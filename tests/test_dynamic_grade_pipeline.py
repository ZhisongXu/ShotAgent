import unittest

import numpy as np
from PIL import Image

from retouch_agent import RetouchExecutor, RetouchParameters
from video_retouch.backends import (
    AnchorGrade,
    HeroAnchorReference,
    ParameterEstimator,
    VLAnchorBackend,
)
from video_retouch.critic import ShotCritique
from video_retouch.models import ShotPlan, StoryboardPlan
from video_retouch.pipeline import DynamicGradePipeline


class FixedShotPlanner:
    def plan(self, frames, fps, instruction, anchors_per_shot=1):
        return StoryboardPlan(
            frame_count=len(frames),
            fps=fps,
            shots=(
                ShotPlan(
                    shot_id=0,
                    start_frame=0,
                    end_frame=len(frames) - 1,
                    anchor_frames=(1,),
                ),
            ),
            planner="fixed-test",
        )


class FixedAnchorBackend:
    name = "fixed-anchor"

    def __init__(self):
        self.executor = RetouchExecutor()

    def grade(self, image, instruction, frame_index, shot_id):
        parameters = RetouchParameters(exposure=0.4 + 0.05 * frame_index)
        return AnchorGrade(
            frame_index=frame_index,
            parameters=parameters,
            preview=self.executor.apply(image, parameters),
            valid=True,
            score=1.0,
            backend=self.name,
        )


class AcceptingCritic:
    name = "accepting-test-critic"

    def evaluate(self, *args, **kwargs):
        return ShotCritique(
            score=0.9,
            accepted=True,
            metrics={"test": 1.0},
            reasons=(),
            recommended_anchor=None,
        )


class ReanchorCritic:
    name = "reanchor-test-critic"

    def __init__(self):
        self.calls = 0

    def evaluate(self, source_frames, *args, **kwargs):
        if len(source_frames) == 1:
            return ShotCritique(
                score=0.9,
                accepted=True,
                metrics={"hero": 1.0},
                reasons=(),
                recommended_anchor=None,
            )
        self.calls += 1
        if self.calls == 1:
            return ShotCritique(
                score=0.2,
                accepted=False,
                metrics={"temporal": 1.0},
                reasons=("temporal_edit_residual",),
                recommended_anchor=3,
            )
        return ShotCritique(
            score=0.8,
            accepted=True,
            metrics={"temporal": 0.0},
            reasons=(),
            recommended_anchor=None,
        )


class RejectingCritic:
    name = "rejecting-test-critic"

    def evaluate(self, *args, **kwargs):
        return ShotCritique(
            score=0.0,
            accepted=False,
            metrics={"unsafe": 1.0},
            reasons=("content_fidelity",),
            recommended_anchor=None,
        )


class FakeAnchorVisionClient:
    model_id = "test/anchor-vision-model"

    def __init__(self):
        self.calls = 0
        self.prompts = []
        self.image_counts = []

    def generate_json(self, labeled_images, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        self.image_counts.append(len(labeled_images))
        updates = (
            {"exposure": 0.6}
            if self.calls == 1
            else {"temperature": 0.15} if self.calls == 2 else {"contrast": 0.1}
        )
        return {
            "diagnosis": {"stage_call": self.calls},
            "parameter_updates": updates,
            "constraints": ["preserve_content"],
            "semantic_correspondences": [
                {"hero": "neutral subject", "target": "neutral subject"}
            ],
            "protected_regions": ["skin", "practical lights"],
            "mkl_decision": "attenuate",
            "mkl_weight": 0.4,
            "confidence": 0.8,
        }


class DynamicGradePipelineTest(unittest.TestCase):
    @staticmethod
    def _frames() -> list[Image.Image]:
        return [
            Image.fromarray(
                np.full((20, 24, 3), 70 + 5 * i, dtype=np.uint8), mode="RGB"
            )
            for i in range(5)
        ]

    def test_outputs_grade_parameters_instead_of_video(self) -> None:
        pipeline = DynamicGradePipeline(
            shot_planner=FixedShotPlanner(),
            anchor_backend=FixedAnchorBackend(),
            critic=AcceptingCritic(),
        )

        result = pipeline.run(self._frames(), fps=5.0, instruction="brighter")

        self.assertEqual(result.frame_parameters.shape, (5, 12))
        self.assertTrue(result.shots[0].accepted)
        self.assertFalse(result.shots[0].rolled_back)
        self.assertGreater(float(result.frame_parameters[:, 0].mean()), 0.0)
        self.assertEqual(result.to_dict()["schema_version"], "dynamic-grade-graph/v1")

    def test_critic_can_insert_anchor_and_retry(self) -> None:
        critic = ReanchorCritic()
        pipeline = DynamicGradePipeline(
            shot_planner=FixedShotPlanner(),
            anchor_backend=FixedAnchorBackend(),
            critic=critic,
            maximum_anchors_per_shot=2,
            maximum_attempts=2,
        )

        result = pipeline.run(self._frames(), fps=5.0, instruction="brighter")

        shot = result.shots[0]
        self.assertTrue(shot.accepted)
        self.assertEqual(len(shot.attempts), 2)
        self.assertEqual(shot.attempts[1].anchor_frames, (3,))
        self.assertEqual(set(shot.parameter_keyframes), {3})
        self.assertEqual(
            shot.search_memory["rounds"][0]["anchor_action"], "replace"
        )

    def test_rejected_shot_rolls_parameter_graph_back_to_identity(self) -> None:
        pipeline = DynamicGradePipeline(
            shot_planner=FixedShotPlanner(),
            anchor_backend=FixedAnchorBackend(),
            critic=RejectingCritic(),
            maximum_attempts=1,
        )

        result = pipeline.run(self._frames(), fps=5.0, instruction="unsafe edit")

        shot = result.shots[0]
        self.assertTrue(shot.rolled_back)
        self.assertEqual(shot.rollback_reason, "content_fidelity")
        np.testing.assert_allclose(result.frame_parameters, 0.0)
        self.assertEqual(shot.parameter_keyframes, {})

    def test_image_only_backend_result_is_recovered_as_parameters(self) -> None:
        array = np.linspace(20, 180, 20 * 24 * 3, dtype=np.uint8).reshape(20, 24, 3)
        source = Image.fromarray(array, mode="RGB")
        target = RetouchExecutor().apply(
            source, RetouchParameters(exposure=0.5, temperature=0.15)
        )

        parameters, error = ParameterEstimator(iterations=80, max_side=64).fit(
            source, target
        )

        self.assertLess(error, 0.01)
        self.assertAlmostEqual(parameters.exposure, 0.5, delta=0.15)

    def test_dedicated_anchor_model_performs_staged_grading(self) -> None:
        client = FakeAnchorVisionClient()
        backend = VLAnchorBackend(client, candidate_count=1, seed=2)
        source = Image.fromarray(np.full((24, 28, 3), 55, dtype=np.uint8), mode="RGB")

        grade = backend.grade(source, "brighter warm cinematic", 3, 0)

        self.assertEqual(client.calls, 3)
        self.assertEqual(grade.backend, "vl-anchor-agent")
        self.assertTrue(grade.valid)
        self.assertGreater(grade.parameters.exposure, 0.0)
        self.assertGreater(grade.parameters.temperature, 0.0)
        self.assertEqual(
            grade.metadata["plan"]["diagnosis"]["model_id"],
            "test/anchor-vision-model",
        )

    def test_vl_hero_keeps_model_grade_for_generic_look_instruction(self) -> None:
        client = FakeAnchorVisionClient()
        backend = VLAnchorBackend(
            client,
            stages=("lighting",),
            candidate_count=1,
            seed=2,
        )
        source = Image.fromarray(
            np.full((24, 28, 3), 128, dtype=np.uint8), mode="RGB"
        )

        grade = backend.grade(source, "coherent natural look", 3, 0)

        self.assertTrue(grade.valid)
        self.assertGreater(grade.parameters.exposure, 0.0)
        self.assertFalse(grade.metadata["rolled_back"])
        self.assertEqual(
            grade.metadata["plan"]["diagnosis"]["target_source"],
            "model_developed_preview",
        )

    def test_vl_anchor_matching_receives_original_and_graded_hero_pair(self) -> None:
        client = FakeAnchorVisionClient()
        backend = VLAnchorBackend(
            client,
            stages=("white_balance_and_color",),
            candidate_count=1,
            seed=2,
        )
        hero_source = Image.fromarray(
            np.full((24, 28, 3), 80, dtype=np.uint8), mode="RGB"
        )
        hero_parameters = RetouchParameters(temperature=0.2, contrast=0.1)
        hero_grade = AnchorGrade(
            frame_index=1,
            parameters=hero_parameters,
            preview=RetouchExecutor().apply(hero_source, hero_parameters),
            valid=True,
            score=0.9,
            backend="hero-editor",
        )
        reference = HeroAnchorReference(1, 0, hero_source, hero_grade)
        target = Image.fromarray(
            np.full((24, 28, 3), 120, dtype=np.uint8), mode="RGB"
        )

        grade = backend.grade_with_reference(
            target, "match the cinematic look", 7, 2, reference
        )

        self.assertIn("<TASK_ANCHOR_MATCH>", client.prompts[0])
        self.assertEqual(client.image_counts[0], 5)
        self.assertEqual(grade.metadata["matched_to_hero_frame"], 1)
        self.assertEqual(
            grade.metadata["mkl_prior"]["semantic_decision"]["decision"],
            "attenuate",
        )
        self.assertEqual(
            grade.metadata["mkl_prior"]["semantic_decision"]["protected_regions"],
            ["skin", "practical lights"],
        )


if __name__ == "__main__":
    unittest.main()
