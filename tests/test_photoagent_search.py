import json
import unittest

import numpy as np
from PIL import Image

from retouch_agent import RetouchExecutor, RetouchParameters
from video_retouch.backends import AnchorGrade
from video_retouch.critic import CriticEnsemble, CriticMember, ShotCritique
from video_retouch.models import ShotPlan, StoryboardPlan
from video_retouch.pipeline import DynamicGradePipeline


class OneShotPlanner:
    def plan(self, frames, fps, instruction, anchors_per_shot=1):
        del instruction, anchors_per_shot
        return StoryboardPlan(
            frame_count=len(frames),
            fps=fps,
            shots=(ShotPlan(0, 0, len(frames) - 1, (1,)),),
            planner="fixed",
        )


class TwoAnchorPlanner:
    def plan(self, frames, fps, instruction, anchors_per_shot=2):
        del instruction, anchors_per_shot
        return StoryboardPlan(
            frame_count=len(frames),
            fps=fps,
            shots=(ShotPlan(0, 0, len(frames) - 1, (0, 2), (0, 2)),),
            planner="fixed",
        )


class TwoShotPlanner:
    def plan(self, frames, fps, instruction, anchors_per_shot=1):
        del instruction, anchors_per_shot
        return StoryboardPlan(
            frame_count=len(frames),
            fps=fps,
            shots=(
                ShotPlan(0, 0, 1, (0,), (0,)),
                ShotPlan(1, 2, 3, (3,), (3,)),
            ),
            planner="fixed",
            hero_anchor_frame=0,
            hero_anchor_candidates=(0,),
        )


class ExposureEditor:
    def __init__(self, name, exposure):
        self.name = name
        self.exposure = exposure
        self.executor = RetouchExecutor()

    def grade(self, image, instruction, frame_index, shot_id):
        del instruction
        parameters = RetouchParameters(exposure=self.exposure)
        return AnchorGrade(
            frame_index=frame_index,
            parameters=parameters,
            preview=self.executor.apply(image, parameters),
            valid=True,
            score=0.5,
            backend=self.name,
            metadata={"shot_id": shot_id},
        )


class BatchExposureEditor(ExposureEditor):
    def __init__(self, name, exposure):
        super().__init__(name, exposure)
        self.batch_calls = []
        self.single_calls = []

    def grade(self, image, instruction, frame_index, shot_id):
        self.single_calls.append(frame_index)
        return super().grade(image, instruction, frame_index, shot_id)

    def batch_grade(self, frames, instruction, frame_indices, shot_id):
        self.batch_calls.append(tuple(frame_indices))
        return tuple(
            super(BatchExposureEditor, self).grade(
                frames[frame_index], instruction, frame_index, shot_id
            )
            for frame_index in frame_indices
        )

    def batch_grade_with_reference(
        self, frames, instruction, frame_indices, shot_id, hero_reference
    ):
        del hero_reference
        return self.batch_grade(frames, instruction, frame_indices, shot_id)


class StoryboardBatchExposureEditor(BatchExposureEditor):
    def __init__(self, name, exposure):
        super().__init__(name, exposure)
        self.storyboard_batch_calls = []

    def batch_grade_storyboard(
        self, frames, instruction, frame_indices, frame_to_shot, hero_reference
    ):
        del hero_reference
        self.storyboard_batch_calls.append((tuple(frame_indices), dict(frame_to_shot)))
        return tuple(
            super(BatchExposureEditor, self).grade(
                frames[frame_index],
                instruction,
                frame_index,
                frame_to_shot[frame_index],
            )
            for frame_index in frame_indices
        )


class ExposurePreferenceCritic:
    name = "exposure-preference"

    def evaluate(
        self,
        source_frames,
        output_frames,
        frame_parameters,
        frame_uncertainty,
        shot,
        instruction,
        anchor_grades,
    ):
        del (
            source_frames,
            output_frames,
            frame_uncertainty,
            shot,
            instruction,
            anchor_grades,
        )
        score = float(np.mean(frame_parameters[:, 0]))
        return ShotCritique(score, True, {"exposure": score}, (), None)


class FixedCritic:
    def __init__(self, name, score, accepted, reason=""):
        self.name = name
        self.score = score
        self.accepted = accepted
        self.reason = reason

    def evaluate(self, *args, **kwargs):
        del args, kwargs
        return ShotCritique(
            self.score,
            self.accepted,
            {"fixed": self.score},
            () if self.accepted else (self.reason,),
            None,
        )


class PhotoAgentSearchTest(unittest.TestCase):
    @staticmethod
    def frames():
        return [
            Image.fromarray(np.full((18, 20, 3), 80, dtype=np.uint8), mode="RGB")
            for _ in range(3)
        ]

    @staticmethod
    def four_frames():
        return [
            Image.fromarray(np.full((18, 20, 3), 80, dtype=np.uint8), mode="RGB")
            for _ in range(4)
        ]

    def test_mcts_explores_multiple_editors_and_keeps_best_branch(self):
        pipeline = DynamicGradePipeline(
            shot_planner=OneShotPlanner(),
            anchor_backends=(
                ExposureEditor("conservative-editor", 0.2),
                ExposureEditor("bright-editor", 0.8),
            ),
            critic=ExposurePreferenceCritic(),
            maximum_attempts=2,
            mcts_seed=3,
        )

        result = pipeline.run(self.frames(), 3.0, "make it bright")
        shot = result.shots[0]

        self.assertTrue(shot.accepted)
        self.assertEqual(len(shot.attempts), 2)
        self.assertAlmostEqual(shot.parameter_keyframes[1][0], 0.8, places=5)
        self.assertEqual(shot.search_memory["algorithm"], "photoagent-uct-mcts")
        self.assertEqual(shot.search_memory["evaluated_trajectories"], 2)
        self.assertEqual(
            set(shot.search_memory["editor_agents"]),
            {"conservative-editor", "bright-editor"},
        )

    def test_mcts_batches_anchor_proposals_when_backend_supports_it(self):
        editor = BatchExposureEditor("batch-editor", 0.25)
        pipeline = DynamicGradePipeline(
            shot_planner=TwoAnchorPlanner(),
            anchor_backends=(editor,),
            critic=ExposurePreferenceCritic(),
            anchors_per_shot=2,
            maximum_anchors_per_shot=2,
            maximum_attempts=1,
            mcts_seed=3,
        )

        result = pipeline.run(self.frames(), 3.0, "make it warm")

        self.assertTrue(result.shots[0].accepted)
        self.assertEqual(editor.batch_calls, [(2,)])
        self.assertEqual(editor.single_calls, [0])

    def test_pipeline_batches_storyboard_anchor_proposals_across_shots(self):
        editor = StoryboardBatchExposureEditor("storyboard-batch-editor", 0.25)
        pipeline = DynamicGradePipeline(
            shot_planner=TwoShotPlanner(),
            anchor_backends=(editor,),
            critic=ExposurePreferenceCritic(),
            anchors_per_shot=1,
            maximum_anchors_per_shot=1,
            maximum_attempts=1,
            maximum_hero_attempts=1,
            mcts_seed=3,
        )

        result = pipeline.run(self.four_frames(), 2.0, "make it warm")

        self.assertTrue(all(shot.accepted for shot in result.shots))
        self.assertEqual(editor.storyboard_batch_calls, [((3,), {0: 0, 3: 1})])
        self.assertEqual(editor.batch_calls, [])
        json.dumps(result.to_dict())

    def test_critic_ensemble_safety_veto_forces_rejection(self):
        ensemble = CriticEnsemble(
            (
                CriticMember(FixedCritic("aesthetic", 0.95, True), weight=0.8),
                CriticMember(
                    FixedCritic("safety", 0.4, False, "clipping"),
                    weight=0.2,
                    veto=True,
                ),
            ),
            acceptance_score=0.5,
        )
        frames = self.frames()
        shot = ShotPlan(0, 0, 2, (1,))
        grade = ExposureEditor("editor", 0.2).grade(frames[1], "", 1, 0)

        result = ensemble.evaluate(
            frames,
            frames,
            np.zeros((3, 12)),
            np.zeros((3, 12)),
            shot,
            "natural",
            (grade,),
        )

        self.assertFalse(result.accepted)
        self.assertIn("safety:clipping", result.reasons)
        self.assertTrue(result.metadata["vetoed"])

    def test_visual_member_can_accept_on_calibrated_score(self):
        ensemble = CriticEnsemble(
            (
                CriticMember(
                    FixedCritic("visual", 0.7, False, "minor mismatch"),
                    accept_on_score=True,
                ),
            ),
            acceptance_score=0.6,
        )
        frames = self.frames()
        shot = ShotPlan(0, 0, 2, (1,))
        grade = ExposureEditor("editor", 0.2).grade(frames[1], "", 1, 0)

        result = ensemble.evaluate(
            frames,
            frames,
            np.zeros((3, 12)),
            np.zeros((3, 12)),
            shot,
            "natural",
            (grade,),
        )

        self.assertTrue(result.accepted)
        member = result.metadata["members"]["visual"]
        self.assertFalse(member["accepted"])
        self.assertTrue(member["effective_accepted"])


if __name__ == "__main__":
    unittest.main()
