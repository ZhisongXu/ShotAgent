import unittest

import numpy as np
from PIL import Image

from retouch_agent import RetouchExecutor, RetouchParameters
from video_retouch.backends import AnchorGrade
from video_retouch.critic import ShotCritique
from video_retouch.models import ShotPlan, StoryboardPlan
from video_retouch.pipeline import DynamicGradePipeline


class TwoShotHeroPlanner:
    def plan(self, frames, fps, instruction, anchors_per_shot=1):
        del instruction, anchors_per_shot
        return StoryboardPlan(
            frame_count=len(frames),
            fps=fps,
            shots=(
                ShotPlan(0, 0, 2, (1,), (1, 2)),
                ShotPlan(1, 3, 5, (4,), (4, 5)),
            ),
            planner="fixed-hero-test",
            hero_anchor_frame=1,
            hero_anchor_candidates=(1, 4),
            hero_selection_reason="ranked across shot Anchors",
        )


class ReferenceRecordingEditor:
    name = "reference-aware-editor"

    def __init__(self):
        self.executor = RetouchExecutor()
        self.hero_matches: list[tuple[int, int]] = []

    def grade(self, image, instruction, frame_index, shot_id):
        del instruction
        parameters = RetouchParameters(exposure=0.3 + 0.05 * shot_id)
        return AnchorGrade(
            frame_index=frame_index,
            parameters=parameters,
            preview=self.executor.apply(image, parameters),
            valid=True,
            score=0.8,
            backend=self.name,
            metadata={"shot_id": shot_id, "look_development": True},
        )

    def grade_with_reference(
        self, image, instruction, frame_index, shot_id, hero_reference
    ):
        del instruction
        self.hero_matches.append((frame_index, hero_reference.frame_index))
        parameters = RetouchParameters(
            exposure=hero_reference.grade.parameters.exposure,
            temperature=0.1,
        )
        return AnchorGrade(
            frame_index=frame_index,
            parameters=parameters,
            preview=self.executor.apply(image, parameters),
            valid=True,
            score=0.9,
            backend=self.name,
            metadata={
                "shot_id": shot_id,
                "matched_to_hero_frame": hero_reference.frame_index,
            },
        )


class HeroAwareCritic:
    name = "hero-aware-critic"

    def __init__(self, reject_first_hero=False):
        self.reject_first_hero = reject_first_hero

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
            output_frames,
            frame_parameters,
            frame_uncertainty,
            instruction,
            anchor_grades,
        )
        if len(source_frames) == 1 and self.reject_first_hero and shot.start_frame == 1:
            return ShotCritique(
                0.1,
                False,
                {"hero_quality": 0.1},
                ("poor_hero_reference",),
                None,
            )
        return ShotCritique(0.9, True, {"quality": 0.9}, (), None)


class RejectSecondShotCritic:
    name = "reject-second-shot"

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
            frame_parameters,
            frame_uncertainty,
            instruction,
            anchor_grades,
        )
        if shot.shot_id == 1 and shot.start_frame != shot.end_frame:
            return ShotCritique(
                0.2,
                False,
                {"match": 0.2},
                ("hero_match_failed",),
                None,
            )
        return ShotCritique(0.9, True, {"quality": 0.9}, (), None)


class HeroAnchorPipelineTest(unittest.TestCase):
    @staticmethod
    def _frames():
        return [
            Image.fromarray(
                np.full((18, 24, 3), 60 + index * 12, dtype=np.uint8),
                mode="RGB",
            )
            for index in range(6)
        ]

    def test_other_shot_anchor_is_matched_to_accepted_hero(self):
        editor = ReferenceRecordingEditor()
        pipeline = DynamicGradePipeline(
            shot_planner=TwoShotHeroPlanner(),
            anchor_backend=editor,
            critic=HeroAwareCritic(),
            maximum_attempts=2,
        )

        result = pipeline.run(self._frames(), 3.0, "coherent cinematic ocean look")

        self.assertEqual(result.hero_anchor.frame_index, 1)
        self.assertIn((4, 1), editor.hero_matches)
        self.assertEqual(
            result.shots[1].search_memory["hero_anchor_frame"], 1
        )
        self.assertEqual(
            result.shots[1].search_memory["proposals"]["4"][0]["metadata"][
                "matched_to_hero_frame"
            ],
            1,
        )

    def test_rejected_hero_rolls_back_and_tries_next_ranked_anchor(self):
        pipeline = DynamicGradePipeline(
            shot_planner=TwoShotHeroPlanner(),
            anchor_backend=ReferenceRecordingEditor(),
            critic=HeroAwareCritic(reject_first_hero=True),
            maximum_attempts=1,
            maximum_hero_attempts=2,
        )

        result = pipeline.run(self._frames(), 3.0, "coherent natural look")

        self.assertEqual(result.hero_anchor.frame_index, 4)
        self.assertEqual(len(result.hero_anchor_attempts), 2)
        self.assertFalse(result.hero_anchor_attempts[0]["accepted"])
        self.assertTrue(result.hero_anchor_attempts[1]["selected_global_attempt"])

    def test_no_fully_accepted_hero_pass_rolls_back_entire_video(self):
        pipeline = DynamicGradePipeline(
            shot_planner=TwoShotHeroPlanner(),
            anchor_backend=ReferenceRecordingEditor(),
            critic=RejectSecondShotCritic(),
            maximum_attempts=2,
            maximum_hero_attempts=1,
        )

        result = pipeline.run(self._frames(), 3.0, "one coherent film look")

        self.assertIsNone(result.hero_anchor)
        self.assertTrue(all(shot.rolled_back for shot in result.shots))
        np.testing.assert_allclose(result.frame_parameters, 0.0)
        self.assertTrue(result.hero_anchor_attempts[0]["best_uncommitted_attempt"])


if __name__ == "__main__":
    unittest.main()
