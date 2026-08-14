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


if __name__ == "__main__":
    unittest.main()
