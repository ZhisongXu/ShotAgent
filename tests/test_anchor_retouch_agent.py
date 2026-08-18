import unittest

import numpy as np
from PIL import Image

from retouch_agent.agent import AnchorRetouchAgent
from retouch_agent.evaluator import CandidateEvaluation
from retouch_agent.parameters import RetouchParameters
from retouch_agent.planner import RetouchPlan


class RejectingEvaluator:
    def evaluate(self, source, output, plan) -> CandidateEvaluation:
        return CandidateEvaluation(
            score=-100.0,
            valid=False,
            metrics={"forced_rejection": 1.0},
        )


class FlatEvaluator:
    def evaluate(self, source, output, plan) -> CandidateEvaluation:
        return CandidateEvaluation(score=1.0, valid=True, metrics={"flat": 1.0})


class ClippingPlanner:
    def plan(self, image, instruction, reference=None, has_local_mask=False):
        return RetouchPlan(
            diagnosis={"planner": "test"},
            initial_parameters=RetouchParameters(exposure=3.0),
            targets={
                "luminance": 1.0,
                "contrast": 0.0,
                "saturation": 0.0,
                "warmth": 0.0,
            },
        )


class AnchorRetouchAgentTest(unittest.TestCase):
    def test_dark_image_is_brightened_and_returns_parameter_uncertainty(self) -> None:
        source = np.full((40, 48, 3), 45, dtype=np.uint8)
        image = Image.fromarray(source, mode="RGB")
        agent = AnchorRetouchAgent(candidate_count=12, covariance_top_k=5, seed=9)
        result = agent.run(image, "make it brighter, warm, and cinematic")

        output = np.asarray(result.image)
        self.assertGreater(float(output.mean()), float(source.mean()) + 20.0)
        self.assertGreater(result.parameters.exposure, 0.2)
        self.assertGreater(result.parameters.temperature, 0.0)
        self.assertEqual(result.parameter_covariance.shape, (12, 12))
        self.assertGreater(result.accepted_candidates, 0)
        self.assertIn("source_statistics", result.plan.diagnosis)
        self.assertFalse(result.rolled_back)

    def test_invalid_candidates_roll_back_to_input_checkpoint(self) -> None:
        source = np.full((24, 32, 3), 80, dtype=np.uint8)
        image = Image.fromarray(source, mode="RGB")
        agent = AnchorRetouchAgent(
            evaluator=RejectingEvaluator(), candidate_count=4, seed=3
        )

        result = agent.run(image, "make it dramatically brighter")

        np.testing.assert_array_equal(np.asarray(result.image), source)
        self.assertTrue(result.rolled_back)
        self.assertEqual(result.rollback_reason, "no_valid_candidate")
        self.assertEqual(result.parameters.exposure, 0.0)
        self.assertFalse(result.evaluation.valid)

    def test_candidate_must_improve_direction_to_avoid_rollback(self) -> None:
        source = np.full((16, 16, 3), 96, dtype=np.uint8)
        agent = AnchorRetouchAgent(
            evaluator=FlatEvaluator(), candidate_count=3, seed=4
        )

        result = agent.run(Image.fromarray(source, mode="RGB"), "make it warmer")

        self.assertTrue(result.rolled_back)
        self.assertEqual(
            result.rollback_reason, "no_directional_improvement"
        )
        self.assertEqual(result.decision["directional_candidates"], 0)
        np.testing.assert_array_equal(np.asarray(result.image), source)

    def test_generic_retouch_commits_visible_target_directed_grade(self) -> None:
        ramp = np.linspace(55, 195, 64, dtype=np.uint8)
        source = np.repeat(ramp[None, :, None], 48, axis=0)
        source = np.repeat(source, 3, axis=2)

        result = AnchorRetouchAgent(candidate_count=12, seed=9).run(
            Image.fromarray(source, mode="RGB"), "retouch"
        )

        output = np.asarray(result.image)
        mean_delta = float(np.abs(output.astype(np.float32) - source).mean())
        self.assertFalse(result.rolled_back)
        self.assertGreater(mean_delta, 5.0)
        self.assertGreater(result.parameters.contrast, 0.0)
        self.assertGreater(result.decision["directional_improvement"], 0.0)
        self.assertGreaterEqual(result.decision["directional_candidates"], 1)

    def test_safe_but_invisible_candidates_roll_back(self) -> None:
        source = np.full((24, 32, 3), 96, dtype=np.uint8)
        result = AnchorRetouchAgent(
            candidate_count=4,
            seed=3,
            minimum_perceptual_delta=1.0,
        ).run(Image.fromarray(source, mode="RGB"), "make it warmer")

        self.assertTrue(result.rolled_back)
        self.assertEqual(result.rollback_reason, "no_perceptible_improvement")
        np.testing.assert_array_equal(np.asarray(result.image), source)

    def test_clipping_is_diagnostic_and_does_not_veto_grade(self) -> None:
        source = np.full((24, 32, 3), 128, dtype=np.uint8)
        result = AnchorRetouchAgent(planner=ClippingPlanner(), candidate_count=1).run(
            Image.fromarray(source, mode="RGB"), "deliberately clip highlights"
        )

        self.assertFalse(result.rolled_back)
        self.assertTrue(result.evaluation.valid)
        self.assertGreater(result.evaluation.metrics["highlight_clipping"], 0.18)
        self.assertGreater(
            float(np.abs(np.asarray(result.image).astype(float) - source).mean()),
            1.0,
        )

    def test_negative_perceptual_threshold_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AnchorRetouchAgent(minimum_perceptual_delta=-0.01)


if __name__ == "__main__":
    unittest.main()
