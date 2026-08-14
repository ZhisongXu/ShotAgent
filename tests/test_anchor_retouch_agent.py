import unittest

import numpy as np
from PIL import Image

from retouch_agent.agent import AnchorRetouchAgent
from retouch_agent.evaluator import CandidateEvaluation


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

    def test_candidate_must_beat_checkpoint_to_avoid_rollback(self) -> None:
        source = np.full((16, 16, 3), 96, dtype=np.uint8)
        agent = AnchorRetouchAgent(
            evaluator=FlatEvaluator(), candidate_count=3, seed=4
        )

        result = agent.run(Image.fromarray(source, mode="RGB"), "make it warmer")

        self.assertTrue(result.rolled_back)
        self.assertEqual(
            result.rollback_reason, "insufficient_quality_improvement"
        )
        np.testing.assert_array_equal(np.asarray(result.image), source)


if __name__ == "__main__":
    unittest.main()
