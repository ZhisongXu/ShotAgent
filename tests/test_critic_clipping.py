import unittest

import numpy as np
from PIL import Image

from retouch_agent import RetouchParameters
from video_retouch.backends import AnchorGrade
from video_retouch.critic import PhotoAgentStyleCritic
from video_retouch.models import ShotPlan


class PhotoAgentStyleCriticClippingTest(unittest.TestCase):
    @staticmethod
    def _evaluate(source: Image.Image, output: Image.Image):
        shot = ShotPlan(
            shot_id=0,
            start_frame=0,
            end_frame=0,
            anchor_frames=(0,),
        )
        grade = AnchorGrade(
            frame_index=0,
            parameters=RetouchParameters(),
            preview=output,
            valid=True,
            score=1.0,
            backend="test",
        )
        critic = PhotoAgentStyleCritic(
            use_vl_review=False,
            maximum_fidelity_l1=1.0,
            maximum_clipping=0.01,
            maximum_anchor_error=1.0,
        )
        return critic.evaluate(
            [source],
            [output],
            np.zeros((1, 12), dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            shot,
            "test",
            [grade],
        )

    def test_preserves_preexisting_clipping_without_rejecting(self) -> None:
        pixels = np.zeros((16, 16, 3), dtype=np.uint8)
        pixels[:, 8:] = 255
        source = Image.fromarray(pixels, mode="RGB")

        critique = self._evaluate(source, source.copy())

        self.assertNotIn("highlight_or_shadow_clipping", critique.reasons)
        self.assertEqual(critique.metrics["source_clipping_fraction"], 1.0)
        self.assertEqual(critique.metrics["added_clipping_fraction"], 0.0)

    def test_rejects_clipping_introduced_by_the_grade(self) -> None:
        source = Image.new("RGB", (16, 16), (128, 128, 128))
        output = Image.new("RGB", (16, 16), (255, 255, 255))

        critique = self._evaluate(source, output)

        self.assertIn("highlight_or_shadow_clipping", critique.reasons)
        self.assertEqual(critique.metrics["source_clipping_fraction"], 0.0)
        self.assertEqual(critique.metrics["added_clipping_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
