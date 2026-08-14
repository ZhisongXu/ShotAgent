import unittest

import numpy as np
from PIL import Image

from bayesgrade.pipeline import BayesGradeRetouchPipeline
from retouch_agent.agent import AnchorRetouchAgent


class BayesGradeRetouchPipelineTest(unittest.TestCase):
    def test_anchor_agent_parameters_feed_video_field(self) -> None:
        frames = []
        for value in (55, 65, 90, 115, 130, 145):
            array = np.full((24, 28, 3), value, dtype=np.uint8)
            frames.append(Image.fromarray(array, mode="RGB"))

        pipeline = BayesGradeRetouchPipeline(
            anchor_agent=AnchorRetouchAgent(candidate_count=8, seed=4)
        )
        result = pipeline.run(
            frames,
            "make the clip brighter and warm",
            anchor_indices=[1],
        )

        self.assertEqual(result.parameter_mean.shape, (6, 12))
        self.assertEqual(result.parameter_variance.shape, (6, 12))
        self.assertEqual(len(result.rendered_frames), 6)
        self.assertEqual(len(result.anchor_results), 1)
        self.assertIsNotNone(result.next_anchor)
        self.assertNotEqual(result.next_anchor, 1)
        self.assertGreater(result.anchor_results[0].parameters.exposure, 0.0)
        self.assertGreater(result.anchor_results[0].parameters.temperature, 0.0)


if __name__ == "__main__":
    unittest.main()
