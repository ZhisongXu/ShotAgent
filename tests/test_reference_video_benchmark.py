import unittest

import numpy as np
from PIL import Image

from evaluation.reference_video_benchmark import (
    VideoData,
    global_reinhard,
    metrics,
)


def _video(color: tuple[int, int, int], frames: int = 3) -> VideoData:
    images = tuple(
        Image.new(
            "RGB", (32, 24), tuple(min(255, value + index * 2) for value in color)
        )
        for index in range(frames)
    )
    return VideoData(images, 6.0, __import__("pathlib").Path("synthetic.mp4"))


class ReferenceVideoBenchmarkTests(unittest.TestCase):
    def test_metrics_do_not_include_edit_strength_or_quality_gates(self) -> None:
        target = _video((40, 80, 120))
        edited_frames = tuple(
            Image.fromarray(
                np.clip(
                    np.asarray(frame, dtype=np.int16) + np.array([45, 15, -20]), 0, 255
                ).astype(np.uint8)
            )
            for frame in target.frames
        )
        result = metrics(target, edited_frames)
        self.assertNotIn("edit_magnitude_delta_e00", result)
        self.assertNotIn("edited_pixel_fraction_delta_e00_gt_2", result)
        self.assertNotIn("strong_change_pass", result)
        self.assertNotIn("strong_style_pass", result)

    def test_identity_has_perfect_structure_preservation(self) -> None:
        target = _video((40, 80, 120))
        result = metrics(target, target.frames)
        self.assertAlmostEqual(result["content_structure_correlation"], 1.0, places=8)
        self.assertAlmostEqual(result["edge_ssim"], 1.0, places=8)
        self.assertGreaterEqual(result["temporal_flow_warp_error"], 0.0)
        self.assertAlmostEqual(result["temporal_transform_drift"], 0.0, places=8)

    def test_global_reinhard_preserves_frame_contract(self) -> None:
        target = _video((30, 60, 90))
        reference = _video((180, 120, 40))
        result = global_reinhard(target, reference, 0.8)
        self.assertEqual(len(result), len(target.frames))
        self.assertTrue(all(frame.size == target.frames[0].size for frame in result))


if __name__ == "__main__":
    unittest.main()
