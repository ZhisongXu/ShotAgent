import unittest

import numpy as np
from PIL import Image

from evaluation.reference_video_benchmark import (
    VideoData,
    _delta_e_ciede2000,
    _lab_histogram_emd,
    global_reinhard,
    metrics,
)


def _video(color: tuple[int, int, int], frames: int = 3) -> VideoData:
    images = tuple(Image.new("RGB", (32, 24), tuple(min(255, value + index * 2) for value in color)) for index in range(frames))
    return VideoData(images, 6.0, __import__("pathlib").Path("synthetic.mp4"))


class ReferenceVideoBenchmarkTests(unittest.TestCase):
    def test_edit_magnitude_is_descriptive_without_quality_gate(self) -> None:
        target = _video((40, 80, 120))
        edited_frames = tuple(
            Image.fromarray(
                np.clip(np.asarray(frame, dtype=np.int16) + np.array([45, 15, -20]), 0, 255).astype(np.uint8)
            )
            for frame in target.frames
        )
        result = metrics(target, edited_frames)
        self.assertGreater(result["edit_magnitude_delta_e00"], 0.0)
        self.assertNotIn("strong_change_pass", result)
        self.assertNotIn("strong_style_pass", result)

    def test_identity_has_zero_edit_and_perfect_structure_preservation(self) -> None:
        target = _video((40, 80, 120))
        result = metrics(target, target.frames)
        self.assertAlmostEqual(result["edit_magnitude_delta_e00"], 0.0, places=8)
        self.assertAlmostEqual(result["content_structure_correlation"], 1.0, places=8)
        self.assertAlmostEqual(result["edge_ssim"], 1.0, places=8)
        self.assertGreaterEqual(result["temporal_flow_warp_error"], 0.0)
        self.assertAlmostEqual(result["temporal_transform_drift"], 0.0, places=8)

    def test_reference_color_emd_is_zero_for_same_color_distribution(self) -> None:
        target = _video((40, 80, 120))
        self.assertAlmostEqual(
            _lab_histogram_emd(target.frames, target.frames), 0.0, places=8
        )

    def test_reference_metrics_are_reported_without_ground_truth(self) -> None:
        target = _video((40, 80, 120))
        reference = _video((180, 120, 40))
        result = metrics(target, target.frames, reference=reference)
        self.assertIn("lab_histogram_emd", result)
        self.assertGreater(result["lab_histogram_emd"], 0.0)

    def test_ciede2000_matches_published_reference_pair(self) -> None:
        left = np.array([[50.0, 2.6772, -79.7751]])
        right = np.array([[50.0, 0.0, -82.7485]])
        self.assertAlmostEqual(float(_delta_e_ciede2000(left, right)[0]), 2.0425, places=4)

    def test_global_reinhard_preserves_frame_contract(self) -> None:
        target = _video((30, 60, 90))
        reference = _video((180, 120, 40))
        result = global_reinhard(target, reference, 0.8)
        self.assertEqual(len(result), len(target.frames))
        self.assertTrue(all(frame.size == target.frames[0].size for frame in result))


if __name__ == "__main__":
    unittest.main()
