import unittest

import numpy as np
from PIL import Image

from evaluation.reference_video_benchmark import (
    LAB_SLICED_WASSERSTEIN_UPPER_BOUND,
    LAB_WASSERSTEIN_UPPER_BOUND,
    VideoData,
    bounded_lab_similarity,
    global_reinhard,
    lab_sliced_wasserstein_distance,
    lab_wasserstein_distance,
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
        self.assertNotIn("vgg_style_gain", result)
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

    def test_lab_distribution_distances_are_zero_for_identical_videos(self) -> None:
        reference = _video((120, 90, 60))

        self.assertAlmostEqual(
            lab_wasserstein_distance(reference.frames, reference.frames), 0.0
        )
        self.assertAlmostEqual(
            lab_sliced_wasserstein_distance(reference.frames, reference.frames), 0.0
        )

    def test_lab_distribution_distances_detect_a_color_shift(self) -> None:
        output = _video((30, 90, 45))
        reference = _video((160, 110, 70))

        self.assertGreater(lab_wasserstein_distance(output.frames, reference.frames), 0)
        self.assertGreater(
            lab_sliced_wasserstein_distance(output.frames, reference.frames), 0
        )

    def test_bounded_lab_similarity_has_fixed_data_independent_scale(self) -> None:
        self.assertEqual(bounded_lab_similarity(0.0, LAB_WASSERSTEIN_UPPER_BOUND), 1.0)
        self.assertEqual(
            bounded_lab_similarity(
                LAB_WASSERSTEIN_UPPER_BOUND, LAB_WASSERSTEIN_UPPER_BOUND
            ),
            0.0,
        )
        self.assertAlmostEqual(
            bounded_lab_similarity(0.3, LAB_SLICED_WASSERSTEIN_UPPER_BOUND), 0.9
        )

    def test_metrics_report_lab_distances_and_bounded_similarities(self) -> None:
        target = _video((30, 90, 45))
        reference = _video((160, 110, 70))

        result = metrics(target, target.frames, reference=reference)

        self.assertAlmostEqual(
            result["lab_wasserstein_similarity"],
            bounded_lab_similarity(
                result["lab_wasserstein_distance"], LAB_WASSERSTEIN_UPPER_BOUND
            ),
        )
        self.assertAlmostEqual(
            result["lab_sliced_wasserstein_similarity"],
            bounded_lab_similarity(
                result["lab_sliced_wasserstein_distance"],
                LAB_SLICED_WASSERSTEIN_UPPER_BOUND,
            ),
        )


if __name__ == "__main__":
    unittest.main()
