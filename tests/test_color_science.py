import unittest

import cv2
import numpy as np
from PIL import Image

from retouch_agent import RetouchExecutor, RetouchParameters
from video_retouch.color_science import (
    LinearMongeKantorovichMatcher,
    SourceGuidedTonalStabilizer,
    spatiotemporal_palette_features,
)


def _lab_statistics(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)
    lab[:, 0] /= 100.0
    lab[:, 1:] /= 128.0
    return np.mean(lab, axis=0), np.cov(lab, rowvar=False)


class ColorScienceTest(unittest.TestCase):
    def test_mkl_moves_first_and_second_order_color_statistics(self) -> None:
        y, x = np.mgrid[0:36, 0:48]
        source_array = np.stack(
            [
                35 + 3 * x,
                25 + 4 * y,
                45 + 2 * x + y,
            ],
            axis=2,
        ).clip(0, 255).astype(np.uint8)
        reference_array = np.stack(
            [
                65 + 2 * x + y,
                40 + 2 * y,
                25 + 4 * x,
            ],
            axis=2,
        ).clip(0, 255).astype(np.uint8)
        source = Image.fromarray(source_array, mode="RGB")
        reference = Image.fromarray(reference_array, mode="RGB")

        output, diagnostics = LinearMongeKantorovichMatcher(
            strength=1.0,
            analysis_max_side=128,
        ).transfer(source, reference)

        source_mean, source_covariance = _lab_statistics(source)
        output_mean, output_covariance = _lab_statistics(output)
        reference_mean, reference_covariance = _lab_statistics(reference)
        self.assertLess(
            np.linalg.norm(output_mean - reference_mean),
            np.linalg.norm(source_mean - reference_mean),
        )
        self.assertLess(
            np.linalg.norm(output_covariance - reference_covariance),
            np.linalg.norm(source_covariance - reference_covariance),
        )
        self.assertLess(
            diagnostics["covariance_error_after_full_transport"],
            diagnostics["covariance_error_before"],
        )

    def test_palette_trace_uses_one_corresponding_palette_over_time(self) -> None:
        frames = []
        for red_width in (8, 16, 24):
            array = np.zeros((24, 32, 3), dtype=np.uint8)
            array[:, :red_width] = [210, 35, 30]
            array[:, red_width:] = [25, 90, 210]
            frames.append(Image.fromarray(array, mode="RGB"))

        features = spatiotemporal_palette_features(frames, palette_size=2)

        self.assertEqual(features.shape, (3, 8))
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertFalse(np.allclose(features[0], features[-1]))
        np.testing.assert_allclose(np.sum(features[:, :2], axis=1), 1.0)

    def test_source_guided_stabilization_preserves_anchor_and_output_tone(self) -> None:
        values = [80, 120, 78, 122, 80, 121, 79, 120, 80]
        frames = [
            Image.fromarray(np.full((28, 32, 3), value, dtype=np.uint8), mode="RGB")
            for value in values
        ]
        trajectory = np.zeros((len(frames), 12), dtype=np.float64)
        anchor_indices = np.asarray([4], dtype=np.int64)
        anchor_values = np.zeros((1, 12), dtype=np.float64)

        stabilized, diagnostics = SourceGuidedTonalStabilizer(
            prior_weight=0.5,
            velocity_weight=0.2,
            curvature_weight=0.2,
        ).stabilize(frames, trajectory, anchor_indices, anchor_values)

        np.testing.assert_allclose(stabilized[4], anchor_values[0])
        self.assertTrue(diagnostics["anchor_constraints_preserved"])
        source_luma = np.asarray(values, dtype=np.float64)
        self.assertLess(np.corrcoef(source_luma, stabilized[:, 0])[0, 1], -0.5)
        executor = RetouchExecutor()
        output_luma = []
        for frame, parameters in zip(frames, stabilized):
            rendered = executor.apply(frame, RetouchParameters.from_vector(parameters))
            output_luma.append(np.asarray(rendered, dtype=np.float64).mean())
        self.assertLess(np.std(output_luma), np.std(source_luma))


if __name__ == "__main__":
    unittest.main()
