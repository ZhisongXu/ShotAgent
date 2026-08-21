import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from video_retouch.operations import (
    OperationExecutor,
    canonicalize_operation_parameters,
)
from video_retouch.unified_backend import EditOperation


def operation(operation_type, parameters, frame_range=(0, 9)):
    return EditOperation(
        operation_id=f"test-{operation_type}",
        operation_type=operation_type,
        shot_id=0,
        frame_range=frame_range,
        keyframe=0,
        parameters=parameters,
    )


class OperationExecutorTest(unittest.TestCase):
    def test_tone_curve_is_validated_and_applied(self):
        executor = OperationExecutor()
        parameters = executor.canonicalize(
            "tone_curve",
            {
                "channel": "rgb",
                "points": [[0, 0], [0.5, 0.7], [1, 1]],
                "strength": 1,
            },
        )
        source = Image.fromarray(np.full((4, 5, 3), 128, dtype=np.uint8), "RGB")

        output = executor.apply(
            source, [operation("tone_curve", parameters)], frame_index=3
        )

        self.assertGreater(float(np.asarray(output).mean()), 128.0)
        with self.assertRaisesRegex(ValueError, "monotonic"):
            canonicalize_operation_parameters(
                "tone_curve",
                {
                    "points": [[0, 0], [0.5, 0.8], [1, 0.7]],
                },
            )

    def test_hsl_grade_selectively_reduces_red_saturation(self):
        executor = OperationExecutor()
        parameters = executor.canonicalize(
            "hsl_grade",
            {
                "hue_center": 0,
                "hue_width": 60,
                "saturation": -0.75,
                "lightness": 0,
                "hue_shift": 0,
                "strength": 1,
            },
        )
        source = Image.fromarray(
            np.asarray([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8), "RGB"
        )

        output = np.asarray(
            executor.apply(source, [operation("hsl_grade", parameters)], frame_index=0)
        )

        red_spread = int(output[0, 0].max()) - int(output[0, 0].min())
        green_spread = int(output[0, 1].max()) - int(output[0, 1].min())
        self.assertLess(red_spread, green_spread)
        self.assertEqual(output[0, 1].tolist(), [0, 255, 0])

    def test_operation_scope_is_respected(self):
        executor = OperationExecutor()
        parameters = executor.canonicalize(
            "tone_curve",
            {"points": [[0, 0], [0.5, 0.8], [1, 1]]},
        )
        source = Image.fromarray(np.full((2, 2, 3), 128, dtype=np.uint8), "RGB")

        output = executor.apply(
            source,
            [operation("tone_curve", parameters, frame_range=(5, 8))],
            frame_index=2,
        )

        np.testing.assert_array_equal(output, source)

    def test_registered_cube_lut_uses_trilinear_interpolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invert.cube"
            rows = []
            for blue in (0.0, 1.0):
                for green in (0.0, 1.0):
                    for red in (0.0, 1.0):
                        rows.append(f"{1-red} {1-green} {1-blue}")
            path.write_text(
                "\n".join(["LUT_3D_SIZE 2", *rows]) + "\n",
                encoding="utf-8",
            )
            executor = OperationExecutor({"invert": path})
            parameters = executor.canonicalize(
                "lut", {"lut_id": "invert", "strength": 1}
            )
            source = Image.fromarray(
                np.asarray([[[64, 128, 192]]], dtype=np.uint8), "RGB"
            )

            output = np.asarray(
                executor.apply(source, [operation("lut", parameters)], frame_index=0)
            )

            np.testing.assert_allclose(output[0, 0], [191, 127, 63], atol=1)


if __name__ == "__main__":
    unittest.main()
