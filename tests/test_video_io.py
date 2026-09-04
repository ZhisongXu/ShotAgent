import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from retouch_video import resample_parameter_trajectory
from video_retouch.io import decode_video


class VideoIoTest(unittest.TestCase):
    def test_decode_video_target_fps_skips_frames_and_preserves_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.avi"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (24, 20)
            )
            if not writer.isOpened():
                self.skipTest("MJPG video writer is unavailable")
            for value in range(8):
                writer.write(np.full((20, 24, 3), value * 20, dtype=np.uint8))
            writer.release()

            decoded = decode_video(path, target_fps=2.0)

            self.assertEqual(len(decoded.frames), 2)
            self.assertAlmostEqual(decoded.fps, 2.0)
            self.assertAlmostEqual(len(decoded.frames) / decoded.fps, 1.0)

    def test_resample_parameter_trajectory_to_render_frame_count(self):
        source = np.zeros((3, 12), dtype=np.float64)
        source[:, 0] = [0.0, 1.0, 0.0]

        result = resample_parameter_trajectory(source, 5)

        self.assertEqual(result.shape, (5, 12))
        np.testing.assert_allclose(result[:, 0], [0.0, 0.5, 1.0, 0.5, 0.0])


if __name__ == "__main__":
    unittest.main()
