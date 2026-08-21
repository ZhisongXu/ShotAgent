import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from tests.test_video_cli_e2e import _VisionHandler


class UnifiedBackendCliTest(unittest.TestCase):
    def test_cli_accepts_one_backend_config(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _VisionHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                video = root / "input.avi"
                writer = cv2.VideoWriter(
                    str(video), cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (24, 20)
                )
                if not writer.isOpened():
                    self.skipTest("MJPG video writer is unavailable")
                for value in (45, 50, 55, 60):
                    writer.write(np.full((20, 24, 3), value, dtype=np.uint8))
                writer.release()

                config = {
                    "backend": {
                        "type": "unified_vl_video",
                        "provider": "openai_compatible",
                        "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                        "model": "one-shared-model",
                        "api_key_env": "TEST_VISION_KEY",
                        "operations": {
                            "global_grade": True,
                            "tone_curve": True,
                        },
                        "editor": {
                            "stages": ["lighting"],
                            "candidate_count": 1,
                            "use_mkl_prior": False,
                        },
                        "review": {
                            "enabled": True,
                            "metric_weight": 0.5,
                            "visual_weight": 0.5,
                            "acceptance_score": 0.5,
                        },
                        "search": {"maximum_evaluations": 2, "seed": 2},
                    }
                }
                config_path = root / "backend.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                output = root / "grade.json"
                environment = dict(os.environ)
                environment["TEST_VISION_KEY"] = "test-only"

                subprocess.run(
                    [
                        sys.executable,
                        "retouch_video.py",
                        "--input",
                        str(video),
                        "--instruction",
                        "make it brighter while preserving content",
                        "--backend-config",
                        str(config_path),
                        "--output",
                        str(output),
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                payload = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(payload["backend"], "unified-vl-video")
                self.assertEqual(
                    payload["backend_runtime"]["mode"],
                    "unified-single-backend/v1",
                )
                self.assertNotIn("agent_runtime", payload)
                self.assertEqual(
                    payload["operation_graph"]["supported_operations"],
                    ["global_grade", "hsl_grade", "lut", "tone_curve"],
                )
                self.assertTrue(payload["operation_graph"]["operations"])
                self.assertTrue(
                    any(
                        operation["type"] == "tone_curve"
                        for operation in payload["operation_graph"]["operations"]
                    )
                )
                self.assertTrue(payload["operation_graph"]["audit"][0]["accepted"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
