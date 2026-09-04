import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np


class _VisionHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        prompt = request["messages"][1]["content"][0]["text"]
        if "<TASK_LONG_VIDEO_OVERVIEW>" in prompt:
            payload = {
                "summary": "one continuous dark shot",
                "recurring_elements": [],
                "visual_regimes": [],
                "continuity_notes": [],
            }
        elif "<TASK_SHOT_WINDOW>" in prompt:
            payload = {
                "boundaries": [],
                "window_description": "one continuous shot",
                "uncertain_ranges": [],
            }
        elif "<TASK_ANCHOR_SELECTION>" in prompt:
            payload = {
                "shots": [
                    {
                        "shot_id": 0,
                        "description": "single dark shot",
                        "anchors": [
                            {
                                "frame": 1,
                                "rank": 1,
                                "confidence": 0.9,
                                "reason": "representative frame",
                            }
                        ],
                    }
                ]
            }
        elif "<TASK_HERO_ANCHOR_SELECTION>" in prompt:
            payload = {
                "ranked_candidates": [
                    {
                        "frame": 1,
                        "shot_id": 0,
                        "confidence": 0.95,
                        "reason": "clean master reference",
                    }
                ]
            }
        elif "<TASK_ANCHOR_GRADE>" in prompt:
            if "Batch Anchor grading request" in prompt:
                payload = {
                    "anchors": [
                        {
                            "frame": 1,
                            "diagnosis": {"issues": ["dark"]},
                            "parameter_updates": {"exposure": 0.4},
                            "stages": [
                                {
                                    "stage": "lighting",
                                    "updates": {"exposure": 0.4},
                                    "reason": "dark frame",
                                }
                            ],
                            "constraints": ["preserve_content"],
                            "confidence": 0.9,
                        }
                    ]
                }
            if 'stage "lighting"' in prompt:
                updates = {"exposure": 0.4}
            else:
                updates = {"temperature": 0.1}
            if "Batch Anchor grading request" not in prompt:
                payload = {
                    "diagnosis": {"issues": ["dark"]},
                    "parameter_updates": updates,
                    "constraints": ["preserve_content"],
                    "confidence": 0.9,
                }
        elif "<TASK_ANCHOR_MATCH>" in prompt:
            payload = {
                "diagnosis": {"match_gaps": ["dark"]},
                "parameter_updates": {"exposure": 0.3},
                "constraints": ["preserve_content"],
                "semantic_correspondences": [{"hero": "tone", "target": "tone"}],
                "protected_regions": [],
                "mkl_decision": "reject",
                "mkl_weight": 0.0,
                "confidence": 0.9,
            }
        elif "<TASK_CRITIQUE>" in prompt:
            payload = {
                "accept": True,
                "score": 0.9,
                "instruction_score": 0.9,
                "content_score": 0.9,
                "consistency_score": 0.9,
                "reasons": [],
            }
        else:
            self.send_error(400)
            return
        body = json.dumps(
            {"choices": [{"message": {"content": json.dumps(payload)}}]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        del format, args


class VideoCliEndToEndTest(unittest.TestCase):
    def test_video_and_text_produce_mcts_grade_graph(self):
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

                endpoint = f"http://127.0.0.1:{server.server_port}/v1"
                config = {
                    "storyboard": {
                        "provider": "openai_compatible",
                        "base_url": endpoint,
                        "model": "perceiver",
                        "api_key_env": "TEST_VISION_KEY",
                    },
                    "editors": [
                        {
                            "name": "lighting-editor",
                            "type": "vision_model",
                            "provider": "openai_compatible",
                            "base_url": endpoint,
                            "model": "lighting",
                            "api_key_env": "TEST_VISION_KEY",
                            "stages": ["lighting"],
                            "candidate_count": 1,
                        },
                        {
                            "name": "color-editor",
                            "type": "vision_model",
                            "provider": "openai_compatible",
                            "base_url": endpoint,
                            "model": "color",
                            "api_key_env": "TEST_VISION_KEY",
                            "stages": ["white_balance_and_color"],
                            "candidate_count": 1,
                        },
                    ],
                    "evaluators": [
                        {
                            "name": "temporal-safety",
                            "type": "metrics",
                            "weight": 0.5,
                            "veto": True,
                        },
                        {
                            "name": "visual-critic",
                            "type": "vision_model",
                            "provider": "openai_compatible",
                            "base_url": endpoint,
                            "model": "critic",
                            "api_key_env": "TEST_VISION_KEY",
                            "weight": 0.5,
                        },
                    ],
                    "acceptance_score": 0.5,
                    "search": {"maximum_evaluations": 2, "seed": 2},
                }
                config_path = root / "agents.json"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                output = root / "grade.json"
                rollouts = root / "rollouts.jsonl"
                video_outputs = root / "videos"
                environment = dict(os.environ)
                environment["TEST_VISION_KEY"] = "test-only"

                completed = subprocess.run(
                    [
                        sys.executable,
                        "retouch_video.py",
                        "--input",
                        str(video),
                        "--instruction",
                        "make it brighter while preserving content",
                        "--agent-config",
                        str(config_path),
                        "--output",
                        str(output),
                        "--trajectory-output",
                        str(rollouts),
                        "--video-output-dir",
                        str(video_outputs),
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                payload = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(payload["orchestrator"], "photoagent-uct-mcts")
                self.assertEqual(len(payload["agent_runtime"]["editors"]), 2)
                self.assertTrue(payload["shots"][0]["accepted"])
                self.assertGreater(
                    max(
                        abs(value)
                        for value in payload["shots"][0][
                            "parameter_keyframes"
                        ]["1"]
                    ),
                    0.0,
                )
                self.assertTrue(rollouts.read_text(encoding="utf-8").strip())
                self.assertTrue((video_outputs / "input.source.mp4").is_file())
                self.assertTrue((video_outputs / "input.graded.mp4").is_file())
                self.assertEqual(
                    payload["video_artifacts"]["result"],
                    str(video_outputs / "input.graded.mp4"),
                )
                self.assertIn(str(output), completed.stdout)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
