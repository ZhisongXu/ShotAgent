import json
import tempfile
import unittest
from pathlib import Path

from video_retouch.agent_config import load_multi_agent_runtime
from video_retouch.backends import NativeRetouchBackend, VLAnchorBackend
from video_retouch.clients import OpenAIResponsesVisionClient
from video_retouch.critic import CriticEnsemble


class MultiAgentConfigTest(unittest.TestCase):
    def test_builds_independent_editor_and_evaluator_pools(self) -> None:
        payload = {
            "storyboard": {
                "provider": "openai_compatible",
                "base_url": "https://storyboard.example/v1",
                "model": "scene-model-a",
                "api_key_env": "STORYBOARD_TEST_KEY",
            },
            "editors": [
                {
                    "name": "grade-editor",
                    "type": "vision_model",
                    "provider": "openai_compatible",
                    "base_url": "https://anchor.example/v1",
                    "model": "grade-model-b",
                    "api_key_env": "ANCHOR_TEST_KEY",
                },
                {"name": "native-editor", "type": "native"},
            ],
            "evaluators": [
                {
                    "name": "temporal-safety",
                    "type": "metrics",
                    "weight": 0.6,
                    "veto": True,
                },
                {
                    "name": "visual-critic",
                    "type": "vision_model",
                    "provider": "openai_compatible",
                    "base_url": "https://critic.example/v1",
                    "model": "critic-model-c",
                    "api_key_env": "CRITIC_TEST_KEY",
                    "weight": 0.4,
                },
            ],
            "search": {"maximum_evaluations": 17},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agents.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            runtime = load_multi_agent_runtime(path)

        self.assertEqual(runtime.storyboard_client.model_id, "scene-model-a")
        self.assertEqual(len(runtime.anchor_backends), 2)
        self.assertIsInstance(runtime.anchor_backends[0], VLAnchorBackend)
        self.assertIsInstance(runtime.anchor_backends[1], NativeRetouchBackend)
        self.assertEqual(runtime.anchor_backends[0].client.model_id, "grade-model-b")
        self.assertIsInstance(runtime.critic, CriticEnsemble)
        self.assertEqual(runtime.critic_client.model_id, "critic-model-c")
        self.assertEqual(runtime.search.maximum_evaluations, 17)
        self.assertNotIn("api_key_env", json.dumps(runtime.manifest))

    def test_builds_gpt56_responses_client_and_long_video_policy(self) -> None:
        payload = {
            "storyboard": {
                "provider": "openai_responses",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-5.6-sol",
                "api_key_env": "OPENAI_API_KEY",
                "reasoning_effort": "high",
                "long_video": {
                    "window_seconds": 30.0,
                    "overlap_seconds": 5.0,
                    "max_window_images": 32,
                },
            },
            "editors": [{"name": "native-editor", "type": "native"}],
            "evaluators": [{"name": "safety", "type": "metrics"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agents.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            runtime = load_multi_agent_runtime(path)

        self.assertIsInstance(
            runtime.storyboard_client, OpenAIResponsesVisionClient
        )
        self.assertEqual(runtime.storyboard_client.model_id, "gpt-5.6-sol")
        self.assertEqual(runtime.storyboard_settings.window_seconds, 30.0)
        self.assertEqual(runtime.storyboard_settings.max_window_images, 32)
        self.assertEqual(
            runtime.manifest["storyboard"]["planner"],
            "hierarchical-vision-storyboard/v2",
        )


if __name__ == "__main__":
    unittest.main()
