import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from video_retouch.critic import CriticEnsemble, VisionReviewCritic
from video_retouch.render import render_grade_frames
from video_retouch.unified_backend import (
    VideoEditRequest,
    build_unified_backend,
)


class ScriptedUnifiedClient:
    model_id = "one-shared-model"

    def generate_json(self, labeled_images, prompt):
        if "<TASK_LONG_VIDEO_OVERVIEW>" in prompt:
            return {
                "summary": "one continuous shot",
                "recurring_elements": [],
                "visual_regimes": [],
                "continuity_notes": [],
            }
        if "<TASK_SHOT_WINDOW>" in prompt:
            return {
                "boundaries": [],
                "window_description": "one continuous shot",
                "uncertain_ranges": [],
            }
        if "<TASK_BOUNDARY_VERIFY>" in prompt:
            return {"decisions": []}
        if "<TASK_ANCHOR_SELECTION>" in prompt:
            by_shot = {}
            for label, _ in labeled_images:
                shot = int(re.search(r"shot_id=(\d+)", label).group(1))
                frame = int(re.search(r"frame_id=(\d+)", label).group(1))
                by_shot.setdefault(shot, frame)
            return {
                "shots": [
                    {
                        "shot_id": shot,
                        "description": "test shot",
                        "anchors": [
                            {
                                "frame": frame,
                                "rank": 1,
                                "confidence": 0.9,
                                "reason": "representative",
                            }
                        ],
                    }
                    for shot, frame in sorted(by_shot.items())
                ]
            }
        if "<TASK_HERO_ANCHOR_SELECTION>" in prompt:
            label = labeled_images[0][0]
            return {
                "ranked_candidates": [
                    {
                        "frame": int(re.search(r"frame_id=(\d+)", label).group(1)),
                        "shot_id": int(re.search(r"shot_id=(\d+)", label).group(1)),
                        "confidence": 0.95,
                        "reason": "master reference",
                    }
                ]
            }
        if "<TASK_ANCHOR_GRADE>" in prompt or "<TASK_ANCHOR_MATCH>" in prompt:
            return {
                "diagnosis": {"issues": ["slightly dark"]},
                "parameter_updates": {"exposure": 0.2},
                "constraints": ["preserve content"],
                "confidence": 0.9,
                "mkl_decision": "reject",
                "mkl_weight": 0.0,
            }
        if "<TASK_CRITIQUE>" in prompt:
            return {
                "accept": True,
                "score": 0.95,
                "instruction_score": 0.95,
                "content_score": 0.95,
                "consistency_score": 0.95,
                "hero_match_score": 0.95,
                "recommended_anchor": None,
                "reasons": [],
            }
        if "<TASK_OPERATION_PLAN>" in prompt:
            return {
                "operations": [
                    {
                        "type": "tone_curve",
                        "parameters": {
                            "channel": "rgb",
                            "points": [[0, 0], [0.5, 0.56], [1, 1]],
                            "strength": 0.5,
                        },
                        "confidence": 0.85,
                        "reason": "gentle midtone lift",
                    }
                ],
                "diagnosis": ["global grade can use a small curve refinement"],
            }
        if "<TASK_OPERATION_REVIEW>" in prompt:
            return {
                "accept": True,
                "score": 0.9,
                "reasons": [],
                "instruction_score": 0.9,
                "preservation_score": 0.95,
            }
        raise AssertionError(f"Unexpected task: {prompt[:120]}")


class RetryOperationClient(ScriptedUnifiedClient):
    def __init__(self):
        self.operation_calls = 0

    def generate_json(self, labeled_images, prompt):
        if "<TASK_OPERATION_PLAN>" in prompt:
            self.operation_calls += 1
            if self.operation_calls == 1:
                return {
                    "operations": [
                        {
                            "type": "tone_curve",
                            "parameters": {"points": [[0, 0], [0.5, 0.8], [1, 0.7]]},
                        }
                    ]
                }
        return super().generate_json(labeled_images, prompt)


def unified_config():
    return {
        "backend": {
            "type": "unified_vl_video",
            "provider": "openai_compatible",
            "base_url": "https://unused.example/v1",
            "model": "one-shared-model",
            "api_key_env": "UNUSED_TEST_KEY",
            "operations": {"global_grade": True},
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
            "search": {"maximum_evaluations": 2, "seed": 3},
        }
    }


class UnifiedBackendTest(unittest.TestCase):
    def test_one_client_and_one_editor_own_every_semantic_role(self):
        client = ScriptedUnifiedClient()
        backend = build_unified_backend(unified_config(), client=client)

        self.assertIs(backend.client, client)
        self.assertIs(backend.shot_planner.client, client)
        self.assertIs(backend.editor.client, client)
        self.assertEqual(backend.pipeline.anchor_backends, (backend.editor,))
        self.assertIsInstance(backend.critic, CriticEnsemble)
        visual = next(
            member.critic
            for member in backend.critic.members
            if isinstance(member.critic, VisionReviewCritic)
        )
        self.assertIs(visual.client, client)
        self.assertTrue(backend.runtime_manifest["backend"]["single_client_for_roles"])

    def test_process_emits_versioned_global_operation_graph(self):
        backend = build_unified_backend(
            unified_config(), client=ScriptedUnifiedClient()
        )
        frames = tuple(
            Image.fromarray(np.full((20, 24, 3), value, dtype=np.uint8), "RGB")
            for value in (50, 52, 54, 56)
        )

        result = backend.process(
            VideoEditRequest(
                frames=frames,
                fps=4.0,
                instruction="make the shot slightly brighter",
            )
        )
        payload = result.to_dict()

        self.assertEqual(payload["backend"], "unified-vl-video")
        self.assertEqual(payload["orchestrator"], "unified-anchor-trajectory-search/v1")
        self.assertEqual(
            payload["operation_graph"]["schema_version"],
            "video-edit-operation-graph/v1",
        )
        self.assertTrue(payload["operation_graph"]["operations"])
        operation = payload["operation_graph"]["operations"][0]
        self.assertEqual(operation["type"], "global_grade")
        self.assertGreater(operation["parameters"]["exposure"], 0.0)
        self.assertNotIn("api_key_env", str(payload["backend_runtime"]))

    def test_rejects_advertised_but_unimplemented_operation(self):
        config = unified_config()
        config["backend"]["operations"]["masked_grade"] = True

        with self.assertRaisesRegex(ValueError, "not implemented"):
            build_unified_backend(config, client=ScriptedUnifiedClient())

    def test_plans_reviews_and_renders_tone_curve_operation(self):
        config = unified_config()
        config["backend"]["operations"]["tone_curve"] = True
        backend = build_unified_backend(config, client=ScriptedUnifiedClient())
        frames = tuple(
            Image.fromarray(np.full((20, 24, 3), value, dtype=np.uint8), "RGB")
            for value in (50, 52, 54, 56)
        )

        result = backend.process(
            VideoEditRequest(
                frames=frames,
                fps=4.0,
                instruction="make the shot slightly brighter",
            )
        )
        post_operations = [
            operation
            for operation in result.operations
            if operation.operation_type != "global_grade"
        ]
        base = list(render_grade_frames(frames, result.grade_graph.frame_parameters))
        enhanced = list(
            render_grade_frames(
                frames,
                result.grade_graph.frame_parameters,
                operations=result.operations,
                operation_executor=backend.operation_executor,
            )
        )

        self.assertEqual(len(post_operations), 1)
        self.assertEqual(post_operations[0].operation_type, "tone_curve")
        self.assertTrue(result.operation_audit[0]["accepted"])
        self.assertGreater(
            float(np.asarray(enhanced[0]).mean()),
            float(np.asarray(base[0]).mean()),
        )

    def test_post_operation_stack_rolls_back_on_deterministic_safety_failure(self):
        config = unified_config()
        config["backend"]["operations"]["tone_curve"] = True
        config["backend"]["operation_policy"] = {
            "maximum_additional_fidelity_l1": 0.0,
            "strict": False,
        }
        backend = build_unified_backend(config, client=ScriptedUnifiedClient())
        frames = tuple(
            Image.fromarray(np.full((20, 24, 3), value, dtype=np.uint8), "RGB")
            for value in (50, 52, 54, 56)
        )

        result = backend.process(
            VideoEditRequest(
                frames=frames,
                fps=4.0,
                instruction="make the shot slightly brighter",
            )
        )

        self.assertFalse(
            any(
                operation.operation_type == "tone_curve"
                for operation in result.operations
            )
        )
        self.assertTrue(result.operation_audit[0]["rolled_back"])
        self.assertIn(
            "additional_edit_too_strong",
            result.operation_audit[0]["review"]["reasons"],
        )

    def test_invalid_operation_json_is_retried_with_validation_feedback(self):
        config = unified_config()
        config["backend"]["operations"]["tone_curve"] = True
        client = RetryOperationClient()
        backend = build_unified_backend(config, client=client)
        frames = tuple(
            Image.fromarray(np.full((20, 24, 3), value, dtype=np.uint8), "RGB")
            for value in (50, 52, 54, 56)
        )

        result = backend.process(
            VideoEditRequest(
                frames=frames,
                fps=4.0,
                instruction="make the shot slightly brighter",
            )
        )

        self.assertEqual(client.operation_calls, 2)
        self.assertEqual(result.operation_audit[0]["planning_attempts"], 2)
        self.assertTrue(result.operation_audit[0]["planning_errors"])
        self.assertTrue(result.operation_audit[0]["accepted"])

    def test_lut_catalog_cannot_escape_config_directory(self):
        config = unified_config()
        config["backend"]["operations"]["lut"] = True
        config["backend"]["lut_catalog"] = {"unsafe": "../outside.cube"}
        with tempfile.TemporaryDirectory() as directory:
            config_root = Path(directory) / "config"
            config_root.mkdir()

            with self.assertRaisesRegex(ValueError, "cannot escape"):
                build_unified_backend(
                    config,
                    client=ScriptedUnifiedClient(),
                    config_root=config_root,
                )

    def test_rejects_legacy_editor_pool_in_unified_config(self):
        config = unified_config()
        config["editors"] = [{"type": "native"}]

        with self.assertRaisesRegex(ValueError, "editor or evaluator pools"):
            build_unified_backend(config, client=ScriptedUnifiedClient())


if __name__ == "__main__":
    unittest.main()
