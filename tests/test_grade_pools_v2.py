import unittest
import json
from pathlib import Path

import numpy as np
from PIL import Image

from video_retouch.grade_pools import (
    POOL_OPERATION_TYPES,
    GradePoolExecutor,
    canonicalize_pool_parameters,
)
from video_retouch.pool_pipeline import PoolGradePipeline
from video_retouch.operations import OperationExecutor
from video_retouch.render import render_grade_frames
from video_retouch.shot_planner import HeuristicShotPlanner
from video_retouch.unified_backend import VideoEditRequest, build_unified_backend


class ScriptedPoolClient:
    model_id = "scripted-pool-v2"

    def generate_json(self, labeled_images, prompt):
        if "<TASK_POOL_REVIEW_V2>" in prompt:
            return {
                "accept": True,
                "score": 0.95,
                "instruction_score": 0.95,
                "preservation_score": 0.95,
                "temporal_score": 0.95,
                "reasons": [],
            }
        if "At stage 'technical'" in prompt:
            return {
                "operations": [
                    {
                        "type": "denoise",
                        "parameters": {"luminance": 5, "color": 8},
                    },
                    {
                        "type": "white_balance",
                        "parameters": {"temperature": 6200, "tint": 2},
                    },
                    {
                        "type": "primary",
                        "parameters": {
                            "exposure": 0.12,
                            "contrast": 5,
                            "highlights": -5,
                            "shadows": 4,
                            "whites": 0,
                            "blacks": -2,
                            "gamma": 1.02,
                        },
                    },
                ],
                "confidence": 0.9,
            }
        if "At stage 'look'" in prompt:
            return {
                "operations": [
                    {
                        "type": "color_wheels",
                        "parameters": {
                            "shadows": {"hue": -150, "saturation": 4},
                            "highlights": {"hue": 35, "saturation": 3},
                            "balance": 0,
                        },
                    },
                    {
                        "type": "curves",
                        "parameters": {
                            "rgb": [[0, 0], [0.5, 0.52], [1, 1]],
                            "strength": 0.4,
                        },
                    },
                    {
                        "type": "global_color",
                        "parameters": {"saturation": 2, "vibrance": 5},
                    },
                ],
                "confidence": 0.9,
            }
        if "At stage 'selective_color'" in prompt:
            return {
                "operations": [
                    {
                        "type": "hsl8",
                        "parameters": {
                            "orange": {"saturation": 3, "luminance": 2},
                            "blue": {"hue": -2, "saturation": -3},
                        },
                    }
                ],
                "confidence": 0.9,
            }
        if "At stage 'texture'" in prompt:
            return {
                "operations": [
                    {
                        "type": "texture",
                        "parameters": {
                            "clarity": 4,
                            "texture": 3,
                            "dehaze": 1,
                            "sharpening": 5,
                        },
                    }
                ],
                "confidence": 0.9,
            }
        if "At stage 'optical'" in prompt:
            return {
                "operations": [
                    {
                        "type": "optical_effects",
                        "parameters": {
                            "vignette": {
                                "amount": -3,
                                "midpoint": 55,
                                "feather": 75,
                            },
                            "grain": {"amount": 2, "size": 20, "roughness": 40},
                        },
                    }
                ],
                "confidence": 0.9,
            }
        raise AssertionError(prompt[:200])


def gradient_frame(offset):
    axis = np.linspace(45 + offset, 185 + offset, 32, dtype=np.float32)
    gray = np.tile(axis[None, :], (24, 1))
    rgb = np.stack((gray * 1.02, gray, gray * 0.96), axis=-1)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


class GradePoolsV2Test(unittest.TestCase):
    def test_every_pool_has_a_canonical_neutral_state(self):
        for operation_type in POOL_OPERATION_TYPES:
            values = canonicalize_pool_parameters(operation_type, {})
            self.assertIsInstance(values, dict)

        with self.assertRaisesRegex(ValueError, "primary.exposure"):
            canonicalize_pool_parameters("primary", {"exposure": 6})

    def test_full_pool_pipeline_renders_and_emits_dynamic_tracks(self):
        frames = tuple(gradient_frame(offset) for offset in (0, 1, -1, 2, 0))
        pipeline = PoolGradePipeline(
            client=ScriptedPoolClient(),
            shot_planner=HeuristicShotPlanner(),
            anchors_per_shot=1,
            maximum_hero_attempts=1,
        )

        result = pipeline.run(frames, 5.0, "subtle cinematic color")
        rendered = tuple(
            render_grade_frames(
                frames,
                result.grade_graph.frame_parameters,
                operations=result.operations,
                operation_executor=OperationExecutor(),
                batch_size=2,
            )
        )

        self.assertEqual(
            {operation.operation_type for operation in result.operations},
            set(POOL_OPERATION_TYPES),
        )
        primary = next(
            operation
            for operation in result.operations
            if operation.operation_type == "primary"
        )
        self.assertEqual(len(primary.parameter_track), len(frames))
        self.assertFalse(result.metadata["legacy_12d_parameters_used"])
        self.assertGreater(
            float(np.mean(np.abs(np.asarray(rendered[0], dtype=np.float32) - np.asarray(frames[0], dtype=np.float32)))),
            0.5,
        )

    def test_grain_is_repeatable_per_frame_and_changes_over_time(self):
        executor = GradePoolExecutor()
        parameters = canonicalize_pool_parameters(
            "optical_effects", {"grain": {"amount": 30, "size": 20, "roughness": 50}}
        )

        class Operation:
            operation_type = "optical_effects"
            frame_range = (0, 2)
            parameter_track = ()

            def __init__(self, values):
                self.parameters = values

        source = gradient_frame(0)
        first = executor.apply(source, [Operation(parameters)], frame_index=0)
        repeated = executor.apply(source, [Operation(parameters)], frame_index=0)
        next_frame = executor.apply(source, [Operation(parameters)], frame_index=1)
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, next_frame))

    def test_unified_pool_backend_hides_legacy_parameter_vector(self):
        config = json.loads(
            (Path(__file__).parents[1] / "configs/unified_vl.gemini-full.json").read_text(
                encoding="utf-8"
            )
        )
        backend = build_unified_backend(
            config,
            client=ScriptedPoolClient(),
            allow_storyboard_fallback=True,
        )
        frames = tuple(gradient_frame(offset) for offset in (0, 1, -1, 2, 0))

        result = backend.process(
            VideoEditRequest(
                frames=frames,
                fps=5.0,
                instruction="subtle cinematic color",
            )
        )
        payload = result.to_dict()

        self.assertEqual(payload["schema_version"], "pool-grade-graph/v2")
        self.assertNotIn("parameter_schema", payload)
        self.assertNotIn("frame_parameters", payload)
        self.assertEqual(
            payload["operation_graph"]["schema_version"],
            "video-edit-operation-graph/v2",
        )
        self.assertEqual(
            set(payload["operation_graph"]["supported_operations"]),
            set(POOL_OPERATION_TYPES),
        )
        compact = result.to_dict(include_frame_parameters=False)
        self.assertTrue(compact["operation_graph"]["operations"])
        self.assertNotIn(
            "parameter_track", compact["operation_graph"]["operations"][0]
        )


if __name__ == "__main__":
    unittest.main()
