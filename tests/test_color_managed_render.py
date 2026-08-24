from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from video_retouch.color_management import COLOR_SPACES, ColorManager
from video_retouch.color_managed_render import render_color_managed_frames
from video_retouch.gpu_pool_executor import TorchGradePoolExecutor
from video_retouch.grade_pools import GradePoolExecutor, canonicalize_pool_parameters
from video_retouch.high_bit_io import decode_video_rgb16, encode_video_high_bit
from video_retouch.semantic_masks import SemanticMaskGenerator, SemanticMaskTracker


def _operation(operation_type: str, parameters: dict, *, mask: str = "global"):
    return SimpleNamespace(
        operation_type=operation_type,
        frame_range=(0, 1),
        parameters=canonicalize_pool_parameters(operation_type, parameters),
        parameter_track=(),
        mask_id=mask,
    )


@pytest.mark.parametrize("space", sorted(COLOR_SPACES))
def test_builtin_color_spaces_round_trip_through_acescg(space: str) -> None:
    values = np.linspace(0.03, 0.8, 72, dtype=np.float32).reshape(4, 6, 3)
    working = ColorManager.convert(values, space, "acescg")
    restored = ColorManager.convert(working, "acescg", space)
    assert np.all(np.isfinite(working))
    assert np.max(np.abs(restored - values)) < 2e-5


def test_torch_pool_batch_supports_parameter_tracks_and_masks() -> None:
    operation = _operation("primary", {"exposure": 0}, mask="sky")
    operation.parameter_track = (
        canonicalize_pool_parameters("primary", {"exposure": 1}),
        canonicalize_pool_parameters("primary", {"exposure": -1}),
    )
    images = torch.full((2, 3, 16, 20), 0.3)
    mask = torch.zeros((2, 16, 20))
    mask[:, :, :10] = 1
    result = TorchGradePoolExecutor().apply_batch(
        images, (operation,), frame_indices=(0, 1), masks={"sky": mask}
    )
    assert torch.allclose(result[:, :, :, 10:], images[:, :, :, 10:], atol=1e-6)
    assert result[0, :, :, :10].mean() > images[0, :, :, :10].mean()
    assert result[1, :, :, :10].mean() < images[1, :, :, :10].mean()


def test_torch_and_numpy_primary_are_close() -> None:
    operation = _operation(
        "primary",
        {
            "exposure": 0.7,
            "contrast": 18,
            "highlights": -20,
            "shadows": 15,
            "gamma": 1.1,
        },
    )
    rng = np.random.default_rng(4)
    rgb = rng.uniform(0.05, 0.85, (24, 32, 3)).astype(np.float32)
    expected = GradePoolExecutor().apply_array(rgb, (operation,), frame_index=0)
    actual = (
        TorchGradePoolExecutor()
        .apply_batch(
            torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0),
            (operation,),
            frame_indices=(0,),
        )[0]
        .permute(1, 2, 0)
        .numpy()
    )
    assert np.max(np.abs(actual - expected)) < 3e-5


def test_semantic_masks_generate_and_track() -> None:
    frame = np.zeros((144, 192, 3), dtype=np.uint8)
    frame[:72] = (80, 150, 230)
    frame[55:125, 75:115] = (210, 150, 120)
    generator = SemanticMaskGenerator()
    for name in ("skin", "sky", "person"):
        mask = generator.generate(Image.fromarray(frame), name)
        assert mask.shape == frame.shape[:2]
        assert mask.dtype == np.float32
        assert 0 <= float(mask.min()) <= float(mask.max()) <= 1
    tracked = SemanticMaskTracker(generator, detection_interval=2).track_many(
        (Image.fromarray(frame), Image.fromarray(np.roll(frame, 2, axis=1))),
        ("skin", "sky", "person"),
    )
    assert all(len(values) == 2 for values in tracked.values())


def test_color_managed_render_preserves_float_pipeline() -> None:
    frames = tuple(np.full((12, 16, 3), 0.18, np.float32) for _ in range(2))
    params = np.zeros((2, 12), np.float32)
    operation = _operation("primary", {"exposure": 1})
    output = tuple(
        render_color_managed_frames(
            frames,
            params,
            operations=(operation,),
            input_color_space="logc3",
            output_color_space="rec2020_pq",
            batch_size=2,
            device="cpu",
        )
    )
    assert len(output) == 2
    assert output[0].dtype == np.float32
    assert output[0].shape == frames[0].shape
    assert np.all(np.isfinite(output[0]))


def test_pq_identity_round_trip_uses_reference_white_scaling() -> None:
    frame = np.linspace(0.05, 0.75, 12 * 16 * 3, dtype=np.float32).reshape(12, 16, 3)
    output = tuple(
        render_color_managed_frames(
            (frame,),
            np.zeros((1, 12), np.float32),
            input_color_space="rec2020_pq",
            output_color_space="rec2020_pq",
            device="cpu",
        )
    )[0]
    assert np.max(np.abs(output - frame)) < 2e-4


@pytest.mark.parametrize("depth", (10, 12))
def test_ffmpeg_high_bit_round_trip(tmp_path: Path, depth: int) -> None:
    height, width = 96, 128
    gradient = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
    frame = np.broadcast_to(gradient, (height, width, 3)).copy()
    path = tmp_path / f"roundtrip-{depth}.mp4"
    encode_video_high_bit((frame, frame), path, 12.0, bit_depth=depth)
    info, decoded = decode_video_rgb16(path, max_frames=2)
    assert info.width == width and info.height == height
    assert len(decoded) == 2
    assert decoded[0].dtype == np.float32
    assert np.mean(np.abs(decoded[0] - frame)) < 0.035
