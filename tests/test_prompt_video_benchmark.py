from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluation.prompt_video_benchmark import _edit_magnitude, _preset
from evaluation.run_cliptone_video_baseline import ailut_transform

ROOT = Path(__file__).resolve().parents[1]


def test_prompt_manifest_is_balanced_and_contains_no_visual_reference() -> None:
    payload = json.loads(
        (ROOT / "evaluation/manifests/prompt_video_grading_v1.json").read_text(
            encoding="utf-8"
        )
    )
    samples = payload["samples"]
    assert len(samples) == 8
    assert {sample["input_id"] for sample in samples} == {"girl", "city"}
    assert {sample["style_id"] for sample in samples} == {
        "neo-noir",
        "bleach-bypass",
        "1970s-35mm",
        "luxury-pastel",
    }
    assert all("reference" not in sample for sample in samples)
    assert all(len(sample["instruction"].split()) >= 35 for sample in samples)


def test_nlut3_prompt_manifest_disables_privileged_style_id_preset() -> None:
    payload = json.loads(
        (
            ROOT
            / "evaluation/manifests/prompt_video_grading_nlut3_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["include_text2preset"] is False
    assert len(payload["samples"]) == 3
    assert all("reference" not in sample for sample in payload["samples"])


def test_prompt_preset_changes_color_without_changing_geometry() -> None:
    pixels = np.zeros((24, 32, 3), dtype=np.uint8)
    pixels[..., 0] = np.arange(32, dtype=np.uint8)[None] * 7
    pixels[..., 1] = 120
    pixels[..., 2] = 190
    source = (Image.fromarray(pixels), Image.fromarray(np.flip(pixels, axis=1).copy()))
    output = _preset(source, "neo-noir")
    assert [frame.size for frame in output] == [frame.size for frame in source]
    assert _edit_magnitude(source, output) > 0.005


def test_pytorch_ailut_interpolator_preserves_identity_lut() -> None:
    values = torch.linspace(0.0, 1.0, 5)
    blue, green, red = torch.meshgrid(values, values, values, indexing="ij")
    lut = torch.stack((red, green, blue), dim=0).unsqueeze(0)
    vertices = values.repeat(1, 3, 1)
    image = torch.rand(1, 3, 9, 11)
    output = ailut_transform(image, lut, vertices)
    torch.testing.assert_close(output, image, atol=1e-6, rtol=1e-6)
