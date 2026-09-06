"""Run the official CLIPtone model on every frame of a prompt-video manifest.

CLIPtone is an image model.  This adapter keeps the text condition fixed for an
entire sequence and applies the official content-adaptive AiLUT frame by frame.
The repository's legacy CUDA extension is replaced by an equivalent inference-
only PyTorch trilinear interpolator so the published checkpoints run on current
PyTorch installations.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from video_retouch.io import decode_video, encode_video


def _gather(lut: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    flat = lut.flatten(2)
    return torch.gather(flat, 2, index[:, None].expand(-1, flat.shape[1], -1))


def ailut_transform(
    image: torch.Tensor, lut: torch.Tensor, vertices: torch.Tensor
) -> torch.Tensor:
    """Inference-only adaptive 3D-LUT transform matching CLIPtone's extension."""

    batch, _, height, width = image.shape
    size = lut.shape[-1]
    pixels = image.flatten(2)
    lower = []
    fractions = []
    for channel in range(3):
        axis = vertices[:, channel].contiguous()
        values = pixels[:, channel].contiguous()
        index = torch.searchsorted(axis, values, right=False) - 1
        index = index.clamp(0, size - 2)
        start = torch.gather(axis, 1, index)
        end = torch.gather(axis, 1, index + 1)
        lower.append(index)
        fractions.append((values - start) / (end - start + 1e-10))
    red, green, blue = lower
    rd, gd, bd = fractions
    base = red + size * green + size * size * blue
    offsets = (0, 1, size, size + 1, size * size, size * size + 1,
               size * size + size, size * size + size + 1)
    weights = (
        (1 - rd) * (1 - gd) * (1 - bd),
        rd * (1 - gd) * (1 - bd),
        (1 - rd) * gd * (1 - bd),
        rd * gd * (1 - bd),
        (1 - rd) * (1 - gd) * bd,
        rd * (1 - gd) * bd,
        (1 - rd) * gd * bd,
        rd * gd * bd,
    )
    output = sum(
        weight[:, None] * _gather(lut, base + offset)
        for weight, offset in zip(weights, offsets)
    )
    return output.reshape(batch, lut.shape[1], height, width)


def _load_modules(repository: Path):
    compatibility = types.ModuleType("ailut")
    compatibility.ailut_transform = ailut_transform
    sys.modules["ailut"] = compatibility
    sys.path.insert(0, str(repository))
    return importlib.import_module("ailutmodel"), importlib.import_module("adaptation")


def _load_clip(device: str):
    import clip

    model, _ = clip.load("RN50", device=device)
    return model.eval().requires_grad_(False)


def _text_direction(model, prompt: str, device: str):
    import clip

    tokens = clip.tokenize(["Normal photo.", prompt + " photo."], truncate=True).to(
        device
    )
    with torch.inference_mode():
        features = model.encode_text(tokens).float()
    features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    direction = features[1:2] - features[0:1]
    return direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _load_models(repository: Path, device: str):
    ailut_module, adaptation_module = _load_modules(repository)
    settings = SimpleNamespace(
        backbone="tpami",
        n_ranks=3,
        n_vertices=33,
        n_colors=3,
        en_adaint=True,
        en_adaint_share=False,
        pretrained=False,
    )
    model = ailut_module.AiLUT(settings).to(device)
    base_path = repository / "checkpoint/base_network/AiLUT-FiveK-sRGB.pth"
    state = torch.load(base_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    text_model = _load_clip(device)
    dummy_direction = _text_direction(text_model, "cinematic", device)
    adaptor = adaptation_module.AdaptationModule(
        settings, dummy_direction.shape[-1], model.backbone.out_channels
    ).to(device)
    adapter_path = repository / "checkpoint/RN50/pretrained.pth"
    adaptor.load_state_dict(
        torch.load(adapter_path, map_location="cpu", weights_only=False)
    )
    return (
        model.eval().requires_grad_(False),
        adaptor.eval().requires_grad_(False),
        text_model,
    )


def _align_size(image: Image.Image, maximum: int | None) -> Image.Image:
    if maximum is None or max(image.size) <= maximum:
        return image.convert("RGB")
    scale = maximum / max(image.size)
    return image.convert("RGB").resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.LANCZOS,
    )


def run(manifest_path: Path, repository: Path, output_dir: Path, batch_size: int) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, adaptor, text_model = _load_models(repository, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sample in payload["samples"]:
        path = (root / sample["input"]).resolve()
        decoded = decode_video(
            path,
            max_frames=sample.get("max_frames"),
            max_side=sample.get("max_side"),
        )
        prompt = str(sample["instruction"])
        direction = _text_direction(text_model, prompt, device)
        with torch.inference_mode():
            deltas = adaptor(direction, 1.0)
        frames = []
        for offset in range(0, len(decoded.frames), batch_size):
            images = [
                _align_size(frame, sample.get("max_side"))
                for frame in decoded.frames[offset : offset + batch_size]
            ]
            tensor = torch.stack(
                [pil_to_tensor(image).float().div(255.0) for image in images]
            ).to(device)
            with torch.inference_mode():
                codes = model.backbone(tensor)
                _, luts = model.lut_generator(codes, deltas[0])
                vertices = model.adaint(codes, deltas[1])
                result = ailut_transform(tensor, luts, vertices).clamp(0, 1)
            frames.extend(to_pil_image(frame.cpu()) for frame in result)
        destination = output_dir / f"{sample['id']}.mp4"
        encode_video(frames, destination, decoded.fps, preset="veryfast")
        records.append(
            {
                "sample": sample["id"],
                "prompt": prompt,
                "output": str(destination),
                "frame_count": len(frames),
            }
        )
        print(destination, flush=True)
    (output_dir / "run.json").write_text(
        json.dumps(
            {
                "method": "CLIPtone (Lee et al., CVPR 2024)",
                "adapter": "official image model applied framewise with fixed prompt",
                "prompt_token_policy": "official CLIP RN50 77-token limit; truncate tail",
                "checkpoint_repository": "https://github.com/hmin970922/CLIPtone",
                "samples": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    run(
        args.manifest.resolve(),
        args.repository.resolve(),
        args.output_dir.resolve(),
        args.batch_size,
    )


if __name__ == "__main__":
    main()
