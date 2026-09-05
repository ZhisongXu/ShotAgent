"""Run the official SA-LUT checkpoint on a target/reference video pair.

SA-LUT accepts one style image. For the common video-reference protocol this
adapter uses the temporal middle frame of the reference video for every target
frame. The repository's custom CUDA interpolation extension is replaced by an
equivalent inference-only PyTorch implementation so the baseline can run on
current PyTorch/CUDA builds.
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from video_retouch.io import decode_video, encode_video


def _quadrilinear(lut: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Evaluate a [B,3,2,D,D,D] LUT at [context,R,G,B] image values."""

    if lut.ndim == 5:
        lut = lut.unsqueeze(0)
    batch, _, height, width = values.shape
    dimension = lut.shape[-1]
    rgb = values[:, 1:4].clamp(0.0, 1.0)
    context = values[:, 0].clamp(0.0, 1.0)
    positions = rgb * (dimension - 1)
    indices = torch.floor(positions).long().clamp(0, dimension - 2)
    fractions = positions - indices
    result = values.new_zeros((batch, 3, height, width))
    for batch_index in range(batch):
        red_index, green_index, blue_index = indices[batch_index]
        red_fraction, green_fraction, blue_fraction = fractions[batch_index]
        context_fraction = context[batch_index]
        for channel in range(3):
            channel_result = values.new_zeros((height, width))
            for context_offset in (0, 1):
                context_weight = (
                    1.0 - context_fraction if context_offset == 0 else context_fraction
                )
                for red_offset in (0, 1):
                    red_weight = 1.0 - red_fraction if red_offset == 0 else red_fraction
                    for green_offset in (0, 1):
                        green_weight = (
                            1.0 - green_fraction
                            if green_offset == 0
                            else green_fraction
                        )
                        for blue_offset in (0, 1):
                            blue_weight = (
                                1.0 - blue_fraction
                                if blue_offset == 0
                                else blue_fraction
                            )
                            channel_result += (
                                context_weight
                                * red_weight
                                * green_weight
                                * blue_weight
                                * lut[
                                    batch_index,
                                    channel,
                                    context_offset,
                                    blue_index + blue_offset,
                                    green_index + green_offset,
                                    red_index + red_offset,
                                ]
                            )
            result[batch_index, channel] = channel_result
    return result


def _install_extension_fallbacks() -> None:
    trilinear = types.ModuleType("trilinear")

    def trilinear_forward(lut, image, output, *unused):
        # This value is computed but unused by the released SA-LUT forward pass.
        del lut, unused
        output.copy_(image)
        return 1

    trilinear.forward = trilinear_forward
    trilinear.backward = lambda *unused: 1
    sys.modules["trilinear"] = trilinear

    quadrilinear = types.ModuleType("quadrilinear4d")

    def quadrilinear_forward(lut, image, output, *unused):
        del unused
        output.copy_(_quadrilinear(lut, image))
        return 1

    quadrilinear.forward = quadrilinear_forward
    quadrilinear.backward = lambda *unused: 1
    sys.modules["quadrilinear4d"] = quadrilinear


def _tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    return (
        pil_to_tensor(image.convert("RGB")).float().div(255.0).unsqueeze(0).to(device)
    )


def run(args: argparse.Namespace) -> None:
    repo_dir = args.repo_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    target_path = args.target.resolve()
    reference_path = args.reference.resolve()
    output_path = args.output.resolve()
    _install_extension_fallbacks()
    sys.path.insert(0, str(repo_dir))
    original_cwd = Path.cwd()
    os.chdir(repo_dir)
    try:
        from inference_cli import load_salut_from_ckpt

        device = torch.device(args.device)
        model = load_salut_from_ckpt(str(checkpoint), device)
        # The released 512 setting forms a 65K x 65K attention matrix. A 256
        # analysis resolution is the official module's supported resize path
        # and fits an 8 GB evaluation GPU.
        model.context_extractor.target_resolution = (args.analysis_size,) * 2
        target = decode_video(
            target_path, max_frames=args.max_frames, max_side=args.max_side
        )
        reference = decode_video(
            reference_path, max_frames=args.reference_frames, max_side=args.max_side
        )
        style = reference.frames[len(reference.frames) // 2].convert("RGB")
        style_tensor = F.interpolate(
            _tensor(style, device),
            size=(args.analysis_size, args.analysis_size),
            mode="bilinear",
            align_corners=True,
        )
        output_frames = []
        with torch.inference_mode():
            for index, frame in enumerate(target.frames, start=1):
                full = _tensor(frame, device)
                content = F.interpolate(
                    full,
                    size=(args.analysis_size, args.analysis_size),
                    mode="bilinear",
                    align_corners=True,
                )
                _, fused_lut, context = model(style_tensor, content)
                context_full = F.interpolate(
                    context,
                    size=full.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                rendered = _quadrilinear(
                    fused_lut, torch.cat([context_full, full], dim=1)
                )
                array = (
                    rendered[0]
                    .clamp(0.0, 1.0)
                    .permute(1, 2, 0)
                    .mul(255.0)
                    .add(0.5)
                    .byte()
                    .cpu()
                    .numpy()
                )
                output_frames.append(Image.fromarray(array, mode="RGB"))
                print(f"SA-LUT frame {index}/{len(target.frames)}", flush=True)
        encode_video(
            output_frames,
            output_path,
            target.fps,
            preset="veryfast",
            quality=args.encode_quality,
        )
    finally:
        os.chdir(original_cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-size", type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=72)
    parser.add_argument("--reference-frames", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-quality", type=float, default=10.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
