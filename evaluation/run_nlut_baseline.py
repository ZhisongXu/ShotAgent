"""Run the official NLUT checkpoint on a target/reference video pair.

NLUT learns one residual 3D LUT from a content image and a style image, then
applies that LUT to a complete video. For the common reference-video protocol,
this adapter uses the temporal middle frames of both videos. The released CUDA
interpolation extension is replaced by differentiable ``grid_sample`` so the
official fine-tuning procedure works on current PyTorch installations.
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
from torch import nn
from torchvision.transforms.functional import pil_to_tensor

from video_retouch.io import decode_video, encode_video


class TorchTrilinearInterpolation(nn.Module):
    """Differentiable evaluation of NLUT's [B, C, B, G, R] LUT tensor."""

    def forward(self, lut: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        if lut.ndim == 4:
            lut = lut.unsqueeze(0)
        if lut.shape[0] == 1 and image.shape[0] > 1:
            lut = lut.expand(image.shape[0], -1, -1, -1, -1)
        if lut.shape[0] != image.shape[0]:
            return torch.stack(
                [self.forward(item.unsqueeze(0), image) for item in lut], dim=1
            )
        # grid_sample's coordinates are ordered x/y/z, corresponding to the
        # LUT's R/G/B axes (W/H/D).
        grid = image.clamp(0.0, 1.0).permute(0, 2, 3, 1).mul(2.0).sub(1.0)
        grid = grid.unsqueeze(1)
        return F.grid_sample(
            lut,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(2)


def _install_extension_stub() -> None:
    # nlut_models imports the released extension at module load time. The
    # class above replaces its interpolation path immediately afterwards.
    extension = types.ModuleType("trilinear")
    extension.forward = lambda *unused: 1
    extension.backward = lambda *unused: 1
    sys.modules["trilinear"] = extension


def _tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    tensor = pil_to_tensor(image.convert("RGB")).float().div(255.0).unsqueeze(0)
    return F.interpolate(
        tensor.to(device), size=(256, 256), mode="bilinear", align_corners=False
    )


def _load_model(repo_dir: Path, checkpoint: Path, device: torch.device):
    _install_extension_stub()
    sys.path.insert(0, str(repo_dir))
    from nlut_models import NLUTNet

    model = NLUTNet("2048+32+32", dim=33).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    incompatible = model.load_state_dict(state["state_dict"], strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            f"NLUT checkpoint is missing required keys: {incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        print(
            "Ignoring obsolete checkpoint modules: "
            + ", ".join(incompatible.unexpected_keys),
            flush=True,
        )
    model.TrilinearInterpolation = TorchTrilinearInterpolation()
    return model


def _fit_lut(model, content: torch.Tensor, style: torch.Tensor, iterations: int):
    from nlut_models import TVMN

    model.train()
    content = content.repeat(2, 1, 1, 1)
    style = style.repeat(2, 1, 1, 1)
    regularizer = TVMN(33).to(content.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for index in range(iterations):
        stylized, _, auxiliary = model(content, content, style, TVMN=regularizer)
        tvmn = auxiliary["tvmn"]
        smooth_monotonic = 2_000_000.0 * (tvmn[0] + 10.0 * tvmn[2])
        smooth_monotonic += 2_000_000.0 * tvmn[1]
        content_loss, style_loss = model.encoder(content, style, stylized)
        loss = content_loss.mean() + style_loss.mean() + 100.0 * smooth_monotonic
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
        optimizer.step()
        if index == 0 or (index + 1) % 10 == 0 or index + 1 == iterations:
            print(
                f"NLUT fine-tune {index + 1}/{iterations}: loss={loss.item():.4f}",
                flush=True,
            )

    with torch.no_grad():
        _, _, auxiliary = model(content, content, style, TVMN=regularizer)
    return auxiliary["LUT"][:1].detach()


def run(args: argparse.Namespace) -> None:
    repo_dir = args.repo_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    output_path = args.output.resolve()
    device = torch.device(args.device)
    target = decode_video(
        args.target.resolve(), max_frames=args.max_frames, max_side=args.max_side
    )
    reference = decode_video(
        args.reference.resolve(),
        max_frames=args.reference_frames,
        max_side=args.max_side,
    )
    content = _tensor(target.frames[len(target.frames) // 2], device)
    style = _tensor(reference.frames[len(reference.frames) // 2], device)

    original_cwd = Path.cwd()
    os.chdir(repo_dir)
    try:
        model = _load_model(repo_dir, checkpoint, device)
        lut = _fit_lut(model, content, style, args.iterations)
        interpolation = TorchTrilinearInterpolation()
        output_frames = []
        with torch.inference_mode():
            for index, frame in enumerate(target.frames, start=1):
                full = pil_to_tensor(frame.convert("RGB")).float().div(255.0)
                full = full.unsqueeze(0).to(device)
                rendered = (full + interpolation(lut, full)).clamp(0.0, 1.0)
                array = (
                    rendered[0]
                    .permute(1, 2, 0)
                    .mul(255.0)
                    .add(0.5)
                    .byte()
                    .cpu()
                    .numpy()
                )
                output_frames.append(Image.fromarray(array, mode="RGB"))
                if index == 1 or index % 8 == 0 or index == len(target.frames):
                    print(f"NLUT frame {index}/{len(target.frames)}", flush=True)
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
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--max-frames", type=int, default=72)
    parser.add_argument("--reference-frames", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-quality", type=float, default=10.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
