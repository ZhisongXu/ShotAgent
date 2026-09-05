"""Run the official CAP-VSTNet video checkpoint with a reference video.

CAP-VSTNet accepts one style image. This adapter selects the temporal middle
frame of the reference video, uses the released photorealistic video model
without semantic masks, and writes an RGB video at the target resolution.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from video_retouch.io import decode_video, encode_video


def _tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    return (
        pil_to_tensor(image.convert("RGB")).float().div(255.0).unsqueeze(0).to(device)
    )


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

    sys.path.insert(0, str(repo_dir))
    original_cwd = Path.cwd()
    os.chdir(repo_dir)
    try:
        from models.cWCT import cWCT
        from models.RevResNet import RevResNet
        from utils.utils import img_resize

        network = RevResNet(
            nBlocks=[10, 10, 10],
            nStrides=[1, 2, 2],
            nChannels=[16, 64, 256],
            in_channel=3,
            mult=4,
            hidden_dim=16,
            sp_steps=2,
        )
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        network.load_state_dict(state["state_dict"])
        network = network.to(device).eval()
        transfer = cWCT()

        style_image = reference.frames[len(reference.frames) // 2].convert("RGB")
        style_image = img_resize(
            style_image, args.max_side, down_scale=network.down_scale
        )
        style = _tensor(style_image, device)
        output_frames = []
        with torch.inference_mode():
            style_features = network(style, forward=True)
            for index, frame in enumerate(target.frames, start=1):
                original_size = frame.size
                content_image = img_resize(
                    frame.convert("RGB"), args.max_side, down_scale=network.down_scale
                )
                content = _tensor(content_image, device)
                content_features = network(content, forward=True)
                stylized_features = transfer.transfer(
                    content_features, style_features, None, None
                )
                stylized = network(stylized_features, forward=False)[0]
                array = (
                    stylized.clamp(0.0, 1.0)
                    .permute(1, 2, 0)
                    .mul(255.0)
                    .add(0.5)
                    .byte()
                    .cpu()
                    .numpy()
                )
                image = Image.fromarray(np.asarray(array), mode="RGB")
                if image.size != original_size:
                    image = image.resize(original_size, Image.Resampling.BICUBIC)
                output_frames.append(image)
                if index == 1 or index % 8 == 0 or index == len(target.frames):
                    print(f"CAP-VSTNet frame {index}/{len(target.frames)}", flush=True)
        encode_video(output_frames, output_path, target.fps, preset="veryfast")
    finally:
        os.chdir(original_cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=72)
    parser.add_argument("--reference-frames", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
