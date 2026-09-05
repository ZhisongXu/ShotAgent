"""Run the official CanonCGT checkpoint with a reference video.

CanonCGT accepts one reference image. This adapter uses the temporal middle
frame of the reference video for every target frame and preserves the target
video's frame size and cadence.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
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
    config_path = args.config.resolve()
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
        from models.networks.SSL_training import CanonCGT_SSL

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model = CanonCGT_SSL(SimpleNamespace(**config))
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=False)
        model = model.to(device).eval()

        style = _tensor(reference.frames[len(reference.frames) // 2], device)
        output_frames = []
        with torch.inference_mode():
            for index, frame in enumerate(target.frames, start=1):
                rendered = model(_tensor(frame, device), style)["restyled"][0]
                array = (
                    rendered.clamp(0.0, 1.0)
                    .permute(1, 2, 0)
                    .mul(255.0)
                    .add(0.5)
                    .byte()
                    .cpu()
                    .numpy()
                )
                output_frames.append(Image.fromarray(np.asarray(array), mode="RGB"))
                if index == 1 or index % 8 == 0 or index == len(target.frames):
                    print(f"CanonCGT frame {index}/{len(target.frames)}", flush=True)
        encode_video(output_frames, output_path, target.fps, preset="veryfast")
    finally:
        os.chdir(original_cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
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
