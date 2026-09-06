"""Run the official ICCV 2025 Video Color Grading implementation."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from video_retouch.io import decode_video, encode_video


def run(args: argparse.Namespace) -> None:
    repo_dir = args.repo_dir.resolve()
    config = args.config.resolve()
    output_path = args.output.resolve()
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
        # The released model vendors the Diffusers 0.21 ReferenceNet source.
        # PositionNet was later removed from the public embeddings module, but
        # the default Stable Diffusion configuration used by this checkpoint
        # never instantiates it.  Supplying the symbol keeps that official
        # source importable on the benchmark's current PyTorch/Diffusers stack.
        from diffusers.models import embeddings
        from diffusers.models.unets import unet_2d_blocks
        from diffusers.pipelines import pipeline_utils

        if not hasattr(embeddings, "PositionNet"):

            class PositionNet(torch.nn.Module):
                def __init__(self, *args: object, **kwargs: object) -> None:
                    super().__init__()
                    raise RuntimeError(
                        "PositionNet is unsupported for gated attention configurations"
                    )

            embeddings.PositionNet = PositionNet
        sys.modules.setdefault("diffusers.models.unet_2d_blocks", unet_2d_blocks)
        sys.modules.setdefault("diffusers.pipeline_utils", pipeline_utils)

        from grading import Inference

        with tempfile.TemporaryDirectory(prefix="video-color-grading-") as temp_text:
            temp = Path(temp_text)
            input_path = temp / "input.mp4"
            raw_output = temp / "official-output.mp4"
            encode_video(
                target.frames,
                input_path,
                target.fps,
                preset="veryfast",
                quality=args.encode_quality,
            )
            reference_image = np.asarray(
                reference.frames[len(reference.frames) // 2]
                .convert("RGB")
                .resize((args.model_size, args.model_size))
            )
            grader = Inference(config=str(config))
            grader(
                reference_image,
                str(input_path),
                str(raw_output),
                args.seed,
                args.steps,
                args.model_size,
                args.color_correction,
            )
            rendered = decode_video(raw_output)
            encode_video(
                rendered.frames,
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=72)
    parser.add_argument("--reference-frames", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument(
        "--model-size",
        type=int,
        default=512,
        help="Diffusion canvas; 512 gives the checkpoint's trained 64x64 LUT grid.",
    )
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--color-correction", action="store_true")
    parser.add_argument("--encode-quality", type=float, default=10.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
