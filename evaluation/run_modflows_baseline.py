"""Run the official ModFlows checkpoint on a target/reference video pair.

ModFlows predicts an invertible RGB color mapping for each image. This video
adapter predicts the source and reference flows once from the temporal middle
frames, then applies that fixed mapping to every target frame. The policy keeps
the published image method temporally deterministic and matches the benchmark's
one-transform-per-sequence contract.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from video_retouch.io import decode_video, encode_video


def _encode_flow(
    encoder: torch.nn.Module,
    neural_ode_type: type[torch.nn.Module],
    preprocess: object,
    image: Image.Image,
    device: torch.device,
) -> torch.nn.Module:
    tensor = preprocess(image.convert("RGB"), False).unsqueeze(0).to(device)
    encoded = encoder(tensor).flatten()
    flow = neural_ode_type(input_dim=3, hidden=encoder.hidden, device=device)
    flow.set_weights(encoded)
    return flow


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
        from src.encoder import Encoder, enc_preprocess
        from src.neural_ode import NeuralODE

        encoder = Encoder(
            k_dim=8195,
            input_dim=4,
            hidden=1024,
            output_dim=3,
            device=device,
        )
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        encoder.load_state_dict(state)
        encoder.eval()

        with torch.inference_mode():
            source_flow = _encode_flow(
                encoder,
                NeuralODE,
                enc_preprocess,
                target.frames[len(target.frames) // 2],
                device,
            )
            reference_flow = _encode_flow(
                encoder,
                NeuralODE,
                enc_preprocess,
                reference.frames[len(reference.frames) // 2],
                device,
            )
            del encoder

            output_frames = []
            for index, frame in enumerate(target.frames, start=1):
                pixels = torch.from_numpy(
                    np.asarray(frame.convert("RGB"), dtype=np.float32).reshape(-1, 3)
                    / 255.0
                )
                rendered_chunks = []
                for chunk in pixels.split(args.pixel_chunk):
                    latent = source_flow.sample(
                        chunk.to(device), N=args.steps, strength=args.strength
                    )
                    rendered_chunks.append(
                        reference_flow.inv_sample(
                            latent, N=args.steps, strength=args.strength
                        ).cpu()
                    )
                rendered = torch.cat(rendered_chunks).clamp(0.0, 1.0)
                array = (
                    rendered.reshape(frame.height, frame.width, 3)
                    .mul(255.0)
                    .add(0.5)
                    .byte()
                    .numpy()
                )
                output_frames.append(Image.fromarray(array, mode="RGB"))
                if index == 1 or index % 8 == 0 or index == len(target.frames):
                    print(f"ModFlows frame {index}/{len(target.frames)}", flush=True)

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
    parser.add_argument("--max-frames", type=int, default=72)
    parser.add_argument("--reference-frames", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--pixel-chunk", type=int, default=131_072)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-quality", type=float, default=10.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
