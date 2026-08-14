"""Build a controlled multi-shot video probe from real motion clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .video_benchmark import load_media


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frames-per-shot", type=int, default=24)
    parser.add_argument("--fps", type=float, default=24.0)
    args = parser.parse_args()
    if len(args.inputs) < 2:
        raise ValueError("Controlled cut probe requires at least two videos.")
    if args.frames_per_shot < 2:
        raise ValueError("frames-per-shot must be at least two.")

    clips = [
        load_media(path.resolve(), fps=args.fps, max_frames=args.frames_per_shot)
        for path in args.inputs
    ]
    output_root = args.output_root.resolve()
    frames_dir = output_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    target_size = clips[0].frames[0].size
    frame_index = 0
    boundaries = []
    for clip_index, clip in enumerate(clips):
        if clip_index > 0:
            boundaries.append(frame_index)
        for frame in clip.frames:
            if frame.size != target_size:
                frame = frame.resize(target_size, Image.Resampling.LANCZOS)
            frame.save(frames_dir / f"{frame_index:06d}.png")
            frame_index += 1

    manifest = {
        "schema_version": "training-free-video-benchmark-manifest/v1",
        "dataset": "controlled real-motion hard-cut probe",
        "profile": "storyboard",
        "fps": args.fps,
        "samples": [
            {
                "id": "controlled-hard-cuts",
                "input": "frames",
                "fps": args.fps,
                "instruction": "识别镜头边界并为视频调色选择代表性 Anchor",
                "shot_boundaries": boundaries,
                "shot_boundary_tolerance": 1,
            }
        ],
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
