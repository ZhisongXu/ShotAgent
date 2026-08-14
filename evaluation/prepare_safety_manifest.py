"""Create a VideoGradeBench safety/rollback manifest from real videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def build_manifest(
    video_root: Path,
    *,
    limit: int | None = None,
    max_frames: int = 48,
) -> dict[str, object]:
    video_root = Path(video_root).resolve()
    videos = sorted(
        path.resolve()
        for path in video_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if limit is not None:
        videos = videos[:limit]
    if not videos:
        raise RuntimeError("No videos were found for the safety benchmark.")
    return {
        "schema_version": "training-free-video-benchmark-manifest/v1",
        "dataset": "Video safety and rollback stress",
        "samples": [
            {
                "id": f"safety-{index:04d}-{video.stem}",
                "input": str(video),
                "max_frames": max_frames,
                "instruction": "保持专业、自然且时间稳定的视频调色",
            }
            for index, video in enumerate(videos)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-frames", type=int, default=48)
    args = parser.parse_args()
    manifest = build_manifest(
        args.video_root, limit=args.limit, max_frames=args.max_frames
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
