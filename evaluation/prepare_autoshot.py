"""Create a storyboard benchmark manifest from the AutoShot SHOT dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_annotations(path: Path) -> list[tuple[str, int, list[int]]]:
    blocks = Path(path).read_text(encoding="utf-8").strip().split("\n\n")
    rows = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        header = lines[0].rsplit(maxsplit=1)
        if len(header) != 2:
            continue
        filename, frame_count_text = header
        boundaries = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) != 2:
                continue
            start, end = int(parts[0]), int(parts[1])
            if end >= start:
                boundaries.append(end)
        rows.append((filename, int(frame_count_text), boundaries))
    return rows


def build_manifest(
    video_root: Path,
    annotations: Path,
    *,
    limit: int | None = None,
) -> dict[str, object]:
    video_root = Path(video_root).resolve()
    videos = {path.name: path.resolve() for path in video_root.rglob("*.mp4")}
    samples = []
    for filename, frame_count, boundaries in parse_annotations(annotations):
        video = videos.get(filename)
        if video is None:
            continue
        samples.append(
            {
                "id": f"autoshot-{Path(filename).stem}",
                "input": str(video),
                "declared_frame_count": frame_count,
                "instruction": "为视频调色识别镜头边界并选择代表性 Anchor",
                "shot_boundaries": boundaries,
                "shot_boundary_tolerance": 2,
            }
        )
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise RuntimeError("No AutoShot annotation/video pairs were found.")
    return {
        "schema_version": "training-free-video-benchmark-manifest/v1",
        "dataset": "AutoShot SHOT test videos",
        "profile": "storyboard",
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest = build_manifest(
        args.video_root, args.annotations, limit=args.limit
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
