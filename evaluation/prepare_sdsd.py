"""Create a paired-video benchmark manifest from an extracted SDSD dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _subset_root(root: Path, subset: str) -> Path:
    candidates = (
        root / subset,
        root / f"{subset}_png",
        root,
    )
    for candidate in candidates:
        if (candidate / "LQ").is_dir() and (candidate / "GT").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Cannot find {subset}/LQ and {subset}/GT below {root}. "
        "Point --root at the extracted SDSD PNG dataset."
    )


def build_manifest(
    root: Path,
    *,
    subsets: tuple[str, ...] = ("indoor", "outdoor"),
    limit: int | None = None,
    max_frames: int = 30,
    fps: float = 30.0,
) -> dict[str, object]:
    root = Path(root).resolve()
    samples = []
    for subset in subsets:
        location = _subset_root(root, subset)
        lq, gt = location / "LQ", location / "GT"
        scene_names = sorted(
            path.name
            for path in lq.iterdir()
            if path.is_dir() and (gt / path.name).is_dir()
        )
        for scene_name in scene_names:
            samples.append(
                {
                    "id": f"sdsd-{subset}-{scene_name}",
                    "subset": subset,
                    "input": str((lq / scene_name).resolve()),
                    "reference": str((gt / scene_name).resolve()),
                    "fps": fps,
                    "max_frames": max_frames,
                    "instruction": (
                        "恢复自然曝光和可见度，保持真实颜色、亮部细节和时间一致性"
                    ),
                    "expect_rollback": False,
                }
            )
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise RuntimeError("No aligned SDSD LQ/GT scene pairs were found.")
    return {
        "schema_version": "training-free-video-benchmark-manifest/v1",
        "dataset": "SDSD paired low-light video",
        "profile": "paired_quality",
        "fps": fps,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--subset",
        choices=("indoor", "outdoor", "all"),
        default="all",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    subsets = (
        ("indoor", "outdoor") if args.subset == "all" else (args.subset,)
    )
    payload = build_manifest(
        args.root,
        subsets=subsets,
        limit=args.limit,
        max_frames=args.max_frames,
        fps=args.fps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
