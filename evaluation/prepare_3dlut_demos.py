"""Download official 3D-LUT examples and create video-to-video demo pairs."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from video_retouch.io import decode_video, encode_video


BASE_URL = "https://raw.githubusercontent.com/seunghyuns98/VideoColorGrading/main/examples"
VIDEO_NAMES = ("video1.mp4", "video2.mp4", "video3.mp4")
PAIRS = (
    ("official-v1-to-v2", "video1.mp4", "video2.mp4"),
    ("official-v2-to-v3", "video2.mp4", "video3.mp4"),
    ("official-v3-to-v1", "video3.mp4", "video1.mp4"),
)


def _download(url: str, path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ShotAgent-benchmark"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as output:
        output.write(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--max-side", type=int, default=512)
    args = parser.parse_args()
    root = args.output_root.resolve()
    source_root = root / "official_sources"
    for name in VIDEO_NAMES:
        _download(f"{BASE_URL}/{name}", source_root / name)

    samples = []
    for sample_id, target_name, reference_name in PAIRS:
        case_dir = root / sample_id
        case_dir.mkdir(parents=True, exist_ok=True)
        target = decode_video(
            source_root / target_name,
            max_frames=args.max_frames,
            max_side=args.max_side,
        )
        reference = decode_video(
            source_root / reference_name,
            max_frames=args.max_frames,
            max_side=args.max_side,
        )
        encode_video(target.frames, case_dir / "target.mp4", target.fps, preset="veryfast")
        encode_video(
            reference.frames,
            case_dir / "reference.mp4",
            reference.fps,
            preset="veryfast",
        )
        samples.append(
            {
                "id": sample_id,
                "target": f"{sample_id}/target.mp4",
                "reference": f"{sample_id}/reference.mp4",
                "max_frames": args.max_frames,
                "max_side": args.max_side,
                "source": "VideoColorGrading official examples",
            }
        )
    manifest = {
        "schema": "reference-video-grade-manifest/v1",
        "dataset": "VideoColorGrading official examples (ICCV 2025)",
        "source_url": "https://github.com/seunghyuns98/VideoColorGrading/tree/main/examples",
        "samples": samples,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(root / "manifest.json")


if __name__ == "__main__":
    main()
