"""Build the standard text-to-video-grade-parameter benchmark track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prepare_safety_manifest import VIDEO_SUFFIXES


CASES = (
    (
        "bright-en",
        "make the video naturally brighter while preserving highlights",
        "en",
        {"exposure": 0.35},
    ),
    (
        "bright-zh",
        "自然提亮视频，同时保留高光细节",
        "zh",
        {"exposure": 0.35},
    ),
    (
        "warm-en",
        "give the video a restrained warm tone",
        "en",
        {"temperature": 0.30},
    ),
    (
        "warm-zh",
        "给视频克制、自然的暖色调",
        "zh",
        {"temperature": 0.30},
    ),
    (
        "cool-en",
        "make the video naturally cool without changing exposure",
        "en",
        {"temperature": -0.30},
    ),
    (
        "cool-zh",
        "让视频呈现自然冷调，同时保持曝光",
        "zh",
        {"temperature": -0.30},
    ),
    (
        "vivid-en",
        "make the colors vivid but not oversaturated",
        "en",
        {"saturation": 0.18, "vibrance": 0.25},
    ),
    (
        "vivid-zh",
        "让视频颜色更鲜活，但不要过饱和",
        "zh",
        {"saturation": 0.18, "vibrance": 0.25},
    ),
    (
        "cinematic-en",
        "apply a restrained cinematic grade with richer contrast",
        "en",
        {"contrast": 0.18, "saturation": -0.08, "tone_curve": 0.10},
    ),
    (
        "cinematic-zh",
        "使用克制的电影感调色，增加层次和对比度",
        "zh",
        {"contrast": 0.18, "saturation": -0.08, "tone_curve": 0.10},
    ),
)


def build_manifest(
    video_root: Path,
    *,
    limit_videos: int | None = None,
    max_frames: int = 48,
    languages: tuple[str, ...] = ("en", "zh"),
) -> dict[str, object]:
    video_root = Path(video_root).resolve()
    videos = sorted(
        path.resolve()
        for path in video_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if limit_videos is not None:
        videos = videos[:limit_videos]
    if not videos:
        raise RuntimeError("No videos were found for the intent benchmark.")
    samples = []
    selected_cases = [case for case in CASES if case[2] in languages]
    for video_index, video in enumerate(videos):
        for case_id, instruction, language, parameters in selected_cases:
            samples.append(
                {
                    "id": f"intent-{video_index:04d}-{video.stem}-{case_id}",
                    "input": str(video),
                    "max_frames": max_frames,
                    "language": language,
                    "intent": case_id.rsplit("-", 1)[0],
                    "instruction": instruction,
                    "target_parameters": parameters,
                    "expect_rollback": False,
                }
            )
    return {
        "schema_version": "training-free-video-benchmark-manifest/v1",
        "dataset": "VideoGradeBench controlled text-to-parameter track",
        "profile": "intent_parameter",
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument(
        "--language", choices=("en", "zh", "all"), default="all"
    )
    args = parser.parse_args()
    languages = ("en", "zh") if args.language == "all" else (args.language,)
    manifest = build_manifest(
        args.video_root,
        limit_videos=args.limit_videos,
        max_frames=args.max_frames,
        languages=languages,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
