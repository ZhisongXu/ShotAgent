"""Build a fixed hard multi-shot reference-video grading benchmark from AutoShot."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from evaluation.prepare_autoshot import parse_annotations
from video_retouch.io import encode_video


@dataclass(frozen=True)
class Candidate:
    path: Path
    frame_count: int
    boundaries: tuple[int, ...]
    fps: float
    shot_density: float
    luminance_swing: float
    chroma_swing: float
    visual_change: float
    window_start: int
    window_cuts: int


def _densest_window(boundaries: tuple[int, ...], length: int) -> tuple[int, int]:
    starts = [0]
    for boundary in boundaries:
        starts.extend((boundary, max(0, boundary - length + 1)))
    ranked = [
        (
            sum(start <= boundary < start + length for boundary in boundaries),
            -start,
            start,
        )
        for start in starts
    ]
    cuts, _, start = max(ranked)
    return start, cuts


def _analyze_video(
    path: Path,
    frame_count: int,
    boundaries: list[int],
    *,
    samples: int,
    clip_frames: int,
) -> Candidate | None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    usable_count = min(frame_count, decoded_count) if decoded_count > 0 else frame_count
    if fps <= 0 or usable_count < clip_frames:
        capture.release()
        return None

    positions = np.linspace(0, usable_count - 1, min(samples, usable_count), dtype=int)
    luminance = []
    chroma = []
    grays = []
    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok, frame = capture.read()
        if not ok:
            continue
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        luminance.append(float(np.mean(lab[..., 0]) / 255.0))
        chroma.append(
            [
                float(np.mean(lab[..., 1]) - 128.0),
                float(np.mean(lab[..., 2]) - 128.0),
            ]
        )
        grays.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32))
    capture.release()
    if len(luminance) < 4:
        return None

    luminance_array = np.asarray(luminance)
    chroma_array = np.asarray(chroma)
    visual_change = float(
        np.mean(
            [
                np.mean(np.abs(current - previous)) / 255.0
                for previous, current in pairwise(grays)
            ]
        )
    )
    valid_boundaries = tuple(
        boundary for boundary in boundaries if 0 <= boundary < usable_count
    )
    window_start, window_cuts = _densest_window(valid_boundaries, clip_frames)
    return Candidate(
        path=path,
        frame_count=usable_count,
        boundaries=valid_boundaries,
        fps=fps,
        shot_density=(len(valid_boundaries) + 1) / (usable_count / fps),
        luminance_swing=float(
            np.percentile(luminance_array, 90) - np.percentile(luminance_array, 10)
        ),
        chroma_swing=float(np.linalg.norm(np.std(chroma_array, axis=0))),
        visual_change=visual_change,
        window_start=window_start,
        window_cuts=window_cuts,
    )


def _percentile_ranks(values: list[float]) -> list[float]:
    order = np.argsort(np.argsort(np.asarray(values), kind="stable"), kind="stable")
    if len(values) == 1:
        return [1.0]
    return (order / (len(values) - 1)).astype(float).tolist()


def _hardness(candidates: list[Candidate]) -> list[float]:
    features = (
        ("shot_density", 0.40),
        ("luminance_swing", 0.25),
        ("chroma_swing", 0.20),
        ("visual_change", 0.15),
    )
    scores = np.zeros(len(candidates), dtype=np.float64)
    for name, weight in features:
        ranks = _percentile_ranks([float(getattr(item, name)) for item in candidates])
        scores += weight * np.asarray(ranks)
    return scores.tolist()


def _extract(candidate: Candidate, output: Path, clip_frames: int) -> None:
    capture = cv2.VideoCapture(str(candidate.path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, candidate.window_start)
    frames = []
    for _ in range(clip_frames):
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mode="RGB"))
    capture.release()
    if len(frames) != clip_frames:
        raise RuntimeError(
            f"Could not extract {clip_frames} frames from {candidate.path}"
        )
    encode_video(frames, output, candidate.fps, preset="medium", quality=5.0)


def build(args: argparse.Namespace) -> dict[str, object]:
    video_root = args.video_root.resolve()
    output = args.output.resolve()
    references = [path.resolve() for path in args.reference]
    videos = {path.name: path.resolve() for path in video_root.rglob("*.mp4")}
    candidates = []
    for filename, frame_count, boundaries in parse_annotations(args.annotations):
        path = videos.get(filename)
        if path is None:
            continue
        candidate = _analyze_video(
            path,
            frame_count,
            boundaries,
            samples=args.analysis_frames,
            clip_frames=args.clip_frames,
        )
        if candidate is not None and candidate.window_cuts >= args.min_window_cuts:
            candidates.append(candidate)
    if len(candidates) < args.limit:
        raise RuntimeError(
            f"Only {len(candidates)} eligible AutoShot videos; need {args.limit}"
        )

    scores = _hardness(candidates)
    selected = sorted(
        zip(candidates, scores), key=lambda item: (-item[1], item[0].path.name)
    )[: args.limit]
    clip_root = output.parent / f"{output.stem}_clips"
    clip_root.mkdir(parents=True, exist_ok=True)
    samples = []
    for index, (candidate, score) in enumerate(selected):
        clip_path = clip_root / f"{index + 1:02d}_{candidate.path.stem}.mp4"
        if not clip_path.is_file() or args.overwrite:
            _extract(candidate, clip_path, args.clip_frames)
        reference = references[index % len(references)]
        samples.append(
            {
                "id": f"autoshot-hard-{index + 1:02d}-{candidate.path.stem}",
                "target": os.path.relpath(clip_path, output.parent),
                "reference": os.path.relpath(reference, output.parent),
                "max_frames": args.clip_frames,
                "max_side": args.max_side,
                "source_video": str(candidate.path),
                "source_start_frame": candidate.window_start,
                "annotated_cuts_in_clip": candidate.window_cuts,
                "hardness": {
                    "score": score,
                    "shot_density_per_second": candidate.shot_density,
                    "luminance_swing": candidate.luminance_swing,
                    "chroma_swing": candidate.chroma_swing,
                    "visual_change": candidate.visual_change,
                },
            }
        )
    return {
        "schema": "reference-video-grade-manifest/v2",
        "dataset": "AutoShot-Hard multi-shot videos x VideoColorGrading references",
        "protocol": {
            "task": "hard multi-shot reference-video photorealistic color grading without ground truth",
            "selection": (
                "output-independent top hardness score: 40% shot density, 25% "
                "luminance swing, 20% chroma swing, 15% sampled visual change"
            ),
            "window": (
                f"densest {args.clip_frames}-frame window with at least "
                f"{args.min_window_cuts} annotated cuts"
            ),
            "pairing": "balanced fixed rotation over official reference videos",
            "aggregation": "compute every metric per sequence, then macro-average",
            "ranking": "per-metric average rank across sequences",
        },
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--clip-frames", type=int, default=144)
    parser.add_argument("--min-window-cuts", type=int, default=3)
    parser.add_argument("--analysis-frames", type=int, default=24)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.reference:
        parser.error("At least one --reference is required")
    manifest = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
