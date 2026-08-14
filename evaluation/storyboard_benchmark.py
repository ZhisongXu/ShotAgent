"""Evaluate Video Perceiver shot boundaries and task-aware Anchor coverage."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from video_retouch import HeuristicShotPlanner, VLShotPlanner
from video_retouch.agent_config import load_multi_agent_runtime

from .video_benchmark import (
    _anchor_metrics,
    _boundary_metrics,
    _resolve_path,
    load_manifest,
    load_media,
)
from .provenance import file_sha256


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    successful = [row for row in rows if "error" not in row]
    names = sorted(
        {
            name
            for row in successful
            for name, value in row["metrics"].items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
    )
    metrics = {
        name: float(np.mean([float(row["metrics"][name]) for row in successful]))
        for name in names
        if all(name in row["metrics"] for row in successful)
    }
    return {
        "sample_count": len(rows),
        "successful_samples": len(successful),
        "failed_samples": len(rows) - len(successful),
        "metrics": metrics,
    }


def evaluate_storyboard_manifest(
    manifest_path: Path,
    *,
    agent_config: Optional[Path] = None,
    anchors_per_shot: int = 1,
    fail_fast: bool = False,
) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve()
    payload = load_manifest(manifest_path)
    if agent_config is None:
        planner = HeuristicShotPlanner()
        runtime = {
            "mode": "offline-native-training-free",
            "storyboard": "heuristic-shot-planner",
        }
    else:
        configured = load_multi_agent_runtime(agent_config)
        planner = VLShotPlanner(
            client=configured.storyboard_client,
            settings=configured.storyboard_settings,
            strict=True,
        )
        runtime = configured.manifest["storyboard"]

    rows = []
    for index, raw_sample in enumerate(payload["samples"]):
        sample = dict(raw_sample)
        sample_id = str(sample.get("id", f"sample-{index:04d}"))
        try:
            media = load_media(
                _resolve_path(sample.get("input"), manifest_path.parent, "input"),
                fps=float(sample.get("fps", payload.get("fps", 30.0))),
                max_frames=(
                    None
                    if sample.get("max_frames") is None
                    else int(sample["max_frames"])
                ),
                max_side=int(sample.get("analysis_max_side", 320)),
            )
            instruction = str(
                sample.get("instruction", "segment shots for consistent color grading")
            )
            storyboard = planner.plan(
                media.frames,
                media.fps,
                instruction,
                anchors_per_shot=anchors_per_shot,
            )
            predicted = [shot.start_frame for shot in storyboard.shots[1:]]
            target = sample.get("shot_boundaries", [])
            if not isinstance(target, list):
                raise ValueError("shot_boundaries must be a list of frame indices.")
            metrics = _boundary_metrics(
                predicted,
                [int(value) for value in target],
                int(sample.get("shot_boundary_tolerance", 2)),
            )
            metrics.update(_anchor_metrics(media.frames, storyboard))
            rows.append(
                {
                    "id": sample_id,
                    "input": str(media.source),
                    "frame_count": len(media.frames),
                    "predicted_shots": len(storyboard.shots),
                    "target_boundaries": len(target),
                    "metrics": metrics,
                    "storyboard": storyboard.to_dict(),
                }
            )
        except Exception as error:
            if fail_fast:
                raise
            rows.append(
                {
                    "id": sample_id,
                    "error": f"{type(error).__name__}: {error}",
                    "metrics": {},
                }
            )

    aggregate = _aggregate(rows)
    metrics = aggregate["metrics"]
    f1 = float(metrics.get("shot_boundary_f1", 0.0))
    coverage = float(metrics.get("anchor_mean_coverage_error", float("inf")))
    primary_score = 0.8 * min(max(f1, 0.0), 1.0) + 0.2 * max(
        0.0, 1.0 - coverage / 2.0
    )
    return {
        "schema_version": "videogradebench-storyboard/v1",
        "dataset": str(payload.get("dataset", manifest_path.stem)),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "runtime": runtime,
        "primary_score": float(primary_score),
        "aggregate": aggregate,
        "samples": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path)
    parser.add_argument("--anchors-per-shot", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    report = evaluate_storyboard_manifest(
        args.manifest,
        agent_config=args.agent_config,
        anchors_per_shot=args.anchors_per_shot,
        fail_fast=args.fail_fast,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
