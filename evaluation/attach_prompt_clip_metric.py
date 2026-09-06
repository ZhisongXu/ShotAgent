"""Attach raw CLIP prompt cosine scores to a completed prompt-video report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluation.perceptual_metrics import LearnedMetricSuite
from video_retouch.io import decode_video


def _statistics(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def attach(manifest_path: Path, benchmark_dir: Path, device: str | None = None) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompts = {str(item["id"]): str(item["instruction"]) for item in manifest["samples"]}
    report_path = benchmark_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    suite = LearnedMetricSuite(frame_count=8, device=device)

    rows = report["rows"]
    for index, row in enumerate(rows, start=1):
        row["llm_balanced_overall"] = float(
            np.mean([row["llm_prompt_style"], row["llm_quality"]])
        )
        sample_id = str(row["sample"])
        video_path = benchmark_dir / sample_id / f"{row['method']}.mp4"
        frames = decode_video(video_path).frames
        row["clip_prompt_similarity"] = suite.clip_prompt_similarity(
            frames, prompts[sample_id]
        )
        print(f"[{index}/{len(rows)}] {sample_id}: {row['method']}", flush=True)

    for method, values in report["aggregate"].items():
        balanced = [
            float(row["llm_balanced_overall"])
            for row in rows
            if row["method"] == method
        ]
        values["llm_balanced_overall"] = _statistics(balanced)
        scores = [
            float(row["clip_prompt_similarity"])
            for row in rows
            if row["method"] == method
        ]
        values["clip_prompt_similarity"] = _statistics(scores)
    for style, methods in report["style_aggregate"].items():
        for method, values in methods.items():
            balanced = [
                float(row["llm_balanced_overall"])
                for row in rows
                if row["style_id"] == style and row["method"] == method
            ]
            values["llm_balanced_overall"] = _statistics(balanced)
            scores = [
                float(row["clip_prompt_similarity"])
                for row in rows
                if row["style_id"] == style and row["method"] == method
            ]
            values["clip_prompt_similarity"] = _statistics(scores)

    report["ranking_policy"]["style"] = (
        "llm_prompt_style (six literal prompt dimensions) plus raw CLIP RN50 prompt cosine"
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (benchmark_dir / "results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows = []
    for method, method_values in report["aggregate"].items():
        row = {"method": method}
        for metric_name, statistics in method_values.items():
            row[f"{metric_name}_mean"] = statistics["mean"]
            row[f"{metric_name}_std"] = statistics["std"]
        aggregate_rows.append(row)
    columns = list(dict.fromkeys(key for row in aggregate_rows for key in row))
    with (benchmark_dir / "aggregate.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(aggregate_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    attach(args.manifest.resolve(), args.benchmark_dir.resolve(), args.device)


if __name__ == "__main__":
    main()
