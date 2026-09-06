"""Run one anonymised MLLM development review over no-GT video outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import rankdata

from video_retouch.clients import OpenAIResponsesVisionClient
from video_retouch.io import decode_video

AXIS_METRICS = {
    "style": (
        "llm_reference_style_rating",
        "vgg_style_similarity",
        "lab_chroma_histogram_bhattacharyya",
    ),
    "content": (
        "content_structure_correlation",
        "dino_content_similarity",
        "edge_ssim",
    ),
    "temporal": (
        "temporal_flow_warp_error",
        "temporal_edit_warp_error",
        "temporal_transform_drift",
    ),
    "quality_artifact": (
        "musiq_score",
        "new_shadow_clip_fraction",
        "new_highlight_clip_fraction",
    ),
}


def attach_axis_rank_summary(report: dict[str, object]) -> dict[str, object]:
    """Attach equal-metric axis ranks and an equal-axis overall rank."""

    average_ranks = report.get("average_ranks")
    if not isinstance(average_ranks, dict):
        return report
    axis_ranks: dict[str, dict[str, float]] = {}
    overall_ranks: dict[str, float] = {}
    for method, raw_ranks in average_ranks.items():
        if not isinstance(raw_ranks, dict):
            continue
        method_axes: dict[str, float] = {}
        for axis, metrics in AXIS_METRICS.items():
            if all(metric in raw_ranks for metric in metrics):
                method_axes[axis] = float(
                    np.mean([float(raw_ranks[metric]) for metric in metrics])
                )
        axis_ranks[str(method)] = method_axes
        if len(method_axes) == len(AXIS_METRICS):
            overall_ranks[str(method)] = float(np.mean(list(method_axes.values())))
    report["axis_average_ranks"] = axis_ranks
    report["overall_axis_average_rank"] = overall_ranks
    report["axis_rank_policy"] = {
        "direction": "lower is better",
        "axis_metrics": {axis: list(metrics) for axis, metrics in AXIS_METRICS.items()},
        "overall": "unweighted mean of the four axis-average ranks",
    }
    return report


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip("'\"")


def _storyboard(path: Path, count: int = 8) -> Image.Image:
    decoded = decode_video(path, max_side=480)
    indices = np.unique(
        np.linspace(0, len(decoded.frames) - 1, min(count, len(decoded.frames)))
        .round()
        .astype(int)
    )
    cells = []
    for order, index in enumerate(indices.tolist(), start=1):
        image = decoded.frames[index].convert("RGB")
        image.thumbnail((320, 180), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (320, 204), "#111820")
        cell.paste(image, ((320 - image.width) // 2, (180 - image.height) // 2))
        ImageDraw.Draw(cell).text((8, 184), f"t{order:02d}", fill="white")
        cells.append(cell)
    columns = 4
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 320, rows * 204), "#111820")
    for index, cell in enumerate(cells):
        sheet.paste(cell, ((index % columns) * 320, (index // columns) * 204))
    return sheet


def _score(value: object, name: str) -> float:
    number = float(value)
    if not 1.0 <= number <= 5.0:
        raise ValueError(f"{name} must be between 1 and 5")
    return number


def _validate(payload: dict[str, object], codes: Sequence[str]) -> dict[str, object]:
    raw_scores = payload.get("candidate_scores")
    if not isinstance(raw_scores, dict) or set(raw_scores) != set(codes):
        raise ValueError(
            "candidate_scores must contain every anonymous candidate exactly once"
        )
    scores: dict[str, dict[str, object]] = {}
    style_fields = (
        "deep_shadow_black_level_match",
        "shadow_chroma_match",
        "midtone_luminance_match",
        "midtone_palette_match",
        "highlight_rolloff_match",
        "neutral_axis_temperature_match",
        "palette_hierarchy_match",
        "saturation_hierarchy_match",
        "local_contrast_depth_match",
    )
    diagnostic_fields = (
        "content_preservation",
        "temporal_consistency",
        "artifact_free",
        "overall_preference",
    )
    for code in codes:
        raw = raw_scores[code]
        if not isinstance(raw, dict):
            raise TypeError(f"{code} score must be an object")
        scores[code] = {
            field: _score(raw.get(field), f"{code}.{field}")
            for field in (*style_fields, *diagnostic_fields)
        }
        scores[code]["reference_style_match"] = float(
            np.mean([scores[code][field] for field in style_fields])
        )
        scores[code]["overall_grade_quality"] = float(
            np.mean(
                [
                    scores[code]["reference_style_match"],
                    scores[code]["content_preservation"],
                    scores[code]["temporal_consistency"],
                    scores[code]["artifact_free"],
                ]
            )
        )
        scores[code]["rationale"] = str(raw.get("rationale", ""))[:500]
    return {
        "candidate_scores": scores,
        "review_summary": str(payload.get("review_summary", ""))[:1000],
    }


def attach_reference_style_similarity(
    report_path: Path,
    key_path: Path,
    review: dict[str, object],
) -> dict[str, object]:
    """Attach normalized, de-anonymized LLM style similarity to a benchmark."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assignments = json.loads(key_path.read_text(encoding="utf-8"))
    sample = str(review["sample"])
    raw_scores = review["candidate_scores"]
    code_to_method = {
        str(item["candidate_code"]): str(item["method"])
        for item in assignments
        if str(item["sample"]) == sample
    }
    reference_rating_by_method = {
        code_to_method[code]: float(values["reference_style_match"])
        for code, values in raw_scores.items()
        if code in code_to_method
    }
    similarity_by_method = {
        code_to_method[code]: (float(values["reference_style_match"]) - 1.0) / 4.0
        for code, values in raw_scores.items()
        if code in code_to_method
    }
    overall_quality_rating_by_method = {
        code_to_method[code]: (
            float(values["overall_grade_quality"])
            if "overall_grade_quality" in values
            else float(
                np.mean(
                    [
                        values["reference_style_match"],
                        values["content_preservation"],
                        values["temporal_consistency"],
                        values["artifact_free"],
                    ]
                )
            )
        )
        for code, values in raw_scores.items()
        if code in code_to_method
    }
    for row in report["rows"]:
        if str(row["sample"]) != sample:
            continue
        method = str(row["method"])
        if method in similarity_by_method:
            row["llm_reference_style_similarity"] = similarity_by_method[method]
            row["llm_reference_style_rating"] = reference_rating_by_method[method]
            row["llm_overall_grade_quality_rating"] = overall_quality_rating_by_method[
                method
            ]
            row["llm_overall_grade_quality"] = (
                overall_quality_rating_by_method[method] - 1.0
            ) / 4.0

    llm_metrics = (
        "llm_reference_style_similarity",
        "llm_reference_style_rating",
        "llm_overall_grade_quality_rating",
        "llm_overall_grade_quality",
    )
    for method, values in report["aggregate"].items():
        for metric in llm_metrics:
            method_scores = [
                float(row[metric])
                for row in report["rows"]
                if row["method"] == method and metric in row
            ]
            if not method_scores:
                continue
            scores = np.asarray(method_scores, dtype=np.float64)
            mean = float(np.mean(scores))
            std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
            radius = 1.96 * std / np.sqrt(len(scores))
            values[metric] = mean
            if "aggregate_statistics" in report:
                report["aggregate_statistics"][method][metric] = {
                    "mean": mean,
                    "std": std,
                    "ci95_low": mean - radius,
                    "ci95_high": mean + radius,
                    "n": len(scores),
                }
    methods = list(report["aggregate"])
    samples = sorted({str(row["sample"]) for row in report["rows"]})
    for metric in llm_metrics:
        ranks: dict[str, list[float]] = {method: [] for method in methods}
        for sample_id in samples:
            sample_rows = {
                str(row["method"]): float(row[metric])
                for row in report["rows"]
                if row["sample"] == sample_id and metric in row
            }
            if len(sample_rows) != len(methods):
                continue
            sample_ranks = rankdata(
                [-sample_rows[method] for method in methods], method="average"
            )
            for method, rank in zip(methods, sample_ranks):
                ranks[method].append(float(rank))
        if "average_ranks" in report:
            for method in methods:
                if ranks[method]:
                    report["average_ranks"][method][metric] = float(
                        np.mean(ranks[method])
                    )
    attach_axis_rank_summary(report)
    report["llm_reference_style_evaluation"] = {
        "judge_model": review["judge_model"],
        "scale": "normalized from 1-5 to 0-1",
        "overall_grade_quality": (
            "unweighted mean of reference-style match, content preservation, "
            "temporal consistency, and artifact-free ratings; raw 1-5 value is "
            "stored as llm_overall_grade_quality_rating and headline quality is "
            "normalized with (rating - 1) / 4"
        ),
        "evidence": review["evidence"],
        "review_file": str(report_path.parent / "mllm_reference_style_review.json"),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = list(report["rows"][0])
    for row in report["rows"]:
        for key in row:
            if key not in columns:
                columns.append(key)
    with (report_path.parent / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(report["rows"])
    if "aggregate_statistics" in report:
        aggregate_rows = []
        for method in report["aggregate"]:
            method_statistics = report["aggregate_statistics"][method]
            aggregate_row: dict[str, object] = {
                "method": method,
                "sequence_count": max(
                    (int(values["n"]) for values in method_statistics.values()),
                    default=0,
                ),
            }
            for metric, statistics in method_statistics.items():
                for statistic in ("mean", "std", "ci95_low", "ci95_high"):
                    aggregate_row[f"{metric}_{statistic}"] = statistics[statistic]
                if metric in report.get("average_ranks", {}).get(method, {}):
                    aggregate_row[f"{metric}_average_rank"] = report["average_ranks"][
                        method
                    ][metric]
            for axis, rank in (
                report.get("axis_average_ranks", {}).get(method, {}).items()
            ):
                aggregate_row[f"{axis}_axis_average_rank"] = rank
            if method in report.get("overall_axis_average_rank", {}):
                aggregate_row["overall_axis_average_rank"] = report[
                    "overall_axis_average_rank"
                ][method]
            aggregate_rows.append(aggregate_row)
        aggregate_columns = list(
            dict.fromkeys(key for row in aggregate_rows for key in row)
        )
        with (report_path.parent / "aggregate.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=aggregate_columns)
            writer.writeheader()
            writer.writerows(aggregate_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--benchmark-report",
        type=Path,
        help="Optionally attach normalized LLM style similarity to report.json.",
    )
    args = parser.parse_args()
    _load_env(args.env_file)
    media = args.review_dir / "blind_review_media" / args.sample
    candidates = sorted(media.glob("C*.mp4"))
    if len(candidates) < 2:
        raise ValueError("At least two anonymous candidates are required")
    labeled = [
        (
            "TARGET: original content video, ordered frames",
            _storyboard(media / "target.mp4"),
        ),
        (
            "REFERENCE: desired grading style, ordered frames",
            _storyboard(media / "reference.mp4"),
        ),
    ]
    codes = [path.stem for path in candidates]
    labeled.extend(
        (f"ANONYMOUS CANDIDATE {path.stem}, ordered frames", _storyboard(path))
        for path in candidates
    )
    prompt = """You are an independent evaluator of production-ready, reference-video controlled color grading with no ground truth.

Judge anonymous candidates only from the supplied ordered storyboards. The TARGET defines the content, geometry, material identity, motion and recoverable detail. The REFERENCE defines the desired relationships among black level, tonal hierarchy, chroma, temperature, saturation, contrast, depth and mood. Because target and reference depict different content, evaluate whether the grading treatment has been translated appropriately to the target. Do not reward literal object-color or histogram coincidence, an indiscriminate global cast, or a larger edit by itself.

Treat a professional grade as a constrained transfer. It should express the reference treatment clearly while remaining plausible and production-ready on the target. Exact palette proximity does not compensate for damaged detail, implausible materials, unstable frames or technical artifacts. Likewise, an almost unchanged target does not deserve a high style score merely because it is clean. Score every field independently before forming an overall opinion, and apply the same standard to every anonymous candidate.

Give each candidate thirteen scores from 1.0 to 5.0. Judge these nine style fields independently:
- deep_shadow_black_level_match: floor, crushing/lift and retained near-black separation;
- shadow_chroma_match: shadow hue bias and chroma, separate from shadow brightness, while retaining meaningful color separation;
- midtone_luminance_match: middle-gray placement and the amount of open/luminous midtone detail;
- midtone_palette_match: dominant midtone hue families and relational remapping such as green to olive/taupe, without forcing unrelated objects toward one hue;
- highlight_rolloff_match: shoulder softness, diffuse-highlight brightness and specular containment;
- neutral_axis_temperature_match: white/gray balance from shadows through highlights, including warm/cool separation;
- palette_hierarchy_match: dominant versus secondary hue families and their prominence, adjusted for the different scene contents rather than raw pixel area;
- saturation_hierarchy_match: which tonal/semantic regions are restrained or emphasized, not merely global saturation;
- local_contrast_depth_match: microcontrast, haze/clarity and perceived depth while ignoring scene geometry;

Then judge these four diagnostic fields separately:
- content_preservation: target identity, geometry, fine edges, texture, local detail and plausible material appearance, without blur, reconstruction damage or semantic recoloring;
- temporal_consistency: one coherent grade across the ordered frames, without flicker, pumping, exposure jumps or changing color casts, acknowledging that storyboard evidence is limited;
- artifact_free: retained shadow and highlight detail with no clipping, crushing, halos, banding, posterization, color bleeding, over-smoothing, contaminated neutrals, or damaged skin, foliage and skies;
- overall_preference: production-ready holistic preference, giving comparable consideration to reference-style transfer, content/detail preservation, temporal consistency and artifact control.

Use these calibration anchors consistently: 5 means an excellent production-ready result with only negligible defects; 4 means clearly successful with minor defects; 3 means usable but with visible mismatch or degradation; 2 means major mismatch or damage; 1 means failure. A result with severe failure on any one of style transfer, content preservation, temporal consistency or artifact control should not receive an excellent overall_preference score.

The evaluator computes reference_style_match as the unweighted mean of the nine style-only fields. It also computes overall_grade_quality as the unweighted mean of reference_style_match, content_preservation, temporal_consistency and artifact_free. Do not return those derived fields yourself. Use the full 1-5 range when evidence warrants it. Identity may preserve content but should score poorly on style fields it does not transfer. Return JSON only:
{"candidate_scores":{"C01":{"deep_shadow_black_level_match":0,"shadow_chroma_match":0,"midtone_luminance_match":0,"midtone_palette_match":0,"highlight_rolloff_match":0,"neutral_axis_temperature_match":0,"palette_hierarchy_match":0,"saturation_hierarchy_match":0,"local_contrast_depth_match":0,"content_preservation":0,"temporal_consistency":0,"artifact_free":0,"overall_preference":0,"rationale":"..."}},"review_summary":"..."}
Include every supplied candidate code exactly once. Do not guess method identities."""
    client = OpenAIResponsesVisionClient(
        base_url="https://api.openai.com/v1",
        model_id=args.model,
        api_key_env="OPENAI_API_KEY",
        reasoning_effort="medium",
        image_detail="high",
        max_output_tokens=4096,
        max_image_side=1280,
    )
    result = _validate(client.generate_json(labeled, prompt), codes)
    report = {
        "schema": "reference-video-blind-mllm-review/v4",
        "sample": args.sample,
        "judge_model": args.model,
        "evidence": (
            "8-frame 1280-pixel ordered storyboards; nine style dimensions; "
            "four production-quality diagnostics; calibrated 1-5 anchors; "
            "development evaluation only"
        ),
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.benchmark_report is not None:
        attach_reference_style_similarity(
            args.benchmark_report,
            args.review_dir / "blind_review_key.json",
            report,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
