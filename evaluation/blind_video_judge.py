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

from video_retouch.clients import OpenAIResponsesVisionClient
from video_retouch.io import decode_video


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
        image.thumbnail((240, 135), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (240, 157), "#111820")
        cell.paste(image, ((240 - image.width) // 2, (135 - image.height) // 2))
        ImageDraw.Draw(cell).text((7, 139), f"t{order:02d}", fill="white")
        cells.append(cell)
    columns = 4
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 240, rows * 157), "#111820")
    for index, cell in enumerate(cells):
        sheet.paste(cell, ((index % columns) * 240, (index // columns) * 157))
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
    similarity_by_method = {
        code_to_method[code]: (float(values["reference_style_match"]) - 1.0) / 4.0
        for code, values in raw_scores.items()
        if code in code_to_method
    }
    for row in report["rows"]:
        if str(row["sample"]) != sample:
            continue
        method = str(row["method"])
        if method in similarity_by_method:
            row["llm_reference_style_similarity"] = similarity_by_method[method]
    for method, values in report["aggregate"].items():
        method_scores = [
            float(row["llm_reference_style_similarity"])
            for row in report["rows"]
            if row["method"] == method and "llm_reference_style_similarity" in row
        ]
        if method_scores:
            values["llm_reference_style_similarity"] = float(np.mean(method_scores))
    report["llm_reference_style_evaluation"] = {
        "judge_model": review["judge_model"],
        "scale": "normalized from 1-5 to 0-1",
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
    prompt = """You are an independent evaluator for reference-video controlled color grading with no ground truth.

Judge anonymous candidates only from the supplied ordered storyboards. The TARGET defines content and geometry. The REFERENCE defines abstract grading style. Target and reference depict different content, so compare relationships and treatment rather than raw object colors or histograms. Do not reward a candidate merely for making a larger edit. Score style similarity independently from content preservation and artifacts: a candidate can match the style yet have poor preservation, and those failures belong in their separate fields.

Give each candidate thirteen scores from 1.0 to 5.0. Judge these nine style fields independently:
- deep_shadow_black_level_match: floor, crushing/lift and retained near-black separation;
- shadow_chroma_match: shadow hue bias and chroma, separate from shadow brightness;
- midtone_luminance_match: middle-gray placement and the amount of open/luminous midtone detail;
- midtone_palette_match: dominant midtone hue families and specific remapping such as green to olive/taupe;
- highlight_rolloff_match: shoulder softness, diffuse-highlight brightness and specular containment;
- neutral_axis_temperature_match: white/gray balance from shadows through highlights, including warm/cool separation;
- palette_hierarchy_match: dominant versus secondary hue families and their relative visual area, without matching unrelated object colors;
- saturation_hierarchy_match: which tonal/semantic regions are restrained or emphasized, not merely global saturation;
- local_contrast_depth_match: microcontrast, haze/clarity and perceived depth while ignoring scene geometry;

Then judge these four diagnostic fields separately:
- content_preservation: target identity, geometry, texture and plausible materials;
- temporal_consistency: consistency across the ordered frames, acknowledging that storyboard evidence is limited;
- artifact_free: no clipping, crushing, halos, banding, unnatural casts or damaged skin/foliage;
- overall_preference: holistic preference balancing all of the above.

The evaluator computes reference_style_match as the unweighted mean of the nine style-only fields. Do not return that derived field yourself. Use the full 1-5 range when evidence warrants it. Identity may preserve content but should score poorly on style fields it does not transfer. Return JSON only:
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
        "schema": "reference-video-blind-mllm-review/v3",
        "sample": args.sample,
        "judge_model": args.model,
        "evidence": (
            "8-frame ordered storyboards; nine style-only dimensions; "
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
