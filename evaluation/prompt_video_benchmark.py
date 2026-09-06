"""No-GT benchmark for pure-language video color grading.

The source video and a frozen text prompt are the only conditioning inputs.
Prompt adherence is judged from anonymous ordered storyboards; content,
temporal stability and technical quality are measured independently.  Edit
magnitude is diagnostic only and never treated as quality.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from evaluation.reference_video_benchmark import VideoData, _align_frames, metrics
from retouch_video import load_env_file
from video_retouch.clients import OpenAIResponsesVisionClient
from video_retouch.io import decode_video, encode_video

STYLE_FIELDS = (
    "black_and_tonal_hierarchy",
    "shadow_and_midtone_palette",
    "warm_cool_relationship",
    "saturation_hierarchy",
    "highlight_rolloff",
    "genre_mood",
)
QUALITY_FIELDS = (
    "content_preservation",
    "temporal_consistency",
    "artifact_free",
)


def _load(path: Path, sample: dict[str, object]) -> VideoData:
    decoded = decode_video(
        path,
        max_frames=sample.get("max_frames"),
        max_side=sample.get("max_side"),
    )
    return VideoData(decoded.frames, decoded.fps, decoded.source)


def _preset(frames: Sequence[Image.Image], style: str) -> tuple[Image.Image, ...]:
    """Frozen prompt-category baseline with no learned model or reference media."""

    settings = {
        "neo-noir": (1.12, 0.78, (-0.030, 0.000, 0.045), (0.045, 0.012, -0.025)),
        "bleach-bypass": (1.22, 0.48, (-0.012, -0.005, 0.018), (0.020, 0.012, 0.000)),
        "1970s-35mm": (0.96, 0.72, (-0.005, 0.010, 0.025), (0.050, 0.025, -0.018)),
        "luxury-pastel": (0.82, 0.82, (0.018, 0.010, 0.025), (0.038, 0.015, 0.035)),
    }
    contrast, saturation, shadow_rgb, highlight_rgb = settings[style]
    output = []
    for frame in frames:
        rgb = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
        luma = np.sum(rgb * np.array([0.2126, 0.7152, 0.0722]), axis=2, keepdims=True)
        gray = np.repeat(luma, 3, axis=2)
        rgb = gray + saturation * (rgb - gray)
        rgb = (rgb - 0.5) * contrast + 0.5
        shadow = np.clip((0.55 - luma) / 0.55, 0.0, 1.0)
        highlight = np.clip((luma - 0.45) / 0.55, 0.0, 1.0)
        rgb += shadow * np.asarray(shadow_rgb) + highlight * np.asarray(highlight_rgb)
        if style == "luxury-pastel":
            rgb = rgb * 0.93 + 0.055
        elif style == "1970s-35mm":
            rgb = np.maximum(rgb, 0.018)
        output.append(Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)))
    return tuple(output)


def _edit_magnitude(source: Sequence[Image.Image], output: Sequence[Image.Image]) -> float:
    values = []
    indices = np.unique(np.linspace(0, len(source) - 1, min(12, len(source))).round().astype(int))
    for index in indices:
        before = np.asarray(source[int(index)].convert("RGB"), dtype=np.float32) / 255.0
        after = np.asarray(output[int(index)].convert("RGB"), dtype=np.float32) / 255.0
        before_lab = cv2.cvtColor(before, cv2.COLOR_RGB2LAB)
        after_lab = cv2.cvtColor(after, cv2.COLOR_RGB2LAB)
        delta = np.abs(after_lab - before_lab) / np.array([100.0, 128.0, 128.0])
        values.append(float(np.mean(delta)))
    return float(np.mean(values))


def _storyboard(frames: Sequence[Image.Image], count: int = 8) -> Image.Image:
    indices = np.unique(np.linspace(0, len(frames) - 1, min(count, len(frames))).round().astype(int))
    canvas = Image.new("RGB", (4 * 320, 2 * 204), "#111820")
    draw = ImageDraw.Draw(canvas)
    for order, index in enumerate(indices.tolist(), start=1):
        image = frames[index].copy()
        image.thumbnail((320, 180), Image.Resampling.LANCZOS)
        x = ((order - 1) % 4) * 320
        y = ((order - 1) // 4) * 204
        canvas.paste(image, (x + (320 - image.width) // 2, y + (180 - image.height) // 2))
        draw.text((x + 8, y + 184), f"t{order:02d}", fill="white")
    return canvas


def _score(value: object, name: str) -> float:
    number = float(value)
    if not 1.0 <= number <= 5.0:
        raise ValueError(f"{name} must be between 1 and 5")
    return number


def _judge(
    client: OpenAIResponsesVisionClient,
    sample_id: str,
    prompt: str,
    source: VideoData,
    outputs: dict[str, tuple[Image.Image, ...]],
) -> tuple[dict[str, dict[str, float | str]], list[dict[str, str]]]:
    ordered = sorted(
        outputs,
        key=lambda method: hashlib.sha256(f"{sample_id}:{method}".encode()).hexdigest(),
    )
    assignments = [
        {"candidate_code": f"C{index:02d}", "method": method}
        for index, method in enumerate(ordered, start=1)
    ]
    labeled = [("SOURCE VIDEO: ordered frames", _storyboard(source.frames))]
    labeled.extend(
        (
            f"ANONYMOUS CANDIDATE {item['candidate_code']}: ordered frames",
            _storyboard(outputs[item["method"]]),
        )
        for item in assignments
    )
    codes = [item["candidate_code"] for item in assignments]
    instruction = f"""You are an independent evaluator of production-ready prompt-controlled video color grading with no visual reference and no ground truth.

The exact frozen grading instruction is:
<grading_instruction>{prompt}</grading_instruction>

Judge every anonymous candidate only from the SOURCE and ordered candidate storyboards. Evaluate literal fulfillment of the requested relationships, not keyword presence or edit size. A larger change is not better by itself; an almost unchanged result must score poorly when it fails to express the requested grade. Do not infer method identity. Use the same standard for every candidate.

Give scores from 1.0 to 5.0 for these six prompt-style fields:
- black_and_tonal_hierarchy: requested black floor, toe, middle-gray placement, contrast and retained shadow separation;
- shadow_and_midtone_palette: requested hue families and relational palette treatment rather than a uniform cast;
- warm_cool_relationship: requested separation among shadows, neutrals, skin/practicals and highlights;
- saturation_hierarchy: which tonal or semantic regions are restrained or emphasized;
- highlight_rolloff: requested shoulder softness, diffuse highlight placement and specular containment;
- genre_mood: how clearly the full grade reads as the named cinematic/commercial treatment.

Also score:
- content_preservation: identity, geometry, edges, texture and plausible material/skin appearance;
- temporal_consistency: a coherent grade through ordered frames without flicker, pumping or cast changes, within storyboard limits;
- artifact_free: no crushing, clipping, halos, banding, posterization, color bleeding, contaminated neutrals or damaged skin/sky;
- overall_preference: production-ready holistic preference considering prompt fulfillment, content, time and artifacts together.

Calibration: 5 excellent and production-ready; 4 clearly successful with minor defects; 3 usable with visible mismatch; 2 major mismatch or damage; 1 failure. Score fields independently. Return JSON only:
{{"candidate_scores":{{"C01":{{"black_and_tonal_hierarchy":0,"shadow_and_midtone_palette":0,"warm_cool_relationship":0,"saturation_hierarchy":0,"highlight_rolloff":0,"genre_mood":0,"content_preservation":0,"temporal_consistency":0,"artifact_free":0,"overall_preference":0,"rationale":"..."}}}},"review_summary":"..."}}
Include each candidate code exactly once."""
    raw = client.generate_json(labeled, instruction)
    candidate_scores = raw.get("candidate_scores")
    if not isinstance(candidate_scores, dict) or set(candidate_scores) != set(codes):
        raise ValueError("Judge response must contain every anonymous candidate once")
    code_to_method = {item["candidate_code"]: item["method"] for item in assignments}
    scores = {}
    for code in codes:
        candidate = candidate_scores[code]
        if not isinstance(candidate, dict):
            raise TypeError(f"{code} score must be an object")
        values = {field: _score(candidate.get(field), f"{code}.{field}") for field in (*STYLE_FIELDS, *QUALITY_FIELDS, "overall_preference")}
        style_rating = float(np.mean([values[field] for field in STYLE_FIELDS]))
        quality_rating = float(np.mean([values[field] for field in QUALITY_FIELDS]))
        balanced_rating = float(np.mean([style_rating, quality_rating]))
        scores[code_to_method[code]] = {
            **{field: (value - 1.0) / 4.0 for field, value in values.items()},
            "llm_prompt_style": (style_rating - 1.0) / 4.0,
            "llm_quality": (quality_rating - 1.0) / 4.0,
            "llm_balanced_overall": (balanced_rating - 1.0) / 4.0,
            "llm_rationale": str(candidate.get("rationale", ""))[:600],
        }
    return scores, assignments


def _mosaic(source: VideoData, outputs: dict[str, tuple[Image.Image, ...]], path: Path) -> None:
    labels = ["SOURCE", *[name.upper() for name in outputs]]
    sequences = [source.frames, *outputs.values()]
    frames = []
    for index in range(len(source.frames)):
        canvas = Image.new("RGB", (320 * len(sequences), 214), "#111820")
        draw = ImageDraw.Draw(canvas)
        for column, (label, sequence) in enumerate(zip(labels, sequences)):
            frame = sequence[index].copy()
            frame.thumbnail((320, 180), Image.Resampling.LANCZOS)
            canvas.paste(frame, (column * 320 + (320 - frame.width) // 2, 34 + (180 - frame.height) // 2))
            draw.text((column * 320 + 8, 9), label, fill="white")
        frames.append(canvas)
    encode_video(frames, path, source.fps, preset="veryfast")


def _external_video(root: Path, sample_id: str) -> Path:
    candidates = (root / f"{sample_id}.mp4", root / f"{sample_id}.result.mp4", root / sample_id / "output.mp4")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"No external output found for {sample_id} under {root}")
    return path


def run(
    manifest_path: Path,
    output_dir: Path,
    external: dict[str, Path],
    learned_metrics: bool,
    judge_model: str | None,
) -> dict[str, object]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any("reference" in sample for sample in payload["samples"]):
        raise ValueError("Pure-prompt manifests must not contain a reference field")
    learned_suite = None
    if learned_metrics:
        from evaluation.perceptual_metrics import LearnedMetricSuite

        learned_suite = LearnedMetricSuite(frame_count=8)
    judge = None
    if judge_model:
        judge = OpenAIResponsesVisionClient(
            base_url="https://api.openai.com/v1",
            model_id=judge_model,
            api_key_env="OPENAI_API_KEY",
            reasoning_effort="medium",
            image_detail="high",
            max_output_tokens=4096,
            max_image_side=1280,
        )
    root = manifest_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    blind_key = []
    for sample in payload["samples"]:
        sample_id = str(sample["id"])
        source_path = (root / sample["input"]).resolve()
        source = _load(source_path, sample)
        outputs = {"Identity": tuple(frame.copy() for frame in source.frames)}
        if bool(payload.get("include_text2preset", True)):
            outputs["Text2Preset"] = _preset(
                source.frames, str(sample["style_id"])
            )
        for method, method_root in external.items():
            decoded = _load(_external_video(method_root, sample_id), sample)
            outputs[method] = _align_frames(decoded.frames, source)
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        encode_video(source.frames, sample_dir / "source.mp4", source.fps, preset="veryfast")
        llm_scores = {}
        if judge is not None:
            llm_scores, assignments = _judge(
                judge, sample_id, str(sample["instruction"]), source, outputs
            )
            blind_key.extend({"sample": sample_id, **item} for item in assignments)
        for method, output in outputs.items():
            path = sample_dir / f"{method}.mp4"
            encode_video(output, path, source.fps, preset="veryfast")
            result_metrics = metrics(source, output)
            result_metrics["edit_magnitude_diagnostic"] = _edit_magnitude(source.frames, output)
            if learned_suite is not None:
                result_metrics["clip_prompt_similarity"] = learned_suite.clip_prompt_similarity(
                    output, str(sample["instruction"])
                )
                result_metrics["dino_content_similarity"] = learned_suite.dino_content_similarity(source.frames, output)
                result_metrics["musiq_score"] = learned_suite.no_reference_quality(output, "musiq")
            result_metrics.update(llm_scores.get(method, {}))
            rows.append(
                {
                    "sample": sample_id,
                    "input_id": sample["input_id"],
                    "style_id": sample["style_id"],
                    "method": method,
                    **result_metrics,
                }
            )
        _mosaic(source, outputs, sample_dir / "comparison.mp4")
        print(sample_id, flush=True)
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    def aggregate_rows(selected: list[dict[str, object]]) -> dict[str, object]:
        grouped = {}
        for method in methods:
            method_rows = [row for row in selected if row["method"] == method]
            if not method_rows:
                continue
            grouped[method] = {}
            for key in method_rows[0]:
                values = [
                    float(row[key])
                    for row in method_rows
                    if isinstance(row.get(key), (int, float))
                ]
                if values:
                    grouped[method][key] = {
                        "mean": float(np.mean(values)),
                        "std": (
                            float(np.std(values, ddof=1))
                            if len(values) > 1
                            else 0.0
                        ),
                        "n": len(values),
                    }
        return grouped

    aggregate = aggregate_rows(rows)
    style_aggregate = {
        style: aggregate_rows([row for row in rows if row["style_id"] == style])
        for style in sorted({str(row["style_id"]) for row in rows})
    }
    report = {
        "schema": "prompt-video-grade-benchmark/v1-no-gt",
        "dataset": payload.get("dataset"),
        "conditioning": "source video plus frozen text prompt only",
        "reference_media": None,
        "ranking_policy": {
            "primary": "llm_balanced_overall and overall_preference on anonymous candidates",
            "style": "llm_prompt_style (six literal prompt dimensions) plus raw CLIP RN50 prompt cosine",
            "quality": "llm_quality plus MUSIQ and artifact fractions",
            "edit_magnitude": "diagnostic only; excluded from quality and ranking",
        },
        "judge_model": judge_model,
        "rows": rows,
        "aggregate": aggregate,
        "style_aggregate": style_aggregate,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "blind_review_key.json").write_text(json.dumps(blind_key, indent=2, ensure_ascii=False), encoding="utf-8")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    aggregate_rows = []
    for method, method_values in aggregate.items():
        row = {"method": method}
        for metric_name, statistics in method_values.items():
            row[f"{metric_name}_mean"] = statistics["mean"]
            row[f"{metric_name}_std"] = statistics["std"]
        aggregate_rows.append(row)
    with (output_dir / "aggregate.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in aggregate_rows for key in row)))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--external", action="append", default=[], metavar="METHOD=DIR")
    parser.add_argument("--learned-metrics", action="store_true")
    parser.add_argument("--judge-model")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    load_env_file(args.env_file)
    external = {}
    for item in args.external:
        name, raw_path = item.split("=", 1)
        external[name] = Path(raw_path).resolve()
    report = run(
        args.manifest.resolve(),
        args.output_dir.resolve(),
        external,
        args.learned_metrics,
        args.judge_model,
    )
    print(json.dumps(report["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
