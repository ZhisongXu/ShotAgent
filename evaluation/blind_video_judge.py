"""Run one anonymised MLLM development review over no-GT video outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

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
        raise ValueError("candidate_scores must contain every anonymous candidate exactly once")
    scores: dict[str, dict[str, object]] = {}
    fields = (
        "reference_style_match",
        "content_preservation",
        "temporal_consistency",
        "artifact_free",
        "overall_preference",
    )
    for code in codes:
        raw = raw_scores[code]
        if not isinstance(raw, dict):
            raise ValueError(f"{code} score must be an object")
        scores[code] = {field: _score(raw.get(field), f"{code}.{field}") for field in fields}
        scores[code]["rationale"] = str(raw.get("rationale", ""))[:500]
    return {"candidate_scores": scores, "review_summary": str(payload.get("review_summary", ""))[:1000]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    _load_env(args.env_file)
    media = args.review_dir / "blind_review_media" / args.sample
    candidates = sorted(media.glob("C*.mp4"))
    if len(candidates) < 2:
        raise ValueError("At least two anonymous candidates are required")
    labeled = [
        ("TARGET: original content video, ordered frames", _storyboard(media / "target.mp4")),
        ("REFERENCE: desired grading style, ordered frames", _storyboard(media / "reference.mp4")),
    ]
    codes = [path.stem for path in candidates]
    labeled.extend((f"ANONYMOUS CANDIDATE {path.stem}, ordered frames", _storyboard(path)) for path in candidates)
    prompt = """You are an independent evaluator for reference-video controlled color grading with no ground truth.

Judge anonymous candidates only from the supplied ordered storyboards. The TARGET defines content and geometry. The REFERENCE defines abstract grading style: tonal hierarchy, contrast character, palette relationships, color temperature, saturation character, highlight/shadow treatment, and atmosphere. Target and reference depict different content, so do not reward raw object-color or histogram coincidence. Do not reward a candidate merely for making a larger edit.

Give each candidate five scores from 1.0 to 5.0:
- reference_style_match: transfer of the reference's abstract grading look;
- content_preservation: target identity, geometry, texture and plausible materials;
- temporal_consistency: consistency across the ordered frames, acknowledging that storyboard evidence is limited;
- artifact_free: no clipping, crushing, halos, banding, unnatural casts or damaged skin/foliage;
- overall_preference: holistic preference balancing all of the above.

Use the full 1-5 range when evidence warrants it. Identity may preserve content but should score poorly on style if it does not transfer the look. Return JSON only:
{"candidate_scores":{"C01":{"reference_style_match":0,"content_preservation":0,"temporal_consistency":0,"artifact_free":0,"overall_preference":0,"rationale":"..."}},"review_summary":"..."}
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
        "schema": "reference-video-blind-mllm-review/v1",
        "sample": args.sample,
        "judge_model": args.model,
        "evidence": "8-frame ordered storyboards; development evaluation only",
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
