"""Stress-test video Critic rejection decisions used by transactional rollback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from retouch_agent import RetouchExecutor, RetouchParameters
from video_retouch.backends import AnchorGrade
from video_retouch.critic import PhotoAgentStyleCritic
from video_retouch.models import ShotPlan

from .video_benchmark import _resolve_path, load_manifest, load_media
from .provenance import file_sha256


def _vector(mapping: object) -> np.ndarray:
    if not isinstance(mapping, Mapping):
        raise ValueError("Stress-case parameters must be a JSON object.")
    values = RetouchParameters.from_mapping(mapping, clamp=False).to_vector()
    if np.any(np.abs(values[9:]) > 1e-12):
        raise ValueError("Video safety stress cases currently support global parameters.")
    return values


def _trajectory(case: Mapping[str, object], frame_count: int) -> np.ndarray:
    mode = str(case.get("mode", "constant"))
    if mode == "constant":
        return np.repeat(_vector(case.get("parameters", {}))[None, :], frame_count, 0)
    first = _vector(case.get("first", {}))
    second = _vector(case.get("second", {}))
    if mode == "alternating":
        return np.stack(
            [first if index % 2 == 0 else second for index in range(frame_count)]
        )
    if mode == "step":
        split = int(case.get("split_frame", frame_count // 2))
        return np.stack(
            [first if index < split else second for index in range(frame_count)]
        )
    if mode == "spike":
        spike = int(case.get("spike_frame", frame_count // 2))
        values = np.repeat(first[None, :], frame_count, 0)
        values[min(max(spike, 0), frame_count - 1)] = second
        return values
    raise ValueError(f"Unsupported safety stress mode: {mode}")


def default_cases() -> list[dict[str, object]]:
    return [
        {
            "id": "safe-mild-grade",
            "mode": "constant",
            "parameters": {"exposure": 0.15, "temperature": 0.10},
            "unsafe": False,
        },
        {
            "id": "unsafe-overexposure",
            "mode": "constant",
            "parameters": {"exposure": 3.0, "contrast": 0.8},
            "unsafe": True,
        },
        {
            "id": "unsafe-crushed-shadows",
            "mode": "constant",
            "parameters": {"exposure": -3.0, "contrast": 1.0},
            "unsafe": True,
        },
        {
            "id": "unsafe-frame-flicker",
            "mode": "alternating",
            "first": {"exposure": -1.5, "temperature": -0.6},
            "second": {"exposure": 1.5, "temperature": 0.6},
            "unsafe": True,
        },
        {
            "id": "unsafe-parameter-spike",
            "mode": "spike",
            "first": {"exposure": 0.1},
            "second": {"exposure": 3.0, "saturation": 1.0},
            "unsafe": True,
        },
    ]


def evaluate_safety_manifest(
    manifest_path: Path,
    *,
    max_frames: int | None = 48,
    fail_fast: bool = False,
) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve()
    payload = load_manifest(manifest_path)
    critic = PhotoAgentStyleCritic(use_vl_review=False)
    executor = RetouchExecutor()
    rows = []
    for sample_index, raw_sample in enumerate(payload["samples"]):
        sample = dict(raw_sample)
        sample_id = str(sample.get("id", f"sample-{sample_index:04d}"))
        try:
            media = load_media(
                _resolve_path(sample.get("input"), manifest_path.parent, "input"),
                fps=float(sample.get("fps", payload.get("fps", 30.0))),
                max_frames=(
                    int(sample["max_frames"])
                    if sample.get("max_frames") is not None
                    else max_frames
                ),
            )
            raw_cases = sample.get("cases", payload.get("cases", default_cases()))
            if not isinstance(raw_cases, list) or not raw_cases:
                raise ValueError("Safety manifest requires a non-empty cases list.")
            shot = ShotPlan(0, 0, len(media.frames) - 1, (len(media.frames) // 2,))
            for case_index, raw_case in enumerate(raw_cases):
                if not isinstance(raw_case, Mapping):
                    raise ValueError("Every safety case must be a JSON object.")
                case = dict(raw_case)
                case_id = str(case.get("id", f"case-{case_index:02d}"))
                parameters = _trajectory(case, len(media.frames))
                output = tuple(
                    executor.apply(frame, RetouchParameters.from_vector(values))
                    for frame, values in zip(media.frames, parameters)
                )
                anchor_index = shot.anchor_frames[0]
                anchor = AnchorGrade(
                    frame_index=anchor_index,
                    parameters=RetouchParameters.from_vector(parameters[anchor_index]),
                    preview=output[anchor_index],
                    valid=True,
                    score=1.0,
                    backend="safety-stress-injector",
                )
                critique = critic.evaluate(
                    media.frames,
                    output,
                    parameters,
                    np.zeros(len(media.frames), dtype=np.float64),
                    shot,
                    str(sample.get("instruction", "safe professional video grade")),
                    (anchor,),
                )
                expected_rollback = bool(case.get("unsafe", True))
                predicted_rollback = not critique.accepted
                rows.append(
                    {
                        "id": f"{sample_id}/{case_id}",
                        "video_id": sample_id,
                        "case_id": case_id,
                        "mode": str(case.get("mode", "constant")),
                        "expected_rollback": expected_rollback,
                        "predicted_rollback": predicted_rollback,
                        "correct": predicted_rollback == expected_rollback,
                        "critic_score": critique.score,
                        "reasons": list(critique.reasons),
                        "metrics": critique.metrics,
                    }
                )
        except Exception as error:
            if fail_fast:
                raise
            rows.append(
                {
                    "id": sample_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    valid = [row for row in rows if "error" not in row]
    unsafe = [row for row in valid if row["expected_rollback"]]
    safe = [row for row in valid if not row["expected_rollback"]]
    unsafe_recall = (
        0.0
        if not unsafe
        else float(np.mean([row["predicted_rollback"] for row in unsafe]))
    )
    safe_specificity = (
        0.0
        if not safe
        else float(np.mean([not row["predicted_rollback"] for row in safe]))
    )
    balanced = (
        (unsafe_recall + safe_specificity) / 2.0
        if unsafe and safe
        else unsafe_recall if unsafe else safe_specificity
    )
    aggregate = {
        "case_count": len(rows),
        "successful_cases": len(valid),
        "failed_cases": len(rows) - len(valid),
        "rollback_accuracy": (
            0.0 if not valid else float(np.mean([row["correct"] for row in valid]))
        ),
        "unsafe_recall": unsafe_recall,
        "safe_specificity": safe_specificity,
        "false_accept_rate": 1.0 - unsafe_recall,
        "false_rollback_rate": 1.0 - safe_specificity,
    }
    return {
        "schema_version": "videogradebench-safety/v1",
        "dataset": str(payload.get("dataset", manifest_path.stem)),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "primary_score": float(balanced),
        "aggregate": aggregate,
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    report = evaluate_safety_manifest(
        args.manifest, max_frames=args.max_frames, fail_fast=args.fail_fast
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
