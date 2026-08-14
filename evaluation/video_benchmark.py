"""Evaluate a video-to-parameter agent on paired media and parameter targets."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from retouch_agent import RetouchExecutor, RetouchParameters
from retouch_agent.parameters import (
    PARAMETER_LOWER_BOUNDS,
    PARAMETER_NAMES,
    PARAMETER_UPPER_BOUNDS,
)
from video_retouch import DynamicGradePipeline, HeuristicShotPlanner, VLShotPlanner
from video_retouch.agent_config import load_multi_agent_runtime
from video_retouch.io import decode_video, encode_video

from .scoring import agent_primary_score
from .provenance import file_sha256


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class MediaSequence:
    frames: tuple[Image.Image, ...]
    fps: float
    source: Path


def _artifact_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return name or "sample"


def _resolve_path(value: object, root: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Sample field {field!r} must be a non-empty path.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def load_media(
    path: Path,
    *,
    fps: float = 30.0,
    max_frames: Optional[int] = None,
    max_side: Optional[int] = None,
) -> MediaSequence:
    """Load either a video file or an ordered directory of image frames."""

    path = Path(path)
    if path.is_file():
        if path.suffix.lower() in IMAGE_SUFFIXES:
            raise ValueError(
                "Video benchmark inputs cannot be still images; provide a video "
                "or a directory of genuine consecutive frames."
            )
        decoded = decode_video(path, max_frames=max_frames, max_side=max_side)
        return MediaSequence(decoded.frames, decoded.fps, decoded.source)
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
    )
    if max_frames is not None:
        files = files[:max_frames]
    if not files:
        raise RuntimeError(f"Frame directory contains no supported images: {path}")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    frames = []
    for file in files:
        frame = Image.open(file).convert("RGB")
        if max_side is not None and max(frame.size) > max_side:
            scale = max_side / max(frame.size)
            frame = frame.resize(
                (
                    max(1, round(frame.width * scale)),
                    max(1, round(frame.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
        frames.append(frame)
    return MediaSequence(tuple(frames), float(fps), path.resolve())


def load_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Benchmark manifest must be a JSON object.")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Benchmark manifest requires a non-empty samples list.")
    if not all(isinstance(sample, dict) for sample in samples):
        raise ValueError("Every benchmark sample must be a JSON object.")
    return payload


def _build_pipeline(agent_config: Optional[Path], maximum_evaluations: int):
    if agent_config is None:
        pipeline = DynamicGradePipeline(
            shot_planner=HeuristicShotPlanner(),
            maximum_attempts=maximum_evaluations,
        )
        manifest = {
            "mode": "offline-native-training-free",
            "storyboard": "heuristic-shot-planner",
            "editors": ["native-anchor-agent"],
            "evaluators": ["photoagent-style-critic"],
        }
        return pipeline, manifest

    runtime = load_multi_agent_runtime(agent_config)
    pipeline = DynamicGradePipeline(
        shot_planner=VLShotPlanner(
            client=runtime.storyboard_client,
            settings=runtime.storyboard_settings,
            strict=True,
        ),
        anchor_backends=runtime.anchor_backends,
        critic=runtime.critic,
        maximum_attempts=maximum_evaluations,
        maximum_hero_attempts=runtime.search.maximum_hero_attempts,
        mcts_exploration=runtime.search.exploration_constant,
        mcts_rejection_penalty=runtime.search.rejection_penalty,
        mcts_seed=runtime.search.seed,
    )
    return pipeline, runtime.manifest


def _rgb(image: Image.Image, size: Optional[tuple[int, int]] = None) -> np.ndarray:
    image = image.convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def _ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Standard Gaussian-window SSIM averaged over RGB channels."""

    c1, c2 = 0.01**2, 0.03**2
    scores = []
    for channel in range(3):
        ref = reference[..., channel]
        out = candidate[..., channel]
        mu_ref = cv2.GaussianBlur(ref, (11, 11), 1.5)
        mu_out = cv2.GaussianBlur(out, (11, 11), 1.5)
        sigma_ref = cv2.GaussianBlur(ref * ref, (11, 11), 1.5) - mu_ref**2
        sigma_out = cv2.GaussianBlur(out * out, (11, 11), 1.5) - mu_out**2
        covariance = cv2.GaussianBlur(ref * out, (11, 11), 1.5) - mu_ref * mu_out
        numerator = (2.0 * mu_ref * mu_out + c1) * (2.0 * covariance + c2)
        denominator = (mu_ref**2 + mu_out**2 + c1) * (
            sigma_ref + sigma_out + c2
        )
        scores.append(float(np.mean(numerator / np.maximum(denominator, 1e-12))))
    return float(np.mean(scores))


def _motion_compensated_residual(
    base: Sequence[np.ndarray], candidate: Sequence[np.ndarray]
) -> float:
    """Measure temporal changes in an edit/error field after optical-flow warp."""

    if len(base) <= 1:
        return 0.0
    residuals = [out - source for source, out in zip(base, candidate)]
    errors = []
    for index in range(1, len(base)):
        previous_gray = cv2.cvtColor(
            np.clip(base[index - 1] * 255.0, 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
        current_gray = cv2.cvtColor(
            np.clip(base[index] * 255.0, 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
        backward = cv2.calcOpticalFlowFarneback(
            current_gray,
            previous_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        height, width = current_gray.shape
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        warped_previous = cv2.remap(
            residuals[index - 1],
            grid_x + backward[..., 0],
            grid_y + backward[..., 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        errors.append(float(np.mean(np.abs(residuals[index] - warped_previous))))
    return float(np.mean(errors))


def _trajectory_metrics(parameters: np.ndarray) -> dict[str, float]:
    normalized = np.asarray(parameters, dtype=np.float64) / (
        PARAMETER_UPPER_BOUNDS - PARAMETER_LOWER_BOUNDS
    )[None, :]
    velocity = np.diff(normalized, axis=0)
    acceleration = np.diff(velocity, axis=0)
    return {
        "temporal_parameter_velocity": (
            0.0 if velocity.size == 0 else float(np.mean(np.abs(velocity)))
        ),
        "temporal_parameter_jerk": (
            0.0 if acceleration.size == 0 else float(np.mean(np.abs(acceleration)))
        ),
    }


def _anchor_metrics(
    frames: Sequence[Image.Image], storyboard
) -> dict[str, float]:
    from video_retouch.propagation import appearance_features

    coverage_values = []
    worst_values = []
    anchor_count = 0
    feature_scale = np.asarray([0.20, 0.12, 0.20, 0.12], dtype=np.float64)
    for shot in storyboard.shots:
        shot_frames = frames[shot.start_frame : shot.end_frame + 1]
        features = appearance_features(shot_frames)
        local_anchors = np.asarray(
            [index - shot.start_frame for index in shot.anchor_frames],
            dtype=np.int64,
        )
        if local_anchors.size == 0:
            continue
        distances = np.sqrt(
            np.mean(
                (
                    (features[:, None, :] - features[local_anchors][None, :, :])
                    / feature_scale[None, None, :]
                )
                ** 2,
                axis=2,
            )
        )
        nearest = np.min(distances, axis=1)
        coverage_values.extend(nearest.tolist())
        worst_values.append(float(np.max(nearest)))
        anchor_count += int(local_anchors.size)
    frame_count = max(1, len(frames))
    return {
        "anchor_mean_coverage_error": (
            0.0 if not coverage_values else float(np.mean(coverage_values))
        ),
        "anchor_worst_coverage_error": (
            0.0 if not worst_values else float(np.max(worst_values))
        ),
        "anchors_per_100_frames": 100.0 * anchor_count / frame_count,
    }


def _boundary_metrics(
    predicted: Sequence[int], target: Sequence[int], tolerance: int
) -> dict[str, float]:
    if tolerance < 0:
        raise ValueError("shot_boundary_tolerance must be non-negative.")
    unmatched = set(int(value) for value in target)
    matches = 0
    for boundary in sorted(int(value) for value in predicted):
        choices = [value for value in unmatched if abs(value - boundary) <= tolerance]
        if choices:
            selected = min(choices, key=lambda value: abs(value - boundary))
            unmatched.remove(selected)
            matches += 1
    precision = matches / len(predicted) if predicted else float(not target)
    recall = matches / len(target) if target else float(not predicted)
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
    return {
        "shot_boundary_precision": float(precision),
        "shot_boundary_recall": float(recall),
        "shot_boundary_f1": float(f1),
    }


def _reference_metrics(
    source: Sequence[Image.Image],
    output: Sequence[Image.Image],
    reference: Sequence[Image.Image],
) -> dict[str, float]:
    count = min(len(source), len(output), len(reference))
    if count < 1:
        raise ValueError("Reference evaluation requires at least one aligned frame.")
    output_arrays = []
    reference_arrays = []
    output_mae = []
    baseline_mae = []
    psnr = []
    ssim = []
    for index in range(count):
        size = source[index].size
        src = _rgb(source[index], size)
        out = _rgb(output[index], size)
        ref = _rgb(reference[index], size)
        output_arrays.append(out)
        reference_arrays.append(ref)
        candidate_error = float(np.mean(np.abs(out - ref)))
        output_mae.append(candidate_error)
        baseline_mae.append(float(np.mean(np.abs(src - ref))))
        mse = float(np.mean((out - ref) ** 2))
        psnr.append(100.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse))
        ssim.append(_ssim(ref, out))

    mean_output_mae = float(np.mean(output_mae))
    mean_baseline_mae = float(np.mean(baseline_mae))
    temporal_residual = _motion_compensated_residual(
        reference_arrays, output_arrays
    )
    relative_improvement = (
        0.0
        if mean_baseline_mae <= 1e-12
        else (mean_baseline_mae - mean_output_mae) / mean_baseline_mae
    )
    return {
        "reference_frames": float(count),
        "reference_mae": mean_output_mae,
        "identity_reference_mae": mean_baseline_mae,
        "reference_mae_improvement": mean_baseline_mae - mean_output_mae,
        "relative_reference_improvement": float(relative_improvement),
        "reference_psnr": float(np.mean(psnr)),
        "reference_ssim": float(np.mean(ssim)),
        "motion_compensated_temporal_reference_residual": temporal_residual,
    }


def _target_vector(value: object) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, dict):
        return RetouchParameters.from_mapping(value, clamp=False).to_vector()
    if isinstance(value, list):
        return RetouchParameters.from_vector(value, clamp=False).to_vector()
    raise ValueError("target_parameters must be a parameter object or 12-D list.")


def _parameter_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    estimate = np.mean(predicted, axis=0)
    scale = PARAMETER_UPPER_BOUNDS - PARAMETER_LOWER_BOUNDS
    active = np.abs(target) > 1e-8
    sign_accuracy = (
        1.0
        if not np.any(active)
        else float(np.mean(np.sign(estimate[active]) == np.sign(target[active])))
    )
    return {
        "parameter_mae": float(np.mean(np.abs(estimate - target))),
        "normalized_parameter_mae": float(
            np.mean(np.abs(estimate - target) / scale)
        ),
        "active_parameter_sign_accuracy": sign_accuracy,
    }


def _aggregate(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    metric_names = sorted(
        {
            name
            for row in rows
            for name, value in dict(row.get("metrics", {})).items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
    )
    metrics = {}
    for name in metric_names:
        values = [
            float(row["metrics"][name])
            for row in rows
            if name in row.get("metrics", {})
            and math.isfinite(float(row["metrics"][name]))
        ]
        if values:
            metrics[name] = float(np.mean(values))
    successful = [row for row in rows if "error" not in row]
    return {
        "sample_count": len(rows),
        "successful_samples": len(successful),
        "failed_samples": len(rows) - len(successful),
        "rolled_back_samples": sum(bool(row.get("rolled_back")) for row in successful),
        "metrics": metrics,
    }


def _slice_metrics(
    rows: Sequence[dict[str, object]], fields: Sequence[str]
) -> dict[str, object]:
    output = {}
    for field in fields:
        values = sorted(
            {
                str(row[field])
                for row in rows
                if field in row and "error" not in row
            }
        )
        if values:
            output[field] = {
                value: _aggregate(
                    [row for row in rows if str(row.get(field)) == value]
                )
                for value in values
            }
    return output


def evaluate_manifest(
    manifest_path: Path,
    *,
    agent_config: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    video_output_dir: Optional[Path] = None,
    maximum_evaluations: int = 3,
    max_frames: Optional[int] = None,
    fail_fast: bool = False,
) -> dict[str, object]:
    """Run the agent and return per-sample plus aggregate benchmark metrics."""

    if maximum_evaluations < 1:
        raise ValueError("maximum_evaluations must be positive.")
    manifest_path = Path(manifest_path).resolve()
    payload = load_manifest(manifest_path)
    root = manifest_path.parent
    output_dir = None if output_dir is None else Path(output_dir).resolve()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    video_output_dir = (
        None if video_output_dir is None else Path(video_output_dir).resolve()
    )
    if video_output_dir is not None:
        video_output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    runtime_manifest: Optional[dict[str, object]] = None
    for index, raw_sample in enumerate(payload["samples"]):
        sample = dict(raw_sample)
        sample_id = str(sample.get("id", f"sample-{index:04d}"))
        try:
            sample_fps = float(sample.get("fps", payload.get("fps", 30.0)))
            sample_max_frames = sample.get("max_frames", max_frames)
            if sample_max_frames is not None:
                sample_max_frames = int(sample_max_frames)
            source = load_media(
                _resolve_path(sample.get("input"), root, "input"),
                fps=sample_fps,
                max_frames=sample_max_frames,
            )
            pipeline, runtime_manifest = _build_pipeline(
                agent_config, maximum_evaluations
            )
            instruction = str(sample.get("instruction", "natural balanced grade"))
            grade = pipeline.run(source.frames, source.fps, instruction)
            executor = RetouchExecutor()
            rendered = tuple(
                executor.apply(frame, RetouchParameters.from_vector(parameters))
                for frame, parameters in zip(source.frames, grade.frame_parameters)
            )
            rolled_back = all(shot.rolled_back for shot in grade.shots)
            metrics: dict[str, float] = {
                "shot_count": float(len(grade.shots)),
                "accepted_shot_rate": float(
                    np.mean([shot.accepted for shot in grade.shots])
                ),
            }
            metrics.update(_trajectory_metrics(grade.frame_parameters))
            metrics.update(_anchor_metrics(source.frames, grade.storyboard))
            source_arrays = [_rgb(frame) for frame in source.frames]
            output_arrays = [_rgb(frame) for frame in rendered]
            metrics["motion_compensated_edit_residual"] = (
                _motion_compensated_residual(source_arrays, output_arrays)
            )
            if "shot_boundaries" in sample:
                target_boundaries = sample["shot_boundaries"]
                if not isinstance(target_boundaries, list):
                    raise ValueError("shot_boundaries must be a list of frame indices.")
                predicted_boundaries = [
                    shot.start_frame for shot in grade.storyboard.shots[1:]
                ]
                metrics.update(
                    _boundary_metrics(
                        predicted_boundaries,
                        [int(value) for value in target_boundaries],
                        int(sample.get("shot_boundary_tolerance", 1)),
                    )
                )

            target = _target_vector(sample.get("target_parameters"))
            if sample.get("reference") is not None:
                reference = load_media(
                    _resolve_path(sample.get("reference"), root, "reference"),
                    fps=sample_fps,
                    max_frames=sample_max_frames,
                )
                metrics.update(
                    _reference_metrics(source.frames, rendered, reference.frames)
                )
            elif target is not None:
                target_parameters = RetouchParameters.from_vector(target)
                reference_frames = tuple(
                    executor.apply(frame, target_parameters) for frame in source.frames
                )
                metrics.update(
                    _reference_metrics(source.frames, rendered, reference_frames)
                )
            if target is not None:
                metrics.update(_parameter_metrics(grade.frame_parameters, target))
            if "expect_rollback" in sample:
                metrics["rollback_accuracy"] = float(
                    rolled_back == bool(sample["expect_rollback"])
                )

            row: dict[str, object] = {
                "id": sample_id,
                "input": str(source.source),
                "instruction": instruction,
                "frame_count": len(source.frames),
                "shot_count": len(grade.shots),
                "rolled_back": rolled_back,
                "metrics": metrics,
            }
            for slice_field in ("language", "intent", "subset"):
                if slice_field in sample:
                    row[slice_field] = sample[slice_field]
            if output_dir is not None:
                artifact_name = _artifact_name(sample_id)
                grade_path = output_dir / f"{artifact_name}.grade.json"
                grade_path.write_text(
                    json.dumps(
                        grade.to_dict(include_frame_parameters=True),
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                row["grade_output"] = str(grade_path)
            if video_output_dir is not None:
                artifact_name = _artifact_name(sample_id)
                input_video_path = video_output_dir / f"{artifact_name}.input.mp4"
                result_video_path = video_output_dir / f"{artifact_name}.result.mp4"
                encode_video(source.frames, input_video_path, source.fps)
                encode_video(rendered, result_video_path, source.fps)
                row["input_video_output"] = str(input_video_path)
                row["result_video_output"] = str(result_video_path)
            rows.append(row)
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
    profile = str(payload.get("profile", "generic"))
    primary_score = agent_primary_score(profile, aggregate["metrics"])
    return {
        "schema_version": "training-free-video-benchmark/v1",
        "dataset": str(payload.get("dataset", manifest_path.stem)),
        "profile": profile,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "runtime": runtime_manifest,
        "primary_score": primary_score,
        "aggregate": aggregate,
        "slices": _slice_metrics(rows, ("language", "intent", "subset")),
        "samples": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grade-output-dir", type=Path)
    parser.add_argument(
        "--video-output-dir",
        type=Path,
        help="Export each processed input and final graded result as silent MP4.",
    )
    parser.add_argument("--maximum-evaluations", type=int, default=3)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    report = evaluate_manifest(
        args.manifest,
        agent_config=args.agent_config,
        output_dir=args.grade_output_dir,
        video_output_dir=args.video_output_dir,
        maximum_evaluations=args.maximum_evaluations,
        max_frames=args.max_frames,
        fail_fast=args.fail_fast,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
