"""No-GT benchmark for reference-video controlled color grading.

The task has no pixel-aligned ground truth.  Consequently this script does not
turn input/output edit magnitude or cross-content color histograms into a
quality score.  It measures content preservation, temporal stability and
technical artifacts, and emits a blinded review sheet for the only defensible
style-match ranking: human or independently configured vision-model review.

Lightweight classical baselines make the protocol runnable without model
checkpoints.  Any black-box method, including the ShotAgent API pool, can be
added with ``--external METHOD=DIR``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.linalg import sqrtm
from scipy.stats import rankdata
from skimage.metrics import structural_similarity

from video_retouch.io import decode_video, encode_video


@dataclass(frozen=True)
class VideoData:
    frames: tuple[Image.Image, ...]
    fps: float
    path: Path


def _load(path: Path, max_frames: int | None, max_side: int | None) -> VideoData:
    decoded = decode_video(path, max_frames=max_frames, max_side=max_side)
    return VideoData(decoded.frames, decoded.fps, decoded.source)


def _lab(image: Image.Image, size: tuple[int, int] | None = None) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB").resize(size or image.size), dtype=np.float32)
    return cv2.cvtColor(rgb / 255.0, cv2.COLOR_RGB2LAB).astype(np.float64)


def _rgb(lab: np.ndarray) -> Image.Image:
    values = cv2.cvtColor(np.asarray(lab, dtype=np.float32), cv2.COLOR_LAB2RGB)
    return Image.fromarray((np.clip(values, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))


def _sample_indices(length: int, count: int = 24) -> np.ndarray:
    return np.unique(np.linspace(0, length - 1, min(length, count)).round().astype(int))


def _pixels(frames: Sequence[Image.Image], sample_count: int = 24) -> np.ndarray:
    rows = []
    for index in _sample_indices(len(frames), sample_count):
        image = frames[int(index)].convert("RGB")
        scale = min(1.0, 96.0 / max(image.size))
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        rows.append(_lab(image, size).reshape(-1, 3))
    values = np.concatenate(rows, axis=0)
    if len(values) > 120_000:
        values = values[np.linspace(0, len(values) - 1, 120_000).astype(int)]
    return values


def _matched_lab_samples(
    left: Sequence[Image.Image],
    right: Sequence[Image.Image],
    maximum: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic, equally sized Lab samples in normalized units."""

    scale = np.asarray([100.0, 128.0, 128.0], dtype=np.float64)
    left_pixels = _pixels(left) / scale
    right_pixels = _pixels(right) / scale
    count = min(len(left_pixels), len(right_pixels), maximum)
    left_indices = np.linspace(0, len(left_pixels) - 1, count).astype(np.int64)
    right_indices = np.linspace(0, len(right_pixels) - 1, count).astype(np.int64)
    return left_pixels[left_indices], right_pixels[right_indices]


def lab_wasserstein_distance(
    output: Sequence[Image.Image], reference: Sequence[Image.Image]
) -> float:
    """Mean 1-D Wasserstein distance over normalized L, a, and b channels."""

    output_pixels, reference_pixels = _matched_lab_samples(output, reference)
    channel_distances = []
    for channel in range(3):
        left = np.sort(output_pixels[:, channel])
        right = np.sort(reference_pixels[:, channel])
        channel_distances.append(float(np.mean(np.abs(left - right))))
    return float(np.mean(channel_distances))


def lab_sliced_wasserstein_distance(
    output: Sequence[Image.Image],
    reference: Sequence[Image.Image],
    projection_count: int = 64,
    seed: int = 7,
) -> float:
    """Sliced Wasserstein distance over the joint normalized Lab distribution."""

    output_pixels, reference_pixels = _matched_lab_samples(output, reference)
    generator = np.random.default_rng(seed)
    directions = generator.normal(size=(3, projection_count))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True).clip(1e-12)
    output_projections = np.sort(output_pixels @ directions, axis=0)
    reference_projections = np.sort(reference_pixels @ directions, axis=0)
    return float(np.mean(np.abs(output_projections - reference_projections)))


def _reinhard_fit(
    source: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    source_std = np.maximum(source.std(axis=0), 1e-3)
    reference_mean = reference.mean(axis=0)
    reference_std = np.maximum(reference.std(axis=0), 1e-3)
    scale = np.clip(reference_std / source_std, 0.35, 3.0)
    return source_mean, reference_mean, scale


def _apply_reinhard(
    image: Image.Image, fit: tuple[np.ndarray, np.ndarray, np.ndarray], strength: float
) -> Image.Image:
    source_mean, reference_mean, scale = fit
    lab = _lab(image)
    mapped = (lab - source_mean) * scale + reference_mean
    return _rgb((1.0 - strength) * lab + strength * mapped)


def _mkl_fit(
    source: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    reference_mean = reference.mean(axis=0)
    cs = np.cov(source, rowvar=False) + np.eye(3) * 1e-3
    cr = np.cov(reference, rowvar=False) + np.eye(3) * 1e-3
    cs_half = np.real(sqrtm(cs))
    cs_inv_half = np.linalg.inv(cs_half)
    transform = cs_inv_half @ np.real(sqrtm(cs_half @ cr @ cs_half)) @ cs_inv_half
    return source_mean, reference_mean, transform


def _apply_mkl(
    image: Image.Image, fit: tuple[np.ndarray, np.ndarray, np.ndarray], strength: float
) -> Image.Image:
    source_mean, reference_mean, transform = fit
    lab = _lab(image)
    mapped = (lab - source_mean) @ transform.T + reference_mean
    return _rgb((1.0 - strength) * lab + strength * mapped)


def identity(
    target: VideoData, reference: VideoData, strength: float
) -> tuple[Image.Image, ...]:
    del reference, strength
    return tuple(frame.copy() for frame in target.frames)


def global_reinhard(
    target: VideoData, reference: VideoData, strength: float
) -> tuple[Image.Image, ...]:
    fit = _reinhard_fit(_pixels(target.frames), _pixels(reference.frames))
    return tuple(_apply_reinhard(frame, fit, strength) for frame in target.frames)


def global_mkl(
    target: VideoData, reference: VideoData, strength: float
) -> tuple[Image.Image, ...]:
    fit = _mkl_fit(_pixels(target.frames), _pixels(reference.frames))
    return tuple(_apply_mkl(frame, fit, strength) for frame in target.frames)


def framewise_reinhard(
    target: VideoData, reference: VideoData, strength: float
) -> tuple[Image.Image, ...]:
    output = []
    for index, frame in enumerate(target.frames):
        reference_index = round(
            index * (len(reference.frames) - 1) / max(1, len(target.frames) - 1)
        )
        fit = _reinhard_fit(
            _pixels((frame,), 1), _pixels((reference.frames[reference_index],), 1)
        )
        output.append(_apply_reinhard(frame, fit, strength))
    return tuple(output)


BASELINES: dict[
    str, Callable[[VideoData, VideoData, float], tuple[Image.Image, ...]]
] = {
    "identity": identity,
    "global-reinhard": global_reinhard,
    "global-mkl": global_mkl,
    "framewise-reinhard": framewise_reinhard,
}
METHOD_NAMES = tuple(BASELINES)

METRIC_DIRECTIONS = {
    "vgg_style_similarity": "higher",
    "llm_reference_style_similarity": "higher",
    "lab_wasserstein_distance": "lower",
    "lab_sliced_wasserstein_distance": "lower",
    "content_structure_correlation": "higher",
    "edge_ssim": "higher",
    "dino_content_similarity": "higher",
    "temporal_flow_warp_error": "lower",
    "temporal_edit_warp_error": "lower",
    "temporal_transform_drift": "lower",
    "musiq_score": "higher",
    "new_shadow_clip_fraction": "lower",
    "new_highlight_clip_fraction": "lower",
}


def _align_frames(
    frames: Sequence[Image.Image], target: VideoData
) -> tuple[Image.Image, ...]:
    """Time-align any black-box API output to the target video contract."""

    if not frames:
        raise ValueError("Method output contains no frames.")
    aligned = []
    for index in range(len(target.frames)):
        source_index = round(index * (len(frames) - 1) / max(1, len(target.frames) - 1))
        aligned.append(
            frames[source_index].convert("RGB").resize(target.frames[index].size)
        )
    return tuple(aligned)


def _local_structure(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Remove local luminance/contrast so legitimate grading is discounted."""

    gray = cv2.cvtColor(
        np.asarray(image.convert("RGB").resize(size), dtype=np.float32) / 255.0,
        cv2.COLOR_RGB2GRAY,
    )
    mean = cv2.GaussianBlur(gray, (0, 0), 3.0)
    variance = np.maximum(
        cv2.GaussianBlur(gray * gray, (0, 0), 3.0) - mean * mean, 1e-4
    )
    return (gray - mean) / np.sqrt(variance)


def _structure_correlation(
    left: Image.Image, right: Image.Image, size: tuple[int, int]
) -> float:
    """Correlation of locally normalised luminance structure."""

    a = _local_structure(left, size).reshape(-1)
    b = _local_structure(right, size).reshape(-1)
    if float(a.std()) < 1e-8 and float(b.std()) < 1e-8:
        return 1.0
    if float(a.std()) < 1e-8 or float(b.std()) < 1e-8:
        return 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1.0, 1.0))


def _edge_ssim(left: Image.Image, right: Image.Image, size: tuple[int, int]) -> float:
    """SSIM on Canny edge maps, discounting ordinary color and tone edits."""

    def edges(image: Image.Image) -> np.ndarray:
        gray = cv2.cvtColor(
            np.asarray(image.convert("RGB").resize(size), dtype=np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
        return cv2.Canny(gray, 80, 160).astype(np.float32) / 255.0

    return float(structural_similarity(edges(left), edges(right), data_range=1.0))


def _valid_transitions(source: Sequence[Image.Image]) -> np.ndarray:
    """Return transitions that are unlikely to be hard cuts."""

    if len(source) < 2:
        return np.zeros(0, dtype=bool)
    scores = []
    for previous, current in pairwise(source):
        size = (min(160, current.width), min(90, current.height))
        a = np.asarray(previous.convert("RGB").resize(size), dtype=np.float32) / 255.0
        b = np.asarray(current.convert("RGB").resize(size), dtype=np.float32) / 255.0
        scores.append(float(np.mean(np.abs(a - b))))
    values = np.asarray(scores)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(0.18, median + 6.0 * max(mad, 1e-4))
    return values <= threshold


def _motion_compensated_edit_residual(
    source: Sequence[Image.Image], output: Sequence[Image.Image]
) -> float:
    if len(source) < 2:
        return 0.0
    errors = []
    valid = _valid_transitions(source)
    for index in range(1, len(source)):
        if not valid[index - 1]:
            continue
        size = (min(256, source[index].width), min(144, source[index].height))
        previous = np.asarray(source[index - 1].resize(size), dtype=np.float32) / 255.0
        current = np.asarray(source[index].resize(size), dtype=np.float32) / 255.0
        previous_out = (
            np.asarray(output[index - 1].resize(size), dtype=np.float32) / 255.0
        )
        current_out = np.asarray(output[index].resize(size), dtype=np.float32) / 255.0
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        grid_x, grid_y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
        map_x = (grid_x - flow[..., 0]).astype(np.float32)
        map_y = (grid_y - flow[..., 1]).astype(np.float32)
        edit_previous = previous_out - previous
        edit_current = current_out - current
        warped = cv2.remap(
            edit_previous, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        errors.append(float(np.mean(np.abs(edit_current - warped))))
    return float(np.mean(errors)) if errors else 0.0


def _motion_compensated_output_residual(
    source: Sequence[Image.Image], output: Sequence[Image.Image]
) -> float:
    """Warp output frames with source-video flow and measure temporal residual."""

    if len(source) < 2:
        return 0.0
    errors = []
    valid = _valid_transitions(source)
    for index in range(1, len(source)):
        if not valid[index - 1]:
            continue
        size = (min(256, source[index].width), min(144, source[index].height))
        previous = np.asarray(source[index - 1].resize(size), dtype=np.float32) / 255.0
        current = np.asarray(source[index].resize(size), dtype=np.float32) / 255.0
        previous_out = (
            np.asarray(output[index - 1].resize(size), dtype=np.float32) / 255.0
        )
        current_out = np.asarray(output[index].resize(size), dtype=np.float32) / 255.0
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        grid_x, grid_y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
        map_x = (grid_x - flow[..., 0]).astype(np.float32)
        map_y = (grid_y - flow[..., 1]).astype(np.float32)
        warped = cv2.remap(
            previous_out, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        errors.append(float(np.mean(np.abs(current_out - warped))))
    return float(np.mean(errors)) if errors else 0.0


def _frame_edit_signature(before: Image.Image, after: Image.Image) -> np.ndarray:
    scale = min(1.0, 96.0 / max(before.size))
    size = (max(1, round(before.width * scale)), max(1, round(before.height * scale)))
    scale_values = np.array([100.0, 128.0, 128.0])
    source = _lab(before, size).reshape(-1, 3) / scale_values
    result = _lab(after, size).reshape(-1, 3) / scale_values
    design = np.column_stack([source, np.ones(len(source))])
    # A compact approximation of the frame's applied color transform. Fixed
    # global grades remain stable even as the scene's pixel distribution moves.
    coefficients = np.linalg.lstsq(design, result - source, rcond=None)[0]
    return coefficients.reshape(-1)


def _temporal_transform_drift(
    source: Sequence[Image.Image], output: Sequence[Image.Image]
) -> float:
    if len(source) < 2:
        return 0.0
    signatures = np.asarray(
        [_frame_edit_signature(a, b) for a, b in zip(source, output)]
    )
    differences = np.linalg.norm(np.diff(signatures, axis=0), axis=1)
    valid = _valid_transitions(source)
    return float(np.mean(differences[valid])) if valid.any() else 0.0


def metrics(
    target: VideoData,
    output: Sequence[Image.Image],
    reference: VideoData | None = None,
    learned_suite=None,
) -> dict[str, float]:
    shadow_clipping, source_shadow_clipping = [], []
    highlight_clipping, source_highlight_clipping = [], []
    structure_scores = []
    edge_ssim_scores = []
    for source, result in zip(target.frames, output):
        result_l = _lab(result)[..., 0]
        source_l = _lab(source)[..., 0]
        shadow_clipping.append(float(np.mean(result_l <= 1.0)))
        source_shadow_clipping.append(float(np.mean(source_l <= 1.0)))
        highlight_clipping.append(float(np.mean(result_l >= 99.0)))
        source_highlight_clipping.append(float(np.mean(source_l >= 99.0)))
        size = (min(256, source.width), min(144, source.height))
        structure_scores.append(_structure_correlation(source, result, size))
        edge_ssim_scores.append(_edge_ssim(source, result, size))
    shadow_clip = float(np.mean(shadow_clipping))
    source_shadow_clip = float(np.mean(source_shadow_clipping))
    highlight_clip = float(np.mean(highlight_clipping))
    source_highlight_clip = float(np.mean(source_highlight_clipping))
    values: dict[str, float] = {
        "content_structure_correlation": float(np.mean(structure_scores)),
        "edge_ssim": float(np.mean(edge_ssim_scores)),
        "temporal_flow_warp_error": _motion_compensated_output_residual(
            target.frames, output
        ),
        "temporal_edit_warp_error": _motion_compensated_edit_residual(
            target.frames, output
        ),
        "temporal_transform_drift": _temporal_transform_drift(target.frames, output),
        "new_shadow_clip_fraction": max(0.0, shadow_clip - source_shadow_clip),
        "new_highlight_clip_fraction": max(0.0, highlight_clip - source_highlight_clip),
    }
    if reference is not None:
        values["lab_wasserstein_distance"] = lab_wasserstein_distance(
            output, reference.frames
        )
        values["lab_sliced_wasserstein_distance"] = lab_sliced_wasserstein_distance(
            output, reference.frames
        )
    if learned_suite is not None:
        if reference is None:
            raise ValueError(
                "Learned reference-style metrics require a reference video."
            )
        values.update(learned_suite.evaluate(target.frames, reference.frames, output))
    return values


def _write_blind_review(
    output_dir: Path,
    samples: Sequence[dict[str, object]],
    methods: Sequence[str],
) -> None:
    """Create anonymised media plus individual and pairwise review sheets."""

    media_root = output_dir / "blind_review_media"
    assignments: list[dict[str, str]] = []
    individual_rows: list[dict[str, str]] = []
    pairwise_rows: list[dict[str, str]] = []
    for sample in samples:
        sample_id = str(sample["id"])
        sample_dir = output_dir / sample_id
        blind_dir = media_root / sample_id
        blind_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample_dir / "target.mp4", blind_dir / "target.mp4")
        shutil.copy2(sample_dir / "reference.mp4", blind_dir / "reference.mp4")
        ordered = sorted(
            methods,
            key=lambda method: hashlib.sha256(
                f"{sample_id}:{method}".encode()
            ).hexdigest(),
        )
        codes: list[str] = []
        for index, method in enumerate(ordered, start=1):
            code = f"C{index:02d}"
            codes.append(code)
            destination = blind_dir / f"{code}.mp4"
            shutil.copy2(sample_dir / f"{method}.mp4", destination)
            relative = destination.relative_to(output_dir).as_posix()
            assignments.append(
                {"sample": sample_id, "candidate_code": code, "method": method}
            )
            individual_rows.append(
                {
                    "sample": sample_id,
                    "candidate_code": code,
                    "target_video": (blind_dir / "target.mp4")
                    .relative_to(output_dir)
                    .as_posix(),
                    "reference_video": (blind_dir / "reference.mp4")
                    .relative_to(output_dir)
                    .as_posix(),
                    "candidate_video": relative,
                    "reference_style_match_1_5": "",
                    "content_preservation_1_5": "",
                    "temporal_consistency_1_5": "",
                    "artifact_free_1_5": "",
                    "overall_rank": "",
                    "notes": "",
                }
            )
        pair_index = 0
        for left_index, left in enumerate(codes):
            for right in codes[left_index + 1 :]:
                pair_index += 1
                pairwise_rows.append(
                    {
                        "sample": sample_id,
                        "pair_id": f"P{pair_index:02d}",
                        "target_video": (blind_dir / "target.mp4")
                        .relative_to(output_dir)
                        .as_posix(),
                        "reference_video": (blind_dir / "reference.mp4")
                        .relative_to(output_dir)
                        .as_posix(),
                        "candidate_a": left,
                        "candidate_a_video": (blind_dir / f"{left}.mp4")
                        .relative_to(output_dir)
                        .as_posix(),
                        "candidate_b": right,
                        "candidate_b_video": (blind_dir / f"{right}.mp4")
                        .relative_to(output_dir)
                        .as_posix(),
                        "style_match_winner_A_B_Tie": "",
                        "overall_winner_A_B_Tie": "",
                        "notes": "",
                    }
                )

    def write_csv(name: str, rows: list[dict[str, str]]) -> None:
        if not rows:
            return
        with (output_dir / name).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv("blind_individual_review.csv", individual_rows)
    write_csv("blind_pairwise_review.csv", pairwise_rows)
    (output_dir / "blind_review_key.json").write_text(
        json.dumps(assignments, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _mosaic(
    target: VideoData,
    reference: VideoData,
    outputs: dict[str, Sequence[Image.Image]],
    path: Path,
) -> None:
    labels = ["TARGET", "REFERENCE"]
    sources: list[Sequence[Image.Image]] = [target.frames, reference.frames]
    labels.extend(name.upper() for name in outputs)
    sources.extend(outputs.values())
    width, height = 320, 180
    frames = []
    for index in range(len(target.frames)):
        canvas = Image.new("RGB", (width * len(sources), height + 34), "#111820")
        draw = ImageDraw.Draw(canvas)
        for column, (label, sequence) in enumerate(zip(labels, sources)):
            source_index = round(
                index * (len(sequence) - 1) / max(1, len(target.frames) - 1)
            )
            image = sequence[source_index].convert("RGB")
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            x = column * width + (width - image.width) // 2
            y = 34 + (height - image.height) // 2
            canvas.paste(image, (x, y))
            draw.text((column * width + 10, 9), label, fill="white")
        frames.append(canvas)
    encode_video(frames, path, target.fps, preset="veryfast")


def run_manifest(
    manifest_path: Path,
    output_dir: Path,
    methods: Sequence[str],
    strength: float,
    external: dict[str, Path] | None = None,
    learned_metrics: bool = False,
    learned_frame_count: int = 8,
    style_vgg_weights: Path | None = None,
) -> dict[str, object]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    rows = []
    learned_suite = None
    if learned_metrics:
        from evaluation.perceptual_metrics import LearnedMetricSuite

        learned_suite = LearnedMetricSuite(
            frame_count=learned_frame_count,
            style_vgg_weights=style_vgg_weights,
        )
    for sample in payload["samples"]:
        sample_id = str(sample["id"])
        target = _load(
            (root / sample["target"]).resolve(),
            sample.get("max_frames"),
            sample.get("max_side"),
        )
        reference = _load(
            (root / sample["reference"]).resolve(),
            sample.get("max_frames"),
            sample.get("max_side"),
        )
        sample_outputs = {}
        requested = list(methods) + list((external or {}).keys())
        method_dir = output_dir / sample_id
        method_dir.mkdir(parents=True, exist_ok=True)
        encode_video(
            target.frames, method_dir / "target.mp4", target.fps, preset="veryfast"
        )
        encode_video(
            reference.frames,
            method_dir / "reference.mp4",
            reference.fps,
            preset="veryfast",
        )
        for method in requested:
            start = time.perf_counter()
            measure_runtime = method not in (external or {})
            if method in (external or {}):
                method_root = (external or {})[method]
                candidates = (
                    method_root / f"{sample_id}.mp4",
                    method_root / sample_id / "output.mp4",
                    method_root / sample_id / f"{method}.mp4",
                )
                method_path = next(
                    (path for path in candidates if path.is_file()), None
                )
                if method_path is None:
                    raise FileNotFoundError(
                        f"No output for external method {method!r}, sample "
                        f"{sample_id!r}."
                    )
                decoded_output = _load(
                    method_path,
                    sample.get("max_frames"),
                    sample.get("max_side"),
                )
                result = _align_frames(decoded_output.frames, target)
            else:
                result = BASELINES[method](target, reference, strength)
            elapsed = time.perf_counter() - start
            encode_video(
                result, method_dir / f"{method}.mp4", target.fps, preset="veryfast"
            )
            result_metrics = metrics(
                target,
                result,
                reference=reference,
                learned_suite=learned_suite,
            )
            if measure_runtime:
                result_metrics["runtime_seconds"] = elapsed
                result_metrics["processing_fps"] = len(result) / max(elapsed, 1e-8)
            rows.append({"sample": sample_id, "method": method, **result_metrics})
            sample_outputs[method] = result
        _mosaic(
            target, reference, sample_outputs, output_dir / sample_id / "comparison.mp4"
        )
    aggregate = {}
    aggregate_statistics = {}
    for method in list(methods) + list((external or {}).keys()):
        method_rows = [row for row in rows if row["method"] == method]
        keys = [
            key
            for key in method_rows[0]
            if any(isinstance(row.get(key), (int, float)) for row in method_rows)
        ]
        aggregate[method] = {}
        aggregate_statistics[method] = {}
        for key in keys:
            samples = np.asarray(
                [
                    row[key]
                    for row in method_rows
                    if isinstance(row.get(key), (int, float))
                ],
                dtype=np.float64,
            )
            mean = float(np.mean(samples))
            std = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
            radius = 1.96 * std / np.sqrt(len(samples))
            aggregate[method][key] = mean
            aggregate_statistics[method][key] = {
                "mean": mean,
                "std": std,
                "ci95_low": mean - radius,
                "ci95_high": mean + radius,
                "n": len(samples),
            }

    method_order = list(methods) + list((external or {}).keys())
    average_ranks: dict[str, dict[str, float]] = {method: {} for method in method_order}
    for metric_name, direction in METRIC_DIRECTIONS.items():
        ranks_by_method: dict[str, list[float]] = {
            method: [] for method in method_order
        }
        for sample in payload["samples"]:
            sample_id = str(sample["id"])
            sample_rows = {
                str(row["method"]): row
                for row in rows
                if row["sample"] == sample_id and metric_name in row
            }
            if len(sample_rows) != len(method_order):
                continue
            values = np.asarray(
                [float(sample_rows[method][metric_name]) for method in method_order]
            )
            if direction == "higher":
                values = -values
            ranks = rankdata(values, method="average")
            for method, rank in zip(method_order, ranks):
                ranks_by_method[method].append(float(rank))
        for method in method_order:
            if ranks_by_method[method]:
                average_ranks[method][metric_name] = float(
                    np.mean(ranks_by_method[method])
                )
    report = {
        "schema": "reference-video-grade-benchmark/v8-no-gt",
        "dataset": payload.get("dataset"),
        "strength": strength,
        "ranking_policy": {
            "primary": "blinded pairwise style-match and overall preference win rates",
            "objective_axes": [
                "reference grading-style transfer",
                "content preservation",
                "temporal stability",
                "no-reference image quality",
                "technical artifacts",
            ],
            "learned_metrics": learned_metrics,
            "composite_score": None,
        },
        "rows": rows,
        "aggregate": aggregate,
        "aggregate_statistics": aggregate_statistics,
        "average_ranks": average_ranks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    columns = [
        "sample",
        "method",
        *[key for key in rows[0] if key not in {"sample", "method"}],
    ]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row[column]) for column in columns))
    (output_dir / "results.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    aggregate_rows = []
    for method in method_order:
        row: dict[str, object] = {
            "method": method,
            "sequence_count": len(payload["samples"]),
        }
        for key, statistics in aggregate_statistics[method].items():
            row[f"{key}_mean"] = statistics["mean"]
            row[f"{key}_std"] = statistics["std"]
            row[f"{key}_ci95_low"] = statistics["ci95_low"]
            row[f"{key}_ci95_high"] = statistics["ci95_high"]
            if key in average_ranks[method]:
                row[f"{key}_average_rank"] = average_ranks[method][key]
        aggregate_rows.append(row)
    with (output_dir / "aggregate.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    _write_blind_review(
        output_dir, payload["samples"], list(methods) + list((external or {}).keys())
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--methods", nargs="*", choices=sorted(METHOD_NAMES), default=list(METHOD_NAMES)
    )
    parser.add_argument("--strength", type=float, default=0.90)
    parser.add_argument(
        "--learned-metrics",
        action="store_true",
        help="Enable DINOv2 and MUSIQ (downloads model weights).",
    )
    parser.add_argument("--learned-frame-count", type=int, default=8)
    parser.add_argument(
        "--style-vgg-weights",
        type=Path,
        help="Path to vgg_normalised.pth for reference-style feature statistics.",
    )
    parser.add_argument(
        "--external",
        action="append",
        default=[],
        metavar="METHOD=DIR",
        help=(
            "Evaluate black-box visual outputs stored as DIR/<sample>.mp4 or "
            "DIR/<sample>/output.mp4."
        ),
    )
    args = parser.parse_args()
    if not 0.0 <= args.strength <= 1.0:
        raise ValueError("--strength must be in [0, 1].")
    external = {}
    for item in args.external:
        if "=" not in item:
            raise ValueError("--external must use METHOD=DIR")
        name, raw_path = item.split("=", 1)
        external[name] = Path(raw_path).resolve()
    report = run_manifest(
        args.manifest.resolve(),
        args.output_dir.resolve(),
        args.methods,
        args.strength,
        external,
        learned_metrics=args.learned_metrics,
        learned_frame_count=args.learned_frame_count,
        style_vgg_weights=(
            None if args.style_vgg_weights is None else args.style_vgg_weights.resolve()
        ),
    )
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
