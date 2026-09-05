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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance
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


def _reinhard_fit(source: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    source_std = np.maximum(source.std(axis=0), 1e-3)
    reference_mean = reference.mean(axis=0)
    reference_std = np.maximum(reference.std(axis=0), 1e-3)
    scale = np.clip(reference_std / source_std, 0.35, 3.0)
    return source_mean, reference_mean, scale


def _apply_reinhard(image: Image.Image, fit: tuple[np.ndarray, np.ndarray, np.ndarray], strength: float) -> Image.Image:
    source_mean, reference_mean, scale = fit
    lab = _lab(image)
    mapped = (lab - source_mean) * scale + reference_mean
    return _rgb((1.0 - strength) * lab + strength * mapped)


def _mkl_fit(source: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    reference_mean = reference.mean(axis=0)
    cs = np.cov(source, rowvar=False) + np.eye(3) * 1e-3
    cr = np.cov(reference, rowvar=False) + np.eye(3) * 1e-3
    cs_half = np.real(sqrtm(cs))
    cs_inv_half = np.linalg.inv(cs_half)
    transform = cs_inv_half @ np.real(sqrtm(cs_half @ cr @ cs_half)) @ cs_inv_half
    return source_mean, reference_mean, transform


def _apply_mkl(image: Image.Image, fit: tuple[np.ndarray, np.ndarray, np.ndarray], strength: float) -> Image.Image:
    source_mean, reference_mean, transform = fit
    lab = _lab(image)
    mapped = (lab - source_mean) @ transform.T + reference_mean
    return _rgb((1.0 - strength) * lab + strength * mapped)


def identity(target: VideoData, reference: VideoData, strength: float) -> tuple[Image.Image, ...]:
    del reference, strength
    return tuple(frame.copy() for frame in target.frames)


def global_reinhard(target: VideoData, reference: VideoData, strength: float) -> tuple[Image.Image, ...]:
    fit = _reinhard_fit(_pixels(target.frames), _pixels(reference.frames))
    return tuple(_apply_reinhard(frame, fit, strength) for frame in target.frames)


def global_mkl(target: VideoData, reference: VideoData, strength: float) -> tuple[Image.Image, ...]:
    fit = _mkl_fit(_pixels(target.frames), _pixels(reference.frames))
    return tuple(_apply_mkl(frame, fit, strength) for frame in target.frames)


def framewise_reinhard(target: VideoData, reference: VideoData, strength: float) -> tuple[Image.Image, ...]:
    output = []
    for index, frame in enumerate(target.frames):
        reference_index = round(index * (len(reference.frames) - 1) / max(1, len(target.frames) - 1))
        fit = _reinhard_fit(_pixels((frame,), 1), _pixels((reference.frames[reference_index],), 1))
        output.append(_apply_reinhard(frame, fit, strength))
    return tuple(output)


BASELINES: dict[str, Callable[[VideoData, VideoData, float], tuple[Image.Image, ...]]] = {
    "identity": identity,
    "global-reinhard": global_reinhard,
    "global-mkl": global_mkl,
    "framewise-reinhard": framewise_reinhard,
}
METHOD_NAMES = tuple(BASELINES)


def _align_frames(frames: Sequence[Image.Image], target: VideoData) -> tuple[Image.Image, ...]:
    """Time-align any black-box API output to the target video contract."""

    if not frames:
        raise ValueError("Method output contains no frames.")
    aligned = []
    for index in range(len(target.frames)):
        source_index = round(
            index * (len(frames) - 1) / max(1, len(target.frames) - 1)
        )
        aligned.append(
            frames[source_index].convert("RGB").resize(target.frames[index].size)
        )
    return tuple(aligned)


def _delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Vectorised CIEDE2000, used only to describe input/output edit size."""

    l1, a1, b1 = np.moveaxis(np.asarray(lab1, dtype=np.float64), -1, 0)
    l2, a2, b2 = np.moveaxis(np.asarray(lab2, dtype=np.float64), -1, 0)
    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)
    c_bar = (c1 + c2) / 2.0
    g = 0.5 * (1.0 - np.sqrt(c_bar**7 / (c_bar**7 + 25.0**7)))
    a1p, a2p = (1.0 + g) * a1, (1.0 + g) * a2
    c1p = np.sqrt(a1p * a1p + b1 * b1)
    c2p = np.sqrt(a2p * a2p + b2 * b2)
    h1p = np.mod(np.degrees(np.arctan2(b1, a1p)), 360.0)
    h2p = np.mod(np.degrees(np.arctan2(b2, a2p)), 360.0)
    dlp = l2 - l1
    dcp = c2p - c1p
    dhp = h2p - h1p
    dhp = np.where(c1p * c2p == 0.0, 0.0, dhp)
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dhp_rad = np.radians(dhp / 2.0)
    dh_term = 2.0 * np.sqrt(c1p * c2p) * np.sin(dhp_rad)

    l_bar = (l1 + l2) / 2.0
    c_bar_p = (c1p + c2p) / 2.0
    hsum = h1p + h2p
    hdiff = np.abs(h1p - h2p)
    h_bar = np.where(c1p * c2p == 0.0, hsum, hsum / 2.0)
    h_bar = np.where((c1p * c2p != 0.0) & (hdiff > 180.0) & (hsum < 360.0), (hsum + 360.0) / 2.0, h_bar)
    h_bar = np.where((c1p * c2p != 0.0) & (hdiff > 180.0) & (hsum >= 360.0), (hsum - 360.0) / 2.0, h_bar)
    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar))
        + 0.32 * np.cos(np.radians(3.0 * h_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar - 63.0))
    )
    sl = 1.0 + 0.015 * (l_bar - 50.0) ** 2 / np.sqrt(20.0 + (l_bar - 50.0) ** 2)
    sc = 1.0 + 0.045 * c_bar_p
    sh = 1.0 + 0.015 * c_bar_p * t
    delta_theta = 30.0 * np.exp(-((h_bar - 275.0) / 25.0) ** 2)
    rc = 2.0 * np.sqrt(c_bar_p**7 / (c_bar_p**7 + 25.0**7))
    rt = -rc * np.sin(np.radians(2.0 * delta_theta))
    dl, dc, dh = dlp / sl, dcp / sc, dh_term / sh
    return np.sqrt(np.maximum(0.0, dl * dl + dc * dc + dh * dh + rt * dc * dh))


def _local_structure(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Remove local luminance/contrast so legitimate grading is discounted."""

    gray = cv2.cvtColor(
        np.asarray(image.convert("RGB").resize(size), dtype=np.float32) / 255.0,
        cv2.COLOR_RGB2GRAY,
    )
    mean = cv2.GaussianBlur(gray, (0, 0), 3.0)
    variance = np.maximum(cv2.GaussianBlur(gray * gray, (0, 0), 3.0) - mean * mean, 1e-4)
    return (gray - mean) / np.sqrt(variance)


def _structure_correlation(left: Image.Image, right: Image.Image, size: tuple[int, int]) -> float:
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


def _lab_histogram_emd(
    output: Sequence[Image.Image], reference: Sequence[Image.Image]
) -> float:
    """Mean normalized 1-D Wasserstein distance over the L*, a*, b* marginals.

    The value is a cross-content color-distribution diagnostic. It is kept as
    its own axis because scene composition can change it even when a grade is
    perceptually correct.
    """

    output_pixels = _pixels(output)
    reference_pixels = _pixels(reference)
    ranges = ((0.0, 100.0), (-128.0, 127.0), (-128.0, 127.0))
    distances = []
    for channel, (lower, upper) in enumerate(ranges):
        out_hist, _ = np.histogram(output_pixels[:, channel], bins=65, range=(lower, upper))
        ref_hist, _ = np.histogram(reference_pixels[:, channel], bins=65, range=(lower, upper))
        # histogram has 65 bins; use the corresponding bin centres.
        edges = np.linspace(lower, upper, 66)
        bin_centers = (edges[:-1] + edges[1:]) / 2.0
        distance = wasserstein_distance(
            bin_centers,
            bin_centers,
            u_weights=out_hist.astype(np.float64),
            v_weights=ref_hist.astype(np.float64),
        )
        distances.append(float(distance / (upper - lower)))
    return float(np.mean(distances))


def _valid_transitions(source: Sequence[Image.Image]) -> np.ndarray:
    """Return transitions that are unlikely to be hard cuts."""

    if len(source) < 2:
        return np.zeros(0, dtype=bool)
    scores = []
    for previous, current in zip(source[:-1], source[1:]):
        size = (min(160, current.width), min(90, current.height))
        a = np.asarray(previous.convert("RGB").resize(size), dtype=np.float32) / 255.0
        b = np.asarray(current.convert("RGB").resize(size), dtype=np.float32) / 255.0
        scores.append(float(np.mean(np.abs(a - b))))
    values = np.asarray(scores)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(0.18, median + 6.0 * max(mad, 1e-4))
    return values <= threshold


def _motion_compensated_edit_residual(source: Sequence[Image.Image], output: Sequence[Image.Image]) -> float:
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
        previous_out = np.asarray(output[index - 1].resize(size), dtype=np.float32) / 255.0
        current_out = np.asarray(output[index].resize(size), dtype=np.float32) / 255.0
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(previous_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        grid_x, grid_y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
        map_x = (grid_x - flow[..., 0]).astype(np.float32)
        map_y = (grid_y - flow[..., 1]).astype(np.float32)
        edit_previous = previous_out - previous
        edit_current = current_out - current
        warped = cv2.remap(edit_previous, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
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
        previous_out = np.asarray(output[index - 1].resize(size), dtype=np.float32) / 255.0
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


def _temporal_transform_drift(source: Sequence[Image.Image], output: Sequence[Image.Image]) -> float:
    if len(source) < 2:
        return 0.0
    signatures = np.asarray([_frame_edit_signature(a, b) for a, b in zip(source, output)])
    differences = np.linalg.norm(np.diff(signatures, axis=0), axis=1)
    valid = _valid_transitions(source)
    return float(np.mean(differences[valid])) if valid.any() else 0.0


def metrics(
    target: VideoData,
    output: Sequence[Image.Image],
    reference: VideoData | None = None,
    learned_suite=None,
    instruction: str | None = None,
) -> dict[str, float | None]:
    target_pixels = _pixels(target.frames)
    output_pixels = _pixels(output)
    edit_delta_e00 = _delta_e_ciede2000(target_pixels, output_pixels)
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
        # Descriptor only: there is deliberately no up/down direction or gate.
        "edit_magnitude_delta_e00": float(np.mean(edit_delta_e00)),
        "edited_pixel_fraction_delta_e00_gt_2": float(np.mean(edit_delta_e00 > 2.0)),
        "content_structure_correlation": float(np.mean(structure_scores)),
        "edge_ssim": float(np.mean(edge_ssim_scores)),
        "lab_histogram_emd": (
            _lab_histogram_emd(output, reference.frames) if reference is not None else None
        ),
        "temporal_flow_warp_error": _motion_compensated_output_residual(target.frames, output),
        "temporal_edit_warp_error": _motion_compensated_edit_residual(target.frames, output),
        "temporal_transform_drift": _temporal_transform_drift(target.frames, output),
        "new_shadow_clip_fraction": max(0.0, shadow_clip - source_shadow_clip),
        "new_highlight_clip_fraction": max(0.0, highlight_clip - source_highlight_clip),
    }
    if learned_suite is not None:
        if reference is None:
            raise ValueError("Learned reference metrics require reference video frames.")
        values.update(
            learned_suite.evaluate(target.frames, output, reference.frames, instruction)
        )
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
            key=lambda method: hashlib.sha256(f"{sample_id}:{method}".encode()).hexdigest(),
        )
        codes: list[str] = []
        for index, method in enumerate(ordered, start=1):
            code = f"C{index:02d}"
            codes.append(code)
            destination = blind_dir / f"{code}.mp4"
            shutil.copy2(sample_dir / f"{method}.mp4", destination)
            relative = destination.relative_to(output_dir).as_posix()
            assignments.append({"sample": sample_id, "candidate_code": code, "method": method})
            individual_rows.append(
                {
                    "sample": sample_id,
                    "candidate_code": code,
                    "target_video": (blind_dir / "target.mp4").relative_to(output_dir).as_posix(),
                    "reference_video": (blind_dir / "reference.mp4").relative_to(output_dir).as_posix(),
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
                        "target_video": (blind_dir / "target.mp4").relative_to(output_dir).as_posix(),
                        "reference_video": (blind_dir / "reference.mp4").relative_to(output_dir).as_posix(),
                        "candidate_a": left,
                        "candidate_a_video": (blind_dir / f"{left}.mp4").relative_to(output_dir).as_posix(),
                        "candidate_b": right,
                        "candidate_b_video": (blind_dir / f"{right}.mp4").relative_to(output_dir).as_posix(),
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


def _mosaic(target: VideoData, reference: VideoData, outputs: dict[str, Sequence[Image.Image]], path: Path) -> None:
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
            source_index = round(index * (len(sequence) - 1) / max(1, len(target.frames) - 1))
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
) -> dict[str, object]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    rows = []
    learned_suite = None
    if learned_metrics:
        from evaluation.perceptual_metrics import LearnedMetricSuite

        learned_suite = LearnedMetricSuite(frame_count=learned_frame_count)
    for sample in payload["samples"]:
        sample_id = str(sample["id"])
        target = _load((root / sample["target"]).resolve(), sample.get("max_frames"), sample.get("max_side"))
        reference = _load((root / sample["reference"]).resolve(), sample.get("max_frames"), sample.get("max_side"))
        sample_outputs = {}
        requested = list(methods) + list((external or {}).keys())
        method_dir = output_dir / sample_id
        method_dir.mkdir(parents=True, exist_ok=True)
        encode_video(target.frames, method_dir / "target.mp4", target.fps, preset="veryfast")
        encode_video(reference.frames, method_dir / "reference.mp4", reference.fps, preset="veryfast")
        for method in requested:
            start = time.perf_counter()
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
            encode_video(result, method_dir / f"{method}.mp4", target.fps, preset="veryfast")
            result_metrics = metrics(
                target,
                result,
                reference=reference,
                learned_suite=learned_suite,
                instruction=sample.get("instruction"),
            )
            result_metrics["runtime_seconds"] = elapsed
            result_metrics["processing_fps"] = len(result) / max(elapsed, 1e-8)
            rows.append({"sample": sample_id, "method": method, **result_metrics})
            sample_outputs[method] = result
        _mosaic(target, reference, sample_outputs, output_dir / sample_id / "comparison.mp4")
    aggregate = {}
    for method in list(methods) + list((external or {}).keys()):
        method_rows = [row for row in rows if row["method"] == method]
        keys = [
            key
            for key in method_rows[0]
            if any(isinstance(row.get(key), (int, float)) for row in method_rows)
        ]
        aggregate[method] = {
            key: float(np.mean([row[key] for row in method_rows if isinstance(row.get(key), (int, float))]))
            for key in keys
        }
    report = {
        "schema": "reference-video-grade-benchmark/v3-no-gt",
        "dataset": payload.get("dataset"),
        "strength": strength,
        "ranking_policy": {
            "primary": "blinded pairwise style-match and overall preference win rates",
            "objective_axes": [
                "reference color distribution",
                "content preservation",
                "temporal stability",
                "no-reference image quality",
                "technical artifacts",
            ],
            "learned_metrics": learned_metrics,
            "edit_magnitude": "descriptor only; no direction, gate, or score contribution",
            "composite_score": None,
        },
        "rows": rows,
        "aggregate": aggregate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    columns = ["sample", "method", *[key for key in rows[0] if key not in {"sample", "method"}]]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row[column]) for column in columns))
    (output_dir / "results.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_blind_review(output_dir, payload["samples"], list(methods) + list((external or {}).keys()))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHOD_NAMES), default=list(METHOD_NAMES))
    parser.add_argument("--strength", type=float, default=0.90)
    parser.add_argument(
        "--learned-metrics",
        action="store_true",
        help="Enable CLIP, DINOv2, MUSIQ and CLIP-IQA (downloads model weights).",
    )
    parser.add_argument("--learned-frame-count", type=int, default=8)
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
    )
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
