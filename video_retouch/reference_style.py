"""Content-independent measurements for reference-video color grading."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from PIL import Image


def _sample_pixels(
    frames: Sequence[Image.Image], frame_count: int = 8, max_side: int = 128
) -> tuple[np.ndarray, list[np.ndarray]]:
    if not frames:
        raise ValueError("A reference-style profile requires at least one frame.")
    indices = np.unique(
        np.linspace(0, len(frames) - 1, min(frame_count, len(frames)))
        .round()
        .astype(int)
    )
    per_frame = []
    for index in indices:
        image = frames[int(index)].convert("RGB")
        scale = min(1.0, max_side / max(image.size))
        size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        rgb = np.asarray(image.resize(size, Image.Resampling.LANCZOS))
        lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        lab[..., 0] /= 100.0
        lab[..., 1:] /= 128.0
        per_frame.append(lab.reshape(-1, 3).astype(np.float64))
    pixels = np.concatenate(per_frame, axis=0)
    if len(pixels) > 80_000:
        pixels = pixels[np.linspace(0, len(pixels) - 1, 80_000).astype(int)]
    return pixels, per_frame


def _round_list(values: np.ndarray, digits: int = 4) -> list[float]:
    return [round(float(value), digits) for value in values]


def _hue_name(a: float, b: float) -> str:
    chroma = float(np.hypot(a, b))
    if chroma < 0.025:
        return "neutral"
    angle = float(np.degrees(np.arctan2(b, a)) % 360.0)
    names = (
        "red",
        "orange",
        "yellow",
        "yellow-green",
        "green",
        "cyan",
        "blue",
        "violet",
        "magenta",
        "red-magenta",
    )
    return names[int(((angle + 18.0) % 360.0) // 36.0)]


def _palette(pixels: np.ndarray, count: int = 5) -> list[dict[str, object]]:
    samples = pixels
    if len(samples) > 20_000:
        samples = samples[np.linspace(0, len(samples) - 1, 20_000).astype(int)]
    centers = [
        samples[int(np.argmin(np.linalg.norm(samples - samples.mean(0), axis=1)))]
    ]
    for _ in range(1, count):
        distances = np.min(
            np.stack(
                [np.sum((samples - center) ** 2, axis=1) for center in centers],
                axis=1,
            ),
            axis=1,
        )
        centers.append(samples[int(np.argmax(distances))])
    centers_array = np.stack(centers)
    for _ in range(12):
        distances = np.sum(
            (samples[:, None, :] - centers_array[None, :, :]) ** 2, axis=2
        )
        labels = np.argmin(distances, axis=1)
        updated = centers_array.copy()
        for cluster in range(count):
            members = samples[labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        if np.allclose(updated, centers_array, atol=1e-5):
            break
        centers_array = updated
    distances = np.sum((pixels[:, None, :] - centers_array[None, :, :]) ** 2, axis=2)
    labels = np.argmin(distances, axis=1)
    weights = np.bincount(labels, minlength=count) / len(labels)
    order = np.argsort(-weights, kind="stable")
    result = []
    for cluster in order:
        lightness, a, b = centers_array[int(cluster)]
        result.append(
            {
                "weight": round(float(weights[int(cluster)]), 4),
                "lightness": round(float(lightness), 4),
                "a_green_to_magenta": round(float(a), 4),
                "b_blue_to_yellow": round(float(b), 4),
                "chroma": round(float(np.hypot(a, b)), 4),
                "hue_family": _hue_name(float(a), float(b)),
            }
        )
    return result


def _profile(frames: Sequence[Image.Image]) -> dict[str, object]:
    pixels, per_frame = _sample_pixels(frames)
    lightness = pixels[:, 0]
    chroma = np.linalg.norm(pixels[:, 1:], axis=1)
    quantiles = np.quantile(lightness, [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98])
    zone_specs = (
        ("deep_shadows", 0.00, 0.20),
        ("shadows", 0.20, 0.40),
        ("midtones", 0.40, 0.70),
        ("highlights", 0.70, 0.90),
        ("speculars", 0.90, 1.01),
    )
    zones: dict[str, object] = {}
    for name, lower, upper in zone_specs:
        mask = (lightness >= lower) & (lightness < upper)
        if not np.any(mask):
            zones[name] = {"pixel_fraction": 0.0}
            continue
        mean = pixels[mask].mean(axis=0)
        zones[name] = {
            "pixel_fraction": round(float(mask.mean()), 4),
            "mean_lightness": round(float(mean[0]), 4),
            "mean_a_green_to_magenta": round(float(mean[1]), 4),
            "mean_b_blue_to_yellow": round(float(mean[2]), 4),
            "mean_chroma": round(float(chroma[mask].mean()), 4),
            "hue_family": _hue_name(float(mean[1]), float(mean[2])),
        }
    frame_medians = np.asarray([np.median(frame[:, 0]) for frame in per_frame])
    frame_chroma = np.asarray(
        [np.mean(np.linalg.norm(frame[:, 1:], axis=1)) for frame in per_frame]
    )
    return {
        "lightness_quantiles_q02_q10_q25_q50_q75_q90_q98": _round_list(quantiles),
        "chroma_quantiles_q10_q50_q90": _round_list(
            np.quantile(chroma, [0.10, 0.50, 0.90])
        ),
        "tone_zones": zones,
        "dominant_palette": _palette(pixels),
        "temporal_dispersion": {
            "median_lightness_std": round(float(frame_medians.std()), 4),
            "mean_chroma_std": round(float(frame_chroma.std()), 4),
        },
    }


def build_reference_style_profile(
    target_frames: Sequence[Image.Image],
    reference_frames: Sequence[Image.Image],
) -> dict[str, object]:
    """Measure target and reference separately without pixel correspondence."""

    target = _profile(target_frames)
    reference = _profile(reference_frames)
    target_q = np.asarray(target["lightness_quantiles_q02_q10_q25_q50_q75_q90_q98"])
    reference_q = np.asarray(
        reference["lightness_quantiles_q02_q10_q25_q50_q75_q90_q98"]
    )
    target_c = np.asarray(target["chroma_quantiles_q10_q50_q90"])
    reference_c = np.asarray(reference["chroma_quantiles_q10_q50_q90"])
    return {
        "schema": "reference-style-profile/v1",
        "coordinate_system": (
            "CIELAB normalized as L/100, a/128, b/128; positive a is magenta, "
            "negative a green, positive b yellow/warm, negative b blue/cool"
        ),
        "target": target,
        "reference": reference,
        "reference_minus_target": {
            "lightness_quantile_delta": _round_list(reference_q - target_q),
            "chroma_quantile_delta": _round_list(reference_c - target_c),
        },
        "usage": (
            "Infer a content-aware grading transform from the two independent "
            "profiles. Do not match raw object colors across unrelated scenes."
        ),
    }
