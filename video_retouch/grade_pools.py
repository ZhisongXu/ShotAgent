"""Typed color-grading pools and their deterministic local executor.

The v2 grading path deliberately models a grade as a sparse graph of typed
operations instead of a single flat parameter vector.  Every value returned by
an untrusted vision-language model is canonicalized here before rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np
from PIL import Image


POOL_OPERATION_TYPES = frozenset(
    {
        "primary",
        "white_balance",
        "global_color",
        "hsl8",
        "color_wheels",
        "curves",
        "texture",
        "optical_effects",
        "denoise",
    }
)

POOL_PROCESSING_ORDER = (
    "denoise",
    "white_balance",
    "primary",
    "color_wheels",
    "curves",
    "hsl8",
    "global_color",
    "texture",
    "optical_effects",
)

POOL_STAGE_TYPES = {
    "technical": ("denoise", "white_balance", "primary"),
    "look": ("color_wheels", "curves", "global_color"),
    "selective_color": ("hsl8",),
    "texture": ("texture",),
    "optical": ("optical_effects",),
}

HSL_REGIONS = {
    "red": 0.0,
    "orange": 30.0,
    "yellow": 60.0,
    "green": 120.0,
    "aqua": 180.0,
    "blue": 240.0,
    "purple": 275.0,
    "magenta": 320.0,
}


def _finite(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _bounded(value: object, name: str, lower: float, upper: float) -> float:
    result = _finite(value, name)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return result


def _mapping(value: object, name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object.")
    return dict(value)


def _only(values: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")


def _slider(
    values: Mapping[str, object], name: str, default: float = 0.0
) -> float:
    return _bounded(values.get(name, default), name, -100.0, 100.0)


def _curve_samples(points: object, name: str) -> list[float]:
    if not isinstance(points, list) or not 2 <= len(points) <= 16:
        raise ValueError(f"{name} must contain 2 to 16 [input,output] points.")
    parsed: list[list[float]] = []
    for index, point in enumerate(points):
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{name}[{index}] must be [input,output].")
        parsed.append(
            [
                _bounded(point[0], f"{name}[{index}][0]", 0.0, 1.0),
                _bounded(point[1], f"{name}[{index}][1]", 0.0, 1.0),
            ]
        )
    xs = np.asarray([point[0] for point in parsed], dtype=np.float32)
    ys = np.asarray([point[1] for point in parsed], dtype=np.float32)
    if xs[0] != 0.0 or xs[-1] != 1.0:
        raise ValueError(f"{name} must start at x=0 and end at x=1.")
    if np.any(np.diff(xs) <= 0.0):
        raise ValueError(f"{name} input coordinates must be strictly increasing.")
    if np.any(np.diff(ys) < 0.0):
        raise ValueError(f"{name} output coordinates must be monotonic.")
    axis = np.linspace(0.0, 1.0, 17, dtype=np.float32)
    return np.interp(axis, xs, ys).astype(np.float32).tolist()


def _wheel(value: object, name: str) -> dict[str, float]:
    raw = _mapping(value, name)
    allowed = {"hue", "saturation", "x", "y"}
    _only(raw, allowed, name)
    cartesian = "x" in raw or "y" in raw
    polar = "hue" in raw or "saturation" in raw
    if cartesian and polar:
        raise ValueError(f"{name} must use hue/saturation or x/y, not both.")
    if cartesian:
        return {
            "x": _bounded(raw.get("x", 0.0), f"{name}.x", -1.0, 1.0),
            "y": _bounded(raw.get("y", 0.0), f"{name}.y", -1.0, 1.0),
        }
    hue = np.deg2rad(_bounded(raw.get("hue", 0.0), f"{name}.hue", -180, 180))
    saturation = _bounded(
        raw.get("saturation", 0.0), f"{name}.saturation", 0.0, 100.0
    ) / 100.0
    return {
        "x": float(np.cos(hue) * saturation),
        "y": float(np.sin(hue) * saturation),
    }


def canonicalize_pool_parameters(
    operation_type: str, parameters: object
) -> dict[str, object]:
    """Validate model-authored parameters and fill deterministic defaults."""

    raw = _mapping(parameters, f"{operation_type}.parameters")
    if operation_type == "primary":
        allowed = {
            "exposure",
            "contrast",
            "highlights",
            "shadows",
            "whites",
            "blacks",
            "gamma",
        }
        _only(raw, allowed, operation_type)
        return {
            "exposure": _bounded(raw.get("exposure", 0.0), "primary.exposure", -5, 5),
            "contrast": _slider(raw, "contrast"),
            "highlights": _slider(raw, "highlights"),
            "shadows": _slider(raw, "shadows"),
            "whites": _slider(raw, "whites"),
            "blacks": _slider(raw, "blacks"),
            "gamma": _bounded(raw.get("gamma", 1.0), "primary.gamma", 0.1, 3.0),
        }

    if operation_type == "white_balance":
        _only(raw, {"temperature", "tint"}, operation_type)
        return {
            "temperature": _bounded(
                raw.get("temperature", 6500.0),
                "white_balance.temperature",
                2000,
                12000,
            ),
            "tint": _bounded(raw.get("tint", 0.0), "white_balance.tint", -100, 100),
        }

    if operation_type == "global_color":
        _only(raw, {"saturation", "vibrance", "hue_shift"}, operation_type)
        return {
            "saturation": _slider(raw, "saturation"),
            "vibrance": _slider(raw, "vibrance"),
            "hue_shift": _bounded(
                raw.get("hue_shift", 0.0), "global_color.hue_shift", -180, 180
            ),
        }

    if operation_type == "hsl8":
        _only(raw, set(HSL_REGIONS), operation_type)
        result: dict[str, object] = {}
        for region in HSL_REGIONS:
            values = _mapping(raw.get(region), f"hsl8.{region}")
            _only(values, {"hue", "saturation", "luminance"}, f"hsl8.{region}")
            result[region] = {
                "hue": _bounded(values.get("hue", 0.0), f"hsl8.{region}.hue", -100, 100),
                "saturation": _bounded(
                    values.get("saturation", 0.0),
                    f"hsl8.{region}.saturation",
                    -100,
                    100,
                ),
                "luminance": _bounded(
                    values.get("luminance", 0.0),
                    f"hsl8.{region}.luminance",
                    -100,
                    100,
                ),
            }
        return result

    if operation_type == "color_wheels":
        _only(raw, {"shadows", "midtones", "highlights", "balance"}, operation_type)
        return {
            "shadows": _wheel(raw.get("shadows"), "color_wheels.shadows"),
            "midtones": _wheel(raw.get("midtones"), "color_wheels.midtones"),
            "highlights": _wheel(raw.get("highlights"), "color_wheels.highlights"),
            "balance": _bounded(
                raw.get("balance", 0.0), "color_wheels.balance", -100, 100
            ),
        }

    if operation_type == "curves":
        _only(raw, {"rgb", "red", "green", "blue", "strength"}, operation_type)
        channels: dict[str, object] = {}
        identity = np.linspace(0.0, 1.0, 17, dtype=np.float32).tolist()
        for channel in ("rgb", "red", "green", "blue"):
            channels[channel] = (
                identity
                if channel not in raw
                else _curve_samples(raw[channel], f"curves.{channel}")
            )
        channels["strength"] = _bounded(
            raw.get("strength", 1.0), "curves.strength", 0.0, 1.0
        )
        return channels

    if operation_type == "texture":
        _only(raw, {"clarity", "texture", "dehaze", "sharpening"}, operation_type)
        return {
            "clarity": _slider(raw, "clarity"),
            "texture": _slider(raw, "texture"),
            "dehaze": _slider(raw, "dehaze"),
            "sharpening": _bounded(
                raw.get("sharpening", 0.0), "texture.sharpening", 0, 150
            ),
        }

    if operation_type == "denoise":
        _only(raw, {"luminance", "color"}, operation_type)
        return {
            "luminance": _bounded(raw.get("luminance", 0.0), "denoise.luminance", 0, 100),
            "color": _bounded(raw.get("color", 0.0), "denoise.color", 0, 100),
        }

    if operation_type == "optical_effects":
        allowed = {
            "vignette",
            "grain",
            "bloom",
            "halation",
            "diffusion",
            "chromatic_aberration",
        }
        _only(raw, allowed, operation_type)
        vignette = _mapping(raw.get("vignette"), "optical_effects.vignette")
        _only(vignette, {"amount", "midpoint", "feather", "roundness"}, "vignette")
        grain = _mapping(raw.get("grain"), "optical_effects.grain")
        _only(grain, {"amount", "size", "roughness"}, "grain")
        bloom = _mapping(raw.get("bloom"), "optical_effects.bloom")
        _only(bloom, {"intensity", "threshold"}, "bloom")
        halation = _mapping(raw.get("halation"), "optical_effects.halation")
        _only(halation, {"intensity", "threshold"}, "halation")
        diffusion = _mapping(raw.get("diffusion"), "optical_effects.diffusion")
        _only(diffusion, {"strength"}, "diffusion")
        aberration = _mapping(
            raw.get("chromatic_aberration"), "optical_effects.chromatic_aberration"
        )
        _only(aberration, {"amount"}, "chromatic_aberration")
        return {
            "vignette": {
                "amount": _bounded(vignette.get("amount", 0), "vignette.amount", -100, 100),
                "midpoint": _bounded(vignette.get("midpoint", 50), "vignette.midpoint", 0, 100),
                "feather": _bounded(vignette.get("feather", 50), "vignette.feather", 0, 100),
                "roundness": _bounded(vignette.get("roundness", 0), "vignette.roundness", -100, 100),
            },
            "grain": {
                "amount": _bounded(grain.get("amount", 0), "grain.amount", 0, 100),
                "size": _bounded(grain.get("size", 25), "grain.size", 1, 100),
                "roughness": _bounded(grain.get("roughness", 50), "grain.roughness", 0, 100),
            },
            "bloom": {
                "intensity": _bounded(bloom.get("intensity", 0), "bloom.intensity", 0, 100),
                "threshold": _bounded(bloom.get("threshold", 0.8), "bloom.threshold", 0, 1),
            },
            "halation": {
                "intensity": _bounded(halation.get("intensity", 0), "halation.intensity", 0, 100),
                "threshold": _bounded(halation.get("threshold", 0.85), "halation.threshold", 0, 1),
            },
            "diffusion": {
                "strength": _bounded(diffusion.get("strength", 0), "diffusion.strength", 0, 100),
            },
            "chromatic_aberration": {
                "amount": _bounded(aberration.get("amount", 0), "chromatic_aberration.amount", -100, 100),
            },
        }

    raise ValueError(f"Unknown grade Pool operation: {operation_type}")


def pool_contract() -> dict[str, object]:
    """Machine-readable schema included in manifests and model prompts."""

    return {
        "schema_version": "grade-pool-contract/v2",
        "working_space": {
            "analysis_proxy": "tone-mapped sRGB for VL inspection",
            "render_precision": "float32 batch; 16-bit RGB decode; 8/10/12-bit delivery",
            "scene_linear": ["ACEScg", "ACES2065-1", "linear Rec.709", "linear Rec.2020"],
            "input_transfers": ["sRGB", "Rec.709", "LogC3", "S-Log3", "V-Log", "PQ", "HLG"],
            "ocio": "optional studio config with explicit input/working/display/output spaces",
            "linear_light": ["white_balance", "primary", "color_wheels"],
            "display_referred": [
                "curves",
                "hsl8",
                "global_color",
                "texture",
                "optical_effects",
            ],
        },
        "processing_order": list(POOL_PROCESSING_ORDER),
        "execution": {
            "primary": "Torch BCHW batch on CUDA when available",
            "fallback": "Torch CPU or deterministic NumPy/OpenCV",
            "semantic_masks": ["global", "person", "skin", "sky"],
            "mask_tracking": "Farneback optical flow with periodic semantic refresh",
        },
        "stages": {key: list(value) for key, value in POOL_STAGE_TYPES.items()},
        "ranges": {
            "primary": {
                "exposure": [-5, 5],
                "contrast": [-100, 100],
                "highlights": [-100, 100],
                "shadows": [-100, 100],
                "whites": [-100, 100],
                "blacks": [-100, 100],
                "gamma": [0.1, 3.0],
            },
            "white_balance": {"temperature": [2000, 12000], "tint": [-100, 100]},
            "global_color": {
                "saturation": [-100, 100],
                "vibrance": [-100, 100],
                "hue_shift": [-180, 180],
            },
            "hsl8": {
                "regions": list(HSL_REGIONS),
                "hue_saturation_luminance": [-100, 100],
            },
            "color_wheels": {
                "hue": [-180, 180],
                "saturation": [0, 100],
                "balance": [-100, 100],
            },
            "curves": {"channels": ["rgb", "red", "green", "blue"], "points": [2, 16]},
            "texture": {
                "clarity": [-100, 100],
                "texture": [-100, 100],
                "dehaze": [-100, 100],
                "sharpening": [0, 150],
            },
            "denoise": {"luminance": [0, 100], "color": [0, 100]},
            "optical_effects": {
                "vignette": "amount/midpoint/feather/roundness",
                "grain": "amount/size/roughness",
                "bloom": "intensity/threshold",
                "halation": "intensity/threshold",
                "diffusion": "strength",
                "chromatic_aberration": "amount",
            },
        },
    }


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        np.power((rgb + 0.055) / 1.055, 2.4),
    )


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.maximum(rgb, 0.0)
    return np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
    )


def _luma(rgb: np.ndarray) -> np.ndarray:
    return np.sum(rgb * np.asarray([0.2126, 0.7152, 0.0722], np.float32), axis=-1)


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    amount = np.clip((value - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return amount * amount * (3.0 - 2.0 * amount)


def _rgb_to_hsl(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    delta = maximum - minimum
    lightness = (maximum + minimum) * 0.5
    saturation = np.zeros_like(lightness)
    chromatic = delta > 1e-7
    saturation[chromatic] = delta[chromatic] / np.maximum(
        1e-7, 1.0 - np.abs(2.0 * lightness[chromatic] - 1.0)
    )
    hue = np.zeros_like(lightness)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red_max = chromatic & (maximum == red)
    green_max = chromatic & (maximum == green) & ~red_max
    blue_max = chromatic & ~(red_max | green_max)
    hue[red_max] = 60.0 * (((green[red_max] - blue[red_max]) / delta[red_max]) % 6.0)
    hue[green_max] = 60.0 * ((blue[green_max] - red[green_max]) / delta[green_max] + 2.0)
    hue[blue_max] = 60.0 * ((red[blue_max] - green[blue_max]) / delta[blue_max] + 4.0)
    return hue, saturation, lightness


def _hsl_to_rgb(hue: np.ndarray, saturation: np.ndarray, lightness: np.ndarray) -> np.ndarray:
    chroma = (1.0 - np.abs(2.0 * lightness - 1.0)) * saturation
    sector = (hue % 360.0) / 60.0
    x_value = chroma * (1.0 - np.abs((sector % 2.0) - 1.0))
    zeros = np.zeros_like(chroma)
    candidates = (
        (chroma, x_value, zeros),
        (x_value, chroma, zeros),
        (zeros, chroma, x_value),
        (zeros, x_value, chroma),
        (x_value, zeros, chroma),
        (chroma, zeros, x_value),
    )
    index = np.floor(sector).astype(np.int64) % 6
    rgb = np.zeros((*hue.shape, 3), dtype=np.float32)
    for sector_index, values in enumerate(candidates):
        mask = index == sector_index
        if np.any(mask):
            rgb[mask] = np.stack(values, axis=-1)[mask]
    return rgb + (lightness - chroma * 0.5)[..., None]


def _apply_denoise(rgb: np.ndarray, p: Mapping[str, object]) -> np.ndarray:
    luminance_strength = float(p["luminance"]) / 100.0
    color_strength = float(p["color"]) / 100.0
    if luminance_strength <= 0 and color_strength <= 0:
        return rgb
    lab = cv2.cvtColor(np.clip(rgb, 0, 1), cv2.COLOR_RGB2LAB)
    light, a_channel, b_channel = cv2.split(lab)
    if luminance_strength > 0:
        diameter = 5 if luminance_strength < 0.5 else 7
        light = cv2.bilateralFilter(
            light,
            diameter,
            8.0 + 28.0 * luminance_strength,
            2.0 + 5.0 * luminance_strength,
        )
    if color_strength > 0:
        sigma = 0.4 + 2.8 * color_strength
        a_channel = cv2.GaussianBlur(a_channel, (0, 0), sigmaX=sigma)
        b_channel = cv2.GaussianBlur(b_channel, (0, 0), sigmaX=sigma)
    return cv2.cvtColor(cv2.merge((light, a_channel, b_channel)), cv2.COLOR_LAB2RGB)


def _apply_white_balance(rgb: np.ndarray, p: Mapping[str, object]) -> np.ndarray:
    linear = _srgb_to_linear(rgb)
    kelvin = float(p["temperature"])
    warmth = ((1_000_000.0 / kelvin) - (1_000_000.0 / 6500.0)) / 350.0
    tint = float(p["tint"]) / 100.0
    gains = np.asarray(
        [
            np.exp(0.48 * warmth + 0.10 * tint),
            np.exp(-0.20 * tint),
            np.exp(-0.48 * warmth + 0.10 * tint),
        ],
        dtype=np.float32,
    )
    return _linear_to_srgb(linear * gains)


def _apply_primary(rgb: np.ndarray, p: Mapping[str, object]) -> np.ndarray:
    linear = _srgb_to_linear(rgb)
    linear *= 2.0 ** float(p["exposure"])
    luminance = np.clip(_luma(linear), 0.0, 1.0)
    highlights = float(p["highlights"]) / 100.0
    shadows = float(p["shadows"]) / 100.0
    scale = 1.0 + 0.75 * highlights * luminance**2
    scale += 0.75 * shadows * (1.0 - luminance) ** 2
    linear *= np.maximum(scale[..., None], 0.05)
    whites = float(p["whites"]) / 100.0
    blacks = float(p["blacks"]) / 100.0
    white_weight = _smoothstep(0.55, 1.0, luminance)
    black_weight = 1.0 - _smoothstep(0.0, 0.40, luminance)
    linear += (0.18 * whites * white_weight + 0.10 * blacks * black_weight)[..., None]
    contrast = float(p["contrast"]) / 100.0
    factor = 2.0 ** (1.5 * contrast)
    linear = (linear - 0.18) * factor + 0.18
    gamma = float(p["gamma"])
    linear = np.power(np.maximum(linear, 1e-7), 1.0 / gamma)
    return _linear_to_srgb(linear)


def _apply_color_wheels(rgb: np.ndarray, p: Mapping[str, object]) -> np.ndarray:
    linear = _srgb_to_linear(rgb)
    luminance = np.clip(_luma(linear), 0.0, 1.0)
    balance = float(p["balance"]) / 250.0
    shadow = 1.0 - _smoothstep(0.05 + balance, 0.58 + balance, luminance)
    highlight = _smoothstep(0.42 + balance, 0.95 + balance, luminance)
    midtone = np.clip(1.0 - shadow - highlight, 0.0, 1.0)
    weights = (shadow, midtone, highlight)
    result = linear.copy()
    for zone, weight in zip(("shadows", "midtones", "highlights"), weights):
        x_value = float(p[zone]["x"])
        y_value = float(p[zone]["y"])
        delta = np.asarray(
            [x_value + 0.5 * y_value, -0.5 * x_value + 0.25 * y_value, -x_value - 0.75 * y_value],
            dtype=np.float32,
        )
        result += weight[..., None] * delta * 0.16
    return _linear_to_srgb(np.maximum(result, 0.0))


def _apply_curves(rgb: np.ndarray, p: Mapping[str, object]) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, 17, dtype=np.float32)
    strength = float(p["strength"])
    result = np.clip(rgb, 0.0, 1.0).copy()
    master = np.asarray(p["rgb"], dtype=np.float32)
    mapped = np.stack(
        [np.interp(result[..., channel], axis, master) for channel in range(3)],
        axis=-1,
    )
    result += strength * (mapped - result)
    for index, channel in enumerate(("red", "green", "blue")):
        samples = np.asarray(p[channel], dtype=np.float32)
        mapped_channel = np.interp(np.clip(result[..., index], 0, 1), axis, samples)
        result[..., index] += strength * (mapped_channel - result[..., index])
    return result


def _apply_hsl8(rgb: np.ndarray, p: Mapping[str, object]) -> np.ndarray:
    hue, saturation, lightness = _rgb_to_hsl(np.clip(rgb, 0.0, 1.0))
    hue_delta = np.zeros_like(hue)
    saturation_delta = np.zeros_like(saturation)
    lightness_delta = np.zeros_like(lightness)
    total = np.zeros_like(hue)
    for region, center in HSL_REGIONS.items():
        distance = np.abs((hue - center + 180.0) % 360.0 - 180.0)
        weight = np.clip(1.0 - distance / 45.0, 0.0, 1.0)
        weight = weight * weight * (3.0 - 2.0 * weight)
        values = p[region]
        hue_delta += weight * float(values["hue"]) * 0.30
        saturation_delta += weight * float(values["saturation"]) / 100.0
        lightness_delta += weight * float(values["luminance"]) / 200.0
        total += weight
    normalization = np.maximum(total, 1.0)
    hue = (hue + hue_delta / normalization) % 360.0
    saturation = np.clip(saturation + saturation_delta / normalization, 0.0, 1.0)
    lightness = np.clip(lightness + lightness_delta / normalization, 0.0, 1.0)
    return _hsl_to_rgb(hue, saturation, lightness)


def _apply_global_color(rgb: np.ndarray, p: Mapping[str, object]) -> np.ndarray:
    hue, saturation, lightness = _rgb_to_hsl(np.clip(rgb, 0.0, 1.0))
    hue = (hue + float(p["hue_shift"])) % 360.0
    saturation_scale = max(0.0, 1.0 + float(p["saturation"]) / 100.0)
    saturation = np.clip(saturation * saturation_scale, 0.0, 1.0)
    vibrance = float(p["vibrance"]) / 100.0
    skin_distance = np.abs((hue - 28.0 + 180.0) % 360.0 - 180.0)
    skin_protection = 1.0 - 0.65 * np.clip(1.0 - skin_distance / 35.0, 0.0, 1.0)
    saturation = np.clip(
        saturation + vibrance * (1.0 - saturation) * skin_protection,
        0.0,
        1.0,
    )
    return _hsl_to_rgb(hue, saturation, lightness)


def _gaussian(rgb: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(rgb, (0, 0), sigmaX=max(float(sigma), 0.05))


def _apply_texture(rgb: np.ndarray, p: Mapping[str, object]) -> np.ndarray:
    result = rgb.copy()
    dehaze = float(p["dehaze"]) / 100.0
    if abs(dehaze) > 1e-8:
        broad = _gaussian(result, max(3.0, min(result.shape[:2]) / 35.0))
        if dehaze > 0:
            result += 0.35 * dehaze * (result - broad)
            luma = _luma(result)
            result = luma[..., None] + (1.0 + 0.25 * dehaze) * (result - luma[..., None])
        else:
            atmospheric = np.mean(broad, axis=(0, 1), keepdims=True)
            result = result * (1.0 + dehaze * 0.30) - dehaze * 0.30 * atmospheric
    clarity = float(p["clarity"]) / 100.0
    if abs(clarity) > 1e-8:
        result += 0.45 * clarity * (result - _gaussian(result, 4.0))
    texture = float(p["texture"]) / 100.0
    if abs(texture) > 1e-8:
        result += 0.30 * texture * (result - _gaussian(result, 1.0))
    sharpening = float(p["sharpening"]) / 150.0
    if sharpening > 0:
        result += 0.55 * sharpening * (result - _gaussian(result, 0.65))
    return result


def _chromatic_aberration(rgb: np.ndarray, amount: float) -> np.ndarray:
    if abs(amount) < 1e-8:
        return rgb
    height, width = rgb.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
    dx = xx - cx
    dy = yy - cy
    radius2 = (dx * dx + dy * dy) / max(cx * cx + cy * cy, 1.0)
    scale = amount / 100.0 * 0.012
    result = rgb.copy()
    for channel, direction in ((0, 1.0), (2, -1.0)):
        map_x = xx + dx * radius2 * scale * direction
        map_y = yy + dy * radius2 * scale * direction
        result[..., channel] = cv2.remap(
            rgb[..., channel], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
    return result


def _apply_optical(rgb: np.ndarray, p: Mapping[str, object], frame_index: int) -> np.ndarray:
    result = _chromatic_aberration(
        rgb, float(p["chromatic_aberration"]["amount"])
    )
    diffusion = float(p["diffusion"]["strength"]) / 100.0
    if diffusion > 0:
        result = result * (1.0 - 0.55 * diffusion) + _gaussian(result, 2.5) * (0.55 * diffusion)
    linear = _srgb_to_linear(np.clip(result, 0, 1))
    luminance = _luma(linear)
    bloom = p["bloom"]
    bloom_amount = float(bloom["intensity"]) / 100.0
    if bloom_amount > 0:
        threshold = float(bloom["threshold"])
        weight = np.clip((luminance - threshold) / max(1.0 - threshold, 1e-4), 0, 1)
        bright = linear * weight[..., None]
        linear += _gaussian(
            bright, max(2.0, min(result.shape[:2]) / 80.0)
        ) * bloom_amount * 0.8
    halation = p["halation"]
    halation_amount = float(halation["intensity"]) / 100.0
    if halation_amount > 0:
        threshold = float(halation["threshold"])
        weight = np.clip((luminance - threshold) / max(1.0 - threshold, 1e-4), 0, 1)
        halo = _gaussian(weight, max(2.5, min(result.shape[:2]) / 65.0))
        linear += halo[..., None] * np.asarray(
            [0.14, 0.035, 0.008], np.float32
        ) * halation_amount
    result = _linear_to_srgb(linear)
    vignette = p["vignette"]
    amount = float(vignette["amount"]) / 100.0
    if abs(amount) > 1e-8:
        height, width = result.shape[:2]
        yy, xx = np.indices((height, width), dtype=np.float32)
        nx = (xx - (width - 1) * 0.5) / max(width * 0.5, 1.0)
        ny = (yy - (height - 1) * 0.5) / max(height * 0.5, 1.0)
        roundness = float(vignette["roundness"]) / 100.0
        nx *= 1.0 + max(0.0, -roundness) * 0.6
        ny *= 1.0 + max(0.0, roundness) * 0.6
        radius = np.sqrt(nx * nx + ny * ny)
        midpoint = 0.15 + 0.70 * float(vignette["midpoint"]) / 100.0
        feather = 0.05 + 0.70 * float(vignette["feather"]) / 100.0
        mask = _smoothstep(midpoint, midpoint + feather, radius)
        result *= (1.0 + amount * 0.65 * mask)[..., None]
    grain = p["grain"]
    grain_amount = float(grain["amount"]) / 100.0
    if grain_amount > 0:
        height, width = result.shape[:2]
        size = float(grain["size"])
        divisor = 1.0 + size / 18.0
        small_width = max(2, round(width / divisor))
        small_height = max(2, round(height / divisor))
        rng = np.random.default_rng(0x4B1D5FD + int(frame_index))
        noise = rng.normal(0.0, 1.0, (small_height, small_width)).astype(np.float32)
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        roughness = float(grain["roughness"]) / 100.0
        fine = rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
        noise = (1.0 - roughness) * noise + roughness * fine
        response = 0.45 + 0.55 * (1.0 - np.abs(2.0 * np.clip(_luma(result), 0, 1) - 1.0))
        result += noise[..., None] * response[..., None] * grain_amount * 0.055
    return result


@dataclass
class GradePoolExecutor:
    """Execute canonical Pool operations in a fixed professional order."""

    def apply_array(
        self,
        rgb: np.ndarray,
        operations: Sequence[object],
        *,
        frame_index: int,
        pre_grade_only: bool = False,
        post_grade_only: bool = False,
        masks: Mapping[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        active = []
        for order, operation_type in enumerate(POOL_PROCESSING_ORDER):
            for source_index, operation in enumerate(operations):
                if str(getattr(operation, "operation_type")) != operation_type:
                    continue
                start, end = getattr(operation, "frame_range")
                if int(start) <= frame_index <= int(end):
                    active.append((order, source_index, operation))
        result = np.asarray(rgb, dtype=np.float32)
        for _, _, operation in sorted(active, key=lambda item: (item[0], item[1])):
            operation_type = str(getattr(operation, "operation_type"))
            if pre_grade_only and operation_type != "denoise":
                continue
            if post_grade_only and operation_type == "denoise":
                continue
            before_operation = result.copy()
            p = getattr(operation, "parameters")
            track = getattr(operation, "parameter_track", ())
            if track:
                start = int(getattr(operation, "frame_range")[0])
                offset = min(max(frame_index - start, 0), len(track) - 1)
                p = track[offset]
            if operation_type == "denoise":
                result = _apply_denoise(result, p)
            elif operation_type == "white_balance":
                result = _apply_white_balance(result, p)
            elif operation_type == "primary":
                result = _apply_primary(result, p)
            elif operation_type == "color_wheels":
                result = _apply_color_wheels(result, p)
            elif operation_type == "curves":
                result = _apply_curves(result, p)
            elif operation_type == "hsl8":
                result = _apply_hsl8(result, p)
            elif operation_type == "global_color":
                result = _apply_global_color(result, p)
            elif operation_type == "texture":
                result = _apply_texture(result, p)
            elif operation_type == "optical_effects":
                result = _apply_optical(result, p, frame_index)
            result = np.clip(result, 0.0, 1.0)
            mask_id = str(getattr(operation, "mask_id", "global"))
            if mask_id != "global":
                if masks is None or mask_id not in masks:
                    raise ValueError(
                        f"Pool operation requires unavailable semantic mask: {mask_id}"
                    )
                mask = np.asarray(masks[mask_id], dtype=np.float32)
                if mask.shape != result.shape[:2]:
                    raise ValueError("Semantic mask and image dimensions do not match.")
                amount = np.clip(mask, 0.0, 1.0)[..., None]
                result = before_operation * (1.0 - amount) + result * amount
        return result

    def apply(
        self,
        image: Image.Image,
        operations: Sequence[object],
        *,
        frame_index: int,
        masks: Mapping[str, np.ndarray] | None = None,
    ) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        output = self.apply_array(
            rgb, operations, frame_index=frame_index, masks=masks
        )
        return Image.fromarray((output * 255.0 + 0.5).astype(np.uint8), mode="RGB")
