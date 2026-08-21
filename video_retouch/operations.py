"""Deterministic executors for versioned video edit operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image


def _finite_float(value: object, field_name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field_name} must be finite.")
    return result


def _bounded_float(
    value: object,
    field_name: str,
    lower: float,
    upper: float,
) -> float:
    result = _finite_float(value, field_name)
    if not lower <= result <= upper:
        raise ValueError(f"{field_name} must be in [{lower}, {upper}].")
    return result


def canonicalize_operation_parameters(
    operation_type: str,
    parameters: object,
    *,
    lut_ids: Sequence[str] = (),
) -> dict[str, object]:
    """Validate untrusted model JSON and return a stable operation payload."""

    if not isinstance(parameters, dict):
        raise ValueError(f"{operation_type} parameters must be an object.")
    if operation_type == "tone_curve":
        unknown = set(parameters) - {"channel", "points", "strength"}
        if unknown:
            raise ValueError(f"Unknown tone_curve fields: {sorted(unknown)}")
        channel = str(parameters.get("channel", "rgb")).lower()
        if channel not in {"rgb", "red", "green", "blue"}:
            raise ValueError("tone_curve.channel must be rgb, red, green, or blue.")
        raw_points = parameters.get("points")
        if not isinstance(raw_points, list) or not 2 <= len(raw_points) <= 8:
            raise ValueError("tone_curve.points must contain 2 to 8 points.")
        points: list[list[float]] = []
        for index, point in enumerate(raw_points):
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f"tone_curve point {index} must be [input, output].")
            points.append(
                [
                    _bounded_float(point[0], f"tone_curve.points[{index}][0]", 0, 1),
                    _bounded_float(point[1], f"tone_curve.points[{index}][1]", 0, 1),
                ]
            )
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        if x_values[0] != 0.0 or x_values[-1] != 1.0:
            raise ValueError("tone_curve points must start at x=0 and end at x=1.")
        if any(right <= left for left, right in zip(x_values[:-1], x_values[1:])):
            raise ValueError(
                "tone_curve input coordinates must be strictly increasing."
            )
        if any(right < left for left, right in zip(y_values[:-1], y_values[1:])):
            raise ValueError("tone_curve output coordinates must be monotonic.")
        return {
            "channel": channel,
            "points": points,
            "strength": _bounded_float(
                parameters.get("strength", 1.0), "tone_curve.strength", 0, 1
            ),
        }

    if operation_type == "hsl_grade":
        allowed = {
            "hue_center",
            "hue_width",
            "hue_shift",
            "saturation",
            "lightness",
            "strength",
        }
        unknown = set(parameters) - allowed
        if unknown:
            raise ValueError(f"Unknown hsl_grade fields: {sorted(unknown)}")
        return {
            "hue_center": _bounded_float(
                parameters.get("hue_center", 0.0), "hsl_grade.hue_center", 0, 360
            ),
            "hue_width": _bounded_float(
                parameters.get("hue_width", 60.0), "hsl_grade.hue_width", 5, 180
            ),
            "hue_shift": _bounded_float(
                parameters.get("hue_shift", 0.0), "hsl_grade.hue_shift", -45, 45
            ),
            "saturation": _bounded_float(
                parameters.get("saturation", 0.0), "hsl_grade.saturation", -0.75, 0.75
            ),
            "lightness": _bounded_float(
                parameters.get("lightness", 0.0), "hsl_grade.lightness", -0.35, 0.35
            ),
            "strength": _bounded_float(
                parameters.get("strength", 1.0), "hsl_grade.strength", 0, 1
            ),
        }

    if operation_type == "lut":
        unknown = set(parameters) - {"lut_id", "strength"}
        if unknown:
            raise ValueError(f"Unknown lut fields: {sorted(unknown)}")
        lut_id = str(parameters.get("lut_id", ""))
        if not lut_id or lut_id not in set(lut_ids):
            raise ValueError("lut.lut_id must name a configured LUT catalog entry.")
        return {
            "lut_id": lut_id,
            "strength": _bounded_float(
                parameters.get("strength", 1.0), "lut.strength", 0, 1
            ),
        }

    raise ValueError(f"No deterministic executor for operation: {operation_type}")


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
    hue[green_max] = 60.0 * (
        (blue[green_max] - red[green_max]) / delta[green_max] + 2.0
    )
    hue[blue_max] = 60.0 * ((red[blue_max] - green[blue_max]) / delta[blue_max] + 4.0)
    return hue, saturation, lightness


def _hsl_to_rgb(
    hue: np.ndarray, saturation: np.ndarray, lightness: np.ndarray
) -> np.ndarray:
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


@dataclass(frozen=True)
class CubeLUT:
    size: int
    table: np.ndarray
    domain_min: np.ndarray
    domain_max: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "CubeLUT":
        size = None
        domain_min = np.zeros(3, dtype=np.float32)
        domain_max = np.ones(3, dtype=np.float32)
        rows: list[list[float]] = []
        for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("TITLE"):
                continue
            parts = line.split()
            if parts[0] == "LUT_3D_SIZE":
                size = int(parts[1])
            elif parts[0] == "DOMAIN_MIN":
                domain_min = np.asarray(parts[1:4], dtype=np.float32)
            elif parts[0] == "DOMAIN_MAX":
                domain_max = np.asarray(parts[1:4], dtype=np.float32)
            elif parts[0].startswith("LUT_"):
                raise ValueError("Only 3D .cube LUTs are supported.")
            else:
                if len(parts) != 3:
                    raise ValueError("Invalid RGB row in .cube LUT.")
                rows.append([float(value) for value in parts])
        if size is None or not 2 <= size <= 65:
            raise ValueError("LUT_3D_SIZE must be between 2 and 65.")
        if len(rows) != size**3:
            raise ValueError(".cube row count does not match LUT_3D_SIZE.")
        if np.any(domain_max <= domain_min):
            raise ValueError(".cube DOMAIN_MAX must exceed DOMAIN_MIN.")
        table = np.asarray(rows, dtype=np.float32).reshape(size, size, size, 3)
        if not np.all(np.isfinite(table)):
            raise ValueError(".cube values must be finite.")
        return cls(size, table, domain_min, domain_max)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        normalized = np.clip(
            (rgb - self.domain_min) / (self.domain_max - self.domain_min), 0, 1
        )
        coordinates = normalized * (self.size - 1)
        lower = np.floor(coordinates).astype(np.int64)
        upper = np.minimum(lower + 1, self.size - 1)
        amount = coordinates - lower
        red0, green0, blue0 = lower[..., 0], lower[..., 1], lower[..., 2]
        red1, green1, blue1 = upper[..., 0], upper[..., 1], upper[..., 2]
        red_amount = amount[..., 0:1]
        green_amount = amount[..., 1:2]
        blue_amount = amount[..., 2:3]

        def sample(red, green, blue):
            # .cube stores red fastest, then green, then blue.
            return self.table[blue, green, red]

        c00 = (
            sample(red0, green0, blue0) * (1 - red_amount)
            + sample(red1, green0, blue0) * red_amount
        )
        c01 = (
            sample(red0, green0, blue1) * (1 - red_amount)
            + sample(red1, green0, blue1) * red_amount
        )
        c10 = (
            sample(red0, green1, blue0) * (1 - red_amount)
            + sample(red1, green1, blue0) * red_amount
        )
        c11 = (
            sample(red0, green1, blue1) * (1 - red_amount)
            + sample(red1, green1, blue1) * red_amount
        )
        c0 = c00 * (1 - green_amount) + c10 * green_amount
        c1 = c01 * (1 - green_amount) + c11 * green_amount
        return c0 * (1 - blue_amount) + c1 * blue_amount


class OperationExecutor:
    """Apply validated post-grade operations without arbitrary code execution."""

    def __init__(self, lut_catalog: Mapping[str, Path] | None = None) -> None:
        self.lut_catalog = {
            str(name): Path(path).resolve()
            for name, path in (lut_catalog or {}).items()
        }
        self._lut_cache: dict[str, CubeLUT] = {}

    @property
    def lut_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.lut_catalog))

    def canonicalize(
        self, operation_type: str, parameters: object
    ) -> dict[str, object]:
        return canonicalize_operation_parameters(
            operation_type, parameters, lut_ids=self.lut_ids
        )

    @staticmethod
    def _tone_curve(rgb: np.ndarray, parameters: Mapping[str, object]) -> np.ndarray:
        points = np.asarray(parameters["points"], dtype=np.float32)
        strength = float(parameters["strength"])
        channel = str(parameters["channel"])
        result = rgb.copy()
        indices = (
            range(3)
            if channel == "rgb"
            else ({"red": 0, "green": 1, "blue": 2}[channel],)
        )
        for index in indices:
            mapped = np.interp(result[..., index], points[:, 0], points[:, 1])
            result[..., index] += strength * (mapped - result[..., index])
        return result

    @staticmethod
    def _hsl_grade(rgb: np.ndarray, parameters: Mapping[str, object]) -> np.ndarray:
        hue, saturation, lightness = _rgb_to_hsl(rgb)
        center = float(parameters["hue_center"])
        half_width = float(parameters["hue_width"]) * 0.5
        distance = np.abs((hue - center + 180.0) % 360.0 - 180.0)
        weight = np.clip(1.0 - distance / half_width, 0.0, 1.0)
        weight = weight * weight * (3.0 - 2.0 * weight)
        weight *= float(parameters["strength"])
        hue = (hue + weight * float(parameters["hue_shift"])) % 360.0
        saturation = np.clip(
            saturation + weight * float(parameters["saturation"]), 0.0, 1.0
        )
        lightness = np.clip(
            lightness + weight * float(parameters["lightness"]), 0.0, 1.0
        )
        return _hsl_to_rgb(hue, saturation, lightness)

    def _lut(self, rgb: np.ndarray, parameters: Mapping[str, object]) -> np.ndarray:
        lut_id = str(parameters["lut_id"])
        if lut_id not in self._lut_cache:
            self._lut_cache[lut_id] = CubeLUT.load(self.lut_catalog[lut_id])
        mapped = self._lut_cache[lut_id].apply(rgb)
        strength = float(parameters["strength"])
        return rgb + strength * (mapped - rgb)

    def apply(
        self,
        image: Image.Image,
        operations: Sequence[object],
        *,
        frame_index: int,
    ) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        for operation in operations:
            operation_type = str(getattr(operation, "operation_type"))
            if operation_type == "global_grade":
                continue
            start, end = getattr(operation, "frame_range")
            if not int(start) <= frame_index <= int(end):
                continue
            parameters = getattr(operation, "parameters")
            if operation_type == "tone_curve":
                rgb = self._tone_curve(rgb, parameters)
            elif operation_type == "hsl_grade":
                rgb = self._hsl_grade(rgb, parameters)
            elif operation_type == "lut":
                rgb = self._lut(rgb, parameters)
            else:
                raise ValueError(f"Unsupported post-grade operation: {operation_type}")
            rgb = np.clip(rgb, 0.0, 1.0)
        output = (rgb * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(output, mode="RGB")
