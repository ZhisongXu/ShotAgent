"""High-bit-depth RGB color management with optional OpenColorIO support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


Array = np.ndarray


def _srgb_decode(value: Array) -> Array:
    return np.where(
        value <= 0.04045,
        value / 12.92,
        np.power(np.maximum((value + 0.055) / 1.055, 0.0), 2.4),
    )


def _srgb_encode(value: Array) -> Array:
    value = np.maximum(value, 0.0)
    return np.where(
        value <= 0.0031308,
        value * 12.92,
        1.055 * np.power(value, 1.0 / 2.4) - 0.055,
    )


def _rec709_decode(value: Array) -> Array:
    return np.where(
        value < 0.081,
        value / 4.5,
        np.power(np.maximum((value + 0.099) / 1.099, 0.0), 1.0 / 0.45),
    )


def _rec709_encode(value: Array) -> Array:
    value = np.maximum(value, 0.0)
    return np.where(
        value < 0.018,
        4.5 * value,
        1.099 * np.power(value, 0.45) - 0.099,
    )


def _logc3_decode(value: Array) -> Array:
    cut = 0.010591
    a, b, c, d, e, f = 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809
    boundary = e * cut + f
    return np.where(
        value > boundary,
        (np.power(10.0, (value - d) / c) - b) / a,
        (value - f) / e,
    )


def _logc3_encode(value: Array) -> Array:
    cut = 0.010591
    a, b, c, d, e, f = 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809
    return np.where(
        value > cut,
        c * np.log10(np.maximum(a * value + b, 1e-10)) + d,
        e * value + f,
    )


def _slog3_decode(value: Array) -> Array:
    code = value * 1023.0
    boundary = 171.2102946929
    return np.where(
        code >= boundary,
        np.power(10.0, (code - 420.0) / 261.5) * 0.19 - 0.01,
        (code - 95.0) * 0.01125 / (boundary - 95.0),
    )


def _slog3_encode(value: Array) -> Array:
    return np.where(
        value >= 0.01125,
        (420.0 + np.log10(np.maximum((value + 0.01) / 0.19, 1e-10)) * 261.5)
        / 1023.0,
        (value * (171.2102946929 - 95.0) / 0.01125 + 95.0) / 1023.0,
    )


def _vlog_decode(value: Array) -> Array:
    return np.where(
        value < 0.181,
        (value - 0.125) / 5.6,
        np.power(10.0, (value - 0.598206) / 0.241514) - 0.00873,
    )


def _vlog_encode(value: Array) -> Array:
    return np.where(
        value < 0.01,
        5.6 * value + 0.125,
        0.241514 * np.log10(np.maximum(value + 0.00873, 1e-10)) + 0.598206,
    )


def _pq_decode(value: Array) -> Array:
    m1, m2 = 2610.0 / 16384.0, 2523.0 / 32.0
    c1, c2, c3 = 3424.0 / 4096.0, 2413.0 / 128.0, 2392.0 / 128.0
    power = np.power(np.clip(value, 0.0, 1.0), 1.0 / m2)
    return np.power(
        np.maximum(power - c1, 0.0) / np.maximum(c2 - c3 * power, 1e-10),
        1.0 / m1,
    )


def _pq_encode(value: Array) -> Array:
    m1, m2 = 2610.0 / 16384.0, 2523.0 / 32.0
    c1, c2, c3 = 3424.0 / 4096.0, 2413.0 / 128.0, 2392.0 / 128.0
    power = np.power(np.maximum(value, 0.0), m1)
    return np.power((c1 + c2 * power) / (1.0 + c3 * power), m2)


def _hlg_decode(value: Array) -> Array:
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    return np.where(
        value <= 0.5,
        value * value / 3.0,
        (np.exp((value - c) / a) + b) / 12.0,
    )


def _hlg_encode(value: Array) -> Array:
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    return np.where(
        value <= 1.0 / 12.0,
        np.sqrt(np.maximum(3.0 * value, 0.0)),
        a * np.log(np.maximum(12.0 * value - b, 1e-10)) + c,
    )


IDENTITY = lambda value: value


@dataclass(frozen=True)
class RGBSpace:
    name: str
    rgb_to_xyz: Array
    white_xy: tuple[float, float]
    decode: Callable[[Array], Array]
    encode: Callable[[Array], Array]


REC709_TO_XYZ = np.asarray(
    [
        [0.4123907993, 0.3575843394, 0.1804807884],
        [0.2126390059, 0.7151686788, 0.0721923154],
        [0.0193308187, 0.1191947798, 0.9505321522],
    ],
    dtype=np.float64,
)
REC2020_TO_XYZ = np.asarray(
    [
        [0.6369580483, 0.1446169036, 0.1688809752],
        [0.2627002120, 0.6779980715, 0.0593017165],
        [0.0000000000, 0.0280726930, 1.0609850577],
    ],
    dtype=np.float64,
)
AP1_TO_XYZ = np.asarray(
    [
        [0.6624541811, 0.1340042065, 0.1561876870],
        [0.2722287168, 0.6740817658, 0.0536895174],
        [-0.0055746495, 0.0040607335, 1.0103391003],
    ],
    dtype=np.float64,
)
AP0_TO_XYZ = np.asarray(
    [
        [0.9525523959, 0.0000000000, 0.0000936786],
        [0.3439664498, 0.7281660966, -0.0721325464],
        [0.0000000000, 0.0000000000, 1.0088251844],
    ],
    dtype=np.float64,
)

D65 = (0.3127, 0.3290)
D60 = (0.32168, 0.33767)


COLOR_SPACES = {
    "srgb": RGBSpace("srgb", REC709_TO_XYZ, D65, _srgb_decode, _srgb_encode),
    "rec709": RGBSpace("rec709", REC709_TO_XYZ, D65, _rec709_decode, _rec709_encode),
    "linear_rec709": RGBSpace("linear_rec709", REC709_TO_XYZ, D65, IDENTITY, IDENTITY),
    "logc3": RGBSpace("logc3", REC709_TO_XYZ, D65, _logc3_decode, _logc3_encode),
    "slog3": RGBSpace("slog3", REC709_TO_XYZ, D65, _slog3_decode, _slog3_encode),
    "vlog": RGBSpace("vlog", REC709_TO_XYZ, D65, _vlog_decode, _vlog_encode),
    "rec2020_pq": RGBSpace("rec2020_pq", REC2020_TO_XYZ, D65, _pq_decode, _pq_encode),
    "rec2020_hlg": RGBSpace("rec2020_hlg", REC2020_TO_XYZ, D65, _hlg_decode, _hlg_encode),
    "linear_rec2020": RGBSpace("linear_rec2020", REC2020_TO_XYZ, D65, IDENTITY, IDENTITY),
    "acescg": RGBSpace("acescg", AP1_TO_XYZ, D60, IDENTITY, IDENTITY),
    "aces2065-1": RGBSpace("aces2065-1", AP0_TO_XYZ, D60, IDENTITY, IDENTITY),
}


def _xy_to_xyz(xy: tuple[float, float]) -> Array:
    x_value, y_value = xy
    return np.asarray([x_value / y_value, 1.0, (1.0 - x_value - y_value) / y_value])


def _chromatic_adaptation(source: tuple[float, float], target: tuple[float, float]) -> Array:
    if source == target:
        return np.eye(3, dtype=np.float64)
    bradford = np.asarray(
        [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]],
        dtype=np.float64,
    )
    source_cone = bradford @ _xy_to_xyz(source)
    target_cone = bradford @ _xy_to_xyz(target)
    return np.linalg.inv(bradford) @ np.diag(target_cone / source_cone) @ bradford


def aces_fitted_tonemap(linear: Array) -> Array:
    """A bounded ACES-inspired display fit for proxy generation."""

    value = np.maximum(linear, 0.0)
    return np.clip(
        value * (2.51 * value + 0.03) / (value * (2.43 * value + 0.59) + 0.14),
        0.0,
        1.0,
    )


class ColorManager:
    """Convert RGB arrays through a linear XYZ connection space."""

    def __init__(self, working_space: str = "acescg") -> None:
        if working_space not in COLOR_SPACES:
            raise ValueError(f"Unknown working color space: {working_space}")
        self.working_space = working_space

    @staticmethod
    def convert(
        rgb: Array,
        source_space: str,
        target_space: str,
        *,
        tone_map: bool = False,
    ) -> Array:
        if source_space not in COLOR_SPACES or target_space not in COLOR_SPACES:
            raise ValueError(
                f"Unsupported color conversion: {source_space} -> {target_space}"
            )
        source = COLOR_SPACES[source_space]
        target = COLOR_SPACES[target_space]
        values = np.asarray(rgb, dtype=np.float32)
        linear_source = source.decode(values.astype(np.float64))
        xyz = linear_source @ source.rgb_to_xyz.T
        xyz = xyz @ _chromatic_adaptation(source.white_xy, target.white_xy).T
        linear_target = xyz @ np.linalg.inv(target.rgb_to_xyz).T
        if tone_map:
            linear_target = aces_fitted_tonemap(linear_target)
        return target.encode(linear_target).astype(np.float32)

    def to_working(self, rgb: Array, source_space: str) -> Array:
        return self.convert(rgb, source_space, self.working_space)

    def from_working(
        self, rgb: Array, target_space: str, *, tone_map: bool = False
    ) -> Array:
        return self.convert(rgb, self.working_space, target_space, tone_map=tone_map)

    def display_proxy(self, working_rgb: Array) -> Array:
        return np.clip(
            self.from_working(working_rgb, "srgb", tone_map=True), 0.0, 1.0
        )


class OCIOColorManager:
    """OpenColorIO adapter for studio configs, including official ACES configs."""

    def __init__(self, config_path: Path | None = None) -> None:
        try:
            import PyOpenColorIO as ocio
        except ImportError as error:
            raise RuntimeError(
                "OpenColorIO support requires the optional 'opencolorio' package."
            ) from error
        self.ocio = ocio
        self.config = (
            ocio.Config.CreateFromFile(str(Path(config_path).resolve()))
            if config_path is not None
            else ocio.GetCurrentConfig()
        )

    def convert(self, rgb: Array, source_space: str, target_space: str) -> Array:
        values = np.ascontiguousarray(rgb, dtype=np.float32).copy()
        if values.ndim < 1 or values.shape[-1] != 3:
            raise ValueError("OCIO RGB arrays must end in a three-channel dimension.")
        processor = self.config.getProcessor(source_space, target_space).getDefaultCPUProcessor()
        pixel_count = int(values.size // 3)
        reshaped = values.reshape(1, pixel_count, 3)
        descriptor = self.ocio.PackedImageDesc(reshaped, pixel_count, 1, 3)
        processor.apply(descriptor)
        return values
