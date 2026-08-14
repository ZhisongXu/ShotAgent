"""Interpretable parameter schema shared by image and video retouching."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


PARAMETER_NAMES = (
    "exposure",
    "temperature",
    "tint",
    "contrast",
    "highlights",
    "shadows",
    "saturation",
    "vibrance",
    "tone_curve",
    "local_exposure",
    "local_temperature",
    "local_saturation",
)

PARAMETER_LOWER_BOUNDS = np.array(
    [-3.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0, -1.0, -1.0],
    dtype=np.float64,
)
PARAMETER_UPPER_BOUNDS = np.array(
    [3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class RetouchParameters:
    exposure: float = 0.0
    temperature: float = 0.0
    tint: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    saturation: float = 0.0
    vibrance: float = 0.0
    tone_curve: float = 0.0
    local_exposure: float = 0.0
    local_temperature: float = 0.0
    local_saturation: float = 0.0

    def to_vector(self, dtype=np.float64) -> np.ndarray:
        return np.asarray([getattr(self, name) for name in PARAMETER_NAMES], dtype=dtype)

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}

    @classmethod
    def from_vector(cls, values: Sequence[float], clamp: bool = False) -> "RetouchParameters":
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size != len(PARAMETER_NAMES):
            raise ValueError(f"Expected {len(PARAMETER_NAMES)} retouch parameters.")
        if not np.all(np.isfinite(vector)):
            raise ValueError("Retouch parameters must be finite.")
        if clamp:
            vector = np.clip(vector, PARAMETER_LOWER_BOUNDS, PARAMETER_UPPER_BOUNDS)
        elif np.any(vector < PARAMETER_LOWER_BOUNDS) or np.any(vector > PARAMETER_UPPER_BOUNDS):
            raise ValueError("Retouch parameter is outside its valid range.")
        return cls(**dict(zip(PARAMETER_NAMES, vector.tolist())))

    @classmethod
    def from_mapping(cls, values: Mapping[str, float], clamp: bool = False) -> "RetouchParameters":
        unknown = set(values) - set(PARAMETER_NAMES)
        if unknown:
            raise ValueError(f"Unknown retouch parameters: {sorted(unknown)}")
        merged = cls().to_dict()
        merged.update({key: float(value) for key, value in values.items()})
        return cls.from_vector([merged[name] for name in PARAMETER_NAMES], clamp=clamp)
