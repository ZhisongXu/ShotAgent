"""Type-aware temporal propagation for Grade Pool operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from bayesgrade.parameter_field import BayesianGradeField

from .grade_pools import canonicalize_pool_parameters
from .models import ShotPlan
from .propagation import grade_field_features


DYNAMIC_POOL_FIELDS = {
    "primary": (
        "exposure",
        "contrast",
        "highlights",
        "shadows",
        "whites",
        "blacks",
        "gamma",
    ),
    "white_balance": ("temperature", "tint"),
    "denoise": ("luminance", "color"),
}

POOL_TEMPORAL_POLICIES = {
    "primary": "bayesian_source_guided",
    "white_balance": "bayesian",
    "denoise": "slow_bayesian",
    "global_color": "shot_static",
    "hsl8": "shot_static",
    "color_wheels": "shot_static",
    "curves": "shot_static",
    "texture": "shot_static",
    "optical_effects": "shot_static_seeded_grain",
}


def _average_payload(values: Sequence[object]) -> object:
    first = values[0]
    if isinstance(first, Mapping):
        keys = first.keys()
        return {
            str(key): _average_payload([value[key] for value in values])
            for key in keys
        }
    if isinstance(first, list):
        array = np.asarray(values, dtype=np.float64)
        return np.mean(array, axis=0).tolist()
    if isinstance(first, (int, float)):
        return float(np.mean(np.asarray(values, dtype=np.float64)))
    return first


def static_pool_consensus(
    operation_type: str,
    anchor_parameters: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Average canonical static nodes from multiple grading Anchors."""

    if not anchor_parameters:
        return canonicalize_pool_parameters(operation_type, {})
    result = _average_payload(anchor_parameters)
    assert isinstance(result, dict)
    return result


class PoolParameterDiffuser:
    """Diffuse only controls whose semantics permit frame-level changes."""

    def __init__(self) -> None:
        self.field = BayesianGradeField(
            temporal_lengthscale=0.28,
            feature_lengthscale=1.5,
            observation_noise=2e-3,
        )

    @staticmethod
    def _source_exposure(frames: Sequence[Image.Image]) -> np.ndarray:
        values = []
        for frame in frames:
            image = frame.convert("RGB")
            scale = min(1.0, 96.0 / max(image.size))
            if scale < 1.0:
                image = image.resize(
                    (
                        max(1, round(image.width * scale)),
                        max(1, round(image.height * scale)),
                    )
                )
            rgb = np.asarray(image, dtype=np.float64) / 255.0
            linear = np.where(
                rgb <= 0.04045,
                rgb / 12.92,
                np.power((rgb + 0.055) / 1.055, 2.4),
            )
            luma = np.sum(linear * np.asarray([0.2126, 0.7152, 0.0722]), axis=2)
            values.append(np.log2(np.median(luma) + 1e-4))
        return np.asarray(values, dtype=np.float64)

    def diffuse(
        self,
        frames: Sequence[Image.Image],
        shot: ShotPlan,
        operation_type: str,
        anchors: Sequence[tuple[int, dict[str, object]]],
    ) -> tuple[dict[str, object], ...]:
        if operation_type not in DYNAMIC_POOL_FIELDS:
            raise ValueError(f"Pool {operation_type} is not frame-diffusible.")
        if not anchors:
            raise ValueError("At least one Pool Anchor is required.")
        shot_frames = frames[shot.start_frame : shot.end_frame + 1]
        fields = DYNAMIC_POOL_FIELDS[operation_type]
        anchor_indices = np.asarray(
            [frame - shot.start_frame for frame, _ in anchors], dtype=np.int64
        )
        anchor_values = np.asarray(
            [[float(parameters[field]) for field in fields] for _, parameters in anchors],
            dtype=np.float64,
        )
        base = np.median(anchor_values, axis=0)
        posterior = self.field.posterior(
            np.arange(len(shot_frames), dtype=np.float64),
            anchor_indices,
            anchor_values,
            features=grade_field_features(shot_frames),
            prior_mean=base,
            anchor_noise=np.full(len(anchors), 0.005, dtype=np.float64),
        )
        trajectory = posterior.mean

        if operation_type == "primary" and len(shot_frames) >= 3:
            exposure = self._source_exposure(shot_frames)
            sigma = float(np.clip(len(shot_frames) / 24.0, 1.0, 8.0))
            trend = gaussian_filter1d(exposure, sigma=sigma, mode="nearest")
            compensation = np.clip(-0.75 * (exposure - trend), -0.6, 0.6)
            trajectory[:, 0] += compensation

        if operation_type == "denoise" and len(shot_frames) >= 3:
            trajectory = gaussian_filter1d(trajectory, sigma=1.5, axis=0, mode="nearest")

        trajectory[anchor_indices] = anchor_values
        result = []
        for row in trajectory:
            payload = {field: float(value) for field, value in zip(fields, row)}
            result.append(canonicalize_pool_parameters(operation_type, payload))
        return tuple(result)
