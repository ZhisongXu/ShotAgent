"""Frozen VideoGradeBench-v1 engineering score definitions."""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence


def _unit(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def agent_primary_score(
    profile: str, metrics: Mapping[str, float]
) -> Optional[float]:
    """Return a transparent 0..1 score for an agent benchmark profile."""

    if profile == "intent_parameter":
        required = {
            "active_parameter_sign_accuracy",
            "normalized_parameter_mae",
            "relative_reference_improvement",
            "motion_compensated_edit_residual",
        }
        if not required.issubset(metrics):
            return None
        return float(
            0.35 * _unit(metrics["active_parameter_sign_accuracy"])
            + 0.25 * _unit(1.0 - metrics["normalized_parameter_mae"] / 0.10)
            + 0.25 * _unit(metrics["relative_reference_improvement"])
            + 0.15 * _unit(1.0 - metrics["motion_compensated_edit_residual"] / 0.02)
        )
    if profile == "paired_quality":
        required = {
            "reference_ssim",
            "relative_reference_improvement",
            "motion_compensated_temporal_reference_residual",
            "temporal_parameter_jerk",
        }
        if not required.issubset(metrics):
            return None
        return float(
            0.35 * _unit(metrics["reference_ssim"])
            + 0.35 * _unit(metrics["relative_reference_improvement"])
            + 0.15
            * _unit(
                1.0
                - metrics["motion_compensated_temporal_reference_residual"] / 0.02
            )
            + 0.15 * _unit(1.0 - metrics["temporal_parameter_jerk"] / 0.02)
        )
    return None


def geometric_mean(scores: Sequence[float]) -> Optional[float]:
    if not scores:
        return None
    bounded = [_unit(value) for value in scores]
    if any(value <= 0.0 for value in bounded):
        return 0.0
    return float(math.exp(sum(math.log(value) for value in bounded) / len(bounded)))
