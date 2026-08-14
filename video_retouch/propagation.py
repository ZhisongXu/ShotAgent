"""Shot-local Bayesian diffusion of editable grade parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from bayesgrade.parameter_field import BayesianGradeField
from retouch_agent.parameters import PARAMETER_LOWER_BOUNDS, PARAMETER_UPPER_BOUNDS
from retouch_agent.planner import image_statistics

from .backends import AnchorGrade
from .color_science import (
    SourceGuidedTonalStabilizer,
    spatiotemporal_palette_features,
)
from .models import ShotPlan


@dataclass(frozen=True)
class DiffusedGrade:
    base_parameters: np.ndarray
    frame_parameters: np.ndarray
    frame_uncertainty: np.ndarray
    keyframes: dict[int, np.ndarray]
    stabilization: dict[str, object]


def appearance_features(frames: Sequence[Image.Image]) -> np.ndarray:
    rows = []
    for frame in frames:
        stats = image_statistics(frame)
        rows.append(
            [
                stats["luminance"],
                stats["contrast"],
                stats["saturation"],
                stats["warmth"],
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def grade_field_features(frames: Sequence[Image.Image]) -> np.ndarray:
    """Combine stable public appearance metrics with dynamic palette traces."""

    global_features = appearance_features(frames)
    palette_features = spatiotemporal_palette_features(frames)
    return np.concatenate([global_features, palette_features], axis=1)


class BayesianParameterDiffuser:
    """Infer shot base grade plus smooth frame-level corrections."""

    def __init__(
        self,
        field: BayesianGradeField | None = None,
        stabilizer: SourceGuidedTonalStabilizer | None = None,
    ) -> None:
        self.field = field or BayesianGradeField(
            temporal_lengthscale=0.28,
            feature_lengthscale=1.5,
            observation_noise=2e-3,
        )
        self.stabilizer = stabilizer or SourceGuidedTonalStabilizer()

    def diffuse(
        self,
        frames: Sequence[Image.Image],
        shot: ShotPlan,
        anchor_grades: Sequence[AnchorGrade],
    ) -> DiffusedGrade:
        shot_frames = frames[shot.start_frame : shot.end_frame + 1]
        if not shot_frames:
            raise ValueError("Shot contains no frames.")
        if not anchor_grades:
            raise ValueError("At least one Anchor grade is required.")
        anchor_indices = np.asarray(
            [grade.frame_index - shot.start_frame for grade in anchor_grades],
            dtype=np.int64,
        )
        anchor_values = np.stack(
            [grade.parameters.to_vector() for grade in anchor_grades], axis=0
        )
        if len(np.unique(anchor_indices)) != anchor_indices.size:
            raise ValueError("Anchor grades must have unique frame indices.")
        base = np.median(anchor_values, axis=0)
        times = np.arange(len(shot_frames), dtype=np.float64)
        posterior = self.field.posterior(
            times,
            anchor_indices,
            anchor_values,
            features=grade_field_features(shot_frames),
            prior_mean=base,
            anchor_noise=np.asarray(
                [
                    max(1e-4, 0.01 * (1.0 - min(grade.score, 0.0)))
                    for grade in anchor_grades
                ],
                dtype=np.float64,
            ),
        )
        raw_trajectory = np.clip(
            posterior.mean,
            PARAMETER_LOWER_BOUNDS[None, :],
            PARAMETER_UPPER_BOUNDS[None, :],
        )
        trajectory, stabilization = self.stabilizer.stabilize(
            shot_frames,
            raw_trajectory,
            anchor_indices,
            anchor_values,
        )
        keyframes = {
            grade.frame_index: grade.parameters.to_vector() for grade in anchor_grades
        }
        return DiffusedGrade(
            base_parameters=base,
            frame_parameters=trajectory,
            frame_uncertainty=posterior.variance,
            keyframes=keyframes,
            stabilization=stabilization,
        )
