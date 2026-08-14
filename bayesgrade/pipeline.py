"""End-to-end connection between AnchorRetouchAgent and BayesGrade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from retouch_agent import AnchorRetouchAgent, AnchorRetouchResult, RetouchExecutor
from retouch_agent.parameters import PARAMETER_LOWER_BOUNDS, PARAMETER_UPPER_BOUNDS, RetouchParameters
from retouch_agent.planner import image_statistics

from .parameter_field import BayesianGradeField, Posterior


@dataclass(frozen=True)
class BayesGradeRetouchResult:
    anchor_indices: tuple[int, ...]
    anchor_results: tuple[AnchorRetouchResult, ...]
    parameter_mean: np.ndarray
    parameter_variance: np.ndarray
    rendered_frames: tuple[Image.Image, ...]
    posterior: Posterior
    next_anchor: Optional[int]
    acquisition_score: Optional[float]


def extract_video_features(frames: Sequence[Image.Image]) -> np.ndarray:
    """Extract lightweight color-state features without a learned video model."""

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


class BayesGradeRetouchPipeline:
    """Retouch selected Anchors, infer a video field, and propose the next Anchor."""

    def __init__(
        self,
        anchor_agent: Optional[AnchorRetouchAgent] = None,
        parameter_field: Optional[BayesianGradeField] = None,
        executor: Optional[RetouchExecutor] = None,
    ) -> None:
        self.anchor_agent = anchor_agent or AnchorRetouchAgent()
        self.parameter_field = parameter_field or BayesianGradeField()
        self.executor = executor or RetouchExecutor()

    @staticmethod
    def _observation_noise(result: AnchorRetouchResult) -> float:
        diagonal = np.diag(result.parameter_covariance)
        # Local dimensions can be deterministically zero when no mask is present;
        # use active global dimensions to avoid a falsely noise-free observation.
        return float(np.clip(np.mean(diagonal[:9]), 1e-6, 0.25))

    def run(
        self,
        frames: Sequence[Image.Image],
        instruction: str,
        anchor_indices: Sequence[int],
        reference: Optional[Image.Image] = None,
        local_masks: Optional[Sequence[Optional[Image.Image]]] = None,
        times: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
    ) -> BayesGradeRetouchResult:
        if not frames:
            raise ValueError("At least one video frame is required.")
        frame_count = len(frames)
        anchors = np.asarray(anchor_indices, dtype=np.int64).reshape(-1)
        if anchors.size == 0 or len(np.unique(anchors)) != anchors.size:
            raise ValueError("At least one unique Anchor is required.")
        if np.any(anchors < 0) or np.any(anchors >= frame_count):
            raise IndexError("Anchor index is outside the video.")
        if local_masks is None:
            masks: list[Optional[Image.Image]] = [None] * frame_count
        else:
            masks = list(local_masks)
            if len(masks) != frame_count:
                raise ValueError("local_masks must contain one item per frame.")

        normalized_frames = [frame.convert("RGB") for frame in frames]
        anchor_results = tuple(
            self.anchor_agent.run(
                normalized_frames[index],
                instruction,
                reference=reference,
                local_mask=masks[index],
            )
            for index in anchors.tolist()
        )
        anchor_values = np.stack(
            [result.parameters.to_vector() for result in anchor_results], axis=0
        )
        anchor_noise = np.asarray(
            [self._observation_noise(result) for result in anchor_results],
            dtype=np.float64,
        )
        if times is None:
            times = np.linspace(0.0, 1.0, frame_count)
        if features is None:
            features = extract_video_features(normalized_frames)

        posterior = self.parameter_field.posterior(
            times,
            anchors,
            anchor_values,
            features=features,
            anchor_noise=anchor_noise,
        )
        parameter_mean = np.clip(
            posterior.mean,
            PARAMETER_LOWER_BOUNDS[None, :],
            PARAMETER_UPPER_BOUNDS[None, :],
        )
        rendered = tuple(
            self.executor.apply(
                frame,
                RetouchParameters.from_vector(parameter_mean[index]),
                mask=masks[index],
            )
            for index, frame in enumerate(normalized_frames)
        )

        next_anchor = acquisition_score = None
        if anchors.size < frame_count:
            next_anchor, acquisition_score = self.parameter_field.select_next_anchor(
                posterior, anchors.tolist()
            )
        return BayesGradeRetouchResult(
            anchor_indices=tuple(anchors.tolist()),
            anchor_results=anchor_results,
            parameter_mean=parameter_mean,
            parameter_variance=np.repeat(
                posterior.variance[:, None], parameter_mean.shape[1], axis=1
            ),
            rendered_frames=rendered,
            posterior=posterior,
            next_anchor=next_anchor,
            acquisition_score=acquisition_score,
        )
