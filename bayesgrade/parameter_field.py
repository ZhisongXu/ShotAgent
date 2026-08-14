"""Bayesian temporal parameter fields and active Anchor acquisition.

This first prototype intentionally stays independent of video decoders, VLMs,
and retouching executors.  It implements the central research object from the
proposal: a posterior over an editable parameter trajectory and an analytic
integrated-variance-reduction acquisition function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Posterior:
    """Posterior of a multi-output grade trajectory with a shared kernel."""

    mean: Array
    covariance: Array

    @property
    def variance(self) -> Array:
        return np.maximum(np.diag(self.covariance), 0.0)

    @property
    def integrated_variance(self) -> float:
        return float(self.variance.sum())


class BayesianGradeField:
    """Gaussian-process parameter field conditioned on temporal video features.

    Parameters share the same temporal covariance but retain independent output
    values. This is sufficient for the initial active-selection experiments and
    keeps the acquisition function analytic.
    """

    def __init__(
        self,
        temporal_lengthscale: float = 0.18,
        feature_lengthscale: float = 1.0,
        signal_variance: float = 1.0,
        observation_noise: float = 1e-4,
        jitter: float = 1e-8,
    ) -> None:
        if temporal_lengthscale <= 0 or feature_lengthscale <= 0:
            raise ValueError("Kernel lengthscales must be positive.")
        if signal_variance <= 0 or observation_noise < 0 or jitter <= 0:
            raise ValueError("Kernel variances and jitter are invalid.")
        self.temporal_lengthscale = float(temporal_lengthscale)
        self.feature_lengthscale = float(feature_lengthscale)
        self.signal_variance = float(signal_variance)
        self.observation_noise = float(observation_noise)
        self.jitter = float(jitter)

    @staticmethod
    def _as_time_vector(times: Array) -> Array:
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        if times.size == 0:
            raise ValueError("At least one timestamp is required.")
        if not np.all(np.isfinite(times)):
            raise ValueError("Timestamps must be finite.")
        span = float(times.max() - times.min())
        if span > 0:
            times = (times - times.min()) / span
        else:
            times = np.zeros_like(times)
        return times

    @staticmethod
    def _standardize_features(features: Array, expected_length: int) -> Array:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features[:, None]
        if features.ndim != 2 or features.shape[0] != expected_length:
            raise ValueError("features must have shape [num_frames, feature_dim].")
        if not np.all(np.isfinite(features)):
            raise ValueError("Video features must be finite.")
        scale = features.std(axis=0, keepdims=True)
        scale[scale < 1e-8] = 1.0
        return (features - features.mean(axis=0, keepdims=True)) / scale

    def kernel(self, times: Array, features: Optional[Array] = None) -> Array:
        """Build a time-by-time prior covariance matrix."""

        t = self._as_time_vector(times)
        dt2 = (t[:, None] - t[None, :]) ** 2
        covariance = self.signal_variance * np.exp(
            -0.5 * dt2 / (self.temporal_lengthscale**2)
        )

        if features is not None:
            z = self._standardize_features(features, t.size)
            dz = z[:, None, :] - z[None, :, :]
            dz2 = np.sum(dz * dz, axis=-1)
            covariance *= np.exp(
                -0.5 * dz2 / (self.feature_lengthscale**2)
            )

        # Numerical symmetrization avoids tiny asymmetric eigenspectrum errors.
        return 0.5 * (covariance + covariance.T)

    def posterior(
        self,
        times: Array,
        anchor_indices: Sequence[int],
        anchor_values: Array,
        features: Optional[Array] = None,
        prior_mean: Optional[Array] = None,
        anchor_noise: Optional[Array] = None,
    ) -> Posterior:
        """Condition the parameter field on observed Anchor parameter values.

        Args:
            times: Frame timestamps, shape ``[T]``.
            anchor_indices: Unique observed frame indices.
            anchor_values: Observed parameters, shape ``[N]`` or ``[N, P]``.
            features: Optional per-frame appearance/motion features ``[T, D]``.
            prior_mean: Optional constant parameter mean ``[P]``.
            anchor_noise: Optional observation variance per Anchor ``[N]``.
        """

        t = self._as_time_vector(times)
        indices = np.asarray(anchor_indices, dtype=np.int64).reshape(-1)
        values = np.asarray(anchor_values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if indices.size == 0:
            raise ValueError("At least one Anchor is required for conditioning.")
        if values.ndim != 2 or values.shape[0] != indices.size:
            raise ValueError("anchor_values must have shape [num_anchors, num_params].")
        if len(np.unique(indices)) != indices.size:
            raise ValueError("Anchor indices must be unique.")
        if np.any(indices < 0) or np.any(indices >= t.size):
            raise IndexError("Anchor index is outside the video.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Anchor parameters must be finite.")

        parameter_dim = values.shape[1]
        if prior_mean is None:
            base = np.zeros(parameter_dim, dtype=np.float64)
        else:
            base = np.asarray(prior_mean, dtype=np.float64).reshape(-1)
            if base.size != parameter_dim:
                raise ValueError("prior_mean must match the parameter dimension.")

        covariance = self.kernel(t, features)
        k_aa = covariance[np.ix_(indices, indices)]
        if anchor_noise is None:
            noise = np.full(indices.size, self.observation_noise, dtype=np.float64)
        else:
            noise = np.asarray(anchor_noise, dtype=np.float64).reshape(-1)
            if noise.size != indices.size or np.any(noise < 0) or not np.all(np.isfinite(noise)):
                raise ValueError("anchor_noise must be finite, nonnegative, and match Anchors.")
        k_aa = k_aa + np.diag(noise + self.jitter)
        chol = np.linalg.cholesky(k_aa)

        centered = values - base[None, :]
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, centered))
        k_xa = covariance[:, indices]
        mean = base[None, :] + k_xa @ alpha

        solved = np.linalg.solve(chol, covariance[indices, :])
        post_covariance = covariance - solved.T @ solved
        post_covariance = 0.5 * (post_covariance + post_covariance.T)
        diagonal = np.maximum(np.diag(post_covariance), 0.0)
        np.fill_diagonal(post_covariance, diagonal)
        return Posterior(mean=mean, covariance=post_covariance)

    def acquisition_scores(
        self,
        posterior: Posterior,
        selected_indices: Iterable[int],
        candidate_costs: Optional[Array] = None,
        candidate_risks: Optional[Array] = None,
        cost_weight: float = 0.0,
        risk_weight: float = 0.0,
    ) -> Array:
        """Compute integrated posterior-variance reduction for every frame.

        For a hypothetical noisy observation at frame ``c``, the covariance
        reduction is ``K[:, c]^2 / (K[c, c] + noise)``. This quantity is label
        independent, so the next Anchor can be chosen without supervision.
        """

        covariance = np.asarray(posterior.covariance, dtype=np.float64)
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("posterior covariance must be square.")
        frame_count = covariance.shape[0]
        denominator = np.maximum(
            np.diag(covariance) + self.observation_noise + self.jitter,
            self.jitter,
        )
        scores = np.sum(covariance * covariance, axis=0) / denominator

        if candidate_costs is not None:
            costs = np.asarray(candidate_costs, dtype=np.float64).reshape(-1)
            if costs.size != frame_count:
                raise ValueError("candidate_costs must contain one value per frame.")
            scores = scores - float(cost_weight) * costs
        if candidate_risks is not None:
            risks = np.asarray(candidate_risks, dtype=np.float64).reshape(-1)
            if risks.size != frame_count:
                raise ValueError("candidate_risks must contain one value per frame.")
            scores = scores - float(risk_weight) * risks

        selected = np.asarray(list(selected_indices), dtype=np.int64)
        if selected.size:
            if np.any(selected < 0) or np.any(selected >= frame_count):
                raise IndexError("Selected Anchor index is outside the video.")
            scores[selected] = -np.inf
        return scores

    def select_next_anchor(
        self,
        posterior: Posterior,
        selected_indices: Iterable[int],
        **score_kwargs: object,
    ) -> tuple[int, float]:
        """Return the best unobserved frame and its acquisition value."""

        scores = self.acquisition_scores(
            posterior,
            selected_indices=selected_indices,
            **score_kwargs,
        )
        index = int(np.argmax(scores))
        if not np.isfinite(scores[index]):
            raise RuntimeError("No valid candidate Anchor remains.")
        return index, float(scores[index])
