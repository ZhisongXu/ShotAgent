"""GP-guided Langevin posterior sampling in editable grade-parameter space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np


Array = np.ndarray
ExtraEnergy = Callable[[Array], tuple[float, Array]]


def linear_spline_basis(times: Array, control_count: int) -> Array:
    """Return a convex piecewise-linear basis with shape ``[T, K]``."""

    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if times.size < 2:
        raise ValueError("At least two timestamps are required.")
    if control_count < 2 or control_count > times.size:
        raise ValueError("control_count must lie in [2, num_frames].")
    span = float(times.max() - times.min())
    normalized = (times - times.min()) / span if span > 0 else np.zeros_like(times)
    knots = np.linspace(0.0, 1.0, control_count)
    spacing = 1.0 / (control_count - 1)
    basis = np.maximum(1.0 - np.abs(normalized[:, None] - knots[None, :]) / spacing, 0.0)
    row_sum = basis.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise RuntimeError("Failed to construct a valid temporal basis.")
    return basis / row_sum


@dataclass(frozen=True)
class LangevinSamples:
    """Posterior samples of spline controls and full video trajectories."""

    controls: Array
    trajectories: Array
    energies: Array

    @property
    def mean(self) -> Array:
        return self.trajectories.mean(axis=0)

    @property
    def variance(self) -> Array:
        return self.trajectories.var(axis=0, ddof=1)

    @property
    def frame_uncertainty(self) -> Array:
        return self.variance.mean(axis=-1)


class LangevinGradeSampler:
    """Sample a non-degenerate posterior over low-dimensional grade curves.

    Langevin iteration is inference time, not video time.  The sampler operates
    on spline controls while all energies are evaluated on the complete temporal
    trajectory. GP posterior mean and variance provide the prior and a diagonal
    preconditioner; additional differentiable energies can be plugged in without
    training a video model.
    """

    def __init__(
        self,
        control_count: int = 10,
        step_size: float = 5e-4,
        iterations: int = 400,
        temperature: float = 0.2,
        prior_weight: float = 0.25,
        prior_variance_floor: float = 3e-2,
        anchor_variance: float = 2.5e-3,
        smoothness_weight: float = 4.0,
        initial_noise: float = 0.03,
        precondition: bool = True,
    ) -> None:
        if control_count < 2 or iterations < 1:
            raise ValueError("control_count and iterations must be positive.")
        if step_size <= 0 or temperature < 0:
            raise ValueError("Invalid Langevin step size or temperature.")
        if prior_variance_floor <= 0 or anchor_variance <= 0:
            raise ValueError("Variances must be positive.")
        self.control_count = int(control_count)
        self.step_size = float(step_size)
        self.iterations = int(iterations)
        self.temperature = float(temperature)
        self.prior_weight = float(prior_weight)
        self.prior_variance_floor = float(prior_variance_floor)
        self.anchor_variance = float(anchor_variance)
        self.smoothness_weight = float(smoothness_weight)
        self.initial_noise = float(initial_noise)
        self.precondition = bool(precondition)

    @staticmethod
    def _expand_prior_variance(variance: Array, frame_count: int, parameter_dim: int) -> Array:
        variance = np.asarray(variance, dtype=np.float64)
        if variance.ndim == 1:
            variance = np.repeat(variance[:, None], parameter_dim, axis=1)
        if variance.shape != (frame_count, parameter_dim):
            raise ValueError("prior_variance must have shape [T] or [T, P].")
        return variance

    def _energy_and_gradient(
        self,
        controls: Array,
        basis: Array,
        prior_mean: Array,
        prior_variance: Array,
        anchor_indices: Array,
        anchor_values: Array,
        smoothness_weights: Array,
        extra_energy: Optional[ExtraEnergy],
    ) -> tuple[float, Array]:
        trajectory = basis @ controls
        difference = trajectory - prior_mean
        precision = 1.0 / (prior_variance + self.prior_variance_floor)
        energy = 0.5 * self.prior_weight * float(np.sum(difference * difference * precision))
        gradient = self.prior_weight * difference * precision

        anchor_difference = trajectory[anchor_indices] - anchor_values
        energy += 0.5 * float(np.sum(anchor_difference * anchor_difference)) / self.anchor_variance
        np.add.at(gradient, anchor_indices, anchor_difference / self.anchor_variance)

        second_difference = trajectory[2:] - 2.0 * trajectory[1:-1] + trajectory[:-2]
        weighted = smoothness_weights[:, None] * second_difference
        energy += 0.5 * self.smoothness_weight * float(
            np.sum(second_difference * weighted)
        )
        smooth_gradient = self.smoothness_weight * weighted
        gradient[:-2] += smooth_gradient
        gradient[1:-1] -= 2.0 * smooth_gradient
        gradient[2:] += smooth_gradient

        if extra_energy is not None:
            extra_value, extra_gradient = extra_energy(trajectory)
            extra_gradient = np.asarray(extra_gradient, dtype=np.float64)
            if extra_gradient.shape != trajectory.shape:
                raise ValueError("extra_energy gradient must match trajectory shape.")
            energy += float(extra_value)
            gradient += extra_gradient

        return energy, basis.T @ gradient

    def sample(
        self,
        times: Array,
        prior_mean: Array,
        prior_variance: Array,
        anchor_indices: Sequence[int],
        anchor_values: Array,
        num_samples: int = 16,
        smoothness_weights: Optional[Array] = None,
        parameter_bounds: Optional[tuple[Array, Array]] = None,
        extra_energy: Optional[ExtraEnergy] = None,
        seed: int = 0,
    ) -> LangevinSamples:
        """Run independent preconditioned Langevin chains and return final states."""

        times = np.asarray(times, dtype=np.float64).reshape(-1)
        prior_mean = np.asarray(prior_mean, dtype=np.float64)
        if prior_mean.ndim == 1:
            prior_mean = prior_mean[:, None]
        if prior_mean.ndim != 2 or prior_mean.shape[0] != times.size:
            raise ValueError("prior_mean must have shape [T, P].")
        frame_count, parameter_dim = prior_mean.shape
        if self.control_count > frame_count:
            raise ValueError("control_count cannot exceed the number of frames.")
        if num_samples < 2:
            raise ValueError("At least two samples are needed to estimate uncertainty.")

        anchors = np.asarray(anchor_indices, dtype=np.int64).reshape(-1)
        values = np.asarray(anchor_values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if anchors.size == 0 or values.shape != (anchors.size, parameter_dim):
            raise ValueError("Anchor indices and values have inconsistent shapes.")
        if len(np.unique(anchors)) != anchors.size:
            raise ValueError("Anchor indices must be unique.")
        if np.any(anchors < 0) or np.any(anchors >= frame_count):
            raise IndexError("Anchor index is outside the video.")

        variance = self._expand_prior_variance(
            prior_variance, frame_count, parameter_dim
        )
        basis = linear_spline_basis(times, self.control_count)
        initial_controls = np.linalg.lstsq(basis, prior_mean, rcond=None)[0]

        if smoothness_weights is None:
            weights = np.ones(frame_count - 2, dtype=np.float64)
        else:
            weights = np.asarray(smoothness_weights, dtype=np.float64).reshape(-1)
            if weights.size != frame_count - 2 or np.any(weights < 0):
                raise ValueError("smoothness_weights must be nonnegative with length T-2.")

        projected_variance = (basis.T @ variance) / np.maximum(
            basis.sum(axis=0)[:, None], 1e-8
        )
        if self.precondition:
            preconditioner = np.clip(
                projected_variance + self.prior_variance_floor,
                2e-2,
                1.0,
            )
        else:
            preconditioner = np.ones_like(projected_variance)

        lower = upper = None
        if parameter_bounds is not None:
            lower = np.asarray(parameter_bounds[0], dtype=np.float64).reshape(-1)
            upper = np.asarray(parameter_bounds[1], dtype=np.float64).reshape(-1)
            if lower.size != parameter_dim or upper.size != parameter_dim:
                raise ValueError("Parameter bounds must have one value per parameter.")
            if np.any(lower >= upper):
                raise ValueError("Every lower parameter bound must be below its upper bound.")

        rng = np.random.default_rng(seed)
        controls_out = np.empty(
            (num_samples, self.control_count, parameter_dim), dtype=np.float64
        )
        trajectories = np.empty(
            (num_samples, frame_count, parameter_dim), dtype=np.float64
        )
        energies = np.empty(num_samples, dtype=np.float64)

        for sample_index in range(num_samples):
            controls = initial_controls + self.initial_noise * np.sqrt(
                preconditioner
            ) * rng.standard_normal(initial_controls.shape)
            if lower is not None:
                controls = np.clip(controls, lower[None, :], upper[None, :])

            for _ in range(self.iterations):
                _, gradient = self._energy_and_gradient(
                    controls,
                    basis,
                    prior_mean,
                    variance,
                    anchors,
                    values,
                    weights,
                    extra_energy,
                )
                noise = rng.standard_normal(controls.shape)
                controls = (
                    controls
                    - 0.5 * self.step_size * preconditioner * gradient
                    + np.sqrt(
                        self.step_size * self.temperature * preconditioner
                    )
                    * noise
                )
                if lower is not None:
                    controls = np.clip(
                        controls, lower[None, :], upper[None, :]
                    )

            final_energy, _ = self._energy_and_gradient(
                controls,
                basis,
                prior_mean,
                variance,
                anchors,
                values,
                weights,
                extra_energy,
            )
            controls_out[sample_index] = controls
            trajectories[sample_index] = basis @ controls
            energies[sample_index] = final_energy

        return LangevinSamples(
            controls=controls_out,
            trajectories=trajectories,
            energies=energies,
        )


def disagreement_acquisition_scores(
    samples: LangevinSamples,
    selected_indices: Iterable[int],
    observation_noise: float = 1e-4,
) -> Array:
    """Estimate information value using empirical cross-frame covariance."""

    trajectories = np.asarray(samples.trajectories, dtype=np.float64)
    if trajectories.ndim != 3 or trajectories.shape[0] < 2:
        raise ValueError("Expected at least two trajectory samples [S, T, P].")
    sample_count, frame_count, parameter_dim = trajectories.shape
    scores = np.zeros(frame_count, dtype=np.float64)
    for parameter_index in range(parameter_dim):
        values = trajectories[:, :, parameter_index]
        centered = values - values.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered / (sample_count - 1)
        denominator = np.maximum(
            np.diag(covariance) + observation_noise,
            1e-12,
        )
        scores += np.sum(covariance * covariance, axis=0) / denominator

    selected = np.asarray(list(selected_indices), dtype=np.int64)
    if selected.size:
        if np.any(selected < 0) or np.any(selected >= frame_count):
            raise IndexError("Selected Anchor index is outside the video.")
        scores[selected] = -np.inf
    return scores


def select_disagreement_anchor(
    samples: LangevinSamples,
    selected_indices: Iterable[int],
    observation_noise: float = 1e-4,
) -> tuple[int, float]:
    scores = disagreement_acquisition_scores(
        samples,
        selected_indices=selected_indices,
        observation_noise=observation_noise,
    )
    index = int(np.argmax(scores))
    if not np.isfinite(scores[index]):
        raise RuntimeError("No valid candidate Anchor remains.")
    return index, float(scores[index])


def hybrid_acquisition_scores(
    gp_scores: Array,
    samples: LangevinSamples,
    selected_indices: Iterable[int],
    disagreement_weight: float = 0.35,
    observation_noise: float = 1e-4,
) -> Array:
    """Fuse calibrated GP variance reduction with non-Gaussian disagreement.

    The GP term is the reliable linear-Gaussian baseline. Langevin disagreement
    is treated as an additional signal for constraints not represented by the GP
    kernel. Both terms are normalized over valid candidates before fusion.
    """

    if not 0.0 <= disagreement_weight <= 1.0:
        raise ValueError("disagreement_weight must lie in [0, 1].")
    gp_scores = np.asarray(gp_scores, dtype=np.float64).reshape(-1)
    ld_scores = disagreement_acquisition_scores(
        samples,
        selected_indices=selected_indices,
        observation_noise=observation_noise,
    )
    if gp_scores.size != ld_scores.size:
        raise ValueError("GP and Langevin scores must cover the same frames.")

    selected = np.asarray(list(selected_indices), dtype=np.int64)
    valid = np.ones(gp_scores.size, dtype=bool)
    valid[selected] = False

    def normalize(values: Array) -> Array:
        result = np.zeros_like(values)
        finite = valid & np.isfinite(values)
        if np.any(finite):
            shifted = values[finite] - np.min(values[finite])
            scale = float(np.max(shifted))
            result[finite] = shifted / scale if scale > 1e-12 else 0.0
        return result

    combined = (
        (1.0 - disagreement_weight) * normalize(gp_scores)
        + disagreement_weight * normalize(ld_scores)
    )
    combined[selected] = -np.inf
    return combined
