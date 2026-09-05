"""Classical color-science priors adapted to an editable video grade graph.

The routines in this module are deliberately parameter-domain building blocks.
They do not post-process the committed RGB video, so a result remains an
editable 12-D grade trajectory and can still be rejected or rolled back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import spsolve

from retouch_agent.parameters import (
    PARAMETER_LOWER_BOUNDS,
    PARAMETER_UPPER_BOUNDS,
)


def _psd_power(matrix: np.ndarray, power: float, epsilon: float) -> np.ndarray:
    """Return a stable power of a real symmetric positive-semidefinite matrix."""

    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, epsilon)
    return (vectors * np.power(values, power)[None, :]) @ vectors.T


def _rgb_to_normalized_lab(array: np.ndarray) -> np.ndarray:
    rgb = np.asarray(array, dtype=np.float32)
    if rgb.max(initial=0.0) > 1.0:
        rgb = rgb / 255.0
    lab = cv2.cvtColor(np.clip(rgb, 0.0, 1.0), cv2.COLOR_RGB2LAB)
    lab[..., 0] /= 100.0
    lab[..., 1:] /= 128.0
    return lab.astype(np.float64)


def _normalized_lab_to_rgb(lab: np.ndarray) -> tuple[np.ndarray, float]:
    cv_lab = np.asarray(lab, dtype=np.float32).copy()
    cv_lab[..., 0] *= 100.0
    cv_lab[..., 1:] *= 128.0
    rgb = cv2.cvtColor(cv_lab, cv2.COLOR_LAB2RGB)
    clipped = np.logical_or(rgb < 0.0, rgb > 1.0)
    return np.clip(rgb, 0.0, 1.0), float(np.mean(clipped))


def _resized_rgb(image: Image.Image, max_side: int) -> np.ndarray:
    source = image.convert("RGB")
    scale = min(1.0, float(max_side) / max(source.size))
    size = (
        max(1, round(source.width * scale)),
        max(1, round(source.height * scale)),
    )
    return np.asarray(source.resize(size, Image.Resampling.LANCZOS))


@dataclass(frozen=True)
class MongeKantorovichFit:
    source_mean: np.ndarray
    reference_mean: np.ndarray
    matrix: np.ndarray
    source_covariance: np.ndarray
    reference_covariance: np.ndarray


class LinearMongeKantorovichMatcher:
    """Pitie-style linear Monge--Kantorovich color-distribution transport.

    This is an independent implementation of the published closed-form map.
    It accepts masks so the same primitive can later be applied to MLLM-selected
    and tracked semantic regions instead of indiscriminate full-frame colors.
    """

    def __init__(
        self,
        strength: float = 0.35,
        covariance_epsilon: float = 1e-4,
        max_samples: int = 20_000,
        analysis_max_side: int = 320,
    ) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("MKL strength must be in [0, 1].")
        if covariance_epsilon <= 0.0 or max_samples < 16:
            raise ValueError("Invalid MKL numerical settings.")
        self.strength = float(strength)
        self.covariance_epsilon = float(covariance_epsilon)
        self.max_samples = int(max_samples)
        self.analysis_max_side = int(analysis_max_side)

    def _pixels(
        self,
        image: Image.Image,
        mask: Optional[Image.Image | np.ndarray],
    ) -> np.ndarray:
        rgb = _resized_rgb(image, self.analysis_max_side)
        lab = _rgb_to_normalized_lab(rgb).reshape(-1, 3)
        if mask is not None:
            if isinstance(mask, Image.Image):
                mask_image = mask.convert("L")
            else:
                values = np.asarray(mask)
                if values.dtype != np.uint8:
                    values = (np.clip(values, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
                mask_image = Image.fromarray(values, mode="L")
            keep = (
                np.asarray(
                    mask_image.resize(
                        (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST
                    )
                ).reshape(-1)
                > 127
            )
            lab = lab[keep]
        lab = lab[np.all(np.isfinite(lab), axis=1)]
        if len(lab) < 16:
            raise ValueError("An MKL region must contain at least 16 valid pixels.")
        if len(lab) > self.max_samples:
            indices = np.linspace(0, len(lab) - 1, self.max_samples, dtype=np.int64)
            lab = lab[indices]
        return lab

    def fit(
        self,
        source: Image.Image,
        reference: Image.Image,
        source_mask: Optional[Image.Image | np.ndarray] = None,
        reference_mask: Optional[Image.Image | np.ndarray] = None,
    ) -> MongeKantorovichFit:
        source_pixels = self._pixels(source, source_mask)
        reference_pixels = self._pixels(reference, reference_mask)
        source_mean = np.mean(source_pixels, axis=0)
        reference_mean = np.mean(reference_pixels, axis=0)
        source_covariance = np.cov(source_pixels, rowvar=False)
        reference_covariance = np.cov(reference_pixels, rowvar=False)
        regularizer = self.covariance_epsilon * np.eye(3, dtype=np.float64)
        source_covariance = source_covariance + regularizer
        reference_covariance = reference_covariance + regularizer
        # The middle covariance product has squared units, so using the
        # covariance regularizer itself as its eigenvalue floor would strongly
        # inflate low-variance chroma directions.
        spectral_epsilon = max(np.finfo(np.float64).eps, self.covariance_epsilon**3)

        source_sqrt = _psd_power(source_covariance, 0.5, spectral_epsilon)
        source_inverse_sqrt = _psd_power(source_covariance, -0.5, spectral_epsilon)
        middle = source_sqrt @ reference_covariance @ source_sqrt
        matrix = (
            source_inverse_sqrt
            @ _psd_power(middle, 0.5, spectral_epsilon)
            @ source_inverse_sqrt
        )
        return MongeKantorovichFit(
            source_mean=source_mean,
            reference_mean=reference_mean,
            matrix=matrix,
            source_covariance=source_covariance,
            reference_covariance=reference_covariance,
        )

    def transfer(
        self,
        source: Image.Image,
        reference: Image.Image,
        source_mask: Optional[Image.Image | np.ndarray] = None,
        reference_mask: Optional[Image.Image | np.ndarray] = None,
    ) -> tuple[Image.Image, dict[str, object]]:
        fit = self.fit(source, reference, source_mask, reference_mask)
        source_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        source_lab = _rgb_to_normalized_lab(source_rgb)
        transported = (
            source_lab - fit.source_mean[None, None, :]
        ) @ fit.matrix.T + fit.reference_mean[None, None, :]
        blended = (1.0 - self.strength) * source_lab + self.strength * transported
        rgb, clipped_fraction = _normalized_lab_to_rgb(blended)

        before_covariance_error = float(
            np.linalg.norm(fit.source_covariance - fit.reference_covariance, ord="fro")
        )
        mapped_covariance = fit.matrix @ fit.source_covariance @ fit.matrix.T
        after_covariance_error = float(
            np.linalg.norm(mapped_covariance - fit.reference_covariance, ord="fro")
        )
        diagnostics: dict[str, object] = {
            "method": "linear-monge-kantorovich-lab",
            "role": "distribution_prior_not_semantic_ground_truth",
            "strength": self.strength,
            "source_mean_lab_normalized": fit.source_mean.tolist(),
            "reference_mean_lab_normalized": fit.reference_mean.tolist(),
            "matrix": fit.matrix.tolist(),
            "covariance_error_before": before_covariance_error,
            "covariance_error_after_full_transport": after_covariance_error,
            "clipped_channel_fraction": clipped_fraction,
            "masked": source_mask is not None or reference_mask is not None,
        }
        output = Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), mode="RGB")
        return output, diagnostics


@dataclass(frozen=True)
class ChromaAffinityFit:
    """One global chroma transform shared by every frame in a video."""

    source_mean_ab: np.ndarray
    reference_mean_ab: np.ndarray
    matrix_ab: np.ndarray

    def to_dict(self, strength: float) -> dict[str, object]:
        return {
            "method": "luma-preserving-chroma-affinity",
            "strength": float(strength),
            "source_mean_ab_normalized": self.source_mean_ab.tolist(),
            "reference_mean_ab_normalized": self.reference_mean_ab.tolist(),
            "matrix_ab": self.matrix_ab.tolist(),
            "luma_policy": "preserve_each_pixel_cielab_lightness",
            "temporal_policy": "one_transform_for_entire_video",
            "reference_source": "input_reference_video_only",
        }


class LumaPreservingChromaMatcher:
    """Transfer reference chroma covariance while preserving source lightness.

    This is a two-channel closed-form transport over CIELAB ``a,b``. It is
    deliberately narrower than full color transfer: the source luminance and
    its spatial structure remain unchanged, and one fit is reused for all
    frames so the operation cannot introduce framewise parameter flicker.
    """

    def __init__(
        self,
        strength: float = 0.6,
        covariance_epsilon: float = 1e-4,
        max_samples: int = 40_000,
        analysis_max_side: int = 128,
    ) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("Chroma-affinity strength must be in [0, 1].")
        self.strength = float(strength)
        self.covariance_epsilon = float(covariance_epsilon)
        self.max_samples = int(max_samples)
        self.analysis_max_side = int(analysis_max_side)

    def _video_ab(self, frames: Sequence[Image.Image]) -> np.ndarray:
        if not frames:
            raise ValueError("Chroma-affinity fitting requires non-empty videos.")
        indices = np.unique(
            np.linspace(0, len(frames) - 1, min(8, len(frames))).round().astype(int)
        )
        rows = []
        for index in indices:
            rgb = _resized_rgb(frames[int(index)], self.analysis_max_side)
            rows.append(_rgb_to_normalized_lab(rgb).reshape(-1, 3)[:, 1:])
        values = np.concatenate(rows, axis=0)
        if len(values) > self.max_samples:
            values = values[
                np.linspace(0, len(values) - 1, self.max_samples).astype(np.int64)
            ]
        return values

    def fit(
        self,
        source_frames: Sequence[Image.Image],
        reference_frames: Sequence[Image.Image],
    ) -> ChromaAffinityFit:
        source = self._video_ab(source_frames)
        reference = self._video_ab(reference_frames)
        regularizer = self.covariance_epsilon * np.eye(2, dtype=np.float64)
        source_covariance = np.cov(source, rowvar=False) + regularizer
        reference_covariance = np.cov(reference, rowvar=False) + regularizer
        source_sqrt = _psd_power(source_covariance, 0.5, 1e-12)
        matrix = (
            _psd_power(source_covariance, -0.5, 1e-12)
            @ _psd_power(
                source_sqrt @ reference_covariance @ source_sqrt,
                0.5,
                1e-12,
            )
            @ _psd_power(source_covariance, -0.5, 1e-12)
        )
        return ChromaAffinityFit(
            source_mean_ab=source.mean(axis=0),
            reference_mean_ab=reference.mean(axis=0),
            matrix_ab=matrix,
        )

    def apply(self, frame: Image.Image, fit: ChromaAffinityFit) -> Image.Image:
        rgb = np.asarray(frame.convert("RGB"), dtype=np.uint8)
        lab = _rgb_to_normalized_lab(rgb)
        source_ab = lab[..., 1:].reshape(-1, 2)
        mapped_ab = (
            source_ab - fit.source_mean_ab[None, :]
        ) @ fit.matrix_ab.T + fit.reference_mean_ab[None, :]
        lab[..., 1:] = (
            (1.0 - self.strength) * source_ab + self.strength * mapped_ab
        ).reshape(lab.shape[:2] + (2,))
        output, _ = _normalized_lab_to_rgb(lab)
        return Image.fromarray((output * 255.0 + 0.5).astype(np.uint8), mode="RGB")

    def transfer_video(
        self,
        source_frames: Sequence[Image.Image],
        reference_frames: Sequence[Image.Image],
    ) -> tuple[tuple[Image.Image, ...], dict[str, object]]:
        fit = self.fit(source_frames, reference_frames)
        output = tuple(self.apply(frame, fit) for frame in source_frames)
        return output, fit.to_dict(self.strength)


def spatiotemporal_palette_features(
    frames: Sequence[Image.Image],
    palette_size: int = 4,
    analysis_side: int = 32,
    iterations: int = 8,
) -> np.ndarray:
    """Build compact, temporally corresponding palette traces in Lab space.

    A single set of color centers is fitted over the whole shot. Per-frame
    cluster occupancy and cluster means then act as time-varying palette
    handles. This is a long-video approximation, not the 4-D skew-polytope
    extraction from the spatial-temporal geometric palette paper.
    """

    if not frames:
        raise ValueError("Palette features require at least one frame.")
    if palette_size < 1:
        raise ValueError("palette_size must be positive.")
    frame_pixels = [
        _rgb_to_normalized_lab(_resized_rgb(frame, analysis_side)).reshape(-1, 3)
        for frame in frames
    ]
    samples = np.concatenate(frame_pixels, axis=0)
    centers = [
        samples[int(np.argmin(np.linalg.norm(samples - samples.mean(0), axis=1)))]
    ]
    for _ in range(1, palette_size):
        distances = np.min(
            np.stack(
                [np.sum((samples - center) ** 2, axis=1) for center in centers],
                axis=1,
            ),
            axis=1,
        )
        centers.append(samples[int(np.argmax(distances))])
    centers_array = np.stack(centers, axis=0)
    for _ in range(iterations):
        distances = np.sum(
            (samples[:, None, :] - centers_array[None, :, :]) ** 2, axis=2
        )
        labels = np.argmin(distances, axis=1)
        updated = centers_array.copy()
        for cluster in range(palette_size):
            members = samples[labels == cluster]
            if len(members):
                updated[cluster] = np.mean(members, axis=0)
        if np.allclose(updated, centers_array, atol=1e-5):
            break
        centers_array = updated
    order = np.lexsort((centers_array[:, 2], centers_array[:, 1], centers_array[:, 0]))
    centers_array = centers_array[order]

    rows = []
    for pixels in frame_pixels:
        distances = np.sum(
            (pixels[:, None, :] - centers_array[None, :, :]) ** 2, axis=2
        )
        labels = np.argmin(distances, axis=1)
        weights = np.bincount(labels, minlength=palette_size).astype(np.float64)
        weights /= max(1, len(pixels))
        means = centers_array.copy()
        for cluster in range(palette_size):
            members = pixels[labels == cluster]
            if len(members):
                means[cluster] = np.mean(members, axis=0)
        rows.append(np.concatenate([weights, means.reshape(-1)]))
    return np.asarray(rows, dtype=np.float64)


class SourceGuidedTonalStabilizer:
    """Stabilize editable parameters using temporal regularity of the source.

    The source-video tonal residual estimates camera flicker. A sparse
    quadratic solve then balances that compensation, the Bayesian parameter
    prior, and first/second-order smoothness while preserving every Anchor
    exactly. This combines the motivation of Tonal Stabilization and blind
    input-guided temporal consistency without committing a pixel post-process.
    """

    def __init__(
        self,
        compensation_strength: float = 0.75,
        prior_weight: float = 2.0,
        velocity_weight: float = 2.0,
        curvature_weight: float = 8.0,
        anchor_weight: float = 1e7,
        maximum_exposure_compensation: float = 0.6,
        maximum_chroma_compensation: float = 0.35,
    ) -> None:
        self.compensation_strength = float(compensation_strength)
        self.prior_weight = float(prior_weight)
        self.velocity_weight = float(velocity_weight)
        self.curvature_weight = float(curvature_weight)
        self.anchor_weight = float(anchor_weight)
        self.maximum_exposure_compensation = float(maximum_exposure_compensation)
        self.maximum_chroma_compensation = float(maximum_chroma_compensation)

    @staticmethod
    def _source_signatures(frames: Sequence[Image.Image]) -> np.ndarray:
        rows = []
        for frame in frames:
            rgb = _resized_rgb(frame, 96).astype(np.float64) / 255.0
            linear = np.where(
                rgb <= 0.04045,
                rgb / 12.92,
                np.power((rgb + 0.055) / 1.055, 2.4),
            )
            luma = np.sum(linear * np.asarray([0.2126, 0.7152, 0.0722]), axis=2)
            exposure = np.log2(np.median(luma) + 1e-4)
            means = np.mean(linear.reshape(-1, 3), axis=0)
            temperature = np.log((means[0] + 1e-4) / (means[2] + 1e-4)) / 0.56
            tint = (
                -np.log((means[1] + 1e-4) / (0.5 * (means[0] + means[2]) + 1e-4)) / 0.26
            )
            rows.append([exposure, temperature, tint])
        return np.asarray(rows, dtype=np.float64)

    @staticmethod
    def _jerk(values: np.ndarray) -> float:
        if len(values) < 3:
            return 0.0
        return float(np.mean(np.abs(np.diff(values, n=2, axis=0))))

    def stabilize(
        self,
        frames: Sequence[Image.Image],
        trajectory: np.ndarray,
        anchor_indices: np.ndarray,
        anchor_values: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, object]]:
        values = np.asarray(trajectory, dtype=np.float64)
        anchors = np.asarray(anchor_indices, dtype=np.int64).reshape(-1)
        observations = np.asarray(anchor_values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != len(frames):
            raise ValueError("Trajectory must contain one parameter vector per frame.")
        if observations.shape != (len(anchors), values.shape[1]):
            raise ValueError("Anchor values do not match the trajectory.")
        if len(values) < 3:
            result = values.copy()
            result[anchors] = observations
            return result, {
                "method": "source-guided-parameter-stabilization",
                "applied": False,
                "reason": "shot_shorter_than_three_frames",
            }

        signatures = self._source_signatures(frames)
        sigma = float(np.clip(len(values) / 24.0, 1.0, 8.0))
        trend = gaussian_filter1d(signatures, sigma=sigma, axis=0, mode="nearest")
        flicker = signatures - trend
        tonal_guide = np.zeros_like(values)
        tonal_guide[:, 0] = np.clip(
            self.compensation_strength * flicker[:, 0],
            -self.maximum_exposure_compensation,
            self.maximum_exposure_compensation,
        )
        tonal_guide[:, 1:3] = np.clip(
            self.compensation_strength * flicker[:, 1:3],
            -self.maximum_chroma_compensation,
            self.maximum_chroma_compensation,
        )

        length = len(values)
        first = diags(
            [-np.ones(length - 1), np.ones(length - 1)],
            [0, 1],
            shape=(length - 1, length),
            format="csr",
        )
        second = diags(
            [np.ones(length - 2), -2.0 * np.ones(length - 2), np.ones(length - 2)],
            [0, 1, 2],
            shape=(length - 2, length),
            format="csr",
        )
        changes = np.linalg.norm(np.diff(signatures, axis=0), axis=1)
        scale = max(float(np.median(changes)), 1e-4)
        coherence = np.exp(-changes / (3.0 * scale))
        velocity = first.T @ diags(coherence, format="csr") @ first
        smoothness = (
            self.velocity_weight * velocity
            + self.curvature_weight * (second.T @ second)
        ).tocsr()
        system: csr_matrix = (
            self.prior_weight * eye(length, format="csr") + smoothness
        ).tocsr()
        # For exposure/WB, smooth the estimated output tonal residual
        # (parameter + source flicker), not the parameter alone. This permits a
        # deliberately high-frequency compensation when the camera flickers.
        right = self.prior_weight * values - smoothness @ tonal_guide
        anchor_diagonal = np.zeros(length, dtype=np.float64)
        anchor_diagonal[anchors] = self.anchor_weight
        system = system + diags(anchor_diagonal, format="csr")
        right[anchors] += self.anchor_weight * observations

        stabilized = np.asarray(spsolve(system, right), dtype=np.float64)
        stabilized[:, 0] = values[:, 0] + np.clip(
            stabilized[:, 0] - values[:, 0],
            -self.maximum_exposure_compensation,
            self.maximum_exposure_compensation,
        )
        stabilized[:, 1:3] = values[:, 1:3] + np.clip(
            stabilized[:, 1:3] - values[:, 1:3],
            -self.maximum_chroma_compensation,
            self.maximum_chroma_compensation,
        )
        stabilized = np.clip(
            stabilized,
            PARAMETER_LOWER_BOUNDS[None, :],
            PARAMETER_UPPER_BOUNDS[None, :],
        )
        # Exact constraints matter for editable graph round-tripping.
        stabilized[anchors] = observations
        diagnostics = {
            "method": "source-guided-parameter-stabilization",
            "paper_inspiration": [
                "tonal-stabilization",
                "blind-input-guided-temporal-consistency",
            ],
            "applied": True,
            "source_trend_sigma_frames": sigma,
            "source_flicker_rms": np.sqrt(np.mean(flicker * flicker, axis=0)).tolist(),
            "parameter_jerk_before": self._jerk(values),
            "parameter_jerk_after": self._jerk(stabilized),
            "parameter_adjustment_rms": float(
                np.sqrt(np.mean((stabilized - values) ** 2))
            ),
            "anchor_constraints_preserved": True,
        }
        return stabilized, diagnostics
