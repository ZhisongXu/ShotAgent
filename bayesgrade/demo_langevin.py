"""Demonstrate GP-guided Langevin sampling and disagreement acquisition."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .demo import make_episode
from .langevin import (
    LangevinGradeSampler,
    hybrid_acquisition_scores,
    select_disagreement_anchor,
)
from .parameter_field import BayesianGradeField


def run_demo(frame_count: int, samples: int, seed: int) -> dict:
    times, features, ground_truth = make_episode(frame_count, seed)
    anchors = [frame_count // 4]
    gp = BayesianGradeField(
        temporal_lengthscale=0.20,
        feature_lengthscale=1.35,
        observation_noise=1e-5,
    )
    posterior = gp.posterior(
        times,
        anchors,
        ground_truth[anchors],
        features=features,
    )
    gp_anchor, gp_score = gp.select_next_anchor(posterior, anchors)
    gp_scores = gp.acquisition_scores(posterior, anchors)

    # Relax temporal smoothing around photometric events instead of treating a
    # real illumination transition as an editing artifact.
    curvature = np.abs(np.diff(features[:, 0], n=2))
    smoothness_weights = np.exp(-5.0 * curvature)
    sampler = LangevinGradeSampler(
        control_count=min(12, frame_count),
        iterations=300,
        temperature=0.15,
        smoothness_weight=3.0,
    )
    sampled = sampler.sample(
        times,
        prior_mean=posterior.mean,
        prior_variance=posterior.variance,
        anchor_indices=anchors,
        anchor_values=ground_truth[anchors],
        num_samples=samples,
        smoothness_weights=smoothness_weights,
        parameter_bounds=(np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
        seed=seed,
    )
    langevin_anchor, langevin_score = select_disagreement_anchor(sampled, anchors)
    hybrid_scores = hybrid_acquisition_scores(gp_scores, sampled, anchors)
    hybrid_anchor = int(np.argmax(hybrid_scores))

    return {
        "frame_count": frame_count,
        "anchors": anchors,
        "gp": {
            "rmse": float(np.sqrt(np.mean((posterior.mean - ground_truth) ** 2))),
            "mean_uncertainty": float(posterior.variance.mean()),
            "next_anchor": gp_anchor,
            "acquisition_score": gp_score,
        },
        "langevin": {
            "num_samples": samples,
            "rmse": float(np.sqrt(np.mean((sampled.mean - ground_truth) ** 2))),
            "mean_uncertainty": float(sampled.frame_uncertainty.mean()),
            "next_anchor": langevin_anchor,
            "acquisition_score": langevin_score,
            "energy_mean": float(sampled.energies.mean()),
            "energy_std": float(sampled.energies.std()),
        },
        "hybrid": {
            "next_anchor": hybrid_anchor,
            "acquisition_score": float(hybrid_scores[hybrid_anchor]),
            "disagreement_weight": 0.35,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.frames, args.samples, args.seed), indent=2))


if __name__ == "__main__":
    main()
