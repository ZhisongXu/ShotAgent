"""Synthetic active-Anchor experiment for the first BayesGrade milestone."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .parameter_field import BayesianGradeField


def make_episode(frame_count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    times = np.linspace(0.0, 1.0, frame_count)
    event = (times >= 0.52).astype(np.float64)
    luminance = 0.25 + 0.18 * np.sin(2.0 * np.pi * times) + 0.55 * event
    luminance += rng.normal(scale=0.015, size=frame_count)
    motion = 0.08 + 0.55 * np.exp(-((times - 0.52) / 0.055) ** 2)
    features = np.stack([luminance, motion, event], axis=1)

    exposure = 0.20 * np.sin(2.0 * np.pi * times) - 0.48 * event
    exposure += 0.12 * np.exp(-((times - 0.78) / 0.12) ** 2)
    temperature = 0.15 * np.cos(np.pi * times) + 0.42 * event
    parameters = np.stack([exposure, temperature], axis=1)
    return times, features, parameters


def run_active_episode(frame_count: int, budget: int, seed: int) -> dict:
    times, features, ground_truth = make_episode(frame_count, seed)
    field = BayesianGradeField(
        temporal_lengthscale=0.20,
        feature_lengthscale=1.35,
        observation_noise=1e-5,
    )

    anchors = [frame_count // 4]
    history = []
    while len(anchors) <= budget:
        posterior = field.posterior(
            times,
            anchors,
            ground_truth[anchors],
            features=features,
        )
        rmse = float(np.sqrt(np.mean((posterior.mean - ground_truth) ** 2)))
        record = {
            "anchor_count": len(anchors),
            "anchors": list(anchors),
            "rmse": rmse,
            "integrated_variance": posterior.integrated_variance,
        }
        history.append(record)
        if len(anchors) == budget:
            break
        next_anchor, score = field.select_next_anchor(posterior, anchors)
        record["next_anchor"] = next_anchor
        record["acquisition_score"] = score
        anchors.append(next_anchor)

    return {
        "frame_count": frame_count,
        "budget": budget,
        "seed": seed,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.frames < 8:
        parser.error("--frames must be at least 8")
    if not 1 <= args.budget < args.frames:
        parser.error("--budget must lie in [1, frames)")
    print(json.dumps(run_active_episode(args.frames, args.budget, args.seed), indent=2))


if __name__ == "__main__":
    main()
