"""Budget-curve benchmark for active, uniform, and random Anchor selection."""

from __future__ import annotations

import argparse

import numpy as np

from .demo import make_episode
from .parameter_field import BayesianGradeField


def reconstruct_rmse(field, times, features, values, anchors) -> float:
    posterior = field.posterior(times, anchors, values[anchors], features=features)
    return float(np.sqrt(np.mean((posterior.mean - values) ** 2)))


def active_anchors(field, times, features, values, budget, initial_anchor):
    anchors = [initial_anchor]
    while len(anchors) < budget:
        posterior = field.posterior(times, anchors, values[anchors], features=features)
        next_anchor, _ = field.select_next_anchor(posterior, anchors)
        anchors.append(next_anchor)
    return anchors


def evaluate(frame_count: int, max_budget: int, seeds: int) -> list[dict]:
    rows = []
    for budget in range(1, max_budget + 1):
        scores = {"active": [], "uniform": [], "random": []}
        for seed in range(seeds):
            times, features, values = make_episode(frame_count, seed)
            field = BayesianGradeField(
                temporal_lengthscale=0.20,
                feature_lengthscale=1.35,
                observation_noise=1e-5,
            )
            initial = frame_count // 4
            active = active_anchors(
                field, times, features, values, budget, initial
            )
            uniform = (
                [initial]
                if budget == 1
                else np.linspace(0, frame_count - 1, budget, dtype=int).tolist()
            )
            rng = np.random.default_rng(10000 + seed * 97 + budget)
            random = [initial]
            if budget > 1:
                pool = np.delete(np.arange(frame_count), initial)
                random.extend(
                    rng.choice(pool, size=budget - 1, replace=False).tolist()
                )

            scores["active"].append(
                reconstruct_rmse(field, times, features, values, active)
            )
            scores["uniform"].append(
                reconstruct_rmse(field, times, features, values, uniform)
            )
            scores["random"].append(
                reconstruct_rmse(field, times, features, values, random)
            )

        rows.append(
            {
                "budget": budget,
                **{name: float(np.mean(result)) for name, result in scores.items()},
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--max-budget", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    rows = evaluate(args.frames, args.max_budget, args.seeds)
    print("| Anchor budget | Active RMSE | Uniform RMSE | Random RMSE |")
    print("|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['budget']} | {row['active']:.4f} | "
            f"{row['uniform']:.4f} | {row['random']:.4f} |"
        )


if __name__ == "__main__":
    main()
