"""Lightweight multi-objective evaluator for Anchor candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .planner import RetouchPlan


@dataclass(frozen=True)
class CandidateEvaluation:
    score: float
    valid: bool
    metrics: dict[str, float]


class RetouchEvaluator:
    @staticmethod
    def _statistics(image: Tensor) -> dict[str, float]:
        weights = image.new_tensor([0.2126, 0.7152, 0.0722])[:, None, None]
        luma = (image * weights).sum(dim=0)
        return {
            "luminance": float(luma.mean()),
            "contrast": float(luma.std()),
            "saturation": float((image.max(dim=0).values - image.min(dim=0).values).mean()),
            "warmth": float((image[0] - image[2]).mean()),
            "highlight_clipping": float((image >= 0.995).float().mean()),
            "shadow_crushing": float((image <= 0.005).float().mean()),
        }

    def evaluate(self, source: Tensor, output: Tensor, plan: RetouchPlan) -> CandidateEvaluation:
        stats = self._statistics(output)
        fidelity_l1 = float(torch.mean(torch.abs(output - source)))
        target = plan.targets

        errors = {
            "luminance_error": abs(stats["luminance"] - target.get("luminance", stats["luminance"])) / 0.20,
            "contrast_error": abs(stats["contrast"] - target.get("contrast", stats["contrast"])) / 0.12,
            "saturation_error": abs(stats["saturation"] - target.get("saturation", stats["saturation"])) / 0.20,
            "warmth_error": abs(stats["warmth"] - target.get("warmth", stats["warmth"])) / 0.12,
        }
        score = -sum(errors.values()) - 1.5 * fidelity_l1
        score -= 5.0 * stats["highlight_clipping"] + 3.0 * stats["shadow_crushing"]
        valid = (
            stats["highlight_clipping"] < 0.18
            and stats["shadow_crushing"] < 0.30
            and fidelity_l1 < 0.45
        )
        metrics = {
            **stats,
            **errors,
            "fidelity_l1": fidelity_l1,
        }
        return CandidateEvaluation(score=float(score), valid=valid, metrics=metrics)
