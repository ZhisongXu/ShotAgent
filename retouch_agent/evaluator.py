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
        perceptual_delta = float(torch.mean((output - source).square()).sqrt())
        target = plan.targets

        errors = {
            "luminance_error": abs(stats["luminance"] - target.get("luminance", stats["luminance"])) / 0.20,
            "contrast_error": abs(stats["contrast"] - target.get("contrast", stats["contrast"])) / 0.12,
            "saturation_error": abs(stats["saturation"] - target.get("saturation", stats["saturation"])) / 0.20,
            "warmth_error": abs(stats["warmth"] - target.get("warmth", stats["warmth"])) / 0.12,
        }
        target_error = sum(errors.values())
        # Fidelity remains a tie-breaker, but target-directed grading must be
        # allowed to beat the identity checkpoint.  The previous 1.5 weight
        # frequently made "do nothing" the highest-scoring result.
        score = -target_error - 0.35 * fidelity_l1
        # Clipping and crushing remain observable metrics, but artistic edits
        # are no longer vetoed by fixed safety thresholds.  ``valid`` now only
        # represents whether the executor produced a numerically usable image.
        valid = bool(torch.isfinite(output).all())
        metrics = {
            **stats,
            **errors,
            "target_error": target_error,
            "fidelity_l1": fidelity_l1,
            "perceptual_delta": perceptual_delta,
        }
        return CandidateEvaluation(score=float(score), valid=valid, metrics=metrics)
