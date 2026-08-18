"""Execution–evaluation Anchor retouching agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .evaluator import CandidateEvaluation, RetouchEvaluator
from .executor import RetouchExecutor
from .parameters import (
    PARAMETER_LOWER_BOUNDS,
    PARAMETER_NAMES,
    PARAMETER_UPPER_BOUNDS,
    RetouchParameters,
)
from .planner import HeuristicRetouchPlanner, RetouchPlan, RetouchPlanner


@dataclass(frozen=True)
class AnchorRetouchResult:
    image: Image.Image
    parameters: RetouchParameters
    parameter_covariance: np.ndarray
    evaluation: CandidateEvaluation
    plan: RetouchPlan
    accepted_candidates: int
    rolled_back: bool
    rollback_reason: Optional[str]
    decision: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "parameters": self.parameters.to_dict(),
            "parameter_names": list(PARAMETER_NAMES),
            "parameter_covariance": self.parameter_covariance.tolist(),
            "evaluation": {
                "score": self.evaluation.score,
                "valid": self.evaluation.valid,
                "metrics": self.evaluation.metrics,
            },
            "plan": self.plan.to_dict(),
            "accepted_candidates": self.accepted_candidates,
            "rolled_back": self.rolled_back,
            "rollback_reason": self.rollback_reason,
            "decision": self.decision,
        }


class AnchorRetouchAgent:
    """Plan, execute, evaluate, select, and expose parameter uncertainty."""

    proposal_scale = np.array(
        [0.22, 0.12, 0.10, 0.12, 0.12, 0.12, 0.12, 0.12, 0.10, 0.16, 0.10, 0.10],
        dtype=np.float64,
    )

    def __init__(
        self,
        planner: Optional[RetouchPlanner] = None,
        executor: Optional[RetouchExecutor] = None,
        evaluator: Optional[RetouchEvaluator] = None,
        candidate_count: int = 24,
        covariance_top_k: int = 6,
        seed: int = 7,
        rollback_on_failure: bool = True,
        minimum_score_improvement: float = 0.0,
        minimum_perceptual_delta: float = 0.01,
    ) -> None:
        if candidate_count < 1 or covariance_top_k < 1:
            raise ValueError("candidate_count and covariance_top_k must be positive.")
        self.planner = planner or HeuristicRetouchPlanner()
        self.executor = executor or RetouchExecutor()
        self.evaluator = evaluator or RetouchEvaluator()
        self.candidate_count = int(candidate_count)
        self.covariance_top_k = int(covariance_top_k)
        self.seed = int(seed)
        self.rollback_on_failure = bool(rollback_on_failure)
        self.minimum_score_improvement = float(minimum_score_improvement)
        if minimum_perceptual_delta < 0.0:
            raise ValueError("minimum_perceptual_delta must be non-negative.")
        self.minimum_perceptual_delta = float(minimum_perceptual_delta)

    @staticmethod
    def _directional_quality(evaluation: CandidateEvaluation) -> float:
        """Measure target alignment independently of edit magnitude.

        Third-party evaluators can omit the target-error metric; their score is
        then used as the compatibility fallback.
        """

        target_error = evaluation.metrics.get("target_error")
        if target_error is None:
            return evaluation.score
        return -float(target_error)

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    @staticmethod
    def _mask_to_tensor(mask: Optional[Image.Image]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        return torch.from_numpy(
            np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
        )

    def _propose(self, initial: RetouchParameters, has_mask: bool) -> np.ndarray:
        center = initial.to_vector()
        rng = np.random.default_rng(self.seed)
        proposals = np.repeat(center[None, :], self.candidate_count, axis=0)
        if self.candidate_count > 1:
            proposals[1:] += rng.normal(
                size=(self.candidate_count - 1, center.size)
            ) * self.proposal_scale[None, :]
        if not has_mask:
            proposals[:, 9:12] = 0.0
        return np.clip(proposals, PARAMETER_LOWER_BOUNDS, PARAMETER_UPPER_BOUNDS)

    def run(
        self,
        image: Image.Image,
        instruction: str,
        reference: Optional[Image.Image] = None,
        local_mask: Optional[Image.Image] = None,
    ) -> AnchorRetouchResult:
        image = image.convert("RGB")
        plan = self.planner.plan(
            image,
            instruction,
            reference=reference,
            has_local_mask=local_mask is not None,
        )
        proposals = self._propose(plan.initial_parameters, local_mask is not None)
        source = self._pil_to_tensor(image)
        mask = self._mask_to_tensor(local_mask)
        batch = source.unsqueeze(0).expand(self.candidate_count, -1, -1, -1)
        parameter_tensor = torch.from_numpy(proposals).to(dtype=source.dtype)
        rendered = self.executor.apply_vector(batch, parameter_tensor, mask=mask)

        evaluations = [
            self.evaluator.evaluate(source, rendered[index], plan)
            for index in range(self.candidate_count)
        ]
        valid_indices = [index for index, result in enumerate(evaluations) if result.valid]
        checkpoint_evaluation = self.evaluator.evaluate(source, source, plan)
        checkpoint_quality = self._directional_quality(checkpoint_evaluation)
        perceptual_deltas = (
            (rendered - source.unsqueeze(0)).square().mean(dim=(1, 2, 3)).sqrt()
        ).detach().cpu().numpy()
        perceptible_indices = [
            index
            for index in valid_indices
            if perceptual_deltas[index] >= self.minimum_perceptual_delta
        ]
        directional_indices = [
            index
            for index in perceptible_indices
            if self._directional_quality(evaluations[index])
            > checkpoint_quality + self.minimum_score_improvement
        ]

        if self.rollback_on_failure:
            pool = directional_indices
        else:
            pool = valid_indices or list(range(self.candidate_count))
        fallback_pool = valid_indices or list(range(self.candidate_count))
        ranked = sorted(
            pool or fallback_pool,
            key=lambda index: evaluations[index].score,
            reverse=True,
        )
        best = ranked[0]
        best_quality = self._directional_quality(evaluations[best])
        decision = {
            "policy": "directional_perceptual_v2",
            "minimum_directional_improvement": self.minimum_score_improvement,
            "minimum_perceptual_delta": self.minimum_perceptual_delta,
            "valid_candidates": len(valid_indices),
            "perceptible_candidates": len(perceptible_indices),
            "directional_candidates": len(directional_indices),
            "checkpoint_directional_quality": checkpoint_quality,
            "best_candidate_directional_quality": best_quality,
            "directional_improvement": best_quality - checkpoint_quality,
            "best_candidate_perceptual_delta": float(perceptual_deltas[best]),
        }

        rollback_reason: Optional[str] = None
        if self.rollback_on_failure:
            if not valid_indices:
                rollback_reason = "no_valid_candidate"
            elif not perceptible_indices:
                rollback_reason = "no_perceptible_improvement"
            elif not directional_indices:
                rollback_reason = "no_directional_improvement"

        if rollback_reason is not None:
            parameter_count = len(PARAMETER_NAMES)
            return AnchorRetouchResult(
                image=image.copy(),
                parameters=RetouchParameters(),
                parameter_covariance=np.zeros(
                    (parameter_count, parameter_count), dtype=np.float64
                ),
                evaluation=checkpoint_evaluation,
                plan=plan,
                accepted_candidates=len(valid_indices),
                rolled_back=True,
                rollback_reason=rollback_reason,
                decision=decision,
            )

        top = ranked[: min(self.covariance_top_k, len(ranked))]
        if len(top) > 1:
            covariance = np.cov(proposals[top], rowvar=False)
        else:
            covariance = np.zeros((len(PARAMETER_NAMES), len(PARAMETER_NAMES)), dtype=np.float64)

        output_array = (
            rendered[best].permute(1, 2, 0).detach().cpu().numpy() * 255.0 + 0.5
        ).astype(np.uint8)
        return AnchorRetouchResult(
            image=Image.fromarray(output_array, mode="RGB"),
            parameters=RetouchParameters.from_vector(proposals[best]),
            parameter_covariance=covariance,
            evaluation=evaluations[best],
            plan=plan,
            accepted_candidates=len(valid_indices),
            rolled_back=False,
            rollback_reason=None,
            decision=decision,
        )
