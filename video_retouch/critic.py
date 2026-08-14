"""PhotoAgent-style closed-loop evaluator for shot parameter trajectories."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import cv2
import numpy as np
from PIL import Image

from .backends import AnchorGrade, HeroAnchorReference
from .clients import VisionLanguageClient
from .models import ShotPlan
from .tasks import critique_prompt


@dataclass(frozen=True)
class ShotCritique:
    score: float
    accepted: bool
    metrics: dict[str, float]
    reasons: tuple[str, ...]
    recommended_anchor: Optional[int]
    metadata: dict[str, object] = field(default_factory=dict)


class ShotCritic(Protocol):
    name: str

    def evaluate(
        self,
        source_frames: Sequence[Image.Image],
        output_frames: Sequence[Image.Image],
        frame_parameters: np.ndarray,
        frame_uncertainty: np.ndarray,
        shot: ShotPlan,
        instruction: str,
        anchor_grades: Sequence[AnchorGrade],
        hero_reference: Optional[HeroAnchorReference] = None,
    ) -> ShotCritique: ...


def _json_boolean(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{field_name} must be a JSON boolean.")


def _unit_score(value: object, field_name: str) -> float:
    score = float(value)
    if not np.isfinite(score):
        raise ValueError(f"{field_name} must be finite.")
    return float(np.clip(score, 0.0, 1.0))


def _rgb(image: Image.Image, max_side: int = 320) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = array.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        array = cv2.resize(
            array,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return array


def _luma(array: np.ndarray) -> np.ndarray:
    return 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]


def _style_signature(image: Image.Image) -> np.ndarray:
    rgb = _rgb(image)
    luma = _luma(rgb)
    chroma = rgb - luma[..., None]
    return np.asarray(
        [
            *np.percentile(luma, [10, 50, 90]).tolist(),
            *rgb.mean(axis=(0, 1)).tolist(),
            *chroma.mean(axis=(0, 1)).tolist(),
            float(np.std(luma)),
        ],
        dtype=np.float64,
    )


def _motion_compensated_residual_error(
    source: Sequence[np.ndarray], output: Sequence[np.ndarray]
) -> tuple[float, np.ndarray]:
    if len(source) <= 1:
        return 0.0, np.zeros(len(source), dtype=np.float64)
    per_frame = np.zeros(len(source), dtype=np.float64)
    residuals = [out - inp for inp, out in zip(source, output)]
    for index in range(1, len(source)):
        current_gray = (_luma(source[index]) * 255.0).astype(np.uint8)
        previous_gray = (_luma(source[index - 1]) * 255.0).astype(np.uint8)
        # Backward flow maps each current pixel into the previous frame.
        backward = cv2.calcOpticalFlowFarneback(
            current_gray,
            previous_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        height, width = current_gray.shape
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        warped = cv2.remap(
            residuals[index - 1],
            grid_x + backward[..., 0],
            grid_y + backward[..., 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        per_frame[index] = float(np.mean(np.abs(residuals[index] - warped)))
    return float(per_frame[1:].mean()), per_frame


class PhotoAgentStyleCritic:
    """Evaluate, remember, and request re-planning like PhotoAgent.

    PhotoAgent's official repository has not released code or reward weights, so
    this class implements the published closed-loop pattern with explicit video
    metrics and an optional independent vision-model review. It is intentionally not named or
    presented as the original PhotoAgent evaluator.
    """

    name = "photoagent-style-temporal-critic"

    def __init__(
        self,
        vl_client: Optional[VisionLanguageClient] = None,
        use_vl_review: bool = True,
        strict_vl: bool = False,
        maximum_fidelity_l1: float = 0.42,
        maximum_clipping: float = 0.20,
        maximum_temporal_error: float = 0.10,
        maximum_parameter_jerk: float = 0.20,
        maximum_anchor_error: float = 0.12,
    ) -> None:
        self.vl_client = vl_client
        self.use_vl_review = bool(use_vl_review and vl_client is not None)
        self.strict_vl = bool(strict_vl)
        self.maximum_fidelity_l1 = float(maximum_fidelity_l1)
        self.maximum_clipping = float(maximum_clipping)
        self.maximum_temporal_error = float(maximum_temporal_error)
        self.maximum_parameter_jerk = float(maximum_parameter_jerk)
        self.maximum_anchor_error = float(maximum_anchor_error)

    def _vl_review(
        self,
        source_frames: Sequence[Image.Image],
        output_frames: Sequence[Image.Image],
        shot: ShotPlan,
        instruction: str,
        hero_reference: Optional[HeroAnchorReference],
    ) -> dict[str, object]:
        assert self.vl_client is not None
        sample_count = min(4, len(source_frames))
        indices = np.unique(
            np.linspace(0, len(source_frames) - 1, sample_count).round().astype(int)
        )
        labeled: list[tuple[str, Image.Image]] = []
        if hero_reference is not None:
            labeled.extend(
                [
                    (
                        f"HeroAnchor source frame {hero_reference.frame_index}",
                        hero_reference.source,
                    ),
                    (
                        f"HeroAnchor accepted grade frame "
                        f"{hero_reference.frame_index}",
                        hero_reference.grade.preview,
                    ),
                ]
            )
        for local_index in indices.tolist():
            absolute = shot.start_frame + local_index
            labeled.append((f"source frame {absolute}", source_frames[local_index]))
            labeled.append((f"graded frame {absolute}", output_frames[local_index]))
        prompt = critique_prompt(
            instruction,
            hero_frame=(
                None if hero_reference is None else hero_reference.frame_index
            ),
        )
        return self.vl_client.generate_json(labeled, prompt)

    def evaluate(
        self,
        source_frames: Sequence[Image.Image],
        output_frames: Sequence[Image.Image],
        frame_parameters: np.ndarray,
        frame_uncertainty: np.ndarray,
        shot: ShotPlan,
        instruction: str,
        anchor_grades: Sequence[AnchorGrade],
        hero_reference: Optional[HeroAnchorReference] = None,
    ) -> ShotCritique:
        if len(source_frames) != len(output_frames) or not source_frames:
            raise ValueError("Critic requires matching non-empty frame sequences.")
        source = [_rgb(image) for image in source_frames]
        output = [_rgb(image) for image in output_frames]
        fidelity = float(
            np.mean(
                [
                    np.mean(np.abs(after - before))
                    for before, after in zip(source, output)
                ]
            )
        )
        clipping = float(
            np.mean([np.mean((after <= 0.005) | (after >= 0.995)) for after in output])
        )
        temporal_error, frame_risk = _motion_compensated_residual_error(source, output)
        if len(frame_parameters) > 2:
            jerk = float(
                np.mean(
                    np.abs(
                        frame_parameters[2:]
                        - 2.0 * frame_parameters[1:-1]
                        + frame_parameters[:-2]
                    )
                )
            )
        else:
            jerk = 0.0

        anchor_errors = []
        for grade in anchor_grades:
            local_index = grade.frame_index - shot.start_frame
            reference = _rgb(grade.preview)
            rendered = output[local_index]
            if reference.shape != rendered.shape:
                reference = cv2.resize(
                    reference,
                    (rendered.shape[1], rendered.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            anchor_errors.append(float(np.mean(np.abs(reference - rendered))))
        anchor_error = float(np.mean(anchor_errors)) if anchor_errors else 1.0
        metrics = {
            "fidelity_l1": fidelity,
            "clipping_fraction": clipping,
            "motion_compensated_residual_error": temporal_error,
            "parameter_jerk": jerk,
            "anchor_reconstruction_l1": anchor_error,
            "mean_uncertainty": float(np.mean(frame_uncertainty)),
        }
        if hero_reference is not None:
            hero_signature = _style_signature(hero_reference.grade.preview)
            metrics["hero_style_distance"] = float(
                np.mean(
                    [
                        np.linalg.norm(_style_signature(image) - hero_signature)
                        / np.sqrt(hero_signature.size)
                        for image in output_frames
                    ]
                )
            )
        reasons = []
        if not all(grade.valid for grade in anchor_grades):
            reasons.append("anchor_backend_rejected")
        if fidelity > self.maximum_fidelity_l1:
            reasons.append("content_fidelity")
        if clipping > self.maximum_clipping:
            reasons.append("highlight_or_shadow_clipping")
        if temporal_error > self.maximum_temporal_error:
            reasons.append("temporal_edit_residual")
        if jerk > self.maximum_parameter_jerk:
            reasons.append("parameter_trajectory_jerk")
        if anchor_error > self.maximum_anchor_error:
            reasons.append("anchor_parameter_reconstruction")

        metadata: dict[str, object] = {}
        vl_accept = True
        vl_score = 1.0
        if self.use_vl_review:
            try:
                review = self._vl_review(
                    source_frames,
                    output_frames,
                    shot,
                    instruction,
                    hero_reference,
                )
                metadata["vl_review"] = review
                metadata["critic_model_id"] = self.vl_client.model_id
                vl_accept = _json_boolean(review.get("accept", False), "accept")
                vl_score = _unit_score(review.get("score", 0.0), "score")
                if not vl_accept:
                    reasons.append("vl_review_rejected")
            except Exception as error:
                if self.strict_vl:
                    raise
                metadata["vl_fallback_reason"] = f"{type(error).__name__}: {error}"

        normalized_penalty = (
            fidelity / max(self.maximum_fidelity_l1, 1e-8)
            + clipping / max(self.maximum_clipping, 1e-8)
            + temporal_error / max(self.maximum_temporal_error, 1e-8)
            + jerk / max(self.maximum_parameter_jerk, 1e-8)
            + anchor_error / max(self.maximum_anchor_error, 1e-8)
        ) / 5.0
        score = float(0.75 * max(0.0, 1.0 - normalized_penalty) + 0.25 * vl_score)
        accepted = not reasons and vl_accept
        combined_risk = frame_risk + np.asarray(frame_uncertainty, dtype=np.float64)
        for grade in anchor_grades:
            combined_risk[grade.frame_index - shot.start_frame] = -np.inf
        recommended = None
        if not accepted and np.any(np.isfinite(combined_risk)):
            recommended = shot.start_frame + int(np.argmax(combined_risk))
        return ShotCritique(
            score=score,
            accepted=accepted,
            metrics=metrics,
            reasons=tuple(reasons),
            recommended_anchor=recommended,
            metadata=metadata,
        )


class VisionReviewCritic:
    """An independent visual specialist in the evaluator ensemble."""

    def __init__(
        self,
        client: VisionLanguageClient,
        name: str,
        focus: str,
        strict: bool = True,
        fallback_score: float = 0.0,
    ) -> None:
        if not name:
            raise ValueError("Vision critic requires a name.")
        self.client = client
        self.name = name
        self.focus = focus
        self.strict = bool(strict)
        self.fallback_score = float(np.clip(fallback_score, 0.0, 1.0))

    def evaluate(
        self,
        source_frames: Sequence[Image.Image],
        output_frames: Sequence[Image.Image],
        frame_parameters: np.ndarray,
        frame_uncertainty: np.ndarray,
        shot: ShotPlan,
        instruction: str,
        anchor_grades: Sequence[AnchorGrade],
        hero_reference: Optional[HeroAnchorReference] = None,
    ) -> ShotCritique:
        del frame_parameters, frame_uncertainty, anchor_grades
        sample_count = min(4, len(source_frames))
        indices = np.unique(
            np.linspace(0, len(source_frames) - 1, sample_count).round().astype(int)
        )
        labeled: list[tuple[str, Image.Image]] = []
        if hero_reference is not None:
            labeled.extend(
                [
                    (
                        f"HeroAnchor source frame {hero_reference.frame_index}",
                        hero_reference.source,
                    ),
                    (
                        f"HeroAnchor accepted grade frame "
                        f"{hero_reference.frame_index}",
                        hero_reference.grade.preview,
                    ),
                ]
            )
        for local_index in indices.tolist():
            absolute = shot.start_frame + local_index
            labeled.append((f"source frame {absolute}", source_frames[local_index]))
            labeled.append((f"graded frame {absolute}", output_frames[local_index]))
        try:
            review = self.client.generate_json(
                labeled,
                critique_prompt(
                    instruction,
                    focus=self.focus,
                    hero_frame=(
                        None
                        if hero_reference is None
                        else hero_reference.frame_index
                    ),
                ),
            )
            score = _unit_score(review.get("score", 0.0), "score")
            accepted = _json_boolean(review.get("accept", False), "accept")
            raw_reasons = review.get("reasons", [])
            reasons = (
                tuple(str(value) for value in raw_reasons)
                if isinstance(raw_reasons, list)
                else (str(raw_reasons),)
            )
            metrics = {}
            for key in (
                "instruction_score",
                "content_score",
                "consistency_score",
                "hero_match_score",
            ):
                if key in review:
                    metrics[key] = _unit_score(review.get(key, 0.0), key)
            recommended = review.get("recommended_anchor")
            if recommended is not None:
                recommended = int(recommended)
                if recommended < shot.start_frame or recommended > shot.end_frame:
                    recommended = None
            return ShotCritique(
                score=score,
                accepted=accepted,
                metrics=metrics,
                reasons=(reasons or ("visual_rejection",)) if not accepted else (),
                recommended_anchor=recommended,
                metadata={
                    "model_id": self.client.model_id,
                    "focus": self.focus,
                    "review": review,
                },
            )
        except Exception as error:
            if self.strict:
                raise
            return ShotCritique(
                score=self.fallback_score,
                accepted=False,
                metrics={"review_available": 0.0},
                reasons=("review_unavailable",),
                recommended_anchor=None,
                metadata={
                    "model_id": self.client.model_id,
                    "focus": self.focus,
                    "error": f"{type(error).__name__}: {error}",
                },
            )


@dataclass(frozen=True)
class CriticMember:
    critic: ShotCritic
    weight: float = 1.0
    veto: bool = False

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("Critic weight must be positive.")


class CriticEnsemble:
    """Aggregate independent reviewers while preserving safety vetoes."""

    name = "photoagent-multi-critic-ensemble"

    def __init__(
        self,
        members: Sequence[CriticMember],
        acceptance_score: float = 0.60,
    ) -> None:
        if not members:
            raise ValueError("Critic ensemble requires at least one member.")
        names = [member.critic.name for member in members]
        if len(set(names)) != len(names):
            raise ValueError("Critic names must be unique.")
        self.members = tuple(members)
        self.acceptance_score = float(np.clip(acceptance_score, 0.0, 1.0))

    def evaluate(
        self,
        source_frames: Sequence[Image.Image],
        output_frames: Sequence[Image.Image],
        frame_parameters: np.ndarray,
        frame_uncertainty: np.ndarray,
        shot: ShotPlan,
        instruction: str,
        anchor_grades: Sequence[AnchorGrade],
        hero_reference: Optional[HeroAnchorReference] = None,
    ) -> ShotCritique:
        reviews: list[tuple[CriticMember, ShotCritique]] = []
        for member in self.members:
            arguments = (
                source_frames,
                output_frames,
                frame_parameters,
                frame_uncertainty,
                shot,
                instruction,
                anchor_grades,
            )
            if "hero_reference" in inspect.signature(
                member.critic.evaluate
            ).parameters:
                result = member.critic.evaluate(
                    *arguments, hero_reference=hero_reference
                )
            else:
                result = member.critic.evaluate(*arguments)
            reviews.append((member, result))
        total_weight = sum(member.weight for member, _ in reviews)
        score = float(
            sum(member.weight * result.score for member, result in reviews)
            / total_weight
        )
        vetoed = any(member.veto and not result.accepted for member, result in reviews)
        accepted_weight = sum(
            member.weight for member, result in reviews if result.accepted
        )
        accepted = (
            not vetoed
            and score >= self.acceptance_score
            and accepted_weight >= total_weight * 0.5
        )
        metrics: dict[str, float] = {}
        reasons: list[str] = []
        recommendations: list[tuple[float, int]] = []
        metadata: dict[str, object] = {"members": {}}
        member_metadata = metadata["members"]
        assert isinstance(member_metadata, dict)
        for member, result in reviews:
            name = member.critic.name
            metrics.update(
                {f"{name}.{key}": value for key, value in result.metrics.items()}
            )
            if not result.accepted:
                reasons.extend(f"{name}:{reason}" for reason in result.reasons)
            if result.recommended_anchor is not None:
                recommendations.append(
                    (member.weight * (1.0 - result.score), result.recommended_anchor)
                )
            member_metadata[name] = {
                "weight": member.weight,
                "veto": member.veto,
                "score": result.score,
                "accepted": result.accepted,
                "metadata": result.metadata,
            }
        recommended = (
            max(recommendations, key=lambda value: value[0])[1]
            if recommendations
            else None
        )
        metadata.update(
            {
                "acceptance_score": self.acceptance_score,
                "accepted_weight_fraction": accepted_weight / total_weight,
                "vetoed": vetoed,
            }
        )
        return ShotCritique(
            score=score,
            accepted=accepted,
            metrics=metrics,
            reasons=tuple(dict.fromkeys(reasons)),
            recommended_anchor=recommended,
            metadata=metadata,
        )
