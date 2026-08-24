"""End-to-end v2 Agent pipeline built around typed grading Pools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from .clients import VisionLanguageClient
from .grade_pools import (
    POOL_OPERATION_TYPES,
    POOL_PROCESSING_ORDER,
    POOL_STAGE_TYPES,
    GradePoolExecutor,
    canonicalize_pool_parameters,
)
from .models import GradeGraph, ShotGrade, ShotPlan, StoryboardPlan
from .pool_propagation import (
    DYNAMIC_POOL_FIELDS,
    POOL_TEMPORAL_POLICIES,
    PoolParameterDiffuser,
    static_pool_consensus,
)
from .shot_planner import ShotPlanner
from .semantic_masks import SEMANTIC_MASK_TYPES, SemanticMaskGenerator
from .tasks import pool_grade_prompt, pool_review_prompt


@dataclass(frozen=True)
class PoolEditOperation:
    operation_id: str
    operation_type: str
    shot_id: int
    frame_range: tuple[int, int]
    keyframe: int
    parameters: dict[str, object]
    parameter_track: tuple[dict[str, object], ...] = ()
    temporal_policy: str = "shot_static"
    mask_id: str = "global"
    dependencies: tuple[str, ...] = ()
    confidence: float = 0.0
    provenance: dict[str, object] = field(default_factory=dict)

    def to_dict(self, include_parameter_track: bool = True) -> dict[str, object]:
        payload = {
            "operation_id": self.operation_id,
            "type": self.operation_type,
            "scope": {
                "shot_id": self.shot_id,
                "start_frame": self.frame_range[0],
                "end_frame": self.frame_range[1],
                "keyframe": self.keyframe,
            },
            "parameters": dict(self.parameters),
            "temporal_policy": self.temporal_policy,
            "mask": self.mask_id,
            "dependencies": list(self.dependencies),
            "confidence": self.confidence,
            "provenance": dict(self.provenance),
        }
        if include_parameter_track:
            payload["parameter_track"] = [
                dict(values) for values in self.parameter_track
            ]
        return payload


@dataclass(frozen=True)
class PoolAnchorResult:
    frame_index: int
    source: Image.Image
    preview: Image.Image
    operations: dict[str, dict[str, object]]
    operation_masks: dict[str, str]
    confidence: float
    accepted: bool
    audit: dict[str, object]


@dataclass(frozen=True)
class PoolPipelineResult:
    grade_graph: GradeGraph
    operations: tuple[PoolEditOperation, ...]
    audit: tuple[dict[str, object], ...]
    metadata: dict[str, object]


class PoolGradePipeline:
    """Storyboard, develop a Hero look, match shots, diffuse, and review."""

    name = "unified-pool-grade-agent/v2"

    def __init__(
        self,
        *,
        client: VisionLanguageClient,
        shot_planner: ShotPlanner,
        stages: Sequence[str] = tuple(POOL_STAGE_TYPES),
        review_enabled: bool = True,
        review_strict: bool = True,
        anchors_per_shot: int = 1,
        maximum_anchors_per_shot: int = 3,
        maximum_hero_attempts: int = 2,
        maximum_stage_attempts: int = 2,
        maximum_fidelity_l1: float = 0.42,
        maximum_clipping: float = 0.20,
        minimum_review_score: float = 0.60,
        transactional: bool = True,
        enabled_operations: Sequence[str] = tuple(POOL_OPERATION_TYPES),
    ) -> None:
        unknown_stages = set(stages) - set(POOL_STAGE_TYPES)
        if unknown_stages:
            raise ValueError(f"Unknown Pool Agent stages: {sorted(unknown_stages)}")
        if anchors_per_shot < 1 or maximum_anchors_per_shot < anchors_per_shot:
            raise ValueError("Invalid Pool Anchor budget.")
        self.client = client
        self.shot_planner = shot_planner
        self.stages = tuple(stages)
        self.review_enabled = bool(review_enabled)
        self.review_strict = bool(review_strict)
        self.anchors_per_shot = int(anchors_per_shot)
        self.maximum_anchors_per_shot = int(maximum_anchors_per_shot)
        self.maximum_hero_attempts = int(maximum_hero_attempts)
        self.maximum_stage_attempts = int(maximum_stage_attempts)
        self.maximum_fidelity_l1 = float(maximum_fidelity_l1)
        self.maximum_clipping = float(maximum_clipping)
        self.minimum_review_score = float(minimum_review_score)
        self.transactional = bool(transactional)
        self.enabled_operations = frozenset(str(value) for value in enabled_operations)
        self.executor = GradePoolExecutor()
        self.diffuser = PoolParameterDiffuser()
        self.mask_generator = SemanticMaskGenerator()

    @staticmethod
    def _operation_payloads(
        operations: dict[str, dict[str, object]],
        operation_masks: Optional[dict[str, str]] = None,
    ) -> list[dict[str, object]]:
        return [
            {
                "type": operation_type,
                "parameters": parameters,
                "mask": (operation_masks or {}).get(operation_type, "global"),
            }
            for operation_type, parameters in operations.items()
        ]

    def _preview(
        self,
        source: Image.Image,
        operations: dict[str, dict[str, object]],
        operation_masks: dict[str, str],
        frame_index: int,
    ) -> Image.Image:
        nodes = tuple(
            PoolEditOperation(
                operation_id=f"preview-{operation_type}",
                operation_type=operation_type,
                shot_id=-1,
                frame_range=(frame_index, frame_index),
                keyframe=frame_index,
                parameters=parameters,
                mask_id=operation_masks.get(operation_type, "global"),
            )
            for operation_type, parameters in operations.items()
        )
        mask_ids = {
            node.mask_id for node in nodes if node.mask_id != "global"
        }
        masks = {
            mask_id: self.mask_generator.generate(source, mask_id)
            for mask_id in mask_ids
        }
        return self.executor.apply(
            source, nodes, frame_index=frame_index, masks=masks
        )

    @staticmethod
    def _metrics(source: Image.Image, preview: Image.Image) -> dict[str, float]:
        before = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
        after = np.asarray(preview.convert("RGB"), dtype=np.float32) / 255.0
        clipping = float(np.mean((after <= 0.005) | (after >= 0.995)))
        source_clipping = float(np.mean((before <= 0.005) | (before >= 0.995)))
        return {
            "fidelity_l1": float(np.mean(np.abs(after - before))),
            "perceptual_rms": float(np.sqrt(np.mean((after - before) ** 2))),
            "clipping": clipping,
            "added_clipping": max(0.0, clipping - source_clipping),
        }

    def _grade_anchor(
        self,
        source: Image.Image,
        instruction: str,
        frame_index: int,
        shot_id: int,
        *,
        hero: Optional[PoolAnchorResult] = None,
    ) -> PoolAnchorResult:
        source = source.convert("RGB")
        preview = source.copy()
        operations: dict[str, dict[str, object]] = {}
        operation_masks: dict[str, str] = {}
        records: list[dict[str, object]] = []
        confidences: list[float] = []
        try:
            for stage in self.stages:
                allowed = [
                    value
                    for value in POOL_STAGE_TYPES[stage]
                    if value in self.enabled_operations
                ]
                if not allowed:
                    continue
                labeled: list[tuple[str, Image.Image]] = []
                if hero is not None:
                    labeled.extend(
                        [
                            (f"Hero source frame {hero.frame_index}", hero.source),
                            (f"Hero accepted grade frame {hero.frame_index}", hero.preview),
                        ]
                    )
                labeled.extend(
                    [
                        (f"target source frame {frame_index}", source),
                        (f"target preview before {stage}", preview),
                    ]
                )
                errors: list[str] = []
                payload: Optional[dict[str, object]] = None
                parsed: dict[str, dict[str, object]] = {}
                for attempt in range(1, self.maximum_stage_attempts + 1):
                    repair = (
                        ""
                        if not errors
                        else "\nPrevious response was invalid: " + errors[-1]
                    )
                    try:
                        candidate = self.client.generate_json(
                            labeled,
                            pool_grade_prompt(
                                instruction,
                                stage,
                                allowed,
                                self._operation_payloads(operations, operation_masks),
                                hero_frame=None if hero is None else hero.frame_index,
                            )
                            + repair,
                        )
                        raw_operations = candidate.get("operations", [])
                        if not isinstance(raw_operations, list):
                            raise ValueError("Pool grade response requires operations list.")
                        seen: set[str] = set()
                        current: dict[str, dict[str, object]] = {}
                        current_masks: dict[str, str] = {}
                        for index, raw_operation in enumerate(raw_operations):
                            if not isinstance(raw_operation, dict):
                                raise ValueError(f"Pool operation {index} must be an object.")
                            operation_type = str(raw_operation.get("type", ""))
                            if operation_type not in allowed:
                                raise ValueError(
                                    f"Pool {operation_type!r} is not allowed at stage {stage}."
                                )
                            if operation_type in seen:
                                raise ValueError(f"Pool {operation_type} appears more than once.")
                            seen.add(operation_type)
                            mask_id = str(raw_operation.get("mask", "global"))
                            if mask_id != "global" and mask_id not in SEMANTIC_MASK_TYPES:
                                raise ValueError(
                                    f"Unknown semantic mask for {operation_type}: {mask_id}"
                                )
                            current[operation_type] = canonicalize_pool_parameters(
                                operation_type, raw_operation.get("parameters", {})
                            )
                            current_masks[operation_type] = mask_id
                        payload = candidate
                        parsed = current
                        records.append(
                            {
                                "stage": stage,
                                "attempt": attempt,
                                "diagnosis": candidate.get("diagnosis", []),
                                "operations": [
                                    {
                                        "type": key,
                                        "parameters": value,
                                        "mask": current_masks[key],
                                    }
                                    for key, value in parsed.items()
                                ],
                                "validation_errors": list(errors),
                            }
                        )
                        break
                    except Exception as error:
                        errors.append(f"{type(error).__name__}: {error}")
                if payload is None:
                    raise ValueError(
                        f"Pool stage {stage} failed validation: {' | '.join(errors)}"
                    )
                operations.update(parsed)
                operation_masks.update(current_masks)
                preview = self._preview(
                    source, operations, operation_masks, frame_index
                )
                confidence = float(payload.get("confidence", 0.0))
                confidences.append(float(np.clip(confidence, 0.0, 1.0)))

            metrics = self._metrics(source, preview)
            reasons: list[str] = []
            if metrics["fidelity_l1"] > self.maximum_fidelity_l1:
                reasons.append("pool_grade_too_strong")
            if metrics["clipping"] > self.maximum_clipping:
                reasons.append("pool_grade_clipping")
            review: Optional[dict[str, object]] = None
            if self.review_enabled and not reasons:
                labeled = []
                if hero is not None:
                    labeled.extend(
                        [
                            (f"Hero source frame {hero.frame_index}", hero.source),
                            (f"Hero accepted grade frame {hero.frame_index}", hero.preview),
                        ]
                    )
                labeled.extend(
                    [
                        (f"shot {shot_id} source frame {frame_index}", source),
                        (f"shot {shot_id} final Pool preview frame {frame_index}", preview),
                    ]
                )
                try:
                    review = self.client.generate_json(
                        labeled,
                        pool_review_prompt(
                            instruction,
                            shot_id,
                            self._operation_payloads(operations, operation_masks),
                            hero_frame=None if hero is None else hero.frame_index,
                        ),
                    )
                    accepted = review.get("accept")
                    if not isinstance(accepted, bool):
                        raise ValueError("Pool review accept must be a JSON boolean.")
                    score = float(review.get("score", 0.0))
                    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                        raise ValueError("Pool review score must be in [0,1].")
                    if not accepted or score < self.minimum_review_score:
                        reasons.append("pool_vl_review_rejected")
                    confidences.append(score)
                except Exception as error:
                    if self.review_strict:
                        raise
                    review = {"error": f"{type(error).__name__}: {error}"}
            return PoolAnchorResult(
                frame_index=frame_index,
                source=source,
                preview=preview,
                operations=operations,
                operation_masks=operation_masks,
                confidence=float(np.mean(confidences)) if confidences else 0.0,
                accepted=not reasons,
                audit={
                    "frame": frame_index,
                    "shot_id": shot_id,
                    "accepted": not reasons,
                    "metrics": metrics,
                    "reasons": reasons,
                    "stages": records,
                    "review": review,
                },
            )
        except Exception as error:
            return PoolAnchorResult(
                frame_index=frame_index,
                source=source,
                preview=source.copy(),
                operations={},
                operation_masks={},
                confidence=0.0,
                accepted=False,
                audit={
                    "frame": frame_index,
                    "shot_id": shot_id,
                    "accepted": False,
                    "reasons": ["pool_anchor_failed"],
                    "error": f"{type(error).__name__}: {error}",
                    "stages": records,
                },
            )

    @staticmethod
    def _shot_for_frame(storyboard: StoryboardPlan, frame_index: int) -> ShotPlan:
        return next(
            shot
            for shot in storyboard.shots
            if shot.start_frame <= frame_index <= shot.end_frame
        )

    def _combine_shot(
        self,
        frames: Sequence[Image.Image],
        shot: ShotPlan,
        anchors: Sequence[PoolAnchorResult],
    ) -> tuple[PoolEditOperation, ...]:
        present_types = {
            operation_type
            for anchor in anchors
            for operation_type in anchor.operations
        }
        operations: list[PoolEditOperation] = []
        for operation_type in sorted(
            present_types,
            key=lambda value: list(POOL_PROCESSING_ORDER).index(value)
            if value in POOL_PROCESSING_ORDER
            else 999,
        ):
            neutral = canonicalize_pool_parameters(operation_type, {})
            values = [anchor.operations.get(operation_type, neutral) for anchor in anchors]
            base = static_pool_consensus(operation_type, values)
            mask_candidates = [
                anchor.operation_masks.get(operation_type, "global")
                for anchor in anchors
                if operation_type in anchor.operations
            ]
            mask_id = (
                max(set(mask_candidates), key=mask_candidates.count)
                if mask_candidates
                else "global"
            )
            track: tuple[dict[str, object], ...] = ()
            if operation_type in DYNAMIC_POOL_FIELDS:
                track = self.diffuser.diffuse(
                    frames,
                    shot,
                    operation_type,
                    [
                        (anchor.frame_index, anchor.operations.get(operation_type, neutral))
                        for anchor in anchors
                    ],
                )
            operations.append(
                PoolEditOperation(
                    operation_id=f"shot-{shot.shot_id}-{operation_type}",
                    operation_type=operation_type,
                    shot_id=shot.shot_id,
                    frame_range=(shot.start_frame, shot.end_frame),
                    keyframe=anchors[0].frame_index,
                    parameters=base,
                    parameter_track=track,
                    temporal_policy=POOL_TEMPORAL_POLICIES[operation_type],
                    mask_id=mask_id,
                    confidence=float(np.mean([anchor.confidence for anchor in anchors])),
                    provenance={
                        "backend": self.name,
                        "planner": "shared-vl-pool-colorist/v2",
                        "anchor_frames": [anchor.frame_index for anchor in anchors],
                        "editable": True,
                    },
                )
            )
        return tuple(operations)

    def _review_shot(
        self,
        frames: Sequence[Image.Image],
        shot: ShotPlan,
        operations: Sequence[PoolEditOperation],
        instruction: str,
        hero: PoolAnchorResult,
    ) -> tuple[bool, dict[str, object]]:
        sample_indices = tuple(
            sorted({shot.start_frame, *shot.anchor_frames, shot.end_frame})
        )
        sources = [frames[index].convert("RGB") for index in sample_indices]
        mask_ids = {
            operation.mask_id
            for operation in operations
            if operation.mask_id != "global"
        }
        outputs = []
        for index, source in zip(sample_indices, sources):
            masks = {
                mask_id: self.mask_generator.generate(source, mask_id)
                for mask_id in mask_ids
            }
            outputs.append(
                self.executor.apply(
                    source,
                    operations,
                    frame_index=index,
                    masks=masks,
                )
            )
        metrics = [self._metrics(source, output) for source, output in zip(sources, outputs)]
        residual_means = []
        for source, output in zip(sources, outputs):
            before = np.asarray(source, dtype=np.float32) / 255.0
            after = np.asarray(output, dtype=np.float32) / 255.0
            residual_means.append(np.mean(after - before, axis=(0, 1)))
        temporal_error = (
            0.0
            if len(residual_means) < 2
            else float(np.mean(np.abs(np.diff(np.asarray(residual_means), axis=0))))
        )
        summary = {
            "mean_fidelity_l1": float(np.mean([item["fidelity_l1"] for item in metrics])),
            "maximum_clipping": float(max(item["clipping"] for item in metrics)),
            "temporal_residual_error": temporal_error,
        }
        reasons: list[str] = []
        if summary["mean_fidelity_l1"] > self.maximum_fidelity_l1:
            reasons.append("pool_shot_too_strong")
        if summary["maximum_clipping"] > self.maximum_clipping:
            reasons.append("pool_shot_clipping")
        if temporal_error > 0.10:
            reasons.append("pool_temporal_inconsistency")
        review: Optional[dict[str, object]] = None
        if self.review_enabled and not reasons:
            labeled: list[tuple[str, Image.Image]] = [
                (f"Hero source frame {hero.frame_index}", hero.source),
                (f"Hero accepted grade frame {hero.frame_index}", hero.preview),
            ]
            for frame_index, source, output in zip(sample_indices, sources, outputs):
                labeled.extend(
                    [
                        (f"shot {shot.shot_id} source frame {frame_index}", source),
                        (f"shot {shot.shot_id} Pool grade frame {frame_index}", output),
                    ]
                )
            try:
                review = self.client.generate_json(
                    labeled,
                    pool_review_prompt(
                        instruction,
                        shot.shot_id,
                        [
                            operation.to_dict(include_parameter_track=False)
                            for operation in operations
                        ],
                        hero_frame=hero.frame_index,
                    ),
                )
                accept = review.get("accept")
                score = float(review.get("score", 0.0))
                if not isinstance(accept, bool):
                    raise ValueError("Pool shot review accept must be a JSON boolean.")
                if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError("Pool shot review score must be in [0,1].")
                if not accept or score < self.minimum_review_score:
                    reasons.append("pool_shot_vl_review_rejected")
            except Exception as error:
                if self.review_strict:
                    reasons.append("pool_shot_review_failed")
                review = {"error": f"{type(error).__name__}: {error}"}
        return not reasons, {
            "sample_frames": list(sample_indices),
            "metrics": summary,
            "reasons": reasons,
            "vl_review": review,
        }

    def _identity_graph(
        self,
        storyboard: StoryboardPlan,
        instruction: str,
        accepted: dict[int, bool],
        reasons: dict[int, Optional[str]],
    ) -> GradeGraph:
        shots = []
        for shot in storyboard.shots:
            length = shot.end_frame - shot.start_frame + 1
            is_accepted = bool(accepted.get(shot.shot_id, False))
            shots.append(
                ShotGrade(
                    shot=shot,
                    base_parameters=np.zeros(12, dtype=np.float64),
                    parameter_keyframes={},
                    frame_parameters=np.zeros((length, 12), dtype=np.float64),
                    confidence=1.0 if is_accepted else 0.0,
                    accepted=is_accepted,
                    rolled_back=not is_accepted,
                    rollback_reason=reasons.get(shot.shot_id),
                    attempts=(),
                    search_memory={
                        "schema": "pool-graph/v2",
                        "legacy_12d_parameters_used": False,
                    },
                )
            )
        return GradeGraph(
            instruction=instruction,
            storyboard=storyboard,
            shots=tuple(shots),
            frame_parameters=np.zeros((storyboard.frame_count, 12), dtype=np.float64),
            backend="unified-vl-video/pool-v2",
            critic="pool-deterministic-safety+vl-review",
            orchestrator=self.name,
            hero_anchor=None,
        )

    def run(
        self,
        frames: Sequence[Image.Image],
        fps: float,
        instruction: str,
        *,
        reference_frames: Optional[Sequence[Image.Image]] = None,
        reference_fps: Optional[float] = None,
    ) -> PoolPipelineResult:
        target = tuple(frame.convert("RGB") for frame in frames)
        storyboard = self.shot_planner.plan(
            target, fps, instruction, anchors_per_shot=self.anchors_per_shot
        )
        external = reference_frames is not None
        hero_frames = (
            tuple(frame.convert("RGB") for frame in reference_frames)
            if external
            else target
        )
        hero_fps = float(reference_fps) if external else fps
        hero_storyboard = (
            self.shot_planner.plan(
                hero_frames,
                hero_fps,
                instruction,
                anchors_per_shot=self.anchors_per_shot,
            )
            if external
            else storyboard
        )
        hero_candidates = list(hero_storyboard.hero_anchor_candidates)
        if hero_storyboard.hero_anchor_frame is not None:
            hero_candidates.insert(0, hero_storyboard.hero_anchor_frame)
        hero_candidates = list(dict.fromkeys(hero_candidates))
        if not hero_candidates:
            hero_candidates = [hero_storyboard.shots[0].anchor_frames[0]]
        hero: Optional[PoolAnchorResult] = None
        hero_attempts: list[dict[str, object]] = []
        for frame_index in hero_candidates[: self.maximum_hero_attempts]:
            hero_shot = self._shot_for_frame(hero_storyboard, frame_index)
            candidate = self._grade_anchor(
                hero_frames[frame_index],
                instruction,
                frame_index,
                hero_shot.shot_id,
            )
            hero_attempts.append(candidate.audit)
            if candidate.accepted:
                hero = candidate
                break

        operations: list[PoolEditOperation] = []
        audits: list[dict[str, object]] = []
        accepted: dict[int, bool] = {}
        reasons: dict[int, Optional[str]] = {}
        if hero is not None:
            for shot in storyboard.shots:
                candidate_frames = list(
                    dict.fromkeys([*shot.anchor_frames, *shot.anchor_candidates])
                )[: self.maximum_anchors_per_shot]
                anchor_results: list[PoolAnchorResult] = []
                accepted_anchors: list[PoolAnchorResult] = []
                shot_operations: tuple[PoolEditOperation, ...] = ()
                shot_review: dict[str, object] | None = None
                shot_accepted = False
                for frame_index in candidate_frames:
                    if not external and frame_index == hero.frame_index:
                        result = hero
                    else:
                        result = self._grade_anchor(
                            target[frame_index],
                            instruction,
                            frame_index,
                            shot.shot_id,
                            hero=hero,
                        )
                    anchor_results.append(result)
                    if not result.accepted:
                        continue
                    accepted_anchors.append(result)
                    if len(accepted_anchors) < self.anchors_per_shot:
                        continue
                    shot_operations = self._combine_shot(
                        target, shot, accepted_anchors
                    )
                    shot_accepted, shot_review = self._review_shot(
                        target,
                        shot,
                        shot_operations,
                        instruction,
                        hero,
                    )
                    if shot_accepted:
                        break
                    shot_operations = ()
                accepted[shot.shot_id] = shot_accepted
                reasons[shot.shot_id] = (
                    None if shot_accepted else "pool_anchor_rejected"
                )
                if shot_accepted:
                    operations.extend(shot_operations)
                audits.append(
                    {
                        "shot_id": shot.shot_id,
                        "accepted": shot_accepted,
                        "rolled_back": not shot_accepted,
                        "anchor_attempts": [anchor.audit for anchor in anchor_results],
                        "shot_review": shot_review,
                        "operations": [
                            operation.to_dict(include_parameter_track=False)
                            for operation in shot_operations
                        ],
                    }
                )
        else:
            for shot in storyboard.shots:
                accepted[shot.shot_id] = False
                reasons[shot.shot_id] = "hero_pool_grade_rejected"

        globally_accepted = hero is not None and all(accepted.values())
        if self.transactional and not globally_accepted:
            operations = []
            for shot in storyboard.shots:
                accepted[shot.shot_id] = False
                reasons[shot.shot_id] = reasons.get(shot.shot_id) or "global_pool_rollback"
            for audit in audits:
                audit["global_rollback"] = True

        graph = self._identity_graph(storyboard, instruction, accepted, reasons)
        return PoolPipelineResult(
            grade_graph=graph,
            operations=tuple(operations),
            audit=tuple(audits),
            metadata={
                "schema_version": "pool-grade-graph/v2",
                "legacy_12d_parameters_used": False,
                "hero": None
                if hero is None
                else {
                    "frame": hero.frame_index,
                    "source_video": "reference_video" if external else "target_video",
                    "confidence": hero.confidence,
                    "operations": self._operation_payloads(
                        hero.operations, hero.operation_masks
                    ),
                },
                "hero_attempts": hero_attempts,
                "transactional": self.transactional,
                "globally_accepted": globally_accepted,
            },
        )
