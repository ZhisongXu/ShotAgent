"""Single public backend for the complete VL video-grading workflow.

The unified backend deliberately exposes one provider/model and one editing
backend to callers.  Storyboarding, Anchor look development, visual review,
deterministic safety checks, parameter diffusion, and rollback remain separate
internal stages so they can be tested without becoming public backends.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from retouch_agent.parameters import PARAMETER_NAMES, RetouchParameters

from .agent_config import SearchSettings, build_vision_client
from .backends import VLAnchorBackend
from .clients import VisionLanguageClient
from .critic import (
    CriticEnsemble,
    CriticMember,
    PhotoAgentStyleCritic,
    ShotCritic,
    VisionReviewCritic,
)
from .models import GradeGraph
from .operations import OperationExecutor
from .pipeline import DynamicGradePipeline
from .shot_planner import LongVideoStoryboardSettings, VLShotPlanner
from .tasks import operation_plan_prompt, operation_review_prompt


SUPPORTED_OPERATIONS = frozenset({"global_grade", "tone_curve", "hsl_grade", "lut"})
RESOLVE_OPERATIONS = frozenset({"global_grade"})
KNOWN_OPERATIONS = frozenset(
    {
        "global_grade",
        "tone_curve",
        "hsl_grade",
        "masked_grade",
        "lut",
        "denoise",
        "deblur",
        "generative_edit",
    }
)


@dataclass(frozen=True)
class VideoEditRequest:
    frames: Sequence[Image.Image]
    fps: float
    instruction: str
    reference_frames: Optional[Sequence[Image.Image]] = None
    reference_fps: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("VideoEditRequest requires at least one frame.")
        if self.fps <= 0.0:
            raise ValueError("VideoEditRequest fps must be positive.")
        if not self.instruction.strip():
            raise ValueError("VideoEditRequest instruction cannot be empty.")
        if self.reference_frames is not None and not self.reference_frames:
            raise ValueError("Reference video must contain at least one frame.")
        if self.reference_frames is not None and (
            self.reference_fps is None or self.reference_fps <= 0.0
        ):
            raise ValueError(
                "reference_fps must be positive when reference_frames are supplied."
            )


@dataclass(frozen=True)
class EditOperation:
    operation_id: str
    operation_type: str
    shot_id: int
    frame_range: tuple[int, int]
    keyframe: int
    parameters: dict[str, object]
    dependencies: tuple[str, ...] = ()
    confidence: float = 0.0
    provenance: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id,
            "type": self.operation_type,
            "scope": {
                "shot_id": self.shot_id,
                "start_frame": self.frame_range[0],
                "end_frame": self.frame_range[1],
                "keyframe": self.keyframe,
            },
            "parameters": dict(self.parameters),
            "dependencies": list(self.dependencies),
            "confidence": self.confidence,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class UnifiedVideoEditResult:
    grade_graph: GradeGraph
    operations: tuple[EditOperation, ...]
    runtime_manifest: dict[str, object]
    operation_audit: tuple[dict[str, object], ...] = ()

    def to_dict(self, include_frame_parameters: bool = True) -> dict[str, object]:
        payload = self.grade_graph.to_dict(
            include_frame_parameters=include_frame_parameters
        )
        runtime_operations = self.runtime_manifest.get("operations", {})
        enabled_operations = (
            runtime_operations.get("enabled", [])
            if isinstance(runtime_operations, dict)
            else []
        )
        payload["operation_graph"] = {
            "schema_version": "video-edit-operation-graph/v1",
            "supported_operations": sorted(SUPPORTED_OPERATIONS),
            "enabled_operations": list(enabled_operations),
            "resolve_supported_operations": sorted(RESOLVE_OPERATIONS),
            "operations": [operation.to_dict() for operation in self.operations],
            "audit": list(self.operation_audit),
        }
        payload["backend_runtime"] = self.runtime_manifest
        return payload


def _object(value: object, field_name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object.")
    return value


def _enabled_operations(config: dict[str, object]) -> tuple[str, ...]:
    raw = config.get("operations", {"global_grade": True})
    operations = _object(raw, "backend.operations")
    unknown = set(operations) - KNOWN_OPERATIONS
    if unknown:
        raise ValueError(
            "Unknown unified backend operations: " + ", ".join(sorted(unknown))
        )
    non_boolean = [
        name for name, value in operations.items() if not isinstance(value, bool)
    ]
    if non_boolean:
        raise ValueError(
            "Unified backend operation flags must be JSON booleans: "
            + ", ".join(sorted(non_boolean))
        )
    enabled = tuple(
        name for name in sorted(KNOWN_OPERATIONS) if bool(operations.get(name, False))
    )
    unsupported = set(enabled) - SUPPORTED_OPERATIONS
    if unsupported:
        raise ValueError(
            "Operations are configured but not implemented in operation-graph/v1: "
            + ", ".join(sorted(unsupported))
        )
    if "global_grade" not in enabled:
        raise ValueError("The v1 unified backend requires global_grade.")
    return enabled


def _search_settings(config: dict[str, object]) -> SearchSettings:
    raw = _object(config.get("search"), "backend.search")
    return SearchSettings(
        maximum_evaluations=int(raw.get("maximum_evaluations", 12)),
        maximum_hero_attempts=int(raw.get("maximum_hero_attempts", 3)),
        exploration_constant=float(raw.get("exploration_constant", 1.41421356237)),
        rejection_penalty=float(raw.get("rejection_penalty", 0.25)),
        seed=int(raw.get("seed", 7)),
    )


def _editor_settings(config: dict[str, object]) -> dict[str, object]:
    editor = dict(_object(config.get("editor"), "backend.editor"))
    raw_stages = editor.get("stages", ["lighting", "white_balance_and_color", "tone"])
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("backend.editor.stages must be a non-empty list.")
    editor["stages"] = [str(stage) for stage in raw_stages]
    if int(editor.get("candidate_count", 16)) < 1:
        raise ValueError("backend.editor.candidate_count must be positive.")
    return editor


def _lut_catalog(
    config: dict[str, object],
    config_root: Path,
) -> dict[str, Path]:
    raw = _object(config.get("lut_catalog"), "backend.lut_catalog")
    root = Path(config_root).resolve()
    catalog: dict[str, Path] = {}
    for raw_name, raw_path in raw.items():
        name = str(raw_name).strip()
        path = Path(str(raw_path))
        if not name:
            raise ValueError("LUT catalog IDs cannot be empty.")
        if path.is_absolute():
            raise ValueError("LUT catalog paths must be relative to the config file.")
        resolved = (root / path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(
                "LUT catalog paths cannot escape the config directory."
            ) from error
        if not resolved.is_file():
            raise ValueError(f"Configured LUT does not exist: {resolved}")
        catalog[name] = resolved
    return catalog


def _operation_policy(config: dict[str, object]) -> dict[str, object]:
    raw = _object(config.get("operation_policy"), "backend.operation_policy")
    allowed = {
        "maximum_operations_per_shot",
        "maximum_additional_fidelity_l1",
        "maximum_added_clipping",
        "minimum_review_score",
        "maximum_planning_attempts",
        "strict",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown operation_policy fields: {sorted(unknown)}")
    maximum_operations = int(raw.get("maximum_operations_per_shot", 3))
    if not 1 <= maximum_operations <= 8:
        raise ValueError("maximum_operations_per_shot must be in [1,8].")
    maximum_delta = float(raw.get("maximum_additional_fidelity_l1", 0.12))
    maximum_clipping = float(raw.get("maximum_added_clipping", 0.03))
    minimum_score = float(raw.get("minimum_review_score", 0.60))
    maximum_planning_attempts = int(raw.get("maximum_planning_attempts", 2))
    if not 0.0 <= maximum_delta <= 1.0:
        raise ValueError("maximum_additional_fidelity_l1 must be in [0,1].")
    if not 0.0 <= maximum_clipping <= 1.0:
        raise ValueError("maximum_added_clipping must be in [0,1].")
    if not 0.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_review_score must be in [0,1].")
    if not 1 <= maximum_planning_attempts <= 3:
        raise ValueError("maximum_planning_attempts must be in [1,3].")
    strict = raw.get("strict", False)
    if not isinstance(strict, bool):
        raise ValueError("operation_policy.strict must be a JSON boolean.")
    return {
        "maximum_operations_per_shot": maximum_operations,
        "maximum_additional_fidelity_l1": maximum_delta,
        "maximum_added_clipping": maximum_clipping,
        "minimum_review_score": minimum_score,
        "maximum_planning_attempts": maximum_planning_attempts,
        "strict": strict,
    }


def _build_critic(
    client: VisionLanguageClient,
    config: dict[str, object],
) -> ShotCritic:
    safety = _object(config.get("safety"), "backend.safety")
    metric_critic = PhotoAgentStyleCritic(
        use_vl_review=False,
        maximum_fidelity_l1=float(safety.get("maximum_fidelity_l1", 0.42)),
        maximum_clipping=float(safety.get("maximum_clipping", 0.20)),
        maximum_temporal_error=float(safety.get("maximum_temporal_error", 0.10)),
        maximum_parameter_jerk=float(safety.get("maximum_parameter_jerk", 0.20)),
        maximum_anchor_error=float(safety.get("maximum_anchor_error", 0.12)),
    )
    metric_critic.name = "unified-temporal-safety"

    review = _object(config.get("review"), "backend.review")
    if not bool(review.get("enabled", True)):
        return metric_critic
    visual_critic = VisionReviewCritic(
        client=client,
        name="unified-vl-review",
        focus=str(
            review.get(
                "focus",
                "instruction adherence, preservation, professional color, and "
                "cross-frame coherence",
            )
        ),
        strict=bool(review.get("strict", True)),
        fallback_score=float(review.get("fallback_score", 0.0)),
    )
    metric_weight = float(review.get("metric_weight", 0.55))
    visual_weight = float(review.get("visual_weight", 0.45))
    return CriticEnsemble(
        (
            CriticMember(metric_critic, weight=metric_weight, veto=True),
            CriticMember(visual_critic, weight=visual_weight, veto=False),
        ),
        acceptance_score=float(review.get("acceptance_score", 0.60)),
    )


class UnifiedVLVideoBackend:
    """Own the complete video-grade workflow behind one public backend."""

    name = "unified-vl-video"

    def __init__(
        self,
        *,
        client: VisionLanguageClient,
        storyboard_settings: LongVideoStoryboardSettings,
        editor_settings: dict[str, object],
        critic: ShotCritic,
        search: SearchSettings,
        enabled_operations: Sequence[str],
        allow_storyboard_fallback: bool,
        anchors_per_shot: int = 1,
        maximum_anchors_per_shot: int = 3,
        maximum_evaluations: Optional[int] = None,
        operation_executor: Optional[OperationExecutor] = None,
        operation_policy: Optional[dict[str, object]] = None,
        visual_operation_review: bool = True,
        manifest: Optional[dict[str, object]] = None,
    ) -> None:
        self.client = client
        self.enabled_operations = tuple(enabled_operations)
        self.search_settings = search
        self.operation_executor = operation_executor or OperationExecutor()
        self.operation_policy = operation_policy or {}
        self.visual_operation_review = bool(visual_operation_review)
        self.editor = VLAnchorBackend(
            client,
            stages=tuple(
                str(stage)
                for stage in editor_settings.get(
                    "stages", ["lighting", "white_balance_and_color", "tone"]
                )
            ),
            candidate_count=int(editor_settings.get("candidate_count", 16)),
            seed=int(editor_settings.get("seed", search.seed)),
            name=self.name,
            use_mkl_prior=bool(editor_settings.get("use_mkl_prior", True)),
            mkl_strength=float(editor_settings.get("mkl_strength", 0.35)),
            mkl_projection_iterations=int(
                editor_settings.get("mkl_projection_iterations", 40)
            ),
        )
        self.critic = critic
        self.shot_planner = VLShotPlanner(
            client=client,
            settings=storyboard_settings,
            strict=not allow_storyboard_fallback,
        )
        self.pipeline = DynamicGradePipeline(
            shot_planner=self.shot_planner,
            anchor_backend=self.editor,
            critic=critic,
            anchors_per_shot=anchors_per_shot,
            maximum_anchors_per_shot=maximum_anchors_per_shot,
            maximum_attempts=(
                search.maximum_evaluations
                if maximum_evaluations is None
                else maximum_evaluations
            ),
            maximum_hero_attempts=search.maximum_hero_attempts,
            mcts_exploration=search.exploration_constant,
            mcts_rejection_penalty=search.rejection_penalty,
            mcts_seed=search.seed,
        )
        # This search still performs Anchor replacement and rollback, but no
        # longer selects among public editor backends.
        self.pipeline.search.name = "unified-anchor-trajectory-search/v1"
        self.runtime_manifest = manifest or {}

    @property
    def executor(self):
        return self.pipeline.executor

    @staticmethod
    def _global_operations(graph: GradeGraph) -> tuple[EditOperation, ...]:
        operations: list[EditOperation] = []
        for shot_grade in graph.shots:
            if not shot_grade.accepted or shot_grade.rolled_back:
                continue
            for frame_index, values in sorted(shot_grade.parameter_keyframes.items()):
                operations.append(
                    EditOperation(
                        operation_id=(
                            f"shot-{shot_grade.shot.shot_id}-frame-{frame_index}-"
                            "global-grade"
                        ),
                        operation_type="global_grade",
                        shot_id=shot_grade.shot.shot_id,
                        frame_range=(
                            shot_grade.shot.start_frame,
                            shot_grade.shot.end_frame,
                        ),
                        keyframe=frame_index,
                        parameters={
                            name: float(value)
                            for name, value in zip(PARAMETER_NAMES, values)
                        },
                        confidence=shot_grade.confidence,
                        provenance={
                            "backend": graph.backend,
                            "propagation": "bayesian-parameter-diffusion",
                            "editable": True,
                        },
                    )
                )
        return tuple(operations)

    def _review_operations(
        self,
        *,
        samples: Sequence[tuple[int, Image.Image, Image.Image, Image.Image]],
        instruction: str,
        shot_id: int,
        operations: Sequence[EditOperation],
    ) -> tuple[bool, dict[str, object]]:
        base = (
            np.stack(
                [
                    np.asarray(global_preview.convert("RGB"), dtype=np.float32)
                    for _, _, global_preview, _ in samples
                ]
            )
            / 255.0
        )
        enhanced = (
            np.stack(
                [
                    np.asarray(enhanced_preview.convert("RGB"), dtype=np.float32)
                    for _, _, _, enhanced_preview in samples
                ]
            )
            / 255.0
        )
        fidelity = float(np.mean(np.abs(enhanced - base)))
        base_clipping = float(np.mean((base <= 0.005) | (base >= 0.995)))
        enhanced_clipping = float(np.mean((enhanced <= 0.005) | (enhanced >= 0.995)))
        added_clipping = max(0.0, enhanced_clipping - base_clipping)
        maximum_delta = float(
            self.operation_policy.get("maximum_additional_fidelity_l1", 0.12)
        )
        maximum_added_clipping = float(
            self.operation_policy.get("maximum_added_clipping", 0.03)
        )
        reasons = []
        if fidelity > maximum_delta:
            reasons.append("additional_edit_too_strong")
        if added_clipping > maximum_added_clipping:
            reasons.append("additional_clipping")
        review: dict[str, object] | None = None
        if self.visual_operation_review and not reasons:
            labeled_images: list[tuple[str, Image.Image]] = []
            for frame_index, source, global_preview, enhanced_preview in samples:
                labeled_images.extend(
                    [
                        (f"source frame {frame_index}", source),
                        (
                            f"accepted global preview frame {frame_index}",
                            global_preview,
                        ),
                        (
                            f"preview after additional operations frame {frame_index}",
                            enhanced_preview,
                        ),
                    ]
                )
            review = self.client.generate_json(
                labeled_images,
                operation_review_prompt(
                    instruction,
                    shot_id,
                    [operation.to_dict() for operation in operations],
                ),
            )
            accepted = review.get("accept")
            if not isinstance(accepted, bool):
                raise ValueError("Operation review accept must be a JSON boolean.")
            score = float(review.get("score", 0.0))
            if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("Operation review score must be in [0,1].")
            minimum_score = float(
                self.operation_policy.get("minimum_review_score", 0.60)
            )
            if not accepted or score < minimum_score:
                reasons.append("vl_operation_review_rejected")
        return not reasons, {
            "metrics": {
                "additional_fidelity_l1": fidelity,
                "base_clipping": base_clipping,
                "enhanced_clipping": enhanced_clipping,
                "added_clipping": added_clipping,
            },
            "limits": {
                "maximum_additional_fidelity_l1": maximum_delta,
                "maximum_added_clipping": maximum_added_clipping,
            },
            "reasons": reasons,
            "vl_review": review,
        }

    def _plan_post_operations(
        self,
        request: VideoEditRequest,
        graph: GradeGraph,
    ) -> tuple[tuple[EditOperation, ...], tuple[dict[str, object], ...]]:
        enabled = [name for name in self.enabled_operations if name != "global_grade"]
        if not enabled:
            return (), ()
        maximum_operations = int(
            self.operation_policy.get("maximum_operations_per_shot", 3)
        )
        if maximum_operations < 1 or maximum_operations > 8:
            raise ValueError("maximum_operations_per_shot must be in [1,8].")
        strict = bool(self.operation_policy.get("strict", False))
        operations: list[EditOperation] = []
        audits: list[dict[str, object]] = []
        for shot_grade in graph.shots:
            shot = shot_grade.shot
            audit: dict[str, object] = {
                "shot_id": shot.shot_id,
                "accepted": False,
                "rolled_back": False,
            }
            if not shot_grade.accepted or shot_grade.rolled_back:
                audit.update(
                    {
                        "rolled_back": True,
                        "reasons": ["base_global_grade_not_accepted"],
                    }
                )
                audits.append(audit)
                continue
            keyframe = next(iter(sorted(shot_grade.parameter_keyframes)))
            sample_indices = tuple(sorted({shot.start_frame, keyframe, shot.end_frame}))
            audit["sample_frames"] = list(sample_indices)
            sample_pairs: list[tuple[int, Image.Image, Image.Image]] = []
            for sample_index in sample_indices:
                source = request.frames[sample_index].convert("RGB")
                sample_parameters = RetouchParameters.from_vector(
                    graph.frame_parameters[sample_index]
                )
                global_preview = self.executor.apply(source, sample_parameters)
                assert isinstance(global_preview, Image.Image)
                sample_pairs.append((sample_index, source, global_preview))
            parameters = RetouchParameters.from_vector(graph.frame_parameters[keyframe])
            try:
                planning_images: list[tuple[str, Image.Image]] = []
                for sample_index, source, global_preview in sample_pairs:
                    planning_images.extend(
                        [
                            (
                                f"shot {shot.shot_id} source frame {sample_index}",
                                source,
                            ),
                            (
                                f"shot {shot.shot_id} accepted global preview "
                                f"frame {sample_index}",
                                global_preview,
                            ),
                        ]
                    )
                base_prompt = operation_plan_prompt(
                    request.instruction,
                    shot.shot_id,
                    shot.start_frame,
                    shot.end_frame,
                    parameters.to_dict(),
                    enabled,
                    list(self.operation_executor.lut_ids),
                    maximum_operations,
                )
                maximum_planning_attempts = int(
                    self.operation_policy.get("maximum_planning_attempts", 2)
                )
                planning_errors: list[str] = []
                payload: dict[str, object] | None = None
                proposed: list[EditOperation] | None = None
                for planning_attempt in range(1, maximum_planning_attempts + 1):
                    repair = (
                        ""
                        if not planning_errors
                        else "\nPrevious response validation failed: "
                        + planning_errors[-1]
                        + "\nReturn a corrected JSON object only."
                    )
                    try:
                        candidate_payload = self.client.generate_json(
                            planning_images,
                            base_prompt + repair,
                        )
                        raw_operations = candidate_payload.get("operations", [])
                        if not isinstance(raw_operations, list):
                            raise ValueError(
                                "Operation plan requires an operations list."
                            )
                        if len(raw_operations) > maximum_operations:
                            raise ValueError(
                                "Operation plan exceeds maximum_operations_per_shot."
                            )
                        parsed: list[EditOperation] = []
                        for index, raw_operation in enumerate(raw_operations):
                            if not isinstance(raw_operation, dict):
                                raise ValueError(
                                    f"Operation {index} must be an object."
                                )
                            operation_type = str(raw_operation.get("type", ""))
                            if operation_type not in enabled:
                                raise ValueError(
                                    f"Operation {operation_type!r} is not enabled "
                                    "for this backend."
                                )
                            canonical = self.operation_executor.canonicalize(
                                operation_type, raw_operation.get("parameters")
                            )
                            confidence = float(raw_operation.get("confidence", 0.0))
                            if not np.isfinite(confidence):
                                raise ValueError("Operation confidence must be finite.")
                            parsed.append(
                                EditOperation(
                                    operation_id=(
                                        f"shot-{shot.shot_id}-post-{index}-"
                                        f"{operation_type}"
                                    ),
                                    operation_type=operation_type,
                                    shot_id=shot.shot_id,
                                    frame_range=(shot.start_frame, shot.end_frame),
                                    keyframe=keyframe,
                                    parameters=canonical,
                                    dependencies=(
                                        f"shot-{shot.shot_id}-frame-{keyframe}-"
                                        "global-grade",
                                    ),
                                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                                    provenance={
                                        "backend": self.name,
                                        "planner": "shared-vl-operation-planner/v1",
                                        "reason": str(raw_operation.get("reason", "")),
                                        "editable": True,
                                    },
                                )
                            )
                        payload = candidate_payload
                        proposed = parsed
                        audit["planning_attempts"] = planning_attempt
                        audit["planning_errors"] = list(planning_errors)
                        break
                    except Exception as error:
                        planning_errors.append(f"{type(error).__name__}: {error}")
                if payload is None or proposed is None:
                    raise ValueError(
                        "Operation planning failed validation after retries: "
                        + " | ".join(planning_errors)
                    )
                if not proposed:
                    audit.update(
                        {
                            "accepted": True,
                            "proposal_count": 0,
                            "diagnosis": payload.get("diagnosis", []),
                            "reasons": ["no_additional_operation_needed"],
                        }
                    )
                    audits.append(audit)
                    continue
                reviewed_samples = tuple(
                    (
                        sample_index,
                        source,
                        global_preview,
                        self.operation_executor.apply(
                            global_preview,
                            proposed,
                            frame_index=sample_index,
                        ),
                    )
                    for sample_index, source, global_preview in sample_pairs
                )
                accepted, review = self._review_operations(
                    samples=reviewed_samples,
                    instruction=request.instruction,
                    shot_id=shot.shot_id,
                    operations=proposed,
                )
                audit.update(
                    {
                        "accepted": accepted,
                        "rolled_back": not accepted,
                        "proposal_count": len(proposed),
                        "proposed_operations": [
                            operation.to_dict() for operation in proposed
                        ],
                        "diagnosis": payload.get("diagnosis", []),
                        "review": review,
                    }
                )
                if accepted:
                    operations.extend(proposed)
                audits.append(audit)
            except Exception as error:
                audit.update(
                    {
                        "rolled_back": True,
                        "reasons": ["operation_planning_failed"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                audits.append(audit)
                if strict:
                    raise
        return tuple(operations), tuple(audits)

    def process(self, request: VideoEditRequest) -> UnifiedVideoEditResult:
        graph = self.pipeline.run(
            request.frames,
            request.fps,
            request.instruction,
            reference_frames=request.reference_frames,
            reference_fps=request.reference_fps,
        )
        post_operations, operation_audit = self._plan_post_operations(request, graph)
        return UnifiedVideoEditResult(
            grade_graph=graph,
            operations=(*self._global_operations(graph), *post_operations),
            runtime_manifest=self.runtime_manifest,
            operation_audit=operation_audit,
        )


def build_unified_backend(
    payload: object,
    *,
    allow_storyboard_fallback: bool = False,
    anchors_per_shot: int = 1,
    maximum_anchors_per_shot: int = 3,
    maximum_evaluations: Optional[int] = None,
    client: Optional[VisionLanguageClient] = None,
    config_root: Optional[Path] = None,
) -> UnifiedVLVideoBackend:
    if not isinstance(payload, dict):
        raise ValueError("Unified backend config must be a JSON object.")
    if any(name in payload for name in ("editors", "evaluators")):
        raise ValueError(
            "Unified backend config accepts one backend object, not editor or "
            "evaluator pools."
        )
    backend = _object(payload.get("backend"), "backend")
    backend_type = str(backend.get("type", "unified_vl_video"))
    if backend_type != "unified_vl_video":
        raise ValueError(f"Unsupported unified backend type: {backend_type}")

    enabled_operations = _enabled_operations(backend)
    shared_client = client or build_vision_client(backend, "unified backend")
    lut_catalog = _lut_catalog(backend, config_root or Path.cwd())
    if "lut" in enabled_operations and not lut_catalog:
        raise ValueError("The lut operation requires at least one LUT catalog entry.")
    operation_executor = OperationExecutor(lut_catalog)
    storyboard = _object(backend.get("storyboard"), "backend.storyboard")
    storyboard_settings = LongVideoStoryboardSettings.from_dict(
        storyboard.get("long_video")
    )
    editor = _editor_settings(backend)
    search = _search_settings(backend)
    critic = _build_critic(shared_client, backend)
    fallback_enabled = bool(storyboard.get("allow_fallback", False)) or bool(
        allow_storyboard_fallback
    )
    review_enabled = bool(
        _object(backend.get("review"), "backend.review").get("enabled", True)
    )
    operation_policy = _operation_policy(backend)
    roles = ["storyboard", "editor"]
    if review_enabled:
        roles.append("review")
    manifest = {
        "mode": "unified-single-backend/v1",
        "backend": {
            "name": UnifiedVLVideoBackend.name,
            "type": backend_type,
            "provider": backend.get("provider"),
            "model": shared_client.model_id,
            "single_client_for_roles": True,
            "roles": roles,
        },
        "operations": {
            "schema_version": "video-edit-operation-graph/v1",
            "enabled": list(enabled_operations),
            "implemented": sorted(SUPPORTED_OPERATIONS),
            "resolve_supported": sorted(RESOLVE_OPERATIONS),
            "lut_ids": list(operation_executor.lut_ids),
            "policy": dict(operation_policy),
        },
        "storyboard": {
            "planner": "hierarchical-vision-storyboard/v2",
            "allow_fallback": fallback_enabled,
            "long_video": {
                field: getattr(storyboard_settings, field)
                for field in storyboard_settings.__dataclass_fields__
            },
        },
        "editor": {
            "name": UnifiedVLVideoBackend.name,
            "stages": list(
                editor.get("stages", ["lighting", "white_balance_and_color", "tone"])
            ),
            "candidate_count": int(editor.get("candidate_count", 16)),
            "use_mkl_prior": bool(editor.get("use_mkl_prior", True)),
        },
        "review": {
            "visual_review": review_enabled,
            "deterministic_safety_veto": True,
        },
        "search": {
            "algorithm": "unified-anchor-trajectory-search/v1",
            "maximum_evaluations": (
                search.maximum_evaluations
                if maximum_evaluations is None
                else maximum_evaluations
            ),
            "maximum_hero_attempts": search.maximum_hero_attempts,
            "seed": search.seed,
        },
    }
    return UnifiedVLVideoBackend(
        client=shared_client,
        storyboard_settings=storyboard_settings,
        editor_settings=editor,
        critic=critic,
        search=search,
        enabled_operations=enabled_operations,
        allow_storyboard_fallback=fallback_enabled,
        anchors_per_shot=anchors_per_shot,
        maximum_anchors_per_shot=maximum_anchors_per_shot,
        maximum_evaluations=maximum_evaluations,
        operation_executor=operation_executor,
        operation_policy=operation_policy,
        visual_operation_review=review_enabled,
        manifest=manifest,
    )


def load_unified_backend(
    path: Path,
    **kwargs,
) -> UnifiedVLVideoBackend:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return build_unified_backend(payload, config_root=path.parent, **kwargs)
