"""Serializable data structures for the video grade graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from retouch_agent.parameters import (
    PARAMETER_LOWER_BOUNDS,
    PARAMETER_NAMES,
    PARAMETER_UPPER_BOUNDS,
)


@dataclass(frozen=True)
class ShotPlan:
    shot_id: int
    start_frame: int
    end_frame: int
    anchor_frames: tuple[int, ...]
    anchor_candidates: tuple[int, ...] = ()
    description: str = ""
    selection_reason: str = ""

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame < self.start_frame:
            raise ValueError("Invalid shot frame range.")
        if not self.anchor_frames:
            raise ValueError("Every shot requires at least one Anchor.")
        if any(
            index < self.start_frame or index > self.end_frame
            for index in self.anchor_frames
        ):
            raise ValueError("Anchor lies outside its shot.")
        if any(
            index < self.start_frame or index > self.end_frame
            for index in self.anchor_candidates
        ):
            raise ValueError("Anchor candidate lies outside its shot.")
        if self.anchor_candidates and not set(self.anchor_frames).issubset(
            self.anchor_candidates
        ):
            raise ValueError("Selected Anchors must be present in anchor_candidates.")

    def to_dict(self) -> dict[str, object]:
        return {
            "shot_id": self.shot_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "anchor_frames": list(self.anchor_frames),
            "anchor_candidates": list(self.anchor_candidates),
            "description": self.description,
            "selection_reason": self.selection_reason,
        }


@dataclass(frozen=True)
class StoryboardPlan:
    frame_count: int
    fps: float
    shots: tuple[ShotPlan, ...]
    planner: str
    hero_anchor_frame: Optional[int] = None
    hero_anchor_candidates: tuple[int, ...] = ()
    hero_selection_reason: str = ""
    diagnosis: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.hero_anchor_frame is not None and not (
            0 <= self.hero_anchor_frame < self.frame_count
        ):
            raise ValueError("HeroAnchor lies outside the video.")
        if any(
            frame < 0 or frame >= self.frame_count
            for frame in self.hero_anchor_candidates
        ):
            raise ValueError("HeroAnchor candidate lies outside the video.")
        if (
            self.hero_anchor_frame is not None
            and self.hero_anchor_candidates
            and self.hero_anchor_frame not in self.hero_anchor_candidates
        ):
            raise ValueError("HeroAnchor must be in hero_anchor_candidates.")

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "fps": self.fps,
            "duration_seconds": self.frame_count / self.fps,
            "planner": self.planner,
            "hero_anchor": (
                None
                if self.hero_anchor_frame is None
                else {
                    "frame": self.hero_anchor_frame,
                    "shot_id": next(
                        (
                            shot.shot_id
                            for shot in self.shots
                            if shot.start_frame
                            <= self.hero_anchor_frame
                            <= shot.end_frame
                        ),
                        None,
                    ),
                    "ranked_candidates": list(self.hero_anchor_candidates),
                    "selection_reason": self.hero_selection_reason,
                }
            ),
            "diagnosis": self.diagnosis,
            "shots": [shot.to_dict() for shot in self.shots],
        }


@dataclass(frozen=True)
class HeroAnchorRecord:
    frame_index: int
    shot_id: int
    parameters: np.ndarray
    backend: str
    score: float
    ranked_candidates: tuple[int, ...]
    selection_reason: str
    attempts: tuple[dict[str, object], ...] = ()
    source_video: str = "target_video"

    def to_dict(self) -> dict[str, object]:
        return {
            "frame": self.frame_index,
            "shot_id": self.shot_id,
            "parameters": self.parameters.tolist(),
            "backend": self.backend,
            "score": self.score,
            "ranked_candidates": list(self.ranked_candidates),
            "selection_reason": self.selection_reason,
            "attempts": list(self.attempts),
            "source_video": self.source_video,
        }


@dataclass(frozen=True)
class GradeAttempt:
    attempt: int
    anchor_frames: tuple[int, ...]
    score: float
    accepted: bool
    metrics: dict[str, float]
    reasons: tuple[str, ...]
    recommended_anchor: Optional[int]
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "anchor_frames": list(self.anchor_frames),
            "score": self.score,
            "accepted": self.accepted,
            "metrics": self.metrics,
            "reasons": list(self.reasons),
            "recommended_anchor": self.recommended_anchor,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ShotGrade:
    shot: ShotPlan
    base_parameters: np.ndarray
    parameter_keyframes: dict[int, np.ndarray]
    frame_parameters: np.ndarray
    confidence: float
    accepted: bool
    rolled_back: bool
    rollback_reason: Optional[str]
    attempts: tuple[GradeAttempt, ...]
    search_memory: dict[str, object] = field(default_factory=dict)

    def to_dict(self, include_frame_parameters: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            **self.shot.to_dict(),
            "parameter_names": list(PARAMETER_NAMES),
            "base_parameters": self.base_parameters.tolist(),
            "parameter_keyframes": {
                str(index): values.tolist()
                for index, values in sorted(self.parameter_keyframes.items())
            },
            "confidence": self.confidence,
            "accepted": self.accepted,
            "rolled_back": self.rolled_back,
            "rollback_reason": self.rollback_reason,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "search_memory": self.search_memory,
        }
        if include_frame_parameters:
            payload["frame_parameters"] = self.frame_parameters.tolist()
        return payload


@dataclass(frozen=True)
class GradeGraph:
    instruction: str
    storyboard: StoryboardPlan
    shots: tuple[ShotGrade, ...]
    frame_parameters: np.ndarray
    backend: str
    critic: str
    orchestrator: str = "photoagent-uct-mcts"
    hero_anchor: Optional[HeroAnchorRecord] = None
    hero_anchor_attempts: tuple[dict[str, object], ...] = ()

    def to_dict(self, include_frame_parameters: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": "dynamic-grade-graph/v1",
            "instruction": self.instruction,
            "parameter_schema": {
                "names": list(PARAMETER_NAMES),
                "lower_bounds": PARAMETER_LOWER_BOUNDS.tolist(),
                "upper_bounds": PARAMETER_UPPER_BOUNDS.tolist(),
                "units": {
                    "exposure": "stops",
                    "local_exposure": "stops",
                    "others": "normalized",
                },
            },
            "backend": self.backend,
            "critic": self.critic,
            "orchestrator": self.orchestrator,
            "hero_anchor": (
                None if self.hero_anchor is None else self.hero_anchor.to_dict()
            ),
            "hero_anchor_attempts": list(self.hero_anchor_attempts),
            "storyboard": self.storyboard.to_dict(),
            "shots": [
                shot.to_dict(include_frame_parameters=False) for shot in self.shots
            ],
        }
        if include_frame_parameters:
            payload["frame_parameters"] = self.frame_parameters.tolist()
        return payload
