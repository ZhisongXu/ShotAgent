"""Canonical multimodal training record shared by all agent roles."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence

from retouch_agent.parameters import PARAMETER_NAMES
from video_retouch.tasks import (
    AGENT_SYSTEM_PROMPT,
    ANCHOR_GRADE_TASK,
    CRITIQUE_TASK,
    STORYBOARD_TASK,
)


class TrainingRole(str, Enum):
    STORYBOARD = "storyboard"
    ANCHOR_GRADE = "anchor_grade"
    CRITIQUE = "critique"

    @property
    def task_token(self) -> str:
        return {
            self.STORYBOARD: STORYBOARD_TASK,
            self.ANCHOR_GRADE: ANCHOR_GRADE_TASK,
            self.CRITIQUE: CRITIQUE_TASK,
        }[self]


SUPPORTED_METHODS = {
    "monet_puzzle_a",
    "monet_puzzle_b",
    "monet_puzzle_c",
    "jarvis_tool_trace",
    "jarvis_interleaved_feedback",
    "photoagent_trajectory",
    "photoagent_preference",
    "video_storyboard",
    "synthetic_temporal_intervention",
    "human_grade",
}


@dataclass(frozen=True)
class AgentTrainingExample:
    example_id: str
    role: TrainingRole
    method: str
    images: tuple[str, ...]
    prompt: str
    response: dict[str, object]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("Training example requires an id.")
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported training method: {self.method}")
        if not self.images:
            raise ValueError("Multimodal training example requires images.")
        self._validate_response()

    def _validate_response(self) -> None:
        if self.role is TrainingRole.STORYBOARD:
            shots = self.response.get("shots")
            if not isinstance(shots, list) or not shots:
                raise ValueError("Storyboard target requires non-empty shots.")
        elif self.role is TrainingRole.ANCHOR_GRADE:
            updates = self.response.get("parameter_updates")
            if not isinstance(updates, dict):
                raise ValueError("Anchor target requires parameter_updates.")
            unknown = set(updates) - set(PARAMETER_NAMES)
            if unknown:
                raise ValueError(f"Unknown target parameters: {sorted(unknown)}")
            for name, value in updates.items():
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(f"Non-finite target parameter: {name}")
        elif self.role is TrainingRole.CRITIQUE:
            required = {"accept", "score", "reasons"}
            if not required.issubset(self.response):
                raise ValueError("Critique target requires accept, score, and reasons.")
            if not isinstance(self.response["accept"], bool):
                raise ValueError("Critique accept target must be boolean.")
            score = float(self.response["score"])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("Critique score target must be in [0,1].")
            if not isinstance(self.response["reasons"], list):
                raise ValueError("Critique reasons target must be a list.")

    @classmethod
    def from_dict(
        cls, payload: dict[str, object], base_directory: Path | None = None
    ) -> "AgentTrainingExample":
        raw_images = payload.get("images", [])
        if not isinstance(raw_images, Sequence) or isinstance(raw_images, str):
            raise ValueError("images must be a sequence of paths.")
        paths = []
        for value in raw_images:
            path = Path(str(value))
            if base_directory is not None and not path.is_absolute():
                path = (base_directory / path).resolve()
            paths.append(str(path))
        response = payload.get("response")
        if not isinstance(response, dict):
            raise ValueError("response must be an object.")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object.")
        return cls(
            example_id=str(payload.get("example_id", "")),
            role=TrainingRole(str(payload.get("role", ""))),
            method=str(payload.get("method", "")),
            images=tuple(paths),
            prompt=str(payload.get("prompt", "")),
            response=response,
            metadata=metadata,
        )

    def to_sharegpt(self) -> dict[str, object]:
        image_placeholders = "\n".join(
            f"<image> image_{index}" for index in range(len(self.images))
        )
        user_content = (
            f"{self.role.task_token}\n{self.prompt.strip()}\n{image_placeholders}"
        )
        return {
            "id": self.example_id,
            "messages": [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {
                    "role": "assistant",
                    "content": json.dumps(self.response, ensure_ascii=False),
                },
            ],
            "images": list(self.images),
            "metadata": {"method": self.method, **self.metadata},
        }
