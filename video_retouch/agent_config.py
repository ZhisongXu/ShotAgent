"""Build a PhotoAgent-style multi-model runtime from a JSON manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .backends import (
    AnchorRetouchBackend,
    CommandRetouchBackend,
    MonetParameterBackend,
    MonetRetouchBackend,
    NativeRetouchBackend,
    VLAnchorBackend,
)
from .clients import (
    OpenAICompatibleVisionClient,
    OpenAIResponsesVisionClient,
    VisionLanguageClient,
)
from .critic import (
    CriticEnsemble,
    CriticMember,
    PhotoAgentStyleCritic,
    ShotCritic,
    VisionReviewCritic,
)
from .shot_planner import LongVideoStoryboardSettings


@dataclass(frozen=True)
class SearchSettings:
    maximum_evaluations: int = 24
    maximum_hero_attempts: int = 3
    exploration_constant: float = 1.41421356237
    rejection_penalty: float = 0.25
    seed: int = 7

    def __post_init__(self) -> None:
        if self.maximum_evaluations < 1:
            raise ValueError("search.maximum_evaluations must be positive.")
        if self.maximum_hero_attempts < 1:
            raise ValueError("search.maximum_hero_attempts must be positive.")
        if self.exploration_constant < 0:
            raise ValueError("search.exploration_constant cannot be negative.")
        if self.rejection_penalty < 0:
            raise ValueError("search.rejection_penalty cannot be negative.")


@dataclass(frozen=True)
class MultiAgentRuntime:
    storyboard_client: VisionLanguageClient
    storyboard_settings: LongVideoStoryboardSettings
    anchor_backends: tuple[AnchorRetouchBackend, ...]
    critic: ShotCritic
    critic_client: Optional[VisionLanguageClient]
    search: SearchSettings
    manifest: dict[str, object]

    @property
    def anchor_backend(self) -> AnchorRetouchBackend:
        """Compatibility accessor for older single-editor callers."""

        return self.anchor_backends[0]


def _vision_client(config: object, role: str) -> VisionLanguageClient:
    if not isinstance(config, dict):
        raise ValueError(f"{role} agent config must be an object.")
    provider = str(config.get("provider", ""))
    common = {
        "base_url": str(config.get("base_url", "")),
        "model_id": str(config.get("model", "")),
        "api_key_env": str(config.get("api_key_env", "")),
        "timeout_seconds": float(config.get("timeout_seconds", 180.0)),
        "max_image_side": int(config.get("max_image_side", 1280)),
    }
    if provider == "openai_compatible":
        return OpenAICompatibleVisionClient(
            **common,
            max_tokens=int(config.get("max_tokens", 1024)),
        )
    if provider == "openai_responses":
        return OpenAIResponsesVisionClient(
            **common,
            max_output_tokens=int(
                config.get("max_output_tokens", config.get("max_tokens", 4096))
            ),
            reasoning_effort=str(config.get("reasoning_effort", "high")),
            image_detail=str(config.get("image_detail", "high")),
        )
    raise ValueError(
        f"Unsupported {role} provider {provider!r}; expected "
        "openai_responses or openai_compatible."
    )


def _anchor_backend(config: object, index: int) -> AnchorRetouchBackend:
    if not isinstance(config, dict):
        raise ValueError(f"editor[{index}] config must be an object.")
    anchor_type = str(config.get("type", "vision_model"))
    name = str(config.get("name", f"editor-{index}"))
    if anchor_type == "vision_model":
        raw_stages = config.get(
            "stages", ["lighting", "white_balance_and_color", "tone"]
        )
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError(f"{name} stages must be a non-empty list.")
        return VLAnchorBackend(
            _vision_client(config, name),
            stages=tuple(str(stage) for stage in raw_stages),
            candidate_count=int(config.get("candidate_count", 16)),
            seed=int(config.get("seed", 7)),
            name=name,
            use_mkl_prior=bool(config.get("use_mkl_prior", True)),
            mkl_strength=float(config.get("mkl_strength", 0.35)),
            mkl_projection_iterations=int(config.get("mkl_projection_iterations", 40)),
            specialty=str(config.get("specialty", "")),
            direct_api_mode=bool(config.get("direct_api_mode", False)),
        )
    if anchor_type == "monet":
        return MonetRetouchBackend(Path(str(config.get("root", ""))), name=name)
    if anchor_type == "monet_parameters":
        return MonetParameterBackend(
            Path(str(config.get("root", ""))),
            python_executable=str(config.get("python_executable", "python")),
            name=name,
            style=str(config.get("style", "balanced")),
            timeout_seconds=float(config.get("timeout_seconds", 600.0)),
            reject_unsupported=bool(config.get("reject_unsupported", True)),
        )
    if anchor_type == "command":
        return CommandRetouchBackend(
            str(config.get("command", "")),
            name=name,
            timeout_seconds=float(config.get("timeout_seconds", 600.0)),
        )
    if anchor_type == "native":
        return NativeRetouchBackend(name=name)
    raise ValueError(f"Unsupported editor Agent type: {anchor_type}")


def _critic_ensemble(
    configs: object,
    acceptance_score: float,
) -> tuple[ShotCritic, Optional[VisionLanguageClient]]:
    if not isinstance(configs, list) or not configs:
        raise ValueError("evaluators must be a non-empty list.")
    members: list[CriticMember] = []
    first_client: Optional[VisionLanguageClient] = None
    for index, config in enumerate(configs):
        if not isinstance(config, dict):
            raise ValueError(f"evaluator[{index}] config must be an object.")
        critic_type = str(config.get("type", "vision_model"))
        name = str(config.get("name", f"critic-{index}"))
        if critic_type == "metrics":
            critic: ShotCritic = PhotoAgentStyleCritic(
                use_vl_review=False,
                maximum_fidelity_l1=float(config.get("maximum_fidelity_l1", 0.42)),
                maximum_clipping=float(config.get("maximum_clipping", 0.20)),
                maximum_temporal_error=float(
                    config.get("maximum_temporal_error", 0.10)
                ),
                maximum_parameter_jerk=float(
                    config.get("maximum_parameter_jerk", 0.20)
                ),
                maximum_anchor_error=float(config.get("maximum_anchor_error", 0.12)),
            )
            # The deterministic safety critic has a fixed semantic name.
            if name != critic.name:
                critic.name = name
        elif critic_type == "vision_model":
            client = _vision_client(config, name)
            if first_client is None:
                first_client = client
            critic = VisionReviewCritic(
                client=client,
                name=name,
                focus=str(config.get("focus", "overall professional quality")),
                strict=bool(config.get("strict", False)),
                fallback_score=float(config.get("fallback_score", 0.0)),
            )
        else:
            raise ValueError(f"Unsupported evaluator type: {critic_type}")
        members.append(
            CriticMember(
                critic=critic,
                weight=float(config.get("weight", 1.0)),
                veto=bool(config.get("veto", critic_type == "metrics")),
                accept_on_score=bool(config.get("accept_on_score", False)),
            )
        )
    return CriticEnsemble(members, acceptance_score), first_client


def load_multi_agent_runtime(path: Path) -> MultiAgentRuntime:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Agent config must be a JSON object.")
    storyboard_config = payload.get("storyboard")
    storyboard = _vision_client(storyboard_config, "storyboard")
    assert isinstance(storyboard_config, dict)
    storyboard_settings = LongVideoStoryboardSettings.from_dict(
        storyboard_config.get("long_video")
    )

    raw_editors = payload.get("editors")
    if raw_editors is None and "anchor" in payload:
        raw_editors = [payload["anchor"]]
    if not isinstance(raw_editors, list) or not raw_editors:
        raise ValueError("At least one editor Agent is required.")
    editors = tuple(
        _anchor_backend(config, index) for index, config in enumerate(raw_editors)
    )

    raw_evaluators = payload.get("evaluators")
    if raw_evaluators is None:
        legacy_critic = payload.get("critic")
        raw_evaluators = [
            {
                "type": "metrics",
                "name": "temporal-safety",
                "weight": 1.0,
                "veto": True,
            }
        ]
        if (
            isinstance(legacy_critic, dict)
            and legacy_critic.get("type") != "metrics_only"
        ):
            raw_evaluators.append({**legacy_critic, "name": "visual-critic"})
    acceptance_score = float(payload.get("acceptance_score", 0.60))
    critic, critic_client = _critic_ensemble(raw_evaluators, acceptance_score)

    raw_search = payload.get("search", {})
    if not isinstance(raw_search, dict):
        raise ValueError("search config must be an object.")
    search = SearchSettings(
        maximum_evaluations=int(raw_search.get("maximum_evaluations", 24)),
        maximum_hero_attempts=int(raw_search.get("maximum_hero_attempts", 3)),
        exploration_constant=float(
            raw_search.get("exploration_constant", 1.41421356237)
        ),
        rejection_penalty=float(raw_search.get("rejection_penalty", 0.25)),
        seed=int(raw_search.get("seed", 7)),
    )

    reference_refinement = payload.get("reference_chroma_refinement")
    sanitized_reference_refinement = None
    if reference_refinement is not None:
        if not isinstance(reference_refinement, dict):
            raise ValueError("reference_chroma_refinement must be an object.")
        refinement_strength = float(reference_refinement.get("strength", 0.6))
        if not 0.0 <= refinement_strength <= 1.0:
            raise ValueError("reference_chroma_refinement.strength must be in [0, 1].")
        refinement_mode = str(
            reference_refinement.get("mode", "luma_preserving_video_global")
        )
        if refinement_mode != "luma_preserving_video_global":
            raise ValueError(
                "reference_chroma_refinement.mode must be "
                "'luma_preserving_video_global'."
            )
        sanitized_reference_refinement = {
            "enabled": bool(reference_refinement.get("enabled", False)),
            "strength": refinement_strength,
            "target_luma_strength": float(
                reference_refinement.get("target_luma_strength", 0.0)
            ),
            "mode": refinement_mode,
            "role": str(reference_refinement.get("role", "pool tool")),
        }
        if not 0.0 <= sanitized_reference_refinement["target_luma_strength"] <= 1.0:
            raise ValueError(
                "reference_chroma_refinement.target_luma_strength must be in [0, 1]."
            )

    sanitized = {
        "storyboard": {
            "provider": storyboard_config.get("provider"),
            "model": storyboard.model_id,
            "planner": "hierarchical-vision-storyboard/v2",
            "long_video": {
                field: getattr(storyboard_settings, field)
                for field in storyboard_settings.__dataclass_fields__
            },
        },
        "editors": [
            {
                "name": editor.name,
                "type": (
                    "vision_model" if isinstance(editor, VLAnchorBackend) else "tool"
                ),
                "model": (
                    editor.client.model_id
                    if isinstance(editor, VLAnchorBackend)
                    else editor.name
                ),
            }
            for editor in editors
        ],
        "evaluators": [
            {
                "name": member.critic.name,
                "weight": member.weight,
                "veto": member.veto,
                "model": (
                    member.critic.client.model_id
                    if isinstance(member.critic, VisionReviewCritic)
                    else "deterministic-metrics"
                ),
            }
            for member in critic.members
        ],
        "search": {
            "algorithm": "uct-mcts",
            "maximum_evaluations": search.maximum_evaluations,
            "maximum_hero_attempts": search.maximum_hero_attempts,
            "exploration_constant": search.exploration_constant,
            "rejection_penalty": search.rejection_penalty,
            "seed": search.seed,
        },
    }
    if sanitized_reference_refinement is not None:
        sanitized["reference_chroma_refinement"] = sanitized_reference_refinement
    return MultiAgentRuntime(
        storyboard_client=storyboard,
        storyboard_settings=storyboard_settings,
        anchor_backends=editors,
        critic=critic,
        critic_client=critic_client,
        search=search,
        manifest=sanitized,
    )
