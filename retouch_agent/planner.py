"""Structured planners for converting an editing request into parameter priors."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np
from PIL import Image

from .executor import RetouchExecutor
from .parameters import RetouchParameters


@dataclass(frozen=True)
class RetouchPlan:
    diagnosis: dict[str, object]
    initial_parameters: RetouchParameters
    targets: dict[str, float]
    constraints: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "diagnosis": self.diagnosis,
            "initial_parameters": self.initial_parameters.to_dict(),
            "targets": dict(self.targets),
            "constraints": list(self.constraints),
        }


class RetouchPlanner(Protocol):
    def plan(
        self,
        image: Image.Image,
        instruction: str,
        reference: Optional[Image.Image] = None,
        has_local_mask: bool = False,
    ) -> RetouchPlan: ...


def image_statistics(image: Image.Image) -> dict[str, float]:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luma = 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
    return {
        "luminance": float(luma.mean()),
        "contrast": float(luma.std()),
        "saturation": float((array.max(axis=-1) - array.min(axis=-1)).mean()),
        "warmth": float((array[..., 0] - array[..., 2]).mean()),
    }


class HeuristicRetouchPlanner:
    """Training-free fallback used for tests and early executor experiments."""

    def plan(
        self,
        image: Image.Image,
        instruction: str,
        reference: Optional[Image.Image] = None,
        has_local_mask: bool = False,
    ) -> RetouchPlan:
        stats = image_statistics(image)
        target = image_statistics(reference) if reference is not None else dict(stats)
        text = instruction.lower()

        target["luminance"] = float(np.clip(target["luminance"], 0.32, 0.62))
        exposure = float(np.clip(math.log2((target["luminance"] + 1e-3) / (stats["luminance"] + 1e-3)), -1.5, 1.5))
        temperature = 0.0
        saturation = 0.0
        vibrance = 0.0
        contrast = 0.0
        tone = 0.0
        tint = 0.0
        highlights = 0.0
        shadows = 0.0
        midtone_lift = any(
            phrase in text
            for phrase in (
                "lift midtones",
                "open midtones",
                "raise midtones",
                "提亮中间调",
                "打开中间调",
                "中间调提亮",
            )
        )

        if any(word in text for word in ("warm", "golden", "温暖", "暖色", "暖")):
            temperature += 0.30
            target["warmth"] = max(target["warmth"], stats["warmth"] + 0.04)
        if any(word in text for word in ("cool", "cold", "清冷", "冷色")):
            temperature -= 0.30
            target["warmth"] = min(target["warmth"], stats["warmth"] - 0.04)
        if (
            any(word in text for word in ("bright", "brighter", "明亮", "提亮", "亮一些"))
            and not midtone_lift
        ):
            exposure += 0.35
            target["luminance"] = min(0.62, target["luminance"] + 0.10)
        if any(word in text for word in ("dark", "moody", "暗调", "压暗")):
            exposure -= 0.25
            target["luminance"] = max(0.25, target["luminance"] - 0.08)
        restrained_chroma = any(
            phrase in text
            for phrase in (
                "not oversaturated",
                "restrained saturation",
                "controlled chroma",
                "不过饱和",
                "避免过饱和",
                "饱和度克制",
                "克制色度",
            )
        )
        if (
            any(word in text for word in ("vivid", "colorful", "鲜艳", "生动"))
            and not restrained_chroma
        ):
            saturation += 0.18
            vibrance += 0.25
            target["saturation"] = min(0.65, target["saturation"] + 0.10)
        if restrained_chroma:
            saturation -= 0.035
            vibrance += 0.04
        if any(word in text for word in ("cinematic", "film", "电影", "胶片")):
            contrast += 0.18
            saturation -= 0.08
            tone += 0.10
            target["contrast"] = min(0.28, target["contrast"] + 0.04)
        if any(
            phrase in text
            for phrase in (
                "gentle contrast",
                "soft contrast",
                "restrained contrast",
                "柔和对比",
                "克制对比",
            )
        ):
            contrast -= 0.08
        if midtone_lift:
            exposure += 0.12
        if any(
            word in text
            for word in (
                "ocean",
                "sea",
                "aqua",
                "marine",
                "coast",
                "海洋",
                "海水",
                "海岸",
                "青蓝",
            )
        ):
            # A restrained cyan bias keeps water clean without globally
            # crushing warm land tones into a stylized LUT look.
            temperature -= 0.10
            tint -= 0.035
        if any(
            word in text
            for word in (
                "clean",
                "crisp",
                "polished",
                "travel commercial",
                "清透",
                "通透",
                "干净",
                "旅行广告",
                "商业风光",
            )
        ):
            contrast += 0.07
            vibrance += 0.08
            highlights -= 0.08
            shadows += 0.05
            target["contrast"] = min(0.28, target["contrast"] + 0.025)
            target["saturation"] = min(0.65, target["saturation"] + 0.035)
        if any(
            word in text
            for word in (
                "protect highlights",
                "preserve highlights",
                "recover highlights",
                "highlight detail",
                "保留高光",
                "保护高光",
                "高光细节",
                "浪花细节",
                "天空细节",
                "roll-off",
                "rolloff",
                "高光滚降",
            )
        ):
            highlights -= 0.16
        if any(
            word in text
            for word in (
                "open shadows",
                "lift shadows",
                "shadow detail",
                "暗部层次",
                "暗部细节",
                "提亮暗部",
            )
        ):
            shadows += 0.12

        local_exposure = 0.0
        if has_local_mask and any(word in text for word in ("face", "person", "subject", "人物", "人脸", "主体")):
            local_exposure = 0.20

        parameters = RetouchParameters.from_mapping(
            {
                "exposure": exposure,
                "temperature": temperature,
                "tint": tint,
                "contrast": contrast,
                "highlights": highlights,
                "shadows": shadows,
                "saturation": saturation,
                "vibrance": vibrance,
                "tone_curve": tone,
                "local_exposure": local_exposure,
            },
            clamp=True,
        )
        if reference is None:
            # Composite professional briefs change several controls at once.
            # Use the planned transform itself to derive mutually consistent
            # luminance/color targets instead of scoring it against a target
            # assembled from independent one-dimensional heuristics.
            global_parameters = RetouchParameters.from_mapping(
                {
                    **parameters.to_dict(),
                    "local_exposure": 0.0,
                    "local_temperature": 0.0,
                    "local_saturation": 0.0,
                }
            )
            target = image_statistics(
                RetouchExecutor().apply(image, global_parameters)
            )
        return RetouchPlan(
            diagnosis={
                "source_statistics": stats,
                "target_source": (
                    "reference" if reference is not None else "planned_transform"
                ),
                "planner": "heuristic",
                "instruction": instruction,
            },
            initial_parameters=parameters,
            targets=target,
            constraints=(
                "avoid_highlight_clipping",
                "avoid_shadow_crushing",
                "preserve_content",
            ),
        )
