"""Optional learned content and quality metrics for the no-GT benchmark.

The models are loaded lazily because they download checkpoints and are not
needed by the lightweight benchmark/test path.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext

import numpy as np
from PIL import Image


def _indices(length: int, count: int) -> np.ndarray:
    return np.unique(np.linspace(0, length - 1, min(length, count)).round().astype(int))


class LearnedMetricSuite:
    """CLIP, DINOv2 and no-reference IQA metrics sampled over video frames."""

    def __init__(self, frame_count: int = 8, device: str | None = None) -> None:
        import torch

        self.torch = torch
        self.frame_count = frame_count
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dino = None
        self._dino_preprocess = None
        self._iqa: dict[str, object] = {}

    def _autocast(self):
        if self.device.startswith("cuda"):
            return self.torch.autocast(device_type="cuda")
        return nullcontext()

    def _sample(self, frames: Sequence[Image.Image]) -> list[Image.Image]:
        return [
            frames[int(index)].convert("RGB")
            for index in _indices(len(frames), self.frame_count)
        ]

    def _load_dino(self) -> None:
        if self._dino is not None:
            return
        from torchvision import transforms

        self._dino = (
            self.torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", trust_repo=True
            )
            .eval()
            .to(self.device)
        )
        self._dino_preprocess = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def _dino_images(self, frames: Sequence[Image.Image]):
        self._load_dino()
        batch = self.torch.stack(
            [self._dino_preprocess(frame) for frame in self._sample(frames)]
        ).to(self.device)
        with self.torch.inference_mode(), self._autocast():
            features = self._dino(batch).float()
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def dino_content_similarity(
        self, target: Sequence[Image.Image], output: Sequence[Image.Image]
    ) -> float:
        target_features = self._dino_images(target)
        output_features = self._dino_images(output)
        count = min(len(target_features), len(output_features))
        return float(
            (target_features[:count] * output_features[:count]).sum(dim=-1).mean()
        )

    def _load_iqa(self, name: str):
        if name not in self._iqa:
            import pyiqa

            self._iqa[name] = pyiqa.create_metric(name, device=self.device).eval()
        return self._iqa[name]

    def no_reference_quality(self, output: Sequence[Image.Image], name: str) -> float:
        from torchvision.transforms.functional import pil_to_tensor

        model = self._load_iqa(name)
        scores = []
        for frame in self._sample(output):
            tensor = (
                pil_to_tensor(frame).float().div(255.0).unsqueeze(0).to(self.device)
            )
            with self.torch.inference_mode():
                scores.append(float(model(tensor).reshape(-1).mean()))
        return float(np.mean(scores))

    def evaluate(
        self,
        target: Sequence[Image.Image],
        output: Sequence[Image.Image],
    ) -> dict[str, float]:
        return {
            "dino_content_similarity": self.dino_content_similarity(target, output),
            "musiq_score": self.no_reference_quality(output, "musiq"),
            "clipiqa_score": self.no_reference_quality(output, "clipiqa"),
        }
