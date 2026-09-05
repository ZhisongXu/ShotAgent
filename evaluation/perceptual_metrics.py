"""Optional learned metrics for the no-GT reference-video benchmark.

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
        self._clip = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
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

    def _load_clip(self) -> None:
        if self._clip is not None:
            return
        import open_clip

        model_name = "ViT-B-32"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="laion2b_s34b_b79k"
        )
        self._clip = model.eval().to(self.device)
        self._clip_preprocess = preprocess
        self._clip_tokenizer = open_clip.get_tokenizer(model_name)

    def _clip_images(self, frames: Sequence[Image.Image]):
        self._load_clip()
        batch = self.torch.stack(
            [self._clip_preprocess(frame) for frame in self._sample(frames)]
        ).to(self.device)
        with self.torch.inference_mode(), self._autocast():
            features = self._clip.encode_image(batch)
        features = features.float()
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def clip_directional_similarity(
        self,
        target: Sequence[Image.Image],
        output: Sequence[Image.Image],
        reference: Sequence[Image.Image],
    ) -> float | None:
        """Cosine between target->output and target->reference CLIP directions.

        With an unrelated-content reference this remains a diagnostic because
        the reference direction also contains semantic content differences.
        """

        target_feature = self._clip_images(target).mean(dim=0)
        output_feature = self._clip_images(output).mean(dim=0)
        reference_feature = self._clip_images(reference).mean(dim=0)
        edit_direction = output_feature - target_feature
        reference_direction = reference_feature - target_feature
        denominator = edit_direction.norm() * reference_direction.norm()
        if float(denominator) < 1e-8:
            return None
        return float(self.torch.dot(edit_direction, reference_direction) / denominator)

    def clip_text_alignment(
        self, output: Sequence[Image.Image], instruction: str
    ) -> float:
        self._load_clip()
        image_features = self._clip_images(output)
        tokens = self._clip_tokenizer([instruction]).to(self.device)
        with self.torch.inference_mode(), self._autocast():
            text_feature = self._clip.encode_text(tokens).float()
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True).clamp_min(
            1e-8
        )
        return float((image_features @ text_feature.T).mean())

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
        reference: Sequence[Image.Image],
        instruction: str | None = None,
    ) -> dict[str, float | None]:
        values: dict[str, float | None] = {
            "clip_directional_similarity": self.clip_directional_similarity(
                target, output, reference
            ),
            "dino_content_similarity": self.dino_content_similarity(target, output),
            "musiq_score": self.no_reference_quality(output, "musiq"),
            "clipiqa_score": self.no_reference_quality(output, "clipiqa"),
        }
        values["clip_text_alignment"] = (
            self.clip_text_alignment(output, instruction) if instruction else None
        )
        return values
