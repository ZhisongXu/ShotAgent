"""Optional learned content and quality metrics for the no-GT benchmark.

The models are loaded lazily because they download checkpoints and are not
needed by the lightweight benchmark/test path.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image


def _indices(length: int, count: int) -> np.ndarray:
    return np.unique(np.linspace(0, length - 1, min(length, count)).round().astype(int))


class LearnedMetricSuite:
    """CLIP, reference-style, DINOv2, and MUSIQ metrics sampled over video frames."""

    def __init__(
        self,
        frame_count: int = 8,
        device: str | None = None,
        style_vgg_weights: Path | None = None,
    ) -> None:
        import torch

        self.torch = torch
        self.frame_count = frame_count
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dino = None
        self._dino_preprocess = None
        self._clip = None
        self._clip_preprocess = None
        self.style_vgg_weights = style_vgg_weights
        self._style_vgg = None
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

    def _load_clip(self) -> None:
        if self._clip is not None:
            return
        import clip

        self._clip, self._clip_preprocess = clip.load("RN50", device=self.device)
        self._clip.eval()

    def clip_prompt_similarity(
        self, output: Sequence[Image.Image], prompt: str
    ) -> float:
        """Mean raw CLIP RN50 cosine similarity between sampled frames and prompt."""

        import clip

        self._load_clip()
        images = self.torch.stack(
            [self._clip_preprocess(frame) for frame in self._sample(output)]
        ).to(self.device)
        text = clip.tokenize([prompt], truncate=True).to(self.device)
        with self.torch.inference_mode(), self._autocast():
            image_features = self._clip.encode_image(images).float()
            text_features = self._clip.encode_text(text).float()
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            return float((image_features @ text_features.T).mean())

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

    def _load_style_vgg(self) -> None:
        if self._style_vgg is not None:
            return
        if self.style_vgg_weights is None:
            raise ValueError(
                "VGG style metrics require --style-vgg-weights. The compatible "
                "vgg_normalised.pth is distributed with SA-LUT/PST models."
            )
        nn = self.torch.nn
        model = nn.Sequential(
            nn.Conv2d(3, 3, 1),
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
        )
        state = self.torch.load(
            self.style_vgg_weights, map_location="cpu", weights_only=False
        )
        prefix = model.state_dict()
        compatible = {key: state[key] for key in prefix if key in state}
        if compatible.keys() != prefix.keys():
            missing = sorted(prefix.keys() - compatible.keys())
            raise ValueError(f"Incompatible VGG style weights; missing keys: {missing}")
        model.load_state_dict(compatible)
        self._style_vgg = model.eval().to(self.device)

    def _style_signature(self, frames: Sequence[Image.Image]):
        from torchvision.transforms.functional import pil_to_tensor, resize

        self._load_style_vgg()
        sampled = self._sample(frames)
        batch = self.torch.stack(
            [pil_to_tensor(frame).float().div(255.0) for frame in sampled]
        ).to(self.device)
        batch = resize(batch, [224, 224], antialias=True)
        capture = {3, 6, 10, 13, 17}
        statistics = []
        with self.torch.inference_mode(), self._autocast():
            feature = batch
            for index, layer in enumerate(self._style_vgg):
                feature = layer(feature)
                if index in capture:
                    flat = feature.float().flatten(2)
                    mean = flat.mean(dim=2).mean(dim=0)
                    std = flat.std(dim=2, correction=0).mean(dim=0)
                    for vector in (mean, std):
                        statistics.append(vector / vector.norm().clamp_min(1e-8))
        return self.torch.cat(statistics)

    @staticmethod
    def _cosine(left, right) -> float:
        return float(
            (left * right).sum() / (left.norm() * right.norm()).clamp_min(1e-8)
        )

    def reference_style_similarity(
        self,
        reference: Sequence[Image.Image],
        output: Sequence[Image.Image],
    ) -> float:
        reference_style = self._style_signature(reference)
        output_style = self._style_signature(output)
        return self._cosine(output_style, reference_style)

    def evaluate(
        self,
        target: Sequence[Image.Image],
        reference: Sequence[Image.Image],
        output: Sequence[Image.Image],
    ) -> dict[str, float]:
        return {
            "vgg_style_similarity": self.reference_style_similarity(reference, output),
            "dino_content_similarity": self.dino_content_similarity(target, output),
            "musiq_score": self.no_reference_quality(output, "musiq"),
        }
