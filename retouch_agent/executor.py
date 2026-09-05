"""Deterministic, differentiable photometric retouching executor."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .parameters import RetouchParameters


ImageLike = Union[Tensor, np.ndarray, Image.Image]


class RetouchExecutor:
    """Apply a fixed sequence of global and mask-local photometric operations.

    Parameter order is deliberately fixed so the same vector can be propagated
    through BayesGrade and exported later. The executor never synthesizes new
    geometry or texture.
    """

    parameter_count = 12

    @staticmethod
    def srgb_to_linear(image: Tensor) -> Tensor:
        image = image.clamp_min(0.0)
        return torch.where(
            image <= 0.04045,
            image / 12.92,
            ((image + 0.055) / 1.055).pow(2.4),
        )

    @staticmethod
    def linear_to_srgb(image: Tensor) -> Tensor:
        image = image.clamp_min(0.0)
        return torch.where(
            image <= 0.0031308,
            image * 12.92,
            1.055 * image.pow(1.0 / 2.4) - 0.055,
        )

    @staticmethod
    def _prepare_image(image: Tensor) -> tuple[Tensor, bool]:
        image = torch.as_tensor(image)
        squeeze = image.ndim == 3
        if squeeze:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [3,H,W] or [B,3,H,W].")
        if not image.is_floating_point():
            image = image.float() / 255.0
        return image, squeeze

    @staticmethod
    def _prepare_parameters(
        parameters: Tensor, batch_size: int, device, dtype
    ) -> Tensor:
        parameters = torch.as_tensor(parameters, device=device, dtype=dtype)
        if parameters.ndim == 1:
            parameters = parameters.unsqueeze(0)
        if parameters.ndim != 2 or parameters.shape[1] != 12:
            raise ValueError("parameters must have shape [12] or [B,12].")
        if parameters.shape[0] == 1 and batch_size > 1:
            parameters = parameters.expand(batch_size, -1)
        if parameters.shape[0] != batch_size:
            raise ValueError("Parameter and image batch sizes do not match.")
        return parameters

    @staticmethod
    def _prepare_mask(mask: Tensor, image: Tensor) -> Tensor:
        mask = torch.as_tensor(mask, device=image.device, dtype=image.dtype)
        if mask.ndim == 2:
            mask = mask[None, None]
        elif mask.ndim == 3:
            mask = mask[:, None] if mask.shape[0] == image.shape[0] else mask[None]
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("mask must have shape [H,W], [1,H,W], or [B,1,H,W].")
        if mask.shape[0] == 1 and image.shape[0] > 1:
            mask = mask.expand(image.shape[0], -1, -1, -1)
        if mask.shape[0] != image.shape[0] or mask.shape[-2:] != image.shape[-2:]:
            raise ValueError("Mask and image sizes do not match.")
        return mask.clamp(0.0, 1.0)

    @staticmethod
    def _luminance(image: Tensor) -> Tensor:
        weights = image.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]
        return (image * weights).sum(dim=1, keepdim=True)

    @staticmethod
    def _white_balance(linear: Tensor, temperature: Tensor, tint: Tensor) -> Tensor:
        red = torch.exp(0.28 * temperature + 0.10 * tint)
        green = torch.exp(-0.16 * tint)
        blue = torch.exp(-0.28 * temperature + 0.10 * tint)
        gains = torch.cat([red, green, blue], dim=1)
        balanced = linear * gains
        weights = linear.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]
        source_luminance = (linear * weights).sum(dim=1, keepdim=True)
        balanced_luminance = (balanced * weights).sum(dim=1, keepdim=True)
        luminance_scale = source_luminance / balanced_luminance.clamp_min(1e-8)
        return balanced * luminance_scale

    def _global_edit(self, image: Tensor, p: Tensor) -> Tensor:
        def value(index: int) -> Tensor:
            return p[:, index : index + 1, None, None]

        linear = self.srgb_to_linear(image)
        linear = linear * torch.pow(linear.new_tensor(2.0), value(0))
        linear = self._white_balance(linear, value(1), value(2))

        luminance = self._luminance(linear).clamp(0.0, 1.0)
        highlight_weight = luminance.square()
        shadow_weight = (1.0 - luminance).square()
        scale = (
            1.0 + 0.55 * value(4) * highlight_weight + 0.55 * value(5) * shadow_weight
        )
        linear = linear * scale.clamp_min(0.05)

        contrast_factor = torch.pow(linear.new_tensor(2.0), value(3))
        linear = (linear - 0.18) * contrast_factor + 0.18
        gamma = torch.exp(-0.70 * value(8))
        linear = linear.clamp_min(1e-7).pow(gamma)

        srgb = self.linear_to_srgb(linear)
        luma = self._luminance(srgb)
        saturation_factor = (1.0 + value(6)).clamp(0.0, 2.0)
        srgb = luma + saturation_factor * (srgb - luma)

        channel_spread = (
            srgb.max(dim=1, keepdim=True).values - srgb.min(dim=1, keepdim=True).values
        )
        vibrance_factor = (
            1.0 + value(7) * (1.0 - channel_spread.clamp(0.0, 1.0))
        ).clamp(0.0, 2.0)
        return luma + vibrance_factor * (srgb - luma)

    def _local_edit(self, image: Tensor, p: Tensor) -> Tensor:
        local = self.srgb_to_linear(image)
        local = local * torch.pow(local.new_tensor(2.0), p[:, 9:10, None, None])
        local = self._white_balance(
            local,
            p[:, 10:11, None, None],
            torch.zeros_like(p[:, 10:11, None, None]),
        )
        local = self.linear_to_srgb(local)
        luma = self._luminance(local)
        factor = (1.0 + p[:, 11:12, None, None]).clamp(0.0, 2.0)
        return luma + factor * (local - luma)

    def apply_vector(
        self,
        image: Tensor,
        parameters: Tensor,
        mask: Optional[Tensor] = None,
        clamp: bool = True,
    ) -> Tensor:
        image, squeeze = self._prepare_image(image)
        parameters = self._prepare_parameters(
            parameters, image.shape[0], image.device, image.dtype
        )
        output = self._global_edit(image, parameters)

        local_active = torch.any(parameters[:, 9:12].detach().abs() > 1e-8).item()
        if mask is not None:
            prepared_mask = self._prepare_mask(mask, image)
            local = self._local_edit(output, parameters)
            output = output * (1.0 - prepared_mask) + local * prepared_mask
        elif local_active:
            raise ValueError("Local retouch parameters require a mask.")

        if clamp:
            output = output.clamp(0.0, 1.0)
        return output.squeeze(0) if squeeze else output

    def apply(
        self,
        image: ImageLike,
        parameters: RetouchParameters,
        mask: Optional[ImageLike] = None,
    ) -> ImageLike:
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            mask_tensor = None
            if mask is not None:
                if isinstance(mask, Image.Image):
                    mask_tensor = torch.from_numpy(
                        np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
                    )
                else:
                    mask_tensor = torch.as_tensor(mask)
            result = self.apply_vector(
                tensor,
                torch.from_numpy(parameters.to_vector(np.float32)),
                mask=mask_tensor,
            )
            output = (
                result.permute(1, 2, 0).detach().cpu().numpy() * 255.0 + 0.5
            ).astype(np.uint8)
            return Image.fromarray(output, mode="RGB")

        if isinstance(image, np.ndarray):
            array = image.astype(np.float32)
            was_uint = np.issubdtype(image.dtype, np.integer)
            if array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError("NumPy image must have shape [H,W,3].")
            if was_uint:
                array /= 255.0
            result = self.apply_vector(
                torch.from_numpy(array).permute(2, 0, 1),
                torch.from_numpy(parameters.to_vector(np.float32)),
                mask=None if mask is None else torch.as_tensor(mask),
            )
            output = result.permute(1, 2, 0).detach().cpu().numpy()
            return (output * 255.0 + 0.5).astype(np.uint8) if was_uint else output

        if isinstance(image, Tensor):
            return self.apply_vector(
                image,
                image.new_tensor(parameters.to_vector()),
                mask=mask,
            )
        raise TypeError(f"Unsupported image type: {type(image)!r}")
