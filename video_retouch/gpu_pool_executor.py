"""Torch batch executor for the v2 grading Pool graph.

The executor accepts a whole ``BCHW`` batch and keeps every Pool operation on
the selected Torch device.  CUDA is used when available; CPU Torch remains a
feature-complete fallback for machines without a supported GPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .grade_pools import HSL_REGIONS, POOL_OPERATION_TYPES, POOL_PROCESSING_ORDER


Tensor = torch.Tensor


def _v(values: Sequence[float], like: Tensor) -> Tensor:
    return torch.as_tensor(values, dtype=like.dtype, device=like.device).view(-1, 1, 1, 1)


def _srgb_decode(x: Tensor) -> Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp_min(0).pow(2.4))


def _srgb_encode(x: Tensor) -> Tensor:
    x = x.clamp_min(0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.pow(1 / 2.4) - 0.055)


def _luma(x: Tensor) -> Tensor:
    weights = x.new_tensor((0.2126, 0.7152, 0.0722)).view(1, 3, 1, 1)
    return (x * weights).sum(1, keepdim=True)


def _smoothstep(edge0: Tensor | float, edge1: Tensor | float, x: Tensor) -> Tensor:
    width = torch.as_tensor(edge1, dtype=x.dtype, device=x.device) - torch.as_tensor(
        edge0, dtype=x.dtype, device=x.device
    )
    t = ((x - edge0) / width.clamp_min(1e-6)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _blur(x: Tensor, radius: int) -> Tensor:
    radius = max(1, min(int(radius), 31))
    kernel = radius * 2 + 1
    return F.avg_pool2d(F.pad(x, (radius,) * 4, mode="reflect"), kernel, stride=1)


def _rgb_to_hsl(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    maximum, max_index = x.max(1, keepdim=True)
    minimum = x.min(1, keepdim=True).values
    delta = maximum - minimum
    lightness = (maximum + minimum) * 0.5
    saturation = torch.where(
        delta > 1e-7,
        delta / (1 - (2 * lightness - 1).abs()).clamp_min(1e-7),
        torch.zeros_like(delta),
    )
    red, green, blue = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    safe = delta.clamp_min(1e-7)
    candidates = torch.cat(
        (
            ((green - blue) / safe).remainder(6),
            (blue - red) / safe + 2,
            (red - green) / safe + 4,
        ),
        dim=1,
    )
    hue = candidates.gather(1, max_index) * 60
    hue = torch.where(delta > 1e-7, hue, torch.zeros_like(hue))
    return hue, saturation, lightness


def _hsl_to_rgb(h: Tensor, s: Tensor, light: Tensor) -> Tensor:
    chroma = (1 - (2 * light - 1).abs()) * s
    sector = h.remainder(360) / 60
    x = chroma * (1 - (sector.remainder(2) - 1).abs())
    z = torch.zeros_like(chroma)
    choices = torch.stack(
        (
            torch.cat((chroma, x, z), 1),
            torch.cat((x, chroma, z), 1),
            torch.cat((z, chroma, x), 1),
            torch.cat((z, x, chroma), 1),
            torch.cat((x, z, chroma), 1),
            torch.cat((chroma, z, x), 1),
        ),
        dim=1,
    )
    index = sector.floor().long().remainder(6).unsqueeze(2).expand(-1, -1, 3, -1, -1)
    rgb = choices.gather(1, index).squeeze(1)
    return rgb + light - chroma * 0.5


def _curve(x: Tensor, samples: Tensor) -> Tensor:
    """Per-frame 17-sample piecewise-linear curves."""

    position = x.clamp(0, 1) * 16
    lower = position.floor().long().clamp(0, 15)
    upper = (lower + 1).clamp(max=16)
    amount = position - lower
    batch = torch.arange(x.shape[0], device=x.device).view(-1, 1, 1, 1).expand_as(lower)
    return samples[batch, lower] * (1 - amount) + samples[batch, upper] * amount


def _at(operation: object, frame_index: int) -> Mapping[str, object]:
    track = getattr(operation, "parameter_track", ())
    if track:
        start = int(getattr(operation, "frame_range")[0])
        return track[min(max(frame_index - start, 0), len(track) - 1)]
    return getattr(operation, "parameters")


def _params(operations: Sequence[object], frame_indices: Sequence[int]) -> list[Mapping[str, object]]:
    return [_at(operation, index) for operation, index in zip(operations, frame_indices)]


def _primary(x: Tensor, ps: Sequence[Mapping[str, object]]) -> Tensor:
    linear = _srgb_decode(x)
    exposure = _v([2 ** float(p["exposure"]) for p in ps], x)
    linear = linear * exposure
    luminance = _luma(linear).clamp(0, 1)
    hi = _v([float(p["highlights"]) / 100 for p in ps], x)
    sh = _v([float(p["shadows"]) / 100 for p in ps], x)
    linear = linear * (1 + 0.75 * hi * luminance.square() + 0.75 * sh * (1 - luminance).square()).clamp_min(0.05)
    whites = _v([float(p["whites"]) / 100 for p in ps], x)
    blacks = _v([float(p["blacks"]) / 100 for p in ps], x)
    linear = linear + 0.18 * whites * _smoothstep(0.55, 1.0, luminance)
    linear = linear + 0.10 * blacks * (1 - _smoothstep(0.0, 0.40, luminance))
    factor = 2 ** (1.5 * _v([float(p["contrast"]) / 100 for p in ps], x))
    linear = (linear - 0.18) * factor + 0.18
    gamma = _v([float(p["gamma"]) for p in ps], x)
    return _srgb_encode(linear.clamp_min(1e-7).pow(1 / gamma))


def _white_balance(x: Tensor, ps: Sequence[Mapping[str, object]]) -> Tensor:
    linear = _srgb_decode(x)
    gains = []
    for p in ps:
        warmth = ((1_000_000 / float(p["temperature"])) - (1_000_000 / 6500)) / 350
        tint = float(p["tint"]) / 100
        gains.append((math.exp(0.48 * warmth + 0.10 * tint), math.exp(-0.20 * tint), math.exp(-0.48 * warmth + 0.10 * tint)))
    return _srgb_encode(linear * x.new_tensor(gains).view(-1, 3, 1, 1))


def _wheels(x: Tensor, ps: Sequence[Mapping[str, object]]) -> Tensor:
    linear = _srgb_decode(x)
    luminance = _luma(linear).clamp(0, 1)
    balance = _v([float(p["balance"]) / 250 for p in ps], x)
    shadow = 1 - _smoothstep(0.05 + balance, 0.58 + balance, luminance)
    highlight = _smoothstep(0.42 + balance, 0.95 + balance, luminance)
    midtone = (1 - shadow - highlight).clamp(0, 1)
    result = linear
    for zone, weight in zip(("shadows", "midtones", "highlights"), (shadow, midtone, highlight)):
        deltas = []
        for p in ps:
            xv, yv = float(p[zone]["x"]), float(p[zone]["y"])
            deltas.append((xv + 0.5 * yv, -0.5 * xv + 0.25 * yv, -xv - 0.75 * yv))
        result = result + weight * x.new_tensor(deltas).view(-1, 3, 1, 1) * 0.16
    return _srgb_encode(result.clamp_min(0))


def _curves(x: Tensor, ps: Sequence[Mapping[str, object]]) -> Tensor:
    result = x.clamp(0, 1)
    strength = _v([float(p["strength"]) for p in ps], x)
    master = x.new_tensor([p["rgb"] for p in ps])
    result = result + strength * (_curve(result, master) - result)
    channels = []
    for index, name in enumerate(("red", "green", "blue")):
        samples = x.new_tensor([p[name] for p in ps])
        value = result[:, index : index + 1]
        channels.append(value + strength * (_curve(value, samples) - value))
    return torch.cat(channels, 1)


def _hsl8(x: Tensor, ps: Sequence[Mapping[str, object]]) -> Tensor:
    hue, saturation, lightness = _rgb_to_hsl(x.clamp(0, 1))
    hd, sd, ld, total = (torch.zeros_like(hue) for _ in range(4))
    for region, center in HSL_REGIONS.items():
        distance = (hue - center + 180).remainder(360).sub(180).abs()
        weight = (1 - distance / 45).clamp(0, 1)
        weight = weight.square() * (3 - 2 * weight)
        hd = hd + weight * _v([float(p[region]["hue"]) * 0.30 for p in ps], x)
        sd = sd + weight * _v([float(p[region]["saturation"]) / 100 for p in ps], x)
        ld = ld + weight * _v([float(p[region]["luminance"]) / 200 for p in ps], x)
        total = total + weight
    normalization = total.clamp_min(1)
    return _hsl_to_rgb((hue + hd / normalization).remainder(360), (saturation + sd / normalization).clamp(0, 1), (lightness + ld / normalization).clamp(0, 1))


def _global_color(x: Tensor, ps: Sequence[Mapping[str, object]]) -> Tensor:
    hue, saturation, lightness = _rgb_to_hsl(x.clamp(0, 1))
    hue = (hue + _v([float(p["hue_shift"]) for p in ps], x)).remainder(360)
    scale = _v([max(0, 1 + float(p["saturation"]) / 100) for p in ps], x)
    saturation = (saturation * scale).clamp(0, 1)
    vibrance = _v([float(p["vibrance"]) / 100 for p in ps], x)
    distance = (hue - 28 + 180).remainder(360).sub(180).abs()
    protection = 1 - 0.65 * (1 - distance / 35).clamp(0, 1)
    return _hsl_to_rgb(hue, (saturation + vibrance * (1 - saturation) * protection).clamp(0, 1), lightness)


def _denoise(x: Tensor, ps: Sequence[Mapping[str, object]]) -> Tensor:
    # Opponent-space chroma smoothing plus an edge-preserving luma blend.
    y = _luma(x)
    chroma = x - y
    y_soft = _blur(y, 2)
    c_soft = _blur(chroma, 3)
    ls = _v([float(p["luminance"]) / 100 for p in ps], x)
    cs = _v([float(p["color"]) / 100 for p in ps], x)
    edge = torch.exp(-16 * (y - y_soft).abs())
    return (y + ls * edge * (y_soft - y)) + chroma + cs * (c_soft - chroma)


def _texture(x: Tensor, ps: Sequence[Mapping[str, object]]) -> Tensor:
    result = x
    broad = _blur(result, max(3, min(x.shape[-2:]) // 35))
    dehaze = _v([float(p["dehaze"]) / 100 for p in ps], x)
    positive = result + 0.35 * dehaze.clamp_min(0) * (result - broad)
    luma = _luma(positive)
    positive = luma + (1 + 0.25 * dehaze.clamp_min(0)) * (positive - luma)
    atmosphere = broad.mean((-2, -1), keepdim=True)
    negative = result * (1 + dehaze.clamp_max(0) * 0.30) - dehaze.clamp_max(0) * 0.30 * atmosphere
    result = torch.where(dehaze >= 0, positive, negative)
    clarity = _v([float(p["clarity"]) / 100 for p in ps], x)
    result = result + 0.45 * clarity * (result - _blur(result, 4))
    texture = _v([float(p["texture"]) / 100 for p in ps], x)
    result = result + 0.30 * texture * (result - _blur(result, 1))
    sharp = _v([float(p["sharpening"]) / 150 for p in ps], x)
    return result + 0.55 * sharp * (result - _blur(result, 1))


def _optical(x: Tensor, ps: Sequence[Mapping[str, object]], frame_indices: Sequence[int]) -> Tensor:
    result = x
    diffusion = _v([float(p["diffusion"]["strength"]) / 100 for p in ps], x)
    result = result * (1 - 0.55 * diffusion) + _blur(result, 3) * (0.55 * diffusion)
    linear = _srgb_decode(result.clamp(0, 1))
    luminance = _luma(linear)
    threshold = _v([float(p["bloom"]["threshold"]) for p in ps], x)
    amount = _v([float(p["bloom"]["intensity"]) / 100 for p in ps], x)
    bright = linear * ((luminance - threshold) / (1 - threshold).clamp_min(1e-4)).clamp(0, 1)
    linear = linear + _blur(bright, max(2, min(x.shape[-2:]) // 80)) * amount * 0.8
    hthreshold = _v([float(p["halation"]["threshold"]) for p in ps], x)
    hamount = _v([float(p["halation"]["intensity"]) / 100 for p in ps], x)
    halo = _blur(((luminance - hthreshold) / (1 - hthreshold).clamp_min(1e-4)).clamp(0, 1), max(3, min(x.shape[-2:]) // 65))
    linear = linear + halo * x.new_tensor((0.14, 0.035, 0.008)).view(1, 3, 1, 1) * hamount
    result = _srgb_encode(linear)

    height, width = x.shape[-2:]
    yy = torch.linspace(-1, 1, height, dtype=x.dtype, device=x.device).view(1, 1, height, 1)
    xx = torch.linspace(-1, 1, width, dtype=x.dtype, device=x.device).view(1, 1, 1, width)
    roundness = _v([float(p["vignette"]["roundness"]) / 100 for p in ps], x)
    radius = ((xx * (1 + (-roundness).clamp_min(0) * 0.6)).square() + (yy * (1 + roundness.clamp_min(0) * 0.6)).square()).sqrt()
    midpoint = 0.15 + 0.70 * _v([float(p["vignette"]["midpoint"]) / 100 for p in ps], x)
    feather = 0.05 + 0.70 * _v([float(p["vignette"]["feather"]) / 100 for p in ps], x)
    vamount = _v([float(p["vignette"]["amount"]) / 100 for p in ps], x)
    result = result * (1 + vamount * 0.65 * _smoothstep(midpoint, midpoint + feather, radius))

    # Counter-based noise is deterministic per absolute frame on CPU and GPU.
    grain = _v([float(p["grain"]["amount"]) / 100 for p in ps], x)
    fi = _v([float(index) for index in frame_indices], x)
    phase = xx * 127.1 + yy * 311.7 + fi * 74.7
    noise = torch.frac(torch.sin(phase) * 43758.5453) * 2 - 1
    response = 0.45 + 0.55 * (1 - (2 * _luma(result).clamp(0, 1) - 1).abs())
    result = result + noise * response * grain * 0.055

    # Radial channel resampling for chromatic aberration.
    aberration = _v([float(p["chromatic_aberration"]["amount"]) / 100 * 0.012 for p in ps], x)
    if torch.any(aberration != 0):
        grid_y = yy.expand(x.shape[0], 1, height, width)
        grid_x = xx.expand(x.shape[0], 1, height, width)
        radius2 = grid_x.square() + grid_y.square()
        base = torch.cat((grid_x, grid_y), 1).permute(0, 2, 3, 1)
        delta = torch.cat((grid_x * radius2, grid_y * radius2), 1).permute(0, 2, 3, 1)
        red = F.grid_sample(result[:, 0:1], base + delta * aberration.permute(0, 2, 3, 1), padding_mode="reflection", align_corners=True)
        blue = F.grid_sample(result[:, 2:3], base - delta * aberration.permute(0, 2, 3, 1), padding_mode="reflection", align_corners=True)
        result = torch.cat((red, result[:, 1:2], blue), 1)
    return result


_FUNCTIONS = {
    "denoise": _denoise,
    "white_balance": _white_balance,
    "primary": _primary,
    "color_wheels": _wheels,
    "curves": _curves,
    "hsl8": _hsl8,
    "global_color": _global_color,
    "texture": _texture,
}


@dataclass
class TorchGradePoolExecutor:
    """Execute Pool graphs over a batch without CPU image round-trips."""

    clamp_output: bool = True

    def apply_batch(
        self,
        images: Tensor,
        operations: Sequence[object],
        *,
        frame_indices: Sequence[int],
        masks: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B,3,H,W].")
        if len(frame_indices) != images.shape[0]:
            raise ValueError("frame_indices length must equal the batch size.")
        unknown = {str(getattr(op, "operation_type")) for op in operations} - POOL_OPERATION_TYPES
        if unknown:
            raise ValueError(f"Torch Pool executor received non-Pool operations: {sorted(unknown)}")
        result = images
        for operation_type in POOL_PROCESSING_ORDER:
            for operation in operations:
                if str(getattr(operation, "operation_type")) != operation_type:
                    continue
                start, end = (int(v) for v in getattr(operation, "frame_range"))
                positions = [i for i, frame in enumerate(frame_indices) if start <= frame <= end]
                if not positions:
                    continue
                index = torch.as_tensor(positions, device=result.device)
                before = result.index_select(0, index)
                selected_frames = [int(frame_indices[i]) for i in positions]
                selected_ops = [operation] * len(positions)
                ps = _params(selected_ops, selected_frames)
                edited = _optical(before, ps, selected_frames) if operation_type == "optical_effects" else _FUNCTIONS[operation_type](before, ps)
                edited = edited.clamp(0, 1) if self.clamp_output else edited
                mask_id = str(getattr(operation, "mask_id", "global"))
                if mask_id != "global":
                    if masks is None or mask_id not in masks:
                        raise ValueError(f"Pool operation requires unavailable semantic mask: {mask_id}")
                    mask = masks[mask_id].to(device=result.device, dtype=result.dtype)
                    if mask.ndim == 3:
                        mask = mask.unsqueeze(1)
                    if mask.shape[0] != result.shape[0] or mask.shape[-2:] != result.shape[-2:]:
                        raise ValueError("Semantic mask batch and image dimensions do not match.")
                    amount = mask.index_select(0, index).clamp(0, 1)
                    edited = before * (1 - amount) + edited * amount
                result = result.index_copy(0, index, edited)
        return result
