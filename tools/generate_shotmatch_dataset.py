"""Generate a synthetic Hero-referenced shot-matching video dataset.

Example:
    python tools/generate_shotmatch_dataset.py \
        --input-dir /datasets/mastered_multishot_clips \
        --output-dir outputs/shotmatch_v1 \
        --variants 3 --track realisp --seed 2026

Every source video is treated as ground truth. One automatically selected Hero
shot is copied unchanged; every other shot receives an independent camera-
linear exposure, correlated CCT/tint white balance, real-camera CCM, and (for
the default ``realisp`` track) nonlinear ISP perturbation. Parameters remain
constant within a shot unless ``--track temporal`` is selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video_retouch.io import decode_video
from video_retouch.shot_planner import HeuristicShotPlanner

VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
TRACKS = ("global", "realisp", "temporal")

# The four empirical XYZ -> camera matrices from the official implementation
# of Brooks et al., "Unprocessing Images for Learned Raw Denoising", CVPR 2019.
XYZ_TO_CAMERA = np.asarray(
    [
        [
            [1.0234, -0.2969, -0.2266],
            [-0.5625, 1.6328, -0.0469],
            [-0.0703, 0.2188, 0.6406],
        ],
        [
            [0.4913, -0.0541, -0.0202],
            [-0.6130, 1.3513, 0.2906],
            [-0.1564, 0.2151, 0.7183],
        ],
        [
            [0.8380, -0.2630, -0.0639],
            [-0.2887, 1.0725, 0.2496],
            [-0.0627, 0.1427, 0.5438],
        ],
        [
            [0.6596, -0.2079, -0.0562],
            [-0.4782, 1.3016, 0.1933],
            [-0.0970, 0.1581, 0.5181],
        ],
    ],
    dtype=np.float64,
)

SRGB_TO_XYZ = np.asarray(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

RANGES = {
    "mild": {
        "exposure_ev": (0.15, 0.40),
        "mired_delta": (20.0, 50.0),
        "tint_uv": (0.002, 0.004),
        "ccm_mix": (0.15, 0.35),
        "black_level": (0.0005, 0.0020),
        "white_level_delta": (0.015, 0.040),
        "tone_delta": (0.04, 0.10),
        "black_lift": (0.002, 0.008),
        "hue_degrees": (1.0, 3.0),
        "saturation_delta": (0.02, 0.05),
        "rolloff": (0.03, 0.08),
        "settling_seconds": (0.25, 0.60),
        "overshoot": (0.10, 0.25),
    },
    "medium": {
        "exposure_ev": (0.35, 0.80),
        "mired_delta": (50.0, 100.0),
        "tint_uv": (0.004, 0.008),
        "ccm_mix": (0.35, 0.65),
        "black_level": (0.0015, 0.0050),
        "white_level_delta": (0.035, 0.090),
        "tone_delta": (0.08, 0.18),
        "black_lift": (0.006, 0.018),
        "hue_degrees": (2.5, 7.0),
        "saturation_delta": (0.04, 0.12),
        "rolloff": (0.07, 0.16),
        "settling_seconds": (0.40, 1.00),
        "overshoot": (0.18, 0.40),
    },
    "hard": {
        "exposure_ev": (0.70, 1.25),
        "mired_delta": (100.0, 180.0),
        "tint_uv": (0.008, 0.015),
        "ccm_mix": (0.60, 1.00),
        "black_level": (0.0040, 0.0120),
        "white_level_delta": (0.070, 0.160),
        "tone_delta": (0.15, 0.32),
        "black_lift": (0.012, 0.035),
        "hue_degrees": (6.0, 14.0),
        "saturation_delta": (0.10, 0.24),
        "rolloff": (0.14, 0.30),
        "settling_seconds": (0.70, 1.60),
        "overshoot": (0.30, 0.60),
    },
}


@dataclass(frozen=True)
class CameraState:
    weights: np.ndarray
    rgb_to_camera: np.ndarray
    red_gain: float
    blue_gain: float
    cct_kelvin: float
    tint_uv: float

    @property
    def gains(self) -> np.ndarray:
        return np.asarray([self.red_gain, 1.0, self.blue_gain], dtype=np.float64)

    @property
    def camera_to_rgb(self) -> np.ndarray:
        return np.linalg.inv(self.rgb_to_camera)

    def to_dict(self) -> dict[str, object]:
        return {
            "ccm_weights": self.weights.tolist(),
            "rgb_to_camera": self.rgb_to_camera.tolist(),
            "camera_to_rgb": self.camera_to_rgb.tolist(),
            "wb_gains": self.gains.tolist(),
            "cct_kelvin": self.cct_kelvin,
            "mired": 1_000_000.0 / self.cct_kelvin,
            "tint_uv": self.tint_uv,
        }


@dataclass(frozen=True)
class ISPState:
    black_level: float = 0.0
    white_level: float = 1.0
    tone_contrast: float = 1.0
    black_lift: float = 0.0
    hue_degrees: float = 0.0
    saturation: float = 1.0
    highlight_rolloff: float = 0.0

    def scaled(self, scale: float) -> ISPState:
        return ISPState(
            black_level=self.black_level * scale,
            white_level=1.0 + (self.white_level - 1.0) * scale,
            tone_contrast=1.0 + (self.tone_contrast - 1.0) * scale,
            black_lift=self.black_lift * scale,
            hue_degrees=self.hue_degrees * scale,
            saturation=1.0 + (self.saturation - 1.0) * scale,
            highlight_rolloff=self.highlight_rolloff * scale,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "black_level": self.black_level,
            "white_level": self.white_level,
            "tone_contrast": self.tone_contrast,
            "black_lift": self.black_lift,
            "hue_degrees": self.hue_degrees,
            "saturation": self.saturation,
            "highlight_rolloff": self.highlight_rolloff,
        }


@dataclass(frozen=True)
class Perturbation:
    exposure_ev: float
    ccm_mix: float
    target: CameraState
    isp: ISPState
    settling_seconds: float = 0.0
    settling_overshoot: float = 0.0


def _signed(rng: np.random.Generator, limits: tuple[float, float]) -> float:
    return (-1.0 if rng.integers(0, 2) == 0 else 1.0) * float(rng.uniform(*limits))


def _cct_xy(cct: float) -> tuple[float, float]:
    t = float(np.clip(cct, 1667.0, 25000.0))
    if t <= 4000.0:
        x = -0.2661239e9 / t**3 - 0.2343580e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2222.0:
        y = -1.1063814 * x**3 - 1.3481102 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000.0:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.8733867 * x**2 + 3.75112997 * x - 0.37001483
    return float(x), float(y)


def _illuminant_xyz(cct: float, tint_uv: float) -> np.ndarray:
    x, y = _cct_xy(cct)
    denominator = -2.0 * x + 12.0 * y + 3.0
    u, v = 4.0 * x / denominator, 6.0 * y / denominator + tint_uv
    denominator = 2.0 * u - 8.0 * v + 4.0
    x, y = 3.0 * u / denominator, 2.0 * v / denominator
    return np.asarray([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def _camera_state(weights: np.ndarray, cct: float, tint_uv: float) -> CameraState:
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 1e-8)
    weights /= weights.sum()
    xyz_to_camera = np.tensordot(weights, XYZ_TO_CAMERA, axes=(0, 0))
    rgb_to_camera = xyz_to_camera @ SRGB_TO_XYZ
    rgb_to_camera /= rgb_to_camera.sum(axis=1, keepdims=True)
    illuminant_rgb = np.linalg.solve(SRGB_TO_XYZ, _illuminant_xyz(cct, tint_uv))
    camera_white = np.maximum(rgb_to_camera @ illuminant_rgb, 1e-4)
    gains = camera_white[1] / camera_white
    gains /= gains[1]
    return CameraState(
        weights=weights,
        rgb_to_camera=rgb_to_camera,
        red_gain=float(gains[0]),
        blue_gain=float(gains[2]),
        cct_kelvin=float(cct),
        tint_uv=float(tint_uv),
    )


def _sample_reference(rng: np.random.Generator) -> CameraState:
    weights = rng.uniform(1e-8, 1.0, len(XYZ_TO_CAMERA))
    mired = float(rng.uniform(135.0, 330.0))
    return _camera_state(weights, 1_000_000.0 / mired, float(rng.normal(0, 0.0015)))


def _interpolate_camera(
    reference: CameraState, target: CameraState, scale: float
) -> CameraState:
    weights = reference.weights + scale * (target.weights - reference.weights)
    reference_mired = 1_000_000.0 / reference.cct_kelvin
    target_mired = 1_000_000.0 / target.cct_kelvin
    mired = float(
        np.clip(reference_mired + scale * (target_mired - reference_mired), 100, 450)
    )
    tint = reference.tint_uv + scale * (target.tint_uv - reference.tint_uv)
    return _camera_state(weights, 1_000_000.0 / mired, tint)


def _sample_perturbation(
    rng: np.random.Generator, reference: CameraState, severity: str, track: str
) -> Perturbation:
    limits = RANGES[severity]
    other = _sample_reference(rng)
    ccm_mix = float(rng.uniform(*limits["ccm_mix"]))
    weights = (1.0 - ccm_mix) * reference.weights + ccm_mix * other.weights
    reference_mired = 1_000_000.0 / reference.cct_kelvin
    target_mired = float(
        np.clip(reference_mired + _signed(rng, limits["mired_delta"]), 100, 450)
    )
    target = _camera_state(
        weights,
        1_000_000.0 / target_mired,
        reference.tint_uv + _signed(rng, limits["tint_uv"]),
    )
    isp = ISPState()
    if track != "global":
        isp = ISPState(
            black_level=_signed(rng, limits["black_level"]),
            white_level=1.0 + _signed(rng, limits["white_level_delta"]),
            tone_contrast=1.0 + _signed(rng, limits["tone_delta"]),
            black_lift=_signed(rng, limits["black_lift"]),
            hue_degrees=_signed(rng, limits["hue_degrees"]),
            saturation=1.0 + _signed(rng, limits["saturation_delta"]),
            highlight_rolloff=float(rng.uniform(*limits["rolloff"])),
        )
    return Perturbation(
        exposure_ev=_signed(rng, limits["exposure_ev"]),
        ccm_mix=ccm_mix,
        target=target,
        isp=isp,
        settling_seconds=(
            float(rng.uniform(*limits["settling_seconds"]))
            if track == "temporal"
            else 0.0
        ),
        settling_overshoot=(
            _signed(rng, limits["overshoot"]) if track == "temporal" else 0.0
        ),
    )


def _inverse_smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return 0.5 - np.sin(np.arcsin(1.0 - 2.0 * value) / 3.0)


def _unprocess(frame: Image.Image, reference: CameraState) -> np.ndarray:
    srgb = np.asarray(frame.convert("RGB"), dtype=np.float64) / 255.0
    linear = np.maximum(_inverse_smoothstep(srgb), 1e-8) ** 2.2
    return (linear @ reference.rgb_to_camera.T) / reference.gains[None, None, :]


def _render(
    raw: np.ndarray,
    camera: CameraState,
    exposure_ev: float,
    isp: ISPState,
) -> tuple[Image.Image, float]:
    white_range = max(isp.white_level - isp.black_level, 1e-3)
    raw = (raw - isp.black_level) / white_range
    camera_rgb = raw * math.exp2(exposure_ev) * camera.gains[None, None, :]
    linear = camera_rgb @ camera.camera_to_rgb.T
    clipped = np.logical_or(linear < 0.0, linear > 1.0)
    linear = np.clip(linear, 0.0, 1.0).astype(np.float32)
    if isp.hue_degrees != 0.0 or isp.saturation != 1.0:
        hsv = cv2.cvtColor(linear, cv2.COLOR_RGB2HSV)
        radians = np.deg2rad(hsv[..., 0])
        hsv[..., 0] = np.mod(hsv[..., 0] + isp.hue_degrees * np.sin(radians), 360)
        hue_dependent_saturation = isp.saturation * (1.0 + 0.15 * np.cos(2 * radians))
        hsv[..., 1] = np.clip(hsv[..., 1] * hue_dependent_saturation, 0.0, 1.0)
        linear = np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0)
    shoulder = max(isp.highlight_rolloff, -0.45)
    linear = (1.0 + shoulder) * linear / (1.0 + shoulder * linear)
    linear += isp.black_lift * (1.0 - linear)
    linear = np.clip(linear, 0.0, 1.0)
    contrast = max(isp.tone_contrast, 0.25)
    powered = np.power(linear, contrast)
    linear = powered / np.maximum(powered + np.power(1.0 - linear, contrast), 1e-8)
    gamma = np.maximum(linear, 1e-8) ** (1.0 / 2.2)
    srgb = gamma * gamma * (3.0 - 2.0 * gamma)
    image = Image.fromarray((np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")
    return image, float(np.mean(clipped))


def _encode_lossless(frames: Iterable[Image.Image], path: Path, fps: float) -> None:
    iterator = iter(frames)
    first = next(iterator).convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(path),
        first.size,
        pix_fmt_in="rgb24",
        pix_fmt_out="bgr0",
        fps=fps,
        codec="ffv1",
        macro_block_size=1,
        ffmpeg_log_level="warning",
    )
    writer.send(None)
    try:
        for frame in chain((first,), iterator):
            writer.send(np.ascontiguousarray(frame.convert("RGB"), dtype=np.uint8))
    finally:
        writer.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_id(path: Path, root: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    value = "__".join(relative.parts)
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-") or "video"


def _variant_seed(seed: int, source_id: str, variant: int) -> int:
    value = hashlib.sha256(f"{seed}:{source_id}:{variant}".encode()).digest()
    return int.from_bytes(value[:8], "little")


def _synthesize(
    frames: Sequence[Image.Image],
    shots,
    hero_shot_id: int,
    fps: float,
    seed: int,
    severity: str,
    track: str,
) -> tuple[list[Image.Image], dict[str, object]]:
    rng = np.random.default_rng(seed)
    reference = _sample_reference(rng)
    output: list[Image.Image | None] = [None] * len(frames)
    labels = []
    for shot in shots:
        row: dict[str, object] = {
            "shot_id": shot.shot_id,
            "start_frame": shot.start_frame,
            "end_frame": shot.end_frame,
            "is_hero": shot.shot_id == hero_shot_id,
        }
        if shot.shot_id == hero_shot_id:
            for index in range(shot.start_frame, shot.end_frame + 1):
                output[index] = frames[index].convert("RGB").copy()
            row["perturbation"] = None
        else:
            perturbation = _sample_perturbation(rng, reference, severity, track)
            clipping = []
            scales = []
            for index in range(shot.start_frame, shot.end_frame + 1):
                elapsed = (index - shot.start_frame) / fps
                scale = (
                    1.0
                    + perturbation.settling_overshoot
                    * math.exp(-elapsed / perturbation.settling_seconds)
                    if track == "temporal"
                    else 1.0
                )
                camera = _interpolate_camera(reference, perturbation.target, scale)
                rendered, fraction = _render(
                    _unprocess(frames[index], reference),
                    camera,
                    perturbation.exposure_ev * scale,
                    perturbation.isp.scaled(scale),
                )
                output[index] = rendered
                clipping.append(fraction)
                scales.append(scale)
            target = perturbation.target
            linear_forward = (
                target.camera_to_rgb
                @ (math.exp2(perturbation.exposure_ev) * np.diag(target.gains))
                @ np.linalg.inv(np.diag(reference.gains))
                @ reference.rgb_to_camera
            )
            row["perturbation"] = {
                "track": track,
                "severity": severity,
                "exposure_ev": perturbation.exposure_ev,
                "reference_cct_kelvin": reference.cct_kelvin,
                "target_cct_kelvin": target.cct_kelvin,
                "mired_delta": 1_000_000 / target.cct_kelvin
                - 1_000_000 / reference.cct_kelvin,
                "tint_uv_delta": target.tint_uv - reference.tint_uv,
                "ccm_mix": perturbation.ccm_mix,
                "target_camera": target.to_dict(),
                "isp": perturbation.isp.to_dict(),
                "linear_core_gt_to_input": linear_forward.tolist(),
                "linear_core_input_to_gt": np.linalg.inv(linear_forward).tolist(),
                "temporal_response": (
                    None
                    if track != "temporal"
                    else {
                        "formula": "scale(t)=1+overshoot*exp(-t/tau)",
                        "tau_seconds": perturbation.settling_seconds,
                        "overshoot": perturbation.settling_overshoot,
                        "first_scale": scales[0],
                        "last_scale": scales[-1],
                    }
                ),
                "clipped_channel_fraction_mean": float(np.mean(clipping)),
                "clipped_channel_fraction_max": float(np.max(clipping)),
            }
        labels.append(row)
    if any(frame is None for frame in output):
        raise RuntimeError("Shot plan did not cover every frame.")
    return [frame for frame in output if frame is not None], {
        "schema": "shotmatch-synthetic/v1",
        "seed": seed,
        "track": track,
        "severity": severity,
        "hero_shot_id": hero_shot_id,
        "reference_camera": reference.to_dict(),
        "shots": labels,
    }


def generate(args: argparse.Namespace) -> dict[str, object]:
    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()
    videos = sorted(
        path.resolve()
        for path in input_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_SUFFIXES
        and path.match(args.include)
    )
    if not videos:
        raise RuntimeError(f"No videos found below {input_root}")
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass --overwrite to regenerate")
    samples = []
    skipped = []
    for video in videos:
        source_id = _safe_id(video, input_root)
        decoded = decode_video(
            video, max_frames=args.max_frames, max_side=args.max_side
        )
        storyboard = HeuristicShotPlanner(
            cut_threshold=args.cut_threshold,
            minimum_shot_seconds=args.minimum_shot_seconds,
        ).plan(decoded.frames, decoded.fps, "shot matching synthesis")
        if len(storyboard.shots) < 2:
            skipped.append({"id": source_id, "reason": "fewer than two shots"})
            continue
        hero_frame = int(storyboard.hero_anchor_frame)
        hero_shot_id = next(
            shot.shot_id
            for shot in storyboard.shots
            if shot.start_frame <= hero_frame <= shot.end_frame
        )
        gt_path = output_root / "ground_truth" / f"{source_id}.mkv"
        if args.overwrite or not gt_path.exists():
            _encode_lossless(decoded.frames, gt_path, decoded.fps)
        for variant in range(args.variants):
            severity = tuple(RANGES)[variant % len(RANGES)]
            sample_seed = _variant_seed(args.seed, source_id, variant)
            sample_id = f"{source_id}__v{variant:02d}_{args.track}_{severity}"
            input_path = output_root / "inputs" / f"{sample_id}.mkv"
            label_path = output_root / "labels" / f"{sample_id}.json"
            frames, label = _synthesize(
                decoded.frames,
                storyboard.shots,
                hero_shot_id,
                decoded.fps,
                sample_seed,
                severity,
                args.track,
            )
            if args.overwrite or not input_path.exists():
                _encode_lossless(frames, input_path, decoded.fps)
            label.update(
                {
                    "id": sample_id,
                    "source": str(video),
                    "source_sha256": _sha256(video),
                    "fps": decoded.fps,
                    "frame_count": len(decoded.frames),
                    "input": str(input_path.relative_to(output_root)),
                    "ground_truth": str(gt_path.relative_to(output_root)),
                }
            )
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(
                json.dumps(label, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            samples.append(
                {
                    "id": sample_id,
                    "source_id": source_id,
                    "severity": severity,
                    "track": args.track,
                    "input": str(input_path.relative_to(output_root)),
                    "ground_truth": str(gt_path.relative_to(output_root)),
                    "label": str(label_path.relative_to(output_root)),
                }
            )
    manifest = {
        "schema": "shotmatch-synthetic-dataset/v1",
        "seed": args.seed,
        "track": args.track,
        "variants_per_source": args.variants,
        "encoding": "FFV1 Matroska (lossless RGB)",
        "hero_policy": "unchanged full shot selected by the existing shot planner",
        "samples": samples,
        "skipped": skipped,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include",
        default="*",
        help="Only process paths matching this glob, for example '*_6fps.mp4'.",
    )
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--track", choices=TRACKS, default="realisp")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cut-threshold", type=float, default=0.42)
    parser.add_argument("--minimum-shot-seconds", type=float, default=0.4)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-side", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.variants < 1:
        raise ValueError("--variants must be positive")
    manifest = generate(args)
    print(
        json.dumps(
            {"samples": len(manifest["samples"]), "skipped": manifest["skipped"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
