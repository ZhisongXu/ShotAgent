"""Video decoding and artifact encoding for the video grading pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Iterable, Optional

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DecodedVideo:
    frames: tuple[Image.Image, ...]
    fps: float
    width: int
    height: int
    source: Path


VIDEO_CODECS = {
    ".avi": "MJPG",
    ".m4v": "mp4v",
    ".mov": "mp4v",
    ".mp4": "mp4v",
}


def decode_video(
    path: Path,
    max_frames: Optional[int] = None,
    max_side: Optional[int] = None,
) -> DecodedVideo:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Video has invalid FPS or dimensions.")
    if max_side is not None:
        if max_side < 64:
            capture.release()
            raise ValueError("max_side must be at least 64 pixels.")
        scale = min(1.0, max_side / max(width, height))
        width = max(1, round(width * scale))
        height = max(1, round(height * scale))
    frames: list[Image.Image] = []
    try:
        while max_frames is None or len(frames) < max_frames:
            success, bgr = capture.read()
            if not success:
                break
            if bgr.shape[1] != width or bgr.shape[0] != height:
                bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb, mode="RGB"))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Video contains no decodable frames: {path}")
    return DecodedVideo(
        frames=tuple(frames),
        fps=fps,
        width=width,
        height=height,
        source=path.resolve(),
    )


def encode_video(
    frames: Iterable[Image.Image],
    path: Path,
    fps: float,
    *,
    codec: Optional[str] = None,
) -> Path:
    """Encode ordered RGB frames as a silent video artifact.

    The encoder deliberately writes exactly the frames evaluated by the Agent.
    This means a run using ``max_frames`` or resized inputs exports that same
    processed view rather than silently copying the original source container.
    """

    path = Path(path).resolve()
    if fps <= 0:
        raise ValueError("fps must be positive.")
    iterator = iter(frames)
    try:
        first = next(iterator).convert("RGB")
    except StopIteration as error:
        raise ValueError("At least one frame is required to encode a video.") from error

    selected_codec = codec or VIDEO_CODECS.get(path.suffix.lower())
    if selected_codec is None:
        supported = ", ".join(sorted(VIDEO_CODECS))
        raise ValueError(f"Unsupported video suffix {path.suffix!r}; use {supported}.")
    if len(selected_codec) != 4:
        raise ValueError("Video codec must be a four-character code.")

    width, height = first.size
    if width <= 0 or height <= 0:
        raise ValueError("Video frames must have positive dimensions.")
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_frames = chain((first,), iterator)
    if path.suffix.lower() in {".mp4", ".m4v", ".mov"} and codec is None:
        writer = imageio_ffmpeg.write_frames(
            str(path),
            (width, height),
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            fps=float(fps),
            quality=7.0,
            codec="libx264",
            macro_block_size=2,
            ffmpeg_log_level="warning",
            output_params=["-preset", "medium", "-movflags", "+faststart"],
        )
        writer.send(None)
        try:
            for frame in ordered_frames:
                rgb_frame = frame.convert("RGB")
                if rgb_frame.size != (width, height):
                    raise ValueError("All video frames must have identical dimensions.")
                writer.send(np.ascontiguousarray(rgb_frame, dtype=np.uint8))
        finally:
            writer.close()
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Video writer produced no output: {path}")
        return path

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*selected_codec),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"Cannot open video writer for {path} with codec {selected_codec}."
        )
    try:
        for frame in ordered_frames:
            rgb_frame = frame.convert("RGB")
            if rgb_frame.size != (width, height):
                raise ValueError("All video frames must have identical dimensions.")
            rgb = np.asarray(rgb_frame, dtype=np.uint8)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Video writer produced no output: {path}")
    return path
