"""FFmpeg-backed 16-bit RGB decode and 10/12-bit delivery encoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Iterable, Iterator, Optional

import cv2
import imageio_ffmpeg
import numpy as np


@dataclass(frozen=True)
class HighBitVideoInfo:
    source: Path
    width: int
    height: int
    fps: float
    frame_count: int
    source_bit_depth: int
    color_space_hint: str


def has_audio_stream(path: Path) -> bool:
    """Return whether FFmpeg can map a first audio stream from the input."""

    completed = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(Path(path).resolve()),
            "-map",
            "0:a:0",
            "-frames:a",
            "1",
            "-f",
            "null",
            "-",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def _video_geometry(path: Path, max_side: Optional[int]) -> HighBitVideoInfo:
    source = Path(path).resolve()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Video has invalid FPS or dimensions.")
    if max_side is not None:
        if max_side < 64:
            raise ValueError("max_side must be at least 64 pixels.")
        scale = min(1.0, float(max_side) / max(width, height))
        width = max(1, round(width * scale))
        height = max(1, round(height * scale))
        # Most 4:2:0 delivery codecs require even dimensions.
        width -= width % 2
        height -= height % 2
    return HighBitVideoInfo(
        source=source,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        source_bit_depth=16,
        color_space_hint="unknown",
    )


def iter_decode_video_rgb16(
    path: Path,
    *,
    max_frames: Optional[int] = None,
    max_side: Optional[int] = None,
) -> tuple[HighBitVideoInfo, Iterator[np.ndarray]]:
    """Stream decoded frames as normalized float32 RGB from rgb48le."""

    info = _video_geometry(path, max_side)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(info.source),
        "-map",
        "0:v:0",
    ]
    original = _video_geometry(path, None)
    if (info.width, info.height) != (original.width, original.height):
        command.extend(
            [
                "-vf",
                f"scale={info.width}:{info.height}:flags=lanczos",
            ]
        )
    command.extend(
        [
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb48le",
            "pipe:1",
        ]
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
    )
    if process.stdout is None:
        process.kill()
        raise RuntimeError("FFmpeg decoder did not expose stdout.")
    frame_bytes = info.width * info.height * 3 * 2

    def frames() -> Iterator[np.ndarray]:
        count = 0
        stopped_early = False
        try:
            while max_frames is None or count < max_frames:
                chunks = bytearray()
                while len(chunks) < frame_bytes:
                    block = process.stdout.read(frame_bytes - len(chunks))
                    if not block:
                        break
                    chunks.extend(block)
                if not chunks:
                    break
                if len(chunks) != frame_bytes:
                    raise RuntimeError("FFmpeg returned a truncated 16-bit frame.")
                frame = np.frombuffer(chunks, dtype="<u2").reshape(
                    info.height, info.width, 3
                )
                yield frame.astype(np.float32) / 65535.0
                count += 1
            stopped_early = max_frames is not None and count >= max_frames
        finally:
            process.stdout.close()
            if stopped_early and process.poll() is None:
                process.terminate()
            return_code = process.wait(timeout=30)
            if return_code != 0 and not stopped_early:
                detail = (
                    process.stderr.read().decode("utf-8", errors="replace")[-2000:]
                    if process.stderr is not None
                    else ""
                )
                raise RuntimeError(f"FFmpeg high-bit decode failed: {detail}")

    return info, frames()


def decode_video_rgb16(
    path: Path,
    *,
    max_frames: Optional[int] = None,
    max_side: Optional[int] = None,
) -> tuple[HighBitVideoInfo, tuple[np.ndarray, ...]]:
    info, iterator = iter_decode_video_rgb16(
        path, max_frames=max_frames, max_side=max_side
    )
    frames = tuple(iterator)
    if not frames:
        raise RuntimeError(f"Video contains no decodable frames: {path}")
    return info, frames


def encode_video_high_bit(
    frames: Iterable[np.ndarray],
    path: Path,
    fps: float,
    *,
    bit_depth: int = 10,
    color_space: str = "rec709",
    audio_source: Optional[Path] = None,
) -> Path:
    """Encode float RGB frames to HEVC 10/12-bit and optionally copy audio."""

    if bit_depth not in {10, 12}:
        raise ValueError("High-bit output bit_depth must be 10 or 12.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    iterator = iter(frames)
    try:
        first = np.asarray(next(iterator), dtype=np.float32)
    except StopIteration as error:
        raise ValueError("At least one frame is required.") from error
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("High-bit frames must have shape [H,W,3].")
    height, width = first.shape[:2]
    if min(height, width) < 64:
        raise ValueError(
            "HEVC high-bit output requires both frame dimensions to be at least 64 pixels."
        )
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if audio_source is not None and Path(audio_source).resolve() == output:
        raise ValueError("High-bit output cannot overwrite its audio source.")
    pix_fmt = "yuv420p10le" if bit_depth == 10 else "yuv444p12le"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb48le",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.9g}",
        "-i",
        "pipe:0",
    ]
    if audio_source is not None:
        command.extend(["-i", str(Path(audio_source).resolve())])
    command.extend(["-map", "0:v:0"])
    if audio_source is not None:
        command.extend(["-map", "1:a?", "-c:a", "copy"])
    command.extend(
        [
            "-c:v",
            "libx265",
            "-preset",
            "medium",
            "-crf",
            "12",
            "-x265-params",
            "pools=1:frame-threads=1",
            "-pix_fmt",
            pix_fmt,
            "-tag:v",
            "hvc1",
        ]
    )
    if color_space == "rec2020_pq":
        command.extend(
            [
                "-color_primaries",
                "bt2020",
                "-color_trc",
                "smpte2084",
                "-colorspace",
                "bt2020nc",
            ]
        )
    elif color_space == "rec2020_hlg":
        command.extend(
            [
                "-color_primaries",
                "bt2020",
                "-color_trc",
                "arib-std-b67",
                "-colorspace",
                "bt2020nc",
            ]
        )
    else:
        command.extend(
            [
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-colorspace",
                "bt709",
            ]
        )
    if audio_source is not None:
        command.append("-shortest")
    command.extend(["-movflags", "+faststart", str(output)])
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdin is None:
        process.kill()
        raise RuntimeError("FFmpeg encoder did not expose stdin.")

    def write(frame: np.ndarray) -> None:
        values = np.asarray(frame, dtype=np.float32)
        if values.shape != (height, width, 3):
            raise ValueError("All high-bit frames must have the same RGB dimensions.")
        encoded = (np.clip(values, 0.0, 1.0) * 65535.0 + 0.5).astype("<u2")
        process.stdin.write(np.ascontiguousarray(encoded).tobytes())

    try:
        write(first)
        for frame in iterator:
            write(frame)
    finally:
        process.stdin.close()
    return_code = process.wait(timeout=120)
    if return_code != 0:
        detail = (
            process.stderr.read().decode("utf-8", errors="replace")[-3000:]
            if process.stderr is not None
            else ""
        )
        raise RuntimeError(f"FFmpeg high-bit encode failed: {detail}")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("FFmpeg high-bit encoder produced no output.")
    return output
