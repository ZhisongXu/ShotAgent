"""High-resolution rendering of dense video grading trajectories."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Optional

import numpy as np
import torch
from PIL import Image

from retouch_agent import RetouchExecutor

from .operations import OperationExecutor


def render_grade_frames(
    frames: Sequence[Image.Image],
    frame_parameters: np.ndarray,
    *,
    executor: Optional[RetouchExecutor] = None,
    operations: Sequence[object] = (),
    operation_executor: Optional[OperationExecutor] = None,
    batch_size: int = 8,
    device: Optional[str] = None,
) -> Iterator[Image.Image]:
    """Render a dense trajectory in GPU/CPU batches without buffering output."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    parameters = np.asarray(frame_parameters, dtype=np.float32)
    if parameters.shape != (len(frames), 12):
        raise ValueError("frame_parameters must have shape [number_of_frames, 12].")
    if not frames:
        return
    size = frames[0].size
    if any(frame.size != size for frame in frames):
        raise ValueError("All video frames must have identical dimensions.")

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    retoucher = executor or RetouchExecutor()
    post_executor = operation_executor or OperationExecutor()
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start : start + batch_size]
            arrays = (
                np.stack(
                    [
                        np.asarray(frame.convert("RGB"), dtype=np.float32)
                        for frame in batch_frames
                    ]
                )
                / 255.0
            )
            images = (
                torch.from_numpy(arrays)
                .permute(0, 3, 1, 2)
                .to(selected_device, non_blocking=True)
            )
            values = torch.from_numpy(parameters[start : start + len(batch_frames)]).to(
                selected_device, non_blocking=True
            )
            rendered = retoucher.apply_vector(images, values)
            output = (
                rendered.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0 + 0.5
            ).astype(np.uint8)
            for offset, array in enumerate(output):
                image = Image.fromarray(array, mode="RGB")
                if operations:
                    image = post_executor.apply(
                        image,
                        operations,
                        frame_index=start + offset,
                    )
                yield image
