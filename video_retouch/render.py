"""High-resolution rendering of dense video grading trajectories."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Optional

import numpy as np
import torch
from PIL import Image

from retouch_agent import RetouchExecutor

from .gpu_pool_executor import TorchGradePoolExecutor
from .grade_pools import POOL_OPERATION_TYPES
from .operations import OperationExecutor
from .semantic_masks import SemanticMaskTracker


def render_grade_frames(
    frames: Sequence[Image.Image],
    frame_parameters: np.ndarray,
    *,
    executor: Optional[RetouchExecutor] = None,
    operations: Sequence[object] = (),
    operation_executor: Optional[OperationExecutor] = None,
    batch_size: int = 8,
    device: Optional[str] = None,
    mask_tracker: Optional[SemanticMaskTracker] = None,
    use_torch_pools: bool = True,
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
    pool_executor = TorchGradePoolExecutor()
    pool_operations = tuple(
        operation
        for operation in operations
        if str(getattr(operation, "operation_type")) in POOL_OPERATION_TYPES
    )
    legacy_operations = tuple(
        operation
        for operation in operations
        if str(getattr(operation, "operation_type")) not in POOL_OPERATION_TYPES
    )
    pre_operations = (
        legacy_operations
        if use_torch_pools
        else legacy_operations + pool_operations
    )
    mask_ids = sorted(
        {
            str(getattr(operation, "mask_id", "global"))
            for operation in operations
            if str(getattr(operation, "mask_id", "global")) != "global"
        }
    )
    mask_tracks = (
        (mask_tracker or SemanticMaskTracker()).track_many(frames, mask_ids)
        if mask_ids
        else {}
    )
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start : start + batch_size]
            if pre_operations:
                batch_frames = tuple(
                    post_executor.apply_pre_grade(
                        frame,
                        pre_operations,
                        frame_index=start + offset,
                        masks={
                            mask_id: mask_tracks[mask_id][start + offset]
                            for mask_id in mask_ids
                        },
                    )
                    for offset, frame in enumerate(batch_frames)
                )
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
            if pool_operations and use_torch_pools:
                batch_masks = {
                    mask_id: torch.from_numpy(
                        np.stack(mask_tracks[mask_id][start : start + len(batch_frames)])
                    ).to(selected_device, non_blocking=True)
                    for mask_id in mask_ids
                }
                rendered = pool_executor.apply_batch(
                    rendered,
                    pool_operations,
                    frame_indices=range(start, start + len(batch_frames)),
                    masks=batch_masks,
                )
            output = (
                rendered.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0 + 0.5
            ).astype(np.uint8)
            for offset, array in enumerate(output):
                image = Image.fromarray(array, mode="RGB")
                if pool_operations and not use_torch_pools:
                    image = post_executor.apply_post_grade(
                        image,
                        pool_operations,
                        frame_index=start + offset,
                        masks={
                            mask_id: mask_tracks[mask_id][start + offset]
                            for mask_id in mask_ids
                        },
                    )
                if legacy_operations:
                    image = post_executor.apply_post_grade(
                        image,
                        legacy_operations,
                        frame_index=start + offset,
                        masks={
                            mask_id: mask_tracks[mask_id][start + offset]
                            for mask_id in mask_ids
                        },
                    )
                yield image
