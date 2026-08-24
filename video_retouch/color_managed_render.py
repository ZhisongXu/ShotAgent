"""Color-managed high-bit rendering for Pool grade graphs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Optional

import numpy as np
import torch

from retouch_agent import RetouchExecutor

from .color_management import ColorManager, OCIOColorManager
from .gpu_pool_executor import TorchGradePoolExecutor
from .grade_pools import POOL_OPERATION_TYPES
from .semantic_masks import SemanticMaskTracker


def _luma(rgb: np.ndarray) -> np.ndarray:
    return np.sum(
        rgb * np.asarray((0.2722287, 0.6740818, 0.0536895), np.float32),
        axis=-1,
        keepdims=True,
    )


def _transfer_proxy_grade(
    original_working: np.ndarray,
    proxy_working: np.ndarray,
    graded_working: np.ndarray,
) -> np.ndarray:
    """Transfer a display-referred grade while retaining scene-linear range."""

    proxy_luma = _luma(proxy_working)
    graded_luma = _luma(graded_working)
    scale = np.clip(
        (graded_luma + 1e-5) / (np.maximum(proxy_luma, 0.0) + 1e-5),
        0.125,
        8.0,
    )
    chroma_change = graded_working - proxy_working * scale
    return np.maximum(original_working * scale + chroma_change, 0.0).astype(np.float32)


def render_color_managed_frames(
    frames: Sequence[np.ndarray],
    frame_parameters: np.ndarray,
    *,
    operations: Sequence[object] = (),
    input_color_space: str = "srgb",
    working_color_space: str = "acescg",
    output_color_space: str = "rec709",
    executor: Optional[RetouchExecutor] = None,
    batch_size: int = 8,
    device: Optional[str] = None,
    mask_tracker: Optional[SemanticMaskTracker] = None,
    pq_reference_white_nits: float = 203.0,
    ocio_manager: OCIOColorManager | None = None,
    ocio_spaces: Mapping[str, str] | None = None,
) -> Iterator[np.ndarray]:
    """Render normalized float RGB frames without an 8-bit intermediate.

    Built-in mode converts through linear ACEScg.  OCIO mode uses the named
    input/working/display/output spaces supplied in ``ocio_spaces``.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if not 0 < pq_reference_white_nits <= 10_000:
        raise ValueError("pq_reference_white_nits must be in (0, 10000].")
    parameters = np.asarray(frame_parameters, dtype=np.float32)
    if parameters.shape != (len(frames), 12):
        raise ValueError("frame_parameters must have shape [number_of_frames, 12].")
    if not frames:
        return
    shape = np.asarray(frames[0]).shape
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError("Color-managed frames must have shape [H,W,3].")
    if any(np.asarray(frame).shape != shape for frame in frames):
        raise ValueError("All video frames must have identical dimensions.")
    non_pool = [
        str(getattr(operation, "operation_type"))
        for operation in operations
        if str(getattr(operation, "operation_type")) not in POOL_OPERATION_TYPES
    ]
    if non_pool:
        raise ValueError(
            "High-bit color-managed rendering accepts Pool v2 operations only; "
            f"unsupported operations: {sorted(set(non_pool))}"
        )

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    retoucher = executor or RetouchExecutor()
    pool_executor = TorchGradePoolExecutor()
    manager = ColorManager(working_color_space)
    spaces = dict(ocio_spaces or {})

    def to_working(values: np.ndarray) -> np.ndarray:
        if ocio_manager is None:
            working = manager.to_working(values, input_color_space)
            if input_color_space == "rec2020_pq":
                working = working * (10_000.0 / pq_reference_white_nits)
            return working
        return ocio_manager.convert(values, spaces["input"], spaces["working"])

    def working_to_proxy(values: np.ndarray) -> np.ndarray:
        if ocio_manager is None:
            return manager.display_proxy(values)
        return np.clip(
            ocio_manager.convert(values, spaces["working"], spaces["display"]),
            0,
            1,
        )

    def proxy_to_working(values: np.ndarray) -> np.ndarray:
        if ocio_manager is None:
            return manager.convert(values, "srgb", working_color_space)
        return ocio_manager.convert(values, spaces["display"], spaces["working"])

    def from_working(values: np.ndarray, graded_proxy: np.ndarray) -> np.ndarray:
        if ocio_manager is not None:
            return ocio_manager.convert(values, spaces["working"], spaces["output"])
        if output_color_space in {"srgb", "rec709"}:
            return ColorManager.convert(graded_proxy, "srgb", output_color_space)
        if output_color_space == "rec2020_pq":
            values = values * (pq_reference_white_nits / 10_000.0)
        return manager.from_working(values, output_color_space)

    mask_ids = sorted(
        {
            str(getattr(operation, "mask_id", "global"))
            for operation in operations
            if str(getattr(operation, "mask_id", "global")) != "global"
        }
    )
    if mask_ids:
        # Segmentation must see the same display-referred view as the VL model,
        # not a flat Log image or perceptually encoded HDR code values.
        mask_frames = tuple(
            working_to_proxy(to_working(np.asarray(frame, dtype=np.float32)))
            for frame in frames
        )
        mask_tracks = (mask_tracker or SemanticMaskTracker()).track_many(
            mask_frames, mask_ids
        )
    else:
        mask_tracks = {}

    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            stop = min(start + batch_size, len(frames))
            encoded = np.stack(
                [np.asarray(frame, dtype=np.float32) for frame in frames[start:stop]]
            )
            working = to_working(encoded)
            proxy = working_to_proxy(working)
            images = torch.from_numpy(np.ascontiguousarray(proxy)).permute(0, 3, 1, 2).to(
                selected_device, non_blocking=True
            )
            values = torch.from_numpy(parameters[start:stop]).to(
                selected_device, non_blocking=True
            )
            graded = retoucher.apply_vector(images, values)
            batch_masks = {
                mask_id: torch.from_numpy(
                    np.stack(mask_tracks[mask_id][start:stop])
                ).to(selected_device, non_blocking=True)
                for mask_id in mask_ids
            }
            if operations:
                graded = pool_executor.apply_batch(
                    graded,
                    operations,
                    frame_indices=range(start, stop),
                    masks=batch_masks,
                )
            graded_proxy = graded.permute(0, 2, 3, 1).cpu().numpy()
            proxy_working = proxy_to_working(proxy)
            graded_working = proxy_to_working(graded_proxy)
            result_working = _transfer_proxy_grade(
                working, proxy_working, graded_working
            )
            output = np.clip(
                from_working(result_working, graded_proxy), 0.0, 1.0
            ).astype(np.float32)
            yield from output
