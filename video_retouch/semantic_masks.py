"""Deterministic semantic masks and optical-flow mask tracking."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from PIL import Image


SEMANTIC_MASK_TYPES = frozenset({"skin", "sky", "person"})


def _rgb(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    values = np.asarray(image)
    if values.dtype != np.uint8:
        values = (np.clip(values, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return values


def _soften(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    values = np.asarray(mask, dtype=np.float32)
    values = cv2.morphologyEx(
        values, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
    )
    values = cv2.morphologyEx(
        values, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8)
    )
    return np.clip(cv2.GaussianBlur(values, (0, 0), sigmaX=sigma), 0.0, 1.0)


class SemanticMaskGenerator:
    """Generate skin, sky, and person masks without a remote segmentation API."""

    def __init__(self) -> None:
        self._hog: cv2.HOGDescriptor | None = None

    @staticmethod
    def skin(image: Image.Image | np.ndarray) -> np.ndarray:
        rgb = _rgb(image)
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        _, cr, cb = cv2.split(ycrcb)
        hue, saturation, value = cv2.split(hsv)
        chroma = (cr >= 132) & (cr <= 182) & (cb >= 72) & (cb <= 138)
        hue_skin = ((hue <= 25) | (hue >= 170)) & (saturation >= 20)
        valid_value = (value >= 25) & (value <= 250)
        return _soften((chroma & hue_skin & valid_value).astype(np.float32), 2.2)

    @staticmethod
    def sky(image: Image.Image | np.ndarray) -> np.ndarray:
        rgb = _rgb(image)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, saturation, value = cv2.split(hsv)
        blue = (hue >= 78) & (hue <= 132) & (saturation >= 25) & (value >= 45)
        neutral_bright = (saturation < 45) & (value >= 155)
        height, width = blue.shape
        vertical = 1.0 - np.arange(height, dtype=np.float32)[:, None] / max(height - 1, 1)
        prior = np.broadcast_to(vertical, (height, width))
        candidate = blue.astype(np.float32) * (0.45 + 0.55 * prior)
        candidate += neutral_bright.astype(np.float32) * np.clip((prior - 0.45) * 1.5, 0, 0.55)
        # Keep components that touch the upper image region or are very large.
        binary = (candidate > 0.35).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        selected = np.zeros_like(candidate)
        for label in range(1, count):
            x, y, component_width, component_height, area = stats[label]
            del x, component_width, component_height
            if y <= height * 0.20 or area >= height * width * 0.12:
                selected[labels == label] = candidate[labels == label]
        return _soften(selected, 3.0)

    def person(self, image: Image.Image | np.ndarray) -> np.ndarray:
        rgb = _rgb(image)
        height, width = rgb.shape[:2]
        scale = min(1.0, 640.0 / max(height, width))
        proxy = (
            cv2.resize(
                rgb,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            if scale < 1.0
            else rgb
        )
        # OpenCV's default HOG window is 64x128; some builds crash instead of
        # returning an empty result when detectMultiScale receives less.
        if proxy.shape[0] >= 128 and proxy.shape[1] >= 64:
            if self._hog is None:
                self._hog = cv2.HOGDescriptor()
                self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            boxes, weights = self._hog.detectMultiScale(
                cv2.cvtColor(proxy, cv2.COLOR_RGB2BGR),
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05,
            )
        else:
            boxes, weights = (), ()
        mask = np.zeros(proxy.shape[:2], dtype=np.float32)
        proxy_bgr = cv2.cvtColor(proxy, cv2.COLOR_RGB2BGR)
        for (x, y, box_width, box_height), confidence in zip(boxes, weights):
            if float(confidence) < 0.2:
                continue
            # Refine the detector rectangle to pixel support with GrabCut.  A
            # soft ellipse remains a safe fallback for degenerate rectangles.
            inset_x = max(1, round(box_width * 0.03))
            inset_y = max(1, round(box_height * 0.01))
            rect = (
                max(0, int(x + inset_x)),
                max(0, int(y + inset_y)),
                min(proxy.shape[1] - int(x + inset_x), max(2, int(box_width - 2 * inset_x))),
                min(proxy.shape[0] - int(y + inset_y), max(2, int(box_height - 2 * inset_y))),
            )
            component = np.zeros(proxy.shape[:2], dtype=np.float32)
            if rect[2] > 1 and rect[3] > 1:
                grab_mask = np.zeros(proxy.shape[:2], dtype=np.uint8)
                background = np.zeros((1, 65), dtype=np.float64)
                foreground = np.zeros((1, 65), dtype=np.float64)
                try:
                    cv2.grabCut(
                        proxy_bgr,
                        grab_mask,
                        rect,
                        background,
                        foreground,
                        2,
                        cv2.GC_INIT_WITH_RECT,
                    )
                    component = np.isin(
                        grab_mask, (cv2.GC_FGD, cv2.GC_PR_FGD)
                    ).astype(np.float32)
                except cv2.error:
                    component.fill(0)
            if np.max(component) <= 0:
                center = (x + box_width // 2, y + box_height // 2)
                axes = (max(1, box_width // 2), max(1, box_height // 2))
                cv2.ellipse(component, center, axes, 0, 0, 360, 1.0, -1)
            mask = np.maximum(mask, component)
        skin = self.skin(proxy)
        if np.max(mask) <= 0.0:
            # The fallback is conservative: expand detected skin into a likely
            # upper-body region rather than marking the full frame as a person.
            seed = (skin > 0.35).astype(np.uint8)
            kernel_size = max(9, round(min(proxy.shape[:2]) * 0.08)) | 1
            expanded = cv2.dilate(
                seed,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
                ),
            )
            mask = expanded.astype(np.float32) * 0.75
        else:
            mask = np.maximum(mask, skin)
        mask = _soften(mask, 3.5)
        if scale < 1.0:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.clip(mask, 0.0, 1.0)

    def generate(
        self, image: Image.Image | np.ndarray, mask_type: str
    ) -> np.ndarray:
        if mask_type == "skin":
            return self.skin(image)
        if mask_type == "sky":
            return self.sky(image)
        if mask_type == "person":
            return self.person(image)
        raise ValueError(f"Unknown semantic mask: {mask_type}")


class SemanticMaskTracker:
    """Track semantic masks with backward optical flow and periodic refresh."""

    def __init__(
        self,
        generator: SemanticMaskGenerator | None = None,
        detection_interval: int = 6,
    ) -> None:
        if detection_interval < 1:
            raise ValueError("detection_interval must be positive.")
        self.generator = generator or SemanticMaskGenerator()
        self.detection_interval = int(detection_interval)

    @staticmethod
    def _gray(frame: Image.Image | np.ndarray) -> np.ndarray:
        return cv2.cvtColor(_rgb(frame), cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _warp_previous(
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        previous_mask: np.ndarray,
    ) -> np.ndarray:
        # Backward flow gives, for each current pixel, its coordinate in the
        # previous frame, which is directly usable by remap.
        backward = cv2.calcOpticalFlowFarneback(
            current_gray,
            previous_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        height, width = current_gray.shape
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        return cv2.remap(
            previous_mask,
            grid_x + backward[..., 0],
            grid_y + backward[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def track(
        self,
        frames: Sequence[Image.Image | np.ndarray],
        mask_type: str,
    ) -> tuple[np.ndarray, ...]:
        if mask_type not in SEMANTIC_MASK_TYPES:
            raise ValueError(f"Unknown semantic mask: {mask_type}")
        if not frames:
            return ()
        result: list[np.ndarray] = []
        previous_gray: np.ndarray | None = None
        previous_mask: np.ndarray | None = None
        for index, frame in enumerate(frames):
            current_gray = self._gray(frame)
            detected = (
                self.generator.generate(frame, mask_type)
                if index == 0
                or index % self.detection_interval == 0
                or mask_type in {"skin", "sky"}
                else None
            )
            if previous_gray is None or previous_mask is None:
                current = detected
                assert current is not None
            else:
                warped = self._warp_previous(previous_gray, current_gray, previous_mask)
                current = warped if detected is None else 0.55 * detected + 0.45 * warped
            current = np.clip(cv2.GaussianBlur(current, (0, 0), 1.2), 0.0, 1.0)
            result.append(current.astype(np.float32))
            previous_gray = current_gray
            previous_mask = current
        return tuple(result)

    def track_many(
        self,
        frames: Sequence[Image.Image | np.ndarray],
        mask_types: Sequence[str],
    ) -> dict[str, tuple[np.ndarray, ...]]:
        return {
            mask_type: self.track(frames, mask_type)
            for mask_type in dict.fromkeys(mask_types)
        }
