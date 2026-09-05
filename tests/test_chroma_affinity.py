import numpy as np
from PIL import Image

from video_retouch.color_science import LumaPreservingChromaMatcher


def _lightness(image: Image.Image) -> np.ndarray:
    import cv2

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]


def test_chroma_affinity_preserves_lightness_and_reuses_one_transform() -> None:
    source = (
        Image.new("RGB", (32, 20), (30, 80, 45)),
        Image.new("RGB", (32, 20), (45, 100, 60)),
    )
    reference = (
        Image.new("RGB", (32, 20), (120, 85, 55)),
        Image.new("RGB", (32, 20), (180, 145, 105)),
    )
    output, audit = LumaPreservingChromaMatcher(strength=0.6).transfer_video(
        source, reference
    )

    assert len(output) == len(source)
    assert audit["temporal_policy"] == "one_transform_for_entire_video"
    assert audit["reference_source"] == "input_reference_video_only"
    for before, after in zip(source, output):
        assert float(np.mean(np.abs(_lightness(before) - _lightness(after)))) < 0.5
