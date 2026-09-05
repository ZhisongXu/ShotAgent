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


def test_chroma_affinity_can_blend_toward_corresponding_target_luma() -> None:
    rendered = (
        Image.new("RGB", (32, 20), (12, 28, 18)),
        Image.new("RGB", (32, 20), (18, 36, 24)),
    )
    target = (
        Image.new("RGB", (32, 20), (90, 110, 95)),
        Image.new("RGB", (32, 20), (120, 140, 125)),
    )
    reference = (Image.new("RGB", (32, 20), (160, 95, 60)),)
    output, audit = LumaPreservingChromaMatcher(
        strength=0.6,
        target_luma_strength=0.5,
    ).transfer_video(rendered, reference, target_luma_frames=target)

    assert audit["target_luma_affinity_strength"] == 0.5
    for before, target_frame, after in zip(rendered, target, output):
        expected = 0.5 * (_lightness(before) + _lightness(target_frame))
        assert float(np.mean(np.abs(_lightness(after) - expected))) < 1.0
