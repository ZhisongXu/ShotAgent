import numpy as np
from PIL import Image

from video_retouch.reference_style import build_reference_style_profile


def _frames(colors: list[tuple[int, int, int]]) -> tuple[Image.Image, ...]:
    return tuple(Image.new("RGB", (24, 16), color) for color in colors)


def test_reference_style_profile_is_deterministic_and_zone_explicit() -> None:
    target = _frames([(24, 52, 31), (40, 80, 48)])
    reference = _frames([(92, 70, 50), (180, 150, 112)])

    first = build_reference_style_profile(target, reference)
    second = build_reference_style_profile(target, reference)

    assert first == second
    assert first["schema"] == "reference-style-profile/v1"
    assert set(first["reference"]["tone_zones"]) == {
        "deep_shadows",
        "shadows",
        "midtones",
        "highlights",
        "speculars",
    }
    assert len(first["reference"]["dominant_palette"]) == 5
    deltas = np.asarray(first["reference_minus_target"]["lightness_quantile_delta"])
    assert deltas.shape == (7,)
    assert float(np.median(deltas)) > 0.0
