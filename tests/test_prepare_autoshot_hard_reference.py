from pathlib import Path

from evaluation.prepare_autoshot_hard_reference import (
    Candidate,
    _densest_window,
    _hardness,
    _percentile_ranks,
)


def _candidate(name: str, **values: float) -> Candidate:
    defaults = {
        "shot_density": 1.0,
        "luminance_swing": 1.0,
        "chroma_swing": 1.0,
        "visual_change": 1.0,
    }
    defaults.update(values)
    return Candidate(
        path=Path(name),
        frame_count=300,
        boundaries=(10, 20, 30),
        fps=25.0,
        window_start=0,
        window_cuts=3,
        **defaults,
    )


def test_densest_window_finds_cluster_ending_at_boundary() -> None:
    start, cuts = _densest_window((10, 100, 110, 120), 21)
    assert start == 100
    assert cuts == 3


def test_percentile_ranks_are_deterministic_for_ties() -> None:
    assert _percentile_ranks([2.0, 1.0, 2.0]) == [0.5, 0.0, 1.0]


def test_hardness_respects_frozen_weights() -> None:
    easy = _candidate("easy.mp4")
    hard = _candidate(
        "hard.mp4",
        shot_density=2.0,
        luminance_swing=2.0,
        chroma_swing=2.0,
        visual_change=2.0,
    )
    assert _hardness([easy, hard]) == [0.0, 1.0]
