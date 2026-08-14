"""VL-first shot segmentation and task-aware Anchor selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Protocol, Sequence

import cv2
import numpy as np
from PIL import Image

from .clients import VisionLanguageClient
from .models import ShotPlan, StoryboardPlan
from .tasks import (
    anchor_selection_prompt,
    boundary_verification_prompt,
    hero_anchor_selection_prompt,
    long_video_overview_prompt,
    shot_window_prompt,
)


class ShotPlanner(Protocol):
    def plan(
        self,
        frames: Sequence[Image.Image],
        fps: float,
        instruction: str,
        anchors_per_shot: int = 1,
    ) -> StoryboardPlan: ...


@dataclass(frozen=True)
class LongVideoStoryboardSettings:
    """Sampling and adjudication policy for hierarchical long-video analysis."""

    overview_frames: int = 24
    window_seconds: float = 20.0
    overlap_seconds: float = 4.0
    max_window_images: int = 28
    technical_candidate_threshold: float = 0.14
    hard_cut_threshold: float = 0.42
    boundary_context_frames: int = 3
    boundary_batch_size: int = 4
    boundary_merge_tolerance_seconds: float = 0.20
    boundary_accept_confidence: float = 0.55
    anchor_candidates_per_shot: int = 10
    anchor_shots_per_call: int = 3
    max_anchor_images_per_call: int = 30
    hero_max_images_per_call: int = 32
    hero_nominees_per_batch: int = 3
    hero_candidate_count: int = 5
    minimum_shot_seconds: float = 0.40

    def __post_init__(self) -> None:
        positive_ints = {
            "overview_frames": self.overview_frames,
            "max_window_images": self.max_window_images,
            "boundary_context_frames": self.boundary_context_frames,
            "boundary_batch_size": self.boundary_batch_size,
            "anchor_candidates_per_shot": self.anchor_candidates_per_shot,
            "anchor_shots_per_call": self.anchor_shots_per_call,
            "max_anchor_images_per_call": self.max_anchor_images_per_call,
            "hero_max_images_per_call": self.hero_max_images_per_call,
            "hero_nominees_per_batch": self.hero_nominees_per_batch,
            "hero_candidate_count": self.hero_candidate_count,
        }
        for name, value in positive_ints.items():
            if int(value) < 1:
                raise ValueError(f"storyboard.long_video.{name} must be positive.")
        if self.window_seconds <= 0:
            raise ValueError("storyboard.long_video.window_seconds must be positive.")
        if self.overlap_seconds < 0 or self.overlap_seconds >= self.window_seconds:
            raise ValueError(
                "storyboard.long_video.overlap_seconds must be non-negative and "
                "smaller than window_seconds."
            )
        if not 0 <= self.technical_candidate_threshold <= 1:
            raise ValueError("technical_candidate_threshold must be in [0, 1].")
        if not 0 <= self.hard_cut_threshold <= 1:
            raise ValueError("hard_cut_threshold must be in [0, 1].")
        if self.boundary_merge_tolerance_seconds < 0:
            raise ValueError("boundary_merge_tolerance_seconds cannot be negative.")
        if not 0 <= self.boundary_accept_confidence <= 1:
            raise ValueError("boundary_accept_confidence must be in [0, 1].")
        if self.minimum_shot_seconds < 0:
            raise ValueError("minimum_shot_seconds cannot be negative.")
        if self.hero_nominees_per_batch >= self.hero_max_images_per_call:
            raise ValueError(
                "hero_nominees_per_batch must be smaller than "
                "hero_max_images_per_call."
            )

    @classmethod
    def from_dict(cls, payload: object) -> "LongVideoStoryboardSettings":
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("storyboard.long_video must be an object.")
        values = {
            field: payload[field]
            for field in cls.__dataclass_fields__
            if field in payload
        }
        return cls(**values)


def _frame_feature(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = array.shape[:2]
    scale = min(1.0, 160.0 / max(height, width))
    if scale < 1.0:
        array = cv2.resize(
            array,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).reshape(-1)
    rgb = array.astype(np.float32) / 255.0
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    sharpness = np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var()) / 10.0
    summary = np.array(
        [
            *rgb.mean(axis=(0, 1)).tolist(),
            float(luma.mean()),
            float(luma.std()),
            float(sharpness),
        ],
        dtype=np.float64,
    )
    return np.concatenate([hist.astype(np.float64), summary])


def _features_and_changes(
    frames: Sequence[Image.Image],
) -> tuple[np.ndarray, np.ndarray]:
    features = np.stack([_frame_feature(frame) for frame in frames], axis=0)
    histogram_size = 48
    changes = np.zeros(len(frames), dtype=np.float64)
    for index in range(1, len(frames)):
        histogram_distance = cv2.compareHist(
            features[index - 1, :histogram_size].astype(np.float32),
            features[index, :histogram_size].astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA,
        )
        summary_distance = np.mean(
            np.abs(
                features[index - 1, histogram_size:-1]
                - features[index, histogram_size:-1]
            )
        )
        changes[index] = 0.8 * histogram_distance + 0.2 * summary_distance
    return features, changes


def _local_change_peaks(changes: np.ndarray, threshold: float) -> list[int]:
    peaks: list[int] = []
    for index in range(1, len(changes)):
        previous = changes[index - 1]
        following = changes[index + 1] if index + 1 < len(changes) else -np.inf
        if changes[index] >= threshold and changes[index] >= previous:
            if changes[index] >= following:
                peaks.append(index)
    return peaks


def _uniform_indices(start: int, end: int, count: int) -> list[int]:
    if end < start:
        return []
    count = min(max(int(count), 1), end - start + 1)
    return np.unique(
        np.linspace(start, end, count).round().astype(np.int64)
    ).astype(int).tolist()


def _select_anchors(
    features: np.ndarray,
    start: int,
    end: int,
    count: int,
) -> tuple[int, ...]:
    indices = np.arange(start, end + 1, dtype=np.int64)
    count = min(max(int(count), 1), indices.size)
    shot_features = features[indices]
    center = np.median(shot_features, axis=0)
    distance_to_center = np.linalg.norm(shot_features - center[None, :], axis=1)
    edge_penalty = np.abs(np.linspace(-1.0, 1.0, indices.size, dtype=np.float64))
    sharpness = shot_features[:, -1]
    quality = -distance_to_center - 0.05 * edge_penalty + 0.03 * sharpness
    selected = [int(indices[int(np.argmax(quality))])]
    while len(selected) < count:
        selected_features = features[np.asarray(selected)]
        coverage = np.min(
            np.linalg.norm(
                shot_features[:, None, :] - selected_features[None, :, :], axis=2
            ),
            axis=1,
        )
        coverage[np.isin(indices, selected)] = -np.inf
        selected.append(int(indices[int(np.argmax(coverage))]))
    return tuple(sorted(selected))


def _rank_hero_candidates(
    features: np.ndarray,
    candidate_frames: Sequence[int],
) -> tuple[int, ...]:
    candidates = np.asarray(list(dict.fromkeys(candidate_frames)), dtype=np.int64)
    if candidates.size == 0:
        raise ValueError("HeroAnchor selection requires at least one Anchor.")
    values = features[candidates]
    center = np.median(values, axis=0)
    representativeness = np.linalg.norm(values - center[None, :], axis=1)
    luminance = values[:, 51]
    tonal_range = values[:, 52]
    sharpness = values[:, 53]
    extreme_exposure = np.abs(luminance - 0.5)
    quality = (
        -representativeness
        - 0.25 * extreme_exposure
        + 0.08 * tonal_range
        + 0.04 * sharpness
    )
    order = np.argsort(-quality, kind="stable")
    return tuple(int(candidates[index]) for index in order)


class HeuristicShotPlanner:
    """Deterministic fallback based on appearance discontinuities and coverage."""

    def __init__(
        self,
        cut_threshold: float = 0.42,
        minimum_shot_seconds: float = 0.4,
    ) -> None:
        self.cut_threshold = float(cut_threshold)
        self.minimum_shot_seconds = float(minimum_shot_seconds)

    def plan(
        self,
        frames: Sequence[Image.Image],
        fps: float,
        instruction: str,
        anchors_per_shot: int = 1,
    ) -> StoryboardPlan:
        if not frames:
            raise ValueError("Video contains no frames.")
        if fps <= 0:
            raise ValueError("fps must be positive.")
        features, changes = _features_and_changes(frames)

        minimum_frames = max(1, round(self.minimum_shot_seconds * fps))
        boundaries = [0]
        for index in range(1, len(frames)):
            if (
                changes[index] >= self.cut_threshold
                and index - boundaries[-1] >= minimum_frames
            ):
                boundaries.append(index)
        boundaries.append(len(frames))
        # A fade-out or damaged final frame can cross the cut threshold but is
        # not a usable standalone shot. Merge a trailing fragment back into
        # the preceding shot so it cannot receive an unrelated grade.
        if len(boundaries) > 2 and boundaries[-1] - boundaries[-2] < minimum_frames:
            boundaries.pop(-2)

        shots = []
        for shot_id, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            anchors = _select_anchors(features, start, stop - 1, anchors_per_shot)
            anchor_candidates = _select_anchors(
                features,
                start,
                stop - 1,
                max(anchors_per_shot, min(5, stop - start)),
            )
            shots.append(
                ShotPlan(
                    shot_id=shot_id,
                    start_frame=start,
                    end_frame=stop - 1,
                    anchor_frames=anchors,
                    anchor_candidates=anchor_candidates,
                    description="appearance-continuous shot",
                    selection_reason="feature medoid plus appearance coverage",
                )
            )
        hero_ranking = _rank_hero_candidates(
            features,
            [frame for shot in shots for frame in shot.anchor_frames],
        )
        return StoryboardPlan(
            frame_count=len(frames),
            fps=float(fps),
            shots=tuple(shots),
            planner="heuristic",
            hero_anchor_frame=hero_ranking[0],
            hero_anchor_candidates=hero_ranking,
            hero_selection_reason=(
                "global feature medoid with exposure, tonal-range, and sharpness safety"
            ),
            diagnosis={
                "instruction": instruction,
                "cut_threshold": self.cut_threshold,
                "cut_scores": changes.tolist(),
            },
        )


class VLShotPlanner:
    """Hierarchical long-video shot segmentation and task-aware Anchor ranking."""

    def __init__(
        self,
        client: VisionLanguageClient,
        fallback: Optional[ShotPlanner] = None,
        settings: Optional[LongVideoStoryboardSettings] = None,
        max_storyboard_frames: Optional[int] = None,
        strict: bool = False,
    ) -> None:
        self.client = client
        self.settings = settings or LongVideoStoryboardSettings()
        # Retain the old constructor argument as an overview-budget alias so
        # downstream callers do not break while still using the new pipeline.
        if max_storyboard_frames is not None:
            self.settings = replace(
                self.settings, overview_frames=int(max_storyboard_frames)
            )
        self.fallback = fallback or HeuristicShotPlanner(
            cut_threshold=self.settings.hard_cut_threshold,
            minimum_shot_seconds=self.settings.minimum_shot_seconds,
        )
        self.strict = bool(strict)

    def _fallback_plan(
        self,
        frames: Sequence[Image.Image],
        fps: float,
        instruction: str,
        anchors_per_shot: int,
        error: Exception,
    ) -> StoryboardPlan:
        if self.strict:
            raise error
        plan = self.fallback.plan(frames, fps, instruction, anchors_per_shot)
        diagnosis = dict(plan.diagnosis)
        diagnosis.update(
            {
                "requested_planner": f"hierarchical-vision:{self.client.model_id}",
                "fallback_reason": f"{type(error).__name__}: {error}",
            }
        )
        return StoryboardPlan(
            frame_count=plan.frame_count,
            fps=plan.fps,
            shots=plan.shots,
            planner="heuristic_fallback",
            hero_anchor_frame=plan.hero_anchor_frame,
            hero_anchor_candidates=plan.hero_anchor_candidates,
            hero_selection_reason=plan.hero_selection_reason,
            diagnosis=diagnosis,
        )

    def _window_ranges(self, frame_count: int, fps: float) -> list[tuple[int, int]]:
        window_frames = max(2, round(self.settings.window_seconds * fps))
        overlap_frames = min(
            window_frames - 1, round(self.settings.overlap_seconds * fps)
        )
        if frame_count <= window_frames:
            return [(0, frame_count - 1)]
        step = window_frames - overlap_frames
        ranges: list[tuple[int, int]] = []
        start = 0
        while start < frame_count:
            end = min(frame_count - 1, start + window_frames - 1)
            ranges.append((start, end))
            if end == frame_count - 1:
                break
            start += step
        return ranges

    def _window_samples(
        self,
        start: int,
        end: int,
        technical_candidates: Sequence[int],
        changes: np.ndarray,
    ) -> list[int]:
        budget = min(self.settings.max_window_images, end - start + 1)
        uniform_budget = min(budget, max(4, budget // 2))
        selected = set(_uniform_indices(start, end, uniform_budget))
        ranked = sorted(
            (index for index in technical_candidates if start < index <= end),
            key=lambda index: float(changes[index]),
            reverse=True,
        )
        for candidate in ranked:
            for offset in (-1, 0, 1):
                frame = min(max(candidate + offset, start), end)
                if len(selected) < budget:
                    selected.add(frame)
        if len(selected) < budget:
            for frame in _uniform_indices(start, end, budget):
                selected.add(frame)
                if len(selected) >= budget:
                    break
        return sorted(selected)

    @staticmethod
    def _confidence(value: object, default: float = 0.0) -> float:
        try:
            return min(max(float(value), 0.0), 1.0)
        except (TypeError, ValueError):
            return default

    def _global_overview(
        self,
        frames: Sequence[Image.Image],
        fps: float,
        instruction: str,
    ) -> tuple[dict[str, object], list[int]]:
        indices = _uniform_indices(
            0, len(frames) - 1, self.settings.overview_frames
        )
        labeled = [
            (
                f"scope=global_overview; frame_id={index}; "
                f"time={index / fps:.3f}s",
                frames[index],
            )
            for index in indices
        ]
        overview = self.client.generate_json(
            labeled,
            long_video_overview_prompt(instruction, len(frames), fps),
        )
        if not isinstance(overview.get("summary"), str):
            raise ValueError("Long-video overview response is missing summary.")
        return overview, indices

    def _collect_window_boundaries(
        self,
        frames: Sequence[Image.Image],
        fps: float,
        instruction: str,
        overview: dict[str, object],
        technical_candidates: Sequence[int],
        changes: np.ndarray,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        records: list[dict[str, object]] = []
        window_diagnostics: list[dict[str, object]] = []
        for window_id, (start, end) in enumerate(
            self._window_ranges(len(frames), fps)
        ):
            sample_indices = self._window_samples(
                start, end, technical_candidates, changes
            )
            labeled = [
                (
                    f"window_id={window_id}; frame_id={index}; "
                    f"time={index / fps:.3f}s",
                    frames[index],
                )
                for index in sample_indices
            ]
            candidates = [
                {"frame": index, "score": round(float(changes[index]), 6)}
                for index in technical_candidates
                if start < index <= end
            ]
            candidates = sorted(
                candidates, key=lambda item: float(item["score"]), reverse=True
            )[:64]
            payload = self.client.generate_json(
                labeled,
                shot_window_prompt(
                    instruction,
                    len(frames),
                    fps,
                    start,
                    end,
                    overview,
                    candidates,
                ),
            )
            raw_boundaries = payload.get("boundaries")
            if not isinstance(raw_boundaries, list):
                raise ValueError(
                    f"Shot-window {window_id} response is missing boundaries."
                )
            accepted_in_window = 0
            for item in raw_boundaries:
                if not isinstance(item, dict):
                    continue
                try:
                    frame = int(item.get("boundary_frame"))
                except (TypeError, ValueError):
                    continue
                if not start < frame <= end or not 0 < frame < len(frames):
                    continue
                records.append(
                    {
                        "frame": frame,
                        "confidence": self._confidence(item.get("confidence"), 0.5),
                        "transition_type": str(
                            item.get("transition_type", "unknown")
                        ),
                        "evidence": str(item.get("evidence", "")),
                        "source": f"vl-window-{window_id}",
                    }
                )
                accepted_in_window += 1
            window_diagnostics.append(
                {
                    "window_id": window_id,
                    "start_frame": start,
                    "end_frame": end,
                    "sampled_frames": sample_indices,
                    "reported_boundaries": accepted_in_window,
                }
            )
        return records, window_diagnostics

    def _cluster_boundary_records(
        self,
        records: Sequence[dict[str, object]],
        fps: float,
    ) -> list[dict[str, object]]:
        tolerance = max(
            1, round(self.settings.boundary_merge_tolerance_seconds * fps)
        )
        clusters: list[list[dict[str, object]]] = []
        for record in sorted(records, key=lambda item: int(item["frame"])):
            if not clusters or int(record["frame"]) - int(
                clusters[-1][-1]["frame"]
            ) > tolerance:
                clusters.append([record])
            else:
                clusters[-1].append(record)
        merged: list[dict[str, object]] = []
        for candidate_id, cluster in enumerate(clusters):
            strongest = max(
                cluster, key=lambda item: float(item.get("confidence", 0.0))
            )
            merged.append(
                {
                    "candidate_id": candidate_id,
                    "approximate_frame": int(strongest["frame"]),
                    "sources": sorted({str(item["source"]) for item in cluster}),
                    "proposals": [
                        {
                            "frame": int(item["frame"]),
                            "confidence": float(item.get("confidence", 0.0)),
                            "transition_type": str(
                                item.get("transition_type", "unknown")
                            ),
                            "evidence": str(item.get("evidence", "")),
                        }
                        for item in cluster
                    ],
                }
            )
        return merged

    def _verify_boundaries(
        self,
        frames: Sequence[Image.Image],
        fps: float,
        candidates: Sequence[dict[str, object]],
    ) -> tuple[list[int], list[dict[str, object]], int]:
        verified: list[dict[str, object]] = []
        call_count = 0
        for offset in range(0, len(candidates), self.settings.boundary_batch_size):
            batch = list(
                candidates[offset : offset + self.settings.boundary_batch_size]
            )
            labeled: list[tuple[str, Image.Image]] = []
            for candidate in batch:
                candidate_id = int(candidate["candidate_id"])
                center = int(candidate["approximate_frame"])
                start = max(0, center - self.settings.boundary_context_frames)
                end = min(
                    len(frames) - 1,
                    center + self.settings.boundary_context_frames,
                )
                for index in range(start, end + 1):
                    labeled.append(
                        (
                            f"candidate_id={candidate_id}; "
                            f"approximate_boundary={center}; frame_id={index}; "
                            f"time={index / fps:.3f}s",
                            frames[index],
                        )
                    )
            payload = self.client.generate_json(
                labeled,
                boundary_verification_prompt(len(frames), fps, batch),
            )
            call_count += 1
            raw_decisions = payload.get("decisions")
            if not isinstance(raw_decisions, list):
                raise ValueError("Boundary verification response is missing decisions.")
            by_id = {
                int(item["candidate_id"]): item
                for item in raw_decisions
                if isinstance(item, dict) and "candidate_id" in item
            }
            expected = {int(item["candidate_id"]) for item in batch}
            if self.strict and set(by_id) != expected:
                raise ValueError(
                    "Boundary verification did not decide every candidate."
                )
            for candidate in batch:
                candidate_id = int(candidate["candidate_id"])
                decision = by_id.get(candidate_id)
                if not isinstance(decision, dict) or not bool(
                    decision.get("accept", False)
                ):
                    continue
                confidence = self._confidence(decision.get("confidence"), 0.0)
                if confidence < self.settings.boundary_accept_confidence:
                    continue
                try:
                    boundary = int(decision.get("boundary_frame"))
                except (TypeError, ValueError):
                    continue
                center = int(candidate["approximate_frame"])
                maximum_refinement = max(
                    1, self.settings.boundary_context_frames * 2
                )
                if (
                    not 0 < boundary < len(frames)
                    or abs(boundary - center) > maximum_refinement
                ):
                    continue
                verified.append(
                    {
                        "candidate_id": candidate_id,
                        "boundary_frame": boundary,
                        "confidence": confidence,
                        "transition_type": str(
                            decision.get("transition_type", "unknown")
                        ),
                        "reason": str(decision.get("reason", "")),
                    }
                )
        minimum_frames = max(1, round(self.settings.minimum_shot_seconds * fps))
        boundaries: list[int] = []
        for record in sorted(verified, key=lambda item: int(item["boundary_frame"])):
            boundary = int(record["boundary_frame"])
            if boundary - (boundaries[-1] if boundaries else 0) < minimum_frames:
                continue
            if len(frames) - boundary < minimum_frames:
                continue
            if boundaries and boundary == boundaries[-1]:
                continue
            boundaries.append(boundary)
        return boundaries, verified, call_count

    def _anchor_candidate_frames(
        self,
        features: np.ndarray,
        start: int,
        end: int,
    ) -> tuple[int, ...]:
        length = end - start + 1
        margin = min(
            self.settings.boundary_context_frames,
            max(0, (length - 1) // 4),
        )
        candidate_start = start + margin
        candidate_end = end - margin
        count = min(
            self.settings.anchor_candidates_per_shot,
            candidate_end - candidate_start + 1,
        )
        return _select_anchors(
            features, candidate_start, candidate_end, max(count, 1)
        )

    def _anchor_batches(
        self,
        contracts: Sequence[dict[str, object]],
    ) -> list[list[dict[str, object]]]:
        batches: list[list[dict[str, object]]] = []
        current: list[dict[str, object]] = []
        image_count = 0
        for contract in contracts:
            candidates = contract["candidate_frames"]
            assert isinstance(candidates, list)
            would_exceed = (
                current
                and (
                    len(current) >= self.settings.anchor_shots_per_call
                    or image_count + len(candidates)
                    > self.settings.max_anchor_images_per_call
                )
            )
            if would_exceed:
                batches.append(current)
                current = []
                image_count = 0
            current.append(contract)
            image_count += len(candidates)
        if current:
            batches.append(current)
        return batches

    def _select_vl_anchors(
        self,
        frames: Sequence[Image.Image],
        fps: float,
        instruction: str,
        overview: dict[str, object],
        boundaries: Sequence[int],
        features: np.ndarray,
        anchors_per_shot: int,
    ) -> tuple[list[ShotPlan], int]:
        starts = [0, *boundaries]
        stops = [*boundaries, len(frames)]
        contracts: list[dict[str, object]] = []
        for shot_id, (start, stop) in enumerate(zip(starts, stops)):
            candidates = self._anchor_candidate_frames(
                features, start, stop - 1
            )
            contracts.append(
                {
                    "shot_id": shot_id,
                    "start_frame": start,
                    "end_frame": stop - 1,
                    "candidate_frames": list(candidates),
                    "requested_anchors": min(
                        anchors_per_shot, stop - start, len(candidates)
                    ),
                }
            )
        selected: dict[int, dict[str, object]] = {}
        call_count = 0
        for batch in self._anchor_batches(contracts):
            labeled: list[tuple[str, Image.Image]] = []
            for contract in batch:
                shot_id = int(contract["shot_id"])
                for index in contract["candidate_frames"]:
                    frame = int(index)
                    labeled.append(
                        (
                            f"shot_id={shot_id}; frame_id={frame}; "
                            f"time={frame / fps:.3f}s; role=anchor_candidate",
                            frames[frame],
                        )
                    )
            payload = self.client.generate_json(
                labeled,
                anchor_selection_prompt(
                    instruction, fps, anchors_per_shot, batch, overview
                ),
            )
            call_count += 1
            raw_shots = payload.get("shots")
            if not isinstance(raw_shots, list):
                raise ValueError("Anchor-selection response is missing shots.")
            for item in raw_shots:
                if isinstance(item, dict) and "shot_id" in item:
                    selected[int(item["shot_id"])] = item

        shots: list[ShotPlan] = []
        for contract in contracts:
            shot_id = int(contract["shot_id"])
            start = int(contract["start_frame"])
            end = int(contract["end_frame"])
            candidate_frames = {int(value) for value in contract["candidate_frames"]}
            target_count = int(contract["requested_anchors"])
            result = selected.get(shot_id)
            if result is None:
                raise ValueError(f"Anchor role omitted shot {shot_id}.")
            raw_anchors = result.get("anchors")
            if not isinstance(raw_anchors, list):
                raise ValueError(f"Anchor role returned invalid shot {shot_id} anchors.")
            ranked: list[tuple[int, int, str]] = []
            for rank_index, item in enumerate(raw_anchors):
                if isinstance(item, dict):
                    raw_frame = item.get("frame")
                    rank = int(item.get("rank", rank_index + 1))
                    reason = str(item.get("reason", ""))
                else:
                    raw_frame = item
                    rank = rank_index + 1
                    reason = ""
                try:
                    frame = int(raw_frame)
                except (TypeError, ValueError):
                    continue
                if frame in candidate_frames:
                    ranked.append((rank, frame, reason))
            ranked.sort(key=lambda item: item[0])
            unique_frames: list[int] = []
            reasons: list[str] = []
            for _, frame, reason in ranked:
                if frame not in unique_frames:
                    unique_frames.append(frame)
                    if reason:
                        reasons.append(reason)
                if len(unique_frames) >= target_count:
                    break
            if len(unique_frames) < target_count:
                if self.strict:
                    raise ValueError(
                        f"Anchor role returned too few valid frames for shot {shot_id}."
                    )
                for frame in contract["candidate_frames"]:
                    value = int(frame)
                    if value not in unique_frames:
                        unique_frames.append(value)
                    if len(unique_frames) >= target_count:
                        break
            shots.append(
                ShotPlan(
                    shot_id=shot_id,
                    start_frame=start,
                    end_frame=end,
                    anchor_frames=tuple(sorted(unique_frames)),
                    anchor_candidates=tuple(sorted(candidate_frames)),
                    description=str(result.get("description", "")),
                    selection_reason="; ".join(reasons)
                    or "VL-ranked representative grading coverage",
                )
            )
        return shots, call_count

    def _hero_model_call(
        self,
        frames: Sequence[Image.Image],
        instruction: str,
        overview: dict[str, object],
        candidates: Sequence[dict[str, object]],
        requested: int,
        final_round: bool,
    ) -> tuple[list[dict[str, object]], int]:
        allowed = {int(item["frame"]): int(item["shot_id"]) for item in candidates}
        labeled = [
            (
                f"shot_id={int(item['shot_id'])}; frame_id={int(item['frame'])}; "
                "role=hero_candidate",
                frames[int(item["frame"])],
            )
            for item in candidates
        ]
        payload = self.client.generate_json(
            labeled,
            hero_anchor_selection_prompt(
                instruction,
                list(candidates),
                overview,
                requested,
                final_round,
            ),
        )
        raw_ranking = payload.get("ranked_candidates")
        if not isinstance(raw_ranking, list):
            raise ValueError("HeroAnchor response is missing ranked_candidates.")
        ranking: list[dict[str, object]] = []
        seen: set[int] = set()
        for item in raw_ranking:
            if not isinstance(item, dict):
                continue
            try:
                frame = int(item.get("frame"))
            except (TypeError, ValueError):
                continue
            if frame not in allowed or frame in seen:
                continue
            ranking.append(
                {
                    "frame": frame,
                    "shot_id": allowed[frame],
                    "confidence": self._confidence(item.get("confidence"), 0.0),
                    "reason": str(item.get("reason", "")),
                }
            )
            seen.add(frame)
            if len(ranking) >= requested:
                break
        if not ranking:
            raise ValueError("HeroAnchor role returned no valid candidate.")
        return ranking, 1

    def _select_hero_anchor(
        self,
        frames: Sequence[Image.Image],
        instruction: str,
        overview: dict[str, object],
        shots: Sequence[ShotPlan],
    ) -> tuple[tuple[int, ...], str, int]:
        current = [
            {"frame": frame, "shot_id": shot.shot_id}
            for shot in shots
            for frame in shot.anchor_frames
        ]
        if not current:
            raise ValueError("No per-shot Anchors are available for HeroAnchor ranking.")
        call_count = 0
        while len(current) > self.settings.hero_max_images_per_call:
            nominees: list[dict[str, object]] = []
            for offset in range(
                0, len(current), self.settings.hero_max_images_per_call
            ):
                batch = current[
                    offset : offset + self.settings.hero_max_images_per_call
                ]
                ranking, calls = self._hero_model_call(
                    frames,
                    instruction,
                    overview,
                    batch,
                    min(self.settings.hero_nominees_per_batch, len(batch)),
                    False,
                )
                nominees.extend(ranking)
                call_count += calls
            current = nominees
        ranking, calls = self._hero_model_call(
            frames,
            instruction,
            overview,
            current,
            min(self.settings.hero_candidate_count, len(current)),
            True,
        )
        call_count += calls
        frames_by_rank = tuple(int(item["frame"]) for item in ranking)
        reason = str(ranking[0].get("reason", "")) or (
            "VL-ranked master reference suitability"
        )
        return frames_by_rank, reason, call_count

    def plan(
        self,
        frames: Sequence[Image.Image],
        fps: float,
        instruction: str,
        anchors_per_shot: int = 1,
    ) -> StoryboardPlan:
        if not frames:
            raise ValueError("Video contains no frames.")
        if fps <= 0:
            raise ValueError("fps must be positive.")
        if anchors_per_shot < 1:
            raise ValueError("anchors_per_shot must be positive.")
        try:
            features, changes = _features_and_changes(frames)
            technical_candidates = _local_change_peaks(
                changes, self.settings.technical_candidate_threshold
            )
            overview, overview_indices = self._global_overview(
                frames, fps, instruction
            )
            boundary_records, window_diagnostics = self._collect_window_boundaries(
                frames,
                fps,
                instruction,
                overview,
                technical_candidates,
                changes,
            )
            for frame in _local_change_peaks(
                changes, self.settings.hard_cut_threshold
            ):
                boundary_records.append(
                    {
                        "frame": frame,
                        "confidence": min(
                            0.95,
                            0.55
                            + float(changes[frame] - self.settings.hard_cut_threshold),
                        ),
                        "transition_type": "physical_discontinuity",
                        "evidence": f"full-scan change score={changes[frame]:.6f}",
                        "source": "full-video-physical-scan",
                    }
                )
            candidates = self._cluster_boundary_records(boundary_records, fps)
            boundaries, verified, boundary_calls = self._verify_boundaries(
                frames, fps, candidates
            )
            shots, anchor_calls = self._select_vl_anchors(
                frames,
                fps,
                instruction,
                overview,
                boundaries,
                features,
                anchors_per_shot,
            )
            hero_ranking, hero_reason, hero_calls = self._select_hero_anchor(
                frames, instruction, overview, shots
            )
            return StoryboardPlan(
                frame_count=len(frames),
                fps=float(fps),
                shots=tuple(shots),
                planner="hierarchical-vision-storyboard/v2",
                hero_anchor_frame=hero_ranking[0],
                hero_anchor_candidates=hero_ranking,
                hero_selection_reason=hero_reason,
                diagnosis={
                    "model_id": self.client.model_id,
                    "instruction": instruction,
                    "duration_seconds": len(frames) / fps,
                    "global_overview": overview,
                    "overview_sampled_frames": overview_indices,
                    "full_scan_frame_count": len(frames),
                    "technical_candidate_frames": technical_candidates,
                    "window_count": len(window_diagnostics),
                    "windows": window_diagnostics,
                    "boundary_candidate_count": len(candidates),
                    "boundary_verification_calls": boundary_calls,
                    "verified_boundaries": verified,
                    "final_boundaries": boundaries,
                    "anchor_selection_calls": anchor_calls,
                    "hero_selection_calls": hero_calls,
                },
            )
        except Exception as error:
            return self._fallback_plan(
                frames, fps, instruction, anchors_per_shot, error
            )
