"""Closed-loop video-to-grade-parameter pipeline."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from retouch_agent import RetouchExecutor, RetouchParameters
from retouch_agent.planner import image_statistics

from .backends import (
    AnchorGrade,
    AnchorRetouchBackend,
    HeroAnchorReference,
    NativeRetouchBackend,
)
from .critic import PhotoAgentStyleCritic, ShotCritic
from .models import GradeGraph, HeroAnchorRecord, ShotGrade, ShotPlan, StoryboardPlan
from .propagation import BayesianParameterDiffuser
from .search import AestheticMCTSSearch
from .shot_planner import HeuristicShotPlanner, ShotPlanner


class DynamicGradePipeline:
    """Storyboard, grade Anchors, diffuse parameters, critique, and rollback."""

    def __init__(
        self,
        shot_planner: Optional[ShotPlanner] = None,
        anchor_backend: Optional[AnchorRetouchBackend] = None,
        anchor_backends: Optional[Sequence[AnchorRetouchBackend]] = None,
        diffuser: Optional[BayesianParameterDiffuser] = None,
        critic: Optional[ShotCritic] = None,
        executor: Optional[RetouchExecutor] = None,
        anchors_per_shot: int = 1,
        maximum_anchors_per_shot: int = 3,
        maximum_attempts: int = 3,
        maximum_hero_attempts: int = 3,
        mcts_exploration: float = np.sqrt(2.0),
        mcts_rejection_penalty: float = 0.25,
        mcts_seed: int = 7,
    ) -> None:
        if anchors_per_shot < 1 or maximum_anchors_per_shot < anchors_per_shot:
            raise ValueError("Invalid Anchor budget.")
        if maximum_attempts < 1:
            raise ValueError("maximum_attempts must be positive.")
        if maximum_hero_attempts < 1:
            raise ValueError("maximum_hero_attempts must be positive.")
        self.shot_planner = shot_planner or HeuristicShotPlanner()
        if anchor_backend is not None and anchor_backends is not None:
            raise ValueError("Use anchor_backend or anchor_backends, not both.")
        selected_backends = (
            tuple(anchor_backends)
            if anchor_backends is not None
            else (anchor_backend or NativeRetouchBackend(),)
        )
        if not selected_backends:
            raise ValueError("At least one Anchor editing Agent is required.")
        self.anchor_backends = selected_backends
        # Kept for callers that inspect the legacy single-backend attribute.
        self.anchor_backend = selected_backends[0]
        self.diffuser = diffuser or BayesianParameterDiffuser()
        self.critic = critic or PhotoAgentStyleCritic(use_vl_review=False)
        self.executor = executor or RetouchExecutor()
        self.anchors_per_shot = int(anchors_per_shot)
        self.maximum_anchors_per_shot = int(maximum_anchors_per_shot)
        self.maximum_attempts = int(maximum_attempts)
        self.maximum_hero_attempts = int(maximum_hero_attempts)
        self.search = AestheticMCTSSearch(
            backends=self.anchor_backends,
            critic=self.critic,
            diffuser=self.diffuser,
            executor=self.executor,
            maximum_evaluations=self.maximum_attempts,
            maximum_anchors=self.maximum_anchors_per_shot,
            exploration_constant=mcts_exploration,
            rejection_penalty=mcts_rejection_penalty,
            seed=mcts_seed,
        )

    def _run_shot(
        self,
        frames: Sequence[Image.Image],
        instruction: str,
        shot: ShotPlan,
        hero_reference: HeroAnchorReference,
        candidate_grid: Optional[dict[int, tuple[AnchorGrade, ...]]] = None,
    ) -> ShotGrade:
        outcome = self.search.search(
            frames,
            instruction,
            shot,
            hero_reference=hero_reference,
            candidate_grid=candidate_grid,
        )
        shot_length = shot.end_frame - shot.start_frame + 1
        if outcome.selected is None:
            zeros = np.zeros((shot_length, 12), dtype=np.float64)
            reason = (
                ",".join(outcome.attempts[-1].reasons)
                if outcome.attempts and outcome.attempts[-1].reasons
                else "critic_rejected"
            )
            attempted_anchors = (
                outcome.attempts[-1].anchor_frames
                if outcome.attempts
                else shot.anchor_frames
            )
            attempted_shot = replace(
                shot,
                anchor_frames=attempted_anchors,
                anchor_candidates=tuple(
                    sorted(set(shot.anchor_candidates) | set(attempted_anchors))
                ),
            )
            return ShotGrade(
                shot=attempted_shot,
                base_parameters=np.zeros(12, dtype=np.float64),
                parameter_keyframes={},
                frame_parameters=zeros,
                confidence=0.0,
                accepted=False,
                rolled_back=True,
                rollback_reason=reason,
                attempts=outcome.attempts,
                search_memory=outcome.memory,
            )
        selected = outcome.selected
        committed_shot = replace(
            shot,
            anchor_frames=selected.anchor_indices,
            anchor_candidates=tuple(
                sorted(set(shot.anchor_candidates) | set(selected.anchor_indices))
            ),
        )
        return ShotGrade(
            shot=committed_shot,
            base_parameters=selected.diffused.base_parameters,
            parameter_keyframes=selected.diffused.keyframes,
            frame_parameters=selected.diffused.frame_parameters,
            confidence=float(selected.critique.score),
            accepted=bool(selected.critique.accepted),
            rolled_back=not bool(selected.critique.accepted),
            rollback_reason=(
                None
                if selected.critique.accepted
                else ",".join(selected.critique.reasons) or "critic_rejected_kept_grade"
            ),
            attempts=outcome.attempts,
            search_memory=outcome.memory,
        )

    @staticmethod
    def _shot_for_frame(storyboard: StoryboardPlan, frame_index: int) -> ShotPlan:
        return next(
            shot
            for shot in storyboard.shots
            if shot.start_frame <= frame_index <= shot.end_frame
        )

    @staticmethod
    def _identity_shot_grade(shot: ShotPlan, reason: str) -> ShotGrade:
        length = shot.end_frame - shot.start_frame + 1
        return ShotGrade(
            shot=shot,
            base_parameters=np.zeros(12, dtype=np.float64),
            parameter_keyframes={},
            frame_parameters=np.zeros((length, 12), dtype=np.float64),
            confidence=0.0,
            accepted=False,
            rolled_back=True,
            rollback_reason=reason,
            attempts=(),
            search_memory={
                "rolled_back": True,
                "rollback_scope": "global_hero_anchor",
                "reason": reason,
            },
        )

    @staticmethod
    def _shot_content_signature(
        frames: Sequence[Image.Image], shot: ShotPlan
    ) -> np.ndarray:
        anchor = int(shot.anchor_frames[0])
        stats = image_statistics(frames[anchor])
        return np.asarray(
            [
                stats["luminance"],
                stats["contrast"],
                stats["saturation"],
                stats["warmth"],
            ],
            dtype=np.float64,
        )

    @classmethod
    def _content_groups(
        cls, frames: Sequence[Image.Image], shots: Sequence[ShotPlan]
    ) -> list[tuple[str, tuple[ShotPlan, ...]]]:
        buckets: dict[str, list[ShotPlan]] = {}
        for shot in shots:
            luminance, contrast, saturation, warmth = cls._shot_content_signature(
                frames, shot
            )
            if luminance < 0.22:
                label = "dark_low_key"
            elif luminance > 0.58 and saturation < 0.20:
                label = "bright_snow_fog"
            elif warmth > 0.06:
                label = "warm_fire_skin"
            elif saturation > 0.26:
                label = "colorful_daylight"
            else:
                label = "neutral_cool"
            buckets.setdefault(label, []).append(shot)
        return [
            (label, tuple(group))
            for label, group in buckets.items()
            if group
        ]

    @classmethod
    def _rank_group_hero_candidates(
        cls, frames: Sequence[Image.Image], shots: Sequence[ShotPlan]
    ) -> list[int]:
        candidates = [
            int(frame)
            for shot in shots
            for frame in shot.anchor_frames
        ]
        if not candidates:
            return []
        signatures = np.stack(
            [
                cls._shot_content_signature(
                    frames,
                    replace(shot, anchor_frames=(int(frame),)),
                )
                for shot in shots
                for frame in shot.anchor_frames
            ],
            axis=0,
        )
        center = np.median(signatures, axis=0)
        distance = np.linalg.norm(signatures - center[None, :], axis=1)
        quality = -distance - 0.35 * np.abs(signatures[:, 0] - 0.45)
        order = np.argsort(-quality, kind="stable")
        return [candidates[int(index)] for index in order]

    @staticmethod
    def _reference_video_sheet(
        frames: Sequence[Image.Image], count: int = 8, cell_side: int = 192
    ) -> Image.Image:
        """Represent the full reference video as an ordered visual storyboard."""

        normalized = tuple(frame.convert("RGB") for frame in frames)
        if not normalized:
            raise ValueError("Reference video must contain at least one frame.")
        indices = np.unique(
            np.linspace(0, len(normalized) - 1, min(count, len(normalized)))
            .round()
            .astype(int)
        )
        sheet = Image.new("RGB", (cell_side * len(indices), cell_side), "black")
        for column, index in enumerate(indices):
            image = normalized[int(index)].copy()
            image.thumbnail((cell_side, cell_side), Image.Resampling.LANCZOS)
            x = column * cell_side + (cell_side - image.width) // 2
            y = (cell_side - image.height) // 2
            sheet.paste(image, (x, y))
        return sheet

    def _run_with_external_reference(
        self,
        frames: Sequence[Image.Image],
        instruction: str,
        storyboard: StoryboardPlan,
        reference_frames: Sequence[Image.Image],
    ) -> GradeGraph:
        """Run the configured editor pool using a user-supplied reference video."""

        reference_sheet = self._reference_video_sheet(reference_frames)
        reference_instruction = (
            f"{instruction}\n"
            "Reference-video requirement: make a clearly visible, strong match "
            "to the reference video's overall palette, white balance, contrast, "
            "black level, highlight roll-off, and saturation. Preserve the target "
            "video's people, objects, geometry, texture, and temporal continuity. "
            "Do not return a timid near-identity grade."
        )
        reference_grade = AnchorGrade(
            frame_index=-1,
            parameters=RetouchParameters(),
            preview=reference_sheet,
            valid=True,
            score=1.0,
            backend="external-reference-video",
            metadata={
                "reference_type": "video_storyboard",
                "sampled_frames": min(8, len(reference_frames)),
                "source_frame_count": len(reference_frames),
            },
        )
        reference = HeroAnchorReference(
            frame_index=-1,
            shot_id=-1,
            source=reference_sheet,
            grade=reference_grade,
        )
        candidate_grid = self.search.prepopulate_storyboard_candidate_grid(
            frames,
            reference_instruction,
            storyboard.shots,
            reference,
        )
        shot_grades = tuple(
            self._run_shot(
                frames,
                reference_instruction,
                shot,
                reference,
                candidate_grid,
            )
            for shot in storyboard.shots
        )
        trajectory = np.zeros((len(frames), 12), dtype=np.float64)
        for grade in shot_grades:
            trajectory[grade.shot.start_frame : grade.shot.end_frame + 1] = (
                grade.frame_parameters
            )
        audit = {
            "reference_type": "external_video",
            "reference_frame_count": len(reference_frames),
            "accepted_shots": sum(grade.accepted for grade in shot_grades),
            "total_shots": len(shot_grades),
            "pool_backends": [backend.name for backend in self.anchor_backends],
        }
        return GradeGraph(
            instruction=instruction,
            storyboard=storyboard,
            shots=shot_grades,
            frame_parameters=trajectory,
            backend=",".join(backend.name for backend in self.anchor_backends),
            critic=self.critic.name,
            orchestrator=f"{self.search.name}+external-reference-video",
            hero_anchor=HeroAnchorRecord(
                frame_index=-1,
                shot_id=-1,
                parameters=reference_grade.parameters.to_vector(),
                backend=reference_grade.backend,
                score=1.0,
                ranked_candidates=(),
                selection_reason="user-supplied reference video storyboard",
                attempts=(audit,),
            ),
            hero_anchor_attempts=(audit,),
        )

    def run(
        self,
        frames: Sequence[Image.Image],
        fps: float,
        instruction: str,
        reference_frames: Optional[Sequence[Image.Image]] = None,
    ) -> GradeGraph:
        normalized = tuple(frame.convert("RGB") for frame in frames)
        if not normalized:
            raise ValueError("At least one video frame is required.")
        storyboard = self.shot_planner.plan(
            normalized,
            fps,
            instruction,
            anchors_per_shot=self.anchors_per_shot,
        )
        if reference_frames is not None:
            return self._run_with_external_reference(
                normalized,
                instruction,
                storyboard,
                reference_frames,
            )
        hero_candidates = list(storyboard.hero_anchor_candidates)
        if storyboard.hero_anchor_frame is not None:
            hero_candidates = [
                storyboard.hero_anchor_frame,
                *(
                    frame
                    for frame in hero_candidates
                    if frame != storyboard.hero_anchor_frame
                ),
            ]
        if not hero_candidates:
            hero_candidates = [
                frame
                for shot in storyboard.shots
                for frame in shot.anchor_frames
            ]
        hero_candidates = list(dict.fromkeys(hero_candidates))

        content_groups = self._content_groups(normalized, storyboard.shots)
        if len(content_groups) > 1:
            grouped_grades: list[ShotGrade] = []
            grouped_attempts: list[dict[str, object]] = []
            primary_reference: Optional[HeroAnchorReference] = None
            ranked_all_heroes: list[int] = []
            for group_index, (group_label, group_shots) in enumerate(content_groups):
                group_candidates = self._rank_group_hero_candidates(
                    normalized, group_shots
                )
                ranked_all_heroes.extend(group_candidates)
                selected_reference: Optional[HeroAnchorReference] = None
                selected_audit: Optional[dict[str, object]] = None
                selected_grid: Optional[dict[int, tuple[AnchorGrade, ...]]] = None
                for frame_index in group_candidates[: self.maximum_hero_attempts]:
                    hero_shot = self._shot_for_frame(storyboard, frame_index)
                    reference, audit, hero_candidate_grid = (
                        self.search.select_hero_reference(
                            normalized,
                            instruction,
                            frame_index,
                            hero_shot.shot_id,
                        )
                    )
                    audit = {
                        "round": len(grouped_attempts) + 1,
                        "content_group": group_label,
                        "group_index": group_index,
                        "group_shot_ids": [shot.shot_id for shot in group_shots],
                        **audit,
                    }
                    grouped_attempts.append(audit)
                    if reference is not None:
                        selected_reference = reference
                        selected_audit = audit
                        selected_grid = hero_candidate_grid
                        break
                if selected_reference is None:
                    continue
                if primary_reference is None:
                    primary_reference = selected_reference
                candidate_grid = self.search.prepopulate_storyboard_candidate_grid(
                    normalized,
                    instruction,
                    group_shots,
                    selected_reference,
                    selected_grid,
                )
                group_grades = tuple(
                    self._run_shot(
                        normalized,
                        instruction,
                        shot,
                        selected_reference,
                        candidate_grid,
                    )
                    for shot in group_shots
                )
                grouped_grades.extend(group_grades)
                if selected_audit is not None:
                    selected_audit.update(
                        {
                            "accepted_shots": sum(
                                grade.accepted for grade in group_grades
                            ),
                            "total_shots": len(group_grades),
                            "mean_shot_score": float(
                                np.mean([grade.confidence for grade in group_grades])
                            ),
                            "rolled_back_shots": [
                                grade.shot.shot_id
                                for grade in group_grades
                                if grade.rolled_back
                            ],
                            "global_committed": True,
                            "local_hero_committed": True,
                        }
                    )
            if grouped_grades and primary_reference is not None:
                by_id = {grade.shot.shot_id: grade for grade in grouped_grades}
                shot_grades = tuple(
                    by_id.get(shot.shot_id)
                    or self._identity_shot_grade(
                        shot, "content_group_hero_selection_failed"
                    )
                    for shot in storyboard.shots
                )
                trajectory = np.zeros((len(normalized), 12), dtype=np.float64)
                for grade in shot_grades:
                    trajectory[
                        grade.shot.start_frame : grade.shot.end_frame + 1
                    ] = grade.frame_parameters
                hero_record = HeroAnchorRecord(
                    frame_index=primary_reference.frame_index,
                    shot_id=primary_reference.shot_id,
                    parameters=primary_reference.grade.parameters.to_vector(),
                    backend=primary_reference.grade.backend,
                    score=primary_reference.grade.score,
                    ranked_candidates=tuple(dict.fromkeys(ranked_all_heroes)),
                    selection_reason=(
                        "content-aware local HeroAnchor matching; each content "
                        "group selects its own representative hero"
                    ),
                    attempts=tuple(grouped_attempts),
                )
                return GradeGraph(
                    instruction=instruction,
                    storyboard=storyboard,
                    shots=shot_grades,
                    frame_parameters=trajectory,
                    backend=",".join(backend.name for backend in self.anchor_backends),
                    critic=self.critic.name,
                    orchestrator=f"{self.search.name}+content-local-heroes",
                    hero_anchor=hero_record,
                    hero_anchor_attempts=tuple(grouped_attempts),
                )

        global_attempts: list[dict[str, object]] = []
        best: Optional[
            tuple[
                tuple[int, float],
                HeroAnchorReference,
                tuple[ShotGrade, ...],
                int,
            ]
        ] = None
        for hero_round, frame_index in enumerate(
            hero_candidates[: self.maximum_hero_attempts], start=1
        ):
            hero_shot = self._shot_for_frame(storyboard, frame_index)
            reference, audit, hero_candidate_grid = self.search.select_hero_reference(
                normalized,
                instruction,
                frame_index,
                hero_shot.shot_id,
            )
            audit = {"round": hero_round, **audit}
            if reference is None:
                audit["global_committed"] = False
                audit["rollback_scope"] = "hero_look_development"
                global_attempts.append(audit)
                continue
            candidate_grid = self.search.prepopulate_storyboard_candidate_grid(
                normalized,
                instruction,
                storyboard.shots,
                reference,
                hero_candidate_grid,
            )
            candidate_grades = tuple(
                self._run_shot(
                    normalized,
                    instruction,
                    shot,
                    reference,
                    candidate_grid,
                )
                for shot in storyboard.shots
            )
            accepted_count = sum(grade.accepted for grade in candidate_grades)
            mean_score = float(
                np.mean([grade.confidence for grade in candidate_grades])
            )
            all_accepted = accepted_count == len(candidate_grades)
            audit.update(
                {
                    "accepted_shots": accepted_count,
                    "total_shots": len(candidate_grades),
                    "mean_shot_score": mean_score,
                    "rolled_back_shots": [
                        grade.shot.shot_id
                        for grade in candidate_grades
                        if grade.rolled_back
                    ],
                    "shot_rollback_reasons": {
                        str(grade.shot.shot_id): grade.rollback_reason
                        for grade in candidate_grades
                        if grade.rolled_back
                    },
                    "global_committed": all_accepted,
                }
            )
            global_attempts.append(audit)
            objective = (accepted_count, mean_score)
            if best is None or objective > best[0]:
                best = (
                    objective,
                    reference,
                    candidate_grades,
                    len(global_attempts) - 1,
                )
            if all_accepted:
                break

        committed = best
        if committed is None:
            if best is not None:
                global_attempts[best[3]]["best_uncommitted_attempt"] = True
            rejection_reasons = [
                reason
                for attempt in global_attempts
                for proposal in attempt.get("proposals", [])
                for reason in proposal.get("reasons", [])
            ]
            shot_rejection_reasons = [
                reason
                for attempt in global_attempts
                for reason in attempt.get("shot_rollback_reasons", {}).values()
                if reason
            ]
            rollback_reason = (
                str((shot_rejection_reasons or rejection_reasons)[-1])
                if shot_rejection_reasons or rejection_reasons
                else "hero_anchor_rejected"
            )
            shot_grades = tuple(
                self._identity_shot_grade(shot, rollback_reason)
                for shot in storyboard.shots
            )
            hero_record = None
        else:
            _, selected_reference, shot_grades, selected_attempt = committed
            global_attempts[selected_attempt]["selected_global_attempt"] = True
            hero_record = HeroAnchorRecord(
                frame_index=selected_reference.frame_index,
                shot_id=selected_reference.shot_id,
                parameters=selected_reference.grade.parameters.to_vector(),
                backend=selected_reference.grade.backend,
                score=selected_reference.grade.score,
                ranked_candidates=tuple(hero_candidates),
                selection_reason=storyboard.hero_selection_reason,
                attempts=tuple(global_attempts),
            )
        trajectory = np.zeros((len(normalized), 12), dtype=np.float64)
        for grade in shot_grades:
            trajectory[grade.shot.start_frame : grade.shot.end_frame + 1] = (
                grade.frame_parameters
            )
        return GradeGraph(
            instruction=instruction,
            storyboard=storyboard,
            shots=shot_grades,
            frame_parameters=trajectory,
            backend=",".join(backend.name for backend in self.anchor_backends),
            critic=self.critic.name,
            orchestrator=self.search.name,
            hero_anchor=hero_record,
            hero_anchor_attempts=tuple(global_attempts),
        )
