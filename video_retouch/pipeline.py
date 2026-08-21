"""Closed-loop video-to-grade-parameter pipeline."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from retouch_agent import RetouchExecutor

from .backends import (
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
    ) -> ShotGrade:
        outcome = self.search.search(
            frames,
            instruction,
            shot,
            hero_reference=hero_reference,
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
            accepted=True,
            rolled_back=False,
            rollback_reason=None,
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

    def run(
        self,
        frames: Sequence[Image.Image],
        fps: float,
        instruction: str,
        *,
        reference_frames: Optional[Sequence[Image.Image]] = None,
        reference_fps: Optional[float] = None,
    ) -> GradeGraph:
        normalized = tuple(frame.convert("RGB") for frame in frames)
        if not normalized:
            raise ValueError("At least one video frame is required.")
        if reference_frames is not None and not reference_frames:
            raise ValueError("Reference video must contain at least one frame.")
        if reference_frames is not None and (
            reference_fps is None or reference_fps <= 0.0
        ):
            raise ValueError("reference_fps must be positive with reference_frames.")
        storyboard = self.shot_planner.plan(
            normalized,
            fps,
            instruction,
            anchors_per_shot=self.anchors_per_shot,
        )
        external_reference = reference_frames is not None
        hero_frames = (
            tuple(frame.convert("RGB") for frame in reference_frames)
            if reference_frames is not None
            else normalized
        )
        hero_storyboard = (
            self.shot_planner.plan(
                hero_frames,
                float(reference_fps),
                instruction,
                anchors_per_shot=self.anchors_per_shot,
            )
            if external_reference
            else storyboard
        )
        hero_candidates = list(hero_storyboard.hero_anchor_candidates)
        if hero_storyboard.hero_anchor_frame is not None:
            hero_candidates = [
                hero_storyboard.hero_anchor_frame,
                *(
                    frame
                    for frame in hero_candidates
                    if frame != hero_storyboard.hero_anchor_frame
                ),
            ]
        if not hero_candidates:
            hero_candidates = [
                frame
                for shot in hero_storyboard.shots
                for frame in shot.anchor_frames
            ]
        hero_candidates = list(dict.fromkeys(hero_candidates))

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
            hero_shot = self._shot_for_frame(hero_storyboard, frame_index)
            reference, audit = self.search.select_hero_reference(
                hero_frames,
                instruction,
                frame_index,
                hero_shot.shot_id,
                external=external_reference,
            )
            audit = {"round": hero_round, **audit}
            if reference is None:
                audit["global_committed"] = False
                audit["rollback_scope"] = "hero_look_development"
                global_attempts.append(audit)
                continue
            candidate_grades = tuple(
                self._run_shot(normalized, instruction, shot, reference)
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

        committed = (
            best
            if best is not None and best[0][0] == len(storyboard.shots)
            else None
        )
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
                selection_reason=hero_storyboard.hero_selection_reason,
                attempts=tuple(global_attempts),
                source_video=(
                    "reference_video" if external_reference else "target_video"
                ),
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
