"""PhotoAgent-style UCT search over video color-grading trajectories."""

from __future__ import annotations

import inspect
import math
import random
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from retouch_agent import RetouchExecutor, RetouchParameters
from retouch_agent.parameters import PARAMETER_LOWER_BOUNDS, PARAMETER_UPPER_BOUNDS

from .backends import (
    AnchorGrade,
    AnchorRetouchBackend,
    HeroAnchorReference,
    STYLE_STRENGTH_GUARDRAILS,
)
from .critic import ShotCritic, ShotCritique
from .models import GradeAttempt, ShotPlan
from .propagation import BayesianParameterDiffuser, DiffusedGrade


@dataclass
class _Node:
    choices: tuple[int, ...]
    parent: Optional["_Node"] = None
    action: Optional[int] = None
    children: dict[int, "_Node"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class SearchEvaluation:
    anchor_indices: tuple[int, ...]
    choices: tuple[int, ...]
    grades: tuple[AnchorGrade, ...]
    diffused: DiffusedGrade
    critique: ShotCritique
    reward: float
    round_index: int


@dataclass(frozen=True)
class SearchOutcome:
    selected: Optional[SearchEvaluation]
    attempts: tuple[GradeAttempt, ...]
    memory: dict[str, object]


CandidateGrid = dict[int, tuple[AnchorGrade, ...]]


class AestheticMCTSSearch:
    """Explore editor/Anchor combinations with UCT and explicit rollback.

    A state is a partial assignment of one editor proposal to every Anchor.
    Terminal states are rendered into a full shot trajectory, independently
    evaluated, and back-propagated through the tree. Rejected trajectories may
    add a critic-recommended Anchor and start a deeper search round.
    """

    name = "photoagent-uct-mcts"

    def __init__(
        self,
        backends: Sequence[AnchorRetouchBackend],
        critic: ShotCritic,
        diffuser: Optional[BayesianParameterDiffuser] = None,
        executor: Optional[RetouchExecutor] = None,
        maximum_evaluations: int = 24,
        maximum_anchors: int = 3,
        exploration_constant: float = math.sqrt(2.0),
        rejection_penalty: float = 0.25,
        seed: int = 7,
    ) -> None:
        if not backends:
            raise ValueError("MCTS requires at least one editing Agent.")
        names = [backend.name for backend in backends]
        if len(set(names)) != len(names):
            raise ValueError("Editing Agent names must be unique.")
        if maximum_evaluations < 1:
            raise ValueError("maximum_evaluations must be positive.")
        if maximum_anchors < 1:
            raise ValueError("maximum_anchors must be positive.")
        self.backends = tuple(backends)
        self.critic = critic
        self.diffuser = diffuser or BayesianParameterDiffuser()
        self.executor = executor or RetouchExecutor()
        self.maximum_evaluations = int(maximum_evaluations)
        self.maximum_anchors = int(maximum_anchors)
        self.exploration_constant = float(exploration_constant)
        self.rejection_penalty = float(rejection_penalty)
        self.random = random.Random(seed)

    @staticmethod
    def _failed_grade(
        frame: Image.Image,
        frame_index: int,
        shot_id: int,
        backend: str,
        error: Exception,
    ) -> AnchorGrade:
        parameters = RetouchParameters.from_mapping(
            STYLE_STRENGTH_GUARDRAILS["preferred_defaults"], clamp=True
        )
        return AnchorGrade(
            frame_index=frame_index,
            parameters=parameters,
            preview=RetouchExecutor().apply(frame.convert("RGB"), parameters),
            valid=True,
            score=-0.25,
            backend=backend,
            metadata={
                "shot_id": shot_id,
                "error": f"{type(error).__name__}: {error}",
                "fallback": "failed_anchor_assertive_floor",
                "identity_fallback_disabled": True,
            },
        )

    def _propose(
        self,
        frames: Sequence[Image.Image],
        instruction: str,
        shot: ShotPlan,
        frame_index: int,
        hero_reference: Optional[HeroAnchorReference] = None,
    ) -> tuple[AnchorGrade, ...]:
        proposals = []
        for backend in self.backends:
            try:
                if (
                    hero_reference is not None
                    and frame_index == hero_reference.frame_index
                ):
                    grade = hero_reference.grade
                elif hero_reference is not None and callable(
                    getattr(backend, "grade_with_reference", None)
                ):
                    grade = backend.grade_with_reference(
                        frames[frame_index],
                        instruction,
                        frame_index,
                        shot.shot_id,
                        hero_reference,
                    )
                else:
                    grade = backend.grade(
                        frames[frame_index], instruction, frame_index, shot.shot_id
                    )
                    if hero_reference is not None:
                        grade = replace(
                            grade,
                            metadata={
                                **grade.metadata,
                                "hero_match_fallback": (
                                    "backend_does_not_support_visual_reference"
                                ),
                                "requested_hero_frame": hero_reference.frame_index,
                            },
                        )
            except Exception as error:
                grade = self._failed_grade(
                    frames[frame_index],
                    frame_index,
                    shot.shot_id,
                    backend.name,
                    error,
                )
            proposals.append(grade)
        return tuple(proposals)

    def _populate_candidate_grid(
        self,
        frames: Sequence[Image.Image],
        instruction: str,
        shot: ShotPlan,
        frame_indices: tuple[int, ...],
        candidate_grid: dict[int, tuple[AnchorGrade, ...]],
        hero_reference: Optional[HeroAnchorReference] = None,
    ) -> None:
        missing = tuple(
            frame_index
            for frame_index in frame_indices
            if frame_index not in candidate_grid
        )
        if not missing:
            return
        for frame_index in missing:
            candidate_grid[frame_index] = tuple()
        for backend in self.backends:
            try:
                if hero_reference is not None and callable(
                    getattr(backend, "batch_grade_with_reference", None)
                ):
                    grades = backend.batch_grade_with_reference(
                        frames,
                        instruction,
                        missing,
                        shot.shot_id,
                        hero_reference,
                    )
                elif hero_reference is None and callable(
                    getattr(backend, "batch_grade", None)
                ):
                    grades = backend.batch_grade(
                        frames,
                        instruction,
                        missing,
                        shot.shot_id,
                    )
                else:
                    proposed = []
                    for frame_index in missing:
                        if (
                            hero_reference is not None
                            and frame_index == hero_reference.frame_index
                        ):
                            grade = hero_reference.grade
                        elif hero_reference is not None and callable(
                            getattr(backend, "grade_with_reference", None)
                        ):
                            grade = backend.grade_with_reference(
                                frames[frame_index],
                                instruction,
                                frame_index,
                                shot.shot_id,
                                hero_reference,
                            )
                        else:
                            grade = backend.grade(
                                frames[frame_index],
                                instruction,
                                frame_index,
                                shot.shot_id,
                            )
                            if hero_reference is not None:
                                grade = replace(
                                    grade,
                                    metadata={
                                        **grade.metadata,
                                        "hero_match_fallback": (
                                            "backend_does_not_support_visual_reference"
                                        ),
                                        "requested_hero_frame": (
                                            hero_reference.frame_index
                                        ),
                                    },
                                )
                        proposed.append(grade)
                    grades = tuple(proposed)
            except Exception as error:
                grades = tuple(
                    self._failed_grade(
                        frames[frame_index],
                        frame_index,
                        shot.shot_id,
                        backend.name,
                        error,
                    )
                    for frame_index in missing
                )
            by_frame = {grade.frame_index: grade for grade in grades}
            for frame_index in missing:
                grade = by_frame.get(frame_index)
                if grade is None:
                    grade = self._failed_grade(
                        frames[frame_index],
                        frame_index,
                        shot.shot_id,
                        backend.name,
                        RuntimeError("batch backend did not return this anchor"),
                    )
                candidate_grid[frame_index] = (*candidate_grid[frame_index], grade)

    def prepopulate_storyboard_candidate_grid(
        self,
        frames: Sequence[Image.Image],
        instruction: str,
        shots: Sequence[ShotPlan],
        hero_reference: HeroAnchorReference,
        hero_candidate_grid: Optional[CandidateGrid] = None,
    ) -> CandidateGrid:
        grid: CandidateGrid = {}
        anchor_shot_ids: dict[int, int] = {}
        for shot in shots:
            for frame_index in dict.fromkeys(sorted(shot.anchor_frames)):
                anchor_shot_ids[int(frame_index)] = shot.shot_id
        anchor_frames = tuple(sorted(anchor_shot_ids))
        if not anchor_frames:
            return grid
        for frame_index in anchor_frames:
            if frame_index == hero_reference.frame_index:
                if hero_candidate_grid is not None:
                    grid[frame_index] = hero_candidate_grid.get(
                        frame_index, (hero_reference.grade,)
                    )
                else:
                    grid[frame_index] = (hero_reference.grade,)
            else:
                grid[frame_index] = tuple()
        missing = tuple(
            frame_index
            for frame_index in anchor_frames
            if frame_index != hero_reference.frame_index
        )
        if not missing:
            return grid
        for backend in self.backends:
            try:
                if callable(getattr(backend, "batch_grade_storyboard", None)):
                    grades = backend.batch_grade_storyboard(
                        frames,
                        instruction,
                        missing,
                        anchor_shot_ids,
                        hero_reference,
                    )
                elif callable(getattr(backend, "batch_grade_with_reference", None)):
                    grades = []
                    for shot in shots:
                        shot_indices = tuple(
                            index
                            for index in sorted(shot.anchor_frames)
                            if index in missing
                        )
                        if not shot_indices:
                            continue
                        grades.extend(
                            backend.batch_grade_with_reference(
                                frames,
                                instruction,
                                shot_indices,
                                shot.shot_id,
                                hero_reference,
                            )
                        )
                    grades = tuple(grades)
                else:
                    grades = []
                    for frame_index in missing:
                        shot_id = anchor_shot_ids[frame_index]
                        if callable(getattr(backend, "grade_with_reference", None)):
                            grades.append(
                                backend.grade_with_reference(
                                    frames[frame_index],
                                    instruction,
                                    frame_index,
                                    shot_id,
                                    hero_reference,
                                )
                            )
                        else:
                            grade = backend.grade(
                                frames[frame_index],
                                instruction,
                                frame_index,
                                shot_id,
                            )
                            grades.append(
                                replace(
                                    grade,
                                    metadata={
                                        **grade.metadata,
                                        "hero_match_fallback": (
                                            "backend_does_not_support_visual_reference"
                                        ),
                                        "requested_hero_frame": (
                                            hero_reference.frame_index
                                        ),
                                    },
                                )
                            )
                    grades = tuple(grades)
            except Exception as error:
                grades = tuple(
                    self._failed_grade(
                        frames[frame_index],
                        frame_index,
                        anchor_shot_ids[frame_index],
                        backend.name,
                        error,
                    )
                    for frame_index in missing
                )
            by_frame = {grade.frame_index: grade for grade in grades}
            for frame_index in missing:
                if frame_index == hero_reference.frame_index:
                    continue
                grade = by_frame.get(frame_index)
                if grade is None:
                    grade = self._failed_grade(
                        frames[frame_index],
                        frame_index,
                        anchor_shot_ids[frame_index],
                        backend.name,
                        RuntimeError("storyboard batch backend omitted this anchor"),
                    )
                grid[frame_index] = (*grid[frame_index], grade)
        return grid

    def propose_hero(
        self,
        frames: Sequence[Image.Image],
        instruction: str,
        frame_index: int,
        shot_id: int,
    ) -> tuple[AnchorGrade, ...]:
        hero_shot = ShotPlan(
            shot_id=shot_id,
            start_frame=frame_index,
            end_frame=frame_index,
            anchor_frames=(frame_index,),
            anchor_candidates=(frame_index,),
            description="HeroAnchor look-development candidate",
        )
        return self._propose(
            frames,
            instruction,
            hero_shot,
            frame_index,
            hero_reference=None,
        )

    def select_hero_reference(
        self,
        frames: Sequence[Image.Image],
        instruction: str,
        frame_index: int,
        shot_id: int,
    ) -> tuple[Optional[HeroAnchorReference], dict[str, object], CandidateGrid]:
        hero_shot = ShotPlan(
            shot_id=shot_id,
            start_frame=frame_index,
            end_frame=frame_index,
            anchor_frames=(frame_index,),
            anchor_candidates=(frame_index,),
            description="HeroAnchor look-development candidate",
        )
        proposals = self.propose_hero(
            frames, instruction, frame_index, shot_id
        )
        candidate_grid = {frame_index: proposals}
        evaluations = [
            self._evaluate(
                frames,
                instruction,
                hero_shot,
                (frame_index,),
                candidate_grid,
                (choice,),
                round_index=0,
                hero_reference=None,
            )
            for choice in range(len(proposals))
        ]
        accepted = [
            evaluation
            for evaluation in evaluations
            if evaluation.grades[0].valid and evaluation.critique.accepted
        ]
        selected = (
            max(accepted, key=lambda item: (item.critique.score, item.reward))
            if accepted
            else None
        )
        if selected is None and evaluations:
            selected = max(
                evaluations,
                key=lambda item: (item.reward, item.critique.score),
            )
        audit = {
            "hero_frame": frame_index,
            "hero_shot_id": shot_id,
            "proposals": [
                {
                    "backend": evaluation.grades[0].backend,
                    "valid": evaluation.grades[0].valid,
                    "proposal_score": evaluation.grades[0].score,
                    "critic_score": evaluation.critique.score,
                    "accepted": evaluation.critique.accepted,
                    "reasons": list(evaluation.critique.reasons),
                    "parameters": evaluation.grades[0].parameters.to_dict(),
                }
                for evaluation in evaluations
            ],
            "selected_backend": (
                None if selected is None else selected.grades[0].backend
            ),
            "accepted": selected is not None and selected.critique.accepted,
            "committed_as_reference": selected is not None,
        }
        if selected is None:
            return None, audit, candidate_grid
        return (
            HeroAnchorReference(
                frame_index=frame_index,
                shot_id=shot_id,
                source=frames[frame_index].convert("RGB"),
                grade=selected.grades[0],
            ),
            audit,
            candidate_grid,
        )

    def _render(
        self,
        frames: Sequence[Image.Image],
        parameters: np.ndarray,
    ) -> tuple[Image.Image, ...]:
        return tuple(
            self.executor.apply(frame, RetouchParameters.from_vector(values))
            for frame, values in zip(frames, parameters)
        )

    @staticmethod
    def _strength_feedback_factor(critique: ShotCritique) -> float:
        text = " ".join(critique.reasons).lower()
        if any(
            phrase in text
            for phrase in (
                "no visible difference",
                "virtually no visible difference",
                "not visible",
                "too subtle",
                "not strong",
                "not enough",
                "failing the requested strong",
            )
        ):
            return 1.25
        return 1.12

    def _amplify_evaluation(
        self,
        frames: Sequence[Image.Image],
        instruction: str,
        shot: ShotPlan,
        evaluation: SearchEvaluation,
        factor: float,
        hero_reference: Optional[HeroAnchorReference] = None,
    ) -> SearchEvaluation:
        def amplify(values: np.ndarray) -> np.ndarray:
            upper = np.minimum(PARAMETER_UPPER_BOUNDS, 0.48)
            lower = np.maximum(PARAMETER_LOWER_BOUNDS, -0.48)
            upper[0] = min(PARAMETER_UPPER_BOUNDS[0], 0.65)
            lower[0] = max(PARAMETER_LOWER_BOUNDS[0], -0.65)
            upper[9] = min(PARAMETER_UPPER_BOUNDS[9], 0.55)
            lower[9] = max(PARAMETER_LOWER_BOUNDS[9], -0.55)
            return np.clip(
                np.asarray(values, dtype=np.float64) * factor,
                lower,
                upper,
            )

        amplified_grades = tuple(
            replace(
                grade,
                parameters=RetouchParameters.from_vector(
                    amplify(grade.parameters.to_vector()), clamp=True
                ),
                preview=self.executor.apply(
                    frames[grade.frame_index],
                    RetouchParameters.from_vector(
                        amplify(grade.parameters.to_vector()), clamp=True
                    ),
                ),
                score=max(float(grade.score), float(evaluation.critique.score)),
                valid=True,
                metadata={
                    **grade.metadata,
                    "critic_feedback_amplified": True,
                    "feedback_factor": factor,
                    "feedback_source": list(evaluation.critique.reasons),
                },
            )
            for grade in evaluation.grades
        )
        diffused = self.diffuser.diffuse(frames, shot, amplified_grades)
        source = frames[shot.start_frame : shot.end_frame + 1]
        output = self._render(source, diffused.frame_parameters)
        critic_arguments = (
            source,
            output,
            diffused.frame_parameters,
            diffused.frame_uncertainty,
            shot,
            instruction,
            amplified_grades,
        )
        if "hero_reference" in inspect.signature(
            self.critic.evaluate
        ).parameters:
            critique = self.critic.evaluate(
                *critic_arguments, hero_reference=hero_reference
            )
        else:
            critique = self.critic.evaluate(*critic_arguments)
        critique = replace(
            critique,
            reasons=(
                *critique.reasons,
                f"critic_feedback_amplified_by_{factor:.2f}x",
            ),
            metadata={
                **critique.metadata,
                "critic_feedback_amplified": True,
                "feedback_factor": factor,
                "previous_reasons": list(evaluation.critique.reasons),
            },
        )
        reward = float(
            np.clip(
                (
                    critique.score
                    if critique.accepted
                    else critique.score - self.rejection_penalty * 0.25
                ),
                -1.0,
                1.0,
            )
        )
        return SearchEvaluation(
            anchor_indices=evaluation.anchor_indices,
            choices=evaluation.choices,
            grades=amplified_grades,
            diffused=diffused,
            critique=critique,
            reward=reward,
            round_index=evaluation.round_index,
        )

    def _evaluate(
        self,
        frames: Sequence[Image.Image],
        instruction: str,
        shot: ShotPlan,
        anchor_indices: tuple[int, ...],
        candidate_grid: dict[int, tuple[AnchorGrade, ...]],
        choices: tuple[int, ...],
        round_index: int,
        hero_reference: Optional[HeroAnchorReference] = None,
    ) -> SearchEvaluation:
        grades = tuple(
            candidate_grid[frame_index][choice]
            for frame_index, choice in zip(anchor_indices, choices)
        )
        diffused = self.diffuser.diffuse(frames, shot, grades)
        source = frames[shot.start_frame : shot.end_frame + 1]
        output = self._render(source, diffused.frame_parameters)
        critic_arguments = (
            source,
            output,
            diffused.frame_parameters,
            diffused.frame_uncertainty,
            shot,
            instruction,
            grades,
        )
        if "hero_reference" in inspect.signature(
            self.critic.evaluate
        ).parameters:
            critique = self.critic.evaluate(
                *critic_arguments, hero_reference=hero_reference
            )
        else:
            critique = self.critic.evaluate(*critic_arguments)
        reward = float(
            np.clip(
                (
                    critique.score
                    if critique.accepted
                    else critique.score - self.rejection_penalty
                ),
                -1.0,
                1.0,
            )
        )
        return SearchEvaluation(
            anchor_indices=anchor_indices,
            choices=choices,
            grades=grades,
            diffused=diffused,
            critique=critique,
            reward=reward,
            round_index=round_index,
        )

    def _select_child(self, node: _Node) -> _Node:
        log_parent = math.log(max(1, node.visits))

        def uct(child: _Node) -> tuple[float, int]:
            if child.visits == 0:
                return (float("inf"), -(child.action or 0))
            bonus = self.exploration_constant * math.sqrt(log_parent / child.visits)
            return (child.mean_value + bonus, -(child.action or 0))

        return max(node.children.values(), key=uct)

    def _tree_policy(self, root: _Node, depth: int, action_count: int) -> _Node:
        node = root
        while len(node.choices) < depth:
            unexpanded = [
                action for action in range(action_count) if action not in node.children
            ]
            if unexpanded:
                action = self.random.choice(unexpanded)
                child = _Node(
                    choices=(*node.choices, action), parent=node, action=action
                )
                node.children[action] = child
                return child
            node = self._select_child(node)
        return node

    def _rollout(self, node: _Node, depth: int, action_count: int) -> tuple[int, ...]:
        choices = list(node.choices)
        while len(choices) < depth:
            choices.append(self.random.randrange(action_count))
        return tuple(choices)

    @staticmethod
    def _backpropagate(node: _Node, reward: float) -> None:
        current: Optional[_Node] = node
        while current is not None:
            current.visits += 1
            current.value_sum += reward
            current = current.parent

    def search(
        self,
        frames: Sequence[Image.Image],
        instruction: str,
        shot: ShotPlan,
        hero_reference: Optional[HeroAnchorReference] = None,
        candidate_grid: Optional[CandidateGrid] = None,
    ) -> SearchOutcome:
        anchor_indices = list(dict.fromkeys(sorted(shot.anchor_frames)))
        if len(anchor_indices) > self.maximum_anchors:
            anchor_indices = anchor_indices[: self.maximum_anchors]
        candidate_grid = {} if candidate_grid is None else candidate_grid
        evaluations: dict[tuple[tuple[int, ...], tuple[int, ...]], SearchEvaluation] = (
            {}
        )
        attempts: list[GradeAttempt] = []
        rounds: list[dict[str, object]] = []
        simulations = 0
        round_index = 0
        used_anchor_frames = set(anchor_indices)

        while simulations < self.maximum_evaluations:
            round_index += 1
            anchors = tuple(sorted(anchor_indices))
            self._populate_candidate_grid(
                frames,
                instruction,
                shot,
                anchors,
                candidate_grid,
                hero_reference,
            )
            action_count = len(self.backends)
            combination_count = action_count ** len(anchors)
            root = _Node(choices=())
            round_keys: set[tuple[int, ...]] = set()
            round_simulations = 0

            while (
                simulations < self.maximum_evaluations
                and len(round_keys) < combination_count
            ):
                node = self._tree_policy(root, len(anchors), action_count)
                choices = self._rollout(node, len(anchors), action_count)
                key = (anchors, choices)
                if key in evaluations:
                    evaluation = evaluations[key]
                    self._backpropagate(node, evaluation.reward)
                    round_simulations += 1
                    if round_simulations > combination_count * 8:
                        break
                    continue
                evaluation = self._evaluate(
                    frames,
                    instruction,
                    shot,
                    anchors,
                    candidate_grid,
                    choices,
                    round_index,
                    hero_reference,
                )
                evaluations[key] = evaluation
                round_keys.add(choices)
                simulations += 1
                round_simulations += 1
                self._backpropagate(node, evaluation.reward)
                path = [
                    {
                        "anchor_frame": frame_index,
                        "editor": grade.backend,
                        "valid": grade.valid,
                        "proposal_score": grade.score,
                    }
                    for frame_index, grade in zip(anchors, evaluation.grades)
                ]
                attempts.append(
                    GradeAttempt(
                        attempt=len(attempts) + 1,
                        anchor_frames=anchors,
                        score=evaluation.critique.score,
                        accepted=evaluation.critique.accepted,
                        metrics=evaluation.critique.metrics,
                        reasons=evaluation.critique.reasons,
                        recommended_anchor=evaluation.critique.recommended_anchor,
                        metadata={
                            **evaluation.critique.metadata,
                            "temporal_stabilization": (
                                evaluation.diffused.stabilization
                            ),
                            "search": {
                                "algorithm": self.name,
                                "round": round_index,
                                "choices": list(choices),
                                "path": path,
                                "reward": evaluation.reward,
                                "hero_anchor_frame": (
                                    None
                                    if hero_reference is None
                                    else hero_reference.frame_index
                                ),
                            },
                        },
                    )
                )

            round_evaluations = [
                value
                for (round_anchors, _), value in evaluations.items()
                if round_anchors == anchors
            ]
            rounds.append(
                {
                    "round": round_index,
                    "anchors": list(anchors),
                    "terminal_combinations": combination_count,
                    "evaluated_combinations": len(round_evaluations),
                    "root_visits": root.visits,
                    "root_value": root.mean_value,
                }
            )
            accepted = [
                evaluation
                for evaluation in round_evaluations
                if evaluation.critique.accepted
            ]
            if accepted:
                selected = max(
                    accepted,
                    key=lambda value: (value.critique.score, value.reward),
                )
                return SearchOutcome(
                    selected=selected,
                    attempts=tuple(attempts),
                    memory=self._memory(
                        candidate_grid,
                        evaluations,
                        rounds,
                        selected,
                        False,
                        hero_reference,
                    ),
                )
            if not round_evaluations:
                break
            best_rejected = max(
                round_evaluations,
                key=lambda value: (value.reward, value.critique.score),
            )
            next_anchor = best_rejected.critique.recommended_anchor
            if (
                next_anchor is None
                or next_anchor in used_anchor_frames
                or next_anchor < shot.start_frame
                or next_anchor > shot.end_frame
            ):
                next_anchor = next(
                    (
                        frame
                        for frame in shot.anchor_candidates
                        if frame not in used_anchor_frames
                    ),
                    None,
                )
            if next_anchor is None:
                amplified = self._amplify_evaluation(
                    frames,
                    instruction,
                    shot,
                    best_rejected,
                    self._strength_feedback_factor(best_rejected.critique),
                    hero_reference,
                )
                evaluations[(anchors, (*best_rejected.choices, simulations))] = amplified
                attempts.append(
                    GradeAttempt(
                        attempt=len(attempts) + 1,
                        anchor_frames=anchors,
                        score=amplified.critique.score,
                        accepted=amplified.critique.accepted,
                        metrics=amplified.critique.metrics,
                        reasons=amplified.critique.reasons,
                        recommended_anchor=amplified.critique.recommended_anchor,
                        metadata={
                            **amplified.critique.metadata,
                            "positive_rollback_feedback": True,
                            "source_attempt": len(attempts),
                        },
                    )
                )
                if amplified.critique.accepted:
                    return SearchOutcome(
                        selected=amplified,
                        attempts=tuple(attempts),
                        memory=self._memory(
                            candidate_grid,
                            evaluations,
                            rounds,
                            amplified,
                            False,
                            hero_reference,
                        ),
                    )
                rounds[-1].update(
                    {
                        "rolled_back": True,
                        "rollback_reasons": list(amplified.critique.reasons),
                        "anchor_action": "amplify",
                        "positive_rollback_feedback": True,
                    }
                )
                break
            previous_anchors = tuple(sorted(anchor_indices))
            # A rejected round is transactional: discard its trajectory and
            # replace an Anchor before regenerating proposals. This avoids
            # repeatedly conditioning on a bad/occluded Anchor.
            if len(anchor_indices) >= self.maximum_anchors:
                anchor_indices = anchor_indices[1:]
            elif anchor_indices:
                anchor_indices = anchor_indices[1:]
            anchor_indices.append(next_anchor)
            used_anchor_frames.add(next_anchor)
            rounds[-1].update(
                {
                    "rolled_back": True,
                    "rollback_reasons": list(best_rejected.critique.reasons),
                    "anchor_action": "replace",
                    "previous_anchors": list(previous_anchors),
                    "next_anchors": sorted(anchor_indices),
                    "replacement_source": (
                        "critic_recommendation"
                        if best_rejected.critique.recommended_anchor == next_anchor
                        else "storyboard_candidate_pool"
                    ),
                }
            )

        best_unaccepted = (
            max(evaluations.values(), key=lambda value: (value.reward, value.critique.score))
            if evaluations
            else None
        )
        return SearchOutcome(
            selected=best_unaccepted,
            attempts=tuple(attempts),
            memory=self._memory(
                candidate_grid,
                evaluations,
                rounds,
                best_unaccepted,
                best_unaccepted is not None,
                hero_reference,
            ),
        )

    def _memory(
        self,
        candidate_grid: dict[int, tuple[AnchorGrade, ...]],
        evaluations: dict[tuple[tuple[int, ...], tuple[int, ...]], SearchEvaluation],
        rounds: list[dict[str, object]],
        selected: Optional[SearchEvaluation],
        rolled_back: bool,
        hero_reference: Optional[HeroAnchorReference],
    ) -> dict[str, object]:
        proposals = {
            str(frame_index): [
                {
                    "editor": grade.backend,
                    "valid": grade.valid,
                    "score": grade.score,
                    "parameters": grade.parameters.to_dict(),
                    "metadata": grade.metadata,
                }
                for grade in grades
            ]
            for frame_index, grades in sorted(candidate_grid.items())
        }
        return {
            "algorithm": self.name,
            "editor_agents": [backend.name for backend in self.backends],
            "critic": self.critic.name,
            "hero_anchor_frame": (
                None if hero_reference is None else hero_reference.frame_index
            ),
            "exploration_constant": self.exploration_constant,
            "maximum_evaluations": self.maximum_evaluations,
            "rounds": rounds,
            "proposals": proposals,
            "evaluated_trajectories": len(evaluations),
            "selected": (
                None
                if selected is None
                else {
                    "anchors": list(selected.anchor_indices),
                    "choices": list(selected.choices),
                    "score": selected.critique.score,
                    "temporal_stabilization": selected.diffused.stabilization,
                }
            ),
            "rolled_back": rolled_back,
        }
