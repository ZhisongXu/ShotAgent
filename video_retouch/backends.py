"""Adapters from single-image retouch agents to editable grade parameters."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np
import torch
from PIL import Image

from retouch_agent import AnchorRetouchAgent, RetouchExecutor, RetouchParameters
from retouch_agent.parameters import PARAMETER_LOWER_BOUNDS, PARAMETER_UPPER_BOUNDS
from retouch_agent.planner import HeuristicRetouchPlanner, RetouchPlan

from .clients import VisionLanguageClient
from .color_science import LinearMongeKantorovichMatcher
from .monet_adapter import convert_monet_adjustments
from .tasks import anchor_grade_prompt, anchor_match_prompt


@dataclass(frozen=True)
class AnchorGrade:
    frame_index: int
    parameters: RetouchParameters
    preview: Image.Image
    valid: bool
    score: float
    backend: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class HeroAnchorReference:
    frame_index: int
    shot_id: int
    source: Image.Image
    grade: AnchorGrade


class AnchorRetouchBackend(Protocol):
    name: str

    def grade(
        self,
        image: Image.Image,
        instruction: str,
        frame_index: int,
        shot_id: int,
    ) -> AnchorGrade: ...


class NativeRetouchBackend:
    """In-repository backend used for tests and as a procedural baseline."""

    name = "native-anchor-agent"

    def __init__(
        self,
        agent: Optional[AnchorRetouchAgent] = None,
        name: str = "native-anchor-agent",
    ) -> None:
        self.agent = agent or AnchorRetouchAgent()
        self.name = name

    def grade(
        self,
        image: Image.Image,
        instruction: str,
        frame_index: int,
        shot_id: int,
    ) -> AnchorGrade:
        result = self.agent.run(image, instruction)
        return AnchorGrade(
            frame_index=frame_index,
            parameters=result.parameters,
            preview=result.image,
            valid=result.evaluation.valid and not result.rolled_back,
            score=result.evaluation.score,
            backend=self.name,
            metadata={"shot_id": shot_id, **result.to_dict()},
        )

    def grade_with_reference(
        self,
        image: Image.Image,
        instruction: str,
        frame_index: int,
        shot_id: int,
        hero_reference: HeroAnchorReference,
    ) -> AnchorGrade:
        result = self.agent.run(
            image,
            instruction,
            reference=hero_reference.grade.preview,
        )
        return AnchorGrade(
            frame_index=frame_index,
            parameters=result.parameters,
            preview=result.image,
            valid=result.evaluation.valid and not result.rolled_back,
            score=result.evaluation.score,
            backend=self.name,
            metadata={
                "shot_id": shot_id,
                "matched_to_hero_frame": hero_reference.frame_index,
                "matched_to_hero_shot": hero_reference.shot_id,
                **result.to_dict(),
            },
        )


class _FixedRetouchPlanner:
    def __init__(self, plan: RetouchPlan) -> None:
        self._plan = plan

    def plan(self, image, instruction, reference=None, has_local_mask=False):
        return self._plan


class VLAnchorBackend:
    """Use a dedicated vision model for operation-aware Anchor grading."""

    name = "vl-anchor-agent"

    def __init__(
        self,
        client: VisionLanguageClient,
        stages: Sequence[str] = ("lighting", "white_balance_and_color", "tone"),
        candidate_count: int = 16,
        seed: int = 7,
        name: str = "vl-anchor-agent",
        use_mkl_prior: bool = True,
        mkl_strength: float = 0.35,
        mkl_projection_iterations: int = 40,
    ) -> None:
        if not stages:
            raise ValueError("VL Anchor grading requires at least one stage.")
        self.client = client
        self.stages = tuple(stages)
        self.candidate_count = int(candidate_count)
        self.seed = int(seed)
        self.name = name
        self.executor = RetouchExecutor()
        self.heuristic = HeuristicRetouchPlanner()
        self.use_mkl_prior = bool(use_mkl_prior)
        self.mkl_matcher = LinearMongeKantorovichMatcher(strength=mkl_strength)
        self.mkl_projection_iterations = int(mkl_projection_iterations)

    def grade(
        self,
        image: Image.Image,
        instruction: str,
        frame_index: int,
        shot_id: int,
    ) -> AnchorGrade:
        return self._grade(
            image,
            instruction,
            frame_index,
            shot_id,
            hero_reference=None,
        )

    def grade_with_reference(
        self,
        image: Image.Image,
        instruction: str,
        frame_index: int,
        shot_id: int,
        hero_reference: HeroAnchorReference,
    ) -> AnchorGrade:
        return self._grade(
            image,
            instruction,
            frame_index,
            shot_id,
            hero_reference=hero_reference,
        )

    def _grade(
        self,
        image: Image.Image,
        instruction: str,
        frame_index: int,
        shot_id: int,
        hero_reference: Optional[HeroAnchorReference],
    ) -> AnchorGrade:
        source = image.convert("RGB")
        preview = source
        parameters = RetouchParameters()
        mkl_preview: Optional[Image.Image] = None
        mkl_parameters: Optional[RetouchParameters] = None
        mkl_metadata: dict[str, object] = {}
        mkl_applied = False
        if hero_reference is not None and self.use_mkl_prior:
            mkl_preview, mkl_metadata = self.mkl_matcher.transfer(
                source,
                hero_reference.grade.preview,
            )
            mkl_parameters, reconstruction_error = ParameterEstimator(
                iterations=self.mkl_projection_iterations,
                max_side=192,
            ).fit(source, mkl_preview)
            # The color transfer remains a proposal until the VL role judges
            # semantic correspondence (skin-to-skin, sky-to-sky, etc.).
            mkl_metadata = {
                **mkl_metadata,
                "projected_parameters": mkl_parameters.to_dict(),
                "parameter_projection_error": reconstruction_error,
            }
        stage_records: list[dict[str, object]] = []
        constraints: list[str] = []
        confidences: list[float] = []
        for stage in self.stages:
            if hero_reference is None:
                labeled_images = [
                    ("source Anchor", source),
                    (f"current preview before {stage}", preview),
                ]
                prompt = anchor_grade_prompt(
                    instruction, stage, parameters.to_dict()
                )
            else:
                labeled_images = [
                    (
                        f"HeroAnchor source frame {hero_reference.frame_index}",
                        hero_reference.source,
                    ),
                    (
                        f"accepted graded HeroAnchor frame "
                        f"{hero_reference.frame_index}",
                        hero_reference.grade.preview,
                    ),
                    (f"target shot {shot_id} Anchor source", source),
                    (f"target shot {shot_id} preview before {stage}", preview),
                ]
                if mkl_preview is not None:
                    labeled_images.append(
                        (
                            "distribution-only MKL proposal for the target Anchor",
                            mkl_preview,
                        )
                    )
                prompt = anchor_match_prompt(
                    instruction,
                    stage,
                    parameters.to_dict(),
                    hero_reference.frame_index,
                    hero_reference.shot_id,
                    mkl_metadata if mkl_preview is not None else None,
                )
            payload = self.client.generate_json(
                labeled_images,
                prompt,
            )
            raw_updates = payload.get("parameter_updates", {})
            if not isinstance(raw_updates, dict):
                raise ValueError("Anchor Agent response requires parameter_updates.")
            merged = parameters.to_dict()
            mkl_decision = "not_available"
            mkl_weight = 0.0
            if (
                hero_reference is not None
                and mkl_parameters is not None
                and not mkl_applied
            ):
                mkl_decision = str(payload.get("mkl_decision", "attenuate")).lower()
                requested_weight = float(payload.get("mkl_weight", 0.5))
                if mkl_decision == "accept":
                    mkl_weight = 1.0
                elif mkl_decision == "attenuate":
                    mkl_weight = float(np.clip(requested_weight, 0.0, 1.0))
                elif mkl_decision == "reject":
                    mkl_weight = 0.0
                else:
                    raise ValueError(
                        "mkl_decision must be accept, attenuate, or reject."
                    )
                prior_values = mkl_parameters.to_dict()
                for name in merged:
                    merged[name] += mkl_weight * prior_values[name]
                mkl_applied = True
            for name, value in raw_updates.items():
                if name not in merged:
                    raise ValueError(f"Anchor Agent returned unknown parameter: {name}")
                merged[name] += float(value)
            parameters = RetouchParameters.from_mapping(merged, clamp=True)
            preview = self.executor.apply(source, parameters)
            raw_constraints = payload.get("constraints", [])
            if isinstance(raw_constraints, list):
                constraints.extend(str(value) for value in raw_constraints)
            confidence = float(payload.get("confidence", 0.0))
            confidences.append(min(max(confidence, 0.0), 1.0))
            stage_records.append(
                {
                    "stage": stage,
                    "diagnosis": payload.get("diagnosis", {}),
                    "parameter_updates": {
                        name: float(value) for name, value in raw_updates.items()
                    },
                    "parameters_after_stage": parameters.to_dict(),
                    "confidence": confidences[-1],
                    "semantic_correspondences": payload.get(
                        "semantic_correspondences", []
                    ),
                    "protected_regions": payload.get("protected_regions", []),
                    "mkl_decision": mkl_decision,
                    "mkl_weight": mkl_weight,
                }
            )

        heuristic_plan = self.heuristic.plan(source, instruction)
        plan = RetouchPlan(
            diagnosis={
                "planner": "dedicated-vl-anchor-agent",
                "model_id": self.client.model_id,
                "stages": stage_records,
            },
            initial_parameters=parameters,
            targets=heuristic_plan.targets,
            constraints=tuple(
                dict.fromkeys([*heuristic_plan.constraints, *constraints])
            ),
        )
        agent = AnchorRetouchAgent(
            planner=_FixedRetouchPlanner(plan),
            candidate_count=self.candidate_count,
            seed=self.seed,
        )
        result = agent.run(source, instruction)
        return AnchorGrade(
            frame_index=frame_index,
            parameters=result.parameters,
            preview=result.image,
            valid=result.evaluation.valid and not result.rolled_back,
            score=result.evaluation.score,
            backend=self.name,
            metadata={
                "shot_id": shot_id,
                "mean_model_confidence": float(np.mean(confidences)),
                "matched_to_hero_frame": (
                    None if hero_reference is None else hero_reference.frame_index
                ),
                "matched_to_hero_shot": (
                    None if hero_reference is None else hero_reference.shot_id
                ),
                "mkl_prior": (
                    None
                    if hero_reference is None or not self.use_mkl_prior
                    else {
                        **mkl_metadata,
                        "semantic_decision": next(
                            (
                                {
                                    "decision": record["mkl_decision"],
                                    "weight": record["mkl_weight"],
                                    "correspondences": record[
                                        "semantic_correspondences"
                                    ],
                                    "protected_regions": record[
                                        "protected_regions"
                                    ],
                                }
                                for record in stage_records
                                if record["mkl_decision"] != "not_available"
                            ),
                            None,
                        ),
                    }
                ),
                **result.to_dict(),
            },
        )


class ParameterEstimator:
    """Recover the shared 12-D parameter schema from a before/after pair.

    MonetGPT yields a rendered image while JarvisArt can yield Lightroom
    settings. For image-only backends this differentiable inverse step converts
    their result into the same editable parameter space used by video diffusion.
    """

    def __init__(
        self,
        iterations: int = 120,
        learning_rate: float = 0.06,
        max_side: int = 256,
    ) -> None:
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.max_side = int(max_side)
        self.executor = RetouchExecutor()

    def fit(
        self, source: Image.Image, target: Image.Image
    ) -> tuple[RetouchParameters, float]:
        source = source.convert("RGB")
        target = target.convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
        scale = min(1.0, self.max_side / max(source.size))
        size = (
            max(1, round(source.width * scale)),
            max(1, round(source.height * scale)),
        )
        source_array = (
            np.asarray(source.resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
            / 255.0
        )
        target_array = (
            np.asarray(target.resize(size, Image.Resampling.LANCZOS), dtype=np.float32)
            / 255.0
        )
        source_tensor = torch.from_numpy(source_array).permute(2, 0, 1)
        target_tensor = torch.from_numpy(target_array).permute(2, 0, 1)
        values = torch.zeros(12, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([values], lr=self.learning_rate)
        lower = values.new_tensor(PARAMETER_LOWER_BOUNDS)
        upper = values.new_tensor(PARAMETER_UPPER_BOUNDS)
        # External image-only agents cannot identify local parameters without a
        # mask, so those dimensions stay zero.
        for _ in range(self.iterations):
            optimizer.zero_grad()
            rendered = self.executor.apply_vector(source_tensor, values)
            reconstruction = torch.mean(torch.abs(rendered - target_tensor))
            regularization = 2e-3 * torch.mean(values[:9].square())
            loss = reconstruction + regularization
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                values.clamp_(lower, upper)
                values[9:] = 0.0
        with torch.no_grad():
            rendered = self.executor.apply_vector(source_tensor, values)
            error = float(torch.mean(torch.abs(rendered - target_tensor)))
        return RetouchParameters.from_vector(values.detach().cpu().numpy()), error


class CommandRetouchBackend:
    """Strict process adapter for a single-image agent.

    Tokens may contain ``{input}``, ``{output}``, ``{instruction}``,
    ``{frame_index}``, and ``{shot_id}``. The command must create the output
    image. No shell is used, which keeps paths and user text from becoming code.
    """

    def __init__(
        self,
        command: str | Sequence[str],
        name: str = "external-command-agent",
        cwd: Optional[Path] = None,
        timeout_seconds: float = 600.0,
        estimator: Optional[ParameterEstimator] = None,
        maximum_reconstruction_error: float = 0.18,
    ) -> None:
        self.command = (
            shlex.split(command) if isinstance(command, str) else list(command)
        )
        if not self.command:
            raise ValueError("External backend command cannot be empty.")
        self.name = name
        self.cwd = None if cwd is None else Path(cwd)
        self.timeout_seconds = float(timeout_seconds)
        self.estimator = estimator or ParameterEstimator()
        self.maximum_reconstruction_error = float(maximum_reconstruction_error)

    def grade(
        self,
        image: Image.Image,
        instruction: str,
        frame_index: int,
        shot_id: int,
    ) -> AnchorGrade:
        with tempfile.TemporaryDirectory(prefix="dynamic-grade-anchor-") as directory:
            work = Path(directory)
            input_path = work / "input.png"
            output_path = work / "output.png"
            image.convert("RGB").save(input_path)
            substitutions = {
                "input": str(input_path),
                "output": str(output_path),
                "instruction": instruction,
                "frame_index": str(frame_index),
                "shot_id": str(shot_id),
            }
            command = [token.format(**substitutions) for token in self.command]
            completed = subprocess.run(
                command,
                cwd=self.cwd,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            if not output_path.is_file():
                raise RuntimeError(
                    f"{self.name} completed without creating {output_path}."
                )
            with Image.open(output_path) as loaded:
                preview = loaded.convert("RGB").copy()
            parameters, error = self.estimator.fit(image, preview)
            return AnchorGrade(
                frame_index=frame_index,
                parameters=parameters,
                preview=preview,
                valid=error <= self.maximum_reconstruction_error,
                score=-error,
                backend=self.name,
                metadata={
                    "shot_id": shot_id,
                    "reconstruction_l1": error,
                    "stdout": completed.stdout[-2000:],
                    "stderr": completed.stderr[-2000:],
                },
            )


class MonetRetouchBackend(CommandRetouchBackend):
    """Adapter for the official MonetGPT ``inference_cli.py single`` command."""

    def __init__(
        self,
        repository: Path,
        python_executable: str = sys.executable,
        name: str = "monetgpt",
        **kwargs: object,
    ) -> None:
        repository = Path(repository).resolve()
        entrypoint = repository / "inference_cli.py"
        if not entrypoint.is_file():
            raise FileNotFoundError(f"MonetGPT entrypoint not found: {entrypoint}")
        super().__init__(
            [
                python_executable,
                str(entrypoint),
                "single",
                "{input}",
                "--output",
                "{output}",
            ],
            name=name,
            cwd=repository,
            **kwargs,
        )


class MonetParameterBackend:
    """Run MonetGPT, consume its JSON, and submit parameters to MCTS/rollback.

    Unlike :class:`MonetRetouchBackend`, this path does not use MonetGPT's
    GIMP/NumPy render or recover parameters from pixels. It renders a preview
    with the shared deterministic executor so the existing critics evaluate
    exactly the transform that will later be exported to Resolve.
    """

    def __init__(
        self,
        repository: Path,
        python_executable: str = sys.executable,
        name: str = "monetgpt-parameters",
        style: str = "balanced",
        timeout_seconds: float = 600.0,
        reject_unsupported: bool = True,
    ) -> None:
        repository = Path(repository).resolve()
        entrypoint = repository / "inference_cli.py"
        if not entrypoint.is_file():
            raise FileNotFoundError(f"MonetGPT entrypoint not found: {entrypoint}")
        if style not in {"balanced", "vibrant", "retro"}:
            raise ValueError("MonetGPT style must be balanced, vibrant, or retro.")
        self.repository = repository
        self.entrypoint = entrypoint
        self.python_executable = str(python_executable)
        self.name = name
        self.style = style
        self.timeout_seconds = float(timeout_seconds)
        self.reject_unsupported = bool(reject_unsupported)
        self.executor = RetouchExecutor()

    def grade(
        self,
        image: Image.Image,
        instruction: str,
        frame_index: int,
        shot_id: int,
    ) -> AnchorGrade:
        # MonetGPT's official CLI currently exposes a style, but not arbitrary
        # edit text. Preserve the requested instruction in the audit so this
        # limitation remains visible to the critics and caller.
        with tempfile.TemporaryDirectory(
            prefix="dynamic-grade-monet-parameters-"
        ) as directory:
            work = Path(directory)
            input_path = work / "input.png"
            output_base = work / "monet"
            parameter_path = output_base.with_suffix(".json")
            image.convert("RGB").save(input_path)
            completed = subprocess.run(
                [
                    self.python_executable,
                    str(self.entrypoint),
                    "single",
                    str(input_path),
                    "--output",
                    str(output_base),
                    "--style",
                    self.style,
                ],
                cwd=self.repository,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            if not parameter_path.is_file():
                raise RuntimeError(
                    "MonetGPT completed without creating final parameter JSON: "
                    f"{parameter_path}"
                )
            payload = json.loads(parameter_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("MonetGPT final parameter JSON must be an object.")
            conversion = convert_monet_adjustments(payload, strict=False)
            unsupported = conversion.unsupported_fields
            valid = not (self.reject_unsupported and unsupported)
            preview = self.executor.apply(image.convert("RGB"), conversion.parameters)
            assert isinstance(preview, Image.Image)
            return AnchorGrade(
                frame_index=frame_index,
                parameters=conversion.parameters,
                preview=preview,
                valid=valid,
                score=0.0 if valid else -1e6,
                backend=self.name,
                metadata={
                    "shot_id": shot_id,
                    "instruction": instruction,
                    "style": self.style,
                    "parameter_source": "monetgpt-final-json",
                    "resolve_transform_preview": True,
                    "conversion": conversion.to_dict(),
                    "rollback_eligible": True,
                    "invalid_reason": (
                        None
                        if valid
                        else "unsupported_monet_adjustments"
                    ),
                    "stdout": completed.stdout[-2000:],
                    "stderr": completed.stderr[-2000:],
                },
            )


def load_parameter_file(path: Path) -> RetouchParameters:
    """Load a backend-produced parameter JSON in the shared schema."""

    payload = json.loads(Path(path).read_text())
    values = payload.get("parameters", payload)
    if not isinstance(values, dict):
        raise ValueError("Parameter JSON must contain an object.")
    return RetouchParameters.from_mapping(values, clamp=True)
