"""Convert a video plus text instruction into an editable grade graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_retouch import (
    DynamicGradePipeline,
    HeuristicShotPlanner,
    VLShotPlanner,
)
from video_retouch.agent_config import load_multi_agent_runtime
from video_retouch.io import decode_video, encode_video
from video_retouch.render import render_grade_frames
from video_retouch.resolve_export import export_resolve_package
from video_retouch.unified_backend import VideoEditRequest, load_unified_backend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input video.")
    parser.add_argument(
        "--reference-video",
        type=Path,
        help=(
            "Optional look-reference video. Its graded HeroShot becomes the "
            "visual reference for all target-video Anchors."
        ),
    )
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Grade JSON.")
    runtime = parser.add_mutually_exclusive_group(required=True)
    runtime.add_argument(
        "--backend-config",
        type=Path,
        help="Single UnifiedVLVideoBackend JSON configuration.",
    )
    runtime.add_argument(
        "--agent-config",
        type=Path,
        help="Legacy PhotoAgent-style multi-model JSON configuration.",
    )
    runtime.add_argument(
        "--offline-native",
        action="store_true",
        help=(
            "Run the fully local training-free baseline with physical shot "
            "planning, the native parameter editor, and deterministic critics."
        ),
    )
    parser.add_argument(
        "--allow-storyboard-fallback",
        action="store_true",
        help="Allow physical shot planning if the storyboard model fails.",
    )
    parser.add_argument("--anchors-per-shot", type=int, default=1)
    parser.add_argument("--max-anchors-per-shot", type=int, default=3)
    parser.add_argument(
        "--max-evaluations",
        "--mcts-simulations",
        "--max-attempts",
        dest="mcts_simulations",
        type=int,
        help="Maximum evaluated Anchor trajectories per shot.",
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        help="Optional JSONL rollout export for memory/distillation.",
    )
    parser.add_argument(
        "--video-output-dir",
        type=Path,
        help=(
            "Optionally export the processed input and final graded video as "
            "silent MP4 files."
        ),
    )
    parser.add_argument(
        "--resolve-package-dir",
        type=Path,
        help=(
            "Export static LUTs, dynamic DCTL trajectories, an apply script, "
            "and a conform manifest."
        ),
    )
    parser.add_argument(
        "--resolve-keyframe-error",
        type=float,
        default=0.015,
        help="Maximum per-parameter error when compressing Resolve DCTL curves.",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--max-side",
        type=int,
        help="Legacy shortcut setting both analysis and render maximum side.",
    )
    parser.add_argument(
        "--analysis-max-side",
        type=int,
        help="Use a smaller proxy for Agent planning while retaining frame indices.",
    )
    parser.add_argument(
        "--render-max-side",
        type=int,
        help="Render video artifacts from a separate decode at this maximum side.",
    )
    parser.add_argument(
        "--render-batch-size",
        type=int,
        default=8,
        help="Number of full-resolution frames rendered together.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Omit the dense per-frame parameter trajectory.",
    )
    args = parser.parse_args()

    analysis_max_side = (
        args.analysis_max_side if args.analysis_max_side is not None else args.max_side
    )
    render_max_side = (
        args.render_max_side if args.render_max_side is not None else args.max_side
    )
    decoded = decode_video(
        args.input,
        max_frames=args.max_frames,
        max_side=analysis_max_side,
    )
    reference_decoded = (
        decode_video(
            args.reference_video,
            max_frames=args.max_frames,
            max_side=analysis_max_side,
        )
        if args.reference_video is not None
        else None
    )
    unified_result = None
    if args.backend_config is not None:
        backend = load_unified_backend(
            args.backend_config,
            allow_storyboard_fallback=args.allow_storyboard_fallback,
            anchors_per_shot=args.anchors_per_shot,
            maximum_anchors_per_shot=args.max_anchors_per_shot,
            maximum_evaluations=args.mcts_simulations,
        )
        unified_result = backend.process(
            VideoEditRequest(
                frames=decoded.frames,
                fps=decoded.fps,
                instruction=args.instruction,
                reference_frames=(
                    None if reference_decoded is None else reference_decoded.frames
                ),
                reference_fps=(
                    None if reference_decoded is None else reference_decoded.fps
                ),
            )
        )
        result = unified_result.grade_graph
        render_executor = backend.executor
        operation_executor = backend.operation_executor
    elif args.offline_native:
        planner = HeuristicShotPlanner()
        maximum_attempts = (
            args.mcts_simulations if args.mcts_simulations is not None else 3
        )
        pipeline = DynamicGradePipeline(
            shot_planner=planner,
            anchors_per_shot=args.anchors_per_shot,
            maximum_anchors_per_shot=args.max_anchors_per_shot,
            maximum_attempts=maximum_attempts,
        )
        runtime_manifest = {
            "mode": "offline-native-training-free",
            "storyboard": {
                "provider": "deterministic",
                "model": "heuristic-shot-planner",
            },
            "editors": [
                {
                    "name": pipeline.anchor_backend.name,
                    "type": "native",
                    "model": "heuristic-parameter-search",
                }
            ],
            "evaluators": [
                {
                    "name": pipeline.critic.name,
                    "weight": 1.0,
                    "veto": True,
                    "model": "deterministic-metrics",
                }
            ],
            "search": {
                "algorithm": "uct-mcts",
                "maximum_evaluations": maximum_attempts,
            },
        }
        result = pipeline.run(
            decoded.frames,
            decoded.fps,
            args.instruction,
            reference_frames=(
                None if reference_decoded is None else reference_decoded.frames
            ),
            reference_fps=(
                None if reference_decoded is None else reference_decoded.fps
            ),
        )
        render_executor = pipeline.executor
        operation_executor = None
    else:
        configured_runtime = load_multi_agent_runtime(args.agent_config)
        planner = VLShotPlanner(
            client=configured_runtime.storyboard_client,
            settings=configured_runtime.storyboard_settings,
            strict=not args.allow_storyboard_fallback,
        )
        pipeline = DynamicGradePipeline(
            shot_planner=planner,
            anchor_backends=configured_runtime.anchor_backends,
            critic=configured_runtime.critic,
            anchors_per_shot=args.anchors_per_shot,
            maximum_anchors_per_shot=args.max_anchors_per_shot,
            maximum_attempts=(
                args.mcts_simulations
                if args.mcts_simulations is not None
                else configured_runtime.search.maximum_evaluations
            ),
            maximum_hero_attempts=configured_runtime.search.maximum_hero_attempts,
            mcts_exploration=configured_runtime.search.exploration_constant,
            mcts_rejection_penalty=configured_runtime.search.rejection_penalty,
            mcts_seed=configured_runtime.search.seed,
        )
        runtime_manifest = configured_runtime.manifest
        result = pipeline.run(
            decoded.frames,
            decoded.fps,
            args.instruction,
            reference_frames=(
                None if reference_decoded is None else reference_decoded.frames
            ),
            reference_fps=(
                None if reference_decoded is None else reference_decoded.fps
            ),
        )
        render_executor = pipeline.executor
        operation_executor = None
    if unified_result is not None:
        payload = unified_result.to_dict(include_frame_parameters=not args.compact)
    else:
        payload = result.to_dict(include_frame_parameters=not args.compact)
        payload["agent_runtime"] = runtime_manifest
    payload["source_video"] = {
        "path": str(decoded.source),
        "width": decoded.width,
        "height": decoded.height,
        "fps": decoded.fps,
        "frame_count": len(decoded.frames),
    }
    if reference_decoded is not None:
        payload["reference_video"] = {
            "path": str(reference_decoded.source),
            "width": reference_decoded.width,
            "height": reference_decoded.height,
            "fps": reference_decoded.fps,
            "frame_count": len(reference_decoded.frames),
            "role": "external_hero_source",
        }
    unsupported_resolve_operations = (
        []
        if unified_result is None
        else [
            operation.operation_type
            for operation in unified_result.operations
            if operation.operation_type != "global_grade"
        ]
    )
    if args.resolve_package_dir is not None and unsupported_resolve_operations:
        raise ValueError(
            "Resolve export cannot yet encode unified post-grade operations: "
            + ", ".join(sorted(set(unsupported_resolve_operations)))
        )
    if args.video_output_dir is not None:
        video_output_dir = args.video_output_dir.resolve()
        source_video_output = video_output_dir / f"{args.input.stem}.source.mp4"
        result_video_output = video_output_dir / f"{args.input.stem}.graded.mp4"
        if decoded.source in {
            source_video_output.resolve(),
            result_video_output.resolve(),
        }:
            raise ValueError("Video artifact path would overwrite the source video.")
        render_video = (
            decoded
            if render_max_side == analysis_max_side
            else decode_video(
                args.input,
                max_frames=args.max_frames,
                max_side=render_max_side,
            )
        )
        if len(render_video.frames) != len(decoded.frames):
            raise RuntimeError(
                "Analysis and render decodes have different frame counts."
            )
        rendered_frames = render_grade_frames(
            render_video.frames,
            result.frame_parameters,
            executor=render_executor,
            operations=(() if unified_result is None else unified_result.operations),
            operation_executor=operation_executor,
            batch_size=args.render_batch_size,
        )
        encode_video(render_video.frames, source_video_output, render_video.fps)
        encode_video(rendered_frames, result_video_output, render_video.fps)
        payload["video_artifacts"] = {
            "input": str(source_video_output),
            "result": str(result_video_output),
            "fps": render_video.fps,
            "frame_count": len(render_video.frames),
            "width": render_video.width,
            "height": render_video.height,
            "analysis_width": decoded.width,
            "analysis_height": decoded.height,
            "video_codec": "H.264/libx264",
            "quality": "CRF approximately 15",
            "audio_preserved": False,
        }
    if args.resolve_package_dir is not None:
        resolve_manifest = export_resolve_package(
            result,
            decoded.source,
            args.resolve_package_dir,
            dynamic_keyframe_error=args.resolve_keyframe_error,
        )
        payload["resolve_package"] = str(resolve_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.trajectory_output is not None:
        rows = []
        for shot in result.shots:
            rows.append(
                {
                    "source_video": str(decoded.source),
                    "instruction": args.instruction,
                    "shot": shot.shot.to_dict(),
                    "accepted": shot.accepted,
                    "rolled_back": shot.rolled_back,
                    "rollback_reason": shot.rollback_reason,
                    "attempts": [attempt.to_dict() for attempt in shot.attempts],
                    "selected_parameters": {
                        str(index): values.tolist()
                        for index, values in sorted(shot.parameter_keyframes.items())
                    },
                    "search_memory": shot.search_memory,
                }
            )
        args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        args.trajectory_output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    print(args.output)
    if args.video_output_dir is not None:
        print(payload["video_artifacts"]["input"])
        print(payload["video_artifacts"]["result"])
    if args.resolve_package_dir is not None:
        print(payload["resolve_package"])
    if args.trajectory_output is not None:
        print(args.trajectory_output)


if __name__ == "__main__":
    main()
