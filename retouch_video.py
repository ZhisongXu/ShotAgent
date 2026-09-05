"""Convert a video plus text instruction into an editable grade graph."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from video_retouch import (
    DynamicGradePipeline,
    HeuristicShotPlanner,
    VLShotPlanner,
)
from video_retouch.agent_config import load_multi_agent_runtime
from video_retouch.color_science import LumaPreservingChromaMatcher
from video_retouch.io import decode_video, encode_video
from video_retouch.render import render_grade_frames
from video_retouch.resolve_export import export_resolve_package


def progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def resample_parameter_trajectory(
    parameters,
    target_frame_count: int,
):
    import numpy as np

    source = np.asarray(parameters, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 12:
        raise ValueError("Parameter trajectory must have shape [frames, 12].")
    if target_frame_count < 1:
        raise ValueError("target_frame_count must be positive.")
    if source.shape[0] == target_frame_count:
        return source
    if source.shape[0] == 1:
        return np.repeat(source, target_frame_count, axis=0)
    source_t = np.linspace(0.0, 1.0, source.shape[0], dtype=np.float64)
    target_t = np.linspace(0.0, 1.0, target_frame_count, dtype=np.float64)
    return np.stack(
        [np.interp(target_t, source_t, source[:, column]) for column in range(12)],
        axis=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input video.")
    parser.add_argument(
        "--reference-video",
        type=Path,
        help="Optional graded reference video used by every editor in the pool.",
    )
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Grade JSON.")
    runtime = parser.add_mutually_exclusive_group(required=True)
    runtime.add_argument(
        "--agent-config",
        type=Path,
        help="PhotoAgent-style multi-model JSON configuration.",
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
        "--mcts-simulations",
        "--max-attempts",
        dest="mcts_simulations",
        type=int,
        help="Maximum evaluated MCTS trajectories per shot.",
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
        "--target-fps",
        type=float,
        help=(
            "Legacy shortcut for --analysis-fps. Decode a lower-frame-rate "
            "working copy while preserving duration."
        ),
    )
    parser.add_argument(
        "--analysis-fps",
        type=float,
        help=(
            "Decode a lower-frame-rate working copy for API planning. "
            "For example, 6 on a 24 FPS source analyzes roughly one frame in four."
        ),
    )
    parser.add_argument(
        "--render-fps",
        type=float,
        help=(
            "Decode/render video artifacts at this FPS. Omit to preserve the "
            "source video's original frame rate."
        ),
    )
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
        "--encode-preset",
        default="medium",
        help="libx264 preset for MP4 artifacts; use ultrafast for quick previews.",
    )
    parser.add_argument(
        "--encode-quality",
        type=float,
        default=7.0,
        help="imageio-ffmpeg quality from 0 to 10; use 10 for metric evaluation.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Omit the dense per-frame parameter trajectory.",
    )
    args = parser.parse_args()
    load_env_file(Path(__file__).resolve().parent / ".env")

    analysis_max_side = (
        args.analysis_max_side if args.analysis_max_side is not None else args.max_side
    )
    render_max_side = (
        args.render_max_side if args.render_max_side is not None else args.max_side
    )
    analysis_fps = (
        args.analysis_fps if args.analysis_fps is not None else args.target_fps
    )
    progress("decoding analysis video")
    decoded = decode_video(
        args.input,
        max_frames=args.max_frames,
        max_side=analysis_max_side,
        target_fps=analysis_fps,
    )
    progress(
        "decoded analysis video "
        f"frames={len(decoded.frames)} fps={decoded.fps:.3f} "
        f"size={decoded.width}x{decoded.height}"
    )
    if args.offline_native:
        progress("building offline-native runtime")
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
    else:
        progress(f"loading agent config {args.agent_config}")
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
    reference_decoded = None
    if args.reference_video is not None:
        progress(f"decoding reference video {args.reference_video}")
        reference_decoded = decode_video(
            args.reference_video,
            max_frames=min(args.max_frames or 96, 96),
            max_side=analysis_max_side,
            target_fps=1.0,
        )
        progress(
            "reference video ready "
            f"frames={len(reference_decoded.frames)} "
            f"size={reference_decoded.width}x{reference_decoded.height}"
        )
    progress("running storyboard, grading search, and API critics")
    result = pipeline.run(
        decoded.frames,
        decoded.fps,
        args.instruction,
        reference_frames=(
            None if reference_decoded is None else reference_decoded.frames
        ),
    )
    progress("pipeline run finished")
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
            "pool_conditioning": "ordered 8-frame storyboard",
        }
    if args.video_output_dir is not None:
        progress("preparing video artifacts")
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
            if render_max_side == analysis_max_side and args.render_fps == analysis_fps
            else decode_video(
                args.input,
                max_frames=args.max_frames,
                max_side=render_max_side,
                target_fps=args.render_fps,
            )
        )
        progress(
            "render source ready "
            f"frames={len(render_video.frames)} fps={render_video.fps:.3f} "
            f"size={render_video.width}x{render_video.height}"
        )
        render_parameters = resample_parameter_trajectory(
            result.frame_parameters,
            len(render_video.frames),
        )
        if len(render_video.frames) != len(decoded.frames):
            progress(
                "resampled parameter trajectory "
                f"analysis_frames={len(decoded.frames)} "
                f"render_frames={len(render_video.frames)}"
            )
        progress("rendering graded frames")
        rendered_frames = tuple(
            render_grade_frames(
                render_video.frames,
                render_parameters,
                executor=pipeline.executor,
                batch_size=args.render_batch_size,
            )
        )
        refinement = runtime_manifest.get("reference_chroma_refinement", {})
        if (
            reference_decoded is not None
            and isinstance(refinement, dict)
            and bool(refinement.get("enabled", False))
        ):
            strength = float(refinement.get("strength", 0.6))
            progress(
                "applying pool-selected luma-preserving chroma affinity "
                f"strength={strength:.3f}"
            )
            rendered_frames, refinement_audit = LumaPreservingChromaMatcher(
                strength=strength,
                target_luma_strength=float(refinement.get("target_luma_strength", 0.0)),
            ).transfer_video(
                rendered_frames,
                reference_decoded.frames,
                target_luma_frames=render_video.frames,
            )
            payload["reference_chroma_refinement"] = refinement_audit
        progress(f"encoding source preview with preset={args.encode_preset}")
        encode_video(
            render_video.frames,
            source_video_output,
            render_video.fps,
            preset=args.encode_preset,
            quality=args.encode_quality,
        )
        progress(f"encoding graded preview with preset={args.encode_preset}")
        encode_video(
            rendered_frames,
            result_video_output,
            render_video.fps,
            preset=args.encode_preset,
            quality=args.encode_quality,
        )
        payload["video_artifacts"] = {
            "input": str(source_video_output),
            "result": str(result_video_output),
            "fps": render_video.fps,
            "frame_count": len(render_video.frames),
            "width": render_video.width,
            "height": render_video.height,
            "analysis_width": decoded.width,
            "analysis_height": decoded.height,
            "analysis_fps": decoded.fps,
            "analysis_frame_count": len(decoded.frames),
            "trajectory_resampled": len(render_video.frames) != len(decoded.frames),
            "video_codec": "H.264/libx264",
            "quality": args.encode_quality,
            "audio_preserved": False,
        }
    if args.resolve_package_dir is not None:
        progress("exporting Resolve package")
        resolve_manifest = export_resolve_package(
            result,
            decoded.source,
            args.resolve_package_dir,
            dynamic_keyframe_error=args.resolve_keyframe_error,
        )
        payload["resolve_package"] = str(resolve_manifest)
    progress(f"writing grade JSON {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.trajectory_output is not None:
        progress(f"writing trajectory JSONL {args.trajectory_output}")
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
    progress("done")
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
