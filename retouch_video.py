"""Convert a video plus text instruction into an editable grade graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from video_retouch import (
    DynamicGradePipeline,
    HeuristicShotPlanner,
    VLShotPlanner,
)
from video_retouch.agent_config import load_multi_agent_runtime
from video_retouch.color_management import COLOR_SPACES, ColorManager, OCIOColorManager
from video_retouch.color_managed_render import render_color_managed_frames
from video_retouch.high_bit_io import (
    decode_video_rgb16,
    encode_video_high_bit,
    has_audio_stream,
)
from video_retouch.io import DecodedVideo, decode_video, encode_video
from video_retouch.render import render_grade_frames
from video_retouch.resolve_export import export_resolve_package
from video_retouch.semantic_masks import SemanticMaskTracker
from video_retouch.unified_backend import VideoEditRequest, load_unified_backend


def _analysis_decode(
    path: Path,
    *,
    color_space: str,
    max_frames: int | None,
    max_side: int | None,
    working_space: str,
    ocio_manager: OCIOColorManager | None,
    ocio_spaces: dict[str, str],
    pq_reference_white_nits: float,
) -> DecodedVideo:
    if color_space == "srgb" and ocio_manager is None:
        return decode_video(path, max_frames=max_frames, max_side=max_side)
    info, arrays = decode_video_rgb16(path, max_frames=max_frames, max_side=max_side)
    if ocio_manager is None:
        manager = ColorManager(working_space)
        proxies = []
        for frame in arrays:
            working = manager.to_working(frame, color_space)
            if color_space == "rec2020_pq":
                working = working * (10_000.0 / pq_reference_white_nits)
            proxies.append(manager.display_proxy(working))
    else:
        proxies = [
            np.clip(
                ocio_manager.convert(
                    ocio_manager.convert(frame, ocio_spaces["input"], ocio_spaces["working"]),
                    ocio_spaces["working"],
                    ocio_spaces["display"],
                ),
                0,
                1,
            )
            for frame in arrays
        ]
    frames = tuple(
        Image.fromarray((proxy * 255 + 0.5).astype(np.uint8), mode="RGB")
        for proxy in proxies
    )
    return DecodedVideo(frames, info.fps, info.width, info.height, info.source)


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
        "--render-device",
        help="Torch device for batch Pool rendering, for example cuda, cuda:1, or cpu.",
    )
    parser.add_argument(
        "--disable-torch-pools",
        action="store_true",
        help="Use the NumPy/OpenCV Pool fallback for 8-bit output.",
    )
    parser.add_argument(
        "--mask-detection-interval",
        type=int,
        default=6,
        help="Refresh person detection every N frames between optical-flow tracks.",
    )
    parser.add_argument(
        "--input-color-space",
        choices=sorted(COLOR_SPACES),
        default="srgb",
        help="Transfer/gamut carried by the input video.",
    )
    parser.add_argument(
        "--reference-color-space",
        choices=sorted(COLOR_SPACES),
        help="Reference-video color space; defaults to --input-color-space.",
    )
    parser.add_argument(
        "--working-color-space",
        choices=("acescg", "aces2065-1", "linear_rec709", "linear_rec2020"),
        default="acescg",
    )
    parser.add_argument(
        "--output-color-space",
        choices=sorted(COLOR_SPACES),
        default="rec709",
    )
    parser.add_argument("--output-bit-depth", type=int, choices=(8, 10, 12), default=8)
    parser.add_argument(
        "--pq-reference-white-nits",
        type=float,
        default=203.0,
        help="Scene-linear value 1.0 luminance when writing Rec.2020 PQ.",
    )
    parser.add_argument("--ocio-config", type=Path, help="Optional OpenColorIO config.ocio.")
    parser.add_argument("--ocio-input-space")
    parser.add_argument("--ocio-working-space")
    parser.add_argument("--ocio-display-space")
    parser.add_argument("--ocio-output-space")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Omit the dense per-frame parameter trajectory.",
    )
    args = parser.parse_args()

    ocio_names = {
        "input": args.ocio_input_space,
        "working": args.ocio_working_space,
        "display": args.ocio_display_space,
        "output": args.ocio_output_space,
    }
    if args.ocio_config is not None and not all(ocio_names.values()):
        parser.error(
            "--ocio-config requires --ocio-input-space, --ocio-working-space, "
            "--ocio-display-space, and --ocio-output-space."
        )
    ocio_manager = (
        OCIOColorManager(args.ocio_config) if args.ocio_config is not None else None
    )

    analysis_max_side = (
        args.analysis_max_side if args.analysis_max_side is not None else args.max_side
    )
    render_max_side = (
        args.render_max_side if args.render_max_side is not None else args.max_side
    )
    decoded = _analysis_decode(
        args.input,
        color_space=args.input_color_space,
        max_frames=args.max_frames,
        max_side=analysis_max_side,
        working_space=args.working_color_space,
        ocio_manager=ocio_manager,
        ocio_spaces=ocio_names,
        pq_reference_white_nits=args.pq_reference_white_nits,
    )
    reference_decoded = (
        _analysis_decode(
            args.reference_video,
            color_space=args.reference_color_space or args.input_color_space,
            max_frames=args.max_frames,
            max_side=analysis_max_side,
            working_space=args.working_color_space,
            ocio_manager=ocio_manager,
            ocio_spaces=ocio_names,
            pq_reference_white_nits=args.pq_reference_white_nits,
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
        "input_color_space": args.input_color_space,
        "working_color_space": (
            ocio_names["working"] if ocio_manager is not None else args.working_color_space
        ),
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
        if (
            unified_result is not None
            and unified_result.pool_metadata.get("schema_version")
            == "pool-grade-graph/v2"
        ):
            raise ValueError(
                "Resolve export cannot faithfully encode pool-graph/v2 spatial, "
                "frequency, and temporal operations. Use --video-output-dir for "
                "the complete render, or use the legacy runtime for 12-D DCTL export."
            )
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
        operations = () if unified_result is None else unified_result.operations
        color_managed = (
            args.output_bit_depth > 8
            or args.input_color_space != "srgb"
            or args.output_color_space not in {"srgb", "rec709"}
            or ocio_manager is not None
        )
        if color_managed and args.disable_torch_pools:
            raise ValueError(
                "--disable-torch-pools is only available for the 8-bit sRGB/Rec.709 path."
            )
        pool_render_device = args.render_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        mask_tracker = SemanticMaskTracker(
            detection_interval=args.mask_detection_interval
        )
        if color_managed:
            render_info, render_arrays = decode_video_rgb16(
                args.input,
                max_frames=args.max_frames,
                max_side=render_max_side,
            )
            if len(render_arrays) != len(decoded.frames):
                raise RuntimeError("Analysis and render decodes have different frame counts.")
            rendered_arrays = render_color_managed_frames(
                render_arrays,
                result.frame_parameters,
                executor=render_executor,
                operations=operations,
                input_color_space=args.input_color_space,
                working_color_space=args.working_color_space,
                output_color_space=args.output_color_space,
                batch_size=args.render_batch_size,
                device=pool_render_device,
                mask_tracker=mask_tracker,
                pq_reference_white_nits=args.pq_reference_white_nits,
                ocio_manager=ocio_manager,
                ocio_spaces=ocio_names,
            )
            delivery_depth = args.output_bit_depth
            if delivery_depth == 8:
                source_proxy = _analysis_decode(
                    args.input,
                    color_space=args.input_color_space,
                    max_frames=args.max_frames,
                    max_side=render_max_side,
                    working_space=args.working_color_space,
                    ocio_manager=ocio_manager,
                    ocio_spaces=ocio_names,
                    pq_reference_white_nits=args.pq_reference_white_nits,
                )
                encode_video(source_proxy.frames, source_video_output, render_info.fps)
                encode_video(
                    (
                        Image.fromarray(
                            (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8),
                            mode="RGB",
                        )
                        for frame in rendered_arrays
                    ),
                    result_video_output,
                    render_info.fps,
                )
                codec = "H.264/libx264 8-bit"
                audio_preserved = False
            else:
                encode_video_high_bit(
                    render_arrays,
                    source_video_output,
                    render_info.fps,
                    bit_depth=delivery_depth,
                    color_space=args.input_color_space,
                    audio_source=args.input,
                )
                encode_video_high_bit(
                    rendered_arrays,
                    result_video_output,
                    render_info.fps,
                    bit_depth=delivery_depth,
                    color_space=args.output_color_space,
                    audio_source=args.input,
                )
                codec = f"HEVC/libx265 {delivery_depth}-bit"
                audio_preserved = has_audio_stream(args.input)
            render_width, render_height, render_fps = (
                render_info.width,
                render_info.height,
                render_info.fps,
            )
        else:
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
                raise RuntimeError("Analysis and render decodes have different frame counts.")
            rendered_frames = render_grade_frames(
                render_video.frames,
                result.frame_parameters,
                executor=render_executor,
                operations=operations,
                operation_executor=operation_executor,
                batch_size=args.render_batch_size,
                device=pool_render_device,
                mask_tracker=mask_tracker,
                use_torch_pools=not args.disable_torch_pools,
            )
            encode_video(render_video.frames, source_video_output, render_video.fps)
            encode_video(rendered_frames, result_video_output, render_video.fps)
            render_width, render_height, render_fps = (
                render_video.width,
                render_video.height,
                render_video.fps,
            )
            codec = "H.264/libx264 8-bit"
            audio_preserved = False
        payload["video_artifacts"] = {
            "input": str(source_video_output),
            "result": str(result_video_output),
            "fps": render_fps,
            "frame_count": len(decoded.frames),
            "width": render_width,
            "height": render_height,
            "analysis_width": decoded.width,
            "analysis_height": decoded.height,
            "video_codec": codec,
            "bit_depth": args.output_bit_depth,
            "input_color_space": args.input_color_space,
            "output_color_space": (
                ocio_names["output"] if ocio_manager is not None else args.output_color_space
            ),
            "color_management": "OpenColorIO" if ocio_manager is not None else "built-in ACES/XYZ",
            "pool_render_device": pool_render_device,
            "gpu_pool_render": (
                not args.disable_torch_pools
                and pool_render_device.lower().startswith("cuda")
            ),
            "semantic_masks": ["person", "skin", "sky"],
            "audio_preserved": audio_preserved,
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
        pool_v2 = (
            unified_result is not None
            and unified_result.pool_metadata.get("schema_version")
            == "pool-grade-graph/v2"
        )
        for shot in result.shots:
            row = {
                "source_video": str(decoded.source),
                "instruction": args.instruction,
                "shot": shot.shot.to_dict(),
                "accepted": shot.accepted,
                "rolled_back": shot.rolled_back,
                "rollback_reason": shot.rollback_reason,
                "search_memory": shot.search_memory,
            }
            if pool_v2:
                row["pool_operations"] = [
                    operation.to_dict()
                    for operation in unified_result.operations
                    if operation.shot_id == shot.shot.shot_id
                ]
                row["pool_audit"] = next(
                    (
                        audit
                        for audit in unified_result.operation_audit
                        if audit.get("shot_id") == shot.shot.shot_id
                    ),
                    {},
                )
            else:
                row["attempts"] = [attempt.to_dict() for attempt in shot.attempts]
                row["selected_parameters"] = {
                    str(index): values.tolist()
                    for index, values in sorted(shot.parameter_keyframes.items())
                }
            rows.append(row)
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
