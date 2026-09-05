"""Render ShotAgent and published baselines on one reference-video manifest."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

METHODS = ("sa-lut", "nlut", "cap-vstnet", "canoncgt", "shotagent-pool")


def _run(command: list[str], cwd: Path) -> float:
    started = time.perf_counter()
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)
    return time.perf_counter() - started


def _baseline_command(
    method: str,
    python: str,
    repo: Path,
    target: Path,
    reference: Path,
    output: Path,
    max_frames: int,
    max_side: int,
    nlut_iterations: int,
) -> list[str]:
    common = [
        "--target",
        str(target),
        "--reference",
        str(reference),
        "--output",
        str(output),
        "--max-frames",
        str(max_frames),
        "--max-side",
        str(max_side),
        "--encode-quality",
        "10",
    ]
    baseline_root = repo / "outputs/reference_video_eval/modern_baselines"
    if method == "sa-lut":
        return [
            python,
            "-m",
            "evaluation.run_salut_baseline",
            "--repo-dir",
            str(baseline_root / "SA-LUT/SA-LUT"),
            "--checkpoint",
            str(
                baseline_root
                / "SA-LUT/SA-LUT/ckpts/salut_ckpt/epoch=100-step=4127466.ckpt.state.pt"
            ),
            *common,
        ]
    if method == "nlut":
        return [
            python,
            "-m",
            "evaluation.run_nlut_baseline",
            "--repo-dir",
            str(baseline_root / "NLUT"),
            "--checkpoint",
            str(baseline_root / "NLUT/experiments/336999_style_lut.pth"),
            "--iterations",
            str(nlut_iterations),
            *common,
        ]
    if method == "cap-vstnet":
        return [
            python,
            "-m",
            "evaluation.run_cap_vstnet_baseline",
            "--repo-dir",
            str(baseline_root / "cap-vstnet"),
            "--checkpoint",
            str(baseline_root / "cap-vstnet/checkpoints/checkpoints/photo_video.pt"),
            *common,
        ]
    if method == "canoncgt":
        return [
            python,
            "-m",
            "evaluation.run_canoncgt_baseline",
            "--repo-dir",
            str(baseline_root / "CanonCGT"),
            "--checkpoint",
            str(
                baseline_root / "CanonCGT/pretrained_downloaded/SSL_updated_251111.pth"
            ),
            "--config",
            str(
                baseline_root
                / "CanonCGT/configs/Stage3_SSL_training_Flickr2K_PPR10K_LSDIR.yaml"
            ),
            *common,
        ]
    raise ValueError(f"Unsupported baseline: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--nlut-iterations", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_root = manifest_path.parent
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timing_path = output_root / "render_timings.json"
    timings: list[dict[str, object]] = (
        json.loads(timing_path.read_text(encoding="utf-8"))
        if timing_path.is_file()
        else []
    )
    python = sys.executable
    instruction = (
        "Match the reference video's complete cinematic color grade: deep-shadow "
        "black level and chroma, shadow tint, midtone luminance and palette, "
        "neutral-axis temperature, highlight tint and roll-off, dominant and "
        "secondary palette weights, saturation hierarchy, and local contrast/depth. "
        "Make a clearly visible, decisive transfer wherever the reference differs. "
        "Preserve object identity, geometry, texture, motion, clean shadow detail, "
        "and highlight detail; keep one temporally coherent grade without flicker, "
        "clipping, gray haze, or a weak near-neutral result."
    )

    for sample in manifest["samples"]:
        sample_id = str(sample["id"])
        target = (manifest_root / str(sample["target"])).resolve()
        reference = (manifest_root / str(sample["reference"])).resolve()
        max_frames = int(sample.get("max_frames", 96))
        max_side = int(sample.get("max_side", 512))
        for method in args.methods:
            method_root = output_root / method
            output = method_root / f"{sample_id}.mp4"
            if output.is_file() and not args.overwrite:
                print(f"SKIP {method} {sample_id}: {output}", flush=True)
                continue
            method_root.mkdir(parents=True, exist_ok=True)
            if method == "shotagent-pool":
                artifact_dir = method_root / "artifacts" / sample_id
                grade_path = method_root / "grades" / f"{sample_id}.json"
                command = [
                    python,
                    "retouch_video.py",
                    "--input",
                    str(target),
                    "--reference-video",
                    str(reference),
                    "--instruction",
                    instruction,
                    "--output",
                    str(grade_path),
                    "--agent-config",
                    str(repo / "configs/reference_video_pool.json"),
                    "--video-output-dir",
                    str(artifact_dir),
                    "--max-frames",
                    str(max_frames),
                    "--analysis-max-side",
                    str(max_side),
                    "--render-max-side",
                    str(max_side),
                    "--encode-preset",
                    "veryfast",
                    "--encode-quality",
                    "10",
                    "--compact",
                ]
                elapsed = _run(command, repo)
                rendered = artifact_dir / f"{target.stem}.graded.mp4"
                shutil.copy2(rendered, output)
            else:
                command = _baseline_command(
                    method,
                    python,
                    repo,
                    target,
                    reference,
                    output,
                    max_frames,
                    max_side,
                    args.nlut_iterations,
                )
                elapsed = _run(command, repo)
            timings.append({"sample": sample_id, "method": method, "seconds": elapsed})
            timing_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
