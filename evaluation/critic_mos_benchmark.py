"""Measure Video Critic correlation with human mean-opinion scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import pearsonr, spearmanr

from retouch_agent import RetouchParameters
from video_retouch.agent_config import load_multi_agent_runtime
from video_retouch.backends import AnchorGrade
from video_retouch.critic import PhotoAgentStyleCritic
from video_retouch.models import ShotPlan

from .video_benchmark import _resolve_path, load_manifest, load_media
from .provenance import file_sha256


def evaluate_critic_mos_manifest(
    manifest_path: Path,
    *,
    agent_config: Optional[Path] = None,
    max_frames: Optional[int] = 48,
    fail_fast: bool = False,
) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve()
    payload = load_manifest(manifest_path)
    if agent_config is None:
        critic = PhotoAgentStyleCritic(use_vl_review=False)
        runtime = {"critic": "photoagent-style-temporal-critic"}
    else:
        configured = load_multi_agent_runtime(agent_config)
        critic = configured.critic
        runtime = configured.manifest["evaluators"]

    rows = []
    for index, raw_sample in enumerate(payload["samples"]):
        sample = dict(raw_sample)
        sample_id = str(sample.get("id", f"sample-{index:04d}"))
        try:
            source = load_media(
                _resolve_path(sample.get("source"), manifest_path.parent, "source"),
                fps=float(sample.get("fps", payload.get("fps", 30.0))),
                max_frames=(
                    int(sample["max_frames"])
                    if sample.get("max_frames") is not None
                    else max_frames
                ),
            )
            candidate = load_media(
                _resolve_path(
                    sample.get("candidate"), manifest_path.parent, "candidate"
                ),
                fps=source.fps,
                max_frames=len(source.frames),
            )
            count = min(len(source.frames), len(candidate.frames))
            if count < 1:
                raise ValueError("Critic MOS sample has no aligned frames.")
            source_frames = source.frames[:count]
            candidate_frames = candidate.frames[:count]
            anchor_index = count // 2
            shot = ShotPlan(0, 0, count - 1, (anchor_index,))
            anchor = AnchorGrade(
                frame_index=anchor_index,
                parameters=RetouchParameters(),
                preview=candidate_frames[anchor_index],
                valid=True,
                score=1.0,
                backend="human-rated-candidate",
            )
            critique = critic.evaluate(
                source_frames,
                candidate_frames,
                np.zeros((count, 12), dtype=np.float64),
                np.zeros(count, dtype=np.float64),
                shot,
                str(
                    sample.get(
                        "instruction",
                        "evaluate professional video enhancement quality",
                    )
                ),
                (anchor,),
            )
            mos = float(sample["mos"])
            if not np.isfinite(mos):
                raise ValueError("MOS must be finite.")
            rows.append(
                {
                    "id": sample_id,
                    "mos": mos,
                    "critic_score": critique.score,
                    "accepted": critique.accepted,
                    "reasons": list(critique.reasons),
                    "metrics": critique.metrics,
                }
            )
        except Exception as error:
            if fail_fast:
                raise
            rows.append(
                {
                    "id": sample_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    valid = [row for row in rows if "error" not in row]
    mos_values = np.asarray([row["mos"] for row in valid], dtype=np.float64)
    critic_values = np.asarray(
        [row["critic_score"] for row in valid], dtype=np.float64
    )
    if len(valid) >= 2 and np.std(mos_values) > 0 and np.std(critic_values) > 0:
        pearson = float(pearsonr(mos_values, critic_values).statistic)
        spearman = float(spearmanr(mos_values, critic_values).statistic)
    else:
        pearson = None
        spearman = None
    primary = None if spearman is None else (spearman + 1.0) / 2.0
    return {
        "schema_version": "videogradebench-critic-mos/v1",
        "dataset": str(payload.get("dataset", manifest_path.stem)),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "runtime": runtime,
        "primary_score": primary,
        "aggregate": {
            "sample_count": len(rows),
            "successful_samples": len(valid),
            "failed_samples": len(rows) - len(valid),
            "pearson": pearson,
            "spearman": spearman,
            "acceptance_rate": (
                0.0
                if not valid
                else float(np.mean([row["accepted"] for row in valid]))
            ),
        },
        "samples": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    report = evaluate_critic_mos_manifest(
        args.manifest,
        agent_config=args.agent_config,
        max_frames=args.max_frames,
        fail_fast=args.fail_fast,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
