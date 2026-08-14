"""Run all VideoGradeBench-v1 tracks and produce one scorecard."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .critic_mos_benchmark import evaluate_critic_mos_manifest
from .safety_benchmark import evaluate_safety_manifest
from .scoring import geometric_mean
from .storyboard_benchmark import evaluate_storyboard_manifest
from .video_benchmark import evaluate_manifest
from .provenance import environment_manifest, file_sha256


def _path(value: object, root: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Suite track requires {field!r}.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _artifact_name(value: str) -> str:
    """Return a stable directory name without allowing path traversal."""

    name = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return name or "track"


def run_suite(
    suite_path: Path,
    *,
    agent_config: Path | None = None,
    artifact_dir: Path | None = None,
    fail_fast: bool = False,
) -> dict[str, object]:
    suite_path = Path(suite_path).resolve()
    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Suite file must be a JSON object.")
    raw_tracks = payload.get("tracks")
    if not isinstance(raw_tracks, list) or not raw_tracks:
        raise ValueError("Suite requires a non-empty tracks list.")
    root = suite_path.parent
    artifact_root = None if artifact_dir is None else Path(artifact_dir).resolve()
    if artifact_root is not None:
        artifact_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, raw_track in enumerate(raw_tracks):
        if not isinstance(raw_track, dict):
            raise ValueError("Every suite track must be a JSON object.")
        track = dict(raw_track)
        name = str(track.get("name", f"track-{index:02d}"))
        track_type = str(track.get("type", "agent"))
        track_artifact_dir = (
            None if artifact_root is None else artifact_root / _artifact_name(name)
        )
        manifest = _path(track.get("manifest"), root, "manifest")
        configured_path = agent_config
        if configured_path is None and track.get("agent_config") is not None:
            configured_path = _path(track.get("agent_config"), root, "agent_config")
        try:
            if track_type == "agent":
                report = evaluate_manifest(
                    manifest,
                    agent_config=configured_path,
                    output_dir=(
                        None
                        if track_artifact_dir is None
                        else track_artifact_dir / "grades"
                    ),
                    video_output_dir=(
                        None
                        if track_artifact_dir is None
                        else track_artifact_dir / "videos"
                    ),
                    maximum_evaluations=int(track.get("maximum_evaluations", 3)),
                    max_frames=(
                        None
                        if track.get("max_frames") is None
                        else int(track["max_frames"])
                    ),
                    fail_fast=fail_fast,
                )
            elif track_type == "storyboard":
                report = evaluate_storyboard_manifest(
                    manifest,
                    agent_config=configured_path,
                    anchors_per_shot=int(track.get("anchors_per_shot", 1)),
                    fail_fast=fail_fast,
                )
            elif track_type == "safety":
                report = evaluate_safety_manifest(
                    manifest,
                    max_frames=(
                        None
                        if track.get("max_frames") is None
                        else int(track["max_frames"])
                    ),
                    fail_fast=fail_fast,
                )
            elif track_type == "critic_mos":
                report = evaluate_critic_mos_manifest(
                    manifest,
                    agent_config=configured_path,
                    max_frames=(
                        None
                        if track.get("max_frames") is None
                        else int(track["max_frames"])
                    ),
                    fail_fast=fail_fast,
                )
            else:
                raise ValueError(f"Unsupported suite track type: {track_type}")
            track_report_path = None
            if track_artifact_dir is not None:
                track_artifact_dir.mkdir(parents=True, exist_ok=True)
                track_report_path = track_artifact_dir / "report.json"
                track_report_path.write_text(
                    json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8",
                )
            reports.append(
                {
                    "name": name,
                    "type": track_type,
                    "primary_score": report.get("primary_score"),
                    **(
                        {}
                        if track_report_path is None
                        else {"artifact_report": str(track_report_path)}
                    ),
                    "report": report,
                }
            )
        except Exception as error:
            if fail_fast:
                raise
            reports.append(
                {
                    "name": name,
                    "type": track_type,
                    "primary_score": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    scored = [
        float(track["primary_score"])
        for track in reports
        if track.get("primary_score") is not None
    ]
    expected_tracks = int(payload.get("expected_track_count", len(reports)))
    partial_score = geometric_mean(scored)
    complete = len(scored) == expected_tracks and len(reports) == expected_tracks
    return {
        "schema_version": "videogradebench-suite/v1",
        "suite": str(payload.get("name", "VideoGradeBench-v1")),
        "suite_file": str(suite_path),
        "suite_sha256": file_sha256(suite_path),
        "artifact_dir": None if artifact_root is None else str(artifact_root),
        "environment": environment_manifest(),
        "overall_score": partial_score if complete else None,
        "partial_score": partial_score,
        "complete": complete,
        "scored_tracks": len(scored),
        "run_tracks": len(reports),
        "expected_tracks": expected_tracks,
        "tracks": reports,
    }


def render_markdown(report: dict[str, object]) -> str:
    overall = report.get("overall_score")
    overall_text = "n/a" if overall is None else f"{float(overall):.4f}"
    partial = report.get("partial_score")
    partial_text = "n/a" if partial is None else f"{float(partial):.4f}"
    lines = [
        f"# {report['suite']}",
        "",
        f"Overall score: **{overall_text}**  ",
        f"Partial score: **{partial_text}**  ",
        f"Track coverage: **{report['scored_tracks']}/{report['expected_tracks']}**",
        "",
        "| Track | Type | Primary score | Status |",
        "|---|---|---:|---|",
    ]
    for track in report["tracks"]:
        score = track.get("primary_score")
        score_text = "n/a" if score is None else f"{float(score):.4f}"
        status = "failed" if "error" in track else "ok"
        lines.append(
            f"| {track['name']} | {track['type']} | {score_text} | {status} |"
        )
    lines.extend(["", "## Key metrics", ""])
    for track in report["tracks"]:
        if "report" not in track:
            continue
        aggregate = track["report"].get("aggregate", {})
        metrics = aggregate.get("metrics", aggregate)
        lines.append(f"### {track['name']}")
        lines.append("")
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                lines.append(f"- `{name}`: {float(value):.6f}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help=(
            "Per-track reports and per-sample grade graphs. Defaults to "
            "<output-stem>_artifacts beside --output."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    artifact_dir = args.artifact_dir
    if artifact_dir is None:
        artifact_dir = args.output.parent / f"{args.output.stem}_artifacts"
    report = run_suite(
        args.suite,
        agent_config=args.agent_config,
        artifact_dir=artifact_dir,
        fail_fast=args.fail_fast,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(args.output)
    if args.markdown is not None:
        print(args.markdown)


if __name__ == "__main__":
    main()
