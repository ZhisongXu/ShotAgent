"""Command-line entry point for the single-image AnchorRetouchAgent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from retouch_agent import AnchorRetouchAgent, HeuristicRetouchPlanner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument(
        "--no-rollback",
        action="store_true",
        help="Keep the best candidate even if it fails the quality gate.",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.0,
        help="Minimum target-alignment gain required to commit an edit.",
    )
    parser.add_argument(
        "--min-perceptual-delta",
        type=float,
        default=0.01,
        help="Minimum RGB RMS change required to commit a visible edit.",
    )
    args = parser.parse_args()

    planner = HeuristicRetouchPlanner()
    agent = AnchorRetouchAgent(
        planner=planner,
        candidate_count=args.candidates,
        rollback_on_failure=not args.no_rollback,
        minimum_score_improvement=args.min_improvement,
        minimum_perceptual_delta=args.min_perceptual_delta,
    )
    image = Image.open(args.input).convert("RGB")
    reference = Image.open(args.reference).convert("RGB") if args.reference else None
    mask = Image.open(args.mask).convert("L") if args.mask else None
    result = agent.run(image, args.instruction, reference=reference, local_mask=mask)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(args.output)
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output)
    print(metadata_path)


if __name__ == "__main__":
    main()
