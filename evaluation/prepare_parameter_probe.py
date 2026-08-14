"""Build a controlled text-to-parameter probe from licensed source media."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retouch_agent import RetouchExecutor, RetouchParameters

from .video_benchmark import load_media


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--parameters",
        required=True,
        help='Target parameter JSON, for example \'{"exposure": 0.35}\'.',
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--max-frames", type=int, default=24)
    args = parser.parse_args()

    raw_parameters = json.loads(args.parameters)
    if not isinstance(raw_parameters, dict):
        raise ValueError("--parameters must decode to a JSON object.")
    target = RetouchParameters.from_mapping(raw_parameters, clamp=False)
    source_path = args.input.resolve()
    media = load_media(source_path, fps=args.fps, max_frames=args.max_frames)
    frames, fps = media.frames, media.fps

    output_root = args.output_root.resolve()
    input_dir = output_root / "input"
    reference_dir = output_root / "reference"
    input_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    executor = RetouchExecutor()
    for index, frame in enumerate(frames):
        frame.save(input_dir / f"{index:06d}.png")
        executor.apply(frame, target).save(reference_dir / f"{index:06d}.png")

    manifest = {
        "schema_version": "training-free-video-benchmark-manifest/v1",
        "dataset": "controlled parameter probe",
        "profile": "intent_parameter",
        "samples": [
            {
                "id": "parameter-probe",
                "input": "input",
                "reference": "reference",
                "fps": fps,
                "instruction": args.instruction,
                "target_parameters": target.to_dict(),
                "expect_rollback": False,
            }
        ],
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
