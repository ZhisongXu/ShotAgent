"""Convert MonetGPT adjustment JSON to a DaVinci Resolve LUT package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from video_retouch.monet_adapter import export_monet_resolve_package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="MonetGPT JSON file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lut-size", type=int, default=33)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a non-zero MonetGPT control cannot be represented.",
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    manifest = export_monet_resolve_package(
        payload,
        args.output_dir,
        lut_size=args.lut_size,
        strict=args.strict,
    )
    print(manifest)


if __name__ == "__main__":
    main()
