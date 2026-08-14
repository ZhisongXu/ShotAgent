"""Convert a source/candidate/MOS CSV into a Video Critic benchmark manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--source-column", default="source")
    parser.add_argument("--candidate-column", default="candidate")
    parser.add_argument("--mos-column", default="mos")
    parser.add_argument("--id-column")
    parser.add_argument("--dataset-name", default="VDPVE video enhancement MOS")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    candidate_root = args.candidate_root.resolve()
    samples = []
    with args.csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            source = (source_root / row[args.source_column]).resolve()
            candidate = (candidate_root / row[args.candidate_column]).resolve()
            if not source.is_file() or not candidate.is_file():
                continue
            samples.append(
                {
                    "id": (
                        row[args.id_column]
                        if args.id_column and row.get(args.id_column)
                        else f"mos-{index:05d}"
                    ),
                    "source": str(source),
                    "candidate": str(candidate),
                    "mos": float(row[args.mos_column]),
                    "instruction": "评价视频增强后的专业画质、色彩和时间一致性",
                }
            )
            if args.limit is not None and len(samples) >= args.limit:
                break
    if not samples:
        raise RuntimeError("No source/candidate/MOS rows resolved to video files.")
    payload = {
        "schema_version": "videogradebench-critic-mos-manifest/v1",
        "dataset": args.dataset_name,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
