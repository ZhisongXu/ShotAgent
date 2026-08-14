"""Validate manifests and export one multimodal dataset per Agent role."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.schema import AgentTrainingExample, TrainingRole


def load_jsonl(path: Path) -> list[AgentTrainingExample]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("record must be an object")
                examples.append(AgentTrainingExample.from_dict(payload, path.parent))
            except Exception as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return examples


def build_role_datasets(inputs: list[Path], output_directory: Path) -> dict[str, int]:
    by_role: dict[TrainingRole, list[AgentTrainingExample]] = {
        role: [] for role in TrainingRole
    }
    ids = set()
    for path in inputs:
        for example in load_jsonl(path):
            if example.example_id in ids:
                raise ValueError(f"Duplicate example id: {example.example_id}")
            ids.add(example.example_id)
            by_role[example.role].append(example)

    output_directory.mkdir(parents=True, exist_ok=True)
    counts = {}
    for role, examples in by_role.items():
        output = output_directory / f"dynamicgrade_{role.value}.json"
        output.write_text(
            json.dumps(
                [example.to_sharegpt() for example in examples],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        counts[role.value] = len(examples)
    catalog = Path(__file__).parent / "configs" / "dataset_info.json"
    (output_directory / "dataset_info.json").write_text(
        catalog.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    counts = build_role_datasets(args.input, args.output_directory)
    print(json.dumps({"examples": sum(counts.values()), "roles": counts}))


if __name__ == "__main__":
    main()
