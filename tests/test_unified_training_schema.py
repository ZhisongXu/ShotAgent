import unittest
import json
import tempfile
from pathlib import Path

from training.build_role_datasets import build_role_datasets
from training.schema import AgentTrainingExample, TrainingRole


class MultiAgentTrainingSchemaTest(unittest.TestCase):
    def test_anchor_example_exports_shared_task_token(self) -> None:
        example = AgentTrainingExample(
            example_id="anchor-1",
            role=TrainingRole.ANCHOR_GRADE,
            method="monet_puzzle_c",
            images=("source.png", "preview.png"),
            prompt="Stage lighting for a natural portrait.",
            response={
                "parameter_updates": {"exposure": 0.25},
                "confidence": 0.9,
            },
        )

        exported = example.to_sharegpt()

        self.assertIn("<TASK_ANCHOR_GRADE>", exported["messages"][1]["content"])
        self.assertEqual(exported["metadata"]["method"], "monet_puzzle_c")
        self.assertEqual(exported["images"], ["source.png", "preview.png"])

    def test_unknown_parameter_is_rejected_before_training(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown target parameters"):
            AgentTrainingExample(
                example_id="bad-anchor",
                role=TrainingRole.ANCHOR_GRADE,
                method="jarvis_tool_trace",
                images=("source.png",),
                prompt="edit",
                response={"parameter_updates": {"nonexistent_slider": 1.0}},
            )

    def test_critique_requires_decision_score_and_reasons(self) -> None:
        with self.assertRaisesRegex(ValueError, "accept, score, and reasons"):
            AgentTrainingExample(
                example_id="bad-critic",
                role=TrainingRole.CRITIQUE,
                method="photoagent_preference",
                images=("before.png", "after.png"),
                prompt="critique",
                response={"accept": True},
            )

    def test_dataset_builder_registers_all_role_outputs(self) -> None:
        records = [
            {
                "example_id": "storyboard-1",
                "role": "storyboard",
                "method": "video_storyboard",
                "images": ["frame.png"],
                "prompt": "split",
                "response": {"shots": [{"start_frame": 0, "end_frame": 1}]},
            },
            {
                "example_id": "editor-1",
                "role": "anchor_grade",
                "method": "photoagent_trajectory",
                "images": ["anchor.png"],
                "prompt": "grade",
                "response": {"parameter_updates": {"exposure": 0.2}},
            },
            {
                "example_id": "critic-1",
                "role": "critique",
                "method": "photoagent_preference",
                "images": ["source.png", "graded.png"],
                "prompt": "judge",
                "response": {"accept": True, "score": 0.8, "reasons": []},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            output = root / "data"

            counts = build_role_datasets([manifest], output)

            self.assertEqual(
                counts, {"storyboard": 1, "anchor_grade": 1, "critique": 1}
            )
            self.assertTrue((output / "dataset_info.json").is_file())
            self.assertTrue((output / "dynamicgrade_anchor_grade.json").is_file())


if __name__ == "__main__":
    unittest.main()
