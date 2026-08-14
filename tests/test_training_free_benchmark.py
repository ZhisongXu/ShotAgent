import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from evaluation.prepare_sdsd import build_manifest
from evaluation.prepare_autoshot import parse_annotations
from evaluation.safety_benchmark import evaluate_safety_manifest
from evaluation.video_benchmark import evaluate_manifest, load_media
from retouch_agent import RetouchExecutor, RetouchParameters
from video_retouch.io import decode_video


class TrainingFreeBenchmarkTest(unittest.TestCase):
    @staticmethod
    def _write_sequence(path: Path, frames) -> None:
        path.mkdir(parents=True)
        for index, frame in enumerate(frames):
            frame.save(path / f"{index:04d}.png")

    def test_paired_frame_manifest_runs_offline_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_frames = [
                Image.fromarray(
                    np.full((20, 24, 3), 45 + index * 3, dtype=np.uint8),
                    mode="RGB",
                )
                for index in range(4)
            ]
            reference_frames = [
                RetouchExecutor().apply(
                    frame, RetouchParameters(exposure=0.35)
                )
                for frame in source_frames
            ]
            self._write_sequence(root / "input", source_frames)
            self._write_sequence(root / "reference", reference_frames)
            manifest = {
                "dataset": "unit-paired-video",
                "profile": "intent_parameter",
                "samples": [
                    {
                        "id": "bright",
                        "input": "input",
                        "reference": "reference",
                        "fps": 4,
                        "instruction": "make it brighter",
                        "target_parameters": {"exposure": 0.35},
                        "expect_rollback": False,
                    }
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = evaluate_manifest(
                manifest_path,
                output_dir=root / "grades",
                video_output_dir=root / "videos",
                maximum_evaluations=1,
                fail_fast=True,
            )

            self.assertEqual(report["aggregate"]["successful_samples"], 1)
            self.assertIsNotNone(report["primary_score"])
            self.assertIn("reference_ssim", report["aggregate"]["metrics"])
            self.assertIn(
                "normalized_parameter_mae", report["aggregate"]["metrics"]
            )
            self.assertIn(
                "motion_compensated_edit_residual",
                report["aggregate"]["metrics"],
            )
            self.assertIn(
                "temporal_parameter_jerk", report["aggregate"]["metrics"]
            )
            self.assertTrue((root / "grades" / "bright.grade.json").is_file())
            input_video = root / "videos" / "bright.input.mp4"
            result_video = root / "videos" / "bright.result.mp4"
            self.assertTrue(input_video.is_file())
            self.assertTrue(result_video.is_file())
            self.assertEqual(len(decode_video(input_video).frames), 4)
            self.assertEqual(len(decode_video(result_video).frames), 4)
            self.assertEqual(
                report["samples"][0]["input_video_output"], str(input_video)
            )
            self.assertEqual(
                report["samples"][0]["result_video_output"], str(result_video)
            )

    def test_benchmark_rejects_single_image_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "single.png"
            Image.new("RGB", (8, 8), (40, 40, 40)).save(image_path)

            with self.assertRaisesRegex(ValueError, "cannot be still images"):
                load_media(image_path)

    def test_sdsd_adapter_pairs_matching_scene_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_sequence(
                root / "indoor" / "LQ" / "scene01",
                [Image.new("RGB", (8, 8), (20, 20, 20))],
            )
            self._write_sequence(
                root / "indoor" / "GT" / "scene01",
                [Image.new("RGB", (8, 8), (80, 80, 80))],
            )

            manifest = build_manifest(
                root, subsets=("indoor",), max_frames=1, fps=25.0
            )

            self.assertEqual(len(manifest["samples"]), 1)
            self.assertEqual(manifest["samples"][0]["fps"], 25.0)
            self.assertEqual(manifest["samples"][0]["max_frames"], 1)

    def test_video_cli_has_offline_training_free_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (24, 20)
            )
            if not writer.isOpened():
                self.skipTest("MJPG video writer is unavailable")
            for value in (35, 40, 45, 50):
                writer.write(np.full((20, 24, 3), value, dtype=np.uint8))
            writer.release()
            output = root / "grade.json"

            subprocess.run(
                [
                    sys.executable,
                    "retouch_video.py",
                    "--input",
                    str(video),
                    "--instruction",
                    "make it brighter",
                    "--offline-native",
                    "--mcts-simulations",
                    "1",
                    "--output",
                    str(output),
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["agent_runtime"]["mode"],
                "offline-native-training-free",
            )
            self.assertEqual(payload["schema_version"], "dynamic-grade-graph/v1")

    def test_autoshot_annotation_parser_uses_transition_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            annotation = Path(directory) / "labels.txt"
            annotation.write_text(
                "clip.mp4 100\n10,11\n30,35\n\nother.mp4 50\n20,21\n",
                encoding="utf-8",
            )

            rows = parse_annotations(annotation)

            self.assertEqual(rows[0], ("clip.mp4", 100, [11, 35]))

    def test_video_safety_track_scores_rollback_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 6.0, (24, 20)
            )
            if not writer.isOpened():
                self.skipTest("MJPG video writer is unavailable")
            for value in (40, 45, 50, 55, 60, 65):
                writer.write(np.full((20, 24, 3), value, dtype=np.uint8))
            writer.release()
            manifest_path = root / "safety.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset": "safety-unit",
                        "samples": [{"id": "clip", "input": str(video)}],
                    }
                ),
                encoding="utf-8",
            )

            report = evaluate_safety_manifest(
                manifest_path, max_frames=6, fail_fast=True
            )

            self.assertEqual(report["aggregate"]["successful_cases"], 5)
            self.assertGreaterEqual(report["primary_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
