import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from retouch_agent import RetouchExecutor, RetouchParameters
from video_retouch import DynamicGradePipeline
from video_retouch.backends import (
    AnchorGrade,
    HeroAnchorReference,
    MonetParameterBackend,
)
from video_retouch.monet_adapter import (
    convert_monet_adjustments,
    export_monet_resolve_package,
)
from video_retouch.resolve_export import export_resolve_package


class MonetAdapterTest(unittest.TestCase):
    @staticmethod
    def _fake_monet_repository(root: Path, adjustments: dict) -> Path:
        repository = root / "monet"
        repository.mkdir()
        script = repository / "inference_cli.py"
        script.write_text(
            "import argparse, json\n"
            "from pathlib import Path\n"
            "p=argparse.ArgumentParser(); p.add_argument('command'); "
            "p.add_argument('image'); p.add_argument('--output'); "
            "p.add_argument('--style'); a=p.parse_args()\n"
            f"Path(a.output + '.json').write_text({json.dumps(json.dumps(adjustments))})\n",
            encoding="utf-8",
        )
        return repository

    def test_maps_native_global_controls_and_audits_selective_controls(self):
        result = convert_monet_adjustments(
            {
                "Exposure": 25,
                "Temperature": -10,
                "Contrast": 20,
                "Highlights": -30,
                "Whites": 20,
                "Blacks": 40,
                "Saturation": 15,
                "HueAdjustmentGreen": 8,
            }
        )

        self.assertAlmostEqual(result.parameters.exposure, 0.25)
        self.assertAlmostEqual(result.parameters.temperature, -0.10)
        self.assertAlmostEqual(result.parameters.highlights, -0.20)
        self.assertAlmostEqual(result.parameters.shadows, 0.20)
        self.assertIn("Whites", result.approximated_fields)
        self.assertEqual(result.unsupported_fields, {"HueAdjustmentGreen": 8.0})

    def test_strict_mode_rejects_unsupported_nonzero_control(self):
        with self.assertRaisesRegex(ValueError, "HueAdjustmentBlue"):
            convert_monet_adjustments(
                {"Exposure": 10, "HueAdjustmentBlue": -20}, strict=True
            )

    def test_exports_one_lut_per_monet_shot(self):
        payload = {
            "shots": [
                {
                    "shot_id": 2,
                    "start_frame": 0,
                    "end_frame": 19,
                    "adjustments": {"Exposure": 10, "Tint": 4},
                },
                {
                    "shot_id": 3,
                    "start_frame": 20,
                    "end_frame": 39,
                    "adjustments": {"Contrast": 15, "Dehaze": 5},
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = export_monet_resolve_package(payload, Path(directory))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(len(manifest["shots"]), 2)
            self.assertTrue(Path(manifest["shots"][0]["lut"]).is_file())
            self.assertEqual(
                manifest["shots"][1]["conversion"]["unsupported_fields"],
                {"Dehaze": 5.0},
            )

    def test_cli_converts_single_adjustment_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "monet.json"
            source.write_text('{"Exposure": 12, "Saturation": 8}', encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "monet_to_resolve.py",
                    "--input",
                    str(source),
                    "--output-dir",
                    str(root / "resolve"),
                    "--lut-size",
                    "17",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest_path = Path(completed.stdout.strip())
            self.assertTrue(manifest_path.is_file())
            lut = Path(json.loads(manifest_path.read_text())["shots"][0]["lut"])
            self.assertIn("LUT_3D_SIZE 17", lut.read_text(encoding="utf-8"))

    def test_parameter_backend_enters_pipeline_and_commits_accepted_grade(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self._fake_monet_repository(
                Path(directory), {"Exposure": 20, "Saturation": 5}
            )
            backend = MonetParameterBackend(repository)
            grade = backend.grade(
                Image.new("RGB", (24, 20), (70, 80, 90)),
                "make it brighter",
                0,
                0,
            )

            self.assertTrue(grade.valid)
            self.assertAlmostEqual(grade.parameters.exposure, 0.2)
            self.assertTrue(grade.metadata["rollback_eligible"])
            self.assertNotEqual(grade.preview.getpixel((0, 0)), (70, 80, 90))

    def test_parameter_backend_matches_hero_through_shared_interface(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self._fake_monet_repository(
                Path(directory), {"Exposure": 20, "Saturation": 5}
            )
            backend = MonetParameterBackend(
                repository,
                hero_match_strength=0.5,
            )
            hero_source = Image.new("RGB", (24, 20), (90, 100, 110))
            hero_parameters = RetouchParameters(exposure=0.8, saturation=0.15)
            hero_grade = AnchorGrade(
                frame_index=4,
                parameters=hero_parameters,
                preview=RetouchExecutor().apply(hero_source, hero_parameters),
                valid=True,
                score=0.9,
                backend="hero-editor",
            )
            reference = HeroAnchorReference(
                4,
                1,
                hero_source,
                hero_grade,
                external=True,
            )

            grade = backend.grade_with_reference(
                Image.new("RGB", (24, 20), (70, 80, 90)),
                "match the reference look",
                7,
                2,
                reference,
            )

            self.assertAlmostEqual(grade.parameters.exposure, 0.5)
            self.assertAlmostEqual(grade.parameters.saturation, 0.10)
            self.assertEqual(
                grade.metadata["hero_match_method"],
                "shared_parameter_blend",
            )
            self.assertEqual(
                grade.metadata["hero_source_video"],
                "reference_video",
            )

    def test_monet_single_image_calls_produce_video_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self._fake_monet_repository(
                Path(directory), {"Exposure": 20, "Saturation": 5}
            )
            frames = tuple(
                Image.fromarray(
                    np.full((20, 24, 3), 70 + index * 3, dtype=np.uint8),
                    mode="RGB",
                )
                for index in range(4)
            )
            result = DynamicGradePipeline(
                anchor_backend=MonetParameterBackend(repository),
                maximum_attempts=1,
                maximum_hero_attempts=1,
            ).run(frames, 4.0, "natural grade")

            self.assertIsNotNone(result.hero_anchor)
            self.assertFalse(result.shots[0].rolled_back)
            self.assertGreater(float(result.frame_parameters[:, 0].mean()), 0.0)

    def test_unsupported_monet_grade_triggers_global_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self._fake_monet_repository(
                Path(directory), {"HueAdjustmentGreen": 30}
            )
            result = DynamicGradePipeline(
                anchor_backend=MonetParameterBackend(repository),
                maximum_attempts=1,
                maximum_hero_attempts=1,
            ).run(
                (Image.new("RGB", (24, 20), (70, 80, 90)),),
                24.0,
                "natural grade",
            )

            self.assertIsNone(result.hero_anchor)
            self.assertTrue(result.shots[0].rolled_back)
            self.assertTrue(result.hero_anchor_attempts)
            proposal = result.hero_anchor_attempts[0]["proposals"][0]
            self.assertFalse(proposal["valid"])
            manifest_path = export_resolve_package(
                result,
                Path(directory) / "input.mp4",
                Path(directory) / "resolve",
                lut_size=17,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["shots"][0]["rolled_back"])
            self.assertTrue(
                all(
                    value == 0.0
                    for value in manifest["shots"][0]["parameters"].values()
                )
            )


if __name__ == "__main__":
    unittest.main()
