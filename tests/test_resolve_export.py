import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from video_retouch import DynamicGradePipeline
from video_retouch.resolve_export import (
    compress_parameter_trajectory,
    export_resolve_package,
    write_dynamic_dctl,
)


class ResolveExportTest(unittest.TestCase):
    @staticmethod
    def _write_fake_resolve_module(path: Path) -> None:
        path.write_text(
            '''import json, os
from pathlib import Path

EVENTS = []
def record(name, *args):
    EVENTS.append([name, *args])
    Path(os.environ["FAKE_RESOLVE_EVENTS"]).write_text(json.dumps(EVENTS))

class Clip:
    def GetStart(self): return 100
    def GetDuration(self): return 4
    def GetNumNodes(self): return 3
    def SetLUT(self, node, path):
        record("SetLUT", node, path)
        return Path(path).is_file()

class Timeline:
    def GetItemListInTrack(self, kind, index):
        record("GetItemListInTrack", kind, index)
        return [Clip()]

class Project:
    def GetCurrentTimeline(self): return Timeline()
    def RefreshLUTList(self):
        record("RefreshLUTList")
        return True
    def SaveProject(self):
        record("SaveProject")
        return True

class Manager:
    def GetCurrentProject(self): return Project()

class Resolve:
    def GetProjectManager(self): return Manager()

def scriptapp(name):
    record("scriptapp", name)
    return Resolve()
''',
            encoding="utf-8",
        )

    def test_grade_graph_exports_resolve_lut_and_manifest(self):
        frames = tuple(
            Image.new("RGB", (24, 20), (40 + index, 55, 70))
            for index in range(4)
        )
        grade = DynamicGradePipeline(maximum_attempts=1).run(
            frames, 4.0, "make the video brighter"
        )

        with tempfile.TemporaryDirectory() as directory:
            manifest_path = export_resolve_package(
                grade, Path(directory) / "input.mp4", Path(directory) / "resolve"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            lut_path = Path(manifest["shots"][0]["lut"])

            self.assertTrue(lut_path.is_file())
            self.assertIn("LUT_3D_SIZE 33", lut_path.read_text(encoding="utf-8"))
            dctl_path = Path(manifest["shots"][0]["dynamic_dctl"])
            self.assertTrue(dctl_path.is_file())
            self.assertIn("TIMELINE_FRAME_INDEX", dctl_path.read_text(encoding="utf-8"))
            apply_script = Path(manifest["dynamic_apply_script"])
            self.assertTrue(apply_script.is_file())
            compile(apply_script.read_text(encoding="utf-8"), str(apply_script), "exec")
            self.assertGreaterEqual(len(manifest["shots"][0]["dynamic_keyframes"]), 1)
            self.assertEqual(manifest["frame_count"], 4)
            self.assertEqual(manifest["shots"][0]["start_frame"], 0)
            self.assertTrue(np.isfinite(grade.frame_parameters).all())

            fake_module_dir = Path(directory) / "fake_api"
            fake_module_dir.mkdir()
            self._write_fake_resolve_module(
                fake_module_dir / "DaVinciResolveScript.py"
            )
            event_path = Path(directory) / "resolve_events.json"
            lut_install = Path(directory) / "resolve_lut_search_path"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(fake_module_dir)
            environment["FAKE_RESOLVE_EVENTS"] = str(event_path)
            subprocess.run(
                [
                    sys.executable,
                    str(apply_script),
                    "--lut-dir",
                    str(lut_install),
                    "--node-index",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            events = json.loads(event_path.read_text(encoding="utf-8"))
            set_lut = next(event for event in events if event[0] == "SetLUT")
            self.assertEqual(set_lut[1], 2)
            installed_dctl = Path(set_lut[2])
            self.assertTrue(installed_dctl.is_file())
            self.assertIn(
                "#define DG_TIMELINE_START 100.0f",
                installed_dctl.read_text(encoding="utf-8"),
            )

    def test_dynamic_keyframe_compression_preserves_turn_and_anchor(self):
        trajectory = np.zeros((9, 12), dtype=np.float64)
        trajectory[:5, 0] = np.linspace(0.0, 0.8, 5)
        trajectory[4:, 0] = np.linspace(0.8, 0.1, 5)
        selected = compress_parameter_trajectory(
            trajectory, maximum_error=1e-6, mandatory_indices=(2,)
        )

        self.assertEqual(selected, (0, 2, 4, 8))

    def test_dynamic_dctl_contains_compressed_parameter_curve(self):
        trajectory = np.zeros((5, 12), dtype=np.float64)
        trajectory[:, 0] = np.linspace(0.0, 0.4, 5)
        with tempfile.TemporaryDirectory() as directory:
            path, keyframes = write_dynamic_dctl(
                trajectory,
                Path(directory) / "shot.dctl",
                timeline_start_frame=120,
                maximum_error=1e-6,
            )
            source = path.read_text(encoding="utf-8")

        self.assertEqual(keyframes, (0, 4))
        self.assertIn("#define DG_TIMELINE_START 120.0f", source)
        self.assertIn("float exposure", source)
        self.assertIn("0.4f", source)


if __name__ == "__main__":
    unittest.main()
