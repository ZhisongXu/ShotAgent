import unittest
import re

import numpy as np
from PIL import Image

from video_retouch.shot_planner import (
    HeuristicShotPlanner,
    LongVideoStoryboardSettings,
    VLShotPlanner,
)


class UnavailableVLClient:
    model_id = "unavailable/test-model"

    def generate_json(self, labeled_images, prompt):
        raise RuntimeError("VL intentionally unavailable")


class RecordingHierarchicalVLClient:
    model_id = "gpt-5.6-sol/test-double"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    @staticmethod
    def _label_value(label: str, name: str) -> int:
        match = re.search(rf"{name}=(\d+)", label)
        if match is None:
            raise AssertionError(f"missing {name} in {label}")
        return int(match.group(1))

    def generate_json(self, labeled_images, prompt):
        labels = [label for label, _ in labeled_images]
        self.calls.append((prompt, labels))
        if "<TASK_LONG_VIDEO_OVERVIEW>" in prompt:
            return {
                "summary": "three continuous camera setups",
                "recurring_elements": ["ocean"],
                "visual_regimes": [],
                "continuity_notes": [],
            }
        if "<TASK_SHOT_WINDOW>" in prompt:
            frame_ids = [self._label_value(label, "frame_id") for label in labels]
            start, end = min(frame_ids), max(frame_ids)
            boundaries = []
            for frame in (20, 50):
                if start < frame <= end:
                    boundaries.append(
                        {
                            "boundary_frame": frame,
                            "transition_type": "semantic_scene_change",
                            "confidence": 0.94,
                            "evidence": "new camera setup",
                        }
                    )
            return {
                "boundaries": boundaries,
                "window_description": "test window",
                "uncertain_ranges": [],
            }
        if "<TASK_BOUNDARY_VERIFY>" in prompt:
            candidates: dict[int, int] = {}
            for label in labels:
                candidate_id = self._label_value(label, "candidate_id")
                candidates[candidate_id] = self._label_value(
                    label, "approximate_boundary"
                )
            return {
                "decisions": [
                    {
                        "candidate_id": candidate_id,
                        "accept": True,
                        "boundary_frame": frame,
                        "transition_type": "hard_cut",
                        "confidence": 0.97,
                        "reason": "before and after are different setups",
                    }
                    for candidate_id, frame in sorted(candidates.items())
                ]
            }
        if "<TASK_ANCHOR_SELECTION>" in prompt:
            by_shot: dict[int, list[int]] = {}
            for label in labels:
                shot_id = self._label_value(label, "shot_id")
                by_shot.setdefault(shot_id, []).append(
                    self._label_value(label, "frame_id")
                )
            return {
                "shots": [
                    {
                        "shot_id": shot_id,
                        "description": f"setup {shot_id}",
                        "anchors": [
                            {
                                "frame": sorted(set(frame_ids))[len(set(frame_ids)) // 2],
                                "rank": 1,
                                "confidence": 0.95,
                                "reason": "sharp representative tonal state",
                            }
                        ],
                    }
                    for shot_id, frame_ids in sorted(by_shot.items())
                ]
            }
        if "<TASK_HERO_ANCHOR_SELECTION>" in prompt:
            candidates = []
            for label in labels:
                candidates.append(
                    {
                        "frame": self._label_value(label, "frame_id"),
                        "shot_id": self._label_value(label, "shot_id"),
                        "confidence": 0.93,
                        "reason": "best whole-video master reference",
                    }
                )
            return {"ranked_candidates": candidates[:5]}
        raise AssertionError("unexpected task prompt")


class VideoShotPlannerTest(unittest.TestCase):
    @staticmethod
    def _solid(color: tuple[int, int, int]) -> Image.Image:
        array = np.zeros((24, 32, 3), dtype=np.uint8)
        array[:] = color
        return Image.fromarray(array, mode="RGB")

    def test_detects_hard_cut_and_selects_anchor_per_shot(self) -> None:
        frames = [self._solid((180, 20, 20)) for _ in range(4)]
        frames += [self._solid((20, 20, 180)) for _ in range(4)]
        planner = HeuristicShotPlanner(cut_threshold=0.3, minimum_shot_seconds=0.1)

        plan = planner.plan(frames, fps=4.0, instruction="cinematic")

        self.assertEqual(len(plan.shots), 2)
        self.assertEqual((plan.shots[0].start_frame, plan.shots[0].end_frame), (0, 3))
        self.assertEqual((plan.shots[1].start_frame, plan.shots[1].end_frame), (4, 7))
        self.assertEqual(len(plan.shots[0].anchor_frames), 1)
        self.assertEqual(len(plan.shots[1].anchor_frames), 1)

    def test_vl_failure_uses_explicit_storyboard_fallback(self) -> None:
        frames = [self._solid((80, 90, 100)) for _ in range(5)]
        planner = VLShotPlanner(client=UnavailableVLClient())

        plan = planner.plan(frames, fps=5.0, instruction="natural warm")

        self.assertEqual(plan.planner, "heuristic_fallback")
        self.assertIn("VL intentionally unavailable", plan.diagnosis["fallback_reason"])
        self.assertEqual(plan.frame_count, 5)

    def test_merges_one_frame_trailing_fragment(self) -> None:
        frames = [self._solid((180, 20, 20)) for _ in range(12)]
        frames.append(self._solid((20, 20, 180)))
        planner = HeuristicShotPlanner(
            cut_threshold=0.3,
            minimum_shot_seconds=0.4,
        )

        plan = planner.plan(frames, fps=10.0, instruction="natural")

        self.assertEqual(len(plan.shots), 1)
        self.assertEqual(plan.shots[0].end_frame, 12)

    def test_long_video_uses_overlapping_windows_verification_and_anchor_role(
        self,
    ) -> None:
        frames = [self._solid((80, 90, 100)) for _ in range(80)]
        client = RecordingHierarchicalVLClient()
        settings = LongVideoStoryboardSettings(
            overview_frames=12,
            window_seconds=5.0,
            overlap_seconds=1.0,
            max_window_images=12,
            technical_candidate_threshold=0.14,
            hard_cut_threshold=0.42,
            boundary_context_frames=2,
            boundary_batch_size=2,
            anchor_candidates_per_shot=6,
            anchor_shots_per_call=2,
            max_anchor_images_per_call=12,
            minimum_shot_seconds=0.5,
        )
        planner = VLShotPlanner(client=client, settings=settings, strict=True)

        plan = planner.plan(
            frames,
            fps=4.0,
            instruction="preserve the ocean while preparing a cinematic grade",
        )

        self.assertEqual(plan.planner, "hierarchical-vision-storyboard/v2")
        self.assertEqual([shot.start_frame for shot in plan.shots], [0, 20, 50])
        self.assertEqual([shot.end_frame for shot in plan.shots], [19, 49, 79])
        self.assertTrue(all(len(shot.anchor_frames) == 1 for shot in plan.shots))
        self.assertGreater(plan.diagnosis["window_count"], 1)
        self.assertEqual(plan.diagnosis["full_scan_frame_count"], 80)
        self.assertEqual(plan.diagnosis["final_boundaries"], [20, 50])
        prompts = [prompt for prompt, _ in client.calls]
        self.assertTrue(any("<TASK_LONG_VIDEO_OVERVIEW>" in p for p in prompts))
        self.assertGreater(
            sum("<TASK_SHOT_WINDOW>" in prompt for prompt in prompts), 1
        )
        self.assertTrue(any("<TASK_BOUNDARY_VERIFY>" in p for p in prompts))
        self.assertTrue(any("<TASK_ANCHOR_SELECTION>" in p for p in prompts))
        self.assertTrue(
            any("<TASK_HERO_ANCHOR_SELECTION>" in p for p in prompts)
        )
        self.assertIsNotNone(plan.hero_anchor_frame)
        self.assertIn(plan.hero_anchor_frame, plan.hero_anchor_candidates)


if __name__ == "__main__":
    unittest.main()
