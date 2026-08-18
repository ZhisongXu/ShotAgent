import unittest

from PIL import Image

from retouch_agent.planner import HeuristicRetouchPlanner


class ProfessionalPlannerTest(unittest.TestCase):
    def test_generic_retouch_has_balanced_visible_default_direction(self):
        plan = HeuristicRetouchPlanner().plan(
            Image.new("RGB", (24, 20), (110, 120, 130)),
            "自动修图",
        )
        parameters = plan.initial_parameters

        self.assertGreater(parameters.exposure, 0.0)
        self.assertGreater(parameters.contrast, 0.0)
        self.assertLess(parameters.highlights, 0.0)
        self.assertGreater(parameters.shadows, 0.0)
        self.assertGreater(parameters.vibrance, 0.0)
        self.assertGreater(parameters.tone_curve, 0.0)

    def test_ocean_commercial_intent_protects_range_and_biases_water_cool(self):
        plan = HeuristicRetouchPlanner().plan(
            Image.new("RGB", (24, 20), (90, 120, 145)),
            "清透的海洋旅行广告，保护高光和暗部细节",
        )
        parameters = plan.initial_parameters

        self.assertLess(parameters.temperature, 0.0)
        self.assertLess(parameters.tint, 0.0)
        self.assertLess(parameters.highlights, 0.0)
        self.assertGreater(parameters.shadows, 0.0)
        self.assertGreater(parameters.contrast, 0.0)
        self.assertGreater(parameters.vibrance, 0.0)

    def test_film_and_highlight_intents_compose(self):
        plan = HeuristicRetouchPlanner().plan(
            Image.new("RGB", (24, 20), (100, 100, 100)),
            "warm cinematic film coast with preserved highlight detail",
        )
        parameters = plan.initial_parameters

        self.assertGreater(parameters.temperature, 0.0)
        self.assertGreater(parameters.contrast, 0.0)
        self.assertLess(parameters.saturation, 0.0)
        self.assertLess(parameters.highlights, 0.0)

    def test_restrained_saturation_overrides_vivid_token(self):
        plan = HeuristicRetouchPlanner().plan(
            Image.new("RGB", (24, 20), (70, 110, 150)),
            "色彩鲜艳但不过饱和，饱和度克制",
        )

        self.assertLessEqual(plan.initial_parameters.saturation, 0.0)
        self.assertLess(plan.initial_parameters.vibrance, 0.10)

    def test_filmic_midtones_soften_contrast_and_roll_off_highlights(self):
        plan = HeuristicRetouchPlanner().plan(
            Image.new("RGB", (24, 20), (100, 100, 100)),
            "胶片式柔和对比，提亮中间调并使用高光 roll-off",
        )
        parameters = plan.initial_parameters

        self.assertAlmostEqual(parameters.exposure, 0.12)
        self.assertAlmostEqual(parameters.contrast, 0.10)
        self.assertLess(parameters.highlights, 0.0)


if __name__ == "__main__":
    unittest.main()
