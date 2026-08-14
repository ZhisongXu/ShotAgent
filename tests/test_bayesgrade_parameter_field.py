import unittest

import numpy as np

from bayesgrade.parameter_field import BayesianGradeField


class BayesianGradeFieldTest(unittest.TestCase):
    def setUp(self) -> None:
        self.times = np.linspace(0.0, 1.0, 40)
        self.values = np.stack(
            [np.sin(2.0 * np.pi * self.times), np.cos(np.pi * self.times)],
            axis=1,
        )
        self.field = BayesianGradeField(
            temporal_lengthscale=0.22,
            observation_noise=1e-7,
        )

    def test_posterior_interpolates_anchor_parameters(self) -> None:
        anchors = [4, 19, 34]
        posterior = self.field.posterior(
            self.times,
            anchors,
            self.values[anchors],
        )
        np.testing.assert_allclose(
            posterior.mean[anchors],
            self.values[anchors],
            atol=2e-5,
        )
        self.assertTrue(np.all(posterior.variance[anchors] < 2e-6))

    def test_adding_anchor_reduces_integrated_variance(self) -> None:
        first = self.field.posterior(self.times, [8], self.values[[8]])
        next_anchor, _ = self.field.select_next_anchor(first, [8])
        second = self.field.posterior(
            self.times,
            [8, next_anchor],
            self.values[[8, next_anchor]],
        )
        self.assertLess(second.integrated_variance, first.integrated_variance)

    def test_acquisition_never_reselects_an_anchor(self) -> None:
        anchors = [3, 20]
        posterior = self.field.posterior(
            self.times,
            anchors,
            self.values[anchors],
        )
        selected, score = self.field.select_next_anchor(posterior, anchors)
        self.assertNotIn(selected, anchors)
        self.assertTrue(np.isfinite(score))

    def test_video_features_isolate_an_appearance_regime(self) -> None:
        regime = (self.times >= 0.5).astype(np.float64)
        features = regime[:, None]
        field = BayesianGradeField(
            temporal_lengthscale=0.5,
            feature_lengthscale=0.25,
            observation_noise=1e-7,
        )
        posterior = field.posterior(
            self.times,
            [8],
            self.values[[8]],
            features=features,
        )
        # The observation explains nearby frames in the same regime but should
        # not produce false confidence after the appearance-state transition.
        self.assertGreater(posterior.variance[30], posterior.variance[10] + 0.4)


if __name__ == "__main__":
    unittest.main()
