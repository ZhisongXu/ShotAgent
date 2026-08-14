import unittest

import numpy as np

from bayesgrade.langevin import (
    LangevinGradeSampler,
    hybrid_acquisition_scores,
    linear_spline_basis,
    select_disagreement_anchor,
)
from bayesgrade.parameter_field import BayesianGradeField


class LangevinGradeSamplerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.times = np.linspace(0.0, 1.0, 48)
        self.features = (self.times >= 0.55).astype(np.float64)[:, None]
        self.parameters = np.stack(
            [
                0.2 * np.sin(2.0 * np.pi * self.times) - 0.4 * self.features[:, 0],
                0.15 * np.cos(np.pi * self.times) + 0.3 * self.features[:, 0],
            ],
            axis=1,
        )
        self.anchors = [10]
        field = BayesianGradeField(
            temporal_lengthscale=0.22,
            feature_lengthscale=0.4,
            observation_noise=1e-6,
        )
        self.posterior = field.posterior(
            self.times,
            self.anchors,
            self.parameters[self.anchors],
            features=self.features,
        )

    def test_linear_basis_is_convex(self) -> None:
        basis = linear_spline_basis(self.times, 8)
        self.assertEqual(basis.shape, (48, 8))
        np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-12)
        self.assertTrue(np.all(basis >= 0.0))

    def test_sampler_returns_finite_bounded_trajectories(self) -> None:
        sampler = LangevinGradeSampler(
            control_count=8,
            iterations=120,
            temperature=0.1,
        )
        samples = sampler.sample(
            self.times,
            self.posterior.mean,
            self.posterior.variance,
            self.anchors,
            self.parameters[self.anchors],
            num_samples=6,
            parameter_bounds=(np.array([-0.8, -0.8]), np.array([0.8, 0.8])),
            seed=4,
        )
        self.assertEqual(samples.trajectories.shape, (6, 48, 2))
        self.assertTrue(np.all(np.isfinite(samples.trajectories)))
        self.assertTrue(np.all(samples.trajectories <= 0.8 + 1e-12))
        self.assertTrue(np.all(samples.trajectories >= -0.8 - 1e-12))
        np.testing.assert_allclose(
            samples.mean[self.anchors],
            self.parameters[self.anchors],
            atol=0.08,
        )

    def test_disagreement_acquisition_does_not_reselect_anchor(self) -> None:
        sampler = LangevinGradeSampler(
            control_count=8,
            iterations=80,
            temperature=0.12,
        )
        samples = sampler.sample(
            self.times,
            self.posterior.mean,
            self.posterior.variance,
            self.anchors,
            self.parameters[self.anchors],
            num_samples=6,
            seed=8,
        )
        index, score = select_disagreement_anchor(samples, self.anchors)
        self.assertNotIn(index, self.anchors)
        self.assertTrue(np.isfinite(score))
        self.assertGreaterEqual(score, 0.0)

        gp_scores = np.linspace(1.0, 2.0, self.times.size)
        hybrid = hybrid_acquisition_scores(gp_scores, samples, self.anchors)
        self.assertEqual(hybrid.shape, (self.times.size,))
        self.assertTrue(np.isneginf(hybrid[self.anchors[0]]))
        self.assertTrue(np.isfinite(hybrid[np.arange(self.times.size) != self.anchors[0]]).all())


if __name__ == "__main__":
    unittest.main()
