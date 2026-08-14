# BayesGrade Experiment Log

## 2026-08-08 — M0 Bayesian field

Implemented:

- video-conditioned Gaussian-process parameter field;
- analytic posterior mean and covariance;
- integrated-variance-reduction Anchor acquisition;
- synthetic exposure/temperature episodes;
- active, uniform, and random budget baselines.

Twenty-seed synthetic RMSE:

| Anchor budget | Active | Uniform | Random |
|---:|---:|---:|---:|
| 1 | 0.3195 | 0.3195 | 0.3195 |
| 2 | 0.1367 | 0.2365 | 0.2399 |
| 3 | 0.1342 | 0.2365 | 0.2096 |
| 4 | 0.1343 | 0.1093 | 0.1556 |
| 5 | 0.1340 | 0.1240 | 0.1586 |

Observation: analytic acquisition is strong in the low-budget regime but pure integrated variance reduction does not monotonically track reconstruction error as the budget increases.

## 2026-08-08 — M0.1 Langevin posterior

Implemented:

- convex piecewise-linear spline parameterization;
- GP posterior mean/variance initialization;
- diagonal GP preconditioning;
- Anchor, GP-prior, event-aware smoothness, parameter-bound energies;
- plug-in interface for differentiable extra energies;
- independent Langevin trajectory chains;
- sample-covariance disagreement acquisition;
- hybrid GP/Langevin acquisition.

Initial one-Anchor example, 120 frames and 16 samples:

```text
GP next Anchor:          93
Langevin next Anchor:    70
GP trajectory RMSE:      0.3195
Langevin mean RMSE:      0.3248
```

Across 20 synthetic seeds, selecting the second Anchor using pure Langevin disagreement was worse than analytic GP acquisition. This is an expected negative result for the current synthetic process, which is generated from a smooth near-Gaussian trajectory and supplies no nonlinear edit energy. The current Langevin sample variance is also under-dispersed relative to GP variance.

The hybrid acquisition with disagreement weight 0.35 preserved most of the GP behavior:

```text
Two-Anchor GP RMSE:       0.1367
Two-Anchor hybrid RMSE:   0.1408
Hybrid better/equal:      60% of seeds
```

This is not yet evidence that Langevin improves the method. It establishes a safe integration path while retaining the calibrated GP term; a positive Langevin result must be demonstrated on a non-Gaussian constraint task.

Next diagnostics:

1. measure effective sample size and chain mixing instead of using final independent states only;
2. calibrate temperature, energy weights, and prior-variance floor;
3. add a genuinely non-Gaussian local/style constraint where GP mean is insufficient;
4. retain GP acquisition as the calibrated base and treat Langevin disagreement as an auxiliary signal;
5. evaluate whether disagreement correlates with constraint violation, not only parameter RMSE.

## 2026-08-08 — M0.2 AnchorRetouchAgent integration

Repository refactor:

- archived the original 4KAgent restoration system under `legacy/4kagent/`;
- replaced the project root with the BayesGrade/retouching entry points;
- retained all original tracked files through Git moves rather than deletion.

Implemented:

- 12-dimensional deterministic differentiable RetouchExecutor;
- global exposure, white balance, tone, saturation, and vibrance operations;
- mask-local exposure, temperature, and saturation;
- heuristic zero-training single-image baseline planner;
- execution/evaluation candidate selection and Anchor parameter covariance;
- BayesGradeRetouchPipeline connecting Anchor observations to video GP inference;
- per-Anchor covariance converted into GP observation noise;
- propagation, frame rendering, and next-Anchor recommendation.

Validation:

```text
14 unit/integration tests passed
single-image CLI produced image + parameter JSON
zero parameters reproduce the input within 2e-6
executor gradients are finite with respect to all parameters
local adjustments remain inside the supplied mask
```
