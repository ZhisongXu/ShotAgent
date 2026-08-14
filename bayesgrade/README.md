# BayesGrade prototype

This directory contains the first mathematical prototype for the research
proposal in [`research/BAYESGRADE_RP.md`](../research/BAYESGRADE_RP.md).

Current scope:

- video-conditioned Gaussian-process grade field;
- multi-parameter posterior mean and uncertainty;
- label-free integrated-variance-reduction Anchor acquisition;
- GP-preconditioned Langevin sampling over spline grade controls;
- posterior-disagreement Anchor acquisition;
- synthetic piecewise-smooth retouching episode.

Run the prototype:

```bash
python -m bayesgrade.demo --frames 120 --budget 5
python -m bayesgrade.demo_langevin --frames 120 --samples 16
```

Run the tests:

```bash
python -m unittest tests.test_bayesgrade_parameter_field tests.test_bayesgrade_langevin
```

Compare quality–Anchor budget curves on synthetic episodes:

```bash
python -m bayesgrade.benchmark_synthetic --max-budget 5 --seeds 20
```

The current prototype does not yet decode videos or call an image retouching
executor. Those interfaces are the next milestone after validating the active
selection behavior on synthetic trajectories.
