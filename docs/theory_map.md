# Theory Map

This file maps the paper-facing concepts to the cleaned implementation layout.

## Core Wasserstein Conditioning

- Manuscript algorithms 1--3 live in `core/wasserstein_conditioning_algorithms.py`.
- Shared validation, time-grid, random-generator, log-sum-exp, and periodic image
  helpers live in `core/conditioning_utils.py`.

## Small H-Transform Examples

- Gaussian mixture terminal examples live in `examples/gaussian_mixture_htransform.py`.
- Fixed-count two-well examples live in `examples/fixed_count_two_well_htransform.py`.
- Factorized two-well examples live in `examples/factorized_two_well_htransform.py`.

## MNIST Weighted Point Clouds

- Image-to-atomic-measure conversion and rasterization live in
  `mnist/weighted_point_cloud.py`.
- Terminal classifier guidance and Monte Carlo h-transform sampling live in
  `mnist/conditioned_diffusion.py`.
- Experiment-6 reparameterized sampling and noisy terminal training live in
  `mnist/experiment6_fixes.py`.
- Random-search utilities for experiment 6 live in
  `mnist/experiment6_hyperparameter_search.py`.
- Score-matching networks, DSM targets, training, diagnostics, and reverse
  samplers live in `mnist/score_matching.py`.

## Notebook Contract

The notebooks import the package modules directly: `core.*`,
`examples.*`, and `mnist.*`. There is no root-level shim layer.
