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

## D0 Fixed-Grid Models

- The active positive-time theory is `docs/d0_patch_theory.tex`.
- The active experiment record, exact result, and next objective-bearing action are
  `docs/experiment12_d0_patch_plan.md`,
  `docs/jacobi_rb_global_dilated_rollout.md`, and root `HANDOFF.md`.
- The current workflow is the exploratory global-dilated
  Jacobi/Rao--Blackwell rollout. Its implemented entry point is
  `python -m mnist.diag_d0_jacobi_rb_global_dilated_rollout`. The successful
  one-path exact 128-step result improved paired squared-L2 by 7.454%; the next
  scientific milestone is an exact same-path complete reconstruction in a fresh
  immutable child.
- The global model, tangent controller, fused bank, and rollout orchestration live
  in `mnist/d0_jacobi_rb_global_dilated.py`,
  `mnist/d0_jacobi_rb_tangent_rollout.py`,
  `mnist/d0_jacobi_rb_tangent_fused.py`, and
  `mnist/diag_d0_jacobi_rb_global_dilated_rollout.py`.
- The boundary-tangent representation, exact midpoint cache, provenance,
  statistical gates, and confirmation aggregation remain in
  `mnist/d0_jacobi_rb_boundary_tangent.py`,
  `mnist/d0_jacobi_rb_boundary_tangent_cache.py`,
  `mnist/d0_jacobi_rb_boundary_tangent_provenance.py`,
  `mnist/d0_jacobi_rb_boundary_tangent_gate.py`, and
  `mnist/d0_jacobi_rb_boundary_tangent_confirmation.py`.
- The completed D0-v0 density-ratio baseline lives in
  `mnist/d0_v0_density_ratio.py`.
- The older D0-v1 potential-gradient diagnostic lives in
  `mnist/d0_v1_potential_gradient.py` and reuses the D0-v0 cache.
- Exact Jacobi certification, multi-path scheduling, refinement, Haar coupling,
  one-image learnability, and coarse-residual modules are the provenance chain for
  the active rollout. Historical proxy gates do not override the saved direct
  trajectory result or authorize mutating its sealed run.

## Notebook Contract

The notebooks import the package modules directly: `core.*`,
`examples.*`, and `mnist.*`. There is no root-level shim layer.
