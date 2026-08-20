# Conventional pixel-DDPM calibration benchmark

Date: 2026-08-15
Research mode: exploratory
Schema: `pixel-ddpm-calibration-v1`

## Decision

Can one fixed class-conditional Gaussian DDPM produce recognizable,
class-consistent, noncollapsed MNIST images from independent Gaussian noise in the
same data, rendering, evaluation, review, and artifact boundary intended for later
Eulerian generators?

This is a calibration control. It does not replace the Eulerian program and cannot
establish an Eulerian generator. A positive result freezes a conventional benchmark
and retires only nearby cutoff, gain, and reweighting changes in the current
Jacobi/Rao--Blackwell learner-controller lineage.

Proxy-only patches since the last objective-bearing experiment: 2. The next approved
mainline action must execute this benchmark; another proxy-only patch is not allowed.

## Data and roles

The only production dataset is `mnist_data/mnist_784.arff`, authenticated by SHA-256
`418c0a60d2b4abc95db2e2bbf676f3af93ddaf18f79ba3f640624ab57007fb4b`.
OpenML order is retained exactly:

- train: rows `0:55000`;
- validation: rows `55000:60000`;
- terminal test: rows `60000:70000`.

Test rows may not be parsed before the selected generator and evaluator checkpoints,
prior starts, generated samples, sampling manifest, and their hashes are frozen.
No Stage E state, target image, mass bank, protected path, or Jacobi checkpoint is a
DDPM input.

## Frozen recipe

- Model: the exact 1,378,593-parameter `ClassConditionalUNet28` defined in
  `mnist.pixel_ddpm`.
- Conditioning: sinusoidal timestep embedding plus a ten-class embedding; no label
  dropout or classifier-free guidance.
- Forward law: 1,000-step linear VP schedule, beta `1e-4` through `2e-2`.
- Objective: unweighted pixelwise epsilon MSE on model-space `[-1,1]` images.
- Training: 40 epochs, batch 128, Adam at `2e-4`, gradient norm cap 1.0, float32,
  no AMP, augmentation, LR schedule, or sweep.
- EMA: decay `0.999`; select the lowest finite fixed validation-bank EMA MSE,
  earliest exact tie, with epoch zero ineligible.
- Sampling: the exact fixed-variance DDPM posterior; no DDIM, learned variance,
  guidance, or alternate sampler.

All scientific settings and seeds are source-level constants. Production CLI options
are operational only.

## Controls and objective artifacts

The analytic epsilon oracle, zero predictor, and learned predictor all call the same
reverse-step implementation and reuse identical per-step reverse noise. Before
training, the oracle must reconstruct the fixed 20-image validation panel from start
timesteps 99, 499, and 999 with maximum endpoint model-space MSE at most `1e-6`
and at least 99% median error reduction relative to zero at every horizon.

The objective-bearing output is 160 prespecified independent-prior samples: 16 for
each requested digit, generated in four fixed balanced batches. Save all initial
Gaussian states, final uint8 images, and model-space trajectory states after 0, 250,
500, 750, and 1,000 completed reverse steps. Failed or noise-like images remain in
the arrays and contact sheets.

A fixed 40-image subset uses within-class indices 0, 5, 10, and 15. Its contact sheet
is blinded and shuffled with seed `0xDD4000`. Human labels are one of `0`--`9`,
`noise`, or `ambiguous`; answers cannot be machine-filled.

## Gates

- Gate A, execution/integrity: exact data hash, split roles, and terminal-test
  firewall.
- Gate B, execution/integrity: analytic oracle validates schedule, orientation, and
  posterior composition before learner attribution.
- Gate C, execution/integrity for classifier metrics only: the frozen evaluator must
  reach at least 97% on validation and terminal test. Failure never suppresses saved
  samples or human review.
- Gate D, execution/integrity: active time, CUDA allocation, storage, and terminal
  reserve remain inside the explicit approval.
- Diagnostic E, exploratory: classifier requested-label accuracy at least 80%, human
  requested-label agreement at least 75%, zero exact duplicate pairs, and median
  within-class diversity ratio at least 0.25.

Diagnostic E is not a confirmatory claim gate. Its failure applies only to this one
fixed bounded recipe.

## Outcome routing

- Data or oracle failure: repair only the localized data/schedule/posterior defect
  and rerun the unchanged recipe.
- Evaluator failure: preserve samples and human review; repair only the evaluator.
- Healthy noise-like learned samples: audit once against a recognized standard DDPM
  implementation before drawing an Eulerian comparison.
- Fidelity without diversity: permit at most one bounded prespecified diversity or
  training correction.
- Diagnostic E pass: freeze the DDPM/common pipeline benchmark and move future
  Eulerian work only to a materially different formulation using the same image
  benchmark interface.

No branch automatically replaces the Eulerian objective with DDPM.

## Resource and authorization boundary

Expected active time is 30--90 minutes on the target RTX 5060 Laptop GPU. The proposed
production cap is 7,200 active seconds, CUDA allocation below 75% of the current
device, storage below 500 MiB, with 900 active seconds reserved after training.
After epoch one the runner must project the frozen remaining work and pause rather
than shorten the recipe.

This implementation note authorizes no production compute. A production run requires
a fresh explicit approval reference and cap. CPU-focused tests and the synthetic
four-step/two-update smoke are permitted implementation checks.

## Claim boundary

A positive run may establish only that this frozen conventional benchmark produced
recognizable, class-consistent, noncollapsed samples under one model seed and 160
fixed Gaussian starts, and that the common pipeline is usable as an exploratory
benchmark. A negative run may establish only that this one bounded recipe did not
meet its diagnostics. Neither result establishes an Eulerian generator, DDPM
superiority, full distribution fidelity, or a confirmatory population claim.
