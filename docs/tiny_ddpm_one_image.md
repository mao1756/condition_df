# Tiny pixel-DDPM one-image experiment

Date: 2026-08-20  
Primary mode: exploratory  
Program objective: establish or decisively falsify a DDPM-like MNIST generator
based on the Eulerian approximation. This experiment is a conventional pixel-DDPM
capacity and pipeline control; it is not evidence for the Eulerian mechanism.

## Decision

Can a fixed 29,913-parameter conventional pixel DDPM memorize the first label-3
MNIST training image and produce target-like complete samples from independent
Gaussian starts?

The objective-bearing artifacts are complete 1,000-step zero, analytic-oracle, and
learned trajectories with shared starts and reverse noise, plus their final images.

## Competing hypotheses

- The tiny model and standard epsilon objective are sufficient. The oracle and
  learned rows should both reconstruct the source, while learned samples improve
  materially over zero prediction.
- The model can denoise forward states but fails from an independent prior. Short
  reconstruction should work while complete sampling fails, implicating terminal
  time coverage or long-horizon composition before width.
- The architecture or optimization is inadequate. The oracle should pass while the
  learned model fails both reconstruction and prior sampling.
- The implementation or reverse orientation is wrong. The analytic oracle should
  fail through the same reverse-step implementation.
- Epsilon validation MSE is a misleading proxy at this scale. Validation can improve
  without visibly useful complete samples, or vice versa.

## Frozen exploratory design

- Source: the first label-3 image in the authenticated MNIST training order.
- Model: `TinyClassConditionalUNet28`, exactly 29,913 trainable parameters.
- Channels: one 8-channel full-resolution block, one 16-channel half-resolution
  stage with three residual blocks, and one 8-channel decoder block.
- Conditioning: 32-dimensional sinusoidal time features mapped to 52 dimensions,
  added to a learned ten-class embedding in every residual block.
- Schedule: 1,000-step linear VP schedule, beta `1e-4` through `2e-2`.
- Loss: unweighted pixel-mean epsilon MSE on the single image in model space
  `[-1,1]`; timesteps and Gaussian noise are sampled independently.
- Training: 10,000 updates, batch 128, Adam at `2e-4`, gradient norm cap 1,
  EMA `0.999`, validation every 250 updates on a fixed independent noise bank.
- Sampling: 16 independent Gaussian starts. Zero, oracle, and learned rows reuse
  the same starts and reverse-noise seeds. Anchors are saved after 0, 250, 500,
  750, and 1,000 reverse steps.

The existing 1,378,593-parameter full-data conventional DDPM is a contextual
positive benchmark, not a paired baseline: it answers a different dataset-level
question. This experiment isolates small-model one-image feasibility.

## Metrics and gates

Primary objective metrics are per-start endpoint squared error and correlation to
the source image, learned improvement over zero, and the saved images themselves.
Validation epsilon MSE is a mechanism diagnostic, not a substitute for sampling.

Execution/integrity gate:

```text
Gate type: execution/integrity
Downstream action or claim controlled: attribution of sampling failure to the learner
Exact proposition tested: the analytic epsilon oracle composes through the same reverse sampler
Why necessary: a broken schedule, orientation, or posterior makes learner evaluation uninterpretable
Statistic and independent unit: maximum endpoint model-space MSE over fixed paired paths, both forward-terminal horizons and independent-prior starts
Pass condition: maximum oracle MSE <= 1e-6 and all saved oracle starts, endpoints, and trajectory anchors finite
Failure means: repair schedule/orientation/composition before interpreting the learner
Failure does not mean: the tiny architecture or epsilon objective failed
Pass action: execute and interpret the paired zero/oracle/learned comparison
Fail action: a preflight failure stops before training; a post-training independent-prior failure marks the run invalid for learner attribution while preserving every output
Ambiguous/invalid action: rerun only after localizing the implementation defect
```

Diagnostic thresholds (non-blocking): at each forward reconstruction horizon, all
four learned endpoints must be finite, at least three must beat zero, and median
relative MSE improvement must be positive. The complete-horizon result is routed
separately from any short-horizon success. For independent-prior sampling, median
learned endpoint MSE improvement over zero must be positive, median learned source
correlation at least `0.8`, and learned must beat zero on at least 12 of 16 starts.
Sampling and artifact writing continue whether these thresholds pass or fail.

## Outcome-to-action table

| Observation | Interpretation | Required next action |
|---|---|---|
| Oracle fails | Reverse composition is invalid | Repair the sampler or schedule; do not change the learner |
| Oracle passes; learned reconstruction and prior sampling work | Roughly 30k parameters suffice for one-image DDPM feasibility | Freeze the result as a system control; do not infer dataset-level generation |
| Reconstruction works; independent-prior sampling fails | Long-horizon or terminal-time weakness | Inspect saved anchors, then change time coverage or parameterization before width |
| Short reconstruction works; complete forward-terminal reconstruction and prior sampling fail | Accumulation, late-time coverage, or on-policy error | Localize by saved horizon before changing model width |
| Independent-prior sampling passes while complete forward-terminal reconstruction fails | Pairing, metric, or reconstruction-panel anomaly | Audit the reconstruction panel before treating prior sampling as evidence |
| Oracle passes; learned reconstruction and sampling both fail | Capacity or optimization is inadequate | Compare one prespecified larger model or x0 parameterization; do not claim DDPM impossibility |
| Validation improves but images do not | Proxy/task mismatch | Select future changes by objective-bearing samples, not epsilon MSE alone |

## Required artifacts

The run saves the source image, exact configuration and command, training history,
selected and final checkpoints, fixed validation bank summary, zero/oracle/learned
reconstruction controls, independent starts, all final samples, trajectory anchors,
contact sheets, metrics, `REPORT.md`, status, and a SHA-256 artifact manifest.
Failures and noise-like images are never suppressed.

## Resource budget and stop rule

```text
Expected wall time: 5--20 minutes on the target laptop GPU
Expected accelerator time: 5--15 minutes
Expected peak memory: below 1 GiB
Expected persisted storage: below 100 MiB
New source/test/artifact complexity: one small model, one runner, focused tests, this specification
Maximum budget before automatic stop: 10,000 updates; one model seed; no sweep
Scientific decision purchased: whether a ~30k conventional DDPM can complete the one-image task
Why a smaller experiment cannot answer it: the CPU smoke validates interfaces but cannot establish MNIST sampling behavior
```

## Commands

Fast deterministic CPU smoke:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_tiny_ddpm_one_image `
  --smoke --device cpu --runs-root runs\experiment13-tiny-ddpm-one-image `
  --run-name smoke
```

Production exploratory run:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_tiny_ddpm_one_image `
  --device cuda --data-root mnist_data `
  --runs-root runs\experiment13-tiny-ddpm-one-image `
  --run-name production-label3-image0
```

The production result can establish only one-image, one-seed feasibility under this
fixed model and schedule. It cannot establish diversity, generalization, a
dataset-level generator, DDPM superiority, or an Eulerian result.
