# condition_df

Research code for conditioning measure-valued diffusions on Wasserstein space,
with small numerical examples and MNIST weighted-point-cloud experiments.

## Active D0 route: global-dilated Jacobi/RB rollout

The active D0 route keeps the certified fixed-grid Jacobi transition and raw
Rao--Blackwell tangent target, but uses a 34,974-parameter circular dilated model
whose receptive field spans the 28x28 torus. On one fresh source-forward path, its
gain-1 controller improved an exact paired 128-step reverse suffix over zero by
7.454%; the source-informed control improved 98.211%, while both signs of the older
local frequency-one controller were adverse. This is exploratory one-image evidence,
not complete generation or a population claim.

The result, claim boundary, and next exact complete-path experiment are in
[`docs/jacobi_rb_global_dilated_rollout.md`](docs/jacobi_rb_global_dilated_rollout.md)
and [`HANDOFF.md`](HANDOFF.md). The implemented entry point is:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_global_dilated_rollout --help
```

The successful run is immutable and must not be resumed with changed source. The
next patch is a fresh child that completes the same path through forward step 511
and compares exact zero/global/source-informed full reverse trajectories. D0-v0,
D0-v1, and earlier boundary-tangent gates remain historical evidence rather than
the current scientific route.

## Layout

- `core/`: shared numerical helpers and manuscript-level algorithms.
- `examples/`: small h-transform examples used by the early notebooks, plus Example 9
  image-mass bridge helpers.
- `mnist/`: MNIST point-cloud conversion, MNIST-CP contour adapters,
  classifiers, guided diffusion, score matching, generation metrics, and
  experiment search utilities.
- `notebooks/`: interactive examples that import the package modules directly.
- `tests/`: lightweight regression/smoke checks.
- `docs/`: paper/reference notes and PDFs.
- `runs/experiment10/`: timestamped Experiment 10 outputs. Each invocation
  creates a fresh subdirectory, optionally tagged with `--run-name`.

## Imports

Use direct package imports:

```python
from core.wasserstein_conditioning_algorithms import simulate_gaussian_terminal_em
from examples.factorized_two_well_htransform import simulate_factorized_gaussian_mixture_em
from examples.eulerian_image_bridge import PositiveHeatPotentialCNN, simulate_conditioned_image_bridge
from mnist.score_matching import train_score_model
from mnist.mnist_cp import load_mnist_cp_splits
```

## Example 9: Eulerian image bridge

`notebooks/example_9_eulerian_house_to_butterfly.ipynb` demonstrates a
self-contained 28x28 house-to-butterfly experiment.  The helper module
`examples/eulerian_image_bridge.py` implements the numerically stable recipe:
train a positive CNN surrogate for the Feynman--Kac heat potential from free
Eulerian rollouts, then simulate the terminally conditioned conservative
edge-flux dynamics.

## Example 10e: MNIST direct edge-flux generation

`mnist/eulerian_flux_mnist.py` implements the laptop-friendly MNIST generation
experiment based on the Eulerian conditioning formula, but learns the two
conditioning-flux channels directly instead of learning a scalar heat potential.
The model is a small label-conditioned U-Net that predicts horizontal and
vertical edge fluxes, and the sampler applies them with conservative incidence
updates so total mass stays on the 28x28 simplex.

The default path is now **stable nearest-matched Poisson-flow with persistent source conditioning and on-policy correction**:

- sample a smooth low-frequency source measure;
- sample a digit label;
- assign each source to a same-label MNIST target by global nearest-neighbour matching in blurred low-resolution features;
- train on the minimum-energy two-channel periodic edge flux whose divergence equals the source-to-target velocity;
- after warmup, train some batches on states visited by the current sampler, using a residual corrective teacher;
- include a limiter-aware one-step loss so training sees the same edge clipping rule used during rollout.

The sampler receives the current mass image, bridge time, digit label, and the
initial random source/latent at every step. It still does not receive the target
MNIST image; the target image is used only to build the supervised training flux.

First sanity check: class-mean flow should produce blurry recognizable digits.

```powershell
.venv\Scripts\python.exe -m mnist.eulerian_flux_mnist `
  --data-root mnist_data `
  --run-name class-mean-smoke `
  --target-mode class-mean-flow `
  --source-mode lowfreq `
  --free-weight 0 `
  --noise-weight 0 `
  --train-steps 1500 `
  --batch-size 256 `
  --base-channels 32 `
  --sample-steps 96 `
  --num-samples 64
```

Then run the stable nearest-matched low-frequency source version:

```powershell
.venv\Scripts\python.exe -m mnist.eulerian_flux_mnist `
  --data-root mnist_data `
  --run-name poisson-ot-stable `
  --target-mode poisson-ot-flow `
  --source-mode lowfreq `
  --condition-on-source `
  --ot-match-mode nearest `
  --ot-nearest-top-k 1 `
  --velocity-target constant `
  --state-jitter-weight 0 `
  --on-policy-prob 0.25 `
  --on-policy-warmup-steps 1500 `
  --on-policy-prefix-steps 16 `
  --step-loss-weight 0.25 `
  --divergence-loss-weight 0.5 `
  --node-loss-weight 1.0 `
  --free-weight 0 `
  --noise-weight 0 `
  --train-steps 8000 `
  --batch-size 256 `
  --base-channels 32 `
  --sample-steps 256 `
  --num-samples 64
```

For a faster first check, use `--train-steps 3000 --sample-steps 192 --on-policy-prob 0.15 --on-policy-prefix-steps 8`.

A useful diagnostic upper-bound run is the coupled target-lowres prior. It does not give the full target image to the model, but during training it builds the source as a heavily downsampled/blurred version of the same target image. This distinguishes source-prior ambiguity from flux-rollout issues.

```powershell
.venv\Scripts\python.exe -m mnist.eulerian_flux_mnist `
  --data-root mnist_data `
  --target-mode poisson-flow `
  --source-mode target-lowres-prior `
  --condition-on-source `
  --velocity-target residual `
  --state-jitter-weight 0.05 `
  --free-weight 0 `
  --noise-weight 0 `
  --train-steps 3000 `
  --batch-size 256 `
  --base-channels 32 `
  --sample-steps 128 `
  --num-samples 64
```

The progress bar reports loss, divergence cosine, predicted/target RMS, step
loss, on-policy usage, mean-anchor probability, sample entropy, max pixel mass,
clipping fraction, and ETA. Every Experiment 10 invocation now creates a new
run directory under `runs/experiment10/`. Use `--run-name my-nickname` to tag
the folder; otherwise the folder is just timestamped. Final run outputs also
save `source_indices`, `source_labels`, `source_unique_count`,
`source_diversity_l2`, and source-label match diagnostics so source-prior
collapse is visible immediately. Training previews are saved under the run
folder, for example `runs/experiment10/<timestamp>_<name>/previews/`, every
`--preview-every` steps. The preview panel has rows for source, generated
sample, assigned target, exact teacher rollout, and class mean. To reproduce the
older mini-batch OT pairing, pass `--ot-match-mode minibatch`; to sample among
several stable neighbours, use `--ot-match-mode topk --ot-nearest-top-k K`. To
reproduce the older independent random pairing, pass `--target-mode poisson-flow`.
To ablate the persistent source channel, pass `--no-condition-on-source`. To
reproduce the older terminal-score proxy, pass `--target-mode terminal-score
--free-weight 1 --noise-weight 1`.
### Experiment 10g: stochastic-aware conditioning flux

The deterministic learned-only sampler is useful for debugging, but the theory
uses the free harmonic drift/noise plus a conditioning flux.  Experiment 10g
therefore adds stochastic-aware training flags.  With `--free-aware-target`, the
supervised Poisson flux is treated as a desired **total** transport flux and the
network is trained on the conditioning part

```text
J_theta ≈ J_total - free_weight * J_free.
```

The on-policy branch and one-step loss can also use the same free/noisy weights
as the sampler, so stochasticity is present during training rather than added
only after training.

Conservative first stochastic run:

```powershell
.venv\Scripts\python.exe -m mnist.eulerian_flux_mnist `
  --data-root mnist_data `
  --target-mode poisson-ot-flow `
  --source-mode lowfreq `
  --condition-on-source `
  --ot-match-mode nearest `
  --velocity-target constant `
  --free-aware-target `
  --sde-curriculum `
  --sde-ramp-steps 3000 `
  --target-free-weight 0.02 `
  --target-noise-weight 0.003 `
  --on-policy-use-free `
  --on-policy-use-noise `
  --stochastic-step-loss `
  --on-policy-prob 0.35 `
  --on-policy-warmup-steps 1500 `
  --on-policy-prefix-steps 16 `
  --step-loss-weight 0.25 `
  --adaptive-sampling `
  --clip-target 0.03 `
  --max-substeps 4 `
  --train-steps 8000 `
  --batch-size 256 `
  --base-channels 32 `
  --sample-steps 256 `
  --num-samples 64
```

The command above ramps the training free/noise weights from zero to the target
values.  Since `--free-weight` and `--noise-weight` are omitted, sampling uses
the same target values at the end of training.  The checkpoint and `.npz` now
record `learned_step_rms`, `free_step_rms`, `noise_step_rms`,
`free_to_learned_ratio`, and `noise_to_learned_ratio`; if either ratio is above
about 0.5, the stochastic terms are already comparable to the learned
conditioning increment.  Add `--save-ablation-samples` to save learned-only,
free-only, and noise-only preview grids from the same initial sources.

The mobility/free-SDE parameter can now be made explicit:

```text
--edge-alpha-mode legacy  # old experiments: alpha_edge = --alpha
--edge-alpha-mode grid    # theory-style 2D grid: alpha_edge = beta / grid_size^2
```

`grid` mode is opt-in because it is much stiffer on a 28x28 grid and may need
smaller free/noise weights or more substeps.

### Experiment 10h: stochastic rollout sharpening and anti-checkerboard controls

Experiment 10h keeps the stochastic-aware conditioning target from 10g and adds
the fixes for the current artifacts: resize-conv upsampling instead of transposed
convolutions, projected/minimum-energy flux parameterization to remove useless
curl components, full-horizon on-policy prefixes, mixed late residual targets,
multi-step rollout consistency, image-gradient sharpening, and small
anti-checkerboard/curl regularizers.

Recommended first 10h stochastic run:

```powershell
.venv\Scripts\python.exe -m mnist.eulerian_flux_mnist `
  --data-root mnist_data `
  --target-mode poisson-ot-flow `
  --source-mode lowfreq `
  --condition-on-source `
  --ot-match-mode nearest `
  --free-aware-target `
  --sde-curriculum `
  --sde-ramp-steps 3000 `
  --target-free-weight 0.015 `
  --target-noise-weight 0.002 `
  --velocity-target mixed `
  --late-residual-fraction 0.25 `
  --late-residual-prob 0.50 `
  --on-policy-use-free `
  --on-policy-use-noise `
  --on-policy-prob 0.40 `
  --on-policy-prefix-mode uniform `
  --on-policy-min-prefix-fraction 0.05 `
  --on-policy-max-prefix-fraction 0.85 `
  --on-policy-batch-size 64 `
  --rollout-loss-weight 0.15 `
  --rollout-loss-steps 8 `
  --rollout-loss-batch-size 64 `
  --image-grad-loss-weight 0.05 `
  --upsample-mode resize-conv `
  --flux-parameterization projected `
  --curl-loss-weight 0.01 `
  --checkerboard-loss-weight 0.001 `
  --stochastic-step-loss `
  --adaptive-sampling `
  --clip-target 0.03 `
  --max-substeps 4 `
  --train-steps 8000 `
  --batch-size 256 `
  --base-channels 32 `
  --sample-steps 256 `
  --num-samples 64 `
  --save-ablation-samples
```

The saved `.npz` includes artifact diagnostics: `sample_total_variation`,
`sample_checkerboard_energy`, and `sample_highfreq_fraction`.  The ablation grids
are now named `conditioning_only`, `free_plus_conditioning_no_noise`,
`full_stochastic`, `free_only`, and `noise_only`; under `--free-aware-target`,
`conditioning_only` is not expected to be a faithful sample because the model is
trained as a correction to the free SDE.

### Experiment 10j: faster replay cache and safer on-policy targets

Experiment 10j keeps the stochastic/free-aware setup and improves the 10i replay cache.
The default cache mode is now `trajectory`: one rollout stores many trajectory snapshots,
so refreshes are much cheaper than independently rolling every cached state from the source.
Replay targets use `--on-policy-target-mode safe-residual`, which clips residual corrections
relative to the constant source-target velocity and avoids pulling the model toward overly
aggressive mid-trajectory targets.  EMA weights can be used for cache refreshes and final
sampling through `--use-ema-for-cache` and `--use-ema-for-sampling`.

The main teacher-forced loss still skips the FFT projection when `--flux-parameterization projected`
is active; projection is used for sampler-consistency losses and generation.  Use
`--save-process-figure` to save trajectory grids from the initial source distribution to the
terminal digit samples.

Recommended laptop-friendly 10j run:

```powershell
.venv\Scripts\python.exe -m mnist.eulerian_flux_mnist `
  --data-root mnist_data `
  --target-mode poisson-ot-flow `
  --source-mode lowfreq `
  --condition-on-source `
  --ot-match-mode nearest `
  --free-aware-target `
  --sde-curriculum `
  --sde-ramp-steps 3000 `
  --target-free-weight 0.015 `
  --target-noise-weight 0.002 `
  --velocity-target mixed `
  --late-residual-fraction 0.25 `
  --late-residual-prob 0.50 `
  --on-policy-use-free `
  --on-policy-use-noise `
  --on-policy-mode replay `
  --on-policy-cache-mode trajectory `
  --on-policy-cache-size 2048 `
  --on-policy-cache-refresh-interval 100 `
  --on-policy-cache-rollout-batch-size 128 `
  --on-policy-cache-snapshots-per-traj 16 `
  --on-policy-target-mode safe-residual `
  --on-policy-residual-max-ratio 1.5 `
  --ema-decay 0.999 `
  --use-ema-for-sampling `
  --use-ema-for-cache `
  --rollout-loss-weight 0.15 `
  --rollout-loss-steps 6 `
  --rollout-loss-batch-size 64 `
  --rollout-loss-every 2 `
  --rollout-image-grad-loss-weight 0.03 `
  --upsample-mode resize-conv `
  --flux-parameterization projected `
  --curl-loss-weight 0.01 `
  --checkerboard-loss-weight 0.001 `
  --stochastic-step-loss `
  --adaptive-sampling `
  --clip-target 0.03 `
  --max-substeps 4 `
  --train-steps 8000 `
  --batch-size 256 `
  --base-channels 32 `
  --sample-steps 256 `
  --num-samples 64 `
  --save-ablation-samples `
  --save-process-figure
```

The process output is saved as `experiment10_diffusion_process.png`,
`experiment10_diffusion_marginal_process.png`, and
`experiment10_diffusion_process.npz`.

### Experiment 10k: terminal classifier diagnostics and endpoint refinement

Experiment 10k keeps the 10j stochastic/free-aware replay setup and adds a small
LeNet-style MNIST classifier as an optional terminal diagnostic and training loss.
The classifier is trained/cached locally on the same normalized MNIST measures and
is used only as a terminal-label score surrogate; generation still receives only a
label and a random/coarse source, never the target image.  The patch also adds
terminal-biased replay snapshots, endpoint L2/BCE/TV losses on rollout endpoints,
classifier-confidence diagnostics, optional classifier-based candidate selection,
and automatic analysis of `samples_goodbad.txt` when present in the output folder.

Recommended 10k raw-quality run:

```powershell
.venv\Scripts\python.exe -m mnist.eulerian_flux_mnist `
  --data-root mnist_data `
  --target-mode poisson-ot-flow `
  --source-mode lowfreq `
  --condition-on-source `
  --ot-match-mode nearest `
  --free-aware-target `
  --sde-curriculum `
  --sde-ramp-steps 3000 `
  --target-free-weight 0.015 `
  --target-noise-weight 0.002 `
  --velocity-target mixed `
  --late-residual-fraction 0.25 `
  --late-residual-prob 0.50 `
  --on-policy-use-free `
  --on-policy-use-noise `
  --on-policy-mode replay `
  --on-policy-cache-mode trajectory `
  --on-policy-cache-size 2048 `
  --on-policy-cache-refresh-interval 100 `
  --on-policy-cache-rollout-batch-size 128 `
  --on-policy-cache-snapshots-per-traj 16 `
  --on-policy-cache-terminal-fraction 0.35 `
  --on-policy-cache-terminal-min-tau 0.02 `
  --on-policy-cache-terminal-max-tau 0.18 `
  --on-policy-target-mode safe-residual `
  --on-policy-residual-max-ratio 1.5 `
  --ema-decay 0.999 `
  --use-ema-for-sampling `
  --use-ema-for-cache `
  --rollout-loss-weight 0.15 `
  --rollout-loss-steps 6 `
  --rollout-loss-batch-size 64 `
  --rollout-loss-every 2 `
  --rollout-image-grad-loss-weight 0.03 `
  --rollout-endpoint-l2-weight 0.05 `
  --rollout-endpoint-bce-weight 0.01 `
  --rollout-endpoint-tv-weight 0.005 `
  --use-classifier-diagnostics `
  --classifier-loss-weight 0.04 `
  --classifier-train-epochs 2 `
  --upsample-mode resize-conv `
  --flux-parameterization projected `
  --curl-loss-weight 0.01 `
  --checkerboard-loss-weight 0.001 `
  --stochastic-step-loss `
  --adaptive-sampling `
  --clip-target 0.03 `
  --max-substeps 4 `
  --train-steps 8000 `
  --batch-size 256 `
  --base-channels 32 `
  --sample-steps 256 `
  --num-samples 64 `
  --save-ablation-samples `
  --save-process-figure
```

For a presentation-style grid, add:

```powershell
  --sample-rejection-factor 4 `
  --sample-selection-metric classifier-confidence
```

This saves `experiment10_samples_raw.png` for the unselected candidates and
`experiment10_samples.png` for the selected grid.  The raw grid should still be
used when measuring actual generator quality.

### Experiment 10n: terminal shape diagnostics and composite sample selection

Experiment 10n treats the classifier as a diagnostic and optional weak terminal
signal, not as the main quality objective.  It adds classwise shape statistics
from real MNIST masses and uses them to penalize the measured failure mode of
bad samples: high entropy, low total variation, and low peak mass near the
terminal endpoint.  The terminal shape loss is gated by the same terminal-time
mask as the 10l endpoint losses.

New options include:

```text
--terminal-shape-loss-weight
--terminal-shape-entropy-weight
--terminal-shape-tv-weight
--terminal-shape-maxmass-weight
--classifier-loss-mode off|terminal|low-confidence-terminal
--classifier-loss-confidence-threshold
--sample-selection-metric composite
--selection-classifier-weight
--selection-entropy-weight
--selection-tv-weight
--selection-maxmass-weight
--selection-checkerboard-weight
```

Recommended raw-quality run keeps classifier loss off and adds only terminal
shape loss:

```powershell
.venv\Scripts\python.exe -m mnist.eulerian_flux_mnist `
  --data-root mnist_data `
  --run-name 10n-terminal-shape-raw `
  --target-mode poisson-ot-flow `
  --source-mode lowfreq `
  --condition-on-source `
  --ot-match-mode nearest `
  --free-aware-target `
  --sde-curriculum `
  --sde-ramp-steps 3000 `
  --target-free-weight 0.015 `
  --target-noise-weight 0.002 `
  --velocity-target mixed `
  --on-policy-use-free `
  --on-policy-use-noise `
  --on-policy-mode replay `
  --on-policy-cache-mode trajectory `
  --on-policy-cache-terminal-fraction 0.35 `
  --on-policy-target-mode safe-residual `
  --ema-decay 0.999 `
  --use-ema-for-sampling `
  --use-ema-for-cache `
  --rollout-loss-weight 0.15 `
  --rollout-loss-steps 6 `
  --rollout-image-grad-loss-weight 0.03 `
  --rollout-endpoint-l2-weight 0 `
  --rollout-endpoint-bce-weight 0 `
  --rollout-endpoint-tv-weight 0 `
  --terminal-shape-loss-weight 0.03 `
  --terminal-shape-entropy-weight 1.0 `
  --terminal-shape-tv-weight 1.0 `
  --terminal-shape-maxmass-weight 0.5 `
  --use-classifier-diagnostics `
  --classifier-loss-mode off `
  --upsample-mode resize-conv `
  --flux-parameterization projected `
  --curl-loss-weight 0.01 `
  --checkerboard-loss-weight 0.001 `
  --stochastic-step-loss `
  --adaptive-sampling `
  --train-steps 8000 `
  --batch-size 256 `
  --base-channels 32 `
  --sample-steps 256 `
  --num-samples 64 `
  --save-ablation-samples `
  --save-process-figure
```

For presentation-quality grids, add:

```powershell
  --sample-rejection-factor 4 `
  --sample-selection-metric composite
```

The composite selector writes `experiment10_selection_report.csv`, and if
`samples_goodbad.txt` exists the run also writes `experiment10_goodbad_report.csv`.

### Experiment 10o: local terminal shape and gap diagnostics

Experiment 10o extends 10n with a blurred low-resolution local terminal shape
loss.  The new loss compares terminal rollout endpoints with the matched target
at low resolution and penalizes mass in classwise negative-space masks.  This is
intended to reduce recurring local topology failures such as closed-bottom 9s,
5-to-8 crossings, and melting strokes without reintroducing checkerboard.

New options include:

```text
--terminal-local-shape-loss-weight
--terminal-target-support-weight
--terminal-target-edge-weight
--terminal-negative-space-weight
--terminal-local-shape-size
--terminal-local-shape-blur-sigma
--sample-selection-metric composite-local
--selection-local-support-weight
--selection-local-edge-weight
--selection-negative-space-weight
```

Recommended first raw-quality run:

```powershell
.venv\Scripts\python.exe -m mnist.eulerian_flux_mnist `
  --data-root mnist_data `
  --run-name 10o-local-terminal-shape-raw `
  --target-mode poisson-ot-flow `
  --source-mode lowfreq `
  --condition-on-source `
  --ot-match-mode nearest `
  --free-aware-target `
  --sde-curriculum `
  --sde-ramp-steps 3000 `
  --target-free-weight 0.015 `
  --target-noise-weight 0.002 `
  --velocity-target mixed `
  --on-policy-use-free `
  --on-policy-use-noise `
  --on-policy-mode replay `
  --on-policy-cache-mode trajectory `
  --on-policy-cache-terminal-fraction 0.35 `
  --on-policy-target-mode safe-residual `
  --ema-decay 0.999 `
  --use-ema-for-sampling `
  --use-ema-for-cache `
  --rollout-loss-weight 0.15 `
  --rollout-loss-steps 6 `
  --rollout-image-grad-loss-weight 0.03 `
  --rollout-endpoint-l2-weight 0 `
  --rollout-endpoint-bce-weight 0 `
  --rollout-endpoint-tv-weight 0 `
  --terminal-shape-loss-weight 0.03 `
  --terminal-shape-entropy-weight 1.0 `
  --terminal-shape-tv-weight 1.0 `
  --terminal-shape-maxmass-weight 0.5 `
  --terminal-local-shape-loss-weight 0.02 `
  --terminal-target-support-weight 1.0 `
  --terminal-target-edge-weight 0.5 `
  --terminal-negative-space-weight 0.5 `
  --terminal-local-shape-size 14 `
  --terminal-local-shape-blur-sigma 0.7 `
  --use-classifier-diagnostics `
  --classifier-loss-mode off `
  --upsample-mode resize-conv `
  --flux-parameterization projected `
  --curl-loss-weight 0.01 `
  --checkerboard-loss-weight 0.001 `
  --stochastic-step-loss `
  --adaptive-sampling `
  --train-steps 8000 `
  --batch-size 256 `
  --base-channels 32 `
  --sample-steps 256 `
  --num-samples 64 `
  --save-ablation-samples `
  --save-process-figure
```

For presentation-quality grids, add:

```powershell
  --sample-rejection-factor 4 `
  --sample-selection-metric composite-local
```

The local diagnostics are saved in `experiment10_local_shape_report.csv`.

### Experiment 10p: one-sided gap/local diagnostics

Experiment 10p keeps 10n/10o diagnostics but changes the local terminal objective to avoid
smoothing valid stroke variation.  The old symmetric local-shape training loss remains available
but is not the recommended default.  New one-sided losses target missing support, extra support,
and explicit low-resolution gap mass near target strokes.  Negative-space diagnostics now support
a stricter high-quantile class mask.

New options include:

```text
--terminal-negative-space-mode mean|strict
--terminal-negative-space-threshold
--terminal-negative-space-temperature
--terminal-gap-loss-weight
--terminal-gap-threshold
--terminal-gap-dilate-radius
--terminal-missing-support-weight
--terminal-extra-support-weight
--terminal-extra-support-margin
--sample-selection-metric composite-gap
--selection-gap-weight
--selection-extra-support-weight
--save-eval-source-batch
--eval-source-batch-path
--eval-fixed-source-seed
```

Recommended first raw-quality run:

```powershell
.venv\Scripts\python.exe -m mnist.eulerian_flux_mnist `
  --data-root mnist_data `
  --run-name 10p-gap-diagnostics-raw `
  --target-mode poisson-ot-flow `
  --source-mode lowfreq `
  --condition-on-source `
  --ot-match-mode nearest `
  --free-aware-target `
  --sde-curriculum `
  --sde-ramp-steps 3000 `
  --target-free-weight 0.015 `
  --target-noise-weight 0.002 `
  --velocity-target mixed `
  --on-policy-use-free `
  --on-policy-use-noise `
  --on-policy-mode replay `
  --on-policy-cache-mode trajectory `
  --on-policy-cache-terminal-fraction 0.35 `
  --on-policy-target-mode safe-residual `
  --ema-decay 0.999 `
  --use-ema-for-sampling `
  --use-ema-for-cache `
  --rollout-loss-weight 0.15 `
  --rollout-loss-steps 6 `
  --rollout-image-grad-loss-weight 0.03 `
  --rollout-endpoint-l2-weight 0 `
  --rollout-endpoint-bce-weight 0 `
  --rollout-endpoint-tv-weight 0 `
  --terminal-shape-loss-weight 0.03 `
  --terminal-shape-entropy-weight 1.0 `
  --terminal-shape-tv-weight 1.0 `
  --terminal-shape-maxmass-weight 0.5 `
  --terminal-local-shape-loss-weight 0 `
  --terminal-gap-loss-weight 0.005 `
  --terminal-missing-support-weight 0.005 `
  --terminal-extra-support-weight 0.002 `
  --terminal-negative-space-mode strict `
  --terminal-negative-space-threshold 0.08 `
  --terminal-gap-threshold 0.12 `
  --terminal-gap-dilate-radius 1 `
  --use-classifier-diagnostics `
  --classifier-loss-mode off `
  --upsample-mode resize-conv `
  --flux-parameterization projected `
  --curl-loss-weight 0.01 `
  --checkerboard-loss-weight 0.001 `
  --stochastic-step-loss `
  --adaptive-sampling `
  --train-steps 8000 `
  --batch-size 256 `
  --base-channels 32 `
  --sample-steps 256 `
  --num-samples 64 `
  --save-ablation-samples `
  --save-process-figure
```

For presentation-quality grids, add:

```powershell
  --sample-rejection-factor 4 `
  --sample-selection-metric composite-gap
```

Use `--save-eval-source-batch` on one run and `--eval-source-batch-path <that npz>` on later
runs to compare models on identical source masses and labels.

### Experiment 10q: terminal-consistent local losses

Experiment 10q keeps the 10p gap/local diagnostics but fixes the timing issue in terminal losses.
Endpoint and local terminal losses are only applied to genuinely near-terminal rollout endpoints,
or after a capped rollout toward tau=0.  The replay cache is biased toward the last few percent of
the bridge, and an optional small terminal microbatch trains recurring hard labels without changing
the whole data stream.

New options include:

```text
--terminal-loss-mode fixed|near-terminal|to-terminal
--terminal-rollout-max-steps
--terminal-loss-every
--terminal-rollout-batch-size
--terminal-batch-prob
--terminal-batch-size
--terminal-tau-min-fraction
--terminal-tau-max-fraction
--hard-label-sampling
--hard-labels 2,5,6,7,9
--hard-label-prob
```

Recommended first raw-quality run:

```powershell
.venv\Scripts\python.exe -m mnist.eulerian_flux_mnist `
  --data-root mnist_data `
  --run-name 10q-terminal-consistent-raw `
  --target-mode poisson-ot-flow `
  --source-mode lowfreq `
  --condition-on-source `
  --ot-match-mode nearest `
  --free-aware-target `
  --sde-curriculum `
  --sde-ramp-steps 3000 `
  --target-free-weight 0.015 `
  --target-noise-weight 0.002 `
  --velocity-target mixed `
  --on-policy-use-free `
  --on-policy-use-noise `
  --on-policy-mode replay `
  --on-policy-cache-mode trajectory `
  --on-policy-cache-terminal-fraction 0.50 `
  --on-policy-cache-terminal-min-tau 0.00 `
  --on-policy-cache-terminal-max-tau 0.08 `
  --on-policy-target-mode safe-residual `
  --ema-decay 0.999 `
  --use-ema-for-sampling `
  --use-ema-for-cache `
  --rollout-loss-weight 0.15 `
  --rollout-loss-steps 6 `
  --rollout-image-grad-loss-weight 0.03 `
  --terminal-loss-mode near-terminal `
  --terminal-rollout-max-steps 16 `
  --terminal-loss-every 4 `
  --terminal-loss-tau-max-fraction 0.06 `
  --terminal-batch-prob 0.25 `
  --terminal-batch-size 64 `
  --terminal-tau-min-fraction 0.00 `
  --terminal-tau-max-fraction 0.06 `
  --terminal-shape-loss-weight 0.03 `
  --terminal-shape-entropy-weight 1.0 `
  --terminal-shape-tv-weight 1.0 `
  --terminal-shape-maxmass-weight 0.5 `
  --terminal-local-shape-loss-weight 0 `
  --terminal-gap-loss-weight 0.02 `
  --terminal-missing-support-weight 0.01 `
  --terminal-extra-support-weight 0.006 `
  --terminal-negative-space-mode strict `
  --terminal-negative-space-threshold 0.08 `
  --terminal-gap-threshold 0.12 `
  --terminal-gap-dilate-radius 1 `
  --hard-label-sampling `
  --hard-labels 2,5,6,7,9 `
  --hard-label-prob 0.35 `
  --use-classifier-diagnostics `
  --classifier-loss-mode off `
  --upsample-mode resize-conv `
  --flux-parameterization projected `
  --curl-loss-weight 0.01 `
  --checkerboard-loss-weight 0.001 `
  --stochastic-step-loss `
  --adaptive-sampling `
  --train-steps 8000 `
  --batch-size 256 `
  --base-channels 32 `
  --sample-steps 256 `
  --num-samples 64 `
  --save-ablation-samples `
  --save-process-figure
```

### Experiment 10r: terminal replay/target fix

Experiment 10r keeps the 10q terminal-consistent loss timing, but fixes two terminal-training details:

- trajectory replay snapshots now preserve duplicate terminal snapshot steps instead of de-duplicating them, so the actual cache terminal fraction should match `--on-policy-cache-terminal-fraction` much more closely;
- dedicated terminal microbatches can use their own target mode and rollout-to-zero behavior.

New/updated options:

```text
--terminal-target-mode mixed|constant|residual|safe-residual
--terminal-batch-rollout-mode fixed|to-zero
```

Recommended first raw-quality run:

```powershell
.venv\Scripts\python.exe -m mnist.eulerian_flux_mnist `
  --data-root mnist_data `
  --run-name 10r-terminal-target-safe-residual `
  --target-mode poisson-ot-flow `
  --source-mode lowfreq `
  --condition-on-source `
  --ot-match-mode nearest `
  --free-aware-target `
  --sde-curriculum `
  --sde-ramp-steps 3000 `
  --target-free-weight 0.015 `
  --target-noise-weight 0.002 `
  --velocity-target mixed `
  --on-policy-use-free `
  --on-policy-use-noise `
  --on-policy-mode replay `
  --on-policy-cache-mode trajectory `
  --on-policy-cache-terminal-fraction 0.50 `
  --on-policy-cache-terminal-min-tau 0.00 `
  --on-policy-cache-terminal-max-tau 0.08 `
  --on-policy-target-mode safe-residual `
  --terminal-target-mode safe-residual `
  --terminal-batch-rollout-mode to-zero `
  --ema-decay 0.999 `
  --use-ema-for-sampling `
  --use-ema-for-cache `
  --rollout-loss-weight 0.15 `
  --rollout-loss-steps 6 `
  --rollout-image-grad-loss-weight 0.03 `
  --terminal-loss-mode near-terminal `
  --terminal-rollout-max-steps 16 `
  --terminal-loss-every 4 `
  --terminal-loss-tau-max-fraction 0.06 `
  --terminal-batch-prob 0.25 `
  --terminal-batch-size 64 `
  --terminal-tau-min-fraction 0.00 `
  --terminal-tau-max-fraction 0.06 `
  --terminal-shape-loss-weight 0.03 `
  --terminal-shape-entropy-weight 1.0 `
  --terminal-shape-tv-weight 1.0 `
  --terminal-shape-maxmass-weight 0.5 `
  --terminal-local-shape-loss-weight 0 `
  --terminal-gap-loss-weight 0.03 `
  --terminal-missing-support-weight 0.005 `
  --terminal-extra-support-weight 0.008 `
  --terminal-negative-space-mode strict `
  --terminal-negative-space-threshold 0.08 `
  --terminal-gap-threshold 0.12 `
  --terminal-gap-dilate-radius 1 `
  --hard-label-sampling `
  --hard-labels 2,6,9 `
  --hard-label-prob 0.50 `
  --use-classifier-diagnostics `
  --classifier-loss-mode off `
  --upsample-mode resize-conv `
  --flux-parameterization projected `
  --curl-loss-weight 0.01 `
  --checkerboard-loss-weight 0.001 `
  --stochastic-step-loss `
  --adaptive-sampling `
  --train-steps 8000 `
  --batch-size 256 `
  --base-channels 32 `
  --sample-steps 256 `
  --num-samples 64 `
  --save-ablation-samples `
  --save-process-figure
```
### Experiment 10s: foreground recall and label-gated local losses

Experiment 10s keeps the stable 10q-style terminal timing and adds a conservative
foreground-recall loss for the remaining faded-stroke failures. Local topology
losses are now label-gated, and their combined contribution is capped so they do
not trade one class-specific artifact for another.

Useful new options:

```text
--terminal-foreground-recall-weight
--terminal-foreground-threshold
--terminal-foreground-temperature
--terminal-foreground-size
--terminal-foreground-blur-sigma
--terminal-gap-labels
--terminal-extra-support-labels
--terminal-foreground-labels
--terminal-local-loss-max-ratio
--selection-foreground-weight
```

A cautious run should keep classifier loss off and use 10q local-loss weights:

```powershell
  --terminal-gap-loss-weight 0.02 `
  --terminal-missing-support-weight 0.01 `
  --terminal-extra-support-weight 0.006 `
  --terminal-foreground-recall-weight 0.01 `
  --terminal-gap-labels 5,9 `
  --terminal-extra-support-labels 5,9 `
  --terminal-foreground-labels 2,3,6,9 `
  --terminal-local-loss-max-ratio 0.25 `
  --classifier-loss-mode off
```

### Experiment 11 / C0: weighted free-rollout innovation matching

Experiment 11 implements the Level-C/C0 endpoint-bridge training recipe.  It
simulates free finite-volume reference trajectories, weights each whole
trajectory by a soft terminal endpoint reward, and trains the network to predict
the weighted mean of the edge Brownian innovation.  The network output is an
edge Brownian-shift field `eta`; at generation time the sampler converts it to a
physical learned flux using the same reference noise scale used during cache
construction.

Run folders follow the Experiment 10 convention and are created under
`runs/experiment11/<timestamp>_<run-name>/`.  Each run stores metadata,
checkpoints, history/metrics, generated samples, and preview images.  Cache
construction, training, and sampling all use ETA progress bars unless
`--no-progress` is passed.

A default C0 run is:

```powershell
python -m mnist.experiment11_c0 `
  --run-name c0-weighted-innovation `
  --base-channels 48 `
  --train-steps 10000 `
  --batch-size 256 `
  --cache-paths 4096 `
  --cache-refresh-every 500 `
  --cache-batch-size 128 `
  --teacher-stride 8 `
  --time-slices-per-path 4 `
  --terminal-epsilon 0 `
  --terminal-ess-target 0.25 `
  --reference-free-weight 0.03 `
  --reference-noise-weight 0.005 `
  --sample-steps 256 `
  --num-samples 64
```

Use `--terminal-epsilon 0` to calibrate the terminal reward width from the free
rollout distances so that the effective sample size is close to
`--terminal-ess-target`.  Add `--save-cache-previews` if you want diagnostic
PNG grids of free endpoints and high-weight cached states at each cache refresh.
