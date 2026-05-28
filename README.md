# condition_df

Research code for conditioning measure-valued diffusions on Wasserstein space,
with small numerical examples and MNIST weighted-point-cloud experiments.

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
