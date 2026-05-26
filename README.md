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
- `artifacts/patches/`: old patch files kept for reference, away from the
  importable source root.

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
clipping fraction, and ETA. Final artifacts also save `source_indices`,
`source_labels`, `source_unique_count`, `source_diversity_l2`, and
source-label match diagnostics so source-prior collapse is visible immediately.
Training previews are saved under `artifacts/experiment10_mnist_flux/previews/`
every `--preview-every` steps. The preview panel has rows for source, generated
sample, assigned target, exact teacher rollout, and class mean. To reproduce the
older mini-batch OT pairing, pass `--ot-match-mode minibatch`; to sample among
several stable neighbours, use `--ot-match-mode topk --ot-nearest-top-k K`. To
reproduce the older independent random pairing, pass `--target-mode poisson-flow`.
To ablate the persistent source channel, pass `--no-condition-on-source`. To
reproduce the older terminal-score proxy, pass `--target-mode terminal-score
--free-weight 1 --noise-weight 1`.

