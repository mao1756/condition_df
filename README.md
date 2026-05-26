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

## Example 10c: MNIST direct edge-flux generation

`mnist/eulerian_flux_mnist.py` implements the laptop-friendly MNIST generation
experiment based on the Eulerian conditioning formula, but learns the two
conditioning-flux channels directly instead of learning a scalar heat potential.
The model is a small label-conditioned U-Net that predicts horizontal and
vertical edge fluxes, and the sampler applies them with conservative incidence
updates so total mass stays on the 28x28 simplex.

The default path is now **OT-coupled Poisson-flow**:

- sample a smooth low-frequency source measure;
- sample a digit label;
- assign each source to a same-label MNIST target by a tiny per-class
  mini-batch optimal-transport problem in blurred low-resolution features;
- train on the minimum-energy two-channel periodic edge flux whose divergence
  equals the source-to-target velocity.

This keeps the direct two-channel flux object from Example 10b, but avoids the
independent random pairing that made the learned MSE flux average over
incompatible same-label target digits.  Sampling still receives only the current
mass image, bridge time, and digit label; the target image is used only to build
the supervised training flux.

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

Then run the OT-coupled stochastic-target version:

```powershell
.venv\Scripts\python.exe -m mnist.eulerian_flux_mnist `
  --data-root mnist_data `
  --target-mode poisson-ot-flow `
  --source-mode lowfreq `
  --ot-cost-mode lowres `
  --ot-lowres-size 7 `
  --ot-blur-sigma 1.0 `
  --mean-flow-prob 0.20 `
  --tau-sampling endpoint-mixture `
  --free-weight 0 `
  --noise-weight 0 `
  --train-steps 5000 `
  --batch-size 256 `
  --base-channels 32 `
  --sample-steps 128 `
  --num-samples 64
```

A useful diagnostic upper-bound run is the coarse class prior.  It does not give
the full target image, but it starts from a heavily downsampled/blurred same-class
source skeleton so you can distinguish source-prior issues from flux-rollout
issues.

```powershell
.venv\Scripts\python.exe -m mnist.eulerian_flux_mnist `
  --data-root mnist_data `
  --target-mode poisson-flow `
  --source-mode class-lowres-prior `
  --source-lowfreq-size 7 `
  --source-blur-sigma 1.25 `
  --free-weight 0 `
  --noise-weight 0 `
  --train-steps 3000 `
  --batch-size 256 `
  --base-channels 32 `
  --sample-steps 128 `
  --num-samples 64
```

The progress bar reports loss, divergence cosine, predicted/target RMS,
mean-anchor probability, sample entropy, max pixel mass, clipping fraction, and
ETA.  Training previews are saved under
`artifacts/experiment10_mnist_flux/previews/` every `--preview-every` steps.  The
preview panel has rows for source, generated sample, assigned target, exact
teacher rollout, and class mean.  To reproduce the older independent random
pairing, pass `--target-mode poisson-flow`.  To reproduce the older terminal-score
proxy, pass `--target-mode terminal-score --free-weight 1 --noise-weight 1`.

For a very quick smoke run, reduce `--train-steps` to 100--300. Outputs are
written to `artifacts/experiment10_mnist_flux/` by default.

## Smoke Tests

Use the virtual environment Python on Windows:

```powershell
.venv\Scripts\python.exe -m tests.test_imports
.venv\Scripts\python.exe -m tests.test_smoke_wasserstein_conditioning_algorithms
.venv\Scripts\python.exe -m tests.test_smoke_mnist_conditioned_diffusion
.venv\Scripts\python.exe -m tests.test_smoke_mnist_score_matching
.venv\Scripts\python.exe -m tests.test_smoke_mnist_cp
.venv\Scripts\python.exe -m tests.test_smoke_mnist_experiment6_hyperparameter_search
```

The system `python` launcher on this machine may resolve to the Windows Store
alias, so prefer `.venv\Scripts\python.exe` inside this repo.
