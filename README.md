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
