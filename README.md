# condition_df

Research code for conditioning measure-valued diffusions on Wasserstein space,
with small numerical examples and MNIST weighted-point-cloud experiments.

## Layout

- `core/`: shared numerical helpers and manuscript-level algorithms.
- `examples/`: small h-transform examples used by the early notebooks.
- `mnist/`: MNIST point-cloud conversion, classifiers, guided diffusion,
  score matching, generation metrics, and experiment search utilities.
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
from mnist.score_matching import train_score_model
```

## Smoke Tests

Use the virtual environment Python on Windows:

```powershell
.venv\Scripts\python.exe -m tests.test_imports
.venv\Scripts\python.exe -m tests.test_smoke_wasserstein_conditioning_algorithms
.venv\Scripts\python.exe -m tests.test_smoke_mnist_conditioned_diffusion
.venv\Scripts\python.exe -m tests.test_smoke_mnist_score_matching
.venv\Scripts\python.exe -m tests.test_smoke_mnist_experiment6_hyperparameter_search
```

The system `python` launcher on this machine may resolve to the Windows Store
alias, so prefer `.venv\Scripts\python.exe` inside this repo.
