# Codex /goal prompt for Experiment 11 overnight loop

Paste the block below into Codex from the repository root.

```text
/goal
You are working in the `condition_df` repository. I need an overnight Experiment 11 diagnose-and-iterate loop. Do not change the theoretical objective casually: the honest target is the Eulerian finite-volume Doob h-transform / weighted-innovation estimator, where the conditioned grid SDE keeps the same edge noise and only adds the heat-potential edge flux. Poisson may be used only as an explicit diagnostic or proposal; do not treat Poisson loss as the final honest objective.

Current situation:
- Experiment 11/C0, C1, C2, and C2.1 still generate mostly noise/blobs.
- C2.1 has positive `innovation_gain`, but `learned_step_rms / noise_step_rms` is tiny. That means the cache objective may be learned on training states but the sampler receives a weak or distribution-shifted control.
- Your job overnight is to run experiments, diagnose metrics and saved PNGs, make minimal patches only when diagnostics justify them, and leave a clear final report.

Before doing anything:
1. Run:
   `python -m py_compile mnist\experiment11_c0.py tools\experiment11_overnight_loop.py`
2. Run:
   `python tools\experiment11_overnight_loop.py diagnose --runs-root runs\experiment11`
3. Read `runs\experiment11\overnight_report.md`.

Then run the overnight suite:
`python tools\experiment11_overnight_loop.py run-suite --data-root mnist_data --train-steps 5000 --max-runs 4`

After each run, inspect:
- `runs/experiment11/overnight_report.md`
- latest `experiment11_c0_metrics.json`
- latest `experiment11_c0_history.json`
- latest `experiment11_c0_cache_diagnostics.csv`
- latest sample PNG and weighted terminal/target PNGs

Decision rules:
1. If `target_label_mismatch_fraction > 0`, stop and fix the target sampler immediately.
2. If `branch_centered_target_rms` is near zero or `branch_weighted_minus_unweighted_dist2` is near zero, the terminal reward is not producing a local score. Patch terminal reward/feature handling, not the optimizer.
3. If `innovation_gain > 0.25` but `learned_step_rms / noise_step_rms < 0.05`, the model is learning on cache states but the sampler control is too weak or off-distribution. First add generation/control-strength ablation from the same checkpoint; do not retrain just to test control strength.
4. If higher control strength produces digit-like structures, patch Experiment 11 to save control-strength ablations automatically and choose a calibrated sampling strength. Keep the training loss unchanged.
5. If higher control strength is still noise, implement the next honest estimator: a scalar heat-potential/value network for `log u_t(s)` with edge-gradient readout, or a local finite-difference value-estimator diagnostic. Do not increase Poisson supervised loss as the final answer.
6. If a patch is made, run `python -m py_compile mnist\experiment11_c0.py` and the Experiment 11 smoke test before continuing.

Always save artifacts under `runs/experiment11/<timestamp>_<run-name>/`. Do not overwrite previous runs. At the end, write `runs/experiment11/CODEX_OVERNIGHT_SUMMARY.md` with:
- commands run
- patches made
- best run and why
- failed hypotheses
- key metrics table
- image observations
- recommended next patch.
```

## Manual fallback commands

Diagnose existing runs only:

```powershell
python tools\experiment11_overnight_loop.py diagnose --runs-root runs\experiment11
```

Run default overnight suite:

```powershell
python tools\experiment11_overnight_loop.py run-suite `
  --data-root mnist_data `
  --train-steps 5000 `
  --max-runs 4
```

Use shorter smoke-size suite:

```powershell
python tools\experiment11_overnight_loop.py run-suite `
  --data-root mnist_data `
  --train-steps 1000 `
  --cache-paths 256 `
  --max-runs 2 `
  --num-samples 32
```
