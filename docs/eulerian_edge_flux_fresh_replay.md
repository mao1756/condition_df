# Eulerian edge-flux factor-one fresh replay

## Purpose and research mode

This is an **exploratory, objective-bearing** compatibility replay. It asks one
decision-changing question:

> Does the hash-pinned historical global conservative edge-flux checkpoint produce
> recognizable, requested-class-consistent, noncollapsed MNIST images from 160 fixed
> fresh low-frequency starts when the new experiment retains exactly one endpoint per
> path and performs no candidate selection or replacement?

The latest objective-bearing result is the finalized K128 Jacobi pilot, not the older
DDPM calibration. It passed its numerical and oracle controls but produced a tiny
learned effect and no recognizable learned-prior images. Its prescribed route was the
materially different Experiment-10 replay. Therefore:

```text
Proxy-only patches since the last objective-bearing experiment: 0
```

The replay moves away from the local Jacobi/Rao--Blackwell learner to a global
Poisson/OT edge-flux architecture. It does not leave the fixed-grid Eulerian program.

## Scope and terminology

The new evidence is **factor-one and candidate-selection-free**:

- exactly 160 declared paths, 16 per requested digit;
- exactly 160 learned endpoints retained;
- no oversized candidate pool, ranking, replacement, classifier selection, or shape
  selection;
- no training, fine-tuning, checkpoint search, gain search, or evaluator tuning.

The historical checkpoint configuration still contains
`sample_rejection_factor=4` and `sample_selection_metric="composite"`. Those bytes are
part of the authenticated historical configuration and must not be rewritten. They
are inert in this standalone runner. The new factor-one/no-selection rule is a
separate experiment policy recorded in the run configuration and verifier.

Do not call the replay generically "rejection-free." The adaptive numerical
integrator deliberately rejects an outer-step attempt and retries it with 2 or 4
substeps when clipping is too large. Those retry counts are mechanism telemetry, not
candidate rejection.

## Frozen authorities

- Legacy checkpoint: 13,947,413 bytes, SHA-256
  `8be77d1701887522f86099673431a928ad7dd2d350a06f7a94ade5c30a658cc3`.
- MNIST ARFF: SHA-256
  `418c0a60d2b4abc95db2e2bbf676f3af93ddaf18f79ba3f640624ab57007fb4b`.
- Current sampler source: `mnist/eulerian_flux_mnist.py`, SHA-256
  `4dca1c40f25eb04b3d615bd0094891c7cedb8cea8a673607eb02e1ab977e4f19`.
- Current benchmark source: `mnist/mnist_generation_benchmark.py`, SHA-256
  `2ebf13e37e03646222b8decde91034f10dff564fdc0d3f7a967af789ef3cbfd6`.
- Frozen evaluator checkpoint: 99,755 bytes, SHA-256
  `3d31d42a14fee0ecc72adc1644718a037cc48e649948407da6c272b981840c92`.
- Finalized K128 authority: manifest SHA-256
  `e6e25b297cf5e407fa0dcfdfd06755db56fc57bf724720723c8ca88631115b7a`,
  tree digest
  `33f191ef2753b12f5dbe8365003cc5b312e4bd35764479809939ce5abe39e039`,
  terminal route `complete`, scientific route
  `v0_negative_pivot_experiment10`, and `full_scale_auto_launched=0`.

The legacy checkpoint is hash-gated before deserialization. It is loaded only with
`weights_only=True` inside the exact audited NumPy safe-global scope. The loader
rejects unexpected payload keys/types, tensor names/shapes/dtypes, nonfinite tensors,
config drift, or a non-strict model load. It never calls `weights_only=False` and
never broadens the allowlist after failure.

The immutable `factor-one-fresh-prior-v1` attempt stopped verifier-clean at
`checkpoint_extract` because it compared the order of PyTorch's set-backed
safe-global registry. It performed no model construction, data parsing, rollout, or
CUDA allocation and produced no scientific result. The v2 hotfix compares the same exact authorities
as an unordered set; it changes no scientific configuration or experiment policy.

## Immutable v2 machine evidence and verifier defect

The immutable `factor-one-fresh-prior-v2-safe-globals` run completed every machine
stage through `machine_terminalization`. It sealed the teacher, null, and learned
160-image populations, scored them, prepared the 80-sample blinded review bundle,
and passed execution/integrity Gates A through D. Gate E remains pending: no human
answers or `outcome.json` exist, so v2 has no final route and is not a scientific
negative result.

The writer correctly retained the configured 30.0-second reserve for the
non-terminal `population_seal_and_scoring` and `review_prepare` operations. The
stale verifier instead expected 0.0 seconds for both, after which the run wrapper
recorded the verifier false negative as `integrity_failed`. A second read-only audit
found that evaluation verification then replayed CUDA-produced logits on CPU rather
than on the scoring device bound in the run configuration. Exact saved-versus-replay
logit authority must use that bound device and exact array comparison.

The v3 changes are verifier-only: they correct reserve replay and device-bound exact
evaluator replay without changing the checkpoint, starts, sampler, controller,
scientific configuration, evaluator, metrics, or gates. The v2 tree remains
immutable, and neither this note nor the v3 command claims that v2 was recovered or
that v3 was run. Its exact five-source closure and sealed-tree bindings are preserved
at
`handoff/source_snapshots/factor-one-fresh-prior-v2-safe-globals-52610de7/`.

## Data roles and fixed raster

The runner uses a strict local ARFF prefix parser:

- training rows `[0, 55000)` provide only the global raster authority;
- validation rows `[55000, 60000)` provide deterministic teacher targets;
- terminal rows `[60000, 70000)` are unavailable until all generated populations
  have been sealed;
- whole-file SHA-256 reads are authority-only and are not content parsing.

The raster scale is derived from the authenticated **training slice only**. The two
central integer pixel sums must be exactly `25470` and `25472`, giving

```text
numerator   = 25471
denominator = 255
float64     = 99.88627450980393
float hex   = 0x1.8f8b8b8b8b8b9p+6
```

Every mass image `m` is rendered by

```python
np.rint(255.0 * np.clip((25471.0 / 255.0) * m.astype(np.float64), 0.0, 1.0)).astype(np.uint8)
```

The verifier recomputes every uint8 array from saved raw masses. Per-image maximum,
percentile, or autocontrast normalization is forbidden.

The teacher bank keeps two distinct uint8 authorities. `source_images_uint8` holds
the exact selected validation pixels before mass normalization. The transport target
is the normalized floating-point `masses` array; `rendered_images_uint8` is exactly
`mass_to_uint8(masses)` under the global raster authority. Gate D measures the
teacher trajectory against `masses` and uses the rendered form for image health. It
does not silently substitute the original validation raster for the normalized mass
target.

## Fixed paths, rows, and dynamics

Path IDs are `efr-v1-000` through `efr-v1-159` in class-major order. Path `i` has
label `i // 16` and source seed `0xE14F1000 + i`. Every source is sampled independently
from the frozen label-independent 7-by-7 low-frequency law and sealed before any row
runs. Rows share the exact sealed start bank but use separate batch-level simulator
roots:

- teacher: `0xE14F3001`;
- null: `0xE14F2001`;
- learned: `0xE14F4001`.

Because adaptive retries consume new random draws, rows are not common-random-number
pairs. Comparisons are aggregate; no path-paired stochastic effect is reported.

All rows use 256 outer steps and save completed-step anchors at
`0, 64, 128, 192, 256`:

1. **Teacher:** recomputes target transport flux at every attempted substep, subtracts
   the configured free drift contribution, then passes the result through the same
   free/noise/limiter/retry interface. It is a known-positive transport control, not
   an exact h-transform.
2. **Null:** supplies an exact zero conditioning flux and retains the same configured
   free drift and stochastic noise.
3. **Learned:** calls the strict-loaded model as
   `predict_flux(tau, state, labels, source_masses=starts)` so the initial source
   remains persistent conditioning. Teacher targets are not accepted by this API.

For the accepted attempt of each outer step, telemetry records conditioning, free,
expected-noise and realized-state-increment RMS, clipping counts, substeps, mass
health, elapsed time, and CUDA allocation. Discarded attempts have separate
retry/clipping summaries and do not silently weight accepted-attempt mechanism
averages.

## Evidence firewall and review

The runner seals checkpoint, source, target, transform, path, seed, raw-mass, anchor,
telemetry, and uint8 authorities before loading evaluator weights, parsing terminal
rows, computing machine metrics, or creating a review key. Generation cannot run
after the terminal-test open event.

Machine scoring is descriptive and separate for teacher, null, and learned rows.
Human review is primary. The fixed blind review contains 40 learned and 40 matching
null endpoints: within-class indices `0, 5, 10, 15` for every digit. Review identity
is concealed until answers are recorded. The evaluator is secondary corroboration;
its historical 97% qualification target is not an execution veto here.

The machine phase ends at `awaiting_human_review`. It does not finalize a scientific
route before the manual review, and it never launches a follow-up experiment.

## Gates and outcome routing

- **Gate A - execution/integrity:** exact source, data, checkpoint, package, config,
  K128, DDPM evaluator, safe-load, strict-load, and sealed-inventory authorities.
- **Gate B - execution/integrity:** 160 fixed starts and 160 retained endpoints per
  row; factor one; no candidate selection/replacement; learned/null target firewall.
- **Gate C - execution/integrity:** deterministic restart excluding nondeterministic
  wall-time/allocator values, finite nonnegative unit-mass states, complete retry
  telemetry, resource compliance, and semantic artifact verification.
- **Gate D - execution/integrity:** the full-interface teacher must materially
  approach its fixed targets and pass its endpoint criterion. This gate controls
  learner attribution: failure localizes the shared
  controller/integrator/limiter/rendering interface and blocks that attribution.
- **Gate E - diagnostic threshold:** human recognizability and requested-label
  agreement are primary; classifier agreement, uniqueness, diversity, and aggregate
  learned-minus-null differences guide the next action. It is not a confirmatory
  gate.

Prespecified actions:

| Observation | Required action |
|---|---|
| Authority, firewall, determinism, resource, or artifact failure | Preserve evidence and repair only the localized engineering defect; replay unchanged if still scientifically identical |
| Teacher fails | Repair the shared interface; do not blame or retrain the learner |
| Human and evaluator are positive, learned exceeds null | Freeze one-checkpoint exploratory feasibility; do not scale or tune automatically |
| Human positive, evaluator negative | Preserve the human-positive result if attribution is clean; record evaluator/render disagreement; no evaluator tuning or automatic scale-up |
| Evaluator positive, humans call outputs noise/ambiguous | Treat image feasibility as negative unless an independent raster defect invalidates the run |
| Teacher passes, learned result is healthy but negative | Stop this historical checkpoint line; recommend a separately planned and approved materially different fixed-grid/on-policy architecture, or stop |
| Null is unexpectedly target-like or learned does not exceed null | Audit leakage/raster attribution; do not credit the checkpoint |

A weighted-point DSM may be mentioned only as a separately authorized Lagrangian
alternative. It is not the same fixed-grid objective and is never launched
automatically by this runner.

## Resource authority

The frozen maximums are 240 active seconds, 100 MiB persisted storage, and 75% of
visible CUDA memory. Lower user-approved caps are allowed; higher values are rejected.
The ledger prospectively prices each nontrivial quantum and persistence step, retains
the last valid evidence on a stop, and charges rendering, evaluator work, review
terminalization, reporting, and sealing. A failed projection does not silently change
batch size, path count, steps, precision, or device.

Warm-up and probe work counts against approved active time even when the warm-up is
excluded from the rate estimate. CUDA timing synchronizes at the start and end of
each outer step so the recorded elapsed time covers that step's queued device work.
The preflight times fixed eight-step probes and projects the complete workload; each
production row is then admitted and charged in eight-step quanta.

Restart authority is append-only in `restart_history.json`; it does not rewrite
`config.json` or `START_BANK_SEALED.json`. Before population sealing, re-entry clears
only unsealed row, telemetry, and image outputs and reruns all three rows from their
original roots--it never resumes a partial row. The resource governor rehydrates the
cumulative ledger rather than resetting it. After `POPULATIONS_SEALED.json` exists,
re-entry cannot regenerate samples and may continue only scoring, review preparation,
or authenticated terminalization recovery. Atomic replacement uses the repository's
bounded Windows retry pattern so transient `WinError 5` does not corrupt an authority
file.

## Commands

These commands describe the next fresh v3 interface. They are not evidence that v3
was launched. CUDA execution requires a fresh explicit approval.

Run the bounded test-only CPU smoke first. It takes no external inputs, creates no
run directory or production artifacts, never opens CUDA, and uses the frozen seed
`0xE14FF001`:

```powershell
.\.venv\Scripts\python.exe -B -m mnist.diag_d0_eulerian_edge_flux_replay smoke
```

It emits a repeatable JSON receipt for the synthetic null/teacher composition and
exits nonzero if its deterministic or numerical assertions fail.

```powershell
$RunDir = '.\runs\experiment15-eulerian-edge-flux-replay\factor-one-fresh-prior-v3-resource-device-verifier'

.\.venv\Scripts\python.exe -B -m mnist.diag_d0_eulerian_edge_flux_replay run `
  --run-dir $RunDir `
  --legacy-checkpoint '.\runs\experiment10\20260601-035019_10q-wide-repeat\experiment10_direct_flux_mnist.pt' `
  --ddpm-run-dir '.\runs\experiment13-conventional-ddpm\pixel-ddpm-calibration-v1-cpu-recovered' `
  --k128-run-dir '.\runs\experiment14-eulerian-jacobi-ddpm\candidate-k128-objective-pilot-v2-rng-aligned' `
  --arff '.\mnist_data\mnist_784.arff' `
  --device 'cuda:0' `
  --approval-id '<fresh-user-approval-id>' `
  --max-active-seconds 240 `
  --max-storage-mib 100 `
  --max-cuda-fraction 0.75
```

After completing the blind CSV outside the sealed run tree:

```powershell
.\.venv\Scripts\python.exe -B -m mnist.diag_d0_eulerian_edge_flux_replay record-review `
  --run-dir $RunDir `
  --answers '.\human_review_answers_edge_flux_v1.csv' `
  --reviewer '<reviewer-id>' `
  --confirm-manual-review
```

Read-only verification:

```powershell
.\.venv\Scripts\python.exe -B -m mnist.diag_d0_eulerian_edge_flux_replay verify `
  --run-dir $RunDir
```

## Claim boundary

A positive result supports only one-checkpoint, one-source-law exploratory fixed-grid
edge-flux feasibility without candidate selection. A healthy negative establishes
only that this pinned checkpoint/current sampler/fixed raster did not meet the frozen
image markers.

Neither result establishes an exact Doob transform, continuum consistency,
population performance, superiority to the DDPM, or success/failure of Eulerian
generation in general. No production run or successor experiment is authorized by
implementation alone.
