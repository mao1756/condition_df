# Read-only Jacobi/RB quartile direction adjudication

## Purpose and restricted scope

This additive workflow diagnoses why the frozen quartile-specialist checkpoint
family failed in forward-time quartiles `q1-q3`. It is a historical,
nonauthorizing adjudication, not a new learner. It first replays the sealed gain
and rank results exactly, then evaluates the already-sealed checkpoints on the
already-open gain and rank caches to separate directional alignment from
prediction energy.

The workflow is implemented by
`mnist.diag_d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication` and
has the restartable stages

```text
preflight -> replay -> decompose -> adjudicate -> report
```

It creates no Jacobi transition, label, optimizer update, checkpoint, evidence
role, path ID, seed, selected system, controller trajectory, reconstruction, or
sample. Every result is historical design evidence only.

## Immutable parent and provenance

The sole primary parent is

```text
runs/experiment12_d0_jacobi_rb_boundary_tangent_quartile_specialist/
  20260807-132351_production-exact-quartile-specialist
```

Preflight requires the terminal decision `no_training_only_quartile_system`, a
valid scientific negative, and the following immutable bindings:

| Artifact or identity | Required value |
|---|---|
| Registered artifacts | `4120` |
| Registry semantic SHA-256 | `e5f6b3ee257b3d4f86ec3ac54f4223540cf76caa24489d22e9c138a54e19c7bb` |
| Registry-file SHA-256 | `e24c7db28081dbceb8f0acf690d779f835379c82a89a2b263976c3e0b631f798` |
| Source fingerprint | `61a8c2fd6a317c05b9eed73e208d40b0cc6c01d6fdd227ae33d71d0be7c6027d` |
| Scientific-config semantic SHA-256 | `05263b7b01c2586e9a771bd71fe28fbb74d8e5d9da873ce4746019c5167c08c1` |
| `gain_table.npz` SHA-256 | `48ec1f17be4869f9a816c0338e8b23cddbdf44dd7000ca60fe317fd787925815` |
| `training_rank_path_tables.npz` SHA-256 | `93f5c4ea39bc658cc5f46b7d31930a0c8c02b2c7ccb106cf314229f0eec32d9b` |
| `training_checkpoint_index.json` SHA-256 | `6446cec12529f5634870c43eb349c3a43b9e1b64f0850c04c680b13c1c749d2b` |
| Checkpoint-index semantic SHA-256 | `c4112fa6c971bac1ca3b0da471c8915a955a1ee760529cd948530963e38e77c7` |
| `gain_calibration_seal.json` SHA-256 | `a165b1d3c601625ebd058cf67ee564ede751ec5356496c6fdb0c7c8e4094e189` |
| `rank_label_open.json` SHA-256 | `9eac05c28339202fafbcf5abdf00e4040679f9087ad929729dbd085736e6e1b6` |

All 492 checkpoint files, their indexed model-state hashes, the cache indexes,
stage seals, and the prerequisites in both role-open records must verify.
Preflight also rejects any `selection_open.json`, `confirmation_open.json`,
selection evidence, confirmation evidence, or selected-system artifact.

The complete parent tree is snapshotted by relative path, size, and SHA-256
before the first evidence read and after the final report. The snapshots must
match exactly.

## Parent artifact schemas

The canonical candidate order is quartile, seed, then update. The 480 nonzero
candidates use three seeds per quartile and updates `100,200,...,4000`. The
checkpoint index additionally contains the 12 update-zero controls, for 492
payloads total.

`gain_table.npz` contains:

| Array | Shape | Type | Meaning |
|---|---:|---|---|
| `quartile` | `[480]` | `int8` | Candidate quartile |
| `seed` | `[480]` | `int64` | Candidate seed |
| `update` | `[480]` | `int16` | Checkpoint update |
| `cross_term` | `[480]` | `float64` | Gain-role `C`; `NaN` for fixed-unit q0/q1 |
| `prediction_energy` | `[480]` | `float64` | Gain-role `P`; `NaN` for q0/q1 |
| `gain` | `[480]` | `float64` | Unit q0/q1 gain, eligible q2/q3 gain, or `NaN` |
| `eligible` | `[480]` | `uint8` | Frozen gain eligibility |
| `reason_code` | `[480]` | `int16` | Code bound by `gain_table.json` |

The sealed gain counts are 120 fixed-unit candidates in each of q0 and q1,
103 eligible plus 17 nonpositive-cross-term candidates in q2, and 37 eligible
plus 83 nonpositive-cross-term candidates in q3. The individual q2/q3 gain
records bind the candidate, `C`, `P`, scalar gain, sample count, reason, and a
semantic hash.

`training_rank_path_tables.npz` contains:

| Array | Shape | Type |
|---|---:|---|
| `candidate_quartile` | `[480]` | `int8` |
| `candidate_seed` | `[480]` | `int64` |
| `candidate_update` | `[480]` | `int16` |
| `path_ids` | `[32]` | `int64` |
| `per_path_pooled_improvement` | `[480,32]` | `float64` |
| `per_path_fine_cell_improvement` | `[480,32,7,8]` | `float64` |
| `per_path_fine_cell_count` | `[480,32,7,8]` | `int64` |
| `pooled_improvement` | `[480]` | `float64` |
| `fine_cell_improvement` | `[480,7,8]` | `float64` |
| `eligible` | `[480]` | `uint8` |

The companion rank CSV contains the frozen gain record and the pooled, seven
phase, eight midpoint, and 56 fine-cell improvements and reason code for each
candidate. It has 80 eligible q0 candidates; q1, q2, and q3 have none. The
deterministic q0 winner is `q0.seed261333.update1800`, with pooled improvement
`0.002227481269259224`.

The checkpoint index has 12 task rows and 492 checkpoint rows. Each checkpoint
row binds `candidate_key`, `checkpoint_path`, `checkpoint_file_sha256`, and
`model_state_sha256`. Every payload binds its quartile, seed, update, training
fingerprint, `state_dict`, `state_sha256`, optimizer state, batch cursor, CPU
and CUDA RNG states, and the `raw_target_unchanged=1` and `training_only=1`
flags. The adjudication loads only the model state.

Each evidence role has a pre-existing semantic role-open record:

- `gain_calibration`: paths `999680-999711`, opened against the training seal;
- `training_rank`: paths `999936-999967`, opened against the gain-calibration
  seal.

Each cache index has 32 paths, 256 shard entries, 57,344 input rows, 57,344
label rows, and separate input and label archives. Selected-shard input arrays
carry `sample_key`, `path_id`, `outer_step`, `phase`, `midpoint_index`,
`midpoint_fraction`, `later_full_state`, `reverse_time`, `color`, `duration`,
and `label`; label arrays carry the same row identity plus
`denoising_target[...,392]` and `certificate_codes[...,392]`. Existing cache
loaders verify artifact hashes, flatten branch arrays, stable-sort by unique
`sample_key`, and preserve C-order.

## Strict read-only evidence firewall

The child uses a dedicated `load_already_open_role` path. It requires and
verifies the existing role-open record and every prerequisite file hash before
opening the associated input and label stores. It refuses a missing record and
never calls a helper that can create or replace one.

Only `gain_calibration` and `training_rank` may be read. Gain and rank q0
evidence is a positive diagnostic control; scientific conclusions concern
q1-q3. Gain and rank path IDs are disjoint and are never paired path-by-path.
Fresh selection, untouched confirmation, bootstrap shards, historical
selection, and confirmation evidence remain forbidden.

The child scientific configuration repeats the exact physical contract but
sets all authorizations to zero, including cache generation, training,
selection, confirmation, controller execution, reconstruction, and sampling.
It records zero new paths, seeds, roles, transitions, labels, optimizer steps,
and checkpoint writes.

## Frozen scientific contract

The target remains

\[
\bar Z=y(1-y)\,\partial_y\log k_u(y\mid x)
\]

under certified binary64 Jacobi transitions, grid 28, `alpha=1`, `K=512`,
`tau_eff=5e-5`, and later-state-only model inputs. The checkpoint family is the
unchanged width-32 boundary-tangent family. Classifiers, quotient or clipped
targets, reverse residuals, Gaussian or Euler proxies, fitted future-data
controls, and phase- or midpoint-specific fitted targets remain forbidden.

## Exact replay

Replay reads only the sealed gain NPZ, rank NPZ, companion indexes and summary
CSV. It reproduces the 480-candidate order, all values and reason codes, gain
eligibility, rank eligibility, q0's deterministic winner, zero q1-q3 winners,
and the terminal `no_training_only_quartile_system` decision. It computes no
alternative threshold, screen, candidate, or gain.

## Quadratic decomposition

For every nonzero candidate, each already-open role, path, phase, midpoint,
and phase-by-midpoint cell, the workflow computes from the raw checkpoint
prediction `m`

\[
C=E[\bar Zm],\qquad P=E[m^2],\qquad
I(\lambda)=2\lambda C-\lambda^2P.
\]

When `C>0` and `P>0`, it also reports

\[
\lambda^*=C/P,\qquad I^*=C^2/P,
\]

without clipping or projection. The parent evaluation gain is one for q0/q1,
the frozen parent gain for eligible q2/q3, and diagnostic unit gain for
ineligible q2/q3. A separate nonauthorizing diagnostic applies each
candidate's pooled gain-role `C/P` algebraically to the disjoint rank-role
moments.

Direct risk improvement and `2*lambda*C-lambda^2*P` must agree within `5e-15`.
The reconstructed rank path and cell table must match the sealed table within
the same tolerance, and pooled q2/q3 gain-role `C`, `P`, and `C/P` must replay
the parent gain table. q0/q1 `C/P` values remain diagnostic only.

Predictions are made in batches of at most 32, converted to binary64 before
products, reduced in canonical C-order with `math.fsum`, and discarded. No
row-level prediction is persisted. The consolidated NPZ contains at least:

```text
candidate_quartile              [480]
candidate_seed                  [480]
candidate_update                [480]
role_code                       [2]
role_path_ids                   [2,32]
cross_term                      [480,2,32,7,8]
prediction_energy               [480,2,32,7,8]
raw_improvement                 [480,2,32,7,8]
parent_gain_improvement         [480,2,32,7,8]
diagnostic_gain_improvement     [480,2,32,7,8]
fine_cell_row_count             [480,2,32,7,8]
```

One reduction shard per candidate and role makes the 960 jobs restartable. A
valid shard is verified and skipped; a missing shard is recomputed; a corrupt
or mismatched shard fails closed and is not silently replaced.

## Diagnostic calculations

Independent-role transfer compares gain-role and rank-role `C`, `P`, `C/P`,
rank improvement at the gain-role optimum, rank improvement at the rank-role
optimum, and their unbounded transfer-efficiency ratio.

Each 7-by-8 `C` map reports pooled alignment, phase and midpoint marginals,
positive-cell count, the q1 sentinel `phase4.midpoint7`, weighted cancellation,
and weighted phase, midpoint, and interaction sums of squares. Within each
quartile and seed, adjacent nonzero 100-update maps are compared by cosine,
cell-sign-flip fraction, pooled-`C` change, and gain change. The three seeds are
also compared at the same update, and each candidate's gain-role map is
compared with its rank-role map. These comparisons never select a checkpoint.

Path stability reports positive pooled-`C` path counts, leave-one-path-out
minima, path standard deviation and standard error, and the analogous
transferred-improvement quantities separately on each role.

## Frozen diagnostic screens and forecasts

A role-level directional screen passes only when pooled `C`, all seven phase
marginals, and all eight midpoint marginals are positive, at least 51 of 56
cells are positive, and the q1 sentinel is positive. Cross-role directional
stability additionally requires this screen on both roles, at least 24 of 32
positive pooled-`C` paths on each role, and every leave-one-path-out pooled `C`
to be positive.

The nonexclusive quartile mechanism flags are:

- `conditional_direction_absent`;
- `direction_present_but_role_unstable`;
- `phase_midpoint_cancellation`;
- `gain_transfer_failure`;
- `optimization_time_rotation`; and
- `strictly_positive_but_too_small`.

Their exact predicates are frozen in the gate module. Rotation requires the
specified low-cosine/high-sign-flip condition in at least two seeds or low
same-update cross-seed cosine. The `strictly_positive_but_too_small` flag is
available only after cross-role stability and every original transferred-risk
point screen pass.

If any required transferred-risk point estimate is nonpositive, the path
forecast is infinite with reason `negative_or_incompatible_point_effect`.
Otherwise the forecast uses the frozen critical value
`7.1588810358178305` and 32 whole-path improvements:

\[
n_{\rm raw}=\left\lceil(7.1588810358178305\,s/\mu)^2\right\rceil,
\]

then rounds upward to a multiple of 32. A quartile has power-only evidence
only when at least two of its three seeds have a cross-role stable candidate
with finite rounded requirement at most 384. This is a planning forecast, not
a confidence interval.

## Gates, decisions, and artifacts

The exact stage flags cover parent identity and immutability, sealed-table
replay, completion of all 960 reductions, direct/algebraic controls, batch and
memory limits, no raw-prediction persistence, deterministic classifications,
and zero new evidence opening.

Closed decisions, in precedence order, are:

1. `quartile_direction_adjudication_parent_provenance_invalid`;
2. `quartile_direction_adjudication_table_replay_invalid`;
3. `quartile_direction_adjudication_decomposition_invalid`;
4. `quartile_direction_adjudication_classification_invalid`;
5. `no_later_quartile_direction_detectable_under_current_class`;
6. `partial_later_quartile_direction_only`;
7. `later_quartile_failure_mechanism_localized`; and
8. `powered_fresh_later_quartile_design_justified`.

Integrity failures return 1, the valid hard stop returns 2, and completed
non-hard-stop diagnostics return 0. Even decision 8 authorizes only drafting
one separately reviewed fresh-learner plan targeted at the diagnosed
mechanism.

The child writes sealed preflight, replay, decomposition, adjudication, and
report artifacts; compact candidate/path/stratum CSVs; the consolidated NPZ;
960 restart shards; parent snapshots; `direction_adjudication_decision.json`;
`REPORT.md`; the workflow gate; and a final registry. Persisted output must be
projected below 512 MiB.

## Production commands

Resolve the parent and freeze deterministic process settings:

```powershell
$parentRun = (
  Resolve-Path `
    "runs/experiment12_d0_jacobi_rb_boundary_tangent_quartile_specialist/20260807-132351_production-exact-quartile-specialist"
).Path

$env:PYTHONHASHSEED = "0"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
```

Create the child and run the preflight:

```powershell
.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication `
  --runs-root `
    runs/experiment12_d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication `
  --run-name production-readonly-q1-q3-direction-adjudication `
  --device cuda `
  --stage preflight `
  --require-gate preflight `
  --parent-quartile-specialist-run-dir $parentRun
if ($LASTEXITCODE -ne 0) { throw "direction-adjudication preflight failed" }

$adjudicationRun = (
  Get-ChildItem `
    "runs/experiment12_d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication" `
    -Directory |
  Where-Object Name -Like "*_production-readonly-q1-q3-direction-adjudication" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName
```

Replay and decompose the sealed evidence:

```powershell
foreach ($stage in @("replay", "decompose")) {
  .\.venv\Scripts\python.exe `
    -m mnist.diag_d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication `
    --device cuda `
    --stage $stage `
    --resume-run-dir $adjudicationRun `
    --require-gate $stage `
    --parent-quartile-specialist-run-dir $parentRun
  if ($LASTEXITCODE -ne 0) { throw "direction-adjudication $stage failed" }
}
```

Commit the closed adjudication. Exit 2 is valid only for the prescribed hard
stop:

```powershell
.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication `
  --device cuda `
  --stage adjudicate `
  --resume-run-dir $adjudicationRun `
  --require-gate adjudicate `
  --parent-quartile-specialist-run-dir $parentRun

$adjudicateExit = $LASTEXITCODE
$decision = Get-Content `
  (Join-Path $adjudicationRun "direction_adjudication_decision.json") `
  -Raw | ConvertFrom-Json
if (
  $adjudicateExit -ne 0 -and
  -not (
    $adjudicateExit -eq 2 -and
    $decision.decision -eq `
      "no_later_quartile_direction_detectable_under_current_class"
  )
) {
  throw "direction adjudication failed with $($decision.decision)"
}
```

Finally write and verify the report:

```powershell
.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication `
  --stage report `
  --resume-run-dir $adjudicationRun `
  --require-gate none `
  --parent-quartile-specialist-run-dir $parentRun
if ($LASTEXITCODE -ne 0) { throw "direction-adjudication report failed" }
```

An interrupted decomposition resumes in the same child directory and reuses
only hash-valid shards. There are deliberately no cache-generation, training,
selection, confirmation, controller, reconstruction, or sampling commands.

## Restricted claim

This workflow can localize failure mechanisms only for the complete frozen
one-image, fixed-grid, four-width-32-expert checkpoint family. It cannot show
that the exact Jacobi/Rao--Blackwell target is invalid, exclude signal in every
architecture, establish multi-image generalization, validate a reverse
controller, establish reconstruction or sampling quality, identify a known
prior, or establish an unsplit-generator or continuum limit.

If the hard stop is reached, the justified claim is:

> Across the complete frozen seed/update grid, no q1-q3 direction satisfying
> the prescribed independent-role and local-geometry conditions is detectable
> under the current later-state-only width-32 expert class.
