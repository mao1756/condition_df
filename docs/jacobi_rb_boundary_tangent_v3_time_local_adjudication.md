# Immutable v3 time-local signal adjudication

## Purpose and scope

This additive workflow explains the scientifically valid
`no_validation_candidate` result from the memory-safe zero-baseline v3
experiment. It reads already sealed train/validation evidence, replays the
original search-aware decision, and decomposes the learned prediction into
target alignment and prediction energy. It does not change or rescue the
historical decision.

The workflow is implemented by
`mnist.diag_d0_jacobi_rb_boundary_tangent_v3_time_local_adjudication` and has
the restartable stages

```text
preflight -> replay -> decompose -> report
```

No stage creates a Jacobi transition, optimizer update, checkpoint, physical
path, confirmation label, controller trajectory, reconstruction, or sample.
The reserved confirmation namespace `0xF2000-0xF203F` remains unopened.

## Immutable evidence

The primary parent is

```text
runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_memory_confirmation/
  20260806-181326_production-zero-baseline-v3-memory-safe
```

It is bound by:

```text
terminal decision                         no_validation_candidate
artifact registry records                 620
artifact registry file SHA-256            a3a697a7a92f2ad6f0cc666d95ebb92b11dfd8c6837005d457356bd326b79076
artifact registry semantic SHA-256        a69ca33be9c3281eb54e9285f3d292d9e8c9cd0775781ed652af0a1adda85626
source fingerprint                        5cde21b7ed36a806f2e872cf8fb3f7ac859d9317ab87aad1e728b95d544a2cee
scientific configuration SHA-256          bbb3b79cc7afc6311b6e6413cde0e7f93c074f8fe885bd4960140bc39e44f61e
```

Its immutable cache and memory contracts passed, all three physical training
tasks completed, and validation selection was performed. No nonzero candidate
passed the original 27,360-member search-aware family, so logical update zero
was selected and confirmation was forbidden.

Two independent controls are also bound:

- The physical coarse witness
  `20260730-135059_production-exact-k512-physical-coarse-signal-jsonfix`
  ended `exact_physical_coarse_signal_detected`. Its 2,616-record registry has
  semantic SHA-256
  `ca405ea7c00d6efed470c0651b8ad28d31f797cf142a1bef5f75d464ee1c3ab3`
  and file SHA-256
  `866984822ef14dbb13f0644a0f23697f6fb42ecec40b07341249d74043319747`.
- The noisy-Bayes calibration
  `20260730-012459_production-noisy-jacobi-bayes-power` ended
  `noisy_bayes_detection_pipeline_calibrated`. Its 74-record registry has
  semantic SHA-256
  `01b5d772299611e9e17b886658b7eba80a7ab50805241e94d2e9a8ba36562e79`
  and file SHA-256
  `4caa9597f1ce7e6e6180ea11bffe55138f10582791b60e5d529e38d9e3b13bec`.

Preflight verifies every registered artifact, transitive parent, cache binding,
selection seal, checkpoint hash, path role, and source/configuration
commitment. The parent directories are never written. Confirmation artifacts
or labels are rejected even if they later appear.

## Sealed selection replay

The authoritative path table is
`validation_candidate_path_tables.npz`. Its required shape is
`[32 paths, 120 candidates, 228 components]`. The 228 components retain their
original ordering:

- 224 `model_vs_zero` contrasts over four forward-time quartiles, seven
  phases, and eight midpoint fractions;
- four quartile-pooled `model_vs_zero` contrasts.

Replay must reproduce the original whole-path max-T family exactly:

```text
candidate count                           120
component count                           228
search-family size                        27,360
bootstrap replicates                      50,000
critical value                            7.1588810358178305
eligible candidates                       0
logical update zero selected              1
```

The partial-discovery census is diagnostic but exactly reproducible:

- 28 positive simultaneous candidate/component bounds across 24 checkpoints;
- discovered component indices `{6,7,15,224}`, all in forward quartile `q0`;
- no positive adjusted bound in `q1`, `q2`, or `q3`;
- no candidate with all 228 point estimates positive; and
- `q1.phase4.midpoint7` nonpositive for every candidate.

The three frozen `q0` nominees are selected only for decomposition by the
largest original adjusted lower bound of component 224, independently within
each model seed:

| Seed | Update | Pooled-q0 point estimate | Adjusted lower bound | Positive q0 fine cells |
|---:|---:|---:|---:|---:|
| 261312 | 900 | 0.0007141943 | 0.0003539920 | 55/56 |
| 261313 | 1600 | 0.0013397223 | 0.0006636222 | 55/56 |
| 261314 | 3900 | 0.0006320900 | 0.0002391796 | 54/56 |

This nomination is historical diagnosis, not a replacement selection rule.
It cannot open confirmation or alter the `no_validation_candidate` result.

## Descriptive resolution ladder

The replay preregisters the following summaries before loading any
checkpoint:

```text
quartile x phase x midpoint
quartile x phase
quartile x midpoint
quartile pooled
phase pooled
midpoint pooled
overall
```

Canonical path weights and component identities are preserved when pooling.
Only the original 228-component simultaneous bounds retain their historical
gate meaning. Additional intervals are marked `posthoc_non_authorizing=1`;
they cannot qualify a candidate, revise a threshold, or rescue the failed
selection.

The coarse witness is replayed independently. Its overall conditional-mean
energy estimate remains positive, and its descriptive quartile estimates are

```text
q0  0.0016264
q1  0.0006213
q2  0.0002102
q3  0.0001358
```

These values describe decreasing signal strength at the witness resolution.
They are neither targets nor estimates fitted to the v3 validation paths.

## Quadratic-risk decomposition

Only the three frozen nominees are loaded. Immutable train and validation
inputs are evaluated in canonical CUDA batches of at most 32. For prediction
`m_hat` and the unchanged exact Rao--Blackwell label `Z_bar`, the workflow
computes, by path and by every resolution-ladder stratum,

\[
I = E[\bar Z^2-(\bar Z-\hat m)^2]
  = 2E[\bar Z\hat m]-E[\hat m^2].
\]

The stored components are:

```text
C = E[Z_bar * m_hat]       target/prediction alignment
P = E[m_hat^2]             prediction energy
I_direct                   direct risk improvement
I_reconstructed = 2*C-P   reconstructed improvement
```

The direct and reconstructed improvements must agree within `5e-15`. All
values must be finite, every model call must contain at most 32 rows, and peak
CUDA allocation must not exceed 80% of device memory. Train and validation
statistics remain separate and their gap is reported.

When `C>0` and `P>0`, the workflow also reports

\[
\lambda^*=C/P,
\qquad
I_{\rm directional}=C^2/P.
\]

These are advisory diagnostics of direction versus amplitude. The scalar is
never applied to predictions or checkpoints and cannot participate in
selection, confirmation, or a future training target.

Each quartile is classified from the median of the three nominees:

- `directional_alignment_missing` if its cross term is nonpositive;
- `prediction_energy_dominates` if alignment is positive but `2*C-P<=0`;
- `positive_but_underpowered` if its point improvement is positive but the
  original adjusted bound is nonpositive; or
- `resolved` only when its original adjusted bound is positive.

A conservative path-count forecast uses the observed whole-path variance and
the frozen critical value. A nonpositive point effect receives an infinite
requirement; path multiplication is not presented as a repair for a wrong or
absent direction.

## Gates, artifacts, and decisions

The CLI exposes:

```text
--stage {preflight,replay,decompose,report,all}
--require-gate {none,preflight,replay,decompose}
--parent-memory-v3-run-dir
--parent-coarse-witness-run-dir
--parent-bayes-power-run-dir
--resume-run-dir
--runs-root
--run-name
--device
```

Closed terminal decisions are:

- `control_provenance_invalid`;
- `sealed_selection_replay_invalid`;
- `coarse_witness_replay_invalid`;
- `quadratic_risk_decomposition_invalid`;
- `no_learned_time_local_signal`;
- `multiplicity_only_underpowered`;
- `mixed_time_local_signal_inconclusive`; and
- `exact_rb_high_reverse_time_only_signal`.

The last decision requires all three frozen nominees to have positive original
adjusted pooled-`q0` bounds, at least 90% positive q0 fine-cell point
estimates, no positive adjusted component in `q1-q3`, no all-positive
228-component candidate, and a finite positive independent coarse-witness
replay.

The workflow writes atomic provenance and immutability reports,
`adjudication_plan.json`, exact replay and partial-discovery records,
resolution-ladder tables, witness energy/signal-capture tables, per-path and
stratified quadratic-risk NPZ/CSV records, train/validation-gap diagnostics,
advisory scalar-calibration tables, a path-count forecast, named replay and
decomposition gates, the terminal decision, status, and artifact registry.

Every artifact records zero new transitions, optimization, confirmation,
controller trajectories, reconstruction, and sampling. Required-gate failures
return nonzero only after the readable evidence is committed.

## Production commands

Resolve the three immutable parents and run preflight:

```powershell
$memoryRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_memory_confirmation/20260806-181326_production-zero-baseline-v3-memory-safe").Path
$witnessRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_physical_coarse_signal_witness/20260730-135059_production-exact-k512-physical-coarse-signal-jsonfix").Path
$bayesRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_bayes_power_calibration/20260730-012459_production-noisy-jacobi-bayes-power").Path

$env:PYTHONHASHSEED = "0"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_boundary_tangent_v3_time_local_adjudication `
  --runs-root runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_time_local_adjudication `
  --run-name production-v3-time-local-adjudication `
  --device cuda `
  --stage preflight `
  --parent-memory-v3-run-dir $memoryRun `
  --parent-coarse-witness-run-dir $witnessRun `
  --parent-bayes-power-run-dir $bayesRun `
  --require-gate preflight
```

Resolve the fresh child after preflight passes:

```powershell
$adjudicationRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_time_local_adjudication" -Directory |
  Where-Object Name -Like "*_production-v3-time-local-adjudication" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName
```

Run replay and decomposition separately, stopping at the first nonzero exit:

```powershell
foreach ($stage in @("replay", "decompose")) {
  .\.venv\Scripts\python.exe `
    -m mnist.diag_d0_jacobi_rb_boundary_tangent_v3_time_local_adjudication `
    --device cuda `
    --stage $stage `
    --resume-run-dir $adjudicationRun `
    --parent-memory-v3-run-dir $memoryRun `
    --parent-coarse-witness-run-dir $witnessRun `
    --parent-bayes-power-run-dir $bayesRun `
    --require-gate $stage

  if ($LASTEXITCODE -ne 0) { break }
}
```

An interrupted stage resumes with the same child directory. A failed gate is
not rerun with changed inputs, thresholds, or parents.

## Restricted claim and next action

This adjudication can establish only that the current width-32 learner found
a search-adjusted signal in forward quartile `q0` (the high-reverse-time
region) while the existing validation evidence did not resolve a learned
signal in later forward quartiles. It cannot establish that another model is
learnable, that the physical score is globally recovered, or that a reverse
controller is executable.

Only `exact_rb_high_reverse_time_only_signal` authorizes planning a fresh
quartile-specialized exact-RB learner with new selection evidence. The next
design follows the decomposition: independent quartile experts for missing
alignment, training-only time-local shrinkage when prediction energy dominates,
or a powered fresh panel only for positive-but-underpowered effects. The raw
Rao--Blackwell label and plain unweighted MSE within each quartile remain
mandatory. No outcome here authorizes confirmation, controller execution,
reconstruction, or sampling.
