# Read-only Jacobi/RB quartile directional representation adjudication

## Experiment purpose

This experiment determines whether one preregistered branch of the frozen
quartile-specialist predictor supplies a stable later-quartile direction that
transfers from historical gain calibration to independent historical rank
evidence. It is an additive, read-only adjudication of an already completed
scientific negative. It is not a new learner and cannot rescue or revise a
parent checkpoint.

The workflow is implemented by
`mnist.diag_d0_jacobi_rb_boundary_tangent_quartile_directional_adjudication`
with the sealed stage order

```text
preflight -> replay -> controls -> fittrace -> nominate -> adjudicate -> report
```

It creates no paths, transitions, targets, optimizer updates, checkpoints,
selection evidence, confirmation evidence, controller trajectories,
reconstructions, or samples.

## Immutable parents

The explicit parents are:

```text
runs/experiment12_d0_jacobi_rb_boundary_tangent_quartile_specialist/
  20260807-132351_production-exact-quartile-specialist

runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_time_local_adjudication/
  20260807-005609_production-v3-time-local-adjudication
```

The specialist parent must remain the terminal scientific negative
`no_training_only_quartile_system` and satisfy these bindings:

| Binding | Required value |
|---|---|
| Artifacts | `4,120` |
| Registry semantic SHA-256 | `e5f6b3ee257b3d4f86ec3ac54f4223540cf76caa24489d22e9c138a54e19c7bb` |
| Registry-file SHA-256 | `e24c7db28081dbceb8f0acf690d779f835379c82a89a2b263976c3e0b631f798` |
| Source fingerprint | `61a8c2fd6a317c05b9eed73e208d40b0cc6c01d6fdd227ae33d71d0be7c6027d` |
| Scientific-config SHA-256 | `05263b7b01c2586e9a771bd71fe28fbb74d8e5d9da873ce4746019c5167c08c1` |

The time-local parent must remain
`exact_rb_high_reverse_time_only_signal` and satisfy these bindings:

| Binding | Required value |
|---|---|
| Artifacts | `29` |
| Registry semantic SHA-256 | `b25256d606f1fea2c9ef78ab5f14a7b8ccd67bc6f5c234bd2ed2a1a0086fd9f5` |
| Registry-file SHA-256 | `15220d3f4ee3e7a4740fd5fae2695e1da1d0b1ea91ee05c70357c3a152569a64` |
| Source fingerprint | `55f259f30ecb1eb47915a44d3ba67a353bac87abd87743f917c55bbcb06a0123` |
| Scientific-config SHA-256 | `faf395317449a842e63de0807d39102f68d7afa49c7700e9cd6c94e0d381b009` |

Preflight verifies all registered payloads, all 492 checkpoint payloads and
their indexed state hashes, every role-cache payload, the transitive parent
chain, and the absence of parent selection or confirmation evidence. Parent
trees are snapshotted before evidence access and checked again before the
terminal registry is sealed.

## Frozen scientific contract

The experiment retains the certified binary64 Jacobi split, grid 28,
`alpha=1`, `K=512`, `tau_eff=5e-5`, the first label-3 image at dataset index
7 with `lambda_mix=0.35`, and the unchanged raw Rao--Blackwell target

\[
\bar Z=y(1-y)\,\partial_y\log k_u(y\mid x).
\]

Predictions remain boundary tangent,
`m(W)=y(1-y)q(W)`, and use only the already permitted later-state inputs.
All 12 trained trajectories and 492 checkpoints are frozen. Evaluation is
deterministic float64, uses model batches of at most 32, and uses no mixed
precision.

For a frozen direction `m`, the workflow accumulates

\[
T=E[\bar Z^2],\qquad C=E[\bar Zm],\qquad P=E[m^2],
\]

so the exact raw-MSE improvement at positive scale `lambda` is

\[
\Delta(\lambda)=2\lambda C-\lambda^2P.
\]

When `C>0` and `P>0`, the positive-ray optimum is
`lambda_plus=C/P` with ceiling `D_plus=C^2/P`. When `C<=0`, both are fixed to
zero. No epsilon denominator, gain clipping, target clipping, quotient target,
residual target, empirical floor, or projection is allowed.

## Exact representation decomposition

Only these three preregistered components are evaluated:

```text
full
local_affine
spatial_cnn
```

The evaluator preserves the model-dtype addition used by the frozen forward
pass. It computes `q_full64` from the model-dtype sum, converts the local
branch separately, and defines the diagnostic spatial branch by exact
subtraction:

```text
q_full64          = float64(q_local_model_dtype + q_spatial_model_dtype)
q_local64         = float64(q_local_model_dtype)
q_spatial64_exact = q_full64 - q_local64
```

Thus `m_full = m_local + m_spatial` exactly up to the frozen `5e-15`
recomposition tolerance, without changing the original full prediction. A
separate direct spatial conversion must remain inside the IEEE forward
rounding bound. The moment identities

```text
C_full = C_local + C_spatial
P_full = P_local + P_spatial + 2*E[m_local*m_spatial]
```

and direct versus reconstructed risk improvements are mandatory
implementation controls.

## Evidence-role firewall

Only historical roles that the specialist parent already opened may be read:

| Role | Permitted use | Forbidden use |
|---|---|---|
| `physical_fit` | In-sample trajectory, cosine, and branch-rotation diagnostics | Nomination, inference, or fitting |
| `gain_calibration` | Compute directional moments, nominate one checkpoint per stream, and freeze `lambda_gain` | Final adjudication |
| `training_rank` | Independent read-only adjudication after the nomination seal exists | Renomination or fitting |

The parent selection and confirmation roles remain unopened. The child cannot
derive or allocate their path IDs. Every output is marked historical design
evidence and nonauthorizing.

## Stage contract

1. `preflight` verifies both immutable parents, freezes candidate/component
   order, role order, 72 max-T family names, bootstrap indices, source closure,
   and the 8-GiB resource projection. It reads inputs only.
2. `replay` reconstructs the sealed specialist gain/rank summaries and their
   480-candidate ordering without loading raw role labels.
3. `controls` runs exact-zero, positive/nonpositive direction,
   energy-dominated, path-instability, cancellation, branch-recomposition,
   rotation, and malformed-moment fixtures without physical labels.
4. `fittrace` opens only `physical_fit` and evaluates all frozen nonzero
   checkpoints for nonauthorizing trajectory and cross-seed rotation
   diagnostics.
5. `nominate` opens only `gain_calibration`, evaluates 40 nonzero checkpoints
   for every quartile, seed, and component, and seals 36 streams. A stream
   without `C_gain>0` remains an explicit no-nominee stream.
6. `adjudicate` verifies the nomination seal before opening `training_rank`,
   then evaluates the sealed nominees under the frozen direction/effect
   family and q0 positive control.
7. `report` applies the closed decision hierarchy, rechecks parent
   immutability, and writes the terminal registry.

An interrupted stage resumes only with the same run directory, source,
parents, plans, and seals. Hash-valid candidate shards are skipped; a corrupt
or incompatible shard fails closed.

## Nomination and inference

For each `(quartile, seed, component)`, the gain role nominates the checkpoint
with largest `D_plus`, breaking an exact tie toward the earlier update. This
creates 36 fixed streams. Rank cannot open until
`direction_nomination_seal.json` verifies.

Each stream contributes two pooled rank statistics to one joint family:

```text
direction: C_rank
effect:    2*lambda_gain*C_rank - lambda_gain^2*P_rank
```

The family has 72 canonical names ordered by quartile, component, seed, then
statistic. It uses 50,000 deterministic whole-path bootstrap replicates,
seed `261352`, one-sided confidence `0.995`, studentization, `higher` quantile
interpolation, and no standard-error floor. A zero-SE statistic is handled
analytically.

A seed-level direction requires positive gain/rank alignment, a positive
simultaneous pooled lower bound, all seven positive phase marginals, all eight
positive midpoint marginals, at least `51/56` positive fine cells, and for q1
a positive `phase4.midpoint7` sentinel. The transferred effect applies the
same local geometry to `Delta_rank(lambda_gain)`. A component is stable only
when at least two of three seed streams pass the complete rule.

The workflow also reports path-sign stability, gain/rank transfer,
phase/midpoint cancellation, branch cross-moments, adjacent-checkpoint and
cross-seed cosines, sign reversals, and advisory path-count forecasts. None of
these diagnostics may nominate another checkpoint or change the family.

## Resource limits

The prelabel forward-only pilot must project:

| Limit | Frozen value |
|---|---:|
| Peak GPU memory | `<=0.80` |
| GPU wall time | `<=48 h` |
| New persisted evidence | `<=1 GiB` |
| Prediction batch size | `32` |
| Mixed precision | `0` |

There is no reduced-candidate, reduced-stratum, lower-bootstrap, CPU-fallback,
or partial-role production mode.

## Closed decisions and authority

Integrity and implementation failures precede all scientific outcomes:

```text
quartile_directional_parent_provenance_invalid
quartile_directional_scientific_contract_invalid
quartile_directional_resource_plan_invalid
quartile_directional_historical_replay_invalid
quartile_directional_prelabel_controls_failed
quartile_directional_fittrace_invalid
quartile_directional_nomination_invalid
quartile_directional_rank_adjudication_invalid
quartile_directional_q0_positive_control_failed
```

Valid scientific outcomes are:

```text
unique_representation_hypothesis_identified
same_class_effect_detected_but_non_authorizing_stop
representation_cancellation_nonidentifying_stop
positive_direction_effect_unresolved_stop
later_quartile_direction_unstable_across_roles_stop
no_later_quartile_signal_detectable_under_permitted_class_stop
```

`unique_representation_hypothesis_identified` is deliberately strict. Exactly
one of `local_affine` or `spatial_cnn` must have stable direction and effect in
all of q1--q3; the competitor must have no stable effect there; full must fail
in at least one later quartile; exact branch algebra must attribute every such
failure to the competing branch; and q0 full must pass the same direction and
effect machinery.

Even that result authorizes only drafting a separate fresh-role,
branch-restricted learner plan with its own pre-cache feasibility gate. All
other valid outcomes are principled stops. No result from this workflow
authorizes new training, confirmation, controller execution, reconstruction,
or sampling.

## Production commands

Resolve the immutable parents and freeze deterministic process settings:

```powershell
$specialistRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_quartile_specialist/20260807-132351_production-exact-quartile-specialist").Path
$timeLocalRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_time_local_adjudication/20260807-005609_production-v3-time-local-adjudication").Path

$env:PYTHONHASHSEED = "0"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
```

Create the child and run preflight:

```powershell
.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_boundary_tangent_quartile_directional_adjudication `
  --runs-root runs/experiment12_d0_jacobi_rb_boundary_tangent_quartile_directional_adjudication `
  --run-name production-read-only-quartile-directional-adjudication `
  --device cuda `
  --stage preflight `
  --require-gate preflight `
  --parent-quartile-specialist-run-dir $specialistRun `
  --parent-time-local-run-dir $timeLocalRun

if ($LASTEXITCODE -ne 0) {
  throw "quartile directional-adjudication preflight did not pass"
}

$auditRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_boundary_tangent_quartile_directional_adjudication" -Directory |
  Where-Object Name -Like "*_production-read-only-quartile-directional-adjudication" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName
```

Run each evidence gate separately:

```powershell
foreach ($stage in @("replay", "controls", "fittrace", "nominate", "adjudicate")) {
  .\.venv\Scripts\python.exe `
    -m mnist.diag_d0_jacobi_rb_boundary_tangent_quartile_directional_adjudication `
    --device cuda `
    --stage $stage `
    --resume-run-dir $auditRun `
    --require-gate $stage `
    --parent-quartile-specialist-run-dir $specialistRun `
    --parent-time-local-run-dir $timeLocalRun

  if ($LASTEXITCODE -ne 0) {
    throw "quartile directional-adjudication $stage did not pass"
  }
}
```

Finalize the valid scientific decision:

```powershell
.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_boundary_tangent_quartile_directional_adjudication `
  --stage report `
  --resume-run-dir $auditRun `
  --require-gate none `
  --parent-quartile-specialist-run-dir $specialistRun `
  --parent-time-local-run-dir $timeLocalRun

if ($LASTEXITCODE -ne 0) {
  throw "quartile directional-adjudication report was invalid"
}

$decision = Get-Content (Join-Path $auditRun "quartile_directional_adjudication_decision.json") -Raw | ConvertFrom-Json
$decision.decision
```

Scientific stop outcomes are valid reports and return zero. Provenance,
resource, control, role-order, or inferential failures return nonzero.

## Validation

Static and focused validation:

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  mnist\d0_jacobi_rb_quartile_directional_adjudication.py `
  mnist\d0_jacobi_rb_quartile_directional_adjudication_inference.py `
  mnist\d0_jacobi_rb_quartile_directional_adjudication_provenance.py `
  mnist\d0_jacobi_rb_quartile_directional_adjudication_gate.py `
  mnist\diag_d0_jacobi_rb_boundary_tangent_quartile_directional_adjudication.py

.\.venv\Scripts\python.exe -m pytest -q `
  tests\test_d0_jacobi_rb_quartile_directional_adjudication.py `
  tests\test_d0_jacobi_rb_quartile_directional_adjudication_inference.py `
  tests\test_d0_jacobi_rb_quartile_directional_adjudication_provenance.py `
  tests\test_d0_jacobi_rb_quartile_directional_adjudication_gate.py `
  tests\test_d0_jacobi_rb_quartile_directional_adjudication_cli.py
```

Then run the adjacent quartile tests, complete suite, and `git diff --check`.
Required fixtures cover exact quadratic and branch algebra, deterministic
nomination, role-order firewalls, 50,000-replicate max-T replay, every
mechanism and terminal decision, q0 invalidation, resume, and parent mutation.

## Restricted claim

A passing branch-identification result may establish only that one frozen
`local_affine` or `spatial_cnn` direction transfers across historical
gain/rank roles and seeds under the exact preregistered rules, thereby
justifying preparation of a separate fresh-role learner plan. A stop may
establish only that no stable later-quartile direction or no single mechanism
is identifiable under this one-image, fixed-grid, later-state-only, width-32
local-affine-plus-CNN class.

Neither outcome proves that the true conditional score is absent or fully
recovered, excludes another architecture, establishes multi-image
generalization, validates a reverse controller, or says anything about
reconstruction or sampling quality.
