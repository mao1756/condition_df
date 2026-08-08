# Exact Jacobi/RB quartile-specialist learnability gate

## Purpose and restricted scope

This additive workflow tests whether four independently trained, time-local
boundary-tangent experts can predict the unchanged exact Jacobi
Rao--Blackwell label across all four forward-time quartiles. It follows the
sealed v3 adjudication result: the earlier shared learner resolved `q0`, had a
positive but underpowered `q1` direction, and spent too much prediction energy
in `q2-q3`.

The workflow is implemented by
`mnist.diag_d0_jacobi_rb_boundary_tangent_quartile_specialist` and has the
restartable stages

```text
preflight -> cache -> controls -> train -> calibrate -> select -> confirm -> report
```

It does not change the physical transition, target, state space, or loss. It
does not execute a reverse controller, construct a reverse trajectory,
reconstruct an image, or sample. Even a successful result authorizes only
planning a separately reviewed reverse-controller control workflow.

## Immutable parent evidence

The authoritative parent is

```text
runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_time_local_adjudication/
  20260807-005609_production-v3-time-local-adjudication
```

It must end `exact_rb_high_reverse_time_only_signal` and verify with:

```text
artifact registry records                 29
artifact registry semantic SHA-256        b25256d606f1fea2c9ef78ab5f14a7b8ccd67bc6f5c234bd2ed2a1a0086fd9f5
source fingerprint                        55f259f30ecb1eb47915a44d3ba67a353bac87abd87743f917c55bbcb06a0123
scientific configuration SHA-256          faf395317449a842e63de0807d39102f68d7afa49c7700e9cd6c94e0d381b009
```

Preflight also verifies its three transitive parents:

- memory-safe v3 selection
  `20260806-181326_production-zero-baseline-v3-memory-safe`;
- physical coarse witness
  `20260730-135059_production-exact-k512-physical-coarse-signal-jsonfix`;
- noisy-Bayes power calibration
  `20260730-012459_production-noisy-jacobi-bayes-power`.

The historical validation table, q0 nominees, resolution ladder, and advisory
scalar optima are design evidence only. They cannot supply a production gain,
checkpoint, threshold, control variate, selection result, or confirmation
result. All parent directories remain immutable.

## Frozen scientific contract

The physical contract remains:

- grid 28, alpha 1, `K=512`, and `tau_eff=5e-5`;
- the first label-3 MNIST image (dataset index 7), mixed with uniform mass at
  `lambda_mix=0.35`;
- the certified binary64 Jacobi split and exact raw Rao--Blackwell target
  `Z_bar = y(1-y) d_y log k_u(y|x)`;
- boundary-tangent coordinates `m(W)=y(1-y)q(W)`;
- width 32, Adam at `1e-3`, zero weight decay, batch 32, unit gradient clip,
  deterministic execution, and no mixed precision;
- 4,000 updates, with update zero and every 100 updates checkpointed; and
- plain unweighted MSE against the raw target within each quartile, normalized
  only by one positive training-only quartile RMS.

The workflow forbids quotient targets, target clipping, loss weights, floors,
limiters, projections, reverse residuals, classifiers, Euler/Gaussian
proxies, historical calibration, and post-hoc path extension.

## Four-expert model

`QuartileSpecialistBoundaryTangentPredictor` owns four independent
`ZeroBaselineBoundaryTangentPredictor` experts. It accepts only `ModelInputs`
and reconstructs the forward quartile from public `fractional_coordinate`
using `reverse_time` and `phase`. Each row is sent to exactly one expert, and
the original row order is restored. The model cannot read an outer step, path
ID, earlier state, target, certificate, or random variable.

The `q0` and `q1` gains are exactly one. The `q2` and `q3` gains are frozen
binary64 scalars loaded from the sealed selected system. Missing, nonfinite,
nonpositive, or unsealed gains are rejected. Mobility-zero rows remain exact
zero.

## Fresh paths and seeds

The role plan is fixed before CUDA work:

| Role | IDs | Paths | Evidence rule |
|---|---|---:|---|
| seam | `0xF3000-0xF3007` | 8 | preflight diagnostic only |
| physical fit | `0xF4000-0xF403F` | 64 | labels open only in `train` |
| gain calibration | `0xF4100-0xF411F` | 32 | labels open first in `calibrate` |
| training rank | `0xF4200-0xF421F` | 32 | labels open only after gain sealing |
| fresh selection | `0xF5000-0xF517F` | 384 | streamed after system sealing |
| confirmation | `0xF7000-0xF717F` | 384 | untouched, opened once after selection |

No cohort mixes roles. Exact generation uses cohorts of at most ten paths;
each 384-path audit uses 38 cohorts of ten and one cohort of four.

The workflow seed is `261331`. Physical seeds are `261332-261343`, three per
quartile in quartile-major order. Selection uses seed `261350` and namespace
`0x51545331`; confirmation uses `261351` and `0x51544331`. Synthetic controls
use `261352-261355`, and the null root is `261356`.

The physical grid contains 480 nonzero candidates plus 12 update-zero
controls, ordered by quartile, seed, and update. No alternate width,
architecture, optimizer, gain family, or repair branch is searched.

## Cache and prelabel controls

Only fit, gain, and rank evidence is persisted as raw cache evidence. Their
expected row counts are 114,688, 57,344, and 57,344. Selection and confirmation
are streamed and retain only whole-path reductions. The cache requires every
transition to be certified, mass error at most `2e-12`, no forbidden events,
throughput at least 1,300 transitions/s, peak CUDA allocation at most 80%,
persisted artifacts at most 3 GiB, and projected exact capture at most 160
hours.

The controls gate runs before any physical label opens:

1. certified CPU/CUDA seam on the eight fresh seam paths;
2. exact-zero outputs from all 12 update-zero physical checkpoints;
3. mixed-batch quartile dispatch and one-expert-only instrumentation;
4. four independent analytic synthetic teachers, each with relative held-out
   MSE at most `0.01` and every held-out path beating zero;
5. exact gain algebra, including `m_raw=2*Z_bar -> lambda=0.5`;
6. one exact-null step per quartile with zero loss and unchanged parameters;
7. the later-state-only input firewall; and
8. batch-32 and 80%-memory enforcement.

Every control record explicitly states `physical_labels_opened=0`.

## Training-only calibration and ranking

Each quartile has its own RMS scale computed only from the 64 fit paths. Three
independent fresh seeds train in each quartile. Batches are sampled uniformly
from that quartile's fit rows; phases, midpoints, paths, edges, and target
amplitudes are not balanced or reweighted.

After all nonzero checkpoints exist, only the 32 gain paths open. For every
`q2` and `q3` checkpoint,

```text
C = mean(Z_bar * m_raw)
P = mean(m_raw^2)
lambda = C / P
```

is reduced in canonical binary64 order. There is no clipping. A gain is
eligible only when `C`, `P`, and `lambda` are finite, `C>0`, `P>0`, and
`0<lambda<1`. All gain records are committed and sealed before rank labels
may open.

The disjoint 32-path rank role then evaluates each checkpoint using its
already-fixed gain. A candidate must have positive pooled improvement,
positive improvements in all seven phase marginals and all eight midpoint
marginals, and at least 51 of 56 positive phase-by-midpoint cells. The q1
sentinel `phase4.midpoint7` must also be positive. Each quartile independently
selects the largest pooled improvement, with ties broken by earlier update
and lower seed. This separable rule searches no Cartesian product.

If any quartile has no candidate, the valid negative
`no_training_only_quartile_system` closes the run without opening selection.
Otherwise checkpoint hashes, scales, gains, role records, and rank-table hash
are frozen in one immutable four-expert system seal.

## Fresh selection and confirmation

Each audit reduces exactly six whole-path contrasts:

```text
specialist_vs_zero.q0.pooled
specialist_vs_zero.q1.pooled
specialist_vs_zero.q2.pooled
specialist_vs_zero.q3.pooled
shrunken_vs_raw.q2.pooled
shrunken_vs_raw.q3.pooled
```

For the last two, raw is the same selected checkpoint at gain one, not a
second candidate. Each audit also retains the final system's `[384,4,56]`
local improvements for the fixed directional screen.

Both audits use 50,000 centered whole-path, one-sided, studentized 99.5%
max-T replicates. Preflight creates and seals 50 independent `uint16` count
shards of shape `[1000,384]` for each audit. Counts come from stateless Philox,
every row sums to 384, all six components share a resample, standard errors
have no floor, and the critical quantile uses NumPy `method="higher"`.

An audit passes only when all six simultaneous lower bounds are strictly
positive and, in every quartile, all phase and midpoint marginals are
positive and at least 51 of 56 cells are positive. The q1 sentinel must be
positive. These local screens are fixed compatibility checks, not separate
confidence claims.

Selection evaluates only the sealed system. It cannot change a checkpoint,
gain, family, screen, or path count. A failure ends
`no_fresh_quartile_specialist_system` and leaves confirmation unopened.
Confirmation uses the same system and a distinct precommitted Philox plan. Its
IDs are burned when `confirmation_open.json` is committed; an interruption
must resume in the same run and cannot allocate a replacement panel.

## Gates and decisions

The CLI exposes:

```text
--stage {preflight,cache,controls,train,calibrate,select,confirm,report,all}
--require-gate {none,preflight,cache,controls,train,calibrate,select,confirm}
--parent-time-local-run-dir
--parent-memory-v3-run-dir
--parent-coarse-witness-run-dir
--parent-bayes-power-run-dir
--resume-run-dir
--runs-root
--run-name
--device
```

Closed decisions, in precedence order, are:

1. `quartile_specialist_parent_provenance_invalid`;
2. `quartile_specialist_scientific_contract_invalid`;
3. `quartile_specialist_path_or_resource_plan_invalid`;
4. `quartile_specialist_exact_cache_invalid`;
5. `quartile_specialist_prelabel_controls_failed`;
6. `quartile_specialist_physical_training_invalid`;
7. `quartile_specialist_gain_calibration_invalid`;
8. `no_training_only_quartile_system`;
9. `quartile_specialist_selection_inference_invalid`;
10. `no_fresh_quartile_specialist_system`;
11. `quartile_specialist_confirmation_invalid`;
12. `quartile_specialist_time_local_signal_not_confirmed`; and
13. `exact_rb_quartile_specialist_time_local_signal_confirmed`.

Decisions 8, 10, and 12 are valid scientific negatives and return exit code
2. Success and authorized stage advancement return 0. Integrity, contract,
resource, or execution failures return 1. Every failure commits all readable
evidence before returning.

Only the final decision sets
`reverse_controller_control_planning_authorized=1`. Controller execution,
sampling, reconstruction, and confirmation reuse remain zero.

## Production commands

Resolve immutable parents and freeze deterministic process settings:

```powershell
$timeLocalRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_time_local_adjudication/20260807-005609_production-v3-time-local-adjudication").Path
$memoryV3Run = (Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_memory_confirmation/20260806-181326_production-zero-baseline-v3-memory-safe").Path
$coarseWitnessRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_physical_coarse_signal_witness/20260730-135059_production-exact-k512-physical-coarse-signal-jsonfix").Path
$bayesPowerRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_bayes_power_calibration/20260730-012459_production-noisy-jacobi-bayes-power").Path

$env:PYTHONHASHSEED = "0"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
```

Create the run and execute preflight:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_boundary_tangent_quartile_specialist `
  --runs-root runs/experiment12_d0_jacobi_rb_boundary_tangent_quartile_specialist `
  --run-name production-exact-quartile-specialist `
  --device cuda --stage preflight --require-gate preflight `
  --parent-time-local-run-dir $timeLocalRun `
  --parent-memory-v3-run-dir $memoryV3Run `
  --parent-coarse-witness-run-dir $coarseWitnessRun `
  --parent-bayes-power-run-dir $bayesPowerRun
if ($LASTEXITCODE -ne 0) { throw "quartile-specialist preflight did not pass" }

$specialistRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_boundary_tangent_quartile_specialist" -Directory |
  Where-Object Name -Like "*_production-exact-quartile-specialist" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName
```

Run each gate separately and stop at the first nonzero exit:

```powershell
foreach ($stage in @("cache", "controls", "train")) {
  .\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_boundary_tangent_quartile_specialist `
    --device cuda --stage $stage --resume-run-dir $specialistRun --require-gate $stage `
    --parent-time-local-run-dir $timeLocalRun `
    --parent-memory-v3-run-dir $memoryV3Run `
    --parent-coarse-witness-run-dir $coarseWitnessRun `
    --parent-bayes-power-run-dir $bayesPowerRun
  if ($LASTEXITCODE -ne 0) { throw "quartile-specialist $stage did not pass" }
}
```

Calibrate and seal one training-only system:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_boundary_tangent_quartile_specialist `
  --device cuda --stage calibrate --resume-run-dir $specialistRun --require-gate calibrate `
  --parent-time-local-run-dir $timeLocalRun `
  --parent-memory-v3-run-dir $memoryV3Run `
  --parent-coarse-witness-run-dir $coarseWitnessRun `
  --parent-bayes-power-run-dir $bayesPowerRun
```

If the decision is `no_training_only_quartile_system`, stop. Otherwise run
the single-system selection audit:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_boundary_tangent_quartile_specialist `
  --device cuda --stage select --resume-run-dir $specialistRun --require-gate select `
  --parent-time-local-run-dir $timeLocalRun `
  --parent-memory-v3-run-dir $memoryV3Run `
  --parent-coarse-witness-run-dir $coarseWitnessRun `
  --parent-bayes-power-run-dir $bayesPowerRun
```

If selection ends `no_fresh_quartile_specialist_system`, confirmation is
forbidden. Only after a passing selection gate, open the one untouched audit:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_boundary_tangent_quartile_specialist `
  --device cuda --stage confirm --resume-run-dir $specialistRun --require-gate confirm `
  --parent-time-local-run-dir $timeLocalRun `
  --parent-memory-v3-run-dir $memoryV3Run `
  --parent-coarse-witness-run-dir $coarseWitnessRun `
  --parent-bayes-power-run-dir $bayesPowerRun

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_boundary_tangent_quartile_specialist `
  --stage report --resume-run-dir $specialistRun --require-gate none `
  --parent-time-local-run-dir $timeLocalRun `
  --parent-memory-v3-run-dir $memoryV3Run `
  --parent-coarse-witness-run-dir $coarseWitnessRun `
  --parent-bayes-power-run-dir $bayesPowerRun
```

An interrupted stage resumes with the same run directory and immutable
parents. An opened confirmation is never replaced.

## Restricted claim

The successful decision establishes only fresh held-out, one-image,
fixed-grid, exact-Jacobi/RB time-local risk improvement for one sealed
four-expert system. It does not establish a valid reverse controller, complete
reverse dynamics, reconstruction quality, generative sampling quality,
multi-image generalization, a known prior, an unsplit-generator limit, or a
continuum limit.
