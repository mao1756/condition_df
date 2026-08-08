# Exact eager-prefix boundary-tangent time-local confirmation v2

## Scope

This workflow integrates the certified eager-prefix CUDA schedule into fresh
one-image boundary-tangent cache generation, optimization, and one sealed
time-local confirmation. It keeps the exact binary64 Jacobi transition and the
unchanged Rao--Blackwell label. It does not execute a learned controller,
construct a complete reverse path, reconstruct an image, or sample.

The workflow is implemented by
`mnist.diag_d0_jacobi_rb_boundary_tangent_eager_confirmation`. Its only
authorizing terminal outcome is
`exact_rb_boundary_tangent_time_local_signal_confirmed`; that outcome permits
planning a separate at-most-eight-phase controller-control experiment.

## Immutable parents

The run binds three immutable parents:

- eager complete-pipeline scheduler:
  `20260803-034008_production-eager-prefix-complete-pipeline`;
- resource-failed v1 tangent preflight:
  `20260802-140158_production-boundary-tangent-rb-controller`;
- successful coarse-residual learner:
  `20260731-140333_production-exact-k512-coarse-residual-one-image`.

The v1 result is re-adjudicated as
`legacy_schedule_resource_projection_superseded`. All of its scientific checks
passed, its production namespaces were never opened, and its only failed check
was the obsolete 32.701-hour adaptive-schedule projection. The eager parent
passed its complete-pipeline gate at a conservative projected 25.985 hours.

## Frozen evidence contract

- Grid 28, alpha 1, K=512, `tau_eff=5e-5`, first label-3 image, and
  `lambda_mix=0.35`.
- Root seed `261311`; model seeds `261312,261313,261314`; bootstrap seed
  `261315`; synthetic/null seeds `261317,261318`.
- Train/validation/confirmation path counts are 64/32/64, using disjoint
  namespaces `0xEC100`, `0xEC200`, and `0xED000`.
- Train and validation execute as `[10x9,6]`; confirmation executes separately
  as `[10x6,4]`. The mixed P10 cohort is split by immutable role before every
  artifact commit.
- Both canonical phase transitions and all eight midpoint branches use the
  frozen `eager_prefix_128_tpb128` authorizer.
- Permitted inputs are stored as float32. Raw exact Rao--Blackwell labels are
  stored separately as float64. Confirmation labels are streamed and never
  persisted.
- Exact row counts are 114,688 train, 57,344 validation, and 114,688
  confirmation. Exact transition counts are 134,873,088, 67,436,544, and
  134,873,088.
- Exact evidence generation, including actual cache and confirmation
  durations, must remain within 108,000 seconds. Optimization and bootstrap
  time are outside this resource total.

The predictor is

```text
m_theta(W) = y(1-y) [q_B(C(W)) + q_residual_theta(W)].
```

The training-only baseline is fitted independently in every `4x7x8x392` cell
as `sum(mu*Zbar)/sum(mu^2)`. Training minimizes only direct, unweighted MSE
against the unchanged raw label. No quotient target, target clipping, loss
weighting, floor, limiter, projection, or stored residualized target is
allowed.

Before physical labels open, the synthetic tangent teacher must attain
relative validation MSE at most 0.01 and beat zero on every validation path;
the exact-baseline null must select update zero. All three physical seeds run
to completion. A nonzero checkpoint must beat the frozen baseline overall and
in the high-reverse-time quartile.

The sealed confirmation uses 50,000 whole-path bootstrap replicates and a
one-sided 99.5% studentized max-T family. All 224 combined-vs-zero cell bounds
and all four combined-vs-baseline quartile bounds must be strictly positive.

## Production sequence

Run preflight first:

```powershell
$eagerRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_eager_pipeline_confirmation/20260803-034008_production-eager-prefix-complete-pipeline").Path
$failedTangentRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_controller_confirmation/20260802-140158_production-boundary-tangent-rb-controller").Path
$coarseRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_coarse_residual_learnability/20260731-140333_production-exact-k512-coarse-residual-one-image").Path

.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_boundary_tangent_eager_confirmation `
  --runs-root runs/experiment12_d0_jacobi_rb_boundary_tangent_eager_confirmation `
  --run-name production-eager-boundary-tangent-time-local `
  --device cuda `
  --stage preflight `
  --parent-eager-pipeline-run-dir $eagerRun `
  --failed-boundary-tangent-run-dir $failedTangentRun `
  --parent-coarse-residual-run-dir $coarseRun `
  --require-gate preflight
```

Resolve the new directory only after preflight returns zero:

```powershell
$tangentV2Run = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_boundary_tangent_eager_confirmation" -Directory |
  Where-Object Name -Like "*_production-eager-boundary-tangent-time-local" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName
```

Run `cache`, inspect `cache_gate.json`, then run `train`, inspect
`train_gate.json`, and only then run `confirm`:

```powershell
foreach ($stage in @("cache", "train", "confirm")) {
  .\.venv\Scripts\python.exe `
    -m mnist.diag_d0_jacobi_rb_boundary_tangent_eager_confirmation `
    --device cuda `
    --stage $stage `
    --resume-run-dir $tangentV2Run `
    --parent-eager-pipeline-run-dir $eagerRun `
    --failed-boundary-tangent-run-dir $failedTangentRun `
    --parent-coarse-residual-run-dir $coarseRun `
    --require-gate $stage
  if ($LASTEXITCODE -ne 0) { break }
}
```

Required-gate failures are committed before the command exits nonzero. A
completed scientific gate is immutable; only incomplete or corrupt shard
tails may be resumed.
