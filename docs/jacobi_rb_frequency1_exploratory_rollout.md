# Frequency-one Jacobi/RB exploratory reverse rollout

Date: 2026-08-11  
Primary research mode: **exploratory**  
Nearest milestone: one-image reverse-suffix and full-path reconstruction

## Decision

This experiment asks one objective-bearing question:

> Starting from fresh exact forward states of the frozen one-image target,
> does the fixed frequency-one tangent controller improve a reverse suffix or
> a complete reverse path relative to paired zero control?

The checkpoint is seed `261372`, update `3700`. It was selected post hoc from
validation-inspected historical evidence and therefore cannot support a new
confirmatory claim. The experiment produces raw reverse states and fixed-scale
images even when they are poor.

The project objective remains a DDPM-like MNIST generator based on the
fixed-grid Eulerian approximation. This patch tests one-image composition from
forward-chain anchors; it does not test prior-start sampling or multi-image
generation.

The immutable frequency-one learner ended
`no_frequency1_coordinate_validation_candidate`: none of its 120 candidates
satisfied the prospective 228-component validation family, update zero was
selected, and confirmation was never opened. That exact negative remains
valid. It does not answer whether a fixed historical checkpoint can have
useful integrated dynamics under an exploratory gain. This rollout is the
direct system test required after the preceding proxy-only sequence; executing
it resets the proxy-only counter to zero even when its images are poor.

## Competing hypotheses

- If the oracle fails even on the short suffix, the controller sign, ordering,
  oracle construction, or split composition is not yet interpretable.
- If the oracle passes but every learned gain is dynamically negligible, the
  historical coefficient is under-calibrated for recursive control.
- If learned control is non-negligible but harms the short suffix, its
  direction or local representation is wrong on rollout states.
- If the short suffix improves but the full path does not, accumulated,
  later-time, or on-policy error is the leading explanation.
- If learned and oracle control improve the full path, the strict validation
  proxy was not necessary for this exploratory one-path objective, although
  the effect may still be too small to matter visually.

## Frozen mechanism

For a matched pair with current head fraction `y`, a controller supplies a
tangent coefficient `q`. One control subflow applies

\[
  y' = \operatorname{logistic}(\operatorname{logit}(y)+2q\,\Delta u).
\]

Every phase preserves the existing reference/control/reference order and uses
two controller microsteps (`M=2`). Reference transitions use the existing exact
certified CUDA Jacobi sampler. The learned controller is evaluated at gains
`0.5, 1, 2, 4`; development chooses minimum final raw squared L2 error, with a
tie going to the smaller gain.

The controls are:

- zero tangent score, which must have exactly zero control displacement;
- the immutable learned checkpoint scaled by the selected gain;
- a source-informed target-fraction oracle. For an interior pair it returns
  `(logit(y_star)-logit(y))/(2*delta_u)`, so the frozen logistic subflow reaches
  the fixed mixed-target fraction exactly before the final reference half-step.

The oracle is a controller-interface positive control, not a learned score and
not the true reverse score.

## Evidence roles and paired randomness

| Role | Initial path ID | Use |
|---|---:|---|
| Preflight | `0xFB000` | CUDA seam and timing only |
| Development | `0xFB100` | Choose one gain on the 128-step suffix |
| Evaluation | `0xFB200` | Fresh 128-step and 512-step paired rollout |
| Optional replication | `0xFB300` | Fresh full path only after a positive full evaluation |

Allocation version `frequency1-rollout-fb-v2-after-fa-smoke` explicitly supersedes
the proposed `0xFA000/100/200/300` slots. The immutable failed preflight
`20260812-005426_production-frequency1-exploratory-rollout` consumed `0xFA000`
in its three-lane exact-CUDA smoke before exposing a phase-benchmark adapter
error. It opened no development, evaluation, or replication path. The full
repository scan found the replacement `0xFB000/100/200/300` slots collision-free;
this is a recorded plan revision, not automatic relocation.

The forward root seed is `261401`; the reverse root seed is `261402`.
Within a role and horizon, zero, learned, and oracle trajectories use the same
path ID, stream role, transition IDs, and underlying random bits. Variant names
never enter the exact-reference RNG key. Development selection is committed
before the evaluation path is generated. Replication is generated only after
its predeclared positive-evaluation condition is met.

Historical confirmation data is neither read nor generated.

## Resource stop rule

Preflight measures three complete one-path M=2 tangent phases with the exact
adaptive reference adapter. The slowest complete-repeat rate projects the
frozen `32,313,344`-transition main workflow. This is an execution/resource
gate: projected time must be no more than six GPU hours, and the measured rate
must be at least 1,300 transitions/second. It may legitimately fail if small
392-lane launches are too slow; the workflow does not silently substitute an
eager, approximate, or batched scientific schedule.

Main persisted storage is capped at 2 GiB. Optional positive replication has a
separate two-GPU-hour budget.

Expected main accelerator time is about three hours at the earlier
3,024-transition/s planning rate, with six hours as the automatic stop. The
preflight projection is authorizing because the adaptive one-path launch shape
may be slower than that historical planning rate. Peak device allocation is
expected to remain below 80% and is recorded as a health metric; an allocation
failure is an execution failure, not evidence against learned signal. New
complexity is limited to one reusable rollout core, one additive CLI, focused
tests, and this experiment note. The cost buys the first reverse images and
paired objective metrics; no existing read-only table can answer that dynamic
question.

### Completed production preflight

The corrected production run
`20260812-005942_production-frequency1-exploratory-rollout-fbv2` stopped validly
at this resource gate. The three complete-phase rates were `622.3636`,
`1224.9556`, and `1226.5807` transitions/s. The preregistered slowest-repeat
projection was `51,920.364 s` (`14.422 h`), above the `21,600 s` cap. The best
warmed repeat alone still projects to about `7.32 h`, so the conclusion is not
caused solely by including cold-start time. All other preflight checks passed,
certificate fraction was one, fallbacks and forbidden events were zero, and
simplex/pair-mass errors remained below `2e-12`. No fresh forward or reverse
scientific role was opened. Continue only through a separately reviewed exact
cross-variant fused-scheduling feasibility patch.

## Failure-preserving artifacts

Every valid branch retains raw float64 anchors and trajectories, eight-step
restart shards, fixed-scale PNGs, endpoint and progress metrics, controller
telemetry, transition health diagnostics, exact configuration and checkpoint
bindings, and the command used. Development failure still produces its images,
decision, `REPORT.md`, manifest, and checksums. Evaluation adds short and full
zero/learned/oracle contact sheets; optional replication is recorded only when
its frozen trigger is met. A post-run `HANDOFF.md` must follow `AGENTS.md` and
identify the outcome-to-action branch actually taken.

## Metrics and renderings

All scientific metrics use raw float64 simplex states:

- squared L2 and L1 error to the fixed mixed target;
- total variation distance (`0.5 * L1`);
- centered contrast correlation relative to the uniform state;
- paired absolute and relative improvement over zero.

Mechanism telemetry records score and logistic-shift magnitudes, reference and
control fraction displacement, their ratio, boundary counts, certificates,
fallbacks, forbidden events, mass errors, timing, and restart identity.

Every selected state has two fixed-scale renderings. Raw density uses one scale
derived from the mixed target. Background-demixed images subtract
`lambda_mix/784`, divide by `1-lambda_mix`, and use the original source-image
maximum. Clipping occurs only for display. Per-image autoscaling is forbidden.

## Typed gates

### Execution and integrity

Downstream action: produce or interpret paired trajectories. The gate checks
immutable source/checkpoint bindings, fresh path allocations, exact reverse
order, restart chains, CUDA availability, certification, finite nonnegative
states, simplex/pair conservation, finite controller values, storage, and the
six-hour projection. Failure means the run is invalid or infeasible under the
frozen schedule; it does not mean the learned controller lacks signal.

### Oracle interpretability

Downstream action: open the fresh evaluation path. The development short-path
oracle must have lower final squared L2 error than paired zero. Failure directs
work to controller sign/order, oracle construction, or composition. It does
not authorize retraining or another representation feature.

### Diagnostic thresholds

Learned/oracle endpoint improvement and a control/reference displacement ratio
below `0.05` are descriptive. They are not hypothesis tests, p-values, or
confirmatory gates.

## Outcome-to-action table

| Observation | Interpretation | Next action |
|---|---|---|
| Integrity/resource gate fails | Invalid or infeasible frozen execution | Repair the exact blocker and rerun the same science |
| Development oracle fails | Controller/oracle/composition is not validated | Repair the full-system positive control |
| Oracle passes; learned control is negligible and no short improvement | Dynamic calibration is too small | Run a bounded scale/calibration experiment |
| Oracle passes; non-negligible learned control does not improve short suffix | Direction or local architecture is suspect | Compare a global/multiscale or rollout-trained learner |
| Learned improves short; oracle improves full; learned fails full | Accumulation, late-time, or on-policy failure | Test anchors 255/383 and frozen late-time-off schedules |
| Oracle improves short but fails full | Long-horizon composition or M=2 is suspect | Compare fixed oracle at M=2/M=8 on a bounded horizon |
| Learned and oracle improve full but effect is tiny | Real but inadequate dynamic signal | Pivot to a materially stronger learner using this harness |
| Learned improves full evaluation | Integrated utility despite proxy-gate failure | Run frozen replication, then M=8 verification |
| Evaluation and replication improve | Stronger exploratory direction evidence | Freeze checkpoint/gain and verify at M=8 |

## Commands

Run focused tests first:

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_d0_jacobi_rb_boundary_tangent.py `
  tests/test_d0_jacobi_rb_boundary_tangent_frequency1_coordinate.py `
  tests/test_d0_jacobi_rb_tangent_rollout.py `
  tests/test_diag_d0_jacobi_rb_frequency1_rollout.py
```

Production:

```powershell
$frequency1Run = (Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_frequency1_coordinate_learnability/20260811-010641_production-frequency1-coordinate-v1-one-image").Path
$sourceRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_coarse_residual_learnability/20260731-140333_production-exact-k512-coarse-residual-one-image").Path

.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_frequency1_rollout `
  --frequency1-run-dir $frequency1Run `
  --source-run-dir $sourceRun `
  --runs-root runs/experiment12_d0_jacobi_rb_frequency1_rollout `
  --run-name production-frequency1-exploratory-rollout `
  --device cuda:0 `
  --stage all
```

Resume any incomplete eight-step shard chain:

```powershell
$rolloutRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_frequency1_rollout" -Directory |
  Where-Object Name -Like "*_production-frequency1-exploratory-rollout" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName

.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_frequency1_rollout `
  --resume-run-dir $rolloutRun `
  --device cuda:0 `
  --stage all
```

Inspect `REPORT.md`, `exploratory_decision.json`, the trajectory NPZs, and the
three contact sheets. `SHA256SUMS.txt` and `artifact_manifest.json` audit the
compact final run.

## Claim boundary

A positive full-path result supports only a one-path exploratory statement for
the fixed post-hoc checkpoint, gain, M=2 split, and forward-terminal start. It
does not overturn the parent's validation result, establish generation from a
prior, establish generalization, or confirm the Eulerian limit.
