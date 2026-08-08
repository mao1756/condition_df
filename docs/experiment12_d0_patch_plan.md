# Experiment 12 D0 patch plan

The active milestone is the completed immutable v3 time-local adjudication
recorded below and in
[`jacobi_rb_boundary_tangent_v3_time_local_adjudication.md`](jacobi_rb_boundary_tangent_v3_time_local_adjudication.md).
The global zero-baseline v3 selection gate remains a valid
`no_validation_candidate` result, but its sealed evidence establishes a
reproducible exact-Rao--Blackwell signal only in forward quartile `q0` (high
reverse time). The next authorized work is planning a fresh
quartile-specialized learner. Confirmation, controller execution,
reconstruction, and sampling remain unauthorized.

## Historical D0-v0 one-image baseline

The original D0-v0 route was a single density-ratio experiment whose purpose
was to reach an honest first generated image. It is retained here as
historical context, not as the current command sequence.

## Workflow

1. Build 64 forward-reference paths from the first label-3 MNIST image.
2. Split whole paths once into 48 training and 16 validation paths.
3. Train the boundary-smooth scalar density-ratio model for 4,000 optimizer
   steps with matched time and label batches.
4. Select the earliest EMA checkpoint with the lowest finite validation BCE.
5. Run paired strength-0 and strength-1 reverse sampling on eight held-out
   terminal states with shared noise.
6. Publish the samples, contact sheet, metrics, and a pass/stop summary.

Run the complete workflow with:

```powershell
python -m mnist.diag_d0_v0_one_image --stage all --device cuda
```

Use `--smoke --device cpu` for the small synthetic integration check. The
scientific settings are frozen in `D0V0Config`; the CLI intentionally exposes
no tuning surface.

## Stop rule

Sampling runs even when validation BCE misses its gate. A failed production
summary ends this experiment after the first run. Inspect the artifacts, but
do not add seeds, thresholds, estimators, or certification layers to rescue
the result.

The detailed pre-v0 plan is archived at
`docs/archive/experiment12_d0_patch_plan_pre_v0_20260727.md`.

<!-- BEGIN EXACT NOISY-JACOBI BAYES-POWER CALIBRATION -->

## Exact noisy-Jacobi Bayes-power calibration

The exact-\(K=512\) one-image run
`20260729-015817_production-exact-k512-rb-one-image-learnability` is immutable
and remains a sealed `no_detectable_one_image_conditional_signal` result. Its
only failed confirmation check was that the learned model did not beat the
analytic zero predictor in aggregate.

The next additive workflow is a controls-only power calibration. It trains the
unchanged width-32 model on exact noisy Rao--Blackwell labels from:

- a bounded teacher
  \(q_0(x)=x+\tfrac12\), with oracle
  \(m(y,u)=y(1-y)e^{-2u}/[1+\tfrac12e^{-2u}(2y-1)]\);
- a stationary null \(q_0=1\), with oracle \(m=0\).

Parent input caches supply pair-mass/time templates only. Parent physical
`*_labels_audit.npz` artifacts are forbidden. The analytic oracle is stored
separately and is audit-only; it is never a model input or training target.
Train, validation, and confirmation each use eight disjoint paths per law,
with three model seeds and a sealed one-time confirmation.

Passing requires a powered oracle panel, teacher recovery of at least half the
oracle gain, aggregate improvement over zero, improvement over metadata on all
eight teacher paths, and no corresponding false-discovery conjunction under
the null. A pass authorizes only planning a fresh physical-signal witness.

The complete rationale, gates, and exact staged commands are in
[`jacobi_rb_bayes_power_calibration.md`](jacobi_rb_bayes_power_calibration.md).

<!-- END EXACT NOISY-JACOBI BAYES-POWER CALIBRATION -->

<!-- BEGIN EXACT JACOBI/RB COARSE-RESIDUAL LEARNABILITY -->

## Exact Jacobi/RB coarse-residual learnability

The terminal coarse-signal witness
`20260730-135059_production-exact-k512-physical-coarse-signal-jsonfix`
detected a nonzero physical coarse conditional mean. Both preregistered
one-sided 99% lower bounds were positive, with point estimate
`0.0006484248701021389`. This does not overturn the immutable earlier
one-image result: the width-32 learner still failed to beat analytic zero.

The completed workflow
`20260731-140333_production-exact-k512-coarse-residual-one-image` froze the
witness-derived shrinkage baseline and trained only an additive neural
residual while retaining plain unweighted MSE against the exact
Rao--Blackwell label. Witness paths were baseline-training evidence only.
Fresh train/validation/confirmation roles contained `64/32/64` paths, and
confirmation was generated only after validation sealed a nonzero checkpoint
that beat the baseline overall and at high reverse time.

The confirmation gate passed its simultaneous tests that the frozen baseline
beats zero and that the combined baseline-plus-residual predictor beats the
frozen baseline. This result is the successful learning parent for the active
boundary-tangent gate. No coarse-residual stage imported or invoked a reverse
sampler.

The derivation, claim boundary, and staged production commands are recorded
in
[`jacobi_rb_coarse_residual_learnability.md`](jacobi_rb_coarse_residual_learnability.md).

<!-- END EXACT JACOBI/RB COARSE-RESIDUAL LEARNABILITY -->

<!-- BEGIN EXACT JACOBI/RB BOUNDARY-TANGENT CONTROLLER CONFIRMATION -->

## 2026-08-02: boundary-tangent reverse-controller confirmation

The immutable affine-controller run
`20260802-040147_production-exact-rb-reverse-controller-control` failed in
preflight before any oracle or physical reverse-law panel opened. Its exact
Jacobi reference transition was certified and healthy, but the learned affine
subflow produced `y=1.0018200816811438` at the first recorded failing cell.
The defect also reproduced for `M=4`, production `M=8`, and an advisory
`M=16` replay. The run's 26-record registry has semantic SHA-256
`2b1c7dc65fa715a6996571bf2cadf8cdbada15eedfa5ef1b8ffa5b5d9c18be8b`.
It is preserved unchanged and re-adjudicated as
`frozen_affine_conormal_flow_boundary_invalid`, not a kernel or target
failure.

The additive repair represents the same conditional mean as

\[
  m=\mu q,\qquad \mu=y(1-y),
\]

trains direct unweighted MSE against the unchanged exact Rao--Blackwell label,
and integrates the frozen finite coefficient through

\[
  \operatorname{logit}(y^+)=\operatorname{logit}(y)+2q\,\delta u.
\]

This flow fixes both facets and has first-order increment `2m*du`, so it
changes finite-step coordinates rather than the learned reverse generator.
Fresh `64/32/64` train/validation/confirmation paths are disjoint from all
inspected evidence. Every path executes the full exact `K=512` chain; rows are
recorded at steps `15,31,...,511`, all seven phase occurrences, and the eight
exact midpoint fractions. The unchanged width-32 residual CNN receives only
later state, reverse time, phase, color, duration, and label. A 228-member
confirmation family and a separate 784-member reverse-law family must both
pass without clipping, floors, limiters, projection, quotient targets, or
other target transformation.

Training uses Adam at `1e-3`, batch 32, 4,000 updates, validation every 100
updates, zero weight decay, unit gradient clipping, deterministic execution,
and no mixed precision. The training-only `4 x 7 x 8 x 392` tangent baseline
is fit by direct least squares. An analytic tangent teacher and an
exact-baseline null must pass before physical labels open; update zero must
remain eligible, and a nonzero checkpoint must beat the baseline overall and
at high reverse time before confirmation is created.

The exact cache/resource contract requires certification fraction one, mass
error at most `2e-12`, at least 1,300 transitions/s, fallback fraction/time at
most `1e-4/0.10`, at most 80% device memory, at most 30 projected hours, and
at most 1.25 GiB persisted evidence. Confirmation uses a one-sided 99.5%
whole-path max-T family; the controller control uses a two-sided 99.5% family
at anchors `127,255,383,511` and microstep counts `2,4,8`. Both use 50,000
deterministic whole-path bootstrap replicates.

The full derivation, training recipe, claim boundary, and staged commands are
in
[`jacobi_rb_boundary_tangent_controller_confirmation.md`](jacobi_rb_boundary_tangent_controller_confirmation.md).

<!-- END EXACT JACOBI/RB BOUNDARY-TANGENT CONTROLLER CONFIRMATION -->

<!-- BEGIN FUSED-LANE BOUNDARY-TANGENT SCHEDULING FEASIBILITY -->

## 2026-08-02: fused-lane boundary-tangent scheduling feasibility

The immutable boundary-tangent preflight
`20260802-140158_production-boundary-tangent-rb-controller` was scientifically
and numerically valid but missed its frozen resource gate.  Its exact
`337,182,720`-transition workload projected to `32.701` hours at approximately
`2,864.17` transitions/s, above the preregistered 30-hour limit.  It produced
no cache, physical training, confirmation, controller trajectory,
reconstruction, or sampling evidence.  The result is therefore re-adjudicated
as `eight_path_cache_schedule_resource_infeasible`, not as a Jacobi law,
Rao--Blackwell target, representation, or certification failure.

The additive scheduling gate keeps every scientific input fixed and tests only
execution packing.  Canonical phases use at most ten independent paths per
CUDA call.  The eight midpoint branches are flattened in exact
`(midpoint,path,edge)` order and split into contiguous launches of at most
4,096 lanes.  Train and validation are projected with the frozen
`[10x9,6]` schedule, while confirmation remains separate with `[10x6,4]`.
Canonical path IDs determine randomness, so packing cannot change any path or
expose validation evidence to training code.

The fixed pilot measures four 16-step windows starting at outer steps
`0,128,256,384`, three deterministic repeats, and all four cache/streaming
profiles.  Timing includes exact base and midpoint transitions, certification,
input/label conversion, atomic cache commits, and a real unchanged width-32
predictor forward/risk commit for streaming profiles.  The slowest repeat is
authorizing.  Passing requires at most 108,000 projected seconds, an effective
rate of at least `3,122.0622/s`, certification fraction one, exact numerical
health, the unchanged memory/storage limits, and no approximate mechanism.

Run the fresh preflight and then the pilot:

```powershell
$failedTangentRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_controller_confirmation/20260802-140158_production-boundary-tangent-rb-controller").Path
$coarseRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_coarse_residual_learnability/20260731-140333_production-exact-k512-coarse-residual-one-image").Path

.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_boundary_tangent_schedule_feasibility `
  --runs-root runs/experiment12_d0_jacobi_rb_boundary_tangent_schedule_feasibility `
  --run-name production-fused-boundary-tangent-schedule `
  --device cuda `
  --stage preflight `
  --failed-boundary-tangent-run-dir $failedTangentRun `
  --parent-coarse-residual-run-dir $coarseRun `
  --require-gate preflight

$scheduleRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_boundary_tangent_schedule_feasibility" -Directory |
  Where-Object Name -Like "*_production-fused-boundary-tangent-schedule" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName

.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_boundary_tangent_schedule_feasibility `
  --device cuda `
  --stage pilot `
  --resume-run-dir $scheduleRun `
  --failed-boundary-tangent-run-dir $failedTangentRun `
  --parent-coarse-residual-run-dir $coarseRun `
  --require-gate pilot
```

Only `exact_boundary_tangent_schedule_feasible` authorizes integrating the
fused scheduler into a fresh v2 cache/training/confirmation workflow.  This
gate itself performs no scientific cache generation, training, controller
trajectory, reconstruction, or sampling.

<!-- END FUSED-LANE BOUNDARY-TANGENT SCHEDULING FEASIBILITY -->

<!-- BEGIN EAGER-PREFIX BOUNDARY-TANGENT SCHEDULING CONFIRMATION -->

## 2026-08-02: exact eager-prefix scheduling confirmation

The fused-lane pilot
`20260802-174811_production-fused-boundary-tangent-schedule` was exact,
certified, deterministic, and numerically clean, but remained resource-only
infeasible: `35.6773` projected hours and `2,625.25/s` versus the frozen
30-hour and `3,122.0622/s` gates. Its slowest P10 base work spent 97.46% of
its time in the certificate authorizer, while only 0.1882% of base lanes
escalated from mode/prefix `128/64` to `4112/128`.

The next additive gate reveals the same transition-local second Philox word
at the initial `m=128` proof bucket. Candidate modes remain 128, bisection
steps remain 56, and the exact Jacobi law, exposures, IDs, rounded later
states, and raw Rao--Blackwell targets are unchanged. Preflight requires
bit-identical scientific outputs; a sealed three-repeat profile stage must
conservatively forecast a passing resource result; only then may the unchanged
complete four-profile pilot run.

Commands, hashes, artifacts, and the restricted controls-only claim are in
[`jacobi_rb_boundary_tangent_prefix_schedule_confirmation.md`](jacobi_rb_boundary_tangent_prefix_schedule_confirmation.md).
Only `exact_boundary_tangent_eager_prefix_schedule_feasible` authorizes
integration into a fresh v2 workflow. It does not authorize cache generation,
training, controller trajectories, reconstruction, or sampling.

### 2026-08-03 eager-prefix preflight repair

The immutable `20260803-005757` preflight failed after its large
adaptive/eager equivalence panel had passed.  A valid active edge in the
zero-mass fixture required a quantile 1,156 ULPs from the nonauthorizing CUDA
proposal, exhausting the legacy +/-256-ULP candidate-local Arb lattice.  The
failure was a certificate-escalation execution defect, not an eager-prefix
equivalence or Jacobi-law failure.

An additive versioned worker now retains the local search and escalates its
exhaustion to full certified Arb inversion on the identical Philox stream.
The direct control is active-mask aware: active rows require the full density
certificate, while structural zero-mass/zero-time rows retain exact code-0
no-op semantics.  Fresh preflight `20260803-021405...arbfix-v2` passed and is
`ready_for_profile`; no cache, training, controller trajectory, or sampling
ran.

<!-- END EAGER-PREFIX BOUNDARY-TANGENT SCHEDULING CONFIRMATION -->

<!-- BEGIN EAGER-PREFIX COMPLETE-PIPELINE CONFIRMATION -->

## 2026-08-03: eager-prefix complete-pipeline continuation

The immutable eager-prefix profile run
`20260803-021405_production-eager-prefix-boundary-tangent-schedule-arbfix-v2`
completed normally and was scientifically and numerically valid. Its three
repeats produced identical exact outputs, certificate fraction one, and zero
fallback or forbidden events. The conservative base-only forecast nevertheless
missed the frozen resource gate by `1060.8496` seconds: `30.2947` projected
hours and `3091.6935/s`, versus 30 hours and `3122.0622/s`.

That forecast credited eager-prefix acceleration only to the separately
measured P10 base-authorizer component. It did not measure eager-prefix effects
on midpoint branches or P6/P4 cohorts. The parent decision remains
`eager_prefix_profile_computationally_infeasible`; the additive continuation
records the narrower derived adjudication `base_only_projection_inconclusive`
and directly times the complete frozen pipeline.

The new preflight proves that the inherited `0xEE000`/`0xEE010` cache and
`0xEE100`/`0xEE110` streaming namespaces were never opened by the parent
profile. The pilot then runs three cyclic repeats over the same four 16-step
windows, using the eager schedule for both base and midpoint authorizers. The
slowest repeat for each profile enters the exact `[10x9,6]` plus `[10x6,4]`
projection. The 30-hour, `3122.0622/s`, certification, conservation, fallback,
memory, storage, launch-size, and no-approximation gates remain unchanged.

Only `exact_boundary_tangent_eager_pipeline_feasible` authorizes integrating
the scheduler into a fresh v2 boundary-tangent workflow. This confirmation
does not generate a production cache, train a model, run a controller
trajectory, reconstruct an image, or sample. Commands, immutable hashes,
restart semantics, and complete acceptance criteria are in
[`jacobi_rb_boundary_tangent_eager_pipeline_confirmation.md`](jacobi_rb_boundary_tangent_eager_pipeline_confirmation.md).

### Production outcome

Run `20260803-034008_production-eager-prefix-complete-pipeline` passed with
decision `exact_boundary_tangent_eager_pipeline_feasible`. The conservative
slowest-repeat projection was `25.984910233561983` hours at
`3,604.471434567184` transitions/s, compared with the unchanged 30-hour and
`3,122.0622222222223/s` gates. All pilot transitions were certified; fallback
and forbidden-event counts were zero, repeat hashes agreed, maximum mass error
was `4.440892098500626e-16`, and the 615-artifact terminal registry verified
without a hash mismatch (semantic SHA-256
`b85907645f1b11be581f1247268729478fb7b4ff49444181663ac90467792eb7`).

The next authorized patch is therefore fresh v2 integration of the eager-prefix
scheduler into boundary-tangent cache generation, training, and sealed
confirmation. Production cache generation, training, controller trajectories,
reconstruction, and sampling remain unauthorized until that separate workflow
passes its own gates.

<!-- END EAGER-PREFIX COMPLETE-PIPELINE CONFIRMATION -->

<!-- BEGIN EAGER BOUNDARY-TANGENT TIME-LOCAL CONFIRMATION V2 -->

## 2026-08-03: exact eager-prefix boundary-tangent time-local confirmation v2

The passing complete-pipeline scheduler now has a separate fresh integration
workflow. `mnist.diag_d0_jacobi_rb_boundary_tangent_eager_confirmation`
generates new 64/32 train/validation paths with the exact eager authorizer,
runs the unchanged synthetic and null controls, trains all three physical
seeds on the raw Rao--Blackwell target, and conditionally opens 64 sealed
confirmation paths.

The v1 boundary-tangent preflight is preserved and re-adjudicated as
`legacy_schedule_resource_projection_superseded`: every scientific check
passed, its obsolete adaptive schedule projected 32.701 hours, and it opened
none of the production namespaces. The v2 execution freezes `[10x9,6]`
train/validation cohorts and separately streamed `[10x6,4]` confirmation
cohorts. The eager 128-prefix policy is used for both the seven canonical
phases and every midpoint branch. Train/validation evidence is split by role
before persistence even where a P10 execution cohort crosses the role
boundary.

The scientific target and optimization contract are unchanged. The model
returns `y(1-y)(q_B+q_residual)` and minimizes plain unweighted MSE against the
exact binary64 Rao--Blackwell label. The cellwise baseline is derived only
from training labels; synthetic and exact-baseline-null controls must pass
before physical labels open. A nonzero checkpoint must beat the baseline on
validation overall and at high reverse time before confirmation can be
created.

The sealed 228-component, one-sided 99.5% whole-path max-T family requires
strictly positive lower bounds for every time-local combined-vs-zero contrast
and every quartile combined-vs-baseline contrast. Only
`exact_rb_boundary_tangent_time_local_signal_confirmed` authorizes planning a
separate at-most-eight-phase controller-control patch. It does not authorize a
controller trajectory, reverse path, reconstruction, or sampling.

Exact provenance, thresholds, artifacts, restart rules, and production
commands are in
[`jacobi_rb_boundary_tangent_eager_confirmation.md`](jacobi_rb_boundary_tangent_eager_confirmation.md).

<!-- END EAGER BOUNDARY-TANGENT TIME-LOCAL CONFIRMATION V2 -->

<!-- BEGIN SEALED BOUNDARY-TANGENT FALSE-DISCOVERY ADJUDICATION -->

## 2026-08-05: sealed boundary-tangent false-discovery adjudication

The completed eager v2 run
`20260803-113404_production-eager-boundary-tangent-time-local` remains
terminally `selection_false_discovery`. Its exact cache and confirmation were
certified, numerically healthy, and resource-valid, but the validation rule
searched 120 nonzero checkpoints while requiring only pointwise improvement
over the fitted baseline. The selected seed `261314`, update `800` improved
over that baseline by just `1.5723577897475138e-6` overall and already lost to
zero. On the sealed 64-path confirmation, every quartile
combined-versus-baseline point estimate and every one of the 224
combined-versus-zero point estimates was negative.

The next additive child workflow is report-only. It binds the immutable
3,457-artifact parent registry (semantic SHA-256
`36bf43c0a108549954617a78625d4fd65820141c950ba84330133de1f8648580`),
reconstructs the omitted `baseline_vs_zero` contrast from the sealed risk
shards, and re-evaluates all existing checkpoints on the sealed validation
cache. It performs no transition, training, new path generation, controller
trajectory, reconstruction, or sampling.

Stages are `preflight -> adjudicate -> decision`. Preflight fails closed on
any registry, checkpoint, cache, seal, shard, identity, role, or hash defect.
Adjudication uses a two-sided 229-member whole-path max-|T| family for the
post-hoc baseline contrast and a one-sided, search-aware 480-member family for
the 120 candidates by four residual quartiles. The child decisions are
`forensic_evidence_invalid`, `implementation_or_replay_defect`,
`retained_baseline_v3_selection_design_ready`,
`baseline_only_requires_fresh_confirmation_design`,
`zero_baseline_v3_learnability_ready`,
`baseline_and_residual_unresolved`, or
`selection_resolution_failure_confirmed`.

The old confirmation paths `0xED000-0xED03F` are permanently burned: their
observed outcomes may inform this preregistered redesign audit, but they can
never again select or confirm a model. No child outcome authorizes a
controller. At most, the child may authorize planning a fresh v3 learner with
a search-aware validation family and a new, unopened confirmation namespace.

```powershell
$parent = (Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_eager_confirmation/20260803-113404_production-eager-boundary-tangent-time-local").Path

.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_boundary_tangent_false_discovery_adjudication `
  --runs-root runs/experiment12_d0_jacobi_rb_boundary_tangent_false_discovery_adjudication `
  --run-name production-sealed-false-discovery-adjudication `
  --parent-run-dir $parent `
  --stage all `
  --bootstrap-replicates 50000 `
  --require-gate decision
```

Full evidence contracts, baseline/residual classifications, and the
prospective v3 selection rule are documented in
[`jacobi_rb_boundary_tangent_false_discovery_adjudication.md`](jacobi_rb_boundary_tangent_false_discovery_adjudication.md).

<!-- END SEALED BOUNDARY-TANGENT FALSE-DISCOVERY ADJUDICATION -->

<!-- BEGIN ZERO-BASELINE BOUNDARY-TANGENT V3 LEARNABILITY -->

## 2026-08-05: exact zero-baseline boundary-tangent v3 workflow

The additive v3 workflow is implemented as
`mnist.diag_d0_jacobi_rb_boundary_tangent_v3_learnability`, with the sealed
stage order `preflight -> cache -> train -> select -> confirm -> report`.
Its only representation change is `q_B := 0`; the predictor remains
`m_theta(W)=y(1-y)q_theta(W)` and is trained by plain, unweighted MSE against
the unchanged exact binary64 Jacobi/Rao--Blackwell label. No fitted baseline
array, buffer, checkpoint key, or baseline-derived target is permitted.

Physical training produces the fixed 3-by-41 checkpoint grid without opening
physical validation labels. The separate selection stage evaluates all 120
nonzero checkpoints using one prospective `120 x 228 = 27,360` whole-path
max-T family. Update zero remains the logical null. A nonzero nominee is
sealed only when all 228 simultaneous lower bounds are strictly positive;
otherwise confirmation paths remain unopened. A single fresh confirmation,
if authorized, uses the same 228-component family.

Fresh roles are frozen at `0xF0000-0xF0007` (preflight),
`0xF1000-0xF103F` (train), `0xF1100-0xF111F` (validation), and
`0xF2000-0xF203F` (reserved confirmation). The implementation verifies every
bound parent artifact row, the source image and mixed target, a transitive
41-file source closure, exact restart tails, candidate path alignment, and
the 30-hour evidence-generation cap. Historical scheduler/cache APIs and
serialized records remain unchanged; exact reviewed successor mappings keep
their immutable parent runs reportable.

Implementation validation passed the 74-test focused plan suite, the
340-test boundary-tangent regression family, and the complete 1,886-test
repository suite. No production v3 path, physical
training task, confirmation, controller trajectory, reconstruction, or
sampling was run during implementation. Exact evidence contracts and staged
commands are in
[`jacobi_rb_boundary_tangent_zero_baseline_v3_learnability.md`](jacobi_rb_boundary_tangent_zero_baseline_v3_learnability.md).

Only
`exact_rb_zero_baseline_boundary_tangent_time_local_signal_confirmed`
authorizes planning a separate controls-only, at-most-eight-phase controller
study. It does not authorize controller execution or sampling.

<!-- END ZERO-BASELINE BOUNDARY-TANGENT V3 LEARNABILITY -->

<!-- BEGIN V3 CERTIFICATE-SEMANTICS COMPARATOR REPAIR -->

## 2026-08-05: v3 certificate-semantics comparator repair

The first production v3 preflight,
`20260805-170727_production-zero-baseline-v3-learnability`, is immutable. Its
19-record registry has file SHA-256
`94fcd4443fe2e45ff148fab9954b372bd3d3d7cf5c5efddb6dce132c8018692d`
and semantic SHA-256
`ed60b3d4130883b39e940ccb3d78f8110ceec8c979e111c21d8b43dfa21ccd3b`.
The historical terminal decision remains `exact_cache_invalid`, but the
verified result is re-adjudicated as
`certificate_semantics_comparator_invalid`: base states and targets, all
midpoint states, targets, and certificate semantics agreed; every transition
was certified; mass and resource checks passed; and no production cache,
training, selection, or confirmation evidence was opened.

The defect was that the seam compared `batch_certificate_sha256`, which also
contains nonauthorizing proof-effort metadata such as mode and prefix counts.
Adaptive and eager-prefix execution may use different proof effort while
certifying the same binary64 state and Rao--Blackwell target. The repaired
semantic comparator gates state, target, and certificate meaning; proof-effort
metadata equality is retained as an advisory diagnostic only. A real semantic
mismatch still fails `exact_cache_invalid`. Because the source fingerprint
changed, the failed directory must not be resumed.

Implementation validation used the real eight-path CUDA seam in
`test-output/v3-certificate-semantics-real-cuda/20260805-213345_codex-validation-zero-baseline-v3-certificate-semantics-fix`.
It ended `ready_for_cache`: scientific payload and normalized authorization
both matched exactly, while proof metadata differed advisory-only as expected.
Certification fraction was `1`, maximum mass error was
`4.44089209850063e-16`, forbidden-event count was `0`, and eager throughput was
approximately `4649.41` transitions/s. This validation directory did not open
production cache, training, selection, or confirmation evidence.

Resolve the immutable parents, then run a fresh preflight:

```powershell
$v2Run = (Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_eager_confirmation/20260803-113404_production-eager-boundary-tangent-time-local").Path
$failedV3Run = (Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_learnability/20260805-170727_production-zero-baseline-v3-learnability").Path
$eagerRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_eager_pipeline_confirmation/20260803-034008_production-eager-prefix-complete-pipeline").Path
$coarseRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_coarse_residual_learnability/20260731-140333_production-exact-k512-coarse-residual-one-image").Path
$adjudicationRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_false_discovery_adjudication/20260805-125856_production-sealed-false-discovery-adjudication").Path

$env:PYTHONHASHSEED = "0"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_boundary_tangent_v3_learnability `
  --runs-root runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_learnability `
  --run-name production-zero-baseline-v3-certificate-semantics-fix `
  --device cuda --stage preflight `
  --failed-v3-preflight-run-dir $failedV3Run `
  --parent-v2-run-dir $v2Run --adjudication-run-dir $adjudicationRun `
  --parent-eager-pipeline-run-dir $eagerRun --parent-coarse-residual-run-dir $coarseRun `
  --require-gate preflight
if ($LASTEXITCODE -ne 0) { throw "fresh v3 preflight did not pass" }

$v3Run = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_learnability" -Directory |
  Where-Object Name -Like "*_production-zero-baseline-v3-certificate-semantics-fix" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName
```

Only after that fresh preflight passes, start the cache in the same shell:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_boundary_tangent_v3_learnability `
  --device cuda --stage cache --resume-run-dir $v3Run `
  --failed-v3-preflight-run-dir $failedV3Run `
  --parent-v2-run-dir $v2Run --adjudication-run-dir $adjudicationRun `
  --parent-eager-pipeline-run-dir $eagerRun --parent-coarse-residual-run-dir $coarseRun `
  --require-gate cache
if ($LASTEXITCODE -ne 0) { throw "fresh v3 cache did not pass" }
```

<!-- END V3 CERTIFICATE-SEMANTICS COMPARATOR REPAIR -->

<!-- BEGIN V3 IMMUTABLE-CACHE STREAMING-MEMORY RECOVERY -->

## 2026-08-06: v3 immutable-cache streaming-memory recovery

The production certificate-semantics repair run
`20260805-224211_production-zero-baseline-v3-certificate-semantics-fix`
completed a valid 64/32-path train/validation cache, then failed before any
scientific control completed. The first unbatched 114,688-row width-32 model
call required a 10.71875-GiB activation on a 7.96-GiB GPU. The failure is
therefore re-adjudicated as `prelabel_control_memory_schedule_invalid`, not as
evidence that a control or the physical learner failed.

The additive recovery binds the parent's 2,085-artifact registry and reuses
its cache read-only. All neural work is streamed from writable host arrays in
fixed batches of 32, with no full-cache CUDA prediction or target tensor and
an 80% peak-memory gate. The exact Jacobi law, raw Rao--Blackwell target,
zero-baseline representation, optimizer, seeds, 120-candidate selection
family, 228 contrasts, and 50,000-replicate max-T gates are unchanged.
Confirmation remains unopened unless a nonzero validation nominee is sealed.

Exact provenance, the OOM calculation, stage commands, restart contract, and
restricted claim are recorded in
[`jacobi_rb_boundary_tangent_v3_memory_confirmation.md`](jacobi_rb_boundary_tangent_v3_memory_confirmation.md).
Only a final
`exact_rb_zero_baseline_boundary_tangent_time_local_signal_confirmed` result
authorizes planning an at-most-eight-phase controller-control patch; this
memory repair itself authorizes no controller trajectory or sampling.

<!-- END V3 IMMUTABLE-CACHE STREAMING-MEMORY RECOVERY -->

<!-- BEGIN V3 TIME-LOCAL SIGNAL ADJUDICATION -->

## 2026-08-06: immutable v3 time-local signal adjudication

The memory-safe v3 run
`20260806-181326_production-zero-baseline-v3-memory-safe` completed all three
physical training tasks and the prospective 120-by-228 validation family, but
ended `no_validation_candidate`. This is a valid negative result for the
original all-cell gate: update zero was selected, confirmation remained
forbidden, and `0xF2000-0xF203F` was never opened.

The sealed table nevertheless contains a reproducible time-local pattern. All
three seeds have positive search-adjusted pooled signal in forward quartile
`q0`, while no adjusted component is positive in `q1-q3`; no candidate has all
228 point estimates positive. The additive read-only workflow
`mnist.diag_d0_jacobi_rb_boundary_tangent_v3_time_local_adjudication` replays
that decision and decomposes improvement as
`2 E[Z_bar*m_hat] - E[m_hat^2]`. It creates no paths, labels, updates,
confirmation evidence, controller trajectories, or samples.

The resolution ladder and scalar calibration are explicitly post-hoc and
nonauthorizing. Only an `exact_rb_high_reverse_time_only_signal` decision may
authorize planning a fresh quartile-specialized exact-RB learner; it cannot
rescue the v3 candidate or authorize confirmation or sampling. Immutable
bindings, estimands, artifacts, decisions, and exact production commands are
documented in
[`jacobi_rb_boundary_tangent_v3_time_local_adjudication.md`](jacobi_rb_boundary_tangent_v3_time_local_adjudication.md).

<!-- END V3 TIME-LOCAL SIGNAL ADJUDICATION -->

<!-- BEGIN V3 TIME-LOCAL ADJUDICATION PRODUCTION OUTCOME -->

## 2026-08-07: v3 time-local adjudication production outcome

Official run
`20260807-005609_production-v3-time-local-adjudication` completed at stage
`decompose` with decision `exact_rb_high_reverse_time_only_signal`. All 29
registered child artifacts verified under semantic SHA-256
`b25256d606f1fea2c9ef78ab5f14a7b8ccd67bc6f5c234bd2ed2a1a0086fd9f5`.
The preflight, sealed replay, quadratic decomposition, batch-32 memory, and
confirmation-firewall gates all passed. The workflow generated no paths or
transitions, performed no optimizer update, did not access confirmation
evidence, and did not execute a controller or sampler.

The historical `32 x 120 x 228` validation table replayed exactly with
critical value `7.1588810358178305`. There were no eligible all-cell
candidates. The 28 positive adjusted candidate-components belonged to 24
checkpoints and only four component IDs `{6,7,15,224}`, all in `q0`. No
component in `q1-q3` had a positive original adjusted lower bound, and no
candidate had positive point estimates in all 228 components.

All three frozen per-seed `q0` nominees were independently positive under the
original adjustment:

- seed `261312`, update `900`: point `0.0007141943`, lower bound
  `0.0003539920`, and `55/56` positive fine cells;
- seed `261313`, update `1600`: point `0.0013397223`, lower bound
  `0.0006636222`, and `55/56` positive fine cells; and
- seed `261314`, update `3900`: point `0.0006320900`, lower bound
  `0.0002391796`, and `54/56` positive fine cells.

The quadratic decomposition classifies `q0` as `resolved`, `q1` as
`positive_but_underpowered`, and `q2-q3` as
`prediction_energy_dominates`. Median net improvements were approximately
`+7.142e-4`, `+7.517e-5`, `-2.870e-5`, and `-6.594e-5`. The independent
coarse-witness energy also decreases across quartiles:
`[0.0016264, 0.0006213, 0.0002102, 0.0001358]`. More paths alone are therefore
not a repair for `q2-q3`; any amplitude shrinkage must be learned from fresh
training paths and frozen before selection.

This outcome authorizes planning only. The next design must retain the exact
raw Jacobi/Rao--Blackwell label, boundary-tangent form, width 32 per quartile,
and plain unweighted MSE within each quartile. Historical validation remains
diagnostic and cannot select or confirm the future learner. Fresh disjoint
training, prospective multiplicity-aware selection, and untouched
confirmation evidence are mandatory. A planning handoff is stored at
`handoff/jacobi_rb_v3_time_local_next_decision_20260807.zip`.

<!-- END V3 TIME-LOCAL ADJUDICATION PRODUCTION OUTCOME -->

<!-- BEGIN EXACT JACOBI RB QUARTILE SPECIALIST -->

## 2026-08-07: exact Jacobi/RB quartile-specialist learnability gate

The official immutable time-local adjudication
`20260807-005609_production-v3-time-local-adjudication` ended
`exact_rb_high_reverse_time_only_signal`. It resolved `q0`, found a positive
but underpowered `q1` direction, and found positive alignment but excessive
prediction energy in `q2-q3`. This authorizes a fresh learnability experiment,
not confirmation of the historical learner and not controller execution.

The additive workflow
`mnist.diag_d0_jacobi_rb_boundary_tangent_quartile_specialist` trains four
independent width-32 zero-baseline boundary-tangent experts on the unchanged
raw Rao--Blackwell target with plain unweighted MSE within each quartile.
Training-only q2/q3 scalar gains are frozen before disjoint checkpoint ranking.
One sealed four-expert system then faces a fresh 384-path selection audit and,
only if that passes, one untouched 384-path confirmation audit. Both use the
same preregistered six-component 99.5% whole-path max-T family and fixed local
compatibility screens.

Fit, gain, rank, selection, and confirmation roles have disjoint paths and
label-opening seals. Historical validation values may justify the fixed path
budget but cannot fit a gain, select an expert, or contribute to either audit.
The workflow performs no controller trajectory, reverse path, reconstruction,
or sampling. Only
`exact_rb_quartile_specialist_time_local_signal_confirmed` authorizes planning
a separate reverse-controller control milestone. The full contract, commands,
and restricted claim are documented in
[`jacobi_rb_boundary_tangent_quartile_specialist.md`](jacobi_rb_boundary_tangent_quartile_specialist.md).

<!-- END EXACT JACOBI RB QUARTILE SPECIALIST -->

## Historical multiscale reproducibility commands

These four commands are retained only so the completed multiscale evidence
can be reproduced with the current parser. They are historical and do not
replace the active exact Jacobi/Rao--Blackwell boundary-tangent workflow above.

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_multiscale_learnability `
  --runs-root runs/experiment12_d0_multiscale_learnability `
  --run-name production-multiscale-preflight `
  --device cuda `
  --stage cache-preflight `
  --zero-residual-run-dir runs/experiment12_d0_zero_residual/20260715-131054_production-manuscript-budget `
  --parent-one-image-run-dir runs/experiment12_d0_one_image/20260715-183915_production-one-image-direct-doob-preflight `
  --temporal-strides 1,16,64,256,1024 `
  --preflight-paths 4 `
  --dataset-seed 260718 `
  --cache-seed 260721 `
  --split-seed 260722 `
  --teacher-seed 260727 `
  --require-gate cache
```

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_multiscale_learnability `
  --runs-root runs/experiment12_d0_multiscale_learnability `
  --run-name production-multiscale-learnability `
  --device cuda `
  --stage all `
  --zero-residual-run-dir runs/experiment12_d0_zero_residual/20260715-131054_production-manuscript-budget `
  --parent-one-image-run-dir runs/experiment12_d0_one_image/20260715-183915_production-one-image-direct-doob-preflight `
  --temporal-strides 1,16,64,256,1024 `
  --cache-paths 64 `
  --anchors-per-path 32 `
  --anchor-bin-counts 4,4,4,4,16 `
  --train-paths 40 `
  --selection-paths 12 `
  --audit-paths 12 `
  --dataset-seed 260718 `
  --cache-seed 260721 `
  --split-seed 260722 `
  --training-seeds 260723,260724,260725 `
  --bootstrap-seed 260726 `
  --teacher-seed 260727 `
  --bootstrap-reps 10000 `
  --base-channels 32 `
  --batch-size 128 `
  --train-steps 3000 `
  --validation-every 250 `
  --checkpoint-every 250 `
  --ema-decay 0.999 `
  --require-gate any-scale
```

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_multiscale_learnability `
  --runs-root runs/experiment12_d0_multiscale_learnability `
  --run-name production-multiscale-confirmation `
  --study-profile confirmation `
  --device cuda `
  --stage cache-preflight `
  --zero-residual-run-dir runs/experiment12_d0_zero_residual/20260715-131054_production-manuscript-budget `
  --parent-one-image-run-dir runs/experiment12_d0_one_image/20260715-183915_production-one-image-direct-doob-preflight `
  --parent-multiscale-run-dir runs/experiment12_d0_multiscale_learnability/20260716-090351_production-multiscale-learnability `
  --dataset-seed 260718 `
  --require-gate cache
```

```powershell
$confirmationRun = "runs/experiment12_d0_multiscale_learnability/<timestamp>_production-multiscale-confirmation"
.\.venv\Scripts\python.exe -m mnist.diag_d0_multiscale_learnability `
  --study-profile confirmation `
  --device cuda `
  --stage all `
  --resume-run-dir $confirmationRun `
  --zero-residual-run-dir runs/experiment12_d0_zero_residual/20260715-131054_production-manuscript-budget `
  --parent-one-image-run-dir runs/experiment12_d0_one_image/20260715-183915_production-one-image-direct-doob-preflight `
  --parent-multiscale-run-dir runs/experiment12_d0_multiscale_learnability/20260716-090351_production-multiscale-learnability `
  --dataset-seed 260718 `
  --require-gate any-scale
```
