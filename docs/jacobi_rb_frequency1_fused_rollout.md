# Exact fused frequency-one rollout on the laptop

Research mode: engineering/infrastructure immediately followed by an exploratory,
objective-bearing one-image reverse rollout.

This workflow removes the measured single-row scheduling blocker without changing
the checkpoint, controller, Jacobi transition, Rao--Blackwell target, controller
gain grid, anchors, common-random-number design, scientific thresholds, or the
six-hour laptop budget. A passing preflight continues directly into the objective
workflow in the same Python process. Scheduler throughput is a health result, not a
scientific success.

## Continuation and evidence authority

The workflow binds the immutable resource-stopped carrier
`20260812-005942_production-frequency1-exploratory-rollout-fbv2`. It verifies the
carrier manifest, checksum inventory, bundle audit, source/configuration, and
terminal resource adjudication. Only preflight path `0xFB000` was realized. The
unopened objective identities transfer unchanged:

- development: `0xFB100`;
- evaluation: `0xFB200`;
- optional replication: `0xFB300`.

The original v1 allocation assigned infrastructure-only paths `0xFC000` through
`0xFC003` and reserved the containing slot `[0xFC000,0xFC010)`. The immutable first
attempt described below realized `0xFC000` and `0xFC001`; `0xFC002` through
`0xFC00F` remain unopened. The carrier is read-only and is never resumed, copied
into, or amended.

## Row identity and random-transition identity

A fused row has two deliberately separate identities:

- `row_key` is unique and controls dispatch, artifact order, summaries, and resume;
- `canonical_path_id` constructs exact transition IDs and may repeat.

Zero, learned-gain, and oracle rows for one path therefore carry different row keys
but the same canonical path ID. At an equal outer step, phase, microstep, edge, and
reference role they receive identical transition IDs and share the exact Philox
random bits. Their states and exposures may differ, so their exact outcomes need not
be equal. Variant, gain, horizon, row index, and batch position never enter the
reference RNG key.

The fused API does not weaken the unique-path contract of existing singleton or
independent-multipath APIs.

## Frozen packing schedule

Development first creates the singleton exact forward path `0xFB100`. Its step-127
anchor starts one six-row family in this order: zero; learned gains 0.5, 1, 2, and 4;
oracle. The 128-step suffix uses 16 atomic eight-step shards and 8,429,568 logical
reference transitions. Gain selection is committed before evaluation is opened.

Evaluation then creates forward path `0xFB200`. Three full-horizon rows (zero,
selected learned, oracle) run together from outer step 511 through 128 in 48 shards.
At the boundary before step 127, their states are joined with three short rows from
the immutable step-127 forward anchor. The resulting six-row family runs the shared
127-through-0 suffix in 16 shards. The join record binds both inputs, row order,
controllers, RNG contract, and next coordinate.

Optional replication uses path `0xFB300` and exactly two full rows: zero and the
already selected learned gain. It is opened only under the existing positive
evaluation and two-hour resource conditions.

## Exact reference and shard telemetry

Each microstep remains reference half-step, exact tangent logistic flow, reference
half-step. The certified Jacobi sampler and binary64 target are unchanged. Fused
execution holds states and telemetry on CUDA through an eight-step shard, validates
one packed health record at the boundary, then atomically commits NPZ followed by
JSON. An invalid shard is not committed; the last valid state and failure artifacts
remain readable.

Each shard binds the ordered row table, duplicate canonical IDs, controller and
checkpoint/target commitments, RNG namespace, exact sequence, input/output hashes,
per-row and aggregate telemetry, elapsed time, and state archive. Resume verifies the
entire binding and hash chain. A committed shard is never sampled again; an orphaned
NPZ without its JSON commit is deterministic replay work.

The largest reference launch is `6*392=2352` lanes, below the 4096-lane cap. Row
permutation, predeclared chunking, singleton/fused execution, the P3-to-P6 join, and
restart are equivalence controls.

## Laptop resource gate

One untimed warm-up uses `0xFC000`. Three timed complete eight-step repeats are then
run for each profile, including state updates, controllers, health transfer, NPZ/JSON
commit, and restart verification:

| Profile | Path | Transitions/shard | Production shards |
|---|---:|---:|---:|
| `forward_p1` | `0xFC001` | 21,952 | 128 |
| `reverse_p3` | `0xFC002` | 263,424 | 48 |
| `reverse_p6` | `0xFC003` | 526,848 | 32 |

The slowest repeat in each profile is authorizing; repeats are never averaged or
rerun to obtain a favorable result. The exact projection is

```text
128 * slowest_forward_p1_seconds
+ 48 * slowest_reverse_p3_seconds
+ 32 * slowest_reverse_p6_seconds
+ 300 seconds
```

The projection must be at most 21,600 seconds; its effective rate over 32,313,344
transitions must be at least 1495.9881481481482/s; every profile must be at least
1300/s. Repeat hashes, exact equivalence, certification, conservation, fallback,
forbidden-event, memory (80%), and persisted-storage (2 GiB) gates must also pass.

Only a completed failure of the timed P1/P3/P6 projection establishes that the
unchanged exact M=2 rollout is infeasible under the current laptop/six-hour contract.
A failure before those profiles supports no resource conclusion. After an
authorizing resource failure, the next choice is to change that contract explicitly
or pivot the experiment—not another packing benchmark and not a suggestion to use
unavailable hardware.

## First production attempt and immutable adjudication

The fresh `--stage all` attempt
`20260812-065538_production-frequency1-exploratory-rollout-fused-laptop-v1`
is terminal and immutable. Its recorded decision is `rollout_integrity_invalid`
with failure code `fused_preflight_forward_health_invalid`, but the complete
artifact audit localizes that stop to an implementation-contract comparator defect:

- all 64 `0xFC001` forward-anchor shards and their hash chain verify;
- all `1,404,928` transitions were certified, with zero fallback and zero forbidden
  operations;
- maximum pair-mass and scheduler simplex errors were
  `3.469446951953614e-18` and `2.220446049250313e-16`;
- the saved states are finite, nonnegative binary64 arrays, and peak CUDA allocation
  was `4,422,144` of `8,546,484,224` bytes;
- `0xFC000` and `0xFC001` were realized, while `0xFC002` and `0xFC003` were not;
- objective roles `0xFB100`, `0xFB200`, and `0xFB300` remain unopened.

The 151-record artifact manifest has semantic SHA-256
`6d13ec893857d0d457fce827923f4e1b4fc746c245fef2f22747769102e2e882`
and file SHA-256
`25cd4e1660f2bab390d9ba4771c05acd389b764f2505acd503df891037ded71e`;
the manifest, 153-entry checksum inventory, and all registered hashes verify. No
objective-bearing reverse trajectory ran, so this stop does not reset the proxy-only
counter.

The adapter omitted aggregate transition/active-count fields when normalizing the
otherwise valid anchor diagnostics, producing sentinel `-1` counts. It also applied
the `1300` transitions/s timed-profile threshold to the untimed 512-step
infrastructure anchor. The observed anchor rate, `595.26049998513` transitions/s,
is advisory resource-risk telemetry only: the complete warm-up, the three frozen
P1/P3/P6 repeat families, and the authorizing projection never ran.

The additive adjudication is therefore
`fused_forward_anchor_diagnostics_aggregation_invalid`, with
`failure_domain="implementation_contract"`, `resource_valid="not_evaluated"`,
and `scientific_evidence_complete=0`. No learner or rollout conclusion follows.
The original run records remain unchanged.

The code now exports the missing counts and makes throughput enforcement explicit:
the FC001 anchor records throughput without authorizing on it, whereas every timed
profile retains the frozen throughput gate. Production-shaped regressions distinguish
anchor validity, timed-profile resource failure, and non-resource integrity failure.
The repaired 25-file source closure is
`bd3195e38c33ec56ba98a98382ee0f6537335e1807ccd306aa702026905f35ce`.

Do not resume or rerun the terminal directory below. Because `0xFC000` and `0xFC001`
are now historical realized identities and the source closure changed, another
production attempt requires a separately reviewed continuation binding and a fresh
preflight namespace. The still-unopened objective identities may be retained only
after that review.

## Historical production command (do not rerun)

This is the exact command that created the immutable stopped run; it is retained for
provenance, not as an executable next step:

```powershell
$frequency1Run = (Resolve-Path 'runs/experiment12_d0_jacobi_rb_boundary_tangent_frequency1_coordinate_learnability/20260811-010641_production-frequency1-coordinate-v1-one-image').Path
$sourceRun = (Resolve-Path 'runs/experiment12_d0_jacobi_rb_coarse_residual_learnability/20260731-140333_production-exact-k512-coarse-residual-one-image').Path
$carrier = (Resolve-Path 'runs/experiment12_d0_jacobi_rb_frequency1_rollout/20260812-005942_production-frequency1-exploratory-rollout-fbv2').Path

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_frequency1_rollout `
  --frequency1-run-dir $frequency1Run `
  --source-run-dir $sourceRun `
  --continuation-run-dir $carrier `
  --runs-root runs/experiment12_d0_jacobi_rb_frequency1_rollout `
  --run-name production-frequency1-exploratory-rollout-fused-laptop-v1 `
  --device cuda:0 `
  --stage all `
  --require-gate evaluation
```

The implementation continues to verify carrier, copied-input, source-closure,
scientific-configuration, family, controller, RNG, and shard-chain bindings on
resume. That generic capability does not authorize resuming this terminal attempt.

## Claim boundary

The saved raw states, intermediate anchors, paired endpoint metrics, and fixed-scale
images are exploratory one-image evidence under exact M=2 split dynamics. They do
not establish protected validation success, prior-start generation, M=8 behavior,
multi-image generalization, spatial Dirichlet--Ferguson convergence, or convergence
of the unsplit Eulerian generator. Failed yet numerically interpretable trajectories
remain part of the evidence.

## Objective-first recovery v3

### Immutable 23:33 v3 adjudication and fresh v4 successor

The terminal child
`20260812-233343_production-frequency1-objective-first-recovery-v3` is immutable.
It committed one certified exact eight-step, three-row FB100 audit shard before
stopping. The saved state is finite and nonnegative with maximum simplex error
`4.44e-16`; all `263424/263424` active transitions are certified, with zero
fallback, unauthorized, invalid, or forbidden events and maximum pair-mass error
`2.22e-16`. Peak CUDA allocation is only `46531584/8546484224` bytes. The stop was
therefore an implementation-contract false negative, not a numerical or resource
failure: the synchronous exact record exposed authoritative per-row active counts
but omitted their aggregate fields, while the recovery health consumer defaulted
the missing aggregates to zero.

The child is not resumed after the source-closure repair. FB100 is now a committed
historical realization and the fresh v4 collision scanner must remap the development
role. No candidate shard or candidate/exact discrepancy audit exists; FB200 and
FB300 remain fresh and unopened. The v3 terminal bundle and predecessor stay
unchanged.

The additive recovery workflow binds the immutable stopped run
`20260812-065538_production-frequency1-exploratory-rollout-fused-laptop-v1`,
reuses its verified step-127 and step-511 states, and makes the reverse trajectory
the first GPU workload. The primary mode is exploratory engineering immediately
followed by an objective-bearing experiment. No new preflight namespace is created.
FB100 is the preferred development role, but the v4 collision scanner remapped it
to FB101 because v3 had already committed FB100; FB200 is evaluation and FB300 is
optional future evidence.

The mandatory result is one paired 128-step family with zero control, the frozen
learned controller at gain 1.0, and the source-informed target-fraction diagnostic.
All three rows share the same canonical path ID and random bits. The source-informed
row is descriptive; only the analytic target-fraction identity is an implementation
gate, and the source row cannot suppress a learned-versus-zero result.

Backend choice is horizon-local. One certified exact eight-step shard is committed
and timed through NPZ/JSON commit and restart verification. If the remaining exact
horizon does not fit the 21,600-second active budget, `auto` restarts from the
original anchor with `candidate_approximate_v1`. That candidate is the fixed
128-mode, 56-bisection CUDA inverse-CDF proposal using the same transition IDs; it
does not claim certification or Arb fallback. Its first shard is compared with the
exact shard. A coarse candidate learned-effect interpretation additionally requires
paired-contrast relative error at most 0.25 and endpoint learned-zero squared-L2
separation at least four times the largest first-shard candidate/exact squared-L2
discrepancy. These are diagnostic claim guards and never hide the trajectory.

Optional gain expansion requires 20% headroom and occurs only after the mandatory
core artifacts are durable. FB200 then attempts fresh forward anchors while each
verified shard fits; otherwise it is explicitly historical-anchor stochastic
evaluation. Short evaluation precedes an affordable full path, and each horizon
repeats the exact-first/automatic-candidate policy. Optional resource stops preserve
the successful mandatory core. Actual setup, execution, commit, verification,
wasted active attempt time, storage, and peak memory are sealed in the final ledger;
idle time between invocations is not charged.

Historical v4 production command (completed; do not rerun), with no
authorization-only gate:

```powershell
$frequency1Run = (Resolve-Path 'runs/experiment12_d0_jacobi_rb_boundary_tangent_frequency1_coordinate_learnability/20260811-010641_production-frequency1-coordinate-v1-one-image').Path
$sourceRun = (Resolve-Path 'runs/experiment12_d0_jacobi_rb_coarse_residual_learnability/20260731-140333_production-exact-k512-coarse-residual-one-image').Path
$carrier = (Resolve-Path 'runs/experiment12_d0_jacobi_rb_frequency1_rollout/20260812-005942_production-frequency1-exploratory-rollout-fbv2').Path
$predecessor = (Resolve-Path 'runs/experiment12_d0_jacobi_rb_frequency1_rollout/20260812-065538_production-frequency1-exploratory-rollout-fused-laptop-v1').Path

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_frequency1_rollout `
  --frequency1-run-dir $frequency1Run `
  --source-run-dir $sourceRun `
  --continuation-run-dir $carrier `
  --predecessor-run-dir $predecessor `
  --runs-root runs/experiment12_d0_jacobi_rb_frequency1_rollout `
  --run-name production-frequency1-objective-first-recovery-v4 `
  --device cuda:0 `
  --reference-backend auto `
  --development-anchor predecessor `
  --core-learned-gain 1.0 `
  --gain-sweep auto `
  --exact-audit-outer-steps 8 `
  --maximum-main-seconds 21600 `
  --stage all
```

A successful terminal requires the mandatory 128-step raw states, fixed-scale
images, paired metrics, numerical health, resource ledger, report, and artifact
registry. It does not establish confirmation, prior-start generation, multi-image
generalization, or exact full-horizon equivalence for candidate trajectories.

### Fresh v4 production outcome

The completed child is
`runs/experiment12_d0_jacobi_rb_frequency1_rollout/20260813-002414_production-frequency1-objective-first-recovery-v4`.
It bound source closure
`cf27a97d85f335ec7fbd288e93946adefecd45ebfdedceeab6e0e68ac79b28c8`,
selected FB101 (`1028353`) for development after the FB100 collision, opened fresh
FB200 (`1028608`) evaluation, and left FB300 (`1028864`) unopened.

The mandatory exact 128-step development family was numerically valid. Final
squared-L2 errors for zero, learned gain 1, and source-informed rows were
`0.00318150204`, `0.00392860435`, and `0.0000542526755`. Thus learned control
worsened the paired endpoint by `0.000747102312` (23.48%), whereas the
source-informed controller visibly reconstructed the label-3 target. An exact gain
sweep selected `0.5` by minimum learned error, but all tested gains remained worse
than zero.

Fresh FB200 forward evidence committed 64 exact shards with
`1,404,928/1,404,928` active certified transitions, one fallback, and no forbidden
event. The exact 128-step evaluation family independently gave zero/learned/source
errors `0.00312099339`, `0.00323938544`, and `0.0000554996424`: selected gain 0.5
again worsened error, by `0.000118392043` (3.79%). The terminal outcome is
`evaluation_short_rollout_direction_not_useful`.

The optional full-horizon exact audit shard passed, but exact completion projected
`36,586.23` seconds and exceeded the hard cap. The candidate backend was selected,
then rejected before its first sampler call because its conservative shard-zero
projection exceeded the remaining budget. Consequently full evaluation was
resource-deferred: there is no candidate trajectory, full endpoint, or
exact-candidate discrepancy claim. The final ledger records `16,394.322` active
seconds, `0.173` wasted seconds, `10,598,962` bytes, and `46,694,400` peak CUDA
bytes. A source-bound report-only recovery sealed the terminal bundle without
running a sampler or retiming a committed shard.

This one-image exploratory result supports a learner direction/calibration or
representation problem rather than failure of the exact integrator/composition
path at 128 steps. It does not establish full-path behavior, prior matching,
multi-image generation, or a general negative claim about learned controllers. The
next objective patch should inspect sign/order and compare a materially different
learner/controller, such as a rollout-trained or global model.

## Successor: global-dilated exact fresh suffix

That required successor is complete. The fresh global-dilated v3 child
`runs/experiment12-d0-jacobi-rb-global-dilated-rollout/20260813-233915_production-global-dilated-exact-five-row-v3`
selected update 3100, opened FB300 path `1028864`, and ran exact paired zero,
v4 `+0.5`, v4 `-0.5`, global `+1`, and source-informed 128-step rows. Global
improved final squared-L2 by `7.45379%`; both v4 signs were adverse, while the
source-informed control improved `98.2113%`. Exact health and the terminal bundle
passed. The same-path complete branch was resource-deferred without a sampler call.

The design, implementation failures, successful result, artifacts, and current
claim boundary are documented in
[`jacobi_rb_global_dilated_rollout.md`](jacobi_rb_global_dilated_rollout.md).
Root `HANDOFF.md` now provides the sole planning handoff for a fresh-budget exact
same-path complete zero/global/source reconstruction. This historical v4 section and
its immutable run remain unchanged.
