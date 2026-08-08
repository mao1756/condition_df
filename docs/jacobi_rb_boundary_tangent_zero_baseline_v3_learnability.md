# Exact Jacobi/RB zero-baseline boundary-tangent learnability v3

## Scope

This additive workflow tests one representation repair and one prospective
selection repair for the fixed-grid, one-image boundary-tangent problem. The
representation is

```text
q_B := 0
m_theta(W) = y(1-y) q_theta(W).
```

The workflow keeps the exact certified binary64 Jacobi transition, the raw
Rao--Blackwell target, direct unweighted MSE, width-32 predictor, optimizer,
training horizon, and one-image law unchanged. It does not fit, store, or
reuse a baseline. It also separates physical checkpoint generation from
validation: training creates the fixed checkpoint grid without opening
physical validation labels, and the later `select` stage accounts
prospectively for every checkpoint and every confirmation-shaped contrast.

The restartable stages are

```text
preflight -> cache -> train -> select -> confirm -> report
```

This experiment performs no controller trajectory, complete reverse path,
reconstruction, image sampling, reverse sampling, or full-data training.

## Immutable evidence

Preflight verifies the complete artifact registries rather than trusting only
terminal summaries.

The immutable eager-prefix v2 parent is

```text
runs/experiment12_d0_jacobi_rb_boundary_tangent_eager_confirmation/
  20260803-113404_production-eager-boundary-tangent-time-local
decision: selection_false_discovery
source fingerprint:
  dfe9c3357c1d1ba614cccfdcaca84b3c3bf2d0967d6a3a3b15e5a0421d04243e
scientific configuration SHA-256:
  fadc1eb31ad0fb1ccb900f41f1eb8523c67c6ae39e09c783698aa5a20634cdec
artifact registry: 3457 records
artifact registry file SHA-256:
  c996bdce5935667d247b6ce24c5e88f008c6038ec42ae25b9ea74b8b64a9a0d4
artifact registry semantic SHA-256:
  36bf43c0a108549954617a78625d4fd65820141c950ba84330133de1f8648580
```

The completed read-only adjudication must report
`zero_baseline_v3_learnability_ready`,
`sealed_baseline_harm_confirmed`, and
`selected_update_below_resolution`. Its binding is

```text
source fingerprint:
  a5bfbfdd84292744aef1317fb28cfd56930d126764df72dbac3c3745eaa75968
scientific configuration SHA-256:
  f7b1ffc044bdb507a6563cf2dd836b31a5aff1090d66730cb672b1287700bd9c
artifact registry: 272 records
artifact registry file SHA-256:
  e2b6f95ddaded8001bcd2cbaf24075a4bfe8423fac75787b906a8465bc4cf12c
artifact registry semantic SHA-256:
  3aac15ae494ffc82ada769509dfaa8ef080444315fdccb045f7bae40fad5896a
decision file SHA-256:
  595630bb87e2bef77f946fb07071a6765267df88ae2950be45ac03fcd80a762e
```

The adjudication authorizes design only: it must say that the old
confirmation paths are burned, their reuse is forbidden, fresh v3 design is
authorized, and cache generation, physical training, confirmation, and
controller planning were not authorized within that completed child. The v3
workflow is a separately reviewed run and never mutates either parent. It
also binds the passing eager scheduler and successful coarse-residual
ancestors used by v2.

## Exact scientific and model contract

- Grid 28, alpha 1, `K=512`, `tau_eff=5e-5`, all 512 outer steps.
- First label-3 MNIST image, mixed with uniform mass at
  `lambda_mix=0.35`; both the source artifact and reconstructed tensor are
  hash-bound before CUDA work.
- Selected outer steps `15,31,...,495,511`; midpoint fractions
  `1/16,3/16,...,15/16`.
- Exact raw target
  `Z_bar = y(1-y) d_y log k_u(y|x)` and plain direct MSE, normalized only by
  the RMS of physical training targets.
- Width-32 `JacobiRBPhasePredictor`, Adam at `1e-3`, batch 32, prediction
  batch 32, zero weight decay, unit gradient clipping, no mixed precision,
  and updates `100,200,...,4000` for three seeds.
- Permitted inputs remain later full state, reverse time, phase, color,
  duration, and fixed class label. Outer step, midpoint index/fraction,
  earlier states, certificates, random bits, and oracle values are not model
  inputs.
- The final layer is exactly zero at update zero. The model has no baseline
  object, baseline-named module, or `_q_values` buffer. Its state dictionary
  contains only the residual network.
- The conceptual C-order float64 zero array of shape `[4,7,8,392]` is not
  persisted; its required SHA-256 is
  `a0cfe4ce7c13acb57ced3803a69321b59b790ae5ec652a6c03476676d6204149`.
- Target clipping or weighting, quotient targets, floors, limiters,
  projections, classifiers, Euler/Gaussian proxies, reverse residuals, fitted
  control variates, and baseline-fit artifacts are forbidden.

## Fresh evidence allocation

The frozen seeds are

```text
root paths                 261311
physical models            261312, 261313, 261314
selection bootstrap        261320, namespace 0x42545633 ("BTV3")
confirmation bootstrap     261322, namespace 0x42544333 ("BTC3")
synthetic teacher          261323
exact-model null           261324
reserved future control    261325
forbidden scheduler seed   261321
```

Fresh 20-bit path slots are frozen before CUDA initialization:

```text
preflight seam  0xF0000-0xF0007   8 paths
training        0xF1000-0xF103F  64 paths
validation      0xF1100-0xF111F  32 paths
confirmation    0xF2000-0xF203F  64 paths, reserved but unopened
```

They are valid suballocations of the historical
`0xF0000-0xFFFFF` allocator reservation, but must be disjoint from every
active or historical realized role. The old v2 confirmation range
`0xED000-0xED03F` is permanently burned. Confirmation reservation is not an
opening: no confirmation artifact or transition may exist until a nonzero
validation nominee is sealed and `confirmation_namespace_open.json` is
committed.

Training and validation use the stable `[10x9,6]` cohort plan. One CUDA cohort
crosses the role boundary, but every payload is split by immutable role before
artifact commit. Confirmation, if authorized, uses `[10x6,4]` separately.

## Stage contracts

### Preflight

All provenance, authorization, immutability, path-collision, scientific
configuration, cohort, zero-baseline, target/input, and source-closure records
are committed before CUDA configuration. Preflight executes only the eight
fresh seam paths. It verifies the exact eager kernel and certificate contract,
source image, selected samples, model-input firewall, absence of baseline
state, exact-zero update-zero output, determinism, and inherited resource
projection. It may not open train, validation, or confirmation paths.

### Cache

Cache creates only 64 training and 32 validation paths: exactly 114,688 and
57,344 selected rows. Permitted inputs are float32 and raw labels are float64
in separate role-specific artifacts. The gate inherits all transition,
certificate, conservation, numerical, persistence, and resource checks from
v2; requires the complete Cartesian sample identity; and rejects any baseline
fit or confirmation artifact.

### Train

Before physical labels open, three controls must pass:

1. Update-zero predictions, baseline diagnostics, and facet outputs are
   exactly zero on all permitted train/validation inputs.
2. The synthetic tangent teacher has relative validation MSE at most `0.01`
   and beats zero on every synthetic validation path.
3. An exactly representable, positive-energy model-null teacher is cloned
   into the student; loss and gradients are exactly zero, update zero is
   selected, and all parameter tensors remain bitwise unchanged.

Only then are physical training labels opened. The physical trainer receives
training inputs and labels only; it cannot access physical validation inputs,
labels, or indexes. It trains all three seeds through update 4000 and saves
update zero plus every 100 updates, producing exactly 120 nonzero candidates.
Task records contain no validation metric, eligibility, or selected field.

### Select

Before validation labels open, the run commits the candidate order and hashes,
228 family names, 27,360 flattening rule, inferential settings, environment
binding, and all bootstrap-count shards. Validation then evaluates all three
exact-zero checkpoints and all 120 nonzero checkpoints on the fresh 32 paths.
Each nonzero candidate is a separately committed restart unit.

The nonzero candidate family is

```text
3 seeds x 40 updates x 228 contrasts = 27,360 components.
```

The 228 contrasts are 224
`model_vs_zero.q{0..3}.phase{0..6}.midpoint{0..7}` cells followed by four
`model_vs_zero.q{0..3}.pooled` quartiles. Quartile is audit-only
`outer_step // 128`. Each row contains

```text
mean_edges(target^2 - (target - prediction)^2).
```

There is no baseline or residual-versus-baseline contrast. A nonzero
candidate qualifies only if all 228 search-adjusted lower bounds are strictly
positive. Rank qualifiers by largest minimum lower bound, then earlier update,
then lower seed. Update zero competes as the exact logical null but is excluded
from studentization because all its contrasts and standard errors are exactly
zero. If no nonzero candidate qualifies, seal `no_validation_candidate` and
never open confirmation.

### Confirm and report

A passing select stage seals one nonzero checkpoint. The confirmation-open
record and 50 fixed count shards are committed before the first transition,
permanently burning `0xF2000-0xF203F`. The 64 paths are streamed once; raw
confirmation inputs and labels are never persisted. The same numeric max-T
core evaluates the same 228 all-versus-zero contrasts. Every simultaneous
lower bound must be strictly positive. There is no re-selection, threshold
change, second confirmation, or reuse of v2 evidence.

`report` performs no scientific computation. It verifies all stage seals,
ordering, namespace semantics, source closure, registry, and parent
immutability before finalizing status and the workflow decision.

## Restartable max-T computation

Validation stores `X[path,candidate,component]` with shape `[32,120,228]`.
Whole-path bootstrap resamples are represented by multiplicity rows, shared
across every candidate and component. The frozen design is:

```text
replicates             50,000
count/maxima shards        50 x 1,000
candidate blocks             6 x 20
component blocks             4 x 57
working block family         1,140
confidence                 0.995 one-sided
quantile interpolation     higher
negative truncation        none
peak working memory        below 64 MiB
```

Count shard `i` uses
`np.random.Generator(np.random.Philox([seed, namespace, i]))`, draws a
`[1000,path_count]` integer index table, and stores `uint8` multiplicities.
Every row sums to the path count. NumPy, byte order, Philox constructor, and
CPU/BLAS environment are resume-bound. The count artifacts are committed
before labels open; maxima shards are restart units and store only 1,000
float64 maxima plus their bindings. Confirmation applies the same procedure
with 64 paths and one candidate.

No standard-error floor is allowed. A nonfinite or nonpositive observed or
bootstrap standard error for a nonzero component ends
`validation_inference_invalid`.

## Gates, decisions, and restart rules

Closed decisions, in precedence order, are:

- `provenance_or_path_plan_invalid`;
- `zero_baseline_contract_invalid`;
- `exact_cache_invalid`;
- `training_controls_failed`;
- `physical_training_invalid`;
- `validation_inference_invalid`;
- `no_validation_candidate`;
- `fresh_confirmation_invalid`;
- `zero_baseline_v3_signal_not_confirmed`;
- `exact_rb_zero_baseline_boundary_tangent_time_local_signal_confirmed`.

Only the final decision authorizes planning a separate controls-only,
at-most-eight-phase controller-control patch. A sealed validation nominee
authorizes only the single confirmation in this workflow.

Resume re-verifies source closure, parent registries, adjudication, scientific
configuration, path/cohort plans, zero-baseline contract, and every completed
seal before mutation. Completed scientific stages are immutable. Cache shards
use metadata-last commits; physical training checkpoints and progress are
atomic per seed; candidate metadata commits a validation replay. Before its
label/namespace open record, a missing stateless bootstrap count shard may be
regenerated; afterward a missing or changed count is fatal. Missing
uncommitted maxima may be recomputed from committed counts. Once selection is
sealed it cannot change. Once confirmation opens, the same IDs and checkpoint
must be resumed and no replacement audit may be allocated. Expected gate
failures and unexpected execution failures commit readable gate, seal,
decision, status, and registry evidence before returning nonzero.

Implementation validation on 2026-08-05 passed the 74-test focused patch
suite, all 340 boundary-tangent regressions, and the complete 1,886-test
repository suite. Ruff is not installed in the project virtual environment;
Python compilation and `git diff --check` passed.

## Production commands

Resolve the immutable parents and adjudication, then freeze the deterministic
process environment in the shell used for every stage:

```powershell
$v2Run = (Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_eager_confirmation/20260803-113404_production-eager-boundary-tangent-time-local").Path
$eagerRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_eager_pipeline_confirmation/20260803-034008_production-eager-prefix-complete-pipeline").Path
$coarseRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_coarse_residual_learnability/20260731-140333_production-exact-k512-coarse-residual-one-image").Path

$adjudicationRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_false_discovery_adjudication/20260805-125856_production-sealed-false-discovery-adjudication").Path

$env:PYTHONHASHSEED = "0"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
```

Run preflight and resolve its directory:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_boundary_tangent_v3_learnability `
  --runs-root runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_learnability `
  --run-name production-zero-baseline-v3-learnability `
  --device cuda --stage preflight `
  --parent-v2-run-dir $v2Run --adjudication-run-dir $adjudicationRun `
  --parent-eager-pipeline-run-dir $eagerRun --parent-coarse-residual-run-dir $coarseRun `
  --require-gate preflight
if ($LASTEXITCODE -ne 0) { throw "v3 preflight did not pass" }

$v3Run = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_learnability" -Directory |
  Where-Object Name -Like "*_production-zero-baseline-v3-learnability" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName
```

Run cache and train, inspecting each committed gate before continuing:

```powershell
foreach ($stage in @("cache", "train")) {
  .\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_boundary_tangent_v3_learnability `
    --device cuda --stage $stage --resume-run-dir $v3Run `
    --parent-v2-run-dir $v2Run --adjudication-run-dir $adjudicationRun `
    --parent-eager-pipeline-run-dir $eagerRun --parent-coarse-residual-run-dir $coarseRun `
    --require-gate $stage
  if ($LASTEXITCODE -ne 0) { throw "v3 $stage did not pass" }
}
```

Run selection separately. A committed `no_validation_candidate` is a valid
negative scientific result and forbids confirmation:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_boundary_tangent_v3_learnability `
  --device cuda --stage select --resume-run-dir $v3Run `
  --parent-v2-run-dir $v2Run --adjudication-run-dir $adjudicationRun `
  --parent-eager-pipeline-run-dir $eagerRun --parent-coarse-residual-run-dir $coarseRun `
  --require-gate select
$selectExit = $LASTEXITCODE
$selectDecision = Get-Content (Join-Path $v3Run "boundary_tangent_v3_decision.json") -Raw | ConvertFrom-Json
if (($selectExit -ne 0) -and ($selectDecision.decision -ne "no_validation_candidate")) {
  throw "v3 select failed with $($selectDecision.decision)"
}
$confirmationAuthorized = $selectExit -eq 0
```

Only a sealed nonzero nominee may open the single confirmation:

```powershell
if (-not $confirmationAuthorized) { throw "No nonzero nominee; confirmation is forbidden" }
$selection = Get-Content (Join-Path $v3Run "validation_selection.json") -Raw | ConvertFrom-Json
if ($selection.decision -ne "zero_baseline_v3_validation_nominee_sealed") {
  throw "Selection does not authorize confirmation"
}

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_boundary_tangent_v3_learnability `
  --device cuda --stage confirm --resume-run-dir $v3Run `
  --parent-v2-run-dir $v2Run --adjudication-run-dir $adjudicationRun `
  --parent-eager-pipeline-run-dir $eagerRun --parent-coarse-residual-run-dir $coarseRun `
  --require-gate confirm
```

An interrupted confirmation is resumed with the same command and run
directory; it must not allocate a new audit. Finally, verify the report:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_boundary_tangent_v3_learnability `
  --stage report --resume-run-dir $v3Run `
  --parent-v2-run-dir $v2Run --adjudication-run-dir $adjudicationRun `
  --parent-eager-pipeline-run-dir $eagerRun --parent-coarse-residual-run-dir $coarseRun `
  --require-gate none
```

## Restricted claim

A pass establishes only that one width-32 phase-conditioned predictor has a
fresh, search-adjusted, time-local all-versus-zero signal for one frozen image
under the exact fixed-`K=512` Jacobi split chain. It does not establish an
executable reverse controller, reconstruction, sample quality, a known prior,
full-data generalization, convergence to the unsplit Eulerian generator, or
spatial Dirichlet--Ferguson convergence.
