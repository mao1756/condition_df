# Frequency-one coordinate Jacobi/RB learnability

## Scope

This additive workflow tests one representation change: a frozen periodic
frequency-one coordinate field is injected into the spatial branch of the
existing width-32 Jacobi/Rao--Blackwell predictor.  The exact Jacobi law, raw
Rao--Blackwell label, zero tangent baseline, plain unweighted MSE, optimizer,
training horizon, path counts, and inferential thresholds remain unchanged.

The scientific question is whether this 128-parameter symmetry-breaking stem
produces a nonzero checkpoint whose complete 228-member all-versus-zero
family is simultaneously positive on fresh validation and one fresh sealed
confirmation.  This is a one-image, fixed-grid, fixed-`K=512` learnability
gate.  It is not a controller, reconstruction, or sampling experiment.

## Immutable rationale and protocol parents

The direct design parent is

```text
runs/experiment12_d0_jacobi_rb_absolute_coordinate_adjudication/
  20260810-211949_production-read-only-absolute-coordinate-adjudication
```

with decision `absolute_coordinate_representation_hypothesis_supported`, 40
registered artifacts, registry semantic SHA-256
`77f7245cd52b8d4e210f75ceb17f2bf3c0b923bbbabde26173e9409a0d6d9218`,
and registry-file SHA-256
`d05804f468c6485c234a6f6f66d55e3c3075e85a9172efbd1af2ab5654a84158`.
Its independently held-out q0--q3 frequency-one lower bounds are all positive.
That historical result is design evidence only.

The prospective protocol parent is the memory-safe zero-baseline v3 run

```text
runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_memory_confirmation/
  20260806-181326_production-zero-baseline-v3-memory-safe
```

with decision `no_validation_candidate`.  The workflow also verifies the
exact physical coarse witness
`20260730-135059_production-exact-k512-physical-coarse-signal-jsonfix`
(registry semantic SHA-256
`ca405ea7c00d6efed470c0651b8ad28d31f797cf142a1bef5f75d464ee1c3ab3`)
and the portable directional terminal archive
`20260808-135158_production-runpod-quartile-directional-continuation.zip`
(archive SHA-256
`0f9914b79011a1182bac8fd9645e7ac0e222618d5be92047c03268e8b9ab3f7d`).
The memory-v3 registry semantic SHA-256 is
`a69ca33be9c3281eb54e9285f3d292d9e8c9cd0775781ed652af0a1adda85626`;
its source and scientific-configuration fingerprints are
`5cde21b7ed36a806f2e872cf8fb3f7ac859d9317ab87aad1e728b95d544a2cee`
and `bbb3b79cc7afc6311b6e6413cde0e7f93c074f8fe885bd4960140bc39e44f61e`.
The design-parent source/configuration fingerprints are
`2a9ba3b2b5078665aaadcab036185f3b5ae8798e1672bc372ff08d32714a707f`
and `8c8eb7e9ee7bf6251dbfc478a7bf2c409b4cbfb7a8f3e55523a68d56040345dc`.
All transitive specialist and time-local provenance is reverified. Parent
trees and archives remain immutable and are rechecked before each mutating
stage.

The remaining direct commitments are frozen as follows:

| Parent | Registry-file SHA-256 | Source/config commitment |
|---|---|---|
| Absolute-coordinate design | `d05804f468c6485c234a6f6f66d55e3c3075e85a9172efbd1af2ab5654a84158` | source/config shown above |
| Memory-safe v3 protocol | `a3a697a7a92f2ad6f0cc666d95ebb92b11dfd8c6837005d457356bd326b79076` | source/config shown above |
| Coarse witness | `866984822ef14dbb13f0644a0f23697f6fb42ecec40b07341249d74043319747` | source `31f1f15008c2db864e282c5d3fa047986a9b576b92c480d50a18d55138e9eafb`; config `b2e28989ef6da6fa2d233b14ee475c04e10326079cf03750f1f427494de90f14` |
| Portable directional result | `3fe04ff3d4a8a5231f6588b8383610e8d283492774f0e7bbe2146824550f50b6` | registry semantic `cf206b49a094ede6196fd794f945c8ecf616e3caf48ef12b32c31afc8cafea64`; config `00f2464129b4c4dcfbd727aed97173abcc59e0e29697b77bafa76d1d28c0d39e` |

## Coordinate-aware predictor

The four frozen planes are

```text
sin(2*pi*row/28), cos(2*pi*row/28),
sin(2*pi*column/28), cos(2*pi*column/28).
```

They use row-major vertex coordinates, the active edge's head site, and the
existing phase/color orientation.  Their binary64 values are committed as
IEEE-754 hexadecimal literals; production forward passes do not call libm or
QR.  Preflight proves that the active-head restriction has rank four and the
same span as the sealed diagnostic frequency-one space.

The old 24-channel first convolution is not widened.  A bias-free parameter

```text
coordinate_stem_weight: [32, 4, 1, 1]
```

is appended after all inherited parameters and initialized with exact zeros.
Its output is added to the old first-convolution preactivation before SiLU.
The three circular 3x3 convolutions, width 32, four-output spatial head, and
25-feature local affine head are otherwise unchanged.  The model therefore
has 25,726 trainable parameters instead of 25,598.  Exact-zero stem weights
map every old state tensor and update-zero function into the new model without
changing inherited construction RNG state.

```text
old 24 state/metadata planes ----> old circular conv1 ----+
                                                          +--> SiLU --> conv2 --> conv3 --> spatial head
frozen 4 frequency-one planes --> zero 1x1 stem ----------+

tail/head masses + unchanged metadata -------------------------> unchanged local affine head
```

The coordinate field is persistent model state, not cached evidence or a
call-site argument.  Permitted inputs remain later full state, reverse time,
phase/color/duration, fixed label, and the internally frozen output-site
coordinate.  Earlier state, audit outer step, midpoint identity, path/sample
identity, RNG bits, certificates, transition internals, targets, parent
directions, and oracle values remain forbidden.

## Fresh roles and seeds

The frozen path roles are:

| Role | IDs | Count |
|---|---|---:|
| Preflight seam | `0xF8000--0xF8007` | 8 |
| Train | `0xF8100--0xF813F` | 64 |
| Validation | `0xF8200--0xF821F` | 32 |
| Confirmation | `0xF9000--0xF903F` | 64 |

Root path seed is `261371`; model seeds are `261372`, `261373`, and
`261374`.  Selection uses seed `261380` and namespace `0x46435631`;
confirmation uses seed `261382` and namespace `0x46434331`.  Synthetic,
null, and initialization controls use `261383`, `261384`, and `261385`.
The confirmation range is reserved at initialization but is burned only by a
sealed nonzero validation nominee.

## Stages and firewalls

Production stages run separately:

```text
preflight -> cache -> controls -> train -> select -> confirm -> report
```

Production `all` is rejected.  Cache creates fresh train/validation inputs
and labels but keeps each role and payload type physically separate.
Validation labels remain unopened until the prospective candidate family and
bootstrap design are sealed.  Controls use inputs and synthetic targets only;
physical train labels open only after all controls pass.  No confirmation
transition or evidence is created unless selection seals a nonzero nominee.

Training uses batch 32, Adam at `1e-3`, no weight decay or AMP, unit global
gradient clipping, exactly 4,000 updates, and checkpoints at update zero and
every 100 updates.  The target scale is the RMS of physical training labels
only.  All three seeds run to completion without early stopping.

The synthetic coordinate teacher and exact-model null prove learnability,
stem connectivity, zero-loss/zero-gradient behavior, batching, memory,
firewalls, and exact resume before physical labels open.

Completed stage seals are immutable. Cache shards use metadata-last atomic
commits; training resumes from its last exact 100-update boundary; candidate
evaluation and bootstrap shards are restartable under their frozen
seed/namespace. After validation or confirmation is opened, missing or
changed committed bootstrap evidence fails closed. A sealed nominee and its
confirmation IDs cannot be replaced.

## Inference

Validation jointly covers 120 nonzero candidates times 228 components:

```text
224 model-vs-zero quartile/phase/midpoint cells
  4 model-vs-zero pooled quartiles
----------------------------------------------
228 components x 120 checkpoints = 27,360
```

The whole-path one-sided 99.5% max-T procedure uses 50,000 Philox bootstrap
replicates, `higher` quantile interpolation, no negative truncation, and no
standard-error floor.  A candidate qualifies only when all 228 adjusted lower
bounds are strictly positive.  Qualifiers rank by largest minimum lower
bound, then earlier update, then lower seed.

The single sealed nominee is evaluated once on 64 fresh confirmation paths
with the same 228-member, 99.5% max-T contract.  Raw confirmation inputs,
labels, and predictions are never persisted; only restart-safe path-level
reductions and inferential artifacts are retained.

## Closed decisions

In precedence order:

1. `frequency1_coordinate_parent_provenance_invalid`
2. `frequency1_coordinate_contract_invalid`
3. `frequency1_coordinate_path_or_resource_plan_invalid`
4. `frequency1_coordinate_exact_cache_invalid`
5. `frequency1_coordinate_prelabel_controls_failed`
6. `frequency1_coordinate_physical_training_invalid`
7. `frequency1_coordinate_validation_inference_invalid`
8. `no_frequency1_coordinate_validation_candidate`
9. `frequency1_coordinate_fresh_confirmation_invalid`
10. `frequency1_coordinate_signal_not_confirmed`
11. `exact_rb_frequency1_coordinate_boundary_tangent_time_local_signal_confirmed`

Decisions 8 and 10 are valid scientific negatives.  Only decision 11 permits
drafting a separate controls-only controller patch.  It does not authorize a
controller trajectory, full reverse path, reconstruction, or sampling.

## Production commands

Resolve the four immutable parents and freeze the inherited deterministic
process settings:

```powershell
$absoluteCoordinateRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_absolute_coordinate_adjudication/20260810-211949_production-read-only-absolute-coordinate-adjudication").Path
$memoryV3Run = (Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_memory_confirmation/20260806-181326_production-zero-baseline-v3-memory-safe").Path
$coarseWitnessRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_physical_coarse_signal_witness/20260730-135059_production-exact-k512-physical-coarse-signal-jsonfix").Path
$directionalZip = (Resolve-Path "$HOME\Downloads\20260808-135158_production-runpod-quartile-directional-continuation.zip").Path

$env:PYTHONHASHSEED = "0"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
```

Create the run with:

```powershell
.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_boundary_tangent_frequency1_coordinate_learnability `
  --runs-root runs/experiment12_d0_jacobi_rb_boundary_tangent_frequency1_coordinate_learnability `
  --run-name production-frequency1-coordinate-v1-one-image `
  --device cuda `
  --stage preflight `
  --require-gate preflight `
  --parent-absolute-coordinate-run-dir $absoluteCoordinateRun `
  --parent-memory-v3-run-dir $memoryV3Run `
  --parent-coarse-witness-run-dir $coarseWitnessRun `
  --parent-directional-result-archive $directionalZip
```

Resolve the unique printed directory:

```powershell
$coordinateRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_boundary_tangent_frequency1_coordinate_learnability" -Directory |
  Where-Object Name -Like "*_production-frequency1-coordinate-v1-one-image" |
  Sort-Object Name -Descending |
  Select-Object -First 1
).FullName
```

Resume it one reviewed stage at a time, repeating all four parent bindings:

```powershell
foreach ($stage in @("cache", "controls", "train", "select")) {
  .\.venv\Scripts\python.exe `
    -m mnist.diag_d0_jacobi_rb_boundary_tangent_frequency1_coordinate_learnability `
    --resume-run-dir $coordinateRun `
    --device cuda `
    --stage $stage `
    --require-gate $stage `
    --parent-absolute-coordinate-run-dir $absoluteCoordinateRun `
    --parent-memory-v3-run-dir $memoryV3Run `
    --parent-coarse-witness-run-dir $coarseWitnessRun `
    --parent-directional-result-archive $directionalZip
  if ($LASTEXITCODE -ne 0) { break }
}
```

Run `confirm` only when selection reports
`frequency1_coordinate_validation_nominee_sealed`; otherwise the negative is
terminal and confirmation must remain unopened. Finally run `report` with
`--require-gate terminal`. Report-only execution may use CPU; every physical
stage is CUDA-only.
