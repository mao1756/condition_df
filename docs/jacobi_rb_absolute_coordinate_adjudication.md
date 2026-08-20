# Read-only Jacobi/RB absolute-coordinate adjudication

## Purpose

The terminal quartile-directional experiment ended with
`representation_cancellation_nonidentifying_stop`.  That result is a valid
stop inside the frozen local-affine-plus-spatial-CNN class; it is not evidence
that the exact Rao--Blackwell target has zero conditional mean.

The independent physical coarse-witness panels already establish a positive
conditional-moment lower bound at the resolution

```text
(forward-time quartile, split phase, absolute oriented edge).
```

The frozen predictor, however, has no absolute-coordinate channels.  Its
spatial branch is three shared circular 3x3 convolutions (a 7x7 receptive
field), and its local-affine branch shares one map across all edges.  This
workflow therefore asks the narrower historical question: does a fixed
periodic absolute-coordinate subspace carry a direction that transfers from
coarse-witness panel A to independent panel B in every quartile?

The answer is design evidence only.  The panels and the representation
hypothesis have already been inspected, so even a positive result can
recommend only drafting a separately reviewed fresh learner plan.

## Workflow

The additive module is

```text
mnist.diag_d0_jacobi_rb_absolute_coordinate_adjudication
```

with sealed stages

```text
preflight -> replay -> symmetry -> decompose -> report
```

It generates no paths or transitions, performs no optimizer update, creates
no model checkpoint, and opens no selection or confirmation role.

## Immutable evidence

The first parent is the verified portable continuation archive

```text
20260808-135158_production-runpod-quartile-directional-continuation.zip
```

with archive SHA-256
`0f9914b79011a1182bac8fd9645e7ac0e222618d5be92047c03268e8b9ab3f7d`,
2,041 registered artifacts, registry semantic SHA-256
`cf206b49a094ede6196fd794f945c8ecf616e3caf48ef12b32c31afc8cafea64`,
and decision `representation_cancellation_nonidentifying_stop`.

The second parent is the physical coarse witness

```text
runs/experiment12_d0_jacobi_rb_physical_coarse_signal_witness/
  20260730-135059_production-exact-k512-physical-coarse-signal-jsonfix
```

with two immutable 64-path arrays of shape `[64,4,7,392]` and decision
`exact_physical_coarse_signal_detected`.

The archive is verified in place.  Safe relative paths, a unique
case-insensitive inventory, CRCs, every registered size and SHA-256, the
registry semantic commitment, manifest/configuration commitments, terminal
status, and decision must all agree.  The witness registry, panel seals, and
panel hashes receive the same fail-closed treatment.

## Coordinate decomposition

For every phase, the 392 matching edges form a periodic oriented-edge
lattice.  A deterministic real orthonormal basis splits it into

```text
H0       phasewise DC/constant
H1-H0    axial sine/cosine modes at periodic frequency one
H2-H1    axial sine/cosine modes at periodic frequency two
residual all remaining edge-position structure
```

The QR sign convention, edge orientation, coordinate origin, and phase/color
mapping are fingerprinted.  Direct projection and coefficient-space
calculations must obey Parseval and reconstruct the signed total cross-panel
energy within the frozen float64 tolerance.  Negative estimates are retained.

Panel A supplies, for each quartile, the normalized frequency-one direction.
Those four directions are sealed before panel B is read.  Panel B contributes
one whole-path signed projection per quartile.  The primary family uses
50,000 deterministic whole-path bootstrap replicates, one-sided 99% max-T
bounds, and `higher` quantile interpolation.  q0 is the positive control;
q1--q3 are the later-time hypothesis tests.

The signed independent-panel cross-spectrum remains the unbiased energy
estimate.  Authorization does not rely on an asymptotic quadratic-null
z-score: the primary held-out statistic is linear conditional on the sealed
panel-A direction, and synthetic DC, single-mode, mixed-mode, and stationary
null fixtures must pass.

## Interpretation

Because absolute edge identity is a coarsening of the permitted output site,
a positive coordinate projection is an exact lower-bound witness of
conditional observability.  It identifies a useful feature family, not a
unique parameterization: periodic coordinate channels, Fourier features, and
per-edge effects span related hypotheses.

The result also cannot prove that the old equivariant CNN could never infer
position indirectly from state content.  The fixed-image physical law is not
translation invariant, so shifted-state tests are partly off support.  Every
report carries this caveat.

## Closed outcomes

Integrity outcomes are

```text
control_provenance_invalid
portable_directional_parent_invalid
coordinate_hypothesis_plan_invalid
coarse_witness_replay_invalid
translation_symmetry_audit_invalid
coordinate_projection_algebra_invalid
coordinate_inference_invalid
```

Valid scientific outcomes are

```text
coarse_signal_nonreplicating_stop
absolute_coordinate_signal_not_detected_stop
absolute_coordinate_signal_partial_stop
absolute_coordinate_representation_hypothesis_supported
```

Only the last outcome recommends drafting a fresh coordinate-aware learner
plan.  It does not authorize cache generation, training, confirmation,
controller execution, reconstruction, or sampling.

## Production commands

This workflow is CPU-only and normally completes in under a few minutes.  It
reads the verified ZIP directly; extraction is neither needed nor permitted.

```powershell
$directionalZip = (
  Resolve-Path "$HOME\Downloads\20260808-135158_production-runpod-quartile-directional-continuation.zip"
).Path
$witnessRun = (
  Resolve-Path "runs/experiment12_d0_jacobi_rb_physical_coarse_signal_witness/20260730-135059_production-exact-k512-physical-coarse-signal-jsonfix"
).Path

.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_absolute_coordinate_adjudication `
  --runs-root runs/experiment12_d0_jacobi_rb_absolute_coordinate_adjudication `
  --run-name production-read-only-absolute-coordinate-adjudication `
  --device cpu `
  --stage preflight `
  --parent-directional-result-archive $directionalZip `
  --parent-coarse-witness-run-dir $witnessRun `
  --require-gate preflight
```

Resolve the new run and execute each remaining gate only after the preceding
one passes:

```powershell
$coordinateRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_absolute_coordinate_adjudication" -Directory |
  Where-Object Name -Like "*_production-read-only-absolute-coordinate-adjudication" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName

foreach ($stage in @("replay", "symmetry", "decompose")) {
  .\.venv\Scripts\python.exe `
    -m mnist.diag_d0_jacobi_rb_absolute_coordinate_adjudication `
    --device cpu `
    --stage $stage `
    --resume-run-dir $coordinateRun `
    --parent-directional-result-archive $directionalZip `
    --parent-coarse-witness-run-dir $witnessRun `
    --require-gate $stage
  if ($LASTEXITCODE -ne 0) { break }
}
```

After `decompose` passes, render the terminal report:

```powershell
.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_absolute_coordinate_adjudication `
  --device cpu `
  --stage report `
  --resume-run-dir $coordinateRun `
  --parent-directional-result-archive $directionalZip `
  --parent-coarse-witness-run-dir $witnessRun
```
