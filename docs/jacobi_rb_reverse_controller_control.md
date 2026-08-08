# Exact Jacobi/Rao--Blackwell reverse-controller control

This additive Experiment 12 workflow tests whether the sealed one-image
coarse-baseline-plus-residual predictor can be used as a time-local reverse
controller. It performs no complete reverse path, image sampling,
reconstruction, checkpoint selection, or new training.

## Immutable parent

The workflow binds

`runs/experiment12_d0_jacobi_rb_coarse_residual_learnability/20260731-140333_production-exact-k512-coarse-residual-one-image`

and requires its terminal decision
`exact_rb_coarse_residual_learnable`. In particular it freezes:

- selected seed/update `261254/3000`;
- selected checkpoint SHA-256
  `24a0893daa31196815463a7396220542003e7dc2557689950ba4dd0eeaa9c914`;
- selected state SHA-256
  `df479e979cf6dd99580bd918377405b665791a4608f45f6cae326cc10e5e6ad9`;
- frozen coarse-table value SHA-256
  `5d4e73153c36a59e26403439befd4e13b7f4fe096f7cbf9af6b77ac26565a9df`;
- grid 28, alpha 1, `K=512`, `tau_eff=5e-5`, the seven phase
  occurrences `(0,1,2,3,2,1,0)`, and durations
  `(1/2,1/2,1/2,1,1/2,1/2,1/2)`.

The certified Jacobi implementation, Rao--Blackwell target, checkpoint,
coarse table, and historical artifacts remain byte-identical.

## Controller under test

For an oriented edge with pair mass `r`, head fraction `y`, and exact phase
exposure `u`, the sealed model predicts

\[
m(S,e,u)=\mathbb E[\bar Z\mid S_{\rm later}=S,e,u]
=y(1-y)\partial_y\log\rho_u,
\qquad \bar Z=L-MY.
\]

The exposure-time reverse generator adds the drift `2m`. One numerical
microstep uses a certified exact reference half-transition, the affine frozen
midpoint flow

\[
y^+=y+2m_{\rm mid}\,\delta u,
\]

and another certified exact reference half-transition. A learned flow that
leaves `[0,1]` is rejected; it is never clipped, projected, floored, limited,
or renormalized. Production is fixed at eight microsteps per phase. Two and
four microsteps are controls only.

The corresponding physical learned head-mass flux at alpha 1 is

\[
\frac{d s_{\rm head}}{dt}=\frac{6a(t)}{h^2}m,
\]

so pair mass cancels from the learned physical flux while remaining in the
exact reference exposure.

## Time-coordinate correction

The historical report's field called `data_end` selected forward quartile 3.
That is actually terminal-near and the beginning of reverse evolution.
Historical files are unchanged. New records use explicit
`forward_outer_quartile`, `reverse_quartile`, and `reverse_start` names.

The fractional adapter derives its coarse row using only permitted reverse
time and phase occurrence. It does not expose forward outer step or any audit
field to the model. It must reproduce the old combined predictor bitwise at
all trained endpoint coordinates.

## Stages and evidence

1. `preflight` verifies immutable provenance, endpoint equivalence, formula,
   sign/orientation, boundary/conservation, namespace/restart invariance, and
   a 40-hour resource projection.
2. `oracle` tests the stationary null and a bounded analytic linear teacher
   against an exact reverse conditional comparator.
3. `cache` generates 64 fresh exact forward paths and independent certified
   partial-phase labels at every fixed `M=8` midpoint. A 228-member one-sided
   99.5% max-T family requires every time/phase/fraction cell to beat zero and
   every forward quartile to beat the frozen baseline.
4. `control` runs paired one-phase and eight-phase reverse-law controls for
   `M=2,4,8`. A fixed 784-member two-sided 99.5% max-T family gates weak-law
   bias and `M=4` to `M=8` refinement.
5. `decide` issues a closed terminal decision from the already sealed gates.

The fresh physical path block is `0xEB000--0xEB03F`; preflight and oracle use
disjoint `0xEA000` and `0xEE000` blocks. `0xEC000`, `0xED000`, and the existing
`0xF0000` production reservation remain untouched.

## Production commands

```powershell
$parentRun = "runs/experiment12_d0_jacobi_rb_coarse_residual_learnability/" +
  "20260731-140333_production-exact-k512-coarse-residual-one-image"
$controllerRoot = "runs/experiment12_d0_jacobi_rb_reverse_controller_control"

.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_reverse_controller `
  --runs-root $controllerRoot `
  --run-name production-exact-rb-reverse-controller-control `
  --device cuda `
  --stage preflight `
  --parent-coarse-residual-run-dir $parentRun `
  --require-gate preflight
```

Resolve the printed directory once:

```powershell
$controllerRun = (Get-ChildItem $controllerRoot -Directory |
  Where-Object Name -Like "*_production-exact-rb-reverse-controller-control" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1).FullName
```

Then run the remaining stages in order, only after each required gate passes:

```powershell
foreach ($stage in @("oracle", "cache", "control", "decide")) {
  .\.venv\Scripts\python.exe `
    -m mnist.diag_d0_jacobi_rb_reverse_controller `
    --device cuda `
    --stage $stage `
    --resume-run-dir $controllerRun `
    --parent-coarse-residual-run-dir $parentRun `
    --require-gate $stage
  if ($LASTEXITCODE -ne 0) { break }
}
```

For the clearest scientific audit, running the four commands individually is
preferred over the convenience loop.

## Closed claim

Only `exact_rb_time_local_reverse_controller_controlled` authorizes planning a
later one-image cycle/reconstruction experiment from a separately bound exact
image-conditional terminal bank. Even a pass leaves reverse sampling,
reconstruction, known-prior, full-data, unsplit-generator, and spatial
Dirichlet--Ferguson claims false.

A failure preserves the sealed overall learnability result. It means the
checkpoint is not supported as a global time-local controller, so a later
patch must retrain or separately select and confirm a time-local controller;
it must not hide the failure with a learned phase mask or exploratory image.
