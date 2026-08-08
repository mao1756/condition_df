# Exact Jacobi/RB coarse-residual learnability gate

## Purpose

The immutable exact-\(K=512\) coarse-signal witness ended
`exact_physical_coarse_signal_detected`. Its two independent 64-path panels
gave

\[
S=\operatorname{mean}_c(a_cb_c)
  =0.0006484248701021389,
\]

with one-sided 99% lower bounds from both the whole-path bootstrap and the
independent influence calculation strictly above zero. The earlier width-32
one-image learner nevertheless failed to beat the analytic zero predictor.

This workflow tests the smallest additive repair consistent with both facts.
It freezes a coarse conditional-mean baseline from the historical witness and
trains a neural residual against the original, unweighted, exact
Rao--Blackwell label. It does not residualize or otherwise alter the persisted
training target.

## Frozen baseline

Let \(A,B\in\mathbb R^{64\times4\times7\times392}\) be the two witness
panels, and let \(a\) and \(b\) be their path means. The frozen shrinkage rule
is

\[
N=\tfrac12\operatorname{mean}_c(a_c-b_c)^2,\qquad
\lambda=\frac{S}{S+N/2},\qquad
B_c=\lambda\frac{a_c+b_c}{2}.
\]

The authorizing constants are:

- \(N=0.00315904482822984\);
- \(\lambda=0.2910413880506186\);
- \(\operatorname{mean}(B^2)=0.00018871847424106853\);
- float64 C-order baseline SHA-256
  `5d4e73153c36a59e26403439befd4e13b7f4fe096f7cbf9af6b77ac26565a9df`.

All signed cell values are retained. The baseline is never refitted after the
witness.

## Fresh evidence and gates

Training, validation, and confirmation use disjoint fresh path sets of
64, 32, and 64 paths. Each cache uses the certified seven-phase Jacobi chain
at \(K=512\), the same 32 selected outer steps, and the unmodified binary64
Rao--Blackwell label.

The predictor is

\[
g_\theta(W)=B(C(W))+r_\theta(W).
\]

Both residual output paths are initialized exactly to zero, so update zero is
the frozen baseline. Training minimizes

\[
\frac{\operatorname{mean}(g_\theta(W)-\bar Z)^2}
     {\operatorname{mean}_{\rm train}\bar Z^2}
\]

with no weights, clipping, target transformation, or mixed precision.
Synthetic residual-teacher and exact-baseline-null controls run before
physical labels are opened.

A nonzero checkpoint must improve over the baseline on validation overall
and in the high reverse-time quartile. Only then is the 64-path confirmation
namespace opened. For every whole path, confirmation records

\[
\Delta_B=R(0)-R(B),\qquad
\Delta_R=R(B)-R(B+r_\theta).
\]

A one-sided 99% studentized max-\(T\) family must give strictly positive
simultaneous lower bounds for both contrasts. A baseline-only result cannot
authorize a reverse sampler.

## Production sequence

```powershell
$witnessRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_physical_coarse_signal_witness/20260730-135059_production-exact-k512-physical-coarse-signal-jsonfix").Path
$oneImageRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_one_image_learnability/20260729-015817_production-exact-k512-rb-one-image-learnability").Path

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_coarse_residual_learnability `
  --runs-root runs/experiment12_d0_jacobi_rb_coarse_residual_learnability `
  --run-name production-exact-k512-coarse-residual-one-image `
  --device cuda `
  --stage preflight `
  --parent-coarse-witness-run-dir $witnessRun `
  --parent-one-image-run-dir $oneImageRun `
  --require-gate preflight
```

Resolve the printed run directory, then run `cache`, `train`, and, only when
the train decision seals a nonzero nominee, `confirm`:

```powershell
$residualRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_coarse_residual_learnability" -Directory |
  Where-Object Name -Like "*_production-exact-k512-coarse-residual-one-image" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_coarse_residual_learnability `
  --device cuda `
  --stage cache `
  --resume-run-dir $residualRun `
  --parent-coarse-witness-run-dir $witnessRun `
  --parent-one-image-run-dir $oneImageRun `
  --require-gate cache

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_coarse_residual_learnability `
  --device cuda `
  --stage train `
  --resume-run-dir $residualRun `
  --parent-coarse-witness-run-dir $witnessRun `
  --parent-one-image-run-dir $oneImageRun `
  --require-gate train
```

Inspect `coarse_residual_decision.json`. Run confirmation only when its
decision is `ready_for_confirm`:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_coarse_residual_learnability `
  --device cuda `
  --stage confirm `
  --resume-run-dir $residualRun `
  --parent-coarse-witness-run-dir $witnessRun `
  --parent-one-image-run-dir $oneImageRun `
  --require-gate confirm
```

## Claim boundary

`exact_rb_coarse_residual_learnable` establishes only fresh held-out
predictive learnability for one frozen image under the fixed-\(K=512\) split
chain. It authorizes planning a reverse-phase construction and sampler
control. It does not establish reconstruction, an exact finite-time reverse
kernel, a known prior, unsplit Eulerian convergence, spatial
Dirichlet--Ferguson convergence, or multi-image generalization.
