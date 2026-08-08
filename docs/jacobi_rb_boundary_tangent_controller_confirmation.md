# Exact Jacobi/Rao--Blackwell boundary-tangent controller confirmation

This additive Experiment 12 workflow repairs the representation used by the
failed affine reverse-controller control. It keeps the certified Jacobi
transition and the raw Rao--Blackwell denoising target unchanged. It trains on
fresh exact evidence and runs only time-local and at-most-eight-phase controls;
it does not run a complete reverse path, sample an image, or claim
reconstruction.

## Why the affine controller failed

The immutable failed run is

`runs/experiment12_d0_jacobi_rb_reverse_controller_control/20260802-040147_production-exact-rb-reverse-controller-control`.

Its preflight failed before the oracle, physical cache, or controller-law
panels opened. The exact reference half-transition was healthy and certified,
but the frozen affine learned subflow left the Jacobi fraction interval. The
first failure occurred at outer step 127, phase occurrence 0, path `0xEA000`,
edge 88:

- the exact reference put the head fraction at
  `0.9995690075155423`;
- the frozen predictor returned
  `m=0.06510134784541541`;
- with `M=2`, `du=0.017288998155204367`, the affine update gave
  `y+2*m*du=1.0018200816811438`;
- the corresponding tail mass was `-6.045029436998786e-06`.

The same structural problem appeared on the first microstep for `M=4` and the
production value `M=8`; an advisory `M=16` replay also crossed a facet. This
is therefore not evidence that the Jacobi kernel or Rao--Blackwell label is
wrong, and merely increasing a fixed microstep count is not a principled
repair. The failed run remains byte-identical and is re-adjudicated as
`frozen_affine_conormal_flow_boundary_invalid`.

The successful learning parent remains

`runs/experiment12_d0_jacobi_rb_coarse_residual_learnability/20260731-140333_production-exact-k512-coarse-residual-one-image`.

## Boundary-tangent representation

For a matched edge, let

\[
  r=s_{\rm tail}+s_{\rm head},\qquad
  y=s_{\rm head}/r,\qquad
  \mu(y)=y(1-y).
\]

For earlier and later head fractions `X` and `Y`, let the ancestral denoising
variable be `H=L-MY`. The exact certified Rao--Blackwell label is

\[
  \bar Z(X,Y,u)
  =\mathbb E[H\mid X,Y,u]
  =Y(1-Y)\,\partial_Y\log k_u(Y\mid X).
\]

The conditional mean seen through the permitted model inputs has conormal form

\[
  m(W)=\mathbb E[\bar Z\mid W]=\mu(y)q(W),
\]

The new width-32 predictor represents the finite coefficient

\[
  q_\theta(W)=q_B(C(W))+q_{\theta,\rm residual}(W)
\]

and returns `m_theta=mu*q_theta`. It still minimizes direct, unweighted MSE
against the raw exact binary64 label:

\[
  \mathcal L(\theta)
  =\frac{\operatorname{mean}(\mu q_\theta-\bar Z)^2}{c^2},
  \qquad c^2=\operatorname{mean}_{\rm train}\bar Z^2.
\]

No quotient label `barZ/mu` is formed or persisted. There is no target
clipping, weighting, floor, limiter, projection, or renormalization. At an
exact facet, `mu=0`, so every finite coefficient produces exactly zero
conormal output.

The frozen baseline is fitted from training paths only, independently in each
of the `4 x 7 x 8 x 392` cells:

\[
  q_B=\frac{\sum \mu\bar Z}{\sum \mu^2}.
\]

Both residual output paths are initialized to zero, so update zero is exactly
the frozen boundary-tangent baseline.

## Boundary-preserving controller flow

At each learned substep, freeze the finite coefficient `q` and integrate

\[
  \frac{dy}{du}=2q\,y(1-y)
\]

exactly. For an interior fraction,

\[
  \operatorname{logit}(y^+)
  =\operatorname{logit}(y)+2q\,\delta u.
\]

The facets `y=0` and `y=1` are fixed exactly. This preserves `[0,1]`, pair
mass, and global simplex mass structurally. Moreover,

\[
  y^+=y+2\mu(y)q\,\delta u+O(\delta u^2)
     =y+2m\,\delta u+O(\delta u^2),
\]

so the new flow has the same infinitesimal learned reverse generator as the
old affine update. Only the finite-step coordinate integration changes.

## Fresh evidence and training recipe

The root seed is `261311`; model seeds are `261312,261313,261314`; the
confirmation and controller bootstrap seeds are `261315` and `261316`.
Synthetic-teacher and exact-baseline-null training use seeds `261317` and
`261318`. A semantic collision scan runs before CUDA work. Fresh path roles
are:

- preflight: `0xEC000--0xEC007` (8 paths);
- train: `0xEC100--0xEC13F` (64 paths);
- validation: `0xEC200--0xEC21F` (32 paths);
- confirmation: `0xED000--0xED03F` (64 paths).

All paths use the exact certified grid-28, alpha-1, `K=512`,
`tau_eff=5e-5` chain and execute all 512 outer steps. The represented datum is
the first label-3 MNIST image mixed with uniform mass at `lambda_mix=0.35`;
its image and mixed-target SHA-256 values are
`0bb39fec59853f789fe366251cd85ed79ffbffb5a1aaa32084d2dbd2bbb4ea7d`
and `00ae86fb69be6d86557f15f6f8fa00f8bb3c2514f331863c9638e36d23d135c5`.
Cache rows are recorded at outer steps `15,31,...,511`, at all seven phase
occurrences, and at the eight exact within-phase fractions
`1/16,3/16,...,15/16`. Each recorded row uses its own exact midpoint-prefix
branch. The frozen row counts are
`114688/57344/114688` for train/validation/confirmation; including the base
rollout, their transition projections are
`134873088/67436544/134873088`. Confirmation paths do not exist until a
checkpoint has been sealed, and their branch inputs and labels are evaluated
streamingly rather than retained as another training cache.

The residual is the unchanged width-32 phase-conditioned local-affine CNN:
three circular `3 x 3` convolutions, a four-color spatial head, and a local
affine edge head. Its complete input firewall is the later full state, reverse
time, phase occurrence, matching color, phase duration, and digit label.
Earlier state, outer-step index, midpoint index, pair mass as a separate
feature, target, and certificate data are not model inputs. The tangent wrapper
derives `mu` from the later state, adds the frozen baseline to the residual
coefficient, and multiplies the result by `mu`.

Training uses batch size 32, Adam at `1e-3`, weight decay zero, 4,000 updates,
validation every 100 updates, gradient clipping at one, deterministic
execution, and no mixed precision. An exact synthetic boundary-tangent teacher
must reach relative validation MSE at most `0.01` and beat zero on every
validation path. An exact-baseline null must select update zero. Physical
labels remain unopened until both controls pass. Update zero then competes
normally; a nonzero candidate is eligible only if it beats the frozen baseline
both overall and in the high-reverse-time quartile. The global selection rule
is lowest validation MSE, then earliest update, then lower model seed.

## Gates

1. `preflight` verifies both immutable parents, the corrected failure
   adjudication, direct-target algebra, exact facets, finite score and stable
   logistic-flow controls, orientation, pair/simplex conservation, model-input
   firewall, namespaces, exact resume, and a production resource projection.
   The projection must cover all fresh roles in at most 30 hours and keep the
   train/validation evidence within 1.25 GiB.
2. `cache` generates fresh train and validation evidence with certificate
   fraction one, mass error at most `2e-12`, no forbidden mechanisms, at least
   1,300 transitions/s, fallback fraction at most `1e-4`, fallback-time
   fraction at most `0.10`, at most 80% device memory, at most 1.25 GiB total
   persisted train/validation evidence, and restartable eight-step shards.
3. `train` fits the training-only `q_B`, runs the synthetic and null controls,
   and selects one nonzero physical checkpoint only from validation evidence.
   A baseline-only result fails closed.
4. `confirm` opens 64 fresh paths once. A one-sided 99.5% studentized
   whole-path max-T family jointly covers 224 combined-vs-zero cells
   (`4 x 7 x 8`) and four pooled-quartile combined-vs-baseline contrasts. All
   228 simultaneous lower bounds must be strictly positive.
5. `control` uses the sealed confirmation paths at anchors
   `127,255,383,511` and runs paired one-phase and at-most-eight-phase
   exact-reference controls at `M=2,4,8`. A 784-component two-sided 99.5%
   whole-path max-T family requires normalized `M=8` weak-law bias at most
   `0.10` and `M=8`-versus-`M=4` discrepancy at most `0.05`, together with
   full certification, conservation, forbidden-event, throughput, memory,
   and persisted-size gates. `M=2` is retained as a numerical trajectory
   diagnostic; the authorizing refinement contrast is `M=8` versus `M=4`.

The statistical gates use 50,000 deterministic whole-path bootstrap
replicates. Paths, not cells or edges, are the resampling unit.

The closed terminal decisions are:

- `control_provenance_invalid`;
- `failed_controller_adjudication_invalid`;
- `boundary_tangent_representation_invalid`;
- `boundary_tangent_design_infeasible`;
- `fresh_exact_cache_invalid`;
- `boundary_tangent_baseline_invalid`;
- `boundary_tangent_optimization_pipeline_invalid`;
- `boundary_tangent_baseline_only_signal`;
- `selection_false_discovery`;
- `boundary_tangent_time_local_signal_not_detected`;
- `boundary_tangent_audit_inconclusive`;
- `paired_risk_inference_invalid`;
- `boundary_tangent_controller_numerically_invalid`;
- `reverse_controller_weak_law_failed`;
- `reverse_controller_microstep_refinement_failed`;
- `exact_rb_boundary_tangent_controller_controlled`.

All nonfinal outcomes fail closed. In particular, a cache, representation,
optimization, inference, numerical, weak-law, or refinement failure does not
authorize a complete reverse sampler.

## Production commands

```powershell
$coarseRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_coarse_residual_learnability/20260731-140333_production-exact-k512-coarse-residual-one-image").Path
$failedControllerRun = (Resolve-Path "runs/experiment12_d0_jacobi_rb_reverse_controller_control/20260802-040147_production-exact-rb-reverse-controller-control").Path

.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_boundary_tangent_controller_confirmation `
  --runs-root runs/experiment12_d0_jacobi_rb_boundary_tangent_controller_confirmation `
  --run-name production-boundary-tangent-rb-controller `
  --device cuda `
  --stage preflight `
  --parent-coarse-residual-run-dir $coarseRun `
  --failed-controller-run-dir $failedControllerRun `
  --require-gate preflight
```

Resolve the new directory without a placeholder:

```powershell
$tangentRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_boundary_tangent_controller_confirmation" -Directory |
  Where-Object Name -Like "*_production-boundary-tangent-rb-controller" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName
```

Run each later stage only after the preceding required gate passes:

```powershell
foreach ($stage in @("cache", "train", "confirm", "control")) {
  .\.venv\Scripts\python.exe `
    -m mnist.diag_d0_jacobi_rb_boundary_tangent_controller_confirmation `
    --device cuda `
    --stage $stage `
    --resume-run-dir $tangentRun `
    --parent-coarse-residual-run-dir $coarseRun `
    --failed-controller-run-dir $failedControllerRun `
    --require-gate $stage
  if ($LASTEXITCODE -ne 0) { break }
}
```

Individual stage commands are preferable when reviewing each gate manually.

## Closed claim

Only `exact_rb_boundary_tangent_controller_controlled` authorizes planning a
separate one-image conditional reconstruction control. A pass establishes
time-local boundary-tangent learnability and at-most-eight-phase controller
controls for one frozen image under the exact certified `K=512` split chain.
It does not authorize or perform reverse sampling, image reconstruction, a
known-prior claim, full-data training, unsplit-generator convergence, or
spatial Dirichlet--Ferguson convergence.
