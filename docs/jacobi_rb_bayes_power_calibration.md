# Exact noisy-Jacobi Bayes-oracle power calibration

## Purpose and claim

This is a controls-only calibration of the existing exact Jacobi/Rao--Blackwell
learnability pipeline. It asks whether the unchanged width-32 predictor and
optimization recipe can recover a known, nonzero conditional mean when it is
trained on the same noisy Rao--Blackwell label used by the one-image
experiment.

The immutable parent run is:

`runs/experiment12_d0_jacobi_rb_one_image_learnability/20260729-015817_production-exact-k512-rb-one-image-learnability`

That run ended `no_detectable_one_image_conditional_signal`. Its exact cache,
teacher, optimization, sealing, and numerical checks passed; its only failed
confirmation subcheck was `aggregate_model_beats_zero`. This is sealed physical
no-signal evidence. The synthetic calibration neither overturns nor
reinterprets it.

A passing calibration establishes only that this finite-sample pipeline can
detect the prescribed bounded synthetic conditional mean. It authorizes
planning a separately named, fresh physical-signal witness. It does not
authorize physical retraining, full-data training, production refinement,
reverse sampling, reconstruction, a known-prior claim, or spatial
Dirichlet--Ferguson convergence.

## Label firewall and exact controls

The new synthetic caches may use the parent's separated input caches only as
pair-mass, time, phase, and metadata templates. Parent
`cache/*_labels_audit.npz` files, selected weights, and audit targets are
forbidden. The synthetic noisy Rao--Blackwell label remains the training
target. The analytic oracle is stored in a separate audit-only cache and is
never a model input, training target, or selection target.

For an edge head fraction \(x\), the bounded teacher has density ratio

\[
q_0(x)=1+\tfrac12(2x-1)=x+\tfrac12,
\]

equivalently the mixture
\(\tfrac12\operatorname{Uniform}(0,1)+\tfrac12\operatorname{Beta}(2,1)\).
After Jacobi exposure \(u\),

\[
q_u(y)=1+\tfrac12e^{-2u}(2y-1),
\qquad
m(y,u)=\frac{y(1-y)e^{-2u}}
{1+\tfrac12e^{-2u}(2y-1)}.
\]

Here \(m(y,u)\) is the exact Bayes conditional mean of the noisy
Rao--Blackwell label. The stationary-null control uses
\(q_0=q_u=1\) and \(m=0\).

Teacher and null each use eight disjoint train paths, eight validation paths,
and eight sealed confirmation paths. Every role uses the same 32 selected
outer steps, seven phases, and 392 edges per phase. The complete workload is
4,214,784 exact certified transitions. Three model seeds
`261201,261202,261203` use the unchanged width-32 predictor, unweighted MSE,
4,000-update training plan, and a target RMS inferred from that law's training
paths only. Validation alone selects checkpoints. The analytic zero predictor
is a legal candidate for both laws.

Confirmation is generated and opened once, only after both laws' selected
checkpoints, metadata baselines, path plan, and gate definition are frozen.

## Gates

The stages and required gates are:

- `preflight`: immutable-parent provenance, label firewall, disjoint path IDs,
  analytic normalization/score/Bayes identities, and stationary-null identity.
- `cache`: four exact train/validation role caches, complete certification,
  role isolation, noisy-target integrity, tower controls, and zero forbidden
  numerical events.
- `train`: all six model tasks complete and finite, separate training-only
  target scales, validation-only selection, a nonzero teacher checkpoint, a
  legal analytic-zero candidate for both laws, and confirmation still absent.
- `controls`: the one-time teacher/null confirmation.

The teacher confirmation requires:

- the oracle beats zero on all eight paths;
- aggregate oracle relative gain is at least `0.01`;
- the selected model beats zero in aggregate;
- the model beats the training-only metadata baseline on all eight paths; and
- the model recovers at least `0.50` of the oracle's aggregate gain over zero.

The null fails closed if it exhibits the same discovery conjunction: its model
beats analytic zero in aggregate and beats its metadata baseline on all eight
paths.

Closed outcomes are:

- `control_provenance_invalid`
- `analytic_bayes_identity_invalid`
- `exact_control_cache_invalid`
- `oracle_panel_underpowered`
- `optimization_pipeline_invalid`
- `null_false_discovery`
- `noisy_bayes_detection_pipeline_calibrated`

## Production commands

Run a fresh preflight:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_bayes_power_calibration `
  --runs-root runs/experiment12_d0_jacobi_rb_bayes_power_calibration `
  --run-name production-noisy-jacobi-bayes-power `
  --device cuda `
  --stage preflight `
  --parent-one-image-run-dir runs/experiment12_d0_jacobi_rb_one_image_learnability/20260729-015817_production-exact-k512-rb-one-image-learnability `
  --require-gate preflight
```

Resolve the generated directory without a placeholder:

```powershell
$bayesRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_bayes_power_calibration" -Directory |
  Where-Object Name -Like "*_production-noisy-jacobi-bayes-power" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName
```

Generate the unsealed train and validation caches:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_bayes_power_calibration `
  --device cuda `
  --stage cache `
  --resume-run-dir $bayesRun `
  --parent-one-image-run-dir runs/experiment12_d0_jacobi_rb_one_image_learnability/20260729-015817_production-exact-k512-rb-one-image-learnability `
  --require-gate cache
```

Only after the cache gate passes, train and freeze the candidates:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_bayes_power_calibration `
  --device cuda `
  --stage train `
  --resume-run-dir $bayesRun `
  --parent-one-image-run-dir runs/experiment12_d0_jacobi_rb_one_image_learnability/20260729-015817_production-exact-k512-rb-one-image-learnability `
  --require-gate train
```

Only after the train gate passes, open the sealed confirmation once:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_bayes_power_calibration `
  --device cuda `
  --stage confirm `
  --resume-run-dir $bayesRun `
  --parent-one-image-run-dir runs/experiment12_d0_jacobi_rb_one_image_learnability/20260729-015817_production-exact-k512-rb-one-image-learnability `
  --require-gate controls
```

Do not run `confirm` again to seek a different result. Resume is for exact
recovery of the frozen workflow, not panel regeneration or threshold tuning.
