# Exact Jacobi/Rao–Blackwell one-image learnability gate

This workflow asks one deliberately narrow question:

> For the already validated exact, certified, phase-serial `K=512` Jacobi
> split chain, is the exact Rao–Blackwell DDPM-like label conditionally
> learnable from the permitted later-state inputs on one frozen MNIST image?

It does not repair or reinterpret the terminal Haar result
`hierarchical_power_infeasible`. It does not establish convergence of the
split chain to the unsplit Eulerian generator. It contains no reverse sampler.

## Why this remains the DDPM-like target

Fix an outer step and one active matching phase. Let
\(\mathcal F^-\) contain all information immediately before the phase. For an
active oriented edge \(e\), write:

- \(r_e\) for its conserved pair mass;
- \(X_e\) for its earlier head fraction;
- \(u_e\) for its state-dependent Jacobi exposure;
- \(Y_e\) for its exact later head fraction;
- \(H_e=L_e-M_eY_e\) for the ancestral denoising label; and
- \(\bar Z_e=\bar Z(X_e,Y_e,u_e)\) for the exact Rao–Blackwell label.

Conditional on \(\mathcal F^-\), disjoint active edges use a product of exact
Jacobi transition kernels with independent edge-local latent randomness.
Consequently,

\[
\mathbb E[H_e\mid\mathcal F^-,S^+]
=\mathbb E[H_e\mid X_e,Y_e,u_e]
=\bar Z_e.
\]

The pair mass is conserved, so the full post-phase state contains the pair
mass needed to recover the exposure together with the known phase duration.
For the permitted model information

\[
W=(S^+,\text{reverse split-chain coordinate},\text{phase},
\text{color},\text{duration},\text{class label}),
\]

the tower property gives

\[
\mathbb E[H_e\mid W]=\mathbb E[\bar Z_e\mid W].
\]

Thus unweighted MSE regression on \(\bar Z\) has the same population
conditional-mean optimum, under the allowed inputs, as regression on the
ancestral DDPM-like label. Earlier fractions construct the supervised label
but are never model inputs. No `K → ∞` argument is used.

## Frozen design

- Grid: `28 × 28`, exact split chain `K=512`, eight-step restart shards.
- Phase palindrome: `H0/2,H1/2,V0/2,V1,V0/2,H1/2,H0/2`.
- Selected rows: all seven phases at outer steps `15,31,…,511`.
- Whole paths: eight train, eight validation, eight sealed confirmation.
- Root seed: `261191`.
- Image: first label-3 MNIST image, mixed with uniform mass at
  `lambda_mix=0.35`, imported byte-for-byte from the immutable Strang run.
- Fresh path slots: train `0xE0000…0xE0007`, validation
  `0xE1000…0xE1007`, confirmation `0xE2000…0xE2007`.

The originally suggested `0x60000/0x61000/0x62000` slots were rejected by the
required repository scan because an immutable phase-observer plan already
claims that namespace.

Historical parent verification remains byte-specific. A reviewed
compatibility table recognizes only the exact additive capture successor of
the multipath scheduler (including its LF/CRLF forms) when replaying old
source fingerprints. The new learnability manifest instead binds the current
scheduler, provenance helper, and compatibility-table bytes directly; any
unlisted source edit still fails closed.

Model input is limited to:

1. later full state;
2. normalized remaining split-chain phase coordinate;
3. phase;
4. color;
5. duration; and
6. class label.

Path IDs, earlier states, randomness, certificates, later edge fractions, and
targets are held in a separate audit object and cannot enter `forward`.

The fixed predictor has a permitted-input local affine skip plus three
same-padding width-32 convolutional layers. Training uses Adam at `1e-3`,
batch 32, at most 4,000 updates, validation every 100 updates, global
gradient clipping at one, and seeds `261201,261202,261203`. The only target
scaling is one positive RMS computed over all training targets.

## Stages and interpretation

1. `preflight` verifies all immutable parents, the source image, namespace,
   eight-path capture/no-capture hash parity, and ten-hour/128-MiB resource
   ceilings. Its storage projection includes selected capture archives,
   restart states, compact caches, and shard metadata.
2. `cache` generates exact train and validation chains. Confirmation remains
   absent. Every restart shard binds its split, step, path IDs, root seed,
   numerical profile, scientific configuration, input/output chain, and NPZ
   hashes.
3. `train` must first pass the exactly representable synthetic teacher, then
   fits all three physical seeds and freezes the validation-selected
   checkpoint, metadata baseline, path plan, and confirmation rule. The
   physical metadata baseline is frozen before the teacher runs, and each
   task writes an exact optimizer/model/RNG resume checkpoint at every
   validation boundary.
4. `confirm` opens exactly eight new paths once. It passes only when the model
   beats the training-only metadata baseline on every path and beats the zero
   predictor in aggregate. Eight strictly positive path signs have the
   preregistered one-sided sign probability `1/256`. An interrupted cache or
   evaluation resumes from committed evidence; a completed confirmation is
   replay-verified and never evaluated again.

The success decision is
`exact_k512_split_chain_rb_label_learnable`. It authorizes planning a larger
exact-discrete-chain training study only. It does not authorize full-data
training, reconstruction, reverse sampling, spatial Dirichlet–Ferguson
claims, or an unsplit-generator approximation claim.

## Production sequence

```powershell
$root = "runs/experiment12_d0_jacobi_rb_one_image_learnability"
$multi = "runs/experiment12_d0_jacobi_rb_cuda_multipath_confirmation/20260723-092105_production-multipath-jacobi-rb"
$strang = "runs/experiment12_d0_jacobi_rb_strang_refinement/20260723-230629_production-state-dependent-strang-refinement"
$haar = "runs/experiment12_d0_jacobi_rb_haar_power_recovery_confirmation/20260726-085126_production-haar-power-recovery"

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_one_image_learnability `
  --runs-root $root `
  --run-name production-exact-k512-rb-one-image-learnability `
  --device cuda --stage preflight `
  --parent-multipath-run-dir $multi `
  --parent-strang-run-dir $strang `
  --parent-haar-run-dir $haar `
  --require-gate preflight

$run = (
  Get-ChildItem $root -Directory |
  Where-Object Name -Like "*_production-exact-k512-rb-one-image-learnability" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_one_image_learnability `
  --device cuda --stage cache --resume-run-dir $run `
  --parent-multipath-run-dir $multi --parent-strang-run-dir $strang `
  --parent-haar-run-dir $haar --require-gate cache

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_one_image_learnability `
  --device cuda --stage train --resume-run-dir $run `
  --parent-multipath-run-dir $multi --parent-strang-run-dir $strang `
  --parent-haar-run-dir $haar --require-gate train

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_one_image_learnability `
  --device cuda --stage confirm --resume-run-dir $run `
  --parent-multipath-run-dir $multi --parent-strang-run-dir $strang `
  --parent-haar-run-dir $haar --require-gate confirm
```

## Completed result and sealed signal diagnostic

The production run

`runs/experiment12_d0_jacobi_rb_one_image_learnability/20260729-015817_production-exact-k512-rb-one-image-learnability`

ended `no_detectable_one_image_conditional_signal`. The exact confirmation
cache and every numerical/control gate passed. The selected model beat the
training-only metadata baseline on all eight confirmation paths, but its
aggregate MSE (`1.5826757011`) was slightly above the zero predictor
(`1.5825289213`). This is a valid negative result for the frozen selected
model, not an execution failure and not proof that the population conditional
mean is identically zero.

The additive report-only diagnostic decomposes

\[
\operatorname{MSE}(f,Z)-\operatorname{MSE}(0,Z)
=\mathbb E[f^2]-2\mathbb E[fZ].
\]

It replays only the already-selected checkpoint on the existing train,
validation, and confirmation caches. It also reports a cross-split estimate
of conditional-mean energy on the already-frozen
`time-quartile × phase × edge` partition. Both analyses are post-hoc and
non-authorizing: no checkpoint is selected, rescaled, ensembled, or fitted;
no paths are generated; and the parent run remains immutable.

Run a fresh diagnostic preflight:

```powershell
$parent = "runs/experiment12_d0_jacobi_rb_one_image_learnability/20260729-015817_production-exact-k512-rb-one-image-learnability"

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_zero_signal_diagnostic `
  --runs-root runs/experiment12_d0_jacobi_rb_zero_signal_diagnostic `
  --run-name production-sealed-rb-zero-signal-diagnostic `
  --device cuda `
  --stage preflight `
  --parent-learnability-run-dir $parent `
  --require-gate preflight
```

Resolve the generated directory and run the analysis:

```powershell
$signalRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_zero_signal_diagnostic" -Directory |
  Where-Object Name -Like "*_production-sealed-rb-zero-signal-diagnostic" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_zero_signal_diagnostic `
  --device cuda `
  --stage analyze `
  --resume-run-dir $signalRun `
  --parent-learnability-run-dir $parent `
  --require-gate analysis
```

A completed diagnostic authorizes no new training, refinement, reconstruction,
or sampling. Its role is to distinguish prediction-energy cost from target
alignment and to inform a separately preregistered theoretical or
signal-identifiability study.
