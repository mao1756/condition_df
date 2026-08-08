# Experiment 12 / D0 patch plan: forward-from-data reverse Doob-residual matching

**Theory-lock revision:** based on `experiment11_advisor_report.tex`, `experiment_c0_weighted_innovation.pdf`, `main.tex`, `eulerian_approx.tex`, `d0_patch_theory.tex`, and the current Experiment 12 implementation. The recommended baseline below is the direct reverse physical Doob-residual mode. Raw forward-innovation regression is retained only as a separately named full-Euler-kernel alternative.

**Current status (implementation audit, 2026-07-17):** the saved P0.8 run `20260613-190443_p08-d0-practical` passed its declared practical Phase-0 gate and marked its bank usable for Phase 1. Its refined `sub64` run still had a mobility/noise-weighted intervention fraction about `0.0397` (raw edge fraction about `0.0752`), so this is not yet the negligible-intervention limit required for a strict h-transform claim. The elementary `r=1` cache, projected/scaled direct target, positive-reference-drift sampler, strict configuration guards, direct oracle replay, and unit/integration contracts are implemented. The production-kernel zero-residual diagnostic now includes a forecast-only limiter preflight, on-device diagnostic accumulation, atomic artifacts/status, source-bound configuration-fingerprinted seed-level resume, seed-selective forward controls, and fail-closed `fixed`, `strict`, and `training-ready` gates. Terminal banks now freeze the mass floor, limiter budget, alpha law, schedule/grid, and integrator identity needed to reproduce the forward law. A deeper saved `4 x 4` run (`20260715-110207_deeper-refinement-hardened`) passes the fixed-grid temporal stationarity/refinement gate at `sub256`. The `28 x 28` preflight (`20260715-123735_production-preflight`) selected the manuscript-aligned full available-mass budget at `sub256`. The three-seed production run `20260715-131054_production-manuscript-budget` then passed `training-ready`: all three direct seeds passed the strict fixed-grid temporal gate, and seed `260715` passed the forward reference control. The one-image run passed its exact-cache gate but failed its elementary optimization gate. Both the first five-stride study and the frozen 128-path/five-seed confirmation completed with passing cache and teacher controls but no robust physical stride; the authoritative confirmation `20260716-202103_production-multiscale-confirmation` ended `no_confirmed_conditional_signal` and closed further unchanged pathwise escalation. The positive-time Dirichlet-form implicit-score gate is now implemented as the next optimization-only experiment, with independent fresh audit paths, exact operator and synthetic controls, training-only linear and Stein comparators, immutable stochastic plans, and fail-closed task resume. No sampler or full-data training is justified unless this score gate returns `implicit_score_signal`.

**2026-07-18 addendum:** the saved implicit-score run failed both synthetic controls before physical training. The boundary-admissible controls-only repair described below is now implemented; no physical-score or sampling run is authorized until it returns `control_pipeline_repaired`.

---

## 1. Diagnosis: one shared failure cause across C0–C3

Every estimator tried so far (C0 weighted innovations, C1 Girsanov-corrected proposals, C2/C2.1 branch posteriors, C3 MC value targets) estimates a conditional law **given that the free reference process reaches a digit-like terminal state**. The `c2_weighted_terminals` figure shows it never does: even the highest-weight free terminals are noise blobs.

This is not an estimator-variance problem that better centering or more branches can fix; it is a rare-event problem. A back-of-envelope check with the C0 parameters (w_sigma = 0.005, K = 256, T inherited from Exp. 10, alpha_bar = 1): from a near-uniform state, the per-edge noise mass transfer per step is sigma_e * sqrt(dt) ~ 5e-4, while a stroke pixel of a simplex-normalized digit holds ~6e-3 mass. Forming one stroke pixel requires ~10 aligned 1-sigma transfers; forming a digit requires this simultaneously over ~10^2 pixels. The probability of the conditioning event under the proposal is effectively e^{-(huge)}, so:

- **C0**: the true conditional innovation mean on free-rollout states *is* essentially zero (learned/noise = 0.007 was the correct answer to an uninformative question).
- **C1**: KL(bridge || free proposal) is enormous, so the path-space Girsanov weights must collapse (observed ESS 1.6e-3). No proposal tweak fixes this while the proposal stays physically unable to reach digits.
- **C2.1**: branch continuations from free states also never reach digits, so the branch posterior only discriminates noise-level fluctuations — a real but useless signal (signal/noise 2.91 about the wrong quantity).
- **C3**: on the free-state distribution, log u is dominated by the smooth coarse-feature distance, which a network fits perfectly (corr ~1.0) while its autograd edge-derivatives are unconstrained by the value MSE — hence high-frequency texture under amplification.
- **ESS calibration masks the problem**: widening epsilon until ESS is in [0.15, 0.40] succeeds precisely by making the reward stop distinguishing digits from noise. Good ESS + no signal is the expected signature of conditioning on an unreachable event.

Two secondary geometric issues that any next experiment must handle:

1. **Mobility freeze on zero background.** theta_e = 0 when s_i = s_j = 0. A simplex-normalized MNIST digit has ~40% of edges frozen at t = 0. Noise cannot enter empty regions and no learned flux can move mass into them. Any scheme that touches data-adjacent states needs the data mixed with a uniform floor, s_0 = (1 - lambda) a + lambda * unif, with lambda part of the data convention (the source sampler already mixes ~15% uniform, so this is consistent).
2. **Drift/clip scale near sparse states.** w_free * (2*alpha_bar + 1) / h^2 * R * dt ~ 0.28 * R per step per edge at w_free = 0.03, K = 256. Near digits, |R| ~ 1 on stroke boundaries, so steps are limiter-dominated exactly where the innovation identity matters. Expect to lower w_free and/or rely on substepping for forward-from-data rollouts; the clip mask must be watched as a first-class diagnostic.

## 2. The pivot: D0, reverse-time learning on forward-from-data rollouts

DDPM works because every training trajectory **starts at data**, so its denoising target is informative without importance weights. At a fixed grid, the corresponding Eulerian construction stays inside the h-transform framework when the free finite-volume process is **nu_h-symmetric**: use `w_free(t) = w_sigma(t)^2` and the same time change in drift and quadratic variation. The Dirichlet–Ferguson refinement claim additionally requires `alpha_h = beta * h^d` with fixed beta; a legacy fixed `alpha_bar` run is a valid fixed-grid practical model, not by itself a continuum-scaled experiment.

**Honest framing for the writeup.** Let p_0 have a density v_0 relative to nu_h and let p_t be the law of the free process started from p_0. By nu_h-symmetry, v_t evolves by the same heat semigroup. The time reversal on [0, T] is then a Doob h-transform with heat potential v_{T - tau}: in reversed time tau the Brownian edge shift is

    eta_rev,e(tau, s) = w_sigma * sqrt(2 theta_e(s)) * d^h_e log v_{T - tau}(s),

the same object class as eq. (23) of the C0 note and the conditioning flux of the Exp. 11 report. Empirical MNIST remains singular relative to nu_h even after a uniform floor. A literal density h-transform therefore needs a specified smoothing of the empirical law or a positive start time `t_min`; uniform flooring alone only removes zero-mobility edges. Without that regularization, report the result as a discrete-data reverse-kernel approximation.

**Training identity (weightless, direct baseline).** Use exact substeps and start with stride `r = 1`. Store the applied forward physical transfer `DeltaK_app,k` and form the realized reverse transfer

    R_k = -DeltaK_app,k.

Let `b_ref,k` denote the drift coefficient assigned by the implemented schedule to the forward interval `[t_k,t_{k+1}]`, spatially evaluated at the later state. Subtract this known positive reverse reference drift and use one run-level scale `c_U > 0`, inferred from the initial cache RMS unless explicitly overridden and then frozen across refreshes:

    U_k = R_k - b_ref,k(S_{k+1}) * dt,
    U_projected,k = Proj_inc(U_k),
    U_scaled,k = U_projected,k / c_U.

On finite projected edge entries, the unweighted L2 minimizer satisfies

    E[U_projected,k | S_{k+1}, y] = Proj_inc(J_rev(tau_{k+1}, S_{k+1}, y)) * dt + o(dt),
    E[U_scaled,k | S_{k+1}, y] = Proj_inc(J_rev(tau_{k+1}, S_{k+1}, y)) * dt / c_U + o(dt).

This target learns only the conditioning residual because the reference reversal is explicit. Project away incidence-null cycle components, or use a potential-gradient head, before calling the learned edge field an h-transform field. There are no terminal weights, ESS, branches, or value Monte Carlo.

**Qualified raw-innovation alternative.** If a sampler instead subtracts a forward-shaped step, raw innovation regression must learn the *full* reverse Euler-kernel mean. With frozen coefficients,

    m_full = -sqrt(dt) * (2 * b_ref / sigma_ref + eta_rev) + finite-step corrections.

Thus `E[xi | later]` is not `-sqrt(dt) * eta_rev` when the reference drift is nonzero. Keep this only under a distinct full-kernel target-space name; do not interpret it as direct Doob-shift regression.

**Where the digit signal now lives.** Near tau ~ T (states close to data), the reverse shift is large and stroke-specific because the trajectory demonstrably came from a digit. This is the exact analogue of DDPM's high-SNR low-noise regime, and it is the regime that every C0–C3 estimator was structurally unable to populate.

## 3. Patch list

### Phase 0 — diagnostics before any training (no model, ~hours)

- **P0.1 Forward-noising preview script** (`mnist/diag_forward_noising.py` or a flag on the cache generator): roll the free reference from normalized MNIST digits (with lambda uniform mix), save an image grid of S_k at k in {0, K/8, K/4, ..., K} for a few digits and seeds.
- **P0.2 Destruction/clip curves**: per time bin, log (i) feature distance D_h(S_k, S_0)^2 and pixel correlation with S_0, (ii) entropy, (iii) clip fraction, (iv) frozen-edge fraction (theta below threshold).
- **P0.3 Schedule calibration sweep** over (w_sigma, K, lambda, w_free; optionally a time-increasing noise schedule w_sigma(t)): choose the smallest noise scale such that by k = K the digit correlation is ~0 and entropy is near the terminal plateau. A declared `< 5%` intervention threshold is only a pragmatic fixed-grid gate; the strict h-transform baseline requires limiter/floor interventions to be negligible and to decrease under substep refinement.
- **P0.4 Terminal bank**: store terminal states of the same calibrated forward discretization. These are controlled samples from the empirical, generally class-conditional p_T^y and remove initial-law mismatch from the first reverse-field experiments. They are not an analytic DDPM prior or automatically samples from nu_h; earn that claim later with mixing and label-leakage tests.
- **Gate**: if no setting destroys digits without clip blowup, fix limiter/substepping first; do not proceed to training.

### Phase 1 — cache generator changes (small diff to the C0 cache path)

- The current `experiment12_d0` cache is deliberately data-initialized: `S_0 = (1 - lambda) a + lambda * unif`, with `a` a training digit carrying label `y`. There is no `--rollout-init` switch in this implementation; the old source-initialized C0 path remains in `experiment11_c0.py`. A shared source-init ablation is deferred until it has an explicit adapter rather than being implied by a nonexistent flag.
- The logical baseline slice is `(state = S_{k+1}, time = t_{k+1} or tau, label = y, realized_reverse_transfer = R_k, physical_residual = U_k, scaled_residual = Proj_inc(U_k) / c_U)`, with exact substeps and `r = 1`. The current NPZ stores `R_k`, interval/schedule metadata, and `c_U`; training derives the residual tensors using the adjacent run configuration. There is no `log_weight`, endpoint A, or source z input. Keep the start digit and raw xi for diagnostics only.
- Store the realized applied physical transfer as well as raw xi. Record per-edge limiter/mobility validity plus aggregate floor-touched counts and floor/renormalization correction magnitudes, but use all finite projected physical transfers in the loss: selecting only no-intervention edges would learn a selected transition law. Require intervention rates to vanish under refinement for the strict claim.

### Phase 2 — loss changes (smaller diff)

- Project target and prediction onto the incidence-relevant edge subspace. Use per-slice MSE with target `Proj_inc(U_k) / c_U` and prediction `u_theta(tau, S_{k+1}, y)`; delete self-normalized sample weighting.
- Set the output L2 coefficient `lambda_m = 0` for the exact conditional-mean baseline. A positive output penalty shrinks the population target and is an ablation. Keep parameter-space AdamW weight decay, gradient clipping, and EMA.
- Sign convention: U_k is already oriented in increasing reverse time, so its Doob-residual mean is positive. A potential-gradient head or incidence-space projection is preferred to suppress cycle components.

### Phase 3 — reverse sampler

Given the later state `s = S_{k+1}`, the direct reverse baseline is

```
u      = project_edges(u_theta(tau, s, y))
dK_rev = b_ref,k(s) * dt + c_U * u + sigma_ref,k(s) * sqrt(dt) * fresh_noise
s      = s + div_h(dK_rev)
```

The known positive reverse reference drift is explicit; the model supplies only the Doob residual. Fresh identity noise supplies the continuous-time reference diffusion, not the exact finite-step conditional covariance. Reusing a limiter implementation does not make the limited forward and reverse kernels Bayes inverses; h-transform claims require intervention rates to vanish under refinement.

- Keep `r = 1` until the elementary contract passes. For `r > 1`, subtract the reference drift at every actual intermediate state and schedule index; uniformly redistributing a block prediction is a coarse-kernel approximation. Promote `r = 4`, then `r = 8`, only with no-degradation tests.
- Keep `--sample-control-strengths` style sweeps from C2.1 for the learned mean term; the honest setting is strength 1.
- Class conditioning via the label embedding only; no z input.

### Phase 4 — acceptance tests and diagnostics

Ordered from cheapest and most decisive:

1. **Zero-residual reference test**: start from nu_h and run the direct reverse reference with `u_theta = 0`. Reference-law observables must be stationary and errors must decrease under dt refinement; this catches the reference-drift sign before training.
2. **Overfit-one-image smoke test**: dataset = a single digit, small model, exact substeps, `r = 1`, `lambda_m = 0`. The reverse sampler from the matching terminal bank must reproduce that digit.
3. **Reconstruction test**: take a held-out forward path, start at its exact terminal state, and check whether a same-class digit re-forms. This isolates the learned field from initial-law mismatch.
4. **Per-time-bin signal**: target RMS of `U_scaled`, residual RMS, and generation-time learned/noise step ratio per bin. Expect a strongly time-dependent profile, large near tau ~ T.
5. Existing image stats (entropy, TV, checkerboard/high-frequency energy), preview grids, and classifier accuracy if the diagnostic classifier is available.
6. Ablations in the C0 Sec. 13 style: reverse drift only (no fresh noise) vs full stochastic reverse. Compare to source-initialized C0 only after an explicit common cache adapter exists.

#### Implemented zero-residual gate

`mnist/diag_d0_zero_residual.py` now calls the same elementary direct-Doob substep as production generation, with the learned transfer set exactly to zero. Refinement levels share one initial Dirichlet bank and coupled Brownian increments; Dirichlet-vs-Dirichlet null thresholds are calibrated once and frozen across levels. The forward reference integrator can be included as a control, but it does not define the reverse pass result. The diagnostic saves `summary.json`, `refinement_metrics.csv`, `states_finest.npz`, and `stationarity_refinement.png`; optional comma-separated seeds produce per-seed artifacts plus `aggregate_summary.json`.

The fixed-grid decision requires at least three temporal levels, a positive reference-rate integral, the finest law to lie inside the calibrated quantile/MMD bands, and entropy and second-moment errors to clear three complementary checks: comparison with an independent Dirichlet bank, comparison with the exact symmetric-Dirichlet expectation, and paired initial-to-terminal drift. Two successive coupled discretization errors must contract, numerical corrections must remain below threshold, and all limiter metrics must be finite and nonincreasing. A separate `strict_h_transform_limit_supported` evidence flag additionally requires the raw and both weighted intervention fractions to fall below their declared thresholds and contract at every tested refinement. It is intentionally possible for the fixed-grid gate to pass while this stricter flag remains false; even a strict-flag pass is finite-resolution temporal evidence, not proof of a continuum or spatial Dirichlet--Ferguson limit.

The first saved implementation run used 1,024 Dirichlet starts, `grid_size=4`, `alpha_eff=1`, `K=8`, `tau_eff=2e-4`, and substeps `{8,16,32,64}`:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_zero_residual --device cpu --run-name finer-smoke-hardened --grid-size 4 --num-paths 1024 --sample-steps 8 --substeps 8,16,32,64 --tau-eff 2e-4 --calibration-reps 8
```

That run's independent-bank checks looked acceptable at `sub64`, but the hardened contract found an analytic entropy error of `6.07` standardized units, paired entropy drift of `8.12`, and paired second-moment drift of `4.48`; it therefore fails stationarity. This is why the independent-bank comparison cannot be the only moment gate.

The deeper temporal refinement run is the current small-grid evidence record:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_zero_residual --device cpu --run-name deeper-refinement-hardened --grid-size 4 --num-paths 1024 --sample-steps 8 --substeps 64,128,256 --tau-eff 2e-4 --calibration-reps 8
```

At `sub256`, quantile distance was `0.0206` against threshold `0.2381` and feature MMD was `0.000735` against threshold `0.00242`. The worst entropy check was the paired drift at `2.77` standardized units; the independent and analytic entropy errors were `0.77` and `1.40`. The independent, analytic, and paired second-moment errors were `0.72`, `0.37`, and `1.68`. Coupled RMS discrepancies contracted `0.00145 -> 0.000909`. Numerical health passed with zero floor correction, renormalization correction about `4.33e-8` per path-substep, and maximum simplex-mass error about `3.58e-7`. Raw intervention fell `0.00825 -> 0.00403 -> 0.00198`, while both weighted fractions fell `0.00139 -> 0.000407 -> 0.000113`. The fixed-grid gate and the finite-resolution strict-refinement evidence flag both passed.

The edge-splitting statement in the manuscript truncates transfer to the mass available at the source endpoint. In the current implementation that is the full directional budget `limiter_fraction=1.0`; the historical default `0.25` is an additional conservative restriction and remains the legacy default. One-step checks show that the full available-mass budget substantially lowers intervention frequency. This is a numerical-policy correction, not evidence by itself, so run the forecast-only production preflight first:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_zero_residual `
  --device cuda `
  --run-name production-preflight `
  --grid-size 28 `
  --sample-steps 512 `
  --substeps 64,128,256 `
  --tau-eff 5e-5 `
  --edge-alpha-mode alpha_eff `
  --alpha-eff 1 `
  --mass-floor 1e-7 `
  --limiter-fraction 1 `
  --preflight-only `
  --preflight-paths 1024 `
  --preflight-reps 4 `
  --preflight-limiter-fractions 0.25,0.5,1 `
  --preflight-max-substeps 4096
```

The preflight evaluates one elementary step at the largest scheduled rate, reusing states and randomness across limiter settings. It saves `preflight_summary.json`, `preflight_metrics.csv`, and `preflight_refinement.png`. These are intervention forecasts only: they cannot set any stationarity or readiness flag.

The saved production preflight `20260715-123735_production-preflight` evaluated four replicates. At `limiter_fraction=1` and `sub256`, the worst direct raw/weighted fractions were `0.001956 / 0.0000401`, and the worst forward raw/weighted fractions were `0.004304 / 0.0000732`; both kernels cleared the fixed `0.005 / 0.0005` forecast thresholds. The direct kernel already cleared them at `sub128`, but the forward raw fraction was `0.00811`, so `sub256` remains the first joint candidate. Fractions `0.25` and `0.5` did not clear both thresholds by `sub256`. This selects the production candidate without constituting stationarity evidence; its run status explicitly records `scientific_gate_evaluated=0`.

The production law gate used the following command. The direct kernel ran all three seeds; the forward cache/reference control ran only the representative seed. `training-ready` requires strict direct evidence on every seed and a passing forward control, and exits nonzero only after all artifacts have been saved:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_zero_residual `
  --device cuda `
  --run-name production-manuscript-budget `
  --grid-size 28 `
  --num-paths 256 `
  --sample-steps 512 `
  --substeps 64,128,256 `
  --tau-eff 5e-5 `
  --edge-alpha-mode alpha_eff `
  --alpha-eff 1 `
  --mass-floor 1e-7 `
  --limiter-fraction 1 `
  --calibration-reps 32 `
  --seeds 260715,260716,260717 `
  --forward-control-seeds 260715 `
  --require-gate training-ready
```

Each run writes `run_config.json`, atomic `run_status.json`, and `aggregate_summary.json`, including for a single seed. Resume an interrupted multi-seed run with the identical scientific options plus `--resume-run-dir <existing-run-dir>`; completed seeds are verified and skipped, while any configuration-fingerprint mismatch fails before computation. On any future `training-ready` failure, retain the evidence and test a separate adaptive or implicit boundary integrator rather than changing the declared thresholds. Do not interpret the saved pass as spatial Dirichlet--Ferguson convergence; fixed-beta grid refinement remains separate.

#### Production one-image direct-Doob gate

The saved production law run `runs/experiment12_d0_zero_residual/20260715-131054_production-manuscript-budget` passed `training-ready`. Its configuration fingerprint is `b850fdd8ed0ab470e24a341e461b7929c159165aafaf06543373b72f42f2693b`. All three direct seeds passed the strict gate at `sub256`; their finest raw limiter fractions were `0.001683`, `0.001683`, and `0.001682`, and their mobility/noise-weighted fractions were about `3.8e-5`. The representative forward control on seed `260715` also passed. This is the required upstream evidence for one-image training, but its claim remains fixed-grid temporal refinement of the reference law.

`mnist.diag_d0_one_image_overfit` is the fail-closed orchestration entry point for the next milestone. It fixes the first label-3 MNIST image, records its hash, mixes it with the uniform measure using `lambda_mix=0.35`, and builds 64 exact substep paths with 16 slices per path. Whole path IDs are deterministically split into 48 training paths and 16 validation paths; slices from one path must never cross the split. The physical target scale is inferred from training paths only and then frozen. The kernel is fixed to `grid=28`, `K=512`, `substeps=256`, elementary stride `r=1`, `tau_eff=5e-5`, `alpha_eff=1`, `mass_floor=1e-7`, and `limiter_fraction=1`. Training uses the direct `doob-physical-residual` target, reference sampler noise, zero auxiliary loss/clip coefficients, seed `260718`, EMA decay `0.999`, 32 base channels, batch size 128, and 10,000 steps.

Run the eight-path cache preflight first. The command intentionally relies on the dedicated CLI's frozen production defaults; `run_manifest.json` is the authoritative expansion of those defaults. Any `--require-gate` run rejects changes to the named production data/cache/training/sampling choices or acceptance thresholds; use `--require-gate none` for exploratory settings:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_one_image_overfit `
  --runs-root runs/experiment12_d0_one_image `
  --run-name production-one-image-direct-doob-preflight `
  --device cuda `
  --stage cache-preflight `
  --zero-residual-run-dir runs/experiment12_d0_zero_residual/20260715-131054_production-manuscript-budget `
  --require-gate cache
```

The cache gate requires exact substep construction and stride 1; a finite positive target scale; target finite fraction 1; direct-oracle replay L1 strictly below positive-reference-only replay L1; and mean absolute terminal-to-mixed-target correlation at most `0.10`. It also requires zero nonfinite edges and floor touches, simplex error at most `2e-6`, floor and renormalization correction L1 per path-substep at most `1e-8` and `1e-6`, raw intervention at most `0.005`, and each weighted intervention at most `0.0005`. This is a cache-contract gate, not evidence that a learned reverse model reconstructs the image. The 64-path production cache separately infers its frozen scale from the 48 training paths only.

Only after that gate passes, run the complete training and paired reconstruction workflow:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_one_image_overfit `
  --runs-root runs/experiment12_d0_one_image `
  --run-name production-one-image-direct-doob `
  --device cuda `
  --stage all `
  --zero-residual-run-dir runs/experiment12_d0_zero_residual/20260715-131054_production-manuscript-budget `
  --validation-paths 16 `
  --validation-every 500 `
  --checkpoint-every 500 `
  --overfit-eval-seeds 260719,260720 `
  --samples-per-seed 8 `
  --sampling-checkpoint-every-outer-steps 8 `
  --require-gate reconstruction
```

Raw and EMA validation run at step 0 and every 500 steps. Checkpoint selection uses only the 16 held-out forward paths: choose the EMA checkpoint with the lowest finite primary residual MSE, breaking ties toward the earliest step. Overall and five fixed `tau/T` bins report target, prediction, and residual RMS; zero-baseline MSE; gain `1 - model_MSE / zero_MSE`; target--prediction covariance; and residual covariance trace. The optimization gate requires positive selected-EMA gain both overall and in the populated data-end bin `tau/T in [0.8,1]`.

Reconstruction evaluates two seeds times eight disjoint validation terminals. Strengths 0 and 1 use the same terminal state and the same standard-normal tensors; mobility remains state-dependent in each branch. The strength-0 branch bypasses model inference because the learned transfer is exactly zero. The reconstruction gate requires, over all 16 paired samples: strength-1 mean correlation to the lambda-mixed target at least `0.90`; mean simplex L1 at most `0.20`; at least 80% of samples with correlation at least `0.85`; mean paired correlation improvement at least `0.20`; relative mean-L1 reduction at least `0.25`; and the cache numerical-health/intervention bounds in both branches. Learned/noise ratios are reported by time bin but are not gated.

The workflow writes atomic `run_manifest.json` and `run_status.json`, cache and split records, exact resumable step checkpoints, raw/EMA validation tables, the selected EMA checkpoint, paired sample arrays/tables/grids, sampling time-bin diagnostics, and `overfit_gate.json`. Exact resume enforces deterministic PyTorch/cuDNN/cuBLAS settings, disables TF32/MKLDNN and reduced-precision-reduction arithmetic changes, and fingerprints the CUDA/NVIDIA driver, backend, device capability, cache identity, checkpoint bytes, RNG state, and source implementation. A required gate exits nonzero only after all available evidence has been written. A cache or optimization failure may skip the later expensive stage, but `run_status.json` must name the skip reason. Resume the production command by replacing `--run-name ...` with `--resume-run-dir <existing-one-image-run-dir>` while keeping all scientific options identical. `--checkpoint-path` is report-only evaluation: legacy or mismatched checkpoints may be inspected with an explicit warning, but cannot satisfy a required gate or exact resume.

The gate target is the lambda-mixed image because that is the represented data law; metrics to the unmixed digit are advisory. A reconstruction pass permits a held-out/multi-image pilot. It proves only one-image reproduction under this frozen fixed-grid temporal kernel, not spatial Dirichlet--Ferguson convergence, an analytic prior, preference conditioning, or full-data sample quality.

##### Saved one-image result: cache pass, elementary learnability failure

The resumed production run `runs/experiment12_d0_one_image/20260715-183915_production-one-image-direct-doob-preflight` completed all 10,000 training steps and failed closed at the optimization gate. This was not a cache or reference-kernel failure. The eight-path preflight passed, and the complete 64-path cache contained 1,024 exact stride-1 slices with frozen physical target scale `1.9628797530721137e-5`. Its raw intervention fraction was `0.00299533`, both weighted fractions were about `7.37355e-5`, and it recorded no floor touches or floor correction.

Checkpoint selection chose the step-500 EMA checkpoint. On all 256 held-out validation slices its primary residual MSE was `1.0104554` against zero-baseline MSE `0.99018345`, for prediction gain `-0.0204729`. In the populated data-end bin `tau/T in [0.8,1]`, containing 147 slices, gain was `-0.0216950`. Target--prediction covariance was only `0.00032458` overall and `0.00021885` in the data-end bin. Later training did not reveal delayed learnability: at step 10,000 the EMA MSE was `2.3812119`, with overall gain `-1.4048189`. The optimization gate therefore failed, paired reconstruction sampling was deliberately skipped, and `run_status.json` records that exact skip reason.

The evidence says that the elementary pathwise label is noise-dominated at this data budget: a trained network did not beat the zero predictor even on the first label-3 image. It does not say that the direct target sign, incidence convention, or reference kernel is wrong; those contracts passed their independent gates. Do not repeat the same stride-1 training with a larger full-data budget, weaken the optimization threshold, or interpret the absent reconstruction metrics as sampled-image evidence. The next question is whether the same physical residual becomes statistically learnable after exact finite-time aggregation.

##### Next gate: multiscale finite-block learnability

The planned entry point is `mnist.diag_d0_multiscale_learnability`. It is a learnability diagnostic only: it builds exact common-trajectory targets at several block lengths, trains independent seed tasks, and audits held-out prediction. It performs no reverse sampling and cannot set a reconstruction flag.

For a block of `r` elementary reference substeps starting at forward index `k`, use the exact physical target from `d0_patch_theory.pdf`, Eq. (6.4), with the incidence projection made explicit:

\[
U_{k}^{(r)}=
\Pi_{\mathrm{inc}}\left[
-\sum_{q=0}^{r-1}\Delta K_{k+q}^{\mathrm{app}}
-\sum_{q=0}^{r-1}b_{k+q}^{\mathrm{ref}}(S_{k+q+1})\,\Delta t
\right].
\]

The second sum must be accumulated at every actual intermediate later state `S_{k+q+1}` with the matching schedule interval. Freezing the drift at `S_{k+r}` and multiplying by `r`, or reconstructing it after the cache has discarded the intermediate states, is not Eq. (6.4). Each stride gets its own independently initialized unchanged `DirectFluxUNet`; no stride embedding or shared multiscale model is used. The model input is the block endpoint `(tau_{k+r}, S_{k+r}, y)`, and its population target is the conditional first moment `E[U_k^(r) | S_{k+r}, y]`.

Project before normalization. For every stride, infer a separate positive scalar

\[
c_{U,r}=\max\left\{\operatorname{RMS}_{\mathrm{train}}(U_k^{(r)}),c_{\min}\right\},
\qquad \widetilde U_k^{(r)}=U_k^{(r)}/c_{U,r},
\]

using the 40 training paths only, then freeze and fingerprint it for selection, audit, and any later sampler. Do not infer scales from the selection or audit paths, normalize per slice, whiten by state-dependent mobility, or select the loss on a random no-intervention mask. The primary loss remains unweighted finite-edge MSE with zero auxiliary output penalty and zero clipping.

The production diagnostic uses common exact trajectories and strides `1,16,64,256,1024`. It builds 64 paths with 32 distinct endpoint anchors per path. The fixed five-bin anchor allocation is `4,4,4,4,16`, so half of the anchors lie in the data-end bin. Whole path IDs are split once into 40 training, 12 selection, and 12 audit paths. Seeds are frozen as follows: cache `260721`, split `260722`, training tasks `260723,260724,260725`, whole-path bootstrap `260726`, and teacher audit `260727`. Each learned task uses 32 base channels, batch size 128, 3,000 steps, validation and checkpoint intervals of 250 steps, and EMA decay `0.999`.

The teacher audit is a pipeline-capacity control, not evidence for the physical target. It must achieve prediction gain at least `0.90` both overall and in the data-end bin before any physical-stride result can pass. A physical stride passes only when all of the following fixed conditions hold:

1. All three training-seed tasks complete, all required metrics are finite, and every task selects a nonzero checkpoint.
2. At least two of the three seeds have positive prediction gain over the zero baseline on both selection and audit data, both overall and in the data-end bin.
3. Across the three seeds, median audit overall gain, median audit data-end gain, and median audit target--prediction covariance are all strictly positive.
4. Deterministic one-sided 90% whole-audit-path bootstrap lower bounds are strictly positive for overall and data-end gain. Use all 12 audit paths, 10,000 bootstrap replicates, and seed `260726`; never resample individual slices as if they were independent paths.
5. Median overall audit gain over the frozen training-only `tau`-bin mean baseline is strictly positive.
6. Audit coverage is complete: all 12 audit paths and all 192 expected data-end slices are present.

The required-gate choices are `none`, `cache`, `teacher`, `any-scale`, and `elementary`. `any-scale` requires the cache gate, teacher gate, and at least one passing physical stride. `elementary` requires the cache gate, teacher gate, and a passing stride `r=1`. Always report every stride and select the smallest passing stride; do not select the largest gain after looking at the audit set. Artifacts must be written before a required-gate failure returns nonzero.

`learnability_decision.json` uses a closed set of scientific outcomes. `elementary_signal` launches a fresh strict `r=1` reconstruction run. `coarse_only_signal` authorizes planning a separately named coarse sampler plus conditional-noise calibration, but this workflow itself still performs no sampling. `path_memorization_only` requires training gain at least `0.50` without held-out audit signal and calls for more independent paths or variance reduction. Positive audit point estimates that miss robustness checks are `inconclusive` and trigger the unchanged 128-path/five-seed confirmation profile. With a passing teacher and finite nonpositive audit gains at every scale, `no_detectable_conditional_signal` sends the cumulative/score target back to theory. Cache and teacher failures are reported as `cache_invalid` and `optimization_pipeline_invalid` and must be repaired before further scientific interpretation.

Run the four-path preflight first. It uses one anchor in each `tau` bin per path and checks exact Eq. (6.4) accumulation, common-trajectory alignment, split/scale isolation, finiteness, and cache numerical health without making a learned-model claim:

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

Only after that cache gate passes, run the full learnability audit:

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

##### Saved multiscale pilot result: valid pipeline, inconclusive physical signal

The production pilot `runs/experiment12_d0_multiscale_learnability/20260716-090351_production-multiscale-learnability` completed all 15 physical tasks and ended with `status=complete`, `outcome=gate_failed`, and decision `inconclusive`. This is a scientific gate failure rather than a crashed or invalid run. The cache passed every arithmetic and numerical-health check: exact `r=1` target identities held, all target scales were finite and positive, the raw intervention fraction was `0.00299657`, the two weighted fractions were about `7.37868e-5`, and there were no nonfinite edges or floor touches. The deterministic teacher also passed strongly, with overall gain `0.984170` and data-end gain `0.982537`, so the model and optimization pipeline could learn a known state-dependent target.

No physical stride passed. Strides `1`, `16`, `64`, and `256` had negative held-out gains. At `r=1024`, two seeds produced a small positive data-end gain and the median data-end gain was `0.00487250`; its one-sided whole-path bootstrap lower bound was also positive at `0.000871361`. That signal did not survive the predeclared complete-stride gate: median audit gain was `-0.00582885` overall, the overall bootstrap lower bound was `-0.00758004`, and median gain relative to the frozen training-only time-bin predictor was `-0.0150556`. Thus the result is at most a localized endpoint hint. It does not establish state-dependent conditional-mean learning, authorize a coarse sampler, or justify weakening any threshold.

##### Independent 128-path/five-seed confirmation

The `confirmation` study profile is a single predeclared replication, not an adaptive continuation of the inspected pilot. It binds the completed pilot above through `--parent-multiscale-run-dir`, verifies that its cache and teacher passed and that its decision was `inconclusive`, and records hashes of its manifest, status, decision, and scientific configuration. The profile preserves the source image, kernel, target, model, optimizer, EMA, training schedule, five strides, anchor plan, and all gate thresholds. It changes only the independent evidence budget:

- 128 newly generated paths, sharded eight paths at a time, with no pilot cache reuse;
- whole-path split `80/24/24` for training, checkpoint selection, and untouched audit;
- five training seeds `260730,260731,260732,260733,260734`, requiring at least three passing seeds per stride;
- fresh cache, split, bootstrap, and teacher seeds `260728`, `260729`, `260735`, and `260736`, while dataset seed `260718` preserves the same label-3 source image; and
- complete audit coverage of 24 paths and 384 prescribed data-end slices per seed, with 10,000 bootstrap replicates over whole audit paths.

Pilot artifacts cannot be reused as confirmation cache shards or training evidence. The confirmation profile may resume only itself, and required gates reject profile overrides. An altered configuration remains available only as an exploratory `--require-gate none` run and cannot issue the authoritative confirmation decision.

Run the confirmation preflight first:

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

After it passes, copy the newly printed run directory into `$confirmationRun` and resume that exact run. This retains the validated preflight and advances through all 16 cache shards, the teacher, and the 25 physical training tasks:

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

The confirmation is terminal for automatic sample-size escalation. A passing `r=1` selects `elementary_signal` and authorizes planning a fresh strict elementary reconstruction run. A first pass only at `r>1` selects `coarse_only_signal` and authorizes planning a separately validated coarse sampler and conditional-noise calibration. A train-only gain of at least `0.50` without audit signal remains `path_memorization_only`. If the cache and teacher remain valid but no stride passes, including another positive-but-nonrobust endpoint hint, the new terminal state is `no_confirmed_conditional_signal`; it records `confirmation_exhausted=1`, `repeat_same_profile_authorized=0`, and `sampling_authorized=0`, and sends the cumulative/score target or a separately justified variance-reduction method back to theory. Do not launch a third larger unchanged rerun.

The command is restartable at both cache-shard and training-task boundaries. Resume an interrupted production run with the same scientific options plus `--resume-run-dir <run-dir>`; completed shard hashes and task checkpoint hashes are verified before they are skipped. To train in a new run from an already completed compatible cache, add `--cache-run-dir <cache-producing-run-dir>`. Cache reuse is rejected if the frozen cache semantics, source implementation, runtime fingerprint, anchor plan, or shard hashes differ. `--stage report --resume-run-dir <run-dir>` recomputes tables and gates from verified completed task results without running a model or sampler.

The atomic run artifacts include `run_manifest.json`, `run_status.json`, `parent_provenance.json`, `anchor_plan.json`, `path_split.json`, `cache_preflight.json`, the sharded cache index, `target_scales.json`, per-task resumable checkpoints/status, `teacher_control.json`, checkpoint/split/time-bin/per-path/stride-seed CSVs, `gain_vs_stride.png`, `learning_curves.png`, and `learnability_decision.json`. A completed confirmation also writes `pilot_confirmation_stride_comparison.csv` and `pilot_confirmation_gain.png`; these display the two studies side by side with `evidence_pooled=0` and are never used by the gate. Every artifact records or inherits `sampling_performed: 0`; no sample arrays or generated-image grids are produced by this gate.

At fixed nonzero `r`, a passing model estimates an integrated coarse reverse-kernel first moment. It is not an exact finite-time reverse transition, because a conditional mean plus fresh reference Gaussian noise does not determine the state-dependent finite-step conditional covariance or higher moments. Uniformly redistributing one predicted block total over `r` reverse substeps is an additional coarse integrator and is not authorized by this learnability gate. A later sampler patch must be separately named, replay-tested, and compared as `r` decreases.

Even an `elementary` pass would remain limited to the lambda-mixed empirical data law, the empirical class-conditional terminal bank, and the frozen 28 by 28 temporal kernel. The present `alpha_eff=1` configuration is not the manuscript spatial scaling `alpha_h=beta h^d`; the empirical data law is singular relative to the grid Dirichlet law at time zero; and the finite limiter tolerance is not an exact Bayes reversal. Therefore this gate cannot establish spatial Dirichlet--Ferguson convergence, a known DDPM prior, a literal time-zero density h-transform, preference conditioning, held-out digit generalization, or full-data sample quality.

##### Saved confirmation result: no confirmed pathwise conditional-mean signal

The authoritative confirmation `runs/experiment12_d0_multiscale_learnability/20260716-202103_production-multiscale-confirmation` completed all 25 physical tasks with a passing cache and teacher, but failed the predeclared `any-scale` gate. Its terminal decision is `no_confirmed_conditional_signal`; `confirmation_exhausted=1`, `repeat_same_profile_authorized=0`, and `sampling_authorized=0`. This closes the automatic pathwise block-residual escalation. It is not permission to add more strides, paths, or seeds to the same target.

The negative conclusion is specific. Strides `1`, `16`, `64`, and `256` had negative selection and audit gains. At `r=1024`, all five seeds had positive audit point estimates overall and at the data end: overall gains ranged from `0.0001483` to `0.0045574`, while data-end gains ranged from `0.0111557` to `0.0164590`. However, every seed remained worse than the frozen training-only time-bin predictor, with gains from `-0.0125083` to `-0.0080434`. Thus temporal aggregation exposed schedule-dependent structure but did not establish a reproducible state-dependent conditional mean beyond time. The passing cache and teacher rule out a broken rollout, target arithmetic, or generic optimizer-capacity failure.

##### Next gate: positive-time Dirichlet-form implicit score

The next entry point is `mnist.diag_d0_dirichlet_score_learnability`. It changes the statistical estimator, not the frozen forward process. Instead of regressing a realized reverse transfer, it estimates the positive-time relative score `f_tau(s,y)=log(dp_tau^y/dnu_h)(s)` by the symmetric Dirichlet-form identity

\[
\mathcal J_\tau(f)=\mathbb E_{p_\tau^y}\left[\Gamma_h(f,f)+2L_hf\right].
\]

Up to an additive constant and a positive normalization, this is the squared Dirichlet-form distance from the true relative score. It needs only noised states, not a pathwise transfer label. The implementation uses a detached-coefficient Hutchinson Hessian-vector estimator, a twice-differentiable scalar potential U-Net, and the physical conversion

\[
J_e^{\mathrm{Doob}}(s,\tau)
=2n^2 w(\tau)\,\theta_e(s)\,\partial_e f_\tau(s,y).
\]

The literal claim starts only at positive forward time `t_min=T/128`, corresponding to minimum forward substep 1,024 of 131,072. It makes no time-zero density claim. The state-only cache reuses the authoritative confirmation's 80 training and 24 checkpoint-selection paths, but never its previously inspected 24 audit paths. It generates 24 new audit paths and four separate preflight paths with seeds `260751` and `260752`. Every path has 32 anchors in the frozen `4,4,4,4,16` reverse-time strata.

The comparator is a training-only cubic B-spline time-dependent state-linear potential. Each of the three neural tasks learns only a nonlinear residual above that frozen baseline, with seeds `260753,260754,260755`, batch 32, 5,000 steps, EMA `0.999`, and FP32 second-order autograd without AMP or TF32. Checkpoint selection uses only the 24 parent selection paths and a fixed 16-probe bank. Final audit uses two independent 32-probe banks on the 24 fresh paths, four-group whole-path median-of-means, and 10,000 one-sided 90% whole-path bootstrap replicates. Each of the two probe-free Stein banks contains 32 smooth periodic linear plus 32 quadratic witnesses; their scales are frozen from training states only, the quadratic generator includes the exact Hessian term, and discrepancies are squared within each path/time bin before equal aggregation. A signal must also have stable nonlinear physical-flux direction across training seeds; the median pathwise cosine must be at least `0.50` and its bootstrap lower bound at least `0.25`.

Before physical optimization, a 4 by 4 exact operator preflight checks constant and simplex-mass gauges, the generator product rule, and Hutchinson trace accuracy. The positive teacher and stationary null use 64 independently sampled path clusters split `40/12/12`; only their time plan is inherited, while physical states and path IDs are not reused. A nonuniform-Dirichlet positive teacher must achieve score gain at least `0.90`, overall flux cosine at least `0.98`, every time-bin cosine at least `0.95`, overall relative flux L2 at most `0.15`, and every bin at most `0.20`. The stationary-Dirichlet null must not produce a positive audit lower bound versus its frozen training-only linear comparator. Failure of either synthetic control is `optimization_pipeline_invalid` and skips physical training under a required gate.

Run the production preflight first:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_dirichlet_score_learnability `
  --runs-root runs/experiment12_d0_dirichlet_score `
  --run-name production-one-image-dirichlet-score `
  --device cuda `
  --stage preflight `
  --zero-residual-run-dir runs/experiment12_d0_zero_residual/20260715-131054_production-manuscript-budget `
  --parent-one-image-run-dir runs/experiment12_d0_one_image/20260715-183915_production-one-image-direct-doob-preflight `
  --parent-multiscale-run-dir runs/experiment12_d0_multiscale_learnability/20260716-202103_production-multiscale-confirmation `
  --require-gate preflight
```

Copy the printed directory into `$scoreRun`, then resume the exact run through cache, controls, training, and report:

```powershell
$scoreRun = "runs/experiment12_d0_dirichlet_score/<timestamp>_production-one-image-dirichlet-score"
.\.venv\Scripts\python.exe -m mnist.diag_d0_dirichlet_score_learnability `
  --device cuda `
  --stage all `
  --resume-run-dir $scoreRun `
  --zero-residual-run-dir runs/experiment12_d0_zero_residual/20260715-131054_production-manuscript-budget `
  --parent-one-image-run-dir runs/experiment12_d0_one_image/20260715-183915_production-one-image-direct-doob-preflight `
  --parent-multiscale-run-dir runs/experiment12_d0_multiscale_learnability/20260716-202103_production-multiscale-confirmation `
  --require-gate score
```

The closed decisions are `implicit_score_signal`, `linear_spatiotemporal_only`, `objective_only_signal`, `trace_estimator_inconclusive`, `boundary_or_outlier_artifact`, `path_memorization_only`, `no_detectable_implicit_score`, `operator_invalid`, `cache_invalid`, and `optimization_pipeline_invalid`. Only `implicit_score_signal` authorizes planning a separately named positive-time score-to-flux one-image reconstruction patch. This workflow never imports a sampler, writes `sampling_performed: 0` throughout, and produces no generated image.

Its artifacts include exact run and parent provenance, immutable probe/control/witness plans, the operator/device preflight, deterministic state-only cache shards and hashes, explicit exclusion of the old audit paths, frozen linear baselines, exact RNG-resumable raw/EMA checkpoints with an atomic best pointer, independent audit/Stein/path tables, cross-seed nonlinear-flux cosines, learning/time-bin plots, `implicit_score_gate.json`, and `score_learnability_decision.json`. Required-gate failure is returned only after all available evidence is committed.

##### Saved implicit-score result and boundary-domain repair gate

The completed run `runs/experiment12_d0_dirichlet_score/20260717-162622_production-one-image-dirichlet-score` stopped at the synthetic controls with `decision=optimization_pipeline_invalid`; it did not train or audit a physical score and did not sample. Its operator and state-cache gates passed. The positive teacher recovered substantial but insufficient signal (`0.8200` overall score gain, `0.8887` flux cosine, and `0.4590` relative flux L2), while the exact stationary `Dirichlet(1)` null produced a false audit improvement of `+263.66` with one-sided 90% lower bound `+210.54`.

The null failure is a model-domain failure, not physical evidence. The old potential U-Net consumed raw `log(Ns)`, allowing `1/s` state gradients. Such candidates need not have vanishing conormal flux at a simplex face, so they lie outside the integration-by-parts domain used by the implicit-score objective. The saved null model exhibits precisely this singularity. Consequently the old nonuniform-Dirichlet teacher is retained only as an advisory singular stress test, and its failed checkpoints cannot satisfy the repaired control gate.

The controls-only repair entry point is `mnist.diag_d0_score_boundary_controls`. Its versioned U-Net uses the closed-simplex-smooth channels `Ns` and `log1p(Ns)`. The exact bounded teacher has density ratio

\[
\frac{p_\tau(s)}{\nu(s)}=\frac12+\frac12N\sum_{j=1}^4w_j(\tau)s_{i_j},
\]

with the fixed quarter-grid anchors and polynomial weights from the patch theory. Teacher states are sampled exactly as the corresponding base/one-count Dirichlet mixture. A deterministic facet-ray gate checks finite potential, gradient, Hessian-vector products, generator, and energy, conormal decay slope at least `0.9`, and four-decade decay at most `1e-3`. The singular `sum_i log(s_i)` family is an explicit negative fixture whose null coefficient is checked against `-12(N-1)`.

The preflight exercises both an analytic `log1p` witness and the actual versioned potential U-Net with deterministic nonzero weights. On CUDA it also runs the frozen production workload shape: batch 64 with four training probes and a backward pass, followed by a 64-probe audit evaluation. The failed parent, its cache time plan, and every required control artifact are checked against the parent's own manifest-bound artifact registry before reuse.

Synthetic teacher and null clusters are independent and split by whole path as `128/32/32`, with 32 anchors per path in the fixed `4,4,4,4,16` time strata. One supervised analytic-score task must pass first. Three implicit teacher tasks and three stationary-null tasks then use four randomized orthogonal-Hadamard training probes, two independent 16-probe selection banks, and two independent 64-probe audit banks. Step zero is selectable; every nonzero checkpoint must have positive whole-path 90% lower bounds in both selection banks, overall and at the data end. The null's primary comparator is the analytic zero potential. The fitted linear comparator is advisory only.

Run the production boundary/operator preflight first:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_score_boundary_controls `
  --runs-root runs/experiment12_d0_score_boundary_controls `
  --run-name production-boundary-admissible-controls `
  --device cuda `
  --stage preflight `
  --failed-score-run-dir runs/experiment12_d0_dirichlet_score/20260717-162622_production-one-image-dirichlet-score `
  --require-gate preflight
```

Copy the printed directory into `$controlRun`, then resume the exact run:

```powershell
$controlRun = "runs/experiment12_d0_score_boundary_controls/<timestamp>_production-boundary-admissible-controls"

.\.venv\Scripts\python.exe -m mnist.diag_d0_score_boundary_controls `
  --device cuda `
  --stage all `
  --resume-run-dir $controlRun `
  --failed-score-run-dir runs/experiment12_d0_dirichlet_score/20260717-162622_production-one-image-dirichlet-score `
  --require-gate controls
```

The terminal outcomes are `control_pipeline_repaired`, `boundary_domain_invalid`, `representation_invalid`, `trace_estimator_inconclusive`, `implicit_objective_unstable`, and `control_provenance_invalid`. Only `control_pipeline_repaired` authorizes planning a separate physical-score experiment with fresh audit paths. This repair workflow performs no physical training and no sampling, so even a pass is only evidence that the bounded synthetic implicit-score controls are trustworthy. It is not a reconstruction result, a known-prior result, or spatial Dirichlet--Ferguson convergence.

The run is fail-closed and exactly resumable. Task fingerprints bind training, selection, and audit arrays; selected-EMA checkpoints carry the optimizer and all RNG/probe states; report mode verifies the prior terminal registry and reconstructs expected fingerprints from the frozen manifest. Artifacts include per-seed and five-time-bin metrics, per-path selection and audit risks, component/gradient histories and quantiles, boundary-ray tables and plots, exact task checkpoints/status, and a terminal registry bound by `run_status.json`.

##### Saved boundary-control result: representation passed, optimizer scaling failed

The completed run `runs/experiment12_d0_score_boundary_controls/20260718-114633_production-boundary-admissible-controls` ended `status=complete`, `outcome=gate_failed`, with the recorded decision `representation_invalid`. Its boundary and operator preflight passed: the model facet-ray slope was `0.99915`, its four-decade endpoint ratio was `1.00936e-4`, all required derivatives and energies were finite, and the legacy logarithmic barrier was correctly rejected.

The bounded supervised teacher in fact passed the scientific representation checks very strongly. Its selected EMA checkpoint at step 3,750 achieved audit score gain `0.998493`, flux cosine `0.999250`, and relative flux L2 `0.038755`, including the required data-end and per-time-bin checks. The only failed subcheck was optimizer health: the post-warmup clipping fraction was `0.989714`, above the unchanged `0.10` limit. The supervised loss had multiplier `1`, whereas loss-scale calibration was scheduled only after this gate and therefore could not repair its optimizer units.

All three implicit-teacher tasks and all three stationary-null tasks were skipped. No implicit objective, trace-estimator, or repaired-null evidence was produced, and empty probe-bank results must not be interpreted as agreement. The run performed no physical-score training and no sampling. It therefore does not justify weakening any analytic threshold or switching to density-ratio classification; it identifies an optimizer-scale and gate-ordering defect.

##### Next gate: boundary-control optimizer-scale repair

The new controls-only entry point is `mnist.diag_d0_score_control_scale_repair`. It binds the immutable failed boundary-control run and verifies that preflight and every supervised analytic metric passed, clipping alone failed, downstream controls were skipped, and neither physical training nor sampling occurred. Only model, operator, schedule, and provenance definitions are reused. Synthetic training, checkpoint-selection, and untouched audit states are regenerated with fresh defaults beginning at seed `260781`; no old synthetic state or task checkpoint is reused.

Before any task training, the workflow calibrates separate supervised and implicit positive loss multipliers on fixed 256-state training-only batches:

\[
c=\min\left(1,\frac{0.10}{\lVert\nabla_\theta L(\theta_0)\rVert_2}\right).
\]

The target initial gradient norm is frozen at `0.10`. The stationary null uses exactly the implicit-teacher multiplier. Scaling is applied before backward and clipping, while all reported analytic losses and score/flux metrics remain in their original units. Raw and scaled loss and gradient diagnostics are recorded, and the gate continues to require a scaled post-warmup clipping fraction at most `0.10` with gradient clipping fixed at `1`.

All scientific settings and thresholds remain unchanged: whole-path split `128/32/32`, 32 anchors per path in strata `4,4,4,4,16`, three paired implicit/null seeds, batch 64, 4,000 steps, AdamW learning rate and weight decay `1e-4`, EMA `0.99`, validation every 250 steps, dual selection/audit probe banks, and the existing score-gain, flux-cosine, relative-L2, bootstrap, and analytic-zero-null requirements. The supervised control must be finite, analytically passing, and optimizer-healthy before implicit and null tasks run. Skipped or incomplete probe banks are explicitly `not_evaluated`; completed banks are `agree` or `disagree`.

Run the optimizer-scale repair preflight first:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_score_control_scale_repair `
  --runs-root runs/experiment12_d0_score_control_scale_repair `
  --run-name production-boundary-control-scale-repair `
  --device cuda `
  --stage preflight `
  --parent-boundary-control-run-dir runs/experiment12_d0_score_boundary_controls/20260718-114633_production-boundary-admissible-controls `
  --require-gate preflight
```

Copy the printed directory into `$repairRun`, then resume that exact run through the fresh controls and report:

```powershell
$repairRun = "runs/experiment12_d0_score_control_scale_repair/<timestamp>_production-boundary-control-scale-repair"

.\.venv\Scripts\python.exe -m mnist.diag_d0_score_control_scale_repair `
  --device cuda `
  --stage all `
  --resume-run-dir $repairRun `
  --parent-boundary-control-run-dir runs/experiment12_d0_score_boundary_controls/20260718-114633_production-boundary-admissible-controls `
  --require-gate controls
```

The closed decisions are `control_pipeline_repaired`, `optimizer_scale_invalid`, `representation_invalid`, `trace_estimator_inconclusive`, `implicit_objective_unstable`, plus the existing provenance and boundary-domain failures. `optimizer_scale_invalid` covers invalid calibration, nonfinite training, or clipping above `0.10`; `representation_invalid` is reserved for optimizer-healthy supervised training that fails the frozen analytic metrics. Only `control_pipeline_repaired` authorizes planning a fresh physical implicit-score experiment. `implicit_objective_unstable` authorizes planning the separately declared density-ratio-classification controls. This repair performs no physical training or sampling, and every artifact records `sampling_performed: 0`.

##### Saved scale-repair result: representation repaired, implicit optimization still unstable

The completed run `runs/experiment12_d0_score_control_scale_repair/20260718-124405_production-boundary-control-scale-repair` ended `status=complete`, `outcome=gate_failed`, with decision `optimizer_scale_invalid`. Its terminal registry contains 231 hash-verified artifacts and records no task failure, physical training, or sampling.

The supervised calibration and task passed decisively. Scaling the initial gradient from `6.81558` to `0.10` produced zero post-warmup clipping; the selected step-4,000 EMA achieved audit score gain `0.998753`, flux cosine `0.999341`, and relative flux L2 `0.036321`. This validates the boundary-smooth representation and supervised optimizer.

The shared implicit calibration also mapped its fixed initial gradient from `37.53796` to `0.10`, but it did not control the subsequent trajectory. All three implicit-teacher and all three null tasks clipped on every one of 3,500 post-warmup steps. Median scaled pre-clip norms were roughly `542–614`, and the negative trace contribution grew faster than the positive energy term. Both independent selection banks rejected every nonzero checkpoint, so every task selected analytic step zero. The null false positive from the earlier singular model did not recur, but the failed optimizer-health prerequisite prevents interpreting this as a clean population-objective failure. Physical-score training remains unauthorized.

##### Next gate: streamed implicit-control stability confirmation

The additive entry point `mnist.diag_d0_score_control_stability_confirmation` binds the immutable scale-repair run and changes only the optimizer experiment. It retains the boundary-smooth model, Dirichlet operator, implicit objective, parent loss multiplier, gradient clip, and all scientific thresholds. Training states are sampled freshly from the exact bounded-teacher or stationary-null law at every optimizer step, with exact per-batch time counts `8,8,8,8,32`. Two stateless four-probe orthogonal-Hadamard banks are averaged for each update.

The workflow first checks analytic Stein identities and records an advisory parent-checkpoint train-versus-fresh forensic replay. It then runs a train/selection-only 1,000-step pilot over learning rates `1e-4,3e-5,1e-5,3e-6`. Only an optimizer-healthy profile with a nonzero dual-bank teacher signal and an analytic-zero null may be frozen. Confirmation uses fresh selection and untouched audit panels plus three paired teacher/null seeds for 4,000 streamed steps. No pilot state, confirmation audit state, probe stream, or checkpoint is reused across roles.

Run the preflight:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_score_control_stability_confirmation `
  --runs-root runs/experiment12_d0_score_control_stability_confirmation `
  --run-name production-streamed-implicit-controls `
  --device cuda `
  --stage preflight `
  --parent-scale-repair-run-dir runs/experiment12_d0_score_control_scale_repair/20260718-124405_production-boundary-control-scale-repair `
  --require-gate preflight
```

Resume the printed directory through the pilot:

```powershell
$stabilityRun = "runs/experiment12_d0_score_control_stability_confirmation/<timestamp>_production-streamed-implicit-controls"

.\.venv\Scripts\python.exe -m mnist.diag_d0_score_control_stability_confirmation `
  --device cuda `
  --stage pilot `
  --resume-run-dir $stabilityRun `
  --parent-scale-repair-run-dir runs/experiment12_d0_score_control_scale_repair/20260718-124405_production-boundary-control-scale-repair `
  --require-gate pilot
```

Only after the pilot passes, run the fresh confirmation:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_score_control_stability_confirmation `
  --device cuda `
  --stage confirm `
  --resume-run-dir $stabilityRun `
  --parent-scale-repair-run-dir runs/experiment12_d0_score_control_scale_repair/20260718-124405_production-boundary-control-scale-repair `
  --require-gate controls
```

The closed outcomes are `control_pipeline_repaired`, `optimizer_stability_unresolved`, `optimizer_stability_invalid`, `trace_estimator_inconclusive`, `implicit_objective_unstable`, `operator_identity_invalid`, `stability_preflight_invalid`, and `control_provenance_invalid`. Only an optimizer-healthy streamed confirmation with passing teacher and null controls authorizes a fresh physical-score patch. Only an optimizer-healthy confirmation failure with agreeing audit banks authorizes density-ratio-classification controls. This workflow performs no physical training or sampling.

##### Saved streamed-control result: stable optimization, unstable implicit objective

The completed controls-only run `runs/experiment12_d0_score_control_stability_confirmation/20260718-232902_production-streamed-implicit-controls` ended `status=complete`, `outcome=gate_failed`, with decision `implicit_objective_unstable`. Its exact 381-record terminal registry verifies, all 14 pilot/confirmation tasks completed, the selected pilot profile was AdamW learning rate `1e-5`, and no physical training or sampling occurred.

Streaming fresh law-matched states repaired the optimizer pathology. Across the three paired confirmation seeds, all 32,000 optimizer updates had zero clipping and the largest scaled pre-clip norm remained below `0.75`. The bounded teacher nevertheless missed every frozen derivative gate: mean audit score gain was `0.5138` overall and `0.7071` at the data end rather than at least `0.90`; mean flux cosine was `0.7214` rather than at least `0.98`; and mean relative flux L2 was `0.6933` rather than at most `0.15`. Signal was weakest near the reference end and strongest near the data end, but no seed approached the required all-bin flux accuracy.

Two stationary-null seeds selected analytic step zero. Seed `260813` nominated step 150 on the fixed selection panel, but both untouched audit lower bounds were negative. This is selection-panel false discovery rather than a reproduced population-null signal. Exact Stein identities passed, and the parent forensic replay showed very large old-train-versus-fresh risk gaps, confirming that streaming solved the previous fixed-state memorization problem. The remaining failure is specific to the empirical implicit trace-plus-drift estimator; it is not evidence against the population identity or the boundary-smooth representation.

##### Next gate: density-ratio classification controls

The additive entry point is `mnist.diag_d0_score_density_ratio_controls`. It binds the immutable streamed-control run and replaces only the statistical estimator. With equal class priors matched at every time anchor, balanced binary cross entropy has the Bayes raw logit

\[
\ell^*(s,\tau)=\log\frac{p_\tau(s)}{\nu_h(s)}.
\]

The same boundary-smooth scalar U-Net is used as one raw logit. Its state gradient is converted to the existing edge score and physical flux; the sigmoid is never differentiated. Teacher positives are sampled from the exact bounded four-anchor mixture and negatives from `Dirichlet(1)`. The null pools independent `Dirichlet(1)` states and assigns balanced labels by a stateless within-anchor swap. There are no Hessian probes, derivative auxiliary losses, class weights, label smoothing, focal loss, temperature scaling, or logit clipping.

A training-only 256-example balanced batch calibrates one positive optimizer multiplier to initial gradient norm `0.10`; that frozen multiplier is shared by every teacher and null arm. The train/selection-only pilot tests learning rates `3e-4,1e-4,3e-5,1e-5` for 2,000 streamed steps. Panel A alone nominates one checkpoint, panel B tests only that nominee, and no audit state participates in profile selection. Confirmation uses three paired seeds `260831,260832,260833`, 4,000 streamed steps, and four fresh independent 32-path panels per law: discovery A, confirmation B, and untouched audits C and D. Rejected null nominees are still evaluated on C and D.

Run the preflight first:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_controls `
  --runs-root runs/experiment12_d0_score_density_ratio_controls `
  --run-name production-density-ratio-controls `
  --device cuda `
  --stage preflight `
  --parent-stability-run-dir runs/experiment12_d0_score_control_stability_confirmation/20260718-232902_production-streamed-implicit-controls `
  --require-gate preflight
```

Copy the printed directory, then run the train/selection-only pilot:

```powershell
$ratioRun = "runs/experiment12_d0_score_density_ratio_controls/<timestamp>_production-density-ratio-controls"

.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_controls `
  --device cuda `
  --stage pilot `
  --resume-run-dir $ratioRun `
  --parent-stability-run-dir runs/experiment12_d0_score_control_stability_confirmation/20260718-232902_production-streamed-implicit-controls `
  --require-gate pilot
```

Only after the pilot passes, run confirmation:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_controls `
  --device cuda `
  --stage confirm `
  --resume-run-dir $ratioRun `
  --parent-stability-run-dir runs/experiment12_d0_score_control_stability_confirmation/20260718-232902_production-streamed-implicit-controls `
  --require-gate controls
```

The closed decisions distinguish provenance/operator failure, unresolved or invalid classifier optimization, selection false discovery, audit disagreement, no detectable ratio signal, value-only ratio learning, and `density_ratio_control_pipeline_repaired`. Only the last outcome authorizes planning a fresh physical one-image density-ratio score experiment. This patch performs no physical training or sampling and produces no generated image.

##### Saved density-ratio pilot result: real teacher signal, unresolved optimizer scale

The completed run `runs/experiment12_d0_score_density_ratio_controls/20260719-233220_production-density-ratio-controls` ended terminally at the pilot with `decision=classification_optimizer_unresolved`. Its 222-record registry verifies, all eight 2,000-step pilot tasks completed finite, and there were no task failures, physical training, or sampling. Confirmation and its C/D audit panels were correctly not created.

The null behaved correctly at every learning rate: analytic step zero was retained, all A/B lower bounds were nonpositive, and null clipping was at most `0.0393`. The teacher produced independently confirmed panel-B signal at `3e-5` and `1e-5`. At `3e-5`, the overall/data-end score gains were `0.3133/0.4927`, flux cosines were `0.5854/0.7404`, and relative flux L2 values were `0.8114/0.6759`; panel-B improvement lower bounds were `+0.002163/+0.010322`. At `1e-5`, the corresponding bounds were `+0.001599/+0.007982`. Neither profile qualified because teacher clipping was `0.2987` and `0.210`, above the frozen `0.10` limit and increasingly concentrated late in training.

This is evidence for a bounded-teacher density-ratio value signal with a clean stationary null, not yet for accurate score recovery. It is neither `no_detectable_density_ratio_signal` nor a repaired control pipeline. No physical task or sampler is authorized.

##### Next gate: paired-mixture density-ratio stability confirmation

The additive entry point is `mnist.diag_d0_score_density_ratio_stability_confirmation`. It preserves the boundary-smooth model, AdamW/EMA settings, exact balanced-BCE population objective, frozen multiplier `0.05173607018770852`, clipping threshold, and analytic score/flux gates. Only estimator variance changes.

For teacher density

\[
p_\tau=(1-\epsilon)\nu+\epsilon\sum_j w_j(\tau)\nu_j,\qquad \epsilon=0.5,
\]

draw shared Gamma coordinates `G`, form `S0=G/sum(G)`, draw one anchor `J~w(tau)` and one independent `E~Gamma(1,1)`, and form `SJ=(G+E e_J)/(sum(G)+E)`. For raw logit `ell`, one matched-time cluster contributes

\[
\tfrac12(1-\epsilon)\operatorname{softplus}[-\ell(S_0)]
+\tfrac12\epsilon\operatorname{softplus}[-\ell(S_J)]
+\tfrac12\operatorname{softplus}[\ell(S_0)].
\]

This Rao--Blackwellizes the mixture coin and couples the shared reference terms without changing the expectation. The null retains independent pooled `Dirichlet(1)` positive/negative states. Deterministic accumulation tests levels `2,4,8` for learning rates `3e-5,1e-5`, stopping after the first complete accumulation level with an eligible profile. Pilot profiles must satisfy the existing science gate and `<=0.10` clipping after warmup, over the final 500 steps, and over the final 200 steps.

Run preflight first:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_stability_confirmation `
  --runs-root runs/experiment12_d0_score_density_ratio_stability_confirmation `
  --run-name production-paired-density-ratio-controls `
  --device cuda `
  --stage preflight `
  --parent-density-ratio-run-dir runs/experiment12_d0_score_density_ratio_controls/20260719-233220_production-density-ratio-controls `
  --require-gate preflight
```

Resume the printed directory for the hierarchical pilot:

```powershell
$stabilityRun = "runs/experiment12_d0_score_density_ratio_stability_confirmation/<timestamp>_production-paired-density-ratio-controls"

.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_stability_confirmation `
  --device cuda `
  --stage pilot `
  --resume-run-dir $stabilityRun `
  --parent-density-ratio-run-dir runs/experiment12_d0_score_density_ratio_controls/20260719-233220_production-density-ratio-controls `
  --require-gate pilot
```

Only after the pilot passes, run fresh three-seed confirmation:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_stability_confirmation `
  --device cuda `
  --stage confirm `
  --resume-run-dir $stabilityRun `
  --parent-density-ratio-run-dir runs/experiment12_d0_score_density_ratio_controls/20260719-233220_production-density-ratio-controls `
  --require-gate controls
```

Only `density_ratio_control_pipeline_repaired` authorizes planning a fresh physical one-image ratio-score experiment. A pilot failure after accumulation level `8` requires a separate function-space trust/coercivity patch. This workflow performs no physical training or sampling.

##### Saved paired-mixture pilot result: variance reduced, coordinate clipping unresolved

The completed run `runs/experiment12_d0_score_density_ratio_stability_confirmation/20260720-011809_production-paired-density-ratio-controls` ended `status=complete`, `outcome=gate_failed`, with `decision=classification_variance_reduction_unresolved`. Its exact 332-record terminal registry verifies, the paired-estimator preflight passed, all twelve teacher/null pilot tasks completed finite with no task failures, and no confirmation, physical training, or sampling occurred.

The paired construction did reduce estimator noise: representative loss-variance ratios against the original IID estimator were about `0.064` and `0.0416`, while directional-gradient variance ratios were about `0.0989` and `0.0726`; every loss and directional-gradient 99% mean-difference interval contained zero. Accumulation improved both optimization and held-out teacher signal, but not enough to meet the frozen clipping limit. The maximum teacher/null clipping fractions were:

| Accumulation | Body LR | Maximum clipping fraction |
|---:|---:|---:|
| 2 | `3e-5` | `0.5060` |
| 2 | `1e-5` | `0.4707` |
| 4 | `3e-5` | `0.4000` |
| 4 | `1e-5` | `0.3360` |
| 8 | `3e-5` | `0.2693` |
| 8 | `1e-5` | `0.2593` |

At accumulation `8`, learning rate `3e-5`, the teacher selected EMA step 2000. Panel-B lower bounds were `+0.003428/+0.002624` overall/data-end; score gains were `0.9016/0.9318`, flux cosines `0.9532/0.9688`, and relative flux L2 values `0.3029/0.2486`. All six null arms selected analytic step zero. A tiny positive null panel-A bound at this profile disappeared on independent panel B; accumulation `8`, learning rate `1e-5` had nonpositive null A/B bounds but weaker teacher derivatives. The gate failure is therefore an optimizer-health result with real teacher signal, not evidence of absent density-ratio signal or a code crash. Do not run confirmation from this directory.

Subsequent layerwise inspection explains why merely shrinking the backbone is not the primary repair. In the width-32 legacy model, the scalar logit sums a 28-by-28 output map. Its 33 final-head parameters contribute roughly `99–99.9%` of the squared gradient norm, while the approximately 779,200-parameter backbone is not driving clipping. The same backbone already passed the supervised analytic teacher at about `0.999` score gain, `0.999` flux cosine, and `0.036` relative flux L2. This points to the spatial-sum parameter coordinate rather than insufficient representational capacity.

##### Next gate: normalized-head density-ratio coordinate repair

The additive entry point is `mnist.diag_d0_score_density_ratio_head_confirmation`. It binds the immutable 332-artifact paired-pilot run and introduces a versioned, function-equivalent scalar head. If `N=784`, the old head is

\[
\ell_{\mathrm{sum}}=\sum_{i=1}^{N}(w^\top h_i+b),
\]

and the new head is

\[
\ell_{\mathrm{mean}}=\frac1N\sum_{i=1}^{N}({w'}^\top h_i+b'),
\qquad w'=Nw,\quad b'=Nb.
\]

The two logits are exactly equal. State gradients, edge scores, physical fluxes, boundary admissibility, function class, width, and parameter count are unchanged. Head gradients scale by `1/N`, while backbone gradients are unchanged. To make an unclipped AdamW step and EMA exactly coordinate-conjugate, the backbone retains learning rate `eta`, epsilon `1e-8`, and weight decay `1e-4`, while final weight and bias use learning rate `N*eta`, epsilon `1e-8/N`, and weight decay `1e-4/N`. Global clipping remains norm `1` and is intentionally applied in the normalized coordinate; it is the optimizer geometry being repaired, not the BCE population objective.

Preflight proves CUDA and float64 logit equivalence, BCE and state-derivative equivalence, edge-score/flux equivalence, `1/N` head-gradient scaling, unchanged backbone gradients, and multi-step unclipped AdamW/EMA conjugacy. It also replays parent accumulation-8 checkpoints at steps `250,500,1000,1500,2000`, requiring finite gradients and median legacy-head squared-gradient share at least `0.95`. Boundary/operator, paired-estimator, and stream-replay checks remain mandatory. Converted legacy checkpoints are forensic-only; all new training starts from the exact zero initialization.

Fresh pilot training uses root seed `260881`, width `32`, accumulation `8`, body learning rates `3e-5,1e-5`, paired teacher/null streams, 2,000 updates, and fresh 16-path A/B panels. The frozen BCE multiplier remains `0.05173607018770852`, EMA remains `0.99`, and every optimizer and scientific threshold is unchanged. If a profile passes, confirmation uses seeds `260891,260892,260893`, 4,000 updates, and entirely fresh 32-path A/B/C/D panels. Only `density_ratio_control_pipeline_repaired` authorizes planning a separate physical one-image score experiment.

Run preflight first:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_head_confirmation `
  --runs-root runs/experiment12_d0_score_density_ratio_head_confirmation `
  --run-name production-normalized-head-density-ratio-controls `
  --device cuda `
  --stage preflight `
  --parent-paired-ratio-run-dir runs/experiment12_d0_score_density_ratio_stability_confirmation/20260720-011809_production-paired-density-ratio-controls `
  --require-gate preflight
```

Resume the printed directory for the pilot:

```powershell
$headRun = "runs/experiment12_d0_score_density_ratio_head_confirmation/<timestamp>_production-normalized-head-density-ratio-controls"

.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_head_confirmation `
  --device cuda `
  --stage pilot `
  --resume-run-dir $headRun `
  --parent-paired-ratio-run-dir runs/experiment12_d0_score_density_ratio_stability_confirmation/20260720-011809_production-paired-density-ratio-controls `
  --require-gate pilot
```

Only after the pilot passes, run confirmation:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_head_confirmation `
  --device cuda `
  --stage confirm `
  --resume-run-dir $headRun `
  --parent-paired-ratio-run-dir runs/experiment12_d0_score_density_ratio_stability_confirmation/20260720-011809_production-paired-density-ratio-controls `
  --require-gate controls
```

Closed decisions are `control_provenance_invalid`, `normalized_head_coordinate_invalid`, `classification_coordinate_repair_unresolved`, `classification_optimizer_invalid`, `selection_false_discovery`, `classification_audit_inconclusive`, `no_detectable_density_ratio_signal`, `density_ratio_value_only`, and `density_ratio_control_pipeline_repaired`. If clipping still fails, the next experiment may test an explicitly gated H1-like function-step trust region. If optimization is healthy but the derivative thresholds fail, report `density_ratio_value_only`; do not respond by shrinking the model. This workflow performs no physical training or sampling, and even a pass establishes only bounded synthetic density-ratio control recovery---not reconstruction, a known prior, an exact reverse kernel, or spatial Dirichlet--Ferguson convergence.

##### Saved normalized-head pilot result: optimizer repaired, selection panel underpowered

The completed normalized-head run `runs/experiment12_d0_score_density_ratio_head_confirmation/20260720-150202_production-normalized-head-density-ratio-controls` ended `classification_coordinate_repair_unresolved`. Its exact 125-artifact terminal registry and the transitive `332 -> 222 -> 381` provenance chain verify. All four pilot teacher/null tasks completed finite and boundary-admissible, every clipping window was exactly zero, and there were no task failures, confirmation tasks, physical training, or sampling. The coordinate repair therefore succeeded at its intended job.

The failed sealed teacher gate was not interpretable as an optimization or derivative failure because the exact analytic Bayes teacher itself failed the same saved 16-path panel B. Using whole-path bootstrap units, exact-oracle panel A had overall/data-end point improvements `+0.01621771096/+0.01721000665` and LB90 values `+0.00872031148/+0.00517841895`. Independent sealed panel B had point improvements `-0.00174841445/+0.00326400978` and LB90 values `-0.00913743689/-0.00565862046`. Panel B was therefore incapable of certifying even the population-risk minimizer on this draw. Do not diagnose insufficient model capacity or add H1 regularization from this run.

##### Next gate: oracle-qualified density-ratio selection power

The additive entry point is `mnist.diag_d0_score_density_ratio_selection_power_confirmation`. It binds the immutable 125-artifact normalized-head run and changes only evidence-panel power. The width-32 spatial-mean model, coordinate-conjugate AdamW groups, balanced BCE population objective, paired estimator, accumulation `8`, frozen loss multiplier `0.05173607018770852`, EMA `0.99`, global clip norm `1`, body learning rates `3e-5,1e-5`, and all scientific thresholds remain unchanged.

Before any new optimizer step, preflight reproduces the saved 16-path oracle forensic and builds one independent 256-path calibration panel. Every path has 32 anchors with fixed time strata `4,4,4,4,16`. The exact bounded-teacher log-density ratio must have finite positive one-sided 99% whole-path lower bounds overall and at the data end on the complete panel. Its two predetermined 128-path halves must each satisfy the existing positive 90% rule in both scopes. Calibration paths are never reused for selection or audit.

The pilot fixes fresh 128-path A/B panels for teacher and null. The exact teacher must qualify both actual teacher panels, overall and at the data end, before training starts. A failed panel is never resized, regenerated, or replaced after inspection; it closes the run as `evidence_panel_underpowered`. Root seed `260931`, 2,000 updates, the existing checkpoint schedule, sealed panel-B semantics, and fresh paired streams are frozen. A qualifying profile is ranked by mean teacher A/B BCE, maximum clipping, then smaller body learning rate.

Only a passing pilot unlocks the fresh three-seed confirmation. Confirmation uses seeds `260941,260942,260943`, 4,000 updates, and immutable 128-path A/B/C/D panels for each law. Every teacher panel is oracle-qualified before training, then the unchanged BCE lower-bound, score-gain, flux-cosine, relative-flux-L2, analytic-zero null, optimizer-health, and two-of-three teacher-seed gates are applied. The conservative legacy null rule remains authorizing. A separate non-authorizing multiplicity report distinguishes an A-only discovery event from a replicated false signal: if panel A alone violates the null rule while sealed B rejects the nominee, report `null_gate_multiplicity_inconclusive` rather than optimizer instability.

Closed decisions are `control_provenance_invalid`, `oracle_power_invalid`, `evidence_panel_underpowered`, `null_gate_multiplicity_inconclusive`, `classification_power_confirmation_unresolved`, `classification_optimizer_invalid`, `selection_false_discovery`, `classification_audit_inconclusive`, `no_detectable_density_ratio_signal`, `density_ratio_value_only`, and `density_ratio_control_pipeline_repaired`. Only an oracle-qualified `classification_power_confirmation_unresolved` result authorizes planning an H1 function-step trust patch. Only `density_ratio_control_pipeline_repaired` authorizes planning fresh physical one-image score training.

Run preflight first:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_selection_power_confirmation `
  --runs-root runs/experiment12_d0_score_density_ratio_selection_power_confirmation `
  --run-name production-density-ratio-selection-power `
  --device cuda `
  --stage preflight `
  --parent-normalized-head-run-dir runs/experiment12_d0_score_density_ratio_head_confirmation/20260720-150202_production-normalized-head-density-ratio-controls `
  --require-gate preflight
```

Resume the printed directory for the pilot:

```powershell
$powerRun = "runs/experiment12_d0_score_density_ratio_selection_power_confirmation/<timestamp>_production-density-ratio-selection-power"

.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_selection_power_confirmation `
  --device cuda `
  --stage pilot `
  --resume-run-dir $powerRun `
  --parent-normalized-head-run-dir runs/experiment12_d0_score_density_ratio_head_confirmation/20260720-150202_production-normalized-head-density-ratio-controls `
  --require-gate pilot
```

Only after the pilot passes:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_selection_power_confirmation `
  --device cuda `
  --stage confirm `
  --resume-run-dir $powerRun `
  --parent-normalized-head-run-dir runs/experiment12_d0_score_density_ratio_head_confirmation/20260720-150202_production-normalized-head-density-ratio-controls `
  --require-gate controls
```

This workflow performs no physical training and no sampling. A pass proves only that the bounded synthetic ratio controls are detectable on fixed, oracle-qualified evidence panels and that the learned derivatives satisfy the frozen control gates. It does not prove physical-score learnability, reconstruction, a known prior, an exact reverse kernel, or spatial Dirichlet--Ferguson convergence.

##### Saved oracle-qualified pilot result: teacher learned, legacy null discovery rule failed

The completed run `runs/experiment12_d0_score_density_ratio_selection_power_confirmation/20260720-204514_production-density-ratio-selection-power` ended `status=complete`, `outcome=gate_failed`, and `decision=null_gate_multiplicity_inconclusive`. Its exact 123-record terminal registry and transitive `125 -> 332 -> 222 -> 381` provenance chain verify. The independent 256-path oracle calibration, both predetermined calibration halves, and the actual fixed 128-path teacher A/B panels all passed their declared power gates before training. All four 2,000-update pilot tasks completed finite and boundary-admissible, every clipping window was zero, and there were no task failures, confirmation tasks, physical training, or sampling.

This powered run resolves the earlier ambiguity. Both teacher arms selected nonzero EMA step 2000 and passed sealed panel B. At body learning rate `3e-5`, teacher A/B BCE was `0.6801083346/0.6814580763`; A LB90 was `+0.0101354213/+0.0137847375` and sealed-B LB90 was `+0.0088853571/+0.0093210751`, overall/data-end. Score gain was `0.886891/0.919885`, flux cosine was `0.949558/0.965450`, and relative flux L2 was `0.317918/0.265274`. The `1e-5` arm also passed A/B, but had higher mean BCE (`0.6817931156` versus `0.6807832055`) and weaker derivatives. The already declared ranking therefore prefers `3e-5`, with accumulation `8`.

Both null arms behaved correctly on independent sealed evidence: their panel-A nominees were rejected by panel B and the primary selected checkpoint remained analytic step zero. The `3e-5` null A LB90 was `-2.0771e-6/+2.8700e-5`, while B was `-3.1140e-5/-3.0777e-5`. The `1e-5` null A LB90 was `-6.6940e-6/+9.0041e-6`, while B was `-4.1019e-6/-5.4485e-6`. Thus each candidate failed only because the data-end bound on discovery panel A was microscopically positive. Requiring a panel used to search checkpoints to also certify the null treats a selected discovery fluctuation as confirmatory evidence. It is a multiplicity-semantics defect, not a capacity, optimizer, boundary-domain, or teacher-signal failure. Do not run the blocked confirmation from this directory, and do not introduce H1 regularization on this evidence.

##### Next gate: multiplicity-aware density-ratio confirmation

The additive entry point is `mnist.diag_d0_score_density_ratio_multiplicity_confirmation`. It binds the immutable 123-record selection-power run, replays its sealed evidence without training, freezes the already ranked profile `(body learning rate 3e-5, accumulation 8)`, and then runs a fresh three-seed confirmation. Parent models, states, and optimizer state are never used to initialize confirmation; only the frozen hyperparameter profile and validated model/operator definitions are reused.

The corrected null semantics are fixed before fresh confirmation. Panel A is discovery-only: it nominates one checkpoint and never authorizes a null signal. Panels B, C, and D are confirmatory. For the family

`F = {three model seeds} x {B,C,D} x {overall,data-end}`,

let `Delta_hat_j` be the whole-path mean BCE improvement at the panel-A nominee and `se_hat_j` its whole-path standard error. A deterministic 50,000-replicate studentized whole-path max-T bootstrap computes

`Z_j* = (Delta_hat_j* - Delta_hat_j) / se_hat_j*` and `c_0.95 = Q_0.95[max_j Z_j*]`

and simultaneous lower bounds `L_j^sim = Delta_hat_j - c_0.95 * se_hat_j`. Common path indices are resampled jointly across model seeds within each panel role, while the independent B/C/D roles are resampled independently. The analytic-zero null passes only if every simultaneous lower bound in the complete 18-comparison family is nonpositive. This gives one-sided simultaneous 95% lower bounds and controls the authorizing family-wise false-signal probability at `0.05`; raw legacy panelwise bounds remain advisory. A positive B bound is a selection false discovery, while a positive C/D bound is an audit false discovery. Panel A is reported, but is excluded from this family by construction.

The replay stage first applies the same rule to the immutable parent sealed-B evidence across its two learning-rate candidates and verifies the unchanged profile ranking. It writes a hash-bound frozen profile and performs no optimizer step. Confirmation then uses fresh root seed `260961`, paired model seeds `260971,260972,260973`, fresh zero initialization, 4,000 updates, and disjoint fixed 128-path A/B/C/D panels for each law. Every teacher panel is exact-oracle-qualified before training. The balanced BCE objective, width-32 normalized-head model, coordinate-conjugate AdamW, loss multiplier `0.05173607018770852`, EMA `0.99`, clip norm `1`, teacher classification and derivative thresholds, and two-of-three teacher-seed rule remain unchanged.

Run provenance/operator preflight first:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_multiplicity_confirmation `
  --runs-root runs/experiment12_d0_score_density_ratio_multiplicity_confirmation `
  --run-name production-multiplicity-aware-density-ratio-controls `
  --device cuda `
  --stage preflight `
  --parent-selection-power-run-dir runs/experiment12_d0_score_density_ratio_selection_power_confirmation/20260720-204514_production-density-ratio-selection-power `
  --require-gate preflight
```

Resume the printed directory to replay the immutable pilot, qualify multiplicity, and freeze the selected profile:

```powershell
$multiplicityRun = "runs/experiment12_d0_score_density_ratio_multiplicity_confirmation/<timestamp>_production-multiplicity-aware-density-ratio-controls"

.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_multiplicity_confirmation `
  --device cuda `
  --stage replay `
  --resume-run-dir $multiplicityRun `
  --parent-selection-power-run-dir runs/experiment12_d0_score_density_ratio_selection_power_confirmation/20260720-204514_production-density-ratio-selection-power `
  --require-gate replay
```

Only after replay passes, run the fresh confirmation:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_multiplicity_confirmation `
  --device cuda `
  --stage confirm `
  --resume-run-dir $multiplicityRun `
  --parent-selection-power-run-dir runs/experiment12_d0_score_density_ratio_selection_power_confirmation/20260720-204514_production-density-ratio-selection-power `
  --require-gate controls
```

This remains a controls-only experiment. Every artifact records `physical_training_performed: 0` and `sampling_performed: 0`. A passing multiplicity-aware synthetic confirmation authorizes only planning a separately named physical one-image density-ratio score experiment with fresh states. It does not authorize sampling and does not establish physical-score learnability, reconstruction, a known prior, an exact finite-time reverse kernel, a time-zero density, or spatial Dirichlet--Ferguson convergence.

##### Saved multiplicity result: values passed, physical-flux derivatives failed

The completed run `runs/experiment12_d0_score_density_ratio_multiplicity_confirmation/20260721-000607_production-multiplicity-aware-density-ratio-controls` ended `status=complete`, `outcome=gate_failed`, and `decision=density_ratio_value_only`. Its exact 263-record terminal registry and transitive `263 -> 123 -> 125 -> 332 -> 222 -> 381` provenance chain verify. All six teacher/null tasks completed finite and boundary-admissible, every clipping window was zero, all teacher nominees passed sealed panel B, all nulls selected analytic step zero, and there were no task failures, physical training, or sampling.

The multiplicity repair worked. The single 18-member B/C/D null family passed with max-T critical value `2.78527696`; all simultaneous lower bounds were negative, from `-1.987539756e-4` to `-5.643400881e-5`, and the minimum adjusted p-value was `0.65210696`. The failure is therefore not a repeated null false discovery.

The teachers learned the density-ratio values, but not the physical derivatives accurately enough. Seeds `260971,260972,260973` selected EMA steps `2750,3750,4000`. On independent C/D panels, overall score gains ranged from `0.905564` to `0.940955` and data-end gains from `0.933611` to `0.954342`; these pass the frozen `0.90` gain target. In contrast, flux cosine ranged from `0.956994` to `0.974420`, below `0.98`, and relative flux L2 ranged from `0.226424` to `0.292159`, above `0.15`. Consequently zero of three seeds passed the derivative gate. This is a `density_ratio_value_only` result, not evidence of insufficient width, clipping, a boundary-domain defect, weak panels, or bad null semantics.

##### Next gate: EMA-proximal H1 function-step trust confirmation

The additive entry point is `mnist.diag_d0_score_density_ratio_h1_trust_confirmation`. It binds the immutable 263-record multiplicity run and preserves width `32`, the paired balanced-BCE population objective, normalized-head coordinate, body learning rate `3e-5`, accumulation `8`, EMA `0.99`, loss multiplier, global clip norm `1`, and all scientific thresholds. It performs no physical training and no sampling.

For raw and EMA logits, define the proximal function increment

`d(s,tau) = f_raw(s,tau) - stopgrad(f_EMA(s,tau))`.

With the existing physical edge derivative `r_e`, harmonic mobility `theta_e`, and `N=784`, use

`Gamma(d,d) = N * sum_e theta_e * r_e(d)^2`, and the implemented fixed-grid
average is `Gamma_bar(d,d) = Gamma(d,d)/(2*N^2) = mean_e theta_e*r_e(d)^2`.

and the reference-law trust penalty

`P(d) = E_nu[d^2/a0^2 + Gamma_bar(d,d)/a1^2]`.

The L2 component is essential because the Dirichlet energy cannot see constant logit shifts. The Gamma component is the mass-simplex H1 geometry already used by the physical score/flux operator; it is not image total variation. The EMA branch is stop-gradient, so the penalty controls the update in function space and vanishes when raw and EMA functions agree rather than permanently shrinking the Bayes log-density ratio.

Calibration is deterministic, fresh, and training-only. From one shadow Adam step relative to the pre-step EMA function, set

`a0 = RMS_nu(d)`, `a1 = sqrt(E_nu Gamma_bar(d,d))`, and

`lambda_base = ||grad(c_BCE R)||_2 / ||grad P||_2`.

Every scale and gradient factor must be finite and strictly greater than `1e-8`; otherwise calibration fails closed without clamping. For `q in {0,0.1,0.3,1}`, optimize

`L_q = c_BCE R_pair + q * lambda_base * (P_A + P_B)/2`,

where `P_A` and `P_B` use two independent 32-state reference banks at every optimizer update.

Preflight verifies exact parent/transitive provenance; Gamma symmetry, nonnegativity, orientation, and analytic fixtures; zero penalty for `raw=EMA`; detection of constant shifts by L2; no EMA gradient; finite boundary behavior and CUDA second-order differentiation; stateless stream isolation and candidate-order invariance; and a 25-step `q=0` regression against the old runner.

The pilot uses root seed `261001`, model seed `261011`, 4,000 updates, and fresh fixed 128-path A/B panels for teacher and null. Panel A nominates one checkpoint and sealed B evaluates it once. The null uses a global max-T family across all four `q` candidates and both scopes. A profile is eligible only if it satisfies the previous strict teacher/null and optimizer gates and improves relative flux L2 by at least `10%` both overall and at the data end versus `q=0`. Rank eligible profiles by lowest worst overall/per-bin relative flux L2, then highest minimum-bin cosine, lowest sealed-B BCE, and smaller `q`.

Only a passing pilot unlocks confirmation. Confirmation freezes the selected profile, uses paired seeds `261021,261022,261023` for 4,000 updates, and evaluates new disjoint 128-path A/B/C/D panels. Every existing strict BCE-improvement, score-gain, flux-cosine, relative-flux-L2, analytic-zero null, optimizer-health, and two-of-three teacher-seed rule remains unchanged. H1-specific health gates only finite calibration, components, and second-order gradients plus the existing clipping limit; no outcome-dependent trust cutoff is added.

Closed decisions are `control_provenance_invalid`, `h1_operator_invalid`, `h1_calibration_invalid`, `evidence_panel_underpowered`, `h1_optimizer_invalid`, `h1_overregularized`, `h1_function_step_unresolved`, `selection_false_discovery`, `classification_audit_inconclusive`, `h1_density_ratio_value_only`, and `density_ratio_control_pipeline_repaired`.

Run preflight first:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_h1_trust_confirmation `
  --runs-root runs/experiment12_d0_score_density_ratio_h1_trust_confirmation `
  --run-name production-h1-function-step-density-ratio-controls `
  --device cuda `
  --stage preflight `
  --parent-multiplicity-run-dir runs/experiment12_d0_score_density_ratio_multiplicity_confirmation/20260721-000607_production-multiplicity-aware-density-ratio-controls `
  --require-gate preflight
```

Resume the printed directory for the pilot:

```powershell
$h1Run = "runs/experiment12_d0_score_density_ratio_h1_trust_confirmation/<timestamp>_production-h1-function-step-density-ratio-controls"

.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_h1_trust_confirmation `
  --device cuda `
  --stage pilot `
  --resume-run-dir $h1Run `
  --parent-multiplicity-run-dir runs/experiment12_d0_score_density_ratio_multiplicity_confirmation/20260721-000607_production-multiplicity-aware-density-ratio-controls `
  --require-gate pilot
```

Only after the pilot passes, run confirmation:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_h1_trust_confirmation `
  --device cuda `
  --stage confirm `
  --resume-run-dir $h1Run `
  --parent-multiplicity-run-dir runs/experiment12_d0_score_density_ratio_multiplicity_confirmation/20260721-000607_production-multiplicity-aware-density-ratio-controls `
  --require-gate controls
```

This remains a controls-only experiment. Every artifact records `physical_training_performed: 0` and `sampling_performed: 0`. Only `density_ratio_control_pipeline_repaired` authorizes planning a fresh physical one-image score experiment. Passing proves only that the bounded synthetic density-ratio control is value- and derivative-accurate under this EMA-proximal function-step coordinate; it does not establish physical-score learnability, reconstruction, a known prior, a time-zero density, an exact reverse kernel, sample quality, or spatial Dirichlet--Ferguson convergence.

##### Saved H1 pilot result: helpful but under-scaled function-step control

The completed run `runs/experiment12_d0_score_density_ratio_h1_trust_confirmation/20260721-114934_production-h1-function-step-density-ratio-controls` ended `status=complete`, `outcome=gate_failed`, and `decision=h1_function_step_unresolved`. Its exact 301-record registry has SHA-256 `26c42bc045903253c504a00499318292b3e5612f54b16e103b0872f8933fc6f4`. Preflight, the H1 operator, training-only calibration, oracle panel power, all eight teacher/null tasks, optimizer health, and the eight-member simultaneous null family passed. Every clipping window was zero, all nulls selected analytic step zero, and there were no task failures, confirmation tasks, physical training, or sampling.

The H1 term moved the learned derivatives in the intended direction, but did not reach the frozen thresholds. The `q=0` teacher selected step 3000 with score gain `0.92538/0.94396`, overall flux cosine `0.96584`, minimum-bin cosine `0.94262`, overall/data-end relative flux L2 `0.26151/0.22549`, and worst-bin relative flux L2 `0.33689`. The strongest arm, `q=1`, selected step 4000 and improved these values to gain `0.94155/0.95497`, overall cosine `0.97444`, minimum-bin cosine `0.96223`, relative L2 `0.22745/0.19746`, and worst-bin relative L2 `0.28084`. This is a `13.0%` overall and `12.4%` data-end relative-L2 reduction, but it still misses cosine `>=0.98` overall and relative L2 `<=0.15` overall and `<=0.20` in every bin.

The one-shadow-step calibration produced `a0=0.03735975761867785`, `a1=0.00541742970034323`, and `lambda_base=5.620791829277054e-8`. During the `q=1` teacher run, the median realized H1-to-BCE gradient-norm ratio was only about `0.0569`. The fixed coefficient therefore supplied a useful but weak perturbation after calibration. There is also a checkpoint-duration confound: `q=0` was judged at step 3000 while every nonzero arm was judged at step 4000. This run does not support increasing model width, weakening derivative thresholds, or beginning physical training. It motivates controlling the realized H1 gradient strength online and comparing all arms at one fixed endpoint.

##### Next gate: online gradient-ratio-controlled H1 confirmation

The additive entry point is `mnist.diag_d0_score_density_ratio_h1_gradient_control_confirmation`. It binds the immutable 301-record H1 run and preserves width `32`, body learning rate `3e-5`, accumulation `8`, paired balanced BCE, normalized-head coordinate, EMA `0.99`, global clipping norm `1`, the existing stopped-EMA H1 geometry and scales, and every scientific threshold. The old `lambda_base` is forensic provenance only and is not used for optimization.

Let

`g_t = grad_theta(c_BCE * L_BCE,t)` and `h_t = grad_theta(H_t)`,

where `H_t` is the mean of the existing two stopped-EMA H1 reference banks. For target ratio `rho in {0,0.1,0.3,1}`, use the frozen ramp

`rho_t = rho * min(1, max(0,(t-1)/100))`

and the stopped coefficient

`lambda_t = rho_t * ||g_t||_2 / ||h_t||_2`,

then apply `g_total = g_t + stopgrad(lambda_t) h_t`. Norms are evaluated in float64 normalized-head parameter coordinates and the unchanged global clip is applied once after composition. Step one has zero H1 contribution. The norm floor is `1e-12`: a post-ramp H1 floor hit or nonfinite coefficient fails closed, while a BCE-floor hit is a recorded stationary no-op. The realized preclip component ratio must track `rho_t` within relative error `1e-4`, and at least `99%` of post-ramp updates must be controller-active. This finite-time vector-field controller preserves the BCE fixed point, but it is not claimed to be gradient descent on one fixed scalar objective.

Preflight proves ratio algebra, stop-gradient semantics, positive H1-rescaling invariance, ramp and floor branches, fixed-point behavior, exact 25-step `rho=0` equivalence, finite CUDA second-order differentiation, boundary admissibility, stream/order invariance, and exact interruption replay. A parent forensic evaluates `q=0` and `q=1` at matched steps 3000 and 4000 on fresh advisory states; those states and weights are never reused for selection.

The pilot fixes root seed `261041`, model seed `261051`, 4,000 updates, and fresh disjoint 128-path A/B panels. All four ratios share initialization, BCE streams, and H1 reference streams, and step 4000 is the only scientific endpoint. Panel A ranks one nonzero ratio; sealed B opens exactly once for that nominee and its matched `rho=0` comparator. No fallback ratio is tried after B is inspected. The candidate must pass the unchanged BCE, score-gain, cosine, per-bin relative-L2, controller-health, and clipping gates. It must also reduce relative flux L2 by at least `10%` against the matched step-4000 baseline overall and at the data end, with positive simultaneous whole-path lower bounds in both scopes. The authorizing null family contains `4 ratios x 2 scopes = 8` sealed-B comparisons.

Only a passing pilot unlocks confirmation. Confirmation freezes only the ratio and uses seeds `261061,261062,261063` with fresh 128-path A/B/C/D panels. For each seed it trains the selected-ratio teacher, an identically coupled `rho=0` teacher, and the selected-ratio null, giving nine tasks. At least two selected-ratio teachers must pass every existing strict derivative gate. A single simultaneous matched-effect family contains `3 seeds x {B,C,D} x {overall,data-end} = 18` members; at least two seeds must have six positive simultaneous bounds and point reductions of at least `10%`. The independent 18-member null family must have every simultaneous lower bound nonpositive. All nine optimizer/controller trajectories must remain healthy.

Closed decisions are `control_provenance_invalid`, `h1_gradient_controller_invalid`, `evidence_panel_underpowered`, `h1_controller_optimizer_invalid`, `h1_controller_overregularized`, `h1_strength_grid_unresolved`, `selection_false_discovery`, `h1_causal_effect_unconfirmed`, `classification_audit_inconclusive`, `h1_effect_audit_inconclusive`, `h1_density_ratio_value_only`, and `density_ratio_control_pipeline_repaired`. If ratio tracking is healthy but the derivative gates still fail, stop tuning this EMA-proximal mechanism and report `h1_density_ratio_value_only`.

Run preflight first:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_h1_gradient_control_confirmation `
  --runs-root runs/experiment12_d0_score_density_ratio_h1_gradient_control_confirmation `
  --run-name production-gradient-controlled-h1-density-ratio-controls `
  --device cuda `
  --stage preflight `
  --parent-h1-run-dir runs/experiment12_d0_score_density_ratio_h1_trust_confirmation/20260721-114934_production-h1-function-step-density-ratio-controls `
  --require-gate preflight
```

Resume the printed directory for the fixed-endpoint pilot:

```powershell
$gradientRun = "runs/experiment12_d0_score_density_ratio_h1_gradient_control_confirmation/<timestamp>_production-gradient-controlled-h1-density-ratio-controls"

.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_h1_gradient_control_confirmation `
  --device cuda `
  --stage pilot `
  --resume-run-dir $gradientRun `
  --parent-h1-run-dir runs/experiment12_d0_score_density_ratio_h1_trust_confirmation/20260721-114934_production-h1-function-step-density-ratio-controls `
  --require-gate pilot
```

Only after the pilot passes, run confirmation:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_score_density_ratio_h1_gradient_control_confirmation `
  --device cuda `
  --stage confirm `
  --resume-run-dir $gradientRun `
  --parent-h1-run-dir runs/experiment12_d0_score_density_ratio_h1_trust_confirmation/20260721-114934_production-h1-function-step-density-ratio-controls `
  --require-gate controls
```

This remains a controls-only experiment. Every artifact records `physical_training_performed: 0` and `sampling_performed: 0`. Only `density_ratio_control_pipeline_repaired` authorizes planning fresh physical one-image score training. A pass proves only optimizer-healthy, derivative-accurate bounded synthetic density-ratio controls under this fixed-grid controller; it does not establish physical-score learnability, reconstruction, a known prior, a time-zero density, an exact reverse kernel, sampling quality, or spatial Dirichlet--Ferguson convergence.

##### Exact Jacobi-split Eulerian denoising feasibility gate

The run `runs/experiment12_d0_score_density_ratio_h1_gradient_control_confirmation/20260722-000701_production-gradient-controlled-h1-density-ratio-controls` has an immutable 277-record registry with SHA-256 `0341f1defa29029fce03c638d86b15db1565c2f4d488b7fce8413fa140dc71ab`. Its saved `selection_false_discovery` decision is a report defect: no nonzero profile was nominated, panel B was never opened, and the null family was `not_evaluated` with size zero. All eight tasks completed 4,000 updates finite, boundary/optimizer healthy, and with zero clipping. The new workflow records the factual result as `h1_strength_grid_unresolved` without rewriting the parent.

The replacement target is the exact conditional score of the fixed-grid Eulerian edge semigroup. For pair total `r=s_tail+s_head` and head fraction `x=s_head/r`, one phase is a neutral Jacobi diffusion. With dimensionless exposure `u`, draw

`M ~ q^(2 alpha)(2u), L|M,x ~ Bin(M,x), Y|M,L ~ Beta(alpha+L,alpha+M-L)`

and train later on the exact latent denoising label

`Z = L - M*Y`, with `E[Z|later,phase] = Y*(1-Y)*d_Y log(p/nu)`.

This is the Jacobi counterpart of DDPM noise prediction. The workflow contains no Euler residual, BCE target, H1 penalty, Gaussian transition proxy, limiter, floor, projection, target clipping, physical training, or reverse sampler.

The additive entry point is `mnist.diag_d0_jacobi_denoising_feasibility`. It implements:

- the four exact torus matchings and palindromic seven-phase Strang sweep;
- an alpha-1 Legendre density/CDF/arrival-score oracle with explicit tail certificates;
- the arbitrary-precision Jenkins--Spano alternating-series ancestral-count sampler, which fails closed on every cap hit;
- production-support and projected 90-million-transition cost checks;
- pair/global mass conservation, Dirichlet stationarity, detailed balance, split refinement, and exact teacher/null denoising controls.

The inexpensive noncommuting-matrix Strang study is advisory only. It does
not stand in for refinement of the state-dependent Eulerian Jacobi split,
because crossing colors change each conserved pair total. The authorizing
`actual_eulerian_refinement_pass` therefore remains fail-closed until the
small-time exact transition passes the kernel and throughput gate and can
support a genuine exact-phase refinement rollout. This prevents a generic
second-order splitting fixture from being reported as Eulerian evidence.
The workflow does retain a non-authorizing local check of the true grid-28
color generator: exact Jacobi eigenmoments are differentiated on Fourier
linear, quadratic-mass, and cubic-mass observables and compared with the
state-dependent `1/r` generator.

Run preflight:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_denoising_feasibility `
  --runs-root runs/experiment12_d0_jacobi_denoising_feasibility `
  --run-name production-exact-jacobi-feasibility `
  --device cuda `
  --stage preflight `
  --parent-gradient-control-run-dir runs/experiment12_d0_score_density_ratio_h1_gradient_control_confirmation/20260722-000701_production-gradient-controlled-h1-density-ratio-controls `
  --require-gate preflight
```

Resolve the generated directory without a literal timestamp placeholder:

```powershell
$jacobiRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_denoising_feasibility" -Directory |
  Where-Object Name -Like "*_production-exact-jacobi-feasibility" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName
```

Run the exact-kernel gate:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_denoising_feasibility `
  --device cuda `
  --stage kernel `
  --resume-run-dir $jacobiRun `
  --parent-gradient-control-run-dir runs/experiment12_d0_score_density_ratio_h1_gradient_control_confirmation/20260722-000701_production-gradient-controlled-h1-density-ratio-controls `
  --require-gate kernel
```

Only if that gate passes, run controls:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_denoising_feasibility `
  --device cuda `
  --stage controls `
  --resume-run-dir $jacobiRun `
  --parent-gradient-control-run-dir runs/experiment12_d0_score_density_ratio_h1_gradient_control_confirmation/20260722-000701_production-gradient-controlled-h1-density-ratio-controls `
  --require-gate controls
```

Only `exact_jacobi_denoising_feasible` authorizes a separate one-image phase-conditioned `Z`-prediction patch. A failure means the exact transition implementation or its production cost remains unresolved; it does not authorize returning to residual, classifier, or Gaussian-proxy targets. This remains a fixed-grid `alpha=1` claim and does not establish spatial Dirichlet--Ferguson convergence.

##### Certified spectral Rao--Blackwell Jacobi kernel gate

The completed run `runs/experiment12_d0_jacobi_denoising_feasibility/20260722-142613_production-exact-jacobi-feasibility` ended `status=complete`, `outcome=gate_failed`, with decision `jacobi_kernel_numerically_unresolved`. Its 16-record terminal registry has SHA-256 `fd8695f463e1bd82f5d0be16059ab1be41fd9dbf727603264600881d5addc836`. The Legendre density/CDF/arrival-score oracle passed normalization, detailed balance, semigroup, eigenmoment, production-support, CUDA-agreement, and distribution controls. The obstruction was confined to the ancestral representation: one small-time resource-cap failure, one invalid alternating-series certificate, 14 uncertified support draws, and a projected 5,080.942 hours for the 89,915,392-transition cache. No split control, physical training, or reverse sampling ran.

The additive repair entry point is `mnist.diag_d0_jacobi_rb_denoising_feasibility`. It eliminates the intractable sampled ancestral count by using the exact Rao--Blackwellized target

`Z_bar(X,Y,u) = Y*(1-Y)*d_Y log k_u(Y|X) = E[L-M*Y | X,Y,u]`.

The tower property gives `E[Z_bar|later,phase] = E[L-M*Y|later,phase]`, so this changes only label variance, not the DDPM-like population optimum. The model contract continues to expose only the later full state, reverse time, phase/color/duration, and class label; the earlier fraction, inverse-CDF uniform, and certificate data are forbidden inputs.

The new alpha-one engine uses a CDF-only Legendre recurrence for interval inverse sampling and evaluates the conormal numerator `G=Y*(1-Y)*d_Y k_u` directly before forming `Z_bar=G/k_u`. `python-flint==0.9.0` Arb resolves every authorizing comparison, including rigorously computed Arb omitted-series radii. A batched CUDA recurrence is retained only for independent point/enclosure diagnostics. The frozen authorizing profile deliberately bypasses its ordinary-float inverse proposal and sends every active transition through transition-local Arb certification; it never mistakes a CUDA point agreement for an interval proof. A draw is accepted only when the inverse lies in one binary64 rounding cell and the target quotient has a positive-density, unique-rounding certificate. The end-to-end benchmark evolves an actual seven-phase, state-dependent grid path, includes complete output-shard I/O, validates resumed shards, and triggers a recorded resource guard from the projected 64-path cost. Consequently this gate may honestly end `spectral_inversion_computationally_infeasible` until a genuinely authorizing batched interval backend exists. There is no Gaussian/Euler fallback, finite-ancestral-count truncation, exposure binning, clipping, floor, limiter, projection, or target clipping.

The first production kernel attempt, `20260722-165831_production-certified-spectral-rb-kernel`, completed all 294 support rows and then raised `Arb returned nonfinite endpoints` in the benchmark probe at `x=0.03`, `u=0.00011484375`. The Jacobi value was finite: a low-precision but rigorous Arb cancellation ball exceeded binary64 range, and diagnostic endpoint conversion aborted before the frozen precision ladder reached its decisive 1024-bit evaluation. Diagnostic endpoint extraction is now non-authorizing and may return an unbounded sentinel while exact Arb comparisons continue. Of the old support shards, 270 are valid certificates and 24 contain that obsolete early-abort result. A fresh run may import the 270 registry-bound certificates with `--support-shard-source-run-dir`; the 24 failed rows are recomputed and the immutable failed run is never resumed under changed sources.

Install the pinned certificate backend and run preflight:

```powershell
.\.venv\Scripts\python.exe -m pip install python-flint==0.9.0

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_denoising_feasibility `
  --runs-root runs/experiment12_d0_jacobi_rb_denoising_feasibility `
  --run-name production-certified-spectral-rb-kernel-retry `
  --device cuda `
  --stage preflight `
  --parent-jacobi-feasibility-run-dir runs/experiment12_d0_jacobi_denoising_feasibility/20260722-142613_production-exact-jacobi-feasibility `
  --require-gate preflight
```

Resolve the printed run and execute the kernel stage. Run the target stage only
if the kernel gate passes; a recorded computational-infeasibility result is a
terminal kernel result, not permission to bypass it.

```powershell
$rbRun = (Get-ChildItem "runs/experiment12_d0_jacobi_rb_denoising_feasibility" -Directory | Where-Object Name -Like "*_production-certified-spectral-rb-kernel-retry" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_denoising_feasibility `
  --device cuda `
  --stage kernel `
  --resume-run-dir $rbRun `
  --support-shard-source-run-dir runs/experiment12_d0_jacobi_rb_denoising_feasibility/20260722-165831_production-certified-spectral-rb-kernel `
  --parent-jacobi-feasibility-run-dir runs/experiment12_d0_jacobi_denoising_feasibility/20260722-142613_production-exact-jacobi-feasibility `
  --require-gate kernel

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_denoising_feasibility `
  --device cuda `
  --stage target `
  --resume-run-dir $rbRun `
  --parent-jacobi-feasibility-run-dir runs/experiment12_d0_jacobi_denoising_feasibility/20260722-142613_production-exact-jacobi-feasibility `
  --require-gate target
```

Only `exact_jacobi_rb_kernel_feasible` authorizes a separately named, production state-dependent Strang-refinement patch. It does not authorize one-image training or reconstruction.

##### Certified fused-CUDA Rao--Blackwell confirmation

The immutable retry run `runs/experiment12_d0_jacobi_rb_denoising_feasibility/20260722-203339_production-certified-spectral-rb-kernel-retry` ended `spectral_inversion_computationally_infeasible`. Its 315-record terminal registry has SHA-256 `309a4296691684f1cc7ce26bfc243accfb51b8dd2b9ea50a6c43d93dea164e9e`. This is a resource result, not a numerical rejection: all 294 support transitions were certified and correctly rounded, the spectral-law controls passed, no approximate transition mechanism was used, and the only failed kernel checks were the 100% Arb fallback/cost, the resulting `0.6147` transitions/s, the projected `40,631.8` hours, and the intentionally skipped full-path benchmark. Target controls were consequently not evaluated. No physical training or reverse sampling ran.

The additive entry point `mnist.diag_d0_jacobi_rb_cuda_confirmation` keeps that run and its eight fingerprinted source files immutable. It replaces scalar Arb-for-every-transition with a header-free NVRTC kernel compiled from PyTorch's bundled CUDA runtime. A float64 inverse-CDF pass starts from the analytic eigenmoment/Cantelli bracket and makes 56 safeguarded bisections, but never authorizes an output. The authorizing pass uses double-double centers, outward radii, directed IEEE operations, one certified evaluation of `q=exp(-2u)`, outward-recursive spectral powers, 16-mode tail checks, and strict inverse-CDF rounding-cell inequalities. To avoid dependency blow-up, Legendre recurrences use directed enclosures of their a posteriori local residuals together with the published rigorous bound `|P_tilde_n-P_n| <= ((n+1)(n+2)/4) max_j |epsilon_j|` (Johansson--Mezzarobba, Proposition 5); this is not an empirical ULP allowance. The Rao--Blackwell label is accepted only when the density has a positive lower bound and the `G/k` enclosure lies in one binary64 rounding cell. Unresolved entries escalate first to a strengthened CUDA certificate and then to candidate-local Arb; there is no Euler/Gaussian transition, finite ancestral proxy, exposure binning, clipping, flooring, limiting, projection, or altered target.

The version-2 Philox stream is keyed by canonical `(path, outer step, phase, edge)` transition IDs, a local refinement block, and a frozen namespace, so chunking and resume cannot shift later randomness. Historical support replay injects the parent run's recorded first version-1 word; rows that consumed more bits verify and reconstruct their exact historical transition-local continuation rather than claiming that the new stream reproduces them. The certificate stage replays all 294 parent rows and independently recertifies a fresh 512-transition four-color/half-full-duration panel. The kernel stage runs a 4,096-transition warm-up, three 65,536-transition probes, and, only after those pass, three complete state-dependent seven-phase paths of 1,404,928 transitions each. CUDA launches contain at most 4,096 transitions and benchmark shards commit every eight outer steps.

The frozen performance gates remain certificate fraction `1`, Arb fallback fraction `<=1e-4`, Arb time fraction `<=0.10`, slowest rate `>=1300` transitions/s, projected 89,915,392-transition time `<=20` hours, peak GPU memory `<=80%`, and identical full-path hashes. Only a passing kernel stage unlocks the existing tower, legacy-mixture, bounded-teacher, stationary-null, orientation, scale, and model-input controls through the new API. The final decision `exact_jacobi_rb_cuda_kernel_and_target_feasible` authorizes only a separate state-dependent Strang-refinement patch.

Run the stages in order:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_cuda_confirmation `
  --runs-root runs/experiment12_d0_jacobi_rb_cuda_confirmation `
  --run-name production-certified-cuda-dd-jacobi-rb `
  --device cuda `
  --stage preflight `
  --parent-rb-kernel-run-dir runs/experiment12_d0_jacobi_rb_denoising_feasibility/20260722-203339_production-certified-spectral-rb-kernel-retry `
  --require-gate preflight

$cudaRbRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_cuda_confirmation" -Directory |
  Where-Object Name -Like "*_production-certified-cuda-dd-jacobi-rb" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_cuda_confirmation `
  --device cuda --stage certificate --resume-run-dir $cudaRbRun `
  --parent-rb-kernel-run-dir runs/experiment12_d0_jacobi_rb_denoising_feasibility/20260722-203339_production-certified-spectral-rb-kernel-retry `
  --require-gate certificate

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_cuda_confirmation `
  --device cuda --stage kernel --resume-run-dir $cudaRbRun `
  --parent-rb-kernel-run-dir runs/experiment12_d0_jacobi_rb_denoising_feasibility/20260722-203339_production-certified-spectral-rb-kernel-retry `
  --require-gate kernel

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_cuda_confirmation `
  --device cuda --stage target --resume-run-dir $cudaRbRun `
  --parent-rb-kernel-run-dir runs/experiment12_d0_jacobi_rb_denoising_feasibility/20260722-203339_production-certified-spectral-rb-kernel-retry `
  --require-gate target
```

This remains a fixed-grid `alpha=1`, controls-only claim. It neither establishes spatial Dirichlet--Ferguson refinement nor authorizes model training, reconstruction, or sampling.

##### Exact multi-path Jacobi Rao--Blackwell CUDA scheduling and target confirmation

The immutable fused-CUDA run `runs/experiment12_d0_jacobi_rb_cuda_confirmation/20260722-233246_codex-final-validation-certified-cuda-dd-jacobi-rb` has a 219-record terminal registry with SHA-256 `74a538caa33fbc5ef28e76e7feeedc77287fc0af36b8679c59e241ca3e43a757` and source fingerprint `c6ab156d467cbf17bd804c1a204e91e11be637b23ba4aa15bd066994e8bba52f`. Its transition mathematics is accepted: preflight and all 806 certificate cases passed, every full-path transition was certified, conservation and replay hashes passed, and there were no fallbacks, cap hits, approximations, nonfinite values, floors, limiters, corrections, or renormalizations. The only failed checks were resource checks: the single-path stateful benchmark ran at `643.1408275980631 < 1300` transitions/s and projected `38.835192396442096 > 20` hours for the 64-path cache. The target stage was consequently not evaluated. This result is re-adjudicated as `single_path_scheduling_resource_infeasible`, without changing the parent record or any of its seven fingerprinted implementation files.

The additive entry point is `mnist.diag_d0_jacobi_rb_cuda_multipath_confirmation`. It changes only launch packing. A single path exposes 392 independent edges in a color phase, whereas the already passing probe showed that the unchanged certified CUDA transition is efficient near a 4,096-lane launch. Independent paths are therefore advanced at the same outer step and phase in the frozen production groups

`[10,10,10,10,10,10,4]`,

which produce phase launches of `10*392=3920` and `4*392=1568` transitions. Within every path the 512 outer steps and palindromic seven-phase order remain serial. For phase `p` at outer step `k`, the scheduler gathers pair states as `[P,392]`, flattens them path-major, calls the unchanged certified API once, reshapes the result, and scatters each update back only to its source path. Canonical identifiers are constructed from `(path, outer_step, phase, edge)`, so Philox randomness depends on path identity rather than cohort position. Permuting, regrouping, chunking, or resuming paths therefore cannot change a path's realized transition.

The preflight requires exact phase-by-phase equality between `P=1` and batched `P=4/P=10` execution for fresh eight-step panels. It also embeds immutable parent path 0 in a ten-path cohort and requires its output and final-state hashes to match. Canonical-ID uniqueness over `64*512*7*392` transitions, the 4,096 launch cap, path isolation, permutation/regrouping invariance, and eight-step resume equivalence are all fail-closed checks.

The scheduling pilot uses separate path identifiers, 64 evolving outer steps, group sizes 10 and 4, three repeats, and atomic eight-step shards. Both group sizes must sustain at least 1,300 transitions/s. Its conservative projection is

`t_projected = 8 * (6*max_repeat(t_10) + max_repeat(t_4))`,

and must satisfy both `t_projected <= 20 hours` and `89,915,392/t_projected >= 1300` transitions/s. A pilot failure stops before the full gate and authorizes only an exact CUDA-graph or persistent-launch scheduling patch.

The full kernel gate uses fresh states and runs three complete `K=512` repeats at each group size: 14,049,280 transitions per `P=10` repeat and 5,619,712 per `P=4` repeat, for 59,006,976 executed transitions. It preserves the certificate-fraction-one, fallback, memory, conservation, numerical-law, and identical-hash requirements. The exact frozen production projection is

`t_64 = 6*max_repeat(t_10) + max_repeat(t_4)`.

It must be at most 20 hours and imply at least 1,300 effective transitions/s. Only a passing kernel unlocks the unchanged Rao--Blackwell tower, legacy-mixture, bounded-teacher, stationary-null, orientation, pair-mass, `h^-2`, invariant-Beta-score, flux-sign, and later-state-only input controls. No model training, Strang-refinement experiment, or reverse sampling occurs in this workflow.

Run preflight and resolve the generated directory:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_cuda_multipath_confirmation `
  --runs-root runs/experiment12_d0_jacobi_rb_cuda_multipath_confirmation `
  --run-name production-multipath-jacobi-rb `
  --device cuda `
  --stage preflight `
  --parent-cuda-run-dir runs/experiment12_d0_jacobi_rb_cuda_confirmation/20260722-233246_codex-final-validation-certified-cuda-dd-jacobi-rb `
  --require-gate preflight

$multiRbRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_cuda_multipath_confirmation" -Directory |
  Where-Object Name -Like "*_production-multipath-jacobi-rb" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName
```

Run the pilot, then the expensive kernel only after the pilot passes:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_cuda_multipath_confirmation `
  --device cuda --stage pilot --resume-run-dir $multiRbRun `
  --parent-cuda-run-dir runs/experiment12_d0_jacobi_rb_cuda_confirmation/20260722-233246_codex-final-validation-certified-cuda-dd-jacobi-rb `
  --require-gate pilot

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_cuda_multipath_confirmation `
  --device cuda --stage kernel --resume-run-dir $multiRbRun `
  --parent-cuda-run-dir runs/experiment12_d0_jacobi_rb_cuda_confirmation/20260722-233246_codex-final-validation-certified-cuda-dd-jacobi-rb `
  --require-gate kernel
```

Only after the kernel passes, run the target gate:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_cuda_multipath_confirmation `
  --device cuda --stage target --resume-run-dir $multiRbRun `
  --parent-cuda-run-dir runs/experiment12_d0_jacobi_rb_cuda_confirmation/20260722-233246_codex-final-validation-certified-cuda-dd-jacobi-rb `
  --require-gate target
```

The closed decisions distinguish provenance/runtime/equivalence failures, pilot or full resource failures, numerical failures, target failures, and `exact_jacobi_rb_multipath_kernel_and_target_feasible`. Only that final decision authorizes planning the state-dependent Strang-refinement experiment. It still does not authorize neural training, reconstruction, reverse sampling, a known-prior claim, or spatial Dirichlet--Ferguson convergence.

##### Exact state-dependent Jacobi Strang-refinement gate

The completed run `runs/experiment12_d0_jacobi_rb_cuda_multipath_confirmation/20260723-092105_production-multipath-jacobi-rb` passed both its exact kernel and Rao--Blackwell target gates. Its immutable 891-record registry has SHA-256 `b1724cb1222baf315b3aff24858ac6d979a2ed36e0331995245220a5861545f5`. The new workflow preserves all 11 source files bound by that run and adds a separate variable-`K` refinement scheduler.

For `K in {128,256,512,1024,2048}`, every phase recomputes the current pair mass and uses the exact exposure

`u = 3*(tau_eff/K)*phase_duration/(h*h*pair_mass)`.

The physical horizon remains `tau_eff=5e-5`, the grid remains 28, and the certified Jacobi transition remains unchanged. `K=2048` is reference-only. Levels share exact Philox uniforms at aligned right endpoints on a finest `K=2048` clock; this is common-random-number variance reduction and does not alter any level's marginal law.

The nonstationary panel starts from the first label-3 MNIST image with `lambda_mix=0.35` and source SHA-256 `0bb39fec59853f789fe366251cd85ed79ffbffb5a1aaa32084d2dbd2bbb4ea7d`. It records eight low Fourier modes, `sum(s**2)`, and `sum(s**3)` at quarter-horizon checkpoints. The quadratic and cubic quantities are centered and normalized using exact Dirichlet(1) moments.

The authorizing refinement evidence requires:

- weak order at least `1.8` for the linear, quadratic, cubic, and pooled families;
- a 90% path-bootstrap interval containing order two with lower endpoint at least `1.5`;
- simultaneous 99% `K=512` versus `K=1024` discrepancy at most `0.005`;
- simultaneous 99% `K=512` versus the Richardson reference `(4*mu_2048-mu_1024)/3` at most `0.01`;
- agreement of the high- and low-resolution Richardson extrapolates within `0.005`;
- exact certification, conservation, and zero approximate/intervention mechanisms.

A disjoint pilot uses variances and timings only to freeze 32 or 64 production paths and a 16- or 32-path `K=2048` subset. It chooses the cheapest design meeting the predeclared confidence-width targets and a 48-hour projected production cap. Pilot means do not enter the final result.

The target run contained one nonauthorizing 16-path stationarity miss. The new preflight explicitly rechecks it on two fixed 128-path panels with a joint 99% max-T family over Legendre degrees 1--8 and reversibility. It also tests exact Dirichlet stationarity and antisymmetric detailed-balance witnesses for full seven-phase sweeps at every refinement level.

Run preflight:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_strang_refinement `
  --runs-root runs/experiment12_d0_jacobi_rb_strang_refinement `
  --run-name production-state-dependent-strang-refinement `
  --device cuda `
  --stage preflight `
  --parent-multipath-run-dir runs/experiment12_d0_jacobi_rb_cuda_multipath_confirmation/20260723-092105_production-multipath-jacobi-rb `
  --require-gate preflight
```

Resolve the generated directory:

```powershell
$strangRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_strang_refinement" -Directory |
  Where-Object Name -Like "*_production-state-dependent-strang-refinement" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName
```

Run the variance-only power stage:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_strang_refinement `
  --device cuda `
  --stage power `
  --resume-run-dir $strangRun `
  --parent-multipath-run-dir runs/experiment12_d0_jacobi_rb_cuda_multipath_confirmation/20260723-092105_production-multipath-jacobi-rb `
  --require-gate power
```

Only after the power gate passes:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_strang_refinement `
  --device cuda `
  --stage refinement `
  --resume-run-dir $strangRun `
  --parent-multipath-run-dir runs/experiment12_d0_jacobi_rb_cuda_multipath_confirmation/20260723-092105_production-multipath-jacobi-rb `
  --require-gate refinement
```

Only `exact_state_dependent_strang_refinement_feasible` authorizes a separate one-image phase-conditioned MSE experiment on the exact Rao--Blackwell target. This workflow performs no training or reverse sampling.

##### Exact Dynkin/Rao--Blackwell Strang power confirmation

The completed run `runs/experiment12_d0_jacobi_rb_strang_refinement/20260723-230629_production-state-dependent-strang-refinement` stopped at `refinement_power_infeasible`. Its immutable 1,308-record registry has SHA-256 `734c93e1e7d0be29041e1d567b36cbd8ea7aac50df7996d5f8c41fbddef8e632`. The run was numerically healthy: all 112,394,240 transitions were certified, all intervention/approximation counts were zero, conservation passed, and the sustained complete-panel rate was about 2,725 transitions per second. Three candidate designs fit the 48-hour resource budget. The failure was that raw terminal-observable fluctuations made the projected simultaneous half-widths many orders of magnitude larger than `0.0025/0.005`.

The additive workflow `mnist.diag_d0_jacobi_rb_dynkin_power_confirmation` keeps every transition, transition identifier, Rao--Blackwell denoising label, and scientific threshold unchanged. It changes only the unbiased estimator used for the weak-refinement diagnostic. For each phase transition with current state \(S_j\) and observable \(f\), it records the exact conditional drift

```text
delta_j = P_phase f(S_j) - f(S_j)
```

and reports

```text
A_K(t) = f(S_0) + sum_{j<t} delta_j.
```

The raw endpoint decomposes as `f(S_t) = A_K(t) + sum_j M_j`, where each `M_j` is a conditional mean-zero martingale increment. Consequently `A_K` has exactly the same expectation as the raw endpoint while removing transition noise. The coefficient is fixed at one; no empirical fit, learned baseline, future state, approximate transition, or modified training target enters the calculation.

For pair mass `r`, head fraction `x`, `z=2*x-1`, and `P2=(3*z*z-1)/2`, the authorizing phase drifts are

```text
delta Fourier = (w_head-w_tail)*r*z*expm1(-2*u)/2
delta Q       = r*r*P2*expm1(-6*u)/3
delta C       = r*r*r*P2*expm1(-6*u)/2
```

summed over the current perfect matching before its exact Jacobi transition. Zero pair mass or zero duration contributes exactly zero. A fused sidecar evaluates these expressions with outward numerical-error bounds and deterministic compensated accumulation. Raw checkpoint observables remain available as advisory evidence, and enabling the sidecar must reproduce the parent transition and state hashes exactly.

Preflight verifies the phase moments against spectral/Arb evaluation, exercises both phase durations and every colour, and runs two immutable 128-cluster tower-identity panels. It also records why a useful distribution-free Hoeffding power guarantee is impossible at this budget: the standardized cubic support would require on the order of \(2.4\times10^{18}\) paths at the frozen tolerance. The subsequent small-panel calculation is therefore an explicitly labelled engineering forecast, never the final scientific interval.

The sealed pilot freezes independent eight-path A and B panels at `K=128,256,512,1024,2048`. Panel A nominates one of the unchanged `32/16`, `32/32`, `64/16`, or `64/32` main/reference designs. Panel B opens once and must confirm that exact nominee; the combined 16-path calculation must also qualify. The existing normal/chi-square/Bonferroni planning construction, `0.0025/0.005` widths, 1,300-transition/s floor, and 48-hour cap remain unchanged. Pilot means never enter the future refinement estimate.

Run preflight:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_dynkin_power_confirmation `
  --runs-root runs/experiment12_d0_jacobi_rb_dynkin_power_confirmation `
  --run-name production-dynkin-strang-power `
  --device cuda `
  --stage preflight `
  --parent-strang-run-dir runs/experiment12_d0_jacobi_rb_strang_refinement/20260723-230629_production-state-dependent-strang-refinement `
  --require-gate preflight
```

Resolve the new run and execute the sealed pilot:

```powershell
$dynkinRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_dynkin_power_confirmation" -Directory |
  Where-Object Name -Like "*_production-dynkin-strang-power" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_dynkin_power_confirmation `
  --device cuda `
  --stage pilot `
  --resume-run-dir $dynkinRun `
  --parent-strang-run-dir runs/experiment12_d0_jacobi_rb_strang_refinement/20260723-230629_production-state-dependent-strang-refinement `
  --require-gate pilot
```

##### Exact phase-local Dynkin observer repair

The fresh namespace-corrected run
`runs/experiment12_d0_jacobi_rb_dynkin_power_confirmation/20260724-184842_production-dynkin-strang-power-idfix`
ended at `dynkin_estimator_numerically_unresolved`. Its immutable 18-record
registry has SHA-256
`57daf61e9686c2257d6579c84130e5e4b4a400a8435916a62ac408c87ad6072d`.
This was not another namespace, transition, or Dynkin-moment failure. All
eleven path-ID checks passed, the legacy K=512 transition/certificate/state
hashes were byte-identical, and the largest float64 and CUDA phase-moment
errors were approximately `3.98e-15` and `8.67e-19`.

Tower panel A stopped during max-T inference with `nonzero degenerate
whole-path statistic`. Horizontal matchings analytically preserve the four
y-Fourier observables, while vertical matchings preserve the four x-Fourier
observables. The old observer evaluated each invariant as two independent
global reductions, `f(state_after)-f(state_before)`, leaving nonzero roundoff
of order `1e-14` with standard error below the frozen `1e-15` degeneracy
floor. The strict max-T implementation correctly failed closed. Panel A was
not sealed, panel B did not run, and the sealed power pilot remains
unauthorized.

The additive workflow
`mnist.diag_d0_jacobi_rb_dynkin_phase_observer_confirmation` changes only
the numerical representation of the realized one-phase observable
increment. For pair total `r`, earlier and later head fractions `x,y`,
`d=y-x`, `c=x+y-1`, it evaluates

```text
delta Fourier_m = sum_edges (w_head-w_tail) * r * d
delta Q         = 2 * sum_edges r^2 * d * c
delta C         = 3 * sum_edges r^3 * d * c
```

before the matching update is scattered. These expressions are algebraically
identical to the global before/after differences under pair conservation.
Endpoint-weight equality makes cross-coordinate Fourier increments bitwise
zero; no tolerance-based snapping is used. The sampler's certified quantile
cell, deterministic ball arithmetic, and the unchanged analytic Dynkin-drift
radius produce the authorizing residual enclosure. Global reductions remain
an advisory roundoff forensic. The generic max-T degeneracy guard and every
scientific threshold remain unchanged.

Fresh controls use root seed `261171`, tower bases `0x60000/0x70000`, pilot
bases `0x80000/0x90000`, and retain `[0xF0000,0x100000)` for future
production paths. Two fresh 128-path tower panels must pass the complete
80-member 99% max-T family before the original sealed A/B power pilot may
run.

Run a fresh phase-observer preflight:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_dynkin_phase_observer_confirmation `
  --runs-root runs/experiment12_d0_jacobi_rb_dynkin_phase_observer_confirmation `
  --run-name production-dynkin-strang-power-phase-observer-fix `
  --device cuda `
  --stage preflight `
  --parent-dynkin-idfix-run-dir runs/experiment12_d0_jacobi_rb_dynkin_power_confirmation/20260724-184842_production-dynkin-strang-power-idfix `
  --require-gate preflight
```

Only after that preflight passes:

```powershell
$observerRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_dynkin_phase_observer_confirmation" -Directory |
  Where-Object Name -Like "*_production-dynkin-strang-power-phase-observer-fix" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_dynkin_phase_observer_confirmation `
  --device cuda `
  --stage pilot `
  --resume-run-dir $observerRun `
  --parent-dynkin-idfix-run-dir runs/experiment12_d0_jacobi_rb_dynkin_power_confirmation/20260724-184842_production-dynkin-strang-power-idfix `
  --require-gate pilot
```

Only `exact_dynkin_refinement_estimator_feasible` authorizes planning a fresh
production refinement run. This observer repair performs no model training
or reverse sampling.

##### Dynkin canonical-ID namespace repair

The first production preflight,
`runs/experiment12_d0_jacobi_rb_dynkin_power_confirmation/20260724-164645_production-dynkin-strang-power`,
ended fail-closed before either tower panel ran. This is an implementation
failure, not a scientific Dynkin result. Parent provenance, the exact legacy
`K=512` observer replay, and the phase-moment oracle all passed; the largest
CUDA phase-moment error was about `8.67e-19`. Execution then raised
`ValueError: path IDs must fit the frozen 20-bit field`.

The original Dynkin CLI assigned tower IDs around five to six million and
pilot IDs around 4.1 to 4.2 million, although the immutable refinement
scheduler requires `0 <= path_id < 2^20`. The 128-path tower also attempted to
call a canonical-ID helper whose cohort contract is at most eight paths. The
recorded top-level decision `control_provenance_invalid` is therefore a
historical classification defect: the saved `parent_provenance.json` passed.
The failed directory remains immutable and must not be resumed.

The correction freezes a versioned `path_id_plan.json` before GPU work:

- legacy replay uses `20,000..20,007`;
- tower A and B use bases `0x20000` and `0x30000`, with eight 128-ID cases;
- pilot A and B use `0x40000..0x40007` and `0x50000..0x50007`;
- `[0xF0000,0x100000)` is reserved for the later production refinement.

Tower IDs are assembled by concatenating sixteen calls to the unchanged
eight-path canonical helper in path-major order. The plan, derived panels,
tower records, shard fingerprints, manifest, and resume contract bind the
same semantic hash. Operational failures now use
`evaluation_status=execution_failed`; scheduler/configuration failures map to
`refinement_scheduler_invalid` and cannot masquerade as provenance failures.
The Jacobi law, Dynkin estimator, root seed, panel sizes, thresholds, and
scientific claim remain unchanged.

Run a fresh preflight:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_dynkin_power_confirmation `
  --runs-root runs/experiment12_d0_jacobi_rb_dynkin_power_confirmation `
  --run-name production-dynkin-strang-power-idfix `
  --device cuda `
  --stage preflight `
  --parent-strang-run-dir runs/experiment12_d0_jacobi_rb_strang_refinement/20260723-230629_production-state-dependent-strang-refinement `
  --require-gate preflight
```

Only after that fresh preflight passes:

```powershell
$dynkinRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_dynkin_power_confirmation" -Directory |
  Where-Object Name -Like "*_production-dynkin-strang-power-idfix" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_dynkin_power_confirmation `
  --device cuda `
  --stage pilot `
  --resume-run-dir $dynkinRun `
  --parent-strang-run-dir runs/experiment12_d0_jacobi_rb_strang_refinement/20260723-230629_production-state-dependent-strang-refinement `
  --require-gate pilot
```

##### Saved phase-observer pilot result: exact tower, insufficient coupling

The completed run
`runs/experiment12_d0_jacobi_rb_dynkin_phase_observer_confirmation/20260724-225759_production-dynkin-strang-power-phase-observer-fix`
ended `dynkin_power_infeasible`. Its immutable 1,061-record registry has
SHA-256
`0cbe181966a79698b9e8d8177b86f3038cd8c1c619c3af384168fbc61f7a11ac`.
The phase-local observer repair was successful: both independent 128-path
tower panels passed, all 87,105,536 pilot transitions were certified, mass
conservation and shard chains passed, and the complete-panel rate was about
2,870 transitions per second. Three candidate production budgets remained
below 48 hours.

The failure was statistical power under the right-endpoint common-quantile
coupling. The best projected Dynkin half-widths were approximately `130.20`
for the main family and `206.83` for the reference family, versus frozen
limits `0.0025` and `0.005`. Panel A nominated no design and panel B correctly
remained sealed. The top-level infinite selected-design values are fail-closed
sentinels, not numerical infinities.

The Dynkin estimator is unbiased, but its predictable part need not have
lower variance than the endpoint: it can be negatively correlated with the
omitted martingale. Here it reduced variance for 66 of 160 features and
increased it for 94; quadratic and cubic mass observables dominated. Raw
endpoint widths were smaller (`22.92/18.93`) but still outside the gates.
Future refinement gates therefore use raw endpoint differences as the
authorizing weak observable and retain Dynkin values as exact advisory tower
diagnostics.

##### Certified Haar-coupled Strang power gate

The additive workflow
`mnist.diag_d0_jacobi_rb_hierarchical_coupling_confirmation` changes only the
joint coupling between temporal levels. For independent normal root/detail
fields,

```text
xi[2K,2k]   = (xi[K,k] + eta[K,k]) / sqrt(2)
xi[2K,2k+1] = (xi[K,k] - eta[K,k]) / sqrt(2)
U[K,k]      = Phi(xi[K,k]).
```

Every level receives independent uniform transition inputs with the correct
marginal law, while adjacent levels share their Brownian-scale innovation.
The Gaussian tree is a copula only: every state update and Rao--Blackwell
target still comes from the exact certified Jacobi inverse CDF. The normal
transforms and derived uniform intervals are certified before applying the
existing strict Jacobi CDF-cell and `G/k` target-cell tests. Ordinary CUDA
normal functions are proposal-only.

The production implementation is fused and fail-closed. It starts from
stateless 128-bit Philox dyadic source cells, uses CUDA `normcdfinv` only to
propose a root bracket, and authorizes the bracket with double-double balls.
The certified normal CDF is expanded around 305 Arb-generated quarter-grid
anchors on `[-38,38]`; its local degree-56 recurrence uses a frozen
`2^-120` remainder bound. The certified normal intervals are combined by the
orthogonal Haar recursion and mapped through the same certified CDF. The
resulting arbitrary uniform ball is passed directly to the existing Jacobi
CDF-cell and conormal `G/k` authorizer; it is never rounded to an ordinary
uniform before authorization. Ambiguity refines only the responsible source
prefix, up to 1,024 bits, and then uses transition-local Arb fallback. Source,
compile-option, CUBIN, constant-table, and runtime hashes are recorded.

The first profile is one nested Haar arm. It uses independent main
`K=128,256,512,1024` and reference `K=512,1024,2048` pools and retains the
`32/16`, `32/32`, `64/16`, and `64/32` candidate grid. If its sealed panel A
nominates nothing, a predeclared pairwise antithetic profile receives one
panel-A attempt: each coarse path is paired with two exact fine paths obtained
by reversing the Haar detail signs, and its only production candidate is
`16/16`. Once either profile opens panel B, no profile fallback is permitted.

The authorizing corrections are

```text
D1 = mu128  - mu256
D2 = mu256  - mu512
D3 = mu512  - mu1024
D4 = mu1024 - mu2048
mu512 - mu_star       = D3 + (4/3) D4
mu_star - mu_star_low = (D3 - 4 D4) / 3.
```

Main and reference pools are independent and their planning variances are
combined accordingly. Both sealed eight-cluster panels and their combined
analysis must satisfy the unchanged `0.0025/0.005` widths, 48-hour cap,
1,300-transition-per-second floor, certification, conservation, and forbidden
event gates. A pass authorizes only a fresh production refinement experiment.
The root seed is frozen at `261181`; single-arm, antithetic, marginal-control,
and future-production path-ID slots are disjoint. The closed outcomes are
`control_provenance_invalid`, `hierarchical_rng_algebra_invalid`,
`certified_normal_transform_invalid`,
`arbitrary_uniform_jacobi_certificate_invalid`,
`hierarchical_marginal_law_invalid`, `hierarchical_scheduler_invalid`,
`hierarchical_coupling_computationally_infeasible`,
`hierarchical_power_infeasible`, `hierarchical_panels_disagree`, and
`exact_haar_hierarchical_refinement_coupling_feasible`.

##### Saved first Haar preflight: scheduler-adapter shape failure

The immutable run
`runs/experiment12_d0_jacobi_rb_hierarchical_coupling_confirmation/20260725-171002_production-certified-haar-strang-power`
contains a 17-record terminal registry with SHA-256
`8d56d08d1658651115bc932698aa42608f0953a09354b3e0c44ba5fbfb9f5db3`.
It stopped during preflight because the scheduler flattened the transition
state from `[P,392]` to `[P*392]` without flattening the corresponding
double-double uniform certificate and prefix tensors. The tensors themselves
were valid; their incompatible adapter shape caused the authorizer's
`source_prefix_bits` contract check to fail.

This is a scheduler-adapter execution failure, not evidence against the Haar
algebra, certified normal transform, Jacobi law, Rao--Blackwell target, or
scientific power gate. The saved top-level
`hierarchical_rng_algebra_invalid` decision is therefore preserved but
re-adjudicated as `hierarchical_scheduler_invalid`, with scientific evidence
incomplete. The failed directory must remain immutable and must not be
resumed. The repair changes only certificate-tensor normalization at the
scheduler boundary; no theory, transition, randomness, threshold, path count,
or power-design change is authorized.

##### Saved second Haar preflight: follow-on diagnostic-contract failures

The immutable adapter-repair acceptance run
`runs/experiment12_d0_jacobi_rb_hierarchical_coupling_confirmation/20260725-180014_production-certified-haar-strang-power-adapter-fix`
contains a 20-record terminal registry with SHA-256
`9dba0e804d928c573414503e995f1282d7db799016c8187f145bd67ef7b82337`.
The scheduler shape repair executed successfully: all `2,634,240`
transitions were certified, sustained throughput was `4300.162`
transitions per second, all forbidden-event counts were zero, and only two
Arb fallbacks occurred (fraction `7.59e-7`). The measured complete-pipeline
projection was `21.055` hours, safely inside the frozen 48-hour cap.

Preflight nevertheless failed on two follow-on implementation diagnostics.
First, the observer batching check required bitwise equality across equivalent
GPU reductions whose reduction ordering differs with batching. Second, one
summary record omitted a redundant `uncertified_count` field even though its
certificate count and fraction proved that every transition was certified.
Neither failure is evidence against the Haar coupling, Jacobi transition,
Rao--Blackwell target, numerical certification, conservation, or resource
feasibility. The bitwise batching assertion and redundant-counter schema have
now both been repaired. This 20-record run remains immutable and must not be
resumed; a fresh `adapter-fix-v2` preflight is required.

Run the fresh v2 preflight:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_hierarchical_coupling_confirmation `
  --runs-root runs/experiment12_d0_jacobi_rb_hierarchical_coupling_confirmation `
  --run-name production-certified-haar-strang-power-adapter-fix-v2 `
  --device cuda `
  --stage preflight `
  --parent-phase-observer-run-dir runs/experiment12_d0_jacobi_rb_dynkin_phase_observer_confirmation/20260724-225759_production-dynkin-strang-power-phase-observer-fix `
  --require-gate preflight
```

##### Saved third Haar preflight: adapter and diagnostic repairs accepted

The fresh immutable run
`runs/experiment12_d0_jacobi_rb_hierarchical_coupling_confirmation/20260725-212650_production-certified-haar-strang-power-adapter-fix-v2`
passed the required preflight gate. Its 20-record terminal registry has
SHA-256
`762695fd027083a5d9aa8f6a26411303280e8c3e0e21ae4571483fd008aa7b1f`.
`run_status.json` records `status=complete`, `outcome=complete`, and
`required_gate_pass=1`; `haar_preflight_gate.json` records `passed=1`.
All named preflight subchecks pass, including deterministic batching,
order/chunk/resume invariance, exact certification, Haar/Jacobi controls, and
the later-state-only target contract.

The complete scheduler shard executed `2,634,240` transitions at
`4315.066` transitions per second. Certificate fraction was exactly one,
mass error was `4.44e-16`, fallback count was two, fallback fraction was
`7.59e-7`, fallback-time fraction was `5.04e-4`, and every forbidden-event
count was zero. The measured nested `32/16` production projection was
`20.982` hours, below the frozen 48-hour limit.

The closed whole-workflow `decision` field still carries
`hierarchical_scheduler_invalid` while the coupling and pilot components are
`not_evaluated`. That legacy pending-stage label is not a failed preflight:
the required-gate result above is authoritative, and
`coupling_stage_authorized=1`. Resume this exact directory for coupling; do
not start another preflight.

Resolve the run, then execute the coupling and sealed pilot stages:

```powershell
$haarRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_hierarchical_coupling_confirmation" -Directory |
  Where-Object Name -Like "*_production-certified-haar-strang-power-adapter-fix-v2" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_hierarchical_coupling_confirmation `
  --device cuda --stage coupling --resume-run-dir $haarRun `
  --parent-phase-observer-run-dir runs/experiment12_d0_jacobi_rb_dynkin_phase_observer_confirmation/20260724-225759_production-dynkin-strang-power-phase-observer-fix `
  --require-gate coupling

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_hierarchical_coupling_confirmation `
  --device cuda --stage pilot --resume-run-dir $haarRun `
  --parent-phase-observer-run-dir runs/experiment12_d0_jacobi_rb_dynkin_phase_observer_confirmation/20260724-225759_production-dynkin-strang-power-phase-observer-fix `
  --require-gate pilot
```

Only `exact_haar_hierarchical_refinement_coupling_feasible` authorizes the
fresh production Strang-refinement patch. It does not authorize neural
training or reverse sampling.

### Phase 5 — reconnecting to the preference-function program (after D0 works)

D0 deliberately learns the data-density h-transform first. The original goal — conditioning on a *preference* g — is then layered on top, where it becomes tractable because the learned generator reaches digit-like states:

- **Guidance**: add the edge flux (2/h) theta_e d^h_e log g(s)-style term (or its Brownian-shift form) to the learned reverse dynamics at sampling time — the conservative-flux analogue of classifier guidance; honest if reported as approximate h-transform of the learned process.
- **C1 revived**: use the D0 generator as the proposal for Girsanov-corrected terminal tilting; ESS will no longer collapse because the proposal mass actually overlaps the conditioning event.
- **C3 revived, on-policy**: branch/value estimation under the learned generator's state distribution, with the derivative-regularized value losses already sketched in the Exp. 11 report.

This ordering also gives the advisor report a clean narrative: the terminal-reward estimators were sound but were applied on a reference measure that gives the conditioning event probability ~0; D0 changes the measure, not the theory.

#### Immutable Haar panel-A recovery and antithetic continuation

The completed run
`runs/experiment12_d0_jacobi_rb_hierarchical_coupling_confirmation/20260725-212650_production-certified-haar-strang-power-adapter-fix-v2`
is now frozen with 197 registered artifacts and registry SHA-256
`4bf1dab4c0905533fe0df885521fb3309ed6344e13f1fd67faad7fa9ae11abfe`.
Its source fingerprint is
`300bcdab17d9cac5605311bf0b513a5c476e88011662fb1e51ac69ca4f431c39`,
and its scientific-configuration SHA-256 is
`cb26ea614d20f7695b02fa063aafca41d0a229ff34a26e9bbb0610bdda1352cf`.
It must not be resumed.

All 16 nested-main and 64 nested-reference shards completed. They contain
`120,823,808` certified transitions, 38 Arb fallbacks (fraction
`3.1450755e-7`), fallback-time fraction `5.989668e-4`, maximum mass error
`1.3323e-15`, no forbidden events, and a conservative rate of approximately
`4,202.43` transitions per second. Aggregation then read `record.schedule`
instead of the frozen canonical `record.identity.schedule`. The corrected
adjudication is `panel_schedule_binding_invalid`, not a Haar, Jacobi, target,
numerical, or resource failure.

The additive workflow
`mnist.diag_d0_jacobi_rb_haar_power_recovery_confirmation` verifies all
parent artifacts and restart chains and replays the 80 archives without GPU
recomputation or parent mutation. It exactly recovers four numerically and
resource-valid but power-ineligible nested candidates:

| Main/reference paths | Main width | Generator width | Stability width | Projected hours |
| --- | ---: | ---: | ---: | ---: |
| 32/16 | 6.83809 | 9.44036 | 7.77044 | 21.5446 |
| 32/32 | 6.83809 | 7.79274 | 5.65563 | 31.9455 |
| 64/16 | 4.83526 | 8.54130 | 7.65398 | 32.6884 |
| 64/32 | 4.83526 | 6.67534 | 5.49453 | 43.0893 |

The recovered nested result is therefore
`panel_a_no_eligible_design`. This authorizes only the untouched
pairwise-antithetic panel A. Panel B opens exactly once only if A nominates
the frozen `16/16` design. An A failure ends `hierarchical_power_infeasible`;
a B or combined-analysis failure ends `hierarchical_panels_disagree`. Only
`exact_haar_hierarchical_refinement_coupling_feasible` authorizes a fresh
production refinement patch.

Run preflight:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_haar_power_recovery_confirmation `
  --runs-root runs/experiment12_d0_jacobi_rb_haar_power_recovery_confirmation `
  --run-name production-haar-power-recovery `
  --device cuda `
  --stage preflight `
  --parent-haar-run-dir runs/experiment12_d0_jacobi_rb_hierarchical_coupling_confirmation/20260725-212650_production-certified-haar-strang-power-adapter-fix-v2 `
  --require-gate preflight
```

Resolve the generated directory and replay the immutable panel:

```powershell
$recoveryRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_haar_power_recovery_confirmation" -Directory |
  Where-Object Name -Like "*_production-haar-power-recovery" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_haar_power_recovery_confirmation `
  --device cuda `
  --stage replay `
  --resume-run-dir $recoveryRun `
  --parent-haar-run-dir runs/experiment12_d0_jacobi_rb_hierarchical_coupling_confirmation/20260725-212650_production-certified-haar-strang-power-adapter-fix-v2 `
  --require-gate replay
```

Only after replay passes:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_haar_power_recovery_confirmation `
  --device cuda `
  --stage pilot `
  --resume-run-dir $recoveryRun `
  --parent-haar-run-dir runs/experiment12_d0_jacobi_rb_hierarchical_coupling_confirmation/20260725-212650_production-certified-haar-strang-power-adapter-fix-v2 `
  --require-gate pilot
```

#### Fresh exact-K=512 physical coarse-signal witness

Before another neural run, use a model-free witness for the exact physical
Rao--Blackwell label. The allowed coarse variable is

\[
C=(\text{time quartile},\text{phase},\text{oriented edge}),
\qquad
\theta_C=\frac1{10976}\sum_c \mathbb E(\bar Z\mid C=c)^2.
\]

Because `C` is contained in the permitted future model inputs,
`theta_C > 0` proves that the full allowed-input conditional mean is not
identically zero. A non-detection does not prove the converse.

The workflow binds three immutable parents:

- the 544-record one-image run
  `20260729-015817_production-exact-k512-rb-one-image-learnability`,
  ending `no_detectable_one_image_conditional_signal`, registry SHA-256
  `5e0b46328b6783614bdb7d394587b32e63d2d33b76f0279abdab6ecdf7d4e18a`;
- the 18-record diagnostic
  `20260730-010919_production-sealed-rb-zero-signal-diagnostic`,
  concluding `frozen_model_does_not_beat_zero`, registry SHA-256
  `11d0a7272dd83b6535c1bc4426ad471f929ec0a1cd2f9c96e8ac80f01483a5e3`;
- the 74-record calibration
  `20260730-012459_production-noisy-jacobi-bayes-power`,
  ending `noisy_bayes_detection_pipeline_calibrated`, registry SHA-256
  `01b5d772299611e9e17b886658b7eba80a7ab50805241e94d2e9a8ba36562e79`.

No parent state, label, or prediction enters the authorizing estimate. The
old physical evidence is used only for a labelled power forecast. Preflight
must reproduce positive 99% lower bounds for every bounded-teacher Bayes
split pair and zero-containing 99% intervals for every stationary-null pair.
Teacher detection uses the preregistered one-sided 99% bounds; the null check
uses a genuine central 99% interval (0.5% and 99.5% endpoints).

Panels A and B are fresh independent 64-path exact `K=512` chains. They use
the first label-3 image, `lambda_mix=0.35`, 32 frozen outer steps, all seven
phases, and all 392 oriented edges. Each path stores only the mean of the
eight unmodified binary64 targets in each quartile/phase/edge cell, yielding
shape `[64,4,7,392]`. Selected labels are folded immediately into restartable
compensated accumulators; raw selected labels, earlier states, and predictions
are not persisted. The estimator is

\[
\widehat\theta_C
=\frac1{10976}\sum_c \overline Z_A(c)\overline Z_B(c).
\]

Independent panels remove the positive sampling-variance bias of a
within-panel square. Negative estimates and confidence limits are retained.
Inference is preregistered and independently implemented as:

1. 50,000 whole-path bootstrap replicates, resampling A and B independently
   with seed `261242`;
2. a delta/Welch bound from the two path-level influence components.

Both use one-sided 99% lower and upper bounds. The closed scientific
decisions are:

- `exact_physical_coarse_signal_detected` when both lower bounds are positive;
- `coarse_signal_below_preregistered_resolution` when neither method detects
  and both upper bounds are at most `5e-4`;
- `physical_coarse_signal_inconclusive` otherwise.

Panel A is sealed before panel B starts. Both panel hashes and the analysis
definition are frozen before joint analysis. The exact workload is
`179,830,784` certified transitions, with a 24-hour projected-runtime cap,
1,300-transition/s floor, certificate fraction one, the existing fallback
and memory limits, mass error at most `2e-12`, and zero approximate or
corrective mechanisms.

Run preflight:

```powershell
$physicalRun = "runs/experiment12_d0_jacobi_rb_one_image_learnability/20260729-015817_production-exact-k512-rb-one-image-learnability"
$zeroRun = "runs/experiment12_d0_jacobi_rb_zero_signal_diagnostic/20260730-010919_production-sealed-rb-zero-signal-diagnostic"
$bayesRun = "runs/experiment12_d0_jacobi_rb_bayes_power_calibration/20260730-012459_production-noisy-jacobi-bayes-power"

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_physical_coarse_signal_witness `
  --runs-root runs/experiment12_d0_jacobi_rb_physical_coarse_signal_witness `
  --run-name production-exact-k512-physical-coarse-signal `
  --device cuda `
  --stage preflight `
  --parent-one-image-run-dir $physicalRun `
  --parent-zero-signal-run-dir $zeroRun `
  --parent-bayes-power-run-dir $bayesRun `
  --require-gate preflight
```

Resolve the fresh run and execute the sealed panels in order:

```powershell
$witnessRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_physical_coarse_signal_witness" -Directory |
  Where-Object Name -Like "*_production-exact-k512-physical-coarse-signal" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_physical_coarse_signal_witness `
  --device cuda --stage panel-a --resume-run-dir $witnessRun `
  --parent-one-image-run-dir $physicalRun `
  --parent-zero-signal-run-dir $zeroRun `
  --parent-bayes-power-run-dir $bayesRun --require-gate panel-a

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_physical_coarse_signal_witness `
  --device cuda --stage panel-b --resume-run-dir $witnessRun `
  --parent-one-image-run-dir $physicalRun `
  --parent-zero-signal-run-dir $zeroRun `
  --parent-bayes-power-run-dir $bayesRun --require-gate panel-b

.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_physical_coarse_signal_witness `
  --device cuda --stage analyze --resume-run-dir $witnessRun `
  --parent-one-image-run-dir $physicalRun `
  --parent-zero-signal-run-dir $zeroRun `
  --parent-bayes-power-run-dir $bayesRun --require-gate witness
```

A positive witness authorizes only planning a coarse baseline plus an exact-RB
residual learner, still trained by unweighted MSE on the unchanged target.
A below-resolution result constrains only this coarse projection at `5e-4`.
This workflow performs no training or reverse sampling and makes no
reconstruction, known-prior, or spatial Dirichlet--Ferguson claim.

## 4. Risks and open items

- **Sign and time conventions**: the direct reverse transfer is added through the positive incidence convention, and schedule indices run `k+r-1, ..., k`. Test a ramped schedule; a constant schedule cannot expose an off-by-one error.
- **Conditional covariance**: the finite-step conditional covariance is state-dependent and nonidentity; it need not be pointwise smaller than I. Fresh N(0, I) is the continuous-time reference-noise prescription, not an exact Euler posterior covariance.
- **Block bias**: the strict baseline is `r = 1`. Any block-conditioned target and uniform redistribution over substeps is an additional approximation.
- **Limiter/floor bias**: these interventions change the transition kernel. Report them without conditioning the loss on a random no-intervention mask, and require their rates to vanish under refinement before making an h-transform claim.
- **Data law**: lambda flooring defines the represented images but does not make the empirical law absolutely continuous relative to nu_h. Specify smoothing or `t_min` for a literal density claim.
- **Grid scaling and prior**: fixed alpha is fixed-grid; fixed beta with `alpha_h = beta h^d` is required for the manuscript scaling. The class-conditional terminal bank remains an empirical p_T^y initializer until mixing and label-leakage gates justify a known prior.

## 5. Minimal checklist for the D0 patch (analogue of C0 Sec. 16)

1. Forward rollouts start at lambda-mixed data with learned control zero and symmetric scaling `w_free = w_sigma^2` for the literal baseline.
2. Realized applied physical transfers and raw Gaussian innovations are stored; interventions are reported and the direct loss uses all finite projected transfers.
3. Baseline slices use exact substeps, `r = 1`, and later state S_{k+1}; no terminal weights appear in the loss.
4. Form `U_k = -DeltaK_app,k - b_ref,k(S_{k+1}) * dt` using the same forward schedule interval `k`, then train on `Proj_inc(U_k) / c_U`; infer the global scale from the initial cache and freeze it run-wide.
5. Network sees (tau, state, label) only; output penalty `lambda_m = 0`.
6. Direct reverse sampling adds reference drift, learned residual, and fresh noise through positive incidence.
7. First runs use the matching empirical terminal bank without calling it an analytic prior.
8. Zero-residual stationarity and overfit-one-image tests pass before any full run.
9. Per-time-bin signal, covariance, intervention, mobility, and preview diagnostics are logged.
