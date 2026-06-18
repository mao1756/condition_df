# Experiment 12 / D0 patch plan: forward-from-data reverse innovation matching

**Status of inputs:** based on `experiment11_advisor_report.tex`, `experiment_c0_weighted_innovation.pdf`, `main.tex`, and `eulerian_approx.tex`. The codebase zip (`condition_df_copy202606121018.zip`) did not upload, so all file/flag names below are inferred from the C0 note and the Experiment 11 appendix commands (`mnist/experiment11_c0.py`, `mnist/experiment11_c3_value.py`) and must be checked against the actual code.

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

DDPM works because every training trajectory **starts at data**, so the regression target (conditional mean of the noise given the noised state) is O(1)-informative without importance weights. The exact analogue exists in the Eulerian setting, and it stays inside the h-transform framework because the free finite-volume process is **nu_h-symmetric** (the closed form (E_h, D(E_h)) is a symmetric Dirichlet form with nu_h-symmetric Hunt process; see the finite-volume manuscript, Sec. on the finite-volume Dirichlet form and the regularity/Hunt-process statement).

**Honest framing for the writeup.** Let p_0 be the (uniform-mixed) data distribution on Sigma_h and p_t the law of the free process started from p_0. By nu_h-symmetry, v_t := dp_t / dnu_h evolves by the *same* heat semigroup, v_t = T_h(t) v_0. The time reversal of the free process on [0, T] is then exactly a Doob h-transform of the free process with heat potential v_{T - tau}: in reversed time tau the Brownian edge shift is

    eta_rev,e(tau, s) = w_sigma * sqrt(2 theta_e(s)) * d^h_e log v_{T - tau}(s),

the same object class as eq. (23) of the C0 note and the conditioning flux of the Exp. 11 report. So D0 is not an abandonment of the conditioning program: it is the h-transform with g chosen as the data density relative to nu_h, with the heat potential propagated *forward from data* — where it is directly sampleable — instead of *backward from a terminal reward* — where it required conditioning on an unreachable event.

**Training identity (weightless).** Run the forward (noising) chain from data and store raw innovations xi_{k,e} exactly as in C0. The only change to the regression: condition on the **later** state. For block innovations Xi^{(r)}_k = r^{-1/2} sum_{q<r} xi_{k+q}, the L2 minimizer of

    E | f(t_{k+r}, S_{k+r}, y) - Xi^{(r)}_k |^2

over the *unweighted* forward law is E[Xi | S_{k+r}, y], which is approx -sqrt(dt_eff) * eta_rev (up to the sign convention fixed in Sec. 4) by the same one-step argument as C0 Sec. 8.2, applied to the reversed chain. No terminal weight, no ESS, no branches, no value MC. All the C0 cache machinery (raw-xi storage, clip masks, stride r, time slices) is reused verbatim; only the keying state and the weights change.

**Where the digit signal now lives.** Near tau ~ T (states close to data), the reverse shift is large and stroke-specific because the trajectory demonstrably came from a digit. This is the exact analogue of DDPM's high-SNR low-noise regime, and it is the regime that every C0–C3 estimator was structurally unable to populate.

## 3. Patch list

### Phase 0 — diagnostics before any training (no model, ~hours)

- **P0.1 Forward-noising preview script** (`mnist/diag_forward_noising.py` or a flag on the cache generator): roll the free reference from normalized MNIST digits (with lambda uniform mix), save an image grid of S_k at k in {0, K/8, K/4, ..., K} for a few digits and seeds.
- **P0.2 Destruction/clip curves**: per time bin, log (i) feature distance D_h(S_k, S_0)^2 and pixel correlation with S_0, (ii) entropy, (iii) clip fraction, (iv) frozen-edge fraction (theta below threshold).
- **P0.3 Schedule calibration sweep** over (w_sigma, K, lambda, w_free; optionally a time-increasing noise schedule w_sigma(t)): choose the smallest noise scale such that by k = K the digit correlation is ~0 and entropy is near the terminal plateau, while clip fraction stays small (target < 5% of edge-steps; verify what the limiter reports in the code). The previous regime (w_sigma = 0.005, w_free = 0.03) is likely either non-mixing or clip-dominated from sparse starts — this sweep settles it empirically.
- **P0.4 Prior bank**: store the terminal states of calibrated forward rollouts. These are exact samples of the reverse process's initial law and remove prior mismatch from the first experiments. (Matching a parametric source sampler to them is a later, separate step.)
- **Gate**: if no setting destroys digits without clip blowup, fix limiter/substepping first; do not proceed to training.

### Phase 1 — cache generator changes (small diff to the C0 cache path)

- `--rollout-init data` (vs existing `source`): S_0 = (1 - lambda) a + lambda * unif, a a training digit with label y; keep `--rollout-init source` so old C0 remains one flag away for ablations.
- Slice format becomes `(state = S_{k+r}, time = t_{k+r} (or tau), label = y, innovation = Xi^{(r)}_k, edge_mask)`. Differences from C0 Sec. 11.4: keyed at the **later** state; **no** `log_weight` (or constant 1 to avoid touching the trainer interface); **no** endpoint A and no source z as model inputs. Keep a (the start digit) in the cache for diagnostics only — same spirit as C0 checklist item 5.
- Keep: raw-xi storage before clipping, per-edge clip masks over the block, mobility-threshold masking, time-slices-per-path, periodic cache refresh (rollouts are model-independent, as in C0).

### Phase 2 — loss changes (smaller diff)

- Same masked per-slice MSE of C0 eq. (43) with target Xi and prediction sqrt(dt_eff) * eta_theta(tau, S_{k+r}, y); delete the self-normalized weighting of eqs. (44)–(46) in this mode (weights identically 1).
- Keep the eta L2 regularizer, gradient clip, AdamW, EMA — unchanged from C0 defaults.
- Sign convention: define eta_theta to predict E[Xi | later state] directly (call it `m_theta`), and put all sign handling in the sampler (Sec. 4). This avoids a silent sign bug, which is the most likely way this patch fails for boring reasons.

### Phase 3 — reverse sampler

Given the later state s' = S_{k+r} and learned mean innovation per step m_theta / sqrt(r) (from the block-trained head), the reverse Euler step approximates the exact reverse kernel by

```
xi_hat ~ N( m_theta(tau, s', y) / sqrt(r) per-step equivalent, I )   # fresh unit noise + learned mean
dK_e   = b_ref_e(s') * dt + sigma_ref_e(s') * sqrt(dt) * xi_hat_e
s      = s' - div_h(dK)        # subtract: undo a forward step
```

with coefficients evaluated at s' (O(dt) consistent), the same conservative incidence, limiter, floor/renorm, and adaptive substeps as the forward sampler. In continuous time this is the standard reverse SDE with unchanged diffusion coefficient; the learned conditional mean absorbs both the score term and the divergence-of-mobility correction to O(dt), so no explicit div(sigma sigma^T) term is needed. Initialization from the Phase 0 prior bank. Practical notes:

- Run reverse with the same stride bookkeeping used in training (predict block mean, apply over r substeps, or train r = 1..8 first to keep this simple).
- Keep `--sample-control-strengths` style sweeps from C2.1 for the learned mean term; the honest setting is strength 1.
- Class conditioning via the label embedding only; no z input.

### Phase 4 — acceptance tests and diagnostics

Ordered from cheapest and most decisive:

1. **Overfit-one-image smoke test**: dataset = a single digit, small model. The reverse sampler from the prior bank must reproduce that digit. Failure here means an implementation bug (sign, indexing, incidence direction), not theory. This should be the first training run of the patch.
2. **Reconstruction test**: take a held-out *forward* path, start the reverse sampler at its exact terminal state, and check whether a digit (any same-class digit, not necessarily the original) re-forms. This isolates the learned field from prior mismatch.
3. **Per-time-bin signal**: target RMS of Xi-residual vs time, and generation-time learned/noise step ratio per bin. Expect a strongly time-dependent profile, large near tau ~ T (close to data). Compare against C0's flat 0.007.
4. Existing image stats (entropy, TV, checkerboard/high-frequency energy), preview grids, and classifier accuracy if the diagnostic classifier is available.
5. Ablations in the C0 Sec. 13 style: reverse drift only (no fresh noise) vs full stochastic reverse; data-init vs source-init cache (one flag).

### Phase 5 — reconnecting to the preference-function program (after D0 works)

D0 deliberately learns the data-density h-transform first. The original goal — conditioning on a *preference* g — is then layered on top, where it becomes tractable because the learned generator reaches digit-like states:

- **Guidance**: add the edge flux (2/h) theta_e d^h_e log g(s)-style term (or its Brownian-shift form) to the learned reverse dynamics at sampling time — the conservative-flux analogue of classifier guidance; honest if reported as approximate h-transform of the learned process.
- **C1 revived**: use the D0 generator as the proposal for Girsanov-corrected terminal tilting; ESS will no longer collapse because the proposal mass actually overlaps the conditioning event.
- **C3 revived, on-policy**: branch/value estimation under the learned generator's state distribution, with the derivative-regularized value losses already sketched in the Exp. 11 report.

This ordering also gives the advisor report a clean narrative: the terminal-reward estimators were sound but were applied on a reference measure that gives the conditioning event probability ~0; D0 changes the measure, not the theory.

## 4. Risks and open items

- **Sign and time conventions** are the dominant bug risk: forward time vs reversed tau in the embedding, the minus sign in the incidence subtraction, and predicting E[Xi] vs -E[Xi]. Mitigate via the overfit-one-image test before any real run.
- **Conditional covariance approximation**: the true posterior of xi given the later state has covariance < I; using fresh N(0, I) is the standard reverse-SDE approximation (exact as dt -> 0). Log the empirical residual covariance on held-out forward steps as a sanity number.
- **Block bias in reverse**: conditioning the block innovation on S_{k+r} is biased if the state moves a lot within the block; near data (large drift, clipping) this is worse than in C0. Start with smaller r (4–8) near tau ~ T; a time-dependent stride is a cheap follow-up.
- **lambda (uniform mix) defines the target**: generated digits will sit on a lambda-floored simplex. Choose lambda once in Phase 0 and use it consistently for data, features, previews, and any classifier.
- **w_free under data starts**: with |R| ~ 1 at stroke edges, the free drift may need to be reduced (or substeps increased) for forward-from-data rollouts; otherwise the clip mask deletes the most informative slices. Phase 0 P0.3 decides this.
- **Code-dependent unknowns** (re-upload the codebase to settle): the exact horizon T and time embedding, the limiter/substep API and what it reports, whether the trainer's weight path can be cleanly bypassed, the network input signature (does dropping z need a model change or a zero-tensor stub), and the cache schema.

## 5. Minimal checklist for the D0 patch (analogue of C0 Sec. 16)

1. Forward rollouts start at lambda-mixed data, learned weight exactly zero, same (w_free, w_sigma, limiter) as the reverse sampler.
2. Raw Gaussian innovations stored before clipping; clipped/invalid edges masked.
3. Slices keyed at the later state S_{k+r}; no terminal weights anywhere in the loss.
4. Network sees (tau, state, label) only; the start digit a is never an input.
5. Reverse step subtracts the incidence update and uses fresh unit noise plus the learned mean.
6. Sampler initialization from banked forward terminals in all first runs.
7. Overfit-one-image test passes before any full run.
8. Per-time-bin signal curves, clip fraction, frozen-edge fraction, and preview grids logged every cache refresh.
