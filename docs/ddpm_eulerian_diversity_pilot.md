# DDPM-to-Eulerian diversity pilot

## Research mode, cadence, and decision

This is an **exploratory, objective-bearing** experiment. It asks one
decision-changing question:

> Can the frozen conventional DDPM predictor drive the existing conservative
> Eulerian integrator while retaining recognizable requested-class digits and
> recovering materially more within-class diversity than the frozen historical
> Eulerian controller?

The completed factor-one V3 replay is the latest objective-bearing experiment. It
formed recognizable, mostly requested-class digits, but its learned diversity ratio
was only `0.08098423855664495`; the historical checkpoint line is stopped.

```text
Proxy-only patches since the last objective-bearing experiment: 0
```

This patch changes the controller representation without training. It does not tune
the historical checkpoint, alter the source/reference law, or authorize production.
A separately approved run must end in one of three substantive decisions: freeze a
promising adapter for fresh replication, make one major on-policy/theory-shaped
pivot, or stop the Eulerian generator hypothesis.

There are no confirmatory-claim thresholds. A positive pilot is feasibility
evidence, not a population, superiority, or theory result.

## Competing hypotheses

| Hypothesis | Predicted observation | Required action |
|---|---|---|
| Historical representation learned class prototypes | DDPM-Eulerian retains fidelity and materially exceeds historical diversity | Freeze adapter/configuration and replicate on fresh protected evidence |
| Conservative conversion loses DDPM mode information | Native DDPM and predicted masses are diverse, but Eulerian endpoints collapse under a passing teacher | Run one separately approved mobility-weighted potential-gradient comparison or stop |
| Off-policy DDPM inputs are the main defect | Predicted masses become collapsed or pathological on rollout states | Move to new on-policy/global or multiscale training or stop |
| Common orientation or composition is defective | Teacher, divergence, conservation, or deterministic replay fails | Repair this experiment before learner interpretation |
| Historical collapse is only late-time over-contraction | A prespecified historical anchor is jointly promising while its endpoint collapses | Permit at most one fresh horizon-only replication; no gain sweep |
| Learned Eulerian control lacks feasibility | Teacher and native DDPM pass, both learned Eulerian rows fail at every decision horizon | Stop unless the user approves one major theory/on-policy program |
| Evaluator framing is partly misaligned | Human fidelity and frozen classifier disagree | Preserve both; do not tune against the evaluator |

## Frozen population and pairing

- `40` independent paths, exactly four per requested digit, in class-major order.
- Path IDs are `d2e-v1-000` through `d2e-v1-039`.
- Exactly one endpoint per path and row is retained. Ranking, replacement, reroll,
  best-of-N generation, and sample-dependent stopping are forbidden.
- Eulerian outer steps: `256`.
- Saved anchors: `[0, 64, 128, 192, 256]`.
- Decision horizons: `64`, `128`, and `256`; `192` is diagnostic only.

| Row | Start/controller | Pairing interpretation |
|---|---|---|
| `null` | sealed Eulerian start; zero conditioning flux | fully matched Eulerian control |
| `teacher` | same start; target velocity through the complete flux/integrator interface | fully matched known-positive control; only target-bearing provider |
| `historical` | same start; frozen DirectFluxUNet | fully matched stopped-line baseline |
| `ddpm_eulerian` | same start; online denoised-target adapter | fully matched candidate |
| `native_ddpm` | sealed Gaussian latent; native reverse sampler | contextual and latent-linked, but not state- or dynamics-paired |

The four Eulerian rows share path-local standard-normal keys. The same standard
normal is supplied for a given path, outer step, attempted substep count, and
sub-index; row-dependent mobility may still produce different physical noise.
Native DDPM uses the same sealed per-path latent as the adapter's persistent latent,
but separate sealed reverse-noise seeds. This is not a paired causal comparison.

## Frozen constants and authorities

```text
INVENTORY_SEED                  = 0xE1600001
SOURCE_SEED_BASE                = 0xE1601000
DDPM_LATENT_SEED_BASE           = 0xE1602000
EULERIAN_EDGE_NOISE_ROOT        = 0xE1603001
NATIVE_DDPM_REVERSE_SEED_BASE   = 0xE1604000
REVIEW_SEED                     = 0xE1605001
SMOKE_SEED                      = 0xE160F001

MASS_SCALE_NUMERATOR            = 25471
MASS_SCALE_DENOMINATOR          = 255
MASS_SCALE_FLOAT64              = 99.88627450980393
MASS_SCALE_FLOAT64_HEX          = 0x1.8f8b8b8b8b8b9p+6
DDPM_STEPS                      = 1000
ANCHORS                         = 0,64,128,192,256
NATIVE_COMPLETED_STEP_ANCHORS   = 0,250,500,750,1000
ADAPTIVE_SUBSTEPS               = 1,2,4
MIN_TAU_FRACTION                = 0.03
POISSON_RESIDUAL_LIMIT          = 2e-4
MASS_SUM_ERROR_LIMIT            = 2e-6
```

Authority hashes:

- historical checkpoint: `8be77d1701887522f86099673431a928ad7dd2d350a06f7a94ade5c30a658cc3`;
- selected DDPM generator: `5f4065da8753ad5611ec4efd61b6d13082ce3c9cccaa62258f8019118e95dfc8`;
- evaluator checkpoint: `3d31d42a14fee0ecc72adc1644718a037cc48e649948407da6c272b981840c92`;
- evaluator selection: `e6cd9e49ca61237d3a10e9ad2fe0ad09f7a33ea22911fdd73fd99f3a07e4c668`;
- MNIST ARFF: 127,888,265 bytes and
  `418c0a60d2b4abc95db2e2bbf676f3af93ddaf18f79ba3f640624ab57007fb4b`.

For Eulerian noise, interpret the first eight SHA-256 bytes in unsigned **big-endian**
byte order as the runner's frozen 64-bit seed:

```text
SHA256("edge-noise-v1|<root>|<path_id>|<outer_step>|<attempted_substeps>|<sub_index>")
```

Seed derivation and generated tensors must be independent of batch order and
chunking. Supplying a standard normal must not advance global PyTorch RNG state.

Scoped deviation from Plan section 8.2: the protected shared Eulerian core remains
byte-exact at SHA-256
`4dca1c40f25eb04b3d615bd0094891c7cedb8cea8a673607eb02e1ab977e4f19`.
The identical supplied-standard-normal stepping seam is additive in the new DDPM
adapter and is called only by this pilot. This preserves the V3 source authority
while keeping path-local common random numbers explicit.

## Data roles and firewalls

- ARFF rows `[0,55000)` supply inherited development/raster authority only.
- Rows `[55000,60000)` supply the first four deterministic validation targets per
  class and evaluator-development roles.
- Terminal real-reference rows, evaluator weights, review keys, and human answers
  cannot open until all five populations are sealed.
- Whole-file hashing is authority inspection, not content parsing.
- Only `TeacherControllerProvider` may possess target images or masses.
  `null`, `historical`, and `ddpm_eulerian` public APIs cannot receive them.
- The adapter calls the epsilon predictor online. A native endpoint cannot enter its
  API or serve as a hidden target.
- Every model is frozen, has no optimizer, and is checked for parameter/buffer,
  training-mode, gradient, and state-digest immutability.
- Scientific constants are absent from the production CLI. No gain, path-count,
  time-map, seed, anchor, or threshold option is exposed.

## Adapter equations

For simplex mass `s`, retain the V3 global scale

\[
c=\frac{25471}{255},\qquad
x(s)=2\,\operatorname{clip}(c s,0,1)-1.
\]

This is a continuous float transform into `(B,1,28,28)`. There is no uint8
round-trip or per-image normalization.

For remaining Eulerian time `tau`, natural horizon `T`, and `K=1000`, use

\[
k(\tau)=\operatorname{clip}\!\left(
\operatorname{round}((K-1)\tau/T),0,K-1\right).
\]

The saved-anchor mapping is approximately `0->999`, `64->749`, `128->500`,
`192->250`; no controller call occurs after step `256`.

With persistent sealed latent `z`, form

\[
x_k=\sqrt{\bar\alpha_k}\,x(s)+\sqrt{1-\bar\alpha_k}\,z,
\qquad
\widehat\epsilon=\epsilon_\theta(x_k,k,y),
\]

and

\[
\widehat x_0=\operatorname{clip}\left(
\frac{x_k-\sqrt{1-\bar\alpha_k}\widehat\epsilon}
{\sqrt{\bar\alpha_k}},-1,1\right).
\]

Using the inherited positive mass floor `epsilon_m`, define

\[
a_i=(\widehat x_{0,i}+1)/2+\epsilon_m,
\qquad \widehat q_i=a_i/\sum_j a_j.
\]

The desired node velocity is

\[
\tau_{\mathrm{eff}}=\max(\tau,0.03T),\qquad
v=(\widehat q-s)/\tau_{\mathrm{eff}},\qquad
v\leftarrow v-\operatorname{mean}(v).
\]

Let `D_per` be the existing periodic tail/head divergence. Reconstruct the
minimum-norm total flux

\[
J_{\mathrm{total}}=\arg\min_J \tfrac12\lVert J\rVert_2^2
\quad\text{subject to}\quad D_{\mathrm{per}}J=v,
\]

then pass

\[
J_{\mathrm{ctrl}}=J_{\mathrm{total}}-w_{\mathrm{free}}J_{\mathrm{free}}(s)
\]

to the inherited integrator. The residual requirement is
`max(abs(D_per J_total - v)) <= 2e-4`. The historical free/noise terms, corrected
harmonic mobility, limiter, mass floor, renormalization, four-color update, and
adaptive `(1,2,4)` attempt policy remain unchanged.

Null uses `J_ctrl=0`. Teacher replaces `q_hat` with its private target mass and uses
the same remaining-time, free-cancellation, Poisson, noise, limiter, and retry path.
Historical uses the exact frozen `DirectFluxUNet` contract. Native DDPM uses the
existing reverse sampler and its sealed latent/noise authorities.

## Theory and boundary scope

The retained theory references are [the primary construction](paper.pdf)
(523,280 bytes; SHA-256
`31c8b3660317954d9cc507e4894891804b379bc1c70aa526f4101f93b4a45fed`) and
[the Eulerian approximation note](eulerian_approx.pdf) (855,994 bytes; SHA-256
`f5310576d5c1a36b1889c4e4cc959a5edc8a09921e818a6a1134edb46838ff2a`). They motivate the program
objective and make the mismatch auditable; neither is treated as evidence that this
adapter is paper-faithful.

The physical grid and Poisson solve are periodic. Opposite image borders are
neighbors; this is not an image-domain no-flux boundary.

The adapter is an exploratory bridge. It approximates a DDPM training-marginal
input by forward-noising a rendered Eulerian state, maps a denoised image to a
simplex target, and uses an unweighted minimum-energy flux. It does **not** implement
an exact Doob transform, prove that the DDPM score is `D log u`, parameterize a
mobility-weighted potential gradient, match a grid-scaled Dirichlet reference law,
or establish a continuous-time boundary law.

The theory-motivated `alpha_h=beta/n^2` intervention is deliberately excluded: it
would simultaneously change source law, free drift, noise, boundary behavior, and
time scale. It remains a possible later major experiment, not a prerequisite for
this direct objective test.

## Lifecycle, sealing, resume, and failure

The stage order is:

1. `binding_preflight`;
2. `cpu_smoke_replay`;
3. `inventory_and_start_seal`;
4. `null_population`;
5. `teacher_population`;
6. `historical_population`;
7. `ddpm_eulerian_population`;
8. `native_ddpm_population`;
9. `population_seal`;
10. `machine_scoring`;
11. `render_and_review_bundle`;
12. `awaiting_human_review`;
13. `human_review_terminalization` after a valid external answer transaction.

`run` requires a nonexistent or empty directory. Before population seal, any
failure is terminal `failed_unsealed`; the runner saves partial evidence and a new
run directory is required. It never mixes stochastic evidence across retries.

After a valid seal, populations are immutable. An explicit post-seal resume may
continue only deterministic scoring, rendering, reporting, or review preparation
after replaying every seal, scientific authority, and the cumulative resource
ledger before deleting any derived artifact. It cannot invoke generation.
`finalize-review` is a separate transaction that first read-only verifies the
awaiting tree, then consumes an external CSV with exactly one answer per blind
member. Invalid submissions are retained in an append-only attempt log and may be
corrected without altering populations or the private key.

The external CSV must retain the template's exact 80 unique `blind_id` values.
For every row, `recognizable` is the integer `0` or `1`, and `perceived_digit` is
an integer from `0` through `9` even when `recognizable=0`; `notes` is optional
free text. The editable answers file stays outside the immutable run directory.

Every population failure retains the last valid raw state and readable image,
using the sealed start or latent bank at step 0 before the first durable boundary,
plus the last completed step, attempted substeps, RNG keys, row/path identities,
telemetry tail, resource snapshot, traceback, and model/config hashes. Resource
events close coherently on success, observed failure, and resource stop.

Resource accounting is cumulative across resume. Population quanta are grouped as
8 Eulerian or 25 native steps plus row finalization. `machine_scoring` and
`render_and_review_bundle` preserve the configured 30-second reserve; only
`machine_terminalization` and `human_review_terminalization` use reserve zero.

`verify` is semantic and strictly read-only: it replays authorities, arrays,
scientific digests, metrics, gates, stages, resources, review, and route without
writing a receipt, cache, manifest amendment, or regenerated artifact.

## Required artifacts

The run saves configuration, command, source/checkpoint bindings, status, stage and
resource ledgers, claim boundary, report, gates, outcome, manifest, and its terminal
verification receipt. Inventory binds the exact path table, start bank, private
teacher bank, DDPM latent bank, RNG contract, and start seal.

Raw populations are saved for `null`, `teacher`, `historical`, `ddpm_eulerian`, and
`native_ddpm`, with every anchor, path ID, label, authority hash, and canonical
scientific digest. Companion uint8 populations retain the same anchor and identity
axes and replay exactly from raw arrays. The population seal binds all rows and telemetry.
Telemetry records retry/clipping, mass health, increment scales, model identity, CRN
keys, adapter inputs/outputs, predicted-mass diversity, and Poisson residuals.

Scoped deviation from the plan's universal NPZ preamble: raw population NPZs and
sealed inventory banks are self-described and scientifically digested. Derived
uint8 populations, adapter telemetry, evaluator predictions, and failure snapshots
are not independently self-certifying containers; their authority comes from the
population seal or manifest plus semantic replay against the raw arrays. This avoids
a late container rewrite and means an extracted derived NPZ is not independently
auditable without its run tree.

Machine evaluation scores every prespecified horizon only after population seal.
The blind review contains all 40 historical and all 40 adapter endpoints, shuffled
by `REVIEW_SEED`; null and teacher remain visible in separate control sheets.

## Metric hierarchy

Machine classifier and diversity metrics are computed at `64`, `128`, and `256`.
Blind-human metrics are endpoint-only at step `256`. The primary objective metrics
are:

- blind human recognizability and requested-label agreement;
- frozen classifier accuracy/confusion and predicted-class coverage, reported
  separately from humans;
- within-class nearest-neighbor diversity relative to the frozen real reference;
- adapter-to-historical diversity ratio, per-class ratios, and exact duplicates.

Mechanism diagnostics include predicted-mass diversity, mass displacement,
render/noisy-input distributions, epsilon/score/velocity/flux/increment RMS,
controller/free/noise ratios, divergence residual, retries, clipping, and
native-versus-adapter latent-linked panels. Health metrics include bindings, seals,
mass/conservation, accepted attempts, model identity, resources, and restart state.
Mechanism and health metrics never replace the image-level objective.

## Typed gates and diagnostic thresholds

### Execution/integrity gates

**I1 -- authority and leakage.** Every source, checkpoint, schedule, dataset role,
scientific constant, inventory, RNG, and firewall must match before generation.
Failure invalidates the run but says nothing about the candidate.

**I2 -- deterministic orientation and CRN smoke.** Synthetic Poisson residual is at
most `2e-4`; two-path/four-step scientific bytes replay exactly; common-noise keys
are batch-order and retry stable; supplied noise does not advance global RNG.

**I3 -- numerical health.** At every saved/terminal step: finite state, minimum mass
at least zero, mass-sum error at most `2e-6`, accepted substeps in `{1,2,4}`, no
unlogged fallback, Poisson residual at most `2e-4`, and unchanged model digest.
Clipping fraction is logged but is not itself a pass threshold.

**I4 -- full-interface teacher.** With path as the unit, at least `36/40` teacher
paths improve endpoint squared L2; median relative squared L2 is at most `0.80` at
step 64 and `0.20` at step 256; endpoint classifier accuracy is at least `0.80`;
I1-I3 pass. Failure requires common-pipeline repair before learner attribution.

**I5 -- seal and completeness.** Exactly one complete endpoint and every anchor exist
for all 40 paths in every row; every output is retained; manifests and the
population seal replay; no evaluator/test/review authority opened early.

There are **no confirmatory-claim gates**.

### Diagnostic routing thresholds

**D1 -- adapter human fidelity at step 256:** recognizability at least `0.90`
(`36/40`) and requested-label agreement at least `0.80` (`32/40`). Classifier
accuracy remains separate.

**D2 -- adapter diversity at step 256:** real-reference ratio at least `0.25`, at
least `2.0x` the fresh historical row, and zero exact duplicate pairs. Candidate
exceeds historical in at least seven classes is supportive only.

**D3 -- native DDPM representation control:** classifier accuracy at least `0.80`,
diversity ratio at least `0.25`, and zero exact duplicate pairs. Failure invalidates
the representation probe rather than counting against the adapter.

**D4 -- promising early machine horizon:** because the blind human review contains
endpoints only, D1 cannot be evaluated at an earlier anchor. At step 64 or 128,
"jointly promising" is therefore an explicitly machine-only exploratory proxy:
frozen-classifier accuracy is at least `0.80` and the full D2 diversity proposition
(real-reference ratio at least `0.25`, adapter-to-historical ratio at least `2.0`,
and zero exact duplicates) holds at that same anchor. Step 192 remains diagnostic
only. This proxy is not renamed human fidelity and cannot make the endpoint D1
claim. An earlier-only result permits one fresh horizon-only replication; the
current endpoint is never retrospectively replaced.

The gate contract is deliberately asymmetric:

- I1-I5 are execution/integrity gates. They control whether population generation,
  terminal-reference access, or scientific interpretation is valid. Each uses the
  path as the independent unit where a path-level statistic is involved. Any
  failure records the exact failed proposition, retains the artifacts, and routes
  to repair in a new run; it does not establish that the DDPM adapter, historical
  learner, or Eulerian hypothesis is scientifically negative.
- D1-D4 are diagnostic thresholds. They control only the prespecified route
  and strength of the exploratory description. A threshold failure never erases
  the images, authorizes post-hoc endpoint selection, or becomes a confirmatory
  rejection.
- I4's exact proposition is that the same complete controller/integrator interface
  can carry a private known target under the four simultaneous prespecified
  statistics. I5's exact proposition is that all five factor-one populations and
  all required authorities are complete, immutable, semantically replayable, and
  evaluator/test/review-free before sealing.
- D1's unit is each of 40 blinded adapter endpoints. D2 and D3 use the full frozen
  40-path population with per-class diagnostics reported separately. D4 examines
  only steps 64 and 128 with its disclosed machine proxy; step 192 cannot be
  selected as a rescue horizon and step 256 is routed by the endpoint gates.

## Outcome routes

| Observation | Exact route | Required action |
|---|---|---|
| Integrity, teacher, or seal failure | lifecycle `failed_unsealed` or `resource_stopped` | Repair the same experiment; do not interpret learned rows |
| Native DDPM D3 failure | `native_ddpm_control_invalid` | Repair the frozen predictor/binding in a new run |
| Adapter passes endpoint D1 and D2 | `adapter_positive_freeze_replication` | Freeze adapter/configuration for fresh protected replication |
| Fidelity passes but diversity fails at every horizon | `adapter_fidelity_only_major_pivot_or_stop` | One separately approved on-policy/multiscale major pivot or stop |
| Endpoint fidelity fails but step 64 or 128 passes the machine-only D4 proxy | `adapter_early_joint_horizon_replication` | One fresh predeclared horizon-only replication; no gain sweep |
| Diversity improves but fidelity fails at every horizon | `adapter_diverse_not_faithful_major_pivot_or_stop` | Major on-policy/potential-gradient bridge or stop |
| Predicted masses are diverse but Eulerian outputs collapse under passing controls | `composition_mode_loss_theory_bridge_or_stop` | One theory-shaped mobility-weighted potential-gradient comparison or stop |
| Predicted masses themselves collapse/pathologize | `off_policy_bridge_on_policy_or_stop` | New on-policy/global training or stop |
| Historical earlier anchor is promising but candidate adds no advantage | `historical_early_horizon_replication` | At most one fresh horizon-only comparison; historical checkpoint remains stopped |
| Teacher/native pass; both learned Eulerian rows fail everywhere | `learned_eulerian_negative_stop_or_major_pivot` | Stop unless user explicitly approves one major theory/on-policy program |
| No prespecified branch fits, including unresolved human/classifier disagreement | `unclassified_stop_redesign` | Do not scale or tune; repair one identified design defect or stop |

The lifecycle routes are exactly `failed_unsealed`, `resource_stopped`,
`postseal_interrupted`, `awaiting_human_review`, and `complete`. The verifier replays
the complete scientific and lifecycle truth tables. A positive route never launches
replication, and a negative route never launches a new architecture or training run
automatically.

## Resource budget and stop rules

- New training: none.
- Expected wall time: 12-25 minutes; hard cap: 60 minutes.
- Expected accelerator-active time: 8-15 minutes; hard cap: 30 minutes.
- Expected peak CUDA allocation: below 1.5 GiB; hard cap: 50% of device memory.
- Expected storage: below 100 MiB; hard cap: 256 MiB.
- Actual new source at the initial implementation freeze: 2,449 runner lines plus 688 adapter lines = 3,137.
- Actual new tests at the initial implementation freeze: 3,447 runner-test lines plus 578 adapter-test lines = 4,025.
- This experiment note had 520 lines at the initial implementation freeze.
- The protected core is byte-exact; the shared smoke test has no content delta (its prior raw-byte identity is not separately authenticated).
- Aggregate visible new-file scope is 7,682 additions and zero deletions.
- This exceeds the plan's preferred 1,500-line source ceiling because the required resource-governed lifecycle, failure retention, resume, sealing, semantic verifier, and adversarial tests are included; it adds no general framework or nonessential abstraction.

Stop and save failure evidence on any integrity failure, leakage, binding or model
drift, nonfinite/negative state, mass error above `2e-6`, unlogged fallback,
prospective or observed resource overrun, unavailable native checkpoint, unplanned
training, nonempty/stale run directory, or need for an unplanned scientific knob.
Human inspection time is outside compute accounting, but answer validation,
semantic replay, report generation, sealing, and terminalization are charged to
`human_review_terminalization`.

## Commands and authorization boundary

Focused tests:

```powershell
python -B -m pytest `
  .\tests\test_ddpm_eulerian_adapter.py `
  .\tests\test_diag_d0_ddpm_eulerian_diversity_pilot.py `
  .\tests\test_smoke_mnist_eulerian_flux.py `
  -q
```

CPU smoke:

```powershell
python -B -m mnist.diag_d0_ddpm_eulerian_diversity_pilot smoke --device cpu
```

The first production attempt, `pilot-v1`, stopped before population rollout with
`failed_unsealed`: the frozen schedule was built on CPU and copied to CUDA, while
the adapter rebuilt its comparison schedule directly on CUDA. Backend-specific
float32 rounding made the mathematically identical arrays differ by a few ULPs.
The failed directory is verifier-clean evidence and must not be resumed or reused.
No population, evaluator, test-content, image, or review evidence was opened.

The correction retains exact equality but constructs the comparison schedule by
the same canonical CPU-then-device path. The real frozen DDPM binding passes on
`cuda:0`, including its exact schedule receipt, after this change.

The second attempt, `pilot-v2`, completed and sealed all five 40-path populations,
then stopped before scoring. The adapter checkpoint binding hashed state-dict keys
in sorted order while population telemetry used PyTorch state-dict order. Both
digests describe the same unchanged 83-tensor model; every per-row pre/post digest
is identical. No evaluator, test-content, rendered-image, or review evidence was
opened. Its failure terminal precheck repeated the false mismatch and left a pending
terminal receipt, so `pilot-v2` is preserved and not resumed. The adapter now uses
the repository-standard state-dict-order digest, and production checks this identity
contract during binding before any rollout.

The corrected production retry must use a fresh directory and approval reference:

```powershell
python -B -m mnist.diag_d0_ddpm_eulerian_diversity_pilot run `
  --run-dir '.\runs\experiment16-ddpm-eulerian-diversity\pilot-v3' `
  --legacy-checkpoint '.\runs\experiment10\20260601-035019_10q-wide-repeat\experiment10_direct_flux_mnist.pt' `
  --ddpm-run-dir '.\runs\experiment13-conventional-ddpm\pixel-ddpm-calibration-v1-cpu-recovered' `
  --arff '.\mnist_data\mnist_784.arff' `
  --device 'cuda:0' `
  --approval-id '<new-explicit-user-approval-reference>' `
  --max-wall-seconds 3600 `
  --max-accelerator-seconds 1800 `
  --max-cuda-fraction 0.50 `
  --max-storage-mib 256
```

Read-only verification is:

```powershell
python -B -m mnist.diag_d0_ddpm_eulerian_diversity_pilot verify `
  --run-dir $RunDir
```

For manual review, copy the sealed blank template to a work file outside
`$RunDir`, edit exactly the vocabulary specified above, and finalize from that
external file:

```powershell
$Answers = Join-Path (Get-Location) 'experiment16-review-answers.csv'
Copy-Item -LiteralPath "$RunDir\review\human_review_template.csv" -Destination $Answers
# Edit $Answers: all 80 rows require recognizable=0/1 and perceived_digit=0..9.
python -B -m mnist.diag_d0_ddpm_eulerian_diversity_pilot finalize-review `
  --run-dir $RunDir `
  --answers $Answers `
  --reviewer 'manual-reviewer-id' `
  --confirm-manual-review
```

Production, review, replication, training, and any follow-up experiment require
explicit user action/approval.

## Claim boundary and deliberate omissions

This pilot can establish only the behavior of one frozen DDPM checkpoint, one
historical checkpoint, one legacy source/process law, one adapter, 40 paths, the
declared exploratory thresholds, and the finite periodic backend.

It cannot establish confirmatory generator quality, population MNIST coverage,
superiority or equivalence to native DDPM, a common stochastic coupling, an exact
Doob transform, a matched Dirichlet-Ferguson reference prior, stationary law,
continuum convergence, an exact simplex-boundary weak law, or general Eulerian
success/failure. It deliberately omits new training, gain/schedule search,
reference-law changes, no-flux image boundaries, protected confirmation evidence,
automatic follow-up execution, and independent self-certification of derived NPZs.
