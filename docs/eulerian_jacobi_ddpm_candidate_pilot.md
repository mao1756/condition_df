# K=128 approximate-candidate Eulerian Jacobi objective pilot

This patch is an exploratory, objective-bearing test of the assembled Eulerian
Jacobi generator. It asks one decision question: after the fixed numerical audit
and a complete ten-class oracle control, does the frozen 34,974-parameter global
learner improve over paired null control on both forward-terminal reversals and
Dirichlet-prior generation?

The CUDA backend emits an **approximate-candidate Rao--Blackwell target**. Nothing
in this experiment calls that target exact, certified, K=512-equivalent,
continuum-consistent, or an exact reverse score. The fixed K=128 chain is an
exploratory finite model. Here K=128 means 128 outer chain steps. The CUDA
candidate profile uses 128 Legendre modes as its frozen minimum and may increase
the realized mode count adaptively, up to the protected kernel's ceiling of 1024;
the audit records the observed maximum and number of lanes above the minimum.

## What the command does

`run` executes one lifecycle in this order:

1. binds the 27 protected source files, the additive patch, authenticated ARFF,
   accepted evaluator authority, path-ID roles, environment, approval, and caps;
2. prepares only the 128-minimum-mode/56-bisection proposal backend and all fixed
   RNG keys used by production;
3. compares 512 frozen lanes using one aligned v2 stateless-Philox stream: the
   candidate and 568-mode arms consume the same rounded initial 64-bit midpoint,
   while an audit-only certified reference uses the CUDA authorizer, bounded Arb
   fallback where required, and the same v2 prefix plus continuation blocks;
4. prices backend preparation and the 512-lane audit with conservative
   sub-60-second quanta, then measures a fresh production-shaped resource smoke
   and stops if the complete pilot does not fit the three-hour envelope;
5. runs complete paired null/oracle reverse paths for one validation image from
   every digit before any cache or training artifact can exist;
6. saves 250 training and 100 validation whole-path forward-record cohorts;
7. trains exactly 750 updates and selects the earliest finite minimum EMA
   checkpoint among updates 250, 500, and 750;
8. saves six assembled path authorities - `null_prior`, `learned_prior`,
   `null_forward_terminal`, `learned_forward_terminal`, `null_oracle`, and
   `oracle`, while reusing rather than recomputing the ten oracle controls;
9. commits `POPULATIONS_SEALED.json`, then and only then loads the accepted
   evaluator on CPU and constructs the fixed blinded 40-image prior review;
10. ends with `review_pending`. `record-review` records the fixed answers and
    applies the prespecified route table.

There is no full-scale launch, tuning flag, audit-only production command, resume
mode, or terminal-MNIST opening. The ARFF's full bytes are read only to authenticate
its SHA-256. The content loader stops after row index 59999: `data_roles.json`
records `content_rows_read=60000`, `last_content_row_index=59999`,
`terminal_content_rows_read=0`, and `full_file_read_purpose=sha256-only`. Thus
hashing the authority is not evidence access to terminal rows 60000 and above.

Each cohort is admitted with nonzero storage and time estimates before compute.
Its final outer-step reservation includes measured persistence time, and the
completed NPZ, telemetry, and PNGs are written before a later admission can stop
the run. Every failed or partial cohort and every rendered image is retained.
Every terminal failure also receives a compact `REPORT.md` and
`experiment_note.md` with the scoped failure and next action.

Failure sealing is itself governed. The runner preserves the original failed
admission for a resource route, then records exactly one terminal-only
`failure_terminalization` admission with a declared 5.0-second and 1,048,576-byte
shutdown envelope, `major_stage=0`, `projected_remaining_seconds=0`, and
`reserve_remaining_seconds=0`. On a failure, the configured 900-second reserve is
released solely for this shutdown path; both the receipt and
`failure.json.failure_terminalization_reserve_seconds` bind the zero remaining
balance. The receipt is retained even if the terminal envelope no longer fits and
never authorizes scientific work. The runner writes the failure artifacts and a
provisional seal, completes exactly one non-raising terminal event, regenerates the
route-specific reports (and `resource_stop.json` when applicable) from the final
ledger, creates the final manifest, and verifies it read-only. `record-review`
rehydrates its governor before review work; any subsequent review exception uses
this same failure-terminalization path.

## Fixed scientific contract

- Grid/state: nonnegative unit-mass vectors on the periodic `28 x 28` grid.
- Chain: 128 outer steps, seven palindromic matching phases, two controller
  microsteps, fixed anchors at completed reverse steps `0, 32, 64, 96, 128`.
- Prior: label-independent `Dirichlet(1,...,1)`.
- Data: ARFF train rows `[0,55000)` and validation rows `[55000,60000)` only;
  content parsing stops before row 60000, independently of full-file SHA-256
  authentication.
- Training: 25 paths/class for training, 10/class for validation, four records per
  path, Adam at `2e-4`, batch 64, EMA `0.999`, 750 updates.
- Objective populations: two prior paths/class, two forward-terminal paths/class,
  and one oracle path/class.
- Pairing: null and candidate controllers share fixed stateless transition-ID and
  role-key inputs. Their later states differ once their controllers diverge.
- Rendering: global demix, unit-mass normalization, and the fixed `25471/255`
  raster scale. Per-image maximum normalization is forbidden.
- Candidate law: no clipping, floors, projection, renormalization, limiter,
  rejection, resampling, certification, or fallback in the candidate runtime.
  The one certified reference call belongs only to the fixed numerical audit
  and is never available to production transitions, training targets, or sampling.

All of these values are serialized in `config.json`; none is a CLI tuning option.

## Gates and diagnostics

Gate A is an execution/integrity gate. It binds source, data roles, path and seed
identities, renderer, evaluator authority, and the post-seal firewall. Failure
invalidates this run instance; it does not imply scientific failure.

Gate B is an execution/integrity gate over the fixed 512-lane bank. All three arms
are paired by `philox4x32-10-canonical-transition-v2`, not merely by equal logical
key names. For each active lane, the audit saves the initial prefix numerator,
prefix length, and rounded midpoint. The candidate and fast arms use that midpoint;
the certified arm continues the same v2 Philox prefix stream. Gate B requires:

- candidate minimum modes/bisections exactly `128/56`, the expected adaptive
  maximum and above-minimum lane count, runtime type `CandidateRuntime`, returned
  batch type `CandidateRBCudaBatch`, and candidate-only production dispatch;
- exact v2 seed/prefix hashes, the certified reference's v2 runtime contract, and
  bit-identical candidate-only versus certified-internal candidate draws;
- expected active, structural-no-op, approximation, and valid masks;
- zero invalid, correction, clipping, floor, limiter, projection, renormalization,
  authorizer, and fallback counts for the candidate arm; audit-reference
  authorizer or Arb activity is separately named and does not authorize candidate
  output;
- maximum later-fraction error at most `2e-10` versus both references;
- maximum target error at most `2e-8` versus both references;
- maximum reconstructed pair-total error at most `2e-12`;
- exact unchanged-fraction/zero-target structural no-ops.

`candidate_audit/outputs.npz` retains
`rng_v2_initial_prefix_numerators`, `rng_v2_initial_prefix_bits`,
`rng_v2_uniform_midpoints`, the candidate and fast outputs, the certified outputs,
the certified arm's internal candidate outputs, and its certified/CUDA/fallback
masks. `candidate_audit/report.json` binds those arrays in `rng_alignment`, along
with the canonical seed, array hashes, internal-candidate exactness, audit-reference
call counts, and certified runtime contract. This makes RNG pairing independently
replayable rather than inferred from identical argument names.

Gate C is an execution/integrity gate required for learner attribution. All ten
oracle and null paths must be healthy, the oracle must beat null final raw-mass L1
on at least 9/10 paths, and aggregate oracle L1 must be lower. Failure directs work
to composition, orientation, schedule, or backend integration before blaming the
learner.

Gate D is an execution/integrity resource gate. Major-stage admission requires

```text
active + 1.25 * projected_remaining + 900 < approved_active_cap
```

and each next quantum must be predicted below 60 seconds, fit active and storage
caps, and keep peak CUDA allocation within the approved fraction. The 900-second
reserve remains unavailable until the population seal exists. A resource stop is
a valid partial tree, not a scientific negative.

Admission receipts bind their declared duration, the last three completed
same-kind durations, the exact event boundary, measured-smoke floor, predicted
bytes, and decision inequality. Backend preparation, the candidate audit, and the
production-shaped smoke use a conservative 55-second pre-smoke bound. A completed
quantum is checked again against time, storage, and CUDA caps; an actual overshoot
is sealed as `resource_stopped` rather than being reported as a successful route.

Diagnostic E is exploratory action routing. The forward marker needs at least
12/20 learned L1 wins, at least 1% aggregate relative improvement, positive finite
controller RMS, and healthy trajectories. The human prior marker needs at least
15/20 recognizable learned images, at least 12/20 requested digits, both counts
strictly above null, 20 distinct learned endpoints, and no within-class duplicate
pair. The accepted evaluator markers are secondary and cannot override a human
negative.

## Resource budget

Expected wall/accelerator time is measured afresh by the production-shaped smoke.
The hard command defaults are 10,800 active seconds, 2 GiB persisted storage, and
0.75 peak CUDA allocation fraction, including a 900-second terminal reserve.
Candidate transition work is fixed at 273,960,960 transitions for the pilot:
122,931,200 forward-record, 140,492,800 reverse, and 10,536,960 forward-evaluation
transitions. The resource decision purchases direct all-class image evidence; a
smaller local prediction proxy cannot answer it.

Admissions are repeated at cache construction, training, objective sampling,
population sealing, sealed evaluation, post-evaluator finalization, and
record-review finalization. Failure routes add the single terminal-only admission
described above. The ledger therefore accounts for source/data overhead, every
bounded compute quantum, durable persistence, evaluator work, report generation,
and the final read-only verification envelope. Successful routes must finish below
all three caps, not merely have passed an earlier projection.

## Artifact and implementation cost

The additive implementation is intentionally larger than a minimal sampler because
this run has three demonstrated failure modes to prevent: Windows-unsafe replacement
of live ledgers, loss of a completed cohort at the next admission boundary, and
post-seal evaluator leakage. The extra source/test complexity buys stage-wise
durability, a sole seal-bound evaluator loader, and semantic replay of coordinated
tampering. It does not add another scientific proxy, numerical certificate family,
resume format, or authorization layer. The run still ends in direct images and a
prespecified action; no positive branch launches full scale automatically.

## Why the sealed v1 run is not a candidate result

The preserved tree at
`runs/experiment14-eulerian-jacobi-ddpm/candidate-k128-objective-pilot-v1`
ended as `candidate_health_failed`, but its paired comparison was invalid. The
candidate arm used the v2 stateless-Philox stream while both spectral references
used the distinct v1 lazy-dyadic stream. Equal logical key and transition-ID values
therefore selected different uniforms. The two references agreed with each other,
and a read-only replay using the candidate's v2 uniform reduced candidate-versus-568
errors to `2.220446049250313e-16` maximum for later fraction and
`4.5075054799781356e-14` maximum for target, with zero failures at the frozen
thresholds.

V1 stopped before the oracle, cache, training, population, or image stages. It
contains no learned-controller or generator result and is not evidence that the
candidate arithmetic, learner, or Eulerian/Jacobi strategy failed. Its exact 32
source-bound files are preserved outside the sealed tree at
`handoff/source_snapshots/candidate-k128-pilot-v1-a216632d`; that snapshot has its
own complete source manifest and checksums. The v1 run itself remains untouched.

## Production command

Run from the repository root only after a fresh explicit compute approval:

```powershell
.\.venv\Scripts\python.exe -B -m mnist.diag_eulerian_jacobi_ddpm_candidate_pilot run `
  --run-dir .\runs\experiment14-eulerian-jacobi-ddpm\candidate-k128-objective-pilot-v2-rng-aligned `
  --arff .\mnist_data\mnist_784.arff `
  --ddpm-run-dir .\runs\experiment13-conventional-ddpm\pixel-ddpm-calibration-v1-cpu-recovered `
  --device cuda:0 `
  --approval-id "<fresh-real-approval>" `
  --max-active-seconds 10800 `
  --max-storage-mib 2048 `
  --max-cuda-fraction 0.75
```

The implementation does not launch this command automatically. Supply a newly
approved identifier for this v2 run; do not reuse the v1 approval identifier.

## Blind review and verification

Complete exactly the generated `review/review_template.csv` without opening the
key, then record it:

```powershell
.\.venv\Scripts\python.exe -B -m mnist.diag_eulerian_jacobi_ddpm_candidate_pilot record-review `
  --run-dir .\runs\experiment14-eulerian-jacobi-ddpm\candidate-k128-objective-pilot-v2-rng-aligned `
  --answers .\human_review_answers.csv `
  --reviewer "REVIEWER NAME" `
  --confirm-manual-review
```

Read-only verification requires no CUDA, kernel compilation, MNIST loading, or
evaluator loading:

```powershell
.\.venv\Scripts\python.exe -B -m mnist.diag_eulerian_jacobi_ddpm_candidate_pilot verify `
  --run-dir .\runs\experiment14-eulerian-jacobi-ddpm\candidate-k128-objective-pilot-v2-rng-aligned
```

`passed: 1` means the saved tree is internally valid. It does not mean the
scientific diagnostic passed. Valid terminal routes are `integrity_failed`,
`candidate_health_failed`, `resource_projection_failed`, `oracle_control_failed`,
`resource_stopped`, `review_pending`, and `complete`.

The verifier recomputes the live source/config/path-ID authorities; rebuilds the
512-lane candidate bank from the saved development roles; replays oracle identities,
forward terminal starts, six cohort-to-assembled-to-raw population authorities,
rendering, evaluator paired effects, blind-review pairing, human markers, resource
admissions, stage order, reports, and the final manifest. On `resource_stopped` it
semantically checks every durable partial oracle, cache, training, and objective
cohort and forbids artifacts beyond the completed/current stage. Verification is
snapshot-checked read-only and never calls `torch.load` on the accepted evaluator.

The accepted evaluator has one loader. That helper independently validates the
current `POPULATIONS_SEALED.json` hash and firewall state before `torch.load`; caller
ordering or a forged in-memory firewall is insufficient.

## Outcome actions

- Numerical failure: repair or reject candidate integration; do not judge learner.
- Oracle failure: repair full-system composition before learner attribution.
- Forward positive, prior human negative: localize terminal/prior mismatch or
  on-policy shift; do not scale v0.
- Prior positive, forward negative: treat as suspicious and audit pairing/rendering.
- Human negative, evaluator positive: treat the task result as negative; audit only
  the evaluator/render/proxy mismatch and do not select samples.
- Forward, human, and evaluator markers all negative: stop this v0 recipe and run
  the materially different Experiment-10 formulation or stop the hypothesis.
- Human/direct positive with evaluator negative: preserve the task-positive result
  and audit evaluator symmetry.
- Both direct/human markers positive with noncollapse: freeze this pilot and seek
  separate approval for one fixed reference audit before any full population.

A healthy negative rejects only this frozen approximate-candidate K=128 v0 recipe.
It does not establish failure of all fixed-grid, Jacobi, or Eulerian approaches.
