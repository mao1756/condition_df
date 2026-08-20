# Candidate complete-path Eulerian MNIST experiment

Date: 2026-08-14

Primary mode: exploratory, with a bounded engineering prerequisite.

This patch removes one demonstrated blocker: the exploratory candidate path used to
prepare the exact CUDA authorizer, and leaves a reusable Stage D/Stage E runner.
Bounded test-only CUDA regressions were exercised during implementation, including a
candidate-backend smoke with every exact path poisoned (candidate binary SHA-256
`456ead387d1afc1b55a21c32b55486ca32aee0537bd14c09f984522aaf2d2494`) and the
pre-existing exact-equivalence regression. This exceeded the intended CPU-only final
verification boundary, but did not run Stage D/E, produce an image, or establish a
scientific result. The candidate smoke is now explicitly opt-in. Any production run
still requires fresh, explicit compute approval.

## Program objective and current milestone

The program objective is a DDPM-like MNIST image generator based on the fixed-grid
Eulerian/Jacobi approximation. The nearest objective-bearing milestone is Stage D:
one complete 512-step reverse reconstruction from the frozen forward-terminal state,
with saved images for zero, learned, and source-informed controls.

The last objective-bearing experiment reached a healthy exact 16-step reverse suffix
on one opened image/path. It did not produce a complete image. Proxy-only patches
since that experiment: 1 (this engineering patch). The next approved mainline run
must execute Stage D, not add another authorization layer.

## One decision

Under the candidate-only fixed-grid transition law, can the frozen one-image
controller complete all 512 reverse steps, improve over paired zero control, and
coexist with a functioning source-informed oracle without invalid state mutation?

Stage D is a mechanism/reconstruction test, not generation. Stage E starts from an
independent Dirichlet prior. Even a successful Stage E result remains an overfit
one-image bridge; dataset-level training and multiple prior seeds are still required.

## Frozen experimental contract

- Named Stage D specification: `stage-d-anchor-v1`.
- Named Stage E specification: `stage-e-prior-v1`.
- Rows, in order: `zero`, `global-plus-1` at gain 1, and `source-informed` oracle.
- Stage D path ID/root seed: `1028864` / `261402`.
- Stage D stream role: `global_dilated_positive_complete_exact`, preserving the
  exact-prefix Philox witness.
- Stage E uses its own PCG64 Dirichlet initialization and distinct path/RNG authority.
- Eight-step atomic shards; 64 shards for 512 reverse steps.
- Shared randomness across the three rows; the unit is one image/path.
- Exact step-8/16 states are comparison-only and can never enter the candidate chain.

Scientific command-line overrides are intentionally absent. Stage E requires a
verified machine-eligible Stage D run and binds its external locator plus the hashes
of its manifest, outcome, and first-16 audit; it is never launched automatically.

## Approximation layers

The executable claim is explicitly limited to:

1. a fixed 28x28 spatial probability simplex;
2. the established split-time Eulerian/Jacobi composition;
3. a binary64 Legendre inverse-CDF candidate, with a 128-mode profile that may
   adapt to 1024 modes and uses 56 bisections;
4. the learned Rao--Blackwell score surrogate.

The candidate proposal is accepted without a correctly-rounded cell proof or Arb
fallback. Candidate preparation compiles/loads only the proposal kernel; it does not
probe or construct the exact authorizer. Certification is `not_applicable`.

## Controls, artifacts, and metrics

The zero row is the null control. The source-informed row is a target-aware positive
system control and is not a realizable generator baseline. The learned row receives
only the established `ModelInputs` interface.

Every graceful completion, resource pause, or scientific failure preserves the
longest valid raw shard chain and writes, when derivation is possible:

- raw boundary states and mixed-target/source metrics at every committed boundary;
- pre-clipping demixing diagnostics;
- raw and demixed individual PNGs/contact sheets at steps 0, 8, 16, 128, 256, 384,
  512, plus the last partial boundary;
- candidate health and controller/reference displacement summaries;
- the optional exact/candidate step-8/16 audit;
- `outcome.json`, `REPORT.md`, and one `artifact_manifest.json`.

On-policy drift is explicitly `not_available` because this compact experiment binds
no training-calibration artifact. It is not silently estimated from a different role.

## Gates and thresholds

### Numerical and artifact integrity

Gate type: execution/integrity gate.

Downstream action controlled: committing another candidate shard or accepting a run.

Exact proposition tested: committed states and candidate telemetry are finite,
nonnegative, in range, mass-conserving within `2e-12`, self-chained, produced under
the frozen row/RNG/schedule/backend contract, and use no forbidden correction,
clipping, flooring, limiting, projection, renormalization, fallback, replay, or exact
authorizer.

Why necessary: violation makes the trajectory mathematically invalid or changes the
declared numerical law.

Statistic and independent unit: every committed shard and every state row.

Pass condition: every named invariant holds.

Failure means: preserve the longest valid prefix and repair the localized numerical
or composition defect.

Failure does not mean: the learner or Jacobi/RB strategy is generally impossible.

Pass action: continue the candidate chain.

Fail action: stop before accepting the invalid shard and render existing evidence.

Ambiguous/invalid action: treat as an integrity failure; do not infer science.

### First-16 candidate/exact comparison

Gate type: diagnostic threshold.

Downstream action controlled: the recommended Stage D-to-Stage E branch, never raw
artifact completion.

Exact proposition tested: at both steps 8 and 16, every row has L1 discrepancy at
most 0.02, maximum absolute discrepancy at most 0.002, centered correlation at least
0.999, and learned-minus-zero vector-contrast relative error at most 0.25.

Why necessary: it is a practical check that the exploratory law is not completely
off on the only matching exact prefix; it is not proof of 512-step equivalence.

Statistic and independent unit: the single opened path at two prespecified horizons.

Pass condition: all disclosed checks pass at both horizons.

Failure means: retain the full Stage D result but audit/fix the candidate discrepancy
before claiming it transfers to the fixed-grid reference law.

Failure does not mean: the candidate-law images are invalid or the learner failed.

Pass action: permit the Stage E recommendation if the complete objective and controls
also support it.

Fail action: follow the candidate-kernel audit branch.

Ambiguous/invalid action: record `unavailable`; finish Stage D without a transfer
claim.

The inherited 1% learned improvement marker is also a diagnostic threshold. It is
reported but does not replace direction, images, or human review with a universal veto.

There is no confirmatory claim gate in this exploratory patch.

## Outcome-to-action table

| Observation | Interpretation | Required next action |
|---|---|---|
| Invalid candidate state/contract | Numerical or composition defect | Repair only the localized defect |
| Resource projection stops | Budget insufficient; science unresolved | Resume only after a larger explicit cap approval |
| First-16 diagnostic is grossly discrepant | Candidate law is a poor local proxy | Audit/fix candidate kernel; keep images |
| Oracle does not improve zero | Schedule/oracle/composition problem | Repair the system control before the learner |
| Oracle works; learned degrades late | Accumulation or on-policy issue | Revise schedule or add rollout-aware training |
| Oracle works; learned is negligible | Learner/scale/representation issue | Make one material learner/controller change |
| Complete healthy Stage D works | Mechanism feasibility on one anchor | Request approval for Stage E prior start |
| Stage E works | Overfit one-image bridge | Implement dataset-level Stage F and multiple seeds |
| All mechanisms are uninformative | Strategy failure remains plausible | Run the conventional DDPM sanity baseline or pivot |

## Resource policy and launch boundary

Provisional future Stage D budget from the audited plan: 3,900 active seconds,
300-second terminal reserve, CUDA memory below 80%, and persisted storage below 2 GiB.
The first two shards receive a conservative bootstrap allowance; after two measured
candidate shards, every remaining shard is projected at 1.2 times the maximum
observed candidate shard time. Setup/orchestration and homogeneous shard time are
reported separately. Execution also preserves a 64-MiB terminal-output allowance
inside the 2-GiB ceiling. A storage-reserve or 80%-CUDA stop is an engineering/
integrity failure, not a time-cap pause; it preserves the longest prefix but requires
an explicit environment/storage repair rather than an irrelevant time-cap increase.

An explicit cap amendment must increase the current cap and record its approval
reference. Scientific fields and committed shards remain unchanged. An active-time
resource pause is resumable; numerical, storage, CUDA-memory, and completed outcomes
are not. Hard-crash reconciliation bills the UTC interval plus five seconds as a
conservative upper bound, which may include offline downtime and is reported
separately from accelerator utilization.

Expected accelerator time: only the disclosed implementation smoke was executed; no
production accelerator time is authorized by this implementation patch.
Expected peak memory/storage for a future run: bounded by the 80%/2-GiB automatic
stops. The exact-prefix runs cannot answer the decision because they ended at 16
steps; another read-only analysis cannot produce the missing complete image.

New/changed complexity is 1,583 physical lines in the standalone runner and 641 in
its focused test file, plus narrow candidate-only preparation changes in the existing
CUDA/fused modules. This exceeds the plan's 400--650-line runner estimate, which is a
YAGNI warning rather than a gate. The retained bulk implements the requested raw
chain, resume, resource stop, derivation, rendering, report, and read-only verifier;
automatic Stage E/F launch, calibration machinery, training, and a new exact stack
were deliberately omitted. Do not insert a cleanup-only proxy patch before Stage D.

## Commands (not executed by this patch)

```powershell
python -m mnist.diag_d0_jacobi_rb_candidate_complete run-anchor `
  --reference-run-dir <sealed-reference-run> --runs-root <runs-root> `
  --run-name <new-stage-d-name> --device cuda:0 `
  --maximum-active-seconds 3900 --approval-reference <fresh-explicit-approval>

python -m mnist.diag_d0_jacobi_rb_candidate_complete verify `
  --run-dir <stage-d-run>

python -m mnist.diag_d0_jacobi_rb_candidate_complete resume `
  --run-dir <resource-paused-run> --device cuda:0 `
  --extend-maximum-active-seconds <larger-approved-cap> `
  --cap-amendment-reason <approval-reference>

python -m mnist.diag_d0_jacobi_rb_candidate_complete run-prior `
  --reference-run-dir <sealed-reference-run> --stage-d-run-dir <successful-stage-d-run> `
  --runs-root <runs-root> --run-name <new-stage-e-name> --device cuda:0 `
  --maximum-active-seconds <approved-cap> --approval-reference <fresh-explicit-approval>
```

No production Stage D, Stage E, exact audit, training, or multi-image sampling was
executed or authorized by this implementation.

## Stage D schedule-window v1 implementation note

This section supersedes the pre-run status above. The exploratory
`stage-d-anchor-v1-20260814` run subsequently completed one full 512-step path on
the candidate backend. Its source-informed control improved paired terminal
mixed-target squared-L2 error by 98.374% over zero, while the always-on learned row
was 43.637% worse than zero. The learned row was transiently positive, with its best
persisted result at completed step 176, its last positive persisted boundary at 216,
and its first negative persisted boundary at 224. The existing first-16
candidate/exact diagnostic passed. These are one opened path under the exploratory
candidate law, not population, generation, or exact-law evidence.

Primary mode for this patch: exploratory experiment implementation. The decision is
whether removing learned control after the post-hoc boundaries 176 or 216 preserves
a positive terminal effect. This is the smallest direct self-chained test of the
late-time/on-policy hypothesis; replaying the existing trajectory cannot answer it
because disabling control changes all later visited states. Proxy-only patches since
the latest objective-bearing experiment: 1 (this implementation).

The named specification is `stage-d-schedule-window-v1`. It fixes the same anchor,
path ID `1028864`, root seed `261402`, stream role, checkpoint, gain, and candidate
backend as Stage D. Its rows are exactly, in order:

1. `zero`;
2. `global-plus-1` (always-on gain 1);
3. `global-cutoff-176`;
4. `global-cutoff-216`;
5. `source-informed`.

All rows share the same random path. A cutoff row is active when
`reverse_time < cutoff_completed_reverse_steps / 512.0`: cutoff 176 includes outer
step `k=336` and is zero from `k=335` onward; cutoff 216 includes `k=296` and is zero
from `k=295` onward. No schedule, gain, row, or sweep option is exposed. The immutable
completed Stage D manifest, config, outcome, trajectory, and first-16 audit are copied
into the new run and independently hash-bound; the external Stage D directory is not
needed after initialization.

The execution/integrity gates remain finite/nonnegative states, mass error at most
`2e-12`, the candidate contract, forbidden-counter zeros, self-chaining, shared-row
identity with the copied Stage D trajectory, cutoff-row equality with always-on
through each cutoff, and exactly zero applied score/logistic-shift telemetry in every
wholly post-cutoff shard. Failures preserve the longest valid raw chain and adverse
images. The mapped first-16 comparison remains a nonblocking diagnostic, using
candidate-to-exact rows `[0,1,1,1,2]`. The inherited 1% marker is also diagnostic;
strictly positive terminal improvement is the exploratory direction criterion.
There is no confirmatory claim gate.

Complete-run images are rendered at steps 0, 8, 16, 128, 176, 192, 216, 224, 256,
384, and 512. Raw shards, the `[65,5,784]` boundary trajectory, 325 metric rows,
five-row raw/demixed images, `140x28` contact sheets, health, schedule identity,
mechanism summaries, outcome, report, resource ledger, and exhaustive manifest are
retained. A positive endpoint selects the larger of the two effects, with an exact
tie going to 176. Selection is unavailable on a partial, unhealthy, or failed run.
Stage E/F eligibility and automatic launch remain zero in every branch.

Outcome routing is fixed: an integrity/identity failure repairs only that defect; a
first-16 discrepancy routes to candidate-kernel audit; oracle failure routes to the
composition/backend interface; a positive cutoff is frozen for human review and a
separately approved Stage E implementation; earlier-positive but terminally
nonpositive cutoffs close nearby cutoff tuning and route to rollout-aware/on-policy
training or a material controller change; never-positive cutoffs route to a learner,
scale, or representation change.

No production compute is authorized by this implementation patch. After fresh
approval for the proposed 1,800 active-second maximum, the one permitted command is:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_candidate_complete `
  run-schedule-window `
  --reference-run-dir .\runs\experiment12-d0-jacobi-rb-global-dilated-continuation\production-v2-prefix-successor-v1 `
  --stage-d-run-dir .\runs\experiment12-d0-jacobi-rb-candidate-complete\stage-d-anchor-v1-20260814 `
  --runs-root .\runs\experiment12-d0-jacobi-rb-candidate-complete `
  --run-name stage-d-schedule-window-v1-20260814 `
  --device cuda:0 `
  --maximum-active-seconds 1800 `
  --approval-reference <fresh-explicit-user-approval-reference>
```

Stop after the runner's read-only verification and report the routed result. Do not
invoke Stage E or Stage F automatically.

## Stage E prior-start cutoff-216 v1 implementation note

Primary mode: exploratory. The decision is whether the Stage D-selected cutoff 216
transfers from the forward anchor to one frozen intended-prior start: does it beat
paired zero and always-on control while the oracle validates composition, or must static-schedule
work stop before Stage F? This is the final static-cutoff test.

The specification `stage-e-prior-cutoff-216-v1` is exposed only by
`run-prior-cutoff-216`. It reuses the legacy PCG64 Dirichlet seed `261403`, path
`1028865`, root seed `261404`, and stream role `global_dilated_positive_prior_candidate`.
The `same-path-four-row` family is fixed, in order, to `zero`, `global-plus-1`,
`global-cutoff-216`, and `source-informed`.
Boundaries are 0, 8, 16, 128, 216, 224, 256, 384, and 512; totals are 351,232
transitions per shard and 22,478,848 over 64 shards.

Initialization authenticates the completed schedule run with its read-only
manifest-tree/link verifier, not its source-sensitive `verify_run()`. Six copied
authorities under `inputs/stage_d_schedule_predecessor/` are pinned as follows:

- manifest: `385c6b82e9fa219ab32096437c62194144c95ae818754e052e85ba4b30bdc94f`;
- config: `693bd8d599261f39fea61a2bea981193f5077c6add13c5c4d1a4d57f9e422537`;
- bindings: `d9706eb66c09267ae4e2ac1f345cb78b2bd9078cc0fd3de19332fef40923d268`;
- outcome: `d7fe1cbbd52dc57a40a6be3d9202335d986f2d45a553d666aa18ded4f1d61db1`;
- health: `2270598922131b7580c06af8db5833e98c095e5942626c3b7c3cdd6d85d60d0a`;
- first-16 audit: `1406c9d27f25a9d01e4c82f1488fd6e1d33c2c4c627282b4025614ee434cd478`.
The predecessor must be healthy and complete through 512, pass schedule identity and
oracle, select cutoff 216, and record `selected_cutoff_ready_for_review=1`. Resume and
verification use only the copies. Its literal `<fresh-approval-reference>` is preserved
as a provenance defect, not approval for this run; new placeholder references fail.

Execution/integrity gates cover finite nonnegative states, mass error at most
`2e-12`, forbidden-operation zeros, valid self-chain and RNG pairing, cutoff
identity, and the existing resource, storage, and CUDA bounds. Cutoff must equal
always-on at every persisted boundary through 216; shards 27--63 must report exactly zero
applied score, unscaled score, and logistic-shift telemetry. Checks run at shard
admission, terminalization, and read-only verification.

The oracle's at-least-1% improvement over zero is a diagnostic prerequisite for
learner attribution. Cutoff must be strictly better than both zero and always-on;
its 1% marker is descriptive only. The report includes both comparisons, relative effect,
centered correlation, oracle-gap closure, boundary metrics, controller magnitudes, and
all endpoint images. Human recognizability is required before Stage F and is not automated.

Routing is fixed: repair a localized integrity/identity defect and rerun unchanged;
repair the prior/oracle/composition path if the oracle fails; if cutoff does not beat both
baselines, stop static-cutoff work and pivot to rollout-aware training or a material
controller; a noise-like numerical win is dynamically negligible and routes on-policy;
only a target-like win permits bounded Stage F planning. Stage F machine eligibility and
all auto-launch flags stay zero.

Claim scope is one opened target-specific model, one fixed prior seed/path, and the approximate
candidate law: no exact-law, population, diversity, confirmatory, or generator claim or gate.

No production compute runs in this implementation patch. A future run is projected
at 450--650 active/GPU seconds, below 64 MiB CUDA and 8 MiB storage. Its 1,200-second
cap requires real explicit approval; amend to 1,500 only if measured projection requires.
No new cutoff, training, exact backend, or Stage F automation is added. Proxy-only patches
since the last objective-bearing experiment: 1; next mainline action must run Stage E.
This preserves `VERSION` and every legacy command/spec; despite exceeding the rough line estimate for authority closure and adversarial tests, it adds no framework, schedule DSL, low-level backend path, tuning surface, or follow-on automation.
