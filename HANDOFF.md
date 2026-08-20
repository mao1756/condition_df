# Authenticated-v2 exact continuation: terminal resource-stop research handoff

Date: 2026-08-14  
Production source SHA-256: f78ba841953e10af6ca44639b56ccfc3b4a701f4f479168e5c9de995bc02296d  
Post-terminal reporting-fix source SHA-256: 281aae8441a1cc1ce9f30221cbd87127348b7af469b80a88554dca451a53babd  
Handoff author: Codex

## 1. Program objective

Final scientific/engineering objective: establish or decisively falsify a DDPM-like
MNIST image generator based on the Eulerian/Jacobi approximation.

Concrete success artifact: a complete one-image reverse trajectory with saved
intermediate and final images, paired zero, learned-global, and source/oracle
controls under shared randomness, followed by reference-prior and multi-image tests
only if the one-image mechanism is promising.

## 2. Current milestone and distance to goal

Nearest objective-bearing milestone: a complete 512-step one-image reverse path on
the existing candidate-only approximate CUDA backend, with its first 16 steps
cross-checked against the healthy exact prefix reported here.

Current principal blocker: the certified exact backend is too expensive under the
prespecified adaptive 20% timing reserve. This successor was the second run in the
same exact-continuation family to stop at a resource-admission boundary.

Last objective-bearing experiment and date: production-v2-prefix-successor-v1,
2026-08-14 JST.

Artifact produced: a healthy exact 16-step reverse suffix containing an imported
eight-step shard and one newly generated eight-step shard. The run is sealed at:

    runs/experiment12-d0-jacobi-rb-global-dilated-continuation/production-v2-prefix-successor-v1

Proxy-only patches since the last objective-bearing experiment: 0.

What remains untested end to end: the remaining 496 reverse steps, a complete
reconstruction or generated image, long-horizon learned-control behavior,
reference-prior behavior, multiple images, fidelity/diversity, and confirmation.

## 3. Strategy review

Strategy status: major modification. Stop the exact-continuation lineage and switch
the next exploratory objective-bearing run to the existing approximate CUDA backend.

Rationale: v2 stopped before generating a successor reverse shard; this approved
22,500-second successor generated one additional healthy shard but stopped before
shard 2. A third provenance successor or a post-hoc relaxation of the 20% reserve
would spend more effort on exact infrastructure without answering the complete-path
question. Exactness should audit a promising complete mechanism, not prevent that
mechanism from being tested.

Strongest alternative strategy: a conventional MNIST DDPM reconstruction sanity
baseline, using the same held-out image and saving comparable milestones.

Evidence that would change this decision: a complete approximate path with a
material learned advantage and a passing oracle control would justify requesting a
new budget for a frozen exact-reference audit. A failed oracle would instead direct
work to composition/backend correctness. A passing oracle with a weak learned path
would direct work to the learner, scale, or on-policy training.

## 4. Research mode and evidence roles

Primary mode: engineering/infrastructure for the authenticated-prefix implementation,
immediately followed by an exploratory objective-bearing run.

Training/development evidence: frozen update 3100 and its opened lineage.

Exploratory-validation evidence: opened path 1028864, the exact v3 parent, v2
predecessor, and this successor.

Protected confirmation evidence: unopened and unused.

Evidence already opened or reused: the source image and mixed target, all 64 v2
forward shards, imported reverse shard 0, and the selected checkpoint. No retraining,
checkpoint reselection, gain tuning, path reselection, or protected-evidence opening
occurred.

## 5. Exact result of the latest run

Design: one opened image/path, paired zero, learned-global, and source/oracle rows,
shared randomness, certified exact fused reverse backend, eight reverse steps per
shard, 64 planned reverse shards, and a hard active-time cap of 22,500 seconds.

Terminal outcome: resource_boundary at reverse shard index 2, before that shard's
sampler was called. The scientific objective is incomplete and the sealed child is
nonresumable.

The successor authenticated and independently copied:

- all 64 v2 forward shards;
- v2 reverse shard 0;
- the 16-file v2 provenance capsule;
- the immutable v3 parent and source bindings;
- the exact carried active cost of 1945.6831628999998 seconds.

It then generated reverse shard 1. Reverse shards 0 and 1 jointly cover 16 of 512
reverse steps. Both pass strict health:

- 526,848 of 526,848 active transitions certified and authorized;
- certificate fraction 1.0;
- fallback, forbidden, nonfinite, and negative counts all zero;
- maximum mass error 4.440892098500626e-16.

Resource admission was correct. Before reverse work:

- authenticated carry: 1945.6831628999998 seconds;
- prepare: 9.43122840000433 seconds;
- controls: 7.483725400001276 seconds;
- forward validation-only stage: 7.082786400002078 seconds;
- active total A: 1969.6809031000075 seconds.

Observed reverse times were 252.79023189999862 and 265.82845950000046 seconds.
The frozen next-shard bound was:

    R_next = 1.2 * max(observed) = 318.99415140000053 seconds

With 62 shards remaining and 630 seconds reserved for postprocessing and reporting,
the admission projection was 22377.318289900042 + W. Even the conservative bound
W >= 265.82845950000046 gives 22643.146749400043 seconds, 143.146749400043
seconds over the approved cap. The reconciled terminal ledger is
2245.3719581000078 seconds with no actual resource breach; the stop was prospective.

Exploratory partial metrics at 16 completed reverse steps follow.

Primary mixed-target squared-L2:

| Row | Error | Improvement versus zero |
|---|---:|---:|
| zero | 0.0029733907954117736 | — |
| learned-global | 0.0029675479817521536 | 0.1965033882742893% |
| source/oracle | 0.0002394655253201298 | 91.94638236959474% |

Secondary pure-source-image squared-L2:

| Row | Error | Improvement versus zero |
|---|---:|---:|
| zero | 0.005598575911869436 | — |
| learned-global | 0.005576005803899502 | 0.4031401614486156% |
| source/oracle | 0.0014497146796270052 | 74.1056528937387% |

The learned row is directionally better than zero but below the frozen 1% practical
threshold. Both zero and learned errors worsened from step 8 to step 16, with learned
worsening less; the oracle improved strongly.

### This result establishes

A healthy certified exact 16-step same-path suffix for one opened image/path. At that
scope, learned-global is slightly better than zero and source/oracle is strongly
better. The approved exact continuation could not preserve its prespecified
resource reserve before shard 2.

### This result does not establish

It does not establish complete-path learned success or failure, image generation,
reference-prior behavior, population performance, diversity, confirmatory validity,
or that the Jacobi/RB strategy should continue unchanged.

## 6. Confirmed facts, current inferences, and open hypotheses

### Confirmed facts

- The exact forward chain has 64 contiguous healthy shards.
- The exact reverse prefix has two contiguous healthy shards and 16 completed steps.
- Reverse shard 2 was not sampled or committed.
- The learned partial effect is positive but practically small at 16 steps.
- The oracle partial effect is large and positive.
- The terminal ledger remains below every actual cap; admission stopped on projected
  remaining cost.
- The terminal verifier passed and all four audited trees were unchanged.
- The sealed child REPORT.md and child HANDOFF.md incorrectly list eight complete-run
  artifacts that do not exist after this early stop. The child remains immutable;
  this external handoff is the correction.

### Current inferences

- The strong oracle result makes a gross orientation error less likely over the
  first 16 steps, but does not validate full-horizon composition.
- Learned control currently has a weak useful direction, not persuasive practical
  scale.
- Exact-backend cost, specifically the compounded 20% reserve, is now obstructing
  the higher-value complete-path decision.
- A complete approximate trajectory has greater decision value than another exact
  budget/provenance patch.

### Open hypotheses

- A composition, controller-interface, or approximate-backend error may appear only
  at longer horizons.
- The learned controller may have inadequate amplitude, calibration, architecture,
  or receptive field.
- Forward/off-policy quality may fail on the learned rollout distribution.
- The learned effect may remain directionally positive but dynamically negligible.
- Late schedule segments may accumulate error even if short horizons pass.
- Terminal/reference-prior mismatch may dominate after same-path reconstruction.
- The 1% diagnostic threshold may be misaligned with visible output quality.
- The Jacobi/RB research strategy itself may have lower value than a conventional
  diffusion baseline.

## 7. Decision the next patch must resolve

Does a complete 512-step one-image path on the existing candidate-only approximate
CUDA backend preserve a valid oracle trajectory and produce a materially better
learned-global reconstruction than the paired zero control?

## 8. Candidate actions and value of information

| Candidate | Approximate cost | Decision value |
|---|---:|---|
| Complete approximate paired path | Up to one GPU-hour planning estimate | Directly tests the nearest objective and separates short from long horizon |
| Third exact-budget successor | More exact infrastructure plus roughly 5–6 GPU-hours projected | Low: repeats the same resource-stop family before feasibility is known |
| Read-only decomposition of the 16-step prefix | Minutes | Low: cannot resolve complete-path behavior |
| Retrain or tune immediately | Hours to days | Premature: composition and long-horizon behavior remain unknown |
| Conventional DDPM sanity baseline | One bounded exploratory run | High if the approximate Jacobi/RB path is invalid or uninformative |

The smallest decisive experiment is the complete approximate paired path. A shorter
suffix is cheaper but cannot answer the long-horizon decision; the exact 16-step
reference already supplies the relevant short-horizon check.

## 9. Recommended next patch

Why it has the highest decision value: it exercises the assembled mechanism to a
visible task artifact and uses the existing exact prefix as a numerical cross-check,
without paying exact-reference cost for an unproven complete mechanism.

What it will implement or execute: one complete 512-step same-path exploratory path
on the existing candidate-only approximate CUDA backend.

Objective-bearing artifacts it will save:

- the starting state and source/mixed targets;
- zero, learned-global, and source/oracle trajectories;
- every prespecified milestone and all final images, including bad outputs;
- shared randomness identifiers;
- controller magnitude relative to stochastic increments;
- health, conservation, fallback, and boundary diagnostics;
- first-16-step comparison against the exact prefix;
- exact configuration, command, checkpoint, source closure, and compact report.

Controls and baselines: zero control, source/oracle known-positive control, current
learned-global controller, shared randomness, and exact-prefix numerical reference
for the first 16 steps.

Primary metrics: complete-path mixed-target squared-L2 and saved reconstruction
images. Mechanism diagnostics: horizon errors, relative effects, controller scale,
and on-policy drift. Health metrics: finiteness, conservation, fallback,
boundary validity, runtime, CUDA memory, and storage.

What it will not claim: confirmation, reference-prior generation, population
performance, diversity, or exact-backend equivalence beyond the audited prefix.

## 10. Gates and claim boundaries

### Gate A: numerical health

Gate type: execution/integrity gate.  
Downstream action or claim controlled: interpretation of any trajectory after the
first invalid boundary.  
Exact proposition tested: states, controller diagnostics, conservation telemetry,
and shared-randomness bindings remain finite, authorized, and internally consistent.  
Why necessary: corrupted dynamics cannot support even an exploratory mechanism
interpretation.  
Statistic and independent unit: every committed boundary on the one path.  
Pass condition: exact schema/binding checks pass, no forbidden/nonfinite/negative
state occurs, and declared conservation tolerances hold. Certified conservative
fallback is allowed and must be reported.  
Failure means: evidence through the last valid boundary is retained, but later
trajectory interpretation stops.  
Failure does not mean: the learner or full research strategy is globally false.  
Pass action: continue to the next boundary.  
Fail action: stop, package all partial states/images, and fix the concrete defect.  
Ambiguous/invalid action: retain evidence and rerun only after the validity question
is resolved.

### Gate B: resource safety

Gate type: execution/integrity gate.  
Downstream action or claim controlled: launching the next sampler or terminal write.  
Exact proposition tested: projected wall time plus declared postprocessing/report
reserve, persisted storage, and CUDA memory remain within an explicitly approved
budget.  
Why necessary: the experiment must stop safely and preserve a truthful terminal
package.  
Statistic and independent unit: cumulative active wall time, remaining-boundary
projection, recursive bytes, and current-device CUDA fraction.  
Pass condition: within a separately approved future cap, storage below 2 GiB, CUDA
below 0.80, and all mandatory reserves preserved.  
Failure means: automatic resource stop before further compute.  
Failure does not mean: scientific failure of the controller.  
Pass action: continue.  
Fail action: save and report the partial objective; do not silently raise a cap.  
Ambiguous/invalid action: reconcile durable evidence before any retry.

### Gate C: source/oracle composition control

Gate type: execution/integrity gate controlling learned interpretation.  
Downstream action or claim controlled: attributing complete-path behavior to the
learned controller.  
Exact proposition tested: source/oracle control is finite, improves over zero in the
positive direction, and reaches at least the prespecified practical scale.  
Why necessary: if a known-positive controller cannot traverse the same composition
path, learned-controller attribution is uninterpretable.  
Statistic and independent unit: paired complete-path mixed-target squared-L2 for the
one exploratory path.  
Pass condition: positive relative improvement of at least 1%.  
Failure means: inspect composition, backend, schedule, and controller interface.  
Failure does not mean: the learned predictor lacks all useful signal.  
Pass action: interpret learned-versus-zero direction and scale exploratorily.  
Fail action: fix or replace the system path before retraining the learner.  
Ambiguous/invalid action: preserve outputs and run a bounded composition sanity test.

### Threshold D: learned practical scale

Gate type: diagnostic threshold.  
Downstream action or claim controlled: selection of the next exploratory action, not
execution and not a confirmatory claim.  
Exact proposition tested: learned-global improves complete-path mixed-target error
over zero by at least 1%.  
Why necessary: the threshold distinguishes a potentially useful effect from a
directionally positive but small effect for planning.  
Statistic and independent unit: paired relative improvement on the one opened path.  
Pass condition: at least 1%.  
Failure means: the observed complete-path effect is below the selected practical
scale.  
Failure does not mean: no signal, no local benefit, or universal learner failure.  
Pass action: consider an exact reference audit and then broader exploration.  
Fail action: inspect horizon, scale, representation, and on-policy behavior or pivot.  
Ambiguous/invalid action: use images and horizon traces; do not convert this
diagnostic into an execution veto.

There is no confirmatory claim gate in the next exploratory patch.

## 11. Outcome-to-action table

| Outcome | Interpretation | Required next action |
|---|---|---|
| Oracle fails or becomes invalid | Composition, controller interface, schedule, or approximate backend is not valid | Fix the system before changing the learner |
| Oracle passes; learned is adverse or negligible | Learner, representation, scale, or training distribution is suspect | Change learner/scale/on-policy data; do not default to more exact infrastructure |
| Short exact-prefix agreement passes; long horizon fails | Accumulation, late schedule, or on-policy shift | Localize by horizon and revise schedule/controller |
| Complete learned path materially beats zero and images are plausible | Mechanism has exploratory feasibility evidence | Request budget for a frozen exact-reference audit |
| Approximate first 16 steps disagree materially with exact prefix | Backend discrepancy matters | Audit the fixed prefix before using approximate full-path conclusions |
| All Jacobi/RB controls are uninformative | Current strategy lacks feasibility evidence | Run the conventional DDPM sanity baseline or stop the hypothesis |
| Result cannot separate these branches | Experiment is underdesigned | Do not scale it; redesign the direct comparison |

## 12. Constraints

### Integrity constraints

- Do not mutate or resume the sealed successor child. Rationale: it is an immutable,
  verified failure terminal. Scope: that child only. Review trigger: none; create a
  new run for new science.
- Do not silently open protected confirmation evidence. Rationale: preserve future
  claim validity. Scope: exploratory design and tuning. Review trigger: a frozen
  confirmatory protocol.
- Preserve shared-randomness pairing, row identities, exact-prefix bytes, failed
  outputs, and source/checkpoint bindings. Rationale: without them the system
  comparison is not interpretable. Scope: the next paired path.
- Do not auto-raise the 22,500-second cap or launch another exact successor.
  Rationale: the user approval applied to the completed successor and the
  prespecified branch stops this lineage after a second resource stop. Review
  trigger: a promising complete approximate result plus new explicit approval.

### Revisable scientific and engineering choices

The architecture, features, loss, optimizer, controller gain, schedule, inference
family, approximate backend, practical threshold, on-policy training strategy, and
Jacobi/RB hypothesis remain revisable. The approximate backend is appropriate for
exploration; exactness is required only when a later claim depends on it.

## 13. Resource budget and stop rule

Completed successor:

- approved hard active-time cap: 22,500 seconds;
- terminal active ledger: 2245.3719581000078 seconds;
- terminal storage: 3,246,841 bytes;
- peak CUDA fraction: approximately 0.00548;
- production-frozen runtime/test complexity: 4497/2998 physical lines;
- current reporting-fix runtime/test complexity: 4498/2999 physical lines.

Next-patch planning envelope, not launch authorization:

- expected wall and accelerator time: at most one GPU-hour for a complete
  candidate-only approximate path;
- expected peak CUDA: below 0.80 of the current device;
- expected persisted storage: well below 2 GiB, including all images and failures;
- new source/test complexity: prefer a compact separate exploratory runner and tests,
  with no growth of the already near-cap exact continuation module;
- maximum compute budget: must be explicitly approved before launch;
- automatic stops: invalid health, oracle invalidity, CUDA at or above 0.80, storage
  at or above 2 GiB, or inability to preserve reporting reserve;
- maximum proxy-only continuation: zero additional proxy-only patches before the
  complete objective-bearing run.

Scientific decision purchased: whether the assembled one-image mechanism works over
the complete path and where it fails if not. The exact 16-step prefix cannot answer
the long-horizon question, and another smaller proxy has lower decision value.

## 14. Alternative and pivot plan

If the complete approximate experiment is invalid or scientifically uninformative,
run a conventional MNIST DDPM reconstruction sanity baseline on the same held-out
image with comparable zero/reference context and saved milestones. If that baseline
works while Jacobi/RB does not, substantially modify or stop the Jacobi/RB route. If
both fail, audit the common data, checkpoint, and reconstruction setup before more
model work.

## 15. Evidence map

All paths below exist at handoff time.

Production child root:

    runs/experiment12-d0-jacobi-rb-global-dilated-continuation/production-v2-prefix-successor-v1

Load-bearing files and roles:

- terminal_failure.json — terminal failure domain, incomplete objective, and
  nonresumable status; raw/semantic SHA-256
  b6326815016ae073d329f5a532a8d615a15fea78dd11c7085f7468e5fa35d75c /
  439e1b0901377d9d3704c1c73002c1a4349af6f8613331beeff63aa02ce3593f.
- verification.json — final terminal verification; semantic SHA-256
  e5276ecb1a0f584447fda8556b2f756fd2ebe061bed3a3ae6ad29c2286b63daa.
- resource_ledger.json — exact carry, child work, caps, and terminal resource state;
  file SHA-256 0bf965417296c94ae0e57abe7ed4b8e019aef4d3c6ca4533f03e282525338bc8.
- last_valid_evidence.json — exact inventory of all usable raw and partial evidence.
- artifact_manifest.json — 174 terminal inventory rows; file/semantic SHA-256
  af10daa0b23a4cf2da2511976a6c5818b67b24a85540c1c4a41d8d2596f80ac0 /
  fd58aac31318d3a21c49cadafbddb3d7ae9ddb943f57b28b14d7603957943786.
- SHA256SUMS.txt — 175 checksum rows; file SHA-256
  2d55e10ddc9cc0054ea3cb3c84567022b306af1d453a9081ba5c058201c8647e.
- predecessor_binding.json — authenticated v2 prefix tree and pinned authorities.
- parent_binding.json — immutable v3 parent and source closure.
- forward/forward_summary.json — exact 64-shard imported forward aggregate.
- reverse/fused_families/same-path-three-row/complete-512/shard-0000.json and
  shard-0000.npz — authenticated imported exact reverse shard.
- reverse/fused_families/same-path-three-row/complete-512/shard-0001.json and
  shard-0001.npz — child-generated exact reverse shard; JSON/NPZ SHA-256
  f89f899f97c82da2c6facf215dc9b2d0aaabae4fdced387cdabb1f101e60271d /
  b4d5415d48118da8cbaaaf562cc827705db03548930bf644ea9d36390083bee3.
- reverse/fused_families/same-path-three-row/complete-512/shard-0002.failure.json
  — pre-sampler resource-stop evidence; SHA-256
  15b3b39226550f8874961547ed8a795a3b818c73d201323b50e4faade8857bc3.

External immutable roots:

- v2 prefix: production-same-path-complete-v2, 158 files, 2,887,822 bytes, tree
  SHA-256 2a88fc8da20188f19a329e5bf3e9fd236d9ba30141e8710feedac2d26a9326f3.
- v3 parent: 20260813-233915_production-global-dilated-exact-five-row-v3,
  212 files, 12,520,738 bytes, tree SHA-256
  2790043c1363cf9f75b7c64bce0b2792c3afa8622b660d2a92f69107d511c452.
- source binding: 20260813-002414_production-frequency1-objective-first-recovery-v4/input_bindings,
  6 files, 188,287 bytes, tree SHA-256
  eea1b7d269649c5c14bd85926e448ad3b4529f8a48eecb93bad4db9864bd6482.

Implementation:

- mnist/diag_d0_jacobi_rb_global_dilated_continuation.py — current post-terminal
  source SHA-256 281aae8441a1cc1ce9f30221cbd87127348b7af469b80a88554dca451a53babd,
  4498 physical lines.
- tests/test_diag_d0_jacobi_rb_global_dilated_continuation.py — current SHA-256
  6032d15a732355fccd314a4afe4ee12dc3ec8d4c83d4827edf2f985b1ef319f8,
  2999 physical lines.
- docs/jacobi_rb_global_dilated_continuation.md — plan/contract SHA-256
  bc9008494e3f13dab76457c3974239f8cb0b9d9a8571b899dad1f6f1bc69b782.

## 16. Deliberate omissions

The resource stop occurred before derived postprocessing. These paths do not exist:

- reverse/trajectory_shard_boundaries.npz;
- reverse/milestones.npz;
- reverse/metrics.csv;
- reverse/mechanism.json;
- reverse/summary.json;
- reverse/family_summary.json;
- outcome.json;
- images/.

Their absence prevents independent auditing of a complete trajectory, milestone
images, complete-path metrics, mechanism trace, or outcome classification. The
sealed child REPORT.md and child HANDOFF.md incorrectly list these eight absent
paths as key artifacts. They were not edited because the child is immutable. The
current runtime fixes future failure reports by listing only actual last-valid
paths, and the focused suite covers that regression.

No complete reverse image, reference-prior sample, multi-image result, uncertainty
interval, or protected confirmation evidence exists.

## 17. Reproduction commands

Original production command:

    .\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_global_dilated_continuation --stage all --repository-root C:\Users\mao17\Workspace\condition_df --runs-root C:\Users\mao17\Workspace\condition_df\runs\experiment12-d0-jacobi-rb-global-dilated-continuation --run-name production-v2-prefix-successor-v1 --prefix-run-dir C:\Users\mao17\Workspace\condition_df\runs\experiment12-d0-jacobi-rb-global-dilated-continuation\production-same-path-complete-v2 --parent-run-dir C:\Users\mao17\Workspace\condition_df\runs\experiment12-d0-jacobi-rb-global-dilated-rollout\20260813-233915_production-global-dilated-exact-five-row-v3 --source-run-dir C:\Users\mao17\Workspace\condition_df\runs\experiment12_d0_jacobi_rb_frequency1_rollout\20260813-002414_production-frequency1-objective-first-recovery-v4\input_bindings --device cuda

Historical read-only verification invocation:

    .\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_global_dilated_continuation --verify-run-dir C:\Users\mao17\Workspace\condition_df\runs\experiment12-d0-jacobi-rb-global-dilated-continuation\production-v2-prefix-successor-v1 --repository-root C:\Users\mao17\Workspace\condition_df --prefix-run-dir C:\Users\mao17\Workspace\condition_df\runs\experiment12-d0-jacobi-rb-global-dilated-continuation\production-same-path-complete-v2 --parent-run-dir C:\Users\mao17\Workspace\condition_df\runs\experiment12-d0-jacobi-rb-global-dilated-rollout\20260813-233915_production-global-dilated-exact-five-row-v3 --source-run-dir C:\Users\mao17\Workspace\condition_df\runs\experiment12_d0_jacobi_rb_frequency1_rollout\20260813-002414_production-frequency1-objective-first-recovery-v4\input_bindings

That direct invocation was run successfully before the reporting-only source fix.
It is historical, not executable against the current module because the sealed child
requires production source SHA-256
f78ba841953e10af6ca44639b56ccfc3b4a701f4f479168e5c9de995bc02296d.

The following current-workspace command reconstructs those exact production bytes
entirely in memory, asserts their size and hash, recomputes the full 26-file source
closure with the reconstructed entry bytes, and runs the production verifier. It
does not write the reconstructed source or mutate any of the four evidence trees:

    @'
    import base64, hashlib, sys, types
    from pathlib import Path

    root = Path(r"C:\Users\mao17\Workspace\condition_df").resolve()
    entry = root / "mnist/diag_d0_jacobi_rb_global_dilated_continuation.py"
    current = entry.read_bytes()
    production = b"".join(
        line for line in current.splitlines(keepends=True)
        if not line.startswith(b"    artifact_paths = ")
    )
    original_report_literal = base64.b64decode(
        "S2V5IGFydGlmYWN0czogYHJldmVyc2UvdHJhamVjdG9yeV9zaGFyZF9ib3VuZGFyaWVzLm5wemAsIGByZXZlcnNlL21pbGVzdG9uZXMubnB6YCwgYHJldmVyc2UvbWV0cmljcy5jc3ZgLCBgcmV2ZXJzZS9tZWNoYW5pc20uanNvbmAsIGByZXZlcnNlL3N1bW1hcnkuanNvbmAsIGByZXZlcnNlL2ZhbWlseV9zdW1tYXJ5Lmpzb25gLCBgb3V0Y29tZS5qc29uYCwgYGltYWdlcy9gLCBgcHJlZGVjZXNzb3JfYmluZGluZy5qc29uYCwgYHJlc291cmNlX2xlZGdlci5qc29uYC4="
    )
    production = production.replace(
        b"Artifact paths: {artifact_paths}.", original_report_literal
    )
    expected = "f78ba841953e10af6ca44639b56ccfc3b4a701f4f479168e5c9de995bc02296d"
    assert len(production) == 265355
    assert hashlib.sha256(production).hexdigest() == expected

    name = "mnist._production_v2_prefix_successor_verifier"
    module = types.ModuleType(name)
    module.__file__ = str(entry)
    module.__package__ = "mnist"
    sys.modules[name] = module
    exec(compile(production, str(entry), "exec"), module.__dict__)

    def exact_closure(repository_root):
        rows = {}
        for path in module.v3_transitive_source_paths((entry,)):
            path = path.resolve()
            raw = production if path == entry else path.read_bytes()
            relative = path.relative_to(repository_root).as_posix()
            rows[relative] = {
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        return rows, module.semantic_sha256(rows)

    module._current_source_closure = exact_closure
    rc = module.main([
        "--verify-run-dir", str(root / "runs/experiment12-d0-jacobi-rb-global-dilated-continuation/production-v2-prefix-successor-v1"),
        "--repository-root", str(root),
        "--prefix-run-dir", str(root / "runs/experiment12-d0-jacobi-rb-global-dilated-continuation/production-same-path-complete-v2"),
        "--parent-run-dir", str(root / "runs/experiment12-d0-jacobi-rb-global-dilated-rollout/20260813-233915_production-global-dilated-exact-five-row-v3"),
        "--source-run-dir", str(root / "runs/experiment12_d0_jacobi_rb_frequency1_rollout/20260813-002414_production-frequency1-objective-first-recovery-v4/input_bindings"),
    ])
    print({
        "passed": int(rc == 0),
        "return_code": rc,
        "production_source_sha256": hashlib.sha256(production).hexdigest(),
    })
    raise SystemExit(rc)
    '@ | .\.venv\Scripts\python.exe -

Expected output:

    {'passed': 1, 'return_code': 0, 'production_source_sha256': 'f78ba841953e10af6ca44639b56ccfc3b4a701f4f479168e5c9de995bc02296d'}

The current source intentionally has a different closure because it corrects future
failure-report path disclosure. The in-memory reconstruction proves that this
single logical reporting change is the complete difference from production bytes.

Current implementation checks:

    .\.venv\Scripts\python.exe -m py_compile mnist\diag_d0_jacobi_rb_global_dilated_continuation.py tests\test_diag_d0_jacobi_rb_global_dilated_continuation.py
    .\.venv\Scripts\python.exe -m pytest tests\test_diag_d0_jacobi_rb_global_dilated_continuation.py -q -rs --basetemp .tmp-successor-recipient

Expected focused result: 148 passed, 1 skipped. The sole skip is Windows symlink
creation unavailable; real v2/v3 audits are not skipped.

## 18. Bundle-integrity audit

Verification command: the read-only command in section 17 returned exit code 0 on
the exact production closure.

Expected child inventory: 178 files, 3,246,841 bytes, tree SHA-256
83c87a037c4ad65852db1c7b51d03476e43372b1e3333ac75bd7052c8eb6b5a4.

Expected manifest/checksum inventory: 174 manifest artifacts and 175 checksum rows.

Expected nonempty directories: forward, imports, inputs, reverse, and stages.
Journals is intentionally empty at terminal.

Manifest/hash locations: artifact_manifest.json and SHA256SUMS.txt at the production
child root.

Representative files opened successfully: imported and child-generated reverse NPZ
states, all 64 forward pairs, resource ledger, terminal failure, verification,
manifest, checksums, v2 capsule, v3 parent, and source input NPZ.

Four-tree read-only audit: child 178 files, v2 158, v3 212, source 6; zero of 554
files changed across verification.

Recipient-path caveat: the eight nonexistent paths in the sealed child reports are
explicitly corrected in sections 15–16. All paths affirmatively cited as evidence in
this external handoff were checked for existence.

## 19. Exact deliverable for the receiving agent

Delivered:

1. the implemented authenticated-v2-prefix successor with the explicitly approved
   22,500-second cap;
2. its focused tests and executable contract;
3. the immutable, independently verified terminal resource-stop child;
4. the post-terminal failure-report path fix and regression test;
5. this implementation and scientific handoff.

The receiving agent must produce an implementation-ready, objective-bearing plan for
one complete approximate CUDA path with paired zero/global/oracle controls and saved
milestones, then request any new compute approval. A positive result must lead to a
frozen exact-reference audit; a negative or invalid result must trigger the
outcome-to-action branch in section 11. Do not resume this child, raise the old cap,
or build a third exact-continuation successor without new evidence and explicit user
approval.
