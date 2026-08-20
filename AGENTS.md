# AGENTS.md

# IMPORTANT
- Do not write overly defensive code. Always prefer simplicity over pathological complexity.
- Remember the YAGNI principle.
- Delete temporary files (such as .`pytest-*`, .`.tmp-*`) after you make it.

## Purpose and scope

This file governs research planning, experiment design, implementation, reporting,
and agent-to-agent handoffs in this repository.

The project goal is not to accumulate locally valid diagnostics. The goal is to
establish, or decisively falsify, a **DDPM-like MNIST image generator based on the
Eulerian approximation**. The Jacobi transition, Rao--Blackwell target, learned
controller, exact certificates, and statistical gates are means to that end.

The precedence order is:

1. the user's current explicit instruction;
2. this `AGENTS.md`;
3. verified source code and immutable evidence;
4. the current `HANDOFF.md`;
5. older plans, reports, and inherited conventions.

`HANDOFF.md` is a fallible scientific proposal and evidence summary. It is not an
unquestionable specification. A receiving agent must audit its framing before
following it. Detail, provenance, or the word “immutable” does not make a scientific
choice correct.

## Prime directives

1. **Keep the objective visible.** Every substantial patch must state how it moves
   the project toward, away from, or to a decision about the end-to-end generator.
2. **Test the assembled mechanism early.** Once basic correctness controls pass, run
   the smallest interpretable end-to-end experiment before adding exhaustive local
   certification.
3. **Separate exploration from confirmation.** Strict confirmatory rules control
   claims and protected evidence; they do not normally forbid cheap exploratory
   rollouts on separate data.
4. **Choose experiments by decision value.** A patch must change a substantive next
   action, not merely authorize another plan or narrower diagnostic.
5. **State results at exactly the tested scope.** A failed strong gate is not evidence
   for every broader negative interpretation.
6. **Save failures.** Bad images, unstable trajectories, negative controls, and
   intermediate states are scientific evidence.
7. **Use rigor proportionately.** Exactness, provenance, and restartability must
   protect a concrete claim or observed failure mode. They must not indefinitely
   postpone feasibility testing.
8. **Permit a pivot.** Architecture, loss, features, inference family, controller,
   numerical backend, and even the Jacobi/RB strategy remain revisable scientific
   choices.

## 1. Required research modes

Every experiment, patch plan, and handoff must declare one primary mode.

### 1.1 Exploratory

Purpose: determine whether a mechanism can work, compare alternatives, localize a
failure, or choose a promising setting.

Exploratory work may use:

- post-hoc checkpoint inspection;
- reused development or exploratory-validation evidence;
- gain, schedule, and hyperparameter sweeps;
- approximate or reduced-scale numerical backends;
- adaptive ablations;
- visual inspection and direct comparison of failed outputs.

All post-hoc choices must be disclosed. Exploratory results may guide the next
experiment, but they must not be presented as confirmatory evidence.

### 1.2 Confirmatory

Purpose: support a prespecified scientific claim.

Confirmatory work requires, as appropriate:

- a frozen claim and model-selection procedure;
- fixed primary metrics, units of analysis, thresholds, and stopping rules;
- protected or fresh evidence disjoint from development evidence;
- prespecified multiplicity handling;
- the reference numerical backend when the claim depends on it;
- no tuning after opening confirmation evidence.

A failed confirmation does not become successful through post-hoc rescue. The opened
confirmation evidence may thereafter be treated as exploratory evidence, and a new
confirmation must be designed with new protected evidence when the claim requires it.

### 1.3 Forensic

Purpose: explain a completed run using existing immutable evidence.

A forensic patch is allowed only when each plausible outcome maps to a different
concrete action. It must not become an indefinite substitute for running the system.
A read-only decomposition whose every outcome leads to another read-only
adjudication has insufficient decision value.

### 1.4 Engineering, control, or infrastructure

Purpose: remove a demonstrated correctness, reliability, resource, or reproducibility
blocker.

The blocker must be named precisely. The patch must end with the smallest experiment
showing that the blocker is removed. Infrastructure work is not automatically
scientific progress and must be budgeted as such.

## 2. Objective-bearing milestones

An **objective-bearing experiment** directly exercises a material part of the final
mechanism and produces an artifact visible at the task level. For this repository,
examples include:

- a reverse suffix from a held-out forward state;
- a complete reverse trajectory;
- a reconstruction or generated MNIST image;
- a teacher-controlled full reverse path;
- a direct comparison with a zero controller or alternative generator;
- a system-level result that decisively localizes where composition fails.

The following are proxies, not objective-bearing artifacts by themselves:

- local prediction loss;
- componentwise risk tables;
- cross-energy or alignment decompositions;
- certificate counts;
- artifact-registry size;
- one-step transition accuracy;
- a plan that authorizes another plan.

A failed objective-bearing experiment still counts as useful progress when it is
interpretable and its outputs are saved.

### End-to-end cadence

Maintain this counter in every handoff:

```text
Proxy-only patches since the last objective-bearing experiment: <integer>
```

After two consecutive proxy-only patches, the next mainline patch must do one of the
following:

1. run an objective-bearing experiment; or
2. repair one concrete defect that makes such an experiment uninterpretable, then
   run the objective-bearing experiment immediately afterward.

“Not authorized by an earlier gate,” “more confidence would be useful,” and “another
decomposition may be informative” are not concrete blockers.

## 3. Mandatory experiment-design workflow

Before implementation or compute begins, write a compact experiment specification
covering the following items.

### 3.1 One decision, not a topic

State one decision in a form that changes the next action. Examples:

- Does the current learned controller improve a reverse suffix over zero control?
- Is failure caused by controller composition or by the learned predictor?
- Does a global architecture materially outperform the local model?
- Is the exact backend necessary for the observed behavior?
- Should the current Jacobi/RB route continue, be substantially modified, or stop?

“Understand the failure better” is not a sufficient decision.

### 3.2 Competing hypotheses

List the leading explanations and the observation each predicts. The list must remain
open to framing failure. When applicable, include:

- an implementation or orientation bug;
- failure of the controller, integrator, splitting, or interface;
- an inadequate architecture or receptive field;
- insufficient prediction amplitude or incorrect calibration;
- a useful aggregate signal with some negative local components;
- an effect that is statistically real but dynamically negligible;
- forward/off-policy success but on-policy distribution shift;
- terminal/reference-prior mismatch;
- a proxy metric or gate misaligned with the final objective;
- excessive approximation error or an unnecessarily expensive exact backend;
- failure of the current research strategy itself;
- another explanation supported by the evidence.

Do not force the next agent to choose only among minor repairs proposed by the prior
agent.

### 3.3 Outcome-to-action table

Define the action before running the experiment.

| Possible observation | Interpretation | Required next action |
|---|---|---|
| Positive control fails | System composition or implementation is invalid | Fix the system before changing the learner |
| Positive control passes; learned controller fails | Learner, representation, scale, or training distribution is suspect | Change the learner or data, not the integrator by default |
| Short horizon passes; long horizon fails | Accumulation, late-time, or on-policy error | Localize by horizon and revise schedule/controller |
| Direct output works despite proxy-gate failure | The proxy gate is too strong or misaligned | Redesign the claim and validation around relevant metrics |
| Approximate backend passes; reference backend fails | Backend discrepancy matters | Audit and rerun the fixed case |
| All tested mechanisms fail | Current strategy lacks feasibility evidence | Run the stated pivot comparison or stop |
| Result cannot distinguish hypotheses | Experiment was underdesigned | Do not scale it; redesign the experiment |

If materially different outcomes all lead to the same next patch, the experiment has
low decision value.

### 3.4 Choose the smallest decisive experiment

Prefer, in order:

1. a no-compute replay of already sufficient evidence;
2. a tiny deterministic or synthetic smoke test;
3. a short-horizon paired system test;
4. a reduced-scale end-to-end run;
5. a full exploratory run;
6. a large exact or confirmatory run.

A cheaper experiment is preferred only when it can change the decision. Do not replace
a decisive system test with a cheap but irrelevant proxy.

### 3.5 Define artifacts before metrics

State what must be saved even on failure. For learned stochastic dynamics, normally
save:

- inputs or starting states;
- shared random seeds or random-bit identifiers;
- null, baseline, teacher, and learned trajectories;
- intermediate states at prespecified anchors;
- final images or reconstructions;
- controller magnitude relative to stochastic/noise increments;
- conservation, boundary, fallback, and numerical-failure diagnostics;
- the exact configuration, source revision, commands, and selected checkpoint;
- a compact human-readable report.

Do not hide failed task artifacts behind a scalar gate.

## 4. Controls and comparisons

Every experiment must use the controls needed to identify its failure mode, not merely
the controls easiest to certify.

### 4.1 Minimum control stack

For the Jacobi/RB generator, the default stack is:

1. **Null control:** zero learned control with otherwise identical randomness.
2. **Known-positive control:** an analytic or synthetic teacher passed through the
   same complete controller and reverse-composition code.
3. **Current baseline:** the best existing learned system under a frozen, disclosed
   selection rule.
4. **Candidate intervention:** the proposed change.
5. **Numerical reference:** the exact/reference backend for a fixed subset when
   approximation error could alter the conclusion.

A local teacher-prediction test is not a substitute for a teacher-driven full-system
trajectory. Positive controls must exercise the interface being tested.

### 4.2 Pairing and randomness

Use matched initial states and shared random numbers whenever possible. Record the
unit of independence explicitly: path, image, training seed, rollout seed, or another
unit. Do not treat correlated cells or edges as independent replicates.

### 4.3 Short and long horizons

For dynamical systems, test multiple horizons. A default exploratory design includes:

- one or more short reverse suffixes;
- an intermediate suffix;
- the complete reverse path;
- intermediate visual and numerical states.

This separates local score quality from accumulated composition error.

### 4.4 On-policy evaluation

Performance on states sampled from the forward cache does not establish performance
on states visited by the learned reverse controller. When a model will be used
recursively, evaluate it on its own rollout distribution or explicitly explain why
that is unnecessary.

## 5. Metrics, gates, and statistical claims

### 5.1 Metric hierarchy

Every experiment should distinguish:

1. **Primary objective metric:** directly reflects the milestone, such as paired
   reconstruction error, image quality, class score, sample diversity, or trajectory
   success.
2. **Mechanism diagnostics:** local denoising risk, alignment, prediction energy,
   quartile/phase behavior, controller scale, and horizon-specific error.
3. **Health metrics:** certification, conservation, runtime, memory, fallback,
   numerical stability, and artifact integrity.

A health metric may block an invalid run. A mechanism diagnostic must not silently
replace the objective metric.

### 5.2 Three kinds of gate

Every threshold must be labeled as exactly one of the following.

#### Execution/integrity gate

Blocks execution because failure would make the result unsafe, corrupted, leaked,
mathematically invalid, or uninterpretable. Examples: broken conservation, invalid
orientation, confirmation leakage, unreadable checkpoints, or a known controller
boundary violation.

#### Confirmatory claim gate

Controls whether a prespecified claim may be made. Failure blocks that claim, not all
future exploratory work.

#### Diagnostic threshold

A heuristic used to select a follow-up or describe behavior. It never becomes a
universal veto merely by being called a gate.

No untyped gate is allowed.

### 5.3 Gate specification

Before a gate may block an action, state:

```text
Gate type:
Downstream action or claim controlled:
Exact proposition tested:
Why the proposition is necessary for that action or claim:
Statistic and independent unit:
Pass condition:
Failure means:
Failure does not mean:
Pass action:
Fail action:
Ambiguous/invalid action:
```

An “all cells must pass” criterion is allowed only when every cell is genuinely
necessary for the final mechanism or exact confirmatory claim. Increasing diagnostic
resolution must not automatically make the acceptance criterion harder.

### 5.4 Negative-result language

Use this form:

```text
This run establishes:
  No candidate satisfied <exact criterion> under <data, search family,
  multiplicity rule, confidence level, and numerical backend>.

This run does not establish:
  <important broader conclusions that were not tested>.
```

Distinguish all of the following:

- point-estimate direction;
- effect magnitude and practical scale;
- unadjusted uncertainty;
- selection- or multiplicity-adjusted uncertainty;
- pooled or integrated performance;
- local componentwise performance;
- short-horizon behavior;
- full end-to-end utility.

Do not write “the learner failed,” “there is no signal,” or “later quartiles fail”
when the actual result is only failure of a stronger simultaneous criterion.

### 5.5 Statistical discipline

- Prespecify the primary confirmatory claim and its independent unit.
- Use whole-path or whole-image resampling when paths or images are the independent
  units.
- Report effect sizes and uncertainty, not only pass/fail labels.
- Separate model selection from confirmatory evaluation.
- Apply multiplicity correction to the family of claims actually intended, not every
  diagnostic cell merely because it was recorded.
- Report pooled and componentwise summaries separately when both are informative.
- Label post-hoc analyses advisory or exploratory.
- Do not interpret statistical significance as practical usefulness; compare the
  effect with the target, noise, or controller scale.

## 6. Model and intervention design

### 6.1 Change one scientific axis or design a real factorial test

A patch should normally make one principal scientific change while keeping a clear
baseline. When several changes are inseparable, use an ablation or factorial design
that can identify which change mattered.

Do not repeatedly add one nearby feature after another while leaving a more plausible
structural limitation, such as global receptive field or on-policy training,
untested.

### 6.2 Local tweaks versus major alternatives

After two targeted variants of the same idea fail to improve an objective-bearing
result, the next comparison must include at least one materially different option,
such as:

- a multiscale or global architecture;
- a different control parameterization;
- on-policy or rollout-augmented training;
- a simpler approximate numerical scheme;
- a conventional diffusion or score baseline;
- a decision to abandon the current hypothesis.

Preservation of sunk work is not evidence for continuing it.

### 6.3 Scientific choices remain revisable

The following are not immutable unless a theorem or explicit user instruction makes
them so:

- architecture and parameter count;
- receptive field and coordinate features;
- loss weighting and optimizer;
- time sampling and curriculum;
- controller gain and schedule;
- validation aggregation and inference family;
- exact versus approximate exploration backend;
- the Jacobi/RB strategy itself.

For every frozen scientific choice, record its rationale, scope, and review trigger.

## 7. Compute, backend, and complexity discipline

### 7.1 Exploration and reference backends

Maintain two clearly labeled modes when useful:

- **Exploration backend:** fast, vectorized, approximate, reduced-scale, or lower
  precision. Use it to discover whether a mechanism works and compare alternatives.
- **Reference backend:** exact or certified. Use it to audit a fixed promising case
  and support final claims.

Document all approximations. Do not generalize a conclusion beyond the backend on
which it was established. Exactness should verify a promising mechanism, not prevent
the project from finding one.

### 7.2 Resource budget

Every nontrivial patch must state:

```text
Expected wall time:
Expected accelerator time:
Expected peak memory:
Expected persisted storage:
New source/test/artifact complexity:
Maximum budget before automatic stop:
Scientific decision purchased by this cost:
Why a smaller existing experiment cannot answer it:
```

Abort or downscale when the resource projection exceeds the stated budget unless the
user explicitly approves the increase.

### 7.3 Rigor debt and infrastructure growth

New provenance schemas, registries, certificate layers, restart mechanisms, or cache
formats require a concrete justification. Add them only when they:

- protect a stated claim;
- prevent a demonstrated failure mode;
- make an otherwise infeasible decisive experiment feasible; or
- materially reduce future compute or risk.

Artifact count, code volume, and certificate depth are costs, not success metrics.

## 8. Implementation and artifact standards

### 8.1 Before a production run

At minimum, verify:

- unit tests for the changed scientific logic;
- deterministic smoke tests where appropriate;
- conservation, orientation, and boundary invariants;
- null and known-positive controls;
- model-input firewalls and data-role separation;
- exact-resume or restart behavior only when the production run needs it;
- a dry-run estimate of runtime, memory, and storage;
- the output report and task artifacts can be generated even after scientific
  failure.

### 8.2 Run artifacts

Each run should contain a compact top-level `REPORT.md` or equivalent with:

- research mode and decision question;
- source revision and scientific configuration;
- exact commands;
- evidence roles and path/image IDs;
- selected checkpoint and selection rule;
- primary objective artifacts and metrics;
- controls and health metrics;
- exact claim boundary;
- outcome-to-action decision;
- deliberate omissions.

Keep immutable raw evidence separate from derived post-hoc analysis. Derived analysis
must cite its source artifacts and may be added without mutating the original run.

### 8.3 Evidence packaging

Before sending a handoff bundle, verify it from a clean recipient's perspective:

- every referenced path exists;
- every promised directory is nonempty and contains the stated files;
- prose counts match manifests;
- hashes are computed after final assembly;
- representative arrays and checkpoints open successfully;
- commands do not depend on unstated local paths;
- deliberate omissions and the conclusions they prevent are explicit;
- the compact decision bundle contains everything needed for the requested decision.

An empty directory is not evidence. A hash proves integrity of included bytes, not
completeness of the intended bundle.

Prefer two layers:

1. a compact decision bundle for the next decision;
2. a referenced full audit archive for exhaustive provenance.

## 9. Repository-specific experimental ladder

The default ladder for this project is below. Steps may be repeated or reordered only
with a stated reason and decision table.

### Stage A: numerical and local correctness

- exact/reference transition checks;
- conservation, orientation, boundary, and stationary-null controls;
- a synthetic or analytic target that the learner can recover;
- a smoke test of the controller interface.

These establish component correctness, not generation.

### Stage B: full-system known-positive control

Drive the complete reverse controller with a known-positive teacher or oracle through
the same composition code used by the learned model. Test both short and full
horizons. Failure here is a system/controller defect and must be fixed before blaming
the learner.

### Stage C: one-image reverse-suffix reconstruction

From held-out forward states at several horizons, compare zero and learned control
with shared randomness. Save intermediate and final images. Test a small disclosed
grid of controller gains or time-window schedules exploratorily when scale is
uncertain.

### Stage D: one-image complete reverse path from a forward terminal state

This isolates long-horizon composition while avoiding uncertainty about the nominal
prior. Compare with zero control and the teacher.

### Stage E: one-image complete reverse path from the intended reference prior

This tests terminal mixing and prior matching. A discrepancy between Stages D and E
indicates a terminal/reference-distribution problem rather than automatically a
learner problem.

### Stage F: small multi-image or class-conditioned MNIST experiment

Only after the one-image mechanism works or its failure is well localized, expand to
multiple images. Report both fidelity and diversity. One-image reconstruction is a
pipeline feasibility result, not proof of a generative model.

### Stage G: frozen reference-backend confirmation

Freeze the architecture, controller, gain/schedule, selection rule, metrics, and
claims. Rerun the fixed promising setting with protected evidence and the required
reference backend.

Once Stage A has passed, a representation-only or read-only adjudication may accompany
Stages B--E, but it must not gate them unless it identifies a concrete defect that
makes the trajectory invalid or uninterpretable. In particular, failure of a strict
componentwise family does not by itself forbid exploratory controller execution,
reconstruction, or sampling on separate evidence.

## 10. Strategy reviews and stopping rules

A high-level strategy review is mandatory when any of the following occurs:

- the same gate fails twice after targeted revisions;
- two consecutive proxy-only patches occur after basic correctness controls pass;
- three local representation variants fail without a materially different model;
- the assembled system has not been run despite components being composable;
- compute, storage, artifact count, or code complexity grows without reducing the
  main uncertainty;
- each positive result authorizes only another planning or diagnostic stage;
- a handoff forbids testing the core objective without a concrete integrity reason;
- the path from the proposed patch to an objective-bearing result is more than two
  substantive decisions long.

The review must choose explicitly among:

1. continue the present strategy;
2. simplify it;
3. run a direct end-to-end falsification test;
4. make a major architecture, training, controller, or numerical change;
5. compare with a simpler alternative baseline;
6. stop the current hypothesis.

State what evidence would reverse the decision. Continuing by inertia is not allowed.

## 11. Required `HANDOFF.md` structure

Use the following template. The executive scientific summary should fit roughly on
one screen; detailed provenance belongs later or in linked artifacts.

```markdown
# <Project or patch name>: research handoff

Date:
Source revision:
Handoff author:

## 1. Program objective
Final scientific/engineering objective:
Concrete success artifact:

## 2. Current milestone and distance to goal
Nearest objective-bearing milestone:
Current principal blocker:
Last objective-bearing experiment and date:
Artifact produced:
Proxy-only patches since then:
What remains untested end to end:

## 3. Strategy review
Strategy status: continue / major modification / pivot / stop / undecided
Rationale:
Strongest alternative strategy:
Evidence that would change this decision:

## 4. Research mode and evidence roles
Primary mode: exploratory / confirmatory / forensic / engineering-infrastructure
Training/development evidence:
Exploratory-validation evidence:
Protected confirmation evidence:
Evidence already opened or reused:

## 5. Exact result of the latest run
State the design, search family, independent unit, backend, criterion, and terminal
outcome without broader interpretation.

### This result establishes
...

### This result does not establish
...

## 6. Confirmed facts, current inferences, and open hypotheses
### Confirmed facts
Only claims directly supported by cited evidence.

### Current inferences
Reasonable but defeasible interpretations, labeled as such.

### Open hypotheses
Include implementation failure, proxy/gate misalignment, on-policy failure,
architecture failure, numerical/controller failure, prior mismatch, and strategy
failure when applicable. The list is not exhaustive.

## 7. Decision the next patch must resolve
One sentence describing a decision that changes the next action.

## 8. Candidate actions and value of information
Compare the smallest direct system test, read-only alternatives, new training, and a
major alternative. Include expected cost and what each action distinguishes.

## 9. Recommended next patch
Why it has the highest decision value:
What it will implement or execute:
Objective-bearing artifacts it will save:
Controls and baselines:
Primary metrics and health metrics:
What it will not claim:

## 10. Gates and claim boundaries
For each threshold, use the gate specification from `AGENTS.md`. Separate
execution/integrity gates, confirmatory claim gates, and diagnostic thresholds.

## 11. Outcome-to-action table
| Outcome | Interpretation | Required next action |
|---|---|---|

## 12. Constraints
### Integrity constraints
Only non-leakage, protected evidence, immutable completed artifacts, safety,
mathematical validity, and other genuinely claim-protecting restrictions.

### Revisable scientific and engineering choices
Architecture, features, loss, optimizer, controller, inference family, backend,
schedule, thresholds, and strategy unless explicitly fixed by the user or theorem.

For every prohibition, state its rationale, scope, and review trigger.

## 13. Resource budget and stop rule
Expected wall time/GPU/memory/storage/code complexity:
Maximum budget:
Automatic stop conditions:
Maximum proxy-only continuation before an objective-bearing run:

## 14. Alternative and pivot plan
What materially different approach is attempted if the recommended patch is
uninformative or negative:

## 15. Evidence map
For every load-bearing claim, list the exact artifact path and role. Distinguish raw,
derived, exploratory, validation-inspected, and protected evidence.

## 16. Deliberate omissions
List omitted artifacts and state which conclusions cannot be independently audited
without them.

## 17. Reproduction commands
Commands that work from a clean checkout with documented dependencies and paths.

## 18. Bundle-integrity audit
Verification command:
Expected file count:
Expected nonempty directories:
Manifest/hash location:
Representative files opened successfully:

## 19. Exact deliverable for the receiving agent
Specify code, executable experiment, report, implementation-ready plan, or other
concrete output. A positive result must not merely authorize another plan.
```

## 12. Receiving-agent responsibilities

Before accepting a handoff, the receiving agent must answer:

1. What is the actual project objective and nearest objective-bearing milestone?
2. When was the objective last tested directly?
3. Does the requested patch test the objective or only a proxy?
4. What exact decision changes after the patch?
5. Is each blocking gate necessary for the action it blocks?
6. Are confirmatory restrictions being incorrectly imposed on exploration?
7. Is there a cheaper direct experiment with greater decision value?
8. Does the negative language match the exact tested claim?
9. Are the central effect size and practical scale reported?
10. Are all cited artifacts present, nonempty, and readable?
11. Could the handoff's framing or the current strategy itself be wrong?
12. Does the patch end in execution, a major pivot, or a justified stop rather than
    another authorization layer?

If the answer reveals a strategic mismatch, state it and propose the corrected patch.
Faithful execution of a bad handoff is not success.

## 13. Prohibited handoff and experiment patterns

Do not produce or follow a plan with any of these patterns unless the user explicitly
requires it and the limitation is disclosed:

- “A positive result authorizes only planning another experiment.”
- “Do not run the end-to-end system until every diagnostic passes,” when failure only
  weakens a claim rather than invalidating the run.
- “Do not reconsider the architecture, loss, inference family, or backend,” without a
  theorem, integrity rationale, or explicit user instruction.
- “The learner failed” when only a stronger uniform or multiplicity-adjusted family
  failed.
- a closed list of minor diagnoses that excludes proxy misalignment, controller
  failure, architecture failure, prior mismatch, or strategy failure;
- a new exact cache or certification layer before a cheap decisive system test;
- adding diagnostic cells to understand behavior while silently requiring all new
  cells to pass;
- suppressing failed images or trajectories because a gate failed;
- opening protected confirmation evidence for exploratory tuning;
- changing several scientific axes without a baseline or ablation;
- reporting thousands of provenance artifacts without a compact decision summary;
- referencing evidence directories that are empty, absent, or omitted from the
  manifest;
- continuing the same local strategy solely because substantial code and compute have
  already been invested.

## 14. Final checklist

Before a production experiment or handoff is considered ready, verify:

- [ ] The final objective and nearest objective-bearing artifact are explicit.
- [ ] The primary research mode is declared.
- [ ] One substantive decision and competing hypotheses are stated.
- [ ] An outcome-to-action table has genuinely different branches.
- [ ] The smallest decisive direct experiment was considered.
- [ ] Null, known-positive, and current-baseline controls are included where needed.
- [ ] Short- and long-horizon behavior are separated for dynamical systems.
- [ ] On-policy evaluation is included or explicitly justified as unnecessary.
- [ ] Objective, mechanism, and health metrics are separated.
- [ ] Every threshold is typed as an execution gate, claim gate, or diagnostic.
- [ ] Negative conclusions are scoped to the exact tested criterion.
- [ ] Effect size and practical scale are reported.
- [ ] Exploratory and confirmatory evidence are separated.
- [ ] No more than two proxy-only patches have accumulated.
- [ ] Failed task artifacts will still be saved.
- [ ] Compute, storage, and code complexity are justified by information gained.
- [ ] A materially different alternative and pivot trigger are stated.
- [ ] Every referenced artifact exists, is nonempty, and is readable.
- [ ] The handoff does not end with permission to write another plan.

## Prime directive

**Preserve rigor, but do not confuse rigor with delay. Build and test the smallest
credible version of the actual generator, learn from its failures, and reserve the
strictest certification for claims that genuinely require it.**


Do not write overly defensive code. Always prefer simplicity over pathological complexity.

You need to write an experiment note for every experiment.