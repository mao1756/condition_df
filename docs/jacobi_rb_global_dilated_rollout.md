# Global-dilated Jacobi/RB exact five-row rollout

Date: 2026-08-14 JST  
Primary research mode: **exploratory**  
Run status: **sealed success**  
Objective-bearing artifact: one fresh, certified-exact, paired 128-step reverse suffix with five controller rows

## Program objective and decision

The program objective remains a DDPM-like MNIST image generator based on the
Eulerian approximation. This experiment tested the nearest objective-bearing
milestone rather than another representation proxy:

> On one fresh held-out forward path, does a genuinely global Jacobi/RB score
> model improve an exact paired 128-step reverse suffix over zero control, and
> does the prespecified frozen-v4 negative-sign diagnostic reveal a controller
> sign or ordering defect?

The run answered the first part positively at this one-path scope. The selected
global controller reduced final squared L2 error by `0.00025784694413619105`, or
`7.45378989614584%`, relative to paired zero. Both signs of the frozen v4 model
were adverse, so this run did not identify a simple sign reversal. The
source-informed positive control passed strongly through the same complete
controller and reverse-composition interface.

This was the direct system test required after the preceding proxy-heavy
sequence. The proxy-only counter is therefore reset:

```text
Proxy-only patches since the last objective-bearing experiment: 0
```

## Prespecified design and contracts

### Global architecture

The candidate was `GlobalDilatedZeroBaselinePredictor`, a 34,974-parameter
periodic 28x28 model. Its spatial branch used four circular 3x3 convolutions at
dilations `1, 2, 4, 8`, constant width 32, and SiLU activations. The resulting
contiguous offset range was `[-15, 15]`, a receptive field of 31 that spans the
28x28 torus. It retained the zero-baseline local-affine path and used frozen
coordinate features. There was no normalization, dropout, pooling, mixed
precision, or post-selection architecture change.

This was a material architecture comparison: a globally receptive model was
tested against the best frozen local frequency-one baseline, not another local
feature variation.

### Target and controller semantics

The load-bearing target contract was

\[
  m_\theta(W)=Y(1-Y)q_\theta(W),\qquad
  L=\frac{\operatorname{MSE}(m_\theta,\bar Z)}
          {\operatorname{RMS}_{\rm train}(\bar Z)^2}.
\]

Training called the wrapped model and compared `m_theta` directly with stored
`bar_Z`. It never formed `bar_Z / mobility`. Rollout called
`score_prediction` to obtain `q_theta`. A controller substep applied

\[
  \operatorname{logit}(Y^+)=\operatorname{logit}(Y)+2q_\theta\,\Delta u.
\]

The sealed theory-to-code audit verified that mobility was applied exactly once
during training, the rollout consumed `q_theta`, positive `q` increased the
declared head logit, zero mobility gave exactly zero wrapped output, and all
fixture values were finite.

### Training and selection

Training was deterministic on CUDA with seed `261372`, Adam learning rate
`1e-3`, betas `[0.9, 0.999]`, epsilon `1e-8`, batch size 32, zero weight decay,
no AMSGrad, and global gradient-norm clipping at 1. Checkpoints were committed
every 100 updates. A timing-only cap ladder of `4000, 3000, 2000, 1000` admitted
the full 4,000 updates.

The prespecified selection rule was the finite, nonzero checkpoint with minimum
validation MSE of wrapped `m` against stored `bar_Z`, with the earlier update
winning an exact tie. All 41 checkpoints from update 0 through 4000 were
retained. Selection was required to continue even if every nonzero checkpoint
was worse than zero; the frozen v4 comparison was never eligible for selection.

| Model/checkpoint | Update | Raw validation MSE | Normalized validation MSE | Selection role |
|---|---:|---:|---:|---|
| Global zero | 0 | 6.807536109081771 | 0.9992301715496035 | Reference only |
| Selected global | 3100 | 6.806853452023982 | 0.9991299691389172 | Prespecified winner |
| Frozen v4 | 3700 | 6.806818434636477 | 0.9991248291837904 | Comparison only |

The training-target RMS was `2.610130414663935`. The selected global checkpoint
state SHA-256 was
`1df9888bef6c63db10f41f89a58891321e058e55ed7d8b36622c9cdf9827a218`.
Its validation risk was slightly worse than frozen v4, while its paired suffix
effect was better. On this single path, cached validation risk therefore did
not rank recursive suffix performance; that observation is descriptive, not a
population result.

### Evidence roles and evaluation freeze

The immutable training-parent registry contained exactly 2,701 registered
artifacts plus its registry file, all remeasured before use. Evidence roles were
kept disjoint:

| Role | Path IDs | Use |
|---|---|---|
| Training/development | `0xF8100`-`0xF813F` | Global training labels |
| Exploratory validation | `0xF8200`-`0xF821F` | Checkpoint selection |
| Protected confirmation | `0xF9000`-`0xF903F` | Remained unopened |
| Exact preflight | `0xFB2F1` | Interface, health, and resource smoke |
| Fresh evaluation pool | `0xFB300`-`0xFB30F` | First collision-free ID selected |

The actual evaluation path was `0xFB300` (`1028864`). Repository-wide path
allocation was completed before fresh evidence was opened. The forward root
seed was `261401`; the reverse root seed was `261402`. The evaluation freeze
bound the target, model states, gains, row order, exact sequence, seeds,
rendering scale, and resource limits before generating the fresh step-127
anchor.

The forward process opened only steps through 127. Step 511 was never opened.
The exact 128-step reverse sequence ran outer steps 127 down to 0 with seven
phases and `M=2` controller microsteps. All five rows used the same canonical
transition IDs and random bits.

### Five-row control stack

The frozen row order was:

1. `zero`: null controller;
2. `v4-plus-0p5`: frozen v4 current baseline at gain `+0.5`;
3. `v4-minus-0p5`: diagnostic-only signed v4 at gain `-0.5`;
4. `global-plus-1`: selected global-dilated candidate at gain `+1`;
5. `source-informed`: mixed-target fraction oracle as the known-positive
   controller-interface control.

The source-informed row was not a learned score or a population baseline. Its
purpose was to test the complete controller and reverse-composition interface.
Before fresh evidence, the actual frozen-v4 input also satisfied
`Signed(v4,-0.5)(W) == -Scaled(v4,+0.5)(W)`, and the v4 state-dict hash was
unchanged across inference.

## Execution and integrity controls

All blocking gates were execution/integrity gates. They protected input and
source bindings, target semantics, controller orientation, shared randomness,
exact certification, numerical health, restart authority, and resource limits;
they did not act as scientific claim gates.

The exact preflight ran one eight-step five-row fused shard and an independent
source singleton. It verified:

- bitwise equality of the singleton and fused source row;
- row-permutation-invariant transition identities;
- full exact certification and finite, nonnegative simplex states;
- zero forbidden events and maximum mass error
  `2.220446049250313e-16`;
- a source final squared L2 error of `0.00013361331524007608`, below paired
  zero at `0.002720298089721669`;
- CUDA peak allocation below the 80% cap; and
- a 20%-margined mandatory-suffix projection of
  `5957.7922809599895` seconds, within the six-hour cap.

Only after those controls passed were the training cap, selection, evaluation
freeze, and fresh path used.

## Sealed v3 result

The successful immutable run is:

```text
runs/experiment12-d0-jacobi-rb-global-dilated-rollout/
  20260813-233915_production-global-dilated-exact-five-row-v3
```

Its terminal classification was `global_material_improvement`. The primary
metric was `Delta = E_zero - E_c`, where `E_c` is the final raw-state squared L2
error to the frozen mixed target. The 1% relative-effect boundary was a
**diagnostic threshold**, not a hypothesis test or confirmatory gate.

| Row | `E_c` squared L2 | `Delta` | Relative `Delta` | L1 error | TV distance | Centered contrast correlation |
|---|---:|---:|---:|---:|---:|---:|
| `zero` | 0.003459273037324502 | 0 | 0% | 1.0575882582911713 | 0.5287941291455857 | 0.11697874299278531 |
| `v4-plus-0p5` | 0.0035312921849536445 | -0.00007201914762914241 | -2.081915675694799% | 1.0239329040125842 | 0.5119664520062921 | 0.19063859802785862 |
| `v4-minus-0p5` | 0.003525732287509452 | -0.00006645925018495006 | -1.921191229136151% | 1.0922240716590559 | 0.5461120358295279 | 0.04487943980691853 |
| `global-plus-1` | 0.003201426093188311 | 0.00025784694413619105 | 7.45378989614584% | 0.921799568676855 | 0.4608997843384275 | 0.33929815601191937 |
| `source-informed` | 0.00006187506618656453 | 0.0033973979711379378 | 98.21132748068883% | 0.1491606646496611 | 0.07458033232483055 | 0.9850915430480526 |

Both v4 signs were adverse. The negative-sign diagnostic was marginally less
adverse than the positive baseline but did not reverse the endpoint effect;
therefore a simple sign flip was not supported. The global row crossed the
prespecified 1% practical label, and the source-informed row demonstrated that
the assembled interface could produce a large improvement.

These facts favor continuing with a complete-path global-controller test over
another local representation tweak. They do not identify why the global model
outperformed v4 on this path. Global receptive field, different recursive
state behavior, controller scale, and path-specific variation remain open
explanations.

## Saved task artifacts and exact health

The mandatory suffix completed all 16 restart shards. It retained five rows by
17 shard-boundary states by 784 cells, plus milestones after 0, 32, 64, 96, and
128 reverse steps. Raw and background-demixed PNGs were saved for every row and
milestone, including adverse v4 outputs. Fourteen fixed-scale contact sheets
were also saved. No adverse task artifact was suppressed.

| Authority | Shards | Active/certified transitions | Certified fallbacks | Forbidden events | Max mass error | Negative/nonfinite states |
|---|---:|---:|---:|---:|---:|---:|
| Fresh forward to step 127 | 16 | 351,232 / 351,232 | 0 | 0 | `2.220446049250313e-16` | 0 / 0 |
| Five-row reverse suffix | 16 | 7,024,640 / 7,024,640 | 2 | 0 | `4.440892098500626e-16` | 0 / 0 |

The two reverse fallbacks were certified exact fallbacks, one in the
`v4-minus-0p5` row and one in the `source-informed` row; they were not invalid
or approximate transitions. Restart chains and all per-row telemetry
cross-bindings passed. Derived authority comprised 85 recomputed metric rows,
17 mechanism records, bitwise-recomputed milestones, and 64 decoded and
reproduced images. The complete boundary-state array hash was
`d4f4e8392a77e8e52fbda5fb6e284dd99368bae69e3f35e321db53aeac3fc1d9`.

## Resource result and optional complete path

The final ledger recorded:

| Resource | Observed | Cap | Result |
|---|---:|---:|---|
| Active/cap-debited wall time | 7,580.6255627 s | 21,600 s | Passed |
| Peak CUDA allocation | 75,522,560 B | 80% of 8,546,484,224 B | Passed |
| Final recursive storage | 12,520,738 B | 2,147,483,648 B | Passed |

There were no abandoned hard-crash intervals and no resource breaches. Exact
suffix sampling accounted for about 5,092.324 seconds; final reporting retained
the frozen 600-second reserve.

The positive global result triggered the optional same-path complete branch,
which would have resumed the same forward path to step 511 and compared
`zero`, `global-plus-1`, and `source-informed`. It was **not launched**. Its
projected totals were:

- forward tail: `2700` seconds;
- three-row exact reverse: `14298.701474303974` seconds;
- optional postprocessing: `30` seconds;
- total ledger active time: `24608.895198303973` seconds; and
- total storage: `29221944` bytes.

Storage fit, but time exceeded the frozen six-hour cap. The branch therefore
ended `triggered=1, attempted=0, completed=0`. No step-511 state, partial full
trajectory, or complete-path claim exists. This is a resource deferral, not an
adverse full-path result.

## Outcome-to-action decision

| Observed outcome | Interpretation at tested scope | Required action |
|---|---|---|
| Source interface control failed | Composition or implementation invalid | Repair the interface before interpreting a learner |
| Negative v4 alone materially improved | Sign/order convention becomes leading | Reconcile theory and code before changing signs |
| Global improved materially, as observed | One-path global-context feasibility | Attempt the same-path complete zero/global/source reconstruction if an exact budget fits |
| Validation improved but suffix was adverse | Cached risk did not compose recursively | Change target derivation or rollout alignment |
| All learned rows adverse with controls valid | Nearby Jacobi learner changes lack feasibility evidence | Run the stated standard-DDPM/direct heat-potential pivot |

The next scientific decision is whether the global advantage persists over a
complete same-path reverse trajectory. It should be answered by a separately
budgeted objective-bearing run, not by another read-only proxy or by rerunning
the sealed v3 command.

## Claim boundary

### This run establishes

- On one fresh path, with the frozen target, selected checkpoints, gains,
  exact `M=2` backend, phase order, and paired randomness, the global row
  improved the 128-step suffix over zero by 7.4538%.
- Under the same conditions, frozen v4 at both `+0.5` and `-0.5` was adverse.
- The source-informed interface control improved by 98.2113%, and exact health
  and resource controls passed.
- A globally receptive Jacobi/RB model therefore has one-path exploratory
  suffix-feasibility evidence that the frozen local v4 model did not show on
  this path.

### This run does not establish

- a complete reverse-path reconstruction;
- generation from the intended reference prior;
- multi-image fidelity, diversity, or population generalization;
- a p-value, confidence interval, or confirmatory claim;
- that global receptive field alone caused the improvement;
- that cached validation risk is generally misaligned; or
- success or impossibility of the broader Eulerian/Jacobi/RB strategy.

One path/image was the independent unit, so no population uncertainty was
estimated. Confirmation paths `0xF9000`-`0xF903F` remained unopened.

## Immutable attempt provenance

Two failed predecessor runs were preserved rather than overwritten:

| Run | Terminal stage and exact failure | Scientific authority |
|---|---|---|
| `20260813-225613_production-global-dilated-exact-five-row` | `controls`: `v4 step-127 development anchor hash changed` | Execution-integrity failure; 0/16 suffix shards, objective incomplete, nonresumable |
| `20260813-230738_production-global-dilated-exact-five-row-v2` | `train_select_freeze`: `RNG state must be a torch.ByteTensor` | Execution-integrity failure; 0/16 suffix shards, objective incomplete, nonresumable |
| `20260813-233915_production-global-dilated-exact-five-row-v3` | All five stages passed | Complete one-path exact suffix authority |

The v1 root cause was a consumer hash-shape mismatch: the immutable v4 anchor
binding used the canonical row shape `(1, 784)`, while the first consumer
hashed the same values as `(784,)`. The repaired consumer retained the archive
file, dtype, and stored-shape checks but compared the canonical reshaped row.

The v2 root cause was CUDA-mapped saved RNG tensors being passed to an API that
requires CPU byte tensors. The repaired restore path CPU-normalized the saved
states, bound exact CUDA topology and state count, and validated CPU and every
CUDA state on scratch generators before mutating any default generator. Both
failed bundles have passing terminal verification records and support no
scientific conclusion about the controller.

## Artifact map and bundle integrity

All paths below are relative to the sealed v3 run root.

| Evidence role | Load-bearing artifact |
|---|---|
| Compact result and handoff | `REPORT.md`, `HANDOFF.md` |
| Scientific design | `scientific_config.json` |
| Input, role, parent, v4, and source bindings | `input_bindings.json` |
| Theory and assembled-interface controls | `controls/theory_to_code.json`, `controls/preflight_controls.json` |
| Checkpoint selection and frozen evaluation | `selection.json`, `evaluation_freeze.json` |
| Training trace and checkpoints | `training/history.csv`, `training/training_cap.json`, `training/checkpoints/` |
| Fresh exact forward evidence | `fresh_forward/forward_shards/`, `fresh_forward/anchor-step-0127.npz`, `fresh_forward/forward_summary.json` |
| Raw exact reverse evidence | `suffix/fused_families/fresh-five-row/suffix-128/`, `suffix/trajectory_shard_boundaries.npz` |
| Derived metrics and mechanism records | `suffix/metrics.csv`, `suffix/summary.json`, `suffix/mechanism.json` |
| Task-level images | `images/raw/`, `images/demixed/`, `images/contact-sheets/` |
| Decision and optional branch | `outcome.json`, `positive_branch.json` |
| Resource authority | `resource_ledger.json`, `terminal_storage_authority.json` |
| Source and command provenance | `run_manifest.json`, `exact_command.txt` |
| Completeness and deep verification | `artifact_manifest.json`, `SHA256SUMS.txt`, `verification.json` |

The sealed bundle contains 212 physical files totaling 12,520,738 bytes. Its
manifest registers 207 non-self-referential artifacts; the checksum inventory
has 208 entries. Deep verification reopened all 16 raw suffix shards,
recomputed the 85 metrics and 17 mechanism records, rebound selection/freeze/
path authority, reproduced every promised image, and passed. Key seals are:

- run manifest semantic SHA-256:
  `8c37b461dbad725f454d5f3f0a219c5329172bc1d27471ba9bb3cc468fc393c4`;
- artifact manifest semantic SHA-256:
  `327885100381d72c996a552a7a543db2c45451c5acb40036d995bb52712ddf47`;
- artifact manifest file SHA-256:
  `6542ff34926dc1298df15420f361fa6f36cbcf84475d9f7e71b5e0de6934eb7c`;
- checksum file SHA-256:
  `bfd253f6b174dd9533d4d7f6a294af08ab54c685505df750727691ed3bdc1f49`;
- verification semantic SHA-256:
  `e11c761732608b2eb1fef3ff4a67c7ab8dfefbdf8edd13ff694d6d2b52a79578`;
- terminal storage authority semantic SHA-256:
  `9d440f7fd679341a6c63e980b1fc6a32a349696fc2188ffb4683b0112f4aeec2`.

The external audited launch plan was not copied into the sealed run. The
executable scientific authority is the combination of `scientific_config.json`,
`input_bindings.json`, `selection.json`, and `evaluation_freeze.json`, all bound
by the run and artifact manifests.

## Historical command — provenance only, do not rerun

The following is the exact command preserved by v3. It is reproduced only as
historical provenance. **Do not execute it:** the named immutable run already
exists, and the current source is FROZEN27 rather than the FROZEN26 source
closure sealed by v3.

```powershell
C:\Users\mao17\Workspace\condition_df\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_global_dilated_rollout --stage all --repository-root C:\Users\mao17\Workspace\condition_df --runs-root C:\Users\mao17\Workspace\condition_df\runs\experiment12-d0-jacobi-rb-global-dilated-rollout --run-name production-global-dilated-exact-five-row-v3 --training-parent C:\Users\mao17\Workspace\condition_df\runs\experiment12_d0_jacobi_rb_boundary_tangent_frequency1_coordinate_learnability\20260811-010641_production-frequency1-coordinate-v1-one-image --v4-run-dir C:\Users\mao17\Workspace\condition_df\runs\experiment12_d0_jacobi_rb_frequency1_rollout\20260813-002414_production-frequency1-objective-first-recovery-v4 --source-run-dir C:\Users\mao17\Workspace\condition_df\runs\experiment12_d0_jacobi_rb_frequency1_rollout\20260813-002414_production-frequency1-objective-first-recovery-v4\input_bindings --device cuda
```

No command in this document authorizes opening confirmation evidence, mutating
v1/v2/v3, or launching a new production run.

## FROZEN26 run closure and FROZEN27 future-run fix

The successful v3 run is immutably bound to **FROZEN26**, not to the current
working source:

- repository commit: `b4bb7f9e1bec0062b44efb99b333bd144e4a119d`;
- dirty-status SHA-256:
  `ef366006c766af0688cc7f95099658933d90fb14ec379ac415f1408d066d3b1d`;
- transitive source-closure SHA-256:
  `65d85cb4345a14fb8b4e442ff978c66bc77bd6bba7051f295706b82bac3a6014`;
- orchestrator source SHA-256:
  `2356ddb38d39e75689ca1193094fc9114660915933235dece67d0b8490e32351`;
- prelaunch orchestrator-test SHA-256:
  `dd94cda44832077bdeb7c96d4ceb18bd446cfdcaaf336179dc8002a4752b72a9`.

After v3 sealed, a narrow read-only-resume defect was found. JSON serialization
stored `training.betas` as a list, while the live scientific-config constructor
returned the source tuple. Those values serialized identically but compared
unequal in memory, so a completed-run resume could reject before reaching its
read-only verifier.

**FROZEN27** fixes only that future-run canonicalization by constructing
`training.betas` as a list after expanding the frozen training mapping. It does
not change the serialized v3 scientific configuration, whose semantic SHA-256
remains
`02149d49ede0379ac96a8c188d7a7904b460ae1df7f770be116c8ff96943847b`.
The frozen future-run files are:

- orchestrator source SHA-256:
  `9258ad5c49474250b7f150c26fe78fa9db892a602e17d085511d4e39391fd98d`;
- orchestrator test SHA-256:
  `a33ae1f29284d22ef4573de663b4bc793f7d9dba4f73b6ca218252167572d2e5`.

FROZEN27 passed Python compilation, 86 focused orchestrator tests, and 192
combined model/tangent/fused/orchestrator tests. Two independent bounded audits
returned GO and reconstructed the exact FROZEN26 bytes by removing only the
canonicalization patch and its two regressions.

This fix is deliberately **not** a legacy source-closure bypass. Running the
FROZEN27 executable against v3 must still reject the changed transitive source
closure. The v3 bundle remains auditable through its sealed manifests and deep
verifier. Its FROZEN26 path/size/hash closure is recorded, but a standalone copy of
the old orchestrator bytes is not archived; both auditors reproduced its exact hash
by reversing the documented one-line FROZEN27 change. FROZEN27 governs future runs.
No v1, v2, or v3 artifact was modified to accommodate the fix.
