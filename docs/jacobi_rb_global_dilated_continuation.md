# Exact same-path global-dilated continuation

## Status and scope

This document specifies the authenticated-prefix successor implemented by
`mnist.diag_d0_jacobi_rb_global_dilated_continuation`. The engineering portion
authenticates the sealed v2 resource-stop child, independently copies its complete
512-step forward path and first eight-step reverse shard into a fresh child, carries
the predecessor's exact active cost once, and then immediately resumes the
exploratory paired reverse reconstruction at shard 1.

The scientific objective is to determine whether the global-dilated controller's
one-path 128-step advantage survives complete recursive composition. The concrete
task artifacts are the complete zero, global, and source-informed reverse
trajectories, their milestone and final images, and paired reconstruction metrics.

Primary patch mode: **engineering/infrastructure**, immediately followed by an
**exploratory** objective-bearing experiment. The continuation reuses one
already-opened path and one selected checkpoint. It is neither a fresh replication
nor a confirmatory experiment. The only policy change is the explicitly approved
`22,500s` child active-time cap.

This patch adds one runner, one focused test module, and this technical document.
Existing scientific source, tests, configurations, and sealed runs remain
byte-identical. Updating the repository-root `HANDOFF.md` after the terminal result
is a mandatory reporting exception to the add-only implementation scope; it is not
part of the runner's source closure and never changes v3's internal handoff.

## Decision and competing hypotheses

Decision: does frozen global `+1` improve the final squared-L2 reconstruction error
over paired zero control on the exact 512-step continuation of path `1028864`?

The freeze records these competing explanations and their predicted observations:

| Hypothesis | Predicted observation |
|---|---|
| Implementation or orientation defect | Exact health or the source-informed full-system control fails |
| Controller, integrator, or interface failure | Source-informed control is adverse or dynamically uninformative despite healthy raw states |
| Global architecture or parameterization is inadequate | Exact/source controls pass, but global is adverse throughout the complete path |
| Gain or late-time calibration failure | Global materially helps at a predeclared intermediate horizon but becomes adverse at the endpoint |
| Useful but dynamically negligible signal | Direction is positive but remains below the 1% practical threshold |
| On-policy distribution shift | Global separates from its calibration distribution as its recursive advantage degrades |
| Terminal/reference-prior mismatch | Same-path Stage D succeeds but the later reference-prior Stage E fails |
| Proxy misalignment | Trajectory utility differs materially from the near-zero cached validation-MSE difference |
| Exact-backend cost or discrepancy | The exact run cannot fit its declared resource budget, or later approximate/reference results diverge |
| Current Jacobi/RB strategy failure | Complete-path global is adverse with valid exact and source controls |
| Other evidence-supported explanation | The saved raw trajectory supports an explanation outside this prespecified list |

The list is deliberately open. A terminal report must not force every result into a
minor controller repair.

## Frozen scientific identity

The continuation freezes:

| Field | Value |
|---|---:|
| Authenticated predecessor | `production-same-path-complete-v2` |
| Parent run | `20260813-233915_production-global-dilated-exact-five-row-v3` |
| Selected update | `3100` |
| Training seed | `261372` |
| Checkpoint state SHA-256 | `1df9888bef6c63db10f41f89a58891321e058e55ed7d8b36622c9cdf9827a218` |
| Path ID | `1028864` |
| Forward root seed | `261401` |
| Reverse root seed | `261402` |
| Reverse stream role | `global_dilated_positive_complete_exact` |
| Label | `3` |
| Phases | `7` |
| Exact microsteps | `2` |
| Forward extent | steps `0..511` |
| Imported forward | steps `0..511`, shards `0..63` |
| Generated forward | none |
| Imported reverse | completed steps `0..8`, shard `0` |
| Generated reverse | completed steps `9..512`, shards `1..63` |
| Rows | `zero`, `global-plus-1`, `source-informed` |
| Global gain | `1.0` |
| Reverse milestones | completed steps `0, 128, 256, 384, 512` |
| Backend | `certified_exact` |
| Independent unit | one already-opened path/image |

Expected active-transition authorities are:

```text
full imported forward 512 * 7 * 392                 =  1,404,928
imported reverse       8 * 7 * 2 * 2 * 3 * 392     =    263,424
generated reverse    504 * 7 * 2 * 2 * 3 * 392     = 16,595,712
complete reverse     512 * 7 * 2 * 2 * 3 * 392     = 16,859,136
```

The reverse sequence is `tuple(reverse_suffix_sequence(511))`, containing 3,584
phase coordinates. Captures are frozen at `(384, 0)`, `(256, 0)`, `(128, 0)`, and
`(0, 0)`, corresponding to 128, 256, 384, and 512 completed reverse steps.

## Predecessor, parent, and producer authentication

The sealed v2 predecessor is immutable evidence, never a resume target. Before any
copy, its pinned verifier requires exactly 158 regular files, 2,887,822 recursive
bytes, 154 manifest rows, 155 checksum entries, a verified terminal resource-stop
package, all 64 healthy forward pairs, one healthy reverse pair, and only the
authenticated pre-sampler failure at reverse shard 1. The canonical predecessor
ledger value is `1945.6831628999998s`, with CUDA authority
`46,834,176/8,546,484,224` bytes and no breach. Its historical `21,600s` policy is
validated as predecessor evidence and is never confused with the successor policy.

Exactly 136 operational files (2,654,708 source bytes) are independently copied to
canonical child paths. Sixteen compact terminal/provenance authorities are copied
under `imports/v2/`. The operational tree receives neither v2's shard-1 failure nor
its terminal failure package. Every copy is byte-hash exact, single-link, and not the
same file as its source.

The parent is immutable evidence, never a resume target. The runner requires its
212 regular files, 207 artifact-manifest entries, 208 checksum entries, exact
terminal authorities, selected checkpoint, source pair, path binding, prefix
chain, outcome, resource deferral, and unopened confirmation role. It also pins the
complete sealed tree hash
`2790043c1363cf9f75b7c64bce0b2792c3afa8622b660d2a92f69107d511c452`
and the excluded report marker's raw and semantic hashes, so a self-consistent
rewrite of a manifest-excluded terminal authority cannot be rebound as the parent.

V3 was produced by FROZEN26:

```text
size    387813
sha256  2356ddb38d39e75689ca1193094fc9114660915933235dece67d0b8490e32351
```

The current historical runner is FROZEN27:

```text
size    387863
sha256  9258ad5c49474250b7f150c26fe78fa9db892a602e17d085511d4e39391fd98d
```

FROZEN27 differs by exactly one 50-byte canonicalization line:

```python
b'                "betas": list(TRAINING["betas"]),\n'
```

Parent authentication removes the unique line in memory, verifies the reconstructed
FROZEN26 size/hash, and never writes the reconstructed bytes. Every other parent
source-closure file is checked against current bytes. FROZEN27 must never be called
the historical producer.

The supplied predecessor, parent, and source locators are snapshotted before and
after every stage. Terminal verification snapshots the child as a fourth tree.
Verification does not follow historical training, registry, or protected
confirmation paths.

## CLI modes

The runner has three mutually exclusive modes.

Fresh child:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_global_dilated_continuation `
  --stage all `
  --repository-root C:\Users\mao17\Workspace\condition_df `
  --runs-root C:\Users\mao17\Workspace\condition_df\runs\experiment12-d0-jacobi-rb-global-dilated-continuation `
  --run-name production-v2-prefix-successor-v1 `
  --prefix-run-dir C:\Users\mao17\Workspace\condition_df\runs\experiment12-d0-jacobi-rb-global-dilated-continuation\production-same-path-complete-v2 `
  --parent-run-dir C:\Users\mao17\Workspace\condition_df\runs\experiment12-d0-jacobi-rb-global-dilated-rollout\20260813-233915_production-global-dilated-exact-five-row-v3 `
  --source-run-dir C:\Users\mao17\Workspace\condition_df\runs\experiment12_d0_jacobi_rb_frequency1_rollout\20260813-002414_production-frequency1-objective-first-recovery-v4\input_bindings `
  --device cuda
```

Fresh mode accepts only `prepare` or `all`; sampling-capable invocations require
CUDA. The final child is created through a temporary sibling and atomic rename.

Resume:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_global_dilated_continuation `
  --stage all `
  --repository-root C:\Users\mao17\Workspace\condition_df `
  --resume-run-dir <child-run> `
  --prefix-run-dir <sealed-v2-resource-stop> `
  --parent-run-dir <sealed-v3> `
  --source-run-dir <source-input-bindings> `
  --device cuda
```

Only a nonterminal successor with the exact prefix, parent, and source locators may
resume. V1, v2, v3, arbitrary directories, locator/source-closure mismatches, and
terminal children are rejected before mutation.

Read-only terminal verification:

```powershell
.\.venv\Scripts\python.exe -m mnist.diag_d0_jacobi_rb_global_dilated_continuation `
  --repository-root C:\Users\mao17\Workspace\condition_df `
  --verify-run-dir <terminal-child> `
  --prefix-run-dir <sealed-v2-resource-stop> `
  --parent-run-dir <sealed-v3> `
  --source-run-dir <source-input-bindings>
```

Verification forbids `--stage`, `--device`, `--runs-root`, `--run-name`, and resume.
It initializes no sampler or CUDA state and leaves child, predecessor, parent, and
source trees byte-identical.

## Five-stage lifecycle

### Prepare

`prepare` opens its durable journal, authenticates v2 plus v3/source, copies and
reopens the 16-file `imports/v2` capsule, and then atomically appends one
`prefix_resource_carry` event whose elapsed value is read from that authenticated
copied ledger. It next copies the 136 operational files and writes child-owned
predecessor, parent, input, forward-import, and reverse-import bindings. Copies may
not be symlinks, junctions, reparse points, same files, or hardlinks.

The carry event uses the normal child resource-event schema and binds v2's ledger,
verification, terminal failure, terminal storage, and run-manifest authorities. It
is idempotent and conflict-closed; v2 event IDs are never transplanted. No sampler or
CUDA initialization is allowed.

### Controls

`controls` revalidates the complete 64-shard imported forward chain, step-511 anchor,
imported reverse shard 0 through the same load-bearing exact-health validator used
for the complete family, exact carry event, partial mixed-target metrics, and the
initial successor resource projection. It writes a child-specific freeze before any
new allocation. The freeze distinguishes zero child sampling from the 64 forward and
one reverse shards already present as predecessor evidence.

There is no CPU replay or extra proxy trajectory. The next sampler call must be
reverse shard 1 and must contribute directly to the objective-bearing path.

### Forward tail

`forward_tail` is retained only for stage compatibility. It calls no sampler,
requires the imported scanner to report all 64 shards, recomputes the complete
1,404,928-transition health authority, proves shard 63 equals the imported step-511
anchor, and writes a child summary reporting 64 imported and zero generated shards.
Its resource event charges validation wall only.

### Complete reverse

The step-511 anchor is repeated for the frozen rows:

- zero control;
- `ScaledTangentScoreController(update-3100, gain=1.0)`;
- `TargetFractionOracleController(mixed_target, microsteps=2)`.

All rows share canonical path `1028864` and the same exact-reference random bits;
row variants never enter the RNG key. Imported shard 0 is byte-protected and not
charged again. The fused runner starts at the clean shard-1 gap and admits shards
1..63 one at a time. Existing NPZ-only or repeated failure evidence for child-owned
shards is archived under `recovery/` before replay so that low-level atomic
replacement cannot destroy it.

After all 64 raw shards exist, a separate admitted CPU attempt reconstructs the `(3,65,784)`
boundary array and `(3,5,784)` milestones, writes 195 metric rows, horizon and
on-policy diagnostics, and all raw/demixed images. Failed and adverse outputs are
never suppressed.

### Report and verify

The reporting reserve is **prepaid**, not merely projected. Admission charges the
full 600 seconds to the ledger before the ledger is frozen and records a monotonic
deadline. Report generation, manifest/checksum construction, terminal fixed-point
serialization, and the final read-only audit must all finish within that deadline.
This prevents terminal work after ledger freeze from escaping the active-time cap.

`stages/report_verify.json` is the final success commit. A completed adverse global
result with passing controls is still a successful scientific terminal state:
"success" means complete and interpretable, not beneficial.

## Metrics and gates

Primary objective: paired final mixed-target squared-L2 error. Reports also retain
the absolute delta, relative improvement, L1, TV, centered correlation, comparison
to the source image, row-versus-zero separation, controller/reference scales,
calibration drift, and health/resource diagnostics.

### Exact numerical health

Gate type: execution/integrity.

Exact health requires the full transition authorities, complete certification,
authorized shared-RNG identities, finite/nonnegative float64 states, zero forbidden
events, and mass error at most `2e-12`. Certified fallbacks are permitted only when
they remain certified, authorized, finite, and conservative. Failure preserves the
last valid state and makes no learned-controller claim.

### Source-informed control

Gate type: execution/integrity gate controlling learned-endpoint interpretation.

The exact proposition is that the known-positive source controller improves the
paired complete-path objective by at least 1%. This is necessary before attributing
the learned row's complete-path behavior to the learner, because a smaller source
effect does not demonstrate that the assembled controller/composition interface has
practically informative authority at this horizon. Failure blocks that
interpretation; it does not suppress artifacts, block separate exploratory repair,
or establish that the learned controller lacks signal.

For finite `E_zero > 0`, define:

```text
relative_source = (E_zero - E_source) / E_zero
```

| Condition | Label | Consequence |
|---|---|---|
| `relative_source >= 0.01` | `source_informative` | Learned endpoint may be interpreted |
| `0 < relative_source < 0.01` | `source_positive_small_uninformative` | Direction is positive but the teacher is dynamically uninformative; block learned interpretation |
| `relative_source <= 0` | `source_adverse` | Composition-control failure; block learned interpretation |
| Nonfinite endpoint or `E_zero <= 0` | `invalid_objective` | Interpretation-invalid failure |

The 1% value is the prespecified practical scale inside this execution/integrity
gate. It is not also typed as a diagnostic or confirmatory gate, and passing it does
not authorize a population claim.

### Global complete-path effect

Gate type: diagnostic threshold selecting the next exploratory action. It does not
invalidate an otherwise healthy run and does not control a confirmatory claim.

For finite `E_zero > 0`, define:

```text
relative_global = (E_zero - E_global) / E_zero
```

Predeclared intermediate relative improvements use the same formula at completed
steps 128, 256, and 384. An intermediate improvement must reach 1% to select the
late-adverse scheduling branch. A smaller positive value is recorded as direction
only and cannot choose strategy.

| Observation, after exact/source controls pass | Label | Required next patch |
|---|---|---|
| Final `relative_global >= 0.01` | `global_material_improvement` | Stage E one-image reference-prior test |
| Final `0 < relative_global < 0.01` | `global_positive_small` | One new independent path/seed replication with model, gain, and schedule frozen |
| Final `relative_global <= 0` and an intermediate is `>=0.01` | `global_early_help_late_adverse` | One predeclared time-window schedule ablation before retraining |
| Final `relative_global <= 0` with no material intermediate | `global_complete_adverse` | Conventional MNIST DDPM reconstruction sanity baseline first |

For the future positive-small replication, an identical deterministic rerun is only
verification. The next patch must allocate and bind a genuinely new exploratory
path/seed. A positive replication routes to Stage E with a weak two-path exploratory
label; a nonpositive replication routes to the DDPM sanity baseline.

If the schedule ablation fails, the next intervention is rollout-aligned training or
a materially different formulation. If the DDPM baseline passes while Jacobi/RB is
adverse, pivot away from the current formulation; if the baseline also fails, audit
the shared metric/data/interface before interpreting either learner.

## Resource contract

Gate type for every limit in this section: execution/integrity. These limits stop
the next write or sampler admission; crossing one yields partial resource-stop
evidence and never a negative scientific conclusion.

```text
successor active cap     22,500 seconds
authenticated v2 carry  1,945.6831628999998 seconds
storage cap               2 GiB
CUDA allocation cap      80% of device total
prepaid reporting reserve 600 seconds
reverse baseline         223.4172105359996 seconds per remaining shard
postprocess reserve       30 seconds
initial carried projection 21,686.624694539896 seconds
nominal initial headroom     813.3753054601038 seconds
```

The predecessor carry is read from the authenticated copied v2 ledger and charged
exactly once. Its imported shard-0 elapsed `252.79023189999862s` remains inside that
carry, participates in adaptive admission, and is never charged again as child work.
The pre-setup projection is
`1945.6831628999998 + 63*303.34827827999834 + 30 + 600 =
21686.624694539896s`. Actual prepare, controls, no-op-forward, resume, reverse, and
postprocess/report wall is charged; nominal headroom alone never authorizes a shard.

Expected remaining production wall is approximately 19,741 seconds (5.5 hours)
conservatively. V2's observed peak CUDA allocation was 46,834,176 of
8,546,484,224 bytes; the successor stops at 80% of the actual device total. Frozen
storage reserves project roughly 24 MB while retaining the conservative 2 GiB cap.

Admission occurs before each sampler. Adaptive next-shard estimates use the larger
of the frozen baseline and 120% of the maximum elapsed among imported shard 0 and all
committed child shards. The durable reverse delta excludes imported shard 0 while the
carried total retains it. Time equality passes; storage equality and CUDA allocation
at or above 80% fail. A resource stop is partial evidence, not a scientific negative.

The reporting reserve is charged in full before packaging. A crash may resume only
within its original monotonic deadline; an overrun records a `report_deadline`
resource breach and can seal only a terminal failure package. Every journal-delete
path reconverges the exact persisted-byte ledger, and success requires final bytes
strictly below the storage cap.

Implementation complexity is also budgeted: one runtime module no larger than 4,500
lines, one focused test module no larger than 3,000 lines, and this document. If the
runner exceeds that limit before an executable integration smoke, it must reuse
authenticated stateless FROZEN27 helpers or reduce administrative surface rather
than grow a new general framework.

This approved cost purchases the decision whether the observed short-suffix
advantage survives complete recursive composition. Reopening v2/v3, rerunning a
local prediction metric, or performing another read-only decomposition cannot answer
that question. Another resource stop ends this exact-continuation branch; it does not
automatically authorize a further cap increase.

## Restart, recovery, and terminal failures

A shard is committed only when its semantic JSON and NPZ both exist and all binding,
hash, chain, shape, health, and committed-state checks pass. Committed pairs form one
contiguous prefix and are never overwritten.

Durable attempt journals reconcile hard crashes exactly once. Accounting uses the
larger of invocation wall time and newly committed child-owned shard time, with a
small conservative CPU/reconciliation allowance. The authenticated predecessor cost
is one explicit child carry event; imported reverse shard 0 is not double charged.

Success verification and failure verification are intentionally different:

- success requires every frozen parent/source/scientific predicate and every planned
  raw/derived artifact;
- a failure package verifies all predicates known to have passed, binds the observed
  versus expected failing predicate, preserves available raw evidence, and requires
  `learned_interpretation_authorized=0` when controls are inadequate;
- parent/source binding failures are not required to satisfy the predicate whose
  failure they document;
- source-closure incompatibility discovered before resume ownership is a no-write
  rejection, not an attempted in-place failure seal;
- unexpected or tampered child files are inventoried as untrusted failure evidence
  or leave the child unsealed/read-only; they can never be accepted by a success path
  set.

A source-control failure after all reverse shards still packages every trajectory,
metric, and image. A valid global adverse result with informative source and exact
controls is terminal scientific success.

## Artifact and claim boundary

The successful child contains the v2 provenance capsule, predecessor/parent/input
bindings, 64 imported forward pairs, imported reverse shard 0, 63 child-generated
reverse pairs, anchors, boundary/milestone arrays, metrics, mechanism summaries, raw
and demixed milestone images/contact sheets, reports, carried and child resource
records, manifest, checksums, storage authority, and terminal verification. Recovery
and failure artifacts appear when needed and are part of the expected path set.

This experiment may establish only a one-path exploratory complete-reconstruction
result under the exact backend. It cannot establish population performance,
reference-prior generation, sample diversity, architectural causality, a confidence
interval, or a confirmatory claim. Protected confirmation evidence remains unopened.

After terminalization, the repository-root handoff must record the exact outcome,
effect sizes, health, artifacts, and next action. A completed objective-bearing
reverse attempt resets the proxy-only counter; a pre-sampler or forward-only stop
does not.
