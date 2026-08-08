# Sealed boundary-tangent false-discovery adjudication

## Scope

This additive workflow performs a read-only forensic adjudication of the
completed eager-prefix boundary-tangent v2 experiment. It creates no Jacobi
transition, path, physical label, checkpoint, optimizer update, controller
trajectory, reconstruction, or sample. It never mutates the parent run.

The workflow is implemented by
`mnist.diag_d0_jacobi_rb_boundary_tangent_false_discovery_adjudication` and
has the restartable stages

```text
preflight -> adjudicate -> decision
```

The historical v2 decision remains `selection_false_discovery` regardless of
the child result. The child can authorize at most the design of a fresh v3
learnability experiment; no child outcome authorizes controller planning or
execution.

## Immutable parent and observed failure

The sole source-of-record parent is

```text
runs/experiment12_d0_jacobi_rb_boundary_tangent_eager_confirmation/
  20260803-113404_production-eager-boundary-tangent-time-local
```

The forensic preflight binds the following exact records:

- terminal decision `selection_false_discovery` with complete scientific
  evidence;
- source fingerprint
  `dfe9c3357c1d1ba614cccfdcaca84b3c3bf2d0967d6a3a3b15e5a0421d04243e`;
- scientific-configuration SHA-256
  `fadc1eb31ad0fb1ccb900f41f1eb8523c67c6ae39e09c783698aa5a20634cdec`;
- 3,457-artifact terminal registry with semantic SHA-256
  `36bf43c0a108549954617a78625d4fd65820141c950ba84330133de1f8648580`.

The v2 optimizer selected seed `261314`, update `800`. On the 32 validation
paths it improved over the fitted baseline by only
`1.5723577897475138e-6` overall and
`1.7809627570919417e-5` at high reverse time, while already losing to zero by
`-0.016184475047380076` overall and `-0.013052578417634741` at high reverse
time. The old selection rule searched 120 nonzero candidates but required only
pointwise improvement over the baseline.

The sealed 64-path confirmation required every one of 224
combined-versus-zero and four combined-versus-baseline simultaneous lower
bounds to be positive. Instead, all four residual-versus-baseline quartile
point estimates and all 224 combined-versus-zero point estimates were
negative. The execution itself was exact, certified, numerically healthy, and
inside the resource limit. This is therefore a selection/confirmation
contract failure, not a transition-kernel failure.

## Why the confirmation paths are burned

The confirmation namespace `0xED000-0xED03F` has been opened and its outcomes
have been inspected. Those paths may be used only for this explicitly
post-hoc, non-authorizing forensic analysis. Reusing them to select a model,
fit a baseline, tune a threshold, or confirm a future learner would make them
part of the design process and destroy their status as untouched audit data.

Accordingly, every child artifact records

```text
posthoc_non_authorizing = 1
old_confirmation_paths_burned = 1
controller_planning_authorized = 0
```

Any later v3 experiment must seal its selection before creating an entirely
fresh confirmation namespace.

## Forensic preflight gate

Preflight verifies the complete parent registry and the sealed train and
confirmation chains. In particular, it requires:

- three physical task records, three histories, and exactly 120 hash-bound
  nonzero checkpoint files at updates `100,200,...,4000` for seeds
  `261312,261313,261314`;
- the exact update-zero baseline checkpoint for each seed and the historical
  selection hash for seed `261314`, update `800`;
- the validation role index, all referenced input/label shards, their
  complete Cartesian identities, and the fixed 32 validation path IDs;
- the selection, train, and confirmation seals;
- all 448 confirmation metadata/state shards, all 224 risk shards, and the
  confirmation index and aggregate artifacts;
- disjoint train, validation, and confirmation namespaces;
- absence of persisted raw confirmation labels.

Missing, changed, duplicated, nonfinite, or role-mismatched evidence ends
`forensic_evidence_invalid`. Confirmation IDs are rejected by every
validation replay function.

## Sealed baseline re-adjudication

Each authoritative risk shard already stores the rowwise float64 contrasts

```text
combined_vs_zero
combined_vs_baseline
baseline_vs_zero
```

The replay first requires

```text
abs(combined_vs_zero
    - baseline_vs_zero
    - combined_vs_baseline) <= 5e-15
```

for every row. It then reconstructs the original 228-member path table and
requires exact agreement with the parent aggregate, point estimates, and
direct/derived identity. Only after that control passes does it aggregate
`baseline_vs_zero` into 229 path-level components: 224
quartile/phase/midpoint cells, four quartiles, and one overall contrast.

A deterministic two-sided 99.5% whole-path studentized max-|T| interval uses
50,000 Philox replicates, `higher` quantile interpolation, and a new fixed
audit namespace. The direction is two-sided because this omitted contrast was
inspected before the child workflow was designed.

Closed baseline classifications are:

- `sealed_baseline_advantage_confirmed` when every simultaneous lower bound
  is positive;
- `sealed_baseline_harm_confirmed` when the overall and all four quartile
  simultaneous upper bounds are negative;
- `sealed_baseline_not_established` otherwise;
- `sealed_baseline_evidence_invalid` on any replay or integrity failure.

This analysis cannot rescue the old v2 confirmation or authorize a
controller.

## Selection-resolution audit

The adjudicator evaluates each of the 120 existing nonzero checkpoints on the
sealed 32-path validation cache. It performs no training, fine-tuning,
averaging, or retrospective checkpoint selection. Evaluation uses only the
permitted later-state inputs and direct float64 MSE against the unchanged raw
Rao--Blackwell target.

Replay controls require update zero to equal the baseline exactly, every
stored aggregate candidate metric to reproduce within the frozen tolerance,
and the historical point-selection rule to recover seed `261314`, update
`800` and its selection hash. Failure ends
`implementation_or_replay_defect`.

For each nonzero candidate the replay produces the confirmation-shaped 228
path contrasts. Search-aware inference jointly resamples whole validation
paths across the 480-component family

```text
120 candidates x 4 combined-vs-baseline quartiles
```

using one-sided 99.5% studentized max-T bounds with 50,000 deterministic
Philox replicates. A residual signal is selection-resolved only if one
candidate has all four simultaneous lower bounds above zero. Separately, all
228 validation point estimates must be positive for directional compatibility
with the later confirmation contract. That sign screen is descriptive and is
not a confidence claim.

Residual classifications are scoped strictly to this width-32 model, these
three seeds, and these 40 nonzero checkpoint times:

- `current_candidate_family_residual_signal_resolved`;
- `selected_update_below_resolution`;
- `residual_signal_directionally_incompatible_with_zero`;
- `selection_audit_inconclusive`;
- `implementation_or_replay_defect`.

## Closed child decisions

The terminal decision is one of:

- `forensic_evidence_invalid`;
- `implementation_or_replay_defect`;
- `retained_baseline_v3_selection_design_ready`;
- `baseline_only_requires_fresh_confirmation_design`;
- `zero_baseline_v3_learnability_ready`;
- `baseline_and_residual_unresolved`;
- `selection_resolution_failure_confirmed`.

The maximum authorization is design work for a fresh v3 learner. In
particular, confirmed baseline harm permits planning a zero-baseline v3
learner, while a below-resolution selected update permits repairing the
selection design. Neither result authorizes reuse of the old confirmation
paths.

## Prospective v3 rule

The minimal future selection repair must account for every searched
checkpoint and every confirmation-shaped validation contrast. For the frozen
three-seed, 40-checkpoint design, the family has

```text
120 x 228 = 27,360 candidate/contrast pairs.
```

Shared whole-path bootstrap resamples are processed in bounded
candidate/component blocks. A candidate is eligible only when all 228
search-adjusted lower bounds are strictly positive. Among eligible candidates,
select the largest minimum lower bound, then the earlier update, then the
lower seed. If none qualifies, seal `no_validation_candidate` and never
create confirmation paths.

If the adjudication confirms that the fitted 87,808-cell baseline is harmful,
the smallest exact-target representation repair is `q_B := 0` with exact zero
initialization. This changes neither the Jacobi law nor the raw
Rao--Blackwell target and introduces no parameter tuned on the burned audit.

## Production command

Run the bounded child workflow once against the immutable parent:

```powershell
$parent = (Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_eager_confirmation/20260803-113404_production-eager-boundary-tangent-time-local").Path

.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_boundary_tangent_false_discovery_adjudication `
  --runs-root runs/experiment12_d0_jacobi_rb_boundary_tangent_false_discovery_adjudication `
  --run-name production-sealed-false-discovery-adjudication `
  --parent-run-dir $parent `
  --stage all `
  --bootstrap-replicates 50000 `
  --require-gate decision
```

Each stage commits readable evidence atomically before a required-gate failure
returns nonzero. Completed parent artifacts are read only and are never
rewritten, copied into place as replacements, or resumed.
