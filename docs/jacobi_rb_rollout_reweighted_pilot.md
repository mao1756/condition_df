# Rollout-reweighted Stage E pilot

Date: 2026-08-15

Primary mode: exploratory.

## Decision

Does a label-preserving, rollout-proximity reweighting of the already authenticated
forward RB examples produce a learned cutoff-216 controller that transfers to three
fresh prior-start paths, or should the project stop this learner-side route and pivot
to a conventional DDPM baseline?

This is the final learner-side intervention in the current Jacobi/RB controller
family. It is objective-bearing because it ends in three complete prior-start reverse
trajectories and saved images. It does not open the protected confirmation paths and
does not create RB labels on reverse-rollout states.

Proxy-only patches since the last objective-bearing experiment: 1.

## Evidence roles

- Training: the opened `0xF8100-0xF813F` forward RB cache.
- Validation and checkpoint selection: the opened `0xF8200-0xF821F` cache.
- Development-only weighting reference: the completed Stage E
  `global-cutoff-216` trajectory on path `0xFB301`.
- Fresh exploratory evaluation: `0xE9008`, `0xE9009`, and `0xE900A`.
- Protected confirmation: `0xF9000-0xF903F`, unopened.

The originally proposed fresh IDs `0xFB302-0xFB304` are inside an existing reserved
replication slot. The replacement block above was frozen only after both the semantic
namespace scanner and committed numerical-artifact scanner found no collision.

## Reweighting intervention

The sealed eager cache contains eight phase midpoints, `(1,3,...,15)/16`, and no
phase-endpoint row. Therefore the literal planned endpoint fraction `1` is absent.
The implementation makes the smallest disclosed correction:

1. retain every original training example exactly once;
2. consider duplication only for the nearest available endpoint midpoint, `15/16`;
3. restrict duplication to selected outer steps `303,319,...,511`, where the
   cutoff-216 learned controller is active;
4. compare each permitted input state `W` with the opened Stage E pre-step boundary
   `R_(511-k)` in binary64 squared L2;
5. compute the median distance independently for every `(outer_step, phase)` using
   training inputs only;
6. duplicate every eligible training row at or below its cell median, including
   ties, in ascending original-row order;
7. apply the frozen training thresholds to validation for diagnostics only.

The `15/16` row is a pre-step-boundary proxy with a disclosed one-sixteenth-phase
coordinate mismatch. The weighting still depends only on permitted model inputs and
fixed opened rollout evidence, never on the denoising target, certificate codes, or
validation outcomes. Positive `W`-dependent weighting preserves the population
conditional mean `E[Zbar | W]`; it does not guarantee that a finite model or finite
optimization path is unchanged.

## Fine-tuning contract

- Model: the frozen 34,974-parameter global-dilated predictor initialized from the
  authenticated update-3100 checkpoint.
- Seed: `261405`.
- Optimizer: fresh Adam, learning rate `1e-3`, betas `(0.9,0.999)`, epsilon `1e-8`,
  no weight decay, batch size 32, gradient clip 1.
- Maximum updates: 4,000; validation every 100 updates.
- Selection: lowest finite reweighted-validation MSE among nonzero updates, with the
  earlier update winning exact ties. Update zero is logged but ineligible.
- Raw RB target and target RMS remain unchanged. No AMP, dropout, schedule sweep, or
  confirmation data.

## Fresh prior-start evaluation

Each fresh path is a separate four-row fused family with shared start and transition
randomness:

1. `zero`;
2. `frozen-cutoff-216` using the parent checkpoint;
3. `reweighted-cutoff-216` using the selected fine-tuned checkpoint;
4. `source-informed` oracle.

The candidate exploration backend uses the frozen approximate Jacobi proposal
contract. Each path runs the complete 512 reverse steps and saves every eight-step
boundary plus raw and demixed images at steps 0, 16, 128, 208, 216, 224, 256, 384,
and 512.

## Metrics and gates

Execution/integrity gates require finite nonnegative states, mass error at most
`2e-12`, no forbidden/invalid/fallback/correction events, exact self-chain and shared
RNG binding, exact cutoff masking after step 216, CUDA allocation below 80%, storage
below 500 MiB, and the approved active-time cap.

The oracle-attribution threshold is diagnostic: the oracle must improve zero by at
least 1% on every healthy path before learner attribution.

The practical-success diagnostic requires all of:

- reweighted beats frozen at the endpoint on at least two of three paths;
- aggregate mean reweighted error is lower than aggregate frozen error;
- median reweighted oracle-gap closure is at least 10%;
- median endpoint centered correlation is at least 0.5;
- human review identifies the reweighted endpoint as target label 3 on at least two
  of three paths.

There is no confirmatory claim gate and no automatic follow-on compute.

## Outcome routing

| Observation | Required action |
|---|---|
| Input, weighting, or checkpoint integrity fails | Repair only the localized defect and rerun unchanged. |
| Numerical, pairing, or cutoff identity fails | Repair only the localized execution defect and rerun the same path. |
| Oracle fails on any healthy path | Repair the common prior/oracle/composition path before learner attribution. |
| Practical-success diagnostic passes | Implement and execute a bounded Stage F experiment after a fresh compute approval. |
| Candidate is materially better near 208/216 but loses most benefit by 512 | Compare one materially different state/time-dependent controller; do not tune another static cutoff. |
| Candidate is numerically small/noise-like or matches/loses to frozen without the collapse signature | Pivot to a conventional DDPM baseline. |

## Resource envelope and stop rule

Expected active time is 2,500-4,500 seconds: roughly 600-1,800 seconds for fine
tuning and 1,600 seconds for three fresh evaluations. Expected peak GPU allocation is
well below 80%; expected storage is below 500 MiB. The initial approved cap is 7,200
active seconds. The user explicitly authorized any resource or compute cap needed for
this implementation-and-run request; any amendment must still be recorded before
additional work.

Stop automatically on projected active-time, storage, CUDA-memory, integrity, or
numerical failure. Preserve the longest valid prefix and all adverse images.

## Claim boundary

The independent unit is one prior path (`n=3`). Results apply only to one opened
target-specific checkpoint, the disclosed rollout-proximity reweighting, the frozen
cutoff-216 controller, and the approximate candidate law. They do not establish an
exact-law, population, diversity, confirmatory, or general generator claim.

