# Legacy Experiment-10 edge-flux factor-one replay

Status: archived optional compatibility diagnostic; not the active mainline plan.

The complete historical proposal remains preserved in the frozen planning bundle at
`handoff/gpt_pro_eulerian_edge_flux_fresh_replay_planning_20260815/`. This archive
note records its corrected scientific role after the fixed-grid Jacobi/Rao--Blackwell
plan was adopted.

## Correct scope

The proposed replay is **candidate-selection-free**, because it would retain one
generated endpoint per request without the historical classifier/composite selector.
It is not generally “rejection-free”: the underlying adaptive numerical routine may
still reject and retry internal substeps as part of its declared integrator.

The pinned Experiment-10 checkpoint predicts a directly trained Poisson/optimal-
transport flux associated with a source-to-image interpolation. It is not a learned
score for the free Eulerian forward marginal and therefore is not the repository's
theoretically matched DDPM-like Eulerian score experiment.

The target-informed teacher in that proposal is an interface/composition control.
It is not a realizable generative baseline and cannot establish learned-model quality.

## If this replay is ever run

- Keep it exploratory and bind the historical checkpoint hash before safe loading.
- Use only the minimum extraction, source, artifact, and verification machinery needed
  for an interpretable compatibility result.
- Preserve all factor-one outputs and disclose the current-source sampler law.
- Do not use a weighted-point DSM as the automatic fixed-grid fallback; that is a
  materially different representation requiring its own decision and experiment.
- Do not treat a positive replay as completion of the Eulerian DDPM-like objective.

The active mainline implementation is documented in
`docs/eulerian_jacobi_ddpm_mnist.md` and uses the matched Jacobi/Rao--Blackwell target,
the boundary-preserving logistic controller, an all-class objective pilot, and a
conditional full population experiment.
