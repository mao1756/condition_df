# Eulerian Jacobi DDPM-v0 MNIST experiment

Date: 2026-08-15  
Primary research mode: exploratory  
Schema: `eulerian-jacobi-ddpm-mnist-v0`

## Decision and objective

This patch asks one objective-bearing question:

> Can one frozen ten-class global network, trained from scratch on exact Jacobi
> Rao--Blackwell denoising labels, drive the declared fixed-grid Eulerian split
> chain from its symmetric Dirichlet prior to recognizable, class-consistent,
> noncollapsed MNIST images?

The concrete success artifact is the complete set of 160 learned prior-start images
and their paired null trajectories, with forward-terminal and oracle controls. Every
endpoint is retained. There is no candidate generation followed by classifier,
shape, or human selection.

This is the first fresh population model in this formulation. It is not a
confirmation run, a continuum-limit claim, or proof that the implemented controller
is the exact discrete reverse kernel.

Proxy-only patches since the last objective-bearing experiment: 0.

## Why this is a major intervention

The historical Stage E experiment used one target-specific model and one prior path.
On that opened path, cutoff 216 improved terminal squared L2 over null by about
`0.5336%`, closed only about `0.5439%` of the null-to-oracle gap, and remained
noise-like. The source-informed oracle produced a recognizable 3 and improved over
null by about `98.12%`. That establishes a healthy one-target interface and a
dynamically negligible learned effect at exactly that scope. It does not establish
population failure or failure of every Eulerian learner.

This patch therefore does not tune the old checkpoint. It changes to a fresh
ten-class denoising model whose training law, terminal prior, reverse controller,
rendering, and end-to-end evaluation are declared together. A healthy negative ends
this v0 learner experiment and triggers a strategy review; it does not automatically
authorize another nearby learner tweak.

## Competing hypotheses

- The target, orientation, split order, path IDs, or controller interface is wrong.
  This predicts a failed theory/code audit, numerical audit, or oracle control.
- K=128 preserves nominal cumulative exposure but changes the finite split-chain
  law enough to matter. This predicts a material paired K=128/K=512 law or oracle
  discrepancy.
- The global predictor learns a useful conditional tangent field. This predicts
  improved forward-terminal reversal and recognizable prior-start images.
- The predictor works off-policy but not recursively. This predicts validation-loss
  improvement with weak or adverse learned rollouts.
- Terminal mixing or the Dirichlet start law is the main mismatch. This predicts a
  learned advantage from forward-terminal states but not from prior starts.
- The class input is weak. This predicts recognizable but systematically off-label
  learned samples.
- The controller is task-useful but collapses randomness. This predicts fidelity
  with duplicate endpoints or low within-class diversity.
- The fixed global architecture, direct target, controller parameterization, or
  present Jacobi strategy is inadequate. This remains an admissible explanation
  after clean controls.
- The evaluator or fixed rendering is misaligned. This predicts human/machine
  disagreement despite exact render replay.

## Fixed scientific contract

### State, data, and forward chain

The state is a unit-mass vector on the fixed 28 by 28 grid. The practical reference
law is `Dirichlet(1 * ones(784))`. MNIST images use the existing D0 uniform mix
`lambda_mix=0.35` before forward noising. Train, validation, and terminal-test roles
are fixed to ARFF rows `0:55000`, `55000:60000`, and `60000:70000`.

The forward model is the finite four-matching, seven-phase palindromic Jacobi split
chain. Production uses K=128 outer steps with the same nominal cumulative exposure
as the existing K=512 schedule. K=128 is a new numerical/scientific intervention:
equal cumulative exposure does not make its finite chain identical to K=512.

The reverse sampler composes the reference phases with the existing
boundary-preserving frozen-score logistic controller. It is described as a
boundary-preserving approximate reverse controller for the declared forward split
chain, not as an exact discrete reverse transition.

The controller uses the canonical `M=2` microsteps per phase. Because each controlled
phase is traversed backward, model times are evaluated in execution order at the
within-phase quantiles `q=0.75` and then `q=0.25`, using
`t=1-(7*k+phase+q)/(7*K)`. Thus every complete reverse path makes exactly
`128 * 7 * 2 = 1,792` model evaluations; the workload projection accounts for that
separately from forward pair transitions.

### Target and model firewall

For earlier and later active-pair fractions `X` and `Y` at exposure `u`, training
uses the direct Rao--Blackwell target

```text
bar_Z = Y * (1 - Y) * d/dY log k_u(Y | X).
```

The model emits finite `q_theta`; its loss output is
`m_theta = Y * (1-Y) * q_theta`. The regression never divides the target by the
mobility.

The only model inputs are later full state, reverse time, phase, matching/color,
phase duration, and requested class. Earlier state, source/target image, random
uniforms or bits, path ID, and the denoising target are forbidden.

The sole architecture is the 34,974-parameter
`GlobalDilatedZeroBaselinePredictor`: width 32, circular dilations 1/2/4/8, ten
classes, and no source conditioning. There is no `DirectFluxUNet` fallback and no
architecture sweep.

### Exact balanced paths and IDs

Training uses exactly 4,000 whole forward paths: 400 per class. Validation uses
exactly 1,000: 100 per class. Each path contributes four prespecified time-quartile
records, each containing all 392 labels from one active matching. The projected
forward workload is exactly `1,756,160,000` active pair transitions.

Sampling is a separate cost. Each reverse row executes
`128 * 7 * 4 * 392 = 1,404,928` reference pair transitions, and each forward start
costs `128 * 7 * 392 = 351,232`. The 420 reverse rows and 50 forward-noising legs
cost `607,631,360` transitions. Full base work is therefore `2,363,791,360`;
including the one-path K=128/K=512 audit it is `2,372,572,160`.

Fresh 20-bit path-ID roles are frozen as follows:

- `0xB0000:0xB2000` remains Haar-reserved;
- `0xB2000:0xB2100` is the fast-kernel preflight pool and
  `0xB2100:0xB2101` is the paired K=128/K=512 audit;
- pilot roles are `0xB2200:0xB22FA` (250 train),
  `0xB2300:0xB2364` (100 validation), `0xB2500:0xB2514` (20 prior),
  `0xB2520:0xB2534` (20 forward-terminal), and `0xB2540:0xB254A`
  (ten oracle);
- `0xB3000:0xB3FA0` is the exact 4,000-path training inventory;
- `0xB4000:0xB43E8` is the exact 1,000-path validation inventory;
- `0xB5000:0xB50A0` is the 160-path prior population;
- `0xB5100:0xB5128` is the 40-path forward-terminal panel;
- `0xB5200:0xB520A` is the ten-path oracle panel.

The runner records counts, half-open ranges, and hashes and scans legacy MNIST
source for collisions before production.

### Optimization

One model seed is trained with Adam at `2e-4`, batch 64, for 10,000 updates. EMA is
`0.999`, gradient norm is capped at 1, and validation runs every 250 updates. The
selected checkpoint is the earliest finite EMA checkpoint attaining the minimum
validation normalized MSE. Update zero is reported as the null baseline but is not
eligible for selection. The selected finite checkpoint is sampled even if it fails
to beat zero. There is no width, learning-rate, gain, schedule, or image-driven
checkpoint search.

### Populations and fixed rendering

Prior generation has 16 paths per digit, 160 total. The null and learned rows use
the same Dirichlet starts and paired transition identities. The forward-terminal
diagnostic uses the first four validation images per class. The interface positive
control uses one validation image per class and the same full controller/reference
composition.

States at completed reverse steps `0,32,64,96,128` are saved for both prior rows;
forward-terminal and oracle endpoints are also saved. Bad images, unstable prefixes,
and failures remain artifacts.
The Dirichlet prior states, requested labels, path IDs, and sample IDs are atomically
committed to `prior_start_authority.npz` and hashed before the first sampling call.
The later start bank and both trajectory anchor-zero arrays must replay it exactly,
so a sampler crash cannot erase or reorder the decision population.
After each named null, learned, or oracle population finishes, its starts, endpoint,
five anchors, identities, and telemetry are atomically written under
`population_stages/` before the next stage or post-stage resource check. Successful
population sealing requires all six stage files and checks them against the assembled
raw population. An operational failure seals whatever completed stage files exist in
the parent failure tree.

Rendering is a new frozen Eulerian binding, not the pixel-DDPM model-space transform.
For display only, mixed masses are background-demixed, clipped at zero, and
renormalized. A unit mass is converted with exact scale `25471/255`:

```text
uint8 = rint(255 * clip(demixed_mass * (25471/255), 0, 1)).
```

Raw mixed masses, demixed masses, and uint8 arrays are retained. Verification
replays this exact transform.

## Controls and preflight

Smoke and unit tests are CPU-only. The executable production path requires CUDA;
this implementation task does not itself launch CUDA compute.

Before full production the runner must pass and save:

1. architecture, firewall, path-ID, split-order, prior, mix, controller, and
   rasterization identity checks;
2. a 4,096-transition fast-versus-certified audit with maximum state error
   `2e-10`, target error `2e-8`, pair-total error `2e-12`, identical orientation
   and IDs, and no nonfinite value;
3. a one-path, real-28x28 paired K=128/K=512 law-and-oracle audit (8,780,800
   transitions) that reports both discrepancies, couples only aligned transition
   randomness, and never claims full-path common random numbers or chain identity;
4. 32 forward records, 25 optimizer updates, eight prior paths, and four
   forward-terminal paths through all 128 steps;
5. separate measured pilot and conditional-full projections. Pilot base work is
   `273,960,960` transitions (`282,741,760` including the shared audit); full base
   work is `2,363,791,360` (`2,372,572,160` including that audit).

That resource smoke cannot authorize the full workload scientifically. Before the
full 4,000/1,000-path run, the same `run` command executes an all-ten-class
objective-bearing pilot: exactly 250 training paths (25/class), 100 validation paths
(10/class), 750 updates, 20 prior paths (2/class), 20 forward-terminal paths
(2/class), and ten oracle paths (1/class). It saves paired null/learned/oracle images
and trajectories. Full scale is admitted only if the pilot is verifier-clean, Gate C
and all health checks pass, learned forward-terminal L1 beats null on at least 12 of
20 paths, aggregate L1 relative improvement is at least 1%, and learned-controller
RMS is finite and strictly positive. The already bound evaluator then scores the 20
fixed-render prior images on CPU without opening terminal-test rows. Scale additionally
requires learned requested-label top-1 accuracy of at least 0.20 and above null,
learned requested-class log probability above null on at least 12 of 20 paired starts,
and a positive mean paired log-probability improvement. Raw predictions, logits,
probabilities, and requested-class log probabilities are saved and verifier-replayed.
These are diagnostic routing thresholds, not
integrity or confirmatory claim gates. A healthy negative is a scientific stop that
preserves the pilot outputs and requires a new patch; it is not a reason to tune the
opened pilot.

If the pilot projection exceeds the supplied caps, the runner writes a sealed
resource stop without suppressing or resizing the science. A positive pilot triggers
a fresh remaining-cap check for the conditional full stage; insufficient full-stage
authority preserves the pilot result in the resource-stop tree. The user-selected
lifecycle is whole-run restart: an existing nonempty run directory fails closed,
there is no `--resume` surface, and an external pilot cannot be supplied or reused.

Every compute-stage cap check preserves the frozen 900-second terminalization reserve;
terminal sealing and review do not consume that reserve through compute admission.
The post-pilot full receipt records the measured active time, storage, CUDA peak,
caps, projected full cost, reserve, and ledger event prefix; verification recomputes
all three admission inequalities. Post-seal evaluation, review packaging, and
finalization are charged as explicit zero-reserve ledger stages.

The `run` command executes the numerical audits, measured resource smoke, and the
all-class objective pilot inside `objective_pilot/` under the same run root. Only
after a positive pilot plus a fresh full-resource admission does it execute the fixed
4,000/1,000-path experiment. A healthy learned-negative stops before scale; an oracle
failure routes to reverse-composition repair; an unhealthy pilot is invalid for
learner interpretation.

## Artifact and evidence firewall

The compact run root contains:

```text
config.json                       command.txt
status.json                       source_bindings.json
path_id_audit.json                resource_ledger.json
kernel_audit.json                 k128_k512_audit.json
preflight_projection.json         data_roles.npz
forward_records.npz               training_history.csv
selected_checkpoint.pt            start_banks.npz
prior_start_authority.npz/json
population_stages/*.npz           objective_pilot_admission.json
objective_pilot/                  pilot_admission_authority/
prior_classifier_outputs.npz      prior_classifier_metrics.json
populations.npz                   uint8_populations.npz
telemetry.csv                     metrics.json
contextual_ddpm_metrics.json      POPULATIONS_SEALED.json
TERMINAL_EVIDENCE_OPENED.json     review/
images/                           outcome.json (after review)
REPORT.md                         HANDOFF.md
artifact_manifest.json            SHA256SUMS.txt
```

The terminal-test loader and review-key creator both require and recheck
`POPULATIONS_SEALED.json`. The seal binds the selected checkpoint, start banks, raw
populations, atomic named-stage records, fixed uint8 populations, and telemetry.
For a full run, verification also replays the positive pilot admission against the
exact embedded `objective_pilot/` tree, source/data/device bindings, and compact
copied pilot admission/metrics/manifest authority. The admission stores only the
canonical relative path `objective_pilot`; absolute, traversal, alternate, and
external pilot paths are rejected. A sealed run therefore retains its pilot decision
authority when the whole tree is moved. Terminal test rows are not parsed before
that boundary.

The evaluator and contextual DDPM row come from the frozen conventional-DDPM run.
The runner admits only the user-accepted recovered run: checkpoint
`3d31d42a14fee0ecc72adc1644718a037cc48e649948407da6c272b981840c92`,
selection `e6cd9e49ca61237d3a10e9ad2fe0ad09f7a33ea22911fdd73fd99f3a07e4c668`,
contextual metrics `2e2fc75b6398f25a84bdaef0558c2f99c51117c71a009a3a94ed0afe8d27be33`,
manifest `79aa5d9ae1ca6615a46c9d699f947bea4b6a380cc32e86547cc7e49cee612953`,
and status `complete`.
Those exact files and hashes are rechecked before terminal evidence opens. The DDPM
row is contextual calibration only;
it is not a common-dynamics comparison or a superiority claim.
The original 0.97 evaluator-accuracy threshold is retained only as a diagnostic.
The user explicitly accepted the observed 0.9546 validation / 0.9476 test evaluator
for this exploratory run, so missing 0.97 does not veto or discard generated output.

## Metrics

Primary objective metrics are learned-prior recognizability, requested-label
agreement, and diversity, plus paired forward-terminal L1/L2/correlation differences.
Mechanism diagnostics are validation normalized MSE versus zero, controller RMS by
time quarter, maximum absolute `q`, maximum logit increment, and horizon-specific
effects. Health metrics are finite/nonnegative/unit-mass checks, pair-total error,
exact facets, fallback/forbidden-operation counts, K=128/K=512 discrepancy, runtime,
memory, storage, hashes, and firewall order.

The blinded review contains two learned and the two paired null endpoints per digit:
20 learned and 20 null. Allowed answers are digits 0 through 9, `noise`, and
`ambiguous`.

## Gates and claim boundaries

### Gate A: theory/code and evidence identity

Gate type: execution/integrity.  
Downstream action or claim controlled: training and interpretation.  
Exact proposition tested: the exact target orientation, permitted inputs, frozen
architecture, controller sign/facets, phase order, data mix, path IDs, prior, and
rasterization all match this contract.  
Why necessary: a mismatch changes the experiment or leaks unavailable information.  
Statistic and independent unit: deterministic contract checks and exact inventories.  
Pass condition: every check passes with no path-ID collision.  
Failure means: a localized implementation/binding defect.  
Failure does not mean: the learner or Eulerian strategy is scientifically false.  
Pass action: run numerical preflight.  
Fail action: repair only the mismatch and restart the same experiment.  
Ambiguous/invalid action: stop before training.

### Gate B: fast backend and K=128 admission

Gate type: execution/integrity.  
Downstream action or claim controlled: scaling the K=128 production chain.  
Exact proposition tested: the fixed fast kernel meets the stated certified-fixture
tolerances, preserves simplex/pair totals, and the paired K=128/K=512 law and oracle
discrepancies are finite and explicitly reported.  
Why necessary: an invalid or unmeasured numerical intervention makes learner
attribution uninterpretable.  
Statistic and independent unit: 4,096 transition cases and the fixed paired oracle
panel.  
Pass condition: all numerical bounds and health checks pass; finite-chain identity
is not claimed.  
Failure means: repair the fast kernel or reconsider K=128.  
Failure does not mean: replace the target or declare the learner failed.  
Pass action: run the measured workload preflight.  
Fail action: stop before full training.  
Ambiguous/invalid action: retain both audit outputs and do not scale.

### Gate C: interface positive control

Gate type: execution/integrity for learned attribution.  
Downstream action or claim controlled: interpreting learned image failure.  
Exact proposition tested: the oracle improves endpoint L1 over null on at least 9 of
10 paths and has lower aggregate L1, with clean health.  
Why necessary: failure would implicate composition rather than the learner alone.  
Statistic and independent unit: ten whole oracle paths.  
Pass condition: at least 9 paired improvements plus aggregate improvement.  
Failure means: repair or revise the controller/reference composition.  
Failure does not mean: no learned signal exists.  
Pass action: interpret the learned rows.  
Fail action: stop; do not retrain.  
Ambiguous/invalid action: preserve paths and localize the invalid control.

### Pilot task-signal scale admission

Gate type: exploratory diagnostic threshold.  
Downstream action or claim controlled: spending the conditional 2.37-billion-transition
full-stage budget; it does not control a confirmatory claim.  
Exact proposition tested: the learned 20-image prior row contains nontrivial paired
requested-class signal under the frozen evaluator and renderer.  
Why necessary: forward/off-policy improvement alone does not justify scaling a
generator whose prior outputs lack task-level signal.  
Statistic and independent unit: 20 whole prior paths, two per requested digit.  
Pass condition: learned requested-label top-1 accuracy is at least 0.20 and strictly
above null; learned requested-class log probability beats null on at least 12 paired
paths; mean paired log-probability improvement is strictly positive.  
Failure means: route `pilot_prior_negative_stop_before_scale` and review terminal-law,
conditioning, architecture, controller, or strategy alternatives.  
Failure does not mean: every Eulerian model or the forward predictor universally fails.  
Pass action: combine with Gate C, health, and forward-dynamics thresholds for full
scale admission.  
Fail action: preserve all pilot outputs and stop before scale.  
Ambiguous/invalid action: repair only the evaluator/render replay discrepancy; do not
tune on the opened pilot.

### Diagnostic D: learned reverse behavior

Gate type: diagnostic threshold.  
Downstream action or claim controlled: strategy review, never sampling execution.  
Exact proposition tested: learned validation and full trajectories improve over
their paired null rows at practically visible scale.  
Statistic and independent unit: whole path/image; validation MSE is a mechanism
proxy.  
Pass condition: report direction, magnitude, horizons, and images; there is no
sampling veto.  
Failure means: this fixed learned mechanism is weak under the tested scope.  
Failure does not mean: every component or all Eulerian models fail.  
Pass action: evaluate exploratory image feasibility.  
Fail action: include the evidence in the final strategy decision.  
Ambiguous/invalid action: do not substitute proxy loss for images.

### Diagnostic E: exploratory image feasibility

Gate type: diagnostic action-routing threshold, not a confirmatory claim gate.  
Downstream action or claim controlled: recommending a frozen model-seed replication.  
Exact proposition tested: among 160 learned-prior outputs, classifier requested-label
accuracy is at least 0.70, at least 150 endpoints are unique, diversity ratio is at
least 0.25, and among the 20 reviewed learned outputs at least 15 are recognizable
and 12 match the request; human and machine agreement both strictly exceed null.  
Why necessary: small scalar improvements can remain noise-like.  
Statistic and independent unit: generated image/path; 20 prespecified human-review
images and all 160 machine-scored images.  
Pass condition: every stated marker plus Gates A--C.  
Failure means: this fixed v0 recipe lacks the required exploratory image evidence.  
Failure does not mean: population-level impossibility, no signal, or universal
Eulerian failure.  
Pass action: freeze v0 and plan one fresh model-seed replication.  
Fail action: stop this v0 learner experiment and perform a high-level strategy review
of a materially different fixed-grid score/controller or stopping the hypothesis.  
Ambiguous/invalid action: audit only the specific evaluator/rendering disagreement;
do not tune on the opened images.

There is no confirmatory claim gate in this patch.

## Outcome-to-action table

| Observation | Interpretation | Required next action |
|---|---|---|
| Gate A fails | Experiment identity or leakage is invalid | Repair the exact mismatch and restart in a new directory |
| Gate B fails | Fast backend or K=128 intervention is not admitted | Repair/audit the kernel or reconsider K=128 before training |
| Gate C fails | Full composition is not a valid positive-control path | Fix composition; do not blame or retrain the learner |
| Oracle passes; forward-terminal improves; prior fails | Terminal mixing/prior mismatch is a leading explanation | Stop v0 and conduct a strategy review focused on terminal law/exposure; no automatic exposure tuning |
| Oracle passes; both learned rows fail or remain noise-like | Architecture, target learning, controller, or strategy is inadequate at tested scope | Stop v0; compare one materially different fixed-grid formulation or stop the Jacobi learner hypothesis |
| Images are recognizable but systematically off-class | Class conditioning is inadequate | Stop and review label conditioning as a major intervention; do not post-hoc retrain on opened outputs |
| Images are class-consistent but collapsed | Learned dynamics do not preserve useful diversity | Stop and review prior mixing/controller randomness; do not add rejection |
| Human and machine judgments diverge | Evaluator or rendering may be misaligned | Audit only the common evaluator/render replay before a learner conclusion |
| Diagnostic E passes | Exploratory fixed-grid Eulerian Jacobi DDPM-v0 works at this scope | Freeze artifacts and plan one fresh model-seed replication; do not auto-launch |
| Result cannot distinguish branches | Experiment or controls were underdesigned | Do not scale or tune; redesign the direct comparison |

## Resource budget and stop rule

The source-level planning estimate is 60,000 wall seconds, 50,000 accelerator
seconds, 8 GiB peak memory, and 1 GiB persisted storage. These are estimates, not
launch authority. Production CLI caps are required numeric arguments and a real,
non-placeholder approval ID is mandatory.

The production preflight replaces the estimate with separately measured
cache-transition, optimizer-update, forward-leg, reverse-sampling, storage, and
terminal-reserve projections. The pilot is allowed when its own budget fits even if
the conditional full stage does not. After a positive pilot, remaining full cost is
checked again. Exceeding the required stage cap produces a sealed resource stop; it
never modifies the science. Pilot and full execution use fresh whole-run restart
under identical verified bindings and explicit caps.

Scientific decision purchased by this cost: whether the assembled fresh denoising
mechanism produces prior-start MNIST images. A smaller local-label or suffix-only
experiment cannot answer it.

## Commands

Run the CPU synthetic lifecycle smoke:

```powershell
.\.venv\Scripts\python.exe -B -m mnist.diag_eulerian_jacobi_ddpm_mnist smoke `
  --output-dir .\tmp\eulerian-jacobi-ddpm-smoke
```

Production is CUDA-only and is not launched by this implementation task. The command
below validates a real approval and numeric caps, executes the objective pilot, and
continues to full scale only on a positive pilot and a second full-resource admission.
Replace `APPROVAL_REFERENCE` with a real non-placeholder approval identifier:

```powershell
.\.venv\Scripts\python.exe -B -m mnist.diag_eulerian_jacobi_ddpm_mnist run `
  --run-dir .\runs\experiment14-eulerian-jacobi-ddpm\fixed-grid-v0 `
  --arff .\mnist_data\mnist_784.arff `
  --ddpm-run-dir .\runs\experiment13-conventional-ddpm\pixel-ddpm-calibration-v1-cpu-recovered `
  --device cuda:0 `
  --approval-id APPROVAL_REFERENCE `
  --max-active-seconds 86400 `
  --max-storage-mib 2048 `
  --max-cuda-fraction 0.75
```

Complete the copied blinded CSV manually, then record it:

```powershell
.\.venv\Scripts\python.exe -B -m mnist.diag_eulerian_jacobi_ddpm_mnist record-review `
  --run-dir .\runs\experiment14-eulerian-jacobi-ddpm\fixed-grid-v0 `
  --answers .\human_review_answers.csv `
  --reviewer "REVIEWER NAME" `
  --confirm-manual-review
```

Verify a sealed tree without running production compute:

```powershell
.\.venv\Scripts\python.exe -B -m mnist.diag_eulerian_jacobi_ddpm_mnist verify `
  --run-dir .\runs\experiment14-eulerian-jacobi-ddpm\fixed-grid-v0
```

## Claim language

A positive result may establish only that this frozen exploratory v0 recipe produced
the stated task artifacts under one model seed and fixed populations. A negative
result establishes only that no candidate satisfied the exact v0 diagnostics under
this architecture, target, K=128 chain/controller, paths, renderer, evaluator, and
opened review. It does not establish absence of useful local signal, failure of all
fixed-grid Eulerian scores, a continuum result, or a population impossibility.
