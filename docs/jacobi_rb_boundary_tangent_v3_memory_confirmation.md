# Immutable-cache v3 streaming-memory confirmation

## Purpose

This additive workflow repairs a deterministic CUDA-memory scheduling defect
in the zero-baseline boundary-tangent v3 experiment. It reuses the completed
train/validation cache read-only and streams all neural work in fixed batches
of 32. It does not regenerate the cache or alter the exact Jacobi transition,
Rao--Blackwell target, model, optimizer, random seeds, candidate family, or
statistical thresholds.

The restartable stages are

```text
preflight -> train -> select -> confirm -> report
```

There is deliberately no `cache` stage. The immutable parent supplies that
evidence. This workflow performs no controller trajectory, complete reverse
path, reconstruction, image sampling, reverse sampling, or full-data
training.

## Failed parent and OOM diagnosis

The immutable parent is

```text
runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_learnability/
  20260805-224211_production-zero-baseline-v3-certificate-semantics-fix
```

Its bindings are:

```text
artifact registry records                 2,085
artifact registry file SHA-256            6d50edf754a49105f70294b9a6bacd948b2155e9d1f4f614da83f7ae7005da91
artifact registry semantic SHA-256        4e018e1913e54ad8cf0dab79d027c609db030a2b13d35a7fbc69e184022ac723
source fingerprint                        860780ae957ea853d3d1254e20ab8d4db68339d9ecacdb48b730825cd07b47f9
scientific configuration SHA-256          c6ab05acd0a7f122646b4a5365fb85368b5a8243e062fbc4a57348793e256eb1
```

The cache gate passed. It contains 64 training paths and 32 validation paths,
with 114,688 and 57,344 selected rows respectively. All 202,309,632
transitions were certified. Maximum mass error was
`4.440892098500626e-16`, the forbidden-event count was zero, and the slowest
role sustained approximately `3,687.69` transitions/s. Cache construction
took `54,453.820496799715` seconds and persisted `1,091,008,340` bytes. Its
cache-plus-confirmation forecast was `92,961.73835921875` seconds, below the
unchanged 30-hour cap.

Training failed before any scientific control completed. The implementation
sent all 114,688 training rows through the width-32 convolutional model at
once. Its first activation alone required

```text
114688 * 32 * 28 * 28 * sizeof(float32) = 10.71875 GiB,
```

which cannot fit on the 7.96-GiB device. The reported 10.72-GiB allocation
therefore identifies a deterministic batching defect, not allocator
fragmentation. No allocator setting or rerun of the unchanged command can
repair it.

The parent directory remains byte-for-byte immutable. Its historical
`training_controls_failed` decision is retained, while the child records the
correct adjudication:

```text
decision                     prelabel_control_memory_schedule_invalid
failure_domain               implementation_contract
stage_execution_valid        0
scientific_evidence_complete 0
physical training            not performed
validation selection         not performed
confirmation                 not performed
```

## Immutable cache and memory contract

The child run writes `immutable_cache_binding.json` and verifies the parent
registry, cache gate, cache seal, role indexes, path plan, and cache-time
commitments before every stage. It reads cache artifacts directly from the
parent without copying, linking, rewriting, or regenerating them.

The memory schedule is frozen as follows:

- Inputs and raw labels remain contiguous, writable NumPy arrays on the host.
- At most 32 rows of model inputs and targets are transferred to CUDA for any
  model call.
- Full-cache activations, predictions, labels, and analytic targets are
  forbidden on CUDA.
- Every model invocation records its batch size; any batch above 32 fails the
  memory-contract gate.
- CUDA peak allocated memory must remain at most 80% of the selected device.
- Batch size is never adapted after observing memory use, and allocator
  environment changes are not part of the repair.

Physical-label firewalls remain unchanged. Controls can access permitted
inputs only. Training labels open only after every prelabel control passes.
Validation labels open only after the candidate and bootstrap plans are
sealed. Confirmation labels are streamed once and are never persisted.
Artifact hash verification may read bytes but must not deserialize protected
labels before the corresponding opening seal.

## Scientific and optimization invariants

The experiment retains all v3 scientific choices:

- Grid 28, alpha 1, `K=512`, and `tau_eff=5e-5`.
- Exact certified binary64 Jacobi transitions and raw Rao--Blackwell labels.
- Zero-baseline boundary-tangent model
  `m_theta(W)=y(1-y)q_theta(W)`.
- Plain unweighted MSE, normalized only by the training-target RMS.
- Width 32, Adam at `1e-3`, batch 32, 4,000 updates, checkpoints every 100
  updates, zero weight decay, unit gradient clipping, deterministic execution,
  and no mixed precision.
- Physical seeds `261312,261313,261314`, synthetic seed `261323`, exact-null
  seed `261324`, selection bootstrap seed `261320`, and confirmation bootstrap
  seed `261322`.
- The fixed 120-nonzero-checkpoint by 228-contrast validation family and the
  one-sided 99.5% whole-path max-T procedure with 50,000 replicates.

Streaming changes only execution:

- Update-zero scans reduce exact-zero diagnostics batchwise.
- Synthetic analytic targets are constructed per batch. Their training-only
  scale uses canonical row-order float64 squared sums combined with
  `math.fsum`.
- The exact-model null evaluates teacher and student batchwise and accumulates
  the exact global mean-loss gradient before its single Adam step. It must
  retain exact-zero gradients and bitwise-identical parameters.
- Physical optimization transfers only its selected 32-row training batch to
  CUDA and preserves the original deterministic batch-index stream.
- Validation predictions are copied from each 32-row CUDA batch into a host
  buffer before unchanged whole-path risk aggregation.

The fixed parent training and validation paths remain the cache evidence.
The reserved confirmation paths `0xF2000-0xF203F` remain scientifically fresh
because the failed parent never opened their namespace. They may be opened
once only after a nonzero validation nominee is sealed.

## Gates and outcomes

The additive implementation exposes:

```text
--stage {preflight,train,select,confirm,report,all}
--require-gate {none,preflight,train,select,confirm}
--failed-v3-train-run-dir
--resume-run-dir
--runs-root
--run-name
--device
```

Memory-specific closed outcomes are:

- `control_provenance_invalid`;
- `immutable_cache_binding_invalid`;
- `training_memory_schedule_invalid`;
- `training_memory_resource_infeasible`.

After preflight, the unchanged v3 scientific outcomes remain authoritative,
including `training_controls_failed`, `physical_training_invalid`,
`no_validation_candidate`, `validation_inference_invalid`,
`zero_baseline_v3_signal_not_confirmed`, and
`exact_rb_zero_baseline_boundary_tangent_time_local_signal_confirmed`.

If no nonzero validation candidate passes the search-aware gate, confirmation
remains unopened. A sealed nominee authorizes one confirmation only. Expected
gate failures and unexpected execution failures write all available gate,
decision, status, and registry evidence before returning nonzero.

## Production commands

Resolve the immutable parent and run a fresh preflight:

```powershell
$failedTrainRun = (
  Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_learnability/20260805-224211_production-zero-baseline-v3-certificate-semantics-fix"
).Path

$env:PYTHONHASHSEED = "0"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_boundary_tangent_v3_memory_confirmation `
  --runs-root runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_memory_confirmation `
  --run-name production-zero-baseline-v3-memory-safe `
  --device cuda `
  --stage preflight `
  --failed-v3-train-run-dir $failedTrainRun `
  --require-gate preflight
```

Resolve the new child after preflight reports `ready_for_train`:

```powershell
$memoryRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_memory_confirmation" -Directory |
  Where-Object Name -Like "*_production-zero-baseline-v3-memory-safe" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName
```

Run each remaining gate separately. Stop at the first nonzero exit:

```powershell
foreach ($stage in @("train", "select", "confirm")) {
  .\.venv\Scripts\python.exe `
    -m mnist.diag_d0_jacobi_rb_boundary_tangent_v3_memory_confirmation `
    --device cuda `
    --stage $stage `
    --resume-run-dir $memoryRun `
    --failed-v3-train-run-dir $failedTrainRun `
    --require-gate $stage

  if ($LASTEXITCODE -ne 0) { break }
}
```

If selection records `no_validation_candidate`, that is a valid negative
scientific result and confirmation is forbidden. An interrupted train or
confirmation stage resumes with the same command and child directory; it must
not create a replacement run or confirmation namespace.

## Restricted claim

A final pass establishes only that one width-32 phase-conditioned predictor
has a fresh, search-adjusted, time-local all-versus-zero signal for one frozen
image under the exact fixed-`K=512` Jacobi split chain. It does not establish
an executable reverse controller, reconstruction, sample quality, a known
prior, full-data generalization, convergence to the unsplit Eulerian
generator, or spatial Dirichlet--Ferguson convergence.

Only `exact_rb_zero_baseline_boundary_tangent_time_local_signal_confirmed`
authorizes planning the separate controls-only, at-most-eight-phase
controller study. It does not authorize controller execution or sampling.
