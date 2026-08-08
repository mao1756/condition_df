# Exact eager-prefix boundary-tangent scheduling confirmation

## Purpose and parent result

This controls-only experiment asks whether changing only the timing of an
already-defined Philox prefix can make the exact `64/32/64` boundary-tangent
workflow fit its frozen 30-hour budget.

It binds the immutable fused-scheduler run
`20260802-174811_production-fused-boundary-tangent-schedule`. That run was
numerically valid and scientifically complete, but its pilot projected
`337,182,720` exact transitions to `128,438.3003` seconds (`35.6773` hours) at
`2,625.2506/s`. The only failures were the unchanged 108,000-second and
`3,122.0622/s` resource thresholds. Certification fraction was one, fallback
fraction/time were approximately `1.2654e-7/5.0262e-6`, maximum mass error was
`4.4409e-16`, and every forbidden-event count was zero. No cache, training,
controller trajectory, reconstruction, or sampling ran.

The parent binding is frozen at 614 registered artifacts, registry semantic
SHA-256 `3a9372dee33287c2bc3d2e2752d6b206a44cd070c91305dc9d921bb2521e688e`,
registry-file SHA-256
`23ae4a0b3448b57b4982c9d61eca2ea5df1834057cc7035dc1084b0c28cf6ea9`,
source fingerprint
`c562b8e39a07bbc19a9f65b9a3187c49d97cb64a277bf9492f4cb7fb92c9b2ee`,
and scientific-configuration SHA-256
`0820d8e5051adae996d8c79bc1e933c8725e958aba62ff6c4bed1525a6cc6845`.
It is re-adjudicated only as an exact scheduling resource failure.

## Eager-prefix invariant

The stateless Philox key and transition ID define one infinite dyadic uniform.
The adaptive certificate normally exposes 64 bits first and reveals the same
second word only after escalation. The eager profile makes those 128 bits
available at the first `m=128` proof bucket. It freezes:

- candidate modes `128` and 56 bisection steps;
- 128 threads per block;
- the existing correctly-rounded CUDA certificate and Arb fallback;
- every transition ID, exposure, Jacobi law, later fraction, and raw
  Rao--Blackwell target.

Thus only proof scheduling and its mode/prefix diagnostics may change. The
rounded later fraction and target must be bit-identical to the adaptive path.
The convenience precision-doubling profile is not used because it would also
change the nonauthorizing proposal.

The parent P10 base evidence explains the intervention: `5,258,565` lanes
finished at mode/prefix `128/64`, while only `9,915` lanes (`0.1882%`) reached
`4112/128`. Those rare lanes dominate divergent warp work. This diagnosis is
performance evidence, not a change to the probability law.

## Staged gates

`preflight` verifies immutable provenance, the same-uniform contract, prefix
interval nesting, exact adaptive/eager state and target equality on fixed
base and midpoint panels, facets and zero cases, Arb agreement, and
permutation/chunk/resume invariance.

`profile` runs three paired, cyclic adaptive/eager timing repeats on sealed
P10 windows. It uses the fastest adaptive and slowest eager authorizer times
in the conservative parent projection. Only the parent's separately recorded
base-authorizer component is accelerated; branch conversion, I/O, model, and
all other unseparated pipeline time remain fixed. This requires about a
`1.354x` authorizer speedup rather than the optimistic `1.214x` obtained by
scaling the whole proof-dominated wall time. It passes only if outputs remain exact,
every transition is certified with zero fixed-panel fallback or forbidden
events, and the projection is at most 108,000 seconds and at least
`3,122.0622/s`.

`pilot` repeats the complete four-profile experiment with the frozen eager
profile. It retains `[10x9,6]` cache and `[10x6,4]` streaming cohorts, four
16-step time windows, three cyclic repeats, slowest-repeat projection, the
4,096-lane cap, and every existing certification, conservation, memory,
storage, persistence, and exact-count threshold. No timing average, rerun, or
post-hoc allowance is permitted.

Principal atomic artifacts include parent provenance and re-adjudication,
the eager-prefix contract and forensic histogram, adaptive/eager equivalence
records, sealed profile timing records and qualification, the selected
profile, complete pilot shard/repeat registries, projection and numerical
metrics, stage gates, final decision, status, and terminal artifact registry.

Only `exact_boundary_tangent_eager_prefix_schedule_feasible` authorizes
integrating this frozen execution profile into a fresh v2 cache/training/
confirmation workflow. It does not authorize cache generation, neural
training, controller trajectories, reconstruction, or sampling.

## 2026-08-03 Arb-escalation repair

The first production preflight,
`20260803-005757_production-eager-prefix-boundary-tangent-schedule`, is
immutable failed execution evidence (19 registered artifacts; registry
semantic SHA-256
`8bcc588b79f9ed45d540a9fab9d2cd4ee3d1b69721e64bbd85b20b0f59cd2d92`).
Its 250,880-transition adaptive/eager comparison passed bit-for-bit.  The
later one-hot zero-mass fixture exposed a genuine rare fallback gap on its
single active edge: the nonauthorizing CUDA candidate was 1,156 binary64
ULPs from the certified quantile, beyond the legacy local Arb search radius
of 256 ULPs.

The versioned additive repair retains that fast local search and, only when
it is exhausted, runs the existing full certified Arb inverse CDF and target
evaluation on the same stateless Philox stream.  It also distinguishes
active density rows (certificate code 15) from exact structural zero-mass or
zero-duration rows (code 0); inactive rows remain exact no-ops and never
pretend to carry a density certificate.  No candidate, law, exposure,
rounded state, target, or random stream changed.

Fresh preflight
`20260803-021405_production-eager-prefix-boundary-tangent-schedule-arbfix-v2`
passed all checks and ended `ready_for_profile`.  Its 20-artifact registry
semantic SHA-256 is
`a04ba41709c1dab211d747cebeef677cf5a700e1b795247ce1ccf1bba1e2b8f5`.
The repaired lane certified the exact values
`Y=0.4996324195179097` and `Z=25.60541796782855` with a 128-bit eager
prefix; all forbidden-event counts were zero.

## Production commands

```powershell
$scheduleParent = (Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_schedule_feasibility/20260802-174811_production-fused-boundary-tangent-schedule").Path

.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_boundary_tangent_prefix_schedule_confirmation `
  --runs-root runs/experiment12_d0_jacobi_rb_boundary_tangent_prefix_schedule_confirmation `
  --run-name production-eager-prefix-boundary-tangent-schedule `
  --device cuda `
  --stage preflight `
  --parent-schedule-run-dir $scheduleParent `
  --require-gate preflight
```

Resolve the fresh run only after preflight passes:

```powershell
$prefixRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_boundary_tangent_prefix_schedule_confirmation" -Directory |
  Where-Object Name -Like "*_production-eager-prefix-boundary-tangent-schedule" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName
```

Run the sealed profile qualification:

```powershell
.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_boundary_tangent_prefix_schedule_confirmation `
  --device cuda `
  --stage profile `
  --resume-run-dir $prefixRun `
  --parent-schedule-run-dir $scheduleParent `
  --require-gate profile
```

Only after the profile gate passes, run the complete pilot:

```powershell
.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_boundary_tangent_prefix_schedule_confirmation `
  --device cuda `
  --stage pilot `
  --resume-run-dir $prefixRun `
  --parent-schedule-run-dir $scheduleParent `
  --require-gate pilot
```
