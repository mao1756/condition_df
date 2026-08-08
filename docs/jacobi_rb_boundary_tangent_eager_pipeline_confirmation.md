# Exact eager-prefix boundary-tangent complete-pipeline confirmation

## Scope

This controls-only workflow asks whether the already verified eager-prefix
certificate schedule makes the complete frozen boundary-tangent workload fit
inside its preregistered 30-hour resource budget. It changes neither the
Jacobi transition law nor the exact Rao--Blackwell target. It generates no
production cache, trains no model, runs no controller trajectory, and performs
no reconstruction or sampling.

The workflow is implemented by
`mnist.diag_d0_jacobi_rb_boundary_tangent_eager_pipeline_confirmation` and has
two authorizing stages: `preflight` and `pilot`.

## Immutable parent result

The parent run is:

```text
runs/experiment12_d0_jacobi_rb_boundary_tangent_prefix_schedule_confirmation/
20260803-021405_production-eager-prefix-boundary-tangent-schedule-arbfix-v2
```

Its immutable terminal evidence contains 33 registered artifacts, with
registry semantic SHA-256
`b3a8a4f7187f49877257ae18b3e94cd66f09bf0b0c9b3b03773c181ad7a01086`
and registry-file SHA-256
`4ce586988c1dd768ea90e9a2d11d003c8bcf894156dbac47f0abe3056de43631`.
The source fingerprint is
`e30e301ffa330108c986dfb80f32ac2d4d17f648b422a9b047ef5e807e826547`
and the scientific-configuration SHA-256 is
`165a716e92d74ad9f78f2439d5eb96cf27f8f5756839e34c26c9d84073a9fd0b`.

The parent profile was exact, deterministic, fully certified, and free of
forbidden events. Its only failed checks were resource checks:

- projected time `109060.84962575609` seconds (`30.2947` hours) exceeded
  `108000` seconds;
- effective rate `3091.693500986353/s` was below
  `3122.0622222222223/s`.

The eager schedule nevertheless saved a projected `19377.4507` seconds from
the fused-schedule parent. The parent forecast accelerated only the separately
measured P10 base-authorizer component. It did not claim an eager-prefix
benefit for midpoint branches or for P6/P4 cohorts. The derived adjudication
for this new workflow is therefore `base_only_projection_inconclusive`; the
parent's recorded `eager_prefix_profile_computationally_infeasible` decision
remains unchanged.

## Frozen direct measurement

The pilot directly measures the complete eager-prefix pipeline on the frozen,
previously unopened namespaces:

| Profile | Path IDs |
| --- | --- |
| cache P10 | `0xEE000`--`0xEE009` |
| cache P6 | `0xEE010`--`0xEE015` |
| stream P10 | `0xEE100`--`0xEE109` |
| stream P4 | `0xEE110`--`0xEE113` |

Preflight proves these namespaces were not consumed by the parent profile,
which used only its `0xEE200` warm-up namespace. It also binds the inherited
cohort, timing, launch-packing, initial-state, eager-profile, CUDA source, and
CUBIN records.

Each of three deterministic repeats covers four 16-step windows beginning at
outer steps `0,128,256,384`. Profile order rotates cyclically. The timed
pipeline includes exact base and eight-way midpoint transitions, matching
updates, certification and fallback, permitted float32 model-input and raw
float64 target conversion, cache-style atomic NPZ/JSON commits, and real
width-32 predictor/risk work for the streaming profiles. Eight-step shards
are atomic and exactly resumable.

For each profile the slowest complete repeat is used. The projection is

\[
T_{\rm projected}=8\left(
9\max T_{\rm cache10}+\max T_{\rm cache6}
+6\max T_{\rm stream10}+\max T_{\rm stream4}
\right).
\]

It represents exactly `224788480` base transitions, `112394240` midpoint
transitions, and `337182720` total transitions. Repeats may not be averaged,
an unfavorable repeat may not be rerun, and no timing allowance may be added.

Passing requires:

- projected time at most `108000` seconds and effective rate at least
  `3122.0622222222223/s`;
- every individual profile at least `1300/s`;
- certificate fraction exactly one;
- fallback fraction/time at most `1e-4/0.10`;
- mass error at most `2e-12`, peak memory at most 80%, and projected cache
  persistence at most 1.25 GiB;
- launch size at most 4096 and identical scientific/final-state hashes across
  repeats;
- zero approximate transitions, nonfinite values, corrections, floors,
  limiters, projections, or renormalizations.

Only `exact_boundary_tangent_eager_pipeline_feasible` authorizes integrating
this execution profile into a fresh v2 cache/training/confirmation workflow.
A failure remains scheduling evidence. The preregistered next scheduling-only
repair is a shallow all-lane proof followed by stable compaction and unchanged
deep certification of unresolved lanes.

## Production commands

Run a fresh preflight:

```powershell
$prefixParent = (
  Resolve-Path "runs/experiment12_d0_jacobi_rb_boundary_tangent_prefix_schedule_confirmation/20260803-021405_production-eager-prefix-boundary-tangent-schedule-arbfix-v2"
).Path

.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_boundary_tangent_eager_pipeline_confirmation `
  --runs-root runs/experiment12_d0_jacobi_rb_boundary_tangent_eager_pipeline_confirmation `
  --run-name production-eager-prefix-complete-pipeline `
  --device cuda `
  --stage preflight `
  --parent-prefix-run-dir $prefixParent `
  --require-gate preflight
```

Resolve the newly printed directory without using a literal timestamp
placeholder:

```powershell
$pipelineRun = (
  Get-ChildItem "runs/experiment12_d0_jacobi_rb_boundary_tangent_eager_pipeline_confirmation" -Directory |
  Where-Object Name -Like "*_production-eager-prefix-complete-pipeline" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
).FullName
```

Only after `eager_pipeline_preflight_gate.json` passes, run the pilot:

```powershell
.\.venv\Scripts\python.exe `
  -m mnist.diag_d0_jacobi_rb_boundary_tangent_eager_pipeline_confirmation `
  --device cuda `
  --stage pilot `
  --resume-run-dir $pipelineRun `
  --parent-prefix-run-dir $prefixParent `
  --require-gate pilot
```

If interrupted, rerun the same pilot command. Completed atomic shards are
verified and skipped; only an incomplete or corrupt tail is recomputed.

## Production result

The production run
`20260803-034008_production-eager-prefix-complete-pipeline` completed all
three deterministic repeats of all four profiles and passed with decision
`exact_boundary_tangent_eager_pipeline_feasible`.

The frozen slowest-repeat projection was `93,545.67684082314` seconds
(`25.984910233561983` hours), leaving `4.015089766438016` hours below the
30-hour limit. Its effective rate was `3,604.471434567184/s`, which is
`15.4516%` above the required `3,122.0622222222223/s`. The authorizing
slowest profile rates were:

- cache P10: `3,655.4593158922094/s`;
- cache P6: `4,011.184507492176/s`;
- streaming P10: `3,549.8024698245536/s`;
- streaming P4: `2,918.7820943591864/s`.

All `23,708,160` executed pilot transitions were certified. There were zero
fallbacks, forbidden events, and repeat-hash mismatches; the maximum mass
error was `4.440892098500626e-16`, and peak GPU-memory fraction was
`0.008759760158424649`. Projected persistence was `1,214,005,704` bytes,
within the unchanged 1.25-GiB limit. The terminal registry contains 615
verified artifacts with semantic SHA-256
`b85907645f1b11be581f1247268729478fb7b4ff49444181663ac90467792eb7`.

This passing result authorizes integrating the exact eager-prefix scheduler
into a fresh v2 boundary-tangent cache/training/confirmation workflow. It did
not generate the production cache, train a model, run a controller trajectory,
reconstruct an image, or sample.
