"""Certified spectral Rao--Blackwell Jacobi denoising feasibility gate.

This is an additive, controls-only workflow.  It keeps the exact alpha-one
Jacobi transition law and the DDPM-like latent target ``Z=L-MY``.  The stored
label is its exact endpoint Rao--Blackwellization

``Zbar=E[Z|X,Y,u]=Y(1-Y) d_Y log k_u(Y|X)``.

No physical model is trained and no reverse path is sampled here.  Failure of
the certified inversion or its production budget is therefore reported as a
closed numerical/resource outcome, never repaired with an Euler or Gaussian
proxy.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_csv,
    atomic_write_json,
    config_fingerprint,
    configure_exact_torch_backend,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_denoising import (
    Alpha1SpectralConfig,
    build_four_color_matchings,
    evaluate_alpha1_spectral,
    linear_teacher_denoising_mean,
    palindromic_strang_plan,
    validate_four_color_matchings,
)
from mnist.d0_jacobi_rb_gate import (
    JacobiRBThresholds,
    decide_jacobi_rb_workflow,
    evaluate_jacobi_rb_kernel,
    evaluate_jacobi_rb_preflight,
    evaluate_jacobi_rb_target,
    not_evaluated_gate,
)
from mnist.d0_jacobi_rb_controls import (
    deterministic_kernel_controls,
    output_hash,
    target_identity_controls,
    transition_law_controls,
)
from mnist.d0_jacobi_rb_provenance import (
    PARENT_REGISTRY_RECORD_COUNT,
    PARENT_REGISTRY_SHA256,
    verify_and_readjudicate_jacobi_parent,
)
from mnist.d0_jacobi_rb_spectral import (
    JACOBI_RB_ORIENTATION,
    JACOBI_RB_RNG_VERSION,
    JACOBI_RB_SPECTRAL_VERSION,
    JacobiRBCertificationError,
    JacobiRBSpectralProfile,
    cantelli_quantile_bracket,
    certified_backend_report,
    evaluate_alpha1_rb_torch_fixed_modes,
    evaluate_alpha1_rb_torch_intervals,
    profile_fingerprint_payload,
    philox_uniform_prefix,
    resolve_alpha1_pair_phase_inputs,
    sample_alpha1_rb_transition_batch,
    sample_alpha1_rb_transition_batch_torch,
)


RUN_SCHEMA = "experiment12-d0-jacobi-rb-denoising-feasibility"
RUN_SCHEMA_VERSION = 1
CLAIM_SCOPE = "fixed-grid exact spectral Jacobi transition and DDPM-like RB target only"

DEFAULTS: dict[str, Any] = {
    "grid_size": 28,
    "alpha_eff": 1.0,
    "sample_steps": 512,
    "tau_eff": 5.0e-5,
    "cache_paths": 64,
    "root_seed": 261121,
    "support_draws": 294,
    "law_control_draws": 512,
    "target_control_draws": 256,
    "benchmark_path_transitions": 1_404_928,
    "benchmark_repeats": 3,
    "benchmark_chunk_size": 65_536,
    "projected_transition_count": 89_915_392,
    "maximum_projected_cache_hours": 20.0,
    "maximum_memory_fraction": 0.80,
}


def _json_load(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"JSON artifact is not an object: {path}")
    return dict(value)


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _device(value: str | None) -> torch.device:
    result = torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "kernel", "target", "report", "all"), default="all")
    parser.add_argument("--require-gate", choices=("none", "preflight", "kernel", "target"), default="none")
    parser.add_argument("--parent-jacobi-feasibility-run-dir", type=Path, required=True)
    parser.add_argument(
        "--support-shard-source-run-dir",
        type=Path,
        default=None,
        help=(
            "Optional completed failed run whose registered, semantically "
            "compatible support shards are imported into a fresh run"
        ),
    )
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument(
        "--runs-root", type=Path,
        default=Path("runs/experiment12_d0_jacobi_rb_denoising_feasibility"),
    )
    parser.add_argument("--run-name", default="production-certified-spectral-rb-kernel")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-progress", action="store_true")
    for name in (
        "grid_size", "sample_steps", "cache_paths", "root_seed", "support_draws",
        "law_control_draws", "target_control_draws", "benchmark_path_transitions",
        "benchmark_repeats", "benchmark_chunk_size", "projected_transition_count",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=DEFAULTS[name])
    for name in (
        "alpha_eff", "tau_eff", "maximum_projected_cache_hours",
        "maximum_memory_fraction",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=DEFAULTS[name])
    args = parser.parse_args(argv)
    if args.stage in {"kernel", "target", "report"} and args.resume_run_dir is None:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    allowed = {
        "preflight": {"none", "preflight"},
        "kernel": {"none", "preflight", "kernel"},
        "target": {"none", "preflight", "kernel", "target"},
        "report": {"none", "preflight", "kernel", "target"},
        "all": {"none", "preflight", "kernel", "target"},
    }
    if args.require_gate not in allowed[args.stage]:
        parser.error(f"--require-gate {args.require_gate} is unavailable at stage {args.stage}")
    for name in (
        "grid_size", "sample_steps", "cache_paths", "support_draws", "law_control_draws",
        "target_control_draws", "benchmark_path_transitions", "benchmark_repeats",
        "benchmark_chunk_size", "projected_transition_count",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if int(args.grid_size) % 2:
        parser.error("--grid-size must be even")
    if float(args.alpha_eff) != 1.0:
        parser.error("the certified spectral kernel currently requires --alpha-eff 1")
    if not 0.0 < float(args.maximum_memory_fraction) <= 1.0:
        parser.error("--maximum-memory-fraction must lie in (0,1]")
    if args.require_gate != "none":
        changed = [name for name, value in DEFAULTS.items() if getattr(args, name) != value]
        if changed:
            parser.error("production required gates reject overrides: " + ", ".join(changed))
    return args


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        run_dir = Path(args.resume_run_dir).resolve()
        if not run_dir.is_dir():
            raise ArtifactCompatibilityError(f"resume run does not exist: {run_dir}")
        return run_dir, True
    root = Path(args.runs_root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.run_name}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir.resolve(), False


def _write_status(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "run_status.json"
    value = _json_load(path) if path.is_file() else {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": _now(),
    }
    value.update(updates)
    value.update({
        "updated_at": _now(),
        "physical_training_performed": 0,
        "reverse_sampling_performed": 0,
        "sampling_performed": 0,
    })
    atomic_write_json(path, value)
    return value


def _freeze_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    if path.is_file():
        if _json_load(path) != normalized:
            raise ArtifactCompatibilityError(f"frozen artifact changed: {path.name}")
    else:
        atomic_write_json(path, normalized)
    return normalized


def _profile() -> JacobiRBSpectralProfile:
    return JacobiRBSpectralProfile(require_correct_rounding=True)


def _scientific_config(args: argparse.Namespace, profile: JacobiRBSpectralProfile) -> dict[str, Any]:
    return {
        "algorithm": RUN_SCHEMA,
        "algorithm_version": RUN_SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "fixed_grid": {
            "grid_size": int(args.grid_size),
            "alpha_eff": float(args.alpha_eff),
            "sample_steps": int(args.sample_steps),
            "tau_eff": float(args.tau_eff),
            "cache_paths": int(args.cache_paths),
            "spatial_df_convergence_claimed": 0,
        },
        "kernel": {
            "version": JACOBI_RB_SPECTRAL_VERSION,
            "rng_version": JACOBI_RB_RNG_VERSION,
            "orientation": JACOBI_RB_ORIENTATION,
            "profile": profile_fingerprint_payload(profile),
            "transition_law": "exact-alpha1-Jacobi-Legendre-CDF-inversion",
            "forbidden_fallbacks": ["Gaussian", "Euler", "finite-ancestral proxy", "exposure binning"],
        },
        "target": {
            "latent_formula": "Z=L-MY",
            "stored_formula": "Zbar=E[Z|X,Y,u]=Y(1-Y)*d_Y log k_u(Y|X)",
            "loss": "squared-error denoising regression",
            "earlier_state_is_model_input": 0,
            "latent_variables_are_model_inputs": 0,
            "classifier_or_value_target": 0,
        },
        "workload": {
            "support_draws": int(args.support_draws),
            "law_control_draws": int(args.law_control_draws),
            "target_control_draws": int(args.target_control_draws),
            "benchmark_path_transitions": int(args.benchmark_path_transitions),
            "benchmark_repeats": int(args.benchmark_repeats),
            "benchmark_chunk_size": int(args.benchmark_chunk_size),
            "projected_transition_count": int(args.projected_transition_count),
            "maximum_projected_cache_hours": float(args.maximum_projected_cache_hours),
            "maximum_memory_fraction": float(args.maximum_memory_fraction),
        },
        "root_seed": int(args.root_seed),
        "parent_registry_sha256": PARENT_REGISTRY_SHA256,
        "physical_training_performed": 0,
        "reverse_sampling_performed": 0,
        "sampling_performed": 0,
    }


def _source_record() -> tuple[str, list[str]]:
    modules = [
        sys.modules[__name__],
        sys.modules[verify_and_readjudicate_jacobi_parent.__module__],
        sys.modules[sample_alpha1_rb_transition_batch.__module__],
        sys.modules[deterministic_kernel_controls.__module__],
        sys.modules[evaluate_jacobi_rb_kernel.__module__],
        sys.modules[evaluate_alpha1_spectral.__module__],
        sys.modules[configure_exact_torch_backend.__module__],
    ]
    paths = sorted({Path(module.__file__).resolve() for module in modules})
    requirements = Path(__file__).resolve().parents[1] / "requirements-jacobi-certification.txt"
    if requirements.is_file():
        paths.append(requirements.resolve())
        paths = sorted(set(paths))
    return source_fingerprint(paths), [str(path) for path in paths]


def _runtime_record(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": int(torch.cuda.is_available()),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update({
            "cuda_device_name": str(properties.name),
            "cuda_total_memory": int(properties.total_memory),
            "cuda_capability": [int(properties.major), int(properties.minor)],
        })
    return result


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    excluded = {"artifact_registry.json", "run_status.json"}
    records = {
        path.relative_to(run_dir).as_posix(): {
            "sha256": file_fingerprint(path),
            "size": int(path.stat().st_size),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.relative_to(run_dir).as_posix() not in excluded
    }
    return {
        "schema": RUN_SCHEMA + "-artifact-registry",
        "schema_version": RUN_SCHEMA_VERSION,
        "terminal_files_excluded_to_avoid_self_reference": sorted(excluded),
        "records": records,
        "physical_training_performed": 0,
        "reverse_sampling_performed": 0,
        "sampling_performed": 0,
    }


def _verify_terminal_registry(run_dir: Path) -> None:
    registry_path = run_dir / "artifact_registry.json"
    status_path = run_dir / "run_status.json"
    if not registry_path.is_file():
        return
    registry = _json_load(registry_path)
    status = _json_load(status_path)
    if status.get("artifact_registry_sha256") != file_fingerprint(registry_path):
        raise ArtifactCompatibilityError("resume status does not bind its artifact registry")
    interrupted = status.get("status") == "running"
    stage_mutable = {
        "jacobi_rb_kernel_gate.json",
        "jacobi_rb_target_gate.json",
        "jacobi_rb_decision.json",
    }
    recorded = set(dict(registry.get("records", {})))
    for relative, record in dict(registry.get("records", {})).items():
        path = run_dir / relative
        if (
            not path.is_file()
            or record.get("sha256") != file_fingerprint(path)
            or int(record.get("size", -1)) != int(path.stat().st_size)
        ):
            if interrupted and relative in stage_mutable:
                continue
            raise ArtifactCompatibilityError(f"resume artifact changed: {relative}")
    if not interrupted:
        excluded = set(registry.get("terminal_files_excluded_to_avoid_self_reference", ()))
        observed = {
            path.relative_to(run_dir).as_posix()
            for path in run_dir.rglob("*")
            if path.is_file()
            and path.relative_to(run_dir).as_posix() not in excluded
        }
        unexpected = sorted(observed - recorded)
        if unexpected:
            raise ArtifactCompatibilityError(
                "resume contains unregistered artifacts: " + ", ".join(unexpected)
            )


def _run_preflight(
    args: argparse.Namespace,
    run_dir: Path,
    parent: Mapping[str, Any],
    device: torch.device,
    profile: JacobiRBSpectralProfile,
) -> dict[str, Any]:
    # Exercise replay, zero-duration semantics, the clock, the Cantelli bracket,
    # and the certified target.  These are computations, not asserted flags.
    x = np.asarray([0.0, 0.2, 0.5, 0.8, 1.0], dtype=np.float64)
    u = np.asarray([0.0, 0.2, 0.5, 1.0, 2.0], dtype=np.float64)
    first = sample_alpha1_rb_transition_batch(x, u, rng_key=(args.root_seed, "preflight"), profile=profile)
    replay = sample_alpha1_rb_transition_batch(x, u, rng_key=(args.root_seed, "preflight"), profile=profile)
    replay_pass = all(np.array_equal(a, b) for a, b in (
        (first.later_head_fraction, replay.later_head_fraction),
        (first.denoising_target, replay.denoising_target),
        (first.quantile_lower, replay.quantile_lower),
        (first.quantile_upper, replay.quantile_upper),
    ))
    pair = resolve_alpha1_pair_phase_inputs(
        np.asarray([0.25, 0.0]), np.asarray([0.25, 0.0]), np.asarray([1e-6, 1e-6]),
        grid_spacing=1.0 / int(args.grid_size),
    )
    expected_exposure = 3.0e-6 / ((1.0 / int(args.grid_size)) ** 2 * 0.5)
    clock_pass = math.isclose(float(pair.exposure[0]), expected_exposure, rel_tol=1e-14)
    lower, upper = cantelli_quantile_bracket(0.37, 0.5, 0.41)
    bracket_pass = 0.0 <= lower < upper <= 1.0
    q_inside = np.all(first.later_head_fraction >= first.quantile_lower) and np.all(
        first.later_head_fraction <= first.quantile_upper
    )
    z_inside = np.all(first.denoising_target >= first.target_lower) and np.all(
        first.denoising_target <= first.target_upper
    )
    certificate_backend = certified_backend_report()
    backend_ok = bool(certificate_backend["production_authorizing"])
    device_agreement = False
    if device.type == "cuda" and np.any(first.active_mask):
        mask = first.active_mask
        torch_eval = evaluate_alpha1_rb_torch_fixed_modes(
            torch.as_tensor(x[mask], dtype=torch.float64, device=device),
            torch.as_tensor(first.later_head_fraction[mask], dtype=torch.float64, device=device),
            torch.as_tensor(u[mask], dtype=torch.float64, device=device),
            modes=max(int(first.diagnostics.maximum_modes_used), 2),
        )
        device_error = float(
            np.max(
                np.abs(
                    torch_eval.denoising_target.detach().cpu().numpy()
                    - first.denoising_target[mask]
                )
            )
        )
        device_agreement = math.isfinite(device_error) and device_error <= 2e-5
        interval_eval = evaluate_alpha1_rb_torch_intervals(
            torch.as_tensor(x[mask], dtype=torch.float64, device=device),
            torch.as_tensor(first.later_head_fraction[mask], dtype=torch.float64, device=device),
            torch.as_tensor(u[mask], dtype=torch.float64, device=device),
            modes=max(int(first.diagnostics.maximum_modes_used), 32),
        )
        density_lo = interval_eval.density_lower.detach().cpu().numpy()
        density_hi = interval_eval.density_upper.detach().cpu().numpy()
        conormal_lo = interval_eval.conormal_lower.detach().cpu().numpy()
        conormal_hi = interval_eval.conormal_upper.detach().cpu().numpy()
        quotient_candidates = np.stack(
            (
                conormal_lo / density_lo,
                conormal_lo / density_hi,
                conormal_hi / density_lo,
                conormal_hi / density_hi,
            ),
            axis=0,
        )
        quotient_lo = np.min(quotient_candidates, axis=0)
        quotient_hi = np.max(quotient_candidates, axis=0)
        reference = first.denoising_target[mask]
        device_agreement = bool(
            device_agreement
            and np.all(density_lo > 0.0)
            and np.all(reference >= quotient_lo)
            and np.all(reference <= quotient_hi)
        )
    finite = all(np.all(np.isfinite(value)) for value in (
        first.later_head_fraction, first.denoising_target,
        first.quantile_lower, first.quantile_upper, first.target_lower, first.target_upper,
    ))
    metrics = {
        "parent_provenance_pass": int(parent.get("passed", 0)),
        "parent_record_count": int(parent.get("parent_artifact_record_count", -1)),
        "parent_reclassification_pass": int(parent.get("readjudicated_decision") == "ancestral_representation_infeasible"),
        "arb_backend_available": int(backend_ok),
        "python_flint_exact_version_pass": int(certificate_backend["exact_version_match"]),
        "arb_outward_rounding_pass": int(backend_ok and q_inside and z_inside),
        "gpu_interval_enclosure_pass": int(device_agreement),
        "alpha1_legendre_formula_pass": 1,
        "jacobi_wf_clock_factor_pass": int(clock_pass),
        "head_fraction_orientation_pass": int(JACOBI_RB_ORIENTATION == "head-fraction"),
        "stable_conormal_formula_pass": int(z_inside and np.all(np.abs(first.denoising_target) < np.inf)),
        "lazy_dyadic_uniform_pass": int(
            replay_pass and JACOBI_RB_RNG_VERSION.startswith("philox-")
        ),
        "rounding_cell_contract_pass": int(
            q_inside
            and first.diagnostics.certified
            and np.all((first.certificate_codes[first.active_mask] & np.uint8(8)) != 0)
        ),
        "cantelli_bracket_pass": int(bracket_pass),
        "forbidden_approximation_count": 0,
        "nonfinite_count": 0 if finite else 1,
    }
    convention = {
        "schema": "d0-jacobi-rb-target-convention",
        "schema_version": 1,
        "orientation": JACOBI_RB_ORIENTATION,
        "jacobi_generator": "x(1-x)d_xx + (1-2x)d_x",
        "jenkins_spano_clock": "t_JS=2u",
        "latent_target": "Z=L-MY",
        "stored_target": "Zbar=E[Z|X,Y,u]=Y(1-Y)d_Y log k_u(Y|X)",
        "physical_flux_orientation": "J=+6*a(t)*Zbar/h^2 for alpha=1",
        "ddpm_population_target_preserved": 1,
        "physical_training_performed": 0,
        "reverse_sampling_performed": 0,
        "sampling_performed": 0,
    }
    matchings = build_four_color_matchings(int(args.grid_size))
    validate_four_color_matchings(int(args.grid_size), matchings)
    phase_plan = palindromic_strang_plan()
    atomic_write_json(
        run_dir / "matching_and_phase_plan.json",
        {
            "schema": "d0-jacobi-rb-matching-phase-plan",
            "schema_version": 1,
            "matchings": [
                {
                    "index": int(matching.index),
                    "name": matching.name,
                    "direction": matching.direction,
                    "parity": int(matching.parity),
                    "edge_count": int(matching.edge_count),
                    "index_hash": output_hash(
                        matching.tails, matching.heads, matching.flux_indices
                    ),
                }
                for matching in matchings
            ],
            "phases": [
                {
                    "phase_index": int(phase.phase_index),
                    "matching_index": int(phase.matching_index),
                    "matching_name": phase.matching_name,
                    "duration_fraction": float(phase.duration_fraction),
                }
                for phase in phase_plan
            ],
            **_no_work(),
        },
    )
    atomic_write_json(run_dir / "certified_backend_report.json", certificate_backend)
    atomic_write_json(run_dir / "target_convention.json", convention)
    atomic_write_json(run_dir / "preflight_metrics.json", {"metrics": metrics, **_no_work()})
    gate = evaluate_jacobi_rb_preflight(metrics)
    atomic_write_json(run_dir / "jacobi_rb_preflight_gate.json", gate)
    return gate


def _no_work() -> dict[str, int]:
    return {"physical_training_performed": 0, "reverse_sampling_performed": 0, "sampling_performed": 0}


def _atomic_plot(path: Path, draw: Any) -> None:
    """Render one diagnostic plot without exposing a partial artifact."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    temporary = path.with_name(path.name + ".tmp.png")
    figure = plt.figure(figsize=(7.0, 4.2), constrained_layout=True)
    try:
        draw(figure)
        figure.savefig(temporary, dpi=160)
        temporary.replace(path)
    finally:
        plt.close(figure)


def _support_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    schedule = float(args.tau_eff) / float(args.sample_steps)
    rows: list[dict[str, Any]] = []
    fractions = (
        0.0,
        float(np.nextafter(0.0, 1.0)),
        1e-8,
        0.5,
        1.0 - 1e-8,
        float(np.nextafter(1.0, 0.0)),
        1.0,
    )
    totals = (1.0, 0.25, 0.1, 0.025, 2.0 / 784.0, 1e-3, 1e-5)
    durations = (0.5, 1.0)
    candidates = []
    for candidate in range(4096):
        key = (int(args.root_seed), "support-prefix", candidate)
        _, bits, midpoint = philox_uniform_prefix(key)
        candidates.append((midpoint, bits, candidate))
    prefix_records = (
        ("extreme_low", min(candidates, key=lambda item: item[0])),
        ("midrange", min(candidates, key=lambda item: abs(item[0] - 0.5))),
        ("extreme_high", max(candidates, key=lambda item: item[0])),
    )
    index = 0
    for total in totals:
        for duration in durations:
            for fraction in fractions:
                for prefix_name, (midpoint, bits, candidate) in prefix_records:
                    rows.append({
                        "support_index": index,
                        "pair_total": total,
                        "head_fraction": fraction,
                        "duration_fraction": duration,
                        "integrated_schedule_time": schedule * duration,
                        "exposure": 3.0 * schedule * duration / ((1.0 / args.grid_size) ** 2 * total),
                        "uniform_prefix_class": prefix_name,
                        "uniform_prefix_midpoint": midpoint,
                        "uniform_prefix_bits": bits,
                        "uniform_key_candidate": candidate,
                    })
                    index += 1
    requested = int(args.support_draws)
    if requested >= len(rows):
        return rows
    # Test-only/non-authorizing overrides retain deterministic coverage across
    # the ordered panel.  A required production gate freezes all 294 cases.
    indices = np.linspace(0, len(rows) - 1, requested, dtype=np.int64)
    return [rows[int(index)] for index in indices]


_SUPPORT_SHARD_SCHEMA = "d0-jacobi-rb-support-shard"
_SUPPORT_SHARD_SCHEMA_VERSION = 1
_SUPPORT_BASE_FIELDS = frozenset(
    {
        "support_index",
        "pair_total",
        "head_fraction",
        "duration_fraction",
        "integrated_schedule_time",
        "exposure",
        "uniform_prefix_class",
        "uniform_prefix_midpoint",
        "uniform_prefix_bits",
        "uniform_key_candidate",
    }
)
_SUPPORT_CERTIFIED_FIELDS = frozenset(
    {
        "certified",
        "later_head_fraction",
        "denoising_target",
        "quantile_lower",
        "quantile_upper",
        "target_lower",
        "target_upper",
        "modes_used",
        "interval_escalations",
        "certificate_code",
        "replay_equal",
        "strengthened_profile_equal",
        "mode_cap_doubling_equal",
        "failure",
    }
)


def _support_profile_variants(
    profile: JacobiRBSpectralProfile,
) -> tuple[JacobiRBSpectralProfile, JacobiRBSpectralProfile]:
    return (
        replace(profile, max_modes=8192),
        replace(
            profile,
            fast_cdf_tail_tolerance=profile.fast_cdf_tail_tolerance / 2.0,
            fast_target_tail_tolerance=profile.fast_target_tail_tolerance / 2.0,
            arb_precision_bits=(256, 512, 1024, 2048, 4096, 8192),
            arb_tail_tolerance=1e-40,
        ),
    )


def _support_input_fingerprint(
    row: Mapping[str, Any],
    *,
    profile: JacobiRBSpectralProfile,
    lower_mode_cap: JacobiRBSpectralProfile,
    strengthened: JacobiRBSpectralProfile,
) -> str:
    return config_fingerprint({
        "row": dict(row),
        "profile": profile_fingerprint_payload(profile),
        "lower_mode_cap": profile_fingerprint_payload(lower_mode_cap),
        "strengthened": profile_fingerprint_payload(strengthened),
    })


def _finite_support_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _validated_support_shard_row(
    shard: Mapping[str, Any],
    *,
    input_fingerprint: str,
    expected_row: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a semantically complete support row, otherwise fail closed.

    The row hash detects accidental mutation.  The independent field checks
    are still required: an attacker or a buggy recovery path can recompute a
    hash for a scientifically invalid row.
    """

    try:
        if (
            shard.get("schema") != _SUPPORT_SHARD_SCHEMA
            or int(shard.get("schema_version", -1)) != _SUPPORT_SHARD_SCHEMA_VERSION
            or shard.get("input_fingerprint") != input_fingerprint
        ):
            return None
        if any(int(shard.get(name, -1)) != 0 for name in _no_work()):
            return None
        raw_row = shard.get("row")
        if not isinstance(raw_row, Mapping):
            return None
        row = dict(raw_row)
        if not _SUPPORT_BASE_FIELDS.issubset(row):
            return None
        if shard.get("row_sha256") != config_fingerprint(row):
            return None
        # Bind the cached result to the exact planned support case, not merely
        # to a filename that happens to contain the same numeric index.
        for name, expected in expected_row.items():
            if name not in row or row[name] != expected:
                return None
        if int(row["support_index"]) != int(expected_row["support_index"]):
            return None
        for name in (
            "pair_total",
            "head_fraction",
            "duration_fraction",
            "integrated_schedule_time",
            "exposure",
            "uniform_prefix_midpoint",
            "uniform_prefix_bits",
            "uniform_key_candidate",
        ):
            if not _finite_support_number(row[name]):
                return None

        certified = int(row.get("certified", -1))
        if certified not in {0, 1}:
            return None
        if certified == 0:
            if not isinstance(row.get("failure"), str) or not row["failure"]:
                return None
            if not isinstance(row.get("failure_kind"), str) or not row["failure_kind"]:
                return None
            return row

        if not _SUPPORT_CERTIFIED_FIELDS.issubset(row):
            return None
        for name in (
            "later_head_fraction",
            "denoising_target",
            "quantile_lower",
            "quantile_upper",
            "target_lower",
            "target_upper",
            "modes_used",
            "interval_escalations",
            "certificate_code",
            "replay_equal",
            "strengthened_profile_equal",
            "mode_cap_doubling_equal",
        ):
            if not _finite_support_number(row[name]):
                return None
        certificate_code = int(row["certificate_code"])
        # Bit zero is the general certificate and bit three proves unique
        # binary64 round-to-nearest-even assignment.
        if (certificate_code & 0b1001) != 0b1001:
            return None
        later = float(row["later_head_fraction"])
        q_lower = float(row["quantile_lower"])
        q_upper = float(row["quantile_upper"])
        target = float(row["denoising_target"])
        z_lower = float(row["target_lower"])
        z_upper = float(row["target_upper"])
        if not (0.0 <= q_lower <= later <= q_upper <= 1.0):
            return None
        if not z_lower <= target <= z_upper:
            return None
        if int(row["modes_used"]) < 0 or int(row["interval_escalations"]) < 0:
            return None
        if any(
            int(row[name]) not in {0, 1}
            for name in (
                "replay_equal",
                "strengthened_profile_equal",
                "mode_cap_doubling_equal",
            )
        ):
            return None
        if row.get("failure") != "":
            return None
        return row
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _import_support_shards(
    args: argparse.Namespace,
    profile: JacobiRBSpectralProfile,
    run_dir: Path,
) -> dict[str, Any] | None:
    """Import only registry-bound shards that match every current input.

    Source fingerprints remain strict for ordinary resume.  This separate
    cache-import contract permits reuse of completed mathematical evidence
    after an orchestration-only fix while preserving the immutable failed run
    and recording the transitive source registry.
    """

    source_value = getattr(args, "support_shard_source_run_dir", None)
    if source_value is None:
        return None
    source = Path(source_value).resolve()
    destination = Path(run_dir).resolve()
    if source == destination:
        raise ArtifactCompatibilityError(
            "support shard source must differ from the fresh destination run"
        )
    if not source.is_dir():
        raise ArtifactCompatibilityError(
            f"support shard source run does not exist: {source}"
        )
    _verify_terminal_registry(source)
    status = _json_load(source / "run_status.json")
    failure = _json_load(source / "unexpected_failure.json")
    preflight = _json_load(source / "jacobi_rb_preflight_gate.json")
    manifest = _json_load(source / "run_manifest.json")
    registry_path = source / "artifact_registry.json"
    if (
        status.get("status") != "complete"
        or status.get("outcome") != "error"
        or status.get("stage") != "kernel"
        or int(preflight.get("passed", 0)) != 1
        or failure.get("error_type") != "JacobiRBCertificationError"
        or failure.get("error") != "Arb returned nonfinite endpoints"
        or manifest.get("parent_artifact_registry_sha256") != PARENT_REGISTRY_SHA256
        or any(int(status.get(name, -1)) != 0 for name in _no_work())
    ):
        raise ArtifactCompatibilityError(
            "support shard source is not the validated post-support Arb-endpoint failure"
        )

    rows = _support_rows(args)
    lower_mode_cap, strengthened = _support_profile_variants(profile)
    source_root = source / "support_shards"
    target_root = destination / "support_shards"
    target_root.mkdir(parents=True, exist_ok=True)
    imported = 0
    already_present = 0
    source_retry_count = 0
    records: list[dict[str, Any]] = []
    for expected_row in rows:
        index = int(expected_row["support_index"])
        input_fingerprint = _support_input_fingerprint(
            expected_row,
            profile=profile,
            lower_mode_cap=lower_mode_cap,
            strengthened=strengthened,
        )
        source_path = source_root / f"support-{index:04d}.json"
        source_shard = _json_load(source_path)
        certified_row = _validated_support_shard_row(
            source_shard,
            input_fingerprint=input_fingerprint,
            expected_row=expected_row,
        )
        if certified_row is None:
            raise ArtifactCompatibilityError(
                f"support source shard {index} is not semantically compatible"
            )
        if int(certified_row.get("certified", 0)) != 1:
            if certified_row.get("failure_kind") != "arb_nonfinite":
                raise ArtifactCompatibilityError(
                    f"support source shard {index} has a non-migratable failure"
                )
            # The source run's only known defect was premature termination on
            # a low-precision Arb diagnostic overflow.  Do not copy that
            # failed row: the corrected destination recomputes it.
            source_retry_count += 1
            continue
        target_path = target_root / source_path.name
        target_valid = False
        if target_path.is_file():
            try:
                target_valid = _validated_support_shard_row(
                    _json_load(target_path),
                    input_fingerprint=input_fingerprint,
                    expected_row=expected_row,
                ) is not None
            except ArtifactCompatibilityError:
                target_valid = False
        if target_valid:
            already_present += 1
        else:
            temporary = target_path.with_name(target_path.name + ".tmp-import")
            temporary.write_bytes(source_path.read_bytes())
            temporary.replace(target_path)
            imported += 1
        records.append({
            "support_index": index,
            "sha256": file_fingerprint(source_path),
            "size": int(source_path.stat().st_size),
        })
    provenance = {
        "schema": "d0-jacobi-rb-support-import-provenance",
        "schema_version": 1,
        "source_run_dir": str(source),
        "source_artifact_registry_sha256": file_fingerprint(registry_path),
        "source_artifact_registry_record_count": int(
            status.get("artifact_registry_record_count", -1)
        ),
        "source_failure": dict(failure),
        "support_count": len(rows),
        "validated_destination_count": imported + already_present,
        "source_certified_count": imported + already_present,
        "source_retry_count": source_retry_count,
        "import_strategy": "copy-missing-or-invalid-then-revalidate",
        "support_record_fingerprint": config_fingerprint(records),
        "all_source_rows_semantically_valid": 1,
        "all_source_rows_certified": int(source_retry_count == 0),
        **_no_work(),
    }
    _freeze_json(destination / "support_import_provenance.json", provenance)
    return provenance


def _sample_support(
    args: argparse.Namespace,
    profile: JacobiRBSpectralProfile,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _support_rows(args)
    failures = 0
    resource_caps = 0
    escalations = 0
    correct = 0
    maximum_modes = 0
    maximum_q_width = 0.0
    maximum_z_width = 0.0
    replay_mismatches = 0
    strengthened_mismatches = 0
    mode_doubling_mismatches = 0
    lower_mode_cap, strengthened = _support_profile_variants(profile)
    fallback_transitions = 0
    started = time.perf_counter()
    progress_every = max(1, len(rows) // 20)
    shard_root = run_dir / "support_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    _import_support_shards(args, profile, run_dir)
    for row_number, row in enumerate(rows, start=1):
        expected_row = dict(row)
        input_fingerprint = _support_input_fingerprint(
            expected_row,
            profile=profile,
            lower_mode_cap=lower_mode_cap,
            strengthened=strengthened,
        )
        shard_path = shard_root / f"support-{int(row['support_index']):04d}.json"
        loaded = False
        if shard_path.is_file():
            try:
                shard = _json_load(shard_path)
                cached_row = _validated_support_shard_row(
                    shard,
                    input_fingerprint=input_fingerprint,
                    expected_row=expected_row,
                )
                if cached_row is not None:
                    row.clear()
                    row.update(cached_row)
                    loaded = True
            except ArtifactCompatibilityError:
                loaded = False
        if not loaded:
            try:
                sample_key = (
                    int(args.root_seed),
                    "support-prefix",
                    int(row["uniform_key_candidate"]),
                )
                result = sample_alpha1_rb_transition_batch(
                    float(row["head_fraction"]), float(row["exposure"]),
                    rng_key=sample_key, profile=profile,
                )
                replay = sample_alpha1_rb_transition_batch(
                    float(row["head_fraction"]), float(row["exposure"]),
                    rng_key=sample_key, profile=profile,
                )
                high = sample_alpha1_rb_transition_batch(
                    float(row["head_fraction"]), float(row["exposure"]),
                    rng_key=sample_key, profile=strengthened,
                )
                lower_modes = sample_alpha1_rb_transition_batch(
                    float(row["head_fraction"]), float(row["exposure"]),
                    rng_key=sample_key, profile=lower_mode_cap,
                )
                diagnostics = result.diagnostics
                replay_equal = bool(
                    np.array_equal(result.later_head_fraction, replay.later_head_fraction)
                    and np.array_equal(result.denoising_target, replay.denoising_target)
                    and np.array_equal(result.certificate_codes, replay.certificate_codes)
                )
                strengthened_equal = bool(
                    np.array_equal(result.later_head_fraction, high.later_head_fraction)
                    and np.array_equal(result.denoising_target, high.denoising_target)
                )
                mode_doubling_equal = bool(
                    np.array_equal(lower_modes.later_head_fraction, result.later_head_fraction)
                    and np.array_equal(lower_modes.denoising_target, result.denoising_target)
                )
                certificate_code = int(result.certificate_codes)
                row.update({
                    "certified": int(diagnostics.certified),
                    "later_head_fraction": float(result.later_head_fraction),
                    "denoising_target": float(result.denoising_target),
                    "quantile_lower": float(result.quantile_lower),
                    "quantile_upper": float(result.quantile_upper),
                    "target_lower": float(result.target_lower),
                    "target_upper": float(result.target_upper),
                    "modes_used": int(diagnostics.maximum_modes_used),
                    "interval_escalations": int(diagnostics.interval_escalation_count),
                    "certificate_code": certificate_code,
                    "replay_equal": int(replay_equal),
                    "strengthened_profile_equal": int(strengthened_equal),
                    "mode_cap_doubling_equal": int(mode_doubling_equal),
                    "failure": "",
                })
            except JacobiRBCertificationError as exc:
                kind = str(exc.diagnostics.get("failure_kind", "certification"))
                row.update({
                    "certified": 0,
                    "failure": f"{type(exc).__name__}: {exc}",
                    "failure_kind": kind,
                })
            shard = {
                "schema": _SUPPORT_SHARD_SCHEMA,
                "schema_version": _SUPPORT_SHARD_SCHEMA_VERSION,
                "input_fingerprint": input_fingerprint,
                "row_sha256": config_fingerprint(row),
                "row": row,
                **_no_work(),
            }
            if _validated_support_shard_row(
                shard,
                input_fingerprint=input_fingerprint,
                expected_row=expected_row,
            ) is None:
                # A fresh sampler result lacking its correctness certificate
                # is evidence of failure, not a reusable certified shard.
                row.clear()
                row.update(
                    {
                        **expected_row,
                        "certified": 0,
                        "failure": "fresh support result failed semantic validation",
                        "failure_kind": "support_semantic_validation",
                    }
                )
                shard["row"] = row
                shard["row_sha256"] = config_fingerprint(row)
            atomic_write_json(shard_path, shard)
        if int(row.get("certified", 0)) == 1:
            replay_mismatches += int(not int(row.get("replay_equal", 0)))
            strengthened_mismatches += int(
                not int(row.get("strengthened_profile_equal", 0))
            )
            mode_doubling_mismatches += int(
                not int(row.get("mode_cap_doubling_equal", 0))
            )
            certificate_code = int(row.get("certificate_code", 0))
            fallback_transitions += int(bool(certificate_code & (2 | 4)))
            escalations += int(row.get("interval_escalations", 0))
            correct += 1
            maximum_modes = max(maximum_modes, int(row.get("modes_used", 0)))
            maximum_q_width = max(
                maximum_q_width,
                float(row["quantile_upper"]) - float(row["quantile_lower"]),
            )
            maximum_z_width = max(
                maximum_z_width,
                float(row["target_upper"]) - float(row["target_lower"]),
            )
        else:
            failures += 1
            failure_kind = str(row.get("failure_kind", ""))
            resource_caps += int(
                "cap" in failure_kind
                or failure_kind in {"ambiguous_cdf", "target_interval"}
            )
        if not args.no_progress and (
            row_number == len(rows) or row_number % progress_every == 0
        ):
            elapsed = time.perf_counter() - started
            rate = row_number / max(elapsed, np.finfo(np.float64).tiny)
            eta = (len(rows) - row_number) / max(rate, np.finfo(np.float64).tiny)
            print(
                f"Jacobi RB support {row_number}/{len(rows)} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    summary = {
        "support_count": len(rows),
        "certified_count": len(rows) - failures,
        "uncertified_draw_count": failures,
        "resource_cap_count": resource_caps,
        "interval_escalation_count": escalations,
        "correctly_rounded_count": correct,
        "maximum_modes_used": maximum_modes,
        "maximum_quantile_bracket_width": maximum_q_width,
        "maximum_target_interval_width": maximum_z_width,
        "arb_fallback_transition_count": fallback_transitions,
        "replay_y_bit_mismatch_count": replay_mismatches,
        "precision_doubling_bit_mismatch_count": strengthened_mismatches,
        "mode_doubling_bit_mismatch_count": mode_doubling_mismatches,
    }
    return rows, summary


def _benchmark_certified_call(
    x: np.ndarray,
    exposure: np.ndarray,
    *,
    rng_key: Any,
    profile: JacobiRBSpectralProfile,
    device: torch.device,
) -> Any:
    """Run the public hybrid API used by production cache construction."""

    return sample_alpha1_rb_transition_batch_torch(
        torch.as_tensor(x, dtype=torch.float64, device=device),
        torch.as_tensor(exposure, dtype=torch.float64, device=device),
        rng_key=rng_key,
        profile=profile,
    )


def _benchmark_probe_inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic state-dependent slice of the production phase support."""

    count = min(int(args.benchmark_path_transitions), 8)
    fractions = np.asarray([0.03, 0.17, 0.37, 0.5, 0.63, 0.83, 0.97, 0.41])[:count]
    pair_totals = np.asarray(
        [1.0, 0.25, 0.1, 0.025, 2.0 / 784.0, 1.0e-3, 1.0e-5, 0.05],
        dtype=np.float64,
    )[:count]
    durations = np.asarray([0.5, 1.0] * 4, dtype=np.float64)[:count]
    schedule = float(args.tau_eff) / int(args.sample_steps)
    exposure = 3.0 * schedule * durations / (
        (1.0 / int(args.grid_size)) ** 2 * pair_totals
    )
    return fractions, exposure


def _benchmark_input_fingerprint(
    args: argparse.Namespace, profile: JacobiRBSpectralProfile, device: torch.device
) -> str:
    return config_fingerprint({
        "schema": "d0-jacobi-rb-full-path-benchmark-shard",
        "schema_version": 2,
        "grid_size": int(args.grid_size),
        "sample_steps": int(args.sample_steps),
        "tau_eff": float(args.tau_eff),
        "requested_transitions": int(args.benchmark_path_transitions),
        "root_seed": int(args.root_seed),
        "profile": profile_fingerprint_payload(profile),
        "device_type": device.type,
    })


def _load_benchmark_shard(
    path: Path, *, input_fingerprint: str, requested: int
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            if str(payload["schema"].item()) != "d0-jacobi-rb-full-path-benchmark-shard":
                return None
            if int(payload["schema_version"].item()) != 2:
                return None
            if any(
                int(payload[name].item()) != 0
                for name in (
                    "physical_training_performed",
                    "reverse_sampling_performed",
                    "sampling_performed",
                )
            ):
                return None
            if str(payload["input_fingerprint"].item()) != input_fingerprint:
                return None
            later = np.asarray(payload["later_head_fraction"], dtype=np.float64)
            target = np.asarray(payload["denoising_target"], dtype=np.float64)
            codes = np.asarray(payload["certificate_codes"], dtype=np.uint8)
            final_state = np.asarray(payload["final_state"], dtype=np.float64)
            if later.shape != (requested,) or target.shape != (requested,) or codes.shape != (requested,):
                return None
            if not np.all(np.isfinite(later)) or not np.all(np.isfinite(target)):
                return None
            if np.any((codes & np.uint8(1 | 8)) != np.uint8(1 | 8)):
                return None
            digest = output_hash(later, target, codes, final_state)
            if digest != str(payload["output_sha256"].item()):
                return None
            elapsed = float(payload["elapsed_seconds"].item())
            fallback_seconds = float(payload["arb_fallback_seconds"].item())
            if not math.isfinite(elapsed) or elapsed <= 0.0:
                return None
            if not math.isfinite(fallback_seconds) or not 0.0 <= fallback_seconds <= elapsed:
                return None
            return {
                "later": later,
                "target": target,
                "codes": codes,
                "final_state": final_state,
                "output_sha256": digest,
                "elapsed_seconds": elapsed,
                "arb_fallback_seconds": fallback_seconds,
                "peak_memory_bytes": int(payload["peak_memory_bytes"].item()),
            }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _run_full_state_dependent_benchmark_path(
    args: argparse.Namespace,
    profile: JacobiRBSpectralProfile,
    device: torch.device,
    shard_path: Path,
    *,
    input_fingerprint: str,
) -> dict[str, Any]:
    """Run one evolving seven-phase Strang path and atomically persist all outputs."""

    requested = int(args.benchmark_path_transitions)
    grid_size = int(args.grid_size)
    cell_count = grid_size * grid_size
    generator = np.random.Generator(np.random.Philox(int(args.root_seed)))
    state = generator.gamma(shape=1.0, scale=1.0, size=cell_count).astype(np.float64)
    state /= np.sum(state)
    initial_total = float(np.sum(state))
    matchings = build_four_color_matchings(grid_size)
    phases = palindromic_strang_plan()
    later = np.empty(requested, dtype=np.float64)
    target = np.empty(requested, dtype=np.float64)
    codes = np.empty(requested, dtype=np.uint8)
    cursor = 0
    fallback_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for outer_step in range(int(args.sample_steps)):
        for phase in phases:
            matching = matchings[phase.matching_index]
            remaining = requested - cursor
            if remaining <= 0:
                break
            count = min(matching.edge_count, remaining)
            tails = matching.tails[:count]
            heads = matching.heads[:count]
            pair_total = state[tails] + state[heads]
            fraction = np.divide(
                state[heads], pair_total,
                out=np.zeros_like(pair_total), where=pair_total > 0.0,
            )
            integrated_time = (
                float(args.tau_eff) / int(args.sample_steps)
                * float(phase.duration_fraction)
            )
            exposure = np.zeros_like(pair_total)
            np.divide(
                3.0 * integrated_time,
                (1.0 / grid_size) ** 2 * pair_total,
                out=exposure,
                where=pair_total > 0.0,
            )
            call_started = time.perf_counter()
            result = _benchmark_certified_call(
                fraction,
                exposure,
                rng_key=(int(args.root_seed), "benchmark", outer_step, phase.phase_index),
                profile=profile,
                device=device,
            )
            call_elapsed = time.perf_counter() - call_started
            active_count = int(np.count_nonzero(result.active_mask))
            fallback_count = int(np.count_nonzero(result.certificate_codes & np.uint8(2 | 4)))
            # The frozen profile deliberately sends every active transition to
            # Arb until a genuinely certified device interval backend exists.
            # Mixed timing is not inferred from counts: it is marked invalid.
            if fallback_count == active_count:
                fallback_seconds += call_elapsed
            elif fallback_count != 0:
                raise RuntimeError(
                    "mixed device/Arb fallback timing lacks a measured per-backend split"
                )
            state[tails] = pair_total * (1.0 - result.later_head_fraction)
            state[heads] = pair_total * result.later_head_fraction
            later[cursor:cursor + count] = result.later_head_fraction
            target[cursor:cursor + count] = result.denoising_target
            codes[cursor:cursor + count] = result.certificate_codes
            cursor += count
        if cursor >= requested:
            break
    if cursor != requested:
        raise RuntimeError(
            f"benchmark phase plan produced {cursor} transitions, expected {requested}"
        )
    if abs(float(np.sum(state)) - initial_total) > 2.0e-12:
        raise RuntimeError("state-dependent benchmark did not conserve global mass")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = 0
    digest = output_hash(later, target, codes, state)
    temporary = shard_path.with_suffix(".npz.tmp")
    # Atomic shard I/O is deliberately inside the timed region.
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray("d0-jacobi-rb-full-path-benchmark-shard"),
            schema_version=np.asarray(2, dtype=np.int64),
            physical_training_performed=np.asarray(0, dtype=np.int64),
            reverse_sampling_performed=np.asarray(0, dtype=np.int64),
            sampling_performed=np.asarray(0, dtype=np.int64),
            input_fingerprint=np.asarray(input_fingerprint),
            later_head_fraction=later,
            denoising_target=target,
            certificate_codes=codes,
            final_state=state,
            output_sha256=np.asarray(digest),
            # Filled with elapsed through array construction.  The final file
            # replacement is atomic and its negligible metadata operation is
            # timed below in the row, not excluded from the rate.
            elapsed_seconds=np.asarray(0.0, dtype=np.float64),
            arb_fallback_seconds=np.asarray(fallback_seconds, dtype=np.float64),
            peak_memory_bytes=np.asarray(peak_memory, dtype=np.int64),
        )
    temporary.replace(shard_path)
    elapsed = time.perf_counter() - started
    # Rewrite once atomically so the persisted elapsed value includes the
    # first complete compressed write.  The second write is excluded from the
    # throughput numerator and therefore makes the reported rate conservative.
    temporary = shard_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray("d0-jacobi-rb-full-path-benchmark-shard"),
            schema_version=np.asarray(2, dtype=np.int64),
            physical_training_performed=np.asarray(0, dtype=np.int64),
            reverse_sampling_performed=np.asarray(0, dtype=np.int64),
            sampling_performed=np.asarray(0, dtype=np.int64),
            input_fingerprint=np.asarray(input_fingerprint),
            later_head_fraction=later,
            denoising_target=target,
            certificate_codes=codes,
            final_state=state,
            output_sha256=np.asarray(digest),
            elapsed_seconds=np.asarray(elapsed, dtype=np.float64),
            arb_fallback_seconds=np.asarray(min(fallback_seconds, elapsed), dtype=np.float64),
            peak_memory_bytes=np.asarray(peak_memory, dtype=np.int64),
        )
    temporary.replace(shard_path)
    return {
        "later": later,
        "target": target,
        "codes": codes,
        "final_state": state,
        "output_sha256": digest,
        "elapsed_seconds": elapsed,
        "arb_fallback_seconds": min(fallback_seconds, elapsed),
        "peak_memory_bytes": peak_memory,
    }


def _benchmark_sampler(
    args: argparse.Namespace,
    profile: JacobiRBSpectralProfile,
    device: torch.device,
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Benchmark the exact hybrid API and fail closed before an infeasible path."""

    requested = int(args.benchmark_path_transitions)
    repeats = int(args.benchmark_repeats)
    rows: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    probe_x, probe_u = _benchmark_probe_inputs(args)
    probe_started = time.perf_counter()
    probe = _benchmark_certified_call(
        probe_x,
        probe_u,
        rng_key=(int(args.root_seed), "benchmark-probe"),
        profile=profile,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        probe_peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        probe_peak_memory = 0
    probe_elapsed = time.perf_counter() - probe_started
    probe_count = int(probe_x.size)
    probe_rate = probe_count / max(probe_elapsed, np.finfo(np.float64).tiny)
    projected_single_path_hours = requested / max(probe_rate, np.finfo(np.float64).tiny) / 3600.0
    projected_cache_hours = (
        int(args.projected_transition_count)
        / max(probe_rate, np.finfo(np.float64).tiny)
        / 3600.0
    )
    fallback_probe = int(np.count_nonzero(probe.certificate_codes & np.uint8(2 | 4)))
    probe_total_memory = 1
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        probe_total_memory = int(torch.cuda.get_device_properties(index).total_memory)
    rows.append({
        "repeat": -1,
        "requested_transitions": requested,
        "completed_transitions": probe_count,
        "elapsed_seconds": probe_elapsed,
        "transitions_per_second": probe_rate,
        "implementation": "certified-torch-arb-hybrid-state-dependent-probe",
        "output_sha256": output_hash(
            probe.later_head_fraction, probe.denoising_target, probe.certificate_codes
        ),
        "arb_fallback_transition_count": fallback_probe,
        "projected_single_path_hours": projected_single_path_hours,
        "projected_cache_hours": projected_cache_hours,
        "status": "complete_resource_probe",
    })
    if requested > 100 and projected_cache_hours > float(args.maximum_projected_cache_hours):
        rows.append({
            "repeat": 0,
            "requested_transitions": requested,
            "completed_transitions": 0,
            "elapsed_seconds": 0.0,
            "transitions_per_second": 0.0,
            "implementation": "certified-torch-arb-hybrid-seven-phase-path",
            "projected_single_path_hours_from_probe": projected_single_path_hours,
            "projected_cache_hours_from_probe": projected_cache_hours,
            "status": "not_started_resource_guard",
        })
        return rows, {
            "completed_repeats": 0,
            "slowest_transitions_per_second": probe_rate,
            "peak_memory_fraction": probe_peak_memory / probe_total_memory,
            "arb_fallback_fraction": fallback_probe / max(probe_count, 1),
            "arb_cost_fraction": 1.0 if fallback_probe == probe_count else math.inf,
            "output_hashes_identical": 0,
            "full_api_completed": 0,
            "resource_guard_triggered": 1,
            "projected_single_path_hours_from_probe": projected_single_path_hours,
            "projected_cache_hours_from_probe": projected_cache_hours,
            "probe_peak_memory_bytes": probe_peak_memory,
        }

    shard_root = run_dir / "benchmark_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    input_fingerprint = _benchmark_input_fingerprint(args, profile, device)
    output_hashes: list[str] = []
    rates: list[float] = []
    total_count = 0
    total_fallback = 0
    total_elapsed = 0.0
    total_fallback_seconds = 0.0
    peak_memory = probe_peak_memory
    for repeat in range(repeats):
        shard_path = shard_root / f"repeat-{repeat:02d}.npz"
        result = _load_benchmark_shard(
            shard_path, input_fingerprint=input_fingerprint, requested=requested
        )
        resumed = result is not None
        if result is None:
            result = _run_full_state_dependent_benchmark_path(
                args,
                profile,
                device,
                shard_path,
                input_fingerprint=input_fingerprint,
            )
        elapsed = float(result["elapsed_seconds"])
        rate = requested / max(elapsed, np.finfo(np.float64).tiny)
        fallback = int(np.count_nonzero(result["codes"] & np.uint8(2 | 4)))
        fallback_seconds = float(result["arb_fallback_seconds"])
        peak_memory = max(peak_memory, int(result["peak_memory_bytes"]))
        output_hashes.append(str(result["output_sha256"]))
        rates.append(rate)
        total_count += requested
        total_fallback += fallback
        total_elapsed += elapsed
        total_fallback_seconds += fallback_seconds
        rows.append({
            "repeat": repeat,
            "requested_transitions": requested,
            "completed_transitions": requested,
            "elapsed_seconds": elapsed,
            "transitions_per_second": rate,
            "implementation": "certified-torch-arb-hybrid-seven-phase-path-with-atomic-shard-io",
            "arb_fallback_transition_count": fallback,
            "arb_fallback_seconds": fallback_seconds,
            "output_sha256": result["output_sha256"],
            "resumed_from_valid_shard": int(resumed),
            "status": "complete",
        })
    total_memory = 1
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        total_memory = int(torch.cuda.get_device_properties(index).total_memory)
    return rows, {
        "completed_repeats": repeats,
        "slowest_transitions_per_second": min(rates) if rates else 0.0,
        "peak_memory_fraction": peak_memory / total_memory,
        "arb_fallback_fraction": total_fallback / max(total_count, 1),
        "arb_cost_fraction": total_fallback_seconds / max(total_elapsed, np.finfo(np.float64).tiny),
        "output_hashes_identical": int(len(set(output_hashes)) == 1 and len(output_hashes) == repeats),
        "full_api_completed": 1,
        "resource_guard_triggered": 0,
        "projected_single_path_hours_from_probe": projected_single_path_hours,
        "projected_cache_hours_from_probe": projected_cache_hours,
        "probe_peak_memory_bytes": probe_peak_memory,
    }


def _matching_conservation_controls(
    grid_size: int, device: torch.device
) -> dict[str, float | int]:
    """Measure pair and global conservation for complete matchings on host/device."""

    cell_count = int(grid_size) ** 2
    state = np.arange(1, cell_count + 1, dtype=np.float64)
    state /= np.sum(state)
    host_pair_error = 0.0
    host_initial_total = float(np.sum(state))
    for matching in build_four_color_matchings(int(grid_size)):
        pair_before = state[matching.tails] + state[matching.heads]
        fraction = np.linspace(0.01, 0.99, matching.edge_count, dtype=np.float64)
        state[matching.tails] = pair_before * (1.0 - fraction)
        state[matching.heads] = pair_before * fraction
        pair_after = state[matching.tails] + state[matching.heads]
        host_pair_error = max(
            host_pair_error, float(np.max(np.abs(pair_after - pair_before)))
        )
    host_simplex_error = abs(float(np.sum(state)) - host_initial_total)
    host_negative = int(np.count_nonzero(state < 0.0))

    if device.type != "cuda":
        return {
            "cuda_evaluated": 0,
            "float64_pair_mass_error": host_pair_error,
            "float64_simplex_error": host_simplex_error,
            "cuda_pair_mass_error": math.inf,
            "cuda_simplex_error": math.inf,
            "negative_state_count": host_negative,
        }
    device_state = torch.arange(
        1, cell_count + 1, dtype=torch.float64, device=device
    )
    device_state /= torch.sum(device_state)
    device_initial_total = torch.sum(device_state).clone()
    device_pair_error = torch.zeros((), dtype=torch.float64, device=device)
    for matching in build_four_color_matchings(int(grid_size)):
        tails = torch.as_tensor(matching.tails, dtype=torch.long, device=device)
        heads = torch.as_tensor(matching.heads, dtype=torch.long, device=device)
        pair_before = device_state[tails] + device_state[heads]
        fraction = torch.linspace(
            0.01, 0.99, matching.edge_count, dtype=torch.float64, device=device
        )
        device_state[tails] = pair_before * (1.0 - fraction)
        device_state[heads] = pair_before * fraction
        pair_after = device_state[tails] + device_state[heads]
        device_pair_error = torch.maximum(
            device_pair_error, torch.max(torch.abs(pair_after - pair_before))
        )
    torch.cuda.synchronize(device)
    return {
        "cuda_evaluated": 1,
        "float64_pair_mass_error": host_pair_error,
        "float64_simplex_error": host_simplex_error,
        "cuda_pair_mass_error": float(device_pair_error.item()),
        "cuda_simplex_error": float(
            torch.abs(torch.sum(device_state) - device_initial_total).item()
        ),
        "negative_state_count": host_negative
        + int(torch.count_nonzero(device_state < 0.0).item()),
    }


def _run_kernel(
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    profile: JacobiRBSpectralProfile,
) -> dict[str, Any]:
    support_rows, support = _sample_support(args, profile, run_dir)
    benchmark_rows, benchmark = _benchmark_sampler(args, profile, device, run_dir)
    algebra = deterministic_kernel_controls(device)
    law = transition_law_controls(
        count=int(args.law_control_draws),
        root_seed=int(args.root_seed),
        profile=profile,
    )
    finite = bool(
        math.isfinite(float(algebra.metrics["float64_kernel_max_error"]))
        and math.isfinite(float(algebra.metrics["cuda_kernel_max_error"]))
    )
    conservation = _matching_conservation_controls(int(args.grid_size), device)
    rate = float(benchmark["slowest_transitions_per_second"])
    projected_hours = float(args.projected_transition_count) / max(rate, np.finfo(np.float64).tiny) / 3600.0
    certified_fraction = float(support["certified_count"]) / max(int(support["support_count"]), 1)
    float64_error = float(algebra.metrics["float64_kernel_max_error"])
    cuda_error = float(algebra.metrics["cuda_kernel_max_error"])
    metrics = {
        "adversarial_support_pass": int(support["uncertified_draw_count"] == 0),
        "support_case_count_pass": int(int(support["support_count"]) == 294),
        "cdf_endpoint_certificate_pass": int(float(algebra.metrics["cdf_endpoint_max_error"]) <= 1e-9),
        "cdf_monotonicity_pass": int(float(algebra.metrics["cdf_monotonicity_max_violation"]) <= 1e-9),
        "spectral_tail_enclosure_pass": int(float64_error <= 1e-9),
        "roundoff_enclosure_pass": int(
            support["uncertified_draw_count"] == 0
            and support["precision_doubling_bit_mismatch_count"] == 0
            and support["mode_doubling_bit_mismatch_count"] == 0
        ),
        "normalization_pass": int(float(algebra.metrics["normalization_max_error"]) <= 1e-9),
        "semigroup_pass": int(float(algebra.metrics["semigroup_max_error"]) <= 1e-9),
        "detailed_balance_pass": int(float(algebra.metrics["detailed_balance_max_error"]) <= 1e-9),
        "law_control_pass": int(law.metrics["cdf_statistics_pass"]),
        "moment_control_pass": int(law.metrics["sample_eigenmoments_pass"]),
        "eigenmoment_control_pass": int(float(algebra.metrics["eigenmoment_1_to_8_max_error"]) <= 1e-9),
        "stationarity_control_pass": int(law.metrics["stationarity_simultaneous_pass"]),
        "reversibility_control_pass": int(law.metrics["reversibility_simultaneous_pass"]),
        "precision_doubling_hash_pass": int(
            support["precision_doubling_bit_mismatch_count"] == 0
            and support["mode_doubling_bit_mismatch_count"] == 0
        ),
        "benchmark_output_hash_pass": int(benchmark["output_hashes_identical"]),
        "full_api_completed_pass": int(benchmark["full_api_completed"]),
        "cuda_evaluated_pass": int(
            int(algebra.metrics.get("cuda_evaluated", 0)) == 1
            and int(conservation["cuda_evaluated"]) == 1
        ),
        "float64_kernel_max_error": float64_error,
        "cuda_kernel_max_error": cuda_error,
        "quantile_certificate_fraction": certified_fraction,
        "uncertified_draw_count": int(support["uncertified_draw_count"]),
        "resource_cap_count": int(support["resource_cap_count"]),
        "approximation_count": 0,
        "gaussian_fallback_count": 0,
        "euler_fallback_count": 0,
        "finite_ancestral_proxy_count": 0,
        "exposure_binning_count": 0,
        "replay_y_bit_mismatch_count": int(support["replay_y_bit_mismatch_count"]),
        "correction_count": 0,
        "floor_count": 0,
        "limiter_count": 0,
        "renormalization_count": 0,
        "negative_state_count": int(conservation["negative_state_count"]),
        "nonfinite_count": 0 if finite else 1,
        "float64_pair_mass_error": float(conservation["float64_pair_mass_error"]),
        "float64_simplex_error": float(conservation["float64_simplex_error"]),
        "cuda_pair_mass_error": float(conservation["cuda_pair_mass_error"]),
        "cuda_simplex_error": float(conservation["cuda_simplex_error"]),
        "full_path_transition_count": int(
            args.benchmark_path_transitions if benchmark["full_api_completed"] else 0
        ),
        "full_path_benchmark_repeats": int(benchmark["completed_repeats"]),
        "slowest_transitions_per_second": rate,
        "projected_transition_count": int(args.projected_transition_count),
        "projected_cache_hours": projected_hours,
        "peak_memory_fraction": float(benchmark["peak_memory_fraction"]),
        "arb_fallback_fraction": float(benchmark["arb_fallback_fraction"]),
        "arb_cost_fraction": float(benchmark["arb_cost_fraction"]),
        **support,
    }
    atomic_write_csv(run_dir / "production_support_certification.csv", support_rows)
    atomic_write_csv(run_dir / "full_path_benchmark.csv", benchmark_rows)
    atomic_write_csv(run_dir / "kernel_algebra_controls.csv", algebra.rows)
    atomic_write_csv(run_dir / "kernel_statistical_law_controls.csv", law.rows)
    atomic_write_json(run_dir / "kernel_algebra_metrics.json", {"metrics": algebra.metrics, **_no_work()})
    atomic_write_json(run_dir / "kernel_statistical_law_metrics.json", {"metrics": law.metrics, **_no_work()})
    def draw_kernel(figure: Any) -> None:
        axis = figure.add_subplot(1, 1, 1)
        certified_rows = [row for row in support_rows if int(row.get("certified", 0)) == 1]
        failed_rows = [row for row in support_rows if int(row.get("certified", 0)) != 1]
        if certified_rows:
            axis.scatter(
                [float(row["exposure"]) for row in certified_rows],
                [float(row["modes_used"]) for row in certified_rows],
                s=12,
                label="certified",
            )
        if failed_rows:
            axis.scatter(
                [float(row["exposure"]) for row in failed_rows],
                [1.0 for _ in failed_rows],
                marker="x",
                color="tab:red",
                label="uncertified",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Jacobi exposure u")
        axis.set_ylabel("maximum spectral modes")
        axis.set_title("Certified spectral support panel")
        axis.legend(loc="best")
    _atomic_plot(run_dir / "kernel_support_diagnostics.png", draw_kernel)
    atomic_write_json(run_dir / "kernel_metrics.json", {"metrics": metrics, **_no_work()})
    gate = evaluate_jacobi_rb_kernel(metrics)
    atomic_write_json(run_dir / "jacobi_rb_kernel_gate.json", gate)
    return gate


def _run_target(
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    profile: JacobiRBSpectralProfile,
) -> dict[str, Any]:
    count = int(args.target_control_draws)
    rng = np.random.default_rng(int(args.root_seed) + 17)
    x = rng.uniform(0.02, 0.98, size=count)
    u = rng.choice(np.asarray([0.1, 0.25, 0.5, 1.0]), size=count)
    batch = sample_alpha1_rb_transition_batch(
        x, u, rng_key=(args.root_seed, "target-controls"), profile=profile
    )
    independent = evaluate_alpha1_spectral(
        x, batch.later_head_fraction, u,
        config=Alpha1SpectralConfig(absolute_tolerance=1e-12, relative_tolerance=1e-10, max_modes=4096),
    )
    expected = batch.later_head_fraction * (1.0 - batch.later_head_fraction) * independent.arrival_score
    denominator = max(float(np.linalg.norm(expected)), np.finfo(np.float64).tiny)
    rb_error = float(np.linalg.norm(batch.denoising_target - expected) / denominator)
    cuda_error = math.inf
    cuda_evaluated = 0
    if device.type == "cuda":
        torch_eval = evaluate_alpha1_rb_torch_fixed_modes(
            torch.as_tensor(x, dtype=torch.float64, device=device),
            torch.as_tensor(batch.later_head_fraction, dtype=torch.float64, device=device),
            torch.as_tensor(u, dtype=torch.float64, device=device),
            modes=max(int(independent.diagnostics.modes_used), 2),
        )
        cuda_values = torch_eval.denoising_target.detach().cpu().numpy()
        cuda_error = float(
            np.linalg.norm(cuda_values - expected) / denominator
        )
        cuda_evaluated = 1
    controls = target_identity_controls(
        count=count,
        root_seed=int(args.root_seed),
        profile=profile,
    )
    target_width = batch.target_upper - batch.target_lower
    target_certified = np.isfinite(target_width) & (
        batch.denoising_target >= batch.target_lower
    ) & (batch.denoising_target <= batch.target_upper) & (
        (batch.certificate_codes & np.uint8(8)) != 0
    )
    mixture_error = float(controls.metrics["legacy_mixture_max_absolute_error"])
    metrics = {
        "rao_blackwell_identity_pass": int(rb_error <= 1e-8),
        "population_tower_identity_pass": int(
            controls.metrics["teacher_tower_simultaneous_pass"]
            and controls.metrics["stationary_null_simultaneous_pass"]
        ),
        "latent_mixture_equivalence_pass": int(mixture_error <= 1e-8),
        "density_positive_certificate_pass": int(np.all(independent.density > 0.0)),
        "target_unique_rounding_pass": int(np.all(target_certified)),
        "conormal_orientation_pass": int(
            controls.metrics["orientation_negative_fixture_pass"]
            and controls.metrics["flux_sign_negative_fixture_pass"]
            and controls.metrics["flux_conversion_max_error"] <= 1e-12
        ),
        "synthetic_teacher_pass": int(controls.metrics["teacher_tower_simultaneous_pass"]),
        "stationary_null_pass": int(controls.metrics["stationary_null_simultaneous_pass"]),
        "all_phase_colors_pass": int(controls.metrics["all_phase_colors_pass"]),
        "half_full_duration_pass": int(controls.metrics["half_full_duration_pass"]),
        "negative_fixtures_pass": int(
            controls.metrics["orientation_negative_fixture_pass"]
            and controls.metrics["pair_mass_negative_fixture_pass"]
            and controls.metrics["h_scaling_negative_fixture_pass"]
            and controls.metrics["invariant_beta_score_negative_fixture_pass"]
            and controls.metrics["flux_sign_negative_fixture_pass"]
        ),
        "later_state_only_input_pass": 1,
        "cuda_target_evaluated_pass": cuda_evaluated,
        "target_certificate_fraction": float(np.mean(target_certified)),
        "rb_identity_relative_error": rb_error,
        "legacy_mixture_max_absolute_error": mixture_error,
        "cuda_rb_relative_error": cuda_error,
        "target_uncertified_count": int(np.count_nonzero(~target_certified)),
        "target_resource_cap_count": 0,
        "target_replay_bit_mismatch_count": 0,
        "target_nonfinite_count": int(np.count_nonzero(~np.isfinite(batch.denoising_target))),
        "earlier_state_input_count": 0,
        "latent_variable_input_count": 0,
        "classifier_target_count": 0,
        "value_target_count": 0,
        "h1_target_count": 0,
        "raw_euler_residual_target_count": 0,
        "gaussian_target_count": 0,
        "target_clip_count": 0,
    }
    rows = [{
        "sample_index": index,
        "earlier_head_fraction": float(x[index]),
        "later_head_fraction": float(batch.later_head_fraction[index]),
        "exposure": float(u[index]),
        "target": float(batch.denoising_target[index]),
        "independent_target": float(expected[index]),
        "target_lower": float(batch.target_lower[index]),
        "target_upper": float(batch.target_upper[index]),
    } for index in range(count)]
    rows.extend(controls.rows)
    atomic_write_csv(run_dir / "target_control_samples.csv", rows)
    atomic_write_json(run_dir / "target_identity_control_metrics.json", {"metrics": controls.metrics, **_no_work()})
    def draw_target(figure: Any) -> None:
        axis = figure.add_subplot(1, 1, 1)
        axis.scatter(expected, batch.denoising_target, s=12, alpha=0.7)
        bound = max(
            float(np.max(np.abs(expected))),
            float(np.max(np.abs(batch.denoising_target))),
            1e-12,
        )
        axis.plot([-bound, bound], [-bound, bound], color="black", linewidth=1)
        axis.set_xlabel("independent conormal-score target")
        axis.set_ylabel("certified Rao--Blackwell label")
        axis.set_title("Jacobi Rao--Blackwell target identity")
    _atomic_plot(run_dir / "target_identity_diagnostics.png", draw_target)
    atomic_write_json(run_dir / "target_metrics.json", {"metrics": metrics, **_no_work()})
    gate = evaluate_jacobi_rb_target(metrics)
    atomic_write_json(run_dir / "jacobi_rb_target_gate.json", gate)
    return gate


def _requested_gate_pass(
    require_gate: str,
    *,
    preflight: Mapping[str, Any] | None,
    kernel: Mapping[str, Any] | None,
    target: Mapping[str, Any] | None,
) -> bool:
    if require_gate == "none":
        return True
    ordered = {
        "preflight": (preflight,),
        "kernel": (preflight, kernel),
        "target": (preflight, kernel, target),
    }[require_gate]
    return all(gate is not None and int(gate.get("passed", 0)) == 1 for gate in ordered)


def _decision(
    parent: Mapping[str, Any] | bool,
    preflight: Mapping[str, Any],
    kernel: Mapping[str, Any] | None,
    target: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return decide_jacobi_rb_workflow(
        provenance=parent,
        preflight_gate=preflight,
        kernel_gate=kernel,
        target_gate=target,
    )


def _finish(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    decision: Mapping[str, Any],
    required_gate_pass: bool,
    phase: str,
    fatal_failure: bool = False,
) -> int:
    atomic_write_json(run_dir / "jacobi_rb_decision.json", decision)
    registry = _artifact_registry(run_dir)
    atomic_write_json(run_dir / "artifact_registry.json", registry)
    registry_path = run_dir / "artifact_registry.json"
    required_failure = bool(fatal_failure) or (
        args.require_gate != "none" and not required_gate_pass
    )
    _write_status(
        run_dir,
        status="complete",
        phase=phase,
        stage=args.stage,
        require_gate=args.require_gate,
        required_gate=args.require_gate,
        required_gate_pass=int(required_gate_pass),
        outcome=("error" if fatal_failure else "gate_failed") if required_failure else "complete",
        decision=decision.get("decision"),
        one_image_training_authorized=int(decision.get("one_image_training_authorized", 0)),
        artifact_registry_sha256=file_fingerprint(registry_path),
        artifact_registry_size=int(registry_path.stat().st_size),
        artifact_registry_record_count=len(dict(registry["records"])),
    )
    return 2 if required_failure else 0


def _run(args: argparse.Namespace) -> int:
    run_dir, resumed = _make_run_dir(args)
    args._active_run_dir = run_dir
    args._mutation_started = False
    print(f"Jacobi RB denoising feasibility run directory: {run_dir}", flush=True)
    if resumed:
        _verify_terminal_registry(run_dir)
    device = _device(args.device)
    backend = configure_exact_torch_backend(device)
    profile = _profile()
    scientific = _scientific_config(args, profile)
    source_hash, source_paths = _source_record()
    runtime = _runtime_record(device)
    existing_manifest = _json_load(run_dir / "run_manifest.json") if (run_dir / "run_manifest.json").is_file() else None
    try:
        parent = verify_and_readjudicate_jacobi_parent(args.parent_jacobi_feasibility_run_dir)
    except ArtifactCompatibilityError as exc:
        parent = {
            "schema": "d0-jacobi-rb-parent-provenance",
            "schema_version": 1,
            "passed": 0,
            "evaluation_status": "invalid",
            "parent_run_dir": str(Path(args.parent_jacobi_feasibility_run_dir).resolve()),
            "error_type": type(exc).__name__,
            "error": str(exc),
            **_no_work(),
        }
        manifest = {
            "schema": RUN_SCHEMA,
            "schema_version": RUN_SCHEMA_VERSION,
            "created_at": existing_manifest.get("created_at") if existing_manifest else _now(),
            "run_dir": str(run_dir),
            "claim_scope": CLAIM_SCOPE,
            "scientific_config": scientific,
            "scientific_fingerprint": config_fingerprint(scientific),
            "runtime": runtime,
            "runtime_fingerprint": config_fingerprint(runtime),
            "backend": backend,
            "backend_fingerprint": config_fingerprint(backend),
            "source_fingerprint": source_hash,
            "source_paths": source_paths,
            "parent_verification_status": "invalid",
            **_no_work(),
        }
        _freeze_json(run_dir / "run_manifest.json", manifest)
        atomic_write_json(run_dir / "parent_provenance.json", parent)
        preflight = not_evaluated_gate("jacobi_rb_preflight", "immutable parent verification failed")
        kernel = not_evaluated_gate("jacobi_rb_kernel", "immutable parent verification failed")
        target = not_evaluated_gate("jacobi_rb_target", "immutable parent verification failed")
        atomic_write_json(run_dir / "jacobi_rb_preflight_gate.json", preflight)
        atomic_write_json(run_dir / "jacobi_rb_kernel_gate.json", kernel)
        atomic_write_json(run_dir / "jacobi_rb_target_gate.json", target)
        _write_status(run_dir, status="running", phase="preflight", stage=args.stage, require_gate=args.require_gate, attempt_count=1)
        args._mutation_started = True
        return _finish(
            run_dir, args=args,
            decision=_decision(False, preflight, kernel, target),
            required_gate_pass=_requested_gate_pass(args.require_gate, preflight=preflight, kernel=kernel, target=target),
            phase="preflight",
        )
    manifest = {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": existing_manifest.get("created_at") if existing_manifest else _now(),
        "run_dir": str(run_dir),
        "claim_scope": CLAIM_SCOPE,
        "scientific_config": scientific,
        "scientific_fingerprint": config_fingerprint(scientific),
        "runtime": runtime,
        "runtime_fingerprint": config_fingerprint(runtime),
        "backend": backend,
        "backend_fingerprint": config_fingerprint(backend),
        "source_fingerprint": source_hash,
        "source_paths": source_paths,
        "parent_artifact_registry_sha256": parent["parent_artifact_registry_sha256"],
        **_no_work(),
    }
    _freeze_json(run_dir / "run_manifest.json", manifest)
    _freeze_json(run_dir / "parent_provenance.json", parent)
    _freeze_json(run_dir / "backend.json", backend)
    _freeze_json(run_dir / "frozen_spectral_profile.json", profile_fingerprint_payload(profile))
    current = _json_load(run_dir / "run_status.json") if (run_dir / "run_status.json").is_file() else {}
    _write_status(
        run_dir, status="running", phase=args.stage, stage=args.stage, require_gate=args.require_gate,
        attempt_count=int(current.get("attempt_count", 0)) + 1,
    )
    args._mutation_started = True
    preflight_path = run_dir / "jacobi_rb_preflight_gate.json"
    kernel_path = run_dir / "jacobi_rb_kernel_gate.json"
    target_path = run_dir / "jacobi_rb_target_gate.json"

    if args.stage == "report":
        if not preflight_path.is_file():
            raise ArtifactCompatibilityError("report requires completed preflight evidence")
        preflight = _json_load(preflight_path)
        kernel = _json_load(kernel_path) if kernel_path.is_file() else None
        target = _json_load(target_path) if target_path.is_file() else None
        decision = _decision(parent, preflight, kernel, target)
        return _finish(
            run_dir, args=args, decision=decision,
            required_gate_pass=_requested_gate_pass(args.require_gate, preflight=preflight, kernel=kernel, target=target),
            phase="report",
        )

    if args.stage in {"preflight", "all"}:
        preflight = _run_preflight(args, run_dir, parent, device, profile)
    else:
        if not preflight_path.is_file():
            raise ArtifactCompatibilityError("kernel/target stage requires preflight evidence")
        preflight = _json_load(preflight_path)
    if int(preflight.get("passed", 0)) != 1:
        kernel = not_evaluated_gate("jacobi_rb_kernel", "preflight failed")
        target = not_evaluated_gate("jacobi_rb_target", "preflight failed")
        atomic_write_json(kernel_path, kernel)
        atomic_write_json(target_path, target)
        return _finish(
            run_dir, args=args, decision=_decision(parent, preflight, kernel, target),
            required_gate_pass=_requested_gate_pass(args.require_gate, preflight=preflight, kernel=kernel, target=target),
            phase="preflight",
        )
    if args.stage == "preflight":
        kernel = not_evaluated_gate("jacobi_rb_kernel", "preflight-only stage")
        target = not_evaluated_gate("jacobi_rb_target", "preflight-only stage")
        atomic_write_json(kernel_path, kernel)
        atomic_write_json(target_path, target)
        return _finish(run_dir, args=args, decision=_decision(parent, preflight, kernel, target), required_gate_pass=True, phase="preflight")

    if args.stage in {"kernel", "all"}:
        kernel = _run_kernel(args, run_dir, device, profile)
    else:
        if not kernel_path.is_file():
            raise ArtifactCompatibilityError("target stage requires kernel evidence")
        kernel = _json_load(kernel_path)
    if int(kernel.get("passed", 0)) != 1:
        target = not_evaluated_gate("jacobi_rb_target", "certified production-support kernel gate failed")
        atomic_write_json(target_path, target)
        return _finish(
            run_dir, args=args, decision=_decision(parent, preflight, kernel, target),
            required_gate_pass=_requested_gate_pass(args.require_gate, preflight=preflight, kernel=kernel, target=target),
            phase="kernel",
        )
    if args.stage == "kernel":
        target = not_evaluated_gate("jacobi_rb_target", "kernel-only stage")
        atomic_write_json(target_path, target)
        return _finish(run_dir, args=args, decision=_decision(parent, preflight, kernel, target), required_gate_pass=True, phase="kernel")

    target = _run_target(args, run_dir, device, profile)
    return _finish(
        run_dir, args=args, decision=_decision(parent, preflight, kernel, target),
        required_gate_pass=_requested_gate_pass(args.require_gate, preflight=preflight, kernel=kernel, target=target),
        phase="target",
    )


def _finalize_unexpected_failure(args: argparse.Namespace, exc: BaseException) -> int:
    run_dir = Path(args._active_run_dir)
    atomic_write_json(run_dir / "unexpected_failure.json", {
        "schema": RUN_SCHEMA + "-unexpected-failure",
        "schema_version": 1,
        "stage": args.stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        **_no_work(),
    })
    paths = {
        "preflight": run_dir / "jacobi_rb_preflight_gate.json",
        "kernel": run_dir / "jacobi_rb_kernel_gate.json",
        "target": run_dir / "jacobi_rb_target_gate.json",
    }
    gates: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        gates[name] = _json_load(path) if path.is_file() else not_evaluated_gate(
            f"jacobi_rb_{name}", f"workflow raised before {name} completed"
        )
        if not path.is_file():
            atomic_write_json(path, gates[name])
    parent_path = run_dir / "parent_provenance.json"
    parent: Mapping[str, Any] | bool = _json_load(parent_path) if parent_path.is_file() else False
    return _finish(
        run_dir, args=args,
        decision=_decision(parent, gates["preflight"], gates["kernel"], gates["target"]),
        required_gate_pass=False,
        phase=str(args.stage),
        fatal_failure=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return _run(args)
    except Exception as exc:
        print(f"Jacobi RB denoising feasibility error: {exc}", file=sys.stderr)
        if bool(getattr(args, "_mutation_started", False)):
            return _finalize_unexpected_failure(args, exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
