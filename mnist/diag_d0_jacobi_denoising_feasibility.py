"""Exact Jacobi-split Eulerian denoising feasibility gate.

This additive Experiment 12 workflow validates an exact fixed-grid Jacobi
reference and the latent denoising target ``Z=L-MY``.  It never trains on
physical MNIST states and never imports or runs a reverse sampler.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import linalg, stats
import torch

from mnist.d0_jacobi_ancestral import (
    ANCESTRAL_SAMPLER_VERSION,
    AncestralSamplerConfig,
    JacobiCertificationError,
    sample_ancestral_count,
    sample_wright_fisher_latent,
)
from mnist.d0_jacobi_denoising import (
    JACOBI_ALPHA1_SPECTRAL_VERSION,
    JACOBI_DENOISING_SCHEMA,
    JACOBI_ORIENTATION,
    JACOBI_STRANG_VERSION,
    Alpha1SpectralConfig,
    SpectralConvergenceError,
    apply_matching_head_fractions,
    build_four_color_matchings,
    evaluate_alpha1_spectral,
    evaluate_alpha1_spectral_torch_fixed_modes,
    jacobi_latent_label,
    jacobi_phase_exposure,
    linear_teacher_arrival_score,
    linear_teacher_denoising_mean,
    palindromic_strang_plan,
    validate_four_color_matchings,
)
from mnist.d0_jacobi_feasibility_gate import (
    JacobiFeasibilityThresholds,
    decide_jacobi_feasibility,
    evaluate_jacobi_controls,
    evaluate_jacobi_kernel,
    evaluate_jacobi_preflight,
    not_evaluated_gate,
)
from mnist.d0_jacobi_provenance import (
    PARENT_REGISTRY_SHA256,
    verify_and_readjudicate_gradient_parent,
)
from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_csv,
    atomic_write_json,
    config_fingerprint,
    configure_exact_torch_backend,
    file_fingerprint,
    source_fingerprint,
)


RUN_SCHEMA = "experiment12-d0-jacobi-denoising-feasibility"
RUN_SCHEMA_VERSION = 1
CLAIM_SCOPE = "fixed-grid exact Jacobi kernel, split, and denoising identity only"

DEFAULTS: dict[str, Any] = {
    "grid_size": 28,
    "alpha_eff": 1.0,
    "sample_steps": 512,
    "tau_eff": 5e-5,
    "root_seed": 261101,
    "quadrature_nodes": 192,
    "kernel_mc_draws": 256,
    "benchmark_draws": 32,
    "cache_paths": 64,
    "spectral_absolute_tolerance": 1e-12,
    "spectral_relative_tolerance": 1e-10,
    "spectral_max_modes": 4096,
    "ancestral_decimal_precision": 80,
    "ancestral_max_count": 4096,
    "ancestral_max_terms": 100_000,
    "ancestral_max_refinements": 20_000,
    "maximum_projected_cache_hours": 24.0,
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
    if value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "kernel", "controls", "report", "all"), default="all")
    parser.add_argument("--require-gate", choices=("none", "preflight", "kernel", "controls"), default="none")
    parser.add_argument("--parent-gradient-control-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument(
        "--runs-root", type=Path,
        default=Path("runs/experiment12_d0_jacobi_denoising_feasibility"),
    )
    parser.add_argument("--run-name", default="production-exact-jacobi-feasibility")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-progress", action="store_true")
    for name in (
        "grid_size", "sample_steps", "root_seed", "quadrature_nodes",
        "kernel_mc_draws", "benchmark_draws", "cache_paths", "spectral_max_modes",
        "ancestral_decimal_precision", "ancestral_max_count", "ancestral_max_terms",
        "ancestral_max_refinements",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=DEFAULTS[name])
    for name in (
        "alpha_eff", "tau_eff", "spectral_absolute_tolerance",
        "spectral_relative_tolerance", "maximum_projected_cache_hours",
        "maximum_memory_fraction",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=DEFAULTS[name])
    args = parser.parse_args(argv)
    if args.stage in {"kernel", "controls", "report"} and args.resume_run_dir is None:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    allowed = {
        "preflight": {"none", "preflight"},
        "kernel": {"none", "preflight", "kernel"},
        "controls": {"none", "preflight", "kernel", "controls"},
        "report": {"none", "preflight", "kernel", "controls"},
        "all": {"none", "preflight", "kernel", "controls"},
    }
    if args.require_gate not in allowed[args.stage]:
        parser.error(f"--require-gate {args.require_gate} is unavailable at stage {args.stage}")
    for name in (
        "grid_size", "sample_steps", "quadrature_nodes", "kernel_mc_draws",
        "benchmark_draws", "cache_paths", "spectral_max_modes",
        "ancestral_decimal_precision", "ancestral_max_count", "ancestral_max_terms",
        "ancestral_max_refinements",
    ):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if int(args.grid_size) % 2:
        parser.error("--grid-size must be even")
    if float(args.alpha_eff) != 1.0:
        parser.error("this first fixed-grid spectral gate requires --alpha-eff 1")
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


def _source_record() -> tuple[str, list[str]]:
    modules = [
        sys.modules[__name__],
        sys.modules[verify_and_readjudicate_gradient_parent.__module__],
        sys.modules[evaluate_alpha1_spectral.__module__],
        sys.modules[sample_ancestral_count.__module__],
        sys.modules[evaluate_jacobi_kernel.__module__],
        sys.modules[configure_exact_torch_backend.__module__],
    ]
    paths = sorted({Path(module.__file__).resolve() for module in modules})
    return source_fingerprint(paths), [str(path) for path in paths]


def _runtime_record(device: torch.device) -> dict[str, Any]:
    value: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "scipy": __import__("scipy").__version__,
        "device": str(device),
        "cuda_available": int(torch.cuda.is_available()),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        value.update({
            "cuda_device_name": str(properties.name),
            "cuda_total_memory": int(properties.total_memory),
            "cuda_capability": [int(properties.major), int(properties.minor)],
        })
    return value


def _scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "algorithm": RUN_SCHEMA,
        "algorithm_version": RUN_SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "fixed_grid": {
            "grid_size": int(args.grid_size),
            "grid_spacing": 1.0 / int(args.grid_size),
            "alpha_eff": float(args.alpha_eff),
            "sample_steps": int(args.sample_steps),
            "tau_eff": float(args.tau_eff),
            "spatial_df_convergence_claimed": 0,
        },
        "kernel": {
            "orientation": JACOBI_ORIENTATION,
            "spectral_version": JACOBI_ALPHA1_SPECTRAL_VERSION,
            "ancestral_sampler_version": ANCESTRAL_SAMPLER_VERSION,
            "jacobi_to_standard_wf_time_factor": 2.0,
            "spectral_absolute_tolerance": float(args.spectral_absolute_tolerance),
            "spectral_relative_tolerance": float(args.spectral_relative_tolerance),
            "spectral_max_modes": int(args.spectral_max_modes),
            "ancestral_config": {
                "decimal_precision": int(args.ancestral_decimal_precision),
                "max_ancestral_count": int(args.ancestral_max_count),
                "max_terms": int(args.ancestral_max_terms),
                "max_refinements": int(args.ancestral_max_refinements),
            },
        },
        "split": {
            "version": JACOBI_STRANG_VERSION,
            "phase_matching_indices": [phase.matching_index for phase in palindromic_strang_plan()],
            "phase_duration_fractions": [phase.duration_fraction for phase in palindromic_strang_plan()],
        },
        "denoising_target": {
            "name": "jacobi-latent-conormal-score",
            "formula": "Z=L-MY",
            "population_identity": "E[Z|later,phase]=Y(1-Y)*d_Y log(p/nu)",
            "raw_euler_residual_used": 0,
            "classifier_target_used": 0,
            "gaussian_proxy_used": 0,
        },
        "workload": {
            "cache_paths": int(args.cache_paths),
            "kernel_mc_draws": int(args.kernel_mc_draws),
            "benchmark_draws": int(args.benchmark_draws),
            "maximum_projected_cache_hours": float(args.maximum_projected_cache_hours),
            "maximum_memory_fraction": float(args.maximum_memory_fraction),
        },
        "root_seed": int(args.root_seed),
        "parent_registry_sha256": PARENT_REGISTRY_SHA256,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


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
        "sampling_performed": 0,
    }


def _write_diagnostic_plot(run_dir: Path) -> None:
    """Write a compact, non-authorizing visualization of committed metrics.

    Plotting is deliberately downstream of the numeric artifacts: an absent
    plotting dependency can never turn a mathematical pass into a pass, nor
    hide a kernel failure.  The status artifact makes that distinction
    explicit.
    """

    status_path = run_dir / "jacobi_plot_status.json"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
        support_path = run_dir / "production_support_certification.csv"
        if support_path.is_file():
            with support_path.open("r", encoding="utf-8", newline="") as handle:
                support = list(csv.DictReader(handle))
            exposures = np.asarray([float(row["jacobi_exposure"]) for row in support])
            workloads = np.asarray([
                float(row["ancestral_count_workload_estimate"]) for row in support
            ])
            certified = np.asarray([int(row["certified"]) for row in support], dtype=bool)
            axes[0].scatter(exposures[certified], workloads[certified], label="certified", marker="o")
            axes[0].scatter(exposures[~certified], workloads[~certified], label="fail closed", marker="x")
            axes[0].set_xscale("log")
            axes[0].set_yscale("log")
            axes[0].set_xlabel("Jacobi exposure $u$")
            axes[0].set_ylabel("ancestral-count workload estimate")
            axes[0].legend()
        else:
            axes[0].text(0.5, 0.5, "kernel not evaluated", ha="center", va="center")
            axes[0].set_axis_off()

        refinement_path = run_dir / "strang_refinement_metrics.csv"
        if refinement_path.is_file():
            with refinement_path.open("r", encoding="utf-8", newline="") as handle:
                refinement = [
                    row for row in csv.DictReader(handle)
                    if row.get("sample_steps") not in {None, ""}
                ]
            steps = np.asarray([int(row["sample_steps"]) for row in refinement])
            errors = np.asarray([float(row["relative_error"]) for row in refinement])
            axes[1].loglog(steps, errors, marker="o")
            axes[1].set_xlabel("split steps $K$")
            axes[1].set_ylabel("relative weak error")
        else:
            axes[1].text(0.5, 0.5, "split controls not evaluated", ha="center", va="center")
            axes[1].set_axis_off()

        output = run_dir / "jacobi_feasibility_summary.png"
        temporary = run_dir / ".jacobi_feasibility_summary.tmp.png"
        figure.savefig(temporary, dpi=160)
        plt.close(figure)
        temporary.replace(output)
        atomic_write_json(status_path, {
            "schema": "d0-jacobi-feasibility-plot-status",
            "schema_version": 1,
            "status": "written",
            "path": output.name,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        })
    except Exception as exc:  # plotting is evidence presentation, never gate evidence
        atomic_write_json(status_path, {
            "schema": "d0-jacobi-feasibility-plot-status",
            "schema_version": 1,
            "status": "not_written",
            "reason": f"{type(exc).__name__}: {exc}",
            "physical_training_performed": 0,
            "sampling_performed": 0,
        })


def _verify_terminal_registry(run_dir: Path) -> None:
    registry_path = run_dir / "artifact_registry.json"
    status_path = run_dir / "run_status.json"
    if not registry_path.is_file():
        return
    registry = _json_load(registry_path)
    status = _json_load(status_path)
    if status.get("artifact_registry_sha256") != file_fingerprint(registry_path):
        raise ArtifactCompatibilityError("resume status does not bind its artifact registry")
    for relative, record in dict(registry.get("records", {})).items():
        path = run_dir / relative
        if (
            not path.is_file()
            or record.get("sha256") != file_fingerprint(path)
            or int(record.get("size", -1)) != int(path.stat().st_size)
        ):
            raise ArtifactCompatibilityError(f"resume artifact changed: {relative}")


def _matching_records(grid_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matchings = build_four_color_matchings(grid_size)
    validate_four_color_matchings(grid_size, matchings)
    matching_rows = [{
        "matching_index": matching.index,
        "name": matching.name,
        "direction": matching.direction,
        "parity": matching.parity,
        "edge_count": matching.edge_count,
        "tails_sha256": config_fingerprint(matching.tails.tolist()),
        "heads_sha256": config_fingerprint(matching.heads.tolist()),
        "flux_indices_sha256": config_fingerprint(matching.flux_indices.tolist()),
    } for matching in matchings]
    phase_rows = [{
        "phase_index": phase.phase_index,
        "matching_index": phase.matching_index,
        "matching_name": phase.matching_name,
        "duration_fraction": phase.duration_fraction,
    } for phase in palindromic_strang_plan()]
    return matching_rows, phase_rows


def _run_preflight(
    args: argparse.Namespace,
    run_dir: Path,
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    matching_rows, phase_rows = _matching_records(int(args.grid_size))
    matching_indices = [int(row["matching_index"]) for row in phase_rows]
    phase_fractions = [float(row["duration_fraction"]) for row in phase_rows]
    convention = {
        "schema": "d0-jacobi-kernel-convention",
        "schema_version": 1,
        "orientation": "tail-to-head edge; x and Y are head fractions",
        "pair_total": "r=s_tail+s_head",
        "dimensionless_exposure": "u=(2alpha+1)*integral(a dt)/(alpha*h^2*r)",
        "standard_wf_time": "t_JS=2u",
        "latent_transition": "M~q^(2alpha)(2u), L~Bin(M,x), Y~Beta(alpha+L,alpha+M-L)",
        "denoising_label": "Z=L-MY",
        "conditional_identity": "E[Z|later,phase]=Y(1-Y)*d_Y log(p/nu)",
        "head_directed_flux": "J=2(2alpha+1)a(t)E[Z|state,phase]/(alpha*h^2)",
        "limiter_used": 0,
        "mass_floor_used": 0,
        "projection_used": 0,
        "target_clipping_used": 0,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    atomic_write_json(run_dir / "parent_readjudication.json", parent)
    atomic_write_json(run_dir / "matching_plan.json", {
        "schema": "d0-jacobi-four-color-matchings",
        "schema_version": 1,
        "grid_size": int(args.grid_size),
        "matchings": matching_rows,
        "phases": phase_rows,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    })
    atomic_write_csv(run_dir / "matching_plan.csv", [*matching_rows, *phase_rows])
    atomic_write_json(run_dir / "jacobi_kernel_convention.json", convention)
    matching_partition_valid = sum(int(row["edge_count"]) for row in matching_rows) == 2 * int(args.grid_size) ** 2
    matching_disjoint = all(int(row["edge_count"]) == int(args.grid_size) ** 2 // 2 for row in matching_rows)
    strang_palindromic = matching_indices == list(reversed(matching_indices)) and all(
        math.isclose(sum(
            phase_fractions[index] for index, value in enumerate(matching_indices) if value == matching
        ), 1.0, rel_tol=0.0, abs_tol=0.0)
        for matching in range(4)
    )
    gate = evaluate_jacobi_preflight(
        parent_registry_valid=True,
        parent_record_count=int(parent["parent_artifact_record_count"]),
        parent_registry_sha256=str(parent["parent_artifact_registry_sha256"]),
        expected_registry_sha256=PARENT_REGISTRY_SHA256,
        parent_readjudication_valid=bool(parent.get("readjudication_valid")),
        matching_partition_valid=matching_partition_valid,
        matching_disjoint=matching_disjoint,
        strang_palindromic=strang_palindromic,
        convention_valid=(
            float(args.alpha_eff) == 1.0
            and convention["denoising_label"] == "Z=L-MY"
            and convention["standard_wf_time"] == "t_JS=2u"
        ),
    )
    atomic_write_json(run_dir / "jacobi_preflight_gate.json", gate)
    return gate


def _spectral_kernel_metrics(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = Alpha1SpectralConfig(
        absolute_tolerance=float(args.spectral_absolute_tolerance),
        relative_tolerance=float(args.spectral_relative_tolerance),
        max_modes=int(args.spectral_max_modes),
    )
    nodes, weights = np.polynomial.legendre.leggauss(int(args.quadrature_nodes))
    z = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    cases = ((0.27, 0.18), (0.51, 0.31), (0.79, 0.67))
    normalization_errors: list[float] = []
    eigen_errors: list[float] = []
    maximum_modes = 0
    production_support_pass = True
    for x, exposure in cases:
        evaluation = evaluate_alpha1_spectral(x, z, exposure, config=config)
        maximum_modes = max(maximum_modes, evaluation.diagnostics.modes_used)
        normalization_errors.append(abs(float(np.dot(w, evaluation.density)) - 1.0))
        zy = 2.0 * z - 1.0
        for degree in (1, 2, 3):
            coefficients = np.zeros(degree + 1)
            coefficients[-1] = 1.0
            polynomial = np.polynomial.legendre.legval(zy, coefficients)
            observed = float(np.dot(w, evaluation.density * polynomial))
            expected = math.exp(-degree * (degree + 1) * exposure) * float(
                np.polynomial.legendre.legval(2.0 * x - 1.0, coefficients)
            )
            eigen_errors.append(abs(observed - expected) / max(1.0, abs(expected)))

    h = 1.0 / int(args.grid_size)
    macro_schedule = float(args.tau_eff) / int(args.sample_steps)
    support_rows: list[dict[str, Any]] = []
    for duration_fraction in (0.5, 1.0):
        for pair_total in (1.0, 0.25, 0.1, 0.025, 2.0 / int(args.grid_size) ** 2, 1e-3, 1e-5):
            exposure_value = float(jacobi_phase_exposure(
                pair_total,
                macro_schedule * duration_fraction,
                alpha=1.0,
                grid_spacing=h,
            ))
            x_value = 0.37
            y_value = float(np.clip(x_value + 0.25 * math.sqrt(exposure_value), 1e-8, 1.0 - 1e-8))
            try:
                support_evaluation = evaluate_alpha1_spectral(
                    x_value, y_value, exposure_value, config=config
                )
                maximum_modes = max(maximum_modes, support_evaluation.diagnostics.modes_used)
                passed = bool(
                    support_evaluation.diagnostics.converged
                    and math.isfinite(float(support_evaluation.density))
                    and math.isfinite(float(support_evaluation.arrival_score))
                    and float(support_evaluation.density) > 0.0
                )
                support_rows.append({
                    "case": "production_support",
                    "pair_total": pair_total,
                    "duration_fraction": duration_fraction,
                    "exposure": exposure_value,
                    "x": x_value,
                    "y": y_value,
                    "density": float(support_evaluation.density),
                    "arrival_score": float(support_evaluation.arrival_score),
                    "modes_used": support_evaluation.diagnostics.modes_used,
                    "density_tail_bound": support_evaluation.diagnostics.max_density_tail_bound,
                    "score_error_bound": support_evaluation.diagnostics.max_arrival_score_error_bound,
                    "certified": int(passed),
                })
                production_support_pass = production_support_pass and passed
            except SpectralConvergenceError as exc:
                production_support_pass = False
                support_rows.append({
                    "case": "production_support",
                    "pair_total": pair_total,
                    "duration_fraction": duration_fraction,
                    "exposure": exposure_value,
                    "x": x_value,
                    "y": y_value,
                    "certified": 0,
                    "failure": str(exc),
                })

    endpoint = evaluate_alpha1_spectral(
        np.array([0.31, 0.31]), np.array([0.0, 1.0]), 0.22, config=config
    )
    cdf_error = float(np.max(np.abs(endpoint.cdf - np.array([0.0, 1.0]))))
    xy = evaluate_alpha1_spectral(0.27, 0.73, 0.23, config=config)
    yx = evaluate_alpha1_spectral(0.73, 0.27, 0.23, config=config)
    detailed_error = abs(float(xy.density) - float(yx.density)) / max(1.0, abs(float(xy.density)))
    first = evaluate_alpha1_spectral(0.27, z, 0.11, config=config)
    second = evaluate_alpha1_spectral(z, 0.73, 0.16, config=config)
    composed = float(np.dot(w, first.density * second.density))
    direct = float(evaluate_alpha1_spectral(0.27, 0.73, 0.27, config=config).density)
    semigroup_error = abs(composed - direct) / max(1.0, abs(direct))

    x, y, exposure = 0.31, 0.58, 0.22
    step = 2e-5
    offsets = np.array([-2.0, -1.0, 1.0, 2.0]) * step
    logs = np.log(evaluate_alpha1_spectral(x, y + offsets, exposure, config=config).density)
    finite_difference = float((logs[0] - 8.0 * logs[1] + 8.0 * logs[2] - logs[3]) / (12.0 * step))
    center = evaluate_alpha1_spectral(x, y, exposure, config=config)
    score_error = abs(finite_difference - float(center.arrival_score)) / max(1.0, abs(float(center.arrival_score)))

    smallest_exposure = float(jacobi_phase_exposure(
        1.0, macro_schedule * 0.5, alpha=1.0, grid_spacing=h
    ))
    typical_exposure = float(jacobi_phase_exposure(
        2.0 / int(args.grid_size) ** 2,
        macro_schedule * 0.5,
        alpha=1.0,
        grid_spacing=h,
    ))
    torch_x = torch.tensor([0.37, 0.43, 0.81], dtype=torch.float64, device=device)
    torch_y = torch.tensor([
        0.37 + 0.25 * math.sqrt(smallest_exposure),
        0.43 + 0.25 * math.sqrt(typical_exposure),
        0.64,
    ], dtype=torch.float64, device=device)
    torch_u = torch.tensor([smallest_exposure, typical_exposure, 0.91], dtype=torch.float64, device=device)
    numpy_eval = evaluate_alpha1_spectral(
        torch_x.cpu().numpy(), torch_y.cpu().numpy(), torch_u.cpu().numpy(), config=config
    )
    modes = int(numpy_eval.diagnostics.modes_used)
    torch_eval = evaluate_alpha1_spectral_torch_fixed_modes(torch_x, torch_y, torch_u, modes=modes)
    cuda_relative_error = max(
        float(np.max(np.abs(torch_eval.density.detach().cpu().numpy() - numpy_eval.density) / np.maximum(1.0, np.abs(numpy_eval.density)))),
        float(np.max(np.abs(torch_eval.cdf.detach().cpu().numpy() - numpy_eval.cdf))),
    )
    cuda_score_error = float(np.max(
        np.abs(torch_eval.arrival_score.detach().cpu().numpy() - numpy_eval.arrival_score)
        / np.maximum(1.0, np.abs(numpy_eval.arrival_score))
    ))
    records = [{
        "x": x_value,
        "exposure": u_value,
        "normalization_error": error,
    } for (x_value, u_value), error in zip(cases, normalization_errors, strict=True)]
    records.extend(support_rows)
    return {
        "normalization_relative_error": max(normalization_errors),
        "cdf_endpoint_error": cdf_error,
        "detailed_balance_relative_error": detailed_error,
        "semigroup_relative_error": semigroup_error,
        "eigenmoment_relative_error": max(eigen_errors),
        "arrival_score_relative_error": score_error,
        "cuda_relative_error": cuda_relative_error,
        "cuda_score_relative_error": cuda_score_error,
        "maximum_modes_used": maximum_modes,
        "production_spectral_support_pass": int(production_support_pass),
        "negative_density_count": 0,
        "nonfinite_count": 0,
        "comparison_device": str(device),
    }, records


def _sample_linear_initial(rng: np.random.Generator, count: int, amplitude: float = 0.5) -> np.ndarray:
    # Rejection from Uniform using v0(x)=1+c(2x-1), bounded by 1+|c|.
    values: list[float] = []
    while len(values) < count:
        proposal = rng.random(max(16, 2 * (count - len(values))))
        accept = rng.random(proposal.size) * (1.0 + abs(amplitude)) <= 1.0 + amplitude * (2.0 * proposal - 1.0)
        values.extend(proposal[accept].tolist())
    return np.asarray(values[:count], dtype=np.float64)


def _ancestral_kernel_metrics(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = AncestralSamplerConfig(
        decimal_precision=int(args.ancestral_decimal_precision),
        max_ancestral_count=int(args.ancestral_max_count),
        max_terms=int(args.ancestral_max_terms),
        max_refinements=int(args.ancestral_max_refinements),
    )
    h = 1.0 / int(args.grid_size)
    macro_schedule = float(args.tau_eff) / int(args.sample_steps)
    cell_count = int(args.grid_size) ** 2
    pair_totals = (
        ("simplex_maximum", 1.0),
        ("boundary_stress", 0.25),
        ("upper_support", 0.1),
        ("moderate_support", 0.025),
        ("dirichlet_mean", 2.0 / cell_count),
        ("small_pair", 1e-3),
        ("very_small_pair", 1e-5),
    )
    support_records: list[dict[str, Any]] = []
    resource_cap_count = 0
    numerical_certificate_failure_count = 0
    support_not_evaluated_count = 0
    numerical_abort = False
    for duration_fraction in (0.5, 1.0):
        for support_name, pair_total in pair_totals:
            exposure = float(jacobi_phase_exposure(
                pair_total,
                macro_schedule * duration_fraction,
                alpha=1.0,
                grid_spacing=h,
            ))
            workload_estimate = int(math.ceil(1.0 / exposure))
            record: dict[str, Any] = {
                "support_name": support_name,
                "pair_total": pair_total,
                "duration_fraction": duration_fraction,
                "jacobi_exposure": exposure,
                "standard_wf_time": 2.0 * exposure,
                "ancestral_count_workload_estimate": workload_estimate,
                "configured_ancestral_count_cap": int(cfg.max_ancestral_count),
            }
            if numerical_abort:
                support_not_evaluated_count += 1
                record.update({
                    "certified": 0,
                    "failure_kind": "not_evaluated_after_numerical_failure",
                    "failure": "earlier support point invalidated the sampler certificate",
                })
                support_records.append(record)
                continue
            started = time.perf_counter()
            try:
                draw = sample_ancestral_count(
                    jacobi_time=exposure,
                    alpha=1.0,
                    uniform=0.37123456789,
                    config=cfg,
                )
                record.update({
                    "certified": 1,
                    "ancestral_count": draw.count,
                    "terms_evaluated": draw.terms_evaluated,
                    "refinements": draw.refinements,
                    "bracket_width": draw.bracket_width,
                })
            except JacobiCertificationError as exc:
                failure_kind = str(exc.diagnostics.get("failure_kind", "numerical_certificate"))
                if failure_kind == "resource_cap":
                    resource_cap_count += 1
                else:
                    numerical_certificate_failure_count += 1
                    numerical_abort = True
                record.update({
                    "certified": 0,
                    "failure": str(exc),
                    "failure_kind": failure_kind,
                    **exc.diagnostics,
                })
            record["elapsed_seconds"] = time.perf_counter() - started
            support_records.append(record)

    rng = np.random.default_rng(int(args.root_seed) + 11)
    benchmark_records: list[dict[str, Any]] = []
    start = time.perf_counter()
    benchmark_certified = 0
    benchmark_terms: list[int] = []
    representative_pair_total = 2.0 / cell_count
    representative_exposure = float(jacobi_phase_exposure(
        representative_pair_total,
        macro_schedule * 0.5,
        alpha=1.0,
        grid_spacing=h,
    ))
    benchmark_numerical_failures = 0
    benchmark_resource_failures = 0
    for index in range(int(args.benchmark_draws)):
        try:
            draw = sample_wright_fisher_latent(
                head_fraction=0.37,
                jacobi_time=representative_exposure,
                alpha=1.0,
                rng=rng,
                config=cfg,
            )
            benchmark_certified += 1
            benchmark_terms.append(draw.ancestral.terms_evaluated)
            benchmark_records.append({
                "draw_index": index,
                "certified": 1,
                "M": draw.ancestral_count,
                "L": draw.head_count,
                "Y": draw.later_head_fraction,
                "Z": draw.denoising_label,
                "terms_evaluated": draw.ancestral.terms_evaluated,
            })
        except JacobiCertificationError as exc:
            kind = str(exc.diagnostics.get("failure_kind", "numerical_certificate"))
            if kind == "resource_cap":
                benchmark_resource_failures += 1
            else:
                benchmark_numerical_failures += 1
            benchmark_records.append({
                "draw_index": index,
                "certified": 0,
                "failure_kind": kind,
                "failure": str(exc),
            })
    elapsed = time.perf_counter() - start
    throughput = benchmark_certified / elapsed if elapsed > 0.0 else 0.0

    distribution_rng = np.random.default_rng(int(args.root_seed) + 12)
    values: list[float] = []
    distribution_certified = True
    for _ in range(int(args.kernel_mc_draws)):
        try:
            draw = sample_wright_fisher_latent(
                head_fraction=0.3,
                jacobi_time=representative_exposure,
                alpha=1.0,
                rng=distribution_rng,
                config=cfg,
            )
            values.append(draw.later_head_fraction)
        except JacobiCertificationError:
            distribution_certified = False
            break
    distribution_pass = False
    distribution_record: dict[str, Any] = {
        "draw_count": len(values),
        "requested_draw_count": int(args.kernel_mc_draws),
        "all_draws_certified": int(distribution_certified and len(values) == int(args.kernel_mc_draws)),
    }
    if distribution_record["all_draws_certified"]:
        samples = np.asarray(values)
        expected_mean = 0.5 + (0.3 - 0.5) * math.exp(-2.0 * representative_exposure)
        observed_mean = float(np.mean(samples))
        standard_error = float(np.std(samples, ddof=1) / math.sqrt(samples.size))

        def cdf(value: Any) -> Any:
            result = evaluate_alpha1_spectral(0.3, value, representative_exposure).cdf
            return float(result) if np.ndim(result) == 0 else np.asarray(result)

        ks = stats.kstest(samples, cdf)
        mean_z = abs(observed_mean - expected_mean) / max(standard_error, 1e-15)
        quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(256)
        quadrature_y = 0.5 * (quadrature_nodes + 1.0)
        quadrature_w = 0.5 * quadrature_weights
        quadrature_density = evaluate_alpha1_spectral(
            0.3, quadrature_y, representative_exposure
        ).density
        expected_moments = np.asarray([
            float(np.dot(quadrature_w, quadrature_density * quadrature_y**degree))
            for degree in (1, 2, 3)
        ])
        cdf_grid = np.linspace(0.01, 0.99, 257)
        cdf_grid_values = np.asarray(evaluate_alpha1_spectral(
            0.3, cdf_grid, representative_exposure
        ).cdf)
        target_probabilities = np.asarray([0.2, 0.4, 0.6, 0.8])
        selected_indices = [
            int(np.argmin(np.abs(cdf_grid_values - probability)))
            for probability in target_probabilities
        ]
        cdf_points = cdf_grid[selected_indices]
        expected_cdf = cdf_grid_values[selected_indices]
        differences = np.column_stack([
            *(samples**degree - expected_moments[degree - 1] for degree in (1, 2, 3)),
            *((samples <= point).astype(np.float64) - expected_cdf[index]
              for index, point in enumerate(cdf_points)),
        ])
        simultaneous_pass, simultaneous_intervals = _simultaneous_mean_intervals(
            differences,
            seed=int(args.root_seed) + 13,
        )
        distribution_pass = bool(simultaneous_pass)
        distribution_record.update({
            "observed_mean": observed_mean,
            "expected_mean": expected_mean,
            "mean_standard_error": standard_error,
            "mean_z_score": mean_z,
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "simultaneous_confidence": 0.99,
            "simultaneous_family_size": int(differences.shape[1]),
            "simultaneous_intervals": simultaneous_intervals,
            "passed": int(distribution_pass),
        })

    transitions = int(args.cache_paths) * int(args.sample_steps) * 7 * (int(args.grid_size) ** 2 // 2)
    projected_hours = transitions / throughput / 3600.0 if throughput > 0.0 else math.inf
    # The implementation is streaming and retains one matching at a time.
    estimated_bytes = int(args.cache_paths) * (int(args.grid_size) ** 2 // 2) * 8 * 8
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        total_memory = int(torch.cuda.get_device_properties(index).total_memory)
    else:
        total_memory = max(estimated_bytes, 8 * 1024**3)
    memory_fraction = estimated_bytes / total_memory
    metrics = {
        "uncertified_draw_count": int(
            resource_cap_count
            + numerical_certificate_failure_count
            + support_not_evaluated_count
            + benchmark_resource_failures
            + benchmark_numerical_failures
        ),
        "resource_cap_count": int(resource_cap_count + benchmark_resource_failures),
        "numerical_certification_failure_count": int(
            numerical_certificate_failure_count + benchmark_numerical_failures
        ),
        "support_not_evaluated_count": int(support_not_evaluated_count),
        "distribution_control_pass": int(distribution_pass),
        "benchmark_elapsed_seconds": elapsed,
        "certified_draws_per_second": throughput,
        "projected_transition_count": transitions,
        "projected_cache_hours": projected_hours,
        "peak_memory_fraction": memory_fraction,
        "maximum_benchmark_terms": max(benchmark_terms) if benchmark_terms else None,
        "production_support_certified": int(
            resource_cap_count == 0 and numerical_certificate_failure_count == 0
        ),
        "benchmark_jacobi_exposure": representative_exposure,
        "benchmark_pair_total": representative_pair_total,
        "benchmark_duration_fraction": 0.5,
        "benchmark_implementation_device": "cpu-mpmath-arbitrary-precision",
        "requested_orchestration_device": str(device),
        "correction_count": 0,
        "floor_count": 0,
        "limiter_count": 0,
        "renormalization_count": 0,
        "sampler_version": ANCESTRAL_SAMPLER_VERSION,
        "distribution": distribution_record,
    }
    return metrics, support_records, benchmark_records


def _run_kernel(args: argparse.Namespace, run_dir: Path, device: torch.device) -> dict[str, Any]:
    spectral_metrics: dict[str, Any]
    spectral_rows: list[dict[str, Any]]
    try:
        spectral_metrics, spectral_rows = _spectral_kernel_metrics(args, device)
    except (SpectralConvergenceError, FloatingPointError, ValueError) as exc:
        diagnostics = getattr(exc, "diagnostics", None)
        spectral_metrics = {
            "normalization_relative_error": math.inf,
            "cdf_endpoint_error": math.inf,
            "detailed_balance_relative_error": math.inf,
            "semigroup_relative_error": math.inf,
            "eigenmoment_relative_error": math.inf,
            "arrival_score_relative_error": math.inf,
            "cuda_relative_error": math.inf,
            "cuda_score_relative_error": math.inf,
            "negative_density_count": int(getattr(diagnostics, "negative_density_count", 0)),
            "nonfinite_count": int(getattr(diagnostics, "nonfinite_count", 1)),
            "spectral_failure": str(exc),
        }
        spectral_rows = []
    ancestral_metrics, support_rows, benchmark_rows = _ancestral_kernel_metrics(args, device)
    metrics = {**spectral_metrics, **ancestral_metrics}
    thresholds = JacobiFeasibilityThresholds(
        maximum_projected_cache_hours=float(args.maximum_projected_cache_hours),
        maximum_memory_fraction=float(args.maximum_memory_fraction),
    )
    gate = evaluate_jacobi_kernel(metrics, thresholds)
    atomic_write_json(run_dir / "jacobi_kernel_metrics.json", {
        "schema": "d0-jacobi-kernel-metrics",
        "schema_version": 1,
        "metrics": metrics,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    })
    atomic_write_csv(run_dir / "spectral_kernel_metrics.csv", spectral_rows)
    atomic_write_csv(run_dir / "production_support_certification.csv", support_rows)
    atomic_write_csv(run_dir / "ancestral_benchmark_draws.csv", benchmark_rows)
    distribution = metrics.get("distribution", {})
    interval_rows = [
        {"family": "moment-and-cdf", **dict(record)}
        for record in distribution.get("simultaneous_intervals", [])
    ] if isinstance(distribution, Mapping) else []
    atomic_write_csv(run_dir / "spectral_mixture_distribution_control.csv", interval_rows)
    atomic_write_json(run_dir / "jacobi_kernel_gate.json", gate)
    _write_diagnostic_plot(run_dir)
    return gate


def _strang_matrix_refinement() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # This noncommuting matrix fixture checks only the generic seven-stage
    # composition algebra.  It is deliberately advisory: it cannot replace a
    # finite-horizon refinement of the state-dependent Eulerian Jacobi phases,
    # whose crossing colors change each pair total.  Until the exact small-time
    # kernel is fast enough for that rollout, the authorizing refinement flag
    # remains false.
    edges = ((0, 1), (2, 3), (1, 2), (3, 0))
    generators: list[np.ndarray] = []
    rates = (1.0, 1.7, 0.8, 1.3)
    for (left, right), rate in zip(edges, rates, strict=True):
        matrix = np.zeros((4, 4), dtype=np.float64)
        matrix[left, right] = rate
        matrix[right, left] = 0.7 * rate
        matrix[left, left] = -rate
        matrix[right, right] = -0.7 * rate
        generators.append(matrix)
    horizon = 2.0
    total = sum(generators)
    reference = linalg.expm(horizon * total)
    observables = np.asarray([
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 4.0, 9.0],
        [0.0, 1.0, 8.0, 27.0],
    ]).T
    initial = np.asarray([0.43, 0.17, 0.29, 0.11])
    expected = initial @ reference @ observables
    plan = palindromic_strang_plan()
    rows: list[dict[str, Any]] = []
    errors: dict[int, float] = {}
    values: dict[int, np.ndarray] = {}
    for steps in (128, 256, 512, 1024):
        dt = horizon / steps
        macro = np.eye(4)
        for phase in plan:
            macro = macro @ linalg.expm(
                dt * float(phase.duration_fraction) * generators[phase.matching_index]
            )
        result = initial @ np.linalg.matrix_power(macro, steps) @ observables
        error = float(np.linalg.norm(result - expected) / max(np.linalg.norm(expected), 1e-15))
        errors[steps] = error
        values[steps] = result
        rows.append({
            "fixture": "generic_palindrome_algebra_smoke",
            "authorizing": 0,
            "sample_steps": steps,
            "linear_value": result[0],
            "quadratic_value": result[1],
            "cubic_value": result[2],
            "relative_error": error,
        })
    pairs = [(128, 256), (256, 512), (512, 1024)]
    orders = [math.log(errors[a] / errors[b], 2.0) for a, b in pairs if errors[a] > 0 and errors[b] > 0]
    observed_order = float(np.median(orders)) if orders else 0.0
    discrepancy = float(np.linalg.norm(values[512] - values[1024]) / max(np.linalg.norm(values[1024]), 1e-15))
    return {
        "advisory_observed_weak_order": observed_order,
        "advisory_k512_k1024_relative_error": discrepancy,
        "advisory_k512_generator_relative_error": errors[512],
        "fixture": "advisory noncommuting-matrix composition algebra",
        "split_reference_evaluated": 0,
        "split_fixture": "not_evaluated",
        "actual_eulerian_refinement_pass": 0,
        "authorizing": 0,
        "not_evaluated_reason": (
            "state-dependent exact Jacobi full-horizon refinement requires a "
            "production-feasible small-time ancestral kernel"
        ),
    }, rows


def _edge_generator_observable_fixture() -> tuple[float, list[dict[str, Any]]]:
    """Check actual grid-28 color generators on linear/quadratic/cubic observables.

    Exact degree-three Jacobi eigenmoments give each one-phase expectation;
    Richardson-extrapolated physical-time differences are compared with the
    state-dependent ``1/r`` Eulerian generator.  This is a local consistency
    check only, not the missing full-horizon split-refinement experiment.
    """

    grid_size = 28
    count = grid_size * grid_size
    h = 1.0 / grid_size
    coordinates = np.arange(count, dtype=np.float64)
    state = 1.0 + 0.22 * np.cos(2.0 * math.pi * coordinates / count)
    state += 0.13 * np.sin(6.0 * math.pi * coordinates / count)
    state /= np.sum(state)
    weights = np.cos(4.0 * math.pi * coordinates / count)
    rows: list[dict[str, Any]] = []
    errors: list[float] = []

    for matching in build_four_color_matchings(grid_size):
        tail = state[matching.tails]
        head = state[matching.heads]
        pair_total = tail + head
        x = head / pair_total
        minimum_pair = float(np.min(pair_total))
        maximum_exposure = 5e-5
        physical_step = maximum_exposure * h * h * minimum_pair / 3.0

        def exact_rates(step: float) -> np.ndarray:
            exposure = 3.0 * step / (h * h * pair_total)
            delta_m1 = (x - 0.5) * np.expm1(-2.0 * exposure)
            p2 = 6.0 * x * x - 6.0 * x + 1.0
            delta_p2 = p2 * np.expm1(-6.0 * exposure)
            delta_m2 = (delta_p2 + 6.0 * delta_m1) / 6.0
            linear_increment = np.sum(
                pair_total
                * (weights[matching.heads] - weights[matching.tails])
                * delta_m1
            )
            quadratic_increment = np.sum(
                pair_total**2 * (-2.0 * delta_m1 + 2.0 * delta_m2)
            ) / count
            cubic_increment = np.sum(
                pair_total**3 * (-3.0 * delta_m1 + 3.0 * delta_m2)
            ) / count
            return np.asarray([
                linear_increment, quadratic_increment, cubic_increment
            ]) / step

        full = exact_rates(physical_step)
        half = exact_rates(0.5 * physical_step)
        measured = 2.0 * half - full
        difference = head - tail
        directional = np.column_stack([
            weights[matching.heads] - weights[matching.tails],
            2.0 * difference / count,
            3.0 * (head * head - tail * tail) / count,
        ])
        directional_second = np.column_stack([
            np.zeros_like(pair_total),
            np.full_like(pair_total, 4.0 / count),
            6.0 * pair_total / count,
        ])
        analytic = 3.0 / (h * h) * np.sum(
            (tail * head / pair_total)[:, None] * directional_second
            + ((tail - head) / pair_total)[:, None] * directional,
            axis=0,
        )
        names = ("linear_fourier", "quadratic_mass", "cubic_mass")
        for index, name in enumerate(names):
            error = abs(float(measured[index] - analytic[index])) / max(
                1.0, abs(float(analytic[index]))
            )
            errors.append(error)
            rows.append({
                "fixture": "actual_grid28_color_generator_local_consistency",
                "authorizing": 0,
                "matching": matching.name,
                "observable": name,
                "analytic_generator": float(analytic[index]),
                "richardson_semigroup_generator": float(measured[index]),
                "relative_error": error,
                "maximum_phase_exposure": maximum_exposure,
            })
    return max(errors), rows


def _mean_interval_contains_zero(values: np.ndarray, confidence: float = 0.99) -> tuple[bool, float, float, float]:
    sample = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(sample))
    if sample.size < 2:
        return False, mean, math.nan, math.nan
    se = float(np.std(sample, ddof=1) / math.sqrt(sample.size))
    radius = float(stats.t.ppf(0.5 + confidence / 2.0, sample.size - 1)) * se
    return bool(mean - radius <= 0.0 <= mean + radius), mean, mean - radius, mean + radius


def _simultaneous_mean_intervals(
    values: np.ndarray,
    *,
    seed: int,
    confidence: float = 0.99,
    replicates: int = 10_000,
) -> tuple[bool, list[dict[str, Any]]]:
    """Studentized whole-path max-T intervals with fail-closed degeneracy.

    Rows are paths and columns are a jointly adjudicated family.  Resampling
    rows, rather than individual observables or edges, preserves every
    within-path dependency used by the stationarity and denoising controls.
    """

    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 2 or sample.shape[0] < 2 or sample.shape[1] < 1:
        return False, []
    if not np.all(np.isfinite(sample)):
        return False, []
    count, width = sample.shape
    means = np.mean(sample, axis=0)
    standard_errors = np.std(sample, axis=0, ddof=1) / math.sqrt(count)
    exact_zero = np.all(sample == 0.0, axis=0)
    invalid_degenerate = (standard_errors == 0.0) & ~exact_zero
    if bool(np.any(invalid_degenerate)):
        return False, [{
            "component": int(index),
            "mean": float(means[index]),
            "standard_error": float(standard_errors[index]),
            "lower": math.nan,
            "upper": math.nan,
            "contains_zero": 0,
            "degenerate_nonzero": int(invalid_degenerate[index]),
        } for index in range(width)]

    rng = np.random.default_rng(int(seed))
    maximum_statistics = np.zeros(int(replicates), dtype=np.float64)
    cursor = 0
    while cursor < int(replicates):
        batch = min(256, int(replicates) - cursor)
        indices = rng.integers(0, count, size=(batch, count))
        resampled = sample[indices]
        resampled_means = np.mean(resampled, axis=1)
        resampled_se = np.std(resampled, axis=1, ddof=1) / math.sqrt(count)
        numerator = np.abs(resampled_means - means[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            statistics = numerator / resampled_se
        statistics[:, exact_zero] = 0.0
        statistics[~np.isfinite(statistics)] = math.inf
        maximum_statistics[cursor:cursor + batch] = np.max(statistics, axis=1)
        cursor += batch
    critical = float(np.quantile(maximum_statistics, confidence, method="higher"))
    if not math.isfinite(critical):
        return False, []
    lower = means - critical * standard_errors
    upper = means + critical * standard_errors
    lower[exact_zero] = 0.0
    upper[exact_zero] = 0.0
    contains = (lower <= 0.0) & (upper >= 0.0)
    records = [{
        "component": int(index),
        "mean": float(means[index]),
        "standard_error": float(standard_errors[index]),
        "simultaneous_confidence": float(confidence),
        "critical_max_t": critical,
        "lower": float(lower[index]),
        "upper": float(upper[index]),
        "contains_zero": int(contains[index]),
        "degenerate_nonzero": 0,
    } for index in range(width)]
    return bool(np.all(contains)), records


def _denoising_identity_controls(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    nodes, weights = np.polynomial.legendre.leggauss(int(args.quadrature_nodes))
    x_nodes = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    config = Alpha1SpectralConfig(
        absolute_tolerance=float(args.spectral_absolute_tolerance),
        relative_tolerance=float(args.spectral_relative_tolerance),
        max_modes=int(args.spectral_max_modes),
    )
    deterministic_errors: list[float] = []
    rows: list[dict[str, Any]] = []
    for exposure in (0.15, 0.5, 1.0):
        for y in (0.17, 0.43, 0.79):
            kernel = evaluate_alpha1_spectral(x_nodes, y, exposure, config=config)
            initial_density = 1.0 + 0.5 * (2.0 * x_nodes - 1.0)
            marginal = float(np.dot(w, initial_density * kernel.density))
            score_numerator = float(np.dot(w, initial_density * kernel.density * kernel.arrival_score))
            observed_score = score_numerator / marginal
            expected_score = float(linear_teacher_arrival_score(y, exposure))
            expected_mean = float(linear_teacher_denoising_mean(y, exposure))
            observed_mean = y * (1.0 - y) * observed_score
            error = abs(observed_mean - expected_mean) / max(1.0, abs(expected_mean))
            deterministic_errors.append(error)
            rows.append({
                "exposure": exposure,
                "later_head_fraction": y,
                "observed_score": observed_score,
                "expected_score": expected_score,
                "observed_denoising_mean": observed_mean,
                "expected_denoising_mean": expected_mean,
                "relative_error": error,
            })

    cfg = AncestralSamplerConfig(
        decimal_precision=int(args.ancestral_decimal_precision),
        max_ancestral_count=int(args.ancestral_max_count),
        max_terms=int(args.ancestral_max_terms),
        max_refinements=int(args.ancestral_max_refinements),
    )
    count = max(128, int(args.kernel_mc_draws))
    rng = np.random.default_rng(int(args.root_seed) + 31)
    initial = _sample_linear_initial(rng, count)
    teacher_differences = np.empty((count, 3), dtype=np.float64)
    null_differences = np.empty((count, 3), dtype=np.float64)
    for index in range(count):
        teacher = sample_wright_fisher_latent(
            head_fraction=float(initial[index]), jacobi_time=0.5, alpha=1.0, rng=rng, config=cfg
        )
        y = teacher.later_head_fraction
        analytic = float(linear_teacher_denoising_mean(y, 0.5))
        basis = np.asarray([1.0, y, y * y])
        teacher_differences[index] = (teacher.denoising_label - analytic) * basis
        null = sample_wright_fisher_latent(
            head_fraction=float(rng.random()), jacobi_time=0.5, alpha=1.0, rng=rng, config=cfg
        )
        yn = null.later_head_fraction
        null_differences[index] = null.denoising_label * np.asarray([1.0, yn, yn * yn])
    teacher_pass, teacher_intervals = _simultaneous_mean_intervals(
        teacher_differences, seed=int(args.root_seed) + 32
    )
    null_pass, null_intervals = _simultaneous_mean_intervals(
        null_differences, seed=int(args.root_seed) + 33
    )
    for task, intervals in (("teacher", teacher_intervals), ("null", null_intervals)):
        for basis_index, interval in enumerate(intervals):
            rows.append({
                "control": task,
                "basis_degree": basis_index,
                "mean_difference": interval["mean"],
                "lower_99_simultaneous": interval["lower"],
                "upper_99_simultaneous": interval["upper"],
                "critical_max_t": interval["critical_max_t"],
                "contains_zero": interval["contains_zero"],
            })
    canonical_z = float(jacobi_latent_label(
        np.array([3]), np.array([2]), np.array([0.4])
    )[0])
    reversed_z = float(jacobi_latent_label(
        np.array([3]), np.array([1]), np.array([0.6])
    )[0])
    correct_pair_exposure = float(jacobi_phase_exposure(
        0.2, 1e-4, alpha=1.0, grid_spacing=1.0 / 28.0
    ))
    omitted_pair_exposure = float(jacobi_phase_exposure(
        1.0, 1e-4, alpha=1.0, grid_spacing=1.0 / 28.0
    ))
    fine_h_exposure = float(jacobi_phase_exposure(
        0.2, 1e-4, alpha=1.0, grid_spacing=1.0 / 28.0
    ))
    coarse_h_exposure = float(jacobi_phase_exposure(
        0.2, 1e-4, alpha=1.0, grid_spacing=2.0 / 28.0
    ))
    auxiliary_y = 0.3
    alpha_two_invariant_score = (2.0 - 1.0) * (
        1.0 / auxiliary_y - 1.0 / (1.0 - auxiliary_y)
    )
    fixture_passes = {
        "head_orientation": abs(canonical_z - 0.8) <= 3e-16 and abs(reversed_z + canonical_z) <= 3e-16,
        "pair_mass_exposure": abs(correct_pair_exposure / omitted_pair_exposure - 5.0) <= 1e-12,
        "h_scaling": abs(fine_h_exposure / coarse_h_exposure - 4.0) <= 1e-12,
        "invariant_beta_not_lebesgue": abs(alpha_two_invariant_score) > 1.0,
        "teacher_sign": float(linear_teacher_arrival_score(0.4, 0.5)) > 0.0,
    }
    for name, passed in fixture_passes.items():
        rows.append({
            "control": "negative_fixture",
            "fixture": name,
            "passed": int(passed),
        })
    return {
        "deterministic_identity_error": max(deterministic_errors),
        "monte_carlo_identity_pass": int(teacher_pass),
        "stationary_null_pass": int(null_pass),
        "orientation_fixtures_pass": int(all(fixture_passes.values())),
    }, rows


def _control_observables(states: np.ndarray, grid_size: int) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    count = grid_size * grid_size
    coordinates = np.arange(count, dtype=np.float64)
    angle = 2.0 * math.pi * coordinates / count
    horizontal = np.roll(values.reshape(values.shape[0], grid_size, grid_size), -1, axis=2)
    vertical = np.roll(values.reshape(values.shape[0], grid_size, grid_size), -1, axis=1)
    image = values.reshape(values.shape[0], grid_size, grid_size)
    return np.column_stack([
        np.sum(values**2, axis=1),
        np.sum(values**3, axis=1),
        values @ np.cos(angle),
        values @ np.sin(angle),
        np.sum(image * horizontal, axis=(1, 2)),
        np.sum(image * vertical, axis=(1, 2)),
    ])


def _apply_exact_control_phases(
    states: np.ndarray,
    phases: Sequence[Any],
    *,
    matchings: Sequence[Any],
    base_clock: float,
    rng: np.random.Generator,
    config: AncestralSamplerConfig,
) -> tuple[np.ndarray, int]:
    out = np.asarray(states, dtype=np.float64).copy()
    failures = 0
    for phase in phases:
        matching = matchings[int(phase.matching_index)]
        pair_totals = out[:, matching.tails] + out[:, matching.heads]
        fractions = out[:, matching.heads] / pair_totals
        later = np.empty_like(fractions)
        for path_index, edge_index in np.ndindex(fractions.shape):
            exposure = base_clock * float(phase.duration_fraction) / float(
                pair_totals[path_index, edge_index]
            )
            try:
                later[path_index, edge_index] = sample_wright_fisher_latent(
                    head_fraction=float(fractions[path_index, edge_index]),
                    jacobi_time=exposure,
                    alpha=1.0,
                    rng=rng,
                    config=config,
                ).later_head_fraction
            except JacobiCertificationError:
                failures += 1
                return out, failures
        out = apply_matching_head_fractions(out, matching, later)
    return out, failures


def _stationarity_detailed_balance_controls(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    # A four-by-four torus is the smallest nondegenerate instance containing
    # all four production color classes.  The exact conditional phase law is
    # unchanged; the enlarged control clock keeps ancestral draws certifiable.
    grid_size = 4
    path_count = min(64, max(32, int(args.kernel_mc_draws) // 4))
    matchings = build_four_color_matchings(grid_size)
    phases = palindromic_strang_plan()
    cfg = AncestralSamplerConfig(
        decimal_precision=int(args.ancestral_decimal_precision),
        max_ancestral_count=int(args.ancestral_max_count),
        max_terms=int(args.ancestral_max_terms),
        max_refinements=int(args.ancestral_max_refinements),
    )
    rng = np.random.default_rng(int(args.root_seed) + 51)
    sweep_initial = rng.dirichlet(np.ones(grid_size * grid_size), size=path_count)
    sweep_later, sweep_failures = _apply_exact_control_phases(
        sweep_initial,
        phases,
        matchings=matchings,
        base_clock=0.05,
        rng=rng,
        config=cfg,
    )
    phase_initial = rng.dirichlet(np.ones(grid_size * grid_size), size=path_count)
    phase_later, phase_failures = _apply_exact_control_phases(
        phase_initial,
        (phases[0],),
        matchings=matchings,
        base_clock=0.10,
        rng=rng,
        config=cfg,
    )
    records: list[dict[str, Any]] = []
    if sweep_failures or phase_failures:
        return {
            "dirichlet_stationarity_pass": 0,
            "full_sweep_detailed_balance_pass": 0,
            "stationarity_certificate_failure_count": sweep_failures + phase_failures,
        }, records

    families: list[tuple[str, np.ndarray]] = []
    for name, initial, later in (
        ("full_sweep", sweep_initial, sweep_later),
        ("single_phase", phase_initial, phase_later),
    ):
        initial_obs = _control_observables(initial, grid_size)
        later_obs = _control_observables(later, grid_size)
        families.append((name + "_stationarity", later_obs - initial_obs))
        partner_initial = np.roll(initial_obs, 1, axis=1)
        partner_later = np.roll(later_obs, 1, axis=1)
        detailed_balance = initial_obs * partner_later - later_obs * partner_initial
        families.append((name + "_detailed_balance", detailed_balance))

    passes: dict[str, bool] = {}
    for family_index, (name, differences) in enumerate(families):
        passed, intervals = _simultaneous_mean_intervals(
            differences,
            seed=int(args.root_seed) + 52 + family_index,
        )
        passes[name] = passed
        for interval in intervals:
            records.append({"family": name, **interval})
    return {
        "dirichlet_stationarity_pass": int(
            passes.get("full_sweep_stationarity", False)
            and passes.get("single_phase_stationarity", False)
        ),
        "full_sweep_detailed_balance_pass": int(
            passes.get("full_sweep_detailed_balance", False)
            and passes.get("single_phase_detailed_balance", False)
        ),
        "stationarity_certificate_failure_count": 0,
        "stationarity_path_count": path_count,
        "stationarity_control_grid_size": grid_size,
        "stationarity_inference": "whole-path studentized max-T, simultaneous 99%",
    }, records


def _run_controls(args: argparse.Namespace, run_dir: Path, device: torch.device) -> dict[str, Any]:
    rng = np.random.default_rng(int(args.root_seed) + 41)
    matchings = build_four_color_matchings(int(args.grid_size))
    states = rng.dirichlet(np.ones(int(args.grid_size) ** 2), size=4)
    maximum_pair_error = 0.0
    maximum_simplex_error = 0.0
    out = states.copy()
    for phase in palindromic_strang_plan():
        matching = matchings[phase.matching_index]
        before = out[:, matching.tails] + out[:, matching.heads]
        later = rng.random(before.shape)
        out = apply_matching_head_fractions(out, matching, later)
        after = out[:, matching.tails] + out[:, matching.heads]
        maximum_pair_error = max(maximum_pair_error, float(np.max(np.abs(after - before))))
        maximum_simplex_error = max(maximum_simplex_error, float(np.max(np.abs(out.sum(axis=1) - 1.0))))
    tensor = torch.from_numpy(states).to(device=device, dtype=torch.float64)
    torch_out = tensor.clone()
    torch_pair_error = 0.0
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.root_seed) + 42)
    for phase in palindromic_strang_plan():
        matching = matchings[phase.matching_index]
        tails = torch.as_tensor(matching.tails, dtype=torch.long, device=device)
        heads = torch.as_tensor(matching.heads, dtype=torch.long, device=device)
        before = torch_out[:, tails] + torch_out[:, heads]
        later = torch.rand(before.shape, dtype=torch.float64, device=device, generator=generator)
        updated = torch_out.clone()
        updated[:, tails] = before * (1.0 - later)
        updated[:, heads] = before * later
        after = updated[:, tails] + updated[:, heads]
        torch_pair_error = max(
            torch_pair_error,
            float(torch.max(torch.abs(after - before)).detach().cpu()),
        )
        torch_out = updated
    torch_simplex_error = float(torch.max(torch.abs(torch_out.sum(dim=1) - 1.0)).detach().cpu())
    split_metrics, refinement_rows = _strang_matrix_refinement()
    edge_generator_error, edge_generator_rows = _edge_generator_observable_fixture()
    refinement_rows.extend(edge_generator_rows)
    identity_metrics, identity_rows = _denoising_identity_controls(args)
    stationarity_metrics, stationarity_rows = _stationarity_detailed_balance_controls(args)
    metrics = {
        "float64_pair_mass_error": maximum_pair_error,
        "float64_simplex_error": maximum_simplex_error,
        "cuda_pair_mass_error": torch_pair_error,
        "cuda_simplex_error": torch_simplex_error,
        **stationarity_metrics,
        **split_metrics,
        "edge_generator_observable_error": edge_generator_error,
        **identity_metrics,
        "intervention_count": 0,
        "correction_count": 0,
        "floor_count": 0,
        "limiter_count": 0,
        "renormalization_count": 0,
        "nonfinite_count": 0,
    }
    gate = evaluate_jacobi_controls(metrics)
    atomic_write_json(run_dir / "jacobi_control_metrics.json", {
        "schema": "d0-jacobi-control-metrics",
        "schema_version": 1,
        "metrics": metrics,
        "stationarity_argument": (
            "each exact matching phase preserves the conditional Beta law; "
            "the palindrome and its reverse coincide, so the complete split is Dirichlet-reversible"
        ),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    })
    atomic_write_csv(run_dir / "strang_refinement_metrics.csv", refinement_rows)
    atomic_write_csv(run_dir / "denoising_identity_metrics.csv", identity_rows)
    atomic_write_csv(run_dir / "stationarity_detailed_balance_metrics.csv", stationarity_rows)
    atomic_write_json(run_dir / "jacobi_control_gate.json", gate)
    _write_diagnostic_plot(run_dir)
    return gate


def _decision_record(
    *,
    preflight: Mapping[str, Any],
    kernel: Mapping[str, Any] | None,
    controls: Mapping[str, Any] | None,
    parent: Mapping[str, Any],
    interim: str | None = None,
) -> dict[str, Any]:
    if interim is not None:
        return {
            "schema": "d0-jacobi-feasibility-decision",
            "schema_version": 1,
            "decision": interim,
            "closed_terminal_scientific_outcome": 0,
            "one_image_training_authorized": 0,
            "physical_training_performed": 0,
            "sampling_authorized": 0,
            "sampling_performed": 0,
        }
    return decide_jacobi_feasibility(
        provenance_valid=True,
        adjudication_valid=bool(parent.get("readjudication_valid")),
        preflight_gate=preflight,
        kernel_gate=kernel,
        controls_gate=controls,
    )


def _requested_gate_pass(
    require_gate: str,
    *,
    preflight: Mapping[str, Any] | None,
    kernel: Mapping[str, Any] | None,
    controls: Mapping[str, Any] | None,
) -> bool:
    if require_gate == "none":
        return True
    selected = {
        "preflight": preflight,
        "kernel": kernel,
        "controls": controls,
    }[str(require_gate)]
    return selected is not None and int(selected.get("passed", 0)) == 1


def _finish(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    decision: Mapping[str, Any],
    required_gate_pass: bool,
    phase: str,
) -> int:
    atomic_write_json(run_dir / "jacobi_feasibility_decision.json", decision)
    registry = _artifact_registry(run_dir)
    atomic_write_json(run_dir / "artifact_registry.json", registry)
    registry_path = run_dir / "artifact_registry.json"
    required_failure = args.require_gate != "none" and not required_gate_pass
    _write_status(
        run_dir,
        status="complete",
        phase=phase,
        stage=args.stage,
        require_gate=args.require_gate,
        required_gate=args.require_gate,
        required_gate_pass=int(required_gate_pass),
        outcome="gate_failed" if required_failure else "complete",
        decision=decision.get("decision"),
        one_image_training_authorized=int(decision.get("one_image_training_authorized", 0)),
        artifact_registry_sha256=file_fingerprint(registry_path),
        artifact_registry_size=int(registry_path.stat().st_size),
    )
    return 2 if required_failure else 0


def _run(args: argparse.Namespace) -> int:
    run_dir, resumed = _make_run_dir(args)
    args._active_run_dir = run_dir
    args._mutation_started = False
    print(f"Jacobi denoising feasibility run directory: {run_dir}", flush=True)
    if resumed:
        _verify_terminal_registry(run_dir)
    device = _device(args.device)
    backend = configure_exact_torch_backend(device)
    scientific = _scientific_config(args)
    source_hash, source_paths = _source_record()
    runtime = _runtime_record(device)
    existing_manifest = (
        _json_load(run_dir / "run_manifest.json")
        if (run_dir / "run_manifest.json").is_file()
        else None
    )
    try:
        parent = verify_and_readjudicate_gradient_parent(args.parent_gradient_control_run_dir)
    except ArtifactCompatibilityError as exc:
        # A required gate must fail only after readable evidence exists.  This
        # run is intentionally non-resumable after repairing its parent: the
        # frozen manifest records the failed binding, so a fresh run is needed.
        failure = {
            "schema": "d0-jacobi-parent-readjudication",
            "schema_version": 1,
            "evaluation_status": "invalid",
            "parent_run_dir": str(Path(args.parent_gradient_control_run_dir).resolve()),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "readjudication_valid": 0,
            "physical_training_performed": 0,
            "sampling_performed": 0,
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
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        _freeze_json(run_dir / "run_manifest.json", manifest)
        atomic_write_json(run_dir / "parent_readjudication.json", failure)
        preflight = not_evaluated_gate("jacobi_preflight", "immutable parent verification failed")
        atomic_write_json(run_dir / "jacobi_preflight_gate.json", preflight)
        atomic_write_json(run_dir / "jacobi_kernel_gate.json", not_evaluated_gate(
            "jacobi_kernel", "immutable parent verification failed"
        ))
        atomic_write_json(run_dir / "jacobi_control_gate.json", not_evaluated_gate(
            "jacobi_controls", "immutable parent verification failed"
        ))
        _write_status(
            run_dir,
            status="running",
            phase="preflight",
            stage=args.stage,
            require_gate=args.require_gate,
            attempt_count=1,
        )
        decision = decide_jacobi_feasibility(
            provenance_valid=False,
            adjudication_valid=False,
            preflight_gate=preflight,
            kernel_gate=None,
            controls_gate=None,
        )
        return _finish(
            run_dir,
            args=args,
            decision=decision,
            required_gate_pass=_requested_gate_pass(
                args.require_gate, preflight=preflight, kernel=None, controls=None
            ),
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
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    _freeze_json(run_dir / "run_manifest.json", manifest)
    current = _json_load(run_dir / "run_status.json") if (run_dir / "run_status.json").is_file() else {}
    _write_status(
        run_dir,
        status="running",
        phase=args.stage,
        stage=args.stage,
        require_gate=args.require_gate,
        attempt_count=int(current.get("attempt_count", 0)) + 1,
    )
    args._mutation_started = True

    preflight_path = run_dir / "jacobi_preflight_gate.json"
    kernel_path = run_dir / "jacobi_kernel_gate.json"
    controls_path = run_dir / "jacobi_control_gate.json"

    if args.stage == "report":
        if not preflight_path.is_file():
            raise ArtifactCompatibilityError("report requires completed preflight evidence")
        preflight = _json_load(preflight_path)
        kernel = _json_load(kernel_path) if kernel_path.is_file() else None
        controls = _json_load(controls_path) if controls_path.is_file() else None
        decision = _decision_record(preflight=preflight, kernel=kernel, controls=controls, parent=parent)
        required = {
            "none": True,
            "preflight": int(preflight.get("passed", 0)) == 1,
            "kernel": kernel is not None and int(kernel.get("passed", 0)) == 1,
            "controls": controls is not None and int(controls.get("passed", 0)) == 1,
        }[args.require_gate]
        return _finish(run_dir, args=args, decision=decision, required_gate_pass=required, phase="report")

    if args.stage in {"preflight", "all"}:
        preflight = _run_preflight(args, run_dir, parent)
    else:
        if not preflight_path.is_file():
            raise ArtifactCompatibilityError("kernel/controls stage requires preflight evidence")
        preflight = _json_load(preflight_path)
    if int(preflight.get("passed", 0)) != 1:
        decision = _decision_record(preflight=preflight, kernel=None, controls=None, parent=parent)
        return _finish(
            run_dir,
            args=args,
            decision=decision,
            required_gate_pass=_requested_gate_pass(
                args.require_gate, preflight=preflight, kernel=None, controls=None
            ),
            phase="preflight",
        )
    if args.stage == "preflight":
        return _finish(
            run_dir, args=args,
            decision=_decision_record(preflight=preflight, kernel=None, controls=None, parent=parent, interim="preflight_passed"),
            required_gate_pass=True,
            phase="preflight",
        )

    if args.stage in {"kernel", "all"}:
        kernel = _run_kernel(args, run_dir, device)
    else:
        if not kernel_path.is_file():
            raise ArtifactCompatibilityError("controls stage requires kernel evidence")
        kernel = _json_load(kernel_path)
    if int(kernel.get("passed", 0)) != 1:
        atomic_write_json(run_dir / "jacobi_control_gate.json", not_evaluated_gate(
            "jacobi_controls", "exact production-support kernel gate failed"
        ))
        decision = _decision_record(preflight=preflight, kernel=kernel, controls=None, parent=parent)
        return _finish(
            run_dir,
            args=args,
            decision=decision,
            required_gate_pass=_requested_gate_pass(
                args.require_gate, preflight=preflight, kernel=kernel, controls=None
            ),
            phase="kernel",
        )
    if args.stage == "kernel":
        return _finish(
            run_dir, args=args,
            decision=_decision_record(preflight=preflight, kernel=kernel, controls=None, parent=parent, interim="kernel_passed"),
            required_gate_pass=True,
            phase="kernel",
        )

    controls = _run_controls(args, run_dir, device)
    decision = _decision_record(preflight=preflight, kernel=kernel, controls=controls, parent=parent)
    return _finish(
        run_dir,
        args=args,
        decision=decision,
        required_gate_pass=_requested_gate_pass(
            args.require_gate, preflight=preflight, kernel=kernel, controls=controls
        ),
        phase="controls",
    )


def _finalize_unexpected_failure(args: argparse.Namespace, exc: BaseException) -> int:
    run_dir = Path(args._active_run_dir)
    atomic_write_json(run_dir / "unexpected_failure.json", {
        "schema": "d0-jacobi-feasibility-unexpected-failure",
        "schema_version": 1,
        "stage": args.stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    })
    preflight_path = run_dir / "jacobi_preflight_gate.json"
    kernel_path = run_dir / "jacobi_kernel_gate.json"
    controls_path = run_dir / "jacobi_control_gate.json"
    preflight = _json_load(preflight_path) if preflight_path.is_file() else not_evaluated_gate(
        "jacobi_preflight", "workflow raised before preflight completed"
    )
    if not preflight_path.is_file():
        atomic_write_json(preflight_path, preflight)
    kernel = _json_load(kernel_path) if kernel_path.is_file() else not_evaluated_gate(
        "jacobi_kernel", "workflow raised before kernel completed"
    )
    if not kernel_path.is_file():
        atomic_write_json(kernel_path, kernel)
    controls = _json_load(controls_path) if controls_path.is_file() else not_evaluated_gate(
        "jacobi_controls", "workflow raised before controls completed"
    )
    if not controls_path.is_file():
        atomic_write_json(controls_path, controls)
    parent_path = run_dir / "parent_readjudication.json"
    parent = _json_load(parent_path) if parent_path.is_file() else {"readjudication_valid": 0}
    decision = decide_jacobi_feasibility(
        provenance_valid=parent_path.is_file(),
        adjudication_valid=bool(parent.get("readjudication_valid", 0)),
        preflight_gate=preflight,
        kernel_gate=kernel,
        controls_gate=controls,
    )
    return _finish(
        run_dir,
        args=args,
        decision=decision,
        required_gate_pass=_requested_gate_pass(
            args.require_gate,
            preflight=preflight,
            kernel=kernel,
            controls=controls,
        ),
        phase=str(args.stage),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return _run(args)
    except (ArtifactCompatibilityError, RuntimeError, ValueError) as exc:
        print(f"Jacobi denoising feasibility error: {exc}", file=sys.stderr)
        if bool(getattr(args, "_mutation_started", False)):
            return _finalize_unexpected_failure(args, exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
