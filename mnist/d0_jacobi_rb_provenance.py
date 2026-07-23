"""Strict parent binding for the spectral Rao--Blackwell Jacobi repair.

The admissible parent is the immutable grid-28 Jacobi feasibility run which
passed its algebraic and spectral-law checks but could not produce ancestral
counts at production-small exposures within the frozen resource budget.  The
verifier below checks the exact 16-record registry and reconstructs that
limited conclusion.  It does not modify or reinterpret any parent file.

The additive repair may replace the *representation* of the label by the
exact conditional expectation ``E[L-MY | X,Y]``.  It is not authorised to
replace the transition law or the DDPM-like population target.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_provenance import verify_and_readjudicate_gradient_parent


PARENT_RUN_SCHEMA = "experiment12-d0-jacobi-denoising-feasibility"
PARENT_REGISTRY_SCHEMA = PARENT_RUN_SCHEMA + "-artifact-registry"
PARENT_REGISTRY_RECORD_COUNT = 16
PARENT_REGISTRY_SHA256 = (
    "fd8695f463e1bd82f5d0be16059ab1be41fd9dbf727603264600881d5addc836"
)
PARENT_RAW_DECISION = "jacobi_kernel_numerically_unresolved"
READJUDICATED_DECISION = "ancestral_representation_infeasible"
EXPECTED_TRANSITION_COUNT = 89_915_392
EXPECTED_PROJECTED_CACHE_HOURS = 5080.94209321022
EXPECTED_LINEAGE_REGISTRY_RECORD_COUNTS = (
    277,
    301,
    263,
    123,
    125,
    332,
    222,
    381,
)

_REQUIRED_FILES = {
    "manifest": "run_manifest.json",
    "status": "run_status.json",
    "registry": "artifact_registry.json",
    "decision": "jacobi_feasibility_decision.json",
    "preflight": "jacobi_preflight_gate.json",
    "kernel_gate": "jacobi_kernel_gate.json",
    "kernel_metrics": "jacobi_kernel_metrics.json",
    "controls": "jacobi_control_gate.json",
    "parent_readjudication": "parent_readjudication.json",
}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"cannot read Jacobi feasibility parent artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(
            f"Jacobi feasibility parent artifact is not an object: {path}"
        )
    return dict(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactCompatibilityError(message)


def _one(value: Any) -> bool:
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return False


def _zero(value: Any) -> bool:
    try:
        return int(value) == 0
    except (TypeError, ValueError):
        return False


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _same_float(actual: Any, expected: float, tolerance: float = 1.0e-12) -> bool:
    return _finite(actual) and math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=float(tolerance)
    )


def _no_work(record: Mapping[str, Any], description: str) -> None:
    _require(
        _zero(record.get("physical_training_performed", 0))
        and _zero(record.get("sampling_performed", 0)),
        f"{description} records physical training or sampling",
    )


def _verify_registry(
    run_dir: Path,
    *,
    registry_path: Path,
    status: Mapping[str, Any],
) -> dict[str, Any]:
    registry = _load(registry_path)
    digest = file_fingerprint(registry_path)
    _require(
        digest == PARENT_REGISTRY_SHA256,
        "parent is not the frozen 16-record Jacobi feasibility run",
    )
    _require(
        registry.get("schema") == PARENT_REGISTRY_SCHEMA
        and int(registry.get("schema_version", -1)) == 1,
        "Jacobi feasibility registry schema is incompatible",
    )
    _require(
        status.get("artifact_registry_sha256") == digest
        and int(status.get("artifact_registry_size", -1))
        == int(registry_path.stat().st_size),
        "Jacobi feasibility status does not bind its registry",
    )
    exclusions = set(
        registry.get("terminal_files_excluded_to_avoid_self_reference", [])
    )
    _require(
        exclusions == {"artifact_registry.json", "run_status.json"},
        "Jacobi feasibility registry exclusions changed",
    )
    raw_records = registry.get("records", {})
    _require(isinstance(raw_records, Mapping), "parent registry records are invalid")
    records = dict(raw_records)
    _require(
        len(records) == PARENT_REGISTRY_RECORD_COUNT,
        f"parent registry must contain exactly {PARENT_REGISTRY_RECORD_COUNT} records",
    )
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.relative_to(run_dir).as_posix() not in exclusions
    }
    _require(
        actual == set(records),
        "Jacobi feasibility registry is incomplete or contains stale records",
    )
    for relative, raw in records.items():
        _require(isinstance(raw, Mapping), f"invalid registry record: {relative}")
        artifact = run_dir / relative
        _require(
            artifact.is_file()
            and raw.get("sha256") == file_fingerprint(artifact)
            and int(raw.get("size", -1)) == int(artifact.stat().st_size),
            f"registered Jacobi feasibility artifact changed: {relative}",
        )
    _no_work(registry, "Jacobi feasibility registry")
    return registry


def _verify_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        manifest.get("schema") == PARENT_RUN_SCHEMA
        and int(manifest.get("schema_version", -1)) == 1,
        "Jacobi feasibility manifest schema is incompatible",
    )
    scientific_raw = manifest.get("scientific_config", {})
    _require(
        isinstance(scientific_raw, Mapping),
        "Jacobi feasibility scientific configuration is missing",
    )
    scientific = dict(scientific_raw)
    _require(
        config_fingerprint(scientific) == manifest.get("scientific_fingerprint"),
        "Jacobi feasibility scientific fingerprint is inconsistent",
    )
    source_paths = [Path(str(value)) for value in manifest.get("source_paths", [])]
    _require(
        bool(source_paths) and all(path.is_file() for path in source_paths),
        "Jacobi feasibility source paths are unavailable",
    )
    _require(
        source_fingerprint(source_paths) == manifest.get("source_fingerprint"),
        "Jacobi feasibility parent source fingerprint changed",
    )

    fixed = scientific.get("fixed_grid", {})
    kernel = scientific.get("kernel", {})
    target = scientific.get("denoising_target", {})
    workload = scientific.get("workload", {})
    _require(
        isinstance(fixed, Mapping)
        and int(fixed.get("grid_size", -1)) == 28
        and int(fixed.get("sample_steps", -1)) == 512
        and _same_float(fixed.get("tau_eff"), 5.0e-5)
        and _same_float(fixed.get("alpha_eff"), 1.0)
        and _zero(fixed.get("spatial_df_convergence_claimed")),
        "Jacobi feasibility fixed-grid contract changed",
    )
    _require(
        isinstance(kernel, Mapping)
        and kernel.get("orientation") == "head-fraction"
        and _same_float(kernel.get("jacobi_to_standard_wf_time_factor"), 2.0)
        and kernel.get("spectral_version") == "alpha1-legendre-certified-v1"
        and kernel.get("ancestral_sampler_version")
        == "jenkins-spano-alternating-series-algorithm2-v1",
        "Jacobi feasibility kernel convention changed",
    )
    _require(
        isinstance(target, Mapping)
        and target.get("formula") == "Z=L-MY"
        and target.get("population_identity")
        == "E[Z|later,phase]=Y(1-Y)*d_Y log(p/nu)"
        and _zero(target.get("classifier_target_used"))
        and _zero(target.get("gaussian_proxy_used"))
        and _zero(target.get("raw_euler_residual_used")),
        "Jacobi feasibility DDPM-like target contract changed",
    )
    _require(
        isinstance(workload, Mapping)
        and int(workload.get("cache_paths", -1)) == 64
        and _same_float(workload.get("maximum_projected_cache_hours"), 24.0),
        "Jacobi feasibility workload contract changed",
    )
    _no_work(manifest, "Jacobi feasibility manifest")
    _no_work(scientific, "Jacobi feasibility scientific configuration")
    return scientific


def _verify_status_and_decision(
    status: Mapping[str, Any], decision: Mapping[str, Any]
) -> None:
    _require(
        status.get("schema") == PARENT_RUN_SCHEMA
        and int(status.get("schema_version", -1)) == 1
        and status.get("status") == "complete"
        and status.get("outcome") == "gate_failed"
        and status.get("phase") == "kernel"
        and status.get("stage") == "kernel"
        and status.get("require_gate") == "kernel"
        and status.get("required_gate") == "kernel"
        and _zero(status.get("required_gate_pass"))
        and status.get("decision") == PARENT_RAW_DECISION,
        "Jacobi feasibility terminal status changed",
    )
    _require(
        decision.get("schema") == "d0-jacobi-feasibility-decision"
        and int(decision.get("schema_version", -1)) == 1
        and decision.get("decision") == PARENT_RAW_DECISION
        and _one(decision.get("closed_terminal_scientific_outcome"))
        and _zero(decision.get("one_image_training_authorized"))
        and _zero(decision.get("sampling_authorized")),
        "Jacobi feasibility decision changed",
    )
    _no_work(status, "Jacobi feasibility status")
    _no_work(decision, "Jacobi feasibility decision")


def _verify_parent_evidence(
    *,
    preflight: Mapping[str, Any],
    kernel_gate: Mapping[str, Any],
    kernel_metrics: Mapping[str, Any],
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        preflight.get("evaluation_status") == "evaluated"
        and _one(preflight.get("passed")),
        "Jacobi feasibility preflight did not pass",
    )
    _require(
        kernel_gate.get("evaluation_status") == "evaluated"
        and _zero(kernel_gate.get("passed"))
        and _zero(kernel_gate.get("numerically_valid"))
        and _zero(kernel_gate.get("computationally_feasible")),
        "Jacobi feasibility kernel outcome changed",
    )
    raw_checks = kernel_gate.get("subchecks", {})
    _require(isinstance(raw_checks, Mapping), "Jacobi kernel subchecks are invalid")
    checks = dict(raw_checks)
    failed = {
        str(name)
        for name, raw in checks.items()
        if not isinstance(raw, Mapping) or not _one(raw.get("passed"))
    }
    expected_failed = {
        "numerical_certification_failure_count",
        "projected_cache_hours",
        "resource_cap_count",
        "uncertified_draw_count",
    }
    _require(
        failed == expected_failed,
        f"Jacobi feasibility failure set changed: {sorted(failed)}",
    )
    raw_metrics = kernel_metrics.get("metrics", {})
    _require(isinstance(raw_metrics, Mapping), "Jacobi kernel metrics are invalid")
    metrics = dict(raw_metrics)
    _require(
        _one(metrics.get("production_spectral_support_pass"))
        and _one(metrics.get("distribution_control_pass"))
        and int(metrics.get("projected_transition_count", -1))
        == EXPECTED_TRANSITION_COUNT
        and _same_float(
            metrics.get("projected_cache_hours"),
            EXPECTED_PROJECTED_CACHE_HOURS,
            tolerance=1.0e-9,
        )
        and int(metrics.get("resource_cap_count", -1)) == 1
        and int(metrics.get("numerical_certification_failure_count", -1)) == 1
        and int(metrics.get("uncertified_draw_count", -1)) == 14
        and int(metrics.get("maximum_modes_used", -1)) == 568
        and metrics.get("benchmark_implementation_device")
        == "cpu-mpmath-arbitrary-precision"
        and _zero(metrics.get("correction_count"))
        and _zero(metrics.get("floor_count"))
        and _zero(metrics.get("limiter_count"))
        and _zero(metrics.get("renormalization_count"))
        and _zero(metrics.get("nonfinite_count")),
        "Jacobi kernel failure is not isolated to the ancestral representation",
    )
    _require(
        controls.get("evaluation_status") == "not_evaluated"
        and _zero(controls.get("passed"))
        and controls.get("reason") == "exact production-support kernel gate failed"
        and not controls.get("subchecks"),
        "Jacobi controls were unexpectedly evaluated",
    )
    for description, record in (
        ("Jacobi preflight", preflight),
        ("Jacobi kernel gate", kernel_gate),
        ("Jacobi kernel metrics", kernel_metrics),
        ("Jacobi controls", controls),
    ):
        _no_work(record, description)
    return {
        "preflight_pass": 1,
        "spectral_support_pass": 1,
        "distribution_control_pass": 1,
        "projected_transition_count": EXPECTED_TRANSITION_COUNT,
        "projected_cache_hours": EXPECTED_PROJECTED_CACHE_HOURS,
        "resource_cap_count": 1,
        "numerical_certification_failure_count": 1,
        "uncertified_draw_count": 14,
        "failed_subchecks": sorted(expected_failed),
        "controls_evaluation_status": "not_evaluated",
    }


def verify_and_readjudicate_jacobi_parent(run_dir: str | Path) -> dict[str, Any]:
    """Verify the frozen parent and isolate its ancestral-sampler failure."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"Jacobi feasibility parent does not exist: {root}")
    paths = {name: root / relative for name, relative in _REQUIRED_FILES.items()}
    for name, path in paths.items():
        _require(path.is_file(), f"Jacobi feasibility parent lacks {name}: {path.name}")

    status = _load(paths["status"])
    registry = _verify_registry(
        root,
        registry_path=paths["registry"],
        status=status,
    )
    manifest = _load(paths["manifest"])
    scientific = _verify_manifest(manifest)
    decision = _load(paths["decision"])
    _verify_status_and_decision(status, decision)

    basis = _verify_parent_evidence(
        preflight=_load(paths["preflight"]),
        kernel_gate=_load(paths["kernel_gate"]),
        kernel_metrics=_load(paths["kernel_metrics"]),
        controls=_load(paths["controls"]),
    )

    stored_readjudication = _load(paths["parent_readjudication"])
    gradient_root = Path(str(stored_readjudication.get("parent_run_dir", "")))
    recomputed = verify_and_readjudicate_gradient_parent(gradient_root)
    _require(
        stored_readjudication == recomputed,
        "Jacobi feasibility transitive parent re-adjudication changed",
    )
    lineage = tuple(int(value) for value in recomputed.get("lineage_registry_record_counts", []))
    _require(
        lineage == EXPECTED_LINEAGE_REGISTRY_RECORD_COUNTS,
        f"Jacobi feasibility lineage changed: {list(lineage)}",
    )

    return {
        "schema": "d0-jacobi-rao-blackwell-parent-provenance",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": 1,
        "parent_run_dir": str(root),
        "parent_run_schema": PARENT_RUN_SCHEMA,
        "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
        "parent_artifact_registry_size": int(paths["registry"].stat().st_size),
        "parent_artifact_record_count": len(dict(registry["records"])),
        "parent_scientific_fingerprint": manifest.get("scientific_fingerprint"),
        "parent_source_fingerprint": manifest.get("source_fingerprint"),
        "lineage_registry_record_counts": [
            PARENT_REGISTRY_RECORD_COUNT,
            *lineage,
        ],
        "saved_decision": PARENT_RAW_DECISION,
        "readjudicated_decision": READJUDICATED_DECISION,
        "readjudication_basis": basis,
        "ddpm_population_target_preserved": 1,
        "rao_blackwell_followup_authorized": 1,
        "parent_mutated": 0,
        "physical_training_performed": 0,
        "sampling_performed": 0,
        "scientific_config": {
            "grid_size": scientific["fixed_grid"]["grid_size"],
            "sample_steps": scientific["fixed_grid"]["sample_steps"],
            "tau_eff": scientific["fixed_grid"]["tau_eff"],
            "alpha_eff": scientific["fixed_grid"]["alpha_eff"],
            "orientation": scientific["kernel"]["orientation"],
        },
    }


verify_parent_jacobi_feasibility_run = verify_and_readjudicate_jacobi_parent


__all__ = [
    "PARENT_RUN_SCHEMA",
    "PARENT_REGISTRY_SCHEMA",
    "PARENT_REGISTRY_RECORD_COUNT",
    "PARENT_REGISTRY_SHA256",
    "PARENT_RAW_DECISION",
    "READJUDICATED_DECISION",
    "EXPECTED_TRANSITION_COUNT",
    "EXPECTED_PROJECTED_CACHE_HOURS",
    "EXPECTED_LINEAGE_REGISTRY_RECORD_COUNTS",
    "verify_and_readjudicate_jacobi_parent",
    "verify_parent_jacobi_feasibility_run",
]
