"""Immutable provenance for the fused-CUDA Jacobi RB follow-up.

The only admissible control is the completed certified spectral run named by
``PARENT_RUN_BASENAME``.  That run established a numerically valid exact
kernel, but failed its frozen resource gate and never evaluated the target.
This module verifies that limited conclusion without changing any parent
artifact or authorising training or reverse sampling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_rb_gate import (
    evaluate_jacobi_rb_kernel,
    evaluate_jacobi_rb_preflight,
)
from mnist.d0_jacobi_rb_provenance import verify_and_readjudicate_jacobi_parent


PARENT_RUN_BASENAME = (
    "20260722-203339_production-certified-spectral-rb-kernel-retry"
)
PARENT_RUN_SCHEMA = "experiment12-d0-jacobi-rb-denoising-feasibility"
PARENT_REGISTRY_SCHEMA = PARENT_RUN_SCHEMA + "-artifact-registry"
PARENT_REGISTRY_RECORD_COUNT = 315
PARENT_REGISTRY_SHA256 = (
    "309a4296691684f1cc7ce26bfc243accfb51b8dd2b9ea50a6c43d93dea164e9e"
)
PARENT_SOURCE_FINGERPRINT = (
    "e7803a1e9fac0719e2a51aa069cb6a0c52fdae6eca049bd593501ac96ba30031"
)
PARENT_SAVED_DECISION = "spectral_inversion_computationally_infeasible"
READJUDICATED_DECISION = (
    "spectral_inversion_numerically_valid_but_computationally_infeasible"
)
EXPECTED_PROJECTED_TRANSITION_COUNT = 89_915_392
EXPECTED_PROJECTED_CACHE_HOURS = 40_631.804049512015
EXPECTED_SLOWEST_TRANSITIONS_PER_SECOND = 0.6147031460218352
EXPECTED_LINEAGE_REGISTRY_RECORD_COUNTS = (
    16,
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
    "decision": "jacobi_rb_decision.json",
    "preflight_gate": "jacobi_rb_preflight_gate.json",
    "preflight_metrics": "preflight_metrics.json",
    "kernel_gate": "jacobi_rb_kernel_gate.json",
    "kernel_metrics": "kernel_metrics.json",
    "target_gate": "jacobi_rb_target_gate.json",
    "parent_provenance": "parent_provenance.json",
}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"cannot read fused-CUDA control artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(
            f"fused-CUDA control artifact is not an object: {path}"
        )
    return dict(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactCompatibilityError(message)


def _one(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 1


def _zero(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 0


def _no_work(record: Mapping[str, Any], description: str) -> None:
    _require(
        _zero(record.get("physical_training_performed", 0))
        and _zero(record.get("sampling_performed", 0))
        and _zero(record.get("reverse_sampling_performed", 0)),
        f"{description} records physical training or reverse sampling",
    )


def _verify_registry(
    root: Path,
    registry_path: Path,
    status: Mapping[str, Any],
) -> dict[str, Any]:
    registry = _load(registry_path)
    digest = file_fingerprint(registry_path)
    _require(
        digest == PARENT_REGISTRY_SHA256,
        "control is not the frozen 315-record certified spectral run",
    )
    _require(
        registry.get("schema") == PARENT_REGISTRY_SCHEMA
        and registry.get("schema_version") == 1,
        "control registry schema is incompatible",
    )
    records_raw = registry.get("records")
    _require(isinstance(records_raw, Mapping), "control registry records are invalid")
    records = dict(records_raw)
    _require(
        len(records) == PARENT_REGISTRY_RECORD_COUNT,
        f"control registry must contain exactly {PARENT_REGISTRY_RECORD_COUNT} records",
    )
    exclusions_raw = registry.get("terminal_files_excluded_to_avoid_self_reference")
    _require(isinstance(exclusions_raw, list), "control registry exclusions are invalid")
    exclusions = {str(value) for value in exclusions_raw}
    _require(
        exclusions == {"artifact_registry.json", "run_status.json"},
        "control registry exclusions changed",
    )
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in exclusions
    }
    _require(actual == set(records), "control registry file set changed")
    for relative, raw in records.items():
        _require(isinstance(raw, Mapping), f"invalid registry record: {relative}")
        artifact = root / relative
        _require(
            artifact.is_file()
            and raw.get("sha256") == file_fingerprint(artifact)
            and raw.get("size") == artifact.stat().st_size,
            f"registered control artifact changed: {relative}",
        )
    _require(
        status.get("artifact_registry_sha256") == digest
        and status.get("artifact_registry_record_count")
        == PARENT_REGISTRY_RECORD_COUNT
        and status.get("artifact_registry_size") == registry_path.stat().st_size,
        "control status does not bind its terminal registry",
    )
    _no_work(registry, "control registry")
    return registry


def _metrics(record: Mapping[str, Any], description: str) -> dict[str, Any]:
    raw = record.get("metrics")
    _require(isinstance(raw, Mapping), f"{description} metrics are invalid")
    return dict(raw)


def verify_and_readjudicate_jacobi_rb_cuda_parent(
    run_dir: str | Path,
) -> dict[str, Any]:
    """Verify the exact control and expose only its closed scientific result."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"fused-CUDA control does not exist: {root}")
    _require(
        root.name == PARENT_RUN_BASENAME,
        f"control basename must be {PARENT_RUN_BASENAME}",
    )
    paths = {name: root / relative for name, relative in _REQUIRED_FILES.items()}
    for name, path in paths.items():
        _require(path.is_file(), f"fused-CUDA control lacks {name}: {path.name}")

    status = _load(paths["status"])
    registry = _verify_registry(root, paths["registry"], status)
    manifest = _load(paths["manifest"])
    decision = _load(paths["decision"])
    preflight_gate = _load(paths["preflight_gate"])
    kernel_gate = _load(paths["kernel_gate"])
    target_gate = _load(paths["target_gate"])

    _require(
        manifest.get("schema") == PARENT_RUN_SCHEMA
        and manifest.get("schema_version") == 1
        and manifest.get("source_fingerprint") == PARENT_SOURCE_FINGERPRINT,
        "control manifest or source fingerprint changed",
    )
    source_paths_raw = manifest.get("source_paths")
    _require(
        isinstance(source_paths_raw, list) and bool(source_paths_raw),
        "control source paths are invalid",
    )
    source_paths = [Path(str(value)) for value in source_paths_raw]
    _require(
        all(path.is_file() for path in source_paths)
        and source_fingerprint(source_paths) == PARENT_SOURCE_FINGERPRINT,
        "control source payloads changed",
    )
    _require(
        status.get("schema") == PARENT_RUN_SCHEMA
        and status.get("schema_version") == 1
        and status.get("status") == "complete"
        and status.get("outcome") == "gate_failed"
        and status.get("phase") == "kernel"
        and status.get("required_gate") == "kernel"
        and _zero(status.get("required_gate_pass"))
        and status.get("decision") == PARENT_SAVED_DECISION,
        "control terminal status changed",
    )
    _require(
        decision.get("decision") == PARENT_SAVED_DECISION
        and _zero(decision.get("one_image_training_authorized"))
        and _zero(decision.get("physical_training_authorized"))
        and _zero(decision.get("sampling_authorized")),
        "control decision changed",
    )

    preflight_metrics_record = _load(paths["preflight_metrics"])
    kernel_metrics_record = _load(paths["kernel_metrics"])
    recomputed_preflight = evaluate_jacobi_rb_preflight(
        _metrics(preflight_metrics_record, "preflight")
    )
    recomputed_kernel = evaluate_jacobi_rb_kernel(
        _metrics(kernel_metrics_record, "kernel")
    )
    _require(
        preflight_gate == recomputed_preflight and _one(preflight_gate.get("passed")),
        "control preflight does not recompute to pass",
    )
    _require(
        kernel_gate == recomputed_kernel
        and _zero(kernel_gate.get("passed"))
        and _one(kernel_gate.get("numerically_valid"))
        and _zero(kernel_gate.get("resource_valid")),
        "control is not numerically valid and resource infeasible",
    )
    metrics = _metrics(kernel_metrics_record, "kernel")
    _require(
        metrics.get("projected_transition_count")
        == EXPECTED_PROJECTED_TRANSITION_COUNT
        and metrics.get("projected_cache_hours") == EXPECTED_PROJECTED_CACHE_HOURS
        and metrics.get("slowest_transitions_per_second")
        == EXPECTED_SLOWEST_TRANSITIONS_PER_SECOND
        and metrics.get("quantile_certificate_fraction") == 1.0
        and metrics.get("uncertified_draw_count") == 0
        and metrics.get("approximation_count") == 0,
        "control numerical/resource evidence changed",
    )
    _require(
        target_gate.get("evaluation_status") == "not_evaluated"
        and _zero(target_gate.get("passed"))
        and target_gate.get("subchecks") == {}
        and target_gate.get("reason")
        == "certified production-support kernel gate failed",
        "control target was evaluated or its skip reason changed",
    )

    stored_parent = _load(paths["parent_provenance"])
    transitive_root = Path(str(stored_parent.get("parent_run_dir", "")))
    recomputed_parent = verify_and_readjudicate_jacobi_parent(transitive_root)
    _require(
        stored_parent == recomputed_parent
        and tuple(stored_parent.get("lineage_registry_record_counts", []))
        == EXPECTED_LINEAGE_REGISTRY_RECORD_COUNTS,
        "control transitive provenance changed",
    )

    for description, record in (
        ("control manifest", manifest),
        ("control status", status),
        ("control decision", decision),
        ("control preflight", preflight_gate),
        ("control kernel", kernel_gate),
        ("control target", target_gate),
        ("control preflight metrics", preflight_metrics_record),
        ("control kernel metrics", kernel_metrics_record),
        ("control transitive provenance", stored_parent),
    ):
        _no_work(record, description)

    return {
        "schema": "d0-jacobi-rb-cuda-parent-provenance",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": 1,
        "parent_run_dir": str(root),
        "parent_run_basename": PARENT_RUN_BASENAME,
        "parent_run_schema": PARENT_RUN_SCHEMA,
        "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
        "parent_artifact_registry_size": paths["registry"].stat().st_size,
        "parent_artifact_record_count": len(dict(registry["records"])),
        "parent_source_fingerprint": PARENT_SOURCE_FINGERPRINT,
        "saved_decision": PARENT_SAVED_DECISION,
        "readjudicated_decision": READJUDICATED_DECISION,
        "parent_numerically_valid": 1,
        "parent_resource_feasible": 0,
        "parent_target_evaluation_status": "not_evaluated",
        "lineage_registry_record_counts": [
            PARENT_REGISTRY_RECORD_COUNT,
            *EXPECTED_LINEAGE_REGISTRY_RECORD_COUNTS,
        ],
        "readjudication_basis": {
            "projected_transition_count": EXPECTED_PROJECTED_TRANSITION_COUNT,
            "projected_cache_hours": EXPECTED_PROJECTED_CACHE_HOURS,
            "slowest_transitions_per_second": (
                EXPECTED_SLOWEST_TRANSITIONS_PER_SECOND
            ),
            "quantile_certificate_fraction": 1.0,
            "uncertified_draw_count": 0,
            "approximation_count": 0,
        },
        "cuda_implementation_followup_authorized": 1,
        "target_evaluation_authorized_by_parent": 0,
        "parent_mutated": 0,
        "physical_training_performed": 0,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
    }


verify_parent_jacobi_rb_cuda_run = verify_and_readjudicate_jacobi_rb_cuda_parent
verify_and_readjudicate_jacobi_rb_parent = verify_and_readjudicate_jacobi_rb_cuda_parent


__all__ = [
    "PARENT_RUN_BASENAME",
    "PARENT_RUN_SCHEMA",
    "PARENT_REGISTRY_SCHEMA",
    "PARENT_REGISTRY_RECORD_COUNT",
    "PARENT_REGISTRY_SHA256",
    "PARENT_SOURCE_FINGERPRINT",
    "PARENT_SAVED_DECISION",
    "READJUDICATED_DECISION",
    "EXPECTED_PROJECTED_TRANSITION_COUNT",
    "EXPECTED_PROJECTED_CACHE_HOURS",
    "EXPECTED_SLOWEST_TRANSITIONS_PER_SECOND",
    "EXPECTED_LINEAGE_REGISTRY_RECORD_COUNTS",
    "verify_and_readjudicate_jacobi_rb_cuda_parent",
    "verify_parent_jacobi_rb_cuda_run",
    "verify_and_readjudicate_jacobi_rb_parent",
]
