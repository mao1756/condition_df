"""Immutable provenance for the multi-path Jacobi RB scheduling follow-up.

The admissible parent is the completed 219-artifact fused-CUDA run.  That run
certified the exact transition and completed three evolving one-path repeats;
it failed only the frozen throughput and projected-wall-time checks.  This
module verifies that complete record without changing any parent artifact.
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
from mnist.d0_jacobi_rb_cuda_gate import (
    evaluate_jacobi_rb_cuda_certificate,
    evaluate_jacobi_rb_cuda_kernel,
    evaluate_jacobi_rb_cuda_preflight,
)


PARENT_RUN_BASENAME = (
    "20260722-233246_codex-final-validation-certified-cuda-dd-jacobi-rb"
)
PARENT_RUN_SCHEMA = "experiment12-d0-jacobi-rb-cuda-confirmation"
PARENT_REGISTRY_SCHEMA = PARENT_RUN_SCHEMA + "-artifact-registry"
PARENT_REGISTRY_RECORD_COUNT = 219
PARENT_REGISTRY_SHA256 = (
    "74a538caa33fbc5ef28e76e7feeedc77287fc0af36b8679c59e241ca3e43a757"
)
PARENT_SOURCE_FINGERPRINT = (
    "c6ab156d467cbf17bd804c1a204e91e11be637b23ba4aa15bd066994e8bba52f"
)
PARENT_SAVED_DECISION = "spectral_inversion_computationally_infeasible"
READJUDICATED_DECISION = "single_path_scheduling_resource_infeasible"
PARENT_SLOWEST_TRANSITIONS_PER_SECOND = 643.1408275980631
PARENT_PROJECTED_CACHE_HOURS = 38.835192396442096
PARENT_PROJECTED_TRANSITION_COUNT = 89_915_392
PARENT_FAILED_KERNEL_CHECKS = frozenset(
    {"slowest_transitions_per_second", "projected_cache_hours"}
)

_REQUIRED_FILES = {
    "manifest": "run_manifest.json",
    "status": "run_status.json",
    "registry": "artifact_registry.json",
    "decision": "jacobi_rb_cuda_decision.json",
    "workflow": "jacobi_rb_cuda_workflow_gate.json",
    "preflight_gate": "jacobi_rb_cuda_preflight_gate.json",
    "preflight_metrics": "preflight_metrics.json",
    "certificate_gate": "jacobi_rb_cuda_certificate_gate.json",
    "certificate_metrics": "certificate_metrics.json",
    "kernel_gate": "jacobi_rb_cuda_kernel_gate.json",
    "kernel_metrics": "kernel_metrics.json",
    "parent_provenance": "parent_provenance.json",
}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"cannot read multi-path parent artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"parent artifact is not an object: {path}")
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


def _metrics(record: Mapping[str, Any], description: str) -> dict[str, Any]:
    raw = record.get("metrics")
    _require(isinstance(raw, Mapping), f"{description} metrics are invalid")
    return dict(raw)


def _failed_checks(gate: Mapping[str, Any]) -> set[str]:
    raw = gate.get("subchecks")
    _require(isinstance(raw, Mapping), "parent kernel subchecks are invalid")
    return {
        str(name)
        for name, value in raw.items()
        if not isinstance(value, Mapping) or not _one(value.get("passed"))
    }


def _verify_registry(
    root: Path, registry_path: Path, status: Mapping[str, Any]
) -> dict[str, Any]:
    registry = _load(registry_path)
    digest = file_fingerprint(registry_path)
    _require(digest == PARENT_REGISTRY_SHA256, "parent registry SHA-256 changed")
    _require(
        registry.get("schema") == PARENT_REGISTRY_SCHEMA
        and registry.get("schema_version") == 1,
        "parent registry schema changed",
    )
    raw_records = registry.get("records")
    _require(isinstance(raw_records, Mapping), "parent registry records are invalid")
    records = dict(raw_records)
    _require(
        len(records) == PARENT_REGISTRY_RECORD_COUNT,
        f"parent registry must contain {PARENT_REGISTRY_RECORD_COUNT} records",
    )
    exclusions = set(registry.get("terminal_files_excluded_to_avoid_self_reference", ()))
    _require(
        exclusions == {"artifact_registry.json", "run_status.json"},
        "parent registry exclusions changed",
    )
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in exclusions
    }
    _require(actual == set(records), "parent registry file set changed")
    for relative, raw in records.items():
        _require(isinstance(raw, Mapping), f"invalid registry row: {relative}")
        artifact = root / relative
        _require(
            artifact.is_file()
            and raw.get("sha256") == file_fingerprint(artifact)
            and raw.get("size") == artifact.stat().st_size,
            f"registered parent artifact changed: {relative}",
        )
    _require(
        status.get("artifact_registry_sha256") == digest
        and status.get("artifact_registry_record_count")
        == PARENT_REGISTRY_RECORD_COUNT
        and status.get("artifact_registry_size") == registry_path.stat().st_size,
        "parent status does not bind its registry",
    )
    _no_work(registry, "parent registry")
    return registry


def verify_and_readjudicate_jacobi_rb_cuda_multipath_parent(
    run_dir: str | Path,
) -> dict[str, Any]:
    """Verify the exact failed run and expose its scheduling-only conclusion."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"multi-path parent does not exist: {root}")
    _require(
        root.name == PARENT_RUN_BASENAME,
        f"parent basename must be {PARENT_RUN_BASENAME}",
    )
    paths = {name: root / relative for name, relative in _REQUIRED_FILES.items()}
    for name, path in paths.items():
        _require(path.is_file(), f"multi-path parent lacks {name}: {path.name}")

    status = _load(paths["status"])
    registry = _verify_registry(root, paths["registry"], status)
    manifest = _load(paths["manifest"])
    decision = _load(paths["decision"])
    workflow = _load(paths["workflow"])
    preflight_gate = _load(paths["preflight_gate"])
    certificate_gate = _load(paths["certificate_gate"])
    kernel_gate = _load(paths["kernel_gate"])

    _require(
        manifest.get("schema") == PARENT_RUN_SCHEMA
        and manifest.get("schema_version") == 1
        and manifest.get("source_fingerprint") == PARENT_SOURCE_FINGERPRINT,
        "parent manifest/source fingerprint changed",
    )
    raw_sources = manifest.get("source_paths")
    _require(isinstance(raw_sources, list) and len(raw_sources) == 7, "parent source set changed")
    source_paths = [Path(str(value)) for value in raw_sources]
    _require(
        all(path.is_file() for path in source_paths)
        and source_fingerprint(source_paths) == PARENT_SOURCE_FINGERPRINT,
        "one of the seven immutable parent sources changed",
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
        "parent terminal status changed",
    )
    _require(
        decision.get("decision") == PARENT_SAVED_DECISION
        and _zero(decision.get("kernel_and_target_followup_authorized"))
        and _zero(decision.get("physical_training_authorized"))
        and _zero(decision.get("sampling_authorized")),
        "parent decision changed",
    )

    preflight_metrics = _metrics(_load(paths["preflight_metrics"]), "preflight")
    certificate_metrics = _metrics(
        _load(paths["certificate_metrics"]), "certificate"
    )
    kernel_metrics = _metrics(_load(paths["kernel_metrics"]), "kernel")
    _require(
        preflight_gate == evaluate_jacobi_rb_cuda_preflight(preflight_metrics)
        and _one(preflight_gate.get("passed")),
        "parent preflight no longer recomputes to pass",
    )
    _require(
        certificate_gate
        == evaluate_jacobi_rb_cuda_certificate(certificate_metrics)
        and _one(certificate_gate.get("passed"))
        and _one(certificate_gate.get("numerically_valid"))
        and _one(certificate_gate.get("fallback_valid")),
        "parent certificate no longer recomputes to pass",
    )
    recomputed_kernel = evaluate_jacobi_rb_cuda_kernel(kernel_metrics)
    _require(
        kernel_gate == recomputed_kernel
        and _zero(kernel_gate.get("passed"))
        and _one(kernel_gate.get("numerically_valid"))
        and _zero(kernel_gate.get("resource_valid"))
        and _failed_checks(kernel_gate) == PARENT_FAILED_KERNEL_CHECKS,
        "parent is not a numerical pass with only the two resource failures",
    )
    _require(
        kernel_metrics.get("slowest_transitions_per_second")
        == PARENT_SLOWEST_TRANSITIONS_PER_SECOND
        and kernel_metrics.get("projected_cache_hours")
        == PARENT_PROJECTED_CACHE_HOURS
        and kernel_metrics.get("projected_transition_count")
        == PARENT_PROJECTED_TRANSITION_COUNT
        and kernel_metrics.get("uncertified_draw_count") == 0
        and kernel_metrics.get("cpu_fallback_count") == 0
        and kernel_metrics.get("approximation_count") == 0,
        "parent resource/numerical evidence changed",
    )
    components = workflow.get("components")
    _require(isinstance(components, Mapping), "parent workflow components are invalid")
    target = components.get("target")
    _require(
        isinstance(target, Mapping)
        and target.get("evaluation_status") == "not_evaluated"
        and target.get("reason") == "target controls not run"
        and _zero(target.get("passed")),
        "parent target was evaluated or relabeled",
    )
    _require(
        not (root / "jacobi_rb_cuda_target_gate.json").exists()
        and not (root / "target_metrics.json").exists(),
        "parent unexpectedly contains target artifacts",
    )
    transitive = _load(paths["parent_provenance"])
    _require(
        _one(transitive.get("passed"))
        and transitive.get("parent_artifact_record_count") == 315
        and transitive.get("parent_artifact_registry_sha256")
        == "309a4296691684f1cc7ce26bfc243accfb51b8dd2b9ea50a6c43d93dea164e9e",
        "parent transitive provenance changed",
    )
    for name, record in (
        ("status", status), ("decision", decision), ("workflow", workflow),
        ("preflight", preflight_gate), ("certificate", certificate_gate),
        ("kernel", kernel_gate), ("transitive provenance", transitive),
    ):
        _no_work(record, f"parent {name}")

    return {
        "schema": "d0-jacobi-rb-cuda-multipath-parent-provenance",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": 1,
        "parent_run_dir": str(root),
        "parent_run_basename": PARENT_RUN_BASENAME,
        "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
        "parent_artifact_registry_size": paths["registry"].stat().st_size,
        "parent_artifact_record_count": len(dict(registry["records"])),
        "parent_source_fingerprint": PARENT_SOURCE_FINGERPRINT,
        "parent_certificate_valid": 1,
        "parent_kernel_numerically_valid": 1,
        "parent_kernel_resource_valid": 0,
        "parent_target_evaluation_status": "not_evaluated",
        "saved_decision": PARENT_SAVED_DECISION,
        "readjudicated_decision": READJUDICATED_DECISION,
        "readjudication_basis": {
            "failed_kernel_checks": sorted(PARENT_FAILED_KERNEL_CHECKS),
            "slowest_transitions_per_second": PARENT_SLOWEST_TRANSITIONS_PER_SECOND,
            "projected_cache_hours": PARENT_PROJECTED_CACHE_HOURS,
            "projected_transition_count": PARENT_PROJECTED_TRANSITION_COUNT,
        },
        "multipath_scheduling_followup_authorized": 1,
        "parent_mutated": 0,
        "physical_training_performed": 0,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
    }


__all__ = [
    "PARENT_REGISTRY_RECORD_COUNT",
    "PARENT_REGISTRY_SHA256",
    "PARENT_RUN_BASENAME",
    "PARENT_SOURCE_FINGERPRINT",
    "READJUDICATED_DECISION",
    "verify_and_readjudicate_jacobi_rb_cuda_multipath_parent",
]
