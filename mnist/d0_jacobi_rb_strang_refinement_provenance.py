"""Immutable provenance for the successful Jacobi RB multipath parent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_rb_cuda_multipath_gate import (
    decide_multipath_workflow,
    evaluate_multipath_kernel,
    evaluate_multipath_pilot,
    evaluate_multipath_preflight,
    evaluate_multipath_target,
)


PARENT_RUN_BASENAME = "20260723-092105_production-multipath-jacobi-rb"
PARENT_RUN_SCHEMA = "experiment12-d0-jacobi-rb-cuda-multipath-confirmation"
PARENT_REGISTRY_SCHEMA = PARENT_RUN_SCHEMA + "-artifact-registry"
PARENT_REGISTRY_RECORD_COUNT = 891
PARENT_REGISTRY_SHA256 = (
    "b1724cb1222baf315b3aff24858ac6d979a2ed36e0331995245220a5861545f5"
)
PARENT_SOURCE_FINGERPRINT = (
    "151eaa6c3fbd3a4beaae61ad5337892187e4338fe629761716f281bb84f7d450"
)
PARENT_SCIENTIFIC_CONFIG_SHA256 = (
    "13de55906b4a9b8696183f16f49e07d2b177d20b8429e6b086b3ce5c36bd1ee9"
)
PARENT_DECISION = "exact_jacobi_rb_multipath_kernel_and_target_feasible"

_REQUIRED_FILES = {
    "manifest": "run_manifest.json",
    "status": "run_status.json",
    "registry": "artifact_registry.json",
    "scientific_config": "scientific_config.json",
    "decision": "multipath_decision.json",
    "workflow": "multipath_workflow_gate.json",
    "preflight_gate": "multipath_preflight_gate.json",
    "preflight_metrics": "preflight_metrics.json",
    "pilot_gate": "multipath_pilot_gate.json",
    "pilot_metrics": "pilot_metrics.json",
    "kernel_gate": "multipath_kernel_gate.json",
    "kernel_metrics": "kernel_metrics.json",
    "target_gate": "multipath_target_gate.json",
    "target_metrics": "target_metrics.json",
    "parent_provenance": "parent_provenance.json",
}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"cannot read Strang-refinement parent artifact {path}: {exc}"
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


def _verify_registry(
    root: Path, path: Path, status: Mapping[str, Any]
) -> dict[str, Any]:
    registry = _load(path)
    digest = file_fingerprint(path)
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
        artifact.relative_to(root).as_posix()
        for artifact in root.rglob("*")
        if artifact.is_file()
        and artifact.relative_to(root).as_posix() not in exclusions
    }
    _require(actual == set(records), "parent registry file set changed")
    for relative, raw in records.items():
        _require(isinstance(raw, Mapping), f"invalid registry row: {relative}")
        artifact = root / str(relative)
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
        and status.get("artifact_registry_size") == path.stat().st_size,
        "parent status does not bind its terminal registry",
    )
    _no_work(registry, "parent registry")
    return registry


def verify_exact_jacobi_rb_multipath_parent(
    run_dir: str | Path,
) -> dict[str, Any]:
    """Verify the exact successful parent and authorize refinement planning."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"Strang-refinement parent does not exist: {root}")
    _require(
        root.name == PARENT_RUN_BASENAME,
        f"parent basename must be {PARENT_RUN_BASENAME}",
    )
    paths = {name: root / relative for name, relative in _REQUIRED_FILES.items()}
    for name, path in paths.items():
        _require(path.is_file(), f"parent lacks {name}: {path.name}")

    status = _load(paths["status"])
    registry = _verify_registry(root, paths["registry"], status)
    manifest = _load(paths["manifest"])
    scientific_config = _load(paths["scientific_config"])
    decision = _load(paths["decision"])
    workflow = _load(paths["workflow"])
    preflight_gate = _load(paths["preflight_gate"])
    pilot_gate = _load(paths["pilot_gate"])
    kernel_gate = _load(paths["kernel_gate"])
    target_gate = _load(paths["target_gate"])

    _require(
        manifest.get("schema") == PARENT_RUN_SCHEMA
        and manifest.get("schema_version") == 1
        and manifest.get("source_fingerprint") == PARENT_SOURCE_FINGERPRINT
        and manifest.get("scientific_config_sha256")
        == PARENT_SCIENTIFIC_CONFIG_SHA256,
        "parent manifest/source/config fingerprint changed",
    )
    raw_sources = manifest.get("source_paths")
    _require(
        isinstance(raw_sources, list) and len(raw_sources) == 11,
        "parent immutable source set changed",
    )
    sources = [Path(str(value)) for value in raw_sources]
    _require(
        all(path.is_file() for path in sources)
        and source_fingerprint(sources) == PARENT_SOURCE_FINGERPRINT,
        "one of the eleven immutable parent sources changed",
    )
    _require(
        config_fingerprint(scientific_config) == PARENT_SCIENTIFIC_CONFIG_SHA256
        and scientific_config.get("schema")
        == PARENT_RUN_SCHEMA + "-scientific-config"
        and scientific_config.get("schema_version") == 1
        and scientific_config.get("root_seed") == 261141
        and scientific_config.get("full_outer_steps") == 512
        and scientific_config.get("projection_path_count") == 64
        and scientific_config.get("projection_group_sizes")
        == [10, 10, 10, 10, 10, 10, 4]
        and scientific_config.get("test_only_reduced_workload") == 0,
        "parent scientific configuration changed",
    )
    _require(
        status.get("schema") == PARENT_RUN_SCHEMA
        and status.get("schema_version") == 1
        and status.get("status") == "complete"
        and status.get("outcome") == "complete"
        and status.get("phase") == "target"
        and status.get("required_gate") == "target"
        and _one(status.get("required_gate_pass"))
        and status.get("decision") == PARENT_DECISION
        and status.get("scientific_config_sha256")
        == PARENT_SCIENTIFIC_CONFIG_SHA256,
        "parent terminal status changed",
    )
    _require(
        decision.get("decision") == PARENT_DECISION
        and _one(decision.get("state_dependent_strang_refinement_authorized"))
        and _zero(decision.get("physical_training_authorized"))
        and _zero(decision.get("sampling_authorized")),
        "parent decision no longer authorizes only Strang refinement",
    )

    preflight_metrics = _metrics(_load(paths["preflight_metrics"]), "preflight")
    pilot_metrics = _metrics(_load(paths["pilot_metrics"]), "pilot")
    kernel_metrics = _metrics(_load(paths["kernel_metrics"]), "kernel")
    target_metrics = _metrics(_load(paths["target_metrics"]), "target")
    _require(
        preflight_gate == evaluate_multipath_preflight(preflight_metrics)
        and _one(preflight_gate.get("passed")),
        "parent preflight no longer recomputes to pass",
    )
    _require(
        pilot_gate == evaluate_multipath_pilot(pilot_metrics)
        and _one(pilot_gate.get("passed"))
        and _one(pilot_gate.get("numerically_valid"))
        and _one(pilot_gate.get("resource_valid")),
        "parent pilot no longer recomputes to pass",
    )
    _require(
        kernel_gate == evaluate_multipath_kernel(kernel_metrics)
        and _one(kernel_gate.get("passed"))
        and _one(kernel_gate.get("numerically_valid"))
        and _one(kernel_gate.get("resource_valid")),
        "parent kernel no longer recomputes to pass",
    )
    _require(
        target_gate == evaluate_multipath_target(target_metrics)
        and _one(target_gate.get("passed")),
        "parent target no longer recomputes to pass",
    )
    recomputed_decision = decide_multipath_workflow(
        provenance=True,
        preflight_gate=preflight_gate,
        pilot_gate=pilot_gate,
        kernel_gate=kernel_gate,
        target_gate=target_gate,
    )
    _require(
        recomputed_decision == decision,
        "parent decision no longer recomputes from its gates",
    )
    components = workflow.get("components")
    _require(
        isinstance(components, Mapping)
        and components.get("preflight") == preflight_gate
        and components.get("pilot") == pilot_gate
        and components.get("kernel") == kernel_gate
        and components.get("target") == target_gate
        and workflow.get("decision") == decision
        and workflow.get("required_gate") == "target"
        and _one(workflow.get("required_gate_pass")),
        "parent workflow no longer binds the four passing components",
    )
    transitive = _load(paths["parent_provenance"])
    _require(
        _one(transitive.get("passed"))
        and transitive.get("parent_artifact_record_count") == 219
        and transitive.get("parent_artifact_registry_sha256")
        == "74a538caa33fbc5ef28e76e7feeedc77287fc0af36b8679c59e241ca3e43a757",
        "parent transitive provenance changed",
    )
    for description, record in (
        ("manifest", manifest),
        ("status", status),
        ("registry", registry),
        ("scientific config", scientific_config),
        ("decision", decision),
        ("workflow", workflow),
        ("preflight", preflight_gate),
        ("pilot", pilot_gate),
        ("kernel", kernel_gate),
        ("target", target_gate),
        ("transitive provenance", transitive),
    ):
        _no_work(record, f"parent {description}")

    return {
        "schema": "d0-jacobi-rb-strang-refinement-parent-provenance",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": 1,
        "parent_run_dir": str(root),
        "parent_run_basename": PARENT_RUN_BASENAME,
        "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
        "parent_artifact_registry_size": paths["registry"].stat().st_size,
        "parent_artifact_record_count": len(dict(registry["records"])),
        "parent_source_fingerprint": PARENT_SOURCE_FINGERPRINT,
        "parent_source_count": len(sources),
        "parent_scientific_config_sha256": PARENT_SCIENTIFIC_CONFIG_SHA256,
        "parent_kernel_pass": 1,
        "parent_target_pass": 1,
        "parent_decision": PARENT_DECISION,
        "state_dependent_strang_refinement_authorized": 1,
        "physical_training_authorized": 0,
        "sampling_authorized": 0,
        "parent_mutated": 0,
        **{
            "physical_training_performed": 0,
            "sampling_performed": 0,
            "reverse_sampling_performed": 0,
        },
    }


__all__ = [
    "PARENT_DECISION",
    "PARENT_REGISTRY_RECORD_COUNT",
    "PARENT_REGISTRY_SHA256",
    "PARENT_RUN_BASENAME",
    "PARENT_SCIENTIFIC_CONFIG_SHA256",
    "PARENT_SOURCE_FINGERPRINT",
    "verify_exact_jacobi_rb_multipath_parent",
]
