"""Immutable provenance for the failed raw-endpoint Strang power parent.

The parent run is scientific evidence, not a cache that this workflow may
rewrite.  This verifier therefore binds the complete terminal registry and
then recomputes the two evaluated parent gates.  It also corrects one reporting
ambiguity: the parent's null selected-design sentinel made its top-level
``resource_valid`` flag false even though three individual candidate designs
fit the frozen 48-hour budget.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_rb_strang_refinement_gate import (
    decide_strang_refinement_workflow,
    evaluate_refinement_power,
    evaluate_strang_preflight,
)


PARENT_RUN_BASENAME = (
    "20260723-230629_production-state-dependent-strang-refinement"
)
PARENT_RUN_SCHEMA = "experiment12-d0-jacobi-rb-strang-refinement"
PARENT_REGISTRY_SCHEMA = PARENT_RUN_SCHEMA + "-artifact-registry"
PARENT_REGISTRY_RECORD_COUNT = 1308
PARENT_REGISTRY_SHA256 = (
    "734c93e1e7d0be29041e1d567b36cbd8ea7aac50df7996d5f8c41fbddef8e632"
)
PARENT_SOURCE_FINGERPRINT = (
    "2f20297eb83b434aa782676119915e9f8883eb116cec0d2b08c2c8c9a8b5ddb0"
)
PARENT_SOURCE_COUNT = 15
PARENT_SCIENTIFIC_CONFIG_SHA256 = (
    "884c181610426e8d0c2adb99fc2835aa98322116d7bfdf2c684fcb7a3c286396"
)
PARENT_DECISION = "refinement_power_infeasible"
PARENT_RE_ADJUDICATION = "raw_endpoint_power_infeasible"
# Backward-compatible spelling used by the first additive CLI draft.
PARENT_READJUDICATION = PARENT_RE_ADJUDICATION
PARENT_TRANSITION_COUNT = 112_394_240
PARENT_CERTIFIED_COUNT = 112_394_240
PARENT_FALLBACK_COUNT = 30
PARENT_CONSERVATIVE_RATE = 2725.4689340542554
PARENT_RESOURCE_FEASIBLE_CANDIDATE_COUNT = 3

_EXPECTED_CANDIDATES = {(32, 16), (32, 32), (64, 16), (64, 32)}
_EXPECTED_POWER_FAILURES = {
    "predicted_main_half_width",
    "predicted_reference_half_width",
    "projected_production_hours",
    "selected_main_paths",
    "selected_reference_paths",
    "variance_only_selection_pass",
}
_FORBIDDEN_COUNTS = (
    "uncertified_count",
    "resource_cap_count",
    "invalid_density_count",
    "approximation_count",
    "correction_count",
    "floor_count",
    "limiter_count",
    "projection_count",
    "renormalization_count",
    "nonfinite_count",
)
_REQUIRED_FILES = {
    "manifest": "run_manifest.json",
    "status": "run_status.json",
    "registry": "artifact_registry.json",
    "scientific_config": "scientific_config.json",
    "decision": "strang_refinement_decision.json",
    "workflow": "strang_workflow_gate.json",
    "preflight_gate": "strang_preflight_gate.json",
    "preflight_metrics": "preflight_metrics.json",
    "power_gate": "strang_power_gate.json",
    "power_metrics": "power_metrics.json",
    "selection": "selected_refinement_design.json",
    "candidate_csv": "refinement_design_candidates.csv",
    "parent_provenance": "parent_provenance.json",
}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"cannot read Dynkin parent artifact {path}: {exc}"
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


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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


def _failed_names(gate: Mapping[str, Any]) -> set[str]:
    raw = gate.get("subchecks")
    _require(isinstance(raw, Mapping), "parent power subchecks are invalid")
    return {
        str(name)
        for name, value in raw.items()
        if not isinstance(value, Mapping) or not _one(value.get("passed"))
    }


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


def _verify_candidates(
    selection: Mapping[str, Any],
    *,
    maximum_hours: float,
    maximum_main_width: float,
    maximum_reference_width: float,
) -> dict[str, Any]:
    raw_candidates = selection.get("candidates")
    _require(isinstance(raw_candidates, Sequence), "parent candidates are invalid")
    candidates: list[dict[str, Any]] = []
    observed: set[tuple[int, int]] = set()
    for raw in raw_candidates:
        _require(isinstance(raw, Mapping), "parent candidate row is invalid")
        row = dict(raw)
        key = (int(row.get("main_paths", -1)), int(row.get("reference_paths", -1)))
        _require(
            key in _EXPECTED_CANDIDATES and key not in observed,
            "parent candidate grid changed",
        )
        observed.add(key)
        for name in (
            "predicted_main_half_width",
            "predicted_reference_half_width",
            "projected_hours",
            "conservative_rate",
        ):
            _require(
                _finite(row.get(name)) and float(row[name]) >= 0.0,
                f"parent candidate {key} has invalid {name}",
            )
        _require(
            float(row["conservative_rate"]) == PARENT_CONSERVATIVE_RATE
            and row.get("variance_bound") == "normal-chi-square-bonferroni"
            and int(row.get("variance_family_size", -1)) == 40
            and _one(row.get("variance_only_pass"))
            and _one(row.get("pilot_production_isolation_pass"))
            and _one(row.get("pilot_means_excluded_pass")),
            f"parent candidate {key} metadata changed",
        )
        expected_eligible = (
            float(row["predicted_main_half_width"]) <= maximum_main_width
            and float(row["predicted_reference_half_width"])
            <= maximum_reference_width
            and float(row["projected_hours"]) <= maximum_hours
        )
        _require(
            _zero(row.get("eligible")) and not expected_eligible,
            f"parent candidate {key} eligibility changed",
        )
        candidates.append(row)
    _require(observed == _EXPECTED_CANDIDATES, "parent candidate grid is incomplete")
    resource_rows = [
        row for row in candidates if float(row["projected_hours"]) <= maximum_hours
    ]
    _require(
        len(resource_rows) == PARENT_RESOURCE_FEASIBLE_CANDIDATE_COUNT,
        "parent runtime-feasible candidate count changed",
    )
    _require(
        all(
            float(row["predicted_main_half_width"]) > maximum_main_width
            or float(row["predicted_reference_half_width"])
            > maximum_reference_width
            for row in candidates
        ),
        "parent is no longer a raw-endpoint power failure",
    )
    return {
        "candidate_count": len(candidates),
        "resource_feasible_candidate_count": len(resource_rows),
        "resource_feasible_candidates": [
            {
                "main_paths": int(row["main_paths"]),
                "reference_paths": int(row["reference_paths"]),
                "projected_hours": float(row["projected_hours"]),
            }
            for row in resource_rows
        ],
        "raw_endpoint_power_failure_pass": 1,
    }


def verify_raw_endpoint_power_infeasible_parent(
    run_dir: str | Path,
) -> dict[str, Any]:
    """Verify and re-adjudicate the exact 1,308-artifact failed parent."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"Dynkin parent does not exist: {root}")
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
    power_gate = _load(paths["power_gate"])
    power_record = _load(paths["power_metrics"])
    selection = _load(paths["selection"])
    transitive = _load(paths["parent_provenance"])

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
        isinstance(raw_sources, list) and len(raw_sources) == PARENT_SOURCE_COUNT,
        "parent immutable source set changed",
    )
    sources = [Path(str(value)) for value in raw_sources]
    _require(
        all(path.is_file() for path in sources)
        and source_fingerprint(sources) == PARENT_SOURCE_FINGERPRINT,
        "one of the fifteen immutable parent sources changed",
    )
    _require(
        config_fingerprint(scientific_config) == PARENT_SCIENTIFIC_CONFIG_SHA256
        and scientific_config.get("schema")
        == PARENT_RUN_SCHEMA + "-scientific-config"
        and scientific_config.get("schema_version") == 1
        and scientific_config.get("root_seed") == 261151
        and scientific_config.get("pilot_main_paths") == 16
        and scientific_config.get("pilot_reference_paths") == 8
        and scientific_config.get("candidate_main_paths") == [32, 64]
        and scientific_config.get("candidate_reference_paths") == [16, 32]
        and scientific_config.get("test_only_reduced_workload") == 0,
        "parent scientific configuration changed",
    )
    _require(
        status.get("schema") == PARENT_RUN_SCHEMA
        and status.get("schema_version") == 1
        and status.get("status") == "complete"
        and status.get("outcome") == "gate_failed"
        and status.get("phase") == "power"
        and status.get("required_gate") == "power"
        and _zero(status.get("required_gate_pass"))
        and status.get("decision") == PARENT_DECISION
        and status.get("scientific_config_sha256")
        == PARENT_SCIENTIFIC_CONFIG_SHA256,
        "parent terminal status changed",
    )
    _require(
        decision.get("decision") == PARENT_DECISION
        and _zero(decision.get("one_image_phase_conditioned_training_patch_authorized"))
        and _zero(decision.get("physical_training_authorized"))
        and _zero(decision.get("sampling_authorized")),
        "parent decision changed",
    )

    preflight_metrics = _metrics(
        _load(paths["preflight_metrics"]), "parent preflight"
    )
    power_metrics = _metrics(power_record, "parent power")
    _require(
        preflight_gate == evaluate_strang_preflight(preflight_metrics)
        and _one(preflight_gate.get("passed")),
        "parent preflight no longer recomputes to pass",
    )
    _require(
        power_gate == evaluate_refinement_power(power_metrics)
        and _zero(power_gate.get("passed"))
        and _one(power_gate.get("numerically_valid"))
        and _zero(power_gate.get("power_valid"))
        and _zero(power_gate.get("resource_valid"))
        and _failed_names(power_gate) == _EXPECTED_POWER_FAILURES,
        "parent power gate no longer has the exact fail-closed sentinel failure",
    )
    _require(
        power_record.get("selection") == selection
        and selection.get("selection_status") == "no_eligible_design"
        and _zero(selection.get("passed"))
        and selection.get("selected") is None,
        "parent null selection record changed",
    )
    thresholds = selection.get("thresholds")
    _require(isinstance(thresholds, Mapping), "parent selection thresholds changed")
    candidate_record = _verify_candidates(
        selection,
        maximum_hours=float(thresholds.get("maximum_projected_hours")),
        maximum_main_width=float(thresholds.get("maximum_main_half_width")),
        maximum_reference_width=float(
            thresholds.get("maximum_reference_half_width")
        ),
    )

    execution = power_record.get("pilot_execution")
    _require(isinstance(execution, Mapping), "parent pilot execution is invalid")
    _require(
        int(execution.get("transition_count", -1)) == PARENT_TRANSITION_COUNT
        and int(execution.get("certified_count", -1)) == PARENT_CERTIFIED_COUNT
        and int(execution.get("fallback_count", -1)) == PARENT_FALLBACK_COUNT
        and float(execution.get("certificate_fraction", -1.0)) == 1.0
        and _one(execution.get("shard_chain_pass"))
        and _one(execution.get("state_updates_device_resident_pass"))
        and all(_zero(execution.get(name)) for name in _FORBIDDEN_COUNTS),
        "parent pilot is no longer numerically clean",
    )

    recomputed_decision = decide_strang_refinement_workflow(
        provenance=True,
        preflight_gate=preflight_gate,
        power_gate=power_gate,
        refinement_gate=workflow.get("components", {}).get("refinement"),
    )
    components = workflow.get("components")
    _require(
        recomputed_decision == decision
        and isinstance(components, Mapping)
        and components.get("preflight") == preflight_gate
        and components.get("power") == power_gate
        and isinstance(components.get("refinement"), Mapping)
        and components["refinement"].get("evaluation_status") == "not_evaluated"
        and workflow.get("decision") == decision
        and workflow.get("required_gate") == "power"
        and _zero(workflow.get("required_gate_pass")),
        "parent workflow no longer binds its failed power outcome",
    )
    _require(
        not (root / "refinement_metrics.json").exists()
        and not (root / "refinement_observables.npz").exists(),
        "parent unexpectedly contains production refinement evidence",
    )
    _require(
        _one(transitive.get("passed"))
        and transitive.get("parent_artifact_record_count") == 891
        and transitive.get("parent_artifact_registry_sha256")
        == "b1724cb1222baf315b3aff24858ac6d979a2ed36e0331995245220a5861545f5",
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
        ("power", power_gate),
        ("power metrics", power_record),
        ("selection", selection),
        ("transitive provenance", transitive),
    ):
        _no_work(record, f"parent {description}")

    return {
        "schema": "d0-jacobi-rb-dynkin-power-parent-provenance",
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
        "parent_decision": PARENT_DECISION,
        "parent_re_adjudication": PARENT_RE_ADJUDICATION,
        "parent_preflight_pass": 1,
        "parent_power_numerically_valid": 1,
        "parent_power_pass": 0,
        "parent_resource_sentinel_invalid": 1,
        **{
            f"parent_{name}": value
            for name, value in candidate_record.items()
        },
        "parent_mutated": 0,
        "physical_training_authorized": 0,
        "sampling_authorized": 0,
        "physical_training_performed": 0,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
    }


__all__ = [
    "PARENT_CERTIFIED_COUNT",
    "PARENT_CONSERVATIVE_RATE",
    "PARENT_DECISION",
    "PARENT_FALLBACK_COUNT",
    "PARENT_RE_ADJUDICATION",
    "PARENT_READJUDICATION",
    "PARENT_REGISTRY_RECORD_COUNT",
    "PARENT_REGISTRY_SHA256",
    "PARENT_RESOURCE_FEASIBLE_CANDIDATE_COUNT",
    "PARENT_RUN_BASENAME",
    "PARENT_SCIENTIFIC_CONFIG_SHA256",
    "PARENT_SOURCE_COUNT",
    "PARENT_SOURCE_FINGERPRINT",
    "PARENT_TRANSITION_COUNT",
    "verify_raw_endpoint_power_infeasible_parent",
]
