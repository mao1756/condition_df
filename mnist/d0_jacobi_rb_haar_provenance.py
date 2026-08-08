"""Immutable provenance for the certified Haar-coupling follow-up.

The admissible parent is the completed phase-local Dynkin observer run.  That
run is retained as evidence: its observer controls passed and its pilot was
numerically/resource valid, but neither the Dynkin nor raw right-endpoint
coupling could meet the frozen power thresholds.  This module verifies every
registered artifact and exposes that scheduling/statistics-only
re-adjudication without mutating the parent directory.
"""

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
from mnist.d0_jacobi_rb_dynkin_phase_observer_gate import (
    evaluate_phase_observer_power,
    evaluate_phase_observer_preflight,
)


PARENT_RUN_BASENAME = (
    "20260724-225759_production-dynkin-strang-power-phase-observer-fix"
)
PARENT_RUN_SCHEMA = (
    "experiment12-d0-jacobi-rb-dynkin-phase-observer-confirmation"
)
PARENT_REGISTRY_SCHEMA = PARENT_RUN_SCHEMA + "-artifact-registry"
PARENT_REGISTRY_RECORD_COUNT = 1061
PARENT_REGISTRY_SHA256 = (
    "0cbe181966a79698b9e8d8177b86f3038cd8c1c619c3af384168fbc61f7a11ac"
)
PARENT_SOURCE_COUNT = 26
PARENT_SOURCE_FINGERPRINT = (
    "19086844e84f141c8aa86a235746ef5ef1376c6d1365253e3045d31df9524e6a"
)
PARENT_SCIENTIFIC_CONFIG_SHA256 = (
    "1730f53acd369c1d33cacc02b2088b865f7ca01d5f563eb1a5e3759ba7201942"
)
PARENT_DECISION = "dynkin_power_infeasible"
PARENT_RE_ADJUDICATION = "right_endpoint_coupling_power_infeasible"

_REQUIRED_FILES = {
    "manifest": "run_manifest.json",
    "status": "run_status.json",
    "registry": "artifact_registry.json",
    "scientific_config": "scientific_config.json",
    "decision": "phase_observer_decision.json",
    "workflow": "phase_observer_workflow_gate.json",
    "preflight_gate": "phase_observer_preflight_gate.json",
    "preflight_metrics": "phase_observer_preflight_metrics.json",
    "pilot_gate": "phase_observer_pilot_gate.json",
    "pilot_metrics": "phase_observer_power_metrics.json",
    "panel_a_metrics": "pilot_panel_a_metrics.json",
    "panel_a_nomination": "panel_a_nomination.json",
    "panel_b_plan": "panel_b_plan.json",
    "sealed_confirmation": "sealed_design_confirmation.json",
    "selected_design": "selected_dynkin_design.json",
    "raw_dynkin_power": "raw_dynkin_power_panel_a.json",
    "transitive_provenance": "parent_provenance.json",
}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"cannot read Haar parent artifact {path}: {exc}"
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
    excluded = set(
        registry.get("terminal_files_excluded_to_avoid_self_reference", ())
    )
    _require(
        excluded == {"artifact_registry.json", "run_status.json"},
        "parent registry exclusions changed",
    )
    actual = {
        artifact.relative_to(root).as_posix()
        for artifact in root.rglob("*")
        if artifact.is_file()
        and artifact.relative_to(root).as_posix() not in excluded
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
        and status.get("artifact_registry_size") == registry_path.stat().st_size,
        "parent status does not bind its terminal registry",
    )
    _no_work(registry, "parent registry")
    return registry


def verify_right_endpoint_coupling_parent(
    run_dir: str | Path,
) -> dict[str, Any]:
    """Verify the immutable parent and re-adjudicate its power failure."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"Haar parent does not exist: {root}")
    _require(
        root.name == PARENT_RUN_BASENAME,
        f"parent basename must be {PARENT_RUN_BASENAME}",
    )
    paths = {name: root / relative for name, relative in _REQUIRED_FILES.items()}
    for name, path in paths.items():
        _require(path.is_file(), f"Haar parent lacks {name}: {path.name}")

    status = _load(paths["status"])
    registry = _verify_registry(root, paths["registry"], status)
    manifest = _load(paths["manifest"])
    config = _load(paths["scientific_config"])
    decision = _load(paths["decision"])
    workflow = _load(paths["workflow"])
    preflight_gate = _load(paths["preflight_gate"])
    preflight_payload = _load(paths["preflight_metrics"])
    pilot_gate = _load(paths["pilot_gate"])
    pilot_payload = _load(paths["pilot_metrics"])
    panel_a_metrics = _load(paths["panel_a_metrics"])
    nomination = _load(paths["panel_a_nomination"])
    panel_b_plan = _load(paths["panel_b_plan"])
    sealed = _load(paths["sealed_confirmation"])
    selected = _load(paths["selected_design"])
    raw_dynkin = _load(paths["raw_dynkin_power"])
    transitive = _load(paths["transitive_provenance"])

    _require(
        manifest.get("schema") == PARENT_RUN_SCHEMA
        and manifest.get("schema_version") == 1
        and manifest.get("source_fingerprint") == PARENT_SOURCE_FINGERPRINT
        and manifest.get("scientific_config_sha256")
        == PARENT_SCIENTIFIC_CONFIG_SHA256
        and manifest.get("source_count") == PARENT_SOURCE_COUNT,
        "parent manifest/source/config binding changed",
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
        "one of the twenty-six immutable parent sources changed",
    )
    _require(
        config_fingerprint(config) == PARENT_SCIENTIFIC_CONFIG_SHA256
        and config.get("schema") == PARENT_RUN_SCHEMA + "-scientific-config"
        and config.get("schema_version") == 1
        and config.get("root_seed") == 261_171
        and config.get("grid_size") == 28
        and config.get("alpha") == 1.0
        and config.get("tau_eff") == 5.0e-5
        and config.get("pilot_panel_paths") == 8
        and config.get("test_only_reduced_workload") == 0,
        "parent scientific configuration changed",
    )
    _require(
        status.get("schema") == PARENT_RUN_SCHEMA
        and status.get("schema_version") == 1
        and status.get("status") == "complete"
        and status.get("outcome") == "gate_failed"
        and status.get("phase") == "pilot"
        and status.get("required_gate") == "pilot"
        and _zero(status.get("required_gate_pass"))
        and status.get("decision") == PARENT_DECISION
        and status.get("source_fingerprint") == PARENT_SOURCE_FINGERPRINT
        and status.get("scientific_config_sha256")
        == PARENT_SCIENTIFIC_CONFIG_SHA256,
        "parent terminal status changed",
    )
    _require(
        decision == workflow
        and decision.get("decision") == PARENT_DECISION
        and _zero(decision.get("production_refinement_patch_authorized"))
        and _zero(decision.get("physical_training_authorized"))
        and _zero(decision.get("sampling_authorized")),
        "parent terminal decision changed",
    )

    recomputed_preflight = evaluate_phase_observer_preflight(
        _metrics(preflight_payload, "parent preflight")
    )
    _require(
        preflight_gate == recomputed_preflight
        and _one(preflight_gate.get("passed"))
        and _one(preflight_gate.get("provenance_valid"))
        and _one(preflight_gate.get("observer_algebra_valid"))
        and _one(preflight_gate.get("tower_identity_valid"))
        and _one(preflight_gate.get("numerically_valid"))
        and _one(preflight_gate.get("resource_valid")),
        "parent phase-observer preflight no longer recomputes to pass",
    )
    recomputed_pilot = evaluate_phase_observer_power(
        _metrics(pilot_payload, "parent pilot")
    )
    _require(
        pilot_gate == recomputed_pilot
        and _zero(pilot_gate.get("passed"))
        and _one(pilot_gate.get("numerically_valid"))
        and _one(pilot_gate.get("resource_valid"))
        and _zero(pilot_gate.get("power_valid"))
        and _zero(pilot_gate.get("panel_a_nominated"))
        and _zero(pilot_gate.get("panels_agree")),
        "parent pilot is not a clean power-only failure",
    )

    execution = panel_a_metrics.get("execution")
    _require(
        isinstance(execution, Mapping)
        and _one(panel_a_metrics.get("complete"))
        and panel_a_metrics.get("levels") == [128, 256, 512, 1024, 2048]
        and execution.get("transition_count") == 87_105_536
        and _one(execution.get("shard_chain_pass"))
        and execution.get("conservative_rate", 0.0) >= 1_300.0
        and execution.get("certificate_fraction") == 1.0
        and execution.get("fallback_fraction", 1.0) <= 1.0e-4
        and execution.get("peak_memory_fraction", 1.0) <= 0.80,
        "parent panel-A execution evidence changed",
    )
    _require(
        nomination.get("evaluation_status") == "evaluated"
        and _zero(nomination.get("passed"))
        and nomination.get("eligible_candidate_count") == 0
        and nomination.get("selection_status") == "panel_a_no_eligible_design"
        and nomination.get("selected") is None,
        "parent panel-A no-nomination evidence changed",
    )
    candidates = nomination.get("candidates")
    _require(
        isinstance(candidates, list)
        and len(candidates) == 4
        and all(
            isinstance(candidate, Mapping)
            and _zero(candidate.get("eligible"))
            and _one(candidate.get("panel_complete_pass"))
            and _one(candidate.get("panel_numerical_health_pass"))
            for candidate in candidates
        ),
        "parent candidate-grid evidence changed",
    )
    _require(
        panel_b_plan.get("panel") == "b"
        and sealed.get("selection_status") == "panel_a_no_eligible_design"
        and _zero(sealed.get("passed"))
        and selected.get("selected") is None
        and _one(selected.get("selected_design_frozen"))
        and raw_dynkin.get("authorizing") == 0,
        "parent sealed panel or advisory-estimator semantics changed",
    )
    raw_families = raw_dynkin.get("families")
    _require(
        isinstance(raw_families, Mapping)
        and isinstance(raw_families.get("main"), Mapping)
        and isinstance(raw_families.get("reference"), Mapping),
        "parent raw/Dynkin power families changed",
    )
    raw_main_block = raw_families["main"].get("maximum_projected_half_width")
    raw_reference_block = raw_families["reference"].get(
        "maximum_projected_half_width"
    )
    _require(
        isinstance(raw_main_block, Mapping)
        and isinstance(raw_reference_block, Mapping),
        "parent raw/Dynkin half-width blocks changed",
    )
    raw_main_widths = raw_main_block.get("raw")
    raw_reference_widths = raw_reference_block.get("raw")
    _require(
        isinstance(raw_main_widths, Mapping)
        and isinstance(raw_reference_widths, Mapping)
        and min(float(value) for value in raw_main_widths.values()) > 0.0025
        and min(float(value) for value in raw_reference_widths.values()) > 0.005,
        "parent raw right-endpoint coupling is no longer power-infeasible",
    )
    records = set(dict(registry["records"]))
    _require(
        not any(
            relative.startswith("dynkin_shards/pilot/b/")
            or relative.startswith("dynkin_shards/refinement/")
            or relative.startswith("refinement_")
            for relative in records
        ),
        "parent contains panel-B or production-refinement execution",
    )
    _require(
        _one(transitive.get("passed"))
        and transitive.get("parent_artifact_record_count") == 18
        and transitive.get("parent_artifact_registry_sha256")
        == "57daf61e9686c2257d6579c84130e5e4b4a400a8435916a62ac408c87ad6072d",
        "parent transitive provenance changed",
    )
    for description, record in (
        ("manifest", manifest),
        ("status", status),
        ("registry", registry),
        ("scientific config", config),
        ("decision", decision),
        ("preflight gate", preflight_gate),
        ("pilot gate", pilot_gate),
        ("panel A metrics", panel_a_metrics),
        ("nomination", nomination),
        ("panel B plan", panel_b_plan),
        ("sealed confirmation", sealed),
        ("selected design", selected),
        ("raw/Dynkin power", raw_dynkin),
        ("transitive provenance", transitive),
    ):
        _no_work(record, f"parent {description}")

    return {
        "schema": "d0-jacobi-rb-haar-parent-provenance",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": 1,
        "parent_run_dir": str(root),
        "parent_run_basename": PARENT_RUN_BASENAME,
        "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
        "parent_artifact_registry_size": paths["registry"].stat().st_size,
        "parent_artifact_record_count": PARENT_REGISTRY_RECORD_COUNT,
        "parent_source_fingerprint": PARENT_SOURCE_FINGERPRINT,
        "parent_source_count": PARENT_SOURCE_COUNT,
        "parent_scientific_config_sha256": PARENT_SCIENTIFIC_CONFIG_SHA256,
        "parent_decision": PARENT_DECISION,
        "parent_re_adjudication": PARENT_RE_ADJUDICATION,
        "parent_preflight_pass": 1,
        "parent_pilot_numerically_valid": 1,
        "parent_pilot_resource_valid": 1,
        "parent_pilot_power_valid": 0,
        "parent_panel_a_nominated": 0,
        "parent_panel_b_opened": 0,
        "parent_selected_design": None,
        "parent_raw_main_best_half_width": min(
            float(value) for value in raw_main_widths.values()
        ),
        "parent_raw_reference_best_half_width": min(
            float(value) for value in raw_reference_widths.values()
        ),
        "parent_production_refinement_performed": 0,
        "parent_mutated": 0,
        "physical_training_authorized": 0,
        "sampling_authorized": 0,
        "physical_training_performed": 0,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
    }


__all__ = [
    "PARENT_DECISION",
    "PARENT_RE_ADJUDICATION",
    "PARENT_REGISTRY_RECORD_COUNT",
    "PARENT_REGISTRY_SHA256",
    "PARENT_RUN_BASENAME",
    "PARENT_RUN_SCHEMA",
    "PARENT_SCIENTIFIC_CONFIG_SHA256",
    "PARENT_SOURCE_COUNT",
    "PARENT_SOURCE_FINGERPRINT",
    "verify_right_endpoint_coupling_parent",
]
