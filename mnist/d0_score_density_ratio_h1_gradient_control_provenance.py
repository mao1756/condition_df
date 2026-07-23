"""Strict provenance for the gradient-controlled H1 density-ratio experiment.

The only admissible parent is the immutable 301-record H1 function-step
pilot that ended ``h1_function_step_unresolved``.  This verifier binds the
known terminal registry, checks every registered payload, recomputes the
saved gates, verifies all eight zero-clipping pilot tasks, and recursively
verifies the 263 -> 123 -> 125 -> 332 -> 222 -> 381 ancestry.

The module is additive: none of the sources fingerprinted by the completed
parent or its ancestors are modified in order to verify them.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from mnist.d0_one_image_gate import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_score_density_ratio_h1_trust_gate import (
    H1TrustThresholds,
    evaluate_h1_calibration,
    evaluate_h1_operator_preflight,
    evaluate_h1_pilot,
    evaluate_h1_preflight,
    evaluate_h1_workflow,
)
from mnist.d0_score_density_ratio_h1_trust_provenance import (
    EXPECTED_HEAD_COORDINATE_VERSION,
    EXPECTED_MODEL_SCHEMA,
    EXPECTED_OPTIMIZER_COORDINATE_VERSION,
    verify_parent_multiplicity_run,
)
from mnist.d0_score_density_ratio_head_provenance import (
    EXPECTED_KERNEL,
    PARENT_LOSS_SCALE,
)


__all__ = [
    "PARENT_RUN_SCHEMA",
    "PARENT_REGISTRY_SCHEMA",
    "PARENT_REGISTRY_RECORD_COUNT",
    "PARENT_REGISTRY_SHA256",
    "EXPECTED_PARENT_DECISION",
    "EXPECTED_MULTIPLIERS",
    "EXPECTED_LINEAGE_REGISTRY_RECORD_COUNTS",
    "verify_parent_h1_trust_run",
    "verify_parent_h1_run",
]


PARENT_RUN_SCHEMA = "experiment12-d0-score-density-ratio-h1-trust-confirmation"
PARENT_REGISTRY_SCHEMA = PARENT_RUN_SCHEMA + "-artifact-registry"
PARENT_REGISTRY_RECORD_COUNT = 301
PARENT_REGISTRY_SHA256 = (
    "26c42bc045903253c504a00499318292b3e5612f54b16e103b0872f8933fc6f4"
)
EXPECTED_PARENT_DECISION = "h1_function_step_unresolved"
EXPECTED_MULTIPLIERS = (0.0, 0.1, 0.3, 1.0)
EXPECTED_PILOT_MODEL_SEED = 261011
EXPECTED_LINEAGE_REGISTRY_RECORD_COUNTS = (263, 123, 125, 332, 222, 381)
EXPECTED_H1_OPERATOR_VERSION = "d0-stopped-ema-l2-gamma-increment-v1"
EXPECTED_H1_CALIBRATION_VERSION = "d0-one-shadow-step-h1-calibration-v1"
EXPECTED_H1_STREAM_VERSION = "d0-reference-h1-trust-stream-v1"
EXPECTED_H1_TASK_VERSION = "d0-paired-bce-stopped-ema-h1-training-v1"
EXPECTED_H1_VALUE_SCALE = 0.03735975761867785
EXPECTED_H1_ENERGY_SCALE = 0.00541742970034323
EXPECTED_H1_LAMBDA_BASE = 5.620791829277054e-8

_REQUIRED_FILES: dict[str, str] = {
    "manifest": "run_manifest.json",
    "status": "run_status.json",
    "registry": "artifact_registry.json",
    "parent_provenance": "parent_provenance.json",
    "operator_raw": "h1_operator_preflight.json",
    "operator_gate": "h1_operator_gate.json",
    "calibration_raw": "h1_calibration.json",
    "calibration_gate": "h1_calibration_gate.json",
    "preflight_gate": "h1_preflight_gate.json",
    "pilot_panel_registry": "h1_pilot_panel_registry.json",
    "pilot_oracle": "pilot_oracle_feasibility.json",
    "pilot_candidates": "h1_pilot_candidates.json",
    "pilot_gate": "h1_pilot_gate.json",
    "pilot_failures": "pilot_task_failures.json",
    "pilot_null_gate": "pilot_null_family_gate.json",
    "pilot_null_max_t": "pilot_null_b_max_t.json",
    "workflow": "h1_control_gate.json",
    "decision": "h1_trust_decision.json",
}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"cannot read H1 parent artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"H1 parent artifact is not a JSON object: {path}")
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


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return _finite(actual) and math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-15
        )
    return actual == expected


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_fingerprint(path),
        "size": int(path.stat().st_size),
    }


def _no_work(value: Mapping[str, Any], description: str) -> None:
    _require(
        _zero(value.get("physical_training_performed", 0))
        and _zero(value.get("sampling_performed", 0)),
        f"{description} records physical training or sampling",
    )


def _verify_record(
    raw: Mapping[str, Any],
    *,
    root: Path,
    expected_path: Path,
    description: str,
) -> None:
    path = expected_path.resolve()
    raw_path = raw.get("path")
    if raw_path is not None:
        _require(Path(str(raw_path)).resolve() == path, f"{description} path mismatch")
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtifactCompatibilityError(f"{description} escapes the parent run") from exc
    _require(path.is_file(), f"{description} is missing")
    _require(
        raw.get("sha256") == file_fingerprint(path)
        and int(raw.get("size", -1)) == int(path.stat().st_size),
        f"{description} hash or size mismatch",
    )


def _verify_registry(
    run_dir: Path,
    *,
    registry_path: Path,
    status_path: Path,
    expected_count: int = PARENT_REGISTRY_RECORD_COUNT,
    expected_sha256: str | None = PARENT_REGISTRY_SHA256,
) -> dict[str, Any]:
    """Verify the exact terminal file set and every registered payload."""

    registry = _load(registry_path)
    status = _load(status_path)
    registry_sha = file_fingerprint(registry_path)
    _require(
        registry.get("schema") == PARENT_REGISTRY_SCHEMA
        and int(registry.get("schema_version", -1)) == 1,
        "H1 terminal registry schema is incompatible",
    )
    if expected_sha256 is not None:
        _require(
            registry_sha == expected_sha256,
            "H1 terminal registry is not the frozen 301-record parent",
        )
    _require(
        status.get("artifact_registry_sha256") == registry_sha
        and int(status.get("artifact_registry_size", -1))
        == int(registry_path.stat().st_size),
        "H1 status does not bind its terminal registry",
    )
    exclusions = set(registry.get("terminal_files_excluded_to_avoid_self_reference", []))
    _require(
        exclusions == {"artifact_registry.json", "run_status.json"},
        "H1 registry exclusions are incompatible",
    )
    raw_records = registry.get("records", {})
    _require(isinstance(raw_records, Mapping), "H1 registry records are invalid")
    records = dict(raw_records)
    _require(
        len(records) == int(expected_count),
        f"H1 terminal registry must contain exactly {expected_count} records",
    )
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in exclusions
    }
    _require(set(records) == actual, "H1 registry is incomplete or contains stale records")
    for relative, raw in records.items():
        _require(isinstance(raw, Mapping), f"invalid H1 registry record {relative}")
        _verify_record(
            dict(raw),
            root=run_dir,
            expected_path=run_dir / relative,
            description=f"registered H1 artifact {relative}",
        )
    return registry


def _verify_scientific_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw = manifest.get("scientific_config", {})
    _require(isinstance(raw, Mapping), "H1 scientific configuration is missing")
    scientific = dict(raw)
    _require(
        scientific.get("algorithm") == PARENT_RUN_SCHEMA
        and int(scientific.get("algorithm_version", -1)) == 1,
        "H1 parent algorithm is incompatible",
    )
    _require(
        config_fingerprint(scientific) == manifest.get("scientific_fingerprint"),
        "H1 scientific fingerprint is inconsistent",
    )
    kernel = scientific.get("kernel", {})
    _require(isinstance(kernel, Mapping), "H1 parent kernel is missing")
    for key, expected in EXPECTED_KERNEL.items():
        _require(_same(kernel.get(key), expected), f"H1 parent kernel mismatch for {key}")
    _require(
        scientific.get("thresholds") == _json_value(H1TrustThresholds().to_dict()),
        "H1 scientific thresholds changed",
    )
    for key, expected in {
        "model_schema": EXPECTED_MODEL_SCHEMA,
        "head_coordinate_version": EXPECTED_HEAD_COORDINATE_VERSION,
        "optimizer_coordinate_version": EXPECTED_OPTIMIZER_COORDINATE_VERSION,
        "paired_estimator_schema": "experiment12-d0-score-density-ratio-paired-mixture",
        "paired_objective_version": "d0-density-ratio-paired-mixture-weighted-softplus-v1",
        "paired_stream_version": "d0-density-ratio-paired-mixture-stream-v1",
        "paired_accumulation_version": "d0-density-ratio-deterministic-gradient-accumulation-v1",
        "h1_operator_version": EXPECTED_H1_OPERATOR_VERSION,
        "h1_calibration_version": EXPECTED_H1_CALIBRATION_VERSION,
        "h1_stream_version": EXPECTED_H1_STREAM_VERSION,
    }.items():
        _require(scientific.get(key) == expected, f"H1 schema mismatch for {key}")
    optimization = scientific.get("optimization", {})
    pilot = scientific.get("pilot", {})
    confirmation = scientific.get("confirmation", {})
    bootstrap = scientific.get("bootstrap", {})
    _require(
        isinstance(optimization, Mapping)
        and _same(optimization.get("body_learning_rate"), 3e-5)
        and _same(optimization.get("body_weight_decay"), 1e-4)
        and _same(optimization.get("ema_decay"), 0.99)
        and _same(optimization.get("global_grad_clip"), 1.0)
        and _same(optimization.get("loss_scale"), PARENT_LOSS_SCALE)
        and int(optimization.get("accumulation_steps", -1)) == 8
        and int(optimization.get("microbatch_clusters", -1)) == 32
        and int(optimization.get("trust_banks_per_update", -1)) == 2
        and int(optimization.get("trust_states_per_bank", -1)) == 32,
        "H1 optimizer configuration changed",
    )
    _require(
        isinstance(pilot, Mapping)
        and tuple(float(value) for value in pilot.get("h1_multipliers", []))
        == EXPECTED_MULTIPLIERS
        and int(pilot.get("model_seed", -1)) == EXPECTED_PILOT_MODEL_SEED
        and int(pilot.get("paths_per_panel", -1)) == 128
        and int(pilot.get("steps", -1)) == 4000,
        "H1 pilot configuration changed",
    )
    _require(
        isinstance(confirmation, Mapping)
        and tuple(int(value) for value in confirmation.get("model_seeds", []))
        == (261021, 261022, 261023)
        and int(confirmation.get("paths_per_panel", -1)) == 128
        and int(confirmation.get("steps", -1)) == 4000,
        "H1 planned confirmation configuration changed",
    )
    _require(
        isinstance(bootstrap, Mapping)
        and int(bootstrap.get("path_replicates", -1)) == 10_000
        and int(bootstrap.get("simultaneous_replicates", -1)) == 50_000
        and _same(bootstrap.get("path_confidence"), 0.9)
        and _same(bootstrap.get("familywise_confidence"), 0.95),
        "H1 bootstrap configuration changed",
    )
    _require(int(scientific.get("root_seed", -1)) == 261001, "H1 root seed changed")
    _no_work(scientific, "H1 scientific configuration")
    return scientific


def _verify_transitive_parent(parent: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        parent.get("schema") == PARENT_RUN_SCHEMA + "-parent-provenance"
        and int(parent.get("schema_version", -1)) == 1
        and _one(parent.get("passed")),
        "H1 multiplicity provenance is incompatible",
    )
    root = Path(str(parent.get("run_dir", ""))).resolve()
    recomputed = verify_parent_multiplicity_run(root)
    normalized = dict(parent)
    normalized.pop("verifier_source_fingerprint", None)
    normalized["schema"] = recomputed["schema"]
    _require(normalized == recomputed, "H1 transitive multiplicity provenance changed")
    _require(
        int(recomputed.get("terminal_registry_record_count", -1)) == 263
        and tuple(recomputed.get("lineage_registry_record_counts", []))
        == EXPECTED_LINEAGE_REGISTRY_RECORD_COUNTS[1:],
        "H1 parent does not bind exact 263/123/125/332/222/381 ancestry",
    )
    _no_work(parent, "H1 transitive parent")
    return recomputed


def _verify_zero_clipping(
    metrics: Mapping[str, Any],
    *,
    history_path: Path,
    windows_path: Path,
    description: str,
) -> None:
    names = (
        "post_warmup_clip_fraction",
        "final_500_clip_fraction",
        "final_200_clip_fraction",
    )
    diagnostics = metrics.get("optimization_diagnostics", {})
    _require(isinstance(diagnostics, Mapping), f"{description} diagnostics are missing")
    for name in names:
        _require(
            _finite(metrics.get(name)) and float(metrics[name]) == 0.0,
            f"{description} does not record zero {name}",
        )
        _require(
            _finite(diagnostics.get(name)) and float(diagnostics[name]) == 0.0,
            f"{description} diagnostics do not record zero {name}",
        )
    windows = diagnostics.get("clipping_windows", [])
    _require(
        isinstance(windows, Sequence)
        and bool(windows)
        and all(
            isinstance(row, Mapping)
            and _finite(row.get("clip_fraction"))
            and float(row["clip_fraction"]) == 0.0
            for row in windows
        ),
        f"{description} contains invalid or nonzero clipping windows",
    )
    _require(
        history_path.is_file() and windows_path.is_file(),
        f"{description} clipping CSV artifacts are missing",
    )
    try:
        with history_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        with windows_path.open("r", encoding="utf-8", newline="") as handle:
            window_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise ArtifactCompatibilityError(f"cannot read {description} clipping CSV: {exc}") from exc
    _require(len(rows) == 4000, f"{description} history must contain 4000 updates")
    _require(
        all(_finite(row.get("clipped")) and float(row["clipped"]) == 0.0 for row in rows),
        f"{description} history contains clipping",
    )
    _require(
        bool(window_rows)
        and all(
            _finite(row.get("clip_fraction")) and float(row["clip_fraction"]) == 0.0
            for row in window_rows
        ),
        f"{description} clipping-window CSV contains clipping",
    )


def _verify_task(
    aggregate: Mapping[str, Any],
    *,
    task_dir: Path,
    expected_task: str,
    expected_multiplier: float,
    manifest: Mapping[str, Any],
) -> int:
    result_path = task_dir / "task_result.json"
    status_path = task_dir / "task_status.json"
    _require(result_path.is_file() and status_path.is_file(), f"missing {expected_task} task artifacts")
    result = _load(result_path)
    status = _load(status_path)
    _require(result == dict(aggregate), f"{expected_task} aggregate/task result mismatch")
    _require(
        result.get("schema") == "experiment12-d0-score-density-ratio-h1-task-result"
        and int(result.get("schema_version", -1)) == 1
        and result.get("task") == expected_task
        and _same(result.get("h1_ratio"), expected_multiplier),
        f"{expected_task} task result identity changed",
    )
    _require(
        status.get("schema") == "experiment12-d0-score-density-ratio-h1-task-status"
        and int(status.get("schema_version", -1)) == 1
        and status.get("status") == "complete"
        and status.get("task") == expected_task
        and int(status.get("training_step", -1)) == 4000
        and _same(status.get("h1_ratio"), expected_multiplier)
        and status.get("task_result_sha256") == file_fingerprint(result_path),
        f"{expected_task} task status is incomplete or unbound",
    )
    metrics = result.get("metrics", {})
    _require(isinstance(metrics, Mapping), f"{expected_task} task metrics are missing")
    _require(
        _one(metrics.get("complete"))
        and _one(metrics.get("finite"))
        and _one(metrics.get("boundary_admissible"))
        and _one(metrics.get("h1_health_pass")),
        f"{expected_task} task is incomplete, nonfinite, boundary-inadmissible, or H1-unhealthy",
    )
    gate = result.get("gate", {})
    _require(isinstance(gate, Mapping) and _one(gate.get("passed")), f"{expected_task} task gate failed")
    fingerprints = result.get("fingerprints", {})
    _require(isinstance(fingerprints, Mapping), f"{expected_task} fingerprints are missing")
    _require(
        fingerprints.get("phase") == "h1-pilot"
        and fingerprints.get("task") == expected_task
        and int(fingerprints.get("model_seed", -1)) == EXPECTED_PILOT_MODEL_SEED
        and int(fingerprints.get("accumulation_level", -1)) == 8
        and _same(fingerprints.get("learning_rate"), 3e-5)
        and _same(fingerprints.get("loss_scale"), PARENT_LOSS_SCALE)
        and _same(fingerprints.get("h1_ratio"), expected_multiplier)
        and _same(fingerprints.get("h1_multiplier"), expected_multiplier)
        and _same(
            fingerprints.get("h1_effective_multiplier"),
            expected_multiplier * EXPECTED_H1_LAMBDA_BASE,
        )
        and _same(fingerprints.get("h1_lambda_base"), EXPECTED_H1_LAMBDA_BASE)
        and _same(fingerprints.get("h1_value_scale"), EXPECTED_H1_VALUE_SCALE)
        and _same(fingerprints.get("h1_energy_scale"), EXPECTED_H1_ENERGY_SCALE)
        and fingerprints.get("h1_operator_version") == EXPECTED_H1_OPERATOR_VERSION
        and fingerprints.get("h1_calibration_version") == EXPECTED_H1_CALIBRATION_VERSION
        and fingerprints.get("h1_stream_version") == EXPECTED_H1_STREAM_VERSION
        and fingerprints.get("h1_task_training_version") == EXPECTED_H1_TASK_VERSION
        and fingerprints.get("model_schema") == EXPECTED_MODEL_SCHEMA
        and fingerprints.get("head_coordinate_version") == EXPECTED_HEAD_COORDINATE_VERSION
        and fingerprints.get("optimizer_coordinate_version")
        == EXPECTED_OPTIMIZER_COORDINATE_VERSION
        and fingerprints.get("scientific_fingerprint")
        == manifest.get("scientific_fingerprint")
        and fingerprints.get("runtime_fingerprint") == manifest.get("runtime_fingerprint")
        and fingerprints.get("source_fingerprint") == manifest.get("source_fingerprint")
        and dict(status.get("fingerprints", {})) == dict(fingerprints),
        f"{expected_task} task fingerprints changed",
    )
    panels = fingerprints.get("selection_panel_identities", {})
    _require(
        isinstance(panels, Mapping)
        and set(panels) == {"a", "b"}
        and all(
            isinstance(value, Mapping)
            and int(value.get("path_count", -1)) == 128
            and value.get("task") == expected_task
            and value.get("role") == role
            for role, value in panels.items()
        )
        and fingerprints.get("audit_panel_identities") is None,
        f"{expected_task} pilot panel fingerprints changed",
    )
    model_seed = int(result.get("model_seed", -1))
    _require(
        model_seed == EXPECTED_PILOT_MODEL_SEED
        and int(status.get("model_seed", -2)) == model_seed,
        f"{expected_task} model seed changed",
    )
    diagnostics = metrics.get("h1_diagnostics", {})
    _require(
        isinstance(diagnostics, Mapping)
        and _one(diagnostics.get("h1_health_pass"))
        and _same(diagnostics.get("h1_ratio"), expected_multiplier)
        and _same(diagnostics.get("lambda_base"), EXPECTED_H1_LAMBDA_BASE)
        and _same(diagnostics.get("value_scale"), EXPECTED_H1_VALUE_SCALE)
        and _same(diagnostics.get("energy_scale"), EXPECTED_H1_ENERGY_SCALE)
        and diagnostics.get("operator_version") == EXPECTED_H1_OPERATOR_VERSION,
        f"{expected_task} H1 diagnostics changed",
    )
    _verify_zero_clipping(
        dict(metrics),
        history_path=task_dir / "training_history.csv",
        windows_path=task_dir / "clipping_windows.csv",
        description=f"{task_dir.name} {expected_task}",
    )
    _no_work(result, f"{expected_task} task result")
    _no_work(status, f"{expected_task} task status")
    return model_seed


def _verify_derivative_only_failure(pilot: Mapping[str, Any]) -> None:
    gates = pilot.get("candidate_gates", [])
    _require(isinstance(gates, Sequence) and len(gates) == 4, "H1 pilot candidate gates are invalid")
    values = [dict(value) for value in gates if isinstance(value, Mapping)]
    _require(
        len(values) == 4
        and tuple(float(value.get("multiplier", math.nan)) for value in values)
        == EXPECTED_MULTIPLIERS,
        "H1 pilot gate multiplier ordering changed",
    )
    baseline = values[0]
    _require(
        _one(baseline.get("passed"))
        and _one(baseline.get("optimizer_health_pass"))
        and _one(baseline.get("classification_pass"))
        and _one(baseline.get("null_pass")),
        "H1 zero-ratio baseline no longer passes its defined checks",
    )
    expected_nonzero_checks = {
        "accumulation_steps",
        "base_channels",
        "known_multiplier",
        "learning_rate",
        "optimizer_and_task_health",
        "relative_l2_reduction_overall_and_data_end",
        "stationary_null",
        "strict_derivative_thresholds",
        "teacher_classification",
    }
    for value in values[1:]:
        subchecks = value.get("subchecks", {})
        _require(isinstance(subchecks, Mapping), "H1 nonzero candidate subchecks are missing")
        failed = {
            str(name)
            for name, check in subchecks.items()
            if not isinstance(check, Mapping) or not _one(check.get("passed"))
        }
        _require(
            set(subchecks) == expected_nonzero_checks
            and failed == {"strict_derivative_thresholds"}
            and _zero(value.get("passed"))
            and _one(value.get("optimizer_health_pass"))
            and _one(value.get("classification_pass"))
            and _one(value.get("null_pass"))
            and _zero(value.get("derivative_pass"))
            and _one(value.get("relative_l2_reduction_pass")),
            "H1 nonzero candidate failure is not restricted to strict derivative thresholds",
        )


def verify_parent_h1_trust_run(path: str | Path) -> dict[str, Any]:
    """Verify and normalize the exact 301-record unresolved H1 pilot."""

    run_dir = Path(path).resolve()
    _require(run_dir.is_dir(), f"H1 parent run does not exist: {run_dir}")
    paths = {name: run_dir / relative for name, relative in _REQUIRED_FILES.items()}
    missing = [str(value) for value in paths.values() if not value.is_file()]
    _require(not missing, "H1 parent artifacts are missing: " + ", ".join(missing))
    values = {name: _load(value) for name, value in paths.items() if name != "registry"}
    manifest, status = values["manifest"], values["status"]
    _require(
        manifest.get("schema") == status.get("schema") == PARENT_RUN_SCHEMA
        and int(manifest.get("schema_version", -1)) == 1
        and int(status.get("schema_version", -1)) == 1,
        "H1 parent schema is incompatible",
    )
    _require(
        Path(str(manifest.get("run_dir", ""))).resolve() == run_dir,
        "H1 manifest run directory is inconsistent",
    )
    _require(
        status.get("status") == "complete"
        and status.get("outcome") == "gate_failed"
        and status.get("phase") == "pilot"
        and status.get("stage") == "pilot"
        and status.get("decision") == EXPECTED_PARENT_DECISION
        and status.get("required_gate") == "pilot"
        and _zero(status.get("required_gate_pass")),
        "H1 parent is not the recorded unresolved pilot",
    )
    skips = status.get("skips", [])
    _require(
        isinstance(skips, Sequence)
        and len(skips) == 1
        and isinstance(skips[0], Mapping)
        and skips[0].get("stage") == "confirmation"
        and skips[0].get("reason") == "H1 pilot failed",
        "H1 parent did not explicitly skip confirmation",
    )
    _require(
        not (run_dir / "confirmation").exists()
        and not (run_dir / "selected_h1_profile.json").exists()
        and not (run_dir / "confirmation_panel_registry.json").exists(),
        "H1 parent contains a selected profile or confirmation evidence",
    )
    registry = _verify_registry(
        run_dir,
        registry_path=paths["registry"],
        status_path=paths["status"],
    )
    scientific = _verify_scientific_manifest(manifest)
    source_paths = [Path(str(value)).resolve() for value in manifest.get("source_paths", [])]
    _require(
        len(source_paths) == 21
        and all(value.is_file() for value in source_paths)
        and source_fingerprint(source_paths) == manifest.get("source_fingerprint"),
        "H1 parent source fingerprint no longer matches",
    )
    _require(
        manifest.get("parent_provenance_sha256")
        == file_fingerprint(paths["parent_provenance"]),
        "H1 manifest does not bind multiplicity provenance",
    )
    transitive = _verify_transitive_parent(values["parent_provenance"])
    _require(
        scientific.get("parent_artifact_registry_sha256")
        == transitive.get("artifact_registry_sha256")
        and scientific.get("parent_scientific_fingerprint")
        == transitive.get("scientific_fingerprint"),
        "H1 scientific configuration does not bind its parent",
    )

    operator_gate = _json_value(evaluate_h1_operator_preflight(values["operator_raw"]))
    calibration_gate = _json_value(evaluate_h1_calibration(values["calibration_raw"]))
    _require(
        operator_gate == values["operator_gate"] and _one(operator_gate.get("passed")),
        "H1 operator gate does not recompute as passing",
    )
    _require(
        calibration_gate == values["calibration_gate"]
        and _one(calibration_gate.get("passed")),
        "H1 calibration gate does not recompute as passing",
    )
    calibration = values["calibration_raw"]
    _require(
        _same(calibration.get("lambda_base"), EXPECTED_H1_LAMBDA_BASE)
        and _same(calibration.get("value_scale"), EXPECTED_H1_VALUE_SCALE)
        and _same(calibration.get("energy_scale"), EXPECTED_H1_ENERGY_SCALE),
        "H1 calibration values changed",
    )
    inherited = values["preflight_gate"].get("inherited_preflight", {})
    _require(isinstance(inherited, Mapping) and _one(inherited.get("passed")), "inherited preflight did not pass")
    preflight_gate = _json_value(
        evaluate_h1_preflight(
            inherited_preflight=dict(inherited),
            operator=operator_gate,
            calibration=calibration_gate,
        )
    )
    _require(
        preflight_gate == values["preflight_gate"]
        and _one(preflight_gate.get("passed")),
        "H1 combined preflight gate does not recompute as passing",
    )
    oracle = values["pilot_oracle"]
    panel_gates = oracle.get("panel_gates", [])
    _require(
        _one(oracle.get("passed"))
        and oracle.get("evaluation_status") == "evaluated"
        and tuple(oracle.get("expected_roles", [])) == ("a", "b")
        and isinstance(panel_gates, Mapping)
        and set(panel_gates) == {"a", "b"}
        and all(
            isinstance(value, Mapping) and _one(value.get("passed"))
            for value in panel_gates.values()
        ),
        "H1 pilot panels were not oracle-qualified",
    )
    panel_registry = values["pilot_panel_registry"]
    disjointness = panel_registry.get("disjointness", {})
    _require(
        _one(panel_registry.get("passed"))
        and _zero(panel_registry.get("panel_regeneration_after_inspection"))
        and list(panel_registry.get("parent_overlap_stream_fingerprints", [])) == []
        and list(panel_registry.get("previous_phase_overlap_stream_fingerprints", [])) == []
        and isinstance(disjointness, Mapping)
        and _one(disjointness.get("passed"))
        and int(disjointness.get("panel_count", -1)) == 4
        and list(disjointness.get("overlaps", [])) == [],
        "H1 pilot panels are not fresh and frozen",
    )

    candidate_wrapper = values["pilot_candidates"]
    raw_candidates = candidate_wrapper.get("candidates", [])
    _require(isinstance(raw_candidates, Sequence) and len(raw_candidates) == 4, "H1 pilot candidates are invalid")
    candidates = [dict(value) for value in raw_candidates if isinstance(value, Mapping)]
    _require(len(candidates) == 4, "H1 pilot candidates are malformed")
    common_task_bindings: dict[str, tuple[Any, ...]] = {}
    for index, (candidate, multiplier) in enumerate(zip(candidates, EXPECTED_MULTIPLIERS, strict=True)):
        _require(
            _same(candidate.get("multiplier"), multiplier)
            and _one(candidate.get("complete"))
            and _one(candidate.get("finite"))
            and _one(candidate.get("boundary_admissible"))
            and _one(candidate.get("optimizer_health_pass"))
            and _one(candidate.get("h1_health_pass"))
            and _finite(candidate.get("maximum_clip_fraction_observed"))
            and float(candidate["maximum_clip_fraction_observed"]) == 0.0,
            f"H1 pilot candidate {multiplier:g} is incomplete or unhealthy",
        )
        seeds: list[int] = []
        for key, task in (("teacher", "bounded_teacher"), ("null", "dirichlet_null")):
            aggregate = candidate.get(key, {})
            _require(isinstance(aggregate, Mapping), f"H1 {task} aggregate is missing")
            task_dir = run_dir / "pilot" / f"q-{index:02d}" / task
            seeds.append(
                _verify_task(
                    dict(aggregate),
                    task_dir=task_dir,
                    expected_task=task,
                    expected_multiplier=multiplier,
                    manifest=manifest,
                )
            )
            fingerprints = dict(aggregate.get("fingerprints", {}))
            binding = (
                fingerprints.get("paired_stream_plan_fingerprint"),
                fingerprints.get("stream_plan_fingerprint"),
                fingerprints.get("h1_trust_plan_fingerprint"),
                config_fingerprint(fingerprints.get("selection_panel_identities", {})),
            )
            if task in common_task_bindings:
                _require(
                    common_task_bindings[task] == binding,
                    f"H1 {task} streams or panels differ across multipliers",
                )
            else:
                common_task_bindings[task] = binding
        _require(seeds == [EXPECTED_PILOT_MODEL_SEED] * 2, "H1 teacher/null initialization is not paired")
    _no_work(candidate_wrapper, "H1 pilot candidate wrapper")

    failures = values["pilot_failures"]
    _require(
        int(failures.get("count", -1)) == 0 and list(failures.get("failures", [])) == [],
        "H1 pilot contains task failures",
    )
    null_gate = values["pilot_null_gate"]
    max_t = values["pilot_null_max_t"]
    _require(
        _one(null_gate.get("passed"))
        and _zero(null_gate.get("familywise_false_discovery"))
        and _one(max_t.get("finite"))
        and _one(max_t.get("valid"))
        and int(max_t.get("family_size", -1)) == 8
        and int(max_t.get("bootstrap_replicates", -1)) == 50_000
        and _same(max_t.get("confidence"), 0.95)
        and _zero(max_t.get("familywise_false_discovery")),
        "H1 simultaneous stationary-null family did not pass",
    )
    recomputed_pilot = _json_value(
        evaluate_h1_pilot(candidates, panel_power=oracle, null_family=null_gate)
    )
    _require(recomputed_pilot == values["pilot_gate"], "H1 pilot gate does not recompute")
    pilot = values["pilot_gate"]
    _verify_derivative_only_failure(pilot)
    _require(
        _zero(pilot.get("passed"))
        and _one(pilot.get("optimizer_health_pass"))
        and _zero(pilot.get("overregularized"))
        and _one(pilot.get("null_family_pass"))
        and _zero(dict(pilot.get("selected_profile", {})).get("selected")),
        "H1 pilot terminal semantics changed",
    )

    workflow = values["workflow"]
    components = workflow.get("components", {})
    _require(isinstance(components, Mapping), "H1 workflow components are missing")
    controls = components.get("controls", {})
    _require(isinstance(controls, Mapping), "H1 skipped controls record is missing")
    confirmation_power = components.get("confirmation_panel_power", {})
    teacher_study = controls.get("teacher_study", {})
    confirmation_null = controls.get("null_family", {})
    _require(
        isinstance(confirmation_power, Mapping)
        and confirmation_power.get("evaluation_status") == "not_evaluated"
        and isinstance(teacher_study, Mapping)
        and teacher_study.get("evaluation_status") == "not_evaluated"
        and isinstance(confirmation_null, Mapping)
        and confirmation_null.get("evaluation_status") == "not_evaluated",
        "H1 skipped confirmation is not fail-closed",
    )
    recomputed_workflow = _json_value(
        evaluate_h1_workflow(
            provenance=values["parent_provenance"],
            operator=operator_gate,
            calibration=calibration_gate,
            preflight=preflight_gate,
            pilot_panel_power=oracle,
            pilot=pilot,
            confirmation_panel_power=dict(confirmation_power),
            teacher_study=dict(teacher_study),
            null_family=dict(confirmation_null),
            require_gate="pilot",
        )
    )
    _require(recomputed_workflow == workflow, "H1 terminal workflow does not recompute")
    decision = values["decision"]
    _require(
        dict(workflow.get("decision", {})) == decision
        and decision.get("decision") == EXPECTED_PARENT_DECISION
        and _zero(decision.get("physical_training_authorized"))
        and _zero(decision.get("sampling_authorized"))
        and _zero(workflow.get("required_gate_pass")),
        "H1 terminal decision changed",
    )
    for name, value in values.items():
        _no_work(value, f"H1 {name}")
    _require(
        _zero(status.get("physical_training_authorized"))
        and _zero(status.get("sampling_authorized")),
        "H1 parent authorized physical training or sampling",
    )
    schedule = transitive.get("schedule_metadata", {})
    _require(isinstance(schedule, Mapping), "H1 parent schedule is missing")
    horizon = schedule.get("horizon", transitive.get("horizon"))
    _require(_finite(horizon) and float(horizon) > 0.0, "H1 parent horizon is invalid")
    return {
        "schema": PARENT_RUN_SCHEMA + "-gradient-control-parent-provenance",
        "schema_version": 1,
        "passed": 1,
        "run_dir": str(run_dir),
        "scientific_fingerprint": str(manifest.get("scientific_fingerprint", "")),
        "runtime_fingerprint": str(manifest.get("runtime_fingerprint", "")),
        "source_fingerprint": str(manifest.get("source_fingerprint", "")),
        "scientific_config": scientific,
        "kernel": dict(scientific["kernel"]),
        "model": {
            "schema": EXPECTED_MODEL_SCHEMA,
            "base_channels": 32,
            "head_coordinate_version": EXPECTED_HEAD_COORDINATE_VERSION,
            "optimizer_coordinate_version": EXPECTED_OPTIMIZER_COORDINATE_VERSION,
        },
        "schedule_metadata": dict(schedule),
        "horizon": float(horizon),
        "parent_decision": EXPECTED_PARENT_DECISION,
        "terminal_registry_record_count": len(dict(registry["records"])),
        "artifact_registry_sha256": file_fingerprint(paths["registry"]),
        "artifact_registry_size": int(paths["registry"].stat().st_size),
        "parent_loss_scale": PARENT_LOSS_SCALE,
        "h1_operator_version": EXPECTED_H1_OPERATOR_VERSION,
        "h1_calibration_version": EXPECTED_H1_CALIBRATION_VERSION,
        "h1_stream_version": EXPECTED_H1_STREAM_VERSION,
        "h1_value_scale": EXPECTED_H1_VALUE_SCALE,
        "h1_energy_scale": EXPECTED_H1_ENERGY_SCALE,
        "h1_lambda_base": EXPECTED_H1_LAMBDA_BASE,
        "h1_calibration_fingerprint": config_fingerprint(calibration),
        "h1_calibration": calibration,
        "preflight_pass": 1,
        "operator_pass": 1,
        "calibration_pass": 1,
        "pilot_oracle_pass": 1,
        "pilot_pass": 0,
        "pilot_task_count": 8,
        "all_tasks_complete_finite_boundary_admissible": 1,
        "all_task_clipping_zero": 1,
        "all_teacher_classification_pass": 1,
        "all_null_pass": 1,
        "nonzero_derivative_passing_count": 0,
        "derivative_only_failure": 1,
        "selected_profile_present": 0,
        "confirmation_performed": 0,
        "task_failure_count": 0,
        "lineage_registry_record_counts": list(EXPECTED_LINEAGE_REGISTRY_RECORD_COUNTS),
        "transitive_parent_provenance": values["parent_provenance"],
        "artifacts": {name: _record(value) for name, value in paths.items()},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def verify_parent_h1_run(path: str | Path) -> dict[str, Any]:
    """Compatibility alias for callers that omit ``trust`` from the name."""

    return verify_parent_h1_trust_run(path)
