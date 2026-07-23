"""Strict parent provenance for the D0 multiplicity confirmation.

The admissible parent is the immutable 128-path selection-power pilot that
ended ``null_gate_multiplicity_inconclusive``.  The verifier binds its exact
123-record terminal registry, recomputes the saved oracle/preflight and pilot
gates, verifies every task and clipping record, and recursively verifies the
stored 125 -> 332 -> 222 -> 381 ancestry.

This module is additive.  In particular, none of the sources fingerprinted by
the completed parent or its ancestors need to change in order to verify it.
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
from mnist.d0_score_density_ratio_head_provenance import (
    EXPECTED_KERNEL,
    PARENT_LOSS_SCALE,
)
from mnist.d0_score_density_ratio_selection_power_gate import (
    SelectionPowerThresholds,
    evaluate_oracle_panel_set,
    evaluate_power_pilot,
    evaluate_selection_power_preflight,
    evaluate_selection_power_workflow,
)
from mnist.d0_score_density_ratio_selection_power_provenance import (
    verify_parent_normalized_head_run,
)


__all__ = [
    "PARENT_RUN_SCHEMA",
    "PARENT_REGISTRY_SCHEMA",
    "PARENT_REGISTRY_RECORD_COUNT",
    "EXPECTED_PARENT_DECISION",
    "EXPECTED_KERNEL",
    "PARENT_LOSS_SCALE",
    "verify_parent_selection_power_run",
]


PARENT_RUN_SCHEMA = (
    "experiment12-d0-score-density-ratio-selection-power-confirmation"
)
PARENT_REGISTRY_SCHEMA = PARENT_RUN_SCHEMA + "-artifact-registry"
PARENT_REGISTRY_RECORD_COUNT = 123
EXPECTED_PARENT_DECISION = "null_gate_multiplicity_inconclusive"
EXPECTED_LEARNING_RATES = (3e-5, 1e-5)
EXPECTED_ACCUMULATION = 8
EXPECTED_MODEL_SCHEMA = "d0-boundary-smooth-potential-unet-mean-head-v2"
EXPECTED_HEAD_COORDINATE_VERSION = "d0-spatial-sum-to-mean-head-coordinate-v1"
EXPECTED_OPTIMIZER_COORDINATE_VERSION = "d0-mean-head-coordinate-adamw-v1"

_REQUIRED_FILES: dict[str, str] = {
    "manifest": "run_manifest.json",
    "status": "run_status.json",
    "registry": "artifact_registry.json",
    "parent_provenance": "parent_provenance.json",
    "stream_plans": "selection_power_stream_plans.json",
    "saved_forensic": "saved_16_path_oracle_forensic.json",
    "oracle_calibration": "oracle_power_calibration.json",
    "preflight_gate": "selection_power_preflight_gate.json",
    "pilot_panel_registry": "pilot_panel_registry.json",
    "pilot_oracle": "pilot_oracle_feasibility.json",
    "pilot_gate": "selection_power_pilot_gate.json",
    "pilot_multiplicity": "pilot_null_multiplicity_analysis.json",
    "pilot_failures": "pilot_task_failures.json",
    "workflow": "selection_power_control_gate.json",
    "decision": "selection_power_decision.json",
}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"cannot read selection-power parent artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(
            f"selection-power parent artifact is not a JSON object: {path}"
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
) -> dict[str, Any]:
    """Verify terminal registry integrity and exact file-set equality."""

    registry = _load(registry_path)
    status = _load(status_path)
    _require(
        registry.get("schema") == PARENT_REGISTRY_SCHEMA
        and int(registry.get("schema_version", -1)) == 1,
        "selection-power terminal registry schema is incompatible",
    )
    _require(
        status.get("artifact_registry_sha256") == file_fingerprint(registry_path)
        and int(status.get("artifact_registry_size", -1))
        == int(registry_path.stat().st_size),
        "selection-power status does not bind its terminal registry",
    )
    exclusions = set(registry.get("terminal_files_excluded_to_avoid_self_reference", []))
    _require(
        exclusions == {"artifact_registry.json", "run_status.json"},
        "selection-power registry exclusions are incompatible",
    )
    raw_records = registry.get("records", {})
    _require(isinstance(raw_records, Mapping), "selection-power registry records are invalid")
    records = dict(raw_records)
    _require(
        len(records) == int(expected_count),
        f"selection-power terminal registry must contain exactly {expected_count} records",
    )
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in exclusions
    }
    _require(
        set(records) == actual,
        "selection-power registry is incomplete or contains stale records",
    )
    for relative, raw in records.items():
        _require(isinstance(raw, Mapping), f"invalid registry record {relative}")
        _verify_record(
            dict(raw),
            root=run_dir,
            expected_path=run_dir / relative,
            description=f"registered selection-power artifact {relative}",
        )
    return registry


def _verify_scientific_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw = manifest.get("scientific_config", {})
    _require(isinstance(raw, Mapping), "selection-power scientific configuration is missing")
    scientific = dict(raw)
    _require(
        scientific.get("algorithm") == PARENT_RUN_SCHEMA
        and int(scientific.get("algorithm_version", -1)) == 1,
        "selection-power parent algorithm is incompatible",
    )
    _require(
        config_fingerprint(scientific) == manifest.get("scientific_fingerprint"),
        "selection-power scientific fingerprint is inconsistent",
    )
    kernel = scientific.get("kernel", {})
    _require(isinstance(kernel, Mapping), "selection-power kernel is missing")
    for key, expected in EXPECTED_KERNEL.items():
        _require(_same(kernel.get(key), expected), f"selection-power kernel mismatch for {key}")
    _require(
        scientific.get("thresholds")
        == _json_value(SelectionPowerThresholds().to_dict()),
        "selection-power scientific thresholds changed",
    )
    for key, expected in {
        "model_schema": EXPECTED_MODEL_SCHEMA,
        "head_coordinate_version": EXPECTED_HEAD_COORDINATE_VERSION,
        "optimizer_coordinate_version": EXPECTED_OPTIMIZER_COORDINATE_VERSION,
        "paired_estimator_schema": "experiment12-d0-score-density-ratio-paired-mixture",
        "paired_objective_version": "d0-density-ratio-paired-mixture-weighted-softplus-v1",
        "paired_stream_version": "d0-density-ratio-paired-mixture-stream-v1",
        "paired_accumulation_version": "d0-density-ratio-deterministic-gradient-accumulation-v1",
    }.items():
        _require(scientific.get(key) == expected, f"selection-power schema mismatch for {key}")
    _require(
        _same(scientific.get("loss_scale"), PARENT_LOSS_SCALE),
        "selection-power loss multiplier changed",
    )
    pilot = scientific.get("pilot", {})
    confirmation = scientific.get("confirmation", {})
    oracle = scientific.get("oracle_power", {})
    optimization = scientific.get("optimization", {})
    _require(
        isinstance(pilot, Mapping)
        and tuple(float(value) for value in pilot.get("learning_rates", []))
        == EXPECTED_LEARNING_RATES
        and tuple(int(value) for value in pilot.get("accumulation_levels", []))
        == (EXPECTED_ACCUMULATION,)
        and int(pilot.get("steps", -1)) == 2000
        and int(pilot.get("paths_per_panel", -1)) == 128
        and tuple(int(value) for value in pilot.get("validation_steps", []))
        == (0, 25, 50, 100, 150, 250, 500, 750, 1000, 1500, 2000),
        "selection-power pilot configuration changed",
    )
    _require(
        isinstance(confirmation, Mapping)
        and tuple(int(value) for value in confirmation.get("model_seeds", []))
        == (260941, 260942, 260943)
        and int(confirmation.get("steps", -1)) == 4000
        and int(confirmation.get("paths_per_selection_panel", -1)) == 128
        and int(confirmation.get("paths_per_audit_panel", -1)) == 128,
        "selection-power planned confirmation configuration changed",
    )
    _require(
        isinstance(oracle, Mapping)
        and int(oracle.get("bootstrap_reps", -1)) == 10000
        and int(oracle.get("calibration_paths", -1)) == 256
        and int(oracle.get("predetermined_half_paths", -1)) == 128
        and _same(oracle.get("calibration_confidence"), 0.99)
        and _same(oracle.get("evaluation_confidence"), 0.9)
        and _zero(oracle.get("panel_regeneration_after_inspection")),
        "selection-power oracle-power configuration changed",
    )
    _require(
        isinstance(optimization, Mapping)
        and optimization.get("optimizer") == "coordinate-conjugate-AdamW"
        and _same(optimization.get("body_weight_decay"), 1e-4)
        and _same(optimization.get("ema_decay"), 0.99)
        and _same(optimization.get("grad_clip"), 1.0)
        and int(optimization.get("clip_warmup_steps", -1)) == 500
        and _zero(optimization.get("adaptive_loss_scaling"))
        and optimization.get("gradient_accumulation") == "mean-then-clip-once",
        "selection-power optimizer configuration changed",
    )
    _require(int(scientific.get("root_seed", -1)) == 260931, "selection-power root seed changed")
    _no_work(scientific, "selection-power scientific configuration")
    return scientific


def _verify_transitive_parent(parent: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        parent.get("schema") == PARENT_RUN_SCHEMA + "-parent-provenance"
        and int(parent.get("schema_version", -1)) == 1
        and _one(parent.get("passed")),
        "selection-power normalized-head provenance is incompatible",
    )
    root = Path(str(parent.get("run_dir", ""))).resolve()
    recomputed = verify_parent_normalized_head_run(root)
    normalized = dict(parent)
    normalized.pop("verifier_source_fingerprint", None)
    normalized["schema"] = recomputed["schema"]
    _require(
        normalized == recomputed,
        "selection-power transitive normalized-head provenance changed",
    )
    parent_332 = dict(recomputed.get("transitive_parent_provenance", {}))
    parent_222 = dict(parent_332.get("transitive_parent_provenance", {}))
    parent_381 = dict(parent_222.get("transitive_parent_provenance", {}))
    _require(
        int(recomputed.get("terminal_registry_record_count", -1)) == 125
        and int(parent_332.get("terminal_registry_record_count", -1)) == 332
        and int(parent_222.get("terminal_registry_record_count", -1)) == 222
        and int(parent_381.get("terminal_registry_record_count", -1)) == 381,
        "selection-power parent does not bind exact 125/332/222/381 ancestry",
    )
    _no_work(parent, "selection-power transitive parent")
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
    for name in names:
        _require(
            _finite(metrics.get(name)) and float(metrics[name]) == 0.0,
            f"{description} does not record zero {name}",
        )
    diagnostics = metrics.get("optimization_diagnostics", {})
    _require(isinstance(diagnostics, Mapping), f"{description} optimization diagnostics are missing")
    for name in names:
        _require(
            _finite(diagnostics.get(name)) and float(diagnostics[name]) == 0.0,
            f"{description} diagnostics do not record zero {name}",
        )
    windows = diagnostics.get("clipping_windows", [])
    _require(
        isinstance(windows, Sequence)
        and len(windows) > 0
        and all(
            isinstance(value, Mapping)
            and _finite(value.get("clip_fraction"))
            and float(value["clip_fraction"]) == 0.0
            for value in windows
        ),
        f"{description} contains invalid or nonzero clipping windows",
    )
    _require(history_path.is_file() and windows_path.is_file(), f"{description} clipping CSV artifacts are missing")
    try:
        with history_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        with windows_path.open("r", encoding="utf-8", newline="") as handle:
            window_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise ArtifactCompatibilityError(f"cannot read {description} clipping CSV: {exc}") from exc
    _require(len(rows) == 2000, f"{description} training history must contain 2000 updates")
    _require(
        all(_finite(row.get("clipped")) and float(row["clipped"]) == 0.0 for row in rows),
        f"{description} training history contains clipping",
    )
    _require(
        bool(window_rows)
        and all(
            _finite(row.get("clip_fraction")) and float(row["clip_fraction"]) == 0.0
            for row in window_rows
        ),
        f"{description} clipping-window CSV contains clipping",
    )


def _verify_task_result(
    aggregate: Mapping[str, Any],
    *,
    task_dir: Path,
    expected_task: str,
    expected_learning_rate: float,
    manifest: Mapping[str, Any],
) -> int:
    result_path = task_dir / "task_result.json"
    status_path = task_dir / "task_status.json"
    _require(result_path.is_file() and status_path.is_file(), f"missing {expected_task} task artifacts")
    result = _load(result_path)
    status = _load(status_path)
    _require(result == dict(aggregate), f"{expected_task} aggregate/task result mismatch")
    _require(
        result.get("schema")
        == "experiment12-d0-score-density-ratio-head-confirmation-task-result"
        and int(result.get("schema_version", -1)) == 1
        and status.get("schema")
        == "experiment12-d0-score-density-ratio-head-confirmation-task-status"
        and int(status.get("schema_version", -1)) == 1,
        f"{expected_task} task schema is incompatible",
    )
    _require(
        status.get("status") == "complete"
        and status.get("task") == expected_task
        and status.get("task_result_sha256") == file_fingerprint(result_path)
        and int(status.get("training_step", -1)) == 2000,
        f"{expected_task} task status is incomplete or unbound",
    )
    metrics = result.get("metrics", {})
    _require(isinstance(metrics, Mapping), f"{expected_task} metrics are missing")
    _require(
        _one(metrics.get("complete"))
        and _one(metrics.get("finite"))
        and _one(metrics.get("boundary_admissible")),
        f"{expected_task} task is incomplete, nonfinite, or boundary-inadmissible",
    )
    gate = result.get("gate", {})
    _require(isinstance(gate, Mapping) and _one(gate.get("passed")), f"{expected_task} task gate failed")
    fingerprints = result.get("fingerprints", {})
    _require(isinstance(fingerprints, Mapping), f"{expected_task} fingerprints are missing")
    _require(
        fingerprints.get("schema")
        == "experiment12-d0-score-density-ratio-head-confirmation-task-fingerprints"
        and int(fingerprints.get("schema_version", -1)) == 1
        and int(fingerprints.get("accumulation_level", -1)) == EXPECTED_ACCUMULATION
        and _same(fingerprints.get("learning_rate"), expected_learning_rate)
        and _same(fingerprints.get("loss_scale"), PARENT_LOSS_SCALE)
        and fingerprints.get("task") == expected_task
        and fingerprints.get("phase") == "selection-power-pilot"
        and fingerprints.get("selection_power_workflow_schema") == PARENT_RUN_SCHEMA
        and fingerprints.get("model_schema") == EXPECTED_MODEL_SCHEMA
        and fingerprints.get("head_coordinate_version") == EXPECTED_HEAD_COORDINATE_VERSION
        and fingerprints.get("optimizer_coordinate_version")
        == EXPECTED_OPTIMIZER_COORDINATE_VERSION
        and fingerprints.get("scientific_fingerprint")
        == manifest.get("scientific_fingerprint")
        and fingerprints.get("runtime_fingerprint") == manifest.get("runtime_fingerprint")
        and fingerprints.get("source_fingerprint") == manifest.get("source_fingerprint"),
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
        f"{expected_task} panel fingerprints changed",
    )
    _require(
        dict(status.get("fingerprints", {})) == dict(fingerprints),
        f"{expected_task} task status fingerprints changed",
    )
    model_seed = int(result.get("model_seed", -1))
    _require(
        model_seed >= 0 and int(status.get("model_seed", -2)) == model_seed,
        f"{expected_task} model seed changed",
    )
    selection = metrics.get("selection", {})
    _require(isinstance(selection, Mapping), f"{expected_task} selection is missing")
    confirmation = selection.get("confirmation", {})
    _require(isinstance(confirmation, Mapping), f"{expected_task} panel-B evidence is missing")
    bounds = confirmation.get("panel_b_lower_bounds", [])
    _require(
        isinstance(bounds, Sequence)
        and len(bounds) == 2
        and all(_finite(value) for value in bounds),
        f"{expected_task} panel-B bounds are invalid",
    )
    if expected_task == "bounded_teacher":
        _require(
            _one(confirmation.get("accepted"))
            and int(selection.get("selected_step", 0)) > 0
            and all(float(value) > 0.0 for value in bounds),
            "bounded-teacher sealed panel B did not confirm",
        )
    else:
        _require(
            _zero(confirmation.get("accepted"))
            and _zero(selection.get("selected_step"))
            and all(float(value) <= 0.0 for value in bounds),
            "Dirichlet-null sealed panel B did not reject the nominee",
        )
    _verify_zero_clipping(
        dict(metrics),
        history_path=task_dir / "training_history.csv",
        windows_path=task_dir / "clipping_windows.csv",
        description=f"{expected_task} task",
    )
    _no_work(result, f"{expected_task} task result")
    _no_work(status, f"{expected_task} task status")
    return model_seed


def _failed_subchecks(gate: Mapping[str, Any]) -> list[str]:
    raw = gate.get("subchecks", {})
    if not isinstance(raw, Mapping):
        return []
    return sorted(
        str(name)
        for name, check in raw.items()
        if isinstance(check, Mapping) and not _one(check.get("passed"))
    )


def _verify_pilot(
    run_dir: Path,
    stored: Mapping[str, Any],
    *,
    oracle: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_candidates = stored.get("candidate_records", [])
    _require(
        isinstance(raw_candidates, Sequence) and len(raw_candidates) == 2,
        "selection-power parent must contain exactly two pilot candidates",
    )
    candidates = [dict(value) for value in raw_candidates if isinstance(value, Mapping)]
    _require(len(candidates) == 2, "selection-power pilot candidates are invalid")
    model_seeds: list[int] = []
    for index, (candidate, learning_rate) in enumerate(
        zip(candidates, EXPECTED_LEARNING_RATES, strict=True)
    ):
        _require(
            int(candidate.get("accumulation_steps", -1)) == EXPECTED_ACCUMULATION
            and _same(candidate.get("learning_rate"), learning_rate),
            "selection-power pilot candidate ordering changed",
        )
        paired: list[int] = []
        for key, task in (
            ("teacher", "bounded_teacher"),
            ("null", "dirichlet_null"),
        ):
            aggregate = candidate.get(key, {})
            _require(isinstance(aggregate, Mapping), f"selection-power {task} result is missing")
            paired.append(
                _verify_task_result(
                    dict(aggregate),
                    task_dir=run_dir / "pilot" / f"lr-{index:02d}" / task,
                    expected_task=task,
                    expected_learning_rate=learning_rate,
                    manifest=manifest,
                )
            )
        _require(paired[0] == paired[1], "selection-power teacher/null initialization is not paired")
        model_seeds.extend(paired)
    _require(len(set(model_seeds)) == 1, "selection-power pilot profiles did not share initialization")

    recomputed = _json_value(evaluate_power_pilot(candidates, panel_power=oracle))
    comparable = {key: stored.get(key) for key in recomputed}
    _require(recomputed == comparable, "selection-power pilot gate does not recompute")
    _require(
        _zero(recomputed.get("passed"))
        and _zero(dict(recomputed.get("selected_profile", {})).get("selected"))
        and _one(recomputed.get("optimizer_health_pass")),
        "selection-power parent has an incompatible pilot outcome",
    )
    gates = recomputed.get("candidate_gates", [])
    _require(isinstance(gates, Sequence) and len(gates) == 2, "selection-power candidate gates are missing")
    for gate in gates:
        _require(isinstance(gate, Mapping), "selection-power candidate gate is invalid")
        base = gate.get("base_gate", {})
        _require(
            isinstance(base, Mapping)
            and _failed_subchecks(base) == ["null_panel_a_lower_bounds"],
            "selection-power candidate did not fail only the null panel-A check",
        )
        null_selection = base.get("null_selection", {})
        _require(isinstance(null_selection, Mapping), "selection-power null selection is missing")
        nomination = null_selection.get("nomination", {})
        confirmation = null_selection.get("confirmation", {})
        _require(
            isinstance(nomination, Mapping)
            and isinstance(confirmation, Mapping)
            and any(
                _finite(value) and float(value) > 0.0
                for value in nomination.get("nominee_panel_a_lower_bounds", [])
            )
            and _zero(confirmation.get("accepted"))
            and _zero(null_selection.get("selected_step"))
            and len(confirmation.get("panel_b_lower_bounds", [])) == 2
            and all(
                _finite(value) and float(value) <= 0.0
                for value in confirmation.get("panel_b_lower_bounds", [])
            ),
            "selection-power A-only failure or sealed-B rejection changed",
        )
        teacher_selection = base.get("teacher_selection", {})
        _require(
            isinstance(teacher_selection, Mapping)
            and _one(dict(teacher_selection.get("confirmation", {})).get("accepted"))
            and int(teacher_selection.get("selected_step", 0)) > 0,
            "selection-power teacher nominee was not confirmed on B",
        )
    multiplicity = recomputed.get("null_multiplicity_analysis", {})
    _require(
        isinstance(multiplicity, Mapping)
        and int(multiplicity.get("candidate_count", -1)) == 2
        and int(multiplicity.get("failed_candidate_count", -1)) == 2
        and int(multiplicity.get("a_only_candidate_count", -1)) == 2
        and _one(multiplicity.get("a_only_explains_failure")),
        "selection-power null multiplicity diagnosis changed",
    )
    _no_work(stored, "selection-power pilot gate")
    return candidates, dict(multiplicity)


def verify_parent_selection_power_run(path: str | Path) -> dict[str, Any]:
    """Verify and normalize the exact 123-record selection-power parent."""

    run_dir = Path(path).resolve()
    _require(run_dir.is_dir(), f"selection-power parent run does not exist: {run_dir}")
    paths = {name: run_dir / relative for name, relative in _REQUIRED_FILES.items()}
    missing = [str(value) for value in paths.values() if not value.is_file()]
    _require(not missing, "selection-power parent artifacts are missing: " + ", ".join(missing))
    values = {name: _load(value) for name, value in paths.items() if name != "registry"}
    manifest, status = values["manifest"], values["status"]
    _require(
        manifest.get("schema") == status.get("schema") == PARENT_RUN_SCHEMA
        and int(manifest.get("schema_version", -1)) == 1
        and int(status.get("schema_version", -1)) == 1,
        "selection-power parent schema is incompatible",
    )
    _require(
        Path(str(manifest.get("run_dir", ""))).resolve() == run_dir,
        "selection-power manifest run directory is inconsistent",
    )
    _require(
        status.get("status") == "complete"
        and status.get("outcome") == "gate_failed"
        and status.get("phase") == "pilot"
        and status.get("stage") == "pilot"
        and status.get("decision") == EXPECTED_PARENT_DECISION
        and status.get("required_gate") == "pilot"
        and _zero(status.get("required_gate_pass")),
        "selection-power parent is not the multiplicity precursor",
    )
    skips = status.get("skips", [])
    _require(
        isinstance(skips, Sequence)
        and any(
            isinstance(value, Mapping)
            and value.get("stage") == "confirmation"
            and "no oracle-qualified pilot profile" in str(value.get("reason", ""))
            for value in skips
        ),
        "selection-power parent did not explicitly skip confirmation",
    )
    forbidden = (
        "confirmation",
        "confirmation_panel_registry.json",
        "confirmation_oracle_feasibility.json",
        "selected_selection_power_profile.json",
        "selection_power_teacher_confirmation.json",
        "selection_power_null_confirmation.json",
    )
    _require(
        all(not (run_dir / relative).exists() for relative in forbidden),
        "selection-power parent contains forbidden profile or confirmation tasks",
    )
    registry = _verify_registry(
        run_dir, registry_path=paths["registry"], status_path=paths["status"]
    )
    scientific = _verify_scientific_manifest(manifest)
    source_paths = [Path(str(value)).resolve() for value in manifest.get("source_paths", [])]
    _require(
        len(source_paths) == 15
        and all(value.is_file() for value in source_paths)
        and source_fingerprint(source_paths) == manifest.get("source_fingerprint"),
        "selection-power parent source fingerprint no longer matches",
    )
    _require(
        manifest.get("parent_provenance_sha256")
        == file_fingerprint(paths["parent_provenance"]),
        "selection-power manifest does not bind normalized-head provenance",
    )
    transitive = _verify_transitive_parent(values["parent_provenance"])
    _require(
        scientific.get("parent_artifact_registry_sha256")
        == transitive.get("artifact_registry_sha256")
        and scientific.get("parent_scientific_fingerprint")
        == transitive.get("scientific_fingerprint"),
        "selection-power scientific configuration does not bind its parent",
    )

    raw_oracle = values["pilot_oracle"].get("raw_evidence", {})
    _require(isinstance(raw_oracle, Mapping), "selection-power pilot oracle raw evidence is missing")
    recomputed_oracle = _json_value(
        evaluate_oracle_panel_set(raw_oracle, expected_roles=("a", "b"))
    )
    recomputed_oracle["raw_evidence"] = _json_value(raw_oracle)
    _require(
        recomputed_oracle == values["pilot_oracle"]
        and _one(recomputed_oracle.get("passed")),
        "selection-power pilot oracle gate does not recompute as passing",
    )
    preflight = _json_value(
        evaluate_selection_power_preflight(
            normalized_head_preflight={
                "evaluation_status": "evaluated",
                "passed": int(transitive.get("preflight_pass", 0)),
            },
            saved_forensic=values["saved_forensic"],
            calibration=values["oracle_calibration"],
        )
    )
    _require(
        preflight == values["preflight_gate"] and _one(preflight.get("passed")),
        "selection-power preflight gate does not recompute as passing",
    )
    candidates, multiplicity = _verify_pilot(
        run_dir,
        values["pilot_gate"],
        oracle=values["pilot_oracle"],
        manifest=manifest,
    )
    _require(
        multiplicity == values["pilot_multiplicity"],
        "selection-power stored multiplicity report changed",
    )
    failures = values["pilot_failures"]
    _require(
        int(failures.get("count", -1)) == 0
        and list(failures.get("failures", [])) == [],
        "selection-power pilot contains task failures",
    )

    confirmation_power = {
        "gate": "confirmation_oracle_panel_power",
        "evaluation_status": "not_evaluated",
        "passed": 0,
        "reason": "confirmation panels were not frozen",
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    report = _json_value(
        evaluate_selection_power_workflow(
            provenance=values["parent_provenance"],
            preflight=values["preflight_gate"],
            pilot_panel_power=values["pilot_oracle"],
            pilot=values["pilot_gate"],
            confirmation_panel_power=confirmation_power,
            teacher_results=[],
            null_results=[],
            require_gate="pilot",
        )
    )
    _require(report == values["workflow"], "selection-power terminal workflow does not recompute")
    _require(
        dict(report.get("decision", {})) == values["decision"]
        and values["decision"].get("decision") == EXPECTED_PARENT_DECISION,
        "selection-power terminal decision changed",
    )
    for name, value in values.items():
        _no_work(value, f"selection-power {name}")
    _require(
        _zero(status.get("physical_training_authorized"))
        and _zero(status.get("sampling_authorized")),
        "selection-power parent authorized physical training or sampling",
    )
    schedule = transitive.get("schedule_metadata", {})
    _require(isinstance(schedule, Mapping), "selection-power parent schedule is missing")
    horizon = schedule.get("horizon", transitive.get("horizon"))
    _require(_finite(horizon) and float(horizon) > 0.0, "selection-power parent horizon is invalid")
    return {
        "schema": PARENT_RUN_SCHEMA + "-multiplicity-parent-provenance",
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
        "preflight_pass": 1,
        "oracle_calibration_pass": 1,
        "pilot_oracle_pass": 1,
        "pilot_pass": 0,
        "pilot_candidate_count": len(candidates),
        "pilot_task_count": 2 * len(candidates),
        "all_tasks_complete_finite_boundary_admissible": 1,
        "all_task_clipping_zero": 1,
        "a_only_failure_count": int(multiplicity["a_only_candidate_count"]),
        "sealed_b_rejection_count": int(multiplicity["a_only_candidate_count"]),
        "task_failure_count": 0,
        "selected_profile": 0,
        "confirmation_performed": 0,
        "lineage_registry_record_counts": [125, 332, 222, 381],
        "transitive_parent_provenance": values["parent_provenance"],
        "artifacts": {name: _record(value) for name, value in paths.items()},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
