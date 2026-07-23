"""Strict provenance for the H1 density-ratio function-step experiment.

The only admissible parent is the completed multiplicity-aware confirmation
that ended ``density_ratio_value_only`` and explicitly authorized the H1
function-step patch.  The verifier binds its exact 263-record terminal
registry (including its known SHA-256), verifies every registered payload,
checks the six completed zero-clipping confirmation tasks, and recursively
verifies the stored 123 -> 125 -> 332 -> 222 -> 381 ancestry.

This file is additive so none of the immutable parent's source hashes change.
"""

from __future__ import annotations

from dataclasses import asdict
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
from mnist.d0_score_density_ratio_multiplicity_provenance import (
    verify_parent_selection_power_run,
)
from mnist.d0_score_density_ratio_sealed_null_gate import SealedNullThresholds


__all__ = [
    "PARENT_RUN_SCHEMA",
    "PARENT_REGISTRY_SCHEMA",
    "PARENT_REGISTRY_RECORD_COUNT",
    "PARENT_REGISTRY_SHA256",
    "EXPECTED_PARENT_DECISION",
    "EXPECTED_LINEAGE_REGISTRY_RECORD_COUNTS",
    "verify_parent_multiplicity_run",
]


PARENT_RUN_SCHEMA = "experiment12-d0-score-density-ratio-multiplicity-confirmation"
PARENT_REGISTRY_SCHEMA = PARENT_RUN_SCHEMA + "-artifact-registry"
PARENT_REGISTRY_RECORD_COUNT = 263
PARENT_REGISTRY_SHA256 = (
    "d500962d8b2c61594884ba6c759de69bfe268b807f85c6dc2dd6b17512c36c8b"
)
EXPECTED_PARENT_DECISION = "density_ratio_value_only"
EXPECTED_LINEAGE_REGISTRY_RECORD_COUNTS = (123, 125, 332, 222, 381)
EXPECTED_MODEL_SCHEMA = "d0-boundary-smooth-potential-unet-mean-head-v2"
EXPECTED_HEAD_COORDINATE_VERSION = "d0-spatial-sum-to-mean-head-coordinate-v1"
EXPECTED_OPTIMIZER_COORDINATE_VERSION = "d0-mean-head-coordinate-adamw-v1"
EXPECTED_CONFIRMATION_SEEDS = (260971, 260972, 260973)

_REQUIRED_FILES: dict[str, str] = {
    "manifest": "run_manifest.json",
    "status": "run_status.json",
    "registry": "artifact_registry.json",
    "parent_provenance": "parent_provenance.json",
    "preflight": "multiplicity_preflight_gate.json",
    "replay": "multiplicity_replay_gate.json",
    "selected_profile": "selected_multiplicity_profile.json",
    "confirmation_oracle": "confirmation_oracle_feasibility.json",
    "panel_registry": "confirmation_panel_registry.json",
    "teacher_results": "multiplicity_teacher_confirmation.json",
    "null_results": "multiplicity_null_confirmation.json",
    "task_failures": "confirmation_task_failures.json",
    "teacher_gate": "confirmation_teacher_study_gate.json",
    "null_gate": "confirmation_null_family_gate.json",
    "max_t": "confirmation_b_c_d_max_t.json",
    "workflow": "multiplicity_control_gate.json",
    "decision": "multiplicity_decision.json",
}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"cannot read multiplicity parent artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(
            f"multiplicity parent artifact is not a JSON object: {path}"
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
    expected_sha256: str | None = PARENT_REGISTRY_SHA256,
) -> dict[str, Any]:
    """Verify terminal registry identity, exact file set, and every payload."""

    registry = _load(registry_path)
    status = _load(status_path)
    registry_sha = file_fingerprint(registry_path)
    _require(
        registry.get("schema") == PARENT_REGISTRY_SCHEMA
        and int(registry.get("schema_version", -1)) == 1,
        "multiplicity terminal registry schema is incompatible",
    )
    if expected_sha256 is not None:
        _require(
            registry_sha == expected_sha256,
            "multiplicity terminal registry is not the frozen 263-record parent",
        )
    _require(
        status.get("artifact_registry_sha256") == registry_sha
        and int(status.get("artifact_registry_size", -1))
        == int(registry_path.stat().st_size),
        "multiplicity status does not bind its terminal registry",
    )
    exclusions = set(registry.get("terminal_files_excluded_to_avoid_self_reference", []))
    _require(
        exclusions == {"artifact_registry.json", "run_status.json"},
        "multiplicity registry exclusions are incompatible",
    )
    raw_records = registry.get("records", {})
    _require(isinstance(raw_records, Mapping), "multiplicity registry records are invalid")
    records = dict(raw_records)
    _require(
        len(records) == int(expected_count),
        f"multiplicity terminal registry must contain exactly {expected_count} records",
    )
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in exclusions
    }
    _require(
        set(records) == actual,
        "multiplicity registry is incomplete or contains stale records",
    )
    for relative, raw in records.items():
        _require(isinstance(raw, Mapping), f"invalid registry record {relative}")
        _verify_record(
            dict(raw),
            root=run_dir,
            expected_path=run_dir / relative,
            description=f"registered multiplicity artifact {relative}",
        )
    return registry


def _verify_scientific_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw = manifest.get("scientific_config", {})
    _require(isinstance(raw, Mapping), "multiplicity scientific configuration is missing")
    scientific = dict(raw)
    _require(
        scientific.get("algorithm") == PARENT_RUN_SCHEMA
        and int(scientific.get("algorithm_version", -1)) == 1,
        "multiplicity parent algorithm is incompatible",
    )
    _require(
        config_fingerprint(scientific) == manifest.get("scientific_fingerprint"),
        "multiplicity scientific fingerprint is inconsistent",
    )
    kernel = scientific.get("kernel", {})
    _require(isinstance(kernel, Mapping), "multiplicity kernel is missing")
    for key, expected in EXPECTED_KERNEL.items():
        _require(_same(kernel.get(key), expected), f"multiplicity kernel mismatch for {key}")
    _require(
        scientific.get("thresholds")
        == _json_value(asdict(SealedNullThresholds())),
        "multiplicity scientific thresholds changed",
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
        _require(scientific.get(key) == expected, f"multiplicity schema mismatch for {key}")
    optimization = scientific.get("optimization", {})
    confirmation = scientific.get("confirmation", {})
    simultaneous = scientific.get("simultaneous_bootstrap", {})
    _require(
        isinstance(optimization, Mapping)
        and optimization.get("optimizer") == "coordinate-conjugate-AdamW"
        and _same(optimization.get("body_learning_rate"), 3e-5)
        and int(optimization.get("accumulation_steps", -1)) == 8
        and _same(optimization.get("loss_scale"), PARENT_LOSS_SCALE)
        and _same(optimization.get("body_weight_decay"), 1e-4)
        and _same(optimization.get("ema_decay"), 0.99)
        and _same(optimization.get("grad_clip"), 1.0)
        and _zero(optimization.get("adaptive_loss_scaling")),
        "multiplicity optimizer configuration changed",
    )
    _require(
        isinstance(confirmation, Mapping)
        and tuple(int(value) for value in confirmation.get("model_seeds", []))
        == EXPECTED_CONFIRMATION_SEEDS
        and int(confirmation.get("steps", -1)) == 4000
        and int(confirmation.get("paths_per_selection_panel", -1)) == 128
        and int(confirmation.get("paths_per_audit_panel", -1)) == 128
        and int(confirmation.get("anchors_per_path", -1)) == 32,
        "multiplicity confirmation configuration changed",
    )
    family = simultaneous.get("confirmation_family", {}) if isinstance(simultaneous, Mapping) else {}
    _require(
        isinstance(simultaneous, Mapping)
        and simultaneous.get("version")
        == "studentized-whole-path-centered-bootstrap-max-t-v1"
        and int(simultaneous.get("bootstrap_replicates", -1)) == 50_000
        and _same(simultaneous.get("one_sided_familywise_confidence"), 0.95)
        and isinstance(family, Mapping)
        and int(family.get("expected_member_count", -1)) == 18,
        "multiplicity simultaneous-bootstrap configuration changed",
    )
    _require(int(scientific.get("root_seed", -1)) == 260961, "multiplicity root seed changed")
    _no_work(scientific, "multiplicity scientific configuration")
    return scientific


def _verify_transitive_parent(parent: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        parent.get("schema") == PARENT_RUN_SCHEMA + "-parent-provenance"
        and int(parent.get("schema_version", -1)) == 1
        and _one(parent.get("passed")),
        "multiplicity selection-power provenance is incompatible",
    )
    root = Path(str(parent.get("run_dir", ""))).resolve()
    recomputed = verify_parent_selection_power_run(root)
    normalized = dict(parent)
    normalized.pop("verifier_source_fingerprint", None)
    normalized["schema"] = recomputed["schema"]
    _require(
        normalized == recomputed,
        "multiplicity transitive selection-power provenance changed",
    )
    _require(
        int(recomputed.get("terminal_registry_record_count", -1)) == 123
        and tuple(recomputed.get("lineage_registry_record_counts", []))
        == EXPECTED_LINEAGE_REGISTRY_RECORD_COUNTS[1:],
        "multiplicity parent does not bind exact 123/125/332/222/381 ancestry",
    )
    _no_work(parent, "multiplicity transitive parent")
    return recomputed


def _task_rows(value: Mapping[str, Any], description: str) -> list[dict[str, Any]]:
    raw = value.get("task_results", [])
    _require(isinstance(raw, Sequence), f"{description} task results are invalid")
    rows = [dict(item) for item in raw if isinstance(item, Mapping)]
    _require(len(rows) == len(raw) == 3, f"{description} must contain three tasks")
    return rows


def _verify_task(
    result: Mapping[str, Any],
    *,
    run_dir: Path,
    expected_task: str,
    expected_seed: int,
    manifest: Mapping[str, Any],
) -> None:
    task_dir = run_dir / "confirmation" / f"seed-{expected_seed}" / expected_task
    result_path = task_dir / "task_result.json"
    status_path = task_dir / "task_status.json"
    _require(result_path.is_file() and status_path.is_file(), f"missing {expected_task} task artifacts")
    stored = _load(result_path)
    status = _load(status_path)
    _require(stored == dict(result), f"{expected_task} aggregate/task result mismatch")
    _require(
        stored.get("schema")
        == "experiment12-d0-score-density-ratio-head-confirmation-task-result"
        and int(stored.get("schema_version", -1)) == 1
        and stored.get("task") == expected_task
        and int(stored.get("model_seed", -1)) == expected_seed,
        f"{expected_task} result schema or identity changed",
    )
    _require(
        status.get("status") == "complete"
        and status.get("task") == expected_task
        and int(status.get("model_seed", -1)) == expected_seed
        and int(status.get("training_step", -1)) == 4000
        and status.get("task_result_sha256") == file_fingerprint(result_path),
        f"{expected_task} task status is incomplete or unbound",
    )
    metrics = stored.get("metrics", {})
    _require(isinstance(metrics, Mapping), f"{expected_task} metrics are missing")
    _require(
        _one(metrics.get("complete"))
        and _one(metrics.get("finite"))
        and _one(metrics.get("boundary_admissible")),
        f"{expected_task} is incomplete, nonfinite, or boundary-inadmissible",
    )
    for name in (
        "post_warmup_clip_fraction",
        "final_500_clip_fraction",
        "final_200_clip_fraction",
    ):
        _require(
            _finite(metrics.get(name)) and float(metrics[name]) == 0.0,
            f"{expected_task} does not record zero {name}",
        )
    diagnostics = metrics.get("optimization_diagnostics", {})
    _require(isinstance(diagnostics, Mapping), f"{expected_task} diagnostics are missing")
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
        f"{expected_task} contains nonzero or invalid clipping windows",
    )
    fingerprints = stored.get("fingerprints", {})
    _require(isinstance(fingerprints, Mapping), f"{expected_task} fingerprints are missing")
    _require(
        fingerprints.get("phase") == "multiplicity-confirmation"
        and fingerprints.get("task") == expected_task
        and _same(fingerprints.get("learning_rate"), 3e-5)
        and int(fingerprints.get("accumulation_level", -1)) == 8
        and _same(fingerprints.get("loss_scale"), PARENT_LOSS_SCALE)
        and fingerprints.get("scientific_fingerprint")
        == manifest.get("scientific_fingerprint")
        and fingerprints.get("runtime_fingerprint")
        == manifest.get("runtime_fingerprint")
        and fingerprints.get("source_fingerprint") == manifest.get("source_fingerprint")
        and dict(status.get("fingerprints", {})) == dict(fingerprints),
        f"{expected_task} fingerprints changed",
    )
    _no_work(stored, f"{expected_task} task result")
    _no_work(status, f"{expected_task} task status")


def verify_parent_multiplicity_run(path: str | Path) -> dict[str, Any]:
    """Verify and normalize the exact 263-record value-only parent run."""

    run_dir = Path(path).resolve()
    _require(run_dir.is_dir(), f"multiplicity parent run does not exist: {run_dir}")
    paths = {name: run_dir / relative for name, relative in _REQUIRED_FILES.items()}
    missing = [str(value) for value in paths.values() if not value.is_file()]
    _require(not missing, "multiplicity parent artifacts are missing: " + ", ".join(missing))
    values = {name: _load(value) for name, value in paths.items() if name != "registry"}
    manifest, status = values["manifest"], values["status"]
    _require(
        manifest.get("schema") == status.get("schema") == PARENT_RUN_SCHEMA
        and int(manifest.get("schema_version", -1)) == 1
        and int(status.get("schema_version", -1)) == 1,
        "multiplicity parent schema is incompatible",
    )
    _require(
        Path(str(manifest.get("run_dir", ""))).resolve() == run_dir,
        "multiplicity manifest run directory is inconsistent",
    )
    _require(
        status.get("status") == "complete"
        and status.get("outcome") == "gate_failed"
        and status.get("phase") == "confirmation"
        and status.get("stage") == "confirm"
        and status.get("decision") == EXPECTED_PARENT_DECISION
        and status.get("required_gate") == "controls"
        and _zero(status.get("required_gate_pass"))
        and _one(status.get("h1_function_step_patch_authorized")),
        "multiplicity parent is not the authorized density-ratio-value-only precursor",
    )
    _require(list(status.get("skips", [])) == [], "multiplicity confirmation contains skipped work")
    registry = _verify_registry(
        run_dir,
        registry_path=paths["registry"],
        status_path=paths["status"],
    )
    scientific = _verify_scientific_manifest(manifest)
    source_paths = [Path(str(value)).resolve() for value in manifest.get("source_paths", [])]
    _require(
        len(source_paths) == 18
        and all(value.is_file() for value in source_paths)
        and source_fingerprint(source_paths) == manifest.get("source_fingerprint"),
        "multiplicity parent source fingerprint no longer matches",
    )
    _require(
        manifest.get("parent_provenance_sha256")
        == file_fingerprint(paths["parent_provenance"]),
        "multiplicity manifest does not bind selection-power provenance",
    )
    transitive = _verify_transitive_parent(values["parent_provenance"])
    _require(
        scientific.get("parent_artifact_registry_sha256")
        == transitive.get("artifact_registry_sha256")
        and scientific.get("parent_scientific_fingerprint")
        == transitive.get("scientific_fingerprint"),
        "multiplicity scientific configuration does not bind its parent",
    )

    _require(_one(values["preflight"].get("passed")), "multiplicity preflight did not pass")
    _require(
        _one(values["replay"].get("passed"))
        and _one(values["replay"].get("optimizer_health_pass"))
        and _zero(values["replay"].get("optimizer_steps_performed")),
        "multiplicity parent replay is incompatible",
    )
    selected = values["selected_profile"]
    _require(
        _same(selected.get("body_learning_rate"), 3e-5)
        and int(selected.get("accumulation_steps", -1)) == 8
        and _zero(selected.get("parent_weights_reused"))
        and _zero(selected.get("parent_states_reused_for_confirmation")),
        "multiplicity selected profile changed",
    )
    _require(
        _one(values["confirmation_oracle"].get("passed")),
        "multiplicity confirmation panels were not oracle-qualified",
    )
    panel_registry = values["panel_registry"]
    _require(
        _one(panel_registry.get("parent_confirmation_isolation_pass"))
        and int(panel_registry.get("parent_overlap_path_count", -1)) == 0
        and _zero(panel_registry.get("panel_regeneration_after_inspection")),
        "multiplicity confirmation panels are not fresh and frozen",
    )
    failures = values["task_failures"]
    _require(
        int(failures.get("count", -1)) == 0
        and list(failures.get("failures", [])) == [],
        "multiplicity confirmation contains task failures",
    )
    teachers = _task_rows(values["teacher_results"], "bounded-teacher")
    nulls = _task_rows(values["null_results"], "Dirichlet-null")
    teacher_seeds = tuple(int(row.get("model_seed", -1)) for row in teachers)
    null_seeds = tuple(int(row.get("model_seed", -1)) for row in nulls)
    _require(
        teacher_seeds == null_seeds == EXPECTED_CONFIRMATION_SEEDS,
        "multiplicity confirmation seed pairing changed",
    )
    for seed, teacher, null in zip(teacher_seeds, teachers, nulls, strict=True):
        _verify_task(
            teacher,
            run_dir=run_dir,
            expected_task="bounded_teacher",
            expected_seed=seed,
            manifest=manifest,
        )
        _verify_task(
            null,
            run_dir=run_dir,
            expected_task="dirichlet_null",
            expected_seed=seed,
            manifest=manifest,
        )

    teacher_gate = values["teacher_gate"]
    _require(
        _zero(teacher_gate.get("passed"))
        and int(teacher_gate.get("classification_passing_seed_count", -1)) == 3
        and int(teacher_gate.get("derivative_passing_seed_count", -1)) == 0
        and _one(teacher_gate.get("optimizer_health_pass"))
        and _zero(teacher_gate.get("panel_disagreement")),
        "multiplicity bounded-teacher value-only result changed",
    )
    null_gate = values["null_gate"]
    _require(
        _one(null_gate.get("passed"))
        and _one(null_gate.get("optimizer_health_pass"))
        and _zero(null_gate.get("familywise_false_discovery"))
        and _zero(null_gate.get("selection_false_discovery"))
        and _zero(null_gate.get("audit_false_discovery")),
        "multiplicity stationary-null family did not pass",
    )
    max_t = values["max_t"]
    _require(
        _one(max_t.get("finite"))
        and int(max_t.get("family_size", -1)) == 18
        and int(max_t.get("bootstrap_replicates", -1)) == 50_000
        and _same(max_t.get("confidence"), 0.95)
        and _zero(max_t.get("familywise_false_discovery")),
        "multiplicity B/C/D max-T family changed",
    )
    workflow = values["workflow"]
    decision = values["decision"]
    _require(
        workflow.get("required_gate") == "controls"
        and _zero(workflow.get("required_gate_pass"))
        and dict(workflow.get("decision", {})) == decision
        and decision.get("decision") == EXPECTED_PARENT_DECISION
        and _one(decision.get("h1_function_step_patch_authorized"))
        and _zero(decision.get("physical_training_authorized"))
        and _zero(decision.get("sampling_authorized")),
        "multiplicity terminal gate or H1 authorization changed",
    )
    for name, value in values.items():
        _no_work(value, f"multiplicity {name}")
    _require(
        _zero(status.get("physical_training_authorized"))
        and _zero(status.get("sampling_authorized")),
        "multiplicity parent authorized physical training or sampling",
    )
    schedule = transitive.get("schedule_metadata", {})
    _require(isinstance(schedule, Mapping), "multiplicity parent schedule is missing")
    horizon = schedule.get("horizon", transitive.get("horizon"))
    _require(_finite(horizon) and float(horizon) > 0.0, "multiplicity parent horizon is invalid")
    return {
        "schema": PARENT_RUN_SCHEMA + "-h1-trust-parent-provenance",
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
        "h1_function_step_patch_authorized": 1,
        "terminal_registry_record_count": len(dict(registry["records"])),
        "artifact_registry_sha256": file_fingerprint(paths["registry"]),
        "artifact_registry_size": int(paths["registry"].stat().st_size),
        "parent_loss_scale": PARENT_LOSS_SCALE,
        "preflight_pass": 1,
        "replay_pass": 1,
        "confirmation_oracle_pass": 1,
        "confirmation_task_count": 6,
        "all_tasks_complete_finite_boundary_admissible": 1,
        "all_task_clipping_zero": 1,
        "teacher_classification_passing_seed_count": 3,
        "teacher_derivative_passing_seed_count": 0,
        "null_family_pass": 1,
        "task_failure_count": 0,
        "lineage_registry_record_counts": list(
            EXPECTED_LINEAGE_REGISTRY_RECORD_COUNTS
        ),
        "transitive_parent_provenance": values["parent_provenance"],
        "artifacts": {name: _record(value) for name, value in paths.items()},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
