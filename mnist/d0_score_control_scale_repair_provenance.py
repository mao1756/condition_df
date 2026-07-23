"""Strict parent-run provenance for the D0 control scale-repair workflow.

The optimizer-scale repair is authorized by one narrow piece of evidence: the
boundary-control parent passed its model/operator and supervised scientific
checks, but the supervised task failed the optimizer-health gate solely because
of gradient clipping.  This module verifies that evidence without importing the
new orchestration CLI or any training code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from mnist.d0_one_image_gate import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
)
from mnist.d0_score_boundary_control_gate import (
    BoundaryControlThresholds,
    evaluate_boundary_preflight,
    evaluate_supervised_teacher,
)


__all__ = [
    "PARENT_RUN_SCHEMA",
    "verify_parent_boundary_control_run",
]


PARENT_RUN_SCHEMA = "experiment12-d0-score-boundary-controls"
PARENT_REGISTRY_SCHEMA = PARENT_RUN_SCHEMA + "-artifact-registry"
FAILED_SCORE_PROVENANCE_SCHEMA = PARENT_RUN_SCHEMA + "-failed-run-provenance"

EXPECTED_KERNEL: dict[str, Any] = {
    "grid_size": 28,
    "sample_steps": 512,
    "reference_substeps": 256,
    "tau_eff": 5e-5,
    "edge_alpha_mode": "alpha_eff",
    "alpha_eff": 1.0,
    "mass_floor": 1e-7,
    "limiter_fraction": 1.0,
    "lambda_mix": 0.35,
}

EXPECTED_MODEL: dict[str, Any] = {
    "schema": "d0-boundary-smooth-potential-unet-v1",
    "schema_version": 1,
    "base_channels": 32,
    "raw_log_density_used": 0,
}

EXPECTED_SYNTHETIC_DATA: dict[str, Any] = {
    "train_paths": 128,
    "selection_paths": 32,
    "audit_paths": 32,
    "anchors_per_path": 32,
    "anchor_bin_counts": [4, 4, 4, 4, 16],
    "teacher_epsilon": 0.5,
    "teacher_version": "d0-bounded-four-anchor-mixture-v1",
    "teacher_and_null_states_independent": 1,
}

EXPECTED_OPTIMIZATION: dict[str, Any] = {
    "batch_size": 64,
    "train_steps": 4000,
    "validation_every": 250,
    "checkpoint_every": 250,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "ema_decay": 0.99,
    "grad_clip": 1.0,
    "clip_warmup_steps": 500,
    "training_probes": 4,
    "selection_probes_per_bank": 16,
    "audit_probes_per_bank": 64,
}

_REQUIRED_FILES: dict[str, str] = {
    "manifest": "run_manifest.json",
    "status": "run_status.json",
    "registry": "artifact_registry.json",
    "operator_preflight": "boundary_operator_preflight.json",
    "preflight_gate": "boundary_preflight_gate.json",
    "control_gate": "boundary_control_gate.json",
    "control_report": "boundary_control_report.json",
    "decision": "control_repair_decision.json",
    "supervised": "supervised_teacher_control.json",
    "implicit_study": "implicit_teacher_study.json",
    "null_study": "null_study.json",
    "task_failures": "task_failures.json",
    "failed_score_provenance": "failed_run_provenance.json",
    "supervised_task_result": "tasks/supervised-teacher/task_result.json",
    "supervised_task_status": "tasks/supervised-teacher/task_status.json",
}

_UPSTREAM_ARTIFACTS = {
    "manifest",
    "status",
    "preflight",
    "cache",
    "controls",
    "operator",
    "cache_index",
    "artifact_registry",
}


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"cannot read parent artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"parent artifact is not a JSON object: {path}")
    return dict(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactCompatibilityError(message)


def _zero(value: Any) -> bool:
    try:
        return int(value) == 0
    except (TypeError, ValueError):
        return False


def _one(value: Any) -> bool:
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return False


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_fingerprint(path),
        "size": int(path.stat().st_size),
    }


def _verify_record(
    raw_record: Mapping[str, Any],
    *,
    expected_path: Path | None = None,
    root: Path | None = None,
    description: str,
) -> Path:
    record = dict(raw_record)
    artifact = Path(str(record.get("path", ""))).resolve()
    if expected_path is not None:
        _require(
            artifact == expected_path.resolve(),
            f"{description} path does not match its recorded location",
        )
    if root is not None:
        try:
            artifact.relative_to(root.resolve())
        except ValueError as exc:
            raise ArtifactCompatibilityError(
                f"{description} escapes its recorded run directory"
            ) from exc
    _require(artifact.is_file(), f"{description} is missing: {artifact}")
    _require(
        record.get("sha256") == file_fingerprint(artifact)
        and int(record.get("size", -1)) == int(artifact.stat().st_size),
        f"{description} hash or size mismatch",
    )
    return artifact


def _verify_terminal_registry(
    run_dir: Path,
    *,
    status: Mapping[str, Any],
    registry_path: Path,
) -> dict[str, Any]:
    _require(registry_path.is_file(), "parent artifact registry is missing")
    _require(
        status.get("artifact_registry_sha256") == file_fingerprint(registry_path)
        and int(status.get("artifact_registry_size", -1))
        == int(registry_path.stat().st_size),
        "parent terminal status does not bind the artifact registry",
    )
    registry = _load_mapping(registry_path)
    _require(
        registry.get("schema") == PARENT_REGISTRY_SCHEMA
        and int(registry.get("schema_version", -1)) == 1,
        "parent artifact registry schema is incompatible",
    )
    exclusions = set(registry.get("terminal_files_excluded_to_avoid_self_reference", []))
    expected_exclusions = {"artifact_registry.json", "run_status.json"}
    _require(exclusions == expected_exclusions, "parent artifact registry exclusions are incompatible")
    records = registry.get("records", {})
    _require(isinstance(records, Mapping), "parent artifact registry records are invalid")
    records = dict(records)
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in exclusions
    }
    _require(
        set(records) == actual,
        "parent artifact registry is incomplete or contains stale records",
    )
    for relative, raw_record in records.items():
        _require(isinstance(raw_record, Mapping), f"invalid registry record for {relative}")
        _verify_record(
            raw_record,
            expected_path=run_dir / relative,
            root=run_dir,
            description=f"registered parent artifact {relative}",
        )
    return registry


def _verify_scientific_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    scientific = manifest.get("scientific_config", {})
    _require(isinstance(scientific, Mapping), "parent scientific configuration is missing")
    scientific = dict(scientific)
    fingerprint = str(manifest.get("scientific_fingerprint", ""))
    _require(
        bool(fingerprint) and config_fingerprint(scientific) == fingerprint,
        "parent scientific fingerprint is internally inconsistent",
    )
    _require(
        scientific.get("algorithm") == PARENT_RUN_SCHEMA
        and int(scientific.get("algorithm_version", -1)) == 1,
        "parent scientific algorithm is incompatible",
    )
    _require(_zero(scientific.get("sampling_performed")), "parent scientific run sampled")
    for section, expected in (
        ("kernel", EXPECTED_KERNEL),
        ("model", EXPECTED_MODEL),
        ("synthetic_data", EXPECTED_SYNTHETIC_DATA),
        ("optimization", EXPECTED_OPTIMIZATION),
    ):
        actual = scientific.get(section, {})
        _require(isinstance(actual, Mapping), f"parent {section} configuration is missing")
        for key, expected_value in expected.items():
            _require(
                _same(actual.get(key), expected_value),
                f"parent {section} mismatch for {key}: {actual.get(key)!r}",
            )
    thresholds = scientific.get("thresholds", {})
    _require(
        isinstance(thresholds, Mapping)
        and dict(thresholds) == BoundaryControlThresholds().to_dict(),
        "parent scientific thresholds are not the frozen control thresholds",
    )
    return scientific


def _verify_transitive_provenance(
    provenance: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    scientific: Mapping[str, Any],
    operator: Mapping[str, Any],
    provenance_path: Path,
) -> dict[str, Any]:
    _require(
        provenance.get("schema") == FAILED_SCORE_PROVENANCE_SCHEMA
        and int(provenance.get("schema_version", -1)) == 1
        and _one(provenance.get("passed")),
        "parent failed-score provenance schema or gate is incompatible",
    )
    _require(
        _zero(provenance.get("physical_training_performed"))
        and _zero(provenance.get("sampling_performed")),
        "transitive failed-score parent trained physically or sampled",
    )
    _require(
        manifest.get("failed_run_provenance_sha256") == file_fingerprint(provenance_path),
        "parent manifest does not bind failed-score provenance",
    )
    upstream_root = Path(str(provenance.get("run_dir", ""))).resolve()
    _require(upstream_root.is_dir(), "transitive failed-score run directory is missing")
    artifacts = provenance.get("artifacts", {})
    _require(
        isinstance(artifacts, Mapping) and _UPSTREAM_ARTIFACTS.issubset(artifacts),
        "transitive failed-score artifact records are incomplete",
    )
    artifacts = dict(artifacts)
    upstream_paths: dict[str, Path] = {}
    for name in sorted(_UPSTREAM_ARTIFACTS):
        raw_record = artifacts[name]
        _require(isinstance(raw_record, Mapping), f"invalid failed-score artifact record {name}")
        upstream_paths[name] = _verify_record(
            raw_record,
            root=upstream_root,
            description=f"transitive failed-score artifact {name}",
        )

    upstream_status = _load_mapping(upstream_paths["status"])
    _require(
        upstream_status.get("status") == "complete"
        and upstream_status.get("outcome") == "gate_failed"
        and upstream_status.get("decision") == "optimization_pipeline_invalid"
        and _zero(upstream_status.get("sampling_performed")),
        "transitive failed-score terminal status is incompatible",
    )
    upstream_manifest = _load_mapping(upstream_paths["manifest"])
    _require(
        upstream_manifest.get("scientific_fingerprint")
        == provenance.get("scientific_fingerprint"),
        "transitive failed-score scientific fingerprint mismatch",
    )
    manifest_registry = dict(dict(upstream_manifest.get("artifacts", {})).get("artifact_registry", {}))
    _require(
        manifest_registry == dict(artifacts["artifact_registry"]),
        "transitive failed-score manifest does not bind its artifact registry",
    )
    _require(
        scientific.get("parent_failed_run_sha256") == artifacts["status"].get("sha256")
        and scientific.get("parent_scientific_fingerprint")
        == provenance.get("scientific_fingerprint"),
        "parent scientific configuration is not transitively bound",
    )
    binding = operator.get("binding", {})
    _require(isinstance(binding, Mapping), "parent operator binding is missing")
    _require(
        binding.get("failed_run_status_sha256") == artifacts["status"].get("sha256")
        and binding.get("scientific_fingerprint") == manifest.get("scientific_fingerprint")
        and binding.get("runtime_fingerprint") == manifest.get("runtime_fingerprint")
        and binding.get("source_fingerprint") == manifest.get("source_fingerprint"),
        "parent operator preflight provenance binding is inconsistent",
    )
    schedule = provenance.get("schedule_metadata", {})
    _require(isinstance(schedule, Mapping), "transitive schedule metadata is missing")
    try:
        horizon = float(schedule.get("horizon"))
    except (TypeError, ValueError) as exc:
        raise ArtifactCompatibilityError("transitive schedule horizon is invalid") from exc
    _require(math.isfinite(horizon) and horizon > 0.0, "transitive schedule horizon is invalid")
    return {
        **dict(provenance),
        "verified_upstream_run_dir": str(upstream_root),
        "verified_upstream_status_sha256": artifacts["status"]["sha256"],
    }


def _verify_supervised_failure(
    supervised: Mapping[str, Any],
    *,
    task_result_path: Path,
    task_status: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metrics = supervised.get("metrics", {})
    stored_gate = supervised.get("gate", {})
    _require(isinstance(metrics, Mapping), "parent supervised metrics are missing")
    _require(isinstance(stored_gate, Mapping), "parent supervised gate is missing")
    recomputed = evaluate_supervised_teacher(dict(metrics), BoundaryControlThresholds())
    _require(dict(stored_gate) == recomputed, "parent supervised gate does not recompute")
    failed = [
        name
        for name, check in dict(recomputed.get("subchecks", {})).items()
        if not _one(dict(check).get("passed"))
    ]
    _require(
        failed == ["post_warmup_clip_fraction"],
        "parent supervised task did not fail solely on gradient clipping",
    )
    _require(
        task_status.get("status") == "complete"
        and task_status.get("task_kind") == "supervised_teacher"
        and task_status.get("task_result_sha256") == file_fingerprint(task_result_path),
        "parent supervised task status is incomplete or unbound",
    )
    science_checks = {
        name: dict(check)
        for name, check in dict(recomputed["subchecks"]).items()
        if name != "post_warmup_clip_fraction"
    }
    scientific_gate = {
        "gate": "supervised_teacher_scientific_metrics",
        "passed": int(all(_one(check.get("passed")) for check in science_checks.values())),
        "subchecks": science_checks,
        "sampling_performed": 0,
    }
    optimizer_gate = {
        "gate": "supervised_teacher_optimizer_health",
        "passed": int(_one(recomputed["subchecks"]["post_warmup_clip_fraction"]["passed"])),
        "subchecks": {
            "post_warmup_clip_fraction": dict(
                recomputed["subchecks"]["post_warmup_clip_fraction"]
            )
        },
        "sampling_performed": 0,
    }
    _require(_one(scientific_gate["passed"]), "parent supervised scientific metrics did not pass")
    _require(_zero(optimizer_gate["passed"]), "parent supervised optimizer-health gate did not fail")
    return recomputed, scientific_gate, optimizer_gate


def _verify_skipped_implicit_controls(
    *,
    run_dir: Path,
    implicit_study: Mapping[str, Any],
    null_study: Mapping[str, Any],
    task_failures: Mapping[str, Any],
) -> None:
    for name, study in (("implicit teacher", implicit_study), ("null", null_study)):
        _require(_zero(study.get("passed")), f"parent {name} study unexpectedly passed")
        _require(
            list(study.get("seed_gates", [])) == []
            and list(study.get("task_results", [])) == [],
            f"parent {name} study contains evaluated tasks",
        )
    _require(
        int(task_failures.get("failure_count", -1)) == 0
        and list(task_failures.get("failures", [])) == [],
        "parent contains an execution failure rather than an optimizer gate failure",
    )
    skips = list(task_failures.get("skips", []))
    _require(
        len(skips) == 1
        and isinstance(skips[0], Mapping)
        and skips[0].get("stage") == "implicit_controls",
        "parent does not record the implicit/null controls as skipped",
    )
    _require(
        not (run_dir / "loss_scale_calibration.json").exists(),
        "parent contains implicit loss-scale calibration despite skipped controls",
    )
    tasks_root = run_dir / "tasks"
    unexpected = []
    if tasks_root.is_dir():
        for path in tasks_root.rglob("*"):
            relative = path.relative_to(tasks_root)
            if relative.parts and relative.parts[0] != "supervised-teacher":
                unexpected.append(relative.as_posix())
    _require(not unexpected, "parent contains implicit, null, or physical task evidence")


def verify_parent_boundary_control_run(path: str | Path) -> dict[str, Any]:
    """Verify the immutable boundary-control failure that authorizes scale repair.

    The returned mapping is intentionally JSON-serializable so a caller can
    write it directly into the new run's provenance artifact.  Any missing,
    modified, ambiguous, or scientifically incompatible evidence fails closed
    with :class:`ArtifactCompatibilityError`.
    """

    run_dir = Path(path).resolve()
    _require(run_dir.is_dir(), f"parent boundary-control run does not exist: {run_dir}")
    paths = {name: run_dir / relative for name, relative in _REQUIRED_FILES.items()}
    missing = [str(item) for item in paths.values() if not item.is_file()]
    _require(not missing, "parent boundary-control artifacts are missing: " + ", ".join(missing))
    values = {name: _load_mapping(item) for name, item in paths.items() if name != "registry"}
    manifest = values["manifest"]
    status = values["status"]
    _require(
        manifest.get("schema") == PARENT_RUN_SCHEMA
        and int(manifest.get("schema_version", -1)) == 1
        and status.get("schema") == PARENT_RUN_SCHEMA
        and int(status.get("schema_version", -1)) == 1,
        "parent boundary-control schema is incompatible",
    )
    _require(
        Path(str(manifest.get("run_dir", ""))).resolve() == run_dir,
        "parent manifest run directory is inconsistent",
    )
    _require(
        status.get("status") == "complete"
        and status.get("outcome") == "gate_failed"
        and status.get("decision") == "representation_invalid"
        and status.get("required_gate") == "controls"
        and _zero(status.get("required_gate_pass")),
        "parent terminal outcome is not the recorded supervised clipping failure",
    )
    _verify_terminal_registry(run_dir, status=status, registry_path=paths["registry"])
    scientific = _verify_scientific_config(manifest)

    operator = values["operator_preflight"]
    recomputed_preflight = evaluate_boundary_preflight(
        dict(operator.get("gate_metrics", {})), BoundaryControlThresholds()
    )
    _require(
        values["preflight_gate"] == recomputed_preflight and _one(recomputed_preflight.get("passed")),
        "parent boundary/operator preflight does not recompute as passing",
    )

    task_result_path = paths["supervised_task_result"]
    _require(
        file_fingerprint(paths["supervised"]) == file_fingerprint(task_result_path),
        "parent supervised summary is not the frozen task result",
    )
    supervised_gate, scientific_gate, optimizer_gate = _verify_supervised_failure(
        values["supervised"],
        task_result_path=task_result_path,
        task_status=values["supervised_task_status"],
    )
    _verify_skipped_implicit_controls(
        run_dir=run_dir,
        implicit_study=values["implicit_study"],
        null_study=values["null_study"],
        task_failures=values["task_failures"],
    )

    control_gate = values["control_gate"]
    components = control_gate.get("components", {})
    _require(isinstance(components, Mapping), "parent boundary-control components are missing")
    _require(
        _zero(control_gate.get("passed"))
        and dict(components.get("boundary_preflight", {})) == recomputed_preflight
        and dict(components.get("supervised_teacher", {})) == supervised_gate
        and dict(components.get("implicit_teacher_study", {})) == values["implicit_study"]
        and dict(components.get("null_study", {})) == values["null_study"]
        and _one(components.get("provenance")),
        "parent aggregate control gate is inconsistent",
    )
    report = values["control_report"]
    _require(
        dict(report.get("controls", {})) == control_gate
        and _one(report.get("preflight_pass"))
        and _zero(report.get("required_gate_pass")),
        "parent boundary-control report is inconsistent",
    )
    decision = values["decision"]
    _require(
        decision.get("decision") == "representation_invalid"
        and dict(report.get("decision", {})) == decision,
        "parent boundary-control decision is inconsistent",
    )

    for name, record in (
        ("manifest", manifest),
        ("status", status),
        ("report", report),
        ("decision", decision),
        ("control gate", control_gate),
    ):
        _require(_zero(record.get("sampling_performed")), f"parent {name} records sampling")
    _require(
        _zero(manifest.get("physical_training_performed"))
        and _zero(status.get("physical_training_performed"))
        and _zero(status.get("physical_training_authorized"))
        and _zero(status.get("sampling_authorized"))
        and _zero(decision.get("physical_training_authorized"))
        and _zero(decision.get("sampling_authorized")),
        "parent authorized or performed physical training or sampling",
    )

    transitive = _verify_transitive_provenance(
        values["failed_score_provenance"],
        manifest=manifest,
        scientific=scientific,
        operator=operator,
        provenance_path=paths["failed_score_provenance"],
    )
    schedule = dict(transitive["schedule_metadata"])
    return {
        "schema": PARENT_RUN_SCHEMA + "-scale-repair-parent-provenance",
        "schema_version": 1,
        "passed": 1,
        "run_dir": str(run_dir),
        "scientific_fingerprint": str(manifest["scientific_fingerprint"]),
        "scientific_config": scientific,
        "runtime_fingerprint": str(manifest.get("runtime_fingerprint", "")),
        "source_fingerprint": str(manifest.get("source_fingerprint", "")),
        "kernel": dict(scientific["kernel"]),
        "schedule_metadata": schedule,
        "horizon": float(schedule["horizon"]),
        "boundary_preflight_gate": recomputed_preflight,
        "supervised_teacher_gate": supervised_gate,
        "supervised_scientific_gate": scientific_gate,
        "supervised_optimizer_gate": optimizer_gate,
        "failed_only_post_warmup_clipping": 1,
        "implicit_teacher_status": "not_evaluated",
        "null_status": "not_evaluated",
        "transitive_failed_score_provenance": transitive,
        "artifacts": {name: _artifact_record(item) for name, item in paths.items()},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
