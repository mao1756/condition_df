"""Strict parent provenance for streamed D0 control-stability confirmation.

The stability workflow is authorized by one specific terminal run: boundary
and supervised controls passed, the shared implicit scale calibrated correctly,
and all six implicit/null tasks completed finitely but violated optimizer
health through clipping.  This verifier binds that evidence, the complete
231-record terminal registry, and the earlier transitive provenance without
importing training code.
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
from mnist.d0_score_control_scale_repair_gate import (
    ProbeBankStatus,
    evaluate_loss_scale_calibration,
    evaluate_optimizer_task_health,
    split_supervised_teacher_gate,
)


__all__ = [
    "PARENT_RUN_SCHEMA",
    "PARENT_REGISTRY_RECORD_COUNT",
    "PARENT_IMPLICIT_LOSS_SCALE",
    "verify_parent_scale_repair_run",
]


PARENT_RUN_SCHEMA = "experiment12-d0-score-control-scale-repair"
PARENT_REGISTRY_SCHEMA = PARENT_RUN_SCHEMA + "-artifact-registry"
PARENT_PROVENANCE_SCHEMA = (
    "experiment12-d0-score-boundary-controls-scale-repair-parent-provenance"
)
PARENT_REGISTRY_RECORD_COUNT = 231
PARENT_IMPLICIT_LOSS_SCALE = 0.00266397028560976
EXPECTED_SEEDS = (260785, 260786, 260787)

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
    "selection_probes": 16,
    "audit_probes": 64,
    "calibration_state_count": 256,
    "supervised_initial_grad_target": 0.1,
    "implicit_initial_grad_target": 0.1,
}

_REQUIRED_FILES: dict[str, str] = {
    "manifest": "run_manifest.json",
    "status": "run_status.json",
    "registry": "artifact_registry.json",
    "parent_provenance": "parent_provenance.json",
    "operator_preflight": "boundary_operator_preflight.json",
    "preflight_gate": "boundary_preflight_gate.json",
    "calibration_gate": "loss_scale_calibration_gate.json",
    "supervised_calibration": "supervised_loss_scale_calibration.json",
    "implicit_calibration": "implicit_loss_scale_calibration.json",
    "supervised": "supervised_teacher_control.json",
    "implicit_study": "implicit_teacher_study.json",
    "null_study": "null_study.json",
    "control_gate": "boundary_control_gate.json",
    "control_report": "boundary_control_report.json",
    "decision": "control_repair_decision.json",
    "task_failures": "task_failures.json",
}


def _load(path: Path) -> dict[str, Any]:
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


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_fingerprint(path),
        "size": int(path.stat().st_size),
    }


def _verify_record(
    raw: Mapping[str, Any],
    *,
    root: Path | None,
    expected_path: Path | None = None,
    description: str,
) -> Path:
    path = Path(str(raw.get("path", ""))).resolve()
    if expected_path is not None:
        _require(path == expected_path.resolve(), f"{description} path mismatch")
    if root is not None:
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ArtifactCompatibilityError(f"{description} escapes its run directory") from exc
    _require(path.is_file(), f"{description} is missing: {path}")
    _require(
        raw.get("sha256") == file_fingerprint(path)
        and int(raw.get("size", -1)) == int(path.stat().st_size),
        f"{description} hash or size mismatch",
    )
    return path


def _verify_registry(
    run_dir: Path,
    *,
    registry_path: Path,
    status_path: Path,
    expected_schema: str | None = None,
    expected_count: int | None = None,
    expected_exclusions: set[str] | None = None,
    require_status_binding: bool = True,
    require_complete: bool = True,
) -> dict[str, Any]:
    registry = _load(registry_path)
    status = _load(status_path)
    if expected_schema is not None:
        _require(
            registry.get("schema") == expected_schema
            and int(registry.get("schema_version", -1)) == 1,
            "parent artifact registry schema is incompatible",
        )
    if require_status_binding:
        _require(
            status.get("artifact_registry_sha256") == file_fingerprint(registry_path)
            and int(status.get("artifact_registry_size", -1))
            == int(registry_path.stat().st_size),
            "terminal status does not bind its artifact registry",
        )
    exclusions = set(registry.get("terminal_files_excluded_to_avoid_self_reference", []))
    if expected_exclusions is None:
        expected_exclusions = {"artifact_registry.json", "run_status.json"}
    _require(
        exclusions == expected_exclusions,
        "artifact registry exclusions are incompatible",
    )
    raw_records = registry.get("records", {})
    _require(isinstance(raw_records, Mapping), "artifact registry records are invalid")
    records = dict(raw_records)
    if expected_count is not None:
        _require(
            len(records) == int(expected_count),
            f"parent terminal registry must contain exactly {expected_count} records",
        )
    filesystem_exclusions = exclusions | {"artifact_registry.json"}
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in filesystem_exclusions
    }
    if require_complete:
        _require(
            set(records) == actual,
            "artifact registry is incomplete or contains stale records",
        )
    for relative, raw in records.items():
        _require(isinstance(raw, Mapping), f"invalid registry record {relative}")
        _verify_record(
            raw,
            root=run_dir,
            expected_path=run_dir / relative,
            description=f"registered artifact {relative}",
        )
    return registry


def _verify_artifact_set(
    artifacts: Mapping[str, Any], *, root: Path, description: str
) -> dict[str, Path]:
    verified: dict[str, Path] = {}
    for name, raw in artifacts.items():
        if not isinstance(raw, Mapping):
            continue
        if not {"path", "sha256", "size"}.issubset(raw):
            continue
        verified[str(name)] = _verify_record(
            raw,
            root=root,
            description=f"{description} artifact {name}",
        )
    return verified


def _verify_transitive_provenance(parent: Mapping[str, Any]) -> None:
    _require(
        parent.get("schema") == PARENT_PROVENANCE_SCHEMA
        and int(parent.get("schema_version", -1)) == 1
        and _one(parent.get("passed")),
        "parent transitive provenance schema or gate is incompatible",
    )
    _require(
        _zero(parent.get("physical_training_performed"))
        and _zero(parent.get("sampling_performed")),
        "parent transitive provenance records physical work or sampling",
    )
    boundary_root = Path(str(parent.get("run_dir", ""))).resolve()
    _require(boundary_root.is_dir(), "transitive boundary-control run is missing")
    artifacts = parent.get("artifacts", {})
    _require(isinstance(artifacts, Mapping), "transitive boundary artifact records are missing")
    verified = _verify_artifact_set(
        dict(artifacts), root=boundary_root, description="transitive boundary-control"
    )
    _require(
        {"manifest", "status", "registry", "control_report"}.issubset(verified),
        "transitive boundary artifact records are incomplete",
    )
    _verify_registry(
        boundary_root,
        registry_path=verified["registry"],
        status_path=verified["status"],
    )

    failed = parent.get("transitive_failed_score_provenance", {})
    _require(isinstance(failed, Mapping) and _one(failed.get("passed")), "failed-score transitive provenance is invalid")
    _require(
        _zero(failed.get("physical_training_performed"))
        and _zero(failed.get("sampling_performed")),
        "transitive failed-score run records physical work or sampling",
    )
    failed_root = Path(str(failed.get("run_dir", ""))).resolve()
    _require(failed_root.is_dir(), "transitive failed-score run is missing")
    failed_artifacts = failed.get("artifacts", {})
    _require(isinstance(failed_artifacts, Mapping), "failed-score artifact records are missing")
    failed_verified = _verify_artifact_set(
        dict(failed_artifacts), root=failed_root, description="transitive failed-score"
    )
    registry_key = "artifact_registry" if "artifact_registry" in failed_verified else "registry"
    _require(
        {"manifest", "status", registry_key}.issubset(failed_verified),
        "failed-score artifact records are incomplete",
    )
    failed_manifest = _load(failed_verified["manifest"])
    manifest_registry = dict(
        dict(failed_manifest.get("artifacts", {})).get("artifact_registry", {})
    )
    _require(
        manifest_registry == dict(failed_artifacts[registry_key]),
        "transitive failed-score manifest does not bind its artifact registry",
    )
    _verify_registry(
        failed_root,
        registry_path=failed_verified[registry_key],
        status_path=failed_verified["status"],
        expected_exclusions=set(),
        require_status_binding=False,
        require_complete=False,
    )


def _verify_scientific_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    scientific = manifest.get("scientific_config", {})
    _require(isinstance(scientific, Mapping), "parent scientific configuration is missing")
    scientific = dict(scientific)
    _require(
        scientific.get("algorithm") == PARENT_RUN_SCHEMA
        and int(scientific.get("algorithm_version", -1)) == 1,
        "parent scientific algorithm is incompatible",
    )
    _require(
        config_fingerprint(scientific) == manifest.get("scientific_fingerprint"),
        "parent scientific fingerprint is inconsistent",
    )
    for section, expected in (
        ("kernel", EXPECTED_KERNEL),
        ("model", EXPECTED_MODEL),
        ("optimization", EXPECTED_OPTIMIZATION),
    ):
        actual = scientific.get(section, {})
        _require(isinstance(actual, Mapping), f"parent {section} configuration is missing")
        for key, value in expected.items():
            _require(
                _same(actual.get(key), value),
                f"parent {section} mismatch for {key}: {actual.get(key)!r}",
            )
    _require(
        scientific.get("thresholds") == BoundaryControlThresholds().to_dict(),
        "parent scientific thresholds are not frozen boundary-control thresholds",
    )
    _require(
        _zero(scientific.get("physical_training_performed"))
        and _zero(scientific.get("sampling_performed")),
        "parent scientific configuration records physical work or sampling",
    )
    return scientific


def _verify_calibrations(values: Mapping[str, Mapping[str, Any]]) -> None:
    supervised = evaluate_loss_scale_calibration(
        values["supervised_calibration"],
        expected_initial_grad_target=0.1,
        expected_objective_kind="supervised_teacher",
    )
    implicit = evaluate_loss_scale_calibration(
        values["implicit_calibration"],
        expected_initial_grad_target=0.1,
        expected_objective_kind="implicit_teacher",
    )
    stored = values["calibration_gate"]
    _require(
        stored.get("schema")
        == PARENT_RUN_SCHEMA + "-loss-scale-calibration-gate"
        and int(stored.get("schema_version", -1)) == 1
        and _one(stored.get("passed"))
        and dict(stored.get("supervised", {})) == supervised
        and dict(stored.get("implicit_shared_teacher_null", {})) == implicit,
        "parent loss-scale calibration gates do not recompute",
    )
    record = values["implicit_calibration"]
    _require(
        math.isclose(
            float(record.get("loss_scale", math.nan)),
            PARENT_IMPLICIT_LOSS_SCALE,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and _one(record.get("shared_by_implicit_teacher_and_null")),
        "parent implicit/null loss multiplier is incompatible",
    )


def _verify_supervised(values: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    supervised = values["supervised"]
    metrics = supervised.get("metrics", {})
    _require(isinstance(metrics, Mapping), "parent supervised metrics are missing")
    recomputed = evaluate_supervised_teacher(dict(metrics), BoundaryControlThresholds())
    _require(
        dict(supervised.get("gate", {})) == recomputed,
        "parent supervised teacher gate does not recompute",
    )
    split = split_supervised_teacher_gate(recomputed)
    _require(
        _one(split["optimizer"].get("passed"))
        and _one(split["representation"].get("passed")),
        "parent supervised optimizer or representation control did not pass",
    )
    return split["optimizer"], split["representation"]


def _task_paths(run_dir: Path, law: str, seed: int) -> tuple[Path, Path]:
    root = run_dir / "tasks" / law / f"seed-{seed}"
    return root / "task_result.json", root / "task_status.json"


def _verify_downstream_tasks(
    run_dir: Path,
    *,
    implicit_study: Mapping[str, Any],
    null_study: Mapping[str, Any],
) -> list[dict[str, Any]]:
    task_gates: list[dict[str, Any]] = []
    for law_dir, task_kind, study in (
        ("implicit-teacher", "implicit_teacher", implicit_study),
        ("null", "null", null_study),
    ):
        _require(
            study.get("evaluation_status") == "evaluated"
            and _zero(study.get("passed"))
            and len(list(study.get("task_results", []))) == 3,
            f"parent {task_kind} study is not the completed failed study",
        )
        by_seed = {
            int(value.get("model_seed", -1)): dict(value)
            for value in study.get("task_results", [])
            if isinstance(value, Mapping)
        }
        _require(set(by_seed) == set(EXPECTED_SEEDS), f"parent {task_kind} seeds are incompatible")
        for seed in EXPECTED_SEEDS:
            result_path, status_path = _task_paths(run_dir, law_dir, seed)
            _require(result_path.is_file() and status_path.is_file(), f"parent {task_kind} task {seed} is missing")
            result = _load(result_path)
            status = _load(status_path)
            _require(result == by_seed[seed], f"parent {task_kind} study/task result mismatch for seed {seed}")
            _require(
                status.get("status") == "complete"
                and status.get("task_kind") == task_kind
                and int(status.get("model_seed", -1)) == seed
                and status.get("task_result_sha256") == file_fingerprint(result_path),
                f"parent {task_kind} task status is incomplete or unbound for seed {seed}",
            )
            metrics = result.get("metrics", {})
            _require(isinstance(metrics, Mapping), f"parent {task_kind} task metrics are missing")
            optimizer = evaluate_optimizer_task_health(dict(metrics), BoundaryControlThresholds())
            failed = [
                name
                for name, check in optimizer.get("subchecks", {}).items()
                if not _one(dict(check).get("passed"))
            ]
            _require(
                failed == ["post_warmup_clip_fraction"]
                and _one(metrics.get("boundary_admissible")),
                f"parent {task_kind} task {seed} did not fail optimizer health solely through clipping",
            )
            _require(_zero(result.get("sampling_performed")), f"parent {task_kind} task sampled")
            task_gates.append(
                {
                    "task_kind": task_kind,
                    "model_seed": seed,
                    "optimizer_gate": optimizer,
                }
            )
    return task_gates


def verify_parent_scale_repair_run(path: str | Path) -> dict[str, Any]:
    """Verify the immutable scale-repair failure authorizing stability work."""

    run_dir = Path(path).resolve()
    _require(run_dir.is_dir(), f"parent scale-repair run does not exist: {run_dir}")
    paths = {name: run_dir / relative for name, relative in _REQUIRED_FILES.items()}
    missing = [str(value) for value in paths.values() if not value.is_file()]
    _require(not missing, "parent scale-repair artifacts are missing: " + ", ".join(missing))
    values = {name: _load(value) for name, value in paths.items() if name != "registry"}
    manifest = values["manifest"]
    status = values["status"]
    _require(
        manifest.get("schema") == PARENT_RUN_SCHEMA
        and int(manifest.get("schema_version", -1)) == 1
        and status.get("schema") == PARENT_RUN_SCHEMA
        and int(status.get("schema_version", -1)) == 1,
        "parent scale-repair schema is incompatible",
    )
    _require(
        Path(str(manifest.get("run_dir", ""))).resolve() == run_dir,
        "parent manifest run directory is inconsistent",
    )
    _require(
        status.get("status") == "complete"
        and status.get("outcome") == "gate_failed"
        and status.get("decision") == "optimizer_scale_invalid"
        and status.get("probe_bank_status") == ProbeBankStatus.AGREE.value
        and status.get("required_gate") == "controls"
        and _zero(status.get("required_gate_pass")),
        "parent terminal outcome is not the recorded downstream clipping failure",
    )
    registry = _verify_registry(
        run_dir,
        registry_path=paths["registry"],
        status_path=paths["status"],
        expected_schema=PARENT_REGISTRY_SCHEMA,
        expected_count=PARENT_REGISTRY_RECORD_COUNT,
    )
    scientific = _verify_scientific_manifest(manifest)
    _require(
        manifest.get("parent_provenance_sha256") == file_fingerprint(paths["parent_provenance"]),
        "parent manifest does not bind transitive provenance",
    )
    _verify_transitive_provenance(values["parent_provenance"])

    operator = values["operator_preflight"]
    metrics = operator.get("gate_metrics", {})
    _require(isinstance(metrics, Mapping), "parent boundary/operator metrics are missing")
    recomputed_preflight = evaluate_boundary_preflight(dict(metrics), BoundaryControlThresholds())
    _require(
        values["preflight_gate"] == recomputed_preflight and _one(recomputed_preflight.get("passed")),
        "parent boundary/operator preflight does not recompute as passing",
    )
    _verify_calibrations(values)
    supervised_optimizer, supervised_representation = _verify_supervised(values)
    task_gates = _verify_downstream_tasks(
        run_dir,
        implicit_study=values["implicit_study"],
        null_study=values["null_study"],
    )

    failures = values["task_failures"]
    _require(
        int(failures.get("failure_count", -1)) == 0
        and list(failures.get("failures", [])) == []
        and list(failures.get("skips", [])) == [],
        "parent contains task failures or skipped controls",
    )
    report = values["control_report"]
    gate = values["control_gate"]
    decision = values["decision"]
    _require(
        report.get("schema") == PARENT_RUN_SCHEMA + "-gate"
        and int(report.get("schema_version", -1)) == 2
        and dict(report.get("controls", {})) == gate
        and dict(report.get("decision", {})) == decision
        and _one(report.get("preflight_pass"))
        and _zero(report.get("required_gate_pass")),
        "parent aggregate control report is inconsistent",
    )
    components = gate.get("components", {})
    _require(isinstance(components, Mapping), "parent aggregate components are missing")
    expected_passes = (
        "provenance",
        "boundary_preflight",
        "supervised_calibration",
        "implicit_calibration",
        "supervised_optimizer",
        "supervised_representation",
    )
    _require(
        _zero(gate.get("passed"))
        and gate.get("probe_bank_status") == ProbeBankStatus.AGREE.value
        and all(_one(dict(components.get(name, {})).get("passed", components.get(name))) for name in expected_passes)
        and _zero(dict(components.get("downstream_optimizer", {})).get("passed"))
        and dict(components.get("supervised_optimizer", {})) == supervised_optimizer
        and dict(components.get("supervised_representation", {})) == supervised_representation,
        "parent aggregate gate is not the optimizer-only failure",
    )
    downstream = dict(components.get("downstream_optimizer", {}))
    _require(
        downstream.get("evaluation_status") == "evaluated"
        and int(downstream.get("task_count", -1)) == 6
        and int(downstream.get("expected_task_count", -1)) == 6
        and _one(downstream.get("complete_task_set"))
        and len(list(downstream.get("task_gates", []))) == 6,
        "parent downstream optimizer gate is incomplete",
    )
    _require(
        decision.get("decision") == "optimizer_scale_invalid"
        and decision.get("probe_bank_status") == ProbeBankStatus.AGREE.value
        and _zero(decision.get("physical_training_authorized"))
        and _zero(decision.get("sampling_authorized")),
        "parent terminal decision is incompatible",
    )
    for name, value in (
        ("manifest", manifest),
        ("status", status),
        ("report", report),
        ("decision", decision),
        ("gate", gate),
    ):
        _require(
            _zero(value.get("physical_training_performed", 0))
            and _zero(value.get("sampling_performed", 0)),
            f"parent {name} records physical training or sampling",
        )
    _require(
        _zero(status.get("physical_training_authorized"))
        and _zero(status.get("sampling_authorized")),
        "parent status authorized physical training or sampling",
    )

    parent_provenance = values["parent_provenance"]
    schedule = parent_provenance.get("schedule_metadata", {})
    _require(isinstance(schedule, Mapping), "parent schedule metadata is missing")
    horizon = schedule.get("horizon", parent_provenance.get("horizon"))
    _require(
        isinstance(horizon, (int, float)) and math.isfinite(float(horizon)) and float(horizon) > 0.0,
        "parent schedule horizon is invalid",
    )
    return {
        "schema": PARENT_RUN_SCHEMA + "-stability-parent-provenance",
        "schema_version": 1,
        "passed": 1,
        "run_dir": str(run_dir),
        "scientific_fingerprint": str(manifest.get("scientific_fingerprint", "")),
        "runtime_fingerprint": str(manifest.get("runtime_fingerprint", "")),
        "source_fingerprint": str(manifest.get("source_fingerprint", "")),
        "scientific_config": scientific,
        "kernel": dict(scientific["kernel"]),
        "model": dict(scientific["model"]),
        "schedule_metadata": dict(schedule),
        "horizon": float(horizon),
        "implicit_loss_scale": PARENT_IMPLICIT_LOSS_SCALE,
        "parent_decision": "optimizer_scale_invalid",
        "probe_bank_status": ProbeBankStatus.AGREE.value,
        "terminal_registry_record_count": len(dict(registry["records"])),
        "all_six_tasks_complete_finite_clipping_only": 1,
        "downstream_optimizer_task_gates": task_gates,
        "transitive_parent_provenance": parent_provenance,
        "artifacts": {name: _record(value) for name, value in paths.items()},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
