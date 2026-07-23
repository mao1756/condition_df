"""Fail-closed provenance contracts for the D0 optimizer-scale repair."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

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
from mnist.d0_score_control_scale_repair_provenance import (
    EXPECTED_KERNEL,
    EXPECTED_MODEL,
    EXPECTED_OPTIMIZATION,
    EXPECTED_SYNTHETIC_DATA,
    FAILED_SCORE_PROVENANCE_SCHEMA,
    PARENT_REGISTRY_SCHEMA,
    PARENT_RUN_SCHEMA,
    verify_parent_boundary_control_run,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_fingerprint(path),
        "size": int(path.stat().st_size),
    }


def _passing_preflight_metrics() -> dict[str, Any]:
    return {
        "potential_finite": 1,
        "gradient_finite": 1,
        "hvp_finite": 1,
        "generator_finite": 1,
        "energy_finite": 1,
        "incident_flux_loglog_slope": 0.99,
        "incident_flux_endpoint_ratio": 1e-4,
        "legacy_barrier_rejected": 1,
        "legacy_coefficient_relative_error": 0.01,
        "operator_pass": 1,
        "orthogonal_probe_pass": 1,
        "aggregate_preflight_pass": 1,
        "production_workload_smoke_pass": 1,
    }


def _supervised_metrics(*, clip_fraction: float = 0.95) -> dict[str, Any]:
    return {
        "complete": 1,
        "finite": 1,
        "selected_step": 3750,
        "audit_overall_score_gain": 0.998,
        "audit_data_end_score_gain": 0.998,
        "overall_flux_cosine": 0.999,
        "time_bin_flux_cosines": [0.998] * 5,
        "overall_relative_flux_l2": 0.04,
        "time_bin_relative_flux_l2": [0.04] * 5,
        "boundary_admissible": 1,
        "post_warmup_clip_fraction": clip_fraction,
    }


def _write_upstream_failed_score(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True)
    values = {
        "status": {
            "status": "complete",
            "outcome": "gate_failed",
            "decision": "optimization_pipeline_invalid",
            "sampling_performed": 0,
        },
        "preflight": {"passed": 1},
        "cache": {"passed": 1},
        "controls": {"passed": 0},
        "operator": {"passed": 1},
        "cache_index": {"metadata": {"schedule_metadata": {"horizon": 4.25e-4}}},
        "artifact_registry": {"schema": "upstream-fixture-registry", "records": {}},
    }
    paths = {
        "status": root / "run_status.json",
        "preflight": root / "preflight_gate.json",
        "cache": root / "cache_gate.json",
        "controls": root / "controls_gate.json",
        "operator": root / "operator_preflight.json",
        "cache_index": root / "cache" / "parent" / "cache_index.json",
        "artifact_registry": root / "artifact_registry.json",
        "manifest": root / "run_manifest.json",
    }
    for name, value in values.items():
        _write_json(paths[name], value)
    registry_record = _record(paths["artifact_registry"])
    _write_json(
        paths["manifest"],
        {
            "schema": "experiment12-d0-dirichlet-score-learnability",
            "scientific_fingerprint": "failed-score-science",
            "artifacts": {"artifact_registry": registry_record},
        },
    )
    return {
        "schema": FAILED_SCORE_PROVENANCE_SCHEMA,
        "schema_version": 1,
        "passed": 1,
        "run_dir": str(root.resolve()),
        "scientific_fingerprint": "failed-score-science",
        "schedule_metadata": {"horizon": 4.25e-4, "sample_steps": 512},
        "artifacts": {name: _record(path) for name, path in paths.items()},
        "failed_teacher_gate": {"passed": 0},
        "failed_null_gate": {"passed": 0},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _finalize_registry(root: Path) -> None:
    registry_path = root / "artifact_registry.json"
    status_path = root / "run_status.json"
    exclusions = {"artifact_registry.json", "run_status.json"}
    records = {
        path.relative_to(root).as_posix(): _record(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in exclusions
    }
    _write_json(
        registry_path,
        {
            "schema": PARENT_REGISTRY_SCHEMA,
            "schema_version": 1,
            "records": records,
            "terminal_files_excluded_to_avoid_self_reference": sorted(exclusions),
            "sampling_performed": 0,
        },
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["artifact_registry_sha256"] = file_fingerprint(registry_path)
    status["artifact_registry_size"] = int(registry_path.stat().st_size)
    _write_json(status_path, status)


def _write_parent(root: Path) -> tuple[Path, dict[str, Path]]:
    upstream_provenance = _write_upstream_failed_score(root.parent / "failed-score")
    root.mkdir()
    paths = {
        "manifest": root / "run_manifest.json",
        "status": root / "run_status.json",
        "operator": root / "boundary_operator_preflight.json",
        "preflight": root / "boundary_preflight_gate.json",
        "control_gate": root / "boundary_control_gate.json",
        "control_report": root / "boundary_control_report.json",
        "decision": root / "control_repair_decision.json",
        "supervised": root / "supervised_teacher_control.json",
        "task_result": root / "tasks" / "supervised-teacher" / "task_result.json",
        "task_status": root / "tasks" / "supervised-teacher" / "task_status.json",
        "implicit": root / "implicit_teacher_study.json",
        "null": root / "null_study.json",
        "task_failures": root / "task_failures.json",
        "provenance": root / "failed_run_provenance.json",
    }
    _write_json(paths["provenance"], upstream_provenance)
    scientific = {
        "algorithm": PARENT_RUN_SCHEMA,
        "algorithm_version": 1,
        "kernel": dict(EXPECTED_KERNEL),
        "model": dict(EXPECTED_MODEL),
        "synthetic_data": dict(EXPECTED_SYNTHETIC_DATA),
        "optimization": dict(EXPECTED_OPTIMIZATION),
        "thresholds": BoundaryControlThresholds().to_dict(),
        "parent_failed_run_sha256": upstream_provenance["artifacts"]["status"]["sha256"],
        "parent_scientific_fingerprint": upstream_provenance["scientific_fingerprint"],
        "sampling_performed": 0,
    }
    fingerprint = config_fingerprint(scientific)
    _write_json(
        paths["manifest"],
        {
            "schema": PARENT_RUN_SCHEMA,
            "schema_version": 1,
            "run_dir": str(root.resolve()),
            "scientific_config": scientific,
            "scientific_fingerprint": fingerprint,
            "runtime_fingerprint": "runtime-fingerprint",
            "source_fingerprint": "source-fingerprint",
            "failed_run_provenance_sha256": file_fingerprint(paths["provenance"]),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    preflight_metrics = _passing_preflight_metrics()
    _write_json(
        paths["operator"],
        {
            "binding": {
                "failed_run_status_sha256": upstream_provenance["artifacts"]["status"]["sha256"],
                "scientific_fingerprint": fingerprint,
                "runtime_fingerprint": "runtime-fingerprint",
                "source_fingerprint": "source-fingerprint",
            },
            "gate_metrics": preflight_metrics,
            "sampling_performed": 0,
        },
    )
    preflight = evaluate_boundary_preflight(preflight_metrics)
    _write_json(paths["preflight"], preflight)
    metrics = _supervised_metrics()
    supervised_gate = evaluate_supervised_teacher(metrics)
    supervised = {
        "task_kind": "supervised_teacher",
        "metrics": metrics,
        "gate": supervised_gate,
        "complete": 1,
        "finite": 1,
        "sampling_performed": 0,
    }
    _write_json(paths["supervised"], supervised)
    _write_json(paths["task_result"], supervised)
    _write_json(
        paths["task_status"],
        {
            "status": "complete",
            "task_kind": "supervised_teacher",
            "task_result_sha256": file_fingerprint(paths["task_result"]),
            "sampling_performed": 0,
        },
    )
    implicit = {"gate": "implicit_teacher_study", "passed": 0, "seed_gates": [], "sampling_performed": 0}
    null = {"gate": "null_study", "passed": 0, "seed_gates": [], "sampling_performed": 0}
    _write_json(paths["implicit"], implicit)
    _write_json(paths["null"], null)
    _write_json(
        paths["task_failures"],
        {
            "failure_count": 0,
            "failures": [],
            "skips": [{"stage": "implicit_controls", "reason": "supervised analytic teacher failed"}],
        },
    )
    decision = {
        "decision": "representation_invalid",
        "physical_training_authorized": 0,
        "sampling_authorized": 0,
        "sampling_performed": 0,
    }
    _write_json(paths["decision"], decision)
    control_gate = {
        "gate": "boundary_controls",
        "passed": 0,
        "components": {
            "boundary_preflight": preflight,
            "supervised_teacher": supervised_gate,
            "implicit_teacher_study": implicit,
            "null_study": null,
            "provenance": 1,
        },
        "sampling_performed": 0,
    }
    _write_json(paths["control_gate"], control_gate)
    _write_json(
        paths["control_report"],
        {
            "controls": control_gate,
            "decision": decision,
            "preflight_pass": 1,
            "required_gate_pass": 0,
            "sampling_performed": 0,
        },
    )
    _write_json(
        paths["status"],
        {
            "schema": PARENT_RUN_SCHEMA,
            "schema_version": 1,
            "status": "complete",
            "outcome": "gate_failed",
            "decision": "representation_invalid",
            "required_gate": "controls",
            "required_gate_pass": 0,
            "physical_training_authorized": 0,
            "physical_training_performed": 0,
            "sampling_authorized": 0,
            "sampling_performed": 0,
        },
    )
    _finalize_registry(root)
    return root, paths


def _replace_supervised_metrics(
    root: Path,
    paths: dict[str, Path],
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    supervised = json.loads(paths["supervised"].read_text(encoding="utf-8"))
    mutate(supervised["metrics"])
    supervised["gate"] = evaluate_supervised_teacher(supervised["metrics"])
    _write_json(paths["supervised"], supervised)
    _write_json(paths["task_result"], supervised)
    task_status = json.loads(paths["task_status"].read_text(encoding="utf-8"))
    task_status["task_result_sha256"] = file_fingerprint(paths["task_result"])
    _write_json(paths["task_status"], task_status)
    control_gate = json.loads(paths["control_gate"].read_text(encoding="utf-8"))
    control_gate["components"]["supervised_teacher"] = supervised["gate"]
    _write_json(paths["control_gate"], control_gate)
    report = json.loads(paths["control_report"].read_text(encoding="utf-8"))
    report["controls"] = control_gate
    _write_json(paths["control_report"], report)
    _finalize_registry(root)


def test_accepts_parent_that_failed_only_supervised_clipping(tmp_path: Path) -> None:
    parent, _ = _write_parent(tmp_path / "boundary-parent")

    result = verify_parent_boundary_control_run(parent)

    assert result["passed"] == 1
    assert result["failed_only_post_warmup_clipping"] == 1
    assert result["supervised_scientific_gate"]["passed"] == 1
    assert result["supervised_optimizer_gate"]["passed"] == 0
    assert result["implicit_teacher_status"] == "not_evaluated"
    assert result["null_status"] == "not_evaluated"
    assert result["horizon"] == pytest.approx(4.25e-4)
    assert result["artifacts"]["status"]["sha256"] == file_fingerprint(parent / "run_status.json")
    assert result["transitive_failed_score_provenance"]["verified_upstream_status_sha256"]
    assert result["physical_training_performed"] == result["sampling_performed"] == 0


def test_rejects_complete_registry_artifact_tampering(tmp_path: Path) -> None:
    parent, paths = _write_parent(tmp_path / "boundary-parent")
    with paths["supervised"].open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(ArtifactCompatibilityError, match="registered parent artifact"):
        verify_parent_boundary_control_run(parent)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda metrics: metrics.__setitem__("audit_overall_score_gain", 0.5), "solely"),
        (lambda metrics: metrics.__setitem__("post_warmup_clip_fraction", 0.05), "solely"),
    ],
)
def test_rejects_scientific_failure_or_nonfailed_clipping(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    parent, paths = _write_parent(tmp_path / "boundary-parent")
    _replace_supervised_metrics(parent, paths, mutation)

    with pytest.raises(ArtifactCompatibilityError, match=message):
        verify_parent_boundary_control_run(parent)


def test_rejects_evaluated_implicit_or_null_tasks(tmp_path: Path) -> None:
    parent, paths = _write_parent(tmp_path / "boundary-parent")
    implicit = json.loads(paths["implicit"].read_text(encoding="utf-8"))
    implicit["seed_gates"] = [{"model_seed": 260781, "passed": 0}]
    _write_json(paths["implicit"], implicit)
    control_gate = json.loads(paths["control_gate"].read_text(encoding="utf-8"))
    control_gate["components"]["implicit_teacher_study"] = implicit
    _write_json(paths["control_gate"], control_gate)
    report = json.loads(paths["control_report"].read_text(encoding="utf-8"))
    report["controls"] = control_gate
    _write_json(paths["control_report"], report)
    _finalize_registry(parent)

    with pytest.raises(ArtifactCompatibilityError, match="contains evaluated tasks"):
        verify_parent_boundary_control_run(parent)


def test_rejects_physical_training_or_sampling_flags(tmp_path: Path) -> None:
    parent, paths = _write_parent(tmp_path / "boundary-parent")
    status = json.loads(paths["status"].read_text(encoding="utf-8"))
    status["physical_training_performed"] = 1
    _write_json(paths["status"], status)

    with pytest.raises(ArtifactCompatibilityError, match="physical training"):
        verify_parent_boundary_control_run(parent)


def test_rejects_modified_transitive_failed_score_artifact(tmp_path: Path) -> None:
    parent, paths = _write_parent(tmp_path / "boundary-parent")
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    upstream_cache = Path(provenance["artifacts"]["cache"]["path"])
    with upstream_cache.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(ArtifactCompatibilityError, match="transitive failed-score artifact cache"):
        verify_parent_boundary_control_run(parent)


def test_rejects_unregistered_extra_parent_artifact(tmp_path: Path) -> None:
    parent, _ = _write_parent(tmp_path / "boundary-parent")
    (parent / "unexpected.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="incomplete or contains stale"):
        verify_parent_boundary_control_run(parent)
