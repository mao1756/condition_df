from __future__ import annotations

import json
from pathlib import Path

import pytest

import mnist.d0_score_control_stability_provenance as provenance
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
    evaluate_loss_scale_calibration,
    split_supervised_teacher_gate,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": file_fingerprint(path),
        "size": path.stat().st_size,
    }


def _preflight_metrics() -> dict[str, object]:
    return {
        "potential_finite": 1,
        "gradient_finite": 1,
        "hvp_finite": 1,
        "generator_finite": 1,
        "energy_finite": 1,
        "incident_flux_loglog_slope": 0.95,
        "incident_flux_endpoint_ratio": 5e-4,
        "legacy_barrier_rejected": 1,
        "legacy_coefficient_relative_error": 0.01,
        "operator_pass": 1,
    }


def _teacher_metrics() -> dict[str, object]:
    return {
        "complete": 1,
        "finite": 1,
        "selected_step": 4000,
        "audit_overall_score_gain": 0.99,
        "audit_data_end_score_gain": 0.99,
        "overall_flux_cosine": 0.999,
        "time_bin_flux_cosines": [0.999] * 5,
        "overall_relative_flux_l2": 0.04,
        "time_bin_relative_flux_l2": [0.04] * 5,
        "boundary_admissible": 1,
        "post_warmup_clip_fraction": 0.0,
    }


def _calibration(kind: str, raw: float) -> dict[str, object]:
    scale = min(1.0, 0.1 / raw)
    return {
        "complete": 1,
        "finite": 1,
        "training_only": 1,
        "calibration_state_count": 256,
        "unscaled_initial_gradient_norm": raw,
        "target_initial_gradient_norm": 0.1,
        "loss_scale": scale,
        "scaled_initial_gradient_norm": raw * scale,
        "objective_kind": kind,
        "calibration_split": "train",
        "calibration_state_sha256": "states",
        "binding": {"scientific_fingerprint": "bound"},
        "shared_by_implicit_teacher_and_null": int(kind == "implicit_teacher"),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _write_semantic_parent(root: Path) -> tuple[Path, dict[str, Path]]:
    root.mkdir(parents=True)
    paths = {name: root / relative for name, relative in provenance._REQUIRED_FILES.items()}
    parent_provenance = {
        "schema": provenance.PARENT_PROVENANCE_SCHEMA,
        "schema_version": 1,
        "passed": 1,
        "run_dir": str((root / "upstream").resolve()),
        "schedule_metadata": {"horizon": 4.25e-4},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    _write(paths["parent_provenance"], parent_provenance)

    scientific = {
        "algorithm": provenance.PARENT_RUN_SCHEMA,
        "algorithm_version": 1,
        "kernel": dict(provenance.EXPECTED_KERNEL),
        "model": dict(provenance.EXPECTED_MODEL),
        "optimization": dict(provenance.EXPECTED_OPTIMIZATION),
        "thresholds": BoundaryControlThresholds().to_dict(),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    manifest = {
        "schema": provenance.PARENT_RUN_SCHEMA,
        "schema_version": 1,
        "run_dir": str(root.resolve()),
        "scientific_config": scientific,
        "scientific_fingerprint": config_fingerprint(scientific),
        "runtime_fingerprint": "runtime",
        "source_fingerprint": "source",
        "parent_provenance_sha256": file_fingerprint(paths["parent_provenance"]),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    _write(paths["manifest"], manifest)
    _write(paths["registry"], {})

    preflight_metrics = _preflight_metrics()
    preflight = evaluate_boundary_preflight(preflight_metrics)
    _write(paths["operator_preflight"], {"gate_metrics": preflight_metrics})
    _write(paths["preflight_gate"], preflight)

    supervised_cal = _calibration("supervised_teacher", 8.0)
    implicit_cal = _calibration(
        "implicit_teacher", 0.1 / provenance.PARENT_IMPLICIT_LOSS_SCALE
    )
    _write(paths["supervised_calibration"], supervised_cal)
    _write(paths["implicit_calibration"], implicit_cal)
    supervised_cal_gate = evaluate_loss_scale_calibration(
        supervised_cal,
        expected_initial_grad_target=0.1,
        expected_objective_kind="supervised_teacher",
    )
    implicit_cal_gate = evaluate_loss_scale_calibration(
        implicit_cal,
        expected_initial_grad_target=0.1,
        expected_objective_kind="implicit_teacher",
    )
    calibration_gate = {
        "schema": provenance.PARENT_RUN_SCHEMA + "-loss-scale-calibration-gate",
        "schema_version": 1,
        "passed": 1,
        "supervised": supervised_cal_gate,
        "implicit_shared_teacher_null": implicit_cal_gate,
        "sampling_performed": 0,
    }
    _write(paths["calibration_gate"], calibration_gate)

    supervised_metrics = _teacher_metrics()
    supervised_gate = evaluate_supervised_teacher(supervised_metrics)
    supervised = {"metrics": supervised_metrics, "gate": supervised_gate, "sampling_performed": 0}
    _write(paths["supervised"], supervised)
    split = split_supervised_teacher_gate(supervised_gate)

    studies: dict[str, dict[str, object]] = {}
    for law_dir, kind, filename in (
        ("implicit-teacher", "implicit_teacher", "implicit_study"),
        ("null", "null", "null_study"),
    ):
        results = []
        for seed in provenance.EXPECTED_SEEDS:
            metrics = {
                "complete": 1,
                "finite": 1,
                "selected_step": 0,
                "boundary_admissible": 1,
                "post_warmup_clip_fraction": 1.0,
            }
            result = {
                "task_kind": kind,
                "model_seed": seed,
                "metrics": metrics,
                "gate": {"passed": 0},
                "sampling_performed": 0,
            }
            result_path = root / "tasks" / law_dir / f"seed-{seed}" / "task_result.json"
            status_path = result_path.with_name("task_status.json")
            _write(result_path, result)
            _write(
                status_path,
                {
                    "status": "complete",
                    "task_kind": kind,
                    "model_seed": seed,
                    "task_result_sha256": file_fingerprint(result_path),
                },
            )
            results.append(result)
        study = {
            "evaluation_status": "evaluated",
            "passed": 0,
            "task_results": results,
            "seed_gates": [{"passed": 0}] * 3,
            "sampling_performed": 0,
        }
        studies[filename] = study
        _write(paths[filename], study)

    downstream = {
        "evaluation_status": "evaluated",
        "passed": 0,
        "task_count": 6,
        "expected_task_count": 6,
        "complete_task_set": 1,
        "task_gates": [{"passed": 0}] * 6,
    }
    components = {
        "provenance": {"passed": 1},
        "boundary_preflight": preflight,
        "supervised_calibration": supervised_cal_gate,
        "implicit_calibration": implicit_cal_gate,
        "supervised_optimizer": split["optimizer"],
        "supervised_representation": split["representation"],
        "downstream_optimizer": downstream,
        "implicit_teacher_study": studies["implicit_study"],
        "null_study": studies["null_study"],
    }
    gate = {
        "passed": 0,
        "probe_bank_status": "agree",
        "components": components,
        "sampling_performed": 0,
    }
    decision = {
        "decision": "optimizer_scale_invalid",
        "probe_bank_status": "agree",
        "physical_training_authorized": 0,
        "sampling_authorized": 0,
        "sampling_performed": 0,
    }
    report = {
        "schema": provenance.PARENT_RUN_SCHEMA + "-gate",
        "schema_version": 2,
        "controls": gate,
        "decision": decision,
        "preflight_pass": 1,
        "required_gate_pass": 0,
        "sampling_performed": 0,
    }
    _write(paths["control_gate"], gate)
    _write(paths["decision"], decision)
    _write(paths["control_report"], report)
    _write(paths["task_failures"], {"failure_count": 0, "failures": [], "skips": []})
    _write(
        paths["status"],
        {
            "schema": provenance.PARENT_RUN_SCHEMA,
            "schema_version": 1,
            "status": "complete",
            "outcome": "gate_failed",
            "decision": "optimizer_scale_invalid",
            "probe_bank_status": "agree",
            "required_gate": "controls",
            "required_gate_pass": 0,
            "physical_training_authorized": 0,
            "physical_training_performed": 0,
            "sampling_authorized": 0,
            "sampling_performed": 0,
        },
    )
    return root, paths


def _mock_registry() -> dict[str, object]:
    return {"records": {str(index): {} for index in range(231)}}


def test_accepts_exact_optimizer_only_parent_semantics(tmp_path: Path, monkeypatch) -> None:
    parent, _ = _write_semantic_parent(tmp_path / "parent")
    monkeypatch.setattr(provenance, "_verify_registry", lambda *args, **kwargs: _mock_registry())
    monkeypatch.setattr(provenance, "_verify_transitive_provenance", lambda value: None)

    result = provenance.verify_parent_scale_repair_run(parent)

    assert result["passed"] == 1
    assert result["terminal_registry_record_count"] == 231
    assert result["all_six_tasks_complete_finite_clipping_only"] == 1
    assert result["implicit_loss_scale"] == pytest.approx(provenance.PARENT_IMPLICIT_LOSS_SCALE)
    assert result["parent_decision"] == "optimizer_scale_invalid"
    assert result["physical_training_performed"] == result["sampling_performed"] == 0


def test_rejects_task_that_did_not_fail_optimizer_only_through_clipping(tmp_path: Path, monkeypatch) -> None:
    parent, _ = _write_semantic_parent(tmp_path / "parent")
    monkeypatch.setattr(provenance, "_verify_registry", lambda *args, **kwargs: _mock_registry())
    monkeypatch.setattr(provenance, "_verify_transitive_provenance", lambda value: None)
    result_path = parent / "tasks/implicit-teacher/seed-260785/task_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["metrics"]["post_warmup_clip_fraction"] = 0.05
    _write(result_path, result)
    status_path = result_path.with_name("task_status.json")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["task_result_sha256"] = file_fingerprint(result_path)
    _write(status_path, status)
    study_path = parent / "implicit_teacher_study.json"
    study = json.loads(study_path.read_text(encoding="utf-8"))
    study["task_results"][0] = result
    _write(study_path, study)

    with pytest.raises(ArtifactCompatibilityError, match="solely through clipping"):
        provenance.verify_parent_scale_repair_run(parent)


def test_terminal_registry_requires_exact_231_records_and_rejects_tampering(tmp_path: Path) -> None:
    root = tmp_path / "registry-parent"
    root.mkdir()
    records: dict[str, object] = {}
    for index in range(provenance.PARENT_REGISTRY_RECORD_COUNT):
        path = root / f"artifact-{index:03d}.json"
        _write(path, {"index": index})
        records[path.name] = _record(path)
    registry_path = root / "artifact_registry.json"
    status_path = root / "run_status.json"
    _write(
        registry_path,
        {
            "schema": provenance.PARENT_REGISTRY_SCHEMA,
            "schema_version": 1,
            "terminal_files_excluded_to_avoid_self_reference": [
                "artifact_registry.json",
                "run_status.json",
            ],
            "records": records,
        },
    )
    _write(
        status_path,
        {
            "artifact_registry_sha256": file_fingerprint(registry_path),
            "artifact_registry_size": registry_path.stat().st_size,
        },
    )
    verified = provenance._verify_registry(
        root,
        registry_path=registry_path,
        status_path=status_path,
        expected_schema=provenance.PARENT_REGISTRY_SCHEMA,
        expected_count=provenance.PARENT_REGISTRY_RECORD_COUNT,
    )
    assert len(verified["records"]) == 231

    with (root / "artifact-010.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ArtifactCompatibilityError, match="registered artifact"):
        provenance._verify_registry(
            root,
            registry_path=registry_path,
            status_path=status_path,
            expected_schema=provenance.PARENT_REGISTRY_SCHEMA,
            expected_count=provenance.PARENT_REGISTRY_RECORD_COUNT,
        )


def test_real_finished_parent_is_accepted_when_available() -> None:
    parent = Path(
        "runs/experiment12_d0_score_control_scale_repair/"
        "20260718-124405_production-boundary-control-scale-repair"
    )
    if not parent.is_dir():
        pytest.skip("production parent is not part of this test checkout")
    result = provenance.verify_parent_scale_repair_run(parent)
    assert result["passed"] == 1
    assert result["terminal_registry_record_count"] == 231
