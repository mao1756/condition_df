from __future__ import annotations

import json
from pathlib import Path

import pytest

import mnist.d0_score_density_ratio_stability_provenance as provenance
from mnist.d0_one_image_gate import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
)
from mnist.d0_score_density_ratio_gate import DensityRatioThresholds


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record(path: Path) -> dict[str, object]:
    return {"sha256": file_fingerprint(path), "size": int(path.stat().st_size)}


def _scientific() -> dict[str, object]:
    return {
        "algorithm": provenance.PARENT_RUN_SCHEMA,
        "algorithm_version": 1,
        "kernel": dict(provenance.EXPECTED_KERNEL),
        "model_schema": "d0-boundary-smooth-potential-unet-v1",
        "model_schema_version": 1,
        "objective_version": "d0-balanced-raw-logit-bce-v1",
        "panel_version": "d0-density-ratio-panel-v1",
        "optimization": {
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "ema_decay": 0.99,
            "grad_clip": 1.0,
            "clip_warmup_steps": 500,
            "calibration_state_count": 256,
            "initial_grad_target": 0.1,
            "adaptive_loss_scaling": 0,
        },
        "pilot": {
            "learning_rates": list(provenance.EXPECTED_PILOT_LEARNING_RATES),
            "steps": 2000,
            "selection_paths_per_panel": 16,
            "audit_paths": 0,
            "selection_panels": ["a", "b"],
        },
        "confirmation": {
            "model_seeds": [260831, 260832, 260833],
            "steps": 4000,
            "selection_paths_per_panel": 32,
            "audit_paths_per_panel": 32,
            "audit_panels": ["c", "d"],
        },
        "preflight": {"paths": 128, "confidence": 0.99},
        "stream": {
            "batch_size": 64,
            "examples_per_class": 32,
            "class_prior": 0.5,
            "class_bin_counts": [4, 4, 4, 4, 16],
        },
        "thresholds": json.loads(json.dumps(DensityRatioThresholds().to_dict())),
        "root_seed": 260821,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def test_scientific_manifest_rejects_frozen_threshold_or_optimizer_changes() -> None:
    scientific = _scientific()
    manifest = {
        "scientific_config": scientific,
        "scientific_fingerprint": config_fingerprint(scientific),
    }
    assert provenance._verify_scientific_manifest(manifest) == scientific

    changed = json.loads(json.dumps(scientific))
    changed["thresholds"]["maximum_clip_fraction"] = 0.11
    with pytest.raises(ArtifactCompatibilityError, match="thresholds changed"):
        provenance._verify_scientific_manifest(
            {
                "scientific_config": changed,
                "scientific_fingerprint": config_fingerprint(changed),
            }
        )

    changed = json.loads(json.dumps(scientific))
    changed["optimization"]["adaptive_loss_scaling"] = 1
    with pytest.raises(ArtifactCompatibilityError, match="optimizer mismatch"):
        provenance._verify_scientific_manifest(
            {
                "scientific_config": changed,
                "scientific_fingerprint": config_fingerprint(changed),
            }
        )


def test_registry_requires_exact_222_records_and_detects_tampering(tmp_path: Path) -> None:
    root = tmp_path / "parent"
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
        root, registry_path=registry_path, status_path=status_path
    )
    assert len(verified["records"]) == 222

    with (root / "artifact-010.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ArtifactCompatibilityError, match="registered density-ratio"):
        provenance._verify_registry(
            root, registry_path=registry_path, status_path=status_path
        )


def test_terminal_verifier_rejects_wrong_parent_outcome_first(tmp_path: Path) -> None:
    root = tmp_path / "parent"
    root.mkdir()
    for relative in provenance._REQUIRED_FILES.values():
        _write(root / relative, {})
    _write(
        root / "run_manifest.json",
        {
            "schema": provenance.PARENT_RUN_SCHEMA,
            "schema_version": 1,
            "run_dir": str(root.resolve()),
        },
    )
    _write(
        root / "run_status.json",
        {
            "schema": provenance.PARENT_RUN_SCHEMA,
            "schema_version": 1,
            "status": "complete",
            "outcome": "gate_failed",
            "phase": "pilot",
            "stage": "pilot",
            "decision": "density_ratio_control_pipeline_repaired",
            "required_gate": "pilot",
            "required_gate_pass": 0,
        },
    )
    with pytest.raises(ArtifactCompatibilityError, match="unresolved pilot"):
        provenance.verify_parent_density_ratio_run(root)


def test_transitive_parent_requires_exact_recursive_381_binding(monkeypatch) -> None:
    recomputed = {
        "schema": "upstream-recomputed-schema",
        "schema_version": 1,
        "passed": 1,
        "run_dir": "upstream",
        "terminal_registry_record_count": 381,
        "scientific_fingerprint": "expected",
        "schedule_metadata": {"horizon": 1.0},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    monkeypatch.setattr(
        provenance, "verify_parent_stability_run", lambda path: recomputed
    )
    stored = {
        **recomputed,
        "schema": "experiment12-d0-score-density-ratio-controls-parent-provenance",
        "verifier_source_fingerprint": "source",
    }
    assert provenance._verify_transitive_parent(stored) == recomputed
    stored["scientific_fingerprint"] = "tampered"
    with pytest.raises(ArtifactCompatibilityError, match="381-record"):
        provenance._verify_transitive_parent(stored)


def test_task_result_status_and_frozen_loss_scale_binding(tmp_path: Path) -> None:
    result_path = tmp_path / "task_result.json"
    status_path = tmp_path / "task_status.json"
    fingerprints = {
        "learning_rate": 3e-5,
        "loss_scale": provenance.PARENT_LOSS_SCALE,
        "model_seed": provenance.EXPECTED_PARENT_MODEL_SEED,
        "task": "bounded_teacher",
    }
    result = {
        "task": "bounded_teacher",
        "model_seed": provenance.EXPECTED_PARENT_MODEL_SEED,
        "fingerprints": fingerprints,
        "metrics": {"complete": 1, "finite": 1, "boundary_admissible": 1},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    _write(result_path, result)
    _write(
        status_path,
        {
            "status": "complete",
            "task": "bounded_teacher",
            "model_seed": provenance.EXPECTED_PARENT_MODEL_SEED,
            "task_result_sha256": file_fingerprint(result_path),
            "fingerprints": fingerprints,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    provenance._verify_task_result(
        result,
        result_path=result_path,
        status_path=status_path,
        expected_task="bounded_teacher",
        expected_learning_rate=3e-5,
    )

    result["fingerprints"]["loss_scale"] *= 0.5
    _write(result_path, result)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["fingerprints"] = result["fingerprints"]
    status["task_result_sha256"] = file_fingerprint(result_path)
    _write(status_path, status)
    with pytest.raises(ArtifactCompatibilityError, match="fingerprints changed"):
        provenance._verify_task_result(
            result,
            result_path=result_path,
            status_path=status_path,
            expected_task="bounded_teacher",
            expected_learning_rate=3e-5,
        )


def test_calibration_requires_exact_training_only_shared_multiplier() -> None:
    calibration = {
        "objective_kind": "density_ratio_balanced_raw_logit_bce",
        "calibration_split": "train",
        "calibration_state_count": 256,
        "training_only": 1,
        "shared_by_teacher_and_null": 1,
        "target_initial_gradient_norm": 0.1,
        "scaled_initial_gradient_norm": 0.1,
        "unscaled_initial_gradient_norm": 1.0,
        "loss_scale": provenance.PARENT_LOSS_SCALE,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    provenance._verify_calibration(calibration)
    calibration["loss_scale"] *= 0.5
    with pytest.raises(ArtifactCompatibilityError, match="calibration changed"):
        provenance._verify_calibration(calibration)


def test_real_finished_222_record_parent_is_accepted_when_available() -> None:
    parent = Path(
        "runs/experiment12_d0_score_density_ratio_controls/"
        "20260719-233220_production-density-ratio-controls"
    )
    if not parent.is_dir():
        pytest.skip("production density-ratio parent is not part of this checkout")
    result = provenance.verify_parent_density_ratio_run(parent)
    assert result["passed"] == 1
    assert result["terminal_registry_record_count"] == 222
    assert result["parent_decision"] == "classification_optimizer_unresolved"
    assert result["parent_loss_scale"] == pytest.approx(0.05173607018770852)
    assert result["teacher_b_confirmed_learning_rates"] == pytest.approx([1e-5, 3e-5])
    assert result["confirmation_performed"] == 0
    assert result["transitive_parent_provenance"]["terminal_registry_record_count"] == 381
    assert result["physical_training_performed"] == result["sampling_performed"] == 0
