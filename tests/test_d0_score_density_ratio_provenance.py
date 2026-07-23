from __future__ import annotations

import json
from pathlib import Path

import pytest

import mnist.d0_score_density_ratio_provenance as provenance
from mnist.d0_one_image_gate import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
)
from mnist.d0_score_control_stability_gate import StabilityThresholds


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
        "optimization": {
            "weight_decay": 1e-4,
            "ema_decay": 0.99,
            "grad_clip": 1.0,
            "clip_warmup_steps": 500,
            "implicit_loss_scale": provenance.PARENT_IMPLICIT_LOSS_SCALE,
            "adaptive_loss_scaling": 0,
        },
        "pilot": {
            "learning_rates": list(provenance.EXPECTED_PILOT_LEARNING_RATES),
            "steps": 1000,
            "selection_paths": 16,
            "audit_paths": 0,
        },
        "confirmation": {
            "model_seeds": list(provenance.EXPECTED_CONFIRMATION_SEEDS),
            "steps": 4000,
            "selection_paths": 32,
            "audit_paths": 32,
        },
        "stream": {
            "batch_size": 64,
            "clusters_per_step": 2,
            "anchor_bin_counts": [4, 4, 4, 4, 16],
            "training_probe_banks": 2,
            "training_probes_per_bank": 4,
        },
        "thresholds": json.loads(json.dumps(StabilityThresholds().to_dict())),
        "root_seed": 260801,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def test_scientific_manifest_rejects_threshold_changes() -> None:
    scientific = _scientific()
    manifest = {
        "scientific_config": scientific,
        "scientific_fingerprint": config_fingerprint(scientific),
    }
    assert provenance._verify_scientific_manifest(manifest) == scientific
    scientific["thresholds"]["maximum_clip_fraction"] = 0.11
    manifest["scientific_fingerprint"] = config_fingerprint(scientific)
    with pytest.raises(ArtifactCompatibilityError, match="thresholds changed"):
        provenance._verify_scientific_manifest(manifest)


def test_terminal_registry_requires_exact_381_records_and_detects_tampering(
    tmp_path: Path,
) -> None:
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
    result = provenance._verify_registry(
        root, registry_path=registry_path, status_path=status_path
    )
    assert len(result["records"]) == 381

    with (root / "artifact-010.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ArtifactCompatibilityError, match="registered stability artifact"):
        provenance._verify_registry(
            root, registry_path=registry_path, status_path=status_path
        )


def test_terminal_verifier_rejects_wrong_decision_before_using_evidence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "parent"
    root.mkdir()
    for relative in provenance._REQUIRED_FILES.values():
        _write(root / relative, {})
    manifest = {
        "schema": provenance.PARENT_RUN_SCHEMA,
        "schema_version": 1,
        "run_dir": str(root.resolve()),
    }
    status = {
        "schema": provenance.PARENT_RUN_SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "outcome": "gate_failed",
        "decision": "control_pipeline_repaired",
        "probe_bank_status": "agree",
        "required_gate": "controls",
        "required_gate_pass": 0,
    }
    _write(root / "run_manifest.json", manifest)
    _write(root / "run_status.json", status)
    with pytest.raises(ArtifactCompatibilityError, match="outcome"):
        provenance.verify_parent_stability_run(root)


def test_transitive_parent_must_match_recursive_verification(monkeypatch) -> None:
    expected = {
        "schema": "experiment12-d0-score-control-scale-repair-stability-parent-provenance",
        "schema_version": 1,
        "passed": 1,
        "run_dir": "upstream",
        "scientific_fingerprint": "expected",
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    monkeypatch.setattr(
        provenance,
        "verify_parent_scale_repair_run",
        lambda path: {key: value for key, value in expected.items() if key != "run_dir"},
    )
    with pytest.raises(ArtifactCompatibilityError, match="scientific_fingerprint"):
        provenance._verify_transitive_parent(
            {**expected, "scientific_fingerprint": "tampered"}
        )


def test_task_result_status_binding_and_optimizer_finiteness(tmp_path: Path) -> None:
    result_path = tmp_path / "task_result.json"
    status_path = tmp_path / "task_status.json"
    result = {
        "task_kind": "null",
        "model_seed": 7,
        "metrics": {"complete": 1, "finite": 1, "boundary_admissible": 1},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    _write(result_path, result)
    _write(
        status_path,
        {
            "status": "complete",
            "task_kind": "null",
            "model_seed": 7,
            "task_result_sha256": file_fingerprint(result_path),
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    provenance._verify_task_result(
        result,
        result_path=result_path,
        status_path=status_path,
        expected_kind="null",
        expected_seed=7,
    )
    result["metrics"]["finite"] = 0
    _write(result_path, result)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["task_result_sha256"] = file_fingerprint(result_path)
    _write(status_path, status)
    with pytest.raises(ArtifactCompatibilityError, match="nonfinite"):
        provenance._verify_task_result(
            result,
            result_path=result_path,
            status_path=status_path,
            expected_kind="null",
            expected_seed=7,
        )


def test_real_finished_parent_is_accepted_when_available() -> None:
    parent = Path(
        "runs/experiment12_d0_score_control_stability_confirmation/"
        "20260718-232902_production-streamed-implicit-controls"
    )
    if not parent.is_dir():
        pytest.skip("production stability parent is not part of this checkout")
    result = provenance.verify_parent_stability_run(parent)
    assert result["passed"] == 1
    assert result["terminal_registry_record_count"] == 381
    assert result["parent_decision"] == "implicit_objective_unstable"
    assert result["selected_learning_rate"] == pytest.approx(1e-5)
    assert result["teacher_passing_seed_count"] == 0
    assert result["null_passing_seed_count"] == 2
    assert result["physical_training_performed"] == result["sampling_performed"] == 0
