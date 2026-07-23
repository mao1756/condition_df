from __future__ import annotations

import json
from pathlib import Path

import pytest

import mnist.d0_score_density_ratio_head_provenance as provenance
from mnist.d0_one_image_gate import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
)
from mnist.d0_score_density_ratio_stability_gate import RatioStabilityThresholds


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
        "paired_estimator_schema": "experiment12-d0-score-density-ratio-paired-mixture",
        "paired_objective_version": "d0-density-ratio-paired-mixture-weighted-softplus-v1",
        "paired_stream_version": "d0-density-ratio-paired-mixture-stream-v1",
        "paired_accumulation_version": "d0-density-ratio-deterministic-gradient-accumulation-v1",
        "loss_scale": provenance.PARENT_LOSS_SCALE,
        "optimization": {
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "ema_decay": 0.99,
            "grad_clip": 1.0,
            "clip_warmup_steps": 500,
            "adaptive_loss_scaling": 0,
            "gradient_accumulation": "mean-then-clip-once",
        },
        "pilot": {
            "accumulation_levels": [2, 4, 8],
            "learning_rates": [3e-5, 1e-5],
            "steps": 2000,
            "selection_paths_per_panel": 16,
        },
        "confirmation": {
            "model_seeds": [260861, 260862, 260863],
            "steps": 4000,
            "selection_paths_per_panel": 32,
            "audit_paths_per_panel": 32,
        },
        "microbatch": {
            "clusters": 32,
            "time_bin_counts": [4, 4, 4, 4, 16],
            "teacher_coupling": "common-gamma-stochastic-anchor",
            "null_coupling": "independent-dirichlet-pooled-label-swap",
        },
        "preflight": {"paths": 128, "confidence": 0.99},
        "root_seed": 260851,
        "thresholds": json.loads(json.dumps(RatioStabilityThresholds().to_dict())),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def test_scientific_manifest_rejects_width_ancestor_or_threshold_changes() -> None:
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
    changed["model_schema"] = "smaller-model"
    with pytest.raises(ArtifactCompatibilityError, match="schema mismatch"):
        provenance._verify_scientific_manifest(
            {
                "scientific_config": changed,
                "scientific_fingerprint": config_fingerprint(changed),
            }
        )


def test_registry_requires_exact_332_records_and_detects_tampering(tmp_path: Path) -> None:
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
    assert len(result["records"]) == 332
    with (root / "artifact-100.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ArtifactCompatibilityError, match="registered paired-ratio"):
        provenance._verify_registry(
            root, registry_path=registry_path, status_path=status_path
        )


def test_transitive_parent_requires_exact_222_and_381_bindings(monkeypatch) -> None:
    recomputed = {
        "schema": "recomputed-schema",
        "schema_version": 1,
        "passed": 1,
        "run_dir": "upstream",
        "terminal_registry_record_count": 222,
        "transitive_parent_provenance": {"terminal_registry_record_count": 381},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    monkeypatch.setattr(
        provenance, "verify_parent_density_ratio_run", lambda path: recomputed
    )
    stored = {
        **recomputed,
        "schema": "experiment12-d0-score-density-ratio-stability-confirmation-parent-provenance",
        "verifier_source_fingerprint": "source",
    }
    assert provenance._verify_transitive_parent(stored) == recomputed
    stored["terminal_registry_record_count"] = 221
    with pytest.raises(ArtifactCompatibilityError, match="222/381-record"):
        provenance._verify_transitive_parent(stored)


def test_task_result_binds_status_fingerprints_and_pairing_fields(tmp_path: Path) -> None:
    result_path = tmp_path / "task_result.json"
    status_path = tmp_path / "task_status.json"
    fingerprints = {
        "accumulation_level": 8,
        "learning_rate": 3e-5,
        "loss_scale": provenance.PARENT_LOSS_SCALE,
        "phase": "pilot",
        "task": "bounded_teacher",
    }
    result = {
        "task": "bounded_teacher",
        "model_seed": 42,
        "fingerprints": fingerprints,
        "gate": {"passed": 1},
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
            "model_seed": 42,
            "training_step": 2000,
            "task_result_sha256": file_fingerprint(result_path),
            "fingerprints": fingerprints,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    assert provenance._verify_task_result(
        result,
        result_path=result_path,
        status_path=status_path,
        expected_task="bounded_teacher",
        expected_accumulation=8,
        expected_learning_rate=3e-5,
    ) == 42
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
            expected_task="bounded_teacher",
            expected_accumulation=8,
            expected_learning_rate=3e-5,
        )


def test_terminal_verifier_rejects_wrong_outcome_before_registry(tmp_path: Path) -> None:
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
    with pytest.raises(ArtifactCompatibilityError, match="coordinate-repair precursor"):
        provenance.verify_parent_paired_ratio_run(root)


def test_real_finished_332_record_parent_is_accepted_when_available() -> None:
    parent = Path(
        "runs/experiment12_d0_score_density_ratio_stability_confirmation/"
        "20260720-011809_production-paired-density-ratio-controls"
    )
    if not parent.is_dir():
        pytest.skip("production paired-ratio parent is not part of this checkout")
    result = provenance.verify_parent_paired_ratio_run(parent)
    assert result["passed"] == 1
    assert result["terminal_registry_record_count"] == 332
    assert result["artifact_registry_sha256"] == file_fingerprint(
        parent / "artifact_registry.json"
    )
    assert result["artifact_registry_size"] == (
        parent / "artifact_registry.json"
    ).stat().st_size
    assert result["pilot_candidate_count"] == 6
    assert result["pilot_task_count"] == 12
    assert result["transitive_parent_provenance"]["terminal_registry_record_count"] == 222
    assert result["transitive_parent_provenance"]["transitive_parent_provenance"][
        "terminal_registry_record_count"
    ] == 381
    assert result["physical_training_performed"] == result["sampling_performed"] == 0
