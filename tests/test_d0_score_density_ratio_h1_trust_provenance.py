from __future__ import annotations

import json
from pathlib import Path

import pytest

import mnist.d0_score_density_ratio_h1_trust_provenance as provenance
from mnist.d0_one_image_gate import ArtifactCompatibilityError, file_fingerprint


REAL_PARENT = Path(
    "runs/experiment12_d0_score_density_ratio_multiplicity_confirmation/"
    "20260721-000607_production-multiplicity-aware-density-ratio-controls"
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _registry_fixture(tmp_path: Path):
    payload = tmp_path / "evidence.json"
    _write_json(payload, {"evidence": 1})
    registry_path = tmp_path / "artifact_registry.json"
    status_path = tmp_path / "run_status.json"
    registry = {
        "schema": provenance.PARENT_REGISTRY_SCHEMA,
        "schema_version": 1,
        "terminal_files_excluded_to_avoid_self_reference": [
            "artifact_registry.json",
            "run_status.json",
        ],
        "records": {
            "evidence.json": {
                "path": str(payload.resolve()),
                "sha256": file_fingerprint(payload),
                "size": payload.stat().st_size,
            }
        },
    }
    _write_json(registry_path, registry)
    _write_json(
        status_path,
        {
            "artifact_registry_sha256": file_fingerprint(registry_path),
            "artifact_registry_size": registry_path.stat().st_size,
        },
    )
    return registry_path, status_path, payload


def test_registry_verifies_exact_payload_set_and_hashes(tmp_path):
    registry_path, status_path, payload = _registry_fixture(tmp_path)
    result = provenance._verify_registry(
        tmp_path,
        registry_path=registry_path,
        status_path=status_path,
        expected_count=1,
        expected_sha256=None,
    )
    assert len(result["records"]) == 1

    payload.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match="hash or size"):
        provenance._verify_registry(
            tmp_path,
            registry_path=registry_path,
            status_path=status_path,
            expected_count=1,
            expected_sha256=None,
        )


def test_registry_rejects_wrong_frozen_identity(tmp_path):
    registry_path, status_path, _ = _registry_fixture(tmp_path)
    with pytest.raises(ArtifactCompatibilityError, match="frozen 263-record parent"):
        provenance._verify_registry(
            tmp_path,
            registry_path=registry_path,
            status_path=status_path,
            expected_count=1,
            expected_sha256="0" * 64,
        )


def test_registry_rejects_unregistered_or_stale_files(tmp_path):
    registry_path, status_path, _ = _registry_fixture(tmp_path)
    (tmp_path / "unregistered.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match="incomplete or contains stale"):
        provenance._verify_registry(
            tmp_path,
            registry_path=registry_path,
            status_path=status_path,
            expected_count=1,
            expected_sha256=None,
        )


def test_transitive_parent_requires_exact_123_125_332_222_381(monkeypatch, tmp_path):
    recomputed = {
        "schema": "selection-power-normalized-provenance",
        "schema_version": 1,
        "passed": 1,
        "run_dir": str(tmp_path.resolve()),
        "terminal_registry_record_count": 123,
        "lineage_registry_record_counts": [125, 332, 222, 381],
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    monkeypatch.setattr(
        provenance,
        "verify_parent_selection_power_run",
        lambda path: dict(recomputed),
    )
    stored = dict(recomputed)
    stored["schema"] = provenance.PARENT_RUN_SCHEMA + "-parent-provenance"
    stored["verifier_source_fingerprint"] = "advisory"
    assert provenance._verify_transitive_parent(stored) == recomputed

    wrong = dict(recomputed)
    wrong["lineage_registry_record_counts"] = [125, 332, 222, 380]
    monkeypatch.setattr(
        provenance,
        "verify_parent_selection_power_run",
        lambda path: dict(wrong),
    )
    stored_wrong = dict(wrong)
    stored_wrong["schema"] = provenance.PARENT_RUN_SCHEMA + "-parent-provenance"
    with pytest.raises(ArtifactCompatibilityError, match="exact 123/125/332/222/381"):
        provenance._verify_transitive_parent(stored_wrong)


@pytest.mark.skipif(not REAL_PARENT.is_dir(), reason="production parent is not present")
def test_real_263_record_parent_is_accepted():
    result = provenance.verify_parent_multiplicity_run(REAL_PARENT)
    assert result["passed"] == 1
    assert result["terminal_registry_record_count"] == 263
    assert result["artifact_registry_sha256"] == provenance.PARENT_REGISTRY_SHA256
    assert result["lineage_registry_record_counts"] == [123, 125, 332, 222, 381]
    assert result["parent_decision"] == "density_ratio_value_only"
    assert result["h1_function_step_patch_authorized"] == 1
    assert result["confirmation_task_count"] == 6
    assert result["all_task_clipping_zero"] == 1
    assert result["teacher_classification_passing_seed_count"] == 3
    assert result["teacher_derivative_passing_seed_count"] == 0
    assert result["null_family_pass"] == 1
    assert result["physical_training_performed"] == 0
    assert result["sampling_performed"] == 0

