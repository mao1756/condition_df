from __future__ import annotations

import json
from pathlib import Path

import pytest

import mnist.d0_score_density_ratio_h1_gradient_control_provenance as provenance
from mnist.d0_one_image_gate import ArtifactCompatibilityError, file_fingerprint


REAL_PARENT = Path(
    "runs/experiment12_d0_score_density_ratio_h1_trust_confirmation/"
    "20260721-114934_production-h1-function-step-density-ratio-controls"
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


def _candidate_gate(multiplier: float) -> dict:
    common = {
        "accumulation_steps": {"passed": 1},
        "base_channels": {"passed": 1},
        "known_multiplier": {"passed": 1},
        "learning_rate": {"passed": 1},
        "optimizer_and_task_health": {"passed": 1},
        "stationary_null": {"passed": 1},
        "teacher_classification": {"passed": 1},
    }
    if multiplier == 0.0:
        return {
            "multiplier": multiplier,
            "passed": 1,
            "optimizer_health_pass": 1,
            "classification_pass": 1,
            "null_pass": 1,
            "subchecks": common,
        }
    return {
        "multiplier": multiplier,
        "passed": 0,
        "optimizer_health_pass": 1,
        "classification_pass": 1,
        "null_pass": 1,
        "derivative_pass": 0,
        "relative_l2_reduction_pass": 1,
        "subchecks": {
            **common,
            "relative_l2_reduction_overall_and_data_end": {"passed": 1},
            "strict_derivative_thresholds": {"passed": 0},
        },
    }


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


def test_registry_rejects_wrong_frozen_parent(tmp_path):
    registry_path, status_path, _ = _registry_fixture(tmp_path)
    with pytest.raises(ArtifactCompatibilityError, match="frozen 301-record parent"):
        provenance._verify_registry(
            tmp_path,
            registry_path=registry_path,
            status_path=status_path,
            expected_count=1,
            expected_sha256="0" * 64,
        )


def test_transitive_parent_requires_exact_263_123_125_332_222_381(
    monkeypatch, tmp_path
):
    recomputed = {
        "schema": "normalized-multiplicity-provenance",
        "schema_version": 1,
        "passed": 1,
        "run_dir": str(tmp_path.resolve()),
        "terminal_registry_record_count": 263,
        "lineage_registry_record_counts": [123, 125, 332, 222, 381],
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    monkeypatch.setattr(
        provenance,
        "verify_parent_multiplicity_run",
        lambda path: dict(recomputed),
    )
    stored = dict(recomputed)
    stored["schema"] = provenance.PARENT_RUN_SCHEMA + "-parent-provenance"
    stored["verifier_source_fingerprint"] = "advisory"
    assert provenance._verify_transitive_parent(stored) == recomputed

    wrong = dict(recomputed)
    wrong["lineage_registry_record_counts"] = [123, 125, 332, 222, 380]
    monkeypatch.setattr(
        provenance,
        "verify_parent_multiplicity_run",
        lambda path: dict(wrong),
    )
    stored_wrong = dict(wrong)
    stored_wrong["schema"] = provenance.PARENT_RUN_SCHEMA + "-parent-provenance"
    with pytest.raises(ArtifactCompatibilityError, match="exact 263/123/125/332/222/381"):
        provenance._verify_transitive_parent(stored_wrong)


def test_derivative_only_failure_rejects_any_other_failed_check():
    pilot = {
        "candidate_gates": [_candidate_gate(value) for value in provenance.EXPECTED_MULTIPLIERS]
    }
    provenance._verify_derivative_only_failure(pilot)

    pilot["candidate_gates"][1]["classification_pass"] = 0
    pilot["candidate_gates"][1]["subchecks"]["teacher_classification"] = {"passed": 0}
    with pytest.raises(ArtifactCompatibilityError, match="not restricted"):
        provenance._verify_derivative_only_failure(pilot)


@pytest.mark.skipif(not REAL_PARENT.is_dir(), reason="production H1 parent is not present")
def test_real_301_record_h1_parent_is_accepted():
    result = provenance.verify_parent_h1_trust_run(REAL_PARENT)
    assert result["passed"] == 1
    assert result["terminal_registry_record_count"] == 301
    assert result["artifact_registry_sha256"] == provenance.PARENT_REGISTRY_SHA256
    assert result["lineage_registry_record_counts"] == [263, 123, 125, 332, 222, 381]
    assert result["parent_decision"] == "h1_function_step_unresolved"
    assert result["pilot_task_count"] == 8
    assert result["all_tasks_complete_finite_boundary_admissible"] == 1
    assert result["all_task_clipping_zero"] == 1
    assert result["all_teacher_classification_pass"] == 1
    assert result["all_null_pass"] == 1
    assert result["nonzero_derivative_passing_count"] == 0
    assert result["derivative_only_failure"] == 1
    assert result["selected_profile_present"] == 0
    assert result["confirmation_performed"] == 0
    assert result["physical_training_performed"] == 0
    assert result["sampling_performed"] == 0
