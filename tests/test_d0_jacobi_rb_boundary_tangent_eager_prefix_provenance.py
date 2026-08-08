from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnist.d0_jacobi_artifacts import config_fingerprint, file_fingerprint
import mnist.d0_jacobi_rb_boundary_tangent_eager_prefix_provenance as provenance


PARENT = Path(
    "runs/experiment12_d0_jacobi_rb_boundary_tangent_schedule_feasibility"
) / provenance.PARENT_RUN_BASENAME


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable fused-schedule parent unavailable")
def test_exact_live_parent_is_accepted() -> None:
    record = provenance.verify_eager_prefix_schedule_parent(
        parent_schedule_run_dir=PARENT
    )
    assert record["passed"] == 1
    assert record["parent_registry"] == {
        "artifact_count": 614,
        "semantic_sha256": provenance.PARENT_REGISTRY_SEMANTIC_SHA256,
        "file_sha256": provenance.PARENT_REGISTRY_FILE_SHA256,
    }
    assert record["parent_decision"] == (
        "boundary_tangent_schedule_computationally_infeasible"
    )
    assert record["parent_failure_domain"] == "resource_gate"
    assert record["parent_stage_execution_valid"] == 1
    assert record["parent_numerically_valid"] == 1
    assert record["parent_resource_valid"] == 0
    assert record["parent_resource_only_failure"] == 1
    assert record["parent_scientific_evidence_complete"] == 1
    assert record["parent_projected_transition_count"] == 337_182_720
    assert record["parent_projected_exact_cache_hours"] > 30.0
    assert record["eager_prefix_schedule_feasibility_authorized"] == 1
    assert record["physical_training_performed"] == 0
    assert record["controller_control_trajectory_performed"] == 0
    assert record["sampling_performed"] == 0
    body = dict(record)
    semantic = body.pop("semantic_sha256")
    assert semantic == config_fingerprint(body)


def test_wrong_parent_basename_fails_closed(tmp_path: Path) -> None:
    wrong = tmp_path / "wrong-parent"
    wrong.mkdir()
    with pytest.raises(
        provenance.EagerPrefixScheduleProvenanceError,
        match="wrong fused-schedule parent basename",
    ):
        provenance.verify_eager_prefix_schedule_parent(
            parent_schedule_run_dir=wrong
        )


def _minimal_registry(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    payload = root / "payload.json"
    _write_json(payload, {"value": 1})
    for name in provenance._REGISTRY_SEMANTICS["excluded_paths"]:
        if name != "artifact_registry.json":
            _write_json(root / name, {"name": name})
    artifacts = [
        {
            "path": "payload.json",
            "sha256": file_fingerprint(payload),
            "size": payload.stat().st_size,
        }
    ]
    semantics = dict(provenance._REGISTRY_SEMANTICS)
    registry = {
        "schema": provenance.PARENT_RUN_SCHEMA + "-artifact-registry",
        "schema_version": 1,
        "artifact_count": 1,
        "artifacts": artifacts,
        "registry_semantics": semantics,
        **{field: 0 for field in provenance.NO_WORK_FIELDS},
    }
    registry["semantic_sha256"] = config_fingerprint(
        {"artifacts": artifacts, "registry_semantics": semantics}
    )
    registry_path = root / "artifact_registry.json"
    _write_json(registry_path, registry)
    monkeypatch.setattr(provenance, "PARENT_REGISTRY_COUNT", 1)
    monkeypatch.setattr(
        provenance,
        "PARENT_REGISTRY_SEMANTIC_SHA256",
        registry["semantic_sha256"],
    )
    monkeypatch.setattr(
        provenance,
        "PARENT_REGISTRY_FILE_SHA256",
        file_fingerprint(registry_path),
    )
    monkeypatch.setattr(
        provenance,
        "_EXCLUDED_FILE_SHA256",
        {
            name: file_fingerprint(root / name)
            for name in provenance._REGISTRY_SEMANTICS["excluded_paths"]
        },
    )
    return payload


def test_minimal_registry_hash_and_file_set_tampering_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _minimal_registry(tmp_path, monkeypatch)
    assert provenance._verify_registry(tmp_path)["artifact_count"] == 1

    _write_json(payload, {"value": 2})
    with pytest.raises(
        provenance.EagerPrefixScheduleProvenanceError,
        match="artifact changed",
    ):
        provenance._verify_registry(tmp_path)

    _write_json(payload, {"value": 1})
    _write_json(tmp_path / "unregistered.json", {"unexpected": True})
    with pytest.raises(
        provenance.EagerPrefixScheduleProvenanceError,
        match="terminal file set changed",
    ):
        provenance._verify_registry(tmp_path)


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable fused-schedule parent unavailable")
def test_config_binding_change_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        provenance, "PARENT_SCIENTIFIC_CONFIG_SHA256", "0" * 64
    )
    with pytest.raises(
        provenance.EagerPrefixScheduleProvenanceError,
        match="manifest binding changed|scientific configuration changed",
    ):
        provenance.verify_eager_prefix_schedule_parent(
            parent_schedule_run_dir=PARENT
        )


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable fused-schedule parent unavailable")
def test_live_source_fingerprint_change_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance, "source_fingerprint", lambda paths: "0" * 64)
    with pytest.raises(
        provenance.EagerPrefixScheduleProvenanceError,
        match="live source fingerprint changed",
    ):
        provenance.verify_eager_prefix_schedule_parent(
            parent_schedule_run_dir=PARENT
        )


def test_nonzero_work_or_authorization_claim_fails_closed() -> None:
    with pytest.raises(
        provenance.EagerPrefixScheduleProvenanceError,
        match="records physical_training_performed",
    ):
        provenance._assert_zero_claims(
            {
                **{field: 0 for field in provenance.NO_WORK_FIELDS},
                "physical_training_performed": 1,
            },
            "fixture",
        )
    with pytest.raises(
        provenance.EagerPrefixScheduleProvenanceError,
        match="authorizes sampling_authorized",
    ):
        provenance._assert_zero_claims(
            {
                **{field: 0 for field in provenance.NO_WORK_FIELDS},
                "sampling_authorized": 1,
            },
            "fixture",
        )
