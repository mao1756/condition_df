from __future__ import annotations

from pathlib import Path

import pytest

from mnist.d0_jacobi_artifacts import config_fingerprint
import mnist.d0_jacobi_rb_boundary_tangent_eager_pipeline_provenance as provenance


PARENT = Path(
    "runs/experiment12_d0_jacobi_rb_boundary_tangent_prefix_schedule_confirmation"
) / provenance.PARENT_RUN_BASENAME


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable prefix parent unavailable")
def test_exact_live_parent_and_base_only_readjudication_are_accepted() -> None:
    record = provenance.verify_eager_pipeline_parent(parent_prefix_run_dir=PARENT)
    assert record["passed"] == 1
    assert record["parent_registry"] == {
        "artifact_count": 33,
        "semantic_sha256": provenance.PARENT_REGISTRY_SEMANTIC_SHA256,
        "file_sha256": provenance.PARENT_REGISTRY_FILE_SHA256,
    }
    assert record["parent_decision"] == (
        "eager_prefix_profile_computationally_infeasible"
    )
    assert record["parent_failure_domain"] == "resource_gate"
    assert record["only_runtime_checks_failed"] == 1
    assert record["pilot_namespaces_unopened"] == 1
    assert record["complete_pipeline_timing_authorized"] == 1
    assert record["parent_projected_transition_count"] == 337_182_720
    assert record["parent_projected_seconds"] == pytest.approx(
        109_060.84962575609
    )
    assert record["parent_projected_effective_rate"] == pytest.approx(
        3_091.693500986353
    )
    assert record["production_cache_generation_performed"] == 0
    assert record["physical_training_performed"] == 0
    assert record["sampling_performed"] == 0
    body = dict(record)
    assert body.pop("semantic_sha256") == config_fingerprint(body)

    readjudication = provenance.build_eager_pipeline_parent_readjudication(record)
    assert readjudication["passed"] == 1
    assert readjudication["historical_decision"] == record["parent_decision"]
    assert readjudication["readjudicated_decision"] == (
        "base_only_projection_inconclusive"
    )
    assert readjudication["historical_gate_mutated"] == 0
    assert readjudication["parent_artifacts_mutated"] == 0
    assert readjudication["complete_pipeline_timing_authorized"] == 1
    assert readjudication["production_cache_generation_performed"] == 0
    body = dict(readjudication)
    assert body.pop("semantic_sha256") == config_fingerprint(body)


def test_wrong_parent_basename_and_unverified_readjudication_fail_closed(
    tmp_path: Path,
) -> None:
    wrong = tmp_path / "wrong-parent"
    wrong.mkdir()
    with pytest.raises(
        provenance.EagerPipelineProvenanceError,
        match="wrong prefix parent basename",
    ):
        provenance.verify_eager_pipeline_parent(parent_prefix_run_dir=wrong)

    with pytest.raises(
        provenance.EagerPipelineProvenanceError,
        match="parent provenance did not pass",
    ):
        provenance.build_eager_pipeline_parent_readjudication({"passed": 0})


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable prefix parent unavailable")
def test_parent_registry_and_scientific_binding_changes_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance, "PARENT_REGISTRY_COUNT", 34)
    with pytest.raises(
        provenance.EagerPipelineProvenanceError,
        match="terminal registry changed",
    ):
        provenance.verify_eager_pipeline_parent(parent_prefix_run_dir=PARENT)

    monkeypatch.setattr(provenance, "PARENT_REGISTRY_COUNT", 33)
    monkeypatch.setattr(provenance, "PARENT_SCIENTIFIC_CONFIG_SHA256", "0" * 64)
    with pytest.raises(provenance.EagerPipelineProvenanceError):
        provenance.verify_eager_pipeline_parent(parent_prefix_run_dir=PARENT)


def test_nonzero_work_or_integration_authorization_fails_closed() -> None:
    zero = {field: 0 for field in provenance.NO_WORK_FIELDS}
    with pytest.raises(
        provenance.EagerPipelineProvenanceError,
        match="records physical_training_performed",
    ):
        provenance._assert_zero_claims(
            {**zero, "physical_training_performed": 1}, "fixture"
        )

    with pytest.raises(
        provenance.EagerPipelineProvenanceError,
        match="authorizes schedule_integration_authorized",
    ):
        provenance._assert_zero_claims(
            {**zero, "schedule_integration_authorized": 1}, "fixture"
        )
