from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError
from mnist.d0_jacobi_rb_cuda_provenance import (
    EXPECTED_LINEAGE_REGISTRY_RECORD_COUNTS,
    PARENT_REGISTRY_RECORD_COUNT,
    PARENT_REGISTRY_SHA256,
    PARENT_RUN_BASENAME,
    PARENT_SOURCE_FINGERPRINT,
    READJUDICATED_DECISION,
    verify_and_readjudicate_jacobi_rb_cuda_parent,
    verify_parent_jacobi_rb_cuda_run,
)


PARENT = Path(
    "runs/experiment12_d0_jacobi_rb_denoising_feasibility"
) / PARENT_RUN_BASENAME


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable CUDA control is unavailable")
def test_exact_control_binds_numerical_validity_and_resource_failure() -> None:
    record = verify_and_readjudicate_jacobi_rb_cuda_parent(PARENT)
    assert record["evaluation_status"] == "evaluated"
    assert record["passed"] == 1
    assert record["parent_run_basename"] == PARENT_RUN_BASENAME
    assert record["parent_artifact_registry_sha256"] == PARENT_REGISTRY_SHA256
    assert record["parent_artifact_record_count"] == PARENT_REGISTRY_RECORD_COUNT
    assert record["parent_source_fingerprint"] == PARENT_SOURCE_FINGERPRINT
    assert record["saved_decision"] == (
        "spectral_inversion_computationally_infeasible"
    )
    assert record["readjudicated_decision"] == READJUDICATED_DECISION
    assert record["parent_numerically_valid"] == 1
    assert record["parent_resource_feasible"] == 0
    assert record["parent_target_evaluation_status"] == "not_evaluated"
    assert record["lineage_registry_record_counts"] == [
        315,
        *EXPECTED_LINEAGE_REGISTRY_RECORD_COUNTS,
    ]
    assert record["physical_training_performed"] == 0
    assert record["sampling_performed"] == 0
    assert record["reverse_sampling_performed"] == 0
    json.dumps(record, allow_nan=False)
    assert verify_parent_jacobi_rb_cuda_run(PARENT) == record


def test_wrong_control_fails_closed() -> None:
    with pytest.raises(ArtifactCompatibilityError):
        verify_and_readjudicate_jacobi_rb_cuda_parent(Path("tests"))
