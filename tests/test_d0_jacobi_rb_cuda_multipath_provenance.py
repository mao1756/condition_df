from __future__ import annotations

from pathlib import Path

import pytest

from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError
from mnist.d0_jacobi_rb_cuda_multipath_provenance import (
    PARENT_REGISTRY_RECORD_COUNT,
    PARENT_REGISTRY_SHA256,
    PARENT_RUN_BASENAME,
    PARENT_SOURCE_FINGERPRINT,
    READJUDICATED_DECISION,
    verify_and_readjudicate_jacobi_rb_cuda_multipath_parent,
)


PARENT = (
    Path("runs/experiment12_d0_jacobi_rb_cuda_confirmation")
    / PARENT_RUN_BASENAME
)


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable CUDA parent unavailable")
def test_exact_parent_is_numerically_valid_and_scheduling_resource_limited() -> None:
    record = verify_and_readjudicate_jacobi_rb_cuda_multipath_parent(PARENT)
    assert record["passed"] == 1
    assert record["parent_artifact_record_count"] == PARENT_REGISTRY_RECORD_COUNT
    assert record["parent_artifact_registry_sha256"] == PARENT_REGISTRY_SHA256
    assert record["parent_source_fingerprint"] == PARENT_SOURCE_FINGERPRINT
    assert record["parent_kernel_numerically_valid"] == 1
    assert record["parent_kernel_resource_valid"] == 0
    assert record["parent_target_evaluation_status"] == "not_evaluated"
    assert record["readjudicated_decision"] == READJUDICATED_DECISION
    assert record["readjudication_basis"]["failed_kernel_checks"] == [
        "projected_cache_hours",
        "slowest_transitions_per_second",
    ]
    assert record["physical_training_performed"] == 0
    assert record["reverse_sampling_performed"] == 0


def test_wrong_parent_fails_closed() -> None:
    with pytest.raises(ArtifactCompatibilityError):
        verify_and_readjudicate_jacobi_rb_cuda_multipath_parent(Path("tests"))

