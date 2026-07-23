from __future__ import annotations

from pathlib import Path

import pytest

from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError
from mnist.d0_jacobi_rb_provenance import (
    EXPECTED_LINEAGE_REGISTRY_RECORD_COUNTS,
    PARENT_REGISTRY_RECORD_COUNT,
    PARENT_REGISTRY_SHA256,
    verify_and_readjudicate_jacobi_parent,
    verify_parent_jacobi_feasibility_run,
)


PARENT = Path(
    "runs/experiment12_d0_jacobi_denoising_feasibility/"
    "20260722-142613_production-exact-jacobi-feasibility"
)


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable production evidence is unavailable")
def test_exact_failed_parent_is_bound_without_changing_the_ddpm_target() -> None:
    record = verify_and_readjudicate_jacobi_parent(PARENT)
    assert record["evaluation_status"] == "evaluated"
    assert record["passed"] == 1
    assert record["parent_artifact_registry_sha256"] == PARENT_REGISTRY_SHA256
    assert record["parent_artifact_record_count"] == PARENT_REGISTRY_RECORD_COUNT
    assert record["saved_decision"] == "jacobi_kernel_numerically_unresolved"
    assert record["readjudicated_decision"] == "ancestral_representation_infeasible"
    assert record["readjudication_basis"]["projected_cache_hours"] == pytest.approx(
        5080.94209321022
    )
    assert record["readjudication_basis"]["uncertified_draw_count"] == 14
    assert record["ddpm_population_target_preserved"] == 1
    assert record["lineage_registry_record_counts"] == [
        16,
        *EXPECTED_LINEAGE_REGISTRY_RECORD_COUNTS,
    ]
    assert record["physical_training_performed"] == 0
    assert record["sampling_performed"] == 0
    assert verify_parent_jacobi_feasibility_run(PARENT) == record


def test_non_parent_directory_fails_closed() -> None:
    with pytest.raises(ArtifactCompatibilityError):
        verify_and_readjudicate_jacobi_parent(Path("tests"))
