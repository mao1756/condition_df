from __future__ import annotations

from pathlib import Path

import pytest

from mnist.d0_jacobi_provenance import (
    PARENT_REGISTRY_SHA256,
    verify_and_readjudicate_gradient_parent,
)


PARENT = Path(
    "runs/experiment12_d0_score_density_ratio_h1_gradient_control_confirmation/"
    "20260722-000701_production-gradient-controlled-h1-density-ratio-controls"
)


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable production evidence is unavailable")
def test_frozen_parent_is_rebound_and_readjudicated_without_mutation() -> None:
    record = verify_and_readjudicate_gradient_parent(PARENT)
    assert record["parent_artifact_registry_sha256"] == PARENT_REGISTRY_SHA256
    assert record["saved_decision"] == "selection_false_discovery"
    assert record["readjudicated_decision"] == "h1_strength_grid_unresolved"
    assert record["readjudication_valid"] == 1
    assert record["lineage_registry_record_counts"] == [277, 301, 263, 123, 125, 332, 222, 381]
    assert record["physical_training_performed"] == 0
    assert record["sampling_performed"] == 0
