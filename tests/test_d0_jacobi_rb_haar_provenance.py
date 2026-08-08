from __future__ import annotations

from pathlib import Path

import pytest

from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError
from mnist.d0_jacobi_rb_haar_provenance import (
    PARENT_DECISION,
    PARENT_RE_ADJUDICATION,
    PARENT_REGISTRY_RECORD_COUNT,
    PARENT_REGISTRY_SHA256,
    PARENT_RUN_BASENAME,
    PARENT_SCIENTIFIC_CONFIG_SHA256,
    PARENT_SOURCE_COUNT,
    PARENT_SOURCE_FINGERPRINT,
    verify_right_endpoint_coupling_parent,
)


PARENT = (
    Path(
        "runs/experiment12_d0_jacobi_rb_dynkin_phase_observer_confirmation"
    )
    / PARENT_RUN_BASENAME
)


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable Haar parent unavailable")
def test_exact_parent_is_verified_and_readjudicated() -> None:
    record = verify_right_endpoint_coupling_parent(PARENT)
    assert record["passed"] == 1
    assert record["parent_artifact_record_count"] == PARENT_REGISTRY_RECORD_COUNT
    assert record["parent_artifact_registry_sha256"] == PARENT_REGISTRY_SHA256
    assert record["parent_source_count"] == PARENT_SOURCE_COUNT
    assert record["parent_source_fingerprint"] == PARENT_SOURCE_FINGERPRINT
    assert (
        record["parent_scientific_config_sha256"]
        == PARENT_SCIENTIFIC_CONFIG_SHA256
    )
    assert record["parent_decision"] == PARENT_DECISION
    assert record["parent_re_adjudication"] == PARENT_RE_ADJUDICATION
    assert record["parent_preflight_pass"] == 1
    assert record["parent_pilot_numerically_valid"] == 1
    assert record["parent_pilot_resource_valid"] == 1
    assert record["parent_pilot_power_valid"] == 0
    assert record["parent_panel_a_nominated"] == 0
    assert record["parent_panel_b_opened"] == 0
    assert record["parent_selected_design"] is None
    assert record["parent_production_refinement_performed"] == 0
    assert record["parent_mutated"] == 0


def test_wrong_parent_fails_closed() -> None:
    with pytest.raises(ArtifactCompatibilityError):
        verify_right_endpoint_coupling_parent(Path("tests"))

