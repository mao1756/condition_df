from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError
from mnist.d0_jacobi_rb_dynkin_phase_observer_provenance import (
    PARENT_DECISION,
    PARENT_FAILURE_CODE,
    PARENT_FAILURE_MESSAGE,
    PARENT_RE_ADJUDICATION,
    PARENT_REGISTRY_RECORD_COUNT,
    PARENT_REGISTRY_SHA256,
    PARENT_RUN_BASENAME,
    PARENT_SCIENTIFIC_CONFIG_SHA256,
    PARENT_SOURCE_COUNT,
    PARENT_SOURCE_FINGERPRINT,
    verify_tower_observer_roundoff_parent,
)


PARENT = (
    Path("runs/experiment12_d0_jacobi_rb_dynkin_power_confirmation")
    / PARENT_RUN_BASENAME
)


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable Dynkin parent unavailable")
def test_exact_failed_parent_is_verified_and_readjudicated() -> None:
    record = verify_tower_observer_roundoff_parent(PARENT)
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
    assert record["parent_failure_code"] == PARENT_FAILURE_CODE
    assert record["parent_failure_message"] == PARENT_FAILURE_MESSAGE
    assert record["parent_path_id_plan_pass"] == 1
    assert record["parent_legacy_k512_replay_pass"] == 1
    assert record["parent_phase_moment_oracle_pass"] == 1
    assert record["parent_tower_inference_performed"] == 0
    assert record["parent_pilot_performed"] == 0
    assert record["parent_re_adjudication"] == PARENT_RE_ADJUDICATION
    assert record["parent_mutated"] == 0
    assert record["physical_training_authorized"] == 0
    assert record["sampling_authorized"] == 0


def test_wrong_parent_fails_closed() -> None:
    with pytest.raises(ArtifactCompatibilityError):
        verify_tower_observer_roundoff_parent(Path("tests"))


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable Dynkin parent unavailable")
def test_registry_or_failure_tampering_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / PARENT_RUN_BASENAME
    shutil.copytree(PARENT, copied)
    failure_path = copied / "preflight_failure.json"
    original = json.loads(failure_path.read_text(encoding="utf-8"))
    original["error"] = "different failure"
    failure_path.write_text(
        json.dumps(original, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ArtifactCompatibilityError,
        match="registry SHA-256 changed|registered parent artifact changed",
    ):
        verify_tower_observer_roundoff_parent(copied)

