from __future__ import annotations

from pathlib import Path

import pytest

from mnist.d0_jacobi_artifacts import config_fingerprint
from mnist.d0_jacobi_rb_boundary_tangent_v3_memory_provenance import (
    BoundaryTangentV3MemoryProvenanceError,
    FAILED_FORWARD_ACTIVATION_BYTES,
    FAILED_FORWARD_ACTIVATION_GIB,
    PARENT_READJUDICATED_DECISION,
    PARENT_REGISTRY_COUNT,
    PARENT_REGISTRY_FILE_SHA256,
    PARENT_REGISTRY_SEMANTIC_SHA256,
    PARENT_RUN_BASENAME,
    TRAIN_ROW_COUNT,
    build_immutable_cache_binding,
    verify_failed_v3_train_parent,
    verify_immutable_cache_binding,
)


def _parent() -> Path:
    return (
        Path("runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_learnability")
        / PARENT_RUN_BASENAME
    ).resolve()


def test_exact_oom_parent_is_read_only_and_readjudicated() -> None:
    root = _parent()
    critical = {
        name: (root / name).read_bytes()
        for name in (
            "artifact_registry.json",
            "cache_artifact_seal.json",
            "train_artifact_seal.json",
            "run_status.json",
        )
    }
    record = verify_failed_v3_train_parent(root)
    assert critical == {name: (root / name).read_bytes() for name in critical}
    assert record["decision"] == PARENT_READJUDICATED_DECISION
    assert record["historical_decision"] == "training_controls_failed"
    assert record["failure_domain"] == "implementation_contract"
    assert record["cache_gate_passed"] == 1
    assert record["cache_seal_verified"] == 1
    assert record["control_evidence_opened"] == 0
    assert record["physical_training_performed"] == 0
    assert record["validation_selection_performed"] == 0
    assert record["confirmation_namespace_opened"] == 0
    assert record["immutable_registry"] == {
        "artifact_count": PARENT_REGISTRY_COUNT,
        "file_sha256": PARENT_REGISTRY_FILE_SHA256,
        "semantic_sha256": PARENT_REGISTRY_SEMANTIC_SHA256,
        "complete_file_set_verified": 1,
    }
    assert record["semantic_sha256"] == config_fingerprint(
        {key: value for key, value in record.items() if key != "semantic_sha256"}
    )


def test_oom_arithmetic_exactly_explains_attempted_activation() -> None:
    assert TRAIN_ROW_COUNT * 32 * 28 * 28 * 4 == FAILED_FORWARD_ACTIVATION_BYTES
    assert FAILED_FORWARD_ACTIVATION_BYTES / 1024**3 == FAILED_FORWARD_ACTIVATION_GIB


def test_external_cache_binding_is_small_complete_and_tamper_evident() -> None:
    binding = build_immutable_cache_binding(_parent())
    assert binding["cache_is_read_only"] == 1
    assert binding["cache_copied"] == 0
    assert binding["cache_linked"] == 0
    assert binding["physical_labels_deserialized_during_binding"] == 0
    assert binding["roles"]["train"]["row_count"] == 114_688
    assert binding["roles"]["validation"]["row_count"] == 57_344
    assert verify_immutable_cache_binding(binding) == binding

    changed = dict(binding)
    changed["cache_is_read_only"] = 0
    with pytest.raises(
        BoundaryTangentV3MemoryProvenanceError, match="semantic hash changed"
    ):
        verify_immutable_cache_binding(changed)


def test_wrong_parent_directory_is_rejected_before_use(tmp_path: Path) -> None:
    wrong = tmp_path / "not-the-frozen-parent"
    wrong.mkdir()
    with pytest.raises(BoundaryTangentV3MemoryProvenanceError, match="basename"):
        verify_failed_v3_train_parent(wrong)

