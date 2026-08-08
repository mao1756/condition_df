from __future__ import annotations

from pathlib import Path

import pytest

from mnist.d0_jacobi_artifacts import file_fingerprint
from mnist.d0_jacobi_rb_boundary_tangent_v3_time_local_provenance import (
    BAYES_PARENT_BASENAME,
    CONFIRMATION_PATH_START,
    CONFIRMATION_PATH_STOP,
    MEMORY_PARENT_BASENAME,
    MEMORY_PARENT_DECISION,
    TimeLocalProvenanceError,
    WITNESS_PARENT_BASENAME,
    snapshot_parent_run,
    verify_parent_immutability_snapshot,
    verify_time_local_adjudication_parents,
)


def _parents() -> tuple[Path, Path, Path]:
    memory = (
        Path("runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_memory_confirmation")
        / MEMORY_PARENT_BASENAME
    ).resolve()
    witness = (
        Path("runs/experiment12_d0_jacobi_rb_physical_coarse_signal_witness")
        / WITNESS_PARENT_BASENAME
    ).resolve()
    bayes = (
        Path("runs/experiment12_d0_jacobi_rb_bayes_power_calibration")
        / BAYES_PARENT_BASENAME
    ).resolve()
    return memory, witness, bayes


def test_exact_terminal_parents_and_unopened_confirmation_verify() -> None:
    memory, witness, bayes = _parents()
    protected = {
        path: file_fingerprint(path)
        for path in (
            memory / "artifact_registry.json",
            memory / "selection_artifact_seal.json",
            witness / "artifact_registry.json",
            bayes / "artifact_registry.json",
        )
    }
    result = verify_time_local_adjudication_parents(
        memory_v3_run_dir=memory,
        coarse_witness_run_dir=witness,
        bayes_power_run_dir=bayes,
        verify_external_cache=False,
    )
    assert result["passed"] == 1
    assert result["parents"]["memory_safe_v3_selection"]["decision"] == (
        MEMORY_PARENT_DECISION
    )
    assert result["all_checkpoint_hashes_verified"] == 1
    assert result["selection_seal_verified"] == 1
    assert result["confirmation_namespace_opened"] == 0
    firewall = result["parents"]["memory_safe_v3_selection"][
        "confirmation_firewall"
    ]
    assert firewall["path_start"] == CONFIRMATION_PATH_START
    assert firewall["path_stop_exclusive"] == CONFIRMATION_PATH_STOP
    assert firewall["confirmation_evidence_paths"] == []
    assert all(row["unchanged"] == 1 for row in result["parent_immutability"].values())
    assert protected == {path: file_fingerprint(path) for path in protected}


def test_parent_snapshot_detects_content_change(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    artifact = parent / "evidence.json"
    artifact.write_text('{"passed":1}\n', encoding="utf-8")
    snapshot = snapshot_parent_run(parent)
    assert verify_parent_immutability_snapshot(parent, snapshot) == snapshot
    artifact.write_text('{"passed":0}\n', encoding="utf-8")
    with pytest.raises(TimeLocalProvenanceError, match="snapshot changed"):
        verify_parent_immutability_snapshot(parent, snapshot)


def test_wrong_memory_parent_is_rejected_before_artifact_use(tmp_path: Path) -> None:
    _, witness, bayes = _parents()
    wrong = tmp_path / "not-the-frozen-memory-parent"
    wrong.mkdir()
    with pytest.raises(TimeLocalProvenanceError, match="basename"):
        verify_time_local_adjudication_parents(
            memory_v3_run_dir=wrong,
            coarse_witness_run_dir=witness,
            bayes_power_run_dir=bayes,
            verify_external_cache=False,
        )

