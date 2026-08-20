from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

from mnist import (
    diag_d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication as cli,
)
from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError
from mnist.d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication import (
    CandidateRoleDecomposition,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_memory import (
    HostInputStore,
    ModelCallBatchGuard,
)
from mnist.d0_jacobi_rb_boundary_tangent_zero_baseline import (
    ZeroBaselineBoundaryTangentPredictor,
)
from mnist.d0_jacobi_rb_learnability import PHASE_DURATIONS, PHASE_MATCHINGS


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        stage="preflight",
        require_gate="preflight",
        parent_quartile_specialist_run_dir=(tmp_path / "parent").resolve(),
        resume_run_dir=None,
        runs_root=(tmp_path / "runs").resolve(),
        run_name="test",
        device="cpu",
    )


def _snapshot(root: Path) -> dict[str, object]:
    return cli._semantic(
        {
            "schema": "fixture-snapshot",
            "schema_version": 1,
            "run_dir": str(root),
            "file_count": 0,
            "files": [],
            "tree_sha256": cli.config_fingerprint([]),
        }
    )


def _decomposition(paths: int = 2) -> CandidateRoleDecomposition:
    shape = (paths, 7, 8)
    cross = np.full(shape, 0.25, dtype=np.float64)
    energy = np.full(shape, 0.5, dtype=np.float64)
    raw = 2.0 * cross - energy
    counts = np.ones(shape, dtype=np.int64)
    return CandidateRoleDecomposition(
        path_ids=np.arange(100, 100 + paths, dtype=np.int64),
        cross_term=cross,
        prediction_energy=energy,
        raw_improvement=raw,
        parent_gain_improvement=raw,
        diagnostic_gain_improvement=np.full(shape, 0.125, dtype=np.float64),
        fine_cell_row_count=counts,
        parent_gain=1.0,
        diagnostic_gain=0.5,
        maximum_raw_identity_error=0.0,
        maximum_parent_gain_identity_error=0.0,
    )


def test_parse_requires_resume_for_later_stages(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--stage",
                "replay",
                "--parent-quartile-specialist-run-dir",
                str(tmp_path / "parent"),
            ]
        )


def test_preflight_writes_read_only_plan_and_named_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.parent_quartile_specialist_run_dir.mkdir()
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    snapshot = _snapshot(args.parent_quartile_specialist_run_dir)
    provenance = {
        "passed": 1,
        "decision": "no_training_only_quartile_system",
        "valid_scientific_negative": 1,
        "all_registered_artifact_hashes_verified": 1,
        "all_checkpoint_hashes_verified": 1,
        "checkpoint_count": 492,
        "cache_bindings_valid": 1,
        "role_open_history_valid": 1,
        "selection_confirmation_absent": 1,
    }
    monkeypatch.setattr(cli, "snapshot_parent_run", lambda _root: snapshot)
    monkeypatch.setattr(cli, "verify_parent", lambda *_a, **_kw: provenance)
    cli._preflight_stage(run_dir, args)
    plan = cli._load_json(run_dir / "adjudication_plan.json")
    gate = cli._load_json(run_dir / "preflight_gate.json")
    firewall = cli._load_json(run_dir / "role_firewall.json")
    assert plan["restartable_job_count"] == 960
    assert plan["checkpoint_selection_performed"] == 0
    assert firewall["permitted_roles"] == ["gain_calibration", "training_rank"]
    assert firewall["role_open_creation_allowed"] == 0
    assert gate["passed"] == 1
    cli._verify_stage_seal(run_dir, "preflight_artifact_seal.json")


def test_candidate_array_contract_uses_canonical_480_order() -> None:
    candidates = cli.NONZERO_CANDIDATE_IDENTITIES
    table = {
        "candidate_quartile": np.asarray([row.quartile for row in candidates]),
        "candidate_seed": np.asarray([row.seed for row in candidates]),
        "candidate_update": np.asarray([row.update for row in candidates]),
    }
    assert cli._candidate_arrays_valid(table, "candidate_")
    table["candidate_update"] = table["candidate_update"].copy()
    table["candidate_update"][1] += 100
    assert not cli._candidate_arrays_valid(table, "candidate_")


def test_restart_shard_binds_checkpoint_role_and_config(tmp_path: Path) -> None:
    candidate = cli.NONZERO_CANDIDATE_IDENTITIES[0]
    row = {
        "checkpoint_path": "checkpoints/value.pt",
        "checkpoint_file_sha256": "a" * 64,
        "model_state_sha256": "b" * 64,
    }
    value = _decomposition()
    cli._save_shard(
        tmp_path,
        role="gain_calibration",
        candidate=candidate,
        decomposition=value,
        checkpoint_row=row,
        role_binding_sha256="c" * 64,
        role_open_sha256="d" * 64,
        scientific_config_sha256="e" * 64,
    )
    loaded = cli._load_shard(
        tmp_path,
        role="gain_calibration",
        candidate=candidate,
        checkpoint_row=row,
        role_binding_sha256="c" * 64,
        role_open_sha256="d" * 64,
        scientific_config_sha256="e" * 64,
    )
    assert loaded is not None
    assert np.array_equal(loaded.cross_term, value.cross_term)
    with pytest.raises(ArtifactCompatibilityError):
        cli._load_shard(
            tmp_path,
            role="gain_calibration",
            candidate=candidate,
            checkpoint_row=row,
            role_binding_sha256="f" * 64,
            role_open_sha256="d" * 64,
            scientific_config_sha256="e" * 64,
        )


def test_streamed_reduction_never_exceeds_batch_32() -> None:
    paths = np.asarray([10, 11], dtype=np.int64)
    cells = [(path, phase, midpoint) for path in paths for phase in range(7) for midpoint in range(8)]
    count = len(cells)
    arrays = {
        "later_full_state": np.full((count, 784), 1.0 / 784.0, dtype=np.float32),
        "reverse_time": np.full(count, 0.5, dtype=np.float64),
        "phase": np.asarray([phase for _, phase, _ in cells], dtype=np.int64),
        "color": np.asarray(
            [PHASE_MATCHINGS[phase] for _, phase, _ in cells], dtype=np.int64
        ),
        "duration": np.asarray(
            [PHASE_DURATIONS[phase] for _, phase, _ in cells], dtype=np.float64
        ),
        "label": np.full(count, 3, dtype=np.int64),
        "path_id": np.asarray([path for path, _, _ in cells], dtype=np.int64),
        "midpoint_index": np.asarray([midpoint for _, _, midpoint in cells], dtype=np.int64),
    }
    store = HostInputStore.from_arrays(arrays, role="train")
    model = ZeroBaselineBoundaryTangentPredictor(zero_residual=True)
    target = np.linspace(-0.1, 0.1, count * 392, dtype=np.float64).reshape(count, 392)
    guard = ModelCallBatchGuard(maximum_batch_size=32)
    result = cli._reduce_streamed_candidate(
        model=model,
        store=store,
        target=target,
        rows=np.arange(count, dtype=np.int64),
        device=torch.device("cpu"),
        guard=guard,
        expected_path_ids=paths,
        parent_gain=1.0,
        diagnostic_gain=0.5,
    )
    assert result.cross_term.shape == (2, 7, 8)
    assert guard.maximum_observed_batch_size == 32
    assert np.count_nonzero(result.prediction_energy) == 0
    assert result.maximum_identity_error <= cli.IDENTITY_TOLERANCE


def test_workflow_hard_stop_is_exit_two_and_never_authorizes_work(tmp_path: Path) -> None:
    for name in ("preflight", "replay", "decompose", "adjudicate"):
        cli.atomic_write_json(
            tmp_path / f"{name}_gate.json",
            {"evaluation_status": "evaluated", "passed": 1},
        )
    cli.atomic_write_json(
        tmp_path / "adjudicate_metrics.json",
        {
            "decision_evidence": {
                "quartiles": {
                    str(q): {
                        "cross_role_stable_candidate_count": 0,
                        "power_only_evidence": 0,
                        "mechanism_localized": 1,
                    }
                    for q in (1, 2, 3)
                }
            }
        },
    )
    workflow = cli._workflow_record(tmp_path, "adjudicate")
    decision = workflow["decision"]
    assert decision["decision"] == "no_later_quartile_direction_detectable_under_current_class"
    assert cli.decision_exit_code(decision) == 2
    assert decision["physical_training_authorized"] == 0
    assert decision["sampling_authorized"] == 0


def test_artifact_registry_records_zero_execution_scope(tmp_path: Path) -> None:
    cli.atomic_write_json(tmp_path / "evidence.json", {"value": 1})
    registry = cli._artifact_registry(tmp_path)
    assert registry["new_transitions_generated"] == 0
    assert registry["optimizer_updates_performed"] == 0
    assert registry["controller_or_sampling_work_performed"] == 0
    assert [row["path"] for row in registry["artifacts"]] == ["evidence.json"]
