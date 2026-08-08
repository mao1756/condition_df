from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_json,
    file_fingerprint,
)
from mnist.d0_jacobi_rb_boundary_tangent_false_discovery_gate import (
    evaluate_preflight_gate,
)
from mnist import (
    diag_d0_jacobi_rb_boundary_tangent_false_discovery_adjudication as cli,
)


def _args(tmp_path: Path, *, stage: str = "all", require_gate: str = "decision"):
    return argparse.Namespace(
        stage=stage,
        require_gate=require_gate,
        parent_run_dir=(tmp_path / "parent").resolve(),
        resume_run_dir=None,
        runs_root=(tmp_path / "runs").resolve(),
        run_name="fixture",
        device="cpu",
        bootstrap_replicates=8,
        test_only=True,
    )


def _passing_preflight() -> dict[str, object]:
    return {
        "evaluation_status": "evaluated",
        "passed": 1,
        "scientific_evidence_complete": 1,
    }


def _failed_preflight() -> dict[str, object]:
    return {
        "evaluation_status": "evaluated",
        "passed": 0,
        "scientific_evidence_complete": 0,
        "failure_domain": "forensic_evidence",
    }


def test_all_stage_preflight_failure_skips_adjudication_and_commits_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    calls = {"adjudicate": 0}

    monkeypatch.setattr(cli, "_initialize", lambda *_args, **_kwargs: None)

    def preflight(run_dir: Path, _args: argparse.Namespace):
        gate = _failed_preflight()
        atomic_write_json(run_dir / "preflight_gate.json", gate)
        return gate

    def adjudicate(*_args, **_kwargs):
        calls["adjudicate"] += 1
        raise AssertionError("adjudication must not run")

    monkeypatch.setattr(cli, "_preflight_stage", preflight)
    monkeypatch.setattr(cli, "_adjudicate_stage", adjudicate)
    assert cli._run(args) == 2
    run_dir = next(args.runs_root.iterdir())
    assert calls["adjudicate"] == 0
    decision = cli._load_json(run_dir / "false_discovery_decision.json")
    assert decision["decision"] == "forensic_evidence_invalid"
    assert (run_dir / "decision_artifact_seal.json").is_file()
    assert (run_dir / "artifact_registry.json").is_file()


def test_all_stage_unexpected_adjudication_failure_commits_closed_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    monkeypatch.setattr(cli, "_initialize", lambda *_args, **_kwargs: None)

    def preflight(run_dir: Path, _args: argparse.Namespace):
        gate = _passing_preflight()
        atomic_write_json(run_dir / "preflight_gate.json", gate)
        return gate

    monkeypatch.setattr(cli, "_preflight_stage", preflight)
    monkeypatch.setattr(
        cli,
        "_adjudicate_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            cli.FalseDiscoveryAdjudicationError(
                "forced replay fault", failure_code="forced_replay_fault"
            )
        ),
    )
    assert cli._run(args) == 2
    run_dir = next(args.runs_root.iterdir())
    assert (run_dir / "all_stage_failure_decision_commit.json").is_file()
    assert (run_dir / "false_discovery_decision.json").is_file()
    assert (run_dir / "decision_gate.json").is_file()
    status = cli._load_json(run_dir / "run_status.json")
    assert status["state"] == "execution_failed"
    assert status["decision"] in {
        "forensic_evidence_invalid",
        "implementation_or_replay_defect",
    }
    registry = cli._load_json(run_dir / "artifact_registry.json")
    registered = {row["path"] for row in registry["artifacts"]}
    assert "all_stage_failure_decision_commit.json" in registered
    assert "adjudicate_execution_failure.json" in registered


def test_decision_requires_adjudication_after_passing_preflight(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(run_dir / "preflight_gate.json", _passing_preflight())
    with pytest.raises(ArtifactCompatibilityError, match="requires committed adjudication"):
        cli._decision_stage(run_dir, _args(tmp_path, stage="decision"))
    assert not (run_dir / "false_discovery_decision.json").exists()

    atomic_write_json(run_dir / "preflight_gate.json", _failed_preflight())
    cli._decision_stage(run_dir, _args(tmp_path, stage="decision"))
    assert (
        cli._load_json(run_dir / "false_discovery_decision.json")["decision"]
        == "forensic_evidence_invalid"
    )


def test_orphan_candidate_commit_is_discarded_but_complete_mismatch_fails(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    candidate = {
        "seed": 261_312,
        "update": 100,
        "checkpoint_file_sha256": "a" * 64,
        "state_sha256": "b" * 64,
    }
    paths = np.arange(0xEC200, 0xEC208, dtype=np.int64)
    npz_path, json_path = cli._candidate_output_paths(run_dir, 261_312, 100)
    cli._atomic_npz(
        npz_path,
        path_ids=paths,
        path_values=np.ones((8, 228), dtype=np.float64),
    )
    assert cli._load_candidate_output(
        run_dir, candidate, expected_paths=paths
    ) is None
    assert not npz_path.exists()

    artifact = cli._atomic_npz(
        npz_path,
        path_ids=paths,
        path_values=np.ones((8, 228), dtype=np.float64),
    )
    record = cli._semantic_record(
        {
            "seed": 261_312,
            "update": 100,
            "checkpoint_file_sha256": "a" * 64,
            "checkpoint_state_sha256": "b" * 64,
            "path_table_sha256": "c" * 64,
            "path_count": 8,
            "family_size": 228,
        }
    )
    atomic_write_json(json_path, record)
    with pytest.raises(ArtifactCompatibilityError, match="binding changed"):
        cli._load_candidate_output(run_dir, candidate, expected_paths=paths)
    assert artifact["sha256"] == file_fingerprint(npz_path)
    assert npz_path.is_file() and json_path.is_file()


def test_complete_candidate_commit_rejects_path_table_tamper(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    candidate = {
        "seed": 261_312,
        "update": 100,
        "checkpoint_file_sha256": "a" * 64,
        "state_sha256": "b" * 64,
    }
    paths = np.arange(0xEC200, 0xEC208, dtype=np.int64)
    npz_path, json_path = cli._candidate_output_paths(run_dir, 261_312, 100)
    artifact = cli._atomic_npz(
        npz_path,
        path_ids=paths + 1,
        path_values=np.ones((8, 228), dtype=np.float64),
    )
    atomic_write_json(
        json_path,
        cli._semantic_record(
            {
                "seed": 261_312,
                "update": 100,
                "checkpoint_file_sha256": "a" * 64,
                "checkpoint_state_sha256": "b" * 64,
                "path_table_sha256": artifact["sha256"],
                "path_count": 8,
                "family_size": 228,
            }
        ),
    )
    with pytest.raises(ArtifactCompatibilityError, match="table contract"):
        cli._load_candidate_output(run_dir, candidate, expected_paths=paths)


def test_preflight_gate_before_seal_is_replayed_and_finalized(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics = {
        "schema": "fixture",
        "evaluation_status": "execution_failed",
        "failure_domain": "forensic_evidence",
        "failure_code": "fixture",
        **cli._scope(),
    }
    for name in ("parent_provenance.json", "candidate_plan.json", "role_firewall.json"):
        atomic_write_json(run_dir / name, cli._semantic_record({"name": name, **cli._scope()}))
    atomic_write_json(run_dir / "preflight_metrics.json", metrics)
    atomic_write_json(run_dir / "preflight_gate.json", evaluate_preflight_gate(metrics))
    cli._preflight_stage(run_dir, _args(tmp_path, stage="preflight"))
    assert (run_dir / "preflight_artifact_seal.json").is_file()
    cli._verify_stage_seal(run_dir, "preflight_artifact_seal.json")


def test_registered_prefix_allows_new_restart_commits_but_rejects_tamper(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tracked = run_dir / "tracked.json"
    atomic_write_json(tracked, {"value": 1})
    cli._artifact_registry(run_dir)
    atomic_write_json(run_dir / "new_candidate.json", {"value": 2})
    cli._verify_registered_prefix(run_dir)
    atomic_write_json(tracked, {"value": 3})
    with pytest.raises(ArtifactCompatibilityError, match="registered child artifact"):
        cli._verify_registered_prefix(run_dir)


def test_candidate_evaluation_uses_exact_cuda_replay_batch_size_32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, int] = {}
    paths = np.arange(0xEC200, 0xEC208, dtype=np.int64)
    target = torch.ones((8, 2), dtype=torch.float64)
    baseline_prediction = torch.full_like(target, 0.5)
    prediction = torch.zeros_like(target)
    candidate = {
        "seed": 261_312,
        "update": 100,
        "checkpoint_path": "checkpoint.pt",
        "checkpoint_file_sha256": "a" * 64,
        "state_sha256": "b" * 64,
        "validation_mse": 1.0,
        "validation_high_reverse_time_mse": 1.0,
        "baseline_validation_mse": 0.25,
        "baseline_high_reverse_time_mse": 0.25,
        "zero_validation_mse": 1.0,
        "zero_high_reverse_time_mse": 1.0,
        "combined_vs_baseline": -0.75,
        "combined_vs_baseline_high_reverse_time": -0.75,
        "combined_vs_zero": 0.0,
        "combined_vs_zero_high_reverse_time": 0.0,
    }

    monkeypatch.setattr(cli._trainer, "_load_candidate_model", lambda *_args: object())

    def predict(_model, _inputs, *, batch_size: int):
        calls["batch_size"] = batch_size
        return prediction

    monkeypatch.setattr(cli._trainer, "_predict_in_batches", predict)
    table = SimpleNamespace(
        path_ids=paths,
        path_values=np.ones((8, 228), dtype=np.float64),
        sample_key_sha256="d" * 64,
    )
    monkeypatch.setattr(cli, "aggregate_confirmation_improvements", lambda **_kwargs: table)
    arrays = {
        "sample_key": np.arange(8, dtype=np.int64),
        "path_id": paths,
        "outer_step": np.zeros(8, dtype=np.int64),
        "phase": np.zeros(8, dtype=np.int64),
        "midpoint_index": np.zeros(8, dtype=np.int64),
    }
    cli._evaluate_candidate(
        tmp_path / "run",
        tmp_path / "parent",
        candidate,
        baseline=object(),
        model_inputs=object(),
        target=target,
        baseline_prediction=baseline_prediction,
        arrays=arrays,
        expected_paths=paths,
        device=torch.device("cpu"),
    )
    assert calls["batch_size"] == 32


def test_direct_predictor_import_and_no_work_scope_are_frozen() -> None:
    source = inspect.getsource(cli._candidate_adjudication)
    assert "BoundaryTangentPredictor(" in source
    assert "_trainer.BoundaryTangentPredictor" not in source
    assert cli.PREDICTION_BATCH_SIZE == 32
    assert cli._scope() == cli.NO_WORK
    assert all(value == 0 for value in cli._scope().values())


def test_resource_limits_are_inclusive_and_fail_nextafter() -> None:
    assert cli._resource_limits_passed(
        elapsed_seconds=cli.MAXIMUM_ADJUDICATION_SECONDS,
        peak_memory_fraction=cli.MAXIMUM_PEAK_MEMORY_FRACTION,
        persisted_bytes=cli.MAXIMUM_PERSISTED_BYTES,
    )
    assert not cli._resource_limits_passed(
        elapsed_seconds=np.nextafter(cli.MAXIMUM_ADJUDICATION_SECONDS, np.inf),
        peak_memory_fraction=0.0,
        persisted_bytes=0,
    )
    assert not cli._resource_limits_passed(
        elapsed_seconds=0.0,
        peak_memory_fraction=np.nextafter(cli.MAXIMUM_PEAK_MEMORY_FRACTION, np.inf),
        persisted_bytes=0,
    )
    assert not cli._resource_limits_passed(
        elapsed_seconds=0.0,
        peak_memory_fraction=0.0,
        persisted_bytes=cli.MAXIMUM_PERSISTED_BYTES + 1,
    )


def test_preflight_metric_contract_includes_exact_parent_bindings() -> None:
    source = inspect.getsource(cli._preflight_metrics)
    for field in (
        '"parent_terminal_decision"',
        '"parent_source_fingerprint"',
        '"parent_scientific_config_sha256"',
        '"parent_registry_semantic_sha256"',
    ):
        assert field in source
    assert "PREFLIGHT_FLAGS" in source
