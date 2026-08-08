from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from mnist import (
    diag_d0_jacobi_rb_boundary_tangent_eager_pipeline_confirmation as cli,
)
from mnist.d0_jacobi_artifacts import (
    atomic_write_json,
    config_fingerprint,
    file_fingerprint,
)
from mnist.d0_jacobi_rb_boundary_tangent_eager_pipeline_provenance import (
    PARENT_RUN_BASENAME,
)


PARENT = Path(
    "runs/experiment12_d0_jacobi_rb_boundary_tangent_prefix_schedule_confirmation"
) / PARENT_RUN_BASENAME


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        stage="all",
        require_gate="none",
        parent_prefix_run_dir=PARENT.resolve(),
        resume_run_dir=None,
        runs_root=tmp_path,
        run_name="test-eager-complete-pipeline",
        device="cpu",
        test_only=True,
    )


def test_parser_and_stage_sequence_expose_only_the_frozen_workflow(
    tmp_path: Path,
) -> None:
    required = ["--parent-prefix-run-dir", str(tmp_path / "parent")]
    args = cli.parse_args(required + ["--stage", "report", "--device", "cpu"])
    assert args.stage == "report"
    assert args.require_gate == "none"
    assert cli.STAGES == ("preflight", "pilot", "report", "all")
    assert cli.REQUIRED_GATES == ("none", "preflight", "pilot")
    assert cli._stage_sequence("all") == ("preflight", "pilot")
    assert cli._stage_sequence("report") == ()
    with pytest.raises(SystemExit):
        cli.parse_args(required + ["--stage", "pilot", "--device", "cpu"])
    with pytest.raises(SystemExit):
        cli.parse_args(
            required + ["--test-only", "--require-gate", "pilot"]
        )


def test_scientific_config_freezes_counts_thresholds_and_no_work(
    tmp_path: Path,
) -> None:
    record = cli._scientific_config(_args(tmp_path))
    assert record["root_seed"] == 261321
    assert record["repeat_count"] == 3
    assert record["projected_base_transitions"] == 224_788_480
    assert record["projected_midpoint_transitions"] == 112_394_240
    assert record["projected_total_transitions"] == 337_182_720
    assert record["maximum_projected_seconds"] == 108_000.0
    assert record["minimum_projected_effective_rate"] == pytest.approx(
        3_122.0622222222223
    )
    assert record["prefix_profile"] == "eager_prefix_128_tpb128"
    assert record["physical_training_performed"] == 0
    assert record["sampling_performed"] == 0
    body = dict(record)
    assert body.pop("semantic_sha256") == config_fingerprint(body)


def test_source_set_binds_gate_provenance_and_exact_parent_closure() -> None:
    names = {path.name for path in cli._source_set()}
    assert {
        "diag_d0_jacobi_rb_boundary_tangent_eager_pipeline_confirmation.py",
        "diag_d0_jacobi_rb_boundary_tangent_prefix_schedule_confirmation.py",
        "d0_jacobi_rb_boundary_tangent_eager_pipeline_gate.py",
        "d0_jacobi_rb_boundary_tangent_eager_pipeline_provenance.py",
        "d0_jacobi_rb_boundary_tangent_prefix_fallback.py",
        "d0_jacobi_rb_boundary_tangent_prefix_schedule.py",
        "d0_jacobi_rb_boundary_tangent_schedule.py",
        "d0_jacobi_rb_boundary_tangent_schedule_provenance.py",
        "d0_jacobi_rb_cuda.py",
    } <= names


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable prefix parent unavailable")
def test_complete_test_only_all_stage_passes_and_writes_authorizing_evidence(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    assert cli._run(args) == 0
    runs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(runs) == 1
    run_dir = runs[0]

    preflight = cli._load_json(run_dir / "eager_pipeline_preflight_gate.json")
    pilot = cli._load_json(run_dir / "eager_pipeline_gate.json")
    workflow = cli._load_json(run_dir / "workflow_gate.json")
    decision = cli._load_json(run_dir / "eager_pipeline_decision.json")
    status = cli._load_json(run_dir / "run_status.json")

    assert preflight["passed"] == 1
    assert pilot["passed"] == 1
    assert pilot["resource_valid"] == 1
    assert workflow["decision"]["decision"] == (
        "exact_boundary_tangent_eager_pipeline_feasible"
    )
    assert decision["schedule_integration_authorized"] == 1
    assert decision["training_authorized"] == 0
    assert decision["sampling_authorized"] == 0
    assert status["state"] == "test_only_complete"

    metrics = cli._load_json(run_dir / "eager_pipeline_metrics.json")
    assert metrics["repeat_count"] == 3
    assert metrics["profile_count"] == 4
    assert metrics["slowest_repeat_selection_valid"] == 1
    assert metrics["repeat_averaging_not_used"] == 1
    assert metrics["unfavorable_repeat_rerun_count"] == 0
    assert metrics["projected_total_transitions"] == 337_182_720
    assert metrics["certificate_fraction"] == 1.0
    assert metrics["forbidden_event_count"] == 0

    for name in (
        "pilot_namespace_audit.json",
        "eager_pipeline_repeat_registry.json",
        "eager_pipeline_projection.json",
        "eager_pipeline_shard_registry.json",
        "artifact_registry.json",
    ):
        assert (run_dir / name).is_file()


def test_workflow_helper_handles_missing_pilot_without_positional_api_error(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(run_dir / "parent_provenance.json", {"passed": 1})
    atomic_write_json(
        run_dir / "eager_pipeline_preflight_gate.json",
        {"evaluation_status": "evaluated", "passed": 1},
    )
    workflow = cli._workflow(run_dir, "preflight")
    assert workflow["required_gate_pass"] == 1
    assert workflow["decision"]["decision"] == "ready_for_pilot"
    assert (run_dir / "eager_pipeline_decision.json").is_file()


def test_execution_failure_writes_artifacts_before_returning_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.stage = "preflight"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(cli, "_make_run_dir", lambda _: (run_dir, False))
    monkeypatch.setattr(cli, "_initialize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda *_: {})

    def fail(*_args: object) -> dict[str, object]:
        raise cli.EagerPipelineCLIError(
            "pipeline fixture failed",
            failure_domain="pipeline_execution",
            failure_code="fixture_pipeline_failed",
        )

    monkeypatch.setattr(cli, "_preflight_stage", fail)
    assert cli._run(args) == 2
    failure = cli._load_json(run_dir / "preflight_execution_failure.json")
    gate = cli._load_json(run_dir / "eager_pipeline_preflight_gate.json")
    status = cli._load_json(run_dir / "run_status.json")
    assert failure["evaluation_status"] == "execution_failed"
    assert failure["failure_code"] == "fixture_pipeline_failed"
    assert gate["stage_execution_valid"] == 0
    assert status["state"] == "execution_failed"
    assert (run_dir / "artifact_registry.json").is_file()


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable prefix parent unavailable")
def test_preflight_then_pilot_resume_preserves_seals_and_skips_completed_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.stage = "preflight"
    assert cli._run(args) == 0
    runs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(runs) == 1
    run_dir = runs[0]
    preflight_seal = file_fingerprint(run_dir / "preflight_artifact_seal.json")
    assert not (run_dir / "eager_pipeline_gate.json").exists()

    args.stage = "pilot"
    args.resume_run_dir = run_dir
    assert cli._run(args) == 0
    assert file_fingerprint(run_dir / "preflight_artifact_seal.json") == preflight_seal
    pilot = cli._load_json(run_dir / "eager_pipeline_gate.json")
    assert pilot["passed"] == 1
    assert pilot["numerically_valid"] == 1
    assert pilot["resource_valid"] == 1
    assert (run_dir / "pilot_artifact_seal.json").is_file()

    def rerun_forbidden() -> list[object]:
        raise AssertionError("a sealed pilot must be skipped on resume")

    monkeypatch.setattr(cli._prefix, "_test_pilot_records", rerun_forbidden)
    assert cli._run(args) == 0


def test_failed_preflight_blocks_pilot_and_commits_gate_failure_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.require_gate = "preflight"
    args.test_only = False
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(run_dir / "parent_provenance.json", {"passed": 1})
    calls: list[str] = []

    monkeypatch.setattr(cli, "_make_run_dir", lambda _: (run_dir, False))
    monkeypatch.setattr(cli, "_initialize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda *_: {})

    def fail_preflight(*_args: object) -> dict[str, object]:
        calls.append("preflight")
        gate: dict[str, object] = {
            "evaluation_status": "evaluated",
            "gate": "preflight",
            "passed": 0,
            "failure_domain": "design",
            "scientific_evidence_complete": 0,
        }
        atomic_write_json(run_dir / "eager_pipeline_preflight_gate.json", gate)
        return gate

    def pilot(*_args: object) -> dict[str, object]:
        calls.append("pilot")
        return {"evaluation_status": "evaluated", "passed": 1}

    monkeypatch.setattr(cli, "_preflight_stage", fail_preflight)
    monkeypatch.setattr(cli, "_pilot_stage", pilot)
    assert cli._run(args) == 2
    assert calls == ["preflight"]
    for name in (
        "workflow_gate.json",
        "eager_pipeline_decision.json",
        "run_status.json",
        "artifact_registry.json",
    ):
        assert (run_dir / name).is_file()
    status = cli._load_json(run_dir / "run_status.json")
    assert status["state"] == "gate_failed"


def test_execution_failure_replaces_only_an_unsealed_passing_gate(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(run_dir / "parent_provenance.json", {"passed": 1})
    atomic_write_json(
        run_dir / "eager_pipeline_preflight_gate.json",
        {"evaluation_status": "evaluated", "passed": 1},
    )
    error = cli.EagerPipelineCLIError(
        "preflight exploded",
        failure_domain="pipeline_execution",
        failure_code="fixture_preflight_failed",
    )

    cli._commit_execution_failure(
        run_dir, stage="preflight", exc=error, require_gate="preflight"
    )

    failure = cli._load_json(run_dir / "preflight_execution_failure.json")
    gate = cli._load_json(run_dir / "eager_pipeline_preflight_gate.json")
    workflow = cli._load_json(run_dir / "workflow_gate.json")
    registry = cli._load_json(run_dir / "artifact_registry.json")
    assert failure["failure_code"] == "fixture_preflight_failed"
    assert gate["evaluation_status"] == "execution_failed"
    assert gate["passed"] == 0
    assert workflow["required_gate_pass"] == 0
    registered = {str(row["path"]) for row in registry["artifacts"]}
    assert "preflight_execution_failure.json" in registered
    assert "eager_pipeline_preflight_gate.json" in registered


def test_pilot_stage_fails_closed_without_a_passing_preflight(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(
        cli.ArtifactCompatibilityError,
        match="pilot requires a passing preflight gate",
    ):
        cli._pilot_stage(run_dir, _args(tmp_path))
