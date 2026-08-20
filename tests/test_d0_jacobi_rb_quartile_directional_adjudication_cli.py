from __future__ import annotations

import argparse
from types import SimpleNamespace
from pathlib import Path

import pytest

from mnist import (
    diag_d0_jacobi_rb_boundary_tangent_quartile_directional_adjudication as cli,
)
from mnist.d0_jacobi_rb_quartile_directional_adjudication_gate import (
    ZERO_AUTHORIZATION_FIELDS,
    ZERO_WORK_FIELDS,
)
from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError


def _base_argv(tmp_path: Path) -> list[str]:
    return [
        "--parent-quartile-specialist-run-dir",
        str(tmp_path / "specialist"),
        "--parent-time-local-run-dir",
        str(tmp_path / "time-local"),
        "--device",
        "cpu",
        "--test-only",
    ]


def _no_signal_evidence() -> dict[str, object]:
    return {
        "q0_full": {"stable_direction": 1, "stable_effect": 1},
        "inferential_and_role_order_valid": 1,
        "branch_algebra_cancellation_valid": 1,
        "quartiles": {
            quartile: {
                "components": {
                    component: {"stable_direction": 0, "stable_effect": 0}
                    for component in cli.COMPONENT_NAMES
                }
            }
            for quartile in ("q1", "q2", "q3")
        },
    }


def _write_passing_gates(run_dir: Path) -> None:
    filenames = {
        "preflight": "preflight_gate.json",
        "replay": "historical_replay_gate.json",
        "controls": "controls_gate.json",
        "fittrace": "fittrace_gate.json",
        "nominate": "nominate_gate.json",
        "adjudicate": "adjudicate_gate.json",
    }
    for stage, filename in filenames.items():
        cli.atomic_write_json(
            run_dir / filename,
            {
                "schema": f"fixture-{stage}",
                "schema_version": 1,
                "gate": stage,
                "evaluation_status": "evaluated",
                "passed": 1,
            },
        )


def test_parse_stage_and_required_gate_contract(tmp_path: Path) -> None:
    args = cli.parse_args(
        [
            "--stage",
            "preflight",
            "--require-gate",
            "preflight",
            *_base_argv(tmp_path),
        ]
    )
    assert args.stage == "preflight"
    assert args.require_gate == "preflight"
    assert args.device == "cpu"
    assert args.test_only

    with pytest.raises(SystemExit):
        cli.parse_args(["--stage", "fittrace", *_base_argv(tmp_path)])
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--stage",
                "nominate",
                "--require-gate",
                "adjudicate",
                "--resume-run-dir",
                str(tmp_path / "run"),
                *_base_argv(tmp_path),
            ]
        )
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--stage",
                "preflight",
                "--device",
                "cpu",
                "--parent-quartile-specialist-run-dir",
                str(tmp_path / "specialist"),
                "--parent-time-local-run-dir",
                str(tmp_path / "time-local"),
            ]
        )


def test_stage_order_and_frozen_candidate_inference_plans() -> None:
    assert cli._stage_sequence("all") == (
        "preflight",
        "replay",
        "controls",
        "fittrace",
        "nominate",
        "adjudicate",
        "report",
    )
    assert cli._stage_sequence("controls") == ("controls",)

    candidate_plan = cli._candidate_component_plan()
    assert candidate_plan["candidate_count"] == 480
    assert candidate_plan["candidate_component_count"] == 1_440
    assert candidate_plan["components"] == list(cli.COMPONENT_NAMES)
    assert candidate_plan["nominee_stream_count"] == 36

    inference_plan = cli._inference_plan()
    assert inference_plan["family_size"] == 72
    assert inference_plan["bootstrap_replicates"] == 50_000
    assert inference_plan["resampling_unit"] == "whole training-rank path"
    assert inference_plan["standard_error_floor"] is None

    bootstrap = cli._bootstrap_seal(path_count=4)
    assert bootstrap["replicates"] == 50_000
    assert bootstrap["path_count"] == 4
    assert bootstrap["seed"] == 261_352
    assert bootstrap["count_dtype"] == "|u1"
    assert len(bootstrap["count_matrix_sha256"]) == 64
    assert bootstrap == cli._bootstrap_seal(path_count=4)


def test_workflow_scientific_stop_and_report_remain_nonauthorizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_passing_gates(tmp_path)
    evidence = _no_signal_evidence()
    cli.atomic_write_json(
        tmp_path / "adjudicate_metrics.json", {"decision_evidence": evidence}
    )

    workflow = cli._workflow_record(tmp_path, "adjudicate")
    decision = workflow["decision"]
    assert decision["decision"] == (
        "no_later_quartile_signal_detectable_under_permitted_class_stop"
    )
    assert decision["valid_scientific_stop"] == 1
    assert workflow["required_gate_pass"] == 1
    for field in ZERO_AUTHORIZATION_FIELDS + ZERO_WORK_FIELDS:
        assert decision[field] == 0

    before = {
        "quartile_specialist": {"tree_sha256": "a" * 64},
        "time_local": {"tree_sha256": "b" * 64},
    }
    cli.atomic_write_json(tmp_path / "parent_immutability_before.json", before)
    monkeypatch.setattr(cli, "_verify_stage_seal", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        cli,
        "snapshot_parent_run",
        lambda root: {"run_dir": str(root), "tree_sha256": "c" * 64},
    )
    monkeypatch.setattr(
        cli,
        "compare_parent_snapshots",
        lambda _before, _after: {"passed": 1},
    )
    args = argparse.Namespace(
        parent_quartile_specialist_run_dir=tmp_path / "specialist",
        parent_time_local_run_dir=tmp_path / "time-local",
    )
    cli._report_stage(tmp_path, args)

    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "no_later_quartile_signal_detectable" in report
    assert "generated no transitions" in report
    assert "does not authorize training" in report
    after = cli._load_json(tmp_path / "parent_immutability_after.json")
    assert after["passed"] == 1
    assert after["quartile_specialist_unchanged"] == 1
    assert after["time_local_unchanged"] == 1


def test_failed_stage_gate_and_artifact_registry_keep_zero_scope(tmp_path: Path) -> None:
    failure = {
        "failure_domain": "prelabel_control",
        "failure_code": "fixture_control_failure",
        "error": "fixture",
    }
    cli._commit_failed_stage_gate(
        tmp_path, stage="controls", failure=failure
    )
    gate = cli._load_json(tmp_path / "controls_gate.json")
    assert gate["evaluation_status"] == "execution_failed"
    assert gate["passed"] == 0
    assert gate["failure_code"] == "fixture_control_failure"
    for field in ZERO_AUTHORIZATION_FIELDS + ZERO_WORK_FIELDS:
        assert gate[field] == 0

    registry = cli._artifact_registry(tmp_path)
    assert registry["artifact_count"] >= 2
    assert {row["path"] for row in registry["artifacts"]} >= {
        "controls_gate.json",
        "controls_metrics.json",
    }
    for field in ZERO_AUTHORIZATION_FIELDS + ZERO_WORK_FIELDS:
        assert registry[field] == 0


def test_fresh_initialize_resumes_exactly_and_rejects_changed_parent(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    args = SimpleNamespace(
        parent_quartile_specialist_run_dir=(tmp_path / "specialist").resolve(),
        parent_time_local_run_dir=(tmp_path / "time-local").resolve(),
        test_only=True,
    )
    cli._initialize(run_dir, args, resumed=False)
    manifest_before = (run_dir / "run_manifest.json").read_bytes()
    config_before = (run_dir / "scientific_config.json").read_bytes()

    cli._initialize(run_dir, args, resumed=True)
    assert (run_dir / "run_manifest.json").read_bytes() == manifest_before
    assert (run_dir / "scientific_config.json").read_bytes() == config_before

    changed = SimpleNamespace(
        parent_quartile_specialist_run_dir=args.parent_quartile_specialist_run_dir,
        parent_time_local_run_dir=(tmp_path / "different-time-local").resolve(),
        test_only=True,
    )
    with pytest.raises(ArtifactCompatibilityError, match="resume manifest"):
        cli._initialize(run_dir, changed, resumed=True)
    assert (run_dir / "run_manifest.json").read_bytes() == manifest_before
