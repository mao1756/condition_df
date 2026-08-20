from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mnist import diag_d0_jacobi_rb_absolute_coordinate_adjudication as cli
from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError
from mnist.d0_jacobi_rb_absolute_coordinate import synthetic_coordinate_fixture
from mnist.d0_jacobi_rb_absolute_coordinate_gate import safety_record


def _base_argv(tmp_path: Path) -> list[str]:
    return [
        "--parent-directional-result-archive",
        str(tmp_path / "directional.zip"),
        "--parent-coarse-witness-run-dir",
        str(tmp_path / "coarse"),
    ]


def _write_gate(run_dir: Path, name: str, *, passed: int = 1) -> None:
    cli.atomic_write_json(
        run_dir / name,
        {
            "schema": f"fixture-{name}",
            "schema_version": 1,
            "evaluation_status": "evaluated",
            "passed": int(passed),
        },
    )


def _supported_evidence() -> dict[str, object]:
    return {
        "control_provenance_valid": 1,
        "portable_directional_parent_valid": 1,
        "coarse_witness_parent_valid": 1,
        "coordinate_hypothesis_plan_valid": 1,
        "coarse_witness_replay_valid": 1,
        "translation_symmetry_audit_valid": 1,
        "coordinate_projection_algebra_valid": 1,
        "coordinate_inference_valid": 1,
        "q0_positive_control": 1,
        "later_quartile_positive": {"q1": 1, "q2": 1, "q3": 1},
    }


def test_parse_contract_and_production_freezes(tmp_path: Path) -> None:
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
    assert args.bootstrap_replicates == 50_000

    with pytest.raises(SystemExit):
        cli.parse_args(["--stage", "replay", *_base_argv(tmp_path)])
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--stage",
                "symmetry",
                "--resume-run-dir",
                str(tmp_path / "run"),
                "--require-gate",
                "replay",
                *_base_argv(tmp_path),
            ]
        )
    with pytest.raises(SystemExit):
        cli.parse_args(["--device", "cuda", *_base_argv(tmp_path)])
    with pytest.raises(SystemExit):
        cli.parse_args(
            ["--bootstrap-replicates", "32", *_base_argv(tmp_path)]
        )

    test_args = cli.parse_args(
        [
            "--test-only",
            "--device",
            "cuda",
            "--bootstrap-replicates",
            "32",
            *_base_argv(tmp_path),
        ]
    )
    assert test_args.device == "cuda"
    assert test_args.bootstrap_replicates == 32


def test_stage_sequence_prerequisites_and_readiness(tmp_path: Path) -> None:
    assert cli._stage_sequence("all") == (
        "preflight",
        "replay",
        "symmetry",
        "decompose",
        "report",
    )
    assert cli._stage_sequence("decompose") == ("decompose",)

    with pytest.raises(ArtifactCompatibilityError, match="preflight_gate"):
        cli._require_prerequisite(tmp_path, "replay")
    assert cli._readiness_decision(tmp_path) == "preflight_not_passed"

    _write_gate(tmp_path, "preflight_gate.json")
    cli._require_prerequisite(tmp_path, "replay")
    assert cli._readiness_decision(tmp_path) == "ready_for_replay"

    _write_gate(tmp_path, "replay_gate.json")
    cli._require_prerequisite(tmp_path, "symmetry")
    assert cli._readiness_decision(tmp_path) == "ready_for_symmetry"

    _write_gate(tmp_path, "symmetry_gate.json")
    cli._require_prerequisite(tmp_path, "decompose")
    assert cli._readiness_decision(tmp_path) == "ready_for_decompose"

    _write_gate(tmp_path, "decomposition_gate.json")
    cli.atomic_write_json(
        tmp_path / "absolute_coordinate_decision_evidence.json",
        _supported_evidence(),
    )
    cli._require_prerequisite(tmp_path, "report")
    assert cli._readiness_decision(tmp_path) == (
        "absolute_coordinate_representation_hypothesis_supported"
    )


def test_execution_failure_commits_readable_artifacts_before_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs"
    args = cli.parse_args(
        [
            "--stage",
            "preflight",
            "--require-gate",
            "preflight",
            "--runs-root",
            str(runs_root),
            "--run-name",
            "fixture",
            *_base_argv(tmp_path),
        ]
    )

    def fail_preflight(_run_dir: Path, _args: argparse.Namespace) -> None:
        raise cli.AbsoluteCoordinateWorkflowError(
            "fixture failure",
            failure_domain="coordinate_plan",
            failure_code="fixture_preflight_failure",
        )

    monkeypatch.setattr(cli, "_preflight_stage", fail_preflight)
    assert cli._run(args) == 2

    run_dirs = list(runs_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    failure = cli._load_json(run_dir / "preflight_execution_failure.json")
    gate = cli._load_json(run_dir / "preflight_gate.json")
    status = cli._load_json(run_dir / "run_status.json")
    registry = cli._load_json(run_dir / "artifact_registry.json")
    assert failure["failure_code"] == "fixture_preflight_failure"
    assert gate["evaluation_status"] == "execution_failed"
    assert status["state"] == "execution_failed"
    assert status["scientific_evidence_complete"] == 0
    registered = {row["path"] for row in registry["artifacts"]}
    assert {
        "preflight_execution_failure.json",
        "preflight_gate.json",
        "run_status.json",
        "workflow_gate.json",
    } <= registered
    for record in (failure, gate, status, registry):
        for field, value in cli.NO_WORK.items():
            assert record[field] == value


@pytest.mark.parametrize(
    ("stage", "error", "expected"),
    [
        (
            "preflight",
            cli.AbsoluteCoordinateProvenanceError("portable archive changed"),
            "portable_directional_parent_invalid",
        ),
        (
            "preflight",
            cli.AbsoluteCoordinateProvenanceError("coarse witness changed"),
            "control_provenance_invalid",
        ),
        (
            "preflight",
            cli.AbsoluteCoordinateWorkflowError("hypothesis plan changed"),
            "coordinate_hypothesis_plan_invalid",
        ),
        (
            "replay",
            cli.AbsoluteCoordinateWorkflowError("replay changed"),
            "coarse_witness_replay_invalid",
        ),
        (
            "symmetry",
            cli.AbsoluteCoordinateWorkflowError("audit changed"),
            "translation_symmetry_audit_invalid",
        ),
        (
            "decompose",
            cli.AbsoluteCoordinateWorkflowError("projection changed"),
            "coordinate_projection_algebra_invalid",
        ),
        (
            "decompose",
            cli.AbsoluteCoordinateWorkflowError("bootstrap inference changed"),
            "coordinate_inference_invalid",
        ),
    ],
)
def test_integrity_failures_map_to_closed_decisions(
    stage: str, error: BaseException, expected: str
) -> None:
    evidence = cli._failure_decision_evidence(stage, error)
    decision = cli.decide_absolute_coordinate(evidence)
    assert decision["decision"] == expected
    assert decision["scientific_evidence_complete"] == 0
    assert decision["invalid_evidence"] == 1


def test_report_and_terminal_decision_keep_zero_work_and_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_verify_stage_seal", lambda *_args, **_kwargs: None)
    cli.atomic_write_json(
        tmp_path / "absolute_coordinate_decision_evidence.json",
        _supported_evidence(),
    )
    cli.atomic_write_json(
        tmp_path / "panel_b_linear_inference.json",
        {
            "scaled_signed_cross_lower_bounds": [0.4, 0.3, 0.2, 0.1],
            "signed_cross_energies": [0.5, 0.4, 0.3, 0.2],
            "inference": {"critical_value": 2.5},
        },
    )
    cli.atomic_write_json(
        tmp_path / "coordinate_decomposition.json",
        {"maximum_reconstruction_error": 1.0e-16},
    )

    decision = cli._write_report(tmp_path)
    assert decision["decision"] == (
        "absolute_coordinate_representation_hypothesis_supported"
    )
    assert decision["fresh_coordinate_learner_plan_drafting_recommended"] == 1
    for field, value in {**cli.NO_WORK, **safety_record()}.items():
        assert decision[field] == value
    report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "read-only, historical, post-hoc" in report
    assert "does not identify a unique architecture" in report
    assert "no new scientific evidence" in report


def test_reduced_all_stage_workflow_is_restartable_and_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs"
    archive_path = tmp_path / "directional.zip"
    archive_path.write_bytes(b"fixture archive")
    witness = tmp_path / "coarse"
    (witness / "panels" / "a").mkdir(parents=True)
    (witness / "panels" / "b").mkdir(parents=True)
    fixture = synthetic_coordinate_fixture(path_count=24, noise_scale=0.01)
    np.savez_compressed(
        witness / "panels" / "a" / "cell_means.npz",
        cell_means=fixture.left.cell_means,
        path_ids=fixture.left.path_ids,
    )
    np.savez_compressed(
        witness / "panels" / "b" / "cell_means.npz",
        cell_means=fixture.right.cell_means,
        path_ids=fixture.right.path_ids,
    )
    analysis = {
        "bootstrap": {
            "point_estimate": cli.EXPECTED_COARSE_POINT,
            "lower_bound": 0.0005,
            "upper_bound": 0.0008,
        },
        "classification": {"decision": "exact_physical_coarse_signal_detected"},
        "left_panel": {
            "shape": [64, 4, 7, 392],
            "path_count": 64,
            "path_ids": list(range(64)),
        },
        "right_panel": {
            "shape": [64, 4, 7, 392],
            "path_count": 64,
            "path_ids": list(range(256, 320)),
        },
    }
    (witness / "physical_coarse_signal_analysis.json").write_text(
        json.dumps(analysis), encoding="utf-8"
    )
    (witness / "physical_coarse_signal_decision.json").write_text(
        json.dumps({"decision": "exact_physical_coarse_signal_detected"}),
        encoding="utf-8",
    )

    snapshots = {
        "schema": "fixture-parent-snapshots",
        "portable_directional": {"sha256": "a" * 64},
        "coarse_witness": {"tree_sha256": "b" * 64},
        "semantic_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        cli,
        "snapshot_absolute_coordinate_parents",
        lambda **_kwargs: snapshots,
    )
    monkeypatch.setattr(
        cli,
        "verify_absolute_coordinate_parents",
        lambda **_kwargs: {
            "evaluation_status": "evaluated",
            "passed": 1,
            "provenance_valid": 1,
            "portable_directional_parent_valid": 1,
            "coarse_witness_parent_valid": 1,
        },
    )
    monkeypatch.setattr(
        cli,
        "verify_absolute_coordinate_parent_immutability",
        lambda **_kwargs: {"evaluation_status": "evaluated", "passed": 1},
    )

    args = cli.parse_args(
        [
            "--stage",
            "all",
            "--require-gate",
            "decompose",
            "--runs-root",
            str(runs_root),
            "--run-name",
            "reduced",
            "--bootstrap-replicates",
            "200",
            "--test-only",
            *_base_argv(tmp_path),
        ]
    )
    assert cli._run(args) == 0
    run_dir = next(runs_root.iterdir())
    status = cli._load_json(run_dir / "run_status.json")
    assert status["state"] == "complete"
    assert status["decision"] == (
        "absolute_coordinate_representation_hypothesis_supported"
    )
    assert (run_dir / "REPORT.md").is_file()
    assert (run_dir / "panel_a_direction_artifact_seal.json").is_file()
    opening = cli._load_json(run_dir / "panel_b_opening_record.json")
    assert opening["opened_after_panel_a_direction_artifact_seal"] == 1
    assert opening["linear_evaluation_commit_count"] == 1
    intent = cli._load_json(run_dir / "panel_b_opening_intent.json")
    assert intent["committed_before_panel_b_array_loading"] == 1
    cli._verify_semantic(
        cli._load_json(run_dir / "coordinate_lattice.json"),
        "coordinate lattice",
    )
    cli._verify_semantic(
        cli._load_json(run_dir / "parent_provenance.json"),
        "parent provenance",
    )
    with np.load(run_dir / "panel_b_linear_evidence.npz", allow_pickle=False) as payload:
            for field, value in {**cli.NO_WORK, **safety_record()}.items():
                assert field in payload.files
                assert np.asarray(payload[field]).size == 1
                assert int(np.asarray(payload[field]).reshape(-1)[0]) == value
    with (run_dir / "frequency1_heldout_inference.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    for row in rows:
        for field, value in {**cli.NO_WORK, **safety_record()}.items():
            assert int(row[field]) == value
    registry_before = (run_dir / "artifact_registry.json").read_bytes()

    report_args = cli.parse_args(
        [
            "--stage",
            "report",
            "--resume-run-dir",
            str(run_dir),
            "--bootstrap-replicates",
            "200",
            "--test-only",
            *_base_argv(tmp_path),
        ]
    )
    assert cli._run(report_args) == 0
    assert (run_dir / "artifact_registry.json").read_bytes() != registry_before
    assert cli._load_json(run_dir / "panel_b_opening_record.json")[
        "linear_evaluation_commit_count"
    ] == 1
    terminal = cli._load_json(run_dir / "absolute_coordinate_adjudication_decision.json")
    for field, value in {**cli.NO_WORK, **safety_record()}.items():
        assert terminal[field] == value
