from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import numpy as np

from mnist import diag_d0_jacobi_rb_boundary_tangent_frequency1_coordinate_learnability as cli
from mnist.d0_jacobi_artifacts import config_fingerprint, file_fingerprint
from mnist.d0_jacobi_rb_boundary_tangent_eager_cache import EagerCohort


def _parents(tmp_path: Path) -> list[str]:
    archive = tmp_path / "directional.zip"
    archive.write_bytes(b"test-only")
    return [
        "--parent-absolute-coordinate-run-dir", str(tmp_path),
        "--parent-memory-v3-run-dir", str(tmp_path),
        "--parent-coarse-witness-run-dir", str(tmp_path),
        "--parent-directional-result-archive", str(archive),
    ]


def _fixture_args(tmp_path: Path, name: str) -> list[str]:
    return [
        "--test-only",
        "--stage", "all",
        "--require-gate", "none",
        "--device", "cpu",
        "--runs-root", str(tmp_path / name),
        "--run-name", name,
        "--test-path-count", "8",
        "--test-outer-steps", "16",
        "--test-bootstrap-replicates", "8",
        *_parents(tmp_path),
    ]


def _only_run(root: Path) -> Path:
    values = [path for path in root.iterdir() if path.is_dir()]
    assert len(values) == 1
    return values[0]


def _json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_production_parser_rejects_all_cpu_and_test_authority(tmp_path: Path) -> None:
    parents = _parents(tmp_path)
    with pytest.raises(SystemExit):
        cli.parse_args(["--stage", "all", *parents])
    with pytest.raises(SystemExit):
        cli.parse_args(["--stage", "preflight", "--device", "cpu", *parents])
    with pytest.raises(SystemExit):
        cli.parse_args(
            ["--test-only", "--stage", "preflight", "--require-gate", "preflight", *parents]
        )
    report = cli.parse_args(
        ["--stage", "report", "--device", "cpu", "--resume-run-dir", str(tmp_path), *parents]
    )
    assert report.device == "cpu"


def test_confirmation_artifact_firewall_catches_inherited_and_raw_names(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    confirmation = root / "confirmation"
    confirmation.mkdir(parents=True)
    (confirmation / "control_anchor_states.npz").write_bytes(b"forbidden")
    (confirmation / "raw-predictions.npz").write_bytes(b"forbidden")
    assert cli._forbidden_confirmation_artifacts(root) == [
        "confirmation/control_anchor_states.npz",
        "confirmation/raw-predictions.npz",
    ]


def test_milestone_reductions_only_confirmation_shard_reuses_without_anchor(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    cohort = EagerCohort(
        kind="confirmation",
        index=0,
        path_ids=(0x1B00, 0x1B01),
        path_roles=("confirmation", "confirmation"),
    )
    start_step = 120
    selected_step = 127
    namespace = "1" * 64
    current = np.full((2, 784), 1.0 / 784.0, dtype=np.float64)
    final = np.roll(current, 1, axis=1).copy(order="C")
    state_path, risk_path, anchor_path, metadata_path = cli._v3_cli._confirmation_shard_paths(
        run, cohort_index=0, start_step=start_step
    )
    state_artifact = cli._atomic_npz(state_path, final_states=final)
    risk_arrays = {
        "sample_keys": np.asarray([11, 12], dtype=np.int64),
        "path_ids": np.asarray(cohort.path_ids, dtype=np.int64),
        "outer_steps": np.full(2, selected_step, dtype=np.int16),
        "phases": np.asarray([0, 1], dtype=np.int8),
        "midpoint_indices": np.asarray([0, 1], dtype=np.int8),
        "model_vs_zero": np.asarray([0.25, 0.5], dtype=np.float64),
    }
    risk_artifact = cli._atomic_npz(risk_path, **risk_arrays)
    input_hash = cli._v3_cli._array_sha(current)
    body = {
        "schema": cli.RUN_SCHEMA + "-confirmation-shard",
        "schema_version": 1,
        "cohort_index": 0,
        "path_ids": list(cohort.path_ids),
        "start_step": start_step,
        "selected_step": selected_step,
        "input_state_sha256": input_hash,
        "namespace_sha256": namespace,
        "state_file_sha256": state_artifact["sha256"],
        "final_state_sha256": cli._v3_cli._array_sha(final),
        "risk_file_sha256": risk_artifact["sha256"],
        "control_anchor_file_sha256": None,
        "execution": {
            "identity": {
                "cohort_kind": "confirmation",
                "cohort_index": 0,
                "start_step": start_step,
                "step_count": 8,
            },
            "path_ids": list(cohort.path_ids),
            "path_roles": list(cohort.path_roles),
            "selected_step": selected_step,
            "input_state_sha256": input_hash,
            "base_record": {},
            "branch_records": [],
            "diagnostics": {},
            "raw_payload_persisted": 0,
        },
        "complete_pipeline_elapsed_seconds": 1.0,
        "raw_confirmation_inputs_persisted": 0,
        "raw_confirmation_labels_persisted": 0,
        "committed": 1,
        **cli.NO_WORK,
    }
    record = {**body, "semantic_sha256": config_fingerprint(body)}
    cli.atomic_write_json(metadata_path, record)

    resumed = cli._load_reductions_only_confirmation_shard(
        run,
        cohort=cohort,
        start_step=start_step,
        current=current,
        selected_step=selected_step,
        namespace_sha256=namespace,
    )
    assert resumed is not None
    np.testing.assert_array_equal(resumed[0], final)
    assert resumed[1] == record
    assert not anchor_path.exists()

    anchor_path.write_bytes(b"forbidden stale anchor")
    assert cli._load_reductions_only_confirmation_shard(
        run,
        cohort=cohort,
        start_step=start_step,
        current=current,
        selected_step=selected_step,
        namespace_sha256=namespace,
    ) is None


def test_reduced_all_no_candidate_is_terminal_and_never_opens_confirmation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "negative"
    assert cli.main(_fixture_args(tmp_path, "negative")) == 0
    run = _only_run(root)
    decision = _json(run / "frequency1_coordinate_learnability_decision.json")
    assert decision["decision"] == "no_frequency1_coordinate_validation_candidate"
    assert decision["valid_scientific_negative"] == 1
    assert not (run / "confirmation_namespace_open.json").exists()
    assert (run / "REPORT.md").is_file()
    registry = _json(run / "artifact_registry.json")
    registered = {row["path"] for row in registry["artifacts"]}
    assert {
        "run_status.json",
        "workflow_gate.json",
        "frequency1_coordinate_learnability_decision.json",
        "REPORT.md",
    } <= registered

    resume = [
        "--test-only", "--stage", "report", "--require-gate", "none",
        "--device", "cpu", "--resume-run-dir", str(run),
        "--test-path-count", "8", "--test-outer-steps", "16",
        "--test-bootstrap-replicates", "8", *_parents(tmp_path),
    ]
    assert cli.main(resume) == 0
    before = {
        path.relative_to(run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*") if path.is_file()
    }
    assert cli.main([*resume, "--test-maximum-updates", "1"]) == 2
    after = {
        path.relative_to(run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*") if path.is_file()
    }
    assert after == before


def test_reduced_nominee_and_confirmation_positive_only_recommends_planning(
    tmp_path: Path,
) -> None:
    args = [
        *_fixture_args(tmp_path, "positive"),
        "--test-maximum-updates", "1",
        "--test-selection-outcome", "nominee",
        "--test-confirmation-outcome", "pass",
    ]
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "positive")
    decision = _json(run / "frequency1_coordinate_learnability_decision.json")
    assert decision["decision"] == (
        "exact_rb_frequency1_coordinate_boundary_tangent_time_local_signal_confirmed"
    )
    assert decision["controller_control_patch_planning_authorized"] == 1
    for name in (
        "controller_execution_authorized",
        "reconstruction_authorized",
        "reverse_sampling_authorized",
        "sampling_authorized",
    ):
        assert decision[name] == 0
    assert _json(run / "control_synthetic_coordinate_teacher.json")[
        "interrupted_vs_uninterrupted_state_hash_equal"
    ] == 1
    assert _json(run / "control_firewall_memory_restart.json")[
        "monkeypatched_audit_fields_cannot_influence_output"
    ] == 1
    assert _json(run / "confirmation_gate.json")["passed"] == 1


def test_stage_exception_commits_failure_gate_status_decision_and_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_run_dir: Path, _args: object) -> dict:
        raise cli.Frequency1CoordinateCLIError(
            "fixture failure", failure_domain="path_or_resource_plan", failure_code="fixture"
        )

    monkeypatch.setattr(cli, "_preflight_stage", fail)
    args = [
        "--test-only", "--stage", "preflight", "--require-gate", "none",
        "--device", "cpu", "--runs-root", str(tmp_path / "failure"),
        "--run-name", "failure", *_parents(tmp_path),
    ]
    assert cli.main(args) == 2
    run = _only_run(tmp_path / "failure")
    assert (run / "preflight_execution_failure.json").is_file()
    assert _json(run / "preflight_gate.json")["passed"] == 0
    assert _json(run / "run_status.json")["state"] == "execution_failed"
    assert (run / "frequency1_coordinate_learnability_decision.json").is_file()
    assert (run / "artifact_registry.json").is_file()


def test_fresh_initialization_failure_commits_minimal_closed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_args: object, *, snapshots: object = None) -> dict:
        del snapshots
        raise cli.Frequency1CoordinateCLIError(
            "parent fixture invalid",
            failure_domain="parent_provenance",
            failure_code="fixture_parent_invalid",
        )

    monkeypatch.setattr(cli, "_verify_parents", fail)
    args = [
        "--test-only", "--stage", "preflight", "--require-gate", "none",
        "--device", "cpu", "--runs-root", str(tmp_path / "init-failure"),
        "--run-name", "init-failure", *_parents(tmp_path),
    ]
    assert cli.main(args) == 2
    run = _only_run(tmp_path / "init-failure")
    for name in (
        "run_manifest.json",
        "parent_provenance_failure.json",
        "initialization_failure.json",
        "preflight_gate.json",
        "frequency1_coordinate_learnability_decision.json",
        "workflow_gate.json",
        "run_status.json",
        "artifact_registry.json",
    ):
        assert (run / name).is_file()
    assert _json(run / "frequency1_coordinate_learnability_decision.json")["decision"] == (
        "frequency1_coordinate_parent_provenance_invalid"
    )
