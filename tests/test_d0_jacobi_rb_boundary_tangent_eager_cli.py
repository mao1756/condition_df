from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from mnist import (
    diag_d0_jacobi_rb_boundary_tangent_eager_confirmation as cli,
)
from mnist.d0_jacobi_artifacts import (
    atomic_write_json,
    config_fingerprint,
    file_fingerprint,
)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        stage="preflight",
        require_gate="none",
        parent_eager_pipeline_run_dir=tmp_path / "eager-parent",
        failed_boundary_tangent_run_dir=tmp_path / "failed-tangent",
        parent_coarse_residual_run_dir=tmp_path / "coarse-parent",
        resume_run_dir=None,
        runs_root=tmp_path,
        run_name="fixture-eager-v2",
        device="cpu",
        test_only=True,
        test_path_count=2,
        test_outer_steps=16,
        test_maximum_updates=0,
        test_bootstrap_replicates=8,
    )


def _parser_parents(tmp_path: Path) -> list[str]:
    return [
        "--parent-eager-pipeline-run-dir",
        str(tmp_path / "eager-parent"),
        "--failed-boundary-tangent-run-dir",
        str(tmp_path / "failed-tangent"),
        "--parent-coarse-residual-run-dir",
        str(tmp_path / "coarse-parent"),
    ]


def _passing_gate(stage: str) -> dict[str, object]:
    return {
        "schema": cli.RUN_SCHEMA + f"-{stage}-gate-fixture",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "gate": stage,
        "passed": 1,
        "scientific_evidence_complete": 1,
        **cli.NO_WORK,
    }


def _training_inputs() -> dict[str, object]:
    return {
        "train_inputs": object(),
        "validation_inputs": object(),
        "train_path_rows": np.asarray([0xEC100], dtype=np.int64),
        "validation_path_rows": np.asarray([0xEC200], dtype=np.int64),
        "train_arrays": {"sample_key": np.asarray([1], dtype=np.int64)},
        "validation_arrays": {"sample_key": np.asarray([2], dtype=np.int64)},
        "train_index": {"semantic_sha256": "train"},
        "validation_index": {"semantic_sha256": "validation"},
    }


def _control_record(run_dir: Path, *, passed: bool) -> dict[str, object]:
    synthetic = {
        "passed": int(passed),
        "relative_validation_mse": 0.005 if passed else 1.0,
        "every_validation_path_beats_zero": int(passed),
    }
    null = {
        "passed": int(passed),
        "selected_update": 0 if passed else 100,
    }
    atomic_write_json(run_dir / "synthetic_teacher_control.json", synthetic)
    (run_dir / "synthetic_teacher_per_path.csv").write_text(
        "path_id,improvement\n966912,1\n", encoding="utf-8"
    )
    atomic_write_json(run_dir / "exact_baseline_null_control.json", null)
    return {
        "passed": int(passed),
        "synthetic_metrics": synthetic,
        "null_metrics": null,
    }


def _physical_result(run_dir: Path, *, update: int) -> dict[str, object]:
    seed = cli.MODEL_SEEDS[0]
    selected = {
        "update": update,
        "eligible_nonzero": int(update > 0),
        "combined_vs_baseline": 1.0 if update > 0 else 0.0,
        "combined_vs_baseline_high_reverse_time": 1.0 if update > 0 else 0.0,
    }
    selection = {"selected_seed": seed, "selected_update": update}
    report = {
        "seed": seed,
        "complete": 1,
        "finite": 1,
        "selected": selected,
    }
    (run_dir / "tangent_baseline.npz").write_bytes(b"fixture")
    atomic_write_json(run_dir / "tangent_baseline.json", {"passed": 1})
    atomic_write_json(run_dir / "checkpoint_selection.json", selection)
    (run_dir / "physical_seed_metrics.csv").write_text(
        "seed,selected_update\n" + f"{seed},{update}\n", encoding="utf-8"
    )
    return {
        "selection": selection,
        "reports": [report],
        "nonzero_reports": [report] if update > 0 else [],
    }


def _write_mock_initialization(run_dir: Path, args: argparse.Namespace) -> None:
    atomic_write_json(run_dir / "scientific_config.json", cli._scientific_config(args))


def test_parser_and_stage_sequence_are_fail_closed(tmp_path: Path) -> None:
    parents = _parser_parents(tmp_path)
    report = cli.parse_args(
        parents
        + [
            "--stage",
            "report",
            "--device",
            "cpu",
            "--resume-run-dir",
            str(tmp_path / "existing-run"),
        ]
    )
    assert report.stage == "report"
    assert report.require_gate == "none"
    assert report.parent_eager_pipeline_run_dir.is_absolute()
    assert report.failed_boundary_tangent_run_dir.is_absolute()
    assert report.parent_coarse_residual_run_dir.is_absolute()

    assert cli.STAGES == (
        "preflight",
        "cache",
        "train",
        "confirm",
        "report",
        "all",
    )
    assert cli.REQUIRED_GATES == (
        "none",
        "preflight",
        "cache",
        "train",
        "confirm",
    )
    assert cli._stage_sequence("all") == (
        "preflight",
        "cache",
        "train",
        "confirm",
    )
    assert cli._stage_sequence("report") == ()
    with pytest.raises(ValueError):
        cli._stage_sequence("control")

    # Production mutation stages require an existing run; hidden reductions
    # are nonauthorizing and may never be combined with a required gate.
    with pytest.raises(SystemExit):
        cli.parse_args(parents + ["--stage", "cache", "--device", "cuda"])
    with pytest.raises(SystemExit):
        cli.parse_args(
            parents
            + ["--test-only", "--require-gate", "preflight", "--stage", "preflight"]
        )

    reduced = cli.parse_args(
        parents
        + [
            "--test-only",
            "--device",
            "cpu",
            "--stage",
            "all",
            "--test-outer-steps",
            "16",
            "--test-maximum-updates",
            "0",
            "--test-bootstrap-replicates",
            "8",
        ]
    )
    assert reduced.test_only
    assert reduced.require_gate == "none"


def test_scientific_config_freezes_v2_paths_seeds_and_no_work(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.test_only = False
    record = cli._scientific_config(args)
    assert record["root_seed"] == 261_311
    assert record["model_seeds"] == [261_312, 261_313, 261_314]
    assert record["bootstrap_seed"] == 261_315
    assert record["reserved_control_seed"] == 261_316
    assert record["synthetic_teacher_seed"] == 261_317
    assert record["baseline_null_seed"] == 261_318
    plan = cli.build_eager_boundary_tangent_path_plan()
    assert plan["roles"]["train"] == list(range(0xEC100, 0xEC140))
    assert plan["roles"]["validation"] == list(range(0xEC200, 0xEC220))
    assert plan["roles"]["confirmation"] == list(range(0xED000, 0xED040))
    assert plan["roles"]["preflight_seam"] == list(range(0xEF000, 0xEF008))
    assert record["path_id_plan_sha256"] == plan["semantic_sha256"]
    cohort_plan = record["cohort_plan"]
    assert [row["size"] for row in cohort_plan["train_validation"]] == [10] * 9 + [6]
    assert [row["size"] for row in cohort_plan["confirmation"]] == [10] * 6 + [4]
    assert record["eager_execution_contract"]["profile"][
        "certificate_effort"
    ] == "strengthened"
    assert record["eager_execution_contract"]["eager_prefix_contract"][
        "first_authorizing_prefix_bits"
    ] == 128
    assert record["eager_execution_contract"]["branch_runner"].endswith(
        ".sample_fused_midpoint_branches"
    )
    assert record["controller_control_trajectory_performed"] == 0
    assert record["sampling_performed"] == 0
    body = dict(record)
    assert body.pop("semantic_sha256") == config_fingerprint(body)


def test_fresh_initialization_and_resume_bind_all_immutable_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    provenance: dict[str, object] = {
        "schema": "fixture-eager-v2-provenance",
        "passed": 1,
    }
    provenance["semantic_sha256"] = config_fingerprint(provenance)
    readjudication: dict[str, object] = {
        "schema": "fixture-eager-v2-readjudication",
        "passed": 1,
        "decision": "legacy_schedule_resource_projection_superseded",
    }
    readjudication["semantic_sha256"] = config_fingerprint(readjudication)
    monkeypatch.setattr(
        cli, "verify_eager_boundary_tangent_parents", lambda **_: provenance
    )
    monkeypatch.setattr(
        cli,
        "build_eager_boundary_tangent_readjudication",
        lambda value: readjudication if value is provenance else None,
    )

    cli._initialize(run_dir, args, resumed=False)
    assert cli._load_json(run_dir / "parent_provenance.json") == provenance
    assert cli._load_json(run_dir / "parent_readjudication.json") == (
        readjudication
    )
    manifest_before = (run_dir / "run_manifest.json").read_bytes()
    config_before = (run_dir / "scientific_config.json").read_bytes()
    path_plan_before = (run_dir / "path_id_plan.json").read_bytes()

    cli._initialize(run_dir, args, resumed=True)
    assert (run_dir / "run_manifest.json").read_bytes() == manifest_before
    assert (run_dir / "scientific_config.json").read_bytes() == config_before
    assert (run_dir / "path_id_plan.json").read_bytes() == path_plan_before

    changed = {**provenance, "semantic_sha256": "changed"}
    monkeypatch.setattr(
        cli, "verify_eager_boundary_tangent_parents", lambda **_: changed
    )
    monkeypatch.setattr(
        cli, "build_eager_boundary_tangent_readjudication", lambda _: readjudication
    )
    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._initialize(run_dir, args, resumed=True)
    assert (run_dir / "run_manifest.json").read_bytes() == manifest_before


def test_test_only_preflight_reaches_a_passing_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(run_dir / "scientific_config.json", cli._scientific_config(args))
    atomic_write_json(
        run_dir / "path_id_plan.json", cli.build_eager_boundary_tangent_path_plan()
    )
    atomic_write_json(run_dir / "parent_provenance.json", {"passed": 1})
    atomic_write_json(
        run_dir / "parent_readjudication.json",
        {
            "passed": 1,
            "decision": "legacy_schedule_resource_projection_superseded",
        },
    )

    args.parent_eager_pipeline_run_dir.mkdir(parents=True)
    thresholds = cli.BoundaryTangentEagerThresholds()
    atomic_write_json(
        args.parent_eager_pipeline_run_dir / "eager_pipeline_metrics.json",
        {
            "projected_total_transitions": thresholds.total_transitions,
            "projected_base_transitions": thresholds.projected_base_transitions,
            "projected_midpoint_transitions": (
                thresholds.projected_midpoint_transitions
            ),
            "candidate_modes": thresholds.candidate_modes,
            "certificate_fraction": 1.0,
            "forbidden_event_count": 0,
            "projected_elapsed_seconds": 90_000.0,
            "projected_effective_transitions_per_second": (
                thresholds.total_transitions / 90_000.0
            ),
            "minimum_individual_profile_rate": 2_000.0,
            "fallback_fraction": 0.0,
            "fallback_time_fraction": 0.0,
            "maximum_mass_error": 0.0,
            "peak_memory_fraction": 0.5,
            "projected_persisted_bytes": 1_024,
            "maximum_launch_lanes": 2_048,
        },
    )
    monkeypatch.setattr(
        cli._legacy,
        "_load_source_target",
        lambda *_: np.full(784, 1.0 / 784.0, dtype=np.float64),
    )
    monkeypatch.setattr(
        cli._legacy,
        "_representation_preflight",
        lambda *_: {"passed": 1},
    )
    monkeypatch.setattr(
        cli,
        "_preflight_seam",
        lambda *_: {
            "passed": 1,
            "maximum_mass_error": 0.0,
        },
    )
    gate = cli._preflight_stage(run_dir, args)
    assert gate["evaluation_status"] == "evaluated"
    assert gate["passed"] == 1
    assert (run_dir / "preflight_gate.json").is_file()
    assert (run_dir / "preflight_artifact_seal.json").is_file()


def test_reduced_all_orchestration_runs_cache_train_then_confirm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.stage = "all"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_mock_initialization(run_dir, args)
    calls: list[str] = []
    monkeypatch.setattr(cli, "_make_run_dir", lambda _: (run_dir, False))
    monkeypatch.setattr(cli, "_initialize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda *_: {})

    def stage(name: str):
        def execute(root: Path, _args: argparse.Namespace) -> dict[str, object]:
            calls.append(name)
            gate = _passing_gate(name)
            atomic_write_json(root / f"{name}_gate.json", gate)
            return gate

        return execute

    monkeypatch.setattr(cli, "_preflight_stage", stage("preflight"))
    monkeypatch.setattr(cli, "_cache_stage", stage("cache"))
    monkeypatch.setattr(cli, "_train_stage", stage("train"))
    monkeypatch.setattr(cli, "_confirm_stage", stage("confirm"))
    assert cli._run(args) == 0
    assert calls == ["preflight", "cache", "train", "confirm"]
    assert (run_dir / "workflow_gate.json").is_file()
    status = cli._load_json(run_dir / "run_status.json")
    assert status["state"] in {"complete", "test_only_complete"}
    registry = cli._load_json(run_dir / "artifact_registry.json")
    assert registry["physical_training_performed"] in {0, 1}
    assert registry["controller_control_trajectory_performed"] == 0
    assert registry["reverse_sampling_performed"] == 0
    assert registry["sampling_performed"] == 0


def test_controls_run_before_any_physical_label_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(run_dir / "cache_gate.json", _passing_gate("cache"))
    events: list[str] = []

    monkeypatch.setattr(
        cli,
        "_load_training_inputs",
        lambda *_: events.append("inputs") or _training_inputs(),
    )
    monkeypatch.setattr(
        cli,
        "_run_prelabel_controls",
        lambda *_args, **_kwargs: events.append("controls")
        or _control_record(run_dir, passed=True),
    )

    def open_labels(*_args: object, **_kwargs: object):
        assert events == ["inputs", "controls"]
        assert (run_dir / "physical_label_open.json").is_file()
        events.append("labels")
        return {"train": {}, "validation": {}}

    monkeypatch.setattr(cli, "_load_physical_training_labels", open_labels)
    monkeypatch.setattr(
        cli,
        "_run_physical_training",
        lambda *_args, **_kwargs: events.append("physical")
        or _physical_result(run_dir, update=100),
    )
    gate = cli._train_stage(run_dir, args)
    assert gate["passed"] == 1
    assert events == ["inputs", "controls", "labels", "physical"]


def test_failed_controls_leave_physical_labels_unopened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(run_dir / "cache_gate.json", _passing_gate("cache"))
    monkeypatch.setattr(cli, "_load_training_inputs", lambda *_: _training_inputs())
    monkeypatch.setattr(
        cli,
        "_run_prelabel_controls",
        lambda *_args, **_kwargs: _control_record(run_dir, passed=False),
    )
    monkeypatch.setattr(
        cli,
        "_load_physical_training_labels",
        lambda *_args, **_kwargs: pytest.fail("failed controls opened physical labels"),
    )
    gate = cli._train_stage(run_dir, args)
    assert gate["passed"] == 0
    assert not (run_dir / "physical_label_open.json").exists()
    assert not (run_dir / "checkpoint_selection.json").exists()


def test_update_zero_stops_before_confirmation_paths_are_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(run_dir / "cache_gate.json", _passing_gate("cache"))
    monkeypatch.setattr(cli, "_load_training_inputs", lambda *_: _training_inputs())
    monkeypatch.setattr(
        cli,
        "_run_prelabel_controls",
        lambda *_args, **_kwargs: _control_record(run_dir, passed=True),
    )
    monkeypatch.setattr(
        cli, "_load_physical_training_labels", lambda *_: {"train": {}, "validation": {}}
    )
    monkeypatch.setattr(
        cli,
        "_run_physical_training",
        lambda *_args, **_kwargs: _physical_result(run_dir, update=0),
    )
    gate = cli._train_stage(run_dir, args)
    assert gate["passed"] == 0
    assert gate["boundary_tangent_baseline_only"] == 1
    assert not (run_dir / "confirmation_opening_seal.json").exists()
    assert not (run_dir / "confirmation").exists()
    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._confirm_stage(run_dir, args)


def test_required_gate_failure_commits_terminal_artifacts_before_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.require_gate = "preflight"
    args.test_only = False
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_mock_initialization(run_dir, args)
    monkeypatch.setattr(cli, "_make_run_dir", lambda _: (run_dir, False))
    monkeypatch.setattr(cli, "_initialize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda *_: {})

    def fail(root: Path, _args: argparse.Namespace) -> dict[str, object]:
        gate = {
            **_passing_gate("preflight"),
            "passed": 0,
            "scientific_evidence_complete": 1,
        }
        atomic_write_json(root / "preflight_gate.json", gate)
        return gate

    monkeypatch.setattr(cli, "_preflight_stage", fail)
    monkeypatch.setattr(
        cli, "_cache_stage", lambda *_: pytest.fail("failed preflight ran cache")
    )
    assert cli._run(args) == 2
    assert cli._load_json(run_dir / "run_status.json")["state"] == "gate_failed"
    assert cli._load_json(run_dir / "workflow_gate.json")["required_gate_pass"] == 0
    assert (run_dir / "boundary_tangent_eager_decision.json").is_file()
    registry = cli._load_json(run_dir / "artifact_registry.json")
    assert registry["controller_control_trajectory_performed"] == 0
    assert registry["sampling_performed"] == 0


def test_execution_failure_is_not_misreported_as_scientific_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.stage = "cache"
    args.resume_run_dir = tmp_path / "run"
    args.resume_run_dir.mkdir()
    run_dir = args.resume_run_dir
    _write_mock_initialization(run_dir, args)
    monkeypatch.setattr(cli, "_make_run_dir", lambda _: (run_dir, True))
    monkeypatch.setattr(cli, "_initialize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda *_: {})
    atomic_write_json(run_dir / "preflight_gate.json", _passing_gate("preflight"))

    def explode(*_args: object) -> dict[str, object]:
        raise cli.EagerBoundaryTangentCLIError(
            "fixture cache execution failed",
            failure_domain="cache_execution",
            failure_code="fixture_cache_failed",
        )

    monkeypatch.setattr(cli, "_cache_stage", explode)
    assert cli._run(args) == 2
    failure = cli._load_json(run_dir / "cache_execution_failure.json")
    gate = cli._load_json(run_dir / "cache_gate.json")
    assert failure["failure_code"] == "fixture_cache_failed"
    assert failure["scientific_evidence_complete"] == 0
    assert gate["evaluation_status"] == "execution_failed"
    assert cli._load_json(run_dir / "run_status.json")["state"] == (
        "execution_failed"
    )
    registry = cli._load_json(run_dir / "artifact_registry.json")
    assert registry["production_cache_generation_performed"] == 0
    assert registry["confirmation_performed"] == 0


def test_resume_registry_rejects_unregistered_extra_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "committed.json").write_text("{}\n", encoding="utf-8")
    cli._artifact_registry(run_dir)
    cli._verify_existing_registry(run_dir)

    (run_dir / "unregistered.bin").write_bytes(b"not in the terminal snapshot")
    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._verify_existing_registry(run_dir)


@pytest.mark.parametrize("malformation", ["unsafe", "duplicate"])
def test_resume_registry_rejects_unsafe_or_duplicate_rows(
    tmp_path: Path, malformation: str
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "committed.json"
    artifact.write_text("{}\n", encoding="utf-8")
    cli._artifact_registry(run_dir)
    registry_path = run_dir / "artifact_registry.json"
    registry = cli._load_json(registry_path)
    row = next(
        item for item in registry["artifacts"] if item["path"] == "committed.json"
    )
    if malformation == "unsafe":
        outside = tmp_path / "outside.json"
        outside.write_text("{}\n", encoding="utf-8")
        rows = [
            {
                "path": "../outside.json",
                "sha256": file_fingerprint(outside),
                "size": outside.stat().st_size,
            }
        ]
    else:
        rows = [dict(row), dict(row)]
    registry["artifacts"] = rows
    registry["artifact_count"] = len(rows)
    registry["semantic_sha256"] = config_fingerprint({"artifacts": rows})
    atomic_write_json(registry_path, registry)

    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._verify_existing_registry(run_dir)


def test_test_only_workflow_never_emits_authorization_or_planning_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.stage = "preflight"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_mock_initialization(run_dir, args)
    monkeypatch.setattr(cli, "_make_run_dir", lambda _: (run_dir, False))
    monkeypatch.setattr(cli, "_initialize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda *_: {})

    def preflight(root: Path, _args: argparse.Namespace) -> dict[str, object]:
        gate = _passing_gate("preflight")
        atomic_write_json(root / "preflight_gate.json", gate)
        return gate

    monkeypatch.setattr(cli, "_preflight_stage", preflight)
    assert cli._run(args) == 0

    def authorization_values(value: object) -> list[int]:
        found: list[int] = []
        if isinstance(value, dict):
            for key, item in value.items():
                if "authorized" in str(key) or "planning" in str(key):
                    if isinstance(item, (bool, int)):
                        found.append(int(item))
                found.extend(authorization_values(item))
        elif isinstance(value, list):
            for item in value:
                found.extend(authorization_values(item))
        return found

    for name in (
        "workflow_gate.json",
        "boundary_tangent_eager_decision.json",
        "run_status.json",
        "artifact_registry.json",
    ):
        values = authorization_values(cli._load_json(run_dir / name))
        assert all(value == 0 for value in values)


def test_scope_requires_committed_metrics_not_failure_gate_or_opening_seal(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(run_dir / "cache_gate.json", {"passed": 0})
    atomic_write_json(run_dir / "cache_artifact_seal.json", {"artifacts": []})
    atomic_write_json(
        run_dir / "confirmation_seal.json", {"opened_once": 1, "committed": 0}
    )
    assert cli._scope(run_dir)["production_cache_generation_performed"] == 0
    assert cli._scope(run_dir)["confirmation_performed"] == 0


def test_confirmation_persistence_is_inside_positive_pipeline_timing(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state = np.full((1, 784), 1.0 / 784.0, dtype=np.float64)
    execution = SimpleNamespace(
        identity=SimpleNamespace(cohort_index=0, start_step=0),
        path_ids=(0xED000,),
        selected_step=None,
        input_state_sha256=cli._eager_array_sha(state),
        committed_final_states=state,
        to_record=lambda: {
            "schema": "fixture-execution",
            "diagnostics": {"transition_count": 1},
        },
    )
    record = cli._persist_confirmation_shard(
        run_dir,
        execution=execution,
        selection_sha256="a" * 64,
        risks=None,
        audit=None,
        evidence={"row_count": 0},
        pipeline_started_at=time.perf_counter() - 0.01,
    )
    assert record["committed"] == 1
    assert record["persistence_in_timed_region"] == 1
    assert record["complete_pipeline_elapsed_seconds"] > 0.0
    assert cli._load_json(
        run_dir
        / "confirmation/shards/cohort-000/shard-000000/metadata.json"
    ) == record
    cohort = cli.EagerCohort(
        "confirmation", 0, (0xED000,), ("confirmation",)
    )
    assert cli._load_valid_confirmation_shard(
        run_dir,
        cohort=cohort,
        start_step=0,
        current=state,
        selected_step=None,
        selection_sha256="a" * 64,
    ) is not None

    metadata_path = (
        run_dir
        / "confirmation/shards/cohort-000/shard-000000/metadata.json"
    )
    provisional = dict(record)
    provisional["committed"] = 0
    provisional["complete_pipeline_elapsed_seconds"] = 0.0
    provisional["semantic_sha256"] = config_fingerprint(
        {
            key: value
            for key, value in provisional.items()
            if key != "semantic_sha256"
        }
    )
    atomic_write_json(metadata_path, provisional)
    assert cli._load_valid_confirmation_shard(
        run_dir,
        cohort=cohort,
        start_step=0,
        current=state,
        selected_step=None,
        selection_sha256="a" * 64,
    ) is None
