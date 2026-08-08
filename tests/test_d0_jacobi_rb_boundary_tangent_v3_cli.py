from __future__ import annotations

import argparse
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mnist import diag_d0_jacobi_rb_boundary_tangent_v3_learnability as cli
from mnist.d0_jacobi_artifacts import atomic_write_json, config_fingerprint


def _args(tmp_path: Path, *, stage: str = "preflight") -> argparse.Namespace:
    return argparse.Namespace(
        stage=stage,
        require_gate="none",
        failed_v3_preflight_run_dir=(tmp_path / "failed-v3-preflight").resolve(),
        parent_v2_run_dir=(tmp_path / "v2-parent").resolve(),
        adjudication_run_dir=(tmp_path / "adjudication-parent").resolve(),
        parent_eager_pipeline_run_dir=(tmp_path / "eager-parent").resolve(),
        parent_coarse_residual_run_dir=(tmp_path / "coarse-parent").resolve(),
        resume_run_dir=None,
        runs_root=(tmp_path / "runs").resolve(),
        run_name="fixture-zero-baseline-v3",
        device="cpu",
        test_only=True,
        test_path_count=2,
        test_outer_steps=16,
        test_maximum_updates=0,
        test_bootstrap_replicates=8,
    )


def _parent_args(tmp_path: Path) -> list[str]:
    return [
        "--failed-v3-preflight-run-dir",
        str(tmp_path / "failed-v3-preflight"),
        "--parent-v2-run-dir",
        str(tmp_path / "v2-parent"),
        "--adjudication-run-dir",
        str(tmp_path / "adjudication-parent"),
        "--parent-eager-pipeline-run-dir",
        str(tmp_path / "eager-parent"),
        "--parent-coarse-residual-run-dir",
        str(tmp_path / "coarse-parent"),
    ]


def _passing_gate(stage: str) -> dict[str, object]:
    return {
        "schema": f"fixture-{stage}-gate",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "gate": stage,
        "passed": 1,
        "scientific_evidence_complete": 1,
        **cli.NO_WORK,
    }


def _write_mock_initialization(run_dir: Path, args: argparse.Namespace) -> None:
    path_plan = cli.build_v3_path_plan(test_only=args.test_only)
    cohort_plan = cli.build_v3_cohort_plan(path_plan, test_only=args.test_only)
    atomic_write_json(
        run_dir / "scientific_config.json",
        cli._scientific_config(args, path_plan, cohort_plan),
    )
    atomic_write_json(run_dir / "path_id_plan.json", path_plan)
    atomic_write_json(run_dir / "cohort_plan.json", cohort_plan)


def test_parser_and_stage_sequence_are_frozen(tmp_path: Path) -> None:
    parents = _parent_args(tmp_path)
    report = cli.parse_args(
        parents
        + [
            "--stage",
            "report",
            "--resume-run-dir",
            str(tmp_path / "existing-run"),
        ]
    )
    assert report.stage == "report"
    assert report.require_gate == "none"
    assert report.failed_v3_preflight_run_dir.is_absolute()
    assert report.parent_v2_run_dir.is_absolute()
    assert report.adjudication_run_dir.is_absolute()
    assert report.parent_eager_pipeline_run_dir.is_absolute()
    assert report.parent_coarse_residual_run_dir.is_absolute()

    assert cli.STAGES == (
        "preflight",
        "cache",
        "train",
        "select",
        "confirm",
        "report",
        "all",
    )
    assert cli.REQUIRED_GATES == (
        "none",
        "preflight",
        "cache",
        "train",
        "select",
        "confirm",
    )
    assert cli._stage_sequence("all") == (
        "preflight",
        "cache",
        "train",
        "select",
        "confirm",
    )
    assert cli._stage_sequence("report") == ()
    with pytest.raises(ValueError):
        cli._stage_sequence("control")

    # Every mutating production stage after preflight resumes the same run.
    with pytest.raises(SystemExit):
        cli.parse_args(parents + ["--stage", "cache", "--device", "cuda"])


def test_test_only_is_nonauthorizing_and_cannot_require_a_gate(
    tmp_path: Path,
) -> None:
    parents = _parent_args(tmp_path)
    with pytest.raises(SystemExit):
        cli.parse_args(
            parents
            + [
                "--test-only",
                "--stage",
                "preflight",
                "--require-gate",
                "preflight",
            ]
        )

    args = cli.parse_args(
        parents
        + [
            "--test-only",
            "--stage",
            "all",
            "--device",
            "cpu",
            "--test-path-count",
            "2",
            "--test-outer-steps",
            "16",
            "--test-maximum-updates",
            "0",
            "--test-bootstrap-replicates",
            "8",
        ]
    )
    assert args.test_only
    assert args.require_gate == "none"
    plan = cli.build_v3_path_plan(test_only=True, test_path_count=2)
    production = set(range(0xF0000, 0x100000))
    realized = {
        int(path_id)
        for role in plan["roles"].values()
        for path_id in role
    }
    assert not realized.intersection(production)
    config = cli._scientific_config(
        args,
        plan,
        cli.build_v3_cohort_plan(plan, test_only=True),
    )
    assert config["authorizing"] == 0
    assert config["test_only"] == 1


def test_production_scientific_config_freezes_paths_cohorts_and_seeds(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.test_only = False
    path_plan = cli.build_v3_path_plan()
    cohort_plan = cli.build_v3_cohort_plan(path_plan)
    record = cli._scientific_config(args, path_plan, cohort_plan)

    assert path_plan["roles"]["preflight_seam"] == list(
        range(0xF0000, 0xF0008)
    )
    assert path_plan["roles"]["train"] == list(range(0xF1000, 0xF1040))
    assert path_plan["roles"]["validation"] == list(range(0xF1100, 0xF1120))
    assert path_plan["roles"]["confirmation"] == list(
        range(0xF2000, 0xF2040)
    )
    assert [row["size"] for row in cohort_plan["train_validation"]] == [
        10
    ] * 9 + [6]
    assert [row["size"] for row in cohort_plan["confirmation"]] == [10] * 6 + [
        4
    ]

    assert record["root_seed"] == 261_311
    assert record["model_seeds"] == [261_312, 261_313, 261_314]
    assert record["selection_bootstrap_seed"] == 261_320
    assert record["confirmation_bootstrap_seed"] == 261_322
    assert record["synthetic_teacher_seed"] == 261_323
    assert record["exact_model_null_seed"] == 261_324
    assert record["reserved_future_control_seed"] == 261_325
    assert record["forbidden_scheduler_benchmark_seed"] == 261_321
    assert record["selection_namespace"] == 0x42545633
    assert record["confirmation_namespace"] == 0x42544333
    assert record["candidate_count"] == 120
    assert record["component_count"] == 228
    assert record["joint_family_size"] == 27_360
    assert record["checkpoint_updates"] == list(range(0, 4_001, 100))
    assert record["physical_training_uses_validation_labels"] == 0
    assert record["confirmation_paths_created"] == 0
    assert record["certificate_semantics_comparator_version"] == (
        cli.CERTIFICATE_SEMANTICS_COMPARATOR_VERSION
    )
    assert record["failed_v3_preflight_registry_count"] == 19
    assert record["failed_v3_preflight_registry_semantic_sha256"] == (
        "ed60b3d4130883b39e940ccb3d78f8110ceec8c979e111c21d8b43dfa21ccd3b"
    )
    assert record["failed_v3_preflight_registry_file_sha256"] == (
        "94fcd4443fe2e45ff148fab9954b372bd3d3d7cf5c5efddb6dce132c8018692d"
    )
    assert record["controller_control_trajectory_performed"] == 0
    assert record["reverse_sampling_performed"] == 0
    assert record["sampling_performed"] == 0
    body = dict(record)
    assert body.pop("semantic_sha256") == config_fingerprint(body)


def test_certificate_semantics_contract_separates_authority_from_proof() -> None:
    record = cli._certificate_semantics_contract()
    assert record["comparator_version"] == (
        cli.CERTIFICATE_SEMANTICS_COMPARATOR_VERSION
    )
    assert record["scientific_payload_equality_required"] == 1
    assert record["certificate_semantics_equality_required"] == 1
    assert record["proof_metadata_equality_required"] == 0
    assert "batch_certificate_sha256" in record["proof_metadata_advisory"]
    body = dict(record)
    assert body.pop("semantic_sha256") == config_fingerprint(body)


def test_test_only_preflight_writes_schema_v2_certificate_artifacts(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    run_dir = tmp_path / "fresh"
    run_dir.mkdir()
    cli._initialize(run_dir, args, resumed=False)
    gate = cli._preflight_stage(run_dir, args)

    assert gate["passed"] == 1
    seam = cli._load_json(run_dir / "preflight_scheduler_seam.json")
    semantics = cli._load_json(run_dir / "preflight_certificate_semantics.json")
    metrics = cli._load_json(run_dir / "preflight_metrics.json")
    assert seam["schema_version"] == 2
    assert seam["scientific_payload_equal"] == 1
    assert seam["certificate_semantics_equal"] == 1
    assert seam["proof_metadata_equal_advisory"] == 0
    assert seam["batch_certificate_sha256_equality_required"] == 0
    assert semantics["comparator_evaluation_valid"] == 1
    assert semantics["scientific_payload_equal"] == 1
    assert semantics["certificate_semantics_equal"] == 1
    assert semantics["proof_metadata_equal_advisory"] == 0
    assert metrics["certificate_semantics_comparator_valid"] == 1
    assert metrics["scheduler_seam_valid"] == 1
    assert (run_dir / "preflight_certificate_proof_metadata.csv").is_file()
    seal = cli._load_json(run_dir / "preflight_artifact_seal.json")
    sealed = {row["path"] for row in seal["artifacts"]}
    assert {
        "certificate_semantics_contract.json",
        "failed_v3_preflight_adjudication.json",
        "preflight_certificate_semantics.json",
        "preflight_certificate_proof_metadata.csv",
        "preflight_scheduler_seam.json",
    }.issubset(sealed)
    registry = cli._artifact_registry(run_dir)
    registered = {row["path"] for row in registry["artifacts"]}
    assert "preflight_certificate_semantics.json" in registered
    assert "preflight_certificate_proof_metadata.csv" in registered


def test_proof_metadata_csv_rows_include_advisory_hashes_and_timings() -> None:
    shard = lambda hash_value, elapsed: SimpleNamespace(  # noqa: E731
        identity=SimpleNamespace(start_step=0),
        base_record={"batch_certificate_sha256": hash_value},
        diagnostics={
            "complete_pipeline_elapsed_seconds": elapsed,
            "fallback_elapsed_seconds": 0.0,
            "backend_elapsed_seconds": elapsed / 2.0,
            "candidate_elapsed_seconds": elapsed / 4.0,
            "transitions_per_second": 1_500.0,
            "peak_memory_fraction": 0.1,
        },
    )
    comparison = {
        "left_proof_metadata": {},
        "right_proof_metadata": {},
        "differing_proof_metadata_fields": [],
    }
    rows = cli._proof_metadata_rows(
        comparison,
        (shard("a" * 64, 1.0),),
        (shard("b" * 64, 0.5),),
    )

    commitment = next(
        row
        for row in rows
        if row["record_kind"] == "base_shard_certificate_commitment"
    )
    assert commitment["equality_required"] == 0
    assert commitment["equality_advisory"] == 0
    timing = {
        row["field"]: row
        for row in rows
        if row["record_kind"] == "shard_proof_timing"
    }
    assert timing["complete_pipeline_elapsed_seconds"]["adaptive_value"] == 1.0
    assert timing["complete_pipeline_elapsed_seconds"]["eager_value"] == 0.5
    assert all(row["equality_required"] == 0 for row in timing.values())


def test_malformed_certificate_semantics_record_is_comparator_failure() -> None:
    malformed = {
        "schema": cli.RUN_SCHEMA + "-preflight-certificate-semantics",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "comparator_version": cli.CERTIFICATE_SEMANTICS_COMPARATOR_VERSION,
        "comparison_executed": 1,
    }
    with pytest.raises(cli.CertificateSemanticsError):
        cli._certificate_semantics_record_valid(malformed)

    metrics: dict[str, Any] = {}
    # Gate classification is deliberately isolated from the scientific seam:
    # a malformed comparator record is an implementation-contract failure.
    from mnist.d0_jacobi_rb_boundary_tangent_v3_gate import (
        PREFLIGHT_BASELINE_FLAGS,
        PREFLIGHT_COMPARATOR_FLAGS,
        PREFLIGHT_EXECUTION_FLAGS,
        PREFLIGHT_PROVENANCE_FLAGS,
    )

    metrics.update(
        {
            name: 1
            for name in (
                PREFLIGHT_PROVENANCE_FLAGS
                + PREFLIGHT_BASELINE_FLAGS
                + PREFLIGHT_COMPARATOR_FLAGS
                + PREFLIGHT_EXECUTION_FLAGS
            )
        }
    )
    metrics.update(
        {
            "certificate_semantics_comparator_valid": 0,
            "preflight_path_ids": list(range(0xF0000, 0xF0008)),
            "preflight_path_count": 8,
            "certificate_fraction": 1.0,
            "maximum_mass_error": 0.0,
            "forbidden_event_count": 0,
            "transitions_per_second": 1_300.0,
            "peak_memory_fraction": 0.0,
        }
    )
    gate = cli.evaluate_preflight_gate(metrics)
    decision = cli.decide_workflow(
        preflight_gate=gate,
        cache_gate=None,
        train_gate=None,
        select_gate=None,
        confirm_gate=None,
    )
    assert gate["failure_domain"] == "implementation_contract"
    assert decision["decision"] == "certificate_semantics_comparator_invalid"


def test_malformed_comparator_stage_writes_artifacts_before_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    run_dir = tmp_path / "malformed-stage"
    run_dir.mkdir()
    cli._initialize(run_dir, args, resumed=False)
    original = cli._preflight_seam

    def malformed_seam(*call_args: object, **call_kwargs: object):
        seam, semantics, rows = original(*call_args, **call_kwargs)
        body = dict(semantics)
        body.pop("semantic_sha256")
        body.pop("left_authorization_counts")
        return seam, cli._semantic(body), rows

    monkeypatch.setattr(cli, "_preflight_seam", malformed_seam)
    stage_gate = cli._preflight_stage(run_dir, args)
    gate = cli.evaluate_preflight_gate(
        cli._load_json(run_dir / "preflight_metrics.json")
    )
    decision = cli.decide_workflow(
        preflight_gate=gate,
        cache_gate=None,
        train_gate=None,
        select_gate=None,
        confirm_gate=None,
    )

    assert stage_gate["passed"] == 0
    assert gate["passed"] == 0
    assert gate["failure_domain"] == "implementation_contract"
    assert decision["decision"] == "certificate_semantics_comparator_invalid"
    for name in (
        "preflight_certificate_semantics.json",
        "preflight_certificate_proof_metadata.csv",
        "preflight_scheduler_seam.json",
        "preflight_gate.json",
        "preflight_artifact_seal.json",
    ):
        assert (run_dir / name).is_file()


def test_failed_v3_run_resume_is_rejected_before_mutation(tmp_path: Path) -> None:
    failed = (
        Path("runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_learnability")
        / "20260805-170727_production-zero-baseline-v3-learnability"
    ).resolve()
    if not failed.is_dir():
        pytest.skip("immutable production failed-v3 fixture is unavailable")
    before = {
        path.relative_to(failed).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in failed.rglob("*")
        if path.is_file()
    }
    args = _args(tmp_path)
    args.resume_run_dir = failed
    args.failed_v3_preflight_run_dir = failed
    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._initialize(failed, args, resumed=True)
    after = {
        path.relative_to(failed).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in failed.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_physical_candidate_generator_has_train_only_signature() -> None:
    parameters = inspect.signature(
        cli._run_physical_candidate_generation
    ).parameters
    assert tuple(parameters) == (
        "run_dir",
        "train_inputs",
        "train_targets",
        "train_index",
        "target_scale",
        "seed",
        "update_config",
        "device",
    )
    assert "validation_inputs" not in parameters
    assert "validation_targets" not in parameters
    assert "validation_index" not in parameters


def test_fixed_candidate_grid_has_no_selection_or_validation_fields() -> None:
    grid = cli._fixed_candidate_grid()
    assert len(grid) == 123
    assert [row["seed"] for row in grid[::41]] == [261_312, 261_313, 261_314]
    assert sum(int(row["update"] > 0) for row in grid) == 120
    assert [(row["seed"], row["update"]) for row in grid] == [
        (seed, update)
        for seed in (261_312, 261_313, 261_314)
        for update in range(0, 4_001, 100)
    ]
    forbidden = {
        "selected",
        "eligible",
        "eligibility",
        "validation_mse",
        "validation_metric",
        "pointwise_eligibility",
    }
    assert all(forbidden.isdisjoint(row) for row in grid)


def test_registry_allows_only_a_known_uncommitted_resume_tail(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(run_dir / "preflight_metrics.json", {"passed": 1})
    cli._artifact_registry(run_dir)
    tail = run_dir / "eager_cache" / "train_validation" / "shard.npz"
    tail.parent.mkdir(parents=True)
    np.savez(tail, value=np.asarray([1], dtype=np.int64))

    cli._verify_existing_registry(
        run_dir, allow_unregistered_incomplete_tail=True
    )
    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._verify_existing_registry(run_dir)

    (run_dir / "unexpected.bin").write_bytes(b"tamper")
    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._verify_existing_registry(
            run_dir, allow_unregistered_incomplete_tail=True
        )


def test_train_stage_opens_only_train_labels_after_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, stage="train")
    args.test_only = False
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(run_dir / "cache_gate.json", _passing_gate("cache"))
    events: list[str] = []

    training_inputs: dict[str, Any] = {
        "train_inputs": object(),
        "validation_inputs": object(),
        "train_arrays": {"sample_key": np.asarray([1], dtype=np.int64)},
        "validation_arrays": {"sample_key": np.asarray([2], dtype=np.int64)},
        "train_index": {"semantic_sha256": "train-index"},
        "validation_index": {"semantic_sha256": "validation-index"},
    }
    monkeypatch.setattr(
        cli,
        "_load_training_inputs",
        lambda *_args, **_kwargs: events.append("inputs") or training_inputs,
    )

    def controls(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("controls")
        return {
            "passed": 1,
            "synthetic_metrics": {"passed": 1},
            "null_metrics": {"passed": 1, "selected_update": 0},
        }

    monkeypatch.setattr(cli, "_run_prelabel_controls", controls)

    def load_train_labels(*_args: object, **_kwargs: object) -> np.ndarray:
        assert events == ["inputs", "controls"]
        assert (run_dir / "training_label_open.json").is_file()
        assert (run_dir / "physical_training_started.json").is_file()
        events.append("train-labels")
        return np.asarray([1.0], dtype=np.float64)

    monkeypatch.setattr(cli, "_load_physical_train_labels", load_train_labels)
    # These symbols deliberately explode if the physical candidate generator
    # ever attempts to cross the validation firewall.
    monkeypatch.setattr(
        cli,
        "_load_physical_validation_labels",
        lambda *_args, **_kwargs: pytest.fail("train opened validation labels"),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "_load_physical_validation_inputs",
        lambda *_args, **_kwargs: pytest.fail("train opened validation inputs"),
        raising=False,
    )

    def candidate_generator(
        root: Path,
        *,
        train_inputs: object,
        train_targets: object,
        train_index: object,
        target_scale: float,
        seed: int,
        update_config: object,
        device: object,
    ) -> dict[str, object]:
        del root, train_targets, update_config, device
        assert train_inputs is training_inputs["train_inputs"]
        assert train_index is training_inputs["train_index"]
        assert target_scale > 0.0
        events.append(f"seed-{seed}")
        return {
            "seed": seed,
            "complete": 1,
            "finite": 1,
            "checkpoint_count": 41,
            "nonzero_candidate_count": 40,
        }

    monkeypatch.setattr(
        cli, "_run_physical_candidate_generation", candidate_generator
    )
    gate = cli._train_stage(run_dir, args)
    assert gate["passed"] == 1
    assert events == [
        "inputs",
        "controls",
        "train-labels",
        "seed-261312",
        "seed-261313",
        "seed-261314",
    ]
    assert not (run_dir / "validation_label_open.json").exists()
    candidate_grid = cli._load_json(run_dir / "candidate_grid.json")
    assert candidate_grid["candidate_count"] == 120
    assert candidate_grid["checkpoint_count"] == 123
    assert "selected_seed" not in candidate_grid
    assert "selected_update" not in candidate_grid


def test_failed_controls_leave_all_physical_labels_unopened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, stage="train")
    args.test_only = False
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(run_dir / "cache_gate.json", _passing_gate("cache"))
    monkeypatch.setattr(
        cli,
        "_load_training_inputs",
        lambda *_args, **_kwargs: {
            "train_inputs": object(),
            "validation_inputs": object(),
            "train_index": {},
            "validation_index": {},
        },
    )
    monkeypatch.setattr(
        cli,
        "_run_prelabel_controls",
        lambda *_args, **_kwargs: {
            "passed": 0,
            "synthetic_metrics": {"passed": 0},
            "null_metrics": {"passed": 0},
        },
    )
    monkeypatch.setattr(
        cli,
        "_load_physical_train_labels",
        lambda *_args, **_kwargs: pytest.fail("failed controls opened labels"),
    )
    monkeypatch.setattr(
        cli,
        "_run_physical_candidate_generation",
        lambda *_args, **_kwargs: pytest.fail("failed controls trained"),
    )
    gate = cli._train_stage(run_dir, args)
    assert gate["passed"] == 0
    assert not (run_dir / "training_label_open.json").exists()
    assert not (run_dir / "physical_training_started.json").exists()
    assert not (run_dir / "candidate_grid.json").exists()


def test_no_validation_candidate_stops_all_before_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, stage="all")
    args.test_only = False
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_mock_initialization(run_dir, args)
    calls: list[str] = []
    monkeypatch.setattr(cli, "_make_run_dir", lambda _: (run_dir, False))
    monkeypatch.setattr(cli, "_initialize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda *_: {})

    def passing_stage(name: str):
        def execute(root: Path, _args: argparse.Namespace) -> dict[str, object]:
            calls.append(name)
            gate = _passing_gate(name)
            atomic_write_json(root / f"{name}_gate.json", gate)
            return gate

        return execute

    monkeypatch.setattr(cli, "_preflight_stage", passing_stage("preflight"))
    monkeypatch.setattr(cli, "_cache_stage", passing_stage("cache"))
    monkeypatch.setattr(cli, "_train_stage", passing_stage("train"))

    def no_candidate(
        root: Path, _args: argparse.Namespace
    ) -> dict[str, object]:
        calls.append("select")
        atomic_write_json(
            root / "validation_selection.json",
            {
                "selected_seed": None,
                "selected_update": 0,
                "selection_kind": "logical_update_zero_null",
            },
        )
        atomic_write_json(root / "no_validation_candidate.json", {"sealed": 1})
        gate = {
            **_passing_gate("select"),
            "passed": 0,
            "no_validation_candidate": 1,
        }
        atomic_write_json(root / "select_gate.json", gate)
        return gate

    monkeypatch.setattr(cli, "_select_stage", no_candidate)
    monkeypatch.setattr(
        cli,
        "_confirm_stage",
        lambda *_args, **_kwargs: pytest.fail("no-candidate run opened confirmation"),
    )
    assert cli._run(args) == 2
    assert calls == ["preflight", "cache", "train", "select"]
    assert not (run_dir / "confirmation_namespace_open.json").exists()
    assert not (run_dir / "confirmation").exists()
    decision = cli._load_json(run_dir / "boundary_tangent_v3_decision.json")
    assert decision["decision"] == "no_validation_candidate"
    assert (run_dir / "run_status.json").is_file()
    assert (run_dir / "artifact_registry.json").is_file()


def test_confirmation_counts_are_sealed_before_namespace_and_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, stage="confirm")
    args.test_only = False
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path_plan = cli.build_v3_path_plan()
    atomic_write_json(run_dir / "path_id_plan.json", path_plan)
    selection = {
        "selected_seed": 261_312,
        "selected_update": 100,
        "checkpoint_file_sha256": "a" * 64,
        "checkpoint_state_sha256": "b" * 64,
        "selection_role": "fresh_validation_search_aware",
        "confirmation_paths_created": 0,
    }
    atomic_write_json(run_dir / "checkpoint_selection.json", selection)
    atomic_write_json(run_dir / "select_gate.json", _passing_gate("select"))
    cli._seal_stage(
        run_dir,
        ["checkpoint_selection.json", "select_gate.json"],
        "selection_artifact_seal.json",
    )
    events: list[str] = []

    def counts(root: Path, **_kwargs: object) -> dict[str, object]:
        assert not (root / "confirmation_namespace_open.json").exists()
        events.append("counts")
        return {"complete": 1, "shard_count": 50}

    def execute(root: Path, *_args: object, **_kwargs: object) -> dict[str, object]:
        opened = cli._load_json(root / "confirmation_namespace_open.json")
        assert opened["path_ids"] == list(range(0xF2000, 0xF2040))
        assert opened["bootstrap_seed"] == 261_322
        assert opened["bootstrap_namespace"] == 0x42544333
        assert events == ["counts"]
        events.append("execute")
        return {"complete": 1, "path_count": 64}

    monkeypatch.setattr(cli, "_prepare_confirmation_count_shards", counts)
    monkeypatch.setattr(cli, "_run_confirmation_execution", execute)
    monkeypatch.setattr(
        cli,
        "_finalize_confirmation_inference",
        lambda root, *_args, **_kwargs: atomic_write_json(
            root / "confirmation_gate.json", _passing_gate("confirm")
        )
        or _passing_gate("confirm"),
    )
    gate = cli._confirm_stage(run_dir, args)
    assert gate["passed"] == 1
    assert events == ["counts", "execute"]
    before = (run_dir / "confirmation_namespace_open.json").read_bytes()

    # Resume may reuse exactly the burned namespace but cannot replace it.
    events.clear()
    gate = cli._confirm_stage(run_dir, args)
    assert gate["passed"] == 1
    assert (run_dir / "confirmation_namespace_open.json").read_bytes() == before
    assert events in ([], ["counts", "execute"])


def test_execution_failure_commits_gate_decision_status_and_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, stage="cache")
    args.resume_run_dir = tmp_path / "run"
    args.resume_run_dir.mkdir()
    run_dir = args.resume_run_dir
    _write_mock_initialization(run_dir, args)
    atomic_write_json(run_dir / "preflight_gate.json", _passing_gate("preflight"))
    monkeypatch.setattr(cli, "_make_run_dir", lambda _: (run_dir, True))
    monkeypatch.setattr(cli, "_initialize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda *_: {})

    def explode(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise cli.BoundaryTangentV3CLIError(
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
    assert gate["scientific_evidence_complete"] == 0
    assert (run_dir / "cache_artifact_seal.json").is_file()
    assert (run_dir / "boundary_tangent_v3_decision.json").is_file()
    assert cli._load_json(run_dir / "run_status.json")["state"] == (
        "execution_failed"
    )
    registry = cli._load_json(run_dir / "artifact_registry.json")
    registered = {row["path"] for row in registry["artifacts"]}
    assert "cache_execution_failure.json" in registered
    assert registry["controller_control_trajectory_performed"] == 0
    assert registry["reverse_sampling_performed"] == 0
    assert registry["sampling_performed"] == 0


def test_successful_all_stage_runs_every_stage_once_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, stage="all")
    args.test_only = False
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

    for name in ("preflight", "cache", "train", "select", "confirm"):
        monkeypatch.setattr(cli, f"_{name}_stage", stage(name))
    assert cli._run(args) == 0
    assert calls == ["preflight", "cache", "train", "select", "confirm"]
    decision = cli._load_json(run_dir / "boundary_tangent_v3_decision.json")
    assert decision["decision"] == (
        "exact_rb_zero_baseline_boundary_tangent_time_local_signal_confirmed"
    )
    status = cli._load_json(run_dir / "run_status.json")
    assert status["state"] in {"complete", "test_only_complete"}
    registry = cli._load_json(run_dir / "artifact_registry.json")
    assert registry["controller_control_trajectory_performed"] == 0
    assert registry["reconstruction_performed"] == 0
    assert registry["reverse_sampling_performed"] == 0
    assert registry["sampling_performed"] == 0
