from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pytest

from mnist import (
    diag_d0_jacobi_rb_boundary_tangent_quartile_specialist as cli,
)
from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_json,
    file_fingerprint,
)
from mnist.d0_jacobi_rb_boundary_tangent_cache import midpoint_sample_key
from mnist.d0_jacobi_rb_boundary_tangent_quartile_selection import (
    aggregate_quartile_audit_improvements,
    prepare_bootstrap_count_shards,
    restartable_quartile_max_t,
)


def _args(tmp_path: Path, *, stage: str = "preflight") -> argparse.Namespace:
    return argparse.Namespace(
        stage=stage,
        require_gate="none",
        parent_time_local_run_dir=(tmp_path / "time-local").resolve(),
        parent_memory_v3_run_dir=(tmp_path / "memory-v3").resolve(),
        parent_coarse_witness_run_dir=(tmp_path / "coarse-witness").resolve(),
        parent_bayes_power_run_dir=(tmp_path / "bayes-power").resolve(),
        resume_run_dir=None,
        runs_root=(tmp_path / "runs").resolve(),
        run_name="fixture-quartile-specialist",
        device="cpu",
        test_only=True,
        test_path_count=2,
        test_outer_steps=16,
        test_maximum_updates=0,
        test_bootstrap_replicates=8,
        test_bootstrap_shard_size=4,
    )


def _parent_args(tmp_path: Path) -> list[str]:
    return [
        "--parent-time-local-run-dir",
        str(tmp_path / "time-local"),
        "--parent-memory-v3-run-dir",
        str(tmp_path / "memory-v3"),
        "--parent-coarse-witness-run-dir",
        str(tmp_path / "coarse-witness"),
        "--parent-bayes-power-run-dir",
        str(tmp_path / "bayes-power"),
    ]


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
    assert report.parent_time_local_run_dir.is_absolute()
    assert report.parent_memory_v3_run_dir.is_absolute()
    assert report.parent_coarse_witness_run_dir.is_absolute()
    assert report.parent_bayes_power_run_dir.is_absolute()

    assert cli.STAGES == (
        "preflight",
        "cache",
        "controls",
        "train",
        "calibrate",
        "select",
        "confirm",
        "report",
        "all",
    )
    assert cli.REQUIRED_GATES == (
        "none",
        "preflight",
        "cache",
        "controls",
        "train",
        "calibrate",
        "select",
        "confirm",
    )
    assert cli._stage_sequence("all") == (
        "preflight",
        "cache",
        "controls",
        "train",
        "calibrate",
        "select",
        "confirm",
    )
    assert cli._stage_sequence("report") == ()
    with pytest.raises(ValueError):
        cli._stage_sequence("sample")

    # Every mutating stage after preflight must resume the same sealed run.
    with pytest.raises(SystemExit):
        cli.parse_args(parents + ["--stage", "cache"])


def test_test_only_overrides_are_nonauthorizing(tmp_path: Path) -> None:
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
            "--test-bootstrap-shard-size",
            "4",
        ]
    )
    assert args.test_only
    assert args.require_gate == "none"
    config = cli._scientific_config(args)
    assert config["test_only"] == 1
    assert config["authorizing"] == 0


def test_effective_production_and_reduced_plans_are_exact(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.test_only = False
    assert cli._effective_counts(args) == {
        "preflight_seam": 8,
        "physical_fit": 64,
        "gain_calibration": 32,
        "training_rank": 32,
        "fresh_selection": 384,
        "untouched_confirmation": 384,
    }
    assert cli._effective_paths(args, "physical_fit") == tuple(
        range(0xF4000, 0xF4040)
    )
    assert cli._effective_paths(args, "gain_calibration") == tuple(
        range(0xF4100, 0xF4120)
    )
    assert cli._effective_paths(args, "training_rank") == tuple(
        range(0xF4200, 0xF4220)
    )
    assert cli._effective_paths(args, "fresh_selection") == tuple(
        range(0xF5000, 0xF5180)
    )
    assert cli._effective_paths(args, "untouched_confirmation") == tuple(
        range(0xF7000, 0xF7180)
    )
    assert cli._effective_outer_steps(args) == 512
    assert cli._effective_updates(args) == 4_000
    assert cli._effective_bootstrap(args) == (50_000, 1_000)

    args.test_only = True
    assert cli._effective_counts(args) == {
        "preflight_seam": 2,
        "physical_fit": 2,
        "gain_calibration": 2,
        "training_rank": 2,
        "fresh_selection": 8,
        "untouched_confirmation": 8,
    }
    assert cli._effective_outer_steps(args) == 16
    assert cli._effective_selected_steps(args) == (15,)
    assert cli._effective_updates(args) == 0
    assert cli._effective_bootstrap(args) == (8, 4)


def test_initialization_freezes_plans_without_opening_evidence(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cli._initialize(run_dir, args, resumed=False)

    required = {
        "run_manifest.json",
        "scientific_config.json",
        "path_id_plan.json",
        "cohort_plan.json",
        "seed_plan.json",
        "role_firewall.json",
        "candidate_grid_plan.json",
        "gain_calibration_plan.json",
        "training_rank_plan.json",
        "selection_inference_plan.json",
        "confirmation_inference_plan.json",
        "run_status.json",
    }
    assert required.issubset({path.name for path in run_dir.iterdir()})
    assert not (run_dir / "fit_label_open.json").exists()
    assert not (run_dir / "selection_open.json").exists()
    assert not (run_dir / "confirmation_open.json").exists()

    path_plan = cli._load_json(run_dir / "path_id_plan.json")
    assert path_plan["roles"]["physical_fit"] == list(
        range(0xF4000, 0xF4040)
    )
    selection = cli._load_json(run_dir / "selection_inference_plan.json")
    confirmation = cli._load_json(run_dir / "confirmation_inference_plan.json")
    assert selection["path_count"] == confirmation["path_count"] == 384
    assert selection["family_names"] == confirmation["family_names"]
    assert selection["namespace"] != confirmation["namespace"]


def test_reduced_cpu_preflight_commits_counts_and_gate_without_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cli._initialize(run_dir, args, resumed=False)

    monkeypatch.setattr(
        cli,
        "verify_quartile_specialist_parents",
        lambda **_kwargs: {"passed": 1, "parents_mutated": 0},
    )
    monkeypatch.setattr(cli, "quartile_source_paths", lambda: ())
    monkeypatch.setattr(
        cli,
        "_source_target",
        lambda _parent: np.full(784, 1.0 / 784.0, dtype=np.float64),
    )
    monkeypatch.setattr(
        cli,
        "_generate_role_cache",
        lambda *_args, **_kwargs: {
            "result": {
                "metrics": {
                    "semantic_sha256": "fixture",
                    "transition_count": 128,
                    "certified_count": 128,
                    "forbidden_event_count": 0,
                    "complete_pipeline_elapsed_seconds": 0.25,
                    "maximum_peak_memory_fraction": 0.01,
                }
            },
            "cohort_plan": {"semantic_sha256": "fixture"},
        },
    )
    monkeypatch.setattr(
        cli,
        "load_eager_role_inputs",
        lambda *_args, **_kwargs: (
            {"sample_key": np.arange(8, dtype=np.int64)},
            {"semantic_sha256": "fixture"},
        ),
    )

    gate = cli._preflight_stage(run_dir, args)
    assert gate["evaluation_status"] == "evaluated"
    assert gate["passed"] == 1
    assert (run_dir / "bootstrap_counts/selection").is_dir()
    assert (run_dir / "bootstrap_counts/confirmation").is_dir()
    assert cli._load_json(run_dir / "selection_bootstrap_count_index.json")[
        "path_count"
    ] == 8
    assert cli._load_json(run_dir / "confirmation_bootstrap_count_index.json")[
        "path_count"
    ] == 8
    cli._verify_stage_seal(run_dir, "preflight_artifact_seal.json")


def test_artifact_registry_records_scope_and_opening_state(tmp_path: Path) -> None:
    atomic_write_json(tmp_path / "evidence.json", {"value": 1})
    registry = cli._artifact_registry(tmp_path)
    assert registry["physical_training_performed"] == 0
    assert registry["selection_paths_opened"] == 0
    assert registry["confirmation_paths_opened"] == 0
    assert registry["controller_execution_performed"] == 0
    assert registry["reconstruction_performed"] == 0
    assert registry["sampling_performed"] == 0
    assert [row["path"] for row in registry["artifacts"]] == ["evidence.json"]


def test_invalid_test_bootstrap_partition_fails_closed(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.test_bootstrap_replicates = 10
    args.test_bootstrap_shard_size = 4
    with pytest.raises(ArtifactCompatibilityError, match="must divide"):
        cli._effective_bootstrap(args)


def _write_minimal_training_checkpoint_index(
    run_dir: Path, args: argparse.Namespace
) -> Path:
    """Commit the reduced update-zero grid used by integrity regressions."""

    checkpoint_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    first_checkpoint: Path | None = None
    for quartile, seeds in enumerate(cli.MODEL_SEEDS_BY_QUARTILE):
        for seed in seeds:
            task_root = run_dir / "checkpoints" / f"q{quartile}"
            checkpoint_root = task_root / f"seed-{seed}"
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            checkpoint = checkpoint_root / "update-0000.pt"
            checkpoint.write_bytes(f"checkpoint-q{quartile}-seed-{seed}".encode())
            first_checkpoint = first_checkpoint or checkpoint
            candidate = cli.CandidateIdentity(quartile, seed, 0)
            checkpoint_rows.append(
                {
                    "candidate_key": candidate.key,
                    "checkpoint_path": checkpoint.relative_to(run_dir).as_posix(),
                    "checkpoint_file_sha256": file_fingerprint(checkpoint),
                    "model_state_sha256": f"{quartile:x}" * 64,
                }
            )

            task = task_root / f"seed-{seed}-task.json"
            history = task_root / f"seed-{seed}-history.csv"
            progress = task_root / f"seed-{seed}-progress.pt"
            task.write_bytes(f"task-q{quartile}-seed-{seed}".encode())
            history.write_bytes(f"history-q{quartile}-seed-{seed}".encode())
            progress.write_bytes(f"progress-q{quartile}-seed-{seed}".encode())
            task_rows.append(
                {
                    "quartile": quartile,
                    "seed": seed,
                    "task_path": task.relative_to(run_dir).as_posix(),
                    "task_sha256": file_fingerprint(task),
                    "history_path": history.relative_to(run_dir).as_posix(),
                    "history_sha256": file_fingerprint(history),
                    "progress_path": progress.relative_to(run_dir).as_posix(),
                    "progress_sha256": file_fingerprint(progress),
                }
            )
    assert first_checkpoint is not None
    index = cli._semantic(
        {
            "schema": cli.RUN_SCHEMA + "-training-checkpoint-index",
            "schema_version": 1,
            "task_count": len(task_rows),
            "checkpoint_count": len(checkpoint_rows),
            "tasks": task_rows,
            "checkpoints": checkpoint_rows,
            "all_boundary_checkpoints_exactly_resumable": 1,
        }
    )
    atomic_write_json(run_dir / "training_checkpoint_index.json", index)
    cli._seal_stage(
        run_dir,
        ("training_checkpoint_index.json",),
        "train_artifact_seal.json",
    )
    atomic_write_json(
        run_dir / "train_gate.json",
        {"evaluation_status": "evaluated", "passed": 1},
    )
    assert cli._effective_updates(args) == 0
    return first_checkpoint


def test_existing_role_open_rejects_changed_prerequisite_hash(tmp_path: Path) -> None:
    prerequisite = tmp_path / "controls_artifact_seal.json"
    prerequisite.write_bytes(b"sealed-controls")
    expected = {prerequisite.name: file_fingerprint(prerequisite)}
    opened = cli._open_role(tmp_path, "physical_fit", prerequisites=expected)
    open_path = tmp_path / "fit_label_open.json"
    original_sha256 = file_fingerprint(open_path)

    with pytest.raises(
        ArtifactCompatibilityError, match="existing role-open contract changed"
    ):
        cli._open_role(
            tmp_path,
            "physical_fit",
            prerequisites={prerequisite.name: "0" * 64},
        )

    assert opened["prerequisite_file_sha256"] == expected
    assert file_fingerprint(open_path) == original_sha256


def test_complete_checkpoint_index_rejects_missing_payload_before_calibration(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, stage="calibrate")
    missing = _write_minimal_training_checkpoint_index(tmp_path, args)
    missing.unlink()

    with pytest.raises(ArtifactCompatibilityError, match="checkpoint payload changed"):
        cli._verify_complete_training_checkpoint_index(tmp_path, args)
    assert not (tmp_path / "gain_label_open.json").exists()
    assert not (tmp_path / "rank_label_open.json").exists()


def test_rank_open_without_gain_seal_cannot_regenerate_gain_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, stage="calibrate")
    _write_minimal_training_checkpoint_index(tmp_path, args)
    sentinel = b"historical-gain-table-must-not-change"
    (tmp_path / "gain_table.npz").write_bytes(sentinel)
    atomic_write_json(tmp_path / "rank_label_open.json", cli._semantic({"role": "training_rank"}))
    monkeypatch.setattr(
        cli,
        "_gain_records",
        lambda *_args, **_kwargs: pytest.fail("gain evidence was regenerated"),
    )

    with pytest.raises(
        ArtifactCompatibilityError,
        match="rank labels opened without an immutable gain seal",
    ):
        cli._calibrate_stage(tmp_path, args)

    assert (tmp_path / "gain_table.npz").read_bytes() == sentinel
    assert not (tmp_path / "gain_calibration_seal.json").exists()


def test_confirmation_verifies_selection_seal_before_opening_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, stage="confirm")
    atomic_write_json(
        tmp_path / "selection_gate.json",
        {"evaluation_status": "evaluated", "passed": 1},
    )
    evidence = tmp_path / "selection_evidence.json"
    atomic_write_json(evidence, {"value": 1})
    cli._seal_stage(
        tmp_path,
        (evidence.name,),
        "selection_artifact_seal.json",
    )
    atomic_write_json(evidence, {"value": 2})
    monkeypatch.setattr(
        cli,
        "_load_selected_system",
        lambda *_args, **_kwargs: pytest.fail(
            "selected system loaded before selection seal verification"
        ),
    )

    with pytest.raises(ArtifactCompatibilityError, match="sealed artifact changed"):
        cli._audit_stage(tmp_path, args, "confirmation")
    assert not (tmp_path / "confirmation_open.json").exists()


def _committed_audit_cohort(
    run_dir: Path,
) -> tuple[cli.EagerCohort, tuple[int, ...], str, str]:
    paths = tuple(range(0xF5000, 0xF5008))
    selected_steps = (15, 143, 271, 399)
    cohort = cli.EagerCohort(
        kind="confirmation",
        index=0,
        path_ids=paths,
        path_roles=("confirmation",) * len(paths),
    )
    identities = [
        (path, step, phase, midpoint)
        for path in paths
        for step in selected_steps
        for phase in range(7)
        for midpoint in range(8)
    ]
    rows = len(identities)
    sample_keys = np.asarray(
        [midpoint_sample_key(*identity) for identity in identities], dtype=np.int64
    )
    table = aggregate_quartile_audit_improvements(
        sample_keys=sample_keys,
        row_path_ids=np.asarray([value[0] for value in identities], dtype=np.int64),
        outer_steps=np.asarray([value[1] for value in identities], dtype=np.int64),
        phases=np.asarray([value[2] for value in identities], dtype=np.int64),
        midpoint_indices=np.asarray([value[3] for value in identities], dtype=np.int64),
        specialist_vs_zero_improvements=np.linspace(
            0.1, 0.2, rows, dtype=np.float64
        ),
        shrunken_vs_raw_improvements=np.linspace(
            0.05, 0.1, rows, dtype=np.float64
        ),
        expected_path_ids=paths,
        selected_outer_steps=selected_steps,
        expected_path_count=None,
    )
    arrays = {
        "path_ids": table.path_ids,
        "primary_values": table.primary_values,
        "primary_counts": table.primary_counts,
        "local_values": table.local_values,
        "local_counts": table.local_counts,
        "selected_outer_steps": table.selected_outer_steps,
    }
    data_path, metadata_path = cli._audit_cohort_paths(run_dir, "selection", 0)
    artifact = cli._atomic_npz(data_path, **arrays)
    arrays_sha256 = cli.config_fingerprint(
        {
            name: hashlib.sha256(value.tobytes(order="C")).hexdigest()
            for name, value in arrays.items()
        }
    )
    source_sha256 = "a" * 64
    cohort_plan_sha256 = "b" * 64
    metadata = cli._semantic(
        {
            "schema": cli.RUN_SCHEMA + "-audit-cohort",
            "role": "selection",
            "cohort_index": 0,
            "cohort_kind": cohort.kind,
            "path_ids": list(paths),
            "path_roles": list(cohort.path_roles),
            "selected_system_sha256": "c" * 64,
            "selected_outer_steps": list(selected_steps),
            "source_sha256": source_sha256,
            "cohort_plan_sha256": cohort_plan_sha256,
            "artifact_sha256": artifact["sha256"],
            "arrays_sha256": arrays_sha256,
            "source_row_count": rows,
            "sample_key_sha256": table.sample_key_sha256,
        }
    )
    atomic_write_json(metadata_path, metadata)
    return cohort, selected_steps, source_sha256, cohort_plan_sha256


def test_committed_audit_cohort_revalidates_canonical_sample_grid(
    tmp_path: Path,
) -> None:
    cohort, selected_steps, source_sha256, plan_sha256 = _committed_audit_cohort(
        tmp_path
    )
    loaded = cli._load_audit_cohort(
        tmp_path,
        "selection",
        cohort,
        selected_system_sha256="c" * 64,
        selected_outer_steps=selected_steps,
        source_sha256=source_sha256,
        cohort_plan_sha256=plan_sha256,
    )
    assert loaded is not None
    metadata_path = cli._audit_cohort_paths(tmp_path, "selection", 0)[1]
    changed = cli._load_json(metadata_path)
    changed["sample_key_sha256"] = "0" * 64
    changed.pop("semantic_sha256")
    atomic_write_json(metadata_path, cli._semantic(changed))
    with pytest.raises(ArtifactCompatibilityError, match="cohort changed"):
        cli._load_audit_cohort(
            tmp_path,
            "selection",
            cohort,
            selected_system_sha256="c" * 64,
            selected_outer_steps=selected_steps,
            source_sha256=source_sha256,
            cohort_plan_sha256=plan_sha256,
        )


def test_audit_evidence_index_recursively_rejects_shard_tamper(
    tmp_path: Path,
) -> None:
    cohort, selected_steps, source_sha256, plan_sha256 = _committed_audit_cohort(
        tmp_path
    )
    counts = prepare_bootstrap_count_shards(
        tmp_path / "bootstrap_counts" / "selection",
        seed=261_350,
        namespace=0x51545331,
        path_count=8,
        replicates=8,
        shard_size=4,
    )
    del counts
    values = np.asarray(
        [[0.02 * path + 0.001 * component for component in range(6)] for path in range(8)],
        dtype=np.float64,
    )
    restartable_quartile_max_t(
        values,
        path_ids=np.asarray(cohort.path_ids, dtype=np.int64),
        count_directory=tmp_path / "bootstrap_counts" / "selection",
        maxima_directory=tmp_path / "bootstrap_maxima" / "selection",
        seed=261_350,
        namespace=0x51545331,
        confidence=0.995,
        replicates=8,
        shard_size=4,
    )
    atomic_write_json(tmp_path / "selected_system_seal.json", {"fixture": 1})
    cli._audit_evidence_index(
        tmp_path,
        "selection",
        cohorts=(cohort,),
        selected_system_sha256="c" * 64,
        selected_outer_steps=selected_steps,
        source_sha256=source_sha256,
        cohort_plan_sha256=plan_sha256,
        replicates=8,
        shard_size=4,
    )
    cli._verify_audit_evidence_index(
        tmp_path,
        "selection",
        cohorts=(cohort,),
        selected_system_sha256="c" * 64,
        selected_outer_steps=selected_steps,
        source_sha256=source_sha256,
        cohort_plan_sha256=plan_sha256,
        replicates=8,
        shard_size=4,
    )
    (tmp_path / "bootstrap_maxima/selection/shard-00000.npz").write_bytes(
        b"tampered"
    )
    with pytest.raises(ArtifactCompatibilityError, match="audit evidence changed"):
        cli._verify_audit_evidence_index(
            tmp_path,
            "selection",
            cohorts=(cohort,),
            selected_system_sha256="c" * 64,
            selected_outer_steps=selected_steps,
            source_sha256=source_sha256,
            cohort_plan_sha256=plan_sha256,
            replicates=8,
            shard_size=4,
        )
