from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist import diag_d0_jacobi_rb_boundary_tangent_schedule_feasibility as cli
from mnist.d0_jacobi_artifacts import atomic_write_json, config_fingerprint
from mnist.d0_jacobi_rb_boundary_tangent_schedule import (
    PILOT_PROFILE_NAMES,
    PILOT_REPEAT_COUNT,
)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        stage="preflight",
        require_gate="none",
        failed_boundary_tangent_run_dir=tmp_path / "failed",
        parent_coarse_residual_run_dir=tmp_path / "coarse",
        resume_run_dir=None,
        runs_root=tmp_path,
        run_name="fixture",
        device="cpu",
        test_only=True,
    )


def _plans(run_dir: Path) -> None:
    atomic_write_json(run_dir / "path_id_plan.json", cli.build_schedule_path_plan())
    atomic_write_json(run_dir / "cohort_plan.json", cli.build_schedule_cohort_plan())
    atomic_write_json(run_dir / "timing_plan.json", cli.build_schedule_timing_plan())
    atomic_write_json(
        run_dir / "parent_provenance.json",
        {"passed": 1, "failed_registry": {"artifact_count": 14}},
    )
    atomic_write_json(
        run_dir / "failed_boundary_tangent_readjudication.json", {"passed": 1}
    )


def test_parser_exposes_only_frozen_authorizing_surface(tmp_path: Path) -> None:
    required = [
        "--failed-boundary-tangent-run-dir",
        str(tmp_path / "failed"),
        "--parent-coarse-residual-run-dir",
        str(tmp_path / "coarse"),
    ]
    args = cli.parse_args(required + ["--stage", "report", "--device", "cpu"])
    assert args.stage == "report"
    assert args.require_gate == "none"
    with pytest.raises(SystemExit):
        cli.parse_args(required + ["--stage", "pilot", "--device", "cpu"])
    with pytest.raises(SystemExit):
        cli.parse_args(required + ["--test-only", "--require-gate", "pilot"])


def test_scientific_config_fingerprints_exact_thirty_hour_limit(tmp_path: Path) -> None:
    record = cli._scientific_config(_args(tmp_path))
    assert record["maximum_projected_exact_cache_hours"] == 30.0
    body = dict(record)
    semantic = body.pop("semantic_sha256")
    assert semantic == config_fingerprint(body)


def test_stage_sequence_is_fail_closed() -> None:
    assert cli._stage_sequence("all") == ("preflight", "pilot")
    assert cli._stage_sequence("report") == ()
    assert cli._stage_sequence("pilot") == ("pilot",)
    with pytest.raises(ValueError):
        cli._stage_sequence("train")


def test_collision_scan_ignores_stop_metadata_and_consumes_unopened_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mnist import d0_jacobi_rb_learnability as learnability
    from mnist.d0_jacobi_rb_learnability import PathIDClaim

    run_dir = tmp_path / "run"
    failed_boundary = tmp_path / "failed-boundary"
    affine = tmp_path / "failed-affine"
    run_dir.mkdir()
    failed_boundary.mkdir()
    affine.mkdir()
    atomic_write_json(run_dir / "path_id_plan.json", cli.build_schedule_path_plan())
    atomic_write_json(
        run_dir / "parent_provenance.json", {"failed_run_dir": str(failed_boundary)}
    )
    atomic_write_json(
        failed_boundary / "parent_provenance.json",
        {
            "parents": {
                "failed_affine_reverse_controller": {"run_dir": str(affine)}
            }
        },
    )
    affine_plan = affine / "path_id_plan.json"
    atomic_write_json(affine_plan, {"roles": {"oracle": [0xEE000]}})
    atomic_write_json(
        affine / "oracle_gate.json",
        {"evaluation_status": "not_evaluated", "passed": 0},
    )
    claims = (
        PathIDClaim(
            str(failed_boundary / "path_id_plan.json"),
            "role_slots.confirmation.stop_exclusive",
            0xED040,
            0xF0000,
        ),
        PathIDClaim(str(affine_plan), "roles.oracle[0]", 0xEE000, 0xEE001),
    )
    monkeypatch.setattr(
        learnability, "discover_repository_path_id_claims", lambda _: claims
    )
    record = cli._semantic_path_collision_scan(run_dir)
    assert record["passed"] == 1
    assert record["consumed_unrealized_oracle_reservation"] == 1
    assert len(record["ignored_stop_exclusive_metadata_claims"]) == 1


def test_reduced_preflight_populates_complete_gate_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _plans(run_dir)
    monkeypatch.setattr(
        cli,
        "_semantic_path_collision_scan",
        lambda root: atomic_write_json(root / "path_collision_scan.json", {"passed": 1})
        or {"passed": 1},
    )
    monkeypatch.setattr(
        cli,
        "_initial_state_plan",
        lambda _: {"schema": "fixture", "passed": 1},
    )
    monkeypatch.setattr(
        cli,
        "_preflight_equivalence",
        lambda *_: {
            "base_equivalence_valid": 1,
            "path_permutation_invariance_valid": 1,
            "fused_branch_equivalence_valid": 1,
            "cross_role_isolation_valid": 1,
            "maximum_observed_launch_lanes": 4096,
            "passed": 1,
        },
    )
    gate = cli._preflight_stage(run_dir, _args(tmp_path))
    assert gate["passed"] == 1
    metrics = cli._load_json(run_dir / "schedule_preflight_metrics.json")
    assert metrics["maximum_projected_exact_cache_hours"] == 30.0
    assert metrics["projected_transition_count"] == 337_182_720
    assert metrics["restart_outer_steps"] == 8


def test_test_only_pilot_exercises_gate_and_projection(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(
        run_dir / "preflight_gate.json",
        {"evaluation_status": "evaluated", "passed": 1},
    )
    atomic_write_json(run_dir / "path_collision_scan.json", {"collision_count": 0})
    atomic_write_json(
        run_dir / "schedule_equivalence_preflight.json",
        {"cross_role_isolation_valid": 1},
    )
    gate = cli._pilot_stage(run_dir, _args(tmp_path))
    assert gate["passed"] == 1
    metrics = cli._load_json(run_dir / "pilot_metrics.json")
    assert metrics["completed_shard_count"] == 96
    assert metrics["pilot_total_executed_transition_count"] == 23_708_160
    assert set(metrics["profile_elapsed_seconds"]) == set(PILOT_PROFILE_NAMES)
    assert all(
        len(values) == PILOT_REPEAT_COUNT
        for values in metrics["profile_elapsed_seconds"].values()
    )
    atomic_write_json(run_dir / "scientific_config.json", {"test_only": 1})
    workflow = cli._workflow_record(run_dir, require_gate="none")
    assert workflow["decision"]["decision"] == "test_only_complete"
    assert workflow["decision"]["schedule_integration_authorized"] == 0


def test_source_set_binds_live_numerical_and_model_closure() -> None:
    names = {path.name for path in cli._source_set()}
    assert {
        "d0_jacobi_artifacts.py",
        "d0_jacobi_source_compat.py",
        "d0_jacobi_rb_boundary_tangent_cache.py",
        "d0_jacobi_rb_cuda.py",
        "d0_jacobi_rb_cuda_certificate.py",
        "d0_jacobi_rb_cuda_fused.py",
        "d0_jacobi_rb_cuda_multipath.py",
        "d0_jacobi_rb_learnability.py",
        "d0_jacobi_rb_reverse_controller.py",
        "d0_jacobi_rb_coarse_residual.py",
        "d0_jacobi_rb_spectral.py",
    } <= names


def test_failed_gate_seal_is_integrity_valid_and_orphan_gate_refinalizes(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(run_dir / "evidence.json", {"value": 1})
    atomic_write_json(
        run_dir / "preflight_gate.json",
        {"evaluation_status": "evaluated", "passed": 0},
    )
    assert (
        cli._sealed_stage(
            run_dir,
            gate_name="preflight_gate.json",
            seal_name="preflight_artifact_seal.json",
        )
        is None
    )
    cli._seal_stage(
        run_dir,
        ("evidence.json", "preflight_gate.json"),
        "preflight_artifact_seal.json",
    )
    gate = cli._sealed_stage(
        run_dir,
        gate_name="preflight_gate.json",
        seal_name="preflight_artifact_seal.json",
    )
    assert gate is not None and gate["passed"] == 0


def test_registry_accepts_only_frozen_restartable_extras(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(run_dir / "evidence.json", {"value": 1})
    cli._artifact_registry(run_dir)
    atomic_write_json(run_dir / "cuda_model_warmup.json", {"passed": 1})
    cli._verify_existing_registry(run_dir)
    atomic_write_json(run_dir / "unexpected.json", {"bad": 1})
    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._verify_existing_registry(run_dir)


def test_branch_conservation_covers_global_and_pair_mass() -> None:
    paths = 2
    base = np.full((cli.PHASE_COUNT, paths, cli.STATE_SIZE), 1.0 / cli.STATE_SIZE)
    batches = []
    for phase in range(cli.PHASE_COUNT):
        later = torch.as_tensor(
            np.repeat(base[phase][None, :, :], 8, axis=0), dtype=torch.float64
        )
        batches.append(
            SimpleNamespace(
                batch=SimpleNamespace(
                    path_ids=tuple(range(paths)), later_full_state=later
                )
            )
        )
    values = cli._branch_conservation_arrays(batches, base)
    assert cli._maximum_branch_mass_error(values) == 0.0
    batches[0].batch.later_full_state[0, 0, 0] += 1.0e-6
    changed = cli._branch_conservation_arrays(batches, base)
    assert cli._maximum_branch_mass_error(changed) > 0.99e-6


def test_completed_repeat_loader_is_reachable_and_tamper_fails(tmp_path: Path) -> None:
    record = cli._test_pilot_records()[0]
    payload = record.to_record()
    payload["semantic_sha256"] = config_fingerprint(payload)
    path = tmp_path / "repeat.json"
    atomic_write_json(path, payload)
    loaded = cli._load_completed_repeat(path)
    assert loaded is not None
    assert loaded.profile_name == record.profile_name
    payload["elapsed_seconds"] += 1.0
    atomic_write_json(path, payload)
    assert cli._load_completed_repeat(path) is None


def test_reduced_cpu_shard_branch_commit_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the real scheduler/commit path with the deterministic CPU sampler."""

    args = _args(tmp_path)
    root = tmp_path / "repeat"
    path_ids = (int(cli.PROFILE_PATH_IDS[cli.PROFILE_CACHE_P10][0]),)
    initial = np.full((1, cli.STATE_SIZE), 1.0 / cli.STATE_SIZE, dtype=np.float64)
    model = cli.JacobiRBPhasePredictor(width=32).to("cpu").eval()

    first, _, first_record = cli._run_base_shard(
        root,
        args,
        profile_name=cli.PROFILE_CACHE_P10,
        repeat_index=0,
        path_ids=path_ids,
        window_start=0,
        shard_start=0,
        input_states=initial,
        capture_expected=False,
        device=torch.device("cpu"),
    )
    second, pre_phase, second_record = cli._run_base_shard(
        root,
        args,
        profile_name=cli.PROFILE_CACHE_P10,
        repeat_index=0,
        path_ids=path_ids,
        window_start=0,
        shard_start=8,
        input_states=first,
        capture_expected=True,
        device=torch.device("cpu"),
    )
    assert pre_phase is not None and pre_phase.shape == (7, 1, cli.STATE_SIZE)
    branch_record = cli._run_branch_commit(
        root,
        args,
        profile_name=cli.PROFILE_CACHE_P10,
        repeat_index=0,
        path_ids=path_ids,
        window_start=0,
        pre_phase_states=pre_phase,
        device=torch.device("cpu"),
        model=model,
    )
    assert first_record["committed"] == second_record["committed"] == 1
    assert branch_record["transition_count"] == 7 * 8 * 392
    assert branch_record["certified_count"] == branch_record["transition_count"]
    assert branch_record["maximum_mass_error"] <= 2.0e-12

    first_hash = first_record["final_state_sha256"]
    second_hash = second_record["final_state_sha256"]
    branch_hash = branch_record["output_sha256"]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a committed shard must be loaded, not recomputed")

    monkeypatch.setattr(cli, "run_exact_multipath_shard", forbidden)
    monkeypatch.setattr(cli, "sample_fused_midpoint_branches", forbidden)
    replay_first, _, replay_first_record = cli._run_base_shard(
        root,
        args,
        profile_name=cli.PROFILE_CACHE_P10,
        repeat_index=0,
        path_ids=path_ids,
        window_start=0,
        shard_start=0,
        input_states=initial,
        capture_expected=False,
        device=torch.device("cpu"),
    )
    replay_second, replay_pre, replay_second_record = cli._run_base_shard(
        root,
        args,
        profile_name=cli.PROFILE_CACHE_P10,
        repeat_index=0,
        path_ids=path_ids,
        window_start=0,
        shard_start=8,
        input_states=replay_first,
        capture_expected=True,
        device=torch.device("cpu"),
    )
    replay_branch = cli._run_branch_commit(
        root,
        args,
        profile_name=cli.PROFILE_CACHE_P10,
        repeat_index=0,
        path_ids=path_ids,
        window_start=0,
        pre_phase_states=replay_pre,
        device=torch.device("cpu"),
        model=model,
    )
    assert np.array_equal(replay_first, first)
    assert np.array_equal(replay_second, second)
    assert replay_first_record["final_state_sha256"] == first_hash
    assert replay_second_record["final_state_sha256"] == second_hash
    assert replay_branch["output_sha256"] == branch_hash


def test_required_resource_failure_commits_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.stage = "pilot"
    args.require_gate = "pilot"
    args.test_only = False
    args.device = "cuda"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(cli, "_make_run_dir", lambda _: (run_dir, False))
    monkeypatch.setattr(cli, "_initialize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda *_: {})
    atomic_write_json(
        run_dir / "preflight_gate.json",
        {"evaluation_status": "evaluated", "passed": 1},
    )
    resource_gate = {
        "evaluation_status": "evaluated",
        "passed": 0,
        "failure_domain": "resource_gate",
        "scientific_evidence_complete": 1,
        "numerically_valid": 1,
        "stage_execution_valid": 1,
    }
    monkeypatch.setattr(cli, "_pilot_stage", lambda *_: resource_gate)
    assert cli._run(args) == 2
    status = cli._load_json(run_dir / "run_status.json")
    assert status["failure_domain"] == "resource_gate"
    assert status["scientific_evidence_complete"] == 1
    assert (run_dir / "artifact_registry.json").is_file()


def test_rejected_resume_is_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _args(tmp_path)
    args.stage = "report"
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    (run_dir / "marker.bin").write_bytes(b"immutable")
    args.resume_run_dir = run_dir
    before = {path.name: path.read_bytes() for path in run_dir.iterdir()}
    monkeypatch.setattr(
        cli,
        "_initialize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            cli.ArtifactCompatibilityError("source changed")
        ),
    )
    assert cli._run(args) == 2
    assert {path.name: path.read_bytes() for path in run_dir.iterdir()} == before
