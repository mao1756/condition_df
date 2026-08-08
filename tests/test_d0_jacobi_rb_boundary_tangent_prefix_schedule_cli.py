from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

from mnist import (
    diag_d0_jacobi_rb_boundary_tangent_prefix_schedule_confirmation as cli,
)
from mnist.d0_jacobi_artifacts import atomic_write_json
from mnist.d0_jacobi_rb_spectral import JacobiRBCertificationError


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        stage="preflight",
        require_gate="none",
        parent_schedule_run_dir=tmp_path / "parent-schedule",
        resume_run_dir=None,
        runs_root=tmp_path,
        run_name="fixture",
        device="cpu",
        test_only=True,
    )


def _write_preflight_inputs(run_dir: Path) -> None:
    atomic_write_json(run_dir / "path_id_plan.json", cli.build_schedule_path_plan())
    atomic_write_json(run_dir / "cohort_plan.json", cli.build_schedule_cohort_plan())
    atomic_write_json(run_dir / "timing_plan.json", cli.build_schedule_timing_plan())
    atomic_write_json(
        run_dir / "parent_provenance.json",
        {
            "passed": 1,
            "parent_resource_only_failure": 1,
            "parent_scientific_evidence_complete": 1,
            "parent_registry": {"artifact_count": 614},
        },
    )
    atomic_write_json(run_dir / "parent_schedule_readjudication.json", {"passed": 1})
    contract = cli.eager_prefix_contract()
    contract["semantic_sha256"] = cli.config_fingerprint(contract)
    atomic_write_json(run_dir / "eager_prefix_contract.json", contract)
    atomic_write_json(run_dir / "prefix_profile_plan.json", {"schema": "fixture"})


def _passing_gate() -> dict[str, object]:
    return {"evaluation_status": "evaluated", "passed": 1}


def test_parser_stages_and_authorizing_surface(tmp_path: Path) -> None:
    required = ["--parent-schedule-run-dir", str(tmp_path / "parent")]
    args = cli.parse_args(required + ["--stage", "report", "--device", "cpu"])
    assert args.stage == "report"
    assert args.require_gate == "none"
    assert cli.STAGES == ("preflight", "profile", "pilot", "report", "all")
    assert cli.REQUIRED_GATES == ("none", "preflight", "profile", "pilot")
    assert cli._stage_sequence("all") == ("preflight", "profile", "pilot")
    assert cli._stage_sequence("report") == ()
    with pytest.raises(SystemExit):
        cli.parse_args(required + ["--stage", "pilot", "--device", "cpu"])
    with pytest.raises(SystemExit):
        cli.parse_args(required + ["--test-only", "--require-gate", "pilot"])
    with pytest.raises(ValueError):
        cli._stage_sequence("train")


def test_source_set_binds_prefix_and_parent_schedule_closure() -> None:
    names = {path.name for path in cli._source_set()}
    assert {
        "diag_d0_jacobi_rb_boundary_tangent_prefix_schedule_confirmation.py",
        "diag_d0_jacobi_rb_boundary_tangent_schedule_feasibility.py",
        "d0_jacobi_rb_boundary_tangent_eager_prefix_provenance.py",
        "d0_jacobi_rb_boundary_tangent_prefix_schedule.py",
        "d0_jacobi_rb_boundary_tangent_prefix_schedule_gate.py",
        "d0_jacobi_rb_boundary_tangent_schedule.py",
        "d0_jacobi_rb_boundary_tangent_schedule_gate.py",
        "d0_jacobi_rb_boundary_tangent_schedule_provenance.py",
        "d0_jacobi_rb_cuda.py",
        "d0_jacobi_rb_cuda_multipath.py",
        "d0_jacobi_rb_learnability.py",
    } <= names


def test_reduced_preflight_populates_complete_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_preflight_inputs(run_dir)

    def collision(root: Path) -> dict[str, int]:
        value = {"passed": 1, "collision_count": 0}
        atomic_write_json(root / "path_collision_scan.json", value)
        return value

    monkeypatch.setattr(cli, "_semantic_path_collision_scan", collision)
    monkeypatch.setattr(cli, "_initial_state_plan", lambda _: {"passed": 1})
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
    monkeypatch.setattr(
        cli,
        "_prefix_equivalence_preflight",
        lambda *_: {
            "prefix_interval_nesting_valid": 1,
            "base_scientific_output_equivalent": 1,
            "branch_scientific_output_equivalent": 1,
            "final_state_hashes_equal": 1,
            "eager_certificate_fraction": 1.0,
            "forbidden_event_count": 0,
        },
    )
    gate = cli._preflight_stage(run_dir, _args(tmp_path))
    assert gate["passed"] == 1
    metrics = cli._load_json(run_dir / "prefix_schedule_preflight_metrics.json")
    assert metrics["parent_record_count"] == 614
    assert metrics["initial_eager_prefix_bits"] == 128
    assert metrics["candidate_modes"] == 128
    assert metrics["cross_role_isolation_valid"] == 1
    assert (run_dir / "preflight_artifact_seal.json").is_file()


def test_test_only_profile_freezes_eager_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(run_dir / "preflight_gate.json", _passing_gate())
    cli._seal_stage(
        run_dir, ("preflight_gate.json",), "preflight_artifact_seal.json"
    )
    monkeypatch.setattr(
        cli,
        "_parent_p10_timing_decomposition",
        lambda _: {
            "passed": 1,
            "parent_projected_seconds": 128_438.0,
            "weighted_p10_base_authorizer_seconds": 80_000.0,
        },
    )
    gate = cli._profile_stage(run_dir, _args(tmp_path))
    assert gate["passed"] == 1
    selected = cli._load_json(run_dir / "selected_eager_prefix_profile.json")
    assert selected["selected"] == 1
    assert selected["profile_name"] == cli.EAGER_PROFILE_NAME
    assert cli.config_fingerprint(selected["profile"]) == cli.config_fingerprint(
        cli.eager_prefix_profile().to_dict()
    )
    qualification = cli._load_json(run_dir / "prefix_profile_qualification.json")
    assert qualification["conservative_authorizer_speedup"] == pytest.approx(10.0 / 6.0)
    assert qualification["projected_elapsed_seconds"] <= 108_000.0
    metrics = cli._load_json(run_dir / "profile_metrics.json")
    assert metrics["eager_prefix_policy_observed"] == 1


def test_test_only_pilot_exercises_unchanged_schedule_gate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(run_dir / "preflight_gate.json", _passing_gate())
    atomic_write_json(run_dir / "profile_gate.json", _passing_gate())
    cli._seal_stage(
        run_dir, ("preflight_gate.json",), "preflight_artifact_seal.json"
    )
    cli._seal_stage(run_dir, ("profile_gate.json",), "profile_artifact_seal.json")
    atomic_write_json(run_dir / "path_collision_scan.json", {"collision_count": 0})
    gate = cli._pilot_stage(run_dir, _args(tmp_path))
    assert gate["passed"] == 1
    metrics = cli._load_json(run_dir / "pilot_metrics.json")
    assert metrics["all_profiles_complete"] == 1
    assert metrics["eager_prefix_policy_applied"] == 1
    assert metrics["candidate_modes"] == 128
    assert metrics["scientific_hashes_match_parent"] == 1
    assert metrics["pilot_total_executed_transition_count"] == 23_708_160
    assert metrics["projected_elapsed_seconds"] <= 108_000.0
    assert (run_dir / "pilot_artifact_seal.json").is_file()


def test_base_and_branch_commits_bind_exact_eager_profile(tmp_path: Path) -> None:
    args = _args(tmp_path)
    root = tmp_path / "repeat"
    path_ids = (int(cli.PROFILE_PATH_IDS[cli.PROFILE_CACHE_P10][0]),)
    initial = np.full((1, cli.STATE_SIZE), 1.0 / cli.STATE_SIZE, dtype=np.float64)
    model = cli.JacobiRBPhasePredictor(width=32).to("cpu").eval()
    first, _, _ = cli._run_base_shard(
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
    _, pre_phase, shard = cli._run_base_shard(
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
    assert pre_phase is not None
    branch = cli._run_branch_commit(
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
    expected = cli._eager_profile_sha256()
    assert shard["prefix_policy"] == cli.EAGER_PREFIX_POLICY
    assert shard["prefix_profile_sha256"] == expected
    assert shard["peak_memory_fraction"] == 0.0
    assert branch["prefix_policy"] == cli.EAGER_PREFIX_POLICY
    assert branch["prefix_profile_sha256"] == expected
    assert branch["peak_memory_fraction"] == 0.0
    assert branch["target_transformed"] == 0

    invalid_shard = dict(shard)
    invalid_shard["peak_memory_fraction"] = 1.01
    invalid_shard["semantic_sha256"] = cli.config_fingerprint(
        {
            key: value
            for key, value in invalid_shard.items()
            if key != "semantic_sha256"
        }
    )
    atomic_write_json(root / "window-000" / "step-008.json", invalid_shard)
    assert cli._valid_base_shard(
        root,
        profile_name=cli.PROFILE_CACHE_P10,
        repeat_index=0,
        path_ids=path_ids,
        window_start=0,
        shard_start=8,
        input_states=first,
        capture_expected=True,
    ) is None

    invalid_branch = dict(branch)
    invalid_branch["peak_memory_fraction"] = -0.01
    invalid_branch["semantic_sha256"] = cli.config_fingerprint(
        {
            key: value
            for key, value in invalid_branch.items()
            if key != "semantic_sha256"
        }
    )
    atomic_write_json(root / "window-000" / "branch.json", invalid_branch)
    assert cli._valid_branch_commit(
        root,
        profile_name=cli.PROFILE_CACHE_P10,
        repeat_index=0,
        window_start=0,
        path_ids=path_ids,
        pre_phase_states=pre_phase,
    ) is None


def test_cached_child_reconstruction_uses_maximum_peak_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.parent_coarse_residual_run_dir = tmp_path / "coarse-parent"
    root = tmp_path / "repeat"
    profile_name = cli.PROFILE_CACHE_P10
    repeat_index = 0
    path_ids = tuple(int(value) for value in cli.PROFILE_PATH_IDS[profile_name])
    base_total, midpoint_total, transition_total = (
        cli.expected_profile_transition_counts(profile_name)
    )
    first_count = base_total // 2
    second_count = base_total - first_count
    initial = np.zeros((len(path_ids), cli.STATE_SIZE), dtype=np.float64)
    first_final = np.full_like(initial, 0.25)
    second_final = np.full_like(initial, 0.5)
    pre_phase = np.zeros(
        (cli.PHASE_COUNT, len(path_ids), cli.STATE_SIZE), dtype=np.float64
    )

    def base_record(count: int, peak: float, digest: str) -> dict[str, object]:
        return {
            "complete_pipeline_elapsed_seconds": 1.0,
            "peak_memory_fraction": peak,
            "scheduler_record": {
                "batch_output_sha256": digest,
                "diagnostics": {
                    "transition_count": count,
                    "certified_count": count,
                    "fallback_count": 0,
                    "fallback_elapsed_seconds": 0.0,
                    "maximum_mass_error": 1.0e-13,
                    "maximum_cuda_launch_lanes": 64,
                },
            },
        }

    first_record = base_record(first_count, 0.20, "a" * 64)
    second_record = base_record(second_count, 0.70, "b" * 64)
    branch_record: dict[str, object] = {
        "complete_pipeline_elapsed_seconds": 1.0,
        "peak_memory_fraction": 0.50,
        "transition_count": midpoint_total,
        "certified_count": midpoint_total,
        "fallback_count": 0,
        "fallback_elapsed_seconds": 0.0,
        "maximum_mass_error": 2.0e-13,
        "maximum_launch_lanes": 128,
        "forbidden_counts": {name: 0 for name in cli.FORBIDDEN_DIAGNOSTICS},
        "output_sha256": "c" * 64,
    }

    monkeypatch.setattr(cli, "WINDOW_START_STEPS", (0,))
    monkeypatch.setattr(cli, "_repeat_child_file_set_valid", lambda _: True)
    monkeypatch.setattr(cli, "_window_files_size", lambda _: 123)
    monkeypatch.setattr(
        cli,
        "_load_benchmark_initial_states",
        lambda *_args, **_kwargs: (initial, {"fixture": 1}),
    )

    def valid_base(
        _root: Path, *, shard_start: int, **_kwargs: object
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]]:
        if shard_start == 0:
            return first_final, None, first_record
        return second_final, pre_phase, second_record

    monkeypatch.setattr(cli, "_valid_base_shard", valid_base)
    monkeypatch.setattr(cli, "_valid_branch_commit", lambda *_args, **_kwargs: branch_record)

    reconstructed = cli._reconstruct_repeat_from_children(
        root,
        args,
        profile_name=profile_name,
        repeat_index=repeat_index,
        execution_order_index=cli.frozen_repeat_order(repeat_index).index(
            profile_name
        ),
        path_ids=path_ids,
    )
    assert reconstructed is not None
    assert reconstructed.certified_count == transition_total
    assert reconstructed.peak_memory_fraction == pytest.approx(0.70)


def test_all_stage_runs_in_order_and_stops_after_failed_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.stage = "all"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(run_dir / "parent_provenance.json", {"passed": 1})
    atomic_write_json(run_dir / "scientific_config.json", {"test_only": 1})
    calls: list[str] = []
    monkeypatch.setattr(cli, "_make_run_dir", lambda _: (run_dir, False))
    monkeypatch.setattr(cli, "_initialize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda *_: {})
    monkeypatch.setattr(
        cli,
        "_preflight_stage",
        lambda *_: calls.append("preflight") or _passing_gate(),
    )
    monkeypatch.setattr(
        cli,
        "_profile_stage",
        lambda *_: calls.append("profile")
        or {"evaluation_status": "evaluated", "passed": 0},
    )
    monkeypatch.setattr(
        cli,
        "_pilot_stage",
        lambda *_: calls.append("pilot") or _passing_gate(),
    )
    assert cli._run(args) == 0  # require-gate none remains nonauthorizing.
    assert calls == ["preflight", "profile"]
    assert not (run_dir / "pilot_gate.json").exists()


def test_typed_execution_failure_maps_and_commits_readable_artifacts(
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
        raise cli.BoundaryTangentScheduleCLIError(
            "prefix contract exploded",
            failure_domain="rng_contract",
            failure_code="eager_prefix_rng_contract_invalid",
        )

    monkeypatch.setattr(cli, "_preflight_stage", fail)
    assert cli._run(args) == 2
    failure = cli._load_json(run_dir / "preflight_execution_failure.json")
    gate = cli._load_json(run_dir / "preflight_gate.json")
    status = cli._load_json(run_dir / "run_status.json")
    assert failure["evaluation_status"] == "execution_failed"
    assert failure["scientific_evidence_complete"] == 0
    assert failure["failure_code"] == "eager_prefix_rng_contract_invalid"
    assert gate["failure_domain"] == "rng_contract"
    assert status["state"] == "execution_failed"
    assert (run_dir / "artifact_registry.json").is_file()


def test_certificate_exhaustion_has_certificate_failure_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.stage = "preflight"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(run_dir / "parent_provenance.json", {"passed": 1})
    monkeypatch.setattr(cli, "_make_run_dir", lambda _: (run_dir, False))
    monkeypatch.setattr(cli, "_initialize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda *_: {})

    def exhaust(*_args: object) -> dict[str, object]:
        raise JacobiRBCertificationError(
            "candidate-local Arb fallback failed closed",
            {
                "failure_kind": "arb_resource_cap",
                "resource_kind": "candidate_lattice",
            },
        )

    monkeypatch.setattr(cli, "_preflight_stage", exhaust)
    assert cli._run(args) == 2

    failure = cli._load_json(run_dir / "preflight_execution_failure.json")
    gate = cli._load_json(run_dir / "preflight_gate.json")
    workflow = cli._load_json(run_dir / "workflow_gate.json")
    assert failure["failure_domain"] == "certificate_execution"
    assert failure["failure_code"] == "eager_prefix_certificate_fallback_failed"
    assert gate["failure_domain"] == "certificate_execution"
    assert workflow["decision"]["decision"] == "eager_prefix_certificate_invalid"
    assert workflow["decision"]["decision"] != "eager_prefix_equivalence_invalid"


def test_execution_failure_replaces_only_an_unsealed_passing_gate(
    tmp_path: Path,
) -> None:
    error = cli.BoundaryTangentScheduleCLIError(
        "profile execution exploded",
        failure_domain="schedule_execution",
        failure_code="profile_execution_exploded",
    )

    unsealed = tmp_path / "unsealed"
    unsealed.mkdir()
    atomic_write_json(unsealed / "profile_gate.json", _passing_gate())
    cli._commit_execution_failure(
        unsealed, stage="profile", exc=error, require_gate="none"
    )
    failed_gate = cli._load_json(unsealed / "profile_gate.json")
    assert failed_gate["evaluation_status"] == "execution_failed"
    assert failed_gate["passed"] == 0
    assert failed_gate["failure_code"] == "profile_execution_exploded"

    sealed = tmp_path / "sealed"
    sealed.mkdir()
    atomic_write_json(sealed / "profile_gate.json", _passing_gate())
    cli._seal_stage(sealed, ("profile_gate.json",), "profile_artifact_seal.json")
    cli._commit_execution_failure(
        sealed, stage="profile", exc=error, require_gate="none"
    )
    preserved_gate = cli._load_json(sealed / "profile_gate.json")
    assert preserved_gate == _passing_gate()
    assert cli._sealed_stage(
        sealed,
        gate_name="profile_gate.json",
        seal_name="profile_artifact_seal.json",
    ) == _passing_gate()
    assert (sealed / "profile_execution_failure.json").is_file()


def test_keyboard_interrupt_records_interruption_without_masking_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.stage = "preflight"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(cli, "_make_run_dir", lambda _: (run_dir, False))
    monkeypatch.setattr(cli, "_initialize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda *_: {})

    def interrupt(*_args: object) -> dict[str, object]:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_preflight_stage", interrupt)
    with pytest.raises(KeyboardInterrupt):
        cli._run(args)

    status = cli._load_json(run_dir / "run_status.json")
    assert status["state"] == "interrupted"
    assert status["failure_code"] == "keyboard_interrupt"
    assert status["scientific_evidence_complete"] == 0
    assert (run_dir / "artifact_registry.json").is_file()


def test_initialization_interrupt_requires_a_fresh_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.stage = "preflight"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(cli, "_make_run_dir", lambda _: (run_dir, False))

    def interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_initialize", interrupt)
    with pytest.raises(KeyboardInterrupt):
        cli._run(args)

    status = cli._load_json(run_dir / "run_status.json")
    assert status["state"] == "initialization_interrupted"
    assert status["failure_code"] == "initialization_interrupted"
    assert "fresh run" in status["message"]


def test_pilot_shard_registry_excludes_atomic_temp_files(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot" / "cache_p10" / "repeat-00"
    pilot.mkdir(parents=True)
    atomic_write_json(pilot / "repeat.json", {"committed": 1})
    (pilot / "repeat.json.tmp").write_text("partial", encoding="utf-8")
    (pilot / "repeat.tmp.deadbeef").write_text("partial", encoding="utf-8")

    artifacts = cli._pilot_shard_artifacts(tmp_path)
    assert [item["path"] for item in artifacts] == [
        "pilot/cache_p10/repeat-00/repeat.json"
    ]


def test_registry_allows_only_frozen_profile_and_pilot_extras(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(run_dir / "evidence.json", {"value": 1})
    cli._artifact_registry(run_dir)
    allowed = run_dir / "profile" / "repeat-00.json"
    allowed.parent.mkdir(parents=True)
    atomic_write_json(allowed, {"committed": 1})
    cli._verify_existing_registry(run_dir)
    atomic_write_json(run_dir / "unexpected.json", {"bad": 1})
    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._verify_existing_registry(run_dir)
