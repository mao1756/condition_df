from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mnist import diag_d0_jacobi_rb_frequency1_rollout as cli
from mnist.d0_jacobi_rb_tangent_rollout import (
    load_verified_source_target,
    source_measure_sha256,
)


def _json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fixture_args(tmp_path: Path, name: str, *extra: str) -> list[str]:
    parent = tmp_path / name / "parent"
    source = tmp_path / name / "source"
    parent.mkdir(parents=True, exist_ok=True)
    source.mkdir(parents=True, exist_ok=True)
    return [
        "--test-only",
        "--device",
        "cpu",
        "--stage",
        "all",
        "--runs-root",
        str(tmp_path / name / "runs"),
        "--run-name",
        name,
        "--frequency1-run-dir",
        str(parent),
        "--source-run-dir",
        str(source),
        *extra,
    ]


def _only_run(root: Path) -> Path:
    values = [path for path in root.iterdir() if path.is_dir()]
    assert len(values) == 1
    return values[0]


def _fused_fixture_args(tmp_path: Path, name: str, *extra: str) -> list[str]:
    carrier = tmp_path / name / "sealed-carrier"
    carrier.mkdir(parents=True, exist_ok=True)
    return [
        *_fixture_args(tmp_path, name),
        "--continuation-run-dir",
        str(carrier),
        *extra,
    ]


def _recovery_fixture_args(tmp_path: Path, name: str, *extra: str) -> list[str]:
    predecessor = tmp_path / name / "immutable-predecessor"
    carrier = tmp_path / name / "sealed-carrier"
    predecessor.mkdir(parents=True, exist_ok=True)
    carrier.mkdir(parents=True, exist_ok=True)
    return [
        *_fixture_args(tmp_path, name),
        "--continuation-run-dir",
        str(carrier),
        "--predecessor-run-dir",
        str(predecessor),
        *extra,
    ]


def _mechanism_rows_fixture(count: int = 3) -> tuple[dict[str, float], ...]:
    return tuple(
        {
            "reference_fraction_displacement_rms": 1.0e-3,
            "control_fraction_displacement_rms": float(index) * 2.0e-4,
            "score_rms": float(index) * 1.0e-3,
            "logistic_shift_rms": float(index) * 5.0e-5,
        }
        for index in range(count)
    )


def test_parser_separates_production_and_test_only_authority(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--device", "cpu", "--stage", "all"])
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--test-only",
                "--device",
                "cpu",
                "--stage",
                "all",
                "--require-gate",
                "evaluation",
            ]
        )
    report = cli.parse_args(
        ["--device", "cpu", "--stage", "report", "--resume-run-dir", str(tmp_path)]
    )
    assert report.stage == "report"
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--device",
                "cuda:0",
                "--stage",
                "preflight",
                "--continuation-run-dir",
                str(tmp_path),
            ]
        )


def test_fused_projection_uses_slowest_repeat_and_exact_profile_arithmetic() -> None:
    repeats = {
        name: [
            {
                "elapsed_seconds": elapsed,
                "transition_count": count,
                "output_sha256": f"{name}-stable",
                "health_passed": 1,
            }
            for elapsed in (1.0, 2.0, 1.5)
        ]
        for name, count in cli.FUSED_PROFILE_TRANSITIONS.items()
    }
    record = cli._fused_resource_projection(repeats, test_only=True)
    assert record["profiles"]["forward_p1"]["slowest_repeat_seconds"] == 2.0
    assert record["profiles"]["forward_p1"]["slowest_repeat_index"] == 1
    expected = (
        128 * 2.0
        + 48 * 2.0
        + 32 * 2.0
        + cli.FUSED_PROJECTION_FIXED_RESERVE_SECONDS
    )
    assert record["projected_main_wall_seconds"] == expected
    assert sum(
        cli.FUSED_PROFILE_TRANSITIONS[name]
        * cli.FUSED_PROFILE_PRODUCTION_SHARDS[name]
        for name in cli.FUSED_PROFILE_TRANSITIONS
    ) == cli.MAIN_WORKFLOW_TRANSITIONS


def test_fused_projection_exact_six_hour_and_nextafter_boundaries() -> None:
    variable_budget = (
        cli.MAXIMUM_MAIN_WALL_SECONDS
        - cli.FUSED_PROJECTION_FIXED_RESERVE_SECONDS
    )
    denominator = sum(cli.FUSED_PROFILE_PRODUCTION_SHARDS.values())
    exact_elapsed = variable_budget / denominator

    def build(elapsed: float) -> dict[str, list[dict[str, object]]]:
        return {
            name: [
                {
                    "elapsed_seconds": elapsed,
                    "transition_count": count,
                    "output_sha256": f"{name}-stable",
                    "health_passed": 1,
                }
                for _ in range(3)
            ]
            for name, count in cli.FUSED_PROFILE_TRANSITIONS.items()
        }

    exact = cli._fused_resource_projection(build(exact_elapsed))
    assert exact["projected_main_wall_seconds"] == cli.MAXIMUM_MAIN_WALL_SECONDS
    assert exact["checks"]["main_wall_time"] is True
    above = cli._fused_resource_projection(
        build(float(np.nextafter(exact_elapsed, np.inf)))
    )
    assert above["checks"]["main_wall_time"] is False
    assert above["passed"] == 0


def test_fused_continuation_all_runs_objective_in_same_process(tmp_path: Path) -> None:
    args = _fused_fixture_args(tmp_path, "fused-positive")
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "fused-positive" / "runs")
    config = _json(run / "scientific_config.json")
    assert config["fused_continuation"] == 1
    assert _json(run / "continuation_binding.json")["transferred_unopened_roles"] == {
        "development": cli.PATH_IDS["development"],
        "evaluation": cli.PATH_IDS["evaluation"],
        "replication": cli.PATH_IDS["replication"],
    }
    assert (run / "development/short/fused_family/family_summary.json").is_file()
    assert (run / "evaluation/full/fused_prefix/family_summary.json").is_file()
    assert (run / "evaluation/evaluation_family_join.json").is_file()
    suffix = _json(
        run / "evaluation/joined_suffix/fused_family/family_summary.json"
    )
    assert len(suffix["row_table"]) == 6
    assert {row["canonical_path_id"] for row in suffix["row_table"]} == {
        cli.PATH_IDS["evaluation"]
    }
    assert _json(run / "run_status.json")["state"] == "complete"


def test_fused_test_continuation_resume_is_exact_and_preflight_is_reverified(
    tmp_path: Path,
) -> None:
    args = _fused_fixture_args(tmp_path, "fused-resume")
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "fused-resume" / "runs")
    before = {
        path.relative_to(run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*")
        if path.is_file()
    }
    assert cli.main(
        [
            "--test-only",
            "--device",
            "cpu",
            "--stage",
            "all",
            "--resume-run-dir",
            str(run),
        ]
    ) == 0
    after = {
        path.relative_to(run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_fused_gain_row_keys_are_core_safe() -> None:
    from mnist.d0_jacobi_rb_tangent_fused import FusedRowSpec

    assert cli._fused_gain_token(0.5) == "0p5"
    plan = cli._fused_schedule_plan(test_only=False)
    assert "development-short-learned-0p5" in plan["development"]["row_order"]
    path_id, rows, _stream = cli._fused_profile_rows("reverse_p6")
    for key, kind, horizon, gain in rows:
        FusedRowSpec(key, path_id, kind, kind, horizon, gain, {})


def test_immutable_production_carrier_verifies_and_transfers_only_unopened_roles() -> None:
    carrier = Path(
        "runs/experiment12_d0_jacobi_rb_frequency1_rollout/"
        "20260812-005942_production-frequency1-exploratory-rollout-fbv2"
    )
    if not carrier.is_dir():
        pytest.skip("immutable rollout carrier is not present")
    before = {
        path.relative_to(carrier).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in carrier.rglob("*")
        if path.is_file()
    }
    binding = cli._verify_continuation_carrier(carrier, test_only=False)
    assert binding["realized_path_ids"] == [cli.PATH_IDS["preflight"]]
    assert binding["transferred_unopened_roles"] == {
        "development": cli.PATH_IDS["development"],
        "evaluation": cli.PATH_IDS["evaluation"],
        "replication": cli.PATH_IDS["replication"],
    }
    after = {
        path.relative_to(carrier).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in carrier.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_full_path_objective_has_precedence_over_short_suffix_direction() -> None:
    decision, action = cli._classify_evaluation_outcome(
        oracle_short=True,
        oracle_full=True,
        learned_short=False,
        learned_full=True,
        learned_short_ratio=1.0,
    )
    assert decision == "learned_full_dynamic_signal"
    assert "replication" in action

    mixed_oracle_decision, _ = cli._classify_evaluation_outcome(
        oracle_short=False,
        oracle_full=True,
        learned_short=False,
        learned_full=True,
        learned_short_ratio=1.0,
    )
    assert mixed_oracle_decision == "learned_full_dynamic_signal"


def test_replication_projection_uses_complete_operation_elapsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    zero_path = run / "evaluation/full/zero/trajectory_summary.json"
    learned_path = run / "evaluation/full/learned-gain-1/trajectory_summary.json"
    zero_path.parent.mkdir(parents=True)
    learned_path.parent.mkdir(parents=True)
    zero_path.write_text("{}", encoding="utf-8")
    learned_path.write_text("{}", encoding="utf-8")

    forward = {
        "end_to_end_elapsed_seconds": 1.0,
        "health": {"elapsed_seconds": 0.1},
    }
    summaries = {
        zero_path: {
            "end_to_end_elapsed_seconds": 2.0,
            "health": {"elapsed_seconds": 0.2},
        },
        learned_path: {
            "end_to_end_elapsed_seconds": 3.0,
            "health": {"elapsed_seconds": 0.3},
        },
    }
    monkeypatch.setattr(cli, "_verify_forward_summary", lambda *_: forward)
    monkeypatch.setattr(
        cli,
        "_verify_trajectory_summary",
        lambda path, *_: summaries[path],
    )

    record = cli._replication_capacity_record(run, test_only=False)
    assert record["projected_seconds"] == 6.0


def test_resource_usage_keeps_resumed_committed_shard_elapsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    forward_path = run / "forward/development/forward_summary.json"
    reverse_path = run / "development/short/zero/trajectory_summary.json"
    forward_path.parent.mkdir(parents=True)
    reverse_path.parent.mkdir(parents=True)
    forward_path.write_text("{}", encoding="utf-8")
    reverse_path.write_text("{}", encoding="utf-8")
    values = {
        forward_path: {
            "role": "development",
            "end_to_end_elapsed_seconds": 70.0,
            "health": {"elapsed_seconds": 60.0, "transition_count": 7},
        },
        reverse_path: {
            "role": "development",
            "horizon": "short",
            "variant": "zero",
            "gain": None,
            "end_to_end_elapsed_seconds": 5.0,
            "health": {"elapsed_seconds": 80.0, "transition_count": 11},
            "metrics": {
                "final": {
                    "squared_l2_error": 1.0,
                    "l1_error": 1.0,
                    "total_variation_distance": 0.5,
                    "centered_contrast_correlation": 0.0,
                }
            },
            "diagnostics": {},
            "final_state_sha256": "state",
        },
    }
    monkeypatch.setattr(cli, "_load_semantic", lambda path, *_: values[path])

    usage = cli._resource_usage(run)
    assert usage["elapsed_seconds"] == 150.0
    assert usage["transition_count"] == 18
    assert cli._trajectory_summaries(run)[0]["elapsed_seconds"] == 80.0


def test_replication_recomputes_resource_and_scientific_authority(
    tmp_path: Path,
) -> None:
    args = _fixture_args(tmp_path, "replication-binding")
    args[args.index("all")] = "preflight"
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "replication-binding" / "runs")
    resumed = [
        "--test-only",
        "--device",
        "cpu",
        "--resume-run-dir",
        str(run),
        "--stage",
        "forward",
    ]
    for stage in ("forward", "development", "evaluation"):
        resumed[-1] = stage
        assert cli.main(resumed) == 0
    assert not (run / "forward/replication").exists()

    stage_args = cli.parse_args(
        [
            "--test-only",
            "--device",
            "cpu",
            "--stage",
            "replication",
            "--resume-run-dir",
            str(run),
        ]
    )
    projection_path = run / "metrics/replication_resource_projection.json"
    original_projection = _json(projection_path)
    changed_projection = {
        key: value
        for key, value in original_projection.items()
        if key != "semantic_sha256"
    }
    changed_projection["passed"] = 0
    cli.atomic_write_json(projection_path, cli._semantic(changed_projection))
    with pytest.raises(cli.ArtifactCompatibilityError, match="projection binding"):
        cli._replication_stage(run, stage_args)

    cli.atomic_write_json(projection_path, original_projection)
    decision_path = run / "exploratory_decision.json"
    changed_decision = {
        key: value for key, value in _json(decision_path).items() if key != "semantic_sha256"
    }
    changed_decision["replication_authorized"] = 0
    cli.atomic_write_json(decision_path, cli._semantic(changed_decision))
    with pytest.raises(cli.ArtifactCompatibilityError, match="authorization changed"):
        cli._replication_stage(run, stage_args)
    assert not (run / "forward/replication").exists()


def test_reverse_execution_failure_preserves_last_valid_state_and_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mnist import d0_jacobi_rb_tangent_rollout as rollout

    run = tmp_path / "run"
    source = np.zeros(784, dtype=np.float64)
    source[0] = 1.0
    target = np.full(784, 0.35 / 784.0, dtype=np.float64)
    target[0] += 0.65
    anchor = target.copy()
    monkeypatch.setattr(cli, "_source_arrays", lambda *_: (source, target))
    monkeypatch.setattr(cli, "_anchor_state", lambda *_: anchor)
    monkeypatch.setattr(cli, "_ensure_execution_budget", lambda *_, **__: {})

    def fail_reverse(*_args: object, **_kwargs: object) -> object:
        raise rollout.TangentRolloutContractError(
            "controller logistic shift must be finite"
        )

    monkeypatch.setattr(rollout, "run_reverse_trajectory", fail_reverse)
    args = argparse.Namespace(test_only=False, device="cpu", test_oracle_fail=False)
    with pytest.raises(
        rollout.TangentRolloutContractError, match="logistic shift must be finite"
    ):
        cli._trajectory(
            run,
            args,
            role="development",
            horizon="short",
            anchor_step=7,
            variant="zero",
        )

    failure_path = run / "development/short/zero/trajectory_failure.json"
    failure = _json(failure_path)
    assert failure["exception_type"] == "TangentRolloutContractError"
    assert failure["committed_shard_count"] == 0
    assert (run / failure["last_valid_state_artifact"]["path"]).is_file()
    assert (run / failure["last_valid_images"]["last_valid"]["raw"]).is_file()
    assert not (run / "development/short/zero/trajectory_summary.json").exists()


def test_fake_sampler_all_writes_objective_artifacts_and_claim_boundary(
    tmp_path: Path,
) -> None:
    assert cli.main(_fixture_args(tmp_path, "positive")) == 0
    run = _only_run(tmp_path / "positive" / "runs")
    status = _json(run / "run_status.json")
    decision = _json(run / "exploratory_decision.json")
    selection = _json(run / "development/development_selection.json")
    assert status["state"] == "complete"
    assert status["objective_bearing_experiment"] == 1
    assert decision["decision"] == "learned_full_replication_agrees"
    assert decision["validation_pass_claim_authorized"] == 0
    assert selection["selected_gain"] == 2.0
    assert selection["committed_before_evaluation"] == 1
    assert selection["evaluation_evidence_opened"] == 0
    assert (run / "evaluation/full/zero/selected_states.npz").is_file()
    assert list((run / "evaluation/full").glob("learned-gain-*/selected_states.npz"))
    assert (run / "evaluation/full/oracle/selected_states.npz").is_file()
    assert (run / "images/short_contact_sheet.png").stat().st_size > 0
    assert (run / "images/full_contact_sheet.png").stat().st_size > 0
    assert (run / "images/trajectory_contact_sheet.png").stat().st_size > 0
    assert (run / "REPORT.md").stat().st_size > 0
    assert (run / "HANDOFF.md").stat().st_size > 0
    manifest = _json(run / "artifact_manifest.json")
    assert manifest["artifact_count"] > 0
    handoff = (run / "HANDOFF.md").read_text(encoding="utf-8")
    assert (
        f"Expected final manifest artifact count: {manifest['artifact_count']}."
        in handoff
    )
    audit = _json(run / "bundle_integrity_audit.json")
    assert audit["artifact_manifest_artifact_count"] == manifest["artifact_count"]
    assert audit["artifact_manifest_semantic_sha256"] == manifest["semantic_sha256"]
    assert audit["representative_npz_opened_and_hashed"]
    assert (run / "SHA256SUMS.txt").stat().st_size > 0
    assert not (run / "confirmation").exists()


def test_evaluation_forward_is_not_realized_before_committed_selection(
    tmp_path: Path,
) -> None:
    args = _fixture_args(tmp_path, "ordering")
    # Stages are operationally separate.  Preflight first, then stop after
    # forward: only development evidence may exist.
    stage_index = args.index("all")
    args[stage_index] = "preflight"
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "ordering" / "runs")
    resumed = [
        "--test-only",
        "--device",
        "cpu",
        "--stage",
        "forward",
        "--resume-run-dir",
        str(run),
    ]
    assert cli.main(resumed) == 0
    assert (run / "forward/development/anchors.npz").is_file()
    assert not (run / "forward/evaluation").exists()

    resumed[resumed.index("forward")] = "development"
    assert cli.main(resumed) == 0
    assert (run / "development/development_selection.json").is_file()
    assert not (run / "forward/evaluation").exists()

    resumed[resumed.index("development")] = "evaluation"
    assert cli.main(resumed) == 0
    assert (run / "forward/evaluation/anchors.npz").is_file()


def test_failed_oracle_gate_saves_images_report_and_never_opens_evaluation(
    tmp_path: Path,
) -> None:
    assert cli.main(_fixture_args(tmp_path, "oracle-fail", "--test-oracle-fail")) == 0
    run = _only_run(tmp_path / "oracle-fail" / "runs")
    decision = _json(run / "exploratory_decision.json")
    gate = _json(run / "development_gate.json")
    assert decision["decision"] == "development_oracle_control_failed"
    assert gate["passed"] == 0
    assert decision["evaluation_performed"] == 0
    assert not (run / "forward/evaluation").exists()
    assert not (run / "evaluation").exists()
    assert list((run / "images/individual").glob("development-short-*-final-*.png"))
    sheet = run / "images/development_short_contact_sheet.png"
    assert sheet.stat().st_size > 0
    from PIL import Image

    with Image.open(sheet) as image:
        assert image.size == (6 * 28 * 5, 28 * 5 + 22)
    assert (run / "REPORT.md").stat().st_size > 0
    assert (run / "artifact_manifest.json").is_file()


def test_resource_stop_report_is_explicit_and_uses_current_path_allocation(
    tmp_path: Path,
) -> None:
    args = _fixture_args(tmp_path, "resource-report")
    args[args.index("all")] = "preflight"
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "resource-report" / "runs")
    cli._failure(
        run,
        "preflight",
        cli.RolloutCLIError(
            "projected exact main rollout exceeds the frozen resource budget",
            failure_domain="resource_budget",
            failure_code="rollout_main_workflow_computationally_infeasible",
        ),
    )
    decision = _json(run / "exploratory_decision.json")
    report = (run / "REPORT.md").read_text(encoding="utf-8")
    handoff = (run / "HANDOFF.md").read_text(encoding="utf-8")
    assert "cross-variant fused" in decision["recommended_next_action"]
    assert "No reverse-control effect was evaluated" in handoff
    assert "Preflight complete-phase repeat rates" in report
    assert f"0x{cli.PATH_IDS['development']:X}" in handoff
    assert "0xFA100" not in handoff


def test_fake_resume_is_idempotent_at_committed_stage_boundaries(tmp_path: Path) -> None:
    args = _fixture_args(tmp_path, "resume")
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "resume" / "runs")
    before = {
        path.relative_to(run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*")
        if path.is_file()
    }
    resumed = [
        "--test-only",
        "--device",
        "cpu",
        "--stage",
        "all",
        "--resume-run-dir",
        str(run),
    ]
    assert cli.main(resumed) == 0
    after = {
        path.relative_to(run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*")
        if path.is_file()
    }
    assert after == before
    # A REPORT-only interruption is repairable: finalization recreates the
    # excluded non-circular audit and leaves a verifiable manifest/checksum.
    (run / "bundle_integrity_audit.json").unlink()
    report_resume = [
        "--test-only",
        "--device",
        "cpu",
        "--stage",
        "report",
        "--resume-run-dir",
        str(run),
    ]
    assert cli.main(report_resume) == 0
    assert (run / "bundle_integrity_audit.json").is_file()
    cli._verify_artifact_manifest(run)


def test_real_source_fixture_uses_semantic_measure_hash_not_raw_bytes() -> None:
    source_run = Path(
        "runs/experiment12_d0_jacobi_rb_coarse_residual_learnability/"
        "20260731-140333_production-exact-k512-coarse-residual-one-image"
    )
    if not source_run.is_dir():
        pytest.skip("production source fixture is not present")
    verified = load_verified_source_target(source_run)
    assert source_measure_sha256(verified.source_image) == cli.SOURCE_IMAGE_SHA256
    assert source_measure_sha256(verified.mixed_target) == cli.MIXED_TARGET_SHA256
    # Raw array commitments are intentionally different semantics.
    assert cli._array_sha256(verified.source_image) != cli.SOURCE_IMAGE_SHA256
    assert cli._array_sha256(verified.mixed_target) != cli.MIXED_TARGET_SHA256


def test_actual_parent_role_slots_do_not_create_stop_endpoint_false_collisions(
    tmp_path: Path,
) -> None:
    run_history = Path("runs")
    if not run_history.is_dir():
        pytest.skip("production run history is not present")
    current_run = run_history / (
        "experiment12_d0_jacobi_rb_frequency1_rollout/"
        "20260812-005942_production-frequency1-exploratory-rollout-fbv2"
    )
    if not current_run.is_dir():
        pytest.skip("current rollout fixture is not present")
    # Exclude the historical carrier itself.  A newer immutable run may now
    # contribute genuine role claims, but no range ``stop_exclusive`` value may
    # be misread as a path identity.
    record = cli._path_collision_record(Path.cwd(), current_run)
    assert all(
        "stop_exclusive" not in str(row["name"]).lower()
        for row in record["collisions"]
    )
    assert all(
        set(row["path_ids"]).issubset(set(cli.PATH_IDS.values()))
        for row in record["collisions"]
    )


def test_health_normalizes_core_cuda_memory_key_and_alias_counters() -> None:
    diagnostics = {
        "transition_count": 100,
        "certified_count": 100,
        "certificate_fraction": 1.0,
        "fallback_count": 0,
        "fallback_seconds": 0.0,
        "elapsed_seconds": 0.01,
        "transitions_per_second": 10_000.0,
        "maximum_simplex_mass_error": 1.0e-15,
        "maximum_pair_mass_error": 1.0e-15,
        "peak_cuda_memory_allocated_bytes": 40,
        "total_cuda_memory_bytes": 100,
        "reference_forbidden_counts": {"floor_count": 0},
        "forbidden_counts": {"floor_count": 0},
        "floor_count": 0,
    }
    record = cli._health_record(
        diagnostics,
        expected_transition_count=100,
        test_only=False,
    )
    assert record["passed"] == 1
    assert record["peak_memory_fraction"] == 0.4
    assert record["forbidden_counts"]["floor_count"] == 0
    clipped = cli._health_record(
        {**diagnostics, "clipping_count": 1},
        expected_transition_count=100,
        test_only=False,
    )
    assert clipped["passed"] == 0
    assert clipped["checks"]["forbidden_events"] is False

    structural = cli._health_record(
        {
            **diagnostics,
            "transition_count": 100,
            "active_count": 96,
            "certified_count": 96,
            "certificate_fraction": 1.0,
        },
        expected_transition_count=100,
        test_only=False,
    )
    assert structural["passed"] == 1
    assert structural["structural_noop_count"] == 4
    unauthorized = cli._health_record(
        {**diagnostics, "active_count": 96, "certified_count": 95},
        expected_transition_count=100,
        test_only=False,
    )
    assert unauthorized["passed"] == 0
    assert unauthorized["checks"]["authorization_counts"] is False


def test_production_shaped_forward_anchor_records_slow_rate_without_gating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mnist import d0_jacobi_rb_tangent_rollout as rollout

    transition_count = 512 * 7 * 392
    observed_rate = 595.26
    diagnostics = {
        "transition_count": transition_count,
        "active_count": transition_count,
        "structural_noop_count": 0,
        "authorized_count": transition_count,
        "authorization_fraction": 1.0,
        "certified_count": transition_count,
        "certificate_fraction": 1.0,
        "fallback_count": 0,
        "fallback_fraction": 0.0,
        "fallback_seconds": 0.0,
        "fallback_time_fraction": 0.0,
        "elapsed_seconds": transition_count / observed_rate,
        "transitions_per_second": observed_rate,
        "maximum_simplex_mass_error": 1.0e-15,
        "maximum_pair_mass_error": 1.0e-15,
        "peak_cuda_memory_allocated_bytes": 40,
        "total_cuda_memory_bytes": 100,
        "forbidden_counts": {},
    }
    target = np.full(784, 1.0 / 784.0, dtype=np.float64)

    def completed_anchor(*_args: object, **_kwargs: object) -> object:
        return type(
            "CompletedForwardAnchor",
            (),
            {
                "diagnostics": diagnostics,
                "anchors": {
                    cli.SHORT_ANCHOR: target[None, :].copy(),
                    cli.FULL_ANCHOR: target[None, :].copy(),
                },
                "transition_count": transition_count,
            },
        )()

    monkeypatch.setattr(rollout, "run_forward_trajectory", completed_anchor)
    _short, _full, record = cli._fused_preflight_forward_anchor(
        tmp_path,
        target=target,
        device=cli.torch.device("cpu"),
        profile=cli.JacobiRBCudaProfile(),
    )
    health = record["health"]
    assert record["passed"] == 1
    assert record["execution_role"] == "untimed_forward_anchor_infrastructure"
    assert record["throughput_gate_applied"] == 0
    assert health["transition_count"] == transition_count
    assert health["active_count"] == transition_count
    assert health["authorized_count"] == transition_count
    assert health["throughput_gate_applied"] == 0
    assert health["throughput_observation_meets_minimum"] == 0
    assert health["transitions_per_second"] == observed_rate

    timed = cli._health_record(
        diagnostics,
        expected_transition_count=transition_count,
        test_only=False,
    )
    assert timed["throughput_gate_applied"] == 1
    assert timed["checks"]["throughput"] is False
    assert timed["passed"] == 0
    assert cli._profile_health_classification(timed) == "resource"
    assert cli._profile_health_classification(
        {**timed, "checks": {**timed["checks"], "transition_count": False}}
    ) == "integrity"

    repeats = {
        name: [
            {
                "elapsed_seconds": (
                    count / observed_rate if name == "forward_p1" else count / 2_000.0
                ),
                "transition_count": count,
                "output_sha256": f"{name}-stable",
                "health_passed": 0 if name == "forward_p1" else 1,
            }
            for _ in range(3)
        ]
        for name, count in cli.FUSED_PROFILE_TRANSITIONS.items()
    }
    projection = cli._fused_resource_projection(repeats)
    assert projection["profiles"]["forward_p1"]["passed"] == 0
    assert projection["checks"]["individual_profile_rates_and_health"] is False
    assert projection["passed"] == 0


def test_endpoint_table_reads_production_shaped_score_and_wrapper_telemetry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evaluation/full/learned/trajectory_summary.json"
    path.parent.mkdir(parents=True)
    record = cli._semantic(
        {
            "schema": "production-shaped-reverse-trajectory-summary",
            "role": "evaluation",
            "horizon": "full",
            "variant": "learned",
            "gain": 2.0,
            "final_state_sha256": "a" * 64,
            "metrics": {
                "final": {
                    "squared_l2_error": 0.1,
                    "l1_error": 0.2,
                    "total_variation_distance": 0.1,
                    "centered_contrast_correlation": 0.3,
                }
            },
            "diagnostics": {
                "score_rms": 1.25,
                "score_maximum_absolute": 2.5,
                "logistic_shift_rms": 0.125,
                "logistic_shift_maximum_absolute": 0.25,
                "control_reference_displacement_ratio": 0.2,
                "controller": {
                    "unscaled_score_rms": 0.625,
                    "scaled_score_rms": 1.25,
                },
            },
            "health": {"certificate_fraction": 1.0},
        }
    )
    path.write_text(json.dumps(record), encoding="utf-8")
    row = cli._trajectory_summaries(tmp_path)[0]
    assert row["score_rms"] == 1.25
    assert row["logistic_shift_rms"] == 0.125
    assert row["unscaled_score_rms"] == 0.625
    assert row["scaled_score_rms"] == 1.25


def test_tampered_copied_source_is_rejected_before_resume_mutation(tmp_path: Path) -> None:
    args = _fixture_args(tmp_path, "input-tamper")
    args[args.index("all")] = "preflight"
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "input-tamper" / "runs")
    source_path = run / "input_bindings/source_image.npz"
    source_path.write_bytes(source_path.read_bytes() + b"tamper")
    before = {
        path.relative_to(run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*")
        if path.is_file()
    }
    assert cli.main(
        [
            "--test-only",
            "--device",
            "cpu",
            "--stage",
            "forward",
            "--resume-run-dir",
            str(run),
        ]
    ) == 2
    after = {
        path.relative_to(run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_noncanonical_historical_path_plan_collision_is_detected(tmp_path: Path) -> None:
    evaluation_id = cli.PATH_IDS["evaluation"]
    plan = tmp_path / "runs/history/haar_path_id_plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(
        json.dumps(
            {
                "selected": {
                    "realized": {
                        "start": evaluation_id,
                        "stop_exclusive": evaluation_id + 1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    record = cli._path_collision_record(tmp_path, tmp_path / "fresh")
    assert record["passed"] == 0
    assert any(evaluation_id in row["path_ids"] for row in record["collisions"])


def test_transitive_source_closure_includes_and_binds_indirect_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = cli._source_closure()
    names = {row["path"] for row in original["files"]}
    assert "mnist/d0_jacobi_rb_boundary_tangent_fused.py" in names
    assert "mnist/d0_jacobi_rb_cuda_deferred.py" in names
    assert "mnist/d0_jacobi_rb_tangent_fused.py" in names
    assert "mnist/d0_jacobi_rb_absolute_coordinate.py" in names
    assert "mnist/d0_jacobi_rb_coarse_residual.py" in names
    assert "mnist/d0_jacobi_rb_spectral.py" in names
    fingerprint = cli.file_fingerprint

    def changed(path: Path) -> str:
        if Path(path).name == "d0_jacobi_rb_spectral.py":
            return "0" * 64
        return fingerprint(path)

    monkeypatch.setattr(cli, "file_fingerprint", changed)
    tampered = cli._source_closure()
    assert tampered["source_fingerprint"] != original["source_fingerprint"]


def test_real_immutable_recovery_predecessor_passes_top_level_precheck(
    tmp_path: Path,
) -> None:
    """The sealed registry binds source through run_manifest, not itself."""

    predecessor = (
        Path(__file__).resolve().parent.parent
        / "runs/experiment12_d0_jacobi_rb_frequency1_rollout"
        / cli.RECOVERY_PREDECESSOR_BASENAME
    )
    assert predecessor.is_dir()
    manifest = _json(predecessor / "artifact_manifest.json")
    assert "source_fingerprint" not in manifest
    run_manifest = _json(predecessor / "run_manifest.json")
    assert (
        run_manifest["source_fingerprint"]
        == cli.RECOVERY_PREDECESSOR_SOURCE_FINGERPRINT
    )
    args = argparse.Namespace(
        predecessor_run_dir=predecessor,
        test_only=False,
        runs_root=tmp_path / "unused-child-root",
        resume_run_dir=None,
    )
    cli._precheck_recovery_predecessor(args)
    assert not (tmp_path / "unused-child-root").exists()


def test_predecessor_precheck_rejects_run_manifest_not_bound_by_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predecessor = (
        Path(__file__).resolve().parent.parent
        / "runs/experiment12_d0_jacobi_rb_frequency1_rollout"
        / cli.RECOVERY_PREDECESSOR_BASENAME
    )
    args = argparse.Namespace(
        predecessor_run_dir=predecessor,
        test_only=False,
        runs_root=tmp_path / "unused-child-root",
        resume_run_dir=None,
    )
    fingerprint = cli.file_fingerprint

    def changed(path: Path) -> str:
        if Path(path).name == "run_manifest.json":
            return "0" * 64
        return fingerprint(path)

    monkeypatch.setattr(cli, "file_fingerprint", changed)
    with pytest.raises(cli.ArtifactCompatibilityError, match="top-level binding"):
        cli._precheck_recovery_predecessor(args)
    assert not (tmp_path / "unused-child-root").exists()


def test_fused_seed_map_is_built_once_outside_profile_and_family_timers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mnist import d0_jacobi_rb_tangent_fused as fused

    calls = {"map_builds": 0, "references": 0}
    shared_map = {"frozen-role": object()}

    def build_map(**_kwargs: object) -> dict[str, object]:
        calls["map_builds"] += 1
        return shared_map

    class FakeReference:
        def __init__(self, **kwargs: object) -> None:
            calls["references"] += 1
            assert kwargs["prepared_rng_seeds"] is shared_map

    monkeypatch.setattr(fused, "prepare_deferred_reference_rng_seed_map", build_map)
    monkeypatch.setattr(fused, "DeferredCertifiedFusedReference", FakeReference)
    cli._PREPARED_FUSED_SEED_MAPS.clear()
    backend = object()
    profile = cli.JacobiRBCudaProfile()
    factory = cli._fused_reference_factory(
        prepared=backend,
        profile=profile,
        stream_role="profile-seed-map-fixture",
    )
    assert calls == {"map_builds": 1, "references": 0}
    factory(0)
    factory(1)
    reused = cli._fused_reference_factory(
        prepared=backend,
        profile=profile,
        stream_role="profile-seed-map-fixture",
    )
    reused(2)
    assert calls == {"map_builds": 1, "references": 3}

    profile_source = inspect.getsource(cli._fused_preflight_reverse_repeat)
    assert profile_source.index("prepared_reference_factory =") < profile_source.index(
        "started = time.perf_counter()"
    )
    factory_source = profile_source[
        profile_source.index("def factory(") : profile_source.index("common = dict(")
    ]
    assert "_fused_reference_factory(" not in factory_source
    warmup_source = inspect.getsource(cli._fused_preflight_warmup)
    assert warmup_source.index("reference_factory = _fused_reference_factory(") < (
        warmup_source.index("started = time.perf_counter()")
    )
    family_source = inspect.getsource(cli._run_and_commit_fused_family)
    assert family_source.index("reference_factory = _fused_reference_factory(") < (
        family_source.index("started = time.perf_counter()")
    )


def test_fused_committed_prefix_is_verified_before_resume_mutation(
    tmp_path: Path,
) -> None:
    from mnist.d0_jacobi_rb_tangent_fused import FusedRowSpec

    root = tmp_path / "family"
    sequence = cli._reverse_sequence(7)
    specs = (
        FusedRowSpec("row-zero", 17, "zero", "zero", "short"),
        FusedRowSpec("row-learned", 17, "learned", "learned", "short", 1.0),
    )
    controller = {"row_table": [row.to_record() for row in specs]}
    rng = {"root_seed": 123, "stream_role": "prefix-test", "variant_in_rng_key": 0}
    state = np.full((2, 784), 1.0 / 784.0, dtype=np.float64)
    artifact = cli._atomic_npz(root / "shard-0000.npz", state=state)
    transition_count = len(sequence) * 4 * len(specs) * 392
    row_transitions = transition_count // len(specs)
    telemetry_rows: list[dict[str, object]] = []
    controller_rows: list[dict[str, object]] = []
    for spec in specs:
        telemetry, _expected = _load_bearing_fused_telemetry_fixture(
            "certified_exact",
            row_key=spec.row_key,
            controller_kind=spec.controller_kind,
            gain=spec.gain,
            transition_count=row_transitions,
        )
        telemetry_rows.append(telemetry["per_row_diagnostics"][0])
        controller_rows.append(telemetry["controller_diagnostics"][0])
    execution = {
        "schema": "test-execution-plan",
        "shard_index": 0,
        "sequence": [list(item) for item in sequence],
        "row_count": len(specs),
        "transition_count": transition_count,
        "input_state_sha256": cli._core_array_sha256(state),
    }
    record = cli._semantic(
        {
            "schema": "test-fused-prefix",
            "schema_version": 1,
            "family_name": "family",
            "segment_name": "complete",
            "shard_index": 0,
            "sequence_start": list(sequence[0]),
            "sequence_end": list(sequence[-1]),
            "sequence_sha256": cli.config_fingerprint([list(item) for item in sequence]),
            "row_table": [row.to_record() for row in specs],
            "row_keys": [row.row_key for row in specs],
            "canonical_path_ids": [17, 17],
            "microsteps": cli.MICROSTEPS,
            "label": 3,
            "input_state_sha256": cli._core_array_sha256(state),
            "controller_binding_sha256": cli.config_fingerprint(controller),
            "rng_binding_sha256": cli.config_fingerprint(rng),
            "variant_in_rng_key": 0,
            "execution_plan": execution,
            "output_state_sha256": cli._core_array_sha256(state),
            "state_file_sha256": artifact["sha256"],
            "state_file_size": artifact["size"],
            "elapsed_seconds": 1.0,
            "transition_count": transition_count,
                "per_row_diagnostics": telemetry_rows,
                "controller_diagnostics": controller_rows,
            "diagnostics": {
                "transition_count": transition_count,
                "maximum_launch_lanes": 2 * 392,
                "certificate_fraction": 1.0,
                "maximum_mass_error": 0.0,
                "reference": {
                    "transition_count": transition_count,
                "active_count": transition_count,
                "structural_noop_count": 0,
                "certified_count": transition_count,
                    "fallback_count": 0,
                    "certificate_fraction": 1.0,
                    "maximum_cuda_memory_allocated": 0,
                    "total_cuda_memory_bytes": 0,
                    "forbidden_counts": {
                    "resource_cap_count": 0,
                    "invalid_density_count": 0,
                    "approximation_count": 0,
                    "clipping_count": 0,
                    "correction_count": 0,
                    "floor_count": 0,
                    "limiter_count": 0,
                    "projection_count": 0,
                    "renormalization_count": 0,
                    "nonfinite_count": 0,
                },
                },
            },
            "committed": 1,
        }
    )
    cli.atomic_write_json(root / "shard-0000.json", record)
    verified = cli._verify_fused_family_prefix(
        root,
        initial_state=state,
        sequence=sequence,
        row_specs=specs,
        controller_binding=controller,
        rng_binding=rng,
        family_name="family",
        segment_name="complete",
    )
    assert verified["transition_count"] == transition_count

    changed = json.loads(json.dumps(
        {key: value for key, value in record.items() if key != "semantic_sha256"}
    ))
    changed["rng_binding_sha256"] = "0" * 64
    cli.atomic_write_json(root / "shard-0000.json", cli._semantic(changed))
    with pytest.raises(cli.ArtifactCompatibilityError, match="prefix changed"):
        cli._verify_fused_family_prefix(
            root,
            initial_state=state,
            sequence=sequence,
            row_specs=specs,
            controller_binding=controller,
            rng_binding=rng,
            family_name="family",
            segment_name="complete",
        )


def test_mid_objective_resource_stop_is_not_complete_scientific_evidence(
    tmp_path: Path,
) -> None:
    args = _fused_fixture_args(tmp_path, "mid-resource-stop")
    args[args.index("all")] = "preflight"
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "mid-resource-stop" / "runs")
    cli.atomic_write_json(
        run / "development/short/example/trajectory_failure.json",
        cli._semantic(
            {
                "schema": "test-interpretable-objective-failure",
                "last_valid_state_sha256": "1" * 64,
                "fixed_scale_image": "images/example.png",
                "numerically_interpretable_partial_evidence": 1,
            }
        ),
    )
    cli._failure(
        run,
        "development",
        cli.RolloutCLIError(
            "next shard exceeds the frozen cap",
            failure_domain="resource_budget",
            failure_code="rollout_resource_budget_exhausted",
        ),
    )
    assert _json(run / "failure.json")["scientific_evidence_complete"] == 0
    assert _json(run / "run_status.json")["scientific_evidence_complete"] == 0
    handoff = (run / "HANDOFF.md").read_text(encoding="utf-8")
    assert "opened that milestone but stopped at a verified atomic shard boundary" in handoff
    assert "This preflight stopped before that milestone" not in handoff
    assert "Proxy-only patches since the last objective-bearing experiment: 0" in handoff


def test_uninterpretable_partial_failure_does_not_reset_proxy_counter(
    tmp_path: Path,
) -> None:
    args = _fused_fixture_args(tmp_path, "uninterpretable-resource-stop")
    args[args.index("all")] = "preflight"
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "uninterpretable-resource-stop" / "runs")
    cli.atomic_write_json(
        run / "development/short/example/trajectory_failure.json",
        cli._semantic(
            {
                "schema": "test-uninterpretable-objective-failure",
                "last_valid_state_sha256": "2" * 64,
                "numerically_interpretable_partial_evidence": 0,
            }
        ),
    )
    cli._failure(
        run,
        "development",
        cli.RolloutCLIError(
            "next shard failed numerical interpretation",
            failure_domain="resource_budget",
            failure_code="rollout_resource_budget_exhausted",
        ),
    )
    handoff = (run / "HANDOFF.md").read_text(encoding="utf-8")
    assert (
        "Proxy-only patches since the last objective-bearing experiment: at least one"
        in handoff
    )
    assert "interpretable failure states/images" not in handoff


@pytest.mark.parametrize("stage", ["forward", "development"])
def test_objective_stage_resource_precheck_does_not_claim_partial_trajectory(
    tmp_path: Path, stage: str
) -> None:
    name = f"{stage}-resource-precheck"
    args = _fused_fixture_args(tmp_path, name)
    args[args.index("all")] = "preflight"
    assert cli.main(args) == 0
    run = _only_run(tmp_path / name / "runs")
    cli._failure(
        run,
        stage,
        cli.RolloutCLIError(
            "the next operation does not fit the frozen cap",
            failure_domain="resource_budget",
            failure_code="rollout_resource_budget_exhausted",
        ),
    )
    handoff = (run / "HANDOFF.md").read_text(encoding="utf-8")
    assert "objective-stage resource precheck" in handoff
    assert "before an interpretable reverse trajectory" in handoff
    assert "partial raw states and failure images were retained" not in handoff
    assert (
        "Proxy-only patches since the last objective-bearing experiment: at least one"
        in handoff
    )


def test_forward_integrity_failure_without_record_does_not_claim_objective_evidence(
    tmp_path: Path,
) -> None:
    args = _fused_fixture_args(tmp_path, "forward-integrity-precheck")
    args[args.index("all")] = "preflight"
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "forward-integrity-precheck" / "runs")
    cli._failure(
        run,
        "forward",
        cli.RolloutCLIError(
            "forward exact-health check failed before commit",
            failure_domain="execution_integrity",
            failure_code="forward_exact_health_invalid",
        ),
    )
    handoff = (run / "HANDOFF.md").read_text(encoding="utf-8")
    assert "integrity failure stopped it before endpoint or image evidence" in handoff
    assert "This run is that milestone" not in handoff
    assert "partial raw states and failure images were retained" not in handoff
    assert "No interpretable reverse trajectory, endpoint/image evidence" in handoff
    assert (
        "Proxy-only patches since the last objective-bearing experiment: at least one"
        in handoff
    )


def test_stage_all_failure_records_the_actual_preflight_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fused_fixture_args(tmp_path, "stage-all-preflight-resource")

    def fail_preflight(run: Path, _args: argparse.Namespace) -> dict[str, object]:
        cli.atomic_write_json(
            run / "preflight/resource_projection.json",
            cli._semantic(
                {
                    "schema": "test-fused-resource-projection",
                    "profiles": {},
                    "projected_main_wall_seconds": cli.MAXIMUM_MAIN_WALL_SECONDS + 1.0,
                    "maximum_main_wall_seconds": cli.MAXIMUM_MAIN_WALL_SECONDS,
                    "effective_rate": 1.0,
                }
            ),
        )
        cli.atomic_write_json(
            run / "preflight/objective_roles_unopened.json",
            cli._semantic(
                {
                    "schema": "test-objective-roles-unopened",
                    "passed": 1,
                }
            ),
        )
        raise cli.RolloutCLIError(
            "projected fused workflow exceeds the cap",
            failure_domain="resource_budget",
            failure_code="rollout_main_workflow_computationally_infeasible",
        )

    monkeypatch.setattr(cli, "_preflight_stage", fail_preflight)
    assert cli.main(args) == 2
    run = _only_run(tmp_path / "stage-all-preflight-resource" / "runs")
    failure = _json(run / "failure.json")
    assert failure["stage"] == "preflight"
    assert failure["scientific_evidence_complete"] == 1
    handoff = (run / "HANDOFF.md").read_text(encoding="utf-8")
    assert "This preflight stopped before that milestone" in handoff


def test_stage_all_preflight_integrity_failure_does_not_claim_objective_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _fused_fixture_args(tmp_path, "stage-all-preflight-integrity")

    def fail_preflight(_run: Path, _args: argparse.Namespace) -> dict[str, object]:
        raise cli.RolloutCLIError(
            "fused equivalence control failed",
            failure_domain="execution_integrity",
            failure_code="fused_equivalence_invalid",
        )

    monkeypatch.setattr(cli, "_preflight_stage", fail_preflight)
    assert cli.main(args) == 2
    run = _only_run(tmp_path / "stage-all-preflight-integrity" / "runs")
    failure = _json(run / "failure.json")
    assert failure["stage"] == "preflight"
    assert failure["scientific_evidence_complete"] == 0
    handoff = (run / "HANDOFF.md").read_text(encoding="utf-8")
    assert "stopped before that milestone during initialization/preflight" in handoff
    assert "No reverse-control trajectory or paired objective effect was evaluated" in handoff
    assert "This run is that milestone" not in handoff
    assert (
        "Proxy-only patches since the last objective-bearing experiment: at least one"
        in handoff
    )


def test_recovery_parser_freezes_objective_first_contract(tmp_path: Path) -> None:
    base = _recovery_fixture_args(tmp_path, "recovery-parser")
    parsed = cli.parse_args(base)
    assert parsed.reference_backend == "auto"
    assert parsed.core_learned_gain == 1.0
    assert parsed.exact_audit_outer_steps == 8
    for extra in (
        ("--stage", "preflight"),
        ("--require-gate", "evaluation"),
        ("--core-learned-gain", "2"),
        ("--exact-audit-outer-steps", "16"),
    ):
        with pytest.raises(SystemExit):
            cli.parse_args([*base, *extra])


def test_recovery_rejects_predecessor_output_before_mutation(tmp_path: Path) -> None:
    predecessor = tmp_path / "immutable"
    predecessor.mkdir()
    args = cli.parse_args(
        [
            "--test-only",
            "--device",
            "cpu",
            "--stage",
            "all",
            "--frequency1-run-dir",
            str(tmp_path / "parent"),
            "--source-run-dir",
            str(tmp_path / "source"),
            "--continuation-run-dir",
            str(tmp_path / "carrier"),
            "--predecessor-run-dir",
            str(predecessor),
            "--runs-root",
            str(predecessor),
        ]
    )
    before = list(predecessor.iterdir())
    assert cli.main(args.invoked_argv if hasattr(args, "invoked_argv") else [
        "--test-only", "--device", "cpu", "--stage", "all",
        "--frequency1-run-dir", str(tmp_path / "parent"),
        "--source-run-dir", str(tmp_path / "source"),
        "--continuation-run-dir", str(tmp_path / "carrier"),
        "--predecessor-run-dir", str(predecessor),
        "--runs-root", str(predecessor),
    ]) == 2
    assert list(predecessor.iterdir()) == before


def test_recovery_projection_has_only_total_time_storage_memory_gates() -> None:
    exact = cli._recovery_resource_projection(
        active_seconds=100.0,
        wasted_active_seconds=5.0,
        observed_shard_seconds=[10.0, 12.0],
        remaining_shards=15,
        maximum_main_seconds=621.0,
        persisted_bytes=100,
        projected_additional_bytes=200,
        peak_memory_fraction=0.5,
    )
    assert exact["projected_total_seconds"] == 105.0 + 1.2 * 12.0 * 15 + 300.0
    assert exact["passed"] == 1
    assert exact["minimum_rate_gate_present"] == 0
    assert exact["setup_only_veto_present"] == 0
    above = cli._recovery_resource_projection(
        active_seconds=100.0,
        wasted_active_seconds=5.0,
        observed_shard_seconds=[float(np.nextafter(12.0, np.inf))],
        remaining_shards=15,
        maximum_main_seconds=exact["projected_total_seconds"],
        persisted_bytes=100,
        projected_additional_bytes=200,
        peak_memory_fraction=0.5,
    )
    assert above["projected_total_seconds"] >= exact["projected_total_seconds"]


def test_recovery_auto_backend_switch_is_selection_not_terminal() -> None:
    fits = {"passed": 1}
    misses = {"passed": 0}
    assert cli._candidate_backend_selected(fits, "auto") == "exact"
    assert cli._candidate_backend_selected(misses, "auto") == "candidate"
    assert cli._candidate_backend_selected(misses, "candidate") == "candidate"
    with pytest.raises(cli.RolloutCLIError) as captured:
        cli._candidate_backend_selected(misses, "exact")
    assert captured.value.failure_code == "exact_core_resource_blocked_by_explicit_backend_choice"


def test_exact_candidate_audit_uses_rows_and_paired_contrasts() -> None:
    exact = np.full((3, 784), 1.0 / 784.0, dtype=np.float64)
    exact[1, 0] += 1.0e-4
    exact[1, 1] -= 1.0e-4
    exact[2, 2] += 2.0e-4
    exact[2, 3] -= 2.0e-4
    candidate = exact.copy()
    candidate[:, 4] += 1.0e-7
    candidate[:, 5] -= 1.0e-7
    record = cli._exact_candidate_audit_record(
        row_keys=("zero", "learned", "source-informed"),
        exact_state=exact,
        candidate_state=candidate,
        exact_reference_rms=(1.0, 1.0, 1.0),
        candidate_reference_rms=(1.1, 1.1, 1.1),
        exact_controller_rms=(0.0, 0.2, 0.3),
        candidate_controller_rms=(0.0, 0.21, 0.31),
    )
    assert len(record["row_metrics"]) == 3
    assert set(record["paired_contrasts"]) == {
        "learned_minus_zero",
        "source_informed_minus_zero",
    }
    assert record["blocks_candidate_artifact_completion"] == 0
    assert record["proves_full_horizon_exact_equivalence"] == 0


@pytest.mark.parametrize("force_candidate", [False, True])
def test_recovery_test_workflow_commits_core_before_optional_work_and_resumes_exactly(
    tmp_path: Path, force_candidate: bool
) -> None:
    name = "recovery-candidate" if force_candidate else "recovery-exact"
    args = _recovery_fixture_args(
        tmp_path,
        name,
        *(["--test-force-candidate"] if force_candidate else []),
    )
    assert cli.main(args) == 0
    run = _only_run(tmp_path / name / "runs")
    core = _json(run / "core_objective.json")
    backend = _json(run / "backend_decision.json")
    assert core["completed_128_step_three_row_family"] == 1
    assert backend["selected"] == ("candidate" if force_candidate else "exact")
    assert backend["exact_audit_shard_committed"] == 1
    assert (run / "objective/development-core-short/selected_states.npz").is_file()
    assert (run / "images/objective_core_contact_sheet.png").is_file()
    assert _json(run / "gain_expansion.json")["core_artifact_already_committed"] == 1
    evaluation = _json(run / "evaluation_plan.json")
    assert evaluation["source_informed_endpoint_did_not_gate_this_decision"] == 1
    assert _json(run / "terminal_outcome.json")["successful_terminal"] == 1
    if force_candidate:
        assert (run / "exact_candidate_audit.json").is_file()
        assert backend["candidate_restarted_from_original_anchor"] == 1
    before = {
        path.relative_to(run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*") if path.is_file()
    }
    resume = [
        "--test-only", "--device", "cpu", "--stage", "all",
        "--resume-run-dir", str(run),
    ]
    assert cli.main(resume) == 0
    after = {
        path.relative_to(run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*") if path.is_file()
    }
    assert after == before


def test_recovery_source_informed_diagnostic_never_becomes_integrity_gate(
    tmp_path: Path
) -> None:
    args = _recovery_fixture_args(tmp_path, "recovery-source-diagnostic")
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "recovery-source-diagnostic" / "runs")
    core = _json(run / "core_objective.json")
    assert core["source_informed_row_is_diagnostic_not_gate"] == 1
    assert _json(run / "evaluation_plan.json")[
        "source_informed_endpoint_did_not_gate_this_decision"
    ] == 1
    assert _json(run / "analytic_target_fraction_control.json")[
        "source_informed_mnist_row_part_of_this_gate"
    ] == 0


def test_candidate_core_resource_block_is_unsuccessful_and_incomplete(
    tmp_path: Path
) -> None:
    args = _recovery_fixture_args(tmp_path, "recovery-resource-block")
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "recovery-resource-block" / "runs")
    cli._failure_recovery(
        run,
        "objective",
        cli.RolloutCLIError(
            "candidate core cannot fit",
            failure_domain="resource_budget",
            failure_code="candidate_core_resource_blocked",
        ),
    )
    outcome = _json(run / "terminal_outcome.json")
    assert outcome["successful_terminal"] == 0
    assert outcome["decision"] == "candidate_core_resource_blocked"
    assert "same patch" in outcome["recommended_next_action"]


def test_recovery_mechanism_rows_reconstructs_nonzero_rms_from_sums() -> None:
    result = type(
        "Result",
        (),
        {
            "per_row_diagnostics": (
                {
                    "score_squared_sum": 18.0,
                    "score_count": 2,
                    "logistic_shift_squared_sum": 32.0,
                    "logistic_shift_count": 2,
                    "reference_fraction_displacement_squared_sum": 50.0,
                    "reference_fraction_displacement_count": 2,
                    "control_fraction_displacement_squared_sum": 8.0,
                    "control_fraction_displacement_count": 2,
                },
            )
        },
    )()
    row = cli._recovery_mechanism_rows(result)[0]
    assert row["score_rms"] == 3.0
    assert row["logistic_shift_rms"] == 4.0
    assert row["reference_fraction_displacement_rms"] == 5.0
    assert row["control_fraction_displacement_rms"] == 2.0
    assert row["control_reference_displacement_ratio"] == 0.4


def test_gain_candidate_audit_has_honest_row_only_semantics() -> None:
    exact = np.full((3, 784), 1.0 / 784.0, dtype=np.float64)
    candidate = exact.copy()
    candidate[:, 0] += 1.0e-7
    candidate[:, 1] -= 1.0e-7
    record = cli._exact_candidate_row_only_audit_record(
        row_keys=("gain-0p5", "gain-2", "gain-4"),
        exact_state=exact,
        candidate_state=candidate,
    )
    assert record["paired_zero_contrasts_not_applicable"] == 1
    assert "paired_contrasts" not in record


def test_optional_resource_stop_does_not_invalidate_completed_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _recovery_fixture_args(tmp_path, "optional-resource")
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "optional-resource" / "runs")

    def stop(*_args: object, **_kwargs: object) -> object:
        raise cli.RolloutCLIError(
            "optional family no longer fits",
            failure_domain="resource_budget",
            failure_code="optional_objective_resource_deferred",
        )

    monkeypatch.setattr(cli, "_run_recovery_core_backend", stop)
    assert cli._run_recovery_optional_backend(
        run,
        cli.parse_args(["--test-only", "--device", "cpu", "--stage", "all", "--resume-run-dir", str(run)]),
    ) is None
    assert _json(run / "core_objective.json")["completed_128_step_three_row_family"] == 1


def test_candidate_full_prelaunch_failure_marker_preserves_short_report_authority(
    tmp_path: Path,
) -> None:
    args = _recovery_fixture_args(tmp_path, "candidate-full-prelaunch")
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "candidate-full-prelaunch" / "runs")

    short_root = (
        run
        / "objective_attempts/exact/fused_families/evaluation-short/short"
    )
    state = np.full((3, 784), 1.0 / 784.0, dtype=np.float64)
    state_artifact = cli._atomic_npz(short_root / "shard-0000.npz", state=state)
    transition_count = 87_808
    per_row = [
        {
            "transition_count": transition_count,
            "active_count": transition_count,
            "certified_count": transition_count,
            "fallback_count": 0,
            "unauthorized_count": 0,
            "invalid_count": 0,
            "certificate_fraction": 1.0,
        }
        for _ in range(3)
    ]
    short_path = short_root / "shard-0000.json"
    cli.atomic_write_json(
        short_path,
        cli._semantic(
            {
                "committed": 1,
                "shard_index": 0,
                "state_file_sha256": state_artifact["sha256"],
                "state_file_size": state_artifact["size"],
                "execution_plan": {"sequence": [[127, 6]] * 56},
                "diagnostics": {
                    "maximum_mass_error": 0.0,
                    "reference": {
                        "forbidden_counts": {
                            name: 0
                            for name in cli.RECOVERY_EXACT_FORBIDDEN_COUNTS
                        },
                        "per_row": per_row,
                    },
                },
            }
        ),
    )

    failure_root = (
        run
        / "objective_attempts/candidate/fused_families/evaluation-full/full"
    )
    failure_path = failure_root / "shard-0000.failure.json"
    cli.atomic_write_json(
        failure_path,
        cli._semantic(
            {
                "schema": "test-production-shaped-prelaunch-failure",
                "family_name": "evaluation-full",
                "segment_name": "full",
                "shard_index": 0,
                "execution_plan": {"sequence": [[511, 6]] * 56},
                "failure_type": "RolloutCLIError",
                "failure_message": (
                    "candidate shard zero projection exceeds remaining main budget"
                ),
                "committed": 0,
            }
        ),
    )
    newer = short_path.stat().st_mtime_ns + 1_000_000_000
    os.utime(failure_path, ns=(newer, newer))
    assert not failure_path.with_suffix(".npz").exists()

    decision = "evaluation_short_rollout_direction_not_useful"
    action = "change the controller before scaling"
    cli.atomic_write_json(
        run / "evaluation_plan.json",
        cli._semantic(
            {
                "schema": cli.RECOVERY_RUN_SCHEMA + "-evaluation-plan",
                "schema_version": 1,
                "performed": 1,
                "anchor_mode": "fresh-forward",
                "selected_gain": 1.0,
                "short_horizon": {
                    "performed": 1,
                    "backend": "exact",
                    "decision": decision,
                    "recommended_next_action": action,
                    "final_squared_l2": {
                        "zero": 0.1,
                        "learned": 0.2,
                        "source_informed": 0.05,
                    },
                    "learned_minus_zero_risk_improvement": -0.1,
                },
                "full_horizon": {
                    "performed": 0,
                    "reason": "full_selected_backend_resource_deferred",
                    "partial_optional_evidence_preserved": 1,
                },
                "short_decision": decision,
                "decision": decision,
                "recommended_next_action": action,
                "optional_future_fb300_unopened": 1,
            }
        ),
    )
    terminal = _json(run / "terminal_outcome.json")
    terminal.pop("semantic_sha256")
    terminal.update(
        {
            "decision": decision,
            "evaluation_performed": 1,
            "evaluation_decision": decision,
            "recommended_next_action": action,
            "successful_terminal": 1,
        }
    )
    cli.atomic_write_json(run / "terminal_outcome.json", cli._semantic(terminal))

    partial = cli._recovery_partial_evidence_summary(run)
    assert partial["committed_shard_count"] == 1
    assert partial["latest_shard_path"] == short_path.relative_to(run).as_posix()
    assert partial["latest_state_path"] == (
        short_root / "shard-0000.npz"
    ).relative_to(run).as_posix()
    assert partial["backend"] == "exact"
    assert partial["partial_outer_steps"] == 8
    assert partial["numerically_interpretable_partial_evidence"] == 1

    cli._write_recovery_report(run)
    manifest = cli._finalize_artifacts(run)
    assert cli._verify_artifact_manifest(run) == manifest
    assert _json(run / "terminal_outcome.json")["decision"] == decision
    report = (run / "REPORT.md").read_text(encoding="utf-8")
    assert f"`{decision}`" in report
    assert (
        "`full_horizon` not performed: "
        "`full_selected_backend_resource_deferred`."
    ) in report
    manifest_paths = {row["path"] for row in manifest["artifacts"]}
    assert short_path.relative_to(run).as_posix() in manifest_paths
    assert failure_path.relative_to(run).as_posix() in manifest_paths
    assert failure_path.with_suffix(".npz").relative_to(run).as_posix() not in (
        manifest_paths
    )


def test_terminal_outcome_prefers_evaluation_over_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _recovery_fixture_args(tmp_path, "evaluation-precedence")
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "evaluation-precedence" / "runs")
    (run / "terminal_outcome.json").unlink()
    (run / "exploratory_decision.json").unlink()
    (run / "artifact_manifest.json").unlink()
    (run / "SHA256SUMS.txt").unlink()

    def adverse(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "performed": 1,
            "decision": "evaluation_short_rollout_direction_not_useful",
            "recommended_next_action": "change the controller",
            "optional_future_fb300_unopened": 1,
        }

    monkeypatch.setattr(cli, "_recovery_evaluation_plan", adverse)
    parsed = cli.parse_args(
        ["--test-only", "--device", "cpu", "--stage", "all", "--resume-run-dir", str(run)]
    )
    result = cli._objective_first_recovery(run, parsed)
    assert result["decision"] == "evaluation_short_rollout_direction_not_useful"
    assert result["core_decision"] == _json(run / "core_objective.json")["decision"]
    assert result["recommended_next_action"] == "change the controller"


def test_candidate_evaluation_claim_guard_can_dominate_positive_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _recovery_fixture_args(tmp_path, "candidate-guard")
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "candidate-guard" / "runs")
    anchor = cli._recovery_anchors(run)[cli.SHORT_ANCHOR]
    saved = cli._recovery_test_family_states(anchor, backend="exact")
    result = type(
        "Result",
        (),
        {"saved_states": saved, "per_row_diagnostics": _mechanism_rows_fixture()},
    )()
    audit = cli._semantic(
        {
            "row_metrics": [
                {"squared_l2_state_discrepancy": 1.0} for _ in range(3)
            ],
            "paired_contrasts": {
                "learned_minus_zero": {"relative_error": 1.0},
                "source_informed_minus_zero": {"relative_error": 1.0},
            },
        }
    )
    artifact = cli._commit_recovery_evaluation_family(
        run,
        horizon="short",
        result=result,
        backend="candidate",
        audit=audit,
        family_summary=cli._semantic({"passed": 1}),
    )
    assert artifact["decision"] == "evaluation_short_approximation_dominates_observed_effect"
    assert artifact["exact_candidate_claim_guard"]["coarse_candidate_dynamic_claim_permitted"] == 0


@pytest.mark.parametrize("audit_relative", [0.0, 1.0])
def test_candidate_post_core_resume_reconstructs_claim_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audit_relative: float,
) -> None:
    args = _recovery_fixture_args(
        tmp_path,
        f"candidate-core-resume-{str(audit_relative).replace('.', 'p')}",
        "--test-force-candidate",
    )
    assert cli.main(args) == 0
    run = _only_run(Path(args[args.index("--runs-root") + 1]))
    core_path = run / "core_objective.json"
    audit_path = run / "exact_candidate_audit.json"
    audit = _json(audit_path)
    audit.pop("semantic_sha256")
    audit["paired_contrasts"]["learned_minus_zero"]["relative_error"] = audit_relative
    audit = cli._semantic(audit)
    cli.atomic_write_json(audit_path, audit)
    decision_path = run / "backend_decision.json"
    decision = _json(decision_path)
    decision.pop("semantic_sha256")
    decision["exact_candidate_audit_sha256"] = audit["semantic_sha256"]
    cli.atomic_write_json(decision_path, cli._semantic(decision))
    core = _json(core_path)
    core.pop("semantic_sha256")
    guard = dict(core["exact_candidate_claim_guard"])
    guard["learned_paired_contrast_relative_error"] = audit_relative
    guard["learned_contrast_guard_passed"] = int(audit_relative <= 0.25)
    guard["coarse_candidate_dynamic_claim_permitted"] = int(
        bool(guard["learned_contrast_guard_passed"])
        and bool(guard["endpoint_scale_guard_passed"])
    )
    core["exact_candidate_claim_guard"] = guard
    if not guard["coarse_candidate_dynamic_claim_permitted"]:
        core["decision"] = "approximation_dominates_observed_effect"
        core["recommended_next_action"] = (
            "retain the objective images; audit more exact shards or improve the "
            "exploration backend before making a learned-utility claim"
        )
    cli.atomic_write_json(core_path, cli._semantic(core))
    # This isolates the post-core verifier's authority logic from the audit's
    # deterministic shard reconstruction, which has its own focused tests.
    monkeypatch.setattr(
        cli,
        "_verify_recovery_candidate_audit",
        lambda *_args, **_kwargs: audit,
    )
    verified = cli._verify_recovery_core_artifact(
        run, cli.parse_args(["--test-only", "--device", "cpu"])
    )
    assert verified["exact_candidate_claim_guard"] == guard
    assert verified["decision"] == core["decision"]


def test_source_informed_mismatch_does_not_override_learned_signal(
    tmp_path: Path,
) -> None:
    args = _recovery_fixture_args(tmp_path, "source-mismatch-precedence")
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "source-mismatch-precedence" / "runs")
    anchor = cli._recovery_anchors(run)[cli.SHORT_ANCHOR]
    _source, target = cli._source_arrays(run)
    zero = anchor.copy()
    learned = 0.9 * zero + 0.1 * target
    source_bad = np.zeros(784, dtype=np.float64)
    source_bad[int(np.argmin(target))] = 1.0
    final = np.stack((zero, learned, source_bad))
    saved = {name: final.copy() for name in ("start", "progress_25", "progress_50", "progress_75", "final")}
    result = type(
        "Result",
        (),
        {"saved_states": saved, "per_row_diagnostics": _mechanism_rows_fixture()},
    )()
    core = cli._commit_recovery_objective_artifacts(
        run, anchor=anchor, result=result, backend="exact", audit=None
    )
    assert core["decision"] == "learned_short_dynamic_signal"
    assert core["source_informed_composition_mismatch_diagnostic"] == 1
    evaluation = cli._commit_recovery_evaluation_family(
        run,
        horizon="short",
        result=result,
        backend="exact",
        audit=None,
        family_summary=cli._semantic({"passed": 1}),
    )
    assert evaluation["decision"] == "learned_short_dynamic_signal_exact"
    assert evaluation["source_informed_composition_mismatch_diagnostic"] == 1


def test_recovery_resource_ledger_counts_forward_shards_and_overhead(
    tmp_path: Path,
) -> None:
    root = tmp_path / "resource"
    shard = root / "objective/evaluation_forward/forward_shards/eval/shard-0000.json"
    shard.parent.mkdir(parents=True)
    cli.atomic_write_json(
        shard,
        cli._semantic(
            {
                "elapsed_seconds": 7.0,
                "diagnostics": {
                    "reference": {
                        "maximum_cuda_memory_allocated": 4,
                        "total_cuda_memory_bytes": 10,
                    }
                },
            }
        ),
    )
    timing = root / "metrics/evaluation_forward_attempt_timing.json"
    timing.parent.mkdir(parents=True)
    cli.atomic_write_json(
        timing,
        cli._semantic(
            {
                "verified_shard_count": 1,
                "commit_verification_overhead_seconds": 2.5,
            }
        ),
    )
    usage = cli._recovery_observed_resource(root)
    assert usage["active_seconds"] == 9.5
    assert usage["forward_shard_count"] == 1
    assert usage["peak_memory_fraction"] == 0.4


def test_cross_horizon_route_marks_short_only_signal() -> None:
    short = {"performed": 1, "decision": "learned_short_dynamic_signal_exact"}
    full = {"performed": 1, "decision": "evaluation_full_rollout_direction_not_useful"}
    decision = (
        "learned_short_only_dynamic_signal"
        if int(full["performed"])
        and "dynamic_signal" in short["decision"]
        and "dynamic_signal" not in full["decision"]
        else full["decision"]
    )
    assert decision == "learned_short_only_dynamic_signal"


@pytest.mark.parametrize(
    ("requested", "fits", "expected", "raises"),
    [
        ("candidate", True, "candidate", False),
        ("exact", True, "exact", False),
        ("auto", False, "candidate", False),
        ("exact", False, None, True),
    ],
)
def test_evaluation_backend_policy_uses_global_request(
    requested: str, fits: bool, expected: str | None, raises: bool
) -> None:
    if raises:
        with pytest.raises(cli.RolloutCLIError):
            cli._candidate_backend_selected({"passed": int(fits)}, requested)
    else:
        assert cli._candidate_backend_selected({"passed": int(fits)}, requested) == expected


def test_fresh_forward_tail_projection_prices_entire_remaining_prefix() -> None:
    early = cli._recovery_forward_tail_projection(
        active_seconds=100.0,
        wasted_active_seconds=0.0,
        observed_forward_shard_seconds=(10.0,),
        remaining_forward_shards=63,
        short_reverse_reserve_seconds=200.0,
        maximum_main_seconds=1000.0,
        persisted_bytes=1000,
        projected_forward_bytes=6300,
        short_reverse_reserve_bytes=2000,
        peak_memory_fraction=0.1,
    )
    late = cli._recovery_forward_tail_projection(
        active_seconds=400.0,
        wasted_active_seconds=0.0,
        observed_forward_shard_seconds=(10.0,),
        remaining_forward_shards=3,
        short_reverse_reserve_seconds=200.0,
        maximum_main_seconds=1000.0,
        persisted_bytes=7300,
        projected_forward_bytes=300,
        short_reverse_reserve_bytes=2000,
        peak_memory_fraction=0.1,
    )
    assert early["remaining_forward_shards"] == 63
    assert early["passed"] == 0
    assert late["remaining_forward_shards"] == 3
    assert late["passed"] == 1
    assert late["short_reverse_reserve_seconds"] == 200.0


def test_fresh_forward_tail_projection_stops_after_runtime_degradation() -> None:
    projection = cli._recovery_forward_tail_projection(
        active_seconds=400.0,
        wasted_active_seconds=0.0,
        observed_forward_shard_seconds=(2.0, 20.0),
        remaining_forward_shards=20,
        short_reverse_reserve_seconds=100.0,
        maximum_main_seconds=900.0,
        persisted_bytes=0,
        projected_forward_bytes=0,
        short_reverse_reserve_bytes=0,
        peak_memory_fraction=0.1,
    )
    assert projection["observed_slowest_complete_forward_shard_seconds"] == 20.0
    assert projection["passed"] == 0


def test_mixed_optional_projection_returns_authorizing_record(
    tmp_path: Path,
) -> None:
    args = cli.parse_args(["--test-only", "--device", "cpu"])
    record = cli._recovery_mixed_optional_projection(
        tmp_path,
        args,
        selected_backend="candidate",
        continuation_shards=16,
        headroom_fraction=0.20,
    )
    assert isinstance(record, dict)
    assert record["exact_audit_shards"] == 1
    assert record["continuation_shards"] == 16
    assert record["passed"] in {0, 1}


def test_source_diagnostic_does_not_override_nonpositive_learned_outcome(
    tmp_path: Path,
) -> None:
    args = _recovery_fixture_args(tmp_path, "source-negative-precedence")
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "source-negative-precedence" / "runs")
    anchor = cli._recovery_anchors(run)[cli.SHORT_ANCHOR]
    _source, target = cli._source_arrays(run)
    zero = anchor.copy()
    learned = np.zeros(784, dtype=np.float64)
    learned[int(np.argmin(target))] = 1.0
    source_bad = learned.copy()
    final = np.stack((zero, learned, source_bad))
    saved = {
        name: final.copy()
        for name in ("start", "progress_25", "progress_50", "progress_75", "final")
    }
    result = type(
        "Result",
        (),
        {"saved_states": saved, "per_row_diagnostics": _mechanism_rows_fixture()},
    )()
    core = cli._commit_recovery_objective_artifacts(
        run, anchor=anchor, result=result, backend="exact", audit=None
    )
    assert core["decision"] == "learned_short_rollout_direction_not_useful"
    assert core["source_informed_composition_mismatch_diagnostic"] == 1
    evaluation = cli._commit_recovery_evaluation_family(
        run,
        horizon="short",
        result=result,
        backend="exact",
        audit=None,
        family_summary=cli._semantic({"passed": 1}),
    )
    assert evaluation["decision"] == "evaluation_short_rollout_direction_not_useful"
    assert evaluation["source_informed_composition_mismatch_diagnostic"] == 1


def test_interrupted_post_core_resume_rejects_self_semantic_metric_tamper(
    tmp_path: Path,
) -> None:
    args = _recovery_fixture_args(tmp_path, "core-resume-tamper")
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "core-resume-tamper" / "runs")
    for relative in (
        "terminal_outcome.json",
        "exploratory_decision.json",
        "artifact_manifest.json",
        "SHA256SUMS.txt",
        "bundle_integrity_audit.json",
        "REPORT.md",
        "HANDOFF.md",
    ):
        (run / relative).unlink(missing_ok=True)
    core_path = run / "core_objective.json"
    core = _json(core_path)
    changed = {key: value for key, value in core.items() if key != "semantic_sha256"}
    changed["row_results"][1]["metrics_to_mixed_target"]["progress_25"][
        "squared_l2_error"
    ] += 123.0
    changed["row_results"][1]["mechanism_diagnostics"]["score_rms"] = 987.0
    cli.atomic_write_json(core_path, cli._semantic(changed))
    before = {
        path.relative_to(run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*")
        if path.is_file()
    }
    assert cli.main(
        [
            "--test-only",
            "--device",
            "cpu",
            "--stage",
            "all",
            "--resume-run-dir",
            str(run),
        ]
    ) == 2
    after = {
        path.relative_to(run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_anchor_archive_and_binding_are_cross_bound_to_manifest_and_predecessor(
    tmp_path: Path,
) -> None:
    args = _recovery_fixture_args(tmp_path, "anchor-binding-tamper")
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "anchor-binding-tamper" / "runs")
    anchor_path = run / "input_bindings/recovery_anchors.npz"
    arrays = cli._load_npz(anchor_path)
    arrays["step_0127"] = np.roll(arrays["step_0127"], 1)
    artifact = cli._atomic_npz(anchor_path, **arrays)
    binding_path = run / "input_bindings/recovery_anchor_binding.json"
    binding = _json(binding_path)
    changed = {key: value for key, value in binding.items() if key != "semantic_sha256"}
    changed["file_sha256"] = artifact["sha256"]
    changed["file_size"] = artifact["size"]
    changed["array_sha256"] = {
        name: cli._core_array_sha256(value.reshape(1, 784))
        for name, value in arrays.items()
    }
    cli.atomic_write_json(binding_path, cli._semantic(changed))
    before = {
        path.relative_to(run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*")
        if path.is_file()
    }
    assert cli.main(
        [
            "--test-only",
            "--device",
            "cpu",
            "--stage",
            "all",
            "--resume-run-dir",
            str(run),
        ]
    ) == 2
    after = {
        path.relative_to(run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_interrupted_post_core_resume_rejects_image_byte_tamper(
    tmp_path: Path,
) -> None:
    args = _recovery_fixture_args(tmp_path, "core-image-tamper")
    assert cli.main(args) == 0
    run = _only_run(tmp_path / "core-image-tamper" / "runs")
    for relative in (
        "terminal_outcome.json",
        "exploratory_decision.json",
        "artifact_manifest.json",
        "SHA256SUMS.txt",
        "bundle_integrity_audit.json",
        "REPORT.md",
        "HANDOFF.md",
    ):
        (run / relative).unlink(missing_ok=True)
    image = run / "images/individual/development-core-learned-1-progress_25-raw.png"
    image.write_bytes(image.read_bytes() + b"tamper")
    before = {
        path.relative_to(run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*")
        if path.is_file()
    }
    assert cli.main(
        [
            "--test-only",
            "--device",
            "cpu",
            "--stage",
            "all",
            "--resume-run-dir",
            str(run),
        ]
    ) == 2
    after = {
        path.relative_to(run).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize("backend", ["exact", "candidate"])
def test_actual_recovery_backend_runner_reaches_audit_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    from mnist import d0_jacobi_rb_tangent_fused as fused
    from mnist.d0_jacobi_rb_tangent_fused import FusedRowSpec

    run = tmp_path / backend
    run.mkdir()
    cli.atomic_write_json(
        run / "path_usage.json",
        cli._semantic(
            {
                "selected": {
                    "development": cli.PATH_IDS["development"],
                    "evaluation": cli.PATH_IDS["evaluation"],
                    "replication": cli.PATH_IDS["replication"],
                },
                "passed": 1,
            }
        ),
    )
    specs = (
        FusedRowSpec("zero", cli.PATH_IDS["development"], "zero", "zero", "short"),
        FusedRowSpec(
            "learned",
            cli.PATH_IDS["development"],
            "learned",
            "learned",
            "short",
            1.0,
        ),
        FusedRowSpec("source", cli.PATH_IDS["development"], "oracle", "oracle", "short"),
    )
    monkeypatch.setattr(
        cli,
        "_fused_controller_family",
        lambda *_args, **_kwargs: (specs, object(), {"row_table": [row.to_record() for row in specs]}),
    )
    monkeypatch.setattr(cli, "_prepared_fused_reference", lambda *_args: object())
    monkeypatch.setattr(cli, "_fused_reference_factory", lambda **_kwargs: lambda _index: object())
    monkeypatch.setattr(
        cli, "_candidate_fused_reference_factory", lambda **_kwargs: lambda _index: object()
    )
    sequence = cli._reverse_sequence(0)
    transitions = len(sequence) * 4 * len(specs) * 392
    reference = {
        "transition_count": transitions,
        "active_count": transitions,
        "structural_noop_count": 0,
        "forbidden_counts": {
            name: 0
            for name in (
                "resource_cap_count",
                "invalid_density_count",
                *(("approximation_count",) if backend == "exact" else ()),
                "clipping_count",
                "correction_count",
                "floor_count",
                "limiter_count",
                "projection_count",
                "renormalization_count",
                "nonfinite_count",
            )
        },
    }
    if backend == "exact":
        reference.update(
            {
                "certified_count": transitions,
                "fallback_count": 0,
                "unauthorized_count": 0,
                "invalid_count": 0,
                "certificate_fraction": 1.0,
                "per_row": [
                    {
                        "transition_count": transitions // len(specs),
                        "active_count": transitions // len(specs),
                        "structural_noop_count": 0,
                        "certified_count": transitions // len(specs),
                        "fallback_count": 0,
                        "unauthorized_count": 0,
                        "invalid_count": 0,
                        "certificate_fraction": 1.0,
                    }
                    for _ in specs
                ],
            }
        )
    else:
        reference.update(
            {
                "reference_contract": "candidate_approximate_v1",
                "approximation_count": transitions,
                "invalid_count": 0,
                "certificate_fraction": "not_applicable",
                "maximum_candidate_bracket_width": 0.0,
                "per_row": [
                    {
                        "transition_count": transitions // len(specs),
                        "active_count": transitions // len(specs),
                        "structural_noop_count": 0,
                        "approximation_count": transitions // len(specs),
                        "invalid_count": 0,
                        "certificate_fraction": "not_applicable",
                        "maximum_candidate_bracket_width": 0.0,
                    }
                    for _ in specs
                ],
            }
        )
    state = np.full((3, 784), 1.0 / 784.0, dtype=np.float64)

    class Result:
        final_state = state
        transition_count = transitions
        diagnostics = {"maximum_mass_error": 0.0}
        per_row_diagnostics: tuple[dict[str, object], ...] = ({}, {}, {})
        saved_states = {"final": state}
        shard_records = ({"diagnostics": {"reference": reference}},)

        def to_record(self) -> dict[str, object]:
            return {
                "transition_count": transitions,
                "row_table": [row.to_record() for row in specs],
            }

    monkeypatch.setattr(fused, "run_fused_reverse_family", lambda *_args, **_kwargs: Result())
    args = cli.parse_args(["--test-only", "--device", "cpu"])
    result, summary, _root = cli._run_recovery_core_backend(
        run,
        args,
        anchor=state[0],
        backend=backend,
        sequence=sequence,
        exact_audit_only=True,
        rows=(("zero", "zero", "short", None), ("learned", "learned", "short", 1.0), ("source", "oracle", "short", None)),
    )
    assert result.transition_count == transitions
    assert summary["passed"] == 1


def test_production_exact_audit_health_requires_direct_aggregate_counts() -> None:
    """Reproduce the 263424-lane synchronous-replay false negative fail closed."""

    transition_count = 263_424
    per_row_count = 87_808
    forbidden = {
        name: 0
        for name in (
            "resource_cap_count",
            "invalid_density_count",
            "approximation_count",
            "clipping_count",
            "correction_count",
            "floor_count",
            "limiter_count",
            "projection_count",
            "renormalization_count",
            "nonfinite_count",
        )
    }
    rows = [
        {
            "transition_count": per_row_count,
            "active_count": per_row_count,
            "structural_noop_count": 0,
            "certified_count": per_row_count,
            "fallback_count": 0,
            "unauthorized_count": 0,
            "invalid_count": 0,
            "certificate_fraction": 1.0,
        }
        for _ in range(3)
    ]
    reference = {
        "transition_count": transition_count,
        "certified_count": transition_count,
        "fallback_count": 0,
        "certificate_fraction": 1.0,
        "forbidden_counts": forbidden,
        "per_row": rows,
    }
    state = np.full((3, 784), 1.0 / 784.0, dtype=np.float64)

    result = type("ProductionExactAuditResult", (), {})()
    result.final_state = state
    result.transition_count = transition_count
    result.diagnostics = {"maximum_mass_error": 2.220446049250313e-16}
    result.shard_records = ({"diagnostics": {"reference": reference}},)

    with pytest.raises(cli.RolloutCLIError) as captured:
        cli._recovery_family_health(
            result, backend="exact", expected_transition_count=transition_count
        )
    assert captured.value.failure_domain == "implementation_contract"
    assert (
        captured.value.failure_code
        == "synchronous_exact_reference_health_schema_invalid"
    )

    reference.update(
        {
            "active_count": transition_count,
            "structural_noop_count": 0,
            "unauthorized_count": 0,
            "invalid_count": 0,
        }
    )
    health = cli._recovery_family_health(
        result, backend="exact", expected_transition_count=transition_count
    )
    assert health["passed"] == 1
    assert health["active_count"] == transition_count
    assert health["certified_count"] == transition_count


def _strict_recovery_family_health_fixture(
    backend: str,
) -> tuple[object, dict[str, object]]:
    transitions = 6
    if backend == "exact":
        rows = [
            {
                "transition_count": 3,
                "active_count": 3,
                "structural_noop_count": 0,
                "certified_count": 3,
                "fallback_count": 0,
                "unauthorized_count": 0,
                "invalid_count": 0,
                "certificate_fraction": 1.0,
            }
            for _ in range(2)
        ]
        reference: dict[str, object] = {
            "transition_count": transitions,
            "active_count": transitions,
            "structural_noop_count": 0,
            "certified_count": transitions,
            "fallback_count": 0,
            "unauthorized_count": 0,
            "invalid_count": 0,
            "certificate_fraction": 1.0,
            "forbidden_counts": {
                name: 0 for name in cli.RECOVERY_EXACT_FORBIDDEN_COUNTS
            },
            "per_row": rows,
        }
    else:
        rows = [
            {
                "transition_count": 3,
                "active_count": 3,
                "structural_noop_count": 0,
                "approximation_count": 3,
                "invalid_count": 0,
                "certificate_fraction": "not_applicable",
                "maximum_candidate_bracket_width": 1.0e-12,
            }
            for _ in range(2)
        ]
        reference = {
            "reference_contract": "candidate_approximate_v1",
            "transition_count": transitions,
            "active_count": transitions,
            "structural_noop_count": 0,
            "approximation_count": transitions,
            "invalid_count": 0,
            "certificate_fraction": "not_applicable",
            "maximum_candidate_bracket_width": 1.0e-12,
            "forbidden_counts": {
                name: 0 for name in cli.RECOVERY_CANDIDATE_FORBIDDEN_COUNTS
            },
            "per_row": rows,
        }
    state = np.full((2, 784), 1.0 / 784.0, dtype=np.float64)
    result = type("StrictRecoveryFamilyHealthResult", (), {})()
    result.final_state = state
    result.transition_count = transitions
    result.diagnostics = {"maximum_mass_error": 0.0}
    result.shard_records = ({"diagnostics": {"reference": reference}},)
    return result, reference


@pytest.mark.parametrize("backend", ["exact", "candidate"])
def test_recovery_family_health_requires_complete_forbidden_schema(
    backend: str,
) -> None:
    result, reference = _strict_recovery_family_health_fixture(backend)
    assert cli._recovery_family_health(
        result, backend=backend, expected_transition_count=6
    )["passed"] == 1
    forbidden = reference["forbidden_counts"]
    assert isinstance(forbidden, dict)
    forbidden.pop(next(iter(forbidden)))
    with pytest.raises(cli.RolloutCLIError) as captured:
        cli._recovery_family_health(
            result, backend=backend, expected_transition_count=6
        )
    assert captured.value.failure_domain == "implementation_contract"


@pytest.mark.parametrize("backend", ["exact", "candidate"])
def test_recovery_family_health_rejects_row_authority_redistribution(
    backend: str,
) -> None:
    result, reference = _strict_recovery_family_health_fixture(backend)
    rows = reference["per_row"]
    assert isinstance(rows, list)
    authority = (
        ("transition_count", "active_count", "certified_count")
        if backend == "exact"
        else ("transition_count", "active_count", "approximation_count")
    )
    for name in authority:
        rows[0][name] -= 1
        rows[1][name] += 1
    with pytest.raises(cli.RolloutCLIError) as captured:
        cli._recovery_family_health(
            result, backend=backend, expected_transition_count=6
        )
    assert captured.value.failure_domain == "implementation_contract"


@pytest.mark.parametrize(
    ("location", "field"),
    [("aggregate", "certified_count"), ("row", "authorized_count")],
)
def test_candidate_family_health_rejects_false_exact_sentinels(
    location: str, field: str
) -> None:
    result, reference = _strict_recovery_family_health_fixture("candidate")
    if location == "aggregate":
        reference[field] = 0
    else:
        rows = reference["per_row"]
        assert isinstance(rows, list)
        rows[0][field] = 0
    with pytest.raises(cli.RolloutCLIError) as captured:
        cli._recovery_family_health(
            result, backend="candidate", expected_transition_count=6
        )
    assert captured.value.failure_code == "candidate_reference_health_schema_invalid"


def test_exact_health_schema_failure_finalizer_seals_partial_audit_usage(
    tmp_path: Path,
) -> None:
    run = tmp_path / "schema-failure"
    run.mkdir()
    cli.atomic_write_json(
        run / "scientific_config.json",
        cli._semantic({"schema": cli.RECOVERY_TEST_RUN_SCHEMA, "test_only": 1}),
    )
    cli.atomic_write_json(
        run / "resource_ledger.json",
        cli._semantic(
            {
                "schema": cli.RECOVERY_TEST_RUN_SCHEMA + "-resource-ledger",
                "maximum_main_seconds": 21_600.0,
                "active_seconds": 0.0,
                "wasted_active_seconds": 0.0,
                "persisted_bytes": 0,
                "passed": 1,
            }
        ),
    )
    cli.atomic_write_json(
        run / "backend_decision.json",
        cli._semantic(
            {
                "schema": cli.RECOVERY_RUN_SCHEMA + "-backend-decision",
                "schema_version": 1,
                "phase": "not_attempted_before_analytic_control",
                "requested": "auto",
                "selected": "not_attempted",
                "exact_audit_shard_committed": 0,
                "selected_family_complete": 0,
            }
        ),
    )
    root = run / "objective_attempts/exact/fused_families/development-core/short"
    state = np.full((3, 784), 1.0 / 784.0, dtype=np.float64)
    artifact = cli._atomic_npz(root / "shard-0000.npz", state=state)
    per_row = [
        {
            "transition_count": 87_808,
            "active_count": 87_808,
            "certified_count": 87_808,
            "fallback_count": 0,
            "unauthorized_count": 0,
            "invalid_count": 0,
            "certificate_fraction": 1.0,
        }
        for _ in range(3)
    ]
    cli.atomic_write_json(
        root / "shard-0000.json",
        cli._semantic(
            {
                "committed": 1,
                "shard_index": 0,
                "elapsed_seconds": 291.9055766,
                "state_file_sha256": artifact["sha256"],
                "state_file_size": artifact["size"],
                "execution_plan": {"sequence": [[127, 6]] * 56},
                "diagnostics": {
                    "maximum_mass_error": 2.220446049250313e-16,
                    "reference": {
                        "transition_count": 263_424,
                        "certified_count": 263_424,
                        "fallback_count": 0,
                        "certificate_fraction": 1.0,
                        "forbidden_counts": {
                            "resource_cap_count": 0,
                            "invalid_density_count": 0,
                            "approximation_count": 0,
                            "clipping_count": 0,
                            "correction_count": 0,
                            "floor_count": 0,
                            "limiter_count": 0,
                            "projection_count": 0,
                            "renormalization_count": 0,
                            "nonfinite_count": 0,
                        },
                        "maximum_cuda_memory_allocated": 46_531_584,
                        "total_cuda_memory_bytes": 8_546_484_224,
                        "per_row": per_row,
                    },
                },
            }
        ),
    )
    cli.atomic_write_json(
        run / "metrics/recovery_active_setup.json",
        cli._semantic({"elapsed_seconds": 2.218354}),
    )
    cli.atomic_write_json(
        run / "metrics/recovery_attempt_timing_development-core-short-exact.json",
        cli._semantic({"commit_verification_overhead_seconds": 0.0932145}),
    )
    cli.atomic_write_json(
        run / "metrics/wasted_active_seconds.json",
        cli._semantic({"wasted_active_seconds": 0.0057578}),
    )
    error = cli.RolloutCLIError(
        "exact reference aggregate health schema changed",
        failure_domain="implementation_contract",
        failure_code="synchronous_exact_reference_health_schema_invalid",
    )
    cli._failure_recovery(run, "objective", error)

    failure = _json(run / "failure.json")
    assert failure["decision"] == "synchronous_exact_reference_health_schema_invalid"
    assert failure["failure_domain"] == "implementation_contract"
    assert failure["executed_shard_numerics_valid"] == 1
    assert failure["executed_shard_resource_valid"] == 1
    assert failure["scientific_evidence_complete"] == 0
    assert failure["partial_outer_steps"] == 8
    backend = _json(run / "backend_decision.json")
    assert backend["selected"] == "not_attempted"
    assert backend["exact_audit_shard_committed"] == 1
    ledger = _json(run / "resource_ledger.json")
    assert ledger["active_seconds"] == pytest.approx(294.2171451)
    assert ledger["wasted_active_seconds"] == pytest.approx(0.0057578)
    assert ledger["peak_memory_bytes"] == 46_531_584
    report = (run / "REPORT.md").read_text(encoding="utf-8")
    handoff = (run / "HANDOFF.md").read_text(encoding="utf-8")
    assert "8-step three-row" in report
    assert "No core endpoint" in report
    assert "Proxy-only patches since the last objective-bearing experiment: 0" in handoff
    audit = _json(run / "bundle_integrity_audit.json")
    representatives = audit["representative_npz_opened_and_hashed"]
    assert any(row["path"].endswith("shard-0000.npz") for row in representatives)


def test_exact_prefix_rejects_approximate_contract_before_authority(
    tmp_path: Path,
) -> None:
    root = tmp_path / "exact-contract"
    state = np.full((1, 784), 1.0 / 784.0, dtype=np.float64)
    artifact = cli._atomic_npz(root / "shard-0000.npz", state=state)
    sequence = cli._reverse_sequence(0)
    transition_count = len(sequence) * 4 * 392
    from mnist.d0_jacobi_rb_tangent_fused import FusedRowSpec

    spec = FusedRowSpec("zero", 1, "zero", "zero", "short")
    binding = {"row_table": [spec.to_record()]}
    rng = {"root_seed": 1, "stream_role": "exact-contract", "variant_in_rng_key": 0}
    forbidden = {
        name: 0
        for name in (
            "resource_cap_count", "invalid_density_count", "approximation_count",
            "clipping_count", "correction_count", "floor_count", "limiter_count",
            "projection_count", "renormalization_count", "nonfinite_count",
        )
    }
    record = cli._semantic(
        {
            "family_name": "family", "segment_name": "short", "shard_index": 0,
            "sequence_start": list(sequence[0]), "sequence_end": list(sequence[-1]),
            "sequence_sha256": cli.config_fingerprint([list(item) for item in sequence]),
            "row_table": [spec.to_record()], "row_keys": ["zero"],
            "canonical_path_ids": [1], "microsteps": cli.MICROSTEPS, "label": 3,
            "variant_in_rng_key": 0, "input_state_sha256": cli._core_array_sha256(state),
            "output_state_sha256": cli._core_array_sha256(state),
            "controller_binding_sha256": cli.config_fingerprint(binding),
            "rng_binding_sha256": cli.config_fingerprint(rng),
            "transition_count": transition_count,
            "execution_plan": {"shard_index": 0, "sequence": [list(item) for item in sequence], "row_count": 1, "transition_count": transition_count, "input_state_sha256": cli._core_array_sha256(state)},
            "state_file_sha256": artifact["sha256"], "state_file_size": artifact["size"],
            "reference_contract": "candidate_approximate_v1",
            "per_row_diagnostics": [{"reference_transition_count": transition_count, "reference_active_count": transition_count, "reference_structural_noop_count": 0, "reference_certified_count": transition_count, "reference_fallback_count": 0}],
            "controller_diagnostics": [{}],
            "diagnostics": {"transition_count": transition_count, "certificate_fraction": 1.0, "maximum_mass_error": 0.0, "reference": {"transition_count": transition_count, "active_count": transition_count, "structural_noop_count": 0, "certified_count": transition_count, "fallback_count": 0, "certificate_fraction": 1.0, "forbidden_counts": forbidden}},
            "committed": 1,
        }
    )
    cli.atomic_write_json(root / "shard-0000.json", record)
    with pytest.raises(cli.ArtifactCompatibilityError, match="approximate reference"):
        cli._verify_fused_family_prefix(root, initial_state=state, sequence=sequence, row_specs=(spec,), controller_binding=binding, rng_binding=rng, family_name="family", segment_name="short")


def test_candidate_health_rejects_fractional_integer_counter() -> None:
    transitions = 392
    forbidden = {
        name: 0
        for name in (
            "resource_cap_count", "invalid_density_count", "clipping_count",
            "correction_count", "floor_count", "limiter_count", "projection_count",
            "renormalization_count", "nonfinite_count",
        )
    }
    forbidden["resource_cap_count"] = 0.5
    record = {
        "reference_contract": "candidate_approximate_v1",
        "per_row_diagnostics": [{"reference_transition_count": transitions, "reference_active_count": transitions, "reference_structural_noop_count": 0, "reference_approximation_count": transitions, "reference_invalid_count": 0}],
        "controller_diagnostics": [{}],
        "diagnostics": {"transition_count": transitions, "certificate_fraction": "not_applicable", "maximum_mass_error": 0.0, "reference": {"reference_contract": "candidate_approximate_v1", "transition_count": transitions, "active_count": transitions, "structural_noop_count": 0, "approximation_count": transitions, "invalid_count": 0, "certificate_fraction": "not_applicable", "forbidden_counts": forbidden, "maximum_candidate_bracket_width": 0.0}},
    }
    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._verify_candidate_fused_shard_health(record, expected_transitions=transitions, row_count=1)


@pytest.mark.parametrize("fractional_field", ["forbidden", "per_row"])
def test_exact_health_rejects_fractional_integer_counter(
    fractional_field: str,
) -> None:
    transitions = 392
    forbidden = {
        name: 0
        for name in (
            "resource_cap_count", "invalid_density_count", "approximation_count",
            "clipping_count", "correction_count", "floor_count", "limiter_count",
            "projection_count", "renormalization_count", "nonfinite_count",
        )
    }
    per_row = {
        "reference_transition_count": transitions,
        "reference_active_count": transitions,
        "reference_structural_noop_count": 0,
        "reference_certified_count": transitions,
        "reference_fallback_count": 0,
        "reference_unauthorized_count": 0,
        "reference_invalid_count": 0,
    }
    if fractional_field == "forbidden":
        forbidden["resource_cap_count"] = 0.5
    else:
        per_row["reference_certified_count"] = transitions - 0.5
    record = {
        "per_row_diagnostics": [per_row],
        "controller_diagnostics": [{}],
        "diagnostics": {
            "transition_count": transitions,
            "certificate_fraction": 1.0,
            "maximum_mass_error": 0.0,
            "reference": {
                "transition_count": transitions,
                "active_count": transitions,
                "structural_noop_count": 0,
                "certified_count": transitions,
                "fallback_count": 0,
                "certificate_fraction": 1.0,
                "forbidden_counts": forbidden,
            },
        },
    }
    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._verify_fused_shard_health(
            record, expected_transitions=transitions, row_count=1
        )


def test_gain_objective_recomputes_selection_and_rejects_risk_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "gain-objective"
    run.mkdir()
    target = np.full(784, 1.0 / 784.0, dtype=np.float64)
    monkeypatch.setattr(cli, "_source_arrays", lambda _run: (target, target))
    required = ("start", "progress_25", "progress_50", "progress_75", "final")
    base = np.repeat(target[None, :], 3, axis=0)
    saved = {name: base.copy() for name in required}
    core_root = run / "objective/development-core-short"
    core_artifact = cli._atomic_npz(
        core_root / "selected_states.npz", **{name: base.copy() for name in required}
    )
    core_artifact["path"] = (core_root / "selected_states.npz").relative_to(run).as_posix()
    core = cli._semantic(
        {
            "raw_states": core_artifact,
            "learned_final_squared_l2": 0.0,
        }
    )
    cli.atomic_write_json(run / "core_objective.json", core)
    result = type(
        "GainResult",
        (),
        {
            "saved_states": saved,
            "per_row_diagnostics": tuple(
                {
                    "score_count": 1,
                    "score_squared_sum": float(index + 1),
                    "logistic_shift_count": 1,
                    "logistic_shift_squared_sum": 0.0,
                    "reference_fraction_displacement_count": 1,
                    "reference_fraction_displacement_squared_sum": 1.0e-6,
                    "control_fraction_displacement_count": 1,
                    "control_fraction_displacement_squared_sum": 0.0,
                    "target_oracle_unreachable_boundary_count": 0,
                }
                for index in range(3)
            ),
        },
    )()
    summary = cli._semantic({"result": {}, "passed": 1})
    written = cli._recovery_gain_objective_artifact(
        run,
        result=result,
        backend="exact",
        family_summary=summary,
        audit=None,
    )
    assert written["selected_gain"] == 0.5
    assert len(written["row_results"]) == 3
    assert (run / written["contact_sheet"]).is_file()
    assert cli._recovery_gain_objective_artifact(
        run,
        result=result,
        backend="exact",
        family_summary=summary,
        audit=None,
        verify_only=True,
    ) == written
    tampered = dict(written)
    tampered["candidates"] = [dict(row) for row in written["candidates"]]
    tampered["candidates"][0]["final_squared_l2"] = 123.0
    tampered.pop("semantic_sha256")
    cli.atomic_write_json(
        run / "objective/development-gain-expansion/gain_objective_family.json",
        cli._semantic(tampered),
    )
    before = cli.file_fingerprint(
        run / "objective/development-gain-expansion/gain_objective_family.json"
    )
    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._recovery_gain_objective_artifact(
            run,
            result=result,
            backend="exact",
            family_summary=summary,
            audit=None,
            verify_only=True,
        )
    assert cli.file_fingerprint(
        run / "objective/development-gain-expansion/gain_objective_family.json"
    ) == before


def test_unperformed_records_cannot_suppress_committed_objective_evidence(
    tmp_path: Path,
) -> None:
    run = tmp_path / "suppression"
    run.mkdir()
    args = cli.parse_args(["--test-only", "--device", "cpu"])
    gain = cli._semantic(
        {
            "schema": cli.RECOVERY_RUN_SCHEMA + "-gain-expansion",
            "schema_version": 1,
            "performed": 0,
            "reason": "test_fixture_core_only",
            "projection": {
                "schema": cli.RECOVERY_RUN_SCHEMA + "-resource-projection",
                "remaining_shards": 16,
            },
            "selected_gain": 1.0,
            "core_artifact_already_committed": 1,
        }
    )
    cli.atomic_write_json(run / "gain_expansion.json", gain)
    shard = run / (
        "objective_attempts/exact/fused_families/"
        "development-gain-expansion/short/shard-0000.json"
    )
    shard.parent.mkdir(parents=True)
    shard.write_text("{}", encoding="utf-8")
    with pytest.raises(cli.ArtifactCompatibilityError, match="unperformed gain"):
        cli._recovery_gain_expansion(
            run, args, anchor=np.full(784, 1.0 / 784.0), backend="exact"
        )

    evaluation = cli._semantic(
        {
            "schema": cli.RECOVERY_RUN_SCHEMA + "-evaluation-plan",
            "schema_version": 1,
            "performed": 0,
            "reason": "test_fixture_core_only",
            "projection": {},
            "selected_gain": 1.0,
            "optional_future_fb300_unopened": 1,
        }
    )
    evaluation_shard = run / (
        "objective_attempts/exact/fused_families/"
        "evaluation-short/short/shard-0000.json"
    )
    evaluation_shard.parent.mkdir(parents=True)
    evaluation_shard.write_text("{}", encoding="utf-8")
    with pytest.raises(cli.ArtifactCompatibilityError, match="committed evidence"):
        cli._verify_existing_recovery_evaluation_plan(
            run, evaluation, selected_gain=1.0, args=args
        )


def test_typed_forward_aggregate_failure_writes_always_initialization_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "typed-aggregate-child"
    run.mkdir()
    args = cli.parse_args(["--test-only", "--device", "cpu"])
    args.invoked_argv = ["--test-only", "--device", "cpu"]
    for name in ("predecessor", "frequency1", "source"):
        (tmp_path / name).mkdir()
    args.predecessor_run_dir = tmp_path / "predecessor"
    args.frequency1_run_dir = tmp_path / "frequency1"
    args.source_run_dir = tmp_path / "source"
    monkeypatch.setattr(
        cli,
        "_verify_recovery_predecessor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            cli.ExactForwardShardAggregateError(
                "injected aggregate contract failure",
                failure_domain="implementation_contract",
            )
        ),
    )
    with pytest.raises(cli.ExactForwardShardAggregateError):
        cli._initialize_recovery(run, args)
    for name in (
        "scientific_config.json",
        "run_manifest.json",
        "resource_ledger.json",
        "backend_decision.json",
    ):
        assert (run / name).is_file()
    assert _json(run / "run_manifest.json")["deep_predecessor_binding_unavailable"] == 1


def test_family_quarter_captures_are_bound_to_committed_shard_outputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "captures"
    sequence = cli._reverse_sequence(cli.SHORT_ANCHOR)
    initial = np.full((3, 784), 1.0 / 784.0, dtype=np.float64)
    saved = {"start": initial.copy()}
    captures = cli._recovery_capture_coordinates(sequence)
    for index in range(len(sequence) // 56):
        state = initial.copy()
        state[:, 0] += (index + 1) * 1.0e-9
        state[:, 1] -= (index + 1) * 1.0e-9
        cli._atomic_npz(root / f"shard-{index:04d}.npz", state=state)
        name = captures.get(sequence[(index + 1) * 56 - 1])
        if name is not None:
            saved[name] = state.copy()
    cli._verify_recovery_family_captures(
        root, initial=initial, sequence=sequence, saved=saved
    )
    tampered = {name: value.copy() for name, value in saved.items()}
    tampered["progress_50"][:, 0] += 1.0e-10
    tampered["progress_50"][:, 1] -= 1.0e-10
    with pytest.raises(cli.ArtifactCompatibilityError, match="progress_50"):
        cli._verify_recovery_family_captures(
            root, initial=initial, sequence=sequence, saved=tampered
        )


def test_evaluation_family_verifier_rejects_derived_metric_and_image_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "evaluation-derived"
    run.mkdir()
    state = np.full((3, 784), 1.0 / 784.0, dtype=np.float64)
    saved = {
        name: state.copy()
        for name in ("start", "progress_25", "progress_50", "progress_75", "final")
    }
    family_summary = cli._semantic(
        {"result": {}, "saved_states_artifact": {}, "passed": 1}
    )
    monkeypatch.setattr(
        cli,
        "_verify_recovery_completed_family_artifacts",
        lambda *_args, **_kwargs: (family_summary, saved),
    )
    monkeypatch.setattr(cli, "_source_arrays", lambda _run: (state[0], state[0]))
    monkeypatch.setattr(
        cli,
        "_recovery_mechanism_rows",
        lambda _result: [
            {
                "row_index": index,
                "score_rms": 0.0,
                "logistic_shift_rms": 0.0,
                "reference_fraction_displacement_rms": 0.0,
                "control_fraction_displacement_rms": 0.0,
                "control_reference_displacement_ratio": 0.0,
                "target_oracle_unreachable_boundary_count": 0,
            }
            for index in range(3)
        ],
    )
    result = type(
        "Result",
        (),
        {"saved_states": saved, "per_row_diagnostics": _mechanism_rows_fixture()},
    )()
    record = cli._commit_recovery_evaluation_family(
        run,
        horizon="short",
        result=result,
        backend="exact",
        audit=None,
        family_summary=family_summary,
    )
    raw_path = run / record["raw_states"]["path"]
    assert raw_path.is_file()
    changed = json.loads(
        json.dumps(
            {key: value for key, value in record.items() if key != "semantic_sha256"}
        )
    )
    changed["row_results"][1]["metrics_to_mixed_target"]["progress_25"][
        "squared_l2_error"
    ] += 7.0
    cli.atomic_write_json(
        run / "objective/evaluation-short/evaluation_family.json",
        cli._semantic(changed),
    )
    args = cli.parse_args(["--test-only", "--device", "cpu"])
    with pytest.raises(cli.ArtifactCompatibilityError, match="derived row evidence"):
        cli._verify_recovery_evaluation_family_artifact(
            run, args, horizon="short", selected_gain=1.0, anchor=state[0]
        )
    cli.atomic_write_json(
        run / "objective/evaluation-short/evaluation_family.json", record
    )
    image = run / record["row_results"][0]["images"]["start"]["raw"]
    image.write_bytes(image.read_bytes() + b"tamper")
    with pytest.raises(cli.ArtifactCompatibilityError, match="image changed"):
        cli._verify_recovery_evaluation_family_artifact(
            run, args, horizon="short", selected_gain=1.0, anchor=state[0]
        )


def test_complete_reverse_time_is_scoped_by_family_backend_and_row_count(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    cli.atomic_write_json(
        metrics / "recovery_attempt_timing_development-core-short-exact.json",
        cli._semantic(
            {
                "schema": cli.RECOVERY_RUN_SCHEMA + "-attempt-timing",
                "schema_version": 1,
                "family_name": "development-core",
                "segment_name": "short",
                "backend": "exact",
                "row_count": 3,
                "verified_shard_count": 2,
                "commit_verification_overhead_seconds": 4.0,
            }
        ),
    )
    # This deliberately enormous unrelated overhead must not dilute or inflate
    # the exact three-row projection.
    cli.atomic_write_json(
        metrics / "recovery_attempt_timing_other-short-candidate.json",
        cli._semantic(
            {
                "schema": cli.RECOVERY_RUN_SCHEMA + "-attempt-timing",
                "schema_version": 1,
                "family_name": "other",
                "segment_name": "short",
                "backend": "candidate",
                "row_count": 1,
                "verified_shard_count": 100,
                "commit_verification_overhead_seconds": 1000.0,
            }
        ),
    )
    assert cli._complete_reverse_shard_seconds(
        tmp_path,
        [10.0, 20.0],
        family_name="development-core",
        segment_name="short",
        backend="exact",
        row_count=3,
    ) == [12.0, 22.0]
    with pytest.raises(cli.ArtifactCompatibilityError, match="scope changed"):
        cli._complete_reverse_shard_seconds(
            tmp_path,
            [10.0, 20.0],
            family_name="development-core",
            segment_name="short",
            backend="exact",
            row_count=1,
        )


def test_backend_opening_reconstructs_selection_inputs_and_allows_prefix_growth(
    tmp_path: Path,
) -> None:
    run = tmp_path / "opening"
    root = run / "objective_attempts/exact/fused_families/development-core/short"
    state = np.full((3, 784), 1.0 / 784.0, dtype=np.float64)

    def commit(index: int, elapsed: float) -> dict[str, object]:
        artifact = cli._atomic_npz(root / f"shard-{index:04d}.npz", state=state)
        record = cli._semantic(
            {
                "schema": "selection-fixture-shard",
                "schema_version": 1,
                "shard_index": index,
                "committed": 1,
                "elapsed_seconds": elapsed,
                "state_file_sha256": artifact["sha256"],
                "state_file_size": artifact["size"],
                "row_table": [{"row": value} for value in range(3)],
                "diagnostics": {
                    "reference": {
                        "maximum_cuda_memory_allocated": 1,
                        "total_cuda_memory_bytes": 10,
                    }
                },
            }
        )
        cli.atomic_write_json(root / f"shard-{index:04d}.json", record)
        return record

    first = commit(0, 10.0)
    timing_path = run / (
        "metrics/recovery_attempt_timing_development-core-short-exact.json"
    )
    cli.atomic_write_json(
        timing_path,
        cli._semantic(
            {
                "schema": cli.RECOVERY_RUN_SCHEMA + "-attempt-timing",
                "schema_version": 1,
                "family_name": "development-core",
                "segment_name": "short",
                "backend": "exact",
                "row_count": 3,
                "verified_shard_count": 1,
                "raw_shard_elapsed_seconds": 10.0,
                "commit_verification_overhead_seconds": 2.0,
            }
        ),
    )
    args = argparse.Namespace(
        reference_backend="auto", test_only=False, maximum_main_seconds=1000.0
    )
    snapshot = cli._recovery_selection_resource_snapshot(run)
    usage = cli._selection_snapshot_usage(snapshot)
    projection = cli._recovery_resource_projection(
        active_seconds=usage["active_seconds"],
        wasted_active_seconds=usage["wasted_active_seconds"],
        observed_shard_seconds=[12.0],
        remaining_shards=15,
        maximum_main_seconds=1000.0,
        persisted_bytes=usage["persisted_bytes"],
        projected_additional_bytes=15 * int(first["state_file_size"]),
        peak_memory_fraction=usage["peak_memory_fraction"],
    )
    assert projection["passed"] == 1
    opening = cli._semantic(
        {
            "schema": cli.RECOVERY_RUN_SCHEMA + "-backend-selection-opening",
            "schema_version": 1,
            "requested": "auto",
            "selected": "exact",
            "exact_audit_semantic_sha256": "a" * 64,
            "exact_projection": projection,
            "projection_inputs": cli._recovery_projection_input_commitments(
                run,
                exact_root=root,
                remaining_shards=15,
                projection=projection,
                resource_snapshot=snapshot,
                test_only=False,
            ),
        }
    )
    assert cli._verify_recovery_backend_opening(
        run,
        args,
        opening=opening,
        exact_root=root,
        total_shards=16,
        expected_schema=cli.RECOVERY_RUN_SCHEMA + "-backend-selection-opening",
        expected_audit_sha256="a" * 64,
    ) == "exact"

    # A later verified exact shard/timing extension is not part of the frozen
    # selection slice and must not make a valid interruption unresumable.
    commit(1, 20.0)
    cli.atomic_write_json(
        timing_path,
        cli._semantic(
            {
                "schema": cli.RECOVERY_RUN_SCHEMA + "-attempt-timing",
                "schema_version": 1,
                "family_name": "development-core",
                "segment_name": "short",
                "backend": "exact",
                "row_count": 3,
                "verified_shard_count": 2,
                "raw_shard_elapsed_seconds": 30.0,
                "commit_verification_overhead_seconds": 4.0,
            }
        ),
    )
    assert cli._verify_recovery_backend_opening(
        run,
        args,
        opening=opening,
        exact_root=root,
        total_shards=16,
        expected_schema=cli.RECOVERY_RUN_SCHEMA + "-backend-selection-opening",
        expected_audit_sha256="a" * 64,
    ) == "exact"

    # Coordinated rehashing of the opening and decision cannot flip selection:
    # the projection is reconstructed from the frozen resource evidence.
    tampered = {
        key: value for key, value in opening.items() if key != "semantic_sha256"
    }
    tampered["selected"] = "candidate"
    changed_projection = dict(tampered["exact_projection"])
    changed_projection["passed"] = 0
    changed_projection["checks"] = {
        **changed_projection["checks"],
        "main_time": False,
    }
    tampered["exact_projection"] = cli._semantic(
        {
            key: value
            for key, value in changed_projection.items()
            if key != "semantic_sha256"
        }
    )
    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._verify_recovery_backend_opening(
            run,
            args,
            opening=cli._semantic(tampered),
            exact_root=root,
            total_shards=16,
            expected_schema=cli.RECOVERY_RUN_SCHEMA
            + "-backend-selection-opening",
            expected_audit_sha256="a" * 64,
        )

    cli.atomic_write_json(run / "backend_selection_opening.json", opening)
    switch_args = argparse.Namespace(
        reference_backend="auto", test_only=False, maximum_main_seconds=100.0
    )
    switch = cli._recovery_switch_event(
        run,
        switch_args,
        exact_root=root,
        total_shards=16,
        exact_audit_sha256="a" * 64,
        selection_opening_sha256=opening["semantic_sha256"],
        schema=cli.RECOVERY_RUN_SCHEMA + "-backend-switch-event",
        restart_field="restart_from_original_anchor",
    )
    assert cli._verify_recovery_switch_event(
        run,
        switch_args,
        event=switch,
        exact_root=root,
        total_shards=16,
        exact_audit_sha256="a" * 64,
        selection_opening_sha256=opening["semantic_sha256"],
        expected_schema=cli.RECOVERY_RUN_SCHEMA + "-backend-switch-event",
        restart_field="restart_from_original_anchor",
    )["selected"] == "candidate"
    forged = {
        key: value for key, value in switch.items() if key != "semantic_sha256"
    }
    forged_projection = {
        key: value
        for key, value in switch["failed_exact_projection"].items()
        if key != "semantic_sha256"
    }
    forged_projection["active_seconds"] -= 1.0
    forged["failed_exact_projection"] = cli._semantic(forged_projection)
    forged["exact_projection"] = forged["failed_exact_projection"]
    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._verify_recovery_switch_event(
            run,
            switch_args,
            event=cli._semantic(forged),
            exact_root=root,
            total_shards=16,
            exact_audit_sha256="a" * 64,
            selection_opening_sha256=opening["semantic_sha256"],
            expected_schema=cli.RECOVERY_RUN_SCHEMA + "-backend-switch-event",
            restart_field="restart_from_original_anchor",
        )


def test_exact_shard_zero_hard_cap_is_checked_before_sampler_authority(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(maximum_main_seconds=299.0, test_only=False)
    with pytest.raises(cli.RolloutCLIError) as captured:
        cli._recovery_exact_shard_zero_admission(
            tmp_path, args, mandatory_core=True
        )
    assert captured.value.failure_code == "exact_audit_resource_blocked"


def _load_bearing_fused_telemetry_fixture(
    contract: str,
    *,
    row_key: str = "learned",
    controller_kind: str = "learned",
    gain: float | None = 1.0,
    transition_count: int = 392,
) -> tuple[dict, list[dict]]:
    prefixes = (
        "reference_fraction_displacement",
        "control_fraction_displacement",
        "score",
        "logistic_shift",
    )
    phase: dict[str, object] = {
        "row_key": row_key,
        "transition_count": transition_count,
        "boundary_fraction_count": 0,
        "maximum_pair_mass_error": 0.0,
        "maximum_simplex_mass_error": 0.0,
        **{name: 0 for name in cli._FUSED_INTERFACE_INVALID_FIELDS},
    }
    for prefix in prefixes:
        phase[f"{prefix}_count"] = 1
        phase[f"{prefix}_squared_sum"] = 1.0
        phase[f"{prefix}_maximum_absolute"] = 1.0
        phase[f"{prefix}_rms"] = 1.0
    if contract == "certified_exact":
        phase.update(
            {
                "reference_transition_count": transition_count,
                "reference_active_count": transition_count,
                "reference_certified_count": transition_count,
                "reference_fallback_count": 0,
                "reference_unauthorized_count": 0,
                "reference_invalid_count": 0,
                "reference_certificate_fraction": 1.0,
            }
        )
    else:
        phase.update(
            {
                "reference_transition_count": transition_count,
                "reference_active_count": transition_count,
                "reference_structural_noop_count": 0,
                "reference_approximation_count": transition_count,
                "reference_invalid_count": 0,
                "reference_maximum_candidate_bracket_width": 1.0e-12,
                "reference_certificate_fraction": "not_applicable",
            }
        )
    controller: dict[str, object] = {
        "row_key": row_key,
        "controller_kind": controller_kind,
        "gain": gain,
        "call_count": 1,
        "lane_count": transition_count,
        "score_count": 1,
        "score_squared_sum": 1.0,
        "score_maximum_absolute": 1.0,
        "unscaled_score_squared_sum": 1.0,
        "unscaled_score_maximum_absolute": 1.0,
        "score_rms": 1.0,
        "unscaled_score_rms": 1.0,
        "movable_count": 0,
        "already_equal_count": 0,
        "zero_pair_mass_count": 0,
        "zero_duration_count": 0,
        "target_oracle_unreachable_boundary_count": 0,
        "clipping_count": 0,
        "floor_count": 0,
        "projection_count": 0,
        "nonfinite_score_count": 0,
    }
    return {
        "per_row_diagnostics": [phase],
        "controller_diagnostics": [controller],
    }, [{"row_key": row_key, "controller_kind": controller_kind, "gain": gain}]


@pytest.mark.parametrize("contract", ["certified_exact", "candidate_approximate_v1"])
@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("per_row_diagnostics", "input_invalid"),
        ("per_row_diagnostics", "control_fraction_displacement_count"),
        ("per_row_diagnostics", "score_squared_sum"),
        ("controller_diagnostics", "score_squared_sum"),
        ("controller_diagnostics", "target_oracle_unreachable_boundary_count"),
    ],
)
def test_fused_prefix_telemetry_is_fail_closed(
    contract: str, section: str, field: str
) -> None:
    record, rows = _load_bearing_fused_telemetry_fixture(contract)
    cli._verify_fused_load_bearing_row_telemetry(
        record,
        expected_rows=rows,
        reference_contract=contract,
        expected_transition_count_per_row=392,
    )
    changed = json.loads(json.dumps(record))
    del changed[section][0][field]
    with pytest.raises(cli.ArtifactCompatibilityError, match="telemetry"):
        cli._verify_fused_load_bearing_row_telemetry(
            changed,
            expected_rows=rows,
            reference_contract=contract,
            expected_transition_count_per_row=392,
        )


@pytest.mark.parametrize("contract", ["certified_exact", "candidate_approximate_v1"])
def test_fused_prefix_rejects_invalid_and_falsified_mechanism_telemetry(
    contract: str,
) -> None:
    record, rows = _load_bearing_fused_telemetry_fixture(contract)
    invalid = json.loads(json.dumps(record))
    invalid["per_row_diagnostics"][0]["score_invalid"] = 1
    with pytest.raises(cli.ArtifactCompatibilityError, match="invalid"):
        cli._verify_fused_load_bearing_row_telemetry(
            invalid,
            expected_rows=rows,
            reference_contract=contract,
            expected_transition_count_per_row=392,
        )
    falsified = json.loads(json.dumps(record))
    falsified["controller_diagnostics"][0]["score_squared_sum"] = 4.0
    with pytest.raises(cli.ArtifactCompatibilityError, match="RMS"):
        cli._verify_fused_load_bearing_row_telemetry(
            falsified,
            expected_rows=rows,
            reference_contract=contract,
            expected_transition_count_per_row=392,
        )


@pytest.mark.parametrize("contract", ["certified_exact", "candidate_approximate_v1"])
def test_fused_prefix_rejects_row_local_authority_shift(contract: str) -> None:
    first, first_rows = _load_bearing_fused_telemetry_fixture(
        contract, row_key="first"
    )
    second, second_rows = _load_bearing_fused_telemetry_fixture(
        contract, row_key="second"
    )
    record = {
        "per_row_diagnostics": [
            first["per_row_diagnostics"][0],
            second["per_row_diagnostics"][0],
        ],
        "controller_diagnostics": [
            first["controller_diagnostics"][0],
            second["controller_diagnostics"][0],
        ],
    }
    rows = [*first_rows, *second_rows]
    cli._verify_fused_load_bearing_row_telemetry(
        record,
        expected_rows=rows,
        reference_contract=contract,
        expected_transition_count_per_row=392,
    )
    changed = json.loads(json.dumps(record))
    field = (
        "reference_certified_count"
        if contract == "certified_exact"
        else "reference_approximation_count"
    )
    changed["per_row_diagnostics"][0][field] -= 1
    changed["per_row_diagnostics"][1][field] += 1
    if contract == "certified_exact":
        changed["per_row_diagnostics"][0]["reference_certificate_fraction"] = 0.5
    with pytest.raises(cli.ArtifactCompatibilityError, match="authority"):
        cli._verify_fused_load_bearing_row_telemetry(
            changed,
            expected_rows=rows,
            reference_contract=contract,
            expected_transition_count_per_row=392,
        )
    redistributed = json.loads(json.dumps(record))
    authority_fields = [
        "transition_count",
        "reference_transition_count",
        "reference_active_count",
        field,
    ]
    for name in authority_fields:
        redistributed["per_row_diagnostics"][0][name] -= 1
        redistributed["per_row_diagnostics"][1][name] += 1
    with pytest.raises(cli.ArtifactCompatibilityError, match="authority"):
        cli._verify_fused_load_bearing_row_telemetry(
            redistributed,
            expected_rows=rows,
            reference_contract=contract,
            expected_transition_count_per_row=392,
        )


@pytest.mark.parametrize(
    ("failure_stage", "expected_complete", "expected_wasted"),
    [("uncommitted-tail", 3.0, 5.0), ("postprocess", 5.0, 3.0)],
)
def test_recovery_attempt_charges_failed_tail_as_wasted_not_shard_overhead(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    expected_complete: float,
    expected_wasted: float,
) -> None:
    from mnist import d0_jacobi_rb_tangent_fused as fused
    from mnist.d0_jacobi_rb_tangent_fused import FusedRowSpec

    run = tmp_path / failure_stage
    run.mkdir()
    cli.atomic_write_json(
        run / "path_usage.json",
        cli._semantic(
            {
                "selected": {
                    "development": cli.PATH_IDS["development"],
                    "evaluation": cli.PATH_IDS["evaluation"],
                    "replication": cli.PATH_IDS["replication"],
                },
                "passed": 1,
            }
        ),
    )
    specs = (
        FusedRowSpec(
            "learned",
            cli.PATH_IDS["development"],
            "learned",
            "learned",
            "short",
            1.0,
        ),
    )
    monkeypatch.setattr(
        cli,
        "_fused_controller_family",
        lambda *_args, **_kwargs: (
            specs,
            object(),
            {"row_table": [row.to_record() for row in specs]},
        ),
    )
    monkeypatch.setattr(cli, "_prepared_fused_reference", lambda *_args: object())
    monkeypatch.setattr(
        cli, "_fused_reference_factory", lambda **_kwargs: lambda _index: object()
    )
    monkeypatch.setattr(cli, "_promote_recovery_failure_states", lambda *_a, **_k: None)

    class Clock:
        value = 0.0

        @classmethod
        def now(cls) -> float:
            return cls.value

    monkeypatch.setattr(cli.time, "perf_counter", Clock.now)
    state = np.full((1, 784), 1.0 / 784.0, dtype=np.float64)
    # Two complete eight-step shards: the callback below is the durable boundary
    # before the uncommitted second shard.
    sequence = cli._reverse_sequence(111)
    transitions = len(sequence) * 4 * 392

    class Result:
        final_state = state
        transition_count = transitions

    def runner(*_args, **kwargs):
        root = (
            Path(kwargs["output_dir"])
            / "fused_families"
            / kwargs["family_name"]
            / kwargs["segment_name"]
        )
        artifact = cli._atomic_npz(root / "shard-0000.npz", state=state)
        cli.atomic_write_json(
            root / "shard-0000.json",
            cli._semantic(
                {
                    "committed": 1,
                    "shard_index": 0,
                    "elapsed_seconds": 2.0,
                    "state_file_sha256": artifact["sha256"],
                    "state_file_size": artifact["size"],
                    "diagnostics": {
                        "reference": {
                            "maximum_cuda_memory_allocated": 0,
                            "total_cuda_memory_bytes": 0,
                        }
                    },
                }
            ),
        )
        if failure_stage == "uncommitted-tail":
            Clock.value = 3.0
            kwargs["before_uncommitted_shard"](None)
            Clock.value = 8.0
            raise RuntimeError("normally closed uncommitted tail")
        Clock.value = 5.0
        return Result()

    monkeypatch.setattr(fused, "run_fused_reverse_family", runner)
    if failure_stage == "postprocess":
        def fail_postprocess(*_args, **_kwargs):
            Clock.value = 8.0
            raise RuntimeError("postprocess failure")

        monkeypatch.setattr(cli, "_recovery_family_health", fail_postprocess)
    args = cli.parse_args(["--test-only", "--device", "cpu"])
    with pytest.raises(Exception, match="uncommitted tail|postprocess failure"):
        cli._run_recovery_core_backend(
            run,
            args,
            anchor=state[0],
            backend="exact",
            sequence=sequence,
            exact_audit_only=True,
            rows=(("learned", "learned", "short", 1.0),),
        )
    timing = _json(
        run
        / "metrics/recovery_attempt_timing_development-core-short-exact.json"
    )
    assert timing["complete_shard_seconds"] == [expected_complete]
    wasted = _json(run / "metrics/wasted_active_seconds.json")
    assert wasted["wasted_active_seconds"] == expected_wasted
    usage = cli._recovery_observed_resource(run)
    assert usage["active_seconds"] == expected_complete
    assert usage["wasted_active_seconds"] == expected_wasted


def test_resume_after_durable_exact_degradation_continues_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kill after the switch seal must not make the candidate path unresumable."""

    from mnist.d0_jacobi_rb_tangent_fused import FusedRowSpec

    run = tmp_path / "degraded-resume"
    exact_root = (
        run
        / "objective_attempts/exact/fused_families/development-core/short"
    )
    state = np.full((3, 784), 1.0 / 784.0, dtype=np.float64)
    artifact = cli._atomic_npz(exact_root / "shard-0000.npz", state=state)
    specs = (
        FusedRowSpec("zero", 17, "zero", "zero", "short"),
        FusedRowSpec("learned", 17, "learned", "learned", "short", 1.0),
        FusedRowSpec("source", 17, "oracle", "oracle", "short"),
    )
    shard = cli._semantic(
        {
            "committed": 1,
            "shard_index": 0,
            "elapsed_seconds": 1.0,
            "state_file_sha256": artifact["sha256"],
            "state_file_size": artifact["size"],
            "row_table": [row.to_record() for row in specs],
        }
    )
    cli.atomic_write_json(exact_root / "shard-0000.json", shard)
    audit = cli._semantic(
        {
            "schema": cli.RECOVERY_RUN_SCHEMA + "-exact-audit-attempt",
            "schema_version": 1,
            "backend": "exact",
            "result": {"row_table": [row.to_record() for row in specs]},
            "passed": 1,
        }
    )
    cli.atomic_write_json(
        run / "exact_audit_attempt_development-core-short.json", audit
    )
    projection = cli._semantic(
        {"schema": "projection", "schema_version": 1, "passed": 1}
    )
    opening = cli._semantic(
        {
            "schema": cli.RECOVERY_RUN_SCHEMA + "-backend-selection-opening",
            "schema_version": 1,
            "requested": "auto",
            "selected": "exact",
            "exact_projection": projection,
            "exact_audit_semantic_sha256": audit["semantic_sha256"],
        }
    )
    cli.atomic_write_json(run / "backend_selection_opening.json", opening)
    degraded = cli._semantic(
        {"schema": "degraded-projection", "schema_version": 1, "passed": 0}
    )
    switch = cli._semantic(
        {
            "schema": cli.RECOVERY_RUN_SCHEMA + "-backend-switch-event",
            "schema_version": 1,
            "selected": "candidate",
            "selection_opening_sha256": opening["semantic_sha256"],
            "failed_exact_projection": degraded,
        }
    )
    cli.atomic_write_json(run / "backend_switch_event.json", switch)
    decision = cli._semantic(
        {
            "schema": cli.RECOVERY_RUN_SCHEMA + "-backend-decision",
            "schema_version": 1,
            "phase": "exact_degraded_then_candidate_selected",
            "requested": "auto",
            "selected": "candidate",
            "exact_audit_shard_committed": 1,
            "exact_audit_outer_steps": 8,
            "exact_audit_semantic_sha256": audit["semantic_sha256"],
            "exact_projection": projection,
            "exact_degraded_projection": degraded,
            "exact_degraded_at_verified_boundary": 1,
            "candidate_restarted_from_original_anchor": 1,
            "mixed_exact_candidate_scientific_prefix": 0,
            "selection_opening_sha256": opening["semantic_sha256"],
            "switch_event_sha256": switch["semantic_sha256"],
            "selected_family_complete": 0,
        }
    )
    cli.atomic_write_json(run / "backend_decision.json", decision)

    monkeypatch.setattr(
        cli, "_recovery_anchors", lambda _run: {cli.SHORT_ANCHOR: state[0]}
    )
    monkeypatch.setattr(
        cli,
        "_recovery_path_ids",
        lambda _run: {"development": 17, "evaluation": 18, "replication": 19},
    )
    monkeypatch.setattr(
        cli,
        "_fused_controller_family",
        lambda *_a, **_k: (
            specs,
            object(),
            {"row_table": [row.to_record() for row in specs]},
        ),
    )
    monkeypatch.setattr(cli, "_verify_fused_family_prefix", lambda *_a, **_k: {})
    monkeypatch.setattr(
        cli, "_verify_recovery_backend_opening", lambda *_a, **_k: "exact"
    )
    monkeypatch.setattr(
        cli, "_verify_recovery_switch_event", lambda *_a, **_k: switch
    )
    monkeypatch.setattr(
        cli, "_complete_reverse_shard_seconds", lambda *_a, **_k: [1.0]
    )
    monkeypatch.setattr(
        cli, "_recovery_max_committed_shard_bytes", lambda *_a, **_k: 1
    )
    monkeypatch.setattr(
        cli,
        "_write_recovery_ledger",
        lambda *_a, **_k: cli._semantic(
            {"schema": "ledger", "schema_version": 1, "passed": 1}
        ),
    )

    class CandidateContinuationReached(RuntimeError):
        pass

    def continuation(*_args, **kwargs):
        assert kwargs["backend"] == "candidate"
        assert np.array_equal(kwargs["anchor"], state[0])
        raise CandidateContinuationReached

    monkeypatch.setattr(cli, "_run_recovery_core_backend", continuation)
    args = argparse.Namespace(
        exact_audit_outer_steps=8,
        core_learned_gain=1.0,
        reference_backend="auto",
        maximum_main_seconds=21_600.0,
        test_only=False,
    )
    with pytest.raises(CandidateContinuationReached):
        cli._recovery_production_core(run, args)


@pytest.mark.parametrize("committed_count", [1, 2])
def test_hard_crash_orphan_shard_timing_is_reconciled_without_retiming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, committed_count: int
) -> None:
    """An atomic shard newer than timing remains resumable with unknown wall time."""

    from mnist import d0_jacobi_rb_tangent_fused as fused
    from mnist.d0_jacobi_rb_tangent_fused import FusedRowSpec

    run = tmp_path / f"orphan-{committed_count}"
    run.mkdir()
    cli.atomic_write_json(
        run / "path_usage.json",
        cli._semantic(
            {
                "selected": {
                    "development": cli.PATH_IDS["development"],
                    "evaluation": cli.PATH_IDS["evaluation"],
                    "replication": cli.PATH_IDS["replication"],
                },
                "passed": 1,
            }
        ),
    )
    spec = FusedRowSpec(
        "learned",
        cli.PATH_IDS["development"],
        "learned",
        "learned",
        "short",
        1.0,
    )
    monkeypatch.setattr(
        cli,
        "_fused_controller_family",
        lambda *_a, **_k: (
            (spec,),
            object(),
            {"row_table": [spec.to_record()]},
        ),
    )
    monkeypatch.setattr(cli, "_verify_fused_family_prefix", lambda *_a, **_k: {})
    monkeypatch.setattr(cli, "_prepared_fused_reference", lambda *_a: object())
    monkeypatch.setattr(
        cli, "_fused_reference_factory", lambda **_k: lambda _index: object()
    )
    monkeypatch.setattr(cli, "_promote_recovery_failure_states", lambda *_a, **_k: None)
    root = run / "objective_attempts/exact/fused_families/development-core/short"
    state = np.full((1, 784), 1.0 / 784.0, dtype=np.float64)
    for index in range(committed_count):
        artifact = cli._atomic_npz(root / f"shard-{index:04d}.npz", state=state)
        cli.atomic_write_json(
            root / f"shard-{index:04d}.json",
            cli._semantic(
                {
                    "committed": 1,
                    "shard_index": index,
                    "elapsed_seconds": float(index + 2),
                    "state_file_sha256": artifact["sha256"],
                    "state_file_size": artifact["size"],
                }
            ),
        )
    timing_path = (
        run / "metrics/recovery_attempt_timing_development-core-short-exact.json"
    )
    prior_count = committed_count - 1
    cli.atomic_write_json(
        timing_path,
        cli._semantic(
            {
                "schema": cli.RECOVERY_RUN_SCHEMA + "-attempt-timing",
                "schema_version": 1,
                "family_name": "development-core",
                "segment_name": "short",
                "backend": "exact",
                "row_count": 1,
                "elapsed_seconds_through_commit_and_restart_verification": float(
                    sum(range(2, 2 + prior_count))
                ),
                "raw_shard_elapsed_seconds": float(sum(range(2, 2 + prior_count))),
                "commit_verification_overhead_seconds": 0.0,
                "complete_shard_seconds": [
                    float(value) for value in range(2, 2 + prior_count)
                ],
                "verified_shard_count": prior_count,
                "completed_attempt_never_retimed": 1,
            }
        ),
    )

    class StopAfterReconcile(RuntimeError):
        pass

    def stop(*_a, **_k):
        raise StopAfterReconcile

    monkeypatch.setattr(fused, "run_fused_reverse_family", stop)
    args = cli.parse_args(["--test-only", "--device", "cpu"])
    with pytest.raises(cli.RolloutCLIError):
        cli._run_recovery_core_backend(
            run,
            args,
            anchor=state[0],
            backend="exact",
            sequence=cli._reverse_sequence(111),
            exact_audit_only=False,
            rows=(("learned", "learned", "short", 1.0),),
        )
    timing = _json(timing_path)
    assert timing["verified_shard_count"] == committed_count
    assert timing["complete_shard_seconds"] == [
        float(value) for value in range(2, 2 + committed_count)
    ]
    unknown = _json(run / "metrics/unknown_active_time.json")
    assert unknown["unknown_active_time"] == 1
