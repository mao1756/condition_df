from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mnist.d0_jacobi_artifacts import atomic_write_json, config_fingerprint, file_fingerprint
from mnist.d0_jacobi_rb_physical_coarse_signal import (
    PANEL_A_PATH_IDS,
    PANEL_B_PATH_IDS,
    frozen_path_plan,
    frozen_statistic_plan,
)
from mnist import diag_d0_jacobi_rb_physical_coarse_signal_witness as cli


def _write_sealed_panel(
    run_dir: Path, *, panel: str, path_ids: tuple[int, ...], value: float
) -> None:
    means = np.full((64, 4, 7, 392), value, dtype=np.float64)
    data_path = run_dir / "panels" / panel / "cell_means.npz"
    cli._atomic_npz(
        data_path,
        {
            "path_ids": np.asarray(path_ids, dtype=np.int64),
            "cell_means": means,
        },
    )
    metrics_path = run_dir / "panels" / panel / "metrics.json"
    audit_path = run_dir / f"panel_{panel}_cell_mean_persistence_audit.json"
    resource_path = run_dir / f"panel_{panel}_resource_summary.json"
    atomic_write_json(metrics_path, {"panel": panel, "fixture": 1})
    atomic_write_json(audit_path, {"panel": panel, "passed": 1})
    atomic_write_json(resource_path, {"panel": panel, "peak_memory_fraction": 0.0})
    atomic_write_json(
        run_dir / f"panel_{panel}_seal.json",
        {
            "schema": cli.RUN_SCHEMA + "-panel-seal",
            "schema_version": 1,
            "panel": panel,
            "path_ids": list(path_ids),
            "cell_means_file_sha256": file_fingerprint(data_path),
            "cell_means_array_sha256": cli._array_sha256(means),
            "path_plan_sha256": frozen_path_plan().fingerprint,
            "statistic_plan_sha256": frozen_statistic_plan().fingerprint,
            "execution_metrics_file_sha256": file_fingerprint(metrics_path),
            "persistence_audit_file_sha256": file_fingerprint(audit_path),
            "resource_summary_file_sha256": file_fingerprint(resource_path),
        },
    )


def test_parse_args_exposes_frozen_stage_and_gate_contract() -> None:
    args = cli.parse_args(
        [
            "--stage",
            "panel-a",
            "--require-gate",
            "panel-a",
            "--parent-one-image-run-dir",
            "physical",
            "--parent-zero-signal-run-dir",
            "zero",
            "--parent-bayes-power-run-dir",
            "bayes",
        ]
    )
    assert args.stage == "panel-a"
    assert args.require_gate == "panel-a"
    assert args.device == "cuda"


def test_source_fingerprint_covers_authorizing_transitive_dependencies() -> None:
    names = {path.name for path in cli._source_paths()}
    assert {
        "d0_jacobi_rb_learnability.py",
        "d0_jacobi_rb_cuda_controls.py",
        "d0_jacobi_rb_spectral.py",
        "d0_jacobi_rb_cuda_certificate.py",
        "d0_jacobi_rb_cuda_fused.py",
        "d0_jacobi_rb_controls.py",
        "d0_jacobi_denoising.py",
    } <= names
    scientific = cli._scientific_config()
    assert scientific["tau_eff"] == 5.0e-5
    assert scientific["phase_durations"] == [0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5]
    assert scientific == json.loads(json.dumps(scientific))


def test_reduced_capture_benchmark_persists_only_restart_accumulators(
    tmp_path: Path, monkeypatch
) -> None:
    calls = 0

    def fake_scheduler(states, *, capture_training_payload, **_kwargs):
        nonlocal calls
        calls += 1
        final = states.detach().cpu().numpy().copy()
        transition_count = 8 * 8 * 7 * 392
        diagnostics = {
            "transition_count": transition_count,
            "certified_count": transition_count,
            "fallback_count": 0,
            "fallback_elapsed_seconds": 0.0,
            "maximum_mass_error": 0.0,
            **{name: 0 for name in cli.FORBIDDEN_COUNTS},
        }
        payload = (
            SimpleNamespace(
                denoising_targets=np.full(
                    (56, 8, 392), 0.125, dtype=np.float64
                ),
                path_ids=tuple(int(value) for value in _kwargs["path_ids"]),
                outer_steps=np.repeat(
                    np.arange(
                        int(_kwargs["start_step"]),
                        int(_kwargs["start_step"]) + 8,
                        dtype=np.int64,
                    ),
                    7,
                ),
                phases=np.tile(np.arange(7, dtype=np.int64), 8),
            )
            if capture_training_payload
            else None
        )
        return SimpleNamespace(
            committed_final_states=final,
            capture_payload=payload,
            diagnostics=diagnostics,
            to_record=lambda: {"diagnostics": diagnostics},
        )

    monkeypatch.setattr(cli, "run_exact_multipath_shard", fake_scheduler)
    record = cli._benchmark_capture(
        tmp_path,
        mixed_target=np.full(784, 1.0 / 784.0, dtype=np.float64),
        device=cli.torch.device("cpu"),
        scientific_config_sha256="a" * 64,
    )
    assert calls == 2
    assert record["raw_target_observations_persisted"] == 0
    with np.load(
        tmp_path / "preflight_benchmark_restart_state.npz",
        allow_pickle=False,
    ) as restart:
        assert set(restart.files) == {"final_states", "path_ids"}
    with np.load(
        tmp_path / "preflight_benchmark_accumulator.npz",
        allow_pickle=False,
    ) as accumulator:
        assert set(accumulator.files) == {
            "cell_compensations",
            "cell_counts",
            "cell_sums",
        }


def test_sealed_analysis_runs_once_and_writes_complete_evidence(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_sealed_panel(
        run_dir, panel="a", path_ids=PANEL_A_PATH_IDS, value=0.02
    )
    _write_sealed_panel(
        run_dir, panel="b", path_ids=PANEL_B_PATH_IDS, value=0.03
    )
    for name in ("panel-a", "panel-b"):
        atomic_write_json(
            run_dir / cli._GATE_FILES[name],
            {"evaluation_status": "evaluated", "passed": 1},
        )
    seal_a = cli._load_json(run_dir / "panel_a_seal.json")
    seal_b = cli._load_json(run_dir / "panel_b_seal.json")
    joint = {
        "schema": cli.RUN_SCHEMA + "-joint-analysis-seal",
        "schema_version": 1,
        "panel_a_seal_sha256": config_fingerprint(seal_a),
        "panel_b_seal_sha256": config_fingerprint(seal_b),
        "panel_a_file_sha256": seal_a["cell_means_file_sha256"],
        "panel_b_file_sha256": seal_b["cell_means_file_sha256"],
        "statistic_plan_sha256": frozen_statistic_plan().fingerprint,
        "analysis_definition_frozen_before_open": 1,
        "analysis_open_count": 0,
    }
    atomic_write_json(run_dir / "joint_analysis_seal.json", joint)

    gate = cli._analysis_stage(run_dir)
    assert gate["passed"] == 1
    result = cli._load_json(run_dir / "physical_coarse_signal_analysis.json")
    assert (
        result["classification"]["decision"]
        == "exact_physical_coarse_signal_detected"
    )
    assert cli._load_json(run_dir / "analysis_open.json")["analysis_open_count"] == 1
    first_open = (run_dir / "analysis_open.json").read_bytes()
    first_analysis = (run_dir / "physical_coarse_signal_analysis.json").read_bytes()
    first_gate = (run_dir / cli._GATE_FILES["witness"]).read_bytes()
    first_completion = (run_dir / "analysis_completion.json").read_bytes()
    replay = cli._analysis_stage(run_dir)
    assert replay["passed"] == 1
    assert (run_dir / "analysis_open.json").read_bytes() == first_open
    assert (
        run_dir / "physical_coarse_signal_analysis.json"
    ).read_bytes() == first_analysis
    assert (run_dir / cli._GATE_FILES["witness"]).read_bytes() == first_gate
    assert (run_dir / "analysis_completion.json").read_bytes() == first_completion


def test_joint_seal_and_orphaned_analysis_finalization_are_idempotent(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_sealed_panel(
        run_dir, panel="a", path_ids=PANEL_A_PATH_IDS, value=0.02
    )
    _write_sealed_panel(
        run_dir, panel="b", path_ids=PANEL_B_PATH_IDS, value=0.03
    )
    for name in ("panel-a", "panel-b"):
        atomic_write_json(
            run_dir / cli._GATE_FILES[name],
            {"evaluation_status": "evaluated", "passed": 1},
        )

    cli._ensure_joint_analysis_seal(run_dir)
    first_joint = (run_dir / "joint_analysis_seal.json").read_bytes()
    cli._ensure_joint_analysis_seal(run_dir)
    assert (run_dir / "joint_analysis_seal.json").read_bytes() == first_joint

    cli._analysis_stage(run_dir)
    expected_decision = (
        run_dir / "physical_coarse_signal_decision.json"
    ).read_bytes()
    expected_completion = (run_dir / "analysis_completion.json").read_bytes()
    (run_dir / "physical_coarse_signal_decision.json").unlink()
    (run_dir / "analysis_completion.json").unlink()

    replay = cli._analysis_stage(run_dir)
    assert replay["passed"] == 1
    assert (
        run_dir / "physical_coarse_signal_decision.json"
    ).read_bytes() == expected_decision
    assert (run_dir / "analysis_completion.json").read_bytes() == expected_completion


def test_panel_seal_binds_supporting_execution_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_sealed_panel(
        run_dir, panel="a", path_ids=PANEL_A_PATH_IDS, value=0.02
    )
    cli._validate_panel_seal(run_dir, "a")
    atomic_write_json(
        run_dir / "panel_a_resource_summary.json",
        {"panel": "a", "peak_memory_fraction": 0.5},
    )
    with pytest.raises(cli.ArtifactCompatibilityError, match="panel a seal changed"):
        cli._validate_panel_seal(run_dir, "a")


def test_execution_failed_preflight_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cli._save_gate(
        run_dir,
        "preflight",
        cli.execution_failed_gate(
            "preflight",
            failure_code="transient",
            failure_domain="resource_benchmark",
            message="transient",
        ),
    )

    class RetryReached(RuntimeError):
        pass

    monkeypatch.setattr(
        cli,
        "verify_physical_coarse_signal_parents",
        lambda **_kwargs: (_ for _ in ()).throw(RetryReached()),
    )
    with pytest.raises(RetryReached):
        cli._preflight_stage(
            run_dir,
            args=SimpleNamespace(
                parent_one_image_run_dir="physical",
                parent_zero_signal_run_dir="zero",
                parent_bayes_power_run_dir="bayes",
            ),
            scientific={},
        )


def test_failure_attempt_history_is_append_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    gate = cli.execution_failed_gate(
        "panel-a",
        failure_code="fixture",
        failure_domain="physical_panel_execution",
        message="first",
    )
    decision = cli._failure_decision(
        stage="panel-a",
        failure_code="fixture",
        failure_domain="physical_panel_execution",
        message="first",
    )
    first = cli._record_failure_attempt(
        run_dir, stage="panel-a", gate=gate, decision=decision
    )
    second = cli._record_failure_attempt(
        run_dir, stage="panel-a", gate=gate, decision=decision
    )
    assert first.name == "panel_a-attempt-001.json"
    assert second.name == "panel_a-attempt-002.json"
    assert first.is_file()
    assert second.is_file()


def test_report_preserves_execution_failure_adjudication(tmp_path: Path) -> None:
    run_dir = tmp_path / "failed-run"
    run_dir.mkdir()
    gate = cli.execution_failed_gate(
        "preflight",
        failure_code="benchmark_capture_failed",
        failure_domain="resource_benchmark",
        message="fixture failure",
    )
    cli._save_gate(run_dir, "preflight", gate)
    original_decision = cli._failure_decision(
        stage="preflight",
        failure_code="benchmark_capture_failed",
        failure_domain="resource_benchmark",
        message="fixture failure",
    )
    atomic_write_json(
        run_dir / "physical_coarse_signal_decision.json",
        original_decision,
    )
    original_bytes = (
        run_dir / "physical_coarse_signal_decision.json"
    ).read_bytes()
    code = cli._finalize(
        run_dir, stage="report", required_gate="none"
    )
    decision = cli._load_json(run_dir / "physical_coarse_signal_decision.json")
    status = cli._load_json(run_dir / "run_status.json")
    assert code == 1
    assert decision["decision"] == (
        "physical_coarse_signal_computationally_infeasible"
    )
    assert decision["evaluation_status"] == "execution_failed"
    assert (
        run_dir / "physical_coarse_signal_decision.json"
    ).read_bytes() == original_bytes
    assert status["state"] == "gate_failed"


def test_new_preflight_provenance_failure_commits_readable_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    code = cli.main(
        [
            "--stage",
            "preflight",
            "--require-gate",
            "preflight",
            "--runs-root",
            str(root),
            "--run-name",
            "fixture",
            "--parent-one-image-run-dir",
            str(tmp_path / "missing-physical"),
            "--parent-zero-signal-run-dir",
            str(tmp_path / "missing-zero"),
            "--parent-bayes-power-run-dir",
            str(tmp_path / "missing-bayes"),
        ]
    )
    assert code == 1
    run_dir = next(root.iterdir())
    status = cli._load_json(run_dir / "run_status.json")
    decision = cli._load_json(run_dir / "physical_coarse_signal_decision.json")
    gate = cli._load_json(run_dir / cli._GATE_FILES["preflight"])
    assert status["state"] == "gate_failed"
    assert decision["decision"] == "control_provenance_invalid"
    assert gate["evaluation_status"] == "execution_failed"
    assert (run_dir / "artifact_registry.json").is_file()


def test_all_stage_failure_is_committed_to_the_active_panel_gate(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "all-run"
    run_dir.mkdir()
    args = cli.parse_args(
        [
            "--stage",
            "all",
            "--parent-one-image-run-dir",
            "physical",
            "--parent-zero-signal-run-dir",
            "zero",
            "--parent-bayes-power-run-dir",
            "bayes",
        ]
    )
    monkeypatch.setattr(cli, "_make_run_dir", lambda _args: (run_dir, False))
    monkeypatch.setattr(
        cli,
        "_initialize_or_validate",
        lambda *_args, **_kwargs: ({}, {"semantic_sha256": "a" * 64}, ()),
    )

    def passing_preflight(*_args, **_kwargs):
        gate = {
            "gate": "preflight",
            "evaluation_status": "evaluated",
            "passed": 1,
        }
        cli._save_gate(run_dir, "preflight", gate)
        return gate

    monkeypatch.setattr(cli, "_preflight_stage", passing_preflight)
    monkeypatch.setattr(
        cli,
        "_panel_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            cli.CoarseWitnessCLIError(
                "panel fixture failed",
                failure_domain="physical_panel_execution",
                failure_code="physical_panel_fixture_failed",
            )
        ),
    )
    assert cli._run(args) == 1
    assert cli._load_json(
        run_dir / cli._GATE_FILES["panel-a"]
    )["evaluation_status"] == "execution_failed"
    assert cli._load_json(
        run_dir / cli._GATE_FILES["preflight"]
    )["evaluation_status"] == "evaluated"


def test_cuda_certificate_failure_is_classified_as_numerical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = cli.parse_args(
        [
            "--stage",
            "preflight",
            "--parent-one-image-run-dir",
            "physical",
            "--parent-zero-signal-run-dir",
            "zero",
            "--parent-bayes-power-run-dir",
            "bayes",
        ]
    )
    monkeypatch.setattr(cli, "_make_run_dir", lambda _args: (run_dir, False))
    monkeypatch.setattr(
        cli,
        "_initialize_or_validate",
        lambda *_args, **_kwargs: ({}, {"semantic_sha256": "a" * 64}, ()),
    )
    monkeypatch.setattr(
        cli,
        "_preflight_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            cli.RigorousCudaControlError("certificate fixture")
        ),
    )
    assert cli._run(args) == 1
    decision = cli._load_json(
        run_dir / "physical_coarse_signal_decision.json"
    )
    assert decision["decision"] == (
        "physical_coarse_signal_numerically_unresolved"
    )
    assert decision["failure_code"] == "certified_jacobi_transition_failed"
