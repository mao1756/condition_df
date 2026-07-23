from __future__ import annotations

import json
from pathlib import Path

import pytest

import mnist.diag_d0_score_boundary_controls as boundary_cli


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_incompatible_resume_preserves_all_existing_terminal_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "completed-boundary-control-run"
    run_dir.mkdir()
    terminal_names = (
        "run_status.json",
        "boundary_control_report.json",
        "boundary_control_gate.json",
        "control_repair_decision.json",
        "artifact_registry.json",
        "run_manifest.json",
    )
    for index, name in enumerate(terminal_names):
        _write_json(run_dir / name, {"fixture": name, "revision": index})
    _write_json(
        run_dir / "failed_run_provenance.json",
        {"schema": "deliberately-incompatible-frozen-provenance"},
    )
    before = {name: (run_dir / name).read_bytes() for name in terminal_names}

    parent = {
        "passed": 1,
        "run_dir": str((tmp_path / "parent").resolve()),
        "scientific_fingerprint": "parent-science",
        "kernel": dict(boundary_cli.EXPECTED_KERNEL),
        "schedule_metadata": {"horizon": 1.0},
        "artifacts": {"status": {"sha256": "parent-status"}},
        "failed_teacher_gate": {"passed": 0},
        "failed_null_gate": {"passed": 0},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    monkeypatch.setattr(boundary_cli, "configure_exact_torch_backend", lambda _device: {})
    monkeypatch.setattr(boundary_cli, "verify_failed_score_run", lambda _path: parent)
    args = boundary_cli.parse_args(
        [
            "--resume-run-dir",
            str(run_dir),
            "--failed-score-run-dir",
            str(tmp_path / "parent"),
            "--stage",
            "all",
            "--device",
            "cpu",
            "--require-gate",
            "controls",
            "--no-progress",
        ]
    )

    try:
        exit_code = boundary_cli._run(args)
    except boundary_cli.ArtifactCompatibilityError:
        pass
    else:
        assert exit_code == 2

    after = {name: (run_dir / name).read_bytes() for name in terminal_names}
    assert after == before


def test_exception_run_is_an_implementation_failure_even_when_no_gate_is_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(boundary_cli, "configure_exact_torch_backend", lambda _device: {})
    exit_code = boundary_cli.main(
        [
            "--runs-root",
            str(tmp_path / "runs"),
            "--run-name",
            "missing-parent-no-required-gate",
            "--device",
            "cpu",
            "--stage",
            "preflight",
            "--failed-score-run-dir",
            str(tmp_path / "missing-parent"),
            "--require-gate",
            "none",
            "--no-progress",
        ]
    )

    assert exit_code == 2
    run_dir = next((tmp_path / "runs").iterdir())
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["outcome"] == "implementation_error"
    assert status["required_gate_pass"] == 0
    assert (run_dir / "failure.json").is_file()


def test_runtime_setup_failure_still_publishes_fail_closed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_backend(_device: object) -> dict[str, object]:
        raise RuntimeError("synthetic CUDA/backend setup failure")

    monkeypatch.setattr(boundary_cli, "configure_exact_torch_backend", fail_backend)
    exit_code = boundary_cli.main(
        [
            "--runs-root",
            str(tmp_path / "runs"),
            "--run-name",
            "runtime-failure",
            "--device",
            "cpu",
            "--stage",
            "preflight",
            "--failed-score-run-dir",
            str(tmp_path / "unused-parent"),
            "--require-gate",
            "none",
            "--no-progress",
        ]
    )
    assert exit_code == 2
    run_dir = next((tmp_path / "runs").iterdir())
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["outcome"] == "implementation_error"
    assert status["phase"] == "runtime_setup_failure"
    assert (run_dir / "failure.json").is_file()
    assert (run_dir / "artifact_registry.json").is_file()
