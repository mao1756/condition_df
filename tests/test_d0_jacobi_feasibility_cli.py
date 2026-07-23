from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import mnist.diag_d0_jacobi_denoising_feasibility as cli
from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError


def _parent_record(path: Path) -> dict[str, object]:
    return {
        "parent_run_dir": str(path),
        "parent_artifact_registry_sha256": cli.PARENT_REGISTRY_SHA256,
        "parent_artifact_registry_size": 1,
        "parent_artifact_record_count": 277,
        "parent_scientific_fingerprint": "scientific",
        "parent_source_fingerprint": "source",
        "saved_decision": "selection_false_discovery",
        "readjudicated_decision": "h1_strength_grid_unresolved",
        "lineage_registry_record_counts": [277, 301, 263, 123, 125, 332, 222, 381],
        "readjudication_valid": 1,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def test_preflight_writes_readable_evidence_before_returning(tmp_path, monkeypatch) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    monkeypatch.setattr(
        cli, "verify_and_readjudicate_gradient_parent", lambda _: _parent_record(parent)
    )
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda _: {"deterministic": 1})
    result = cli.main([
        "--runs-root", str(tmp_path / "runs"),
        "--run-name", "tiny",
        "--device", "cpu",
        "--stage", "preflight",
        "--parent-gradient-control-run-dir", str(parent),
        "--require-gate", "none",
    ])
    assert result == 0
    run_dir = next((tmp_path / "runs").iterdir())
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    gate = json.loads((run_dir / "jacobi_preflight_gate.json").read_text(encoding="utf-8"))
    decision = json.loads((run_dir / "jacobi_feasibility_decision.json").read_text(encoding="utf-8"))
    assert gate["passed"] == 1
    assert decision["decision"] == "preflight_passed"
    assert status["artifact_registry_sha256"]
    assert (run_dir / "artifact_registry.json").is_file()
    assert status["physical_training_performed"] == 0
    assert status["sampling_performed"] == 0


def test_stage_resume_and_production_override_contract(tmp_path) -> None:
    missing = tmp_path / "missing"
    result = cli.main([
        "--stage", "kernel",
        "--resume-run-dir", str(missing),
        "--parent-gradient-control-run-dir", str(tmp_path / "parent"),
    ])
    assert result == 2


def test_parent_failure_commits_evidence_before_required_gate_failure(tmp_path, monkeypatch) -> None:
    parent = tmp_path / "invalid-parent"
    parent.mkdir()
    monkeypatch.setattr(
        cli,
        "verify_and_readjudicate_gradient_parent",
        lambda _: (_ for _ in ()).throw(ArtifactCompatibilityError("registry mismatch")),
    )
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda _: {"deterministic": 1})
    result = cli.main([
        "--runs-root", str(tmp_path / "runs"),
        "--run-name", "parent-failure",
        "--device", "cpu",
        "--stage", "preflight",
        "--parent-gradient-control-run-dir", str(parent),
        "--require-gate", "preflight",
    ])
    assert result == 2
    run_dir = next((tmp_path / "runs").iterdir())
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    decision = json.loads((run_dir / "jacobi_feasibility_decision.json").read_text(encoding="utf-8"))
    parent_record = json.loads((run_dir / "parent_readjudication.json").read_text(encoding="utf-8"))
    assert decision["decision"] == "control_provenance_invalid"
    assert parent_record["evaluation_status"] == "invalid"
    assert status["outcome"] == "gate_failed"
    assert status["artifact_registry_sha256"]
    assert (run_dir / "artifact_registry.json").is_file()


def test_controls_only_import_does_not_load_physical_training_or_sampler() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; import mnist.diag_d0_jacobi_denoising_feasibility; "
            "assert 'mnist.experiment12_d0' not in sys.modules; "
            "assert 'mnist.eulerian_flux_mnist' not in sys.modules; "
            "assert 'mnist.d0_one_image_sampler' not in sys.modules"
        ),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def test_required_gate_matrix_uses_the_requested_gate_only() -> None:
    passed = {"passed": 1}
    failed = {"passed": 0}
    assert cli._requested_gate_pass("none", preflight=failed, kernel=None, controls=None)
    assert cli._requested_gate_pass("preflight", preflight=passed, kernel=failed, controls=failed)
    assert cli._requested_gate_pass("kernel", preflight=passed, kernel=passed, controls=failed)
    assert not cli._requested_gate_pass("controls", preflight=passed, kernel=passed, controls=failed)


def test_unexpected_post_manifest_failure_is_terminal_and_readable(tmp_path, monkeypatch) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    monkeypatch.setattr(cli, "verify_and_readjudicate_gradient_parent", lambda _: _parent_record(parent))
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda _: {"deterministic": 1})
    monkeypatch.setattr(cli, "_run_preflight", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
    result = cli.main([
        "--runs-root", str(tmp_path / "runs"),
        "--run-name", "unexpected",
        "--device", "cpu",
        "--stage", "preflight",
        "--parent-gradient-control-run-dir", str(parent),
        "--require-gate", "preflight",
    ])
    assert result == 2
    run_dir = next((tmp_path / "runs").iterdir())
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["outcome"] == "gate_failed"
    assert (run_dir / "unexpected_failure.json").is_file()
    assert (run_dir / "artifact_registry.json").is_file()


def test_whole_path_max_t_handles_exact_zero_and_fails_nonzero_degeneracy() -> None:
    passed, records = cli._simultaneous_mean_intervals(
        np.zeros((8, 3)), seed=17, replicates=100
    )
    assert passed
    assert all(record["lower"] == record["upper"] == 0.0 for record in records)
    failed, records = cli._simultaneous_mean_intervals(
        np.ones((8, 2)), seed=17, replicates=100
    )
    assert not failed
    assert all(record["degenerate_nonzero"] == 1 for record in records)


def test_generic_strang_smoke_is_never_authorizing_and_actual_local_generator_agrees() -> None:
    metrics, rows = cli._strang_matrix_refinement()
    assert metrics["actual_eulerian_refinement_pass"] == 0
    assert metrics["split_reference_evaluated"] == 0
    assert "observed_weak_order" not in metrics
    assert rows and all(row["authorizing"] == 0 for row in rows)
    error, generator_rows = cli._edge_generator_observable_fixture()
    assert error <= 1e-8
    assert len(generator_rows) == 12
    assert all(row["authorizing"] == 0 for row in generator_rows)


def test_exact_phase_stationarity_and_detailed_balance_use_whole_paths() -> None:
    args = cli.parse_args([
        "--stage", "preflight",
        "--parent-gradient-control-run-dir", "unused",
        "--kernel-mc-draws", "1",
        "--ancestral-max-count", "512",
        "--ancestral-max-terms", "20000",
        "--ancestral-max-refinements", "5000",
    ])
    metrics, rows = cli._stationarity_detailed_balance_controls(args)
    assert metrics["dirichlet_stationarity_pass"] == 1
    assert metrics["full_sweep_detailed_balance_pass"] == 1
    assert metrics["stationarity_certificate_failure_count"] == 0
    assert metrics["stationarity_inference"].startswith("whole-path")
    assert rows
