from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch
import numpy as np

from mnist import (
    diag_d0_jacobi_rb_boundary_tangent_v3_time_local_adjudication as cli,
)
from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError
from mnist.d0_jacobi_rb_boundary_tangent_v3_time_local import (
    QuadraticRiskDecomposition,
)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        stage="preflight",
        require_gate="preflight",
        parent_memory_v3_run_dir=(tmp_path / "memory").resolve(),
        parent_coarse_witness_run_dir=(tmp_path / "witness").resolve(),
        parent_bayes_power_run_dir=(tmp_path / "bayes").resolve(),
        resume_run_dir=None,
        runs_root=(tmp_path / "runs").resolve(),
        run_name="test",
        device="cpu",
    )


def test_preflight_commits_plan_before_any_checkpoint_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    provenance = {
        "parent_immutability": {
            role: {"unchanged": 1} for role in ("memory", "witness", "bayes")
        },
        "selection_seal_verified": 1,
        "all_checkpoint_hashes_verified": 1,
        "confirmation_namespace_opened": 0,
        "parents_mutated": 0,
    }
    monkeypatch.setattr(cli, "_verify_parents", lambda _args: provenance)
    cli._preflight_stage(run_dir, args)
    plan = cli._load_json(run_dir / "adjudication_plan.json")
    gate = cli._load_json(run_dir / "preflight_gate.json")
    assert plan["created_before_checkpoint_loading"] == 1
    assert [(row["seed"], row["update"]) for row in plan["nominees"]] == [
        (261312, 900),
        (261313, 1600),
        (261314, 3900),
    ]
    assert gate["passed"] == 1
    cli._verify_stage_seal(run_dir, "preflight_artifact_seal.json")


def test_artifact_registry_records_no_work_scope(tmp_path: Path) -> None:
    cli.atomic_write_json(tmp_path / "evidence.json", {"value": 1})
    registry = cli._artifact_registry(tmp_path)
    assert registry["new_exact_transitions"] == 0
    assert registry["physical_training_performed"] == 0
    assert registry["confirmation_evidence_accessed"] == 0
    assert [row["path"] for row in registry["artifacts"]] == ["evidence.json"]


def test_parse_requires_resume_for_later_stages(tmp_path: Path) -> None:
    parents = [
        "--parent-memory-v3-run-dir",
        str(tmp_path / "memory"),
        "--parent-coarse-witness-run-dir",
        str(tmp_path / "witness"),
        "--parent-bayes-power-run-dir",
        str(tmp_path / "bayes"),
    ]
    with pytest.raises(SystemExit):
        cli.parse_args(["--stage", "replay", *parents])


def test_workflow_records_ready_states_without_authority(tmp_path: Path) -> None:
    cli.atomic_write_json(
        tmp_path / "decomposition_metrics.json",
        {
            "quartile_mechanisms": [
                "resolved",
                "directional_alignment_missing",
                "prediction_energy_dominates",
                "positive_but_underpowered",
            ]
        },
    )
    cli.atomic_write_json(
        tmp_path / "preflight_gate.json",
        {
            "evaluation_status": "evaluated",
            "passed": 1,
        },
    )
    workflow = cli._workflow_record(tmp_path, "preflight")
    assert workflow["required_gate_pass"] == 1
    assert workflow["decision"]["decision"] == "ready_for_replay"
    assert workflow["decision"]["fresh_quartile_specialist_planning_authorized"] == 0
    policy = workflow["decision"]["mechanism_conditioned_recommendation_policy"]
    assert "independent width-32 expert" in policy["directional_alignment_missing"]
    assert "training-only" in policy["prediction_energy_dominates"]
    assert "powered fresh" in policy["positive_but_underpowered"]
    assert len(workflow["decision"]["observed_quartile_recommendations"]) == 4
    assert workflow["confirmation_performed"] == 0


def test_later_adjusted_count_includes_later_pooled_components() -> None:
    lower = np.zeros((120, 228), dtype=np.float64)
    lower[0, 55] = 1.0
    assert cli._later_adjusted_positive_count(lower) == 0
    lower[0, 56] = 1.0
    lower[1, 225] = 1.0
    lower[2, 227] = 1.0
    assert cli._later_adjusted_positive_count(lower) == 3


def test_sealed_validation_risk_comparator_rejects_perturbation() -> None:
    paths = np.arange(8, dtype=np.int64)
    sealed = np.zeros((8, 3, 228), dtype=np.float64)
    passing = cli._compare_sealed_validation_risks(
        current_path_ids=paths,
        current_values=sealed,
        sealed_path_ids=paths,
        sealed_values=sealed,
    )
    assert passing["passed"] == 1
    changed = sealed.copy()
    changed[0, 0, 0] = np.nextafter(cli.IDENTITY_TOLERANCE, np.inf)
    failed = cli._compare_sealed_validation_risks(
        current_path_ids=paths,
        current_values=changed,
        sealed_path_ids=paths,
        sealed_values=sealed,
    )
    assert failed["passed"] == 0


def test_typed_replay_failure_commits_closed_named_gate(tmp_path: Path) -> None:
    failure = {
        "failure_domain": "sealed_evidence",
        "failure_code": "coarse_witness_replay_invalid",
        "error": "fixture",
    }
    cli.atomic_write_json(
        tmp_path / "replay_execution_failure.json",
        {"evaluation_status": "execution_failed", **failure},
    )
    cli._commit_failed_stage_gate(tmp_path, stage="replay", failure=failure)
    gate = cli._load_json(tmp_path / "time_local_replay_gate.json")
    assert gate["evaluation_status"] == "execution_failed"
    assert gate["failure_code"] == "coarse_witness_replay_invalid"
    decision = cli._workflow_record(tmp_path, "none")["decision"]
    assert decision["decision"] == "control_provenance_invalid"
    cli.atomic_write_json(
        tmp_path / "preflight_gate.json",
        {"evaluation_status": "evaluated", "passed": 1},
    )
    decision = cli._workflow_record(tmp_path, "none")["decision"]
    assert decision["decision"] == "coarse_witness_replay_invalid"
    assert not cli._completed_stage(
        tmp_path,
        gate_name="time_local_replay_gate.json",
        seal_name="replay_artifact_seal.json",
    )
    assert (tmp_path / "execution_attempts/replay/attempt-001/retry_authorization.json").is_file()
    assert not (tmp_path / "time_local_replay_gate.json").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_nominee_cpu_cuda_streaming_agreement() -> None:
    memory_parent = Path(
        "runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_memory_confirmation/"
        "20260806-181326_production-zero-baseline-v3-memory-safe"
    ).resolve()
    if not memory_parent.is_dir():
        pytest.skip("immutable production parent is unavailable")
    binding = cli._load_json(memory_parent / "immutable_cache_binding.json")
    cache_root = Path(binding["parent_run_dir"])
    inputs = cli.open_external_input_store(cache_root, "validation")
    grid = cli._candidate_grid(memory_parent)
    cpu_model, _ = cli._load_nominee_model(
        memory_parent, seed=261312, update=900, grid=grid, device="cpu"
    )
    cuda_model, _ = cli._load_nominee_model(
        memory_parent, seed=261312, update=900, grid=grid, device="cuda"
    )
    rows = np.arange(32, dtype=np.int64)
    cpu_guard = cli.ModelCallBatchGuard(maximum_batch_size=32)
    cuda_guard = cli.ModelCallBatchGuard(maximum_batch_size=32)
    with torch.no_grad():
        cpu = cpu_guard.call(cpu_model, inputs.batch(rows, device="cpu")).double()
        torch.cuda.reset_peak_memory_stats()
        cuda = cuda_guard.call(cuda_model, inputs.batch(rows, device="cuda")).double()
    cuda_cpu = cuda.cpu()
    assert tuple(cpu.shape) == (32, 392)
    assert torch.isfinite(cpu).all() and torch.isfinite(cuda_cpu).all()
    output_error = torch.abs(cpu - cuda_cpu)
    assert float(torch.max(output_error)) <= 4e-5
    assert float(torch.mean(output_error)) <= 3e-6
    target = torch.linspace(-0.1, 0.1, cpu.numel(), dtype=torch.float64).reshape_as(cpu)
    cpu_c = torch.mean(target * cpu)
    cuda_c = torch.mean(target * cuda_cpu)
    cpu_p = torch.mean(cpu * cpu)
    cuda_p = torch.mean(cuda_cpu * cuda_cpu)
    cpu_i = 2.0 * cpu_c - cpu_p
    cuda_i = 2.0 * cuda_c - cuda_p
    assert abs(float(cpu_c - cuda_c)) <= 3e-8
    assert abs(float(cpu_p - cuda_p)) <= 3e-8
    assert abs(float(cpu_i - cuda_i)) <= 3e-8
    assert cpu_guard.maximum_observed_batch_size == 32
    assert cuda_guard.maximum_observed_batch_size == 32
    peak = torch.cuda.max_memory_allocated()
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    assert peak / total <= cli.MAXIMUM_PEAK_MEMORY_FRACTION


def test_decomposition_csv_rows_cover_original_and_descriptive_resolutions() -> None:
    values = np.ones((2, 1, 228), dtype=np.float64)
    decomposition = QuadraticRiskDecomposition(
        path_ids=np.asarray([1, 2], dtype=np.int64),
        candidate_labels=("candidate",),
        cross_terms=values,
        prediction_energies=values,
        direct_improvements=values,
        reconstructed_improvements=values,
        maximum_identity_error=0.0,
    )
    path_rows, summary_rows = cli._decomposition_rows("validation", decomposition)
    assert len(path_rows) == 2 * 228
    assert {row["resolution"] for row in summary_rows} == {
        "original_228",
        "quartile_phase",
        "quartile_midpoint",
        "phase",
        "midpoint",
        "overall",
    }
    assert sum(row["authorizing"] for row in summary_rows) == 228


def test_completed_stage_verifies_seal_and_skips_recomputation(tmp_path: Path) -> None:
    cli.atomic_write_json(tmp_path / "metrics.json", {"value": 1})
    cli.atomic_write_json(
        tmp_path / "gate.json", {"evaluation_status": "evaluated", "passed": 1}
    )
    cli._seal_stage(tmp_path, ("metrics.json", "gate.json"), "seal.json")
    assert cli._completed_stage(
        tmp_path, gate_name="gate.json", seal_name="seal.json"
    )
    cli.atomic_write_json(tmp_path / "metrics.json", {"value": 2})
    with pytest.raises(ArtifactCompatibilityError):
        cli._completed_stage(tmp_path, gate_name="gate.json", seal_name="seal.json")
