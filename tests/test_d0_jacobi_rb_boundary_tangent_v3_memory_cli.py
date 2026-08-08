from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from mnist import diag_d0_jacobi_rb_boundary_tangent_v3_memory_confirmation as cli
from mnist import d0_jacobi_rb_boundary_tangent_v3_memory as memory
from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_json,
)
from mnist.d0_jacobi_rb_learnability import EDGES_PER_PHASE, ModelInputs


def _input_arrays(rows: int) -> dict[str, np.ndarray]:
    return {
        "sample_key": np.arange(rows, dtype=np.int64),
        "path_id": np.full(rows, 17, dtype=np.int64),
        "outer_step": np.arange(rows, dtype=np.int16),
        "midpoint_index": np.zeros(rows, dtype=np.int8),
        "later_full_state": np.full(
            (rows, 784), 1.0 / 784.0, dtype=np.float32
        ),
        "reverse_time": np.linspace(0.1, 0.9, rows, dtype=np.float64),
        "phase": np.zeros(rows, dtype=np.int8),
        "color": np.zeros(rows, dtype=np.int8),
        "duration": np.full(rows, 0.5, dtype=np.float64),
        "label": np.full(rows, 3, dtype=np.int64),
    }


def _input_store(root: Path, role: str, rows: int = 4) -> memory.HostInputStore:
    return memory.HostInputStore.from_arrays(
        _input_arrays(rows), role=role, cache_root=root
    )


def _label_store(
    root: Path,
    role: str,
    purpose: str,
    digest: str,
    rows: int = 4,
) -> memory.HostLabelStore:
    authorization = memory.LabelOpenAuthorization(root, role, purpose, digest)
    arrays = {
        "sample_key": np.arange(rows, dtype=np.int64),
        "path_id": np.full(rows, 17, dtype=np.int64),
        "denoising_target": np.zeros(
            (rows, EDGES_PER_PHASE), dtype=np.float64
        ),
    }
    return memory.HostLabelStore.from_arrays(
        arrays, authorization=authorization
    )


def test_parser_and_stage_sequence_expose_only_continuation_stages(
    tmp_path: Path,
) -> None:
    args = cli.parse_args(
        [
            "--failed-v3-train-run-dir",
            str(tmp_path / "parent"),
            "--resume-run-dir",
            str(tmp_path / "child"),
            "--stage",
            "report",
        ]
    )
    assert args.failed_v3_train_run_dir.is_absolute()
    assert args.resume_run_dir.is_absolute()
    assert cli.STAGES == (
        "preflight",
        "train",
        "select",
        "confirm",
        "report",
        "all",
    )
    assert cli._stage_sequence("all") == (
        "preflight",
        "train",
        "select",
        "confirm",
    )
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--failed-v3-train-run-dir",
                str(tmp_path / "parent"),
                "--stage",
                "train",
            ]
        )


def test_training_label_open_is_semantically_verified_before_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    parent = tmp_path / "parent"
    run_dir.mkdir()
    parent.mkdir()
    inputs = _input_store(parent, "train")
    opening = cli._semantic(
        {
            "schema": cli.RUN_SCHEMA + "-training-label-open",
            "schema_version": 1,
            "role": "train",
            "controls_passed": 1,
            "validation_labels_opened": 0,
            "confirmation_labels_opened": 0,
            **cli.NO_WORK,
        }
    )
    atomic_write_json(run_dir / "training_label_open.json", opening)
    calls = 0

    def open_labels(root: Path, role: str, *, authorization):
        nonlocal calls
        calls += 1
        assert root == parent.resolve()
        assert role == "train"
        assert authorization.opening_seal_sha256 == opening["semantic_sha256"]
        return _label_store(
            parent,
            "train",
            "physical_training",
            opening["semantic_sha256"],
        )

    monkeypatch.setattr(memory, "open_external_label_store", open_labels)
    labels = cli._training_label_store(run_dir, inputs)
    assert labels.row_count == inputs.row_count
    assert calls == 1

    damaged = dict(opening)
    damaged["controls_passed"] = 0
    atomic_write_json(run_dir / "training_label_open.json", damaged)
    with pytest.raises(ArtifactCompatibilityError):
        cli._training_label_store(run_dir, inputs)
    assert calls == 1


def test_validation_adapter_checks_opening_and_restores_parent_hooks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    parent = tmp_path / "parent"
    run_dir.mkdir()
    parent.mkdir()
    search = cli._semantic({"schema": "fixture-search", "schema_version": 1})
    grid = cli._semantic({"schema": "fixture-grid", "schema_version": 1})
    atomic_write_json(run_dir / "validation_search_plan.json", search)
    atomic_write_json(run_dir / "candidate_grid.json", grid)
    opening = cli._semantic(
        {
            "schema": cli._v3.RUN_SCHEMA + "-validation-label-open",
            "schema_version": 1,
            "search_plan_sha256": search["semantic_sha256"],
            "candidate_grid_sha256": grid["semantic_sha256"],
            "count_shards_committed": 1,
            "confirmation_namespace_opened": 0,
            **cli.NO_WORK,
        }
    )
    atomic_write_json(run_dir / "validation_label_open.json", opening)
    inputs = _input_store(parent, "validation")
    labels = _label_store(
        parent,
        "validation",
        "validation_selection",
        opening["semantic_sha256"],
    )
    monkeypatch.setattr(cli, "_external_root", lambda _run: parent.resolve())
    monkeypatch.setattr(
        memory, "open_external_input_store", lambda root, role: inputs
    )
    label_calls = 0

    def open_labels(root: Path, role: str, *, authorization):
        nonlocal label_calls
        label_calls += 1
        assert authorization.opening_seal_sha256 == opening["semantic_sha256"]
        return labels

    monkeypatch.setattr(memory, "open_external_label_store", open_labels)
    old_loader = cli._v3._load_validation_evidence
    old_predict = cli._v3._predict_in_batches
    args = argparse.Namespace(device="cpu")
    guard = memory.ModelCallBatchGuard()
    with cli._v3_memory_adapter(run_dir, args, guard, stage="select"):
        arrays, adapted, target, _ = cli._v3._load_validation_evidence(
            run_dir, args
        )
        assert adapted is inputs
        assert target.shape == (4, EDGES_PER_PHASE)
        assert arrays["sample_key"].tolist() == [0, 1, 2, 3]
    assert cli._v3._load_validation_evidence is old_loader
    assert cli._v3._predict_in_batches is old_predict
    assert label_calls == 1

    damaged = dict(opening)
    damaged["confirmation_namespace_opened"] = 1
    atomic_write_json(run_dir / "validation_label_open.json", damaged)
    with cli._v3_memory_adapter(run_dir, args, guard, stage="select"):
        with pytest.raises(ArtifactCompatibilityError):
            cli._v3._load_validation_evidence(run_dir, args)
    assert label_calls == 1


@pytest.mark.parametrize("stage", ["select", "confirm"])
def test_legacy_stage_snapshot_recovers_interrupted_child_finalization(
    tmp_path: Path, stage: str
) -> None:
    run_dir = tmp_path / stage
    run_dir.mkdir()
    prefix = "selection" if stage == "select" else "confirmation"
    metrics_name = "select_metrics.json" if stage == "select" else "confirmation_metrics.json"
    gate_name = "select_gate.json" if stage == "select" else "confirm_gate.json"
    seal_name = "selection_artifact_seal.json" if stage == "select" else "confirm_artifact_seal.json"
    immutable_name = f"{prefix}_immutable.json"
    original_metrics = {"schema": "legacy-metrics", "value": 17}
    original_gate = {"schema": "legacy-gate", "passed": 1}
    atomic_write_json(run_dir / metrics_name, original_metrics)
    atomic_write_json(run_dir / gate_name, original_gate)
    atomic_write_json(run_dir / immutable_name, {"sealed": 1})
    if stage == "confirm":
        atomic_write_json(run_dir / "confirmation_gate.json", original_gate)
        names = (
            metrics_name,
            gate_name,
            "confirmation_gate.json",
            immutable_name,
        )
    else:
        names = (metrics_name, gate_name, immutable_name)
    cli._seal_stage(run_dir, names, seal_name)

    metrics, gate = cli._prepare_legacy_stage_snapshot(
        run_dir,
        stage=stage,
        seal_name=seal_name,
        metrics_name=metrics_name,
        gate_name=gate_name,
    )
    assert metrics == original_metrics
    assert gate == original_gate

    # Simulate interruption after the child rewrote mutable JSON but before it
    # replaced the legacy seal. Resume must use the frozen legacy copies and
    # must not rerun the scientific panel.
    atomic_write_json(run_dir / metrics_name, {"schema": "child", "value": 99})
    atomic_write_json(run_dir / gate_name, {"schema": "child", "passed": 0})
    restored_metrics, restored_gate = cli._prepare_legacy_stage_snapshot(
        run_dir,
        stage=stage,
        seal_name=seal_name,
        metrics_name=metrics_name,
        gate_name=gate_name,
    )
    assert restored_metrics == original_metrics
    assert restored_gate == original_gate

    atomic_write_json(run_dir / immutable_name, {"sealed": 0})
    with pytest.raises(ArtifactCompatibilityError):
        cli._prepare_legacy_stage_snapshot(
            run_dir,
            stage=stage,
            seal_name=seal_name,
            metrics_name=metrics_name,
            gate_name=gate_name,
        )


def test_finalized_memory_seal_requires_diagnostic_and_snapshot_membership(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "seal"
    run_dir.mkdir()
    for name in (
        "select_gate.json",
        "selection_memory_diagnostics.json",
        "selection_legacy_stage_snapshot.json",
    ):
        atomic_write_json(run_dir / name, {"name": name})
    cli._seal_stage(
        run_dir,
        (
            "select_gate.json",
            "selection_memory_diagnostics.json",
            "selection_legacy_stage_snapshot.json",
        ),
        "selection_artifact_seal.json",
    )
    paths = cli._stage_seal_paths(run_dir, "selection_artifact_seal.json")
    assert "selection_memory_diagnostics.json" in paths
    assert "selection_legacy_stage_snapshot.json" in paths


def test_adapter_progress_survives_orphan_finalization_and_merges_resume_calls(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "progress"
    run_dir.mkdir()
    first = memory.ModelCallBatchGuard()
    first.call_count = 2
    first.maximum_observed_batch_size = 32
    first.observed_batch_sizes = [32, 7]
    committed = cli._commit_adapter_progress(
        run_dir,
        stage="select",
        guard=first,
        base=None,
        device=cli.torch.device("cpu"),
    )
    assert committed["model_call_batches"]["call_count"] == 2
    assert committed["model_call_batches"]["maximum_observed_batch_size"] == 32

    second = memory.ModelCallBatchGuard()
    second.call_count = 1
    second.maximum_observed_batch_size = 4
    second.observed_batch_sizes = [4]
    merged = cli._commit_adapter_progress(
        run_dir,
        stage="select",
        guard=second,
        base=cli._load_adapter_progress(run_dir, "select"),
        device=cli.torch.device("cpu"),
    )
    assert merged["model_call_batches"]["call_count"] == 3
    assert merged["model_call_batches"]["maximum_observed_batch_size"] == 32
    assert cli._load_adapter_progress(run_dir, "select") == merged

    damaged = dict(merged)
    damaged["peak_memory_bytes"] = 1
    atomic_write_json(run_dir / "selection_memory_progress.json", damaged)
    with pytest.raises(ArtifactCompatibilityError):
        cli._load_adapter_progress(run_dir, "select")


def test_confirmation_adapter_keeps_560_midpoint_rows_host_backed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    parent = tmp_path / "parent"
    run_dir.mkdir()
    parent.mkdir()
    monkeypatch.setattr(cli, "_external_root", lambda _run: parent.resolve())
    args = argparse.Namespace(device="cpu")
    guard = memory.ModelCallBatchGuard()
    original_builder = cli._v3._legacy._model_inputs_from_arrays

    class ZeroModel(nn.Module):
        def forward(self, inputs: ModelInputs) -> torch.Tensor:
            return torch.zeros(
                (inputs.batch_size, EDGES_PER_PHASE),
                dtype=torch.float64,
                device=inputs.later_full_state.device,
            )

    arrays = _input_arrays(560)
    with cli._v3_memory_adapter(run_dir, args, guard, stage="confirm"):
        host = cli._v3._legacy._model_inputs_from_arrays(
            arrays, torch.device("cpu")
        )
        assert isinstance(host, memory.HostInputStore)
        assert host.row_count == 560
        prediction = cli._v3._predict_in_batches(ZeroModel(), host)
        assert prediction.shape == (560, EDGES_PER_PHASE)
        assert prediction.device.type == "cpu"
        assert guard.maximum_observed_batch_size == 32
        assert guard.observed_batch_sizes[-1] == 16
    assert cli._v3._legacy._model_inputs_from_arrays is original_builder
    progress = cli._load_adapter_progress(run_dir, "confirm")
    assert progress is not None
    assert progress["model_call_batches"]["maximum_observed_batch_size"] == 32


@pytest.mark.parametrize("stage", ["train", "select", "confirm"])
def test_execution_failure_is_archived_and_reopened_without_progress_loss(
    tmp_path: Path, stage: str
) -> None:
    run_dir = tmp_path / stage
    run_dir.mkdir()
    metrics_name = {
        "train": "train_metrics.json",
        "select": "select_metrics.json",
        "confirm": "confirmation_metrics.json",
    }[stage]
    seal_name = {
        "train": "train_artifact_seal.json",
        "select": "selection_artifact_seal.json",
        "confirm": "confirm_artifact_seal.json",
    }[stage]
    failure_name = f"{stage}_execution_failure.json"
    gate_name = f"{stage}_gate.json"
    metrics = {
        "schema": "fixture-execution-failure",
        "evaluation_status": "execution_failed",
        "failure_domain": "training_memory_schedule",
        "failure_code": "fixture_failure",
    }
    gate = {
        "schema": "fixture-execution-gate",
        "evaluation_status": "execution_failed",
        "passed": 0,
        "failure_code": "fixture_failure",
    }
    atomic_write_json(run_dir / metrics_name, metrics)
    atomic_write_json(run_dir / failure_name, metrics)
    atomic_write_json(run_dir / gate_name, gate)
    names = [metrics_name, failure_name, gate_name]
    if stage == "confirm":
        atomic_write_json(run_dir / "confirmation_gate.json", gate)
        names.append("confirmation_gate.json")
    cli._seal_stage(run_dir, names, seal_name)
    progress = run_dir / "checkpoints" / "progress.pt"
    progress.parent.mkdir()
    progress.write_bytes(b"exact-progress")
    cli._artifact_registry(run_dir)

    assert cli._prepare_execution_retry(run_dir, stage)
    assert not (run_dir / gate_name).exists()
    assert not (run_dir / seal_name).exists()
    assert not (run_dir / failure_name).exists()
    assert not (run_dir / metrics_name).exists()
    assert progress.read_bytes() == b"exact-progress"
    assert not (run_dir / "artifact_registry.json").exists()
    archive = run_dir / "execution_attempts" / stage / "attempt-001"
    assert (archive / "retry_authorization.json").is_file()
    assert (archive / gate_name).is_file()
    assert (archive / "artifact_registry.json").is_file()
    assert not cli._prepare_execution_retry(run_dir, stage)

    # A scientific evaluated failure remains terminal and is never archived.
    evaluated = {**gate, "evaluation_status": "evaluated"}
    atomic_write_json(run_dir / gate_name, evaluated)
    assert not cli._prepare_execution_retry(run_dir, stage)
    assert (run_dir / gate_name).is_file()


def test_fresh_initialization_failure_commits_readable_provenance_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_parent(_path: Path):
        raise ArtifactCompatibilityError("fixture parent registry changed")

    monkeypatch.setattr(
        cli._provenance, "verify_failed_v3_train_parent", fail_parent
    )
    args = argparse.Namespace(
        stage="preflight",
        require_gate="preflight",
        failed_v3_train_run_dir=(tmp_path / "parent").resolve(),
        resume_run_dir=None,
        runs_root=(tmp_path / "runs").resolve(),
        run_name="fresh-provenance-failure",
        device="cpu",
        test_only=True,
        test_maximum_updates=0,
        test_bootstrap_replicates=8,
        test_outer_steps=16,
    )
    assert cli._run(args) == 2
    run_dir = next(args.runs_root.iterdir())
    status = cli._load_json(run_dir / "run_status.json")
    gate = cli._load_json(run_dir / "preflight_gate.json")
    decision = cli._load_json(
        run_dir / "boundary_tangent_v3_memory_decision.json"
    )
    assert status["state"] == "execution_failed"
    assert gate["failure_domain"] == "control_provenance"
    assert decision["decision"] == "control_provenance_invalid"
    assert (run_dir / "initialization_failure.json").is_file()
    assert (run_dir / "preflight_artifact_seal.json").is_file()
    assert (run_dir / "artifact_registry.json").is_file()


def test_incompatible_resume_initialization_failure_does_not_mutate_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "resume"
    run_dir.mkdir()
    marker = run_dir / "marker.bin"
    marker.write_bytes(b"immutable")

    def fail_initialize(*_args, **_kwargs):
        raise ArtifactCompatibilityError("resume manifest changed")

    monkeypatch.setattr(cli, "_initialize", fail_initialize)
    args = argparse.Namespace(
        stage="report",
        require_gate="none",
        failed_v3_train_run_dir=(tmp_path / "parent").resolve(),
        resume_run_dir=run_dir.resolve(),
        runs_root=(tmp_path / "runs").resolve(),
        run_name="unused",
        device="cpu",
        test_only=True,
        test_maximum_updates=0,
        test_bootstrap_replicates=8,
        test_outer_steps=16,
    )
    before = sorted(path.relative_to(run_dir) for path in run_dir.rglob("*"))
    assert cli._run(args) == 2
    after = sorted(path.relative_to(run_dir) for path in run_dir.rglob("*"))
    assert before == after == [Path("marker.bin")]
    assert marker.read_bytes() == b"immutable"


def test_fresh_cache_binding_initialization_failure_keeps_binding_domain(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "binding-failure"
    run_dir.mkdir()
    error = cli.MemoryConfirmationError(
        "cache seal changed",
        failure_domain="immutable_cache_binding",
        failure_code="immutable_parent_cache_binding_invalid",
    )
    cli._commit_initialization_failure(
        run_dir, error=error, require_gate="preflight"
    )
    gate = cli._load_json(run_dir / "preflight_gate.json")
    decision = cli._load_json(
        run_dir / "boundary_tangent_v3_memory_decision.json"
    )
    assert gate["failure_domain"] == "immutable_cache_binding"
    assert gate["immutable_cache_binding_valid"] == 0
    assert decision["decision"] == "immutable_cache_binding_invalid"
