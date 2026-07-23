from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pytest
import numpy as np
import torch

import mnist.diag_d0_score_density_ratio_stability_confirmation as cli
from mnist.d0_one_image_gate import ArtifactCompatibilityError
from mnist.d0_score_density_ratio import (
    build_density_ratio_panel,
    build_density_ratio_stream_plan,
)
from mnist.d0_score_density_ratio_paired import (
    build_paired_mixture_stream_plan,
    generate_accumulated_paired_stream,
)
from mnist.d0_score_density_ratio_stability_gate import RatioStabilityThresholds
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, natural_horizon


def test_production_cli_defaults_are_frozen() -> None:
    args = cli.parse_args(
        [
            "--parent-density-ratio-run-dir", "parent",
            "--stage", "preflight",
            "--require-gate", "preflight",
        ]
    )
    assert args.root_seed == 260851
    assert args.pilot_learning_rates == (3e-5, 1e-5)
    assert args.accumulation_levels == (2, 4, 8)
    assert args.loss_scale == pytest.approx(0.05173607018770852, abs=0.0)
    assert args.confirm_model_seeds == (260861, 260862, 260863)
    assert args.pilot_steps == 2_000
    assert args.confirm_steps == 4_000
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--parent-density-ratio-run-dir", "parent",
                "--require-gate", "preflight",
                "--base-channels", "4",
            ]
        )
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--parent-density-ratio-run-dir", "parent",
                "--loss-scale", "0.1",
            ]
        )


def test_cli_has_no_sampler_import() -> None:
    tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("sampler" in value.lower() for value in imported)
    _, source_paths = cli._source_record()
    names = {Path(value).name for value in source_paths}
    assert "d0_score_density_ratio_paired.py" in names
    assert "d0_score_density_ratio_stability.py" not in names


def _dynamics() -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=4,
        num_steps=8,
        alpha_eff=1.0,
        edge_alpha_mode="alpha_eff",
        mass_floor=1e-7,
        limiter_fraction=1.0,
        source_lowfreq_size=4,
        ot_lowres_size=4,
    )


def _task_fixture():
    dynamics = _dynamics()
    horizon = float(natural_horizon(dynamics))
    evaluation = build_density_ratio_stream_plan(
        root_seed=260851, grid_size=4, horizon=horizon
    )
    paired = build_paired_mixture_stream_plan(
        root_seed=260851, grid_size=4, horizon=horizon
    )
    panels = {
        role: build_density_ratio_panel(
            evaluation,
            phase="paired-unit",
            role=role,
            task="dirichlet_null",
            path_count=1,
            start_step=index * 10,
        )
        for index, role in enumerate(("a", "b"))
    }
    args = argparse.Namespace(
        base_channels=4,
        weight_decay=0.0,
        validation_steps=(0, 1, 2),
        train_steps=2,
        root_seed=260851,
        grad_clip=1.0,
        ema_decay=0.99,
        bootstrap_reps=16,
        bootstrap_confidence=0.90,
        validation_batch_size=64,
        clip_warmup_steps=0,
    )
    return dynamics, evaluation, paired, panels, args


def _fake_panel_record_factory(calls: dict[str, int]):
    def fake(model, panel, **kwargs):
        del model, kwargs
        calls[panel.role] += 1
        index = calls[panel.role]
        bce = (0.69, 0.55, 0.60)[index - 1] if panel.role == "a" else 0.70
        lower = -0.01
        scope = {
            "bce": bce,
            "model_bce": bce,
            "classification_risk": bce,
            "zero_logit_bce": 0.6931471805599453,
            "improvement": 0.6931471805599453 - bce,
            "objective_improvement": 0.6931471805599453 - bce,
            "lower_bound": lower,
            "upper_bound": lower + 0.01,
            "confidence": 0.90,
            "bootstrap": {"path_ids": [1], "path_values": [lower]},
        }
        return (
            {
                "evaluation_status": "evaluated",
                "finite": 1,
                "role": panel.role,
                "task": panel.task,
                "panel_fingerprint": panel.fingerprint,
                "path_count": 1,
                "anchors_per_path": 32,
                "confidence": 0.90,
                "overall": dict(scope),
                "data_end": dict(scope),
                "physical_training_performed": 0,
                "sampling_performed": 0,
            },
            [],
        )
    return fake


def _run_task(
    path: Path, *, interrupt=None, interrupt_checkpoint=None, interrupt_sealed=False
):
    dynamics, evaluation, paired, panels, args = _task_fixture()
    return cli.run_paired_density_ratio_task(
        task_dir=path,
        task="dirichlet_null",
        selection_panels=panels,
        audit_panels=None,
        dynamics=dynamics,
        args=args,
        device=torch.device("cpu"),
        model_seed=17,
        learning_rate=1e-5,
        accumulation_level=2,
        stream_plan=evaluation,
        paired_stream_plan=paired,
        fingerprints={"fixture": "paired-task"},
        phase="paired-unit",
        thresholds=RatioStabilityThresholds(),
        show_progress=False,
        interrupt_during_accumulation=interrupt,
        interrupt_after_checkpoint_step=interrupt_checkpoint,
        interrupt_after_sealed_panel_b=interrupt_sealed,
    )


def _assert_nested_exact(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_exact(a, b)
    else:
        assert left == right


def test_task_panel_b_one_shot_and_completed_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"a": 0, "b": 0}
    monkeypatch.setattr(cli.base, "_classification_panel_record", _fake_panel_record_factory(calls))
    result = _run_task(tmp_path / "task")
    assert calls == {"a": 3, "b": 1}
    assert result["metrics"]["nominee_step"] == 1
    reused = _run_task(tmp_path / "task")
    assert reused == result
    assert calls == {"a": 3, "b": 1}

    task_dir = tmp_path / "task"
    checkpoints = task_dir / "checkpoints"
    original_final = checkpoints / "step-00000002.pt"
    cli.atomic_write_json(
        checkpoints / "latest.json",
        {
            "schema": cli.RUN_SCHEMA + "-latest",
            "schema_version": 1,
            "filename": original_final.name,
            "sha256": cli.file_fingerprint(original_final),
            "step": 2,
            "stream_cursor": 2,
            "accumulation_cursor": 0,
            "fingerprints": {"fixture": "paired-task"},
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    (task_dir / "task_result.json").unlink()
    (task_dir / "task_status.json").unlink()
    recovered = _run_task(task_dir)
    assert recovered == result
    assert calls == {"a": 3, "b": 1}
    assert cli._json_load(checkpoints / "latest.json")["filename"] == (
        "finalized-step-00000002.pt"
    )

    with (task_dir / "sealed_panel_b.json").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ArtifactCompatibilityError):
        _run_task(task_dir)


def test_interruption_during_accumulation_restarts_uncommitted_step_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli.base,
        "_classification_panel_record",
        _fake_panel_record_factory({"a": 0, "b": 0}),
    )
    uninterrupted = _run_task(tmp_path / "full")
    monkeypatch.setattr(
        cli.base,
        "_classification_panel_record",
        _fake_panel_record_factory({"a": 0, "b": 0}),
    )
    with pytest.raises(RuntimeError, match="during accumulation"):
        _run_task(tmp_path / "resumed", interrupt=(1, 0))
    resumed = _run_task(tmp_path / "resumed")
    assert resumed["metrics"] == uninterrupted["metrics"]
    full_payload = torch.load(
        tmp_path / "full" / "checkpoints" / "finalized-step-00000002.pt",
        map_location="cpu", weights_only=False,
    )
    resumed_payload = torch.load(
        tmp_path / "resumed" / "checkpoints" / "finalized-step-00000002.pt",
        map_location="cpu", weights_only=False,
    )
    _assert_nested_exact(full_payload, resumed_payload)


def test_terminal_registry_detects_tampering(tmp_path: Path) -> None:
    cli.atomic_write_json(tmp_path / "evidence.json", {"passed": 1})
    cli.atomic_write_json(tmp_path / "run_status.json", {"status": "complete"})
    registry = cli._artifact_registry(tmp_path)
    cli.atomic_write_json(tmp_path / "artifact_registry.json", registry)
    registry_path = tmp_path / "artifact_registry.json"
    cli.atomic_write_json(
        tmp_path / "run_status.json",
        {
            "status": "complete",
            "artifact_registry_sha256": cli.file_fingerprint(registry_path),
            "artifact_registry_size": registry_path.stat().st_size,
        },
    )
    assert cli._verify_terminal_registry(tmp_path)["records"]
    cli.atomic_write_json(tmp_path / "evidence.json", {"passed": 0})
    with pytest.raises(ArtifactCompatibilityError):
        cli._verify_terminal_registry(tmp_path)


def test_hierarchical_training_levels_share_exact_nested_stream_prefixes() -> None:
    dynamics = _dynamics()
    plan = build_paired_mixture_stream_plan(
        root_seed=260851,
        grid_size=4,
        horizon=float(natural_horizon(dynamics)),
    )
    streams = {
        level: generate_accumulated_paired_stream(
            plan,
            phase="pilot",
            task="bounded_teacher",
            optimizer_step=83,
            accumulation_level=level,
        )
        for level in (2, 4, 8)
    }
    fingerprints = {
        level: [value.fingerprint for value in stream.canonical_microbatches]
        for level, stream in streams.items()
    }
    assert fingerprints[4][:2] == fingerprints[2]
    assert fingerprints[8][:4] == fingerprints[4]
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert 'training_phase = "pilot"' in source


def test_task_fails_closed_on_finite_loss_with_nonfinite_gradient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli.base,
        "_classification_panel_record",
        _fake_panel_record_factory({"a": 0, "b": 0}),
    )

    def finite_forward_nan_backward(model, batch, *, task):
        del batch, task
        parameter = next(model.parameters()).reshape(-1)[0]
        loss = parameter * 0.0 + 0.5
        loss.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        return loss, {
            "base_positive": 0.0,
            "mixture_positive": 0.0,
            "reference_negative": 0.5,
            "total": 0.5,
        }

    monkeypatch.setattr(
        cli, "_microbatch_objective_and_components", finite_forward_nan_backward
    )
    with pytest.raises(FloatingPointError, match="nonfinite accumulated gradient"):
        _run_task(tmp_path / "nonfinite")


def test_sealed_panel_b_survives_finalization_crash_without_reevaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"a": 0, "b": 0}
    monkeypatch.setattr(
        cli.base, "_classification_panel_record", _fake_panel_record_factory(calls)
    )
    task_dir = tmp_path / "sealed"
    with pytest.raises(RuntimeError, match="after sealed panel B"):
        _run_task(task_dir, interrupt_sealed=True)
    assert calls == {"a": 3, "b": 1}
    sealed = cli._json_load(task_dir / "sealed_panel_b.json")
    assert sealed["nominee_step"] == 1
    result = _run_task(task_dir)
    assert result["metrics"]["nominee_step"] == 1
    assert calls == {"a": 3, "b": 1}


def test_forensics_reuses_common_panels_across_parent_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dynamics, evaluation, paired, _, _ = _task_fixture()

    class ZeroModel(torch.nn.Module):
        def forward(self, tau, states, labels):
            del tau, labels
            return states.sum(dim=1, keepdim=True) * 0.0

    def fake_parent(*args, task, candidate_index, **kwargs):
        del args, kwargs
        return ZeroModel(), {
            "task": task,
            "candidate_index": candidate_index,
            "checkpoint_path": f"fixture-{candidate_index}-{task}",
            "checkpoint_sha256": "fixture",
            "nominee_step": 1,
        }

    monkeypatch.setattr(cli, "_load_parent_forensic_model", fake_parent)
    args = argparse.Namespace(
        parent_density_ratio_run_dir=tmp_path / "parent",
        base_channels=4,
        preflight_paths=2,
        preflight_confidence=0.99,
        bootstrap_reps=16,
        root_seed=260851,
    )
    record = cli._run_variance_forensics(
        tmp_path,
        args=args,
        dynamics=dynamics,
        device=torch.device("cpu"),
        stream_plan=evaluation,
        paired_stream_plan=paired,
    )
    assert record["common_forensic_panels_across_checkpoints_pass"] == 1
    assert record["stream_fingerprint_isolation_pass"] == 1
    assert record["forensic_law_namespaces_disjoint"] == 1


def test_successful_partial_stages_report_interim_next_action() -> None:
    original = {
        "decision": {"decision": "classification_variance_reduction_unresolved"},
        "required_gate_pass": 1,
    }
    preflight = cli._mark_interim_stage_success(original, stage="preflight")
    assert preflight["decision"]["decision"] == "paired_ratio_preflight_passed"
    assert preflight["decision"]["physical_training_authorized"] == 0
    assert preflight["decision"]["closed_terminal_scientific_outcome"] == 0
    pilot = cli._mark_interim_stage_success(original, stage="pilot")
    assert pilot["decision"]["decision"] == "paired_ratio_pilot_passed"
    assert "confirmation" in pilot["decision"]["recommended_next_action"]


def test_interim_preflight_report_finishes_with_explicit_status(tmp_path: Path) -> None:
    report = cli._mark_interim_stage_success(
        {
            "decision": {"decision": "classification_variance_reduction_unresolved"},
            "required_gate": "preflight",
            "required_gate_pass": 1,
        },
        stage="preflight",
    )
    cli._save_report(tmp_path, report)
    assert cli._finish(
        tmp_path,
        report=report,
        stage="preflight",
        phase="preflight",
    ) == 0
    status = cli._json_load(tmp_path / "run_status.json")
    assert status["status"] == "complete"
    assert status["decision"] == "paired_ratio_preflight_passed"
    assert status["physical_training_authorized"] == 0


def test_confirmation_learning_and_optimizer_plot_is_written(tmp_path: Path) -> None:
    history_path = (
        tmp_path / "confirmation" / "seed-1" / "bounded_teacher"
        / "training_history.csv"
    )
    cli.atomic_write_csv(
        history_path,
        [
            {
                "step": 1,
                "scaled_loss": 0.5,
                "scaled_preclip_gradient_norm": 0.8,
                "clipped": 0,
            },
            {
                "step": 2,
                "scaled_loss": 0.4,
                "scaled_preclip_gradient_norm": 1.2,
                "clipped": 1,
            },
        ],
    )
    written = cli._write_plot_artifacts(tmp_path)
    name = "paired_confirmation_learning_optimizer_health.png"
    assert name in written
    assert (tmp_path / name).is_file()
