from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pytest
import torch

import mnist.diag_d0_score_density_ratio_controls as cli
from mnist.d0_one_image_gate import ArtifactCompatibilityError
from mnist.d0_score_density_ratio import (
    build_density_ratio_panel,
    build_density_ratio_stream_plan,
)
from mnist.d0_score_density_ratio_gate import DensityRatioThresholds
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, natural_horizon


def test_production_gate_locks_frozen_defaults() -> None:
    args = cli.parse_args(
        [
            "--parent-stability-run-dir",
            "parent",
            "--stage",
            "preflight",
            "--require-gate",
            "preflight",
        ]
    )
    assert args.root_seed == 260821
    assert args.pilot_learning_rates == (3e-4, 1e-4, 3e-5, 1e-5)
    assert args.pilot_steps == 2_000
    assert args.confirm_steps == 4_000
    assert args.confirm_model_seeds == (260831, 260832, 260833)
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--parent-stability-run-dir",
                "parent",
                "--require-gate",
                "preflight",
                "--base-channels",
                "4",
            ]
        )


def test_cli_has_no_sampler_import() -> None:
    tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("sampler" in name.lower() for name in imported)


def _tiny_task_args() -> argparse.Namespace:
    return argparse.Namespace(
        base_channels=4,
        weight_decay=0.0,
        validation_steps=(0, 1, 2),
        train_steps=2,
        root_seed=97,
        grad_clip=1.0,
        ema_decay=0.99,
        bootstrap_reps=16,
        bootstrap_confidence=0.90,
        validation_batch_size=64,
        clip_warmup_steps=0,
    )


def _assert_nested_exact(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            _assert_nested_exact(left_value, right_value)
    else:
        assert left == right


def test_panel_b_is_evaluated_once_only_on_frozen_a_nominee(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dynamics = DirectFluxMNISTConfig(
        grid_size=4,
        num_steps=8,
        alpha_eff=1.0,
        edge_alpha_mode="alpha_eff",
        mass_floor=1e-7,
        limiter_fraction=1.0,
        source_lowfreq_size=4,
        ot_lowres_size=4,
    )
    plan = build_density_ratio_stream_plan(
        root_seed=97,
        grid_size=4,
        horizon=float(natural_horizon(dynamics)),
    )
    panels = {
        role: build_density_ratio_panel(
            plan,
            phase="unit-selection",
            role=role,
            task="dirichlet_null",
            path_count=1,
            start_step=index * 10,
        )
        for index, role in enumerate(("a", "b"))
    }
    calls = {"a": 0, "b": 0}

    def fake_panel_record(model, panel, **kwargs):
        del model, kwargs
        calls[panel.role] += 1
        index = calls[panel.role]
        if panel.role == "a":
            # Step one is the unique panel-A nominee; panel B must never be
            # consulted while these three checkpoints are scanned.
            bce = (0.69, 0.55, 0.60)[index - 1]
            lower = -0.01
        else:
            bce = 0.70
            lower = -0.02
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

    monkeypatch.setattr(cli, "_classification_panel_record", fake_panel_record)
    task_dir = tmp_path / "task"
    fingerprints = {"unit": "one-shot-b"}
    result = cli.run_density_ratio_task(
        task_dir=task_dir,
        task="dirichlet_null",
        selection_panels=panels,
        audit_panels=None,
        dynamics=dynamics,
        args=_tiny_task_args(),
        device=torch.device("cpu"),
        model_seed=13,
        learning_rate=1e-4,
        loss_scale=0.1,
        stream_plan=plan,
        fingerprints=fingerprints,
        phase="unit",
        thresholds=DensityRatioThresholds(
            pilot_learning_rates=(1e-4,), audit_paths_per_panel=1
        ),
        show_progress=False,
    )
    checkpoints = result["metrics"]["checkpoints"]
    assert calls == {"a": 3, "b": 1}
    assert result["metrics"]["nominee_step"] == 1
    assert [row["step"] for row in checkpoints if "b" in row["panels"]] == [1]

    # A completed-task reuse performs no new panel evaluation, and the two
    # externally named checkpoint copies remain inside the verified hash chain.
    reused = cli.run_density_ratio_task(
        task_dir=task_dir,
        task="dirichlet_null",
        selection_panels=panels,
        audit_panels=None,
        dynamics=dynamics,
        args=_tiny_task_args(),
        device=torch.device("cpu"),
        model_seed=13,
        learning_rate=1e-4,
        loss_scale=0.1,
        stream_plan=plan,
        fingerprints=fingerprints,
        phase="unit",
        thresholds=DensityRatioThresholds(
            pilot_learning_rates=(1e-4,), audit_paths_per_panel=1
        ),
        show_progress=False,
    )
    assert reused == result
    assert calls == {"a": 3, "b": 1}

    # Simulate a crash after the distinct finalized checkpoint was committed
    # but before latest.json was swapped.  Recovery must consume that orphan
    # and must not evaluate sealed panel B a second time.
    checkpoints_dir = task_dir / "checkpoints"
    original_final = checkpoints_dir / "step-00000002.pt"
    cli.atomic_write_json(
        checkpoints_dir / "latest.json",
        {
            "schema": cli.RUN_SCHEMA + "-latest",
            "schema_version": 1,
            "filename": original_final.name,
            "sha256": cli.file_fingerprint(original_final),
            "step": 2,
            "stream_cursor": 2,
            "fingerprints": fingerprints,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    (task_dir / "task_result.json").unlink()
    (task_dir / "task_status.json").unlink()
    recovered = cli.run_density_ratio_task(
        task_dir=task_dir,
        task="dirichlet_null",
        selection_panels=panels,
        audit_panels=None,
        dynamics=dynamics,
        args=_tiny_task_args(),
        device=torch.device("cpu"),
        model_seed=13,
        learning_rate=1e-4,
        loss_scale=0.1,
        stream_plan=plan,
        fingerprints=fingerprints,
        phase="unit",
        thresholds=DensityRatioThresholds(
            pilot_learning_rates=(1e-4,), audit_paths_per_panel=1
        ),
        show_progress=False,
    )
    assert recovered == result
    assert calls == {"a": 3, "b": 1}
    assert cli._json_load(checkpoints_dir / "latest.json")["filename"] == (
        "finalized-step-00000002.pt"
    )

    with (task_dir / "checkpoints" / "nominee_ema.pt").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ArtifactCompatibilityError):
        cli.run_density_ratio_task(
            task_dir=task_dir,
            task="dirichlet_null",
            selection_panels=panels,
            audit_panels=None,
            dynamics=dynamics,
            args=_tiny_task_args(),
            device=torch.device("cpu"),
            model_seed=13,
            learning_rate=1e-4,
            loss_scale=0.1,
            stream_plan=plan,
            fingerprints=fingerprints,
            phase="unit",
            thresholds=DensityRatioThresholds(
                pilot_learning_rates=(1e-4,), audit_paths_per_panel=1
            ),
            show_progress=False,
        )


def test_interrupted_task_resume_matches_uninterrupted_exactly(
    tmp_path: Path,
) -> None:
    dynamics = DirectFluxMNISTConfig(
        grid_size=4,
        num_steps=8,
        alpha_eff=1.0,
        edge_alpha_mode="alpha_eff",
        mass_floor=1e-7,
        limiter_fraction=1.0,
        source_lowfreq_size=4,
        ot_lowres_size=4,
    )
    plan = build_density_ratio_stream_plan(
        root_seed=101,
        grid_size=4,
        horizon=float(natural_horizon(dynamics)),
    )
    panels = {
        role: build_density_ratio_panel(
            plan,
            phase="resume-selection",
            role=role,
            task="dirichlet_null",
            path_count=1,
            start_step=index * 10,
        )
        for index, role in enumerate(("a", "b"))
    }
    common = {
        "task": "dirichlet_null",
        "selection_panels": panels,
        "audit_panels": None,
        "dynamics": dynamics,
        "args": _tiny_task_args(),
        "device": torch.device("cpu"),
        "model_seed": 103,
        "learning_rate": 1e-4,
        "loss_scale": 0.1,
        "stream_plan": plan,
        "fingerprints": {"fixture": "exact-resume"},
        "phase": "unit-resume",
        "thresholds": DensityRatioThresholds(
            pilot_learning_rates=(1e-4,), audit_paths_per_panel=1
        ),
        "show_progress": False,
    }
    interrupted_dir = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="injected interruption"):
        cli.run_density_ratio_task(
            task_dir=interrupted_dir,
            interrupt_after_checkpoint_step=1,
            **common,
        )
    resumed = cli.run_density_ratio_task(task_dir=interrupted_dir, **common)
    uninterrupted_dir = tmp_path / "uninterrupted"
    uninterrupted = cli.run_density_ratio_task(
        task_dir=uninterrupted_dir, **common
    )
    assert resumed["metrics"] == uninterrupted["metrics"]
    for key in (
        "selected_step",
        "nominee_step",
        "training_step",
        "post_warmup_clip_fraction",
        "checkpoint_selection",
        "optimization_diagnostics",
    ):
        assert resumed["training_summary"][key] == uninterrupted[
            "training_summary"
        ][key]

    def latest_payload(task_dir: Path) -> dict[str, Any]:
        pointer = json.loads(
            (task_dir / "checkpoints" / "latest.json").read_text()
        )
        return torch.load(
            task_dir / "checkpoints" / pointer["filename"],
            map_location="cpu",
            weights_only=False,
        )

    resumed_payload = latest_payload(interrupted_dir)
    uninterrupted_payload = latest_payload(uninterrupted_dir)
    for key in (
        "model_state_dict",
        "ema_state_dict",
        "optimizer_state_dict",
        "history",
        "validation_records",
        "checkpoint_selection",
        "stream_cursor",
    ):
        _assert_nested_exact(resumed_payload[key], uninterrupted_payload[key])


def test_required_gate_failure_commits_terminal_evidence(tmp_path: Path) -> None:
    cli.atomic_write_json(tmp_path / "available_evidence.json", {"value": 1})
    failed = cli._not_evaluated_gate("density_ratio_preflight", "fixture failure")
    report = cli._workflow_report(
        provenance={"passed": 1},
        preflight=failed,
        pilot=cli._not_evaluated_gate("density_ratio_pilot", "not run"),
        teacher_results=[],
        null_results=[],
        require_gate="preflight",
        thresholds=DensityRatioThresholds(),
    )
    cli._save_report(tmp_path, report)
    assert cli._finish(
        tmp_path, report=report, stage="preflight", phase="preflight"
    ) == 2
    status = cli._json_load(tmp_path / "run_status.json")
    registry = cli._verify_terminal_registry(tmp_path)
    assert status["outcome"] == "gate_failed"
    assert status["required_gate_pass"] == 0
    assert "available_evidence.json" in registry["records"]
    assert "density_ratio_control_report.json" in registry["records"]


def test_terminal_report_resume_and_source_fingerprint_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dynamics = DirectFluxMNISTConfig(
        grid_size=28,
        num_steps=512,
        alpha_eff=1.0,
        edge_alpha_mode="alpha_eff",
        mass_floor=1e-7,
        limiter_fraction=1.0,
        source_lowfreq_size=4,
        ot_lowres_size=4,
    )
    parent = {
        "passed": 1,
        "horizon": float(natural_horizon(dynamics)),
        "scientific_fingerprint": "parent-science",
        "artifact_registry_sha256": "parent-registry",
        "transitive_parent_provenance": {
            "artifacts": {"operator_preflight": {"sha256": "operator"}}
        },
    }
    monkeypatch.setattr(cli, "verify_parent_stability_run", lambda path: parent)
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda device: {})
    monkeypatch.setattr(cli, "_source_record", lambda: ("source-a", ["cli.py"]))
    monkeypatch.setattr(cli, "_write_plot_artifacts", lambda run_dir: [])
    monkeypatch.setattr(cli, "_write_summary_csvs", lambda run_dir: [])

    def fake_preflight(run_dir: Path, **kwargs):
        del kwargs
        gate = {
            "gate": "density_ratio_preflight",
            "evaluation_status": "evaluated",
            "passed": 1,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        calibration = {
            "loss_scale": 0.1,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        cli.atomic_write_json(run_dir / "density_ratio_preflight_gate.json", gate)
        cli.atomic_write_json(
            run_dir / "density_ratio_loss_scale_calibration.json", calibration
        )
        return gate, calibration

    monkeypatch.setattr(cli, "_run_preflight", fake_preflight)
    base = [
        "--parent-stability-run-dir",
        "parent",
        "--runs-root",
        str(tmp_path / "runs"),
        "--run-name",
        "fixture",
        "--device",
        "cpu",
        "--require-gate",
        "preflight",
        "--no-progress",
    ]
    assert cli.main([*base, "--stage", "preflight"]) == 0
    run_dir = next((tmp_path / "runs").iterdir())
    assert cli.main(
        [
            "--parent-stability-run-dir",
            "parent",
            "--resume-run-dir",
            str(run_dir),
            "--device",
            "cpu",
            "--stage",
            "report",
            "--require-gate",
            "preflight",
            "--no-progress",
        ]
    ) == 0
    status_before = (run_dir / "run_status.json").read_bytes()
    monkeypatch.setattr(cli, "_source_record", lambda: ("source-b", ["cli.py"]))
    assert cli.main(
        [
            "--parent-stability-run-dir",
            "parent",
            "--resume-run-dir",
            str(run_dir),
            "--device",
            "cpu",
            "--stage",
            "report",
            "--require-gate",
            "preflight",
            "--no-progress",
        ]
    ) == 2
    assert (run_dir / "run_status.json").read_bytes() == status_before
