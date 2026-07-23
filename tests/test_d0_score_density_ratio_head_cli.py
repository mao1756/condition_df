from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import mnist.diag_d0_score_density_ratio_head_confirmation as cli
from mnist.d0_one_image_gate import ArtifactCompatibilityError
from mnist.d0_score_density_ratio import (
    build_density_ratio_panel,
    build_density_ratio_stream_plan,
)
from mnist.d0_score_density_ratio_head_gate import HeadCoordinateThresholds
from mnist.d0_score_density_ratio_paired import build_paired_mixture_stream_plan
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, natural_horizon


torch.set_num_threads(1)


def test_production_cli_defaults_and_required_gate_overrides_are_frozen() -> None:
    args = cli.parse_args(
        [
            "--parent-paired-ratio-run-dir",
            "parent",
            "--stage",
            "preflight",
            "--require-gate",
            "preflight",
        ]
    )
    assert args.root_seed == 260881
    assert args.base_channels == 32
    assert args.pilot_learning_rates == (3e-5, 1e-5)
    assert args.accumulation_levels == (8,)
    assert args.loss_scale == pytest.approx(0.05173607018770852, abs=0.0)
    assert args.confirm_model_seeds == (260891, 260892, 260893)
    assert args.pilot_steps == 2_000
    assert args.confirm_steps == 4_000

    for override in (
        ("--base-channels", "16"),
        ("--accumulation-levels", "4"),
        ("--pilot-learning-rates", "1e-5"),
        ("--loss-scale", "0.1"),
    ):
        with pytest.raises(SystemExit):
            cli.parse_args(
                [
                    "--parent-paired-ratio-run-dir",
                    "parent",
                    "--require-gate",
                    "preflight",
                    *override,
                ]
            )


def test_cli_and_bound_sources_have_no_sampler_import() -> None:
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
    assert {
        "diag_d0_score_density_ratio_head_confirmation.py",
        "d0_score_density_ratio_head.py",
        "d0_score_density_ratio_head_gate.py",
        "d0_score_density_ratio_head_provenance.py",
        "d0_score_density_ratio_paired.py",
    }.issubset(names)
    assert not any("sampler" in name.lower() for name in names)


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
        root_seed=260881, grid_size=4, horizon=horizon
    )
    paired = build_paired_mixture_stream_plan(
        root_seed=260881, grid_size=4, horizon=horizon
    )
    panels = {
        role: build_density_ratio_panel(
            evaluation,
            phase="normalized-head-unit",
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
        root_seed=260881,
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
    path: Path,
    *,
    fingerprints: dict[str, Any] | None = None,
    interrupt: tuple[int, int] | None = None,
    interrupt_sealed: bool = False,
) -> dict[str, Any]:
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
        accumulation_level=8,
        stream_plan=evaluation,
        paired_stream_plan=paired,
        fingerprints=fingerprints or {"fixture": "normalized-head-task"},
        phase="normalized-head-unit",
        thresholds=HeadCoordinateThresholds(),
        show_progress=False,
        interrupt_during_accumulation=interrupt,
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


def test_task_exact_resume_completed_reuse_and_fingerprint_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"a": 0, "b": 0}
    monkeypatch.setattr(
        cli.base, "_classification_panel_record", _fake_panel_record_factory(calls)
    )
    result = _run_task(tmp_path / "task")
    assert calls == {"a": 3, "b": 1}
    assert result["training_summary"]["optimizer_coordinate"]["groups"][1][
        "name"
    ] == "normalized_head"
    assert result["physical_training_performed"] == 0
    assert result["sampling_performed"] == 0

    assert _run_task(tmp_path / "task") == result
    assert calls == {"a": 3, "b": 1}
    with pytest.raises(ArtifactCompatibilityError):
        _run_task(tmp_path / "task", fingerprints={"fixture": "changed"})


def test_checkpoint_rejects_drift_in_saved_optimizer_state_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fingerprints = {"fixture": "normalized-head-task"}
    monkeypatch.setattr(
        cli.base,
        "_classification_panel_record",
        _fake_panel_record_factory({"a": 0, "b": 0}),
    )
    _run_task(tmp_path / "task", fingerprints=fingerprints)
    source = tmp_path / "task" / "checkpoints" / "step-00000002.pt"
    payload = torch.load(source, map_location="cpu", weights_only=False)
    payload["optimizer_state_dict"]["param_groups"][1]["lr"] *= 2.0
    tampered = tmp_path / "tampered.pt"
    cli.atomic_torch_save(tampered, payload)

    with pytest.raises(
        ArtifactCompatibilityError,
        match="optimizer state group lr changed",
    ):
        cli._load_checkpoint(
            tampered,
            device=torch.device("cpu"),
            task="dirichlet_null",
            fingerprints=fingerprints,
        )


def test_history_plot_reader_handles_quoted_list_fields(tmp_path: Path) -> None:
    path = tmp_path / "training_history.csv"
    cli.atomic_write_csv(
        path,
        [
            {
                "step": 1,
                "scaled_loss": 0.5,
                "scaled_preclip_gradient_norm": 0.25,
                "clipped": 0,
                "microbatch_losses": [0.1, 0.2, 0.3],
            }
        ],
    )
    columns = cli._read_history_columns(
        path,
        ("step", "scaled_loss", "scaled_preclip_gradient_norm", "clipped"),
    )
    assert columns["step"].tolist() == [1.0]
    assert columns["scaled_loss"].tolist() == [0.5]


def test_interruption_restarts_uncommitted_accumulation_step_exactly(
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
        _run_task(tmp_path / "resumed", interrupt=(1, 3))
    resumed = _run_task(tmp_path / "resumed")
    assert resumed["metrics"] == uninterrupted["metrics"]

    full_payload = torch.load(
        tmp_path / "full" / "checkpoints" / "finalized-step-00000002.pt",
        map_location="cpu",
        weights_only=False,
    )
    resumed_payload = torch.load(
        tmp_path / "resumed" / "checkpoints" / "finalized-step-00000002.pt",
        map_location="cpu",
        weights_only=False,
    )
    _assert_nested_exact(full_payload, resumed_payload)


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


def _passing_coordinate_forensics() -> dict[str, Any]:
    return {
        "float64_logit_max_abs_error": 0.0,
        "float64_bce_max_abs_error": 0.0,
        "cuda_logit_max_abs_error": 0.0,
        "cuda_bce_max_abs_error": 0.0,
        "state_gradient_relative_error": 0.0,
        "edge_score_relative_error": 0.0,
        "flux_relative_error": 0.0,
        "head_gradient_scale_relative_error": 0.0,
        "backbone_gradient_relative_error": 0.0,
        "adamw_coordinate_max_relative_error": 0.0,
        "ema_coordinate_max_relative_error": 0.0,
        "adamw_group_learning_rate_pass": 1,
        "adamw_group_epsilon_pass": 1,
        "adamw_group_weight_decay_pass": 1,
        "parent_forensic_finite": 1,
        "median_legacy_head_squared_gradient_share": 0.99,
        "legacy_checkpoint_report_only_pass": 1,
    }


def test_tiny_cpu_preflight_exercises_real_stream_accumulation_and_backward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dynamics = _dynamics()
    horizon = float(natural_horizon(dynamics))
    evaluation = build_density_ratio_stream_plan(
        root_seed=260881, grid_size=4, horizon=horizon
    )
    paired = build_paired_mixture_stream_plan(
        root_seed=260881, grid_size=4, horizon=horizon
    )
    args = argparse.Namespace(
        root_seed=260881,
        grid_size=4,
        base_channels=4,
        preflight_paths=2,
        preflight_confidence=0.99,
        bootstrap_reps=16,
        loss_scale=cli.PARENT_LOSS_SCALE,
    )
    monkeypatch.setattr(
        cli,
        "_run_head_coordinate_forensics",
        lambda *args, **kwargs: _passing_coordinate_forensics(),
    )
    monkeypatch.setattr(cli.base, "_boundary_certificate", lambda model: {"passed": 1})

    gate = cli._run_preflight(
        tmp_path,
        args=args,
        manifest={"scientific_fingerprint": "fixture"},
        provenance={"passed": 1},
        dynamics=dynamics,
        device=torch.device("cpu"),
        stream_plan=evaluation,
        paired_stream_plan=paired,
        thresholds=HeadCoordinateThresholds(),
    )
    metrics = cli._json_load(tmp_path / "normalized_head_preflight.json")
    assert metrics["paired_estimator_pass"] == 1
    assert metrics["stream_replay_pass"] == 1
    assert metrics["finite_device_backward_pass"] == 1
    assert metrics["actual_paired_accumulation_records"][0][
        "effective_clusters"
    ] == 256
    # This tiny fixture deliberately keeps the production gate frozen at 28x28/32.
    assert gate["passed"] == 0
    assert gate["subchecks"]["grid_cells"]["passed"] == 0
    assert gate["subchecks"]["base_channels"]["passed"] == 0


def test_required_gate_failure_commits_readable_terminal_artifacts(
    tmp_path: Path,
) -> None:
    report = {
        "required_gate": "pilot",
        "required_gate_pass": 0,
        "decision": {
            "decision": "classification_coordinate_repair_unresolved",
            "recommended_next_action": "investigate function-space trust",
            "physical_training_authorized": 0,
        },
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    cli._save_report(tmp_path, report)
    assert cli._finish(
        tmp_path, report=report, stage="pilot", phase="pilot"
    ) == 2

    assert (tmp_path / "normalized_head_control_gate.json").is_file()
    assert (tmp_path / "normalized_head_decision.json").is_file()
    assert (tmp_path / "artifact_registry.json").is_file()
    status = cli._json_load(tmp_path / "run_status.json")
    assert status["status"] == "complete"
    assert status["outcome"] == "gate_failed"
    assert status["required_gate_pass"] == 0
    assert status["physical_training_performed"] == 0
    assert status["sampling_performed"] == 0
    assert cli._verify_terminal_registry(tmp_path)["records"]
