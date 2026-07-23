from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import mnist.d0_score_density_ratio_h1_gradient_control_task as task_runner
import mnist.diag_d0_score_density_ratio_head_confirmation as head_runner
from mnist.d0_one_image_gate import ArtifactCompatibilityError
from mnist.d0_score_density_ratio import (
    build_density_ratio_panel,
    build_density_ratio_stream_plan,
)
from mnist.d0_score_density_ratio_h1_gradient_control import (
    GradientRatioControllerConfig,
)
from mnist.d0_score_density_ratio_h1_trust import build_h1_trust_plan
from mnist.d0_score_density_ratio_head_gate import HeadCoordinateThresholds
from mnist.d0_score_density_ratio_paired import build_paired_mixture_stream_plan
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, natural_horizon


torch.set_num_threads(1)


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


def _calibration() -> dict[str, Any]:
    return {
        "schema": "experiment12-d0-score-density-ratio-h1-trust-calibration",
        "schema_version": 1,
        "calibration_version": "d0-one-shadow-step-h1-calibration-v1",
        "passed": 1,
        "value_scale": 0.1,
        "energy_scale": 0.1,
        "lambda_base": 0.01,
    }


def _fixture() -> tuple[Any, ...]:
    dynamics = _dynamics()
    horizon = float(natural_horizon(dynamics))
    evaluation = build_density_ratio_stream_plan(
        root_seed=261041, grid_size=4, horizon=horizon
    )
    paired = build_paired_mixture_stream_plan(
        root_seed=261041, grid_size=4, horizon=horizon
    )
    trust = build_h1_trust_plan(grid_size=4, horizon=horizon, root_seed=261041)
    panels = {
        role: build_density_ratio_panel(
            evaluation,
            phase="h1-gradient-task-unit",
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
        root_seed=261041,
        grad_clip=1.0,
        ema_decay=0.99,
        bootstrap_reps=16,
        bootstrap_confidence=0.90,
        validation_batch_size=64,
        clip_warmup_steps=0,
    )
    return dynamics, evaluation, paired, trust, panels, args


def _fake_panel_record_factory(calls: dict[str, int]):
    def fake(model, panel, **kwargs):
        del model, kwargs
        calls[panel.role] += 1
        # Panel A is discovery-only.  The sealed null B record remains
        # negative, so the analytic-zero checkpoint is selected.
        bce = 0.69 if panel.role == "a" and calls[panel.role] == 1 else 0.55
        if panel.role == "b":
            bce = 0.70
        scope = {
            "bce": bce,
            "model_bce": bce,
            "classification_risk": bce,
            "zero_logit_bce": 0.6931471805599453,
            "improvement": 0.6931471805599453 - bce,
            "objective_improvement": 0.6931471805599453 - bce,
            "lower_bound": -0.01,
            "upper_bound": 0.0,
            "confidence": 0.90,
            "bootstrap": {"path_ids": [1], "path_values": [-0.01]},
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


def _run(
    path: Path,
    *,
    fingerprints: dict[str, Any] | None = None,
    interrupt_trust: tuple[int, int] | None = None,
    h1_ratio: float = 0.1,
    defer_panel_b: bool = False,
) -> dict[str, Any]:
    dynamics, evaluation, paired, trust, panels, args = _fixture()
    return task_runner.run_gradient_control_paired_density_ratio_task(
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
        trust_plan=trust,
        calibration=_calibration(),
        h1_ratio=h1_ratio,
        # A one-step ramp makes the two-step fixture exercise an active,
        # post-ramp controller update while preserving production semantics.
        controller_config=GradientRatioControllerConfig(ramp_steps=1),
        fingerprints=fingerprints
        or {"fixture": "h1-gradient-task", "grid_cells": 16},
        phase="h1-gradient-task-unit",
        thresholds=HeadCoordinateThresholds(),
        show_progress=False,
        interrupt_during_trust_bank=interrupt_trust,
        defer_panel_b=defer_panel_b,
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
        for a, b in zip(left, right, strict=True):
            _assert_nested_exact(a, b)
    else:
        assert left == right


def test_task_fingerprint_binds_controller_ratio_and_forensic_lambda() -> None:
    _, _, _, trust, _, _ = _fixture()
    config = GradientRatioControllerConfig(ramp_steps=1)
    first = task_runner.gradient_control_task_fingerprints(
        {"fixture": 1},
        trust_plan=trust,
        calibration=_calibration(),
        target_ratio=0.1,
        controller_config=config,
    )
    second = task_runner.gradient_control_task_fingerprints(
        {"fixture": 1},
        trust_plan=trust,
        calibration=_calibration(),
        target_ratio=0.3,
        controller_config=config,
    )
    assert first != second
    assert first["h1_target_gradient_ratio"] == pytest.approx(0.1)
    assert first["h1_controller_config_fingerprint"] == config.fingerprint
    assert first["h1_controller_config"]["fingerprint"] == config.fingerprint
    assert first["h1_legacy_lambda_base_advisory"] == pytest.approx(0.01)
    assert first["h1_legacy_lambda_used_for_optimization"] == 0
    with pytest.raises(ArtifactCompatibilityError, match="conflicts"):
        task_runner.gradient_control_task_fingerprints(
            {"h1_target_gradient_ratio": 1.0},
            trust_plan=trust,
            calibration=_calibration(),
            target_ratio=0.1,
            controller_config=config,
        )


def test_fixed_endpoint_controller_health_and_sealed_b_single_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"a": 0, "b": 0}
    monkeypatch.setattr(
        task_runner.base, "_classification_panel_record", _fake_panel_record_factory(calls)
    )
    result = _run(tmp_path / "task")
    metrics = result["metrics"]
    assert metrics["fixed_endpoint_step"] == 2
    assert metrics["nominee_step"] == 2
    assert metrics["selected_step"] == 0
    assert metrics["selection"]["endpoint_step"] == 2
    assert metrics["selection"]["endpoint_policy"] == "ema-step-2"
    assert metrics["controller_health_pass"] == 1
    assert metrics["h1_health_pass"] == 1
    assert metrics["controller_active_fraction_post_ramp"] == 1.0
    assert metrics["controller_maximum_tracking_error_post_ramp"] <= 1e-4
    assert calls == {"a": 3, "b": 1}

    payload = torch.load(
        tmp_path / "task" / "checkpoints" / "finalized-step-00000002.pt",
        map_location="cpu",
        weights_only=False,
    )
    history = payload["history"]
    assert len(history) == 2
    assert history[0]["controller_target_ratio"] == 0.0
    assert history[0]["controller_ramp_zero_noop"] == 1
    assert history[1]["controller_target_ratio"] == pytest.approx(0.1)
    assert history[1]["controller_active"] == 1
    assert history[1]["controller_ratio_tracking_pass"] == 1
    assert history[1]["controller_ratio_tracking_relative_error"] <= 1e-4
    assert history[1]["h1_legacy_multiplier_used_for_optimization"] == 0
    assert history[1]["h1_effective_multiplier"] != pytest.approx(0.001)

    before = dict(calls)
    assert _run(tmp_path / "task") == result
    assert calls == before


def test_uncommitted_trust_bank_resume_matches_uninterrupted_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        task_runner.base,
        "_classification_panel_record",
        _fake_panel_record_factory({"a": 0, "b": 0}),
    )
    uninterrupted = _run(tmp_path / "full")

    calls = {"a": 0, "b": 0}
    monkeypatch.setattr(
        task_runner.base, "_classification_panel_record", _fake_panel_record_factory(calls)
    )
    with pytest.raises(RuntimeError, match="trust banks"):
        _run(tmp_path / "resume", interrupt_trust=(2, 0))
    resumed = _run(tmp_path / "resume")
    assert resumed["metrics"] == uninterrupted["metrics"]

    full_payload = torch.load(
        tmp_path / "full" / "checkpoints" / "finalized-step-00000002.pt",
        map_location="cpu",
        weights_only=False,
    )
    resumed_payload = torch.load(
        tmp_path / "resume" / "checkpoints" / "finalized-step-00000002.pt",
        map_location="cpu",
        weights_only=False,
    )
    _assert_nested_exact(full_payload, resumed_payload)

    with pytest.raises(ArtifactCompatibilityError):
        _run(
            tmp_path / "resume",
            fingerprints={"fixture": "changed", "grid_cells": 16},
        )


def test_panel_b_is_deferred_for_nonselected_task_and_opened_once_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"a": 0, "b": 0}
    monkeypatch.setattr(
        task_runner.base, "_classification_panel_record", _fake_panel_record_factory(calls)
    )
    task_dir = tmp_path / "deferred"
    a_only = _run(task_dir, defer_panel_b=True)
    assert calls == {"a": 3, "b": 0}
    assert a_only["panel_b_deferred"] == 1
    assert a_only["panel_b_evaluation_count"] == 0
    assert a_only["metrics"]["panel_b_deferred"] == 1
    assert a_only["metrics"]["panel_b_evaluation_count"] == 0
    assert a_only["metrics"]["selection"]["evaluation_status"] == "not_evaluated"
    assert a_only["metrics"]["selection"]["selected_step"] == 0
    assert not (task_dir / "sealed_panel_b.json").exists()
    assert not (task_dir / "checkpoints" / "best.json").exists()
    status = task_runner._json_load(task_dir / "task_status.json")
    assert status["status"] == "awaiting_panel_b"

    # Re-reading an unselected task in deferred mode neither trains nor opens B.
    repeated = _run(task_dir, defer_panel_b=True)
    assert repeated == a_only
    assert calls == {"a": 3, "b": 0}

    finalized = _run(task_dir, defer_panel_b=False)
    assert calls == {"a": 3, "b": 1}
    assert finalized["panel_b_deferred"] == 0
    assert finalized["panel_b_evaluation_count"] == 1
    assert (task_dir / "sealed_panel_b.json").is_file()
    assert (task_dir / "checkpoints" / "best.json").is_file()
    assert task_runner._json_load(task_dir / "task_status.json")["status"] == "complete"

    # A complete selected task is reused, keeping the sealed panel single-use.
    assert _run(task_dir, defer_panel_b=False) == finalized
    assert calls == {"a": 3, "b": 1}


def test_deferred_panel_b_seal_survives_interruption_without_double_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"a": 0, "b": 0}
    monkeypatch.setattr(
        task_runner.base, "_classification_panel_record", _fake_panel_record_factory(calls)
    )
    dynamics, evaluation, paired, trust, panels, args = _fixture()
    task_dir = tmp_path / "sealed-resume"
    _run(task_dir, defer_panel_b=True)

    kwargs = {
        "task_dir": task_dir,
        "task": "dirichlet_null",
        "selection_panels": panels,
        "audit_panels": None,
        "dynamics": dynamics,
        "args": args,
        "device": torch.device("cpu"),
        "model_seed": 17,
        "learning_rate": 1e-5,
        "accumulation_level": 8,
        "stream_plan": evaluation,
        "paired_stream_plan": paired,
        "trust_plan": trust,
        "calibration": _calibration(),
        "h1_ratio": 0.1,
        "controller_config": GradientRatioControllerConfig(ramp_steps=1),
        "fingerprints": {"fixture": "h1-gradient-task", "grid_cells": 16},
        "phase": "h1-gradient-task-unit",
        "thresholds": HeadCoordinateThresholds(),
        "show_progress": False,
    }
    with pytest.raises(RuntimeError, match="sealed panel B"):
        task_runner.run_gradient_control_paired_density_ratio_task(
            **kwargs, interrupt_after_sealed_panel_b=True
        )
    assert calls == {"a": 3, "b": 1}
    assert (task_dir / "sealed_panel_b.json").is_file()
    assert task_runner._json_load(task_dir / "task_status.json")["status"] == (
        "awaiting_panel_b"
    )

    finalized = task_runner.run_gradient_control_paired_density_ratio_task(**kwargs)
    assert finalized["panel_b_evaluation_count"] == 1
    assert calls == {"a": 3, "b": 1}
    assert task_runner._json_load(task_dir / "task_status.json")["status"] == "complete"


def test_zero_ratio_matches_unchanged_normalized_head_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dynamics, evaluation, paired, _, panels, args = _fixture()
    monkeypatch.setattr(
        task_runner.base,
        "_classification_panel_record",
        _fake_panel_record_factory({"a": 0, "b": 0}),
    )
    zero = _run(tmp_path / "controlled-zero", h1_ratio=0.0)
    assert zero["metrics"]["controller_health_pass"] == 1
    assert zero["metrics"]["controller_active_fraction_post_ramp"] == 1.0

    monkeypatch.setattr(
        head_runner.base,
        "_classification_panel_record",
        _fake_panel_record_factory({"a": 0, "b": 0}),
    )
    head_runner.run_paired_density_ratio_task(
        task_dir=tmp_path / "baseline",
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
        fingerprints={"fixture": "baseline", "grid_cells": 16},
        phase="h1-gradient-task-unit",
        thresholds=HeadCoordinateThresholds(),
        show_progress=False,
    )
    controlled_payload = torch.load(
        tmp_path / "controlled-zero" / "checkpoints" / "finalized-step-00000002.pt",
        map_location="cpu",
        weights_only=False,
    )
    baseline_payload = torch.load(
        tmp_path / "baseline" / "checkpoints" / "finalized-step-00000002.pt",
        map_location="cpu",
        weights_only=False,
    )
    for name in ("model_state_dict", "ema_state_dict", "optimizer_state_dict"):
        _assert_nested_exact(controlled_payload[name], baseline_payload[name])
    assert all(
        row["controller_h1_coefficient"] == 0.0
        and row["controller_realized_ratio"] == 0.0
        for row in controlled_payload["history"]
    )


def test_gradient_control_task_source_has_no_sampler_import() -> None:
    tree = ast.parse(Path(task_runner.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("sampler" in value.lower() for value in imported)
