from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import mnist.d0_score_density_ratio_h1_task as task_runner
import mnist.diag_d0_score_density_ratio_head_confirmation as head_runner
from mnist.d0_one_image_gate import ArtifactCompatibilityError
from mnist.d0_score_density_ratio import (
    build_density_ratio_panel,
    build_density_ratio_stream_plan,
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
        root_seed=261001, grid_size=4, horizon=horizon
    )
    paired = build_paired_mixture_stream_plan(
        root_seed=261001, grid_size=4, horizon=horizon
    )
    trust = build_h1_trust_plan(grid_size=4, horizon=horizon, root_seed=261001)
    panels = {
        role: build_density_ratio_panel(
            evaluation,
            phase="h1-task-unit",
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
        root_seed=261001,
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
) -> dict[str, Any]:
    dynamics, evaluation, paired, trust, panels, args = _fixture()
    return task_runner.run_h1_paired_density_ratio_task(
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
        fingerprints=fingerprints or {"fixture": "h1-task", "grid_cells": 16},
        phase="h1-task-unit",
        thresholds=HeadCoordinateThresholds(),
        show_progress=False,
        interrupt_during_trust_bank=interrupt_trust,
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


def test_h1_fingerprints_bind_ratio_plan_and_calibration() -> None:
    dynamics, _, _, trust, _, _ = _fixture()
    del dynamics
    first = task_runner.h1_task_fingerprints(
        {"fixture": 1}, trust_plan=trust, calibration=_calibration(), h1_ratio=0.1
    )
    second = task_runner.h1_task_fingerprints(
        {"fixture": 1}, trust_plan=trust, calibration=_calibration(), h1_ratio=0.3
    )
    assert first["h1_effective_multiplier"] == pytest.approx(0.001)
    assert first["h1_trust_plan_fingerprint"] == trust.fingerprint
    assert first != second
    with pytest.raises(ArtifactCompatibilityError):
        task_runner.h1_task_fingerprints(
            {"h1_ratio": 1.0},
            trust_plan=trust,
            calibration=_calibration(),
            h1_ratio=0.1,
        )


def test_uncommitted_trust_bank_replays_exactly_and_completed_task_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        task_runner.base,
        "_classification_panel_record",
        _fake_panel_record_factory({"a": 0, "b": 0}),
    )
    uninterrupted = _run(tmp_path / "full")
    assert uninterrupted["training_summary"]["h1_ratio"] == pytest.approx(0.1)
    assert uninterrupted["metrics"]["h1_health_pass"] == 1
    assert uninterrupted["training_summary"]["h1_health_pass"] == 1
    assert uninterrupted["training_summary"]["h1_diagnostics"][
        "operator_version"
    ] == "d0-stopped-ema-l2-gamma-increment-v1"
    history = task_runner._json_load(
        tmp_path / "full" / "training_summary.json"
    )
    assert history["h1_health_pass"] == 1

    calls = {"a": 0, "b": 0}
    monkeypatch.setattr(
        task_runner.base, "_classification_panel_record", _fake_panel_record_factory(calls)
    )
    with pytest.raises(RuntimeError, match="trust banks"):
        _run(tmp_path / "resume", interrupt_trust=(2, 0))
    resumed = _run(tmp_path / "resume")
    assert resumed["metrics"] == uninterrupted["metrics"]
    before = dict(calls)
    assert _run(tmp_path / "resume") == resumed
    assert calls == before

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
    for row in full_payload["history"]:
        assert row["h1_value"] == row["h1_normalized_objective"]
        assert row["h1_effective_loss"] == row["h1_scaled_optimizer_loss"]
        assert row["bce_gradient_norm"] == row["bce_scaled_gradient_norm"]
        assert row["h1_gradient_norm"] == row["h1_scaled_gradient_norm"]

    with pytest.raises(ArtifactCompatibilityError):
        _run(
            tmp_path / "resume",
            fingerprints={"fixture": "changed", "grid_cells": 16},
        )


def test_h1_task_source_has_no_sampler_import() -> None:
    tree = ast.parse(Path(task_runner.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("sampler" in value.lower() for value in imported)


def test_zero_ratio_preserves_normalized_head_training_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dynamics, evaluation, paired, _, panels, args = _fixture()
    monkeypatch.setattr(
        task_runner.base,
        "_classification_panel_record",
        _fake_panel_record_factory({"a": 0, "b": 0}),
    )
    _run(tmp_path / "h1-zero", h1_ratio=0.0)
    monkeypatch.setattr(
        head_runner.base,
        "_classification_panel_record",
        _fake_panel_record_factory({"a": 0, "b": 0}),
    )
    head_runner.run_paired_density_ratio_task(
        task_dir=tmp_path / "legacy",
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
        fingerprints={"fixture": "legacy", "grid_cells": 16},
        phase="h1-task-unit",
        thresholds=HeadCoordinateThresholds(),
        show_progress=False,
    )
    h1_payload = torch.load(
        tmp_path / "h1-zero" / "checkpoints" / "finalized-step-00000002.pt",
        map_location="cpu",
        weights_only=False,
    )
    legacy_payload = torch.load(
        tmp_path / "legacy" / "checkpoints" / "finalized-step-00000002.pt",
        map_location="cpu",
        weights_only=False,
    )
    _assert_nested_exact(
        h1_payload["model_state_dict"], legacy_payload["model_state_dict"]
    )
    _assert_nested_exact(
        h1_payload["ema_state_dict"], legacy_payload["ema_state_dict"]
    )
    _assert_nested_exact(
        h1_payload["optimizer_state_dict"], legacy_payload["optimizer_state_dict"]
    )
