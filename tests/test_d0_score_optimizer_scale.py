from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from mnist.d0_score_optimizer_scale import (
    calibrate_initial_loss_scale,
    derive_initial_loss_scale,
    scaled_backward_and_clip,
    summarize_scaled_gradient_history,
)


def _linear_model(value: float = 2.0) -> nn.Linear:
    model = nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(value)
    return model


def _quadratic_batches(model: nn.Module, count: int, chunk: int = 64):
    def factory():
        for start in range(0, count, chunk):
            size = min(chunk, count - start)
            inputs = torch.ones(size, 1, dtype=torch.float64)
            yield model(inputs).square().mean(), size

    return factory


def test_derive_initial_loss_scale_is_positive_and_capped() -> None:
    assert derive_initial_loss_scale(4.0, 0.1) == pytest.approx(0.025)
    assert derive_initial_loss_scale(0.01, 0.1) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="raw_gradient_norm"):
        derive_initial_loss_scale(0.0, 0.1)
    with pytest.raises(ValueError, match="target_gradient_norm"):
        derive_initial_loss_scale(1.0, float("nan"))


def test_calibration_uses_exactly_256_states_and_clears_gradients() -> None:
    model = _linear_model()
    calibration = calibrate_initial_loss_scale(
        model,
        _quadratic_batches(model, 256),
        objective_kind="supervised_teacher",
        calibration_state_sha256="train-state-hash",
        binding={"role": "teacher_train", "initialization_seed": 260781},
    )
    # mean((2*x)^2) has gradient 4 at x=1.
    assert calibration.unscaled_initial_gradient_norm == pytest.approx(4.0)
    assert calibration.loss_scale == pytest.approx(0.025)
    assert calibration.scaled_initial_gradient_norm == pytest.approx(0.1)
    assert calibration.training_only == 1
    assert calibration.calibration_split == "train"
    assert calibration.to_record()["binding"]["role"] == "teacher_train"
    assert all(parameter.grad is None for parameter in model.parameters())


def test_supervised_and_implicit_calibrations_are_separate_and_deterministic() -> None:
    first_supervised = _linear_model(2.0)
    second_supervised = _linear_model(2.0)
    implicit = _linear_model(3.0)
    supervised_a = calibrate_initial_loss_scale(
        first_supervised,
        _quadratic_batches(first_supervised, 256, chunk=32),
        objective_kind="supervised_teacher",
        calibration_state_sha256="shared-training-states",
        binding={"initialization_seed": 17, "probe_plan": None},
    )
    supervised_b = calibrate_initial_loss_scale(
        second_supervised,
        _quadratic_batches(second_supervised, 256, chunk=128),
        objective_kind="supervised_teacher",
        calibration_state_sha256="shared-training-states",
        binding={"initialization_seed": 17, "probe_plan": None},
    )
    implicit_result = calibrate_initial_loss_scale(
        implicit,
        _quadratic_batches(implicit, 256),
        objective_kind="implicit_teacher_and_null",
        calibration_state_sha256="shared-training-states",
        binding={"initialization_seed": 17, "probe_plan": "fixed-probe-hash"},
    )
    assert supervised_a.to_record() == supervised_b.to_record()
    assert implicit_result.loss_scale != supervised_a.loss_scale
    assert implicit_result.binding["probe_plan"] == "fixed-probe-hash"


def test_calibration_rejects_incomplete_or_oversized_state_plans() -> None:
    model = _linear_model()
    with pytest.raises(ValueError, match="yielded 255 states"):
        calibrate_initial_loss_scale(
            model,
            _quadratic_batches(model, 255),
            objective_kind="supervised_teacher",
            calibration_state_sha256="train",
            binding={},
        )
    with pytest.raises(ValueError, match="more states"):
        calibrate_initial_loss_scale(
            model,
            _quadratic_batches(model, 257),
            objective_kind="supervised_teacher",
            calibration_state_sha256="train",
            binding={},
        )
    with pytest.raises(ValueError, match="train split"):
        calibrate_initial_loss_scale(
            model,
            _quadratic_batches(model, 256),
            objective_kind="supervised_teacher",
            calibration_state_sha256="audit",
            binding={},
            calibration_split="audit",
        )


@pytest.mark.parametrize("objective_kind", ["supervised_teacher", "implicit_teacher", "null"])
def test_scaled_backward_reports_raw_and_scaled_preclip_norms(objective_kind: str) -> None:
    parameter = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    unscaled_loss = parameter.square().sum()
    result = scaled_backward_and_clip(
        unscaled_loss,
        [parameter],
        loss_scale=0.5,
        grad_clip=0.5,
    )
    assert objective_kind  # Both objective families use this identical path.
    assert result.unscaled_loss == pytest.approx(1.0)
    assert result.scaled_loss == pytest.approx(0.5)
    assert result.raw_gradient_norm == pytest.approx(2.0)
    assert result.scaled_preclip_gradient_norm == pytest.approx(1.0)
    assert result.clipped == 1
    assert parameter.grad is not None
    assert float(parameter.grad.norm()) == pytest.approx(0.5)


def test_scaled_backward_clip_gate_uses_scaled_not_raw_norm() -> None:
    parameter = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    result = scaled_backward_and_clip(
        parameter.square().sum(),
        [parameter],
        loss_scale=0.1,
        grad_clip=1.0,
    )
    assert result.raw_gradient_norm == pytest.approx(2.0)
    assert result.scaled_preclip_gradient_norm == pytest.approx(0.2)
    assert result.clipped == 0


def test_history_summary_fails_closed_and_uses_scaled_preclip_norm() -> None:
    history = [
        {
            "step": 1,
            "raw_gradient_norm": 20.0,
            "scaled_preclip_gradient_norm": 0.2,
            "unscaled_loss": 4.0,
            "scaled_loss": 0.04,
            "clipped": 0,
        },
        {
            "step": 2,
            "raw_gradient_norm": 30.0,
            "scaled_preclip_gradient_norm": 1.5,
            "unscaled_loss": 3.0,
            "scaled_loss": 0.03,
            "clipped": 1,
        },
    ]
    summary = summarize_scaled_gradient_history(history, warmup_steps=0, grad_clip=1.0)
    assert summary["gradient_norm_source"] == "scaled_preclip_gradient_norm"
    assert summary["post_warmup_clip_fraction"] == pytest.approx(0.5)
    assert summary["recorded_clip_flags_consistent"] == 1
    assert summary["quantiles"]["raw_gradient_norm"]["q100"] == pytest.approx(30.0)

    corrupted = [dict(history[0], clipped=1)]
    mismatch = summarize_scaled_gradient_history(corrupted, warmup_steps=0, grad_clip=1.0)
    assert mismatch["recorded_clip_flags_consistent"] == 0
    assert mismatch["post_warmup_clip_fraction"] == 0.0

    with pytest.raises(ValueError, match="scaled_preclip_gradient_norm"):
        summarize_scaled_gradient_history(
            [{"step": 1, "raw_gradient_norm": 3.0}],
            warmup_steps=0,
            grad_clip=1.0,
        )


def test_scaled_backward_rejects_nonfinite_and_invalid_scale() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    with pytest.raises(ValueError, match="loss_scale"):
        scaled_backward_and_clip(
            parameter.square().sum(), [parameter], loss_scale=0.0, grad_clip=1.0
        )
    with pytest.raises(FloatingPointError, match="unscaled loss"):
        scaled_backward_and_clip(
            parameter * float("nan"), [parameter], loss_scale=0.1, grad_clip=1.0
        )
    assert math.isfinite(float(parameter.detach()))
