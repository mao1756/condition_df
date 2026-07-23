from __future__ import annotations

import json
import math

import pytest
import torch
from torch import nn

from mnist.d0_score_density_ratio_h1_gradient_control import (
    H1_GRADIENT_CONTROL_COORDINATES,
    H1_GRADIENT_CONTROL_NORM_FLOOR,
    GradientRatioControllerConfig,
    assign_controlled_gradients,
    compose_gradient_ratio_update,
    copy_parameter_gradients,
    gradient_ratio_ramp,
)


def _norm(values: tuple[torch.Tensor | None, ...]) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for value in values:
        if value is not None:
            squared = squared + value.detach().double().square().sum()
    return float(torch.sqrt(squared))


def test_gradient_ratio_ramp_is_one_indexed_and_frozen() -> None:
    assert gradient_ratio_ramp(1) == 0.0
    assert gradient_ratio_ramp(2) == 0.01
    assert gradient_ratio_ramp(51) == 0.5
    assert gradient_ratio_ramp(101) == 1.0
    assert gradient_ratio_ramp(1000) == 1.0
    assert gradient_ratio_ramp(3, ramp_steps=4) == 0.5
    with pytest.raises(ValueError, match="at least one"):
        gradient_ratio_ramp(0)
    with pytest.raises(ValueError, match="positive"):
        gradient_ratio_ramp(1, ramp_steps=0)


def test_config_record_is_stable_and_fail_closed() -> None:
    first = GradientRatioControllerConfig()
    second = GradientRatioControllerConfig()
    assert first.fingerprint == second.fingerprint
    record = first.to_record()
    assert record["fingerprint"] == first.fingerprint
    assert record["coordinate_system"] == H1_GRADIENT_CONTROL_COORDINATES
    assert record["norm_floor"] == H1_GRADIENT_CONTROL_NORM_FLOOR
    assert record["coefficient_semantics"] == "stop-gradient"
    assert record["global_clipping_owned_by_caller"] == 1
    assert record["physical_training_performed"] == 0
    assert record["sampling_performed"] == 0
    json.dumps(record, sort_keys=True)

    with pytest.raises(ValueError, match="ramp_steps"):
        GradientRatioControllerConfig(ramp_steps=0)
    with pytest.raises(ValueError, match="norm_floor"):
        GradientRatioControllerConfig(norm_floor=0.0)
    with pytest.raises(ValueError, match="tracking_rtol"):
        GradientRatioControllerConfig(tracking_rtol=-1.0)


def test_controller_hits_exact_float64_target_before_global_clip() -> None:
    bce = (
        torch.tensor([3.0, 4.0], dtype=torch.float32),
        None,
        torch.tensor([-1.0], dtype=torch.float32),
    )
    h1 = (
        torch.tensor([1.0, -2.0], dtype=torch.float32),
        torch.tensor([7.0], dtype=torch.float32),
        None,
    )
    result = compose_gradient_ratio_update(
        bce, h1, target_ratio=0.3, optimizer_step=101
    )

    assert result.controller_pass == 1
    assert result.controller_active == 1
    assert result.ratio_tracking_pass == 1
    assert result.ramp_fraction == 1.0
    assert result.target_ratio == 0.3
    assert math.isclose(
        result.h1_contribution_gradient_norm / result.bce_gradient_norm,
        0.3,
        rel_tol=1e-7,
    )
    assert result.ratio_tracking_relative_error <= 1e-7
    assert math.isclose(_norm(result.gradients), result.combined_gradient_norm)
    assert result.gradients[1] is not None
    assert torch.allclose(
        result.gradients[1], h1[1] * result.h1_coefficient
    )
    record = result.detached_record()
    assert "gradients" not in record
    assert record["global_clipping_applied"] == 0
    assert record["coordinate_system"] == H1_GRADIENT_CONTROL_COORDINATES
    json.dumps(record, sort_keys=True)


def test_positive_h1_rescaling_leaves_combined_gradient_invariant() -> None:
    bce = (torch.tensor([0.25, -0.5, 2.0], dtype=torch.float64),)
    h1 = (torch.tensor([-3.0, 1.0, 0.75], dtype=torch.float64),)
    original = compose_gradient_ratio_update(
        bce, h1, target_ratio=1.0, optimizer_step=101
    )
    for scale in (1e-7, 0.25, 11.0, 1e7):
        rescaled = compose_gradient_ratio_update(
            bce,
            (h1[0] * scale,),
            target_ratio=1.0,
            optimizer_step=101,
        )
        assert torch.allclose(
            original.gradients[0], rescaled.gradients[0], rtol=2e-15, atol=2e-15
        )
        assert math.isclose(
            rescaled.h1_coefficient,
            original.h1_coefficient / scale,
            rel_tol=2e-15,
        )


def test_step_one_and_zero_ratio_are_exact_bce_noops() -> None:
    bce = (torch.tensor([1.0, -2.0], dtype=torch.float64),)
    zero_h1 = (torch.zeros(2, dtype=torch.float64),)
    step_one = compose_gradient_ratio_update(
        bce, zero_h1, target_ratio=1.0, optimizer_step=1
    )
    assert step_one.ramp_zero_noop == 1
    assert step_one.h1_gradient_floor_hit == 1
    assert step_one.post_ramp_h1_floor_failure == 0
    assert step_one.controller_pass == 1
    assert step_one.h1_coefficient == 0.0
    assert torch.equal(step_one.gradients[0], bce[0])

    ratio_zero = compose_gradient_ratio_update(
        bce, (None,), target_ratio=0.0, optimizer_step=4000
    )
    assert ratio_zero.ramp_zero_noop == 1
    assert ratio_zero.controller_pass == 1
    assert ratio_zero.controller_active == 0
    assert ratio_zero.realized_ratio == 0.0
    assert torch.equal(ratio_zero.gradients[0], bce[0])


def test_bce_floor_is_stationary_noop_even_after_ramp() -> None:
    bce = (torch.tensor([1e-14, -1e-14], dtype=torch.float64),)
    h1 = (torch.tensor([2.0, 3.0], dtype=torch.float64),)
    result = compose_gradient_ratio_update(
        bce, h1, target_ratio=1.0, optimizer_step=500
    )
    assert result.bce_gradient_floor_hit == 1
    assert result.stationary_bce_noop == 1
    assert result.h1_coefficient == 0.0
    assert result.controller_active == 0
    assert result.ratio_tracking_pass == 1
    assert result.post_ramp_h1_floor_failure == 0
    assert result.controller_pass == 1
    assert torch.equal(result.gradients[0], bce[0])


def test_h1_floor_records_ramp_miss_and_fails_after_completed_ramp() -> None:
    bce = (torch.tensor([1.0], dtype=torch.float64),)
    h1 = (torch.tensor([1e-15], dtype=torch.float64),)
    during = compose_gradient_ratio_update(
        bce, h1, target_ratio=1.0, optimizer_step=50
    )
    assert during.h1_gradient_floor_hit == 1
    assert during.ratio_tracking_pass == 0
    assert during.post_ramp_h1_floor_failure == 0
    assert during.controller_pass == 1
    assert torch.equal(during.gradients[0], bce[0])

    after = compose_gradient_ratio_update(
        bce, h1, target_ratio=1.0, optimizer_step=101
    )
    assert after.h1_gradient_floor_hit == 1
    assert after.ratio_tracking_pass == 0
    assert after.post_ramp_h1_floor_failure == 1
    assert after.controller_pass == 0
    assert torch.equal(after.gradients[0], bce[0])


def test_controller_rejects_invalid_or_nonfinite_vectors() -> None:
    finite = (torch.tensor([1.0]),)
    with pytest.raises(ValueError, match="nonnegative"):
        compose_gradient_ratio_update(
            finite, finite, target_ratio=-0.1, optimizer_step=101
        )
    with pytest.raises(ValueError, match="differ in length"):
        compose_gradient_ratio_update(
            finite, (finite[0], None), target_ratio=0.1, optimizer_step=101
        )
    with pytest.raises(ValueError, match="shapes differ"):
        compose_gradient_ratio_update(
            finite,
            (torch.ones(2),),
            target_ratio=0.1,
            optimizer_step=101,
        )
    with pytest.raises(FloatingPointError, match="nonfinite"):
        compose_gradient_ratio_update(
            (torch.tensor([math.inf]),),
            finite,
            target_ratio=0.1,
            optimizer_step=101,
        )


def test_copy_and_assign_do_not_clip_or_alias_parameters() -> None:
    model = nn.Linear(2, 1, bias=True).double()
    parameters = tuple(model.parameters())
    loss = model(torch.tensor([[2.0, -1.0]], dtype=torch.float64)).square().sum()
    loss.backward()
    snapshot = copy_parameter_gradients(parameters)
    assert all(value is not None for value in snapshot)

    h1 = tuple(
        None if value is None else torch.ones_like(value) for value in snapshot
    )
    result = compose_gradient_ratio_update(
        snapshot, h1, target_ratio=0.1, optimizer_step=101
    )
    assign_controlled_gradients(parameters, result)
    for parameter, expected in zip(parameters, result.gradients, strict=True):
        assert expected is not None
        assert torch.equal(parameter.grad, expected)
        assert parameter.grad.data_ptr() != expected.data_ptr()

    # Assignment is intentionally preclip: its norm is exactly the controller
    # record even when it is larger than a hypothetical clipping threshold.
    assigned = copy_parameter_gradients(parameters)
    assert math.isclose(_norm(assigned), result.combined_gradient_norm)

    with pytest.raises(ValueError, match="differ in length"):
        assign_controlled_gradients(parameters, (snapshot[0],))

