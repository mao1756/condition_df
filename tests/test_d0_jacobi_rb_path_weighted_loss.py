from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from mnist.d0_jacobi_rb_path_weighted_loss import (
    PathWeightedLossConfig,
    PathWeightedLossError,
    mobility_weight_statistics,
    path_weighted_raw_target_mse,
    path_weighted_target_scale_squared,
)


def test_path_weighted_loss_equals_jacobi_q_metric_without_floor_hits() -> None:
    mobility = torch.linspace(0.01, 0.24, 392, dtype=torch.float64)[None, :].repeat(2, 1)
    q_true = torch.linspace(-2.0, 2.0, 392, dtype=torch.float64)[None, :].repeat(2, 1)
    q_prediction = q_true + 0.25
    target = mobility * q_true
    prediction = mobility * q_prediction
    scale = path_weighted_target_scale_squared(target, mobility)
    normalized, raw_weighted, raw_unweighted = path_weighted_raw_target_mse(
        prediction,
        target,
        mobility,
        target_scale_squared=scale,
    )
    expected = torch.mean(mobility * (q_prediction - q_true).square())
    assert torch.allclose(raw_weighted, expected, atol=0.0, rtol=2e-15)
    assert torch.allclose(normalized, expected / scale, atol=0.0, rtol=2e-15)
    assert torch.allclose(
        raw_unweighted,
        torch.mean(mobility.square() * (q_prediction - q_true).square()),
        atol=0.0,
        rtol=2e-15,
    )


def test_zero_mobility_lanes_are_excluded_but_must_have_zero_target() -> None:
    mobility = torch.full((1, 392), 0.2, dtype=torch.float64)
    mobility[0, 0] = 0.0
    target = torch.ones_like(mobility)
    target[0, 0] = 0.0
    prediction = target.clone()
    scale = path_weighted_target_scale_squared(target, mobility)
    normalized, raw, unweighted = path_weighted_raw_target_mse(
        prediction, target, mobility, target_scale_squared=scale
    )
    assert normalized.item() == raw.item() == unweighted.item() == 0.0
    target[0, 0] = 1.0
    with pytest.raises(PathWeightedLossError, match="zero-mobility"):
        path_weighted_target_scale_squared(target, mobility)


def test_mobility_floor_caps_weights_without_forming_target_quotient() -> None:
    mobility = np.asarray([0.0, 1e-12, 1e-5, 1e-4, 0.25], dtype=np.float64)
    statistics = mobility_weight_statistics(
        mobility, config=PathWeightedLossConfig(mobility_floor=1e-4)
    )
    assert statistics.maximum_weight == 10_000.0
    assert statistics.floor_hit_count == 2
    assert statistics.zero_mobility_count == 1
    source = inspect.getsource(path_weighted_raw_target_mse)
    assert "target64 / mobility" not in source
    assert "exact_target / mobility" not in source


def test_weighted_conditional_mean_is_unchanged() -> None:
    # Two W groups with different positive weights.  Each group's weighted risk
    # is minimized at its ordinary conditional target mean because the weight
    # is constant after conditioning on W.
    targets = np.asarray([-2.0, 0.0, 4.0, 8.0], dtype=np.float64)
    groups = np.asarray([0, 0, 1, 1], dtype=np.int64)
    weights = np.asarray([100.0, 100.0, 2.0, 2.0], dtype=np.float64)
    means = np.asarray([targets[groups == group].mean() for group in (0, 1)])
    grid = np.linspace(-4.0, 10.0, 1401)
    minimizers = []
    for group in (0, 1):
        selected = groups == group
        risks = [
            np.sum(weights[selected] * (value - targets[selected]) ** 2)
            for value in grid
        ]
        minimizers.append(grid[int(np.argmin(risks))])
    np.testing.assert_allclose(minimizers, means, atol=0.011)
