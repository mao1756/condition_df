from __future__ import annotations

import numpy as np
import torch

from mnist.d0_jacobi_denoising import evaluate_alpha1_spectral
from mnist.d0_jacobi_rb_controls import (
    _whole_path_max_t_intervals,
    deterministic_kernel_controls,
    legacy_mixture_rb_target,
    target_identity_controls,
)
from mnist.d0_jacobi_rb_spectral import JacobiRBSpectralProfile


def test_legacy_ancestral_mixture_matches_rb_spectral_target() -> None:
    for x, y, exposure in (
        (0.2, 0.3, 0.75),
        (0.5, 0.5, 1.0),
        (0.8, 0.7, 0.75),
    ):
        spectral = evaluate_alpha1_spectral(x, y, exposure)
        expected = y * (1.0 - y) * float(spectral.arrival_score)
        measured = legacy_mixture_rb_target(x, y, exposure)
        assert abs(measured - expected) <= 1e-12


def test_deterministic_kernel_algebra_tiny_cpu_control() -> None:
    panel = deterministic_kernel_controls(torch.device("cpu"))
    metrics = panel.metrics
    assert metrics["float64_kernel_max_error"] <= 1e-9
    assert metrics["normalization_max_error"] <= 1e-9
    assert metrics["cdf_endpoint_max_error"] <= 1e-9
    assert metrics["cdf_monotonicity_max_violation"] == 0.0
    assert metrics["detailed_balance_max_error"] <= 1e-9
    assert metrics["semigroup_max_error"] <= 1e-9
    assert metrics["eigenmoment_1_to_8_max_error"] <= 1e-9
    assert metrics["cuda_evaluated"] == 0
    assert metrics["cuda_finite"] == 1
    assert panel.rows


def test_target_identity_tiny_control_covers_teacher_and_stationary_null() -> None:
    panel = target_identity_controls(
        count=4,
        root_seed=261121,
        profile=JacobiRBSpectralProfile(),
    )
    metrics = panel.metrics
    assert metrics["legacy_mixture_max_absolute_error"] <= 1e-8
    assert metrics["teacher_tower_simultaneous_pass"] == 1
    assert metrics["stationary_null_simultaneous_pass"] == 1
    assert metrics["all_phase_colors_pass"] == 1
    assert metrics["half_full_duration_pass"] == 1
    assert metrics["phase_duration_simultaneous_family_size"] == 27
    assert metrics["flux_conversion_max_error"] == 0.0
    assert metrics["orientation_negative_fixture_pass"] == 1
    assert metrics["h_scaling_negative_fixture_pass"] == 1
    assert metrics["invariant_beta_score_negative_fixture_pass"] == 1
    assert metrics["teacher_nonfinite_count"] == 0
    assert metrics["null_nonfinite_count"] == 0
    assert np.isfinite([row.get("mean", 0.0) for row in panel.rows]).all()
    phase_rows = [
        row for row in panel.rows
        if row.get("control") in {
            "teacher_phase_duration_orthogonality",
            "null_phase_duration_orthogonality",
        }
    ]
    assert len(phase_rows) == 4 * 2 * 3 * 2
    assert all(row["covered"] == 1 for row in phase_rows)


def test_whole_path_max_t_is_finite_replayable_and_order_invariant() -> None:
    path_ids = np.repeat(np.arange(8, dtype=np.int64), [1, 2, 3, 1, 4, 2, 3, 2])
    path_signal = np.asarray([0.8, 0.9, 1.1, 0.7, 1.2, 1.0, 0.85, 1.05])
    values = np.repeat(path_signal, [1, 2, 3, 1, 4, 2, 3, 2])[:, None]
    values = np.concatenate((values, 2.0 * values), axis=1)
    first = _whole_path_max_t_intervals(
        values, path_ids, seed=77, replicates=500, confidence=0.99
    )
    permutation = np.arange(values.shape[0])[::-1]
    second = _whole_path_max_t_intervals(
        values[permutation], path_ids[permutation],
        seed=77, replicates=500, confidence=0.99,
    )
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(np.asarray(left), np.asarray(right))
    lower, upper, critical = first
    assert np.isfinite(critical)
    assert np.all(lower > 0.0)
    assert np.all(upper >= lower)


def test_whole_path_max_t_handles_exact_zero_and_rejects_invalid_data() -> None:
    zeros = np.zeros((8, 2), dtype=np.float64)
    path_ids = np.arange(8, dtype=np.int64)
    lower, upper, critical = _whole_path_max_t_intervals(
        zeros, path_ids, seed=9, replicates=100, confidence=0.99
    )
    assert critical == 0.0
    assert np.array_equal(lower, np.zeros(2))
    assert np.array_equal(upper, np.zeros(2))
    with np.testing.assert_raises(ValueError):
        _whole_path_max_t_intervals(
            np.asarray([0.0, np.nan]), path_ids[:2], seed=1
        )
    with np.testing.assert_raises(ValueError):
        _whole_path_max_t_intervals(
            np.asarray([0.0, 1.0]), np.asarray([0]), seed=1
        )
