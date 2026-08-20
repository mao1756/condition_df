from __future__ import annotations

import math

import numpy as np
import pytest

from mnist.d0_jacobi_rb_boundary_tangent_quartile_direction_adjudication import (
    CANDIDATE_COUNT,
    CRITICAL_VALUE,
    ROLE_ORDER,
    DirectionAdjudicationError,
    cancellation_diagnostics,
    canonical_candidate_order,
    classify_mechanism_flags,
    classify_power_only_evidence,
    compare_direction_maps,
    directional_compatibility_screen,
    evaluate_cross_role_directional_stability,
    forecast_required_paths,
    gain_transfer_diagnostics,
    path_stability_diagnostics,
    quadratic_improvement,
    reduce_quadratic_cells,
    scalar_optimum,
    summarize_optimization_rotation,
)


def _row_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path_ids = np.repeat(np.arange(10_000, 10_032, dtype=np.int64), 56)
    phases = np.tile(np.repeat(np.arange(7, dtype=np.int64), 8), 32)
    midpoints = np.tile(np.arange(8, dtype=np.int64), 32 * 7)
    return path_ids, phases, midpoints


def test_candidate_and_role_order_are_frozen() -> None:
    candidates = canonical_candidate_order()
    assert len(candidates) == CANDIDATE_COUNT == 480
    assert candidates[0] == "q0.seed261332.update0100"
    assert candidates[-1] == "q3.seed261343.update4000"
    assert ROLE_ORDER == ("gain_calibration", "training_rank")


def test_binary64_quadratic_reduction_reconstructs_unit_and_parent_gains() -> None:
    paths, phases, midpoints = _row_grid()
    target = np.stack(
        (
            np.linspace(-1.0, 1.0, paths.size),
            np.linspace(0.25, 1.25, paths.size),
        ),
        axis=1,
    )
    prediction = 0.4 * target
    result = reduce_quadratic_cells(
        targets=target,
        predictions=prediction.astype(np.float32),
        row_path_ids=paths,
        phases=phases,
        midpoint_indices=midpoints,
        expected_path_ids=np.arange(10_000, 10_032, dtype=np.int64),
        parent_gain=0.25,
        diagnostic_gain=0.75,
    )
    assert result.cross_term.shape == (32, 7, 8)
    assert np.array_equal(result.fine_cell_row_count, np.ones((32, 7, 8)))
    assert np.allclose(
        result.raw_improvement,
        quadratic_improvement(result.cross_term, result.prediction_energy, 1.0),
        rtol=0.0,
        atol=5e-15,
    )
    assert np.allclose(
        result.parent_gain_improvement,
        quadratic_improvement(result.cross_term, result.prediction_energy, 0.25),
        rtol=0.0,
        atol=5e-15,
    )
    assert np.array_equal(
        result.diagnostic_gain_improvement,
        quadratic_improvement(result.cross_term, result.prediction_energy, 0.75),
    )
    assert result.maximum_identity_error <= 5e-15
    assert not result.cross_term.flags.writeable

    keep = np.arange(paths.size) != 0
    with pytest.raises(DirectionAdjudicationError, match="every path/phase/midpoint"):
        reduce_quadratic_cells(
            targets=target[keep],
            predictions=prediction[keep],
            row_path_ids=paths[keep],
            phases=phases[keep],
            midpoint_indices=midpoints[keep],
        )


def test_unclipped_optimum_and_transfer_efficiency_keep_negative_or_undefined_values() -> None:
    optimum = scalar_optimum(2.0, 0.5)
    assert optimum["lambda_star"] == 4.0
    assert optimum["optimal_improvement"] == 8.0
    assert optimum["gain_clipped_or_projected"] == 0
    assert scalar_optimum(-1.0, 2.0)["lambda_star"] is None

    negative = gain_transfer_diagnostics(
        gain_cross_term=1.0,
        gain_prediction_energy=0.25,
        rank_cross_term=0.1,
        rank_prediction_energy=1.0,
        gain_permitted=True,
    )
    assert negative["lambda_gain_star"] == 4.0
    assert negative["I_rank_at_lambda_gain"] < 0.0
    assert negative["transfer_efficiency"] < 0.0
    undefined = gain_transfer_diagnostics(
        gain_cross_term=1.0,
        gain_prediction_energy=1.0,
        rank_cross_term=-1.0,
        rank_prediction_energy=1.0,
    )
    assert undefined["I_rank_at_lambda_rank"] is None
    assert undefined["transfer_efficiency"] is None


def test_directional_screen_has_exact_51_of_56_and_q1_sentinel_rules() -> None:
    cells = np.ones((7, 8), dtype=np.float64)
    for phase, midpoint in ((0, 0), (1, 1), (2, 2), (3, 3), (4, 7)):
        cells[phase, midpoint] = -0.01
    q2 = directional_compatibility_screen(cells, quartile=2)
    q1 = directional_compatibility_screen(cells, quartile=1)
    assert q2["positive_fine_cell_count"] == 51
    assert q2["passed"] == 1
    assert q1["passed"] == 0
    assert q1["reason_code"] == "q1_phase4_midpoint7_nonpositive"

    cells[5, 5] = -0.01
    assert (
        directional_compatibility_screen(cells, quartile=2)["reason_code"]
        == "positive_fine_cells_below_51"
    )


def test_path_stability_and_cross_role_rule_use_24_paths_and_all_loo_values() -> None:
    cross = np.ones((32, 7, 8), dtype=np.float64)
    transferred = np.full_like(cross, 0.5)
    stability = path_stability_diagnostics(
        cross, transferred_improvements=transferred
    )
    assert stability["positive_cross_term_path_count"] == 32
    assert stability["minimum_leave_one_path_out_cross_term"] == 1.0
    assert stability["path_standard_deviation"] == 0.0
    assert stability["positive_transferred_improvement_path_count"] == 32
    screen = directional_compatibility_screen(cross, quartile=2)
    cross_role = evaluate_cross_role_directional_stability(
        gain_screen=screen,
        rank_screen=screen,
        gain_path_stability={
            **stability,
            "positive_cross_term_path_count": 24,
        },
        rank_path_stability=stability,
    )
    assert cross_role["passed"] == 1
    assert cross_role["paths_paired_across_roles"] == 0
    failed = evaluate_cross_role_directional_stability(
        gain_screen=screen,
        rank_screen=screen,
        gain_path_stability={
            **stability,
            "positive_cross_term_path_count": 23,
        },
        rank_path_stability=stability,
    )
    assert failed["passed"] == 0


def test_cancellation_and_rotation_use_the_frozen_map_geometry() -> None:
    phase = np.arange(7, dtype=np.float64)[:, None]
    midpoint = np.arange(8, dtype=np.float64)[None, :]
    additive = phase + 2.0 * midpoint
    cancellation = cancellation_diagnostics(additive)
    assert cancellation["interaction_sum_of_squares"] < 1e-25
    assert cancellation["two_way_identity_error"] < 1e-12

    first = np.ones((7, 8), dtype=np.float64)
    comparison = compare_direction_maps(first, -first)
    assert comparison["cosine_similarity"] == -1.0
    assert comparison["cell_sign_flip_fraction"] == 1.0
    assert comparison["checkpoint_selected"] == 0

    maps = {
        seed: {100: first, 200: -first, 300: first}
        for seed in (1, 2, 3)
    }
    rotation = summarize_optimization_rotation(maps)
    assert rotation["rotating_seed_count"] == 3
    assert rotation["optimization_time_rotation"] == 1


def test_forecast_is_infinite_for_incompatible_points_and_uses_frozen_formula() -> None:
    positive = np.stack(
        [np.full((7, 8), 1.0 + index / 31.0) for index in range(32)]
    )
    forecast = forecast_required_paths(positive, quartile=2)
    path_values = np.asarray(forecast["path_point_estimate"])
    expected = math.ceil(
        (CRITICAL_VALUE * forecast["path_standard_deviation"] / path_values) ** 2
    )
    assert forecast["n_raw"] == expected
    assert forecast["n_rounded"] % 32 == 0
    assert forecast["reason"] == "finite_power_forecast"

    incompatible = positive.copy()
    incompatible[:, 0, :] = -100.0
    failed = forecast_required_paths(incompatible, quartile=2)
    assert math.isinf(failed["required_path_count"])
    assert failed["n_raw"] is None
    assert failed["reason"] == "negative_or_incompatible_point_effect"


def _candidate(**updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "C_gain": 1.0,
        "C_rank": 1.0,
        "gain_directional_screen_passed": 1,
        "rank_directional_screen_passed": 1,
        "cross_role_directionally_stable": 1,
        "lambda_gain_star": 1.0,
        "gain_permitted": 1,
        "I_rank_at_lambda_gain": 1.0,
        "transferred_rank_point_screen_passed": 1,
        "fixed_design_margin": 1.0,
    }
    record.update(updates)
    return record


@pytest.mark.parametrize(
    ("records", "rotation", "expected"),
    [
        (
            [_candidate(C_gain=-1.0, C_rank=-1.0, cross_role_directionally_stable=0)],
            False,
            "conditional_direction_absent",
        ),
        (
            [_candidate(cross_role_directionally_stable=0)],
            False,
            "direction_present_but_role_unstable",
        ),
        (
            [_candidate(gain_directional_screen_passed=0)],
            False,
            "phase_midpoint_cancellation",
        ),
        (
            [_candidate(I_rank_at_lambda_gain=0.0)],
            False,
            "gain_transfer_failure",
        ),
        ([_candidate()], True, "optimization_time_rotation"),
        (
            [_candidate(fixed_design_margin=0.0)],
            False,
            "strictly_positive_but_too_small",
        ),
    ],
)
def test_each_nonexclusive_mechanism_flag(
    records: list[dict[str, object]], rotation: bool, expected: str
) -> None:
    result = classify_mechanism_flags(
        records, optimization_time_rotation=rotation
    )
    assert result[expected] == 1
    assert result["authorizing"] == 0


def test_power_only_requires_two_of_the_three_frozen_seeds() -> None:
    records = [
        {
            "seed": seed,
            "cross_role_directionally_stable": int(seed != 3),
            "n_rounded": 384 if seed != 3 else 32,
        }
        for seed in (1, 2, 3)
    ]
    assert classify_power_only_evidence(records)["power_only_evidence"] == 1
    records[1]["n_rounded"] = 416
    assert classify_power_only_evidence(records)["power_only_evidence"] == 0
