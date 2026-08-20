from __future__ import annotations

import math

import numpy as np
import pytest

from mnist.d0_jacobi_rb_boundary_tangent_quartile_specialist import (
    CHECKPOINT_UPDATES,
)
from mnist.d0_jacobi_rb_quartile_directional_adjudication_inference import (
    COMPONENTS,
    FAMILY_NAMES,
    FAMILY_NAMES_SHA256,
    FAMILY_SIZE,
    STREAM_IDENTITIES,
    DirectionalInferenceError,
    StreamIdentity,
    adjudicate_seed_streams,
    build_stream_path_table,
    direction_effect_family_names,
    forecast_required_paths,
    local_compatibility_screen,
    one_sided_direction_effect_max_t,
    path_stability_summary,
    positive_ray_moment,
    quadratic_improvement,
    select_direction_nominees,
    trajectory_rotation_diagnostics,
)
from mnist.d0_jacobi_artifacts import config_fingerprint


def _candidate_grid(*, ties: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stream in STREAM_IDENTITIES:
        for update in CHECKPOINT_UPDATES:
            cross = -1.0 if stream.component == "spatial_cnn" else update / 4_000
            if ties and stream.component != "spatial_cnn" and update in (100, 200):
                cross = 2.0
            rows.append(
                {
                    **stream.to_record(),
                    "update": update,
                    "target_energy": 4.0,
                    "cross_term": cross,
                    "prediction_energy": 1.0,
                }
            )
    return rows


def _stream_records(*, path_count: int = 8) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    direction = np.linspace(0.5, 1.5, path_count, dtype=np.float64)
    effect = np.linspace(0.25, 1.25, path_count, dtype=np.float64)
    for stream in STREAM_IDENTITIES:
        records.append(
            {
                **stream.to_record(),
                "nominee_present": 1,
                "C_gain": 1.0,
                "C_rank": 1.0,
                "direction_path_values": direction.copy(),
                "effect_path_values": effect.copy(),
                "direction_cells": np.ones((7, 8), dtype=np.float64),
                "effect_cells": np.ones((7, 8), dtype=np.float64),
                "effect_point": 1.0,
            }
        )
    return records


def test_frozen_family_has_canonical_quartile_component_seed_statistic_order() -> None:
    assert len(STREAM_IDENTITIES) == 36
    assert len(FAMILY_NAMES) == FAMILY_SIZE == 72
    assert FAMILY_NAMES[:6] == (
        "q0.full.seed261332.direction",
        "q0.full.seed261332.effect",
        "q0.full.seed261333.direction",
        "q0.full.seed261333.effect",
        "q0.full.seed261334.direction",
        "q0.full.seed261334.effect",
    )
    assert FAMILY_NAMES[-1] == "q3.spatial_cnn.seed261343.effect"
    assert direction_effect_family_names() == FAMILY_NAMES
    assert FAMILY_NAMES_SHA256 == config_fingerprint(list(FAMILY_NAMES))


def test_positive_ray_moment_and_quadratic_identity_are_exact() -> None:
    moment = positive_ray_moment(9.0, 2.0, 4.0)
    assert moment.rho == pytest.approx(1.0 / 3.0)
    assert moment.lambda_plus == 0.5
    assert moment.directional_ceiling == 1.0
    assert quadratic_improvement(2.0, 4.0, moment.lambda_plus) == 1.0

    nonpositive = positive_ray_moment(9.0, -2.0, 4.0)
    assert nonpositive.lambda_plus == 0.0
    assert nonpositive.directional_ceiling == 0.0
    assert not nonpositive.positive_direction

    zero = positive_ray_moment(0.0, 0.0, 0.0)
    assert zero.rho == zero.lambda_plus == zero.directional_ceiling == 0.0

    with pytest.raises(DirectionalInferenceError, match="P=0"):
        positive_ray_moment(1.0, 1.0, 0.0)
    with pytest.raises(DirectionalInferenceError, match="cannot be negative"):
        positive_ray_moment(1.0, 0.0, -1.0)


def test_gain_nomination_uses_only_d_plus_and_exact_earlier_tie_break() -> None:
    nominees = select_direction_nominees(_candidate_grid(ties=True))
    assert len(nominees) == 36
    full = nominees[0]
    assert full["status"] == "gain_direction_nominee"
    assert full["update"] == 100
    assert full["D_plus_gain"] == 4.0
    assert full["rank_evidence_used"] == 0

    spatial = next(row for row in nominees if row["component"] == "spatial_cnn")
    assert spatial["status"] == "no_positive_gain_direction"
    assert spatial["update"] is None
    assert spatial["lambda_gain"] is None

    with pytest.raises(DirectionalInferenceError, match="incomplete"):
        select_direction_nominees(_candidate_grid()[:-1])


def test_local_screen_retains_every_original_point_rule_and_q1_sentinel() -> None:
    values = np.ones((7, 8), dtype=np.float64)
    passed = local_compatibility_screen(values, quartile=1)
    assert passed["passed"] == 1
    assert passed["positive_fine_cell_count"] == 56

    values[4, 7] = -0.1
    q1 = local_compatibility_screen(values, quartile=1)
    assert q1["positive_fine_cell_count"] == 55
    assert q1["q1_sentinel_passed"] == 0
    assert q1["passed"] == 0
    assert local_compatibility_screen(values, quartile=2)["passed"] == 1

    values = np.ones((7, 8), dtype=np.float64)
    values.flat[:6] = -0.01
    assert local_compatibility_screen(values, quartile=3)["positive_fine_cell_count"] == 50
    assert local_compatibility_screen(values, quartile=3)["passed"] == 0


def test_max_t_matches_direct_philox_resampling_and_higher_quantile() -> None:
    path_ids = np.arange(8, dtype=np.int64) + 100
    base = np.arange(8, dtype=np.float64) - 3.5
    values = np.column_stack(
        [base * (1.0 + index / 100.0) + index for index in range(FAMILY_SIZE)]
    ).astype(np.float64)
    values[:, 1] = 2.5  # exact constant: handled analytically
    result = one_sided_direction_effect_max_t(
        values,
        path_ids=path_ids,
        confidence=0.9,
        replicates=200,
        seed=17,
        namespace=29,
        chunk_size=37,
    )

    generator = np.random.Generator(np.random.Philox([17, 29]))
    maxima = []
    point = np.mean(values, axis=0, dtype=np.float64)
    for _ in range(200):
        indices = generator.integers(0, 8, size=(8,), dtype=np.int64)
        draw = values[indices]
        draw_mean = np.mean(draw, axis=0, dtype=np.float64)
        draw_error = np.std(draw[:, result.standard_errors > 0.0], axis=0, ddof=1) / math.sqrt(8)
        statistic = (
            draw_mean[result.standard_errors > 0.0]
            - point[result.standard_errors > 0.0]
        ) / draw_error
        maxima.append(max(0.0, float(np.max(statistic))))
    expected_critical = float(np.quantile(maxima, 0.9, method="higher"))
    assert np.array_equal(result.maxima, np.asarray(maxima, dtype=np.float64))
    assert result.critical_value == expected_critical
    assert result.analytic_constant_mask[1]
    assert result.standard_errors[1] == 0.0
    assert result.lower_bounds[1] == 2.5
    assert result.to_record()["standard_error_floor_used"] == 0


def test_max_t_is_path_order_and_chunk_invariant_and_family_locked() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(size=(12, FAMILY_SIZE)).astype(np.float64)
    path_ids = np.arange(12, dtype=np.int64) + 200
    first = one_sided_direction_effect_max_t(
        values,
        path_ids=path_ids,
        replicates=300,
        seed=7,
        chunk_size=17,
    )
    permutation = np.asarray([3, 0, 11, 2, 5, 1, 10, 4, 6, 9, 7, 8])
    replay = one_sided_direction_effect_max_t(
        values[permutation],
        path_ids=path_ids[permutation],
        replicates=300,
        seed=7,
        chunk_size=113,
    )
    assert np.array_equal(first.maxima, replay.maxima)
    assert np.array_equal(first.lower_bounds, replay.lower_bounds)
    with pytest.raises(DirectionalInferenceError, match="family order"):
        one_sided_direction_effect_max_t(
            values,
            path_ids=path_ids,
            family_names=FAMILY_NAMES[::-1],
            replicates=10,
        )


def test_all_constant_family_is_valid_and_uses_analytic_bounds() -> None:
    values = np.broadcast_to(
        np.arange(FAMILY_SIZE, dtype=np.float64), (8, FAMILY_SIZE)
    ).copy()
    result = one_sided_direction_effect_max_t(
        values,
        path_ids=np.arange(8, dtype=np.int64),
        replicates=20,
    )
    assert result.critical_value == 0.0
    assert np.all(result.analytic_constant_mask)
    assert np.array_equal(result.lower_bounds, values[0])


def test_stream_table_retains_no_nominee_as_fixed_failing_effect() -> None:
    records = _stream_records()
    records[0]["nominee_present"] = 0
    records[0]["effect_path_values"] = np.zeros(8, dtype=np.float64)
    table = build_stream_path_table(records, path_ids=np.arange(8, dtype=np.int64))
    assert table.shape == (8, 72)
    assert np.array_equal(table[:, 1], np.zeros(8))
    records[0]["effect_path_values"][0] = 1.0
    with pytest.raises(DirectionalInferenceError, match="no-nominee"):
        build_stream_path_table(records, path_ids=np.arange(8, dtype=np.int64))


def test_stream_adjudication_applies_two_of_three_and_full_local_geometry() -> None:
    records = _stream_records()
    table = build_stream_path_table(records, path_ids=np.arange(8, dtype=np.int64))
    inference = one_sided_direction_effect_max_t(
        table,
        path_ids=np.arange(8, dtype=np.int64),
        replicates=200,
        seed=101,
    )
    result = adjudicate_seed_streams(records, inference)
    assert len(result["seed_rows"]) == 36
    assert len(result["component_rows"]) == 12
    assert all(row["stable_direction"] == 1 for row in result["component_rows"])
    assert all(row["stable_effect"] == 1 for row in result["component_rows"])

    # One failed seed does not defeat a component; two failed seeds do.
    q1_full = [
        row
        for row in records
        if row["quartile"] == 1 and row["component"] == "full"
    ]
    q1_full[0]["direction_cells"] = np.full((7, 8), -1.0)
    q1_full[1]["direction_cells"] = np.full((7, 8), -1.0)
    changed = adjudicate_seed_streams(records, inference)
    summary = next(
        row
        for row in changed["component_rows"]
        if row["quartile"] == 1 and row["component"] == "full"
    )
    assert summary["direction_passing_seed_count"] == 1
    assert summary["stable_direction"] == 0


def test_path_stability_and_power_forecast_distinguish_instability_and_negative_effect() -> None:
    values = np.asarray([3.0, 3.0, -1.0, -1.0], dtype=np.float64)
    summary = path_stability_summary(values, simultaneous_lower_bound=-0.1)
    assert summary["mean"] == 1.0
    assert summary["positive_path_count"] == 2
    assert summary["sign_entropy_bits"] == 1.0
    assert summary["path_unstable"] == 1

    forecast = forecast_required_paths(
        values,
        critical_value=2.0,
        local_point_screen_passed=True,
    )
    expected_raw = math.ceil((2.0 * np.std(values, ddof=1) / np.mean(values)) ** 2)
    assert forecast["n_required"] == expected_raw
    assert forecast["n_required_rounded"] % 32 == 0

    negative = forecast_required_paths(
        -values,
        critical_value=2.0,
        local_point_screen_passed=True,
    )
    assert negative["n_required"] == "not_finite"
    incompatible = forecast_required_paths(
        values,
        critical_value=2.0,
        local_point_screen_passed=False,
    )
    assert incompatible["n_required"] == "not_finite"


def test_trajectory_rotation_detects_only_preregistered_exact_events() -> None:
    updates = np.asarray([100, 200, 300], dtype=np.int64)
    predictions = np.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]], dtype=np.float64
    )
    gain_profile = np.arange(56, dtype=np.float64).reshape(7, 8)
    rank_profile = -gain_profile
    result = trajectory_rotation_diagnostics(
        updates=updates,
        prediction_vectors=predictions,
        pooled_cross_terms=np.asarray([1.0, -1.0, 1.0]),
        gain_cell_profile=gain_profile,
        rank_cell_profile=rank_profile,
        gain_nominee_rank_cross_term=0.0,
    )
    assert result["negative_adjacent_prediction_cosine"] == 1
    assert result["pooled_cross_term_changes_sign_at_least_twice"] == 1
    assert result["gain_nominee_rank_cross_term_nonpositive"] == 1
    assert result["gain_rank_cell_profile_correlation_nonpositive"] == 1
    assert result["rotation_event_count"] == 4
    assert result["optimization_time_rotation"] == 1


def test_stream_identity_rejects_seed_component_mismatch() -> None:
    assert StreamIdentity(0, COMPONENTS[0], 261_332).key == "q0.full.seed261332"
    with pytest.raises(DirectionalInferenceError, match="another quartile"):
        StreamIdentity(0, "full", 261_335)
