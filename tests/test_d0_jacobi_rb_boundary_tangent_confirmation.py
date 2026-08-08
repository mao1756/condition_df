from __future__ import annotations

import json

import numpy as np
import pytest

from mnist.d0_jacobi_rb_boundary_tangent_cache import midpoint_sample_key
from mnist.d0_jacobi_rb_boundary_tangent_confirmation import (
    BoundaryTangentConfirmationError,
    aggregate_confirmation_improvements,
    aggregate_confirmation_risks,
    normalized_controller_trajectory_max_t,
)
from mnist.d0_jacobi_rb_boundary_tangent_gate import (
    BoundaryTangentThresholds,
    COMBINED_VS_ZERO_FAMILY_SIZE,
    CONFIRMATION_FAMILY_NAMES,
    CONFIRMATION_FAMILY_SIZE,
    CONTROLLER_FAMILY_SIZE,
    validate_controller_family,
)


def _fixture() -> dict[str, np.ndarray | tuple[int, ...]]:
    path_ids = tuple(range(0xED000, 0xED008))
    selected_steps = (15, 143, 271, 399)
    identity = np.asarray(
        [
            (path, step, phase, midpoint)
            for path in path_ids
            for step in selected_steps
            for phase in range(7)
            for midpoint in range(8)
        ],
        dtype=np.int64,
    )
    rows = identity.shape[0]
    keys = np.asarray(
        [midpoint_sample_key(*row) for row in identity], dtype=np.int64
    )
    return {
        "sample_keys": keys,
        "row_path_ids": identity[:, 0],
        "outer_steps": identity[:, 1],
        "phases": identity[:, 2],
        "midpoint_indices": identity[:, 3],
        "targets": np.full((rows, 392), 2.0, dtype=np.float64),
        "combined_predictions": np.full(
            (rows, 392), 1.0, dtype=np.float64
        ),
        "baseline_predictions": np.full(
            (rows, 392), 0.5, dtype=np.float64
        ),
        "expected_path_ids": np.asarray(path_ids, dtype=np.int64),
        "selected_outer_steps": selected_steps,
    }


def test_confirmation_aggregation_builds_exact_228_whole_path_family() -> None:
    result = aggregate_confirmation_risks(**_fixture())
    assert result.path_values.shape == (8, CONFIRMATION_FAMILY_SIZE)
    assert np.all(result.path_values[:, :COMBINED_VS_ZERO_FAMILY_SIZE] == 3.0)
    assert np.all(result.path_values[:, COMBINED_VS_ZERO_FAMILY_SIZE:] == 1.25)
    assert np.all(result.cell_counts[:, :COMBINED_VS_ZERO_FAMILY_SIZE] == 1)
    assert np.all(result.cell_counts[:, COMBINED_VS_ZERO_FAMILY_SIZE:] == 56)
    record = result.to_record()
    assert tuple(record["family_names"]) == CONFIRMATION_FAMILY_NAMES
    assert record["family_size"] == 228
    assert record["bootstrap_unit"] == "whole_path"
    assert record["negative_values_truncated"] == 0
    assert "data_end" not in json.dumps(record)


def test_confirmation_aggregation_is_input_order_invariant() -> None:
    fixture = _fixture()
    first = aggregate_confirmation_risks(**fixture)
    permutation = np.random.default_rng(91).permutation(
        np.asarray(fixture["sample_keys"]).size
    )
    shuffled = {
        name: (
            np.asarray(value)[permutation]
            if name
            not in {"expected_path_ids", "selected_outer_steps"}
            else value
        )
        for name, value in fixture.items()
    }
    second = aggregate_confirmation_risks(**shuffled)
    assert np.array_equal(first.path_ids, second.path_ids)
    assert np.array_equal(first.path_values, second.path_values)
    assert np.array_equal(first.cell_counts, second.cell_counts)
    assert first.sample_key_sha256 == second.sample_key_sha256


def test_streamed_improvement_aggregation_matches_raw_risk_exactly() -> None:
    fixture = _fixture()
    target = np.asarray(fixture["targets"])
    combined = np.asarray(fixture["combined_predictions"])
    baseline = np.asarray(fixture["baseline_predictions"])
    zero_improvement = np.mean(
        target * target - (target - combined) ** 2,
        axis=1,
        dtype=np.float64,
    )
    baseline_improvement = np.mean(
        (target - baseline) ** 2 - (target - combined) ** 2,
        axis=1,
        dtype=np.float64,
    )
    streamed = aggregate_confirmation_improvements(
        sample_keys=fixture["sample_keys"],
        row_path_ids=fixture["row_path_ids"],
        outer_steps=fixture["outer_steps"],
        phases=fixture["phases"],
        midpoint_indices=fixture["midpoint_indices"],
        combined_vs_zero_improvements=np.ascontiguousarray(zero_improvement),
        combined_vs_baseline_improvements=np.ascontiguousarray(
            baseline_improvement
        ),
        expected_path_ids=fixture["expected_path_ids"],
        selected_outer_steps=fixture["selected_outer_steps"],
    )
    raw = aggregate_confirmation_risks(**fixture)
    assert np.array_equal(streamed.path_ids, raw.path_ids)
    assert np.array_equal(streamed.path_values, raw.path_values)
    assert np.array_equal(streamed.cell_counts, raw.cell_counts)
    assert streamed.to_record() == raw.to_record()


def test_streamed_improvements_require_finite_binary64_vectors() -> None:
    fixture = _fixture()
    rows = np.asarray(fixture["sample_keys"]).size
    kwargs = {
        "sample_keys": fixture["sample_keys"],
        "row_path_ids": fixture["row_path_ids"],
        "outer_steps": fixture["outer_steps"],
        "phases": fixture["phases"],
        "midpoint_indices": fixture["midpoint_indices"],
        "combined_vs_zero_improvements": np.ones(rows, dtype=np.float32),
        "combined_vs_baseline_improvements": np.ones(rows, dtype=np.float64),
        "expected_path_ids": fixture["expected_path_ids"],
        "selected_outer_steps": fixture["selected_outer_steps"],
    }
    with pytest.raises(BoundaryTangentConfirmationError, match="binary64"):
        aggregate_confirmation_improvements(**kwargs)
    kwargs["combined_vs_zero_improvements"] = np.ones(rows, dtype=np.float64)
    kwargs["combined_vs_baseline_improvements"] = np.ones(
        (rows, 1), dtype=np.float64
    )
    with pytest.raises(BoundaryTangentConfirmationError, match="finite"):
        aggregate_confirmation_improvements(**kwargs)


def test_confirmation_aggregation_rejects_repeated_or_wrong_identity() -> None:
    fixture = _fixture()
    duplicate = {
        name: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        for name, value in fixture.items()
    }
    for name in (
        "sample_keys",
        "row_path_ids",
        "outer_steps",
        "phases",
        "midpoint_indices",
        "targets",
        "combined_predictions",
        "baseline_predictions",
    ):
        duplicate[name][-1] = duplicate[name][0]
    with pytest.raises(BoundaryTangentConfirmationError, match="not unique"):
        aggregate_confirmation_risks(**duplicate)

    wrong_key = _fixture()
    wrong_key["sample_keys"][17] += 1 << 45
    with pytest.raises(BoundaryTangentConfirmationError, match="do not match"):
        aggregate_confirmation_risks(**wrong_key)


def test_confirmation_aggregation_rejects_missing_quartile_and_nonbinary64() -> None:
    fixture = _fixture()
    with pytest.raises(BoundaryTangentConfirmationError, match="every forward quartile"):
        aggregate_confirmation_risks(
            **{**fixture, "selected_outer_steps": (15, 31, 47, 63)}
        )
    float32_target = _fixture()
    float32_target["targets"] = np.asarray(
        float32_target["targets"], dtype=np.float32
    )
    with pytest.raises(BoundaryTangentConfirmationError, match="binary64"):
        aggregate_confirmation_risks(**float32_target)


def test_confirmation_baseline_contrast_is_pooled_only_within_quartile() -> None:
    fixture = _fixture()
    baseline = np.asarray(fixture["baseline_predictions"])
    quartile = np.asarray(fixture["outer_steps"]) // 128
    # R(B)-R(g): with target=2, g=1, B=0 gives 3; with B=2 gives -1.
    baseline[quartile == 0] = 0.0
    baseline[quartile == 1] = 2.0
    result = aggregate_confirmation_risks(**fixture)
    pooled = result.path_values[:, COMBINED_VS_ZERO_FAMILY_SIZE:]
    assert np.all(pooled[:, 0] == 3.0)
    assert np.all(pooled[:, 1] == -1.0)
    assert np.all(pooled[:, 2:] == 1.25)


def _controller_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    paths = np.arange(0xED100, 0xED108, dtype=np.int64)
    row = np.arange(paths.size, dtype=np.float64)[:, None]
    column = np.arange(CONTROLLER_FAMILY_SIZE, dtype=np.float64)[None, :]
    numerator = np.ascontiguousarray(
        0.01 * (row - 3.5) + 1.0e-5 * column, dtype=np.float64
    )
    denominator = np.ascontiguousarray(
        0.5 + 0.03 * row + 1.0e-6 * column, dtype=np.float64
    )
    names = [f"controller.feature.{index:03d}.bias" for index in range(392)] + [
        f"controller.feature.{index:03d}.M8_vs_M4" for index in range(392)
    ]
    return paths, numerator, denominator, names


def test_controller_trajectory_max_t_is_exact_784_family_and_order_invariant() -> None:
    paths, numerator, denominator, names = _controller_fixture()
    first = normalized_controller_trajectory_max_t(
        numerators=numerator,
        forward_changes=denominator,
        path_ids=paths,
        names=names,
        confidence=0.9,
        replicates=32,
        seed=701,
        namespace=9,
        chunk_size=7,
    )
    permutation = np.asarray([5, 2, 7, 0, 6, 3, 1, 4], dtype=np.int64)
    second = normalized_controller_trajectory_max_t(
        numerators=numerator[permutation],
        forward_changes=denominator[permutation],
        path_ids=paths[permutation],
        names=names,
        confidence=0.9,
        replicates=32,
        seed=701,
        namespace=9,
        chunk_size=11,
    )
    assert first == second
    assert first["family_size"] == 784
    assert first["bootstrap_unit"] == "whole_path"
    assert first["path_ids"] == paths.tolist()
    assert first["denominator_recomputed_per_resample"] == 1
    expected = numerator.mean(axis=0) / np.sqrt(np.mean(denominator**2, axis=0))
    actual = np.asarray([first["point_estimates"][name] for name in names])
    assert np.array_equal(actual, expected)
    validation = validate_controller_family(
        first,
        thresholds=BoundaryTangentThresholds(
            controller_paths=8,
            bootstrap_replicates=32,
            simultaneous_confidence=0.9,
            controller_bootstrap_seed=701,
            maximum_weak_law_bias=10.0,
            maximum_microstep_refinement_error=10.0,
        ),
    )
    assert validation["controller_family_valid"] == 1


def test_controller_trajectory_max_t_rejects_nonbinary64_and_wrong_family() -> None:
    paths, numerator, denominator, names = _controller_fixture()
    kwargs = {
        "numerators": numerator,
        "forward_changes": denominator,
        "path_ids": paths,
        "names": names,
        "confidence": 0.9,
        "replicates": 8,
        "seed": 2,
    }
    with pytest.raises(BoundaryTangentConfirmationError, match="binary64"):
        normalized_controller_trajectory_max_t(
            **{**kwargs, "numerators": numerator.astype(np.float32)}
        )
    with pytest.raises(BoundaryTangentConfirmationError, match="784 unique"):
        normalized_controller_trajectory_max_t(**{**kwargs, "names": names[:-1]})
    with pytest.raises(BoundaryTangentConfirmationError, match="unique 20-bit"):
        normalized_controller_trajectory_max_t(
            **{**kwargs, "path_ids": np.zeros(paths.size, dtype=np.int64)}
        )


def test_controller_trajectory_max_t_rejects_degenerate_forward_change() -> None:
    paths, numerator, denominator, names = _controller_fixture()
    denominator[:, 17] = 0.0
    with pytest.raises(BoundaryTangentConfirmationError, match="RMS"):
        normalized_controller_trajectory_max_t(
            numerators=numerator,
            forward_changes=denominator,
            path_ids=paths,
            names=names,
            confidence=0.9,
            replicates=8,
            seed=3,
        )
