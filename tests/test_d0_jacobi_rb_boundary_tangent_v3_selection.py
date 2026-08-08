from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from mnist.d0_jacobi_rb_boundary_tangent_cache import (
    SELECTED_OUTER_STEPS,
    midpoint_sample_key,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_selection import (
    CONFIRMATION_NAMESPACE,
    DEFAULT_SELECTION_SEED,
    SELECTION_NAMESPACE,
    V3_CANDIDATE_COUNT,
    V3_COMPONENT_COUNT,
    V3_FAMILY_NAMES,
    V3_MODEL_SEEDS,
    V3_NONZERO_UPDATES,
    V3_SEARCH_FAMILY_SIZE,
    V3_SEARCH_FAMILY_NAMES,
    CandidateValidationTableV3,
    NumericMaxTResult,
    V3SelectionError,
    aggregate_zero_baseline_improvements,
    build_candidate_validation_table_v3,
    compute_bootstrap_maxima_shard,
    generate_bootstrap_count_shard,
    load_bootstrap_count_shards,
    numeric_v3_max_t,
    one_sided_v3_confirmation_max_t,
    prepare_bootstrap_count_shards,
    rank_validation_nominee,
    restartable_numeric_v3_max_t,
    search_aware_validation_max_t,
)


def _candidate_grid() -> tuple[np.ndarray, np.ndarray]:
    seeds = np.repeat(np.asarray(V3_MODEL_SEEDS, dtype=np.int64), 40)
    updates = np.tile(np.asarray(V3_NONZERO_UPDATES, dtype=np.int64), 3)
    return seeds, updates


def _candidate_table(seed: int = 19) -> CandidateValidationTableV3:
    seeds, updates = _candidate_grid()
    paths = np.arange(0xF1100, 0xF1120, dtype=np.int64)
    generator = np.random.default_rng(seed)
    path_values = generator.normal(
        loc=0.02,
        scale=0.004,
        size=(paths.size, V3_CANDIDATE_COUNT, V3_COMPONENT_COUNT),
    ).astype(np.float64)
    return build_candidate_validation_table_v3(
        seeds=seeds,
        updates=updates,
        path_ids=paths,
        path_values=np.ascontiguousarray(path_values),
        forbidden_path_ids=np.arange(0xF2000, 0xF2040, dtype=np.int64),
    )


def test_v3_family_names_have_frozen_224_plus_4_order() -> None:
    assert len(V3_FAMILY_NAMES) == V3_COMPONENT_COUNT == 228
    assert V3_SEARCH_FAMILY_SIZE == 120 * 228 == 27_360
    assert len(V3_SEARCH_FAMILY_NAMES) == V3_SEARCH_FAMILY_SIZE
    assert (
        V3_SEARCH_FAMILY_NAMES[0]
        == "seed261312.update0100.model_vs_zero.q0.phase0.midpoint0"
    )
    assert (
        V3_SEARCH_FAMILY_NAMES[-1]
        == "seed261314.update4000.model_vs_zero.q3.pooled"
    )
    assert V3_FAMILY_NAMES[0] == "model_vs_zero.q0.phase0.midpoint0"
    assert V3_FAMILY_NAMES[55] == "model_vs_zero.q0.phase6.midpoint7"
    assert V3_FAMILY_NAMES[56] == "model_vs_zero.q1.phase0.midpoint0"
    assert V3_FAMILY_NAMES[223] == "model_vs_zero.q3.phase6.midpoint7"
    assert V3_FAMILY_NAMES[224:] == (
        "model_vs_zero.q0.pooled",
        "model_vs_zero.q1.pooled",
        "model_vs_zero.q2.pooled",
        "model_vs_zero.q3.pooled",
    )


def _complete_row_fixture() -> dict[str, np.ndarray]:
    rows: list[tuple[int, int, int, int]] = []
    for path_id in range(0xF0000, 0xF0008):
        for outer_step in SELECTED_OUTER_STEPS:
            for phase in range(7):
                for midpoint in range(8):
                    rows.append((path_id, outer_step, phase, midpoint))
    identities = np.asarray(rows, dtype=np.int64)
    path_ids, steps, phases, midpoints = identities.T
    keys = np.asarray(
        [
            midpoint_sample_key(path, step, phase, midpoint)
            for path, step, phase, midpoint in rows
        ],
        dtype=np.int64,
    )
    improvement = (
        (path_ids - 0xF0000) * 0.01
        + (steps // 128) * 0.1
        + phases * 0.001
        + midpoints * 0.0001
        + (steps % 128) * 1.0e-7
    ).astype(np.float64)
    return {
        "sample_keys": keys,
        "row_path_ids": path_ids,
        "outer_steps": steps,
        "phases": phases,
        "midpoint_indices": midpoints,
        "model_vs_zero_improvements": improvement,
        "expected_path_ids": np.arange(0xF0000, 0xF0008, dtype=np.int64),
    }


def test_zero_baseline_aggregation_reuses_one_source_and_checks_pooled_cells() -> None:
    fixture = _complete_row_fixture()
    table = aggregate_zero_baseline_improvements(**fixture)
    assert table.path_values.shape == (8, 228)
    assert np.all(table.cell_counts[:, :224] == 8)
    assert np.all(table.cell_counts[:, 224:] == 8 * 7 * 8)
    for quartile in range(4):
        left = quartile * 56
        expected = np.average(
            table.path_values[:, left : left + 56],
            axis=1,
            weights=table.cell_counts[:, left : left + 56],
        )
        assert np.allclose(table.path_values[:, 224 + quartile], expected, atol=1e-15)
    record = table.to_record()
    assert record["family_names"] == list(V3_FAMILY_NAMES)
    assert record["baseline_contrast_present"] == 0

    changed = np.array(fixture["model_vs_zero_improvements"], copy=True)
    changed[0] += 1.0
    with pytest.raises(V3SelectionError, match="same binary64 source"):
        aggregate_zero_baseline_improvements(
            **fixture,
            pooled_model_vs_zero_improvements=changed,
        )


def test_candidate_table_canonicalizes_candidate_and_path_order_and_firewall() -> None:
    table = _candidate_table()
    candidate_order = np.random.default_rng(7).permutation(V3_CANDIDATE_COUNT)
    path_order = np.random.default_rng(8).permutation(table.path_count)
    reordered = build_candidate_validation_table_v3(
        seeds=table.seeds[candidate_order],
        updates=table.updates[candidate_order],
        path_ids=table.path_ids[path_order],
        path_values=table.path_values[path_order][:, candidate_order],
    )
    assert np.array_equal(reordered.seeds, table.seeds)
    assert np.array_equal(reordered.updates, table.updates)
    assert np.array_equal(reordered.path_ids, table.path_ids)
    assert np.array_equal(reordered.path_values, table.path_values)
    assert reordered.path_values.shape == (32, 120, 228)

    with pytest.raises(V3SelectionError, match="confirmation path IDs") as error:
        build_candidate_validation_table_v3(
            seeds=table.seeds,
            updates=table.updates,
            path_ids=table.path_ids,
            path_values=table.path_values,
            forbidden_path_ids=np.asarray([table.path_ids[0]], dtype=np.int64),
        )
    assert error.value.failure_code == "confirmation_path_firewall_violated"


def test_count_shard_has_frozen_philox_fixture_and_shared_whole_path_counts() -> None:
    counts = generate_bootstrap_count_shard(
        seed=DEFAULT_SELECTION_SEED,
        namespace=SELECTION_NAMESPACE,
        shard_index=0,
        path_count=32,
        shard_size=1_000,
    )
    assert counts.dtype == np.uint8
    assert counts.shape == (1_000, 32)
    assert np.all(counts.sum(axis=1, dtype=np.int64) == 32)
    assert (
        hashlib.sha256(counts.tobytes(order="C")).hexdigest()
        == "2e9305711a536c6070a3aa05817673e06ef4d13c30f070ff707db319dd13e105"
    )
    assert SELECTION_NAMESPACE != CONFIRMATION_NAMESPACE


def test_count_matrix_maxima_matches_naive_repeated_path_resampling() -> None:
    path_count = 8
    paths = np.arange(path_count, dtype=np.int64)
    base = np.arange(path_count, dtype=np.float64)
    values = np.empty((path_count, 2, 228), dtype=np.float64)
    for candidate in range(2):
        for component in range(228):
            values[:, candidate, component] = (
                base + 0.125 * candidate + 0.03125 * component
            )
    counts = generate_bootstrap_count_shard(
        seed=1,
        namespace=2,
        shard_index=0,
        path_count=path_count,
        shard_size=16,
    )
    actual = compute_bootstrap_maxima_shard(
        values,
        counts,
        path_ids=paths,
        candidate_block_size=1,
        component_block_size=57,
    )
    point = values.mean(axis=0)
    expected = np.empty(16, dtype=np.float64)
    for draw_index, multiplicities in enumerate(counts):
        repeated = np.repeat(
            np.arange(path_count, dtype=np.int64), multiplicities.astype(np.int64)
        )
        draw = values[repeated]
        mean = draw.mean(axis=0)
        second = np.mean(draw * draw, axis=0, dtype=np.float64)
        variance = path_count * (second - mean * mean) / (path_count - 1)
        error = np.sqrt(variance / path_count)
        expected[draw_index] = np.max((mean - point) / error)
    assert np.array_equal(actual, expected)


def test_numeric_core_is_block_and_input_order_invariant_and_confirmation_named() -> None:
    generator = np.random.default_rng(31)
    paths = np.arange(0xF1100, 0xF1110, dtype=np.int64)
    values = generator.normal(0.2, 0.03, size=(16, 2, 228)).astype(np.float64)
    counts = [
        generate_bootstrap_count_shard(
            seed=41,
            namespace=42,
            shard_index=index,
            path_count=16,
            shard_size=16,
        )
        for index in range(2)
    ]
    first = numeric_v3_max_t(
        values,
        path_ids=paths,
        count_shards=counts,
        confidence=0.9,
        candidate_block_size=1,
        component_block_size=57,
    )
    permutation = np.random.default_rng(43).permutation(paths.size)
    second = numeric_v3_max_t(
        values[permutation],
        path_ids=paths[permutation],
        count_shards=counts,
        confidence=0.9,
        candidate_block_size=2,
        component_block_size=19,
    )
    assert first.critical_value == second.critical_value
    assert np.array_equal(first.lower_bounds, second.lower_bounds)
    assert np.array_equal(first.maxima, second.maxima)

    confirmation = one_sided_v3_confirmation_max_t(
        values[:, 0],
        path_ids=paths,
        count_shards=counts,
        confidence=0.9,
    )
    assert confirmation["family_names"] == list(V3_FAMILY_NAMES)
    assert set(confirmation["lower_bounds"]) == set(V3_FAMILY_NAMES)
    assert confirmation["candidate_count"] == 1


def _ranking_result(
    table: CandidateValidationTableV3,
    lower: np.ndarray,
) -> NumericMaxTResult:
    return NumericMaxTResult(
        path_ids=table.path_ids,
        point_estimates=np.ones_like(lower),
        standard_errors=np.full_like(lower, 0.1),
        lower_bounds=np.ascontiguousarray(lower),
        maxima=np.asarray([1.0, 2.0], dtype=np.float64),
        critical_value=2.0,
        confidence=0.995,
    )


def test_nominee_requires_every_component_and_uses_frozen_tie_breaks() -> None:
    table = _candidate_table()
    lower = np.full((120, 228), -0.1, dtype=np.float64)
    no_candidate = rank_validation_nominee(table, _ranking_result(table, lower))
    assert no_candidate["decision"] == "no_validation_candidate"
    assert no_candidate["selected_update"] == 0
    assert no_candidate["confirmation_authorized"] == 0

    later = table.candidate_index(261_312, 200)
    earlier_high_seed = table.candidate_index(261_313, 100)
    earlier_low_seed = table.candidate_index(261_312, 100)
    lower[[later, earlier_high_seed, earlier_low_seed]] = 0.2
    # One failed fine cell makes an otherwise positive candidate ineligible.
    lower[later, 17] = -np.finfo(np.float64).eps
    selected = rank_validation_nominee(table, _ranking_result(table, lower))
    assert selected["selected_seed"] == 261_312
    assert selected["selected_update"] == 100
    assert selected["selected_minimum_lower_bound"] == 0.2
    assert selected["eligible_candidate_count"] == 2


def test_tiny_positive_candidate_fails_complete_search_adjustment() -> None:
    seeds, updates = _candidate_grid()
    paths = np.arange(0xF1100, 0xF1120, dtype=np.int64)
    path_noise = np.linspace(-0.01, 0.01, 32, dtype=np.float64)[:, None, None]
    component_scale = 1.0 + np.arange(228, dtype=np.float64)[None, None, :] / 1_000
    values = np.broadcast_to(
        -0.1 + path_noise * component_scale,
        (32, 120, 228),
    ).copy()
    values[:, 0, :] = (1.0e-6 + path_noise * component_scale)[:, 0, :]
    table = build_candidate_validation_table_v3(
        seeds=seeds,
        updates=updates,
        path_ids=paths,
        path_values=np.ascontiguousarray(values),
    )
    counts = [
        generate_bootstrap_count_shard(
            seed=3,
            namespace=4,
            shard_index=0,
            path_count=32,
            shard_size=32,
        )
    ]
    result, selection = search_aware_validation_max_t(
        table,
        count_shards=counts,
        confidence=0.9,
    )
    assert np.all(result.point_estimates[0] > 0.0)
    assert np.any(result.lower_bounds[0] <= 0.0)
    assert selection["decision"] == "no_validation_candidate"


def test_degenerate_observed_or_bootstrap_studentization_fails_without_floor() -> None:
    paths = np.arange(8, dtype=np.int64)
    counts = [
        generate_bootstrap_count_shard(
            seed=1,
            namespace=2,
            shard_index=0,
            path_count=8,
            shard_size=8,
        )
    ]
    with pytest.raises(V3SelectionError, match="observed studentization") as error:
        numeric_v3_max_t(
            np.ones((8, 1, 228), dtype=np.float64),
            path_ids=paths,
            count_shards=counts,
        )
    assert error.value.failure_code == "max_t_studentization_invalid"

    values = np.random.default_rng(4).normal(size=(8, 1, 228)).astype(np.float64)
    degenerate_counts = np.zeros((1, 8), dtype=np.uint8)
    degenerate_counts[0, 0] = 8
    with pytest.raises(V3SelectionError, match="bootstrap produced") as error:
        numeric_v3_max_t(
            values,
            path_ids=paths,
            count_shards=[degenerate_counts],
        )
    assert error.value.failure_code == "max_t_bootstrap_studentization_invalid"


def test_restartable_shards_reuse_counts_and_recompute_only_uncommitted_maximum(
    tmp_path: Path,
) -> None:
    paths = np.arange(0xF1100, 0xF1110, dtype=np.int64)
    values = np.random.default_rng(51).normal(
        0.1, 0.02, size=(16, 2, 228)
    ).astype(np.float64)
    environment = {
        "numpy_version": np.__version__,
        "philox_constructor": "fixture",
        "byte_order": "little",
        "cpu_blas_environment_sha256": "a" * 64,
    }
    count_dir = tmp_path / "counts"
    maxima_dir = tmp_path / "maxima"
    prepared = prepare_bootstrap_count_shards(
        count_dir,
        seed=61,
        namespace=62,
        path_count=16,
        replicates=32,
        shard_size=16,
        environment=environment,
    )
    count_hashes = [record["artifact_sha256"] for record in prepared]
    first, _, maxima_records = restartable_numeric_v3_max_t(
        values,
        path_ids=paths,
        count_directory=count_dir,
        maxima_directory=maxima_dir,
        seed=61,
        namespace=62,
        confidence=0.9,
        replicates=32,
        shard_size=16,
        environment=environment,
    )
    assert len(maxima_records) == 2
    (maxima_dir / "shard-00001.metadata.json").unlink()
    (maxima_dir / "shard-00001.npz").unlink()
    second, loaded_counts, _ = restartable_numeric_v3_max_t(
        values,
        path_ids=paths,
        count_directory=count_dir,
        maxima_directory=maxima_dir,
        seed=61,
        namespace=62,
        confidence=0.9,
        replicates=32,
        shard_size=16,
        environment=environment,
    )
    assert count_hashes == [record["artifact_sha256"] for record in loaded_counts]
    assert first.critical_value == second.critical_value
    assert np.array_equal(first.maxima, second.maxima)
    assert np.array_equal(first.lower_bounds, second.lower_bounds)

    # Once prospective generation is closed, a missing count is fatal.
    (count_dir / "shard-00001.metadata.json").unlink()
    with pytest.raises(V3SelectionError, match="missing"):
        load_bootstrap_count_shards(
            count_dir,
            seed=61,
            namespace=62,
            path_count=16,
            replicates=32,
            shard_size=16,
            environment=environment,
        )
