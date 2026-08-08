from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from mnist.d0_jacobi_rb_boundary_tangent_cache import (
    SELECTED_OUTER_STEPS,
    midpoint_sample_key,
)
from mnist.d0_jacobi_rb_boundary_tangent_quartile_selection import (
    CONFIRMATION_NAMESPACE,
    DEFAULT_CONFIRMATION_SEED,
    DEFAULT_REPLICATES,
    DEFAULT_SELECTION_SEED,
    DEFAULT_SHARD_SIZE,
    LOCAL_FAMILY_NAMES,
    LOCAL_FAMILY_NAMES_SHA256,
    PRIMARY_FAMILY_NAMES,
    PRIMARY_FAMILY_NAMES_SHA256,
    PRODUCTION_PATH_COUNT,
    Q1_SENTINEL_MIDPOINT,
    Q1_SENTINEL_PHASE,
    SELECTION_NAMESPACE,
    LocalCompatibilityScreen,
    QuartileAuditPathTable,
    QuartileMaxTResult,
    QuartileSelectionError,
    aggregate_quartile_audit_improvements,
    aggregate_quartile_audit_risks,
    bootstrap_plan,
    compute_bootstrap_maxima_shard,
    confirmation_record,
    count_shard_index_record,
    evaluate_local_compatibility_screen,
    expected_audit_sample_key_sha256,
    generate_bootstrap_count_shard,
    load_bootstrap_count_shards,
    prepare_bootstrap_count_shards,
    quartile_max_t,
    restartable_quartile_max_t,
    selection_record,
)


def test_exact_six_family_and_224_local_family_order_and_hashes() -> None:
    assert PRIMARY_FAMILY_NAMES == (
        "specialist_vs_zero.q0.pooled",
        "specialist_vs_zero.q1.pooled",
        "specialist_vs_zero.q2.pooled",
        "specialist_vs_zero.q3.pooled",
        "shrunken_vs_raw.q2.pooled",
        "shrunken_vs_raw.q3.pooled",
    )
    assert (
        PRIMARY_FAMILY_NAMES_SHA256
        == "af508d27ef89501bd04800a455ab672e78e270ddff159db71cb531e445a4671e"
    )
    assert len(LOCAL_FAMILY_NAMES) == 224
    assert LOCAL_FAMILY_NAMES[0] == "specialist_vs_zero.q0.phase0.midpoint0"
    assert LOCAL_FAMILY_NAMES[55] == "specialist_vs_zero.q0.phase6.midpoint7"
    assert LOCAL_FAMILY_NAMES[56] == "specialist_vs_zero.q1.phase0.midpoint0"
    assert LOCAL_FAMILY_NAMES[-1] == "specialist_vs_zero.q3.phase6.midpoint7"
    assert (
        LOCAL_FAMILY_NAMES_SHA256
        == "4c8bec71884e2707fdf5fb34c9eeb4c807972251916c9d9323827b81709c4a86"
    )
    assert PRODUCTION_PATH_COUNT == 384
    assert DEFAULT_REPLICATES == 50_000
    assert DEFAULT_SHARD_SIZE == 1_000
    assert DEFAULT_SELECTION_SEED == 261_350
    assert DEFAULT_CONFIRMATION_SEED == 261_351
    assert SELECTION_NAMESPACE == 0x51545331
    assert CONFIRMATION_NAMESPACE == 0x51544331


def _row_fixture(
    *,
    path_count: int = 8,
    steps: tuple[int, ...] = (15, 143, 271, 399),
) -> dict[str, np.ndarray]:
    rows = [
        (path, step, phase, midpoint)
        for path in range(0xF5000, 0xF5000 + path_count)
        for step in steps
        for phase in range(7)
        for midpoint in range(8)
    ]
    identity = np.asarray(rows, dtype=np.int64)
    paths, outer_steps, phases, midpoints = identity.T
    keys = np.fromiter(
        (
            midpoint_sample_key(path, step, phase, midpoint)
            for path, step, phase, midpoint in rows
        ),
        dtype=np.int64,
        count=len(rows),
    )
    zero = (
        0.1
        + (paths - 0xF5000) * 1.0e-3
        + (outer_steps // 128) * 1.0e-2
        + phases * 1.0e-4
        + midpoints * 1.0e-5
    ).astype(np.float64)
    shrink = (0.05 + zero * 0.25).astype(np.float64)
    return {
        "sample_keys": keys,
        "row_path_ids": paths,
        "outer_steps": outer_steps,
        "phases": phases,
        "midpoint_indices": midpoints,
        "specialist_vs_zero_improvements": zero,
        "shrunken_vs_raw_improvements": shrink,
        "expected_path_ids": np.arange(
            0xF5000, 0xF5000 + path_count, dtype=np.int64
        ),
    }


def test_aggregate_224_cells_and_six_primary_contrasts_exactly() -> None:
    fixture = _row_fixture()
    table = aggregate_quartile_audit_improvements(
        **fixture,
        selected_outer_steps=(15, 143, 271, 399),
        expected_path_count=None,
    )
    assert isinstance(table, QuartileAuditPathTable)
    assert table.primary_values.shape == (8, 6)
    assert table.local_values.shape == (8, 4, 56)
    assert np.all(table.local_counts == 1)
    assert np.all(table.primary_counts[:, :4] == 56)
    assert np.all(table.primary_counts[:, 4:] == 56)
    zero = fixture["specialist_vs_zero_improvements"].reshape(8, 4, 7, 8)
    shrink = fixture["shrunken_vs_raw_improvements"].reshape(8, 4, 7, 8)
    assert np.array_equal(table.local_values, zero.reshape(8, 4, 56))
    assert np.allclose(table.primary_values[:, :4], zero.mean(axis=(2, 3)), atol=2e-16)
    assert np.allclose(table.primary_values[:, 4], shrink[:, 2].mean(axis=(1, 2)), atol=2e-16)
    assert np.allclose(table.primary_values[:, 5], shrink[:, 3].mean(axis=(1, 2)), atol=2e-16)
    assert table.to_record()["production_path_count_match"] == 0

    permutation = np.random.default_rng(91).permutation(fixture["sample_keys"].size)
    reordered = {
        name: value[permutation] if np.asarray(value).shape == permutation.shape else value
        for name, value in fixture.items()
    }
    permuted = aggregate_quartile_audit_improvements(
        **reordered,
        selected_outer_steps=(15, 143, 271, 399),
        expected_path_count=None,
    )
    assert np.array_equal(table.primary_values, permuted.primary_values)
    assert np.array_equal(table.local_values, permuted.local_values)

    duplicate = dict(fixture)
    duplicate["sample_keys"] = np.array(fixture["sample_keys"], copy=True)
    duplicate["sample_keys"][1] = duplicate["sample_keys"][0]
    with pytest.raises(QuartileSelectionError, match="not unique"):
        aggregate_quartile_audit_improvements(
            **duplicate,
            selected_outer_steps=(15, 143, 271, 399),
            expected_path_count=None,
        )


def test_path_table_rejects_noncartesian_counts_and_row_count() -> None:
    table, _ = _manual_evidence()
    bad_counts = np.array(table.primary_counts, copy=True)
    bad_counts[0, 0] -= 1
    with pytest.raises(QuartileSelectionError, match="Cartesian counts"):
        QuartileAuditPathTable(
            path_ids=table.path_ids,
            primary_values=table.primary_values,
            local_values=table.local_values,
            primary_counts=bad_counts,
            local_counts=table.local_counts,
            selected_outer_steps=table.selected_outer_steps,
            sample_key_sha256=table.sample_key_sha256,
            row_count=table.row_count,
        )
    with pytest.raises(QuartileSelectionError, match="Cartesian counts"):
        QuartileAuditPathTable(
            path_ids=table.path_ids,
            primary_values=table.primary_values,
            local_values=table.local_values,
            primary_counts=table.primary_counts,
            local_counts=table.local_counts,
            selected_outer_steps=table.selected_outer_steps,
            sample_key_sha256=table.sample_key_sha256,
            row_count=table.row_count - 1,
        )


def test_direct_contrast_algebra_uses_raw_target_and_same_gain_one_checkpoint() -> None:
    fixture = _row_fixture()
    rows = fixture["sample_keys"].size
    target = np.linspace(-0.2, 0.4, rows * 392, dtype=np.float64).reshape(rows, 392)
    raw = np.linspace(0.3, -0.1, rows * 392, dtype=np.float64).reshape(rows, 392)
    gains = np.asarray([1.0, 1.0, 0.4, 0.7], dtype=np.float64)
    table = aggregate_quartile_audit_risks(
        sample_keys=fixture["sample_keys"],
        row_path_ids=fixture["row_path_ids"],
        outer_steps=fixture["outer_steps"],
        phases=fixture["phases"],
        midpoint_indices=fixture["midpoint_indices"],
        targets=target,
        raw_predictions=raw,
        gains=gains,
        expected_path_ids=fixture["expected_path_ids"],
        selected_outer_steps=(15, 143, 271, 399),
        expected_path_count=None,
    )
    q = fixture["outer_steps"] // 128
    final = raw * gains[q, None]
    direct_zero = np.mean(target * target - (target - final) ** 2, axis=1)
    direct_shrink = np.mean((target - raw) ** 2 - (target - final) ** 2, axis=1)
    replay = aggregate_quartile_audit_improvements(
        sample_keys=fixture["sample_keys"],
        row_path_ids=fixture["row_path_ids"],
        outer_steps=fixture["outer_steps"],
        phases=fixture["phases"],
        midpoint_indices=fixture["midpoint_indices"],
        specialist_vs_zero_improvements=np.ascontiguousarray(direct_zero),
        shrunken_vs_raw_improvements=np.ascontiguousarray(direct_shrink),
        expected_path_ids=fixture["expected_path_ids"],
        selected_outer_steps=(15, 143, 271, 399),
        expected_path_count=None,
    )
    assert np.array_equal(table.primary_values, replay.primary_values)
    assert np.array_equal(table.local_values, replay.local_values)
    with pytest.raises(QuartileSelectionError, match="gains"):
        aggregate_quartile_audit_risks(
            sample_keys=fixture["sample_keys"],
            row_path_ids=fixture["row_path_ids"],
            outer_steps=fixture["outer_steps"],
            phases=fixture["phases"],
            midpoint_indices=fixture["midpoint_indices"],
            targets=target,
            raw_predictions=raw,
            gains=np.asarray([1.0, 0.9, 0.4, 0.7], dtype=np.float64),
            expected_path_ids=fixture["expected_path_ids"],
            selected_outer_steps=(15, 143, 271, 399),
            expected_path_count=None,
        )


def test_local_screen_enforces_phase_midpoint_51_of_56_and_q1_sentinel() -> None:
    values = np.ones((8, 4, 56), dtype=np.float64)
    values[:, 0, :5] = -0.01
    passed = evaluate_local_compatibility_screen(values)
    assert isinstance(passed, LocalCompatibilityScreen)
    assert passed.positive_cell_counts.tolist() == [51, 56, 56, 56]
    assert passed.passed
    assert passed.to_record()["inferential_claim_made"] == 0

    six_negative = np.array(values, copy=True)
    six_negative[:, 0, 5] = -0.01
    assert not evaluate_local_compatibility_screen(six_negative).quartile_passed[0]

    sentinel = np.ones((8, 4, 56), dtype=np.float64)
    sentinel[:, 1, Q1_SENTINEL_PHASE * 8 + Q1_SENTINEL_MIDPOINT] = 0.0
    assert not evaluate_local_compatibility_screen(sentinel).quartile_passed[1]

    phase_fail = np.ones((8, 4, 56), dtype=np.float64)
    phase_fail[:, 2, :8] = -1.0
    assert not evaluate_local_compatibility_screen(phase_fail).quartile_passed[2]

    midpoint_fail = np.ones((8, 4, 56), dtype=np.float64)
    midpoint_fail[:, 3, np.arange(7) * 8 + 2] = -1.0
    assert not evaluate_local_compatibility_screen(midpoint_fail).quartile_passed[3]


def test_uint16_384_path_philox_count_fixture_and_plan() -> None:
    counts = generate_bootstrap_count_shard(
        seed=DEFAULT_SELECTION_SEED,
        namespace=SELECTION_NAMESPACE,
        shard_index=0,
    )
    assert counts.dtype == np.uint16
    assert counts.shape == (1_000, 384)
    assert np.all(counts.sum(axis=1, dtype=np.int64) == 384)
    assert (
        hashlib.sha256(counts.tobytes(order="C")).hexdigest()
        == "7103510961f9291e18bf4f3e3ed1db887241771a207320c9152ed0515332a7af"
    )
    confirmation = generate_bootstrap_count_shard(
        seed=DEFAULT_CONFIRMATION_SEED,
        namespace=CONFIRMATION_NAMESPACE,
        shard_index=0,
    )
    assert not np.array_equal(counts, confirmation)
    plan = bootstrap_plan(
        seed=DEFAULT_SELECTION_SEED,
        namespace=SELECTION_NAMESPACE,
    )
    assert plan["path_count"] == 384
    assert plan["replicates"] == 50_000
    assert plan["shard_count"] == 50
    assert plan["count_dtype"] == np.dtype(np.uint16).str
    assert plan["quantile_method"] == "higher"
    assert plan["standard_error_floor_used"] == 0


def test_centered_studentized_max_t_matches_naive_resampling_and_higher_quantile() -> None:
    path_count = 16
    path_ids = np.arange(0xF5000, 0xF5000 + path_count, dtype=np.int64)
    base = np.arange(path_count, dtype=np.float64)
    values = np.stack(
        [base + 0.125 * component + 0.01 * base**2 for component in range(6)],
        axis=1,
    )
    counts = generate_bootstrap_count_shard(
        seed=17,
        namespace=18,
        shard_index=0,
        path_count=path_count,
        shard_size=21,
    )
    actual = compute_bootstrap_maxima_shard(values, counts, path_ids=path_ids)
    point = values.mean(axis=0)
    expected = []
    for multiplicities in counts:
        repeated = np.repeat(
            np.arange(path_count), multiplicities.astype(np.int64)
        )
        draw = values[repeated]
        draw_mean = draw.mean(axis=0)
        draw_se = draw.std(axis=0, ddof=1) / np.sqrt(path_count)
        expected.append(np.max((draw_mean - point) / draw_se))
    assert np.allclose(actual, np.asarray(expected, dtype=np.float64), atol=2e-14)
    result = quartile_max_t(
        values,
        path_ids=path_ids,
        count_shards=[counts],
        confidence=0.9,
    )
    assert result.critical_value == np.quantile(actual, 0.9, method="higher")
    assert np.array_equal(
        result.lower_bounds,
        result.point_estimates - result.critical_value * result.standard_errors,
    )
    permutation = np.random.default_rng(19).permutation(path_count)
    permuted = quartile_max_t(
        values[permutation],
        path_ids=path_ids[permutation],
        count_shards=[counts],
        confidence=0.9,
    )
    assert np.array_equal(result.maxima, permuted.maxima)
    assert np.array_equal(result.lower_bounds, permuted.lower_bounds)


def test_no_standard_error_floor_and_degenerate_bootstrap_fail_closed() -> None:
    path_ids = np.arange(8, dtype=np.int64)
    counts = generate_bootstrap_count_shard(
        seed=1,
        namespace=2,
        shard_index=0,
        path_count=8,
        shard_size=8,
    )
    with pytest.raises(QuartileSelectionError, match="observed studentization") as exc:
        quartile_max_t(
            np.ones((8, 6), dtype=np.float64),
            path_ids=path_ids,
            count_shards=[counts],
        )
    assert exc.value.failure_code == "quartile_max_t_studentization_invalid"

    values = np.random.default_rng(2).normal(size=(8, 6)).astype(np.float64)
    degenerate = np.zeros((1, 8), dtype=np.uint16)
    degenerate[0, 0] = 8
    with pytest.raises(QuartileSelectionError, match="bootstrap produced") as exc:
        compute_bootstrap_maxima_shard(values, degenerate, path_ids=path_ids)
    assert exc.value.failure_code == "quartile_max_t_bootstrap_studentization_invalid"
    with pytest.raises(QuartileSelectionError, match="malformed"):
        compute_bootstrap_maxima_shard(
            values, degenerate.astype(np.uint8), path_ids=path_ids
        )


def _manual_evidence(
    *, mean: float = 0.2, path_start: int = 0xF5000
) -> tuple[
    QuartileAuditPathTable,
    LocalCompatibilityScreen,
]:
    path_ids = np.arange(path_start, path_start + 384, dtype=np.int64)
    centered = np.linspace(-0.01, 0.01, 384, dtype=np.float64)
    primary = mean + centered[:, None] * (
        1.0 + np.arange(6, dtype=np.float64)[None, :] / 10.0
    )
    local = np.broadcast_to(primary[:, :4, None], (384, 4, 56)).copy()
    table = QuartileAuditPathTable(
        path_ids=path_ids,
        primary_values=np.ascontiguousarray(primary),
        local_values=local,
        primary_counts=np.full((384, 6), 8 * 56, dtype=np.int64),
        local_counts=np.full((384, 4, 56), 8, dtype=np.int64),
        selected_outer_steps=np.asarray(SELECTED_OUTER_STEPS, dtype=np.int64),
        sample_key_sha256=expected_audit_sample_key_sha256(
            path_ids, selected_outer_steps=SELECTED_OUTER_STEPS
        ),
        row_count=384 * 32 * 56,
    )
    return table, evaluate_local_compatibility_screen(local)


def test_selection_and_confirmation_records_require_both_six_family_and_screen(
    tmp_path: Path,
) -> None:
    table, screen = _manual_evidence()
    counts = prepare_bootstrap_count_shards(
        tmp_path / "selection-counts",
        seed=DEFAULT_SELECTION_SEED,
        namespace=SELECTION_NAMESPACE,
    )
    result, _, maxima_records = restartable_quartile_max_t(
        table.primary_values,
        path_ids=table.path_ids,
        count_directory=tmp_path / "selection-counts",
        maxima_directory=tmp_path / "selection-maxima",
        seed=DEFAULT_SELECTION_SEED,
        namespace=SELECTION_NAMESPACE,
    )
    selected = selection_record(
        result,
        screen,
        path_table=table,
        count_records=counts,
        maxima_records=maxima_records,
    )
    assert selected["passed"] == 1
    assert selected["confirmation_authorized"] == 1
    assert selected["decision"] == "quartile_specialist_selection_passed"
    assert selected["sampling_authorized"] == 0

    confirmation_table, confirmation_screen = _manual_evidence(path_start=0xF7000)
    confirmation_counts = prepare_bootstrap_count_shards(
        tmp_path / "confirmation-counts",
        seed=DEFAULT_CONFIRMATION_SEED,
        namespace=CONFIRMATION_NAMESPACE,
    )
    confirmation_result, _, confirmation_maxima = restartable_quartile_max_t(
        confirmation_table.primary_values,
        path_ids=confirmation_table.path_ids,
        count_directory=tmp_path / "confirmation-counts",
        maxima_directory=tmp_path / "confirmation-maxima",
        seed=DEFAULT_CONFIRMATION_SEED,
        namespace=CONFIRMATION_NAMESPACE,
    )
    confirmed = confirmation_record(
        confirmation_result,
        confirmation_screen,
        path_table=confirmation_table,
        count_records=confirmation_counts,
        maxima_records=confirmation_maxima,
    )
    assert confirmed["passed"] == 1
    assert (
        confirmed["decision"]
        == "exact_rb_quartile_specialist_time_local_signal_confirmed"
    )
    assert confirmed["reverse_controller_control_planning_authorized"] == 1
    assert confirmed["controller_execution_authorized"] == 0
    assert confirmed["confirmation_reuse_authorized"] == 0
    wrong_path_result, _, wrong_path_maxima = restartable_quartile_max_t(
        table.primary_values,
        path_ids=table.path_ids,
        count_directory=tmp_path / "confirmation-counts",
        maxima_directory=tmp_path / "wrong-confirmation-path-maxima",
        seed=DEFAULT_CONFIRMATION_SEED,
        namespace=CONFIRMATION_NAMESPACE,
    )
    with pytest.raises(QuartileSelectionError, match="384-path/50000-draw plan"):
        confirmation_record(
            wrong_path_result,
            screen,
            path_table=table,
            count_records=confirmation_counts,
            maxima_records=wrong_path_maxima,
        )

    negative_table, negative_screen = _manual_evidence(mean=-0.2)
    negative_result, negative_counts, negative_maxima = restartable_quartile_max_t(
        negative_table.primary_values,
        path_ids=negative_table.path_ids,
        count_directory=tmp_path / "selection-counts",
        maxima_directory=tmp_path / "negative-maxima",
        seed=DEFAULT_SELECTION_SEED,
        namespace=SELECTION_NAMESPACE,
    )
    failed_inference = selection_record(
        negative_result,
        negative_screen,
        path_table=negative_table,
        count_records=negative_counts,
        maxima_records=negative_maxima,
    )
    assert failed_inference["passed"] == 0
    assert failed_inference["confirmation_authorized"] == 0
    bad_local = np.array(table.local_values, copy=True)
    bad_local[:, 1, Q1_SENTINEL_PHASE * 8 + Q1_SENTINEL_MIDPOINT] = 0.0
    bad_primary = np.array(table.primary_values, copy=True)
    bad_primary[:, :4] = bad_local.mean(axis=2)
    bad_table = QuartileAuditPathTable(
        path_ids=table.path_ids,
        primary_values=bad_primary,
        local_values=bad_local,
        primary_counts=table.primary_counts,
        local_counts=table.local_counts,
        selected_outer_steps=table.selected_outer_steps,
        sample_key_sha256=table.sample_key_sha256,
        row_count=table.row_count,
    )
    bad_result, _, bad_maxima = restartable_quartile_max_t(
        bad_table.primary_values,
        path_ids=bad_table.path_ids,
        count_directory=tmp_path / "selection-counts",
        maxima_directory=tmp_path / "bad-local-maxima",
        seed=DEFAULT_SELECTION_SEED,
        namespace=SELECTION_NAMESPACE,
    )
    failed_screen = selection_record(
        bad_result,
        evaluate_local_compatibility_screen(bad_table.local_values),
        path_table=bad_table,
        count_records=counts,
        maxima_records=bad_maxima,
    )
    assert failed_screen["passed"] == 0
    assert failed_screen["decision"] == "no_fresh_quartile_specialist_system"

    test_only = selection_record(
        QuartileMaxTResult(
            path_ids=np.arange(8, dtype=np.int64),
            point_estimates=np.ones(6, dtype=np.float64),
            standard_errors=np.ones(6, dtype=np.float64),
            lower_bounds=np.full(6, -1.0, dtype=np.float64),
            maxima=np.asarray([2.0], dtype=np.float64),
            critical_value=2.0,
            confidence=0.995,
        ),
        evaluate_local_compatibility_screen(np.ones((8, 4, 56), dtype=np.float64)),
        authorizing=False,
    )
    assert test_only["passed"] == 0
    assert test_only["decision"] == "selection_test_only_nonauthorizing"

    # Authorizing records re-open and verify their committed bootstrap data.
    maxima_path = Path(str(maxima_records[0]["artifact_path"]))
    maxima_path.write_bytes(b"tampered maxima")
    with pytest.raises(QuartileSelectionError, match="maxima artifact changed"):
        selection_record(
            result,
            screen,
            path_table=table,
            count_records=counts,
            maxima_records=maxima_records,
        )


def test_restartable_uint16_count_and_maxima_shards_are_exact(tmp_path: Path) -> None:
    path_count = 16
    paths = np.arange(0xF5000, 0xF5000 + path_count, dtype=np.int64)
    values = np.random.default_rng(71).normal(
        0.1, 0.02, size=(path_count, 6)
    ).astype(np.float64)
    count_dir = tmp_path / "counts"
    maxima_dir = tmp_path / "maxima"
    records = prepare_bootstrap_count_shards(
        count_dir,
        seed=81,
        namespace=82,
        path_count=path_count,
        replicates=32,
        shard_size=16,
    )
    assert all(record["dtype"] == np.dtype(np.uint16).str for record in records)
    count_hashes = [record["artifact_sha256"] for record in records]
    first, loaded, maxima_records = restartable_quartile_max_t(
        values,
        path_ids=paths,
        count_directory=count_dir,
        maxima_directory=maxima_dir,
        seed=81,
        namespace=82,
        confidence=0.9,
        replicates=32,
        shard_size=16,
    )
    assert count_hashes == [record["artifact_sha256"] for record in loaded]
    assert len(maxima_records) == 2
    # A corrupted derived payload is reconstructed from sealed counts and
    # committed evidence without changing the result.
    corrupt_payload = maxima_dir / "shard-00000.npz"
    corrupt_payload.write_bytes(b"corrupt derived maxima")
    recovered, _, _ = restartable_quartile_max_t(
        values,
        path_ids=paths,
        count_directory=count_dir,
        maxima_directory=maxima_dir,
        seed=81,
        namespace=82,
        confidence=0.9,
        replicates=32,
        shard_size=16,
    )
    assert np.array_equal(first.maxima, recovered.maxima)
    (maxima_dir / "shard-00001.metadata.json").unlink()
    (maxima_dir / "shard-00001.npz").unlink()
    second, loaded_again, _ = restartable_quartile_max_t(
        values,
        path_ids=paths,
        count_directory=count_dir,
        maxima_directory=maxima_dir,
        seed=81,
        namespace=82,
        confidence=0.9,
        replicates=32,
        shard_size=16,
    )
    assert count_hashes == [record["artifact_sha256"] for record in loaded_again]
    assert np.array_equal(first.maxima, second.maxima)
    assert np.array_equal(first.lower_bounds, second.lower_bounds)

    # Prospective counts are immutable once the preflight seal has closed.
    (count_dir / "shard-00001.metadata.json").unlink()
    with pytest.raises(QuartileSelectionError, match="missing"):
        load_bootstrap_count_shards(
            count_dir,
            seed=81,
            namespace=82,
            path_count=path_count,
            replicates=32,
            shard_size=16,
        )


def test_count_shard_index_binds_all_metadata_and_rejects_tamper(tmp_path: Path) -> None:
    records = prepare_bootstrap_count_shards(
        tmp_path / "selection",
        seed=DEFAULT_SELECTION_SEED,
        namespace=SELECTION_NAMESPACE,
        path_count=16,
        replicates=32,
        shard_size=16,
    )
    index = count_shard_index_record(records, role="selection")
    assert index["sealed_before_physical_labels"] == 0
    assert index["authorizing_count_plan"] == 0
    assert index["count_dtype"] == np.dtype(np.uint16).str
    assert index["replicate_count"] == 32
    with pytest.raises(QuartileSelectionError, match="nonproduction"):
        count_shard_index_record(records, role="selection", authorizing=True)
    changed = [dict(row) for row in records]
    changed[0]["artifact_sha256"] = "0" * 64
    with pytest.raises(QuartileSelectionError, match="semantic hash"):
        count_shard_index_record(changed, role="selection")
    Path(str(records[0]["artifact_path"])).write_bytes(b"tampered counts")
    with pytest.raises(QuartileSelectionError, match="artifact changed"):
        count_shard_index_record(records, role="selection")
