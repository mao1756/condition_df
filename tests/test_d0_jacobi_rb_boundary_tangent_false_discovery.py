from __future__ import annotations

import copy
import random

import numpy as np
import pytest

from mnist.d0_jacobi_rb_boundary_tangent_cache import midpoint_sample_key
from mnist.d0_jacobi_rb_boundary_tangent_false_discovery import (
    BASELINE_FAMILY_NAMES,
    BASELINE_FAMILY_SIZE,
    HISTORICAL_SEEDS,
    HISTORICAL_UPDATES,
    BaselineRiskTable,
    CandidateAuditError,
    HistoricalSelectionReplayError,
    MaxTInferenceError,
    ThreeContrastEvidenceError,
    aggregate_validated_three_contrasts,
    build_candidate_validation_table,
    classify_candidate_audit,
    classify_sealed_baseline,
    corrected_point_candidate_eligible,
    replay_historical_selection,
    require_exact_confirmation_replay,
    search_aware_candidate_max_t,
    two_sided_baseline_max_abs_t,
    validate_three_contrast_rows,
)
from mnist.d0_jacobi_rb_boundary_tangent_gate import (
    COMBINED_VS_ZERO_FAMILY_SIZE,
    CONFIRMATION_FAMILY_SIZE,
)


def _three_contrast_fixture() -> dict[str, object]:
    paths = np.arange(0xED000, 0xED008, dtype=np.int64)
    steps = (15, 143, 271, 399)
    identity = np.asarray(
        [
            (path, step, phase, midpoint)
            for path in paths.tolist()
            for step in steps
            for phase in range(7)
            for midpoint in range(8)
        ],
        dtype=np.int64,
    )
    row = np.arange(identity.shape[0], dtype=np.float64)
    baseline = np.ascontiguousarray(0.02 + row * 1.0e-8)
    residual = np.ascontiguousarray(-0.001 + (row % 17.0) * 1.0e-9)
    zero = np.ascontiguousarray(baseline + residual)
    return {
        "sample_keys": np.asarray(
            [midpoint_sample_key(*value) for value in identity], dtype=np.int64
        ),
        "row_path_ids": identity[:, 0],
        "outer_steps": identity[:, 1],
        "phases": identity[:, 2],
        "midpoint_indices": identity[:, 3],
        "combined_vs_zero": zero,
        "combined_vs_baseline": residual,
        "baseline_vs_zero": baseline,
        "expected_path_ids": paths,
        "selected_outer_steps": steps,
    }


def test_three_contrast_replay_builds_exact_228_and_229_tables_order_invariant() -> None:
    fixture = _three_contrast_fixture()
    first_rows = validate_three_contrast_rows(**fixture)
    first = aggregate_validated_three_contrasts(first_rows)
    assert first.confirmation.path_values.shape == (8, 228)
    assert first.baseline.path_values.shape == (8, 229)
    assert first.baseline.cell_counts[:, :224].min() == 1
    assert first.baseline.cell_counts[:, 224:228].min() == 56
    assert first.baseline.cell_counts[:, 228].min() == 224
    assert (
        first.baseline.sample_key_sha256
        == first.confirmation.sample_key_sha256
    )
    assert first.baseline.to_record()["controller_planning_authorized"] == 0

    permutation = np.random.default_rng(17).permutation(first_rows.row_count)
    shuffled = {
        name: (
            np.asarray(value)[permutation]
            if name not in {"expected_path_ids", "selected_outer_steps"}
            else value
        )
        for name, value in fixture.items()
    }
    second = aggregate_validated_three_contrasts(
        validate_three_contrast_rows(**shuffled)
    )
    assert np.array_equal(first.confirmation.path_values, second.confirmation.path_values)
    assert np.array_equal(first.baseline.path_values, second.baseline.path_values)
    assert first.baseline.sample_key_sha256 == second.baseline.sample_key_sha256
    require_exact_confirmation_replay(
        second.confirmation,
        parent_path_ids=first.confirmation.path_ids,
        parent_path_values=first.confirmation.path_values,
        parent_cell_counts=first.confirmation.cell_counts,
    )


def test_three_contrast_identity_above_five_e_minus_fifteen_is_fatal() -> None:
    fixture = _three_contrast_fixture()
    zero = np.array(fixture["combined_vs_zero"], copy=True)
    zero[0] += 5.1e-15
    with pytest.raises(
        ThreeContrastEvidenceError, match="three-contrast identity"
    ) as raised:
        validate_three_contrast_rows(**{**fixture, "combined_vs_zero": zero})
    assert raised.value.failure_code == "sealed_three_contrast_identity_invalid"


def test_three_contrast_rejects_missing_identity_and_parent_replay_difference() -> None:
    fixture = _three_contrast_fixture()
    with pytest.raises(ThreeContrastEvidenceError, match="row count"):
        validate_three_contrast_rows(
            **{
                name: (
                    value[:-1]
                    if isinstance(value, np.ndarray) and name != "expected_path_ids"
                    else value
                )
                for name, value in fixture.items()
            }
        )
    tables = aggregate_validated_three_contrasts(validate_three_contrast_rows(**fixture))
    changed = np.array(tables.confirmation.path_values, copy=True)
    changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
    with pytest.raises(ThreeContrastEvidenceError, match="differs from parent"):
        require_exact_confirmation_replay(
            tables.confirmation,
            parent_path_ids=tables.confirmation.path_ids,
            parent_path_values=changed,
        )


def _baseline_table(center: np.ndarray) -> BaselineRiskTable:
    paths = np.arange(0xED000, 0xED010, dtype=np.int64)
    path_noise = np.linspace(-0.01, 0.01, paths.size, dtype=np.float64)[:, None]
    feature_noise = np.linspace(-0.002, 0.002, BASELINE_FAMILY_SIZE)[None, :]
    values = np.ascontiguousarray(center[None, :] + path_noise + feature_noise)
    return BaselineRiskTable(
        path_ids=paths,
        path_values=values,
        cell_counts=np.ones_like(values, dtype=np.int64),
        sample_key_sha256="a" * 64,
    )


def test_two_sided_baseline_classifies_advantage_harm_and_mixed() -> None:
    positive = _baseline_table(np.full(BASELINE_FAMILY_SIZE, 0.2))
    positive_record = two_sided_baseline_max_abs_t(
        positive, confidence=0.9, replicates=64, seed=11, chunk_size=7
    )
    assert classify_sealed_baseline(positive_record) == "sealed_baseline_advantage_confirmed"

    harmful = _baseline_table(np.full(BASELINE_FAMILY_SIZE, -0.2))
    harmful_record = two_sided_baseline_max_abs_t(
        harmful, confidence=0.9, replicates=64, seed=11, chunk_size=9
    )
    assert classify_sealed_baseline(harmful_record) == "sealed_baseline_harm_confirmed"

    mixed_center = np.zeros(BASELINE_FAMILY_SIZE, dtype=np.float64)
    mixed_center[:224] = 0.2
    mixed_center[224:] = np.asarray([0.2, -0.2, 0.2, -0.2, 0.0])
    mixed_record = two_sided_baseline_max_abs_t(
        _baseline_table(mixed_center), confidence=0.9, replicates=64, seed=11
    )
    assert classify_sealed_baseline(mixed_record) == "sealed_baseline_not_established"
    assert positive_record["posthoc_non_authorizing"] == 1
    assert positive_record["old_confirmation_paths_burned"] == 1
    assert positive_record["controller_planning_authorized"] == 0


def test_two_sided_baseline_degenerate_family_fails_closed() -> None:
    paths = np.arange(8, dtype=np.int64)
    values = np.ones((8, BASELINE_FAMILY_SIZE), dtype=np.float64)
    table = BaselineRiskTable(
        path_ids=paths,
        path_values=values,
        cell_counts=np.ones_like(values, dtype=np.int64),
        sample_key_sha256="b" * 64,
    )
    with pytest.raises(MaxTInferenceError, match="degenerate"):
        two_sided_baseline_max_abs_t(table, replicates=8)
    assert classify_sealed_baseline(None) == "sealed_baseline_evidence_invalid"


def _candidate_fixture(*, strong_residual: bool = False):
    seeds = np.repeat(np.asarray(HISTORICAL_SEEDS, dtype=np.int64), 40)
    updates = np.tile(np.arange(100, 4_001, 100, dtype=np.int64), 3)
    paths = np.arange(0xEC200, 0xEC210, dtype=np.int64)
    values = np.empty((120, paths.size, CONFIRMATION_FAMILY_SIZE), dtype=np.float64)
    path_noise = np.linspace(-0.02, 0.02, paths.size, dtype=np.float64)
    component_noise = np.linspace(-0.001, 0.001, CONFIRMATION_FAMILY_SIZE)
    for candidate in range(120):
        values[candidate] = (
            -0.2
            + 0.01 * path_noise[:, None]
            + component_noise[None, :]
            + candidate * 1.0e-7
        )
    selected = int(
        np.flatnonzero((seeds == 261_314) & (updates == 800))[0]
    )
    residual_center = 0.5 if strong_residual else 1.0e-5
    values[selected, :, 224:] = residual_center + path_noise[:, None]
    # Even a strongly resolved residual remains incompatible with zero.
    values[selected, :, :224] = -0.5 + 0.01 * path_noise[:, None]
    return build_candidate_validation_table(
        seeds=seeds,
        updates=updates,
        path_ids=paths,
        path_values=np.ascontiguousarray(values),
        forbidden_path_ids=np.arange(0xED000, 0xED040, dtype=np.int64),
    )


def test_unique_tiny_positive_candidate_fails_search_aware_resolution() -> None:
    table = _candidate_fixture()
    record = search_aware_candidate_max_t(
        table,
        confidence=0.9,
        replicates=96,
        seed=19,
        chunk_size=13,
        component_block_size=37,
    )
    selected = next(
        row
        for row in record["candidate_rows"]
        if row["seed"] == 261_314 and row["update"] == 800
    )
    assert selected["selection_resolved_residual_signal"] == 0
    assert selected["all_228_point_estimates_positive"] == 0
    assert record["residual_resolved_candidate_count"] == 0
    assert record["qualifying_candidate_count"] == 0
    assert record["selected_update_residual_resolved"] == 0
    assert classify_candidate_audit(record) == "selected_update_below_resolution"
    assert record["controller_planning_authorized"] == 0


def test_resolved_residual_that_loses_to_zero_is_directionally_incompatible() -> None:
    record = search_aware_candidate_max_t(
        _candidate_fixture(strong_residual=True),
        confidence=0.9,
        replicates=64,
        seed=23,
    )
    assert record["selection_resolved_candidate_count"] >= 1
    assert record["fully_qualified_candidate_count"] == 0
    assert (
        classify_candidate_audit(record)
        == "residual_signal_directionally_incompatible_with_zero"
    )


def test_candidate_search_is_candidate_path_and_block_order_invariant() -> None:
    table = _candidate_fixture()
    first = search_aware_candidate_max_t(
        table,
        confidence=0.9,
        replicates=64,
        seed=29,
        chunk_size=8,
        component_block_size=41,
    )
    candidate_permutation = np.random.default_rng(7).permutation(120)
    path_permutation = np.random.default_rng(8).permutation(table.path_count)
    reordered = build_candidate_validation_table(
        seeds=table.seeds[candidate_permutation],
        updates=table.updates[candidate_permutation],
        path_ids=table.path_ids[path_permutation],
        path_values=table.path_values[candidate_permutation][:, path_permutation],
    )
    second = search_aware_candidate_max_t(
        reordered,
        confidence=0.9,
        replicates=64,
        seed=29,
        chunk_size=16,
        component_block_size=73,
    )
    assert first["critical_value"] == second["critical_value"]
    assert first["lower_bounds"] == second["lower_bounds"]
    assert first["candidate_rows"] == second["candidate_rows"]


def test_confirmation_paths_cannot_enter_candidate_audit() -> None:
    table = _candidate_fixture()
    with pytest.raises(CandidateAuditError, match="confirmation path IDs") as raised:
        build_candidate_validation_table(
            seeds=table.seeds,
            updates=table.updates,
            path_ids=table.path_ids,
            path_values=table.path_values,
            forbidden_path_ids=np.asarray([table.path_ids[0]], dtype=np.int64),
        )
    assert raised.value.failure_code == "confirmation_path_firewall_violated"


def _historical_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for seed in HISTORICAL_SEEDS:
        for update in HISTORICAL_UPDATES:
            baseline = 5.0
            baseline_high = 5.0
            zero = 4.0
            zero_high = 4.0
            validation = 5.0 + update * 1.0e-8
            high = 5.0 + update * 2.0e-8
            if seed == 261_314 and update == 700:
                validation = high = 4.95
            if seed == 261_314 and update == 800:
                validation = high = 4.90
            eligible = int(update > 0 and validation < baseline and high < baseline_high)
            records.append(
                {
                    "seed": seed,
                    "update": update,
                    "finite": 1,
                    "validation_mse": validation,
                    "validation_high_reverse_time_mse": high,
                    "baseline_validation_mse": baseline,
                    "baseline_high_reverse_time_mse": baseline_high,
                    "zero_validation_mse": zero,
                    "zero_high_reverse_time_mse": zero_high,
                    "combined_vs_baseline": baseline - validation,
                    "combined_vs_baseline_high_reverse_time": baseline_high - high,
                    "combined_vs_zero": zero - validation,
                    "combined_vs_zero_high_reverse_time": zero_high - high,
                    "eligible_nonzero": eligible,
                    "state_sha256": f"{seed + update:064x}"[-64:],
                    "checkpoint_path": (
                        f"checkpoints/physical/seed-{seed}/update-{update:04d}.pt"
                    ),
                    "checkpoint_file_sha256": f"{2 * seed + update:064x}"[-64:],
                }
            )
    return records


def test_historical_selection_replay_is_exact_and_order_invariant() -> None:
    records = _historical_records()
    first = replay_historical_selection(records)
    shuffled = copy.deepcopy(records)
    random.Random(31).shuffle(shuffled)
    second = replay_historical_selection(shuffled)
    assert first == second
    assert first["candidate_count"] == 123
    assert first["selected_seed"] == 261_314
    assert first["selected_update"] == 800
    assert first["historical_selection_reproduced"] == 1
    assert first["controller_planning_authorized"] == 0


def test_historical_selection_missing_record_or_bad_metric_fails_closed() -> None:
    records = _historical_records()
    with pytest.raises(HistoricalSelectionReplayError, match="incomplete"):
        replay_historical_selection(records[:-1])
    changed = copy.deepcopy(records)
    changed[17]["combined_vs_baseline"] = 1.0
    with pytest.raises(HistoricalSelectionReplayError, match="does not replay"):
        replay_historical_selection(changed)


def test_candidate_beating_baseline_but_losing_zero_is_not_corrected_eligible() -> None:
    record = {
        "update": 800,
        "finite": 1,
        "combined_vs_baseline": 1.0e-6,
        "combined_vs_baseline_high_reverse_time": 2.0e-6,
        "combined_vs_zero": -1.0e-2,
        "combined_vs_zero_high_reverse_time": -2.0e-2,
    }
    assert corrected_point_candidate_eligible(record) is False
    record["combined_vs_zero"] = 1.0e-6
    record["combined_vs_zero_high_reverse_time"] = 1.0e-6
    assert corrected_point_candidate_eligible(record) is True


def test_baseline_family_order_is_frozen() -> None:
    assert len(BASELINE_FAMILY_NAMES) == 229
    assert BASELINE_FAMILY_NAMES[0] == "baseline_vs_zero.q0.phase0.midpoint0"
    assert BASELINE_FAMILY_NAMES[223] == "baseline_vs_zero.q3.phase6.midpoint7"
    assert BASELINE_FAMILY_NAMES[224:228] == tuple(
        f"baseline_vs_zero.q{q}" for q in range(4)
    )
    assert BASELINE_FAMILY_NAMES[-1] == "baseline_vs_zero.overall"
