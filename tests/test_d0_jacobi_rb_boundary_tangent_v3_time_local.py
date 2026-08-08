from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from mnist.d0_jacobi_rb_boundary_tangent_cache import (
    SELECTED_OUTER_STEPS,
    midpoint_sample_key,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_selection import (
    V3_MODEL_SEEDS,
    V3_NONZERO_UPDATES,
    CandidateValidationTableV3,
    NumericMaxTResult,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_time_local import (
    PRODUCTION_CRITICAL_VALUE,
    PRODUCTION_DISCOVERED_COMPONENTS,
    Q0Nominee,
    TimeLocalAdjudicationError,
    advisory_scalar_calibration,
    aggregate_quadratic_risk_decomposition,
    build_resolution_ladder,
    build_sealed_selection_replay,
    classify_adjudication_decision,
    classify_quartile_signal,
    forecast_required_path_count,
    replay_sealed_selection,
)


def _candidate_identities() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.repeat(np.asarray(V3_MODEL_SEEDS, dtype=np.int64), 40),
        np.tile(np.asarray(V3_NONZERO_UPDATES, dtype=np.int64), 3),
    )


def _synthetic_evidence(seed: int = 17) -> tuple[CandidateValidationTableV3, NumericMaxTResult]:
    seeds, updates = _candidate_identities()
    paths = np.arange(0xF1000, 0xF1020, dtype=np.int64)
    generator = np.random.default_rng(seed)
    values = generator.normal(-0.002, 0.003, size=(32, 120, 228)).astype(np.float64)
    for model_seed in V3_MODEL_SEEDS:
        index = int(np.flatnonzero((seeds == model_seed) & (updates == 900))[0])
        values[:, index, :56] += 0.004
        values[:, index, 224] = np.mean(values[:, index, :56], axis=1)
    table = CandidateValidationTableV3(
        seeds=seeds,
        updates=updates,
        path_ids=paths,
        path_values=np.ascontiguousarray(values),
    )
    point = np.mean(table.path_values, axis=0, dtype=np.float64)
    error = np.std(table.path_values, axis=0, ddof=1, dtype=np.float64) / math.sqrt(32)
    maxima = np.full(100, 7.0, dtype=np.float64)
    lower = point - 7.0 * error
    result = NumericMaxTResult(
        path_ids=table.path_ids,
        point_estimates=np.ascontiguousarray(point),
        standard_errors=np.ascontiguousarray(error),
        lower_bounds=np.ascontiguousarray(lower),
        maxima=maxima,
        critical_value=7.0,
        confidence=0.995,
    )
    return table, result


def test_sealed_replay_recomputes_every_numeric_array_and_recovers_q0_nominees() -> None:
    table, result = _synthetic_evidence()
    replay = replay_sealed_selection(table, result)
    assert replay.table.path_values.shape == (32, 120, 228)
    assert [nominee.seed for nominee in replay.nominees] == list(V3_MODEL_SEEDS)
    assert [nominee.update for nominee in replay.nominees] == [900, 900, 900]
    assert replay.to_record()["logical_update_zero_selected"] == 1
    assert replay.to_record()["confirmation_authorized"] == 0

    changed = np.array(result.point_estimates, copy=True)
    changed[0, 0] = np.nextafter(changed[0, 0], math.inf)
    invalid = NumericMaxTResult(
        path_ids=result.path_ids,
        point_estimates=changed,
        standard_errors=result.standard_errors,
        lower_bounds=result.lower_bounds,
        maxima=result.maxima,
        critical_value=result.critical_value,
        confidence=result.confidence,
    )
    with pytest.raises(TimeLocalAdjudicationError, match="do not replay exactly"):
        replay_sealed_selection(table, invalid)


def test_build_replay_rejects_confirmation_path_ids() -> None:
    table, result = _synthetic_evidence()
    paths = np.array(table.path_ids, copy=True)
    paths[0] = 0xF2000
    with pytest.raises(TimeLocalAdjudicationError) as error:
        build_sealed_selection_replay(
            seeds=table.seeds,
            updates=table.updates,
            path_ids=paths,
            path_values=table.path_values,
            point_estimates=result.point_estimates,
            standard_errors=result.standard_errors,
            lower_bounds=result.lower_bounds,
            maxima=result.maxima,
        )
    assert error.value.failure_code == "confirmation_path_firewall_violated"


def test_resolution_ladder_has_frozen_levels_and_exact_pooling() -> None:
    seeds, updates = _candidate_identities()
    paths = np.arange(0xF1000, 0xF1020, dtype=np.int64)
    generator = np.random.default_rng(19)
    fine = generator.normal(size=(32, 120, 224)).astype(np.float64)
    quartile = fine.reshape(32, 120, 4, 7, 8).mean(axis=(3, 4))
    table = CandidateValidationTableV3(
        seeds=seeds,
        updates=updates,
        path_ids=paths,
        path_values=np.ascontiguousarray(np.concatenate((fine, quartile), axis=2)),
    )
    ladder = build_resolution_ladder(table, critical_value=3.0)
    assert [level.level for level in ladder] == [
        "quartile_phase_midpoint",
        "quartile_phase",
        "quartile_midpoint",
        "quartile",
        "phase",
        "midpoint",
        "overall",
    ]
    assert [level.path_values.shape[2] for level in ladder] == [224, 28, 32, 4, 7, 8, 1]
    assert np.array_equal(ladder[3].path_values, quartile)
    assert np.allclose(
        ladder[-1].path_values[:, :, 0], fine.mean(axis=2), rtol=0.0, atol=0.0
    )

    changed = np.array(table.path_values, copy=True)
    changed[0, 0, 224] += 1e-6
    malformed = CandidateValidationTableV3(
        seeds=seeds, updates=updates, path_ids=paths, path_values=changed
    )
    with pytest.raises(TimeLocalAdjudicationError, match="quartile pools"):
        build_resolution_ladder(malformed, critical_value=3.0)


def _complete_rows() -> dict[str, np.ndarray]:
    rows: list[tuple[int, int, int, int]] = []
    for path_id in range(0xF1000, 0xF1008):
        for outer_step in SELECTED_OUTER_STEPS:
            for phase in range(7):
                for midpoint in range(8):
                    rows.append((path_id, outer_step, phase, midpoint))
    identity = np.asarray(rows, dtype=np.int64)
    path_ids, steps, phases, midpoints = identity.T
    keys = np.asarray(
        [
            midpoint_sample_key(path, step, phase, midpoint)
            for path, step, phase, midpoint in rows
        ],
        dtype=np.int64,
    )
    return {
        "sample_keys": keys,
        "row_path_ids": path_ids,
        "outer_steps": steps,
        "phases": phases,
        "midpoint_indices": midpoints,
        "expected_path_ids": np.arange(0xF1000, 0xF1008, dtype=np.int64),
    }


def test_quadratic_risk_decomposition_preserves_exact_raw_target_identity() -> None:
    rows = _complete_rows()
    row_count = rows["sample_keys"].size
    generator = np.random.default_rng(23)
    targets = generator.normal(size=(row_count, 5)).astype(np.float64)
    predictions = np.stack((0.25 * targets, -0.1 * targets), axis=1)
    decomposition = aggregate_quadratic_risk_decomposition(
        **rows,
        targets=targets,
        predictions=np.ascontiguousarray(predictions),
        candidate_labels=("first", "second"),
    )
    assert decomposition.cross_terms.shape == (8, 2, 228)
    assert decomposition.maximum_identity_error <= 5e-15
    assert np.allclose(
        decomposition.direct_improvements,
        2.0 * decomposition.cross_terms - decomposition.prediction_energies,
        atol=5e-15,
        rtol=0.0,
    )
    first = advisory_scalar_calibration(
        float(decomposition.cross_terms.mean()),
        float(decomposition.prediction_energies.mean()),
    )
    assert first["authorizing"] == 0
    assert first["checkpoint_or_prediction_modified"] == 0


@pytest.mark.parametrize(
    ("cross", "energy", "lower", "expected"),
    [
        ((1.0, 1.0, 1.0), (0.5, 0.5, 0.5), (0.1, 0.1, 0.1), "resolved"),
        ((-0.1, 0.0, 0.1), (0.1, 0.1, 0.1), (-1.0, -1.0, -1.0), "directional_alignment_missing"),
        ((0.1, 0.1, 0.1), (0.3, 0.3, 0.3), (-1.0, -1.0, -1.0), "prediction_energy_dominates"),
        ((0.2, 0.2, 0.2), (0.1, 0.1, 0.1), (-1.0, -1.0, -1.0), "positive_but_underpowered"),
    ],
)
def test_quartile_classification(
    cross: tuple[float, ...],
    energy: tuple[float, ...],
    lower: tuple[float, ...],
    expected: str,
) -> None:
    assert (
        classify_quartile_signal(
            nominee_cross_terms=cross,
            nominee_prediction_energies=energy,
            nominee_adjusted_lower_bounds=lower,
        )
        == expected
    )


def test_path_count_forecast_is_strict_and_nonpositive_effect_is_infinite() -> None:
    assert (
        forecast_required_path_count(
            point_estimate=0.0, path_standard_deviation=1.0, critical_value=2.0
        )
        is None
    )
    required = forecast_required_path_count(
        point_estimate=0.5, path_standard_deviation=1.0, critical_value=2.0
    )
    assert required == 17
    assert 0.5 - 2.0 / math.sqrt(required) > 0.0
    assert 0.5 - 2.0 / math.sqrt(required - 1) <= 0.0


def test_exact_high_reverse_time_decision_requires_all_frozen_conditions() -> None:
    table, result = _synthetic_evidence()
    replay = replay_sealed_selection(table, result)
    nominees = tuple(
        Q0Nominee(
            seed=seed,
            update=900,
            candidate_index=index * 40 + 8,
            point_estimate=0.01,
            adjusted_lower_bound=0.001,
            positive_fine_cell_count=51,
        )
        for index, seed in enumerate(V3_MODEL_SEEDS)
    )
    qualifying = dataclasses.replace(
        replay,
        nominees=nominees,
        positive_component_count=3,
        discovered_component_indices=(6, 7, 224),
        candidate_with_all_point_estimates_positive=False,
    )
    assert (
        classify_adjudication_decision(
            qualifying,
            witness_quartile_energies=(0.0016, 0.0006, 0.0002, 0.0001),
        )
        == "exact_rb_high_reverse_time_only_signal"
    )
    contaminated = dataclasses.replace(
        qualifying, discovered_component_indices=(6, 56, 224)
    )
    assert (
        classify_adjudication_decision(
            contaminated,
            witness_quartile_energies=(0.0016, 0.0006, 0.0002, 0.0001),
        )
        == "mixed_time_local_signal_inconclusive"
    )

    multiplicity_only = dataclasses.replace(
        qualifying,
        nominees=replay.nominees,
        positive_component_count=0,
        discovered_component_indices=(),
        candidate_with_all_point_estimates_positive=True,
    )
    assert (
        classify_adjudication_decision(
            multiplicity_only,
            witness_quartile_energies=(0.0016, 0.0006, 0.0002, 0.0001),
        )
        == "multiplicity_only_underpowered"
    )

    negative_result = dataclasses.replace(
        result,
        point_estimates=np.ascontiguousarray(-np.abs(result.point_estimates)),
    )
    no_signal = dataclasses.replace(
        replay,
        result=negative_result,
        nominees=tuple(
            dataclasses.replace(value, adjusted_lower_bound=-1.0)
            for value in replay.nominees
        ),
        positive_component_count=0,
        discovered_component_indices=(),
        candidate_with_all_point_estimates_positive=False,
    )
    assert (
        classify_adjudication_decision(
            no_signal,
            witness_quartile_energies=(0.0016, 0.0006, 0.0002, 0.0001),
        )
        == "no_learned_time_local_signal"
    )


def test_local_production_artifacts_replay_exactly_when_present() -> None:
    run = Path(
        "runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_memory_confirmation/"
        "20260806-181326_production-zero-baseline-v3-memory-safe"
    )
    table_path = run / "validation_candidate_path_tables.npz"
    max_t_path = run / "validation_search_max_t.npz"
    if not table_path.is_file() or not max_t_path.is_file():
        pytest.skip("immutable production evidence is not part of this checkout")
    with np.load(table_path, allow_pickle=False) as table_archive:
        table_arrays = {name: np.array(table_archive[name], copy=True) for name in table_archive.files}
    with np.load(max_t_path, allow_pickle=False) as max_t_archive:
        max_t_arrays = {name: np.array(max_t_archive[name], copy=True) for name in max_t_archive.files}
    replay = build_sealed_selection_replay(
        **table_arrays,
        **max_t_arrays,
        require_production_fixture=True,
    )
    assert replay.result.critical_value == PRODUCTION_CRITICAL_VALUE
    assert replay.discovered_component_indices == PRODUCTION_DISCOVERED_COMPONENTS
    assert [(value.seed, value.update) for value in replay.nominees] == [
        (261_312, 900),
        (261_313, 1_600),
        (261_314, 3_900),
    ]
