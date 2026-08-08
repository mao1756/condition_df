from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mnist.d0_jacobi_rb_bayes_power import (
    BAYES_FORBIDDEN_MODEL_INPUT_FIELDS,
    BAYES_LABEL_CACHE_FIELDS,
    BAYES_ORACLE_AUDIT_FIELDS,
    EDGES_PER_PHASE,
    NULL_LAW,
    ROOT_SEED,
    TEACHER_LAW,
    BayesPowerContractError,
    ControlTransitionBatch,
    bounded_teacher_arrival_density_ratio,
    bounded_teacher_arrival_score,
    bounded_teacher_initial_density_ratio,
    bounded_teacher_oracle_mean,
    build_control_cache_bundle,
    canonical_control_transition_ids,
    construct_later_full_states,
    expected_control_transition_count,
    extract_pair_mass_templates,
    frozen_path_plan,
    frozen_scientific_config,
    generate_control_role_cache,
    load_bayes_label_cache,
    load_bayes_oracle_audit_cache,
    load_control_cache_bundle,
    null_discovery_signal,
    null_oracle_mean,
    oracle_metric_summary,
    sample_bounded_teacher_initial,
    save_bayes_label_cache,
    save_bayes_oracle_audit_cache,
    save_control_cache_bundle,
    teacher_confirmation_pass,
    tower_witness_products,
)
from mnist.d0_jacobi_rb_learnability import (
    FORBIDDEN_MODEL_INPUT_FIELDS,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    STATE_SIZE,
    LearnabilityContractError,
    LearnabilityInputCache,
    model_inputs_from_mapping,
    sample_key,
    selected_reverse_time,
)


def _template(path_id: int = 0xE0000) -> LearnabilityInputCache:
    rows = [(path_id, 15, 0), (path_id, 31, 3)]
    n = len(rows)
    phases = np.asarray([row[2] for row in rows], dtype=np.int8)
    return LearnabilityInputCache(
        sample_key=np.asarray([sample_key(*row) for row in rows], dtype=np.int64),
        later_full_state=np.full(
            (n, STATE_SIZE), 1.0 / STATE_SIZE, dtype=np.float64
        ),
        reverse_time=np.asarray(
            [selected_reverse_time(row[1], row[2]) for row in rows],
            dtype=np.float64,
        ),
        phase=phases,
        color=np.asarray([PHASE_MATCHINGS[p] for p in phases], dtype=np.int8),
        duration=np.asarray([PHASE_DURATIONS[p] for p in phases], dtype=np.float64),
        label=np.full(n, 3, dtype=np.int64),
    )


def _fake_certified_sampler(
    earlier: np.ndarray,
    exposure: np.ndarray,
    *,
    rng_key: object,
    transition_ids: np.ndarray,
) -> ControlTransitionBatch:
    del rng_key
    assert transition_ids.dtype == np.uint64
    later = 0.1 + 0.8 * np.asarray(earlier, dtype=np.float64)
    target = later * (1.0 - later) * np.exp(-2.0 * exposure)
    return ControlTransitionBatch(
        later_head_fraction=later,
        denoising_target=target,
        certificate_codes=np.full(later.shape, 15, dtype=np.uint8),
        diagnostics={"call_count": 1},
    )


def test_frozen_config_paths_and_workload_round_trip() -> None:
    config = frozen_scientific_config()
    plan = frozen_path_plan()
    assert config.total_transition_count == 4_214_784
    assert expected_control_transition_count(48) == 4_214_784
    assert len(plan.all_path_ids) == 48
    assert len(set(plan.all_path_ids)) == 48
    assert min(plan.teacher_train) == 0xE3000
    assert min(plan.null_train) == 0xE4000
    config_record = json.loads(json.dumps(config.to_record()))
    path_record = json.loads(json.dumps(plan.to_record()))
    assert type(config).from_record(config_record) == config
    assert type(plan).from_record(path_record) == plan
    tampered = dict(path_record)
    tampered["path_id_plan_sha256"] = "0" * 64
    with pytest.raises(BayesPowerContractError, match="hash"):
        type(plan).from_record(tampered)


def test_bounded_teacher_formulas_normalization_score_and_sampling() -> None:
    x = np.linspace(0.0, 1.0, 100_001, dtype=np.float64)
    q0 = bounded_teacher_initial_density_ratio(x)
    assert np.trapezoid(q0, x) == pytest.approx(1.0, abs=2.0e-12)
    for exposure in (0.0, 0.01, 1.0, 100.0):
        u = np.full_like(x, exposure)
        qu = bounded_teacher_arrival_density_ratio(x, u)
        score = bounded_teacher_arrival_score(x, u)
        oracle = bounded_teacher_oracle_mean(x, u)
        assert np.trapezoid(qu, x) == pytest.approx(1.0, abs=2.0e-12)
        assert np.max(np.abs(score * qu - np.exp(-2.0 * exposure))) < 1e-14
        assert np.array_equal(oracle, x * (1.0 - x) * score)
    assert np.array_equal(
        null_oracle_mean(x, np.ones_like(x)), np.zeros_like(x)
    )

    first = sample_bounded_teacher_initial(
        np.random.Generator(np.random.Philox(123)), 200_000
    )
    second = sample_bounded_teacher_initial(
        np.random.Generator(np.random.Philox(123)), 200_000
    )
    assert np.array_equal(first, second)
    assert first.mean() == pytest.approx(7.0 / 12.0, abs=3.0e-3)
    assert np.mean(first * first) == pytest.approx(5.0 / 12.0, abs=3.0e-3)


def test_pair_mass_template_reconstruction_preserves_full_simplex() -> None:
    rng = np.random.Generator(np.random.Philox(99))
    pair_mass = rng.dirichlet(np.ones(EDGES_PER_PHASE), size=8)
    arrival = rng.random((8, EDGES_PER_PHASE))
    colors = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    states = construct_later_full_states(pair_mass, arrival, colors)
    assert states.shape == (8, STATE_SIZE)
    assert np.max(np.abs(states.sum(axis=1) - 1.0)) < 2e-15
    recovered = extract_pair_mass_templates(states, colors)
    assert np.max(np.abs(recovered - pair_mass)) < 2.0e-18
    with pytest.raises(BayesPowerContractError, match="partition"):
        construct_later_full_states(pair_mass * 0.9, arrival, colors)


def test_cache_oracle_is_physically_separate_and_round_trips(tmp_path: Path) -> None:
    template = _template()
    paths = np.asarray([0xE3000, 0xE3000], dtype=np.int64)
    steps = np.asarray([15, 31], dtype=np.int16)
    phases = np.asarray([0, 3], dtype=np.int8)
    colors = np.asarray([PHASE_MATCHINGS[p] for p in phases], dtype=np.int8)
    pair_mass = extract_pair_mass_templates(template.later_full_state, colors)
    arrival = np.full((2, EDGES_PER_PHASE), 0.4, dtype=np.float64)
    earlier = np.full((2, EDGES_PER_PHASE), 0.3, dtype=np.float64)
    exposure = np.full((2, EDGES_PER_PHASE), 0.2, dtype=np.float64)
    target = np.full((2, EDGES_PER_PHASE), 0.1, dtype=np.float64)
    bundle = build_control_cache_bundle(
        path_id=paths,
        outer_step=steps,
        phase=phases,
        pair_mass_templates=pair_mass,
        earlier_head_fraction=earlier,
        arrival_head_fraction=arrival,
        exposure=exposure,
        denoising_target=target,
        certificate_codes=np.full(target.shape, 15, dtype=np.uint8),
        law=TEACHER_LAW,
    )
    assert set(bundle.inputs.arrays()) == {
        "sample_key",
        "later_full_state",
        "reverse_time",
        "phase",
        "color",
        "duration",
        "label",
    }
    assert "oracle_conditional_mean" not in bundle.labels.arrays()
    assert "denoising_target" not in bundle.oracle_audit.arrays()
    assert "later_full_state" not in bundle.oracle_audit.arrays()
    assert np.array_equal(bundle.oracle_audit.earlier_head_fraction, earlier)
    assert np.array_equal(
        bundle.oracle_audit.oracle_conditional_mean,
        bounded_teacher_oracle_mean(arrival, exposure),
    )
    training = bundle.training_bundle()
    assert np.array_equal(
        training.labels_audit.denoising_target, bundle.labels.denoising_target
    )

    label_path = tmp_path / "labels.npz"
    oracle_path = tmp_path / "oracle.npz"
    save_bayes_label_cache(label_path, bundle.labels)
    save_bayes_oracle_audit_cache(oracle_path, bundle.oracle_audit)
    with np.load(label_path, allow_pickle=False) as archive:
        assert set(archive.files) == set(BAYES_LABEL_CACHE_FIELDS)
        assert "oracle_conditional_mean" not in archive.files
    with np.load(oracle_path, allow_pickle=False) as archive:
        assert set(archive.files) == set(BAYES_ORACLE_AUDIT_FIELDS)
        assert "denoising_target" not in archive.files
    assert np.array_equal(
        load_bayes_label_cache(label_path).denoising_target, target
    )
    assert np.array_equal(
        load_bayes_oracle_audit_cache(oracle_path).oracle_conditional_mean,
        bundle.oracle_audit.oracle_conditional_mean,
    )
    input_path = tmp_path / "inputs.npz"
    save_control_cache_bundle(input_path, label_path, oracle_path, bundle)
    joined = load_control_cache_bundle(
        input_path,
        label_path,
        oracle_path,
        expected_path_ids=(0xE3000,),
        expected_outer_steps=(15, 31),
    )
    assert np.array_equal(
        joined.oracle_audit.oracle_conditional_mean,
        bundle.oracle_audit.oracle_conditional_mean,
    )


def test_oracle_fields_are_forbidden_from_model_input_firewall() -> None:
    assert set(FORBIDDEN_MODEL_INPUT_FIELDS).issubset(
        BAYES_FORBIDDEN_MODEL_INPUT_FIELDS
    )
    assert {
        "earlier_head_fraction",
        "arrival_head_fraction",
        "exposure",
        "oracle_conditional_mean",
    }.issubset(BAYES_FORBIDDEN_MODEL_INPUT_FIELDS)
    template = _template()
    values = {
        name: np.asarray(getattr(template, name))
        for name in (
            "later_full_state",
            "reverse_time",
            "phase",
            "color",
            "duration",
            "label",
        )
    }
    import torch

    tensor_values = {
        name: torch.as_tensor(value.copy())
        for name, value in values.items()
    }
    with pytest.raises(LearnabilityContractError):
        model_inputs_from_mapping(
            {
                **tensor_values,
                "oracle_conditional_mean": torch.zeros(
                    (2, EDGES_PER_PHASE), dtype=torch.float64
                ),
            }
        )


def test_role_generation_is_replayable_and_uses_exact_coordinates() -> None:
    template = _template()
    first = generate_control_role_cache(
        template,
        target_path_ids=(0xE3000,),
        law="bounded_teacher",
        sampler=_fake_certified_sampler,
        maximum_rows_per_call=2,
    )
    second = generate_control_role_cache(
        template,
        target_path_ids=(0xE3000,),
        law=TEACHER_LAW,
        sampler=_fake_certified_sampler,
        maximum_rows_per_call=1,
    )
    assert first.diagnostics["transition_count"] == 2 * EDGES_PER_PHASE
    assert first.diagnostics["certificate_fraction"] == 1.0
    assert np.array_equal(
        first.bundle.oracle_audit.earlier_head_fraction,
        second.bundle.oracle_audit.earlier_head_fraction,
    )
    assert np.array_equal(
        first.bundle.labels.denoising_target,
        second.bundle.labels.denoising_target,
    )
    assert set(np.unique(first.bundle.labels.path_id)) == {0xE3000}
    ids = canonical_control_transition_ids(0xE3000, 15, 0)
    assert ids.size == EDGES_PER_PHASE
    assert len(np.unique(ids)) == EDGES_PER_PHASE
    assert np.all(np.bitwise_and(first.bundle.labels.certificate_codes, 15) == 15)


def test_uncertified_role_generation_fails_closed() -> None:
    def bad_sampler(
        earlier: np.ndarray,
        exposure: np.ndarray,
        *,
        rng_key: object,
        transition_ids: np.ndarray,
    ) -> ControlTransitionBatch:
        del exposure, rng_key, transition_ids
        return ControlTransitionBatch(
            earlier,
            np.zeros_like(earlier),
            np.zeros_like(earlier, dtype=np.uint8),
        )

    with pytest.raises(BayesPowerContractError, match="uncertified"):
        generate_control_role_cache(
            _template(),
            target_path_ids=(0xE3000,),
            law=NULL_LAW,
            sampler=bad_sampler,
        )


def test_tower_witnesses_and_confirmation_gate_boundaries() -> None:
    arrival = np.linspace(0.1, 0.9, 2 * EDGES_PER_PHASE).reshape(
        2, EDGES_PER_PHASE
    )
    oracle = np.full_like(arrival, 0.2)
    products = tower_witness_products(oracle, oracle, arrival)
    assert products.shape == (2, EDGES_PER_PHASE, 4)
    assert np.count_nonzero(products) == 0

    path_ids = np.arange(8, dtype=np.int64)
    target = np.full((8, EDGES_PER_PHASE), 0.2, dtype=np.float64)
    perfect_oracle = target.copy()
    metadata = np.zeros_like(target)
    passing = oracle_metric_summary(
        perfect_oracle, target, perfect_oracle, metadata, path_ids
    )
    assert passing.oracle_relative_gain_over_zero == pytest.approx(1.0)
    assert passing.oracle_gain_recovery == pytest.approx(1.0)
    assert teacher_confirmation_pass(passing)
    assert null_discovery_signal(passing)

    zero_model = oracle_metric_summary(
        np.zeros_like(target), target, perfect_oracle, metadata, path_ids
    )
    assert not teacher_confirmation_pass(zero_model)
    assert not null_discovery_signal(zero_model)
