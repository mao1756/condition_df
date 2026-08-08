from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from mnist.d0_jacobi_rb_learnability import (
    AuditTargets,
    CheckpointCandidate,
    CONFIRMATION_PATH_IDS,
    EDGES_PER_PHASE,
    FORBIDDEN_MODEL_INPUT_FIELDS,
    INPUT_CACHE_FIELDS,
    LABEL_AUDIT_CACHE_FIELDS,
    MODEL_INPUT_FIELDS,
    PHASE_COUNT,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    STATE_SIZE,
    TRAIN_PATH_IDS,
    JacobiRBPhasePredictor,
    LearnabilityCacheBundle,
    LearnabilityContractError,
    LearnabilityInputCache,
    LearnabilityLabelAuditCache,
    LearnabilityPathPlan,
    ModelInputs,
    TrainingPlan,
    VALIDATION_PATH_IDS,
    all_positive_path_sign_test,
    audit_targets_from_cache,
    call_model,
    configure_exact_synthetic_teacher,
    deterministic_batch_indices,
    discover_repository_path_id_claims,
    exact_global_target_scale,
    expected_selected_sample_count,
    expected_transition_count,
    fit_metadata_baseline,
    frozen_path_plan,
    frozen_scientific_config,
    globally_scaled_mse,
    load_cache_bundle,
    model_inputs_from_cache,
    model_inputs_from_mapping,
    path_mse_summary,
    sample_key,
    save_cache_bundle,
    scan_path_id_collisions,
    select_checkpoint_candidate,
    selected_reverse_time,
    state_dict_sha256,
    synthetic_teacher_target,
    train_deterministic_regressor,
    validate_cache_bundle,
    validate_path_id,
)


def _bundle(
    path_ids: tuple[int, ...] = (0xE0000, 0xE0001),
    steps: tuple[int, ...] = (15, 143, 271, 399),
) -> LearnabilityCacheBundle:
    rows = [
        (path, step, phase)
        for path in path_ids
        for step in steps
        for phase in range(PHASE_COUNT)
    ]
    n = len(rows)
    keys = np.asarray([sample_key(*row) for row in rows], dtype=np.int64)
    path = np.asarray([row[0] for row in rows], dtype=np.int64)
    outer = np.asarray([row[1] for row in rows], dtype=np.int16)
    phase = np.asarray([row[2] for row in rows], dtype=np.int8)
    state = np.full((n, STATE_SIZE), 1.0 / STATE_SIZE, dtype=np.float64)
    # Give every row a state-dependent local contrast without changing mass.
    state[:, 0] += np.linspace(0.0, 1.0e-5, n)
    state[:, 1] -= np.linspace(0.0, 1.0e-5, n)
    reverse = np.asarray(
        [selected_reverse_time(row[1], row[2]) for row in rows], dtype=np.float64
    )
    inputs = LearnabilityInputCache(
        sample_key=keys,
        later_full_state=state,
        reverse_time=reverse,
        phase=phase,
        color=np.asarray([PHASE_MATCHINGS[value] for value in phase], dtype=np.int8),
        duration=np.asarray([PHASE_DURATIONS[value] for value in phase], dtype=np.float64),
        label=np.full(n, 3, dtype=np.int64),
    )
    target = np.arange(n * EDGES_PER_PHASE, dtype=np.float64).reshape(
        n, EDGES_PER_PHASE
    )
    target = target / max(target.size, 1)
    audit = LearnabilityLabelAuditCache(
        sample_key=keys,
        path_id=path,
        outer_step=outer,
        phase=phase,
        denoising_target=target,
        certificate_codes=np.full((n, EDGES_PER_PHASE), 15, dtype=np.uint8),
    )
    return LearnabilityCacheBundle(inputs, audit)


def test_frozen_configuration_and_collision_free_roles() -> None:
    config = frozen_scientific_config()
    plan = frozen_path_plan()
    assert config.selected_outer_steps == tuple(15 + 16 * value for value in range(32))
    assert config.selected_outer_step_count == 32
    assert len(config.sha256) == 64
    assert plan.train == TRAIN_PATH_IDS
    assert plan.validation == VALIDATION_PATH_IDS
    assert plan.confirmation == CONFIRMATION_PATH_IDS
    assert len(set(plan.all_path_ids)) == 24
    assert max(plan.all_path_ids) < 2**20
    assert len(plan.sha256) == 64
    assert type(config).from_record(config.to_record()) == config
    assert type(plan).from_record(plan.to_record()) == plan
    tampered = plan.to_record()
    tampered["roles"]["train"][0] += 1
    with pytest.raises(LearnabilityContractError, match="hash"):
        type(plan).from_record(tampered)
    assert expected_transition_count(24) == 33_718_272
    assert expected_selected_sample_count(8) == 1_792


@pytest.mark.parametrize(
    ("value", "valid"),
    [(0, True), (2**20 - 1, True), (2**20, False), (-1, False), (True, False)],
)
def test_path_id_boundary(value: object, valid: bool) -> None:
    if valid:
        assert validate_path_id(value) == value
    else:
        with pytest.raises((TypeError, LearnabilityContractError)):
            validate_path_id(value)


def test_path_plan_collision_scanner_and_source_discovery(tmp_path: Path) -> None:
    root = tmp_path
    (root / "mnist").mkdir()
    (root / "runs" / "old").mkdir(parents=True)
    (root / "mnist" / "old_path_ids.py").write_text(
        "TOWER_A_START = 0xE0000\nRESERVED_PRODUCTION_START=0xF0000\n"
        "RESERVED_PRODUCTION_STOP=0x100000\n",
        encoding="utf-8",
    )
    (root / "runs" / "old" / "path_id_plan.json").write_text(
        json.dumps({"roles": {"path_ids": [0xE1000]}, "slot": [0xD0000, 0xD1000]}),
        encoding="utf-8",
    )
    claims = discover_repository_path_id_claims(root)
    collisions = scan_path_id_collisions(
        [0xE0000, 0xE1000, 0xE2000], claims
    )
    assert {item.path_ids for item in collisions} >= {(0xE0000,), (0xE1000,)}
    with pytest.raises(LearnabilityContractError, match="collision"):
        LearnabilityPathPlan().assert_collision_free(claims)
    LearnabilityPathPlan(
        train=(0x01000,),
        validation=(0x02000,),
        confirmation=(0x03000,),
    ).assert_collision_free(claims)


def test_cache_schemas_round_trip_and_are_physically_separate(tmp_path: Path) -> None:
    bundle = _bundle()
    input_path = tmp_path / "train_inputs.npz"
    audit_path = tmp_path / "train_labels_audit.npz"
    save_cache_bundle(input_path, audit_path, bundle)
    with np.load(input_path, allow_pickle=False) as archive:
        assert set(archive.files) == set(INPUT_CACHE_FIELDS)
        assert not set(archive.files).intersection(
            {"path_id", "outer_step", "denoising_target", "certificate_codes"}
        )
    with np.load(audit_path, allow_pickle=False) as archive:
        assert set(archive.files) == set(LABEL_AUDIT_CACHE_FIELDS)
        assert "later_full_state" not in archive.files
    loaded = load_cache_bundle(
        input_path,
        audit_path,
        expected_path_ids=(0xE0000, 0xE0001),
        expected_outer_steps=(15, 143, 271, 399),
    )
    assert np.array_equal(loaded.inputs.later_full_state, bundle.inputs.later_full_state)
    assert np.array_equal(
        loaded.labels_audit.denoising_target,
        bundle.labels_audit.denoising_target,
    )
    assert not loaded.inputs.later_full_state.flags.writeable
    assert not loaded.labels_audit.denoising_target.flags.writeable


def test_cache_join_and_frozen_schedule_validation_fail_closed() -> None:
    bundle = _bundle(path_ids=(0xE0000,), steps=(15,))
    with pytest.raises(LearnabilityContractError, match="dtype"):
        LearnabilityInputCache(
            sample_key=bundle.inputs.sample_key,
            later_full_state=np.asarray(bundle.inputs.later_full_state, dtype=np.float32),
            reverse_time=bundle.inputs.reverse_time,
            phase=bundle.inputs.phase,
            color=bundle.inputs.color,
            duration=bundle.inputs.duration,
            label=bundle.inputs.label,
        )
    bad_keys = np.asarray(bundle.labels_audit.sample_key).copy()
    bad_keys[0] += 100
    with pytest.raises(LearnabilityContractError, match="encode"):
        LearnabilityLabelAuditCache(
            sample_key=bad_keys,
            path_id=bundle.labels_audit.path_id,
            outer_step=bundle.labels_audit.outer_step,
            phase=bundle.labels_audit.phase,
            denoising_target=bundle.labels_audit.denoising_target,
            certificate_codes=bundle.labels_audit.certificate_codes,
        )
    with pytest.raises(LearnabilityContractError, match="outer-step"):
        validate_cache_bundle(bundle, expected_outer_steps=(15, 31))


def test_model_input_firewall_rejects_audit_and_oracle_fields() -> None:
    bundle = _bundle(path_ids=(0xE0000,), steps=(15,))
    inputs = model_inputs_from_cache(bundle.inputs)
    mapping = {name: getattr(inputs, name) for name in MODEL_INPUT_FIELDS}
    reconstructed = model_inputs_from_mapping(mapping)
    assert all(
        getattr(reconstructed, name) is getattr(inputs, name)
        for name in MODEL_INPUT_FIELDS
    )
    for forbidden in FORBIDDEN_MODEL_INPUT_FIELDS:
        with pytest.raises(LearnabilityContractError):
            model_inputs_from_mapping({**mapping, forbidden: torch.zeros(1)})
    with pytest.raises(LearnabilityContractError):
        model_inputs_from_mapping({name: value for name, value in mapping.items() if name != "label"})


def test_audit_targets_remain_separate_from_model_inputs() -> None:
    bundle = _bundle(path_ids=(0xE0000,), steps=(15,))
    audit = audit_targets_from_cache(bundle.labels_audit)
    assert isinstance(audit, AuditTargets)
    assert audit.sample_count == bundle.sample_count
    assert audit.denoising_target.dtype == torch.float64
    assert not {
        "later_full_state",
        "reverse_time",
        "color",
        "duration",
        "label",
    }.intersection(audit.__dataclass_fields__)


def test_model_output_matching_gather_and_exact_teacher_skip() -> None:
    bundle = _bundle(path_ids=(0xE0000,), steps=(15,))
    inputs = model_inputs_from_cache(bundle.inputs)
    model = JacobiRBPhasePredictor(width=4)
    output = call_model(model, inputs)
    assert output.shape == (PHASE_COUNT, EDGES_PER_PHASE)
    configure_exact_synthetic_teacher(model)
    output = call_model(model, inputs).to(torch.float64)
    target = synthetic_teacher_target(inputs)
    assert torch.max(torch.abs(output - target)).item() < 1.0e-6


def test_model_rejects_inconsistent_phase_color() -> None:
    bundle = _bundle(path_ids=(0xE0000,), steps=(15,))
    inputs = model_inputs_from_cache(bundle.inputs)
    wrong = ModelInputs(
        inputs.later_full_state,
        inputs.reverse_time,
        inputs.phase,
        torch.remainder(inputs.color + 1, 4),
        inputs.duration,
        inputs.label,
    )
    with pytest.raises(LearnabilityContractError, match="color"):
        JacobiRBPhasePredictor(width=2)(wrong)


def test_metadata_baseline_uses_only_supplied_training_rows() -> None:
    bundle = _bundle()
    target = bundle.labels_audit.denoising_target
    baseline = fit_metadata_baseline(
        target,
        bundle.labels_audit.outer_step,
        bundle.labels_audit.phase,
    )
    prediction = baseline.predict(
        bundle.labels_audit.outer_step, bundle.labels_audit.phase
    )
    assert prediction.shape == target.shape
    assert len(baseline.sha256) == 64
    modified = target.copy()
    modified[bundle.labels_audit.path_id == 0xE0001] += 10
    other = fit_metadata_baseline(
        modified,
        bundle.labels_audit.outer_step,
        bundle.labels_audit.phase,
    )
    assert other.sha256 != baseline.sha256


def test_global_scale_is_one_constant_multiple_of_raw_mse() -> None:
    target = torch.linspace(-2.0, 3.0, EDGES_PER_PHASE, dtype=torch.float64)[None]
    prediction = target + 0.25
    scale = exact_global_target_scale(target)
    scaled, raw = globally_scaled_mse(prediction, target, scale)
    assert torch.equal(scaled, raw / (scale * scale))
    assert raw.item() == pytest.approx(0.25**2)
    with pytest.raises(LearnabilityContractError):
        exact_global_target_scale(np.zeros((2, EDGES_PER_PHASE)))


def test_deterministic_batches_and_checkpoint_ties() -> None:
    first = deterministic_batch_indices(17, 32, 5, 1234)
    second = deterministic_batch_indices(17, 32, 5, 1234)
    assert np.array_equal(first, second)
    assert len(first) == 32
    state = {"weight": torch.tensor([1.0])}
    digest = state_dict_sha256(state)
    candidates = [
        CheckpointCandidate(2, 100, 0.5, digest, state),
        CheckpointCandidate(1, 200, 0.5, digest, state),
        CheckpointCandidate(1, 100, 0.5, digest, state),
    ]
    selected = select_checkpoint_candidate(candidates)
    assert (selected.seed, selected.update) == (1, 100)


class _TinyAllowedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: ModelInputs) -> torch.Tensor:
        return self.bias.expand(inputs.batch_size, EDGES_PER_PHASE)


def test_reduced_deterministic_training_replays() -> None:
    bundle = _bundle()
    inputs = model_inputs_from_cache(bundle.inputs)
    target = torch.as_tensor(
        np.asarray(bundle.labels_audit.denoising_target).copy(), dtype=torch.float64
    )
    scale = exact_global_target_scale(target)
    first = train_deterministic_regressor(
        _TinyAllowedModel,
        inputs,
        target,
        inputs,
        target,
        target_scale=scale,
        seed=261201,
        plan=TrainingPlan(),
        maximum_updates=2,
    )
    second = train_deterministic_regressor(
        _TinyAllowedModel,
        inputs,
        target,
        inputs,
        target,
        target_scale=scale,
        seed=261201,
        plan=TrainingPlan(),
        maximum_updates=2,
    )
    assert first.finite and second.finite
    assert first.selected.state_sha256 == second.selected.state_sha256
    assert first.selected.validation_mse == second.selected.validation_mse


def test_deterministic_training_resume_preserves_optimizer_trajectory() -> None:
    bundle = _bundle()
    inputs = model_inputs_from_cache(bundle.inputs)
    target = torch.as_tensor(
        np.asarray(bundle.labels_audit.denoising_target).copy(),
        dtype=torch.float64,
    )
    scale = exact_global_target_scale(target)
    partial_snapshots = []
    train_deterministic_regressor(
        _TinyAllowedModel,
        inputs,
        target,
        inputs,
        target,
        target_scale=scale,
        seed=261201,
        maximum_updates=2,
        checkpoint_callback=partial_snapshots.append,
    )
    resumed_snapshots = []
    train_deterministic_regressor(
        _TinyAllowedModel,
        inputs,
        target,
        inputs,
        target,
        target_scale=scale,
        seed=261201,
        maximum_updates=4,
        resume_snapshot=partial_snapshots[-1],
        checkpoint_callback=resumed_snapshots.append,
    )
    uninterrupted_snapshots = []
    train_deterministic_regressor(
        _TinyAllowedModel,
        inputs,
        target,
        inputs,
        target,
        target_scale=scale,
        seed=261201,
        maximum_updates=4,
        checkpoint_callback=uninterrupted_snapshots.append,
    )
    assert state_dict_sha256(
        resumed_snapshots[-1].model_state_dict
    ) == state_dict_sha256(uninterrupted_snapshots[-1].model_state_dict)
    assert (
        resumed_snapshots[-1].completed_update
        == uninterrupted_snapshots[-1].completed_update
        == 4
    )


def test_path_metrics_and_strict_sign_rule() -> None:
    target = np.ones((16, EDGES_PER_PHASE), dtype=np.float64)
    paths = np.repeat(np.arange(8, dtype=np.int64), 2)
    model = np.full_like(target, 0.9)
    metadata = np.zeros_like(target)
    summary = path_mse_summary(model, target, metadata, paths)
    assert all(value > 0 for value in summary.improvements)
    sign = all_positive_path_sign_test(summary.improvements)
    assert sign.all_strictly_positive
    assert sign.one_sided_all_positive_p_value == pytest.approx(1 / 256)
    failed = all_positive_path_sign_test((*summary.improvements[:-1], 0.0))
    assert not failed.all_strictly_positive
    assert failed.one_sided_all_positive_p_value == 1.0
