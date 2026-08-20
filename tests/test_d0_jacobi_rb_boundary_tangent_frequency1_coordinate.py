from __future__ import annotations

from collections import OrderedDict
import hashlib

import numpy as np
import pytest
import torch

from mnist.d0_jacobi_rb_absolute_coordinate import (
    edge_translation_permutation,
    model_translation_equivariance_record,
)
from mnist.d0_jacobi_rb_boundary_tangent_frequency1_coordinate import (
    COORDINATE_FREE_PARAMETER_COUNT,
    FREQUENCY1_COORDINATE_CHANNELS,
    FREQUENCY1_COORDINATE_PARAMETER_COUNT,
    FREQUENCY1_COORDINATE_PARAMETER_COUNT_TOTAL,
    FREQUENCY1_COORDINATE_SHA256,
    FREQUENCY1_COORDINATE_SHAPE,
    FREQUENCY1_COORDINATE_VERSION,
    FrequencyOneCoordinateContractError,
    FrequencyOneCoordinateJacobiRBPhasePredictor,
    FrequencyOneCoordinateZeroBaselinePredictor,
    active_head_frequency1_coordinates,
    canonical_frequency1_coordinate_array,
    configure_exact_synthetic_frequency1_coordinate_teacher,
    configure_exact_synthetic_frequency1_coordinate_zero_baseline_teacher,
    configure_frequency1_coordinate_symmetry_break_fixture,
    frequency1_coordinate_architecture_contract,
    frequency1_coordinate_array_audit,
    frequency1_coordinate_contract,
    frequency1_coordinate_input_contract,
    frequency1_coordinate_span_audit,
    frequency1_coordinate_teacher_contract,
    synthetic_frequency1_coordinate_target,
    upgrade_coordinate_free_state_dict,
    zero_initialize_frequency1_coordinate_residual,
)
from mnist.d0_jacobi_rb_boundary_tangent_zero_baseline import (
    ZeroBaselineBoundaryTangentPredictor,
)
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    GRID_SIZE,
    MODEL_INPUT_FIELDS,
    PHASE_COUNT,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    STATE_SIZE,
    JacobiRBPhasePredictor,
    LearnabilityContractError,
    ModelInputs,
    call_model,
)


def _inputs(*, batch: int = 7, phase: int | None = None) -> ModelInputs:
    phases = (
        torch.arange(batch, dtype=torch.long) % PHASE_COUNT
        if phase is None
        else torch.full((batch,), phase, dtype=torch.long)
    )
    colors = torch.as_tensor(PHASE_MATCHINGS, dtype=torch.long)[phases]
    durations = torch.as_tensor(PHASE_DURATIONS, dtype=torch.float64)[phases]
    state = torch.arange(1, batch * STATE_SIZE + 1, dtype=torch.float64).reshape(
        batch, STATE_SIZE
    )
    state /= torch.sum(state, dim=1, keepdim=True)
    return ModelInputs(
        later_full_state=state,
        reverse_time=torch.linspace(0.1, 0.9, batch, dtype=torch.float64),
        phase=phases,
        color=colors,
        duration=durations,
        label=torch.full((batch,), 3, dtype=torch.long),
    )


def test_prevalidated_one_row_forward_and_score_are_bit_identical() -> None:
    inputs = _inputs(batch=1, phase=3)
    model = FrequencyOneCoordinateJacobiRBPhasePredictor(width=32).eval()
    wrapped = FrequencyOneCoordinateZeroBaselinePredictor(
        model, zero_residual=False
    ).eval()
    with torch.inference_mode():
        ordinary = model(inputs)
        prevalidated = model.forward_prevalidated(inputs)
        ordinary_score = wrapped.score_prediction(inputs)
        prevalidated_score = wrapped.score_prediction_prevalidated(inputs)
    assert torch.equal(ordinary, prevalidated)
    assert torch.equal(ordinary_score, prevalidated_score)


def test_frozen_coordinate_values_layout_hash_and_contract() -> None:
    array = canonical_frequency1_coordinate_array()
    repeated = canonical_frequency1_coordinate_array()
    assert array.shape == FREQUENCY1_COORDINATE_SHAPE == (4, 28, 28)
    assert array.dtype == np.dtype("<f8")
    assert array.flags.c_contiguous and not array.flags.writeable
    assert np.array_equal(array, repeated)
    assert hashlib.sha256(array.tobytes(order="C")).hexdigest() == (
        FREQUENCY1_COORDINATE_SHA256
    )
    assert tuple(frequency1_coordinate_contract()["channel_order"]) == (
        FREQUENCY1_COORDINATE_CHANNELS
    )
    assert array[0, 0, 0] == 0.0
    assert array[1, 0, 0] == 1.0
    assert array[2, 0, 0] == 0.0
    assert array[3, 0, 0] == 1.0
    assert np.array_equal(array[0, :, 0], array[0, :, -1])
    assert np.array_equal(array[2, 0, :], array[2, -1, :])
    assert frequency1_coordinate_array_audit()["passed"] == 1

    writable = canonical_frequency1_coordinate_array()
    with pytest.raises(ValueError):
        writable[0, 0, 0] = 1.0


def test_active_head_mapping_has_rank_four_and_sealed_span() -> None:
    active = active_head_frequency1_coordinates()
    assert active.shape == (PHASE_COUNT, EDGES_PER_PHASE, 4)
    assert not active.flags.writeable
    audit = frequency1_coordinate_span_audit()
    assert audit["phase_ranks"] == [4] * PHASE_COUNT
    assert audit["maximum_projector_discrepancy"] <= 5.0e-14
    assert audit["sealed_basis_sha256"] == (
        "5fcaf84ca50d523fd750c5b1fe3f30e3464f0a8ef0883590dded812912b7f9c6"
    )
    assert audit["passed"] == 1


def test_architecture_adds_only_zero_coordinate_stem_and_internal_buffer() -> None:
    model = FrequencyOneCoordinateJacobiRBPhasePredictor()
    contract = frequency1_coordinate_architecture_contract(model)
    assert contract["passed"] == 1
    assert contract["coordinate_free_parameter_count"] == (
        COORDINATE_FREE_PARAMETER_COUNT
    )
    assert contract["added_parameter_count"] == FREQUENCY1_COORDINATE_PARAMETER_COUNT
    assert contract["trainable_parameter_count"] == (
        FREQUENCY1_COORDINATE_PARAMETER_COUNT_TOTAL
    )
    assert tuple(model.coordinate_stem_weight.shape) == (32, 4, 1, 1)
    assert torch.count_nonzero(model.coordinate_stem_weight) == 0
    assert model.conv1.in_channels == 24
    assert model.local_affine.in_features == 25
    assert tuple(dict(model.named_parameters()))[-1] == "coordinate_stem_weight"
    assert model.frequency1_coordinate.dtype == torch.float64
    assert model.frequency1_coordinate.requires_grad is False
    assert "frequency1_coordinate" in model.state_dict()
    assert "frequency1_coordinate" not in dict(model.named_parameters())
    assert tuple(field for field in MODEL_INPUT_FIELDS if "coordinate" in field) == ()
    assert frequency1_coordinate_input_contract()["passed"] == 1


def test_construction_preserves_inherited_rng_state_and_zero_stem_function() -> None:
    inputs = _inputs()
    torch.manual_seed(261385)
    old = JacobiRBPhasePredictor(width=32)
    old_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(261385)
    new = FrequencyOneCoordinateJacobiRBPhasePredictor(width=32)
    new_rng = torch.random.get_rng_state().clone()
    assert torch.equal(old_rng, new_rng)

    old_state = old.state_dict()
    new_state = new.state_dict()
    for name, value in old_state.items():
        assert name in new_state
        assert torch.equal(value, new_state[name])
    assert set(new_state) - set(old_state) == {
        "coordinate_stem_weight",
        "frequency1_coordinate",
    }
    assert torch.equal(call_model(old, inputs), call_model(new, inputs))

    old_wrapped = ZeroBaselineBoundaryTangentPredictor(old)
    wrapped = FrequencyOneCoordinateZeroBaselinePredictor(new)
    assert torch.equal(call_model(old_wrapped, inputs), call_model(wrapped, inputs))
    assert torch.equal(
        call_model(wrapped, inputs), torch.zeros_like(call_model(wrapped, inputs))
    )
    assert tuple(dict(wrapped.named_parameters()))[-1] == (
        "residual_score.coordinate_stem_weight"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_construction_preserves_rng_and_zero_stem_output_bitwise() -> None:
    device = torch.device("cuda")
    inputs = _inputs().to(device)
    torch.manual_seed(261385)
    old = JacobiRBPhasePredictor(width=32).to(device)
    old_cpu_rng = torch.random.get_rng_state().clone()
    old_cuda_rng = torch.cuda.get_rng_state(device).clone()
    torch.manual_seed(261385)
    new = FrequencyOneCoordinateJacobiRBPhasePredictor(width=32).to(device)
    new_cpu_rng = torch.random.get_rng_state().clone()
    new_cuda_rng = torch.cuda.get_rng_state(device).clone()
    assert torch.equal(old_cpu_rng, new_cpu_rng)
    assert torch.equal(old_cuda_rng, new_cuda_rng)
    with torch.no_grad():
        assert torch.equal(call_model(old, inputs), call_model(new, inputs))


def test_strict_state_upgrade_preserves_old_bytes_and_rejects_mutations() -> None:
    torch.manual_seed(73)
    old = JacobiRBPhasePredictor(width=32)
    old_state = old.state_dict()
    before_rng = torch.random.get_rng_state().clone()
    upgraded = upgrade_coordinate_free_state_dict(old_state)
    after_rng = torch.random.get_rng_state().clone()
    assert isinstance(upgraded, OrderedDict)
    assert torch.equal(before_rng, after_rng)
    for name, value in old_state.items():
        assert torch.equal(upgraded[name], value)
        assert upgraded[name].data_ptr() != value.data_ptr()
    assert torch.count_nonzero(upgraded["coordinate_stem_weight"]) == 0
    assert hashlib.sha256(
        upgraded["frequency1_coordinate"].numpy().astype("<f8", copy=False).tobytes()
    ).hexdigest() == FREQUENCY1_COORDINATE_SHA256
    new = FrequencyOneCoordinateJacobiRBPhasePredictor()
    new.load_state_dict(upgraded, strict=True)
    assert torch.equal(call_model(old, _inputs()), call_model(new, _inputs()))

    missing = dict(old_state)
    missing.pop("conv1.bias")
    with pytest.raises(FrequencyOneCoordinateContractError, match="key set"):
        upgrade_coordinate_free_state_dict(missing)
    additional = {**old_state, "forbidden": torch.zeros(1)}
    with pytest.raises(FrequencyOneCoordinateContractError, match="key set"):
        upgrade_coordinate_free_state_dict(additional)
    wrong_dtype = {name: value.clone() for name, value in old_state.items()}
    wrong_dtype["conv1.weight"] = wrong_dtype["conv1.weight"].double()
    with pytest.raises(FrequencyOneCoordinateContractError, match="dtype"):
        upgrade_coordinate_free_state_dict(wrong_dtype)
    nonfinite = {name: value.clone() for name, value in old_state.items()}
    nonfinite["conv1.weight"].reshape(-1)[0] = float("nan")
    with pytest.raises(FrequencyOneCoordinateContractError, match="nonfinite"):
        upgrade_coordinate_free_state_dict(nonfinite)


def test_zero_initialization_covers_both_outputs_and_coordinate_stem() -> None:
    model = FrequencyOneCoordinateJacobiRBPhasePredictor()
    with torch.no_grad():
        model.spatial_output.weight.fill_(1.0)
        model.spatial_output.bias.fill_(1.0)
        model.local_affine.weight.fill_(1.0)
        model.local_affine.bias.fill_(1.0)
        model.coordinate_stem_weight.fill_(1.0)
    zero_initialize_frequency1_coordinate_residual(model)
    assert torch.count_nonzero(model.spatial_output.weight) == 0
    assert torch.count_nonzero(model.spatial_output.bias) == 0
    assert torch.count_nonzero(model.local_affine.weight) == 0
    assert torch.count_nonzero(model.local_affine.bias) == 0
    assert torch.count_nonzero(model.coordinate_stem_weight) == 0
    assert torch.equal(call_model(model, _inputs()), torch.zeros(7, EDGES_PER_PHASE))
    with pytest.raises(FrequencyOneCoordinateContractError, match="exact"):
        zero_initialize_frequency1_coordinate_residual(  # type: ignore[arg-type]
            JacobiRBPhasePredictor(width=32)
        )


def test_nonzero_coordinate_stem_breaks_translation_symmetry_in_spatial_branch() -> None:
    model = FrequencyOneCoordinateJacobiRBPhasePredictor()
    configure_frequency1_coordinate_symmetry_break_fixture(model)
    inputs = _inputs(batch=2, phase=0)
    uniform = ModelInputs(
        later_full_state=torch.full_like(inputs.later_full_state, 1.0 / STATE_SIZE),
        reverse_time=inputs.reverse_time,
        phase=inputs.phase,
        color=inputs.color,
        duration=inputs.duration,
        label=inputs.label,
    )
    output = call_model(model, uniform)
    assert torch.var(output) > 0.0
    record = model_translation_equivariance_record(
        model, uniform, row_shift=2, column_shift=2, tolerance=0.0
    )
    assert record["passed"] == 0
    for color in range(4):
        permutation = edge_translation_permutation(color, 2, 2)
        np.testing.assert_array_equal(np.sort(permutation), np.arange(EDGES_PER_PHASE))


def test_frozen_synthetic_teacher_and_exact_model_null() -> None:
    inputs = _inputs(batch=7)
    teacher = FrequencyOneCoordinateZeroBaselinePredictor()
    configure_exact_synthetic_frequency1_coordinate_zero_baseline_teacher(teacher)
    target_from_model = call_model(teacher, inputs).detach().clone()
    analytic = synthetic_frequency1_coordinate_target(inputs)
    assert torch.mean(target_from_model.square()) > 0.0
    torch.testing.assert_close(target_from_model, analytic, rtol=2.0e-6, atol=2.0e-7)
    contract = frequency1_coordinate_teacher_contract()
    assert len(contract["state_dict_sha256"]) == 64
    assert contract["physical_labels_used"] == 0

    student = FrequencyOneCoordinateZeroBaselinePredictor()
    student.load_state_dict(teacher.state_dict(), strict=True)
    before = {name: value.detach().clone() for name, value in student.state_dict().items()}
    optimizer = torch.optim.Adam(student.parameters(), lr=1.0e-3, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    prediction = call_model(student, inputs)
    assert torch.equal(prediction, target_from_model)
    loss = torch.mean((prediction - target_from_model).square())
    assert loss == 0.0
    loss.backward()
    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
        for parameter in student.parameters()
    )
    optimizer.step()
    assert all(
        torch.equal(before[name], value) for name, value in student.state_dict().items()
    )
    # Optimizer state is keyed by Parameter, not its string name; make the
    # inclusion check directly against the named parameter object.
    stem = dict(student.named_parameters())["residual_score.coordinate_stem_weight"]
    assert stem in optimizer.state


def test_forward_firewall_and_changed_width_fail_closed() -> None:
    model = FrequencyOneCoordinateJacobiRBPhasePredictor()
    with pytest.raises(LearnabilityContractError, match="ModelInputs"):
        model(object())  # type: ignore[arg-type]
    with pytest.raises(FrequencyOneCoordinateContractError, match="width 32"):
        FrequencyOneCoordinateJacobiRBPhasePredictor(width=16)
    with pytest.raises(FrequencyOneCoordinateContractError, match="exact width-32"):
        FrequencyOneCoordinateZeroBaselinePredictor(  # type: ignore[arg-type]
            JacobiRBPhasePredictor(width=32)
        )
    assert FREQUENCY1_COORDINATE_VERSION.endswith("-v1")
