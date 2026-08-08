from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from mnist.d0_jacobi_rb_coarse_residual import (
    CoarseResidualPredictor,
    FrozenCoarseBaseline,
)
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    FORBIDDEN_MODEL_INPUT_FIELDS,
    PHASE_COUNT,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    STATE_SIZE,
    ModelInputs,
    matching_indices,
    selected_reverse_time,
)
from mnist.d0_jacobi_rb_reverse_controller import (
    ALLOWED_FRACTIONAL_COORDINATES,
    CLAIM_FLAG_NAMES,
    LOCAL_BOOTSTRAP_SEED,
    LOCAL_RISK_FAMILY_SIZE,
    MICROSTEP_REFINEMENT_MARGIN,
    NAMESPACE_VERSION,
    OBSERVABLE_NAMES,
    PHYSICAL_CONTROL_PATH_IDS,
    REFINEMENT_CONTROL_MICROSTEPS,
    REVERSE_LAW_BIAS_MARGIN,
    TRAJECTORY_BOOTSTRAP_SEED,
    TRAJECTORY_COMPONENT_COUNT,
    TRANSITION_ROLES,
    ControllerBoundaryStepRejected,
    EXPECTED_PARENT_CHECKPOINT_SHA256,
    EXPECTED_PARENT_STATE_SHA256,
    FractionalFrozenController,
    ReverseControllerContractError,
    assert_unambiguous_metric_schema,
    bounded_linear_teacher_score,
    claim_boundary,
    controlled_reverse_phase,
    controller_transition_id,
    controller_transition_ids,
    fractional_coordinate,
    frozen_control_half_flow,
    frozen_fractional_prediction,
    internal_reverse_time,
    learned_mass_flux,
    local_risk_gate,
    paired_observables,
    phase_exposure,
    reverse_execution_order,
    trajectory_gate,
    validate_claim_boundary,
    validate_controller_path_plan,
)


def _baseline_fixture() -> FrozenCoarseBaseline:
    coordinate = np.arange(4 * PHASE_COUNT * EDGES_PER_PHASE, dtype=np.float64)
    values = np.ascontiguousarray(coordinate.reshape(4, PHASE_COUNT, EDGES_PER_PHASE) / 1e6)
    raw = np.ascontiguousarray(2.0 * values)
    return FrozenCoarseBaseline(
        raw_values=raw,
        values=values,
        left_path_ids=np.arange(0x10000, 0x10040, dtype=np.int64),
        right_path_ids=np.arange(0x10100, 0x10140, dtype=np.int64),
        shrinkage=0.5,
        signal_energy=1.0,
        panel_mean_noise=2.0,
        averaged_table_noise=1.0,
        left_cell_means_file_sha256="a" * 64,
        right_cell_means_file_sha256="b" * 64,
        left_cell_means_array_sha256="c" * 64,
        right_cell_means_array_sha256="d" * 64,
        witness_registry_sha256="e" * 64,
    )


def _inputs(
    step: int,
    phase: int,
    q: float,
    *,
    batch: int = 1,
) -> ModelInputs:
    return ModelInputs(
        later_full_state=torch.full(
            (batch, STATE_SIZE), 1.0 / STATE_SIZE, dtype=torch.float32
        ),
        reverse_time=torch.full(
            (batch,), internal_reverse_time(step, phase, q), dtype=torch.float64
        ),
        phase=torch.full((batch,), phase, dtype=torch.long),
        color=torch.full((batch,), PHASE_MATCHINGS[phase], dtype=torch.long),
        duration=torch.full((batch,), PHASE_DURATIONS[phase], dtype=torch.float32),
        label=torch.full((batch,), 3, dtype=torch.long),
    )


def _fractional_controller() -> tuple[CoarseResidualPredictor, FractionalFrozenController]:
    original = CoarseResidualPredictor(_baseline_fixture(), zero_residual=True)
    return original, FractionalFrozenController(original)


def test_sealed_checkpoint_and_unchanged_target_bindings_are_frozen() -> None:
    assert EXPECTED_PARENT_CHECKPOINT_SHA256 == (
        "24a0893daa31196815463a7396220542003e7dc2557689950ba4dd0eeaa9c914"
    )
    assert EXPECTED_PARENT_STATE_SHA256 == (
        "df479e979cf6dd99580bd918377405b665791a4608f45f6cae326cc10e5e6ad9"
    )
    # The pure controller exposes no persisted target, residual target, or
    # target transformation: it consumes only the sealed predictor output.
    import mnist.d0_jacobi_rb_reverse_controller as module

    assert not any(
        name in module.__all__
        for name in ("residualized_target", "normalized_target", "score_target")
    )


def test_internal_reverse_time_endpoint_and_monotonic_reverse_order() -> None:
    for step in (0, 127, 511):
        for phase in range(PHASE_COUNT):
            assert internal_reverse_time(step, phase, 1.0) == selected_reverse_time(
                step, phase
            )
    order = reverse_execution_order()
    assert order[:8] == tuple((511, phase) for phase in range(6, -1, -1)) + ((510, 6),)
    assert order[-1] == (0, 0)
    times = [internal_reverse_time(step, phase, 1.0) for step, phase in order]
    assert all(right >= left for left, right in zip(times, times[1:]))
    assert PHASE_MATCHINGS[0] == PHASE_MATCHINGS[6] == 0
    assert order[0][1] == 6 and order[6][1] == 0


@pytest.mark.parametrize("q", ALLOWED_FRACTIONAL_COORDINATES)
def test_fractional_coordinate_recovers_only_permitted_input_coordinates(q: float) -> None:
    inputs = _inputs(255, 5, q)
    coordinate = fractional_coordinate(inputs.reverse_time, inputs.phase)
    assert coordinate.outer_step.item() == 255
    assert coordinate.within_phase_fraction.item() == q
    assert coordinate.forward_outer_quartile.item() == 1
    assert coordinate.reverse_quartile.item() == 2
    assert not coordinate.reverse_start.item()


@pytest.mark.parametrize("q", [0.0, 0.2, 0.5001, 1.1])
def test_fractional_coordinate_rejects_boundary_or_untrained_fraction(q: float) -> None:
    phase = 3
    time = internal_reverse_time(15, phase, q) if 0.0 <= q <= 1.0 else 0.5
    inputs = _inputs(15, phase, 1.0)
    inputs = ModelInputs(
        later_full_state=inputs.later_full_state,
        reverse_time=torch.tensor([time], dtype=torch.float64),
        phase=inputs.phase,
        color=inputs.color,
        duration=inputs.duration,
        label=inputs.label,
    )
    with pytest.raises(ReverseControllerContractError):
        fractional_coordinate(inputs.reverse_time, inputs.phase)


def test_fractional_adapter_is_bitwise_endpoint_equivalent() -> None:
    original, adapter = _fractional_controller()
    for step in (15, 127, 255, 383, 511):
        for phase in range(PHASE_COUNT):
            inputs = _inputs(step, phase, 1.0, batch=2)
            with torch.no_grad():
                old_residual = original.residual(inputs)
                new_residual = adapter.residual_prediction(inputs)
                old = original(inputs)
                new = adapter(inputs)
            assert torch.equal(old_residual, new_residual)
            assert torch.equal(old, new)
            assert new.dtype == torch.float64


def test_fractional_adapter_uses_piecewise_constant_quartile_without_outer_step() -> None:
    _, adapter = _fractional_controller()
    phase = 4
    first = _inputs(130, phase, 1.0 / 16.0)
    second = _inputs(250, phase, 15.0 / 16.0)
    prediction_a = adapter.baseline_prediction(first)
    prediction_b = adapter.baseline_prediction(second)
    assert torch.equal(prediction_a, prediction_b)
    assert torch.equal(prediction_a, adapter.predictor._coarse_values[1, phase][None])


def test_fractional_prediction_enforces_exact_six_field_firewall() -> None:
    _, adapter = _fractional_controller()
    inputs = _inputs(15, 0, 1.0 / 16.0)
    mapping = {name: getattr(inputs, name) for name in (
        "later_full_state", "reverse_time", "phase", "color", "duration", "label"
    )}
    frozen_fractional_prediction(adapter, mapping)
    for forbidden in FORBIDDEN_MODEL_INPUT_FIELDS | {"branch_identity", "witness_identity"}:
        invalid = dict(mapping)
        invalid[forbidden] = torch.zeros(1)
        with pytest.raises(Exception):
            frozen_fractional_prediction(adapter, invalid)


@pytest.mark.parametrize("mass", [0.25, 0.1])
@pytest.mark.parametrize("duration", [0.5, 1.0])
def test_phase_exposure_and_flux_have_exact_scaling_and_mass_cancellation(
    mass: float, duration: float
) -> None:
    expected = 3.0 * (5e-5 / 512.0) * duration / ((1.0 / 28.0) ** 2 * mass)
    assert phase_exposure(mass, duration) == pytest.approx(expected, rel=2e-16)
    m = 0.125
    physical = mass * 2.0 * m * phase_exposure(mass, duration)
    assert physical == pytest.approx(learned_mass_flux(m, duration=duration), rel=2e-16)
    other_mass = 0.8
    assert other_mass * 2.0 * m * phase_exposure(
        other_mass, duration
    ) == pytest.approx(physical, rel=2e-16)
    assert phase_exposure(0.0, duration) == 0.0


def test_phase_exposure_tensor_and_wrong_constants_fail_closed() -> None:
    mass = torch.tensor([0.0, 0.25], dtype=torch.float64)
    result = phase_exposure(mass, torch.tensor([0.5, 1.0]))
    assert result[0].item() == 0.0
    assert result[1].item() > 0.0
    with pytest.raises(ReverseControllerContractError):
        phase_exposure(-1.0, 0.5)
    with pytest.raises(ReverseControllerContractError):
        phase_exposure(1.0, 0.25)


def _edge_state(*, tail: float, head: float, color: int = 0) -> Tensor:
    state = torch.zeros(STATE_SIZE, dtype=torch.float64)
    tails, heads = matching_indices()
    state[tails[color][0]] = tail
    state[heads[color][0]] = head
    return state


def test_frozen_control_flow_sign_noop_and_pair_conservation() -> None:
    state = _edge_state(tail=0.6, head=0.4)
    zero = torch.zeros(EDGES_PER_PHASE, dtype=torch.float64)
    assert torch.equal(frozen_control_half_flow(state, 0, zero, 0.1), state)
    positive = zero.clone()
    positive[0] = 0.25
    output = frozen_control_half_flow(state, 0, positive, 0.1)
    tails, heads = matching_indices()
    assert output[heads[0][0]] > state[heads[0][0]]
    assert output[tails[0][0]] < state[tails[0][0]]
    assert (output[heads[0][0]] + output[tails[0][0]]).item() == pytest.approx(1.0)
    assert torch.sum(output).item() == pytest.approx(torch.sum(state).item())


def test_frozen_control_flow_rejects_outward_boundary_without_clipping() -> None:
    zero_head = _edge_state(tail=1.0, head=0.0)
    negative = torch.zeros(EDGES_PER_PHASE, dtype=torch.float64)
    negative[0] = -0.1
    with pytest.raises(ControllerBoundaryStepRejected) as error:
        frozen_control_half_flow(zero_head, 0, negative, 0.1)
    assert error.value.failure_code == "controller_boundary_step_rejected"
    zero_tail = _edge_state(tail=0.0, head=1.0)
    with pytest.raises(ControllerBoundaryStepRejected):
        frozen_control_half_flow(zero_tail, 0, -negative, 0.1)


def test_zero_pair_mass_is_exact_noop_even_with_nonzero_prediction() -> None:
    state = torch.zeros(STATE_SIZE, dtype=torch.float64)
    prediction = torch.ones(EDGES_PER_PHASE, dtype=torch.float64)
    assert torch.equal(frozen_control_half_flow(state, 0, prediction, 10.0), state)


class _ZeroController(nn.Module):
    def forward(self, inputs: ModelInputs) -> Tensor:
        return torch.zeros(
            (inputs.batch_size, EDGES_PER_PHASE),
            dtype=torch.float64,
            device=inputs.later_full_state.device,
        )


def _identity_reference(**kwargs: Any) -> dict[str, Tensor]:
    return {"later_head_fraction": kwargs["head_fraction"].clone()}


def test_controlled_reverse_phase_uses_only_exact_callback_and_frozen_midpoints() -> None:
    state = torch.full((2, STATE_SIZE), 1.0 / STATE_SIZE, dtype=torch.float64)
    calls: list[tuple[str, Tensor]] = []

    def reference(**kwargs: Any) -> dict[str, Tensor]:
        calls.append((kwargs["role"], kwargs["transition_ids"].clone()))
        return _identity_reference(**kwargs)

    result = controlled_reverse_phase(
        state,
        511,
        6,
        8,
        NAMESPACE_VERSION,
        controller=_ZeroController(),
        reference_transition=reference,
        path_ids=PHYSICAL_CONTROL_PATH_IDS[:2],
        label=3,
    )
    assert len(calls) == 16
    assert result.transition_count == 16 * 2 * EDGES_PER_PHASE
    assert result.boundary_rejection_count == 0
    assert result.projection_count == result.floor_count == result.limiter_count == 0
    expected_times = tuple(
        internal_reverse_time(511, 6, (j - 0.5) / 8.0) for j in range(8, 0, -1)
    )
    assert result.midpoint_reverse_times == expected_times
    assert torch.equal(result.state, state)
    assert all(role.startswith("reverse_reference_") for role, _ in calls)
    assert not torch.equal(calls[0][1], calls[1][1])


def test_controller_transition_ids_are_injective_role_separated_and_order_invariant() -> None:
    identifiers: set[int] = set()
    for role in TRANSITION_ROLES:
        for edge in (0, 391):
            value = controller_transition_id(
                path_id=0xEB000,
                outer_step=511,
                phase=6,
                reverse_microstep=7,
                edge=edge,
                role=role,
            )
            assert 0 <= value < 2**63
            identifiers.add(value)
    assert len(identifiers) == 2 * len(TRANSITION_ROLES)
    paths = (0xEB000, 0xEB001)
    together = controller_transition_ids(
        paths,
        outer_step=3,
        phase=2,
        reverse_microstep=1,
        role="reverse_reference_pre_control_M8",
        device="cpu",
    )
    reversed_rows = controller_transition_ids(
        paths[::-1],
        outer_step=3,
        phase=2,
        reverse_microstep=1,
        role="reverse_reference_pre_control_M8",
        device="cpu",
    )
    assert together.dtype == torch.uint64
    assert torch.equal(together[0], reversed_rows[1])
    assert torch.equal(together[1], reversed_rows[0])
    assert torch.unique(together).numel() == together.numel()
    with pytest.raises(ReverseControllerContractError):
        controller_transition_id(
            path_id=0xEB000,
            outer_step=0,
            phase=0,
            reverse_microstep=16,
            edge=0,
            role=TRANSITION_ROLES[0],
        )


def test_packed_transition_namespace_is_exhaustively_unique_across_frozen_roles() -> None:
    identifiers = {
        controller_transition_id(
            path_id=0xEB000,
            outer_step=127,
            phase=phase,
            reverse_microstep=microstep,
            edge=edge,
            role=role,
        )
        for role in TRANSITION_ROLES
        for phase in range(PHASE_COUNT)
        for microstep in range(8)
        for edge in range(EDGES_PER_PHASE)
    }
    assert len(identifiers) == len(TRANSITION_ROLES) * PHASE_COUNT * 8 * EDGES_PER_PHASE


def test_controller_path_plan_is_disjoint_and_frozen() -> None:
    record = validate_controller_path_plan()
    assert record["collision_free"] == 1
    roles = record["roles"]
    flattened = [value for values in roles.values() for value in values]
    assert len(flattened) == len(set(flattened))
    assert roles["physical_control"] == list(range(0xEB000, 0xEB040))


def test_paired_observables_have_fourteen_entries_and_structural_zero() -> None:
    before = np.full((3, STATE_SIZE), 1.0 / STATE_SIZE, dtype=np.float64)
    after = before.copy()
    tails, heads = matching_indices()
    # Horizontal within-row transfers leave the (0,1) coefficient invariant.
    transfer = np.float64(2.0**-20)
    pair = before[:, tails[0][0]] + before[:, heads[0][0]]
    after[:, tails[0][0]] = before[:, tails[0][0]] - transfer
    after[:, heads[0][0]] = pair - after[:, tails[0][0]]
    result = paired_observables(before, after, phase=0)
    assert result.names == OBSERVABLE_NAMES
    assert result.before.shape == result.after.shape == result.difference.shape == (3, 14)
    assert result.structural_invariant.sum() == 2
    assert np.array_equal(result.difference[:, 2:4], np.zeros((3, 2)))
    assert np.any(result.difference[:, 0:2] != 0.0)
    corrupted = after.copy()
    corrupted[:, heads[0][0]] += 1e-14
    with pytest.raises(ReverseControllerContractError):
        paired_observables(before, corrupted, phase=0)
    eight_phase = paired_observables(
        before, corrupted, phase=0, structural_phase_invariants=False
    )
    assert not eight_phase.structural_invariant.any()


def test_bounded_linear_teacher_is_finite_boundary_vanishing_and_signed() -> None:
    y = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    value = bounded_linear_teacher_score(y, 0.3)
    assert value[0] == value[-1] == 0.0
    assert np.isfinite(value).all()
    assert np.all(value[1:-1] > 0.0)
    np.testing.assert_allclose(
        bounded_linear_teacher_score(y, 0.3, c=-0.5),
        -2 * 0.5 * np.exp(-0.6) * y * (1-y) / (1 - 0.5*np.exp(-0.6)*(2*y-1)),
    )


def test_metric_schema_rejects_ambiguous_historical_name_recursively() -> None:
    assert_unambiguous_metric_schema(
        {"forward_outer_quartile": 3, "reverse_start": True}
    )
    with pytest.raises(ReverseControllerContractError):
        assert_unambiguous_metric_schema({"nested": [{"data_end": 1}]})
    with pytest.raises(ReverseControllerContractError):
        assert_unambiguous_metric_schema({"name": "old.data_end.signal"})


def test_gate_algebra_is_fail_closed_at_exact_boundaries() -> None:
    assert LOCAL_BOOTSTRAP_SEED == 261302
    assert TRAJECTORY_BOOTSTRAP_SEED == 261303
    assert local_risk_gate(np.full(LOCAL_RISK_FAMILY_SIZE, 1e-12))
    lower = np.ones(LOCAL_RISK_FAMILY_SIZE)
    lower[-1] = 0.0
    assert not local_risk_gate(lower)
    assert trajectory_gate(
        np.full(TRAJECTORY_COMPONENT_COUNT, REVERSE_LAW_BIAS_MARGIN),
        np.full(TRAJECTORY_COMPONENT_COUNT, MICROSTEP_REFINEMENT_MARGIN),
    )
    bias = np.zeros(TRAJECTORY_COMPONENT_COUNT)
    bias[0] = np.nextafter(REVERSE_LAW_BIAS_MARGIN, math.inf)
    assert not trajectory_gate(bias, np.zeros(TRAJECTORY_COMPONENT_COUNT))
    with pytest.raises(ReverseControllerContractError):
        local_risk_gate([1.0])


def test_claim_boundary_authorizes_planning_only_and_never_sampling() -> None:
    record = claim_boundary(controlled=True)
    assert record["one_image_reconstruction_planning_authorized"] == 1
    assert record["maximum_control_trajectory_phase_count"] == 8
    assert all(record[name] == 0 for name in CLAIM_FLAG_NAMES)
    validate_claim_boundary(record)
    invalid = dict(record)
    invalid["reverse_sampling_authorized"] = 1
    with pytest.raises(ReverseControllerContractError):
        validate_claim_boundary(invalid)


def test_module_has_no_approximate_transition_or_image_surface() -> None:
    import mnist.d0_jacobi_rb_reverse_controller as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "Euler" not in source.replace("Euler/Gaussian fallback", "")
    assert "imagegen" not in source
    assert not hasattr(module, "sample_image")
    assert REFINEMENT_CONTROL_MICROSTEPS == (2, 4, 8)
