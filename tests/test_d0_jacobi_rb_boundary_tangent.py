from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    STATE_SIZE,
    ModelInputs,
    matching_indices,
)
from mnist.d0_jacobi_rb_reverse_controller import (
    NAMESPACE_VERSION,
    internal_reverse_time,
    phase_exposure,
)
from mnist.d0_jacobi_rb_boundary_tangent import (
    BoundaryTangentContractError,
    BoundaryTangentPredictor,
    M8_KNOTS,
    TANGENT_BASELINE_SHAPE,
    TANGENT_INTERPOLATION_RULE,
    TangentBaseline,
    configure_exact_synthetic_tangent_teacher,
    controlled_reverse_phase_tangent,
    derive_tangent_baseline,
    direct_raw_target_mse,
    edge_pair_geometry,
    frozen_score_logistic_flow,
    frozen_score_logistic_fraction,
    interpolate_tangent_baseline,
    load_tangent_baseline,
    save_tangent_baseline,
    synthetic_tangent_target,
)
from mnist.d0_jacobi_rb_boundary_tangent_fused import (
    TangentControlledPhaseResult,
    TangentScoreController,
    controlled_reverse_phase_tangent as controlled_reverse_phase_tangent_additive,
)


class _StructuralTangentController:
    def __init__(self, value: float = 0.0) -> None:
        self.value = float(value)
        self.input_types: list[type[object]] = []

    def score_prediction(self, inputs: ModelInputs) -> torch.Tensor:
        self.input_types.append(type(inputs))
        return torch.full(
            (inputs.batch_size, EDGES_PER_PHASE),
            self.value,
            dtype=torch.float64,
            device=inputs.later_full_state.device,
        )


def _identity_reference(**kwargs: object) -> dict[str, torch.Tensor]:
    return {"later_head_fraction": kwargs["head_fraction"]}


def _baseline(values: np.ndarray | None = None) -> TangentBaseline:
    q_values = (
        np.zeros(TANGENT_BASELINE_SHAPE, dtype=np.float64)
        if values is None
        else np.ascontiguousarray(values, dtype=np.float64)
    )
    denominator = np.ones(TANGENT_BASELINE_SHAPE, dtype=np.float64)
    return TangentBaseline(
        q_values=q_values,
        numerators=q_values.copy(),
        denominators=denominator,
        counts=np.ones(TANGENT_BASELINE_SHAPE, dtype=np.int64),
        training_path_ids=np.asarray([101, 102], dtype=np.int64),
        training_inputs_sha256="1" * 64,
        training_targets_sha256="2" * 64,
        training_row_path_ids_sha256="3" * 64,
    )


def _inputs(
    *,
    batch: int = 2,
    outer_step: int = 15,
    phase: int = 0,
    midpoint: float = M8_KNOTS[0],
    state: torch.Tensor | None = None,
) -> ModelInputs:
    active_state = (
        torch.full((batch, STATE_SIZE), 1.0 / STATE_SIZE, dtype=torch.float64)
        if state is None
        else state
    )
    reverse_time = internal_reverse_time(outer_step, phase, midpoint)
    return ModelInputs(
        later_full_state=active_state,
        reverse_time=torch.full((batch,), reverse_time, dtype=torch.float64),
        phase=torch.full((batch,), phase, dtype=torch.long),
        color=torch.full((batch,), PHASE_MATCHINGS[phase], dtype=torch.long),
        duration=torch.full((batch,), PHASE_DURATIONS[phase], dtype=torch.float64),
        label=torch.full((batch,), 3, dtype=torch.long),
    )


def _complete_training_fixture() -> tuple[ModelInputs, torch.Tensor, torch.Tensor, np.ndarray]:
    states: list[torch.Tensor] = []
    times: list[float] = []
    phases: list[int] = []
    colors: list[int] = []
    durations: list[float] = []
    coefficients: list[np.ndarray] = []
    path_ids: list[int] = []
    quartile_steps = (15, 143, 271, 399)
    edges = np.arange(EDGES_PER_PHASE, dtype=np.float64)
    for quartile, step in enumerate(quartile_steps):
        for phase in range(7):
            for midpoint_index, midpoint in enumerate(M8_KNOTS):
                states.append(
                    torch.full((STATE_SIZE,), 1.0 / STATE_SIZE, dtype=torch.float64)
                )
                times.append(internal_reverse_time(step, phase, midpoint))
                phases.append(phase)
                colors.append(PHASE_MATCHINGS[phase])
                durations.append(PHASE_DURATIONS[phase])
                coefficients.append(
                    0.5
                    + quartile
                    + 0.1 * phase
                    + 0.01 * midpoint_index
                    + 1.0e-5 * edges
                )
                path_ids.append(10_000 + len(path_ids))
    inputs = ModelInputs(
        later_full_state=torch.stack(states),
        reverse_time=torch.tensor(times, dtype=torch.float64),
        phase=torch.tensor(phases, dtype=torch.long),
        color=torch.tensor(colors, dtype=torch.long),
        duration=torch.tensor(durations, dtype=torch.float64),
        label=torch.full((len(states),), 3, dtype=torch.long),
    )
    mobility = edge_pair_geometry(inputs).mobility
    coefficient = torch.tensor(np.stack(coefficients), dtype=torch.float64)
    return inputs, mobility * coefficient, coefficient, np.asarray(path_ids, dtype=np.int64)


def test_pair_geometry_has_exact_facets_and_zero_mass() -> None:
    state = torch.full((1, STATE_SIZE), 1.0 / STATE_SIZE, dtype=torch.float64)
    tails, heads = matching_indices()
    tail = int(tails[0, 0])
    head = int(heads[0, 0])
    state[0, tail] = 0.0
    state[0, head] = 0.0
    state[0, int(tails[0, 1])] = 0.0
    inputs = _inputs(batch=1, state=state)
    geometry = edge_pair_geometry(inputs)
    assert geometry.pair_mass[0, 0] == 0.0
    assert geometry.head_fraction[0, 0] == 0.0
    assert geometry.mobility[0, 0] == 0.0
    assert geometry.head_fraction[0, 1] == 1.0
    assert geometry.mobility[0, 1] == 0.0


def test_pair_geometry_rejects_phase_color_mismatch() -> None:
    inputs = _inputs()
    bad = ModelInputs(
        later_full_state=inputs.later_full_state,
        reverse_time=inputs.reverse_time,
        phase=inputs.phase,
        color=torch.ones_like(inputs.color),
        duration=inputs.duration,
        label=inputs.label,
    )
    with pytest.raises(BoundaryTangentContractError, match="color"):
        edge_pair_geometry(bad)


def test_tangent_baseline_uses_exact_direct_formula() -> None:
    inputs, target, coefficient, path_ids = _complete_training_fixture()
    baseline = derive_tangent_baseline(inputs, target, path_ids)
    assert baseline.q_values.shape == TANGENT_BASELINE_SHAPE
    assert np.all(baseline.counts == 1)
    np.testing.assert_allclose(
        baseline.q_values.reshape(-1, EDGES_PER_PHASE),
        coefficient.numpy(),
        rtol=2.0e-16,
        atol=2.0e-16,
    )
    assert baseline.to_record()["formula"] == "q_B=sum_train(mu*Zbar)/sum_train(mu^2)"
    assert baseline.to_record()["quotient_target_persisted"] == 0


def test_tangent_baseline_fails_on_nonpositive_denominator() -> None:
    inputs, _, _, path_ids = _complete_training_fixture()
    zero_target = torch.zeros((inputs.batch_size, EDGES_PER_PHASE), dtype=torch.float64)
    state = inputs.later_full_state.clone()
    tails, heads = matching_indices()
    # Make one edge a facet in every row belonging to its matching color.
    for row, color in enumerate(inputs.color.tolist()):
        state[row, int(tails[color, 0])] = 0.0
    bad_inputs = ModelInputs(
        later_full_state=state,
        reverse_time=inputs.reverse_time,
        phase=inputs.phase,
        color=inputs.color,
        duration=inputs.duration,
        label=inputs.label,
    )
    with pytest.raises(BoundaryTangentContractError, match="denominator"):
        derive_tangent_baseline(bad_inputs, zero_target, path_ids)


def test_tangent_baseline_rejects_non_m8_fit_coordinate() -> None:
    inputs, target, _, path_ids = _complete_training_fixture()
    times = inputs.reverse_time.clone()
    times[0] = internal_reverse_time(15, 0, 0.25)
    bad = ModelInputs(
        later_full_state=inputs.later_full_state,
        reverse_time=times,
        phase=inputs.phase,
        color=inputs.color,
        duration=inputs.duration,
        label=inputs.label,
    )
    with pytest.raises(BoundaryTangentContractError, match="M8"):
        derive_tangent_baseline(bad, target, path_ids)


def test_baseline_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    inputs, target, _, path_ids = _complete_training_fixture()
    baseline = derive_tangent_baseline(inputs, target, path_ids)
    path = tmp_path / "tangent-baseline.npz"
    saved = save_tangent_baseline(path, baseline)
    loaded = load_tangent_baseline(path, expected_sha256=saved["sha256"])
    assert loaded.to_record() == baseline.to_record()
    assert loaded.q_values_sha256 == baseline.q_values_sha256

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["q_values"].flat[0] += 1.0
    np.savez_compressed(path, **arrays)
    with pytest.raises(BoundaryTangentContractError):
        load_tangent_baseline(path)


def test_tangent_baseline_formula_is_validated() -> None:
    baseline = _baseline(np.full(TANGENT_BASELINE_SHAPE, 2.0, dtype=np.float64))
    with pytest.raises(BoundaryTangentContractError, match="formula"):
        TangentBaseline(
            q_values=baseline.q_values,
            numerators=np.ones(TANGENT_BASELINE_SHAPE, dtype=np.float64),
            denominators=np.ones(TANGENT_BASELINE_SHAPE, dtype=np.float64),
            counts=baseline.counts,
            training_path_ids=baseline.training_path_ids,
            training_inputs_sha256="1" * 64,
            training_targets_sha256="2" * 64,
            training_row_path_ids_sha256="3" * 64,
        )


def test_interpolation_is_bitwise_at_knots_and_linear_for_m2_m4() -> None:
    values = np.empty(TANGENT_BASELINE_SHAPE, dtype=np.float64)
    for index in range(8):
        values[:, :, index, :] = index + np.arange(EDGES_PER_PHASE) * 1.0e-4
    baseline = _baseline(values)
    exact = _inputs(batch=1, midpoint=M8_KNOTS[3])
    exact_value = interpolate_tangent_baseline(baseline, exact)
    assert torch.equal(
        exact_value, torch.from_numpy(np.array(values[0, 0, 3], copy=True))[None]
    )

    midpoint = _inputs(batch=1, midpoint=0.25)
    interpolated = interpolate_tangent_baseline(baseline, midpoint)
    expected = 0.5 * (values[0, 0, 1] + values[0, 0, 2])
    torch.testing.assert_close(interpolated[0], torch.from_numpy(expected), rtol=0, atol=0)
    assert baseline.interpolation_rule == TANGENT_INTERPOLATION_RULE

    continuous = _inputs(batch=1, midpoint=0.2)
    continuous_value = interpolate_tangent_baseline(baseline, continuous)
    position = 8.0 * 0.2 - 0.5
    continuous_expected = values[0, 0, 1] + (position - 1.0) * (
        values[0, 0, 2] - values[0, 0, 1]
    )
    torch.testing.assert_close(
        continuous_value[0],
        torch.from_numpy(np.array(continuous_expected, copy=True)),
        rtol=0,
        atol=2.0e-12,
    )


def test_interpolation_rejects_endpoint_extrapolation() -> None:
    with pytest.raises(BoundaryTangentContractError, match="outside"):
        interpolate_tangent_baseline(_baseline(), _inputs(batch=1, midpoint=1.0))


def test_zero_residual_predictor_equals_tangent_baseline_and_facets_zero() -> None:
    values = np.full(TANGENT_BASELINE_SHAPE, 3.0, dtype=np.float64)
    model = BoundaryTangentPredictor(_baseline(values))
    inputs = _inputs(batch=1)
    expected = edge_pair_geometry(inputs).mobility * 3.0
    assert torch.equal(model(inputs), expected)

    state = inputs.later_full_state.clone()
    tails, heads = matching_indices()
    state[0, int(tails[0, 0])] = 0.0
    state[0, int(heads[0, 1])] = 0.0
    state[0, int(tails[0, 2])] = 0.0
    state[0, int(heads[0, 2])] = 0.0
    facet_inputs = _inputs(batch=1, state=state)
    prediction = model(facet_inputs)
    assert prediction[0, 0] == 0.0
    assert prediction[0, 1] == 0.0
    assert prediction[0, 2] == 0.0


def test_predictor_requires_unchanged_width_32_model() -> None:
    from mnist.d0_jacobi_rb_learnability import JacobiRBPhasePredictor

    with pytest.raises(BoundaryTangentContractError, match="width-32"):
        BoundaryTangentPredictor(
            _baseline(), JacobiRBPhasePredictor(width=4)
        )


def test_exact_synthetic_tangent_teacher_is_representable() -> None:
    model = BoundaryTangentPredictor(_baseline())
    configure_exact_synthetic_tangent_teacher(model)
    inputs = _inputs(batch=3)
    torch.testing.assert_close(
        model(inputs), synthetic_tangent_target(inputs), rtol=2.0e-6, atol=2.0e-7
    )


def test_direct_mse_uses_raw_target_and_scale() -> None:
    prediction = torch.zeros((1, EDGES_PER_PHASE), dtype=torch.float32)
    target = torch.zeros((1, EDGES_PER_PHASE), dtype=torch.float64)
    prediction[0, :2] = torch.tensor([1.0, 2.0])
    target[0, :2] = torch.tensor([2.0, 4.0])
    optimizer, raw = direct_raw_target_mse(prediction, target, 2.0)
    assert raw == 5.0 / EDGES_PER_PHASE
    assert optimizer == 5.0 / (4.0 * EDGES_PER_PHASE)


def test_logistic_fraction_facets_and_zero_duration_are_exact() -> None:
    y = torch.tensor([0.0, 0.25, 1.0], dtype=torch.float64)
    q = torch.tensor([100.0, -3.0, -100.0], dtype=torch.float64)
    result = frozen_score_logistic_fraction(y, q, 0.0)
    assert torch.equal(result, y)
    moved = frozen_score_logistic_fraction(y, q, 0.5)
    assert moved[0] == 0.0
    assert moved[2] == 1.0
    assert 0.0 < moved[1] < 1.0


def test_logistic_fraction_semigroup_and_local_derivative() -> None:
    y = torch.tensor([0.1, 0.4, 0.8], dtype=torch.float64)
    q = torch.tensor([-1.2, 0.7, 2.1], dtype=torch.float64)
    direct = frozen_score_logistic_fraction(y, q, 0.3)
    split = frozen_score_logistic_fraction(
        frozen_score_logistic_fraction(y, q, 0.1), q, 0.2
    )
    torch.testing.assert_close(split, direct, rtol=3.0e-15, atol=3.0e-16)

    epsilon = 1.0e-7
    derivative = (frozen_score_logistic_fraction(y, q, epsilon) - y) / epsilon
    expected = 2.0 * y * (1.0 - y) * q
    torch.testing.assert_close(derivative, expected, rtol=5.0e-7, atol=2.0e-9)


def test_logistic_flow_orientation_and_mass_conservation() -> None:
    generator = torch.Generator().manual_seed(12)
    state = torch.rand((2, STATE_SIZE), generator=generator, dtype=torch.float64)
    state /= state.sum(dim=1, keepdim=True)
    tails, heads = matching_indices()
    q = torch.linspace(-2.0, 2.0, EDGES_PER_PHASE, dtype=torch.float64)[None].repeat(2, 1)
    result = frozen_score_logistic_flow(state, (tails[0], heads[0]), q, 0.07)
    reversed_result = frozen_score_logistic_flow(
        state, (heads[0], tails[0]), -q, 0.07
    )
    torch.testing.assert_close(result, reversed_result, rtol=0, atol=2.0e-18)
    torch.testing.assert_close(result.sum(dim=1), state.sum(dim=1), rtol=0, atol=2.0e-16)
    torch.testing.assert_close(
        result[:, tails[0]] + result[:, heads[0]],
        state[:, tails[0]] + state[:, heads[0]],
        rtol=0,
        atol=5.0e-19,
    )
    assert bool(torch.all(result >= 0.0))


@pytest.mark.parametrize("microsteps", [2, 4, 8])
def test_controlled_tangent_phase_preserves_split_semantics(microsteps: int) -> None:
    values = np.full(TANGENT_BASELINE_SHAPE, 0.4, dtype=np.float64)
    controller = BoundaryTangentPredictor(_baseline(values))
    state = torch.full((2, STATE_SIZE), 1.0 / STATE_SIZE, dtype=torch.float64)
    calls: list[tuple[str, torch.Tensor]] = []

    def identity_reference(**kwargs: object) -> dict[str, torch.Tensor]:
        calls.append((str(kwargs["role"]), kwargs["transition_ids"].clone()))
        return {"later_head_fraction": kwargs["head_fraction"]}

    result = controlled_reverse_phase_tangent(
        state,
        127,
        0,
        microsteps,
        NAMESPACE_VERSION,
        controller=controller,
        reference_transition=identity_reference,
        path_ids=(100, 101),
        label=3,
    )
    assert len(calls) == 2 * microsteps
    assert result.transition_count == 2 * microsteps * 2 * EDGES_PER_PHASE
    assert result.midpoint_reverse_times == tuple(
        internal_reverse_time(127, 0, (j - 0.5) / microsteps)
        for j in range(microsteps, 0, -1)
    )
    assert result.maximum_pair_mass_error <= 2.0e-19
    assert result.maximum_simplex_mass_error <= 3.0e-16
    assert bool(torch.all(result.state >= 0.0))

    pair = state[:, 0::2] + state[:, 1::2]
    exposure = phase_exposure(pair, 0.5)
    expected_y = frozen_score_logistic_fraction(
        torch.full_like(pair, 0.5), torch.full_like(pair, 0.4), exposure
    )
    torch.testing.assert_close(
        result.state[:, 1::2] / pair, expected_y, rtol=2.0e-7, atol=2.0e-8
    )


def test_controlled_tangent_phase_rejects_wrong_namespace() -> None:
    controller = BoundaryTangentPredictor(_baseline())
    state = torch.full((1, STATE_SIZE), 1.0 / STATE_SIZE, dtype=torch.float64)
    with pytest.raises(BoundaryTangentContractError, match="namespace"):
        controlled_reverse_phase_tangent(
            state,
            0,
            0,
            8,
            "changed",
            controller=controller,
            reference_transition=lambda **kwargs: kwargs["head_fraction"],
            path_ids=(100,),
            label=3,
        )


def test_structural_tangent_controller_and_telemetry() -> None:
    controller = _StructuralTangentController(0.4)
    assert isinstance(controller, TangentScoreController)
    state = torch.full((2, STATE_SIZE), 1.0 / STATE_SIZE, dtype=torch.float64)
    result = controlled_reverse_phase_tangent_additive(
        state,
        127,
        0,
        2,
        NAMESPACE_VERSION,
        controller=controller,
        reference_transition=_identity_reference,
        path_ids=(100, 101),
        label=3,
    )
    assert isinstance(result, TangentControlledPhaseResult)
    assert all(item is ModelInputs for item in controller.input_types)
    assert result.reference_fraction_displacement_squared_sum == 0.0
    assert result.reference_fraction_displacement_count == 8 * EDGES_PER_PHASE
    assert result.reference_fraction_displacement_maximum_absolute == 0.0
    assert result.control_fraction_displacement_squared_sum > 0.0
    assert result.control_fraction_displacement_count == 2 * 2 * EDGES_PER_PHASE
    assert result.control_fraction_displacement_maximum_absolute > 0.0
    assert result.score_squared_sum == pytest.approx(
        2 * 2 * EDGES_PER_PHASE * 0.4**2
    )
    assert result.score_count == 2 * 2 * EDGES_PER_PHASE
    assert result.score_maximum_absolute == 0.4
    assert result.logistic_shift_squared_sum > 0.0
    assert result.logistic_shift_count == 2 * 2 * EDGES_PER_PHASE
    assert result.logistic_shift_maximum_absolute > 0.0
    assert result.boundary_fraction_count == 0


def test_frequency_one_controller_is_structurally_accepted() -> None:
    from mnist.d0_jacobi_rb_boundary_tangent_frequency1_coordinate import (
        FrequencyOneCoordinateZeroBaselinePredictor,
    )

    controller = FrequencyOneCoordinateZeroBaselinePredictor(zero_residual=True)
    state = torch.full((1, STATE_SIZE), 1.0 / STATE_SIZE, dtype=torch.float64)
    result = controlled_reverse_phase_tangent_additive(
        state,
        0,
        0,
        2,
        NAMESPACE_VERSION,
        controller=controller,
        reference_transition=_identity_reference,
        path_ids=(100,),
        label=3,
    )
    assert isinstance(result, TangentControlledPhaseResult)
    assert result.control_fraction_displacement_squared_sum == 0.0
    assert result.control_fraction_displacement_maximum_absolute == 0.0


def test_tangent_phase_rejects_missing_score_protocol() -> None:
    state = torch.full((1, STATE_SIZE), 1.0 / STATE_SIZE, dtype=torch.float64)
    with pytest.raises(BoundaryTangentContractError, match="score_prediction"):
        controlled_reverse_phase_tangent_additive(
            state,
            0,
            0,
            2,
            NAMESPACE_VERSION,
            controller=object(),  # type: ignore[arg-type]
            reference_transition=_identity_reference,
            path_ids=(100,),
            label=3,
        )


@pytest.mark.parametrize(
    "kind", ["not_tensor", "shape", "integer", "wrong_device", "nonfinite"]
)
def test_tangent_phase_rejects_invalid_structural_score(kind: str) -> None:
    class InvalidController:
        def score_prediction(self, inputs: ModelInputs) -> object:
            if kind == "not_tensor":
                return object()
            if kind == "shape":
                return torch.zeros(
                    (inputs.batch_size, EDGES_PER_PHASE - 1), dtype=torch.float64
                )
            if kind == "integer":
                return torch.zeros(
                    (inputs.batch_size, EDGES_PER_PHASE), dtype=torch.int64
                )
            if kind == "wrong_device":
                return torch.empty(
                    (inputs.batch_size, EDGES_PER_PHASE),
                    dtype=torch.float64,
                    device="meta",
                )
            return torch.full(
                (inputs.batch_size, EDGES_PER_PHASE),
                float("nan"),
                dtype=torch.float64,
            )

    state = torch.full((1, STATE_SIZE), 1.0 / STATE_SIZE, dtype=torch.float64)
    with pytest.raises(BoundaryTangentContractError, match="finite floating"):
        controlled_reverse_phase_tangent_additive(
            state,
            0,
            0,
            2,
            NAMESPACE_VERSION,
            controller=InvalidController(),  # type: ignore[arg-type]
            reference_transition=_identity_reference,
            path_ids=(100,),
            label=3,
        )


def test_tangent_phase_rejects_nonfinite_logistic_shift() -> None:
    class HugeFiniteController:
        def score_prediction(self, inputs: ModelInputs) -> torch.Tensor:
            return torch.full(
                (inputs.batch_size, EDGES_PER_PHASE),
                1.0e308,
                dtype=torch.float64,
                device=inputs.later_full_state.device,
            )

    state = torch.full((1, STATE_SIZE), 1.0 / STATE_SIZE, dtype=torch.float64)
    with pytest.raises(BoundaryTangentContractError, match="logistic shift"):
        controlled_reverse_phase_tangent_additive(
            state,
            0,
            0,
            2,
            NAMESPACE_VERSION,
            controller=HugeFiniteController(),
            reference_transition=_identity_reference,
            path_ids=(100,),
            label=3,
        )


def test_metadata_is_json_serializable() -> None:
    json.dumps(_baseline().to_record(), allow_nan=False)
