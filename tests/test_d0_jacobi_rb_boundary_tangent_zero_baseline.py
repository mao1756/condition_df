from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from mnist.d0_jacobi_rb_boundary_tangent import (
    BoundaryTangentContractError,
    edge_pair_geometry,
    synthetic_tangent_target,
)
from mnist.d0_jacobi_rb_boundary_tangent_zero_baseline import (
    ZERO_BASELINE_BYTE_LENGTH,
    ZERO_BASELINE_SHA256,
    ZERO_BASELINE_SHAPE,
    ZeroBaselineBoundaryTangentPredictor,
    configure_exact_synthetic_zero_baseline_teacher,
    exact_zero_baseline_prediction,
    zero_baseline_contract,
)
from mnist.d0_jacobi_rb_learnability import (
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    STATE_SIZE,
    JacobiRBPhasePredictor,
    ModelInputs,
    call_model,
    matching_indices,
)
from mnist.d0_jacobi_rb_reverse_controller import internal_reverse_time


def _inputs(
    *,
    batch: int = 2,
    phase: int = 0,
    state: torch.Tensor | None = None,
) -> ModelInputs:
    active_state = (
        torch.full((batch, STATE_SIZE), 1.0 / STATE_SIZE, dtype=torch.float64)
        if state is None
        else state
    )
    return ModelInputs(
        later_full_state=active_state,
        reverse_time=torch.full(
            (batch,), internal_reverse_time(15, phase, 1.0 / 16.0), dtype=torch.float64
        ),
        phase=torch.full((batch,), phase, dtype=torch.long),
        color=torch.full((batch,), PHASE_MATCHINGS[phase], dtype=torch.long),
        duration=torch.full((batch,), PHASE_DURATIONS[phase], dtype=torch.float64),
        label=torch.full((batch,), 3, dtype=torch.long),
    )


def test_zero_baseline_contract_has_frozen_conceptual_hash_without_array() -> None:
    conceptual = np.zeros(ZERO_BASELINE_SHAPE, dtype=np.float64, order="C")
    assert conceptual.nbytes == ZERO_BASELINE_BYTE_LENGTH == 702_464
    assert hashlib.sha256(conceptual.tobytes(order="C")).hexdigest() == (
        ZERO_BASELINE_SHA256
    )

    record = zero_baseline_contract()
    assert record["formula"] == "q_B := 0"
    assert record["baseline_kind"] == "fixed_exact_zero"
    assert record["fitted_parameter_count"] == 0
    assert record["baseline_array_persisted"] == 0
    assert record["training_labels_used"] == 0
    assert record["validation_labels_used"] == 0
    assert record["confirmation_labels_used"] == 0
    assert record["target_modified"] == 0
    assert record["conceptual_array_sha256"] == ZERO_BASELINE_SHA256
    assert record["conceptual_array_byte_length"] == ZERO_BASELINE_BYTE_LENGTH
    json.dumps(record, allow_nan=False)


def test_predictor_has_only_the_unchanged_residual_network_state() -> None:
    model = ZeroBaselineBoundaryTangentPredictor()
    assert model.residual_score.width == 32
    assert set(model._modules) == {"residual_score"}  # noqa: SLF001
    assert set(model._buffers) == set()  # noqa: SLF001
    state_keys = tuple(model.state_dict())
    assert state_keys
    assert all(name.startswith("residual_score.") for name in state_keys)
    assert all("baseline" not in name and "_q_values" not in name for name in state_keys)
    assert not hasattr(model, "baseline")
    assert not hasattr(model, "baseline_score")
    assert not hasattr(model, "baseline_prediction")


def test_update_zero_score_and_prediction_are_bitwise_zero_on_facets() -> None:
    state = torch.full((2, STATE_SIZE), 1.0 / STATE_SIZE, dtype=torch.float64)
    tails, heads = matching_indices()
    state[0, int(tails[0, 0])] = 0.0
    state[0, int(heads[0, 1])] = 0.0
    state[0, int(tails[0, 2])] = 0.0
    state[0, int(heads[0, 2])] = 0.0
    inputs = _inputs(state=state)
    model = ZeroBaselineBoundaryTangentPredictor()

    score = model.score_prediction(inputs)
    prediction = call_model(model, inputs)
    baseline = exact_zero_baseline_prediction(inputs)
    assert score.dtype == torch.float64
    assert prediction.dtype == torch.float64
    assert torch.equal(score, torch.zeros_like(score))
    assert torch.equal(prediction, torch.zeros_like(prediction))
    assert torch.equal(baseline, torch.zeros_like(baseline))
    geometry = edge_pair_geometry(inputs)
    assert prediction[0, 0] == 0.0
    assert prediction[0, 1] == 0.0
    assert prediction[0, 2] == 0.0
    assert geometry.mobility[0, 0] == 0.0
    assert geometry.mobility[0, 1] == 0.0
    assert geometry.mobility[0, 2] == 0.0


def test_score_prediction_is_exactly_the_residual_score() -> None:
    residual = JacobiRBPhasePredictor(width=32)
    model = ZeroBaselineBoundaryTangentPredictor(residual, zero_residual=False)
    inputs = _inputs(batch=3, phase=4)
    expected_score = call_model(residual, inputs).to(dtype=torch.float64)
    assert torch.equal(model.score_prediction(inputs), expected_score)
    expected_prediction = edge_pair_geometry(inputs).mobility * expected_score
    assert torch.equal(call_model(model, inputs), expected_prediction)


def test_predictor_rejects_changed_width_and_non_model_inputs() -> None:
    with pytest.raises(BoundaryTangentContractError, match="width-32"):
        ZeroBaselineBoundaryTangentPredictor(JacobiRBPhasePredictor(width=4))
    model = ZeroBaselineBoundaryTangentPredictor()
    with pytest.raises(BoundaryTangentContractError, match="ModelInputs"):
        model.score_prediction(object())  # type: ignore[arg-type]


def test_exact_synthetic_teacher_is_representable_without_baseline_state() -> None:
    model = ZeroBaselineBoundaryTangentPredictor()
    configure_exact_synthetic_zero_baseline_teacher(model)
    inputs = _inputs(batch=3, phase=3)
    target = synthetic_tangent_target(inputs)
    assert bool(torch.mean(target.square()) > 0.0)
    torch.testing.assert_close(
        call_model(model, inputs), target, rtol=2.0e-6, atol=2.0e-7
    )
    assert all("baseline" not in name for name in model.state_dict())


def test_exact_model_null_step_leaves_parameters_bitwise_unchanged() -> None:
    teacher = ZeroBaselineBoundaryTangentPredictor()
    configure_exact_synthetic_zero_baseline_teacher(teacher)
    student = ZeroBaselineBoundaryTangentPredictor()
    student.load_state_dict(teacher.state_dict())
    inputs = _inputs(batch=3, phase=6)
    with torch.no_grad():
        target = call_model(teacher, inputs).detach().clone()
    assert bool(torch.mean(target.square()) > 0.0)
    before = {
        name: value.detach().clone() for name, value in student.state_dict().items()
    }
    optimizer = torch.optim.Adam(student.parameters(), lr=1.0e-3, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.mean((call_model(student, inputs) - target).square())
    assert loss == 0.0
    loss.backward()
    optimizer.step()
    assert all(
        torch.equal(before[name], value)
        for name, value in student.state_dict().items()
    )

