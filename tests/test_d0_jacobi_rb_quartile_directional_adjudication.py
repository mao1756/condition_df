from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mnist.d0_jacobi_rb_quartile_directional_adjudication import (
    COMPONENT_NAMES,
    ComponentMomentAccumulator,
    ComponentMomentCube,
    ComponentPredictions,
    QuartileDirectionalAdjudicationError,
    component_summary,
    evaluate_frozen_components,
    marginalize,
    normalized_cosine,
    positive_ray_optimum,
    quadratic_improvement,
    weighted_pooled,
)
from mnist.d0_jacobi_rb_boundary_tangent_zero_baseline import (
    ZeroBaselineBoundaryTangentPredictor,
)
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    PHASE_COUNT,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    STATE_SIZE,
    ModelInputs,
)


def _inputs(batch: int = PHASE_COUNT) -> ModelInputs:
    phases = torch.arange(batch, dtype=torch.long) % PHASE_COUNT
    return ModelInputs(
        later_full_state=torch.full(
            (batch, STATE_SIZE), 1.0 / STATE_SIZE, dtype=torch.float32
        ),
        reverse_time=torch.linspace(0.1, 0.9, batch, dtype=torch.float64),
        phase=phases,
        color=torch.tensor(
            [PHASE_MATCHINGS[int(value)] for value in phases], dtype=torch.long
        ),
        duration=torch.tensor(
            [PHASE_DURATIONS[int(value)] for value in phases], dtype=torch.float64
        ),
        label=torch.full((batch,), 3, dtype=torch.long),
    )


def test_quadratic_identity_and_positive_ray_optimum_are_exact() -> None:
    cross = np.asarray([-1.0, 0.0, 1.5], dtype=np.float64)
    energy = np.asarray([2.0, 0.0, 0.5], dtype=np.float64)
    gain = 0.25
    expected = 2.0 * gain * cross - gain * gain * energy
    assert np.array_equal(quadratic_improvement(cross, energy, gain), expected)
    assert quadratic_improvement(1.5, 0.5, 2.0) == 4.0

    optimum = positive_ray_optimum(9.0, 1.5, 0.5)
    assert optimum == {"lambda_plus": 3.0, "D_plus": 4.5, "rho": math.sqrt(0.5)}
    assert positive_ray_optimum(9.0, -1.0, 2.0)["lambda_plus"] == 0.0
    assert positive_ray_optimum(0.0, 0.0, 0.0) == {
        "lambda_plus": 0.0,
        "D_plus": 0.0,
        "rho": 0.0,
    }
    with pytest.raises(QuartileDirectionalAdjudicationError, match="nonnegative"):
        quadratic_improvement(1.0, 1.0, -1.0)


def test_frozen_forward_decomposes_exactly_into_local_and_spatial_branches() -> None:
    model = ZeroBaselineBoundaryTangentPredictor()
    with torch.no_grad():
        model.residual_score.local_affine.bias.fill_(0.5)
        model.residual_score.spatial_output.bias.fill_(0.25)
    model.eval()

    inputs = _inputs()
    result = evaluate_frozen_components(model, inputs)
    production = model(inputs).to(dtype=torch.float64)

    assert tuple(result.as_mapping()) == COMPONENT_NAMES
    assert torch.equal(result.full, production)
    assert torch.equal(result.full, result.local_affine + result.spatial_cnn)
    assert torch.count_nonzero(result.local_affine) > 0
    assert torch.count_nonzero(result.spatial_cnn) > 0
    assert result.maximum_prediction_recomposition_error == 0.0
    assert result.maximum_spatial_rounding_error == 0.0

    with pytest.raises(QuartileDirectionalAdjudicationError, match="firewall"):
        evaluate_frozen_components(model, _inputs(33))


def _constant_predictions(rows: int) -> ComponentPredictions:
    shape = (rows, EDGES_PER_PHASE)
    full = torch.full(shape, 0.75, dtype=torch.float64)
    local = torch.full(shape, 0.5, dtype=torch.float64)
    spatial = torch.full(shape, 0.25, dtype=torch.float64)
    return ComponentPredictions(
        full=full,
        local_affine=local,
        spatial_cnn=spatial,
        q_full64=full,
        q_local64=local,
        q_spatial64_exact=spatial,
        q_spatial64_direct=spatial,
        rounding_bound=torch.zeros_like(full),
        maximum_prediction_recomposition_error=0.0,
        maximum_spatial_rounding_error=0.0,
    )


def test_streamed_accumulator_reconstructs_component_moments_and_marginals() -> None:
    path_ids = np.asarray([101, 102], dtype=np.int64)
    rows = [
        (path_id, phase, midpoint)
        for path_id in path_ids
        for phase in range(PHASE_COUNT)
        for midpoint in range(8)
    ]
    count = len(rows)
    accumulator = ComponentMomentAccumulator(path_ids)
    accumulator.add_batch(
        path_id=np.asarray([row[0] for row in rows], dtype=np.int64),
        phase=np.asarray([row[1] for row in rows], dtype=np.int64),
        midpoint=np.asarray([row[2] for row in rows], dtype=np.int64),
        target=torch.full((count, EDGES_PER_PHASE), 2.0, dtype=torch.float64),
        predictions=_constant_predictions(count),
    )
    cube = accumulator.finish()

    assert cube.target_energy.shape == (2, 7, 8)
    assert np.array_equal(cube.counts, np.ones((2, 7, 8), dtype=np.int64))
    assert np.all(cube.target_energy == 4.0)
    assert np.all(cube.cross_terms[0] == 1.5)
    assert np.all(cube.cross_terms[1] == 1.0)
    assert np.all(cube.cross_terms[2] == 0.5)
    assert np.all(cube.prediction_energies[0] == 0.75**2)
    assert np.all(cube.local_spatial_cross == 0.125)
    assert cube.maximum_recomposition_error == 0.0
    assert cube.maximum_risk_identity_error == 0.0
    assert not cube.cross_terms.flags.writeable

    summary = component_summary(cube, "full")
    assert summary["T"] == 4.0
    assert summary["C"] == 1.5
    assert summary["P"] == 0.75**2
    assert summary["lambda_plus"] == 1.5 / 0.75**2
    assert np.array_equal(summary["cell_C"], np.full((7, 8), 1.5))

    reduced = marginalize(cube.cross_terms[0], cube.counts)
    assert reduced["pooled"] == 1.5
    assert np.array_equal(reduced["path"], np.full(2, 1.5))
    assert np.array_equal(reduced["phase"], np.full(7, 1.5))
    assert np.array_equal(reduced["midpoint"], np.full(8, 1.5))
    assert weighted_pooled(np.asarray([1.0, 3.0]), np.asarray([1, 3])) == 2.5

    restored = ComponentMomentCube.from_arrays(cube.to_arrays())
    assert np.array_equal(restored.cross_terms, cube.cross_terms)


def test_cosine_and_cube_contracts_fail_closed() -> None:
    assert normalized_cosine([1.0, 2.0], [2.0, 4.0]) == 1.0
    assert normalized_cosine([1.0, 0.0], [-1.0, 0.0]) == -1.0
    assert normalized_cosine([0.0, 0.0], [0.0, 0.0]) == 1.0
    assert normalized_cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(QuartileDirectionalAdjudicationError, match="malformed"):
        normalized_cosine([1.0], [1.0, 2.0])

    shape = (1, 7, 8)
    with pytest.raises(QuartileDirectionalAdjudicationError, match="empty"):
        ComponentMomentAccumulator([1]).finish()
    with pytest.raises(QuartileDirectionalAdjudicationError, match="shape/content"):
        ComponentMomentCube(
            path_ids=np.asarray([1], dtype=np.int64),
            target_energy=np.ones(shape, dtype=np.float64),
            cross_terms=np.ones((3, *shape), dtype=np.float64),
            prediction_energies=np.ones((3, *shape), dtype=np.float64),
            local_spatial_cross=np.ones(shape, dtype=np.float64),
            counts=np.zeros(shape, dtype=np.int64),
        )
