from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mnist.d0_jacobi_rb_absolute_coordinate import (
    ABSOLUTE_COORDINATE_VERSION,
    COORDINATE_COMPONENTS,
    ADirectionSeal,
    AbsoluteCoordinateError,
    BLinearEvidence,
    CoordinatePanel,
    build_coordinate_lattice,
    coordinate_family_names,
    decompose_cross_panel_signal,
    edge_translation_permutation,
    evaluate_panel_b_linear,
    model_translation_equivariance_record,
    one_sided_b_linear_max_t,
    phase_predictor_architecture_contract,
    project_coordinate_component,
    project_coordinate_components,
    scaled_signed_cross_bounds,
    seal_panel_a_directions,
    synthetic_coordinate_fixture,
    synthetic_model_inputs,
)
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    PHASE_COUNT,
    PHASE_MATCHINGS,
    PhaseConditionedLocalAffineCNN,
)


def test_coordinate_lattice_is_nested_orthonormal_and_reproducible() -> None:
    lattice = build_coordinate_lattice()
    repeated = build_coordinate_lattice()

    assert lattice.version == ABSOLUTE_COORDINATE_VERSION
    assert lattice.basis_sha256 == repeated.basis_sha256
    assert lattice.phase_colors.tolist() == list(PHASE_MATCHINGS)
    assert lattice.dc_basis.shape == (PHASE_COUNT, EDGES_PER_PHASE, 1)
    assert lattice.frequency1_basis.shape == (PHASE_COUNT, EDGES_PER_PHASE, 4)
    assert lattice.frequency2_basis.shape == (PHASE_COUNT, EDGES_PER_PHASE, 4)
    assert lattice.maximum_gram_error <= 5e-14

    for phase in range(PHASE_COUNT):
        basis = np.concatenate(
            (
                lattice.dc_basis[phase],
                lattice.frequency1_basis[phase],
                lattice.frequency2_basis[phase],
            ),
            axis=1,
        )
        np.testing.assert_allclose(
            basis.T @ basis,
            np.eye(9, dtype=np.float64),
            rtol=0.0,
            atol=5e-14,
        )
    for first, second in ((0, 6), (1, 5), (2, 4)):
        np.testing.assert_array_equal(
            lattice.head_indices[first], lattice.head_indices[second]
        )
        np.testing.assert_array_equal(lattice.dc_basis[first], lattice.dc_basis[second])
        np.testing.assert_array_equal(
            lattice.frequency1_basis[first], lattice.frequency1_basis[second]
        )
        np.testing.assert_array_equal(
            lattice.frequency2_basis[first], lattice.frequency2_basis[second]
        )

    record = lattice.to_record()
    assert record["basis_sha256"] == lattice.basis_sha256
    assert record["residual_rank"] == EDGES_PER_PHASE - 9


def test_coordinate_projection_reconstructs_and_is_orthogonal() -> None:
    lattice = build_coordinate_lattice()
    generator = np.random.Generator(np.random.Philox(12))
    values = generator.standard_normal((3, 4, PHASE_COUNT, EDGES_PER_PHASE))
    components = project_coordinate_components(values, lattice=lattice)

    assert tuple(components) == COORDINATE_COMPONENTS
    reconstruction = sum(components.values())
    np.testing.assert_allclose(reconstruction, values, rtol=0.0, atol=5e-15)

    flattened = {name: value.reshape(-1) for name, value in components.items()}
    for left_index, left in enumerate(COORDINATE_COMPONENTS):
        for right in COORDINATE_COMPONENTS[left_index + 1 :]:
            scale = max(
                1.0,
                np.linalg.norm(flattened[left]) * np.linalg.norm(flattened[right]),
            )
            assert abs(float(flattened[left] @ flattened[right])) / scale <= 5e-15
    for component in COORDINATE_COMPONENTS:
        projected = project_coordinate_component(
            components[component], component, lattice=lattice
        )
        np.testing.assert_allclose(
            projected, components[component], rtol=0.0, atol=5e-14
        )


def test_cross_panel_decomposition_reconstructs_signed_signal() -> None:
    fixture = synthetic_coordinate_fixture(path_count=12, noise_scale=0.0)
    decomposition = decompose_cross_panel_signal(fixture.left, fixture.right)

    expected = np.asarray(
        [fixture.component_amplitudes[name] for name in COORDINATE_COMPONENTS],
        dtype=np.float64,
    ) ** 2
    np.testing.assert_allclose(
        decomposition.component_point_energies, expected, rtol=0.0, atol=2e-15
    )
    np.testing.assert_allclose(
        decomposition.full_point_energies,
        np.sum(expected, axis=0),
        rtol=0.0,
        atol=2e-15,
    )
    assert decomposition.maximum_reconstruction_error <= 2e-15

    # A sign reversal on B remains negative.  The decomposition never takes an
    # absolute value or truncates a scientific estimate at zero.
    negative_b = CoordinatePanel(
        role="negative-panel-b",
        path_ids=np.arange(0x10200, 0x1020C, dtype=np.int64),
        cell_means=np.ascontiguousarray(-fixture.right.cell_means),
    )
    negative = decompose_cross_panel_signal(fixture.left, negative_b)
    np.testing.assert_allclose(
        negative.component_point_energies, -expected, rtol=0.0, atol=2e-15
    )
    assert np.all(negative.component_point_energies < 0.0)


def test_panel_a_seal_and_panel_b_linear_evidence_match_cross_energy() -> None:
    fixture = synthetic_coordinate_fixture(path_count=20, noise_scale=0.02)
    decomposition = decompose_cross_panel_signal(fixture.left, fixture.right)
    seal = seal_panel_a_directions(fixture.left)
    evidence = evaluate_panel_b_linear(seal, fixture.right)

    expected = decomposition.component_point_energies.T.reshape(-1)
    np.testing.assert_allclose(
        evidence.signed_cross_energies, expected, rtol=0.0, atol=3e-15
    )
    assert evidence.a_direction_seal_sha256 == seal.seal_sha256
    assert seal.family_names == coordinate_family_names()
    assert np.all(seal.direction_active_mask)
    assert not seal.directions.flags.writeable

    inference = one_sided_b_linear_max_t(
        evidence,
        confidence=0.90,
        replicates=256,
        seed=88,
        namespace=99,
        chunk_size=64,
    )
    repeated = one_sided_b_linear_max_t(
        evidence,
        confidence=0.90,
        replicates=256,
        seed=88,
        namespace=99,
        chunk_size=31,
    )
    np.testing.assert_array_equal(inference.bootstrap_maxima, repeated.bootstrap_maxima)
    np.testing.assert_array_equal(inference.lower_bounds, repeated.lower_bounds)
    np.testing.assert_allclose(
        scaled_signed_cross_bounds(seal, inference),
        seal.direction_norms * inference.lower_bounds,
        rtol=0.0,
        atol=0.0,
    )


def test_zero_a_direction_is_an_analytic_constant_member() -> None:
    cells = np.zeros((8, 4, PHASE_COUNT, EDGES_PER_PHASE), dtype=np.float64)
    panel_a = CoordinatePanel(
        role="zero-a",
        path_ids=np.arange(0x11000, 0x11008, dtype=np.int64),
        cell_means=cells,
    )
    panel_b = CoordinatePanel(
        role="zero-b",
        path_ids=np.arange(0x11100, 0x11108, dtype=np.int64),
        cell_means=cells.copy(),
    )
    seal = seal_panel_a_directions(panel_a, components=("frequency1",))
    assert not seal.direction_active_mask.any()
    evidence = evaluate_panel_b_linear(seal, panel_b)
    inference = one_sided_b_linear_max_t(
        evidence, replicates=64, seed=1, namespace=2, chunk_size=17
    )

    np.testing.assert_array_equal(inference.point_estimates, np.zeros(4))
    np.testing.assert_array_equal(inference.lower_bounds, np.zeros(4))
    assert inference.analytic_constant_mask.all()
    assert inference.critical_value == 0.0


def test_max_t_retains_negative_point_estimates() -> None:
    path_ids = np.arange(0x12000, 0x12010, dtype=np.int64)
    positive = np.linspace(0.1, 0.2, path_ids.size, dtype=np.float64)
    negative = -np.linspace(0.2, 0.4, path_ids.size, dtype=np.float64)
    values = np.stack((positive, negative), axis=1)
    evidence = BLinearEvidence(
        family_names=("positive", "negative"),
        panel_b_role="synthetic-b",
        path_ids=path_ids,
        path_values=values,
        point_estimates=np.mean(values, axis=0),
        direction_norms=np.ones(2, dtype=np.float64),
        signed_cross_energies=np.mean(values, axis=0),
        a_direction_seal_sha256="1" * 64,
    )
    result = one_sided_b_linear_max_t(
        evidence, confidence=0.90, replicates=128, seed=9, namespace=8
    )
    assert result.point_estimates[0] > 0.0
    assert result.point_estimates[1] < 0.0
    assert result.lower_bounds[1] < result.point_estimates[1] < 0.0
    assert result.to_record()["negative_values_truncated"] == 0


def test_coordinate_panel_and_family_contracts_fail_closed() -> None:
    valid_cells = np.zeros((3, 4, PHASE_COUNT, EDGES_PER_PHASE), dtype=np.float64)
    with pytest.raises(AbsoluteCoordinateError, match="canonical"):
        CoordinatePanel(
            role="bad-order",
            path_ids=np.asarray([3, 1, 2], dtype=np.int64),
            cell_means=valid_cells,
        )
    with pytest.raises(AbsoluteCoordinateError, match="finite binary64"):
        project_coordinate_components(
            np.zeros((4, PHASE_COUNT, EDGES_PER_PHASE), dtype=np.float32)
        )
    with pytest.raises(AbsoluteCoordinateError, match="malformed"):
        coordinate_family_names(("frequency1", "frequency1"))

    panel_a = CoordinatePanel(
        role="a",
        path_ids=np.arange(10, 13, dtype=np.int64),
        cell_means=valid_cells,
    )
    panel_b = CoordinatePanel(
        role="b",
        path_ids=np.arange(12, 15, dtype=np.int64),
        cell_means=valid_cells,
    )
    with pytest.raises(AbsoluteCoordinateError, match="independent"):
        decompose_cross_panel_signal(panel_a, panel_b)


def test_model_contract_and_translation_equivariance() -> None:
    torch.manual_seed(44)
    model = PhaseConditionedLocalAffineCNN(width=32)
    with torch.no_grad():
        model.spatial_output.weight.normal_(mean=0.0, std=0.02)
        model.spatial_output.bias.normal_(mean=0.0, std=0.02)
    contract = phase_predictor_architecture_contract(model)
    assert contract["passed"] == 1
    assert contract["effective_receptive_field"] == 7
    assert contract["periodic_coordinate_buffers"] == []

    record = model_translation_equivariance_record(
        model,
        synthetic_model_inputs(batch_size=7),
        row_shift=2,
        column_shift=2,
        tolerance=2e-6,
    )
    assert record["passed"] == 1
    assert record["maximum_translation_equivariance_error"] <= 2e-6

    for color in range(4):
        permutation = edge_translation_permutation(color, 2, 2)
        np.testing.assert_array_equal(np.sort(permutation), np.arange(EDGES_PER_PHASE))
    with pytest.raises(AbsoluteCoordinateError, match="does not preserve"):
        edge_translation_permutation(0, 0, 1)


def test_seal_constructor_rejects_non_unit_active_direction() -> None:
    families = coordinate_family_names(("dc",))
    with pytest.raises(AbsoluteCoordinateError, match="unit RMS"):
        ADirectionSeal(
            family_names=families,
            components=("dc",),
            panel_a_role="a",
            panel_a_path_ids=np.arange(4, dtype=np.int64),
            directions=np.full(
                (4, PHASE_COUNT, EDGES_PER_PHASE), 2.0, dtype=np.float64
            ),
            direction_norms=np.ones(4, dtype=np.float64),
            direction_active_mask=np.ones(4, dtype=np.bool_),
            basis_sha256="2" * 64,
            panel_a_sha256="3" * 64,
        )


def test_fixture_amplitudes_have_unit_rms_coordinate_directions() -> None:
    fixture = synthetic_coordinate_fixture(path_count=4, noise_scale=0.0)
    signal_energy = np.mean(fixture.signal * fixture.signal, axis=(1, 2))
    expected = np.sum(
        np.asarray(
            [fixture.component_amplitudes[name] for name in COORDINATE_COMPONENTS],
            dtype=np.float64,
        )
        ** 2,
        axis=0,
    )
    np.testing.assert_allclose(signal_energy, expected, rtol=0.0, atol=2e-15)
    assert math.isfinite(float(np.sum(signal_energy)))
