from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest

from mnist import d0_jacobi_rb_haar as haar


def _midpoint(cell: haar.CertifiedInterval) -> float:
    return float(cell.midpoint)


def test_binary64_ball_contains_both_outward_interval_endpoints() -> None:
    cell = haar.CertifiedUniformCell(
        Fraction(1, 3) - Fraction(1, 1 << 180),
        Fraction(1, 3) + Fraction(1, 1 << 180),
    )
    hi, lo, radius = haar._binary64_ball_for_interval(cell)
    center = Fraction.from_float(hi) + Fraction.from_float(lo)
    exact_radius = Fraction.from_float(radius)
    lower, upper = cell.float_bounds()
    assert center - exact_radius <= Fraction.from_float(lower)
    assert center + exact_radius >= Fraction.from_float(upper)


def test_frozen_role_slots_and_twenty_bit_boundaries() -> None:
    report = haar.verify_haar_id_plan()
    assert report["passed"] == 1
    assert report["twenty_bit_bounds_pass"] == 1
    assert report["role_disjointness_pass"] == 1
    assert report["production_reservation_disjoint_pass"] == 1

    for role, (lower, upper) in haar.HAAR_ROLE_SLOTS.items():
        haar.validate_role_path_id(role, lower)
        haar.validate_role_path_id(role, upper - 1)
        with pytest.raises(ValueError, match="outside"):
            haar.validate_role_path_id(role, upper)
    with pytest.raises(TypeError, match="integer"):
        haar.validate_role_path_id("nested_a", True)
    with pytest.raises(ValueError, match="20-bit"):
        haar.validate_role_path_id("nested_a", 1 << 20)


def test_stateless_source_and_transition_ids_are_stable_and_separated() -> None:
    path = haar.path_ids_for_role("nested_a", 1)[0]
    root = haar.canonical_haar_source_id(
        role="nested_a",
        path_id=path,
        coarsest_step=3,
        phase=2,
        edge_id=17,
        depth=0,
        node=3,
        kind="root",
    )
    assert root == haar.canonical_haar_source_id(
        role="nested_a",
        path_id=path,
        coarsest_step=3,
        phase=2,
        edge_id=17,
        depth=0,
        node=3,
        kind="root",
    )
    detail = haar.canonical_haar_source_id(
        role="nested_a",
        path_id=path,
        coarsest_step=3,
        phase=2,
        edge_id=17,
        depth=1,
        node=3,
        kind="detail",
    )
    assert detail != root
    event = haar.HaarEventIdentity("nested_a", path, 256, 6, 2, 17, 1)
    assert haar.canonical_haar_transition_id(event) != (
        haar.canonical_haar_transition_id(
            haar.HaarEventIdentity("nested_a", path, 256, 7, 2, 17, 1)
        )
    )
    with pytest.raises(ValueError, match="root coordinates"):
        haar.canonical_haar_source_id(
            role="nested_a",
            path_id=path,
            coarsest_step=3,
            phase=2,
            edge_id=17,
            depth=0,
            node=4,
            kind="root",
        )


def test_structural_ids_round_trip_injectively_and_reject_overflow() -> None:
    sources: set[int] = set()
    transitions: set[int] = set()
    for role in haar.HAAR_ROLE_SLOTS:
        path = haar.path_ids_for_role(role, 2)[1]
        for edge in (0, 391):
            source = haar.canonical_haar_source_id(
                role=role,
                path_id=path,
                coarsest_step=127,
                phase=6,
                edge_id=edge,
                depth=1,
                node=127,
                kind="detail",
                tree_root_steps=1024,
            )
            assert source not in sources
            sources.add(source)
            decoded_source = haar.unpack_haar_source_id(source)
            assert decoded_source["role"] == role
            assert decoded_source["path_id"] == path
            assert decoded_source["edge_id"] == edge
            assert decoded_source["tree_root_steps"] == 1024

            event = haar.HaarEventIdentity(
                role, path, 2048, 2047, 6, edge, -1, 1024
            )
            transition = haar.canonical_haar_transition_id(event)
            assert transition not in transitions
            transitions.add(transition)
            decoded_transition = haar.unpack_haar_transition_id(transition)
            assert decoded_transition["sample_steps"] == 2048
            assert decoded_transition["outer_step"] == 2047
            assert decoded_transition["arm"] == -1
    assert sources.isdisjoint(transitions)
    with pytest.raises(ValueError, match="edge_id"):
        haar.canonical_haar_source_id(
            role="nested_a",
            path_id=haar.path_ids_for_role("nested_a", 1)[0],
            coarsest_step=0,
            phase=0,
            edge_id=512,
            depth=0,
            node=0,
            kind="root",
        )
    with pytest.raises(ValueError, match="outer_step"):
        haar.HaarEventIdentity(
            "nested_a",
            haar.path_ids_for_role("nested_a", 1)[0],
            2048,
            2048,
            0,
            0,
        )


def test_haar_split_is_orthogonal_and_antithetic_swap_is_exact() -> None:
    rng = np.random.default_rng(261181)
    parent = rng.standard_normal(200_000)
    detail = rng.standard_normal(200_000)
    left, right = haar.haar_split(parent, detail)
    np.testing.assert_allclose(haar.haar_parent(left, right), parent, rtol=0, atol=1e-15)
    np.testing.assert_allclose(haar.haar_detail(left, right), detail, rtol=0, atol=1e-15)
    swapped_left, swapped_right = haar.haar_split(parent, -detail)
    np.testing.assert_allclose(swapped_left, right, rtol=0, atol=0)
    np.testing.assert_allclose(swapped_right, left, rtol=0, atol=0)
    covariance = np.cov(np.stack((left, right)))
    assert abs(covariance[0, 1]) < 0.01
    assert abs(covariance[0, 0] - 1.0) < 0.01
    assert abs(covariance[1, 1] - 1.0) < 0.01


@pytest.mark.skipif(haar.arb is None, reason="python-flint/Arb unavailable")
def test_certified_nested_children_reconstruct_parent_and_antithetic_swap() -> None:
    profile = haar.HaarCouplingProfile()
    path = haar.path_ids_for_role("marginal_c", 1)
    common = dict(
        root_seed=261181,
        role="marginal_c",
        path_ids=path,
        phase=1,
        edge_ids=[9],
        profile=profile,
    )
    parent = haar.build_certified_haar_uniform_batch(
        sample_steps=128, outer_step=4, **common
    )
    child_left = haar.build_certified_haar_uniform_batch(
        sample_steps=256, outer_step=8, detail_sign=1, **common
    )
    child_right = haar.build_certified_haar_uniform_batch(
        sample_steps=256, outer_step=9, detail_sign=1, **common
    )
    reconstructed = haar.haar_parent(
        [_midpoint(child_left.normal_cells[0])],
        [_midpoint(child_right.normal_cells[0])],
    )[0]
    assert math.isclose(
        reconstructed, _midpoint(parent.normal_cells[0]), rel_tol=0, abs_tol=2e-15
    )
    assert int(parent.source_prefix_ids.item()) == int(
        child_left.source_prefix_ids.item()
    )
    assert parent.runtime_report["arb_authorizing"] is True
    assert parent.runtime_report["torch_normal_authorizing"] is False
    assert parent.diagnostics["certificate_count"] == 1


@pytest.mark.skipif(haar.arb is None, reason="python-flint/Arb unavailable")
@pytest.mark.parametrize("coarse_steps", [128, 256, 512, 1024])
def test_each_adjacent_pair_uses_one_local_antithetic_detail(
    coarse_steps: int,
) -> None:
    profile = haar.HaarCouplingProfile()
    common = dict(
        root_seed=261181,
        role="antithetic_a",
        path_ids=haar.path_ids_for_role("antithetic_a", 1),
        phase=3,
        edge_ids=[11],
        profile=profile,
        pair_coarse_steps=coarse_steps,
    )
    coarse = haar.build_certified_haar_uniform_batch(
        sample_steps=coarse_steps, outer_step=3, **common
    )
    plus_left = haar.build_certified_haar_uniform_batch(
        sample_steps=2 * coarse_steps,
        outer_step=6,
        detail_sign=1,
        **common,
    )
    plus_right = haar.build_certified_haar_uniform_batch(
        sample_steps=2 * coarse_steps,
        outer_step=7,
        detail_sign=1,
        **common,
    )
    minus_left = haar.build_certified_haar_uniform_batch(
        sample_steps=2 * coarse_steps,
        outer_step=6,
        detail_sign=-1,
        **common,
    )
    minus_right = haar.build_certified_haar_uniform_batch(
        sample_steps=2 * coarse_steps,
        outer_step=7,
        detail_sign=-1,
        **common,
    )
    assert minus_left.uniform_cells[0] == plus_right.uniform_cells[0]
    assert minus_right.uniform_cells[0] == plus_left.uniform_cells[0]
    assert int(coarse.source_prefix_ids.item()) == int(
        plus_left.source_prefix_ids.item()
    )
    assert plus_left.diagnostics["pairwise_local_detail_count"] == 1
    assert plus_left.diagnostics["tree_root_steps"] == coarse_steps


@pytest.mark.skipif(haar.arb is None, reason="python-flint/Arb unavailable")
def test_order_and_chunking_do_not_change_certified_cells() -> None:
    profile = haar.HaarCouplingProfile()
    paths = haar.path_ids_for_role("marginal_d", 2)
    kwargs = dict(
        root_seed=("seed", 261181),
        role="marginal_d",
        sample_steps=1024,
        outer_step=17,
        phase=5,
        profile=profile,
    )
    batch = haar.build_certified_haar_uniform_batch(
        path_ids=paths, edge_ids=[2, 7], **kwargs
    )
    reversed_batch = haar.build_certified_haar_uniform_batch(
        path_ids=paths[::-1], edge_ids=[7, 2], **kwargs
    )
    expected = {
        (path, edge): batch.uniform_cells[path_index * 2 + edge_index]
        for path_index, path in enumerate(paths)
        for edge_index, edge in enumerate((2, 7))
    }
    observed = {
        (path, edge): reversed_batch.uniform_cells[path_index * 2 + edge_index]
        for path_index, path in enumerate(paths[::-1])
        for edge_index, edge in enumerate((7, 2))
    }
    assert observed == expected
    assert batch.diagnostics["transition_id_collision_count"] == 0
    assert batch.diagnostics["source_id_collision_count"] == 0


def test_interval_records_and_level_validation_fail_closed() -> None:
    with pytest.raises(ValueError, match="reversed"):
        haar.CertifiedInterval(Fraction(2, 3), Fraction(1, 3))
    with pytest.raises(ValueError, match="strictly"):
        haar.CertifiedUniformCell(Fraction(0), Fraction(1, 2))
    profile = haar.HaarCouplingProfile()
    with pytest.raises(ValueError, match="dyadic"):
        haar._validate_level(profile, 384)
    assert haar.haar_ancestor_step(2048, 31) == 1


@pytest.mark.skipif(haar.arb is None, reason="python-flint/Arb unavailable")
@pytest.mark.parametrize(
    ("numerator", "bits"),
    [(1, 64), ((1 << 63) - 1, 64), ((1 << 64) - 2, 64)],
)
def test_public_extreme_prefix_oracle_certifies_normal_round_trip(
    numerator: int, bits: int
) -> None:
    result = haar.certify_normal_uniform_from_prefix(
        numerator, bits, haar.HaarCouplingProfile()
    )
    source_lower = Fraction(numerator, 1 << bits)
    source_upper = Fraction(numerator + 1, 1 << bits)
    assert result.uniform.lower <= source_lower
    assert result.uniform.upper >= source_upper
    assert result.normal.lower < result.normal.upper
    assert result.backend == "python-flint/Arb"
    with pytest.raises(haar.HaarCertificationError, match="facet"):
        haar.certify_normal_uniform_from_prefix(
            0, bits, haar.HaarCouplingProfile()
        )


@pytest.mark.skipif(haar.arb is None, reason="python-flint/Arb unavailable")
def test_refinement_callback_is_versioned_nested_and_capped() -> None:
    batch = haar.build_certified_haar_uniform_batch(
        root_seed=261181,
        role="marginal_c",
        path_ids=haar.path_ids_for_role("marginal_c", 1),
        sample_steps=512,
        outer_step=17,
        phase=2,
        edge_ids=[4],
        profile=haar.HaarCouplingProfile(),
    )
    response = batch.refinement_callback(
        haar.UniformCellRefinementRequest(
            sample_index=0,
            requested_source_prefix_bits=192,
            current_cell=batch.uniform_cells[0],
        )
    )
    assert response.source_prefix_bits >= 192
    assert response.cell.lower >= batch.uniform_cells[0].lower
    assert response.cell.upper <= batch.uniform_cells[0].upper
    assert response.cell.width < batch.uniform_cells[0].width
    with pytest.raises(ValueError, match=r"\[1,1024\]"):
        haar.UniformCellRefinementRequest(
            0, 1025, batch.uniform_cells[0]
        )
