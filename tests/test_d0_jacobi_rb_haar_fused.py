from __future__ import annotations

from fractions import Fraction
import json
import math
from typing import Any

import numpy as np
import pytest
import torch

from mnist import d0_jacobi_rb_cuda as rb_cuda
from mnist import d0_jacobi_rb_haar as haar
from mnist import d0_jacobi_rb_haar_cuda as haar_cuda
from mnist import d0_jacobi_rb_haar_fused as fused
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile


_ROOT_SEED = ("seed", 261181)
_NORMAL_CASES = (
    # sample_steps, outer_step, phase, edge, detail_sign
    (128, 3, 0, 1, 1),
    (256, 11, 1, 2, -1),
    (512, 37, 2, 3, 1),
    (1024, 333, 3, 4, -1),
    (2048, 1025, 4, 5, 1),
)
_JACOBI_X = np.array([0.2, 0.37, 0.8, 0.6], dtype=np.float64)
_JACOBI_EXPOSURE = np.array([0.1, 0.5, 1.0, 0.0], dtype=np.float64)
_JACOBI_CENTER_HI = np.array([0.2, 0.5, 0.8, 0.3], dtype=np.float64)
_JACOBI_CENTER_LO = np.array(
    [
        float.fromhex("0x1.23456789abcdep-100"),
        -float.fromhex("0x1.3579bdf2468acp-101"),
        float.fromhex("0x1.abcdef0123456p-100"),
        float.fromhex("0x1.13579bdf02468p-100"),
    ],
    dtype=np.float64,
)
_JACOBI_RADIUS = float.fromhex("0x1p-180")
_JACOBI_IDS = np.array([101, 102, 103, 104], dtype=np.uint64)


def _arb_available() -> bool:
    return (
        haar.arb is not None
        and haar.flint_ctx is not None
        and str(getattr(haar.flint, "__version__", "")) == "0.9.0"
    )


def _cuda_arb_available() -> bool:
    return bool(torch.cuda.is_available() and _arb_available())


def _frozen_jacobi_cuda_available() -> bool:
    if not _cuda_arb_available():
        return False
    properties = torch.cuda.get_device_properties(0)
    return (
        str(torch.__version__) == "2.11.0+cu128"
        and str(torch.version.cuda) == "12.8"
        and (properties.major, properties.minor) == (12, 0)
    )


def _cuda_tensor(values: Any, dtype: Any = torch.float64) -> torch.Tensor:
    return torch.as_tensor(values, dtype=dtype, device="cuda").contiguous()


def _normal_panel(
    order: tuple[int, ...] | None = None,
) -> tuple[
    tuple[haar.HaarEventIdentity, ...],
    tuple[int, ...],
    fused.HaarFusedLaunch,
]:
    path_id = haar.path_ids_for_role("marginal_c", 1)[0]
    events = tuple(
        haar.HaarEventIdentity(
            "marginal_c",
            path_id,
            sample_steps,
            outer_step,
            phase,
            edge,
        )
        for sample_steps, outer_step, phase, edge, _sign in _NORMAL_CASES
    )
    signs = tuple(case[-1] for case in _NORMAL_CASES)
    if order is not None:
        events = tuple(events[index] for index in order)
        signs = tuple(signs[index] for index in order)

    source_id_sets: list[tuple[int, ...]] = []
    depths: list[int] = []
    for event in events:
        source_ids, depth, _coarsest_step = haar._source_ids_for_event(event)
        source_id_sets.append(source_ids)
        depths.append(depth)
    source_matrix = [
        list(source_ids)
        + [0] * (fused.HAAR_NORMAL_MAX_SOURCES - len(source_ids))
        for source_ids in source_id_sets
    ]
    launch = fused.launch_certified_haar_normal_transform(
        _cuda_tensor(source_matrix, torch.uint64),
        _cuda_tensor(depths, torch.int32),
        _cuda_tensor([event.outer_step for event in events], torch.int32),
        _cuda_tensor(signs, torch.int32),
        root_seed=_ROOT_SEED,
    )
    return events, signs, launch


def _jacobi_cells(
    order: tuple[int, ...] | None = None,
) -> tuple[haar.CertifiedUniformCell, ...]:
    indices = tuple(range(_JACOBI_X.size)) if order is None else order
    radius = Fraction.from_float(_JACOBI_RADIUS)
    return tuple(
        haar.CertifiedUniformCell(
            Fraction.from_float(float(_JACOBI_CENTER_HI[index]))
            + Fraction.from_float(float(_JACOBI_CENTER_LO[index]))
            - radius,
            Fraction.from_float(float(_JACOBI_CENTER_HI[index]))
            + Fraction.from_float(float(_JACOBI_CENTER_LO[index]))
            + radius,
        )
        for index in indices
    )


def _jacobi_panel(
    order: tuple[int, ...] | None = None,
) -> tuple[fused.HaarUniformFusedLaunch, tuple[haar.CertifiedUniformCell, ...]]:
    indices = np.arange(_JACOBI_X.size) if order is None else np.asarray(order)
    center_hi = _JACOBI_CENTER_HI[indices]
    cells = _jacobi_cells(
        None if order is None else tuple(int(index) for index in indices)
    )
    bounds = [cell.float_bounds() for cell in cells]
    uniform_lower = np.asarray([value[0] for value in bounds])
    uniform_upper = np.asarray([value[1] for value in bounds])
    launch = fused.launch_certified_jacobi_from_uniform_cells(
        _cuda_tensor(_JACOBI_X[indices]),
        _cuda_tensor(_JACOBI_EXPOSURE[indices]),
        _cuda_tensor(uniform_lower),
        _cuda_tensor(uniform_upper),
        _cuda_tensor(_JACOBI_IDS[indices], torch.uint64),
        uniform_center_hi=_cuda_tensor(center_hi),
        uniform_center_lo=_cuda_tensor(_JACOBI_CENTER_LO[indices]),
        uniform_radius=_cuda_tensor(
            np.full(indices.size, _JACOBI_RADIUS, dtype=np.float64)
        ),
        mode_cap=8192,
    )
    return launch, cells


def _host(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


@pytest.mark.skipif(not _arb_available(), reason="python-flint==0.9.0 unavailable")
def test_frozen_normal_constants_and_source_binding() -> None:
    report = fused.verify_normal_constants()
    json.dumps(report, allow_nan=False)

    assert report["passed"] == 1
    assert report["errors"] == []
    assert report["version"] == "d0-jacobi-rb-haar-normal-dd-quarter-anchor-v1"
    assert report["anchor_count"] == 305
    assert report["series_degree"] == 56
    assert report["series_remainder_hex"] == "0x1.0p-120"
    assert report["analytic_tail_bound"] < float.fromhex(
        report["series_remainder_hex"]
    )
    assert report["constants_sha256"] == (
        "bc887be3ea8447d533a2de2b8a11308fadb4059c2fa9b54e3d4e3e3d133b5428"
    )
    assert len(report["source_sha256"]) == 64
    int(report["source_sha256"], 16)

    assert fused.HAAR_NORMAL_SOURCE_BITS == 128
    assert fused.HAAR_NORMAL_MAX_SOURCES == 5
    assert fused.HAAR_NORMAL_ANCHOR_QUARTERS == 152
    assert fused.HAAR_NORMAL_SERIES_DEGREE == 56
    assert fused.HAAR_NORMAL_SERIES_REMAINDER_HEX == "0x1.0p-120"
    assert fused.HAAR_NORMAL_REQUIRED_SELFTEST_MASK == 0x1F
    assert fused.HAAR_FUSED_COMPILE_FLAGS == (
        "--std=c++17",
        "--fmad=false",
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    )
    assert fused.verify_normal_constants() == report


def test_canonical_haar_seed_is_cpu_visible_stable_and_compatible() -> None:
    assert fused.canonical_haar_seed(0) == 0x3ACAF20C33B5472B
    assert fused.canonical_haar_seed(261181) == 0xF798E923217C6420
    assert fused.canonical_haar_seed(_ROOT_SEED) == 0x4085DAAC867F66F0
    assert fused.canonical_haar_seed({"b": 2, "a": 1}) == 0xDF7EC3680207C2E2
    assert fused.canonical_haar_seed({"a": 1, "b": 2}) == fused.canonical_haar_seed(
        {"b": 2, "a": 1}
    )
    assert fused.canonical_haar_seed(b"abc") != fused.canonical_haar_seed("abc")

    for seed in (0, 261181, _ROOT_SEED, {"a": 1, "b": 2}, b"abc", "abc"):
        assert fused.canonical_haar_seed(seed) == rb_cuda._canonical_seed(seed)

    with pytest.raises(ValueError, match="uint64"):
        fused.canonical_haar_seed(-1)
    with pytest.raises(ValueError, match="uint64"):
        fused.canonical_haar_seed(1 << 64)
    with pytest.raises(TypeError, match="canonical JSON"):
        fused.canonical_haar_seed(float("nan"))
    with pytest.raises(TypeError, match="canonical JSON"):
        fused.canonical_haar_seed(object())


@pytest.mark.skipif(not _cuda_arb_available(), reason="CUDA/Arb runtime unavailable")
def test_cuda_normal_panel_is_certified_against_arb() -> None:
    events, signs, launch = _normal_panel()
    assert launch.certificate_mask.detach().cpu().tolist() == [True] * len(events)
    assert launch.fallback_reason_codes.detach().cpu().tolist() == [0] * len(events)
    assert launch.prefix_bits.detach().cpu().tolist() == [128] * len(events)
    assert launch.bundle.selftest_mask == fused.HAAR_NORMAL_REQUIRED_SELFTEST_MASK

    profile = haar.HaarCouplingProfile()
    for index, (event, sign) in enumerate(zip(events, signs, strict=True)):
        arb_normal, arb_uniform, _source_ids, prefix_bits = haar._certify_event(
            event,
            root_seed=_ROOT_SEED,
            profile=profile,
            detail_sign=sign,
        )
        assert prefix_bits == 128

        normal_lower = Fraction.from_float(float(launch.normal_lower[index]))
        normal_upper = Fraction.from_float(float(launch.normal_upper[index]))
        uniform_lower = Fraction.from_float(float(launch.uniform_lower[index]))
        uniform_upper = Fraction.from_float(float(launch.uniform_upper[index]))
        assert normal_lower <= arb_normal.lower <= arb_normal.upper <= normal_upper
        assert uniform_lower <= arb_uniform.lower <= arb_uniform.upper <= uniform_upper

        normal_center = Fraction.from_float(
            float(launch.normal_center_hi[index])
        ) + Fraction.from_float(float(launch.normal_center_lo[index]))
        normal_radius = Fraction.from_float(float(launch.normal_radius[index]))
        uniform_center = Fraction.from_float(
            float(launch.uniform_center_hi[index])
        ) + Fraction.from_float(float(launch.uniform_center_lo[index]))
        uniform_radius = Fraction.from_float(float(launch.uniform_radius[index]))
        assert normal_lower <= normal_center - normal_radius
        assert normal_center + normal_radius <= normal_upper
        assert uniform_lower <= uniform_center - uniform_radius
        assert uniform_center + uniform_radius <= uniform_upper


@pytest.mark.skipif(not _cuda_arb_available(), reason="CUDA/Arb runtime unavailable")
def test_injected_extreme_prefixes_are_fused_certified_against_arb() -> None:
    numerators = (
        2,
        1 << 32,
        1 << 127,
        (1 << 128) - (1 << 32) - 1,
        (1 << 128) - 3,
    )
    mask = (1 << 64) - 1
    launch = fused.launch_certified_normal_prefix_transform(
        _cuda_tensor([value >> 64 for value in numerators], torch.uint64),
        _cuda_tensor([value & mask for value in numerators], torch.uint64),
    )
    certified = launch.certificate_mask.detach().cpu().tolist()
    reasons = launch.fallback_reason_codes.detach().cpu().tolist()
    assert sum(certified) >= 2
    profile = haar.HaarCouplingProfile()
    for index, numerator in enumerate(numerators):
        reference = haar.certify_normal_uniform_from_prefix(
            numerator, 128, profile
        )
        if certified[index]:
            assert Fraction.from_float(
                float(launch.normal_lower[index])
            ) <= reference.normal.lower
            assert Fraction.from_float(
                float(launch.normal_upper[index])
            ) >= reference.normal.upper
            source_lower = Fraction(numerator, 1 << 128)
            source_upper = Fraction(numerator + 1, 1 << 128)
            fused_lower = Fraction.from_float(
                float(launch.uniform_lower[index])
            )
            fused_upper = Fraction.from_float(
                float(launch.uniform_upper[index])
            )
            # Phi(Phi^-1(U)) is exactly U.  Both independent enclosures must
            # contain that exact prefix cell; neither must contain the
            # other's dependency overestimation.
            assert fused_lower <= source_lower <= source_upper <= fused_upper
            assert fused_lower <= reference.uniform.upper
            assert fused_upper >= reference.uniform.lower
        else:
            assert reasons[index] in {2, 3, 5}
            source_lower = Fraction(numerator, 1 << 128)
            source_upper = Fraction(numerator + 1, 1 << 128)
            assert reference.uniform.lower <= source_lower
            assert reference.uniform.upper >= source_upper


@pytest.mark.skipif(not _cuda_arb_available(), reason="CUDA/Arb runtime unavailable")
def test_production_batch_keeps_certified_cells_on_device() -> None:
    path = haar.path_ids_for_role("marginal_c", 1)
    batch = haar.build_certified_haar_uniform_batch(
        root_seed=_ROOT_SEED,
        role="marginal_c",
        path_ids=path,
        sample_steps=128,
        outer_step=0,
        phase=0,
        edge_ids=range(8),
        profile=haar.HaarCouplingProfile(),
        device="cuda",
    )
    assert batch.uniform_cells == ()
    assert batch.normal_cells == ()
    assert batch.uniform_lower.is_cuda
    assert batch.normal_lower.is_cuda
    assert batch.runtime_report["host_interval_materialization_count"] == 0
    assert batch.runtime_report["device_resident_certified_output"] == 1


@pytest.mark.skipif(
    not _frozen_jacobi_cuda_available(),
    reason="frozen Jacobi CUDA/Arb runtime unavailable",
)
def test_arbitrary_uniform_jacobi_cells_match_cpu_arb_authorization() -> None:
    launch, cells = _jacobi_panel()
    active = _JACOBI_EXPOSURE > 0.0
    assert launch.authorized_mask.detach().cpu().tolist() == [True] * active.size
    assert _host(launch.fallback_reason_codes).tolist() == [0] * active.size
    assert _host(launch.certificate_codes)[active].tolist() == [15] * int(
        active.sum()
    )
    assert all(
        haar_cuda.enclosing_dyadic_prefix(cell)[1] >= 128
        for cell, is_active in zip(cells, active, strict=True)
        if is_active
    )

    cpu = haar_cuda.sample_alpha1_rb_transition_batch_from_uniform_cells_cpu(
        _JACOBI_X,
        _JACOBI_EXPOSURE,
        cells,
        transition_ids=_JACOBI_IDS,
        profile=JacobiRBCudaProfile(),
    )
    assert cpu.runtime_report["authorization_backend"] == "python-flint/Arb"
    assert cpu.certified_mask[active].tolist() == [True] * int(active.sum())

    np.testing.assert_array_equal(
        _host(launch.later)[active], cpu.later_head_fraction[active]
    )
    np.testing.assert_array_equal(
        _host(launch.target)[active], cpu.denoising_target[active]
    )
    np.testing.assert_array_equal(
        _host(launch.quantile_lower)[active], cpu.quantile_lower[active]
    )
    np.testing.assert_array_equal(
        _host(launch.quantile_upper)[active], cpu.quantile_upper[active]
    )

    gpu_target_lower = _host(launch.target_lower)
    gpu_target_upper = _host(launch.target_upper)
    gpu_target = _host(launch.target)
    for index in np.flatnonzero(active):
        assert gpu_target_lower[index] <= gpu_target[index] <= gpu_target_upper[index]
        assert (
            cpu.target_lower[index]
            <= cpu.denoising_target[index]
            <= cpu.target_upper[index]
        )
        assert gpu_target_lower[index] <= cpu.target_upper[index]
        assert cpu.target_lower[index] <= gpu_target_upper[index]

    inactive = ~active
    np.testing.assert_array_equal(_host(launch.later)[inactive], _JACOBI_X[inactive])
    np.testing.assert_array_equal(_host(launch.target)[inactive], 0.0)
    assert _host(launch.certificate_codes)[inactive].tolist() == [0]


@pytest.mark.skipif(
    not _frozen_jacobi_cuda_available(),
    reason="frozen Jacobi CUDA/Arb runtime unavailable",
)
def test_arbitrary_uniform_authorizer_rejects_under_radius_dd_ball() -> None:
    lower = _JACOBI_CENTER_HI - 1.0e-12
    upper = _JACOBI_CENTER_HI + 1.0e-12
    launch = fused.launch_certified_jacobi_from_uniform_cells(
        _cuda_tensor(_JACOBI_X),
        _cuda_tensor(_JACOBI_EXPOSURE),
        _cuda_tensor(lower),
        _cuda_tensor(upper),
        _cuda_tensor(_JACOBI_IDS, torch.uint64),
        uniform_center_hi=_cuda_tensor(_JACOBI_CENTER_HI),
        uniform_center_lo=_cuda_tensor(_JACOBI_CENTER_LO),
        uniform_radius=_cuda_tensor(
            np.zeros(_JACOBI_X.size, dtype=np.float64)
        ),
        mode_cap=8192,
    )
    active = _JACOBI_EXPOSURE > 0.0
    assert not bool(_host(launch.authorized_mask)[active].any())
    assert np.all(_host(launch.fallback_reason_codes)[active] == 6)


@pytest.mark.skipif(
    not _frozen_jacobi_cuda_available(),
    reason="frozen Jacobi CUDA/Arb runtime unavailable",
)
def test_both_fused_authorizers_are_order_invariant() -> None:
    permutation = (3, 0, 4, 1, 2)
    _events, _signs, normal = _normal_panel()
    _permuted_events, _permuted_signs, permuted_normal = _normal_panel(permutation)
    inverse = np.argsort(permutation)
    for field in (
        "normal_center_hi",
        "normal_center_lo",
        "normal_radius",
        "normal_lower",
        "normal_upper",
        "uniform_center_hi",
        "uniform_center_lo",
        "uniform_radius",
        "uniform_lower",
        "uniform_upper",
        "certificate_mask",
        "fallback_reason_codes",
        "prefix_bits",
    ):
        np.testing.assert_array_equal(
            _host(getattr(normal, field)),
            _host(getattr(permuted_normal, field))[inverse],
        )

    jacobi_permutation = (2, 0, 3, 1)
    jacobi, _cells = _jacobi_panel()
    permuted_jacobi, _permuted_cells = _jacobi_panel(jacobi_permutation)
    jacobi_inverse = np.argsort(jacobi_permutation)
    for field in (
        "later",
        "target",
        "quantile_lower",
        "quantile_upper",
        "target_lower",
        "target_upper",
        "modes_used",
        "certificate_codes",
        "authorized_mask",
        "fallback_reason_codes",
        "candidate",
    ):
        np.testing.assert_array_equal(
            _host(getattr(jacobi, field)),
            _host(getattr(permuted_jacobi, field))[jacobi_inverse],
        )
    assert jacobi.launch_count == permuted_jacobi.launch_count


def test_host_and_structural_inputs_are_rejected_before_launch() -> None:
    source_ids = torch.zeros((1, fused.HAAR_NORMAL_MAX_SOURCES), dtype=torch.uint64)
    vector = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(TypeError, match="CUDA tensor"):
        fused.launch_certified_haar_normal_transform(
            source_ids,
            vector,
            vector,
            torch.ones_like(vector),
            root_seed=1,
        )

    host_float = torch.tensor([0.2], dtype=torch.float64)
    host_id = torch.tensor([1], dtype=torch.uint64)
    with pytest.raises(ValueError, match="CUDA"):
        fused.launch_certified_jacobi_from_uniform_cells(
            host_float,
            torch.tensor([0.5], dtype=torch.float64),
            torch.tensor([0.25], dtype=torch.float64),
            torch.tensor([0.26], dtype=torch.float64),
            host_id,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_malformed_cuda_values_are_rejected_fail_closed() -> None:
    source_ids = torch.ones((2, 2), dtype=torch.uint64, device="cuda")
    depths = _cuda_tensor([0, 1], torch.int32)
    branches = _cuda_tensor([0, 1], torch.int32)
    signs = _cuda_tensor([1, -1], torch.int32)

    with pytest.raises(ValueError):
        fused.launch_certified_haar_normal_transform(
            source_ids,
            _cuda_tensor([0, 2], torch.int32),
            branches,
            signs,
            root_seed=1,
        )
    with pytest.raises(ValueError):
        fused.launch_certified_haar_normal_transform(
            source_ids,
            depths,
            _cuda_tensor([0, -1], torch.int32),
            signs,
            root_seed=1,
        )
    with pytest.raises(ValueError):
        fused.launch_certified_haar_normal_transform(
            source_ids,
            depths,
            branches,
            _cuda_tensor([1, 0], torch.int32),
            root_seed=1,
        )
    with pytest.raises(TypeError, match="int32"):
        fused.launch_certified_haar_normal_transform(
            source_ids,
            depths.to(torch.int64),
            branches,
            signs,
            root_seed=1,
        )
    with pytest.raises(ValueError, match="threads_per_block"):
        fused.launch_certified_haar_normal_transform(
            source_ids,
            depths,
            branches,
            signs,
            root_seed=1,
            threads_per_block=16,
        )

    x = _cuda_tensor([0.2])
    exposure = _cuda_tensor([0.5])
    lower = _cuda_tensor([0.25])
    upper = _cuda_tensor([0.250000000000001])
    transition_ids = _cuda_tensor([7], torch.uint64)
    for malformed_exposure in (-0.1, math.nan, math.inf):
        with pytest.raises(ValueError):
            fused.launch_certified_jacobi_from_uniform_cells(
                x,
                _cuda_tensor([malformed_exposure]),
                lower,
                upper,
                transition_ids,
            )
    for malformed_x in (-0.1, 1.1, math.nan, math.inf):
        with pytest.raises(ValueError):
            fused.launch_certified_jacobi_from_uniform_cells(
                _cuda_tensor([malformed_x]),
                exposure,
                lower,
                upper,
                transition_ids,
            )
    with pytest.raises(ValueError, match="strictly inside"):
        fused.launch_certified_jacobi_from_uniform_cells(
            x,
            exposure,
            _cuda_tensor([0.0]),
            upper,
            transition_ids,
        )
    with pytest.raises(ValueError, match="mode_cap"):
        fused.launch_certified_jacobi_from_uniform_cells(
            x,
            exposure,
            lower,
            upper,
            transition_ids,
            mode_cap=64,
        )
