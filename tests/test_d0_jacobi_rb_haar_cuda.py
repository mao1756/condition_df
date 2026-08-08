from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from mnist import d0_jacobi_rb_haar as haar
from mnist import d0_jacobi_rb_haar_cuda as haar_cuda
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile


def test_enclosing_prefix_contains_complete_arbitrary_interval() -> None:
    cell = haar.CertifiedUniformCell(
        Fraction(123456789, 1 << 80),
        Fraction(123456790, 1 << 80),
    )
    numerator, bits = haar_cuda.enclosing_dyadic_prefix(cell)
    assert Fraction(numerator, 1 << bits) <= cell.lower
    assert cell.upper <= Fraction(numerator + 1, 1 << bits)
    assert bits >= 78

    crossing = haar.CertifiedUniformCell(
        Fraction(1, 2) - Fraction(1, 1 << 100),
        Fraction(1, 2) + Fraction(1, 1 << 100),
    )
    with pytest.raises(Exception, match="first dyadic boundary"):
        haar_cuda.enclosing_dyadic_prefix(crossing)


@pytest.mark.skipif(haar.arb is None, reason="python-flint/Arb unavailable")
def test_cpu_interval_adapter_authorizes_exact_jacobi_and_zero_duration() -> None:
    uniform = haar.build_certified_haar_uniform_batch(
        root_seed=261181,
        role="marginal_c",
        path_ids=haar.path_ids_for_role("marginal_c", 1),
        sample_steps=256,
        outer_step=3,
        phase=2,
        edge_ids=[0, 1],
        profile=haar.HaarCouplingProfile(),
    )
    result = haar_cuda.sample_alpha1_rb_transition_batch_from_uniform_cells_cpu(
        np.array([[0.3, 0.7]], dtype=np.float64),
        np.array([[0.5, 0.0]], dtype=np.float64),
        uniform.uniform_cells,
        transition_ids=np.array([[7, 8]], dtype=np.uint64),
        profile=JacobiRBCudaProfile(),
    )
    assert result.certified_mask.tolist() == [[True, False]]
    assert result.fallback_mask.tolist() == [[True, False]]
    assert result.certificate_codes.tolist() == [[15, 0]]
    assert result.later_head_fraction[0, 1] == 0.7
    assert result.denoising_target[0, 1] == 0.0
    assert result.quantile_lower[0, 0] < result.later_head_fraction[0, 0]
    assert result.quantile_upper[0, 0] > result.later_head_fraction[0, 0]
    assert result.diagnostics["approximation_count"] == 0
    assert result.runtime_report["arb_authorizing"] is True
    assert result.runtime_report["cuda_authorizing"] is False


@pytest.mark.skipif(haar.arb is None, reason="python-flint/Arb unavailable")
def test_cpu_interval_adapter_is_order_invariant() -> None:
    batch = haar.build_certified_haar_uniform_batch(
        root_seed=99,
        role="marginal_d",
        path_ids=haar.path_ids_for_role("marginal_d", 1),
        sample_steps=128,
        outer_step=5,
        phase=0,
        edge_ids=[3, 4],
        profile=haar.HaarCouplingProfile(),
    )
    first = haar_cuda.sample_alpha1_rb_transition_batch_from_uniform_cells_cpu(
        [0.2, 0.8],
        [0.4, 0.4],
        batch.uniform_cells,
        transition_ids=np.array([11, 12], dtype=np.uint64),
        profile=JacobiRBCudaProfile(),
    )
    second = haar_cuda.sample_alpha1_rb_transition_batch_from_uniform_cells_cpu(
        [0.8, 0.2],
        [0.4, 0.4],
        batch.uniform_cells[::-1],
        transition_ids=np.array([12, 11], dtype=np.uint64),
        profile=JacobiRBCudaProfile(),
    )
    np.testing.assert_array_equal(
        first.later_head_fraction, second.later_head_fraction[::-1]
    )
    np.testing.assert_array_equal(
        first.denoising_target, second.denoising_target[::-1]
    )


def test_cuda_adapter_contract_rejects_host_tensors() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="CUDA tensor"):
        haar_cuda.sample_alpha1_rb_transition_batch_cuda_from_uniform_cells(
            torch.tensor([0.2], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64),
            torch.tensor([0.25], dtype=torch.float64),
            torch.tensor([0.26], dtype=torch.float64),
            transition_ids=torch.tensor([1], dtype=torch.uint64),
            refinement_callback=None,
            profile=JacobiRBCudaProfile(),
        )


def test_source_contains_no_approximate_jacobi_authorizer() -> None:
    source = open(haar_cuda.__file__, encoding="utf-8").read()
    assert "Gaussian/Euler" not in source
    assert "sample_alpha1_rb_transition_batch_cuda_from_uniform_cells" in source
    assert "launch_certified_jacobi_from_uniform_cells" in source
    assert "direct_dd_uniform_cell_authorization" in source
    assert "cuda_authorizing\": False" in source
