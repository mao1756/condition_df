from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist.d0_jacobi_rb_boundary_tangent_cache import (
    BoundaryTangentCacheError,
    MIDPOINT_FRACTIONS,
    flatten_midpoint_batches,
    midpoint_sample_key,
    sample_midpoint_branches,
)
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile


def _fake_sampler(head, exposure, *, rng_key, transition_ids, profile):
    del rng_key, transition_ids, profile
    later = torch.clamp(head + 0.01 * exposure, 0.0, 1.0)
    target = later * (1.0 - later)
    return SimpleNamespace(
        later_head_fraction=later,
        denoising_target=target,
        certificate_codes=torch.full_like(later, 0b1111, dtype=torch.uint8),
        mode_counts=torch.full_like(later, 16, dtype=torch.int32),
        prefix_bits=torch.full_like(later, 64, dtype=torch.int32),
        fallback_mask=torch.zeros_like(later, dtype=torch.bool),
        strengthened_mask=torch.zeros_like(later, dtype=torch.bool),
    )


def test_midpoint_sample_key_is_unique_and_validates() -> None:
    values = {
        midpoint_sample_key(path, step, phase, midpoint)
        for path in (0xEC100, 0xEC101)
        for step in (15, 31)
        for phase in range(7)
        for midpoint in range(8)
    }
    assert len(values) == 2 * 2 * 7 * 8
    with pytest.raises(BoundaryTangentCacheError):
        midpoint_sample_key(1 << 20, 15, 0, 0)
    with pytest.raises(BoundaryTangentCacheError):
        midpoint_sample_key(0, 15, 0, 8)


def test_exact_midpoint_branches_preserve_input_and_flatten() -> None:
    states = torch.full((2, 784), 1.0 / 784.0, dtype=torch.float64)
    original = states.clone()
    result = sample_midpoint_branches(
        states.contiguous(),
        path_ids=(0xEC100, 0xEC101),
        outer_step=15,
        phase=0,
        profile=JacobiRBCudaProfile(),
        sampler=_fake_sampler,
    )
    assert torch.equal(states, original)
    assert result.later_full_state.shape == (8, 2, 784)
    assert result.denoising_target.shape == (8, 2, 392)
    assert result.certified_count == result.transition_count
    assert result.midpoint_fractions == MIDPOINT_FRACTIONS
    assert torch.allclose(
        result.later_full_state.sum(dim=2),
        torch.ones((8, 2), dtype=torch.float64),
        rtol=0.0,
        atol=2.0e-15,
    )

    inputs, audit = flatten_midpoint_batches((result,))
    assert inputs["later_full_state"].shape == (16, 784)
    assert audit["denoising_target"].shape == (16, 392)
    assert len(np.unique(inputs["sample_key"])) == 16
    assert np.array_equal(inputs["sample_key"], audit["sample_key"])
    # Canonical ordering is path-major, then split coordinate.
    assert audit["path_id"].tolist()[:8] == [0xEC100] * 8
    assert audit["midpoint_index"].tolist()[:8] == list(range(8))


def test_uncertified_midpoint_fails_closed() -> None:
    def bad_sampler(*args, **kwargs):
        result = _fake_sampler(*args, **kwargs)
        result.certificate_codes[0] = 0
        return result

    states = torch.full((1, 784), 1.0 / 784.0, dtype=torch.float64)
    with pytest.raises(BoundaryTangentCacheError, match="uncertified"):
        sample_midpoint_branches(
            states.contiguous(),
            path_ids=(0xEC100,),
            outer_step=15,
            phase=0,
            profile=JacobiRBCudaProfile(),
            sampler=bad_sampler,
        )
