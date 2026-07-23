from __future__ import annotations

import math

import numpy as np
import pytest

from mnist.d0_jacobi_ancestral import (
    AncestralSamplerConfig,
    JacobiCertificationError,
    sample_ancestral_count,
    sample_wright_fisher_latent,
    standard_wf_time,
)


def _config() -> AncestralSamplerConfig:
    return AncestralSamplerConfig(
        decimal_precision=60,
        max_ancestral_count=128,
        max_terms=3000,
        max_refinements=3000,
    )


def test_ancestral_draw_is_replayable_and_has_a_strict_cdf_certificate() -> None:
    first = sample_ancestral_count(
        jacobi_time=0.5, alpha=1.0, uniform=0.37123456789, config=_config()
    )
    second = sample_ancestral_count(
        jacobi_time=0.5, alpha=1.0, uniform=0.37123456789, config=_config()
    )
    assert first == second
    assert first.certified
    assert first.lower_cdf_bound > first.uniform
    assert first.upper_cdf_bound >= first.lower_cdf_bound
    assert first.terms_evaluated > 0
    assert first.standard_wf_time == 1.0
    assert standard_wf_time(0.5) == 1.0


def test_small_time_work_cap_fails_closed_without_approximation() -> None:
    with pytest.raises(JacobiCertificationError, match="workload") as captured:
        sample_ancestral_count(
            jacobi_time=1e-5,
            alpha=1.0,
            uniform=0.4,
            config=_config(),
        )
    assert captured.value.diagnostics["certified"] == 0
    assert "Gaussian" not in str(captured.value)


def test_latent_transition_has_the_exact_first_eigenmoment_and_label() -> None:
    rng = np.random.default_rng(261102)
    draws = [
        sample_wright_fisher_latent(
            head_fraction=0.3,
            jacobi_time=0.5,
            alpha=1.0,
            rng=rng,
            config=_config(),
        )
        for _ in range(800)
    ]
    y = np.asarray([draw.later_head_fraction for draw in draws])
    expected = 0.5 + (0.3 - 0.5) * math.exp(-1.0)
    standard_error = float(np.std(y, ddof=1) / math.sqrt(y.size))
    assert abs(float(np.mean(y)) - expected) <= 3.5 * standard_error
    for draw in draws:
        assert draw.denoising_label == pytest.approx(
            draw.head_count - draw.ancestral_count * draw.later_head_fraction,
            abs=0.0,
        )
        assert 0.0 < draw.later_head_fraction < 1.0

