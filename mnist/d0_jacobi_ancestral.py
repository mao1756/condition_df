"""Certified neutral Wright--Fisher ancestral-mixture draws.

This implements Algorithm 2 of Jenkins and Spano (2017) for the ancestral
line-count distribution.  It uses arbitrary-precision alternating-series
bounds and fails closed when a configured work cap is reached.  No Gaussian
or finite-population approximation is used.

Our Jacobi generator convention is

    u * [x(1-x) d_xx + alpha(1-2x) d_x],

whereas the standard Wright--Fisher convention has one half of that
generator.  Consequently the ancestral process is evaluated at ``t = 2u``
with total mutation parameter ``theta = 2alpha``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import mpmath as mp
import numpy as np


ANCESTRAL_SAMPLER_VERSION = "jenkins-spano-alternating-series-algorithm2-v1"


class JacobiCertificationError(RuntimeError):
    """Raised when an exact draw cannot be certified within frozen caps."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


@dataclass(frozen=True)
class AncestralSamplerConfig:
    decimal_precision: int = 80
    max_ancestral_count: int = 4096
    max_terms: int = 100_000
    max_refinements: int = 20_000

    def __post_init__(self) -> None:
        if self.decimal_precision < 40:
            raise ValueError("decimal_precision must be at least 40")
        if self.max_ancestral_count < 1 or self.max_terms < 4 or self.max_refinements < 1:
            raise ValueError("ancestral sampler caps must be positive")

    def to_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class AncestralDraw:
    count: int
    uniform: float
    lower_cdf_bound: float
    upper_cdf_bound: float
    bracket_width: float
    terms_evaluated: int
    refinements: int
    maximum_monotone_offset: int
    standard_wf_time: float
    theta: float
    certified: bool = True
    sampler_version: str = ANCESTRAL_SAMPLER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WrightFisherLatentDraw:
    ancestral_count: int
    head_count: int
    later_head_fraction: float
    denoising_label: float
    ancestral: AncestralDraw

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["ancestral"] = self.ancestral.to_dict()
        value["latent_identity"] = "Z=L-MY"
        return value


def standard_wf_time(jacobi_time: float) -> float:
    value = 2.0 * float(jacobi_time)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("jacobi_time must be finite and positive")
    return value


def _exact_mpf_from_float(value: float) -> mp.mpf:
    numerator, denominator = float(value).as_integer_ratio()
    return mp.mpf(numerator) / mp.mpf(denominator)


def _log_coefficient(k: int, m: int, theta: mp.mpf) -> mp.mpf:
    if m < 0 or k < m:
        raise ValueError("coefficient indices require k >= m >= 0")
    # a^theta_km = (theta+2k-1) (theta+m)_(k-1) / (m! (k-m)!)
    first = theta + 2 * k - 1
    if first <= 0:
        raise ValueError("theta is outside the supported entrance-law domain")
    return (
        mp.log(first)
        + mp.loggamma(theta + m + k - 1)
        - mp.loggamma(theta + m)
        - mp.loggamma(m + 1)
        - mp.loggamma(k - m + 1)
    )


def _log_b(k: int, m: int, t: mp.mpf, theta: mp.mpf) -> mp.mpf:
    return _log_coefficient(k, m, theta) - mp.mpf("0.5") * k * (k + theta - 1) * t


def _coefficient_ratio(k: int, m: int, t: mp.mpf, theta: mp.mpf) -> mp.mpf:
    """Return b_{k+1}(m)/b_k(m) without forming either coefficient."""

    return (
        (theta + 2 * k + 1) / (theta + 2 * k - 1)
        * (theta + m + k - 1) / (k + 1 - m)
        * mp.exp(-mp.mpf("0.5") * (2 * k + theta) * t)
    )


def _monotone_offset(
    m: int,
    t: mp.mpf,
    theta: mp.mpf,
    *,
    max_terms: int,
) -> int:
    for offset in range(max_terms):
        if _coefficient_ratio(m + offset, m, t, theta) < 1:
            return offset
    raise JacobiCertificationError(
        "alternating coefficients did not enter their monotone tail",
        {
            "ancestral_count": int(m),
            "max_terms": int(max_terms),
            "failure_kind": "resource_cap",
        },
    )


class _ComponentBounds:
    def __init__(
        self,
        m: int,
        t: mp.mpf,
        theta: mp.mpf,
        config: AncestralSamplerConfig,
    ) -> None:
        self.m = int(m)
        self.t = t
        self.theta = theta
        self.config = config
        self.monotone_offset = _monotone_offset(
            self.m, t, theta, max_terms=config.max_terms
        )
        self.k_pairs = int(math.ceil(self.monotone_offset / 2.0))
        self.terms = 0
        self.upper = mp.mpf("0")
        self.lower = mp.mpf("0")
        self._initialize()

    def _b(self, offset: int) -> mp.mpf:
        if self.terms >= self.config.max_terms:
            raise JacobiCertificationError(
                "ancestral alternating-series term cap reached",
                {
                    "ancestral_count": self.m,
                    "terms_evaluated": self.terms,
                    "max_terms": self.config.max_terms,
                    "failure_kind": "resource_cap",
                },
            )
        self.terms += 1
        return mp.exp(_log_b(self.m + int(offset), self.m, self.t, self.theta))

    def _initialize(self) -> None:
        even_last = 2 * self.k_pairs
        total = mp.mpf("0")
        for offset in range(even_last + 1):
            term = self._b(offset)
            total = total + term if offset % 2 == 0 else total - term
        self.upper = total
        self.lower = total - self._b(even_last + 1)

    def refine(self) -> None:
        next_even = 2 * self.k_pairs + 2
        new_upper = self.lower + self._b(next_even)
        new_lower = new_upper - self._b(next_even + 1)
        if new_lower + mp.eps < self.lower or new_upper - mp.eps > self.upper:
            raise JacobiCertificationError(
                "alternating-series CDF bounds lost monotonicity",
                {
                    "ancestral_count": self.m,
                    "k_pairs": self.k_pairs,
                    "failure_kind": "numerical_certificate",
                },
            )
        self.k_pairs += 1
        self.upper = new_upper
        self.lower = new_lower


def sample_ancestral_count(
    *,
    jacobi_time: float,
    alpha: float,
    uniform: float,
    config: AncestralSamplerConfig | None = None,
) -> AncestralDraw:
    """Draw the entrance-law ancestral count with a certified CDF bracket."""

    cfg = config or AncestralSamplerConfig()
    if not math.isfinite(alpha) or alpha <= 0.5:
        # The coefficient formula includes Gamma(theta-1) at m=k=0.  The
        # production configuration alpha=1 lies safely in this domain.
        raise ValueError("this certified implementation requires alpha > 0.5")
    if not math.isfinite(uniform) or not 0.0 < uniform < 1.0:
        raise ValueError("uniform must lie strictly between zero and one")
    t_float = standard_wf_time(jacobi_time)
    # E[A_infinity(t)] is asymptotically of order 2/t.  This is not used as an
    # approximation to a draw; it is an early work-cap guard.  A cap failure is
    # explicitly uncertified and therefore cannot pass a production gate.
    workload_estimate = int(math.ceil(2.0 / t_float))
    if workload_estimate > cfg.max_ancestral_count:
        raise JacobiCertificationError(
            "small-time ancestral workload exceeds the certified count cap",
            {
                "standard_wf_time": t_float,
                "ancestral_count_workload_estimate": workload_estimate,
                "max_ancestral_count": cfg.max_ancestral_count,
                "certified": 0,
                "failure_kind": "resource_cap",
            },
        )

    with mp.workdps(cfg.decimal_precision):
        t = _exact_mpf_from_float(t_float)
        theta = _exact_mpf_from_float(2.0 * float(alpha))
        u = _exact_mpf_from_float(float(uniform))
        components: list[_ComponentBounds] = []
        refinements = 0
        for candidate in range(cfg.max_ancestral_count + 1):
            components.append(_ComponentBounds(candidate, t, theta, cfg))
            while True:
                raw_lower = mp.fsum(item.lower for item in components)
                raw_upper = mp.fsum(item.upper for item in components)
                if raw_lower > raw_upper:
                    raise JacobiCertificationError(
                        "ancestral CDF lower bound exceeded upper bound",
                        {
                            "candidate": candidate,
                            "lower": str(raw_lower),
                            "upper": str(raw_upper),
                            "failure_kind": "numerical_certificate",
                        },
                    )
                # Alternating partial sums may initially give a negative lower
                # or an upper above one, but a lower above one or upper below
                # zero cannot enclose a probability.  In that situation the
                # finite-precision cancellation has invalidated the numerical
                # certificate and must never be mistaken for a sampled count.
                if raw_lower > 1 or raw_upper < 0:
                    raise JacobiCertificationError(
                        "ancestral CDF bracket left the probability range",
                        {
                            "candidate": candidate,
                            "lower": str(raw_lower),
                            "upper": str(raw_upper),
                            "decimal_precision": cfg.decimal_precision,
                            "failure_kind": "numerical_certificate",
                            "certified": 0,
                        },
                    )
                # Intersect a mathematically valid (possibly loose) series
                # bracket with the known probability range.  This tightens the
                # certificate; it does not change the sampled transition.
                lower = max(mp.mpf("0"), raw_lower)
                upper = min(mp.mpf("1"), raw_upper)
                if lower > u:
                    total_terms = sum(item.terms for item in components)
                    return AncestralDraw(
                        count=int(candidate),
                        uniform=float(uniform),
                        lower_cdf_bound=float(lower),
                        upper_cdf_bound=float(upper),
                        bracket_width=float(upper - lower),
                        terms_evaluated=int(total_terms),
                        refinements=int(refinements),
                        maximum_monotone_offset=max(item.monotone_offset for item in components),
                        standard_wf_time=t_float,
                        theta=float(theta),
                    )
                if upper < u:
                    break
                if refinements >= cfg.max_refinements:
                    raise JacobiCertificationError(
                        "ancestral alternating-series refinement cap reached",
                        {
                            "candidate": candidate,
                            "refinements": refinements,
                            "max_refinements": cfg.max_refinements,
                            "lower": str(lower),
                            "upper": str(upper),
                            "certified": 0,
                            "failure_kind": "resource_cap",
                        },
                    )
                for item in components:
                    item.refine()
                refinements += 1
        raise JacobiCertificationError(
            "ancestral count cap reached before the CDF bracket crossed the uniform",
            {
                "max_ancestral_count": cfg.max_ancestral_count,
                "standard_wf_time": t_float,
                "certified": 0,
                "failure_kind": "resource_cap",
            },
        )


def sample_wright_fisher_latent(
    *,
    head_fraction: float,
    jacobi_time: float,
    alpha: float,
    rng: np.random.Generator,
    config: AncestralSamplerConfig | None = None,
) -> WrightFisherLatentDraw:
    """Return an exact neutral transition and its Jacobi denoising label."""

    x = float(head_fraction)
    if not math.isfinite(x) or not 0.0 <= x <= 1.0:
        raise ValueError("head_fraction must lie in [0, 1]")
    uniform = float(rng.random())
    # NumPy may produce exactly zero.  Moving to the next representable value
    # only selects a valid open-interval inversion variate; it does not alter
    # the transition law at machine precision.
    if uniform == 0.0:
        uniform = float(np.nextafter(0.0, 1.0))
    ancestral = sample_ancestral_count(
        jacobi_time=jacobi_time,
        alpha=alpha,
        uniform=uniform,
        config=config,
    )
    m = int(ancestral.count)
    l = int(rng.binomial(m, x)) if m else 0
    y = float(rng.beta(float(alpha) + l, float(alpha) + m - l))
    if not math.isfinite(y) or not 0.0 < y < 1.0:
        raise JacobiCertificationError(
            "Beta transition draw was nonfinite or reached a boundary",
            {
                "m": m,
                "l": l,
                "y": y,
                "certified": 0,
                "failure_kind": "numerical_certificate",
            },
        )
    z = float(l - m * y)
    return WrightFisherLatentDraw(
        ancestral_count=m,
        head_count=l,
        later_head_fraction=y,
        denoising_label=z,
        ancestral=ancestral,
    )


__all__ = [
    "ANCESTRAL_SAMPLER_VERSION",
    "AncestralDraw",
    "AncestralSamplerConfig",
    "JacobiCertificationError",
    "WrightFisherLatentDraw",
    "sample_ancestral_count",
    "sample_wright_fisher_latent",
    "standard_wf_time",
]
