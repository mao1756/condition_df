r"""Certified spectral Jacobi transitions and Rao--Blackwell denoising labels.

This module is deliberately additive.  In particular it does not import or
modify the ancestral-mixture implementation used by the immutable Jacobi
feasibility run.

For ``alpha=1`` the two-cell head fraction has transition kernel, relative to
the invariant uniform law,

.. math::

   k_u(y|x)=1+\sum_{n\geq1}(2n+1)e^{-n(n+1)u}P_n(2x-1)P_n(2y-1).

The sampler in :func:`sample_alpha1_rb_transition_batch` inverts the matching
Legendre CDF and returns the exact Rao--Blackwell population target

.. math::

   \bar Z(x,y,u)=y(1-y)\partial_y\log k_u(y|x).

``bar Z`` is evaluated through its conormal numerator, avoiding a singular
score followed by multiplication by ``y(1-y)``.  It satisfies
``bar Z = E[L-MY | X=x,Y=y,u]`` and hence has exactly the same regression
function as the latent target ``L-MY``.

Numerical semantics
-------------------

The portable implementation is a CPU reference kernel.  Every branch is made
against an interval consisting of propagated binary64 roundoff plus an
analytic uniform Legendre tail.  Ambiguous comparisons escalate only to
``python-flint``/Arb ball arithmetic and otherwise fail closed.  ``mpmath``
intervals are deliberately not an authorizing backend.  No point estimate is
used to resolve an ambiguous branch.  The production profile continues until
the quantile is assigned to a unique IEEE-754 binary64 rounding cell.

The device helpers provide a real batched Torch/CUDA fixed-mode evaluator and
inverse-CDF proposal.  Their result is explicitly non-authorizing: final
machine decisions still require the interval/Arb reference path.  This split
keeps the expensive proposal measurable on the production device without
silently treating ordinary CUDA arithmetic as a numerical certificate.

The pseudo-random uniform is represented by deterministic lazy dyadic prefixes
derived from ``rng_key``.  More bits are consumed only when an interval
comparison is ambiguous.  Replay is exact for a fixed input shape, key, and
profile.  Rebatching changes flat sample indices and is intentionally outside
this low-level API; callers that need batching invariance should include the
global sample offset in ``rng_key``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Any, Iterable
import threading

import numpy as np

try:  # Optional, but required whenever the fast certificate is ambiguous.
    from flint import arb as _arb
    from flint import ctx as _flint_ctx
except ImportError:  # pragma: no cover - production preflight reports this.
    _arb = None
    _flint_ctx = None

try:  # Optional at import time; production device benchmarks require Torch.
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - Torch is part of the project env.
    torch = None
    Tensor = Any


_ARB_CONTEXT_LOCK = threading.Lock()


JACOBI_RB_SPECTRAL_VERSION = "alpha1-legendre-rb-arb-v3"
JACOBI_RB_RNG_VERSION = "philox-transition-local-lazy-dyadic-v1"
JACOBI_RB_ORIENTATION = "head-fraction"
JACOBI_RB_DEVICE_VERSION = "torch-fixed-legendre-proposal-v1"


class JacobiRBCertificationError(RuntimeError):
    """Raised when a transition cannot be certified under the frozen caps."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


@dataclass(frozen=True)
class JacobiRBSpectralProfile:
    """Versioned numerical contract for the certified reference sampler.

    Defaults are the production, fail-closed contract.  Tests may construct a
    smaller profile, but any override is fingerprint-visible and therefore
    cannot masquerade as the production feasibility claim.
    """

    schema_version: int = 2
    max_modes: int = 16384
    fast_cdf_tail_tolerance: float = 2e-14
    fast_target_tail_tolerance: float = 2e-13
    quantile_tolerance: float = 1e-10
    target_interval_tolerance: float = 1e-8
    max_bisection_steps: int = 96
    initial_prefix_bits: int = 64
    prefix_block_bits: int = 64
    max_prefix_bits: int = 1024
    roundoff_ulps: int = 4
    allow_interval_escalation: bool = True
    arb_precision_bits: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096, 8192)
    max_arb_precision_bits: int = 8192
    arb_tail_tolerance: float = 1e-30
    require_correct_rounding: bool = True
    device_proposal_modes: int = 568
    device_bisection_steps: int = 56
    device_dtype: str = "float64"
    authorize_device_intervals: bool = False

    def __post_init__(self) -> None:
        if int(self.schema_version) != 2:
            raise ValueError("unsupported Jacobi RB profile schema_version")
        if int(self.max_modes) < 4:
            raise ValueError("max_modes must be at least four")
        for name in (
            "fast_cdf_tail_tolerance",
            "fast_target_tail_tolerance",
            "quantile_tolerance",
            "target_interval_tolerance",
            "arb_tail_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if int(self.max_bisection_steps) < 2:
            raise ValueError("max_bisection_steps must be at least two")
        if not 1 <= int(self.initial_prefix_bits) <= 64:
            raise ValueError("initial_prefix_bits must lie in [1,64]")
        if int(self.prefix_block_bits) != 64:
            raise ValueError("the version-1 dyadic stream uses 64-bit blocks")
        if int(self.max_prefix_bits) < int(self.initial_prefix_bits):
            raise ValueError("max_prefix_bits is smaller than the initial prefix")
        if int(self.roundoff_ulps) < 1:
            raise ValueError("roundoff_ulps must be positive")
        if not self.arb_precision_bits or any(
            int(value) < 64 for value in self.arb_precision_bits
        ):
            raise ValueError("Arb precisions must contain values >=64 bits")
        if tuple(sorted(set(self.arb_precision_bits))) != tuple(self.arb_precision_bits):
            raise ValueError("Arb precisions must be strictly increasing")
        if int(self.max_arb_precision_bits) != int(self.arb_precision_bits[-1]):
            raise ValueError("max_arb_precision_bits must equal the final Arb precision")
        if int(self.max_arb_precision_bits) > 8192:
            raise ValueError("production contract caps Arb precision at 8192 bits")
        if int(self.device_proposal_modes) < 2:
            raise ValueError("device_proposal_modes must be at least two")
        if int(self.device_bisection_steps) < 1:
            raise ValueError("device_bisection_steps must be positive")
        if self.device_dtype not in {"float32", "float64"}:
            raise ValueError("device_dtype must be float32 or float64")
        if not isinstance(self.authorize_device_intervals, bool):
            raise ValueError("authorize_device_intervals must be boolean")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["sampler_version"] = JACOBI_RB_SPECTRAL_VERSION
        value["rng_version"] = JACOBI_RB_RNG_VERSION
        return value


@dataclass(frozen=True)
class JacobiRBSpectralDiagnostics:
    certified: bool
    sample_count: int
    active_count: int
    zero_duration_count: int
    interval_escalation_count: int
    correctly_rounded_count: int
    maximum_modes_used: int
    maximum_bisection_steps: int
    maximum_prefix_bits: int
    maximum_quantile_bracket_width: float
    maximum_target_interval_width: float
    sampler_version: str = JACOBI_RB_SPECTRAL_VERSION
    rng_version: str = JACOBI_RB_RNG_VERSION
    orientation: str = JACOBI_RB_ORIENTATION
    interval_backend: str = "python-flint/Arb" if _arb is not None else "unavailable"
    constrained_semantics: str = (
        "authorizing outputs require a unique binary64 rounding cell; "
        "ordinary Torch/CUDA proposals are non-authorizing"
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JacobiRBTransitionBatch:
    earlier_head_fraction: np.ndarray
    later_head_fraction: np.ndarray
    denoising_target: np.ndarray
    exposure: np.ndarray
    active_mask: np.ndarray
    quantile_lower: np.ndarray
    quantile_upper: np.ndarray
    target_lower: np.ndarray
    target_upper: np.ndarray
    prefix_bits: np.ndarray
    certificate_codes: np.ndarray
    diagnostics: JacobiRBSpectralDiagnostics


# The public name in the experiment contract.  The compatibility alias keeps
# the initially developed additive API readable without changing its cache
# layout behind users' backs.
CertifiedRBTransitionBatch = JacobiRBTransitionBatch


@dataclass(frozen=True)
class JacobiPairPhaseInputs:
    pair_total: np.ndarray
    head_fraction: np.ndarray
    exposure: np.ndarray
    active_mask: np.ndarray


@dataclass(frozen=True)
class JacobiRBTorchEvaluation:
    """Non-authorizing fixed-truncation device evaluation."""

    cdf: Tensor
    density: Tensor
    conormal_numerator: Tensor
    denoising_target: Tensor
    modes_used: int
    device_version: str = JACOBI_RB_DEVICE_VERSION


@dataclass(frozen=True)
class JacobiRBDeviceProposal:
    """Batched inverse-CDF proposal requiring subsequent CPU certification."""

    earlier_head_fraction: Tensor
    proposed_later_head_fraction: Tensor
    uniform_midpoint: Tensor
    proposal_lower: Tensor
    proposal_upper: Tensor
    exposure: Tensor
    active_mask: Tensor
    modes_used: int
    bisection_steps: int
    device_version: str = JACOBI_RB_DEVICE_VERSION


@dataclass(frozen=True)
class JacobiRBTorchIntervalEvaluation:
    """Outward-rounded device intervals with analytic omitted-mode tails."""

    cdf_lower: Tensor
    cdf_upper: Tensor
    density_lower: Tensor
    density_upper: Tensor
    conormal_lower: Tensor
    conormal_upper: Tensor
    modes_used: int
    device_version: str = "torch-outward-legendre-interval-v1"


@dataclass(frozen=True)
class JacobiRBDeviceIntervalProposal:
    later_head_fraction: Tensor
    denoising_target: Tensor
    quantile_lower: Tensor
    quantile_upper: Tensor
    target_lower: Tensor
    target_upper: Tensor
    certified_mask: Tensor
    fallback_mask: Tensor
    active_mask: Tensor
    certificate_codes: Tensor
    modes_used: int
    bisection_steps: int
    device_version: str = "torch-outward-legendre-inverse-rb-v1"


def _torch_down(value: Tensor, ulps: int = 1) -> Tensor:
    result = value
    target = torch.full_like(result, -math.inf)
    for _ in range(int(ulps)):
        result = torch.nextafter(result, target)
    return result


def _torch_up(value: Tensor, ulps: int = 1) -> Tensor:
    result = value
    target = torch.full_like(result, math.inf)
    for _ in range(int(ulps)):
        result = torch.nextafter(result, target)
    return result


def _torch_iadd(
    a_lower: Tensor, a_upper: Tensor, b_lower: Tensor, b_upper: Tensor
) -> tuple[Tensor, Tensor]:
    return _torch_down(a_lower + b_lower), _torch_up(a_upper + b_upper)


def _torch_isub(
    a_lower: Tensor, a_upper: Tensor, b_lower: Tensor, b_upper: Tensor
) -> tuple[Tensor, Tensor]:
    return _torch_down(a_lower - b_upper), _torch_up(a_upper - b_lower)


def _torch_imul(
    a_lower: Tensor, a_upper: Tensor, b_lower: Tensor, b_upper: Tensor
) -> tuple[Tensor, Tensor]:
    products = torch.stack(
        (
            a_lower * b_lower,
            a_lower * b_upper,
            a_upper * b_lower,
            a_upper * b_upper,
        ),
        dim=0,
    )
    return _torch_down(torch.amin(products, dim=0)), _torch_up(
        torch.amax(products, dim=0)
    )


def _torch_iscale(
    lower: Tensor, upper: Tensor, scalar: float
) -> tuple[Tensor, Tensor]:
    value = float(scalar)
    if value >= 0.0:
        return _torch_down(lower * value), _torch_up(upper * value)
    return _torch_down(upper * value), _torch_up(lower * value)


def _torch_iexp(lower: Tensor, upper: Tensor) -> tuple[Tensor, Tensor]:
    # CUDA libdevice exp is enclosed conservatively by eight neighbouring
    # binary64 values.  The production support panel independently checks the
    # resulting enclosure against Arb before this backend can authorize work.
    return _torch_down(torch.exp(lower), 8), _torch_up(torch.exp(upper), 8)


def evaluate_alpha1_rb_torch_intervals(
    head_fraction: Tensor,
    later_head_fraction: Tensor,
    exposure: Tensor,
    *,
    modes: int,
) -> JacobiRBTorchIntervalEvaluation:
    """Outward interval Legendre recurrence on CPU or CUDA.

    Inputs are interpreted as exact IEEE values.  Each elementary recurrence
    operation is widened by ``nextafter`` and exponential coefficients by
    eight ulps; analytic geometric tails enclose every omitted Legendre mode.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("Torch is unavailable")
    if int(modes) < 2:
        raise ValueError("modes must be at least two")
    x, y, u = torch.broadcast_tensors(head_fraction, later_head_fraction, exposure)
    if x.dtype != torch.float64 or y.dtype != torch.float64 or u.dtype != torch.float64:
        raise ValueError("authorizing device intervals require torch.float64")
    if bool((~torch.isfinite(x)).any()) or bool(((x < 0) | (x > 1)).any()):
        raise ValueError("head_fraction must lie in [0,1]")
    if bool((~torch.isfinite(y)).any()) or bool(((y < 0) | (y > 1)).any()):
        raise ValueError("later_head_fraction must lie in [0,1]")
    if bool((~torch.isfinite(u)).any()) or bool((u <= 0).any()):
        raise ValueError("exposure must be positive")

    x_lo = x_hi = x
    y_lo = y_hi = y
    zx_lo, zx_hi = _torch_iscale(x_lo, x_hi, 2.0)
    zx_lo, zx_hi = _torch_isub(zx_lo, zx_hi, torch.ones_like(x), torch.ones_like(x))
    zy_lo, zy_hi = _torch_iscale(y_lo, y_hi, 2.0)
    zy_lo, zy_hi = _torch_isub(zy_lo, zy_hi, torch.ones_like(y), torch.ones_like(y))

    px_prev_lo = px_prev_hi = torch.ones_like(x)
    px_cur_lo, px_cur_hi = zx_lo, zx_hi
    py_prev_lo = py_prev_hi = torch.ones_like(y)
    py_cur_lo, py_cur_hi = zy_lo, zy_hi
    cdf_lo, cdf_hi = y, y
    density_lo = density_hi = torch.ones_like(x)
    conormal_lo = conormal_hi = torch.zeros_like(x)

    for degree in range(1, int(modes)):
        zy_py_lo, zy_py_hi = _torch_imul(zy_lo, zy_hi, py_cur_lo, py_cur_hi)
        first_lo, first_hi = _torch_iscale(
            zy_py_lo, zy_py_hi, float(2 * degree + 1)
        )
        second_lo, second_hi = _torch_iscale(
            py_prev_lo, py_prev_hi, float(degree)
        )
        py_next_lo, py_next_hi = _torch_isub(
            first_lo, first_hi, second_lo, second_hi
        )
        py_next_lo, py_next_hi = _torch_iscale(
            py_next_lo, py_next_hi, 1.0 / float(degree + 1)
        )

        exponent_lo, exponent_hi = _torch_iscale(
            u, u, -float(degree * (degree + 1))
        )
        decay_lo, decay_hi = _torch_iexp(exponent_lo, exponent_hi)
        coefficient_lo, coefficient_hi = _torch_iscale(
            decay_lo, decay_hi, float(2 * degree + 1)
        )
        coefficient_px_lo, coefficient_px_hi = _torch_imul(
            coefficient_lo, coefficient_hi, px_cur_lo, px_cur_hi
        )
        density_term_lo, density_term_hi = _torch_imul(
            coefficient_px_lo, coefficient_px_hi, py_cur_lo, py_cur_hi
        )
        density_lo, density_hi = _torch_iadd(
            density_lo, density_hi, density_term_lo, density_term_hi
        )

        delta_lo, delta_hi = _torch_isub(
            py_next_lo, py_next_hi, py_prev_lo, py_prev_hi
        )
        cdf_term_lo, cdf_term_hi = _torch_imul(
            decay_lo, decay_hi, px_cur_lo, px_cur_hi
        )
        cdf_term_lo, cdf_term_hi = _torch_imul(
            cdf_term_lo, cdf_term_hi, delta_lo, delta_hi
        )
        cdf_term_lo, cdf_term_hi = _torch_iscale(
            cdf_term_lo, cdf_term_hi, 0.5
        )
        cdf_lo, cdf_hi = _torch_iadd(cdf_lo, cdf_hi, cdf_term_lo, cdf_term_hi)

        zy_py_lo, zy_py_hi = _torch_imul(zy_lo, zy_hi, py_cur_lo, py_cur_hi)
        basis_lo, basis_hi = _torch_isub(
            py_prev_lo, py_prev_hi, zy_py_lo, zy_py_hi
        )
        basis_lo, basis_hi = _torch_iscale(
            basis_lo, basis_hi, 0.5 * float(degree)
        )
        conormal_term_lo, conormal_term_hi = _torch_imul(
            coefficient_px_lo, coefficient_px_hi, basis_lo, basis_hi
        )
        conormal_lo, conormal_hi = _torch_iadd(
            conormal_lo, conormal_hi, conormal_term_lo, conormal_term_hi
        )

        zx_px_lo, zx_px_hi = _torch_imul(zx_lo, zx_hi, px_cur_lo, px_cur_hi)
        px_first_lo, px_first_hi = _torch_iscale(
            zx_px_lo, zx_px_hi, float(2 * degree + 1)
        )
        px_second_lo, px_second_hi = _torch_iscale(
            px_prev_lo, px_prev_hi, float(degree)
        )
        px_next_lo, px_next_hi = _torch_isub(
            px_first_lo, px_first_hi, px_second_lo, px_second_hi
        )
        px_next_lo, px_next_hi = _torch_iscale(
            px_next_lo, px_next_hi, 1.0 / float(degree + 1)
        )
        px_prev_lo, px_prev_hi = px_cur_lo, px_cur_hi
        px_cur_lo, px_cur_hi = px_next_lo, px_next_hi
        py_prev_lo, py_prev_hi = py_cur_lo, py_cur_hi
        py_cur_lo, py_cur_hi = py_next_lo, py_next_hi

    omitted = int(modes)
    omitted_decay = torch.exp(-float(omitted * (omitted + 1)) * u)
    step_decay = torch.exp(-2.0 * float(omitted + 1) * u)
    cdf_tail = omitted_decay / (1.0 - step_decay)
    density_ratio = (
        float(2 * omitted + 3) / float(2 * omitted + 1) * step_decay
    )
    conormal_ratio = (
        float(2 * omitted + 3)
        / float(2 * omitted + 1)
        * float(omitted + 1)
        / float(omitted)
        * step_decay
    )
    infinity = torch.full_like(u, math.inf)
    density_tail = torch.where(
        density_ratio < 1.0,
        float(2 * omitted + 1) * omitted_decay / (1.0 - density_ratio),
        infinity,
    )
    conormal_tail = torch.where(
        conormal_ratio < 1.0,
        float(omitted * (2 * omitted + 1))
        * omitted_decay
        / (1.0 - conormal_ratio),
        infinity,
    )
    cdf_lo = _torch_down(cdf_lo - _torch_up(cdf_tail, 8), 2)
    cdf_hi = _torch_up(cdf_hi + _torch_up(cdf_tail, 8), 2)
    density_lo = _torch_down(density_lo - _torch_up(density_tail, 8), 2)
    density_hi = _torch_up(density_hi + _torch_up(density_tail, 8), 2)
    conormal_lo = _torch_down(conormal_lo - _torch_up(conormal_tail, 8), 2)
    conormal_hi = _torch_up(conormal_hi + _torch_up(conormal_tail, 8), 2)
    cdf_lo = torch.where(y == 0.0, torch.zeros_like(cdf_lo), cdf_lo)
    cdf_hi = torch.where(y == 0.0, torch.zeros_like(cdf_hi), cdf_hi)
    cdf_lo = torch.where(y == 1.0, torch.ones_like(cdf_lo), cdf_lo)
    cdf_hi = torch.where(y == 1.0, torch.ones_like(cdf_hi), cdf_hi)
    return JacobiRBTorchIntervalEvaluation(
        cdf_lower=cdf_lo,
        cdf_upper=cdf_hi,
        density_lower=density_lo,
        density_upper=density_hi,
        conormal_lower=conormal_lo,
        conormal_upper=conormal_hi,
        modes_used=int(modes),
    )


def evaluate_alpha1_rb_torch_fixed_modes(
    head_fraction: Tensor,
    later_head_fraction: Tensor,
    exposure: Tensor,
    *,
    modes: int,
) -> JacobiRBTorchEvaluation:
    """Evaluate the CDF, density and stable conormal numerator on Torch.

    This is an independently benchmarkable device implementation.  It is not
    by itself a numerical certificate: production evidence compares it with
    the outward-rounded CPU/Arb enclosure returned by the certified API.
    """

    if torch is None:  # pragma: no cover - project dependency
        raise RuntimeError("Torch is unavailable")
    if int(modes) < 2:
        raise ValueError("modes must be at least two")
    x, y, u = torch.broadcast_tensors(head_fraction, later_head_fraction, exposure)
    if not x.is_floating_point() or x.dtype != y.dtype or x.dtype != u.dtype:
        raise ValueError("device spectral inputs must share a floating dtype")
    if bool((~torch.isfinite(x)).any()) or bool(((x < 0) | (x > 1)).any()):
        raise ValueError("head_fraction must lie in [0,1]")
    if bool((~torch.isfinite(y)).any()) or bool(((y < 0) | (y > 1)).any()):
        raise ValueError("later_head_fraction must lie in [0,1]")
    if bool((~torch.isfinite(u)).any()) or bool((u <= 0).any()):
        raise ValueError("exposure must be positive")
    zx = 2.0 * x - 1.0
    zy = 2.0 * y - 1.0
    px_previous, px_current = torch.ones_like(zx), zx
    py_previous, py_current = torch.ones_like(zy), zy
    density = torch.ones_like(zx)
    cdf = y.clone()
    conormal = torch.zeros_like(zx)
    for degree in range(1, int(modes)):
        py_next = (
            float(2 * degree + 1) * zy * py_current
            - float(degree) * py_previous
        ) / float(degree + 1)
        decay = torch.exp(-float(degree * (degree + 1)) * u)
        coefficient = float(2 * degree + 1) * decay
        density = density + coefficient * px_current * py_current
        cdf = cdf + 0.5 * decay * px_current * (py_next - py_previous)
        conormal = conormal + coefficient * px_current * (
            0.5 * float(degree) * (py_previous - zy * py_current)
        )
        px_next = (
            float(2 * degree + 1) * zx * px_current
            - float(degree) * px_previous
        ) / float(degree + 1)
        px_previous, px_current = px_current, px_next
        py_previous, py_current = py_current, py_next
    cdf = torch.where(y == 0, torch.zeros_like(cdf), torch.where(y == 1, torch.ones_like(cdf), cdf))
    return JacobiRBTorchEvaluation(
        cdf=cdf,
        density=density,
        conormal_numerator=conormal,
        denoising_target=conormal / density,
        modes_used=int(modes),
    )


def propose_alpha1_rb_transition_batch_torch(
    head_fraction: Tensor,
    exposure: Tensor,
    *,
    rng_key: Any,
    profile: JacobiRBSpectralProfile,
    transition_ids: Tensor | np.ndarray | None = None,
) -> JacobiRBDeviceProposal:
    """Return a fast device inverse-CDF proposal for later certification."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("Torch is unavailable")
    x, u = torch.broadcast_tensors(head_fraction, exposure)
    if not x.is_floating_point() or x.dtype != u.dtype:
        raise ValueError("proposal inputs must share a floating dtype")
    if bool((~torch.isfinite(x)).any()) or bool(((x < 0) | (x > 1)).any()):
        raise ValueError("head_fraction must lie in [0,1]")
    if bool((~torch.isfinite(u)).any()) or bool((u < 0).any()):
        raise ValueError("exposure must be nonnegative")
    flat_count = int(x.numel())
    key = _key_bytes(rng_key)
    explicit_ids: np.ndarray | None = None
    if transition_ids is not None:
        ids = (
            transition_ids.detach().cpu().numpy()
            if isinstance(transition_ids, Tensor)
            else np.asarray(transition_ids)
        )
        if ids.shape != tuple(x.shape) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("transition_ids must be an integral array matching the inputs")
        explicit_ids = ids.astype(np.uint64, copy=False).reshape(-1)
        if np.unique(explicit_ids).size != flat_count:
            raise ValueError("transition_ids must be unique within a proposal batch")
    uniforms = np.zeros(flat_count, dtype=np.float64)
    for index in range(flat_count):
        local_key = key
        local_index = index
        if explicit_ids is not None:
            local_key = key + b"\0transition-id:" + int(explicit_ids[index]).to_bytes(8, "big")
            local_index = 0
        prefix = _LazyDyadicPrefix(
            local_key,
            local_index,
            initial_bits=64,
            max_bits=int(profile.max_prefix_bits),
        )
        uniforms[index] = prefix.midpoint()
    uniform = torch.as_tensor(uniforms.reshape(tuple(x.shape)), dtype=x.dtype, device=x.device)
    active = u > 0
    lower = torch.zeros_like(x)
    upper = torch.ones_like(x)
    for _ in range(int(profile.device_bisection_steps)):
        midpoint = 0.5 * (lower + upper)
        safe_u = torch.where(active, u, torch.ones_like(u))
        evaluated = evaluate_alpha1_rb_torch_fixed_modes(
            x, midpoint, safe_u, modes=int(profile.device_proposal_modes)
        )
        go_left = uniform < evaluated.cdf
        upper = torch.where(active & go_left, midpoint, upper)
        lower = torch.where(active & ~go_left, midpoint, lower)
    proposed = torch.where(active, 0.5 * (lower + upper), x)
    lower = torch.where(active, lower, x)
    upper = torch.where(active, upper, x)
    return JacobiRBDeviceProposal(
        earlier_head_fraction=x,
        proposed_later_head_fraction=proposed,
        uniform_midpoint=uniform,
        proposal_lower=lower,
        proposal_upper=upper,
        exposure=u,
        active_mask=active,
        modes_used=int(profile.device_proposal_modes),
        bisection_steps=int(profile.device_bisection_steps),
    )


def propose_alpha1_rb_transition_batch_torch_intervals(
    head_fraction: Tensor,
    exposure: Tensor,
    *,
    rng_key: Any,
    profile: JacobiRBSpectralProfile,
) -> JacobiRBDeviceIntervalProposal:
    """Perform fail-closed outward-interval inverse CDF and RB evaluation.

    Entries whose device comparison is ever ambiguous are marked for Arb
    fallback.  They are never assigned an approximate certificate.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("Torch is unavailable")
    x, u = torch.broadcast_tensors(head_fraction, exposure)
    if x.dtype != torch.float64 or u.dtype != torch.float64:
        raise ValueError("authorizing interval proposals require torch.float64")
    if bool((~torch.isfinite(x)).any()) or bool(((x < 0) | (x > 1)).any()):
        raise ValueError("head_fraction must lie in [0,1]")
    if bool((~torch.isfinite(u)).any()) or bool((u < 0).any()):
        raise ValueError("exposure must be nonnegative")
    flat_count = int(x.numel())
    key = _key_bytes(rng_key)
    uniform_lower = np.empty(flat_count, dtype=np.float64)
    uniform_upper = np.empty(flat_count, dtype=np.float64)
    for index in range(flat_count):
        prefix = _LazyDyadicPrefix(
            key,
            index,
            initial_bits=int(profile.initial_prefix_bits),
            max_bits=int(profile.max_prefix_bits),
        )
        lower_fraction = Fraction(prefix.numerator, prefix.denominator)
        upper_fraction = Fraction(prefix.numerator + 1, prefix.denominator)
        uniform_lower[index] = np.nextafter(float(lower_fraction), -math.inf)
        uniform_upper[index] = np.nextafter(float(upper_fraction), math.inf)
    uniform_lo = torch.as_tensor(
        uniform_lower.reshape(tuple(x.shape)), dtype=torch.float64, device=x.device
    )
    uniform_hi = torch.as_tensor(
        uniform_upper.reshape(tuple(x.shape)), dtype=torch.float64, device=x.device
    )
    active = u > 0.0
    lower = torch.zeros_like(x)
    upper = torch.ones_like(x)
    fallback = torch.zeros_like(active)
    safe_u = torch.where(active, u, torch.ones_like(u))
    for _ in range(int(profile.device_bisection_steps)):
        midpoint = lower + 0.5 * (upper - lower)
        intervals = evaluate_alpha1_rb_torch_intervals(
            x,
            midpoint,
            safe_u,
            modes=int(profile.device_proposal_modes),
        )
        go_left = uniform_hi < intervals.cdf_lower
        go_right = uniform_lo > intervals.cdf_upper
        unresolved = active & ~fallback & ~(go_left | go_right)
        fallback = fallback | unresolved
        decisive = active & ~fallback
        upper = torch.where(decisive & go_left, midpoint, upper)
        lower = torch.where(decisive & go_right, midpoint, lower)
    adjacent = torch.nextafter(lower, torch.full_like(lower, math.inf)) >= upper
    quantile_certified = active & ~fallback & adjacent
    later = torch.where(active, lower + 0.5 * (upper - lower), x)
    target_eval = evaluate_alpha1_rb_torch_intervals(
        x,
        torch.where(active, later, torch.full_like(later, 0.5)),
        safe_u,
        modes=int(profile.device_proposal_modes),
    )
    density_positive = target_eval.density_lower > 0.0
    quotient_candidates = torch.stack(
        (
            target_eval.conormal_lower / target_eval.density_lower,
            target_eval.conormal_lower / target_eval.density_upper,
            target_eval.conormal_upper / target_eval.density_lower,
            target_eval.conormal_upper / target_eval.density_upper,
        ),
        dim=0,
    )
    target_lo = _torch_down(torch.amin(quotient_candidates, dim=0), 2)
    target_hi = _torch_up(torch.amax(quotient_candidates, dim=0), 2)
    target_value = target_lo + 0.5 * (target_hi - target_lo)
    previous = torch.nextafter(target_value, torch.full_like(target_value, -math.inf))
    following = torch.nextafter(target_value, torch.full_like(target_value, math.inf))
    lower_cell_upper = _torch_up(0.5 * previous + 0.5 * target_value, 2)
    upper_cell_lower = _torch_down(0.5 * target_value + 0.5 * following, 2)
    target_certified = (
        density_positive
        & (target_lo > lower_cell_upper)
        & (target_hi < upper_cell_lower)
        & torch.isfinite(target_value)
    )
    # Eager float64 nextafter intervals are a screening implementation.  They
    # are not production-authorizing until a separately validated double-
    # double backend replaces them; the frozen profile therefore sends all
    # rows through transition-local Arb certification.
    certified = quantile_certified & target_certified & bool(
        profile.authorize_device_intervals
    )
    fallback = active & ~certified
    target_value = torch.where(active, target_value, torch.zeros_like(target_value))
    target_lo = torch.where(active, target_lo, torch.zeros_like(target_lo))
    target_hi = torch.where(active, target_hi, torch.zeros_like(target_hi))
    codes = torch.zeros_like(x, dtype=torch.uint8)
    codes = torch.where(certified, torch.full_like(codes, 9), codes)
    codes = torch.where(fallback, torch.full_like(codes, 2 | 4), codes)
    return JacobiRBDeviceIntervalProposal(
        later_head_fraction=later,
        denoising_target=target_value,
        quantile_lower=torch.where(active, lower, x),
        quantile_upper=torch.where(active, upper, x),
        target_lower=target_lo,
        target_upper=target_hi,
        certified_mask=certified,
        fallback_mask=fallback,
        active_mask=active,
        certificate_codes=codes,
        modes_used=int(profile.device_proposal_modes),
        bisection_steps=int(profile.device_bisection_steps),
    )


@dataclass(frozen=True)
class _Interval:
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if math.isnan(self.lower) or math.isnan(self.upper) or self.lower > self.upper:
            raise JacobiRBCertificationError(
                "invalid floating-point interval",
                {"lower": self.lower, "upper": self.upper},
            )

    @property
    def width(self) -> float:
        return float(self.upper - self.lower)


def _down(value: float, count: int = 1) -> float:
    result = float(value)
    for _ in range(max(1, int(count))):
        result = float(np.nextafter(result, -math.inf))
    return result


def _up(value: float, count: int = 1) -> float:
    result = float(value)
    for _ in range(max(1, int(count))):
        result = float(np.nextafter(result, math.inf))
    return result


def _point(value: float, ulps: int = 1) -> _Interval:
    if not math.isfinite(float(value)):
        raise JacobiRBCertificationError("nonfinite interval point", {"value": value})
    return _Interval(_down(float(value), ulps), _up(float(value), ulps))


def _add(left: _Interval, right: _Interval, ulps: int) -> _Interval:
    return _Interval(
        _down(left.lower + right.lower, ulps),
        _up(left.upper + right.upper, ulps),
    )


def _sub(left: _Interval, right: _Interval, ulps: int) -> _Interval:
    return _Interval(
        _down(left.lower - right.upper, ulps),
        _up(left.upper - right.lower, ulps),
    )


def _mul(left: _Interval, right: _Interval, ulps: int) -> _Interval:
    products = (
        left.lower * right.lower,
        left.lower * right.upper,
        left.upper * right.lower,
        left.upper * right.upper,
    )
    return _Interval(_down(min(products), ulps), _up(max(products), ulps))


def _scale(value: _Interval, scalar: float, ulps: int) -> _Interval:
    return _mul(value, _point(float(scalar), ulps), ulps)


def _divide(left: _Interval, right: _Interval, ulps: int) -> _Interval:
    if right.lower <= 0.0 <= right.upper:
        raise JacobiRBCertificationError(
            "interval division encountered a zero denominator",
            {"denominator": [right.lower, right.upper]},
        )
    reciprocal = _Interval(
        _down(1.0 / right.upper, ulps),
        _up(1.0 / right.lower, ulps),
    )
    return _mul(left, reciprocal, ulps)


def _exp(value: _Interval, ulps: int) -> _Interval:
    return _Interval(
        _down(math.exp(value.lower), ulps),
        _up(math.exp(value.upper), ulps),
    )


def _expand(value: _Interval, radius: float, ulps: int) -> _Interval:
    bound = max(0.0, float(radius))
    return _Interval(
        _down(value.lower - bound, ulps),
        _up(value.upper + bound, ulps),
    )


def _exact_float_interval(value: float) -> _Interval:
    """A binary64 input is an exact mathematical input to this module."""

    if not math.isfinite(float(value)):
        raise ValueError("Jacobi inputs must be finite")
    return _Interval(float(value), float(value))


def _legendre_next(
    previous: _Interval,
    current: _Interval,
    coordinate: _Interval,
    degree: int,
    ulps: int,
) -> _Interval:
    first = _scale(_mul(coordinate, current, ulps), float(2 * degree + 1), ulps)
    second = _scale(previous, float(degree), ulps)
    return _scale(_sub(first, second, ulps), 1.0 / float(degree + 1), ulps)


def _tail_bounds(first_omitted: int, exposure: float, ulps: int) -> tuple[float, float, float]:
    """Uniform absolute tails for CDF, density, and conormal numerator."""

    mode = int(first_omitted)
    u = float(exposure)
    exponent = -float(mode * (mode + 1)) * u
    decay = _up(math.exp(exponent), ulps)
    common_ratio = _up(math.exp(-2.0 * float(mode + 1) * u), ulps)
    if not common_ratio < 1.0:
        return math.inf, math.inf, math.inf
    cdf_tail = _up(decay / _down(1.0 - common_ratio, ulps), ulps)

    density_ratio = _up(
        (float(2 * mode + 3) / float(2 * mode + 1)) * common_ratio,
        ulps,
    )
    density_tail = math.inf
    if density_ratio < 1.0:
        density_tail = _up(
            float(2 * mode + 1)
            * decay
            / _down(1.0 - density_ratio, ulps),
            ulps,
        )

    conormal_ratio = _up(
        (float(2 * mode + 3) / float(2 * mode + 1))
        * (float(mode + 1) / float(mode))
        * common_ratio,
        ulps,
    )
    conormal_tail = math.inf
    if conormal_ratio < 1.0:
        conormal_tail = _up(
            float((2 * mode + 1) * mode)
            * decay
            / _down(1.0 - conormal_ratio, ulps),
            ulps,
        )
    return cdf_tail, density_tail, conormal_tail


def _spectral_intervals_fast(
    x: float,
    y: float,
    exposure: float,
    *,
    profile: JacobiRBSpectralProfile,
    cdf_only: bool,
    tail_tolerance: float,
) -> tuple[_Interval, _Interval | None, _Interval | None, int]:
    """Propagated binary64 intervals plus analytic truncation tails."""

    ulps = int(profile.roundoff_ulps)
    x_i = _exact_float_interval(x)
    y_i = _exact_float_interval(y)
    u_i = _exact_float_interval(exposure)
    zx = _sub(_scale(x_i, 2.0, ulps), _exact_float_interval(1.0), ulps)
    zy = _sub(_scale(y_i, 2.0, ulps), _exact_float_interval(1.0), ulps)

    px_previous = _exact_float_interval(1.0)
    px_current = zx
    py_previous = _exact_float_interval(1.0)
    py_current = zy
    cdf = y_i
    density = _exact_float_interval(1.0)
    conormal = _exact_float_interval(0.0)
    modes_used = 1
    final_tails = (math.inf, math.inf, math.inf)

    for degree in range(1, int(profile.max_modes)):
        py_next = _legendre_next(py_previous, py_current, zy, degree, ulps)
        exponent = _scale(u_i, -float(degree * (degree + 1)), ulps)
        decay = _exp(exponent, ulps)
        cdf_term = _scale(
            _mul(
                _mul(decay, px_current, ulps),
                _sub(py_next, py_previous, ulps),
                ulps,
            ),
            0.5,
            ulps,
        )
        cdf = _add(cdf, cdf_term, ulps)

        if not cdf_only:
            coefficient = _scale(decay, float(2 * degree + 1), ulps)
            density = _add(
                density,
                _mul(_mul(coefficient, px_current, ulps), py_current, ulps),
                ulps,
            )
            # y(1-y) d_y P_n(2y-1)
            #   = n/2 [P_{n-1}(2y-1) - (2y-1)P_n(2y-1)].
            conormal_basis = _scale(
                _sub(py_previous, _mul(zy, py_current, ulps), ulps),
                0.5 * float(degree),
                ulps,
            )
            conormal = _add(
                conormal,
                _mul(_mul(coefficient, px_current, ulps), conormal_basis, ulps),
                ulps,
            )

        modes_used = degree + 1
        final_tails = _tail_bounds(degree + 1, exposure, ulps)
        relevant_tail = final_tails[0] if cdf_only else max(final_tails)
        if relevant_tail <= float(tail_tolerance):
            break

        px_next = _legendre_next(px_previous, px_current, zx, degree, ulps)
        px_previous, px_current = px_current, px_next
        py_previous, py_current = py_current, py_next
    else:
        raise JacobiRBCertificationError(
            "Legendre mode cap reached before an analytic tail certificate",
            {
                "x": x,
                "y": y,
                "exposure": exposure,
                "max_modes": profile.max_modes,
                "cdf_only": int(cdf_only),
                "failure_kind": "mode_cap",
            },
        )

    cdf = _expand(cdf, final_tails[0], ulps)
    if cdf_only:
        return cdf, None, None, modes_used
    return (
        cdf,
        _expand(density, final_tails[1], ulps),
        _expand(conormal, final_tails[2], ulps),
        modes_used,
    )


def _arb_exact(value: float | Fraction | int) -> Any:
    """Construct an Arb value containing one exact rational input."""

    if _arb is None:  # pragma: no cover - depends on optional python-flint.
        raise JacobiRBCertificationError(
            "python-flint/Arb is unavailable",
            {"failure_kind": "arb_backend_unavailable"},
        )
    if isinstance(value, Fraction):
        numerator, denominator = value.numerator, value.denominator
    elif isinstance(value, int):
        numerator, denominator = int(value), 1
    else:
        numerator, denominator = float(value).as_integer_ratio()
    return _arb(int(numerator)) / _arb(int(denominator))


def _arb_bounds(value: Any) -> _Interval:
    """Convert rigorous Arb endpoints to a conservative binary64 interval."""

    try:
        lower = float(value.lower())
        upper = float(value.upper())
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise JacobiRBCertificationError(
            "could not extract finite Arb endpoints",
            {"failure_kind": "arb_endpoint_conversion"},
        ) from exc
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise JacobiRBCertificationError(
            "Arb returned nonfinite endpoints",
            {"lower": lower, "upper": upper, "failure_kind": "arb_nonfinite"},
        )
    # Endpoint-to-double conversion may round inward by at most one ulp.
    return _Interval(_down(lower, 2), _up(upper, 2))


def _arb_bounds_or_unbounded(value: Any) -> _Interval:
    """Keep an inconclusive low-precision Arb ball available for escalation.

    Cancellation at small exposure can make a rigorous early-precision Arb
    radius overflow binary64.  That does not invalidate the transition: it
    means only that this precision is inconclusive.  Authorizing comparisons
    remain in Arb, while ``[-inf,+inf]`` is used as a diagnostic sentinel and
    the caller advances to the next frozen precision.
    """

    try:
        return _arb_bounds(value)
    except JacobiRBCertificationError as exc:
        if exc.diagnostics.get("failure_kind") != "arb_nonfinite":
            raise
        return _Interval(-math.inf, math.inf)


def _arb_error_ball(radius: Any) -> Any:
    """Return an Arb zero ball containing an already rigorous radius.

    ``python-flint`` interprets an Arb radius by its absolute upper bound.  In
    particular, no conversion through an ordinary float is permitted here:
    this helper is used only on authorizing omitted-series bounds.
    """

    assert _arb is not None
    return _arb(0, radius)


def _arb_geometric_tail_radii(
    first_omitted: int,
    exposure: Any,
) -> tuple[Any | None, Any | None, Any | None]:
    """Rigorous CDF, density, and conormal omitted-tail radii.

    The Legendre bounds ``|P_n|<=1`` and
    ``|n(P_{n-1}-zP_n)/2|<=n`` reduce all three tails to decreasing geometric
    majorants.  Every coefficient, exponential, ratio, and denominator below
    remains in Arb.  ``None`` means the corresponding ratio is not yet
    provably contractive at the active precision; callers must continue the
    recurrence rather than infer a radius from binary64 arithmetic.
    """

    if _arb is None:
        raise JacobiRBCertificationError(
            "rigorous spectral tails require python-flint/Arb",
            {"failure_kind": "arb_backend_unavailable"},
        )
    mode = int(first_omitted)
    if mode < 1:
        raise ValueError("first_omitted must be positive")
    one = _arb(1)
    decay = (-mode * (mode + 1) * exposure).exp()
    step_decay = (-2 * (mode + 1) * exposure).exp()
    if not step_decay < one:
        return None, None, None
    cdf_radius = decay / (one - step_decay)

    density_ratio = (
        _arb_exact(Fraction(2 * mode + 3, 2 * mode + 1)) * step_decay
    )
    density_radius = None
    if density_ratio < one:
        density_radius = (2 * mode + 1) * decay / (one - density_ratio)

    conormal_ratio = (
        _arb_exact(Fraction(2 * mode + 3, 2 * mode + 1))
        * _arb_exact(Fraction(mode + 1, mode))
        * step_decay
    )
    conormal_radius = None
    if conormal_ratio < one:
        conormal_radius = (
            mode * (2 * mode + 1) * decay / (one - conormal_ratio)
        )
    return cdf_radius, density_radius, conormal_radius


def _spectral_intervals_arb(
    x: float,
    y: float | Fraction,
    exposure: float,
    *,
    profile: JacobiRBSpectralProfile,
    cdf_only: bool,
    precision_bits: int,
) -> tuple[_Interval, _Interval | None, _Interval | None, int]:
    if _arb is None or _flint_ctx is None:
        raise JacobiRBCertificationError(
            "ambiguous spectral comparison requires python-flint/Arb",
            {"failure_kind": "arb_backend_unavailable"},
        )
    bits = int(precision_bits)
    if bits < 64 or bits > int(profile.max_arb_precision_bits):
        raise JacobiRBCertificationError(
            "requested Arb precision lies outside the frozen cap",
            {
                "precision_bits": bits,
                "max_arb_precision_bits": profile.max_arb_precision_bits,
                "failure_kind": "arb_precision_cap",
            },
        )
    with _ARB_CONTEXT_LOCK:
        previous_precision = int(_flint_ctx.prec)
        try:
            _flint_ctx.prec = bits
            one = _arb(1)
            zx = 2 * _arb_exact(x) - one
            zy = 2 * _arb_exact(y) - one
            u_value = _arb_exact(exposure)
            px_previous, px_current = one, zx
            py_previous, py_current = one, zy
            cdf = _arb_exact(y)
            density = one
            conormal = _arb(0)
            modes_used = 1
            final_tails: tuple[Any | None, Any | None, Any | None] = (
                None,
                None,
                None,
            )
            exact_tail_tolerance = _arb_exact(profile.arb_tail_tolerance)
            for degree in range(1, int(profile.max_modes)):
                py_next = (
                    (2 * degree + 1) * zy * py_current
                    - degree * py_previous
                ) / (degree + 1)
                decay = (-degree * (degree + 1) * u_value).exp()
                cdf += _arb_exact(Fraction(1, 2)) * decay * px_current * (
                    py_next - py_previous
                )
                if not cdf_only:
                    coefficient = (2 * degree + 1) * decay
                    density += coefficient * px_current * py_current
                    basis = _arb_exact(Fraction(degree, 2)) * (
                        py_previous - zy * py_current
                    )
                    conormal += coefficient * px_current * basis
                modes_used = degree + 1
                # Binary64 is only a conservative performance screen.  It may
                # cause an early *attempt* to stop, but the authorizing
                # decision is the independent Arb majorant below.
                screen = _tail_bounds(degree + 1, exposure, 8)
                screen_tail = screen[0] if cdf_only else max(screen)
                if screen_tail <= float(profile.arb_tail_tolerance):
                    candidate_tails = _arb_geometric_tail_radii(
                        degree + 1, u_value
                    )
                    required = (
                        candidate_tails[:1]
                        if cdf_only
                        else candidate_tails
                    )
                    if all(
                        radius is not None
                        and radius < exact_tail_tolerance
                        for radius in required
                    ):
                        final_tails = candidate_tails
                        break
                px_next = (
                    (2 * degree + 1) * zx * px_current
                    - degree * px_previous
                ) / (degree + 1)
                px_previous, px_current = px_current, px_next
                py_previous, py_current = py_current, py_next
            else:
                raise JacobiRBCertificationError(
                    "Arb Legendre mode cap reached",
                    {
                        "failure_kind": "arb_mode_cap",
                        "max_modes": profile.max_modes,
                        "precision_bits": bits,
                    },
                )

            if final_tails[0] is None:
                raise JacobiRBCertificationError(
                    "Arb CDF tail was not certified",
                    {"failure_kind": "arb_tail_unresolved"},
                )
            cdf += _arb_error_ball(final_tails[0])
            cdf_interval = _arb_bounds(cdf)
            if cdf_only:
                return cdf_interval, None, None, modes_used
            if final_tails[1] is None or final_tails[2] is None:
                raise JacobiRBCertificationError(
                    "Arb density/conormal tails were not certified",
                    {"failure_kind": "arb_tail_unresolved"},
                )
            density += _arb_error_ball(final_tails[1])
            conormal += _arb_error_ball(final_tails[2])
            return (
                cdf_interval,
                _arb_bounds(density),
                _arb_bounds(conormal),
                modes_used,
            )
        finally:
            _flint_ctx.prec = previous_precision


def _rounding_cell(candidate: float) -> tuple[Fraction, Fraction]:
    """Return the exact open round-to-nearest-even cell around a finite float.

    Exact midpoint ties are deliberately left unresolved.  They have zero
    probability under the continuous transition and escalating rather than
    silently choosing a tie keeps the numerical certificate simple.
    """

    value = float(candidate)
    if not math.isfinite(value):
        raise JacobiRBCertificationError(
            "nonfinite candidate cannot be correctly rounded",
            {"failure_kind": "target_rounding_nonfinite"},
        )
    previous = float(np.nextafter(value, -math.inf))
    following = float(np.nextafter(value, math.inf))
    return (
        (Fraction.from_float(previous) + Fraction.from_float(value)) / 2,
        (Fraction.from_float(value) + Fraction.from_float(following)) / 2,
    )


def _unique_rounding_value(interval: _Interval) -> float | None:
    candidate = float(interval.lower + 0.5 * (interval.upper - interval.lower))
    if not math.isfinite(candidate):
        return None
    lower_boundary, upper_boundary = _rounding_cell(candidate)
    if (
        Fraction.from_float(interval.lower) > lower_boundary
        and Fraction.from_float(interval.upper) < upper_boundary
    ):
        return candidate
    return None


def _arb_target_rounding(
    x: float,
    y: float,
    exposure: float,
    *,
    profile: JacobiRBSpectralProfile,
    precision_bits: int,
) -> tuple[float | None, _Interval, int]:
    """Evaluate ``G/k`` in Arb and certify one binary64 rounding cell."""

    if _arb is None or _flint_ctx is None:
        raise JacobiRBCertificationError(
            "target rounding requires python-flint/Arb",
            {"failure_kind": "arb_backend_unavailable"},
        )
    bits = int(precision_bits)
    with _ARB_CONTEXT_LOCK:
        previous_precision = int(_flint_ctx.prec)
        try:
            _flint_ctx.prec = bits
            one = _arb(1)
            zx = 2 * _arb_exact(x) - one
            zy = 2 * _arb_exact(y) - one
            u_value = _arb_exact(exposure)
            px_previous, px_current = one, zx
            py_previous, py_current = one, zy
            density = one
            conormal = _arb(0)
            modes_used = 1
            first_omitted = 1
            target_tails: tuple[Any | None, Any | None, Any | None] = (
                None,
                None,
                None,
            )
            for degree in range(1, int(profile.max_modes)):
                py_next = (
                    (2 * degree + 1) * zy * py_current
                    - degree * py_previous
                ) / (degree + 1)
                decay = (-degree * (degree + 1) * u_value).exp()
                coefficient = (2 * degree + 1) * decay
                density += coefficient * px_current * py_current
                basis = _arb_exact(Fraction(degree, 2)) * (
                    py_previous - zy * py_current
                )
                conormal += coefficient * px_current * basis
                modes_used = degree + 1
                first_omitted = degree + 1
                # Correctly rounding a target near zero needs an omitted tail
                # far below the normal binary64 scale.  A fixed 1e-30 cutoff
                # is insufficient (and was the source of false failures in
                # nearly stationary phases).  -1200 is below half the
                # smallest binary64 subnormal while remaining easy for Arb.
                log_first_conormal = (
                    math.log(float(first_omitted * (2 * first_omitted + 1)))
                    - float(first_omitted * (first_omitted + 1)) * exposure
                )
                if log_first_conormal < -1200.0:
                    candidate_tails = _arb_geometric_tail_radii(
                        first_omitted, u_value
                    )
                    if (
                        candidate_tails[1] is not None
                        and candidate_tails[2] is not None
                    ):
                        target_tails = candidate_tails
                        break
                px_next = (
                    (2 * degree + 1) * zx * px_current
                    - degree * px_previous
                ) / (degree + 1)
                px_previous, px_current = px_current, px_next
                py_previous, py_current = py_current, py_next
            else:
                raise JacobiRBCertificationError(
                    "Arb target mode cap reached",
                    {
                        "failure_kind": "arb_mode_cap",
                        "max_modes": profile.max_modes,
                        "precision_bits": bits,
                    },
                )
            density_tail = target_tails[1]
            conormal_tail = target_tails[2]
            if density_tail is None or conormal_tail is None:
                raise JacobiRBCertificationError(
                    "Arb target tails were not certified",
                    {"failure_kind": "arb_tail_unresolved"},
                )
            density += _arb_error_ball(density_tail)
            conormal += _arb_error_ball(conormal_tail)
            if not density > _arb(0):
                return None, _Interval(-math.inf, math.inf), modes_used
            target = conormal / density
            target_interval = _arb_bounds_or_unbounded(target)
            if not math.isfinite(target_interval.lower) or not math.isfinite(
                target_interval.upper
            ):
                return None, target_interval, modes_used
            candidate = float(target.mid())
            if not math.isfinite(candidate):
                return None, target_interval, modes_used
            lower_boundary, upper_boundary = _rounding_cell(candidate)
            if target > _arb_exact(lower_boundary) and target < _arb_exact(upper_boundary):
                return candidate, target_interval, modes_used
            return None, target_interval, modes_used
        finally:
            _flint_ctx.prec = previous_precision


def _arb_cdf_prefix_decision(
    prefix: "_LazyDyadicPrefix",
    x: float,
    y: float | Fraction,
    exposure: float,
    *,
    profile: JacobiRBSpectralProfile,
    precision_bits: int,
) -> tuple[int, int, _Interval]:
    """Compare an exact dyadic prefix with an Arb CDF ball.

    Keeping both operands in Arb is essential: converting the CDF ball to a
    binary64 interval would impose a permanent one-ulp width and could never
    decide a correctly-rounded quantile boundary.
    """

    if _arb is None or _flint_ctx is None:
        raise JacobiRBCertificationError(
            "ambiguous spectral comparison requires python-flint/Arb",
            {"failure_kind": "arb_backend_unavailable"},
        )
    bits = int(precision_bits)
    with _ARB_CONTEXT_LOCK:
        previous_precision = int(_flint_ctx.prec)
        try:
            _flint_ctx.prec = bits
            one = _arb(1)
            zx = 2 * _arb_exact(x) - one
            zy = 2 * _arb_exact(y) - one
            u_value = _arb_exact(exposure)
            px_previous, px_current = one, zx
            py_previous, py_current = one, zy
            cdf = _arb_exact(y)
            modes_used = 1
            final_tail = None
            exact_tail_tolerance = _arb_exact(profile.arb_tail_tolerance)
            for degree in range(1, int(profile.max_modes)):
                py_next = (
                    (2 * degree + 1) * zy * py_current
                    - degree * py_previous
                ) / (degree + 1)
                decay = (-degree * (degree + 1) * u_value).exp()
                cdf += _arb_exact(Fraction(1, 2)) * decay * px_current * (
                    py_next - py_previous
                )
                modes_used = degree + 1
                screen_tail = _tail_bounds(degree + 1, exposure, 8)[0]
                if screen_tail <= float(profile.arb_tail_tolerance):
                    candidate_tail = _arb_geometric_tail_radii(
                        degree + 1, u_value
                    )[0]
                    if (
                        candidate_tail is not None
                        and candidate_tail < exact_tail_tolerance
                    ):
                        final_tail = candidate_tail
                        break
                px_next = (
                    (2 * degree + 1) * zx * px_current
                    - degree * px_previous
                ) / (degree + 1)
                px_previous, px_current = px_current, px_next
                py_previous, py_current = py_current, py_next
            else:
                raise JacobiRBCertificationError(
                    "Arb CDF mode cap reached",
                    {
                        "failure_kind": "arb_mode_cap",
                        "max_modes": profile.max_modes,
                        "precision_bits": bits,
                    },
                )
            if final_tail is None:
                raise JacobiRBCertificationError(
                    "Arb CDF tail was not certified",
                    {"failure_kind": "arb_tail_unresolved"},
                )
            cdf += _arb_error_ball(final_tail)
            lower_uniform = _arb_exact(Fraction(prefix.numerator, prefix.denominator))
            upper_uniform = _arb_exact(
                Fraction(prefix.numerator + 1, prefix.denominator)
            )
            if upper_uniform < cdf:
                return -1, modes_used, _arb_bounds_or_unbounded(cdf)
            if lower_uniform > cdf:
                return 1, modes_used, _arb_bounds_or_unbounded(cdf)
            return 0, modes_used, _arb_bounds_or_unbounded(cdf)
        finally:
            _flint_ctx.prec = previous_precision


def _key_bytes(rng_key: Any) -> bytes:
    if isinstance(rng_key, bytes):
        return b"bytes:" + rng_key
    if isinstance(rng_key, str):
        return b"str:" + rng_key.encode("utf-8")
    try:
        encoded = json.dumps(rng_key, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TypeError("rng_key must be bytes, text, or JSON-serializable") from exc
    return b"json:" + encoded.encode("utf-8")


class _LazyDyadicPrefix:
    def __init__(
        self,
        key: bytes,
        sample_index: int,
        *,
        initial_bits: int,
        max_bits: int,
    ) -> None:
        self.key = bytes(key)
        self.sample_index = int(sample_index)
        self.max_bits = int(max_bits)
        self.numerator = 0
        self.bits = 0
        # Each transition gets a disjoint Philox namespace.  Refining this
        # transition therefore cannot consume or shift any later transition's
        # primary bits.  SHA-256 is used only as a collision-resistant key
        # derivation function; all random words come from Philox.
        digest = hashlib.sha256(
            b"d0-jacobi-rb-philox-key\0"
            + self.key
            + self.sample_index.to_bytes(8, "big", signed=False)
        ).digest()
        seed_words = np.frombuffer(digest, dtype="<u8").copy()
        self._bit_generator = np.random.Philox(seed_words)
        self._append_bits(int(initial_bits))

    def _word(self) -> int:
        return int(self._bit_generator.random_raw())

    def _append_bits(self, count: int) -> None:
        remaining = int(count)
        while remaining:
            word = self._word()
            take = min(64, remaining)
            fragment = word >> (64 - take)
            self.numerator = (self.numerator << take) | fragment
            self.bits += take
            remaining -= take

    def refine(self, count: int) -> None:
        if self.bits + int(count) > self.max_bits:
            raise JacobiRBCertificationError(
                "lazy dyadic prefix cap reached",
                {
                    "sample_index": self.sample_index,
                    "prefix_bits": self.bits,
                    "max_prefix_bits": self.max_bits,
                    "failure_kind": "random_bit_cap",
                },
            )
        self._append_bits(int(count))

    @property
    def denominator(self) -> int:
        return 1 << self.bits

    def midpoint(self) -> float:
        return float(Fraction(2 * self.numerator + 1, 2 * self.denominator))

    def upper_leq_float(self, value: float) -> bool:
        return Fraction(self.numerator + 1, self.denominator) <= Fraction.from_float(
            float(value)
        )

    def lower_geq_float(self, value: float) -> bool:
        return Fraction(self.numerator, self.denominator) >= Fraction.from_float(
            float(value)
        )


def philox_uniform_prefix(
    rng_key: Any, *, sample_index: int = 0, bits: int = 64
) -> tuple[int, int, float]:
    """Expose a deterministic prefix for support planning and replay tests."""

    if not 1 <= int(bits) <= 1024:
        raise ValueError("bits must lie in [1,1024]")
    prefix = _LazyDyadicPrefix(
        _key_bytes(rng_key), int(sample_index), initial_bits=min(int(bits), 64), max_bits=int(bits)
    )
    if int(bits) > prefix.bits:
        prefix.refine(int(bits) - prefix.bits)
    return int(prefix.numerator), int(prefix.bits), float(prefix.midpoint())

def _prefix_comparison(prefix: _LazyDyadicPrefix, cdf: _Interval) -> int:
    """Return -1 for U<F, +1 for U>F, and zero when unresolved."""

    if prefix.upper_leq_float(cdf.lower):
        return -1
    if prefix.lower_geq_float(cdf.upper):
        return 1
    return 0


def _analytic_moments(x: float, exposure: float) -> tuple[float, float]:
    z = 2.0 * float(x) - 1.0
    first = math.exp(-2.0 * float(exposure)) * z
    p2 = 0.5 * (3.0 * z * z - 1.0)
    second = (1.0 + 2.0 * math.exp(-6.0 * float(exposure)) * p2) / 3.0
    mean = 0.5 * (1.0 + first)
    variance = max(0.0, 0.25 * (second - first * first))
    return mean, variance


def cantelli_quantile_bracket(
    head_fraction: float,
    exposure: float,
    uniform: float,
) -> tuple[float, float]:
    """Return a conservative moment-based quantile bracket candidate.

    The production sampler verifies both endpoints against certified CDF
    intervals and falls back to ``[0,1]`` if floating-point evaluation of the
    analytic moment formula is not decisive.
    """

    x = float(head_fraction)
    u = float(exposure)
    probability = float(uniform)
    if not 0.0 < probability < 1.0:
        raise ValueError("uniform must lie strictly inside (0,1)")
    if not 0.0 <= x <= 1.0 or not u > 0.0:
        raise ValueError("head_fraction must lie in [0,1] and exposure be positive")
    mean, variance = _analytic_moments(x, u)
    lower_radius = math.sqrt(variance * (1.0 - probability) / probability)
    upper_radius = math.sqrt(variance * probability / (1.0 - probability))
    return (
        max(0.0, _down(mean - lower_radius, 4)),
        min(1.0, _up(mean + upper_radius, 4)),
    )


def _cdf_interval(
    x: float,
    y: float | Fraction,
    exposure: float,
    *,
    profile: JacobiRBSpectralProfile,
    force_interval: bool = False,
    precision_bits: int | None = None,
) -> tuple[_Interval, int, bool]:
    if y == 0:
        return _Interval(0.0, 0.0), 0, False
    if y == 1:
        return _Interval(1.0, 1.0), 0, False
    if not force_interval and isinstance(y, float):
        try:
            result, _, _, modes = _spectral_intervals_fast(
                x,
                y,
                exposure,
                profile=profile,
                cdf_only=True,
                tail_tolerance=profile.fast_cdf_tail_tolerance,
            )
            return result, modes, False
        except JacobiRBCertificationError:
            if not profile.allow_interval_escalation:
                raise
    if not profile.allow_interval_escalation:
        raise JacobiRBCertificationError(
            "CDF evaluation requires Arb but interval escalation is disabled",
            {"failure_kind": "interval_escalation_disabled"},
        )
    precision = int(precision_bits or profile.arb_precision_bits[0])
    result, _, _, modes = _spectral_intervals_arb(
        x,
        y,
        exposure,
        profile=profile,
        cdf_only=True,
        precision_bits=precision,
    )
    return result, modes, True


def _decide_cdf(
    prefix: _LazyDyadicPrefix,
    x: float,
    y: float | Fraction,
    exposure: float,
    profile: JacobiRBSpectralProfile,
) -> tuple[int, int, bool]:
    # The portable binary64 enclosure remains useful for non-authorizing test
    # profiles.  A production correctly-rounded decision goes directly to
    # Arb so no branch can depend on an accidentally underbounded float tail.
    if y == 0 or y == 1 or not profile.require_correct_rounding:
        cdf, maximum_modes, escalated = _cdf_interval(
            x,
            y,
            exposure,
            profile=profile,
            force_interval=not isinstance(y, float),
        )
        decision = _prefix_comparison(prefix, cdf)
        if decision:
            return decision, maximum_modes, escalated
    else:
        cdf = _Interval(0.0, 1.0)
        maximum_modes = 0
        escalated = True
    if not profile.allow_interval_escalation:
        raise JacobiRBCertificationError(
            "CDF interval overlapped the dyadic uniform prefix",
            {"failure_kind": "ambiguous_cdf"},
        )
    for precision in profile.arb_precision_bits:
        decision, modes, cdf = _arb_cdf_prefix_decision(
            prefix,
            x,
            y,
            exposure,
            profile=profile,
            precision_bits=int(precision),
        )
        maximum_modes = max(maximum_modes, modes)
        escalated = True
        if decision:
            return decision, maximum_modes, escalated
    while prefix.bits < int(profile.max_prefix_bits):
        prefix.refine(min(int(profile.prefix_block_bits), profile.max_prefix_bits - prefix.bits))
        decision, modes, cdf = _arb_cdf_prefix_decision(
            prefix,
            x,
            y,
            exposure,
            profile=profile,
            precision_bits=int(profile.max_arb_precision_bits),
        )
        maximum_modes = max(maximum_modes, modes)
        if decision:
            return decision, maximum_modes, escalated
    raise JacobiRBCertificationError(
        "CDF comparison remained ambiguous after precision and random-bit refinement",
        {
            "x": x,
            "y": str(y),
            "exposure": exposure,
            "cdf_interval": [cdf.lower, cdf.upper],
            "prefix_bits": prefix.bits,
            "failure_kind": "ambiguous_cdf",
        },
    )


def _verified_initial_bracket(
    x: float,
    exposure: float,
    prefix: _LazyDyadicPrefix,
    profile: JacobiRBSpectralProfile,
) -> tuple[float, float, int, int]:
    lower, upper = cantelli_quantile_bracket(x, exposure, prefix.midpoint())
    maximum_modes = 0
    escalations = 0
    if lower > 0.0:
        decision, modes, escalated = _decide_cdf(
            prefix, x, lower, exposure, profile
        )
        maximum_modes = max(maximum_modes, modes)
        escalations += int(escalated)
        if decision != 1:  # Need U > F(lower).
            lower = 0.0
    if upper < 1.0:
        decision, modes, escalated = _decide_cdf(
            prefix, x, upper, exposure, profile
        )
        maximum_modes = max(maximum_modes, modes)
        escalations += int(escalated)
        if decision != -1:  # Need U < F(upper).
            upper = 1.0
    if not lower < upper:
        raise JacobiRBCertificationError(
            "verified Cantelli bracket is empty",
            {"lower": lower, "upper": upper, "failure_kind": "invalid_bracket"},
        )
    return lower, upper, maximum_modes, escalations


def _rounding_boundary(lower: float, upper: float) -> Fraction:
    return (Fraction.from_float(lower) + Fraction.from_float(upper)) / 2


def _invert_one(
    x: float,
    exposure: float,
    prefix: _LazyDyadicPrefix,
    profile: JacobiRBSpectralProfile,
) -> tuple[float, float, float, int, int, int, bool]:
    lower, upper, maximum_modes, escalations = _verified_initial_bracket(
        x, exposure, prefix, profile
    )
    correctly_rounded = False
    for step in range(1, int(profile.max_bisection_steps) + 1):
        if profile.require_correct_rounding and np.nextafter(lower, math.inf) >= upper:
            if lower == upper:
                correctly_rounded = True
                return lower, lower, upper, step - 1, maximum_modes, escalations, True
            boundary = _rounding_boundary(lower, upper)
            decision, modes, escalated = _decide_cdf(
                prefix, x, boundary, exposure, profile
            )
            maximum_modes = max(maximum_modes, modes)
            escalations += int(escalated)
            result = lower if decision == -1 else upper
            correctly_rounded = True
            return result, lower, upper, step - 1, maximum_modes, escalations, True
        if not profile.require_correct_rounding and (
            upper - lower <= float(profile.quantile_tolerance)
        ):
            result = lower + 0.5 * (upper - lower)
            return result, lower, upper, step - 1, maximum_modes, escalations, False

        midpoint = lower + 0.5 * (upper - lower)
        if midpoint == lower or midpoint == upper:
            if profile.require_correct_rounding:
                boundary = _rounding_boundary(lower, upper)
                decision, modes, escalated = _decide_cdf(
                    prefix, x, boundary, exposure, profile
                )
                maximum_modes = max(maximum_modes, modes)
                escalations += int(escalated)
                result = lower if decision == -1 else upper
                return result, lower, upper, step, maximum_modes, escalations, True
            return midpoint, lower, upper, step, maximum_modes, escalations, False
        decision, modes, escalated = _decide_cdf(
            prefix, x, midpoint, exposure, profile
        )
        maximum_modes = max(maximum_modes, modes)
        escalations += int(escalated)
        if decision == -1:
            upper = midpoint
        else:
            lower = midpoint
    raise JacobiRBCertificationError(
        "inverse CDF bisection cap reached",
        {
            "x": x,
            "exposure": exposure,
            "lower": lower,
            "upper": upper,
            "prefix_bits": prefix.bits,
            "correct_rounding_required": int(profile.require_correct_rounding),
            "failure_kind": "bisection_cap",
        },
    )


def _target_interval(
    x: float,
    y: float,
    exposure: float,
    profile: JacobiRBSpectralProfile,
) -> tuple[float, _Interval, int, bool]:
    modes = 0
    escalated = False
    if not profile.require_correct_rounding:
        try:
            _, density, conormal, modes = _spectral_intervals_fast(
                x,
                y,
                exposure,
                profile=profile,
                cdf_only=False,
                tail_tolerance=profile.fast_target_tail_tolerance,
            )
            assert density is not None and conormal is not None
            if density.lower > 0.0:
                target = _divide(conormal, density, int(profile.roundoff_ulps))
                rounded = _unique_rounding_value(target)
                if rounded is not None:
                    return rounded, target, modes, escalated
        except JacobiRBCertificationError:
            if not profile.allow_interval_escalation:
                raise
    if not profile.allow_interval_escalation:
        raise JacobiRBCertificationError(
            "Rao--Blackwell target was not certified on the fast path",
            {
                "density_interval": [density.lower, density.upper]
                if "density" in locals() and density is not None
                else None,
                "failure_kind": "target_interval",
            },
        )
    for precision in profile.arb_precision_bits:
        rounded, target, arb_modes = _arb_target_rounding(
            x,
            y,
            exposure,
            profile=profile,
            precision_bits=int(precision),
        )
        modes = max(modes, arb_modes)
        escalated = True
        if rounded is not None:
            return rounded, target, modes, escalated
    raise JacobiRBCertificationError(
        "Rao--Blackwell target interval did not meet the frozen tolerance",
        {
            "x": x,
            "y": y,
            "exposure": exposure,
            "density_interval": [density.lower, density.upper]
            if "density" in locals() and density is not None
            else None,
            "target_interval": [target.lower, target.upper]
            if "target" in locals()
            else None,
            "failure_kind": "target_interval",
        },
    )


def sample_alpha1_rb_transition_batch(
    head_fraction: np.ndarray | float,
    exposure: np.ndarray | float,
    *,
    rng_key: Any,
    profile: JacobiRBSpectralProfile,
    transition_ids: np.ndarray | Tensor | None = None,
) -> JacobiRBTransitionBatch:
    """Sample alpha=1 Jacobi transitions and exact-population RB targets.

    ``exposure == 0`` is an exact no-op: the later fraction equals the earlier
    fraction, the target is the masked sentinel zero, and no random bits are
    consumed for that entry.  Negative exposure is rejected.
    """

    x_array, u_array = np.broadcast_arrays(
        np.asarray(head_fraction, dtype=np.float64),
        np.asarray(exposure, dtype=np.float64),
    )
    if not np.all(np.isfinite(x_array)) or np.any(x_array < 0.0) or np.any(
        x_array > 1.0
    ):
        raise ValueError("head_fraction must be finite and lie in [0,1]")
    if not np.all(np.isfinite(u_array)) or np.any(u_array < 0.0):
        raise ValueError("exposure must be finite and nonnegative")
    key = _key_bytes(rng_key)
    explicit_ids: np.ndarray | None = None
    if transition_ids is not None:
        ids = (
            transition_ids.detach().cpu().numpy()
            if isinstance(transition_ids, Tensor)
            else np.asarray(transition_ids)
        )
        if ids.shape != x_array.shape or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("transition_ids must be an integral array matching the inputs")
        explicit_ids = ids.astype(np.uint64, copy=False).reshape(-1)
        if np.unique(explicit_ids).size != int(x_array.size):
            raise ValueError("transition_ids must be unique within a certified batch")
    later = np.array(x_array, copy=True)
    target = np.zeros_like(x_array)
    active = u_array > 0.0
    q_lower = np.array(x_array, copy=True)
    q_upper = np.array(x_array, copy=True)
    z_lower = np.zeros_like(x_array)
    z_upper = np.zeros_like(x_array)
    prefix_bits = np.zeros(x_array.shape, dtype=np.int32)
    certificate_codes = np.zeros(x_array.shape, dtype=np.uint8)

    maximum_modes = 0
    maximum_steps = 0
    maximum_prefix = 0
    maximum_quantile_width = 0.0
    maximum_target_width = 0.0
    escalation_count = 0
    correctly_rounded_count = 0

    flat_x = x_array.reshape(-1)
    flat_u = u_array.reshape(-1)
    flat_active = active.reshape(-1)
    for index, (x_value, u_value, is_active) in enumerate(
        zip(flat_x, flat_u, flat_active, strict=True)
    ):
        if not bool(is_active):
            continue
        local_key = key
        local_index = index
        if explicit_ids is not None:
            local_key = key + b"\0transition-id:" + int(explicit_ids[index]).to_bytes(8, "big")
            local_index = 0
        prefix = _LazyDyadicPrefix(
            local_key,
            local_index,
            initial_bits=int(profile.initial_prefix_bits),
            max_bits=int(profile.max_prefix_bits),
        )
        try:
            (
                y_value,
                lower,
                upper,
                steps,
                modes,
                escalations,
                correctly_rounded,
            ) = _invert_one(float(x_value), float(u_value), prefix, profile)
            target_value, target_i, target_modes, target_escalated = _target_interval(
                float(x_value), float(y_value), float(u_value), profile
            )
        except JacobiRBCertificationError as exc:
            diagnostics = {
                "sample_index": index,
                "head_fraction": float(x_value),
                "exposure": float(u_value),
                "profile": profile.to_dict(),
                **exc.diagnostics,
            }
            raise JacobiRBCertificationError(str(exc), diagnostics) from exc
        later.reshape(-1)[index] = y_value
        target.reshape(-1)[index] = target_value
        q_lower.reshape(-1)[index] = lower
        q_upper.reshape(-1)[index] = upper
        z_lower.reshape(-1)[index] = target_i.lower
        z_upper.reshape(-1)[index] = target_i.upper
        prefix_bits.reshape(-1)[index] = prefix.bits
        certificate_codes.reshape(-1)[index] = np.uint8(
            1 | (2 if escalations else 0) | (4 if target_escalated else 0) | 8
        )
        maximum_modes = max(maximum_modes, modes, target_modes)
        maximum_steps = max(maximum_steps, steps)
        maximum_prefix = max(maximum_prefix, prefix.bits)
        maximum_quantile_width = max(maximum_quantile_width, upper - lower)
        maximum_target_width = max(maximum_target_width, target_i.width)
        escalation_count += int(escalations) + int(target_escalated)
        correctly_rounded_count += int(correctly_rounded)

    diagnostics = JacobiRBSpectralDiagnostics(
        certified=True,
        sample_count=int(x_array.size),
        active_count=int(np.count_nonzero(active)),
        zero_duration_count=int(x_array.size - np.count_nonzero(active)),
        interval_escalation_count=int(escalation_count),
        correctly_rounded_count=int(correctly_rounded_count),
        maximum_modes_used=int(maximum_modes),
        maximum_bisection_steps=int(maximum_steps),
        maximum_prefix_bits=int(maximum_prefix),
        maximum_quantile_bracket_width=float(maximum_quantile_width),
        maximum_target_interval_width=float(maximum_target_width),
    )
    return JacobiRBTransitionBatch(
        earlier_head_fraction=x_array.copy(),
        later_head_fraction=later,
        denoising_target=target,
        exposure=u_array.copy(),
        active_mask=active,
        quantile_lower=q_lower,
        quantile_upper=q_upper,
        target_lower=z_lower,
        target_upper=z_upper,
        prefix_bits=prefix_bits,
        certificate_codes=certificate_codes,
        diagnostics=diagnostics,
    )


def sample_alpha1_rb_transition_batch_torch(
    head_fraction: Tensor,
    exposure: Tensor,
    *,
    rng_key: Any,
    profile: JacobiRBSpectralProfile,
) -> JacobiRBTransitionBatch:
    """Certified batched API: CUDA intervals with transition-local Arb fallback.

    The returned cache payload is host NumPy because atomic cache shards are
    host artifacts.  Device interval decisions remain visible in compact
    certificate codes: code 9 is device-only certification and code 15 is an
    Arb-resolved transition.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("Torch is unavailable")
    x_tensor, u_tensor = torch.broadcast_tensors(head_fraction, exposure)
    x_array = x_tensor.detach().cpu().numpy().astype(np.float64, copy=True)
    u_array = u_tensor.detach().cpu().numpy().astype(np.float64, copy=True)
    if profile.authorize_device_intervals:
        proposal = propose_alpha1_rb_transition_batch_torch_intervals(
            x_tensor, u_tensor, rng_key=rng_key, profile=profile
        )
        later = proposal.later_head_fraction.detach().cpu().numpy().astype(
            np.float64, copy=True
        )
        target = proposal.denoising_target.detach().cpu().numpy().astype(
            np.float64, copy=True
        )
        q_lower = proposal.quantile_lower.detach().cpu().numpy().astype(
            np.float64, copy=True
        )
        q_upper = proposal.quantile_upper.detach().cpu().numpy().astype(
            np.float64, copy=True
        )
        z_lower = proposal.target_lower.detach().cpu().numpy().astype(
            np.float64, copy=True
        )
        z_upper = proposal.target_upper.detach().cpu().numpy().astype(
            np.float64, copy=True
        )
        active = proposal.active_mask.detach().cpu().numpy().astype(bool, copy=True)
        fallback = proposal.fallback_mask.detach().cpu().numpy().astype(bool, copy=True)
        codes = proposal.certificate_codes.detach().cpu().numpy().astype(
            np.uint8, copy=True
        )
        maximum_modes = int(proposal.modes_used)
        maximum_steps = int(proposal.bisection_steps)
    else:
        # The frozen production profile does not authorize eager CUDA
        # intervals.  Running their full bisection before inevitably falling
        # back would add cost without evidence, so active entries enter the
        # transition-local Arb path directly.  Zero-duration entries retain
        # exact no-op state/target/code semantics and consume no random bits.
        if (
            not np.all(np.isfinite(x_array))
            or np.any(x_array < 0.0)
            or np.any(x_array > 1.0)
        ):
            raise ValueError("head_fraction must be finite and lie in [0,1]")
        if not np.all(np.isfinite(u_array)) or np.any(u_array < 0.0):
            raise ValueError("exposure must be finite and nonnegative")
        active = u_array > 0.0
        fallback = active.copy()
        later = x_array.copy()
        target = np.zeros_like(x_array)
        q_lower = x_array.copy()
        q_upper = x_array.copy()
        z_lower = np.zeros_like(x_array)
        z_upper = np.zeros_like(x_array)
        codes = np.zeros_like(x_array, dtype=np.uint8)
        maximum_modes = 0
        maximum_steps = 0
    prefix_bits = np.zeros_like(x_array, dtype=np.int32)
    key = _key_bytes(rng_key)
    maximum_prefix = 0
    if np.any(fallback) and not profile.allow_interval_escalation:
        raise JacobiRBCertificationError(
            "device interval proposal requires Arb fallback",
            {
                "fallback_count": int(np.count_nonzero(fallback)),
                "failure_kind": "interval_escalation_disabled",
            },
        )
    for flat_index in np.flatnonzero(fallback.reshape(-1)):
        prefix = _LazyDyadicPrefix(
            key,
            int(flat_index),
            initial_bits=int(profile.initial_prefix_bits),
            max_bits=int(profile.max_prefix_bits),
        )
        try:
            (
                y_value,
                lower,
                upper,
                steps,
                modes,
                _escalations,
                correctly_rounded,
            ) = _invert_one(
                float(x_array.reshape(-1)[flat_index]),
                float(u_array.reshape(-1)[flat_index]),
                prefix,
                profile,
            )
            target_value, target_interval, target_modes, _ = _target_interval(
                float(x_array.reshape(-1)[flat_index]),
                float(y_value),
                float(u_array.reshape(-1)[flat_index]),
                profile,
            )
        except JacobiRBCertificationError as exc:
            raise JacobiRBCertificationError(
                str(exc),
                {
                    "sample_index": int(flat_index),
                    "device_interval_fallback": 1,
                    **exc.diagnostics,
                },
            ) from exc
        if not correctly_rounded:
            raise JacobiRBCertificationError(
                "Arb fallback did not correctly round its quantile",
                {"sample_index": int(flat_index), "failure_kind": "rounding_cell"},
            )
        later.reshape(-1)[flat_index] = y_value
        target.reshape(-1)[flat_index] = target_value
        q_lower.reshape(-1)[flat_index] = lower
        q_upper.reshape(-1)[flat_index] = upper
        z_lower.reshape(-1)[flat_index] = target_interval.lower
        z_upper.reshape(-1)[flat_index] = target_interval.upper
        prefix_bits.reshape(-1)[flat_index] = prefix.bits
        codes.reshape(-1)[flat_index] = np.uint8(15)
        maximum_modes = max(maximum_modes, modes, target_modes)
        maximum_steps = max(maximum_steps, steps)
        maximum_prefix = max(maximum_prefix, prefix.bits)
    # Device-certified active entries consumed the initial fixed prefix.
    prefix_bits[(active) & (~fallback)] = int(profile.initial_prefix_bits)
    maximum_prefix = max(
        maximum_prefix,
        int(profile.initial_prefix_bits) if np.any(active & ~fallback) else 0,
    )
    quantile_width = q_upper - q_lower
    target_width = z_upper - z_lower
    diagnostics = JacobiRBSpectralDiagnostics(
        certified=True,
        sample_count=int(x_array.size),
        active_count=int(np.count_nonzero(active)),
        zero_duration_count=int(x_array.size - np.count_nonzero(active)),
        interval_escalation_count=int(np.count_nonzero(fallback)),
        correctly_rounded_count=int(np.count_nonzero(active)),
        maximum_modes_used=int(maximum_modes),
        maximum_bisection_steps=int(maximum_steps),
        maximum_prefix_bits=int(maximum_prefix),
        maximum_quantile_bracket_width=(
            float(np.max(quantile_width)) if quantile_width.size else 0.0
        ),
        maximum_target_interval_width=(
            float(np.max(target_width)) if target_width.size else 0.0
        ),
    )
    return JacobiRBTransitionBatch(
        earlier_head_fraction=x_array,
        later_head_fraction=later,
        denoising_target=target,
        exposure=u_array,
        active_mask=active,
        quantile_lower=q_lower,
        quantile_upper=q_upper,
        target_lower=z_lower,
        target_upper=z_upper,
        prefix_bits=prefix_bits,
        certificate_codes=codes,
        diagnostics=diagnostics,
    )


def resolve_alpha1_pair_phase_inputs(
    tail_mass: np.ndarray | float,
    head_mass: np.ndarray | float,
    integrated_schedule_time: np.ndarray | float,
    *,
    grid_spacing: float,
) -> JacobiPairPhaseInputs:
    """Resolve pair fractions/exposures with exact zero-pair/no-op semantics.

    A zero pair has no mass to move.  Its conventional fraction and exposure
    are both zero and ``active_mask`` is false, even at positive schedule time.
    A zero-duration phase is likewise inactive and never evaluates ``1/r``.
    """

    tail, head, schedule = np.broadcast_arrays(
        np.asarray(tail_mass, dtype=np.float64),
        np.asarray(head_mass, dtype=np.float64),
        np.asarray(integrated_schedule_time, dtype=np.float64),
    )
    if not np.all(np.isfinite(tail)) or not np.all(np.isfinite(head)):
        raise ValueError("pair masses must be finite")
    if np.any(tail < 0.0) or np.any(head < 0.0):
        raise ValueError("pair masses must be nonnegative")
    if not np.all(np.isfinite(schedule)) or np.any(schedule < 0.0):
        raise ValueError("integrated_schedule_time must be finite and nonnegative")
    spacing = float(grid_spacing)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("grid_spacing must be finite and positive")
    total = tail + head
    active = (total > 0.0) & (schedule > 0.0)
    fraction = np.zeros_like(total)
    np.divide(head, total, out=fraction, where=total > 0.0)
    exposure = np.zeros_like(total)
    # alpha=1 gives (2 alpha + 1)/(alpha h^2 r) = 3/(h^2 r).
    np.divide(3.0 * schedule, spacing * spacing * total, out=exposure, where=active)
    return JacobiPairPhaseInputs(
        pair_total=total,
        head_fraction=fraction,
        exposure=exposure,
        active_mask=active,
    )


def reconstruct_pair_masses(
    pair_total: np.ndarray | float,
    later_head_fraction: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return tail/head masses without floors, limiters, or renormalization."""

    total, fraction = np.broadcast_arrays(
        np.asarray(pair_total, dtype=np.float64),
        np.asarray(later_head_fraction, dtype=np.float64),
    )
    if not np.all(np.isfinite(total)) or np.any(total < 0.0):
        raise ValueError("pair_total must be finite and nonnegative")
    if not np.all(np.isfinite(fraction)) or np.any(fraction < 0.0) or np.any(
        fraction > 1.0
    ):
        raise ValueError("later_head_fraction must be finite and lie in [0,1]")
    head = total * fraction
    tail = total - head
    return tail, head


def profile_fingerprint_payload(profile: JacobiRBSpectralProfile) -> dict[str, Any]:
    """Return the complete versioned profile payload for run fingerprints."""

    return profile.to_dict()


def certified_backend_report() -> dict[str, Any]:
    """Report the only backend allowed to authorize production evidence."""

    version = None
    if _arb is not None:
        try:
            import flint  # imported lazily so missing-backend reports are clean

            version = str(flint.__version__)
        except (ImportError, AttributeError):  # pragma: no cover - defensive
            version = None
    exact = version == "0.9.0"
    return {
        "backend": "python-flint/Arb" if _arb is not None else "unavailable",
        "python_flint_version": version,
        "required_python_flint_version": "0.9.0",
        "available": int(_arb is not None and _flint_ctx is not None),
        "exact_version_match": int(exact),
        "production_authorizing": int(exact and _flint_ctx is not None),
    }


__all__ = [
    "JACOBI_RB_ORIENTATION",
    "JACOBI_RB_RNG_VERSION",
    "JACOBI_RB_SPECTRAL_VERSION",
    "JacobiPairPhaseInputs",
    "CertifiedRBTransitionBatch",
    "JacobiRBDeviceIntervalProposal",
    "JacobiRBTorchIntervalEvaluation",
    "JacobiRBCertificationError",
    "JacobiRBSpectralDiagnostics",
    "JacobiRBSpectralProfile",
    "JacobiRBTransitionBatch",
    "cantelli_quantile_bracket",
    "certified_backend_report",
    "evaluate_alpha1_rb_torch_fixed_modes",
    "evaluate_alpha1_rb_torch_intervals",
    "philox_uniform_prefix",
    "profile_fingerprint_payload",
    "propose_alpha1_rb_transition_batch_torch",
    "propose_alpha1_rb_transition_batch_torch_intervals",
    "reconstruct_pair_masses",
    "resolve_alpha1_pair_phase_inputs",
    "sample_alpha1_rb_transition_batch",
    "sample_alpha1_rb_transition_batch_torch",
]
