r"""Fused CUDA/Arb backend for certified alpha-one Jacobi transitions.

The proposal kernel remains deliberately non-authorizing.  A second fused
kernel proves exact inverse-CDF and target rounding cells with DD balls,
directed radii, a certified degree-24 exponential, and omitted-mode tails.
Only unresolved device lanes reach candidate-local Arb certification.

The kernel is header-free CUDA C and is compiled by the NVRTC loader bundled
with PyTorch.  No CUDA toolkit, compiler subprocess, or writable build tree is
needed.  Compilation and authorizing-backend failures are fail-closed.
"""

from __future__ import annotations

import atexit
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import functools
import hashlib
import json
import math
import multiprocessing
import os
import threading
import time
from typing import Any, Mapping

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - exercised in stripped environments.
    torch = None
    Tensor = Any

from mnist import d0_jacobi_rb_spectral as _reference
from mnist.d0_jacobi_rb_cuda_certificate import (
    FallbackReason as _FallbackReason,
    run_certificate_arithmetic_preflight,
)
from mnist.d0_jacobi_rb_cuda_fused import (
    FUSED_CUDA_VERSION,
    launch_fused_cuda_authorizer_with_neighbors,
    probe_fused_cuda_authorizer,
)
from mnist.d0_jacobi_rb_nvrtc_compat import compile_cuda_kernels


_CUDA_VERSION = "alpha1-jacobi-rb-nvrtc-candidate-arb-certificate-v2-multiruntime"
_RNG_VERSION = "philox4x32-10-canonical-transition-v2"
_KERNEL_NAME = "jacobi_rb_candidate_v1"
_FROZEN_TORCH_VERSION = "2.11.0+cu128"
_FROZEN_CUDA_VERSION = "12.8"
_FROZEN_COMPUTE_CAPABILITY = "12.0"
_H100_TORCH_VERSION = "2.8.0+cu128"
_H100_CUDA_VERSION = "12.8"
_H100_COMPUTE_CAPABILITY = "9.0"
_COMPILE_FLAGS = (
    "--std=c++17",
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
)
_U32_MASK = (1 << 32) - 1
_U64_MASK = (1 << 64) - 1
_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85
_PHILOX_NAMESPACE = 0x4A524232  # ASCII "JRB2" in the fourth counter word.
_ARB_CANDIDATE_LATTICE_ULPS = 256


@dataclass(frozen=True)
class JacobiRBCudaProfile:
    """Frozen numerical and launch contract for the CUDA candidate backend."""

    schema_version: int = 1
    candidate_modes: int = 128
    candidate_bisection_steps: int = 56
    threads_per_block: int = 128
    initial_prefix_bits: int = 64
    prefix_block_bits: int = 64
    max_prefix_bits: int = 1024
    require_correct_rounding: bool = True
    allow_certified_cpu_fallback: bool = True
    certificate_effort: str = "adaptive"
    frozen_torch_version: str = _FROZEN_TORCH_VERSION
    frozen_cuda_version: str = _FROZEN_CUDA_VERSION
    frozen_compute_capability: str = _FROZEN_COMPUTE_CAPABILITY
    compile_flags: tuple[str, ...] = _COMPILE_FLAGS

    def __post_init__(self) -> None:
        if int(self.schema_version) not in {1, 2}:
            raise ValueError("unsupported Jacobi RB CUDA profile schema_version")
        if not 2 <= int(self.candidate_modes) <= 4096:
            raise ValueError("candidate_modes must lie in [2,4096]")
        if not 1 <= int(self.candidate_bisection_steps) <= 128:
            raise ValueError("candidate_bisection_steps must lie in [1,128]")
        if int(self.threads_per_block) not in {32, 64, 128, 256, 512}:
            raise ValueError("threads_per_block must be a supported whole warp count")
        if int(self.initial_prefix_bits) != 64 or int(self.prefix_block_bits) != 64:
            raise ValueError("version-1 stateless Philox uses 64-bit prefix blocks")
        if not 64 <= int(self.max_prefix_bits) <= 1024:
            raise ValueError("max_prefix_bits must lie in [64,1024]")
        if not bool(self.require_correct_rounding):
            raise ValueError("the CUDA API cannot disable correct rounding")
        if not bool(self.allow_certified_cpu_fallback):
            raise ValueError("unresolved certificate lanes require the Arb fallback")
        if self.certificate_effort not in {"adaptive", "strengthened"}:
            raise ValueError("certificate_effort must be adaptive or strengthened")
        if int(self.schema_version) == 1:
            expected_runtime = (
                _FROZEN_TORCH_VERSION,
                _FROZEN_CUDA_VERSION,
                _FROZEN_COMPUTE_CAPABILITY,
            )
        else:
            expected_runtime = (
                _H100_TORCH_VERSION,
                _H100_CUDA_VERSION,
                _H100_COMPUTE_CAPABILITY,
            )
        observed_runtime = (
            str(self.frozen_torch_version),
            str(self.frozen_cuda_version),
            str(self.frozen_compute_capability),
        )
        if observed_runtime != expected_runtime:
            raise ValueError(
                "frozen Torch/CUDA/device runtime does not match the selected "
                f"Jacobi RB profile schema {self.schema_version}: "
                f"expected={expected_runtime}, observed={observed_runtime}"
            )
        if tuple(self.compile_flags) != _COMPILE_FLAGS:
            raise ValueError("compile_flags are immutable in Jacobi RB CUDA profiles")

    @classmethod
    def h100_pytorch28(cls, **overrides: Any) -> "JacobiRBCudaProfile":
        """Return the explicitly frozen H100 / PyTorch 2.8 RunPod profile."""

        values: dict[str, Any] = {
            "schema_version": 2,
            "frozen_torch_version": _H100_TORCH_VERSION,
            "frozen_cuda_version": _H100_CUDA_VERSION,
            "frozen_compute_capability": _H100_COMPUTE_CAPABILITY,
        }
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            backend_version=_CUDA_VERSION,
            rng_version=_RNG_VERSION,
            cuda_candidate_authorizing=False,
            fused_cuda_authorizer_implemented=True,
            fused_cuda_authorizer_available=None,
            fused_cuda_authorizer_requires_runtime_selftest=True,
            double_double_interval_certified=True,
            certified_device_exponential=True,
            arb_target_stop_rule="adaptive-g-over-k-rounding-cell-margin",
            arb_candidate_lattice_ulps=_ARB_CANDIDATE_LATTICE_ULPS,
        )
        return result

    @property
    def double_double_interval_certified(self) -> bool:
        return True

    @property
    def certified_device_exponential(self) -> bool:
        return True

    def strengthened(self) -> "JacobiRBCudaProfile":
        """Fingerprint-visible proposal strengthening; proof caps stay frozen.

        The fused certificate always owns its 4096/8192 primary/strengthened
        caps.  This method doubles only the non-authorizing proposal work and
        is used by precision-doubling controls without changing the law.
        """

        return replace(
            self,
            candidate_modes=min(1024, 2 * int(self.candidate_modes)),
            certificate_effort="strengthened",
        )


@dataclass(frozen=True)
class CertifiedRBCudaBatch:
    """Certified transition payload whose tensors remain on the input device."""

    earlier_head_fraction: Tensor
    later_head_fraction: Tensor
    denoising_target: Tensor
    exposure: Tensor
    transition_ids: Tensor
    active_mask: Tensor
    certified_mask: Tensor
    candidate_later_head_fraction: Tensor
    candidate_denoising_target: Tensor
    candidate_match_mask: Tensor
    cuda_certified_mask: Tensor
    fallback_mask: Tensor
    strengthened_mask: Tensor
    arb_fallback_reason_codes: Tensor
    arb_fallback_mode_counts: Tensor
    mode_counts: Tensor
    quantile_lower: Tensor
    quantile_upper: Tensor
    target_lower: Tensor
    target_upper: Tensor
    prefix_bits: Tensor
    certificate_codes: Tensor
    diagnostics: Mapping[str, Tensor]
    runtime_report: Mapping[str, Any]


def _canonical_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return b"bytes:" + value
    if isinstance(value, str):
        return b"str:" + value.encode("utf-8")
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("identifier material must be bytes, text, or canonical JSON") from exc
    return b"json:" + encoded.encode("utf-8")


def _canonical_transition_id(value: Any) -> int:
    """Return one stable unsigned-64 transition namespace identifier.

    Unsigned integers are preserved.  Structured identifiers are hashed with
    an explicit domain separator, making coordinates independent of batching.
    """

    if isinstance(value, int) and not isinstance(value, bool):
        if not 0 <= value <= _U64_MASK:
            raise ValueError("integer transition IDs must lie in uint64 range")
        return int(value)
    digest = hashlib.sha256(
        b"d0-jacobi-rb-canonical-transition-v1\0" + _canonical_bytes(value)
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _canonical_seed(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        if not 0 <= value <= _U64_MASK:
            raise ValueError("rng_seed must lie in uint64 range")
    digest = hashlib.sha256(
        b"d0-jacobi-rb-philox-key-v2\0" + _canonical_bytes(value)
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def canonical_v2_transition_id(
    path: Any, outer_step: int, phase: int, edge: Any
) -> int:
    """Canonical path/step/phase/edge identity for the v2 Philox stream."""

    if int(outer_step) < 0 or int(phase) < 0:
        raise ValueError("outer_step and phase must be nonnegative")
    return _canonical_transition_id(
        ("alpha1-jacobi-rb-v2", path, int(outer_step), int(phase), edge)
    )


def _mulhilo32(left: int, right: int) -> tuple[int, int]:
    product = (int(left) & _U32_MASK) * (int(right) & _U32_MASK)
    return (product >> 32) & _U32_MASK, product & _U32_MASK


def _philox4x32_10(
    counter: tuple[int, int, int, int], key: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Pure-Python Random123 Philox4x32-10 reference implementation."""

    c0, c1, c2, c3 = (int(value) & _U32_MASK for value in counter)
    k0, k1 = (int(value) & _U32_MASK for value in key)
    for _ in range(10):
        hi0, lo0 = _mulhilo32(_PHILOX_M0, c0)
        hi1, lo1 = _mulhilo32(_PHILOX_M1, c2)
        c0, c1, c2, c3 = (
            (hi1 ^ c1 ^ k0) & _U32_MASK,
            lo1,
            (hi0 ^ c3 ^ k1) & _U32_MASK,
            lo0,
        )
        k0 = (k0 + _PHILOX_W0) & _U32_MASK
        k1 = (k1 + _PHILOX_W1) & _U32_MASK
    return c0, c1, c2, c3


def _philox_u64_from_canonical_seed(
    seed: int, transition_id: int, block_index: int = 0
) -> int:
    transition = _canonical_transition_id(transition_id)
    if not 0 <= int(block_index) <= _U32_MASK:
        raise ValueError("v2 refinement block must lie in uint32 range")
    words = _philox4x32_10(
        (
            transition & _U32_MASK,
            transition >> 32,
            int(block_index) & _U32_MASK,
            _PHILOX_NAMESPACE,
        ),
        (seed & _U32_MASK, seed >> 32),
    )
    return ((words[0] << 32) | words[1]) & _U64_MASK


def _philox_u64(rng_seed: Any, transition_id: int, block_index: int = 0) -> int:
    return _philox_u64_from_canonical_seed(
        _canonical_seed(rng_seed), transition_id, block_index
    )


def _philox_uniform_midpoint(rng_seed: int, transition_id: int) -> float:
    """Return the binary64 midpoint of the first 53 Philox prefix bits."""

    top53 = _philox_u64(rng_seed, transition_id) >> 11
    return math.ldexp(float(top53) + 0.5, -53)


def _strict_rounding_cell_contains(lower: float, upper: float, candidate: float) -> bool:
    """Test strict containment in one exact binary64 round-to-nearest cell."""

    lo, hi, value = float(lower), float(upper), float(candidate)
    if not (math.isfinite(lo) and math.isfinite(hi) and math.isfinite(value)):
        return False
    if lo > hi:
        return False
    cell_lo, cell_hi = _reference._rounding_cell(value)
    return (
        Fraction.from_float(lo) > cell_lo
        and Fraction.from_float(hi) < cell_hi
    )


class _StatelessPhiloxPrefix:
    """Lazy prefix compatible with the reference certifier, without RNG state."""

    def __init__(
        self, seed: Any, transition_id: int, max_bits: int, *, seed_is_canonical: bool = False
    ) -> None:
        self.seed = int(seed) if seed_is_canonical else _canonical_seed(seed)
        self.transition_id = _canonical_transition_id(transition_id)
        self.max_bits = int(max_bits)
        self.numerator = 0
        self.bits = 0
        self._next_block = 0
        self.refine(64)

    def refine(self, count: int) -> None:
        remaining = int(count)
        if remaining < 1 or self.bits + remaining > self.max_bits:
            raise _reference.JacobiRBCertificationError(
                "stateless Philox prefix cap reached",
                {
                    "transition_id": self.transition_id,
                    "prefix_bits": self.bits,
                    "max_prefix_bits": self.max_bits,
                    "failure_kind": "random_bit_cap",
                },
            )
        while remaining:
            word = _philox_u64_from_canonical_seed(
                self.seed, self.transition_id, self._next_block
            )
            self._next_block += 1
            take = min(64, remaining)
            self.numerator = (self.numerator << take) | (word >> (64 - take))
            self.bits += take
            remaining -= take

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


class _FixedDyadicPrefix:
    """A recorded v1 dyadic prefix; exhaustion fails instead of inventing bits."""

    def __init__(self, numerator: int, bits: int) -> None:
        if not 1 <= int(bits) <= 64:
            raise ValueError("recorded prefix_bits must lie in [1,64]")
        if not 0 <= int(numerator) < (1 << int(bits)):
            raise ValueError("recorded prefix_numerator does not fit prefix_bits")
        self.numerator = int(numerator)
        self.bits = int(bits)
        self.max_bits = int(bits)

    def refine(self, count: int) -> None:
        raise _reference.JacobiRBCertificationError(
            "recorded v1 dyadic prefix was exhausted",
            {
                "prefix_bits": self.bits,
                "requested_additional_bits": int(count),
                "failure_kind": "recorded_prefix_exhausted",
            },
        )

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


@functools.lru_cache(maxsize=1)
def _parent_v1_prefix_candidate_map() -> Mapping[int, int]:
    """Map the immutable 4096-key parent panel's first words to its keys."""

    result: dict[int, int] = {}
    for candidate in range(4096):
        numerator, bits, _midpoint = _reference.philox_uniform_prefix(
            (261_121, "support-prefix", candidate), bits=64
        )
        if int(bits) != 64 or int(numerator) in result:
            raise RuntimeError("parent-v1 support-prefix map is not injective")
        result[int(numerator)] = int(candidate)
    return result


_CUDA_SOURCE = r"""
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ u32 mul_hi(u32 a, u32 b) {
    return (u32)(((u64)a * (u64)b) >> 32);
}

__device__ __forceinline__ u64 philox_word(u64 seed, u64 transition) {
    u32 c0=(u32)transition, c1=(u32)(transition>>32), c2=0U, c3=0x4A524232U;
    u32 k0=(u32)seed, k1=(u32)(seed>>32);
    #pragma unroll
    for (int round=0; round<10; ++round) {
        u32 hi0=mul_hi(0xD2511F53U,c0), lo0=0xD2511F53U*c0;
        u32 hi1=mul_hi(0xCD9E8D57U,c2), lo1=0xCD9E8D57U*c2;
        u32 n0=hi1^c1^k0, n1=lo1, n2=hi0^c3^k1, n3=lo0;
        c0=n0; c1=n1; c2=n2; c3=n3;
        k0+=0x9E3779B9U; k1+=0xBB67AE85U;
    }
    return ((u64)c0<<32)|(u64)c1;
}

__device__ __forceinline__ double legendre_cdf(
    double x, double y, double exposure, int modes
) {
    double zx=2.0*x-1.0, zy=2.0*y-1.0;
    double px0=1.0, px1=zx, py0=1.0, py1=zy;
    double result=y, correction=0.0;
    for (int degree=1; degree<modes; ++degree) {
        double py2=((2.0*degree+1.0)*zy*py1-degree*py0)/(degree+1.0);
        double decay=exp(-(double)(degree*(degree+1))*exposure);
        double term=0.5*decay*px1*(py2-py0);
        // Neumaier compensation makes the non-authorizing proposal accurate
        // enough that the DD cell certifier normally needs no neighbour.
        double updated=result+term;
        if ((result<0.0?-result:result)>=(term<0.0?-term:term))
            correction+=(result-updated)+term;
        else correction+=(term-updated)+result;
        result=updated;
        double px2=((2.0*degree+1.0)*zx*px1-degree*px0)/(degree+1.0);
        px0=px1; px1=px2; py0=py1; py1=py2;
    }
    return result+correction;
}

__device__ __forceinline__ int adaptive_candidate_modes(double exposure, int minimum) {
    int modes=minimum<128?128:minimum;
    if (modes>1024) modes=1024;
    while (modes<1024) {
        double decay=exp(-(double)(modes*(modes+1))*exposure);
        double ratio=exp(-2.0*(double)(modes+1)*exposure);
        double tail=ratio<1.0?decay/(1.0-ratio):1.0;
        if (tail<0x1.0p-62) break;
        modes*=2;
        if (modes>1024) modes=1024;
    }
    return modes;
}

__device__ __forceinline__ double pow2_candidate(int exponent) {
    return __longlong_as_double((u64)(exponent+1023)<<52);
}
__device__ __forceinline__ double dyadic_midpoint_candidate(u64 numerator, int bits) {
    if (bits<=32)
        return ((double)(u32)numerator+0.5)*pow2_candidate(-bits);
    return (double)(u32)(numerator>>32)*pow2_candidate(32-bits)+
        ((double)(u32)numerator+0.5)*pow2_candidate(-bits);
}
__device__ __forceinline__ void cantelli_candidate_bracket(
    double x, double exposure, double probability, double* lower, double* upper
) {
    double z=2.0*x-1.0;
    double first=exp(-2.0*exposure)*z;
    double p2=0.5*(3.0*z*z-1.0);
    double second=(1.0+2.0*exp(-6.0*exposure)*p2)/3.0;
    double mean=0.5*(1.0+first);
    double variance=0.25*(second-first*first);
    if (variance<0.0) variance=0.0;
    double lr=sqrt(variance*(1.0-probability)/probability);
    double ur=sqrt(variance*probability/(1.0-probability));
    // This bracket only seeds a non-authorizing proposal.  The padding is
    // intentionally much wider than accumulated binary64 moment roundoff.
    double padding=0x1.0p-46;
    double lo=mean-lr-padding, hi=mean+ur+padding;
    *lower=lo>0.0?lo:0.0; *upper=hi<1.0?hi:1.0;
    if (!(*lower<*upper)) { *lower=0.0; *upper=1.0; }
}

__device__ __forceinline__ double rb_target(
    double x, double y, double exposure, int modes
) {
    double zx=2.0*x-1.0, zy=2.0*y-1.0;
    double px0=1.0, px1=zx, py0=1.0, py1=zy;
    double density=1.0, conormal=0.0;
    for (int degree=1; degree<modes; ++degree) {
        double py2=((2.0*degree+1.0)*zy*py1-degree*py0)/(degree+1.0);
        double coefficient=(2.0*degree+1.0)*
            exp(-(double)(degree*(degree+1))*exposure);
        density += coefficient*px1*py1;
        conormal += coefficient*px1*(0.5*degree)*(py0-zy*py1);
        double px2=((2.0*degree+1.0)*zx*px1-degree*px0)/(degree+1.0);
        px0=px1; px1=px2; py0=py1; py1=py2;
    }
    return conormal/density;
}

extern "C" __global__ void jacobi_rb_candidate_v1(
    const double* x, const double* exposure, const u64* transition_ids,
    const u64* seed_pointer, const u64* recorded_prefix,
    const int* recorded_bits, int prefix_kind, int count, int modes, int steps,
    double* later, double* target, double* lower, double* upper
) {
    int index=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    if (index>=count) return;
    double xv=x[index], uv=exposure[index];
    if (uv==0.0) {
        later[index]=xv; target[index]=0.0; lower[index]=xv; upper[index]=xv;
        return;
    }
    u64 random=prefix_kind?recorded_prefix[index]:
        philox_word(seed_pointer[0],transition_ids[index]);
    // Rounded image of the complete 64-bit prefix midpoint, rather than the
    // midpoint of only its leading 53 bits.
    int random_bits=prefix_kind?recorded_bits[index]:64;
    double uniform=dyadic_midpoint_candidate(random,random_bits);
    int adaptive_modes=adaptive_candidate_modes(uv,modes);
    double lo,hi;
    cantelli_candidate_bracket(xv,uv,uniform,&lo,&hi);
    for (int step=0; step<steps; ++step) {
        double midpoint=lo+0.5*(hi-lo);
        if (legendre_cdf(xv,midpoint,uv,adaptive_modes)>uniform) hi=midpoint;
        else lo=midpoint;
    }
    double y=lo+0.5*(hi-lo);
    later[index]=y; target[index]=rb_target(xv,y,uv,adaptive_modes);
    lower[index]=lo; upper[index]=hi;
}
"""

_KERNEL_SHA256 = hashlib.sha256(_CUDA_SOURCE.encode("utf-8")).hexdigest()
_KERNEL_CACHE: dict[tuple[int, int, int, int], tuple[Any, str]] = {}
_KERNEL_LOCK = threading.Lock()


@functools.lru_cache(maxsize=1)
def _certificate_arithmetic_preflight() -> Mapping[str, Any]:
    return run_certificate_arithmetic_preflight()


def _cuda_driver_version() -> int | None:
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        import ctypes
        from torch.cuda._utils import _get_gpu_runtime_library

        value = ctypes.c_int()
        result = int(_get_gpu_runtime_library().cuDriverGetVersion(ctypes.byref(value)))
        return int(value.value) if result == 0 else None
    except Exception:
        return None


def _runtime_report(
    device: Any | None = None,
    *,
    profile: JacobiRBCudaProfile | None = None,
    probe_authorizer: bool = True,
) -> dict[str, Any]:
    selected = profile if profile is not None else JacobiRBCudaProfile()
    report: dict[str, Any] = {
        "backend_version": _CUDA_VERSION,
        "rng_version": _RNG_VERSION,
        "kernel_sha256": _KERNEL_SHA256,
        "loader": "torch.cuda._utils._nvrtc_compile/_cuda_load_module",
        "header_free": True,
        "cuda_candidate_authorizing": False,
        "fused_cuda_authorizer_available": False,
        "fused_cuda_authorizer_unavailable_reason": "runtime arithmetic self-test not run",
        "fused_cuda_version": FUSED_CUDA_VERSION,
        "frozen_torch_version": str(selected.frozen_torch_version),
        "frozen_cuda_version": str(selected.frozen_cuda_version),
        "frozen_compute_capability": str(selected.frozen_compute_capability),
        "runtime_profile_schema": int(selected.schema_version),
        "compile_flags": list(_COMPILE_FLAGS),
        "compile_options_sha256": hashlib.sha256(
            "\0".join(_COMPILE_FLAGS).encode("utf-8")
        ).hexdigest(),
        "authorizing_backend": _reference.certified_backend_report(),
        "torch_available": torch is not None,
        "cuda_driver_version": _cuda_driver_version(),
        "arb_target_stop_rule": "adaptive-g-over-k-rounding-cell-margin",
        "arb_candidate_lattice_ulps": _ARB_CANDIDATE_LATTICE_ULPS,
    }
    if torch is None:
        report.update(cuda_available=False, loader_available=False)
        return report
    report.update(
        torch_version=str(torch.__version__),
        torch_cuda_build=str(torch.version.cuda),
        cuda_available=bool(torch.cuda.is_available()),
        loader_available=callable(getattr(torch.cuda, "_compile_kernel", None)),
        frozen_runtime_match=(
            str(torch.__version__) == str(selected.frozen_torch_version)
            and str(torch.version.cuda) == str(selected.frozen_cuda_version)
        ),
    )
    if device is not None and bool(torch.cuda.is_available()):
        index = torch.device(device).index
        if index is None:
            index = int(torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(index)
        report.update(
            device_index=int(index),
            device_name=str(properties.name),
            compute_capability=f"{properties.major}.{properties.minor}",
            device_uuid=str(properties.uuid),
        )
        if probe_authorizer and report.get("frozen_runtime_match") and (
            report.get("compute_capability") == str(selected.frozen_compute_capability)
        ):
            _bundle, fused = probe_fused_cuda_authorizer(
                device,
                compile_flags=_COMPILE_FLAGS,
                cpu_preflight=dict(_certificate_arithmetic_preflight()),
            )
            report.update(fused)
            report["candidate_kernel_sha256"] = _KERNEL_SHA256
            report["kernel_sha256"] = fused.get("fused_source_sha256", _KERNEL_SHA256)
            report["source_sha256"] = report["kernel_sha256"]
            report["binary_sha256"] = fused.get("fused_binary_sha256")
            report["cubin_sha256"] = report["binary_sha256"]
            report["directed_rounding_intrinsics_pass"] = bool(
                fused.get("arithmetic_selftest_pass", False)
            )
            report["compile_flags_exact_pass"] = True
            report["runtime_contract_pass"] = True
            report["device_contract_pass"] = True
    return report


def _load_cuda_kernel(device: Any, profile: JacobiRBCudaProfile) -> tuple[Any, str]:
    report = _runtime_report(device, profile=profile, probe_authorizer=False)
    if not report.get("cuda_available") or not report.get("loader_available"):
        raise RuntimeError(f"Jacobi RB CUDA backend unavailable: {report}")
    if not report.get("frozen_runtime_match"):
        raise RuntimeError(
            "Jacobi RB CUDA backend runtime differs from the frozen "
            f"Torch/CUDA contract: {report}"
        )
    device_index = torch.device(device).index
    if device_index is None:
        device_index = int(torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device_index)
    actual_capability = f"{props.major}.{props.minor}"
    if actual_capability != profile.frozen_compute_capability:
        raise RuntimeError(
            "Jacobi RB CUDA backend device differs from the frozen compute "
            f"capability {profile.frozen_compute_capability}: {actual_capability}"
        )
    cache_key = (
        int(device_index),
        int(props.major),
        int(props.minor),
        int(profile.schema_version),
    )
    with _KERNEL_LOCK:
        cached = _KERNEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            loaded, binary_sha256, lowered_name = compile_cuda_kernels(
                _CUDA_SOURCE,
                primary_name=_KERNEL_NAME,
                kernel_names=(_KERNEL_NAME,),
                device_index=int(device_index),
                compute_capability=f"{props.major}{props.minor}",
                compile_flags=tuple(profile.compile_flags),
            )
            kernel = loaded[lowered_name]
        except Exception as exc:
            raise RuntimeError(
                "Jacobi RB CUDA NVRTC compilation failed closed; no candidate "
                "or certified active output was emitted"
            ) from exc
        result = (kernel, binary_sha256)
        _KERNEL_CACHE[cache_key] = result
        return result


def _require_cuda_inputs(head_fraction: Tensor, exposure: Tensor, transition_ids: Tensor) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is unavailable")
    values = (
        ("head_fraction", head_fraction, torch.float64),
        ("exposure", exposure, torch.float64),
        ("transition_ids", transition_ids, torch.uint64),
    )
    shape = None
    device = None
    for name, tensor, dtype in values:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if shape is None:
            shape, device = tensor.shape, tensor.device
        elif tensor.shape != shape or tensor.device != device:
            raise ValueError("all inputs must have identical shapes and CUDA devices")
    if head_fraction.numel() > 4096:
        raise ValueError("CUDA backend calls are capped at 4096 transition lanes")
    if not bool(torch.isfinite(head_fraction).all().item()) or bool(
        ((head_fraction < 0.0) | (head_fraction > 1.0)).any().item()
    ):
        raise ValueError("head_fraction must be finite and lie in [0,1]")
    if not bool(torch.isfinite(exposure).all().item()) or bool((exposure < 0.0).any().item()):
        raise ValueError("exposure must be finite and nonnegative")


def _reference_profile(profile: JacobiRBCudaProfile) -> Any:
    return _reference.JacobiRBSpectralProfile(
        initial_prefix_bits=int(profile.initial_prefix_bits),
        prefix_block_bits=int(profile.prefix_block_bits),
        max_prefix_bits=int(profile.max_prefix_bits),
        require_correct_rounding=True,
        allow_interval_escalation=True,
        authorize_device_intervals=False,
    )


def _device_count(value: int, device: Any) -> Tensor:
    return torch.tensor(int(value), dtype=torch.int64, device=device)


def _fraction_down(value: Fraction) -> float:
    rounded = float(value)
    if Fraction.from_float(rounded) > value:
        rounded = math.nextafter(rounded, -math.inf)
    return rounded


def _fraction_up(value: Fraction) -> float:
    rounded = float(value)
    if Fraction.from_float(rounded) < value:
        rounded = math.nextafter(rounded, math.inf)
    return rounded


def _arb_candidate_target_rounding(
    x: float,
    y: float,
    exposure: float,
    *,
    profile: Any,
    precision_bits: int,
) -> tuple[float | None, Any, int]:
    """Certify ``G/K`` against its actual binary64 rounding-cell margin.

    This candidate-local fallback deliberately does not inherit the reference
    backend's historical ``log(first omitted term) < -1200`` stop.  At every
    16-mode bucket it encloses both omitted tails, forms the Arb quotient, and
    stops exactly when that quotient is strictly inside one rounding cell.
    """

    if _reference._arb is None or _reference._flint_ctx is None:
        raise _reference.JacobiRBCertificationError(
            "candidate target rounding requires python-flint/Arb",
            {"failure_kind": "arb_backend_unavailable"},
        )
    bits = int(precision_bits)
    latest_interval = _reference._Interval(-math.inf, math.inf)
    with _reference._ARB_CONTEXT_LOCK:
        previous_precision = int(_reference._flint_ctx.prec)
        try:
            _reference._flint_ctx.prec = bits
            arb = _reference._arb
            one = arb(1)
            zx = 2 * _reference._arb_exact(x) - one
            zy = 2 * _reference._arb_exact(y) - one
            u_value = _reference._arb_exact(exposure)
            px_previous, px_current = one, zx
            py_previous, py_current = one, zy
            density = one
            conormal = arb(0)
            modes_used = 1
            for degree in range(1, int(profile.max_modes)):
                py_next = (
                    (2 * degree + 1) * zy * py_current
                    - degree * py_previous
                ) / (degree + 1)
                decay = (-degree * (degree + 1) * u_value).exp()
                coefficient = (2 * degree + 1) * decay
                density += coefficient * px_current * py_current
                basis = _reference._arb_exact(Fraction(degree, 2)) * (
                    py_previous - zy * py_current
                )
                conormal += coefficient * px_current * basis
                modes_used = degree + 1
                first_omitted = degree + 1

                check_bucket = (
                    first_omitted >= 16
                    and (
                        first_omitted % 16 == 0
                        or first_omitted + 1 == int(profile.max_modes)
                    )
                )
                if check_bucket:
                    tails = _reference._arb_geometric_tail_radii(
                        first_omitted, u_value
                    )
                    density_tail, conormal_tail = tails[1], tails[2]
                    if density_tail is not None and conormal_tail is not None:
                        enclosed_density = density + _reference._arb_error_ball(
                            density_tail
                        )
                        enclosed_conormal = conormal + _reference._arb_error_ball(
                            conormal_tail
                        )
                        if enclosed_density > arb(0):
                            target = enclosed_conormal / enclosed_density
                            latest_interval = _reference._arb_bounds_or_unbounded(
                                target
                            )
                            candidate = float(target.mid())
                            if math.isfinite(candidate):
                                lower_boundary, upper_boundary = (
                                    _reference._rounding_cell(candidate)
                                )
                                if (
                                    target > _reference._arb_exact(lower_boundary)
                                    and target < _reference._arb_exact(upper_boundary)
                                ):
                                    return candidate, latest_interval, modes_used

                px_next = (
                    (2 * degree + 1) * zx * px_current
                    - degree * px_previous
                ) / (degree + 1)
                px_previous, px_current = px_current, px_next
                py_previous, py_current = py_current, py_next
            return None, latest_interval, modes_used
        finally:
            _reference._flint_ctx.prec = previous_precision


def _arb_candidate_target_interval(
    x: float, y: float, exposure: float, profile: Any
) -> tuple[float, Any, int]:
    maximum_modes = 0
    latest_interval = _reference._Interval(-math.inf, math.inf)
    for precision in profile.arb_precision_bits:
        rounded, interval, modes = _arb_candidate_target_rounding(
            x,
            y,
            exposure,
            profile=profile,
            precision_bits=int(precision),
        )
        maximum_modes = max(maximum_modes, int(modes))
        latest_interval = interval
        if rounded is not None:
            return float(rounded), interval, maximum_modes
    raise _reference.JacobiRBCertificationError(
        "candidate-local Arb target did not fit a binary64 rounding cell",
        {
            "x": float(x),
            "y": float(y),
            "exposure": float(exposure),
            "target_interval": [
                float(latest_interval.lower), float(latest_interval.upper)
            ],
            "maximum_modes": int(maximum_modes),
            "stop_rule": "adaptive-g-over-k-rounding-cell-margin",
            "failure_kind": "arb_resource_cap",
        },
    )


def _arb_candidate_cell_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Candidate-local Arb certificate used only for unresolved CUDA lanes."""

    x_value = float(payload["x"])
    u_value = float(payload["exposure"])
    candidate = float(payload["candidate"])
    profile = payload["profile"]
    if payload["prefix_kind"] == "parent-v1-verified-continuation":
        key = (
            261_121,
            "support-prefix",
            int(payload["v1_key_candidate"]),
        )
        prefix = _reference._LazyDyadicPrefix(
            _reference._key_bytes(key),
            0,
            initial_bits=64,
            max_bits=int(payload["max_prefix_bits"]),
        )
        if int(prefix.numerator) != int(payload["prefix_numerator"]):
            raise _reference.JacobiRBCertificationError(
                "parent-v1 continuation does not match its recorded first word",
                {"failure_kind": "parent_v1_prefix_mismatch"},
            )
    elif payload["prefix_kind"] == "recorded":
        prefix: Any = _FixedDyadicPrefix(
            int(payload["prefix_numerator"]), int(payload["prefix_bits"])
        )
    else:
        prefix = _StatelessPhiloxPrefix(
            int(payload["seed"]), int(payload["transition_id"]),
            int(payload["max_prefix_bits"]), seed_is_canonical=True,
        )
    candidates = [candidate]
    lower, upper = candidate, candidate
    for _ in range(_ARB_CANDIDATE_LATTICE_ULPS):
        lower = math.nextafter(lower, -math.inf)
        upper = math.nextafter(upper, math.inf)
        candidates.extend((lower, upper))
    maximum_modes = 0

    def decide(boundary: Fraction) -> tuple[int, int]:
        nonlocal maximum_modes
        while True:
            for precision in profile.arb_precision_bits:
                decision, used, _interval = _reference._arb_cdf_prefix_decision(
                    prefix,
                    x_value,
                    boundary,
                    u_value,
                    profile=profile,
                    precision_bits=int(precision),
                )
                maximum_modes = max(maximum_modes, int(used))
                if decision:
                    return int(decision), int(used)
            if isinstance(prefix, _FixedDyadicPrefix):
                return 0, maximum_modes
            if prefix.bits >= prefix.max_bits:
                return 0, maximum_modes
            prefix.refine(min(int(profile.prefix_block_bits), prefix.max_bits-prefix.bits))

    for value in candidates:
        if not (0.0 < value < 1.0 and math.isfinite(value)):
            continue
        lower_boundary, upper_boundary = _reference._rounding_cell(value)
        lower_decision, modes = decide(lower_boundary)
        maximum_modes = max(maximum_modes, int(modes))
        if lower_decision != 1:  # Need F(b-) < U.
            continue
        upper_decision, modes = decide(upper_boundary)
        maximum_modes = max(maximum_modes, int(modes))
        if upper_decision != -1:  # Need F(b+) > U.
            continue
        target, target_interval, target_modes = _arb_candidate_target_interval(
            x_value, value, u_value, profile
        )
        maximum_modes = max(maximum_modes, int(target_modes))
        return {
            "later": float(value),
            "target": float(target),
            "quantile_lower": _fraction_down(lower_boundary),
            "quantile_upper": _fraction_up(upper_boundary),
            "target_lower": min(float(target), float(target_interval.lower)),
            "target_upper": max(float(target), float(target_interval.upper)),
            "prefix_bits": int(prefix.bits),
            "modes": int(maximum_modes),
            "certificate_code": 15,
        }
    raise _reference.JacobiRBCertificationError(
        "candidate-local Arb lattice did not find a certified rounding cell",
        {
            "failure_kind": "arb_resource_cap",
            "resource_kind": "candidate_lattice",
            "candidate": candidate,
            "maximum_neighbor_ulps": _ARB_CANDIDATE_LATTICE_ULPS,
            "prefix_bits": int(prefix.bits),
        },
    )


_ARB_POOL: ProcessPoolExecutor | None = None
_ARB_POOL_LOCK = threading.Lock()


def _shutdown_arb_pool() -> None:
    global _ARB_POOL
    with _ARB_POOL_LOCK:
        pool, _ARB_POOL = _ARB_POOL, None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


def _arb_pool() -> ProcessPoolExecutor:
    global _ARB_POOL
    with _ARB_POOL_LOCK:
        if _ARB_POOL is None:
            workers = max(1, min(8, int(os.cpu_count() or 1)))
            _ARB_POOL = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
            atexit.register(_shutdown_arb_pool)
        return _ARB_POOL


def _sample_alpha1_rb_transition_batch_cuda_core(
    head_fraction: Tensor,
    exposure: Tensor,
    *,
    rng_key: Any,
    transition_ids: Tensor,
    profile: JacobiRBCudaProfile,
    _prefix_factory: Any | None = None,
    _rng_contract: str = _RNG_VERSION,
    _recorded_prefix_numerators: Tensor | None = None,
    _recorded_prefix_bits: Tensor | None = None,
    _parent_v1_key_candidates: list[int | None] | None = None,
) -> CertifiedRBCudaBatch:
    """Return device-resident, correctly-rounded certified Jacobi transitions.

    Inputs are intentionally strict: three contiguous, same-shape CUDA tensors
    with dtypes ``float64``, ``float64``, and ``uint64``.  ``transition_ids``
    are the batching-invariant Philox namespace.  Candidate agreement remains
    diagnostic; only the fused cell proof or candidate-local Arb authorizes.
    Zero-duration rows are exact no-ops and consume no Philox prefix.
    """

    selected = profile
    if not isinstance(selected, JacobiRBCudaProfile):
        raise TypeError("profile must be a JacobiRBCudaProfile")
    _require_cuda_inputs(head_fraction, exposure, transition_ids)
    seed = _canonical_seed(rng_key)
    device, shape = head_fraction.device, head_fraction.shape
    count = int(head_fraction.numel())
    flat_x = head_fraction.reshape(-1)
    flat_u = exposure.reshape(-1)
    flat_ids = transition_ids.reshape(-1)
    candidate_y = torch.empty_like(flat_x)
    candidate_z = torch.empty_like(flat_x)
    candidate_lower = torch.empty_like(flat_x)
    candidate_upper = torch.empty_like(flat_x)
    active = flat_u > 0.0
    active_count = int(active.sum().item())
    candidate_started = time.perf_counter()
    binary_sha256: str | None = None

    if count == 0:
        candidate_y.copy_(flat_x)
        candidate_z.zero_()
        candidate_lower.copy_(flat_x)
        candidate_upper.copy_(flat_x)
    elif active_count == 0:
        # Exact no-ops need neither NVRTC nor an authorizing numerical backend.
        candidate_y.copy_(flat_x)
        candidate_z.zero_()
        candidate_lower.copy_(flat_x)
        candidate_upper.copy_(flat_x)
    else:
        kernel, binary_sha256 = _load_cuda_kernel(device, selected)
        seed_tensor = torch.tensor([seed], dtype=torch.uint64, device=device)
        recorded_values = (
            _recorded_prefix_numerators.reshape(-1)
            if _recorded_prefix_numerators is not None
            else torch.zeros(count, dtype=torch.uint64, device=device)
        )
        recorded_lengths = (
            _recorded_prefix_bits.reshape(-1).to(torch.int32)
            if _recorded_prefix_bits is not None
            else torch.zeros(count, dtype=torch.int32, device=device)
        )
        prefix_kind = int(_recorded_prefix_numerators is not None)
        threads = int(selected.threads_per_block)
        blocks = (count + threads - 1) // threads
        kernel(
            grid=(blocks, 1, 1),
            block=(threads, 1, 1),
            args=[
                flat_x,
                flat_u,
                flat_ids,
                seed_tensor,
                recorded_values,
                recorded_lengths,
                prefix_kind,
                count,
                int(selected.candidate_modes),
                int(selected.candidate_bisection_steps),
                candidate_y,
                candidate_z,
                candidate_lower,
                candidate_upper,
            ],
            stream=torch.cuda.current_stream(device),
        )
        # The following host certification necessarily synchronizes; do it
        # here so candidate and Arb timings do not charge each other's work.
        torch.cuda.synchronize(device)
    candidate_elapsed = time.perf_counter() - candidate_started
    later = flat_x.clone()
    target = torch.zeros_like(flat_x)
    q_lower = flat_x.clone()
    q_upper = flat_x.clone()
    z_lower = torch.zeros_like(flat_x)
    z_upper = torch.zeros_like(flat_x)
    prefix_bits = torch.zeros(count, dtype=torch.int32, device=device)
    codes = torch.zeros(count, dtype=torch.uint8, device=device)
    cuda_certified = torch.zeros(count, dtype=torch.bool, device=device)
    strengthened = torch.zeros(count, dtype=torch.bool, device=device)
    fallback_reasons = torch.zeros(count, dtype=torch.uint8, device=device)
    fallback_modes = torch.zeros(count, dtype=torch.int32, device=device)
    device_modes = torch.zeros(count, dtype=torch.int32, device=device)
    maximum_cuda_launch_lanes = count if active_count else 0
    fused_authorizer_launch_count = 0
    fused_report: dict[str, Any] = {
        "fused_cuda_authorizer_available": False,
        "fused_cuda_authorizer_unavailable_reason": "no active rows; runtime not exercised",
    }
    fused_elapsed = 0.0
    certification_started = time.perf_counter()
    if active_count:
        cpu_preflight = dict(_certificate_arithmetic_preflight())
        bundle, fused_report = probe_fused_cuda_authorizer(
            device, compile_flags=tuple(selected.compile_flags),
            cpu_preflight=cpu_preflight,
        )
        if bundle is not None:
            fused_started = time.perf_counter()
            fused = launch_fused_cuda_authorizer_with_neighbors(
                bundle,
                flat_x,
                flat_u,
                flat_ids,
                candidate_y,
                seed=seed,
                threads_per_block=int(selected.threads_per_block),
                max_prefix_bits=int(selected.max_prefix_bits),
                recorded_prefix_numerators=(
                    None if _recorded_prefix_numerators is None
                    else _recorded_prefix_numerators.reshape(-1)
                ),
                recorded_prefix_bits=(
                    None if _recorded_prefix_bits is None
                    else _recorded_prefix_bits.reshape(-1).to(torch.int32)
                ),
                force_strengthened=(selected.certificate_effort == "strengthened"),
            )
            torch.cuda.synchronize(device)
            fused_elapsed = time.perf_counter() - fused_started
            later, target = fused.later, fused.target
            q_lower, q_upper = fused.quantile_lower, fused.quantile_upper
            z_lower, z_upper = fused.target_lower, fused.target_upper
            prefix_bits, codes = fused.prefix_bits, fused.certificate_codes
            device_modes = fused.modes_used
            cuda_certified = fused.authorized_mask
            strengthened = fused.strengthened_mask
            fallback_reasons = fused.fallback_reason_codes
            maximum_cuda_launch_lanes = max(
                maximum_cuda_launch_lanes, int(fused.maximum_launch_lanes)
            )
            fused_authorizer_launch_count = int(fused.launch_count)
        else:
            raise RuntimeError(
                "fused CUDA authorizer contract failed before numerical "
                f"certification: {fused_report.get('fused_cuda_authorizer_unavailable_reason')}"
            )

    fallback = active & ~cuda_certified
    device_prefix_bits = prefix_bits.clone()
    strengthened |= fallback  # Arb is reached only after the strengthened pass.
    fallback_count = int(fallback.sum().item())
    arb_started = time.perf_counter()
    if fallback_count:
        host_x = flat_x.detach().cpu().tolist()
        host_u = flat_u.detach().cpu().tolist()
        host_ids = flat_ids.detach().cpu().tolist()
        # Unresolved fused rows carry a non-authorizing DD Newton suggestion
        # in ``later``; this is the centre of the bounded Arb cell lattice.
        host_candidate = later.detach().cpu().tolist()
        host_fallback = fallback.detach().cpu().tolist()
        host_later = later.detach().cpu().tolist()
        host_target = target.detach().cpu().tolist()
        host_q_lower = q_lower.detach().cpu().tolist()
        host_q_upper = q_upper.detach().cpu().tolist()
        host_z_lower = z_lower.detach().cpu().tolist()
        host_z_upper = z_upper.detach().cpu().tolist()
        host_prefix = prefix_bits.detach().cpu().tolist()
        host_codes = codes.detach().cpu().tolist()
        host_modes = [0] * count
        recorded_numerators = (
            None if _recorded_prefix_numerators is None
            else _recorded_prefix_numerators.reshape(-1).detach().cpu().tolist()
        )
        recorded_bits = (
            None if _recorded_prefix_bits is None
            else _recorded_prefix_bits.reshape(-1).detach().cpu().tolist()
        )
        ref_profile = _reference_profile(selected)
        pending: list[tuple[int, Any]] = []
        pool = _arb_pool()
        for index, needs_fallback in enumerate(host_fallback):
            if not bool(needs_fallback):
                continue
            payload: dict[str, Any] = {
                "x": float(host_x[index]),
                "exposure": float(host_u[index]),
                "candidate": float(host_candidate[index]),
                "transition_id": int(host_ids[index]),
                "profile": ref_profile,
                "max_prefix_bits": int(selected.max_prefix_bits),
            }
            if recorded_numerators is not None and recorded_bits is not None:
                parent_candidate = (
                    None if _parent_v1_key_candidates is None
                    else _parent_v1_key_candidates[index]
                )
                payload.update(
                    prefix_kind=(
                        "recorded" if parent_candidate is None
                        else "parent-v1-verified-continuation"
                    ),
                    prefix_numerator=int(recorded_numerators[index]),
                    prefix_bits=int(recorded_bits[index]),
                    v1_key_candidate=parent_candidate,
                )
            else:
                payload.update(prefix_kind="philox-v2", seed=int(seed))
            pending.append((index, pool.submit(_arb_candidate_cell_worker, payload)))
        for index, future in pending:
            try:
                row = future.result()
            except Exception as exc:
                diagnostics = getattr(exc, "diagnostics", {})
                raise _reference.JacobiRBCertificationError(
                    "candidate-local Arb fallback failed closed",
                    {
                        "sample_index": int(index),
                        "transition_id": int(host_ids[index]),
                        "cuda_fallback_reason_code": int(
                            fallback_reasons[index].item()
                        ),
                        **dict(diagnostics),
                    },
                ) from exc
            host_later[index] = row["later"]
            host_target[index] = row["target"]
            host_q_lower[index] = row["quantile_lower"]
            host_q_upper[index] = row["quantile_upper"]
            host_z_lower[index] = row["target_lower"]
            host_z_upper[index] = row["target_upper"]
            host_prefix[index] = row["prefix_bits"]
            host_codes[index] = row["certificate_code"]
            host_modes[index] = row["modes"]

        def copy_host(destination: Tensor, values: Any, dtype: Any) -> None:
            destination.copy_(torch.tensor(values, dtype=dtype, device=device))

        copy_host(later, host_later, torch.float64)
        copy_host(target, host_target, torch.float64)
        copy_host(q_lower, host_q_lower, torch.float64)
        copy_host(q_upper, host_q_upper, torch.float64)
        copy_host(z_lower, host_z_lower, torch.float64)
        copy_host(z_upper, host_z_upper, torch.float64)
        copy_host(prefix_bits, host_prefix, torch.int32)
        copy_host(codes, host_codes, torch.uint8)
        copy_host(fallback_modes, host_modes, torch.int32)

    arb_elapsed = time.perf_counter() - arb_started
    certification_elapsed = time.perf_counter() - certification_started

    candidate_match = active & (candidate_y == later) & (candidate_z == target)
    certified = (codes & 0b1111) == 0b1111
    if bool((active & ~certified).any().item()):  # defensive fail-closed invariant.
        raise RuntimeError("internal error: an active CUDA transition lacks certification")
    final_finite = (
        torch.isfinite(later) & torch.isfinite(target)
        & torch.isfinite(q_lower) & torch.isfinite(q_upper)
        & torch.isfinite(z_lower) & torch.isfinite(z_upper)
    )
    if bool((active & ~final_finite).any().item()):
        raise RuntimeError("internal error: a certified transition has nonfinite output")
    candidate_finite = torch.isfinite(candidate_y) & torch.isfinite(candidate_z)
    diagnostics: dict[str, Tensor] = {
        "sample_count": _device_count(count, device),
        "active_count": _device_count(active_count, device),
        "zero_duration_count": _device_count(count - active_count, device),
        "candidate_count": _device_count(active_count, device),
        "cuda_authorized_count": cuda_certified.sum(dtype=torch.int64),
        "fallback_count": fallback.sum(dtype=torch.int64),
        "certified_count": _device_count(active_count, device),
        "candidate_match_count": candidate_match.sum(dtype=torch.int64),
        "strengthened_count": strengthened.sum(dtype=torch.int64),
        "candidate_elapsed_seconds": torch.tensor(
            candidate_elapsed, dtype=torch.float64, device=device
        ),
        "arb_fallback_elapsed_seconds": torch.tensor(
            arb_elapsed, dtype=torch.float64, device=device
        ),
        "fused_authorizer_elapsed_seconds": torch.tensor(
            fused_elapsed, dtype=torch.float64, device=device
        ),
        "maximum_cuda_launch_lanes": _device_count(
            maximum_cuda_launch_lanes, device
        ),
        "candidate_kernel_launch_count": _device_count(
            1 if active_count else 0, device
        ),
        "fused_authorizer_launch_count": _device_count(
            fused_authorizer_launch_count, device
        ),
        "certificate_elapsed_seconds": torch.tensor(
            certification_elapsed, dtype=torch.float64, device=device
        ),
        "mode_cap_hit_count": (fallback & (device_modes >= 8192)).sum(
            dtype=torch.int64
        ),
        "prefix_cap_hit_count": (
            fallback & (device_prefix_bits >= int(selected.max_prefix_bits))
        ).sum(dtype=torch.int64),
        "cuda_strengthened_exhaustion_count": (
            fallback
            & (
                (device_modes >= 8192)
                | (device_prefix_bits >= int(selected.max_prefix_bits))
            )
        ).sum(dtype=torch.int64),
        # A returned Arb fallback is a completed certificate, not a resource
        # failure.  Exhausting the CUDA escalation is reported separately
        # above; an actual Arb resource failure raises before a result exists.
        "resource_cap_count": _device_count(0, device),
        "invalid_density_count": (
            fallback_reasons == int(_FallbackReason.NONPOSITIVE_DENSITY)
        ).sum(dtype=torch.int64),
        "nonfinite_count": (active & ~final_finite).sum(dtype=torch.int64),
        "candidate_nonfinite_count": (active & ~candidate_finite).sum(
            dtype=torch.int64
        ),
        "approximation_count": _device_count(0, device),
        "candidate_repair_count": (cuda_certified & (candidate_y != later)).sum(
            dtype=torch.int64
        ),
        "correction_count": _device_count(0, device),
        "floor_count": _device_count(0, device),
        "limiter_count": _device_count(0, device),
        "renormalization_count": _device_count(0, device),
        "maximum_cuda_modes": (
            device_modes.max() if count else _device_count(0, device).to(torch.int32)
        ),
        "maximum_arb_fallback_modes": (
            fallback_modes.max() if count else _device_count(0, device).to(torch.int32)
        ),
    }
    runtime = _runtime_report(
        device, profile=selected, probe_authorizer=bool(active_count)
    )
    runtime["profile"] = selected.to_dict()
    runtime["candidate_kernel_sha256"] = _KERNEL_SHA256
    runtime["candidate_binary_sha256"] = binary_sha256
    runtime.update(fused_report)
    runtime["kernel_sha256"] = fused_report.get("fused_source_sha256", _KERNEL_SHA256)
    runtime["source_sha256"] = runtime["kernel_sha256"]
    runtime["binary_sha256"] = fused_report.get("fused_binary_sha256", binary_sha256)
    runtime["cubin_sha256"] = runtime["binary_sha256"]
    runtime["directed_rounding_intrinsics_pass"] = bool(
        fused_report.get("arithmetic_selftest_pass", False)
    )
    runtime["compile_flags_exact_pass"] = tuple(selected.compile_flags) == _COMPILE_FLAGS
    runtime["runtime_contract_pass"] = bool(runtime.get("frozen_runtime_match", False))
    runtime["device_contract_pass"] = (
        runtime.get("compute_capability") == str(selected.frozen_compute_capability)
    )
    runtime["rng_contract"] = str(_rng_contract)
    return CertifiedRBCudaBatch(
        earlier_head_fraction=head_fraction,
        later_head_fraction=later.reshape(shape),
        denoising_target=target.reshape(shape),
        exposure=exposure,
        transition_ids=transition_ids,
        active_mask=active.reshape(shape),
        certified_mask=certified.reshape(shape),
        candidate_later_head_fraction=candidate_y.reshape(shape),
        candidate_denoising_target=candidate_z.reshape(shape),
        candidate_match_mask=candidate_match.reshape(shape),
        cuda_certified_mask=cuda_certified.reshape(shape),
        fallback_mask=fallback.reshape(shape),
        strengthened_mask=strengthened.reshape(shape),
        arb_fallback_reason_codes=fallback_reasons.reshape(shape),
        arb_fallback_mode_counts=fallback_modes.reshape(shape),
        mode_counts=torch.maximum(device_modes, fallback_modes).reshape(shape),
        quantile_lower=q_lower.reshape(shape),
        quantile_upper=q_upper.reshape(shape),
        target_lower=z_lower.reshape(shape),
        target_upper=z_upper.reshape(shape),
        prefix_bits=prefix_bits.reshape(shape),
        certificate_codes=codes.reshape(shape),
        diagnostics=diagnostics,
        runtime_report=runtime,
    )


def sample_alpha1_rb_transition_batch_cuda(
    head_fraction: Tensor,
    exposure: Tensor,
    *,
    rng_key: Any,
    transition_ids: Tensor,
    profile: JacobiRBCudaProfile,
) -> CertifiedRBCudaBatch:
    """Public v2 stateless-Philox CUDA candidate/certificate/fallback API.

    This entry point has no parent-v1 RNG compatibility mode.  Callers
    replaying recorded dyadic prefixes must use the explicitly named internal
    helper below, so a historical control cannot be mislabeled as v2 RNG
    continuity.
    """

    return _sample_alpha1_rb_transition_batch_cuda_core(
        head_fraction,
        exposure,
        rng_key=rng_key,
        transition_ids=transition_ids,
        profile=profile,
    )


def certify_alpha1_rb_transition_batch_cuda_with_dyadic_prefixes(
    head_fraction: Tensor,
    exposure: Tensor,
    prefix_numerators: Tensor,
    prefix_bits: Tensor,
    *,
    transition_ids: Tensor,
    profile: JacobiRBCudaProfile,
) -> CertifiedRBCudaBatch:
    """Replay explicitly recorded parent-v1 dyadic prefixes, fail-closed.

    Prefix tensors must be contiguous CUDA ``uint64`` tensors matching the
    transition shape; ``prefix_bits`` may be ``uint8``, ``int16``, ``int32``,
    or ``int64``.  Only 1--64 recorded bits are supported.  If Arb needs a
    further random bit, replay fails rather than fabricating RNG continuation.
    The result report says ``parent-v1-recorded-dyadic-prefix`` and never
    claims continuity with the public stateless-Philox contract.
    """

    _require_cuda_inputs(head_fraction, exposure, transition_ids)
    for name, tensor in (
        ("prefix_numerators", prefix_numerators),
        ("prefix_bits", prefix_bits),
    ):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            raise TypeError(f"{name} must be a CUDA tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.shape != head_fraction.shape or tensor.device != head_fraction.device:
            raise ValueError(f"{name} must match the transition shape and device")
    if prefix_numerators.dtype != torch.uint64:
        raise TypeError("prefix_numerators must have dtype torch.uint64")
    if prefix_bits.dtype not in {torch.uint8, torch.int16, torch.int32, torch.int64}:
        raise TypeError("prefix_bits must have an integer dtype")
    host_numerators = prefix_numerators.reshape(-1).detach().cpu().tolist()
    host_bits = prefix_bits.reshape(-1).detach().cpu().tolist()
    host_exposure = exposure.reshape(-1).detach().cpu().tolist()
    for numerator, bits, duration in zip(
        host_numerators, host_bits, host_exposure, strict=True
    ):
        if float(duration) == 0.0:
            if int(numerator) != 0 or int(bits) != 0:
                raise ValueError("zero-duration rows must carry the empty recorded prefix")
        else:
            _FixedDyadicPrefix(int(numerator), int(bits))

    parent_map = _parent_v1_prefix_candidate_map()
    parent_candidates: list[int | None] = [
        (
            parent_map.get(int(numerator))
            if float(duration) > 0.0 and int(bits) == 64
            else None
        )
        for numerator, bits, duration in zip(
            host_numerators, host_bits, host_exposure, strict=True
        )
    ]
    verified_parent_continuation = all(
        float(duration) == 0.0 or candidate is not None
        for duration, candidate in zip(host_exposure, parent_candidates, strict=True)
    )

    def factory(index: int, _transition_id: int) -> _FixedDyadicPrefix:
        return _FixedDyadicPrefix(int(host_numerators[index]), int(host_bits[index]))

    result = _sample_alpha1_rb_transition_batch_cuda_core(
        head_fraction,
        exposure,
        # Candidate RNG is isolated and explicitly non-authorizing.  The
        # recorded prefixes alone drive every authorizing comparison.
        rng_key=("parent-v1-recorded-prefix-candidate-only", 0),
        transition_ids=transition_ids,
        profile=profile,
        _prefix_factory=factory,
        _rng_contract=(
            "parent-v1-recorded-prefix+verified-continuation"
            if verified_parent_continuation
            else "parent-v1-recorded-dyadic-prefix"
        ),
        _recorded_prefix_numerators=prefix_numerators,
        _recorded_prefix_bits=prefix_bits,
        _parent_v1_key_candidates=parent_candidates,
    )
    return result


__all__ = [
    "JacobiRBCudaProfile",
    "CertifiedRBCudaBatch",
    "sample_alpha1_rb_transition_batch_cuda",
]
