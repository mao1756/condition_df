r"""Header-free NVRTC fused certificate for alpha-one Jacobi transitions.

This module is intentionally lower level than :mod:`mnist.d0_jacobi_rb_cuda`.
It owns the device arithmetic, its mandatory arithmetic self-test, and one
kernel launch.  It does not own a fallback and it never turns an unresolved
comparison into a certificate.

The device proof represents a value as a double-double centre plus an outward
binary64 radius.  Centre operations use error-free TwoSum and FMA TwoProd;
all radius operations use explicit IEEE directed-rounding intrinsics.  The
small local-error padding used after double-double compression is deliberately
loose (``2^-90`` times the operation scale plus sixteen subnormals).  This is
more than 65,000 times the standard ``u^2`` double-double error constant and
is therefore valid for the finite, non-overflowing operations admitted by the
kernel.  Any failed precondition sets an arithmetic-fault reason code.

``exp`` is not delegated to libdevice for authorization.  It uses an exact
two-float bracket for ln(2), a fixed degree-24 Taylor polynomial whose
coefficient loads are bracketed by adjacent binary64 constants, and the
rigorous remainder bound ``2^-116`` on ``|r| < 3/8``.  Analytic geometric
tails enclose all omitted Legendre modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import threading
from typing import Any, Mapping

from mnist.d0_jacobi_rb_cuda_certificate import (
    EXP24_REMAINDER_BOUND,
    LN2_LOWER,
    LN2_UPPER,
)

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover
    torch = None
    Tensor = Any


FUSED_CUDA_VERSION = "alpha1-jacobi-rb-fused-dd-exp24-v1"
FUSED_KERNEL_NAME = "jacobi_rb_fused_authorize_v1"
SELFTEST_KERNEL_NAME = "jacobi_rb_fused_selftest_v1"
LEGENDRE_PROBE_KERNEL_NAME = "jacobi_rb_fused_legendre_probe_v1"
LEGENDRE_RECURRENCE_THEOREM = (
    "Johansson-Mezzarobba-2018-Proposition-5"
)
LEGENDRE_RECURRENCE_THEOREM_URI = "https://arxiv.org/abs/1802.03948"
LEGENDRE_RECURRENCE_ERROR_FACTOR = "(N+1)*(N+2)/4"

# Internal device-only CDF miss directions.  They deliberately sit outside
# the public fallback enum: a final Arb fallback still records the precise
# CUDA miss, while the wrapper can avoid probing the mathematically wrong
# side of the candidate lattice.
_CDF_CANDIDATE_TOO_HIGH = 23
_CDF_CANDIDATE_TOO_LOW = 24

# add-down/up, mul-down/up, div-down/up, TwoSum, FMA-TwoProd, exp24,
# exact rounding boundary, DD-ball propagation, certified exponential
# underflow, and conservative rejection of subnormal rounding cells.
REQUIRED_SELFTEST_MASK = (1 << 17) - 1

LN2_DD_HEX = (
    "0x1.62e42fefa39efp-1",
    "0x1.abc9e3b39803fp-56",
    "0x1.7b57a079a1934p-111",
)
EXP_DD_HI_HEX = tuple("""0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p-1,0x1.5555555555555p-3,0x1.5555555555555p-5,0x1.1111111111111p-7,0x1.6c16c16c16c17p-10,0x1.a01a01a01a01ap-13,0x1.a01a01a01a01ap-16,0x1.71de3a556c734p-19,0x1.27e4fb7789f5cp-22,0x1.ae64567f544e4p-26,0x1.1eed8eff8d898p-29,0x1.6124613a86d09p-33,0x1.93974a8c07c9dp-37,0x1.ae7f3e733b81fp-41,0x1.ae7f3e733b81fp-45,0x1.952c77030ad4ap-49,0x1.6827863b97d97p-53,0x1.2f49b46814157p-57,0x1.e542ba4020225p-62,0x1.71b8ef6dcf572p-66,0x1.0ce396db7f853p-70,0x1.761b41316381ap-75,0x1.f2cf01972f578p-80""".split(","))
EXP_DD_LO_HEX = tuple("""0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.5555555555555p-57,0x1.5555555555555p-59,0x1.1111111111111p-63,-0x1.f49f49f49f49fp-65,0x1.a01a01a01a01ap-73,0x1.a01a01a01a01ap-76,-0x1.c154f8ddc6c00p-73,0x1.cbbc05b4fa99ap-76,-0x1.c062e06d1f209p-80,-0x1.2aec959e14c06p-83,0x1.f28e0cc748ebep-87,0x1.05d6f8a2efd1fp-92,0x1.1d8656b0ee8cbp-97,0x1.1d8656b0ee8cbp-101,0x1.ac981465ddc6cp-103,0x1.eec01221a8b0bp-107,0x1.2650f61dbdcb4p-112,0x1.ea72b4afe3c2fp-120,-0x1.d043ae40c4647p-120,-0x1.aebcdbd20331cp-124,-0x1.3423c7d91404fp-130,-0x1.9ada5fcc1ab14p-135""".split(","))
EXP_DD_RAD_HEX = tuple("""0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.5555555555556p-111,0x1.5555555555556p-113,0x1.1111111111112p-119,0x1.27d27d27d27d3p-119,0x1.a01a01a01a01bp-133,0x1.a01a01a01a01bp-136,0x1.71de3a556c734p-127,0x1.c6d278883e8f5p-132,0x1.c7880adcbc46ep-136,0x1.2fb0073dd2d9fp-139,0x1.7b2c4c8a840bcp-141,0x1.3aa3346236a5ep-147,0x1.6e142a138f825p-157,0x1.6e142a138f825p-161,0x1.588b72e53bc5fp-165,0x1.568798662118bp-161,0x1.69502917cbf3bp-166,0x1.44020dfd65c8dp-174,0x1.486121e81d5fep-176,0x1.38a88578b4d75p-178,0x1.e6135bfc1194ap-185,0x1.440ce7fd610dcp-189""".split(","))


_CUDA_SOURCE = r"""
typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;

struct DD { double hi; double lo; };
struct Ball { DD c; double r; int ok; };

__device__ __forceinline__ double dabs(double x) {
    return __longlong_as_double((long long)(__double_as_longlong(x) & 0x7fffffffffffffffULL));
}
__device__ __forceinline__ int finite_d(double x) {
    return ((__double_as_longlong(x) >> 52) & 0x7ffULL) != 0x7ffULL;
}
__device__ __forceinline__ double dmax(double a, double b) { return a > b ? a : b; }
__device__ __forceinline__ double dmin(double a, double b) { return a < b ? a : b; }

__device__ __forceinline__ DD two_sum(double a, double b) {
    double s = __dadd_rn(a, b);
    double bb = __dsub_rn(s, a);
    double aa = __dsub_rn(s, bb);
    double da = __dsub_rn(a, aa);
    double db = __dsub_rn(b, bb);
    DD out = {s, __dadd_rn(da, db)};
    return out;
}
__device__ __forceinline__ DD quick_two_sum(double a, double b) {
    double s = __dadd_rn(a, b);
    DD out = {s, __dsub_rn(b, __dsub_rn(s, a))};
    return out;
}
__device__ __forceinline__ DD two_prod(double a, double b) {
    double p = __dmul_rn(a, b);
    DD out = {p, __fma_rn(a, b, -p)};
    return out;
}
__device__ __forceinline__ DD dd_renorm(double hi, double lo) {
    return dabs(hi) >= dabs(lo) ? quick_two_sum(hi, lo) : quick_two_sum(lo, hi);
}
__device__ __forceinline__ DD dd_add_raw(DD a, DD b) {
    DD s = two_sum(a.hi, b.hi);
    DD t = two_sum(a.lo, b.lo);
    DD u = two_sum(s.lo, t.hi);
    DD v = two_sum(s.hi, u.hi);
    double tail = __dadd_rn(__dadd_rn(u.lo, t.lo), v.lo);
    return dd_renorm(v.hi, tail);
}
__device__ __forceinline__ DD dd_neg(DD a) { DD out = {-a.hi, -a.lo}; return out; }
__device__ __forceinline__ DD dd_sub_raw(DD a, DD b) { return dd_add_raw(a, dd_neg(b)); }
__device__ __forceinline__ DD dd_mul_raw(DD a, DD b) {
    DD p = two_prod(a.hi, b.hi);
    double cross = __dadd_rn(__dmul_rn(a.hi, b.lo), __dmul_rn(a.lo, b.hi));
    double tail = __dadd_rn(__dadd_rn(p.lo, cross), __dmul_rn(a.lo, b.lo));
    return dd_renorm(p.hi, tail);
}
__device__ __forceinline__ DD dd_div_raw(DD a, DD b) {
    double q1 = __ddiv_rn(a.hi, b.hi);
    DD q = {q1, 0.0};
    DD residual = dd_sub_raw(a, dd_mul_raw(b, q));
    double q2 = __ddiv_rn(__dadd_rn(residual.hi, residual.lo), b.hi);
    DD correction = {q2, 0.0};
    return dd_add_raw(q, correction);
}
__device__ __forceinline__ double dd_down(DD a) { return __dadd_rd(a.hi, a.lo); }
__device__ __forceinline__ double dd_up(DD a) { return __dadd_ru(a.hi, a.lo); }
__device__ __forceinline__ double dd_rn(DD a) { return __dadd_rn(a.hi, a.lo); }
__device__ __forceinline__ double dd_mag(DD a) {
    return __dadd_ru(dabs(a.hi), dabs(a.lo));
}
__device__ __forceinline__ double local_error(double scale) {
    // 2^-90 is deliberately far larger than a binary64 DD operation's u^2 bound.
    return __dadd_ru(__dmul_ru(dabs(scale), 0x1.0p-90), 0x1.0p-1070);
}
__device__ __forceinline__ Ball invalid_ball() {
    Ball out = {{0.0, 0.0}, 0.0, 0}; return out;
}
__device__ __forceinline__ Ball exact_double(double x) {
    Ball out = {{x, 0.0}, 0.0, finite_d(x)}; return out;
}
__device__ __forceinline__ Ball exact_dd(DD x) {
    Ball out = {x, 0.0, finite_d(x.hi) && finite_d(x.lo)}; return out;
}
__device__ __forceinline__ double ball_lower(Ball a) {
    return __dsub_rd(dd_down(a.c), a.r);
}
__device__ __forceinline__ double ball_upper(Ball a) {
    return __dadd_ru(dd_up(a.c), a.r);
}
__device__ __forceinline__ Ball ball_neg(Ball a) {
    Ball out = {dd_neg(a.c), a.r, a.ok}; return out;
}
__device__ __forceinline__ Ball ball_add(Ball a, Ball b) {
    if (!a.ok || !b.ok) return invalid_ball();
    Ball out;
    out.c = dd_add_raw(a.c, b.c);
    double scale = __dadd_ru(dd_mag(a.c), dd_mag(b.c));
    out.r = __dadd_ru(__dadd_ru(a.r, b.r), local_error(scale));
    out.ok = finite_d(out.c.hi) && finite_d(out.c.lo) && finite_d(out.r);
    return out;
}
__device__ __forceinline__ Ball ball_sub(Ball a, Ball b) { return ball_add(a, ball_neg(b)); }
__device__ __forceinline__ Ball ball_mul(Ball a, Ball b) {
    if (!a.ok || !b.ok) return invalid_ball();
    Ball out;
    out.c = dd_mul_raw(a.c, b.c);
    double ma = dd_mag(a.c), mb = dd_mag(b.c);
    double propagated = __dadd_ru(
        __dadd_ru(__dmul_ru(ma, b.r), __dmul_ru(mb, a.r)),
        __dmul_ru(a.r, b.r));
    double scale = __dmul_ru(ma, mb);
    out.r = __dadd_ru(propagated, local_error(scale));
    out.ok = finite_d(out.c.hi) && finite_d(out.c.lo) && finite_d(out.r);
    return out;
}
__device__ __forceinline__ Ball ball_div(Ball a, Ball b) {
    if (!a.ok || !b.ok) return invalid_ball();
    double blo = ball_lower(b), bhi = ball_upper(b);
    if (!(blo > 0.0 || bhi < 0.0)) return invalid_ball();
    double denominator = blo > 0.0 ? blo : -bhi;
    if (!(denominator > 0.0)) return invalid_ball();
    Ball out;
    out.c = dd_div_raw(a.c, b.c);
    if (!finite_d(out.c.hi) || !finite_d(out.c.lo)) return invalid_ball();
    // Certify the computed quotient directly.  For q=out.c,
    // A/B-q=(A-Bq)/B.  Ball multiplication/subtraction enclose the complete
    // residual, including the DD division error and both input radii.
    Ball quotient_centre=exact_dd(out.c);
    Ball residual=ball_sub(a,ball_mul(b,quotient_centre));
    if (!residual.ok) return invalid_ball();
    double residual_mag=dmax(dabs(ball_lower(residual)),dabs(ball_upper(residual)));
    out.r=__ddiv_ru(residual_mag,denominator);
    out.ok = finite_d(out.c.hi) && finite_d(out.c.lo) && finite_d(out.r);
    return out;
}
__device__ __forceinline__ Ball ball_expand(Ball a, double radius) {
    if (!a.ok || !(radius >= 0.0) || !finite_d(radius)) return invalid_ball();
    a.r = __dadd_ru(a.r, radius);
    a.ok = finite_d(a.r);
    return a;
}
__device__ __forceinline__ Ball ball_scale(Ball a, double scalar) {
    return ball_mul(a, exact_double(scalar));
}
__device__ __forceinline__ int ball_strict_less(Ball a, Ball b) {
    Ball difference=ball_sub(b,a);
    return difference.ok && ball_lower(difference)>0.0;
}

__device__ __forceinline__ double pow2_exact(int exponent) {
    if (exponent > 1023) return __longlong_as_double(0x7ff0000000000000ULL);
    if (exponent >= -1022) {
        return __longlong_as_double((u64)(exponent + 1023) << 52);
    }
    if (exponent >= -1074) return __longlong_as_double(1ULL << (exponent + 1074));
    return 0.0;
}
__device__ __forceinline__ double next_up(double x) {
    u64 bits = (u64)__double_as_longlong(x);
    if (x == 0.0) return __longlong_as_double(1ULL);
    bits = x > 0.0 ? bits + 1ULL : bits - 1ULL;
    return __longlong_as_double(bits);
}
__device__ __forceinline__ double next_down(double x) {
    u64 bits = (u64)__double_as_longlong(x);
    if (x == 0.0) return __longlong_as_double(0x8000000000000001ULL);
    bits = x > 0.0 ? bits - 1ULL : bits + 1ULL;
    return __longlong_as_double(bits);
}
__device__ __forceinline__ Ball exact_midpoint(double a, double b) {
    // Multiplication by 1/2 is exact unless a nonzero operand underflows.
    double ah = __dmul_rn(a, 0.5), bh = __dmul_rn(b, 0.5);
    if ((a != 0.0 && ah == 0.0) || (b != 0.0 && bh == 0.0)) return invalid_ball();
    return exact_dd(two_sum(ah, bh));
}
__device__ __forceinline__ int rounding_boundaries(double candidate, Ball* lower, Ball* upper) {
    if (!finite_d(candidate)) return 0;
    double previous = next_down(candidate), following = next_up(candidate);
    if (!finite_d(previous) || !finite_d(following)) return 0;
    // Halfway points involving a subnormal endpoint can contain a half-
    // minsub component, which is not representable as a two-double sum.
    // Reject if the candidate or either neighbour is nonzero subnormal; this
    // also covers the min-normal boundary.  Arb handles these rare cells.
    if ((candidate != 0.0 && dabs(candidate) < 0x1.0p-1022) ||
        (previous != 0.0 && dabs(previous) < 0x1.0p-1022) ||
        (following != 0.0 && dabs(following) < 0x1.0p-1022)) return 0;
    *lower = exact_midpoint(previous, candidate);
    *upper = exact_midpoint(candidate, following);
    return lower->ok && upper->ok;
}

// Exact 1/n! is enclosed as HI[n] + LO[n] +/- RAD[n].  Adjacent
// single-double coefficient brackets are much too wide for a rounding proof.
__device__ __constant__ double EXP_DD_HI[25] = {
  0x1.0000000000000p+0,0x1.0000000000000p+0,0x1.0000000000000p-1,
  0x1.5555555555555p-3,0x1.5555555555555p-5,0x1.1111111111111p-7,
  0x1.6c16c16c16c17p-10,0x1.a01a01a01a01ap-13,0x1.a01a01a01a01ap-16,
  0x1.71de3a556c734p-19,0x1.27e4fb7789f5cp-22,0x1.ae64567f544e4p-26,
  0x1.1eed8eff8d898p-29,0x1.6124613a86d09p-33,0x1.93974a8c07c9dp-37,
  0x1.ae7f3e733b81fp-41,0x1.ae7f3e733b81fp-45,0x1.952c77030ad4ap-49,
  0x1.6827863b97d97p-53,0x1.2f49b46814157p-57,0x1.e542ba4020225p-62,
  0x1.71b8ef6dcf572p-66,0x1.0ce396db7f853p-70,0x1.761b41316381ap-75,
  0x1.f2cf01972f578p-80};
__device__ __constant__ double EXP_DD_LO[25] = {
  0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.5555555555555p-57,
  0x1.5555555555555p-59,0x1.1111111111111p-63,-0x1.f49f49f49f49fp-65,
  0x1.a01a01a01a01ap-73,0x1.a01a01a01a01ap-76,-0x1.c154f8ddc6c00p-73,
  0x1.cbbc05b4fa99ap-76,-0x1.c062e06d1f209p-80,-0x1.2aec959e14c06p-83,
  0x1.f28e0cc748ebep-87,0x1.05d6f8a2efd1fp-92,0x1.1d8656b0ee8cbp-97,
  0x1.1d8656b0ee8cbp-101,0x1.ac981465ddc6cp-103,0x1.eec01221a8b0bp-107,
  0x1.2650f61dbdcb4p-112,0x1.ea72b4afe3c2fp-120,-0x1.d043ae40c4647p-120,
  -0x1.aebcdbd20331cp-124,-0x1.3423c7d91404fp-130,-0x1.9ada5fcc1ab14p-135};
__device__ __constant__ double EXP_DD_RAD[25] = {
  0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.5555555555556p-111,
  0x1.5555555555556p-113,0x1.1111111111112p-119,0x1.27d27d27d27d3p-119,
  0x1.a01a01a01a01bp-133,0x1.a01a01a01a01bp-136,0x1.71de3a556c734p-127,
  0x1.c6d278883e8f5p-132,0x1.c7880adcbc46ep-136,0x1.2fb0073dd2d9fp-139,
  0x1.7b2c4c8a840bcp-141,0x1.3aa3346236a5ep-147,0x1.6e142a138f825p-157,
  0x1.6e142a138f825p-161,0x1.588b72e53bc5fp-165,0x1.568798662118bp-161,
  0x1.69502917cbf3bp-166,0x1.44020dfd65c8dp-174,0x1.486121e81d5fep-176,
  0x1.38a88578b4d75p-178,0x1.e6135bfc1194ap-185,0x1.440ce7fd610dcp-189};

__device__ __forceinline__ Ball ball_from_bounds(double lo, double hi) {
    if (!finite_d(lo) || !finite_d(hi) || lo > hi) return invalid_ball();
    Ball middle = exact_midpoint(lo, hi);
    if (!middle.ok) return invalid_ball();
    double left = __dsub_ru(dd_up(middle.c), lo);
    double right = __dsub_ru(hi, dd_down(middle.c));
    middle.r = dmax(left, right);
    middle.ok = finite_d(middle.r);
    return middle;
}
__device__ __forceinline__ Ball exp_coefficient(int degree) {
    Ball out = {{EXP_DD_HI[degree], EXP_DD_LO[degree]}, EXP_DD_RAD[degree], 1};
    return out;
}
__device__ __forceinline__ Ball ln2_ball() {
    // Frozen 80-decimal ln(2) bracket split into a DD centre and radius.
    Ball out = {{0x1.62e42fefa39efp-1, 0x1.abc9e3b39803fp-56},
                0x1.7b57a079a1934p-111, 1};
    return out;
}
__device__ __forceinline__ Ball exp24(Ball x) {
    if (!x.ok || ball_upper(x) > 0.0) return invalid_ball();
    // -745 < -1074*ln(2), so exp(x) is strictly below one binary64
    // min-subnormal whenever the whole input ball lies at or below -745.
    // Handle this before the range-reduction integer conversion; converting
    // an arbitrarily negative finite double to long long is not admissible.
    if (ball_upper(x) <= -0x1.748p+9) {
        Ball tiny = {{0.0, 0.0}, 0x1.0p-1074, 1};
        return tiny;
    }
    double midpoint = dd_rn(x.c);
    int k = (int)__double2ll_rn(__dmul_rn(midpoint, 0x1.71547652b82fep+0));
    if (k < -16384 || k > 0) return invalid_ball();
    Ball reduction = ball_sub(x, ball_scale(ln2_ball(), (double)k));
    if (!reduction.ok || !(ball_lower(reduction) > -0.375) ||
        !(ball_upper(reduction) < 0.375)) return invalid_ball();
    Ball polynomial = exp_coefficient(24);
    for (int degree=23; degree>=0; --degree)
        polynomial = ball_add(ball_mul(polynomial, reduction), exp_coefficient(degree));
    // For |r|<3/8: exp(|r|)|r|^25/25! < 2^-116.
    polynomial = ball_expand(polynomial, 0x1.0p-116);
    if (!polynomial.ok) return invalid_ball();
    if (k < -1074) {
        // exp(r)<3/2, hence k<=-1075 is enclosed by one min-subnormal.
        Ball tiny = {{0.0, 0.0}, 0x1.0p-1074, 1};
        return tiny;
    }
    return ball_scale(polynomial, pow2_exact(k));
}

__device__ __forceinline__ u32 mul_hi(u32 a, u32 b) {
    return (u32)(((u64)a * (u64)b) >> 32);
}
__device__ __forceinline__ u64 philox_word(u64 seed, u64 transition, u64 block_index) {
    u32 c0=(u32)transition, c1=(u32)(transition>>32);
    u32 c2=(u32)block_index, c3=0x4A524232U;
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
__device__ __forceinline__ Ball dyadic_word(u64 word, int bits, int upper) {
    if (bits < 1 || bits > 64) return invalid_ball();
    Ball result = exact_double(0.0);
    if (bits <= 32) {
        result = ball_scale(exact_double((double)(u32)word), pow2_exact(-bits));
    } else {
        u32 hi=(u32)(word>>32), lo=(u32)word;
        result = ball_add(
            ball_scale(exact_double((double)hi), pow2_exact(32-bits)),
            ball_scale(exact_double((double)lo), pow2_exact(-bits)));
    }
    if (upper) result = ball_add(result, exact_double(pow2_exact(-bits)));
    return result;
}
__device__ __forceinline__ Ball dyadic128(u64 first, u64 second, int upper) {
    Ball result = dyadic_word(first, 64, 0);
    Ball tail = ball_scale(dyadic_word(second, 64, 0), pow2_exact(-64));
    result = ball_add(result, tail);
    if (upper) result = ball_add(result, exact_double(pow2_exact(-128)));
    return result;
}

__device__ __forceinline__ Ball exact_affine_coordinate(Ball value) {
    // The transition and rounding-boundary inputs are exact twofold values.
    // Multiplication by two and TwoSum with -1 retain that exactness.
    if (!value.ok || value.r != 0.0) return invalid_ball();
    DD two = {2.0,0.0};
    DD minus_one = {-1.0,0.0};
    return exact_dd(dd_add_raw(dd_mul_raw(value.c,two),minus_one));
}
__device__ __forceinline__ Ball legendre_next(
    Ball previous, Ball current, Ball z, int n, double* maximum_local_error
) {
    if (!previous.ok || !current.ok || !z.ok || z.r!=0.0 ||
        !finite_d(*maximum_local_error) || *maximum_local_error<0.0 ||
        ball_lower(z)<-1.0 || ball_upper(z)>1.0) return invalid_ball();
    double zv=dd_rn(z.c);
    if (z.r==0.0 && zv==1.0 && z.c.lo==0.0) return exact_double(1.0);
    if (z.r==0.0 && zv==-1.0 && z.c.lo==0.0)
        return exact_double((n+1)&1 ? -1.0 : 1.0);
    DD first_scale={(double)(2*n+1),0.0}, second_scale={(double)n,0.0};
    DD divisor={(double)(n+1),0.0};
    DD first=dd_mul_raw(dd_mul_raw(z.c,current.c),first_scale);
    DD second=dd_mul_raw(previous.c,second_scale);
    DD numerator=dd_sub_raw(first,second);
    DD centre=dd_div_raw(numerator,divisor);
    if (!finite_d(centre.hi) || !finite_d(centre.lo)) return invalid_ball();

    // A posteriori local defect: treat every DD centre as an exact real and
    // enclose the unnormalised Bonnet residual with Ball operations,
    //   R_n=(n+1)p~_(n+1)-(2n+1)z p~_n+n p~_(n-1).
    // Then |eps_n|<=|R_n|/(n+1) for the post-division perturbation used by
    // Proposition 5.  This avoids interval division inside every recurrence.
    Ball exact_previous=exact_dd(previous.c), exact_current=exact_dd(current.c);
    Ball exact_z=exact_dd(z.c);
    Ball residual=ball_add(
        ball_sub(
            ball_scale(exact_dd(centre),(double)(n+1)),
            ball_scale(ball_mul(exact_z,exact_current),(double)(2*n+1))),
        ball_scale(exact_previous,(double)n));
    if (!residual.ok) return invalid_ball();
    double residual_bound=dmax(
        dabs(ball_lower(residual)),dabs(ball_upper(residual)));
    double local_error_bound=__ddiv_ru(residual_bound,(double)(n+1));
    *maximum_local_error=dmax(*maximum_local_error,local_error_bound);

    // Johansson--Mezzarobba, Prop. 5 (arXiv:1802.03948, section 3): for
    // exact z in [-1,1], a Bonnet sequence with |eps_k|<=eps_bar obeys
    // |p~_N-P_N(z)| <= (N+1)(N+2) eps_bar / 4.  Here N=n+1.
    double output_degree=(double)(n+1);
    double factor=__dmul_ru(
        __dmul_ru(__dadd_ru(output_degree,1.0),__dadd_ru(output_degree,2.0)),
        0.25);
    double radius=__dmul_ru(factor,*maximum_local_error);
    Ball out={centre,radius,finite_d(radius)};
    return out;
}

struct Tail { double cdf; double density; double conormal; int ok; };
__device__ __forceinline__ Tail tail_bounds_from_powers(
    int m, Ball omitted_decay, Ball q_to_m
) {
    Tail bad = {0.0,0.0,0.0,0};
    if (!omitted_decay.ok || !q_to_m.ok) return bad;
    double q = ball_upper(q_to_m), d = ball_upper(omitted_decay);
    double denominator = __dsub_rd(1.0, q);
    if (!(denominator > 0.0)) return bad;
    Tail out;
    out.cdf = __ddiv_ru(d, denominator);
    double density_factor = __ddiv_ru((double)(2*m+3), (double)(2*m+1));
    double density_ratio = __dmul_ru(density_factor, q);
    double density_denominator = __dsub_rd(1.0, density_ratio);
    double conormal_factor = __dmul_ru(density_factor,
        __ddiv_ru((double)(m+1),(double)m));
    double conormal_ratio = __dmul_ru(conormal_factor, q);
    double conormal_denominator = __dsub_rd(1.0, conormal_ratio);
    if (!(density_denominator > 0.0) || !(conormal_denominator > 0.0)) return bad;
    out.density = __ddiv_ru(__dmul_ru((double)(2*m+1),d), density_denominator);
    out.conormal = __ddiv_ru(
        __dmul_ru(__dmul_ru((double)m,(double)(2*m+1)),d), conormal_denominator);
    out.ok = finite_d(out.cdf) && finite_d(out.density) && finite_d(out.conormal);
    return out;
}
__device__ __forceinline__ Tail tail_bounds(int m, Ball exposure) {
    Tail bad = {0.0,0.0,0.0,0};
    Ball decay = exp24(ball_scale(exposure, -(double)(m*(m+1))));
    Ball step = exp24(ball_scale(exposure, -(double)(2*(m+1))));
    if (!decay.ok || !step.ok) return bad;
    double q = ball_upper(step), d = ball_upper(decay);
    double denominator = __dsub_rd(1.0, q);
    if (!(denominator > 0.0)) return bad;
    Tail out;
    out.cdf = __ddiv_ru(d, denominator);
    double density_factor = __ddiv_ru((double)(2*m+3), (double)(2*m+1));
    double density_ratio = __dmul_ru(density_factor, q);
    double density_denominator = __dsub_rd(1.0, density_ratio);
    double conormal_factor = __dmul_ru(density_factor,
        __ddiv_ru((double)(m+1),(double)m));
    double conormal_ratio = __dmul_ru(conormal_factor, q);
    double conormal_denominator = __dsub_rd(1.0, conormal_ratio);
    if (!(density_denominator > 0.0) || !(conormal_denominator > 0.0)) return bad;
    out.density = __ddiv_ru(__dmul_ru((double)(2*m+1),d), density_denominator);
    out.conormal = __ddiv_ru(
        __dmul_ru(__dmul_ru((double)m,(double)(2*m+1)),d), conormal_denominator);
    out.ok = finite_d(out.cdf) && finite_d(out.density) && finite_d(out.conormal);
    return out;
}

struct SpectralState {
    Ball previous; Ball current; Ball cdf; Ball density; Ball conormal;
    double maximum_legendre_local_error;
};
__device__ __forceinline__ SpectralState state_init(Ball y, Ball zy, int full) {
    SpectralState s;
    s.previous=exact_double(1.0); s.current=zy; s.cdf=y;
    s.density=exact_double(1.0); s.conormal=exact_double(0.0);
    s.maximum_legendre_local_error=0.0;
    return s;
}
__device__ __forceinline__ int state_ok(SpectralState s) {
    return s.previous.ok && s.current.ok && s.cdf.ok && s.density.ok &&
        s.conormal.ok && finite_d(s.maximum_legendre_local_error);
}

__device__ __forceinline__ int try_certificate(
    Ball cdf_lower_boundary, Ball cdf_upper_boundary,
    Ball density, Ball conormal, Tail tail,
    Ball uniform_lower, Ball uniform_upper,
    double* target_value, double* target_lo, double* target_hi, int* reason
) {
    if (!tail.ok || !cdf_lower_boundary.ok || !cdf_upper_boundary.ok ||
        !density.ok || !conormal.ok || !uniform_lower.ok || !uniform_upper.ok) {
        *reason=7; return 0;
    }
    Ball left = ball_expand(cdf_lower_boundary, tail.cdf);
    Ball right = ball_expand(cdf_upper_boundary, tail.cdf);
    if (!ball_strict_less(left,uniform_lower) ||
        !ball_strict_less(uniform_upper,right)) {
        *target_value=dd_rn(left.c);
        *target_lo=ball_lower(left); *target_hi=ball_upper(left);
        // If the complete uniform interval is below the candidate cell's
        // lower-bound CDF, monotonicity permits only smaller candidates.  The
        // symmetric comparison permits only larger candidates.  Otherwise
        // the balls genuinely overlap and no neighbour direction is proved.
        if (ball_strict_less(uniform_upper,left)) *reason=23;
        else if (ball_strict_less(right,uniform_lower)) *reason=24;
        else *reason=1;
        return 0;
    }
    Ball k = ball_expand(density, tail.density);
    Ball g = ball_expand(conormal, tail.conormal);
    if (!k.ok || !(ball_lower(k)>0.0)) { *reason=2; return 0; }
    Ball quotient = ball_div(g,k);
    if (!quotient.ok) { *reason=7; return 0; }
    double candidate = dd_rn(quotient.c);
    Ball cell_lower,cell_upper;
    if (!finite_d(candidate) || !rounding_boundaries(candidate,&cell_lower,&cell_upper)) {
        *reason=11; return 0;
    }
    double qlo=ball_lower(quotient), qhi=ball_upper(quotient);
    if (!ball_strict_less(cell_lower,quotient) ||
        !ball_strict_less(quotient,cell_upper)) {
        *reason=3; return 0;
    }
    *target_value=candidate;
    *target_lo=dmin(qlo,candidate); *target_hi=dmax(qhi,candidate);
    *reason=0; return 1;
}

extern "C" __global__ void jacobi_rb_fused_selftest_v1(u64* output) {
    if (blockIdx.x || threadIdx.x) return;
    u64 mask=0;
    double half_ulp=0x1.0p-53, up=next_up(1.0);
    if (__dadd_rd(1.0,half_ulp)==1.0) mask|=1ULL<<0;
    if (__dadd_ru(1.0,half_ulp)==up) mask|=1ULL<<1;
    double a=__dadd_rn(1.0,0x1.0p-27), b=__dsub_rn(1.0,0x1.0p-27);
    if (__dmul_rd(a,b)<1.0) mask|=1ULL<<2;
    if (__dmul_ru(a,b)==1.0) mask|=1ULL<<3;
    if (__ddiv_rd(1.0,10.0)<__ddiv_ru(1.0,10.0)) mask|=1ULL<<4;
    if (__ddiv_ru(1.0,10.0)==next_up(__ddiv_rd(1.0,10.0))) mask|=1ULL<<5;
    DD s=two_sum(1.0,half_ulp);
    if (s.hi==1.0 && s.lo==half_ulp) mask|=1ULL<<6;
    DD p=two_prod(a,b);
    if (p.hi==1.0 && p.lo==-0x1.0p-54) mask|=1ULL<<7;
    Ball e=exp24(ball_neg(ln2_ball()));
    if (e.ok && ball_lower(e)<0.5 && ball_upper(e)>0.5) mask|=1ULL<<8;
    Ball lo,hi;
    if (rounding_boundaries(1.0,&lo,&hi) &&
        dd_down(lo.c)==0x1.fffffffffffffp-1 && dd_up(lo.c)==1.0) mask|=1ULL<<9;
    Ball tenth=ball_div(exact_double(1.0),exact_double(10.0));
    if (tenth.ok && ball_lower(tenth)<0.1 && ball_upper(tenth)>0.1) mask|=1ULL<<10;
    double tiny=__longlong_as_double(1ULL);
    if (__dmul_rn(tiny,1.0)==tiny &&
        __dadd_rn(tiny,tiny)==__longlong_as_double(2ULL)) mask|=1ULL<<11;
    if (__dadd_rn(__dmul_rn(a,b),-1.0)==0.0 &&
        __fma_rn(a,b,-1.0)==-0x1.0p-54) mask|=1ULL<<12;
    Ball under=exp24(exact_double(-0x1.0p+20));
    if (under.ok && ball_lower(under)<=0.0 &&
        ball_upper(under)==tiny) mask|=1ULL<<13;
    Ball sublo,subhi;
    if (!rounding_boundaries(__longlong_as_double(3ULL),&sublo,&subhi) &&
        !rounding_boundaries(0x1.0p-1022,&sublo,&subhi))
        mask|=1ULL<<14;
    Ball interval_num={{1.0,0.0},0.125,1};
    Ball interval_den={{2.0,0.0},0.25,1};
    Ball divided=ball_div(interval_num,interval_den);
    double exact_lo=__ddiv_rd(0.875,2.25);
    double exact_hi=__ddiv_ru(1.125,1.75);
    if (divided.ok && ball_lower(divided)<=exact_lo &&
        ball_upper(divided)>=exact_hi) mask|=1ULL<<15;
    double legendre_error=0.0;
    Ball legendre_z=exact_double(0.25);
    Ball legendre_p2=legendre_next(
        exact_double(1.0),legendre_z,legendre_z,1,&legendre_error);
    Ball legendre_p3=legendre_next(
        legendre_z,legendre_p2,legendre_z,2,&legendre_error);
    if (legendre_p3.ok && ball_lower(legendre_p3)<=-0x1.58p-2 &&
        ball_upper(legendre_p3)>=-0x1.58p-2) mask|=1ULL<<16;
    output[0]=mask;
}

extern "C" __global__ void jacobi_rb_fused_legendre_probe_v1(
    const double* z_values, const int* degrees,
    const double* injected_maximum_local_error, int count,
    double* centres, double* lowers, double* uppers, double* radii, u8* valid
) {
    int index=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    if (index>=count) return;
    double zvalue=z_values[index], maximum_error=injected_maximum_local_error[index];
    int degree=degrees[index];
    valid[index]=0; centres[index]=0.0; lowers[index]=0.0;
    uppers[index]=0.0; radii[index]=0.0;
    if (!finite_d(zvalue) || zvalue<-1.0 || zvalue>1.0 ||
        !finite_d(maximum_error) || maximum_error<0.0 ||
        degree<0 || degree>8192) return;
    Ball z=exact_double(zvalue), previous=exact_double(1.0), current=z;
    if (degree==0) current=previous;
    for (int n=1; n<degree; ++n) {
        Ball next=legendre_next(previous,current,z,n,&maximum_error);
        if (!next.ok) return;
        previous=current; current=next;
    }
    centres[index]=dd_rn(current.c);
    lowers[index]=ball_lower(current); uppers[index]=ball_upper(current);
    radii[index]=current.r;
    valid[index]=(u8)(current.ok && finite_d(lowers[index]) && finite_d(uppers[index]));
}

extern "C" __global__ void jacobi_rb_fused_authorize_v1(
    const double* x, const double* exposure, const u64* transition_ids,
    const u64* seed_pointer, const double* proposed_y,
    const u64* recorded_prefix, const int* recorded_bits, int prefix_kind,
    int count, int primary_cap, int strengthened_cap, int max_prefix_bits,
    double* later, double* target, double* quantile_lower, double* quantile_upper,
    double* target_lower, double* target_upper, int* modes_used, int* prefix_used,
    u8* certificate_codes, u8* authorized, u8* strengthened, u8* reasons
) {
    int index=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    if (index>=count) return;
    double xv=x[index], uv=exposure[index], yv=proposed_y[index];
    later[index]=xv; target[index]=0.0; quantile_lower[index]=xv; quantile_upper[index]=xv;
    target_lower[index]=0.0; target_upper[index]=0.0; modes_used[index]=0;
    prefix_used[index]=0; certificate_codes[index]=0; authorized[index]=0;
    strengthened[index]=0; reasons[index]=0;
    if (uv==0.0) return;
    if (!finite_d(xv)||!finite_d(uv)||!finite_d(yv)||xv<0.0||xv>1.0||uv<=0.0||yv<=0.0||yv>=1.0) {
        reasons[index]=6; return;
    }
    Ball ycell_lo,ycell_hi;
    if (!rounding_boundaries(yv,&ycell_lo,&ycell_hi)) { reasons[index]=11; return; }
    Ball xball=exact_double(xv), uball=exact_double(uv), yball=exact_double(yv);
    Ball zx=exact_affine_coordinate(xball);
    Ball zylo=exact_affine_coordinate(ycell_lo);
    Ball zyhi=exact_affine_coordinate(ycell_hi);
    Ball zyt=exact_affine_coordinate(yball);
    Ball px_prev=exact_double(1.0), px_cur=zx;
    double px_maximum_legendre_local_error=0.0;
    SpectralState slo=state_init(ycell_lo,zylo,0);
    SpectralState shi=state_init(ycell_hi,zyhi,0);
    SpectralState st=state_init(yball,zyt,1);
    u64 first,second=0;
    int bits;
    if (prefix_kind) { first=recorded_prefix[index]; bits=recorded_bits[index]; }
    else { first=philox_word(seed_pointer[0],transition_ids[index],0); bits=64;
           if (max_prefix_bits>=128) second=philox_word(seed_pointer[0],transition_ids[index],1); }
    Ball uniform_lo=dyadic_word(first,bits,0), uniform_hi=dyadic_word(first,bits,1);
    int last_reason=8;
    int first_bucket_reason=0;
    int last_mode=1;
    int cap=strengthened_cap;
    // q=exp(-2u) is certified once.  Exact spectral decays then obey
    // d_1=q and d_{n+1}=d_n*q^{n+1}; no per-mode transcendental is used.
    Ball qbase=exp24(ball_scale(uball,-2.0));
    Ball qpower=qbase, decay=qbase;
    if (!qbase.ok) { reasons[index]=4; return; }
    for (int n=1; n<cap; ++n) {
        last_mode=n+1;
        Ball lo_next=legendre_next(slo.previous,slo.current,zylo,n,
            &slo.maximum_legendre_local_error);
        Ball hi_next=legendre_next(shi.previous,shi.current,zyhi,n,
            &shi.maximum_legendre_local_error);
        Ball t_next=legendre_next(st.previous,st.current,zyt,n,
            &st.maximum_legendre_local_error);
        if (!lo_next.ok) { last_reason=12; break; }
        if (!hi_next.ok) { last_reason=13; break; }
        if (!t_next.ok) { last_reason=14; break; }
        if (!decay.ok) { last_reason=15; break; }
        Ball coefficient=ball_scale(decay,(double)(2*n+1));
        if (!coefficient.ok) { last_reason=16; break; }
        slo.cdf=ball_add(slo.cdf,ball_scale(ball_mul(ball_mul(decay,px_cur),
            ball_sub(lo_next,slo.previous)),0.5));
        if (!slo.cdf.ok) { last_reason=17; break; }
        shi.cdf=ball_add(shi.cdf,ball_scale(ball_mul(ball_mul(decay,px_cur),
            ball_sub(hi_next,shi.previous)),0.5));
        if (!shi.cdf.ok) { last_reason=18; break; }
        st.cdf=ball_add(st.cdf,ball_scale(ball_mul(ball_mul(decay,px_cur),
            ball_sub(t_next,st.previous)),0.5));
        if (!st.cdf.ok) { last_reason=18; break; }
        st.density=ball_add(st.density,ball_mul(ball_mul(coefficient,px_cur),st.current));
        if (!st.density.ok) { last_reason=19; break; }
        Ball basis=ball_scale(
            ball_sub(st.previous,ball_mul(zyt,st.current)),
            __dmul_rn(0.5,(double)n));
        if (!basis.ok) { last_reason=20; break; }
        st.conormal=ball_add(st.conormal,ball_mul(ball_mul(coefficient,px_cur),basis));
        if (!st.conormal.ok) { last_reason=21; break; }
        Ball px_next=legendre_next(px_prev,px_cur,zx,n,
            &px_maximum_legendre_local_error);
        if (!px_next.ok) { last_reason=22; break; }
        slo.previous=slo.current; slo.current=lo_next;
        shi.previous=shi.current; shi.current=hi_next;
        st.previous=st.current; st.current=t_next;
        px_prev=px_cur; px_cur=px_next;
        int m=n+1;
        qpower=ball_mul(qpower,qbase);       // q^m
        decay=ball_mul(decay,qpower);        // q^{m(m+1)/2}
        int bucket=(m>=128 && (m%16)==0) || m==primary_cap || m==strengthened_cap;
        if (!bucket) continue;
        if (!state_ok(slo)||!state_ok(shi)||!state_ok(st)||!px_cur.ok) {
            last_reason=7; break;
        }
        if (!prefix_kind && m>primary_cap && bits==64 && max_prefix_bits>=128) {
            uniform_lo=dyadic128(first,second,0); uniform_hi=dyadic128(first,second,1); bits=128;
        }
        Tail tail=tail_bounds_from_powers(m,decay,qpower);
        double zvalue=0.0,zlo=0.0,zhi=0.0;
        if (try_certificate(slo.cdf,shi.cdf,st.density,st.conormal,tail,
              uniform_lo,uniform_hi,&zvalue,&zlo,&zhi,&last_reason)) {
            later[index]=yv; target[index]=zvalue;
            quantile_lower[index]=dd_down(ycell_lo.c);
            quantile_upper[index]=dd_up(ycell_hi.c);
            target_lower[index]=zlo; target_upper[index]=zhi;
            modes_used[index]=m; prefix_used[index]=bits;
            certificate_codes[index]=15; authorized[index]=1;
            strengthened[index]=(m>primary_cap || bits>64) ? 1 : 0;
            reasons[index]=0; return;
        }
        if (!first_bucket_reason) first_bucket_reason=last_reason;
        if (tail.ok && tail.cdf<0x1.0p-79) {
            Ball full_cdf=ball_expand(st.cdf,tail.cdf);
            Ball full_density=ball_expand(st.density,tail.density);
            if (full_cdf.ok && full_density.ok && ball_lower(full_density)>0.0) {
                Ball uniform_mid=ball_scale(ball_add(uniform_lo,uniform_hi),0.5);
                Ball delta=ball_div(ball_sub(uniform_mid,full_cdf),full_density);
                Ball repaired=ball_add(yball,delta);
                double repair_value=dd_rn(repaired.c);
                if (repaired.ok && finite_d(repair_value) &&
                    repair_value>0.0 && repair_value<1.0) later[index]=repair_value;
            }
        }
        // Failure diagnostics are never authorizing, but retaining the last
        // CDF/uniform triple makes arithmetic faults auditable in GPU tests.
        target[index]=zvalue; target_lower[index]=zlo; target_upper[index]=zhi;
        if (m>=primary_cap) strengthened[index]=1;
    }
    modes_used[index]=last_mode; prefix_used[index]=bits;
    int reported=(last_reason==23 || last_reason==24)
        ? last_reason : (first_bucket_reason ? first_bucket_reason : last_reason);
    if (reported>11 && reported!=23 && reported!=24) reported=7;
    reasons[index]=(u8)(reported ? reported : 8);
}
"""

SOURCE_SHA256 = hashlib.sha256(_CUDA_SOURCE.encode("utf-8")).hexdigest()
CONSTANTS_SHA256 = hashlib.sha256(
    "|".join(
        LN2_DD_HEX + EXP_DD_HI_HEX + EXP_DD_LO_HEX + EXP_DD_RAD_HEX
    ).encode("ascii")
).hexdigest()


def verify_fused_device_constants() -> dict[str, Any]:
    """Exact host proof that every frozen DD device constant encloses truth."""

    errors: list[str] = []
    if not (
        len(EXP_DD_HI_HEX) == len(EXP_DD_LO_HEX)
        == len(EXP_DD_RAD_HEX) == 25
    ):
        errors.append("exp24 constant table length")
    for degree, (hi_hex, lo_hex, radius_hex) in enumerate(
        zip(EXP_DD_HI_HEX, EXP_DD_LO_HEX, EXP_DD_RAD_HEX, strict=True)
    ):
        hi = Fraction.from_float(float.fromhex(hi_hex))
        lo = Fraction.from_float(float.fromhex(lo_hex))
        radius = Fraction.from_float(float.fromhex(radius_hex))
        if abs(Fraction(1, math.factorial(degree)) - hi - lo) > radius:
            errors.append(f"exp24 coefficient {degree}")
        if hi_hex not in _CUDA_SOURCE or lo_hex not in _CUDA_SOURCE or radius_hex not in _CUDA_SOURCE:
            errors.append(f"source binding {degree}")
    ln_hi, ln_lo, ln_radius = (
        Fraction.from_float(float.fromhex(item)) for item in LN2_DD_HEX
    )
    ln_center = ln_hi + ln_lo
    if not (
        ln_center - ln_radius <= LN2_LOWER
        and LN2_UPPER <= ln_center + ln_radius
    ):
        errors.append("ln2 DD enclosure")
    if not Fraction.from_float(float.fromhex("0x1.0p-116")) >= EXP24_REMAINDER_BOUND:
        errors.append("degree-24 remainder")
    if any(item not in _CUDA_SOURCE for item in LN2_DD_HEX):
        errors.append("ln2 source binding")
    legendre_markers = (
        "Johansson--Mezzarobba, Prop. 5 (arXiv:1802.03948, section 3)",
        "z.r!=0.0",
        "double local_error_bound=__ddiv_ru(residual_bound,(double)(n+1));",
        "*maximum_local_error=dmax(*maximum_local_error,local_error_bound);",
        "__dadd_ru(output_degree,1.0),__dadd_ru(output_degree,2.0)",
    )
    missing_legendre_markers = [
        marker for marker in legendre_markers if marker not in _CUDA_SOURCE
    ]
    if missing_legendre_markers:
        errors.append("Legendre Proposition 5 source binding")
    return {
        "passed": int(not errors),
        "fused_constant_fingerprint": CONSTANTS_SHA256,
        "source_sha256": SOURCE_SHA256,
        "coefficient_count": 25,
        "remainder_hex": "0x1.0p-116",
        "legendre_recurrence_certificate_pass": int(
            not missing_legendre_markers
        ),
        "legendre_recurrence_theorem": LEGENDRE_RECURRENCE_THEOREM,
        "legendre_recurrence_theorem_uri": LEGENDRE_RECURRENCE_THEOREM_URI,
        "legendre_recurrence_error_factor": LEGENDRE_RECURRENCE_ERROR_FACTOR,
        "legendre_recurrence_hypotheses": {
            "exact_p0": True,
            "exact_p1_equals_z": True,
            "exact_zero_radius_z_in_closed_unit_interval": True,
            "post_division_local_defect_enclosed": True,
        },
        "errors": errors,
    }


@dataclass(frozen=True)
class FusedCudaBundle:
    authorizer: Any
    legendre_probe: Any
    selftest_mask: int
    binary_sha256: str
    source_sha256: str = SOURCE_SHA256


@dataclass(frozen=True)
class FusedCudaLaunch:
    later: Tensor
    target: Tensor
    quantile_lower: Tensor
    quantile_upper: Tensor
    target_lower: Tensor
    target_upper: Tensor
    modes_used: Tensor
    prefix_bits: Tensor
    certificate_codes: Tensor
    authorized_mask: Tensor
    strengthened_mask: Tensor
    fallback_reason_codes: Tensor
    maximum_launch_lanes: int
    launch_count: int
    bundle: FusedCudaBundle


_CACHE: dict[tuple[int, int, int, tuple[str, ...]], FusedCudaBundle] = {}
_LOCK = threading.Lock()


def _compile(device: Any, compile_flags: tuple[str, ...]) -> FusedCudaBundle:
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    index = torch.device(device).index
    if index is None:
        index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(index)
    key = (int(index), int(properties.major), int(properties.minor), tuple(compile_flags))
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        from torch.utils import cpp_extension
        from torch.cuda._utils import _cuda_load_module, _nvrtc_compile

        previous_cuda_home = cpp_extension.CUDA_HOME
        if previous_cuda_home is None:
            cpp_extension.CUDA_HOME = str(Path(torch.__file__).resolve().parent)
        try:
            with torch.cuda.device(index):
                binary, lowered = _nvrtc_compile(
                    _CUDA_SOURCE,
                    FUSED_KERNEL_NAME,
                    compute_capability=f"{properties.major}{properties.minor}",
                    nvcc_options=list(compile_flags),
                )
                loaded = _cuda_load_module(
                    binary,
                    [lowered, SELFTEST_KERNEL_NAME, LEGENDRE_PROBE_KERNEL_NAME],
                )
                selftest_output = torch.zeros(1, dtype=torch.uint64, device=device)
                loaded[SELFTEST_KERNEL_NAME](
                    grid=(1, 1, 1), block=(1, 1, 1), args=[selftest_output],
                    stream=torch.cuda.current_stream(device),
                )
                torch.cuda.synchronize(device)
                observed = int(selftest_output.item())
                if observed != REQUIRED_SELFTEST_MASK:
                    raise RuntimeError(
                        "fused CUDA arithmetic self-test failed: "
                        f"observed=0x{observed:x}, required=0x{REQUIRED_SELFTEST_MASK:x}"
                    )
                result = FusedCudaBundle(
                    authorizer=loaded[lowered],
                    legendre_probe=loaded[LEGENDRE_PROBE_KERNEL_NAME],
                    selftest_mask=observed,
                    binary_sha256=hashlib.sha256(binary).hexdigest(),
                )
        finally:
            cpp_extension.CUDA_HOME = previous_cuda_home
        _CACHE[key] = result
        return result


def probe_fused_cuda_authorizer(
    device: Any,
    *,
    compile_flags: tuple[str, ...],
    cpu_preflight: Mapping[str, Any],
) -> tuple[FusedCudaBundle | None, dict[str, Any]]:
    """Compile and self-test the kernel, returning a fail-closed report."""

    report: dict[str, Any] = {
        "fused_cuda_version": FUSED_CUDA_VERSION,
        "fused_source_sha256": SOURCE_SHA256,
        "required_selftest_mask": REQUIRED_SELFTEST_MASK,
        "cpu_arithmetic_preflight": dict(cpu_preflight),
        "double_double_interval_algebra_pass": int(
            cpu_preflight.get("double_double_interval_algebra_pass", 0)
        ),
        "certified_exponential_pass": int(
            cpu_preflight.get("certified_exponential_pass", 0)
        ),
    }
    constant_preflight = verify_fused_device_constants()
    report["fused_device_constant_preflight"] = constant_preflight
    report["fused_device_constants_pass"] = int(constant_preflight["passed"])
    report["legendre_recurrence_certificate_pass"] = int(
        constant_preflight["legendre_recurrence_certificate_pass"]
    )
    report["legendre_recurrence_theorem"] = constant_preflight[
        "legendre_recurrence_theorem"
    ]
    report["legendre_recurrence_error_factor"] = constant_preflight[
        "legendre_recurrence_error_factor"
    ]
    report["fused_constant_fingerprint"] = CONSTANTS_SHA256
    if not (
        report["double_double_interval_algebra_pass"]
        and report["certified_exponential_pass"]
        and report["fused_device_constants_pass"]
        and report["legendre_recurrence_certificate_pass"]
    ):
        report.update(
            fused_cuda_authorizer_available=False,
            fused_cuda_authorizer_unavailable_reason="CPU certificate arithmetic preflight failed",
        )
        return None, report
    try:
        bundle = _compile(device, compile_flags)
    except Exception as exc:
        report.update(
            fused_cuda_authorizer_available=False,
            fused_cuda_authorizer_unavailable_reason=f"{type(exc).__name__}: {exc}",
        )
        return None, report
    report.update(
        fused_cuda_authorizer_available=True,
        fused_cuda_authorizer_unavailable_reason=None,
        arithmetic_selftest_mask=int(bundle.selftest_mask),
        arithmetic_selftest_pass=True,
        fused_binary_sha256=bundle.binary_sha256,
    )
    return bundle, report


def launch_fused_cuda_authorizer(
    bundle: FusedCudaBundle,
    head_fraction: Tensor,
    exposure: Tensor,
    transition_ids: Tensor,
    proposed_y: Tensor,
    *,
    seed: int,
    threads_per_block: int,
    max_prefix_bits: int,
    recorded_prefix_numerators: Tensor | None = None,
    recorded_prefix_bits: Tensor | None = None,
    _primary_cap: int = 4096,
    _strengthened_cap: int = 8192,
) -> FusedCudaLaunch:
    """Run the authorizer.  Every output mask/code is written on the device."""

    count = int(head_fraction.numel())
    if count > 4096:
        raise ValueError("fused CUDA authorizer launch exceeds the 4096-lane cap")
    device = head_fraction.device
    zeros_u64 = torch.zeros(count, dtype=torch.uint64, device=device)
    zeros_i32 = torch.zeros(count, dtype=torch.int32, device=device)
    prefix_kind = int(recorded_prefix_numerators is not None)
    prefix_values = recorded_prefix_numerators if prefix_kind else zeros_u64
    prefix_lengths = recorded_prefix_bits if prefix_kind else zeros_i32
    later = torch.empty_like(head_fraction)
    target = torch.empty_like(head_fraction)
    qlo = torch.empty_like(head_fraction)
    qhi = torch.empty_like(head_fraction)
    zlo = torch.empty_like(head_fraction)
    zhi = torch.empty_like(head_fraction)
    modes = torch.empty(count, dtype=torch.int32, device=device)
    bits = torch.empty(count, dtype=torch.int32, device=device)
    codes = torch.empty(count, dtype=torch.uint8, device=device)
    authorized = torch.empty(count, dtype=torch.uint8, device=device)
    strengthened = torch.empty(count, dtype=torch.uint8, device=device)
    reasons = torch.empty(count, dtype=torch.uint8, device=device)
    if count:
        seed_tensor = torch.tensor([int(seed)], dtype=torch.uint64, device=device)
        threads = int(threads_per_block)
        bundle.authorizer(
            grid=((count + threads - 1) // threads, 1, 1),
            block=(threads, 1, 1),
            args=[
                head_fraction, exposure, transition_ids, seed_tensor, proposed_y,
                prefix_values, prefix_lengths, prefix_kind, count, int(_primary_cap),
                int(_strengthened_cap),
                int(max_prefix_bits), later, target, qlo, qhi, zlo, zhi, modes, bits,
                codes, authorized, strengthened, reasons,
            ],
            stream=torch.cuda.current_stream(device),
        )
    return FusedCudaLaunch(
        later=later,
        target=target,
        quantile_lower=qlo,
        quantile_upper=qhi,
        target_lower=zlo,
        target_upper=zhi,
        modes_used=modes,
        prefix_bits=bits,
        certificate_codes=codes,
        authorized_mask=authorized.bool(),
        strengthened_mask=strengthened.bool(),
        fallback_reason_codes=reasons,
        maximum_launch_lanes=count if count else 0,
        launch_count=1 if count else 0,
        bundle=bundle,
    )


def probe_fused_legendre_enclosures(
    bundle: FusedCudaBundle,
    z_values: Tensor,
    degrees: Tensor,
    *,
    injected_maximum_local_error: Tensor | None = None,
    threads_per_block: int = 128,
) -> Mapping[str, Tensor]:
    """Exercise the theorem-backed Legendre enclosure kernel for audit tests."""

    if z_values.dtype != torch.float64 or not z_values.is_cuda:
        raise ValueError("z_values must be a CUDA float64 tensor")
    if degrees.dtype != torch.int32 or degrees.device != z_values.device:
        raise ValueError("degrees must be a same-device int32 tensor")
    if z_values.shape != degrees.shape:
        raise ValueError("z_values and degrees must have identical shapes")
    flat_z = z_values.contiguous().reshape(-1)
    flat_degrees = degrees.contiguous().reshape(-1)
    count = int(flat_z.numel())
    if injected_maximum_local_error is None:
        injected = torch.zeros_like(flat_z)
    else:
        if (
            injected_maximum_local_error.dtype != torch.float64
            or injected_maximum_local_error.device != z_values.device
            or injected_maximum_local_error.shape != z_values.shape
        ):
            raise ValueError(
                "injected_maximum_local_error must match z_values"
            )
        injected = injected_maximum_local_error.contiguous().reshape(-1)
    centres = torch.empty_like(flat_z)
    lowers = torch.empty_like(flat_z)
    uppers = torch.empty_like(flat_z)
    radii = torch.empty_like(flat_z)
    valid = torch.empty(count, dtype=torch.uint8, device=z_values.device)
    if count:
        threads = int(threads_per_block)
        bundle.legendre_probe(
            grid=((count + threads - 1) // threads, 1, 1),
            block=(threads, 1, 1),
            args=[
                flat_z, flat_degrees, injected, count,
                centres, lowers, uppers, radii, valid,
            ],
            stream=torch.cuda.current_stream(z_values.device),
        )
    shape = z_values.shape
    return {
        "centre": centres.reshape(shape),
        "lower": lowers.reshape(shape),
        "upper": uppers.reshape(shape),
        "radius": radii.reshape(shape),
        "valid": valid.bool().reshape(shape),
    }


def launch_fused_cuda_authorizer_with_neighbors(
    bundle: FusedCudaBundle,
    head_fraction: Tensor,
    exposure: Tensor,
    transition_ids: Tensor,
    proposed_y: Tensor,
    *,
    seed: int,
    threads_per_block: int,
    max_prefix_bits: int,
    recorded_prefix_numerators: Tensor | None = None,
    recorded_prefix_bits: Tensor | None = None,
    maximum_neighbor_ulps: int = 4,
    force_strengthened: bool = False,
) -> FusedCudaLaunch:
    """Primary/strengthened certificate with a bounded candidate lattice.

    Candidate repair is not a numerical correction: each neighbouring float
    receives its own exact rounding-cell certificate.  Resolved lanes are
    replaced by NaN in later launches and therefore do no spectral work.
    """

    count = int(head_fraction.numel())
    device = head_fraction.device
    if count and (
        not bool(torch.isfinite(exposure).all().item())
        or bool((exposure < 0.0).any().item())
    ):
        raise ValueError("exposure must be finite and nonnegative")
    combined = {
        "later": head_fraction.clone(),
        "target": torch.zeros_like(head_fraction),
        "quantile_lower": head_fraction.clone(),
        "quantile_upper": head_fraction.clone(),
        "target_lower": torch.zeros_like(head_fraction),
        "target_upper": torch.zeros_like(head_fraction),
        "modes_used": torch.zeros(count, dtype=torch.int32, device=device),
        "prefix_bits": torch.zeros(count, dtype=torch.int32, device=device),
        "certificate_codes": torch.zeros(count, dtype=torch.uint8, device=device),
        "authorized_mask": torch.zeros(count, dtype=torch.bool, device=device),
        "strengthened_mask": torch.zeros(count, dtype=torch.bool, device=device),
        "fallback_reason_codes": torch.zeros(count, dtype=torch.uint8, device=device),
    }
    active = exposure > 0.0
    negative_infinity = torch.full_like(proposed_y, float("-inf"))
    positive_infinity = torch.full_like(proposed_y, float("inf"))
    base_candidate = proposed_y
    maximum_cuda_launch_lanes = 0
    cuda_authorizer_launch_count = 0

    def record_launch(attempt: FusedCudaLaunch) -> None:
        nonlocal maximum_cuda_launch_lanes, cuda_authorizer_launch_count
        maximum_cuda_launch_lanes = max(
            maximum_cuda_launch_lanes, int(attempt.maximum_launch_lanes)
        )
        cuda_authorizer_launch_count += int(attempt.launch_count)

    def candidate_at(offset: int) -> Tensor:
        value = base_candidate
        direction = negative_infinity if offset < 0 else positive_infinity
        for _ in range(abs(int(offset))):
            value = torch.nextafter(value, direction)
        unresolved = active & ~combined["authorized_mask"]
        return torch.where(unresolved, value, torch.full_like(value, float("nan")))

    def absorb(attempt: FusedCudaLaunch, *, strengthened_pass: bool) -> None:
        unresolved = active & ~combined["authorized_mask"]
        take = unresolved & attempt.authorized_mask
        combined["modes_used"] = torch.where(
            unresolved, attempt.modes_used, combined["modes_used"]
        )
        combined["prefix_bits"] = torch.where(
            unresolved, attempt.prefix_bits, combined["prefix_bits"]
        )
        for name in (
            "later", "target", "quantile_lower", "quantile_upper",
            "target_lower", "target_upper",
            "certificate_codes",
        ):
            combined[name] = torch.where(take, getattr(attempt, name), combined[name])
        combined["strengthened_mask"] |= take & (
            attempt.strengthened_mask | bool(strengthened_pass)
        )
        # The last unresolved attempt owns the precise device reason.
        combined["fallback_reason_codes"] = torch.where(
            unresolved & ~take,
            attempt.fallback_reason_codes,
            combined["fallback_reason_codes"],
        )
        combined["fallback_reason_codes"] = torch.where(
            take, torch.zeros_like(combined["fallback_reason_codes"]),
            combined["fallback_reason_codes"],
        )
        combined["authorized_mask"] |= take

    def adopt_newton_suggestion(attempt: FusedCudaLaunch) -> bool:
        nonlocal base_candidate
        unresolved = active & ~combined["authorized_mask"]
        suggested = (
            unresolved
            & (
                (attempt.fallback_reason_codes == 1)
                | (attempt.fallback_reason_codes == _CDF_CANDIDATE_TOO_HIGH)
                | (attempt.fallback_reason_codes == _CDF_CANDIDATE_TOO_LOW)
            )
            & torch.isfinite(attempt.later)
            & (attempt.later > 0.0)
            & (attempt.later < 1.0)
            & (attempt.later != base_candidate)
        )
        base_candidate = torch.where(suggested, attempt.later, base_candidate)
        return bool(suggested.any().item())

    def run_lattice(
        lattice_offsets: list[int],
        *,
        strengthened_pass: bool,
        stage_max_prefix_bits: int,
        stage_primary_cap: int,
        stage_strengthened_cap: int,
        eligible_mask: Tensor | None = None,
    ) -> None:
        """Evaluate an ordered neighbour lattice in watchdog-safe batches."""

        unresolved = active & ~combined["authorized_mask"]
        if eligible_mask is not None:
            unresolved &= eligible_mask
        if not lattice_offsets or not bool(unresolved.any().item()):
            return
        copies = len(lattice_offsets)
        unresolved_indices = torch.nonzero(unresolved, as_tuple=False).reshape(-1)
        # The production watchdog cap is 4096 kernel lanes.  A neighbour cell
        # is a proof lane, but we nevertheless enforce the same hard cap by
        # compacting unresolved transitions and processing at most 4096/copies
        # of them per expanded launch.
        rows_per_launch = max(1, 4096 // copies)
        for chunk_start in range(0, int(unresolved_indices.numel()), rows_per_launch):
            indices = unresolved_indices[chunk_start:chunk_start + rows_per_launch]
            local_count = int(indices.numel())
            candidates = torch.cat(
                [candidate_at(offset).index_select(0, indices) for offset in lattice_offsets],
                dim=0,
            )
            repeated_prefix = (
                None
                if recorded_prefix_numerators is None
                else recorded_prefix_numerators.index_select(0, indices).repeat(copies)
            )
            repeated_bits = (
                None
                if recorded_prefix_bits is None
                else recorded_prefix_bits.index_select(0, indices).repeat(copies)
            )
            expanded = launch_fused_cuda_authorizer(
                bundle,
                head_fraction.index_select(0, indices).repeat(copies),
                exposure.index_select(0, indices).repeat(copies),
                transition_ids.index_select(0, indices).repeat(copies),
                candidates,
                seed=seed,
                threads_per_block=threads_per_block,
                max_prefix_bits=stage_max_prefix_bits,
                recorded_prefix_numerators=repeated_prefix,
                recorded_prefix_bits=repeated_bits,
                _primary_cap=stage_primary_cap,
                _strengthened_cap=stage_strengthened_cap,
            )
            record_launch(expanded)
            for lattice_index in range(copies):
                start = lattice_index * local_count
                end = start + local_count
                still_unresolved = ~combined["authorized_mask"].index_select(0, indices)
                attempt_authorized = expanded.authorized_mask[start:end]
                take = still_unresolved & attempt_authorized
                for name, values in (
                    ("modes_used", expanded.modes_used),
                    ("prefix_bits", expanded.prefix_bits),
                ):
                    current = combined[name].index_select(0, indices)
                    combined[name][indices] = torch.where(
                        still_unresolved, values[start:end], current
                    )
                for name, values in (
                    ("later", expanded.later),
                    ("target", expanded.target),
                    ("quantile_lower", expanded.quantile_lower),
                    ("quantile_upper", expanded.quantile_upper),
                    ("target_lower", expanded.target_lower),
                    ("target_upper", expanded.target_upper),
                    ("certificate_codes", expanded.certificate_codes),
                ):
                    current = combined[name].index_select(0, indices)
                    combined[name][indices] = torch.where(take, values[start:end], current)
                current_strengthened = combined["strengthened_mask"].index_select(
                    0, indices
                )
                combined["strengthened_mask"][indices] = current_strengthened | (
                    take
                    & (expanded.strengthened_mask[start:end] | bool(strengthened_pass))
                )
                current_reason = combined["fallback_reason_codes"].index_select(
                    0, indices
                )
                current_reason = torch.where(
                    still_unresolved & ~take,
                    expanded.fallback_reason_codes[start:end],
                    current_reason,
                )
                combined["fallback_reason_codes"][indices] = torch.where(
                    take, torch.zeros_like(current_reason), current_reason
                )
                current_authorized = combined["authorized_mask"].index_select(
                    0, indices
                )
                combined["authorized_mask"][indices] = current_authorized | take

    def run_nearest_directional_lattice(
        *,
        strengthened_pass: bool,
        stage_max_prefix_bits: int,
        stage_primary_cap: int,
        stage_strengthened_cap: int,
    ) -> None:
        """Probe only a rigorously established CDF direction, nearest first."""

        unresolved = active & ~combined["authorized_mask"]
        reasons = combined["fallback_reason_codes"]
        stage_high = unresolved & (reasons == _CDF_CANDIDATE_TOO_HIGH)
        stage_low = unresolved & (reasons == _CDF_CANDIDATE_TOO_LOW)
        for distance in range(1, int(maximum_neighbor_ulps) + 1):
            # Keep moving in a direction only while the preceding cell is
            # still definitely on that same side.  An overlap or a direction
            # reversal means that stronger arithmetic, not a farther cell, is
            # required.
            run_lattice(
                [-distance],
                strengthened_pass=strengthened_pass,
                stage_max_prefix_bits=stage_max_prefix_bits,
                stage_primary_cap=stage_primary_cap,
                stage_strengthened_cap=stage_strengthened_cap,
                eligible_mask=(
                    stage_high
                    & (
                        combined["fallback_reason_codes"]
                        == _CDF_CANDIDATE_TOO_HIGH
                    )
                ),
            )
            run_lattice(
                [distance],
                strengthened_pass=strengthened_pass,
                stage_max_prefix_bits=stage_max_prefix_bits,
                stage_primary_cap=stage_primary_cap,
                stage_strengthened_cap=stage_strengthened_cap,
                eligible_mask=(
                    stage_low
                    & (
                        combined["fallback_reason_codes"]
                        == _CDF_CANDIDATE_TOO_LOW
                    )
                ),
            )

    # First pass also emits a non-authorizing DD Newton suggestion for a CDF
    # miss.  The suggestion must subsequently pass its own exact cell proof.
    initial_suggestion_adopted = False
    if bool(active.any().item()):
        initial = launch_fused_cuda_authorizer(
            bundle, head_fraction, exposure, transition_ids, candidate_at(0),
            seed=seed, threads_per_block=threads_per_block,
            max_prefix_bits=(
                int(max_prefix_bits) if force_strengthened
                else min(64, int(max_prefix_bits))
            ),
            recorded_prefix_numerators=recorded_prefix_numerators,
            recorded_prefix_bits=recorded_prefix_bits,
            _primary_cap=(0 if force_strengthened else 4096),
            _strengthened_cap=(8192 if force_strengthened else 4096),
        )
        record_launch(initial)
        absorb(initial, strengthened_pass=bool(force_strengthened))
        initial_suggestion_adopted = adopt_newton_suggestion(initial)

    # A Newton value remains non-authorizing: each update receives a fresh
    # exact rounding-cell proof.  The initial primary suggestion gets one
    # retest before the fixed +/-4 ULP primary lattice.
    if (
        not force_strengthened
        and initial_suggestion_adopted
        and bool((active & ~combined["authorized_mask"]).any().item())
    ):
        attempt = launch_fused_cuda_authorizer(
            bundle, head_fraction, exposure, transition_ids, candidate_at(0),
            seed=seed, threads_per_block=threads_per_block,
            max_prefix_bits=min(64, int(max_prefix_bits)),
            recorded_prefix_numerators=recorded_prefix_numerators,
            recorded_prefix_bits=recorded_prefix_bits,
            _primary_cap=4096, _strengthened_cap=4096,
        )
        record_launch(attempt)
        absorb(attempt, strengthened_pass=False)

    # Probe the nearest cells first and compact away every success before the
    # next distance.  Device-proved CDF direction also excludes the opposite
    # side; genuinely overlapping CDF balls proceed to stronger arithmetic.
    if not force_strengthened:
        run_nearest_directional_lattice(
            strengthened_pass=False,
            stage_max_prefix_bits=min(64, int(max_prefix_bits)),
            stage_primary_cap=4096,
            stage_strengthened_cap=4096,
        )

    # Strengthened mode permits 8192 modes and a second v2 Philox block.  Its
    # first attempt may emit a better DD Newton value; adopt that value once
    # and retest it before the strengthened +/-4 ULP lattice.
    strengthened_suggestion_adopted = initial_suggestion_adopted if force_strengthened else False
    if (
        not force_strengthened
        and bool((active & ~combined["authorized_mask"]).any().item())
    ):
        attempt = launch_fused_cuda_authorizer(
            bundle, head_fraction, exposure, transition_ids, candidate_at(0),
            seed=seed, threads_per_block=threads_per_block,
            max_prefix_bits=int(max_prefix_bits),
            recorded_prefix_numerators=recorded_prefix_numerators,
            recorded_prefix_bits=recorded_prefix_bits,
            _primary_cap=(0 if force_strengthened else 4096),
            _strengthened_cap=8192,
        )
        record_launch(attempt)
        absorb(attempt, strengthened_pass=True)
        strengthened_suggestion_adopted = adopt_newton_suggestion(attempt)

    if (
        strengthened_suggestion_adopted
        and bool((active & ~combined["authorized_mask"]).any().item())
    ):
        attempt = launch_fused_cuda_authorizer(
            bundle, head_fraction, exposure, transition_ids, candidate_at(0),
            seed=seed, threads_per_block=threads_per_block,
            max_prefix_bits=int(max_prefix_bits),
            recorded_prefix_numerators=recorded_prefix_numerators,
            recorded_prefix_bits=recorded_prefix_bits,
            _primary_cap=(0 if force_strengthened else 4096),
            _strengthened_cap=8192,
        )
        record_launch(attempt)
        absorb(attempt, strengthened_pass=True)

    run_nearest_directional_lattice(
        strengthened_pass=True,
        stage_max_prefix_bits=int(max_prefix_bits),
        stage_primary_cap=(0 if force_strengthened else 4096),
        stage_strengthened_cap=8192,
    )

    combined["fallback_reason_codes"] = torch.where(
        active & ~combined["authorized_mask"] & (combined["fallback_reason_codes"] == 0),
        torch.full_like(combined["fallback_reason_codes"], 8),
        combined["fallback_reason_codes"],
    )
    unresolved = active & ~combined["authorized_mask"]
    # This remains non-authorizing.  It is retained solely so the local Arb
    # fallback starts from the DD Newton cell rather than restarting inversion.
    combined["later"] = torch.where(unresolved, base_candidate, combined["later"])
    return FusedCudaLaunch(
        bundle=bundle,
        maximum_launch_lanes=maximum_cuda_launch_lanes,
        launch_count=cuda_authorizer_launch_count,
        **combined,
    )


__all__ = [
    "FUSED_CUDA_VERSION",
    "LEGENDRE_RECURRENCE_THEOREM",
    "LEGENDRE_RECURRENCE_THEOREM_URI",
    "LEGENDRE_RECURRENCE_ERROR_FACTOR",
    "CONSTANTS_SHA256",
    "EXP_DD_HI_HEX",
    "EXP_DD_LO_HEX",
    "EXP_DD_RAD_HEX",
    "FusedCudaBundle",
    "FusedCudaLaunch",
    "REQUIRED_SELFTEST_MASK",
    "SOURCE_SHA256",
    "verify_fused_device_constants",
    "launch_fused_cuda_authorizer",
    "launch_fused_cuda_authorizer_with_neighbors",
    "probe_fused_cuda_authorizer",
    "probe_fused_legendre_enclosures",
]
