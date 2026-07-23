r"""Rigorous CPU reference arithmetic for the fused Jacobi RB CUDA certifier.

This module contains no CUDA and never treats an approximate point value as an
authorization.  It is the executable specification used by preflight and by
tests of the device implementation:

* IEEE-754 binary64 rounding cells are represented by exact :class:`Fraction`
  boundaries and are tested with strict inequalities;
* :class:`Ball` operations round exact endpoint algebra outwards;
* error-free TwoSum/FMA-TwoProd and double-double reference operations expose
  the arithmetic contract expected from the kernel;
* ``exp`` is enclosed by a frozen, degree-24 Taylor kernel after a certified
  range reduction by an exact decimal bracket for ``ln(2)``;
* Legendre CDF, density, and conormal (``G``) recurrences are widened by
  analytic geometric bounds for every omitted mode.

Python-flint/Arb is optional.  Its helpers are independent cross-checks, not a
substitute for any of the proofs above.  Every inability to prove a strict
comparison is reported as a fallback; equality never authorizes a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
from fractions import Fraction
import hashlib
import math
from typing import Any, Iterable, Mapping

try:  # Optional independent oracle.
    from flint import arb as _arb
    from flint import ctx as _flint_ctx
except ImportError:  # pragma: no cover - exercised in stripped deployments.
    _arb = None
    _flint_ctx = None


CERTIFICATE_ARITHMETIC_VERSION = "jacobi-rb-cuda-certificate-fraction-exp24-v1"
EXP_DEGREE = 24
LN2_HEX = "0x1.62e42fefa39efp-1"
INV_LN2_HEX = "0x1.71547652b82fep+0"

# The bracket is frozen as decimal integers so importing the module performs
# no transcendental calculation.  The displayed digits come from a directed
# 80-decimal enclosure and are checked against Arb when it is available.
LN2_LOWER = Fraction(
    69314718055994530941723212145817656807550013436025525412068000949339362196969471,
    10**80,
)
LN2_UPPER = Fraction(
    69314718055994530941723212145817656807550013436025525412068000949339362196969472,
    10**80,
)

# Frozen binary64 coefficients are part of the device ABI.  The reference
# proof evaluates the mathematical coefficients 1/n! exactly; device code is
# separately required to enclose its coefficient loads and arithmetic.
EXP24_COEFFICIENT_HEX = (
    "0x1.0000000000000p+0", "0x1.0000000000000p+0",
    "0x1.0000000000000p-1", "0x1.5555555555555p-3",
    "0x1.5555555555555p-5", "0x1.1111111111111p-7",
    "0x1.6c16c16c16c17p-10", "0x1.a01a01a01a01ap-13",
    "0x1.a01a01a01a01ap-16", "0x1.71de3a556c734p-19",
    "0x1.27e4fb7789f5cp-22", "0x1.ae64567f544e4p-26",
    "0x1.1eed8eff8d898p-29", "0x1.6124613a86d09p-33",
    "0x1.93974a8c07c9dp-37", "0x1.ae7f3e733b81fp-41",
    "0x1.ae7f3e733b81fp-45", "0x1.952c77030ad4ap-49",
    "0x1.6827863b97d97p-53", "0x1.2f49b46814157p-57",
    "0x1.e542ba4020225p-62", "0x1.71b8ef6dcf572p-66",
    "0x1.0ce396db7f853p-70", "0x1.761b41316381ap-75",
    "0x1.f2cf01972f578p-80",
)
EXP24_REMAINDER_BOUND = Fraction(3, 2) * Fraction(3, 8) ** 25 / math.factorial(25)

_MAX_FLOAT = float.fromhex("0x1.fffffffffffffp+1023")
_MIN_SUBNORMAL = float.fromhex("0x0.0000000000001p-1022")


class ArithmeticCertificationError(RuntimeError):
    """An arithmetic fact required for authorization could not be proved."""

    def __init__(self, message: str, reason: "FallbackReason", **diagnostics: Any):
        super().__init__(message)
        self.reason = FallbackReason(reason)
        self.diagnostics = dict(diagnostics)


class CertificateBit(IntFlag):
    """Per-row bits; all four are required for CUDA-only authorization."""

    NONE = 0
    CDF_INVERSE = 1 << 0
    DENSITY_POSITIVE = 1 << 1
    TARGET_ENCLOSURE = 1 << 2
    CORRECT_ROUNDING = 1 << 3
    ALL = CDF_INVERSE | DENSITY_POSITIVE | TARGET_ENCLOSURE | CORRECT_ROUNDING

    # Readable compatibility aliases for backend code.
    QUANTILE_BRACKET = CDF_INVERSE
    FINITE_POSITIVE_DENSITY = DENSITY_POSITIVE
    TARGET_QUOTIENT = TARGET_ENCLOSURE
    UNIQUE_ROUNDING_CELL = CORRECT_ROUNDING


class FallbackReason(IntEnum):
    """Stable, JSON/device-friendly reason codes.  Zero means no fallback."""

    NONE = 0
    AMBIGUOUS_CDF = 1
    NONPOSITIVE_DENSITY = 2
    TARGET_CELL_AMBIGUOUS = 3
    EXP_RANGE_REDUCTION = 4
    TAIL_NOT_CONTRACTIVE = 5
    NONFINITE_INPUT = 6
    ARITHMETIC_FAULT = 7
    MODE_CAP = 8
    ARB_UNAVAILABLE = 9
    ARB_DISAGREEMENT = 10
    ROUNDING_CELL_UNREPRESENTABLE = 11


def _finite(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ArithmeticCertificationError(
            "certificate arithmetic requires finite binary64 inputs",
            FallbackReason.NONFINITE_INPUT,
            value=repr(result),
        )
    return result


def _as_fraction(value: float | int | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    return Fraction.from_float(_finite(value))


def fraction_to_float_down(value: Fraction | int) -> float:
    """Largest binary64 not exceeding an exact rational (directed down)."""

    exact = Fraction(value)
    try:
        rounded = float(exact)
    except OverflowError:
        return _MAX_FLOAT if exact > 0 else -math.inf
    if math.isinf(rounded):
        return _MAX_FLOAT if rounded > 0 else -math.inf
    if Fraction.from_float(rounded) > exact:
        rounded = math.nextafter(rounded, -math.inf)
    return rounded


def fraction_to_float_up(value: Fraction | int) -> float:
    """Smallest binary64 not less than an exact rational (directed up)."""

    exact = Fraction(value)
    try:
        rounded = float(exact)
    except OverflowError:
        return math.inf if exact > 0 else -_MAX_FLOAT
    if math.isinf(rounded):
        return math.inf if rounded > 0 else -_MAX_FLOAT
    if Fraction.from_float(rounded) < exact:
        rounded = math.nextafter(rounded, math.inf)
    return rounded


@dataclass(frozen=True)
class Twofold:
    """Unevaluated sum of two binary64 values, conventionally ``hi + lo``."""

    hi: float
    lo: float = 0.0

    def __post_init__(self) -> None:
        _finite(self.hi)
        _finite(self.lo)

    @property
    def exact(self) -> Fraction:
        return Fraction.from_float(self.hi) + Fraction.from_float(self.lo)

    @property
    def value(self) -> float:
        return float(self.hi + self.lo)


def fraction_to_twofold(value: Fraction | int) -> Twofold:
    """Split a rational into two binary64 terms, failing if not exact."""

    exact = Fraction(value)
    hi = float(exact)
    if not math.isfinite(hi):
        raise ArithmeticCertificationError(
            "twofold high component overflowed",
            FallbackReason.ROUNDING_CELL_UNREPRESENTABLE,
        )
    residual = exact - Fraction.from_float(hi)
    lo = float(residual)
    result = Twofold(hi, lo)
    if result.exact != exact:
        raise ArithmeticCertificationError(
            "exact rational needs more than two binary64 components",
            FallbackReason.ROUNDING_CELL_UNREPRESENTABLE,
            residual=str(exact - result.exact),
        )
    return result


@dataclass(frozen=True)
class RoundingCell:
    """Exact open round-to-nearest boundary pair around one binary64."""

    candidate: float
    lower: Fraction
    upper: Fraction

    def contains_fraction(self, value: Fraction | int) -> bool:
        exact = Fraction(value)
        return self.lower < exact < self.upper

    def contains_ball(self, value: "Ball") -> bool:
        return (
            Fraction.from_float(value.lower) > self.lower
            and Fraction.from_float(value.upper) < self.upper
        )

    def as_twofold(self) -> tuple[Twofold, Twofold]:
        return fraction_to_twofold(self.lower), fraction_to_twofold(self.upper)


def rounding_cell(value: float) -> RoundingCell:
    """Return exact binary64 rounding boundaries, including zero/max-finite."""

    candidate = _finite(value)
    exact = Fraction.from_float(candidate)
    previous = math.nextafter(candidate, -math.inf)
    following = math.nextafter(candidate, math.inf)
    if math.isinf(previous):  # candidate == -max-finite
        previous_exact = -Fraction(1 << 1024)
    else:
        previous_exact = Fraction.from_float(previous)
    if math.isinf(following):  # candidate == +max-finite
        following_exact = Fraction(1 << 1024)
    else:
        following_exact = Fraction.from_float(following)
    return RoundingCell(
        candidate=candidate,
        lower=(previous_exact + exact) / 2,
        upper=(exact + following_exact) / 2,
    )


def rounding_cell_boundaries(value: float) -> tuple[Fraction, Fraction]:
    cell = rounding_cell(value)
    return cell.lower, cell.upper


def rounding_cell_twofold(value: float) -> tuple[Twofold, Twofold]:
    return rounding_cell(value).as_twofold()


@dataclass(frozen=True)
class Ball:
    """Closed finite binary64 interval with rigorously outward operations."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        lower, upper = _finite(self.lower), _finite(self.upper)
        if lower > upper:
            raise ArithmeticCertificationError(
                "ball endpoints are reversed",
                FallbackReason.ARITHMETIC_FAULT,
                lower=lower,
                upper=upper,
            )

    @classmethod
    def point(cls, value: float | int | Fraction) -> "Ball":
        if isinstance(value, float):
            exact = _as_fraction(value)
        else:
            exact = Fraction(value)
        return cls(fraction_to_float_down(exact), fraction_to_float_up(exact))

    @classmethod
    def from_fractions(cls, lower: Fraction, upper: Fraction) -> "Ball":
        if lower > upper:
            raise ArithmeticCertificationError(
                "exact ball endpoints are reversed", FallbackReason.ARITHMETIC_FAULT
            )
        lo, hi = fraction_to_float_down(lower), fraction_to_float_up(upper)
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ArithmeticCertificationError(
                "outward ball overflowed binary64",
                FallbackReason.ARITHMETIC_FAULT,
            )
        return cls(lo, hi)

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def midpoint(self) -> float:
        return self.lower + 0.5 * (self.upper - self.lower)

    def fractions(self) -> tuple[Fraction, Fraction]:
        return Fraction.from_float(self.lower), Fraction.from_float(self.upper)

    def contains(self, value: float | int | Fraction) -> bool:
        exact = _as_fraction(value)
        lo, hi = self.fractions()
        return lo <= exact <= hi

    def contains_ball(self, other: "Ball") -> bool:
        return self.lower <= other.lower and other.upper <= self.upper

    def add(self, other: "Ball") -> "Ball":
        a, b = self.fractions(), other.fractions()
        return Ball.from_fractions(a[0] + b[0], a[1] + b[1])

    def sub(self, other: "Ball") -> "Ball":
        a, b = self.fractions(), other.fractions()
        return Ball.from_fractions(a[0] - b[1], a[1] - b[0])

    def mul(self, other: "Ball") -> "Ball":
        a, b = self.fractions(), other.fractions()
        products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
        return Ball.from_fractions(min(products), max(products))

    def reciprocal(self) -> "Ball":
        lo, hi = self.fractions()
        if lo <= 0 <= hi:
            raise ArithmeticCertificationError(
                "division ball contains zero",
                FallbackReason.NONPOSITIVE_DENSITY,
                lower=self.lower,
                upper=self.upper,
            )
        return Ball.from_fractions(min(Fraction(1, lo), Fraction(1, hi)), max(Fraction(1, lo), Fraction(1, hi)))

    def div(self, other: "Ball") -> "Ball":
        return self.mul(other.reciprocal())

    def scale(self, scalar: float | int | Fraction) -> "Ball":
        return self.mul(Ball.point(scalar))

    def inflate(self, radius: float | int | Fraction) -> "Ball":
        exact = _as_fraction(radius)
        if exact < 0:
            raise ValueError("radius must be nonnegative")
        lo, hi = self.fractions()
        return Ball.from_fractions(lo - exact, hi + exact)


def ball_add(left: Ball, right: Ball) -> Ball:
    return left.add(right)


def ball_sub(left: Ball, right: Ball) -> Ball:
    return left.sub(right)


def ball_mul(left: Ball, right: Ball) -> Ball:
    return left.mul(right)


def ball_div(left: Ball, right: Ball) -> Ball:
    return left.div(right)


def two_sum(left: float, right: float) -> Twofold:
    """Knuth TwoSum: an error-free sum when the finite result does not overflow."""

    a, b = _finite(left), _finite(right)
    high = a + b
    if not math.isfinite(high):
        raise ArithmeticCertificationError("TwoSum overflow", FallbackReason.ARITHMETIC_FAULT)
    virtual_b = high - a
    low = (a - (high - virtual_b)) + (b - virtual_b)
    result = Twofold(high, low)
    if result.exact != Fraction.from_float(a) + Fraction.from_float(b):
        raise ArithmeticCertificationError(
            "TwoSum lost an unrepresentable underflow residue",
            FallbackReason.ARITHMETIC_FAULT,
        )
    return result


def fast_two_sum(left: float, right: float) -> Twofold:
    """Error-free FastTwoSum, requiring ``abs(left) >= abs(right)``."""

    a, b = _finite(left), _finite(right)
    if abs(a) < abs(b):
        raise ValueError("FastTwoSum requires abs(left) >= abs(right)")
    high = a + b
    if not math.isfinite(high):
        raise ArithmeticCertificationError("FastTwoSum overflow", FallbackReason.ARITHMETIC_FAULT)
    low = b - (high - a)
    result = Twofold(high, low)
    if result.exact != Fraction.from_float(a) + Fraction.from_float(b):
        raise ArithmeticCertificationError(
            "FastTwoSum was not error-free", FallbackReason.ARITHMETIC_FAULT
        )
    return result


def two_prod_fma(left: float, right: float) -> Twofold:
    """FMA TwoProd reference, checked against the exact rational product."""

    a, b = _finite(left), _finite(right)
    high = a * b
    if not math.isfinite(high):
        raise ArithmeticCertificationError("TwoProd overflow", FallbackReason.ARITHMETIC_FAULT)
    if hasattr(math, "fma"):
        low = math.fma(a, b, -high)
    else:  # pragma: no cover - Python builds without fma.
        low = float(Fraction.from_float(a) * Fraction.from_float(b) - Fraction.from_float(high))
    result = Twofold(high, low)
    exact = Fraction.from_float(a) * Fraction.from_float(b)
    if result.exact != exact:
        raise ArithmeticCertificationError(
            "FMA TwoProd lost an underflow residue", FallbackReason.ARITHMETIC_FAULT
        )
    return result


def dd_from_fraction(value: Fraction | int) -> Twofold:
    """Round an exact rational to a normalized two-component expansion."""

    exact = Fraction(value)
    high = float(exact)
    if not math.isfinite(high):
        raise ArithmeticCertificationError(
            "double-double high component overflowed", FallbackReason.ARITHMETIC_FAULT
        )
    low = float(exact - Fraction.from_float(high))
    return Twofold(high, low)


def dd_add(left: Twofold, right: Twofold) -> Twofold:
    return dd_from_fraction(left.exact + right.exact)


def dd_sub(left: Twofold, right: Twofold) -> Twofold:
    return dd_from_fraction(left.exact - right.exact)


def dd_mul(left: Twofold, right: Twofold) -> Twofold:
    # A two-component result cannot generally retain the exact four-component
    # product.  This is the correctly split CPU reference approximation;
    # authorizing consumers use ``dd_mul_enclosure`` below.
    return dd_from_fraction(left.exact * right.exact)


def dd_add_enclosure(left: Twofold, right: Twofold) -> Ball:
    return Ball.point(left.exact + right.exact)


def dd_mul_enclosure(left: Twofold, right: Twofold) -> Ball:
    return Ball.point(left.exact * right.exact)


@dataclass(frozen=True)
class InverseCellDecision:
    left_strict: bool
    right_strict: bool

    @property
    def certified(self) -> bool:
        return self.left_strict and self.right_strict


def strict_inverse_cell_inequalities(
    cdf_at_lower_boundary: Ball,
    cdf_at_upper_boundary: Ball,
    uniform_lower: Fraction | float,
    uniform_upper: Fraction | float | None = None,
) -> InverseCellDecision:
    """Prove ``F(cell_lo) < U < F(cell_hi)`` with strict exact comparisons.

    ``U`` may itself be a dyadic prefix interval.  Its closed endpoints are
    used in the hardest direction, so equality is necessarily inconclusive.
    """

    u_lo = _as_fraction(uniform_lower)
    u_hi = u_lo if uniform_upper is None else _as_fraction(uniform_upper)
    if u_lo > u_hi:
        raise ValueError("uniform endpoints are reversed")
    f_lo_upper = Fraction.from_float(cdf_at_lower_boundary.upper)
    f_hi_lower = Fraction.from_float(cdf_at_upper_boundary.lower)
    return InverseCellDecision(f_lo_upper < u_lo, f_hi_lower > u_hi)


def strict_rounding_cell_contains(value: Ball, candidate: float) -> bool:
    """True only when the entire closed ball is strictly inside one cell."""

    try:
        return rounding_cell(candidate).contains_ball(value)
    except ArithmeticCertificationError:
        return False


@dataclass(frozen=True)
class _QBall:
    lower: Fraction
    upper: Fraction

    def add(self, other: "_QBall") -> "_QBall":
        return _QBall(self.lower + other.lower, self.upper + other.upper)

    def mul(self, other: "_QBall") -> "_QBall":
        values = (
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        )
        return _QBall(min(values), max(values))


def _nearest_integer_ratio(numerator: Fraction, denominator: Fraction) -> int:
    value = numerator / denominator
    floor = value.numerator // value.denominator
    remainder = value - floor
    if remainder < Fraction(1, 2):
        return floor
    if remainder > Fraction(1, 2):
        return floor + 1
    return floor if floor % 2 == 0 else floor + 1


def _exp24_fraction_enclosure(value: Ball) -> tuple[_QBall, int, Fraction]:
    x_lo, x_hi = value.fractions()
    midpoint = (x_lo + x_hi) / 2
    k = _nearest_integer_ratio(midpoint, (LN2_LOWER + LN2_UPPER) / 2)
    if abs(k) > 16384:
        raise ArithmeticCertificationError(
            "exponential range reduction exceeds frozen cap",
            FallbackReason.EXP_RANGE_REDUCTION,
            reduction=k,
        )
    if k >= 0:
        r_lo = x_lo - k * LN2_UPPER
        r_hi = x_hi - k * LN2_LOWER
    else:
        r_lo = x_lo - k * LN2_LOWER
        r_hi = x_hi - k * LN2_UPPER
    reduced = _QBall(r_lo, r_hi)
    radius = max(abs(r_lo), abs(r_hi))
    if radius > Fraction(3, 8):
        raise ArithmeticCertificationError(
            "degree-24 exp range reduction is ambiguous",
            FallbackReason.EXP_RANGE_REDUCTION,
            reduction=k,
            reduced_lower=str(r_lo),
            reduced_upper=str(r_hi),
        )
    polynomial = _QBall(Fraction(1, math.factorial(EXP_DEGREE)), Fraction(1, math.factorial(EXP_DEGREE)))
    for degree in range(EXP_DEGREE - 1, -1, -1):
        coefficient = Fraction(1, math.factorial(degree))
        polynomial = polynomial.mul(reduced).add(_QBall(coefficient, coefficient))
    remainder = Fraction(3, 2) * radius**25 / math.factorial(25)
    scale = Fraction(1 << k) if k >= 0 else Fraction(1, 1 << (-k))
    enclosed = _QBall(
        max(Fraction(0), (polynomial.lower - remainder) * scale),
        (polynomial.upper + remainder) * scale,
    )
    return enclosed, k, remainder * scale


@dataclass(frozen=True)
class Exp24Enclosure:
    value: Ball
    reduction: int
    scaled_remainder_bound: Fraction
    degree: int = EXP_DEGREE


def exp24_enclosure(value: float | Ball) -> Exp24Enclosure:
    """Certified enclosure of ``exp(value)`` using the frozen degree-24 kernel."""

    source = value if isinstance(value, Ball) else Ball.point(_finite(value))
    exact, reduction, remainder = _exp24_fraction_enclosure(source)
    return Exp24Enclosure(
        Ball.from_fractions(exact.lower, exact.upper), reduction, remainder
    )


@dataclass(frozen=True)
class TailBounds:
    cdf: Ball
    density: Ball
    conormal: Ball
    first_omitted: int


def legendre_tail_bounds(first_omitted: int, exposure: float) -> TailBounds:
    """Uniform absolute CDF/density/G tails for all ``x,y in [0,1]``."""

    mode = int(first_omitted)
    u = _finite(exposure)
    if mode < 1:
        raise ValueError("first_omitted must be positive")
    if u <= 0.0:
        raise ArithmeticCertificationError(
            "positive exposure is required for a spectral tail",
            FallbackReason.TAIL_NOT_CONTRACTIVE,
        )
    exposure_ball = Ball.point(u)
    decay = exp24_enclosure(exposure_ball.scale(-mode * (mode + 1))).value
    step = exp24_enclosure(exposure_ball.scale(-2 * (mode + 1))).value
    q = Fraction.from_float(step.upper)
    if q >= 1:
        raise ArithmeticCertificationError(
            "CDF tail ratio is not strictly contractive",
            FallbackReason.TAIL_NOT_CONTRACTIVE,
        )
    decay_hi = Fraction.from_float(decay.upper)
    cdf_radius = decay_hi / (1 - q)
    density_ratio = Fraction(2 * mode + 3, 2 * mode + 1) * q
    conormal_ratio = density_ratio * Fraction(mode + 1, mode)
    if density_ratio >= 1 or conormal_ratio >= 1:
        raise ArithmeticCertificationError(
            "density/G tail ratio is not strictly contractive",
            FallbackReason.TAIL_NOT_CONTRACTIVE,
            density_ratio=str(density_ratio),
            conormal_ratio=str(conormal_ratio),
        )
    density_radius = Fraction(2 * mode + 1) * decay_hi / (1 - density_ratio)
    conormal_radius = Fraction(mode * (2 * mode + 1)) * decay_hi / (1 - conormal_ratio)
    return TailBounds(
        cdf=Ball.point(cdf_radius),
        density=Ball.point(density_radius),
        conormal=Ball.point(conormal_radius),
        first_omitted=mode,
    )


@dataclass(frozen=True)
class LegendreSpectralEnclosure:
    cdf: Ball
    density: Ball
    conormal: Ball
    tail: TailBounds
    modes_used: int

    @property
    def G(self) -> Ball:  # noqa: N802 - mathematical public name.
        return self.conormal


def _validate_spectral_inputs(x: float, y: float, exposure: float, modes: int) -> tuple[float, float, float]:
    x, y, exposure = _finite(x), _finite(y), _finite(exposure)
    if not 0 <= x <= 1 or not 0 <= y <= 1:
        raise ValueError("Legendre coordinates must lie in [0,1]")
    if exposure <= 0:
        raise ValueError("exposure must be positive")
    if int(modes) < 2:
        raise ValueError("modes must be at least two")
    return x, y, exposure


def legendre_spectral_enclosure(
    x: float,
    y: float,
    exposure: float,
    *,
    modes: int = 32,
) -> LegendreSpectralEnclosure:
    """Outward recurrence plus rigorous tails for CDF, density, and ``G``.

    ``modes`` counts the constant mode, so the recurrence evaluates degrees
    ``1 .. modes-1`` and bounds degrees ``modes .. infinity``.
    """

    x, y, exposure = _validate_spectral_inputs(x, y, exposure, modes)
    one = Ball.point(1)
    zx = Ball.point(x).scale(2).sub(one)
    zy = Ball.point(y).scale(2).sub(one)
    px_previous, px_current = one, zx
    py_previous, py_current = one, zy
    cdf, density, conormal = Ball.point(y), one, Ball.point(0)

    for degree in range(1, int(modes)):
        py_next = zy.mul(py_current).scale(2 * degree + 1).sub(
            py_previous.scale(degree)
        ).scale(Fraction(1, degree + 1))
        decay = exp24_enclosure(
            Ball.point(exposure).scale(-degree * (degree + 1))
        ).value
        coefficient = decay.scale(2 * degree + 1)
        cdf_term = decay.mul(px_current).mul(py_next.sub(py_previous)).scale(Fraction(1, 2))
        density_term = coefficient.mul(px_current).mul(py_current)
        basis = py_previous.sub(zy.mul(py_current)).scale(Fraction(degree, 2))
        conormal_term = coefficient.mul(px_current).mul(basis)
        cdf = cdf.add(cdf_term)
        density = density.add(density_term)
        conormal = conormal.add(conormal_term)
        px_next = zx.mul(px_current).scale(2 * degree + 1).sub(
            px_previous.scale(degree)
        ).scale(Fraction(1, degree + 1))
        px_previous, px_current = px_current, px_next
        py_previous, py_current = py_current, py_next

    tail = legendre_tail_bounds(int(modes), exposure)
    cdf = cdf.inflate(tail.cdf.upper)
    density = density.inflate(tail.density.upper)
    conormal = conormal.inflate(tail.conormal.upper)
    if y == 0.0:
        cdf = Ball.point(0.0)
    elif y == 1.0:
        cdf = Ball.point(1.0)
    return LegendreSpectralEnclosure(cdf, density, conormal, tail, int(modes))


def quotient_enclosure(numerator: Ball, denominator: Ball) -> Ball:
    """Sign-correct quotient enclosure; a denominator touching zero rejects."""

    return numerator.div(denominator)


def target_quotient_enclosure(spectral: LegendreSpectralEnclosure) -> Ball:
    if spectral.density.lower <= 0.0:
        raise ArithmeticCertificationError(
            "spectral density is not provably positive",
            FallbackReason.NONPOSITIVE_DENSITY,
            density_lower=spectral.density.lower,
        )
    return quotient_enclosure(spectral.conormal, spectral.density)


@dataclass(frozen=True)
class ArbCrossCheck:
    available: bool
    passed: bool
    certified: Ball | None
    oracle: Ball | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": int(self.available),
            "passed": int(self.passed),
            "certified": None if self.certified is None else [self.certified.lower, self.certified.upper],
            "oracle": None if self.oracle is None else [self.oracle.lower, self.oracle.upper],
            "detail": self.detail,
        }


def arb_available() -> bool:
    return _arb is not None and _flint_ctx is not None


def _arb_exact_fraction(value: Fraction) -> Any:
    assert _arb is not None
    return _arb(value.numerator) / _arb(value.denominator)


def _arb_to_ball(value: Any) -> Ball:
    lower, upper = float(value.lower()), float(value.upper())
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ArithmeticCertificationError(
            "Arb returned nonfinite endpoints", FallbackReason.ARB_DISAGREEMENT
        )
    return Ball(math.nextafter(lower, -math.inf), math.nextafter(upper, math.inf))


def arb_exp_enclosure(value: float, *, precision_bits: int = 256) -> Ball:
    if not arb_available():
        raise ArithmeticCertificationError(
            "python-flint/Arb is unavailable", FallbackReason.ARB_UNAVAILABLE
        )
    assert _flint_ctx is not None
    previous = int(_flint_ctx.prec)
    try:
        _flint_ctx.prec = int(precision_bits)
        result = _arb_exact_fraction(_as_fraction(value)).exp()
        return _arb_to_ball(result)
    finally:
        _flint_ctx.prec = previous


def arb_cross_check_exp24(value: float, *, precision_bits: int = 256) -> ArbCrossCheck:
    certified = exp24_enclosure(value).value
    if not arb_available():
        return ArbCrossCheck(False, False, certified, None, "python-flint/Arb unavailable")
    try:
        assert _arb is not None and _flint_ctx is not None
        previous = int(_flint_ctx.prec)
        try:
            _flint_ctx.prec = int(precision_bits)
            exact_oracle = _arb_exact_fraction(_as_fraction(value)).exp()
            passed = (
                exact_oracle >= _arb_exact_fraction(Fraction.from_float(certified.lower))
                and exact_oracle <= _arb_exact_fraction(Fraction.from_float(certified.upper))
            )
            oracle = _arb_to_ball(exact_oracle)
        finally:
            _flint_ctx.prec = previous
        return ArbCrossCheck(True, passed, certified, oracle, "contained" if passed else "Arb ball escaped certificate")
    except (ArithmeticError, ValueError, ArithmeticCertificationError) as exc:
        return ArbCrossCheck(True, False, certified, None, f"{type(exc).__name__}: {exc}")


def arb_cross_check_legendre(
    x: float,
    y: float,
    exposure: float,
    *,
    modes: int = 32,
    precision_bits: int = 256,
) -> Mapping[str, Any]:
    """Cross-check a finite partial recurrence in Arb against our full balls.

    The certified balls include omitted tails.  The Arb values deliberately
    stop at the same finite mode; containment is therefore a useful
    independent recurrence/arithmetic check without duplicating the tail proof.
    """

    certified = legendre_spectral_enclosure(x, y, exposure, modes=modes)
    if not arb_available():
        return {"available": 0, "passed": 0, "detail": "python-flint/Arb unavailable"}
    assert _arb is not None and _flint_ctx is not None
    previous_precision = int(_flint_ctx.prec)
    try:
        _flint_ctx.prec = int(precision_bits)
        one = _arb(1)
        zx = 2 * _arb_exact_fraction(_as_fraction(x)) - one
        zy = 2 * _arb_exact_fraction(_as_fraction(y)) - one
        u = _arb_exact_fraction(_as_fraction(exposure))
        px_prev, px_cur, py_prev, py_cur = one, zx, one, zy
        cdf, density, conormal = _arb_exact_fraction(_as_fraction(y)), one, _arb(0)
        for degree in range(1, int(modes)):
            py_next = ((2 * degree + 1) * zy * py_cur - degree * py_prev) / (degree + 1)
            decay = (-degree * (degree + 1) * u).exp()
            coefficient = (2 * degree + 1) * decay
            cdf += _arb(Fraction(1, 2).numerator) / _arb(2) * decay * px_cur * (py_next - py_prev)
            density += coefficient * px_cur * py_cur
            conormal += coefficient * px_cur * (_arb(degree) / _arb(2)) * (py_prev - zy * py_cur)
            px_next = ((2 * degree + 1) * zx * px_cur - degree * px_prev) / (degree + 1)
            px_prev, px_cur, py_prev, py_cur = px_cur, px_next, py_cur, py_next
        # The full CDF has exact boundary values.  A finite partial sum need
        # not have cancelled its omitted boundary terms yet, so compare the
        # theorem-level endpoint rather than that deliberately unfinished sum.
        cdf_oracle = Ball.point(y) if y in {0.0, 1.0} else _arb_to_ball(cdf)
        oracle = (cdf_oracle, _arb_to_ball(density), _arb_to_ball(conormal))
        checks = (
            certified.cdf.contains_ball(oracle[0]),
            certified.density.contains_ball(oracle[1]),
            certified.conormal.contains_ball(oracle[2]),
        )
        return {
            "available": 1,
            "passed": int(all(checks)),
            "cdf_pass": int(checks[0]),
            "density_pass": int(checks[1]),
            "conormal_pass": int(checks[2]),
        }
    except (ArithmeticError, ValueError, ArithmeticCertificationError) as exc:
        return {"available": 1, "passed": 0, "detail": f"{type(exc).__name__}: {exc}"}
    finally:
        _flint_ctx.prec = previous_precision


_CONSTANT_FINGERPRINT = hashlib.sha256(
    (LN2_HEX + INV_LN2_HEX + "|".join(EXP24_COEFFICIENT_HEX)).encode("ascii")
).hexdigest()


def run_certificate_arithmetic_preflight(
    *, faults: Iterable[str] = ()
) -> dict[str, Any]:
    """Run deterministic, JSON-serializable certificate arithmetic checks.

    ``faults`` is a test-only fail-closed injection surface.  Unknown fault
    names are themselves failures so misspellings cannot accidentally pass.
    The two top-level metrics are the fields consumed by CUDA preflight.
    """

    injected = {str(item) for item in faults}
    known = {"rounding_cell", "two_sum", "two_prod", "ball", "quotient", "exp", "legendre", "constants", "arb"}
    subchecks: dict[str, int] = {}
    errors: dict[str, str] = {}

    def check(name: str, operation: Any) -> None:
        try:
            passed = bool(operation()) and name not in injected
        except Exception as exc:  # Preflight is deliberately fail-closed.
            passed = False
            errors[name] = f"{type(exc).__name__}: {exc}"
        if name in injected:
            errors[name] = "fault injected"
        subchecks[name] = int(passed)

    check("rounding_cell", lambda: rounding_cell(1.0).lower == Fraction(2**54 - 1, 2**54) and rounding_cell(0.0).upper == Fraction(1, 2**1075))
    check("two_sum", lambda: two_sum(1.0, 2.0**-53).exact == Fraction.from_float(1.0) + Fraction.from_float(2.0**-53))
    check("two_prod", lambda: two_prod_fma(1.0 + 2.0**-27, 1.0 - 2.0**-27).exact == Fraction.from_float(1.0 + 2.0**-27) * Fraction.from_float(1.0 - 2.0**-27))
    check("ball", lambda: Ball.point(Fraction(1, 10)).mul(Ball.point(Fraction(-3, 7))).contains(Fraction(-3, 70)))
    check("quotient", lambda: quotient_enclosure(Ball(-3.0, -2.0), Ball(-5.0, -4.0)).contains(Fraction(1, 2)))
    def constants_check() -> bool:
        structural = (
            len(EXP24_COEFFICIENT_HEX) == 25
            and all(float.fromhex(item).hex() == item for item in EXP24_COEFFICIENT_HEX)
            and Fraction.from_float(float.fromhex(LN2_HEX)) < LN2_LOWER < LN2_UPPER
            and LN2_UPPER - LN2_LOWER == Fraction(1, 10**80)
        )
        if not structural or not arb_available():
            return structural
        assert _arb is not None and _flint_ctx is not None
        previous = int(_flint_ctx.prec)
        try:
            _flint_ctx.prec = 384
            actual = _arb(2).log()
            return actual > _arb_exact_fraction(LN2_LOWER) and actual < _arb_exact_fraction(LN2_UPPER)
        finally:
            _flint_ctx.prec = previous

    check("constants", constants_check)

    def exp_check() -> bool:
        for point in (-20.0, -1.0, -0.0, 0.1, 1.0, 20.0):
            enclosure = exp24_enclosure(point).value
            # Diagnostic libm containment is supplemented by Arb below when
            # available; authorization follows from the exact construction.
            if not enclosure.contains(math.exp(point)):
                return False
        return True

    check("exp", exp_check)

    def legendre_check() -> bool:
        value = legendre_spectral_enclosure(0.25, 0.75, 1.0, modes=16)
        target = target_quotient_enclosure(value)
        return value.cdf.lower <= value.cdf.upper and value.density.lower > 0 and target.lower <= target.upper

    check("legendre", legendre_check)
    if arb_available():
        check("arb", lambda: all(arb_cross_check_exp24(point).passed for point in (-3.0, -0.1, 0.1, 3.0)))
    else:
        subchecks["arb"] = 0
        errors["arb"] = "optional python-flint/Arb unavailable"
    unknown = sorted(injected - known)
    if unknown:
        errors["unknown_faults"] = ",".join(unknown)

    algebra_names = ("rounding_cell", "two_sum", "two_prod", "ball", "quotient", "constants", "legendre")
    algebra_pass = all(subchecks.get(name) == 1 for name in algebra_names) and not unknown
    exponential_pass = subchecks.get("exp") == 1 and subchecks.get("constants") == 1 and not unknown
    # Arb is an independent optional cross-check; when installed, a mismatch
    # is fatal.  Absence is reported but does not invalidate the native proof.
    if arb_available() and subchecks.get("arb") != 1:
        exponential_pass = False
    return {
        "schema_version": 1,
        "arithmetic_version": CERTIFICATE_ARITHMETIC_VERSION,
        "constant_fingerprint": _CONSTANT_FINGERPRINT,
        "passed": int(algebra_pass and exponential_pass),
        "double_double_interval_algebra_pass": int(algebra_pass),
        "certified_exponential_pass": int(exponential_pass),
        "subchecks": subchecks,
        "errors": errors,
        "arb_available": int(arb_available()),
    }


# Common backend spellings.
two_prod = two_prod_fma
certified_exp24 = exp24_enclosure
spectral_enclosure = legendre_spectral_enclosure


__all__ = [
    "ArithmeticCertificationError",
    "ArbCrossCheck",
    "Ball",
    "CERTIFICATE_ARITHMETIC_VERSION",
    "CertificateBit",
    "EXP24_COEFFICIENT_HEX",
    "EXP24_REMAINDER_BOUND",
    "EXP_DEGREE",
    "Exp24Enclosure",
    "FallbackReason",
    "INV_LN2_HEX",
    "InverseCellDecision",
    "LN2_HEX",
    "LN2_LOWER",
    "LN2_UPPER",
    "LegendreSpectralEnclosure",
    "RoundingCell",
    "TailBounds",
    "Twofold",
    "arb_available",
    "arb_cross_check_exp24",
    "arb_cross_check_legendre",
    "arb_exp_enclosure",
    "ball_add",
    "ball_div",
    "ball_mul",
    "ball_sub",
    "certified_exp24",
    "dd_add",
    "dd_add_enclosure",
    "dd_from_fraction",
    "dd_mul",
    "dd_mul_enclosure",
    "dd_sub",
    "exp24_enclosure",
    "fast_two_sum",
    "fraction_to_float_down",
    "fraction_to_float_up",
    "fraction_to_twofold",
    "legendre_spectral_enclosure",
    "legendre_tail_bounds",
    "quotient_enclosure",
    "rounding_cell",
    "rounding_cell_boundaries",
    "rounding_cell_twofold",
    "run_certificate_arithmetic_preflight",
    "spectral_enclosure",
    "strict_inverse_cell_inequalities",
    "strict_rounding_cell_contains",
    "target_quotient_enclosure",
    "two_prod",
    "two_prod_fma",
    "two_sum",
]
