from __future__ import annotations

from fractions import Fraction
import json
import math
import random

import pytest

from mnist import d0_jacobi_rb_cuda_certificate as cert


def _fraction(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def test_rounding_cells_are_exact_at_asymmetry_zero_subnormal_and_overflow() -> None:
    one = cert.rounding_cell(1.0)
    assert one.lower == Fraction(2**54 - 1, 2**54)
    assert one.upper == Fraction(2**53 + 1, 2**53)
    assert one.contains_fraction(Fraction(1))
    assert not one.contains_fraction(one.lower)
    assert not one.contains_fraction(one.upper)

    zero = cert.rounding_cell(0.0)
    assert zero.lower == -Fraction(1, 2**1075)
    assert zero.upper == Fraction(1, 2**1075)

    smallest = float.fromhex("0x0.0000000000001p-1022")
    smallest_cell = cert.rounding_cell(smallest)
    assert smallest_cell.lower == Fraction(1, 2**1075)
    assert smallest_cell.upper == Fraction(3, 2**1075)

    largest = float.fromhex("0x1.fffffffffffffp+1023")
    largest_cell = cert.rounding_cell(largest)
    assert largest_cell.upper == (_fraction(largest) + Fraction(1 << 1024)) / 2
    negative_largest_cell = cert.rounding_cell(-largest)
    assert negative_largest_cell.lower == (-Fraction(1 << 1024) - _fraction(largest)) / 2


def test_rounding_cell_twofold_is_exact_and_tiny_boundary_fails_closed() -> None:
    lower, upper = cert.rounding_cell_twofold(1.0)
    exact_lower, exact_upper = cert.rounding_cell_boundaries(1.0)
    assert lower.exact == exact_lower
    assert upper.exact == exact_upper
    with pytest.raises(cert.ArithmeticCertificationError) as caught:
        cert.rounding_cell_twofold(0.0)
    assert caught.value.reason == cert.FallbackReason.ROUNDING_CELL_UNREPRESENTABLE


def test_strict_rounding_and_inverse_cell_inequalities_reject_ties() -> None:
    candidate = 0.5
    cell = cert.rounding_cell(candidate)
    inside = cert.Ball.point(candidate)
    assert cert.strict_rounding_cell_contains(inside, candidate)
    touching = cert.Ball(
        cert.fraction_to_float_down(cell.lower),
        cert.fraction_to_float_up(_fraction(candidate)),
    )
    assert not cert.strict_rounding_cell_contains(touching, candidate)

    uniform_lo, uniform_hi = Fraction(4, 10), Fraction(5, 10)
    decision = cert.strict_inverse_cell_inequalities(
        cert.Ball(0.1, 0.39), cert.Ball(0.51, 0.9), uniform_lo, uniform_hi
    )
    assert decision.certified
    assert not cert.strict_inverse_cell_inequalities(
        cert.Ball(0.1, 0.4), cert.Ball(0.51, 0.9), uniform_lo, uniform_hi
    ).certified
    assert not cert.strict_inverse_cell_inequalities(
        cert.Ball(0.1, 0.39), cert.Ball(0.5, 0.9), uniform_lo, uniform_hi
    ).certified


def test_directed_rational_conversion_and_ball_operations_enclose_exact_algebra() -> None:
    tiny = Fraction(1, 2**1075)
    assert cert.fraction_to_float_down(tiny) == 0.0
    assert cert.fraction_to_float_up(tiny) == float.fromhex("0x0.0000000000001p-1022")
    assert cert.fraction_to_float_down(-tiny) == -float.fromhex("0x0.0000000000001p-1022")
    assert cert.fraction_to_float_up(-tiny) == -0.0

    rng = random.Random(94731)
    for _ in range(200):
        a = Fraction(rng.randint(-1000, 1000), rng.randint(1, 1000))
        b = Fraction(rng.randint(-1000, 1000), rng.randint(1, 1000))
        left, right = cert.Ball.point(a), cert.Ball.point(b)
        assert left.add(right).contains(a + b)
        assert left.sub(right).contains(a - b)
        assert left.mul(right).contains(a * b)
        if b:
            assert left.div(right).contains(a / b)


def test_ball_division_and_quotient_are_sign_correct_and_zero_fails_closed() -> None:
    cases = (
        (cert.Ball(2.0, 3.0), cert.Ball(4.0, 5.0), Fraction(1, 2)),
        (cert.Ball(-3.0, -2.0), cert.Ball(4.0, 5.0), Fraction(-1, 2)),
        (cert.Ball(2.0, 3.0), cert.Ball(-5.0, -4.0), Fraction(-1, 2)),
        (cert.Ball(-3.0, -2.0), cert.Ball(-5.0, -4.0), Fraction(1, 2)),
    )
    for numerator, denominator, witness in cases:
        assert cert.quotient_enclosure(numerator, denominator).contains(witness)
    for denominator in (cert.Ball(-1.0, 0.0), cert.Ball(0.0, 1.0), cert.Ball(-1.0, 1.0)):
        with pytest.raises(cert.ArithmeticCertificationError) as caught:
            cert.quotient_enclosure(cert.Ball.point(1), denominator)
        assert caught.value.reason == cert.FallbackReason.NONPOSITIVE_DENSITY


def test_two_sum_and_two_prod_are_error_free_on_adversarial_finite_inputs() -> None:
    sums = (
        (1.0, 2.0**-53),
        (1.0, -math.nextafter(1.0, 0.0)),
        (2.0**500, 2.0**447),
        (float.fromhex("0x0.0000000000002p-1022"), -float.fromhex("0x0.0000000000001p-1022")),
    )
    for left, right in sums:
        result = cert.two_sum(left, right)
        assert result.exact == _fraction(left) + _fraction(right)

    products = (
        (1.0 + 2.0**-27, 1.0 - 2.0**-27),
        (math.pi, math.e),
        (2.0**500, 2.0**-500),
        (-1.25, 3.5),
    )
    for left, right in products:
        result = cert.two_prod_fma(left, right)
        assert result.exact == _fraction(left) * _fraction(right)

    with pytest.raises(cert.ArithmeticCertificationError):
        cert.two_sum(float.fromhex("0x1.fffffffffffffp+1023"), float.fromhex("0x1.fffffffffffffp+1023"))
    with pytest.raises(cert.ArithmeticCertificationError):
        cert.two_prod_fma(float.fromhex("0x1.fffffffffffffp+1023"), 2.0)


def test_double_double_reference_and_authorizing_enclosures() -> None:
    a = cert.Twofold(1.0, 2.0**-53)
    b = cert.Twofold(-1.0, 2.0**-54)
    added = cert.dd_add(a, b)
    multiplied = cert.dd_mul(a, b)
    exact_sum = a.exact + b.exact
    exact_product = a.exact * b.exact
    assert abs(added.exact - exact_sum) <= Fraction(1, 2**106)
    assert abs(multiplied.exact - exact_product) <= Fraction(1, 2**105)
    assert cert.dd_add_enclosure(a, b).contains(exact_sum)
    assert cert.dd_mul_enclosure(a, b).contains(exact_product)


@pytest.mark.parametrize("point", [-700.0, -100.0, -20.0, -1.0, -0.1, -0.0, 0.1, 1.0, 20.0, 100.0, 700.0])
def test_exp24_encloses_libm_and_has_frozen_contract(point: float) -> None:
    result = cert.exp24_enclosure(point)
    assert result.degree == 24
    assert result.scaled_remainder_bound >= 0
    assert result.value.contains(math.exp(point))
    assert len(cert.EXP24_COEFFICIENT_HEX) == 25
    assert all(float.fromhex(value).hex() == value for value in cert.EXP24_COEFFICIENT_HEX)
    assert cert.LN2_LOWER < cert.LN2_UPPER
    assert cert.EXP24_REMAINDER_BOUND > 0


def test_exp24_ball_contains_endpoint_exponentials_and_rejects_ambiguous_wide_input() -> None:
    source = cert.Ball(-0.25, 0.25)
    result = cert.exp24_enclosure(source).value
    assert result.contains(math.exp(source.lower))
    assert result.contains(math.exp(source.upper))
    with pytest.raises(cert.ArithmeticCertificationError) as caught:
        cert.exp24_enclosure(cert.Ball(-1.0, 1.0))
    assert caught.value.reason == cert.FallbackReason.EXP_RANGE_REDUCTION


@pytest.mark.skipif(not cert.arb_available(), reason="python-flint/Arb unavailable")
def test_exp24_randomized_arb_cross_check() -> None:
    rng = random.Random(89217)
    for _ in range(80):
        point = rng.uniform(-700.0, 700.0)
        checked = cert.arb_cross_check_exp24(point, precision_bits=256)
        assert checked.available
        assert checked.passed, checked.to_dict()


def test_legendre_tail_bounds_contract_and_monotone_decay() -> None:
    previous = None
    for mode in range(4, 28):
        tail = cert.legendre_tail_bounds(mode, 0.5)
        assert tail.first_omitted == mode
        assert tail.cdf.lower >= 0
        assert tail.density.lower >= 0
        assert tail.conormal.lower >= 0
        if previous is not None:
            assert tail.cdf.upper < previous.cdf.upper
            assert tail.density.upper < previous.density.upper
            assert tail.conormal.upper < previous.conormal.upper
        previous = tail

    with pytest.raises(cert.ArithmeticCertificationError) as caught:
        cert.legendre_tail_bounds(1, 1e-20)
    assert caught.value.reason == cert.FallbackReason.TAIL_NOT_CONTRACTIVE


def test_legendre_cdf_density_g_enclosures_and_endpoint_identity() -> None:
    center = cert.legendre_spectral_enclosure(0.25, 0.75, 1.0, modes=20)
    assert center.cdf.lower < center.cdf.upper
    assert center.density.lower > 0.0
    assert center.conormal.lower < center.conormal.upper
    assert center.G == center.conormal
    target = cert.target_quotient_enclosure(center)
    assert math.isfinite(target.lower) and math.isfinite(target.upper)

    left = cert.legendre_spectral_enclosure(0.25, 0.0, 1.0, modes=20)
    right = cert.legendre_spectral_enclosure(0.25, 1.0, 1.0, modes=20)
    assert left.cdf == cert.Ball.point(0)
    assert right.cdf == cert.Ball.point(1)

    # Increasing the truncation length materially tightens the analytic tail.
    coarse = cert.legendre_spectral_enclosure(0.3, 0.6, 0.5, modes=12)
    fine = cert.legendre_spectral_enclosure(0.3, 0.6, 0.5, modes=20)
    assert fine.tail.cdf.upper < coarse.tail.cdf.upper
    assert fine.tail.density.upper < coarse.tail.density.upper
    assert fine.tail.conormal.upper < coarse.tail.conormal.upper
    assert fine.cdf.lower <= coarse.cdf.upper and coarse.cdf.lower <= fine.cdf.upper
    assert fine.density.lower <= coarse.density.upper and coarse.density.lower <= fine.density.upper


@pytest.mark.skipif(not cert.arb_available(), reason="python-flint/Arb unavailable")
@pytest.mark.parametrize(
    ("x", "y", "exposure"),
    [(0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.25, 0.75, 0.2), (0.5, 0.5, 2.0)],
)
def test_legendre_recurrence_cross_checks_against_arb(x: float, y: float, exposure: float) -> None:
    checked = cert.arb_cross_check_legendre(x, y, exposure, modes=24)
    assert checked["available"] == 1
    assert checked["passed"] == 1, checked


def test_certificate_and_fallback_codes_are_stable_and_complete() -> None:
    assert int(cert.CertificateBit.CDF_INVERSE) == 1
    assert int(cert.CertificateBit.DENSITY_POSITIVE) == 2
    assert int(cert.CertificateBit.TARGET_ENCLOSURE) == 4
    assert int(cert.CertificateBit.CORRECT_ROUNDING) == 8
    assert int(cert.CertificateBit.ALL) == 15
    assert cert.FallbackReason.NONE == 0
    assert len({int(value) for value in cert.FallbackReason}) == len(cert.FallbackReason)


def test_preflight_is_json_serializable_complete_and_faults_fail_closed() -> None:
    report = cert.run_certificate_arithmetic_preflight()
    json.dumps(report, allow_nan=False)
    assert report["passed"] == 1
    assert report["double_double_interval_algebra_pass"] == 1
    assert report["certified_exponential_pass"] == 1
    assert report["constant_fingerprint"]

    algebra_faults = ("rounding_cell", "two_sum", "two_prod", "ball", "quotient", "legendre", "constants")
    for fault in algebra_faults:
        failed = cert.run_certificate_arithmetic_preflight(faults=(fault,))
        assert failed["passed"] == 0
        assert failed["double_double_interval_algebra_pass"] == 0
        assert failed["subchecks"][fault] == 0
    exp_failed = cert.run_certificate_arithmetic_preflight(faults=("exp",))
    assert exp_failed["passed"] == 0
    assert exp_failed["certified_exponential_pass"] == 0
    unknown = cert.run_certificate_arithmetic_preflight(faults=("misspelled-fault",))
    assert unknown["passed"] == 0
    assert "unknown_faults" in unknown["errors"]


def test_nonfinite_and_invalid_inputs_never_authorize() -> None:
    for value in (math.inf, -math.inf, math.nan):
        with pytest.raises(cert.ArithmeticCertificationError):
            cert.rounding_cell(value)
        with pytest.raises(cert.ArithmeticCertificationError):
            cert.exp24_enclosure(value)
    with pytest.raises(ValueError):
        cert.legendre_spectral_enclosure(-0.1, 0.5, 1.0)
    with pytest.raises(ValueError):
        cert.legendre_spectral_enclosure(0.5, 1.1, 1.0)
    with pytest.raises(ValueError):
        cert.legendre_spectral_enclosure(0.5, 0.5, 0.0)

