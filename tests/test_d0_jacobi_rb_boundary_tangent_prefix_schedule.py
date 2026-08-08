from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from mnist.d0_jacobi_rb_boundary_tangent_prefix_schedule import (
    ADAPTIVE_PROFILE_NAME,
    EAGER_PREFIX_POLICY,
    EAGER_PROFILE_NAME,
    EagerPrefixScheduleError,
    PrefixProfileTimingRecord,
    adaptive_prefix_profile,
    eager_prefix_contract,
    eager_prefix_profile,
    qualify_eager_prefix_profile,
    validate_eager_prefix_equivalence,
)


def _batch(
    *,
    target_delta: float = 0.0,
    certificate_code: int = 0b1111,
    mode_count: int = 128,
    prefix_bits: int = 64,
) -> SimpleNamespace:
    later = np.asarray([0.125, 0.5, 0.875], dtype=np.float64)
    target = np.asarray([-0.25, 0.0, 0.25], dtype=np.float64)
    target = target + np.float64(target_delta)
    return SimpleNamespace(
        later_head_fraction=later,
        denoising_target=target,
        certificate_codes=np.full(3, certificate_code, dtype=np.uint8),
        mode_counts=np.full(3, mode_count, dtype=np.int32),
        prefix_bits=np.full(3, prefix_bits, dtype=np.int32),
    )


def _timing(
    repeat: int,
    *,
    adaptive: float = 10.0,
    eager: float = 8.0,
    fallback_count: int = 0,
    forbidden_count: int = 0,
) -> PrefixProfileTimingRecord:
    return PrefixProfileTimingRecord(
        repeat_index=repeat,
        adaptive_authorizer_seconds=adaptive,
        eager_authorizer_seconds=eager,
        scientific_output_equal=1,
        eager_prefix_policy_observed=1,
        eager_certificate_fraction=1.0,
        eager_fallback_count=fallback_count,
        eager_forbidden_count=forbidden_count,
    )


def test_eager_profile_changes_only_prefix_proof_effort() -> None:
    adaptive = adaptive_prefix_profile()
    eager = eager_prefix_profile()

    assert ADAPTIVE_PROFILE_NAME == "adaptive_prefix_64_then_128_tpb128"
    assert EAGER_PROFILE_NAME == "eager_prefix_128_tpb128"
    assert EAGER_PREFIX_POLICY == "same-philox-uniform-eager-second-word-at-m128"
    assert adaptive.candidate_modes == eager.candidate_modes == 128
    assert adaptive.candidate_bisection_steps == eager.candidate_bisection_steps == 56
    assert adaptive.threads_per_block == eager.threads_per_block == 128
    assert adaptive.max_prefix_bits == eager.max_prefix_bits == 1024
    assert adaptive.certificate_effort == "adaptive"
    assert eager.certificate_effort == "strengthened"

    # The convenience precision-doubling profile is deliberately not this
    # one-variable repair: it also changes the nonauthorizing proposal.
    assert adaptive.strengthened().candidate_modes == 256
    assert eager.candidate_modes != adaptive.strengthened().candidate_modes

    contract = eager_prefix_contract()
    assert contract["candidate_unchanged"] == 1
    assert contract["thread_geometry_unchanged"] == 1
    assert contract["same_philox_key_and_counter"] == 1
    assert contract["same_infinite_dyadic_uniform"] == 1
    assert contract["second_word_revealed_earlier_only"] == 1
    assert contract["transition_law_changed"] == 0
    assert contract["exposure_changed"] == 0
    assert contract["target_changed"] == 0
    assert contract["approximate_transition_used"] == 0


def test_equivalence_authorizes_scientific_outputs_not_proof_metadata() -> None:
    adaptive = _batch(mode_count=4112, prefix_bits=128)
    eager = _batch(mode_count=128, prefix_bits=128)
    result = validate_eager_prefix_equivalence(adaptive, eager)

    assert result["later_fraction_bit_identical"] == 1
    assert result["rb_target_bit_identical"] == 1
    assert result["certificate_codes_bit_identical"] == 1
    assert result["adaptive_certificate_fraction_one"] == 1
    assert result["eager_certificate_fraction_one"] == 1
    assert result["scientific_output_equivalent"] == 1
    assert result["certificate_metadata_equality_required"] == 0


def test_equivalence_fails_closed_on_target_state_or_certificate_change() -> None:
    reference = _batch()

    target_changed = validate_eager_prefix_equivalence(
        reference, _batch(target_delta=np.nextafter(0.0, 1.0))
    )
    assert target_changed["rb_target_bit_identical"] == 0
    assert target_changed["scientific_output_equivalent"] == 0

    later_changed = _batch()
    later_changed.later_head_fraction = later_changed.later_head_fraction.copy()
    later_changed.later_head_fraction[0] = np.nextafter(
        later_changed.later_head_fraction[0], 1.0
    )
    assert validate_eager_prefix_equivalence(reference, later_changed)[
        "scientific_output_equivalent"
    ] == 0

    uncertified = validate_eager_prefix_equivalence(
        reference, _batch(certificate_code=0b0111)
    )
    assert uncertified["eager_certificate_fraction_one"] == 0
    assert uncertified["scientific_output_equivalent"] == 0


def test_profile_qualification_uses_fastest_adaptive_and_slowest_eager() -> None:
    records = [
        _timing(2, adaptive=10.5, eager=7.5),
        _timing(0, adaptive=10.0, eager=8.0),
        _timing(1, adaptive=11.0, eager=7.0),
    ]
    result = qualify_eager_prefix_profile(
        records,
        parent_projected_seconds=120_000.0,
        parent_weighted_p10_authorizer_seconds=60_000.0,
    )
    assert result["fastest_adaptive_authorizer_seconds"] == 10.0
    assert result["slowest_eager_authorizer_seconds"] == 8.0
    assert result["conservative_authorizer_speedup"] == 1.25
    assert result["projected_elapsed_seconds"] == 108_000.0
    assert result["passed"] == 1
    assert [row["repeat_index"] for row in result["records"]] == [0, 1, 2]


def test_profile_runtime_boundary_is_fail_closed() -> None:
    exact = [_timing(index, adaptive=10.0, eager=8.0) for index in range(3)]
    assert qualify_eager_prefix_profile(
        exact,
        parent_projected_seconds=120_000.0,
        parent_weighted_p10_authorizer_seconds=60_000.0,
    )["passed"] == 1

    just_slow = [
        _timing(index, adaptive=10.0, eager=math.nextafter(8.0, math.inf))
        for index in range(3)
    ]
    failed = qualify_eager_prefix_profile(
        just_slow,
        parent_projected_seconds=120_000.0,
        parent_weighted_p10_authorizer_seconds=60_000.0,
    )
    assert failed["projected_elapsed_seconds"] > 108_000.0
    assert failed["passed"] == 0


def test_profile_qualification_requires_clean_fixed_replay() -> None:
    forbidden = [_timing(index, forbidden_count=int(index == 2)) for index in range(3)]
    result = qualify_eager_prefix_profile(
        forbidden,
        parent_projected_seconds=120_000.0,
        parent_weighted_p10_authorizer_seconds=60_000.0,
    )
    assert result["numerically_clean"] == 0
    assert result["passed"] == 0

    # The fixed profile panel requires zero fallback. A certified fallback is
    # legal in the later full pilot under its rate threshold, but it is not a
    # clean eager-prefix equivalence replay.
    fallback = [_timing(index, fallback_count=int(index == 1)) for index in range(3)]
    result = qualify_eager_prefix_profile(
        fallback,
        parent_projected_seconds=120_000.0,
        parent_weighted_p10_authorizer_seconds=60_000.0,
    )
    assert result["numerically_clean"] == 0
    assert result["passed"] == 0


@pytest.mark.parametrize(
    "records",
    [
        [_timing(0), _timing(1)],
        [_timing(0), _timing(1), _timing(1)],
    ],
)
def test_profile_qualification_requires_exact_repeat_set(records) -> None:
    with pytest.raises(EagerPrefixScheduleError):
        qualify_eager_prefix_profile(
            records,
            parent_projected_seconds=120_000.0,
            parent_weighted_p10_authorizer_seconds=60_000.0,
        )
