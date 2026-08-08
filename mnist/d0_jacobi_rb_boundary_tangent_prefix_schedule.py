"""Exact eager-prefix scheduling helpers for the boundary-tangent workflow.

The transition's stateless Philox stream already defines an infinite dyadic
uniform.  The legacy adaptive certificate first exposes 64 bits and reveals
the next word only after its primary spectral cap.  This module changes only
that proof schedule: the same second word is available at the first proof
bucket.  The Jacobi law, candidate, transition ID, rounded state, and
Rao--Blackwell target are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mnist.d0_jacobi_rb_boundary_tangent_schedule import PilotRepeatRecord
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile


PREFIX_SCHEDULE_VERSION = "d0-jacobi-rb-eager-prefix-128-v1"
EAGER_PREFIX_POLICY = "same-philox-uniform-eager-second-word-at-m128"
EAGER_PROFILE_NAME = "eager_prefix_128_tpb128"
ADAPTIVE_PROFILE_NAME = "adaptive_prefix_64_then_128_tpb128"
PROFILE_REPEAT_COUNT = 3
PROJECTED_TRANSITION_COUNT = 337_182_720
MAXIMUM_PROJECTED_SECONDS = 108_000.0
MINIMUM_EFFECTIVE_RATE = PROJECTED_TRANSITION_COUNT / MAXIMUM_PROJECTED_SECONDS


class EagerPrefixScheduleError(ValueError):
    """The exact eager-prefix scheduling contract was violated."""


def adaptive_prefix_profile() -> JacobiRBCudaProfile:
    """Return the immutable legacy profile used by the parent pilot."""

    return JacobiRBCudaProfile(
        candidate_modes=128,
        candidate_bisection_steps=56,
        threads_per_block=128,
        certificate_effort="adaptive",
    )


def eager_prefix_profile() -> JacobiRBCudaProfile:
    """Return the exact eager-128 profile without changing the proposal.

    Constructing the profile directly is intentional.  Calling
    :meth:`JacobiRBCudaProfile.strengthened` would also double the
    nonauthorizing candidate modes and would no longer be the one-variable
    scheduling repair being tested.
    """

    return replace(
        adaptive_prefix_profile(),
        certificate_effort="strengthened",
    )


def eager_prefix_contract() -> dict[str, Any]:
    adaptive = adaptive_prefix_profile()
    eager = eager_prefix_profile()
    return {
        "schema": PREFIX_SCHEDULE_VERSION + "-contract",
        "schema_version": 1,
        "policy": EAGER_PREFIX_POLICY,
        "adaptive_profile": adaptive.to_dict(),
        "eager_profile": eager.to_dict(),
        "candidate_unchanged": int(
            adaptive.candidate_modes == eager.candidate_modes == 128
            and adaptive.candidate_bisection_steps
            == eager.candidate_bisection_steps
            == 56
        ),
        "thread_geometry_unchanged": int(
            adaptive.threads_per_block == eager.threads_per_block == 128
        ),
        "same_philox_key_and_counter": 1,
        "same_infinite_dyadic_uniform": 1,
        "second_word_revealed_earlier_only": 1,
        "first_authorizing_prefix_bits": 128,
        "transition_law_changed": 0,
        "exposure_changed": 0,
        "target_changed": 0,
        "approximate_transition_used": 0,
    }


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _field(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    raise EagerPrefixScheduleError(f"missing eager-prefix comparison field: {names[0]}")


def validate_eager_prefix_equivalence(adaptive: Any, eager: Any) -> dict[str, Any]:
    """Compare scientific outputs while allowing proof metadata to differ."""

    adaptive_y = _as_numpy(
        _field(adaptive, "later_head_fraction", "later_full_state", "later")
    )
    eager_y = _as_numpy(
        _field(eager, "later_head_fraction", "later_full_state", "later")
    )
    adaptive_z = _as_numpy(
        _field(adaptive, "denoising_target", "target")
    )
    eager_z = _as_numpy(_field(eager, "denoising_target", "target"))
    adaptive_codes = _as_numpy(
        _field(adaptive, "certificate_codes", "certificate_code")
    )
    eager_codes = _as_numpy(
        _field(eager, "certificate_codes", "certificate_code")
    )
    shape_equal = adaptive_y.shape == eager_y.shape and adaptive_z.shape == eager_z.shape
    later_equal = bool(shape_equal and np.array_equal(adaptive_y, eager_y))
    target_equal = bool(shape_equal and np.array_equal(adaptive_z, eager_z))
    code_equal = bool(
        adaptive_codes.shape == eager_codes.shape
        and np.array_equal(adaptive_codes, eager_codes)
    )
    adaptive_certified = bool(np.all((adaptive_codes.astype(np.uint8) & 0b1111) == 0b1111))
    eager_certified = bool(np.all((eager_codes.astype(np.uint8) & 0b1111) == 0b1111))
    return {
        "schema": PREFIX_SCHEDULE_VERSION + "-equivalence",
        "schema_version": 1,
        "shape_equal": int(shape_equal),
        "later_fraction_bit_identical": int(later_equal),
        "rb_target_bit_identical": int(target_equal),
        "certificate_codes_bit_identical": int(code_equal),
        "adaptive_certificate_fraction_one": int(adaptive_certified),
        "eager_certificate_fraction_one": int(eager_certified),
        "scientific_output_equivalent": int(
            shape_equal and later_equal and target_equal and code_equal
            and adaptive_certified and eager_certified
        ),
        "certificate_metadata_equality_required": 0,
    }


@dataclass(frozen=True)
class PrefixProfileTimingRecord:
    """One paired adaptive/eager authorizer timing observation."""

    repeat_index: int
    adaptive_authorizer_seconds: float
    eager_authorizer_seconds: float
    scientific_output_equal: int
    eager_prefix_policy_observed: int
    eager_certificate_fraction: float
    eager_fallback_count: int
    eager_forbidden_count: int

    def __post_init__(self) -> None:
        if not 0 <= int(self.repeat_index) < PROFILE_REPEAT_COUNT:
            raise EagerPrefixScheduleError("profile repeat index is outside the frozen panel")
        for value in (self.adaptive_authorizer_seconds, self.eager_authorizer_seconds):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise EagerPrefixScheduleError("profile timings must be finite and positive")
        if int(self.scientific_output_equal) != 1:
            raise EagerPrefixScheduleError("profile outputs are not scientifically identical")
        if int(self.eager_prefix_policy_observed) != 1:
            raise EagerPrefixScheduleError("profile did not observe the eager-prefix policy")
        if float(self.eager_certificate_fraction) != 1.0:
            raise EagerPrefixScheduleError("eager profile did not certify every transition")
        if int(self.eager_fallback_count) < 0 or int(self.eager_forbidden_count) < 0:
            raise EagerPrefixScheduleError("profile counts must be nonnegative")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PREFIX_SCHEDULE_VERSION + "-profile-timing",
            "schema_version": 1,
            "repeat_index": int(self.repeat_index),
            "adaptive_authorizer_seconds": float(self.adaptive_authorizer_seconds),
            "eager_authorizer_seconds": float(self.eager_authorizer_seconds),
            "scientific_output_equal": int(self.scientific_output_equal),
            "eager_prefix_policy_observed": int(self.eager_prefix_policy_observed),
            "eager_certificate_fraction": float(self.eager_certificate_fraction),
            "eager_fallback_count": int(self.eager_fallback_count),
            "eager_forbidden_count": int(self.eager_forbidden_count),
        }


def qualify_eager_prefix_profile(
    records: Sequence[PrefixProfileTimingRecord],
    *,
    parent_projected_seconds: float,
    parent_weighted_p10_authorizer_seconds: float,
) -> dict[str, Any]:
    """Compute the frozen conservative profile-stage runtime forecast."""

    ordered = sorted(records, key=lambda value: int(value.repeat_index))
    if [value.repeat_index for value in ordered] != list(range(PROFILE_REPEAT_COUNT)):
        raise EagerPrefixScheduleError("profile qualification requires exactly three repeats")
    parent_seconds = float(parent_projected_seconds)
    weighted = float(parent_weighted_p10_authorizer_seconds)
    if (
        not math.isfinite(parent_seconds)
        or not math.isfinite(weighted)
        or parent_seconds <= 0.0
        or not 0.0 < weighted <= parent_seconds
    ):
        raise EagerPrefixScheduleError("parent timing decomposition is invalid")
    fastest_adaptive = min(value.adaptive_authorizer_seconds for value in ordered)
    slowest_eager = max(value.eager_authorizer_seconds for value in ordered)
    conservative_speedup = fastest_adaptive / slowest_eager
    saved = (
        weighted * (1.0 - 1.0 / conservative_speedup)
        if conservative_speedup > 1.0
        else 0.0
    )
    projected = parent_seconds - saved
    effective = PROJECTED_TRANSITION_COUNT / projected
    clean = all(
        value.scientific_output_equal == 1
        and value.eager_prefix_policy_observed == 1
        and value.eager_certificate_fraction == 1.0
        and value.eager_fallback_count == 0
        and value.eager_forbidden_count == 0
        for value in ordered
    )
    passed = (
        clean
        and projected <= MAXIMUM_PROJECTED_SECONDS
        and effective >= MINIMUM_EFFECTIVE_RATE
    )
    return {
        "schema": PREFIX_SCHEDULE_VERSION + "-profile-qualification",
        "schema_version": 1,
        "profile_name": EAGER_PROFILE_NAME,
        "records": [value.to_record() for value in ordered],
        "fastest_adaptive_authorizer_seconds": fastest_adaptive,
        "slowest_eager_authorizer_seconds": slowest_eager,
        "conservative_authorizer_speedup": conservative_speedup,
        "parent_projected_seconds": parent_seconds,
        "parent_weighted_p10_authorizer_seconds": weighted,
        "conservative_saved_seconds": saved,
        "projected_elapsed_seconds": projected,
        "projected_effective_transitions_per_second": effective,
        "maximum_projected_seconds": MAXIMUM_PROJECTED_SECONDS,
        "minimum_effective_transitions_per_second": MINIMUM_EFFECTIVE_RATE,
        "numerically_clean": int(clean),
        "passed": int(passed),
    }


PrefixScheduleRepeatRecord = PilotRepeatRecord


__all__ = [
    "ADAPTIVE_PROFILE_NAME",
    "EAGER_PREFIX_POLICY",
    "EAGER_PROFILE_NAME",
    "EagerPrefixScheduleError",
    "PREFIX_SCHEDULE_VERSION",
    "PrefixProfileTimingRecord",
    "PrefixScheduleRepeatRecord",
    "adaptive_prefix_profile",
    "eager_prefix_contract",
    "eager_prefix_profile",
    "qualify_eager_prefix_profile",
    "validate_eager_prefix_equivalence",
]
