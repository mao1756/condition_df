"""Versioned exact Arb escalation for the eager-prefix scheduler.

The immutable CUDA backend first asks a cheap candidate-local Arb search to
certify an unresolved lane.  That search is intentionally bounded to a
small binary64 neighbourhood.  The eager-prefix control discovered a valid
lane whose nonauthorizing CUDA proposal was farther away than that bound.

This additive adapter preserves the immutable backend files and its fast
path.  Only when the local lattice is exhausted does it run the existing
full certified Arb inverse CDF and target evaluator on the same stateless
Philox stream.  No approximate value can be returned by this module.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
import threading
from typing import Any, Iterator, Mapping

from mnist import d0_jacobi_rb_cuda as _backend
from mnist import d0_jacobi_rb_spectral as _reference


EAGER_ARB_ESCALATION_VERSION = "eager-prefix-arb-full-inversion-v1"
EAGER_ARB_INITIAL_PREFIX_BITS = 128

_PARENT_ARB_WORKER = _backend._arb_candidate_cell_worker
_WORKER_PATCH_LOCK = threading.RLock()


class EagerPrefixArbEscalationError(_reference.JacobiRBCertificationError):
    """The exact eager-prefix Arb escalation failed closed."""


def _candidate_lattice_exhausted(exc: BaseException) -> bool:
    diagnostics = getattr(exc, "diagnostics", {})
    return bool(
        isinstance(exc, _reference.JacobiRBCertificationError)
        and isinstance(diagnostics, Mapping)
        and diagnostics.get("failure_kind") == "arb_resource_cap"
        and diagnostics.get("resource_kind") == "candidate_lattice"
    )


def _refine_to_eager_prefix(prefix: Any) -> None:
    bits = int(prefix.bits)
    if bits >= EAGER_ARB_INITIAL_PREFIX_BITS:
        return
    if int(prefix.max_bits) < EAGER_ARB_INITIAL_PREFIX_BITS:
        raise EagerPrefixArbEscalationError(
            "eager Arb prefix cannot reveal the frozen initial 128 bits",
            {
                "failure_kind": "eager_prefix_unavailable",
                "prefix_bits": bits,
                "maximum_prefix_bits": int(prefix.max_bits),
            },
        )
    prefix.refine(EAGER_ARB_INITIAL_PREFIX_BITS - bits)


def _prefix_from_payload(payload: Mapping[str, Any]) -> Any:
    kind = str(payload["prefix_kind"])
    if kind == "parent-v1-verified-continuation":
        candidate = int(payload["v1_key_candidate"])
        key = (261_121, "support-prefix", candidate)
        prefix = _reference._LazyDyadicPrefix(
            _reference._key_bytes(key),
            0,
            initial_bits=64,
            max_bits=int(payload["max_prefix_bits"]),
        )
        if int(prefix.numerator) != int(payload["prefix_numerator"]):
            raise EagerPrefixArbEscalationError(
                "parent-v1 continuation does not match its recorded first word",
                {"failure_kind": "parent_v1_prefix_mismatch"},
            )
    elif kind == "recorded":
        # A fixed historical prefix has no legal continuation.  The parent
        # local worker remains the sole authority for such a row.
        raise EagerPrefixArbEscalationError(
            "recorded fixed prefix cannot enter eager full inversion",
            {
                "failure_kind": "recorded_prefix_exhausted",
                "prefix_bits": int(payload["prefix_bits"]),
            },
        )
    elif kind == "philox-v2":
        prefix = _backend._StatelessPhiloxPrefix(
            int(payload["seed"]),
            int(payload["transition_id"]),
            int(payload["max_prefix_bits"]),
            seed_is_canonical=True,
        )
    else:
        raise EagerPrefixArbEscalationError(
            "unknown eager Arb prefix kind",
            {"failure_kind": "prefix_kind", "prefix_kind": kind},
        )
    _refine_to_eager_prefix(prefix)
    return prefix


def eager_prefix_arb_candidate_cell_worker(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the fast local certificate, then exact full Arb inversion.

    This function is top-level and spawn-safe because the immutable backend
    submits it to a persistent :class:`ProcessPoolExecutor`.
    """

    try:
        row = dict(_PARENT_ARB_WORKER(payload))
        # Revealing the same second Philox word cannot invalidate a proof
        # already valid for the enclosing 64-bit interval.  Retain truthful
        # eager-schedule telemetry for the returned certified row.
        if str(payload.get("prefix_kind")) != "recorded":
            row["prefix_bits"] = max(
                int(row.get("prefix_bits", 0)), EAGER_ARB_INITIAL_PREFIX_BITS
            )
        row["fallback_strategy"] = "candidate_local_lattice"
        row["full_arb_inversion_used"] = 0
        return row
    except Exception as exc:
        if not _candidate_lattice_exhausted(exc):
            raise
        local_diagnostics = dict(getattr(exc, "diagnostics", {}))

    x_value = float(payload["x"])
    exposure = float(payload["exposure"])
    profile = payload["profile"]
    prefix = _prefix_from_payload(payload)
    try:
        (
            later,
            _quantile_lower,
            _quantile_upper,
            _steps,
            quantile_modes,
            _escalations,
            correctly_rounded,
        ) = _reference._invert_one(x_value, exposure, prefix, profile)
        if not correctly_rounded:
            raise EagerPrefixArbEscalationError(
                "full Arb inversion did not certify a binary64 quantile",
                {"failure_kind": "quantile_not_correctly_rounded"},
            )
        target, target_interval, target_modes, _target_escalated = (
            _reference._target_interval(
                x_value, float(later), exposure, profile
            )
        )
    except Exception as exc:
        diagnostics = dict(getattr(exc, "diagnostics", {}))
        raise EagerPrefixArbEscalationError(
            "full eager-prefix Arb inversion failed closed",
            {
                "failure_kind": "full_arb_inversion_failed",
                "transition_id": int(payload["transition_id"]),
                "local_lattice_diagnostics": local_diagnostics,
                **diagnostics,
            },
        ) from exc

    if not (
        math.isfinite(float(later))
        and math.isfinite(float(target))
        and 0.0 <= float(later) <= 1.0
    ):
        raise EagerPrefixArbEscalationError(
            "full eager-prefix Arb inversion returned a nonfinite value",
            {
                "failure_kind": "full_arb_nonfinite",
                "transition_id": int(payload["transition_id"]),
            },
        )
    lower_boundary, upper_boundary = _reference._rounding_cell(float(later))
    return {
        "later": float(later),
        "target": float(target),
        "quantile_lower": _backend._fraction_down(lower_boundary),
        "quantile_upper": _backend._fraction_up(upper_boundary),
        "target_lower": min(float(target), float(target_interval.lower)),
        "target_upper": max(float(target), float(target_interval.upper)),
        "prefix_bits": int(prefix.bits),
        "modes": max(int(quantile_modes), int(target_modes)),
        "certificate_code": 15,
        "fallback_strategy": "full_arb_inversion",
        "full_arb_inversion_used": 1,
        "candidate_lattice_diagnostics": local_diagnostics,
        "escalation_version": EAGER_ARB_ESCALATION_VERSION,
    }


@contextmanager
def _installed_eager_worker() -> Iterator[None]:
    """Install the additive worker for one complete immutable API call."""

    with _WORKER_PATCH_LOCK:
        previous = _backend._arb_candidate_cell_worker
        _backend._arb_candidate_cell_worker = eager_prefix_arb_candidate_cell_worker
        try:
            yield
        finally:
            _backend._arb_candidate_cell_worker = previous


def sample_alpha1_rb_transition_batch_cuda_eager(
    head_fraction: Any,
    exposure: Any,
    *,
    rng_key: Any,
    transition_ids: Any,
    profile: Any,
    **kwargs: Any,
) -> Any:
    """Call the immutable CUDA API with the versioned exact Arb escalation."""

    with _installed_eager_worker():
        return _backend.sample_alpha1_rb_transition_batch_cuda(
            head_fraction,
            exposure,
            rng_key=rng_key,
            transition_ids=transition_ids,
            profile=profile,
            **kwargs,
        )


__all__ = [
    "EAGER_ARB_ESCALATION_VERSION",
    "EAGER_ARB_INITIAL_PREFIX_BITS",
    "EagerPrefixArbEscalationError",
    "eager_prefix_arb_candidate_cell_worker",
    "sample_alpha1_rb_transition_batch_cuda_eager",
]
