"""Schedule-independent semantics for certified Jacobi/RB transitions.

Adaptive and eager proof schedules can return the same correctly rounded
transition through different mode counts, prefix lengths, enclosure widths,
or CUDA/Arb routes.  Those fields remain useful evidence, but they are not
part of the represented transition.  This module compares the scientific
payload and a normalized authorization mask while retaining proof metadata as
advisory diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch


CERTIFICATE_SEMANTICS_COMPARATOR_VERSION = (
    "d0-jacobi-rb-certificate-semantics-comparator-v1"
)
CERTIFICATE_SEMANTICS_PAYLOAD_ADAPTER_VERSION = (
    "d0-jacobi-rb-certificate-semantics-payload-adapter-v1"
)
DEFAULT_MAXIMUM_MISMATCH_RECORDS = 16
MAXIMUM_MISMATCH_RECORDS = 64


class CertificateSemanticsError(ValueError):
    """A certificate payload is malformed and cannot be compared safely."""


_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "transition_ids": ("transition_ids", "transition_id"),
    "earlier_head_fraction": (
        "earlier_head_fraction",
        "earlier_fraction",
        "earlier_state",
    ),
    "exposure": ("exposure",),
    "later_head_fraction": (
        "later_head_fraction",
        "later_fraction",
        "later",
    ),
    "denoising_target": ("denoising_target", "target"),
    "active_mask": ("active_mask",),
    "certified_mask": ("certified_mask", "authorized_mask"),
}

_PROOF_FIELDS: dict[str, tuple[str, ...]] = {
    "certificate_codes": ("certificate_codes", "certificate_code"),
    "prefix_bits": ("prefix_bits",),
    "mode_counts": ("mode_counts", "modes_used"),
    "strengthened_mask": ("strengthened_mask",),
    "cuda_certified_mask": ("cuda_certified_mask", "cuda_authorized_mask"),
    "fallback_mask": ("fallback_mask",),
    "fallback_reason_codes": (
        "arb_fallback_reason_codes",
        "fallback_reason_codes",
    ),
    "fallback_mode_counts": ("arb_fallback_mode_counts",),
    "quantile_lower": ("quantile_lower",),
    "quantile_upper": ("quantile_upper",),
    "target_lower": ("target_lower",),
    "target_upper": ("target_upper",),
    "candidate_later_head_fraction": ("candidate_later_head_fraction",),
    "candidate_denoising_target": ("candidate_denoising_target",),
    "candidate_match_mask": ("candidate_match_mask",),
}


def _field(value: Any, names: Sequence[str], *, required: bool) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    if required:
        raise CertificateSemanticsError(
            f"certificate payload is missing required field {names[0]}"
        )
    return None


def _array(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().contiguous().numpy()
    else:
        result = np.ascontiguousarray(np.asarray(value))
    if result.dtype.hasobject:
        raise CertificateSemanticsError(f"{name} must not have object dtype")
    return np.ascontiguousarray(result)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _record_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_scalar(value: Any) -> Any:
    scalar = value.item() if isinstance(value, np.generic) else value
    if isinstance(scalar, float):
        if math.isnan(scalar):
            return "nan"
        if math.isinf(scalar):
            return "inf" if scalar > 0.0 else "-inf"
        return float(scalar)
    if isinstance(scalar, (bool, int, str)) or scalar is None:
        return scalar
    return repr(scalar)


def _proof_summary(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    flat = array.reshape(-1)
    record: dict[str, Any] = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": _array_sha256(array),
        "element_count": int(array.size),
    }
    if not flat.size:
        return record
    if array.dtype.kind in "biu":
        unique, counts = np.unique(flat, return_counts=True)
        limit = 32
        record.update(
            {
                "minimum": _json_scalar(unique[0]),
                "maximum": _json_scalar(unique[-1]),
                "unique_value_count": int(unique.size),
                "histogram": {
                    str(_json_scalar(key)): int(count)
                    for key, count in zip(
                        unique[:limit], counts[:limit], strict=True
                    )
                },
                "histogram_truncated": int(unique.size > limit),
            }
        )
    elif array.dtype.kind in "fc":
        finite = np.isfinite(flat)
        record.update(
            {
                "finite_count": int(np.count_nonzero(finite)),
                "nonfinite_count": int(flat.size - np.count_nonzero(finite)),
                "minimum_finite": (
                    float(np.min(flat[finite])) if np.any(finite) else None
                ),
                "maximum_finite": (
                    float(np.max(flat[finite])) if np.any(finite) else None
                ),
            }
        )
    return record


def _view(value: Any) -> dict[str, Any]:
    result = {
        name: _array(_field(value, aliases, required=True), name)
        for name, aliases in _REQUIRED_FIELDS.items()
    }
    shape = result["transition_ids"].shape
    if any(array.shape != shape for array in result.values()):
        details = {name: list(array.shape) for name, array in result.items()}
        raise CertificateSemanticsError(
            f"certificate payload fields do not share one transition shape: {details}"
        )
    if result["transition_ids"].dtype.kind not in "iu":
        raise CertificateSemanticsError("transition_ids must have integer dtype")
    for name in (
        "earlier_head_fraction",
        "exposure",
        "later_head_fraction",
        "denoising_target",
    ):
        if result[name].dtype != np.dtype(np.float64):
            raise CertificateSemanticsError(f"{name} must have float64 dtype")
    for name in ("active_mask", "certified_mask"):
        mask = result[name]
        if mask.dtype.kind not in "biu" or np.any((mask != 0) & (mask != 1)):
            raise CertificateSemanticsError(
                f"{name} must be boolean or an integer 0/1 mask"
            )
        result[name] = mask.astype(bool, copy=False)
    proof: dict[str, np.ndarray] = {}
    for name, aliases in _PROOF_FIELDS.items():
        raw = _field(value, aliases, required=False)
        if raw is None:
            continue
        array = _array(raw, name)
        if array.shape != shape:
            raise CertificateSemanticsError(
                f"proof telemetry {name} has shape {array.shape}, expected {shape}"
            )
        if name in {
            "quantile_lower",
            "quantile_upper",
            "target_lower",
            "target_upper",
        } and array.dtype != np.dtype(np.float64):
            raise CertificateSemanticsError(
                f"proof telemetry {name} must have float64 dtype"
            )
        proof[name] = array
    result["proof"] = proof
    return result


def _authorization_valid(view: Mapping[str, Any]) -> tuple[bool, dict[str, int]]:
    active = np.asarray(view["active_mask"], dtype=bool)
    certified = np.asarray(view["certified_mask"], dtype=bool)
    transition_ids = np.asarray(view["transition_ids"])
    earlier = np.asarray(view["earlier_head_fraction"])
    later = np.asarray(view["later_head_fraction"])
    target = np.asarray(view["denoising_target"])
    exposure = np.asarray(view["exposure"])
    counts = {
        "active_duplicate_transition_id_count": int(
            np.count_nonzero(active)
            - np.unique(transition_ids[active]).size
        ),
        "active_uncertified_count": int(np.count_nonzero(active & ~certified)),
        "inactive_certified_count": int(np.count_nonzero(~active & certified)),
        "active_nonfinite_output_count": int(
            np.count_nonzero(active & (~np.isfinite(later) | ~np.isfinite(target)))
        ),
        "inactive_state_change_count": int(np.count_nonzero(~active & (later != earlier))),
        "inactive_nonzero_target_count": int(np.count_nonzero(~active & (target != 0.0))),
        "active_exposure_nonpositive_count": int(np.count_nonzero(active & (exposure <= 0.0))),
        "inactive_exposure_nonzero_count": int(np.count_nonzero(~active & (exposure != 0.0))),
    }
    proof = view["proof"]
    enclosure_names = ("quantile_lower", "quantile_upper", "target_lower", "target_upper")
    present = tuple(name in proof for name in enclosure_names)
    if any(present) and not all(present):
        raise CertificateSemanticsError(
            "quantile and target enclosures must be supplied together"
        )
    if all(present):
        qlo = proof["quantile_lower"]
        qhi = proof["quantile_upper"]
        zlo = proof["target_lower"]
        zhi = proof["target_upper"]
        counts["active_invalid_enclosure_count"] = int(
            np.count_nonzero(
                active
                & (
                    ~np.isfinite(qlo)
                    | ~np.isfinite(qhi)
                    | ~np.isfinite(zlo)
                    | ~np.isfinite(zhi)
                    | (qlo > later)
                    | (later > qhi)
                    | (zlo > target)
                    | (target > zhi)
                )
            )
        )
    else:
        counts["active_invalid_enclosure_count"] = 0
    return not any(counts.values()), counts


def comparator_payload_from_certified_cuda_batch(batch: Any) -> dict[str, Any]:
    """Expose a complete ``CertifiedRBCudaBatch`` as comparator input.

    The adapter is intentionally structural, so callers can use a compatible
    test double without importing or initializing the CUDA backend.  All
    scientific fields and all retained proof telemetry are validated by
    :func:`_view` before the payload is returned.
    """

    payload = {
        name: _field(batch, aliases, required=True)
        for name, aliases in _REQUIRED_FIELDS.items()
    }
    retained: list[str] = []
    for name, aliases in _PROOF_FIELDS.items():
        value = _field(batch, aliases, required=False)
        if value is not None:
            payload[name] = value
            retained.append(name)
    payload["payload_adapter"] = CERTIFICATE_SEMANTICS_PAYLOAD_ADAPTER_VERSION
    payload["payload_source"] = "certified_cuda_batch"
    payload["retained_proof_metadata_fields"] = retained
    payload["unavailable_proof_metadata_fields"] = sorted(
        set(_PROOF_FIELDS) - set(retained)
    )
    _view(payload)
    return payload


def _midpoint_scientific_batch(value: Any) -> tuple[Any, Any | None]:
    """Return a midpoint scientific batch and its optional fused wrapper."""

    holder = _field(value, ("batch",), required=False)
    if holder is None:
        return value, None
    nested = _field(holder, ("batch",), required=False)
    if nested is None:
        return holder, None
    return nested, holder


def comparator_payload_from_eager_midpoint_branch(
    branch: Any,
    *,
    pre_phase_states: Any | None = None,
) -> dict[str, Any]:
    """Reconstruct exact comparator lanes from an eager midpoint branch.

    ``EagerBranchExecution`` supplies ``pre_phase_states`` itself.  A bare
    ``MidpointBranchBatch`` or ``FusedMidpointBranchBatch`` may instead pass
    them explicitly.  The resulting layout is ``[midpoint,path,edge]`` and
    therefore matches the scheduler's canonical fused-lane order.
    """

    from mnist.d0_jacobi_rb_boundary_tangent_cache import (
        MIDPOINT_FRACTIONS,
        phase_base_exposure,
    )
    from mnist.d0_jacobi_rb_learnability import (
        EDGES_PER_PHASE,
        PHASE_MATCHINGS,
        matching_indices,
    )
    from mnist.d0_jacobi_rb_reverse_controller import controller_transition_ids

    scientific, fused = _midpoint_scientific_batch(branch)
    if pre_phase_states is None:
        pre_phase_states = _field(
            branch, ("pre_phase_states",), required=False
        )
    if not isinstance(pre_phase_states, torch.Tensor):
        raise CertificateSemanticsError(
            "midpoint conversion requires pre_phase_states"
        )
    if (
        pre_phase_states.dtype != torch.float64
        or pre_phase_states.ndim != 2
        or not pre_phase_states.is_contiguous()
    ):
        raise CertificateSemanticsError(
            "pre_phase_states must be contiguous float64 [path,state]"
        )
    path_ids = tuple(
        int(item) for item in _field(scientific, ("path_ids",), required=True)
    )
    if len(path_ids) != int(pre_phase_states.shape[0]):
        raise CertificateSemanticsError(
            "midpoint path IDs do not match pre-phase states"
        )
    outer_step = int(_field(scientific, ("outer_step",), required=True))
    phase = int(_field(scientific, ("phase",), required=True))
    fractions = tuple(
        float(item)
        for item in _field(scientific, ("midpoint_fractions",), required=True)
    )
    if fractions != tuple(float(item) for item in MIDPOINT_FRACTIONS):
        raise CertificateSemanticsError("midpoint fractions changed")
    device = pre_phase_states.device
    tails_all, heads_all = matching_indices(device=device)
    try:
        matching = int(PHASE_MATCHINGS[phase])
    except (IndexError, TypeError) as exc:
        raise CertificateSemanticsError("midpoint phase is invalid") from exc
    tails = tails_all[matching]
    heads = heads_all[matching]
    if tails.numel() != EDGES_PER_PHASE or heads.numel() != EDGES_PER_PHASE:
        raise CertificateSemanticsError("midpoint matching size changed")
    tail_mass = pre_phase_states.index_select(1, tails)
    head_mass = pre_phase_states.index_select(1, heads)
    pair_mass = tail_mass + head_mass
    positive = pair_mass > 0.0
    safe = torch.where(positive, pair_mass, torch.ones_like(pair_mass))
    earlier = torch.where(
        positive, head_mass / safe, torch.zeros_like(pair_mass)
    )
    full_exposure = phase_base_exposure(pair_mass, phase)
    midpoint_count = len(fractions)
    shape = (midpoint_count, len(path_ids), EDGES_PER_PHASE)
    earlier_lanes = earlier.unsqueeze(0).expand(shape).contiguous()
    fraction_tensor = torch.as_tensor(
        fractions, dtype=torch.float64, device=device
    ).reshape(midpoint_count, 1, 1)
    exposure_lanes = (full_exposure.unsqueeze(0) * fraction_tensor).contiguous()
    id_lanes = torch.stack(
        tuple(
            controller_transition_ids(
                path_ids,
                outer_step=outer_step,
                phase=phase,
                reverse_microstep=index,
                role="partial_phase_target_prefix",
                device=device,
            )
            for index in range(midpoint_count)
        )
    ).reshape(shape).contiguous()
    active = exposure_lanes > 0.0
    later = _field(
        scientific, ("later_head_fraction",), required=True
    )
    target = _field(scientific, ("denoising_target",), required=True)
    codes = _field(scientific, ("certificate_codes",), required=True)
    if not all(isinstance(item, torch.Tensor) for item in (later, target, codes)):
        raise CertificateSemanticsError(
            "midpoint scientific outputs must be device tensors"
        )
    if any(item.shape != shape for item in (later, target, codes)):
        raise CertificateSemanticsError("midpoint scientific output shape changed")
    certified = active & ((codes.to(torch.uint8) & 0x0F) == 0x0F)
    payload: dict[str, Any] = {
        "transition_ids": id_lanes,
        "earlier_head_fraction": earlier_lanes,
        "exposure": exposure_lanes,
        "later_head_fraction": later,
        "denoising_target": target,
        "active_mask": active,
        "certified_mask": certified,
        "certificate_codes": codes,
        "payload_adapter": CERTIFICATE_SEMANTICS_PAYLOAD_ADAPTER_VERSION,
        "payload_source": "eager_midpoint_branch",
        "final_state": _field(
            scientific, ("later_full_state",), required=True
        ),
    }
    retained = ["certificate_codes"]
    for name in (
        "mode_counts",
        "prefix_bits",
        "fallback_mask",
        "strengthened_mask",
    ):
        value = _field(scientific, (name,), required=False)
        if value is not None:
            payload[name] = value
            retained.append(name)
    fallback_reasons = _field(
        fused, ("fallback_reason_codes",), required=False
    ) if fused is not None else None
    if fallback_reasons is not None:
        payload["fallback_reason_codes"] = fallback_reasons
        retained.append("fallback_reason_codes")
    payload["retained_proof_metadata_fields"] = retained
    payload["unavailable_proof_metadata_fields"] = sorted(
        set(_PROOF_FIELDS) - set(retained)
    )
    _view(payload)
    return payload


def comparator_payload_from_multipath_capture(
    capture: Any,
    *,
    initial_states: Any,
    initial_state_path_ids: Sequence[int],
) -> dict[str, Any]:
    """Rehydrate a complete base-transition payload from a raw capture.

    ``EagerShardExecution`` does not retain this capture, so this adapter is
    usable only while the underlying ``ExactMultipathShardResult`` (or its raw
    capture) is still available.  Explicit initial-state IDs remove any
    ambiguity caused by the capture's canonical path sorting.
    """

    from mnist.d0_jacobi_rb_cuda_multipath import (
        EDGES_PER_PHASE,
        PHASE_DURATIONS,
        PHASE_MATCHINGS,
        canonical_same_phase_transition_ids,
    )
    from mnist.d0_jacobi_rb_learnability import matching_indices

    raw_capture = _field(capture, ("capture_payload",), required=False)
    if raw_capture is not None:
        capture = raw_capture
    path_ids = tuple(
        int(item) for item in _field(capture, ("path_ids",), required=True)
    )
    provided_ids = tuple(int(item) for item in initial_state_path_ids)
    if len(provided_ids) != len(set(provided_ids)) or set(provided_ids) != set(
        path_ids
    ):
        raise CertificateSemanticsError(
            "initial-state path IDs must equal captured path IDs"
        )
    initial = _array(initial_states, "initial_states")
    if initial.dtype != np.dtype(np.float64) or initial.shape != (
        len(provided_ids),
        28 * 28,
    ):
        raise CertificateSemanticsError(
            "initial states must be float64 [path,784]"
        )
    by_id = {path_id: index for index, path_id in enumerate(provided_ids)}
    initial = np.ascontiguousarray(
        initial[[by_id[path_id] for path_id in path_ids]], dtype=np.float64
    )
    later = _array(
        _field(capture, ("later_head_fractions",), required=True),
        "later_head_fractions",
    )
    target = _array(
        _field(capture, ("denoising_targets",), required=True),
        "denoising_targets",
    )
    codes = _array(
        _field(capture, ("certificate_codes",), required=True),
        "certificate_codes",
    )
    post_states = _array(
        _field(capture, ("post_phase_states",), required=True),
        "post_phase_states",
    )
    outer_steps = tuple(
        int(item) for item in _field(capture, ("outer_steps",), required=True)
    )
    phases = tuple(
        int(item) for item in _field(capture, ("phases",), required=True)
    )
    blocks = len(outer_steps)
    shape = (blocks, len(path_ids), EDGES_PER_PHASE)
    if (
        blocks == 0
        or len(phases) != blocks
        or later.dtype != np.dtype(np.float64)
        or target.dtype != np.dtype(np.float64)
        or later.shape != shape
        or target.shape != shape
        or codes.shape != shape
        or post_states.dtype != np.dtype(np.float64)
        or post_states.shape != (blocks, len(path_ids), 28 * 28)
    ):
        raise CertificateSemanticsError("multipath capture shape or dtype changed")
    tails_all, heads_all = matching_indices(device="cpu")
    tails_array = tails_all.numpy()
    heads_array = heads_all.numpy()
    earlier = np.empty(shape, dtype=np.float64)
    exposure = np.empty(shape, dtype=np.float64)
    transition_ids = np.empty(shape, dtype=np.uint64)
    active = np.empty(shape, dtype=bool)
    pre_states = initial
    for block, (outer_step, phase) in enumerate(
        zip(outer_steps, phases, strict=True)
    ):
        if not 0 <= phase < len(PHASE_MATCHINGS):
            raise CertificateSemanticsError("captured phase is invalid")
        matching = int(PHASE_MATCHINGS[phase])
        tails = tails_array[matching]
        heads = heads_array[matching]
        tail_mass = pre_states[:, tails]
        head_mass = pre_states[:, heads]
        pair_mass = tail_mass + head_mass
        positive = pair_mass > 0.0
        safe = np.where(positive, pair_mass, 1.0)
        earlier[block] = np.where(positive, head_mass / safe, 0.0)
        coefficient = (
            3.0
            * (5.0e-5 / 512.0)
            * float(PHASE_DURATIONS[phase])
            / (1.0 / 28.0) ** 2
        )
        exposure[block] = np.where(positive, coefficient / safe, 0.0)
        transition_ids[block] = (
            canonical_same_phase_transition_ids(
                path_ids,
                outer_step=outer_step,
                phase=phase,
                device=torch.device("cpu"),
            )
            .reshape(len(path_ids), EDGES_PER_PHASE)
            .numpy()
        )
        active[block] = positive
        pre_states = post_states[block]
    certified = active & ((codes.astype(np.uint8, copy=False) & 0x0F) == 0x0F)
    payload = {
        "transition_ids": transition_ids,
        "earlier_head_fraction": earlier,
        "exposure": exposure,
        "later_head_fraction": later,
        "denoising_target": target,
        "active_mask": active,
        "certified_mask": certified,
        "certificate_codes": codes,
        "final_state": post_states[-1].copy(),
        "payload_adapter": CERTIFICATE_SEMANTICS_PAYLOAD_ADAPTER_VERSION,
        "payload_source": "multipath_capture",
        "retained_proof_metadata_fields": ["certificate_codes"],
        "unavailable_proof_metadata_fields": sorted(
            set(_PROOF_FIELDS) - {"certificate_codes"}
        ),
    }
    _view(payload)
    return payload


def eager_base_record_payload_availability(execution: Any) -> dict[str, Any]:
    """Describe the safe comparison possible from an eager summary object.

    The combined output commitment covers ``Y``, ``Z``, and raw certificate
    codes.  Equality is therefore sufficient evidence of payload equality,
    but inequality cannot distinguish scientific drift from proof-code drift.
    It must be classified as unresolved, never as a scientific mismatch.
    """

    record = _field(execution, ("base_record",), required=True)
    if not isinstance(record, Mapping):
        raise CertificateSemanticsError("base_record must be a mapping")
    capture = _field(execution, ("capture_payload",), required=False)
    diagnostics = record.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    forbidden = diagnostics.get("forbidden_counts", {})
    forbidden_count = (
        sum(int(item) for item in forbidden.values())
        if isinstance(forbidden, Mapping)
        else None
    )
    identity = _field(execution, ("identity",), required=False)
    if identity is not None and hasattr(identity, "to_record"):
        identity = identity.to_record()
    path_ids = _field(execution, ("path_ids",), required=False)
    committed = _field(
        execution, ("committed_final_states",), required=False
    )
    return {
        "schema": CERTIFICATE_SEMANTICS_PAYLOAD_ADAPTER_VERSION
        + "-base-availability",
        "schema_version": 1,
        "raw_capture_available": int(capture is not None),
        "comparator_payload_available": int(capture is not None),
        "execution_identity": identity,
        "path_ids": list(path_ids) if path_ids is not None else None,
        "input_state_sha256": _field(
            execution, ("input_state_sha256",), required=False
        ),
        "direct_final_state_available": int(committed is not None),
        "missing_scientific_fields_without_capture": [
            "transition_ids",
            "earlier_head_fraction",
            "exposure",
            "active_mask",
            "later_head_fraction",
            "denoising_target",
            "certified_mask",
        ] if capture is None else [],
        "available_batch_output_sha256": record.get("batch_output_sha256"),
        "available_batch_final_state_sha256": record.get(
            "batch_final_state_sha256"
        ),
        "advisory_batch_certificate_sha256": record.get(
            "batch_certificate_sha256"
        ),
        "combined_output_hash_includes_raw_certificate_codes": 1,
        "combined_output_hash_equal_is_sufficient_but_not_necessary": 1,
        "combined_output_hash_mismatch_status": (
            "unresolved_scientific_or_proof_code_difference"
        ),
        "transition_count": diagnostics.get("transition_count"),
        "certified_count": diagnostics.get("certified_count"),
        "forbidden_event_count": forbidden_count,
        "safe_hash_only_policy": (
            "require identical source/config/RNG binding, execution identity, "
            "path IDs, input/final-state/combined-output commitments, and "
            "independently healthy authorization; otherwise fail closed as "
            "unresolved"
        ),
    }


def _difference_mask(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape:
        raise CertificateSemanticsError("difference mask requires equal shapes")
    return np.not_equal(left, right)


def compare_certificate_semantics(
    left: Any,
    right: Any,
    *,
    left_final_state: Any | None = None,
    right_final_state: Any | None = None,
    maximum_mismatch_records: int = DEFAULT_MAXIMUM_MISMATCH_RECORDS,
) -> dict[str, Any]:
    """Compare exact transition semantics while allowing proof-route drift.

    ``comparator_valid`` and ``scheduler_seam_valid`` are authorizing outputs.
    Metadata equality is always advisory and cannot rescue or invalidate an
    otherwise valid scientific comparison.
    """

    limit = int(maximum_mismatch_records)
    if not 1 <= limit <= MAXIMUM_MISMATCH_RECORDS:
        raise CertificateSemanticsError(
            f"maximum_mismatch_records must lie in [1,{MAXIMUM_MISMATCH_RECORDS}]"
        )
    left_view = _view(left)
    right_view = _view(right)
    left_shape = left_view["transition_ids"].shape
    right_shape = right_view["transition_ids"].shape
    shape_equal = left_shape == right_shape

    scientific_fields = (
        "transition_ids",
        "earlier_head_fraction",
        "exposure",
        "active_mask",
        "later_head_fraction",
        "denoising_target",
    )
    mismatch_counts: dict[str, int] = {}
    mismatch_examples: list[dict[str, Any]] = []
    if shape_equal:
        left_ids = left_view["transition_ids"].reshape(-1)
        right_ids = right_view["transition_ids"].reshape(-1)
        for field in scientific_fields:
            left_array = left_view[field]
            right_array = right_view[field]
            mask = _difference_mask(left_array, right_array).reshape(-1)
            indices = np.flatnonzero(mask)
            mismatch_counts[field] = int(indices.size)
            for index in indices:
                if len(mismatch_examples) >= limit:
                    break
                flat = int(index)
                mismatch_examples.append(
                    {
                        "field": field,
                        "flat_index": flat,
                        "left_transition_id": _json_scalar(left_ids[flat]),
                        "right_transition_id": _json_scalar(right_ids[flat]),
                        "left": _json_scalar(left_array.reshape(-1)[flat]),
                        "right": _json_scalar(right_array.reshape(-1)[flat]),
                    }
                )
    else:
        for field in scientific_fields:
            mismatch_counts[field] = max(
                int(left_view[field].size), int(right_view[field].size)
            )
        mismatch_examples.append(
            {
                "field": "transition_shape",
                "left": list(left_shape),
                "right": list(right_shape),
            }
        )

    final_state_equal = True
    final_state_compared = left_final_state is not None or right_final_state is not None
    if final_state_compared:
        if left_final_state is None or right_final_state is None:
            raise CertificateSemanticsError(
                "left and right final states must be supplied together"
            )
        left_final = _array(left_final_state, "left_final_state")
        right_final = _array(right_final_state, "right_final_state")
        if (
            left_final.dtype != np.dtype(np.float64)
            or right_final.dtype != np.dtype(np.float64)
        ):
            raise CertificateSemanticsError("final states must have float64 dtype")
        final_state_equal = bool(
            left_final.shape == right_final.shape
            and np.array_equal(left_final, right_final)
        )
        mismatch_counts["final_state"] = (
            0
            if final_state_equal
            else max(int(left_final.size), int(right_final.size))
        )
        if not final_state_equal and len(mismatch_examples) < limit:
            mismatch_examples.append(
                {
                    "field": "final_state",
                    "left_shape": list(left_final.shape),
                    "right_shape": list(right_final.shape),
                    "left_sha256": _array_sha256(left_final),
                    "right_sha256": _array_sha256(right_final),
                }
            )

    left_authorization_valid, left_authorization_counts = _authorization_valid(
        left_view
    )
    right_authorization_valid, right_authorization_counts = _authorization_valid(
        right_view
    )
    authorization_equal = bool(
        shape_equal
        and np.array_equal(left_view["active_mask"], right_view["active_mask"])
        and np.array_equal(
            left_view["certified_mask"], right_view["certified_mask"]
        )
    )
    if shape_equal:
        authorization_mismatch = _difference_mask(
            left_view["certified_mask"], right_view["certified_mask"]
        ).reshape(-1)
        mismatch_counts["normalized_authorization"] = int(
            np.count_nonzero(authorization_mismatch)
        )
        ids_left = left_view["transition_ids"].reshape(-1)
        ids_right = right_view["transition_ids"].reshape(-1)
        for index in np.flatnonzero(authorization_mismatch):
            if len(mismatch_examples) >= limit:
                break
            flat = int(index)
            mismatch_examples.append(
                {
                    "field": "normalized_authorization",
                    "flat_index": flat,
                    "left_transition_id": _json_scalar(ids_left[flat]),
                    "right_transition_id": _json_scalar(ids_right[flat]),
                    "left": bool(left_view["certified_mask"].reshape(-1)[flat]),
                    "right": bool(right_view["certified_mask"].reshape(-1)[flat]),
                }
            )
    else:
        mismatch_counts["normalized_authorization"] = max(
            int(left_view["certified_mask"].size),
            int(right_view["certified_mask"].size),
        )

    proof_names = sorted(set(left_view["proof"]) | set(right_view["proof"]))
    left_proof = {
        name: _proof_summary(left_view["proof"][name])
        for name in proof_names
        if name in left_view["proof"]
    }
    right_proof = {
        name: _proof_summary(right_view["proof"][name])
        for name in proof_names
        if name in right_view["proof"]
    }
    differing_proof_fields = [
        name for name in proof_names if left_proof.get(name) != right_proof.get(name)
    ]
    proof_metadata_equal = not differing_proof_fields

    scientific_payload_equal = bool(
        shape_equal
        and all(mismatch_counts[name] == 0 for name in scientific_fields)
        and final_state_equal
    )
    certificate_semantics_equal = bool(
        authorization_equal
        and left_authorization_valid
        and right_authorization_valid
    )
    valid = bool(scientific_payload_equal and certificate_semantics_equal)
    total_mismatches = sum(mismatch_counts.values())
    left_payload_hashes = {
        name: _array_sha256(left_view[name]) for name in scientific_fields
    }
    right_payload_hashes = {
        name: _array_sha256(right_view[name]) for name in scientific_fields
    }
    left_authorization_hash = _array_sha256(
        np.where(
            left_view["active_mask"] & left_view["certified_mask"],
            np.uint8(0x0F),
            np.uint8(0),
        )
    )
    right_authorization_hash = _array_sha256(
        np.where(
            right_view["active_mask"] & right_view["certified_mask"],
            np.uint8(0x0F),
            np.uint8(0),
        )
    )
    record = {
        "schema": CERTIFICATE_SEMANTICS_COMPARATOR_VERSION,
        "schema_version": 1,
        "comparison_executed": 1,
        "comparator_valid": int(valid),
        "certificate_semantics_comparator_valid": int(valid),
        "scientific_payload_equal": int(scientific_payload_equal),
        "certificate_semantics_equal": int(certificate_semantics_equal),
        "scheduler_seam_valid": int(valid),
        "transition_shape_equal": int(shape_equal),
        "transition_shape": list(left_shape) if shape_equal else None,
        "final_state_compared": int(final_state_compared),
        "final_state_equal": int(final_state_equal),
        "left_authorization_valid": int(left_authorization_valid),
        "right_authorization_valid": int(right_authorization_valid),
        "left_authorization_counts": left_authorization_counts,
        "right_authorization_counts": right_authorization_counts,
        "left_scientific_payload_sha256": _record_sha256(left_payload_hashes),
        "right_scientific_payload_sha256": _record_sha256(right_payload_hashes),
        "left_normalized_authorization_sha256": left_authorization_hash,
        "right_normalized_authorization_sha256": right_authorization_hash,
        "proof_metadata_advisory": 1,
        "proof_metadata_equality_required": 0,
        "proof_metadata_equal": int(proof_metadata_equal),
        "differing_proof_metadata_fields": differing_proof_fields,
        "left_proof_metadata": left_proof,
        "right_proof_metadata": right_proof,
        "mismatch_counts": mismatch_counts,
        "mismatch_count": int(total_mismatches),
        "maximum_mismatch_records": limit,
        "mismatch_records": mismatch_examples,
        "mismatch_record_count": len(mismatch_examples),
        "mismatch_records_truncated": int(total_mismatches > len(mismatch_examples)),
    }
    return record


compare_jacobi_rb_certificate_semantics = compare_certificate_semantics


__all__ = [
    "CERTIFICATE_SEMANTICS_COMPARATOR_VERSION",
    "CERTIFICATE_SEMANTICS_PAYLOAD_ADAPTER_VERSION",
    "CertificateSemanticsError",
    "DEFAULT_MAXIMUM_MISMATCH_RECORDS",
    "MAXIMUM_MISMATCH_RECORDS",
    "compare_certificate_semantics",
    "compare_jacobi_rb_certificate_semantics",
    "comparator_payload_from_certified_cuda_batch",
    "comparator_payload_from_eager_midpoint_branch",
    "comparator_payload_from_multipath_capture",
    "eager_base_record_payload_availability",
]
