"""No-host-sync CUDA adapters for fused exploratory rollouts.

The exact synchronous sampler and fused authorizer remain in their sealed
historical modules.  This adapter prepares their kernels and RNG seed tensors
outside shard timing, then exposes a speculative device-only enqueue result.
Unresolved lanes never authorize a commit; callers replay the whole shard
through the unchanged synchronous sampler.

The separate candidate preparation/enqueue path exposes only the approximate
proposal kernel and carries no certificate, replay, or Arb authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from mnist import d0_jacobi_rb_cuda as _base
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_cuda_certificate import FallbackReason as _FallbackReason
from mnist.d0_jacobi_rb_cuda_fused import (
    FusedCudaBundle,
    FusedCudaLaunch,
    probe_fused_cuda_authorizer,
)

_canonical_seed = _base._canonical_seed
_certificate_arithmetic_preflight = _base._certificate_arithmetic_preflight
_load_cuda_kernel = _base._load_cuda_kernel
_reference = _base._reference
sample_alpha1_rb_transition_batch_cuda = _base.sample_alpha1_rb_transition_batch_cuda


@dataclass(frozen=True)
class PreparedDeferredRBCudaBackend:
    """Warm, self-tested CUDA handles for the no-host-sync enqueue path.

    Construction is deliberately separate from enqueueing because compiling and
    self-testing the fused authorizer synchronizes the device.  Production code
    prepares this handle during its explicit untimed warm-up, then reuses it for
    every speculative call in an eight-step shard.
    """

    device: Any
    profile: JacobiRBCudaProfile
    candidate_kernel: Any
    candidate_binary_sha256: str
    fused_bundle: FusedCudaBundle
    fused_report: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedCandidateRBCudaBackend:
    """Loaded CUDA proposal handle with no exact-authorizer authority."""

    device: Any
    profile: JacobiRBCudaProfile
    candidate_kernel: Any
    candidate_binary_sha256: str


PreparedRBCudaBackend = (
    PreparedDeferredRBCudaBackend | PreparedCandidateRBCudaBackend
)


@dataclass(frozen=True)
class PreparedDeferredRBCudaSeed:
    """One stateless-Philox seed already resident on the CUDA device.

    Seed construction is deliberately separate from the enqueue API.  A
    fused scheduler prepares the finite set of frozen role keys before shard
    timing, then passes these records through the phase loop without a
    per-transition host-to-device scalar transfer.
    """

    device: Any
    seed: int
    tensor: Tensor


@dataclass(frozen=True)
class DeferredRBCudaBatch:
    """Device-resident speculative transition payload.

    Only lanes selected by ``certified_mask`` are authorizing active
    transitions.  Exact zero-duration lanes are selected by
    ``structural_noop_mask``.  ``authorized_mask`` is their union.  A caller may
    advance a speculative shard with ``later_head_fraction``, but it must not
    commit that shard unless every lane is valid and every active lane is
    certified.  Otherwise the whole shard is replayed through the unchanged
    synchronous exact sampler, which owns candidate-local Arb fallback.
    """

    earlier_head_fraction: Tensor
    later_head_fraction: Tensor
    denoising_target: Tensor
    exposure: Tensor
    transition_ids: Tensor
    active_mask: Tensor
    structural_noop_mask: Tensor
    authorized_mask: Tensor
    certified_mask: Tensor
    cuda_certified_mask: Tensor
    fallback_mask: Tensor
    valid_mask: Tensor
    candidate_later_head_fraction: Tensor
    candidate_denoising_target: Tensor
    candidate_match_mask: Tensor
    strengthened_mask: Tensor
    fallback_reason_codes: Tensor
    mode_counts: Tensor
    prefix_bits: Tensor
    certificate_codes: Tensor
    quantile_lower: Tensor
    quantile_upper: Tensor
    target_lower: Tensor
    target_upper: Tensor
    device_diagnostics: Mapping[str, Tensor]

    @property
    def arb_fallback_reason_codes(self) -> Tensor:
        """Compatibility spelling for CUDA fallback-reason telemetry."""

        return self.fallback_reason_codes

    @property
    def diagnostics(self) -> Mapping[str, Tensor]:
        """Expose the device-only record under the established batch name."""

        return self.device_diagnostics


@dataclass(frozen=True)
class CandidateRBCudaBatch:
    """Non-authorizing CUDA candidate transition payload.

    Every active lane is explicitly marked as an approximation.  This record
    intentionally has no certified or authorization mask: the candidate
    kernel is useful for exploratory rollouts, but it cannot support an exact
    transition claim.
    """

    earlier_head_fraction: Tensor
    later_head_fraction: Tensor
    denoising_target: Tensor
    exposure: Tensor
    transition_ids: Tensor
    active_mask: Tensor
    structural_noop_mask: Tensor
    approximation_mask: Tensor
    valid_mask: Tensor
    candidate_lower: Tensor
    candidate_upper: Tensor
    device_diagnostics: Mapping[str, Tensor]

    @property
    def diagnostics(self) -> Mapping[str, Tensor]:
        """Expose the device-only integrity record under the batch spelling."""

        return self.device_diagnostics



def launch_fused_cuda_authorizer(
    bundle: FusedCudaBundle,
    head_fraction: Tensor,
    exposure: Tensor,
    transition_ids: Tensor,
    proposed_y: Tensor,
    *,
    seed: int,
    seed_tensor: Tensor | None = None,
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
    active_seed_tensor = seed_tensor
    if seed_tensor is not None:
        if (
            not isinstance(seed_tensor, Tensor)
            or not seed_tensor.is_cuda
            or seed_tensor.device != device
            or seed_tensor.dtype != torch.uint64
            or seed_tensor.shape != (1,)
            or not seed_tensor.is_contiguous()
        ):
            raise ValueError(
                "seed_tensor must be contiguous same-device CUDA uint64[1]"
            )
    if count:
        if active_seed_tensor is None:
            active_seed_tensor = torch.tensor(
                [int(seed)], dtype=torch.uint64, device=device
            )
        threads = int(threads_per_block)
        bundle.authorizer(
            grid=((count + threads - 1) // threads, 1, 1),
            block=(threads, 1, 1),
            args=[
                head_fraction, exposure, transition_ids, active_seed_tensor, proposed_y,
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



def _require_deferred_cuda_inputs(
    head_fraction: Tensor,
    exposure: Tensor,
    transition_ids: Tensor,
    prepared: PreparedRBCudaBackend,
) -> tuple[Any, tuple[int, ...], int]:
    """Validate only static tensor metadata for the asynchronous fast path."""

    if torch is None:
        raise RuntimeError("PyTorch is unavailable")
    if not isinstance(
        prepared, (PreparedDeferredRBCudaBackend, PreparedCandidateRBCudaBackend)
    ):
        raise TypeError("prepared must be a prepared Jacobi RB CUDA backend")
    values = (
        ("head_fraction", head_fraction, torch.float64),
        ("exposure", exposure, torch.float64),
        ("transition_ids", transition_ids, torch.uint64),
    )
    shape: tuple[int, ...] | None = None
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
            shape, device = tuple(tensor.shape), tensor.device
        elif tuple(tensor.shape) != shape or tensor.device != device:
            raise ValueError("all inputs must have identical shapes and CUDA devices")
    if device != prepared.device:
        raise ValueError("deferred inputs are on a different CUDA device")
    count = int(head_fraction.numel())
    if count > 4096:
        raise ValueError("deferred CUDA calls are capped at 4096 transition lanes")
    assert shape is not None
    return device, shape, count


def prepare_alpha1_rb_transition_batch_cuda_deferred(
    *,
    device: str | Any,
    profile: JacobiRBCudaProfile,
) -> PreparedDeferredRBCudaBackend:
    """Compile and self-test the exact kernels before deferred shard timing.

    This preparation function is intentionally synchronous.  It belongs in an
    explicit compile/warm-up stage and must never be called from the fused hot
    loop.  The returned opaque handles are the authority passed to
    :func:`enqueue_alpha1_rb_transition_batch_cuda_no_fallback`.
    """

    if torch is None:
        raise RuntimeError("PyTorch is unavailable")
    if not isinstance(profile, JacobiRBCudaProfile):
        raise TypeError("profile must be a JacobiRBCudaProfile")
    selected_device = torch.device(device)
    if selected_device.type != "cuda":
        raise ValueError("deferred Jacobi preparation requires a CUDA device")
    index = selected_device.index
    if index is None:
        index = int(torch.cuda.current_device())
    selected_device = torch.device("cuda", int(index))
    candidate_kernel, candidate_binary_sha256 = _load_cuda_kernel(
        selected_device, profile
    )
    fused_bundle, fused_report = probe_fused_cuda_authorizer(
        selected_device,
        compile_flags=tuple(profile.compile_flags),
        cpu_preflight=dict(_certificate_arithmetic_preflight()),
    )
    if fused_bundle is None or not bool(
        fused_report.get("fused_cuda_authorizer_available", False)
    ):
        raise RuntimeError(
            "deferred Jacobi authorizer preparation failed closed: "
            f"{fused_report.get('fused_cuda_authorizer_unavailable_reason')}"
        )
    return PreparedDeferredRBCudaBackend(
        device=selected_device,
        profile=profile,
        candidate_kernel=candidate_kernel,
        candidate_binary_sha256=str(candidate_binary_sha256),
        fused_bundle=fused_bundle,
        fused_report=dict(fused_report),
    )


def prepare_alpha1_rb_transition_batch_cuda_candidate(
    *,
    device: str | Any,
    profile: JacobiRBCudaProfile,
) -> PreparedCandidateRBCudaBackend:
    """Compile/load only the candidate proposal kernel.

    The first explicitly approved smoke or science shard performs any device
    warm-up; preparation does not launch hidden sampling work.
    """

    if torch is None:
        raise RuntimeError("PyTorch is unavailable")
    if not isinstance(profile, JacobiRBCudaProfile):
        raise TypeError("profile must be a JacobiRBCudaProfile")
    selected_device = torch.device(device)
    if selected_device.type != "cuda":
        raise ValueError("candidate Jacobi preparation requires a CUDA device")
    index = selected_device.index
    if index is None:
        index = int(torch.cuda.current_device())
    selected_device = torch.device("cuda", int(index))
    candidate_kernel, candidate_binary_sha256 = _load_cuda_kernel(
        selected_device, profile
    )
    return PreparedCandidateRBCudaBackend(
        device=selected_device,
        profile=profile,
        candidate_kernel=candidate_kernel,
        candidate_binary_sha256=str(candidate_binary_sha256),
    )


def prepare_alpha1_rb_transition_cuda_rng_seed(
    *,
    rng_key: Any,
    prepared: PreparedRBCudaBackend,
) -> PreparedDeferredRBCudaSeed:
    """Materialize one canonical RNG seed before a deferred hot loop."""

    if torch is None:
        raise RuntimeError("PyTorch is unavailable")
    if not isinstance(
        prepared, (PreparedDeferredRBCudaBackend, PreparedCandidateRBCudaBackend)
    ):
        raise TypeError("prepared must be a prepared Jacobi RB CUDA backend")
    seed = _canonical_seed(rng_key)
    tensor = torch.tensor(
        [seed], dtype=torch.uint64, device=prepared.device
    ).contiguous()
    return PreparedDeferredRBCudaSeed(
        device=prepared.device,
        seed=int(seed),
        tensor=tensor,
    )


def _require_prepared_deferred_rng_seed(
    value: PreparedDeferredRBCudaSeed | None,
    *,
    seed: int,
    device: Any,
) -> Tensor:
    """Static-only validation of a seed prepared outside shard timing."""

    if not isinstance(value, PreparedDeferredRBCudaSeed):
        raise TypeError(
            "prepared_rng_seed must be a PreparedDeferredRBCudaSeed"
        )
    tensor = value.tensor
    if (
        value.device != device
        or int(value.seed) != int(seed)
        or not isinstance(tensor, torch.Tensor)
        or not tensor.is_cuda
        or tensor.device != device
        or tensor.dtype != torch.uint64
        or tensor.shape != (1,)
        or not tensor.is_contiguous()
    ):
        raise ValueError("prepared deferred RNG seed does not match this call")
    return tensor


def validate_prepared_alpha1_rb_transition_cuda_rng_seed(
    *,
    prepared_seed: PreparedDeferredRBCudaSeed,
    rng_key: Any,
    prepared: PreparedRBCudaBackend,
) -> None:
    """Validate a prebuilt seed using metadata only.

    This public additive validator deliberately performs no tensor reduction,
    device transfer, or CUDA synchronization.  Fused shard references use it
    when accepting a seed map prepared at the untimed family boundary.
    """

    if not isinstance(
        prepared, (PreparedDeferredRBCudaBackend, PreparedCandidateRBCudaBackend)
    ):
        raise TypeError("prepared must be a prepared Jacobi RB CUDA backend")
    seed = _canonical_seed(rng_key)
    _require_prepared_deferred_rng_seed(
        prepared_seed,
        seed=seed,
        device=prepared.device,
    )


def enqueue_alpha1_rb_transition_batch_cuda_no_fallback(
    head_fraction: Tensor,
    exposure: Tensor,
    *,
    rng_key: Any,
    transition_ids: Tensor,
    prepared: PreparedDeferredRBCudaBackend,
    prepared_rng_seed: PreparedDeferredRBCudaSeed | None = None,
) -> DeferredRBCudaBatch:
    """Enqueue one exact device-certificate attempt without a host barrier.

    Duplicate transition IDs are deliberate and accepted: stateless Philox
    then supplies common random bits to distinct state rows.  This function
    performs static metadata validation only.  Value/range checks, certificate
    checks, and every diagnostic reduction remain device tensors.

    No unresolved active lane is authorizing.  The caller may use its proposed
    value only while executing a speculative shard and must synchronously
    replay the complete shard through the existing exact sampler before any
    commit when ``fallback_mask`` or the complement of ``valid_mask`` is set.
    """

    if not isinstance(prepared, PreparedDeferredRBCudaBackend):
        raise TypeError("prepared must be a PreparedDeferredRBCudaBackend")
    device, shape, count = _require_deferred_cuda_inputs(
        head_fraction, exposure, transition_ids, prepared
    )
    selected = prepared.profile
    seed = _canonical_seed(rng_key)
    seed_tensor = _require_prepared_deferred_rng_seed(
        prepared_rng_seed, seed=seed, device=device
    )
    flat_x = head_fraction.reshape(-1)
    flat_u = exposure.reshape(-1)
    flat_ids = transition_ids.reshape(-1)

    input_valid = (
        torch.isfinite(flat_x)
        & torch.isfinite(flat_u)
        & (flat_x >= 0.0)
        & (flat_x <= 1.0)
        & (flat_u >= 0.0)
    )
    safe_x = torch.where(input_valid, flat_x, torch.full_like(flat_x, 0.5))
    safe_u = torch.where(input_valid, flat_u, torch.zeros_like(flat_u))
    active = input_valid & (safe_u > 0.0)
    structural_noop = input_valid & (safe_u == 0.0)

    candidate_y = torch.empty_like(safe_x)
    candidate_z = torch.empty_like(safe_x)
    candidate_lower = torch.empty_like(safe_x)
    candidate_upper = torch.empty_like(safe_x)
    if count:
        recorded_values = torch.zeros(count, dtype=torch.uint64, device=device)
        recorded_lengths = torch.zeros(count, dtype=torch.int32, device=device)
        threads = int(selected.threads_per_block)
        prepared.candidate_kernel(
            grid=((count + threads - 1) // threads, 1, 1),
            block=(threads, 1, 1),
            args=[
                safe_x,
                safe_u,
                flat_ids,
                seed_tensor,
                recorded_values,
                recorded_lengths,
                0,
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
    else:
        candidate_y.copy_(safe_x)
        candidate_z.zero_()
        candidate_lower.copy_(safe_x)
        candidate_upper.copy_(safe_x)

    primary = launch_fused_cuda_authorizer(
        prepared.fused_bundle,
        safe_x,
        safe_u,
        flat_ids,
        candidate_y,
        seed=seed,
        seed_tensor=seed_tensor,
        threads_per_block=int(selected.threads_per_block),
        max_prefix_bits=min(64, int(selected.max_prefix_bits)),
        _primary_cap=4096,
        _strengthened_cap=4096,
    )
    # A primary CDF miss carries a non-authorizing DD Newton suggestion in
    # ``later``.  Retest that suggestion unconditionally in a second fixed
    # launch.  Unlike the adaptive neighbour wrapper, this fixed two-launch
    # schedule has no data-dependent host branch or compaction barrier.
    retest = launch_fused_cuda_authorizer(
        prepared.fused_bundle,
        safe_x,
        safe_u,
        flat_ids,
        primary.later,
        seed=seed,
        seed_tensor=seed_tensor,
        threads_per_block=int(selected.threads_per_block),
        max_prefix_bits=int(selected.max_prefix_bits),
        _primary_cap=4096,
        _strengthened_cap=8192,
    )
    take_retest = active & ~primary.authorized_mask & retest.authorized_mask
    cuda_authorized = primary.authorized_mask | take_retest

    def selected_tensor(name: str) -> Tensor:
        return torch.where(
            take_retest, getattr(retest, name), getattr(primary, name)
        )

    fused_later = selected_tensor("later")
    fused_target = selected_tensor("target")
    fused_quantile_lower = selected_tensor("quantile_lower")
    fused_quantile_upper = selected_tensor("quantile_upper")
    fused_target_lower = selected_tensor("target_lower")
    fused_target_upper = selected_tensor("target_upper")
    fused_codes = selected_tensor("certificate_codes")
    fused_modes = torch.where(
        active & ~primary.authorized_mask,
        retest.modes_used,
        primary.modes_used,
    )
    fused_prefix_bits = torch.where(
        active & ~primary.authorized_mask,
        retest.prefix_bits,
        primary.prefix_bits,
    )
    fused_strengthened = (
        (primary.authorized_mask & primary.strengthened_mask)
        | (take_retest & retest.strengthened_mask)
    )
    fused_reasons = torch.where(
        active & ~primary.authorized_mask & ~take_retest,
        retest.fallback_reason_codes,
        primary.fallback_reason_codes,
    )
    fused_reasons = torch.where(
        cuda_authorized, torch.zeros_like(fused_reasons), fused_reasons
    )
    active_cuda_authorized = active & cuda_authorized
    active_code_valid = (fused_codes & 0b1111) == 0b1111
    active_output_valid = (
        torch.isfinite(fused_later)
        & torch.isfinite(fused_target)
        & torch.isfinite(fused_quantile_lower)
        & torch.isfinite(fused_quantile_upper)
        & torch.isfinite(fused_target_lower)
        & torch.isfinite(fused_target_upper)
        & (fused_later >= 0.0)
        & (fused_later <= 1.0)
        & (fused_quantile_lower <= fused_later)
        & (fused_later <= fused_quantile_upper)
        & (fused_target_lower <= fused_target)
        & (fused_target <= fused_target_upper)
    )
    certified = active_cuda_authorized & active_code_valid & active_output_valid
    fallback = active & ~certified
    authorized = structural_noop | certified
    valid = input_valid & authorized
    later = torch.where(structural_noop, safe_x, fused_later)
    target = torch.where(structural_noop, torch.zeros_like(fused_target), fused_target)
    candidate_match = certified & (candidate_y == later) & (candidate_z == target)
    candidate_finite = torch.isfinite(candidate_y) & torch.isfinite(candidate_z)
    invalid_output = active & ~active_output_valid
    zero_i64 = torch.zeros((), dtype=torch.int64, device=device)
    diagnostics: dict[str, Tensor] = {
        "sample_count": torch.full((), count, dtype=torch.int64, device=device),
        "active_count": active.sum(dtype=torch.int64),
        "zero_duration_count": structural_noop.sum(dtype=torch.int64),
        "cuda_authorized_count": active_cuda_authorized.sum(dtype=torch.int64),
        "certified_count": certified.sum(dtype=torch.int64),
        "fallback_count": fallback.sum(dtype=torch.int64),
        "invalid_input_count": (~input_valid).sum(dtype=torch.int64),
        "invalid_output_count": invalid_output.sum(dtype=torch.int64),
        "candidate_match_count": candidate_match.sum(dtype=torch.int64),
        "strengthened_count": (certified & fused_strengthened).sum(
            dtype=torch.int64
        ),
        "mode_cap_hit_count": (fallback & (fused_modes >= 8192)).sum(
            dtype=torch.int64
        ),
        "prefix_cap_hit_count": (
            fallback & (fused_prefix_bits >= int(selected.max_prefix_bits))
        ).sum(dtype=torch.int64),
        "cuda_strengthened_exhaustion_count": (
            fallback
            & (
                (fused_modes >= 8192)
                | (fused_prefix_bits >= int(selected.max_prefix_bits))
            )
        ).sum(dtype=torch.int64),
        "resource_cap_count": zero_i64.clone(),
        "invalid_density_count": (
            fallback
            & (
                fused_reasons
                == int(_FallbackReason.NONPOSITIVE_DENSITY)
            )
        ).sum(dtype=torch.int64),
        "nonfinite_count": (
            (~torch.isfinite(flat_x))
            | (~torch.isfinite(flat_u))
            | invalid_output
        ).sum(dtype=torch.int64),
        "candidate_nonfinite_count": (~candidate_finite).sum(dtype=torch.int64),
        "approximation_count": zero_i64.clone(),
        "candidate_repair_count": (certified & (candidate_y != later)).sum(
            dtype=torch.int64
        ),
        "correction_count": zero_i64.clone(),
        "floor_count": zero_i64.clone(),
        "limiter_count": zero_i64.clone(),
        "projection_count": zero_i64.clone(),
        "renormalization_count": zero_i64.clone(),
        "maximum_cuda_launch_lanes": torch.full(
            (), count, dtype=torch.int64, device=device
        ),
        "candidate_kernel_launch_count": torch.full(
            (), int(bool(count)), dtype=torch.int64, device=device
        ),
        "fused_authorizer_launch_count": torch.full(
            (), 2 * int(bool(count)), dtype=torch.int64, device=device
        ),
        "replay_required": (~valid).any(),
    }
    return DeferredRBCudaBatch(
        earlier_head_fraction=head_fraction,
        later_head_fraction=later.reshape(shape),
        denoising_target=target.reshape(shape),
        exposure=exposure,
        transition_ids=transition_ids,
        active_mask=active.reshape(shape),
        structural_noop_mask=structural_noop.reshape(shape),
        authorized_mask=authorized.reshape(shape),
        certified_mask=certified.reshape(shape),
        cuda_certified_mask=active_cuda_authorized.reshape(shape),
        fallback_mask=fallback.reshape(shape),
        valid_mask=valid.reshape(shape),
        candidate_later_head_fraction=candidate_y.reshape(shape),
        candidate_denoising_target=candidate_z.reshape(shape),
        candidate_match_mask=candidate_match.reshape(shape),
        strengthened_mask=fused_strengthened.reshape(shape),
        fallback_reason_codes=fused_reasons.reshape(shape),
        mode_counts=fused_modes.reshape(shape),
        prefix_bits=fused_prefix_bits.reshape(shape),
        certificate_codes=fused_codes.reshape(shape),
        quantile_lower=fused_quantile_lower.reshape(shape),
        quantile_upper=fused_quantile_upper.reshape(shape),
        target_lower=fused_target_lower.reshape(shape),
        target_upper=fused_target_upper.reshape(shape),
        device_diagnostics=diagnostics,
    )


def enqueue_alpha1_rb_transition_batch_cuda_candidate(
    head_fraction: Tensor,
    exposure: Tensor,
    *,
    rng_key: Any,
    transition_ids: Tensor,
    prepared: PreparedRBCudaBackend,
    prepared_rng_seed: PreparedDeferredRBCudaSeed | None = None,
) -> CandidateRBCudaBatch:
    """Enqueue only the frozen CUDA proposal kernel, without certification.

    This is a deliberately approximate exploratory backend.  It preserves
    canonical transition IDs and stateless-Philox pairing, but it never
    launches the double-double authorizer and never invokes Arb fallback.
    Value and bracket checks remain device-resident until a caller's explicit
    shard boundary.
    """

    device, shape, count = _require_deferred_cuda_inputs(
        head_fraction, exposure, transition_ids, prepared
    )
    selected = prepared.profile
    if (
        int(selected.candidate_modes) != 128
        or int(selected.candidate_bisection_steps) != 56
    ):
        raise ValueError(
            "candidate approximate contract requires the frozen 128-mode, "
            "56-bisection profile"
        )
    seed = _canonical_seed(rng_key)
    seed_tensor = _require_prepared_deferred_rng_seed(
        prepared_rng_seed, seed=seed, device=device
    )
    flat_x = head_fraction.reshape(-1)
    flat_u = exposure.reshape(-1)
    flat_ids = transition_ids.reshape(-1)

    input_valid = (
        torch.isfinite(flat_x)
        & torch.isfinite(flat_u)
        & (flat_x >= 0.0)
        & (flat_x <= 1.0)
        & (flat_u >= 0.0)
    )
    safe_x = torch.where(input_valid, flat_x, torch.full_like(flat_x, 0.5))
    safe_u = torch.where(input_valid, flat_u, torch.zeros_like(flat_u))
    active = input_valid & (safe_u > 0.0)
    structural_noop = input_valid & (safe_u == 0.0)

    candidate_y = torch.empty_like(safe_x)
    candidate_z = torch.empty_like(safe_x)
    candidate_lower = torch.empty_like(safe_x)
    candidate_upper = torch.empty_like(safe_x)
    if count:
        recorded_values = torch.zeros(count, dtype=torch.uint64, device=device)
        recorded_lengths = torch.zeros(count, dtype=torch.int32, device=device)
        threads = int(selected.threads_per_block)
        prepared.candidate_kernel(
            grid=((count + threads - 1) // threads, 1, 1),
            block=(threads, 1, 1),
            args=[
                safe_x,
                safe_u,
                flat_ids,
                seed_tensor,
                recorded_values,
                recorded_lengths,
                0,
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
    else:
        candidate_y.copy_(safe_x)
        candidate_z.zero_()
        candidate_lower.copy_(safe_x)
        candidate_upper.copy_(safe_x)

    # Structural zero-exposure transitions are exact no-ops even if a future
    # kernel implementation changes its zero-duration fast path.
    later = torch.where(active, candidate_y, safe_x)
    target = torch.where(
        active, candidate_z, torch.zeros_like(candidate_z)
    )
    lower = torch.where(active, candidate_lower, safe_x)
    upper = torch.where(active, candidate_upper, safe_x)
    width = upper - lower
    output_valid = (
        torch.isfinite(later)
        & torch.isfinite(target)
        & torch.isfinite(lower)
        & torch.isfinite(upper)
        & torch.isfinite(width)
        & (later >= 0.0)
        & (later <= 1.0)
        & (lower >= 0.0)
        & (lower <= later)
        & (later <= upper)
        & (upper <= 1.0)
        & (width >= 0.0)
    )
    noop_valid = structural_noop & (later == safe_x) & (target == 0.0)
    valid = input_valid & (noop_valid | (active & output_valid))
    approximation = active
    invalid_output = input_valid & ~valid
    zero_i64 = torch.zeros((), dtype=torch.int64, device=device)
    active_width = torch.where(active & output_valid, width, torch.zeros_like(width))
    diagnostics: dict[str, Tensor] = {
        "sample_count": torch.full((), count, dtype=torch.int64, device=device),
        "active_count": active.sum(dtype=torch.int64),
        "structural_noop_count": structural_noop.sum(dtype=torch.int64),
        "approximation_count": approximation.sum(dtype=torch.int64),
        "invalid_input_count": (~input_valid).sum(dtype=torch.int64),
        "invalid_output_count": invalid_output.sum(dtype=torch.int64),
        "nonfinite_count": (
            (~torch.isfinite(flat_x))
            | (~torch.isfinite(flat_u))
            | (active & ~torch.isfinite(candidate_y))
            | (active & ~torch.isfinite(candidate_z))
            | (active & ~torch.isfinite(candidate_lower))
            | (active & ~torch.isfinite(candidate_upper))
        ).sum(dtype=torch.int64),
        "negative_bracket_width_count": (
            active & torch.isfinite(width) & (width < 0.0)
        ).sum(dtype=torch.int64),
        "bracket_order_invalid_count": (
            active
            & ~(
                torch.isfinite(lower)
                & torch.isfinite(later)
                & torch.isfinite(upper)
                & (lower >= 0.0)
                & (lower <= later)
                & (later <= upper)
                & (upper <= 1.0)
            )
        ).sum(dtype=torch.int64),
        "maximum_candidate_bracket_width": torch.amax(active_width)
        if count
        else torch.zeros((), dtype=torch.float64, device=device),
        "candidate_kernel_launch_count": torch.full(
            (), int(bool(count)), dtype=torch.int64, device=device
        ),
        "resource_cap_count": zero_i64.clone(),
        "invalid_density_count": zero_i64.clone(),
        "correction_count": zero_i64.clone(),
        "clipping_count": zero_i64.clone(),
        "floor_count": zero_i64.clone(),
        "limiter_count": zero_i64.clone(),
        "projection_count": zero_i64.clone(),
        "renormalization_count": zero_i64.clone(),
    }
    return CandidateRBCudaBatch(
        earlier_head_fraction=head_fraction,
        later_head_fraction=later.reshape(shape),
        denoising_target=target.reshape(shape),
        exposure=exposure,
        transition_ids=transition_ids,
        active_mask=active.reshape(shape),
        structural_noop_mask=structural_noop.reshape(shape),
        approximation_mask=approximation.reshape(shape),
        valid_mask=valid.reshape(shape),
        candidate_lower=lower.reshape(shape),
        candidate_upper=upper.reshape(shape),
        device_diagnostics=diagnostics,
    )




__all__ = [
    "CandidateRBCudaBatch",
    "DeferredRBCudaBatch",
    "PreparedCandidateRBCudaBackend",
    "PreparedDeferredRBCudaBackend",
    "PreparedDeferredRBCudaSeed",
    "enqueue_alpha1_rb_transition_batch_cuda_candidate",
    "enqueue_alpha1_rb_transition_batch_cuda_no_fallback",
    "prepare_alpha1_rb_transition_batch_cuda_candidate",
    "prepare_alpha1_rb_transition_batch_cuda_deferred",
    "prepare_alpha1_rb_transition_cuda_rng_seed",
    "validate_prepared_alpha1_rb_transition_cuda_rng_seed",
]
