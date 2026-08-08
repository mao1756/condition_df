r"""Exact Jacobi authorization for certified Haar-uniform enclosures.

The CUDA adapter does not evaluate a Gaussian transition.  It encloses each
certified Haar uniform in one dyadic prefix and passes that prefix to the
unchanged fused Jacobi inverse-CDF/target certifier.  Authorization therefore
still requires the strict Jacobi rounding-cell proof.  A wide or boundary-
straddling interval fails closed unless the caller supplies a deterministic
refinement callback.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import math
import time
from typing import Any, Callable, Iterable

import numpy as np

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - stripped test environments.
    torch = None
    Tensor = Any

from mnist import d0_jacobi_rb_cuda as _rb_cuda
from mnist import d0_jacobi_rb_spectral as _spectral
from mnist.d0_jacobi_rb_cuda import (
    CertifiedRBCudaBatch,
    JacobiRBCudaProfile,
)
from mnist.d0_jacobi_rb_cuda_certificate import (
    fraction_to_float_down,
    fraction_to_float_up,
)
from mnist.d0_jacobi_rb_haar import (
    CertifiedUniformCell,
    UniformCellRefinementRequest,
    UniformCellRefinementResult,
)
from mnist.d0_jacobi_rb_haar_fused import (
    HAAR_FUSED_CUDA_VERSION,
    launch_certified_jacobi_from_uniform_cells,
)


HAAR_UNIFORM_JACOBI_ADAPTER_VERSION = (
    "d0-jacobi-rb-certified-haar-uniform-adapter-v1"
)
JACOBI_RECORDED_PREFIX_LIMIT_BITS = 64


@dataclass(frozen=True)
class CertifiedRBIntervalBatch:
    """Portable Arb-authorized reference result for exact uniform cells."""

    earlier_head_fraction: np.ndarray
    later_head_fraction: np.ndarray
    denoising_target: np.ndarray
    exposure: np.ndarray
    transition_ids: np.ndarray
    active_mask: np.ndarray
    certified_mask: np.ndarray
    fallback_mask: np.ndarray
    quantile_lower: np.ndarray
    quantile_upper: np.ndarray
    target_lower: np.ndarray
    target_upper: np.ndarray
    prefix_bits: np.ndarray
    certificate_codes: np.ndarray
    mode_counts: np.ndarray
    uniform_cells: tuple[CertifiedUniformCell, ...]
    diagnostics: dict[str, Any]
    runtime_report: dict[str, Any]


class _EnclosingDyadicPrefix:
    """One immutable dyadic cell containing a certified uniform interval."""

    def __init__(self, numerator: int, bits: int) -> None:
        if not 1 <= int(bits) <= 1024:
            raise ValueError("enclosing prefix bits must lie in [1,1024]")
        if not 0 <= int(numerator) < (1 << int(bits)):
            raise ValueError("enclosing prefix numerator does not fit")
        self.numerator = int(numerator)
        self.bits = int(bits)
        self.max_bits = int(bits)

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

    def refine(self, count: int) -> None:
        raise _spectral.JacobiRBCertificationError(
            "derived Haar prefix needs source refinement",
            {
                "failure_kind": "haar_uniform_prefix_exhausted",
                "prefix_bits": self.bits,
                "requested_additional_bits": int(count),
            },
        )


def enclosing_dyadic_prefix(
    cell: CertifiedUniformCell, *, max_bits: int = 1024
) -> tuple[int, int]:
    """Return the longest dyadic cell containing the complete interval."""

    if not isinstance(cell, CertifiedUniformCell):
        raise TypeError("cell must be a CertifiedUniformCell")
    selected_numerator = 0
    selected_bits = 0
    for bits in range(1, int(max_bits) + 1):
        scale = 1 << bits
        numerator = (cell.lower.numerator * scale) // cell.lower.denominator
        if cell.upper <= Fraction(numerator + 1, scale):
            selected_numerator, selected_bits = numerator, bits
            continue
        break
    if selected_bits == 0:
        raise _spectral.JacobiRBCertificationError(
            "certified Haar uniform straddles the first dyadic boundary",
            {
                "failure_kind": "haar_uniform_prefix_unresolved",
                "uniform_lower": str(cell.lower),
                "uniform_upper": str(cell.upper),
            },
        )
    return selected_numerator, selected_bits


def _coerce_cells(
    cells: Iterable[CertifiedUniformCell], shape: tuple[int, ...]
) -> tuple[CertifiedUniformCell, ...]:
    result = tuple(cells)
    if len(result) != math.prod(shape):
        raise ValueError("uniform cell count does not match the transition shape")
    if any(not isinstance(cell, CertifiedUniformCell) for cell in result):
        raise TypeError("uniform_cells must contain CertifiedUniformCell records")
    return result


def _as_ids(values: Any, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(values, dtype=np.uint64)
    if result.shape != shape:
        raise ValueError("transition_ids must have the transition shape")
    if np.unique(result).size != result.size:
        raise ValueError("transition_ids must be unique")
    return result


def _refine_cell(
    callback: Callable[..., Any] | None,
    *,
    index: int,
    requested_bits: int,
    current: CertifiedUniformCell,
) -> CertifiedUniformCell:
    if callback is None:
        raise _spectral.JacobiRBCertificationError(
            "certified Haar uniform needs refinement but no callback was supplied",
            {
                "failure_kind": "haar_uniform_refinement_unavailable",
                "sample_index": int(index),
                "requested_bits": int(requested_bits),
            },
        )
    response = callback(
        UniformCellRefinementRequest(
            sample_index=int(index),
            requested_source_prefix_bits=int(requested_bits),
            current_cell=current,
        )
    )
    if not isinstance(response, UniformCellRefinementResult):
        raise TypeError(
            "refinement_callback must return UniformCellRefinementResult"
        )
    replacement = response.cell
    if replacement.lower < current.lower or replacement.upper > current.upper:
        raise ValueError("refined uniform cell must be nested in the previous cell")
    if replacement.width >= current.width:
        raise ValueError("refined uniform cell must be strictly narrower")
    return replacement


def _prefix_for_cell(
    cell: CertifiedUniformCell,
    *,
    max_bits: int,
    cuda_recorded_limit: bool,
) -> _EnclosingDyadicPrefix:
    numerator, bits = enclosing_dyadic_prefix(cell, max_bits=max_bits)
    if cuda_recorded_limit and bits > JACOBI_RECORDED_PREFIX_LIMIT_BITS:
        numerator >>= bits - JACOBI_RECORDED_PREFIX_LIMIT_BITS
        bits = JACOBI_RECORDED_PREFIX_LIMIT_BITS
    return _EnclosingDyadicPrefix(numerator, bits)


def sample_alpha1_rb_transition_batch_from_uniform_cells_cpu(
    head_fraction: Any,
    exposure: Any,
    uniform_cells: Iterable[CertifiedUniformCell],
    *,
    transition_ids: Any,
    profile: JacobiRBCudaProfile,
    refinement_callback: Callable[..., Any] | None = None,
) -> CertifiedRBIntervalBatch:
    """Arb-authorized reference inversion from arbitrary certified cells."""

    if not isinstance(profile, JacobiRBCudaProfile):
        raise TypeError("profile must be a JacobiRBCudaProfile")
    x, duration = np.broadcast_arrays(
        np.asarray(head_fraction, dtype=np.float64),
        np.asarray(exposure, dtype=np.float64),
    )
    if not np.all(np.isfinite(x)) or np.any((x < 0.0) | (x > 1.0)):
        raise ValueError("head_fraction must be finite and lie in [0,1]")
    if not np.all(np.isfinite(duration)) or np.any(duration < 0.0):
        raise ValueError("exposure must be finite and nonnegative")
    ids = _as_ids(transition_ids, x.shape)
    cells = list(_coerce_cells(uniform_cells, x.shape))
    reference_profile = _rb_cuda._reference_profile(profile)
    later = np.array(x, copy=True)
    target = np.zeros_like(x)
    active = duration > 0.0
    certified = np.zeros(x.shape, dtype=bool)
    fallback = np.zeros(x.shape, dtype=bool)
    quantile_lower = np.array(x, copy=True)
    quantile_upper = np.array(x, copy=True)
    target_lower = np.zeros_like(x)
    target_upper = np.zeros_like(x)
    prefix_bits = np.zeros(x.shape, dtype=np.int32)
    codes = np.zeros(x.shape, dtype=np.uint8)
    modes = np.zeros(x.shape, dtype=np.int32)
    started = time.perf_counter()

    for index, (x_value, exposure_value, is_active) in enumerate(
        zip(x.reshape(-1), duration.reshape(-1), active.reshape(-1), strict=True)
    ):
        if not bool(is_active):
            continue
        attempts = 0
        while True:
            cell = cells[index]
            try:
                prefix = _prefix_for_cell(
                    cell,
                    max_bits=int(profile.max_prefix_bits),
                    cuda_recorded_limit=False,
                )
                (
                    y_value,
                    _lower,
                    _upper,
                    _steps,
                    inverse_modes,
                    _escalations,
                    correctly_rounded,
                ) = _spectral._invert_one(
                    float(x_value),
                    float(exposure_value),
                    prefix,
                    reference_profile,
                )
                if not correctly_rounded:
                    raise _spectral.JacobiRBCertificationError(
                        "Jacobi inversion did not prove correct rounding",
                        {"failure_kind": "haar_jacobi_rounding_unresolved"},
                    )
                (
                    target_value,
                    target_interval,
                    target_modes,
                    _target_escalated,
                ) = _spectral._target_interval(
                    float(x_value),
                    float(y_value),
                    float(exposure_value),
                    reference_profile,
                )
                break
            except _spectral.JacobiRBCertificationError:
                maximum_attempts = max(
                    0,
                    (
                        int(profile.max_prefix_bits)
                        - int(profile.initial_prefix_bits)
                    )
                    // int(profile.prefix_block_bits)
                    - 1,
                )
                if attempts >= maximum_attempts:
                    raise
                requested = min(
                    int(profile.max_prefix_bits),
                    int(profile.initial_prefix_bits)
                    + (attempts + 2) * int(profile.prefix_block_bits),
                )
                cells[index] = _refine_cell(
                    refinement_callback,
                    index=index,
                    requested_bits=requested,
                    current=cells[index],
                )
                attempts += 1

        cell_lower, cell_upper = _spectral._rounding_cell(float(y_value))
        flat_index = np.unravel_index(index, x.shape)
        later[flat_index] = float(y_value)
        target[flat_index] = float(target_value)
        quantile_lower[flat_index] = fraction_to_float_down(cell_lower)
        quantile_upper[flat_index] = fraction_to_float_up(cell_upper)
        target_lower[flat_index] = min(
            float(target_value), float(target_interval.lower)
        )
        target_upper[flat_index] = max(
            float(target_value), float(target_interval.upper)
        )
        prefix_bits[flat_index] = int(prefix.bits)
        modes[flat_index] = max(int(inverse_modes), int(target_modes))
        codes[flat_index] = 15
        certified[flat_index] = True
        fallback[flat_index] = True

    elapsed = time.perf_counter() - started
    return CertifiedRBIntervalBatch(
        earlier_head_fraction=x,
        later_head_fraction=later,
        denoising_target=target,
        exposure=duration,
        transition_ids=ids,
        active_mask=active,
        certified_mask=certified,
        fallback_mask=fallback,
        quantile_lower=quantile_lower,
        quantile_upper=quantile_upper,
        target_lower=target_lower,
        target_upper=target_upper,
        prefix_bits=prefix_bits,
        certificate_codes=codes,
        mode_counts=modes,
        uniform_cells=tuple(cells),
        diagnostics={
            "sample_count": int(x.size),
            "active_count": int(active.sum()),
            "certified_count": int(certified.sum()),
            "fallback_count": int(fallback.sum()),
            "uncertified_count": int((active & ~certified).sum()),
            "approximation_count": 0,
            "correction_count": 0,
            "floor_count": 0,
            "limiter_count": 0,
            "renormalization_count": 0,
        },
        runtime_report={
            "adapter_version": HAAR_UNIFORM_JACOBI_ADAPTER_VERSION,
            "authorization_backend": "python-flint/Arb",
            "cuda_authorizing": False,
            "arb_authorizing": True,
            "elapsed_seconds": elapsed,
        },
    )


def _validate_cuda_inputs(
    head_fraction: Tensor,
    exposure: Tensor,
    uniform_lower: Tensor,
    uniform_upper: Tensor,
    transition_ids: Tensor,
) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for the CUDA adapter")
    values = (
        ("head_fraction", head_fraction, torch.float64),
        ("exposure", exposure, torch.float64),
        ("uniform_lower", uniform_lower, torch.float64),
        ("uniform_upper", uniform_upper, torch.float64),
        ("transition_ids", transition_ids, torch.uint64),
    )
    shape = head_fraction.shape
    device = head_fraction.device
    for name, value, dtype in values:
        if not isinstance(value, torch.Tensor) or not value.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if value.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}")
        if value.shape != shape or value.device != device:
            raise ValueError(f"{name} must share shape and device")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if not bool(torch.isfinite(head_fraction).all()):
        raise ValueError("head_fraction must be finite")
    if bool(((head_fraction < 0.0) | (head_fraction > 1.0)).any()):
        raise ValueError("head_fraction must lie in [0,1]")
    if not bool(torch.isfinite(exposure).all()) or bool((exposure < 0.0).any()):
        raise ValueError("exposure must be finite and nonnegative")
    active = exposure > 0
    valid_uniform = (
        torch.isfinite(uniform_lower)
        & torch.isfinite(uniform_upper)
        & (uniform_lower > 0.0)
        & (uniform_lower < uniform_upper)
        & (uniform_upper < 1.0)
    )
    if bool((active & ~valid_uniform).any()):
        raise ValueError(
            "active certified uniform bounds must satisfy 0 < lower < upper < 1"
        )


def _cells_from_cuda_bounds(
    lower: Tensor, upper: Tensor, active: Tensor
) -> tuple[CertifiedUniformCell, ...]:
    lower_values = lower.detach().cpu().reshape(-1).tolist()
    upper_values = upper.detach().cpu().reshape(-1).tolist()
    active_values = active.detach().cpu().reshape(-1).tolist()
    return tuple(
        (
            CertifiedUniformCell(
                Fraction.from_float(float(lo)), Fraction.from_float(float(hi))
            )
            if bool(is_active)
            else CertifiedUniformCell(Fraction(1, 4), Fraction(3, 4))
        )
        for lo, hi, is_active in zip(
            lower_values, upper_values, active_values, strict=True
        )
    )


def sample_alpha1_rb_transition_batch_cuda_from_uniform_cells(
    head_fraction: Tensor,
    exposure: Tensor,
    uniform_lower: Tensor,
    uniform_upper: Tensor,
    *,
    transition_ids: Tensor,
    refinement_callback: Callable[..., Any] | None,
    profile: JacobiRBCudaProfile,
    uniform_center_hi: Tensor | None = None,
    uniform_center_lo: Tensor | None = None,
    uniform_radius: Tensor | None = None,
    source_prefix_bits: Tensor | None = None,
) -> CertifiedRBCudaBatch:
    """Authorize exact Jacobi transitions directly from DD uniform cells.

    The normal/Haar kernel returns an outward binary64 view and, when supplied,
    its double-double centre/radius.  The latter goes directly into the strict
    Jacobi CDF rounding-cell proof.  No dyadic approximation participates in
    authorization.  Unresolved lanes alone escalate to the existing Arb
    interval inversion through the deterministic refinement callback.
    """

    if not isinstance(profile, JacobiRBCudaProfile):
        raise TypeError("profile must be a JacobiRBCudaProfile")
    _validate_cuda_inputs(
        head_fraction, exposure, uniform_lower, uniform_upper, transition_ids
    )
    device, shape = head_fraction.device, head_fraction.shape
    if source_prefix_bits is not None and (
        not isinstance(source_prefix_bits, torch.Tensor)
        or not source_prefix_bits.is_cuda
        or source_prefix_bits.dtype not in {torch.int32, torch.int64}
        or source_prefix_bits.shape != shape
        or source_prefix_bits.device != device
        or not source_prefix_bits.is_contiguous()
        or bool((source_prefix_bits < 1).any())
        or bool((source_prefix_bits > 1024).any())
    ):
        raise ValueError(
            "source_prefix_bits must be contiguous int32/int64 in [1,1024] "
            "on the common CUDA device"
        )

    def flat(value: Tensor) -> Tensor:
        return value.reshape(-1).contiguous()

    started = time.perf_counter()
    fused = launch_certified_jacobi_from_uniform_cells(
        flat(head_fraction),
        flat(exposure),
        flat(uniform_lower),
        flat(uniform_upper),
        flat(transition_ids),
        uniform_center_hi=(
            None if uniform_center_hi is None else flat(uniform_center_hi)
        ),
        uniform_center_lo=(
            None if uniform_center_lo is None else flat(uniform_center_lo)
        ),
        uniform_radius=(
            None if uniform_radius is None else flat(uniform_radius)
        ),
        threads_per_block=int(profile.threads_per_block),
        mode_cap=8192,
    )
    active = flat(exposure) > 0.0
    cuda_certified = fused.authorized_mask & active
    fallback = active & ~fused.authorized_mask
    later = fused.later.clone()
    target = fused.target.clone()
    q_lower = fused.quantile_lower.clone()
    q_upper = fused.quantile_upper.clone()
    z_lower = fused.target_lower.clone()
    z_upper = fused.target_upper.clone()
    modes = fused.modes_used.clone()
    codes = fused.certificate_codes.clone()
    fallback_modes = torch.zeros_like(modes)
    used_prefix_bits = (
        torch.zeros_like(modes)
        if source_prefix_bits is None
        else flat(source_prefix_bits).to(dtype=torch.int32).clone()
    )
    arb_elapsed = 0.0
    if bool(fallback.any()):
        arb_started = time.perf_counter()
        fallback_indices = torch.nonzero(
            fallback, as_tuple=False
        ).reshape(-1)
        original_indices = [
            int(value)
            for value in fallback_indices.detach().cpu().tolist()
        ]
        cells = _cells_from_cuda_bounds(
            flat(uniform_lower)
            .index_select(0, fallback_indices)
            .contiguous(),
            flat(uniform_upper)
            .index_select(0, fallback_indices)
            .contiguous(),
            torch.ones(
                len(original_indices),
                dtype=torch.bool,
                device=device,
            ),
        )
        refined_source_bits: dict[int, int] = {}

        def remapped_refinement(
            request: UniformCellRefinementRequest,
        ) -> UniformCellRefinementResult:
            local_index = int(request.sample_index)
            if not 0 <= local_index < len(original_indices):
                raise IndexError("fallback refinement index is out of range")
            if refinement_callback is None:
                raise _spectral.JacobiRBCertificationError(
                    "certified Haar uniform needs refinement but no callback "
                    "was supplied",
                    {
                        "failure_kind": "haar_uniform_refinement_unavailable",
                        "sample_index": original_indices[local_index],
                    },
                )
            response = refinement_callback(
                replace(
                    request,
                    sample_index=original_indices[local_index],
                )
            )
            if not isinstance(response, UniformCellRefinementResult):
                raise TypeError(
                    "refinement_callback must return "
                    "UniformCellRefinementResult"
                )
            refined_source_bits[local_index] = int(
                response.source_prefix_bits
            )
            return response

        host = sample_alpha1_rb_transition_batch_from_uniform_cells_cpu(
            flat(head_fraction)
            .index_select(0, fallback_indices)
            .detach()
            .cpu()
            .numpy(),
            flat(exposure)
            .index_select(0, fallback_indices)
            .detach()
            .cpu()
            .numpy(),
            cells,
            transition_ids=(
                flat(transition_ids)
                .index_select(0, fallback_indices)
                .detach()
                .cpu()
                .numpy()
            ),
            profile=profile,
            refinement_callback=remapped_refinement,
        )

        def host_tensor(value: Any, dtype: Any = torch.float64) -> Tensor:
            return torch.as_tensor(
                value, dtype=dtype, device=device
            ).reshape(-1)

        later.index_copy_(
            0, fallback_indices, host_tensor(host.later_head_fraction)
        )
        target.index_copy_(
            0, fallback_indices, host_tensor(host.denoising_target)
        )
        q_lower.index_copy_(
            0, fallback_indices, host_tensor(host.quantile_lower)
        )
        q_upper.index_copy_(
            0, fallback_indices, host_tensor(host.quantile_upper)
        )
        z_lower.index_copy_(
            0, fallback_indices, host_tensor(host.target_lower)
        )
        z_upper.index_copy_(
            0, fallback_indices, host_tensor(host.target_upper)
        )
        host_modes = host_tensor(host.mode_counts, torch.int32)
        modes.index_copy_(0, fallback_indices, host_modes)
        fallback_modes.index_copy_(0, fallback_indices, host_modes)
        codes.index_copy_(
            0,
            fallback_indices,
            host_tensor(host.certificate_codes, torch.uint8),
        )
        fallback_prefix_values = []
        host_prefix_values = np.asarray(host.prefix_bits).reshape(-1)
        for local_index, original_index in enumerate(original_indices):
            if local_index in refined_source_bits:
                fallback_prefix_values.append(
                    refined_source_bits[local_index]
                )
            elif source_prefix_bits is not None:
                fallback_prefix_values.append(
                    int(
                        flat(source_prefix_bits)[original_index].item()
                    )
                )
            else:
                fallback_prefix_values.append(
                    int(host_prefix_values[local_index])
                )
        used_prefix_bits.index_copy_(
            0,
            fallback_indices,
            torch.tensor(
                fallback_prefix_values,
                dtype=torch.int32,
                device=device,
            ),
        )
        arb_elapsed = time.perf_counter() - arb_started
    certified = (~active) | cuda_certified | fallback
    finite = (
        torch.isfinite(later)
        & torch.isfinite(target)
        & torch.isfinite(q_lower)
        & torch.isfinite(q_upper)
        & torch.isfinite(z_lower)
        & torch.isfinite(z_upper)
    )
    zero_i64 = torch.tensor(0, dtype=torch.int64, device=device)
    diagnostics: dict[str, Tensor] = {
        "sample_count": torch.tensor(
            int(head_fraction.numel()), dtype=torch.int64, device=device
        ),
        "active_count": active.sum(dtype=torch.int64),
        "certified_count": (certified & active).sum(dtype=torch.int64),
        "cuda_authorized_count": cuda_certified.sum(dtype=torch.int64),
        "fallback_count": fallback.sum(dtype=torch.int64),
        "candidate_count": active.sum(dtype=torch.int64),
        "candidate_match_count": zero_i64,
        "strengthened_count": zero_i64,
        "resource_cap_count": zero_i64,
        "invalid_density_count": zero_i64,
        "nonfinite_count": (active & ~finite).sum(dtype=torch.int64),
        "approximation_count": zero_i64,
        "candidate_repair_count": zero_i64,
        "correction_count": zero_i64,
        "floor_count": zero_i64,
        "limiter_count": zero_i64,
        "projection_count": zero_i64,
        "renormalization_count": zero_i64,
        "uniform_interval_count": torch.tensor(
            int(head_fraction.numel()), dtype=torch.int64, device=device
        ),
        "fused_authorizer_launch_count": torch.tensor(
            int(fused.launch_count), dtype=torch.int64, device=device
        ),
        "maximum_cuda_modes": modes.max() if modes.numel() else zero_i64,
        "maximum_arb_fallback_modes": (
            fallback_modes.max() if fallback_modes.numel() else zero_i64
        ),
        "fused_authorizer_elapsed_seconds": torch.tensor(
            float(fused.elapsed_seconds), dtype=torch.float64, device=device
        ),
        "arb_fallback_elapsed_seconds": torch.tensor(
            float(arb_elapsed), dtype=torch.float64, device=device
        ),
    }
    elapsed = time.perf_counter() - started
    active_count = int(active.sum().item())
    fallback_count = int(fallback.sum().item())
    runtime = {
        "uniform_adapter_version": HAAR_UNIFORM_JACOBI_ADAPTER_VERSION,
        "fused_haar_cuda_version": HAAR_FUSED_CUDA_VERSION,
        "arbitrary_uniform_interval_authorizing": True,
        "arbitrary_uniform_cuda_authorizer_available": True,
        "direct_dd_uniform_cell_authorization": True,
        "gaussian_transition_approximation": False,
        "source_refinement_callback_available": int(
            refinement_callback is not None
        ),
        "arb_fallback_fraction": (
            fallback_count / active_count if active_count else 0.0
        ),
        "arb_fallback_time_fraction": (
            arb_elapsed / elapsed if elapsed > 0.0 else 0.0
        ),
        "fused_source_sha256": fused.bundle.source_sha256,
        "fused_binary_sha256": fused.bundle.binary_sha256,
        "elapsed_seconds": elapsed,
        "profile": profile.to_dict(),
    }
    candidate = fused.candidate
    return CertifiedRBCudaBatch(
        earlier_head_fraction=head_fraction,
        later_head_fraction=later.reshape(shape),
        denoising_target=target.reshape(shape),
        exposure=exposure,
        transition_ids=transition_ids,
        active_mask=active.reshape(shape),
        certified_mask=certified.reshape(shape),
        candidate_later_head_fraction=candidate.reshape(shape),
        candidate_denoising_target=torch.zeros_like(candidate).reshape(shape),
        candidate_match_mask=torch.zeros_like(active).reshape(shape),
        cuda_certified_mask=cuda_certified.reshape(shape),
        fallback_mask=fallback.reshape(shape),
        strengthened_mask=torch.zeros_like(active).reshape(shape),
        arb_fallback_reason_codes=fused.fallback_reason_codes.reshape(shape),
        arb_fallback_mode_counts=fallback_modes.reshape(shape),
        mode_counts=modes.reshape(shape),
        quantile_lower=q_lower.reshape(shape),
        quantile_upper=q_upper.reshape(shape),
        target_lower=z_lower.reshape(shape),
        target_upper=z_upper.reshape(shape),
        prefix_bits=used_prefix_bits.reshape(shape),
        certificate_codes=codes.reshape(shape),
        diagnostics=diagnostics,
        runtime_report=runtime,
    )


sample_alpha1_rb_transition_batch_cuda_from_uniform_cells.haar_interval_authorizer_contract = {
    "arbitrary_uniform_jacobi_authorizer": True,
    "direct_dd_uniform_cell_authorization": True,
    "fused_cuda_authorizer": True,
    "version": HAAR_UNIFORM_JACOBI_ADAPTER_VERSION,
}


__all__ = [
    "HAAR_UNIFORM_JACOBI_ADAPTER_VERSION",
    "JACOBI_RECORDED_PREFIX_LIMIT_BITS",
    "CertifiedRBIntervalBatch",
    "enclosing_dyadic_prefix",
    "sample_alpha1_rb_transition_batch_cuda_from_uniform_cells",
    "sample_alpha1_rb_transition_batch_from_uniform_cells_cpu",
]
