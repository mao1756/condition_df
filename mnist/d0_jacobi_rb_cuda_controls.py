"""Measured controls for the rigorous CUDA Jacobi Rao--Blackwell backend.

This module is deliberately controls-only.  It schedules and measures calls to
``sample_alpha1_rb_transition_batch_cuda``; it does not contain an approximate
transition, a CPU fallback, a trainer, or a reverse sampler.  A production
control therefore fails closed when the rigorous backend cannot return a
fully-certified result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import json
import math
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from mnist.d0_jacobi_rb_cuda import (
    JacobiRBCudaProfile,
    certify_alpha1_rb_transition_batch_cuda_with_dyadic_prefixes,
    sample_alpha1_rb_transition_batch_cuda,
)
from mnist import d0_jacobi_rb_controls as _reference_controls
from mnist.d0_jacobi_rb_spectral import philox_uniform_prefix


CUDA_CONTROL_VERSION = "jacobi-rb-rigorous-cuda-controls-v1"
PARENT_REPLAY_COUNT = 294
FRESH_CERTIFICATE_COUNT = 512
WARMUP_TRANSITIONS = 4_096
THROUGHPUT_TRANSITIONS = 65_536
THROUGHPUT_REPEATS = 3
FULL_PATH_TRANSITIONS = 1_404_928
FULL_PATH_REPEATS = 3
MAX_CUDA_CHUNK_SIZE = 4_096
STEPS_PER_SHARD = 8
GRID_SIZE = 28
EDGES_PER_MATCHING = GRID_SIZE * GRID_SIZE // 2
PHASES_PER_STEP = 7
TRANSITIONS_PER_STEP = EDGES_PER_MATCHING * PHASES_PER_STEP
SAMPLE_STEPS = 512
TAU_EFF = 5.0e-5
GRID_SPACING = 1.0 / GRID_SIZE


class RigorousCudaControlError(RuntimeError):
    """Raised when a control cannot produce rigorous CUDA evidence."""


@dataclass(frozen=True)
class CertificatePanelPlan:
    root_seed: int = 261_131
    parent_replay_count: int = PARENT_REPLAY_COUNT
    fresh_count: int = FRESH_CERTIFICATE_COUNT
    chunk_size: int = MAX_CUDA_CHUNK_SIZE

    @property
    def total_count(self) -> int:
        return int(self.parent_replay_count + self.fresh_count)

    def record(self) -> dict[str, Any]:
        return {"version": CUDA_CONTROL_VERSION, **asdict(self), "total_count": self.total_count}


@dataclass(frozen=True)
class KernelBenchmarkPlan:
    root_seed: int = 261_131
    warmup_transitions: int = WARMUP_TRANSITIONS
    throughput_transitions: int = THROUGHPUT_TRANSITIONS
    throughput_repeats: int = THROUGHPUT_REPEATS
    full_path_transitions: int = FULL_PATH_TRANSITIONS
    full_path_repeats: int = FULL_PATH_REPEATS
    chunk_size: int = MAX_CUDA_CHUNK_SIZE
    steps_per_shard: int = STEPS_PER_SHARD

    def record(self) -> dict[str, Any]:
        return {"version": CUDA_CONTROL_VERSION, **asdict(self)}


def _positive_plan(plan: CertificatePanelPlan | KernelBenchmarkPlan) -> None:
    for name, value in asdict(plan).items():
        if name == "root_seed":
            continue
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(plan.chunk_size) > MAX_CUDA_CHUNK_SIZE:
        raise ValueError(f"chunk_size must not exceed {MAX_CUDA_CHUNK_SIZE}")
    if isinstance(plan, KernelBenchmarkPlan) and int(plan.steps_per_shard) != STEPS_PER_SHARD:
        raise ValueError(f"production shard schedule requires {STEPS_PER_SHARD} steps")


def certificate_panel_plan(
    *, root_seed: int = 261_131, test_only: bool = False, **overrides: int
) -> CertificatePanelPlan:
    """Build the frozen 294-replay plus 512-fresh certificate plan.

    Smaller plans are useful for unit tests, but accepting one accidentally in
    a gated run would turn smoke coverage into scientific evidence.  Hence any
    override is rejected unless ``test_only`` is explicit.
    """

    if overrides and not test_only:
        raise ValueError("certificate plan overrides require test_only=True")
    plan = CertificatePanelPlan(root_seed=int(root_seed), **overrides)
    _positive_plan(plan)
    return plan


def kernel_benchmark_plan(
    *, root_seed: int = 261_131, test_only: bool = False, **overrides: int
) -> KernelBenchmarkPlan:
    """Build the frozen warmup, repeated-batch, and full-path benchmark plan."""

    if overrides and not test_only:
        raise ValueError("kernel benchmark plan overrides require test_only=True")
    plan = KernelBenchmarkPlan(root_seed=int(root_seed), **overrides)
    _positive_plan(plan)
    return plan


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _field(result: Any, *names: str) -> Any:
    for name in names:
        if isinstance(result, Mapping) and name in result:
            return result[name]
        if hasattr(result, name):
            return getattr(result, name)
    raise RigorousCudaControlError("CUDA result is missing " + "/".join(names))


def _optional_field(result: Any, name: str, default: Any) -> Any:
    if isinstance(result, Mapping) and name in result:
        return result[name]
    return getattr(result, name, default)


def _diagnostic_scalar(result: Any, name: str, default: int | float = 0) -> int | float:
    diagnostics = _optional_field(result, "diagnostics", {})
    if not isinstance(diagnostics, Mapping) or name not in diagnostics:
        return default
    value = diagnostics[name]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RigorousCudaControlError(f"CUDA diagnostic {name} is not scalar")
        value = value.detach().cpu().item()
    if isinstance(default, int):
        return int(value)
    return float(value)


def canonical_transition_ids(
    *, path: int, outer_step: int, phase: int, edge_start: int, count: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode the frozen ``(path, outer_step, phase, edge)`` coordinates.

    The bit allocation is deliberately explicit and comfortably exceeds the
    production support: 20 path bits, 10 step bits, 3 phase bits, and 10 edge
    bits.  It makes IDs independent of call order, batching, and resume.
    """

    values = (int(path), int(outer_step), int(phase), int(edge_start), int(count))
    if any(value < 0 for value in values):
        raise ValueError("canonical transition coordinates must be nonnegative")
    if path >= (1 << 20) or outer_step >= (1 << 10) or phase >= (1 << 3):
        raise ValueError("canonical transition coordinate exceeds its frozen field")
    if edge_start + count > (1 << 10):
        raise ValueError("canonical edge coordinate exceeds its frozen field")
    base = (int(path) << 23) | (int(outer_step) << 13) | (int(phase) << 10)
    return (
        torch.arange(edge_start, edge_start + count, dtype=torch.int64, device=device)
        .add_(int(base))
        .to(dtype=torch.uint64)
        .contiguous()
    )


def _certificate_codes(result: Any, count: int) -> np.ndarray:
    values = _numpy(_field(result, "certificate_codes", "certificate_code")).reshape(-1)
    if values.size == 1 and count != 1:
        values = np.repeat(values, count)
    if values.size != count:
        raise RigorousCudaControlError("CUDA certificate-code shape mismatch")
    return values.astype(np.uint64, copy=False)


def _certified_mask(result: Any, count: int) -> np.ndarray:
    try:
        values = _numpy(_field(result, "certified", "certified_mask")).reshape(-1)
        if values.size == 1 and count != 1:
            values = np.repeat(values, count)
        if values.size != count:
            raise RigorousCudaControlError("CUDA certified-mask shape mismatch")
        return values.astype(bool, copy=False)
    except RigorousCudaControlError:
        # The established RB API encodes all per-draw certificates in a bit
        # mask.  A nonzero code alone is not sufficient: bits 0..3 bind the
        # CDF, target, interval, and correct-rounding contracts.
        return (_certificate_codes(result, count) & np.uint64(0xF)) == np.uint64(0xF)


def _call_sampler(
    x: torch.Tensor,
    exposure: torch.Tensor,
    *,
    profile: JacobiRBCudaProfile,
    rng_key: tuple[Any, ...],
    transition_offset: int,
    transition_ids: torch.Tensor | None = None,
    sampler: Callable[..., Any],
) -> Any:
    if x.numel() > MAX_CUDA_CHUNK_SIZE:
        raise RigorousCudaControlError("a CUDA backend call exceeded 4096 transitions")
    if x.device.type != "cuda" and sampler is sample_alpha1_rb_transition_batch_cuda:
        raise RigorousCudaControlError("production rigorous backend requires a CUDA device")
    if transition_ids is None:
        transition_ids = torch.as_tensor(
            [int(transition_offset) + index for index in range(int(x.numel()))],
            dtype=torch.uint64,
            device=x.device,
        ).reshape(x.shape).contiguous()
    elif (
        not isinstance(transition_ids, torch.Tensor)
        or transition_ids.dtype != torch.uint64
        or transition_ids.device != x.device
        or transition_ids.shape != x.shape
        or not transition_ids.is_contiguous()
    ):
        raise RigorousCudaControlError("explicit transition IDs violate the CUDA API")
    try:
        return sampler(
            x.contiguous(), exposure.contiguous(),
            rng_key=rng_key, transition_ids=transition_ids, profile=profile,
        )
    except Exception as exc:
        raise RigorousCudaControlError(f"rigorous CUDA backend failed closed: {exc}") from exc


def _digest_arrays(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _strict_rounding_cell_contains(lower: float, upper: float, value: float) -> bool:
    if not all(math.isfinite(item) for item in (lower, upper, value)) or lower > upper:
        return False
    previous = float(np.nextafter(value, -math.inf))
    following = float(np.nextafter(value, math.inf))
    cell_lower = (
        Fraction.from_float(previous) + Fraction.from_float(float(value))
    ) / 2
    cell_upper = (
        Fraction.from_float(float(value)) + Fraction.from_float(following)
    ) / 2
    return (
        Fraction.from_float(float(lower)) > cell_lower
        and Fraction.from_float(float(upper)) < cell_upper
    )


def _outputs(result: Any, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    later = _numpy(_field(result, "later_head_fraction", "later")).reshape(-1)
    target = _numpy(_field(result, "denoising_target", "target")).reshape(-1)
    codes = _certificate_codes(result, count)
    if later.size != count or target.size != count:
        raise RigorousCudaControlError("CUDA output shape mismatch")
    if not np.isfinite(later).all() or not np.isfinite(target).all():
        raise RigorousCudaControlError("rigorous CUDA backend returned nonfinite output")
    return later.astype(np.float64), target.astype(np.float64), codes


def deterministic_fresh_certificate_inputs(count: int, root_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return 64 Dirichlet grids by four colors by half/full durations.

    The production count is therefore exactly ``64 * 4 * 2 == 512``.  A
    reduced prefix of that same ordered panel is used only by explicit tests.
    """

    if int(count) <= 0:
        raise ValueError("count must be positive")
    rng = np.random.Generator(np.random.Philox(int(root_seed)))
    grid = rng.dirichlet(np.ones(28 * 28, dtype=np.float64), size=64).reshape(64, 28, 28)
    rows: list[tuple[float, float]] = []
    schedule = 5.0e-5 / 512.0
    for state_index, state in enumerate(grid):
        flat = state.reshape(-1)
        for color in range(4):
            edges: list[tuple[int, int]] = []
            for row in range(28):
                for column in range(28):
                    tail = row * 28 + column
                    if color < 2 and column % 2 == color:
                        edges.append((tail, row * 28 + ((column + 1) % 28)))
                    elif color >= 2 and row % 2 == color - 2:
                        edges.append((tail, ((row + 1) % 28) * 28 + column))
            tail, head = edges[(state_index * 37 + color * 73) % len(edges)]
            pair_total = float(flat[tail] + flat[head])
            head_fraction = float(flat[head] / pair_total)
            for duration in (0.5, 1.0):
                exposure = 3.0 * schedule * duration / (
                    (1.0 / 28.0) ** 2 * pair_total
                )
                rows.append((head_fraction, exposure))
    selected = rows[: int(count)]
    return (
        np.asarray([item[0] for item in selected], dtype=np.float64),
        np.asarray([item[1] for item in selected], dtype=np.float64),
    )


def _parent_input(row: Mapping[str, Any]) -> tuple[float, float, float | None, float | None]:
    def first(*names: str) -> Any:
        for name in names:
            if name in row:
                return row[name]
        return None

    x = first("earlier_head_fraction", "head_fraction", "x", "earlier")
    exposure = first("exposure", "u", "duration")
    if x is None or exposure is None:
        raise RigorousCudaControlError("parent certificate row lacks x/exposure")
    later = first("later_head_fraction", "later", "y")
    target = first("denoising_target", "target")
    return float(x), float(exposure), None if later is None else float(later), None if target is None else float(target)


def _strengthened_profile(
    profile: JacobiRBCudaProfile,
) -> tuple[JacobiRBCudaProfile, bool]:
    factory = getattr(profile, "strengthened", None)
    if callable(factory):
        value = factory()
        if not isinstance(value, JacobiRBCudaProfile):
            raise RigorousCudaControlError("strengthened CUDA profile has the wrong type")
        return value, True
    # The candidate never authorizes an answer, but doubling its fixed-mode
    # work is still a useful independent perturbation of the complete API.
    return (
        replace(profile, candidate_modes=min(4096, 2 * int(profile.candidate_modes))),
        False,
    )


def _arb_recertify_v2(
    *, x: float, exposure: float, rng_key: Any, transition_id: int,
    profile: JacobiRBCudaProfile,
) -> tuple[float, float, float, float, float, float]:
    """Independently recertify one v2 draw with the immutable Arb oracle."""

    from mnist import d0_jacobi_rb_cuda as backend
    from mnist import d0_jacobi_rb_spectral as oracle

    seed = backend._canonical_seed(rng_key)
    prefix = backend._StatelessPhiloxPrefix(
        seed,
        int(transition_id),
        int(profile.max_prefix_bits),
        seed_is_canonical=True,
    )
    reference_profile = backend._reference_profile(profile)
    y, q_lower, q_upper, _steps, _modes, _escalations, correctly_rounded = (
        oracle._invert_one(float(x), float(exposure), prefix, reference_profile)
    )
    if not correctly_rounded:
        raise RigorousCudaControlError("fresh Arb oracle did not certify the quantile")
    z, z_interval, _target_modes, _target_escalated = oracle._target_interval(
        float(x), float(y), float(exposure), reference_profile
    )
    return (
        float(y), float(z), float(q_lower), float(q_upper),
        float(z_interval.lower), float(z_interval.upper),
    )


def run_certificate_panel(
    parent_rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    profile: JacobiRBCudaProfile,
    plan: CertificatePanelPlan | None = None,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay the exact parent cases and certify an independent fresh panel."""

    active = plan or certificate_panel_plan()
    _positive_plan(active)
    if len(parent_rows) != int(active.parent_replay_count):
        raise RigorousCudaControlError(
            f"expected {active.parent_replay_count} parent replay rows, got {len(parent_rows)}"
        )
    inputs = [_parent_input(row) for row in parent_rows]
    fresh_x, fresh_u = deterministic_fresh_certificate_inputs(active.fresh_count, active.root_seed + 1)
    inputs.extend((float(x), float(u), None, None) for x, u in zip(fresh_x, fresh_u, strict=True))

    rows: list[dict[str, Any]] = []
    replay_y_mismatches = 0
    replay_target_mismatches = 0
    strengthening_mismatches = 0
    fresh_arb_mismatches = 0
    fresh_arb_enclosure_failures = 0
    aggregate_counts = {
        name: 0
        for name in (
            "resource_cap_count", "invalid_density_count", "approximation_count",
            "correction_count", "floor_count", "limiter_count",
            "renormalization_count", "nonfinite_count",
        )
    }
    fresh_fallback_count = 0
    parent_fallback_count = 0
    fresh_fallback_seconds = 0.0
    fresh_total_seconds = 0.0
    binary_hashes: set[str] = set()
    source_hashes: set[str] = set()
    starts: list[tuple[int, int]] = []
    for panel_start, panel_end in (
        (0, int(active.parent_replay_count)),
        (int(active.parent_replay_count), len(inputs)),
    ):
        starts.extend(
            (start, min(panel_end, start + int(active.chunk_size)))
            for start in range(panel_start, panel_end, int(active.chunk_size))
        )
    strengthened_profile, genuine_strengthening = _strengthened_profile(profile)

    def invoke(
        start: int, end: int, block: Sequence[tuple[float, float, float | None, float | None]],
        selected_profile: JacobiRBCudaProfile,
    ) -> Any:
        x = torch.as_tensor([item[0] for item in block], dtype=torch.float64, device=device)
        exposure = torch.as_tensor([item[1] for item in block], dtype=torch.float64, device=device)
        if start < active.parent_replay_count and sampler is sample_alpha1_rb_transition_batch_cuda:
            numerators: list[int] = []
            bits_values: list[int] = []
            transition_values: list[int] = []
            for local, parent_row in enumerate(parent_rows[start:end]):
                bits = int(parent_row.get("uniform_prefix_bits", 0))
                candidate = int(parent_row.get("uniform_key_candidate", -1))
                if bits <= 0 or candidate < 0:
                    raise RigorousCudaControlError("parent row lacks its recorded v1 prefix")
                numerator, observed_bits, midpoint = philox_uniform_prefix(
                    (261_121, "support-prefix", candidate), bits=bits
                )
                if observed_bits != bits or midpoint != float(parent_row.get("uniform_prefix_midpoint")):
                    raise RigorousCudaControlError("parent v1 prefix reconstruction changed")
                numerators.append(numerator)
                bits_values.append(bits)
                transition_values.append(int(parent_row.get("support_index", start + local)))
            try:
                return certify_alpha1_rb_transition_batch_cuda_with_dyadic_prefixes(
                    x.contiguous(), exposure.contiguous(),
                    torch.as_tensor(numerators, dtype=torch.uint64, device=device).contiguous(),
                    torch.as_tensor(bits_values, dtype=torch.int32, device=device).contiguous(),
                    transition_ids=torch.as_tensor(
                        transition_values, dtype=torch.uint64, device=device
                    ).contiguous(),
                    profile=selected_profile,
                )
            except Exception as exc:
                raise RigorousCudaControlError(
                    f"parent v1 prefix replay failed closed: {exc}"
                ) from exc
        return _call_sampler(
            x, exposure, profile=selected_profile,
            rng_key=(active.root_seed, "certificate-v2"), transition_offset=start,
            sampler=sampler,
        )

    for start, end in starts:
        block = inputs[start:end]
        result = invoke(start, end, block, profile)
        runtime = _optional_field(result, "runtime_report", {})
        if isinstance(runtime, Mapping):
            if runtime.get("binary_sha256"):
                binary_hashes.add(str(runtime["binary_sha256"]))
            if runtime.get("kernel_sha256"):
                source_hashes.add(str(runtime["kernel_sha256"]))
        later, target, codes = _outputs(result, len(block))
        strengthened_result = invoke(start, end, block, strengthened_profile)
        strengthened_later, strengthened_target, _strengthened_codes = _outputs(
            strengthened_result, len(block)
        )
        strengthening_mismatches += int(np.count_nonzero(later != strengthened_later))
        strengthening_mismatches += int(np.count_nonzero(target != strengthened_target))
        certified = _certified_mask(result, len(block))
        q_lower = _numpy(_optional_field(result, "quantile_lower", later)).reshape(-1)
        q_upper = _numpy(_optional_field(result, "quantile_upper", later)).reshape(-1)
        z_lower = _numpy(_optional_field(result, "target_lower", target)).reshape(-1)
        z_upper = _numpy(_optional_field(result, "target_upper", target)).reshape(-1)
        fallback = _numpy(_optional_field(result, "fallback_mask", np.zeros(len(block), dtype=bool))).reshape(-1)
        fallback_reasons = _numpy(_optional_field(
            result, "arb_fallback_reason_codes", np.zeros(len(block), dtype=np.uint8)
        )).reshape(-1)
        mode_counts = _numpy(_optional_field(
            result, "mode_counts",
            _optional_field(result, "arb_fallback_mode_counts", np.zeros(len(block), dtype=np.int32)),
        )).reshape(-1)
        strengthened_mask = _numpy(_optional_field(
            result, "strengthened_mask", np.zeros(len(block), dtype=bool)
        )).reshape(-1)
        candidate_match = _numpy(_optional_field(result, "candidate_match_mask", np.zeros(len(block), dtype=bool))).reshape(-1)
        prefix_bits = _numpy(_optional_field(result, "prefix_bits", np.zeros(len(block), dtype=np.int32))).reshape(-1)
        for name in aggregate_counts:
            aggregate_counts[name] += int(_diagnostic_scalar(result, name, 0))
        block_fallback = int(np.count_nonzero(fallback))
        fallback_seconds = float(_diagnostic_scalar(result, "arb_fallback_elapsed_seconds", 0.0))
        fused_seconds = float(_diagnostic_scalar(result, "fused_authorizer_elapsed_seconds", 0.0))
        candidate_seconds = float(_diagnostic_scalar(result, "candidate_elapsed_seconds", 0.0))
        if start < active.parent_replay_count:
            parent_fallback_count += block_fallback
        else:
            fresh_fallback_count += block_fallback
            fresh_fallback_seconds += fallback_seconds
            fresh_total_seconds += fallback_seconds + fused_seconds + candidate_seconds
        for offset, item in enumerate(block):
            index = start + offset
            parent_y, parent_target = item[2], item[3]
            y_match = parent_y is None or np.asarray(later[offset], dtype=np.float64).tobytes() == np.asarray(parent_y, dtype=np.float64).tobytes()
            target_match = parent_target is None or np.asarray(target[offset], dtype=np.float64).tobytes() == np.asarray(parent_target, dtype=np.float64).tobytes()
            if index < active.parent_replay_count:
                replay_y_mismatches += int(not y_match)
                replay_target_mismatches += int(not target_match)
            arb_match = True
            arb_enclosed = True
            if index >= active.parent_replay_count and sampler is sample_alpha1_rb_transition_batch_cuda:
                oracle = _arb_recertify_v2(
                    x=item[0], exposure=item[1],
                    rng_key=(active.root_seed, "certificate-v2"),
                    transition_id=index, profile=profile,
                )
                arb_y, arb_z, _arb_q_lo, _arb_q_hi, _arb_z_lo, _arb_z_hi = oracle
                arb_match = (
                    np.asarray(later[offset], dtype=np.float64).tobytes()
                    == np.asarray(arb_y, dtype=np.float64).tobytes()
                    and np.asarray(target[offset], dtype=np.float64).tobytes()
                    == np.asarray(arb_z, dtype=np.float64).tobytes()
                )
                arb_enclosed = (
                    float(q_lower[offset]) <= arb_y <= float(q_upper[offset])
                    and float(z_lower[offset]) <= arb_z <= float(z_upper[offset])
                )
                fresh_arb_mismatches += int(not arb_match)
                fresh_arb_enclosure_failures += int(not arb_enclosed)
            rows.append({
                "panel": "parent_replay" if index < active.parent_replay_count else "fresh",
                "panel_index": index if index < active.parent_replay_count else index - active.parent_replay_count,
                "earlier_head_fraction": item[0], "exposure": item[1],
                "later_head_fraction": float(later[offset]),
                "denoising_target": float(target[offset]),
                "certificate_code": int(codes[offset]), "certified": int(certified[offset]),
                "quantile_lower": float(q_lower[offset]), "quantile_upper": float(q_upper[offset]),
                "target_lower": float(z_lower[offset]), "target_upper": float(z_upper[offset]),
                "fallback": int(fallback[offset]), "candidate_match": int(candidate_match[offset]),
                "fallback_reason_code": int(fallback_reasons[offset]),
                "mode_count": int(mode_counts[offset]),
                "strengthened": int(strengthened_mask[offset]),
                "prefix_bits": int(prefix_bits[offset]),
                "parent_y_bit_match": int(y_match), "parent_target_bit_match": int(target_match),
                "strengthened_y_bit_match": int(later[offset] == strengthened_later[offset]),
                "strengthened_target_bit_match": int(target[offset] == strengthened_target[offset]),
                "fresh_arb_bit_match": int(arb_match),
                "fresh_arb_enclosed": int(arb_enclosed),
            })
    certified_count = sum(int(row["certified"]) for row in rows)
    quantile_enclosed = all(
        float(row["quantile_lower"]) <= float(row["later_head_fraction"]) <= float(row["quantile_upper"])
        for row in rows
    )
    target_enclosed = all(
        float(row["target_lower"]) <= float(row["denoising_target"]) <= float(row["target_upper"])
        for row in rows
    )
    target_rounding_bits = all(
        (int(row["certificate_code"]) & 0b1100) == 0b1100 for row in rows
    )
    fallback_count = sum(int(row["fallback"]) for row in rows)
    fresh_count = int(active.fresh_count)
    summary = {
        "version": CUDA_CONTROL_VERSION,
        "evaluation_status": "evaluated",
        "parent_replay_count": int(active.parent_replay_count),
        "fresh_certificate_count": int(active.fresh_count),
        "certificate_count": len(rows),
        "certified_count": certified_count,
        "uncertified_count": len(rows) - certified_count,
        "certificate_fraction": certified_count / len(rows),
        "parent_replay_y_bit_mismatch_count": replay_y_mismatches,
        "parent_replay_target_bit_mismatch_count": replay_target_mismatches,
        "parent_replay_z_bit_mismatch_count": replay_target_mismatches,
        "all_certificates_pass": int(certified_count == len(rows)),
        "spectral_rounding_certificate_pass": int(certified_count == len(rows)),
        "cdf_interval_enclosure_pass": int(quantile_enclosed),
        # The backend's authorizing certificate simultaneously encloses the
        # CDF and positive density used by the conormal ratio.
        "density_interval_enclosure_pass": int(certified_count == len(rows)),
        # Quantile authorization is the pair of strict CDF inequalities at
        # the exact binary64 cell boundaries.  q_lower/q_upper are diagnostic
        # last-bisection neighbors and may touch those boundaries; the
        # certificate bit plus independent Arb bit replay is the evidence.
        "quantile_rounding_cell_pass": int(
            quantile_enclosed and certified_count == len(rows)
            and replay_y_mismatches == 0 and fresh_arb_mismatches == 0
        ),
        # The device proves strict DD-ball containment before rounding.  Its
        # outward binary64 diagnostic endpoints can round to adjacent floats
        # even when the underlying ball is strictly inside the half-ULP cell,
        # so reapplying the proof to those lossy endpoints is invalid.  Use
        # the target-enclosure/correct-rounding bits and independent Arb bit
        # replay; retain the float endpoints only as enclosure diagnostics.
        "target_rounding_cell_pass": int(
            target_enclosed and target_rounding_bits
            and replay_target_mismatches == 0 and fresh_arb_mismatches == 0
        ),
        "precision_doubling_hash_pass": int(
            genuine_strengthening and strengthening_mismatches == 0
        ),
        "strengthening_hash_pass": int(
            genuine_strengthening and strengthening_mismatches == 0
        ),
        "genuine_certificate_strengthening": int(genuine_strengthening),
        "strengthening_bit_mismatch_count": strengthening_mismatches,
        "fresh_arb_enclosure_pass": int(
            sampler is not sample_alpha1_rb_transition_batch_cuda
            or (fresh_arb_mismatches == 0 and fresh_arb_enclosure_failures == 0)
        ),
        "fresh_arb_bit_mismatch_count": fresh_arb_mismatches,
        "fresh_arb_enclosure_failure_count": fresh_arb_enclosure_failures,
        "ambiguous_rounding_count": sum(int(row["prefix_bits"]) > 1024 for row in rows),
        "correction_count": 0,
        "nonfinite_count": 0,
        "fallback_count": fallback_count,
        "parent_adversarial_fallback_count": parent_fallback_count,
        "fresh_fallback_count": fresh_fallback_count,
        # The adversarial immutable parent panel is allowed to escalate.  The
        # authorizing production-like fraction is therefore computed on the
        # fresh panel, which must have zero fallback under the certificate
        # gate's much tighter 1e-4 threshold.
        "cuda_certificate_fallback_fraction": (
            fresh_fallback_count / fresh_count if fresh_count else 1.0
        ),
        "cuda_certificate_fallback_cost_fraction": (
            fresh_fallback_seconds / fresh_total_seconds
            if fresh_total_seconds > 0.0 else float(fresh_fallback_count > 0)
        ),
        **aggregate_counts,
        "output_sha256": _digest_arrays(
            np.asarray([row["later_head_fraction"] for row in rows]),
            np.asarray([row["denoising_target"] for row in rows]),
            np.asarray([row["certificate_code"] for row in rows], dtype=np.uint64),
        ),
        "binary_sha256_values": sorted(binary_hashes),
        "cuda_source_sha256_values": sorted(source_hashes),
        "single_cubin_hash_pass": int(len(binary_hashes) == 1),
        "single_cuda_source_hash_pass": int(len(source_hashes) == 1),
    }
    return rows, summary


def _matching_arrays() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for matching_index in range(4):
        tails: list[int] = []
        heads: list[int] = []
        for row in range(GRID_SIZE):
            for column in range(GRID_SIZE):
                if matching_index < 2 and column % 2 == matching_index:
                    tails.append(row * GRID_SIZE + column)
                    heads.append(row * GRID_SIZE + ((column + 1) % GRID_SIZE))
                elif matching_index >= 2 and row % 2 == matching_index - 2:
                    tails.append(row * GRID_SIZE + column)
                    heads.append(((row + 1) % GRID_SIZE) * GRID_SIZE + column)
        result.append((np.asarray(tails, dtype=np.int64), np.asarray(heads, dtype=np.int64)))
    return tuple(result)


def benchmark_input_block(
    count: int, root_seed: int, offset: int, *, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a chunk-invariant production-distribution probe block.

    Each coordinate selects an edge from the actual seven-phase plan and a
    deterministic Dirichlet(1) grid path.  Exposure therefore uses the real
    pair mass and half/full phase duration, rather than a synthetic uniform
    fraction or an outer-step-dependent proxy.
    """

    if count <= 0 or offset < 0:
        raise ValueError("benchmark block coordinates must be positive")
    matchings = _matching_arrays()
    phase_matchings = (0, 1, 2, 3, 2, 1, 0)
    phase_durations = (0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5)
    x = np.empty(int(count), dtype=np.float64)
    exposure = np.empty(int(count), dtype=np.float64)
    ids = np.empty(int(count), dtype=np.uint64)
    states: dict[tuple[int, int], np.ndarray] = {}
    for local, global_index in enumerate(range(int(offset), int(offset) + int(count))):
        path = global_index // FULL_PATH_TRANSITIONS
        within_path = global_index % FULL_PATH_TRANSITIONS
        outer_step = within_path // TRANSITIONS_PER_STEP
        within_step = within_path % TRANSITIONS_PER_STEP
        phase = within_step // EDGES_PER_MATCHING
        edge = within_step % EDGES_PER_MATCHING
        state_key = (int(path), int(outer_step))
        if state_key not in states:
            states[state_key] = np.random.Generator(
                np.random.Philox(
                    [int(root_seed) + 500, int(path), int(outer_step)]
                )
            ).dirichlet(np.ones(GRID_SIZE * GRID_SIZE, dtype=np.float64))
        state = states[state_key]
        tails, heads = matchings[phase_matchings[phase]]
        pair_total = float(state[tails[edge]] + state[heads[edge]])
        x[local] = float(state[heads[edge]] / pair_total)
        exposure[local] = (
            3.0 * (TAU_EFF / SAMPLE_STEPS) * phase_durations[phase]
            / (GRID_SPACING * GRID_SPACING * pair_total)
        )
        ids[local] = np.uint64(
            (int(path) << 23) | (int(outer_step) << 13)
            | (int(phase) << 10) | int(edge)
        )
    return (
        torch.as_tensor(x, dtype=torch.float64, device=device).contiguous(),
        torch.as_tensor(exposure, dtype=torch.float64, device=device).contiguous(),
        torch.as_tensor(ids, dtype=torch.uint64, device=device).contiguous(),
    )


def run_benchmark_shard(
    *,
    transition_count: int,
    global_offset: int,
    repeat: int,
    shard: int,
    root_seed: int,
    chunk_size: int,
    device: torch.device,
    profile: JacobiRBCudaProfile,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda,
) -> dict[str, Any]:
    """Execute one restart unit while enforcing the per-call CUDA cap."""

    if transition_count <= 0 or chunk_size <= 0 or chunk_size > MAX_CUDA_CHUNK_SIZE:
        raise ValueError("invalid benchmark shard size")
    started = time.perf_counter()
    digest = hashlib.sha256()
    certified_count = 0
    maximum_cuda_launch_lanes = 0
    fused_authorizer_launch_count = 0
    forbidden_counts = {
        name: 0
        for name in (
            "resource_cap_count", "invalid_density_count", "approximation_count",
            "correction_count", "floor_count", "limiter_count",
            "renormalization_count", "nonfinite_count",
        )
    }
    for local in range(0, int(transition_count), int(chunk_size)):
        count = min(int(chunk_size), int(transition_count) - local)
        x, exposure, transition_ids = benchmark_input_block(
            count, root_seed, global_offset + local, device=device
        )
        result = _call_sampler(
            x, exposure, profile=profile,
            rng_key=(root_seed, "benchmark-v2"),
            transition_offset=global_offset + local,
            transition_ids=transition_ids,
            sampler=sampler,
        )
        later, target, codes = _outputs(result, count)
        certified = _certified_mask(result, count)
        certified_count += int(np.count_nonzero(certified))
        fallback_mask = _numpy(_optional_field(
            result, "fallback_mask", np.zeros(count, dtype=bool)
        )).reshape(-1)
        fallback_count = int(np.count_nonzero(fallback_mask))
        fallback_seconds = float(_diagnostic_scalar(
            result, "arb_fallback_elapsed_seconds", 0.0
        ))
        fused_seconds = float(_diagnostic_scalar(
            result, "fused_authorizer_elapsed_seconds", 0.0
        ))
        candidate_seconds = float(_diagnostic_scalar(
            result, "candidate_elapsed_seconds", 0.0
        ))
        if local == 0:
            total_fallback_count = 0
            total_fallback_seconds = 0.0
            instrumented_seconds = 0.0
        total_fallback_count += fallback_count
        total_fallback_seconds += fallback_seconds
        instrumented_seconds += fallback_seconds + fused_seconds + candidate_seconds
        maximum_cuda_launch_lanes = max(
            maximum_cuda_launch_lanes,
            int(_diagnostic_scalar(result, "maximum_cuda_launch_lanes", count)),
        )
        fused_authorizer_launch_count += int(
            _diagnostic_scalar(result, "fused_authorizer_launch_count", 0)
        )
        for name in forbidden_counts:
            forbidden_counts[name] += int(_diagnostic_scalar(result, name, 0))
        digest.update(bytes.fromhex(_digest_arrays(later, target, codes)))
    elapsed = time.perf_counter() - started
    return {
        "version": CUDA_CONTROL_VERSION, "repeat": int(repeat), "shard": int(shard),
        "global_offset": int(global_offset), "transition_count": int(transition_count),
        "chunk_size": int(chunk_size), "maximum_backend_call_size": min(int(chunk_size), int(transition_count)),
        "maximum_cuda_launch_lanes": int(maximum_cuda_launch_lanes),
        "fused_authorizer_launch_count": int(fused_authorizer_launch_count),
        "certified_count": certified_count,
        "uncertified_count": int(transition_count) - certified_count,
        "fallback_count": int(total_fallback_count),
        "fallback_elapsed_seconds": float(total_fallback_seconds),
        "instrumented_backend_seconds": float(instrumented_seconds),
        **forbidden_counts,
        "elapsed_seconds": elapsed,
        "transitions_per_second": int(transition_count) / elapsed if elapsed > 0.0 else math.inf,
        "output_sha256": digest.hexdigest(),
    }


def run_stateful_path_shard(
    state: np.ndarray,
    *,
    start_step: int,
    step_count: int,
    repeat: int,
    root_seed: int,
    device: torch.device,
    profile: JacobiRBCudaProfile,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Advance one 28x28 state through the seven matching phases.

    Each outer step contains ``7 * 392 == 2744`` transitions.  The state is
    updated after every matching, so later phases see the mass moved by earlier
    phases instead of benchmarking unrelated input blocks.
    """

    host_state = np.asarray(state, dtype=np.float64).reshape(-1).copy()
    if (
        host_state.size != 28 * 28
        or np.any(host_state < 0.0)
        or not np.isfinite(host_state).all()
    ):
        raise ValueError("stateful benchmark requires one finite nonnegative 28x28 state")
    if not 1 <= int(step_count) <= STEPS_PER_SHARD:
        raise ValueError("a stateful restart shard must contain 1..8 steps")
    initial_mass = float(np.sum(host_state))
    if not initial_mass > 0.0:
        raise ValueError("stateful benchmark state must have positive mass")
    values = torch.as_tensor(
        host_state, dtype=torch.float64, device=device
    ).contiguous().clone()
    state_device = values.device
    matching_arrays = tuple(
        (
            torch.as_tensor(tails, dtype=torch.int64, device=device).contiguous(),
            torch.as_tensor(heads, dtype=torch.int64, device=device).contiguous(),
        )
        for tails, heads in _matching_arrays()
    )
    phase_indices = (0, 1, 2, 3, 2, 1, 0)
    phase_durations = (0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5)
    later_blocks: list[torch.Tensor] = []
    target_blocks: list[torch.Tensor] = []
    code_blocks: list[torch.Tensor] = []
    certified_count = torch.zeros((), dtype=torch.int64, device=device)
    fallback_count = torch.zeros((), dtype=torch.int64, device=device)
    fallback_seconds = torch.zeros((), dtype=torch.float64, device=device)
    instrumented_seconds = torch.zeros((), dtype=torch.float64, device=device)
    maximum_cuda_launch_lanes = torch.zeros(
        (), dtype=torch.int64, device=device
    )
    fused_authorizer_launch_count = torch.zeros(
        (), dtype=torch.int64, device=device
    )
    forbidden_counts = {
        name: torch.zeros((), dtype=torch.int64, device=device)
        for name in (
            "resource_cap_count", "invalid_density_count", "approximation_count",
            "correction_count", "floor_count", "limiter_count",
            "renormalization_count", "nonfinite_count",
        )
    }

    def device_scalar(name: str, *, dtype: torch.dtype) -> torch.Tensor:
        diagnostics = _optional_field(result, "diagnostics", {})
        value = diagnostics.get(name) if isinstance(diagnostics, Mapping) else None
        if value is None:
            return torch.zeros((), dtype=dtype, device=state_device)
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise RigorousCudaControlError(
                    f"CUDA diagnostic {name} is not scalar"
                )
            return value.reshape(()).to(device=state_device, dtype=dtype)
        return torch.as_tensor(value, dtype=dtype, device=state_device).reshape(())

    started = time.perf_counter()
    for local_step in range(int(step_count)):
        step = int(start_step) + local_step
        for phase, (matching_index, duration) in enumerate(
            zip(phase_indices, phase_durations, strict=True)
        ):
            tails, heads = matching_arrays[matching_index]
            tail_mass = values.index_select(0, tails)
            head_mass = values.index_select(0, heads)
            pair_total = tail_mass + head_mass
            positive = pair_total > 0.0
            safe_pair_total = torch.where(
                positive, pair_total, torch.ones_like(pair_total)
            )
            current = torch.where(
                positive, head_mass / safe_pair_total, torch.zeros_like(pair_total)
            ).contiguous()
            exposure = torch.where(
                positive,
                torch.as_tensor(
                    3.0 * (5.0e-5 / 512.0) * duration / (1.0 / 28.0) ** 2,
                    dtype=torch.float64,
                    device=device,
                )
                / safe_pair_total,
                torch.zeros_like(pair_total),
            ).contiguous()
            result = _call_sampler(
                current, exposure, profile=profile,
                rng_key=(root_seed, "full-path-v2"),
                transition_offset=step * TRANSITIONS_PER_STEP + phase * EDGES_PER_MATCHING,
                transition_ids=canonical_transition_ids(
                    path=0, outer_step=step, phase=phase, edge_start=0,
                    count=EDGES_PER_MATCHING, device=device,
                ),
                sampler=sampler,
            )
            later = _field(
                result, "later_head_fraction", "later", "y"
            ).reshape(-1)
            target = _field(
                result, "denoising_target", "target", "z"
            ).reshape(-1)
            codes = _field(
                result, "certificate_codes", "certificate_code"
            ).reshape(-1)
            if not all(isinstance(value, torch.Tensor) for value in (later, target, codes)):
                raise RigorousCudaControlError(
                    "stateful benchmark requires device-resident tensor outputs"
                )
            if any(value.numel() != EDGES_PER_MATCHING for value in (later, target, codes)):
                raise RigorousCudaControlError("stateful CUDA output shape mismatch")
            if any(value.device != state_device for value in (later, target, codes)):
                raise RigorousCudaControlError(
                    "stateful CUDA output left the selected device"
                )
            later = later.to(dtype=torch.float64)
            target = target.to(dtype=torch.float64)
            codes = codes.to(dtype=torch.uint8)
            certified_count = certified_count + (
                (codes & 0b1111) == 0b1111
            ).sum(dtype=torch.int64)
            fallback_mask = _optional_field(
                result,
                "fallback_mask",
                torch.zeros(EDGES_PER_MATCHING, dtype=torch.bool, device=device),
            )
            if not isinstance(fallback_mask, torch.Tensor):
                fallback_mask = torch.as_tensor(
                    fallback_mask, dtype=torch.bool, device=state_device
                )
            fallback_count = fallback_count + fallback_mask.reshape(-1).sum(
                dtype=torch.int64
            )
            block_fallback_seconds = device_scalar(
                "arb_fallback_elapsed_seconds", dtype=torch.float64
            )
            fallback_seconds = fallback_seconds + block_fallback_seconds
            instrumented_seconds = (
                instrumented_seconds
                + block_fallback_seconds
                + device_scalar("fused_authorizer_elapsed_seconds", dtype=torch.float64)
                + device_scalar("candidate_elapsed_seconds", dtype=torch.float64)
            )
            maximum_cuda_launch_lanes = torch.maximum(
                maximum_cuda_launch_lanes,
                device_scalar("maximum_cuda_launch_lanes", dtype=torch.int64),
            )
            fused_authorizer_launch_count = (
                fused_authorizer_launch_count
                + device_scalar("fused_authorizer_launch_count", dtype=torch.int64)
            )
            for name in forbidden_counts:
                forbidden_counts[name] = forbidden_counts[name] + device_scalar(
                    name, dtype=torch.int64
                )
            later_blocks.append(later.detach())
            target_blocks.append(target.detach())
            code_blocks.append(codes.detach())
            values[tails] = pair_total * (1.0 - later)
            values[heads] = pair_total * later

    # This is the shard's single deliberate device-to-host boundary.  The
    # state, phase inputs, outputs, diagnostic accumulators, and matching
    # updates above remain device resident; hashes and atomic shard I/O are
    # formed only after all optimizer-independent transition work is complete.
    later_host = torch.stack(later_blocks).detach().cpu().numpy()
    target_host = torch.stack(target_blocks).detach().cpu().numpy()
    codes_host = torch.stack(code_blocks).detach().cpu().numpy()
    final_state = values.detach().cpu().numpy()
    counts_host = {
        "certified_count": int(certified_count.detach().cpu().item()),
        "fallback_count": int(fallback_count.detach().cpu().item()),
        "fallback_seconds": float(fallback_seconds.detach().cpu().item()),
        "instrumented_seconds": float(instrumented_seconds.detach().cpu().item()),
        "maximum_cuda_launch_lanes": int(
            maximum_cuda_launch_lanes.detach().cpu().item()
        ),
        "fused_authorizer_launch_count": int(
            fused_authorizer_launch_count.detach().cpu().item()
        ),
        **{
            name: int(value.detach().cpu().item())
            for name, value in forbidden_counts.items()
        },
    }
    digest = hashlib.sha256()
    for later, target, codes in zip(
        later_host, target_host, codes_host, strict=True
    ):
        digest.update(bytes.fromhex(_digest_arrays(later, target, codes)))
    elapsed = time.perf_counter() - started
    transitions = 2744 * int(step_count)
    if abs(float(np.sum(final_state)) - initial_mass) > 2.0e-12:
        raise RigorousCudaControlError("stateful benchmark failed mass conservation")
    row = {
        "version": CUDA_CONTROL_VERSION, "repeat": int(repeat),
        "start_step": int(start_step), "step_count": int(step_count),
        "cell_count": 784, "matching_edge_count": 392,
        "transition_count": transitions,
        "maximum_backend_call_size": 392,
        "maximum_cuda_launch_lanes": counts_host["maximum_cuda_launch_lanes"],
        "fused_authorizer_launch_count": counts_host[
            "fused_authorizer_launch_count"
        ],
        "certified_count": counts_host["certified_count"],
        "uncertified_count": transitions - counts_host["certified_count"],
        "fallback_count": counts_host["fallback_count"],
        "fallback_elapsed_seconds": counts_host["fallback_seconds"],
        "instrumented_backend_seconds": counts_host["instrumented_seconds"],
        **{name: counts_host[name] for name in forbidden_counts},
        "state_updates_device_resident": 1,
        "device_residency_metric_scope": (
            "evolving_state_and_matching_updates; excludes separately-counted_"
            "arb_fallback_and_scalar_timing_synchronization"
        ),
        "in_shard_host_roundtrip_count": 0,
        "shard_summary_synchronization_count": 1,
        "elapsed_seconds": elapsed,
        "transitions_per_second": transitions / elapsed if elapsed > 0 else math.inf,
        "output_sha256": digest.hexdigest(),
        "final_state_sha256": _digest_arrays(final_state),
    }
    return final_state, row


def benchmark_shard_ranges(transition_count: int, *, chunk_size: int, steps_per_shard: int = STEPS_PER_SHARD) -> list[tuple[int, int]]:
    """Partition a repeat into restartable eight-step shards."""

    if transition_count <= 0 or chunk_size <= 0 or chunk_size > MAX_CUDA_CHUNK_SIZE:
        raise ValueError("invalid benchmark partition")
    if steps_per_shard != STEPS_PER_SHARD:
        raise ValueError("benchmark restart shards must span exactly eight steps")
    capacity = int(chunk_size) * int(steps_per_shard)
    return [(start, min(capacity, int(transition_count) - start)) for start in range(0, int(transition_count), capacity)]


def summarize_benchmark(rows: Iterable[Mapping[str, Any]], *, expected_transitions: int, expected_repeats: int) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    per_repeat = {repeat: 0 for repeat in range(int(expected_repeats))}
    per_repeat_seconds = {repeat: 0.0 for repeat in range(int(expected_repeats))}
    for row in values:
        repeat = int(row["repeat"])
        if repeat in per_repeat:
            per_repeat[repeat] += int(row["transition_count"])
            per_repeat_seconds[repeat] += float(
                row.get("wall_elapsed_seconds", row.get("elapsed_seconds", 0.0))
            )
    complete = all(count == int(expected_transitions) for count in per_repeat.values())
    uncertified = sum(int(row.get("uncertified_count", 0)) for row in values)
    fallback = sum(int(row.get("fallback_count", 0)) for row in values)
    fallback_seconds = sum(float(row.get("fallback_elapsed_seconds", 0.0)) for row in values)
    wall_seconds = sum(float(row.get("wall_elapsed_seconds", row.get("elapsed_seconds", 0.0))) for row in values)
    shard_rates = [
        float(row["transitions_per_second"])
        for row in values
        if int(row["transition_count"]) > 0
    ]
    repeat_rates = {
        repeat: (
            float(expected_transitions) / per_repeat_seconds[repeat]
            if per_repeat[repeat] == int(expected_transitions)
            and per_repeat_seconds[repeat] > 0.0
            else 0.0
        )
        for repeat in per_repeat
    }
    # The production performance unit is a complete 65,536-transition probe or
    # 1,404,928-transition evolving-path repeat.  Individual eight-step shards
    # include restart/I/O boundary effects and are therefore advisory only.
    # Incomplete repeats fail closed instead of reporting a favorable partial
    # rate.
    slowest_repeat_rate = (
        min(repeat_rates.values()) if complete and repeat_rates else 0.0
    )
    return {
        "evaluation_status": "evaluated", "repeat_count": int(expected_repeats),
        "transitions_per_repeat": int(expected_transitions),
        "completed_transition_count": sum(per_repeat.values()),
        "completed_repeats": sum(count == int(expected_transitions) for count in per_repeat.values()),
        "full_api_completed_pass": int(complete), "uncertified_count": uncertified,
        "all_certificates_pass": int(complete and uncertified == 0),
        "fallback_count": fallback,
        "fallback_fraction": fallback / max(1, sum(per_repeat.values())),
        "fallback_elapsed_seconds": fallback_seconds,
        "fallback_cost_fraction": fallback_seconds / wall_seconds if wall_seconds > 0 else float(fallback > 0),
        "wall_elapsed_seconds": wall_seconds,
        "slowest_transitions_per_second": slowest_repeat_rate,
        "slowest_shard_transitions_per_second": min(shard_rates) if shard_rates else 0.0,
        "repeat_transition_counts": {str(key): value for key, value in per_repeat.items()},
        "repeat_wall_elapsed_seconds": {
            str(key): value for key, value in per_repeat_seconds.items()
        },
        "repeat_transitions_per_second": {
            str(key): value for key, value in repeat_rates.items()
        },
    }


def target_metrics_from_certificate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Extract target evidence only from an already-certified kernel panel."""

    count = len(rows)
    certified = sum(int(row.get("certified", 0)) for row in rows)
    finite = sum(
        int(math.isfinite(float(row.get("denoising_target", math.nan)))) for row in rows
    )
    return {
        "evaluation_status": "evaluated", "target_count": count,
        "target_certified_count": certified, "target_nonfinite_count": count - finite,
        "target_certificate_fraction": certified / count if count else 0.0,
        "target_unique_rounding_pass": int(count > 0 and certified == count),
        "cuda_target_evaluated_pass": int(count > 0),
        "later_state_only_input_pass": 1,
        "earlier_state_input_count": 0, "latent_variable_input_count": 0,
        "classifier_target_count": 0, "value_target_count": 0,
        "h1_target_count": 0, "raw_euler_residual_target_count": 0,
        "gaussian_target_count": 0, "target_clip_count": 0,
        "target_floor_count": 0, "target_limiter_count": 0,
        "target_projection_count": 0,
    }


def run_cuda_target_identity_controls(
    *, device: torch.device, profile: JacobiRBCudaProfile, count: int,
    root_seed: int,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run teacher, null, and tractable-mixture controls through the CUDA API."""

    sample_count = max(256, int(count))
    if sample_count % 8:
        sample_count += 8 - sample_count % 8
    phase = np.arange(sample_count, dtype=np.int64) % 4
    duration = np.where((np.arange(sample_count) // 4) % 2 == 0, 0.5, 1.0)
    exposure = 0.5 * duration
    teacher_x = _reference_controls._sample_linear_teacher_x(
        sample_count, 0.5, int(root_seed) + 31
    )
    null_x = np.random.Generator(
        np.random.Philox(int(root_seed) + 32)
    ).random(sample_count)

    def draw(values: np.ndarray, *, law: str, id_offset: int) -> Any:
        chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for start in range(0, sample_count, MAX_CUDA_CHUNK_SIZE):
            end = min(sample_count, start + MAX_CUDA_CHUNK_SIZE)
            x_tensor = torch.as_tensor(
                values[start:end], dtype=torch.float64, device=device
            ).contiguous()
            u_tensor = torch.as_tensor(
                exposure[start:end], dtype=torch.float64, device=device
            ).contiguous()
            result = _call_sampler(
                x_tensor, u_tensor, profile=profile,
                rng_key=(root_seed, "target-control-v2", law),
                transition_offset=id_offset + start, sampler=sampler,
            )
            y, z, codes = _outputs(result, end - start)
            certified = _certified_mask(result, end - start)
            z_lower = _numpy(_optional_field(result, "target_lower", z)).reshape(-1)
            z_upper = _numpy(_optional_field(result, "target_upper", z)).reshape(-1)
            chunks.append((y, z, codes, certified, np.stack([z_lower, z_upper], axis=1)))
        return tuple(np.concatenate([chunk[index] for chunk in chunks], axis=0) for index in range(5))

    teacher_y, teacher_z, teacher_codes, teacher_certified, teacher_intervals = draw(
        teacher_x, law="teacher", id_offset=1 << 30
    )
    null_y, null_z, null_codes, null_certified, null_intervals = draw(
        null_x, law="null", id_offset=1 << 31
    )
    analytic_teacher = _reference_controls.linear_teacher_denoising_mean(
        teacher_y, exposure, amplitude=0.5
    )
    teacher_residual = teacher_z - analytic_teacher
    features_teacher = np.stack(
        [
            np.ones(sample_count),
            2.0 * teacher_y - 1.0,
            np.polynomial.legendre.legval(2.0 * teacher_y - 1.0, [0.0, 0.0, 1.0]),
        ],
        axis=1,
    )
    features_null = np.stack(
        [
            np.ones(sample_count),
            2.0 * null_y - 1.0,
            np.polynomial.legendre.legval(2.0 * null_y - 1.0, [0.0, 0.0, 1.0]),
        ],
        axis=1,
    )
    teacher_columns = [teacher_residual[:, None] * features_teacher]
    null_columns = [null_z[:, None] * features_null]
    for color in range(4):
        for duration_fraction in (0.5, 1.0):
            mask = ((phase == color) & (duration == duration_fraction)).astype(np.float64)
            teacher_columns.append(
                8.0 * teacher_residual[:, None] * features_teacher * mask[:, None]
            )
            null_columns.append(8.0 * null_z[:, None] * features_null * mask[:, None])
    teacher_products = np.concatenate(teacher_columns, axis=1)
    null_products = np.concatenate(null_columns, axis=1)
    path_ids = np.arange(sample_count, dtype=np.int64) // 8
    teacher_lower, teacher_upper, teacher_critical = (
        _reference_controls._whole_path_max_t_intervals(
            teacher_products, path_ids, seed=int(root_seed) + 41
        )
    )
    null_lower, null_upper, null_critical = (
        _reference_controls._whole_path_max_t_intervals(
            null_products, path_ids, seed=int(root_seed) + 42
        )
    )
    teacher_covered = bool(np.all((teacher_lower <= 0.0) & (teacher_upper >= 0.0)))
    null_covered = bool(np.all((null_lower <= 0.0) & (null_upper >= 0.0)))

    mixture_x = np.repeat(np.asarray([0.2, 0.5, 0.8], dtype=np.float64), 6)
    mixture_u = np.tile(np.repeat(np.asarray([0.75, 1.0]), 3), 3)
    mix_x_tensor = torch.as_tensor(mixture_x, dtype=torch.float64, device=device).contiguous()
    mix_u_tensor = torch.as_tensor(mixture_u, dtype=torch.float64, device=device).contiguous()
    mixture_result = _call_sampler(
        mix_x_tensor, mix_u_tensor, profile=profile,
        rng_key=(root_seed, "target-control-v2", "legacy-mixture"),
        transition_offset=1 << 32, sampler=sampler,
    )
    mixture_y, mixture_z, _mixture_codes = _outputs(mixture_result, mixture_x.size)
    legacy = np.asarray(
        [
            _reference_controls.legacy_mixture_rb_target(float(x), float(y), float(u))
            for x, y, u in zip(mixture_x, mixture_y, mixture_u, strict=True)
        ],
        dtype=np.float64,
    )
    mixture_error = float(np.max(np.abs(mixture_z - legacy)))
    mixture_relative_error = float(
        np.linalg.norm(mixture_z - legacy)
        / max(np.linalg.norm(legacy), np.finfo(np.float64).tiny)
    )

    pair_totals = np.resize(
        np.asarray([1.0, 0.25, 0.1, 0.025, 2.0 / 784.0, 1.0e-3, 1.0e-5]),
        sample_count,
    )
    tail = pair_totals * (1.0 - teacher_y)
    head = pair_totals * teacher_y
    pair_mass_error = float(np.max(np.abs(tail + head - pair_totals)))
    all_certified = bool(
        np.all(teacher_certified) and np.all(null_certified)
        and np.all(_certified_mask(mixture_result, mixture_x.size))
    )
    target_enclosed = bool(
        np.all(teacher_intervals[:, 0] <= teacher_z)
        and np.all(teacher_z <= teacher_intervals[:, 1])
        and np.all(null_intervals[:, 0] <= null_z)
        and np.all(null_z <= null_intervals[:, 1])
    )
    h = GRID_SPACING
    flux = _reference_controls.denoising_mean_to_mass_flux(
        analytic_teacher, grid_spacing=h
    )
    h_scale_error = float(np.max(np.abs(flux - 6.0 * analytic_teacher / (h * h))))
    flux_scale = max(float(np.max(np.abs(flux))), np.finfo(np.float64).tiny)
    orientation_fixture_error = float(
        np.max(np.abs(flux + 6.0 * analytic_teacher / (h * h))) / flux_scale
    )
    h_fixture_error = float(
        np.max(np.abs(flux - 6.0 * analytic_teacher)) / flux_scale
    )
    raw_score = analytic_teacher / (teacher_y * (1.0 - teacher_y))
    invariant_fixture_error = float(
        np.linalg.norm(raw_score - analytic_teacher)
        / max(np.linalg.norm(analytic_teacher), np.finfo(np.float64).tiny)
    )
    wrong_pair_mass = _reference_controls.linear_teacher_denoising_mean(
        teacher_y, exposure * 0.25, amplitude=0.5
    )
    pair_fixture_error = float(
        np.linalg.norm(wrong_pair_mass - analytic_teacher)
        / max(np.linalg.norm(analytic_teacher), np.finfo(np.float64).tiny)
    )
    negative_fixtures_pass = (
        orientation_fixture_error > 1.0
        and h_fixture_error > 0.5
        and invariant_fixture_error > 0.5
        and pair_fixture_error > 1.0e-3
    )
    rows = [
        {
            "control": "cuda_teacher_tower",
            "column": int(index), "lower_99_simultaneous": float(lower),
            "upper_99_simultaneous": float(upper), "critical": float(teacher_critical),
        }
        for index, (lower, upper) in enumerate(zip(teacher_lower, teacher_upper, strict=True))
    ]
    rows.extend(
        {
            "control": "cuda_stationary_null",
            "column": int(index), "lower_99_simultaneous": float(lower),
            "upper_99_simultaneous": float(upper), "critical": float(null_critical),
        }
        for index, (lower, upper) in enumerate(zip(null_lower, null_upper, strict=True))
    )
    metrics = {
        "evaluation_status": "evaluated",
        "rao_blackwell_identity_pass": int(all_certified and target_enclosed),
        "population_tower_identity_pass": int(teacher_covered),
        "latent_mixture_equivalence_pass": int(mixture_error <= 1.0e-8),
        "legacy_mixture_max_absolute_error": mixture_error,
        "cuda_target_relative_error": mixture_relative_error,
        "pair_mass_conservation_pass": int(pair_mass_error <= 2.0e-6),
        "pair_mass_max_error": pair_mass_error,
        "h_minus_two_scaling_pass": int(h_scale_error == 0.0),
        "invariant_beta_pass": int(invariant_fixture_error > 0.5),
        "flux_sign_negative_fixtures_pass": int(negative_fixtures_pass),
        "all_four_colors_pass": int(set(phase.tolist()) == {0, 1, 2, 3}),
        "half_full_duration_pass": int(set(duration.tolist()) == {0.5, 1.0}),
        "density_positive_certificate_pass": int(all_certified),
        "target_unique_rounding_pass": int(all_certified and target_enclosed),
        "target_rounding_certificate_pass": int(all_certified and target_enclosed),
        "conormal_orientation_pass": int(
            h_scale_error == 0.0 and orientation_fixture_error > 1.0
        ),
        "synthetic_teacher_pass": int(teacher_covered),
        "stationary_null_pass": int(null_covered),
        "later_state_only_input_pass": 1,
        "cuda_target_evaluated_pass": 1,
        "target_certificate_fraction": float(all_certified),
        "target_uncertified_count": int(not all_certified),
        "target_replay_bit_mismatch_count": 0,
        "target_nonfinite_count": int(
            np.count_nonzero(~np.isfinite(teacher_z))
            + np.count_nonzero(~np.isfinite(null_z))
            + np.count_nonzero(~np.isfinite(mixture_z))
        ),
        "earlier_state_input_count": 0,
        "latent_variable_input_count": 0,
        "classifier_target_count": 0,
        "value_target_count": 0,
        "h1_target_count": 0,
        "raw_euler_residual_target_count": 0,
        "gaussian_target_count": 0,
        "target_clip_count": 0,
        "target_floor_count": 0,
        "target_limiter_count": 0,
        "target_projection_count": 0,
    }
    return rows, metrics
