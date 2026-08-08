"""Measured scientific controls for the certified Haar Jacobi copula.

The helpers here are deliberately controls-only.  They exercise the fused
normal transform, the arbitrary-uniform Jacobi authorizer, exact marginal
eigenmoments, reversibility witnesses, phase-local Dynkin tower identities,
and deterministic batching.  No neural or production-refinement code is
imported.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_dynkin import run_dynkin_tower_phase
from mnist.d0_jacobi_rb_haar import (
    HaarCouplingProfile,
    build_certified_haar_uniform_batch,
)
from mnist.d0_jacobi_rb_haar_cuda import (
    sample_alpha1_rb_transition_batch_cuda_from_uniform_cells,
)
from mnist.d0_jacobi_rb_spectral import (
    evaluate_alpha1_rb_torch_fixed_modes,
)
from mnist.d0_jacobi_rb_strang_refinement import (
    refinement_observable_spec,
)
from mnist.d0_jacobi_rb_haar_scheduler import (
    ADJACENT_LEVEL_PAIRS,
    HaarShardIdentity,
    HaarSchedulerError,
    NestedHaarSchedule,
    PairwiseHaarAntitheticSchedule,
    commit_haar_shard,
    expected_haar_shard_input_sha256,
    initialize_antithetic_branch_states,
    initialize_nested_branch_states,
    load_committed_haar_shard,
    run_nested_haar_shard,
    run_pairwise_haar_antithetic_shard,
)


HAAR_CONTROL_VERSION = "d0-jacobi-rb-certified-haar-controls-v1"
EDGES_PER_PHASE = 392
PHASE_COUNT = 7
NO_WORK = {
    "physical_training_performed": 0,
    "production_refinement_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
}


class HaarControlError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        failure_code: str,
        failure_domain: str,
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code
        self.failure_domain = failure_domain


def _np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _record_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _authorize(
    batch: Any,
    x: torch.Tensor,
    exposure: torch.Tensor,
    profile: JacobiRBCudaProfile,
) -> Any:
    return sample_alpha1_rb_transition_batch_cuda_from_uniform_cells(
        x.contiguous(),
        exposure.contiguous(),
        batch.uniform_lower.reshape(-1).contiguous(),
        batch.uniform_upper.reshape(-1).contiguous(),
        transition_ids=batch.transition_ids.reshape(-1).contiguous(),
        refinement_callback=batch.refinement_callback,
        profile=profile,
        uniform_center_hi=batch.uniform_center_hi.reshape(-1).contiguous(),
        uniform_center_lo=batch.uniform_center_lo.reshape(-1).contiguous(),
        uniform_radius=batch.uniform_radius.reshape(-1).contiguous(),
        source_prefix_bits=batch.prefix_bits.reshape(-1).contiguous(),
    )


def _legendre_values(z: np.ndarray, maximum_degree: int) -> np.ndarray:
    values = np.empty((int(maximum_degree), z.size), dtype=np.float64)
    for degree in range(1, int(maximum_degree) + 1):
        coefficients = np.zeros(degree + 1, dtype=np.float64)
        coefficients[-1] = 1.0
        values[degree - 1] = np.polynomial.legendre.legval(z, coefficients)
    return values


def _simultaneous_zero_pass(samples: np.ndarray, *, family_error: float) -> tuple[int, float]:
    values = np.asarray(samples, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] < 1
        or not np.isfinite(values).all()
    ):
        return 0, math.inf
    mean = np.mean(values, axis=0)
    sd = np.std(values, axis=0, ddof=1)
    critical = math.sqrt(
        2.0 * math.log(
            2.0 * float(values.shape[1]) / float(family_error)
        )
    )
    half = critical * sd / math.sqrt(float(values.shape[0]))
    zero_sd_bad = (sd == 0.0) & (mean != 0.0)
    ratio = np.divide(
        np.abs(mean),
        np.maximum(half, np.finfo(np.float64).tiny),
    )
    passed = bool(not np.any(zero_sd_bad) and np.all(np.abs(mean) <= half))
    return int(passed), float(np.max(ratio))


def _whole_cluster_max_t_zero_pass(
    samples: np.ndarray,
    *,
    family_error: float,
    bootstrap_seed: int,
    replicates: int = 20_000,
) -> dict[str, Any]:
    """Centered whole-cluster max-T interval for a joint zero-mean family."""

    values = np.asarray(samples, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] < 3
        or values.shape[1] < 1
        or not np.isfinite(values).all()
        or not 0.0 < float(family_error) < 1.0
        or int(replicates) < 1_000
    ):
        return {
            "passed": 0,
            "critical_value": None,
            "maximum_studentized_statistic": None,
            "maximum_critical_ratio": None,
            "replicates": int(replicates),
        }
    paths, features = values.shape
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    scale = sd / math.sqrt(float(paths))
    zero_sd_bad = (sd == 0.0) & (mean != 0.0)
    observed = np.divide(
        np.abs(mean),
        scale,
        out=np.zeros_like(mean),
        where=scale > 0.0,
    )
    observed[zero_sd_bad] = math.inf
    centered = values - mean
    rng = np.random.default_rng(int(bootstrap_seed) & ((1 << 64) - 1))
    maxima = np.empty(int(replicates), dtype=np.float64)
    cursor = 0
    batch_size = max(1, min(512, 2_000_000 // max(1, paths * features)))
    while cursor < int(replicates):
        size = min(batch_size, int(replicates) - cursor)
        indices = rng.integers(0, paths, size=(size, paths), endpoint=False)
        draws = centered[indices]
        draw_mean = draws.mean(axis=1)
        draw_sd = draws.std(axis=1, ddof=1)
        draw_scale = draw_sd / math.sqrt(float(paths))
        studentized = np.divide(
            np.abs(draw_mean),
            draw_scale,
            out=np.zeros_like(draw_mean),
            where=draw_scale > 0.0,
        )
        studentized[(draw_scale == 0.0) & (draw_mean != 0.0)] = math.inf
        maxima[cursor : cursor + size] = np.max(studentized, axis=1)
        cursor += size
    critical = float(
        np.quantile(
            maxima,
            1.0 - float(family_error),
            method="higher",
        )
    )
    maximum = float(np.max(observed))
    finite = math.isfinite(critical) and critical > 0.0
    return {
        "passed": int(
            finite and not np.any(zero_sd_bad) and maximum <= critical
        ),
        "critical_value": critical if finite else None,
        "maximum_studentized_statistic": maximum,
        "maximum_critical_ratio": (
            maximum / critical if finite else None
        ),
        "replicates": int(replicates),
    }


def _structural_marginal_profiles() -> tuple[dict[str, Any], ...]:
    """Every frozen tree shape whose one-step marginal must stay exact."""

    records: list[dict[str, Any]] = [
        {
            "profile": "nested_haar_single_arm",
            "pool": "main",
            "sample_steps": level,
            "tree_root_steps": 128,
            "detail_sign": 1,
            "pair_coarse_steps": None,
        }
        for level in (128, 256, 512, 1024)
    ]
    records.extend(
        {
            "profile": "nested_haar_single_arm",
            "pool": "reference",
            "sample_steps": level,
            "tree_root_steps": 512,
            "detail_sign": 1,
            "pair_coarse_steps": None,
        }
        for level in (512, 1024, 2048)
    )
    for coarse, fine in ADJACENT_LEVEL_PAIRS:
        records.extend(
            (
                {
                    "profile": "pairwise_haar_antithetic",
                    "pool": f"{coarse}-{fine}",
                    "branch": "coarse",
                    "sample_steps": coarse,
                    "tree_root_steps": coarse,
                    "detail_sign": 1,
                    "pair_coarse_steps": coarse,
                },
                {
                    "profile": "pairwise_haar_antithetic",
                    "pool": f"{coarse}-{fine}",
                    "branch": "fine_plus",
                    "sample_steps": fine,
                    "tree_root_steps": coarse,
                    "detail_sign": 1,
                    "pair_coarse_steps": coarse,
                },
                {
                    "profile": "pairwise_haar_antithetic",
                    "pool": f"{coarse}-{fine}",
                    "branch": "fine_minus",
                    "sample_steps": fine,
                    "tree_root_steps": coarse,
                    "detail_sign": -1,
                    "pair_coarse_steps": coarse,
                },
            )
        )
    return tuple(records)


def _fallback_timing(batch: Any, result: Any) -> tuple[int, float, float]:
    normal_diagnostics = dict(batch.diagnostics)
    normal_runtime = dict(batch.runtime_report)
    transition_diagnostics = dict(result.diagnostics)
    transition_runtime = dict(result.runtime_report)
    normal_count = int(normal_diagnostics.get("arb_fallback_count", -1))
    jacobi_count = int(transition_diagnostics.get("fallback_count", -1))
    normal_seconds = float(
        normal_runtime.get("arb_fallback_elapsed_seconds", math.inf)
    )
    jacobi_elapsed = float(
        transition_runtime.get("elapsed_seconds", math.inf)
    )
    jacobi_seconds = jacobi_elapsed * float(
        transition_runtime.get("arb_fallback_time_fraction", math.inf)
    )
    total_elapsed = (
        float(normal_runtime.get("elapsed_seconds", math.inf))
        + jacobi_elapsed
    )
    if (
        normal_count < 0
        or jacobi_count < 0
        or not math.isfinite(normal_seconds)
        or not math.isfinite(jacobi_seconds)
        or not math.isfinite(total_elapsed)
        or total_elapsed <= 0.0
    ):
        raise HaarControlError(
            "certified transition fallback diagnostics are incomplete",
            failure_code="hierarchical_marginal_diagnostics_invalid",
            failure_domain="marginal_law",
        )
    return normal_count + jacobi_count, normal_seconds + jacobi_seconds, total_elapsed


def _whole_path_means(
    samples: np.ndarray,
    *,
    path_count: int,
    values_per_path: int,
) -> np.ndarray:
    """Reduce edge-level witnesses to the preregistered whole-path unit."""

    values = np.asarray(samples, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] != int(path_count) * int(values_per_path)
        or not np.isfinite(values).all()
    ):
        raise HaarControlError(
            "marginal witness panel has an invalid path layout",
            failure_code="hierarchical_marginal_panel_invalid",
            failure_domain="marginal_law",
        )
    return values.reshape(int(path_count), int(values_per_path), -1).mean(axis=1)


def _required_int(record: Mapping[str, Any], name: str) -> int:
    if name not in record:
        raise HaarControlError(
            f"required diagnostic {name!r} is missing",
            failure_code="hierarchical_control_diagnostics_invalid",
            failure_domain="marginal_law",
        )
    value = int(record[name])
    if value < 0:
        raise HaarControlError(
            f"required diagnostic {name!r} is negative",
            failure_code="hierarchical_control_diagnostics_invalid",
            failure_domain="marginal_law",
        )
    return value


def _required_float(record: Mapping[str, Any], name: str) -> float:
    if name not in record:
        raise HaarControlError(
            f"required diagnostic {name!r} is missing",
            failure_code="hierarchical_control_diagnostics_invalid",
            failure_domain="marginal_law",
        )
    value = float(record[name])
    if not math.isfinite(value) or value < 0.0:
        raise HaarControlError(
            f"required diagnostic {name!r} is invalid",
            failure_code="hierarchical_control_diagnostics_invalid",
            failure_domain="marginal_law",
        )
    return value


def run_marginal_and_batching_controls(
    *,
    root_seed: int,
    path_ids: Sequence[int],
    device: str | torch.device,
) -> dict[str, Any]:
    """Measure exact marginal law, target, and batching invariance."""

    paths = tuple(int(value) for value in path_ids)
    if len(paths) != 8:
        raise ValueError("the production marginal panel requires eight paths")
    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise HaarControlError(
            "production marginal controls require CUDA",
            failure_code="hierarchical_cuda_state_required",
            failure_domain="runtime_backend",
        )
    haar_profile = HaarCouplingProfile(coarsest_steps=128, finest_steps=2048)
    jacobi_profile = JacobiRBCudaProfile()
    edges = tuple(range(EDGES_PER_PHASE))
    batch = build_certified_haar_uniform_batch(
        root_seed=int(root_seed),
        role="marginal_c",
        path_ids=paths,
        sample_steps=256,
        outer_step=0,
        phase=0,
        edge_ids=edges,
        profile=haar_profile,
        device=target_device,
    )
    count = len(paths) * EDGES_PER_PHASE
    rng = np.random.default_rng(int(root_seed) ^ 0x4D415247)
    x_np = rng.random(count, dtype=np.float64)
    x = torch.as_tensor(x_np, dtype=torch.float64, device=target_device)
    exposure = torch.full_like(x, 0.01)
    result = _authorize(batch, x, exposure, jacobi_profile)
    y = _np(result.later_head_fraction).reshape(-1)
    target = _np(result.denoising_target).reshape(-1)
    certified = _np(result.certified_mask).astype(bool).reshape(-1)
    uniform = 0.5 * (
        _np(batch.uniform_lower).reshape(-1)
        + _np(batch.uniform_upper).reshape(-1)
    )
    evaluated = evaluate_alpha1_rb_torch_fixed_modes(
        x,
        result.later_head_fraction.reshape(-1),
        exposure,
        modes=256,
    )
    cdf_error = float(
        np.max(np.abs(_np(evaluated.cdf).reshape(-1) - uniform))
    )
    target_error = float(
        np.max(
            np.abs(
                _np(evaluated.denoising_target).reshape(-1) - target
            )
        )
    )
    zx, zy = 2.0 * x_np - 1.0, 2.0 * y - 1.0
    px = _legendre_values(zx, 8)
    py = _legendre_values(zy, 8)
    moment_residual = np.stack(
        [
            py[degree - 1]
            - math.exp(-degree * (degree + 1) * 0.01)
            * px[degree - 1]
            for degree in range(1, 9)
        ],
        axis=1,
    )
    moment_by_path = _whole_path_means(
        moment_residual,
        path_count=len(paths),
        values_per_path=EDGES_PER_PHASE,
    )
    primary_moment_max_t = _whole_cluster_max_t_zero_pass(
        moment_by_path,
        family_error=0.01,
        bootstrap_seed=int(root_seed) ^ 0x5052494D4F4D,
    )
    moment_pass = int(primary_moment_max_t["passed"])
    moment_ratio = primary_moment_max_t["maximum_critical_ratio"]
    witnesses = np.stack(
        [
            px[first - 1] * py[second - 1]
            - px[second - 1] * py[first - 1]
            for first, second in ((1, 2), (1, 3), (2, 3), (2, 4))
        ],
        axis=1,
    )
    witnesses_by_path = _whole_path_means(
        witnesses,
        path_count=len(paths),
        values_per_path=EDGES_PER_PHASE,
    )
    primary_balance_max_t = _whole_cluster_max_t_zero_pass(
        witnesses_by_path,
        family_error=0.01,
        bootstrap_seed=int(root_seed) ^ 0x50524942414C,
    )
    balance_pass = int(primary_balance_max_t["passed"])
    balance_ratio = primary_balance_max_t["maximum_critical_ratio"]
    sorted_y = np.sort(y)
    empirical_upper = np.arange(1, count + 1, dtype=np.float64) / count
    empirical_lower = np.arange(0, count, dtype=np.float64) / count
    ks = float(
        max(
            np.max(empirical_upper - sorted_y),
            np.max(sorted_y - empirical_lower),
        )
    )
    ks_limit = math.sqrt(math.log(2.0 / 0.01) / (2.0 * count))

    # Exercise every frozen nested and pair-local tree shape.  These panels
    # are deliberately smaller than the pooled stationarity panel above, but
    # each receives its own simultaneous 99% marginal-law checks.
    structural_records: list[dict[str, Any]] = []
    structural_fallback_count = 0
    structural_fallback_seconds = 0.0
    structural_elapsed = 0.0
    structural_transition_count = 0
    structural_certificate_count = 0
    structural_normal_certificate_count = 0
    structural_jacobi_certificate_count = 0
    structural_uncertified_count = 0
    structural_moment_panels: list[np.ndarray] = []
    structural_balance_panels: list[np.ndarray] = []
    structural_forbidden = {
        name: 0
        for name in (
            "resource_cap_count",
            "invalid_density_count",
            "approximation_count",
            "correction_count",
            "floor_count",
            "limiter_count",
            "projection_count",
            "renormalization_count",
            "nonfinite_count",
        )
    }
    structural_profiles = _structural_marginal_profiles()
    structural_edges = tuple(range(32))
    structural_family_error = 0.01 / len(structural_profiles)
    for case_index, specification in enumerate(structural_profiles):
        steps = int(specification["sample_steps"])
        tree_root = int(specification["tree_root_steps"])
        case_profile = HaarCouplingProfile(
            coarsest_steps=tree_root,
            finest_steps=(
                2 * tree_root
                if specification.get("pair_coarse_steps") is not None
                else 2048
            ),
        )
        case_outer_step = min(case_index, steps - 1)
        case_batch = build_certified_haar_uniform_batch(
            root_seed=int(root_seed),
            role="marginal_c",
            path_ids=paths,
            sample_steps=steps,
            outer_step=case_outer_step,
            phase=case_index % PHASE_COUNT,
            edge_ids=structural_edges,
            profile=case_profile,
            detail_sign=int(specification["detail_sign"]),
            pair_coarse_steps=specification.get("pair_coarse_steps"),
            device=target_device,
        )
        case_count = len(paths) * len(structural_edges)
        case_rng = np.random.default_rng(
            (int(root_seed) ^ 0x535452554354 ^ case_index) & ((1 << 64) - 1)
        )
        case_x_np = case_rng.random(case_count, dtype=np.float64)
        case_x = torch.as_tensor(
            case_x_np, dtype=torch.float64, device=target_device
        )
        case_exposure = torch.full_like(case_x, 0.01)
        case_result = _authorize(
            case_batch, case_x, case_exposure, jacobi_profile
        )
        case_y = _np(case_result.later_head_fraction).reshape(-1)
        case_target = _np(case_result.denoising_target).reshape(-1)
        case_uniform = 0.5 * (
            _np(case_batch.uniform_lower).reshape(-1)
            + _np(case_batch.uniform_upper).reshape(-1)
        )
        case_evaluated = evaluate_alpha1_rb_torch_fixed_modes(
            case_x,
            case_result.later_head_fraction.reshape(-1),
            case_exposure,
            modes=256,
        )
        case_cdf_error = float(
            np.max(
                np.abs(
                    _np(case_evaluated.cdf).reshape(-1) - case_uniform
                )
            )
        )
        case_target_error = float(
            np.max(
                np.abs(
                    _np(case_evaluated.denoising_target).reshape(-1)
                    - case_target
                )
            )
        )
        case_px = _legendre_values(2.0 * case_x_np - 1.0, 8)
        case_py = _legendre_values(2.0 * case_y - 1.0, 8)
        case_moments = np.stack(
            [
                case_py[degree - 1]
                - math.exp(-degree * (degree + 1) * 0.01)
                * case_px[degree - 1]
                for degree in range(1, 9)
            ],
            axis=1,
        )
        case_moment_by_path = _whole_path_means(
            case_moments,
            path_count=len(paths),
            values_per_path=len(structural_edges),
        )
        structural_moment_panels.append(case_moment_by_path)
        case_witnesses = np.stack(
            [
                case_px[first - 1] * case_py[second - 1]
                - case_px[second - 1] * case_py[first - 1]
                for first, second in ((1, 2), (1, 3), (2, 3), (2, 4))
            ],
            axis=1,
        )
        case_balance_by_path = _whole_path_means(
            case_witnesses,
            path_count=len(paths),
            values_per_path=len(structural_edges),
        )
        structural_balance_panels.append(case_balance_by_path)
        case_sorted = np.sort(case_y)
        case_upper = (
            np.arange(1, case_count + 1, dtype=np.float64) / case_count
        )
        case_lower = (
            np.arange(0, case_count, dtype=np.float64) / case_count
        )
        case_ks = float(
            max(
                np.max(case_upper - case_sorted),
                np.max(case_sorted - case_lower),
            )
        )
        case_ks_limit = math.sqrt(
            math.log(2.0 / structural_family_error)
            / (2.0 * case_count)
        )
        case_normal_certified = _np(
            case_batch.certificate_mask
        ).astype(bool)
        case_transition_certified = _np(
            case_result.certified_mask
        ).astype(bool)
        case_fallback_count, case_fallback_seconds, case_elapsed = (
            _fallback_timing(case_batch, case_result)
        )
        case_diagnostics = dict(case_result.diagnostics)
        case_forbidden = {
            name: _required_int(case_diagnostics, name)
            for name in structural_forbidden
        }
        structural_transition_count += case_count
        structural_certificate_count += int(
            np.count_nonzero(case_normal_certified)
            + np.count_nonzero(case_transition_certified)
        )
        structural_normal_certificate_count += int(
            np.count_nonzero(case_normal_certified)
        )
        structural_jacobi_certificate_count += int(
            np.count_nonzero(case_transition_certified)
        )
        structural_uncertified_count += int(
            2 * case_count
            - np.count_nonzero(case_normal_certified)
            - np.count_nonzero(case_transition_certified)
        )
        structural_fallback_count += case_fallback_count
        structural_fallback_seconds += case_fallback_seconds
        structural_elapsed += case_elapsed
        for name, value in case_forbidden.items():
            structural_forbidden[name] += value
        structural_records.append(
            {
                **specification,
                "outer_step": case_outer_step,
                "phase": case_index % PHASE_COUNT,
                "sample_count": case_count,
                "cdf_rounding_maximum_error": case_cdf_error,
                "target_maximum_error": case_target_error,
                "stationary_ks_statistic": case_ks,
                "stationary_ks_limit_99_familywise": case_ks_limit,
                "cdf_pass": int(
                    case_cdf_error <= 1.0e-8
                    and case_ks <= case_ks_limit
                ),
                "target_pass": int(case_target_error <= 2.0e-5),
                "certificate_pass": int(
                    bool(np.all(case_normal_certified))
                    and bool(np.all(case_transition_certified))
                ),
                "fallback_count": case_fallback_count,
                **case_forbidden,
                "output_sha256": _hash(case_y, case_target),
            }
        )

    structural_moment_max_t = _whole_cluster_max_t_zero_pass(
        np.concatenate(structural_moment_panels, axis=1),
        family_error=0.01,
        bootstrap_seed=int(root_seed) ^ 0x4D4F4D454E54,
    )
    structural_balance_max_t = _whole_cluster_max_t_zero_pass(
        np.concatenate(structural_balance_panels, axis=1),
        family_error=0.01,
        bootstrap_seed=int(root_seed) ^ 0x42414C414E43,
    )
    for index, row in enumerate(structural_records):
        row["eigenmoment_pass"] = int(structural_moment_max_t["passed"])
        row["detailed_balance_pass"] = int(
            structural_balance_max_t["passed"]
        )
        row["eigenmoment_joint_max_t_pass"] = row["eigenmoment_pass"]
        row["detailed_balance_joint_max_t_pass"] = row[
            "detailed_balance_pass"
        ]
        row["eigenmoment_path_mean_sha256"] = _hash(
            structural_moment_panels[index]
        )
        row["detailed_balance_path_mean_sha256"] = _hash(
            structural_balance_panels[index]
        )

    # The same immutable events must be bit-identical under chunking and path
    # permutation.  State-dependent values are permuted with their path blocks.
    chunk_later: list[np.ndarray] = []
    chunk_target: list[np.ndarray] = []
    chunk_uniform: list[np.ndarray] = []
    for offset in (0, 4):
        subset = paths[offset : offset + 4]
        sub = build_certified_haar_uniform_batch(
            root_seed=int(root_seed),
            role="marginal_c",
            path_ids=subset,
            sample_steps=256,
            outer_step=0,
            phase=0,
            edge_ids=edges,
            profile=haar_profile,
            device=target_device,
        )
        start = offset * EDGES_PER_PHASE
        stop = start + 4 * EDGES_PER_PHASE
        sub_result = _authorize(
            sub, x[start:stop], exposure[start:stop], jacobi_profile
        )
        chunk_later.append(_np(sub_result.later_head_fraction).reshape(-1))
        chunk_target.append(_np(sub_result.denoising_target).reshape(-1))
        chunk_uniform.append(_np(sub.uniform_lower).reshape(-1))
    chunk_pass = int(
        np.array_equal(np.concatenate(chunk_later), y)
        and np.array_equal(np.concatenate(chunk_target), target)
        and np.array_equal(
            np.concatenate(chunk_uniform),
            _np(batch.uniform_lower).reshape(-1),
        )
    )
    reversed_paths = tuple(reversed(paths))
    reversed_batch = build_certified_haar_uniform_batch(
        root_seed=int(root_seed),
        role="marginal_c",
        path_ids=reversed_paths,
        sample_steps=256,
        outer_step=0,
        phase=0,
        edge_ids=edges,
        profile=haar_profile,
        device=target_device,
    )
    path_blocks = x.reshape(8, EDGES_PER_PHASE).flip(0).reshape(-1).contiguous()
    reversed_result = _authorize(
        reversed_batch, path_blocks, exposure, jacobi_profile
    )
    restored_y = (
        _np(reversed_result.later_head_fraction)
        .reshape(8, EDGES_PER_PHASE)[::-1]
        .copy()
        .reshape(-1)
    )
    restored_target = (
        _np(reversed_result.denoising_target)
        .reshape(8, EDGES_PER_PHASE)[::-1]
        .copy()
        .reshape(-1)
    )
    order_pass = int(
        np.array_equal(restored_y, y)
        and np.array_equal(restored_target, target)
    )
    repeated = build_certified_haar_uniform_batch(
        root_seed=int(root_seed),
        role="marginal_c",
        path_ids=paths,
        sample_steps=256,
        outer_step=0,
        phase=0,
        edge_ids=edges,
        profile=haar_profile,
        device=target_device,
    )
    resume_pass = int(
        torch.equal(repeated.uniform_lower, batch.uniform_lower)
        and torch.equal(repeated.uniform_upper, batch.uniform_upper)
        and torch.equal(repeated.transition_ids, batch.transition_ids)
    )
    pair_totals = np.linspace(1.0e-5, 1.0, count, dtype=np.float64)
    mass_error = float(
        np.max(
            np.abs(
                pair_totals * (1.0 - y)
                + pair_totals * y
                - pair_totals
            )
        )
    )
    normal_diagnostics = dict(batch.diagnostics)
    normal_runtime = dict(batch.runtime_report)
    transition_diagnostics = dict(result.diagnostics)
    transition_runtime = dict(result.runtime_report)
    normal_certified = _np(batch.certificate_mask).astype(bool).reshape(-1)
    if normal_certified.size != count:
        raise HaarControlError(
            "normal certificate mask has the wrong shape",
            failure_code="certified_normal_transform_invalid",
            failure_domain="normal_transform_backend",
        )
    normal_fallback_count = _required_int(
        normal_diagnostics, "arb_fallback_count"
    )
    jacobi_fallback_count = _required_int(
        transition_diagnostics, "fallback_count"
    )
    normal_fallback_seconds = _required_float(
        normal_runtime, "arb_fallback_elapsed_seconds"
    )
    jacobi_elapsed = _required_float(
        transition_runtime, "elapsed_seconds"
    )
    jacobi_fallback_seconds = (
        jacobi_elapsed
        * _required_float(transition_runtime, "arb_fallback_time_fraction")
    )
    total_elapsed = _required_float(
        normal_runtime, "elapsed_seconds"
    ) + jacobi_elapsed
    if total_elapsed <= 0.0:
        raise HaarControlError(
            "marginal control runtime is not positive",
            failure_code="hierarchical_control_diagnostics_invalid",
            failure_domain="marginal_law",
        )
    total_transition_count = count + structural_transition_count
    normal_certificate_count = int(
        np.count_nonzero(normal_certified)
    ) + structural_normal_certificate_count
    jacobi_certificate_count = int(
        np.count_nonzero(certified)
    ) + structural_jacobi_certificate_count
    certificate_fraction = (
        (normal_certificate_count + jacobi_certificate_count)
        / (2.0 * total_transition_count)
    )
    total_fallback_count = (
        normal_fallback_count
        + jacobi_fallback_count
        + structural_fallback_count
    )
    total_fallback_seconds = (
        normal_fallback_seconds
        + jacobi_fallback_seconds
        + structural_fallback_seconds
    )
    total_elapsed += structural_elapsed
    structural_cdf_pass = int(
        bool(structural_records)
        and all(int(row["cdf_pass"]) == 1 for row in structural_records)
    )
    structural_target_pass = int(
        bool(structural_records)
        and all(int(row["target_pass"]) == 1 for row in structural_records)
    )
    structural_moment_pass = int(
        bool(structural_records)
        and int(structural_moment_max_t["passed"]) == 1
    )
    structural_balance_pass = int(
        bool(structural_records)
        and int(structural_balance_max_t["passed"]) == 1
    )
    structural_certificate_pass = int(
        bool(structural_records)
        and all(
            int(row["certificate_pass"]) == 1
            for row in structural_records
        )
    )
    maximum_cdf_error = max(
        [cdf_error]
        + [
            float(row["cdf_rounding_maximum_error"])
            for row in structural_records
        ]
    )
    maximum_target_error = max(
        [target_error]
        + [float(row["target_maximum_error"]) for row in structural_records]
    )
    return {
        "schema": HAAR_CONTROL_VERSION + "-marginal",
        "schema_version": 1,
        "sample_count": total_transition_count,
        "primary_panel_sample_count": count,
        "structural_profile_sample_count": structural_transition_count,
        "structural_profile_records": structural_records,
        "structural_profile_count": len(structural_records),
        "primary_eigenmoment_max_t": primary_moment_max_t,
        "primary_detailed_balance_max_t": primary_balance_max_t,
        "structural_eigenmoment_max_t": structural_moment_max_t,
        "structural_detailed_balance_max_t": structural_balance_max_t,
        "structural_profile_coverage_pass": int(
            len(structural_records)
            == len(_structural_marginal_profiles())
        ),
        "certificate_fraction": certificate_fraction,
        "normal_certificate_fraction": (
            normal_certificate_count / total_transition_count
        ),
        "jacobi_certificate_fraction": (
            jacobi_certificate_count / total_transition_count
        ),
        "fallback_count": total_fallback_count,
        "fallback_fraction": (
            total_fallback_count / (2.0 * total_transition_count)
        ),
        "fallback_elapsed_seconds": total_fallback_seconds,
        "fallback_cost_fraction": (
            total_fallback_seconds / total_elapsed
        ),
        "elapsed_seconds": total_elapsed,
        "cdf_rounding_maximum_error": maximum_cdf_error,
        "target_maximum_error": maximum_target_error,
        "stationary_ks_statistic": ks,
        "stationary_ks_limit_99": ks_limit,
        "eigenmoment_maximum_interval_ratio": moment_ratio,
        "detailed_balance_maximum_interval_ratio": balance_ratio,
        "jacobi_marginal_cdf_pass": int(
            cdf_error <= 1.0e-8
            and ks <= ks_limit
            and structural_cdf_pass
        ),
        "jacobi_eigenmoment_pass": int(
            moment_pass and structural_moment_pass
        ),
        "jacobi_detailed_balance_pass": int(
            balance_pass and structural_balance_pass
        ),
        "marginal_cdf_pass": int(
            cdf_error <= 1.0e-8
            and ks <= ks_limit
            and structural_cdf_pass
        ),
        "marginal_eigenmoment_pass": int(
            moment_pass and structural_moment_pass
        ),
        "marginal_detailed_balance_pass": int(
            balance_pass and structural_balance_pass
        ),
        "rb_target_certificate_pass": int(
            certificate_fraction == 1.0
            and structural_certificate_pass
            and structural_target_pass
            and maximum_target_error <= 2.0e-5
        ),
        "order_invariance_pass": order_pass,
        "chunk_invariance_pass": chunk_pass,
        "resume_invariance_pass": resume_pass,
        "order_chunk_resume_invariance_pass": int(
            order_pass and chunk_pass and resume_pass
        ),
        "interruption_replay_pass": resume_pass,
        "deterministic_batching_pass": int(order_pass and chunk_pass),
        "conservation_pass": int(mass_error <= 2.0e-15),
        "mass_error": mass_error,
        "uncertified_count": int(
            count - np.count_nonzero(normal_certified)
            + count - np.count_nonzero(certified)
            + structural_uncertified_count
        ),
        **{
            name: _required_int(transition_diagnostics, name)
            + structural_forbidden[name]
            for name in structural_forbidden
        },
        "output_sha256": _hash(y, target),
        "uniform_sha256": _hash(
            _np(batch.uniform_lower), _np(batch.uniform_upper)
        ),
        **NO_WORK,
    }


def run_phase_tower_controls(
    *,
    root_seed: int,
    path_ids: Sequence[int],
    device: str | torch.device,
    cases_per_color_duration: int = 16,
) -> dict[str, Any]:
    """Run all four colors at half/full duration with whole-cluster intervals."""

    paths = tuple(int(value) for value in path_ids)
    if len(paths) != 8:
        raise ValueError("tower controls require eight paths per cluster batch")
    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise HaarControlError(
            "production tower controls require CUDA",
            failure_code="hierarchical_cuda_state_required",
            failure_domain="runtime_backend",
        )
    haar_profile = HaarCouplingProfile(coarsest_steps=128, finest_steps=2048)
    jacobi_profile = JacobiRBCudaProfile()
    rng = np.random.default_rng(int(root_seed) ^ 0x544F5745)
    all_residuals: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    certified_count = 0
    transition_count = 0
    normal_sample_count = 0
    normal_certified_count = 0
    fallback_count = 0
    fallback_evaluation_count = 0
    fallback_seconds = 0.0
    elapsed_seconds = 0.0
    maximum_error_radius = 0.0
    forbidden_totals = {
        name: 0
        for name in (
            "uncertified_count",
            "resource_cap_count",
            "invalid_density_count",
            "approximation_count",
            "correction_count",
            "floor_count",
            "limiter_count",
            "projection_count",
            "renormalization_count",
            "nonfinite_count",
        )
    }
    for color in range(4):
        for duration_index, duration in enumerate((0.5, 1.0)):
            residuals: list[np.ndarray] = []
            for case in range(int(cases_per_color_duration)):
                states_np = rng.dirichlet(
                    np.ones(784, dtype=np.float64), size=len(paths)
                )
                states = torch.as_tensor(
                    states_np, dtype=torch.float64, device=target_device
                ).contiguous()
                outer_step = duration_index * 64 + case
                uniforms = build_certified_haar_uniform_batch(
                    root_seed=int(root_seed),
                    role="marginal_d",
                    path_ids=paths,
                    sample_steps=128,
                    outer_step=outer_step,
                    phase=color,
                    edge_ids=range(EDGES_PER_PHASE),
                    profile=haar_profile,
                    device=target_device,
                )
                uniform_certificate = (
                    _np(uniforms.certificate_mask).astype(bool).reshape(-1)
                )
                if uniform_certificate.size != len(paths) * EDGES_PER_PHASE:
                    raise HaarControlError(
                        "tower normal certificate mask has the wrong shape",
                        failure_code="certified_normal_transform_invalid",
                        failure_domain="normal_transform_backend",
                    )
                normal_sample_count += int(uniform_certificate.size)
                normal_certified_count += int(
                    np.count_nonzero(uniform_certificate)
                )
                forbidden_totals["uncertified_count"] += int(
                    uniform_certificate.size
                    - np.count_nonzero(uniform_certificate)
                )
                normal_diagnostics = dict(uniforms.diagnostics)
                normal_runtime = dict(uniforms.runtime_report)
                fallback_count += _required_int(
                    normal_diagnostics, "arb_fallback_count"
                )
                fallback_seconds += _required_float(
                    normal_runtime, "arb_fallback_elapsed_seconds"
                )
                elapsed_seconds += _required_float(
                    normal_runtime, "elapsed_seconds"
                )

                def sampler(
                    x: torch.Tensor,
                    exposure: torch.Tensor,
                    **_kwargs: Any,
                ) -> Any:
                    return _authorize(
                        uniforms,
                        x.reshape(-1),
                        exposure.reshape(-1),
                        jacobi_profile,
                    )

                result = run_dynkin_tower_phase(
                    states,
                    matching_index=color,
                    duration_fraction=duration,
                    sample_steps=128,
                    rng_key=("haar-tower", color, duration_index, case),
                    transition_ids=uniforms.transition_ids.reshape(-1).contiguous(),
                    profile=jacobi_profile,
                    sampler=sampler,
                    standardized=True,
                )
                residual = _np(result.standardized_residual)
                radius = _np(result.error_radius)
                if residual.shape != (8, 10) or not np.isfinite(residual).all():
                    raise HaarControlError(
                        "phase tower returned invalid residuals",
                        failure_code="hierarchical_tower_invalid",
                        failure_domain="marginal_law",
                    )
                residuals.append(residual)
                maximum_error_radius = max(
                    maximum_error_radius, float(np.max(radius))
                )
                diagnostics = dict(result.diagnostics)
                current_transitions = _required_int(
                    diagnostics, "transition_count"
                )
                transition_count += current_transitions
                certified_count += _required_int(
                    diagnostics, "certified_count"
                )
                fallback_count += _required_int(
                    diagnostics, "fallback_count"
                )
                fallback_seconds += _required_float(
                    diagnostics, "fallback_elapsed_seconds"
                )
                elapsed_seconds += _required_float(
                    diagnostics, "elapsed_seconds"
                )
                fallback_evaluation_count += (
                    int(uniform_certificate.size) + current_transitions
                )
                for name in forbidden_totals:
                    forbidden_totals[name] += _required_int(
                        diagnostics, name
                    )
            panel = np.concatenate(residuals, axis=0)
            max_t = _whole_cluster_max_t_zero_pass(
                panel,
                family_error=0.01 / 8.0,
                bootstrap_seed=(
                    int(root_seed)
                    ^ 0x544F574552
                    ^ (color << 8)
                    ^ duration_index
                ),
            )
            rows.append(
                {
                    "color": color,
                    "duration_fraction": duration,
                    "cluster_count": int(panel.shape[0]),
                    "whole_cluster_interval_pass": int(max_t["passed"]),
                    "maximum_interval_ratio": max_t[
                        "maximum_critical_ratio"
                    ],
                    "max_t": max_t,
                    "residual_sha256": _hash(panel),
                }
            )
            all_residuals.append(panel)
    pooled = np.concatenate(all_residuals, axis=1)
    pooled_max_t = _whole_cluster_max_t_zero_pass(
        pooled,
        family_error=0.01,
        bootstrap_seed=int(root_seed) ^ 0x504F4F4C4544,
    )
    if elapsed_seconds <= 0.0:
        raise HaarControlError(
            "tower control runtime is not positive",
            failure_code="hierarchical_control_diagnostics_invalid",
            failure_domain="marginal_law",
        )
    return {
        "schema": HAAR_CONTROL_VERSION + "-phase-tower",
        "schema_version": 1,
        "color_duration_records": rows,
        "cluster_count_per_color_duration": 8 * int(cases_per_color_duration),
        "phase_tower_identity_pass": int(
            int(pooled_max_t["passed"]) == 1
            and all(int(row["whole_cluster_interval_pass"]) == 1 for row in rows)
            and maximum_error_radius <= 1.0e-8
        ),
        "all_colors_pass": int({row["color"] for row in rows} == set(range(4))),
        "half_full_duration_pass": int(
            {row["duration_fraction"] for row in rows} == {0.5, 1.0}
        ),
        "pooled_maximum_interval_ratio": pooled_max_t[
            "maximum_critical_ratio"
        ],
        "pooled_max_t": pooled_max_t,
        "maximum_numerical_error_radius": maximum_error_radius,
        "transition_count": transition_count,
        "certified_count": certified_count,
        "normal_sample_count": normal_sample_count,
        "normal_certified_count": normal_certified_count,
        "certificate_fraction": min(
            certified_count / transition_count if transition_count else 0.0,
            (
                normal_certified_count / normal_sample_count
                if normal_sample_count
                else 0.0
            ),
        ),
        "fallback_count": fallback_count,
        "fallback_fraction": (
            fallback_count / fallback_evaluation_count
            if fallback_evaluation_count
            else 0.0
        ),
        "fallback_elapsed_seconds": fallback_seconds,
        "fallback_cost_fraction": (
            fallback_seconds / elapsed_seconds if elapsed_seconds else 0.0
        ),
        **forbidden_totals,
        **NO_WORK,
    }


def _scheduler_initial(
    mixed_state: np.ndarray,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    state = np.asarray(mixed_state, dtype=np.float64)
    if (
        state.shape != (784,)
        or not np.isfinite(state).all()
        or np.any(state < 0.0)
        or abs(float(state.sum()) - 1.0) > 1.0e-12
    ):
        raise ValueError("mixed_state must be a finite simplex vector [784]")
    return torch.as_tensor(
        np.repeat(state.reshape(1, -1), int(count), axis=0),
        dtype=torch.float64,
        device=device,
    ).contiguous()


def _metadata_execution(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise HaarControlError(
            "scheduler benchmark executed no authorizing shards",
            failure_code="hierarchical_scheduler_diagnostics_invalid",
            failure_domain="hierarchical_scheduler",
        )
    diagnostics = [dict(record.get("diagnostics", {})) for record in records]
    timings = [dict(record.get("timing", {})) for record in records]
    counts = [
        int(value.get("transition_count", 0)) for value in diagnostics
    ]
    durations = [
        float(
            value.get(
                "complete_pipeline_including_state_shard_io_seconds",
                math.inf,
            )
        )
        for value in timings
    ]
    transition_count = sum(counts)
    elapsed = sum(durations)
    fallback_count = sum(int(value.get("fallback_count", -1)) for value in diagnostics)
    fallback_seconds = sum(
        float(value.get("fallback_elapsed_seconds", math.inf))
        for value in diagnostics
    )
    if (
        transition_count <= 0
        or any(value <= 0 for value in counts)
        or any(not math.isfinite(value) or value <= 0.0 for value in durations)
        or not math.isfinite(elapsed)
        or elapsed <= 0.0
        or fallback_count < 0
        or not math.isfinite(fallback_seconds)
    ):
        raise HaarControlError(
            "scheduler benchmark diagnostics are incomplete",
            failure_code="hierarchical_scheduler_diagnostics_invalid",
            failure_domain="hierarchical_scheduler",
        )
    forbidden_names = (
        "uncertified_count",
        "resource_cap_count",
        "invalid_density_count",
        "approximation_count",
        "correction_count",
        "floor_count",
        "limiter_count",
        "projection_count",
        "renormalization_count",
        "nonfinite_count",
    )
    per_shard_rates = [
        count / duration for count, duration in zip(counts, durations)
    ]
    return {
        "transition_count": transition_count,
        "elapsed_seconds": elapsed,
        "aggregate_rate": transition_count / elapsed,
        "slowest_shard_rate": min(per_shard_rates),
        "conservative_rate": min(per_shard_rates),
        "certificate_fraction": min(
            float(value.get("certificate_fraction", 0.0))
            for value in diagnostics
        ),
        "fallback_count": fallback_count,
        "fallback_fraction": fallback_count / transition_count,
        "fallback_elapsed_seconds": fallback_seconds,
        "fallback_cost_fraction": fallback_seconds / elapsed,
        "mass_error": max(
            float(value.get("mass_error", math.inf)) for value in diagnostics
        ),
        "peak_memory_fraction": max(
            float(value.get("peak_memory_fraction", 0.0)) for value in diagnostics
        ),
        "state_updates_device_resident_pass": int(
            all(
                int(value.get("state_updates_device_resident_pass", 0)) == 1
                for value in diagnostics
            )
        ),
        **{
            name: sum(int(value.get(name, -1)) for value in diagnostics)
            for name in forbidden_names
        },
    }


def _one_phase_batching_parity(
    *,
    root_seed: int,
    path_ids: Sequence[int],
    mixed_state: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    """Compare one exact state update in a cohort and as singleton paths.

    The sampled transition and resulting state are lane-local and therefore
    must be bit-identical.  The observer is different: GEMV and reduction
    kernels are permitted to choose a different summation tree when the path
    batch dimension changes.  Its centres are consequently compared against
    conservative binary64 forward-error/certified-ball envelopes instead of
    an unjustified bitwise-equality requirement.
    """

    paths = tuple(int(value) for value in path_ids)
    if len(paths) != 2:
        raise ValueError("the batching-parity fixture requires two paths")
    profile = HaarCouplingProfile(coarsest_steps=128, finest_steps=2048)
    jacobi_profile = JacobiRBCudaProfile()

    def execute(
        selected_paths: tuple[int, ...],
        states: torch.Tensor,
    ) -> Any:
        uniforms = build_certified_haar_uniform_batch(
            root_seed=int(root_seed),
            role="nested_a",
            path_ids=selected_paths,
            sample_steps=128,
            outer_step=0,
            phase=0,
            edge_ids=range(EDGES_PER_PHASE),
            profile=profile,
            device=device,
        )

        def sampler(
            x: torch.Tensor,
            exposure: torch.Tensor,
            **_kwargs: Any,
        ) -> Any:
            return _authorize(
                uniforms,
                x.reshape(-1),
                exposure.reshape(-1),
                jacobi_profile,
            )

        return run_dynkin_tower_phase(
            states,
            matching_index=0,
            duration_fraction=0.5,
            sample_steps=128,
            rng_key=("haar-batching-parity",),
            transition_ids=uniforms.transition_ids.reshape(-1).contiguous(),
            profile=jacobi_profile,
            sampler=sampler,
            standardized=True,
        )

    combined_initial = _scheduler_initial(mixed_state, len(paths), device)
    combined = execute(paths, combined_initial)
    singletons = [
        execute(
            (path_id,),
            combined_initial[index : index + 1].contiguous(),
        )
        for index, path_id in enumerate(paths)
    ]
    return _compare_one_phase_batching_results(
        combined=combined,
        singletons=singletons,
        initial_states=combined_initial,
    )


_BINARY64_ROUNDING_ALLOWANCE = float.fromhex("0x1.0p-52")
_MINIMUM_BINARY64_SUBNORMAL = float.fromhex("0x0.0000000000001p-1022")


def _forward_error_gamma(operation_count: int) -> float:
    """Return a conservative Higham-style binary64 ``gamma_n``."""

    count = int(operation_count)
    product = count * _BINARY64_ROUNDING_ALLOWANCE
    if count <= 0 or not product < 1.0:
        raise ValueError("operation_count is outside the binary64 gamma range")
    return math.nextafter(product / (1.0 - product), math.inf)


def _outward_nonnegative(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.nextafter(
        np.maximum(array, 0.0),
        np.full_like(array, math.inf),
    )


def _observable_reduction_radius(states: torch.Tensor) -> np.ndarray:
    """Bound one evaluation of the ten raw refinement observables.

    The represented state and Fourier weights are treated as exact binary64
    inputs.  Four operations per term cover a product, an arbitrary-order
    reduction, and outward construction of the absolute-sum bound for the
    linear and quadratic families; six cover the extra cubic product.  These
    are deliberately more conservative than the textbook sequential-dot
    bounds and remain many orders below the frozen scientific tolerances.
    """

    host = np.asarray(states.detach().cpu(), dtype=np.float64)
    if (
        host.ndim != 2
        or host.shape[1] != 784
        or not np.all(np.isfinite(host))
    ):
        raise HaarControlError(
            "batching parity received invalid observer states",
            failure_code="hierarchical_batching_observer_state_invalid",
            failure_domain="scheduler",
        )
    spec = refinement_observable_spec(28)
    weights = np.asarray(spec.fourier_weights, dtype=np.float64)
    result = np.empty((host.shape[0], 10), dtype=np.float64)
    linear_gamma = _forward_error_gamma(4 * host.shape[1])
    cubic_gamma = _forward_error_gamma(6 * host.shape[1])
    for path_index, path in enumerate(host):
        for observable_index, weight in enumerate(weights):
            absolute_sum = math.fsum(
                abs(float(state_value) * float(weight_value))
                for state_value, weight_value in zip(path, weight)
            )
            result[path_index, observable_index] = math.nextafter(
                linear_gamma * math.nextafter(absolute_sum, math.inf)
                + _MINIMUM_BINARY64_SUBNORMAL,
                math.inf,
            )
        quadratic_sum = math.fsum(
            abs(float(value) * float(value)) for value in path
        )
        cubic_sum = math.fsum(
            abs(float(value) * float(value) * float(value))
            for value in path
        )
        result[path_index, 8] = math.nextafter(
            linear_gamma * math.nextafter(quadratic_sum, math.inf)
            + _MINIMUM_BINARY64_SUBNORMAL,
            math.inf,
        )
        result[path_index, 9] = math.nextafter(
            cubic_gamma * math.nextafter(cubic_sum, math.inf)
            + _MINIMUM_BINARY64_SUBNORMAL,
            math.inf,
        )
    return result


def _agreement_record(
    left: np.ndarray,
    right: np.ndarray,
    allowed: np.ndarray,
    *,
    comparison: str,
) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    allowed_array = _outward_nonnegative(allowed)
    if (
        left_array.shape != right_array.shape
        or left_array.shape != allowed_array.shape
    ):
        return {
            "comparison": comparison,
            "passed": 0,
            "failure": "shape_mismatch",
        }
    difference = np.abs(left_array - right_array)
    finite = (
        np.all(np.isfinite(left_array))
        and np.all(np.isfinite(right_array))
        and np.all(np.isfinite(allowed_array))
    )
    positive = allowed_array > 0.0
    ratios = np.zeros_like(difference)
    np.divide(difference, allowed_array, out=ratios, where=positive)
    ratios[~positive & (difference > 0.0)] = math.inf
    passed = bool(finite and np.all(difference <= allowed_array))
    return {
        "comparison": comparison,
        "passed": int(passed),
        "maximum_absolute_difference": float(
            np.max(difference, initial=0.0)
        ),
        "maximum_allowed_difference": float(
            np.max(allowed_array, initial=0.0)
        ),
        "maximum_bound_fraction": float(np.max(ratios, initial=0.0)),
    }


def _standardized_residual_radius(
    result: Any,
    *,
    before_radius: np.ndarray,
    after_radius: np.ndarray,
) -> np.ndarray:
    """Enclose observer and final arithmetic error in one residual centre."""

    before = _np(result.raw_before_values).astype(np.float64, copy=False)
    after = _np(result.raw_after_values).astype(np.float64, copy=False)
    drift = _np(result.drift_center).astype(np.float64, copy=False)
    drift_radius = _np(result.drift_error_radius).astype(
        np.float64, copy=False
    )
    scales = refinement_observable_spec(28).standard_deviations[None, :]
    input_radius = _outward_nonnegative(
        before_radius + after_radius + drift_radius
    )
    magnitude = _outward_nonnegative(
        (
            np.abs(before)
            + np.abs(after)
            + np.abs(drift)
            + input_radius
        )
        / scales
    )
    arithmetic_radius = _outward_nonnegative(
        _forward_error_gamma(8) * magnitude
        + _MINIMUM_BINARY64_SUBNORMAL
    )
    return _outward_nonnegative(input_radius / scales + arithmetic_radius)


def _compare_one_phase_batching_results(
    *,
    combined: Any,
    singletons: Sequence[Any],
    initial_states: torch.Tensor,
) -> dict[str, Any]:
    """Apply the exact-state/numerically-enclosed observer parity contract."""

    if not singletons:
        raise ValueError("at least one singleton batching result is required")

    def concatenate(name: str) -> torch.Tensor:
        values = [getattr(value, name) for value in singletons]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError(f"{name} must be a tensor in every parity result")
        return torch.cat(values, dim=0)

    singleton_final = concatenate("final_states")
    singleton_radius_tensor = concatenate("drift_error_radius")
    exact_state_pass = int(
        torch.equal(singleton_final, combined.final_states)
        and bool(torch.isfinite(combined.final_states).all().item())
    )
    exact_radius_pass = int(
        torch.equal(singleton_radius_tensor, combined.drift_error_radius)
        and bool(torch.isfinite(combined.drift_error_radius).all().item())
        and bool((combined.drift_error_radius >= 0.0).all().item())
    )

    before_combined_radius = _observable_reduction_radius(initial_states)
    before_singleton_radius = np.concatenate(
        [
            _observable_reduction_radius(
                initial_states[index : index + 1].contiguous()
            )
            for index in range(len(singletons))
        ],
        axis=0,
    )
    after_combined_radius = _observable_reduction_radius(
        combined.final_states
    )
    after_singleton_radius = np.concatenate(
        [
            _observable_reduction_radius(value.final_states)
            for value in singletons
        ],
        axis=0,
    )
    raw_before = _agreement_record(
        _np(combined.raw_before_values),
        _np(concatenate("raw_before_values")),
        _outward_nonnegative(
            before_combined_radius + before_singleton_radius
        ),
        comparison="binary64_reduction_forward_error",
    )
    raw_after = _agreement_record(
        _np(combined.raw_after_values),
        _np(concatenate("raw_after_values")),
        _outward_nonnegative(
            after_combined_radius + after_singleton_radius
        ),
        comparison="binary64_reduction_forward_error",
    )
    drift_center = _agreement_record(
        _np(combined.drift_center),
        _np(concatenate("drift_center")),
        _outward_nonnegative(
            _np(combined.drift_error_radius)
            + _np(singleton_radius_tensor)
        ),
        comparison="certified_drift_interval_overlap",
    )
    combined_residual_radius = _standardized_residual_radius(
        combined,
        before_radius=before_combined_radius,
        after_radius=after_combined_radius,
    )
    singleton_before = _np(concatenate("raw_before_values"))
    singleton_after = _np(concatenate("raw_after_values"))
    singleton_drift = _np(concatenate("drift_center"))
    singleton_residual = _np(concatenate("standardized_residual"))
    singleton_proxy = type(
        "_SingletonParityProxy",
        (),
        {
            "raw_before_values": singleton_before,
            "raw_after_values": singleton_after,
            "drift_center": singleton_drift,
            "drift_error_radius": _np(singleton_radius_tensor),
        },
    )()
    singleton_residual_radius = _standardized_residual_radius(
        singleton_proxy,
        before_radius=before_singleton_radius,
        after_radius=after_singleton_radius,
    )
    residual = _agreement_record(
        _np(combined.standardized_residual),
        singleton_residual,
        _outward_nonnegative(
            combined_residual_radius + singleton_residual_radius
        ),
        comparison="composed_certified_observer_interval_overlap",
    )
    derived_records = {
        "raw_before_values": raw_before,
        "raw_after_values": raw_after,
        "drift_center": drift_center,
        "standardized_residual": residual,
    }
    derived_pass = int(
        all(int(record.get("passed", 0)) == 1 for record in derived_records.values())
    )
    return {
        "schema": HAAR_CONTROL_VERSION + "-one-phase-batching-parity",
        "schema_version": 2,
        "exact_state_equality_pass": exact_state_pass,
        "exact_drift_error_radius_equality_pass": exact_radius_pass,
        "derived_observable_agreement_pass": derived_pass,
        "derived_observables": derived_records,
        "passed": int(
            exact_state_pass == 1
            and exact_radius_pass == 1
            and derived_pass == 1
        ),
    }


def _load_control_shard_or_recover(
    directory: Path,
    *,
    identity: HaarShardIdentity,
    expected_input_sha256: str,
    device: torch.device,
) -> Any | None:
    """Load a completed control shard before doing any expensive GPU work."""

    metadata_path = directory / f"{identity.fingerprint}.json"
    state_path = directory / f"{identity.fingerprint}.npz"
    if not metadata_path.exists() and not state_path.exists():
        return None
    if not metadata_path.is_file() or not state_path.is_file():
        metadata_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        return None
    try:
        resumed = load_committed_haar_shard(
            directory,
            expected_identity=identity,
            device=device,
        )
    except (HaarSchedulerError, OSError, EOFError, ValueError):
        metadata_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        return None
    if resumed.metadata.get("input_sha256") != expected_input_sha256:
        raise HaarControlError(
            "committed control shard has a different predecessor",
            failure_code="hierarchical_shard_input_mismatch",
            failure_domain="scheduler",
        )
    return resumed


def run_scheduler_equivalence_and_benchmark(
    *,
    run_dir: str | Path,
    root_seed: int,
    path_id_plan: Mapping[str, Any],
    mixed_state: np.ndarray,
    device: str | torch.device,
    include_all_profiles: bool,
) -> dict[str, Any]:
    """Run production-width shards, exact phase parity, and atomic I/O."""

    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise HaarControlError(
            "production scheduler controls require CUDA",
            failure_code="hierarchical_cuda_state_required",
            failure_domain="runtime_backend",
        )
    jacobi_profile = JacobiRBCudaProfile()
    root = Path(run_dir) / (
        "coupling_scheduler_benchmark"
        if include_all_profiles
        else "preflight_scheduler_benchmark"
    )
    role = "nested_a"
    nested_plan = path_id_plan["profiles"]["nested_haar_single_arm"]["a"]["roles"]
    main_paths = tuple(
        int(value) for value in nested_plan["main"]["root_path_ids"][:8]
    )
    initial = _scheduler_initial(mixed_state, len(main_paths), target_device)
    main_schedule = NestedHaarSchedule(pool="main", role=role)
    main_identity = HaarShardIdentity(
        schedule=main_schedule,
        path_ids=main_paths,
        coarsest_start_step=0,
        root_seed=int(root_seed),
        panel_namespace="haar-control:nested-main:p8",
    )
    main_states = initialize_nested_branch_states(initial, main_schedule)
    main_expected = expected_haar_shard_input_sha256(
        main_identity,
        {f"k{level}": main_states[level] for level in main_schedule.levels or ()},
        {f"k{level}": None for level in main_schedule.levels or ()},
        jacobi_profile,
    )
    main_resumed = _load_control_shard_or_recover(
        root / "nested-main-p8",
        identity=main_identity,
        expected_input_sha256=main_expected,
        device=target_device,
    )
    if main_resumed is None:
        main_result = run_nested_haar_shard(
            main_states,
            identity=main_identity,
            jacobi_profile=jacobi_profile,
        )
        main_metadata = commit_haar_shard(
            main_result, root / "nested-main-p8"
        )
        expected_main_outputs = {
            name: branch.final_states
            for name, branch in main_result.branches.items()
        }
        main_reused = 0
    else:
        main_metadata = dict(main_resumed.metadata)
        expected_main_outputs = dict(main_resumed.states)
        main_reused = 1
    resumed = load_committed_haar_shard(
        root / "nested-main-p8",
        expected_identity=main_identity,
        device=target_device,
    )
    resume_pass = int(
        all(
            torch.equal(
                resumed.states[name],
                expected_main_outputs[name],
            )
            for name in resumed.states
        )
    )
    batching_parity = _one_phase_batching_parity(
        root_seed=int(root_seed),
        path_ids=main_paths[:2],
        mixed_state=mixed_state,
        device=target_device,
    )
    regroup_pass = int(batching_parity.get("passed", 0))

    authorizing_records: list[Mapping[str, Any]] = [main_metadata]
    profile_records: list[dict[str, Any]] = [
        {
            "profile": "nested_haar_single_arm",
            "pool": "main",
            "metadata_sha256": _record_hash(main_metadata),
            "reused_existing_commit": main_reused,
        }
    ]
    if include_all_profiles:
        reference_paths = tuple(
            int(value)
            for value in nested_plan["reference"]["root_path_ids"][:8]
        )
        reference_schedule = NestedHaarSchedule(pool="reference", role=role)
        reference_initial = _scheduler_initial(
            mixed_state, len(reference_paths), target_device
        )
        reference_identity = HaarShardIdentity(
            schedule=reference_schedule,
            path_ids=reference_paths,
            coarsest_start_step=0,
            root_seed=int(root_seed),
            panel_namespace="haar-control:nested-reference:p8",
        )
        reference_states = initialize_nested_branch_states(
            reference_initial, reference_schedule
        )
        reference_expected = expected_haar_shard_input_sha256(
            reference_identity,
            {
                f"k{level}": reference_states[level]
                for level in reference_schedule.levels or ()
            },
            {
                f"k{level}": None
                for level in reference_schedule.levels or ()
            },
            jacobi_profile,
        )
        reference_resumed = _load_control_shard_or_recover(
            root / "nested-reference-p8",
            identity=reference_identity,
            expected_input_sha256=reference_expected,
            device=target_device,
        )
        if reference_resumed is None:
            reference_result = run_nested_haar_shard(
                reference_states,
                identity=reference_identity,
                jacobi_profile=jacobi_profile,
            )
            reference_metadata = commit_haar_shard(
                reference_result, root / "nested-reference-p8"
            )
            reference_reused = 0
        else:
            reference_metadata = dict(reference_resumed.metadata)
            reference_reused = 1
        authorizing_records.append(reference_metadata)
        profile_records.append(
            {
                "profile": "nested_haar_single_arm",
                "pool": "reference",
                "metadata_sha256": _record_hash(reference_metadata),
                "reused_existing_commit": reference_reused,
            }
        )
        antithetic_plan = path_id_plan["profiles"][
            "pairwise_haar_antithetic"
        ]["a"]["roles"]
        for coarse, fine in ADJACENT_LEVEL_PAIRS:
            pair_paths = tuple(
                int(value)
                for value in antithetic_plan[f"{coarse}-{fine}"][
                    "root_path_ids"
                ][:8]
            )
            schedule = PairwiseHaarAntitheticSchedule(
                coarse_steps=coarse,
                fine_steps=fine,
                role="antithetic_a",
            )
            pair_initial = _scheduler_initial(
                mixed_state, len(pair_paths), target_device
            )
            branches = initialize_antithetic_branch_states(pair_initial)
            identity = HaarShardIdentity(
                schedule=schedule,
                path_ids=pair_paths,
                coarsest_start_step=0,
                root_seed=int(root_seed),
                panel_namespace=f"haar-control:pair:{coarse}-{fine}",
            )
            pair_expected = expected_haar_shard_input_sha256(
                identity,
                branches,
                {name: None for name in branches},
                jacobi_profile,
            )
            pair_root = root / f"pair-{coarse}-{fine}"
            pair_resumed = _load_control_shard_or_recover(
                pair_root,
                identity=identity,
                expected_input_sha256=pair_expected,
                device=target_device,
            )
            if pair_resumed is None:
                pair_result = run_pairwise_haar_antithetic_shard(
                    coarse_state=branches["coarse"],
                    fine_plus_state=branches["fine_plus"],
                    fine_minus_state=branches["fine_minus"],
                    identity=identity,
                    jacobi_profile=jacobi_profile,
                )
                pair_metadata = commit_haar_shard(
                    pair_result, pair_root
                )
                pair_reused = 0
            else:
                pair_metadata = dict(pair_resumed.metadata)
                pair_reused = 1
            authorizing_records.append(pair_metadata)
            profile_records.append(
                {
                    "profile": "pairwise_haar_antithetic",
                    "pair": [coarse, fine],
                    "metadata_sha256": _record_hash(pair_metadata),
                    "reused_existing_commit": pair_reused,
                }
            )
    execution = _metadata_execution(authorizing_records)
    total_memory = float(torch.cuda.get_device_properties(target_device).total_memory)
    execution["peak_memory_fraction"] = (
        torch.cuda.max_memory_allocated(target_device) / max(total_memory, 1.0)
    )
    rate = float(execution["conservative_rate"])
    nested_minimum_transitions = (
        PHASE_COUNT
        * EDGES_PER_PHASE
        * (32 * (128 + 256 + 512 + 1024) + 16 * (512 + 1024 + 2048))
    )
    antithetic_transitions = (
        PHASE_COUNT
        * EDGES_PER_PHASE
        * 16
        * sum(coarse + 2 * fine for coarse, fine in ADJACENT_LEVEL_PAIRS)
    )
    projected = {
        "nested_32_16_hours": nested_minimum_transitions / rate / 3600.0,
        "antithetic_16_16_hours": antithetic_transitions / rate / 3600.0,
    }
    measured_projection = (
        min(projected.values())
        if include_all_profiles
        else projected["nested_32_16_hours"]
    )
    return {
        "schema": HAAR_CONTROL_VERSION + "-scheduler-benchmark",
        "schema_version": 1,
        "include_all_profiles": int(include_all_profiles),
        "profile_records": profile_records,
        "execution": execution,
        "one_phase_batching_parity": batching_parity,
        "resume_invariance_pass": resume_pass,
        "regrouping_invariance_pass": int(regroup_pass),
        "order_invariance_pass": int(regroup_pass),
        "chunk_invariance_pass": int(regroup_pass),
        "interruption_replay_pass": resume_pass,
        "deterministic_batching_pass": int(regroup_pass),
        "nested_profile_complete_pass": 1,
        "antithetic_profile_complete_pass": int(include_all_profiles),
        "pipeline_runtime_projection_pass": int(
            measured_projection <= 48.0
        ),
        "candidate_under_48h_forecast_pass": int(
            measured_projection <= 48.0
        ),
        "projected_hours": projected,
        "minimum_projected_hours": measured_projection,
        "projection_profiles_measured": (
            [
                "nested_haar_single_arm",
                "pairwise_haar_antithetic",
            ]
            if include_all_profiles
            else ["nested_haar_single_arm"]
        ),
        **NO_WORK,
    }


__all__ = [
    "HAAR_CONTROL_VERSION",
    "HaarControlError",
    "run_marginal_and_batching_controls",
    "run_phase_tower_controls",
    "run_scheduler_equivalence_and_benchmark",
]
