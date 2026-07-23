"""Measured controls for the certified alpha-one Jacobi RB feasibility gate.

The routines in this module deliberately return raw errors and path-level
records.  The orchestration layer may gate those measurements, but it must not
turn mere finiteness into evidence for normalization, stationarity, or the
Rao--Blackwell tower identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
import torch

from mnist.d0_jacobi_denoising import (
    Alpha1SpectralConfig,
    denoising_mean_to_mass_flux,
    evaluate_alpha1_spectral,
    linear_teacher_denoising_mean,
)
from mnist.d0_jacobi_rb_spectral import (
    JacobiRBSpectralProfile,
    evaluate_alpha1_rb_torch_fixed_modes,
    sample_alpha1_rb_transition_batch,
)


CONTROL_VERSION = "jacobi-rb-measured-controls-v1"


@dataclass(frozen=True)
class ControlPanel:
    metrics: dict[str, Any]
    rows: list[dict[str, Any]]


def _legendre(degree: int, values: np.ndarray) -> np.ndarray:
    coefficient = np.zeros(int(degree) + 1, dtype=np.float64)
    coefficient[-1] = 1.0
    return np.polynomial.legendre.legval(values, coefficient)


def _spectral_config() -> Alpha1SpectralConfig:
    return Alpha1SpectralConfig(
        absolute_tolerance=1e-13, relative_tolerance=1e-12, max_modes=16_384
    )


def deterministic_kernel_controls(device: torch.device) -> ControlPanel:
    """Check spectral algebra with independent Gauss--Legendre quadrature."""

    nodes, weights = np.polynomial.legendre.leggauss(384)
    y = 0.5 * (nodes + 1.0)
    dy_weights = 0.5 * weights
    xs = np.asarray([0.13, 0.5, 0.87], dtype=np.float64)
    exposures = np.asarray([0.1, 0.35, 0.8], dtype=np.float64)
    normalization_error = 0.0
    eigenmoment_error = 0.0
    monotonicity_violation = 0.0
    rows: list[dict[str, Any]] = []
    config = _spectral_config()
    for x in xs:
        for u in exposures:
            evaluation = evaluate_alpha1_spectral(x, y, u, config=config)
            normalization = float(np.dot(dy_weights, evaluation.density))
            normalization_error = max(normalization_error, abs(normalization - 1.0))
            monotonicity_violation = max(
                monotonicity_violation,
                float(max(0.0, -np.min(np.diff(evaluation.cdf)))),
            )
            for degree in range(1, 9):
                measured = float(
                    np.dot(dy_weights, evaluation.density * _legendre(degree, nodes))
                )
                expected = math.exp(-degree * (degree + 1) * u) * float(
                    _legendre(degree, np.asarray(2.0 * x - 1.0))
                )
                error = abs(measured - expected)
                eigenmoment_error = max(eigenmoment_error, error)
                rows.append(
                    {
                        "control": "eigenmoment",
                        "x": x,
                        "exposure": u,
                        "degree": degree,
                        "measured": measured,
                        "expected": expected,
                        "absolute_error": error,
                    }
                )

    grid = np.linspace(0.03, 0.97, 17)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    direct = evaluate_alpha1_spectral(xx, yy, 0.35, config=config)
    reverse = evaluate_alpha1_spectral(yy, xx, 0.35, config=config)
    detailed_balance_error = float(np.max(np.abs(direct.density - reverse.density)))
    symmetry = evaluate_alpha1_spectral(1.0 - xx, 1.0 - yy, 0.35, config=config)
    reflection_error = float(np.max(np.abs(direct.density - symmetry.density)))

    # Semigroup: integrate k_u(z|x) k_v(y|z) over invariant Uniform(dz).
    semigroup_error = 0.0
    for x, arrival in ((0.21, 0.68), (0.5, 0.4), (0.83, 0.12)):
        first = evaluate_alpha1_spectral(x, y, 0.17, config=config).density
        second = evaluate_alpha1_spectral(y, arrival, 0.29, config=config).density
        composed = float(np.dot(dy_weights, first * second))
        expected = float(
            evaluate_alpha1_spectral(x, arrival, 0.46, config=config).density
        )
        semigroup_error = max(semigroup_error, abs(composed - expected))
        rows.append(
            {
                "control": "semigroup",
                "x": x,
                "later": arrival,
                "measured": composed,
                "expected": expected,
                "absolute_error": abs(composed - expected),
            }
        )

    endpoint = evaluate_alpha1_spectral(
        np.asarray([0.0, 0.5, 1.0, 0.2]),
        np.asarray([0.0, 0.0, 1.0, 1.0]),
        np.full(4, 0.4),
        config=config,
    )
    endpoint_error = float(
        np.max(np.abs(endpoint.cdf - np.asarray([0.0, 0.0, 1.0, 1.0])))
    )

    cuda_kernel_error = 0.0
    cuda_target_error = 0.0
    cuda_finite = True
    if device.type == "cuda":
        test_x = torch.as_tensor(xx.reshape(-1), dtype=torch.float64, device=device)
        test_y = torch.as_tensor(yy.reshape(-1), dtype=torch.float64, device=device)
        test_u = torch.full_like(test_x, 0.35)
        torch_eval = evaluate_alpha1_rb_torch_fixed_modes(
            test_x, test_y, test_u, modes=256
        )
        reference_target = (
            yy.reshape(-1)
            * (1.0 - yy.reshape(-1))
            * direct.arrival_score.reshape(-1)
        )
        cuda_kernel_error = max(
            float(np.max(np.abs(torch_eval.density.detach().cpu().numpy() - direct.density.reshape(-1)))),
            float(np.max(np.abs(torch_eval.cdf.detach().cpu().numpy() - direct.cdf.reshape(-1)))),
        )
        cuda_target_error = float(
            np.max(
                np.abs(
                    torch_eval.denoising_target.detach().cpu().numpy()
                    - reference_target
                )
            )
        )
        cuda_finite = bool(
            torch.isfinite(torch_eval.density).all()
            and torch.isfinite(torch_eval.cdf).all()
            and torch.isfinite(torch_eval.denoising_target).all()
        )

    metrics = {
        "control_version": CONTROL_VERSION,
        "normalization_max_error": normalization_error,
        "cdf_endpoint_max_error": endpoint_error,
        "cdf_monotonicity_max_violation": monotonicity_violation,
        "detailed_balance_max_error": detailed_balance_error,
        "reflection_symmetry_max_error": reflection_error,
        "semigroup_max_error": semigroup_error,
        "eigenmoment_1_to_8_max_error": eigenmoment_error,
        "float64_kernel_max_error": max(
            normalization_error,
            endpoint_error,
            monotonicity_violation,
            detailed_balance_error,
            reflection_error,
            semigroup_error,
            eigenmoment_error,
        ),
        "cuda_kernel_max_error": cuda_kernel_error,
        "cuda_target_max_error": cuda_target_error,
        "cuda_finite": int(cuda_finite),
        "cuda_evaluated": int(device.type == "cuda"),
    }
    return ControlPanel(metrics=metrics, rows=rows)


def _normal_interval(values: np.ndarray, confidence_z: float = 3.5) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(data))
    if data.size <= 1:
        return mean, mean
    half = confidence_z * float(np.std(data, ddof=1)) / math.sqrt(data.size)
    return mean - half, mean + half


def _whole_path_max_t_intervals(
    values: np.ndarray,
    path_ids: np.ndarray,
    *,
    seed: int,
    replicates: int = 2_000,
    confidence: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Simultaneous two-sided intervals by whole-path max-T bootstrap."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("values must be a nonempty one- or two-dimensional array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("whole-path values must be finite")
    identifiers = np.asarray(path_ids, dtype=np.int64)
    if identifiers.ndim != 1 or identifiers.shape[0] != matrix.shape[0]:
        raise ValueError("path_ids must contain one identifier per value row")
    if int(replicates) <= 0:
        raise ValueError("replicates must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie in (0,1)")
    unique = np.unique(identifiers)
    if unique.size < 2:
        raise ValueError("whole-path inference requires at least two paths")
    clustered = np.stack(
        [np.mean(matrix[identifiers == path], axis=0) for path in unique], axis=0
    )
    point = np.mean(clustered, axis=0)
    standard_error = np.std(clustered, axis=0, ddof=1) / math.sqrt(unique.size)
    rng = np.random.Generator(np.random.Philox(int(seed)))
    maxima = np.zeros(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        sampled = clustered[rng.integers(0, unique.size, size=unique.size)]
        mean = np.mean(sampled, axis=0)
        statistic = np.zeros_like(point)
        # Studentize by the fixed whole-path standard error.  Re-estimating it
        # inside a bootstrap resample creates artificial infinities whenever a
        # resample happens to contain only one distinct path.  A zero original
        # standard error is legitimate only for a genuinely constant family,
        # in which case every resampled mean is exactly the point estimate.
        nondegenerate = standard_error > 0.0
        statistic[nondegenerate] = np.abs(
            (mean[nondegenerate] - point[nondegenerate])
            / standard_error[nondegenerate]
        )
        inconsistent = (~nondegenerate) & (mean != point)
        if np.any(inconsistent):
            raise ValueError("degenerate whole-path family changed under resampling")
        maxima[replicate] = float(np.max(statistic))
    critical = float(np.quantile(maxima, confidence, method="higher"))
    if not math.isfinite(critical):
        raise ValueError("whole-path max-T critical value is nonfinite")
    lower = point - critical * standard_error
    upper = point + critical * standard_error
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("whole-path max-T interval is nonfinite")
    return lower, upper, critical


def transition_law_controls(
    *,
    count: int,
    root_seed: int,
    profile: JacobiRBSpectralProfile,
) -> ControlPanel:
    """Check inverse-CDF laws, moments, stationarity, and reversibility."""

    sample_count = max(64, int(count))
    config = _spectral_config()
    fixed_x = np.full(sample_count, 0.37)
    fixed_u = np.full(sample_count, 0.5)
    fixed = sample_alpha1_rb_transition_batch(
        fixed_x, fixed_u, rng_key=(root_seed, "law-fixed"), profile=profile
    )
    cdf = evaluate_alpha1_spectral(
        fixed_x, fixed.later_head_fraction, fixed_u, config=config
    ).cdf
    sorted_cdf = np.sort(cdf)
    grid = np.arange(1, sample_count + 1, dtype=np.float64) / sample_count
    ks = float(
        max(
            np.max(grid - sorted_cdf),
            np.max(sorted_cdf - np.arange(sample_count) / sample_count),
        )
    )
    ks_limit = math.sqrt(math.log(2.0 / 0.01) / (2.0 * sample_count))
    moment_covered = True
    moment_rows: list[dict[str, Any]] = []
    y_coordinate = 2.0 * fixed.later_head_fraction - 1.0
    anchors_per_path = min(32, max(8, sample_count // 8))
    path_ids = np.arange(sample_count, dtype=np.int64) // anchors_per_path
    fixed_values = np.stack(
        [_legendre(degree, y_coordinate) for degree in range(1, 9)], axis=1
    )
    fixed_expected = np.asarray(
        [
            math.exp(-degree * (degree + 1) * 0.5)
            * float(_legendre(degree, np.asarray(2.0 * 0.37 - 1.0)))
            for degree in range(1, 9)
        ],
        dtype=np.float64,
    )
    fixed_lower, fixed_upper, fixed_critical = _whole_path_max_t_intervals(
        fixed_values - fixed_expected[None, :],
        path_ids,
        seed=root_seed + 11,
    )
    for degree in range(1, 9):
        values = fixed_values[:, degree - 1]
        expected = float(fixed_expected[degree - 1])
        lower = float(fixed_lower[degree - 1] + expected)
        upper = float(fixed_upper[degree - 1] + expected)
        covered = fixed_lower[degree - 1] <= 0.0 <= fixed_upper[degree - 1]
        moment_covered = moment_covered and covered
        moment_rows.append(
            {
                "control": "sample_eigenmoment",
                "degree": degree,
                "measured": float(np.mean(values)),
                "expected": expected,
                "lower_99_simultaneous": lower,
                "upper_99_simultaneous": upper,
                "covered": int(covered),
                "max_t_critical": fixed_critical,
                "path_count": int(np.unique(path_ids).size),
            }
        )

    rng = np.random.Generator(np.random.Philox(int(root_seed) + 1))
    stationary_x = rng.random(sample_count)
    stationary = sample_alpha1_rb_transition_batch(
        stationary_x,
        np.full(sample_count, 0.5),
        rng_key=(root_seed, "law-stationary"),
        profile=profile,
    )
    stationary_coordinate = 2.0 * stationary.later_head_fraction - 1.0
    stationary_values = np.stack(
        [_legendre(degree, stationary_coordinate) for degree in range(1, 9)],
        axis=1,
    )
    stationary_lower, stationary_upper, stationary_critical = _whole_path_max_t_intervals(
        stationary_values,
        path_ids,
        seed=root_seed + 12,
    )
    stationarity_covered = bool(
        np.all((stationary_lower <= 0.0) & (stationary_upper >= 0.0))
    )
    for degree in range(1, 9):
        lower = float(stationary_lower[degree - 1])
        upper = float(stationary_upper[degree - 1])
        covered = lower <= 0.0 <= upper
        moment_rows.append(
            {
                "control": "stationarity_eigenmoment",
                "degree": degree,
                "measured": float(np.mean(_legendre(degree, stationary_coordinate))),
                "expected": 0.0,
                "lower_99_simultaneous": lower,
                "upper_99_simultaneous": upper,
                "covered": int(covered),
                "max_t_critical": stationary_critical,
                "path_count": int(np.unique(path_ids).size),
            }
        )
    zx = 2.0 * stationary_x - 1.0
    zy = stationary_coordinate
    reverse_statistic = _legendre(1, zx) * _legendre(2, zy) - _legendre(2, zx) * _legendre(1, zy)
    reverse_lower, reverse_upper, reverse_critical = _whole_path_max_t_intervals(
        reverse_statistic,
        path_ids,
        seed=root_seed + 13,
    )
    reversible_interval = (float(reverse_lower[0]), float(reverse_upper[0]))
    moment_rows.append(
        {
            "control": "reversibility_antisymmetric",
            "measured": float(np.mean(reverse_statistic)),
            "expected": 0.0,
            "lower_99_simultaneous": reversible_interval[0],
            "upper_99_simultaneous": reversible_interval[1],
            "max_t_critical": reverse_critical,
            "path_count": int(np.unique(path_ids).size),
            "covered": int(reversible_interval[0] <= 0.0 <= reversible_interval[1]),
        }
    )
    replay = sample_alpha1_rb_transition_batch(
        stationary_x,
        np.full(sample_count, 0.5),
        rng_key=(root_seed, "law-stationary"),
        profile=profile,
    )
    replay_pass = bool(
        np.array_equal(stationary.later_head_fraction, replay.later_head_fraction)
        and np.array_equal(stationary.denoising_target, replay.denoising_target)
        and np.array_equal(stationary.certificate_codes, replay.certificate_codes)
    )
    metrics = {
        "sample_count": sample_count,
        "cdf_ks_statistic": ks,
        "cdf_ks_99_limit": ks_limit,
        "cdf_statistics_pass": int(ks <= ks_limit),
        "sample_eigenmoments_pass": int(moment_covered),
        "stationarity_simultaneous_pass": int(stationarity_covered),
        "reversibility_simultaneous_pass": int(
            reversible_interval[0] <= 0.0 <= reversible_interval[1]
        ),
        "replay_pass": int(replay_pass),
        "uncertified_draw_count": 0,
    }
    return ControlPanel(metrics=metrics, rows=moment_rows)


def _ancestral_probability(m: int, exposure: float, tolerance: float = 1e-17) -> float:
    """Evaluate q_m^2(2u) by its legacy alternating entrance-law series."""

    total = 0.0
    previous = math.inf
    entered_decrease = False
    for k in range(int(m), 4096):
        log_coefficient = (
            math.log(2.0 * k + 1.0)
            + math.lgamma(2.0 + m + k - 1.0)
            - math.lgamma(2.0 + m)
            - math.lgamma(m + 1.0)
            - math.lgamma(k - m + 1.0)
        )
        term = math.exp(log_coefficient - k * (k + 1.0) * exposure)
        total += term if (k - m) % 2 == 0 else -term
        if term < previous:
            entered_decrease = True
        previous = term
        if entered_decrease and term < tolerance:
            break
    return total


def legacy_mixture_rb_target(x: float, y: float, exposure: float) -> float:
    """Posterior E[L-MY|x,y] from the legacy ancestral mixture at tractable u."""

    density = 0.0
    numerator = 0.0
    cumulative_q = 0.0
    for m in range(128):
        q = _ancestral_probability(m, exposure)
        if q < -1e-12:
            raise RuntimeError("legacy mixture produced a negative component")
        q = max(q, 0.0)
        cumulative_q += q
        for l in range(m + 1):
            log_binomial = (
                math.lgamma(m + 1.0)
                - math.lgamma(l + 1.0)
                - math.lgamma(m - l + 1.0)
            )
            if x in (0.0, 1.0):
                probability_l = float((x == 0.0 and l == 0) or (x == 1.0 and l == m))
            else:
                probability_l = math.exp(
                    log_binomial + l * math.log(x) + (m - l) * math.log1p(-x)
                )
            log_beta_density = (
                math.lgamma(m + 2.0)
                - math.lgamma(l + 1.0)
                - math.lgamma(m - l + 1.0)
                + l * math.log(y)
                + (m - l) * math.log1p(-y)
            )
            component = q * probability_l * math.exp(log_beta_density)
            density += component
            numerator += component * (l - m * y)
        if m > 8 and 1.0 - cumulative_q < 1e-14:
            break
    if not density > 0.0 or not math.isfinite(density):
        raise RuntimeError("legacy mixture density is not finite and positive")
    return numerator / density


def _sample_linear_teacher_x(count: int, amplitude: float, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.Philox(seed))
    # Exact rejection sampler from density 1+c(2x-1), bounded by 1+|c|.
    result: list[float] = []
    while len(result) < count:
        proposal = rng.random(max(32, 2 * (count - len(result))))
        accept = rng.random(proposal.size) * (1.0 + abs(amplitude))
        density = 1.0 + amplitude * (2.0 * proposal - 1.0)
        result.extend(proposal[accept <= density].tolist())
    return np.asarray(result[:count], dtype=np.float64)


def target_identity_controls(
    *, count: int, root_seed: int, profile: JacobiRBSpectralProfile
) -> ControlPanel:
    """Check legacy-mixture equivalence and teacher/null tower identities."""

    config = _spectral_config()
    mixture_error = 0.0
    rows: list[dict[str, Any]] = []
    for exposure in (0.75, 1.0):
        for x in (0.2, 0.5, 0.8):
            for y in (0.2, 0.5, 0.8):
                spectral = evaluate_alpha1_spectral(x, y, exposure, config=config)
                exact = y * (1.0 - y) * float(spectral.arrival_score)
                mixture = legacy_mixture_rb_target(x, y, exposure)
                error = abs(exact - mixture)
                mixture_error = max(mixture_error, error)
                rows.append(
                    {
                        "control": "legacy_mixture_equivalence",
                        "x": x,
                        "y": y,
                        "exposure": exposure,
                        "spectral_target": exact,
                        "legacy_mixture_target": mixture,
                        "absolute_error": error,
                    }
                )

    # The frozen panel uses 32 whole paths with the eight exact
    # color-by-duration anchors per path.  Tiny callers may request less work,
    # but a statistical authorization is never based on fewer paths.
    sample_count = max(256, int(count))
    phase_indices = np.arange(sample_count) % 4
    duration = np.where((np.arange(sample_count) // 4) % 2 == 0, 0.5, 1.0)
    exposure = 0.5 * duration
    teacher_x = _sample_linear_teacher_x(sample_count, 0.5, root_seed + 31)
    teacher = sample_alpha1_rb_transition_batch(
        teacher_x, exposure, rng_key=(root_seed, "target-teacher"), profile=profile
    )
    analytic_teacher = linear_teacher_denoising_mean(
        teacher.later_head_fraction, exposure, amplitude=0.5
    )
    rng = np.random.Generator(np.random.Philox(root_seed + 32))
    null_x = rng.random(sample_count)
    null = sample_alpha1_rb_transition_batch(
        null_x, exposure, rng_key=(root_seed, "target-null"), profile=profile
    )
    features_teacher = np.stack(
        [
            np.ones(sample_count),
            2.0 * teacher.later_head_fraction - 1.0,
            _legendre(2, 2.0 * teacher.later_head_fraction - 1.0),
        ],
        axis=1,
    )
    features_null = np.stack(
        [
            np.ones(sample_count),
            2.0 * null.later_head_fraction - 1.0,
            _legendre(2, 2.0 * null.later_head_fraction - 1.0),
        ],
        axis=1,
    )
    teacher_residual = teacher.denoising_target - analytic_teacher
    anchors_per_path = 8
    path_ids = np.arange(sample_count, dtype=np.int64) // anchors_per_path
    teacher_columns = [teacher_residual[:, None] * features_teacher]
    null_columns = [null.denoising_target[:, None] * features_null]
    column_metadata: list[tuple[int | None, float | None, int]] = [
        (None, None, feature_index)
        for feature_index in range(features_teacher.shape[1])
    ]
    # Every path contains all 4 colors x both durations equally often.
    # Multiplying by eight makes each clustered column equal the phase-local
    # mean rather than a zero-padded eighth of it.
    for phase_index in range(4):
        for duration_fraction in (0.5, 1.0):
            mask = (
                (phase_indices == phase_index)
                & (duration == duration_fraction)
            ).astype(np.float64)
            teacher_columns.append(
                8.0 * teacher_residual[:, None] * features_teacher * mask[:, None]
            )
            null_columns.append(
                8.0 * null.denoising_target[:, None] * features_null * mask[:, None]
            )
            column_metadata.extend(
                (phase_index, duration_fraction, feature_index)
                for feature_index in range(features_teacher.shape[1])
            )
    teacher_products = np.concatenate(teacher_columns, axis=1)
    null_products = np.concatenate(null_columns, axis=1)
    teacher_lower, teacher_upper, teacher_critical = _whole_path_max_t_intervals(
        teacher_products,
        path_ids,
        seed=root_seed + 41,
    )
    null_lower, null_upper, null_critical = _whole_path_max_t_intervals(
        null_products,
        path_ids,
        seed=root_seed + 42,
    )
    teacher_component_covered = (teacher_lower <= 0.0) & (teacher_upper >= 0.0)
    null_component_covered = (null_lower <= 0.0) & (null_upper >= 0.0)
    teacher_covered = bool(np.all(teacher_component_covered))
    null_covered = bool(np.all(null_component_covered))
    for feature_index in range(features_teacher.shape[1]):
        teacher_values = teacher_residual * features_teacher[:, feature_index]
        null_values = null.denoising_target * features_null[:, feature_index]
        rows.extend(
            [
                {
                    "control": "teacher_tower_orthogonality",
                    "feature": feature_index,
                    "mean": float(np.mean(teacher_values)),
                    "lower_99_simultaneous": float(teacher_lower[feature_index]),
                    "upper_99_simultaneous": float(teacher_upper[feature_index]),
                    "max_t_critical": teacher_critical,
                    "path_count": int(np.unique(path_ids).size),
                },
                {
                    "control": "stationary_null_orthogonality",
                    "feature": feature_index,
                    "mean": float(np.mean(null_values)),
                    "lower_99_simultaneous": float(null_lower[feature_index]),
                    "upper_99_simultaneous": float(null_upper[feature_index]),
                    "max_t_critical": null_critical,
                    "path_count": int(np.unique(path_ids).size),
                },
            ]
        )

    for column_index, (phase_index, duration_fraction, feature_index) in enumerate(
        column_metadata[features_teacher.shape[1]:],
        start=features_teacher.shape[1],
    ):
        rows.extend(
            [
                {
                    "control": "teacher_phase_duration_orthogonality",
                    "phase_index": phase_index,
                    "duration_fraction": duration_fraction,
                    "feature": feature_index,
                    "mean": float(np.mean(teacher_products[:, column_index])),
                    "lower_99_simultaneous": float(teacher_lower[column_index]),
                    "upper_99_simultaneous": float(teacher_upper[column_index]),
                    "max_t_critical": teacher_critical,
                    "path_count": int(np.unique(path_ids).size),
                    "covered": int(teacher_component_covered[column_index]),
                },
                {
                    "control": "null_phase_duration_orthogonality",
                    "phase_index": phase_index,
                    "duration_fraction": duration_fraction,
                    "feature": feature_index,
                    "mean": float(np.mean(null_products[:, column_index])),
                    "lower_99_simultaneous": float(null_lower[column_index]),
                    "upper_99_simultaneous": float(null_upper[column_index]),
                    "max_t_critical": null_critical,
                    "path_count": int(np.unique(path_ids).size),
                    "covered": int(null_component_covered[column_index]),
                },
            ]
        )

    h = 1.0 / 28.0
    flux = denoising_mean_to_mass_flux(analytic_teacher, grid_spacing=h)
    flux_error = float(np.max(np.abs(flux - 6.0 * analytic_teacher / (h * h))))
    # Negative fixtures must disagree decisively with the approved convention.
    scale = max(float(np.max(np.abs(flux))), np.finfo(np.float64).tiny)
    orientation_fixture_error = float(np.max(np.abs(flux + 6.0 * analytic_teacher / (h * h)))) / scale
    h_fixture_error = float(np.max(np.abs(flux - 6.0 * analytic_teacher))) / scale
    raw_score = analytic_teacher / (
        teacher.later_head_fraction * (1.0 - teacher.later_head_fraction)
    )
    invariant_score_fixture_error = float(
        np.linalg.norm(raw_score - analytic_teacher)
        / max(np.linalg.norm(analytic_teacher), np.finfo(np.float64).tiny)
    )
    wrong_pair_mass_teacher = linear_teacher_denoising_mean(
        teacher.later_head_fraction,
        exposure * 0.25,
        amplitude=0.5,
    )
    pair_mass_fixture_error = float(
        np.linalg.norm(wrong_pair_mass_teacher - analytic_teacher)
        / max(np.linalg.norm(analytic_teacher), np.finfo(np.float64).tiny)
    )
    metrics = {
        "legacy_mixture_max_absolute_error": mixture_error,
        "teacher_tower_simultaneous_pass": int(teacher_covered),
        "stationary_null_simultaneous_pass": int(null_covered),
        "all_phase_colors_pass": int(
            set(phase_indices.tolist()) == {0, 1, 2, 3}
            and teacher_covered
            and null_covered
        ),
        "half_full_duration_pass": int(
            set(duration.tolist()) == {0.5, 1.0}
            and teacher_covered
            and null_covered
        ),
        "phase_duration_simultaneous_family_size": int(teacher_products.shape[1]),
        "flux_conversion_max_error": flux_error,
        "orientation_negative_fixture_pass": int(orientation_fixture_error > 1.0),
        "pair_mass_negative_fixture_pass": int(pair_mass_fixture_error > 1e-3),
        "h_scaling_negative_fixture_pass": int(h_fixture_error > 0.5),
        "invariant_beta_score_negative_fixture_pass": int(invariant_score_fixture_error > 0.5),
        "flux_sign_negative_fixture_pass": int(orientation_fixture_error > 1.0),
        "teacher_nonfinite_count": int(np.count_nonzero(~np.isfinite(teacher.denoising_target))),
        "null_nonfinite_count": int(np.count_nonzero(~np.isfinite(null.denoising_target))),
    }
    return ControlPanel(metrics=metrics, rows=rows)


def output_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


__all__ = [
    "CONTROL_VERSION",
    "ControlPanel",
    "deterministic_kernel_controls",
    "legacy_mixture_rb_target",
    "output_hash",
    "target_identity_controls",
    "transition_law_controls",
]
