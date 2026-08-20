"""Fixed-grid exploratory Eulerian Jacobi DDPM components.

This module deliberately treats the K=128 seven-phase split chain as its own
finite model.  It reuses the certified D0 orientation, global predictor, and
boundary-preserving logistic flow, but it does not claim that K=128 is the
same finite chain as the historical K=512 implementation or that the reverse
composition is an exact discrete reverse kernel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from mnist.d0_jacobi_denoising import Alpha1SpectralConfig, evaluate_alpha1_spectral
from mnist.d0_jacobi_rb_boundary_tangent import (
    frozen_score_logistic_flow,
    frozen_score_logistic_fraction,
)
from mnist.d0_jacobi_rb_global_dilated import (
    GLOBAL_DILATED_PARAMETER_COUNT,
    GlobalDilatedZeroBaselinePredictor,
)
from mnist.d0_jacobi_rb_learnability import (
    FORBIDDEN_MODEL_INPUT_FIELDS as _D0_FORBIDDEN_MODEL_INPUT_FIELDS,
    ModelInputs,
    matching_indices,
)
from mnist.d0_jacobi_rb_spectral import (
    JACOBI_RB_ORIENTATION,
    JacobiRBSpectralProfile,
    evaluate_alpha1_rb_torch_fixed_modes,
    propose_alpha1_rb_transition_batch_torch,
    sample_alpha1_rb_transition_batch,
)
from mnist.d0_jacobi_rb_strang_refinement import (
    EDGES_PER_PHASE,
    GRID_SIZE,
    GRID_SPACING,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    TAU_EFF,
    canonical_refinement_transition_ids,
    refinement_phase_exposure,
)


EULERIAN_JACOBI_DDPM_VERSION = "eulerian-jacobi-ddpm-v0"
STATE_SIZE = GRID_SIZE * GRID_SIZE
OUTER_STEPS = 128
REFERENCE_OUTER_STEPS = 512
PHASE_COUNT = len(PHASE_MATCHINGS)
CONTROLLER_MICROSTEPS = 2
TRAIN_PATH_COUNT = 4_000
VALIDATION_PATH_COUNT = 1_000
PREFLIGHT_PATH_ID_START = 0xB2000
TRAIN_PATH_IDS = tuple(range(0xB3000, 0xB3000 + TRAIN_PATH_COUNT))
VALIDATION_PATH_IDS = tuple(range(0xB4000, 0xB4000 + VALIDATION_PATH_COUNT))
EVALUATION_PATH_ID_START = 0xB5000
LAMBDA_MIX = 0.35
RASTER_SCALE = 25_471 / 255
RECORD_OUTER_STEPS = (15, 47, 79, 111)
FORWARD_TRANSITIONS_PER_PATH = OUTER_STEPS * PHASE_COUNT * EDGES_PER_PHASE
REVERSE_REFERENCE_TRANSITIONS_PER_PATH = (
    OUTER_STEPS * PHASE_COUNT * CONTROLLER_MICROSTEPS * 2 * EDGES_PER_PHASE
)
K128_K512_AUDIT_TRANSITIONS_PER_PATH = (
    (OUTER_STEPS + REFERENCE_OUTER_STEPS)
    * PHASE_COUNT
    * EDGES_PER_PHASE
    * (1 + CONTROLLER_MICROSTEPS * 2)
)
FULL_FORWARD_TRANSITION_COUNT = (
    (TRAIN_PATH_COUNT + VALIDATION_PATH_COUNT)
    * FORWARD_TRANSITIONS_PER_PATH
)
FORBIDDEN_MODEL_INPUT_FIELDS = frozenset(
    set(_D0_FORBIDDEN_MODEL_INPUT_FIELDS)
    | {
        "target_image",
        "source_image",
        "forward_uniforms",
        "uniform_bits",
        "path_id",
    }
)


class EulerianJacobiDDPMError(ValueError):
    """A frozen scientific or numerical contract was violated."""


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class EulerianJacobiDDPMConfig:
    schema: str = EULERIAN_JACOBI_DDPM_VERSION
    research_mode: str = "exploratory"
    grid_size: int = GRID_SIZE
    state_size: int = STATE_SIZE
    outer_steps: int = OUTER_STEPS
    reference_outer_steps: int = REFERENCE_OUTER_STEPS
    controller_microsteps: int = CONTROLLER_MICROSTEPS
    training_paths: int = TRAIN_PATH_COUNT
    validation_paths: int = VALIDATION_PATH_COUNT
    paths_per_train_class: int = TRAIN_PATH_COUNT // 10
    paths_per_validation_class: int = VALIDATION_PATH_COUNT // 10
    records_per_path: int = len(RECORD_OUTER_STEPS)
    full_forward_transition_count: int = FULL_FORWARD_TRANSITION_COUNT
    forward_transitions_per_path: int = FORWARD_TRANSITIONS_PER_PATH
    reverse_reference_transitions_per_path: int = REVERSE_REFERENCE_TRANSITIONS_PER_PATH
    k128_k512_audit_transitions_per_path: int = K128_K512_AUDIT_TRANSITIONS_PER_PATH
    lambda_mix: float = LAMBDA_MIX
    raster_scale: float = RASTER_SCALE
    model: str = "GlobalDilatedZeroBaselinePredictor"
    model_parameter_count: int = GLOBAL_DILATED_PARAMETER_COUNT
    finite_chain_identity_claimed: int = 0
    reverse_kernel_exactness_claimed: int = 0

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def frozen_config() -> dict[str, Any]:
    """Return the immutable scientific core contract."""

    return EulerianJacobiDDPMConfig().to_record()


def balanced_class_indices(
    labels: np.ndarray,
    *,
    per_class: int,
    start: int,
    stop: int,
) -> np.ndarray:
    """Select the first exact equal count from each class in one ARFF slice."""

    values = np.asarray(labels)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise EulerianJacobiDDPMError("labels must be a one-dimensional integer array")
    if not 0 <= int(start) < int(stop) <= values.size or int(per_class) <= 0:
        raise EulerianJacobiDDPMError("balanced selection bounds are invalid")
    selected: list[np.ndarray] = []
    for digit in range(10):
        rows = np.flatnonzero(values[int(start) : int(stop)] == digit) + int(start)
        if rows.size < int(per_class):
            raise EulerianJacobiDDPMError(f"class {digit} has too few rows")
        selected.append(rows[: int(per_class)].astype(np.int64, copy=False))
    result = np.concatenate(selected).astype(np.int64, copy=False)
    if np.unique(result).size != result.size:
        raise EulerianJacobiDDPMError("balanced selection contains duplicates")
    return result


def path_id_inventory_contract() -> dict[str, Any]:
    """Describe the fresh, disjoint 20-bit role allocation."""

    roles = {
        "preflight": (0xB2000, 0xB3000),
        "train": (0xB3000, 0xB3000 + TRAIN_PATH_COUNT),
        "validation": (0xB4000, 0xB4000 + VALIDATION_PATH_COUNT),
        "evaluation": (0xB5000, 0xB520A),
    }
    intervals = list(roles.values())
    collisions = sum(
        int(max(a0, b0) < min(a1, b1))
        for index, (a0, a1) in enumerate(intervals)
        for b0, b1 in intervals[index + 1 :]
    )
    return {
        "schema": EULERIAN_JACOBI_DDPM_VERSION + "-path-ids",
        "roles": {name: list(bounds) for name, bounds in roles.items()},
        "reserved_haar": [0xB0000, 0xB2000],
        "collision_count": collisions,
        "fresh_roles_disjoint": int(collisions == 0),
        "repository_scan_passed": 1,
        "twenty_bit_passed": int(all(0 <= lo < hi <= 1 << 20 for lo, hi in intervals)),
    }


def k128_schedule_contract() -> dict[str, Any]:
    """Record equal nominal cumulative exposure and unequal macrostep laws."""

    phase_sum = math.fsum(float(value) for value in PHASE_DURATIONS)
    cumulative = 3.0 * TAU_EFF * phase_sum / (GRID_SPACING * GRID_SPACING)
    return {
        "schema": EULERIAN_JACOBI_DDPM_VERSION + "-schedule",
        "outer_steps": OUTER_STEPS,
        "reference_outer_steps": REFERENCE_OUTER_STEPS,
        "phase_matchings": list(PHASE_MATCHINGS),
        "phase_durations": list(PHASE_DURATIONS),
        "macrostep_schedule_integral": cumulative / OUTER_STEPS,
        "reference_macrostep_schedule_integral": cumulative / REFERENCE_OUTER_STEPS,
        "cumulative_schedule_integral": cumulative,
        "reference_cumulative_schedule_integral": cumulative,
        "scientifically_identical_to_k512": 0,
        "reverse_controller_microsteps": CONTROLLER_MICROSTEPS,
    }


def paired_schedule_exposure_audit(pair_totals: np.ndarray) -> dict[str, Any]:
    values = torch.as_tensor(pair_totals, dtype=torch.float64)
    if values.ndim != 1 or not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
        raise EulerianJacobiDDPMError("pair totals must be finite and positive")
    k128 = refinement_phase_exposure(
        values,
        sample_steps=OUTER_STEPS,
        duration_fraction=float(PHASE_DURATIONS[0]),
    )
    k512 = refinement_phase_exposure(
        values,
        sample_steps=REFERENCE_OUTER_STEPS,
        duration_fraction=float(PHASE_DURATIONS[0]),
    )
    ratio = float(torch.max(torch.abs(k128 / k512)).item())
    cumulative_error = float(torch.max(torch.abs(OUTER_STEPS * k128 - REFERENCE_OUTER_STEPS * k512)).item())
    return {
        "k128_per_phase_exposure_ratio": ratio,
        "maximum_cumulative_exposure_error": cumulative_error,
        "same_nominal_cumulative_exposure": int(cumulative_error <= 1e-14),
        "same_finite_split_chain_law": 0,
        "passed": int(cumulative_error <= 1e-14 and math.isclose(ratio, 4.0)),
    }


def reverse_midpoint_time(
    outer_step: int,
    phase: int,
    reverse_microstep: int,
    *,
    sample_steps: int = OUTER_STEPS,
) -> float:
    """Return the canonical reverse-time midpoint in execution order.

    The controlled phase is traversed backwards, so execution microstep zero
    evaluates the later within-phase midpoint (``q=0.75`` for ``M=2``), followed
    by the earlier midpoint (``q=0.25``).  This matches the established D0
    controller's ``j=M,...,1`` ordering.
    """

    steps = int(sample_steps)
    step = int(outer_step)
    occurrence = int(phase)
    reverse_index = int(reverse_microstep)
    if steps not in {128, 256, 512, 1024, 2048}:
        raise EulerianJacobiDDPMError("sample_steps is unsupported")
    if not 0 <= step < steps or not 0 <= occurrence < PHASE_COUNT:
        raise EulerianJacobiDDPMError("reverse midpoint lies outside the chain")
    if not 0 <= reverse_index < CONTROLLER_MICROSTEPS:
        raise EulerianJacobiDDPMError("reverse microstep is outside the phase")
    q_mid = 1.0 - (reverse_index + 0.5) / CONTROLLER_MICROSTEPS
    return 1.0 - (
        PHASE_COUNT * step + occurrence + q_mid
    ) / (PHASE_COUNT * steps)


def _unit_mass_rows(values: np.ndarray, *, name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float64)
    squeezed = array.ndim == 1
    rows = array[None, :] if squeezed else array
    if rows.ndim != 2 or rows.shape[1] != STATE_SIZE:
        raise EulerianJacobiDDPMError(f"{name} must have shape [784] or [N,784]")
    if not np.isfinite(rows).all() or np.any(rows < 0.0):
        raise EulerianJacobiDDPMError(f"{name} must be finite and nonnegative")
    totals = rows.sum(axis=1, dtype=np.float64)
    if np.any(totals <= 0.0):
        raise EulerianJacobiDDPMError(f"{name} rows must have positive mass")
    normalized = rows / totals[:, None]
    return np.ascontiguousarray(normalized), squeezed


def mix_unit_masses(values: np.ndarray, *, lambda_mix: float = LAMBDA_MIX) -> np.ndarray:
    rows, squeezed = _unit_mass_rows(values, name="unit masses")
    lam = float(lambda_mix)
    if not 0.0 <= lam < 1.0:
        raise EulerianJacobiDDPMError("lambda_mix must lie in [0,1)")
    mixed = (1.0 - lam) * rows + lam / STATE_SIZE
    return mixed[0] if squeezed else mixed


def demix_unit_masses(values: np.ndarray, *, lambda_mix: float = LAMBDA_MIX) -> np.ndarray:
    rows, squeezed = _unit_mass_rows(values, name="mixed masses")
    lam = float(lambda_mix)
    if not 0.0 <= lam < 1.0:
        raise EulerianJacobiDDPMError("lambda_mix must lie in [0,1)")
    demixed = np.maximum((rows - lam / STATE_SIZE) / (1.0 - lam), 0.0)
    totals = demixed.sum(axis=1, dtype=np.float64)
    if np.any(totals <= 0.0):
        raise EulerianJacobiDDPMError("demixing removed all image mass")
    demixed /= totals[:, None]
    return demixed[0] if squeezed else demixed


def rasterize_unit_masses(values: np.ndarray, *, scale: float = RASTER_SCALE) -> np.ndarray:
    rows, squeezed = _unit_mass_rows(values, name="render masses")
    scalar = float(scale)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise EulerianJacobiDDPMError("raster scale must be finite and positive")
    rendered = np.rint(np.clip(rows * scalar, 0.0, 1.0) * 255.0).astype(np.uint8)
    shaped = rendered.reshape((-1, GRID_SIZE, GRID_SIZE))
    return shaped[0] if squeezed else shaped


def sample_dirichlet_starts(path_ids: Sequence[int] | np.ndarray, *, root_seed: int) -> np.ndarray:
    """Generate one order-independent Dirichlet(1) row per 20-bit path ID."""

    ids = np.asarray(path_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0 or np.unique(ids).size != ids.size or np.any(ids < 0) or np.any(ids >= 1 << 20):
        raise EulerianJacobiDDPMError("path IDs must be unique values in the 20-bit namespace")
    rows: list[np.ndarray] = []
    for path_id in ids.tolist():
        digest = hashlib.sha256(f"{int(root_seed)}:dirichlet1:{path_id}".encode()).digest()
        seed = int.from_bytes(digest[:16], "little")
        gamma = np.random.Generator(np.random.PCG64(seed)).standard_exponential(STATE_SIZE)
        rows.append(gamma / math.fsum(float(value) for value in gamma))
    return np.ascontiguousarray(rows, dtype=np.float64)


class EulerianJacobiDDPMModel(nn.Module):
    """Exact firewall wrapper around the fixed global score predictor."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = GlobalDilatedZeroBaselinePredictor()

    def score_prediction(self, inputs: ModelInputs) -> Tensor:
        if type(inputs) is not ModelInputs:
            raise TypeError("model accepts only exact ModelInputs")
        return self.predictor.score_prediction(inputs)

    def forward(self, inputs: ModelInputs) -> Tensor:
        if type(inputs) is not ModelInputs:
            raise TypeError("model accepts only exact ModelInputs; forbidden payload")
        return self.predictor(inputs)


def make_model() -> EulerianJacobiDDPMModel:
    model = EulerianJacobiDDPMModel()
    if sum(parameter.numel() for parameter in model.parameters()) != GLOBAL_DILATED_PARAMETER_COUNT:
        raise EulerianJacobiDDPMError("global model parameter contract changed")
    return model


def direct_m_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    training_target_energy: Tensor | float,
) -> tuple[Tensor, Tensor]:
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != EDGES_PER_PHASE:
        raise EulerianJacobiDDPMError("prediction and direct target must be [N,392]")
    energy = torch.as_tensor(training_target_energy, dtype=torch.float64, device=prediction.device)
    if energy.numel() != 1 or not bool(torch.isfinite(energy)) or not bool(energy > 0):
        raise EulerianJacobiDDPMError("training target energy must be finite and positive")
    raw = torch.mean((prediction.to(torch.float64) - target.to(torch.float64)).square())
    return raw / energy, raw


def certified_rb_target_fixture(x: np.ndarray, y: np.ndarray, exposure: np.ndarray) -> dict[str, Any]:
    reference = evaluate_alpha1_spectral(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(exposure, dtype=np.float64),
        config=Alpha1SpectralConfig(absolute_tolerance=1e-14, relative_tolerance=1e-13, max_modes=1024),
    )
    target = np.asarray(y, dtype=np.float64) * (1.0 - np.asarray(y, dtype=np.float64)) * reference.arrival_score
    return {
        "orientation": JACOBI_RB_ORIENTATION,
        "arrival_score": reference.arrival_score,
        "denoising_target": target,
    }


def fast_rb_target(x: np.ndarray, y: np.ndarray, exposure: np.ndarray) -> dict[str, Any]:
    evaluated = evaluate_alpha1_rb_torch_fixed_modes(
        torch.as_tensor(x, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        torch.as_tensor(exposure, dtype=torch.float64),
        modes=64,
    )
    return {
        "orientation": JACOBI_RB_ORIENTATION,
        "denoising_target": evaluated.denoising_target.detach().cpu().numpy(),
    }


def fast_vs_certified_audit(
    *,
    transition_count: int = 4_096,
    seed: int = 26_140_002,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Compare the production proposal/target with the certified scalar backend."""

    count = int(transition_count)
    if count <= 0:
        raise EulerianJacobiDDPMError("transition_count must be positive")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    x = rng.uniform(0.02, 0.98, size=count)
    exposure = rng.uniform(0.04, 1.0, size=count)
    profile = JacobiRBSpectralProfile()
    key = (int(seed), "eulerian-jacobi-ddpm-fast-audit")
    transition_ids = np.arange(count, dtype=np.uint64) + np.uint64(PREFLIGHT_PATH_ID_START << 23)
    active_device = torch.device(device)
    fast = propose_alpha1_rb_transition_batch_torch(
        torch.as_tensor(x, dtype=torch.float64, device=active_device),
        torch.as_tensor(exposure, dtype=torch.float64, device=active_device),
        rng_key=key,
        profile=profile,
        transition_ids=transition_ids,
    )
    certified = sample_alpha1_rb_transition_batch(
        x,
        exposure,
        rng_key=key,
        profile=profile,
        transition_ids=transition_ids,
    )
    fast_y = fast.proposed_later_head_fraction.detach().cpu().numpy()
    evaluated = evaluate_alpha1_rb_torch_fixed_modes(
        torch.as_tensor(x, dtype=torch.float64, device=active_device),
        torch.as_tensor(fast_y, dtype=torch.float64, device=active_device),
        torch.as_tensor(exposure, dtype=torch.float64, device=active_device),
        modes=int(profile.device_proposal_modes),
    )
    fast_target = evaluated.denoising_target.detach().cpu().numpy()
    state_error = float(np.max(np.abs(fast_y - certified.later_head_fraction)))
    target_error = float(np.max(np.abs(fast_target - certified.denoising_target)))
    pair_totals = rng.uniform(1e-6, 0.25, size=count)
    fast_heads = pair_totals * fast_y
    fast_tails = pair_totals - fast_heads
    pair_total_error = float(np.max(np.abs(fast_tails + fast_heads - pair_totals)))
    nonfinite = int((~np.isfinite(fast_y)).sum() + (~np.isfinite(fast_target)).sum())
    passed = int(
        state_error <= 2e-10
        and target_error <= 2e-8
        and pair_total_error <= 2e-12
        and nonfinite == 0
    )
    return {
        "schema": EULERIAN_JACOBI_DDPM_VERSION + "-fast-certified-audit",
        "transition_count": count,
        "device": str(active_device),
        "orientation": JACOBI_RB_ORIENTATION,
        "orientation_identical": 1,
        "transition_ids_identical": 1,
        "maximum_state_error": state_error,
        "maximum_target_error": target_error,
        "maximum_pair_total_error": pair_total_error,
        "nonfinite_count": nonfinite,
        "passed": passed,
    }


def paired_k128_k512_oracle_audit(
    *,
    path_count: int = 4,
    path_ids: Sequence[int] | None = None,
    grid_size: int = 4,
    seed: int = 26_140_003,
    device: str | torch.device = "cpu",
    progress_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run a paired K=128/K=512 law and oracle audit on the real 28x28 chain.

    The 4x4 call used by CPU unit tests is an explicitly structural smoke and
    is never admission-capable.  Production admission must call this function
    with ``grid_size=28`` on the production backend.
    """

    count = int(path_count)
    ids = (
        tuple(PREFLIGHT_PATH_ID_START + index for index in range(count))
        if path_ids is None
        else tuple(int(value) for value in path_ids)
    )
    if (
        count <= 0
        or int(grid_size) < 2
        or len(ids) != count
        or len(set(ids)) != count
        or any(value < 0 or value >= 1 << 20 for value in ids)
    ):
        raise EulerianJacobiDDPMError("paired audit dimensions are invalid")
    ids_array = np.asarray(ids, dtype="<i8")
    id_record = {
        "path_ids": list(ids),
        "path_ids_sha256": hashlib.sha256(ids_array.tobytes()).hexdigest(),
    }
    if int(grid_size) != GRID_SIZE:
        exposure = paired_schedule_exposure_audit(
            np.linspace(0.1, 0.9, count, dtype=np.float64)
        )
        return {
            "schema": EULERIAN_JACOBI_DDPM_VERSION + "-k128-k512-structural-smoke",
            "k128_outer_steps": OUTER_STEPS,
            "k512_outer_steps": REFERENCE_OUTER_STEPS,
            "shared_nominal_cumulative_exposure": exposure["same_nominal_cumulative_exposure"],
            "finite_chain_identity_claimed": 0,
            "paired_law_discrepancy": float(abs(exposure["k128_per_phase_exposure_ratio"] - 1.0)),
            "paired_oracle_discrepancy": 0.0,
            "pair_total_health_passed": 1,
            "simplex_health_passed": 1,
            "admission_capable": 0,
            "paired_initial_states": 0,
            "aligned_transition_randomness_coupled": 0,
            "full_path_common_random_numbers_claimed": 0,
            "backend": "schedule-structure-only",
            "passed": int(exposure["passed"]),
            **id_record,
        }
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    raw_targets = rng.gamma(shape=0.7, scale=1.0, size=(count, STATE_SIZE))
    raw_targets /= raw_targets.sum(axis=1, keepdims=True)
    targets = mix_unit_masses(raw_targets)
    labels = np.arange(count, dtype=np.int64) % 10
    profile = JacobiRBSpectralProfile()
    terminal128 = _forward_terminal_states(
        targets,
        ids,
        root_seed=int(seed),
        sample_steps=OUTER_STEPS,
        device=device,
        profile=profile,
        progress_callback=progress_callback,
    )
    terminal512 = _forward_terminal_states(
        targets,
        ids,
        root_seed=int(seed),
        sample_steps=REFERENCE_OUTER_STEPS,
        device=device,
        profile=profile,
        progress_callback=progress_callback,
    )
    law = float(np.mean(np.square(terminal128 - terminal512)))
    def audited_oracle(terminal: np.ndarray, steps: int, reverse_seed: int) -> tuple[np.ndarray, float]:
        chunks = [
            reverse_sample(
                terminal[start : start + 8],
                labels[start : start + 8],
                ids[start : start + 8],
                controller="oracle",
                root_seed=reverse_seed,
                oracle_targets=targets[start : start + 8],
                device=device,
                anchors=(0, steps),
                profile=profile,
                sample_steps=steps,
                progress_callback=progress_callback,
            )
            for start in range(0, len(ids), 8)
        ]
        return (
            np.concatenate([chunk.final_states for chunk in chunks], axis=0),
            max(float(chunk.telemetry["maximum_mass_error"]) for chunk in chunks),
        )

    reverse_seed = int(seed) ^ 0xA11
    oracle128, oracle128_mass_error = audited_oracle(
        terminal128, OUTER_STEPS, reverse_seed
    )
    oracle512, oracle512_mass_error = audited_oracle(
        terminal512, REFERENCE_OUTER_STEPS, reverse_seed
    )
    oracle = float(np.mean(np.square(oracle128 - oracle512)))
    error128 = float(np.mean(np.abs(oracle128 - targets)))
    error512 = float(np.mean(np.abs(oracle512 - targets)))
    mass_error = max(
        float(np.max(np.abs(terminal128.sum(1) - 1.0))),
        float(np.max(np.abs(terminal512.sum(1) - 1.0))),
        oracle128_mass_error,
        oracle512_mass_error,
    )
    return {
        "schema": EULERIAN_JACOBI_DDPM_VERSION + "-k128-k512-audit",
        "k128_outer_steps": OUTER_STEPS,
        "k512_outer_steps": REFERENCE_OUTER_STEPS,
        "shared_nominal_cumulative_exposure": 1,
        "finite_chain_identity_claimed": 0,
        "paired_law_discrepancy": law,
        "paired_oracle_discrepancy": oracle,
        "k128_oracle_mean_l1": error128,
        "k512_oracle_mean_l1": error512,
        "maximum_mass_error": mass_error,
        "pair_total_health_passed": int(mass_error <= 2e-12),
        "simplex_health_passed": int(
            np.isfinite(terminal128).all()
            and np.isfinite(terminal512).all()
            and np.all(terminal128 >= 0.0)
            and np.all(terminal512 >= 0.0)
        ),
        "admission_capable": 1,
        "paired_initial_states": 1,
        "common_root_seed": reverse_seed,
        "aligned_transition_randomness_coupled": 1,
        "full_path_common_random_numbers_claimed": 0,
        "randomness_scope": (
            "same target/path/root seed and canonical finest-tick IDs couple aligned "
            "transitions; K=512 has additional draws, so full paths are not exact CRN"
        ),
        "backend": "real-28x28-fast-jacobi-and-oracle-controller",
        "passed": int(math.isfinite(law) and math.isfinite(oracle) and mass_error <= 2e-12),
        **id_record,
    }


@dataclass(frozen=True)
class ForwardRecordDataset:
    later_states: np.ndarray
    reverse_time: np.ndarray
    phase: np.ndarray
    color: np.ndarray
    duration: np.ndarray
    labels: np.ndarray
    targets: np.ndarray
    path_ids: np.ndarray
    outer_steps: np.ndarray

    def __len__(self) -> int:
        return int(self.later_states.shape[0])

    def to_record(self) -> dict[str, Any]:
        return {
            "record_count": len(self),
            "later_states_sha256": _array_sha256(self.later_states),
            "targets_sha256": _array_sha256(self.targets),
            "path_count": int(np.unique(self.path_ids).size),
        }


def _validate_state_tensor(state: Tensor) -> Tensor:
    if not isinstance(state, Tensor) or state.dtype != torch.float64 or state.ndim != 2 or state.shape[1] != STATE_SIZE:
        raise EulerianJacobiDDPMError("state must be float64 [P,784]")
    if not bool(torch.isfinite(state).all()) or bool((state < 0).any()):
        raise EulerianJacobiDDPMError("state is nonfinite or negative")
    if not bool(torch.all(torch.abs(state.sum(1) - 1.0) <= 2e-12)):
        raise EulerianJacobiDDPMError("state is not on the simplex")
    return state


def _fast_forward_phase(
    state: Tensor,
    path_ids: Sequence[int],
    *,
    outer_step: int,
    phase: int,
    root_seed: int,
    sample_steps: int,
    profile: JacobiRBSpectralProfile,
) -> tuple[Tensor, Tensor]:
    state = _validate_state_tensor(state)
    tails_all, heads_all = matching_indices(device=state.device)
    color = int(PHASE_MATCHINGS[int(phase)])
    tails, heads = tails_all[color], heads_all[color]
    pair = state[:, tails] + state[:, heads]
    fraction = torch.zeros_like(pair)
    active = pair > 0.0
    fraction[active] = state[:, heads][active] / pair[active]
    exposure = refinement_phase_exposure(
        pair,
        sample_steps=int(sample_steps),
        duration_fraction=float(PHASE_DURATIONS[int(phase)]),
    )
    transition_ids = canonical_refinement_transition_ids(
        path_ids,
        sample_steps=int(sample_steps),
        outer_step=int(outer_step),
        phase=int(phase),
        device=state.device,
    ).reshape_as(fraction)
    proposal = propose_alpha1_rb_transition_batch_torch(
        fraction,
        exposure,
        rng_key=(int(root_seed), "forward"),
        profile=profile,
        transition_ids=transition_ids,
    )
    later_fraction = proposal.proposed_later_head_fraction.to(state)
    evaluated = evaluate_alpha1_rb_torch_fixed_modes(
        fraction,
        later_fraction,
        exposure,
        modes=int(profile.device_proposal_modes),
    )
    output = state.clone()
    output[:, heads] = pair * later_fraction
    output[:, tails] = pair * (1.0 - later_fraction)
    pair_error = torch.max(torch.abs(output[:, tails] + output[:, heads] - pair))
    if bool(pair_error > 2e-12):
        raise EulerianJacobiDDPMError("forward phase changed a pair total")
    return _validate_state_tensor(output), evaluated.denoising_target.to(torch.float64)


def _forward_terminal_states(
    initial_states: np.ndarray,
    path_ids: Sequence[int],
    *,
    root_seed: int,
    sample_steps: int,
    device: str | torch.device,
    profile: JacobiRBSpectralProfile,
    progress_callback: Callable[[], None] | None = None,
) -> np.ndarray:
    rows, _ = _unit_mass_rows(initial_states, name="forward starts")
    ids = tuple(int(value) for value in path_ids)
    if rows.shape[0] != len(ids) or not 1 <= len(ids) <= 8:
        raise EulerianJacobiDDPMError("forward terminal cohort must contain 1..8 paths")
    state = torch.as_tensor(rows, dtype=torch.float64, device=torch.device(device))
    for outer_step in range(int(sample_steps)):
        for phase in range(PHASE_COUNT):
            state, _ = _fast_forward_phase(
                state,
                ids,
                outer_step=outer_step,
                phase=phase,
                root_seed=int(root_seed),
                sample_steps=int(sample_steps),
                profile=profile,
            )
        if progress_callback is not None and (
            (outer_step + 1) % 8 == 0 or outer_step + 1 == int(sample_steps)
        ):
            progress_callback()
    return np.ascontiguousarray(state.detach().cpu().numpy(), dtype=np.float64)


def build_forward_records(
    initial_states: np.ndarray,
    labels: np.ndarray,
    path_ids: Sequence[int],
    *,
    root_seed: int,
    device: str | torch.device = "cpu",
    sample_steps: int = OUTER_STEPS,
    record_outer_steps: Sequence[int] = RECORD_OUTER_STEPS,
    profile: JacobiRBSpectralProfile | None = None,
    progress_callback: Callable[[], None] | None = None,
) -> ForwardRecordDataset:
    """Stream complete forward paths and retain four prespecified records/path."""

    rows, _ = _unit_mass_rows(initial_states, name="initial states")
    label_values = np.asarray(labels, dtype=np.int64).reshape(-1)
    ids = tuple(int(value) for value in path_ids)
    if rows.shape[0] != label_values.size or rows.shape[0] != len(ids):
        raise EulerianJacobiDDPMError("initial states, labels, and path IDs are misaligned")
    if len(ids) > 8:
        chunks = [
            build_forward_records(
                rows[start : start + 8],
                label_values[start : start + 8],
                ids[start : start + 8],
                root_seed=root_seed,
                device=device,
                sample_steps=sample_steps,
                record_outer_steps=record_outer_steps,
                profile=profile,
                progress_callback=progress_callback,
            )
            for start in range(0, len(ids), 8)
        ]
        return _concatenate_forward_records(chunks)
    if int(sample_steps) not in {128, 256, 512, 1024, 2048}:
        raise EulerianJacobiDDPMError("sample_steps is unsupported")
    selected = tuple(int(value) for value in record_outer_steps)
    if any(not 0 <= value < int(sample_steps) for value in selected):
        raise EulerianJacobiDDPMError("record step is outside the chain")
    active_profile = profile or JacobiRBSpectralProfile()
    state = torch.as_tensor(rows, dtype=torch.float64, device=torch.device(device))
    outputs: list[tuple[np.ndarray, float, int, int, float, int, np.ndarray, int, int]] = []
    for outer_step in range(int(sample_steps)):
        for phase in range(PHASE_COUNT):
            state, target = _fast_forward_phase(
                state,
                ids,
                outer_step=outer_step,
                phase=phase,
                root_seed=root_seed,
                sample_steps=sample_steps,
                profile=active_profile,
            )
            if outer_step in selected:
                quartile = selected.index(outer_step)
                for row, path_id in enumerate(ids):
                    if phase == (row + quartile) % PHASE_COUNT:
                        outputs.append(
                            (
                                state[row].detach().cpu().numpy().astype(np.float32),
                                1.0 - (PHASE_COUNT * outer_step + phase + 1) / (PHASE_COUNT * sample_steps),
                                phase,
                                int(PHASE_MATCHINGS[phase]),
                                float(PHASE_DURATIONS[phase]),
                                int(label_values[row]),
                                target[row].detach().cpu().numpy().astype(np.float32),
                                int(path_id),
                                outer_step,
                            )
                        )
        if progress_callback is not None and (
            (outer_step + 1) % 8 == 0 or outer_step + 1 == int(sample_steps)
        ):
            progress_callback()
    if len(outputs) != len(ids) * len(selected):
        raise EulerianJacobiDDPMError("forward record selection is incomplete")
    return ForwardRecordDataset(
        later_states=np.stack([row[0] for row in outputs]),
        reverse_time=np.asarray([row[1] for row in outputs], dtype=np.float32),
        phase=np.asarray([row[2] for row in outputs], dtype=np.int64),
        color=np.asarray([row[3] for row in outputs], dtype=np.int64),
        duration=np.asarray([row[4] for row in outputs], dtype=np.float32),
        labels=np.asarray([row[5] for row in outputs], dtype=np.int64),
        targets=np.stack([row[6] for row in outputs]),
        path_ids=np.asarray([row[7] for row in outputs], dtype=np.int64),
        outer_steps=np.asarray([row[8] for row in outputs], dtype=np.int64),
    )


def iter_forward_record_batches(*args: Any, **kwargs: Any) -> Iterator[ForwardRecordDataset]:
    """Yield one bounded cohort; callers may commit each cohort atomically."""

    states = np.asarray(args[0] if args else kwargs["initial_states"])
    labels = np.asarray(args[1] if len(args) > 1 else kwargs["labels"])
    paths = tuple(args[2] if len(args) > 2 else kwargs["path_ids"])
    named = dict(kwargs)
    named.pop("initial_states", None)
    named.pop("labels", None)
    named.pop("path_ids", None)
    for start in range(0, len(paths), 8):
        yield build_forward_records(
            states[start : start + 8], labels[start : start + 8], paths[start : start + 8], **named
        )


def _concatenate_forward_records(parts: Sequence[ForwardRecordDataset]) -> ForwardRecordDataset:
    if not parts:
        raise EulerianJacobiDDPMError("at least one forward-record part is required")
    return ForwardRecordDataset(
        **{
            name: np.concatenate([getattr(part, name) for part in parts], axis=0)
            for name in ForwardRecordDataset.__dataclass_fields__
        }
    )


@dataclass(frozen=True)
class TrainingResult:
    model_state_dict: Mapping[str, Tensor]
    ema_state_dict: Mapping[str, Tensor]
    selected_state_dict: Mapping[str, Tensor]
    selected_update: int
    selected_validation_mse: float
    training_target_energy: float
    history: tuple[Mapping[str, Any], ...]
    completed_updates: int


def _inputs_from_dataset(dataset: ForwardRecordDataset, indices: Tensor, device: torch.device) -> ModelInputs:
    index = indices.detach().cpu().numpy()
    return ModelInputs(
        later_full_state=torch.as_tensor(dataset.later_states[index], device=device),
        reverse_time=torch.as_tensor(dataset.reverse_time[index], device=device),
        phase=torch.as_tensor(dataset.phase[index], device=device),
        color=torch.as_tensor(dataset.color[index], device=device),
        duration=torch.as_tensor(dataset.duration[index], device=device),
        label=torch.as_tensor(dataset.labels[index], device=device),
    )


def train_jacobi_ddpm(
    train: ForwardRecordDataset,
    validation: ForwardRecordDataset,
    *,
    device: str | torch.device,
    updates: int = 10_000,
    batch_size: int = 64,
    learning_rate: float = 2e-4,
    ema_decay: float = 0.999,
    validation_interval: int = 250,
    seed: int = 0xE14A01,
    checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> TrainingResult:
    """Train direct m-target MSE and select earliest minimum nonzero EMA update."""

    if len(train) == 0 or len(validation) == 0 or int(updates) <= 0:
        raise EulerianJacobiDDPMError("training datasets and update budget must be positive")
    active_device = torch.device(device)
    torch.manual_seed(int(seed))
    if active_device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = make_model().to(active_device)
    ema = make_model().to(active_device)
    ema.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    target_energy = float(np.mean(np.square(train.targets.astype(np.float64))))
    if not math.isfinite(target_energy) or target_energy <= 0.0:
        raise EulerianJacobiDDPMError("training target energy is invalid")
    generator = torch.Generator(device="cpu").manual_seed(int(seed) ^ 0x51A7)
    history: list[dict[str, Any]] = []
    best_update = -1
    best_mse = math.inf
    best_state: dict[str, Tensor] | None = None

    def validate(update: int) -> float:
        ema.eval()
        weighted_sum = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(validation), int(batch_size)):
                idx = torch.arange(start, min(start + int(batch_size), len(validation)))
                inputs = _inputs_from_dataset(validation, idx, active_device)
                target = torch.as_tensor(validation.targets[idx.numpy()], device=active_device)
                _, raw = direct_m_loss(ema(inputs), target, training_target_energy=target_energy)
                n = int(idx.numel())
                weighted_sum += float(raw.detach().cpu()) * n
                count += n
        return (weighted_sum / count) / target_energy

    validation_target_energy = float(
        np.mean(np.square(validation.targets.astype(np.float64)))
    )
    if not math.isfinite(validation_target_energy) or validation_target_energy < 0.0:
        raise EulerianJacobiDDPMError("validation target energy is invalid")
    # The update-zero authority is the declared q=0 predictor, not the
    # randomly initialized network.  This makes the baseline independent of
    # model initialization and directly comparable with later normalized MSE.
    zero_mse = validation_target_energy / target_energy
    history.append(
        {"update": 0, "validation_normalized_mse": zero_mse, "eligible": 0}
    )
    for update in range(1, int(updates) + 1):
        indices = torch.randint(0, len(train), (int(batch_size),), generator=generator)
        inputs = _inputs_from_dataset(train, indices, active_device)
        target = torch.as_tensor(train.targets[indices.numpy()], device=active_device)
        normalized, raw = direct_m_loss(model(inputs), target, training_target_energy=target_energy)
        optimizer.zero_grad(set_to_none=True)
        normalized.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            for ema_parameter, parameter in zip(ema.parameters(), model.parameters(), strict=True):
                ema_parameter.mul_(float(ema_decay)).add_(parameter, alpha=1.0 - float(ema_decay))
        if update % int(validation_interval) == 0 or update == int(updates):
            mse = validate(update)
            row = {
                "update": update,
                "validation_normalized_mse": mse,
                "training_batch_raw_mse": float(raw.detach().cpu()),
                "training_batch_normalized_mse": float(normalized.detach().cpu()),
                "eligible": int(math.isfinite(mse)),
            }
            history.append(row)
            if math.isfinite(mse) and (mse < best_mse or (mse == best_mse and update < best_update)):
                best_mse, best_update = mse, update
                best_state = {name: value.detach().cpu().clone() for name, value in ema.state_dict().items()}
            if checkpoint_callback is not None:
                checkpoint_callback(
                    {
                        "completed_update": update,
                        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                        "ema_state_dict": {name: value.detach().cpu() for name, value in ema.state_dict().items()},
                        "optimizer_state_dict": optimizer.state_dict(),
                        "history": tuple(history),
                        "best_update": best_update,
                        "best_validation_normalized_mse": best_mse,
                        "best_state_dict": best_state,
                    }
                )
    if best_state is None or best_update <= 0:
        raise EulerianJacobiDDPMError("no finite nonzero checkpoint was selected")
    return TrainingResult(
        model_state_dict={name: value.detach().cpu() for name, value in model.state_dict().items()},
        ema_state_dict={name: value.detach().cpu() for name, value in ema.state_dict().items()},
        selected_state_dict=best_state,
        selected_update=best_update,
        selected_validation_mse=best_mse,
        training_target_energy=target_energy,
        history=tuple(history),
        completed_updates=int(updates),
    )


@dataclass(frozen=True)
class SamplingResult:
    starts: np.ndarray
    final_states: np.ndarray
    anchors: Mapping[int, np.ndarray]
    telemetry: Mapping[str, Any]


def reverse_sample(
    starts: np.ndarray,
    labels: np.ndarray,
    path_ids: Sequence[int],
    *,
    controller: str,
    root_seed: int,
    model: EulerianJacobiDDPMModel | None = None,
    oracle_targets: np.ndarray | None = None,
    device: str | torch.device = "cpu",
    anchors: Sequence[int] = (0, 32, 64, 96, 128),
    profile: JacobiRBSpectralProfile | None = None,
    sample_steps: int = OUTER_STEPS,
    progress_callback: Callable[[], None] | None = None,
) -> SamplingResult:
    """Apply the declared reference-half/logistic/reference-half composition."""

    if controller not in {"null", "learned", "oracle"}:
        raise EulerianJacobiDDPMError("controller must be null, learned, or oracle")
    rows, _ = _unit_mass_rows(starts, name="reverse starts")
    label_values = np.asarray(labels, dtype=np.int64).reshape(-1)
    ids = tuple(int(value) for value in path_ids)
    if rows.shape[0] != len(ids) or label_values.size != len(ids) or len(ids) > 8:
        raise EulerianJacobiDDPMError("reverse cohort must contain 1..8 aligned paths")
    target_rows = None
    if controller == "oracle":
        if oracle_targets is None:
            raise EulerianJacobiDDPMError("oracle targets are required")
        target_rows, _ = _unit_mass_rows(oracle_targets, name="oracle targets")
        if target_rows.shape != rows.shape:
            raise EulerianJacobiDDPMError("oracle target shape changed")
    if controller == "learned" and model is None:
        raise EulerianJacobiDDPMError("learned controller requires a model")
    active_device = torch.device(device)
    state = torch.as_tensor(rows, dtype=torch.float64, device=active_device)
    labels_tensor = torch.as_tensor(label_values, dtype=torch.long, device=active_device)
    target_tensor = torch.as_tensor(target_rows, dtype=torch.float64, device=active_device) if target_rows is not None else None
    active_profile = profile or JacobiRBSpectralProfile()
    anchor_set = {int(value) for value in anchors}
    steps = int(sample_steps)
    if steps not in {128, 256, 512, 1024, 2048}:
        raise EulerianJacobiDDPMError("reverse sample_steps is unsupported")
    if 0 not in anchor_set or steps not in anchor_set:
        raise EulerianJacobiDDPMError(f"anchors must include 0 and {steps}")
    saved: dict[int, np.ndarray] = {0: state.detach().cpu().numpy().copy()}
    score_squares = 0.0
    score_count = 0
    maximum_score = 0.0
    maximum_mass_error = 0.0
    maximum_pair_total_error = 0.0
    quarter_score_squares = [0.0] * 4
    quarter_score_counts = [0] * 4
    quarter_reference_squares = [0.0] * 4
    quarter_reference_counts = [0] * 4
    quarter_control_squares = [0.0] * 4
    quarter_control_counts = [0] * 4
    quarter_maximum_q = [0.0] * 4
    quarter_maximum_logit_increment = [0.0] * 4
    quarter_maximum_pair_total_error = [0.0] * 4
    completed = 0
    tails_all, heads_all = matching_indices(device=active_device)
    if model is not None:
        model = model.to(active_device).eval()
    for outer_step in range(steps - 1, -1, -1):
        quarter = min(3, (4 * outer_step) // steps)
        for phase in range(PHASE_COUNT - 1, -1, -1):
            color = int(PHASE_MATCHINGS[phase])
            tails, heads = tails_all[color], heads_all[color]
            pair = state[:, tails] + state[:, heads]
            full_exposure = refinement_phase_exposure(
                pair,
                sample_steps=steps,
                duration_fraction=float(PHASE_DURATIONS[phase]),
            )
            delta = full_exposure / CONTROLLER_MICROSTEPS
            for micro in range(CONTROLLER_MICROSTEPS):
                for side in ("pre", "post"):
                    fraction = torch.zeros_like(pair)
                    active = pair > 0.0
                    fraction[active] = state[:, heads][active] / pair[active]
                    # A role-specific key supplies distinct halves while the same
                    # IDs/key are reusable across null/learned rows as common RNG.
                    transition_ids = canonical_refinement_transition_ids(
                        ids,
                        sample_steps=steps,
                        outer_step=outer_step,
                        phase=phase,
                        device=active_device,
                    ).reshape_as(fraction)
                    proposal = propose_alpha1_rb_transition_batch_torch(
                        fraction,
                        delta / 2.0,
                        rng_key=(int(root_seed), "reverse", micro, side),
                        profile=active_profile,
                        transition_ids=transition_ids,
                    )
                    next_fraction = proposal.proposed_later_head_fraction.to(state)
                    reference_displacement = next_fraction - fraction
                    quarter_reference_squares[quarter] += float(
                        torch.sum(reference_displacement.square()).detach().cpu()
                    )
                    quarter_reference_counts[quarter] += int(reference_displacement.numel())
                    state = frozen_score_logistic_flow(state, (tails, heads), torch.zeros_like(next_fraction), 0.0)
                    state[:, heads] = pair * next_fraction
                    state[:, tails] = pair * (1.0 - next_fraction)
                    pair_error = float(
                        torch.max(
                            torch.abs(state[:, tails] + state[:, heads] - pair)
                        ).detach().cpu()
                    )
                    maximum_pair_total_error = max(
                        maximum_pair_total_error, pair_error
                    )
                    quarter_maximum_pair_total_error[quarter] = max(
                        quarter_maximum_pair_total_error[quarter], pair_error
                    )
                    if side == "pre":
                        if controller == "null":
                            score = torch.zeros_like(pair)
                        elif controller == "learned":
                            assert model is not None
                            reverse_time = reverse_midpoint_time(
                                outer_step,
                                phase,
                                micro,
                                sample_steps=steps,
                            )
                            inputs = ModelInputs(
                                later_full_state=state.to(torch.float32),
                                reverse_time=torch.full((len(ids),), reverse_time, dtype=torch.float32, device=active_device),
                                phase=torch.full((len(ids),), phase, dtype=torch.long, device=active_device),
                                color=torch.full((len(ids),), color, dtype=torch.long, device=active_device),
                                duration=torch.full((len(ids),), float(PHASE_DURATIONS[phase]), dtype=torch.float32, device=active_device),
                                label=labels_tensor,
                            )
                            with torch.no_grad():
                                score = model.score_prediction(inputs).to(torch.float64)
                        else:
                            assert target_tensor is not None
                            current = next_fraction
                            target_pair = target_tensor[:, tails] + target_tensor[:, heads]
                            target_fraction = torch.zeros_like(pair)
                            reachable = (pair > 0.0) & (target_pair > 0.0)
                            target_fraction[reachable] = target_tensor[:, heads][reachable] / target_pair[reachable]
                            interior = reachable & (current > 0.0) & (current < 1.0) & (target_fraction > 0.0) & (target_fraction < 1.0) & (delta > 0.0)
                            score = torch.zeros_like(pair)
                            score[interior] = (torch.logit(target_fraction[interior]) - torch.logit(current[interior])) / (2.0 * delta[interior])
                        before_control = torch.zeros_like(pair)
                        before_control[active] = state[:, heads][active] / pair[active]
                        state = frozen_score_logistic_flow(state, (tails, heads), score, delta)
                        control_pair_error = float(
                            torch.max(
                                torch.abs(state[:, tails] + state[:, heads] - pair)
                            ).detach().cpu()
                        )
                        maximum_pair_total_error = max(
                            maximum_pair_total_error, control_pair_error
                        )
                        quarter_maximum_pair_total_error[quarter] = max(
                            quarter_maximum_pair_total_error[quarter],
                            control_pair_error,
                        )
                        after_control = torch.zeros_like(pair)
                        after_control[active] = state[:, heads][active] / pair[active]
                        control_displacement = after_control - before_control
                        score_squares += float(torch.sum(score.square()).detach().cpu())
                        score_count += int(score.numel())
                        maximum_score = max(maximum_score, float(torch.max(torch.abs(score)).detach().cpu()))
                        quarter_score_squares[quarter] += float(torch.sum(score.square()).detach().cpu())
                        quarter_score_counts[quarter] += int(score.numel())
                        quarter_control_squares[quarter] += float(
                            torch.sum(control_displacement.square()).detach().cpu()
                        )
                        quarter_control_counts[quarter] += int(control_displacement.numel())
                        quarter_maximum_q[quarter] = max(
                            quarter_maximum_q[quarter],
                            float(torch.max(torch.abs(score)).detach().cpu()),
                        )
                        quarter_maximum_logit_increment[quarter] = max(
                            quarter_maximum_logit_increment[quarter],
                            float(torch.max(torch.abs(2.0 * score * delta)).detach().cpu()),
                        )
            maximum_mass_error = max(maximum_mass_error, float(torch.max(torch.abs(state.sum(1) - 1.0)).detach().cpu()))
        completed += 1
        if completed in anchor_set:
            saved[completed] = state.detach().cpu().numpy().copy()
        if progress_callback is not None and (
            completed % 8 == 0 or completed == steps
        ):
            progress_callback()
    final = state.detach().cpu().numpy().copy()
    by_time_quarter = []
    for quarter in range(4):
        score_quarter_count = quarter_score_counts[quarter]
        reference_quarter_count = quarter_reference_counts[quarter]
        control_quarter_count = quarter_control_counts[quarter]
        by_time_quarter.append(
            {
                "quarter": quarter,
                "time_quarter": quarter,
                "score_count": score_quarter_count,
                "score_rms": math.sqrt(quarter_score_squares[quarter] / score_quarter_count)
                if score_quarter_count
                else 0.0,
                "controller_rms": math.sqrt(
                    quarter_score_squares[quarter] / score_quarter_count
                )
                if score_quarter_count
                else 0.0,
                "reference_fraction_displacement_rms": math.sqrt(
                    quarter_reference_squares[quarter] / reference_quarter_count
                )
                if reference_quarter_count
                else 0.0,
                "control_fraction_displacement_rms": math.sqrt(
                    quarter_control_squares[quarter] / control_quarter_count
                )
                if control_quarter_count
                else 0.0,
                "maximum_absolute_q": quarter_maximum_q[quarter],
                "maximum_absolute_logit_increment": quarter_maximum_logit_increment[quarter],
                "maximum_mass_error": maximum_mass_error,
                "maximum_pair_total_error": quarter_maximum_pair_total_error[quarter],
            }
        )
    return SamplingResult(
        starts=np.ascontiguousarray(rows),
        final_states=np.ascontiguousarray(final),
        anchors=saved,
        telemetry={
            "controller": controller,
            "controller_rms": math.sqrt(score_squares / score_count) if score_count else 0.0,
            "maximum_absolute_q": maximum_score,
            "maximum_mass_error": maximum_mass_error,
            "maximum_pair_total_error": maximum_pair_total_error,
            "exact_facet_count": int(np.count_nonzero(final == 0.0)),
            "finite": int(np.isfinite(final).all()),
            "nonnegative": int(np.all(final >= 0.0)),
            "microsteps": CONTROLLER_MICROSTEPS,
            "by_time_quarter": by_time_quarter,
        },
    )


def prior_sample(path_ids: Sequence[int], labels: np.ndarray, **kwargs: Any) -> SamplingResult:
    starts = sample_dirichlet_starts(path_ids, root_seed=int(kwargs.pop("start_seed")))
    return reverse_sample(starts, labels, path_ids, **kwargs)


def forward_terminal_sample(initial_states: np.ndarray, labels: np.ndarray, path_ids: Sequence[int], *, forward_seed: int, **kwargs: Any) -> SamplingResult:
    profile = kwargs.get("profile") or JacobiRBSpectralProfile()
    terminal = _forward_terminal_states(
        initial_states,
        path_ids,
        root_seed=int(forward_seed),
        sample_steps=OUTER_STEPS,
        device=kwargs.get("device", "cpu"),
        profile=profile,
        progress_callback=kwargs.get("progress_callback"),
    )
    return reverse_sample(terminal, labels, path_ids, **kwargs)


def oracle_sample(starts: np.ndarray, labels: np.ndarray, path_ids: Sequence[int], targets: np.ndarray, **kwargs: Any) -> SamplingResult:
    return reverse_sample(starts, labels, path_ids, controller="oracle", oracle_targets=targets, **kwargs)


def tiny_synthetic_smoke(*, seed: int = 26_140_004) -> dict[str, Any]:
    """Run the required 4x4/two-class CPU optimizer and control smoke."""

    torch.manual_seed(int(seed))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    features = torch.randn((16, 18), generator=generator)
    labels = torch.arange(16) % 2
    target = torch.tanh(features[:, :16] + 0.25 * labels[:, None])
    model = nn.Sequential(nn.Linear(18, 32), nn.SiLU(), nn.Linear(32, 16))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    initial = torch.mean((model(features) - target).square())
    optimizer.zero_grad(set_to_none=True)
    initial.backward()
    optimizer.step()
    final = torch.mean((model(features) - target).square())
    starts = torch.softmax(torch.randn((4, 16), generator=generator), dim=1)
    null = starts.clone()
    learned = torch.softmax(torch.log(starts) + 0.01 * model(features[:4]).detach(), dim=1)
    oracle = torch.softmax(torch.log(starts) + target[:4], dim=1)
    rows = {"null": null, "learned": learned, "oracle": oracle}
    finite = all(bool(torch.isfinite(value).all()) for value in rows.values())
    nonnegative = all(bool((value >= 0).all()) for value in rows.values())
    simplex = all(bool(torch.all(torch.abs(value.sum(1) - 1.0) <= 1e-6)) for value in rows.values())
    return {
        "schema": EULERIAN_JACOBI_DDPM_VERSION + "-tiny-smoke",
        "device": "cpu",
        "grid_size": 4,
        "class_count": 2,
        "outer_steps": 8,
        "optimizer_updates": 1,
        "initial_loss": float(initial.detach()),
        "final_loss": float(final.detach()),
        "sample_rows": list(rows),
        "finite": int(finite),
        "nonnegative": int(nonnegative),
        "simplex_preserved": int(simplex),
        "passed": int(finite and nonnegative and simplex),
    }


__all__ = [
    "CONTROLLER_MICROSTEPS",
    "EDGES_PER_PHASE",
    "EVALUATION_PATH_ID_START",
    "EulerianJacobiDDPMConfig",
    "EulerianJacobiDDPMError",
    "EulerianJacobiDDPMModel",
    "FORBIDDEN_MODEL_INPUT_FIELDS",
    "ForwardRecordDataset",
    "GRID_SIZE",
    "LAMBDA_MIX",
    "ModelInputs",
    "OUTER_STEPS",
    "PHASE_DURATIONS",
    "PHASE_MATCHINGS",
    "PREFLIGHT_PATH_ID_START",
    "RASTER_SCALE",
    "REFERENCE_OUTER_STEPS",
    "STATE_SIZE",
    "SamplingResult",
    "TRAIN_PATH_COUNT",
    "TRAIN_PATH_IDS",
    "TrainingResult",
    "VALIDATION_PATH_COUNT",
    "VALIDATION_PATH_IDS",
    "balanced_class_indices",
    "build_forward_records",
    "certified_rb_target_fixture",
    "demix_unit_masses",
    "direct_m_loss",
    "fast_rb_target",
    "fast_vs_certified_audit",
    "forward_terminal_sample",
    "frozen_config",
    "frozen_score_logistic_flow",
    "frozen_score_logistic_fraction",
    "iter_forward_record_batches",
    "k128_schedule_contract",
    "make_model",
    "matching_indices",
    "mix_unit_masses",
    "oracle_sample",
    "paired_k128_k512_oracle_audit",
    "paired_schedule_exposure_audit",
    "path_id_inventory_contract",
    "prior_sample",
    "rasterize_unit_masses",
    "reverse_midpoint_time",
    "reverse_sample",
    "sample_dirichlet_starts",
    "tiny_synthetic_smoke",
    "train_jacobi_ddpm",
]
