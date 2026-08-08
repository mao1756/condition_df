"""Sealed raw-endpoint power panels for certified Haar Jacobi coupling.

This module is additive to the immutable Jacobi, Strang, and Dynkin sources.
It executes the exact hierarchical schedulers, commits every aligned shard,
and turns the resulting *raw* endpoint observables into the preregistered
engineering power forecasts.  Dynkin observables are retained only as
advisory evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
from zipfile import BadZipFile

import numpy as np
import torch

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_json,
    file_fingerprint,
)
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_haar_gate import (
    ANTITHETIC_HAAR_PROFILE,
    NESTED_HAAR_PROFILE,
    HaarCouplingThresholds,
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


HAAR_POWER_VERSION = "d0-jacobi-rb-certified-haar-power-v1"
OBSERVATION_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
OBSERVABLE_COUNT = 10
EDGES_PER_PHASE = 392
PHASE_COUNT = 7
POWER_TOTAL_FAMILY_ERROR = 0.01
POWER_VARIANCE_FAMILY_ERROR = 0.005
POWER_MEAN_FAMILY_ERROR = 0.005
FORBIDDEN_COUNTS = (
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
NO_WORK = {
    "physical_training_performed": 0,
    "production_refinement_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
}


class HaarPowerError(RuntimeError):
    """A sealed panel could not produce complete authorizing evidence."""

    def __init__(
        self,
        message: str,
        *,
        failure_code: str,
        failure_domain: str = "hierarchical_scheduler",
        **diagnostics: Any,
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code
        self.failure_domain = failure_domain
        self.diagnostics = diagnostics


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{
                str(name): np.ascontiguousarray(value)
                for name, value in arrays.items()
            },
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_mixed_state(parent_run_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    metadata_path = parent_run_dir / "source_image.json"
    payload_path = parent_run_dir / "source_image.npz"
    if not metadata_path.is_file() or not payload_path.is_file():
        raise ArtifactCompatibilityError("parent source-image binding is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("source_npz_sha256") != file_fingerprint(payload_path)
    ):
        raise ArtifactCompatibilityError("parent source-image hash changed")
    with np.load(payload_path, allow_pickle=False) as archive:
        if set(archive.files) != {"image", "mixed_target"}:
            raise ArtifactCompatibilityError("parent source-image payload changed")
        state = np.asarray(archive["mixed_target"], dtype=np.float64)
    if (
        state.shape != (784,)
        or not np.isfinite(state).all()
        or np.any(state < 0.0)
        or abs(float(state.sum()) - 1.0) > 1.0e-12
    ):
        raise ArtifactCompatibilityError("parent mixed target is invalid")
    return np.array(state, copy=True, order="C"), dict(metadata)


def _initial_states(
    parent_run_dir: Path,
    path_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    state, metadata = _load_mixed_state(parent_run_dir)
    values = np.repeat(state.reshape(1, -1), int(path_count), axis=0)
    return (
        torch.as_tensor(values, dtype=torch.float64, device=device).contiguous(),
        metadata,
    )


def _branch_checkpoints(
    completed_steps: Sequence[int],
    values: np.ndarray,
    *,
    sample_steps: int,
    destination: dict[int, np.ndarray],
) -> None:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 3
        or array.shape[0] != len(completed_steps)
        or array.shape[2] != OBSERVABLE_COUNT
        or not np.isfinite(array).all()
    ):
        raise HaarPowerError(
            "a shard returned invalid observable checkpoints",
            failure_code="hierarchical_observable_checkpoint_invalid",
        )
    requested = {
        int(round(fraction * int(sample_steps)))
        for fraction in OBSERVATION_FRACTIONS
    }
    for index, step in enumerate(completed_steps):
        if int(step) in requested:
            if int(step) in destination:
                raise HaarPowerError(
                    "a checkpoint was recorded twice",
                    failure_code="hierarchical_observable_checkpoint_duplicate",
                )
            destination[int(step)] = np.array(array[index], copy=True, order="C")


def _ordered_level_observables(
    checkpoints: Mapping[int, np.ndarray],
    sample_steps: int,
) -> np.ndarray:
    steps = tuple(
        int(round(fraction * int(sample_steps)))
        for fraction in OBSERVATION_FRACTIONS
    )
    if set(checkpoints) != set(steps):
        raise HaarPowerError(
            f"K={sample_steps} lacks one or more frozen observation times",
            failure_code="hierarchical_observable_checkpoint_missing",
        )
    result = np.stack([np.asarray(checkpoints[step]) for step in steps], axis=1)
    if result.ndim != 3 or result.shape[2] != OBSERVABLE_COUNT:
        raise HaarPowerError(
            "ordered level observables have an invalid shape",
            failure_code="hierarchical_observable_checkpoint_invalid",
        )
    return result


def _check_resumed_input(
    metadata: Mapping[str, Any],
    expected: str,
) -> None:
    if metadata.get("input_sha256") != expected:
        raise ArtifactCompatibilityError(
            "committed Haar shard does not follow the current predecessor"
        )


def _discard_corrupt_tail(
    shard_root: Path,
    *,
    preserved_stems: set[str],
) -> None:
    """Remove only the untrusted suffix of one profile/panel shard chain."""

    if not shard_root.exists():
        return
    for path in shard_root.iterdir():
        if (
            path.is_file()
            and path.suffix in {".json", ".npz"}
            and path.stem not in preserved_stems
        ):
            path.unlink()


def _resume_shard_or_recover_tail(
    *,
    shard_root: Path,
    identity: HaarShardIdentity,
    expected_input: str,
    device: torch.device,
    preserved_stems: set[str],
) -> Any | None:
    metadata_path = shard_root / f"{identity.fingerprint}.json"
    state_path = shard_root / f"{identity.fingerprint}.npz"
    if not metadata_path.exists() and not state_path.exists():
        return None
    if not metadata_path.is_file() or not state_path.is_file():
        _discard_corrupt_tail(
            shard_root, preserved_stems=preserved_stems
        )
        return None
    try:
        resumed = load_committed_haar_shard(
            shard_root,
            expected_identity=identity,
            device=device,
        )
        _check_resumed_input(resumed.metadata, expected_input)
    except (
        ArtifactCompatibilityError,
        HaarSchedulerError,
        OSError,
        EOFError,
        ValueError,
        BadZipFile,
    ):
        _discard_corrupt_tail(
            shard_root, preserved_stems=preserved_stems
        )
        return None
    return resumed


def _nested_pool(
    *,
    run_dir: Path,
    panel: str,
    pool: str,
    role: str,
    path_ids: tuple[int, ...],
    initial: torch.Tensor,
    root_seed: int,
    jacobi_profile: JacobiRBCudaProfile,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[dict[str, Any]]]:
    schedule = NestedHaarSchedule(pool=pool, role=role)
    assert schedule.levels is not None
    states = initialize_nested_branch_states(initial, schedule)
    accumulators: dict[int, Any] = {int(level): None for level in schedule.levels}
    raw: dict[int, dict[int, np.ndarray]] = {
        int(level): {} for level in schedule.levels
    }
    dynkin: dict[int, dict[int, np.ndarray]] = {
        int(level): {} for level in schedule.levels
    }
    records: list[dict[str, Any]] = []
    preserved_stems: set[str] = set()
    shard_root = run_dir / "haar_power_shards" / NESTED_HAAR_PROFILE / panel / pool
    for first in range(0, schedule.coarsest_steps, 8):
        identity = HaarShardIdentity(
            schedule=schedule,
            path_ids=path_ids,
            coarsest_start_step=first,
            root_seed=int(root_seed),
            panel_namespace=f"haar-power:{panel}:{pool}",
        )
        named_states = {f"k{level}": states[int(level)] for level in schedule.levels}
        named_accumulators = {
            f"k{level}": accumulators[int(level)] for level in schedule.levels
        }
        expected_input = expected_haar_shard_input_sha256(
            identity, named_states, named_accumulators, jacobi_profile
        )
        resumed = _resume_shard_or_recover_tail(
            shard_root=shard_root,
            identity=identity,
            expected_input=expected_input,
            device=initial.device,
            preserved_stems=preserved_stems,
        )
        if resumed is not None:
            metadata = dict(resumed.metadata)
            states = {
                int(level): resumed.states[f"k{level}"]
                for level in schedule.levels
            }
            accumulators = {
                int(level): resumed.accumulators[f"k{level}"]
                for level in schedule.levels
            }
            raw_values = resumed.raw_observables
            dynkin_values = resumed.dynkin_observables
        else:
            result = run_nested_haar_shard(
                states,
                identity=identity,
                jacobi_profile=jacobi_profile,
                accumulators_by_level=accumulators,
            )
            metadata = commit_haar_shard(result, shard_root)
            states = {
                int(level): result.branches[f"k{level}"].final_states
                for level in schedule.levels
            }
            accumulators = {
                int(level): result.branches[f"k{level}"].accumulator_state
                for level in schedule.levels
            }
            raw_values = {
                name: branch.raw_observables
                for name, branch in result.branches.items()
            }
            dynkin_values = {
                name: branch.dynkin_observables
                for name, branch in result.branches.items()
            }
        preserved_stems.add(identity.fingerprint)
        for level in schedule.levels:
            name = f"k{level}"
            completed = metadata["branches"][name]["completed_steps"]
            _branch_checkpoints(
                completed,
                raw_values[name],
                sample_steps=int(level),
                destination=raw[int(level)],
            )
            _branch_checkpoints(
                completed,
                dynkin_values[name],
                sample_steps=int(level),
                destination=dynkin[int(level)],
            )
        records.append(metadata)
    return (
        {
            int(level): _ordered_level_observables(raw[int(level)], int(level))
            for level in schedule.levels
        },
        {
            int(level): _ordered_level_observables(
                dynkin[int(level)], int(level)
            )
            for level in schedule.levels
        },
        records,
    )


def _antithetic_pair(
    *,
    run_dir: Path,
    panel: str,
    role: str,
    path_ids: tuple[int, ...],
    initial: torch.Tensor,
    root_seed: int,
    coarse_steps: int,
    fine_steps: int,
    jacobi_profile: JacobiRBCudaProfile,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    schedule = PairwiseHaarAntitheticSchedule(
        coarse_steps=coarse_steps,
        fine_steps=fine_steps,
        role=role,
    )
    states = initialize_antithetic_branch_states(initial)
    accumulators: dict[str, Any] = {
        "coarse": None,
        "fine_plus": None,
        "fine_minus": None,
    }
    raw: dict[str, dict[int, np.ndarray]] = {
        name: {} for name in states
    }
    dynkin: dict[str, dict[int, np.ndarray]] = {
        name: {} for name in states
    }
    records: list[dict[str, Any]] = []
    preserved_stems: set[str] = set()
    pair_name = f"K{coarse_steps:04d}-K{fine_steps:04d}"
    shard_root = (
        run_dir
        / "haar_power_shards"
        / ANTITHETIC_HAAR_PROFILE
        / panel
        / pair_name
    )
    for first in range(0, int(coarse_steps), 8):
        identity = HaarShardIdentity(
            schedule=schedule,
            path_ids=path_ids,
            coarsest_start_step=first,
            root_seed=int(root_seed),
            panel_namespace=f"haar-power:{panel}:{pair_name}",
        )
        expected_input = expected_haar_shard_input_sha256(
            identity, states, accumulators, jacobi_profile
        )
        resumed = _resume_shard_or_recover_tail(
            shard_root=shard_root,
            identity=identity,
            expected_input=expected_input,
            device=initial.device,
            preserved_stems=preserved_stems,
        )
        if resumed is not None:
            metadata = dict(resumed.metadata)
            states = dict(resumed.states)
            accumulators = dict(resumed.accumulators)
            raw_values = resumed.raw_observables
            dynkin_values = resumed.dynkin_observables
        else:
            result = run_pairwise_haar_antithetic_shard(
                coarse_state=states["coarse"],
                fine_plus_state=states["fine_plus"],
                fine_minus_state=states["fine_minus"],
                coarse_accumulator=accumulators["coarse"],
                fine_plus_accumulator=accumulators["fine_plus"],
                fine_minus_accumulator=accumulators["fine_minus"],
                identity=identity,
                jacobi_profile=jacobi_profile,
            )
            metadata = commit_haar_shard(result, shard_root)
            states = {
                name: branch.final_states
                for name, branch in result.branches.items()
            }
            accumulators = {
                name: branch.accumulator_state
                for name, branch in result.branches.items()
            }
            raw_values = {
                name: branch.raw_observables
                for name, branch in result.branches.items()
            }
            dynkin_values = {
                name: branch.dynkin_observables
                for name, branch in result.branches.items()
            }
        preserved_stems.add(identity.fingerprint)
        for name, level in (
            ("coarse", coarse_steps),
            ("fine_plus", fine_steps),
            ("fine_minus", fine_steps),
        ):
            _branch_checkpoints(
                metadata["branches"][name]["completed_steps"],
                raw_values[name],
                sample_steps=level,
                destination=raw[name],
            )
            _branch_checkpoints(
                metadata["branches"][name]["completed_steps"],
                dynkin_values[name],
                sample_steps=level,
                destination=dynkin[name],
            )
        records.append(metadata)
    coarse_raw = _ordered_level_observables(raw["coarse"], coarse_steps)
    fine_raw = 0.5 * (
        _ordered_level_observables(raw["fine_plus"], fine_steps)
        + _ordered_level_observables(raw["fine_minus"], fine_steps)
    )
    coarse_dynkin = _ordered_level_observables(dynkin["coarse"], coarse_steps)
    fine_dynkin = 0.5 * (
        _ordered_level_observables(dynkin["fine_plus"], fine_steps)
        + _ordered_level_observables(dynkin["fine_minus"], fine_steps)
    )
    return coarse_raw - fine_raw, coarse_dynkin - fine_dynkin, records


def _execution_record(
    records: Sequence[Mapping[str, Any]],
    *,
    peak_memory_fraction: float,
) -> dict[str, Any]:
    if not records:
        raise HaarPowerError(
            "panel executed no shards",
            failure_code="hierarchical_panel_empty",
        )
    diagnostics = [dict(value.get("diagnostics", {})) for value in records]
    timings = [dict(value.get("timing", {})) for value in records]

    def count(record: Mapping[str, Any], name: str, *, positive: bool = False) -> int:
        value = record.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < int(positive)
        ):
            raise HaarPowerError(
                "panel shard diagnostics are incomplete",
                failure_code="hierarchical_panel_diagnostics_invalid",
                diagnostic=name,
                value=value,
            )
        return int(value)

    def finite(
        record: Mapping[str, Any], name: str, *, positive: bool = False
    ) -> float:
        value = record.get(name)
        try:
            result = float(value)
        except (TypeError, ValueError):
            result = math.nan
        if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
            raise HaarPowerError(
                "panel shard diagnostics are incomplete",
                failure_code="hierarchical_panel_diagnostics_invalid",
                diagnostic=name,
                value=value,
            )
        return result

    transition_counts = [
        count(value, "transition_count", positive=True) for value in diagnostics
    ]
    elapsed_values = [
        finite(
            value,
            "complete_pipeline_including_state_shard_io_seconds",
            positive=True,
        )
        for value in timings
    ]
    fallback_counts = [
        count(value, "fallback_count") for value in diagnostics
    ]
    fallback_elapsed_values = [
        finite(value, "fallback_elapsed_seconds") for value in diagnostics
    ]
    for value in diagnostics:
        certificate_fraction = finite(value, "certificate_fraction")
        finite(value, "mass_error")
        state_residency = count(value, "state_updates_device_resident_pass")
        if certificate_fraction > 1.0 or state_residency not in {0, 1}:
            raise HaarPowerError(
                "panel shard diagnostics are incomplete",
                failure_code="hierarchical_panel_diagnostics_invalid",
            )
        for name in FORBIDDEN_COUNTS:
            try:
                count(value, name)
            except HaarPowerError as exc:
                raise HaarPowerError(
                    "forbidden-event diagnostics are incomplete",
                    failure_code="hierarchical_panel_diagnostics_invalid",
                    diagnostic=name,
                    value=value.get(name),
                ) from exc
    transition_count = sum(transition_counts)
    elapsed = sum(elapsed_values)
    fallback_count = sum(fallback_counts)
    fallback_elapsed = sum(fallback_elapsed_values)
    certificate_ok = bool(diagnostics) and all(
        float(value["certificate_fraction"]) == 1.0
        for value in diagnostics
    )
    if (
        transition_count <= 0
        or any(value <= 0 for value in transition_counts)
        or any(
            not math.isfinite(value) or value <= 0.0
            for value in elapsed_values
        )
        or not math.isfinite(elapsed)
        or elapsed <= 0.0
        or fallback_count < 0
        or not math.isfinite(fallback_elapsed)
    ):
        raise HaarPowerError(
            "panel shard diagnostics are incomplete",
            failure_code="hierarchical_panel_diagnostics_invalid",
        )
    forbidden_totals = {
        name: sum(int(value[name]) for value in diagnostics)
        for name in FORBIDDEN_COUNTS
    }
    grouped_counts: dict[str, int] = {}
    grouped_elapsed: dict[str, float] = {}
    for record, count, duration in zip(
        records, transition_counts, elapsed_values
    ):
        schedule = record.get("schedule")
        if not isinstance(schedule, Mapping):
            raise HaarPowerError(
                "panel shard schedule binding is missing",
                failure_code="hierarchical_panel_diagnostics_invalid",
            )
        key = json.dumps(
            schedule, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        grouped_counts[key] = grouped_counts.get(key, 0) + count
        grouped_elapsed[key] = grouped_elapsed.get(key, 0.0) + duration
    group_rates = [
        grouped_counts[key] / grouped_elapsed[key]
        for key in grouped_counts
    ]
    mass_error = max(
        float(value["mass_error"]) for value in diagnostics
    )
    if not math.isfinite(float(peak_memory_fraction)) or not (
        0.0 <= float(peak_memory_fraction) <= 1.0
    ):
        raise HaarPowerError(
            "panel shard diagnostics are incomplete",
            failure_code="hierarchical_panel_diagnostics_invalid",
            diagnostic="peak_memory_fraction",
            value=peak_memory_fraction,
        )
    result = {
        "shard_count": len(records),
        "transition_count": transition_count,
        "certified_count": transition_count if certificate_ok else 0,
        "certificate_fraction": 1.0 if certificate_ok else 0.0,
        "fallback_count": fallback_count,
        "fallback_fraction": fallback_count / transition_count,
        "fallback_elapsed_seconds": fallback_elapsed,
        "fallback_cost_fraction": fallback_elapsed / elapsed,
        "elapsed_seconds": elapsed,
        "complete_wall_upper_seconds": elapsed,
        "aggregate_rate": transition_count / elapsed,
        "slowest_schedule_rate": min(group_rates),
        "conservative_rate": min(group_rates),
        "mass_error": mass_error,
        "peak_memory_fraction": float(peak_memory_fraction),
        "state_updates_device_resident_pass": int(
            all(
                int(value.get("state_updates_device_resident_pass", 0)) == 1
                for value in diagnostics
            )
        ),
        "shard_chain_pass": 1,
        **forbidden_totals,
    }
    return result


def _flatten_differences(values: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    if (
        not arrays
        or any(array.ndim != 3 or array.shape[1:] != (4, 10) for array in arrays)
        or any(array.shape[0] != arrays[0].shape[0] for array in arrays)
    ):
        raise HaarPowerError(
            "coupled differences have an invalid path/time/observable shape",
            failure_code="hierarchical_power_feature_invalid",
        )
    result = np.concatenate(
        [array.reshape(array.shape[0], -1) for array in arrays], axis=1
    )
    if not np.isfinite(result).all():
        raise HaarPowerError(
            "coupled differences are nonfinite",
            failure_code="hierarchical_power_feature_invalid",
        )
    return result


def _variance_upper(
    samples: np.ndarray,
    *,
    family_size: int,
    family_error: float = POWER_TOTAL_FAMILY_ERROR,
) -> tuple[np.ndarray, float]:
    """Return a variance envelope and mean critical with joint 99% coverage.

    The total familywise error is split between two prerequisite events:
    the pilot variance envelope and the future mean bound.  A union bound
    therefore leaves at least ``1 - family_error`` simultaneous coverage.
    """

    from scipy import stats

    values = np.asarray(samples, dtype=np.float64)
    total_error = float(family_error)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or not np.isfinite(values).all()
        or int(family_size) < 1
        or not 0.0 < total_error < 1.0
    ):
        raise HaarPowerError(
            "variance planning samples are invalid",
            failure_code="hierarchical_power_projection_invalid",
        )
    variance_error = total_error / 2.0
    mean_error = total_error - variance_error
    lower = float(
        stats.chi2.ppf(
            variance_error / float(family_size),
            values.shape[0] - 1,
        )
    )
    if not math.isfinite(lower) or lower <= 0.0:
        raise HaarPowerError(
            "chi-square variance envelope is unresolved",
            failure_code="hierarchical_power_projection_invalid",
        )
    variance = np.var(values, axis=0, ddof=1)
    upper = (values.shape[0] - 1) * variance / lower
    critical = math.sqrt(
        2.0 * math.log(2.0 * float(family_size) / mean_error)
    )
    return upper, critical


def _production_transition_count(
    profile: str,
    main_paths: int,
    reference_paths: int,
) -> int:
    per_step = PHASE_COUNT * EDGES_PER_PHASE
    if profile == NESTED_HAAR_PROFILE:
        levels_main = 128 + 256 + 512 + 1024
        levels_reference = 512 + 1024 + 2048
        return per_step * (
            int(main_paths) * levels_main
            + int(reference_paths) * levels_reference
        )
    if profile == ANTITHETIC_HAAR_PROFILE:
        pair_steps = sum(coarse + 2 * fine for coarse, fine in ADJACENT_LEVEL_PAIRS)
        # The main/reference counts are both frozen at 16.  Pair D1..D3 and
        # pair D4 are independent namespaces but all four workloads execute.
        if int(main_paths) != 16 or int(reference_paths) != 16:
            raise HaarPowerError(
                "antithetic production design changed",
                failure_code="hierarchical_profile_invalid",
            )
        return per_step * int(main_paths) * pair_steps
    raise HaarPowerError(
        f"unknown Haar profile {profile}",
        failure_code="hierarchical_profile_invalid",
    )


def build_power_candidates(
    *,
    profile: str,
    main_differences: np.ndarray,
    d3_differences: np.ndarray,
    d4_differences: np.ndarray,
    execution: Mapping[str, Any],
    evidence_flags: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Build the frozen candidate rows with independent-pool variances."""

    main = np.asarray(main_differences, dtype=np.float64)
    d3 = np.asarray(d3_differences, dtype=np.float64)
    d4 = np.asarray(d4_differences, dtype=np.float64)
    if (
        main.ndim != 2
        or main.shape[1] != 120
        or d3.ndim != 2
        or d3.shape[1] != 40
        or d4.ndim != 2
        or d4.shape[1] != 40
    ):
        raise HaarPowerError(
            "power matrices do not match the frozen feature families",
            failure_code="hierarchical_power_feature_invalid",
        )
    main_variance, main_critical = _variance_upper(main, family_size=120)
    # Generator and stability are one 80-feature family (40 each).  Each
    # component variance is bounded separately with the same family size.
    d3_variance, reference_critical = _variance_upper(d3, family_size=80)
    d4_variance, _ = _variance_upper(d4, family_size=80)
    designs = (
        [(main_paths, reference_paths) for main_paths in (32, 64) for reference_paths in (16, 32)]
        if profile == NESTED_HAAR_PROFILE
        else [(16, 16)]
    )
    rate = float(execution.get("conservative_rate", 0.0))
    rows: list[dict[str, Any]] = []
    for main_paths, reference_paths in designs:
        main_width = main_critical * np.sqrt(main_variance / main_paths)
        generator_width = reference_critical * np.sqrt(
            d3_variance / main_paths
            + (16.0 / 9.0) * d4_variance / reference_paths
        )
        stability_width = reference_critical * np.sqrt(
            d3_variance / (9.0 * main_paths)
            + (16.0 / 9.0) * d4_variance / reference_paths
        )
        transitions = _production_transition_count(
            profile, main_paths, reference_paths
        )
        rows.append(
            {
                "main_paths": main_paths,
                "reference_paths": reference_paths,
                "predicted_main_half_width": float(np.max(main_width)),
                "predicted_generator_reference_half_width": float(
                    np.max(generator_width)
                ),
                "predicted_reference_stability_half_width": float(
                    np.max(stability_width)
                ),
                "projected_transition_count": transitions,
                "projected_hours": (
                    transitions / rate / 3600.0 if rate > 0.0 else math.inf
                ),
                "conservative_rate": rate,
                "variance_upper_confidence": (
                    1.0 - POWER_VARIANCE_FAMILY_ERROR
                ),
                "mean_bound_confidence": 1.0 - POWER_MEAN_FAMILY_ERROR,
                "joint_power_width_confidence_lower_bound": (
                    1.0 - POWER_TOTAL_FAMILY_ERROR
                ),
                "familywise_error_budget": {
                    "total": POWER_TOTAL_FAMILY_ERROR,
                    "variance_envelope": POWER_VARIANCE_FAMILY_ERROR,
                    "future_mean_bound": POWER_MEAN_FAMILY_ERROR,
                    "combination": "union_bound",
                },
                "main_variance_family_size": 120,
                "reference_variance_family_size": 80,
                "independent_pool_covariance": 0.0,
                "forecast_only": 1,
                **{str(name): int(value) for name, value in evidence_flags.items()},
                **NO_WORK,
            }
        )
    return rows


def _panel_evidence_flags(
    execution: Mapping[str, Any],
    *,
    complete: bool = True,
    finite: bool = True,
    pilot_production_isolation: bool = True,
) -> dict[str, int]:
    thresholds = HaarCouplingThresholds()
    forbidden_clean = all(
        int(execution.get(name, -1)) == 0 for name in FORBIDDEN_COUNTS
    )
    numerical = (
        forbidden_clean
        and float(execution.get("certificate_fraction", 0.0)) == 1.0
        and float(execution.get("fallback_fraction", math.inf))
        <= thresholds.maximum_fallback_fraction
        and float(execution.get("fallback_cost_fraction", math.inf))
        <= thresholds.maximum_fallback_cost_fraction
        and float(execution.get("peak_memory_fraction", math.inf))
        <= thresholds.maximum_peak_memory_fraction
        and int(execution.get("state_updates_device_resident_pass", 0)) == 1
        and int(execution.get("timing_coverage_pass", 0)) == 1
    )
    return {
        "panel_complete_pass": int(bool(complete)),
        "panel_finite_pass": int(bool(finite)),
        "panel_certification_pass": int(
            float(execution.get("certificate_fraction", 0.0)) == 1.0
        ),
        "panel_numerical_health_pass": int(numerical),
        "mass_conservation_pass": int(
            float(execution.get("mass_error", math.inf))
            <= thresholds.maximum_cuda_mass_error
        ),
        "shard_chain_pass": int(execution.get("shard_chain_pass", 0) == 1),
        "pilot_production_isolation_pass": int(
            bool(pilot_production_isolation)
        ),
        "pilot_means_excluded_pass": 1,
        "raw_endpoint_authorizing_pass": 1,
        "dynkin_advisory_only_pass": 1,
    }


def _payload(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    require_existing: bool,
) -> dict[str, Any]:
    if require_existing:
        if not path.is_file():
            raise ArtifactCompatibilityError("sealed panel payload is missing")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays) or any(
                not np.array_equal(np.asarray(archive[name]), value)
                for name, value in arrays.items()
            ):
                raise ArtifactCompatibilityError("sealed panel payload changed")
    else:
        _atomic_npz(path, arrays)
    return {
        "path": path.name,
        "sha256": file_fingerprint(path),
        "size": path.stat().st_size,
        "array_hashes": {
            name: _array_hash(value) for name, value in sorted(arrays.items())
        },
    }


def verify_certified_haar_power_panel_evidence(
    *,
    run_dir: str | Path,
    evidence: Mapping[str, Any],
    expected_profile: str,
    expected_panel: str,
) -> dict[str, Any]:
    """Verify one sealed panel and rederive its authorizing statistics.

    This verifier is intentionally independent of the panel-A nomination or
    panel-B confirmation record.  A crash may leave those control-plane files
    outside the previous terminal registry, so resume must first bind the raw
    NPZ, recompute every power candidate, and only then rederive the sealed
    decision record.
    """

    record = dict(evidence)
    if (
        record.get("schema") != HAAR_POWER_VERSION + "-panel"
        or record.get("schema_version") != 1
        or record.get("evaluation_status") != "evaluated"
        or record.get("profile") != expected_profile
        or record.get("panel") != expected_panel
    ):
        raise ArtifactCompatibilityError("sealed panel evidence identity changed")

    payload = record.get("observable_payload")
    if not isinstance(payload, Mapping):
        raise ArtifactCompatibilityError("sealed panel payload binding is missing")
    relative = payload.get("path")
    if (
        not isinstance(relative, str)
        or Path(relative).name != relative
        or relative
        != f"{expected_profile}_panel_{expected_panel}_observables.npz"
    ):
        raise ArtifactCompatibilityError("sealed panel payload path changed")
    path = Path(run_dir).resolve() / relative
    if (
        not path.is_file()
        or payload.get("sha256") != file_fingerprint(path)
        or int(payload.get("size", -1)) != path.stat().st_size
    ):
        raise ArtifactCompatibilityError("sealed panel payload hash changed")
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                name: np.asarray(archive[name], dtype=np.float64)
                for name in archive.files
            }
    except (OSError, ValueError, BadZipFile) as exc:
        raise ArtifactCompatibilityError(
            f"sealed panel payload is unreadable: {exc}"
        ) from exc
    required_arrays = {
        "raw_main_differences": 120,
        "raw_d3_differences": 40,
        "raw_d4_differences": 40,
        "dynkin_main_differences": 120,
        "dynkin_d3_differences": 40,
        "dynkin_d4_differences": 40,
    }
    cluster_count = int(record.get("cluster_count", -1))
    if (
        set(arrays) != set(required_arrays)
        or cluster_count <= 1
        or any(
            value.shape != (cluster_count, required_arrays[name])
            or not np.isfinite(value).all()
            for name, value in arrays.items()
        )
    ):
        raise ArtifactCompatibilityError("sealed panel payload schema changed")
    expected_hashes = payload.get("array_hashes")
    if not isinstance(expected_hashes, Mapping) or any(
        expected_hashes.get(name) != _array_hash(value)
        for name, value in arrays.items()
    ):
        raise ArtifactCompatibilityError("sealed panel array hash changed")

    execution = record.get("execution")
    if not isinstance(execution, Mapping):
        raise ArtifactCompatibilityError("sealed panel execution evidence is missing")
    required_counts = (
        "transition_count",
        "certified_count",
        "fallback_count",
        *FORBIDDEN_COUNTS,
    )
    for name in required_counts:
        value = execution.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise ArtifactCompatibilityError(
                f"sealed panel execution has invalid {name}"
            )
    transition_count = int(execution["transition_count"])
    certified_count = int(execution["certified_count"])
    finite_nonnegative = (
        "fallback_fraction",
        "fallback_elapsed_seconds",
        "fallback_cost_fraction",
        "elapsed_seconds",
        "conservative_rate",
        "mass_error",
        "peak_memory_fraction",
    )
    for name in finite_nonnegative:
        try:
            value = float(execution.get(name))
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value) or value < 0.0:
            raise ArtifactCompatibilityError(
                f"sealed panel execution has invalid {name}"
            )
    if (
        transition_count <= 0
        or certified_count != transition_count
        or float(execution.get("certificate_fraction", 0.0)) != 1.0
        or int(execution.get("state_updates_device_resident_pass", 0)) != 1
        or int(execution.get("shard_chain_pass", 0)) != 1
        or int(execution.get("timing_coverage_pass", 0)) != 1
    ):
        raise ArtifactCompatibilityError("sealed panel execution is incomplete")

    pools = record.get("path_id_pools")
    reserved = record.get("reserved_production_slot")
    if not isinstance(reserved, Sequence) or isinstance(reserved, (str, bytes)):
        # The production slot lives in the path-ID plan rather than the panel
        # record in schema v1.  Retain compatibility by using the frozen slot.
        reserved = (0xF0000, 0x100000)
    if (
        not isinstance(pools, Mapping)
        or not pools
        or len(reserved) != 2
    ):
        raise ArtifactCompatibilityError("sealed panel path pools are missing")
    flattened: list[int] = []
    for values in pools.values():
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or len(values) != cluster_count
        ):
            raise ArtifactCompatibilityError("sealed panel path pool changed")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ArtifactCompatibilityError("sealed panel path ID changed")
            flattened.append(int(value))
    isolation = (
        len(flattened) == len(set(flattened))
        and all(
            not int(reserved[0]) <= value < int(reserved[1])
            for value in flattened
        )
    )
    if not isolation:
        raise ArtifactCompatibilityError("sealed panel path isolation changed")

    flags = _panel_evidence_flags(
        execution,
        complete=record.get("complete") == 1,
        finite=True,
        pilot_production_isolation=isolation,
    )
    expected_candidates = build_power_candidates(
        profile=expected_profile,
        main_differences=arrays["raw_main_differences"],
        d3_differences=arrays["raw_d3_differences"],
        d4_differences=arrays["raw_d4_differences"],
        execution=execution,
        evidence_flags=flags,
    )
    if record.get("candidates") != expected_candidates:
        raise ArtifactCompatibilityError(
            "sealed panel candidates do not match the bound observables"
        )
    expected_authorizing = int(all(int(value) == 1 for value in flags.values()))
    required_equal = {
        "complete": 1,
        "finite": 1,
        "main_feature_count": 120,
        "reference_feature_count": 80,
        "raw_endpoint_authorizing": 1,
        "dynkin_advisory_only": 1,
        "production_authorizing_pass": expected_authorizing,
        "production_authorizing": expected_authorizing,
        "raw_endpoint_authorizing_pass": 1,
        "dynkin_advisory_only_pass": 1,
        "independent_pool_variance_pass": 1,
        "richardson_formula_pass": 1,
        **flags,
    }
    if any(record.get(name) != value for name, value in required_equal.items()):
        raise ArtifactCompatibilityError(
            "sealed panel evidence flags do not recompute"
        )
    return record


def run_certified_haar_power_panel(
    *,
    run_dir: str | Path,
    parent_run_dir: str | Path,
    root_seed: int,
    profile: str,
    panel: str,
    path_id_plan: Mapping[str, Any],
    device: str | torch.device,
) -> dict[str, Any]:
    """Execute or reconstruct one immutable eight-cluster power panel."""

    if profile not in {NESTED_HAAR_PROFILE, ANTITHETIC_HAAR_PROFILE}:
        raise ValueError("unknown Haar profile")
    if panel not in {"a", "b"}:
        raise ValueError("panel must be a or b")
    root = Path(run_dir)
    parent = Path(parent_run_dir)
    payload_path = root / f"{profile}_panel_{panel}_observables.npz"
    evidence_path = root / f"{profile}_panel_{panel}_evidence.json"
    role = (
        f"nested_{panel}"
        if profile == NESTED_HAAR_PROFILE
        else f"antithetic_{panel}"
    )
    plan = path_id_plan["profiles"][profile][panel]["roles"]
    device_value = torch.device(device)
    if device_value.type != "cuda":
        raise HaarPowerError(
            "production power panels require CUDA",
            failure_code="hierarchical_cuda_state_required",
            failure_domain="runtime_backend",
        )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_value)
    jacobi_profile = JacobiRBCudaProfile()
    all_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    if profile == NESTED_HAAR_PROFILE:
        main_ids = tuple(int(value) for value in plan["main"]["root_path_ids"])
        reference_ids = tuple(
            int(value) for value in plan["reference"]["root_path_ids"]
        )
        main_initial, source_metadata = _initial_states(
            parent, len(main_ids), device_value
        )
        reference_initial, _ = _initial_states(
            parent, len(reference_ids), device_value
        )
        main_raw, main_dynkin, records = _nested_pool(
            run_dir=root,
            panel=panel,
            pool="main",
            role=role,
            path_ids=main_ids,
            initial=main_initial,
            root_seed=int(root_seed),
            jacobi_profile=jacobi_profile,
        )
        all_records.extend(records)
        reference_raw, reference_dynkin, records = _nested_pool(
            run_dir=root,
            panel=panel,
            pool="reference",
            role=role,
            path_ids=reference_ids,
            initial=reference_initial,
            root_seed=int(root_seed),
            jacobi_profile=jacobi_profile,
        )
        all_records.extend(records)
        raw_d1 = main_raw[128] - main_raw[256]
        raw_d2 = main_raw[256] - main_raw[512]
        raw_d3 = main_raw[512] - main_raw[1024]
        raw_d4 = reference_raw[1024] - reference_raw[2048]
        dynkin_d1 = main_dynkin[128] - main_dynkin[256]
        dynkin_d2 = main_dynkin[256] - main_dynkin[512]
        dynkin_d3 = main_dynkin[512] - main_dynkin[1024]
        dynkin_d4 = reference_dynkin[1024] - reference_dynkin[2048]
        main_path_ids = main_ids
        reference_path_ids = reference_ids
        path_id_pools = {
            "main": list(main_ids),
            "reference": list(reference_ids),
        }
    else:
        source_metadata = {}
        raw_values: list[np.ndarray] = []
        dynkin_values: list[np.ndarray] = []
        pair_path_ids: list[tuple[int, ...]] = []
        for coarse, fine in ADJACENT_LEVEL_PAIRS:
            pair_plan = plan[f"{coarse}-{fine}"]
            paths = tuple(int(value) for value in pair_plan["root_path_ids"])
            initial, source_metadata = _initial_states(
                parent, len(paths), device_value
            )
            raw_difference, dynkin_difference, records = _antithetic_pair(
                run_dir=root,
                panel=panel,
                role=role,
                path_ids=paths,
                initial=initial,
                root_seed=int(root_seed),
                coarse_steps=coarse,
                fine_steps=fine,
                jacobi_profile=jacobi_profile,
            )
            raw_values.append(raw_difference)
            dynkin_values.append(dynkin_difference)
            pair_path_ids.append(paths)
            all_records.extend(records)
        raw_d1, raw_d2, raw_d3, raw_d4 = raw_values
        dynkin_d1, dynkin_d2, dynkin_d3, dynkin_d4 = dynkin_values
        main_path_ids = pair_path_ids[0]
        reference_path_ids = pair_path_ids[-1]
        path_id_pools = {
            f"{coarse}-{fine}": list(paths)
            for (coarse, fine), paths in zip(
                ADJACENT_LEVEL_PAIRS, pair_path_ids
            )
        }
    elapsed_outer = time.perf_counter() - started
    peak_fraction = (
        torch.cuda.max_memory_allocated(device_value)
        / max(float(torch.cuda.get_device_properties(device_value).total_memory), 1.0)
    )
    execution = _execution_record(
        all_records, peak_memory_fraction=peak_fraction
    )
    # Authorize with the larger of committed shard timing and the observed
    # end-to-end panel wall time.  This cannot hide orchestration overhead.
    committed_elapsed = float(execution["elapsed_seconds"])
    authorizing_elapsed = max(committed_elapsed, elapsed_outer)
    execution["outer_wall_seconds"] = elapsed_outer
    execution["committed_timing_coverage_pass"] = int(
        committed_elapsed + 1.0e-9 >= elapsed_outer * 0.95
    )
    execution["complete_wall_upper_seconds"] = authorizing_elapsed
    execution["elapsed_seconds"] = authorizing_elapsed
    execution["outer_wall_rate"] = (
        float(execution["transition_count"]) / authorizing_elapsed
    )
    execution["conservative_rate"] = min(
        float(execution["conservative_rate"]),
        float(execution["outer_wall_rate"]),
    )
    execution["timing_coverage_pass"] = 1
    main = _flatten_differences((raw_d1, raw_d2, raw_d3))
    d3 = _flatten_differences((raw_d3,))
    d4 = _flatten_differences((raw_d4,))
    dynkin_main = _flatten_differences((dynkin_d1, dynkin_d2, dynkin_d3))
    dynkin_d3_flat = _flatten_differences((dynkin_d3,))
    dynkin_d4_flat = _flatten_differences((dynkin_d4,))
    all_pool_ids = [
        int(value)
        for values in path_id_pools.values()
        for value in values
    ]
    reserved = path_id_plan.get("reserved_production_slot")
    reserved_valid = (
        isinstance(reserved, Sequence)
        and not isinstance(reserved, (str, bytes))
        and len(reserved) == 2
    )
    isolation = bool(
        len(all_pool_ids) == len(set(all_pool_ids))
        and reserved_valid
        and all(
            not int(reserved[0]) <= value < int(reserved[1])
            for value in all_pool_ids
        )
    )
    panel_finite = all(
        np.isfinite(value).all()
        for value in (
            main,
            d3,
            d4,
            dynkin_main,
            dynkin_d3_flat,
            dynkin_d4_flat,
        )
    )
    evidence_flags = _panel_evidence_flags(
        execution,
        complete=True,
        finite=panel_finite,
        pilot_production_isolation=isolation,
    )
    candidates = build_power_candidates(
        profile=profile,
        main_differences=main,
        d3_differences=d3,
        d4_differences=d4,
        execution=execution,
        evidence_flags=evidence_flags,
    )
    arrays = {
        "raw_main_differences": main,
        "raw_d3_differences": d3,
        "raw_d4_differences": d4,
        "dynkin_main_differences": dynkin_main,
        "dynkin_d3_differences": dynkin_d3_flat,
        "dynkin_d4_differences": dynkin_d4_flat,
    }
    existing = evidence_path.is_file()
    payload_record = _payload(
        payload_path, arrays, require_existing=existing
    )
    record = {
        "schema": HAAR_POWER_VERSION + "-panel",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "profile": profile,
        "panel": panel,
        "root_seed": int(root_seed),
        "path_id_plan_sha256": path_id_plan["path_id_plan_sha256"],
        "main_path_ids": list(main_path_ids),
        "reference_path_ids": list(reference_path_ids),
        "path_id_pools": path_id_pools,
        "cluster_count": int(main.shape[0]),
        "main_feature_count": int(main.shape[1]),
        "reference_feature_count": 80,
        "source_npz_sha256": source_metadata.get("source_npz_sha256"),
        "observable_payload": payload_record,
        "execution": execution,
        "candidates": candidates,
        "complete": 1,
        "finite": int(panel_finite),
        "raw_endpoint_authorizing": 1,
        "dynkin_advisory_only": 1,
        "production_authorizing_pass": int(
            all(int(value) == 1 for value in evidence_flags.values())
            and isolation
        ),
        "raw_endpoint_authorizing_pass": 1,
        "dynkin_advisory_only_pass": 1,
        "independent_pool_variance_pass": int(
            isolation
        ),
        "richardson_formula_pass": int(
            all(
                float(row.get("independent_pool_covariance", math.inf)) == 0.0
                for row in candidates
            )
        ),
        **evidence_flags,
        "panel_means_excluded_from_refinement": 1,
        "production_authorizing": int(
            all(int(value) == 1 for value in evidence_flags.values())
            and isolation
        ),
        **NO_WORK,
    }
    normalized = json.loads(json.dumps(record, sort_keys=True, allow_nan=False))
    if existing:
        previous = json.loads(evidence_path.read_text(encoding="utf-8"))
        if previous != normalized:
            raise ArtifactCompatibilityError("sealed panel evidence changed")
        return previous
    atomic_write_json(evidence_path, normalized)
    return normalized


def panel_confirmation_record(
    panel: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = panel.get("candidates")
    if not isinstance(candidates, list):
        raise ArtifactCompatibilityError("sealed panel candidates are missing")
    key = (int(selected["main_paths"]), int(selected["reference_paths"]))
    row = next(
        (
            dict(value)
            for value in candidates
            if (
                int(value["main_paths"]),
                int(value["reference_paths"]),
            )
            == key
        ),
        None,
    )
    if row is None:
        raise ArtifactCompatibilityError("selected design is absent from panel B")
    execution = panel.get("execution")
    if not isinstance(execution, Mapping):
        raise ArtifactCompatibilityError("panel execution evidence is missing")
    return {
        "schema": HAAR_POWER_VERSION + "-sealed-confirmation",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "profile": panel["profile"],
        "panel": panel["panel"],
        "main_paths": key[0],
        "reference_paths": key[1],
        "complete_pass": int(panel.get("complete", 0) == 1),
        "finite_pass": int(panel.get("finite", 0) == 1),
        "certification_pass": int(
            float(execution.get("certificate_fraction", 0.0)) == 1.0
        ),
        "numerical_health_pass": int(
            int(row.get("panel_numerical_health_pass", 0)) == 1
        ),
        "mass_conservation_pass": int(
            int(row.get("mass_conservation_pass", 0)) == 1
        ),
        "shard_chain_pass": int(row.get("shard_chain_pass", 0)),
        "main_half_width": row["predicted_main_half_width"],
        "generator_reference_half_width": row[
            "predicted_generator_reference_half_width"
        ],
        "reference_stability_half_width": row[
            "predicted_reference_stability_half_width"
        ],
        "projected_hours": row["projected_hours"],
        "minimum_rate": row["conservative_rate"],
        "certificate_fraction": execution["certificate_fraction"],
        "fallback_fraction": execution["fallback_fraction"],
        "fallback_cost_fraction": execution["fallback_cost_fraction"],
        "peak_memory_fraction": execution["peak_memory_fraction"],
        "mass_error": execution["mass_error"],
        "production_authorizing_pass": int(
            panel.get("production_authorizing_pass", 0) == 1
        ),
        "raw_endpoint_authorizing_pass": int(
            panel.get("raw_endpoint_authorizing_pass", 0) == 1
        ),
        "dynkin_advisory_only_pass": int(
            panel.get("dynkin_advisory_only_pass", 0) == 1
        ),
        "independent_pool_variance_pass": int(
            panel.get("independent_pool_variance_pass", 0) == 1
        ),
        "richardson_formula_pass": int(
            panel.get("richardson_formula_pass", 0) == 1
        ),
        "pilot_production_isolation_pass": int(
            row.get("pilot_production_isolation_pass", 0)
        ),
        **{
            name: int(execution.get(name, -1)) for name in FORBIDDEN_COUNTS
        },
        "panel_evidence_sha256": hashlib.sha256(
            json.dumps(panel, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        **NO_WORK,
    }


def combine_certified_haar_power_panels(
    *,
    run_dir: str | Path,
    profile: str,
    selected: Mapping[str, Any],
    panel_a: Mapping[str, Any],
    panel_b: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the 16-cluster forecast from the two sealed raw arrays."""

    if panel_a.get("panel") != "a" or panel_b.get("panel") != "b":
        raise ArtifactCompatibilityError("combined panels have the wrong roles")
    if (
        panel_a.get("profile") != profile
        or panel_b.get("profile") != profile
        or selected.get("profile", profile) != profile
    ):
        raise ArtifactCompatibilityError("combined panel profiles differ")
    if (
        panel_a.get("path_id_plan_sha256")
        != panel_b.get("path_id_plan_sha256")
        or panel_a.get("root_seed") != panel_b.get("root_seed")
        or panel_a.get("source_npz_sha256")
        != panel_b.get("source_npz_sha256")
    ):
        raise ArtifactCompatibilityError("combined panel provenance differs")
    if (
        int(panel_a.get("cluster_count", -1)) <= 0
        or int(panel_a.get("cluster_count", -1))
        != int(panel_b.get("cluster_count", -2))
    ):
        raise ArtifactCompatibilityError("combined panel cluster counts differ")

    def bound_path_ids(record: Mapping[str, Any]) -> set[int]:
        pools = record.get("path_id_pools")
        if not isinstance(pools, Mapping) or not pools:
            raise ArtifactCompatibilityError("panel path-ID pools are missing")
        flattened = [
            int(value)
            for values in pools.values()
            for value in values
        ]
        if len(flattened) != len(set(flattened)):
            raise ArtifactCompatibilityError("panel path-ID pools overlap")
        return set(flattened)

    if not bound_path_ids(panel_a).isdisjoint(bound_path_ids(panel_b)):
        raise ArtifactCompatibilityError("sealed A/B path IDs overlap")
    root = Path(run_dir)

    def arrays(record: Mapping[str, Any]) -> dict[str, np.ndarray]:
        payload = record.get("observable_payload")
        if not isinstance(payload, Mapping):
            raise ArtifactCompatibilityError("panel payload binding is missing")
        path = root / str(payload.get("path", ""))
        if not path.is_file() or file_fingerprint(path) != payload.get("sha256"):
            raise ArtifactCompatibilityError("panel payload hash changed")
        with np.load(path, allow_pickle=False) as archive:
            result = {name: np.asarray(archive[name]) for name in archive.files}
        expected = payload.get("array_hashes")
        if not isinstance(expected, Mapping) or any(
            expected.get(name) != _array_hash(value)
            for name, value in result.items()
        ):
            raise ArtifactCompatibilityError("panel array hash changed")
        return result

    a, b = arrays(panel_a), arrays(panel_b)
    if set(a) != set(b):
        raise ArtifactCompatibilityError("panel payload schemas differ")
    combined = {
        name: np.concatenate((a[name], b[name]), axis=0)
        for name in sorted(a)
    }
    executions = [dict(panel_a["execution"]), dict(panel_b["execution"])]
    transition_count = sum(int(value["transition_count"]) for value in executions)
    elapsed = sum(float(value["elapsed_seconds"]) for value in executions)
    execution = {
        "transition_count": transition_count,
        "certified_count": sum(int(value["certified_count"]) for value in executions),
        "certificate_fraction": min(
            float(value["certificate_fraction"]) for value in executions
        ),
        "fallback_count": sum(int(value["fallback_count"]) for value in executions),
        "fallback_fraction": (
            sum(int(value["fallback_count"]) for value in executions)
            / transition_count
        ),
        "fallback_elapsed_seconds": sum(
            float(value["fallback_elapsed_seconds"]) for value in executions
        ),
        "fallback_cost_fraction": (
            sum(float(value["fallback_elapsed_seconds"]) for value in executions)
            / elapsed
        ),
        "elapsed_seconds": elapsed,
        "complete_wall_upper_seconds": sum(
            float(value.get("complete_wall_upper_seconds", math.inf))
            for value in executions
        ),
        "conservative_rate": min(
            float(value["conservative_rate"]) for value in executions
        ),
        "mass_error": max(float(value["mass_error"]) for value in executions),
        "peak_memory_fraction": max(
            float(value["peak_memory_fraction"]) for value in executions
        ),
        "state_updates_device_resident_pass": int(
            all(int(value["state_updates_device_resident_pass"]) == 1 for value in executions)
        ),
        "shard_chain_pass": int(
            all(int(value["shard_chain_pass"]) == 1 for value in executions)
        ),
        "timing_coverage_pass": int(
            all(int(value.get("timing_coverage_pass", 0)) == 1 for value in executions)
        ),
        **{
            name: sum(int(value[name]) for value in executions)
            for name in FORBIDDEN_COUNTS
        },
    }
    combined_finite = all(
        np.isfinite(value).all() for value in combined.values()
    )
    flags = _panel_evidence_flags(
        execution,
        complete=bool(
            panel_a.get("complete", 0) == 1
            and panel_b.get("complete", 0) == 1
        ),
        finite=combined_finite,
        pilot_production_isolation=True,
    )
    candidates = build_power_candidates(
        profile=profile,
        main_differences=combined["raw_main_differences"],
        d3_differences=combined["raw_d3_differences"],
        d4_differences=combined["raw_d4_differences"],
        execution=execution,
        evidence_flags=flags,
    )
    synthetic_panel = {
        "profile": profile,
        "panel": "combined",
        "complete": int(flags["panel_complete_pass"]),
        "finite": int(combined_finite),
        "execution": execution,
        "candidates": candidates,
        "production_authorizing_pass": int(
            panel_a.get("production_authorizing_pass", 0) == 1
            and panel_b.get("production_authorizing_pass", 0) == 1
            and all(int(value) == 1 for value in flags.values())
        ),
        "raw_endpoint_authorizing_pass": int(
            panel_a.get("raw_endpoint_authorizing_pass", 0) == 1
            and panel_b.get("raw_endpoint_authorizing_pass", 0) == 1
        ),
        "dynkin_advisory_only_pass": int(
            panel_a.get("dynkin_advisory_only_pass", 0) == 1
            and panel_b.get("dynkin_advisory_only_pass", 0) == 1
        ),
        "independent_pool_variance_pass": int(
            panel_a.get("independent_pool_variance_pass", 0) == 1
            and panel_b.get("independent_pool_variance_pass", 0) == 1
        ),
        "richardson_formula_pass": int(
            panel_a.get("richardson_formula_pass", 0) == 1
            and panel_b.get("richardson_formula_pass", 0) == 1
        ),
    }
    result = panel_confirmation_record(synthetic_panel, selected)
    result["panel_a_evidence_sha256"] = hashlib.sha256(
        json.dumps(panel_a, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result["panel_b_evidence_sha256"] = hashlib.sha256(
        json.dumps(panel_b, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result["combined_cluster_count"] = int(
        combined["raw_main_differences"].shape[0]
    )
    return result


__all__ = [
    "FORBIDDEN_COUNTS",
    "HAAR_POWER_VERSION",
    "HaarPowerError",
    "build_power_candidates",
    "combine_certified_haar_power_panels",
    "panel_confirmation_record",
    "run_certified_haar_power_panel",
    "verify_certified_haar_power_panel_evidence",
]
