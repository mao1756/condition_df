"""Restartable exact-K=512 panels for the physical coarse-signal witness.

This module is deliberately only an execution/persistence adapter around
``run_exact_multipath_shard``.  It does not alter the certified Jacobi
transition, the Rao--Blackwell label, its binary64 values, or the canonical
random-number namespace.

Only restart state and sufficient statistics are retained.  Selected labels
are folded immediately into compensated per-cell accumulators and are never
written as raw observations.  The full training-style capture (later states,
phase states, and certificate rows) is never written.  Final per-path means
have shape ``[paths, 4, 7, 392]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_json,
    config_fingerprint,
    file_fingerprint,
)
from mnist.d0_jacobi_rb_cuda import (
    JacobiRBCudaProfile,
    sample_alpha1_rb_transition_batch_cuda,
)
from mnist.d0_jacobi_rb_cuda_multipath import (
    EDGES_PER_PHASE,
    PATH_STATE_SIZE,
    PHASE_MATCHINGS,
    SHARD_STEPS,
    ExactMultipathCapturePayload,
    run_exact_multipath_shard,
)


PANEL_EXECUTION_SCHEMA = "jacobi-rb-physical-coarse-panel-v2"
OUTER_STEPS = 512
SELECTED_OUTER_STEPS = tuple(range(15, OUTER_STEPS, 16))
PANEL_PATH_COUNT = 64
GROUP_SIZE = 8
QUARTILE_COUNT = 4
PHASE_COUNT = len(PHASE_MATCHINGS)
CELL_SHAPE = (QUARTILE_COUNT, PHASE_COUNT, EDGES_PER_PHASE)
FORBIDDEN_COUNTS = (
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
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
}


class PhysicalPanelError(RuntimeError):
    """A physical panel failed its execution or persistence contract."""

    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "physical_panel_execution",
        failure_code: str = "physical_panel_execution_failed",
    ) -> None:
        super().__init__(message)
        self.failure_domain = str(failure_domain)
        self.failure_code = str(failure_code)


@dataclass(frozen=True)
class PhysicalPanelRunResult:
    """One complete panel and its compact authorizing statistic."""

    panel: str
    path_ids: tuple[int, ...]
    cell_means: np.ndarray
    cell_means_path: Path
    metrics_path: Path
    metrics: Mapping[str, Any]


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(value)).tobytes(order="C")
    ).hexdigest()


def _accumulator_sha256(
    cell_sums: np.ndarray,
    cell_compensations: np.ndarray,
    cell_counts: np.ndarray,
) -> str:
    return config_fingerprint(
        {
            "cell_sums": _array_sha256(cell_sums),
            "cell_compensations": _array_sha256(cell_compensations),
            "cell_counts": _array_sha256(cell_counts),
        }
    )


def update_cell_accumulator(
    cell_sums: np.ndarray,
    cell_compensations: np.ndarray,
    cell_counts: np.ndarray,
    contribution: np.ndarray,
    *,
    quartile: int,
) -> None:
    """Fold one selected exact-label block into a Kahan accumulator."""

    if (
        cell_sums.dtype != np.float64
        or cell_compensations.dtype != np.float64
        or cell_sums.shape != (contribution.shape[0], *CELL_SHAPE)
        or cell_compensations.shape != cell_sums.shape
        or cell_counts.dtype != np.int16
        or cell_counts.shape != (QUARTILE_COUNT,)
        or contribution.dtype != np.float64
        or contribution.shape
        != (cell_sums.shape[0], PHASE_COUNT, EDGES_PER_PHASE)
        or not 0 <= int(quartile) < QUARTILE_COUNT
        or int(cell_counts[int(quartile)]) >= 8
    ):
        raise PhysicalPanelError(
            "physical-panel accumulator schema changed",
            failure_code="physical_panel_accumulator_invalid",
        )
    index = int(quartile)
    correction = contribution - cell_compensations[:, index]
    updated = cell_sums[:, index] + correction
    cell_compensations[:, index] = (
        updated - cell_sums[:, index]
    ) - correction
    cell_sums[:, index] = updated
    cell_counts[index] += 1


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {
        str(name): np.ascontiguousarray(np.asarray(value))
        for name, value in sorted(arrays.items())
    }
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **normalized)
    os.replace(temporary, path)
    return {
        "path": path.as_posix(),
        "sha256": file_fingerprint(path),
        "size": int(path.stat().st_size),
        "array_hashes": {
            name: _array_sha256(value) for name, value in normalized.items()
        },
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {
                name: np.array(archive[name], order="C", copy=True)
                for name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise ArtifactCompatibilityError(
            f"cannot read physical-panel NPZ {path}: {exc}"
        ) from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"cannot read physical-panel JSON {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(
            f"physical-panel JSON is not an object: {path}"
        )
    return dict(value)


def validate_selected_outer_steps(
    selected_outer_steps: Sequence[int] = SELECTED_OUTER_STEPS,
    *,
    outer_steps: int = OUTER_STEPS,
) -> tuple[int, ...]:
    """Validate the frozen equally populated quartile capture schedule."""

    selected = tuple(int(value) for value in selected_outer_steps)
    if (
        int(outer_steps) <= 0
        or int(outer_steps) % (QUARTILE_COUNT * SHARD_STEPS) != 0
        or not selected
        or len(selected) % QUARTILE_COUNT
        or len(set(selected)) != len(selected)
        or tuple(sorted(selected)) != selected
    ):
        raise ValueError("selected outer steps do not define four equal quartiles")
    counts = [0] * QUARTILE_COUNT
    for value in selected:
        if (
            not 0 <= value < int(outer_steps)
            or value % SHARD_STEPS != SHARD_STEPS - 1
        ):
            raise ValueError(
                "every selected outer step must end an eight-step shard"
            )
        counts[min(QUARTILE_COUNT - 1, value * QUARTILE_COUNT // outer_steps)] += 1
    if len(set(counts)) != 1 or counts[0] <= 0:
        raise ValueError("every time quartile must contain the same selected count")
    return selected


def selected_target_contribution(
    payload: ExactMultipathCapturePayload,
    *,
    selected_outer_step: int,
    expected_path_ids: Sequence[int],
) -> np.ndarray:
    """Extract one unmodified ``[path, phase, edge]`` RB-label contribution."""

    paths = tuple(int(value) for value in expected_path_ids)
    if tuple(payload.path_ids) != paths:
        raise PhysicalPanelError(
            "capture payload changed canonical path order",
            failure_code="physical_panel_capture_path_order_invalid",
        )
    outer = np.asarray(payload.outer_steps, dtype=np.int64)
    phases = np.asarray(payload.phases, dtype=np.int64)
    targets = np.asarray(payload.denoising_targets)
    expected_blocks = SHARD_STEPS * PHASE_COUNT
    if (
        outer.shape != (expected_blocks,)
        or phases.shape != (expected_blocks,)
        or targets.dtype != np.float64
        or targets.shape != (expected_blocks, len(paths), EDGES_PER_PHASE)
        or not np.isfinite(targets).all()
    ):
        raise PhysicalPanelError(
            "capture payload has an invalid target schema",
            failure_code="physical_panel_capture_schema_invalid",
        )
    indices = np.flatnonzero(outer == int(selected_outer_step))
    if (
        indices.size != PHASE_COUNT
        or not np.array_equal(phases[indices], np.arange(PHASE_COUNT))
    ):
        raise PhysicalPanelError(
            "capture payload lacks the selected complete seven-phase step",
            failure_code="physical_panel_capture_coverage_invalid",
        )
    # Payload paths are canonical and target axes are [block,path,edge].
    contribution = np.transpose(targets[indices], (1, 0, 2))
    return np.array(contribution, dtype=np.float64, order="C", copy=True)


def reduce_selected_contributions(
    contributions: Sequence[tuple[int, np.ndarray]],
    *,
    path_count: int,
    outer_steps: int = OUTER_STEPS,
    selected_outer_steps: Sequence[int] = SELECTED_OUTER_STEPS,
) -> np.ndarray:
    """Chronologically reduce sufficient statistics to ``[P,4,7,392]``."""

    selected = validate_selected_outer_steps(
        selected_outer_steps, outer_steps=outer_steps
    )
    if tuple(step for step, _value in contributions) != selected:
        raise PhysicalPanelError(
            "selected target contributions are missing or out of order",
            failure_code="physical_panel_contribution_chain_invalid",
        )
    sums = np.zeros((int(path_count), *CELL_SHAPE), dtype=np.float64)
    counts = np.zeros(QUARTILE_COUNT, dtype=np.int64)
    for outer_step, value in contributions:
        array = np.asarray(value)
        if (
            array.dtype != np.float64
            or array.shape != (int(path_count), PHASE_COUNT, EDGES_PER_PHASE)
            or not np.isfinite(array).all()
        ):
            raise PhysicalPanelError(
                "selected target contribution is invalid",
                failure_code="physical_panel_contribution_invalid",
            )
        quartile = min(
            QUARTILE_COUNT - 1, int(outer_step) * QUARTILE_COUNT // outer_steps
        )
        # Deliberately chronological binary64 addition; no centering, scaling,
        # clipping, weighting, or fitted variance reduction is performed.
        sums[:, quartile] += array
        counts[quartile] += 1
    expected_count = len(selected) // QUARTILE_COUNT
    if not np.all(counts == expected_count):
        raise PhysicalPanelError(
            "coarse cells do not have the frozen observation count",
            failure_code="physical_panel_cell_count_invalid",
        )
    means = sums / counts.reshape(1, QUARTILE_COUNT, 1, 1)
    if not np.isfinite(means).all():
        raise PhysicalPanelError(
            "coarse cell means are nonfinite",
            failure_code="physical_panel_cell_mean_nonfinite",
        )
    return np.ascontiguousarray(means)


def _shard_paths(
    panel_dir: Path, *, group_index: int, start_step: int
) -> tuple[Path, Path, Path]:
    group_dir = panel_dir / f"group-{int(group_index):02d}"
    stem = f"step-{int(start_step):03d}"
    return (
        group_dir / f"{stem}-state.npz",
        group_dir / f"{stem}-accumulator.npz",
        group_dir / f"{stem}.json",
    )


def _persist_shard(
    panel_dir: Path,
    *,
    panel: str,
    group_index: int,
    start_step: int,
    path_ids: Sequence[int],
    root_seed: int,
    input_states: np.ndarray,
    scientific_config_sha256: str,
    path_plan_sha256: str,
    profile_sha256: str,
    result: Any,
    input_accumulator_sha256: str,
    cell_sums: np.ndarray,
    cell_compensations: np.ndarray,
    cell_counts: np.ndarray,
    accumulator_expected: bool,
    complete_pipeline_started_at: float,
) -> dict[str, Any]:
    state_path, accumulator_path, metadata_path = _shard_paths(
        panel_dir, group_index=group_index, start_step=start_step
    )
    state_record = _atomic_npz(
        state_path,
        {"final_states": np.asarray(result.committed_final_states, dtype=np.float64)},
    )
    if not accumulator_expected and accumulator_path.exists():
        accumulator_path.unlink()
    accumulator_record = (
        None
        if not accumulator_expected
        else _atomic_npz(
            accumulator_path,
            {
                "cell_sums": cell_sums,
                "cell_compensations": cell_compensations,
                "cell_counts": cell_counts,
            },
        )
    )
    scheduler = result.to_record()
    record: dict[str, Any] = {
        "schema": PANEL_EXECUTION_SCHEMA + "-shard",
        "schema_version": 1,
        "identity": {
            "panel": str(panel),
            "group": int(group_index),
            "start_step": int(start_step),
        },
        "step_count": SHARD_STEPS,
        "path_ids": [int(value) for value in path_ids],
        "root_seed": int(root_seed),
        "scientific_config_sha256": str(scientific_config_sha256),
        "path_plan_sha256": str(path_plan_sha256),
        "profile_sha256": str(profile_sha256),
        "input_state_sha256": _array_sha256(input_states),
        "final_state_sha256": _array_sha256(result.committed_final_states),
        "input_accumulator_sha256": str(input_accumulator_sha256),
        "output_accumulator_sha256": _accumulator_sha256(
            cell_sums, cell_compensations, cell_counts
        ),
        "state_file": state_record,
        "accumulator_expected": int(accumulator_expected),
        "accumulator_file": accumulator_record,
        "raw_target_observations_persisted": 0,
        # Includes exact execution and both atomic NPZ writes.  The final
        # metadata replace itself is intentionally the commit marker.
        "complete_pipeline_elapsed_seconds": float(
            time.perf_counter() - complete_pipeline_started_at
        ),
        "scheduler_record": scheduler,
        "scheduler_record_sha256": config_fingerprint(scheduler),
        **NO_WORK,
    }
    record["semantic_sha256"] = config_fingerprint(record)
    atomic_write_json(metadata_path, record)
    return record


def _valid_committed_shard(
    panel_dir: Path,
    *,
    panel: str,
    group_index: int,
    start_step: int,
    path_ids: Sequence[int],
    root_seed: int,
    expected_input_states: np.ndarray,
    expected_cell_sums: np.ndarray,
    expected_cell_compensations: np.ndarray,
    expected_cell_counts: np.ndarray,
    accumulator_expected: bool,
    scientific_config_sha256: str,
    path_plan_sha256: str,
    profile_sha256: str,
) -> tuple[
    bool,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, Any] | None,
]:
    state_path, accumulator_path, metadata_path = _shard_paths(
        panel_dir, group_index=group_index, start_step=start_step
    )
    if not state_path.is_file() or not metadata_path.is_file():
        return False, None, None, None, None, None
    try:
        from mnist import d0_jacobi_rb_cuda_controls as controls

        record = _load_json(metadata_path)
        body = dict(record)
        claimed = body.pop("semantic_sha256", None)
        if claimed != config_fingerprint(body):
            return False, None, None, None, None, None
        if (
            record.get("schema") != PANEL_EXECUTION_SCHEMA + "-shard"
            or record.get("identity")
            != {
                "panel": str(panel),
                "group": int(group_index),
                "start_step": int(start_step),
            }
            or int(record.get("step_count", -1)) != SHARD_STEPS
            or record.get("path_ids") != [int(value) for value in path_ids]
            or int(record.get("root_seed", -1)) != int(root_seed)
            or record.get("scientific_config_sha256")
            != str(scientific_config_sha256)
            or record.get("path_plan_sha256") != str(path_plan_sha256)
            or record.get("profile_sha256") != str(profile_sha256)
            or int(record.get("accumulator_expected", -1))
            != int(accumulator_expected)
            or not math.isfinite(
                float(record.get("complete_pipeline_elapsed_seconds", math.nan))
            )
            or float(record.get("complete_pipeline_elapsed_seconds", 0.0)) <= 0.0
            or record.get("input_state_sha256")
            != _array_sha256(expected_input_states)
            or record.get("input_accumulator_sha256")
            != _accumulator_sha256(
                expected_cell_sums,
                expected_cell_compensations,
                expected_cell_counts,
            )
            or int(record.get("raw_target_observations_persisted", -1)) != 0
            or any(record.get(name) != 0 for name in NO_WORK)
        ):
            return False, None, None, None, None, None
        state_record = record.get("state_file")
        if not isinstance(state_record, Mapping):
            return False, None, None, None, None, None
        if (
            state_record.get("sha256") != file_fingerprint(state_path)
            or int(state_record.get("size", -1)) != state_path.stat().st_size
        ):
            return False, None, None, None, None, None
        state_arrays = _load_npz(state_path)
        if set(state_arrays) != {"final_states"}:
            return False, None, None, None, None, None
        final_states = state_arrays["final_states"]
        if (
            final_states.dtype != np.float64
            or final_states.shape != (len(path_ids), PATH_STATE_SIZE)
            or not np.isfinite(final_states).all()
            or np.any(final_states < 0.0)
            or record.get("final_state_sha256") != _array_sha256(final_states)
        ):
            return False, None, None, None, None, None
        hashes = state_record.get("array_hashes")
        if (
            not isinstance(hashes, Mapping)
            or hashes.get("final_states") != _array_sha256(final_states)
        ):
            return False, None, None, None, None, None
        scheduler = record.get("scheduler_record")
        if (
            not isinstance(scheduler, Mapping)
            or record.get("scheduler_record_sha256")
            != config_fingerprint(scheduler)
        ):
            return False, None, None, None, None, None
        diagnostics = scheduler.get("diagnostics")
        if (
            scheduler.get("schema")
            != "jacobi-rb-cuda-exact-multipath-v1-shard"
            or not isinstance(diagnostics, Mapping)
            or int(diagnostics.get("start_step", -1)) != int(start_step)
            or diagnostics.get("path_ids") != [int(value) for value in path_ids]
            or diagnostics.get("group_sizes") != [len(path_ids)]
            or int(diagnostics.get("transition_count", -1))
            != len(path_ids) * SHARD_STEPS * PHASE_COUNT * EDGES_PER_PHASE
            or int(diagnostics.get("phase_state_trace_enabled", -1))
            != int(accumulator_expected)
            or scheduler.get("batch_final_state_sha256")
            != controls._digest_arrays(final_states)
        ):
            return False, None, None, None, None, None
        path_records = scheduler.get("path_records")
        initial = np.asarray(expected_input_states, dtype=np.float64)
        if not isinstance(path_records, list) or len(path_records) != len(path_ids):
            return False, None, None, None, None, None
        for index, (path_id, path_record) in enumerate(
            zip(path_ids, path_records, strict=True)
        ):
            if (
                not isinstance(path_record, Mapping)
                or int(path_record.get("path_id", -1)) != int(path_id)
                or path_record.get("input_state_sha256")
                != controls._digest_arrays(initial[index])
                or path_record.get("final_state_sha256")
                != controls._digest_arrays(final_states[index])
            ):
                return False, None, None, None, None, None
        cell_sums = np.array(
            expected_cell_sums, dtype=np.float64, order="C", copy=True
        )
        cell_compensations = np.array(
            expected_cell_compensations,
            dtype=np.float64,
            order="C",
            copy=True,
        )
        cell_counts = np.array(
            expected_cell_counts, dtype=np.int16, order="C", copy=True
        )
        accumulator_record = record.get("accumulator_file")
        if accumulator_expected:
            if (
                not accumulator_path.is_file()
                or not isinstance(accumulator_record, Mapping)
                or accumulator_record.get("sha256")
                != file_fingerprint(accumulator_path)
                or int(accumulator_record.get("size", -1))
                != accumulator_path.stat().st_size
            ):
                return False, None, None, None, None, None
            arrays = _load_npz(accumulator_path)
            if set(arrays) != {
                "cell_compensations",
                "cell_counts",
                "cell_sums",
            }:
                return False, None, None, None, None, None
            cell_sums = arrays["cell_sums"]
            cell_compensations = arrays["cell_compensations"]
            cell_counts = arrays["cell_counts"]
            if (
                cell_sums.dtype != np.float64
                or cell_sums.shape != (len(path_ids), *CELL_SHAPE)
                or cell_compensations.dtype != np.float64
                or cell_compensations.shape != cell_sums.shape
                or cell_counts.dtype != np.int16
                or cell_counts.shape != (QUARTILE_COUNT,)
                or not np.isfinite(cell_sums).all()
                or not np.isfinite(cell_compensations).all()
                or np.any(cell_counts < 0)
                or np.any(cell_counts > 8)
                or accumulator_record.get("array_hashes")
                != {
                    name: _array_sha256(value)
                    for name, value in sorted(arrays.items())
                }
            ):
                return False, None, None, None, None, None
        elif accumulator_record is not None or accumulator_path.exists():
            return False, None, None, None, None, None
        expected_output_counts = np.array(
            expected_cell_counts, dtype=np.int16, order="C", copy=True
        )
        if accumulator_expected:
            expected_output_counts[
                (int(start_step) + SHARD_STEPS - 1) // (OUTER_STEPS // 4)
            ] += 1
        if not np.array_equal(cell_counts, expected_output_counts):
            return False, None, None, None, None, None
        if record.get("output_accumulator_sha256") != _accumulator_sha256(
            cell_sums, cell_compensations, cell_counts
        ):
            return False, None, None, None, None, None
        return (
            True,
            final_states,
            cell_sums,
            cell_compensations,
            cell_counts,
            record,
        )
    except (ArtifactCompatibilityError, KeyError, TypeError, ValueError):
        return False, None, None, None, None, None


def _aggregate_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    panel: str,
    path_count: int,
    invocation_wall_seconds: float,
) -> dict[str, Any]:
    diagnostics = [
        record["scheduler_record"]["diagnostics"] for record in records
    ]
    transitions = sum(int(item["transition_count"]) for item in diagnostics)
    certified = sum(int(item["certified_count"]) for item in diagnostics)
    fallback = sum(int(item["fallback_count"]) for item in diagnostics)
    fallback_seconds = sum(
        float(item["fallback_elapsed_seconds"]) for item in diagnostics
    )
    committed_seconds = sum(
        float(record["complete_pipeline_elapsed_seconds"]) for record in records
    )
    expected = int(path_count) * OUTER_STEPS * PHASE_COUNT * EDGES_PER_PHASE
    return {
        "schema": PANEL_EXECUTION_SCHEMA + "-metrics",
        "schema_version": 1,
        "panel": str(panel),
        "path_count": int(path_count),
        "group_count": int(path_count) // GROUP_SIZE,
        "shard_count": len(records),
        "expected_shard_count": int(path_count) // GROUP_SIZE
        * (OUTER_STEPS // SHARD_STEPS),
        "transition_count": transitions,
        "expected_transition_count": expected,
        "certified_count": certified,
        "uncertified_count": transitions - certified,
        "certificate_fraction": certified / transitions if transitions else 0.0,
        "fallback_count": fallback,
        "fallback_fraction": fallback / transitions if transitions else math.inf,
        "fallback_elapsed_seconds": fallback_seconds,
        "complete_pipeline_elapsed_seconds": committed_seconds,
        "current_invocation_elapsed_seconds": float(invocation_wall_seconds),
        "complete_pipeline_transitions_per_second": (
            transitions / committed_seconds if committed_seconds > 0.0 else math.inf
        ),
        "fallback_time_fraction": (
            fallback_seconds / committed_seconds
            if committed_seconds > 0.0
            else math.inf
        ),
        "maximum_mass_error": max(
            float(item["maximum_mass_error"]) for item in diagnostics
        ),
        "maximum_cuda_launch_lanes": max(
            int(item["maximum_cuda_launch_lanes"]) for item in diagnostics
        ),
        "state_updates_device_resident_pass": int(
            all(int(item["state_updates_device_resident"]) == 1 for item in diagnostics)
        ),
        "shard_chain_pass": 1,
        "target_modification_count": 0,
        "projection_count": 0,
        **NO_WORK,
        **{
            name: sum(int(item.get(name, 0)) for item in diagnostics)
            for name in FORBIDDEN_COUNTS
        },
    }


def run_physical_panel(
    run_dir: str | Path,
    *,
    panel: str,
    path_ids: Sequence[int],
    mixed_target: np.ndarray,
    root_seed: int,
    profile: JacobiRBCudaProfile,
    device: torch.device,
    scientific_config_sha256: str,
    path_plan_sha256: str,
    sampler: Callable[..., Any] = sample_alpha1_rb_transition_batch_cuda,
) -> PhysicalPanelRunResult:
    """Run or exactly resume one frozen 64-path physical panel."""

    paths = tuple(int(value) for value in path_ids)
    if (
        str(panel) not in {"a", "b"}
        or len(paths) != PANEL_PATH_COUNT
        or len(set(paths)) != len(paths)
        or any(not 0 <= value < (1 << 20) for value in paths)
    ):
        raise ValueError("a production physical panel requires 64 unique 20-bit IDs")
    if tuple(sorted(paths)) != paths:
        raise ValueError("physical panel path IDs must be in canonical order")
    target = np.asarray(mixed_target)
    if (
        target.dtype != np.float64
        or target.shape != (PATH_STATE_SIZE,)
        or not np.isfinite(target).all()
        or np.any(target < 0.0)
        or abs(float(target.sum()) - 1.0) > 1.0e-12
    ):
        raise ValueError("mixed_target must be a finite float64 simplex state")
    selected = validate_selected_outer_steps()
    root = Path(run_dir)
    panel_dir = root / "panels" / str(panel)
    panel_dir.mkdir(parents=True, exist_ok=True)
    profile_sha = config_fingerprint(profile.to_dict())
    all_group_means: list[tuple[tuple[int, ...], np.ndarray]] = []
    all_records: list[dict[str, Any]] = []
    stage_started = time.perf_counter()
    processed_shards = 0
    total_shards = (PANEL_PATH_COUNT // GROUP_SIZE) * (
        OUTER_STEPS // SHARD_STEPS
    )

    for group_index in range(PANEL_PATH_COUNT // GROUP_SIZE):
        group_paths = paths[
            group_index * GROUP_SIZE : (group_index + 1) * GROUP_SIZE
        ]
        current = np.repeat(target[None, :], GROUP_SIZE, axis=0)
        cell_sums = np.zeros((GROUP_SIZE, *CELL_SHAPE), dtype=np.float64)
        cell_compensations = np.zeros_like(cell_sums)
        cell_counts = np.zeros(QUARTILE_COUNT, dtype=np.int16)
        recompute_tail = False
        for start_step in range(0, OUTER_STEPS, SHARD_STEPS):
            shard_started = time.perf_counter()
            selected_step = start_step + SHARD_STEPS - 1
            capture_expected = selected_step in selected
            valid = False
            restored: np.ndarray | None = None
            restored_sums: np.ndarray | None = None
            restored_compensations: np.ndarray | None = None
            restored_counts: np.ndarray | None = None
            record: dict[str, Any] | None = None
            if not recompute_tail:
                (
                    valid,
                    restored,
                    restored_sums,
                    restored_compensations,
                    restored_counts,
                    record,
                ) = _valid_committed_shard(
                    panel_dir,
                    panel=panel,
                    group_index=group_index,
                    start_step=start_step,
                    path_ids=group_paths,
                    root_seed=root_seed,
                    expected_input_states=current,
                    expected_cell_sums=cell_sums,
                    expected_cell_compensations=cell_compensations,
                    expected_cell_counts=cell_counts,
                    accumulator_expected=capture_expected,
                    scientific_config_sha256=scientific_config_sha256,
                    path_plan_sha256=path_plan_sha256,
                    profile_sha256=profile_sha,
                )
            if valid:
                assert (
                    restored is not None
                    and restored_sums is not None
                    and restored_compensations is not None
                    and restored_counts is not None
                    and record is not None
                )
                # Always materialize a writable buffer before torch.as_tensor;
                # this avoids PyTorch's undefined non-writable NumPy behavior.
                current = np.array(restored, dtype=np.float64, order="C", copy=True)
                cell_sums = np.array(
                    restored_sums, dtype=np.float64, order="C", copy=True
                )
                cell_compensations = np.array(
                    restored_compensations,
                    dtype=np.float64,
                    order="C",
                    copy=True,
                )
                cell_counts = np.array(
                    restored_counts, dtype=np.int16, order="C", copy=True
                )
            else:
                recompute_tail = True
                input_accumulator_sha = _accumulator_sha256(
                    cell_sums, cell_compensations, cell_counts
                )
                states = torch.as_tensor(
                    np.array(current, dtype=np.float64, order="C", copy=True),
                    dtype=torch.float64,
                    device=device,
                ).contiguous()
                result = run_exact_multipath_shard(
                    states,
                    path_ids=group_paths,
                    start_step=start_step,
                    root_seed=int(root_seed),
                    profile=profile,
                    group_sizes=(GROUP_SIZE,),
                    sampler=sampler,
                    capture_training_payload=capture_expected,
                )
                if capture_expected:
                    if result.capture_payload is None:
                        raise PhysicalPanelError(
                            "selected shard returned no target capture",
                            failure_code="physical_panel_capture_missing",
                        )
                    contribution = selected_target_contribution(
                        result.capture_payload,
                        selected_outer_step=selected_step,
                        expected_path_ids=group_paths,
                    )
                    update_cell_accumulator(
                        cell_sums,
                        cell_compensations,
                        cell_counts,
                        contribution,
                        quartile=selected_step // (OUTER_STEPS // 4),
                    )
                record = _persist_shard(
                    panel_dir,
                    panel=panel,
                    group_index=group_index,
                    start_step=start_step,
                    path_ids=group_paths,
                    root_seed=root_seed,
                    input_states=current,
                    scientific_config_sha256=scientific_config_sha256,
                    path_plan_sha256=path_plan_sha256,
                    profile_sha256=profile_sha,
                    result=result,
                    input_accumulator_sha256=input_accumulator_sha,
                    cell_sums=cell_sums,
                    cell_compensations=cell_compensations,
                    cell_counts=cell_counts,
                    accumulator_expected=capture_expected,
                    complete_pipeline_started_at=shard_started,
                )
                current = np.array(
                    result.committed_final_states,
                    dtype=np.float64,
                    order="C",
                    copy=True,
                )
            assert record is not None
            all_records.append(record)
            processed_shards += 1
            if processed_shards % 8 == 0 or processed_shards == total_shards:
                progress_elapsed = time.perf_counter() - stage_started
                progress_rate = (
                    processed_shards / progress_elapsed
                    if progress_elapsed > 0.0
                    else math.inf
                )
                eta_seconds = (
                    (total_shards - processed_shards) / progress_rate
                    if progress_rate > 0.0
                    else math.inf
                )
                print(
                    f"physical panel {panel} shards "
                    f"{processed_shards}/{total_shards} "
                    f"elapsed={progress_elapsed:.1f}s eta={eta_seconds:.1f}s",
                    flush=True,
                )
        if not np.array_equal(
            cell_counts,
            np.full(QUARTILE_COUNT, len(selected) // QUARTILE_COUNT, dtype=np.int16),
        ):
            raise PhysicalPanelError(
                "physical-panel accumulator counts are incomplete",
                failure_code="physical_panel_accumulator_count_invalid",
            )
        group_means = np.ascontiguousarray(
            cell_sums / cell_counts.reshape(1, QUARTILE_COUNT, 1, 1)
        )
        group_path = panel_dir / f"group-{group_index:02d}-cell-means.npz"
        _atomic_npz(
            group_path,
            {
                "path_ids": np.asarray(group_paths, dtype=np.int64),
                "cell_means": group_means,
            },
        )
        all_group_means.append((group_paths, group_means))

    ordered_paths = tuple(
        path_id for group_paths, _means in all_group_means for path_id in group_paths
    )
    if ordered_paths != paths:
        raise AssertionError("panel groups changed canonical path order")
    cell_means = np.ascontiguousarray(
        np.concatenate([means for _group_paths, means in all_group_means], axis=0)
    )
    if cell_means.shape != (PANEL_PATH_COUNT, *CELL_SHAPE):
        raise AssertionError("assembled physical panel has the wrong cell shape")
    cell_means_path = panel_dir / "cell_means.npz"
    cell_record = _atomic_npz(
        cell_means_path,
        {
            "path_ids": np.asarray(paths, dtype=np.int64),
            "cell_means": cell_means,
        },
    )
    metrics = _aggregate_metrics(
        all_records,
        panel=panel,
        path_count=PANEL_PATH_COUNT,
        invocation_wall_seconds=time.perf_counter() - stage_started,
    )
    metrics.update(
        {
            "selected_outer_steps": list(selected),
            "observations_per_path_cell": len(selected) // QUARTILE_COUNT,
            "selected_observation_count": (
                PANEL_PATH_COUNT * len(selected)
            ),
            "expected_selected_observation_count": (
                PANEL_PATH_COUNT * len(selected)
            ),
            "raw_target_observations_persisted": 0,
            "accumulation_method": "online_kahan_float64",
            "cell_shape": list(cell_means.shape),
            "cell_means": cell_record,
            "all_shards_complete_pass": int(
                int(metrics["shard_count"])
                == int(metrics["expected_shard_count"])
            ),
            "cell_means_finite_pass": int(np.isfinite(cell_means).all()),
            "panel_path_order_pass": 1,
        }
    )
    metrics_path = panel_dir / "metrics.json"
    atomic_write_json(metrics_path, metrics)
    return PhysicalPanelRunResult(
        panel=str(panel),
        path_ids=paths,
        cell_means=cell_means,
        cell_means_path=cell_means_path,
        metrics_path=metrics_path,
        metrics=metrics,
    )


__all__ = [
    "CELL_SHAPE",
    "FORBIDDEN_COUNTS",
    "GROUP_SIZE",
    "OUTER_STEPS",
    "PANEL_EXECUTION_SCHEMA",
    "PANEL_PATH_COUNT",
    "PhysicalPanelError",
    "PhysicalPanelRunResult",
    "SELECTED_OUTER_STEPS",
    "reduce_selected_contributions",
    "run_physical_panel",
    "selected_target_contribution",
    "update_cell_accumulator",
    "validate_selected_outer_steps",
]
