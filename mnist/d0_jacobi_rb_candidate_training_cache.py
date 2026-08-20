"""Fresh K=512 eager-prefix evidence for the path-weighted Jacobi experiment.

The historical experiment runners depend on cache artifacts that are not part
of the repository snapshot.  This module builds a self-contained training or
validation cache from one mixed MNIST source image using the existing
approximate-candidate CUDA transition backend.

At outer steps 15, 31, ..., 511, every split phase is branched at the eight
midpoints ``(2j+1)/16``.  Prefix branches reuse the full transition's RNG key
and canonical transition IDs, so they share the underlying uniform stream.
The main forward state advances only through the full phase transition.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch

from mnist import eulerian_jacobi_ddpm as core
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    PHASE_COUNT,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    STATE_SIZE,
    ModelInputs,
    matching_indices,
)
from mnist.eulerian_jacobi_ddpm_candidate import (
    CandidateRuntime,
    candidate_forward_phase,
    candidate_forward_phase_prefixes,
    finish_candidate_outer_step,
)

CANDIDATE_PREFIX_CACHE_VERSION = "d0-jacobi-rb-candidate-prefix-cache-v1"
M8_PREFIX_FRACTIONS = tuple((2 * index + 1) / 16 for index in range(8))
K512_RECORD_OUTER_STEPS = tuple(range(15, 512, 16))
_ARRAY_DTYPES: Mapping[str, np.dtype[Any]] = {
    "later_states": np.dtype(np.float32),
    "reverse_time": np.dtype(np.float32),
    "phase": np.dtype(np.int8),
    "color": np.dtype(np.int8),
    "duration": np.dtype(np.float32),
    "labels": np.dtype(np.int8),
    "targets": np.dtype(np.float32),
    "path_ids": np.dtype(np.int64),
    "outer_steps": np.dtype(np.int16),
    "midpoint_indices": np.dtype(np.int8),
}


class CandidatePrefixCacheError(core.EulerianJacobiDDPMError):
    """The fresh cache contract was violated."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _unit_state(value: np.ndarray) -> np.ndarray:
    state = np.asarray(value, dtype=np.float64).reshape(-1)
    if state.shape != (STATE_SIZE,) or not np.isfinite(state).all() or np.any(state < 0.0):
        raise CandidatePrefixCacheError("source state must be finite nonnegative [784]")
    total = float(np.sum(state, dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise CandidatePrefixCacheError("source state must have positive mass")
    result = np.ascontiguousarray(state / total, dtype=np.float64)
    if abs(float(result.sum(dtype=np.float64)) - 1.0) > 2.0e-12:
        raise CandidatePrefixCacheError("source normalization failed")
    return result


def _canonical_path_ids(path_ids: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in path_ids)
    if (
        not result
        or len(set(result)) != len(result)
        or any(not 0 <= value < (1 << 20) for value in result)
    ):
        raise CandidatePrefixCacheError("path IDs must be unique unsigned 20-bit values")
    return result


@dataclass(frozen=True)
class CandidatePrefixCacheSpec:
    sample_steps: int = 512
    record_outer_steps: tuple[int, ...] = K512_RECORD_OUTER_STEPS
    prefix_fractions: tuple[float, ...] = M8_PREFIX_FRACTIONS

    def __post_init__(self) -> None:
        steps = int(self.sample_steps)
        record_steps = tuple(int(value) for value in self.record_outer_steps)
        fractions = tuple(float(value) for value in self.prefix_fractions)
        if steps not in {128, 512}:
            raise CandidatePrefixCacheError("cache sample_steps must be 128 or 512")
        if (
            not record_steps
            or len(set(record_steps)) != len(record_steps)
            or tuple(sorted(record_steps)) != record_steps
            or any(not 0 <= value < steps for value in record_steps)
        ):
            raise CandidatePrefixCacheError("cache record_outer_steps are invalid")
        if (
            not fractions
            or len(set(fractions)) != len(fractions)
            or tuple(sorted(fractions)) != fractions
            or any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in fractions)
        ):
            raise CandidatePrefixCacheError("cache prefix_fractions are invalid")

    @property
    def records_per_path(self) -> int:
        return len(self.record_outer_steps) * PHASE_COUNT * len(self.prefix_fractions)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["record_outer_steps"] = list(self.record_outer_steps)
        record["prefix_fractions"] = list(self.prefix_fractions)
        record["records_per_path"] = self.records_per_path
        return record


class CandidatePrefixCache:
    """Read-only memory-mapped evidence directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise CandidatePrefixCacheError(f"missing cache manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != CANDIDATE_PREFIX_CACHE_VERSION:
            raise CandidatePrefixCacheError("cache schema changed")
        self.length = int(self.manifest["record_count"])
        if self.length <= 0:
            raise CandidatePrefixCacheError("cache is empty")
        self._arrays: dict[str, np.ndarray] = {}
        shapes = {
            "later_states": (self.length, STATE_SIZE),
            "reverse_time": (self.length,),
            "phase": (self.length,),
            "color": (self.length,),
            "duration": (self.length,),
            "labels": (self.length,),
            "targets": (self.length, EDGES_PER_PHASE),
            "path_ids": (self.length,),
            "outer_steps": (self.length,),
            "midpoint_indices": (self.length,),
        }
        for name, shape in shapes.items():
            path = self.root / f"{name}.npy"
            if not path.is_file():
                raise CandidatePrefixCacheError(f"missing cache array: {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != _ARRAY_DTYPES[name]:
                raise CandidatePrefixCacheError(f"cache array {name} changed shape or dtype")
            self._arrays[name] = array
        terminal = np.load(self.root / "terminal_states.npy", mmap_mode="r", allow_pickle=False)
        expected_paths = int(self.manifest["path_count"])
        if terminal.shape != (expected_paths, STATE_SIZE) or terminal.dtype != np.float64:
            raise CandidatePrefixCacheError("terminal-state artifact is malformed")
        self.terminal_states = terminal

    def __len__(self) -> int:
        return self.length

    def array(self, name: str) -> np.ndarray:
        try:
            return self._arrays[name]
        except KeyError as exc:
            raise CandidatePrefixCacheError(f"unknown cache array {name!r}") from exc

    def batch(
        self,
        indices: np.ndarray | Sequence[int],
        *,
        device: str | torch.device,
    ) -> tuple[ModelInputs, torch.Tensor]:
        selected = np.asarray(indices, dtype=np.int64).reshape(-1)
        if selected.size == 0 or np.any((selected < 0) | (selected >= self.length)):
            raise CandidatePrefixCacheError("batch indices are empty or out of range")
        active_device = torch.device(device)
        inputs = ModelInputs(
            later_full_state=torch.as_tensor(
                np.asarray(self.array("later_states")[selected]),
                dtype=torch.float32,
                device=active_device,
            ),
            reverse_time=torch.as_tensor(
                np.asarray(self.array("reverse_time")[selected]),
                dtype=torch.float32,
                device=active_device,
            ),
            phase=torch.as_tensor(
                np.asarray(self.array("phase")[selected]),
                dtype=torch.long,
                device=active_device,
            ),
            color=torch.as_tensor(
                np.asarray(self.array("color")[selected]),
                dtype=torch.long,
                device=active_device,
            ),
            duration=torch.as_tensor(
                np.asarray(self.array("duration")[selected]),
                dtype=torch.float32,
                device=active_device,
            ),
            label=torch.as_tensor(
                np.asarray(self.array("labels")[selected]),
                dtype=torch.long,
                device=active_device,
            ),
        )
        target = torch.as_tensor(
            np.asarray(self.array("targets")[selected]),
            dtype=torch.float64,
            device=active_device,
        )
        return inputs, target

    def iter_indices(self, batch_size: int) -> Iterator[np.ndarray]:
        size = int(batch_size)
        if size <= 0:
            raise CandidatePrefixCacheError("batch_size must be positive")
        for start in range(0, self.length, size):
            yield np.arange(start, min(self.length, start + size), dtype=np.int64)

    def verify_hashes(self) -> dict[str, Any]:
        expected = dict(self.manifest.get("array_sha256", {}))
        observed: dict[str, str] = {}
        for name in tuple(_ARRAY_DTYPES) + ("terminal_states",):
            observed[name] = _sha256_file(self.root / f"{name}.npy")
        mismatches = {
            name: {"expected": expected.get(name), "observed": digest}
            for name, digest in observed.items()
            if expected.get(name) != digest
        }
        return {
            "schema": CANDIDATE_PREFIX_CACHE_VERSION + "-verification",
            "passed": int(not mismatches),
            "mismatches": mismatches,
            "observed_sha256": observed,
        }


def _allocate_arrays(root: Path, count: int) -> dict[str, np.memmap]:
    shapes = {
        "later_states": (count, STATE_SIZE),
        "reverse_time": (count,),
        "phase": (count,),
        "color": (count,),
        "duration": (count,),
        "labels": (count,),
        "targets": (count, EDGES_PER_PHASE),
        "path_ids": (count,),
        "outer_steps": (count,),
        "midpoint_indices": (count,),
    }
    return {
        name: np.lib.format.open_memmap(
            root / f"{name}.npy",
            mode="w+",
            dtype=_ARRAY_DTYPES[name],
            shape=shape,
        )
        for name, shape in shapes.items()
    }


def _flush_and_close_memmaps(values: Mapping[str, np.memmap]) -> None:
    """Flush and close NumPy memmaps without masking the original failure."""

    for value in values.values():
        try:
            value.flush()
        finally:
            mapping = getattr(value, "_mmap", None)
            if mapping is not None:
                mapping.close()


def _flush_and_close_memmap(value: np.memmap | None) -> None:
    if value is None:
        return
    try:
        value.flush()
    finally:
        mapping = getattr(value, "_mmap", None)
        if mapping is not None:
            mapping.close()


def build_candidate_prefix_cache(
    root: str | Path,
    source_state: np.ndarray,
    *,
    label: int,
    path_ids: Sequence[int],
    root_seed: int,
    runtime: CandidateRuntime,
    spec: CandidatePrefixCacheSpec | None = None,
    outer_step_callback: Callable[[Mapping[str, Any]], None] | None = None,
    overwrite: bool = False,
) -> CandidatePrefixCache:
    """Build one role's fresh evidence and terminal forward endpoints."""

    active_spec = spec or CandidatePrefixCacheSpec()
    source = _unit_state(source_state)
    paths = _canonical_path_ids(path_ids)
    digit = int(label)
    if not 0 <= digit <= 9:
        raise CandidatePrefixCacheError("label must be a decimal digit")
    target_root = Path(root)
    if (target_root / "manifest.json").is_file() and not overwrite:
        return CandidatePrefixCache(target_root)
    temporary = target_root.with_name(target_root.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)

    record_count = len(paths) * active_spec.records_per_path
    arrays: dict[str, np.memmap] = {}
    terminal: np.memmap | None = None
    selected_steps = set(active_spec.record_outer_steps)
    cursor = 0
    telemetry_rows: list[dict[str, Any]] = []
    try:
        arrays = _allocate_arrays(temporary, record_count)
        terminal = np.lib.format.open_memmap(
            temporary / "terminal_states.npy",
            mode="w+",
            dtype=np.float64,
            shape=(len(paths), STATE_SIZE),
        )
        for cohort_start in range(0, len(paths), 8):
            cohort_paths = paths[cohort_start : cohort_start + 8]
            state = torch.as_tensor(
                np.repeat(source[None, :], len(cohort_paths), axis=0),
                dtype=torch.float64,
                device=runtime.device,
            )
            for outer_step in range(active_spec.sample_steps):
                started = time.perf_counter()
                health_parts: list[Mapping[str, Any]] = []
                for phase in range(PHASE_COUNT):
                    if outer_step in selected_steps:
                        branches = candidate_forward_phase_prefixes(
                            state,
                            cohort_paths,
                            outer_step=outer_step,
                            phase=phase,
                            root_seed=int(root_seed),
                            sample_steps=active_spec.sample_steps,
                            prefix_fractions=active_spec.prefix_fractions,
                            runtime=runtime,
                        )
                        color = int(PHASE_MATCHINGS[phase])
                        duration = float(PHASE_DURATIONS[phase])
                        later_stack = (
                            torch.stack([branch[0] for branch in branches])
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                        target_stack = (
                            torch.stack([branch[1] for branch in branches])
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                        health_parts.extend(branch[2] for branch in branches)
                        for midpoint_index in range(len(branches)):
                            row_count = len(cohort_paths)
                            selected = slice(cursor, cursor + row_count)
                            prefix = active_spec.prefix_fractions[midpoint_index]
                            reverse_time = 1.0 - (
                                PHASE_COUNT * outer_step + phase + prefix
                            ) / (PHASE_COUNT * active_spec.sample_steps)
                            arrays["later_states"][selected] = later_stack[midpoint_index]
                            arrays["reverse_time"][selected] = np.float32(reverse_time)
                            arrays["phase"][selected] = np.int8(phase)
                            arrays["color"][selected] = np.int8(color)
                            arrays["duration"][selected] = np.float32(duration)
                            arrays["labels"][selected] = np.int8(digit)
                            arrays["targets"][selected] = target_stack[midpoint_index]
                            arrays["path_ids"][selected] = np.asarray(
                                cohort_paths, dtype=np.int64
                            )
                            arrays["outer_steps"][selected] = np.int16(outer_step)
                            arrays["midpoint_indices"][selected] = np.int8(midpoint_index)
                            cursor += row_count
                    state, _target, full_health = candidate_forward_phase(
                        state,
                        cohort_paths,
                        outer_step=outer_step,
                        phase=phase,
                        root_seed=int(root_seed),
                        sample_steps=active_spec.sample_steps,
                        runtime=runtime,
                    )
                    health_parts.append(full_health)
                record = finish_candidate_outer_step(
                    health_parts,
                    runtime=runtime,
                    direction="forward-prefix-cache",
                    outer_step=outer_step,
                    elapsed_started=started,
                )
                record["cohort_start"] = cohort_start
                telemetry_rows.append(record)
                if outer_step_callback is not None:
                    outer_step_callback(record)
            terminal[cohort_start : cohort_start + len(cohort_paths)] = (
                state.detach().cpu().numpy()
            )
        if cursor != record_count:
            raise CandidatePrefixCacheError(
                f"cache population is incomplete: {cursor} != {record_count}"
            )
        _flush_and_close_memmaps(arrays)
        arrays = {}
        _flush_and_close_memmap(terminal)
        terminal = None

        array_sha256 = {
            name: _sha256_file(temporary / f"{name}.npy")
            for name in tuple(_ARRAY_DTYPES) + ("terminal_states",)
        }
        telemetry = {
            "outer_step_records": len(telemetry_rows),
            "maximum_mass_error": max(
                (float(row["maximum_mass_error"]) for row in telemetry_rows), default=0.0
            ),
            "maximum_pair_total_error": max(
                (float(row["maximum_pair_total_error"]) for row in telemetry_rows),
                default=0.0,
            ),
            "candidate_maximum_bracket_width": max(
                (
                    float(row["candidate_maximum_bracket_width"])
                    for row in telemetry_rows
                ),
                default=0.0,
            ),
            "total_outer_step_seconds": float(
                math.fsum(float(row["outer_step_seconds"]) for row in telemetry_rows)
            ),
        }
        _atomic_json(temporary / "telemetry.json", telemetry)
        manifest = {
            "schema": CANDIDATE_PREFIX_CACHE_VERSION,
            "schema_version": 1,
            "spec": active_spec.to_record(),
            "record_count": record_count,
            "path_count": len(paths),
            "path_ids": list(paths),
            "label": digit,
            "root_seed": int(root_seed),
            "source_state_sha256": hashlib.sha256(
                source.astype("<f8", copy=False).tobytes(order="C")
            ).hexdigest(),
            "candidate_binary_sha256": runtime.candidate_binary_sha256,
            "candidate_backend": "cuda-approximate-candidate-128m-56b",
            "target_semantics": "approximate-candidate Rao--Blackwell target",
            "array_sha256": array_sha256,
            "array_dtypes": {name: value.str for name, value in _ARRAY_DTYPES.items()},
            "terminal_states_dtype": np.dtype(np.float64).str,
            "telemetry": telemetry,
        }
        _atomic_json(temporary / "manifest.json", manifest)
        if target_root.exists():
            if not overwrite:
                raise CandidatePrefixCacheError("cache destination appeared during build")
            shutil.rmtree(target_root)
        os.replace(temporary, target_root)
        return CandidatePrefixCache(target_root)
    except Exception:
        try:
            _flush_and_close_memmaps(arrays)
        except Exception:
            pass
        try:
            _flush_and_close_memmap(terminal)
        except Exception:
            pass
        raise


def cache_mobility_numpy(
    later_states: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    """Compute ``Y(1-Y)`` for cache rows without constructing model inputs."""

    state = np.asarray(later_states, dtype=np.float64)
    active_colors = np.asarray(colors, dtype=np.int64).reshape(-1)
    if state.ndim != 2 or state.shape != (active_colors.size, STATE_SIZE):
        raise CandidatePrefixCacheError("mobility inputs are misaligned")
    tails, heads = matching_indices(device="cpu")
    tails_np = tails.numpy()[active_colors]
    heads_np = heads.numpy()[active_colors]
    tail = np.take_along_axis(state, tails_np, axis=1)
    head = np.take_along_axis(state, heads_np, axis=1)
    pair = tail + head
    fraction = np.divide(head, pair, out=np.zeros_like(head), where=pair > 0.0)
    mobility = fraction * (1.0 - fraction)
    return np.ascontiguousarray(mobility, dtype=np.float64)


__all__ = [
    "CANDIDATE_PREFIX_CACHE_VERSION",
    "K512_RECORD_OUTER_STEPS",
    "M8_PREFIX_FRACTIONS",
    "CandidatePrefixCache",
    "CandidatePrefixCacheError",
    "CandidatePrefixCacheSpec",
    "build_candidate_prefix_cache",
    "cache_mobility_numpy",
]
