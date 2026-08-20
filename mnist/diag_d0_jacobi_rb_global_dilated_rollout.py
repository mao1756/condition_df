"""Exploratory global-dilated Jacobi/RB training and exact rollout.

This is deliberately a small experiment orchestrator rather than another
general artifact framework.  It trains the mobility-wrapped global predictor
against the stored Rao--Blackwell tangent target, freezes every evaluation
choice, and only then allocates and opens one fresh exact path.  The mandatory
objective is a paired five-row, 128-outer-step exact reverse suffix.

Restart boundaries are the existing 100-update training checkpoints, exact
eight-step forward shards, and exact eight-step fused reverse shards.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from mnist.d0_jacobi_artifacts import (
    atomic_write_json,
    config_fingerprint,
    file_fingerprint,
)
from mnist.d0_jacobi_rb_boundary_tangent import (
    direct_raw_target_mse,
    edge_pair_geometry,
)
from mnist.d0_jacobi_rb_boundary_tangent_fused import (
    frozen_score_logistic_fraction,
)
from mnist.d0_jacobi_rb_boundary_tangent_frequency1_coordinate_learnability import (
    MODEL_SEEDS,
    TRAINING,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_memory import (
    CanonicalRowSquareReducer,
    HostInputStore,
    HostLabelStore,
    LabelOpenAuthorization,
    ModelCallBatchGuard,
    open_external_input_store,
    open_external_label_store,
)
from mnist.d0_jacobi_rb_boundary_tangent_v3_provenance import (
    v3_transitive_source_paths,
)
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_global_dilated import (
    GLOBAL_DILATED_PARAMETER_COUNT,
    GlobalDilatedZeroBaselinePredictor,
    global_dilated_architecture_contract,
)
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    OUTER_STEPS,
    PHASE_COUNT,
    PHASE_DURATIONS,
    STATE_SIZE,
    ModelInputs,
    deterministic_batch_indices,
    enable_deterministic_torch,
    semantic_sha256,
    state_dict_sha256,
)
from mnist.d0_jacobi_rb_reverse_controller import internal_reverse_time
from mnist.d0_jacobi_rb_tangent_fused import (
    DeferredCertifiedFusedReference,
    FUSED_SHARD_PHASES,
    FusedRowSpec,
    FusedShardExecutionPlan,
    FusedTangentControllerBank,
    build_fused_transition_id_plan,
    prepare_deferred_reference_rng_seed_map,
    run_fused_reverse_family,
)
from mnist.d0_jacobi_rb_tangent_rollout import (
    FixedRenderingScale,
    ScaledTangentScoreController,
    SignedDiagnosticTangentScoreController,
    TargetFractionOracleController,
    ZeroTangentScoreController,
    atomic_rollout_npz,
    fixed_rendering_scale,
    load_verified_frequency1_checkpoint,
    load_verified_source_target,
    paired_metric_improvement,
    raw_state_metrics,
    render_background_demixed,
    render_raw_density,
    reverse_suffix_sequence,
    rollout_array_sha256,
    rollout_file_sha256,
    rollout_semantic_record,
    run_forward_trajectory,
    save_png,
)
from mnist.diag_d0_jacobi_rb_frequency1_rollout import (
    _verify_fused_load_bearing_row_telemetry,
    _verify_parent_registry as _verify_frequency1_parent_registry,
)


VERSION = "d0-jacobi-rb-global-dilated-rollout-v1"
RUN_SCHEMA = "experiment12-d0-jacobi-rb-global-dilated-rollout"
V4_BASENAME = "20260813-002414_production-frequency1-objective-first-recovery-v4"
V4_MANIFEST_FILE_SHA256 = (
    "d7f3520c0d0a6d9254cf303a1d20f6d1250280cbb26e2deec365e7bf22b549e4"
)
V4_MANIFEST_SEMANTIC_SHA256 = (
    "5747765518aeda7ae0b20a49dd681880e29ac417f42c72b5c353521799ee7f98"
)
V4_MANIFEST_COUNT = 381
SOURCE_IMAGE_MEASURE_SHA256 = (
    "0bb39fec59853f789fe366251cd85ed79ffbffb5a1aaa32084d2dbd2bbb4ea7d"
)
MIXED_TARGET_MEASURE_SHA256 = (
    "00ae86fb69be6d86557f15f6f8fa00f8bb3c2514f331863c9638e36d23d135c5"
)
STAGES = (
    "prepare",
    "controls",
    "train_select_freeze",
    "evaluate_exact",
    "report_verify",
)

TRAINING_SEED = 261_372
FORWARD_ROOT_SEED = 261_401
REVERSE_ROOT_SEED = 261_402
MICROSTEPS = 2
ANCHOR_STEP = 127
MANDATORY_SUFFIX_STEPS = 128
CHECKPOINT_INTERVAL = 100
CAP_LADDER = (4_000, 3_000, 2_000, 1_000)
FRESH_PATH_POOL = tuple(range(0xFB300, 0xFB310))
PREFLIGHT_PATH_POOL = tuple(range(0xFB2F0, 0xFB300))
ROW_ORDER = (
    "zero",
    "v4-plus-0p5",
    "v4-minus-0p5",
    "global-plus-1",
    "source-informed",
)
MILESTONE_STEPS = (0, 32, 64, 96, 128)

ACTIVE_SECONDS_CAP = 21_600.0
STORAGE_CAP_BYTES = 2 * 1024**3
CUDA_MEMORY_FRACTION_CAP = 0.80
REPORT_RESERVE_SECONDS = 600.0
FORWARD_RESERVE_SECONDS = 900.0
OPTIONAL_POSTPROCESS_TIME_MULTIPLIER = 1.50
OPTIONAL_POSTPROCESS_MIN_SECONDS = 30.0
OPTIONAL_POSTPROCESS_MIN_STORAGE_BYTES = 16 * 1024**2
PRACTICAL_RELATIVE_THRESHOLD = 0.01

DEFAULT_TRAINING_PARENT = Path(
    "runs/experiment12_d0_jacobi_rb_boundary_tangent_frequency1_coordinate_learnability/"
    "20260811-010641_production-frequency1-coordinate-v1-one-image"
)
DEFAULT_V4_RUN = Path(
    "runs/experiment12_d0_jacobi_rb_frequency1_rollout/"
    "20260813-002414_production-frequency1-objective-first-recovery-v4"
)

_PREPARED_BACKENDS: dict[str, Any] = {}
_PREPARED_BACKEND_SECONDS: dict[str, float] = {}
_PREPARED_SEED_MAPS: dict[tuple[int, int, str], Mapping[str, Any]] = {}

_FUSED_INVALID_FIELDS = (
    "input_invalid",
    "reference_fraction_invalid",
    "score_invalid",
    "logistic_shift_invalid",
    "state_invalid",
    "mass_invalid",
    "metadata_invalid",
    "reference_invalid_count",
    "reference_unauthorized_count",
)
_FORBIDDEN_EXACT_COUNTS = (
    "resource_cap_count",
    "invalid_density_count",
    "approximation_count",
    "clipping_count",
    "correction_count",
    "floor_count",
    "limiter_count",
    "projection_count",
    "renormalization_count",
    "nonfinite_count",
)
_RAW_SUFFIX_CONVERSION_INTENT = "raw_suffix_conversion_intent.json"


class GlobalDilatedRolloutError(RuntimeError):
    """A frozen experiment, integrity, numerical, or resource contract failed."""


class ResourceBoundaryError(GlobalDilatedRolloutError):
    """A trusted resource admission or durably measured cap boundary stopped work."""


class PostprocessingResourceStopError(GlobalDilatedRolloutError):
    """Raw exact suffix is complete, but cap exhaustion forbids derived replay."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: str | Path, *, semantic: bool = False) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GlobalDilatedRolloutError(f"cannot read JSON artifact: {source}") from exc
    if not isinstance(value, dict):
        raise GlobalDilatedRolloutError(f"JSON artifact is not an object: {source}")
    if semantic:
        body = dict(value)
        recorded = body.pop("semantic_sha256", None)
        if (
            not isinstance(recorded, str)
            or len(recorded) != 64
            or any(character not in "0123456789abcdef" for character in recorded)
            or semantic_sha256(body) != recorded
        ):
            raise GlobalDilatedRolloutError(f"semantic hash changed: {source}")
    return value


def _semantic(body: Mapping[str, Any]) -> dict[str, Any]:
    return rollout_semantic_record(body)


def _write_semantic(path: str | Path, body: Mapping[str, Any]) -> dict[str, Any]:
    record = _semantic(body)
    atomic_write_json(path, record)
    return record


def _atomic_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(destination, output.getvalue())


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True, order="C") for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise GlobalDilatedRolloutError(f"cannot open NPZ artifact: {source}") from exc


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _source_revision(repository_root: Path) -> dict[str, Any]:
    def command(*argv: str) -> str | None:
        try:
            return subprocess.run(
                argv,
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = command("git", "rev-parse", "HEAD")
    status = command("git", "status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else int(bool(status)),
        "status_sha256": None if status is None else hashlib.sha256(status.encode()).hexdigest(),
    }


def _resource_ledger(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "resource_ledger.json"
    if path.is_file():
        value = _read_json(path, semantic=True)
        if value.get("schema") != VERSION + "-resource-ledger":
            raise GlobalDilatedRolloutError("resource ledger schema changed")
        events = value.get("events")
        allowed_breaches = {
            "active_seconds_cap",
            "cuda_memory_fraction_cap",
            "persisted_storage_cap",
        }
        event_keys = {
            "event_id",
            "role",
            "elapsed_seconds",
            "peak_cuda_memory_bytes",
            "total_cuda_memory_bytes",
            "detail",
            "recorded_at",
            "limits_passed",
            "breaches",
        }
        if not isinstance(events, list):
            raise GlobalDilatedRolloutError("resource ledger events changed")
        seen: set[str] = set()
        for event in events:
            if not isinstance(event, Mapping) or set(event) != event_keys:
                raise GlobalDilatedRolloutError("resource ledger event schema changed")
            elapsed = event.get("elapsed_seconds")
            peak = event.get("peak_cuda_memory_bytes")
            total = event.get("total_cuda_memory_bytes")
            detail = event.get("detail")
            breaches = event.get("breaches")
            identifier = event.get("event_id")
            expected_id = semantic_sha256({"role": event.get("role"), "detail": detail})
            if (
                not isinstance(event.get("role"), str)
                or not event.get("role")
                or not isinstance(detail, Mapping)
                or not isinstance(elapsed, (int, float))
                or isinstance(elapsed, bool)
                or not math.isfinite(float(elapsed))
                or float(elapsed) < 0.0
                or not isinstance(peak, int)
                or isinstance(peak, bool)
                or peak < 0
                or not isinstance(total, int)
                or isinstance(total, bool)
                or total < 0
                or (total > 0 and peak > total)
                or identifier != expected_id
                or identifier in seen
                or not isinstance(breaches, list)
                or any(item not in allowed_breaches for item in breaches)
                or event.get("limits_passed") != int(not breaches)
            ):
                raise GlobalDilatedRolloutError("resource ledger event authority changed")
            seen.add(str(identifier))
        active = math.fsum(float(item["elapsed_seconds"]) for item in events)
        peak = max((int(item["peak_cuda_memory_bytes"]) for item in events), default=0)
        total = max((int(item["total_cuda_memory_bytes"]) for item in events), default=0)
        persisted = value.get("persisted_storage_bytes")
        aggregate_breaches: list[str] = []
        if active > ACTIVE_SECONDS_CAP + 1.0e-9:
            aggregate_breaches.append("active_seconds_cap")
        if total > 0 and peak / total >= CUDA_MEMORY_FRACTION_CAP:
            aggregate_breaches.append("cuda_memory_fraction_cap")
        if isinstance(persisted, int) and persisted >= STORAGE_CAP_BYTES:
            aggregate_breaches.append("persisted_storage_cap")
        if (
            float(value.get("active_seconds", math.nan)) != active
            or value.get("maximum_peak_cuda_memory_bytes") != peak
            or value.get("maximum_total_cuda_memory_bytes") != total
            or value.get("active_seconds_cap") != ACTIVE_SECONDS_CAP
            or value.get("storage_cap_bytes") != STORAGE_CAP_BYTES
            or value.get("cuda_memory_fraction_cap") != CUDA_MEMORY_FRACTION_CAP
            or value.get("breaches") != aggregate_breaches
            or value.get("limits_passed") != int(not aggregate_breaches)
            or not isinstance(persisted, int)
            or isinstance(persisted, bool)
            or int(persisted) < 0
        ):
            raise GlobalDilatedRolloutError("resource ledger aggregate authority changed")
        return value
    return _semantic(
        {
            "schema": VERSION + "-resource-ledger",
            "schema_version": 1,
            "active_seconds_cap": ACTIVE_SECONDS_CAP,
            "storage_cap_bytes": STORAGE_CAP_BYTES,
            "cuda_memory_fraction_cap": CUDA_MEMORY_FRACTION_CAP,
            "events": [],
            "active_seconds": 0.0,
            "maximum_peak_cuda_memory_bytes": 0,
            "maximum_total_cuda_memory_bytes": 0,
            "persisted_storage_bytes": _directory_bytes(run_dir),
            "limits_passed": 1,
            "breaches": [],
        }
    )


def _record_resource_event(
    run_dir: Path,
    *,
    role: str,
    elapsed_seconds: float,
    peak_cuda_memory_bytes: int = 0,
    total_cuda_memory_bytes: int = 0,
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current = _resource_ledger(run_dir)
    events = [dict(item) for item in current.get("events", [])]
    event_detail = dict(detail or {})
    event_id = semantic_sha256({"role": str(role), "detail": event_detail})
    if any(item.get("event_id") == event_id for item in events):
        if current.get("limits_passed") != 1:
            raise ResourceBoundaryError(
                "resource limit remains crossed in the durable ledger: "
                + ", ".join(str(item) for item in current.get("breaches", []))
            )
        return current
    events.append(
        {
            "event_id": event_id,
            "role": str(role),
            "elapsed_seconds": float(elapsed_seconds),
            "peak_cuda_memory_bytes": int(peak_cuda_memory_bytes),
            "total_cuda_memory_bytes": int(total_cuda_memory_bytes),
            "detail": event_detail,
            "recorded_at": _utc_now(),
        }
    )
    active = math.fsum(float(item["elapsed_seconds"]) for item in events)
    peak = max((int(item["peak_cuda_memory_bytes"]) for item in events), default=0)
    total = max((int(item["total_cuda_memory_bytes"]) for item in events), default=0)
    persisted = _directory_bytes(run_dir)
    breaches: list[str] = []
    if active > ACTIVE_SECONDS_CAP + 1.0e-9:
        breaches.append("active_seconds_cap")
    if total > 0 and peak / total >= CUDA_MEMORY_FRACTION_CAP:
        breaches.append("cuda_memory_fraction_cap")
    if persisted >= STORAGE_CAP_BYTES:
        breaches.append("persisted_storage_cap")
    events[-1]["limits_passed"] = int(not breaches)
    events[-1]["breaches"] = breaches
    record = _write_semantic(
        run_dir / "resource_ledger.json",
        {
            "schema": VERSION + "-resource-ledger",
            "schema_version": 1,
            "active_seconds_cap": ACTIVE_SECONDS_CAP,
            "storage_cap_bytes": STORAGE_CAP_BYTES,
            "cuda_memory_fraction_cap": CUDA_MEMORY_FRACTION_CAP,
            "events": events,
            "active_seconds": active,
            "maximum_peak_cuda_memory_bytes": peak,
            "maximum_total_cuda_memory_bytes": total,
            "persisted_storage_bytes": persisted,
            "limits_passed": int(not breaches),
            "breaches": breaches,
        },
    )
    if breaches:
        raise ResourceBoundaryError(
            "resource limit crossed and durably recorded: " + ", ".join(breaches)
        )
    return record


@contextmanager
def _timed_resource(run_dir: Path, role: str) -> Iterator[dict[str, Any]]:
    started = time.perf_counter()
    attempt = sum(
        1
        for item in _resource_ledger(run_dir).get("events", [])
        if item.get("role") == role
    )
    detail: dict[str, Any] = {"attempt": attempt}
    try:
        yield detail
    except Exception:
        detail["failed"] = 1
        detail["failure_accounted"] = 1
        try:
            _record_resource_event(
                run_dir,
                role=role,
                elapsed_seconds=time.perf_counter() - started,
                detail=detail,
            )
        except ResourceBoundaryError:
            # The boundary is durably recorded, but a simultaneous integrity
            # exception remains the failure authority and must not be upgraded
            # into same-run resume authorization merely by accounting order.
            pass
        raise
    else:
        _record_resource_event(
            run_dir,
            role=role,
            elapsed_seconds=time.perf_counter() - started,
            detail=detail,
        )


def _maximum_numeric_key(value: Any, keys: set[str]) -> float:
    found: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if key in keys and isinstance(child, (int, float)) and math.isfinite(float(child)):
                    found.append(float(child))
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return max(found, default=0.0)


def _reconcile_durable_attempt_journal(
    run_dir: Path,
    *,
    journal_relative: str,
    role: str,
) -> dict[str, Any] | None:
    """Charge an abandoned hard-crash attempt before replaying its work."""

    journal_path = run_dir / journal_relative
    if not journal_path.is_file():
        return None
    journal = _read_json(journal_path, semantic=True)
    if (
        journal.get("schema") != VERSION + "-durable-attempt"
        or journal.get("role") != role
        or not isinstance(journal.get("attempt"), int)
        or isinstance(journal.get("attempt"), bool)
        or not isinstance(journal.get("attempt_id"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", str(journal.get("attempt_id")))
        or not isinstance(journal.get("started_unix_seconds"), (int, float))
        or isinstance(journal.get("started_unix_seconds"), bool)
        or not math.isfinite(float(journal["started_unix_seconds"]))
        or not isinstance(journal.get("detail"), Mapping)
    ):
        raise GlobalDilatedRolloutError("durable attempt journal authority changed")
    attempt_id = str(journal["attempt_id"])
    ledger = _resource_ledger(run_dir)
    covered = any(
        event.get("detail", {}).get("durable_attempt_id") == attempt_id
        for event in ledger.get("events", [])
    )
    if not covered:
        durable_root_relative = journal["detail"].get("durable_root_relative")
        durable_roots_relative = journal["detail"].get("durable_roots_relative")
        durable_minimum_start = journal["detail"].get("durable_minimum_start_step")
        durable_committed = 0.0
        if isinstance(durable_root_relative, str) and durable_root_relative:
            if durable_minimum_start is not None and (
                not isinstance(durable_minimum_start, int)
                or isinstance(durable_minimum_start, bool)
                or durable_minimum_start < 0
            ):
                raise GlobalDilatedRolloutError(
                    "durable attempt journal minimum shard step changed"
                )
            durable_committed = _committed_shard_elapsed(
                run_dir / durable_root_relative,
                minimum_start_step=durable_minimum_start,
            )
        elif durable_roots_relative is not None:
            if (
                not isinstance(durable_roots_relative, list)
                or not durable_roots_relative
                or any(
                    not isinstance(relative, str) or not relative
                    for relative in durable_roots_relative
                )
            ):
                raise GlobalDilatedRolloutError(
                    "durable attempt journal root list changed"
                )
            durable_committed = math.fsum(
                _committed_shard_elapsed(run_dir / relative)
                for relative in durable_roots_relative
            )
        wall_upper_bound = max(
            time.time() - float(journal["started_unix_seconds"]), 0.0
        )
        try:
            _record_resource_event(
                run_dir,
                role=role + "_abandoned_attempt",
                elapsed_seconds=wall_upper_bound + 5.0,
                detail={
                    "attempt": int(journal["attempt"]),
                    "durable_attempt_id": attempt_id,
                    "journal_relative": journal_relative,
                    "abandoned_hard_crash": 1,
                    "wall_to_resume_upper_bound_seconds": wall_upper_bound,
                    "unknown_active_interval_seconds": wall_upper_bound,
                    "idle_or_powered_off_time_may_be_included": 1,
                    "accounting_classification": (
                        "conservative_upper_bound_not_measured_active_compute"
                    ),
                    "conservative_commit_overhead_seconds": 5.0,
                    "durable_committed_shard_seconds": durable_committed,
                    "original_detail": dict(journal["detail"]),
                },
            )
        except ResourceBoundaryError:
            covered_after_breach = any(
                event.get("detail", {}).get("durable_attempt_id") == attempt_id
                for event in _resource_ledger(run_dir).get("events", [])
            )
            if covered_after_breach:
                journal_path.unlink(missing_ok=True)
            raise
    journal_path.unlink(missing_ok=True)
    if covered and ledger.get("limits_passed") != 1:
        # The resource event committed before a hard death at journal cleanup.
        # Retire the now-covered journal, but preserve the cap boundary as the
        # caller's routing authority instead of silently replaying work.
        raise ResourceBoundaryError(
            "resource limit remains crossed in the durable covered-attempt ledger: "
            + ", ".join(str(item) for item in ledger.get("breaches", []))
        )
    return journal


def _begin_durable_attempt(
    run_dir: Path,
    *,
    journal_relative: str,
    role: str,
    detail: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    _reconcile_durable_attempt_journal(
        run_dir, journal_relative=journal_relative, role=role
    )
    attempt = sum(
        1
        for event in _resource_ledger(run_dir).get("events", [])
        if event.get("role") in {role, role + "_abandoned_attempt"}
    )
    started_perf = time.perf_counter()
    started_unix = time.time()
    attempt_id = semantic_sha256(
        {
            "role": role,
            "attempt": attempt,
            "started_unix_seconds": started_unix,
            "detail": dict(detail),
        }
    )
    record = _write_semantic(
        run_dir / journal_relative,
        {
            "schema": VERSION + "-durable-attempt",
            "schema_version": 1,
            "role": role,
            "attempt": attempt,
            "attempt_id": attempt_id,
            "started_unix_seconds": started_unix,
            "detail": dict(detail),
        },
    )
    return record, started_perf


def _finish_durable_attempt(
    run_dir: Path,
    *,
    journal_relative: str,
    role: str,
    journal: Mapping[str, Any],
    elapsed_seconds: float,
    failed: bool,
    peak_cuda_memory_bytes: int = 0,
    total_cuda_memory_bytes: int = 0,
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current = _read_json(run_dir / journal_relative, semantic=True)
    if current != journal or current.get("role") != role:
        raise GlobalDilatedRolloutError("durable attempt journal changed while active")
    try:
        result = _record_resource_event(
            run_dir,
            role=role,
            elapsed_seconds=float(elapsed_seconds),
            peak_cuda_memory_bytes=peak_cuda_memory_bytes,
            total_cuda_memory_bytes=total_cuda_memory_bytes,
            detail={
                "attempt": int(journal["attempt"]),
                "durable_attempt_id": str(journal["attempt_id"]),
                "failed": int(failed),
                **dict(detail or {}),
            },
        )
    except ResourceBoundaryError:
        # `_record_resource_event` commits the cap-crossing event before it
        # raises.  Once that durable event covers this attempt, the start
        # journal is no longer live; retaining it would double-debit the same
        # interval if failure packaging itself later resumes.
        ledger = _resource_ledger(run_dir)
        covered = any(
            event.get("detail", {}).get("durable_attempt_id")
            == str(journal["attempt_id"])
            for event in ledger.get("events", [])
        )
        if covered:
            (run_dir / journal_relative).unlink(missing_ok=True)
        raise
    (run_dir / journal_relative).unlink(missing_ok=True)
    return result


def _cuda_memory_snapshot(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda":
        return 0, 0
    selected = torch.device(device)
    peak = int(torch.cuda.max_memory_allocated(selected))
    properties = torch.cuda.get_device_properties(selected)
    total = int(properties.total_memory)
    return peak, total


def _cuda_reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def _record_cuda_timed_event(
    run_dir: Path,
    *,
    role: str,
    elapsed_seconds: float,
    device: torch.device,
    detail: Mapping[str, Any],
) -> dict[str, Any]:
    peak, total = _cuda_memory_snapshot(device)
    return _record_resource_event(
        run_dir,
        role=role,
        elapsed_seconds=elapsed_seconds,
        peak_cuda_memory_bytes=peak,
        total_cuda_memory_bytes=total,
        detail={**dict(detail), "device": str(device), "cuda_memory_measured": 1},
    )


def _verify_external_manifest(root: Path) -> dict[str, Any]:
    if root.name != V4_BASENAME:
        raise GlobalDilatedRolloutError("frozen v4 run basename changed")
    path = root / "artifact_manifest.json"
    if not path.is_file() or file_fingerprint(path) != V4_MANIFEST_FILE_SHA256:
        raise GlobalDilatedRolloutError("frozen v4 manifest file commitment changed")
    manifest = _read_json(path, semantic=True)
    rows = manifest.get("artifacts")
    if (
        manifest.get("semantic_sha256") != V4_MANIFEST_SEMANTIC_SHA256
        or manifest.get("artifact_count") != V4_MANIFEST_COUNT
        or not isinstance(rows, list)
        or len(rows) != V4_MANIFEST_COUNT
    ):
        raise GlobalDilatedRolloutError("external artifact manifest rows changed")
    registered: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise GlobalDilatedRolloutError("external artifact manifest row changed")
        relative = str(row["path"])
        if relative in registered:
            raise GlobalDilatedRolloutError("external artifact manifest path duplicated")
        registered.add(relative)
        candidate = root / relative
        if (
            not candidate.is_file()
            or candidate.stat().st_size != int(row.get("size", -1))
            or file_fingerprint(candidate) != row.get("sha256")
        ):
            raise GlobalDilatedRolloutError(f"external immutable artifact changed: {candidate}")
    actual = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file()
    }
    allowed_inventory_self_files = {
        "artifact_manifest.json",
        "SHA256SUMS.txt",
        "bundle_integrity_audit.json",
    }
    if actual != registered | allowed_inventory_self_files:
        raise GlobalDilatedRolloutError("frozen v4 registered artifact path set changed")
    return {
        "present": 1,
        "verified": 1,
        "file_sha256": file_fingerprint(path),
        "semantic_sha256": manifest.get("semantic_sha256"),
        "artifact_count": len(rows),
        "registered_path_set_exact": 1,
    }


def _measure_input_bindings(args: argparse.Namespace) -> dict[str, Any]:
    """Remeasure every immutable input without writing into the active run."""

    parent = Path(args.training_parent).resolve()
    v4 = Path(args.v4_run_dir).resolve()
    source_root = Path(args.source_run_dir).resolve()
    if not parent.is_dir() or not v4.is_dir() or not source_root.is_dir():
        raise GlobalDilatedRolloutError("a required immutable parent directory is absent")

    # The parent registry is the load-bearing commitment for all 2,701 parent
    # artifacts.  The historical helper deliberately knows the frozen basename,
    # registry hashes, source/config seals, terminal decision, and unopened
    # confirmation contract.  It currently returns no record on success, so we
    # retain a deterministic local measurement after it has verified every row.
    try:
        _verify_frequency1_parent_registry(parent, test_only=False)
    except Exception as exc:
        raise GlobalDilatedRolloutError("sealed frequency-one parent registry failed") from exc
    parent_registry = _read_json(parent / "artifact_registry.json", semantic=True)
    parent_registered_paths = {
        str(row["path"]) for row in parent_registry.get("artifacts", [])
    }
    parent_actual_paths = {
        path.relative_to(parent).as_posix()
        for path in parent.rglob("*")
        if path.is_file()
    }
    if parent_actual_paths != parent_registered_paths | {"artifact_registry.json"}:
        raise GlobalDilatedRolloutError("sealed parent registered artifact path set changed")
    parent_registry_binding = {
        "file_sha256": file_fingerprint(parent / "artifact_registry.json"),
        "semantic_sha256": parent_registry["semantic_sha256"],
        "artifact_count": int(parent_registry["artifact_count"]),
        "source_fingerprint": _read_json(parent / "run_manifest.json").get(
            "source_fingerprint"
        ),
        "scientific_config_semantic_sha256": _read_json(
            parent / "scientific_config.json", semantic=True
        )["semantic_sha256"],
        "terminal_decision": _read_json(parent / "run_status.json").get("decision"),
        "confirmation_evidence_opened": 0,
        "all_registry_rows_remeasured": 1,
        "registered_path_set_exact": 1,
    }

    path_plan = _read_json(parent / "path_id_plan.json", semantic=True)
    roles = path_plan.get("roles", {})
    expected = {
        "training": list(range(0xF8100, 0xF8140)),
        "validation": list(range(0xF8200, 0xF8220)),
        "confirmation": list(range(0xF9000, 0xF9040)),
    }
    for role, values in expected.items():
        if roles.get(role) != values:
            raise GlobalDilatedRolloutError(f"immutable {role} path role changed")
    if (parent / "confirmation_namespace_open.json").exists():
        raise GlobalDilatedRolloutError("protected confirmation evidence is already open")

    train_index = _read_json(parent / "eager_cache/train_index.json", semantic=True)
    validation_index = _read_json(parent / "eager_cache/validation_index.json", semantic=True)
    if train_index.get("path_ids") != expected["training"]:
        raise GlobalDilatedRolloutError("training index path role changed")
    if validation_index.get("path_ids") != expected["validation"]:
        raise GlobalDilatedRolloutError("validation index path role changed")

    checkpoint = load_verified_frequency1_checkpoint(parent, device="cpu")
    source = load_verified_source_target(source_root)
    if (
        source.metadata.get("image_sha256") != SOURCE_IMAGE_MEASURE_SHA256
        or source.metadata.get("mixed_target_sha256") != MIXED_TARGET_MEASURE_SHA256
        or int(source.metadata.get("label", -1)) != 3
        or int(source.metadata.get("dataset_index", -1)) != 7
        or float(source.metadata.get("lambda_mix", math.nan)) != 0.35
    ):
        raise GlobalDilatedRolloutError("frozen source target measure changed")
    v4_manifest = _verify_external_manifest(v4)
    copied_checkpoint_path = v4 / "input_bindings/checkpoint.pt"
    if copied_checkpoint_path.is_file():
        copied = torch.load(copied_checkpoint_path, map_location="cpu", weights_only=True)
        copied_state = copied.get("state_dict") if isinstance(copied, Mapping) else None
        if not isinstance(copied_state, Mapping) or state_dict_sha256(copied_state) != checkpoint.state_sha256:
            raise GlobalDilatedRolloutError("v4 copied checkpoint differs from the frozen parent")

    train_open = parent / "physical_train_label_open.json"
    validation_open = parent / "validation_label_open.json"
    for opening in (train_open, validation_open):
        if not opening.is_file():
            raise GlobalDilatedRolloutError("required already-open development role is absent")
    return _semantic(
        {
            "schema": VERSION + "-input-bindings",
            "schema_version": 1,
            "training_parent": str(parent),
            "v4_run_dir": str(v4),
            "source_run_dir": str(source_root),
            "execution_device": str(args.device),
            "parent_registry": parent_registry_binding,
            "path_plan_file_sha256": file_fingerprint(parent / "path_id_plan.json"),
            "path_plan_semantic_sha256": path_plan["semantic_sha256"],
            "train_index_file_sha256": file_fingerprint(parent / "eager_cache/train_index.json"),
            "train_index_semantic_sha256": train_index["semantic_sha256"],
            "validation_index_file_sha256": file_fingerprint(parent / "eager_cache/validation_index.json"),
            "validation_index_semantic_sha256": validation_index["semantic_sha256"],
            "train_label_open_file_sha256": file_fingerprint(train_open),
            "train_label_open_semantic_sha256": _read_json(train_open, semantic=True)[
                "semantic_sha256"
            ],
            "validation_label_open_file_sha256": file_fingerprint(validation_open),
            "validation_label_open_semantic_sha256": _read_json(
                validation_open, semantic=True
            )["semantic_sha256"],
            "confirmation_namespace_opened": 0,
            "roles": {key: values for key, values in expected.items()},
            "frozen_v4_checkpoint": checkpoint.to_record(),
            "source_target": source.to_record(),
            "v4_manifest": v4_manifest,
            "development_evidence_reused": 1,
            "protected_confirmation_evidence_used": 0,
        }
    )


def _verify_inputs_and_roles(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Verify reused development roles and seal their complete measurement."""

    measured = _measure_input_bindings(args)
    atomic_write_json(run_dir / "input_bindings.json", measured)
    return measured


def _scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    architecture = global_dilated_architecture_contract()
    if int(architecture.get("passed", 0)) != 1:
        raise GlobalDilatedRolloutError("global-dilated architecture contract failed")
    return _semantic(
        {
            "schema": VERSION + "-scientific-config",
            "schema_version": 1,
            "research_mode": "exploratory",
            "execution_device": str(args.device),
            "program_objective": "DDPM-like MNIST generator based on the Eulerian approximation",
            "decision": (
                "On one fresh held-out forward path, does a global Jacobi/RB score model "
                "improve an exact paired 128-step suffix over zero, and does frozen v4 -0.5 "
                "reveal a controller sign/order defect?"
            ),
            "competing_hypotheses": [
                "global_receptive_field_is_needed",
                "controller_sign_or_order_is_wrong",
                "cached_tangent_risk_does_not_compose_on_policy",
                "controller_authority_is_dynamically_negligible",
                "terminal_or_prior_mismatch",
                "current_jacobi_rb_strategy_lacks_feasibility",
            ],
            "architecture": architecture,
            "target_contract": {
                "stored_label": "bar_Z",
                "score_model_output": "q_theta",
                "wrapped_training_output": "m_theta=Y(1-Y)*q_theta",
                "training_objective": "MSE(m_theta,bar_Z)/RMS_train(bar_Z)^2",
                "quotient_bar_Z_over_mobility_formed": 0,
            },
            "training": {
                **TRAINING,
                "betas": list(TRAINING["betas"]),
                "device": str(args.device),
                "seed": TRAINING_SEED,
                "cap_ladder": list(CAP_LADDER),
                "checkpoint_interval": CHECKPOINT_INTERVAL,
                "selection": "finite nonzero minimum validation MSE, earlier update tie",
            },
            "evaluation": {
                "row_order": list(ROW_ORDER),
                "gains": {"v4-plus-0p5": 0.5, "v4-minus-0p5": -0.5, "global-plus-1": 1.0},
                "fresh_path_pool": list(FRESH_PATH_POOL),
                "forward_root_seed": FORWARD_ROOT_SEED,
                "reverse_root_seed": REVERSE_ROOT_SEED,
                "anchor_step": ANCHOR_STEP,
                "reverse_outer_steps": MANDATORY_SUFFIX_STEPS,
                "phase_count": PHASE_COUNT,
                "microsteps": MICROSTEPS,
                "reference_backend": "certified_exact",
                "primary_metric": "paired final squared L2 improvement over zero",
                "independent_unit": "one path/image",
                "population_inference": 0,
                "practical_relative_threshold": PRACTICAL_RELATIVE_THRESHOLD,
            },
            "gate_types": {
                "blocking": "execution/integrity",
                "practical_effect_label": "diagnostic threshold",
                "positive_claim": "exploratory claim boundary",
            },
            "resource_budget": {
                "active_seconds": ACTIVE_SECONDS_CAP,
                "storage_bytes": STORAGE_CAP_BYTES,
                "cuda_fraction": CUDA_MEMORY_FRACTION_CAP,
                "report_reserve_seconds": REPORT_RESERVE_SECONDS,
                "mandatory_exact_suffix_never_shortened": 1,
            },
            "outcome_actions": {
                "global_material_improvement": "attempt same-path complete zero/global/source reconstruction",
                "sign_order_leading": "reconcile theory/code convention before any sign change",
                "validation_better_suffix_adverse": "pivot to rollout alignment or target derivation",
                "all_learned_adverse_controls_pass": "stop nearby Jacobi repairs and run stated major pivot",
                "resource_failure": "preserve partial exact evidence; approximate output is nonauthorizing",
            },
            "deliberate_omissions": [
                "no confirmation labels",
                "no approximate primary suffix",
                "no hot-loop 129-state capture",
                "no support-radius execution gate",
            ],
        }
    )


def _initialize_run(run_dir: Path, args: argparse.Namespace) -> None:
    existing = run_dir.exists()
    if existing:
        if not run_dir.is_dir():
            raise GlobalDilatedRolloutError("run path exists but is not a directory")
        if (run_dir / _RAW_SUFFIX_CONVERSION_INTENT).is_file():
            # The intent itself binds the already-verified immutable source
            # package, failed ledger, and raw suffix.  Main completes this
            # packaging-only transaction before the ordinary compatibility
            # path, which may no longer have a live source terminal after a
            # crash in supersession cleanup.
            _raw_suffix_conversion_intent(run_dir)
            return
        if (
            _live_failure_capture(run_dir) is not None
            and not (run_dir / "terminal_failure.json").is_file()
        ):
            # A captured failure predates its terminal package.  Main routes
            # this authenticated state to packaging-only recovery before any
            # resume accounting or scientific-stage mutation.
            return
        _verify_resume_compatibility(run_dir, args)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "controls").mkdir(exist_ok=True)
    (run_dir / "training/checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "stages").mkdir(exist_ok=True)
    command = " ".join([sys.executable, "-m", "mnist.diag_d0_jacobi_rb_global_dilated_rollout", *sys.argv[1:]])
    _atomic_text(run_dir / "exact_command.txt", command + "\n")
    config = _scientific_config(args)
    config_path = run_dir / "scientific_config.json"
    if config_path.is_file():
        if _read_json(config_path, semantic=True) != config:
            raise GlobalDilatedRolloutError("scientific configuration changed on resume")
    else:
        atomic_write_json(config_path, config)
    _write_run_manifest(run_dir, args)
    if not (run_dir / "resource_ledger.json").is_file():
        atomic_write_json(run_dir / "resource_ledger.json", _resource_ledger(run_dir))


def _mark_stage(run_dir: Path, stage: str, detail: Mapping[str, Any] | None = None) -> None:
    _write_semantic(
        run_dir / "stages" / f"{stage}.json",
        {
            "schema": VERSION + "-stage",
            "schema_version": 1,
            "stage": stage,
            "passed": 1,
            "completed_at": _utc_now(),
            "detail": dict(detail or {}),
        },
    )


def _stage_complete(run_dir: Path, stage: str) -> bool:
    path = run_dir / "stages" / f"{stage}.json"
    return path.is_file() and int(_read_json(path, semantic=True).get("passed", 0)) == 1


def _verify_completed_stage(
    run_dir: Path, args: argparse.Namespace, stage: str
) -> dict[str, Any]:
    marker = _read_json(run_dir / "stages" / f"{stage}.json", semantic=True)
    if marker.get("stage") != stage or marker.get("passed") != 1:
        raise GlobalDilatedRolloutError(f"completed {stage} marker changed")
    detail = marker.get("detail")
    if not isinstance(detail, Mapping):
        raise GlobalDilatedRolloutError(f"completed {stage} detail changed")
    if stage == "prepare":
        bindings = _read_json(run_dir / "input_bindings.json", semantic=True)
        manifest = _read_json(run_dir / "run_manifest.json", semantic=True)
        if (
            detail.get("input_bindings_sha256") != bindings.get("semantic_sha256")
            or detail.get("run_manifest_sha256") != manifest.get("semantic_sha256")
        ):
            raise GlobalDilatedRolloutError("completed prepare artifact binding changed")
        # External inputs were already remeasured by the mutation-free resume
        # compatibility gate.  Rehashing the 2,701-file parent again for every
        # completed-stage dependency would add unledgered work without adding
        # a new trust boundary.
    elif stage == "controls":
        theory = _read_json(run_dir / "controls/theory_to_code.json", semantic=True)
        preflight = _read_json(run_dir / "controls/preflight_controls.json", semantic=True)
        if (
            theory.get("passed") != 1
            or preflight.get("passed") != 1
            or detail.get("theory_to_code_sha256") != theory.get("semantic_sha256")
            or detail.get("preflight_controls_sha256") != preflight.get("semantic_sha256")
            or float(preflight.get("projected_suffix_seconds_with_20pct_margin", math.inf))
            + FORWARD_RESERVE_SECONDS
            + REPORT_RESERVE_SECONDS
            >= ACTIVE_SECONDS_CAP
        ):
            raise GlobalDilatedRolloutError("completed controls artifact binding changed")
    elif stage == "train_select_freeze":
        selection = _read_json(run_dir / "selection.json", semantic=True)
        freeze = _read_json(run_dir / "evaluation_freeze.json", semantic=True)
        if (
            detail.get("selection_sha256") != selection.get("semantic_sha256")
            or detail.get("evaluation_freeze_sha256") != freeze.get("semantic_sha256")
            or freeze.get("selection_file_sha256")
            != file_fingerprint(run_dir / "selection.json")
            or selection.get("selected") != freeze.get("global_checkpoint")
        ):
            raise GlobalDilatedRolloutError("completed selection/freeze binding changed")
        _selected_global_model(run_dir, torch.device("cpu"))
    elif stage == "evaluate_exact":
        outcome = _read_json(run_dir / "outcome.json", semantic=True)
        path_usage = _read_json(run_dir / "path_usage.json", semantic=True)
        _read_json(run_dir / "positive_branch.json", semantic=True)
        if (
            detail.get("outcome") != outcome.get("outcome")
            or detail.get("fresh_path_id") != path_usage.get("fresh_path_id")
        ):
            raise GlobalDilatedRolloutError("completed evaluation marker binding changed")
        _verify_scientific_evidence_read_only(run_dir, args)
    elif stage == "report_verify":
        _verify_completed_report_read_only(run_dir, args)
    return marker


def _require_stage(run_dir: Path, args: argparse.Namespace, stage: str) -> None:
    index = STAGES.index(stage)
    if index and not _stage_complete(run_dir, STAGES[index - 1]):
        raise GlobalDilatedRolloutError(f"{stage} requires completed {STAGES[index - 1]}")
    if index:
        _verify_completed_stage(run_dir, args, STAGES[index - 1])


def _prepared_exact_backend(device: torch.device, profile: JacobiRBCudaProfile) -> Any:
    if device.type != "cuda":
        raise GlobalDilatedRolloutError("the production exact fused backend requires CUDA")
    from mnist.d0_jacobi_rb_cuda_deferred import prepare_alpha1_rb_transition_batch_cuda_deferred

    key = f"{device}:{config_fingerprint(profile.to_dict())}"
    if key not in _PREPARED_BACKENDS:
        started = time.perf_counter()
        _PREPARED_BACKENDS[key] = prepare_alpha1_rb_transition_batch_cuda_deferred(
            device=device, profile=profile
        )
        _PREPARED_BACKEND_SECONDS[key] = time.perf_counter() - started
    return _PREPARED_BACKENDS[key]


def _strict_fused_exact_health(
    *,
    final_state: np.ndarray,
    shard_records: Sequence[Mapping[str, Any]],
    row_count: int,
) -> dict[str, Any]:
    state = np.asarray(final_state)

    def exact_count(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise GlobalDilatedRolloutError(f"fused exact count {name} is not an integer")
        return int(value)

    def finite_number(value: Any, name: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
        ):
            raise GlobalDilatedRolloutError(f"fused exact numeric {name} is not finite")
        return float(value)

    def reject_nonfinite_numerics(value: Any, name: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                reject_nonfinite_numerics(child, f"{name}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                reject_nonfinite_numerics(child, f"{name}[{index}]")
        elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            raise GlobalDilatedRolloutError(f"fused exact telemetry {name} is nonfinite")

    row_count_value = exact_count(row_count, "row_count")
    if row_count_value <= 0:
        raise GlobalDilatedRolloutError("fused exact row count is invalid")

    if (
        state.dtype != np.float64
        or state.shape != (row_count_value, STATE_SIZE)
        or not np.isfinite(state).all()
        or np.any(state < 0.0)
        or float(np.max(np.abs(np.sum(state, axis=1) - 1.0))) > 2.0e-12
    ):
        raise GlobalDilatedRolloutError("fused exact final-state health failed")
    total_active = 0
    total_certified = 0
    total_fallback = 0
    total_transition_count = 0
    maximum_mass_error = 0.0
    for shard_index, shard in enumerate(shard_records):
        reject_nonfinite_numerics(shard, f"shard[{shard_index}]")
        diagnostics = shard.get("diagnostics")
        rows = shard.get("per_row_diagnostics")
        controller_rows = shard.get("controller_diagnostics")
        row_table = shard.get("row_table")
        if (
            not isinstance(diagnostics, Mapping)
            or not isinstance(rows, list)
            or len(rows) != row_count_value
            or not isinstance(controller_rows, list)
            or len(controller_rows) != row_count_value
            or not isinstance(row_table, list)
            or len(row_table) != row_count_value
            or exact_count(shard.get("committed", 0), "committed") != 1
        ):
            raise GlobalDilatedRolloutError("fused exact shard health schema changed")
        row_table_keys = {
            "row_key",
            "canonical_path_id",
            "controller_kind",
            "variant",
            "horizon",
            "gain",
            "controller_binding",
        }
        if any(
            not isinstance(item, Mapping)
            or set(item) != row_table_keys
            or not isinstance(item.get("row_key"), str)
            or not item.get("row_key")
            or exact_count(item.get("canonical_path_id"), "canonical_path_id") < 0
            or not isinstance(item.get("controller_binding"), Mapping)
            or (
                item.get("gain") is not None
                and not math.isfinite(finite_number(item.get("gain"), "gain"))
            )
            for item in row_table
        ):
            raise GlobalDilatedRolloutError("fused exact row-table schema changed")
        expected_row_keys = [str(item["row_key"]) for item in row_table]
        if len(set(expected_row_keys)) != row_count_value:
            raise GlobalDilatedRolloutError("fused exact shard row identities changed")
        if (
            [row.get("row_key") for row in rows] != expected_row_keys
            or [row.get("row_key") for row in controller_rows] != expected_row_keys
        ):
            raise GlobalDilatedRolloutError("fused exact row telemetry identity changed")
        if finite_number(diagnostics.get("certificate_fraction"), "certificate_fraction") != 1.0:
            raise GlobalDilatedRolloutError("fused exact shard certificate fraction is not one")
        maximum_mass_error = max(
            maximum_mass_error,
            finite_number(diagnostics.get("maximum_mass_error"), "maximum_mass_error"),
        )
        if maximum_mass_error > 2.0e-12:
            raise GlobalDilatedRolloutError("fused exact shard mass health failed")
        forbidden = diagnostics.get("forbidden_counts")
        if not isinstance(forbidden, Mapping) or set(forbidden) != set(_FORBIDDEN_EXACT_COUNTS):
            raise GlobalDilatedRolloutError("fused exact forbidden-count schema is absent")
        for name in _FORBIDDEN_EXACT_COUNTS:
            if exact_count(forbidden.get(name), f"forbidden_counts.{name}") != 0:
                raise GlobalDilatedRolloutError(
                    f"fused exact shard {shard_index} recorded forbidden {name}"
                )
        reference = diagnostics.get("reference")
        if not isinstance(reference, Mapping):
            raise GlobalDilatedRolloutError("fused exact reference health is absent")
        reference_rows = reference.get("per_row")
        if not isinstance(reference_rows, list) or len(reference_rows) != row_count_value:
            raise GlobalDilatedRolloutError("fused exact reference per-row health changed")
        reference_forbidden = reference.get("forbidden_counts")
        if (
            not isinstance(reference_forbidden, Mapping)
            or set(reference_forbidden) != set(_FORBIDDEN_EXACT_COUNTS)
        ):
            raise GlobalDilatedRolloutError("fused reference forbidden counts are absent")
        for name in _FORBIDDEN_EXACT_COUNTS:
            if exact_count(
                reference_forbidden.get(name), f"reference.forbidden_counts.{name}"
            ) != 0:
                raise GlobalDilatedRolloutError(
                    f"fused reference shard {shard_index} recorded forbidden {name}"
                )
        execution_plan = shard.get("execution_plan")
        sequence = execution_plan.get("sequence") if isinstance(execution_plan, Mapping) else None
        microsteps = exact_count(shard.get("microsteps"), "microsteps")
        if (
            not isinstance(execution_plan, Mapping)
            or not isinstance(sequence, list)
            or not sequence
            or any(
                not isinstance(item, list)
                or len(item) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in item)
                for item in sequence
            )
            or exact_count(execution_plan.get("row_count"), "execution_plan.row_count")
            != row_count_value
        ):
            raise GlobalDilatedRolloutError("fused exact execution-plan schema changed")
        expected_per_row = len(sequence) * 2 * microsteps * EDGES_PER_PHASE
        deterministic_total = expected_per_row * row_count_value
        shard_fallback = 0
        shard_transition_count = exact_count(shard.get("transition_count"), "transition_count")
        if (
            shard_transition_count != deterministic_total
            or exact_count(
                execution_plan.get("transition_count"), "execution_plan.transition_count"
            )
            != deterministic_total
        ):
            raise GlobalDilatedRolloutError("fused exact shard transition count changed")
        reference_counts = {
            name: exact_count(reference.get(name), f"reference.{name}")
            for name in (
                "transition_count",
                "active_count",
                "structural_noop_count",
                "certified_count",
                "fallback_count",
                "unauthorized_count",
                "invalid_count",
            )
        }
        if (
            exact_count(diagnostics.get("transition_count"), "diagnostics.transition_count")
            != shard_transition_count
            or reference_counts["transition_count"] != shard_transition_count
            or reference_counts["active_count"] <= 0
            or reference_counts["active_count"]
            + reference_counts["structural_noop_count"]
            != reference_counts["transition_count"]
            or reference_counts["certified_count"] != reference_counts["active_count"]
            or reference_counts["unauthorized_count"] != 0
            or reference_counts["invalid_count"] != 0
            or reference_counts["fallback_count"] < 0
            or reference_counts["fallback_count"] > reference_counts["transition_count"]
            or exact_count(diagnostics.get("fallback_count"), "diagnostics.fallback_count")
            != reference_counts["fallback_count"]
        ):
            raise GlobalDilatedRolloutError("fused exact aggregate transition authority changed")
        try:
            _verify_fused_load_bearing_row_telemetry(
                shard,
                expected_rows=row_table,
                reference_contract="certified_exact",
                expected_transition_count_per_row=expected_per_row,
            )
        except Exception as exc:
            raise GlobalDilatedRolloutError(
                "fused load-bearing per-row telemetry changed"
            ) from exc
        row_reference_totals = {name: 0 for name in reference_counts}
        for row_index, (row, reference_row, controller_row) in enumerate(
            zip(rows, reference_rows, controller_rows, strict=True)
        ):
            if not all(isinstance(value, Mapping) for value in (row, reference_row, controller_row)):
                raise GlobalDilatedRolloutError("fused exact per-row schema changed")
            required_controller_counts = {
                "call_count",
                "lane_count",
                "score_count",
                "movable_count",
                "already_equal_count",
                "zero_pair_mass_count",
                "zero_duration_count",
                "target_oracle_unreachable_boundary_count",
                "clipping_count",
                "floor_count",
                "projection_count",
                "nonfinite_score_count",
            }
            if not required_controller_counts.issubset(controller_row):
                raise GlobalDilatedRolloutError("fused controller telemetry schema changed")
            for name in required_controller_counts:
                if exact_count(controller_row.get(name), f"controller.{name}") < 0:
                    raise GlobalDilatedRolloutError("fused controller count is negative")
            for name in (
                "reference_fraction_displacement_count",
                "control_fraction_displacement_count",
                "score_count",
                "logistic_shift_count",
                "boundary_fraction_count",
                "transition_count",
            ):
                if exact_count(row.get(name), f"per_row.{name}") < 0:
                    raise GlobalDilatedRolloutError("fused phase count is negative")
            if any(exact_count(row.get(name, 0), name) != 0 for name in _FUSED_INVALID_FIELDS):
                raise GlobalDilatedRolloutError(
                    f"fused exact shard {shard_index} row {row_index} health failed"
                )
            active = exact_count(reference_row.get("active_count", -1), "active_count")
            certified = exact_count(reference_row.get("certified_count", -1), "certified_count")
            transition = exact_count(reference_row.get("transition_count"), "transition_count")
            noops = exact_count(
                reference_row.get("structural_noop_count"), "structural_noop_count"
            )
            fallback = exact_count(reference_row.get("fallback_count"), "fallback_count")
            if (
                active <= 0
                or certified != active
                or transition != active + noops
                or noops < 0
                or finite_number(
                    reference_row.get("certificate_fraction"), "certificate_fraction"
                )
                != 1.0
                or exact_count(reference_row.get("invalid_count", -1), "invalid_count") != 0
                or exact_count(reference_row.get("unauthorized_count", -1), "unauthorized_count") != 0
                or fallback < 0
                or fallback > transition
                or transition != expected_per_row
                or exact_count(row.get("transition_count", -1), "row_transition_count") != expected_per_row
            ):
                raise GlobalDilatedRolloutError(
                    f"fused exact shard {shard_index} row {row_index} authorization failed"
                )
            total_active += active
            total_certified += certified
            shard_fallback += fallback
            for name in row_reference_totals:
                value = exact_count(reference_row.get(name), f"reference.per_row.{name}")
                row_reference_totals[name] += value
                if exact_count(row.get(f"reference_{name}"), f"per_row.reference_{name}") != value:
                    raise GlobalDilatedRolloutError(
                        "fused merged/reference per-row authority changed"
                    )
            for key in (
                "nonfinite_score_count",
                "clipping_count",
                "floor_count",
                "projection_count",
            ):
                if exact_count(controller_row.get(key, 0), key) != 0:
                    raise GlobalDilatedRolloutError(
                        f"fused controller shard {shard_index} row {row_index} health failed"
                    )
        if shard_fallback != reference_counts["fallback_count"]:
            raise GlobalDilatedRolloutError(
                f"fused exact shard {shard_index} fallback aggregate changed"
            )
        total_fallback += shard_fallback
        if row_reference_totals != reference_counts:
            raise GlobalDilatedRolloutError("fused exact per-row authorization sum changed")
        total_transition_count += shard_transition_count
    if total_active <= 0 or total_active != total_certified:
        raise GlobalDilatedRolloutError("fused exact aggregate row authorization failed")
    return {
        "passed": 1,
        "row_count": int(row_count),
        "shard_count": len(shard_records),
        "active_count": total_active,
        "certified_count": total_certified,
        "certificate_fraction": 1.0,
        "maximum_mass_error": maximum_mass_error,
        "forbidden_event_count": 0,
        "fallback_count": total_fallback,
        "transition_count": total_transition_count,
        "final_state_nonfinite_count": 0,
        "final_state_negative_count": 0,
    }


def _committed_shard_elapsed(
    root: Path, *, minimum_start_step: int | None = None
) -> float:
    """Return durable sampler-through-NPZ commit time for committed shards."""

    elapsed: list[float] = []
    if root.is_dir():
        for path in sorted(root.glob("shard-*.json")):
            if path.name.endswith(".failure.json"):
                continue
            try:
                record = _read_json(path, semantic=True)
            except GlobalDilatedRolloutError:
                continue
            if record.get("committed") == 1 and (
                minimum_start_step is None
                or int(record.get("start_step", -1)) >= minimum_start_step
            ):
                value = record.get("elapsed_seconds")
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) >= 0.0
                ):
                    elapsed.append(float(value))
    return math.fsum(elapsed)


def _resource_attempt_index(run_dir: Path, role: str) -> int:
    return sum(
        1
        for item in _resource_ledger(run_dir).get("events", [])
        if item.get("role") == role
    )


def _record_attempt_wall(
    run_dir: Path,
    *,
    role: str,
    started: float,
    durable_before_seconds: float = 0.0,
    durable_elapsed_seconds: float = 0.0,
    failed: bool,
    peak_cuda_memory_bytes: int = 0,
    total_cuda_memory_bytes: int = 0,
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Conservatively charge one invocation, including durable crash-prefix work."""

    prior_events = [
        item
        for item in _resource_ledger(run_dir).get("events", [])
        if item.get("role") == role
    ]
    attempt = len(prior_events)
    wall = time.perf_counter() - started
    covered_durable = max(
        (
            float(item.get("detail", {}).get("durable_committed_shard_seconds", 0.0))
            for item in prior_events
        ),
        default=0.0,
    )
    durable_before = max(float(durable_before_seconds), 0.0)
    durable_after = max(float(durable_elapsed_seconds), durable_before)
    recovered_prefix = max(durable_before - covered_durable, 0.0)
    new_durable = max(durable_after - durable_before, 0.0)
    accounted = recovered_prefix + max(wall, new_durable)
    return _record_resource_event(
        run_dir,
        role=role,
        elapsed_seconds=accounted,
        peak_cuda_memory_bytes=peak_cuda_memory_bytes,
        total_cuda_memory_bytes=total_cuda_memory_bytes,
        detail={
            "attempt": attempt,
            "failed": int(failed),
            "invocation_wall_seconds": wall,
            "durable_before_seconds": durable_before,
            "durable_committed_shard_seconds": durable_after,
            "recovered_unaccounted_prefix_seconds": recovered_prefix,
            "accounted_seconds": accounted,
            **dict(detail or {}),
        },
    )


def _exact_reference_factory(
    *,
    prepared: Any,
    profile: JacobiRBCudaProfile,
    root_seed: int,
    stream_role: str,
    row_chunk_size: int | None = None,
) -> Any:
    key = (id(prepared), int(root_seed), str(stream_role))
    if key not in _PREPARED_SEED_MAPS:
        _PREPARED_SEED_MAPS[key] = prepare_deferred_reference_rng_seed_map(
            prepared_backend=prepared,
            root_seed=int(root_seed),
            stream_role=str(stream_role),
        )
    seed_map = _PREPARED_SEED_MAPS[key]

    def factory(_shard_index: int) -> DeferredCertifiedFusedReference:
        return DeferredCertifiedFusedReference(
            profile=profile,
            root_seed=int(root_seed),
            stream_role=str(stream_role),
            prepared_backend=prepared,
            prepared_rng_seeds=seed_map,
            row_chunk_size=row_chunk_size,
        )

    return factory


def _fixture_inputs(*, batch: int = 1, boundary: bool = False) -> ModelInputs:
    state = torch.full((batch, STATE_SIZE), 1.0 / STATE_SIZE, dtype=torch.float32)
    if boundary:
        # Matching zero contains (tail=0, head=1).  Force its head fraction to
        # an exact endpoint while retaining a valid simplex state.
        state[:, 0] += state[:, 1]
        state[:, 1] = 0.0
    return ModelInputs(
        later_full_state=state,
        reverse_time=torch.full((batch,), internal_reverse_time(127, 6, 0.75), dtype=torch.float64),
        phase=torch.full((batch,), 6, dtype=torch.long),
        color=torch.full((batch,), 0, dtype=torch.long),
        duration=torch.full((batch,), float(PHASE_DURATIONS[6]), dtype=torch.float32),
        label=torch.full((batch,), 3, dtype=torch.long),
    )


def _wrapped_training_loss(
    model: nn.Module,
    inputs: ModelInputs,
    denoising_target: Tensor,
    train_target_rms: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Call wrapped ``forward`` and compare ``m`` directly with ``bar_Z``."""

    if type(inputs) is not ModelInputs:
        raise GlobalDilatedRolloutError("training requires exact ModelInputs")
    target = denoising_target.to(dtype=torch.float64)
    prediction_m = model(inputs)
    if prediction_m.shape != target.shape:
        raise GlobalDilatedRolloutError("wrapped training prediction shape changed")
    loss, raw = direct_raw_target_mse(prediction_m, target, float(train_target_rms))
    return loss, raw, prediction_m


def _run_theory_to_code_control(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Commit the bar_Z/m/q, mobility, sign, and time convention contract."""

    torch.manual_seed(91_271)
    wrapped = GlobalDilatedZeroBaselinePredictor(zero_residual=True)
    with torch.no_grad():
        wrapped.residual_score.local_affine.bias.fill_(0.375)
    inputs = _fixture_inputs(batch=2, boundary=True)
    with torch.no_grad():
        q = wrapped.score_prediction(inputs)
        m = wrapped(inputs)
        mobility = edge_pair_geometry(inputs).mobility
    mobility_once = torch.equal(m, mobility * q)
    zero_mobility_exact = bool(torch.all(m[mobility == 0.0] == 0.0))

    target = torch.full_like(m, 0.125)
    loss, raw, prediction = _wrapped_training_loss(wrapped, inputs, target, 0.5)
    expected_raw = torch.mean((m - target).square())
    training_forward_direct = bool(torch.equal(prediction, m))
    training_target_direct = bool(torch.equal(raw, expected_raw))
    training_normalization = math.isclose(
        float(loss.detach()),
        float((expected_raw / (0.5**2)).detach()),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )

    class ScoreOnlySpy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.score_calls = 0

        def score_prediction(self, value: ModelInputs) -> Tensor:
            self.score_calls += 1
            return torch.ones((value.batch_size, EDGES_PER_PHASE), dtype=torch.float64)

        def forward(self, _value: ModelInputs) -> Tensor:  # pragma: no cover - failure sentinel
            raise AssertionError("rollout must not call wrapped forward")

    spy = ScoreOnlySpy()
    rollout_wrapper = ScaledTangentScoreController(spy, 0.5)
    rollout_value = rollout_wrapper.score_prediction(inputs)
    rollout_uses_score = spy.score_calls == 1 and bool(torch.all(rollout_value == 0.5))

    y = torch.tensor([[0.4]], dtype=torch.float64)
    score = torch.tensor([[0.7]], dtype=torch.float64)
    delta_u = 0.03
    evolved = frozen_score_logistic_fraction(y, score, delta_u)
    logit_before = torch.logit(y)
    logit_after = torch.logit(evolved)
    expected_shift = 2.0 * score * delta_u
    logistic_exact = bool(torch.allclose(logit_after - logit_before, expected_shift, rtol=0.0, atol=2e-15))
    positive_score_increases_head_logit = bool(torch.all(evolved > y))

    checks = {
        "mobility_applied_once_in_training": int(mobility_once),
        "zero_mobility_wrapped_output_exact_zero": int(zero_mobility_exact),
        "training_calls_wrapped_forward": int(training_forward_direct),
        "training_compares_m_directly_with_bar_Z": int(training_target_direct),
        "training_uses_train_target_rms_squared": int(training_normalization),
        "rollout_calls_score_prediction": int(rollout_uses_score),
        "exact_logit_shift_equals_2_q_delta_u": int(logistic_exact),
        "positive_q_increases_declared_head_logit": int(positive_score_increases_head_logit),
        "bar_Z_divided_by_mobility": 0,
        "finite_values": int(
            all(bool(torch.isfinite(item).all()) for item in (q, m, mobility, evolved, loss, raw))
        ),
    }
    passed = int(all(value == 1 for key, value in checks.items() if key != "bar_Z_divided_by_mobility") and checks["bar_Z_divided_by_mobility"] == 0)
    table = [
        {
            "cache_reverse_time": "1-(7*k+phase+q_mid)/(7*512)",
            "outer_step": 127,
            "phase": 6,
            "phase_duration": float(PHASE_DURATIONS[6]),
            "controller_delta_u": "phase_exposure(pair_mass,duration)/M; nonnegative",
            "reverse_execution_order": "k decreases; phase occurrence decreases",
            "logistic_update": "logit(Y_plus)=logit(Y)+2*q_theta*delta_u",
            "orientation": "Y=head/(tail+head)",
        }
    ]
    record = _write_semantic(
        run_dir / "controls/theory_to_code.json",
        {
            "schema": VERSION + "-theory-to-code",
            "schema_version": 1,
            "passed": passed,
            "equations": {
                "bar_Z": "Y(1-Y) * d/dY log k_u(Y|X)",
                "mobility": "mu(Y)=Y(1-Y)",
                "conditional_target": "m(W)=E[bar_Z|W]=mu(Y)*q(W)",
                "controller_ode": "dY/du=2*q_theta*Y(1-Y)",
                "controller_exact_substep": "logit(Y_plus)=logit(Y)+2*q_theta*delta_u",
            },
            "symbols_to_code": {
                "bar_Z": "HostLabelStore.arrays['denoising_target']",
                "q_theta": "GlobalDilatedZeroBaselinePredictor.score_prediction",
                "m_theta": "GlobalDilatedZeroBaselinePredictor.forward",
                "mu": "edge_pair_geometry(inputs).mobility",
                "training": "_wrapped_training_loss",
                "rollout_dispatch": "ScaledTangentScoreController.score_prediction",
                "logistic_flow": "frozen_score_logistic_fraction",
            },
            "time_convention": table,
            "checks": checks,
            "numerical_fixture": {
                "q_sha256": rollout_array_sha256(q),
                "m_sha256": rollout_array_sha256(m),
                "mobility_sha256": rollout_array_sha256(mobility),
                "y": float(y.item()),
                "q": float(score.item()),
                "delta_u": delta_u,
                "y_plus": float(evolved.item()),
            },
            "failure_action": "repair the concrete contract defect in this patch before fresh evidence",
            "failure_does_not_mean": "the end-to-end controller is scientifically adverse",
        },
    )
    if not passed:
        raise GlobalDilatedRolloutError("theory-to-code execution gate failed")
    return record


def _committed_numerical_path_ids(repository_root: Path) -> set[int]:
    realized: set[int] = set()
    runs_root = repository_root / "runs"
    if not runs_root.is_dir():
        return realized
    for path in runs_root.rglob("*.json"):
        if "shard-" not in path.name and path.name not in {
            "trajectory_summary.json",
            "family_summary.json",
            "forward_summary.json",
        }:
            continue
        try:
            value: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue

        def visit(item: Any) -> None:
            if isinstance(item, Mapping):
                if int(item.get("committed", 1)) == 0:
                    return
                for key, child in item.items():
                    if key in {"path_id", "canonical_path_id"} and isinstance(child, int):
                        realized.add(int(child))
                    elif key in {"path_ids", "canonical_path_ids"} and isinstance(child, list):
                        realized.update(int(entry) for entry in child if isinstance(entry, int))
                    else:
                        visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(value)
    return realized


def _load_preflight_anchor(args: argparse.Namespace) -> np.ndarray:
    v4 = Path(args.v4_run_dir).resolve()
    archive_path = v4 / "input_bindings/recovery_anchors.npz"
    binding_path = v4 / "input_bindings/recovery_anchor_binding.json"
    binding = _read_json(binding_path, semantic=True)
    if file_fingerprint(archive_path) != binding.get("file_sha256"):
        raise GlobalDilatedRolloutError("v4 development anchor archive changed")
    arrays = _load_npz(archive_path)
    anchor = arrays.get("step_0127")
    if anchor is None or anchor.dtype != np.float64 or anchor.shape != (STATE_SIZE,):
        raise GlobalDilatedRolloutError("v4 step-127 development anchor changed")
    # The immutable v4 producer committed each vector through its historical
    # canonical one-row convention while storing the NPZ member as `[784]`.
    # Preserve the stored-shape check above, but compare the exact producer
    # hash domain here rather than silently changing the sealed evidence.
    if rollout_array_sha256(anchor.reshape(1, STATE_SIZE)) != binding.get(
        "array_sha256", {}
    ).get("step_0127"):
        raise GlobalDilatedRolloutError("v4 step-127 development anchor hash changed")
    return np.ascontiguousarray(anchor)


def _existing_preflight_path_id(run_dir: Path) -> int | None:
    """Recover the already-committed smoke path before doing a global rescan."""

    record_path = (
        run_dir
        / "controls/exact_smoke/fused_families/five-row/one-shard/shard-0000.json"
    )
    if not record_path.is_file():
        return None
    record = _read_json(record_path, semantic=True)
    if int(record.get("committed", 0)) != 1:
        raise GlobalDilatedRolloutError("existing preflight shard is not committed")
    paths = record.get("canonical_path_ids")
    if (
        not isinstance(paths, list)
        or len(paths) != len(ROW_ORDER)
        or len(set(paths)) != 1
        or int(paths[0]) not in PREFLIGHT_PATH_POOL
    ):
        raise GlobalDilatedRolloutError("existing preflight shard path binding changed")
    return int(paths[0])


def _build_row_family(
    *,
    canonical_path_id: int,
    v4_model: nn.Module,
    global_model: nn.Module,
    mixed_target: np.ndarray,
    horizon: str,
    device: torch.device,
) -> tuple[tuple[FusedRowSpec, ...], FusedTangentControllerBank, dict[str, Any]]:
    v4_hash_before = state_dict_sha256(v4_model.state_dict())
    specs = (
        FusedRowSpec("zero", canonical_path_id, "zero", "zero", horizon),
        FusedRowSpec(
            "v4-plus-0p5",
            canonical_path_id,
            "learned",
            "frozen-v4",
            horizon,
            gain=0.5,
            controller_binding={"checkpoint_state_sha256": v4_hash_before},
        ),
        FusedRowSpec(
            "v4-minus-0p5",
            canonical_path_id,
            "signed_diagnostic",
            "frozen-v4-sign-diagnostic",
            horizon,
            gain=-0.5,
            controller_binding={"checkpoint_state_sha256": v4_hash_before, "diagnostic_only": 1},
        ),
        FusedRowSpec(
            "global-plus-1",
            canonical_path_id,
            "learned",
            "global-dilated",
            horizon,
            gain=1.0,
            controller_binding={"checkpoint_state_sha256": state_dict_sha256(global_model.state_dict())},
        ),
        FusedRowSpec(
            "source-informed",
            canonical_path_id,
            "oracle",
            "mixed-target-fraction",
            horizon,
            controller_binding={"target_sha256": rollout_array_sha256(mixed_target)},
        ),
    )
    controllers = {
        "v4-plus-0p5": ScaledTangentScoreController(v4_model, 0.5),
        "v4-minus-0p5": SignedDiagnosticTangentScoreController(v4_model, -0.5),
        "global-plus-1": ScaledTangentScoreController(global_model, 1.0),
        "source-informed": TargetFractionOracleController(mixed_target, microsteps=MICROSTEPS).to(device),
    }
    bank = FusedTangentControllerBank(specs, controllers)
    v4_hash_after = state_dict_sha256(v4_model.state_dict())
    if v4_hash_before != v4_hash_after:
        raise GlobalDilatedRolloutError("signed diagnostic mutated the frozen v4 state")
    binding = {
        "row_table": [spec.to_record() for spec in specs],
        "v4_state_sha256": v4_hash_before,
        "global_state_sha256": state_dict_sha256(global_model.state_dict()),
        "target_sha256": rollout_array_sha256(mixed_target),
        "dispatch": "stable_one_row_canonical_order",
        "model_input_contract": "exact_ModelInputs_six_fields",
    }
    return specs, bank, binding


def _run_five_row_exact_smoke_impl(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    durable_attempt_id: str | None = None,
    recovered_durable_covered_seconds: float = 0.0,
) -> dict[str, Any]:
    """Run the only expensive blocking control: one exact five-row shard."""

    existing_control_path = run_dir / "controls/preflight_controls.json"
    if existing_control_path.is_file():
        existing = _read_json(existing_control_path, semantic=True)
        if int(existing.get("passed", 0)) != 1:
            raise GlobalDilatedRolloutError("existing exact preflight control is not passing")
        return existing

    device = torch.device(args.device)
    profile = JacobiRBCudaProfile()
    full_preflight_started = time.perf_counter()
    existing_path_id = _existing_preflight_path_id(run_dir)
    fused_root = run_dir / "controls/exact_smoke/fused_families/five-row/one-shard"
    singleton_root = (
        run_dir
        / "controls/exact_smoke/fused_families/source-singleton/one-shard"
    )
    recovered_prefix_before = _committed_shard_elapsed(
        fused_root
    ) + _committed_shard_elapsed(singleton_root)
    if existing_path_id is not None:
        path_id = existing_path_id
    else:
        realized = _committed_numerical_path_ids(Path(args.repository_root).resolve())
        candidates = [path for path in PREFLIGHT_PATH_POOL if path not in realized]
        if not candidates:
            raise GlobalDilatedRolloutError("no collision-free preflight path remains")
        path_id = int(candidates[0])

    checkpoint = load_verified_frequency1_checkpoint(args.training_parent, device=device)
    source = load_verified_source_target(args.source_run_dir)
    global_model = GlobalDilatedZeroBaselinePredictor(zero_residual=True).to(device).eval()
    global_model.requires_grad_(False)
    specs, bank, controller_binding = _build_row_family(
        canonical_path_id=path_id,
        v4_model=checkpoint.model,
        global_model=global_model,
        mixed_target=source.mixed_target,
        horizon="preflight-eight-step",
        device=device,
    )
    sequence = tuple(reverse_suffix_sequence(ANCHOR_STEP)[:FUSED_SHARD_PHASES])

    # Administrative row names never enter the transition identity.  Verify
    # this before the exact backend is opened.
    plan = build_fused_transition_id_plan(specs, sequence, microsteps=MICROSTEPS, device="cpu")
    permuted = tuple(reversed(specs))
    permuted_plan = build_fused_transition_id_plan(permuted, sequence, microsteps=MICROSTEPS, device="cpu")
    transition_ids_row_permutation_invariant = bool(
        torch.equal(plan.ids, torch.flip(permuted_plan.ids, dims=(3,)))
    )
    if not transition_ids_row_permutation_invariant:
        raise GlobalDilatedRolloutError("row permutation changed canonical transition IDs")

    anchor = _load_preflight_anchor(args)
    actual_v4_inputs = ModelInputs(
        later_full_state=torch.as_tensor(anchor[None, :], dtype=torch.float32, device=device),
        reverse_time=torch.tensor(
            [internal_reverse_time(127, 6, 0.75)], dtype=torch.float64, device=device
        ),
        phase=torch.tensor([6], dtype=torch.long, device=device),
        color=torch.tensor([0], dtype=torch.long, device=device),
        duration=torch.tensor([float(PHASE_DURATIONS[6])], dtype=torch.float32, device=device),
        label=torch.tensor([3], dtype=torch.long, device=device),
    )
    v4_hash_before_inference = state_dict_sha256(checkpoint.model.state_dict())
    with torch.no_grad():
        plus_score = ScaledTangentScoreController(
            checkpoint.model, 0.5
        ).score_prediction(actual_v4_inputs)
        minus_score = SignedDiagnosticTangentScoreController(
            checkpoint.model, -0.5
        ).score_prediction(actual_v4_inputs)
    v4_hash_after_inference = state_dict_sha256(checkpoint.model.state_dict())
    signed_exact_negative = bool(torch.equal(minus_score, -plus_score))
    signed_state_unchanged = v4_hash_before_inference == v4_hash_after_inference
    initial = np.repeat(anchor[None, :], len(specs), axis=0)
    prepared = _prepared_exact_backend(device, profile)
    factory = _exact_reference_factory(
        prepared=prepared,
        profile=profile,
        root_seed=REVERSE_ROOT_SEED,
        stream_role="global_dilated_preflight_shared",
    )
    started = time.perf_counter()
    result = run_fused_reverse_family(
        initial,
        sequence=sequence,
        output_dir=run_dir / "controls/exact_smoke",
        family_name="five-row",
        segment_name="one-shard",
        row_specs=specs,
        controller_bank=bank,
        reference_factory=factory,
        controller_binding=controller_binding,
        rng_binding={
            "root_seed": REVERSE_ROOT_SEED,
            "stream_role": "global_dilated_preflight_shared",
            "canonical_path_id": path_id,
        },
        label=3,
        microsteps=MICROSTEPS,
        device=device,
        reference_contract="certified_exact",
    )
    fused_elapsed = time.perf_counter() - started

    # A one-row source-informed replay is the smallest exact singleton/fused
    # equivalence check while also exercising the known-positive interface.
    oracle_spec = (specs[4],)
    oracle_bank = FusedTangentControllerBank(
        oracle_spec,
        {"source-informed": TargetFractionOracleController(source.mixed_target, microsteps=MICROSTEPS).to(device)},
    )
    singleton = run_fused_reverse_family(
        initial[4:5],
        sequence=sequence,
        output_dir=run_dir / "controls/exact_smoke",
        family_name="source-singleton",
        segment_name="one-shard",
        row_specs=oracle_spec,
        controller_bank=oracle_bank,
        reference_factory=factory,
        controller_binding={"row_table": [oracle_spec[0].to_record()]},
        rng_binding={
            "root_seed": REVERSE_ROOT_SEED,
            "stream_role": "global_dilated_preflight_shared",
            "canonical_path_id": path_id,
        },
        label=3,
        microsteps=MICROSTEPS,
        device=device,
        reference_contract="certified_exact",
    )
    singleton_fused_equal = bool(np.array_equal(singleton.final_state[0], result.final_state[4]))
    fused_health = _strict_fused_exact_health(
        final_state=result.final_state,
        shard_records=result.shard_records,
        row_count=len(specs),
    )
    singleton_health = _strict_fused_exact_health(
        final_state=singleton.final_state,
        shard_records=singleton.shard_records,
        row_count=1,
    )
    zero_error = raw_state_metrics(result.final_state[0], source.mixed_target).squared_l2_error
    oracle_error = raw_state_metrics(result.final_state[4], source.mixed_target).squared_l2_error
    source_improves = oracle_error < zero_error
    maximum_mass_error = float(result.diagnostics.get("maximum_mass_error", math.inf))
    certificate_fraction = float(result.diagnostics.get("certificate_fraction", 0.0))
    peak = int(_maximum_numeric_key(result.shard_records, {"peak_cuda_memory_bytes", "peak_cuda_memory_allocated_bytes", "maximum_cuda_memory_allocated"}))
    total = int(_maximum_numeric_key(result.shard_records, {"total_cuda_memory_bytes"}))
    durable_five_row_seconds = max(fused_elapsed, float(result.elapsed_seconds))
    durable_preflight_seconds = _committed_shard_elapsed(
        fused_root
    ) + _committed_shard_elapsed(singleton_root)
    projected_suffix = durable_five_row_seconds * (MANDATORY_SUFFIX_STEPS // 8) * 1.20
    checks = {
        "transition_ids_row_permutation_invariant": int(transition_ids_row_permutation_invariant),
        "singleton_fused_source_row_bitwise_equal": int(singleton_fused_equal),
        "source_informed_improves_over_zero": int(source_improves),
        "signed_minus_0p5_exact_negative_of_plus_0p5": int(signed_exact_negative),
        "signed_inference_preserves_v4_state_hash": int(signed_state_unchanged),
        "mass_nonnegativity_valid": int(
            maximum_mass_error <= 2.0e-12
            and np.isfinite(result.final_state).all()
            and np.all(result.final_state >= 0.0)
        ),
        "exact_certificate_fraction_one": int(certificate_fraction == 1.0),
        "strict_five_row_exact_health": int(fused_health["passed"]),
        "strict_singleton_exact_health": int(singleton_health["passed"]),
        "projected_suffix_inside_active_cap": int(
            projected_suffix + FORWARD_RESERVE_SECONDS + REPORT_RESERVE_SECONDS < ACTIVE_SECONDS_CAP
        ),
        "cuda_memory_below_cap": int(total == 0 or peak / total < CUDA_MEMORY_FRACTION_CAP),
    }
    passed = int(all(value == 1 for value in checks.values()))
    current_preflight_wall = time.perf_counter() - full_preflight_started
    uncovered_recovered_prefix = max(
        recovered_prefix_before - max(float(recovered_durable_covered_seconds), 0.0),
        0.0,
    )
    accounted_preflight_seconds = max(
        current_preflight_wall,
        durable_preflight_seconds - recovered_prefix_before,
    ) + uncovered_recovered_prefix
    record = _write_semantic(
        run_dir / "controls/preflight_controls.json",
        {
            "schema": VERSION + "-preflight-controls",
            "schema_version": 1,
            "passed": passed,
            "gate_type": "execution/integrity",
            "downstream_action": "opening one fresh exact path",
            "preflight_path_id": path_id,
            "sequence": [list(item) for item in sequence],
            "row_table": [spec.to_record() for spec in specs],
            "checks": checks,
            "five_row_exact_shard_elapsed_seconds": fused_elapsed,
            "durable_five_row_exact_shard_seconds": durable_five_row_seconds,
            "accounted_preflight_seconds": accounted_preflight_seconds,
            "current_invocation_wall_seconds": current_preflight_wall,
            "durable_preflight_shard_seconds": durable_preflight_seconds,
            "durable_recovered_prefix_seconds": recovered_prefix_before,
            "durable_prefix_seconds_already_covered_by_abandoned_attempt": min(
                recovered_prefix_before,
                max(float(recovered_durable_covered_seconds), 0.0),
            ),
            "recovered_committed_prefix_present": int(existing_path_id is not None),
            "unknown_crash_interval_excluded_from_favorable_projection": int(
                existing_path_id is not None
            ),
            "projected_suffix_seconds_with_20pct_margin": projected_suffix,
            "zero_final_squared_l2": zero_error,
            "source_final_squared_l2": oracle_error,
            "peak_cuda_memory_bytes": peak,
            "total_cuda_memory_bytes": total,
            "exact_diagnostics": dict(result.diagnostics),
            "strict_five_row_exact_health": fused_health,
            "strict_singleton_exact_health": singleton_health,
            "signed_diagnostic": {
                "actual_input_state_sha256": rollout_array_sha256(anchor),
                "plus_score_sha256": rollout_array_sha256(plus_score),
                "minus_score_sha256": rollout_array_sha256(minus_score),
                "v4_state_sha256_before": v4_hash_before_inference,
                "v4_state_sha256_after": v4_hash_after_inference,
            },
            "failure_means": "the assembled interface/resource plan is invalid for fresh evaluation",
            "failure_does_not_mean": "the learned scientific mechanism is adverse",
        },
    )
    _record_resource_event(
        run_dir,
        role="five_row_exact_preflight",
        elapsed_seconds=accounted_preflight_seconds,
        peak_cuda_memory_bytes=peak,
        total_cuda_memory_bytes=total,
        detail={
            "preflight_path_id": path_id,
            "projected_suffix_seconds": projected_suffix,
            "includes_backend_prepare_compile_and_commits": 1,
            "durable_attempt_id": durable_attempt_id,
        },
    )
    if not passed:
        resource_keys = {
            "projected_suffix_inside_active_cap",
            "cuda_memory_below_cap",
        }
        if all(
            value == 1 for key, value in checks.items() if key not in resource_keys
        ):
            raise ResourceBoundaryError(
                "blocking exact preflight resource admission failed"
            )
        raise GlobalDilatedRolloutError("blocking exact preflight controls failed")
    return record


def _run_five_row_exact_smoke(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Account every preflight invocation, including failed or resumed work."""

    control_path = run_dir / "controls/preflight_controls.json"
    resumed_control = control_path.is_file()
    fused_root = run_dir / "controls/exact_smoke/fused_families/five-row/one-shard"
    singleton_root = (
        run_dir / "controls/exact_smoke/fused_families/source-singleton/one-shard"
    )
    journal_relative = "controls/active-exact-preflight-attempt.json"
    recovered = _reconcile_durable_attempt_journal(
        run_dir,
        journal_relative=journal_relative,
        role="five_row_exact_preflight_attempt",
    )
    if resumed_control:
        started = time.perf_counter()
        result = _run_five_row_exact_smoke_impl(run_dir, args)
        _record_attempt_wall(
            run_dir,
            role="five_row_exact_preflight_resume_validation",
            started=started,
            failed=False,
            detail={
                "control_semantic_sha256": result["semantic_sha256"],
                "validation_only_no_durable_preflight_recharge": 1,
            },
        )
        return result

    recovered_durable_covered = 0.0
    if recovered is not None:
        recovered_durable_covered = _committed_shard_elapsed(
            fused_root
        ) + _committed_shard_elapsed(singleton_root)
    journal, started = _begin_durable_attempt(
        run_dir,
        journal_relative=journal_relative,
        role="five_row_exact_preflight_attempt",
        detail={
            "durable_roots_relative": [
                fused_root.relative_to(run_dir).as_posix(),
                singleton_root.relative_to(run_dir).as_posix(),
            ],
            "contains_fused_and_singleton_exact_calls": 1,
        },
    )
    durable_before = _committed_shard_elapsed(fused_root) + _committed_shard_elapsed(
        singleton_root
    )
    try:
        result = _run_five_row_exact_smoke_impl(
            run_dir,
            args,
            durable_attempt_id=str(journal["attempt_id"]),
            recovered_durable_covered_seconds=recovered_durable_covered,
        )
    except Exception as exc:
        ledger = _resource_ledger(run_dir)
        covered = any(
            item.get("detail", {}).get("durable_attempt_id")
            == str(journal["attempt_id"])
            for item in ledger.get("events", [])
        )
        if covered:
            (run_dir / journal_relative).unlink(missing_ok=True)
            raise
        durable_after = _committed_shard_elapsed(
            fused_root
        ) + _committed_shard_elapsed(singleton_root)
        wall = time.perf_counter() - started
        try:
            _finish_durable_attempt(
                run_dir,
                journal_relative=journal_relative,
                role="five_row_exact_preflight_attempt",
                journal=journal,
                elapsed_seconds=max(wall, durable_after - durable_before),
                failed=True,
                detail={
                    "invocation_wall_seconds": wall,
                    "durable_before_seconds": durable_before,
                    "durable_committed_shard_seconds": durable_after,
                },
            )
        except ResourceBoundaryError:
            if not isinstance(exc, ResourceBoundaryError):
                raise exc
            raise
        raise
    ledger = _resource_ledger(run_dir)
    if not any(
        item.get("detail", {}).get("durable_attempt_id")
        == str(journal["attempt_id"])
        for item in ledger.get("events", [])
    ):
        raise GlobalDilatedRolloutError(
            "completed exact preflight omitted durable attempt coverage"
        )
    (run_dir / journal_relative).unlink(missing_ok=True)
    return result


def _label_authorization(parent: Path, role: str) -> LabelOpenAuthorization:
    names = {
        "train": ("physical_train_label_open.json", "physical_training"),
        "validation": ("validation_label_open.json", "validation_selection"),
    }
    path_name, purpose = names[role]
    opening = parent / path_name
    opening_record = _read_json(opening, semantic=True)
    return LabelOpenAuthorization(
        cache_root=parent,
        role=role,
        purpose=purpose,
        opening_seal_sha256=str(opening_record["semantic_sha256"]),
    )


def _target_rms(labels: HostLabelStore) -> float:
    reducer = CanonicalRowSquareReducer()
    for start in range(0, labels.row_count, 32):
        rows = np.arange(start, min(start + 32, labels.row_count), dtype=np.int64)
        reducer.update(labels.target_batch(rows, device="cpu"))
    if not math.isfinite(reducer.rms) or reducer.rms <= 0.0:
        raise GlobalDilatedRolloutError("stored bar_Z training target RMS is invalid")
    return float(reducer.rms)


def _clone_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _save_training_checkpoint(
    run_dir: Path,
    *,
    model: nn.Module,
    update: int,
    fingerprint: str,
) -> dict[str, Any]:
    state = _clone_state_dict(model)
    state_hash = state_dict_sha256(state)
    path = run_dir / "training/checkpoints" / f"update-{update:04d}.pt"
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("update") != int(update)
            or payload.get("training_fingerprint") != fingerprint
            or payload.get("state_sha256") != state_hash
            or state_dict_sha256(payload.get("state_dict", {})) != state_hash
        ):
            raise GlobalDilatedRolloutError("existing global checkpoint changed")
    else:
        _atomic_torch(
            path,
            {
                "schema": VERSION + "-global-candidate",
                "schema_version": 1,
                "seed": TRAINING_SEED,
                "update": int(update),
                "training_fingerprint": fingerprint,
                "state_dict": state,
                "state_sha256": state_hash,
                "training_prediction": "m_theta=mobility*q_theta",
                "stored_target": "bar_Z",
                "validation_evidence_used": 0,
            },
        )
    return {
        "update": int(update),
        "path": path.relative_to(run_dir).as_posix(),
        "state_sha256": state_hash,
        "file_sha256": file_fingerprint(path),
    }


def _save_training_progress(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_update: int,
    fingerprint: str,
    checkpoints: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    target_rms: float,
    device: torch.device,
) -> None:
    cuda_device_count = torch.cuda.device_count() if device.type == "cuda" else 0
    cuda_rng_states = (
        tuple(torch.cuda.get_rng_state_all()) if device.type == "cuda" else ()
    )
    if (
        (device.type == "cuda" and cuda_device_count <= 0)
        or len(cuda_rng_states) != cuda_device_count
    ):
        raise GlobalDilatedRolloutError("training CUDA RNG topology changed during save")
    _atomic_torch(
        path,
        {
            "schema": VERSION + "-training-progress",
            "schema_version": 1,
            "fingerprint": fingerprint,
            "completed_update": int(completed_update),
            "model_state_dict": _clone_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "checkpoints": [dict(item) for item in checkpoints],
            "history": [dict(item) for item in history],
            "target_rms": float(target_rms),
            "torch_rng_state": torch.get_rng_state().clone(),
            "cuda_device_count": cuda_device_count,
            "cuda_rng_states": cuda_rng_states,
        },
    )


def _verify_training_progress(
    run_dir: Path,
    saved: Mapping[str, Any],
    *,
    fingerprint: str,
    target_rms: float,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        raw_completed = saved["completed_update"]
        if isinstance(raw_completed, bool) or not isinstance(raw_completed, int):
            raise TypeError("completed_update is not an exact integer")
        completed = int(raw_completed)
        if not isinstance(saved["checkpoints"], (list, tuple)) or not isinstance(
            saved["history"], (list, tuple)
        ):
            raise TypeError("training progress sequences changed")
        checkpoints = [dict(item) for item in saved["checkpoints"]]
        history = [dict(item) for item in saved["history"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise GlobalDilatedRolloutError("training progress schema changed") from exc
    if (
        saved.get("schema") != VERSION + "-training-progress"
        or saved.get("fingerprint") != fingerprint
        or float(saved.get("target_rms", -1.0)) != target_rms
        or completed < 0
        or completed > max(CAP_LADDER)
        or completed % CHECKPOINT_INTERVAL
        or any(
            isinstance(item.get("update"), bool)
            or not isinstance(item.get("update"), int)
            for item in checkpoints
        )
        or [item["update"] for item in checkpoints]
        != list(range(0, completed + 1, CHECKPOINT_INTERVAL))
        or any(
            isinstance(item.get("update"), bool)
            or not isinstance(item.get("update"), int)
            for item in history
        )
        or [item["update"] for item in history]
        != list(range(CHECKPOINT_INTERVAL, completed + 1, CHECKPOINT_INTERVAL))
        or not isinstance(saved.get("optimizer_state_dict"), Mapping)
        or not isinstance(saved.get("torch_rng_state"), Tensor)
        or saved["torch_rng_state"].dtype != torch.uint8
        or saved["torch_rng_state"].ndim != 1
        or not isinstance(saved.get("cuda_rng_states"), (tuple, list))
    ):
        raise GlobalDilatedRolloutError("training progress authority changed")
    _validated_training_rng_states(saved, device=device)
    optimizer_state = saved["optimizer_state_dict"]
    if (
        not isinstance(optimizer_state.get("state"), Mapping)
        or not isinstance(optimizer_state.get("param_groups"), list)
        or not optimizer_state["param_groups"]
    ):
        raise GlobalDilatedRolloutError("training optimizer/RNG authority changed")
    for index, row in enumerate(history, start=1):
        candidate = checkpoints[index]
        numerical_names = (
            "train_raw_mse",
            "normalized_loss",
            "preclip_gradient_norm",
            "accounted_interval_seconds",
        )
        if (
            row.get("checkpoint_state_sha256") != candidate.get("state_sha256")
            or row.get("prediction") != "m_theta"
            or row.get("target") != "bar_Z"
            or not isinstance(row.get("resource_attempt_id"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["resource_attempt_id"])
            or any(
                isinstance(row.get(name), bool)
                or not isinstance(row.get(name), (int, float))
                or not math.isfinite(float(row[name]))
                or float(row[name]) < 0.0
                for name in numerical_names
            )
        ):
            raise GlobalDilatedRolloutError("training history authority changed")
    for candidate in checkpoints:
        path = run_dir / str(candidate.get("path", ""))
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            raise GlobalDilatedRolloutError("training progress checkpoint cannot reopen") from exc
        state = payload.get("state_dict") if isinstance(payload, Mapping) else None
        if (
            not path.is_file()
            or file_fingerprint(path) != candidate.get("file_sha256")
            or payload.get("seed") != TRAINING_SEED
            or payload.get("update") != int(candidate["update"])
            or payload.get("training_fingerprint") != fingerprint
            or not isinstance(state, Mapping)
            or state_dict_sha256(state) != candidate.get("state_sha256")
            or payload.get("state_sha256") != candidate.get("state_sha256")
        ):
            raise GlobalDilatedRolloutError("training progress checkpoint binding changed")
    progress_state = saved.get("model_state_dict")
    if (
        not isinstance(progress_state, Mapping)
        or not checkpoints
        or state_dict_sha256(progress_state) != checkpoints[-1]["state_sha256"]
    ):
        raise GlobalDilatedRolloutError("training progress model state differs from last checkpoint")
    return completed, checkpoints, history


def _validated_training_rng_states(
    saved: Mapping[str, Any], *, device: torch.device
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Validate the serialized RNG state and its exact visible-device topology."""

    torch_state = saved.get("torch_rng_state")
    cuda_states = saved.get("cuda_rng_states")
    bound_device_count = saved.get("cuda_device_count")
    runtime_device_count = torch.cuda.device_count() if device.type == "cuda" else 0
    if (
        not isinstance(torch_state, Tensor)
        or torch_state.dtype != torch.uint8
        or torch_state.ndim != 1
        or not isinstance(cuda_states, (tuple, list))
        or isinstance(bound_device_count, bool)
        or not isinstance(bound_device_count, int)
        or bound_device_count != runtime_device_count
        or (device.type == "cuda" and runtime_device_count <= 0)
        or len(cuda_states) != runtime_device_count
        or any(
            not isinstance(state, Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 1
            or state.numel() <= 0
            for state in cuda_states
        )
        or torch_state.numel() <= 0
    ):
        raise GlobalDilatedRolloutError("training optimizer/RNG authority changed")
    normalized_cuda_states = tuple(state.cpu() for state in cuda_states)
    try:
        # Validate payload size and generator-specific contents without touching
        # either default generator.  Structural uint8/rank checks alone do not
        # reject every malformed state accepted by serialization.
        torch.Generator(device="cpu").set_state(torch_state.cpu())
        for index, state in enumerate(normalized_cuda_states):
            torch.Generator(device=torch.device("cuda", index)).set_state(state)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise GlobalDilatedRolloutError(
            "training optimizer/RNG authority changed"
        ) from exc
    return torch_state, normalized_cuda_states


def _restore_training_rng_state(
    saved: Mapping[str, Any], *, device: torch.device
) -> None:
    """Restore CPU/CUDA RNGs from a progress payload loaded on either device."""

    torch_state, cuda_states = _validated_training_rng_states(saved, device=device)
    # ``torch.load(..., map_location=device)`` also relocates serialized RNG
    # tensors.  Both CPU and CUDA generators require their state argument to be
    # a CPU ByteTensor, including the CUDA states passed to set_rng_state_all.
    torch.set_rng_state(torch_state.cpu())
    if device.type == "cuda" and cuda_states:
        torch.cuda.set_rng_state_all(list(cuda_states))


def _restore_training_history_resource_events(
    run_dir: Path,
    history: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> None:
    """Recover only intervals not already covered by their durable attempt ID."""

    for row in history:
        row_attempt_id = str(row["resource_attempt_id"])
        already_charged = any(
            item.get("detail", {}).get("durable_attempt_id") == row_attempt_id
            for item in _resource_ledger(run_dir).get("events", [])
        )
        if "accounted_interval_seconds" in row and not already_charged:
            _record_resource_event(
                run_dir,
                role="global_training_checkpoint_interval",
                elapsed_seconds=float(row["accounted_interval_seconds"]),
                peak_cuda_memory_bytes=int(row.get("peak_cuda_memory_bytes", 0)),
                total_cuda_memory_bytes=int(row.get("total_cuda_memory_bytes", 0)),
                detail={
                    "end_update": int(row["update"]),
                    "device": str(device),
                    "cuda_memory_measured": 1,
                    "attempt": int(row.get("resource_attempt", 0)),
                    "durable_attempt_id": row_attempt_id,
                    "failed": 0,
                },
            )


def _validation_mse(
    model: nn.Module,
    inputs: HostInputStore,
    labels: HostLabelStore,
    *,
    target_rms: float,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    squared_sum = 0.0
    count = 0
    guard = ModelCallBatchGuard(maximum_batch_size=32)
    with torch.no_grad():
        for rows in inputs.sequential_batches(batch_size=32):
            batch = inputs.batch(rows, device=device)
            target = labels.target_batch(rows, device=device)
            prediction = guard.call(model, batch).to(dtype=torch.float64)
            difference = prediction - target
            squared_sum += float(torch.sum(difference.square()).cpu())
            count += int(difference.numel())
    raw = squared_sum / count
    return raw, raw / (float(target_rms) ** 2)


def _quartile_reference(
    inputs: HostInputStore,
    *,
    training_reference: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    states = np.asarray(inputs.row_array("later_full_state"), dtype=np.float64)
    outer = np.asarray(inputs.row_array("outer_step"), dtype=np.int64)
    quartiles = np.clip(outer // 128, 0, 3)
    means = np.zeros((4, STATE_SIZE), dtype=np.float64)
    p95 = np.zeros(4, dtype=np.float64)
    sorted_ratios: list[np.ndarray] = []
    for quartile in range(4):
        selected = states[quartiles == quartile]
        if selected.size == 0:
            raise GlobalDilatedRolloutError("cache omitted a reverse quartile")
        if training_reference is None:
            mean = np.mean(selected, axis=0, dtype=np.float64)
            radius = np.linalg.norm(selected - mean[None, :], axis=1)
            scale = float(np.quantile(radius, 0.95))
            if not math.isfinite(scale) or scale <= 0.0:
                raise GlobalDilatedRolloutError("training quartile radius scale is invalid")
            means[quartile] = mean
            p95[quartile] = scale
            sorted_ratios.append(np.sort(radius / scale))
        else:
            mean = np.asarray(training_reference["means"])[quartile]
            scale = float(np.asarray(training_reference["p95"])[quartile])
            radius = np.linalg.norm(selected - mean[None, :], axis=1)
            means[quartile] = mean
            p95[quartile] = scale
            sorted_ratios.append(np.sort(radius / scale))
    maximum = max(values.size for values in sorted_ratios)
    # Counts distinguish real values from the finite zero padding.  Persisted
    # numerical evidence never relies on NaN padding or object arrays.
    padded = np.zeros((4, maximum), dtype=np.float64)
    counts = np.zeros(4, dtype=np.int64)
    for index, values in enumerate(sorted_ratios):
        padded[index, : values.size] = values
        counts[index] = values.size
    return {"means": means, "p95": p95, "sorted_ratios": padded, "counts": counts}


def _choose_training_cap(
    run_dir: Path,
    *,
    first_hundred_seconds: float,
    validation_probe_seconds: float,
) -> dict[str, Any]:
    smoke = _read_json(run_dir / "controls/preflight_controls.json", semantic=True)
    ledger = _resource_ledger(run_dir)
    suffix_reserve = max(
        9_000.0,
        float(
            smoke.get(
                "durable_five_row_exact_shard_seconds",
                smoke["five_row_exact_shard_elapsed_seconds"],
            )
        )
        * (MANDATORY_SUFFIX_STEPS // 8)
        * 1.20,
    )
    per_update = first_hundred_seconds / 100.0
    per_validation = max(float(validation_probe_seconds), 0.0)
    projections: list[dict[str, Any]] = []
    chosen: int | None = None
    for cap in CAP_LADDER:
        checkpoint_count = cap // CHECKPOINT_INTERVAL + 1
        projected_training = max(cap - 100, 0) * per_update
        # All candidate checkpoints plus one frozen-v4 comparison traverse
        # the same validation store.  The update-100 timing probe is retained
        # scientifically but is still rerun during deterministic ranking.
        projected_validation = (checkpoint_count + 1) * per_validation
        projected_total = (
            float(ledger.get("active_seconds", 0.0))
            + projected_training
            + projected_validation
            + suffix_reserve
            + FORWARD_RESERVE_SECONDS
            + REPORT_RESERVE_SECONDS
        )
        fits = projected_total <= ACTIVE_SECONDS_CAP
        projections.append(
            {
                "cap": cap,
                "projected_remaining_training_seconds": projected_training,
                "projected_validation_seconds": projected_validation,
                "projected_total_active_seconds": projected_total,
                "fits": int(fits),
            }
        )
        if chosen is None and fits:
            chosen = cap
    if chosen is None:
        raise ResourceBoundaryError(
            "even the 1000-update cap cannot preserve the mandatory exact suffix reserve"
        )
    return _semantic(
        {
            "schema": VERSION + "-training-cap",
            "schema_version": 1,
            "chosen_cap": chosen,
            "cap_ladder": list(CAP_LADDER),
            "first_100_updates_seconds": first_hundred_seconds,
            "seconds_per_update": per_update,
            "validation_probe_seconds": per_validation,
            "suffix_reserve_seconds": suffix_reserve,
            "forward_reserve_seconds": FORWARD_RESERVE_SECONDS,
            "report_reserve_seconds": REPORT_RESERVE_SECONDS,
            "projections": projections,
            "timing_only_choice": 1,
            "first_100_updates_retained": 1,
        }
    )


def _run_validation_timing_probe(
    run_dir: Path,
    *,
    parent: Path,
    model: torch.nn.Module,
    target_rms: float,
    device: torch.device,
) -> tuple[float, float, float, int, str]:
    """Run the retained update-100 traversal under hard-crash accounting."""

    journal_relative = "training/active-validation-timing-probe.json"
    role = "validation_timing_probe"
    journal, started = _begin_durable_attempt(
        run_dir,
        journal_relative=journal_relative,
        role=role,
        detail={"update": 100, "device": str(device)},
    )
    _cuda_reset_peak(device)
    try:
        validation_inputs = open_external_input_store(parent, "validation")
        validation_labels = open_external_label_store(
            parent,
            "validation",
            authorization=_label_authorization(parent, "validation"),
        )
        probe_raw, probe_normalized = _validation_mse(
            model,
            validation_inputs,
            validation_labels,
            target_rms=target_rms,
            device=device,
        )
        elapsed = time.perf_counter() - started + 5.0
        peak, total = _cuda_memory_snapshot(device)
        _finish_durable_attempt(
            run_dir,
            journal_relative=journal_relative,
            role=role,
            journal=journal,
            elapsed_seconds=elapsed,
            failed=False,
            peak_cuda_memory_bytes=peak,
            total_cuda_memory_bytes=total,
            detail={"update": 100, "device": str(device), "cuda_memory_measured": 1},
        )
    except Exception:
        # Python exceptions are charged immediately.  A hard process death or
        # KeyboardInterrupt deliberately leaves the durable start journal for
        # conservative wall-to-resume reconciliation at the next stage entry.
        if (run_dir / journal_relative).is_file():
            try:
                peak, total = _cuda_memory_snapshot(device)
                _finish_durable_attempt(
                    run_dir,
                    journal_relative=journal_relative,
                    role=role,
                    journal=journal,
                    elapsed_seconds=time.perf_counter() - started + 5.0,
                    failed=True,
                    peak_cuda_memory_bytes=peak,
                    total_cuda_memory_bytes=total,
                    detail={
                        "update": 100,
                        "device": str(device),
                        "cuda_memory_measured": 1,
                        "caught_exception": 1,
                    },
                )
            except Exception:
                pass
        raise
    return (
        float(probe_raw),
        float(probe_normalized),
        float(elapsed),
        int(journal["attempt"]),
        str(journal["attempt_id"]),
    )


def _prepare_validation_calibration(
    run_dir: Path,
    *,
    parent: Path,
    history: Sequence[Mapping[str, Any]],
    training_reference: Mapping[str, np.ndarray],
    device: torch.device,
) -> tuple[HostInputStore, HostLabelStore, dict[str, np.ndarray]]:
    """Open validation evidence and persist quartile calibration durably."""

    journal_relative = "training/active-validation-calibration.json"
    role = "validation_calibration_preparation"
    journal, started = _begin_durable_attempt(
        run_dir,
        journal_relative=journal_relative,
        role=role,
        detail={"device": str(device), "history_rows": len(history)},
    )
    _cuda_reset_peak(device)
    try:
        _atomic_csv(run_dir / "training/history.csv", history)
        validation_inputs = open_external_input_store(parent, "validation")
        validation_labels = open_external_label_store(
            parent,
            "validation",
            authorization=_label_authorization(parent, "validation"),
        )
        validation_reference = _quartile_reference(
            validation_inputs,
            training_reference=training_reference,
        )
        atomic_rollout_npz(
            run_dir / "training/on_policy_validation_calibration.npz",
            {
                "training_means": training_reference["means"],
                "training_p95": training_reference["p95"],
                "validation_sorted_ratios": validation_reference["sorted_ratios"],
                "validation_counts": validation_reference["counts"],
            },
        )
        elapsed = time.perf_counter() - started + 5.0
        peak, total = _cuda_memory_snapshot(device)
        _finish_durable_attempt(
            run_dir,
            journal_relative=journal_relative,
            role=role,
            journal=journal,
            elapsed_seconds=elapsed,
            failed=False,
            peak_cuda_memory_bytes=peak,
            total_cuda_memory_bytes=total,
            detail={"device": str(device), "cuda_memory_measured": 1},
        )
    except Exception:
        if (run_dir / journal_relative).is_file():
            try:
                peak, total = _cuda_memory_snapshot(device)
                _finish_durable_attempt(
                    run_dir,
                    journal_relative=journal_relative,
                    role=role,
                    journal=journal,
                    elapsed_seconds=time.perf_counter() - started + 5.0,
                    failed=True,
                    peak_cuda_memory_bytes=peak,
                    total_cuda_memory_bytes=total,
                    detail={
                        "device": str(device),
                        "cuda_memory_measured": 1,
                        "caught_exception": 1,
                    },
                )
            except Exception:
                pass
        raise
    return validation_inputs, validation_labels, validation_reference


def _train_and_select_global(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Train the wrapped m predictor, rank all nonzero checkpoints, and resume exactly."""

    # Reconcile abandoned work before *any* repeated setup can consume or cross
    # the cap.  The order is fixed because an interval may have died before the
    # probe, and either may have died before a recursively repeated preparation.
    training_journal_relative = "training/active-checkpoint-interval.json"
    validation_probe_journal_relative = (
        "training/active-validation-timing-probe.json"
    )
    preparation_journal_relative = "training/active-stage-preparation.json"
    calibration_journal_relative = "training/active-validation-calibration.json"
    selection_journal_relative = "training/active-checkpoint-selection-validation.json"
    for journal_relative, role in (
        (training_journal_relative, "global_training_checkpoint_interval"),
        (validation_probe_journal_relative, "validation_timing_probe"),
        (
            preparation_journal_relative,
            "training_store_open_target_rms_and_quartile_reference",
        ),
        (calibration_journal_relative, "validation_calibration_preparation"),
        (selection_journal_relative, "checkpoint_selection_validation"),
    ):
        _reconcile_durable_attempt_journal(
            run_dir, journal_relative=journal_relative, role=role
        )

    parent = Path(args.training_parent).resolve()
    device = torch.device(args.device)
    enable_deterministic_torch()
    torch.manual_seed(TRAINING_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(TRAINING_SEED)
    _cuda_reset_peak(device)

    preparation_journal, preparation_started = _begin_durable_attempt(
        run_dir,
        journal_relative=preparation_journal_relative,
        role="training_store_open_target_rms_and_quartile_reference",
        detail={"device": str(device)},
    )
    _cuda_reset_peak(device)
    try:
        train_inputs = open_external_input_store(parent, "train")
        train_labels = open_external_label_store(
            parent,
            "train",
            authorization=_label_authorization(parent, "train"),
        )
        if train_inputs.row_count != train_labels.row_count:
            raise GlobalDilatedRolloutError("training input/label row count differs")
        target_rms = _target_rms(train_labels)
        train_reference = _quartile_reference(train_inputs)
        atomic_rollout_npz(
            run_dir / "training/on_policy_training_reference.npz",
            {
                "means": train_reference["means"],
                "p95": train_reference["p95"],
                "training_sorted_ratios": train_reference["sorted_ratios"],
                "training_counts": train_reference["counts"],
            },
        )
        preparation_elapsed = time.perf_counter() - preparation_started + 5.0
        preparation_peak, preparation_total = _cuda_memory_snapshot(device)
        _finish_durable_attempt(
            run_dir,
            journal_relative=preparation_journal_relative,
            role="training_store_open_target_rms_and_quartile_reference",
            journal=preparation_journal,
            elapsed_seconds=preparation_elapsed,
            failed=False,
            peak_cuda_memory_bytes=preparation_peak,
            total_cuda_memory_bytes=preparation_total,
            detail={
                "train_rows": train_inputs.row_count,
                "device": str(device),
                "cuda_memory_measured": 1,
            },
        )
    except Exception:
        if (run_dir / preparation_journal_relative).is_file():
            try:
                preparation_peak, preparation_total = _cuda_memory_snapshot(device)
                _finish_durable_attempt(
                    run_dir,
                    journal_relative=preparation_journal_relative,
                    role="training_store_open_target_rms_and_quartile_reference",
                    journal=preparation_journal,
                    elapsed_seconds=time.perf_counter() - preparation_started + 5.0,
                    failed=True,
                    peak_cuda_memory_bytes=preparation_peak,
                    total_cuda_memory_bytes=preparation_total,
                    detail={
                        "device": str(device),
                        "cuda_memory_measured": 1,
                        "caught_exception": 1,
                    },
                )
            except Exception:
                pass
        raise

    model = GlobalDilatedZeroBaselinePredictor(zero_residual=True).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(TRAINING["learning_rate"]),
        betas=tuple(float(value) for value in TRAINING["betas"]),
        eps=float(TRAINING["epsilon"]),
        weight_decay=float(TRAINING["weight_decay"]),
        amsgrad=bool(TRAINING["amsgrad"]),
    )
    fingerprint = config_fingerprint(
        {
            "schema": VERSION + "-training",
            "scientific_config_sha256": _read_json(run_dir / "scientific_config.json", semantic=True)["semantic_sha256"],
            "seed": TRAINING_SEED,
            "train_input_index": dict(train_inputs.index),
            "train_label_index": dict(train_labels.index),
            "train_label_opening_sha256": train_labels.opening_seal_sha256,
            "target_semantics": "bar_Z",
            "prediction_semantics": "m=mobility*q",
            "target_rms": target_rms,
            "optimizer": dict(TRAINING),
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "cap_ladder": list(CAP_LADDER),
        }
    )
    progress_path = run_dir / "training/progress.pt"
    completed = 0
    checkpoints: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    if progress_path.is_file():
        try:
            saved = torch.load(progress_path, map_location=device, weights_only=False)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            raise GlobalDilatedRolloutError("training progress cannot reopen") from exc
        completed, checkpoints, history = _verify_training_progress(
            run_dir,
            saved,
            fingerprint=fingerprint,
            target_rms=target_rms,
            device=device,
        )
        # If a process died after its progress commit but before the resource
        # ledger commit, restore the conservative durable interval charge.  A
        # hard-crash reconciliation event covers the same interval authority:
        # its role differs, but its durable attempt ID is load-bearing.  Never
        # debit elapsed time twice merely to recover end-update metadata.
        _restore_training_history_resource_events(
            run_dir, history, device=device
        )
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        _restore_training_rng_state(saved, device=device)

    guard = ModelCallBatchGuard(maximum_batch_size=32)
    if not checkpoints:
        with torch.no_grad():
            rows = np.arange(min(32, train_inputs.row_count), dtype=np.int64)
            prediction = guard.call(model, train_inputs.batch(rows, device=device))
            if not bool(torch.all(prediction == 0.0)):
                raise GlobalDilatedRolloutError("global wrapped update-zero output is not exact zero")
        checkpoints.append(
            _save_training_checkpoint(run_dir, model=model, update=0, fingerprint=fingerprint)
        )
        _save_training_progress(
            progress_path,
            model=model,
            optimizer=optimizer,
            completed_update=0,
            fingerprint=fingerprint,
            checkpoints=checkpoints,
            history=history,
            target_rms=target_rms,
            device=device,
        )

    cap_path = run_dir / "training/training_cap.json"
    sealed_cap = _read_json(cap_path, semantic=True) if cap_path.is_file() else None
    if sealed_cap is not None and not isinstance(sealed_cap.get("validation_probe"), Mapping):
        raise GlobalDilatedRolloutError("sealed training cap omits retained validation probe")
    if sealed_cap is not None and not (
        run_dir / "training/validation-probe-update-0100.json"
    ).is_file():
        _write_semantic(
            run_dir / "training/validation-probe-update-0100.json",
            dict(sealed_cap["validation_probe"]),
        )
    chosen_cap = int(sealed_cap["chosen_cap"]) if sealed_cap is not None else None
    first_hundred_started = time.perf_counter() if completed < 100 and chosen_cap is None else None
    # A cap is chosen only after update 100 is durably committed.  Until then
    # the loop limit is exactly the timing probe boundary.
    loop_limit = int(chosen_cap or 100)
    model.train()
    interval_journal: dict[str, Any] | None = None
    interval_started = time.perf_counter()
    if completed < loop_limit:
        interval_journal, interval_started = _begin_durable_attempt(
            run_dir,
            journal_relative=training_journal_relative,
            role="global_training_checkpoint_interval",
            detail={
                "start_update": completed + 1,
                "end_update": min(completed + CHECKPOINT_INTERVAL, loop_limit),
                "training_fingerprint": fingerprint,
                "device": str(device),
            },
        )
    try:
        for update in range(completed + 1, loop_limit + 1):
            indices = deterministic_batch_indices(
                train_inputs.row_count, 32, update - 1, TRAINING_SEED
            )
            inputs = train_inputs.batch(indices, device=device)
            target = train_labels.target_batch(indices, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss, raw, _prediction = _wrapped_training_loss(model, inputs, target, target_rms)
            if not bool(torch.isfinite(loss)):
                raise GlobalDilatedRolloutError("global training loss became nonfinite")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not math.isfinite(float(gradient)):
                raise GlobalDilatedRolloutError("global training gradient became nonfinite")
            optimizer.step()
            if update % CHECKPOINT_INTERVAL != 0:
                continue
            if interval_journal is None:
                raise GlobalDilatedRolloutError("training interval journal disappeared")
            candidate = _save_training_checkpoint(
                run_dir, model=model, update=update, fingerprint=fingerprint
            )
            checkpoints.append(candidate)
            accounted_interval_seconds = (
                time.perf_counter() - interval_started + 5.0
            )
            interval_peak, interval_total = _cuda_memory_snapshot(device)
            history.append(
                {
                    "update": update,
                    "train_raw_mse": float(raw.detach().cpu()),
                    "normalized_loss": float(loss.detach().cpu()),
                    "preclip_gradient_norm": float(gradient),
                    "checkpoint_state_sha256": candidate["state_sha256"],
                    "prediction": "m_theta",
                    "target": "bar_Z",
                    "accounted_interval_seconds": accounted_interval_seconds,
                    "peak_cuda_memory_bytes": interval_peak,
                    "total_cuda_memory_bytes": interval_total,
                    "resource_attempt": int(interval_journal["attempt"]),
                    "resource_attempt_id": str(interval_journal["attempt_id"]),
                    **(
                        {
                            # The projection uses the same conservative interval
                            # charge as the durable ledger, including checkpoint,
                            # progress commit, and boundary overhead.
                            "first_100_updates_seconds": accounted_interval_seconds
                        }
                        if update == 100
                        and chosen_cap is None
                        and first_hundred_started is not None
                        else {}
                    ),
                }
            )
            _save_training_progress(
                progress_path,
                model=model,
                optimizer=optimizer,
                completed_update=update,
                fingerprint=fingerprint,
                checkpoints=checkpoints,
                history=history,
                target_rms=target_rms,
                device=device,
            )
            _finish_durable_attempt(
                run_dir,
                journal_relative=training_journal_relative,
                role="global_training_checkpoint_interval",
                journal=interval_journal,
                elapsed_seconds=accounted_interval_seconds,
                failed=False,
                peak_cuda_memory_bytes=interval_peak,
                total_cuda_memory_bytes=interval_total,
                detail={
                    "end_update": update,
                    "device": str(device),
                    "cuda_memory_measured": 1,
                },
            )
            interval_journal = None
            _cuda_reset_peak(device)
            if update < loop_limit:
                interval_journal, interval_started = _begin_durable_attempt(
                    run_dir,
                    journal_relative=training_journal_relative,
                    role="global_training_checkpoint_interval",
                    detail={
                        "start_update": update + 1,
                        "end_update": min(update + CHECKPOINT_INTERVAL, loop_limit),
                        "training_fingerprint": fingerprint,
                        "device": str(device),
                    },
                )
    except Exception:
        if interval_journal is not None:
            covered = any(
                event.get("detail", {}).get("durable_attempt_id")
                == interval_journal.get("attempt_id")
                for event in _resource_ledger(run_dir).get("events", [])
            )
            if not covered:
                try:
                    peak, total = _cuda_memory_snapshot(device)
                    _finish_durable_attempt(
                        run_dir,
                        journal_relative=training_journal_relative,
                        role="global_training_checkpoint_interval",
                        journal=interval_journal,
                        elapsed_seconds=time.perf_counter() - interval_started + 5.0,
                        failed=True,
                        peak_cuda_memory_bytes=peak,
                        total_cuda_memory_bytes=total,
                        detail={"device": str(device), "caught_exception": 1},
                    )
                except Exception:
                    pass
        raise
    completed = loop_limit

    if chosen_cap is None:
        if completed != 100:
            raise GlobalDilatedRolloutError("training timing probe did not end at update 100")
        probe_rows = [row for row in history if int(row.get("update", -1)) == 100]
        if len(probe_rows) != 1 or not math.isfinite(
            float(probe_rows[0].get("first_100_updates_seconds", math.nan))
        ):
            raise GlobalDilatedRolloutError("durable first-100-update timing is absent")
        first_hundred_seconds = float(probe_rows[0]["first_100_updates_seconds"])
        # Time one complete validation traversal on update 100.  This result is
        # retained in selection and no production update is discarded.
        del train_inputs, train_labels
        (
            probe_raw,
            probe_normalized,
            validation_probe_seconds,
            probe_attempt,
            probe_attempt_id,
        ) = _run_validation_timing_probe(
            run_dir,
            parent=parent,
            model=model,
            target_rms=target_rms,
            device=device,
        )
        cap_record = _choose_training_cap(
            run_dir,
            first_hundred_seconds=first_hundred_seconds,
            validation_probe_seconds=validation_probe_seconds,
        )
        probe_record = {
            "schema": VERSION + "-validation-probe",
            "schema_version": 1,
            "update": 100,
            "raw_mse": probe_raw,
            "normalized_mse": probe_normalized,
            "elapsed_seconds": validation_probe_seconds,
            "resource_attempt": probe_attempt,
            "resource_attempt_id": probe_attempt_id,
            "used_for_scientific_selection": 1,
            "used_for_cap_choice": "timing_only",
        }
        cap_body = dict(cap_record)
        cap_body.pop("semantic_sha256", None)
        cap_record = _semantic({**cap_body, "validation_probe": probe_record})
        # This single atomic commit both authorizes continued training and
        # durably retains the scientific/timing probe that bought the cap.
        atomic_write_json(cap_path, cap_record)
        _write_semantic(
            run_dir / "training/validation-probe-update-0100.json", probe_record
        )
        chosen_cap = int(cap_record["chosen_cap"])
        # Resume the same optimizer/model/RNG state rather than restarting the
        # first production hundred updates.
        return _train_and_select_global(run_dir, args)

    if completed < chosen_cap:
        # This only occurs when a resumed progress artifact predates the sealed
        # cap and the first loop used its saved boundary.
        return _train_and_select_global(run_dir, args)
    if completed != chosen_cap:
        raise GlobalDilatedRolloutError("training progressed beyond the sealed timing cap")

    del train_inputs, train_labels
    validation_inputs, validation_labels, validation_reference = (
        _prepare_validation_calibration(
            run_dir,
            parent=parent,
            history=history,
            training_reference=train_reference,
            device=device,
        )
    )

    risk_rows: list[dict[str, Any]] = []
    selection_journal, validation_started = _begin_durable_attempt(
        run_dir,
        journal_relative=selection_journal_relative,
        role="checkpoint_selection_validation",
        detail={
            "candidate_count": len(checkpoints),
            "v4_comparison": 1,
            "training_fingerprint": fingerprint,
            "device": str(device),
        },
    )
    _cuda_reset_peak(device)
    try:
        for candidate in checkpoints:
            update = int(candidate["update"])
            checkpoint_path = run_dir / str(candidate["path"])
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            payload_state = payload.get("state_dict") if isinstance(payload, Mapping) else None
            if (
                file_fingerprint(checkpoint_path) != candidate["file_sha256"]
                or payload.get("seed") != TRAINING_SEED
                or payload.get("update") != update
                or payload.get("training_fingerprint") != fingerprint
                or not isinstance(payload_state, Mapping)
                or state_dict_sha256(payload_state) != candidate["state_sha256"]
                or payload.get("state_sha256") != candidate["state_sha256"]
            ):
                raise GlobalDilatedRolloutError("global candidate checkpoint file changed")
            model.load_state_dict(payload_state, strict=True)
            raw, normalized = _validation_mse(
                model,
                validation_inputs,
                validation_labels,
                target_rms=target_rms,
                device=device,
            )
            risk_rows.append(
                {
                    "model": "global-dilated",
                    "update": update,
                    "raw_validation_mse_m_vs_bar_Z": raw,
                    "normalized_validation_mse": normalized,
                    "state_sha256": candidate["state_sha256"],
                    "checkpoint_path": candidate["path"],
                    "checkpoint_file_sha256": candidate["file_sha256"],
                    "finite": int(math.isfinite(raw) and math.isfinite(normalized)),
                }
            )
        frozen_v4 = load_verified_frequency1_checkpoint(parent, device=device)
        v4_raw, v4_normalized = _validation_mse(
            frozen_v4.model,
            validation_inputs,
            validation_labels,
            target_rms=target_rms,
            device=device,
        )
    except Exception:
        try:
            peak, total = _cuda_memory_snapshot(device)
            _finish_durable_attempt(
                run_dir,
                journal_relative=selection_journal_relative,
                role="checkpoint_selection_validation",
                journal=selection_journal,
                elapsed_seconds=time.perf_counter() - validation_started + 5.0,
                failed=True,
                peak_cuda_memory_bytes=peak,
                total_cuda_memory_bytes=total,
                detail={
                    "candidate_count_completed": len(risk_rows),
                    "v4_comparison_completed": 0,
                    "device": str(device),
                    "cuda_memory_measured": 1,
                },
            )
        except Exception:
            pass
        raise
    validation_elapsed = time.perf_counter() - validation_started
    selection_peak, selection_total = _cuda_memory_snapshot(device)
    _finish_durable_attempt(
        run_dir,
        journal_relative=selection_journal_relative,
        role="checkpoint_selection_validation",
        journal=selection_journal,
        elapsed_seconds=validation_elapsed,
        failed=False,
        peak_cuda_memory_bytes=selection_peak,
        total_cuda_memory_bytes=selection_total,
        detail={
            "candidate_count": len(risk_rows),
            "v4_comparison": 1,
            "device": str(device),
            "cuda_memory_measured": 1,
        },
    )
    nonzero = [
        row
        for row in risk_rows
        if int(row["update"]) > 0 and int(row["finite"]) == 1
    ]
    if not nonzero:
        raise GlobalDilatedRolloutError("all nonzero global validation candidates are nonfinite")
    selected = min(
        nonzero,
        key=lambda row: (float(row["raw_validation_mse_m_vs_bar_Z"]), int(row["update"])),
    )
    zero_row = next(row for row in risk_rows if int(row["update"]) == 0)
    record = _write_semantic(
        run_dir / "selection.json",
        {
            "schema": VERSION + "-selection",
            "schema_version": 1,
            "selected": selected,
            "rule": "finite nonzero minimum validation MSE of wrapped m against stored bar_Z; earlier update tie",
            "continue_if_worse_than_zero": 1,
            "all_nonzero_worse_than_zero": int(
                all(float(row["raw_validation_mse_m_vs_bar_Z"]) >= float(zero_row["raw_validation_mse_m_vs_bar_Z"]) for row in nonzero)
            ),
            "zero_checkpoint": zero_row,
            "frozen_v4_comparison": {
                "checkpoint_state_sha256": frozen_v4.state_sha256,
                "raw_validation_mse_m_vs_bar_Z": v4_raw,
                "normalized_validation_mse": v4_normalized,
                "used_for_selection": 0,
            },
            "train_target_rms_bar_Z": target_rms,
            "candidate_count": len(risk_rows),
            "candidates": risk_rows,
            "confirmation_evidence_used": 0,
            "post_hoc_choice": "timing-selected update cap only; model selection rule prespecified",
        },
    )
    return record


def _selected_global_model(run_dir: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    selection = _read_json(run_dir / "selection.json", semantic=True)
    selected = dict(selection["selected"])
    path = run_dir / str(selected["checkpoint_path"])
    if file_fingerprint(path) != selected["checkpoint_file_sha256"]:
        raise GlobalDilatedRolloutError("selected global checkpoint file changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict")
    cap = _read_json(run_dir / "training/training_cap.json", semantic=True)
    progress = torch.load(
        run_dir / "training/progress.pt", map_location="cpu", weights_only=False
    )
    if (
        not isinstance(state, Mapping)
        or state_dict_sha256(state) != selected["state_sha256"]
        or payload.get("seed") != TRAINING_SEED
        or payload.get("update") != int(selected["update"])
        or int(selected["update"]) <= 0
        or int(selected["update"]) > int(cap["chosen_cap"])
        or payload.get("training_fingerprint") != progress.get("fingerprint")
        or payload.get("state_sha256") != selected["state_sha256"]
    ):
        raise GlobalDilatedRolloutError("selected global checkpoint state changed")
    architecture = global_dilated_architecture_contract()
    if (
        int(architecture.get("passed", 0)) != 1
        or int(architecture.get("trainable_parameter_count", -1))
        != GLOBAL_DILATED_PARAMETER_COUNT
    ):
        raise GlobalDilatedRolloutError("selected global architecture contract changed")
    model = GlobalDilatedZeroBaselinePredictor(zero_residual=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    if state_dict_sha256(model.state_dict()) != selected["state_sha256"]:
        raise GlobalDilatedRolloutError("strict-loaded selected global state differs")
    return model, selected


def _seal_evaluation(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Seal every scientific evaluation choice before fresh path allocation."""

    theory_path = run_dir / "controls/theory_to_code.json"
    preflight_path = run_dir / "controls/preflight_controls.json"
    path = run_dir / "evaluation_freeze.json"
    if not path.is_file() and (
        (run_dir / "path_usage.json").exists() or (run_dir / "fresh_forward").exists()
    ):
        raise GlobalDilatedRolloutError("fresh evidence exists before evaluation freeze")
    selection = _read_json(run_dir / "selection.json", semantic=True)
    selected_checkpoint_path = run_dir / str(selection["selected"]["checkpoint_path"])
    selected_payload = torch.load(
        selected_checkpoint_path, map_location="cpu", weights_only=True
    )
    bindings = _read_json(run_dir / "input_bindings.json", semantic=True)
    source = load_verified_source_target(args.source_run_dir)
    scale = fixed_rendering_scale(
        source.source_image, source.mixed_target, float(source.metadata["lambda_mix"])
    )
    sequence = tuple(reverse_suffix_sequence(ANCHOR_STEP))
    if len(sequence) != MANDATORY_SUFFIX_STEPS * PHASE_COUNT:
        raise GlobalDilatedRolloutError("exact suffix sequence length changed")
    row_table = [
        {"row": 0, "key": "zero", "controller": "exact zero", "gain": None},
        {"row": 1, "key": "v4-plus-0p5", "controller": "frozen v4", "gain": 0.5},
        {"row": 2, "key": "v4-minus-0p5", "controller": "signed diagnostic frozen v4", "gain": -0.5},
        {"row": 3, "key": "global-plus-1", "controller": "selected global-dilated", "gain": 1.0},
        {"row": 4, "key": "source-informed", "controller": "mixed-target fraction oracle", "gain": None},
    ]
    body = {
        "schema": VERSION + "-evaluation-freeze",
        "schema_version": 1,
        "sealed": 1,
        "sealed_at": _utc_now(),
        "global_checkpoint": dict(selection["selected"]),
        "global_training_fingerprint": selected_payload.get("training_fingerprint"),
        "selection_file_sha256": file_fingerprint(run_dir / "selection.json"),
        "v4_checkpoint": dict(bindings["frozen_v4_checkpoint"]),
        "theory_to_code_file_sha256": file_fingerprint(theory_path),
        "preflight_controls_file_sha256": file_fingerprint(preflight_path),
        "row_order": list(ROW_ORDER),
        "row_table": row_table,
        "reverse_sequence": [list(item) for item in sequence],
        "reverse_sequence_sha256": semantic_sha256([list(item) for item in sequence]),
        "microsteps": MICROSTEPS,
        "phase_count": PHASE_COUNT,
        "forward_root_seed": FORWARD_ROOT_SEED,
        "reverse_root_seed": REVERSE_ROOT_SEED,
        "fresh_path_pool": list(FRESH_PATH_POOL),
        "exact_backend": "certified_exact",
        "approximate_primary_authorized": 0,
        "no_postprocessing": 1,
        "primary_metric": "E_zero-E_candidate where E is final squared L2 to mixed target",
        "practical_effect_labels": {
            "materially_improves": f"relative improvement >= {PRACTICAL_RELATIVE_THRESHOLD}",
            "positive_small": f"0 < relative improvement < {PRACTICAL_RELATIVE_THRESHOLD}",
            "adverse": "improvement <= 0",
            "gate_type": "diagnostic threshold",
        },
        "rendering_scale": scale.to_dict(),
        "resource_caps": {
            "active_seconds": ACTIVE_SECONDS_CAP,
            "storage_bytes": STORAGE_CAP_BYTES,
            "cuda_fraction": CUDA_MEMORY_FRACTION_CAP,
        },
        "optional_positive_branch": {
            "trigger": "global relative improvement >= practical threshold",
            "rows": ["zero", "global-plus-1", "source-informed"],
            "same_forward_path_resumed_to_step": 511,
            "admission": "remaining exact budget only",
        },
        "source_target_sha256": rollout_array_sha256(source.mixed_target),
        "confirmation_evidence_used": 0,
    }
    if path.is_file():
        existing = _read_json(path, semantic=True)
        # Timestamp is evidentiary, not a resumable scientific choice.
        comparable_existing = dict(existing)
        comparable_existing.pop("semantic_sha256", None)
        comparable_existing.pop("sealed_at", None)
        comparable_body = dict(body)
        comparable_body.pop("sealed_at", None)
        if comparable_existing != comparable_body:
            raise GlobalDilatedRolloutError("evaluation choices changed after sealing")
        return existing
    return _write_semantic(path, body)


def _allocate_fresh_path(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    freeze_path = run_dir / "evaluation_freeze.json"
    if not freeze_path.is_file():
        raise GlobalDilatedRolloutError("fresh path allocation requires sealed evaluation choices")
    freeze = _read_json(freeze_path, semantic=True)
    if int(freeze.get("sealed", 0)) != 1:
        raise GlobalDilatedRolloutError("evaluation freeze is not sealed")
    freeze_hash = file_fingerprint(freeze_path)
    path = run_dir / "path_usage.json"
    if path.is_file():
        existing = _read_json(path, semantic=True)
        if existing.get("evaluation_freeze_file_sha256") != freeze_hash:
            raise GlobalDilatedRolloutError("fresh allocation no longer matches evaluation freeze")
        return existing
    started = time.perf_counter()
    try:
        realized = _committed_numerical_path_ids(Path(args.repository_root).resolve())
        available = [candidate for candidate in FRESH_PATH_POOL if candidate not in realized]
        if not available:
            raise GlobalDilatedRolloutError("fresh path pool is exhausted by committed numerical artifacts")
        selected = int(available[0])
        record = _write_semantic(
            path,
            {
                "schema": VERSION + "-path-usage",
                "schema_version": 1,
                "evaluation_freeze_file_sha256": freeze_hash,
                "scan_root": str(Path(args.repository_root).resolve() / "runs"),
                "committed_path_id_count": len(realized),
                "committed_path_ids_sha256": semantic_sha256(sorted(realized)),
                "fresh_pool": list(FRESH_PATH_POOL),
                "collisions_in_fresh_pool": sorted(set(FRESH_PATH_POOL) & realized),
                "fresh_path_id": selected,
                "allocation_rule": "smallest collision-free path after evaluation seal",
                "all_five_rows_share_canonical_path_id": 1,
            },
        )
    except Exception:
        try:
            _record_attempt_wall(
                run_dir,
                role="fresh_path_repository_scan_and_allocation",
                started=started,
                failed=True,
                detail={"evaluation_freeze_file_sha256": freeze_hash},
            )
        except Exception:
            pass
        raise
    _record_attempt_wall(
        run_dir,
        role="fresh_path_repository_scan_and_allocation",
        started=started,
        failed=False,
        detail={
            "fresh_path_id": selected,
            "path_usage_file_sha256": file_fingerprint(path),
        },
    )
    return record


def _run_forward_to_127(
    run_dir: Path, args: argparse.Namespace, path_usage: Mapping[str, Any]
) -> dict[str, Any]:
    forward_root = run_dir / "fresh_forward/forward_shards/fresh-main-path"
    path_id = int(path_usage.get("fresh_path_id", -1))
    journal_relative = "fresh_forward/active-forward-to-127-attempt.json"
    journal, started = _begin_durable_attempt(
        run_dir,
        journal_relative=journal_relative,
        role="fresh_exact_forward_to_127_attempt",
        detail={
            "path_id": path_id,
            "step_limit": ANCHOR_STEP + 1,
            "durable_root_relative": forward_root.relative_to(run_dir).as_posix(),
        },
    )
    durable_before = _committed_shard_elapsed(forward_root)
    try:
        freeze_hash = file_fingerprint(run_dir / "evaluation_freeze.json")
        if path_usage.get("evaluation_freeze_file_sha256") != freeze_hash:
            raise GlobalDilatedRolloutError("forward path allocation predates the evaluation seal")
        source = load_verified_source_target(args.source_run_dir)
        freeze = _read_json(run_dir / "evaluation_freeze.json", semantic=True)
        if rollout_array_sha256(source.mixed_target) != freeze.get("source_target_sha256"):
            raise GlobalDilatedRolloutError("fresh forward source target differs from evaluation freeze")
        profile = JacobiRBCudaProfile()
        result = run_forward_trajectory(
            source.mixed_target,
            anchor_steps=(ANCHOR_STEP,),
            output_dir=run_dir / "fresh_forward",
            trajectory_name="fresh-main-path",
            path_ids=(path_id,),
            root_seed=FORWARD_ROOT_SEED,
            profile=profile,
            step_limit=ANCHOR_STEP + 1,
            device=torch.device(args.device),
        )
        anchor = np.ascontiguousarray(result.anchors[ANCHOR_STEP], dtype=np.float64)
        atomic_rollout_npz(
            run_dir / "fresh_forward/anchor-step-0127.npz", {"state": anchor}
        )
        diagnostics = result.diagnostics
        if (
            int(diagnostics.get("passed", 0)) != 1
            or int(diagnostics.get("restart_chain_valid", 0)) != 1
            or float(diagnostics.get("authorization_fraction", 0.0)) != 1.0
            or float(diagnostics.get("certificate_fraction", 0.0)) != 1.0
            or int(diagnostics.get("forbidden_event_count", -1)) != 0
            or int(diagnostics.get("output_state_nonfinite_count", -1)) != 0
            or int(diagnostics.get("output_state_negative_count", -1)) != 0
            or float(diagnostics.get("maximum_output_state_mass_error", math.inf)) > 2.0e-12
        ):
            raise GlobalDilatedRolloutError("fresh exact forward aggregate health failed")
        elapsed_before_summary_commit = time.perf_counter() - started
        record = _write_semantic(
            run_dir / "fresh_forward/forward_summary.json",
            {
                "schema": VERSION + "-fresh-forward",
                "schema_version": 1,
                "evaluation_freeze_file_sha256": freeze_hash,
                "path_usage_file_sha256": file_fingerprint(run_dir / "path_usage.json"),
                "path_id": path_id,
                "root_seed": FORWARD_ROOT_SEED,
                "step_limit": ANCHOR_STEP + 1,
                "step_511_opened": 0,
                "anchor_step": ANCHOR_STEP,
                "source_target_sha256": rollout_array_sha256(source.mixed_target),
                "anchor_state_sha256": rollout_array_sha256(anchor),
                "anchor_archive_sha256": rollout_file_sha256(run_dir / "fresh_forward/anchor-step-0127.npz"),
                "wall_seconds_through_anchor_commit_and_health": elapsed_before_summary_commit,
                "result": result.to_record(),
                "diagnostics": dict(result.diagnostics),
            },
        )
    except Exception as exc:
        durable_after = _committed_shard_elapsed(forward_root)
        wall = time.perf_counter() - started
        try:
            _finish_durable_attempt(
                run_dir,
                journal_relative=journal_relative,
                role="fresh_exact_forward_to_127_attempt",
                journal=journal,
                elapsed_seconds=max(wall, durable_after - durable_before),
                failed=True,
                detail={
                    "path_id": path_id,
                    "invocation_wall_seconds": wall,
                    "durable_before_seconds": durable_before,
                    "durable_committed_shard_seconds": durable_after,
                },
            )
        except ResourceBoundaryError:
            if not isinstance(exc, ResourceBoundaryError):
                raise exc
            raise
        raise
    elapsed = time.perf_counter() - started
    peak = int(_maximum_numeric_key(result.shard_records, {"peak_cuda_memory_bytes", "peak_cuda_memory_allocated_bytes"}))
    total = int(_maximum_numeric_key(result.shard_records, {"total_cuda_memory_bytes"}))
    durable_after = _committed_shard_elapsed(forward_root)
    _finish_durable_attempt(
        run_dir,
        journal_relative=journal_relative,
        role="fresh_exact_forward_to_127_attempt",
        journal=journal,
        elapsed_seconds=max(elapsed, durable_after - durable_before),
        failed=False,
        peak_cuda_memory_bytes=peak,
        total_cuda_memory_bytes=total,
        detail={
            "path_id": path_id,
            "shards": len(result.shard_records),
            "invocation_wall_seconds": elapsed,
            "durable_before_seconds": durable_before,
            "durable_committed_shard_seconds": durable_after,
        },
    )
    return record


def _build_five_rows(
    run_dir: Path, args: argparse.Namespace
) -> tuple[tuple[FusedRowSpec, ...], FusedTangentControllerBank, dict[str, Any]]:
    path_usage = _read_json(run_dir / "path_usage.json", semantic=True)
    device = torch.device(args.device)
    v4 = load_verified_frequency1_checkpoint(args.training_parent, device=device)
    global_model, selected = _selected_global_model(run_dir, device)
    source = load_verified_source_target(args.source_run_dir)
    freeze = _read_json(run_dir / "evaluation_freeze.json", semantic=True)
    bindings = _verify_bound_paths(run_dir, args)
    frozen_v4 = freeze.get("v4_checkpoint", {})
    if (
        v4.state_sha256 != frozen_v4.get("state_sha256")
        or v4.checkpoint_file_sha256 != frozen_v4.get("checkpoint_file_sha256")
        or v4.state_sha256 != bindings.get("frozen_v4_checkpoint", {}).get("state_sha256")
    ):
        raise GlobalDilatedRolloutError("runtime v4 checkpoint differs from evaluation freeze")
    if selected != freeze.get("global_checkpoint"):
        raise GlobalDilatedRolloutError("runtime global checkpoint differs from evaluation freeze")
    selected_payload = torch.load(
        run_dir / str(selected["checkpoint_path"]), map_location="cpu", weights_only=True
    )
    if (
        file_fingerprint(run_dir / "selection.json")
        != freeze.get("selection_file_sha256")
        or selected_payload.get("training_fingerprint")
        != freeze.get("global_training_fingerprint")
    ):
        raise GlobalDilatedRolloutError("selection/training binding differs from evaluation freeze")
    if rollout_array_sha256(source.mixed_target) != freeze.get("source_target_sha256"):
        raise GlobalDilatedRolloutError("runtime source target differs from evaluation freeze")
    specs, bank, binding = _build_row_family(
        canonical_path_id=int(path_usage["fresh_path_id"]),
        v4_model=v4.model,
        global_model=global_model,
        mixed_target=source.mixed_target,
        horizon="fresh-exact-128-step-suffix",
        device=device,
    )
    if tuple(spec.row_key for spec in specs) != ROW_ORDER:
        raise GlobalDilatedRolloutError("five-row order differs from evaluation freeze")
    if [item["key"] for item in freeze["row_table"]] != list(ROW_ORDER):
        raise GlobalDilatedRolloutError("sealed row order changed")
    binding.update(
        evaluation_freeze_file_sha256=file_fingerprint(run_dir / "evaluation_freeze.json"),
        selected_global=selected,
    )
    return specs, bank, binding


def _suffix_admission_callback(
    run_dir: Path,
    family_root: Path,
    *,
    attempt_started: float,
    additional_active_reserve_seconds: float = 0.0,
    additional_storage_reserve_bytes: int = 0,
) -> Any:
    smoke = _read_json(run_dir / "controls/preflight_controls.json", semantic=True)
    conservative = max(
        float(
            smoke.get(
                "durable_five_row_exact_shard_seconds",
                smoke["five_row_exact_shard_elapsed_seconds"],
            )
        )
        * 1.20,
        1.0,
    )
    attempt_role = (
        "positive_branch_complete_three_row_exact_attempt"
        if "positive" in family_root.parts
        else "mandatory_exact_five_row_suffix_attempt"
    )

    def callback(plan: FusedShardExecutionPlan) -> None:
        ledger = _resource_ledger(run_dir)
        durable = _committed_shard_elapsed(family_root)
        covered = max(
            (
                float(item.get("detail", {}).get("durable_committed_shard_seconds", 0.0))
                for item in ledger.get("events", [])
                if item.get("role")
                in {attempt_role, attempt_role + "_abandoned_attempt"}
            ),
            default=0.0,
        )
        unledgered = max(durable - covered, 0.0)
        current_wall = max(time.perf_counter() - attempt_started, 0.0)
        if (
            float(ledger.get("active_seconds", 0.0))
            + max(unledgered, current_wall)
            + conservative
            + max(float(additional_active_reserve_seconds), 0.0)
            + REPORT_RESERVE_SECONDS
            > ACTIVE_SECONDS_CAP
        ):
            raise ResourceBoundaryError(
                f"exact suffix shard {plan.shard_index} cannot preserve report reserve"
            )
        if (
            _directory_bytes(run_dir)
            + max(int(additional_storage_reserve_bytes), 0)
            >= STORAGE_CAP_BYTES
        ):
            raise ResourceBoundaryError("exact suffix cannot admit another shard under storage cap")

    return callback


def _run_exact_suffix(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Run/resume all 16 certified-exact five-row reverse shards."""

    family_root = run_dir / "suffix/fused_families/fresh-five-row/suffix-128"
    row_count = len(ROW_ORDER)
    journal_relative = "suffix/active-mandatory-exact-attempt.json"
    journal, started = _begin_durable_attempt(
        run_dir,
        journal_relative=journal_relative,
        role="mandatory_exact_five_row_suffix_attempt",
        detail={
            "row_count": row_count,
            "reverse_steps": MANDATORY_SUFFIX_STEPS,
            "durable_root_relative": family_root.relative_to(run_dir).as_posix(),
        },
    )
    durable_before = _committed_shard_elapsed(family_root)
    try:
        anchor_arrays = _load_npz(run_dir / "fresh_forward/anchor-step-0127.npz")
        anchor = anchor_arrays.get("state")
        if anchor is None or anchor.dtype != np.float64 or anchor.shape != (STATE_SIZE,):
            raise GlobalDilatedRolloutError("fresh step-127 anchor artifact changed")
        # Model/checkpoint loads, immutable-input remeasurement, backend
        # preparation, and seed/controller construction are part of the exact
        # invocation wall and therefore start after the timer above.
        specs, bank, controller_binding = _build_five_rows(run_dir, args)
        row_count = len(specs)
        initial = np.repeat(anchor[None, :], row_count, axis=0)
        sequence = tuple(reverse_suffix_sequence(ANCHOR_STEP))
        freeze = _read_json(run_dir / "evaluation_freeze.json", semantic=True)
        if semantic_sha256([list(item) for item in sequence]) != freeze["reverse_sequence_sha256"]:
            raise GlobalDilatedRolloutError("reverse sequence differs from evaluation freeze")
        profile = JacobiRBCudaProfile()
        device = torch.device(args.device)
        prepared = _prepared_exact_backend(device, profile)
        stream_role = "global_dilated_fresh_five_row_exact"
        factory = _exact_reference_factory(
            prepared=prepared,
            profile=profile,
            root_seed=REVERSE_ROOT_SEED,
            stream_role=stream_role,
        )
        result = run_fused_reverse_family(
            initial,
            sequence=sequence,
            output_dir=run_dir / "suffix",
            family_name="fresh-five-row",
            segment_name="suffix-128",
            row_specs=specs,
            controller_bank=bank,
            reference_factory=factory,
            controller_binding=controller_binding,
            rng_binding={
                "root_seed": REVERSE_ROOT_SEED,
                "stream_role": stream_role,
                "canonical_path_id": int(specs[0].canonical_path_id),
                "all_rows_shared": 1,
            },
            label=3,
            microsteps=MICROSTEPS,
            device=device,
            capture_coordinates={
                (96, 0): "completed-032",
                (64, 0): "completed-064",
                (32, 0): "completed-096",
                (0, 0): "completed-128",
            },
            before_uncommitted_shard=_suffix_admission_callback(
                run_dir, family_root, attempt_started=started
            ),
            reference_contract="certified_exact",
        )
        if len(result.shard_records) != MANDATORY_SUFFIX_STEPS // 8:
            raise GlobalDilatedRolloutError("exact suffix did not commit all 16 shards")
        if tuple(spec.row_key for spec in result.row_specs) != ROW_ORDER:
            raise GlobalDilatedRolloutError("exact suffix result row order changed")
        strict_health = _strict_fused_exact_health(
            final_state=result.final_state,
            shard_records=result.shard_records,
            row_count=row_count,
        )
        record = _write_semantic(
            run_dir / "suffix/family_summary.json",
            {
                "schema": VERSION + "-exact-five-row-suffix",
                "schema_version": 1,
                "completed": 1,
                "reference_contract": "certified_exact",
                "row_order": list(ROW_ORDER),
                "result": result.to_record(),
                "controller_binding": controller_binding,
                "strict_exact_health": strict_health,
                "shard_record_paths": [
                    f"suffix/fused_families/fresh-five-row/suffix-128/shard-{index:04d}.json"
                    for index in range(len(result.shard_records))
                ],
                "failed_rows_suppressed": 0,
            },
        )
    except Exception as exc:
        durable_after = _committed_shard_elapsed(family_root)
        wall = time.perf_counter() - started
        try:
            _finish_durable_attempt(
                run_dir,
                journal_relative=journal_relative,
                role="mandatory_exact_five_row_suffix_attempt",
                journal=journal,
                elapsed_seconds=max(wall, durable_after - durable_before),
                failed=True,
                detail={
                    "row_count": row_count,
                    "invocation_wall_seconds": wall,
                    "durable_before_seconds": durable_before,
                    "durable_committed_shard_seconds": durable_after,
                },
            )
        except ResourceBoundaryError:
            if not isinstance(exc, ResourceBoundaryError):
                raise exc
            raise
        raise
    elapsed = time.perf_counter() - started
    peak = int(_maximum_numeric_key(result.shard_records, {"peak_cuda_memory_bytes", "peak_cuda_memory_allocated_bytes", "maximum_cuda_memory_allocated"}))
    total = int(_maximum_numeric_key(result.shard_records, {"total_cuda_memory_bytes"}))
    durable_after = _committed_shard_elapsed(family_root)
    _finish_durable_attempt(
        run_dir,
        journal_relative=journal_relative,
        role="mandatory_exact_five_row_suffix_attempt",
        journal=journal,
        elapsed_seconds=max(elapsed, durable_after - durable_before),
        failed=False,
        peak_cuda_memory_bytes=peak,
        total_cuda_memory_bytes=total,
        detail={
            "shard_count": len(result.shard_records),
            "row_count": row_count,
            "invocation_wall_seconds": elapsed,
            "durable_before_seconds": durable_before,
            "durable_committed_shard_seconds": durable_after,
        },
    )
    return record


def _aggregate_existing_shards(run_dir: Path) -> dict[str, Any]:
    """Reopen the 16 committed shard endpoints and create the raw 17-state archive."""

    anchor = _load_npz(run_dir / "fresh_forward/anchor-step-0127.npz").get("state")
    if anchor is None or anchor.dtype != np.float64 or anchor.shape != (STATE_SIZE,):
        raise GlobalDilatedRolloutError("fresh anchor cannot seed trajectory aggregation")
    root = run_dir / "suffix/fused_families/fresh-five-row/suffix-128"
    states = [np.repeat(anchor[None, :], len(ROW_ORDER), axis=0)]
    previous_hash = rollout_array_sha256(states[0])
    for index in range(MANDATORY_SUFFIX_STEPS // 8):
        record_path = root / f"shard-{index:04d}.json"
        archive_path = root / f"shard-{index:04d}.npz"
        record = _read_json(record_path, semantic=True)
        if int(record.get("committed", 0)) != 1:
            raise GlobalDilatedRolloutError("trajectory aggregation encountered an uncommitted shard")
        if record.get("input_state_sha256") != previous_hash:
            raise GlobalDilatedRolloutError("fused reverse shard chain input hash changed")
        arrays = _load_npz(archive_path)
        if set(arrays) != {"state"}:
            raise GlobalDilatedRolloutError("fused reverse shard archive schema changed")
        state = np.ascontiguousarray(arrays["state"])
        if state.dtype != np.float64 or state.shape != (len(ROW_ORDER), STATE_SIZE):
            raise GlobalDilatedRolloutError("fused reverse endpoint shape/dtype changed")
        if (
            not np.isfinite(state).all()
            or np.any(state < 0.0)
            or float(np.max(np.abs(np.sum(state, axis=1) - 1.0))) > 2.0e-12
        ):
            raise GlobalDilatedRolloutError(
                "fused reverse intermediate endpoint violates numerical health"
            )
        if record.get("state_file_sha256") != rollout_file_sha256(archive_path):
            raise GlobalDilatedRolloutError("fused reverse endpoint file hash changed")
        previous_hash = rollout_array_sha256(state)
        if record.get("output_state_sha256") != previous_hash:
            raise GlobalDilatedRolloutError("fused reverse endpoint array hash changed")
        states.append(state)
    stacked = np.ascontiguousarray(np.stack(states, axis=1), dtype=np.float64)
    completed = np.arange(0, MANDATORY_SUFFIX_STEPS + 1, 8, dtype=np.int64)
    if stacked.shape != (len(ROW_ORDER), 17, STATE_SIZE):
        raise GlobalDilatedRolloutError("trajectory shard-boundary aggregate shape changed")
    trajectory_path = run_dir / "suffix/trajectory_shard_boundaries.npz"
    atomic_rollout_npz(
        trajectory_path,
        {"states": stacked, "completed_reverse_steps": completed},
    )
    milestone_indices = np.asarray([step // 8 for step in MILESTONE_STEPS], dtype=np.int64)
    milestone_states = np.ascontiguousarray(stacked[:, milestone_indices, :])
    milestone_path = run_dir / "suffix/milestones.npz"
    atomic_rollout_npz(
        milestone_path,
        {
            "states": milestone_states,
            "completed_reverse_steps": np.asarray(MILESTONE_STEPS, dtype=np.int64),
        },
    )
    return _write_semantic(
        run_dir / "suffix/trajectory_aggregation.json",
        {
            "schema": VERSION + "-trajectory-aggregation",
            "schema_version": 1,
            "source": "start plus 16 existing eight-outer-step exact shard endpoints",
            "states_shape": list(stacked.shape),
            "completed_reverse_steps": completed.tolist(),
            "states_sha256": rollout_array_sha256(stacked),
            "trajectory_file_sha256": rollout_file_sha256(trajectory_path),
            "milestone_steps": list(MILESTONE_STEPS),
            "milestone_states_sha256": rollout_array_sha256(milestone_states),
            "milestone_file_sha256": rollout_file_sha256(milestone_path),
            "chain_valid": 1,
            "hot_loop_modified_for_capture": 0,
        },
    )


def _save_contact_sheet(path: Path, cells: Sequence[np.ndarray], *, columns: int) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise GlobalDilatedRolloutError("Pillow is required for contact sheets") from exc
    if not cells or columns <= 0:
        raise GlobalDilatedRolloutError("contact sheet cells/columns are invalid")
    rows = math.ceil(len(cells) / columns)
    canvas = Image.new("L", (columns * 28, rows * 28), color=0)
    for index, cell in enumerate(cells):
        value = np.asarray(cell)
        if value.dtype != np.uint8 or value.shape != (28, 28):
            raise GlobalDilatedRolloutError("contact sheet image cell changed")
        canvas.paste(Image.fromarray(value, mode="L"), ((index % columns) * 28, (index // columns) * 28))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".png", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        canvas.save(temporary, format="PNG")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _row_quarter_mechanism(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row_index, row_key in enumerate(ROW_ORDER):
        quarters: dict[str, Any] = {}
        for quarter in range(4):
            selected = []
            for shard_index, shard in enumerate(shards):
                # This is reverse-progress quarter, deliberately distinct
                # from the matching training-cache outer-step quartile used
                # by the radius calibration below.
                if shard_index // 4 == quarter:
                    values = shard.get("per_row_diagnostics", [])
                    if len(values) > row_index:
                        selected.append(values[row_index])
            sums: dict[str, float] = {}
            maxima: dict[str, float] = {}
            for entry in selected:
                for key, value in entry.items():
                    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        continue
                    if key.endswith("_squared_sum") or key.endswith("_count"):
                        sums[key] = sums.get(key, 0.0) + float(value)
                    elif "maximum" in key:
                        maxima[key] = max(maxima.get(key, 0.0), float(value))

            def rms(prefix: str) -> float | None:
                count = sums.get(prefix + "_count", 0.0)
                square = sums.get(prefix + "_squared_sum", 0.0)
                return math.sqrt(square / count) if count > 0.0 else None

            score_rms = rms("score")
            control_rms = rms("control_fraction_displacement")
            reference_rms = rms("reference_fraction_displacement")
            quarters[str(quarter)] = {
                "reverse_progress_steps": [quarter * 32, (quarter + 1) * 32],
                "shard_count": len(selected),
                "score_rms": score_rms,
                "control_fraction_displacement_rms": control_rms,
                "reference_fraction_displacement_rms": reference_rms,
                "control_reference_ratio": (
                    control_rms / reference_rms
                    if control_rms is not None and reference_rms not in (None, 0.0)
                    else None
                ),
                "maxima": maxima,
            }
        result[row_key] = quarters
    return result


def _compute_metrics_and_images(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Compute raw objective/secondary metrics, drift descriptions, and every image."""

    trajectory = _load_npz(run_dir / "suffix/trajectory_shard_boundaries.npz")
    states = trajectory.get("states")
    completed = trajectory.get("completed_reverse_steps")
    if (
        states is None
        or completed is None
        or states.dtype != np.float64
        or states.shape != (len(ROW_ORDER), 17, STATE_SIZE)
        or not np.array_equal(completed, np.arange(0, 129, 8, dtype=np.int64))
    ):
        raise GlobalDilatedRolloutError("trajectory metric input changed")
    source = load_verified_source_target(args.source_run_dir)
    freeze = _read_json(run_dir / "evaluation_freeze.json", semantic=True)
    scale = FixedRenderingScale(**freeze["rendering_scale"])
    rows: list[dict[str, Any]] = []
    for boundary_index, reverse_steps in enumerate(completed.tolist()):
        zero_state = states[0, boundary_index]
        for row_index, row_key in enumerate(ROW_ORDER):
            state = states[row_index, boundary_index]
            mixed = raw_state_metrics(state, source.mixed_target).to_dict()
            unmixed = raw_state_metrics(state, source.source_image).to_dict()
            separation = state - zero_state
            rows.append(
                {
                    "row_index": row_index,
                    "row_key": row_key,
                    "boundary_index": boundary_index,
                    "completed_reverse_steps": int(reverse_steps),
                    **{f"mixed_target_{key}": value for key, value in mixed.items()},
                    **{f"unmixed_source_{key}": value for key, value in unmixed.items()},
                    "row_vs_zero_squared_separation": float(np.dot(separation, separation)),
                    "nonfinite_count": int(np.count_nonzero(~np.isfinite(state))),
                    "negative_count": int(np.count_nonzero(state < 0.0)),
                    "is_milestone": int(int(reverse_steps) in MILESTONE_STEPS),
                }
            )
    _atomic_csv(run_dir / "suffix/metrics.csv", rows)

    final_objectives: dict[str, Any] = {}
    zero_metric = raw_state_metrics(states[0, -1], source.mixed_target)
    for row_index, row_key in enumerate(ROW_ORDER):
        metric = raw_state_metrics(states[row_index, -1], source.mixed_target)
        final_objectives[row_key] = {
            **metric.to_dict(),
            **(
                {
                    "paired_squared_l2_improvement_over_zero": 0.0,
                    "relative_paired_squared_l2_improvement_over_zero": 0.0,
                }
                if row_index == 0
                else paired_metric_improvement(metric, zero_metric)
            ),
        }

    calibration = _load_npz(run_dir / "training/on_policy_validation_calibration.npz")
    means = calibration.get("training_means")
    p95 = calibration.get("training_p95")
    validation_ratios = calibration.get("validation_sorted_ratios")
    counts = calibration.get("validation_counts")
    if means is None or p95 is None or validation_ratios is None or counts is None:
        raise GlobalDilatedRolloutError("on-policy calibration artifact changed")
    drift_rows: list[dict[str, Any]] = []
    global_index = ROW_ORDER.index("global-plus-1")
    for boundary_index, reverse_steps in enumerate(completed.tolist()):
        current_outer_step = max(ANCHOR_STEP - int(reverse_steps), 0)
        quartile = min(current_outer_step // 128, 3)
        radius = float(np.linalg.norm(states[global_index, boundary_index] - means[quartile]))
        ratio = radius / float(p95[quartile])
        valid = np.asarray(validation_ratios[quartile, : int(counts[quartile])])
        percentile = float(np.searchsorted(valid, ratio, side="right") / valid.size)
        separation = states[global_index, boundary_index] - states[0, boundary_index]
        drift_rows.append(
            {
                "boundary_index": boundary_index,
                "completed_reverse_steps": int(reverse_steps),
                "matching_training_quartile": int(quartile),
                "global_zero_squared_separation": float(np.dot(separation, separation)),
                "global_radius_from_training_quartile_mean": radius,
                "global_radius_normalized_by_training_p95": ratio,
                "validation_calibrated_percentile": percentile,
                "threshold_or_execution_gate": 0,
            }
        )

    shard_root = run_dir / "suffix/fused_families/fresh-five-row/suffix-128"
    shard_records = [
        _read_json(shard_root / f"shard-{index:04d}.json", semantic=True)
        for index in range(MANDATORY_SUFFIX_STEPS // 8)
    ]
    mechanism = _write_semantic(
        run_dir / "suffix/mechanism.json",
        {
            "schema": VERSION + "-mechanism",
            "schema_version": 1,
            "telemetry_source": "fused bank per-row diagnostics",
            "per_reverse_quarter": _row_quarter_mechanism(shard_records),
            "on_policy_drift": drift_rows,
            "drift_is_descriptive_only": 1,
            "off_policy_failure_inferred": 0,
        },
    )

    image_root = run_dir / "images"
    milestone_indices = [step // 8 for step in MILESTONE_STEPS]
    raw_cells: list[np.ndarray] = []
    demixed_cells: list[np.ndarray] = []
    image_records: list[dict[str, Any]] = []
    for milestone, boundary_index in zip(MILESTONE_STEPS, milestone_indices, strict=True):
        milestone_raw: list[np.ndarray] = []
        milestone_demixed: list[np.ndarray] = []
        for row_index, row_key in enumerate(ROW_ORDER):
            state = states[row_index, boundary_index]
            raw_image = render_raw_density(state, scale)
            demixed_image = render_background_demixed(state, scale)
            raw_path = image_root / f"raw/{row_key}/step-{milestone:03d}.png"
            demixed_path = image_root / f"demixed/{row_key}/step-{milestone:03d}.png"
            save_png(raw_path, raw_image)
            save_png(demixed_path, demixed_image)
            raw_cells.append(raw_image)
            demixed_cells.append(demixed_image)
            milestone_raw.append(raw_image)
            milestone_demixed.append(demixed_image)
            image_records.append(
                {
                    "row_key": row_key,
                    "completed_reverse_steps": milestone,
                    "state_sha256": rollout_array_sha256(state),
                    "raw_path": raw_path.relative_to(run_dir).as_posix(),
                    "raw_sha256": file_fingerprint(raw_path),
                    "demixed_path": demixed_path.relative_to(run_dir).as_posix(),
                    "demixed_sha256": file_fingerprint(demixed_path),
                }
            )
        _save_contact_sheet(
            image_root / f"contact-sheets/milestone-{milestone:03d}-raw.png",
            milestone_raw,
            columns=5,
        )
        _save_contact_sheet(
            image_root / f"contact-sheets/milestone-{milestone:03d}-demixed.png",
            milestone_demixed,
            columns=5,
        )
    _save_contact_sheet(image_root / "contact-sheets/all-milestones-raw.png", raw_cells, columns=5)
    _save_contact_sheet(image_root / "contact-sheets/all-milestones-demixed.png", demixed_cells, columns=5)
    _save_contact_sheet(image_root / "contact-sheets/final-raw.png", raw_cells[-5:], columns=5)
    _save_contact_sheet(image_root / "contact-sheets/final-demixed.png", demixed_cells[-5:], columns=5)

    return _write_semantic(
        run_dir / "suffix/summary.json",
        {
            "schema": VERSION + "-objective-summary",
            "schema_version": 1,
            "research_mode": "exploratory",
            "independent_unit": "one path/image",
            "p_values_or_confidence_intervals": 0,
            "row_order": list(ROW_ORDER),
            "primary_final_objectives": final_objectives,
            "metric_row_count": len(rows),
            "metric_file_sha256": file_fingerprint(run_dir / "suffix/metrics.csv"),
            "mechanism_file_sha256": file_fingerprint(run_dir / "suffix/mechanism.json"),
            "rendering_scale": scale.to_dict(),
            "images": image_records,
            "failed_or_adverse_outputs_suppressed": 0,
        },
    )


def _effect_label(relative: float | None) -> str:
    if relative is None or not math.isfinite(float(relative)):
        return "invalid"
    if float(relative) >= PRACTICAL_RELATIVE_THRESHOLD:
        return "materially_improves"
    if float(relative) > 0.0:
        return "positive_small"
    return "adverse"


def _outcome_body(run_dir: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "suffix/summary.json", semantic=True)
    selection = _read_json(run_dir / "selection.json", semantic=True)
    objectives = summary["primary_final_objectives"]
    labels = {
        row: _effect_label(objectives[row].get("relative_paired_squared_l2_improvement_over_zero"))
        for row in ROW_ORDER[1:]
    }
    global_label = labels["global-plus-1"]
    sign_label = labels["v4-minus-0p5"]
    plus_label = labels["v4-plus-0p5"]
    source_label = labels["source-informed"]
    selected_risk = float(selection["selected"]["normalized_validation_mse"])
    zero_risk = float(selection["zero_checkpoint"]["normalized_validation_mse"])
    if source_label not in {"materially_improves", "positive_small"}:
        outcome = "source_interface_control_failed"
        action = "repair_the_complete_source_controller_composition_before_interpreting_or_scaling_learned_rows"
    elif global_label == "materially_improves":
        outcome = "global_material_improvement"
        action = "attempt_same_path_complete_zero_global_source_reconstruction_if_exact_budget_fits"
    elif sign_label == "materially_improves" and global_label != "materially_improves":
        outcome = "sign_order_leading"
        action = "reconcile_exact_theory_to_code_sign_table_before_any_controller_sign_change"
    elif global_label == "positive_small":
        outcome = "global_positive_small"
        action = "at_most_one_unchanged_fresh_replication_then_complete_path_or_training_objective_question"
    elif selected_risk < zero_risk and global_label == "adverse":
        outcome = "validation_better_suffix_adverse"
        action = "preserve_architecture_and_pivot_to_rollout_alignment_or_target_derivation"
    elif (
        plus_label == sign_label == global_label == "adverse"
        and source_label in {"materially_improves", "positive_small"}
    ):
        outcome = "all_learned_adverse_controls_pass"
        action = "stop_nearby_jacobi_repairs_run_small_standard_ddpm_sanity_then_direct_heat_potential_edge_score_pivot"
    else:
        outcome = "mixed_or_uninformative"
        action = "do_not_scale_redesign_one_direct_decisive_experiment"
    return {
            "schema": VERSION + "-outcome",
            "schema_version": 1,
            "outcome": outcome,
            "required_next_action": action,
            "effect_labels": labels,
            "practical_relative_threshold": PRACTICAL_RELATIVE_THRESHOLD,
            "threshold_type": "diagnostic threshold",
            "global_validation_normalized_mse": selected_risk,
            "zero_validation_normalized_mse": zero_risk,
            "primary_objectives": objectives,
            "one_path_scope": 1,
        }


def _classify_outcome(run_dir: Path) -> dict[str, Any]:
    return _write_semantic(run_dir / "outcome.json", _outcome_body(run_dir))


def _positive_branch_rows(
    run_dir: Path, args: argparse.Namespace
) -> tuple[tuple[FusedRowSpec, ...], FusedTangentControllerBank, dict[str, Any]]:
    path_usage = _read_json(run_dir / "path_usage.json", semantic=True)
    device = torch.device(args.device)
    global_model, selected = _selected_global_model(run_dir, device)
    source = load_verified_source_target(args.source_run_dir)
    path_id = int(path_usage["fresh_path_id"])
    specs = (
        FusedRowSpec("zero", path_id, "zero", "zero", "same-path-complete"),
        FusedRowSpec(
            "global-plus-1",
            path_id,
            "learned",
            "global-dilated",
            "same-path-complete",
            gain=1.0,
            controller_binding={"checkpoint_state_sha256": selected["state_sha256"]},
        ),
        FusedRowSpec(
            "source-informed",
            path_id,
            "oracle",
            "mixed-target-fraction",
            "same-path-complete",
            controller_binding={"target_sha256": rollout_array_sha256(source.mixed_target)},
        ),
    )
    controllers = {
        "global-plus-1": ScaledTangentScoreController(global_model, 1.0),
        "source-informed": TargetFractionOracleController(source.mixed_target, microsteps=MICROSTEPS).to(device),
    }
    bank = FusedTangentControllerBank(specs, controllers)
    binding = {
        "row_table": [spec.to_record() for spec in specs],
        "selected_global": selected,
        "target_sha256": rollout_array_sha256(source.mixed_target),
        "trigger_outcome_file_sha256": file_fingerprint(run_dir / "outcome.json"),
    }
    return specs, bank, binding


def _optional_postprocess_reserve(run_dir: Path) -> dict[str, Any] | None:
    """Seal a conservative reserve derived from completed mandatory work.

    Optional reconstruction is never allowed to consume the report reserve or
    to finish its sampler with no budget left to reopen the raw chain and save
    the task images.  The mandatory postprocessor is the closest measured
    operation.  Its largest successful wall time supplies the timing baseline;
    its complete derived artifact set supplies the storage baseline.
    """

    successful = [
        event
        for event in _resource_ledger(run_dir).get("events", [])
        if event.get("role") == "mandatory_objective_postprocessing"
        and event.get("detail", {}).get("failed") == 0
        and float(event.get("elapsed_seconds", 0.0)) > 0.0
    ]
    if not successful:
        return None
    mandatory_images = sorted((run_dir / "images").rglob("*.png"))
    if not mandatory_images:
        return None
    source_paths = [
        run_dir / "suffix/trajectory_shard_boundaries.npz",
        run_dir / "suffix/milestones.npz",
        run_dir / "suffix/metrics.csv",
        run_dir / "suffix/mechanism.json",
        run_dir / "suffix/summary.json",
        *mandatory_images,
    ]
    if not source_paths or any(not path.is_file() for path in source_paths):
        return None
    source_rows = [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "size": int(path.stat().st_size),
            "sha256": file_fingerprint(path),
        }
        for path in source_paths
    ]
    mandatory_seconds = max(float(event["elapsed_seconds"]) for event in successful)
    mandatory_bytes = sum(int(row["size"]) for row in source_rows)
    body = {
        "schema": VERSION + "-optional-postprocess-reserve",
        "schema_version": 1,
        "source_role": "mandatory_objective_postprocessing",
        "source_successful_event_ids": sorted(
            str(event["event_id"]) for event in successful
        ),
        "source_max_elapsed_seconds": mandatory_seconds,
        "source_artifact_bytes": mandatory_bytes,
        "source_artifacts": source_rows,
        "time_multiplier": OPTIONAL_POSTPROCESS_TIME_MULTIPLIER,
        "minimum_seconds": OPTIONAL_POSTPROCESS_MIN_SECONDS,
        "minimum_storage_bytes": OPTIONAL_POSTPROCESS_MIN_STORAGE_BYTES,
        "reserve_seconds": max(
            OPTIONAL_POSTPROCESS_MIN_SECONDS,
            mandatory_seconds * OPTIONAL_POSTPROCESS_TIME_MULTIPLIER,
        ),
        "reserve_storage_bytes": max(
            OPTIONAL_POSTPROCESS_MIN_STORAGE_BYTES,
            int(math.ceil(mandatory_bytes * 3.0)),
        ),
        "report_reserve_seconds_separate": REPORT_RESERVE_SECONDS,
    }
    reserve_path = run_dir / "positive/postprocess_reserve.json"
    expected = _semantic(body)
    if reserve_path.is_file():
        existing = _read_json(reserve_path, semantic=True)
        if existing != expected:
            raise GlobalDilatedRolloutError(
                "optional postprocess reserve differs from mandatory authority"
            )
        return existing
    return _write_semantic(reserve_path, body)


def _optional_committed_shard_count(root: Path) -> int:
    count = 0
    if not root.is_dir():
        return 0
    for record_path in sorted(root.glob("shard-*.json")):
        archive_path = record_path.with_suffix(".npz")
        record = _read_json(record_path, semantic=True)
        if archive_path.is_file() and record.get("committed") == 1:
            count += 1
    return count


def _optional_forward_observed_shard_seconds(forward_root: Path) -> tuple[float, ...]:
    values: list[float] = []
    if not forward_root.is_dir():
        return ()
    for record_path in sorted(forward_root.glob("shard-*.json")):
        record = _read_json(record_path, semantic=True)
        if record.get("committed") != 1 or int(record.get("start_step", -1)) < 128:
            continue
        elapsed = record.get("elapsed_seconds")
        if (
            not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0.0
        ):
            raise GlobalDilatedRolloutError(
                "optional forward shard elapsed authority changed"
            )
        values.append(float(elapsed))
    return tuple(values)


def _admit_optional_forward_shard(
    run_dir: Path,
    *,
    forward_root: Path,
    attempt_started: float,
    shard_index: int,
    projected_reverse_seconds: float,
    postprocess_reserve_seconds: float,
    postprocess_reserve_storage_bytes: int,
) -> dict[str, float]:
    """Admit exactly one new forward shard using all observed current work."""

    ledger = _resource_ledger(run_dir)
    observed = _optional_forward_observed_shard_seconds(forward_root)
    baseline = FORWARD_RESERVE_SECONDS / (MANDATORY_SUFFIX_STEPS // 8)
    next_shard = max(
        baseline,
        (max(observed) * 1.20) if observed else 0.0,
    )
    current_wall = max(time.perf_counter() - attempt_started, 0.0)
    projected = (
        float(ledger.get("active_seconds", 0.0))
        + current_wall
        + next_shard
        + max(float(projected_reverse_seconds), 0.0)
        + max(float(postprocess_reserve_seconds), 0.0)
        + REPORT_RESERVE_SECONDS
    )
    if projected > ACTIVE_SECONDS_CAP:
        raise ResourceBoundaryError(
            f"optional forward shard {shard_index} cannot preserve exact reverse, postprocess, and report reserves"
        )
    if (
        _directory_bytes(run_dir)
        + max(int(postprocess_reserve_storage_bytes), 0)
        >= STORAGE_CAP_BYTES
    ):
        raise ResourceBoundaryError(
            f"optional forward shard {shard_index} cannot fit under storage cap"
        )
    return {
        "current_invocation_wall_seconds": current_wall,
        "next_shard_projection_seconds": next_shard,
        "projected_reverse_seconds": max(float(projected_reverse_seconds), 0.0),
        "postprocess_reserve_seconds": max(float(postprocess_reserve_seconds), 0.0),
        "postprocess_reserve_storage_bytes": max(
            int(postprocess_reserve_storage_bytes), 0
        ),
        "projected_total_active_seconds": projected,
    }


def _build_optional_positive_postprocess_artifacts(
    run_dir: Path,
    *,
    source: Any,
    anchor: np.ndarray,
    forward: Any,
    result: Any,
    strict_health: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen the complete optional chain and build every promised artifact."""

    zero = raw_state_metrics(result.final_state[0], source.mixed_target)
    global_metric = raw_state_metrics(result.final_state[1], source.mixed_target)
    oracle = raw_state_metrics(result.final_state[2], source.mixed_target)
    positive_root = run_dir / "positive/fused_families/same-path-three-row/complete-512"
    boundary_states = [np.repeat(anchor[None, :], 3, axis=0)]
    previous_hash = rollout_array_sha256(boundary_states[0])
    for index in range(OUTER_STEPS // 8):
        shard = _read_json(positive_root / f"shard-{index:04d}.json", semantic=True)
        arrays = _load_npz(positive_root / f"shard-{index:04d}.npz")
        state = np.ascontiguousarray(arrays["state"], dtype=np.float64)
        if (
            shard.get("input_state_sha256") != previous_hash
            or state.shape != (3, STATE_SIZE)
            or not np.isfinite(state).all()
            or np.any(state < 0.0)
            or float(np.max(np.abs(np.sum(state, axis=1) - 1.0))) > 2.0e-12
        ):
            raise GlobalDilatedRolloutError("positive-branch raw shard chain changed")
        previous_hash = rollout_array_sha256(state)
        if shard.get("output_state_sha256") != previous_hash:
            raise GlobalDilatedRolloutError("positive-branch raw shard hash changed")
        boundary_states.append(state)
    positive_states = np.ascontiguousarray(np.stack(boundary_states, axis=1))
    atomic_rollout_npz(
        run_dir / "positive/trajectory_shard_boundaries.npz",
        {
            "states": positive_states,
            "completed_reverse_steps": np.arange(0, 513, 8, dtype=np.int64),
        },
    )
    positive_milestones = np.ascontiguousarray(
        positive_states[:, [0, 16, 32, 48, 64], :]
    )
    atomic_rollout_npz(
        run_dir / "positive/milestones.npz",
        {
            "states": positive_milestones,
            "completed_reverse_steps": np.asarray(
                (0, 128, 256, 384, 512), dtype=np.int64
            ),
        },
    )
    scale = fixed_rendering_scale(
        source.source_image, source.mixed_target, float(source.metadata["lambda_mix"])
    )
    positive_cells: list[np.ndarray] = []
    for milestone_index, milestone in enumerate((0, 128, 256, 384, 512)):
        for row_index, row_key in enumerate(
            ("zero", "global-plus-1", "source-informed")
        ):
            image = render_raw_density(
                positive_milestones[row_index, milestone_index], scale
            )
            save_png(
                run_dir / f"positive/images/raw/{row_key}/step-{milestone:03d}.png",
                image,
            )
            positive_cells.append(image)
    _save_contact_sheet(
        run_dir / "positive/images/contact-sheet-raw.png",
        positive_cells,
        columns=3,
    )
    return {
        "schema": VERSION + "-positive-branch",
        "schema_version": 1,
        "triggered": 1,
        "attempted": 1,
        "completed": 1,
        "same_forward_path_resumed": 1,
        "forward": forward.to_record(),
        "reverse": result.to_record(),
        "strict_exact_health": dict(strict_health),
        "trajectory_states_sha256": rollout_array_sha256(positive_states),
        "milestone_states_sha256": rollout_array_sha256(positive_milestones),
        "task_images_saved": 15,
        "final_metrics": {
            "zero": zero.to_dict(),
            "global-plus-1": {
                **global_metric.to_dict(),
                **paired_metric_improvement(global_metric, zero),
            },
            "source-informed": {
                **oracle.to_dict(),
                **paired_metric_improvement(oracle, zero),
            },
        },
        "claim_scope": "optional exploratory same-path complete reconstruction only",
    }


def _run_optional_positive_postprocessing(
    run_dir: Path,
    *,
    source: Any,
    anchor: np.ndarray,
    forward: Any,
    result: Any,
    strict_health: Mapping[str, Any],
    reserve: Mapping[str, Any],
) -> dict[str, Any]:
    """Durably account optional shard reopening, arrays, images, and record."""

    journal_relative = "positive/active-positive-postprocessing.json"
    role = "optional_positive_postprocessing"
    _reconcile_durable_attempt_journal(
        run_dir, journal_relative=journal_relative, role=role
    )
    reserve_seconds = float(reserve["reserve_seconds"])
    reserve_storage = int(reserve["reserve_storage_bytes"])
    ledger = _resource_ledger(run_dir)
    if (
        float(ledger.get("active_seconds", 0.0))
        + reserve_seconds
        + REPORT_RESERVE_SECONDS
        > ACTIVE_SECONDS_CAP
    ):
        raise ResourceBoundaryError(
            "optional positive postprocessing cannot preserve its measured reserve and report reserve"
        )
    if _directory_bytes(run_dir) + reserve_storage >= STORAGE_CAP_BYTES:
        raise ResourceBoundaryError(
            "optional positive postprocessing cannot preserve its storage reserve"
        )
    journal, started = _begin_durable_attempt(
        run_dir,
        journal_relative=journal_relative,
        role=role,
        detail={
            "operations": [
                "reopen_64_exact_shards",
                "stack_boundary_and_milestone_arrays",
                "write_two_npz",
                "write_15_raw_pngs_and_contact_sheet",
                "commit_positive_branch_record",
            ],
            "reserve_semantic_sha256": reserve["semantic_sha256"],
            "reserve_seconds": reserve_seconds,
            "reserve_storage_bytes": reserve_storage,
        },
    )
    try:
        body = _build_optional_positive_postprocess_artifacts(
            run_dir,
            source=source,
            anchor=anchor,
            forward=forward,
            result=result,
            strict_health=strict_health,
        )
        _finish_durable_attempt(
            run_dir,
            journal_relative=journal_relative,
            role=role,
            journal=journal,
            elapsed_seconds=time.perf_counter() - started + 5.0,
            failed=False,
            detail={
                "cuda_memory_measured": 1,
                "device": "cpu",
                "reserve_semantic_sha256": reserve["semantic_sha256"],
            },
        )
    except Exception:
        if (run_dir / journal_relative).is_file():
            try:
                _finish_durable_attempt(
                    run_dir,
                    journal_relative=journal_relative,
                    role=role,
                    journal=journal,
                    elapsed_seconds=time.perf_counter() - started + 5.0,
                    failed=True,
                    detail={
                        "cuda_memory_measured": 1,
                        "device": "cpu",
                        "reserve_semantic_sha256": reserve["semantic_sha256"],
                        "caught_exception": 1,
                    },
                )
            except Exception:
                pass
        raise
    return _write_semantic(run_dir / "positive_branch.json", body)


def _run_positive_complete_path_impl(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Attempt the predeclared same-path full reconstruction only when it fits."""

    outcome = _read_json(run_dir / "outcome.json", semantic=True)
    path = run_dir / "positive_branch.json"
    try:
        _reconcile_durable_attempt_journal(
            run_dir,
            journal_relative="positive/active-positive-postprocessing.json",
            role="optional_positive_postprocessing",
        )
    except ResourceBoundaryError as exc:
        return _write_semantic(
            path,
            {
                "schema": VERSION + "-positive-branch",
                "schema_version": 1,
                "triggered": int(
                    outcome.get("outcome") == "global_material_improvement"
                ),
                "attempted": 1,
                "completed": 0,
                "mandatory_suffix_objective_preserved": 1,
                "failure_domain": "resource_budget",
                "failure_type": ResourceBoundaryError.__name__,
                "failure_message": str(exc),
                "reason": (
                    "an interrupted optional postprocessing attempt was durably "
                    "reconciled and exhausted the frozen resource boundary"
                ),
                "required_next_action": (
                    "seal the completed mandatory objective and defer optional reconstruction"
                ),
                "complete_path_claim_authorized": 0,
            },
        )
    if path.is_file():
        return _read_json(path, semantic=True)
    if outcome.get("outcome") != "global_material_improvement":
        return _write_semantic(
            path,
            {
                "schema": VERSION + "-positive-branch",
                "schema_version": 1,
                "triggered": 0,
                "attempted": 0,
                "reason": "global material-improvement trigger was not met",
            },
        )

    reserve = _optional_postprocess_reserve(run_dir)
    if reserve is None:
        return _write_semantic(
            path,
            {
                "schema": VERSION + "-positive-branch",
                "schema_version": 1,
                "triggered": 1,
                "attempted": 0,
                "completed": 0,
                "reason": (
                    "optional postprocessing reserve was unavailable from completed "
                    "mandatory postprocessing authority"
                ),
                "mandatory_suffix_objective_preserved": 1,
                "complete_path_claim_authorized": 0,
            },
        )
    smoke = _read_json(run_dir / "controls/preflight_controls.json", semantic=True)
    ledger = _resource_ledger(run_dir)
    # A complete three-row reverse path has 64 shards.  Scale the measured
    # five-row shard conservatively by row count and retain 20% margin.
    projected_reverse_shard = (
        float(
            smoke.get(
                "durable_five_row_exact_shard_seconds",
                smoke["five_row_exact_shard_elapsed_seconds"],
            )
        )
        * (3.0 / 5.0)
        * 1.20
    )
    forward_root = run_dir / "fresh_forward/forward_shards/fresh-main-path"
    positive_root = run_dir / "positive/fused_families/same-path-three-row/complete-512"
    committed_forward = len(_optional_forward_observed_shard_seconds(forward_root))
    committed_reverse = _optional_committed_shard_count(positive_root)
    remaining_forward = max((OUTER_STEPS - 128) // 8 - committed_forward, 0)
    remaining_reverse = max(OUTER_STEPS // 8 - committed_reverse, 0)
    projected_reverse = projected_reverse_shard * remaining_reverse
    projected_forward_tail = (
        FORWARD_RESERVE_SECONDS / (MANDATORY_SUFFIX_STEPS // 8)
    ) * remaining_forward
    projected_total = (
        float(ledger.get("active_seconds", 0.0))
        + projected_forward_tail
        + projected_reverse
        + float(reserve["reserve_seconds"])
        + REPORT_RESERVE_SECONDS
    )
    projected_storage = _directory_bytes(run_dir) + int(
        reserve["reserve_storage_bytes"]
    )
    if projected_total > ACTIVE_SECONDS_CAP or projected_storage >= STORAGE_CAP_BYTES:
        return _write_semantic(
            path,
            {
                "schema": VERSION + "-positive-branch",
                "schema_version": 1,
                "triggered": 1,
                "attempted": 0,
                "reason": "same-path full exact branch did not fit the remaining frozen budget",
                "projected_forward_tail_seconds": projected_forward_tail,
                "projected_reverse_seconds": projected_reverse,
                "projected_optional_postprocess_seconds": reserve["reserve_seconds"],
                "projected_optional_postprocess_storage_bytes": reserve[
                    "reserve_storage_bytes"
                ],
                "projected_total_active_seconds": projected_total,
                "projected_total_storage_bytes": projected_storage,
                "active_seconds_cap": ACTIVE_SECONDS_CAP,
                "storage_cap_bytes": STORAGE_CAP_BYTES,
            },
        )

    forward_journal_relative = "positive/active-forward-tail-attempt.json"
    forward_journal, forward_started = _begin_durable_attempt(
        run_dir,
        journal_relative=forward_journal_relative,
        role="positive_branch_forward_128_to_511_attempt",
        detail={
            "start_step": 128,
            "step_limit": OUTER_STEPS,
            "durable_root_relative": forward_root.relative_to(run_dir).as_posix(),
            "durable_minimum_start_step": 128,
        },
    )
    forward_durable_before = _committed_shard_elapsed(
        forward_root, minimum_start_step=128
    )
    path_id = -1
    try:
        source = load_verified_source_target(args.source_run_dir)
        path_usage = _read_json(run_dir / "path_usage.json", semantic=True)
        path_id = int(path_usage["fresh_path_id"])
        profile = JacobiRBCudaProfile()
        device = torch.device(args.device)
        first_missing = MANDATORY_SUFFIX_STEPS // 8
        while first_missing < OUTER_STEPS // 8:
            state_path = forward_root / f"shard-{first_missing:04d}.npz"
            record_path = forward_root / f"shard-{first_missing:04d}.json"
            if not (state_path.is_file() and record_path.is_file()):
                break
            first_missing += 1

        forward = None
        if first_missing > MANDATORY_SUFFIX_STEPS // 8:
            # Strictly validate the complete durable prefix once before using
            # it to choose the next boundary.  This invocation executes no new
            # sampler work.
            forward = run_forward_trajectory(
                source.mixed_target,
                anchor_steps=(first_missing * 8 - 1,),
                output_dir=run_dir / "fresh_forward",
                trajectory_name="fresh-main-path",
                path_ids=(path_id,),
                root_seed=FORWARD_ROOT_SEED,
                profile=profile,
                step_limit=first_missing * 8,
                device=device,
            )
        for shard_index in range(first_missing, OUTER_STEPS // 8):
            _admit_optional_forward_shard(
                run_dir,
                forward_root=forward_root,
                attempt_started=forward_started,
                shard_index=shard_index,
                projected_reverse_seconds=projected_reverse,
                postprocess_reserve_seconds=float(reserve["reserve_seconds"]),
                postprocess_reserve_storage_bytes=int(
                    reserve["reserve_storage_bytes"]
                ),
            )
            limit = (shard_index + 1) * 8
            forward = run_forward_trajectory(
                source.mixed_target,
                anchor_steps=(limit - 1,),
                output_dir=run_dir / "fresh_forward",
                trajectory_name="fresh-main-path",
                path_ids=(path_id,),
                root_seed=FORWARD_ROOT_SEED,
                profile=profile,
                step_limit=limit,
                device=device,
            )
        if forward is None:
            raise GlobalDilatedRolloutError(
                "positive forward durable prefix did not reach the mandatory anchor"
            )
        forward_health = forward.diagnostics
        if (
            int(forward_health.get("passed", 0)) != 1
            or int(forward_health.get("restart_chain_valid", 0)) != 1
            or float(forward_health.get("authorization_fraction", 0.0)) != 1.0
            or float(forward_health.get("certificate_fraction", 0.0)) != 1.0
            or int(forward_health.get("forbidden_event_count", -1)) != 0
            or int(forward_health.get("output_state_nonfinite_count", -1)) != 0
            or int(forward_health.get("output_state_negative_count", -1)) != 0
            or float(forward_health.get("maximum_output_state_mass_error", math.inf))
            > 2.0e-12
        ):
            raise GlobalDilatedRolloutError("positive-branch full forward health failed")
        anchor = np.ascontiguousarray(forward.anchors[511], dtype=np.float64)
        atomic_rollout_npz(run_dir / "positive/anchor-step-0511.npz", {"state": anchor})
    except Exception as exc:
        durable_after = _committed_shard_elapsed(
            forward_root, minimum_start_step=128
        )
        wall = time.perf_counter() - forward_started
        try:
            _finish_durable_attempt(
                run_dir,
                journal_relative=forward_journal_relative,
                role="positive_branch_forward_128_to_511_attempt",
                journal=forward_journal,
                elapsed_seconds=max(wall, durable_after - forward_durable_before),
                failed=True,
                detail={
                    "same_path_id": path_id,
                    "invocation_wall_seconds": wall,
                    "durable_before_seconds": forward_durable_before,
                    "durable_committed_shard_seconds": durable_after,
                },
            )
        except ResourceBoundaryError:
            if not isinstance(exc, ResourceBoundaryError):
                raise exc
            raise
        raise
    forward_elapsed = time.perf_counter() - forward_started
    forward_durable_after = _committed_shard_elapsed(
        forward_root, minimum_start_step=128
    )
    _finish_durable_attempt(
        run_dir,
        journal_relative=forward_journal_relative,
        role="positive_branch_forward_128_to_511_attempt",
        journal=forward_journal,
        elapsed_seconds=max(
            forward_elapsed, forward_durable_after - forward_durable_before
        ),
        failed=False,
        detail={
            "same_path_id": path_id,
            "invocation_wall_seconds": forward_elapsed,
            "durable_before_seconds": forward_durable_before,
            "durable_committed_shard_seconds": forward_durable_after,
            "boundary_admission_count": max(
                OUTER_STEPS // 8 - first_missing, 0
            ),
        },
    )

    positive_root = run_dir / "positive/fused_families/same-path-three-row/complete-512"
    reverse_journal_relative = "positive/active-complete-reverse-attempt.json"
    reverse_journal, reverse_started = _begin_durable_attempt(
        run_dir,
        journal_relative=reverse_journal_relative,
        role="positive_branch_complete_three_row_exact_attempt",
        detail={
            "row_count": 3,
            "reverse_steps": OUTER_STEPS,
            "durable_root_relative": positive_root.relative_to(run_dir).as_posix(),
        },
    )
    reverse_durable_before = _committed_shard_elapsed(positive_root)
    try:
        specs, bank, binding = _positive_branch_rows(run_dir, args)
        prepared = _prepared_exact_backend(device, profile)
        stream_role = "global_dilated_positive_complete_exact"
        result = run_fused_reverse_family(
            np.repeat(anchor[None, :], 3, axis=0),
            sequence=tuple(reverse_suffix_sequence(511)),
            output_dir=run_dir / "positive",
            family_name="same-path-three-row",
            segment_name="complete-512",
            row_specs=specs,
            controller_bank=bank,
            reference_factory=_exact_reference_factory(
                prepared=prepared,
                profile=profile,
                root_seed=REVERSE_ROOT_SEED,
                stream_role=stream_role,
            ),
            controller_binding=binding,
            rng_binding={
                "root_seed": REVERSE_ROOT_SEED,
                "stream_role": stream_role,
                "canonical_path_id": int(path_usage["fresh_path_id"]),
            },
            label=3,
            microsteps=MICROSTEPS,
            device=device,
            capture_coordinates={
                (384, 0): "completed-128",
                (256, 0): "completed-256",
                (128, 0): "completed-384",
                (0, 0): "completed-512",
            },
            before_uncommitted_shard=_suffix_admission_callback(
                run_dir,
                positive_root,
                attempt_started=reverse_started,
                additional_active_reserve_seconds=float(
                    reserve["reserve_seconds"]
                ),
                additional_storage_reserve_bytes=int(
                    reserve["reserve_storage_bytes"]
                ),
            ),
            reference_contract="certified_exact",
        )
        if (
            len(result.shard_records) != OUTER_STEPS // 8
            or tuple(spec.row_key for spec in result.row_specs)
            != ("zero", "global-plus-1", "source-informed")
        ):
            raise GlobalDilatedRolloutError("positive-branch exact family is incomplete")
        strict_health = _strict_fused_exact_health(
            final_state=result.final_state,
            shard_records=result.shard_records,
            row_count=3,
        )
    except Exception as exc:
        durable_after = _committed_shard_elapsed(positive_root)
        wall = time.perf_counter() - reverse_started
        finish_error: Exception | None = None
        try:
            _finish_durable_attempt(
                run_dir,
                journal_relative=reverse_journal_relative,
                role="positive_branch_complete_three_row_exact_attempt",
                journal=reverse_journal,
                elapsed_seconds=max(wall, durable_after - reverse_durable_before),
                failed=True,
                detail={
                    "resource_boundary_stop": int(
                        isinstance(exc, ResourceBoundaryError)
                    ),
                    "invocation_wall_seconds": wall,
                    "durable_before_seconds": reverse_durable_before,
                    "durable_committed_shard_seconds": durable_after,
                },
            )
        except Exception as accounting_exc:
            finish_error = accounting_exc
        effective = (
            finish_error
            if finish_error is not None
            and not isinstance(finish_error, ResourceBoundaryError)
            else exc
        )
        if isinstance(effective, ResourceBoundaryError):
            return _write_semantic(
                path,
                {
                    "schema": VERSION + "-positive-branch",
                    "schema_version": 1,
                    "triggered": 1,
                    "attempted": 1,
                    "completed": 0,
                    "reason": "positive complete path stopped at an exact shard resource boundary",
                    "failure_message": str(effective),
                    "partial_exact_evidence_preserved": int(durable_after > 0.0),
                },
            )
        raise effective
    reverse_elapsed = time.perf_counter() - reverse_started
    reverse_durable_after = _committed_shard_elapsed(positive_root)
    _finish_durable_attempt(
        run_dir,
        journal_relative=reverse_journal_relative,
        role="positive_branch_complete_three_row_exact_attempt",
        journal=reverse_journal,
        elapsed_seconds=max(
            reverse_elapsed, reverse_durable_after - reverse_durable_before
        ),
        failed=False,
        detail={
            "shard_count": len(result.shard_records),
            "invocation_wall_seconds": reverse_elapsed,
            "durable_before_seconds": reverse_durable_before,
            "durable_committed_shard_seconds": reverse_durable_after,
        },
    )
    return _run_optional_positive_postprocessing(
        run_dir,
        source=source,
        anchor=anchor,
        forward=forward,
        result=result,
        strict_health=strict_health,
        reserve=reserve,
    )


def _maybe_run_positive_complete_path(
    run_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """Keep every optional failure subordinate to the completed mandatory suffix."""

    started = time.perf_counter()
    active_before = float(_resource_ledger(run_dir).get("active_seconds", 0.0))
    forward_root = run_dir / "fresh_forward/forward_shards/fresh-main-path"
    positive_root = run_dir / "positive/fused_families/same-path-three-row/complete-512"
    durable_before = _committed_shard_elapsed(
        forward_root, minimum_start_step=128
    ) + _committed_shard_elapsed(positive_root)
    try:
        result = _run_positive_complete_path_impl(run_dir, args)
        ledger_after_inner = _resource_ledger(run_dir)
        active_after_inner = float(ledger_after_inner.get("active_seconds", 0.0))
        if ledger_after_inner.get("limits_passed") != 1:
            # A reconciled optional postprocess attempt can itself be the
            # durable cap boundary.  Do not append a second outer event after
            # that authority; the evaluate marker/finalizer will package the
            # completed mandatory objective and the optional deferral.
            return result
        outer_wall = time.perf_counter() - started
        orchestration_wall = max(
            outer_wall - max(active_after_inner - active_before, 0.0), 0.0
        )
        _record_resource_event(
            run_dir,
            role="positive_branch_orchestration_and_postprocess_attempt",
            elapsed_seconds=orchestration_wall,
            detail={
                "attempt": _resource_attempt_index(
                    run_dir, "positive_branch_orchestration_and_postprocess_attempt"
                ),
                "inner_accounted_seconds": max(
                    active_after_inner - active_before, 0.0
                ),
                "outer_invocation_wall_seconds": outer_wall,
                "positive_branch_semantic_sha256": result["semantic_sha256"],
            },
        )
        return result
    except Exception as exc:
        # The mandatory exact family, aggregate, metrics, images, and outcome
        # already completed before this optional call.  A failure in extra
        # same-path reconstruction must not retroactively erase that result.
        durable_after = _committed_shard_elapsed(
            forward_root, minimum_start_step=128
        ) + _committed_shard_elapsed(positive_root)
        try:
            active_after_inner = float(
                _resource_ledger(run_dir).get("active_seconds", 0.0)
            )
            current_ledger = _resource_ledger(run_dir)
            inner_accounted = max(active_after_inner - active_before, 0.0)
            outer_wall = time.perf_counter() - started
            covered_durable = math.fsum(
                max(
                    (
                        float(
                            event.get("detail", {}).get(
                                "durable_committed_shard_seconds", 0.0
                            )
                        )
                        for event in current_ledger.get("events", [])
                        if event.get("role") == role
                    ),
                    default=0.0,
                )
                for role in (
                    "positive_branch_forward_128_to_511_attempt",
                    "positive_branch_complete_three_row_exact_attempt",
                )
            )
            unreconciled = max(
                outer_wall - inner_accounted,
                durable_after - covered_durable,
                0.0,
            )
            _record_resource_event(
                run_dir,
                role="positive_branch_failed_attempt",
                elapsed_seconds=unreconciled,
                detail={
                    "attempt": _resource_attempt_index(
                        run_dir, "positive_branch_failed_attempt"
                    ),
                    "failed": 1,
                    "failure_type": type(exc).__name__,
                    "outer_invocation_wall_seconds": outer_wall,
                    "inner_accounted_seconds": inner_accounted,
                    "durable_before_seconds": durable_before,
                    "durable_committed_shard_seconds": durable_after,
                    "durable_seconds_covered_by_inner_events": covered_durable,
                    "unreconciled_seconds": unreconciled,
                },
            )
        except Exception:
            pass
        partial = sorted(positive_root.glob("shard-*.npz")) if positive_root.is_dir() else []
        last_valid_saved = 0
        if partial:
            try:
                latest = _load_npz(partial[-1]).get("state")
                if latest is not None and latest.shape == (3, STATE_SIZE):
                    atomic_rollout_npz(
                        run_dir / "positive/failure/last_valid_states.npz",
                        {"state": np.ascontiguousarray(latest, dtype=np.float64)},
                    )
                    last_valid_saved = 1
            except Exception:
                pass
        message = str(exc)
        resource = isinstance(exc, ResourceBoundaryError)
        return _write_semantic(
            run_dir / "positive_branch.json",
            {
                "schema": VERSION + "-positive-branch",
                "schema_version": 1,
                "triggered": 1,
                "attempted": 1,
                "completed": 0,
                "mandatory_suffix_objective_preserved": 1,
                "failure_domain": (
                    "resource_budget" if resource else "optional_execution_integrity"
                ),
                "failure_type": type(exc).__name__,
                "failure_message": message,
                "reason": "optional complete-path attempt failed after mandatory objective completion",
                "committed_partial_shard_count": len(partial),
                "last_valid_optional_states_saved": last_valid_saved,
                "partial_exact_evidence_preserved": int(bool(partial)),
                "required_next_action": (
                    "resume the optional exact branch only if its unchanged resource boundary admits it"
                    if resource
                    else "diagnose the optional full-path interface/numerical failure without changing the mandatory suffix conclusion"
                ),
                "complete_path_claim_authorized": 0,
            },
        )


def _write_reports(run_dir: Path, args: argparse.Namespace) -> None:
    summary = _read_json(run_dir / "suffix/summary.json", semantic=True)
    selection = _read_json(run_dir / "selection.json", semantic=True)
    outcome = _read_json(run_dir / "outcome.json", semantic=True)
    ledger = _resource_ledger(run_dir)
    abandoned_events = [
        event
        for event in ledger.get("events", [])
        if event.get("detail", {}).get("abandoned_hard_crash") == 1
    ]
    unknown_active_upper_bound = math.fsum(
        float(event.get("detail", {}).get("unknown_active_interval_seconds", 0.0))
        for event in abandoned_events
    )
    unknown_active_cap_charge = math.fsum(
        float(event.get("elapsed_seconds", 0.0)) for event in abandoned_events
    )
    completed_resource_stop_path = run_dir / "completed_objective_resource_stop.json"
    completed_resource_stop = (
        _read_json(completed_resource_stop_path, semantic=True)
        if completed_resource_stop_path.is_file()
        else None
    )
    resource_stop_block = (
        "\n\nResource-stop status: the mandatory 16-shard scientific objective completed, "
        "but terminal resource accounting (including exact final storage) crossed a configured cap. "
        "The bundle is sealed truthfully in `completed_objective_resource_stop.json`; "
        "this is a reporting/resource failure, not an incomplete scientific suffix."
        if completed_resource_stop is not None
        else ""
    )
    path_usage = _read_json(run_dir / "path_usage.json", semantic=True)
    objectives = summary["primary_final_objectives"]
    global_metric = objectives["global-plus-1"]
    sign_metric = objectives["v4-minus-0p5"]
    source_metric = objectives["source-informed"]
    suffix_family = _read_json(run_dir / "suffix/family_summary.json", semantic=True)
    exact_health = suffix_family["strict_exact_health"]
    positive = _read_json(run_dir / "positive_branch.json", semantic=True)
    objective_table = "\n".join(
        "| {row} | {risk:.9g} | {delta:.9g} | {relative:.6g}% |".format(
            row=row,
            risk=float(objectives[row]["squared_l2_error"]),
            delta=float(objectives[row]["paired_squared_l2_improvement_over_zero"]),
            relative=100.0
            * float(objectives[row]["relative_paired_squared_l2_improvement_over_zero"]),
        )
        for row in ROW_ORDER
    )
    positive_block = (
        "Optional same-path complete branch: triggered={triggered}, attempted={attempted}, "
        "completed={completed}.".format(
            triggered=int(positive.get("triggered", 0)),
            attempted=int(positive.get("attempted", 0)),
            completed=int(positive.get("completed", 0)),
        )
    )
    if int(positive.get("completed", 0)) == 1:
        full = positive["final_metrics"]
        positive_block += (
            " Complete-path global squared-L2={risk:.9g}, delta={delta:.9g}; "
            "strict exact certificate fraction={certificate:.1f}. Raw trajectory/images: "
            "`positive/trajectory_shard_boundaries.npz`, `positive/images/`."
        ).format(
            risk=float(full["global-plus-1"]["squared_l2_error"]),
            delta=float(
                full["global-plus-1"]["paired_squared_l2_improvement_over_zero"]
            ),
            certificate=float(positive["strict_exact_health"]["certificate_fraction"]),
        )
    else:
        positive_block += (
            f" Reason: {positive.get('reason', 'not completed')}. No complete-path claim is made; "
            "any exact partial prefix remains under `positive/`."
        )
    result_sentence = (
        f"The global row's paired squared-L2 improvement was "
        f"{global_metric['paired_squared_l2_improvement_over_zero']:.9g} "
        f"({100.0 * global_metric['relative_paired_squared_l2_improvement_over_zero']:.4g}%), "
        f"and the frozen v4 -0.5 diagnostic was "
        f"{sign_metric['paired_squared_l2_improvement_over_zero']:.9g} "
        f"({100.0 * sign_metric['relative_paired_squared_l2_improvement_over_zero']:.4g}%)."
    )
    report = f"""# Global-dilated exact fresh suffix report

Research mode: exploratory

Decision: does a genuinely global Jacobi/RB score model improve one fresh exact paired 128-step suffix over zero, and does the predeclared negative frozen-v4 diagnostic reveal a sign/order defect?

## Outcome

{result_sentence}

Terminal classification: `{outcome['outcome']}`. Required next action: `{outcome['required_next_action']}`.

The source-informed exact control's paired squared-L2 improvement was {source_metric['paired_squared_l2_improvement_over_zero']:.9g}. All five rows, all 17 raw shard-boundary states, five milestones, mechanism telemetry, and raw/demixed milestone images were retained.

| Row | E_c squared L2 | Delta = E_zero - E_c | Relative Delta |
|---|---:|---:|---:|
{objective_table}

## Training and selection

The model was trained through its wrapper: `m_theta = Y(1-Y) q_theta` was compared directly with stored `bar_Z`, normalized by squared training-target RMS. No `bar_Z/mobility` quotient was formed. Selected update: {selection['selected']['update']}; selected raw/normalized validation MSE: {selection['selected']['raw_validation_mse_m_vs_bar_Z']:.9g} / {selection['selected']['normalized_validation_mse']:.9g}; frozen-v4 raw/normalized comparison: {selection['frozen_v4_comparison']['raw_validation_mse_m_vs_bar_Z']:.9g} / {selection['frozen_v4_comparison']['normalized_validation_mse']:.9g}.

## Exact evaluation

Fresh path ID: {path_usage['fresh_path_id']}; forward seed: {FORWARD_ROOT_SEED}; reverse seed: {REVERSE_ROOT_SEED}; exact backend; seven phases; M={MICROSTEPS}; shared canonical transition IDs. One path/image is the independent unit. No p-value, interval, or population claim is made.

Mandatory exact health: {exact_health['active_count']} active / {exact_health['certified_count']} certified transitions, {exact_health['fallback_count']} certified fallbacks, maximum mass error {exact_health['maximum_mass_error']:.3g}, forbidden events {exact_health['forbidden_event_count']}.

{positive_block}

## Claim boundary

This run establishes only the recorded one-path exact paired effects under the frozen target, checkpoint rule, gains, schedule, and backend. It does not establish multi-image generation, prior-start sampling, population performance, or impossibility of Eulerian score generation.

## Resources

Recorded active/cap-debited seconds: {ledger['active_seconds']:.3f} / {ACTIVE_SECONDS_CAP:.0f}. Resource limits passed: {ledger.get('limits_passed', 0)}; breaches: {ledger.get('breaches', [])}. Abandoned hard-crash intervals: {len(abandoned_events)}, charging {unknown_active_cap_charge:.3f}s conservatively; {unknown_active_upper_bound:.3f}s is an unknown-active wall-to-resume upper bound that may include idle or powered-off time, not measured accelerator compute. Peak/total CUDA bytes: {ledger.get('maximum_peak_cuda_memory_bytes', 0)} / {ledger.get('maximum_total_cuda_memory_bytes', 0)}. Persisted storage: {ledger.get('persisted_storage_bytes', _directory_bytes(run_dir))} / {STORAGE_CAP_BYTES} bytes.{resource_stop_block}
"""
    _atomic_text(run_dir / "REPORT.md", report)

    handoff = f"""# Global-dilated Jacobi/RB exact suffix: research handoff

Date: {_utc_now()}
Source revision: {_source_revision(Path(args.repository_root).resolve())}
Handoff author: Codex

## 1. Program objective

Final scientific/engineering objective: a DDPM-like MNIST image generator based on the Eulerian approximation.

Concrete success artifact: generated or reconstructed MNIST images from complete reverse trajectories.

## 2. Current milestone and distance to goal

Nearest objective-bearing milestone: one fresh exact paired 128-step reverse suffix.

Current principal blocker: `{outcome['outcome']}`.

Last objective-bearing experiment and date: this run, {_utc_now()}.

Artifact produced: `suffix/trajectory_shard_boundaries.npz` and milestone images.

Proxy-only patches since then: 0

What remains untested end to end: multi-image behavior and reference-prior starts; same-path full reverse is recorded only if admitted in `positive_branch.json`.

## 3. Strategy review

Strategy status: {('continue' if outcome['outcome'] == 'global_material_improvement' else 'major modification' if outcome['outcome'] in {'sign_order_leading','validation_better_suffix_adverse'} else 'pivot')}

Rationale: {result_sentence}

Strongest alternative strategy: a small standard MNIST DDPM sanity baseline followed by a direct backward-heat-potential edge-score formulation.

Evidence that would change this decision: an unchanged fresh exact replication or a theory-located sign convention defect, as applicable.

## 4. Research mode and evidence roles

Primary mode: exploratory. Training and validation reuse the opened F8100/F8200 stores. Confirmation F9000-F903F remained unopened. Fresh evidence used path {path_usage['fresh_path_id']}.

## 5. Exact result of the latest run

One fresh path, five paired rows, certified exact M=2 backend, 128 reverse outer steps. {result_sentence}

{positive_block}

### This result establishes

The exact one-path paired effects in `suffix/summary.json` under the frozen design.

### This result does not establish

Population utility, multi-image generation, prior matching, or impossibility of another Eulerian/controller/training strategy.

## 6. Confirmed facts, current inferences, and open hypotheses

### Confirmed facts

All five exact rows completed; the source interface and certificate/health controls passed; selected global validation MSE was {selection['selected']['normalized_validation_mse']:.9g}.

### Current inferences

The outcome classification is `{outcome['outcome']}` and is an exploratory one-path inference.

### Open hypotheses

Implementation/sign convention, recursive on-policy shift, target derivation, controller authority, prior mismatch, and strategy failure remain distinguishable at broader scope.

## 7. Decision the next patch must resolve

Carry out: `{outcome['required_next_action']}`.

## 8. Candidate actions and value of information

The required action above has highest decision value. Read-only radius analysis is descriptive and cannot replace it. New local representation tweaks are excluded by the stopping rule.

## 9. Recommended next patch

Why it has the highest decision value: it follows the predeclared outcome branch. It must save task-level trajectories/images with zero and known-positive controls and make no population claim.

## 10. Gates and claim boundaries

Execution/integrity gates protected parent roles, target/sign semantics, shared RNG, exact certificates, numerical health, and resource admission. The 1% practical-effect label was diagnostic only. Failure of a claim gate never blocked exploratory suffix execution.

## 11. Outcome-to-action table

| Outcome | Interpretation | Required next action |
|---|---|---|
| Global material | One-path global-context feasibility | Same-path complete path, then multi-image/prior start |
| Negative v4 material only | Sign/order convention leads | Locate theory/code convention defect before changing sign |
| Validation better, suffix adverse | Cached risk does not establish recursive utility | Rollout-aligned training or target derivation |
| All learned adverse, controls pass | Nearby Jacobi learner repair lacks feasibility | Standard DDPM sanity plus direct heat-potential pivot |

## 12. Constraints

Integrity constraints: do not open confirmation, mutate immutable parents, change sealed evaluation choices, substitute approximate primary evidence, or suppress adverse outputs.

Revisable scientific and engineering choices: architecture, loss, controller, schedule, inference family, and Jacobi/RB strategy after this scoped result.

## 13. Resource budget and stop rule

Recorded active/cap-debited seconds {ledger['active_seconds']:.3f}; cap {ACTIVE_SECONDS_CAP:.0f}; limits passed {ledger.get('limits_passed', 0)}; breaches {ledger.get('breaches', [])}; storage cap 2 GiB. Abandoned hard-crash accounting contributes {unknown_active_cap_charge:.3f}s, including an unknown-active wall-to-resume upper bound of {unknown_active_upper_bound:.3f}s that may contain idle/off time and is not labeled measured compute. No further nearby local/global Jacobi feature or gain repair after an adverse all-learned branch.{resource_stop_block}

## 14. Alternative and pivot plan

Run the stated small DDPM compute/task sanity baseline, then formulate direct heat-potential edge scores faithful to the backward heat equation.

## 15. Evidence map

Raw: `fresh_forward/`, `suffix/fused_families/`, and `suffix/trajectory_shard_boundaries.npz`. Derived exploratory: `suffix/metrics.csv`, `suffix/mechanism.json`, `images/`, and `outcome.json`. Protected confirmation evidence was not opened.

## 16. Deliberate omissions

No confirmation labels, approximate primary suffix, 129-state hot-loop trace, population inference, or self-contained bundle verifier.

## 17. Reproduction commands

See `exact_command.txt`; resume with `--resume-run-dir {run_dir} --stage all` from the documented repository/dependencies.

## 18. Bundle-integrity audit

Verification command: rerun `--stage report_verify --resume-run-dir {run_dir}`. Manifest/checksums are at top level; representative NPZ/PNG recomputation is recorded in `verification.json`.

## 19. Exact deliverable for the receiving agent

Implement the predeclared required next action `{outcome['required_next_action']}` as executable objective-bearing code/experiment, not another authorization-only plan.
"""
    _atomic_text(run_dir / "HANDOFF.md", handoff)


_FINAL_EXCLUDED = frozenset(
    {
        "artifact_manifest.json",
        "SHA256SUMS.txt",
        "verification.json",
        # This record is bound by verification and the terminal stage marker.
        # Excluding it from the inventory removes a hash cycle while its exact
        # physical bytes remain included in terminal storage accounting.
        "terminal_storage_authority.json",
        # Written last as the terminal commit marker; excluding it avoids a
        # manifest/marker hash cycle while making interrupted finalization
        # distinguishable from completion.
        "stages/report_verify.json",
    }
)


def _manifest_rows(run_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "size": int(path.stat().st_size),
            "sha256": file_fingerprint(path),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and path.relative_to(run_dir).as_posix() not in _FINAL_EXCLUDED
        and path.relative_to(run_dir).as_posix() != _RAW_SUFFIX_CONVERSION_INTENT
        and path.suffix != ".tmp"
    ]


def _verify_scientific_evidence_read_only(
    run_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """Recompute the complete load-bearing raw/derived scientific authority."""

    freeze_path = run_dir / "evaluation_freeze.json"
    freeze = _read_json(freeze_path, semantic=True)
    selection = _read_json(run_dir / "selection.json", semantic=True)
    path_usage = _read_json(run_dir / "path_usage.json", semantic=True)
    if (
        file_fingerprint(run_dir / "selection.json") != freeze.get("selection_file_sha256")
        or selection.get("selected") != freeze.get("global_checkpoint")
        or path_usage.get("evaluation_freeze_file_sha256") != file_fingerprint(freeze_path)
        or path_usage.get("fresh_path_id") not in FRESH_PATH_POOL
    ):
        raise GlobalDilatedRolloutError("selection/freeze/path authority changed")
    source = load_verified_source_target(args.source_run_dir)
    if rollout_array_sha256(source.mixed_target) != freeze.get("source_target_sha256"):
        raise GlobalDilatedRolloutError("final source target differs from evaluation freeze")

    forward = _read_json(run_dir / "fresh_forward/forward_summary.json", semantic=True)
    anchor_path = run_dir / "fresh_forward/anchor-step-0127.npz"
    anchor = _load_npz(anchor_path).get("state")
    diagnostics = forward.get("diagnostics")
    if (
        anchor is None
        or anchor.dtype != np.float64
        or anchor.shape != (STATE_SIZE,)
        or not np.isfinite(anchor).all()
        or np.any(anchor < 0.0)
        or abs(float(np.sum(anchor)) - 1.0) > 2.0e-12
        or forward.get("evaluation_freeze_file_sha256") != file_fingerprint(freeze_path)
        or forward.get("path_usage_file_sha256")
        != file_fingerprint(run_dir / "path_usage.json")
        or forward.get("path_id") != path_usage.get("fresh_path_id")
        or forward.get("source_target_sha256") != freeze.get("source_target_sha256")
        or forward.get("anchor_state_sha256") != rollout_array_sha256(anchor)
        or forward.get("anchor_archive_sha256") != rollout_file_sha256(anchor_path)
        or not isinstance(diagnostics, Mapping)
        or diagnostics.get("passed") != 1
        or diagnostics.get("restart_chain_valid") != 1
        or diagnostics.get("authorization_fraction") != 1.0
        or diagnostics.get("certificate_fraction") != 1.0
        or diagnostics.get("forbidden_event_count") != 0
        or diagnostics.get("output_state_nonfinite_count") != 0
        or diagnostics.get("output_state_negative_count") != 0
        or float(diagnostics.get("maximum_output_state_mass_error", math.inf)) > 2.0e-12
    ):
        raise GlobalDilatedRolloutError("fresh forward anchor/health authority changed")

    aggregate = _load_npz(run_dir / "suffix/trajectory_shard_boundaries.npz")
    states = aggregate.get("states")
    completed = aggregate.get("completed_reverse_steps")
    if (
        states is None
        or completed is None
        or states.dtype != np.float64
        or states.shape != (len(ROW_ORDER), 17, STATE_SIZE)
        or not np.array_equal(completed, np.arange(0, 129, 8, dtype=np.int64))
        or not np.array_equal(states[:, 0], np.repeat(anchor[None, :], len(ROW_ORDER), axis=0))
    ):
        raise GlobalDilatedRolloutError("raw aggregate trajectory authority changed")
    shard_root = run_dir / "suffix/fused_families/fresh-five-row/suffix-128"
    shard_records: list[dict[str, Any]] = []
    previous_hash = rollout_array_sha256(states[:, 0])
    for index in range(16):
        record_path = shard_root / f"shard-{index:04d}.json"
        archive_path = shard_root / f"shard-{index:04d}.npz"
        record = _read_json(record_path, semantic=True)
        arrays = _load_npz(archive_path)
        endpoint = arrays.get("state")
        if (
            set(arrays) != {"state"}
            or endpoint is None
            or endpoint.dtype != np.float64
            or endpoint.shape != (len(ROW_ORDER), STATE_SIZE)
            or not np.isfinite(endpoint).all()
            or np.any(endpoint < 0.0)
            or float(np.max(np.abs(np.sum(endpoint, axis=1) - 1.0))) > 2.0e-12
            or record.get("committed") != 1
            or record.get("input_state_sha256") != previous_hash
            or record.get("state_file_sha256") != rollout_file_sha256(archive_path)
            or record.get("output_state_sha256") != rollout_array_sha256(endpoint)
            or not np.array_equal(endpoint, states[:, index + 1])
        ):
            raise GlobalDilatedRolloutError("raw exact suffix shard authority changed")
        previous_hash = rollout_array_sha256(endpoint)
        shard_records.append(record)
    strict = _strict_fused_exact_health(
        final_state=states[:, -1], shard_records=shard_records, row_count=len(ROW_ORDER)
    )
    family = _read_json(run_dir / "suffix/family_summary.json", semantic=True)
    if (
        family.get("strict_exact_health") != strict
        or family.get("row_order") != list(ROW_ORDER)
        or family.get("reference_contract") != "certified_exact"
        or family.get("completed") != 1
    ):
        raise GlobalDilatedRolloutError("exact suffix family summary authority changed")

    expected_metric_rows: list[dict[str, Any]] = []
    for boundary_index, reverse_steps in enumerate(completed.tolist()):
        zero_state = states[0, boundary_index]
        for row_index, row_key in enumerate(ROW_ORDER):
            state = states[row_index, boundary_index]
            mixed = raw_state_metrics(state, source.mixed_target).to_dict()
            unmixed = raw_state_metrics(state, source.source_image).to_dict()
            separation = state - zero_state
            expected_metric_rows.append(
                {
                    "row_index": row_index,
                    "row_key": row_key,
                    "boundary_index": boundary_index,
                    "completed_reverse_steps": int(reverse_steps),
                    **{f"mixed_target_{key}": value for key, value in mixed.items()},
                    **{f"unmixed_source_{key}": value for key, value in unmixed.items()},
                    "row_vs_zero_squared_separation": float(np.dot(separation, separation)),
                    "nonfinite_count": int(np.count_nonzero(~np.isfinite(state))),
                    "negative_count": int(np.count_nonzero(state < 0.0)),
                    "is_milestone": int(int(reverse_steps) in MILESTONE_STEPS),
                }
            )
    try:
        with (run_dir / "suffix/metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            metric_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise GlobalDilatedRolloutError("metrics CSV cannot reopen") from exc
    if len(metric_rows) != len(expected_metric_rows):
        raise GlobalDilatedRolloutError("metrics CSV row count changed")
    for actual, expected in zip(metric_rows, expected_metric_rows, strict=True):
        if set(actual) != set(expected):
            raise GlobalDilatedRolloutError("metrics CSV schema changed")
        for key, value in expected.items():
            observed = actual[key]
            if isinstance(value, str):
                equal = observed == value
            elif isinstance(value, int):
                equal = int(observed) == value
            else:
                equal = float(observed) == float(value)
            if not equal:
                raise GlobalDilatedRolloutError("metrics CSV value changed")

    calibration = _load_npz(run_dir / "training/on_policy_validation_calibration.npz")
    means = calibration["training_means"]
    p95 = calibration["training_p95"]
    ratios = calibration["validation_sorted_ratios"]
    counts = calibration["validation_counts"]
    drift_rows: list[dict[str, Any]] = []
    global_index = ROW_ORDER.index("global-plus-1")
    for boundary_index, reverse_steps in enumerate(completed.tolist()):
        current_outer_step = max(ANCHOR_STEP - int(reverse_steps), 0)
        quartile = min(current_outer_step // 128, 3)
        radius = float(np.linalg.norm(states[global_index, boundary_index] - means[quartile]))
        ratio = radius / float(p95[quartile])
        valid = np.asarray(ratios[quartile, : int(counts[quartile])])
        separation = states[global_index, boundary_index] - states[0, boundary_index]
        drift_rows.append(
            {
                "boundary_index": boundary_index,
                "completed_reverse_steps": int(reverse_steps),
                "matching_training_quartile": int(quartile),
                "global_zero_squared_separation": float(np.dot(separation, separation)),
                "global_radius_from_training_quartile_mean": radius,
                "global_radius_normalized_by_training_p95": ratio,
                "validation_calibrated_percentile": float(
                    np.searchsorted(valid, ratio, side="right") / valid.size
                ),
                "threshold_or_execution_gate": 0,
            }
        )
    mechanism = _read_json(run_dir / "suffix/mechanism.json", semantic=True)
    if (
        mechanism.get("telemetry_source") != "fused bank per-row diagnostics"
        or mechanism.get("per_reverse_quarter") != _row_quarter_mechanism(shard_records)
        or mechanism.get("on_policy_drift") != drift_rows
        or mechanism.get("drift_is_descriptive_only") != 1
        or mechanism.get("off_policy_failure_inferred") != 0
    ):
        raise GlobalDilatedRolloutError("mechanism telemetry summary changed")
    summary = _read_json(run_dir / "suffix/summary.json", semantic=True)
    if (
        summary.get("metric_file_sha256") != file_fingerprint(run_dir / "suffix/metrics.csv")
        or summary.get("mechanism_file_sha256")
        != file_fingerprint(run_dir / "suffix/mechanism.json")
        or summary.get("metric_row_count") != 85
        or summary.get("row_order") != list(ROW_ORDER)
    ):
        raise GlobalDilatedRolloutError("objective summary derived-file binding changed")
    return {
        "raw_suffix_shards_reopened": 16,
        "strict_exact_health": strict,
        "metric_rows_recomputed": len(expected_metric_rows),
        "mechanism_records_recomputed": len(drift_rows),
        "forward_health_recomputed": 1,
        "selection_freeze_path_rebound": 1,
    }


def _verify_all_generated_images_read_only(
    run_dir: Path,
    *,
    states: np.ndarray,
    summary: Mapping[str, Any],
    scale: FixedRenderingScale,
) -> None:
    """Decode and reproduce every promised individual and sheet PNG."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise GlobalDilatedRolloutError("final image decoder is unavailable") from exc

    def decode(relative: str) -> np.ndarray:
        try:
            with Image.open(run_dir / relative) as opened:
                return np.asarray(opened.convert("L"))
        except OSError as exc:
            raise GlobalDilatedRolloutError(f"generated image cannot decode: {relative}") from exc

    def sheet(cells: Sequence[np.ndarray], columns: int) -> np.ndarray:
        value = np.zeros(
            (math.ceil(len(cells) / columns) * 28, columns * 28), dtype=np.uint8
        )
        for index, cell in enumerate(cells):
            row, column = divmod(index, columns)
            value[row * 28 : (row + 1) * 28, column * 28 : (column + 1) * 28] = cell
        return value

    expected_paths: set[str] = set()
    expected_records: list[dict[str, Any]] = []
    raw_cells: list[np.ndarray] = []
    demixed_cells: list[np.ndarray] = []
    for milestone in MILESTONE_STEPS:
        boundary_index = milestone // 8
        milestone_raw: list[np.ndarray] = []
        milestone_demixed: list[np.ndarray] = []
        for row_index, row_key in enumerate(ROW_ORDER):
            state = states[row_index, boundary_index]
            raw = render_raw_density(state, scale)
            demixed = render_background_demixed(state, scale)
            raw_relative = f"images/raw/{row_key}/step-{milestone:03d}.png"
            demixed_relative = f"images/demixed/{row_key}/step-{milestone:03d}.png"
            if not np.array_equal(decode(raw_relative), raw) or not np.array_equal(
                decode(demixed_relative), demixed
            ):
                raise GlobalDilatedRolloutError(
                    "a generated milestone image does not reproduce from frozen scale"
                )
            expected_paths.update((raw_relative, demixed_relative))
            expected_records.append(
                {
                    "row_key": row_key,
                    "completed_reverse_steps": milestone,
                    "state_sha256": rollout_array_sha256(state),
                    "raw_path": raw_relative,
                    "raw_sha256": file_fingerprint(run_dir / raw_relative),
                    "demixed_path": demixed_relative,
                    "demixed_sha256": file_fingerprint(run_dir / demixed_relative),
                }
            )
            raw_cells.append(raw)
            demixed_cells.append(demixed)
            milestone_raw.append(raw)
            milestone_demixed.append(demixed)
        for kind, cells in (("raw", milestone_raw), ("demixed", milestone_demixed)):
            relative = f"images/contact-sheets/milestone-{milestone:03d}-{kind}.png"
            expected_paths.add(relative)
            if not np.array_equal(decode(relative), sheet(cells, 5)):
                raise GlobalDilatedRolloutError("a milestone contact sheet changed")
    for relative, cells in (
        ("images/contact-sheets/all-milestones-raw.png", raw_cells),
        ("images/contact-sheets/all-milestones-demixed.png", demixed_cells),
        ("images/contact-sheets/final-raw.png", raw_cells[-5:]),
        ("images/contact-sheets/final-demixed.png", demixed_cells[-5:]),
    ):
        expected_paths.add(relative)
        if not np.array_equal(decode(relative), sheet(cells, 5)):
            raise GlobalDilatedRolloutError("an aggregate contact sheet changed")
    actual_paths = {
        path.relative_to(run_dir).as_posix()
        for path in (run_dir / "images").rglob("*.png")
    }
    if actual_paths != expected_paths or summary.get("images") != expected_records:
        raise GlobalDilatedRolloutError("generated image path/record set changed")

    positive_path = run_dir / "positive_branch.json"
    if positive_path.is_file():
        positive = _read_json(positive_path, semantic=True)
        if positive.get("completed") == 1:
            positive_states = _load_npz(run_dir / "positive/milestones.npz").get("states")
            if positive_states is None or positive_states.shape != (3, 5, STATE_SIZE):
                raise GlobalDilatedRolloutError("positive milestone image source changed")
            positive_paths: set[str] = set()
            positive_cells: list[np.ndarray] = []
            for milestone_index, milestone in enumerate((0, 128, 256, 384, 512)):
                for row_index, row_key in enumerate(
                    ("zero", "global-plus-1", "source-informed")
                ):
                    relative = f"positive/images/raw/{row_key}/step-{milestone:03d}.png"
                    expected = render_raw_density(
                        positive_states[row_index, milestone_index], scale
                    )
                    positive_paths.add(relative)
                    positive_cells.append(expected)
                    if not np.array_equal(decode(relative), expected):
                        raise GlobalDilatedRolloutError("a positive-branch image changed")
            contact = "positive/images/contact-sheet-raw.png"
            positive_paths.add(contact)
            if not np.array_equal(decode(contact), sheet(positive_cells, 3)):
                raise GlobalDilatedRolloutError("positive-branch contact sheet changed")
            actual_positive = {
                path.relative_to(run_dir).as_posix()
                for path in (run_dir / "positive/images").rglob("*.png")
            }
            if actual_positive != positive_paths:
                raise GlobalDilatedRolloutError("positive image path set changed")


def _verify_raw_and_derived(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    deep = _verify_scientific_evidence_read_only(run_dir, args)
    aggregate = _load_npz(run_dir / "suffix/trajectory_shard_boundaries.npz")
    states = aggregate["states"]
    completed = aggregate["completed_reverse_steps"]
    aggregation = _read_json(run_dir / "suffix/trajectory_aggregation.json", semantic=True)
    if rollout_array_sha256(states) != aggregation["states_sha256"]:
        raise GlobalDilatedRolloutError("final trajectory aggregate hash failed")
    if not np.array_equal(completed, np.arange(0, 129, 8, dtype=np.int64)):
        raise GlobalDilatedRolloutError("final trajectory boundary coordinates changed")
    milestones = _load_npz(run_dir / "suffix/milestones.npz")
    expected_milestones = states[:, [step // 8 for step in MILESTONE_STEPS], :]
    if not np.array_equal(milestones["states"], expected_milestones):
        raise GlobalDilatedRolloutError("milestone slices no longer equal aggregate states")

    source = load_verified_source_target(args.source_run_dir)
    summary = _read_json(run_dir / "suffix/summary.json", semantic=True)
    zero = raw_state_metrics(states[0, -1], source.mixed_target)
    for row_index, row_key in enumerate(ROW_ORDER):
        metric = raw_state_metrics(states[row_index, -1], source.mixed_target)
        recorded = summary["primary_final_objectives"][row_key]
        if metric.to_dict() != {key: recorded[key] for key in metric.to_dict()}:
            raise GlobalDilatedRolloutError("final primary metrics do not recompute")
        if row_index:
            recomputed = paired_metric_improvement(metric, zero)
            for key, value in recomputed.items():
                if recorded[key] != value:
                    raise GlobalDilatedRolloutError("paired final effect does not recompute")

    freeze = _read_json(run_dir / "evaluation_freeze.json", semantic=True)
    scale = FixedRenderingScale(**freeze["rendering_scale"])
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise GlobalDilatedRolloutError("final image decoder is unavailable") from exc

    def decode(relative: str) -> np.ndarray:
        try:
            with Image.open(run_dir / relative) as opened:
                return np.asarray(opened.convert("L"))
        except OSError as exc:
            raise GlobalDilatedRolloutError(f"generated image cannot decode: {relative}") from exc

    def sheet(cells: Sequence[np.ndarray], columns: int) -> np.ndarray:
        rows = math.ceil(len(cells) / columns)
        value = np.zeros((rows * 28, columns * 28), dtype=np.uint8)
        for index, cell in enumerate(cells):
            value[
                (index // columns) * 28 : (index // columns + 1) * 28,
                (index % columns) * 28 : (index % columns + 1) * 28,
            ] = cell
        return value

    expected_paths: set[str] = set()
    expected_records: list[dict[str, Any]] = []
    raw_cells: list[np.ndarray] = []
    demixed_cells: list[np.ndarray] = []
    milestone_indices = [step // 8 for step in MILESTONE_STEPS]
    for milestone, boundary_index in zip(MILESTONE_STEPS, milestone_indices, strict=True):
        milestone_raw: list[np.ndarray] = []
        milestone_demixed: list[np.ndarray] = []
        for row_index, row_key in enumerate(ROW_ORDER):
            state = states[row_index, boundary_index]
            raw_expected = render_raw_density(state, scale)
            demixed_expected = render_background_demixed(state, scale)
            raw_relative = f"images/raw/{row_key}/step-{milestone:03d}.png"
            demixed_relative = f"images/demixed/{row_key}/step-{milestone:03d}.png"
            if (
                not np.array_equal(decode(raw_relative), raw_expected)
                or not np.array_equal(decode(demixed_relative), demixed_expected)
            ):
                raise GlobalDilatedRolloutError(
                    "a generated milestone image does not reproduce from frozen scale"
                )
            expected_paths.update((raw_relative, demixed_relative))
            expected_records.append(
                {
                    "row_key": row_key,
                    "completed_reverse_steps": milestone,
                    "state_sha256": rollout_array_sha256(state),
                    "raw_path": raw_relative,
                    "raw_sha256": file_fingerprint(run_dir / raw_relative),
                    "demixed_path": demixed_relative,
                    "demixed_sha256": file_fingerprint(run_dir / demixed_relative),
                }
            )
            raw_cells.append(raw_expected)
            demixed_cells.append(demixed_expected)
            milestone_raw.append(raw_expected)
            milestone_demixed.append(demixed_expected)
        for kind, cells in (("raw", milestone_raw), ("demixed", milestone_demixed)):
            relative = f"images/contact-sheets/milestone-{milestone:03d}-{kind}.png"
            expected_paths.add(relative)
            if not np.array_equal(decode(relative), sheet(cells, 5)):
                raise GlobalDilatedRolloutError("a milestone contact sheet changed")
    for relative, cells in (
        ("images/contact-sheets/all-milestones-raw.png", raw_cells),
        ("images/contact-sheets/all-milestones-demixed.png", demixed_cells),
        ("images/contact-sheets/final-raw.png", raw_cells[-5:]),
        ("images/contact-sheets/final-demixed.png", demixed_cells[-5:]),
    ):
        expected_paths.add(relative)
        if not np.array_equal(decode(relative), sheet(cells, 5)):
            raise GlobalDilatedRolloutError("an aggregate contact sheet changed")
    actual_paths = {
        path.relative_to(run_dir).as_posix()
        for path in (run_dir / "images").rglob("*.png")
    }
    if actual_paths != expected_paths or summary.get("images") != expected_records:
        raise GlobalDilatedRolloutError("generated image path/record set changed")

    positive_path = run_dir / "positive_branch.json"
    if positive_path.is_file():
        positive = _read_json(positive_path, semantic=True)
        if exact_count := int(positive.get("completed", 0)):
            positive_arrays = _load_npz(run_dir / "positive/milestones.npz")
            positive_states = positive_arrays.get("states")
            if positive_states is None or positive_states.shape != (3, 5, STATE_SIZE):
                raise GlobalDilatedRolloutError("positive milestone image source changed")
            positive_expected_paths: set[str] = set()
            positive_cells: list[np.ndarray] = []
            for milestone_index, milestone in enumerate((0, 128, 256, 384, 512)):
                for row_index, row_key in enumerate(
                    ("zero", "global-plus-1", "source-informed")
                ):
                    relative = f"positive/images/raw/{row_key}/step-{milestone:03d}.png"
                    expected = render_raw_density(
                        positive_states[row_index, milestone_index], scale
                    )
                    positive_expected_paths.add(relative)
                    positive_cells.append(expected)
                    if not np.array_equal(decode(relative), expected):
                        raise GlobalDilatedRolloutError("a positive-branch image changed")
            contact = "positive/images/contact-sheet-raw.png"
            positive_expected_paths.add(contact)
            if not np.array_equal(decode(contact), sheet(positive_cells, 3)):
                raise GlobalDilatedRolloutError("positive-branch contact sheet changed")
            positive_actual_paths = {
                path.relative_to(run_dir).as_posix()
                for path in (run_dir / "positive/images").rglob("*.png")
            }
            if positive_actual_paths != positive_expected_paths:
                raise GlobalDilatedRolloutError("positive image path set changed")

    required_rows = set(ROW_ORDER)
    if set(summary["primary_final_objectives"]) != required_rows:
        raise GlobalDilatedRolloutError("a scientific row was suppressed from the final summary")
    manifest_rows = _manifest_rows(run_dir)
    manifest = _write_semantic(
        run_dir / "artifact_manifest.json",
        {
            "schema": VERSION + "-artifact-manifest",
            "schema_version": 1,
            "artifact_count": len(manifest_rows),
            "excluded_self_referential_paths": sorted(_FINAL_EXCLUDED),
            "artifacts": manifest_rows,
        },
    )
    checksum_paths = [
        path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and path.relative_to(run_dir).as_posix()
        not in {
            "SHA256SUMS.txt",
            "verification.json",
            "terminal_storage_authority.json",
            "stages/report_verify.json",
        }
        and path.suffix != ".tmp"
    ]
    checksum_text = "".join(
        f"{file_fingerprint(path)}  {path.relative_to(run_dir).as_posix()}\n"
        for path in checksum_paths
    )
    _atomic_text(run_dir / "SHA256SUMS.txt", checksum_text)

    # Reopen and verify the just-created compact integrity inventories.
    manifest_reopened = _read_json(run_dir / "artifact_manifest.json", semantic=True)
    if manifest_reopened != manifest:
        raise GlobalDilatedRolloutError("artifact manifest did not reopen identically")
    for row in manifest_reopened["artifacts"]:
        path = run_dir / row["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size"])
            or file_fingerprint(path) != row["sha256"]
        ):
            raise GlobalDilatedRolloutError("artifact manifest verification failed")
    checksum_lines = (run_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    for line in checksum_lines:
        digest, relative = line.split("  ", 1)
        if file_fingerprint(run_dir / relative) != digest:
            raise GlobalDilatedRolloutError("checksum inventory verification failed")
    return {
        "trajectory_states_sha256": rollout_array_sha256(states),
        "milestones_bitwise_recomputed": 1,
        "primary_metrics_recomputed": 1,
        "images_decoded_and_reproduced": 1,
        "generated_image_path_set_exact": 1,
        "all_five_rows_present": 1,
        "artifact_count": len(manifest_rows),
        "artifact_manifest_file_sha256": file_fingerprint(run_dir / "artifact_manifest.json"),
        "checksum_file_sha256": file_fingerprint(run_dir / "SHA256SUMS.txt"),
        "checksum_entry_count": len(checksum_lines),
        "deep_scientific_authority": deep,
    }


_TERMINAL_STORAGE_ROLE = "terminal_exact_recursive_storage_measurement"


def _serialized_json_size(record: Mapping[str, Any]) -> int:
    return len(
        (
            json.dumps(
                dict(record), indent=2, sort_keys=True, ensure_ascii=False
            )
            + "\n"
        ).encode("utf-8")
    )


def _write_terminal_storage_ledger(
    run_dir: Path,
    *,
    exact_final_bytes: int,
    recorded_at: str,
) -> dict[str, Any]:
    """Replace the provisional terminal measurement with one exact value."""

    current = _resource_ledger(run_dir)
    events = [
        dict(item)
        for item in current.get("events", [])
        if item.get("role") != _TERMINAL_STORAGE_ROLE
    ]
    detail = {
        "measurement_scope": "all_recursive_regular_files_after_terminal_stage_commit",
        "exact_final_storage_authority": 1,
        "temporary_files_present": 0,
    }
    event_id = semantic_sha256({"role": _TERMINAL_STORAGE_ROLE, "detail": detail})
    active = math.fsum(float(item["elapsed_seconds"]) for item in events)
    peak = max((int(item["peak_cuda_memory_bytes"]) for item in events), default=0)
    total = max((int(item["total_cuda_memory_bytes"]) for item in events), default=0)
    breaches: list[str] = []
    if active > ACTIVE_SECONDS_CAP + 1.0e-9:
        breaches.append("active_seconds_cap")
    if total > 0 and peak / total >= CUDA_MEMORY_FRACTION_CAP:
        breaches.append("cuda_memory_fraction_cap")
    if exact_final_bytes >= STORAGE_CAP_BYTES:
        breaches.append("persisted_storage_cap")
    events.append(
        {
            "event_id": event_id,
            "role": _TERMINAL_STORAGE_ROLE,
            "elapsed_seconds": 0.0,
            "peak_cuda_memory_bytes": 0,
            "total_cuda_memory_bytes": 0,
            "detail": detail,
            "recorded_at": recorded_at,
            "limits_passed": int(not breaches),
            "breaches": breaches,
        }
    )
    return _write_semantic(
        run_dir / "resource_ledger.json",
        {
            "schema": VERSION + "-resource-ledger",
            "schema_version": 1,
            "active_seconds_cap": ACTIVE_SECONDS_CAP,
            "storage_cap_bytes": STORAGE_CAP_BYTES,
            "cuda_memory_fraction_cap": CUDA_MEMORY_FRACTION_CAP,
            "events": events,
            "active_seconds": active,
            "maximum_peak_cuda_memory_bytes": peak,
            "maximum_total_cuda_memory_bytes": total,
            "persisted_storage_bytes": int(exact_final_bytes),
            "limits_passed": int(not breaches),
            "breaches": breaches,
        },
    )


def _write_completed_objective_resource_stop(
    run_dir: Path, *, ledger: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = run_dir / "completed_objective_resource_stop.json"
    if ledger.get("limits_passed") == 1:
        path.unlink(missing_ok=True)
        return None
    objective = _mandatory_objective_completion(run_dir)
    if objective["scientific_objective_completed"] != 1:
        raise ResourceBoundaryError(
            "terminal resources crossed a cap before the mandatory objective completed"
        )
    positive = _read_json(run_dir / "positive_branch.json", semantic=True)
    return _write_semantic(
        path,
        {
            "schema": VERSION + "-completed-objective-resource-stop",
            "schema_version": 1,
            "scientific_objective_completed": 1,
            "mandatory_objective_authority": objective,
            "resource_failure_message": (
                "terminal exact resource measurement crossed configured cap(s): "
                + ", ".join(str(item) for item in ledger.get("breaches", []))
            ),
            "resource_ledger_semantic_sha256": ledger["semantic_sha256"],
            "breaches": list(ledger.get("breaches", [])),
            "active_seconds_after_reserved_charge": float(ledger["active_seconds"]),
            "active_seconds_cap": ACTIVE_SECONDS_CAP,
            "exact_final_persisted_storage_bytes": int(
                ledger["persisted_storage_bytes"]
            ),
            "persisted_storage_cap_bytes": STORAGE_CAP_BYTES,
            "optional_branch": {
                "triggered": positive.get("triggered", 0),
                "attempted": positive.get("attempted", 0),
                "completed": positive.get("completed", 0),
                "reason": positive.get("reason"),
            },
            "classification": (
                "mandatory_scientific_objective_complete; terminal exact resource "
                "measurement crossed configured cap"
            ),
        },
    )


def _pending_terminal_storage_finalization(run_dir: Path) -> bool:
    if _stage_complete(run_dir, "report_verify"):
        return False
    ledger = _resource_ledger(run_dir)
    terminal = [
        event
        for event in ledger.get("events", [])
        if event.get("role") == _TERMINAL_STORAGE_ROLE
    ]
    return bool(
        len(terminal) == 1
        and terminal[0].get("detail", {}).get("exact_final_storage_authority") == 1
        and _mandatory_objective_completion(run_dir)[
            "scientific_objective_completed"
        ]
        == 1
    )


def _finalize_and_verify(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if _stage_complete(run_dir, "report_verify"):
        return _verify_completed_report_read_only(run_dir, args)
    stage_path = run_dir / "stages/report_verify.json"
    if stage_path.is_file():
        raise GlobalDilatedRolloutError(
            "nonpassing terminal report stage marker cannot be rebuilt"
        )

    existing_authority_path = run_dir / "terminal_storage_authority.json"
    existing_authority = (
        _read_json(existing_authority_path, semantic=True)
        if existing_authority_path.is_file()
        else None
    )
    terminal_at = (
        str(existing_authority["measured_at"])
        if existing_authority is not None
        else _utc_now()
    )
    ledger_before = _resource_ledger(run_dir)
    terminal_event_exists = any(
        event.get("role") == _TERMINAL_STORAGE_ROLE
        for event in ledger_before.get("events", [])
    )
    if not terminal_event_exists:
        try:
            _record_resource_event(
                run_dir,
                role="report_and_final_verification_reserved_charge",
                elapsed_seconds=REPORT_RESERVE_SECONDS,
                detail={
                    "charge_type": "conservative_frozen_reserve",
                    "covers": "reports_final_inventories_reopen_and_terminal_read_only_audit",
                },
            )
        except ResourceBoundaryError:
            if _mandatory_objective_completion(run_dir)[
                "scientific_objective_completed"
            ] != 1:
                raise

    guess = (
        int(existing_authority["exact_recursive_file_bytes"])
        if existing_authority is not None
        else _directory_bytes(run_dir) + 4096
    )
    record: dict[str, Any] | None = None
    for _iteration in range(16):
        ledger = _write_terminal_storage_ledger(
            run_dir, exact_final_bytes=guess, recorded_at=terminal_at
        )
        report_resource_stop = _write_completed_objective_resource_stop(
            run_dir, ledger=ledger
        )
        authority = _write_semantic(
            existing_authority_path,
            {
                "schema": VERSION + "-terminal-storage-authority",
                "schema_version": 1,
                "measurement_scope": "all_recursive_regular_files_after_terminal_stage_commit",
                "exact_recursive_file_bytes": guess,
                "storage_cap_bytes": STORAGE_CAP_BYTES,
                "resource_limits_passed": int(report_resource_stop is None),
                "breaches": list(ledger.get("breaches", [])),
                "resource_ledger_semantic_sha256": ledger["semantic_sha256"],
                "temporary_files_present": 0,
                "measured_at": terminal_at,
            },
        )
        _write_reports(run_dir, args)
        # The first pass absorbs any provisional inventory bytes; the second
        # proves the registered set and checksum text are stable.
        _verify_raw_and_derived(run_dir, args)
        verification_detail = _verify_raw_and_derived(run_dir, args)
        record = _write_semantic(
            run_dir / "verification.json",
            {
                "schema": VERSION + "-verification",
                "schema_version": 1,
                "passed": 1,
                "verified_at": terminal_at,
                "checks": verification_detail,
                "self_referential_exclusions": sorted(_FINAL_EXCLUDED),
                "representative_npz": "suffix/trajectory_shard_boundaries.npz",
                "representative_png": "images/raw/global-plus-1/step-128.png",
                "scientific_objective_completed": 1,
                "resource_limits_passed": int(report_resource_stop is None),
                "completed_objective_resource_stop_semantic_sha256": (
                    None
                    if report_resource_stop is None
                    else report_resource_stop["semantic_sha256"]
                ),
                "terminal_storage_authority_semantic_sha256": authority[
                    "semantic_sha256"
                ],
            },
        )
        stage_record = _semantic(
            {
                "schema": VERSION + "-stage",
                "schema_version": 1,
                "stage": "report_verify",
                "passed": 1,
                "completed_at": terminal_at,
                "detail": {
                    "raw_and_derived_checks_passed": 1,
                    "final_inventory_stable": 1,
                    "scientific_objective_completed": 1,
                    "resource_limits_passed": int(report_resource_stop is None),
                    "terminal_storage_authority_semantic_sha256": authority[
                        "semantic_sha256"
                    ],
                    "exact_recursive_file_bytes": guess,
                },
            }
        )
        candidate_total = _directory_bytes(run_dir) + _serialized_json_size(
            stage_record
        )
        if candidate_total != guess:
            guess = candidate_total
            continue
        atomic_write_json(stage_path, stage_record)
        if _directory_bytes(run_dir) != guess:
            raise GlobalDilatedRolloutError(
                "terminal exact storage fixed point changed at commit"
            )
        break
    else:
        raise GlobalDilatedRolloutError(
            "terminal exact storage accounting did not converge"
        )
    if record is None:
        raise GlobalDilatedRolloutError("terminal verification record is absent")
    _verify_completed_report_read_only(run_dir, args)
    return record


def _verify_completed_report_read_only(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Strictly verify a completed bundle without rewriting a byte."""

    _verify_scientific_evidence_read_only(run_dir, args)

    verification_path = run_dir / "verification.json"
    manifest_path = run_dir / "artifact_manifest.json"
    checksums_path = run_dir / "SHA256SUMS.txt"
    for path in (verification_path, manifest_path, checksums_path, run_dir / "REPORT.md", run_dir / "HANDOFF.md"):
        if not path.is_file():
            raise GlobalDilatedRolloutError(f"completed report omits {path.name}")
    verification = _read_json(verification_path, semantic=True)
    storage_authority = _read_json(
        run_dir / "terminal_storage_authority.json", semantic=True
    )
    report_stage = _read_json(
        run_dir / "stages/report_verify.json", semantic=True
    )
    ledger = _resource_ledger(run_dir)
    if int(verification.get("passed", 0)) != 1:
        raise GlobalDilatedRolloutError("completed verification is not passing")
    resource_stop_path = run_dir / "completed_objective_resource_stop.json"
    resource_stop = (
        _read_json(resource_stop_path, semantic=True)
        if resource_stop_path.is_file()
        else None
    )
    if (
        verification.get("scientific_objective_completed") != 1
        or verification.get("resource_limits_passed")
        != int(resource_stop is None)
        or verification.get(
            "completed_objective_resource_stop_semantic_sha256"
        )
        != (None if resource_stop is None else resource_stop["semantic_sha256"])
        or verification.get("terminal_storage_authority_semantic_sha256")
        != storage_authority.get("semantic_sha256")
        or storage_authority.get("schema")
        != VERSION + "-terminal-storage-authority"
        or storage_authority.get("resource_ledger_semantic_sha256")
        != ledger.get("semantic_sha256")
        or storage_authority.get("resource_limits_passed")
        != ledger.get("limits_passed")
        or storage_authority.get("breaches") != ledger.get("breaches")
        or storage_authority.get("exact_recursive_file_bytes")
        != ledger.get("persisted_storage_bytes")
        or storage_authority.get("temporary_files_present") != 0
        or storage_authority.get("exact_recursive_file_bytes")
        != _directory_bytes(run_dir)
        or report_stage.get("stage") != "report_verify"
        or report_stage.get("passed") != 1
        or report_stage.get("detail", {}).get(
            "terminal_storage_authority_semantic_sha256"
        )
        != storage_authority.get("semantic_sha256")
        or report_stage.get("detail", {}).get("exact_recursive_file_bytes")
        != storage_authority.get("exact_recursive_file_bytes")
    ):
        raise GlobalDilatedRolloutError("completed objective/resource status changed")
    if resource_stop is not None:
        positive = _read_json(run_dir / "positive_branch.json", semantic=True)
        if (
            resource_stop.get("scientific_objective_completed") != 1
            or ledger.get("limits_passed") != 0
            or resource_stop.get("resource_ledger_semantic_sha256")
            != ledger.get("semantic_sha256")
            or resource_stop.get("exact_final_persisted_storage_bytes")
            != ledger.get("persisted_storage_bytes")
            or resource_stop.get("optional_branch")
            != {
                "triggered": positive.get("triggered", 0),
                "attempted": positive.get("attempted", 0),
                "completed": positive.get("completed", 0),
                "reason": positive.get("reason"),
            }
        ):
            raise GlobalDilatedRolloutError("completed resource-stop authority changed")
    manifest = _read_json(manifest_path, semantic=True)
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or int(manifest.get("artifact_count", -1)) != len(rows):
        raise GlobalDilatedRolloutError("completed artifact manifest schema changed")
    for row in rows:
        path = run_dir / str(row.get("path", ""))
        if (
            not path.is_file()
            or path.stat().st_size != int(row.get("size", -1))
            or file_fingerprint(path) != row.get("sha256")
        ):
            raise GlobalDilatedRolloutError("completed registered artifact changed")
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.relative_to(run_dir).as_posix() not in _FINAL_EXCLUDED
        and path.suffix != ".tmp"
    }
    if actual != {str(row["path"]) for row in rows}:
        raise GlobalDilatedRolloutError("completed registered artifact path set changed")
    checksum_rows: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise GlobalDilatedRolloutError("completed checksum row changed") from exc
        if (
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not relative
            or relative in checksum_rows
        ):
            raise GlobalDilatedRolloutError("completed checksum row authority changed")
        checksum_rows[relative] = digest
        if file_fingerprint(run_dir / relative) != digest:
            raise GlobalDilatedRolloutError("completed checksum inventory changed")
    checksum_expected = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.relative_to(run_dir).as_posix()
        not in {
            "SHA256SUMS.txt",
            "verification.json",
            "terminal_storage_authority.json",
            "stages/report_verify.json",
        }
        and path.suffix != ".tmp"
    }
    if set(checksum_rows) != checksum_expected:
        raise GlobalDilatedRolloutError("completed checksum path set changed")
    # Recompute representative raw/derived claims without touching inventories.
    aggregate = _load_npz(run_dir / "suffix/trajectory_shard_boundaries.npz")
    aggregation = _read_json(run_dir / "suffix/trajectory_aggregation.json", semantic=True)
    if rollout_array_sha256(aggregate["states"]) != aggregation["states_sha256"]:
        raise GlobalDilatedRolloutError("completed trajectory aggregate changed")
    source = load_verified_source_target(args.source_run_dir)
    summary = _read_json(run_dir / "suffix/summary.json", semantic=True)
    freeze = _read_json(run_dir / "evaluation_freeze.json", semantic=True)
    _verify_all_generated_images_read_only(
        run_dir,
        states=aggregate["states"],
        summary=summary,
        scale=FixedRenderingScale(**freeze["rendering_scale"]),
    )
    zero = raw_state_metrics(aggregate["states"][0, -1], source.mixed_target)
    for index, row_key in enumerate(ROW_ORDER):
        metric = raw_state_metrics(aggregate["states"][index, -1], source.mixed_target)
        recorded = summary["primary_final_objectives"][row_key]
        if metric.to_dict() != {key: recorded[key] for key in metric.to_dict()}:
            raise GlobalDilatedRolloutError("completed primary metric changed")
        if index and paired_metric_improvement(metric, zero) != {
            key: recorded[key] for key in paired_metric_improvement(metric, zero)
        }:
            raise GlobalDilatedRolloutError("completed paired objective changed")
    return verification


def _failure_classification(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, PostprocessingResourceStopError):
        return {
            "failure_domain": "resource_budget",
            "failure_code": "raw_exact_suffix_complete_postprocessing_incomplete",
            "resume_same_frozen_run_authorized": 0,
            "required_next_action": (
                "preserve and report the sealed 16-of-16 raw exact suffix; do not "
                "replay postprocessing or optional compute under the exhausted cap"
            ),
        }
    resource = isinstance(exc, ResourceBoundaryError)
    return {
        "failure_domain": "resource_budget" if resource else "execution_integrity",
        "failure_code": (
            "global_rollout_resource_boundary"
            if resource
            else "global_rollout_integrity_or_numerical_failure"
        ),
        "resume_same_frozen_run_authorized": int(resource),
        "required_next_action": (
            "resume the unchanged frozen run from the last committed exact boundary if the same resource cap admits it"
            if resource
            else "diagnose the sealed input/control/numerical defect; repair it in a new immutable run before resuming scientific interpretation"
        ),
    }


def _write_failure_record_from_capture(
    run_dir: Path, capture: Mapping[str, Any]
) -> dict[str, Any]:
    return _write_semantic(
        run_dir / "failure/failure.json",
        {
            "schema": VERSION + "-failure",
            "schema_version": 1,
            "failure_generation": int(capture["failure_generation"]),
            "stage": str(capture["stage"]),
            "exception_type": str(capture["exception_type"]),
            "message": str(capture["message"]),
            "failure_domain": str(capture["failure_domain"]),
            "failure_code": str(capture["failure_code"]),
            "resume_same_frozen_run_authorized": int(
                capture["resume_same_frozen_run_authorized"]
            ),
            "mandatory_objective_authority": dict(
                capture["mandatory_objective_authority"]
            ),
            "last_valid_states_saved_at_atomic_capture": int(
                capture["last_valid_states_saved_at_atomic_capture"]
            ),
            "approximate_primary_substitution_performed": 0,
            "failed_at": str(capture["captured_at"]),
            "active_failure_capture_semantic_sha256": str(
                capture["semantic_sha256"]
            ),
        },
    )


def _write_failure_evidence_record(
    run_dir: Path,
    capture: Mapping[str, Any],
    *,
    evidence_attempt_completed: bool,
    evidence_error: BaseException | None,
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for relative in (
        "failure/last_valid_states.npz",
        "failure/last_valid_contact_sheet.png",
    ):
        path = run_dir / relative
        if path.is_file():
            artifacts.append(
                {
                    "path": relative,
                    "size": int(path.stat().st_size),
                    "sha256": file_fingerprint(path),
                }
            )
    return _write_semantic(
        run_dir / "failure/failure_evidence.json",
        {
            "schema": VERSION + "-failure-evidence",
            "schema_version": 1,
            "failure_generation": int(capture["failure_generation"]),
            "active_failure_capture_semantic_sha256": str(
                capture["semantic_sha256"]
            ),
            "evidence_attempt_completed": int(evidence_attempt_completed),
            "capture_recovery_after_interruption": int(
                not evidence_attempt_completed and evidence_error is None
            ),
            "last_valid_states_saved": int(
                any(row["path"].endswith("last_valid_states.npz") for row in artifacts)
            ),
            "last_valid_contact_sheet_saved": int(
                any(
                    row["path"].endswith("last_valid_contact_sheet.png")
                    for row in artifacts
                )
            ),
            "completed_artifacts": artifacts,
            "evidence_error": (
                None
                if evidence_error is None
                else {
                    "exception_type": type(evidence_error).__name__,
                    "message": str(evidence_error),
                }
            ),
            "approximate_primary_substitution_performed": 0,
        },
    )


def _ensure_failure_evidence_record(
    run_dir: Path, capture: Mapping[str, Any]
) -> dict[str, Any]:
    path = run_dir / "failure/failure_evidence.json"
    if not path.is_file():
        return _write_failure_evidence_record(
            run_dir,
            capture,
            evidence_attempt_completed=False,
            evidence_error=None,
        )
    record = _read_json(path, semantic=True)
    rows = record.get("completed_artifacts")
    if (
        record.get("schema") != VERSION + "-failure-evidence"
        or record.get("failure_generation") != capture.get("failure_generation")
        or record.get("active_failure_capture_semantic_sha256")
        != capture.get("semantic_sha256")
        or not isinstance(rows, list)
    ):
        raise GlobalDilatedRolloutError("failure evidence authority changed")
    for row in rows:
        artifact = run_dir / str(row.get("path", ""))
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(row.get("size", -1))
            or file_fingerprint(artifact) != row.get("sha256")
        ):
            raise GlobalDilatedRolloutError("failure evidence artifact changed")
    return record


def _capture_failure(
    run_dir: Path, stage: str, exc: BaseException
) -> dict[str, Any]:
    active_capture_path = run_dir / "active_failure_capture.json"
    if active_capture_path.is_file():
        raise GlobalDilatedRolloutError(
            "a live failure capture already exists and cannot be overwritten"
        )
    classification = _failure_classification(exc)
    objective = _mandatory_objective_completion(run_dir)
    generation = len(_failure_supersession_records(run_dir))
    immutable_context = {
        relative: file_fingerprint(run_dir / relative)
        for relative in (
            "scientific_config.json",
            "exact_command.txt",
            "run_manifest.json",
            "input_bindings.json",
            "evaluation_freeze.json",
        )
        if (run_dir / relative).is_file()
    }
    capture = _write_semantic(
        active_capture_path,
        {
            "schema": VERSION + "-active-failure-capture",
            "schema_version": 1,
            "failure_generation": generation,
            "stage": stage,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            **classification,
            "mandatory_objective_authority": objective,
            "immutable_context_file_sha256": immutable_context,
            "last_valid_states_saved_at_atomic_capture": 0,
            "failure_evidence_attempted_at_atomic_capture": 0,
            "approximate_primary_substitution_performed": 0,
            "captured_at": _utc_now(),
        },
    )
    failure_root = run_dir / "failure"
    failure_root.mkdir(parents=True, exist_ok=True)
    _write_failure_record_from_capture(run_dir, capture)
    evidence_warning: BaseException | None = None
    try:
        last_state: np.ndarray | None = None
        shard_root = run_dir / "suffix/fused_families/fresh-five-row/suffix-128"
        if shard_root.is_dir():
            committed = sorted(shard_root.glob("shard-*.json"))
            for record_path in reversed(committed):
                try:
                    record = _read_json(record_path, semantic=True)
                    archive_path = record_path.with_suffix(".npz")
                    if int(record.get("committed", 0)) == 1 and archive_path.is_file():
                        candidate = _load_npz(archive_path).get("state")
                        if candidate is not None and candidate.shape == (
                            len(ROW_ORDER),
                            STATE_SIZE,
                        ):
                            last_state = np.ascontiguousarray(candidate, dtype=np.float64)
                            break
                except Exception:
                    continue
        if last_state is None:
            anchor_path = run_dir / "fresh_forward/anchor-step-0127.npz"
            if anchor_path.is_file():
                anchor = _load_npz(anchor_path).get("state")
                if anchor is not None and anchor.shape == (STATE_SIZE,):
                    last_state = np.repeat(
                        anchor[None, :], len(ROW_ORDER), axis=0
                    )
        if last_state is not None:
            atomic_rollout_npz(
                failure_root / "last_valid_states.npz", {"state": last_state}
            )
            try:
                freeze = _read_json(run_dir / "evaluation_freeze.json", semantic=True)
                scale = FixedRenderingScale(**freeze["rendering_scale"])
                cells = [render_raw_density(row, scale) for row in last_state]
                _save_contact_sheet(
                    failure_root / "last_valid_contact_sheet.png", cells, columns=5
                )
            except Exception as contact_exc:
                evidence_warning = contact_exc
    except Exception as evidence_exc:
        _write_failure_evidence_record(
            run_dir,
            capture,
            evidence_attempt_completed=False,
            evidence_error=evidence_exc,
        )
        raise
    _write_failure_evidence_record(
        run_dir,
        capture,
        evidence_attempt_completed=True,
        evidence_error=evidence_warning,
    )
    return capture


def _mandatory_objective_completion(run_dir: Path) -> dict[str, Any]:
    shard_root = run_dir / "suffix/fused_families/fresh-five-row/suffix-128"
    committed: list[str] = []
    if shard_root.is_dir():
        for path in sorted(shard_root.glob("shard-*.json")):
            try:
                record = _read_json(path, semantic=True)
            except GlobalDilatedRolloutError:
                continue
            if record.get("committed") == 1:
                committed.append(path.relative_to(run_dir).as_posix())
    family_path = run_dir / "suffix/family_summary.json"
    family_complete = 0
    if family_path.is_file():
        family = _read_json(family_path, semantic=True)
        family_complete = int(
            family.get("completed") == 1
            and family.get("row_order") == list(ROW_ORDER)
            and family.get("strict_exact_health", {}).get("passed") == 1
        )
    evaluate_marker_complete = int(_stage_complete(run_dir, "evaluate_exact"))
    completed = int(
        evaluate_marker_complete == 1
        and family_complete == 1
        and len(committed) == MANDATORY_SUFFIX_STEPS // 8
    )
    return {
        "scientific_objective_completed": completed,
        "evaluate_exact_marker_complete": evaluate_marker_complete,
        "family_summary_complete": family_complete,
        "committed_exact_suffix_shard_count": len(committed),
        "committed_exact_suffix_shards": committed,
    }


def _verify_raw_exact_suffix_complete_read_only(run_dir: Path) -> dict[str, Any]:
    """Authenticate the 16-shard objective when postprocessing never began."""

    anchor_path = run_dir / "fresh_forward/anchor-step-0127.npz"
    anchor = _load_npz(anchor_path).get("state")
    forward = _read_json(run_dir / "fresh_forward/forward_summary.json", semantic=True)
    freeze_path = run_dir / "evaluation_freeze.json"
    freeze = _read_json(freeze_path, semantic=True)
    path_usage_path = run_dir / "path_usage.json"
    path_usage = _read_json(path_usage_path, semantic=True)
    diagnostics = forward.get("diagnostics")
    if (
        anchor is None
        or anchor.dtype != np.float64
        or anchor.shape != (STATE_SIZE,)
        or not np.isfinite(anchor).all()
        or np.any(anchor < 0.0)
        or abs(float(np.sum(anchor)) - 1.0) > 2.0e-12
        or forward.get("anchor_state_sha256") != rollout_array_sha256(anchor)
        or forward.get("anchor_archive_sha256") != rollout_file_sha256(anchor_path)
        or forward.get("evaluation_freeze_file_sha256")
        != file_fingerprint(freeze_path)
        or forward.get("path_usage_file_sha256")
        != file_fingerprint(path_usage_path)
        or forward.get("path_id") != path_usage.get("fresh_path_id")
        or forward.get("source_target_sha256") != freeze.get("source_target_sha256")
        or path_usage.get("evaluation_freeze_file_sha256")
        != file_fingerprint(freeze_path)
        or path_usage.get("fresh_path_id") not in FRESH_PATH_POOL
        or not isinstance(diagnostics, Mapping)
        or diagnostics.get("passed") != 1
        or diagnostics.get("restart_chain_valid") != 1
        or diagnostics.get("authorization_fraction") != 1.0
        or diagnostics.get("certificate_fraction") != 1.0
        or diagnostics.get("forbidden_event_count") != 0
        or diagnostics.get("output_state_nonfinite_count") != 0
        or diagnostics.get("output_state_negative_count") != 0
        or float(diagnostics.get("maximum_output_state_mass_error", math.inf))
        > 2.0e-12
    ):
        raise GlobalDilatedRolloutError(
            "raw-suffix resource stop forward anchor authority changed"
        )
    root = run_dir / "suffix/fused_families/fresh-five-row/suffix-128"
    expected_json = {f"shard-{index:04d}.json" for index in range(16)}
    expected_npz = {f"shard-{index:04d}.npz" for index in range(16)}
    if (
        {path.name for path in root.glob("shard-*.json")} != expected_json
        or {path.name for path in root.glob("shard-*.npz")} != expected_npz
    ):
        raise GlobalDilatedRolloutError(
            "raw-suffix resource stop shard path set changed"
        )
    records: list[dict[str, Any]] = []
    state = np.repeat(anchor[None, :], len(ROW_ORDER), axis=0)
    previous_hash = rollout_array_sha256(state)
    for index in range(16):
        record_path = root / f"shard-{index:04d}.json"
        archive_path = root / f"shard-{index:04d}.npz"
        record = _read_json(record_path, semantic=True)
        arrays = _load_npz(archive_path)
        endpoint = arrays.get("state")
        if (
            set(arrays) != {"state"}
            or endpoint is None
            or endpoint.dtype != np.float64
            or endpoint.shape != (len(ROW_ORDER), STATE_SIZE)
            or not np.isfinite(endpoint).all()
            or np.any(endpoint < 0.0)
            or float(np.max(np.abs(np.sum(endpoint, axis=1) - 1.0))) > 2.0e-12
            or record.get("committed") != 1
            or record.get("input_state_sha256") != previous_hash
            or record.get("state_file_sha256") != rollout_file_sha256(archive_path)
            or record.get("output_state_sha256") != rollout_array_sha256(endpoint)
        ):
            raise GlobalDilatedRolloutError(
                "raw-suffix resource stop shard authority changed"
            )
        state = np.ascontiguousarray(endpoint)
        previous_hash = rollout_array_sha256(state)
        records.append(record)
    strict = _strict_fused_exact_health(
        final_state=state, shard_records=records, row_count=len(ROW_ORDER)
    )
    family_path = run_dir / "suffix/family_summary.json"
    family = _read_json(family_path, semantic=True)
    expected_record_paths = [
        f"suffix/fused_families/fresh-five-row/suffix-128/shard-{index:04d}.json"
        for index in range(16)
    ]
    if (
        family.get("completed") != 1
        or family.get("reference_contract") != "certified_exact"
        or family.get("row_order") != list(ROW_ORDER)
        or family.get("strict_exact_health") != strict
        or family.get("shard_record_paths") != expected_record_paths
        or family.get("failed_rows_suppressed") != 0
    ):
        raise GlobalDilatedRolloutError(
            "raw-suffix resource stop family authority changed"
        )
    return {
        "raw_exact_suffix_complete": 1,
        "committed_exact_suffix_shard_count": 16,
        "family_summary_semantic_sha256": family["semantic_sha256"],
        "strict_exact_health": strict,
        "final_state_sha256": rollout_array_sha256(state),
        "anchor_state_sha256": rollout_array_sha256(anchor),
    }


def _pending_raw_suffix_postprocessing_resource_stop(run_dir: Path) -> bool:
    """Recognize the exact cap seam after family commit, before aggregation."""

    if _stage_complete(run_dir, "evaluate_exact") or _stage_complete(
        run_dir, "report_verify"
    ):
        return False
    ledger = _resource_ledger(run_dir)
    if ledger.get("limits_passed") != 0 or not ledger.get("breaches"):
        return False
    finish_events = [
        event
        for event in ledger.get("events", [])
        if event.get("role") == "mandatory_exact_five_row_suffix_attempt"
        and event.get("detail", {}).get("failed") == 0
        and event.get("detail", {}).get("shard_count") == 16
        and event.get("detail", {}).get("row_count") == len(ROW_ORDER)
        and event.get("limits_passed") == 0
        and event.get("breaches") == ledger.get("breaches")
    ]
    if len(finish_events) != 1:
        return False
    raw = _mandatory_objective_completion(run_dir)
    if (
        raw["evaluate_exact_marker_complete"] != 0
        or raw["family_summary_complete"] != 1
        or raw["committed_exact_suffix_shard_count"] != 16
    ):
        return False
    derived_paths = (
        "suffix/trajectory_shard_boundaries.npz",
        "suffix/milestones.npz",
        "suffix/trajectory_aggregation.json",
        "suffix/metrics.csv",
        "suffix/mechanism.json",
        "suffix/summary.json",
        "outcome.json",
        "positive_branch.json",
    )
    return not any((run_dir / relative).exists() for relative in derived_paths) and not any(
        (run_dir / "images").rglob("*.png")
    )


def _raw_suffix_conversion_error() -> PostprocessingResourceStopError:
    return PostprocessingResourceStopError(
        "16-of-16 raw exact suffix completed, but its durable finish event "
        "crossed the resource cap before mandatory aggregation, metrics, images, "
        "and outcome could begin"
    )


def _raw_suffix_conversion_intent(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / _RAW_SUFFIX_CONVERSION_INTENT
    if not path.is_file():
        return None
    intent = _read_json(path, semantic=True)
    ledger = _resource_ledger(run_dir)
    raw_authority = _verify_raw_exact_suffix_complete_read_only(run_dir)
    classification = _failure_classification(_raw_suffix_conversion_error())
    source_generation = intent.get("source_failure_generation")
    if (
        intent.get("schema") != VERSION + "-raw-suffix-conversion-intent"
        or not isinstance(source_generation, int)
        or isinstance(source_generation, bool)
        or source_generation < 0
        or intent.get("target_failure_generation") != source_generation + 1
        or intent.get("source_failure_code") != "global_rollout_resource_boundary"
        or intent.get("source_resume_same_frozen_run_authorized") != 1
        or not isinstance(intent.get("source_terminal_semantic_sha256"), str)
        or not re.fullmatch(
            r"[0-9a-f]{64}", str(intent.get("source_terminal_semantic_sha256"))
        )
        or not isinstance(intent.get("source_active_package_semantic_sha256"), str)
        or not isinstance(intent.get("source_capture_semantic_sha256"), str)
        or not isinstance(intent.get("source_verification_file_sha256"), str)
        or intent.get("resource_ledger_semantic_sha256")
        != ledger.get("semantic_sha256")
        or intent.get("raw_exact_suffix_authority") != raw_authority
        or intent.get("target_failure_domain") != classification["failure_domain"]
        or intent.get("target_failure_code") != classification["failure_code"]
        or intent.get("target_resume_same_frozen_run_authorized")
        != classification["resume_same_frozen_run_authorized"]
        or intent.get("target_exception_type")
        != PostprocessingResourceStopError.__name__
        or intent.get("target_message") != str(_raw_suffix_conversion_error())
    ):
        raise GlobalDilatedRolloutError(
            "raw-suffix conversion intent authority changed"
        )
    return intent


def _begin_raw_suffix_conversion_intent(run_dir: Path) -> dict[str, Any]:
    existing = _raw_suffix_conversion_intent(run_dir)
    if existing is not None:
        return existing
    terminal = _live_terminal_failure(run_dir)
    capture = _live_failure_capture(run_dir)
    active = _read_json(run_dir / "active_failure_package.json", semantic=True)
    if (
        terminal is None
        or capture is None
        or terminal.get("resume_same_frozen_run_authorized") != 1
        or terminal.get("failure_code") != "global_rollout_resource_boundary"
        or terminal.get("exception_type") != ResourceBoundaryError.__name__
        or capture.get("resume_same_frozen_run_authorized") != 1
        or capture.get("failure_code") != "global_rollout_resource_boundary"
        or capture.get("exception_type") != ResourceBoundaryError.__name__
        or not _failure_verification_matches_terminal(run_dir, terminal)
        or not _pending_raw_suffix_postprocessing_resource_stop(run_dir)
    ):
        raise GlobalDilatedRolloutError(
            "raw-suffix resource conversion authority is absent"
        )
    _verify_completed_failure_read_only(run_dir)
    records = _failure_supersession_records(run_dir)
    generation = len(records)
    if terminal.get("failure_generation") != generation:
        raise GlobalDilatedRolloutError(
            "raw-suffix source failure generation changed"
        )
    error = _raw_suffix_conversion_error()
    classification = _failure_classification(error)
    return _write_semantic(
        run_dir / _RAW_SUFFIX_CONVERSION_INTENT,
        {
            "schema": VERSION + "-raw-suffix-conversion-intent",
            "schema_version": 1,
            "source_failure_generation": generation,
            "target_failure_generation": generation + 1,
            "source_failure_code": terminal["failure_code"],
            "source_resume_same_frozen_run_authorized": 1,
            "source_terminal_semantic_sha256": terminal["semantic_sha256"],
            "source_active_package_semantic_sha256": active["semantic_sha256"],
            "source_capture_semantic_sha256": capture["semantic_sha256"],
            "source_verification_file_sha256": file_fingerprint(
                run_dir / "verification.json"
            ),
            "resource_ledger_semantic_sha256": _resource_ledger(run_dir)[
                "semantic_sha256"
            ],
            "raw_exact_suffix_authority": (
                _verify_raw_exact_suffix_complete_read_only(run_dir)
            ),
            "target_failure_domain": classification["failure_domain"],
            "target_failure_code": classification["failure_code"],
            "target_resume_same_frozen_run_authorized": classification[
                "resume_same_frozen_run_authorized"
            ],
            "target_exception_type": type(error).__name__,
            "target_message": str(error),
            "scientific_stage_reentry_authorized": 0,
            "resume_accounting_authorized": 0,
        },
    )


def _complete_raw_suffix_conversion_intent(run_dir: Path) -> dict[str, Any]:
    intent = _raw_suffix_conversion_intent(run_dir)
    if intent is None:
        raise GlobalDilatedRolloutError("raw-suffix conversion intent is absent")
    source_generation = int(intent["source_failure_generation"])
    target_generation = int(intent["target_failure_generation"])
    records = _failure_supersession_records(run_dir)
    if len(records) == source_generation:
        terminal = _live_terminal_failure(run_dir)
        active = _read_json(run_dir / "active_failure_package.json", semantic=True)
        capture = _live_failure_capture(run_dir)
        if (
            terminal is None
            or capture is None
            or terminal.get("semantic_sha256")
            != intent["source_terminal_semantic_sha256"]
            or active.get("semantic_sha256")
            != intent["source_active_package_semantic_sha256"]
            or capture.get("semantic_sha256")
            != intent["source_capture_semantic_sha256"]
            or file_fingerprint(run_dir / "verification.json")
            != intent["source_verification_file_sha256"]
        ):
            raise GlobalDilatedRolloutError(
                "raw-suffix conversion source package changed"
            )
        _supersede_authorized_failure(run_dir)
        records = _failure_supersession_records(run_dir)
    if len(records) != target_generation:
        raise GlobalDilatedRolloutError(
            "raw-suffix conversion supersession generation changed"
        )
    source_record = records[source_generation]
    if (
        source_record.get("sequence_index") != source_generation
        or source_record.get("terminal_failure_semantic_sha256")
        != intent["source_terminal_semantic_sha256"]
        or source_record.get("active_failure_capture_semantic_sha256")
        != intent["source_capture_semantic_sha256"]
    ):
        raise GlobalDilatedRolloutError(
            "raw-suffix conversion supersession binding changed"
        )
    _complete_pending_failure_retirement(run_dir)

    error = _raw_suffix_conversion_error()
    raw_authority = intent["raw_exact_suffix_authority"]
    stop_body = {
        "schema": VERSION + "-postprocessing-resource-stop",
        "schema_version": 1,
        "raw_exact_suffix_complete": 1,
        "raw_exact_suffix_authority": raw_authority,
        "mandatory_postprocessing_complete": 0,
        "optional_branch_attempted": 0,
        "resource_boundary_message": str(error),
        "resource_ledger_semantic_sha256": intent[
            "resource_ledger_semantic_sha256"
        ],
        "raw_suffix_conversion_intent_semantic_sha256": intent[
            "semantic_sha256"
        ],
        "resume_same_frozen_run_authorized": 0,
        "required_next_action": (
            "preserve and report the sealed 16-of-16 raw exact suffix; do not "
            "replay postprocessing or optional compute under the exhausted cap"
        ),
    }
    stop_path = run_dir / "postprocessing_resource_stop.json"
    expected_stop = _semantic(stop_body)
    if stop_path.is_file():
        if _read_json(stop_path, semantic=True) != expected_stop:
            raise GlobalDilatedRolloutError(
                "raw-suffix postprocessing-stop record changed"
            )
    else:
        _write_semantic(stop_path, stop_body)

    capture = _live_failure_capture(run_dir)
    if capture is None:
        capture = _capture_failure(run_dir, "evaluate_exact", error)
    if (
        capture.get("failure_generation") != target_generation
        or capture.get("exception_type") != type(error).__name__
        or capture.get("message") != str(error)
        or capture.get("resume_same_frozen_run_authorized") != 0
        or capture.get("failure_code")
        != "raw_exact_suffix_complete_postprocessing_incomplete"
    ):
        raise GlobalDilatedRolloutError(
            "raw-suffix conversion target capture changed"
        )
    # The new atomic capture is now the live recovery authority.  Only after
    # that commit may the transaction intent be retired.  A death after this
    # unlink is handled by the generic capture-only packaging path.
    (run_dir / _RAW_SUFFIX_CONVERSION_INTENT).unlink(missing_ok=True)
    verification = _finalize_failure(run_dir, "evaluate_exact", error)
    terminal = _live_terminal_failure(run_dir)
    if (
        terminal is None
        or terminal.get("resume_same_frozen_run_authorized") != 0
        or terminal.get("scientific_objective_completed") != 0
        or terminal.get("committed_exact_suffix_shard_count") != 16
        or verification.get("failure_code")
        != "raw_exact_suffix_complete_postprocessing_incomplete"
    ):
        raise GlobalDilatedRolloutError(
            "raw-suffix nonresumable terminal conversion failed"
        )
    return verification


def _convert_raw_suffix_resource_failure_to_nonresumable(
    run_dir: Path,
) -> dict[str, Any]:
    """Replace a resumable cap package via a crash-safe conversion intent."""

    _begin_raw_suffix_conversion_intent(run_dir)
    return _complete_raw_suffix_conversion_intent(run_dir)


def _live_failure_capture(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "active_failure_capture.json"
    records = _failure_supersession_records(run_dir)
    if records:
        retirement = (
            run_dir
            / "failure_history"
            / f"retirement-{int(records[-1]['sequence_index']):04d}.json"
        )
        if not retirement.is_file():
            return None
    if not path.is_file():
        return None
    capture = _read_json(path, semantic=True)
    classification_keys = {
        "failure_domain",
        "failure_code",
        "resume_same_frozen_run_authorized",
        "required_next_action",
    }
    if (
        capture.get("schema") != VERSION + "-active-failure-capture"
        or capture.get("failure_generation") != len(records)
        or not isinstance(capture.get("stage"), str)
        or not capture.get("stage")
        or not isinstance(capture.get("exception_type"), str)
        or not capture.get("exception_type")
        or not isinstance(capture.get("message"), str)
        or capture.get("resume_same_frozen_run_authorized") not in {0, 1}
        or not isinstance(capture.get("mandatory_objective_authority"), Mapping)
        or capture.get("mandatory_objective_authority")
        != _mandatory_objective_completion(run_dir)
        or not isinstance(capture.get("captured_at"), str)
        or any(key not in capture for key in classification_keys)
    ):
        raise GlobalDilatedRolloutError("active failure capture authority changed")
    resource = capture["resume_same_frozen_run_authorized"] == 1
    if capture.get("exception_type") == PostprocessingResourceStopError.__name__:
        expected_domain = "resource_budget"
        expected_code = "raw_exact_suffix_complete_postprocessing_incomplete"
        expected_action = (
            "preserve and report the sealed 16-of-16 raw exact suffix; do not "
            "replay postprocessing or optional compute under the exhausted cap"
        )
    else:
        expected_domain = "resource_budget" if resource else "execution_integrity"
        expected_code = (
            "global_rollout_resource_boundary"
            if resource
            else "global_rollout_integrity_or_numerical_failure"
        )
        expected_action = (
            "resume the unchanged frozen run from the last committed exact boundary if the same resource cap admits it"
            if resource
            else "diagnose the sealed input/control/numerical defect; repair it in a new immutable run before resuming scientific interpretation"
        )
    immutable_context = capture.get("immutable_context_file_sha256")
    if (
        capture.get("failure_domain") != expected_domain
        or capture.get("failure_code") != expected_code
        or capture.get("required_next_action") != expected_action
        or (resource and capture.get("exception_type") != ResourceBoundaryError.__name__)
        or (
            capture.get("exception_type") == PostprocessingResourceStopError.__name__
            and capture.get("resume_same_frozen_run_authorized") != 0
        )
        or not isinstance(immutable_context, Mapping)
        or any(
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not (run_dir / relative).is_file()
            or file_fingerprint(run_dir / relative) != digest
            for relative, digest in (
                immutable_context.items() if isinstance(immutable_context, Mapping) else ()
            )
        )
    ):
        raise GlobalDilatedRolloutError(
            "active failure capture typed classification changed"
        )
    failure_path = run_dir / "failure/failure.json"
    if failure_path.is_file():
        failure = _read_json(failure_path, semantic=True)
        if (
            failure.get("active_failure_capture_semantic_sha256")
            != capture["semantic_sha256"]
            or failure.get("failure_generation") != capture["failure_generation"]
            or failure.get("stage") != capture["stage"]
            or failure.get("exception_type") != capture["exception_type"]
            or failure.get("message") != capture["message"]
            or failure.get("failure_domain") != capture["failure_domain"]
            or failure.get("failure_code") != capture["failure_code"]
            or failure.get("mandatory_objective_authority")
            != capture["mandatory_objective_authority"]
        ):
            raise GlobalDilatedRolloutError(
                "active failure capture record binding changed"
            )
    return capture


def _pending_completed_objective_report_reserve_stop(run_dir: Path) -> bool:
    """Recognize the one crash seam after the durable reserve cap crossing."""

    if (
        (run_dir / "completed_objective_resource_stop.json").is_file()
        or _stage_complete(run_dir, "report_verify")
    ):
        return False
    ledger = _resource_ledger(run_dir)
    reserve_events = [
        event
        for event in ledger.get("events", [])
        if event.get("role") == "report_and_final_verification_reserved_charge"
        and event.get("detail")
        == {
            "charge_type": "conservative_frozen_reserve",
            "covers": "reports_final_inventories_reopen_and_terminal_read_only_audit",
        }
    ]
    if not reserve_events:
        return False
    if len(reserve_events) != 1:
        raise GlobalDilatedRolloutError(
            "completed-objective report reserve event multiplicity changed"
        )
    event = reserve_events[0]
    allowed_breaches = {
        "active_seconds_cap",
        "cuda_memory_fraction_cap",
        "persisted_storage_cap",
    }
    event_breaches = event.get("breaches")
    ledger_breaches = ledger.get("breaches")
    if (
        not isinstance(event_breaches, list)
        or not event_breaches
        or any(item not in allowed_breaches for item in event_breaches)
        or event_breaches != ledger_breaches
    ):
        return False
    objective = _mandatory_objective_completion(run_dir)
    return bool(
        event.get("limits_passed") == 0
        and ledger.get("limits_passed") == 0
        and objective["scientific_objective_completed"] == 1
    )


def _pending_completed_objective_resource_packaging(run_dir: Path) -> bool:
    """Route any completed-objective cap stop before resume accounting."""

    if _stage_complete(run_dir, "report_verify"):
        return False
    ledger = _resource_ledger(run_dir)
    if ledger.get("limits_passed") != 0 or not ledger.get("breaches"):
        return False
    objective = _mandatory_objective_completion(run_dir)
    return bool(objective["scientific_objective_completed"] == 1)


def _recover_unmarked_completed_objective_resource_stop(
    run_dir: Path, args: argparse.Namespace
) -> dict[str, Any] | None:
    """Commit only missing evaluation authority after a post-objective cap crash.

    An exact or postprocessing finish event commits before it raises its typed
    resource boundary.  A hard death in the caller before `positive_branch`
    and the evaluate marker must not turn the next invocation into scientific
    replay.  This recovery is intentionally unavailable unless all 16 raw
    shards and every mandatory derived artifact first pass the same deep,
    read-only verifier used by terminal assembly.
    """

    if _stage_complete(run_dir, "evaluate_exact") or _stage_complete(
        run_dir, "report_verify"
    ):
        return None
    ledger = _resource_ledger(run_dir)
    if ledger.get("limits_passed") != 0 or not ledger.get("breaches"):
        return None
    if not _stage_complete(run_dir, "train_select_freeze"):
        raise GlobalDilatedRolloutError(
            "failed resource ledger predates the frozen evaluation authority"
        )
    raw = _mandatory_objective_completion(run_dir)
    if (
        raw["evaluate_exact_marker_complete"] != 0
        or raw["family_summary_complete"] != 1
        or raw["committed_exact_suffix_shard_count"]
        != MANDATORY_SUFFIX_STEPS // 8
    ):
        return None

    # No mutation is permitted above or during these checks.  In particular,
    # recompute the outcome from raw evidence, reopen all mandatory images,
    # and verify the selected/frozen path before repairing administrative
    # terminal records.
    outcome = _verify_mandatory_postprocessing_complete(run_dir, args)
    path_usage = _read_json(run_dir / "path_usage.json", semantic=True)
    positive_path = run_dir / "positive_branch.json"
    positive: dict[str, Any] | None = None
    if positive_path.is_file():
        positive = _read_json(positive_path, semantic=True)
        if (
            positive.get("triggered")
            != int(outcome.get("outcome") == "global_material_improvement")
            or positive.get("attempted") not in {0, 1}
            or positive.get("completed", 0) not in {0, 1}
            or (
                positive.get("completed", 0) == 1
                and positive.get("attempted") != 1
            )
        ):
            raise GlobalDilatedRolloutError(
                "unmarked resource-stop optional status changed"
            )

    if positive is None:
        optional_compute_roles = {
            "positive_branch_forward_128_to_511_attempt",
            "positive_branch_forward_128_to_511_attempt_abandoned_attempt",
            "positive_branch_complete_three_row_exact_attempt",
            "positive_branch_complete_three_row_exact_attempt_abandoned_attempt",
            "optional_positive_postprocessing",
            "optional_positive_postprocessing_abandoned_attempt",
        }
        optional_attempted = int(
            any(
                event.get("role") in optional_compute_roles
                for event in ledger.get("events", [])
            )
            or (run_dir / "positive/anchor-step-0511.npz").is_file()
            or (run_dir / "positive/fused_families").is_dir()
        )
        positive_body = {
            "schema": VERSION + "-positive-branch",
            "schema_version": 1,
            "triggered": int(
                outcome.get("outcome") == "global_material_improvement"
            ),
            "attempted": optional_attempted,
            "completed": 0,
            "mandatory_suffix_objective_preserved": 1,
            "failure_domain": "resource_budget",
            "failure_type": ResourceBoundaryError.__name__,
            "failure_message": (
                "authoritative resource ledger crossed configured cap(s): "
                + ", ".join(str(item) for item in ledger.get("breaches", []))
            ),
            "reason": (
                "a resource boundary committed after mandatory derived evidence "
                "and before the evaluate-stage terminal marker"
            ),
            "required_next_action": (
                "seal the completed mandatory objective and defer optional reconstruction"
            ),
            "complete_path_claim_authorized": 0,
        }
    else:
        positive_body = {
            key: value
            for key, value in positive.items()
            if key != "semantic_sha256"
        }

    # Both writes are deterministic except the final stage timestamp.  A hard
    # death after the first write simply revalidates the same semantic body;
    # a hard death after the atomic marker is handled by the ordinary
    # completed-objective packaging predicate.
    repaired_positive = (
        positive
        if positive is not None
        else _write_semantic(positive_path, positive_body)
    )
    _mark_stage(
        run_dir,
        "evaluate_exact",
        {
            "outcome": outcome["outcome"],
            "fresh_path_id": path_usage["fresh_path_id"],
            "positive_branch_attempted": repaired_positive.get("attempted", 0),
            "packaging_only_recovery_after_resource_boundary": 1,
        },
    )
    recovered = _mandatory_objective_completion(run_dir)
    if recovered["scientific_objective_completed"] != 1:
        raise GlobalDilatedRolloutError(
            "resource-stop evaluation marker did not complete objective authority"
        )
    return {
        "outcome": outcome,
        "positive_branch": repaired_positive,
        "mandatory_objective_authority": recovered,
    }


def _finalize_failure(run_dir: Path, stage: str, exc: Exception) -> dict[str, Any]:
    """Package partial exact evidence without assuming the objective completed."""

    capture = _live_failure_capture(run_dir)
    if capture is None:
        raise GlobalDilatedRolloutError(
            "failure finalization requires an active atomic capture"
        )
    classification = _failure_classification(exc)
    resume_authorized = bool(
        classification["resume_same_frozen_run_authorized"]
    )
    resource_failure = classification["failure_domain"] == "resource_budget"
    failure_domain = str(classification["failure_domain"])
    failure_code = str(classification["failure_code"])
    required_next_action = str(classification["required_next_action"])
    objective = _mandatory_objective_completion(run_dir)
    committed_shards = list(objective["committed_exact_suffix_shards"])
    objective_completed = int(objective["scientific_objective_completed"])
    generation = len(_failure_supersession_records(run_dir))
    if (
        capture.get("failure_generation") != generation
        or capture.get("stage") != stage
        or capture.get("exception_type") != type(exc).__name__
        or capture.get("message") != str(exc)
        or capture.get("failure_domain") != failure_domain
        or capture.get("failure_code") != failure_code
        or capture.get("resume_same_frozen_run_authorized")
        != int(resume_authorized)
        or capture.get("mandatory_objective_authority") != objective
    ):
        raise GlobalDilatedRolloutError(
            "active failure capture differs from finalization authority"
        )
    failure_record = _write_failure_record_from_capture(run_dir, capture)
    failure_evidence = _ensure_failure_evidence_record(run_dir, capture)
    terminal = _write_semantic(
        run_dir / "terminal_failure.json",
        {
            "schema": VERSION + "-terminal-failure",
            "schema_version": 1,
            "failure_generation": generation,
            "active_failure_capture_semantic_sha256": capture["semantic_sha256"],
            "failure_record_semantic_sha256": failure_record["semantic_sha256"],
            "failure_evidence_semantic_sha256": failure_evidence[
                "semantic_sha256"
            ],
            "scientific_objective_completed": objective_completed,
            "failed_stage": stage,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "failure_domain": failure_domain,
            "failure_code": failure_code,
            "committed_exact_suffix_shard_count": len(committed_shards),
            "committed_exact_suffix_shards": committed_shards,
            "mandatory_objective_authority": objective,
            "last_valid_states_present": int(
                failure_evidence.get("last_valid_states_saved") == 1
            ),
            "approximate_primary_substitution_performed": 0,
            "resume_same_frozen_run_authorized": int(resume_authorized),
            "required_next_action": required_next_action,
        },
    )
    active_marker = _write_semantic(
        run_dir / "active_failure_package.json",
        {
            "schema": VERSION + "-active-failure-package",
            "schema_version": 1,
            "failure_generation": generation,
            "active_failure_capture_semantic_sha256": capture["semantic_sha256"],
            "failure_evidence_semantic_sha256": failure_evidence[
                "semantic_sha256"
            ],
            "terminal_failure_semantic_sha256": terminal["semantic_sha256"],
            "resume_same_frozen_run_authorized": int(resume_authorized),
        },
    )
    report = f"""# Global-dilated exact suffix: terminal failure report

Research mode: exploratory. The mandatory objective {('completed before report/resource packaging failed' if objective_completed else 'did not complete')}.

Failure stage: `{stage}`. Domain/code: `{failure_domain}` / `{failure_code}`. Error: `{type(exc).__name__}: {exc}`.

Committed certified-exact suffix shards preserved: {len(committed_shards)} / 16. Last valid states and a contact sheet were saved when available under `failure/`. No approximate trajectory was substituted. {('The completed mandatory suffix remains scientifically authoritative at its tested scope; this terminal is a resource/reporting failure.' if objective_completed else 'No scientific success/negative mechanism claim is made from an incomplete suffix.')}

Required next action: {required_next_action}.
"""
    _atomic_text(run_dir / "REPORT.md", report)
    handoff = f"""# Global-dilated Jacobi/RB partial-run handoff

Primary mode: {('engineering/resource failure' if resource_failure else 'engineering/integrity failure')} during an exploratory objective-bearing run.

Program objective: a DDPM-like MNIST generator based on the Eulerian approximation.

Exact result: the mandatory five-row suffix {('completed' if objective_completed else 'is incomplete')} ({len(committed_shards)}/16 committed exact shards). {('Its sealed metrics/images remain the objective-bearing result, while final reporting hit the failure below.' if objective_completed else 'This establishes no final paired effect.') } The concrete failure was `{type(exc).__name__}: {exc}`.

Proxy-only patches since the last objective-bearing experiment: 0

Decision for the next agent: {required_next_action}. Do not substitute an approximate primary or tune after opening fresh evidence.

Evidence map: `terminal_failure.json`, `failure/`, `fresh_forward/`, and any committed `suffix/fused_families/` shards. Confirmation evidence remains outside this exploratory failure.
"""
    _atomic_text(run_dir / "HANDOFF.md", handoff)
    rows = _manifest_rows(run_dir)
    manifest = _write_semantic(
        run_dir / "artifact_manifest.json",
        {
            "schema": VERSION + "-artifact-manifest",
            "schema_version": 1,
            "terminal_failure": 1,
            "artifact_count": len(rows),
            "excluded_self_referential_paths": sorted(_FINAL_EXCLUDED),
            "artifacts": rows,
        },
    )
    checksum_paths = [
        path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and path.name not in {"SHA256SUMS.txt", "verification.json"}
        and path.relative_to(run_dir).as_posix() != _RAW_SUFFIX_CONVERSION_INTENT
        and path.suffix != ".tmp"
    ]
    _atomic_text(
        run_dir / "SHA256SUMS.txt",
        "".join(
            f"{file_fingerprint(path)}  {path.relative_to(run_dir).as_posix()}\n"
            for path in checksum_paths
        ),
    )
    verification = _write_semantic(
        run_dir / "verification.json",
        {
            "schema": VERSION + "-verification",
            "schema_version": 1,
            "passed": 1,
            "scientific_objective_completed": objective_completed,
            "terminal_failure_semantic_sha256": terminal["semantic_sha256"],
            "active_failure_package_semantic_sha256": active_marker["semantic_sha256"],
            "active_failure_capture_semantic_sha256": capture["semantic_sha256"],
            "failure_evidence_semantic_sha256": failure_evidence[
                "semantic_sha256"
            ],
            "failure_generation": generation,
            "artifact_manifest_semantic_sha256": manifest["semantic_sha256"],
            "committed_exact_suffix_shard_count": len(committed_shards),
            "partial_failure_evidence_packaged": 1,
            "failure_domain": failure_domain,
            "failure_code": failure_code,
        },
    )
    # Strictly reopen the failure inventories without invoking objective-only
    # metric/image verification.
    for row in _read_json(run_dir / "artifact_manifest.json", semantic=True)["artifacts"]:
        path = run_dir / row["path"]
        if file_fingerprint(path) != row["sha256"] or path.stat().st_size != int(row["size"]):
            raise GlobalDilatedRolloutError("terminal failure manifest verification failed")
    for line in (run_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if file_fingerprint(run_dir / relative) != digest:
            raise GlobalDilatedRolloutError("terminal failure checksum verification failed")
    _verify_completed_failure_read_only(run_dir)
    return verification


def _verify_completed_failure_read_only(run_dir: Path) -> dict[str, Any]:
    """Verify a terminal failure package without rewriting any artifact."""

    terminal = _read_json(run_dir / "terminal_failure.json", semantic=True)
    active = _read_json(run_dir / "active_failure_package.json", semantic=True)
    capture = _read_json(run_dir / "active_failure_capture.json", semantic=True)
    live_capture = _live_failure_capture(run_dir)
    failure = _read_json(run_dir / "failure/failure.json", semantic=True)
    evidence = _read_json(run_dir / "failure/failure_evidence.json", semantic=True)
    verified_evidence = _ensure_failure_evidence_record(run_dir, capture)
    verification = _read_json(run_dir / "verification.json", semantic=True)
    manifest = _read_json(run_dir / "artifact_manifest.json", semantic=True)
    if (
        terminal.get("scientific_objective_completed") not in {0, 1}
        or live_capture != capture
        or verified_evidence != evidence
        or verification.get("passed") != 1
        or verification.get("scientific_objective_completed")
        != terminal.get("scientific_objective_completed")
        or verification.get("terminal_failure_semantic_sha256")
        != terminal.get("semantic_sha256")
        or verification.get("active_failure_package_semantic_sha256")
        != active.get("semantic_sha256")
        or verification.get("active_failure_capture_semantic_sha256")
        != capture.get("semantic_sha256")
        or verification.get("failure_generation")
        != terminal.get("failure_generation")
        or active.get("schema") != VERSION + "-active-failure-package"
        or active.get("failure_generation") != terminal.get("failure_generation")
        or capture.get("failure_generation") != terminal.get("failure_generation")
        or terminal.get("active_failure_capture_semantic_sha256")
        != capture.get("semantic_sha256")
        or active.get("active_failure_capture_semantic_sha256")
        != capture.get("semantic_sha256")
        or failure.get("active_failure_capture_semantic_sha256")
        != capture.get("semantic_sha256")
        or terminal.get("failure_record_semantic_sha256")
        != failure.get("semantic_sha256")
        or terminal.get("failure_evidence_semantic_sha256")
        != evidence.get("semantic_sha256")
        or active.get("failure_evidence_semantic_sha256")
        != evidence.get("semantic_sha256")
        or verification.get("failure_evidence_semantic_sha256")
        != evidence.get("semantic_sha256")
        or evidence.get("active_failure_capture_semantic_sha256")
        != capture.get("semantic_sha256")
        or capture.get("stage") != terminal.get("failed_stage")
        or capture.get("exception_type") != terminal.get("exception_type")
        or capture.get("message") != terminal.get("message")
        or capture.get("failure_domain") != terminal.get("failure_domain")
        or capture.get("failure_code") != terminal.get("failure_code")
        or capture.get("resume_same_frozen_run_authorized")
        != terminal.get("resume_same_frozen_run_authorized")
        or capture.get("mandatory_objective_authority")
        != terminal.get("mandatory_objective_authority")
        or active.get("terminal_failure_semantic_sha256")
        != terminal.get("semantic_sha256")
        or verification.get("artifact_manifest_semantic_sha256")
        != manifest.get("semantic_sha256")
        or manifest.get("terminal_failure") != 1
    ):
        raise GlobalDilatedRolloutError("terminal failure package authority changed")
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or manifest.get("artifact_count") != len(rows):
        raise GlobalDilatedRolloutError("terminal failure manifest schema changed")
    expected_manifest_paths = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.relative_to(run_dir).as_posix() not in _FINAL_EXCLUDED
        and path.relative_to(run_dir).as_posix() != _RAW_SUFFIX_CONVERSION_INTENT
        and path.suffix != ".tmp"
    }
    if {str(row.get("path")) for row in rows} != expected_manifest_paths:
        raise GlobalDilatedRolloutError("terminal failure manifest path set changed")
    for row in rows:
        path = run_dir / str(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row.get("size", -1))
            or file_fingerprint(path) != row.get("sha256")
        ):
            raise GlobalDilatedRolloutError("terminal failure registered artifact changed")
    checksum_rows: dict[str, str] = {}
    for line in (run_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise GlobalDilatedRolloutError("terminal failure checksum row changed") from exc
        if (
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            or relative in checksum_rows
            or file_fingerprint(run_dir / relative) != digest
        ):
            raise GlobalDilatedRolloutError("terminal failure checksum changed")
        checksum_rows[relative] = digest
    expected_checksum_paths = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.name not in {"SHA256SUMS.txt", "verification.json"}
        and path.relative_to(run_dir).as_posix() != _RAW_SUFFIX_CONVERSION_INTENT
        and path.suffix != ".tmp"
    }
    if set(checksum_rows) != expected_checksum_paths:
        raise GlobalDilatedRolloutError("terminal failure checksum path set changed")
    return verification


_FAILURE_PACKAGE_PATHS = (
    "terminal_failure.json",
    "active_failure_package.json",
    "active_failure_capture.json",
    "verification.json",
    "artifact_manifest.json",
    "SHA256SUMS.txt",
    "REPORT.md",
    "HANDOFF.md",
    "failure/failure_evidence.json",
    "failure/failure.json",
)


def _failure_supersession_records(run_dir: Path) -> list[dict[str, Any]]:
    root = run_dir / "failure_history"
    records: list[dict[str, Any]] = []
    previous: str | None = None
    for index, path in enumerate(sorted(root.glob("supersession-*.json")) if root.is_dir() else []):
        record = _read_json(path, semantic=True)
        files = record.get("snapshot_utf8_files")
        if (
            path.name != f"supersession-{index:04d}.json"
            or record.get("schema") != VERSION + "-failure-supersession"
            or record.get("sequence_index") != index
            or record.get("failure_generation") != index
            or record.get("previous_supersession_semantic_sha256") != previous
            or not isinstance(files, Mapping)
            or set(files) != set(_FAILURE_PACKAGE_PATHS)
        ):
            raise GlobalDilatedRolloutError("failure supersession chain changed")
        for relative, snapshot in files.items():
            if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("utf8_text"), str):
                raise GlobalDilatedRolloutError("failure supersession snapshot changed")
            encoded = snapshot["utf8_text"].encode("utf-8")
            if (
                snapshot.get("size") != len(encoded)
                or snapshot.get("sha256") != hashlib.sha256(encoded).hexdigest()
            ):
                raise GlobalDilatedRolloutError("failure supersession byte binding changed")
        terminal_snapshot = json.loads(files["terminal_failure.json"]["utf8_text"])
        active_snapshot = json.loads(files["active_failure_package.json"]["utf8_text"])
        capture_snapshot = json.loads(files["active_failure_capture.json"]["utf8_text"])
        evidence_snapshot = json.loads(files["failure/failure_evidence.json"]["utf8_text"])
        failure_snapshot = json.loads(files["failure/failure.json"]["utf8_text"])
        if (
            terminal_snapshot.get("semantic_sha256")
            != record.get("terminal_failure_semantic_sha256")
            or terminal_snapshot.get("failure_generation")
            != record.get("failure_generation")
            or active_snapshot.get("failure_generation")
            != record.get("failure_generation")
            or active_snapshot.get("terminal_failure_semantic_sha256")
            != terminal_snapshot.get("semantic_sha256")
            or capture_snapshot.get("failure_generation")
            != record.get("failure_generation")
            or capture_snapshot.get("semantic_sha256")
            != record.get("active_failure_capture_semantic_sha256")
            or terminal_snapshot.get("active_failure_capture_semantic_sha256")
            != capture_snapshot.get("semantic_sha256")
            or active_snapshot.get("active_failure_capture_semantic_sha256")
            != capture_snapshot.get("semantic_sha256")
            or failure_snapshot.get("active_failure_capture_semantic_sha256")
            != capture_snapshot.get("semantic_sha256")
            or evidence_snapshot.get("active_failure_capture_semantic_sha256")
            != capture_snapshot.get("semantic_sha256")
            or terminal_snapshot.get("failure_evidence_semantic_sha256")
            != evidence_snapshot.get("semantic_sha256")
            or evidence_snapshot.get("semantic_sha256")
            != record.get("failure_evidence_semantic_sha256")
            or active_snapshot.get("failure_evidence_semantic_sha256")
            != evidence_snapshot.get("semantic_sha256")
        ):
            raise GlobalDilatedRolloutError("failure supersession terminal binding changed")
        previous = str(record["semantic_sha256"])
        records.append(record)
    return records


def _live_terminal_failure(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "terminal_failure.json"
    active_path = run_dir / "active_failure_package.json"
    records = _failure_supersession_records(run_dir)
    if records:
        latest = records[-1]
        retirement = (
            run_dir
            / "failure_history"
            / f"retirement-{int(latest['sequence_index']):04d}.json"
        )
        if not retirement.is_file():
            # The immutable supersession is already the commit authority.  A
            # later cleanup pass may have removed any subset of live files.
            return None
    if not path.is_file() and not active_path.is_file():
        return None
    if not path.is_file():
        raise GlobalDilatedRolloutError("active failure package marker is incomplete")
    terminal = _read_json(path, semantic=True)
    capture = _live_failure_capture(run_dir)
    if (
        capture is None
        or terminal.get("active_failure_capture_semantic_sha256")
        != capture.get("semantic_sha256")
    ):
        raise GlobalDilatedRolloutError(
            "terminal failure lacks its active capture authority"
        )
    expected_generation = len(records)
    if not active_path.is_file():
        if (
            (run_dir / "verification.json").is_file()
            or terminal.get("failure_generation") != expected_generation
        ):
            raise GlobalDilatedRolloutError("active failure package marker is incomplete")
        # Crash after terminal commit and before marker commit.  Packaging-only
        # recovery deterministically recreates the marker; compatibility stays
        # mutation-free here.
        return terminal
    active = _read_json(active_path, semantic=True)
    if (
        active.get("schema") != VERSION + "-active-failure-package"
        or active.get("failure_generation") != expected_generation
        or terminal.get("failure_generation") != expected_generation
        or active.get("terminal_failure_semantic_sha256")
        != terminal.get("semantic_sha256")
        or active.get("active_failure_capture_semantic_sha256")
        != capture.get("semantic_sha256")
    ):
        raise GlobalDilatedRolloutError("active failure generation authority changed")
    return terminal


def _failure_verification_matches_terminal(
    run_dir: Path, terminal: Mapping[str, Any]
) -> bool:
    path = run_dir / "verification.json"
    if not path.is_file():
        return False
    verification = _read_json(path, semantic=True)
    return (
        verification.get("terminal_failure_semantic_sha256")
        == terminal.get("semantic_sha256")
    )


def _complete_pending_failure_retirement(run_dir: Path) -> None:
    records = _failure_supersession_records(run_dir)
    if not records:
        return
    record = records[-1]
    index = int(record["sequence_index"])
    retirement_path = run_dir / "failure_history" / f"retirement-{index:04d}.json"
    if retirement_path.is_file():
        retirement = _read_json(retirement_path, semantic=True)
        if (
            retirement.get("supersession_semantic_sha256")
            != record["semantic_sha256"]
            or retirement.get("active_failure_package_retired") != 1
            or retirement.get("active_failure_capture_retired") != 1
        ):
            raise GlobalDilatedRolloutError("failure retirement authority changed")
        return
    terminal_path = run_dir / "terminal_failure.json"
    if terminal_path.is_file() and file_fingerprint(terminal_path) != record[
        "snapshot_utf8_files"
    ]["terminal_failure.json"]["sha256"]:
        raise GlobalDilatedRolloutError("new live failure appeared before retirement completed")
    # Every live package byte is already authenticated in the append-only
    # supersession snapshot.  Retire the old failure record too: leaving it in
    # the live namespace creates an ambiguity if the next generation commits
    # its minimal capture marker and dies before replacing failure/failure.json.
    # The retirement marker remains the final commit, so a death during these
    # removals is recovered from the already-durable supersession snapshot.
    for relative in _FAILURE_PACKAGE_PATHS:
        (run_dir / relative).unlink(missing_ok=True)
    (run_dir / "stages/report_verify.json").unlink(missing_ok=True)
    _write_semantic(
        retirement_path,
        {
            "schema": VERSION + "-failure-retirement",
            "schema_version": 1,
            "sequence_index": index,
            "failure_generation": int(record["failure_generation"]),
            "supersession_semantic_sha256": record["semantic_sha256"],
            "active_failure_package_retired": 1,
            "active_failure_capture_retired": 1,
            "scientific_stage_reentered": 0,
        },
    )


def _supersede_authorized_failure(run_dir: Path) -> dict[str, Any]:
    terminal = _live_terminal_failure(run_dir)
    if terminal is None or terminal.get("resume_same_frozen_run_authorized") != 1:
        raise GlobalDilatedRolloutError("no authorized live failure package to supersede")
    if not _failure_verification_matches_terminal(run_dir, terminal):
        raise GlobalDilatedRolloutError("authorized failure package is incomplete")
    _verify_completed_failure_read_only(run_dir)
    records = _failure_supersession_records(run_dir)
    snapshots: dict[str, Any] = {}
    for relative in _FAILURE_PACKAGE_PATHS:
        path = run_dir / relative
        text = path.read_text(encoding="utf-8")
        encoded = text.encode("utf-8")
        snapshots[relative] = {
            "utf8_text": text,
            "size": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    index = len(records)
    record = _write_semantic(
        run_dir / "failure_history" / f"supersession-{index:04d}.json",
        {
            "schema": VERSION + "-failure-supersession",
            "schema_version": 1,
            "sequence_index": index,
            "failure_generation": int(terminal["failure_generation"]),
            "previous_supersession_semantic_sha256": (
                None if not records else records[-1]["semantic_sha256"]
            ),
            "terminal_failure_semantic_sha256": terminal["semantic_sha256"],
            "active_failure_capture_semantic_sha256": terminal[
                "active_failure_capture_semantic_sha256"
            ],
            "failure_evidence_semantic_sha256": terminal[
                "failure_evidence_semantic_sha256"
            ],
            "resume_same_frozen_run_authorized": 1,
            "snapshot_utf8_files": snapshots,
            "snapshot_scope": (
                "failure terminal, compact reports, manifest, checksums, verification, "
                "and failure record; manifest/checksums bind the contemporaneous run"
            ),
        },
    )
    _complete_pending_failure_retirement(run_dir)
    return record


def _exception_from_failure_authority(authority: Mapping[str, Any]) -> Exception:
    if (
        authority.get("failure_code")
        == "raw_exact_suffix_complete_postprocessing_incomplete"
    ):
        if (
            authority.get("exception_type")
            != PostprocessingResourceStopError.__name__
            or authority.get("resume_same_frozen_run_authorized") != 0
        ):
            raise GlobalDilatedRolloutError(
                "postprocessing resource-stop authority changed"
            )
        return PostprocessingResourceStopError(str(authority["message"]))
    if authority.get("failure_code") == "global_rollout_resource_boundary":
        if (
            authority.get("exception_type") != ResourceBoundaryError.__name__
            or authority.get("resume_same_frozen_run_authorized") != 1
        ):
            raise GlobalDilatedRolloutError(
                "resource failure authority lacks typed boundary classification"
            )
        return ResourceBoundaryError(str(authority["message"]))
    if (
        authority.get("failure_code")
        != "global_rollout_integrity_or_numerical_failure"
        or authority.get("resume_same_frozen_run_authorized") != 0
    ):
        raise GlobalDilatedRolloutError(
            "integrity failure authority has invalid resume classification"
        )
    recovered_type = type(
        str(authority["exception_type"]), (GlobalDilatedRolloutError,), {}
    )
    return recovered_type(str(authority["message"]))


def _recover_captured_failure_package(run_dir: Path) -> dict[str, Any]:
    """Create the terminal package from the atomic pre-terminal capture only."""

    if (run_dir / "terminal_failure.json").is_file():
        raise GlobalDilatedRolloutError(
            "captured-only recovery cannot replace an existing terminal"
        )
    capture = _live_failure_capture(run_dir)
    if capture is None:
        raise GlobalDilatedRolloutError("no live captured failure requires packaging")
    _write_failure_record_from_capture(run_dir, capture)
    exc = _exception_from_failure_authority(capture)
    return _finalize_failure(run_dir, str(capture["stage"]), exc)


def _recover_incomplete_failure_package(run_dir: Path) -> dict[str, Any]:
    """Finish packaging only; never re-enter a scientific stage."""

    terminal = _read_json(run_dir / "terminal_failure.json", semantic=True)
    if _failure_verification_matches_terminal(run_dir, terminal):
        return _verify_completed_failure_read_only(run_dir)
    failure = _read_json(run_dir / "failure/failure.json", semantic=True)
    capture = _live_failure_capture(run_dir)
    objective = _mandatory_objective_completion(run_dir)
    if (
        capture is None
        or terminal.get("active_failure_capture_semantic_sha256")
        != capture.get("semantic_sha256")
        or failure.get("active_failure_capture_semantic_sha256")
        != capture.get("semantic_sha256")
        or terminal.get("scientific_objective_completed")
        != objective["scientific_objective_completed"]
        or terminal.get("mandatory_objective_authority") != objective
        or terminal.get("failed_stage") != failure.get("stage")
        or terminal.get("exception_type") != failure.get("exception_type")
        or terminal.get("message") != failure.get("message")
    ):
        raise GlobalDilatedRolloutError("incomplete failure package authority changed")
    exc = _exception_from_failure_authority(terminal)
    return _finalize_failure(run_dir, str(terminal["failed_stage"]), exc)


def _write_run_manifest(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    repository_root = Path(args.repository_root).resolve()
    entry_points = (
        repository_root / "mnist/d0_jacobi_rb_global_dilated.py",
        repository_root / "mnist/d0_jacobi_rb_tangent_rollout.py",
        repository_root / "mnist/d0_jacobi_rb_tangent_fused.py",
        repository_root / "mnist/diag_d0_jacobi_rb_global_dilated_rollout.py",
    )
    source_paths = v3_transitive_source_paths(entry_points)
    closure = {
        path.relative_to(repository_root).as_posix(): {
            "sha256": file_fingerprint(path),
            "size": int(path.stat().st_size),
        }
        for path in source_paths
        if path.is_file()
    }
    body = {
        "schema": VERSION + "-run-manifest",
        "schema_version": 1,
        "run_dir": str(run_dir.resolve()),
        "created_at": _utc_now(),
        "source_revision": _source_revision(repository_root),
        "source_closure": closure,
        "source_closure_sha256": semantic_sha256(closure),
        "scientific_config_file_sha256": file_fingerprint(run_dir / "scientific_config.json"),
        "exact_command_file_sha256": file_fingerprint(run_dir / "exact_command.txt"),
        "research_mode": "exploratory",
    }
    path = run_dir / "run_manifest.json"
    if path.is_file():
        existing = _read_json(path, semantic=True)
        # Creation time and dirty worktree summary are immutable once opened.
        comparable = dict(existing)
        comparable.pop("semantic_sha256", None)
        expected = dict(body)
        expected["created_at"] = comparable.get("created_at")
        expected["source_revision"] = comparable.get("source_revision")
        if comparable != expected:
            raise GlobalDilatedRolloutError("run source closure changed on resume")
        return existing
    return _write_semantic(path, body)


def _current_source_closure(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    repository_root = Path(args.repository_root).resolve()
    entry_points = (
        repository_root / "mnist/d0_jacobi_rb_global_dilated.py",
        repository_root / "mnist/d0_jacobi_rb_tangent_rollout.py",
        repository_root / "mnist/d0_jacobi_rb_tangent_fused.py",
        repository_root / "mnist/diag_d0_jacobi_rb_global_dilated_rollout.py",
    )
    paths = v3_transitive_source_paths(entry_points)
    closure = {
        path.relative_to(repository_root).as_posix(): {
            "sha256": file_fingerprint(path),
            "size": int(path.stat().st_size),
        }
        for path in paths
    }
    return closure, semantic_sha256(closure)


def _verify_bound_paths(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    bindings = _read_json(run_dir / "input_bindings.json", semantic=True)
    current = _measure_input_bindings(args)
    if current != bindings:
        changed = sorted(
            key
            for key in set(bindings) | set(current)
            if bindings.get(key) != current.get(key)
        )
        raise GlobalDilatedRolloutError(
            "resumed immutable input measurement differs from sealed bindings: "
            + ", ".join(changed)
        )
    return bindings


def _verify_resume_compatibility(run_dir: Path, args: argparse.Namespace) -> None:
    """Read-only fail-closed verification before any resume mutation."""

    if args.resume_run_dir is None:
        raise GlobalDilatedRolloutError("existing run requires --resume-run-dir")
    for relative in ("scientific_config.json", "exact_command.txt", "run_manifest.json"):
        if not (run_dir / relative).is_file():
            raise GlobalDilatedRolloutError(f"resume run omits immutable {relative}")
    terminal_failure = _live_terminal_failure(run_dir)
    if terminal_failure is not None:
        if _failure_verification_matches_terminal(run_dir, terminal_failure):
            _verify_completed_failure_read_only(run_dir)
        if (
            _failure_verification_matches_terminal(run_dir, terminal_failure)
            and terminal_failure.get("resume_same_frozen_run_authorized") != 1
        ):
            raise GlobalDilatedRolloutError(
                "terminal failure does not authorize resuming the same frozen run"
            )
    expected_config = _scientific_config(args)
    if _read_json(run_dir / "scientific_config.json", semantic=True) != expected_config:
        raise GlobalDilatedRolloutError("scientific configuration changed on resume")
    if (run_dir / "input_bindings.json").is_file():
        _verify_bound_paths(run_dir, args)
    manifest = _read_json(run_dir / "run_manifest.json", semantic=True)
    closure, closure_hash = _current_source_closure(args)
    if manifest.get("source_closure") != closure or manifest.get("source_closure_sha256") != closure_hash:
        raise GlobalDilatedRolloutError("transitive runtime source closure changed on resume")
    if manifest.get("exact_command_file_sha256") != file_fingerprint(run_dir / "exact_command.txt"):
        raise GlobalDilatedRolloutError("original exact command changed on resume")

    # A completed marker is an authority claim, so verify its bound artifacts
    # here, still before the resume-attempt ledger event mutates the run.  Stage
    # markers must form one canonical prefix.
    missing_predecessor = False
    for stage in STAGES:
        marker_path = run_dir / "stages" / f"{stage}.json"
        pending_retirement = any(
            not (
                run_dir
                / "failure_history"
                / f"retirement-{int(record['sequence_index']):04d}.json"
            ).is_file()
            for record in _failure_supersession_records(run_dir)
        )
        if stage == "report_verify" and pending_retirement:
            continue
        if not marker_path.is_file():
            missing_predecessor = True
            continue
        if missing_predecessor:
            raise GlobalDilatedRolloutError("completed stage markers are not a canonical prefix")
        if not _stage_complete(run_dir, stage):
            raise GlobalDilatedRolloutError(f"completed {stage} marker is not passing")
        _verify_completed_stage(run_dir, args, stage)


def _run_mandatory_objective_postprocessing(
    run_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """Build all mandatory derived evidence under durable CPU/IO accounting."""

    journal_relative = "suffix/active-mandatory-objective-postprocessing.json"
    role = "mandatory_objective_postprocessing"
    journal, started = _begin_durable_attempt(
        run_dir,
        journal_relative=journal_relative,
        role=role,
        detail={
            "operations": [
                "aggregate_exact_shards",
                "metrics_and_images",
                "classify_outcome",
            ]
        },
    )
    try:
        _aggregate_existing_shards(run_dir)
        _compute_metrics_and_images(run_dir, args)
        outcome = _classify_outcome(run_dir)
        _finish_durable_attempt(
            run_dir,
            journal_relative=journal_relative,
            role=role,
            journal=journal,
            elapsed_seconds=time.perf_counter() - started + 5.0,
            failed=False,
            detail={"cuda_memory_measured": 1, "device": "cpu"},
        )
    except Exception:
        if (run_dir / journal_relative).is_file():
            try:
                _finish_durable_attempt(
                    run_dir,
                    journal_relative=journal_relative,
                    role=role,
                    journal=journal,
                    elapsed_seconds=time.perf_counter() - started + 5.0,
                    failed=True,
                    detail={
                        "cuda_memory_measured": 1,
                        "device": "cpu",
                        "caught_exception": 1,
                    },
                )
            except Exception:
                pass
        raise
    return outcome


def _verify_mandatory_postprocessing_complete(
    run_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """Read-only proof that every mandatory derived artifact fully committed."""

    _verify_scientific_evidence_read_only(run_dir, args)
    aggregate = _load_npz(run_dir / "suffix/trajectory_shard_boundaries.npz")
    summary = _read_json(run_dir / "suffix/summary.json", semantic=True)
    freeze = _read_json(run_dir / "evaluation_freeze.json", semantic=True)
    _verify_all_generated_images_read_only(
        run_dir,
        states=aggregate["states"],
        summary=summary,
        scale=FixedRenderingScale(**freeze["rendering_scale"]),
    )
    outcome = _read_json(run_dir / "outcome.json", semantic=True)
    if outcome != _semantic(_outcome_body(run_dir)):
        raise GlobalDilatedRolloutError(
            "mandatory postprocessing outcome authority changed"
        )
    return outcome


def _run_stage(run_dir: Path, args: argparse.Namespace, stage: str) -> None:
    _require_stage(run_dir, args, stage)
    if stage == "prepare":
        with _timed_resource(run_dir, "prepare_parent_and_role_integrity"):
            bindings = _verify_inputs_and_roles(run_dir, args)
            manifest = _write_run_manifest(run_dir, args)
        _mark_stage(
            run_dir,
            stage,
            {
                "input_bindings_sha256": bindings["semantic_sha256"],
                "run_manifest_sha256": manifest["semantic_sha256"],
            },
        )
        return
    if stage == "controls":
        with _timed_resource(run_dir, "theory_to_code_control"):
            theory = _run_theory_to_code_control(run_dir, args)
        preflight = _run_five_row_exact_smoke(run_dir, args)
        _mark_stage(
            run_dir,
            stage,
            {
                "theory_to_code_sha256": theory["semantic_sha256"],
                "preflight_controls_sha256": preflight["semantic_sha256"],
            },
        )
        return
    if stage == "train_select_freeze":
        selection = _train_and_select_global(run_dir, args)
        freeze = _seal_evaluation(run_dir, args)
        _mark_stage(
            run_dir,
            stage,
            {
                "selection_sha256": selection["semantic_sha256"],
                "evaluation_freeze_sha256": freeze["semantic_sha256"],
                "fresh_path_allocated": 0,
            },
        )
        return
    if stage == "evaluate_exact":
        reconcile_stop: ResourceBoundaryError | None = None
        try:
            _reconcile_durable_attempt_journal(
                run_dir,
                journal_relative=(
                    "suffix/active-mandatory-objective-postprocessing.json"
                ),
                role="mandatory_objective_postprocessing",
            )
        except ResourceBoundaryError as exc:
            reconcile_stop = exc
        if reconcile_stop is not None:
            try:
                outcome = _verify_mandatory_postprocessing_complete(run_dir, args)
            except Exception as verification_exc:
                raw = _mandatory_objective_completion(run_dir)
                _write_semantic(
                    run_dir / "postprocessing_resource_stop.json",
                    {
                        "schema": VERSION + "-postprocessing-resource-stop",
                        "schema_version": 1,
                        "raw_exact_suffix_complete": int(
                            raw["family_summary_complete"] == 1
                            and raw["committed_exact_suffix_shard_count"] == 16
                        ),
                        "raw_exact_suffix_authority": raw,
                        "mandatory_postprocessing_complete": 0,
                        "optional_branch_attempted": 0,
                        "resource_boundary_message": str(reconcile_stop),
                        "derived_verification_failure_type": type(
                            verification_exc
                        ).__name__,
                        "derived_verification_failure_message": str(
                            verification_exc
                        ),
                        "resume_same_frozen_run_authorized": 0,
                        "required_next_action": (
                            "preserve the 16-of-16 raw exact suffix; do not replay "
                            "postprocessing or optional compute under the exhausted cap"
                        ),
                    },
                )
                raise PostprocessingResourceStopError(
                    "raw exact suffix complete but mandatory postprocessing is "
                    "incomplete after a durably reconciled resource cap crossing"
                ) from verification_exc
            positive = _write_semantic(
                run_dir / "positive_branch.json",
                {
                    "schema": VERSION + "-positive-branch",
                    "schema_version": 1,
                    "triggered": int(
                        outcome.get("outcome") == "global_material_improvement"
                    ),
                    "attempted": 0,
                    "completed": 0,
                    "mandatory_suffix_objective_preserved": 1,
                    "failure_domain": "resource_budget",
                    "failure_type": ResourceBoundaryError.__name__,
                    "failure_message": str(reconcile_stop),
                    "reason": (
                        "reconciled postprocessing attempt crossed the cap after "
                        "all mandatory derived evidence committed"
                    ),
                    "required_next_action": (
                        "seal the completed mandatory objective and do not launch "
                        "optional full-path compute"
                    ),
                    "complete_path_claim_authorized": 0,
                },
            )
            path_usage = _read_json(run_dir / "path_usage.json", semantic=True)
            _mark_stage(
                run_dir,
                stage,
                {
                    "outcome": outcome["outcome"],
                    "fresh_path_id": path_usage["fresh_path_id"],
                    "positive_branch_attempted": positive.get("attempted", 0),
                },
            )
            return
        path_usage = _allocate_fresh_path(run_dir, args)
        _run_forward_to_127(run_dir, args, path_usage)
        _run_exact_suffix(run_dir, args)
        postprocess_resource_stop: ResourceBoundaryError | None = None
        try:
            outcome = _run_mandatory_objective_postprocessing(run_dir, args)
        except ResourceBoundaryError as exc:
            # The derived mandatory evidence and outcome commit precede the
            # finish event.  Preserve that completed objective, skip all
            # optional compute, and let report finalization seal the resource
            # stop truthfully.
            outcome_path = run_dir / "outcome.json"
            if not outcome_path.is_file():
                raise
            outcome = _read_json(outcome_path, semantic=True)
            postprocess_resource_stop = exc
        if postprocess_resource_stop is None:
            positive = _maybe_run_positive_complete_path(run_dir, args)
        else:
            positive = _write_semantic(
                run_dir / "positive_branch.json",
                {
                    "schema": VERSION + "-positive-branch",
                    "schema_version": 1,
                    "triggered": int(
                        outcome.get("outcome") == "global_material_improvement"
                    ),
                    "attempted": 0,
                    "completed": 0,
                    "mandatory_suffix_objective_preserved": 1,
                    "failure_domain": "resource_budget",
                    "failure_type": ResourceBoundaryError.__name__,
                    "failure_message": str(postprocess_resource_stop),
                    "reason": (
                        "mandatory postprocessing completed but its durable "
                        "resource boundary did not admit optional full-path compute"
                    ),
                    "required_next_action": (
                        "seal the completed mandatory objective and do not launch "
                        "optional full-path compute"
                    ),
                    "complete_path_claim_authorized": 0,
                },
            )
        _mark_stage(
            run_dir,
            stage,
            {
                "outcome": outcome["outcome"],
                "fresh_path_id": path_usage["fresh_path_id"],
                "positive_branch_attempted": positive.get("attempted", 0),
            },
        )
        return
    if stage == "report_verify":
        if _stage_complete(run_dir, "report_verify"):
            _verify_completed_report_read_only(run_dir, args)
            return
        _finalize_and_verify(run_dir, args)
        return
    raise GlobalDilatedRolloutError(f"unknown stage: {stage}")


def _resolve_paths(args: argparse.Namespace) -> None:
    repository_root = Path(args.repository_root).resolve()
    args.repository_root = repository_root

    def absolute(value: str | Path) -> Path:
        path = Path(value)
        return (repository_root / path).resolve() if not path.is_absolute() else path.resolve()

    args.training_parent = absolute(args.training_parent)
    args.v4_run_dir = absolute(args.v4_run_dir)
    args.source_run_dir = absolute(args.source_run_dir)
    args.runs_root = absolute(args.runs_root)
    if args.resume_run_dir is not None:
        args.resume_run_dir = absolute(args.resume_run_dir)


def _make_run_dir(args: argparse.Namespace) -> Path:
    if args.resume_run_dir is not None:
        run_dir = Path(args.resume_run_dir).resolve()
        if not run_dir.is_dir():
            raise GlobalDilatedRolloutError("resume run directory is absent")
        return run_dir
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.runs_root) / f"{timestamp}_{args.run_name}"
    if run_dir.exists():
        raise GlobalDilatedRolloutError("new run directory already exists")
    return run_dir.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the global-dilated Jacobi/RB controller and run a fresh exact five-row suffix."
    )
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--runs-root", type=Path, default=Path("runs") / RUN_SCHEMA)
    parser.add_argument("--run-name", default="production-global-dilated-exact-five-row")
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--training-parent", type=Path, default=DEFAULT_TRAINING_PARENT)
    parser.add_argument("--v4-run-dir", type=Path, default=DEFAULT_V4_RUN)
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=DEFAULT_V4_RUN / "input_bindings",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _resolve_paths(args)
    run_dir = _make_run_dir(args)
    resume_started = time.perf_counter()
    was_resume = run_dir.exists()
    try:
        _initialize_run(run_dir, args)
    except Exception as exc:
        # A rejected resume must remain byte-identical.  A partially-created
        # fresh run may be packaged because this process owns its new bytes.
        if not was_resume and run_dir.is_dir():
            try:
                _capture_failure(run_dir, "initialization", exc)
                _finalize_failure(run_dir, "initialization", exc)
            except Exception as final_exc:
                print(
                    f"failure finalization error: {type(final_exc).__name__}: {final_exc}",
                    file=sys.stderr,
                )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if was_resume and (run_dir / _RAW_SUFFIX_CONVERSION_INTENT).is_file():
        # This transaction outranks every live/retired failure namespace.  It
        # was durably committed before source-package supersession, so resume
        # may only complete its nonresumable raw-suffix package.
        try:
            _complete_raw_suffix_conversion_intent(run_dir)
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(run_dir)
        return 1
    live_capture = _live_failure_capture(run_dir) if was_resume else None
    live_terminal = _live_terminal_failure(run_dir) if was_resume else None
    if (
        was_resume
        and live_terminal is not None
        and live_terminal.get("failure_code")
        == "global_rollout_resource_boundary"
        and live_terminal.get("resume_same_frozen_run_authorized") == 1
        and _pending_raw_suffix_postprocessing_resource_stop(run_dir)
    ):
        # The exact suffix is complete, but its own finish event exhausted the
        # cap before derived work began.  Convert the earlier generic
        # resumable resource package into a nonresumable raw-suffix terminal
        # before supersession can authorize another scientific invocation.
        try:
            _convert_raw_suffix_resource_failure_to_nonresumable(run_dir)
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(run_dir)
        return 1
    if was_resume and live_capture is not None and live_terminal is None:
        # The atomic capture is the commit point for a failed attempt.  A hard
        # death before terminal creation may only finish packaging; it must not
        # append a resume event or re-enter the failed scientific stage.
        try:
            _recover_captured_failure_package(run_dir)
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(run_dir)
        return 1
    if (
        was_resume
        and live_terminal is not None
        and not _failure_verification_matches_terminal(run_dir, live_terminal)
    ):
        # The terminal failure record can exist before packaging finishes if
        # the process died between atomic commits.  This narrowly scoped path
        # completes only REPORT/HANDOFF/inventories/verification; it never
        # records a resume event or re-enters training/evaluation.
        try:
            _recover_incomplete_failure_package(run_dir)
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(run_dir)
        return 1
    if was_resume and live_terminal is not None:
        # Compatibility already authenticated the complete current package and
        # its resume authorization.  Archive and retire it atomically before
        # the first new ledger write; a crash at any retirement seam is
        # recoverable from the append-only supersession record.
        try:
            _supersede_authorized_failure(run_dir)
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    if was_resume:
        try:
            _complete_pending_failure_retirement(run_dir)
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        try:
            _recover_unmarked_completed_objective_resource_stop(run_dir, args)
        except Exception as exc:
            # This gate is deliberately before resume accounting.  A failed
            # deep verification must leave the suspect run byte-identical;
            # a passing recovery performs packaging-only terminal commits.
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    if (
        was_resume
        and (
            (run_dir / "completed_objective_resource_stop.json").is_file()
            or _pending_completed_objective_report_reserve_stop(run_dir)
            or _pending_terminal_storage_finalization(run_dir)
            or _pending_completed_objective_resource_packaging(run_dir)
        )
        and not _stage_complete(run_dir, "report_verify")
    ):
        # A terminal resource observation is already durable, so a crash in
        # administrative bundle construction must resume that construction
        # without appending another resource event or rerunning science.
        try:
            _finalize_and_verify(run_dir, args)
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(run_dir)
        return int((run_dir / "completed_objective_resource_stop.json").is_file())
    if was_resume and _stage_complete(run_dir, "report_verify"):
        # `_initialize_run` performed the full mutation-free compatibility and
        # completed-bundle audit.  Repeat the terminal read-only verifier for
        # an explicit handoff boundary, then exit before appending any resume
        # accounting event that would invalidate the sealed inventories.
        try:
            _verify_completed_report_read_only(run_dir, args)
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(run_dir)
        return int((run_dir / "completed_objective_resource_stop.json").is_file())
    try:
        if was_resume:
            _record_attempt_wall(
                run_dir,
                role="resume_compatibility_verification",
                started=resume_started,
                failed=False,
                detail={"resume_run_dir": str(run_dir)},
            )
        else:
            _record_attempt_wall(
                run_dir,
                role="fresh_run_initialization_and_source_closure",
                started=resume_started,
                failed=False,
                detail={"run_dir": str(run_dir)},
            )
    except Exception as exc:
        try:
            _capture_failure(run_dir, "initialization_accounting", exc)
            _finalize_failure(run_dir, "initialization_accounting", exc)
        except Exception as final_exc:
            print(
                f"failure finalization error: {type(final_exc).__name__}: {final_exc}",
                file=sys.stderr,
            )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    selected_stages = STAGES if args.stage == "all" else (args.stage,)
    current = "initialization"
    try:
        for stage in selected_stages:
            current = stage
            if _stage_complete(run_dir, stage):
                if not was_resume:
                    _verify_completed_stage(run_dir, args, stage)
                continue
            _run_stage(run_dir, args, stage)
    except Exception as exc:
        try:
            _capture_failure(run_dir, current, exc)
            _finalize_failure(run_dir, current, exc)
        except Exception as final_exc:
            print(
                f"failure finalization error: {type(final_exc).__name__}: {final_exc}",
                file=sys.stderr,
            )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"partial run preserved at {run_dir}", file=sys.stderr)
        return 1
    print(run_dir)
    return int((run_dir / "completed_objective_resource_stop.json").is_file())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
