"""Workflow primitives for the frequency-one coordinate learnability patch.

This module composes the existing exact eager cache, streaming-memory, and
restartable v3 max-T implementations.  It defines only the new path/seed
allocation, candidate identities, role firewall, coordinate-model training,
and seed-aware ranking needed by the additive representation change.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from mnist.d0_jacobi_artifacts import (
    atomic_write_json,
    config_fingerprint,
    file_fingerprint,
)
from mnist.d0_jacobi_rb_boundary_tangent import direct_raw_target_mse
from mnist.d0_jacobi_rb_boundary_tangent_eager_cache import EagerCohort
from mnist.d0_jacobi_rb_boundary_tangent_v3_memory import (
    CanonicalRowSquareReducer,
    HostInputStore,
    HostLabelStore,
    ModelCallBatchGuard,
    predict_to_cpu,
)
from mnist import d0_jacobi_rb_boundary_tangent_frequency1_coordinate_provenance as _provenance
from mnist import d0_jacobi_rb_boundary_tangent_v3_selection as _v3_selection
from mnist.d0_jacobi_rb_learnability import (
    deterministic_batch_indices,
    enable_deterministic_torch,
    state_dict_sha256,
)


FREQUENCY1_COORDINATE_LEARNABILITY_VERSION = (
    "d0-jacobi-rb-boundary-tangent-frequency1-coordinate-learnability-v1"
)
RUN_SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-frequency1-coordinate-learnability"
TEST_RUN_SCHEMA = RUN_SCHEMA + "-nonauthorizing-test"

STAGES = ("preflight", "cache", "controls", "train", "select", "confirm", "report")
PHYSICAL_STAGES = STAGES[:-1]
REQUIRED_GATES = ("none", "preflight", "cache", "controls", "train", "select", "confirm", "terminal")
STAGE_PREDECESSOR = {
    "preflight": None,
    "cache": "preflight",
    "controls": "cache",
    "train": "controls",
    "select": "train",
    "confirm": "select",
    "report": None,
}
STAGE_SEAL_NAMES = {
    "preflight": "preflight_artifact_seal.json",
    "cache": "cache_artifact_seal.json",
    "controls": "controls_artifact_seal.json",
    "train": "train_artifact_seal.json",
    "select": "selection_artifact_seal.json",
    "confirm": "confirmation_artifact_seal.json",
}

ROOT_SEED = 261_371
MODEL_SEEDS = (261_372, 261_373, 261_374)
SELECTION_BOOTSTRAP_SEED = 261_380
SELECTION_NAMESPACE = 0x46435631
FORBIDDEN_SCHEDULER_SEED = 261_381
CONFIRMATION_BOOTSTRAP_SEED = 261_382
CONFIRMATION_NAMESPACE = 0x46434331
SYNTHETIC_COORDINATE_TEACHER_SEED = 261_383
EXACT_MODEL_NULL_SEED = 261_384
INITIALIZATION_CONTROL_SEED = 261_385
RESERVED_FUTURE_CONTROL_SEED = 261_386

TRAINING = {
    "width": 32,
    "batch_size": 32,
    "prediction_batch_size": 32,
    "maximum_updates": 4_000,
    "checkpoint_interval": 100,
    "learning_rate": 1.0e-3,
    "betas": (0.9, 0.999),
    "epsilon": 1.0e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
    "gradient_norm_clip": 1.0,
    "mixed_precision": 0,
}

PREFLIGHT_PATH_IDS = tuple(range(0xF8000, 0xF8008))
TRAIN_PATH_IDS = tuple(range(0xF8100, 0xF8140))
VALIDATION_PATH_IDS = tuple(range(0xF8200, 0xF8220))
CONFIRMATION_PATH_IDS = tuple(range(0xF9000, 0xF9040))
PATH_ROLES = ("preflight_seam", "train", "validation", "confirmation")
PRODUCTION_PATHS = {
    "preflight_seam": PREFLIGHT_PATH_IDS,
    "train": TRAIN_PATH_IDS,
    "validation": VALIDATION_PATH_IDS,
    "confirmation": CONFIRMATION_PATH_IDS,
}

CHECKPOINT_UPDATES = tuple(range(0, 4_001, 100))
NONZERO_UPDATES = CHECKPOINT_UPDATES[1:]
FAMILY_NAMES = _v3_selection.V3_FAMILY_NAMES
FAMILY_NAMES_SHA256 = _v3_selection.V3_FAMILY_NAMES_SHA256
COMPONENT_COUNT = len(FAMILY_NAMES)
CANDIDATE_COUNT = len(MODEL_SEEDS) * len(NONZERO_UPDATES)
SEARCH_FAMILY_SIZE = CANDIDATE_COUNT * COMPONENT_COUNT
SEARCH_FAMILY_NAMES = tuple(
    f"seed{seed}.update{update:04d}.{component}"
    for seed in MODEL_SEEDS
    for update in NONZERO_UPDATES
    for component in FAMILY_NAMES
)
SEARCH_FAMILY_NAMES_SHA256 = config_fingerprint(list(SEARCH_FAMILY_NAMES))
SELECTED_OUTER_STEPS = tuple(range(15, 512, 16))

NO_WORK = {
    "controller_control_trajectory_performed": 0,
    "full_reverse_path_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
    "image_sampling_performed": 0,
    "reconstruction_performed": 0,
    "full_dataset_training_performed": 0,
}


class Frequency1CoordinateWorkflowError(RuntimeError):
    """A frozen workflow, role, or numerical contract was violated."""

    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "workflow_execution",
        failure_code: str = "frequency1_coordinate_workflow_invalid",
    ) -> None:
        super().__init__(message)
        self.failure_domain = str(failure_domain)
        self.failure_code = str(failure_code)


def _semantic(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("semantic_sha256", None)
    return {**body, "semantic_sha256": config_fingerprint(body)}


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(config_fingerprint(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


def build_path_plan(
    *, test_only: bool = False, test_path_count: int = 8
) -> dict[str, Any]:
    """Return the frozen production allocation or an isolated test fixture."""

    if not test_only:
        return _provenance.build_frequency1_path_plan()
    if test_only:
        if not isinstance(test_path_count, int) or isinstance(test_path_count, bool):
            raise Frequency1CoordinateWorkflowError("test path count must be integral")
        if not 2 <= test_path_count <= 8:
            raise Frequency1CoordinateWorkflowError("test path count must lie in [2,8]")
        roles = {
            "preflight_seam": list(range(0x1800, 0x1800 + test_path_count)),
            "train": list(range(0x1900, 0x1900 + test_path_count)),
            "validation": list(range(0x1A00, 0x1A00 + test_path_count)),
            "confirmation": list(range(0x1B00, 0x1B00 + test_path_count)),
        }
    flattened = [int(path) for name in PATH_ROLES for path in roles[name]]
    if (
        len(flattened) != len(set(flattened))
        or any(path < 0 or path >= 1 << 20 for path in flattened)
    ):
        raise Frequency1CoordinateWorkflowError(
            "path roles overlap or leave the 20-bit namespace",
            failure_domain="path_or_resource_plan",
            failure_code="frequency1_coordinate_path_plan_invalid",
        )
    record = {
        "schema": (TEST_RUN_SCHEMA if test_only else RUN_SCHEMA) + "-path-plan",
        "schema_version": 1,
        "test_only": int(test_only),
        "roles": roles,
        "role_order": list(PATH_ROLES),
        "all_roles_disjoint": 1,
        "production_path_ids_opened": 0,
        "authorizing": int(not test_only),
    }
    return _semantic(record)


def _partition(values: Sequence[int], sizes: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    offset = 0
    for size in sizes:
        groups.append(tuple(int(value) for value in values[offset : offset + int(size)]))
        offset += int(size)
    if offset != len(values) or any(not group for group in groups):
        raise Frequency1CoordinateWorkflowError("cohort partition is incomplete")
    return tuple(groups)


def build_cohort_plan(path_plan: Mapping[str, Any]) -> dict[str, Any]:
    roles = path_plan.get("roles")
    if not isinstance(roles, Mapping):
        raise Frequency1CoordinateWorkflowError("path plan has no roles")
    test_only = bool(int(path_plan.get("test_only", 0)))
    if not test_only:
        return _provenance.build_frequency1_cohort_plan(path_plan)
    train = tuple(int(value) for value in roles["train"])
    validation = tuple(int(value) for value in roles["validation"])
    confirmation = tuple(int(value) for value in roles["confirmation"])
    combined = train + validation
    if test_only:
        train_validation_groups = tuple(
            tuple(combined[start : start + 10])
            for start in range(0, len(combined), 10)
        )
        confirmation_groups = tuple(
            tuple(confirmation[start : start + 10])
            for start in range(0, len(confirmation), 10)
        )
    else:
        train_validation_groups = _partition(combined, (10,) * 9 + (6,))
        confirmation_groups = _partition(confirmation, (10,) * 6 + (4,))
    role_by_path = {
        **{path: "train" for path in train},
        **{path: "validation" for path in validation},
        **{path: "confirmation" for path in confirmation},
    }

    def records(kind: str, groups: Sequence[Sequence[int]]) -> list[dict[str, Any]]:
        return [
            {
                "kind": kind,
                "index": index,
                "size": len(group),
                "path_ids": list(group),
                "path_roles": [role_by_path[int(path)] for path in group],
            }
            for index, group in enumerate(groups)
        ]

    return _semantic(
        {
            "schema": (TEST_RUN_SCHEMA if test_only else RUN_SCHEMA) + "-cohort-plan",
            "schema_version": 1,
            "test_only": int(test_only),
            "path_id_plan_sha256": path_plan["semantic_sha256"],
            "train_validation": records("train_validation", train_validation_groups),
            "confirmation": records("confirmation", confirmation_groups),
            "train_validation_sizes": [len(group) for group in train_validation_groups],
            "confirmation_sizes": [len(group) for group in confirmation_groups],
            "authorizing": int(not test_only),
        }
    )


def eager_cohorts(
    cohort_plan: Mapping[str, Any], kind: str
) -> tuple[EagerCohort, ...]:
    if kind not in {"train_validation", "confirmation"}:
        raise Frequency1CoordinateWorkflowError("unknown cohort kind")
    rows = cohort_plan[kind]
    roles = cohort_plan.get("test_only")
    if roles:
        return tuple(
            EagerCohort(
                kind=str(row["kind"]),
                index=int(row["index"]),
                path_ids=tuple(int(value) for value in row["path_ids"]),
                path_roles=tuple(str(value) for value in row["path_roles"]),
            )
            for row in rows
        )
    training = set(TRAIN_PATH_IDS)
    validation = set(VALIDATION_PATH_IDS)
    return tuple(
        EagerCohort(
            kind=kind,
            index=index,
            path_ids=tuple(int(value) for value in row),
            path_roles=tuple(
                "train"
                if int(value) in training
                else "validation"
                if int(value) in validation
                else "confirmation"
                for value in row
            ),
        )
        for index, row in enumerate(rows)
    )


def seed_plan() -> dict[str, Any]:
    return _provenance.build_frequency1_seed_plan()


def checkpoint_plan(
    *, test_only: bool = False, test_maximum_updates: int = 0
) -> dict[str, Any]:
    maximum = int(test_maximum_updates) if test_only else int(TRAINING["maximum_updates"])
    if maximum < 0:
        raise Frequency1CoordinateWorkflowError("maximum updates must be nonnegative")
    interval = 1 if test_only and maximum < 100 else int(TRAINING["checkpoint_interval"])
    updates = tuple(range(0, maximum + 1, interval))
    if updates[-1] != maximum:
        updates = (*updates, maximum)
    return _semantic(
        {
            "schema": (TEST_RUN_SCHEMA if test_only else RUN_SCHEMA) + "-checkpoint-plan",
            "schema_version": 1,
            "model_seeds": list(MODEL_SEEDS),
            "maximum_updates": maximum,
            "checkpoint_interval": interval,
            "checkpoint_updates": list(updates),
            "nonzero_candidate_count": len(MODEL_SEEDS) * max(len(updates) - 1, 0),
            "early_stopping_forbidden": 1,
            "validation_evidence_available_to_training": 0,
            "authorizing": int(not test_only),
        }
    )


def selection_inference_plan(
    *,
    test_only: bool = False,
    test_replicates: int = 8,
    checkpoint_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = dict(checkpoint_record or checkpoint_plan(test_only=test_only))
    replicates = int(test_replicates) if test_only else 50_000
    shard_size = replicates if test_only else 1_000
    seeds = tuple(int(value) for value in checkpoint["model_seeds"])
    updates = tuple(int(value) for value in checkpoint["checkpoint_updates"] if int(value) > 0)
    names = tuple(
        f"seed{seed}.update{update:04d}.{component}"
        for seed in seeds
        for update in updates
        for component in FAMILY_NAMES
    )
    body = {
        "schema": (TEST_RUN_SCHEMA if test_only else RUN_SCHEMA) + "-selection-plan",
        "schema_version": 1,
        "test_only": int(test_only),
        "candidate_order": [[seed, update] for seed in seeds for update in updates],
        "candidate_count": len(seeds) * len(updates),
        "component_names": list(FAMILY_NAMES),
        "component_names_sha256": FAMILY_NAMES_SHA256,
        "component_count": COMPONENT_COUNT,
        "search_family_size": len(names),
        "search_family_names_sha256": config_fingerprint(list(names)),
        "search_flattening_order": "candidate_major_then_component",
        "bootstrap_seed": SELECTION_BOOTSTRAP_SEED,
        "bootstrap_namespace": SELECTION_NAMESPACE,
        "bootstrap_replicates": replicates,
        "bootstrap_shard_size": shard_size,
        "bootstrap_shard_count": replicates // shard_size,
        "candidate_block_size": 20,
        "component_block_size": 57,
        "working_family_per_block": 1_140,
        "confidence": 0.995,
        "quantile_method": "higher",
        "negative_values_truncated": 0,
        "standard_error_floor_used": 0,
        "ranking_rule": [
            "largest_minimum_lower_bound",
            "earlier_update",
            "lower_model_seed",
        ],
        "authorizing": int(not test_only),
    }
    return _semantic(body)


def confirmation_inference_plan(
    *, test_only: bool = False, test_replicates: int = 8
) -> dict[str, Any]:
    replicates = int(test_replicates) if test_only else 50_000
    shard_size = replicates if test_only else 1_000
    return _semantic(
        {
            "schema": (TEST_RUN_SCHEMA if test_only else RUN_SCHEMA) + "-confirmation-plan",
            "schema_version": 1,
            "path_ids": list(CONFIRMATION_PATH_IDS) if not test_only else [],
            "candidate_count": 1,
            "component_names": list(FAMILY_NAMES),
            "component_names_sha256": FAMILY_NAMES_SHA256,
            "component_count": COMPONENT_COUNT,
            "bootstrap_seed": CONFIRMATION_BOOTSTRAP_SEED,
            "bootstrap_namespace": CONFIRMATION_NAMESPACE,
            "bootstrap_replicates": replicates,
            "bootstrap_shard_size": shard_size,
            "confidence": 0.995,
            "quantile_method": "higher",
            "negative_values_truncated": 0,
            "standard_error_floor_used": 0,
            "raw_inputs_persisted": 0,
            "raw_labels_persisted": 0,
            "raw_predictions_persisted": 0,
            "authorizing": int(not test_only),
        }
    )


@dataclass(frozen=True)
class Frequency1CandidateTable:
    """Canonical seed-aware ``[path,candidate,228]`` validation evidence."""

    seeds: np.ndarray
    updates: np.ndarray
    path_ids: np.ndarray
    path_values: np.ndarray

    def __post_init__(self) -> None:
        seeds = np.asarray(self.seeds)
        updates = np.asarray(self.updates)
        paths = np.asarray(self.path_ids)
        values = np.asarray(self.path_values)
        if (
            seeds.dtype != np.dtype(np.int64)
            or seeds.ndim != 1
            or updates.dtype != np.dtype(np.int64)
            or updates.shape != seeds.shape
            or paths.dtype != np.dtype(np.int64)
            or paths.ndim != 1
            or paths.size < 8
            or np.unique(paths).size != paths.size
            or np.any(paths < 0)
            or np.any(paths >= 1 << 20)
            or values.dtype != np.dtype(np.float64)
            or values.shape != (paths.size, seeds.size, COMPONENT_COUNT)
            or not np.isfinite(values).all()
            or seeds.size == 0
        ):
            raise Frequency1CoordinateWorkflowError(
                "candidate validation table is malformed",
                failure_domain="validation_inference",
                failure_code="frequency1_coordinate_candidate_table_invalid",
            )
        pairs = tuple(zip(seeds.tolist(), updates.tolist(), strict=True))
        if len(set(pairs)) != len(pairs) or any(
            int(seed) not in MODEL_SEEDS or int(update) <= 0 for seed, update in pairs
        ):
            raise Frequency1CoordinateWorkflowError("candidate identity is invalid")
        candidate_order = np.lexsort((updates, seeds))
        path_order = np.argsort(paths, kind="stable")
        object.__setattr__(self, "seeds", _readonly(seeds[candidate_order]))
        object.__setattr__(self, "updates", _readonly(updates[candidate_order]))
        object.__setattr__(self, "path_ids", _readonly(paths[path_order]))
        object.__setattr__(
            self,
            "path_values",
            _readonly(values[path_order][:, candidate_order, :]),
        )

    @property
    def candidate_count(self) -> int:
        return int(self.seeds.size)

    @property
    def fingerprint(self) -> str:
        return config_fingerprint(
            {
                "seeds": self.seeds.tolist(),
                "updates": self.updates.tolist(),
                "path_ids": self.path_ids.tolist(),
                "path_values_sha256": _array_sha256(self.path_values),
                "family_names_sha256": FAMILY_NAMES_SHA256,
            }
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": FREQUENCY1_COORDINATE_LEARNABILITY_VERSION + "-candidate-table",
            "schema_version": 1,
            "shape": list(self.path_values.shape),
            "path_ids": self.path_ids.tolist(),
            "seeds": self.seeds.tolist(),
            "updates": self.updates.tolist(),
            "candidate_count": self.candidate_count,
            "component_count": COMPONENT_COUNT,
            "search_family_size": self.candidate_count * COMPONENT_COUNT,
            "family_names": list(FAMILY_NAMES),
            "family_names_sha256": FAMILY_NAMES_SHA256,
            "fingerprint": self.fingerprint,
            "negative_values_truncated": 0,
        }


def build_candidate_table(
    *,
    seeds: Any,
    updates: Any,
    path_ids: Any,
    path_values: Any,
    forbidden_path_ids: Any | None = None,
) -> Frequency1CandidateTable:
    paths = np.asarray(path_ids)
    if paths.dtype.kind not in "iu":
        raise Frequency1CoordinateWorkflowError("path IDs must be integral")
    canonical_paths = np.asarray(paths, dtype=np.int64)
    if forbidden_path_ids is not None:
        forbidden = np.asarray(forbidden_path_ids)
        if forbidden.ndim != 1 or forbidden.dtype.kind not in "iu":
            raise Frequency1CoordinateWorkflowError("forbidden path IDs are malformed")
        if np.intersect1d(canonical_paths, forbidden.astype(np.int64, copy=False)).size:
            raise Frequency1CoordinateWorkflowError(
                "confirmation paths entered validation",
                failure_domain="role_firewall",
                failure_code="frequency1_coordinate_confirmation_path_firewall_violated",
            )
    return Frequency1CandidateTable(
        seeds=np.asarray(seeds, dtype=np.int64),
        updates=np.asarray(updates, dtype=np.int64),
        path_ids=canonical_paths,
        path_values=np.asarray(path_values),
    )


def rank_validation_nominee(
    table: Frequency1CandidateTable,
    result: _v3_selection.NumericMaxTResult,
) -> dict[str, Any]:
    if (
        not isinstance(table, Frequency1CandidateTable)
        or not isinstance(result, _v3_selection.NumericMaxTResult)
        or result.candidate_count != table.candidate_count
        or not np.array_equal(result.path_ids, table.path_ids)
    ):
        raise Frequency1CoordinateWorkflowError("ranking evidence is not aligned")
    minimum = np.min(result.lower_bounds, axis=1)
    eligible = np.all(result.lower_bounds > 0.0, axis=1)
    rows = [
        {
            "seed": int(table.seeds[index]),
            "update": int(table.updates[index]),
            "minimum_lower_bound": float(minimum[index]),
            "all_228_lower_bounds_strictly_positive": int(eligible[index]),
            "positive_lower_bound_count": int(np.count_nonzero(result.lower_bounds[index] > 0.0)),
        }
        for index in range(table.candidate_count)
    ]
    eligible_indices = np.flatnonzero(eligible)
    if eligible_indices.size == 0:
        return {
            "schema": FREQUENCY1_COORDINATE_LEARNABILITY_VERSION + "-nominee-ranking",
            "schema_version": 1,
            "decision": "no_frequency1_coordinate_validation_candidate",
            "candidate_count": table.candidate_count,
            "eligible_candidate_count": 0,
            "selected_seed": None,
            "selected_update": 0,
            "logical_update_zero_selected": 1,
            "confirmation_authorized": 0,
            "candidate_rows": rows,
        }
    selected = min(
        eligible_indices.tolist(),
        key=lambda index: (
            -float(minimum[index]),
            int(table.updates[index]),
            int(table.seeds[index]),
        ),
    )
    return {
        "schema": FREQUENCY1_COORDINATE_LEARNABILITY_VERSION + "-nominee-ranking",
        "schema_version": 1,
        "decision": "frequency1_coordinate_validation_nominee_sealed",
        "candidate_count": table.candidate_count,
        "eligible_candidate_count": int(eligible_indices.size),
        "selected_seed": int(table.seeds[selected]),
        "selected_update": int(table.updates[selected]),
        "selected_minimum_lower_bound": float(minimum[selected]),
        "logical_update_zero_selected": 0,
        "confirmation_authorized": 1,
        "ranking_rule": [
            "largest_minimum_lower_bound",
            "earlier_update",
            "lower_model_seed",
        ],
        "candidate_rows": rows,
    }


def restartable_selection_max_t(
    table: Frequency1CandidateTable,
    *,
    count_directory: str | Path,
    maxima_directory: str | Path,
    replicates: int = 50_000,
    shard_size: int = 1_000,
    environment: Mapping[str, Any] | None = None,
) -> tuple[_v3_selection.NumericMaxTResult, dict[str, Any]]:
    result, count_records, maxima_records = _v3_selection.restartable_numeric_v3_max_t(
        table.path_values,
        path_ids=table.path_ids,
        count_directory=count_directory,
        maxima_directory=maxima_directory,
        seed=SELECTION_BOOTSTRAP_SEED,
        namespace=SELECTION_NAMESPACE,
        confidence=0.995,
        replicates=int(replicates),
        shard_size=int(shard_size),
        candidate_block_size=20,
        component_block_size=57,
        environment=environment,
    )
    ranking = rank_validation_nominee(table, result)
    ranking.update(
        {
            "critical_value": float(result.critical_value),
            "confidence": float(result.confidence),
            "replicates": int(result.maxima.size),
            "seed": SELECTION_BOOTSTRAP_SEED,
            "namespace": SELECTION_NAMESPACE,
            "family_names_sha256": FAMILY_NAMES_SHA256,
            "candidate_table_fingerprint": table.fingerprint,
            "count_metadata_semantic_sha256": [
                row["semantic_sha256"] for row in count_records
            ],
            "maxima_metadata_semantic_sha256": [
                row["semantic_sha256"] for row in maxima_records
            ],
            "maxima_sha256": _array_sha256(result.maxima),
            "lower_bounds_sha256": _array_sha256(result.lower_bounds),
        }
    )
    return result, ranking


def restartable_confirmation_max_t(
    path_values: Any,
    *,
    path_ids: Any,
    count_directory: str | Path,
    maxima_directory: str | Path,
    replicates: int = 50_000,
    shard_size: int = 1_000,
    environment: Mapping[str, Any] | None = None,
) -> tuple[_v3_selection.NumericMaxTResult, dict[str, Any]]:
    result, count_records, maxima_records = _v3_selection.restartable_numeric_v3_max_t(
        np.asarray(path_values, dtype=np.float64)[:, None, :],
        path_ids=path_ids,
        count_directory=count_directory,
        maxima_directory=maxima_directory,
        seed=CONFIRMATION_BOOTSTRAP_SEED,
        namespace=CONFIRMATION_NAMESPACE,
        confidence=0.995,
        replicates=int(replicates),
        shard_size=int(shard_size),
        candidate_block_size=20,
        component_block_size=57,
        environment=environment,
    )
    record = {
        **_v3_selection.v3_confirmation_max_t_record(result),
        "count_metadata_semantic_sha256": [row["semantic_sha256"] for row in count_records],
        "maxima_metadata_semantic_sha256": [row["semantic_sha256"] for row in maxima_records],
        "seed": CONFIRMATION_BOOTSTRAP_SEED,
        "namespace": CONFIRMATION_NAMESPACE,
    }
    return result, record


aggregate_zero_baseline_improvements = _v3_selection.aggregate_zero_baseline_improvements
aggregate_zero_baseline_risks = _v3_selection.aggregate_zero_baseline_risks
prepare_bootstrap_count_shards = _v3_selection.prepare_bootstrap_count_shards
load_bootstrap_count_shards = _v3_selection.load_bootstrap_count_shards


def _relative_file(run_dir: Path, name: str) -> Path:
    root = run_dir.resolve()
    path = (root / name).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise Frequency1CoordinateWorkflowError("artifact path escapes the run") from exc
    return path


def seal_artifacts(
    run_dir: str | Path,
    artifact_names: Iterable[str],
    seal_name: str,
) -> dict[str, Any]:
    """Commit or verify a metadata-last immutable stage seal."""

    root = Path(run_dir).resolve()
    names = tuple(sorted(set(str(name) for name in artifact_names)))
    if not names or seal_name in names:
        raise Frequency1CoordinateWorkflowError("stage seal input is malformed")
    records: list[dict[str, Any]] = []
    for name in names:
        path = _relative_file(root, name)
        if not path.is_file():
            raise Frequency1CoordinateWorkflowError(f"required stage artifact is missing: {name}")
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": int(path.stat().st_size),
                "sha256": file_fingerprint(path),
            }
        )
    record = _semantic(
        {
            "schema": FREQUENCY1_COORDINATE_LEARNABILITY_VERSION + "-artifact-seal",
            "schema_version": 1,
            "seal": str(seal_name),
            "artifacts": records,
            "artifact_count": len(records),
            "committed": 1,
        }
    )
    path = _relative_file(root, seal_name)
    if path.is_file():
        existing = load_json(path)
        if existing != record:
            raise Frequency1CoordinateWorkflowError("completed stage seal changed")
    else:
        atomic_write_json(path, record)
    return record


def load_json(path: str | Path) -> dict[str, Any]:
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Frequency1CoordinateWorkflowError(f"JSON record is not an object: {path}")
    return value


def verify_artifact_seal(run_dir: str | Path, seal_name: str) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    record = load_json(_relative_file(root, seal_name))
    body = dict(record)
    semantic = body.pop("semantic_sha256", None)
    if semantic != config_fingerprint(body) or int(record.get("committed", 0)) != 1:
        raise Frequency1CoordinateWorkflowError("stage seal is malformed")
    rows = record.get("artifacts")
    if not isinstance(rows, list) or len(rows) != int(record.get("artifact_count", -1)):
        raise Frequency1CoordinateWorkflowError("stage seal inventory is malformed")
    for row in rows:
        if not isinstance(row, Mapping):
            raise Frequency1CoordinateWorkflowError("stage seal row is malformed")
        path = _relative_file(root, str(row.get("path", "")))
        if (
            not path.is_file()
            or int(row.get("size", -1)) != path.stat().st_size
            or row.get("sha256") != file_fingerprint(path)
        ):
            raise Frequency1CoordinateWorkflowError("sealed stage artifact changed")
    return record


def gate_passed(record: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(record, Mapping)
        and str(record.get("evaluation_status")) == "evaluated"
        and int(record.get("passed", 0)) == 1
    )


def validate_stage_entry(run_dir: str | Path, stage: str) -> None:
    """Enforce sequential stages and immutable predecessor seals."""

    if stage not in STAGES:
        raise Frequency1CoordinateWorkflowError(f"unknown stage: {stage}")
    root = Path(run_dir).resolve()
    predecessor = STAGE_PREDECESSOR[stage]
    if predecessor is not None:
        gate_path = root / f"{predecessor}_gate.json"
        if not gate_path.is_file() or not gate_passed(load_json(gate_path)):
            raise Frequency1CoordinateWorkflowError(
                f"{stage} requires a passing {predecessor} gate",
                failure_domain="role_order",
                failure_code=f"frequency1_coordinate_{stage}_predecessor_invalid",
            )
        verify_artifact_seal(root, STAGE_SEAL_NAMES[predecessor])
    current_gate = root / f"{stage}_gate.json"
    if current_gate.is_file() and stage != "report":
        verify_artifact_seal(root, STAGE_SEAL_NAMES[stage])


def _opening_record_path(role: str) -> str:
    names = {
        "train": "physical_train_label_open.json",
        "validation": "validation_label_open.json",
        "confirmation": "confirmation_namespace_open.json",
    }
    try:
        return names[role]
    except KeyError as exc:
        raise Frequency1CoordinateWorkflowError("unknown opening role") from exc


def open_label_role(
    run_dir: str | Path,
    role: str,
    *,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit a one-time role opening after its exact prerequisite seal."""

    root = Path(run_dir).resolve()
    prerequisites = {
        "train": ("controls_gate.json", STAGE_SEAL_NAMES["controls"]),
        "validation": ("selection_opening_intent_seal.json", None),
        "confirmation": ("selection_decision.json", STAGE_SEAL_NAMES["select"]),
    }
    if role not in prerequisites:
        raise Frequency1CoordinateWorkflowError("unknown label role")
    gate_name, stage_seal = prerequisites[role]
    prerequisite_path = root / gate_name
    if not prerequisite_path.is_file():
        raise Frequency1CoordinateWorkflowError(
            f"{role} opening prerequisite is absent",
            failure_domain="role_firewall",
            failure_code=f"frequency1_coordinate_{role}_opening_invalid",
        )
    if role == "validation":
        # The prospective candidate/family/count-shard intent must itself be
        # immutable before the first validation label byte is deserialized.
        verify_artifact_seal(root, "selection_opening_intent_seal.json")
    if role == "confirmation":
        # Confirmation has a second, nominee-bound intent that is sealed only
        # after selection and before any fresh transition is generated.
        verify_artifact_seal(root, "confirmation_opening_intent_seal.json")
    prerequisite = load_json(prerequisite_path)
    if role == "train" and not gate_passed(prerequisite):
        raise Frequency1CoordinateWorkflowError("train labels require passing controls")
    if role == "confirmation" and (
        prerequisite.get("decision") != "frequency1_coordinate_validation_nominee_sealed"
        or int(prerequisite.get("confirmation_authorized", 0)) != 1
    ):
        raise Frequency1CoordinateWorkflowError("confirmation requires the sealed nominee")
    if stage_seal is not None:
        verify_artifact_seal(root, stage_seal)
    opening_path = root / _opening_record_path(role)
    record = _semantic(
        {
            "schema": FREQUENCY1_COORDINATE_LEARNABILITY_VERSION + f"-{role}-open",
            "schema_version": 1,
            "role": role,
            "opened_once": 1,
            "binding": dict(binding),
            "prerequisite_sha256": file_fingerprint(prerequisite_path),
            "confirmation_namespace_opened": int(role == "confirmation"),
            "namespace_permanently_burned": int(role == "confirmation"),
            **NO_WORK,
        }
    )
    if opening_path.is_file():
        existing = load_json(opening_path)
        if existing != record:
            raise Frequency1CoordinateWorkflowError(f"opened {role} role changed")
        return existing
    atomic_write_json(opening_path, record)
    return record


def assert_role_firewall(run_dir: str | Path, stage: str) -> None:
    root = Path(run_dir).resolve()
    opened = {
        role: (root / _opening_record_path(role)).is_file()
        for role in ("train", "validation", "confirmation")
    }
    allowed = {
        "preflight": set(),
        "cache": set(),
        "controls": set(),
        "train": {"train"},
        "select": {"train", "validation"},
        "confirm": {"train", "validation", "confirmation"},
        "report": {role for role, active in opened.items() if active},
    }[stage]
    forbidden = {role for role, active in opened.items() if active and role not in allowed}
    if forbidden:
        raise Frequency1CoordinateWorkflowError(
            f"roles opened too early: {sorted(forbidden)}",
            failure_domain="role_firewall",
            failure_code="frequency1_coordinate_label_firewall_violated",
        )
    if opened["confirmation"] and not opened["validation"]:
        raise Frequency1CoordinateWorkflowError("confirmation opened before validation")


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _clone_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in model.state_dict().items()
    }


def _target_scale(labels: HostLabelStore, *, batch_size: int = 32) -> float:
    reducer = CanonicalRowSquareReducer()
    for start in range(0, labels.row_count, batch_size):
        rows = np.arange(start, min(labels.row_count, start + batch_size), dtype=np.int64)
        reducer.update(labels.target_batch(rows, device="cpu"))
    value = reducer.rms
    if not math.isfinite(value) or value <= 0.0:
        raise Frequency1CoordinateWorkflowError("training target RMS is invalid")
    return value


def train_frequency1_coordinate_candidate(
    *,
    model: nn.Module,
    train_inputs: HostInputStore,
    train_labels: HostLabelStore,
    seed: int,
    progress_path: str | Path,
    checkpoint_root: str | Path,
    scientific_config_sha256: str,
    maximum_updates: int = 4_000,
    checkpoint_interval: int = 100,
    stop_after_update: int | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run or resume one training-only trajectory using v3 streaming stores."""

    if int(seed) not in MODEL_SEEDS:
        raise Frequency1CoordinateWorkflowError("physical model seed changed")
    if train_inputs.role != "train" or train_labels.role != "train":
        raise Frequency1CoordinateWorkflowError("physical trainer received another role")
    if train_labels.purpose != "physical_training":
        raise Frequency1CoordinateWorkflowError("training label purpose changed")
    if train_inputs.row_count != train_labels.row_count:
        raise Frequency1CoordinateWorkflowError("training input/label rows differ")
    if maximum_updates < 0 or checkpoint_interval <= 0:
        raise Frequency1CoordinateWorkflowError("training update plan is invalid")
    active_device = torch.device(device)
    enable_deterministic_torch()
    torch.manual_seed(int(seed))
    if active_device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = model.to(active_device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(TRAINING["learning_rate"]),
        betas=tuple(float(value) for value in TRAINING["betas"]),
        eps=float(TRAINING["epsilon"]),
        weight_decay=float(TRAINING["weight_decay"]),
        amsgrad=bool(TRAINING["amsgrad"]),
    )
    stem = getattr(getattr(model, "residual_score", model), "coordinate_stem_weight", None)
    if not isinstance(stem, nn.Parameter) or not any(stem is value for group in optimizer.param_groups for value in group["params"]):
        raise Frequency1CoordinateWorkflowError("coordinate stem is absent from optimizer")
    scale = _target_scale(train_labels)
    progress = Path(progress_path)
    checkpoint_dir = Path(checkpoint_root)
    fingerprint = config_fingerprint(
        {
            "schema": FREQUENCY1_COORDINATE_LEARNABILITY_VERSION + "-physical-training",
            "seed": int(seed),
            "scientific_config_sha256": str(scientific_config_sha256),
            "input_index": dict(train_inputs.index),
            "label_index": dict(train_labels.index),
            "label_opening_seal_sha256": train_labels.opening_seal_sha256,
            "target_scale": scale,
            "maximum_updates": int(maximum_updates),
            "checkpoint_interval": int(checkpoint_interval),
        }
    )
    completed = 0
    records: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    if progress.is_file():
        saved = torch.load(progress, map_location=active_device, weights_only=False)
        if saved.get("fingerprint") != fingerprint:
            raise Frequency1CoordinateWorkflowError("training resume fingerprint changed")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        completed = int(saved["completed_update"])
        records = [dict(row) for row in saved["checkpoint_records"]]
        history = [dict(row) for row in saved["history"]]
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        if active_device.type == "cuda" and saved.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(list(saved["cuda_rng_states"]))

    def save_candidate(update: int) -> dict[str, Any]:
        state = _clone_state_dict(model)
        state_hash = state_dict_sha256(state)
        path = checkpoint_dir / f"seed-{seed}" / f"update-{update:04d}.pt"
        _atomic_torch(
            path,
            {
                "schema": FREQUENCY1_COORDINATE_LEARNABILITY_VERSION + "-candidate",
                "schema_version": 1,
                "seed": int(seed),
                "update": int(update),
                "training_fingerprint": fingerprint,
                "state_dict": state,
                "state_sha256": state_hash,
                "training_only": 1,
                "validation_evidence_used": 0,
                **NO_WORK,
            },
        )
        row = {
            "seed": int(seed),
            "update": int(update),
            "state_sha256": state_hash,
            "checkpoint_path": path.relative_to(checkpoint_dir.parent.parent).as_posix(),
            "checkpoint_file_sha256": file_fingerprint(path),
        }
        records.append(row)
        return row

    def save_progress(update: int) -> None:
        _atomic_torch(
            progress,
            {
                "schema": FREQUENCY1_COORDINATE_LEARNABILITY_VERSION + "-progress",
                "schema_version": 1,
                "fingerprint": fingerprint,
                "completed_update": int(update),
                "sampler_position": int(update),
                "model_state_dict": _clone_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "checkpoint_records": records,
                "history": history,
                "target_scale": scale,
                "torch_rng_state": torch.get_rng_state().clone(),
                "cuda_rng_states": tuple(torch.cuda.get_rng_state_all())
                if active_device.type == "cuda"
                else (),
            },
        )

    guard = ModelCallBatchGuard(maximum_batch_size=32)
    if not records:
        with torch.no_grad():
            rows = np.arange(min(32, train_inputs.row_count), dtype=np.int64)
            prediction = guard.call(model, train_inputs.batch(rows, device=active_device))
            if not bool(torch.all(prediction == 0.0)):
                raise Frequency1CoordinateWorkflowError("update-zero prediction is not exact zero")
        save_candidate(0)
        save_progress(0)
    limit = int(maximum_updates)
    if stop_after_update is not None:
        limit = min(limit, int(stop_after_update))
    model.train()
    for update in range(completed + 1, limit + 1):
        indices = deterministic_batch_indices(
            train_inputs.row_count,
            int(TRAINING["batch_size"]),
            update - 1,
            int(seed),
        )
        batch = train_inputs.batch(indices, device=active_device)
        target = train_labels.target_batch(indices, device=active_device)
        optimizer.zero_grad(set_to_none=True)
        prediction = guard.call(model, batch)
        loss, raw = direct_raw_target_mse(prediction, target, scale)
        if not bool(torch.isfinite(loss)):
            raise Frequency1CoordinateWorkflowError("training loss became nonfinite")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(TRAINING["gradient_norm_clip"])
        )
        if not math.isfinite(float(gradient)):
            raise Frequency1CoordinateWorkflowError("training gradient became nonfinite")
        optimizer.step()
        if update % int(checkpoint_interval) == 0 or update == int(maximum_updates):
            candidate = save_candidate(update)
            history.append(
                {
                    "update": int(update),
                    "train_raw_mse": float(raw.detach().cpu()),
                    "scaled_loss": float(loss.detach().cpu()),
                    "preclip_gradient_norm": float(gradient),
                    "checkpoint_state_sha256": candidate["state_sha256"],
                }
            )
            # Restart only from a sealed prospective checkpoint boundary.  A
            # process interruption between boundaries deterministically
            # replays at most one interval and never creates 4,000 large
            # progress writes per seed.
            save_progress(update)
    complete = int(limit == maximum_updates)
    return _semantic(
        {
            "schema": FREQUENCY1_COORDINATE_LEARNABILITY_VERSION + "-physical-task",
            "schema_version": 1,
            "seed": int(seed),
            "complete": complete,
            "completed_update": limit,
            "maximum_updates": int(maximum_updates),
            "checkpoint_interval": int(checkpoint_interval),
            "checkpoint_count": len(records),
            "checkpoints": records,
            "history": history,
            "training_fingerprint": fingerprint,
            "target_scale": scale,
            "validation_inputs_received": 0,
            "validation_labels_received": 0,
            "selection_performed": 0,
            "model_call_batches": guard.record(),
            "physical_training_performed": 1,
            **NO_WORK,
        }
    )


def candidate_path_risk_table(
    model: nn.Module,
    inputs: HostInputStore,
    labels: HostLabelStore,
    *,
    expected_path_ids: Sequence[int],
    device: str | torch.device,
    selected_outer_steps: Sequence[int] | None = None,
) -> _v3_selection.ZeroBaselineRiskTableV3:
    if inputs.role != "validation" or labels.role != "validation":
        raise Frequency1CoordinateWorkflowError("candidate evaluation requires validation role")
    if labels.purpose != "validation_selection" or inputs.row_count != labels.row_count:
        raise Frequency1CoordinateWorkflowError("validation label authorization is invalid")
    prediction, _ = predict_to_cpu(model, inputs, device=device, batch_size=32)
    target = np.ascontiguousarray(labels.row_array("denoising_target"), dtype=np.float64)
    kwargs: dict[str, Any] = {}
    if selected_outer_steps is not None:
        kwargs["selected_outer_steps"] = tuple(int(value) for value in selected_outer_steps)
    return aggregate_zero_baseline_risks(
        sample_keys=np.asarray(inputs.row_array("sample_key"), dtype=np.int64),
        row_path_ids=np.asarray(inputs.row_array("path_id"), dtype=np.int64),
        outer_steps=np.asarray(inputs.row_array("outer_step"), dtype=np.int64),
        phases=np.asarray(inputs.row_array("phase"), dtype=np.int64),
        midpoint_indices=np.asarray(inputs.row_array("midpoint_index"), dtype=np.int64),
        targets=target,
        predictions=prediction,
        expected_path_ids=np.asarray(expected_path_ids, dtype=np.int64),
        **kwargs,
    )


def scientific_config(
    *,
    test_only: bool = False,
    test_maximum_updates: int = 0,
    test_bootstrap_replicates: int = 8,
) -> dict[str, Any]:
    checkpoint = checkpoint_plan(
        test_only=test_only, test_maximum_updates=test_maximum_updates
    )
    selection = selection_inference_plan(
        test_only=test_only,
        test_replicates=test_bootstrap_replicates,
        checkpoint_record=checkpoint,
    )
    return _semantic(
        {
            "schema": (TEST_RUN_SCHEMA if test_only else RUN_SCHEMA) + "-scientific-config",
            "schema_version": 1,
            "test_only": int(test_only),
            "authorizing": int(not test_only),
            "grid_size": 28,
            "alpha": 1.0,
            "jacobi_outer_steps": 512,
            "tau_eff": 5.0e-5,
            "source_dataset_index": 7,
            "source_label": 3,
            "lambda_mix": 0.35,
            "selected_outer_steps": list(range(15, 512, 16)),
            "midpoint_fractions": [value / 16 for value in range(1, 16, 2)],
            "target": "exact raw binary64 Jacobi Rao-Blackwell label",
            "objective": "plain unweighted MSE divided only by physical-train target RMS squared",
            "boundary_representation": "m_theta(W)=y(1-y)q_theta(W), exact zero baseline",
            "only_scientific_change": "rank-4 periodic frequency-one output-site coordinates in spatial branch",
            "coordinate_cache_channels": 0,
            "width": 32,
            "effective_receptive_field": [7, 7],
            "training": {**TRAINING, "maximum_updates": checkpoint["maximum_updates"]},
            "checkpoint_plan_sha256": checkpoint["semantic_sha256"],
            "selection_plan_sha256": selection["semantic_sha256"],
            "component_count": COMPONENT_COUNT,
            "production_search_family_size": SEARCH_FAMILY_SIZE,
            "family_names_sha256": FAMILY_NAMES_SHA256,
            "search_family_names_sha256": SEARCH_FAMILY_NAMES_SHA256,
            "physical_training_uses_validation_labels": 0,
            "confirmation_reselection_allowed": 0,
            "target_clipping_or_weighting": 0,
            "early_stopping": 0,
            **NO_WORK,
        }
    )


__all__ = [
    "CANDIDATE_COUNT",
    "CHECKPOINT_UPDATES",
    "COMPONENT_COUNT",
    "CONFIRMATION_BOOTSTRAP_SEED",
    "CONFIRMATION_NAMESPACE",
    "CONFIRMATION_PATH_IDS",
    "EXACT_MODEL_NULL_SEED",
    "FAMILY_NAMES",
    "FAMILY_NAMES_SHA256",
    "FORBIDDEN_SCHEDULER_SEED",
    "FREQUENCY1_COORDINATE_LEARNABILITY_VERSION",
    "Frequency1CandidateTable",
    "Frequency1CoordinateWorkflowError",
    "INITIALIZATION_CONTROL_SEED",
    "MODEL_SEEDS",
    "NO_WORK",
    "NONZERO_UPDATES",
    "PATH_ROLES",
    "PHYSICAL_STAGES",
    "PREFLIGHT_PATH_IDS",
    "REQUIRED_GATES",
    "RESERVED_FUTURE_CONTROL_SEED",
    "ROOT_SEED",
    "RUN_SCHEMA",
    "SEARCH_FAMILY_NAMES",
    "SEARCH_FAMILY_NAMES_SHA256",
    "SELECTED_OUTER_STEPS",
    "SEARCH_FAMILY_SIZE",
    "SELECTION_BOOTSTRAP_SEED",
    "SELECTION_NAMESPACE",
    "STAGES",
    "STAGE_PREDECESSOR",
    "STAGE_SEAL_NAMES",
    "SYNTHETIC_COORDINATE_TEACHER_SEED",
    "TEST_RUN_SCHEMA",
    "TRAINING",
    "TRAIN_PATH_IDS",
    "VALIDATION_PATH_IDS",
    "aggregate_zero_baseline_improvements",
    "aggregate_zero_baseline_risks",
    "assert_role_firewall",
    "build_candidate_table",
    "build_cohort_plan",
    "build_path_plan",
    "candidate_path_risk_table",
    "checkpoint_plan",
    "confirmation_inference_plan",
    "eager_cohorts",
    "gate_passed",
    "load_bootstrap_count_shards",
    "load_json",
    "open_label_role",
    "prepare_bootstrap_count_shards",
    "rank_validation_nominee",
    "restartable_confirmation_max_t",
    "restartable_selection_max_t",
    "scientific_config",
    "seal_artifacts",
    "seed_plan",
    "selection_inference_plan",
    "train_frequency1_coordinate_candidate",
    "validate_stage_entry",
    "verify_artifact_seal",
]
