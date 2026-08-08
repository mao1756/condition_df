"""Controls-only exact Jacobi/Rao--Blackwell reverse-controller gate.

The workflow binds the sealed one-image coarse-residual checkpoint and tests
only the local reverse generator that it implies.  It never runs a complete
terminal-to-data reverse trajectory, never creates an image, and never
changes the certified Jacobi transition or the raw Rao--Blackwell label.

Production stages are intentionally bounded and have no scientific override
flags.  ``preflight`` binds immutable evidence and controller algebra,
``oracle`` runs analytic null/teacher controls, ``cache`` opens one fresh
physical panel and streams internal-time raw-risk statistics, ``control``
runs at most eight reverse phase occurrences, and ``decide`` adjudicates the
already frozen gates.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_json,
    config_fingerprint,
    configure_exact_torch_backend,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_rb_cuda import (
    JacobiRBCudaProfile,
    sample_alpha1_rb_transition_batch_cuda,
)
from mnist.d0_jacobi_rb_cuda_multipath import (
    EDGES_PER_PHASE,
    PATH_STATE_SIZE,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    SHARD_STEPS,
    run_exact_multipath_shard,
)
from mnist.d0_jacobi_rb_learnability import (
    FORBIDDEN_MODEL_INPUT_FIELDS,
    MODEL_INPUT_FIELDS,
    ModelInputs,
    selected_reverse_time,
)
from mnist import d0_jacobi_rb_cuda_controls as _cuda_controls
from mnist import diag_d0_jacobi_rb_coarse_residual_learnability as _parent_cli
from mnist import d0_jacobi_rb_reverse_controller as _controller


RUN_SCHEMA = "experiment12-d0-jacobi-rb-reverse-controller-control"
RUN_SCHEMA_VERSION = 1
NAMESPACE_VERSION = "d0-jacobi-rb-reverse-controller-control-v1"
OUTER_STEPS = 512
SELECTED_OUTER_STEPS = tuple(range(15, OUTER_STEPS, 16))
CONTROL_ANCHORS = (127, 255, 383, 511)
PHASE_OCCURRENCES = (0, 1, 2, 3, 2, 1, 0)
PRODUCTION_M = 8
REFINEMENT_M = (2, 4, 8)
CONTROLLER_ROOT_SEED = 261_301
LOCAL_BOOTSTRAP_SEED = 261_302
TRAJECTORY_BOOTSTRAP_SEED = 261_303
ORACLE_ROOT_SEED = 261_304
BOOTSTRAP_REPLICATES = 50_000
SIMULTANEOUS_CONFIDENCE = 0.995
PREFLIGHT_PATH_IDS = tuple(range(0xEA000, 0xEA008))
PHYSICAL_PATH_IDS = tuple(range(0xEB000, 0xEB040))
ORACLE_PATH_IDS = tuple(range(0xEE000, 0xEE020))
RESERVED_FUTURE_PATH_IDS = {
    "fresh_selection": (0xEC000, 0xEC040),
    "fresh_confirmation": (0xED000, 0xED040),
    "production": (0xF0000, 0x100000),
}

EXPECTED_PARENT_REGISTRY_FILE_SHA256 = (
    "45408753658b575d2bab52e3a1e991d97c79082923b5698c4057e1893e0d930a"
)
EXPECTED_PARENT_REGISTRY_SEMANTIC_SHA256 = (
    "308c6452158c1198fd6bb0b7996eeb97a708913baa36b91531fd8e5e3af2c291"
)
EXPECTED_PARENT_REGISTRY_COUNT = 3471
EXPECTED_PARENT_SOURCE_FINGERPRINT = (
    "42b7129b8850d5e4036137e1781799fe0d9b37d8ee98867d3e1a7b7a57b7906c"
)
EXPECTED_PARENT_SCIENTIFIC_CONFIG_SHA256 = (
    "b49f50be2f414b5c5a7c402850ac40aa2e6d129325116e13033acd3884de3378"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "24a0893daa31196815463a7396220542003e7dc2557689950ba4dd0eeaa9c914"
)
EXPECTED_STATE_DICT_SHA256 = (
    "df479e979cf6dd99580bd918377405b665791a4608f45f6cae326cc10e5e6ad9"
)
EXPECTED_BASELINE_SHA256 = (
    "5d4e73153c36a59e26403439befd4e13b7f4fe096f7cbf9af6b77ac26565a9df"
)
EXPECTED_SHRINKAGE = 0.2910413880506186
EXPECTED_SEED = 261_254
EXPECTED_UPDATE = 3000
EXPECTED_IMAGE_SHA256 = (
    "0bb39fec59853f789fe366251cd85ed79ffbffb5a1aaa32084d2dbd2bbb4ea7d"
)
EXPECTED_MIXED_TARGET_SHA256 = (
    "00ae86fb69be6d86557f15f6f8fa00f8bb3c2514f331863c9638e36d23d135c5"
)

MAIN_TRANSITIONS = 89_915_392
LOCAL_BRANCH_TRANSITIONS = 44_957_696
ONE_PHASE_CONTROL_TRANSITIONS = 19_668_992
EIGHT_PHASE_CONTROL_TRANSITIONS = 22_478_848
TOTAL_TRANSITIONS = 177_020_928
RESOURCE_THRESHOLDS = {
    "minimum_transitions_per_second": 1300.0,
    "maximum_projected_hours": 40.0,
    "maximum_peak_memory_fraction": 0.80,
    "maximum_persisted_bytes": 512 * 1024**2,
    "maximum_mass_error": 2.0e-12,
    "maximum_fallback_fraction": 1.0e-4,
    "maximum_fallback_time_fraction": 0.10,
}
FORBIDDEN_DIAGNOSTICS = (
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
CLAIM_BOUNDARY = {
    "reverse_sampling_authorized": 0,
    "sampling_authorized": 0,
    "reconstruction_authorized": 0,
    "reconstruction_claim_authorized": 0,
    "known_prior_claim_authorized": 0,
    "full_dataset_training_authorized": 0,
    "unsplit_generator_claim_authorized": 0,
    "spatial_dirichlet_ferguson_claim_authorized": 0,
    "reverse_sampling_performed": 0,
    "sampling_performed": 0,
    "image_sampling_performed": 0,
    "reconstruction_performed": 0,
    "full_reverse_path_performed": 0,
}
STAGES = ("preflight", "oracle", "cache", "control", "decide", "report", "all")
REQUIRED_GATES = ("none", "preflight", "oracle", "cache", "control", "decide")
_REGISTRY_EXCLUDED = {"artifact_registry.json", "run_status.json"}
_DECISION_GATE_CHECKS = {
    "preflight": frozenset(
        {
            "parent_provenance",
            "package_manifest",
            "path_namespace",
            "endpoint_equivalence",
            "formula_controls",
            "negative_controls",
            "boundary_controls",
            "model_input_firewall",
            "restart_grouping_invariance",
            "path_order_invariance",
            "interrupted_resume_invariance",
            "controller_grouping_invariance",
            "controller_path_order_invariance",
            "controller_phase_restart_invariance",
            "pair_mass_preservation",
            "simplex_mass_preservation",
            "controlled_states_finite",
            "controlled_states_nonnegative",
            "reference_launch_cap",
            "certificate_fraction",
            "fallback_fraction",
            "fallback_time",
            "throughput",
            "projected_time",
            "memory",
            "persisted_size",
            "forbidden_counts",
            "no_physical_panel_opened",
        }
    ),
    "oracle": frozenset(
        {
            "stationary_null",
            "stationary_null_bitwise_reference_composition",
            "bounded_linear_teacher",
            "teacher_M2_M4_M8_complete",
            "teacher_boundary_finite",
            "certificate_fraction",
            "fallback_fraction",
            "fallback_time",
            "forbidden_counts",
            "physical_panel_unopened",
        }
    ),
    "cache": frozenset(
        {
            "main_paths_complete",
            "checkpoint_hash_ledger",
            "main_transition_count",
            "branch_transition_count",
            "local_family_size",
            "local_all_simultaneous_lower_positive",
            "terminal_near_reverse_start_controlled",
            "certificate_fraction",
            "fallback_fraction",
            "fallback_time",
            "mass_error",
            "branch_health",
            "forbidden_counts",
            "throughput",
            "persisted_size",
            "no_joined_cache",
            "target_immutable",
        }
    ),
    "control": frozenset(
        {
            "trajectory_family_size",
            "one_phase_reverse_law",
            "eight_phase_reverse_law",
            "M8_refinement",
            "structural_invariants",
            "states_finite",
            "states_nonnegative",
            "pair_mass",
            "simplex_mass",
            "boundary_rejections",
            "reference_launch_cap",
            "certificate_fraction",
            "fallback_fraction",
            "fallback_time",
            "forbidden_counts",
            "controller_forbidden_counts",
            "throughput",
            "peak_memory",
            "persisted_size",
            "maximum_phase_count",
            "no_full_reverse_path",
            "no_image_artifacts",
        }
    ),
}


class ReverseControllerCLIError(RuntimeError):
    """Typed fail-closed workflow error."""

    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "workflow_execution",
        failure_code: str = "reverse_controller_execution_failed",
    ) -> None:
        super().__init__(message)
        self.failure_domain = str(failure_domain)
        self.failure_code = str(failure_code)


class ParentBindingError(ArtifactCompatibilityError):
    failure_domain = "provenance"
    failure_code = "reverse_controller_parent_provenance_invalid"


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"JSON artifact is not an object: {path}")
    return dict(value)


def _normalized(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))


def _freeze_json(path: Path, value: Mapping[str, Any], *, require_existing: bool = False) -> dict[str, Any]:
    record = _normalized(value)
    if path.is_file():
        if _load_json(path) != record:
            raise ArtifactCompatibilityError(f"frozen artifact changed: {path.name}")
    elif require_existing:
        raise ArtifactCompatibilityError(f"resume lacks frozen artifact: {path.name}")
    else:
        atomic_write_json(path, record)
    return record


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {str(k): np.ascontiguousarray(np.asarray(v)) for k, v in sorted(arrays.items())}
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)
    return {"path": path.as_posix(), "sha256": file_fingerprint(path), "size": path.stat().st_size}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ArtifactCompatibilityError(f"cannot read NPZ artifact {path}: {exc}") from exc


def _load_control_audit(
    run_dir: Path,
    *,
    anchor: int,
    phase: int | None,
) -> dict[str, np.ndarray]:
    """Load and semantically validate a frozen forward control endpoint pair."""

    if anchor not in CONTROL_ANCHORS or (phase is not None and phase not in range(7)):
        raise ArtifactCompatibilityError("control-audit coordinate is outside the frozen plan")
    filename = (
        f"one-phase-anchor-{anchor:04d}-phase-{phase}.npz"
        if phase is not None
        else f"eight-phase-anchor-{anchor:04d}.npz"
    )
    arrays = _load_npz(run_dir / "control" / "audit" / filename)
    if set(arrays) != {"earlier_state", "later_state", "path_ids"}:
        raise ArtifactCompatibilityError(
            f"control audit {filename} has an invalid field set"
        )
    path_ids = np.asarray(arrays["path_ids"])
    earlier = np.asarray(arrays["earlier_state"])
    later = np.asarray(arrays["later_state"])
    expected_shape = (len(PHYSICAL_PATH_IDS), PATH_STATE_SIZE)
    if (
        path_ids.dtype != np.dtype(np.int64)
        or path_ids.shape != (len(PHYSICAL_PATH_IDS),)
        or not np.array_equal(path_ids, np.asarray(PHYSICAL_PATH_IDS, dtype=np.int64))
    ):
        raise ArtifactCompatibilityError(
            f"control audit {filename} does not bind the frozen physical path order"
        )
    if (
        earlier.dtype != np.dtype(np.float64)
        or later.dtype != np.dtype(np.float64)
        or earlier.shape != expected_shape
        or later.shape != expected_shape
        or not np.isfinite(earlier).all()
        or not np.isfinite(later).all()
        or np.any(earlier < 0.0)
        or np.any(later < 0.0)
        or float(np.max(np.abs(earlier.sum(axis=1) - 1.0)))
        > RESOURCE_THRESHOLDS["maximum_mass_error"]
        or float(np.max(np.abs(later.sum(axis=1) - 1.0)))
        > RESOURCE_THRESHOLDS["maximum_mass_error"]
    ):
        raise ArtifactCompatibilityError(
            f"control audit {filename} has invalid state semantics"
        )
    if phase is not None:
        tails, heads = _cuda_controls._matching_arrays()[  # noqa: SLF001
            PHASE_MATCHINGS[phase]
        ]
        earlier_pair = earlier[:, tails] + earlier[:, heads]
        later_pair = later[:, tails] + later[:, heads]
        if float(np.max(np.abs(earlier_pair - later_pair))) > RESOURCE_THRESHOLDS[
            "maximum_mass_error"
        ]:
            raise ArtifactCompatibilityError(
                f"control audit {filename} violates phase pair-mass conservation"
            )
    return {
        "earlier_state": np.ascontiguousarray(earlier),
        "later_state": np.ascontiguousarray(later),
        "path_ids": np.ascontiguousarray(path_ids),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows([dict(row) for row in rows])
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _passed(record: Mapping[str, Any] | None) -> bool:
    return bool(record and record.get("evaluation_status") == "evaluated" and int(record.get("passed", 0)) == 1)


def _contains_forbidden_metric_name(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(str(key) == "data_end" or _contains_forbidden_metric_name(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_metric_name(item) for item in value)
    return False


def _assert_unambiguous_metric_schema(value: Any) -> None:
    try:
        _controller.assert_unambiguous_metric_schema(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ReverseControllerCLIError(
            "new reverse-controller schemas may not contain the ambiguous data_end key",
            failure_domain="schema",
            failure_code="ambiguous_time_metric_name",
        ) from exc


def _semantic_hash(value: Mapping[str, Any]) -> str:
    return config_fingerprint(_normalized(value))


def _artifact_registry_record(run_dir: Path) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in _REGISTRY_EXCLUDED or ".tmp" in path.name:
            continue
        artifacts.append({
            "path": path.relative_to(run_dir).as_posix(),
            "sha256": file_fingerprint(path),
            "size": int(path.stat().st_size),
        })
    semantic = config_fingerprint({"artifacts": artifacts})
    record = {
        "schema": RUN_SCHEMA + "-artifact-registry",
        "schema_version": 1,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "semantic_sha256": semantic,
        **CLAIM_BOUNDARY,
    }
    return record


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    record = _artifact_registry_record(run_dir)
    atomic_write_json(run_dir / "artifact_registry.json", record)
    return record


def _verify_registry(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "artifact_registry.json"
    if not path.is_file():
        return None
    expected = _load_json(path)
    artifacts = expected.get("artifacts")
    if not isinstance(artifacts, list) or int(expected.get("artifact_count", -1)) != len(artifacts):
        raise ArtifactCompatibilityError("artifact registry structure is invalid")
    if expected.get("semantic_sha256") != config_fingerprint({"artifacts": artifacts}):
        raise ArtifactCompatibilityError("artifact registry semantic hash changed")
    for item in artifacts:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise ArtifactCompatibilityError("artifact registry entry is malformed")
        artifact = run_dir / str(item["path"])
        if (
            not artifact.is_file()
            or file_fingerprint(artifact) != item.get("sha256")
            or artifact.stat().st_size != int(item.get("size", -1))
        ):
            raise ArtifactCompatibilityError(
                f"registered artifact changed: {item.get('path')}"
            )
    # Extra files may be unregistered partial shards from an interrupted stage;
    # stage-specific recovery validates or atomically replaces them.
    return expected


def _verify_terminal_registry_exact(run_dir: Path) -> dict[str, Any]:
    expected = _load_json(run_dir / "artifact_registry.json")
    actual = _artifact_registry_record(run_dir)
    if expected != actual:
        raise ArtifactCompatibilityError("terminal artifact registry is not exact")
    return expected


def _registered_stage_gate(
    run_dir: Path, gate_path: Path
) -> dict[str, Any] | None:
    """Return only gates committed by the last sealed artifact registry."""

    if not gate_path.is_file():
        return None
    relative = gate_path.relative_to(run_dir).as_posix()
    registry_path = run_dir / "artifact_registry.json"
    registered = False
    if registry_path.is_file():
        registry = _load_json(registry_path)
        artifacts = registry.get("artifacts", ())
        registered = any(
            isinstance(item, Mapping)
            and item.get("path") == relative
            and item.get("sha256") == file_fingerprint(gate_path)
            and int(item.get("size", -1)) == gate_path.stat().st_size
            for item in artifacts
        )
    if registered:
        return _load_json(gate_path)
    # The gate was written after the last registry seal.  It is an orphan,
    # not committed evidence; remove just this mutable child marker so the
    # stage revalidates/reuses its independently checked supporting artifacts.
    gate_path.unlink()
    return None


def _artifact_is_registered(run_dir: Path, artifact: Path) -> bool:
    registry_path = run_dir / "artifact_registry.json"
    if not artifact.is_file() or not registry_path.is_file():
        return False
    relative = artifact.relative_to(run_dir).as_posix()
    registry = _load_json(registry_path)
    return any(
        isinstance(item, Mapping)
        and item.get("path") == relative
        and item.get("sha256") == file_fingerprint(artifact)
        and int(item.get("size", -1)) == artifact.stat().st_size
        for item in registry.get("artifacts", ())
    )


def _commit_recoverable_json(
    run_dir: Path,
    path: Path,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = _normalized(value)
    if path.is_file() and _load_json(path) != normalized:
        if _artifact_is_registered(run_dir, path):
            raise ArtifactCompatibilityError(
                f"registered artifact changed: {path.relative_to(run_dir).as_posix()}"
            )
        atomic_write_json(path, normalized)
        return normalized
    return _freeze_json(path, normalized)


def _gate_record(name: str, checks: Mapping[str, bool], **evidence: Any) -> dict[str, Any]:
    passed = int(bool(checks) and all(bool(value) for value in checks.values()))
    record = {
        "schema": RUN_SCHEMA + f"-{name}-gate",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "gate": name,
        "checks": {str(k): int(bool(v)) for k, v in checks.items()},
        "passed": passed,
        "numerically_valid": int(bool(evidence.pop("numerically_valid", passed))),
        "resource_valid": int(bool(evidence.pop("resource_valid", passed))),
        **evidence,
        **CLAIM_BOUNDARY,
    }
    _assert_unambiguous_metric_schema(record)
    return record


def _not_evaluated(name: str, reason: str) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + f"-{name}-gate",
        "schema_version": 1,
        "evaluation_status": "not_evaluated",
        "gate": name,
        "reason": str(reason),
        "passed": 0,
        **CLAIM_BOUNDARY,
    }


def _status(run_dir: Path, *, state: str, stage: str, decision: str | None = None, message: str | None = None, failure_domain: str | None = None, failure_code: str | None = None) -> dict[str, Any]:
    control_gate = (
        _load_json(run_dir / "control_gate.json")
        if (run_dir / "control_gate.json").is_file()
        else None
    )
    control_performed = int(
        isinstance(control_gate, Mapping)
        and control_gate.get("evaluation_status") == "evaluated"
        and int(control_gate.get("controller_control_trajectory_performed", 0)) == 1
    )
    control_phase_count = (
        int(control_gate.get("maximum_control_trajectory_phase_count", 0))
        if control_performed and isinstance(control_gate, Mapping)
        else 0
    )
    committed_decision = (
        _load_json(run_dir / "controller_decision.json")
        if (run_dir / "controller_decision.json").is_file()
        else {}
    )
    planning_authorized = int(
        committed_decision.get("one_image_reconstruction_planning_authorized", 0)
    )
    record = {
        "schema": RUN_SCHEMA + "-status",
        "schema_version": 1,
        "state": str(state),
        "stage": str(stage),
        "decision": decision,
        "message": message,
        "failure_domain": failure_domain,
        "failure_code": failure_code,
        "controller_control_trajectory_performed": control_performed,
        "maximum_control_trajectory_phase_count": control_phase_count,
        "one_image_reconstruction_planning_authorized": planning_authorized,
        "updated_at": _now(),
        **CLAIM_BOUNDARY,
    }
    atomic_write_json(run_dir / "run_status.json", record)
    return record


def _source_paths(parent_run: Path) -> tuple[Path, ...]:
    module_paths = (
        Path(__file__),
        Path(_controller.__file__),
        Path(_parent_cli.__file__),
    )
    inherited = tuple(
        Path(path)
        for path in _load_json(parent_run.resolve() / "run_manifest.json").get("source_paths", ())
        if Path(path).is_file()
    )
    return tuple(sorted({path.resolve() for path in (*module_paths, *inherited)}))


def _path_plan() -> dict[str, Any]:
    core_plan = _controller.validate_controller_path_plan()
    record = {
        "schema": RUN_SCHEMA + "-path-id-plan",
        "schema_version": 1,
        "namespace_version": NAMESPACE_VERSION,
        "roles": {
            "preflight": list(PREFLIGHT_PATH_IDS),
            "physical_control": list(PHYSICAL_PATH_IDS),
            "oracle": list(ORACLE_PATH_IDS),
        },
        "reserved_roles": {
            name: [int(bounds[0]), int(bounds[1])]
            for name, bounds in RESERVED_FUTURE_PATH_IDS.items()
        },
        "core_plan_sha256": _semantic_hash(core_plan),
        "canonical_field_bits": 20,
        "collision_free": int(core_plan.get("collision_free", 0)),
        "physical_cohorts": [
            list(PHYSICAL_PATH_IDS[index : index + 8])
            for index in range(0, len(PHYSICAL_PATH_IDS), 8)
        ],
        "selected_outer_steps": list(SELECTED_OUTER_STEPS),
        "control_anchors": list(CONTROL_ANCHORS),
    }
    future_explicit = [
        *range(*RESERVED_FUTURE_PATH_IDS["fresh_selection"]),
        *range(*RESERVED_FUTURE_PATH_IDS["fresh_confirmation"]),
    ]
    flat = [
        *PREFLIGHT_PATH_IDS,
        *PHYSICAL_PATH_IDS,
        *ORACLE_PATH_IDS,
        *future_explicit,
    ]
    if len(set(flat)) != len(flat) or any(not 0 <= value < 2**20 for value in flat):
        raise ReverseControllerCLIError(
            "controller path IDs collide or exceed the 20-bit field",
            failure_domain="namespace",
            failure_code="controller_path_namespace_invalid",
        )
    record["semantic_sha256"] = _semantic_hash(record)
    return record


def _integers_under_path_keys(value: Any, *, under_path_key: bool = False) -> Iterable[int]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _integers_under_path_keys(
                item,
                under_path_key=under_path_key or "path" in str(key).lower(),
            )
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _integers_under_path_keys(item, under_path_key=under_path_key)
    elif under_path_key and isinstance(value, int) and not isinstance(value, bool):
        yield int(value)


def _semantic_path_collision_scan(run_dir: Path) -> dict[str, Any]:
    proposed = set(
        (
            *PREFLIGHT_PATH_IDS,
            *PHYSICAL_PATH_IDS,
            *ORACLE_PATH_IDS,
            *range(*RESERVED_FUTURE_PATH_IDS["fresh_selection"]),
            *range(*RESERVED_FUTURE_PATH_IDS["fresh_confirmation"]),
        )
    )
    collisions: list[dict[str, Any]] = []
    scanned: list[str] = []
    workspace = Path.cwd().resolve()
    excluded = {
        Path(__file__).resolve(),
        Path(_controller.__file__).resolve(),
        (workspace / "docs/jacobi_rb_reverse_controller_control.md").resolve(),
    }
    # Historical frozen path plans are the strongest semantic evidence.
    for path in sorted((workspace / "runs").rglob("*path*plan*.json")):
        if run_dir in path.parents or not path.is_file():
            continue
        scanned.append(path.relative_to(workspace).as_posix())
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for identifier in _integers_under_path_keys(value):
            if identifier in proposed:
                collisions.append({"path": path.relative_to(workspace).as_posix(), "path_id": identifier, "representation": "json_path_field"})
    # Source/document declarations conventionally use hexadecimal path blocks.
    for root in (workspace / "mnist", workspace / "docs"):
        patterns = ("*.py",) if root.name == "mnist" else ("*.md", "*.tex")
        for pattern in patterns:
            for path in sorted(root.rglob(pattern)):
                if path.resolve() in excluded:
                    continue
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                found = False
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if "path" not in line.lower() and "0x" not in line.lower():
                        continue
                    for token in re.findall(r"0x[0-9a-fA-F]+", line):
                        identifier = int(token, 16)
                        if identifier in proposed:
                            collisions.append({"path": path.relative_to(workspace).as_posix(), "line": line_number, "path_id": identifier, "representation": token})
                        found = True
                if found:
                    scanned.append(path.relative_to(workspace).as_posix())
    record = {
        "schema": RUN_SCHEMA + "-semantic-path-collision-scan",
        "schema_version": 1,
        "proposed_path_count": len(proposed),
        "scanned_files": sorted(set(scanned)),
        "collision_count": len(collisions),
        "collisions": collisions,
        "silent_remapping_performed": 0,
        "passed": int(not collisions),
    }
    if collisions:
        raise ReverseControllerCLIError(
            "controller path IDs collide with an existing semantic declaration",
            failure_domain="namespace",
            failure_code="controller_path_namespace_collision",
        )
    return record


def _transition_namespace_record() -> dict[str, Any]:
    roles = tuple(_controller.TRANSITION_ROLES)
    required = {
        "partial_phase_target_prefix",
        *{
            f"reverse_reference_{side}_control_M{microsteps}"
            for side in ("pre", "post")
            for microsteps in REFINEMENT_M
        },
        "analytic_teacher_exact_reverse",
    }
    if not required.issubset(set(roles)):
        raise ReverseControllerCLIError(
            "core transition namespace lacks a frozen role",
            failure_domain="namespace",
            failure_code="controller_transition_role_missing",
        )
    return {
        "schema": RUN_SCHEMA + "-transition-namespace",
        "schema_version": 1,
        "namespace_version": NAMESPACE_VERSION,
        "root_seeds": {
            "controller": CONTROLLER_ROOT_SEED,
            "local_bootstrap": LOCAL_BOOTSTRAP_SEED,
            "trajectory_bootstrap": TRAJECTORY_BOOTSTRAP_SEED,
            "oracle": ORACLE_ROOT_SEED,
        },
        "roles": list(roles),
        "state_independent_randomness": 1,
        "transition_randomness_forbidden_as_model_input": 1,
    }


def _model_input_contract() -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + "-model-input-contract",
        "schema_version": 1,
        "permitted_fields": [
            "later_full_state",
            "reverse_time",
            "phase",
            "color",
            "duration",
            "label",
        ],
        "forbidden_fields": [
            "earlier_state",
            "path_id",
            "outer_step",
            "sample_key",
            "transition_randomness",
            "later_head_fraction",
            "certificate_data",
            "denoising_target",
            "oracle_target",
            "witness_identity",
            "branch_identity",
        ],
        "coarse_lookup_uses_only_reverse_time_and_phase": 1,
        "audit_stream_physically_separated": 1,
        "target_modified": 0,
    }


def _controller_convention() -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + "-controller-convention",
        "schema_version": 1,
        "alpha": 1.0,
        "grid_spacing": 1.0 / 28.0,
        "sample_steps": OUTER_STEPS,
        "tau_eff": 5.0e-5,
        "macrostep_schedule_integral": 5.0e-5 / OUTER_STEPS,
        "phase_occurrences": list(PHASE_OCCURRENCES),
        "phase_durations": list(PHASE_DURATIONS),
        "reverse_execution": {"outer_steps": "511..0", "phases": "6..0"},
        "controller_microsteps_per_phase": PRODUCTION_M,
        "refinement_control_microsteps": list(REFINEMENT_M),
        "microstep_split": "exact-reference-half / frozen-affine-control / exact-reference-half",
        "fraction_flow": "y_plus = y + 2*m_mid*delta_u",
        "outward_boundary_action": "reject",
        "clipping": 0,
        "floor": 0,
        "limiter": 0,
        "projection": 0,
        "renormalization": 0,
        "full_reverse_path_performed": 0,
    }


def _scientific_config() -> dict[str, Any]:
    record = {
        "schema": RUN_SCHEMA + "-scientific-config",
        "schema_version": 1,
        "authorizing": 1,
        "parent_registry_file_sha256": EXPECTED_PARENT_REGISTRY_FILE_SHA256,
        "parent_registry_semantic_sha256": EXPECTED_PARENT_REGISTRY_SEMANTIC_SHA256,
        "selected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "selected_state_dict_sha256": EXPECTED_STATE_DICT_SHA256,
        "frozen_baseline_sha256": EXPECTED_BASELINE_SHA256,
        "selected_seed": EXPECTED_SEED,
        "selected_update": EXPECTED_UPDATE,
        "global_shrinkage": EXPECTED_SHRINKAGE,
        "K": OUTER_STEPS,
        "grid_spacing": 1.0 / 28.0,
        "alpha": 1.0,
        "edges_per_phase": EDGES_PER_PHASE,
        "phase_occurrences": list(PHASE_OCCURRENCES),
        "phase_durations": list(PHASE_DURATIONS),
        "tau_eff": 5.0e-5,
        "production_microsteps": PRODUCTION_M,
        "refinement_microsteps": list(REFINEMENT_M),
        "physical_paths": len(PHYSICAL_PATH_IDS),
        "selected_outer_steps": list(SELECTED_OUTER_STEPS),
        "control_anchors": list(CONTROL_ANCHORS),
        "local_family_size": 228,
        "trajectory_family_size": 784,
        "simultaneous_confidence": SIMULTANEOUS_CONFIDENCE,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "resource_thresholds": dict(RESOURCE_THRESHOLDS),
        "transition_counts": {
            "main": MAIN_TRANSITIONS,
            "internal_prefix": LOCAL_BRANCH_TRANSITIONS,
            "one_phase_control": ONE_PHASE_CONTROL_TRANSITIONS,
            "eight_phase_control": EIGHT_PHASE_CONTROL_TRANSITIONS,
            "total": TOTAL_TRANSITIONS,
        },
        "path_plan_sha256": _path_plan()["semantic_sha256"],
        "namespace_sha256": _semantic_hash(_transition_namespace_record()),
        "target": "unchanged binary64 Rao-Blackwell L-MY",
        "target_modified": 0,
        **CLAIM_BOUNDARY,
    }
    record["semantic_sha256"] = _semantic_hash(record)
    return record


def _verify_package_manifest() -> dict[str, Any]:
    root = Path("handoff/jacobi_rb_coarse_residual_next_decision_20260801").resolve()
    manifest = root / "PACKAGE_MANIFEST.sha256"
    readme = root / "README_FOR_CHATGPT_PRO.md"
    if not manifest.is_file() or not readme.is_file():
        raise ParentBindingError("minimal handoff package or README is missing")
    # The handoff contract explicitly requires this document to be opened
    # before any other packaged artifact is inspected.
    readme_text = readme.read_text(encoding="utf-8")
    if "controls-only" not in readme_text or "reverse" not in readme_text.lower():
        raise ParentBindingError("handoff README does not contain its claim boundary")
    rows: list[dict[str, Any]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        path = root / relative.strip()
        observed = file_fingerprint(path) if path.is_file() else None
        rows.append({"path": relative.strip(), "expected": digest, "observed": observed, "passed": int(observed == digest)})
    if not rows or not all(row["passed"] for row in rows):
        raise ParentBindingError("minimal handoff package manifest failed verification")
    return {
        "schema": RUN_SCHEMA + "-package-manifest-verification",
        "schema_version": 1,
        "package_root": str(root),
        "manifest_sha256": file_fingerprint(manifest),
        "readme_sha256": file_fingerprint(readme),
        "artifact_count": len(rows),
        "all_hashes_verified": 1,
        "readme_read_first_contract": 1,
    }


def _verify_parent(parent: Path) -> dict[str, Any]:
    parent = parent.resolve()
    try:
        status = _load_json(parent / "run_status.json")
        decision = _load_json(parent / "coarse_residual_decision.json")
        manifest = _load_json(parent / "run_manifest.json")
        config = _load_json(parent / "scientific_config.json")
        selected = _load_json(parent / "selected_model.json")
        baseline = _load_json(parent / "frozen_coarse_baseline.json")
        source_image = _load_json(parent / "source_image.json")
        source_arrays = _load_npz(parent / "source_image.npz")
        registry = _load_json(parent / "artifact_registry.json")
        parent_registry_verified = _parent_cli._verify_registry(parent)  # noqa: SLF001
    except (ArtifactCompatibilityError, OSError, ValueError) as exc:
        raise ParentBindingError(f"cannot verify parent: {exc}") from exc
    candidate = selected.get("candidate") if isinstance(selected.get("candidate"), Mapping) else {}
    checkpoint = parent / "selected_model.pt"
    checks = {
        "terminal_status": status.get("state") == "complete" and status.get("stage") == "confirm" and status.get("decision") == "exact_rb_coarse_residual_learnable",
        "decision": decision.get("decision") == "exact_rb_coarse_residual_learnable" and int(decision.get("reverse_controller_planning_authorized", 0)) == 1,
        "no_sampling": all(int(status.get(name, 1)) == 0 for name in ("sampling_performed", "reverse_sampling_performed", "reconstruction_performed")),
        "registry_file": file_fingerprint(parent / "artifact_registry.json") == EXPECTED_PARENT_REGISTRY_FILE_SHA256,
        "registry_count": int(registry.get("artifact_count", -1)) == EXPECTED_PARENT_REGISTRY_COUNT,
        "registry_semantic": registry.get("semantic_sha256") == EXPECTED_PARENT_REGISTRY_SEMANTIC_SHA256,
        "registry_verified": parent_registry_verified == registry,
        "source": manifest.get("source_fingerprint") == EXPECTED_PARENT_SOURCE_FINGERPRINT,
        "live_inherited_source": source_fingerprint(
            tuple(Path(path).resolve() for path in manifest.get("source_paths", ()))
        ) == EXPECTED_PARENT_SOURCE_FINGERPRINT,
        "scientific_config": config.get("semantic_sha256") == EXPECTED_PARENT_SCIENTIFIC_CONFIG_SHA256,
        "checkpoint": checkpoint.is_file() and file_fingerprint(checkpoint) == EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_record": selected.get("selected_model_sha256") == EXPECTED_CHECKPOINT_SHA256 and candidate.get("checkpoint_file_sha256") == EXPECTED_CHECKPOINT_SHA256,
        "state": candidate.get("state_sha256") == EXPECTED_STATE_DICT_SHA256,
        "seed_update": int(candidate.get("seed", -1)) == EXPECTED_SEED and int(candidate.get("update", -1)) == EXPECTED_UPDATE,
        "baseline": baseline.get("values_c_order_sha256") == EXPECTED_BASELINE_SHA256 and float(baseline.get("shrinkage", math.nan)) == EXPECTED_SHRINKAGE,
        "target_immutable": int(baseline.get("target_modified", 1)) == 0,
        "image_identity": (
            source_image.get("image_sha256") == EXPECTED_IMAGE_SHA256
            and source_image.get("mixed_target_sha256")
            == EXPECTED_MIXED_TARGET_SHA256
            and source_image.get("npz_sha256")
            == file_fingerprint(parent / "source_image.npz")
            and set(source_arrays) == {"image", "mixed_target"}
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ParentBindingError("parent binding checks failed: " + ", ".join(failed))
    return {
        "schema": RUN_SCHEMA + "-parent-provenance",
        "schema_version": 1,
        "parent_run_dir": str(parent),
        "checks": {name: int(value) for name, value in checks.items()},
        "registry_file_sha256": EXPECTED_PARENT_REGISTRY_FILE_SHA256,
        "registry_semantic_sha256": EXPECTED_PARENT_REGISTRY_SEMANTIC_SHA256,
        "registry_count": EXPECTED_PARENT_REGISTRY_COUNT,
        "source_fingerprint": EXPECTED_PARENT_SOURCE_FINGERPRINT,
        "scientific_config_sha256": EXPECTED_PARENT_SCIENTIFIC_CONFIG_SHA256,
        "selected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "selected_state_dict_sha256": EXPECTED_STATE_DICT_SHA256,
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "image_sha256": EXPECTED_IMAGE_SHA256,
        "mixed_target_sha256": EXPECTED_MIXED_TARGET_SHA256,
        "transitive_parent_binding_preserved": 1,
        "passed": 1,
        **CLAIM_BOUNDARY,
    }


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        path = args.resume_run_dir.resolve()
        if not path.is_dir():
            raise ArtifactCompatibilityError(f"resume run does not exist: {path}")
        return path, True
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = (args.runs_root / f"{stamp}_{args.run_name}").resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path, False


def _initialize_run(run_dir: Path, args: argparse.Namespace, *, resumed: bool) -> None:
    parent = _verify_parent(args.parent_coarse_residual_run_dir)
    package = _verify_package_manifest()
    path_plan = _path_plan()
    namespace = _transition_namespace_record()
    contract = _model_input_contract()
    convention = _controller_convention()
    config = _scientific_config()
    collision_scan = _semantic_path_collision_scan(run_dir)
    sources = _source_paths(args.parent_coarse_residual_run_dir)
    source_hash = source_fingerprint(sources)
    records = {
        "parent_provenance.json": parent,
        "package_manifest_verification.json": package,
        "path_id_plan.json": path_plan,
        "transition_namespace.json": namespace,
        "model_input_contract.json": contract,
        "controller_convention.json": convention,
        "scientific_config.json": config,
        "selected_checkpoint_binding.json": {
            "schema": RUN_SCHEMA + "-selected-checkpoint-binding",
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "state_dict_sha256": EXPECTED_STATE_DICT_SHA256,
            "seed": EXPECTED_SEED,
            "update": EXPECTED_UPDATE,
        },
        "frozen_baseline_binding.json": {
            "schema": RUN_SCHEMA + "-frozen-baseline-binding",
            "values_c_order_sha256": EXPECTED_BASELINE_SHA256,
            "shrinkage": EXPECTED_SHRINKAGE,
            "signed_values_retained": 1,
            "refit_performed": 0,
        },
        "claim_constraints.json": {
            "schema": RUN_SCHEMA + "-claim-constraints",
            "conditional_planning_authorization_recorded_at_decide_only": 1,
            **CLAIM_BOUNDARY,
        },
        "path_id_collision_scan.json": collision_scan,
    }
    for filename, value in records.items():
        _freeze_json(run_dir / filename, value, require_existing=resumed)
    manifest = {
        "schema": RUN_SCHEMA + "-manifest",
        "schema_version": 1,
        "created_at": _load_json(run_dir / "run_manifest.json").get("created_at") if resumed and (run_dir / "run_manifest.json").is_file() else _now(),
        "parent_coarse_residual_run_dir": str(args.parent_coarse_residual_run_dir.resolve()),
        "device": args.device,
        "scientific_config_sha256": config["semantic_sha256"],
        "path_plan_sha256": path_plan["semantic_sha256"],
        "source_fingerprint": source_hash,
        "source_paths": [path.as_posix() for path in sources],
        **CLAIM_BOUNDARY,
    }
    _freeze_json(run_dir / "run_manifest.json", manifest, require_existing=resumed)
    if resumed:
        _verify_registry(run_dir)


def _tensor_scalar(value: Any, default: float = 0.0) -> float:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ReverseControllerCLIError("CUDA diagnostic is not scalar")
        return float(value.detach().cpu().item())
    return float(default if value is None else value)


class _CertifiedReference:
    """Exact CUDA reference callback plus streaming certificate diagnostics."""

    def __init__(self, *, root_seed: int, profile: JacobiRBCudaProfile, stream_role: str = "default") -> None:
        self.root_seed = int(root_seed)
        self.profile = profile
        self.stream_role = str(stream_role)
        self.transition_count = 0
        self.certified_count = 0
        self.fallback_count = 0
        self.fallback_seconds = 0.0
        self.elapsed_seconds = 0.0
        self.maximum_transition_count_per_call = 0
        self.forbidden = {name: 0 for name in FORBIDDEN_DIAGNOSTICS}

    def __call__(
        self,
        *,
        head_fraction: Tensor,
        exposure: Tensor,
        transition_ids: Tensor,
        role: str,
    ) -> Any:
        count = int(head_fraction.numel())
        self.maximum_transition_count_per_call = max(
            self.maximum_transition_count_per_call, count
        )
        if count > 4096:
            raise ReverseControllerCLIError(
                "certified reference launch exceeds 4096 transitions",
                failure_domain="resource",
                failure_code="controller_reference_launch_too_large",
            )
        started = time.perf_counter()
        result = sample_alpha1_rb_transition_batch_cuda(
            head_fraction.contiguous(),
            exposure.contiguous(),
            rng_key=(self.root_seed, NAMESPACE_VERSION, self.stream_role, str(role)),
            transition_ids=transition_ids.contiguous(),
            profile=self.profile,
        )
        if result.later_head_fraction.shape != head_fraction.shape:
            raise ReverseControllerCLIError(
                "certified reference returned the wrong transition shape",
                failure_domain="exact_reference",
                failure_code="controller_reference_shape_invalid",
            )
        certified = int(result.certified_mask.sum().detach().cpu().item())
        if certified != count:
            raise ReverseControllerCLIError(
                "an exact controller reference transition was uncertified",
                failure_domain="exact_reference",
                failure_code="controller_reference_uncertified",
            )
        self.transition_count += count
        self.certified_count += certified
        self.fallback_count += int(result.fallback_mask.sum().detach().cpu().item())
        diagnostics = result.diagnostics if isinstance(result.diagnostics, Mapping) else {}
        self.fallback_seconds += _tensor_scalar(
            diagnostics.get("arb_fallback_elapsed_seconds"), 0.0
        )
        for name in FORBIDDEN_DIAGNOSTICS:
            self.forbidden[name] += int(_tensor_scalar(diagnostics.get(name), 0.0))
        self.elapsed_seconds += time.perf_counter() - started
        return result

    def record(self) -> dict[str, Any]:
        elapsed = max(self.elapsed_seconds, np.finfo(float).tiny)
        return {
            "transition_count": self.transition_count,
            "certified_count": self.certified_count,
            "certificate_fraction": (
                self.certified_count / self.transition_count
                if self.transition_count
                else 1.0
            ),
            "fallback_count": self.fallback_count,
            "fallback_fraction": (
                self.fallback_count / self.transition_count
                if self.transition_count
                else 0.0
            ),
            "fallback_seconds": self.fallback_seconds,
            "fallback_time_fraction": self.fallback_seconds / elapsed,
            "elapsed_seconds": self.elapsed_seconds,
            "transitions_per_second": self.transition_count / elapsed,
            "maximum_transition_count_per_call": self.maximum_transition_count_per_call,
            "forbidden_counts": dict(self.forbidden),
        }


def _load_mixed_target(parent: Path) -> np.ndarray:
    metadata = _load_json(parent / "source_image.json")
    arrays = _load_npz(parent / "source_image.npz")
    if set(arrays) != {"image", "mixed_target"}:
        raise ParentBindingError("parent source-image schema changed")
    image = np.asarray(arrays["image"])
    target = np.asarray(arrays["mixed_target"])
    if (
        metadata.get("image_sha256") != EXPECTED_IMAGE_SHA256
        or metadata.get("mixed_target_sha256") != EXPECTED_MIXED_TARGET_SHA256
        or metadata.get("npz_sha256") != file_fingerprint(parent / "source_image.npz")
        or image.dtype != np.dtype(np.float64)
        or target.dtype != np.dtype(np.float64)
        or image.shape != (PATH_STATE_SIZE,)
        or target.shape != (PATH_STATE_SIZE,)
        or not np.isfinite(image).all()
        or not np.isfinite(target).all()
        or np.any(target < 0.0)
        or not math.isclose(float(target.sum()), 1.0, rel_tol=0.0, abs_tol=2.0e-12)
    ):
        raise ParentBindingError("parent mixed target identity changed")
    return np.ascontiguousarray(target)


def _initial_states(parent: Path, path_count: int, device: torch.device) -> Tensor:
    target = _load_mixed_target(parent)
    values = np.repeat(target[None, :], int(path_count), axis=0).copy(order="C")
    return torch.as_tensor(values, dtype=torch.float64, device=device).contiguous()


def _endpoint_equivalence(
    controller: Any,
    *,
    device: torch.device,
) -> dict[str, Any]:
    steps = list(SELECTED_OUTER_STEPS)
    step_rows = [step for step in steps for _ in range(7)]
    phases = [phase for _ in steps for phase in range(7)]
    states = torch.full((len(step_rows), PATH_STATE_SIZE), 1.0 / PATH_STATE_SIZE, dtype=torch.float32, device=device)
    inputs = ModelInputs(
        later_full_state=states,
        reverse_time=torch.tensor(
            [selected_reverse_time(step, phase) for step, phase in zip(step_rows, phases, strict=True)],
            dtype=torch.float64,
            device=device,
        ),
        phase=torch.tensor(phases, dtype=torch.long, device=device),
        color=torch.tensor([PHASE_MATCHINGS[p] for p in phases], dtype=torch.long, device=device),
        duration=torch.tensor([PHASE_DURATIONS[p] for p in phases], dtype=torch.float32, device=device),
        label=torch.full((len(step_rows),), 3, dtype=torch.long, device=device),
    )
    with torch.no_grad():
        old_residual = controller.predictor.residual(inputs)
        new_residual = controller.residual_prediction(inputs)
        old_combined = controller.predictor(inputs)
        new_combined = controller(inputs)
    return {
        "schema": RUN_SCHEMA + "-adapter-endpoint-equivalence",
        "probe_count": len(step_rows),
        "residual_bitwise_equal": int(torch.equal(old_residual, new_residual)),
        "combined_bitwise_equal": int(torch.equal(old_combined, new_combined)),
        "passed": int(torch.equal(old_residual, new_residual) and torch.equal(old_combined, new_combined)),
    }


def _formula_controls() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for pair_mass in (0.125, 0.75):
        for phase, duration in enumerate(PHASE_DURATIONS):
            for spacing in (1.0 / 28.0, 1.0 / 14.0):
                exposure = float(
                    _controller.phase_exposure(pair_mass, duration, h=spacing)
                )
                expected = 3.0 * (5.0e-5 / 512.0) * duration / (spacing**2 * pair_mass)
                m = (-1.0 if phase % 2 else 1.0) * 1.0e-3
                fraction_mass_transfer = pair_mass * 2.0 * m * exposure
                physical_flux = float(
                    _controller.learned_mass_flux(m, duration=duration, h=spacing)
                )
                rows.append({
                    "pair_mass": pair_mass,
                    "phase": phase,
                    "duration": duration,
                    "grid_spacing": spacing,
                    "exposure": exposure,
                    "expected_exposure": expected,
                    "exposure_error": abs(exposure - expected),
                    "fraction_mass_transfer": fraction_mass_transfer,
                    "physical_flux": physical_flux,
                    "flux_error": abs(fraction_mass_transfer - physical_flux),
                    "passed": int(exposure == expected and fraction_mass_transfer == physical_flux),
                })
    r, duration, spacing, m = 0.125, 0.5, 1.0 / 28.0, 0.1
    exposure = float(_controller.phase_exposure(r, duration, h=spacing))
    expected_flux = float(_controller.learned_mass_flux(m, duration=duration, h=spacing))
    expected_time = _controller.internal_reverse_time(11, 5, 0.5)
    expected_order = _controller.reverse_execution_order()
    mutants = {
        "reversed_orientation": -r * 2.0 * m * exposure,
        "wrong_sign": r * 2.0 * (-m) * exposure,
        "erroneous_pair_mass_factor": expected_flux * r,
        "omitted_duration": float(_controller.learned_mass_flux(m, duration=1.0, h=spacing)),
        "omitted_schedule": 6.0 * duration * m / spacing**2,
        "omitted_h_inverse_square": 6.0 * (5.0e-5 / 512.0) * duration * m,
    }
    negative = [
        {
            "fixture": name,
            "expected": expected_flux,
            "mutated": value,
            "rejected": int(not math.isclose(value, expected_flux, rel_tol=0.0, abs_tol=1.0e-18)),
        }
        for name, value in mutants.items()
    ]
    collapsed_time = _controller.internal_reverse_time(11, PHASE_OCCURRENCES.index(PHASE_OCCURRENCES[5]), 0.5)
    negative.extend(
        (
            {
                "fixture": "collapsed_repeated_occurrence",
                "expected": expected_time,
                "mutated": collapsed_time,
                "rejected": int(collapsed_time != expected_time),
            },
            {
                "fixture": "forward_phase_order",
                "expected": str(expected_order[:2]),
                "mutated": str(tuple(reversed(expected_order))[:2]),
                "rejected": int(tuple(reversed(expected_order)) != expected_order),
            },
        )
    )
    return rows, negative


def _boundary_controls(device: torch.device) -> dict[str, Any]:
    state = torch.zeros((1, PATH_STATE_SIZE), dtype=torch.float64, device=device)
    state[:, 0] = 0.75
    state[:, 1] = 0.25
    prediction = torch.zeros((1, EDGES_PER_PHASE), dtype=torch.float64, device=device)
    no_op = _controller.frozen_control_half_flow(state, 0, prediction, 0.1)
    outward_rejected = 0
    try:
        huge = torch.full_like(prediction, 1.0e6)
        _controller.frozen_control_half_flow(state, 0, huge, 1.0)
    except _controller.ControllerBoundaryStepRejected:
        outward_rejected = 1
    zero_mass = torch.zeros_like(state)
    zero_mass_noop = _controller.frozen_control_half_flow(zero_mass, 0, prediction, 1.0)
    return {
        "zero_prediction_bitwise_noop": int(torch.equal(no_op, state)),
        "zero_mass_bitwise_noop": int(torch.equal(zero_mass_noop, zero_mass)),
        "outward_boundary_rejected": outward_rejected,
        "clip_count": 0,
        "floor_count": 0,
        "limiter_count": 0,
        "projection_count": 0,
        "renormalization_count": 0,
        "passed": int(torch.equal(no_op, state) and torch.equal(zero_mass_noop, zero_mass) and outward_rejected),
    }


def _model_input_firewall_controls(
    controller: Any,
    *,
    device: torch.device,
) -> dict[str, Any]:
    batch = 1
    phase = 3
    valid = {
        "later_full_state": torch.full(
            (batch, PATH_STATE_SIZE),
            1.0 / PATH_STATE_SIZE,
            dtype=torch.float64,
            device=device,
        ),
        "reverse_time": torch.tensor(
            [selected_reverse_time(127, phase)], dtype=torch.float64, device=device
        ),
        "phase": torch.tensor([phase], dtype=torch.long, device=device),
        "color": torch.tensor([PHASE_MATCHINGS[phase]], dtype=torch.long, device=device),
        "duration": torch.tensor([PHASE_DURATIONS[phase]], dtype=torch.float64, device=device),
        "label": torch.tensor([3], dtype=torch.long, device=device),
    }
    exact_output = _controller.frozen_fractional_prediction(controller, valid)
    exact_six_pass = bool(
        exact_output.shape == (batch, EDGES_PER_PHASE)
        and torch.isfinite(exact_output).all()
        and set(valid) == set(MODEL_INPUT_FIELDS)
    )
    declared_forbidden = set(FORBIDDEN_MODEL_INPUT_FIELDS).union(
        {
            "transition_randomness",
            "certificate_data",
            "witness_identity",
            "branch_identity",
        }
    )
    rejected: dict[str, int] = {}
    for field in sorted(declared_forbidden):
        fixture = dict(valid)
        fixture[field] = torch.zeros((batch,), dtype=torch.float64, device=device)
        try:
            _controller.frozen_fractional_prediction(controller, fixture)
        except (TypeError, ValueError, RuntimeError):
            rejected[field] = 1
        else:
            rejected[field] = 0

    inputs_a = ModelInputs(**valid)
    alternate = dict(valid)
    alternate["later_full_state"] = torch.zeros_like(valid["later_full_state"])
    alternate["later_full_state"][:, 0] = 1.0
    alternate["color"] = torch.tensor([0], dtype=torch.long, device=device)
    alternate["duration"] = torch.tensor([0.5], dtype=torch.float64, device=device)
    alternate["label"] = torch.tensor([9], dtype=torch.long, device=device)
    inputs_b = ModelInputs(**alternate)
    baseline_only_time_phase = bool(
        torch.equal(
            controller.baseline_prediction(inputs_a),
            controller.baseline_prediction(inputs_b),
        )
    )
    return {
        "schema": RUN_SCHEMA + "-model-input-firewall-controls",
        "schema_version": 1,
        "allowed_fields": list(MODEL_INPUT_FIELDS),
        "exact_six_fields_pass": int(exact_six_pass),
        "forbidden_field_rejections": rejected,
        "all_forbidden_fields_rejected": int(bool(rejected) and all(rejected.values())),
        "coarse_lookup_uses_only_reverse_time_and_phase": int(baseline_only_time_phase),
        "passed": int(
            exact_six_pass
            and bool(rejected)
            and all(rejected.values())
            and baseline_only_time_phase
        ),
    }


def _preflight_controller_partition(
    initial: Tensor,
    *,
    path_ids: Sequence[int],
    group_sizes: Sequence[int],
    sequence: Sequence[tuple[int, int]],
    controller: Any,
    profile: JacobiRBCudaProfile,
    stream_role: str,
) -> tuple[Tensor, dict[str, Any]]:
    if sum(group_sizes) != len(path_ids):
        raise ValueError("controller preflight partition does not cover its paths")
    state = initial.clone().contiguous()
    reference = _CertifiedReference(
        root_seed=CONTROLLER_ROOT_SEED,
        profile=profile,
        stream_role=stream_role,
    )
    maximum_pair_error = 0.0
    maximum_simplex_error = 0.0
    for outer_step, phase in sequence:
        offset = 0
        parts: list[Tensor] = []
        for size in group_sizes:
            result = _controller.controlled_reverse_phase(
                state[offset : offset + size],
                outer_step,
                phase,
                2,
                NAMESPACE_VERSION,
                controller=controller,
                reference_transition=reference,
                path_ids=path_ids[offset : offset + size],
                label=3,
            )
            parts.append(result.state)
            maximum_pair_error = max(
                maximum_pair_error, result.maximum_pair_mass_error
            )
            maximum_simplex_error = max(
                maximum_simplex_error, result.maximum_simplex_mass_error
            )
            offset += size
        state = torch.cat(parts, dim=0).contiguous()
    record = reference.record()
    record.update(
        maximum_pair_mass_error=maximum_pair_error,
        maximum_simplex_mass_error=maximum_simplex_error,
    )
    return state, record


def _preflight_benchmark(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    controller: Any,
    device: torch.device,
) -> dict[str, Any]:
    path = run_dir / "resource_projection.json"
    if path.is_file():
        if _artifact_is_registered(run_dir, path):
            return _load_json(path)
        path.unlink()
    profile = JacobiRBCudaProfile()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial = _initial_states(args.parent_coarse_residual_run_dir, 8, device)
    started = time.perf_counter()
    first = run_exact_multipath_shard(
        initial.clone(),
        path_ids=PREFLIGHT_PATH_IDS,
        start_step=0,
        root_seed=CONTROLLER_ROOT_SEED,
        profile=profile,
        group_sizes=(8,),
        capture_phase_state_trace=True,
    )
    second = run_exact_multipath_shard(
        initial.clone(),
        path_ids=PREFLIGHT_PATH_IDS,
        start_step=0,
        root_seed=CONTROLLER_ROOT_SEED,
        profile=profile,
        group_sizes=(4, 4),
        capture_phase_state_trace=True,
    )
    grouping_invariant = bool(
        first.batch_output_sha256 == second.batch_output_sha256
        and first.batch_final_state_sha256 == second.batch_final_state_sha256
        and np.array_equal(first.committed_final_states, second.committed_final_states)
    )
    reversed_paths = tuple(reversed(PREFLIGHT_PATH_IDS))
    reversed_result = run_exact_multipath_shard(
        initial.flip(0).contiguous(),
        path_ids=reversed_paths,
        start_step=0,
        root_seed=CONTROLLER_ROOT_SEED,
        profile=profile,
        group_sizes=(8,),
        capture_phase_state_trace=True,
    )
    path_order_invariant = bool(
        first.batch_output_sha256 == reversed_result.batch_output_sha256
        and first.batch_final_state_sha256 == reversed_result.batch_final_state_sha256
    )
    restart_artifact = _atomic_npz(
        run_dir / "preflight_restart_probe.npz",
        {"states": first.committed_final_states},
    )
    reloaded = _load_npz(run_dir / "preflight_restart_probe.npz")["states"]
    direct_continuation = run_exact_multipath_shard(
        first.final_states.clone(),
        path_ids=PREFLIGHT_PATH_IDS,
        start_step=8,
        root_seed=CONTROLLER_ROOT_SEED,
        profile=profile,
        group_sizes=(8,),
    )
    resumed_continuation = run_exact_multipath_shard(
        torch.as_tensor(reloaded.copy(order="C"), dtype=torch.float64, device=device).contiguous(),
        path_ids=PREFLIGHT_PATH_IDS,
        start_step=8,
        root_seed=CONTROLLER_ROOT_SEED,
        profile=profile,
        group_sizes=(8,),
    )
    restart_invariant = bool(
        direct_continuation.batch_output_sha256 == resumed_continuation.batch_output_sha256
        and direct_continuation.batch_final_state_sha256 == resumed_continuation.batch_final_state_sha256
    )
    reference = _CertifiedReference(root_seed=CONTROLLER_ROOT_SEED, profile=profile)
    # Controller timing must use law-matched states at the actual internal
    # reverse time.  Reusing the eight-step scheduler state at late anchors
    # would benchmark an out-of-distribution controller input.  The sealed
    # parent validation input cache contains only model inputs (no labels), is
    # hash-bound by parent provenance, and provides 32 paths at every selected
    # (outer step, phase) coordinate.
    representative_cache = _parent_cli._load_input_cache_for_role(  # noqa: SLF001
        args.parent_coarse_residual_run_dir, "validation"
    )
    representative_state_hashes: dict[str, str] = {}
    control_started = time.perf_counter()
    maximum_pair_mass_error = 0.0
    maximum_simplex_mass_error = 0.0
    controlled_states_finite = True
    controlled_states_nonnegative = True
    for outer_step in CONTROL_ANCHORS:
        for phase in range(7):
            target_time = float(selected_reverse_time(outer_step, phase))
            indices = np.flatnonzero(
                (representative_cache.phase == phase)
                & (representative_cache.reverse_time == target_time)
            )
            if indices.size < len(PREFLIGHT_PATH_IDS):
                raise ArtifactCompatibilityError(
                    "sealed parent validation inputs lack a representative "
                    f"state panel for outer_step={outer_step}, phase={phase}"
                )
            representative_np = np.ascontiguousarray(
                representative_cache.later_full_state[
                    indices[: len(PREFLIGHT_PATH_IDS)]
                ],
                dtype=np.float64,
            )
            representative_state_hashes[f"{outer_step}:{phase}"] = hashlib.sha256(
                representative_np.tobytes(order="C")
            ).hexdigest()
            representative_state = torch.as_tensor(
                representative_np.copy(), dtype=torch.float64, device=device
            ).contiguous()
            for microsteps in (2, 4):
                controlled = _controller.controlled_reverse_phase(
                    representative_state,
                    outer_step,
                    phase,
                    microsteps,
                    NAMESPACE_VERSION,
                    controller=controller,
                    reference_transition=reference,
                    path_ids=PREFLIGHT_PATH_IDS,
                    label=3,
                )
                maximum_pair_mass_error = max(maximum_pair_mass_error, controlled.maximum_pair_mass_error)
                maximum_simplex_mass_error = max(maximum_simplex_mass_error, controlled.maximum_simplex_mass_error)
                controlled_states_finite &= bool(torch.isfinite(controlled.state).all())
                controlled_states_nonnegative &= not bool(torch.any(controlled.state < 0.0))
    invariant_sequence = ((511, 6), (511, 5))
    invariant_role = "preflight-controller-partition-invariance"
    controller_group8, controller_group8_record = _preflight_controller_partition(
        representative_state,
        path_ids=PREFLIGHT_PATH_IDS,
        group_sizes=(8,),
        sequence=invariant_sequence,
        controller=controller,
        profile=profile,
        stream_role=invariant_role,
    )
    controller_group44, controller_group44_record = _preflight_controller_partition(
        representative_state,
        path_ids=PREFLIGHT_PATH_IDS,
        group_sizes=(4, 4),
        sequence=invariant_sequence,
        controller=controller,
        profile=profile,
        stream_role=invariant_role,
    )
    controller_reversed, controller_reversed_record = _preflight_controller_partition(
        representative_state.flip(0).contiguous(),
        path_ids=tuple(reversed(PREFLIGHT_PATH_IDS)),
        group_sizes=(8,),
        sequence=invariant_sequence,
        controller=controller,
        profile=profile,
        stream_role=invariant_role,
    )
    first_occurrence, controller_restart_first_record = _preflight_controller_partition(
        representative_state,
        path_ids=PREFLIGHT_PATH_IDS,
        group_sizes=(8,),
        sequence=invariant_sequence[:1],
        controller=controller,
        profile=profile,
        stream_role=invariant_role,
    )
    controller_restart_artifact = _atomic_npz(
        run_dir / "preflight_controller_restart_probe.npz",
        {"states": first_occurrence.detach().cpu().numpy()},
    )
    controller_restart_np = _load_npz(
        run_dir / "preflight_controller_restart_probe.npz"
    )["states"]
    controller_resumed, controller_restart_second_record = _preflight_controller_partition(
        torch.as_tensor(
            controller_restart_np.copy(), dtype=torch.float64, device=device
        ).contiguous(),
        path_ids=PREFLIGHT_PATH_IDS,
        group_sizes=(8,),
        sequence=invariant_sequence[1:],
        controller=controller,
        profile=profile,
        stream_role=invariant_role,
    )
    def invariant_signature(item: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            int(item["transition_count"]),
            int(item["certified_count"]),
            int(item["fallback_count"]),
            tuple(int(item["forbidden_counts"][name]) for name in FORBIDDEN_DIAGNOSTICS),
            float(item["maximum_pair_mass_error"]),
            float(item["maximum_simplex_mass_error"]),
        )

    restart_signature = (
        int(controller_restart_first_record["transition_count"])
        + int(controller_restart_second_record["transition_count"]),
        int(controller_restart_first_record["certified_count"])
        + int(controller_restart_second_record["certified_count"]),
        int(controller_restart_first_record["fallback_count"])
        + int(controller_restart_second_record["fallback_count"]),
        tuple(
            int(controller_restart_first_record["forbidden_counts"][name])
            + int(controller_restart_second_record["forbidden_counts"][name])
            for name in FORBIDDEN_DIAGNOSTICS
        ),
        max(
            float(controller_restart_first_record["maximum_pair_mass_error"]),
            float(controller_restart_second_record["maximum_pair_mass_error"]),
        ),
        max(
            float(controller_restart_first_record["maximum_simplex_mass_error"]),
            float(controller_restart_second_record["maximum_simplex_mass_error"]),
        ),
    )
    base_signature = invariant_signature(controller_group8_record)
    controller_grouping_invariant = bool(
        torch.equal(controller_group8, controller_group44)
        and base_signature == invariant_signature(controller_group44_record)
    )
    controller_path_order_invariant = bool(
        torch.equal(controller_group8, controller_reversed.flip(0))
        and base_signature == invariant_signature(controller_reversed_record)
    )
    controller_restart_invariant = bool(
        torch.equal(controller_group8, controller_resumed)
        and base_signature == restart_signature
    )
    invariance_reference_records = (
        controller_group8_record,
        controller_group44_record,
        controller_reversed_record,
        controller_restart_first_record,
        controller_restart_second_record,
    )
    maximum_pair_mass_error = max(
        maximum_pair_mass_error,
        *(float(item["maximum_pair_mass_error"]) for item in invariance_reference_records),
    )
    maximum_simplex_mass_error = max(
        maximum_simplex_mass_error,
        *(float(item["maximum_simplex_mass_error"]) for item in invariance_reference_records),
    )
    control_elapsed = time.perf_counter() - control_started
    elapsed = time.perf_counter() - started
    diag = first.diagnostics
    scheduler_results = (first, second, reversed_result, direct_continuation, resumed_continuation)
    full_count = sum(int(item.diagnostics.get("transition_count", 0)) for item in scheduler_results)
    invariance_transition_count = sum(
        int(item["transition_count"]) for item in invariance_reference_records
    )
    total_count = full_count + reference.transition_count + invariance_transition_count
    effective_rate = total_count / max(elapsed, np.finfo(float).tiny)
    parent_rate = float(
        _load_json(args.parent_coarse_residual_run_dir / "confirmation_cache_gate.json")
        ["subchecks"]["transitions_per_second"]["value"]
    )
    conservative_rate = min(effective_rate, parent_rate)
    projected_hours = TOTAL_TRANSITIONS / conservative_rate / 3600.0
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    total_memory = (
        int(torch.cuda.get_device_properties(device).total_memory)
        if device.type == "cuda"
        else 1
    )
    reference_record = reference.record()
    scheduler_fallback_count = sum(int(item.diagnostics.get("fallback_count", 0)) for item in scheduler_results)
    scheduler_fallback_seconds = sum(float(item.diagnostics.get("fallback_elapsed_seconds", 0.0)) for item in scheduler_results)
    scheduler_backend_seconds = sum(
        float(item.diagnostics.get("elapsed_seconds", 0.0))
        for item in scheduler_results
    )
    aggregate_fallback_count = (
        scheduler_fallback_count
        + int(reference_record["fallback_count"])
        + sum(int(item["fallback_count"]) for item in invariance_reference_records)
    )
    aggregate_fallback_seconds = (
        scheduler_fallback_seconds
        + float(reference_record["fallback_seconds"])
        + sum(float(item["fallback_seconds"]) for item in invariance_reference_records)
    )
    aggregate_backend_seconds = (
        scheduler_backend_seconds
        + float(reference_record["elapsed_seconds"])
        + sum(float(item["elapsed_seconds"]) for item in invariance_reference_records)
    )
    persistence_components = {
        "forward_state_arrays": 64 * 64 * PATH_STATE_SIZE * 8,
        "local_contrast_values": 32 * 64 * 56 * 2 * 8,
        "control_audit_endpoint_arrays": 4 * 8 * 2 * 64 * PATH_STATE_SIZE * 8,
        "control_final_state_arrays": (4 * 7 * 3 + 4 * 3) * 64 * PATH_STATE_SIZE * 8,
        "control_phase_checkpoint_arrays": (4 * 7 * 3 + 4 * 3 * 8) * 64 * PATH_STATE_SIZE * 8,
        "bootstrap_and_tabular_allowance": 16 * 1024**2,
    }
    raw_persisted_bytes = sum(persistence_components.values())
    persistence_safety_factor = 1.75
    projected_persisted_bytes = math.ceil(
        raw_persisted_bytes * persistence_safety_factor
    )
    record = {
        "schema": RUN_SCHEMA + "-resource-projection",
        "schema_version": 1,
        "benchmark_paths": list(PREFLIGHT_PATH_IDS),
        "benchmark_outer_step_anchors": list(CONTROL_ANCHORS),
        "benchmark_microsteps": [2, 4],
        "controller_benchmark_state_source": "sealed_parent_validation_model_inputs",
        "controller_benchmark_state_hashes": representative_state_hashes,
        "grouping_8_vs_4_plus_4_hash_equal": int(grouping_invariant),
        "path_order_reversal_hash_equal": int(path_order_invariant),
        "restart_state_artifact": restart_artifact,
        "interrupted_resume_hash_equal": int(restart_invariant),
        "controller_grouping_8_vs_4_plus_4_equal": int(controller_grouping_invariant),
        "controller_path_order_reversal_equal": int(controller_path_order_invariant),
        "controller_phase_restart_equal": int(controller_restart_invariant),
        "controller_restart_state_artifact": controller_restart_artifact,
        "exact_reference": reference_record,
        "controller_invariance_exact_references": list(invariance_reference_records),
        "full_phase_transition_count": full_count,
        "controller_transition_count": reference.transition_count,
        "benchmark_transition_count": total_count,
        "benchmark_elapsed_seconds": elapsed,
        "controller_elapsed_seconds": control_elapsed,
        "effective_complete_pipeline_rate": effective_rate,
        "sealed_parent_rate": parent_rate,
        "conservative_transition_rate": conservative_rate,
        "projected_transition_count": TOTAL_TRANSITIONS,
        "projected_total_hours": projected_hours,
        "peak_device_memory_bytes": peak_memory,
        "total_device_memory_bytes": total_memory,
        "peak_device_memory_fraction": peak_memory / total_memory,
        "maximum_pair_mass_error": maximum_pair_mass_error,
        "maximum_simplex_mass_error": maximum_simplex_mass_error,
        "controlled_states_finite": int(controlled_states_finite),
        "controlled_states_nonnegative": int(controlled_states_nonnegative),
        "persistence_projection": {
            "components": persistence_components,
            "raw_bytes": raw_persisted_bytes,
            "serialization_and_metadata_safety_factor": persistence_safety_factor,
            "projected_bytes": projected_persisted_bytes,
        },
        "projected_persisted_bytes": projected_persisted_bytes,
        "certificate_fraction": min(
            min(float(item.diagnostics.get("certified_count", 0)) / max(int(item.diagnostics.get("transition_count", 1)), 1) for item in scheduler_results),
            float(reference_record["certificate_fraction"]),
            min(float(item["certificate_fraction"]) for item in invariance_reference_records),
        ),
        "scheduler_fallback_count": scheduler_fallback_count,
        "scheduler_fallback_seconds": scheduler_fallback_seconds,
        "scheduler_backend_seconds": scheduler_backend_seconds,
        "exact_backend_seconds": aggregate_backend_seconds,
        "fallback_count": aggregate_fallback_count,
        "fallback_fraction": aggregate_fallback_count / max(total_count, 1),
        "fallback_time_fraction": aggregate_fallback_seconds / max(aggregate_backend_seconds, np.finfo(float).tiny),
        "forbidden_counts": {
            name: (
                sum(int(item.diagnostics.get(name, 0)) for item in scheduler_results)
                + int(reference_record["forbidden_counts"][name])
                + sum(int(item["forbidden_counts"][name]) for item in invariance_reference_records)
            )
            for name in FORBIDDEN_DIAGNOSTICS
        },
        "maximum_transition_count_per_call": max(
            reference.maximum_transition_count_per_call,
            *(int(item["maximum_transition_count_per_call"]) for item in invariance_reference_records),
        ),
    }
    _freeze_json(path, record)
    return record


def _preflight_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    gate_path = run_dir / "preflight_gate.json"
    committed_gate = _registered_stage_gate(run_dir, gate_path)
    if committed_gate is not None:
        return committed_gate
    configure_exact_torch_backend()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ReverseControllerCLIError(
            "authorizing preflight requires CUDA",
            failure_domain="resource",
            failure_code="controller_cuda_unavailable",
        )
    controller = _controller.load_frozen_controller(
        args.parent_coarse_residual_run_dir, device=device
    )
    endpoint = _endpoint_equivalence(controller, device=device)
    formula_rows, negative_rows = _formula_controls()
    boundary = _boundary_controls(device)
    firewall = _model_input_firewall_controls(controller, device=device)
    _write_csv(run_dir / "preflight_formula_controls.csv", formula_rows)
    _write_csv(run_dir / "preflight_negative_controls.csv", negative_rows)
    _freeze_json(run_dir / "adapter_endpoint_equivalence.json", endpoint)
    _freeze_json(run_dir / "model_input_firewall_controls.json", firewall)
    # The two files are physically separated and contain no joined row.
    probe_state = np.full((1, PATH_STATE_SIZE), 1.0 / PATH_STATE_SIZE, dtype=np.float64)
    _atomic_npz(run_dir / "input_probe.npz", {"later_full_state": probe_state})
    _atomic_npz(run_dir / "label_audit_probe.npz", {"denoising_target": np.zeros((1, EDGES_PER_PHASE), dtype=np.float64)})
    resources = _preflight_benchmark(
        run_dir, args, controller=controller, device=device
    )
    checks = {
        "parent_provenance": _load_json(run_dir / "parent_provenance.json").get("passed") == 1,
        "package_manifest": _load_json(run_dir / "package_manifest_verification.json").get("all_hashes_verified") == 1,
        "path_namespace": _load_json(run_dir / "path_id_plan.json").get("collision_free") == 1,
        "endpoint_equivalence": endpoint.get("passed") == 1,
        "formula_controls": bool(formula_rows) and all(int(row["passed"]) == 1 for row in formula_rows),
        "negative_controls": all(int(row["rejected"]) == 1 for row in negative_rows),
        "boundary_controls": boundary.get("passed") == 1,
        "model_input_firewall": firewall.get("passed") == 1,
        "restart_grouping_invariance": resources.get("grouping_8_vs_4_plus_4_hash_equal") == 1,
        "path_order_invariance": resources.get("path_order_reversal_hash_equal") == 1,
        "interrupted_resume_invariance": resources.get("interrupted_resume_hash_equal") == 1,
        "controller_grouping_invariance": resources.get("controller_grouping_8_vs_4_plus_4_equal") == 1,
        "controller_path_order_invariance": resources.get("controller_path_order_reversal_equal") == 1,
        "controller_phase_restart_invariance": resources.get("controller_phase_restart_equal") == 1,
        "pair_mass_preservation": float(resources.get("maximum_pair_mass_error", math.inf)) <= RESOURCE_THRESHOLDS["maximum_mass_error"],
        "simplex_mass_preservation": float(resources.get("maximum_simplex_mass_error", math.inf)) <= RESOURCE_THRESHOLDS["maximum_mass_error"],
        "controlled_states_finite": resources.get("controlled_states_finite") == 1,
        "controlled_states_nonnegative": resources.get("controlled_states_nonnegative") == 1,
        "reference_launch_cap": int(resources.get("maximum_transition_count_per_call", 1 << 30)) <= 4096,
        "certificate_fraction": float(resources.get("certificate_fraction", 0.0)) == 1.0,
        "fallback_fraction": float(resources.get("fallback_fraction", math.inf)) <= RESOURCE_THRESHOLDS["maximum_fallback_fraction"],
        "fallback_time": float(resources.get("fallback_time_fraction", math.inf)) <= RESOURCE_THRESHOLDS["maximum_fallback_time_fraction"],
        "throughput": float(resources.get("conservative_transition_rate", 0.0)) >= RESOURCE_THRESHOLDS["minimum_transitions_per_second"],
        "projected_time": float(resources.get("projected_total_hours", math.inf)) <= RESOURCE_THRESHOLDS["maximum_projected_hours"],
        "memory": float(resources.get("peak_device_memory_fraction", math.inf)) <= RESOURCE_THRESHOLDS["maximum_peak_memory_fraction"],
        "persisted_size": int(resources.get("projected_persisted_bytes", 1 << 62)) <= RESOURCE_THRESHOLDS["maximum_persisted_bytes"],
        "forbidden_counts": all(int(value) == 0 for value in resources.get("forbidden_counts", {}).values()),
        "no_physical_panel_opened": not (run_dir / "forward").exists() and not (run_dir / "local_risk_max_t.json").exists(),
    }
    gate = _gate_record(
        "preflight",
        checks,
        numerically_valid=all(
            checks[name]
            for name in (
                "parent_provenance", "package_manifest", "path_namespace",
                "endpoint_equivalence", "formula_controls", "negative_controls",
                "boundary_controls", "model_input_firewall",
                "restart_grouping_invariance", "path_order_invariance",
                "interrupted_resume_invariance", "pair_mass_preservation",
                "controller_grouping_invariance",
                "controller_path_order_invariance",
                "controller_phase_restart_invariance",
                "simplex_mass_preservation", "controlled_states_finite",
                "controlled_states_nonnegative", "reference_launch_cap",
                "certificate_fraction", "forbidden_counts",
            )
        ),
        resource_valid=all(
            checks[name]
            for name in (
                "fallback_fraction", "fallback_time", "throughput",
                "projected_time", "memory", "persisted_size",
            )
        ),
        adapter_endpoint_equivalence=endpoint,
        boundary_controls=boundary,
        model_input_firewall=firewall,
        resource_projection=resources,
        target_modified=0,
        physical_authorizing_statistics_opened=0,
    )
    _freeze_json(gate_path, gate)
    return gate


def _higher_quantile(values: np.ndarray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:  # pragma: no cover - NumPy <1.22 compatibility.
        return float(np.quantile(values, probability, interpolation="higher"))


def _two_sided_studentized_max_t(
    values: np.ndarray,
    *,
    confidence: float,
    replicates: int,
    seed: int,
    names: Sequence[str],
) -> dict[str, Any]:
    """Deterministic whole-unit two-sided studentized max-T intervals."""

    data = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] != len(names):
        raise ValueError("max-T input must be [units, named features]")
    if not np.isfinite(data).all() or not 0.0 < confidence < 1.0 or replicates <= 0:
        raise ValueError("invalid max-T input")
    n = data.shape[0]
    point = data.mean(axis=0)
    se = data.std(axis=0, ddof=1) / math.sqrt(n)
    if np.any(se == 0.0):
        raise ValueError("nonstructural max-T feature is degenerate")
    rng = np.random.default_rng(int(seed))
    maxima = np.empty(int(replicates), dtype=np.float64)
    block = 512
    for start in range(0, int(replicates), block):
        size = min(block, int(replicates) - start)
        indices = rng.integers(0, n, size=(size, n), endpoint=False)
        draws = data[indices]
        means = draws.mean(axis=1)
        errors = draws.std(axis=1, ddof=1) / math.sqrt(n)
        statistic = np.divide(
            np.abs(means - point[None, :]),
            errors,
            out=np.full_like(means, math.inf),
            where=errors > 0.0,
        )
        maxima[start : start + size] = statistic.max(axis=1)
    critical = _higher_quantile(maxima, confidence)
    upper_absolute = np.abs(point) + critical * se
    lower = point - critical * se
    upper = point + critical * se
    return {
        "schema": RUN_SCHEMA + "-two-sided-studentized-max-t",
        "method": "centered_whole_path_two_sided_studentized_max_t",
        "confidence": float(confidence),
        "replicates": int(replicates),
        "seed": int(seed),
        "bootstrap_unit": "whole_path",
        "quantile_method": "higher",
        "family_size": len(names),
        "family_names": list(names),
        "critical_value": critical,
        "point_estimates": {name: float(value) for name, value in zip(names, point, strict=True)},
        "standard_errors": {name: float(value) for name, value in zip(names, se, strict=True)},
        "simultaneous_upper_absolute": {name: float(value) for name, value in zip(names, upper_absolute, strict=True)},
        "simultaneous_lower": {name: float(value) for name, value in zip(names, lower, strict=True)},
        "simultaneous_upper": {name: float(value) for name, value in zip(names, upper, strict=True)},
    }


def _one_sided_matrix_max_t(
    values: np.ndarray,
    *,
    confidence: float,
    replicates: int,
    seed: int,
    names: Sequence[str],
) -> dict[str, Any]:
    data = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if data.ndim != 2 or data.shape[0] < 8 or data.shape[1] != len(names):
        raise ValueError("one-sided max-T input must be [at-least-8 paths,features]")
    if not np.isfinite(data).all() or len(set(names)) != len(names):
        raise ValueError("one-sided max-T family is invalid")
    n = data.shape[0]
    point = data.mean(axis=0)
    se = data.std(axis=0, ddof=1) / math.sqrt(n)
    if np.any(se <= 0.0):
        raise ValueError("one-sided max-T family is degenerate")
    rng = np.random.Generator(np.random.Philox([int(seed), 0, 0x52424354]))
    maxima = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 256):
        size = min(256, replicates - start)
        indices = rng.integers(0, n, size=(size, n), dtype=np.int64)
        draw = data[indices]
        mean = draw.mean(axis=1)
        error = draw.std(axis=1, ddof=1) / math.sqrt(n)
        if np.any(error <= 0.0):
            raise ValueError("one-sided bootstrap studentization is degenerate")
        maxima[start : start + size] = ((mean - point[None, :]) / error).max(axis=1)
    critical = _higher_quantile(maxima, confidence)
    lower = point - critical * se
    return {
        "schema": RUN_SCHEMA + "-one-sided-studentized-max-t",
        "method": "centered_whole_path_studentized_max_t",
        "bootstrap_unit": "whole_path_jointly_across_family",
        "quantile_method": "higher",
        "family_names": list(names),
        "family_size": len(names),
        "point_estimates": {name: float(value) for name, value in zip(names, point, strict=True)},
        "standard_errors": {name: float(value) for name, value in zip(names, se, strict=True)},
        "lower_bounds": {name: float(value) for name, value in zip(names, lower, strict=True)},
        "critical_value": critical,
        "path_count": n,
        "confidence": confidence,
        "replicates": replicates,
        "seed": seed,
        "negative_values_truncated": 0,
        "passed": int(np.all(lower > 0.0)),
    }


def _normalized_trajectory_max_t(
    numerators: np.ndarray,
    forward_changes: np.ndarray,
    *,
    confidence: float,
    replicates: int,
    seed: int,
    names: Sequence[str],
) -> dict[str, Any]:
    """Studentized max-T with RMS normalization recomputed per path draw."""

    num = np.ascontiguousarray(np.asarray(numerators, dtype=np.float64))
    den = np.ascontiguousarray(np.asarray(forward_changes, dtype=np.float64))
    if num.shape != den.shape or num.ndim != 2 or num.shape[1] != len(names) or num.shape[0] < 2:
        raise ValueError("trajectory arrays must be matching [paths,features]")
    if not np.isfinite(num).all() or not np.isfinite(den).all():
        raise ValueError("trajectory arrays are nonfinite")
    n = num.shape[0]

    def estimate(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale = np.sqrt(np.mean(b * b, axis=-2))
        if np.any(scale <= 0.0):
            raise ValueError("nonstructural forward-change denominator is degenerate")
        point = np.mean(a, axis=-2) / scale
        # Delta-method influence for mean(a)/sqrt(mean(b^2)).
        mean_a = np.mean(a, axis=-2)
        centered = (
            a / scale[..., None, :] - point[..., None, :]
            if a.ndim == 3
            else a / scale - point
        )
        correction = (
            mean_a[..., None, :] * (b * b - scale[..., None, :] ** 2) / (2.0 * scale[..., None, :] ** 3)
            if a.ndim == 3
            else mean_a * (b * b - scale**2) / (2.0 * scale**3)
        )
        influence = centered - correction
        error = np.std(influence, axis=-2, ddof=1) / math.sqrt(n)
        return point, error

    point, se = estimate(num, den)
    if np.any(se <= 0.0):
        raise ValueError("trajectory studentization is degenerate")
    rng = np.random.default_rng(int(seed))
    maxima = np.empty(int(replicates), dtype=np.float64)
    block = 128
    for start in range(0, int(replicates), block):
        size = min(block, int(replicates) - start)
        indices = rng.integers(0, n, size=(size, n), endpoint=False)
        draw_point, draw_se = estimate(num[indices], den[indices])
        statistic = np.divide(
            np.abs(draw_point - point[None, :]),
            draw_se,
            out=np.full_like(draw_point, math.inf),
            where=draw_se > 0.0,
        )
        maxima[start : start + size] = statistic.max(axis=1)
    critical = _higher_quantile(maxima, confidence)
    upper = np.abs(point) + critical * se
    return {
        "schema": RUN_SCHEMA + "-normalized-trajectory-max-t",
        "method": "whole_path_rms_normalized_two_sided_studentized_max_t",
        "confidence": confidence,
        "replicates": replicates,
        "seed": seed,
        "bootstrap_unit": "whole_path",
        "denominator_recomputed_per_resample": 1,
        "quantile_method": "higher",
        "family_size": len(names),
        "family_names": list(names),
        "critical_value": critical,
        "point_estimates": {name: float(value) for name, value in zip(names, point, strict=True)},
        "standard_errors": {name: float(value) for name, value in zip(names, se, strict=True)},
        "simultaneous_upper_absolute": {name: float(value) for name, value in zip(names, upper, strict=True)},
    }


def _legendre_matrix(values: np.ndarray, maximum_degree: int = 8) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    z = 2.0 * x - 1.0
    output = np.empty((x.size, maximum_degree), dtype=np.float64)
    p0 = np.ones_like(z)
    p1 = z
    output[:, 0] = p1
    for degree in range(2, maximum_degree + 1):
        pn = ((2 * degree - 1) * z * p1 - (degree - 1) * p0) / degree
        output[:, degree - 1] = pn
        p0, p1 = p1, pn
    return output


def _oracle_ids(
    sample_indices: np.ndarray,
    *,
    operation: int,
    role: str,
    device: torch.device,
) -> Tensor:
    """Structured EE-path oracle IDs, including split coordinates and role."""

    indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    cluster_size = 262_144 // len(ORACLE_PATH_IDS)
    cluster = indices // cluster_size
    within = indices % cluster_size
    if np.any(cluster < 0) or np.any(cluster >= len(ORACLE_PATH_IDS)):
        raise ValueError("oracle sample index lies outside frozen EE path clusters")
    edge = within % EDGES_PER_PHASE
    phase = (within // EDGES_PER_PHASE) % 7
    outer = within // (7 * EDGES_PER_PHASE)
    role_map = {name: index for index, name in enumerate(_controller.TRANSITION_ROLES)}
    if role not in role_map or not 0 <= int(operation) < (1 << 23):
        raise ValueError("oracle transition role/operation is outside the frozen layout")
    paths = np.asarray(ORACLE_PATH_IDS, dtype=np.uint64)[cluster]
    packed = (
        (paths << np.uint64(44))
        | (outer.astype(np.uint64) << np.uint64(39))
        | (phase.astype(np.uint64) << np.uint64(36))
        | (edge.astype(np.uint64) << np.uint64(27))
        | (np.uint64(role_map[role]) << np.uint64(23))
        | np.uint64(operation)
    )
    if np.unique(packed).size != packed.size:
        raise ValueError("oracle transition IDs are not injective")
    return torch.as_tensor(packed.copy(), dtype=torch.uint64, device=device).contiguous()


def _sample_reference_vector(
    x: np.ndarray,
    exposure: float,
    *,
    device: torch.device,
    reference: _CertifiedReference,
    role: str,
    operation: int,
    sample_indices: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    output = np.empty_like(values)
    indices = (
        np.arange(values.size, dtype=np.int64)
        if sample_indices is None
        else np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    )
    if indices.shape != values.shape:
        raise ValueError("oracle sample indices do not match transition values")
    for start in range(0, values.size, 4096):
        stop = min(values.size, start + 4096)
        head = torch.as_tensor(values[start:stop].copy(), dtype=torch.float64, device=device).contiguous()
        duration = torch.full_like(head, float(exposure))
        result = reference(
            head_fraction=head,
            exposure=duration,
            transition_ids=_oracle_ids(
                indices[start:stop], operation=operation, role=role, device=device
            ),
            role=role,
        )
        output[start:stop] = result.later_head_fraction.detach().cpu().numpy()
    return output


def _sample_linear_density(uniforms: np.ndarray, coefficient: float) -> np.ndarray:
    u = np.asarray(uniforms, dtype=np.float64)
    c = float(coefficient)
    if abs(c) < 1.0e-15:
        return u.copy()
    discriminant = (1.0 - c) ** 2 + 4.0 * c * u
    return np.ascontiguousarray(2.0 * u / ((1.0 - c) + np.sqrt(discriminant)))


def _analytic_teacher_numerical_reverse(
    y: np.ndarray,
    *,
    s: float,
    delta: float,
    microsteps: int,
    device: torch.device,
    reference: _CertifiedReference,
    operation_base: int,
    score_function: Callable[[np.ndarray, float], np.ndarray] | None = None,
) -> np.ndarray:
    current = np.asarray(y, dtype=np.float64).copy()
    du = float(delta) / microsteps
    cursor = int(operation_base)
    sample_indices = np.arange(current.size, dtype=np.int64)
    for reverse_index in range(microsteps):
        current = _sample_reference_vector(
            current,
            du / 2.0,
            device=device,
            reference=reference,
            role=f"reverse_reference_pre_control_M{microsteps}",
            operation=cursor,
            sample_indices=sample_indices,
        )
        cursor += 1
        midpoint_s = s - (reverse_index + 0.5) * du
        evaluator = score_function or _controller.bounded_linear_teacher_score
        score = np.asarray(evaluator(current, midpoint_s), dtype=np.float64)
        next_values = current + 2.0 * score * du
        if not np.isfinite(next_values).all() or np.any((next_values < 0.0) | (next_values > 1.0)):
            raise ReverseControllerCLIError(
                "analytic teacher controller crossed a boundary",
                failure_domain="formula",
                failure_code="analytic_teacher_boundary_rejected",
            )
        current = next_values
        current = _sample_reference_vector(
            current,
            du / 2.0,
            device=device,
            reference=reference,
            role=f"reverse_reference_post_control_M{microsteps}",
            operation=cursor,
            sample_indices=sample_indices,
        )
        cursor += 1
    return current


def _analytic_teacher_exact_reverse(
    y: np.ndarray,
    *,
    s: float,
    delta: float,
    device: torch.device,
    reference: _CertifiedReference,
    rng: np.random.Generator,
    operation_base: int,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    output = np.empty_like(y)
    pending = np.arange(y.size)
    cursor = int(operation_base)
    coefficient = 0.5 * math.exp(-2.0 * (s - delta))
    maximum = 1.0 + abs(coefficient)
    round_index = 0
    while pending.size:
        proposal = _sample_reference_vector(
            y[pending],
            delta,
            device=device,
            reference=reference,
            role="analytic_teacher_exact_reverse",
            operation=cursor,
            sample_indices=pending,
        )
        cursor += 1
        acceptance = (1.0 + coefficient * (2.0 * proposal - 1.0)) / maximum
        uniform = rng.random(pending.size)
        accepted = uniform < acceptance
        output[pending[accepted]] = proposal[accepted]
        pending = pending[~accepted]
        round_index += 1
        if round_index > 256:
            raise ReverseControllerCLIError(
                "analytic reverse rejection sampler exceeded its frozen cap",
                failure_domain="oracle",
                failure_code="analytic_teacher_rejection_cap",
            )
    return output


def _pure_reference_composition(
    y: np.ndarray,
    *,
    delta: float,
    microsteps: int,
    device: torch.device,
    reference: _CertifiedReference,
    operation_base: int,
) -> np.ndarray:
    current = np.asarray(y, dtype=np.float64).copy()
    indices = np.arange(current.size, dtype=np.int64)
    du = float(delta) / microsteps
    operation = int(operation_base)
    for _ in range(microsteps):
        current = _sample_reference_vector(
            current,
            du / 2.0,
            device=device,
            reference=reference,
            role=f"reverse_reference_pre_control_M{microsteps}",
            operation=operation,
            sample_indices=indices,
        )
        operation += 1
        current = _sample_reference_vector(
            current,
            du / 2.0,
            device=device,
            reference=reference,
            role=f"reverse_reference_post_control_M{microsteps}",
            operation=operation,
            sample_indices=indices,
        )
        operation += 1
    return current


def _oracle_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    gate_path = run_dir / "oracle_gate.json"
    committed_gate = _registered_stage_gate(run_dir, gate_path)
    if committed_gate is not None:
        return committed_gate
    configure_exact_torch_backend()
    if not _passed(_load_json(run_dir / "preflight_gate.json")):
        raise ArtifactCompatibilityError("oracle stage requires a passing preflight")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    profile = JacobiRBCudaProfile()
    reference = _CertifiedReference(root_seed=ORACLE_ROOT_SEED, profile=profile)
    pair_count = 262_144
    cluster_count = len(ORACLE_PATH_IDS)
    cluster_size = pair_count // cluster_count
    rng = np.random.default_rng(ORACLE_ROOT_SEED)
    s, delta = 0.40, 0.05
    coefficient = 0.5 * math.exp(-2.0 * s)
    y = _sample_linear_density(rng.random(pair_count), coefficient)
    exact = _analytic_teacher_exact_reverse(
        y,
        s=s,
        delta=delta,
        device=device,
        reference=reference,
        rng=rng,
        operation_base=1_000,
    )
    numerical: dict[int, np.ndarray] = {}
    id_cursor = 2_000
    for microsteps in REFINEMENT_M:
        numerical[microsteps] = _analytic_teacher_numerical_reverse(
            y,
            s=s,
            delta=delta,
            microsteps=microsteps,
            device=device,
            reference=reference,
            operation_base=id_cursor,
        )
        id_cursor += 100
    exact_legendre = _legendre_matrix(exact)
    y_legendre = _legendre_matrix(y)
    denominator = np.sqrt(np.mean((exact_legendre - y_legendre) ** 2, axis=0))
    if np.any(denominator <= 0.0):
        raise ReverseControllerCLIError("analytic teacher normalization is degenerate")
    teacher_rows: list[dict[str, Any]] = []
    family_blocks: list[np.ndarray] = []
    family_names: list[str] = []
    for microsteps in REFINEMENT_M:
        difference = (_legendre_matrix(numerical[microsteps]) - exact_legendre) / denominator
        clusters = difference.reshape(cluster_count, cluster_size, 8).mean(axis=1)
        family_blocks.append(clusters)
        family_names.extend([f"teacher_M{microsteps}_legendre_{degree}" for degree in range(1, 9)])
        for degree in range(8):
            teacher_rows.append({
                "microsteps": microsteps,
                "legendre_degree": degree + 1,
                "normalized_bias": float(clusters[:, degree].mean()),
            })
    refinement = (_legendre_matrix(numerical[8]) - _legendre_matrix(numerical[4])) / denominator
    refinement_clusters = refinement.reshape(cluster_count, cluster_size, 8).mean(axis=1)
    family_blocks.append(refinement_clusters)
    family_names.extend([f"teacher_M8_vs_M4_legendre_{degree}" for degree in range(1, 9)])
    teacher_max_t = _two_sided_studentized_max_t(
        np.concatenate(family_blocks, axis=1),
        confidence=SIMULTANEOUS_CONFIDENCE,
        replicates=BOOTSTRAP_REPLICATES,
        seed=ORACLE_ROOT_SEED,
        names=family_names,
    )
    teacher_upper = teacher_max_t["simultaneous_upper_absolute"]
    teacher_pass = all(float(value) <= 0.01 for value in teacher_upper.values())

    null_y = rng.random(pair_count)
    null_outputs: dict[int, np.ndarray] = {}
    null_bitwise: dict[int, int] = {}
    for index, microsteps in enumerate(REFINEMENT_M):
        operation = 10_000 + index * 100
        null_outputs[microsteps] = _analytic_teacher_numerical_reverse(
            null_y,
            s=s,
            delta=delta,
            microsteps=microsteps,
            device=device,
            reference=reference,
            operation_base=operation,
            score_function=lambda values, _time: np.zeros_like(values),
        )
        pure = _pure_reference_composition(
            null_y,
            delta=delta,
            microsteps=microsteps,
            device=device,
            reference=reference,
            operation_base=operation,
        )
        null_bitwise[microsteps] = int(np.array_equal(null_outputs[microsteps], pure))
    null_blocks: list[np.ndarray] = []
    null_names: list[str] = []
    for microsteps in REFINEMENT_M:
        block = _legendre_matrix(null_outputs[microsteps]).reshape(
            cluster_count, cluster_size, 8
        ).mean(axis=1)
        null_blocks.append(block)
        null_names.extend(
            [f"stationary_M{microsteps}_legendre_{degree}" for degree in range(1, 9)]
        )
    null_clusters = np.concatenate(null_blocks, axis=1)
    null_max_t = _two_sided_studentized_max_t(
        null_clusters,
        confidence=0.99,
        replicates=BOOTSTRAP_REPLICATES,
        seed=ORACLE_ROOT_SEED + 1,
        names=null_names,
    )
    # Existing law control is a confidence-containment test, not a 0.01 teacher margin.
    null_pass = all(
        float(null_max_t["simultaneous_lower"][name]) <= 0.0 <= float(null_max_t["simultaneous_upper"][name])
        for name in null_max_t["family_names"]
    ) and all(null_bitwise.values())
    _write_csv(run_dir / "oracle_linear_teacher_metrics.csv", teacher_rows)
    _write_csv(
        run_dir / "oracle_stationary_null_metrics.csv",
        [
            {
                "microsteps": microsteps,
                "legendre_degree": degree,
                "mean": float(null_clusters[:, index * 8 + degree - 1].mean()),
                "simultaneous_lower": float(null_max_t["simultaneous_lower"][f"stationary_M{microsteps}_legendre_{degree}"]),
                "simultaneous_upper": float(null_max_t["simultaneous_upper"][f"stationary_M{microsteps}_legendre_{degree}"]),
            }
            for index, microsteps in enumerate(REFINEMENT_M)
            for degree in range(1, 9)
        ],
    )
    _freeze_json(run_dir / "oracle_teacher_max_t.json", teacher_max_t)
    _freeze_json(run_dir / "oracle_null_max_t.json", null_max_t)
    reference_record = reference.record()
    checks = {
        "stationary_null": null_pass,
        "stationary_null_bitwise_reference_composition": all(null_bitwise.values()),
        "bounded_linear_teacher": teacher_pass,
        "teacher_M2_M4_M8_complete": set(numerical) == set(REFINEMENT_M),
        "teacher_boundary_finite": all(np.isfinite(value).all() and np.all((value >= 0.0) & (value <= 1.0)) for value in numerical.values()),
        "certificate_fraction": float(reference_record["certificate_fraction"]) == 1.0,
        "fallback_fraction": float(reference_record["fallback_fraction"]) <= RESOURCE_THRESHOLDS["maximum_fallback_fraction"],
        "fallback_time": float(reference_record["fallback_time_fraction"]) <= RESOURCE_THRESHOLDS["maximum_fallback_time_fraction"],
        "forbidden_counts": all(int(value) == 0 for value in reference_record["forbidden_counts"].values()),
        "physical_panel_unopened": not (run_dir / "forward").exists(),
    }
    gate = _gate_record(
        "oracle",
        checks,
        numerically_valid=all(
            checks[name]
            for name in (
                "teacher_M2_M4_M8_complete", "teacher_boundary_finite",
                "certificate_fraction", "forbidden_counts",
            )
        ),
        resource_valid=all(
            checks[name] for name in ("fallback_fraction", "fallback_time")
        ),
        scalar_pair_count=pair_count,
        stationary_null_max_t=null_max_t,
        bounded_teacher_max_t=teacher_max_t,
        exact_reference=reference_record,
        image_state_inspected=0,
        physical_authorizing_statistics_opened=0,
    )
    _freeze_json(gate_path, gate)
    return gate


def _matching_indices(device: torch.device) -> tuple[tuple[Tensor, Tensor], ...]:
    return tuple(
        (
            torch.as_tensor(tails, dtype=torch.long, device=device),
            torch.as_tensor(heads, dtype=torch.long, device=device),
        )
        for tails, heads in _cuda_controls._matching_arrays()  # noqa: SLF001
    )


def _scatter_later_fraction(
    states: Tensor,
    *,
    phase: int,
    later: Tensor,
) -> Tensor:
    tails, heads = _matching_indices(states.device)[PHASE_MATCHINGS[int(phase)]]
    pair = states[:, tails] + states[:, heads]
    active = pair > 0.0
    output = states.clone()
    tail_values = output[:, tails]
    head_values = output[:, heads]
    next_head = pair * later
    next_tail = pair - next_head
    tail_values[active] = next_tail[active]
    head_values[active] = next_head[active]
    output[:, tails] = tail_values
    output[:, heads] = head_values
    return output


def _prefix_risk_rows(
    pre_states: Tensor,
    *,
    path_ids: Sequence[int],
    outer_step: int,
    phase: int,
    midpoint_index: int,
    midpoint_fraction: float,
    controller: Any,
    reference: _CertifiedReference,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    device = pre_states.device
    tails, heads = _matching_indices(device)[PHASE_MATCHINGS[int(phase)]]
    pair = pre_states[:, tails] + pre_states[:, heads]
    active = pair > 0.0
    current = torch.zeros_like(pair)
    current[active] = pre_states[:, heads][active] / pair[active]
    full_exposure = _controller.phase_exposure(pair, PHASE_DURATIONS[int(phase)])
    ids = _controller.controller_transition_ids(
        path_ids,
        outer_step=outer_step,
        phase=phase,
        reverse_microstep=midpoint_index,
        role="partial_phase_target_prefix",
        device=device,
    )
    result = reference(
        head_fraction=current,
        exposure=(full_exposure * float(midpoint_fraction)).contiguous(),
        transition_ids=ids,
        role="partial_phase_target_prefix",
    )
    later = result.later_head_fraction.to(dtype=torch.float64)
    target = result.denoising_target.to(dtype=torch.float64)
    endpoint = _scatter_later_fraction(pre_states, phase=phase, later=later)
    reverse_time = _controller.internal_reverse_time(
        outer_step, phase, midpoint_fraction
    )
    count = endpoint.shape[0]
    inputs = ModelInputs(
        later_full_state=endpoint.to(dtype=torch.float32),
        reverse_time=torch.full((count,), reverse_time, dtype=torch.float64, device=device),
        phase=torch.full((count,), phase, dtype=torch.long, device=device),
        color=torch.full((count,), PHASE_MATCHINGS[phase], dtype=torch.long, device=device),
        duration=torch.full((count,), PHASE_DURATIONS[phase], dtype=torch.float32, device=device),
        label=torch.full((count,), 3, dtype=torch.long, device=device),
    )
    with torch.no_grad():
        combined = controller(inputs).to(dtype=torch.float64)
        baseline = controller.baseline_prediction(inputs).to(dtype=torch.float64)
    zero_improvement = torch.mean(target * target - (target - combined) ** 2, dim=1)
    baseline_improvement = torch.mean(
        (target - baseline) ** 2 - (target - combined) ** 2, dim=1
    )
    health = {
        "target_finite": int(torch.isfinite(target).all()),
        "prediction_finite": int(torch.isfinite(combined).all()),
        "state_finite": int(torch.isfinite(endpoint).all()),
        "state_nonnegative": int(not torch.any(endpoint < 0.0)),
        "maximum_mass_error": float(torch.max(torch.abs(endpoint.sum(dim=1) - pre_states.sum(dim=1))).item()),
        "target_modified": 0,
        "branch_replaced_main_state": 0,
    }
    return (
        zero_improvement.detach().cpu().numpy(),
        baseline_improvement.detach().cpu().numpy(),
        health,
    )


def _state_checkpoint_paths(run_dir: Path, step: int) -> tuple[Path, Path]:
    directory = run_dir / "forward"
    return (
        directory / f"checkpoint-step-{step:04d}.npz",
        directory / f"checkpoint-step-{step:04d}.json",
    )


def _load_valid_state_checkpoint_unchecked(
    run_dir: Path,
    step: int,
    *,
    expected_input_sha256: str | None,
) -> tuple[Tensor | None, dict[str, Any] | None]:
    state_path, record_path = _state_checkpoint_paths(run_dir, step)
    if not state_path.is_file() or not record_path.is_file():
        return None, None
    record = _load_json(record_path)
    manifest = _load_json(run_dir / "run_manifest.json")
    selected_step = step - 1 if step % 16 == 0 else None
    expected_local_paths: set[str] = set()
    expected_control_paths: set[str] = set()
    if selected_step is not None:
        expected_local_paths = {
            path.relative_to(run_dir).as_posix()
            for path in _expected_local_paths(run_dir, selected_step)
        }
        expected_local_paths.add(
            (run_dir / "local" / f"summary-step-{selected_step:04d}.json")
            .relative_to(run_dir)
            .as_posix()
        )
        if selected_step in CONTROL_ANCHORS:
            expected_control_paths = {
                *{
                    (
                        run_dir
                        / "control"
                        / "audit"
                        / f"one-phase-anchor-{selected_step:04d}-phase-{phase}.npz"
                    )
                    .relative_to(run_dir)
                    .as_posix()
                    for phase in range(7)
                },
                (
                    run_dir
                    / "control"
                    / "audit"
                    / f"eight-phase-anchor-{selected_step:04d}.npz"
                )
                .relative_to(run_dir)
                .as_posix(),
            }
    if (
        int(record.get("committed", 0)) != 1
        or
        record.get("state_file_sha256") != file_fingerprint(state_path)
        or int(record.get("state_file_size", -1)) != state_path.stat().st_size
        or record.get("input_state_sha256") != expected_input_sha256
        or int(record.get("step", -1)) != step
        or int(record.get("start_step", -1)) != step - SHARD_STEPS
        or int(record.get("end_step", -1)) != step
        or record.get("root_seed") != CONTROLLER_ROOT_SEED
        or record.get("namespace_version") != NAMESPACE_VERSION
        or record.get("source_fingerprint") != manifest.get("source_fingerprint")
        or record.get("scientific_config_sha256")
        != manifest.get("scientific_config_sha256")
        or record.get("path_plan_sha256") != manifest.get("path_plan_sha256")
    ):
        return None, None
    for field in ("local_artifacts", "control_artifacts"):
        values = record.get(field, ())
        if not isinstance(values, list):
            return None, None
        for item in values:
            if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
                return None, None
            artifact = run_dir / str(item["path"])
            if (
                not artifact.is_file()
                or item.get("sha256") != file_fingerprint(artifact)
                or ("size" in item and int(item["size"]) != artifact.stat().st_size)
            ):
                return None, None
    actual_local = [str(item["path"]) for item in record["local_artifacts"]]
    actual_control = [str(item["path"]) for item in record["control_artifacts"]]
    if (
        len(actual_local) != len(set(actual_local))
        or set(actual_local) != expected_local_paths
        or len(actual_control) != len(set(actual_control))
        or set(actual_control) != expected_control_paths
    ):
        return None, None
    arrays = _load_npz(state_path)
    if set(arrays) != {"states"}:
        return None, None
    raw_states = np.asarray(arrays["states"])
    if raw_states.dtype != np.dtype(np.float64):
        return None, None
    states = np.ascontiguousarray(raw_states)
    if (
        states.shape != (len(PHYSICAL_PATH_IDS), PATH_STATE_SIZE)
        or not np.isfinite(states).all()
        or np.any(states < 0.0)
        or float(np.max(np.abs(states.sum(axis=1) - 1.0)))
        > RESOURCE_THRESHOLDS["maximum_mass_error"]
    ):
        return None, None
    digest = hashlib.sha256(states.tobytes(order="C")).hexdigest()
    if record.get("state_array_sha256") != digest:
        return None, None
    return torch.from_numpy(states.copy()), record


def _load_valid_state_checkpoint(
    run_dir: Path,
    step: int,
    *,
    expected_input_sha256: str | None,
) -> tuple[Tensor | None, dict[str, Any] | None]:
    try:
        return _load_valid_state_checkpoint_unchecked(
            run_dir, step, expected_input_sha256=expected_input_sha256
        )
    except (ArtifactCompatibilityError, OSError, ValueError, KeyError, TypeError):
        # This validator is used only for mutable child restart tails.  A bad
        # child shard is uncommitted evidence and is recomputed atomically;
        # immutable parent provenance is verified by a separate fail-closed path.
        return None, None


def _persist_state_checkpoint(
    run_dir: Path,
    *,
    step: int,
    states: np.ndarray,
    input_state_sha256: str | None,
    scheduler_record: Mapping[str, Any],
    local_artifacts: Sequence[Mapping[str, Any]],
    control_artifacts: Sequence[Mapping[str, Any]],
    branch_diagnostics: Mapping[str, Any],
    wall_elapsed_seconds: float,
) -> dict[str, Any]:
    state_path, record_path = _state_checkpoint_paths(run_dir, step)
    state_record = _atomic_npz(state_path, {"states": states})
    state_hash = hashlib.sha256(np.ascontiguousarray(states).tobytes(order="C")).hexdigest()
    manifest = _load_json(run_dir / "run_manifest.json")
    record = {
        "schema": RUN_SCHEMA + "-forward-checkpoint",
        "schema_version": 1,
        "step": step,
        "start_step": step - SHARD_STEPS,
        "end_step": step,
        "root_seed": CONTROLLER_ROOT_SEED,
        "namespace_version": NAMESPACE_VERSION,
        "source_fingerprint": manifest["source_fingerprint"],
        "scientific_config_sha256": manifest["scientific_config_sha256"],
        "path_plan_sha256": manifest["path_plan_sha256"],
        "input_state_sha256": input_state_sha256,
        "state_file_sha256": state_record["sha256"],
        "state_file_size": state_record["size"],
        "state_array_sha256": state_hash,
        "scheduler": dict(scheduler_record),
        "local_artifacts": list(local_artifacts),
        "control_artifacts": list(control_artifacts),
        "branch_diagnostics": dict(branch_diagnostics),
        "wall_elapsed_seconds": float(wall_elapsed_seconds),
        "committed": 1,
    }
    atomic_write_json(record_path, _normalized(record))
    return record


def _expected_local_paths(run_dir: Path, selected_step: int) -> tuple[Path, ...]:
    return tuple(
        run_dir / "local" / f"shard-step-{selected_step:04d}-pathgroup-{group:02d}.json"
        for group in range(8)
    )


def _valid_local_artifacts(run_dir: Path, selected_step: int) -> bool:
    try:
        paths = _expected_local_paths(run_dir, selected_step)
        if not all(path.is_file() and _load_json(path).get("committed") == 1 for path in paths):
            return False
        summary = run_dir / "local" / f"summary-step-{selected_step:04d}.json"
        record = _load_json(summary)
        bindings = record.get("path_group_artifacts")
        if int(record.get("committed", 0)) != 1 or not isinstance(bindings, list) or len(bindings) != 8:
            return False
        expected = {path.relative_to(run_dir).as_posix(): path for path in paths}
        bound_names = [str(item.get("path")) for item in bindings if isinstance(item, Mapping)]
        return (
            len(bound_names) == len(bindings)
            and len(set(bound_names)) == len(bindings)
            and set(bound_names) == set(expected)
            and all(
            isinstance(item, Mapping)
            and item.get("path") in expected
            and item.get("sha256") == file_fingerprint(expected[str(item["path"])])
            and int(item.get("size", -1)) == expected[str(item["path"])].stat().st_size
            for item in bindings
            )
        )
    except (ArtifactCompatibilityError, OSError, ValueError, KeyError, TypeError):
        return False


def _control_anchor_artifacts_exist(run_dir: Path, step: int) -> bool:
    if step not in CONTROL_ANCHORS:
        return True
    one = [run_dir / "control" / "audit" / f"one-phase-anchor-{step:04d}-phase-{phase}.npz" for phase in range(7)]
    eight = run_dir / "control" / "audit" / f"eight-phase-anchor-{step:04d}.npz"
    return all(path.is_file() for path in one) and eight.is_file()


def _generate_forward_and_local(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    controller: Any,
    device: torch.device,
) -> dict[str, Any]:
    profile = JacobiRBCudaProfile()
    states = _initial_states(args.parent_coarse_residual_run_dir, len(PHYSICAL_PATH_IDS), device)
    initial_hash = hashlib.sha256(states.detach().cpu().numpy().tobytes(order="C")).hexdigest()
    input_hash: str | None = initial_hash
    reference = _CertifiedReference(root_seed=CONTROLLER_ROOT_SEED, profile=profile)
    scheduler_transition_count = 0
    scheduler_certified_count = 0
    scheduler_fallback_count = 0
    scheduler_fallback_seconds = 0.0
    scheduler_elapsed = 0.0
    forbidden = {name: 0 for name in FORBIDDEN_DIAGNOSTICS}
    maximum_mass_error = 0.0
    branch_health_violation_count = 0
    started = time.perf_counter()
    for start_step in range(0, OUTER_STEPS, SHARD_STEPS):
        selected_step = start_step + 7 if start_step % 16 == 8 else None
        final_step = start_step + SHARD_STEPS
        checkpoint, checkpoint_record = _load_valid_state_checkpoint(
            run_dir, final_step, expected_input_sha256=input_hash
        )
        checkpoint_complete = (
            checkpoint is not None
            and (selected_step is None or _valid_local_artifacts(run_dir, selected_step))
            and (selected_step is None or _control_anchor_artifacts_exist(run_dir, selected_step))
        )
        if checkpoint_complete:
            states = checkpoint.to(device=device, dtype=torch.float64).contiguous()
            input_hash = str(checkpoint_record["state_array_sha256"])
            continue
        start_states = states.detach().clone()
        shard_started = time.perf_counter()
        capture = selected_step is not None
        result = run_exact_multipath_shard(
            states,
            path_ids=PHYSICAL_PATH_IDS,
            start_step=start_step,
            root_seed=CONTROLLER_ROOT_SEED,
            profile=profile,
            group_sizes=(8,) * 8,
            capture_phase_state_trace=capture,
            capture_training_payload=capture,
        )
        diag = result.diagnostics
        scheduler_transition_count += int(diag.get("transition_count", 0))
        scheduler_certified_count += int(diag.get("certified_count", 0))
        scheduler_fallback_count += int(diag.get("fallback_count", 0))
        scheduler_fallback_seconds += float(diag.get("fallback_elapsed_seconds", 0.0))
        scheduler_elapsed += float(diag.get("elapsed_seconds", 0.0))
        maximum_mass_error = max(maximum_mass_error, float(diag.get("maximum_mass_error", math.inf)))
        for name in FORBIDDEN_DIAGNOSTICS:
            forbidden[name] += int(diag.get(name, 0))
        local_records: list[dict[str, Any]] = []
        control_records: list[dict[str, Any]] = []
        branch_diagnostics: dict[str, Any] = {
            "transition_count": 0,
            "certified_count": 0,
            "fallback_count": 0,
            "fallback_seconds": 0.0,
            "elapsed_seconds": 0.0,
            "forbidden_counts": {name: 0 for name in FORBIDDEN_DIAGNOSTICS},
        }
        if selected_step is not None:
            before_reference = {
                "transition_count": reference.transition_count,
                "certified_count": reference.certified_count,
                "fallback_count": reference.fallback_count,
                "fallback_seconds": reference.fallback_seconds,
                "elapsed_seconds": reference.elapsed_seconds,
                "forbidden_counts": dict(reference.forbidden),
            }
            payload = result.capture_payload
            if payload is None:
                raise ReverseControllerCLIError("selected shard lacks exact capture payload")
            phase_states = np.asarray(payload.post_phase_states, dtype=np.float64)
            if phase_states.shape != (56, len(PHYSICAL_PATH_IDS), PATH_STATE_SIZE):
                raise ReverseControllerCLIError("selected shard phase-state trace is malformed")
            path_rows = [
                {"path_id": int(path_id), "cells": []}
                for path_id in PHYSICAL_PATH_IDS
            ]
            branch_maximum_mass_error = 0.0
            branch_health_violation_count = 0
            for phase in range(7):
                pre_index = 7 * 7 + phase - 1
                pre_np = phase_states[pre_index]
                pre = torch.as_tensor(pre_np.copy(order="C"), dtype=torch.float64, device=device).contiguous()
                for midpoint_index, fraction in enumerate(_controller.MIDPOINT_FRACTIONS[8]):
                    for group in range(8):
                        first, last = group * 8, (group + 1) * 8
                        zero, baseline, health = _prefix_risk_rows(
                            pre[first:last],
                            path_ids=PHYSICAL_PATH_IDS[first:last],
                            outer_step=selected_step,
                            phase=phase,
                            midpoint_index=midpoint_index,
                            midpoint_fraction=fraction,
                            controller=controller,
                            reference=reference,
                        )
                        if not all(int(health[name]) == 1 for name in ("target_finite", "prediction_finite", "state_finite", "state_nonnegative")) or float(health["maximum_mass_error"]) > RESOURCE_THRESHOLDS["maximum_mass_error"]:
                            branch_health_violation_count += 1
                        branch_maximum_mass_error = max(
                            branch_maximum_mass_error,
                            float(health["maximum_mass_error"]),
                        )
                        for offset, path_index in enumerate(range(first, last)):
                            path_rows[path_index]["cells"].append({
                                "phase": phase,
                                "midpoint_index": midpoint_index,
                                "midpoint_fraction": fraction,
                                "combined_vs_zero": float(zero[offset]),
                                "combined_vs_baseline": float(baseline[offset]),
                            })
            for group in range(8):
                local_path = _expected_local_paths(run_dir, selected_step)[group]
                record = {
                    "schema": RUN_SCHEMA + "-local-risk-sufficient-statistics",
                    "schema_version": 1,
                    "selected_outer_step": selected_step,
                    "forward_outer_quartile": selected_step // 128,
                    "reverse_quartile": 3 - selected_step // 128,
                    "reverse_start": int(selected_step // 128 == 3),
                    "path_rows": path_rows[group * 8 : (group + 1) * 8],
                    "joined_input_target_rows_persisted": 0,
                    "target_modified": 0,
                    "committed": 1,
                }
                atomic_write_json(local_path, _normalized(record))
                local_records.append({"path": local_path.relative_to(run_dir).as_posix(), "sha256": file_fingerprint(local_path), "size": local_path.stat().st_size})
            branch_diagnostics = {
                "transition_count": reference.transition_count - int(before_reference["transition_count"]),
                "certified_count": reference.certified_count - int(before_reference["certified_count"]),
                "fallback_count": reference.fallback_count - int(before_reference["fallback_count"]),
                "fallback_seconds": reference.fallback_seconds - float(before_reference["fallback_seconds"]),
                "elapsed_seconds": reference.elapsed_seconds - float(before_reference["elapsed_seconds"]),
                "forbidden_counts": {
                    name: reference.forbidden[name] - int(before_reference["forbidden_counts"][name])
                    for name in FORBIDDEN_DIAGNOSTICS
                },
                "maximum_mass_error": branch_maximum_mass_error,
                "health_violation_count": branch_health_violation_count,
            }
            summary_path = run_dir / "local" / f"summary-step-{selected_step:04d}.json"
            summary_record = {
                "schema": RUN_SCHEMA + "-local-risk-shard-summary",
                "schema_version": 1,
                "selected_outer_step": selected_step,
                "branch_diagnostics": branch_diagnostics,
                "path_group_artifacts": list(local_records),
                "committed": 1,
            }
            atomic_write_json(summary_path, _normalized(summary_record))
            local_records.append({"path": summary_path.relative_to(run_dir).as_posix(), "sha256": file_fingerprint(summary_path), "size": summary_path.stat().st_size})
            if selected_step in CONTROL_ANCHORS:
                audit_dir = run_dir / "control" / "audit"
                for phase in range(7):
                    pre_index = 7 * 7 + phase - 1
                    post_index = 7 * 7 + phase
                    artifact = _atomic_npz(
                        audit_dir / f"one-phase-anchor-{selected_step:04d}-phase-{phase}.npz",
                        {"earlier_state": phase_states[pre_index], "later_state": phase_states[post_index], "path_ids": np.asarray(PHYSICAL_PATH_IDS, dtype=np.int64)},
                    )
                    control_records.append({**artifact, "path": (audit_dir / f"one-phase-anchor-{selected_step:04d}-phase-{phase}.npz").relative_to(run_dir).as_posix()})
                end_index = 7 * 7 + 6
                earlier_index = end_index - 8
                artifact = _atomic_npz(
                    audit_dir / f"eight-phase-anchor-{selected_step:04d}.npz",
                    {"earlier_state": phase_states[earlier_index], "later_state": phase_states[end_index], "path_ids": np.asarray(PHYSICAL_PATH_IDS, dtype=np.int64)},
                )
                control_records.append({**artifact, "path": (audit_dir / f"eight-phase-anchor-{selected_step:04d}.npz").relative_to(run_dir).as_posix()})
        checkpoint_record = _persist_state_checkpoint(
            run_dir,
            step=final_step,
            states=result.committed_final_states,
            input_state_sha256=input_hash,
            scheduler_record=result.to_record(),
            local_artifacts=local_records,
            control_artifacts=control_records,
            branch_diagnostics=branch_diagnostics,
            wall_elapsed_seconds=time.perf_counter() - shard_started,
        )
        states = result.final_states
        input_hash = str(checkpoint_record["state_array_sha256"])
        print(
            f"reverse-controller cache step {final_step}/{OUTER_STEPS} "
            f"elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )
    return _aggregate_committed_cache(run_dir, initial_state_sha256=initial_hash)


def _aggregate_committed_cache(
    run_dir: Path,
    *,
    initial_state_sha256: str,
) -> dict[str, Any]:
    scheduler_transitions = 0
    scheduler_certified = 0
    branch_transitions = 0
    branch_certified = 0
    fallback_count = 0
    fallback_seconds = 0.0
    backend_seconds = 0.0
    wall_seconds = 0.0
    maximum_mass_error = 0.0
    branch_health_violation_count = 0
    forbidden = {name: 0 for name in FORBIDDEN_DIAGNOSTICS}
    expected_input: str | None = initial_state_sha256
    final_hash: str | None = None
    for step in range(8, 513, 8):
        state, record = _load_valid_state_checkpoint(
            run_dir, step, expected_input_sha256=expected_input
        )
        if state is None or record is None or int(record.get("committed", 0)) != 1:
            raise ReverseControllerCLIError(
                f"committed cache checkpoint {step} is missing or corrupt",
                failure_domain="restart",
                failure_code="controller_cache_checkpoint_invalid",
            )
        scheduler = record.get("scheduler")
        diagnostics = scheduler.get("diagnostics") if isinstance(scheduler, Mapping) else None
        branch = record.get("branch_diagnostics")
        if not isinstance(diagnostics, Mapping) or not isinstance(branch, Mapping):
            raise ReverseControllerCLIError("cache checkpoint diagnostics are malformed")
        scheduler_transitions += int(diagnostics.get("transition_count", 0))
        scheduler_certified += int(diagnostics.get("certified_count", 0))
        branch_transitions += int(branch.get("transition_count", 0))
        branch_certified += int(branch.get("certified_count", 0))
        fallback_count += int(diagnostics.get("fallback_count", 0)) + int(branch.get("fallback_count", 0))
        fallback_seconds += float(diagnostics.get("fallback_elapsed_seconds", 0.0)) + float(branch.get("fallback_seconds", 0.0))
        backend_seconds += float(diagnostics.get("elapsed_seconds", 0.0)) + float(branch.get("elapsed_seconds", 0.0))
        wall_seconds += float(record.get("wall_elapsed_seconds", 0.0))
        maximum_mass_error = max(maximum_mass_error, float(diagnostics.get("maximum_mass_error", math.inf)))
        maximum_mass_error = max(maximum_mass_error, float(branch.get("maximum_mass_error", math.inf)))
        branch_health_violation_count += int(branch.get("health_violation_count", 0))
        branch_forbidden = branch.get("forbidden_counts") if isinstance(branch.get("forbidden_counts"), Mapping) else {}
        for name in FORBIDDEN_DIAGNOSTICS:
            forbidden[name] += int(diagnostics.get(name, 0)) + int(branch_forbidden.get(name, 0))
        expected_input = str(record["state_array_sha256"])
        final_hash = expected_input
    total = scheduler_transitions + branch_transitions
    certified = scheduler_certified + branch_certified
    return {
        "initial_state_sha256": initial_state_sha256,
        "scheduler_transition_count_committed": scheduler_transitions,
        "branch_transition_count_committed": branch_transitions,
        "total_transition_count_committed": total,
        "certificate_fraction": certified / max(total, 1),
        "fallback_count": fallback_count,
        "fallback_fraction": fallback_count / max(total, 1),
        "fallback_time_fraction": fallback_seconds / max(backend_seconds, np.finfo(float).tiny),
        "maximum_mass_error": maximum_mass_error,
        "branch_health_violation_count": branch_health_violation_count,
        "forbidden_counts": forbidden,
        "committed_wall_elapsed_seconds": wall_seconds,
        "transitions_per_second_committed": total / max(wall_seconds, np.finfo(float).tiny),
        "final_state_sha256": final_hash,
        "checkpoint_count": 64,
        "resume_evidence_aggregated_from_committed_records": 1,
    }


def _checkpoint_hash_ledger(
    run_dir: Path,
    *,
    initial_state_sha256: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    expected_input: str | None = initial_state_sha256
    for step in range(8, 513, 8):
        state_path, record_path = _state_checkpoint_paths(run_dir, step)
        state, record = _load_valid_state_checkpoint(
            run_dir, step, expected_input_sha256=expected_input
        )
        if state is None or record is None:
            raise ReverseControllerCLIError(
                f"checkpoint ledger cannot bind step {step}",
                failure_domain="restart",
                failure_code="controller_checkpoint_ledger_invalid",
            )
        output_hash = str(record["state_array_sha256"])
        rows.append(
            {
                "step": step,
                "input_state_sha256": expected_input,
                "output_state_sha256": output_hash,
                "state_file_sha256": file_fingerprint(state_path),
                "record_file_sha256": file_fingerprint(record_path),
            }
        )
        expected_input = output_hash
    body = {
        "schema": RUN_SCHEMA + "-forward-checkpoint-hash-ledger",
        "schema_version": 1,
        "checkpoint_count": len(rows),
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": expected_input,
        "checkpoints": rows,
        "chain_complete": int(len(rows) == 64),
    }
    body["semantic_sha256"] = _semantic_hash(body)
    return body


def _assemble_local_risk(run_dir: Path) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    zero_sums = np.zeros((64, 4, 7, 8), dtype=np.float64)
    zero_counts = np.zeros((64, 4, 7, 8), dtype=np.int64)
    baseline_sums = np.zeros((64, 4), dtype=np.float64)
    baseline_counts = np.zeros((64, 4), dtype=np.int64)
    id_to_row = {path_id: index for index, path_id in enumerate(PHYSICAL_PATH_IDS)}
    for step in SELECTED_OUTER_STEPS:
        quartile = step // 128
        for group, path in enumerate(_expected_local_paths(run_dir, step)):
            record = _load_json(path)
            path_rows = record.get("path_rows")
            expected_ids = list(PHYSICAL_PATH_IDS[group * 8 : (group + 1) * 8])
            if (
                int(record.get("committed", 0)) != 1
                or int(record.get("selected_outer_step", -1)) != step
                or int(record.get("forward_outer_quartile", -1)) != quartile
                or not isinstance(path_rows, list)
                or len(path_rows) != 8
                or [int(row.get("path_id", -1)) for row in path_rows] != expected_ids
            ):
                raise ArtifactCompatibilityError(
                    f"local-risk artifact binding is invalid: {path.name}"
                )
            for row in path_rows:
                cells = row.get("cells")
                if not isinstance(cells, list) or len(cells) != 56:
                    raise ArtifactCompatibilityError(
                        f"local-risk cell count is invalid: {path.name}"
                    )
                coordinates: set[tuple[int, int]] = set()
                index = id_to_row[int(row["path_id"])]
                for cell in cells:
                    phase = int(cell["phase"])
                    midpoint = int(cell["midpoint_index"])
                    coordinate = (phase, midpoint)
                    zero_value = float(cell["combined_vs_zero"])
                    baseline_value = float(cell["combined_vs_baseline"])
                    if (
                        not 0 <= phase < 7
                        or not 0 <= midpoint < 8
                        or coordinate in coordinates
                        or float(cell.get("midpoint_fraction", math.nan))
                        != float(_controller.MIDPOINT_FRACTIONS[8][midpoint])
                        or not math.isfinite(zero_value)
                        or not math.isfinite(baseline_value)
                    ):
                        raise ArtifactCompatibilityError(
                            f"local-risk cell semantics are invalid: {path.name}"
                        )
                    coordinates.add(coordinate)
                    zero_sums[index, quartile, phase, midpoint] += zero_value
                    zero_counts[index, quartile, phase, midpoint] += 1
                    baseline_sums[index, quartile] += baseline_value
                    baseline_counts[index, quartile] += 1
                if len(coordinates) != 56:
                    raise ArtifactCompatibilityError(
                        f"local-risk coordinates are incomplete: {path.name}"
                    )
    if not np.all(zero_counts == 8) or not np.all(baseline_counts == 448):
        raise ReverseControllerCLIError("local risk cells have invalid aggregate counts")
    zero = zero_sums / zero_counts
    baseline = baseline_sums / baseline_counts
    names: list[str] = []
    columns: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    fractions = _controller.MIDPOINT_FRACTIONS[8]
    for quartile in range(4):
        for phase in range(7):
            for midpoint, fraction in enumerate(fractions):
                name = f"combined_vs_zero.q{quartile}.phase{phase}.midpoint{midpoint}"
                names.append(name)
                columns.append(zero[:, quartile, phase, midpoint])
                rows.append({
                    "contrast": "combined_vs_zero",
                    "forward_outer_quartile": quartile,
                    "reverse_quartile": 3 - quartile,
                    "reverse_start": int(quartile == 3),
                    "phase": phase,
                    "midpoint_index": midpoint,
                    "midpoint_fraction": fraction,
                    "mean_improvement": float(zero[:, quartile, phase, midpoint].mean()),
                })
    for quartile in range(4):
        name = f"combined_vs_baseline.q{quartile}"
        names.append(name)
        columns.append(baseline[:, quartile])
        rows.append({
            "contrast": "combined_vs_baseline",
            "forward_outer_quartile": quartile,
            "reverse_quartile": 3 - quartile,
            "reverse_start": int(quartile == 3),
            "phase": "pooled",
            "midpoint_index": "pooled",
            "midpoint_fraction": "pooled",
            "mean_improvement": float(baseline[:, quartile].mean()),
        })
    matrix = np.stack(columns, axis=1)
    if matrix.shape != (64, 228):
        raise AssertionError("local-risk family is not the frozen 64x228 matrix")
    return matrix, names, rows


def _separated_cache_schema_valid(run_dir: Path) -> bool:
    input_keys = {
        "later_full_state", "earlier_state", "later_state", "state", "states"
    }
    target_keys = {
        "denoising_target", "oracle_target", "target", "labels", "label"
    }
    try:
        for path in run_dir.rglob("*.npz"):
            keys = set(_load_npz(path))
            if keys.intersection(input_keys) and keys.intersection(target_keys):
                return False
        input_probe = _load_npz(run_dir / "input_probe.npz")
        label_probe = _load_npz(run_dir / "label_audit_probe.npz")
        if set(input_probe) != {"later_full_state"} or set(label_probe) != {
            "denoising_target"
        }:
            return False
        for step in SELECTED_OUTER_STEPS:
            for path in _expected_local_paths(run_dir, step):
                record = _load_json(path)
                if any(
                    key in record
                    for key in (
                        "later_full_state", "earlier_state", "denoising_target",
                        "oracle_target", "target",
                    )
                ):
                    return False
        return True
    except (ArtifactCompatibilityError, OSError, ValueError, KeyError, TypeError):
        return False


def _cache_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    gate_path = run_dir / "cache_gate.json"
    committed_gate = _registered_stage_gate(run_dir, gate_path)
    if committed_gate is not None:
        return committed_gate
    configure_exact_torch_backend()
    if not _passed(_load_json(run_dir / "oracle_gate.json")):
        raise ArtifactCompatibilityError("cache stage requires a passing oracle gate")
    device = torch.device(args.device)
    controller = _controller.load_frozen_controller(args.parent_coarse_residual_run_dir, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    execution = _generate_forward_and_local(run_dir, args, controller=controller, device=device)
    checkpoint_ledger = _checkpoint_hash_ledger(
        run_dir,
        initial_state_sha256=str(execution["initial_state_sha256"]),
    )
    _commit_recoverable_json(
        run_dir, run_dir / "forward" / "checkpoint_hashes.json", checkpoint_ledger
    )
    matrix, names, rows = _assemble_local_risk(run_dir)
    result = _one_sided_matrix_max_t(
        matrix,
        names=names,
        confidence=SIMULTANEOUS_CONFIDENCE,
        replicates=BOOTSTRAP_REPLICATES,
        seed=LOCAL_BOOTSTRAP_SEED,
    )
    lower = result.get("lower_bounds", {})
    if not isinstance(lower, Mapping):
        raise ReverseControllerCLIError("local max-T result lacks simultaneous lower bounds")
    for row in rows:
        name = names[rows.index(row)]
        row["simultaneous_lower_bound"] = float(lower[name])
    _write_csv(run_dir / "local_time_phase_metrics.csv", rows)
    _write_csv(
        run_dir / "local_internal_fraction_metrics.csv",
        [row for row in rows if row["midpoint_fraction"] != "pooled"],
    )
    _freeze_json(run_dir / "local_risk_max_t.json", result)
    persisted = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    checks = {
        "main_paths_complete": all(_state_checkpoint_paths(run_dir, step)[0].is_file() for step in range(8, 513, 8)),
        "checkpoint_hash_ledger": (
            int(checkpoint_ledger.get("chain_complete", 0)) == 1
            and int(checkpoint_ledger.get("checkpoint_count", -1)) == 64
            and checkpoint_ledger.get("final_state_sha256")
            == execution.get("final_state_sha256")
        ),
        "main_transition_count": int(execution["scheduler_transition_count_committed"]) == MAIN_TRANSITIONS,
        "branch_transition_count": int(execution["branch_transition_count_committed"]) == LOCAL_BRANCH_TRANSITIONS,
        "local_family_size": len(names) == 228 and int(result.get("family_size", -1)) == 228,
        "local_all_simultaneous_lower_positive": len(lower) == 228 and all(float(value) > 0.0 for value in lower.values()),
        "terminal_near_reverse_start_controlled": all(float(lower[name]) > 0.0 for name in names if ".q3." in name or name.endswith(".q3")),
        "certificate_fraction": float(execution["certificate_fraction"]) == 1.0,
        "fallback_fraction": float(execution["fallback_fraction"]) <= RESOURCE_THRESHOLDS["maximum_fallback_fraction"],
        "fallback_time": float(execution["fallback_time_fraction"]) <= RESOURCE_THRESHOLDS["maximum_fallback_time_fraction"],
        "mass_error": float(execution["maximum_mass_error"]) <= RESOURCE_THRESHOLDS["maximum_mass_error"],
        "branch_health": int(execution["branch_health_violation_count"]) == 0,
        "forbidden_counts": all(int(value) == 0 for value in execution["forbidden_counts"].values()),
        "throughput": float(execution["transitions_per_second_committed"]) >= RESOURCE_THRESHOLDS["minimum_transitions_per_second"],
        "persisted_size": persisted <= RESOURCE_THRESHOLDS["maximum_persisted_bytes"],
        "no_joined_cache": _separated_cache_schema_valid(run_dir),
        "target_immutable": all(int(_load_json(path).get("target_modified", 1)) == 0 for step in SELECTED_OUTER_STEPS for path in _expected_local_paths(run_dir, step)),
    }
    gate = _gate_record(
        "cache",
        checks,
        numerically_valid=all(
            checks[name]
            for name in (
                "main_paths_complete", "checkpoint_hash_ledger",
                "main_transition_count", "branch_transition_count",
                "local_family_size", "certificate_fraction", "mass_error",
                "branch_health", "forbidden_counts", "no_joined_cache",
                "target_immutable",
            )
        ),
        resource_valid=all(
            checks[name]
            for name in (
                "fallback_fraction", "fallback_time", "throughput",
                "persisted_size",
            )
        ),
        execution=execution,
        checkpoint_hash_ledger=checkpoint_ledger,
        local_risk_max_t=result,
        local_family_size=228,
        physical_path_count=64,
        persisted_artifact_bytes=persisted,
        internal_time_target_controlled=int(checks["local_all_simultaneous_lower_positive"]),
        terminal_near_reverse_start_controlled=int(checks["terminal_near_reverse_start_controlled"]),
    )
    _freeze_json(gate_path, gate)
    return gate


def _valid_control_result(
    state_path: Path,
    record_path: Path,
    *,
    expected: Mapping[str, Any],
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    try:
        if not state_path.is_file() or not record_path.is_file():
            return None, None
        record = _load_json(record_path)
        if (
            int(record.get("committed", 0)) != 1
            or record.get("state_file_sha256") != file_fingerprint(state_path)
            or int(record.get("state_file_size", -1)) != state_path.stat().st_size
            or any(record.get(key) != value for key, value in expected.items())
        ):
            return None, None
        arrays = _load_npz(state_path)
        if set(arrays) != {"state"}:
            return None, None
        raw_state = np.asarray(arrays["state"])
        if raw_state.dtype != np.dtype(np.float64):
            return None, None
        state = np.ascontiguousarray(raw_state)
        if (
            state.shape != (64, PATH_STATE_SIZE)
            or not np.isfinite(state).all()
            or np.any(state < 0.0)
            or float(np.max(np.abs(state.sum(axis=1) - 1.0)))
            > RESOURCE_THRESHOLDS["maximum_mass_error"]
        ):
            return None, None
        if record.get("state_array_sha256") != hashlib.sha256(state.tobytes(order="C")).hexdigest():
            return None, None
        diagnostics = record.get("reference_diagnostics")
        sequence = expected.get("sequence")
        microsteps = int(expected.get("microsteps", -1))
        expected_transitions = (
            len(PHYSICAL_PATH_IDS)
            * EDGES_PER_PHASE
            * 2
            * microsteps
            * len(sequence)
            if isinstance(sequence, list) and microsteps in REFINEMENT_M
            else -1
        )
        if (
            not isinstance(diagnostics, Mapping)
            or int(record.get("phase_count", -1)) != len(sequence)
            or int(record.get("full_reverse_path", 1)) != 0
            or int(record.get("states_finite", 0)) != 1
            or int(record.get("states_nonnegative", 0)) != 1
            or int(diagnostics.get("transition_count", -1)) != expected_transitions
            or int(diagnostics.get("certified_count", -1))
            != int(diagnostics.get("transition_count", -1))
            or int(diagnostics.get("fallback_count", -1)) < 0
            or int(diagnostics.get("maximum_transition_count_per_call", 1 << 30))
            > 4096
            or not math.isfinite(float(record.get("maximum_pair_mass_error", math.nan)))
            or not math.isfinite(float(record.get("maximum_simplex_mass_error", math.nan)))
            or float(record.get("maximum_pair_mass_error", math.inf))
            > RESOURCE_THRESHOLDS["maximum_mass_error"]
            or float(record.get("maximum_simplex_mass_error", math.inf))
            > RESOURCE_THRESHOLDS["maximum_mass_error"]
            or not isinstance(diagnostics.get("forbidden_counts"), Mapping)
            or any(
                int(diagnostics["forbidden_counts"].get(name, -1)) != 0
                for name in FORBIDDEN_DIAGNOSTICS
            )
        ):
            return None, None
        return state, record
    except (ArtifactCompatibilityError, OSError, ValueError, KeyError, TypeError):
        return None, None


def _reference_delta(reference: _CertifiedReference, before: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "transition_count": reference.transition_count - int(before["transition_count"]),
        "certified_count": reference.certified_count - int(before["certified_count"]),
        "fallback_count": reference.fallback_count - int(before["fallback_count"]),
        "fallback_seconds": reference.fallback_seconds - float(before["fallback_seconds"]),
        "elapsed_seconds": reference.elapsed_seconds - float(before["elapsed_seconds"]),
        "maximum_transition_count_per_call": reference.maximum_transition_count_per_call,
        "forbidden_counts": {
            name: reference.forbidden[name] - int(before["forbidden_counts"][name])
            for name in FORBIDDEN_DIAGNOSTICS
        },
    }


def _run_control_trajectory(
    run_dir: Path,
    *,
    stem: str,
    later_state: np.ndarray,
    sequence: Sequence[tuple[int, int]],
    microsteps: int,
    controller: Any,
    device: torch.device,
    profile: JacobiRBCudaProfile,
    stream_role: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    directory = run_dir / "control" / "results"
    state_path = directory / f"{stem}-M{microsteps}.npz"
    record_path = directory / f"{stem}-M{microsteps}.json"
    input_sha256 = hashlib.sha256(
        np.ascontiguousarray(later_state).tobytes(order="C")
    ).hexdigest()
    binding = {
        "stem": stem,
        "microsteps": int(microsteps),
        "sequence": [[int(step), int(phase)] for step, phase in sequence],
        "input_state_sha256": input_sha256,
        "stream_role": str(stream_role),
        "root_seed": CONTROLLER_ROOT_SEED,
        "namespace_version": NAMESPACE_VERSION,
        "profile_sha256": config_fingerprint(profile.to_dict()),
        "source_fingerprint": _load_json(run_dir / "run_manifest.json")["source_fingerprint"],
    }
    state = torch.as_tensor(
        np.ascontiguousarray(later_state).copy(), dtype=torch.float64, device=device
    ).contiguous()
    phase_records: list[dict[str, Any]] = []
    progress_dir = run_dir / "control" / "phase_checkpoints"
    for phase_index, (outer_step, phase) in enumerate(sequence):
        phase_state_path = progress_dir / (
            f"{stem}-M{microsteps}-occurrence-{phase_index:02d}.npz"
        )
        phase_record_path = progress_dir / (
            f"{stem}-M{microsteps}-occurrence-{phase_index:02d}.json"
        )
        phase_input_np = np.ascontiguousarray(
            state.detach().cpu().numpy(), dtype=np.float64
        )
        phase_binding = {
            **binding,
            "sequence": [[int(outer_step), int(phase)]],
            "input_state_sha256": hashlib.sha256(
                phase_input_np.tobytes(order="C")
            ).hexdigest(),
            "phase_occurrence_index": phase_index,
            "full_sequence_sha256": config_fingerprint(
                {"sequence": binding["sequence"]}
            ),
        }
        cached_state, cached_record = _valid_control_result(
            phase_state_path, phase_record_path, expected=phase_binding
        )
        if cached_state is not None and cached_record is not None:
            state = torch.as_tensor(
                cached_state.copy(), dtype=torch.float64, device=device
            ).contiguous()
            phase_records.append(cached_record)
            continue
        reference = _CertifiedReference(
            root_seed=CONTROLLER_ROOT_SEED,
            profile=profile,
            stream_role=stream_role,
        )
        before = {
            "transition_count": 0,
            "certified_count": 0,
            "fallback_count": 0,
            "fallback_seconds": 0.0,
            "elapsed_seconds": 0.0,
            "forbidden_counts": {name: 0 for name in FORBIDDEN_DIAGNOSTICS},
        }
        maximum_pair_error = 0.0
        maximum_simplex_error = 0.0
        phase_started = time.perf_counter()
        cohort_states: list[Tensor] = []
        for first in range(0, len(PHYSICAL_PATH_IDS), 8):
            last = min(first + 8, len(PHYSICAL_PATH_IDS))
            result = _controller.controlled_reverse_phase(
                state[first:last],
                outer_step,
                phase,
                microsteps,
                NAMESPACE_VERSION,
                controller=controller,
                reference_transition=reference,
                path_ids=PHYSICAL_PATH_IDS[first:last],
                label=3,
            )
            cohort_states.append(result.state)
            maximum_pair_error = max(
                maximum_pair_error, result.maximum_pair_mass_error
            )
            maximum_simplex_error = max(
                maximum_simplex_error, result.maximum_simplex_mass_error
            )
        state = torch.cat(cohort_states, dim=0).contiguous()
        phase_output = np.ascontiguousarray(
            state.detach().cpu().numpy(), dtype=np.float64
        )
        phase_artifact = _atomic_npz(
            phase_state_path, {"state": phase_output}
        )
        phase_record = {
            "schema": RUN_SCHEMA + "-control-phase-checkpoint",
            "schema_version": 1,
            **phase_binding,
            "phase_count": 1,
            "full_reverse_path": 0,
            "state_file_sha256": phase_artifact["sha256"],
            "state_file_size": phase_artifact["size"],
            "state_array_sha256": hashlib.sha256(
                phase_output.tobytes(order="C")
            ).hexdigest(),
            "reference_diagnostics": _reference_delta(reference, before),
            "maximum_pair_mass_error": maximum_pair_error,
            "maximum_simplex_mass_error": maximum_simplex_error,
            "states_finite": int(np.isfinite(phase_output).all()),
            "states_nonnegative": int(np.all(phase_output >= 0.0)),
            "wall_elapsed_seconds": time.perf_counter() - phase_started,
            "boundary_rejection_count": 0,
            "clip_count": 0,
            "floor_count": 0,
            "limiter_count": 0,
            "projection_count": 0,
            "renormalization_count": 0,
            "committed": 1,
        }
        atomic_write_json(phase_record_path, _normalized(phase_record))
        phase_records.append(phase_record)

    phase_artifacts = [
        {
            "path": (
                progress_dir
                / f"{stem}-M{microsteps}-occurrence-{index:02d}.json"
            ).relative_to(run_dir).as_posix(),
            "sha256": file_fingerprint(
                progress_dir
                / f"{stem}-M{microsteps}-occurrence-{index:02d}.json"
            ),
        }
        for index in range(len(sequence))
    ]
    final_binding = {**binding, "phase_checkpoint_artifacts": phase_artifacts}
    output = np.ascontiguousarray(state.detach().cpu().numpy(), dtype=np.float64)
    valid_state, valid_record = _valid_control_result(
        state_path, record_path, expected=final_binding
    )
    if (
        valid_state is not None
        and valid_record is not None
        and np.array_equal(valid_state, output)
    ):
        return valid_state, valid_record
    artifact = _atomic_npz(state_path, {"state": output})
    reference_record = {
        "transition_count": sum(int(item["reference_diagnostics"]["transition_count"]) for item in phase_records),
        "certified_count": sum(int(item["reference_diagnostics"]["certified_count"]) for item in phase_records),
        "fallback_count": sum(int(item["reference_diagnostics"]["fallback_count"]) for item in phase_records),
        "fallback_seconds": sum(float(item["reference_diagnostics"]["fallback_seconds"]) for item in phase_records),
        "elapsed_seconds": sum(float(item["reference_diagnostics"]["elapsed_seconds"]) for item in phase_records),
        "maximum_transition_count_per_call": max(int(item["reference_diagnostics"]["maximum_transition_count_per_call"]) for item in phase_records),
        "forbidden_counts": {
            name: sum(int(item["reference_diagnostics"]["forbidden_counts"][name]) for item in phase_records)
            for name in FORBIDDEN_DIAGNOSTICS
        },
    }
    record = {
        "schema": RUN_SCHEMA + "-control-trajectory",
        "schema_version": 1,
        **final_binding,
        "phase_count": len(sequence),
        "full_reverse_path": 0,
        "state_file_sha256": artifact["sha256"],
        "state_file_size": artifact["size"],
        "state_array_sha256": hashlib.sha256(output.tobytes(order="C")).hexdigest(),
        "reference_diagnostics": reference_record,
        "maximum_pair_mass_error": max(float(item["maximum_pair_mass_error"]) for item in phase_records),
        "maximum_simplex_mass_error": max(float(item["maximum_simplex_mass_error"]) for item in phase_records),
        "states_finite": int(np.isfinite(output).all()),
        "states_nonnegative": int(np.all(output >= 0.0)),
        "wall_elapsed_seconds": sum(float(item["wall_elapsed_seconds"]) for item in phase_records),
        "boundary_rejection_count": 0,
        "clip_count": 0,
        "floor_count": 0,
        "limiter_count": 0,
        "projection_count": 0,
        "renormalization_count": 0,
        "committed": 1,
    }
    atomic_write_json(record_path, _normalized(record))
    return output, record


def _control_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    gate_path = run_dir / "control_gate.json"
    committed_gate = _registered_stage_gate(run_dir, gate_path)
    if committed_gate is not None:
        return committed_gate
    configure_exact_torch_backend()
    if not _passed(_load_json(run_dir / "cache_gate.json")):
        raise ArtifactCompatibilityError("control stage requires a passing cache gate")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    profile = JacobiRBCudaProfile()
    controller = _controller.load_frozen_controller(args.parent_coarse_residual_run_dir, device=device)
    numerator_columns: list[np.ndarray] = []
    forward_columns: list[np.ndarray] = []
    feature_names: list[str] = []
    health_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    structural_pass = True
    for anchor in CONTROL_ANCHORS:
        for phase in range(7):
            audit = _load_control_audit(run_dir, anchor=anchor, phase=phase)
            earlier = np.ascontiguousarray(audit["earlier_state"], dtype=np.float64)
            later = np.ascontiguousarray(audit["later_state"], dtype=np.float64)
            forward = _controller.paired_observables(earlier, later, phase=phase)
            outputs: dict[int, np.ndarray] = {}
            records: dict[int, dict[str, Any]] = {}
            for microsteps in REFINEMENT_M:
                outputs[microsteps], records[microsteps] = _run_control_trajectory(
                    run_dir,
                    stem=f"one-phase-anchor-{anchor:04d}-phase-{phase}",
                    later_state=later,
                    sequence=((anchor, phase),),
                    microsteps=microsteps,
                    controller=controller,
                    device=device,
                    profile=profile,
                    stream_role=f"one-phase-anchor-{anchor}-phase-{phase}-M{microsteps}",
                )
                all_records.append(records[microsteps])
            reverse8 = _controller.paired_observables(earlier, outputs[8], phase=phase)
            refine = _controller.paired_observables(outputs[4], outputs[8], phase=phase)
            for microsteps in REFINEMENT_M:
                descriptive = _controller.paired_observables(
                    earlier, outputs[microsteps], phase=phase
                )
                for observable, name in enumerate(descriptive.names):
                    raw_rows.append({
                        "scope": "one_phase",
                        "anchor": anchor,
                        "phase": phase,
                        "microsteps": microsteps,
                        "observable": name,
                        "mean_reverse_minus_earlier": float(descriptive.difference[:, observable].mean()),
                        "authorizing": int(microsteps == 8),
                    })
            structural = np.asarray(forward.structural_invariant, dtype=bool)
            structural_pass &= bool(np.all(forward.difference[:, structural] == 0.0))
            structural_pass &= bool(np.all(reverse8.difference[:, structural] == 0.0))
            for observable, name in enumerate(forward.names):
                if structural[observable]:
                    continue
                base = f"one_phase.anchor{anchor}.phase{phase}.{name}"
                numerator_columns.extend((reverse8.difference[:, observable], refine.difference[:, observable]))
                forward_columns.extend((forward.difference[:, observable], forward.difference[:, observable]))
                feature_names.extend((base + ".bias", base + ".M8_vs_M4"))
            for microsteps in REFINEMENT_M:
                record = records[microsteps]
                health_rows.append({
                    "scope": "one_phase",
                    "anchor": anchor,
                    "phase": phase,
                    "microsteps": microsteps,
                    "maximum_pair_mass_error": record["maximum_pair_mass_error"],
                    "maximum_simplex_mass_error": record["maximum_simplex_mass_error"],
                    "states_finite": record["states_finite"],
                    "states_nonnegative": record["states_nonnegative"],
                    "boundary_rejection_count": record["boundary_rejection_count"],
                })
        audit = _load_control_audit(run_dir, anchor=anchor, phase=None)
        earlier = np.ascontiguousarray(audit["earlier_state"], dtype=np.float64)
        later = np.ascontiguousarray(audit["later_state"], dtype=np.float64)
        sequence = tuple([(anchor, phase) for phase in range(6, -1, -1)] + [(anchor - 1, 6)])
        forward = _controller.paired_observables(
            earlier, later, phase=6, structural_phase_invariants=False
        )
        outputs = {}
        records = {}
        for microsteps in REFINEMENT_M:
            outputs[microsteps], records[microsteps] = _run_control_trajectory(
                run_dir,
                stem=f"eight-phase-anchor-{anchor:04d}",
                later_state=later,
                sequence=sequence,
                microsteps=microsteps,
                controller=controller,
                device=device,
                profile=profile,
                stream_role=f"eight-phase-anchor-{anchor}-M{microsteps}",
            )
            all_records.append(records[microsteps])
        reverse8 = _controller.paired_observables(
            earlier, outputs[8], phase=6, structural_phase_invariants=False
        )
        refine = _controller.paired_observables(
            outputs[4], outputs[8], phase=6, structural_phase_invariants=False
        )
        for microsteps in REFINEMENT_M:
            descriptive = _controller.paired_observables(
                earlier,
                outputs[microsteps],
                phase=6,
                structural_phase_invariants=False,
            )
            for observable, name in enumerate(descriptive.names):
                raw_rows.append({
                    "scope": "eight_phase",
                    "anchor": anchor,
                    "phase": "eight_occurrences",
                    "microsteps": microsteps,
                    "observable": name,
                    "mean_reverse_minus_earlier": float(descriptive.difference[:, observable].mean()),
                    "authorizing": int(microsteps == 8),
                })
        for observable, name in enumerate(forward.names):
            base = f"eight_phase.anchor{anchor}.{name}"
            numerator_columns.extend((reverse8.difference[:, observable], refine.difference[:, observable]))
            forward_columns.extend((forward.difference[:, observable], forward.difference[:, observable]))
            feature_names.extend((base + ".bias", base + ".M8_vs_M4"))
        for microsteps in REFINEMENT_M:
            record = records[microsteps]
            health_rows.append({
                "scope": "eight_phase",
                "anchor": anchor,
                "phase": "eight_occurrences",
                "microsteps": microsteps,
                "maximum_pair_mass_error": record["maximum_pair_mass_error"],
                "maximum_simplex_mass_error": record["maximum_simplex_mass_error"],
                "states_finite": record["states_finite"],
                "states_nonnegative": record["states_nonnegative"],
                "boundary_rejection_count": record["boundary_rejection_count"],
            })
    numerators = np.stack(numerator_columns, axis=1)
    forward_changes = np.stack(forward_columns, axis=1)
    if numerators.shape != (64, 784) or len(feature_names) != 784:
        raise AssertionError("trajectory family is not the frozen 64x784 matrix")
    max_t = _normalized_trajectory_max_t(
        numerators,
        forward_changes,
        confidence=SIMULTANEOUS_CONFIDENCE,
        replicates=BOOTSTRAP_REPLICATES,
        seed=TRAJECTORY_BOOTSTRAP_SEED,
        names=feature_names,
    )
    upper = max_t["simultaneous_upper_absolute"]
    bias_names = [name for name in feature_names if name.endswith(".bias")]
    refinement_names = [name for name in feature_names if name.endswith(".M8_vs_M4")]
    total_transitions = sum(int(record["reference_diagnostics"]["transition_count"]) for record in all_records)
    certified = sum(int(record["reference_diagnostics"]["certified_count"]) for record in all_records)
    fallback = sum(int(record["reference_diagnostics"]["fallback_count"]) for record in all_records)
    fallback_seconds = sum(float(record["reference_diagnostics"]["fallback_seconds"]) for record in all_records)
    backend_seconds = sum(float(record["reference_diagnostics"]["elapsed_seconds"]) for record in all_records)
    wall_seconds = sum(float(record["wall_elapsed_seconds"]) for record in all_records)
    forbidden = {
        name: sum(int(record["reference_diagnostics"]["forbidden_counts"][name]) for record in all_records)
        for name in FORBIDDEN_DIAGNOSTICS
    }
    controller_forbidden_names = (
        "clip_count",
        "floor_count",
        "limiter_count",
        "projection_count",
        "renormalization_count",
    )
    controller_forbidden = {
        name: sum(int(record.get(name, 0)) for record in all_records)
        for name in controller_forbidden_names
    }
    _write_csv(run_dir / "one_phase_health_metrics.csv", [row for row in health_rows if row["scope"] == "one_phase"])
    _write_csv(run_dir / "eight_phase_health_metrics.csv", [row for row in health_rows if row["scope"] == "eight_phase"])
    _write_csv(run_dir / "trajectory_raw_M2_M4_M8_metrics.csv", raw_rows)
    _freeze_json(run_dir / "trajectory_max_t.json", max_t)
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    total_memory = int(torch.cuda.get_device_properties(device).total_memory) if device.type == "cuda" else 1
    persisted_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    checks = {
        "trajectory_family_size": len(feature_names) == 784,
        "one_phase_reverse_law": all(float(upper[name]) <= 0.10 for name in bias_names if name.startswith("one_phase")),
        "eight_phase_reverse_law": all(float(upper[name]) <= 0.10 for name in bias_names if name.startswith("eight_phase")),
        "M8_refinement": all(float(upper[name]) <= 0.05 for name in refinement_names),
        "structural_invariants": structural_pass,
        "states_finite": all(int(row["states_finite"]) == 1 for row in health_rows),
        "states_nonnegative": all(int(row["states_nonnegative"]) == 1 for row in health_rows),
        "pair_mass": all(float(row["maximum_pair_mass_error"]) <= RESOURCE_THRESHOLDS["maximum_mass_error"] for row in health_rows),
        "simplex_mass": all(float(row["maximum_simplex_mass_error"]) <= RESOURCE_THRESHOLDS["maximum_mass_error"] for row in health_rows),
        "boundary_rejections": all(int(row["boundary_rejection_count"]) == 0 for row in health_rows),
        "reference_launch_cap": all(
            int(record["reference_diagnostics"].get("maximum_transition_count_per_call", 1 << 30))
            <= 4096
            for record in all_records
        ),
        "certificate_fraction": certified == total_transitions and total_transitions == ONE_PHASE_CONTROL_TRANSITIONS + EIGHT_PHASE_CONTROL_TRANSITIONS,
        "fallback_fraction": fallback / max(total_transitions, 1) <= RESOURCE_THRESHOLDS["maximum_fallback_fraction"],
        "fallback_time": fallback_seconds / max(backend_seconds, np.finfo(float).tiny) <= RESOURCE_THRESHOLDS["maximum_fallback_time_fraction"],
        "forbidden_counts": all(value == 0 for value in forbidden.values()),
        "controller_forbidden_counts": all(
            value == 0 for value in controller_forbidden.values()
        ),
        "throughput": total_transitions / max(wall_seconds, np.finfo(float).tiny) >= RESOURCE_THRESHOLDS["minimum_transitions_per_second"],
        "peak_memory": peak_memory / total_memory <= RESOURCE_THRESHOLDS["maximum_peak_memory_fraction"],
        "persisted_size": persisted_bytes <= RESOURCE_THRESHOLDS["maximum_persisted_bytes"],
        "maximum_phase_count": max(int(record["phase_count"]) for record in all_records) == 8,
        "no_full_reverse_path": all(int(record["full_reverse_path"]) == 0 for record in all_records),
        "no_image_artifacts": not any(path.suffix.lower() in {".png", ".jpg", ".jpeg"} for path in run_dir.rglob("*")),
    }
    gate = _gate_record(
        "control",
        checks,
        numerically_valid=all(
            checks[name]
            for name in (
                "trajectory_family_size", "structural_invariants",
                "states_finite", "states_nonnegative", "pair_mass",
                "simplex_mass", "boundary_rejections", "reference_launch_cap",
                "certificate_fraction", "forbidden_counts",
                "controller_forbidden_counts", "maximum_phase_count",
                "no_full_reverse_path", "no_image_artifacts",
            )
        ),
        resource_valid=all(
            checks[name]
            for name in (
                "fallback_fraction", "fallback_time", "throughput",
                "peak_memory", "persisted_size",
            )
        ),
        trajectory_max_t=max_t,
        total_control_transition_count=total_transitions,
        certificate_fraction=certified / max(total_transitions, 1),
        fallback_fraction=fallback / max(total_transitions, 1),
        fallback_time_fraction=fallback_seconds / max(backend_seconds, np.finfo(float).tiny),
        transitions_per_second=total_transitions / max(wall_seconds, np.finfo(float).tiny),
        peak_device_memory_bytes=peak_memory,
        peak_device_memory_fraction=peak_memory / total_memory,
        total_persisted_bytes=persisted_bytes,
        forbidden_counts=forbidden,
        controller_forbidden_counts=controller_forbidden,
        maximum_control_trajectory_phase_count=8,
        controller_control_trajectory_performed=1,
        one_phase_reverse_law_controlled=int(checks["one_phase_reverse_law"]),
        eight_phase_reverse_law_controlled=int(checks["eight_phase_reverse_law"]),
        M8_refinement_controlled=int(checks["M8_refinement"]),
    )
    _freeze_json(gate_path, gate)
    return gate


def _decision_from_gates(
    preflight: Mapping[str, Any] | None,
    oracle: Mapping[str, Any] | None,
    cache: Mapping[str, Any] | None,
    control: Mapping[str, Any] | None,
) -> tuple[str, str]:
    if not _passed(preflight):
        checks = preflight.get("checks", {}) if isinstance(preflight, Mapping) else {}
        formula_names = (
            "parent_provenance", "package_manifest", "path_namespace",
            "endpoint_equivalence", "formula_controls", "negative_controls",
            "model_input_firewall",
        )
        if any(not bool(checks.get(name, 0)) for name in formula_names):
            return "controller_formula_or_orientation_failed", "repair controller formula, orientation, namespace, or provenance"
        if not bool(preflight.get("numerically_valid", 0)):
            return "controller_boundary_or_conservation_failed", "repair controller boundary/conservation implementation"
        if not bool(preflight.get("resource_valid", 0)):
            return "reverse_controller_control_resource_infeasible", "repair controller control resources without changing the law or target"
        return "controller_formula_or_orientation_failed", "repair controller formula/orientation against exact controls"
    if not _passed(oracle):
        if not bool(oracle.get("numerically_valid", 0)):
            return "controller_boundary_or_conservation_failed", "repair exact oracle numerical health"
        if not bool(oracle.get("resource_valid", 0)):
            return "reverse_controller_control_resource_infeasible", "repair exact oracle execution resources"
        return "controller_formula_or_orientation_failed", "repair controller formula/orientation against exact analytic controls"
    if not _passed(cache):
        checks = cache.get("checks", {}) if isinstance(cache, Mapping) else {}
        if not bool(cache.get("numerically_valid", 0)):
            return "controller_boundary_or_conservation_failed", "repair exact cache numerical health"
        if not bool(cache.get("resource_valid", 0)):
            return "reverse_controller_control_resource_infeasible", "repair controls-only execution resources"
        if not bool(checks.get("local_all_simultaneous_lower_positive", 0)):
            return "frozen_predictor_not_time_local_reverse_controller", "train a separately selected time-local residual or predeclare a fresh truncated window"
        return "controller_boundary_or_conservation_failed", "repair exact cache gate implementation"
    if not _passed(control):
        checks = control.get("checks", {}) if isinstance(control, Mapping) else {}
        if not bool(control.get("numerically_valid", 0)):
            return "controller_boundary_or_conservation_failed", "repair controller boundary/conservation implementation"
        if not bool(control.get("resource_valid", 0)):
            return "reverse_controller_control_resource_infeasible", "repair controls-only execution resources"
        if not bool(checks.get("M8_refinement", 0)):
            return "reverse_controller_microstep_refinement_failed", "repair or further refine the exact-reference controller discretization"
        if not bool(checks.get("one_phase_reverse_law", 0)) or not bool(checks.get("eight_phase_reverse_law", 0)):
            return "reverse_controller_weak_law_failed", "improve time-local controller prediction on fresh evidence"
        return "reverse_controller_weak_law_failed", "repair reverse-controller control adjudication"
    return "exact_rb_time_local_reverse_controller_controlled", "plan a separate one-image conditional reconstruction/cycle control"


def _validate_decision_gate_evidence(
    name: str,
    gate: Mapping[str, Any],
    *,
    allow_not_evaluated: bool,
) -> str:
    """Validate gate completeness before it may influence a closed decision."""

    expected_schema = RUN_SCHEMA + f"-{name}-gate"
    if gate.get("schema") != expected_schema or gate.get("gate") != name:
        raise ArtifactCompatibilityError(f"{name} gate identity is incompatible")
    status = str(gate.get("evaluation_status", ""))
    if status == "not_evaluated":
        if not allow_not_evaluated or int(gate.get("passed", -1)) != 0:
            raise ArtifactCompatibilityError(
                f"{name} gate is unexpectedly not evaluated"
            )
        return status
    if status != "evaluated":
        raise ArtifactCompatibilityError(
            f"decide stage cannot use {status or 'missing-status'} {name} evidence"
        )
    checks = gate.get("checks")
    required = _DECISION_GATE_CHECKS[name]
    if not isinstance(checks, Mapping) or not required.issubset(set(checks)):
        missing = sorted(required - set(checks) if isinstance(checks, Mapping) else required)
        raise ArtifactCompatibilityError(
            f"{name} gate lacks required decision fields: {missing}"
        )
    if "numerically_valid" not in gate or "resource_valid" not in gate:
        raise ArtifactCompatibilityError(
            f"{name} gate lacks numerical/resource classification"
        )
    expected_pass = int(all(bool(checks[key]) for key in checks))
    if int(gate.get("passed", -1)) != expected_pass:
        raise ArtifactCompatibilityError(f"{name} gate pass flag is inconsistent")
    return status


def _write_not_evaluated_after_failure(run_dir: Path, failed_stage: str) -> None:
    order = ("preflight", "oracle", "cache", "control")
    if failed_stage not in order:
        return
    failed_index = order.index(failed_stage)
    for name in order[failed_index + 1 :]:
        path = run_dir / f"{name}_gate.json"
        _freeze_json(
            path,
            _not_evaluated(name, f"skipped_after_failed_{failed_stage}_gate"),
        )


def _decide_stage(run_dir: Path, _args: argparse.Namespace) -> dict[str, Any]:
    path = run_dir / "controller_decision.json"
    gates = {
        name: _load_json(run_dir / f"{name}_gate.json")
        if (run_dir / f"{name}_gate.json").is_file()
        else None
        for name in ("preflight", "oracle", "cache", "control")
    }
    prior_failure = False
    for name in ("preflight", "oracle", "cache", "control"):
        gate = gates[name]
        if gate is None:
            raise ArtifactCompatibilityError(
                f"decide stage requires a committed upstream {name} gate artifact"
            )
        status = _validate_decision_gate_evidence(
            name, gate, allow_not_evaluated=prior_failure
        )
        if status == "not_evaluated":
            continue
        if not _passed(gate):
            prior_failure = True
    decision, next_action = _decision_from_gates(
        gates["preflight"], gates["oracle"], gates["cache"], gates["control"]
    )
    passed = decision == "exact_rb_time_local_reverse_controller_controlled"
    control_performed = int(
        gates["control"].get("evaluation_status") == "evaluated"
        and int(gates["control"].get("controller_control_trajectory_performed", 0))
        == 1
    )
    control_phase_count = (
        int(gates["control"].get("maximum_control_trajectory_phase_count", 0))
        if control_performed
        else 0
    )
    terminal_claim = _controller.claim_boundary(controlled=passed)
    terminal_claim["controller_control_trajectory_performed"] = control_performed
    terminal_claim["maximum_control_trajectory_phase_count"] = control_phase_count
    derived_record = {
        "schema": RUN_SCHEMA + "-decision",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "decision": decision,
        "recommended_next_action": next_action,
        "controller_formula_controlled": int(_passed(gates["preflight"]) and _passed(gates["oracle"])),
        "internal_time_target_controlled": int(_passed(gates["cache"])),
        "terminal_near_reverse_start_controlled": int(_passed(gates["cache"]) and bool(gates["cache"].get("terminal_near_reverse_start_controlled", 0))),
        "one_phase_reverse_law_controlled": int(_passed(gates["control"]) and bool(gates["control"].get("one_phase_reverse_law_controlled", 0))),
        "eight_phase_reverse_law_controlled": int(_passed(gates["control"]) and bool(gates["control"].get("eight_phase_reverse_law_controlled", 0))),
        "M8_refinement_controlled": int(_passed(gates["control"]) and bool(gates["control"].get("M8_refinement_controlled", 0))),
        "one_image_reconstruction_planning_authorized": int(passed),
        "controller_control_trajectory_performed": control_performed,
        "maximum_control_trajectory_phase_count": control_phase_count,
        "claim_scope": "time-local and at-most-eight-phase reverse-controller controls for one frozen image under the exact certified K512 split chain",
        **CLAIM_BOUNDARY,
    }
    _controller.validate_claim_boundary(derived_record)
    _assert_unambiguous_metric_schema(derived_record)
    if path.is_file() and _load_json(path) != _normalized(derived_record):
        raise ArtifactCompatibilityError("orphan controller decision changed")
    record = _freeze_json(path, derived_record)
    _freeze_json(
        run_dir / "claim_boundary.json",
        {
            "schema": RUN_SCHEMA + "-claim-boundary",
            "schema_version": 1,
            **terminal_claim,
        },
    )
    # Keep a gate artifact for uniform --require-gate handling.
    decide_gate = _gate_record(
        "decide",
        {
            "preflight": _passed(gates["preflight"]),
            "oracle": _passed(gates["oracle"]),
            "cache": _passed(gates["cache"]),
            "control": _passed(gates["control"]),
            "terminal_decision": passed,
        },
        decision=decision,
        one_image_reconstruction_planning_authorized=int(passed),
    )
    _freeze_json(run_dir / "decide_gate.json", decide_gate)
    return record


def _gate_for_stage(run_dir: Path, stage: str) -> dict[str, Any] | None:
    filename = "decide_gate.json" if stage == "decide" else f"{stage}_gate.json"
    return _load_json(run_dir / filename) if (run_dir / filename).is_file() else None


def _stage_sequence(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return ("preflight", "oracle", "cache", "control", "decide")
    if stage == "report":
        return ()
    return (stage,)


def _execution_failure_decision(exc: BaseException, stage: str) -> str:
    domain = str(getattr(exc, "failure_domain", "workflow_execution"))
    if domain == "provenance" or isinstance(exc, ArtifactCompatibilityError):
        return "controller_formula_or_orientation_failed"
    if isinstance(exc, _controller.ControllerBoundaryStepRejected):
        return "controller_boundary_or_conservation_failed"
    if domain == "resource":
        return "reverse_controller_control_resource_infeasible"
    return "reverse_controller_execution_invalid"


def _commit_execution_failure(
    run_dir: Path,
    *,
    stage: str,
    exc: BaseException,
) -> None:
    failure_path = run_dir / f"{stage}_execution_failure.json"
    atomic_write_json(
            failure_path,
            {
                "schema": RUN_SCHEMA + f"-{stage}-execution-failure",
                "schema_version": 1,
                "evaluation_status": "execution_failed",
                "stage_execution_valid": 0,
                "scientific_evidence_complete": 0,
                "passed": 0,
                "failure_domain": str(getattr(exc, "failure_domain", "workflow_execution")),
                "failure_code": str(getattr(exc, "failure_code", type(exc).__name__)),
                "message": str(exc),
                **CLAIM_BOUNDARY,
            },
        )
    decision = _execution_failure_decision(exc, stage)
    # Preserve the last successfully sealed registry.  Current-stage shards
    # and this failure diagnostic remain unregistered and therefore retryable.
    _status(
        run_dir,
        state="execution_failed",
        stage=stage,
        decision=decision,
        message=str(exc),
        failure_domain=str(getattr(exc, "failure_domain", "workflow_execution")),
        failure_code=str(getattr(exc, "failure_code", type(exc).__name__)),
    )


def _commit_control_boundary_rejection(
    run_dir: Path,
    *,
    exc: BaseException,
) -> dict[str, Any]:
    completed_by_stem: dict[str, int] = {}
    for path in (run_dir / "control" / "phase_checkpoints").glob(
        "*-occurrence-*.json"
    ):
        match = re.match(r"(.+)-occurrence-(\d+)\.json$", path.name)
        if match and _load_json(path).get("committed") == 1:
            completed_by_stem[match.group(1)] = max(
                completed_by_stem.get(match.group(1), 0), int(match.group(2)) + 1
            )
    attempted_phase_count = min(
        8, max((value + 1 for value in completed_by_stem.values()), default=1)
    )
    checks = {name: False for name in _DECISION_GATE_CHECKS["control"]}
    checks["no_full_reverse_path"] = True
    checks["no_image_artifacts"] = True
    gate = _gate_record(
        "control",
        checks,
        numerically_valid=0,
        resource_valid=0,
        boundary_rejection={
            "failure_code": str(
                getattr(exc, "failure_code", "controller_boundary_step_rejected")
            ),
            "message": str(exc),
        },
        controller_control_trajectory_performed=1,
        maximum_control_trajectory_phase_count=attempted_phase_count,
        one_phase_reverse_law_controlled=0,
        eight_phase_reverse_law_controlled=0,
        M8_refinement_controlled=0,
    )
    _freeze_json(run_dir / "control_gate.json", gate)
    return gate


def _commit_preflight_boundary_rejection(
    run_dir: Path,
    *,
    exc: BaseException,
) -> dict[str, Any]:
    checks = {name: False for name in _DECISION_GATE_CHECKS["preflight"]}
    for name in (
        "parent_provenance",
        "package_manifest",
        "path_namespace",
        "endpoint_equivalence",
        "formula_controls",
        "negative_controls",
        "model_input_firewall",
        "no_physical_panel_opened",
    ):
        checks[name] = True
    gate = _gate_record(
        "preflight",
        checks,
        numerically_valid=0,
        resource_valid=0,
        boundary_rejection={
            "failure_code": str(
                getattr(exc, "failure_code", "controller_boundary_step_rejected")
            ),
            "message": str(exc),
        },
        physical_authorizing_statistics_opened=0,
    )
    _freeze_json(run_dir / "preflight_gate.json", gate)
    return gate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact Jacobi/RB time-local reverse-controller controls"
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/experiment12_d0_jacobi_rb_reverse_controller_control"),
    )
    parser.add_argument(
        "--run-name",
        default="production-exact-rb-reverse-controller-control",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--require-gate", choices=REQUIRED_GATES, default="none")
    parser.add_argument(
        "--parent-coarse-residual-run-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--resume-run-dir", type=Path)
    args = parser.parse_args(argv)
    args.runs_root = args.runs_root.resolve()
    args.parent_coarse_residual_run_dir = args.parent_coarse_residual_run_dir.resolve()
    if args.resume_run_dir is not None:
        args.resume_run_dir = args.resume_run_dir.resolve()
    return args


def _run(args: argparse.Namespace) -> int:
    run_dir: Path | None = None
    initialized = False
    fresh_run = False
    active_stage = str(args.stage)
    try:
        run_dir, resumed = _make_run_dir(args)
        fresh_run = not resumed
        _initialize_run(run_dir, args, resumed=resumed)
        initialized = True
        if not (run_dir / "artifact_registry.json").is_file():
            _artifact_registry(run_dir)
        print(f"Jacobi/RB reverse-controller run directory: {run_dir}", flush=True)
        _status(run_dir, state="running", stage=active_stage)
        scientific_failure = False
        for stage in _stage_sequence(args.stage):
            active_stage = stage
            _status(run_dir, state="running", stage=stage)
            if stage == "preflight":
                gate = _preflight_stage(run_dir, args)
            elif stage == "oracle":
                gate = _oracle_stage(run_dir, args)
            elif stage == "cache":
                gate = _cache_stage(run_dir, args)
            elif stage == "control":
                gate = _control_stage(run_dir, args)
            elif stage == "decide":
                _decide_stage(run_dir, args)
                gate = _gate_for_stage(run_dir, "decide")
            else:  # pragma: no cover - argparse prevents this.
                raise AssertionError(stage)
            _artifact_registry(run_dir)
            if not _passed(gate):
                # Expensive downstream stages never run on incomplete evidence.
                scientific_failure = True
                if stage != "decide":
                    _write_not_evaluated_after_failure(run_dir, stage)
                    _decide_stage(run_dir, args)
                break
        if args.stage == "all" and not scientific_failure and not (run_dir / "controller_decision.json").is_file():
            _decide_stage(run_dir, args)
        decision = (
            _load_json(run_dir / "controller_decision.json")
            if (run_dir / "controller_decision.json").is_file()
            else None
        )
        _artifact_registry(run_dir)
        _verify_terminal_registry_exact(run_dir)
        requested = None if args.require_gate == "none" else _gate_for_stage(run_dir, args.require_gate)
        required_pass = args.require_gate == "none" or _passed(requested)
        stage_gate = _gate_for_stage(run_dir, args.stage) if args.stage in REQUIRED_GATES else None
        stage_pass = _passed(stage_gate) if stage_gate is not None else True
        status_decision = (
            str(decision.get("decision"))
            if decision is not None
            else (f"ready_after_{args.stage}" if stage_pass else None)
        )
        _status(
            run_dir,
            state="complete" if required_pass and not scientific_failure else "gate_failed",
            stage=args.stage if args.stage != "report" else "report",
            decision=status_decision,
        )
        return 0 if required_pass else 1
    except Exception as exc:  # artifacts-before-failure is part of the contract.
        if (
            initialized
            and run_dir is not None
            and active_stage == "preflight"
            and isinstance(exc, _controller.ControllerBoundaryStepRejected)
        ):
            _commit_preflight_boundary_rejection(run_dir, exc=exc)
            _write_not_evaluated_after_failure(run_dir, "preflight")
            _decide_stage(run_dir, args)
            _artifact_registry(run_dir)
            _verify_terminal_registry_exact(run_dir)
            _status(
                run_dir,
                state="gate_failed",
                stage="preflight",
                decision="controller_boundary_or_conservation_failed",
                message=str(exc),
                failure_domain="controller_boundary",
                failure_code=str(
                    getattr(exc, "failure_code", "controller_boundary_step_rejected")
                ),
            )
            print(f"Jacobi/RB reverse-controller boundary rejection: {exc}", file=sys.stderr)
            return 0 if args.require_gate == "none" else 1
        if (
            initialized
            and run_dir is not None
            and active_stage == "control"
            and isinstance(exc, _controller.ControllerBoundaryStepRejected)
        ):
            _commit_control_boundary_rejection(run_dir, exc=exc)
            _decide_stage(run_dir, args)
            _artifact_registry(run_dir)
            _verify_terminal_registry_exact(run_dir)
            _status(
                run_dir,
                state="gate_failed",
                stage="control",
                decision="controller_boundary_or_conservation_failed",
                message=str(exc),
                failure_domain="controller_boundary",
                failure_code=str(
                    getattr(exc, "failure_code", "controller_boundary_step_rejected")
                ),
            )
            print(f"Jacobi/RB reverse-controller boundary rejection: {exc}", file=sys.stderr)
            return 0 if args.require_gate == "none" else 1
        if (initialized or fresh_run) and run_dir is not None and run_dir.is_dir():
            _commit_execution_failure(run_dir, stage=active_stage, exc=exc)
        print(f"Jacobi/RB reverse-controller error: {exc}", file=sys.stderr)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
