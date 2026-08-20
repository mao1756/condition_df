"""Exploratory objective-bearing Jacobi/RB frequency-one reverse rollout.

This command intentionally differs from the preceding learnability gates.  It
uses one validation-inspected historical checkpoint to produce paired reverse
states and images on fresh exploratory paths.  The output is diagnostic: it
cannot rescue the parent's simultaneous validation claim and it never opens
the parent's protected confirmation role.

Production scientific choices are frozen below.  ``--test-only`` selects a
small deterministic fake-sampler workflow used to exercise orchestration,
restart, artifact, oracle-failure, and report behavior without CUDA.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_csv,
    atomic_write_json,
    config_fingerprint,
    configure_exact_torch_backend,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_tangent_rollout import ExactForwardShardAggregateError


RUN_SCHEMA = "experiment12-d0-jacobi-rb-frequency1-exploratory-rollout-v1"
TEST_RUN_SCHEMA = RUN_SCHEMA + "-test-only"
FUSED_RUN_SCHEMA = (
    "experiment12-d0-jacobi-rb-frequency1-exploratory-rollout-fused-laptop-v1"
)
FUSED_TEST_RUN_SCHEMA = FUSED_RUN_SCHEMA + "-test-only"
RECOVERY_RUN_SCHEMA = (
    "experiment12-d0-jacobi-rb-frequency1-objective-first-recovery-v3"
)
RECOVERY_TEST_RUN_SCHEMA = RECOVERY_RUN_SCHEMA + "-test-only"
STAGES = (
    "initialize",
    "preflight",
    "forward",
    "development",
    "evaluation",
    "replication",
    "report",
    "all",
)
REQUIRED_GATES = (
    "none",
    "initialize",
    "preflight",
    "forward",
    "development",
    "evaluation",
    "replication",
)

# Frozen exploratory science configuration.
CHECKPOINT_SEED = 261_372
CHECKPOINT_UPDATE = 3_700
CHECKPOINT_FILE_SHA256 = (
    "49c8211020eb0169c020d4de3c11c0f123319e359a24c2f6444d8b35cdc23326"
)
CHECKPOINT_STATE_SHA256 = (
    "4f63dd991e028418e1a514d568615a2aae6606fd94910d6ffc21086dba486fe3"
)
PARENT_BASENAME = "20260811-010641_production-frequency1-coordinate-v1-one-image"
PARENT_REGISTRY_COUNT = 2_701
PARENT_REGISTRY_SEMANTIC_SHA256 = (
    "bd814a776b43e032acdaec8a8337a99c91db9f44769ba4bc2cea4e26bcb4c2d7"
)
PARENT_REGISTRY_FILE_SHA256 = (
    "77037e7ed485b3178dbb1dfc0cd8bc2cfafb761ae893e677d7772cf7c4348542"
)
PARENT_SOURCE_FINGERPRINT = (
    "7efb3bd306cd942c6721686501fe57ed1f342caa7891b8d4933b11dcff837be4"
)
PARENT_CONFIG_SHA256 = (
    "eac7021d4c71c5c7dae9ff187952749c4fa5ed236c77e0001ea78887537fc827"
)
PARENT_DECISION = "no_frequency1_coordinate_validation_candidate"
SOURCE_IMAGE_SHA256 = (
    "0bb39fec59853f789fe366251cd85ed79ffbffb5a1aaa32084d2dbd2bbb4ea7d"
)
MIXED_TARGET_SHA256 = (
    "00ae86fb69be6d86557f15f6f8fa00f8bb3c2514f331863c9638e36d23d135c5"
)
LAMBDA_MIX = 0.35
SHORT_ANCHOR = 127
FULL_ANCHOR = 511
MICROSTEPS = 2
LEARNED_GAINS = (0.5, 1.0, 2.0, 4.0)
FORWARD_ROOT_SEED = 261_401
REVERSE_ROOT_SEED = 261_402
PATH_IDS = {
    "preflight": 0xFB000,
    "development": 0xFB100,
    "evaluation": 0xFB200,
    "replication": 0xFB300,
}
PATH_ID_ALLOCATION_VERSION = "frequency1-rollout-fb-v2-after-fa-smoke"
FUSED_PATH_ID_ALLOCATION_VERSION = "frequency1-rollout-fused-fc-preflight-v1"
FAILED_FA_PREFLIGHT_RUN = "20260812-005426_production-frequency1-exploratory-rollout"
CONTINUATION_BASENAME = (
    "20260812-005942_production-frequency1-exploratory-rollout-fbv2"
)
CONTINUATION_ARTIFACT_COUNT = 21
CONTINUATION_REGISTRY_SEMANTIC_SHA256 = (
    "b7fed7a23d7ff7cea31ff4e5d87c7ec452432f4d06df3f88a244a4c9bec12338"
)
CONTINUATION_REGISTRY_FILE_SHA256 = (
    "53bb72b0ce1ec1891071d10e7b0198a47d591f9aa22e0ad70543981fb2709ace"
)
CONTINUATION_CHECKSUM_FILE_SHA256 = (
    "4aedd8b79821e992e466da240fcaf92273241ed9daaae878fff3a6859a7121bf"
)
CONTINUATION_BUNDLE_AUDIT_FILE_SHA256 = (
    "45112d7c55799c5129b12c569f43ae78ae250b5688db665079781002509488e4"
)
CONTINUATION_RUN_MANIFEST_FILE_SHA256 = (
    "e098c029aa85da3e3a7e1cf36bb31fc64016aa023cd0ca16ecf825df6ffcdbc3"
)
CONTINUATION_SOURCE_FINGERPRINT = (
    "3453a7decc91e32875c6f57a96eaf1082e8353fd9923633bc5c531c283880291"
)
CONTINUATION_CONFIG_SEMANTIC_SHA256 = (
    "3a247507125cd0fcbed0ab38b1e0738d9edb05fbf8251b6b5e988336e025695a"
)
CONTINUATION_PATH_PLAN_SEMANTIC_SHA256 = (
    "7dcd60c999f9c958f464e34330b5b9f2bb0610980a7b9d5988903362850b94e5"
)
CONTINUATION_PREFLIGHT_GATE_SEMANTIC_SHA256 = (
    "3fce922c5772766d24007c1d5653ffebfc5586654abc1aa2685b9f4050f07e3c"
)
CONTINUATION_RESOURCE_PROJECTION_SEMANTIC_SHA256 = (
    "fd6b52926aaa21c866cb7f3c58996ca6b1bed1dc598f23f0fb267d4465066b05"
)
FUSED_PREFLIGHT_PATH_IDS = {
    "warmup": 0xFC000,
    "forward_profile": 0xFC001,
    "reverse_p3_profile": 0xFC002,
    "reverse_p6_profile": 0xFC003,
}
FUSED_PREFLIGHT_SLOT = (0xFC000, 0xFC010)
FUSED_PROFILE_TRANSITIONS = {
    "forward_p1": 21_952,
    "reverse_p3": 263_424,
    "reverse_p6": 526_848,
}
FUSED_PROFILE_PRODUCTION_SHARDS = {
    "forward_p1": 128,
    "reverse_p3": 48,
    "reverse_p6": 32,
}
FUSED_PROJECTION_FIXED_RESERVE_SECONDS = 300.0
FUSED_PROJECTION_FIXED_STORAGE_RESERVE_BYTES = 64 * 1024**2
MINIMUM_EFFECTIVE_MAIN_RATE = 1_495.9881481481482
FUSED_PREFLIGHT_LEARNED_GAIN = 4.0
RECOVERY_PREDECESSOR_BASENAME = (
    "20260812-065538_production-frequency1-exploratory-rollout-fused-laptop-v1"
)
RECOVERY_PREDECESSOR_ARTIFACT_COUNT = 151
RECOVERY_PREDECESSOR_MANIFEST_SEMANTIC_SHA256 = (
    "6d13ec893857d0d457fce827923f4e1b4fc746c245fef2f22747769102e2e882"
)
RECOVERY_PREDECESSOR_MANIFEST_FILE_SHA256 = (
    "25cd4e1660f2bab390d9ba4771c05acd389b764f2505acd503df891037ded71e"
)
RECOVERY_PREDECESSOR_CHECKSUM_FILE_SHA256 = (
    "6f4dfb7a410d5989af8505b9a3d44ed4ae9ae056325e2c21bdbabd6f8eae0d8f"
)
RECOVERY_PREDECESSOR_CONFIG_SEMANTIC_SHA256 = (
    "aa4c1b88e38ac229f50bdf34029dfdf710c6594da9dd4ff2f50faf5c8723862f"
)
RECOVERY_PREDECESSOR_SOURCE_FINGERPRINT = (
    "2c606a35849374e6041fbf3ce04a73639d9a58cf8a9a29f09d790547a2e660d0"
)
RECOVERY_ANCHOR_BINDINGS = {
    SHORT_ANCHOR: {
        "shard_index": 15,
        "json_sha256": "623927efcc5afeef6ad881b5331827e435f818874954193a03c9a7220aefb5c7",
        "npz_sha256": "9160e413b221176559f8948cff4e626782e98fc93fbedce36d1afec931049be0",
        "output_state_sha256": "d4a51f6cbaaa23455a2357f23b79d57bb63228f774fcef78de6a62f110a6c376",
    },
    FULL_ANCHOR: {
        "shard_index": 63,
        "json_sha256": "d08613b424c3e811beec1ba33b64bc4d44fc21ea2a80c26122f5a153d115f663",
        "npz_sha256": "870124657c94c679b0d54ee4d3e7c764bc12d224d3ee0276da55e67c87daa907",
        "output_state_sha256": "59095bec8f0ec8390842229e5a2b22589c501dca624127da9f589f2fc58fd7c9",
    },
}
RECOVERY_REPORT_RESERVE_SECONDS = 300.0
RECOVERY_PROJECTION_FACTOR = 1.20
RECOVERY_EXACT_AUDIT_OUTER_STEPS = 8
RECOVERY_CORE_LEARNED_GAIN = 1.0
RECOVERY_REFERENCE_CONTRACTS = {
    "exact": "certified_exact",
    "candidate": "candidate_approximate",
}
RECOVERY_EXACT_FORBIDDEN_COUNTS = frozenset(
    {
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
    }
)
RECOVERY_CANDIDATE_FORBIDDEN_COUNTS = frozenset(
    RECOVERY_EXACT_FORBIDDEN_COUNTS - {"approximation_count"}
)
STREAM_ROLES = {
    "development": "frequency1-rollout-development-v1",
    "evaluation": "frequency1-rollout-evaluation-v1",
    "replication": "frequency1-rollout-replication-v1",
}
MAXIMUM_MAIN_WALL_SECONDS = 6.0 * 3600.0
MAXIMUM_REPLICATION_WALL_SECONDS = 2.0 * 3600.0
MAXIMUM_PERSISTED_BYTES = 2 * 1024**3
MAXIMUM_MASS_ERROR = 2.0e-12
MINIMUM_TRANSITIONS_PER_SECOND = 1_300.0
MAXIMUM_FALLBACK_FRACTION = 1.0e-4
MAXIMUM_FALLBACK_TIME_FRACTION = 0.10
MAXIMUM_MEMORY_FRACTION = 0.80
WEAK_CONTROL_RATIO = 0.05
MAIN_WORKFLOW_TRANSITIONS = 32_313_344
NO_WORK = {
    "model_optimization_performed": 0,
    "physical_training_performed": 0,
    "confirmation_evidence_opened": 0,
    "prior_start_sampling_performed": 0,
    "multi_image_training_performed": 0,
}
_FINAL_MANIFEST_EXCLUDED = {
    "artifact_manifest.json",
    "SHA256SUMS.txt",
    "bundle_integrity_audit.json",
}
_LOAD_BEARING_SOURCE_NAMES = (
    "diag_d0_jacobi_rb_frequency1_rollout.py",
    "d0_jacobi_rb_tangent_rollout.py",
    "d0_jacobi_rb_boundary_tangent.py",
    "d0_jacobi_rb_boundary_tangent_fused.py",
    "d0_jacobi_rb_boundary_tangent_frequency1_coordinate.py",
    "d0_jacobi_rb_boundary_tangent_zero_baseline.py",
    "d0_jacobi_rb_reverse_controller.py",
    "d0_jacobi_rb_cuda.py",
    "d0_jacobi_rb_cuda_deferred.py",
    "d0_jacobi_rb_cuda_controls.py",
    "d0_jacobi_rb_cuda_multipath.py",
    "d0_jacobi_rb_learnability.py",
    "d0_jacobi_artifacts.py",
)
_PREPARED_FUSED_BACKENDS: dict[str, Any] = {}
_PREPARED_FUSED_ELAPSED: dict[str, float] = {}
_PREPARED_FUSED_SEED_MAPS: dict[tuple[int, int, str], Mapping[str, Any]] = {}


class RolloutCLIError(RuntimeError):
    """Typed orchestration/integrity failure committed before returning."""

    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "execution_integrity",
        failure_code: str = "frequency1_rollout_execution_invalid",
    ) -> None:
        super().__init__(message)
        self.failure_domain = str(failure_domain)
        self.failure_code = str(failure_code)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fused_mode(args_or_config: Any) -> bool:
    if isinstance(args_or_config, Mapping):
        return bool(
            int(args_or_config.get("fused_continuation", 0))
            or args_or_config.get("continuation_run_dir")
            or str(args_or_config.get("schema", "")).startswith(FUSED_RUN_SCHEMA)
        )
    return getattr(args_or_config, "continuation_run_dir", None) is not None


def _recovery_mode(args_or_config: Any) -> bool:
    """Return whether the additive objective-first successor is selected."""

    if isinstance(args_or_config, Mapping):
        return bool(
            int(args_or_config.get("objective_first_recovery", 0))
            or args_or_config.get("predecessor_run_dir")
            or str(args_or_config.get("schema", "")).startswith(RECOVERY_RUN_SCHEMA)
        )
    return getattr(args_or_config, "predecessor_run_dir", None) is not None


def _schema(args_or_config: Any) -> str:
    value = (
        args_or_config.get("test_only", 0)
        if isinstance(args_or_config, Mapping)
        else getattr(args_or_config, "test_only", False)
    )
    if _recovery_mode(args_or_config):
        return RECOVERY_TEST_RUN_SCHEMA if int(value) else RECOVERY_RUN_SCHEMA
    if _fused_mode(args_or_config):
        return FUSED_TEST_RUN_SCHEMA if int(value) else FUSED_RUN_SCHEMA
    return TEST_RUN_SCHEMA if int(value) else RUN_SCHEMA


def _semantic(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("semantic_sha256", None)
    return {**body, "semantic_sha256": config_fingerprint(body)}


def _is_semantic_record(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    body = dict(value)
    recorded = body.pop("semantic_sha256", None)
    return isinstance(recorded, str) and config_fingerprint(body) == recorded


def _load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"cannot read JSON artifact {target}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"JSON artifact is not an object: {target}")
    return dict(value)


def _load_semantic(path: str | Path, name: str | None = None) -> dict[str, Any]:
    record = _load_json(path)
    body = dict(record)
    measured = body.pop("semantic_sha256", None)
    if not isinstance(measured, str) or config_fingerprint(body) != measured:
        raise ArtifactCompatibilityError(
            f"{name or Path(path).name} semantic commitment changed"
        )
    return record


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            return {
                name: np.array(archive[name], copy=True, order="C")
                for name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise ArtifactCompatibilityError(f"cannot read NPZ artifact {path}: {exc}") from exc


def _atomic_npz(path: str | Path, **arrays: np.ndarray) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                **{
                    name: np.ascontiguousarray(value)
                    for name, value in arrays.items()
                },
            )
            handle.flush()
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": target.as_posix(),
        "size": int(target.stat().st_size),
        "sha256": file_fingerprint(target),
    }


def _atomic_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _record(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_record"):
        result = value.to_record()
    elif is_dataclass(value):
        result = asdict(value)
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise TypeError(f"cannot serialize record of type {type(value).__name__}")
    return dict(result)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _state_row(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu").numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (1, 784):
        array = array[0]
    if array.shape != (784,):
        raise RolloutCLIError(
            f"state must have shape [784] or [1,784], got {array.shape}",
            failure_code="rollout_state_shape_invalid",
        )
    return np.array(array, dtype=np.float64, copy=True, order="C")


def _status(
    run_dir: Path,
    *,
    state: str,
    stage: str,
    decision: str | None = None,
    message: str | None = None,
    failure_domain: str | None = None,
    failure_code: str | None = None,
    scientific_evidence_complete: int = 0,
) -> dict[str, Any]:
    config = _load_json(run_dir / "scientific_config.json") if (
        run_dir / "scientific_config.json"
    ).is_file() else {}
    record = {
        "schema": _schema(config) + "-status",
        "schema_version": 1,
        "updated_at": _now(),
        "state": str(state),
        "stage": str(stage),
        "decision": decision,
        "message": message,
        "failure_domain": failure_domain,
        "failure_code": failure_code,
        "scientific_evidence_complete": int(scientific_evidence_complete),
        "research_mode": "exploratory",
        "objective_bearing_experiment": 1,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "run_status.json", record)
    return record


def _artifact_manifest(run_dir: Path) -> dict[str, Any]:
    config = _load_json(run_dir / "scientific_config.json") if (
        run_dir / "scientific_config.json"
    ).is_file() else {}
    run_schema = _schema(config)
    records: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir).as_posix()
        if relative in _FINAL_MANIFEST_EXCLUDED or relative.endswith(".tmp"):
            continue
        records.append(
            {
                "path": relative,
                "size": int(path.stat().st_size),
                "sha256": file_fingerprint(path),
            }
        )
    record = _semantic(
        {
            "schema": run_schema + "-artifact-manifest",
            "schema_version": 1,
            "artifact_count": len(records),
            "artifacts": records,
        }
    )
    atomic_write_json(run_dir / "artifact_manifest.json", record)
    checksum_rows = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS.txt" or path.suffix == ".tmp":
            continue
        checksum_rows.append(
            f"{file_fingerprint(path)}  {path.relative_to(run_dir).as_posix()}"
        )
    _atomic_text(run_dir / "SHA256SUMS.txt", "\n".join(checksum_rows) + "\n")
    return record


def _finalize_artifacts(run_dir: Path) -> dict[str, Any]:
    """Produce a non-circular final manifest plus an excluded audit record."""

    manifest = _artifact_manifest(run_dir)
    representatives: list[dict[str, Any]] = []
    candidates = [
        *sorted(run_dir.glob("*/*/*/selected_states.npz")),
        *sorted(run_dir.glob("forward/*/anchors.npz")),
        *sorted(run_dir.glob("failure_artifacts/*/last_valid_states.npz")),
        *sorted(
            run_dir.glob(
                "objective_attempts/*/fused_families/*/*/shard-*.npz"
            )
        ),
    ]
    unique_candidates = list(dict.fromkeys(candidates))
    for path in unique_candidates[:3]:
        arrays = _load_npz(path)
        if not arrays:
            raise ArtifactCompatibilityError("representative NPZ is empty")
        representatives.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "sha256": file_fingerprint(path),
                "array_names": sorted(arrays),
            }
        )
    audit = _semantic(
        {
            "schema": _schema(
                _load_json(run_dir / "scientific_config.json")
                if (run_dir / "scientific_config.json").is_file()
                else {}
            )
            + "-bundle-integrity-audit",
            "schema_version": 1,
            "artifact_manifest_artifact_count": manifest["artifact_count"],
            "artifact_manifest_semantic_sha256": manifest["semantic_sha256"],
            "artifact_manifest_file_sha256": file_fingerprint(
                run_dir / "artifact_manifest.json"
            ),
            "representative_npz_opened_and_hashed": representatives,
            "excluded_from_manifest_to_avoid_a_hash_cycle": 1,
            "passed": 1,
        }
    )
    atomic_write_json(run_dir / "bundle_integrity_audit.json", audit)
    # The manifest is stable because the audit is deliberately excluded; this
    # second pass adds the audit itself to SHA256SUMS.txt.
    final_manifest = _artifact_manifest(run_dir)
    if final_manifest != manifest:
        raise ArtifactCompatibilityError("final artifact manifest was not stable")
    return final_manifest


def _directory_bytes(run_dir: Path) -> int:
    return sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())


def _source_revision(repository_root: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=20.0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=normal")
    return {
        "git_head": head,
        "git_dirty": None if status is None else int(bool(status)),
        "git_status_sha256": None if status is None else hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _source_closure() -> dict[str, Any]:
    from mnist.d0_jacobi_rb_boundary_tangent_v3_provenance import (
        v3_transitive_source_paths,
    )

    repository_root = Path(__file__).resolve().parent.parent
    package = Path(__file__).resolve().parent
    paths = v3_transitive_source_paths(
        tuple(
            path
            for path in (
            Path(__file__).resolve(),
            package / "d0_jacobi_rb_tangent_rollout.py",
            package / "d0_jacobi_rb_boundary_tangent.py",
            package / "d0_jacobi_rb_boundary_tangent_fused.py",
            package / "d0_jacobi_rb_tangent_fused.py",
            package / "d0_jacobi_rb_cuda_deferred.py",
            )
            if path.is_file()
        )
    )
    records = [
        {
            "path": path.resolve().relative_to(repository_root).as_posix(),
            "sha256": file_fingerprint(path),
        }
        for path in paths
    ]
    return {
        "files": records,
        "source_fingerprint": config_fingerprint(records),
    }


def _health_record(
    diagnostics: Mapping[str, Any],
    *,
    expected_transition_count: int,
    test_only: bool,
    enforce_throughput: bool = True,
) -> dict[str, Any]:
    """Normalize and fail closed on exact transition health semantics."""

    if type(enforce_throughput) is not bool:
        raise TypeError("enforce_throughput must be a bool")
    value = dict(diagnostics)
    transition_count = int(
        value.get("reference_transition_count", value.get("transition_count", -1))
    )
    certified_count = int(
        value.get("reference_certified_count", value.get("certified_count", -1))
    )
    active_count = int(
        value.get("reference_active_count", value.get("active_count", transition_count))
    )
    if active_count < 0 or active_count > transition_count:
        active_count = -1
    structural_noop_count = int(
        value.get(
            "reference_structural_noop_count",
            value.get(
                "structural_noop_count",
                transition_count - active_count if active_count >= 0 else -1,
            ),
        )
    )
    authorized_count = int(
        value.get(
            "reference_authorized_count",
            value.get(
                "authorized_count",
                certified_count + structural_noop_count
                if structural_noop_count >= 0
                else -1,
            ),
        )
    )
    authorization_fraction = (
        float(value.get("authorization_fraction"))
        if value.get("authorization_fraction") is not None
        else authorized_count / transition_count
        if transition_count > 0 and authorized_count >= 0
        else 1.0
        if transition_count == 0 and authorized_count == 0
        else float("nan")
    )
    certificate_fraction = (
        float(value.get("certificate_fraction"))
        if value.get("certificate_fraction") is not None
        else certified_count / active_count if active_count > 0 else 1.0
    )
    fallback_count = int(
        value.get("reference_fallback_count", value.get("fallback_count", 0))
    )
    fallback_fraction = (
        float(value.get("fallback_fraction"))
        if value.get("fallback_fraction") is not None
        else fallback_count / transition_count if transition_count > 0 else float("nan")
    )
    elapsed = float(
        value.get("reference_elapsed_seconds", value.get("elapsed_seconds", float("nan")))
    )
    fallback_seconds = float(
        value.get("reference_fallback_seconds", value.get("fallback_seconds", 0.0))
    )
    fallback_time_fraction = (
        float(value.get("fallback_time_fraction"))
        if value.get("fallback_time_fraction") is not None
        else fallback_seconds / elapsed if elapsed > 0.0 else 0.0
    )
    rate = float(
        value.get(
            "transitions_per_second",
            transition_count / elapsed if elapsed > 0.0 else float("nan"),
        )
    )
    simplex_error = float(
        value.get("maximum_simplex_mass_error", value.get("maximum_global_mass_error", float("inf")))
    )
    pair_error = float(value.get("maximum_pair_mass_error", float("inf")))
    # Several core summaries expose the same counters both in a nested map and
    # as convenience top-level fields.  They are aliases, not independent
    # events, so normalize with ``max`` instead of summing them twice.
    forbidden: dict[str, int] = {}
    for source_name in ("reference_forbidden_counts", "forbidden_counts"):
        source = value.get(source_name, {})
        if isinstance(source, Mapping):
            for name, count in source.items():
                forbidden[str(name)] = max(forbidden.get(str(name), 0), int(count))
    for name in (
        "boundary_rejection_count",
        "clipping_count",
        "correction_count",
        "floor_count",
        "limiter_count",
        "projection_count",
        "renormalization_count",
        "nonfinite_count",
        "approximation_count",
        "resource_cap_count",
        "invalid_density_count",
    ):
        forbidden[name] = max(forbidden.get(name, 0), int(value.get(name, 0)))
    peak = int(
        value.get(
            "maximum_cuda_memory_allocated",
            value.get(
                "peak_cuda_memory_allocated_bytes",
                value.get("peak_cuda_memory_bytes", 0),
            ),
        )
    )
    total = int(
        value.get("cuda_total_memory", value.get("total_cuda_memory_bytes", 0))
    )
    memory_fraction = peak / total if total > 0 else (0.0 if test_only else float("nan"))
    throughput_observation_meets_minimum = bool(
        math.isfinite(rate) and rate >= MINIMUM_TRANSITIONS_PER_SECOND
    )
    checks = {
        "transition_count": bool(test_only or transition_count == int(expected_transition_count)),
        "authorization_counts": bool(
            active_count >= 0
            and structural_noop_count >= 0
            and active_count + structural_noop_count == transition_count
            and certified_count == active_count
            and authorized_count == transition_count
            and authorization_fraction == 1.0
        ),
        "certificate_fraction": bool(certificate_fraction == 1.0),
        "fallback_fraction": bool(
            math.isfinite(fallback_fraction)
            and fallback_fraction <= MAXIMUM_FALLBACK_FRACTION
        ),
        "fallback_time_fraction": bool(
            math.isfinite(fallback_time_fraction)
            and fallback_time_fraction <= MAXIMUM_FALLBACK_TIME_FRACTION
        ),
        "simplex_mass": bool(math.isfinite(simplex_error) and simplex_error <= MAXIMUM_MASS_ERROR),
        "pair_mass": bool(math.isfinite(pair_error) and pair_error <= MAXIMUM_MASS_ERROR),
        "throughput": bool(
            test_only
            or not enforce_throughput
            or throughput_observation_meets_minimum
        ),
        "memory": bool(test_only or (math.isfinite(memory_fraction) and memory_fraction <= MAXIMUM_MEMORY_FRACTION)),
        "forbidden_events": not any(forbidden.values()),
    }
    return {
        "expected_transition_count": int(expected_transition_count),
        "transition_count": transition_count,
        "active_count": active_count,
        "structural_noop_count": structural_noop_count,
        "authorized_count": authorized_count,
        "authorization_fraction": authorization_fraction,
        "certified_count": certified_count,
        "certificate_fraction": certificate_fraction,
        "fallback_count": fallback_count,
        "fallback_fraction": fallback_fraction,
        "fallback_seconds": fallback_seconds,
        "fallback_time_fraction": fallback_time_fraction,
        "elapsed_seconds": elapsed,
        "transitions_per_second": rate,
        "minimum_transitions_per_second": MINIMUM_TRANSITIONS_PER_SECOND,
        "throughput_gate_applied": int(enforce_throughput and not test_only),
        "throughput_observation_meets_minimum": int(
            throughput_observation_meets_minimum
        ),
        "maximum_simplex_mass_error": simplex_error,
        "maximum_pair_mass_error": pair_error,
        "maximum_cuda_memory_allocated": peak,
        "total_cuda_memory_bytes": total,
        "peak_memory_fraction": memory_fraction,
        "forbidden_counts": forbidden,
        "checks": checks,
        "passed": int(all(checks.values())),
    }


def _profile_health_classification(health: Mapping[str, Any]) -> str:
    """Separate timed-profile resource failures from integrity failures."""

    checks = health.get("checks", {})
    if not isinstance(checks, Mapping) or not checks:
        return "integrity"
    failed = {str(name) for name, passed in checks.items() if not bool(passed)}
    if not failed:
        return "passed"
    if failed.issubset({"throughput", "memory"}):
        return "resource"
    return "integrity"


def _resource_usage(run_dir: Path, *, role: str | None = None) -> dict[str, Any]:
    """Measure committed wall/storage evidence without trusting a mutable ledger."""

    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "forward").glob("*/forward_summary.json")):
        summary = _load_semantic(path, "forward resource summary")
        current_role = str(summary.get("role"))
        if role is not None and current_role != role:
            continue
        health = summary.get("health", {})
        rows.append(
            {
                "artifact": path.relative_to(run_dir).as_posix(),
                "role": current_role,
                # A resumed invocation's end-to-end timer covers only its
                # tail, whereas health elapsed time is reconstructed from the
                # complete committed shard chain.  An uninterrupted
                # invocation usually has the opposite ordering because it
                # includes commit/render overhead.  The resource gate must
                # conservatively retain both cases.
                "elapsed_seconds": max(
                    float(summary.get("end_to_end_elapsed_seconds", 0.0)),
                    float(health.get("elapsed_seconds", 0.0)),
                ),
                "transition_count": int(health.get("transition_count", 0)),
            }
        )
    for path in sorted(run_dir.glob("*/*/*/trajectory_summary.json")):
        summary = _load_semantic(path, "reverse resource summary")
        if isinstance(summary.get("fused_family_binding"), list):
            # Fused wall time belongs to its batch family, never duplicated
            # into each scientific row.
            continue
        current_role = str(summary.get("role"))
        if role is not None and current_role != role:
            continue
        health = summary.get("health", {})
        rows.append(
            {
                "artifact": path.relative_to(run_dir).as_posix(),
                "role": current_role,
                "elapsed_seconds": max(
                    float(summary.get("end_to_end_elapsed_seconds", 0.0)),
                    float(health.get("elapsed_seconds", 0.0)),
                ),
                "transition_count": int(health.get("transition_count", 0)),
            }
        )
    accounted_family_roots: set[Path] = set()
    transition_only_rate: float | None = None
    profile_rates: dict[int, float] = {}
    projection_path = run_dir / "preflight/resource_projection.json"
    if projection_path.is_file():
        try:
            projection_record = _load_semantic(projection_path, "resource projection")
            candidate_rate = float(
                projection_record.get("transition_only_effective_rate", float("nan"))
            )
            if math.isfinite(candidate_rate) and candidate_rate > 0.0:
                transition_only_rate = candidate_rate
            profile_table = projection_record.get("profiles", {})
            if isinstance(profile_table, Mapping):
                for row_count, name in ((3, "reverse_p3"), (6, "reverse_p6")):
                    value = profile_table.get(name, {})
                    if isinstance(value, Mapping):
                        measured = float(value.get("slowest_profile_rate", float("nan")))
                        if math.isfinite(measured) and measured > 0.0:
                            profile_rates[row_count] = measured
                if 3 in profile_rates and 6 in profile_rates:
                    profile_rates[2] = min(profile_rates[3], profile_rates[6])
        except ArtifactCompatibilityError:
            raise
    for path in sorted(run_dir.rglob("family_summary.json")):
        summary = _load_semantic(path, "fused family resource summary")
        if int(summary.get("resource_accounting_authority", 0)) != 1:
            continue
        family_role = str(summary.get("family_name", "")).split("-", 1)[0]
        if role is not None and family_role != role:
            continue
        rows.append(
            {
                "artifact": path.relative_to(run_dir).as_posix(),
                "role": family_role,
                "elapsed_seconds": max(
                    float(summary.get("end_to_end_elapsed_seconds", 0.0)),
                    float(summary.get("health", {}).get("elapsed_seconds", 0.0)),
                ),
                "transition_count": int(
                    summary.get("health", {}).get("transition_count", 0)
                ),
            }
        )
        accounted_family_roots.add(path.parent.resolve())
    # An interrupted family has no summary yet.  Its verified committed shard
    # elapsed and transitions still consume the cap and must survive resume.
    for path in sorted(run_dir.rglob("fused_families/*/*/shard-*.json")):
        destination = path.parents[3].resolve()
        if destination in accounted_family_roots:
            continue
        try:
            relative = destination.relative_to(run_dir.resolve())
        except ValueError:
            continue
        if relative.parts and relative.parts[0] == "preflight":
            continue
        family_role = path.parents[1].name.split("-", 1)[0]
        if role is not None and family_role != role:
            continue
        record = _load_semantic(path, "partial fused family shard")
        if int(record.get("committed", 0)) != 1:
            continue
        row_count = len(record.get("row_table", ()))
        committed_rate = profile_rates.get(row_count, transition_only_rate)
        rows.append(
            {
                "artifact": path.relative_to(run_dir).as_posix(),
                "role": family_role,
                "elapsed_seconds": max(
                    float(record.get("elapsed_seconds", 0.0)),
                    (
                        int(record.get("transition_count", 0)) / committed_rate
                        if committed_rate is not None
                        else 0.0
                    ),
                ),
                "transition_count": int(record.get("transition_count", 0)),
            }
        )
    setup_path = run_dir / "metrics/resume_fused_backend_setup.json"
    if role is None and setup_path.is_file():
        setup = _load_semantic(setup_path, "resume fused backend setup")
        rows.append(
            {
                "artifact": setup_path.relative_to(run_dir).as_posix(),
                "role": "shared_setup",
                "elapsed_seconds": float(setup.get("elapsed_seconds", 0.0)),
                "transition_count": 0,
            }
        )
    # The fused projection freezes 300 seconds of non-shard work.  Debit it
    # exactly once from the main ledger, while per-operation planning uses the
    # transition-only measured rate.  This prevents the reserve from vanishing
    # when a projected operation is replaced by its measured elapsed time.
    config_path = run_dir / "scientific_config.json"
    if role is None and config_path.is_file():
        config = _load_semantic(config_path, "scientific configuration")
        if _fused_mode(config) and (run_dir / "preflight_gate.json").is_file():
            rows.append(
                {
                    "artifact": "preflight/fused_resource_projection.json#fixed-reserve",
                    "role": "shared_fixed_reserve",
                    "elapsed_seconds": FUSED_PROJECTION_FIXED_RESERVE_SECONDS,
                    "transition_count": 0,
                }
            )
    elapsed = math.fsum(row["elapsed_seconds"] for row in rows)
    transitions = sum(row["transition_count"] for row in rows)
    return {
        "elapsed_seconds": elapsed,
        "transition_count": transitions,
        "persisted_bytes": _directory_bytes(run_dir),
        "artifacts": rows,
    }


def _planning_rate(run_dir: Path, *, profile_name: str | None = None) -> float:
    projection = _load_semantic(
        run_dir / "preflight/resource_projection.json", "resource projection"
    )
    profiles = projection.get("profiles", {})
    if profile_name is not None and isinstance(profiles, Mapping):
        if profile_name == "reverse_p2":
            candidates = [
                float(profiles[name].get("slowest_profile_rate", float("nan")))
                for name in ("reverse_p3", "reverse_p6")
                if isinstance(profiles.get(name), Mapping)
            ]
            rate = min(candidates, default=float("nan"))
        else:
            row = profiles.get(profile_name, {})
            rate = (
                float(row.get("slowest_profile_rate", float("nan")))
                if isinstance(row, Mapping)
                else float("nan")
            )
    else:
        rate = float(
            projection.get(
                "transition_only_effective_rate",
                projection.get(
                    "slowest_complete_repeat_rate",
                    projection.get("effective_rate", float("nan")),
                ),
            )
        )
    if not math.isfinite(rate) or rate <= 0.0:
        raise ArtifactCompatibilityError("resource projection rate is invalid")
    return rate


def _profile_shard_storage_estimate(run_dir: Path, profile_name: str | None) -> int:
    """Conservative committed-byte allowance for one future profile shard."""

    if profile_name is None:
        return 0
    projection = _load_semantic(
        run_dir / "preflight/resource_projection.json", "resource projection"
    )
    profiles = projection.get("profiles", {})
    names = (
        ("reverse_p3", "reverse_p6")
        if profile_name == "reverse_p2"
        else (profile_name,)
    )
    candidates: list[int] = []
    if isinstance(profiles, Mapping):
        for name in names:
            row = profiles.get(name, {})
            if isinstance(row, Mapping):
                values = row.get("repeat_persisted_bytes", ())
                if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                    candidates.extend(int(value) for value in values)
    if not candidates or any(value < 0 for value in candidates):
        raise ArtifactCompatibilityError("profile storage estimate is unavailable")
    return max(candidates)


def _ensure_execution_budget(
    run_dir: Path,
    *,
    role: str,
    additional_transitions: int,
    test_only: bool,
    operation: str,
    profile_name: str | None = None,
    additional_persisted_bytes: int = 0,
) -> dict[str, Any]:
    """Fail before opening a role/variant that cannot fit the frozen cap."""

    usage = _resource_usage(run_dir)
    role_usage = _resource_usage(run_dir, role="replication")
    rate = _planning_rate(run_dir, profile_name=profile_name)
    projected_additional = float(additional_transitions) / rate
    if int(additional_persisted_bytes) < 0:
        raise ArtifactCompatibilityError("additional persisted-byte projection is negative")
    config = (
        _load_semantic(run_dir / "scientific_config.json", "scientific configuration")
        if (run_dir / "scientific_config.json").is_file()
        else {}
    )
    fixed_storage_reserve = (
        FUSED_PROJECTION_FIXED_STORAGE_RESERVE_BYTES if _fused_mode(config) else 0
    )
    projected_persisted = (
        int(usage["persisted_bytes"])
        + int(additional_persisted_bytes)
        + fixed_storage_reserve
    )
    if role == "replication":
        elapsed = float(role_usage["elapsed_seconds"])
        limit = MAXIMUM_REPLICATION_WALL_SECONDS
    else:
        elapsed = float(usage["elapsed_seconds"] - role_usage["elapsed_seconds"])
        limit = MAXIMUM_MAIN_WALL_SECONDS
    record = _semantic(
        {
            "schema": (
                FUSED_RUN_SCHEMA if _fused_mode(config) else RUN_SCHEMA
            )
            + "-resource-capacity-check",
            "schema_version": 1,
            "operation": operation,
            "role": role,
            "committed_elapsed_seconds": elapsed,
            "additional_transition_count": int(additional_transitions),
            "planning_rate": rate,
            "planning_profile": profile_name,
            "projected_additional_seconds": projected_additional,
            "projected_total_seconds": elapsed + projected_additional,
            "maximum_seconds": limit,
            "persisted_bytes": int(usage["persisted_bytes"]),
            "additional_persisted_bytes": int(additional_persisted_bytes),
            "fixed_uncommitted_storage_reserve_bytes": fixed_storage_reserve,
            "projected_persisted_bytes": projected_persisted,
            "maximum_persisted_bytes": MAXIMUM_PERSISTED_BYTES,
            "passed": int(
                test_only
                or (
                    elapsed + projected_additional <= limit
                    and projected_persisted <= MAXIMUM_PERSISTED_BYTES
                )
            ),
        }
    )
    safe = operation.replace("/", "-").replace(" ", "-")
    atomic_write_json(run_dir / "resource_checks" / f"{safe}.json", record)
    if not int(record["passed"]):
        raise RolloutCLIError(
            f"frozen resource budget cannot accommodate {operation}",
            failure_domain="resource_budget",
            failure_code="rollout_resource_budget_exhausted",
        )
    return record


def _save_png(path: Path, pixels: np.ndarray) -> None:
    from PIL import Image

    array = np.asarray(pixels, dtype=np.uint8)
    if array.ndim != 2:
        raise RolloutCLIError("rendered image must be a two-dimensional uint8 array")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp.png", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        Image.fromarray(array, mode="L").save(temporary, format="PNG")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _contact_sheet(path: Path, cells: Sequence[tuple[str, np.ndarray]]) -> None:
    from PIL import Image, ImageDraw

    if not cells:
        return
    scale = 5
    cell_width, cell_height = 28 * scale, 28 * scale + 22
    sheet = Image.new("L", (cell_width * len(cells), cell_height), color=255)
    draw = ImageDraw.Draw(sheet)
    for index, (label, pixels) in enumerate(cells):
        image = Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="L").resize(
            (28 * scale, 28 * scale), resample=Image.Resampling.NEAREST
        )
        sheet.paste(image, (index * cell_width, 0))
        draw.text((index * cell_width + 2, 28 * scale + 3), str(label)[:20], fill=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp.png", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        sheet.save(temporary, format="PNG")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_parent_registry(parent: Path, *, test_only: bool) -> dict[str, Any]:
    if test_only:
        return _semantic(
            {
                "schema": TEST_RUN_SCHEMA + "-frequency1-parent-binding",
                "schema_version": 1,
                "passed": 1,
                "production_parent_evidence_used": 0,
            }
        )
    if parent.name != PARENT_BASENAME:
        raise ArtifactCompatibilityError("frequency-one parent basename changed")
    registry_path = parent / "artifact_registry.json"
    if file_fingerprint(registry_path) != PARENT_REGISTRY_FILE_SHA256:
        raise ArtifactCompatibilityError("frequency-one registry file hash changed")
    registry = _load_json(registry_path)
    if (
        int(registry.get("artifact_count", -1)) != PARENT_REGISTRY_COUNT
        or registry.get("semantic_sha256") != PARENT_REGISTRY_SEMANTIC_SHA256
    ):
        raise ArtifactCompatibilityError("frequency-one registry commitment changed")
    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != PARENT_REGISTRY_COUNT:
        raise ArtifactCompatibilityError("frequency-one registry rows are malformed")
    forbidden_confirmation = []
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise ArtifactCompatibilityError("frequency-one registry row is malformed")
        relative = str(row.get("path", ""))
        if relative.startswith("confirmation/") or relative.startswith("confirm/"):
            forbidden_confirmation.append(relative)
            continue
        path = parent / relative
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(row.get("size", -1))
            or file_fingerprint(path) != row.get("sha256")
        ):
            raise ArtifactCompatibilityError(f"frequency-one artifact changed: {relative}")
    if forbidden_confirmation:
        raise ArtifactCompatibilityError("frequency-one confirmation role was opened")
    manifest = _load_json(parent / "run_manifest.json")
    config = _load_json(parent / "scientific_config.json")
    status = _load_json(parent / "run_status.json")
    decision = _load_json(parent / "frequency1_coordinate_learnability_decision.json")
    if (
        manifest.get("source_fingerprint") != PARENT_SOURCE_FINGERPRINT
        or config.get("semantic_sha256") != PARENT_CONFIG_SHA256
        or status.get("state") != "complete"
        or status.get("decision") != PARENT_DECISION
        or decision.get("decision") != PARENT_DECISION
        or int(decision.get("confirmation_performed", -1)) != 0
        or (parent / "confirmation_namespace_open.json").exists()
    ):
        raise ArtifactCompatibilityError("frequency-one terminal contract changed")
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-frequency1-parent-binding",
            "schema_version": 1,
            "passed": 1,
            "run_dir": str(parent.resolve()),
            "registry_count": PARENT_REGISTRY_COUNT,
            "registry_semantic_sha256": PARENT_REGISTRY_SEMANTIC_SHA256,
            "registry_file_sha256": PARENT_REGISTRY_FILE_SHA256,
            "source_fingerprint": PARENT_SOURCE_FINGERPRINT,
            "scientific_config_sha256": PARENT_CONFIG_SHA256,
            "terminal_decision": PARENT_DECISION,
            "confirmation_evidence_opened": 0,
        }
    )


def _verify_continuation_carrier(
    carrier: Path,
    *,
    test_only: bool,
) -> dict[str, Any]:
    """Verify the sealed resource-stop run and transfer only unopened roles.

    The carrier is never resumed or mutated.  Its historical path plan used
    ``used_initially`` for all reserved roles, so path authority is derived
    from the terminal stage and artifact inventory rather than that misleading
    planning field.
    """

    carrier = carrier.resolve()
    if not carrier.is_dir():
        raise ArtifactCompatibilityError(f"continuation carrier does not exist: {carrier}")
    if test_only:
        return _semantic(
            {
                "schema": FUSED_TEST_RUN_SCHEMA + "-continuation-binding",
                "schema_version": 1,
                "passed": 1,
                "production_carrier_evidence_used": 0,
                "carrier_run_dir": str(carrier),
                "carrier_terminal_decision": "rollout_resource_budget_exhausted",
                "carrier_failure_code": "rollout_main_workflow_computationally_infeasible",
                "realized_path_ids": [PATH_IDS["preflight"]],
                "transferred_unopened_roles": {
                    role: path_id
                    for role, path_id in PATH_IDS.items()
                    if role != "preflight"
                },
                "objective_evidence_opened": 0,
                "test_only": 1,
            }
        )
    if carrier.name != CONTINUATION_BASENAME:
        raise ArtifactCompatibilityError("continuation carrier basename changed")
    for relative, expected in (
        ("artifact_manifest.json", CONTINUATION_REGISTRY_FILE_SHA256),
        ("SHA256SUMS.txt", CONTINUATION_CHECKSUM_FILE_SHA256),
        ("bundle_integrity_audit.json", CONTINUATION_BUNDLE_AUDIT_FILE_SHA256),
        ("run_manifest.json", CONTINUATION_RUN_MANIFEST_FILE_SHA256),
    ):
        path = carrier / relative
        if not path.is_file() or file_fingerprint(path) != expected:
            raise ArtifactCompatibilityError(
                f"continuation carrier commitment changed: {relative}"
            )
    registry = _verify_artifact_manifest(carrier)
    if (
        int(registry.get("artifact_count", -1)) != CONTINUATION_ARTIFACT_COUNT
        or registry.get("semantic_sha256") != CONTINUATION_REGISTRY_SEMANTIC_SHA256
    ):
        raise ArtifactCompatibilityError("continuation carrier registry changed")
    manifest = _load_semantic(carrier / "run_manifest.json", "carrier run manifest")
    config = _load_semantic(carrier / "scientific_config.json", "carrier scientific config")
    path_plan = _load_semantic(carrier / "path_id_plan.json", "carrier path plan")
    gate = _load_semantic(carrier / "preflight_gate.json", "carrier preflight gate")
    projection = _load_semantic(
        carrier / "preflight/resource_projection.json", "carrier resource projection"
    )
    status = _load_json(carrier / "run_status.json")
    failure = _load_semantic(carrier / "failure.json", "carrier failure")
    decision = _load_semantic(
        carrier / "exploratory_decision.json", "carrier decision"
    )
    if (
        manifest.get("source_fingerprint") != CONTINUATION_SOURCE_FINGERPRINT
        or config.get("semantic_sha256") != CONTINUATION_CONFIG_SEMANTIC_SHA256
        or path_plan.get("semantic_sha256") != CONTINUATION_PATH_PLAN_SEMANTIC_SHA256
        or gate.get("semantic_sha256") != CONTINUATION_PREFLIGHT_GATE_SEMANTIC_SHA256
        or projection.get("semantic_sha256")
        != CONTINUATION_RESOURCE_PROJECTION_SEMANTIC_SHA256
    ):
        raise ArtifactCompatibilityError("continuation carrier source/configuration changed")
    checks = gate.get("checks")
    if not isinstance(checks, Mapping):
        raise ArtifactCompatibilityError("continuation preflight checks are malformed")
    nonresource_checks = {
        str(name): bool(value)
        for name, value in checks.items()
        if str(name) != "resource_projection"
    }
    if (
        not nonresource_checks
        or not all(nonresource_checks.values())
        or bool(checks.get("resource_projection"))
        or int(gate.get("passed", -1)) != 0
    ):
        raise ArtifactCompatibilityError("continuation preflight adjudication changed")
    if (
        status.get("state") != "failed"
        or status.get("stage") != "preflight"
        or status.get("decision") != "rollout_resource_budget_exhausted"
        or status.get("failure_code")
        != "rollout_main_workflow_computationally_infeasible"
        or failure.get("failure_code")
        != "rollout_main_workflow_computationally_infeasible"
        or decision.get("decision") != "rollout_resource_budget_exhausted"
    ):
        raise ArtifactCompatibilityError("continuation terminal status changed")
    forbidden_roots = (
        "forward",
        "development",
        "evaluation",
        "replication",
        "training",
        "sampling",
        "confirmation",
    )
    opened = [name for name in forbidden_roots if (carrier / name).exists()]
    if opened:
        raise ArtifactCompatibilityError(
            f"continuation carrier opened objective evidence: {opened}"
        )
    if any(int(status.get(name, 0)) for name in NO_WORK):
        raise ArtifactCompatibilityError("continuation carrier work boundary changed")
    transferred = {
        role: PATH_IDS[role]
        for role in ("development", "evaluation", "replication")
    }
    return _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-continuation-binding",
            "schema_version": 1,
            "passed": 1,
            "carrier_run_dir": str(carrier),
            "carrier_artifact_count": CONTINUATION_ARTIFACT_COUNT,
            "carrier_registry_semantic_sha256": CONTINUATION_REGISTRY_SEMANTIC_SHA256,
            "carrier_registry_file_sha256": CONTINUATION_REGISTRY_FILE_SHA256,
            "carrier_checksum_file_sha256": CONTINUATION_CHECKSUM_FILE_SHA256,
            "carrier_bundle_audit_file_sha256": CONTINUATION_BUNDLE_AUDIT_FILE_SHA256,
            "carrier_source_fingerprint": CONTINUATION_SOURCE_FINGERPRINT,
            "carrier_scientific_config_sha256": CONTINUATION_CONFIG_SEMANTIC_SHA256,
            "carrier_terminal_decision": status["decision"],
            "carrier_failure_code": status["failure_code"],
            "carrier_nonresource_preflight_checks": nonresource_checks,
            "carrier_resource_projection_passed": 0,
            "realized_path_ids": [PATH_IDS["preflight"]],
            "transferred_unopened_roles": transferred,
            "objective_evidence_opened": 0,
            "carrier_mutated": 0,
        }
    )


def _test_source() -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((28, 28), dtype=np.float64)
    image[4:24, 12:16] = 1.0
    image[4:8, 8:18] = 1.0
    image /= image.sum()
    mixed = (1.0 - LAMBDA_MIX) * image.reshape(-1) + LAMBDA_MIX / 784.0
    return np.ascontiguousarray(image.reshape(-1)), np.ascontiguousarray(mixed)


def _copy_and_bind_inputs(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    destination = run_dir / "input_bindings"
    destination.mkdir(parents=True, exist_ok=True)
    if args.test_only:
        source, target = _test_source()
        _atomic_npz(destination / "source_image.npz", image=source, mixed_target=target)
        atomic_write_json(
            destination / "source_image.json",
            {
                "label": 3,
                "dataset_index": 7,
                "lambda_mix": LAMBDA_MIX,
                "image_sha256": _array_sha256(source),
                "mixed_target_sha256": _array_sha256(target),
                "test_only": 1,
            },
        )
        # The fake workflow deliberately has no production checkpoint bytes.
        (destination / "checkpoint.pt").write_bytes(b"test-only-frequency1-checkpoint")
        record = {
            "schema": TEST_RUN_SCHEMA + "-input-binding",
            "schema_version": 1,
            "checkpoint_seed": CHECKPOINT_SEED,
            "checkpoint_update": CHECKPOINT_UPDATE,
            "checkpoint_file_sha256": file_fingerprint(destination / "checkpoint.pt"),
            "source_image_array_sha256": _array_sha256(source),
            "mixed_target_array_sha256": _array_sha256(target),
            "source_image_json_sha256": file_fingerprint(
                destination / "source_image.json"
            ),
            "source_image_npz_sha256": file_fingerprint(
                destination / "source_image.npz"
            ),
            "test_only": 1,
        }
        atomic_write_json(destination / "input_binding.json", _semantic(record))
        return _semantic(record)

    from mnist.d0_jacobi_rb_tangent_rollout import (
        load_verified_frequency1_checkpoint,
        load_verified_source_target,
        source_measure_sha256,
    )

    verified_checkpoint = load_verified_frequency1_checkpoint(
        args.frequency1_run_dir,
        device="cpu",
        expected_seed=CHECKPOINT_SEED,
        expected_update=CHECKPOINT_UPDATE,
    )
    verified_source = load_verified_source_target(args.source_run_dir)
    checkpoint_path = args.frequency1_run_dir / (
        f"checkpoints/physical/seed-{CHECKPOINT_SEED}/update-{CHECKPOINT_UPDATE:04d}.pt"
    )
    if (
        file_fingerprint(checkpoint_path) != CHECKPOINT_FILE_SHA256
        or str(_field(verified_checkpoint, "state_sha256", "")) != CHECKPOINT_STATE_SHA256
    ):
        raise ArtifactCompatibilityError("frozen checkpoint commitment changed")
    source = _state_row(_field(verified_source, "source_image", _field(verified_source, "image")))
    target = _state_row(_field(verified_source, "mixed_target", _field(verified_source, "target")))
    if (
        source_measure_sha256(source) != SOURCE_IMAGE_SHA256
        or source_measure_sha256(target) != MIXED_TARGET_SHA256
        or verified_source.metadata.get("image_sha256") != SOURCE_IMAGE_SHA256
        or verified_source.metadata.get("mixed_target_sha256") != MIXED_TARGET_SHA256
    ):
        raise ArtifactCompatibilityError("frozen source or mixed-target bytes changed")
    source_json = args.source_run_dir / "source_image.json"
    source_npz = args.source_run_dir / "source_image.npz"
    shutil.copy2(checkpoint_path, destination / "checkpoint.pt")
    shutil.copy2(source_json, destination / "source_image.json")
    shutil.copy2(source_npz, destination / "source_image.npz")
    record = _semantic(
        {
            "schema": RUN_SCHEMA + "-input-binding",
            "schema_version": 1,
            "frequency1_parent_run_dir": str(args.frequency1_run_dir.resolve()),
            "source_parent_run_dir": str(args.source_run_dir.resolve()),
            "checkpoint_seed": CHECKPOINT_SEED,
            "checkpoint_update": CHECKPOINT_UPDATE,
            "checkpoint_file_sha256": CHECKPOINT_FILE_SHA256,
            "checkpoint_state_sha256": CHECKPOINT_STATE_SHA256,
            "source_image_measure_sha256": SOURCE_IMAGE_SHA256,
            "mixed_target_measure_sha256": MIXED_TARGET_SHA256,
            "source_image_array_sha256": _array_sha256(source),
            "mixed_target_array_sha256": _array_sha256(target),
            "source_image_json_sha256": file_fingerprint(source_json),
            "source_image_npz_sha256": file_fingerprint(source_npz),
            "copied_inputs_only": 1,
            "confirmation_evidence_opened": 0,
        }
    )
    atomic_write_json(destination / "input_binding.json", record)
    return record


def _path_collision_record(
    repository_root: Path,
    run_dir: Path,
    *,
    proposed_path_ids: Mapping[str, int] | None = None,
    continuation_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_learnability import discover_repository_path_id_claims

    proposed_mapping = dict(PATH_IDS if proposed_path_ids is None else proposed_path_ids)
    proposed = set(int(value) for value in proposed_mapping.values())
    collisions: list[dict[str, Any]] = []
    claims: list[tuple[str, str, int, int]] = []
    # Python declarations are already paired by the repository scanner.  JSON
    # role-slot records need explicit pairing: treating ``stop_exclusive`` as
    # another SLOT start invents a spurious 0x1000-wide claim.
    for claim in discover_repository_path_id_claims(repository_root):
        if Path(claim.source).suffix == ".py":
            claims.append((claim.source, claim.name, int(claim.start), int(claim.stop)))

    def visit_json(value: Any, *, source: str, name: str = "") -> None:
        if isinstance(value, Mapping):
            start = value.get("start", value.get("start_inclusive"))
            stop = value.get("stop_exclusive", value.get("stop"))
            count = value.get("count")
            if isinstance(start, int) and not isinstance(start, bool):
                if not isinstance(stop, int) and isinstance(count, int):
                    stop = start + count
                if isinstance(stop, int) and 0 <= start < stop <= (1 << 20):
                    claims.append((source, name or "role", int(start), int(stop)))
            for key in ("path_ids", "used_initially"):
                ids = value.get(key)
                if isinstance(ids, list):
                    for index, item in enumerate(ids):
                        if isinstance(item, int) and not isinstance(item, bool) and 0 <= item < (1 << 20):
                            claims.append((source, f"{name}.{key}[{index}]", item, item + 1))
            slot = value.get("slot")
            if (
                isinstance(slot, list)
                and len(slot) == 2
                and all(isinstance(item, int) and not isinstance(item, bool) for item in slot)
                and 0 <= slot[0] < slot[1] <= (1 << 20)
            ):
                claims.append((source, f"{name}.slot", int(slot[0]), int(slot[1])))
            # Realized-ID records are not uniformly schemaed across this
            # repository.  Recognize direct scalar path claims without
            # mistaking range endpoints/counts for identities.
            for key, scalar in value.items():
                lowered_key = str(key).lower()
                context_is_path_map = name.lower().endswith("path_ids") or ".path_ids" in name.lower()
                scalar_claim = (
                    lowered_key == "path_id"
                    or lowered_key.endswith("_path_id")
                    or context_is_path_map
                )
                if (
                    scalar_claim
                    and isinstance(scalar, int)
                    and not isinstance(scalar, bool)
                    and 0 <= scalar < (1 << 20)
                ):
                    claims.append(
                        (source, f"{name}.{key}" if name else str(key), scalar, scalar + 1)
                    )
            for key, child in value.items():
                if isinstance(child, (Mapping, list)):
                    visit_json(child, source=source, name=f"{name}.{key}" if name else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, (Mapping, list)):
                    visit_json(child, source=source, name=f"{name}[{index}]")

    for base in (repository_root / "runs",):
        if not base.is_dir():
            continue
        # Historical workflows used several noncanonical names, including
        # ``haar_path_id_plan.json`` and ``phase_observer_path_id_plan.json``.
        # Scan every plausible path/ID plan rather than silently omitting one.
        for path in base.rglob("*.json"):
            lowered = path.name.lower()
            if "path" not in lowered or "id" not in lowered:
                continue
            if any(part in {"shards", "checkpoints"} for part in path.parts):
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            visit_json(value, source=str(path))

    for source_text, claim_name, claim_start, claim_stop in claims:
        source = Path(source_text)
        try:
            source.resolve().relative_to(run_dir.resolve())
            continue
        except (ValueError, OSError):
            pass
        overlap = sorted(item for item in proposed if claim_start <= item < claim_stop)
        if not overlap:
            continue
        carrier_authority = False
        if continuation_binding is not None:
            transferred = continuation_binding.get("transferred_unopened_roles", {})
            carrier_root_value = continuation_binding.get("carrier_run_dir")
            if isinstance(transferred, Mapping) and carrier_root_value:
                transferred_ids = {int(value) for value in transferred.values()}
                try:
                    source.resolve().relative_to(Path(str(carrier_root_value)).resolve())
                    source_is_carrier = True
                except (ValueError, OSError):
                    source_is_carrier = False
                carrier_authority = bool(
                    source_is_carrier
                    and set(overlap).issubset(transferred_ids)
                    and int(continuation_binding.get("passed", 0)) == 1
                    and int(continuation_binding.get("objective_evidence_opened", -1)) == 0
                )
        # Historical parents reserved this top interval precisely for later
        # production roles.  Consuming a fresh subslot is not a realized-ID
        # collision; every narrower or realized claim remains fatal.
        generic_reservation = (
            claim_start == 0xF0000
            and claim_stop == 0x100000
            and (
                claim_name.lower().endswith("allocator_reservation")
                or any(token in claim_name.lower() for token in ("reserved", "future", "production"))
            )
        )
        if not generic_reservation and not carrier_authority:
            collisions.append(
                {
                    "source": source_text,
                    "name": claim_name,
                    "path_ids": overlap,
                }
            )
    return _semantic(
        {
            "schema": RUN_SCHEMA + "-path-collision-scan",
            "schema_version": 1,
            "proposed_path_ids": proposed_mapping,
            "collision_count": len(collisions),
            "collisions": collisions[:32],
            "passed": int(not collisions),
            "generic_future_reservation_consumption": 1,
            "continuation_authority_exemption": int(continuation_binding is not None),
        }
    )


def _scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    short_anchor = int(args.test_short_anchor) if args.test_only else SHORT_ANCHOR
    full_anchor = int(args.test_full_anchor) if args.test_only else FULL_ANCHOR
    body: dict[str, Any] = {
            "schema": _schema(args) + "-scientific-config",
            "schema_version": 1,
            "research_mode": "exploratory",
            "decision_question": (
                "Does the fixed frequency-one checkpoint have dynamically useful "
                "reverse-control signal on fresh trajectories?"
            ),
            "objective_bearing_experiment": 1,
            "proxy_only_patches_since_last_objective_bearing_experiment": 0,
            "checkpoint": {
                "seed": CHECKPOINT_SEED,
                "update": CHECKPOINT_UPDATE,
                "role": "historical_validation_inspected_post_hoc_diagnostic",
            },
            "grid_size": 28,
            "alpha": 1.0,
            "jacobi_outer_steps": 512,
            "tau_eff": 5.0e-5,
            "lambda_mix": LAMBDA_MIX,
            "source_label": 3,
            "source_dataset_index": 7,
            "short_anchor": short_anchor,
            "full_anchor": full_anchor,
            "controller_microsteps": MICROSTEPS,
            "learned_gain_grid": list(LEARNED_GAINS),
            "development_selection": (
                "minimum final raw squared L2 error to mixed target; ties smaller gain"
            ),
            "forward_root_seed": FORWARD_ROOT_SEED,
            "reverse_root_seed": REVERSE_ROOT_SEED,
            "path_ids": dict(PATH_IDS),
            "path_id_allocation_version": PATH_ID_ALLOCATION_VERSION,
            "failed_fa_preflight_run": FAILED_FA_PREFLIGHT_RUN,
            "stream_roles": dict(STREAM_ROLES),
            "restart_outer_steps": 8,
            "exploration_backend": "exact certified CUDA Jacobi reference with M=2 control",
            "maximum_main_wall_seconds": MAXIMUM_MAIN_WALL_SECONDS,
            "maximum_replication_wall_seconds": MAXIMUM_REPLICATION_WALL_SECONDS,
            "maximum_persisted_bytes": MAXIMUM_PERSISTED_BYTES,
            "test_only": int(args.test_only),
            "authorizing": 0,
        }
    if _recovery_mode(args):
        body.update(
            {
                "objective_first_recovery": 1,
                "decision_question": (
                    "Starting from a verified MNIST forward anchor, does the frozen "
                    "frequency-one controller at gain 1.0 produce a numerically "
                    "interpretable 128-step reverse suffix relative to paired zero "
                    "control, and what does the source-informed row reveal about "
                    "composition?"
                ),
                "predecessor_run_dir": str(args.predecessor_run_dir.resolve()),
                "predecessor_basename": RECOVERY_PREDECESSOR_BASENAME,
                "development_anchor": str(args.development_anchor),
                "reference_backend": str(args.reference_backend),
                "core_learned_gain": float(args.core_learned_gain),
                "gain_sweep": str(args.gain_sweep),
                "exact_audit_outer_steps": int(args.exact_audit_outer_steps),
                "maximum_main_wall_seconds": float(args.maximum_main_seconds),
                "report_reserve_seconds": RECOVERY_REPORT_RESERVE_SECONDS,
                "projection_safety_factor": RECOVERY_PROJECTION_FACTOR,
                "path_ids": {
                    "development": PATH_IDS["development"],
                    "evaluation": PATH_IDS["evaluation"],
                    "optional_future": PATH_IDS["replication"],
                },
                "path_id_allocation_version": (
                    "frequency1-objective-first-fb-objective-roles-v1"
                ),
                "no_preflight_path_namespace": 1,
                "core_rows": [
                    "development-core-zero",
                    "development-core-learned-1",
                    "development-core-source-informed",
                ],
                "source_informed_row_is_diagnostic_not_integrity_gate": 1,
                "analytic_identity_is_the_only_controller_integrity_gate": 1,
                "backend_contracts": dict(RECOVERY_REFERENCE_CONTRACTS),
                "exploration_backend": (
                    "horizon-local certified-exact first-shard audit followed by "
                    "frozen exact/candidate/auto selection; candidate results are "
                    "explicitly approximate"
                ),
                "candidate_approximation": (
                    "fixed 128-mode Legendre CUDA inverse-CDF candidate with 56 "
                    "bisection steps and stateless Philox; no correct-rounding "
                    "certificate or Arb fallback"
                ),
                "same_process_objective_execution_required": 1,
                "exact_resource_failure_is_backend_selection_not_terminal": 1,
                "confirmation_evidence_opened": 0,
                "proxy_only_patches_since_last_objective_bearing_experiment": 1,
            }
        )
    elif _fused_mode(args):
        body.update(
            {
                "fused_continuation": 1,
                "continuation_run_dir": str(args.continuation_run_dir.resolve()),
                "continuation_basename": (
                    CONTINUATION_BASENAME if not args.test_only else args.continuation_run_dir.name
                ),
                "fused_scheduler_version": "frequency1-fused-laptop-v1",
                "row_identity_separate_from_canonical_path_identity": 1,
                "variant_in_rng_key": 0,
                "fused_preflight_path_ids": dict(FUSED_PREFLIGHT_PATH_IDS),
                "fused_preflight_slot": list(FUSED_PREFLIGHT_SLOT),
                "fused_profile_transitions": dict(FUSED_PROFILE_TRANSITIONS),
                "fused_profile_production_shards": dict(
                    FUSED_PROFILE_PRODUCTION_SHARDS
                ),
                "fused_projection_fixed_reserve_seconds": (
                    FUSED_PROJECTION_FIXED_RESERVE_SECONDS
                ),
                "fused_projection_fixed_storage_reserve_bytes": (
                    FUSED_PROJECTION_FIXED_STORAGE_RESERVE_BYTES
                ),
                "minimum_effective_main_rate": MINIMUM_EFFECTIVE_MAIN_RATE,
                "fused_preflight_learned_gain": FUSED_PREFLIGHT_LEARNED_GAIN,
                "same_process_preflight_to_objective_required": 1,
                "development_fused_rows": 6,
                "evaluation_prefix_fused_rows": 3,
                "evaluation_suffix_fused_rows": 6,
                "replication_fused_rows": 2,
                # The carrier stopped before the objective-bearing rollout.
                # Do not reset the program-level cadence counter merely
                # because this continuation intends to run the objective.
                "proxy_only_patches_since_last_objective_bearing_experiment": 1,
                "proxy_only_counter_scope": (
                    "at_least_one_since_last_objective; exact historical count not established"
                ),
            }
        )
    return _semantic(body)


def _claim_boundary(args_or_config: Any | None = None) -> dict[str, Any]:
    recovery = _recovery_mode(args_or_config or {})
    return _semantic(
        {
            "schema": (
                _schema(args_or_config) if args_or_config is not None else RUN_SCHEMA
            )
            + "-claim-boundary",
            "schema_version": 1,
            "research_mode": "exploratory",
            "permitted_positive_language": (
                "On one exploratory historical forward anchor, the fixed post-hoc "
                "checkpoint at gain 1.0 changed a 128-step reverse suffix relative "
                "to paired zero control under the explicitly recorded exact or "
                "candidate reference backend."
                if recovery
                else "On one fresh exploratory forward-terminal path, the fixed post-hoc "
                "checkpoint and gain reduced reconstruction error relative to paired "
                "zero control under the M=2 exact-reference split."
            ),
            "candidate_results_are_approximate": int(recovery),
            "source_informed_endpoint_is_an_exploratory_diagnostic": int(recovery),
            "validation_pass_claim_authorized": 0,
            "general_generator_claim_authorized": 0,
            "controller_generalization_claim_authorized": 0,
            "prior_start_claim_authorized": 0,
            "eulerian_convergence_claim_authorized": 0,
            "confirmatory_inference_performed": 0,
        }
    )


def _fused_proposed_path_ids() -> dict[str, int]:
    result = {
        "development": PATH_IDS["development"],
        "evaluation": PATH_IDS["evaluation"],
        "replication": PATH_IDS["replication"],
        **{f"preflight_{name}": value for name, value in FUSED_PREFLIGHT_PATH_IDS.items()},
    }
    for value in range(*FUSED_PREFLIGHT_SLOT):
        result.setdefault(f"preflight_slot_{value - FUSED_PREFLIGHT_SLOT[0]:02d}", value)
    return result


def _fused_gain_token(gain: float) -> str:
    """Filesystem/core-safe deterministic token for a frozen gain."""

    return f"{float(gain):g}".replace(".", "p")


def _fused_schedule_plan(*, test_only: bool) -> dict[str, Any]:
    return _semantic(
        {
            "schema": (
                FUSED_TEST_RUN_SCHEMA if test_only else FUSED_RUN_SCHEMA
            )
            + "-schedule-plan",
            "schema_version": 1,
            "scheduler_version": "frequency1-fused-laptop-v1",
            "row_key_unique": 1,
            "duplicate_canonical_path_ids_intentional": 1,
            "variant_in_rng_key": 0,
            "restart_outer_steps": 8,
            "maximum_reference_launch_lanes": 2_352,
            "lane_cap": 4_096,
            "development": {
                "forward_path_id": PATH_IDS["development"],
                "anchor_step": SHORT_ANCHOR,
                "row_order": [
                    "development-short-zero",
                    "development-short-learned-0p5",
                    "development-short-learned-1",
                    "development-short-learned-2",
                    "development-short-learned-4",
                    "development-short-oracle",
                ],
                "row_count": 6,
                "shard_count": 16,
                "transition_count": 8_429_568,
            },
            "evaluation_prefix": {
                "forward_path_id": PATH_IDS["evaluation"],
                "sequence": {"outer_step_start": 511, "outer_step_stop": 128},
                "row_order": ["full-zero", "full-learned", "full-oracle"],
                "row_count": 3,
                "shard_count": 48,
                "transition_count": 12_644_352,
            },
            "evaluation_suffix": {
                "forward_path_id": PATH_IDS["evaluation"],
                "sequence": {"outer_step_start": 127, "outer_step_stop": 0},
                "row_order": [
                    "full-zero",
                    "full-learned",
                    "full-oracle",
                    "short-zero",
                    "short-learned",
                    "short-oracle",
                ],
                "row_count": 6,
                "shard_count": 16,
                "transition_count": 8_429_568,
            },
            "replication": {
                "forward_path_id": PATH_IDS["replication"],
                "row_order": ["full-zero", "full-learned"],
                "row_count": 2,
                "shard_count": 64,
            },
            "profiles": {
                name: {
                    "path_id": FUSED_PREFLIGHT_PATH_IDS[
                        {
                            "forward_p1": "forward_profile",
                            "reverse_p3": "reverse_p3_profile",
                            "reverse_p6": "reverse_p6_profile",
                        }[name]
                    ],
                    "transition_count_per_repeat": FUSED_PROFILE_TRANSITIONS[name],
                    "production_shard_count": FUSED_PROFILE_PRODUCTION_SHARDS[name],
                    "timed_repeats": 3,
                }
                for name in FUSED_PROFILE_TRANSITIONS
            },
            "reverse_profile_learned_gain": FUSED_PREFLIGHT_LEARNED_GAIN,
            "main_transition_count": MAIN_WORKFLOW_TRANSITIONS,
            "fixed_nonshard_reserve_seconds": FUSED_PROJECTION_FIXED_RESERVE_SECONDS,
            "fixed_storage_reserve_bytes": FUSED_PROJECTION_FIXED_STORAGE_RESERVE_BYTES,
            "test_only": int(test_only),
        }
    )


def _fused_resource_projection(
    repeats_by_profile: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    test_only: bool = False,
    current_persisted_bytes: int = 0,
) -> dict[str, Any]:
    """Apply the frozen slowest-repeat projection without favorable averaging."""

    profiles: dict[str, Any] = {}
    output_hashes_identical = True
    individual_profiles_pass = True
    projected_terms: dict[str, float] = {}
    projected_storage_terms: dict[str, int] = {}
    maximum_memory_fraction = 0.0
    for name in ("forward_p1", "reverse_p3", "reverse_p6"):
        rows = list(repeats_by_profile.get(name, ()))
        if len(rows) != 3:
            raise RolloutCLIError(
                f"fused profile {name} must contain exactly three timed repeats",
                failure_domain="resource_budget",
                failure_code="fused_resource_projection_invalid",
            )
        elapsed_values = [float(row.get("elapsed_seconds", float("nan"))) for row in rows]
        if not all(math.isfinite(value) and value > 0.0 for value in elapsed_values):
            raise RolloutCLIError(
                f"fused profile {name} contains an invalid elapsed time",
                failure_domain="resource_budget",
                failure_code="fused_resource_projection_invalid",
            )
        expected_count = FUSED_PROFILE_TRANSITIONS[name]
        if any(int(row.get("transition_count", -1)) != expected_count for row in rows):
            raise RolloutCLIError(
                f"fused profile {name} transition count changed",
                failure_domain="resource_budget",
                failure_code="fused_resource_projection_invalid",
            )
        hashes = [str(row.get("output_sha256", "")) for row in rows]
        hashes_equal = bool(hashes[0] and len(set(hashes)) == 1)
        output_hashes_identical &= hashes_equal
        slowest_seconds = max(elapsed_values)
        rate = expected_count / slowest_seconds
        health_passed = all(int(row.get("health_passed", 1)) == 1 for row in rows)
        memory_values = [float(row.get("peak_memory_fraction", 0.0)) for row in rows]
        if not all(math.isfinite(value) and value >= 0.0 for value in memory_values):
            raise RolloutCLIError(
                f"fused profile {name} contains invalid memory telemetry",
                failure_domain="resource_budget",
                failure_code="fused_resource_projection_invalid",
            )
        maximum_memory_fraction = max(maximum_memory_fraction, *memory_values)
        persisted_values = [int(row.get("persisted_bytes", 0)) for row in rows]
        if any(value < 0 for value in persisted_values):
            raise RolloutCLIError(
                f"fused profile {name} contains invalid storage telemetry",
                failure_domain="resource_budget",
                failure_code="fused_resource_projection_invalid",
            )
        profile_passed = bool(
            (test_only or rate >= MINIMUM_TRANSITIONS_PER_SECOND)
            and hashes_equal
            and health_passed
        )
        individual_profiles_pass &= profile_passed
        projected_seconds = FUSED_PROFILE_PRODUCTION_SHARDS[name] * slowest_seconds
        projected_terms[name] = projected_seconds
        projected_storage_terms[name] = (
            FUSED_PROFILE_PRODUCTION_SHARDS[name] * max(persisted_values)
        )
        profiles[name] = {
            "repeat_elapsed_seconds": elapsed_values,
            "repeat_output_sha256": hashes,
            "repeat_health_passed": [int(row.get("health_passed", 1)) for row in rows],
            "transition_count_per_repeat": expected_count,
            "slowest_repeat_seconds": slowest_seconds,
            "slowest_repeat_index": elapsed_values.index(slowest_seconds),
            "slowest_profile_rate": rate,
            "production_shard_count": FUSED_PROFILE_PRODUCTION_SHARDS[name],
            "projected_seconds": projected_seconds,
            "output_hashes_identical": int(hashes_equal),
            "repeat_peak_memory_fraction": memory_values,
            "repeat_persisted_bytes": persisted_values,
            "projected_persisted_bytes": projected_storage_terms[name],
            "passed": int(profile_passed),
        }
    projected_main_seconds = math.fsum(projected_terms.values()) + float(
        FUSED_PROJECTION_FIXED_RESERVE_SECONDS
    )
    effective_rate = MAIN_WORKFLOW_TRANSITIONS / projected_main_seconds
    transition_only_seconds = math.fsum(projected_terms.values())
    transition_only_effective_rate = (
        MAIN_WORKFLOW_TRANSITIONS / transition_only_seconds
        if transition_only_seconds > 0.0
        else float("nan")
    )
    projected_persisted_bytes = (
        int(current_persisted_bytes)
        + sum(projected_storage_terms.values())
        + FUSED_PROJECTION_FIXED_STORAGE_RESERVE_BYTES
    )
    checks = {
        "profile_repeat_hashes": output_hashes_identical,
        "individual_profile_rates_and_health": individual_profiles_pass,
        "main_wall_time": bool(
            test_only or projected_main_seconds <= MAXIMUM_MAIN_WALL_SECONDS
        ),
        "effective_rate": bool(
            test_only or effective_rate >= MINIMUM_EFFECTIVE_MAIN_RATE
        ),
        "memory": bool(
            test_only or maximum_memory_fraction <= MAXIMUM_MEMORY_FRACTION
        ),
        "persisted_storage": bool(
            test_only or projected_persisted_bytes <= MAXIMUM_PERSISTED_BYTES
        ),
        "exact_transition_arithmetic": bool(
            sum(
                FUSED_PROFILE_TRANSITIONS[name]
                * FUSED_PROFILE_PRODUCTION_SHARDS[name]
                for name in FUSED_PROFILE_TRANSITIONS
            )
            == MAIN_WORKFLOW_TRANSITIONS
        ),
    }
    return _semantic(
        {
            "schema": (
                FUSED_TEST_RUN_SCHEMA if test_only else FUSED_RUN_SCHEMA
            )
            + "-resource-projection",
            "schema_version": 1,
            "gate_type": "execution_integrity_resource_budget",
            "profiles": profiles,
            "projection_terms_seconds": projected_terms,
            "fixed_nonshard_reserve_seconds": FUSED_PROJECTION_FIXED_RESERVE_SECONDS,
            "projected_storage_terms_bytes": projected_storage_terms,
            "fixed_storage_reserve_bytes": FUSED_PROJECTION_FIXED_STORAGE_RESERVE_BYTES,
            "current_persisted_bytes": int(current_persisted_bytes),
            "projected_persisted_bytes": projected_persisted_bytes,
            "maximum_persisted_bytes": MAXIMUM_PERSISTED_BYTES,
            "maximum_peak_memory_fraction": maximum_memory_fraction,
            "maximum_memory_fraction": MAXIMUM_MEMORY_FRACTION,
            "projected_main_transition_count": MAIN_WORKFLOW_TRANSITIONS,
            "projected_main_wall_seconds": projected_main_seconds,
            "effective_rate": effective_rate,
            "transition_only_projected_seconds": transition_only_seconds,
            "transition_only_effective_rate": transition_only_effective_rate,
            "maximum_main_wall_seconds": MAXIMUM_MAIN_WALL_SECONDS,
            "minimum_effective_main_rate": MINIMUM_EFFECTIVE_MAIN_RATE,
            "minimum_individual_profile_rate": MINIMUM_TRANSITIONS_PER_SECOND,
            "slowest_repeat_selected_not_average": 1,
            "checks": checks,
            "passed": int(all(checks.values())),
        }
    )


def _initialize(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    config_path = run_dir / "scientific_config.json"
    if config_path.is_file():
        existing = _load_semantic(config_path, "scientific configuration")
        expected = _scientific_config(args)
        if existing != expected:
            raise ArtifactCompatibilityError("resume scientific configuration changed")
        binding = _load_semantic(
            run_dir / "input_bindings/input_binding.json", "input binding"
        )
        manifest = _load_semantic(run_dir / "run_manifest.json", "run manifest")
        closure = _source_closure()
        if (
            manifest.get("source_fingerprint") != closure["source_fingerprint"]
            or manifest.get("source_files") != closure["files"]
            or manifest.get("input_binding_sha256") != binding.get("semantic_sha256")
        ):
            raise ArtifactCompatibilityError("resume load-bearing source closure changed")
        _verify_copied_inputs(run_dir, binding, test_only=bool(args.test_only))
        continuation_binding: dict[str, Any] | None = None
        if _fused_mode(args):
            continuation_binding = _load_semantic(
                run_dir / "continuation_binding.json", "continuation binding"
            )
            expected_continuation = _verify_continuation_carrier(
                args.continuation_run_dir,
                test_only=bool(args.test_only),
            )
            if continuation_binding != expected_continuation:
                raise ArtifactCompatibilityError("resume continuation binding changed")
            if (
                manifest.get("continuation_binding_sha256")
                != continuation_binding.get("semantic_sha256")
                or manifest.get("continuation_run_dir")
                != str(args.continuation_run_dir.resolve())
            ):
                raise ArtifactCompatibilityError("resume continuation authority changed")
        if not args.test_only:
            if (
                Path(binding["frequency1_parent_run_dir"]).resolve()
                != args.frequency1_run_dir.resolve()
                or Path(binding["source_parent_run_dir"]).resolve()
                != args.source_run_dir.resolve()
            ):
                raise ArtifactCompatibilityError("resume parent input paths changed")
        return existing

    parent_record = _verify_parent_registry(args.frequency1_run_dir, test_only=args.test_only)
    continuation_binding = (
        _verify_continuation_carrier(
            args.continuation_run_dir,
            test_only=bool(args.test_only),
        )
        if _fused_mode(args)
        else None
    )
    proposed_path_ids = (
        _fused_proposed_path_ids() if continuation_binding is not None else dict(PATH_IDS)
    )
    collisions = (
        _semantic(
            {
                "schema": _schema(args) + "-path-collision-scan",
                "schema_version": 1,
                "proposed_path_ids": proposed_path_ids,
                "collision_count": 0,
                "collisions": [],
                "passed": 1,
                "continuation_authority_exemption": int(
                    continuation_binding is not None
                ),
                "test_only": 1,
            }
        )
        if args.test_only
        else _path_collision_record(
            Path.cwd(),
            run_dir,
            proposed_path_ids=proposed_path_ids,
            continuation_binding=continuation_binding,
        )
    )
    if not int(collisions["passed"]):
        raise RolloutCLIError(
            "fresh rollout path allocation collides with existing history",
            failure_domain="path_namespace",
            failure_code="rollout_path_id_collision",
        )
    config = _scientific_config(args)
    atomic_write_json(config_path, config)
    atomic_write_json(run_dir / "claim_boundary.json", _claim_boundary(args))
    if continuation_binding is not None:
        path_plan_body = {
            "schema": _schema(args) + "-path-id-plan",
            "schema_version": 1,
            "allocation_version": FUSED_PATH_ID_ALLOCATION_VERSION,
            "continuation_binding_sha256": continuation_binding["semantic_sha256"],
            "carrier_realized_path_ids": continuation_binding["realized_path_ids"],
            "transferred_unopened_roles": continuation_binding[
                "transferred_unopened_roles"
            ],
            "objective_roles": {
                role: {
                    "path_id": PATH_IDS[role],
                    "authority": "transferred_unopened_from_carrier",
                    "realized_initially": 0,
                }
                for role in ("development", "evaluation", "replication")
            },
            "fused_preflight": {
                "slot": list(FUSED_PREFLIGHT_SLOT),
                "path_ids": dict(FUSED_PREFLIGHT_PATH_IDS),
                "realized_initially": [],
            },
            "collision_scan_sha256": collisions["semantic_sha256"],
            "passed": 1,
        }
    else:
        path_plan_body = {
                "schema": _schema(args) + "-path-id-plan",
                "schema_version": 1,
                "allocation_version": PATH_ID_ALLOCATION_VERSION,
                "superseded_allocation": {
                    "roles": {
                        "preflight": 0xFA000,
                        "development": 0xFA100,
                        "evaluation": 0xFA200,
                        "replication": 0xFA300,
                    },
                    "failed_run": FAILED_FA_PREFLIGHT_RUN,
                    "realized_ids": [0xFA000],
                    "reason": "implementation preflight consumed the three-lane CUDA smoke before its phase-benchmark adapter failed",
                },
                "roles": {
                    name: {"slot": [value, value + 0x10], "used_initially": [value]}
                    for name, value in PATH_IDS.items()
                },
                "collision_scan_sha256": collisions["semantic_sha256"],
                "passed": 1,
            }
    atomic_write_json(run_dir / "path_id_plan.json", _semantic(path_plan_body))
    atomic_write_json(run_dir / "path_collision_scan.json", collisions)
    atomic_write_json(run_dir / "parent_binding.json", parent_record)
    if continuation_binding is not None:
        atomic_write_json(run_dir / "continuation_binding.json", continuation_binding)
        atomic_write_json(
            run_dir / "fused_schedule_plan.json",
            _fused_schedule_plan(test_only=bool(args.test_only)),
        )
    input_binding = _copy_and_bind_inputs(run_dir, args)
    closure = _source_closure()
    manifest_body: dict[str, Any] = {
            "schema": _schema(args) + "-manifest",
            "schema_version": 1,
            "created_at": _now(),
            "research_mode": "exploratory",
            "objective_bearing_experiment": 1,
            "device": str(args.device),
            "frequency1_run_dir": str(args.frequency1_run_dir.resolve()),
            "source_run_dir": str(args.source_run_dir.resolve()),
            "scientific_config_sha256": config["semantic_sha256"],
            "input_binding_sha256": input_binding["semantic_sha256"],
            "source_fingerprint": closure["source_fingerprint"],
            "source_files": closure["files"],
            "source_revision": _source_revision(Path.cwd()),
            "invocation": {
                "python_executable": sys.executable,
                "argv": list(getattr(args, "invoked_argv", [])),
                "normalized_command": " ".join(
                    [str(sys.executable), "-m", "mnist.diag_d0_jacobi_rb_frequency1_rollout", *map(str, getattr(args, "invoked_argv", []))]
                ),
            },
            "test_only": int(args.test_only),
            "authorizing": 0,
            **NO_WORK,
        }
    if continuation_binding is not None:
        manifest_body.update(
            {
                "fused_continuation": 1,
                "continuation_run_dir": str(args.continuation_run_dir.resolve()),
                "continuation_binding_sha256": continuation_binding[
                    "semantic_sha256"
                ],
                "fused_schedule_plan_sha256": _load_semantic(
                    run_dir / "fused_schedule_plan.json", "fused schedule plan"
                )["semantic_sha256"],
            }
        )
    manifest = _semantic(manifest_body)
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    _status(run_dir, state="ready_for_preflight", stage="initialize")
    return config


def _verify_copied_inputs(
    run_dir: Path,
    binding: Mapping[str, Any],
    *,
    test_only: bool,
) -> None:
    """Verify copied immutable inputs before any resumed stage may write."""

    root = run_dir / "input_bindings"
    checkpoint = root / "checkpoint.pt"
    if (
        not checkpoint.is_file()
        or file_fingerprint(checkpoint) != binding.get("checkpoint_file_sha256")
    ):
        raise ArtifactCompatibilityError("copied checkpoint commitment changed")
    arrays = _load_npz(root / "source_image.npz")
    if set(arrays) != {"image", "mixed_target"}:
        raise ArtifactCompatibilityError("copied source archive schema changed")
    source = _state_row(arrays["image"])
    target = _state_row(arrays["mixed_target"])
    if (
        _array_sha256(source) != binding.get("source_image_array_sha256")
        or _array_sha256(target) != binding.get("mixed_target_array_sha256")
    ):
        raise ArtifactCompatibilityError("copied source array commitment changed")
    for relative, key in (
        ("source_image.npz", "source_image_npz_sha256"),
        ("source_image.json", "source_image_json_sha256"),
    ):
        path = root / relative
        if not path.is_file() or file_fingerprint(path) != binding.get(key):
            raise ArtifactCompatibilityError(f"copied input changed: {relative}")


def _source_arrays(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    arrays = _load_npz(run_dir / "input_bindings/source_image.npz")
    if set(arrays) != {"image", "mixed_target"}:
        raise ArtifactCompatibilityError("copied source NPZ schema changed")
    source = _state_row(arrays["image"])
    target = _state_row(arrays["mixed_target"])
    for name, value in (("source", source), ("mixed target", target)):
        if not np.isfinite(value).all() or np.any(value < 0.0):
            raise ArtifactCompatibilityError(f"{name} is nonfinite or negative")
        if abs(float(value.sum()) - 1.0) > 2.0e-15:
            raise ArtifactCompatibilityError(f"{name} simplex mass changed")
    return source, target


def _core_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _verify_restart_chain(
    root: Path,
    *,
    expected_shards: int,
    initial_state: np.ndarray,
) -> list[tuple[dict[str, Any], np.ndarray]]:
    json_paths = sorted(root.glob("shard-*.json"))
    npz_paths = sorted(root.glob("shard-*.npz"))
    if len(json_paths) != expected_shards or len(npz_paths) != expected_shards:
        raise ArtifactCompatibilityError(f"restart shard count changed under {root}")
    result: list[tuple[dict[str, Any], np.ndarray]] = []
    previous = _core_array_sha256(np.ascontiguousarray(initial_state[None, :]))
    for index, (json_path, npz_path) in enumerate(zip(json_paths, npz_paths, strict=True)):
        if json_path.stem != f"shard-{index:04d}" or npz_path.stem != f"shard-{index:04d}":
            raise ArtifactCompatibilityError("restart shard ordering changed")
        record = _load_semantic(json_path, f"restart shard {index}")
        if int(record.get("shard_index", -1)) != index or int(record.get("committed", 0)) != 1:
            raise ArtifactCompatibilityError("restart shard identity changed")
        if record.get("input_state_sha256") != previous:
            raise ArtifactCompatibilityError("restart input/output chain changed")
        if (
            record.get("state_file_sha256") != file_fingerprint(npz_path)
            or int(record.get("state_file_size", -1)) != int(npz_path.stat().st_size)
        ):
            raise ArtifactCompatibilityError("restart state file commitment changed")
        arrays = _load_npz(npz_path)
        if set(arrays) != {"state"}:
            raise ArtifactCompatibilityError("restart state schema changed")
        state = np.asarray(arrays["state"])
        if state.dtype != np.float64 or state.shape != (1, 784):
            raise ArtifactCompatibilityError("restart state shape or dtype changed")
        measured = _core_array_sha256(state)
        if record.get("output_state_sha256") != measured:
            raise ArtifactCompatibilityError("restart state array commitment changed")
        previous = measured
        result.append((record, np.array(state[0], copy=True, order="C")))
    return result


def _verify_forward_summary(run_dir: Path, role: str) -> dict[str, Any]:
    role_dir = run_dir / "forward" / role
    summary = _load_semantic(role_dir / "forward_summary.json", f"{role} forward summary")
    anchors_archive = _load_npz(role_dir / "anchors.npz")
    if "initial_mixed_target" not in anchors_archive:
        raise ArtifactCompatibilityError("forward anchor archive omitted initial state")
    anchors = {int(value) for value in summary.get("anchors", {})}
    if not anchors:
        raise ArtifactCompatibilityError("forward summary has no anchors")
    for anchor in anchors:
        key = f"step_{anchor:04d}"
        if key not in anchors_archive:
            raise ArtifactCompatibilityError("forward anchor archive is incomplete")
        if _array_sha256(_state_row(anchors_archive[key])) != summary["anchors"][str(anchor)]["state_sha256"]:
            raise ArtifactCompatibilityError("forward anchor commitment changed")
    if not int(summary.get("test_only", 0)):
        chain = _verify_restart_chain(
            role_dir / "forward_shards" / role,
            expected_shards=(max(anchors) + 1) // 8,
            initial_state=_state_row(anchors_archive["initial_mixed_target"]),
        )
        by_step = {(index + 1) * 8 - 1: state for index, (_record_value, state) in enumerate(chain)}
        for anchor in anchors:
            if _array_sha256(by_step[anchor]) != summary["anchors"][str(anchor)]["state_sha256"]:
                raise ArtifactCompatibilityError("forward restart anchor differs from summary")
    if int(summary.get("health", {}).get("passed", 0)) != 1:
        raise ArtifactCompatibilityError("forward summary health is not passing")
    return summary


def _verify_trajectory_summary(path: Path, run_dir: Path) -> dict[str, Any]:
    summary = _load_semantic(path, "reverse trajectory summary")
    state_artifact = summary.get("selected_states_artifact", {})
    if not isinstance(state_artifact, Mapping):
        raise ArtifactCompatibilityError("selected-state artifact binding is malformed")
    state_path = Path(str(state_artifact.get("path", "")))
    if not state_path.is_absolute():
        state_path = run_dir / state_path
    if (
        not state_path.is_file()
        or file_fingerprint(state_path) != state_artifact.get("sha256")
        or int(state_path.stat().st_size) != int(state_artifact.get("size", -1))
    ):
        raise ArtifactCompatibilityError("selected-state artifact commitment changed")
    states = _load_npz(state_path)
    if "start" not in states or "final" not in states:
        raise ArtifactCompatibilityError("selected-state archive is incomplete")
    if _array_sha256(_state_row(states["final"])) != summary.get("final_state_sha256"):
        raise ArtifactCompatibilityError("trajectory final-state commitment changed")
    for progress, images in summary.get("images", {}).items():
        if progress not in states or not isinstance(images, Mapping):
            raise ArtifactCompatibilityError("trajectory image/state mapping changed")
        for relative in images.values():
            if not (run_dir / str(relative)).is_file():
                raise ArtifactCompatibilityError("trajectory image artifact is missing")
    if int(summary.get("passed_integrity", 0)) != 1 or int(summary.get("health", {}).get("passed", 0)) != 1:
        raise ArtifactCompatibilityError("trajectory integrity summary is not passing")
    fused_bindings = summary.get("fused_family_binding")
    if isinstance(fused_bindings, list):
        if not fused_bindings:
            raise ArtifactCompatibilityError("fused trajectory family binding is empty")
        covered_states: set[str] = set()
        for binding in fused_bindings:
            if not isinstance(binding, Mapping):
                raise ArtifactCompatibilityError("fused trajectory binding is malformed")
            family_path = run_dir / str(binding.get("path", ""))
            family = _verify_fused_family_summary(family_path, run_dir)
            row_index = int(binding.get("row_index", -1))
            row_table = family.get("row_table", ())
            family_saved = family.get("result", {}).get("saved_state_sha256", {})
            saved_artifact = family.get("saved_states_artifact", {})
            saved_path = run_dir / str(saved_artifact.get("path", ""))
            saved_arrays = _load_npz(saved_path) if saved_path.is_file() else {}
            bound_family_saved = binding.get("family_saved_state_sha256", {})
            bound_row_saved = binding.get("row_saved_state_sha256", {})
            state_sources = binding.get("state_sources", {})
            if (
                family.get("semantic_sha256") != binding.get("sha256")
                or not 0 <= row_index < len(row_table)
                or row_table[row_index].get("row_key") != binding.get("row_key")
                or int(row_table[row_index].get("canonical_path_id", -1))
                != int(summary.get("path_id", -2))
                or not isinstance(family_saved, Mapping)
                or not isinstance(bound_family_saved, Mapping)
                or any(family_saved.get(name) != value for name, value in bound_family_saved.items())
                or not isinstance(bound_row_saved, Mapping)
                or not isinstance(state_sources, Mapping)
                or not state_sources
                or covered_states.intersection(state_sources)
                or set(bound_row_saved) != set(state_sources)
                or binding.get("family_saved_state_artifact_sha256")
                != saved_artifact.get("sha256")
            ):
                raise ArtifactCompatibilityError("fused trajectory family authority changed")
            for trajectory_name, family_name in state_sources.items():
                if (
                    trajectory_name not in states
                    or family_name not in saved_arrays
                    or family_name not in bound_family_saved
                ):
                    raise ArtifactCompatibilityError("fused trajectory state source changed")
                matrix = np.asarray(saved_arrays[family_name])
                if matrix.dtype != np.float64 or matrix.shape != (len(row_table), 784):
                    raise ArtifactCompatibilityError("fused trajectory source matrix changed")
                row_state = _state_row(matrix[row_index])
                if (
                    not np.array_equal(row_state, _state_row(states[trajectory_name]))
                    or bound_row_saved.get(trajectory_name) != _array_sha256(row_state)
                ):
                    raise ArtifactCompatibilityError("fused trajectory row association changed")
            covered_states.update(str(name) for name in state_sources)
        if covered_states != set(states):
            raise ArtifactCompatibilityError("fused trajectory state coverage changed")
    elif not int(summary.get("schema", "").endswith("test-only-reverse-trajectory-summary")):
        anchor = int(summary["anchor_step"])
        key = f"{summary['role']}-{summary['horizon']}-{summary['variant_name']}"
        chain = _verify_restart_chain(
            path.parent / "reverse_shards" / key,
            expected_shards=(anchor + 1) // 8,
            initial_state=_state_row(states["start"]),
        )
        if _array_sha256(chain[-1][1]) != summary.get("final_state_sha256"):
            raise ArtifactCompatibilityError("reverse restart chain differs from final summary")
    return summary


def _verify_evaluation_join(run_dir: Path) -> dict[str, Any]:
    """Reconstruct the exact P3-prefix/P6-suffix join from bound parents."""

    from mnist.d0_jacobi_rb_tangent_fused import (
        FusedRowSpec,
        join_fused_family_rows,
    )

    prefix_path = run_dir / "evaluation/full/fused_prefix/family_summary.json"
    prefix = _verify_fused_family_summary(prefix_path, run_dir)
    archive_path = run_dir / str(prefix["saved_states_artifact"]["path"])
    prefix_final = np.asarray(_load_npz(archive_path)["final"])
    prefix_specs = tuple(FusedRowSpec(**dict(row)) for row in prefix["row_table"])
    selection = _verify_development_selection(run_dir)
    selected_gain = float(selection["selected_gain"])
    append_rows = (
        ("short-zero", "zero", "short", None),
        ("short-learned", "learned", "short", selected_gain),
        ("short-oracle", "oracle", "short", None),
    )
    append_specs = tuple(
        FusedRowSpec(
            row_key=key,
            canonical_path_id=PATH_IDS["evaluation"],
            controller_kind=kind,
            variant=kind,
            horizon=horizon,
            gain=gain,
            controller_binding={
                "checkpoint_state_sha256": (
                    CHECKPOINT_STATE_SHA256 if kind == "learned" else None
                ),
                "target_measure_sha256": MIXED_TARGET_SHA256 if kind == "oracle" else None,
                "microsteps": MICROSTEPS,
            },
        )
        for key, kind, horizon, gain in append_rows
    )
    short_anchor = _anchor_state(run_dir, "evaluation", SHORT_ANCHOR)
    expected = join_fused_family_rows(
        prefix_final,
        prefix_specs,
        np.repeat(short_anchor[None, :], 3, axis=0),
        append_specs,
        next_coordinate=(127, 6),
        bindings={
            "prefix_family_sha256": prefix["semantic_sha256"],
            "short_anchor_sha256": _array_sha256(short_anchor),
            "development_selection_sha256": selection["semantic_sha256"],
            "rng_root_seed": REVERSE_ROOT_SEED,
            "stream_role": STREAM_ROLES["evaluation"],
            "variant_in_rng_key": 0,
        },
    )
    recorded = _load_semantic(
        run_dir / "evaluation/evaluation_family_join.json", "evaluation family join"
    )
    if recorded != dict(expected.record):
        raise ArtifactCompatibilityError("evaluation family join authority changed")
    return recorded


def _expected_fused_family_contract(path: Path, run_dir: Path) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(run_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ArtifactCompatibilityError("fused family lies outside its run") from exc
    selection_path = run_dir / "development/development_selection.json"
    selected_gain = None
    if selection_path.is_file():
        selected_gain = float(
            _load_semantic(selection_path, "development selection")["selected_gain"]
        )
    definitions: dict[str, tuple[str, str, str, int, int, list[tuple[str, str, str, float | None]]]] = {
        "development/short/fused_family/family_summary.json": (
            "development-short", "complete", "development", 16, 8_429_568,
            [
                ("development-short-zero", "zero", "short", None),
                *[
                    (f"development-short-learned-{_fused_gain_token(gain)}", "learned", "short", gain)
                    for gain in LEARNED_GAINS
                ],
                ("development-short-oracle", "oracle", "short", None),
            ],
        ),
        "evaluation/full/fused_prefix/family_summary.json": (
            "evaluation-full", "prefix-511-to-128", "evaluation", 48, 12_644_352,
            [
                ("full-zero", "zero", "full", None),
                ("full-learned", "learned", "full", selected_gain),
                ("full-oracle", "oracle", "full", None),
            ],
        ),
        "evaluation/joined_suffix/fused_family/family_summary.json": (
            "evaluation-joined", "suffix-127-to-0", "evaluation", 16, 8_429_568,
            [
                ("full-zero", "zero", "full", None),
                ("full-learned", "learned", "full", selected_gain),
                ("full-oracle", "oracle", "full", None),
                ("short-zero", "zero", "short", None),
                ("short-learned", "learned", "short", selected_gain),
                ("short-oracle", "oracle", "short", None),
            ],
        ),
        "replication/full/fused_family/family_summary.json": (
            "replication-full", "complete", "replication", 64, 11_239_424,
            [
                ("replication-full-zero", "zero", "full", None),
                ("replication-full-learned", "learned", "full", selected_gain),
            ],
        ),
    }
    if relative not in definitions:
        raise ArtifactCompatibilityError("fused family path is not in the frozen schedule")
    family_name, segment_name, role, shard_count, transition_count, rows = definitions[relative]
    if any(kind == "learned" and gain is None for _key, kind, _horizon, gain in rows):
        raise ArtifactCompatibilityError("fused family selection is unavailable")
    row_table = []
    for row_key, kind, horizon, gain in rows:
        row_table.append(
            {
                "row_key": row_key,
                "canonical_path_id": PATH_IDS[role],
                "controller_kind": kind,
                "variant": kind,
                "horizon": horizon,
                "gain": None if gain is None else float(gain),
                "controller_binding": {
                    "checkpoint_state_sha256": CHECKPOINT_STATE_SHA256 if kind == "learned" else None,
                    "target_measure_sha256": MIXED_TARGET_SHA256 if kind == "oracle" else None,
                    "microsteps": MICROSTEPS,
                },
            }
        )
    controller_binding = {
        "row_table": row_table,
        "checkpoint_state_sha256": CHECKPOINT_STATE_SHA256,
        "target_measure_sha256": MIXED_TARGET_SHA256,
        "controller_dispatch": "stable_one_row_canonical_order",
        "model_input_contract": "exact_six_fields_only",
    }
    rng_binding = {
        "root_seed": REVERSE_ROOT_SEED,
        "stream_role": STREAM_ROLES[role],
        "variant_in_rng_key": 0,
    }
    if role == "development":
        initial = np.repeat(_anchor_state(run_dir, role, SHORT_ANCHOR)[None, :], len(rows), axis=0)
        initial_hash = _core_array_sha256(initial)
        sequence = _reverse_sequence(SHORT_ANCHOR)
    elif relative.endswith("fused_prefix/family_summary.json"):
        initial = np.repeat(_anchor_state(run_dir, role, FULL_ANCHOR)[None, :], len(rows), axis=0)
        initial_hash = _core_array_sha256(initial)
        sequence = tuple(
            coordinate
            for coordinate in _reverse_sequence(FULL_ANCHOR)
            if coordinate[0] >= 128
        )
    elif relative.endswith("joined_suffix/fused_family/family_summary.json"):
        join = _verify_evaluation_join(run_dir)
        initial_hash = str(join.get("joined_state_sha256", ""))
        sequence = _reverse_sequence(SHORT_ANCHOR)
    else:
        initial = np.repeat(_anchor_state(run_dir, role, FULL_ANCHOR)[None, :], len(rows), axis=0)
        initial_hash = _core_array_sha256(initial)
        sequence = _reverse_sequence(FULL_ANCHOR)
    return {
        "family_name": family_name,
        "segment_name": segment_name,
        "role": role,
        "row_table": row_table,
        "shard_count": shard_count,
        "transition_count": transition_count,
        "controller_binding_sha256": config_fingerprint(controller_binding),
        "rng_binding_sha256": config_fingerprint(rng_binding),
        "initial_state_sha256": initial_hash,
        "sequence": tuple(sequence),
    }


def _verify_fused_family_summary(path: Path, run_dir: Path) -> dict[str, Any]:
    summary = _load_semantic(path, "fused family summary")
    expected = _expected_fused_family_contract(path, run_dir)
    row_table = summary.get("row_table")
    result = summary.get("result")
    health = summary.get("health")
    if (
        not isinstance(row_table, list)
        or not row_table
        or not isinstance(result, Mapping)
        or not isinstance(health, Mapping)
        or int(summary.get("passed", 0)) != 1
        or int(health.get("passed", 0)) != 1
        or result.get("row_table") != row_table
        or row_table != expected["row_table"]
        or summary.get("family_name") != expected["family_name"]
        or summary.get("segment_name") != expected["segment_name"]
        or int(result.get("transition_count", -1)) != expected["transition_count"]
        or int(result.get("shard_count", -1)) != expected["shard_count"]
    ):
        raise ArtifactCompatibilityError("fused family summary schema changed")
    row_keys = [str(row.get("row_key", "")) for row in row_table]
    if len(set(row_keys)) != len(row_keys):
        raise ArtifactCompatibilityError("fused family row order changed")
    saved_artifact = summary.get("saved_states_artifact", {})
    if not isinstance(saved_artifact, Mapping):
        raise ArtifactCompatibilityError("fused family saved-state binding is malformed")
    saved_path = run_dir / str(saved_artifact.get("path", ""))
    if (
        not saved_path.is_file()
        or file_fingerprint(saved_path) != saved_artifact.get("sha256")
        or int(saved_path.stat().st_size) != int(saved_artifact.get("size", -1))
    ):
        raise ArtifactCompatibilityError("fused family saved-state artifact changed")
    saved_arrays = _load_npz(saved_path)
    expected_saved = result.get("saved_state_sha256", {})
    if (
        not isinstance(expected_saved, Mapping)
        or set(saved_arrays) != set(expected_saved)
        or not {"start", "final"}.issubset(saved_arrays)
        or any(
            np.asarray(value).dtype != np.float64
            or np.asarray(value).shape != (len(row_table), 784)
            or _core_array_sha256(np.asarray(value)) != expected_saved[name]
            for name, value in saved_arrays.items()
        )
    ):
        raise ArtifactCompatibilityError("fused family saved-state matrices changed")
    shard_root = (
        path.parent
        / "fused_families"
        / str(summary.get("family_name"))
        / str(summary.get("segment_name"))
    )
    json_paths = sorted(shard_root.glob("shard-*.json"))
    npz_paths = sorted(shard_root.glob("shard-*.npz"))
    if (
        len(json_paths) != int(result.get("shard_count", -1))
        or len(npz_paths) != len(json_paths)
    ):
        raise ArtifactCompatibilityError("fused family shard count changed")
    previous: str | None = None
    final_hash: str | None = None
    transition_count = 0
    controller_hash: str | None = None
    rng_hash: str | None = None
    for index, (record_path, state_path) in enumerate(zip(json_paths, npz_paths, strict=True)):
        record = _load_semantic(record_path, "fused family shard")
        if (
            int(record.get("shard_index", -1)) != index
            or int(record.get("committed", 0)) != 1
            or record.get("row_table") != row_table
            or record.get("state_file_sha256") != file_fingerprint(state_path)
            or record.get("family_name") != expected["family_name"]
            or record.get("segment_name") != expected["segment_name"]
            or record.get("row_keys") != row_keys
            or record.get("canonical_path_ids")
            != [row["canonical_path_id"] for row in row_table]
            or int(record.get("microsteps", -1)) != MICROSTEPS
            or int(record.get("label", -1)) != 3
            or int(record.get("variant_in_rng_key", -1)) != 0
        ):
            raise ArtifactCompatibilityError("fused family shard binding changed")
        if index == 0 and record.get("input_state_sha256") != expected["initial_state_sha256"]:
            raise ArtifactCompatibilityError("fused family initial state changed")
        if previous is not None and record.get("input_state_sha256") != previous:
            raise ArtifactCompatibilityError("fused family shard chain changed")
        execution_plan = record.get("execution_plan", {})
        sequence = execution_plan.get("sequence", ()) if isinstance(execution_plan, Mapping) else ()
        expected_sequence = [
            list(item) for item in expected["sequence"][index * 56 : (index + 1) * 56]
        ]
        if (
            not isinstance(sequence, list)
            or sequence != expected_sequence
            or record.get("sequence_start") != sequence[0]
            or record.get("sequence_end") != sequence[-1]
            or record.get("sequence_sha256") != config_fingerprint(sequence)
            or int(execution_plan.get("shard_index", -1)) != index
            or int(execution_plan.get("row_count", -1)) != len(row_table)
            or int(execution_plan.get("transition_count", -1))
            != int(record.get("transition_count", -2))
            or execution_plan.get("input_state_sha256") != record.get("input_state_sha256")
        ):
            raise ArtifactCompatibilityError("fused family sequence/execution plan changed")
        _verify_fused_shard_health(
            record,
            expected_transitions=int(record.get("transition_count", -1)),
            row_count=len(row_table),
        )
        if controller_hash is None:
            controller_hash = str(record.get("controller_binding_sha256", ""))
            rng_hash = str(record.get("rng_binding_sha256", ""))
        if (
            record.get("controller_binding_sha256") != controller_hash
            or record.get("rng_binding_sha256") != rng_hash
        ):
            raise ArtifactCompatibilityError("fused family controller/RNG chain changed")
        transition_count += int(record.get("transition_count", 0))
        arrays = _load_npz(state_path)
        state = np.asarray(arrays.get("state"))
        if state.dtype != np.float64 or state.shape != (len(row_table), 784):
            raise ArtifactCompatibilityError("fused family state archive changed")
        measured = _core_array_sha256(state)
        # Core and CLI array hashes intentionally have identical dtype/shape
        # semantics; bind without trusting a filename alone.
        if record.get("output_state_sha256") != measured:
            raise ArtifactCompatibilityError("fused family output state changed")
        previous = measured
        final_hash = measured
    if final_hash != result.get("final_state_sha256"):
        raise ArtifactCompatibilityError("fused family final state changed")
    if (
        transition_count != expected["transition_count"]
        or controller_hash != expected["controller_binding_sha256"]
        or rng_hash != expected["rng_binding_sha256"]
        or result.get("diagnostics", {}).get("initial_state_sha256")
        != expected["initial_state_sha256"]
        or result.get("diagnostics", {}).get("final_state_sha256") != final_hash
        or int(summary.get("health", {}).get("transition_count", -1))
        != expected["transition_count"]
        or not isinstance(result.get("saved_state_sha256"), Mapping)
        or not {"start", "final"}.issubset(result["saved_state_sha256"])
    ):
        raise ArtifactCompatibilityError("fused family aggregate authority changed")
    return summary


def _load_model(run_dir: Path, device: torch.device, *, test_only: bool) -> Any:
    if test_only:
        from mnist.d0_jacobi_rb_tangent_rollout import ZeroTangentScoreController

        return ZeroTangentScoreController()
    from mnist.d0_jacobi_rb_boundary_tangent_frequency1_coordinate import (
        FrequencyOneCoordinateZeroBaselinePredictor,
    )
    from mnist.d0_jacobi_rb_learnability import state_dict_sha256

    payload = torch.load(
        run_dir / "input_bindings/checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(payload, Mapping)
        or int(payload.get("seed", -1)) != CHECKPOINT_SEED
        or int(payload.get("update", -1)) != CHECKPOINT_UPDATE
        or payload.get("state_sha256") != CHECKPOINT_STATE_SHA256
        or not isinstance(payload.get("state_dict"), Mapping)
    ):
        raise ArtifactCompatibilityError("copied checkpoint schema changed")
    if state_dict_sha256(payload["state_dict"]) != CHECKPOINT_STATE_SHA256:
        raise ArtifactCompatibilityError("copied checkpoint state hash changed")
    model = FrequencyOneCoordinateZeroBaselinePredictor(zero_residual=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _reverse_sequence(anchor_step: int) -> tuple[tuple[int, int], ...]:
    from mnist.d0_jacobi_rb_tangent_rollout import reverse_suffix_sequence

    return tuple(reverse_suffix_sequence(int(anchor_step)))


def _fused_controller_family(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    rows: Sequence[tuple[str, str, str, float | None]],
    canonical_path_id: int,
) -> tuple[tuple[Any, ...], Any, dict[str, Any]]:
    """Build canonical row specs and stable one-row controller dispatch."""

    from mnist.d0_jacobi_rb_tangent_fused import (
        FusedRowSpec,
        FusedTangentControllerBank,
    )
    from mnist.d0_jacobi_rb_tangent_rollout import (
        ScaledTangentScoreController,
        TargetFractionOracleController,
    )

    device = torch.device(args.device)
    _source, target = _source_arrays(run_dir)
    learned_model = (
        _load_model(run_dir, device, test_only=False)
        if any(kind == "learned" for _key, kind, _horizon, _gain in rows)
        else None
    )
    specs: list[Any] = []
    controllers: dict[str, Any] = {}
    for row_key, kind, horizon, gain in rows:
        binding = {
            "checkpoint_state_sha256": (
                CHECKPOINT_STATE_SHA256 if kind == "learned" else None
            ),
            "target_measure_sha256": MIXED_TARGET_SHA256 if kind == "oracle" else None,
            "microsteps": MICROSTEPS,
        }
        spec = FusedRowSpec(
            row_key=row_key,
            canonical_path_id=int(canonical_path_id),
            controller_kind=kind,
            variant=kind,
            horizon=horizon,
            gain=gain,
            controller_binding=binding,
        )
        specs.append(spec)
        if kind == "learned":
            controllers[row_key] = ScaledTangentScoreController(
                learned_model, float(gain)
            )
        elif kind == "oracle":
            controllers[row_key] = TargetFractionOracleController(
                target, microsteps=MICROSTEPS
            ).to(device=device)
    bank = FusedTangentControllerBank(tuple(specs), controllers)
    binding = {
        "row_table": [item.to_record() for item in specs],
        "checkpoint_state_sha256": CHECKPOINT_STATE_SHA256,
        "target_measure_sha256": MIXED_TARGET_SHA256,
        "controller_dispatch": "stable_one_row_canonical_order",
        "model_input_contract": "exact_six_fields_only",
    }
    return tuple(specs), bank, binding


def _prepared_fused_reference(
    device: torch.device,
    profile: JacobiRBCudaProfile,
) -> Any:
    from mnist.d0_jacobi_rb_cuda_deferred import (
        prepare_alpha1_rb_transition_batch_cuda_deferred,
    )

    key = f"{device}:{config_fingerprint(profile.to_dict())}"
    if key not in _PREPARED_FUSED_BACKENDS:
        started = time.perf_counter()
        _PREPARED_FUSED_BACKENDS[key] = prepare_alpha1_rb_transition_batch_cuda_deferred(
        device=device, profile=profile
        )
        _PREPARED_FUSED_ELAPSED[key] = time.perf_counter() - started
    return _PREPARED_FUSED_BACKENDS[key]


def _fused_reference_factory(
    *,
    prepared: Any,
    profile: JacobiRBCudaProfile,
    stream_role: str,
    row_chunk_size: int | None = None,
) -> Any:
    from mnist.d0_jacobi_rb_tangent_fused import (
        DeferredCertifiedFusedReference,
        prepare_deferred_reference_rng_seed_map,
    )

    seed_map_key = (id(prepared), int(REVERSE_ROOT_SEED), str(stream_role))
    if seed_map_key not in _PREPARED_FUSED_SEED_MAPS:
        _PREPARED_FUSED_SEED_MAPS[seed_map_key] = (
            prepare_deferred_reference_rng_seed_map(
                prepared_backend=prepared,
                root_seed=REVERSE_ROOT_SEED,
                stream_role=stream_role,
            )
        )
    prepared_rng_seeds = _PREPARED_FUSED_SEED_MAPS[seed_map_key]

    def factory(_shard_index: int) -> Any:
        return DeferredCertifiedFusedReference(
            profile=profile,
            root_seed=REVERSE_ROOT_SEED,
            stream_role=stream_role,
            prepared_backend=prepared,
            prepared_rng_seeds=prepared_rng_seeds,
            row_chunk_size=row_chunk_size,
        )

    return factory


def _candidate_fused_reference_factory(
    *,
    prepared: Any,
    profile: JacobiRBCudaProfile,
    stream_role: str,
    row_chunk_size: int | None = None,
) -> Any:
    from mnist.d0_jacobi_rb_tangent_fused import (
        CandidateApproximateFusedReference,
        prepare_deferred_reference_rng_seed_map,
    )

    seed_map_key = (id(prepared), int(REVERSE_ROOT_SEED), str(stream_role))
    if seed_map_key not in _PREPARED_FUSED_SEED_MAPS:
        _PREPARED_FUSED_SEED_MAPS[seed_map_key] = prepare_deferred_reference_rng_seed_map(
            prepared_backend=prepared,
            root_seed=REVERSE_ROOT_SEED,
            stream_role=stream_role,
        )
    prepared_rng_seeds = _PREPARED_FUSED_SEED_MAPS[seed_map_key]

    def factory(_shard_index: int) -> Any:
        return CandidateApproximateFusedReference(
            profile=profile,
            root_seed=REVERSE_ROOT_SEED,
            stream_role=stream_role,
            prepared_backend=prepared,
            prepared_rng_seeds=prepared_rng_seeds,
            row_chunk_size=row_chunk_size,
        )

    return factory


def _fused_family_health(
    result: Any,
    *,
    expected_transition_count: int,
    test_only: bool,
    end_to_end_elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    """Normalize authorizing family health from committed shard records."""

    records = tuple(_field(result, "shard_records", ()))
    references = [
        dict(record.get("diagnostics", {}).get("reference", {}))
        for record in records
        if isinstance(record, Mapping)
    ]
    transition_count = int(_field(result, "transition_count", 0))
    elapsed = max(
        float(_field(result, "elapsed_seconds", 0.0)),
        float(end_to_end_elapsed_seconds or 0.0),
    )
    fallback_count = sum(int(row.get("fallback_count", 0)) for row in references)
    active_count = sum(int(row.get("active_count", 0)) for row in references)
    certified_count = sum(int(row.get("certified_count", 0)) for row in references)
    fallback_seconds = math.fsum(
        float(row.get("fallback_seconds", 0.0)) for row in references
    )
    forbidden: dict[str, int] = {}
    for row in references:
        values = row.get("forbidden_counts", {})
        if isinstance(values, Mapping):
            for name, count in values.items():
                forbidden[str(name)] = forbidden.get(str(name), 0) + int(count)
    diagnostics = dict(_field(result, "diagnostics", {}))
    maximum_mass_error = float(diagnostics.get("maximum_mass_error", float("inf")))
    peak = max(
        (int(row.get("maximum_cuda_memory_allocated", 0)) for row in references),
        default=0,
    )
    total_memory = max(
        (int(row.get("total_cuda_memory_bytes", 0)) for row in references),
        default=0,
    )
    normalized = {
        "transition_count": transition_count,
        "active_count": active_count,
        "certified_count": certified_count,
        "certificate_fraction": float(
            diagnostics.get("certificate_fraction", float("nan"))
        ),
        "fallback_count": fallback_count,
        "fallback_fraction": (
            fallback_count / transition_count if transition_count else 0.0
        ),
        "fallback_seconds": fallback_seconds,
        "fallback_time_fraction": (
            fallback_seconds / elapsed if elapsed > 0.0 else 0.0
        ),
        "elapsed_seconds": elapsed,
        "transitions_per_second": (
            transition_count / elapsed if elapsed > 0.0 else float("nan")
        ),
        "maximum_simplex_mass_error": maximum_mass_error,
        "maximum_pair_mass_error": maximum_mass_error,
        "maximum_cuda_memory_allocated": peak,
        "cuda_total_memory": total_memory,
        "forbidden_counts": forbidden,
    }
    return _health_record(
        normalized,
        expected_transition_count=int(expected_transition_count),
        test_only=bool(test_only),
    )


def _aggregate_fused_controller_row(result: Any, row_index: int) -> dict[str, Any]:
    records = []
    for shard in _field(result, "shard_records", ()):
        values = shard.get("controller_diagnostics", ())
        if isinstance(values, Sequence) and row_index < len(values):
            records.append(dict(values[row_index]))
    aggregate: dict[str, Any] = {}
    keys = {key for row in records for key in row}
    for name in keys:
        values = [row[name] for row in records if name in row]
        if name.endswith("count") or name.endswith("squared_sum"):
            aggregate[name] = math.fsum(float(value) for value in values)
            if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
                aggregate[name] = int(aggregate[name])
        elif "maximum" in name:
            aggregate[name] = max(float(value) for value in values)
        elif values and all(value == values[0] for value in values):
            aggregate[name] = values[0]
    count = int(aggregate.get("score_count", 0))
    for prefix in ("unscaled_score", "scaled_score"):
        squared = float(aggregate.get(f"{prefix}_squared_sum", 0.0))
        aggregate[f"{prefix}_rms"] = math.sqrt(squared / count) if count else 0.0
    return aggregate


def _combine_fused_row_diagnostics(
    parts: Sequence[tuple[Any, int]],
) -> dict[str, Any]:
    phase_rows = [dict(_field(result, "per_row_diagnostics", ())[row]) for result, row in parts]
    aggregate: dict[str, Any] = {}
    keys = {key for value in phase_rows for key in value if key != "row_key"}
    for name in keys:
        values = [row.get(name, 0) for row in phase_rows]
        if name.endswith("count") or name in {
            "transition_count",
            "input_invalid",
            "reference_fraction_invalid",
            "score_invalid",
            "logistic_shift_invalid",
            "state_invalid",
            "mass_invalid",
            "metadata_invalid",
        }:
            aggregate[name] = sum(int(value) for value in values)
        elif "maximum" in name:
            aggregate[name] = max(float(value) for value in values)
        elif name.endswith("squared_sum"):
            aggregate[name] = math.fsum(float(value) for value in values)
    for prefix in (
        "reference_fraction_displacement",
        "control_fraction_displacement",
        "score",
        "logistic_shift",
    ):
        count = int(aggregate.get(f"{prefix}_count", 0))
        squared = float(aggregate.get(f"{prefix}_squared_sum", 0.0))
        aggregate[f"{prefix}_rms"] = math.sqrt(squared / count) if count else 0.0
    controllers = [_aggregate_fused_controller_row(result, row) for result, row in parts]
    aggregate["controller"] = _combine_plain_numeric_records(controllers)
    reference_rms = float(aggregate.get("reference_fraction_displacement_rms", 0.0))
    control_rms = float(aggregate.get("control_fraction_displacement_rms", 0.0))
    aggregate["control_reference_displacement_ratio"] = (
        control_rms / reference_rms if reference_rms > 0.0 else None
    )
    aggregate["target_oracle_unreachable_boundary_count"] = int(
        aggregate["controller"].get("target_oracle_unreachable_boundary_count", 0)
    )
    return aggregate


def _combine_plain_numeric_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    keys = {key for record in records for key in record}
    for name in keys:
        values = [record[name] for record in records if name in record]
        if name.endswith("count") or name.endswith("squared_sum"):
            result[name] = math.fsum(float(value) for value in values)
            if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
                result[name] = int(result[name])
        elif "maximum" in name:
            result[name] = max(float(value) for value in values)
        elif values and all(value == values[0] for value in values):
            result[name] = values[0]
    count = int(result.get("score_count", 0))
    for prefix in ("unscaled_score", "scaled_score"):
        squared = float(result.get(f"{prefix}_squared_sum", 0.0))
        result[f"{prefix}_rms"] = math.sqrt(squared / count) if count else 0.0
    return result


def _fused_cuda_duplicate_control(
    *, device: torch.device, profile: JacobiRBCudaProfile
) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_cuda import sample_alpha1_rb_transition_batch_cuda
    from mnist.d0_jacobi_rb_tangent_fused import FusedRowSpec, fused_transition_ids

    rows = tuple(
        FusedRowSpec(
            row_key=f"duplicate-{index}",
            canonical_path_id=FUSED_PREFLIGHT_PATH_IDS["warmup"],
            controller_kind="zero",
            variant="zero",
            horizon="short",
        )
        for index in range(3)
    )
    base = torch.linspace(0.05, 0.95, 392, dtype=torch.float64, device=device)
    head_fraction = torch.stack((base, base, torch.flip(base, dims=(0,))), dim=0).contiguous()
    exposure = torch.full_like(head_fraction, 1.0e-3)
    ids = fused_transition_ids(
        rows,
        outer_step=127,
        phase=6,
        reverse_microstep=0,
        role="reverse_reference_pre_control_M2",
        device=device,
    )
    rng_key = (REVERSE_ROOT_SEED, "frequency1-fused-duplicate-control")
    fused = sample_alpha1_rb_transition_batch_cuda(
        head_fraction,
        exposure,
        rng_key=rng_key,
        transition_ids=ids,
        profile=profile,
    )
    field_names = (
        "later_head_fraction",
        "denoising_target",
        "certificate_codes",
        "fallback_mask",
        "mode_counts",
        "prefix_bits",
    )
    singleton_equal: dict[str, bool] = {name: True for name in field_names}
    singleton_results = []
    for row in range(3):
        singleton = sample_alpha1_rb_transition_batch_cuda(
            head_fraction[row : row + 1].contiguous(),
            exposure[row : row + 1].contiguous(),
            rng_key=rng_key,
            transition_ids=ids[row : row + 1].contiguous(),
            profile=profile,
        )
        singleton_results.append(singleton)
        for name in field_names:
            fused_value = _field(fused, name)
            singleton_value = _field(singleton, name)
            if isinstance(fused_value, torch.Tensor) and isinstance(singleton_value, torch.Tensor):
                singleton_equal[name] &= bool(
                    torch.equal(fused_value[row : row + 1], singleton_value)
                )
    identical_duplicate = all(
        not isinstance(_field(fused, name), torch.Tensor)
        or torch.equal(_field(fused, name)[0], _field(fused, name)[1])
        for name in field_names
    )
    codes = _field(fused, "certificate_codes")
    certified = (
        ((codes.to(torch.uint8) & 0b1111) == 0b1111)
        if isinstance(codes, torch.Tensor)
        else torch.zeros_like(head_fraction, dtype=torch.bool)
    )
    passed = bool(
        torch.equal(ids[0], ids[1])
        and torch.equal(ids[1], ids[2])
        and identical_duplicate
        and all(singleton_equal.values())
        and bool(torch.all(certified))
    )
    return _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-duplicate-transition-id-control",
            "schema_version": 1,
            "path_id": FUSED_PREFLIGHT_PATH_IDS["warmup"],
            "row_count": 3,
            "transition_id_rows_identical": int(
                torch.equal(ids[0], ids[1]) and torch.equal(ids[1], ids[2])
            ),
            "identical_duplicate_rows_bit_identical": int(identical_duplicate),
            "fused_singleton_field_identity": singleton_equal,
            "certificate_fraction": float(certified.to(torch.float64).mean().item()),
            "passed": int(passed),
        }
    )


def _fused_cuda_equivalence_controls(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    anchor: np.ndarray,
    prepared: Any,
    profile: JacobiRBCudaProfile,
) -> dict[str, dict[str, Any]]:
    from mnist.d0_jacobi_rb_tangent_fused import (
        FusedTangentControllerBank,
        join_fused_family_rows,
        run_fused_reverse_family,
        run_fused_reverse_shard,
    )
    from mnist.d0_jacobi_rb_tangent_rollout import (
        CertifiedExploratoryReference,
        ScaledTangentScoreController,
        TargetFractionOracleController,
        ZeroTangentScoreController,
        run_reverse_shard,
    )

    device = torch.device(args.device)

    def integer_signature(
        diagnostics: Mapping[str, Any], controller: Mapping[str, Any]
    ) -> dict[str, int]:
        reference = diagnostics.get("reference", {})
        if not isinstance(reference, Mapping):
            reference = {}
        result = {
            name: int(diagnostics.get(name, 0))
            for name in (
                "transition_count",
                "boundary_fraction_count",
                "boundary_rejection_count",
                "input_invalid",
                "reference_fraction_invalid",
                "score_invalid",
                "logistic_shift_invalid",
                "state_invalid",
                "mass_invalid",
                "metadata_invalid",
                "clipping_count",
                "correction_count",
                "floor_count",
                "limiter_count",
                "projection_count",
                "renormalization_count",
                "nonfinite_count",
                "approximation_count",
            )
        }
        result.update(
            reference_transition_count=int(
                diagnostics.get("reference_transition_count", reference.get("transition_count", 0))
            ),
            reference_certified_count=int(
                diagnostics.get("reference_certified_count", reference.get("certified_count", 0))
            ),
            reference_active_count=int(
                diagnostics.get("reference_active_count", reference.get("active_count", 0))
            ),
            reference_fallback_count=int(
                diagnostics.get("reference_fallback_count", reference.get("fallback_count", 0))
            ),
            reference_unauthorized_count=int(
                diagnostics.get(
                    "reference_unauthorized_count", reference.get("unauthorized_count", 0)
                )
            ),
            reference_invalid_count=int(
                diagnostics.get("reference_invalid_count", reference.get("invalid_count", 0))
            ),
            oracle_unreachable_count=int(
                controller.get("target_oracle_unreachable_boundary_count", 0)
            ),
            score_count=int(controller.get("score_count", diagnostics.get("score_count", 0))),
        )
        return result

    def score_signature(
        diagnostics: Mapping[str, Any], controller: Mapping[str, Any]
    ) -> dict[str, int | float]:
        result: dict[str, int | float] = {}
        for name in (
            "score_count",
            "score_squared_sum",
            "score_maximum_absolute",
            "logistic_shift_count",
            "logistic_shift_squared_sum",
            "logistic_shift_maximum_absolute",
        ):
            value = diagnostics.get(name, 0)
            result[name] = int(value) if name.endswith("count") else float(value)
        for name in (
            "unscaled_score_squared_sum",
            "scaled_score_squared_sum",
            "unscaled_score_maximum_absolute",
            "scaled_score_maximum_absolute",
        ):
            if name in controller:
                result[name] = float(controller[name])
        return result
    p3_rows = [
        ("equivalence-zero", "zero", "short", None),
        (
            "equivalence-learned",
            "learned",
            "short",
            FUSED_PREFLIGHT_LEARNED_GAIN,
        ),
        ("equivalence-oracle", "oracle", "short", None),
    ]
    specs, bank, _binding = _fused_controller_family(
        run_dir,
        args,
        rows=p3_rows,
        canonical_path_id=FUSED_PREFLIGHT_PATH_IDS["reverse_p3_profile"],
    )
    state = np.repeat(anchor[None, :], 3, axis=0)
    stream = "frequency1-fused-equivalence-p3"
    phase_result = run_fused_reverse_shard(
        torch.as_tensor(state, dtype=torch.float64, device=device).contiguous(),
        ((127, 6),),
        row_specs=specs,
        controller_bank=bank,
        reference_transition=_fused_reference_factory(
            prepared=prepared, profile=profile, stream_role=stream
        )(0),
        label=3,
        microsteps=MICROSTEPS,
        device=device,
    )
    singleton_controllers = (
        ZeroTangentScoreController(),
        ScaledTangentScoreController(
            _load_model(run_dir, device, test_only=False),
            FUSED_PREFLIGHT_LEARNED_GAIN,
        ),
        TargetFractionOracleController(
            _source_arrays(run_dir)[1], microsteps=MICROSTEPS
        ).to(device=device),
    )
    singleton_phase_results = []
    for controller in singleton_controllers:
        reference = CertifiedExploratoryReference(
            profile=profile,
            root_seed=REVERSE_ROOT_SEED,
            stream_role=stream,
        )
        singleton_phase_results.append(
            run_reverse_shard(
                torch.as_tensor(anchor[None, :], dtype=torch.float64, device=device),
                ((127, 6),),
                controller=controller,
                reference_transition=reference,
                path_ids=(FUSED_PREFLIGHT_PATH_IDS["reverse_p3_profile"],),
                label=3,
                microsteps=MICROSTEPS,
            )
        )
    phase_equal = np.array_equal(
        phase_result.final_state,
        np.stack([item.final_state[0] for item in singleton_phase_results], axis=0),
    )
    fused_phase_rows = tuple(_field(phase_result, "per_row_diagnostics", ()))
    fused_phase_controllers = tuple(_field(phase_result, "controller_diagnostics", ()))
    phase_integer_equal = bool(
        len(fused_phase_rows) == len(singleton_phase_results)
        and all(
            integer_signature(fused_phase_rows[index], fused_phase_controllers[index])
            == integer_signature(item.diagnostics, item.controller_diagnostics)
            for index, item in enumerate(singleton_phase_results)
        )
    )
    phase_score_equal = bool(
        len(fused_phase_rows) == len(singleton_phase_results)
        and all(
            score_signature(fused_phase_rows[index], fused_phase_controllers[index])
            == score_signature(item.diagnostics, item.controller_diagnostics)
            for index, item in enumerate(singleton_phase_results)
        )
    )
    phase = _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-singleton-phase-equivalence",
            "schema_version": 1,
            "row_order": [item.row_key for item in specs],
            "final_states_bit_identical": int(phase_equal),
            "integer_diagnostics_equal": int(phase_integer_equal),
            "controller_scores_bit_identical": int(phase_score_equal),
            "expected_transition_count": 3 * 4 * 392,
            "passed": int(phase_equal and phase_integer_equal and phase_score_equal),
        }
    )

    p6_rows = [
        ("shard-zero", "zero", "short", None),
        *[
            (f"shard-learned-{_fused_gain_token(gain)}", "learned", "short", gain)
            for gain in LEARNED_GAINS
        ],
        ("shard-oracle", "oracle", "short", None),
    ]
    p6_specs, p6_bank, _p6_binding = _fused_controller_family(
        run_dir,
        args,
        rows=p6_rows,
        canonical_path_id=FUSED_PREFLIGHT_PATH_IDS["reverse_p6_profile"],
    )
    p6_state = np.repeat(anchor[None, :], 6, axis=0)
    sequence = _reverse_sequence(127)[: 8 * 7]
    shard_stream = "frequency1-fused-equivalence-p6"
    fused_shard = run_fused_reverse_shard(
        torch.as_tensor(p6_state, dtype=torch.float64, device=device),
        sequence,
        row_specs=p6_specs,
        controller_bank=p6_bank,
        reference_transition=_fused_reference_factory(
            prepared=prepared, profile=profile, stream_role=shard_stream
        )(0),
        label=3,
        microsteps=MICROSTEPS,
        device=device,
    )
    singleton_shard_results = []
    for spec in p6_specs:
        if spec.controller_kind == "zero":
            controller = ZeroTangentScoreController()
        elif spec.controller_kind == "oracle":
            controller = TargetFractionOracleController(
                _source_arrays(run_dir)[1], microsteps=MICROSTEPS
            ).to(device=device)
        else:
            controller = ScaledTangentScoreController(
                _load_model(run_dir, device, test_only=False), float(spec.gain)
            )
        singleton_shard_results.append(
            run_reverse_shard(
                torch.as_tensor(anchor[None, :], dtype=torch.float64, device=device),
                sequence,
                controller=controller,
                reference_transition=CertifiedExploratoryReference(
                    profile=profile,
                    root_seed=REVERSE_ROOT_SEED,
                    stream_role=shard_stream,
                ),
                path_ids=(FUSED_PREFLIGHT_PATH_IDS["reverse_p6_profile"],),
                label=3,
                microsteps=MICROSTEPS,
            )
        )
    shard_equal = np.array_equal(
        fused_shard.final_state,
        np.stack([item.final_state[0] for item in singleton_shard_results], axis=0),
    )
    fused_shard_rows = tuple(_field(fused_shard, "per_row_diagnostics", ()))
    fused_shard_controllers = tuple(_field(fused_shard, "controller_diagnostics", ()))
    shard_integer_equal = bool(
        len(fused_shard_rows) == len(singleton_shard_results)
        and all(
            integer_signature(fused_shard_rows[index], fused_shard_controllers[index])
            == integer_signature(item.diagnostics, item.controller_diagnostics)
            for index, item in enumerate(singleton_shard_results)
        )
    )
    shard_score_equal = bool(
        len(fused_shard_rows) == len(singleton_shard_results)
        and all(
            score_signature(fused_shard_rows[index], fused_shard_controllers[index])
            == score_signature(item.diagnostics, item.controller_diagnostics)
            for index, item in enumerate(singleton_shard_results)
        )
    )
    from mnist.d0_jacobi_rb_tangent_fused import build_fused_transition_id_plan

    transition_plan = build_fused_transition_id_plan(
        p6_specs, sequence, microsteps=MICROSTEPS, device="cpu"
    )
    transition_ids_rows_equal = bool(
        all(
            torch.equal(transition_plan.ids[:, :, :, 0], transition_plan.ids[:, :, :, row])
            for row in range(1, len(p6_specs))
        )
    )
    shard = _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-singleton-shard-equivalence",
            "schema_version": 1,
            "row_order": [item.row_key for item in p6_specs],
            "final_states_bit_identical": int(shard_equal),
            "integer_diagnostics_equal": int(shard_integer_equal),
            "controller_score_reductions_equal": int(shard_score_equal),
            "selected_final_capture_bit_identical": int(shard_equal),
            "transition_id_rows_bit_identical": int(transition_ids_rows_equal),
            "certificate_fraction": float(
                _field(fused_shard, "diagnostics", {}).get("certificate_fraction", 0.0)
            ),
            "transition_count": fused_shard.transition_count,
            "passed": int(
                shard_equal and shard_integer_equal and shard_score_equal and transition_ids_rows_equal
                and float(_field(fused_shard, "diagnostics", {}).get("certificate_fraction", 0.0)) == 1.0
            ),
        }
    )

    permutation = (5, 0, 4, 1, 3, 2)
    perm_specs = tuple(p6_specs[index] for index in permutation)
    perm_controllers = {
        spec.row_key: p6_bank.controllers[spec.row_key]
        for spec in perm_specs
        if spec.controller_kind != "zero"
    }
    perm_bank = FusedTangentControllerBank(perm_specs, perm_controllers)
    permuted = run_fused_reverse_shard(
        torch.as_tensor(p6_state[list(permutation)], dtype=torch.float64, device=device),
        ((127, 6),),
        row_specs=perm_specs,
        controller_bank=perm_bank,
        reference_transition=_fused_reference_factory(
            prepared=prepared, profile=profile, stream_role="frequency1-fused-permute"
        )(0),
        label=3,
        microsteps=MICROSTEPS,
        device=device,
    )
    canonical_bank_specs, canonical_bank, _ = _fused_controller_family(
        run_dir,
        args,
        rows=p6_rows,
        canonical_path_id=FUSED_PREFLIGHT_PATH_IDS["reverse_p6_profile"],
    )
    canonical = run_fused_reverse_shard(
        torch.as_tensor(p6_state, dtype=torch.float64, device=device),
        ((127, 6),),
        row_specs=canonical_bank_specs,
        controller_bank=canonical_bank,
        reference_transition=_fused_reference_factory(
            prepared=prepared, profile=profile, stream_role="frequency1-fused-permute"
        )(0),
        label=3,
        microsteps=MICROSTEPS,
        device=device,
    )
    restored = np.empty_like(permuted.final_state)
    for permuted_index, canonical_index in enumerate(permutation):
        restored[canonical_index] = permuted.final_state[permuted_index]
    chunk_specs, chunk_bank, _ = _fused_controller_family(
        run_dir,
        args,
        rows=p6_rows,
        canonical_path_id=FUSED_PREFLIGHT_PATH_IDS["reverse_p6_profile"],
    )
    chunked = run_fused_reverse_shard(
        torch.as_tensor(p6_state, dtype=torch.float64, device=device),
        ((127, 6),),
        row_specs=chunk_specs,
        controller_bank=chunk_bank,
        reference_transition=_fused_reference_factory(
            prepared=prepared,
            profile=profile,
            stream_role="frequency1-fused-permute",
            row_chunk_size=2,
        )(0),
        label=3,
        microsteps=MICROSTEPS,
        device=device,
    )
    permutation_equal = np.array_equal(restored, canonical.final_state)
    chunk_equal = np.array_equal(chunked.final_state, canonical.final_state)

    # Exercise the scientific P3-prefix -> P6-suffix join, not merely array
    # concatenation.  The independently run full and short arms use the same
    # exact transition IDs and stream as their joined counterparts.
    join_full_rows = [
        ("join-full-zero", "zero", "full", None),
        ("join-full-learned", "learned", "full", FUSED_PREFLIGHT_LEARNED_GAIN),
        ("join-full-oracle", "oracle", "full", None),
    ]
    join_short_rows = [
        ("join-short-zero", "zero", "short", None),
        ("join-short-learned", "learned", "short", FUSED_PREFLIGHT_LEARNED_GAIN),
        ("join-short-oracle", "oracle", "short", None),
    ]
    full_specs, full_bank, _ = _fused_controller_family(
        run_dir,
        args,
        rows=join_full_rows,
        canonical_path_id=FUSED_PREFLIGHT_PATH_IDS["reverse_p3_profile"],
    )
    join_stream = "frequency1-fused-join-control"
    prefix = run_fused_reverse_shard(
        torch.as_tensor(np.repeat(anchor[None, :], 3, axis=0), dtype=torch.float64, device=device),
        ((127, 6),),
        row_specs=full_specs,
        controller_bank=full_bank,
        reference_transition=_fused_reference_factory(
            prepared=prepared, profile=profile, stream_role=join_stream
        )(0),
        label=3,
        microsteps=MICROSTEPS,
        device=device,
    )
    short_anchor = np.ascontiguousarray(np.roll(anchor, 1), dtype=np.float64)
    short_specs, _short_bank, _ = _fused_controller_family(
        run_dir,
        args,
        rows=join_short_rows,
        canonical_path_id=FUSED_PREFLIGHT_PATH_IDS["reverse_p3_profile"],
    )
    append_state = np.repeat(short_anchor[None, :], 3, axis=0)
    join = join_fused_family_rows(
        prefix.final_state,
        prefix.row_specs,
        append_state,
        short_specs,
        next_coordinate=(127, 5),
        bindings={"control": "preflight-executed-join"},
    )
    joined_specs, joined_bank, _ = _fused_controller_family(
        run_dir,
        args,
        rows=(*join_full_rows, *join_short_rows),
        canonical_path_id=FUSED_PREFLIGHT_PATH_IDS["reverse_p3_profile"],
    )
    if tuple(join.row_specs) != tuple(joined_specs):
        raise RolloutCLIError("joined preflight row contract changed")
    joined = run_fused_reverse_shard(
        torch.as_tensor(join.state, dtype=torch.float64, device=device),
        ((127, 5),),
        row_specs=joined_specs,
        controller_bank=joined_bank,
        reference_transition=_fused_reference_factory(
            prepared=prepared, profile=profile, stream_role=join_stream
        )(1),
        label=3,
        microsteps=MICROSTEPS,
        device=device,
    )
    independent_full_specs, independent_full_bank, _ = _fused_controller_family(
        run_dir,
        args,
        rows=join_full_rows,
        canonical_path_id=FUSED_PREFLIGHT_PATH_IDS["reverse_p3_profile"],
    )
    independent_full = run_fused_reverse_shard(
        torch.as_tensor(np.repeat(anchor[None, :], 3, axis=0), dtype=torch.float64, device=device),
        ((127, 6), (127, 5)),
        row_specs=independent_full_specs,
        controller_bank=independent_full_bank,
        reference_transition=_fused_reference_factory(
            prepared=prepared, profile=profile, stream_role=join_stream
        )(2),
        label=3,
        microsteps=MICROSTEPS,
        device=device,
    )
    independent_short_specs, independent_short_bank, _ = _fused_controller_family(
        run_dir,
        args,
        rows=join_short_rows,
        canonical_path_id=FUSED_PREFLIGHT_PATH_IDS["reverse_p3_profile"],
    )
    independent_short = run_fused_reverse_shard(
        torch.as_tensor(append_state, dtype=torch.float64, device=device),
        ((127, 5),),
        row_specs=independent_short_specs,
        controller_bank=independent_short_bank,
        reference_transition=_fused_reference_factory(
            prepared=prepared, profile=profile, stream_role=join_stream
        )(3),
        label=3,
        microsteps=MICROSTEPS,
        device=device,
    )
    join_state_equal = bool(
        np.array_equal(joined.final_state[:3], independent_full.final_state)
        and np.array_equal(joined.final_state[3:], independent_short.final_state)
    )
    _source_values, target_values = _source_arrays(run_dir)
    joined_metrics = [
        _metrics_dict(value, target_values) for value in joined.final_state
    ]
    independent_metrics = [
        *[_metrics_dict(value, target_values) for value in independent_full.final_state],
        *[_metrics_dict(value, target_values) for value in independent_short.final_state],
    ]
    join_metrics_equal = joined_metrics == independent_metrics
    joined_plan = build_fused_transition_id_plan(
        joined_specs, ((127, 5),), microsteps=MICROSTEPS, device="cpu"
    )
    full_plan = build_fused_transition_id_plan(
        independent_full_specs,
        ((127, 6), (127, 5)),
        microsteps=MICROSTEPS,
        device="cpu",
    )
    short_plan = build_fused_transition_id_plan(
        independent_short_specs, ((127, 5),), microsteps=MICROSTEPS, device="cpu"
    )
    join_ids_equal = bool(
        torch.equal(joined_plan.ids[:, :, :, :3], full_plan.ids[1:2])
        and torch.equal(joined_plan.ids[:, :, :, 3:], short_plan.ids)
    )
    join_equal = bool(join_state_equal and join_metrics_equal and join_ids_equal)
    invariance = _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-permutation-chunk-invariance",
            "schema_version": 1,
            "row_permutation_invariant": int(permutation_equal),
            "chunk_size_two_invariant": int(chunk_equal),
            "join_preserves_prefix_and_append": int(
                np.array_equal(join.state[:3], prefix.final_state)
                and np.array_equal(join.state[3:], append_state)
            ),
            "joined_short_full_final_states_bit_identical": int(join_state_equal),
            "joined_short_full_metrics_bit_identical": int(join_metrics_equal),
            "joined_short_full_transition_ids_bit_identical": int(join_ids_equal),
            "passed": int(permutation_equal and chunk_equal and join_equal),
        }
    )

    restart_root = run_dir / "preflight/restart_identity"
    restart_rows = p3_rows
    restart_specs, restart_bank, restart_binding = _fused_controller_family(
        run_dir,
        args,
        rows=restart_rows,
        canonical_path_id=FUSED_PREFLIGHT_PATH_IDS["reverse_p3_profile"],
    )
    calls = {"count": 0}

    def restart_factory(index: int) -> Any:
        calls["count"] += 1
        return _fused_reference_factory(
            prepared=prepared,
            profile=profile,
            stream_role="frequency1-fused-restart",
        )(index)

    first = run_fused_reverse_family(
        torch.as_tensor(state, dtype=torch.float64, device=device),
        sequence=sequence,
        output_dir=restart_root,
        family_name="restart-control",
        segment_name="complete",
        row_specs=restart_specs,
        controller_bank=restart_bank,
        reference_factory=restart_factory,
        controller_binding=restart_binding,
        rng_binding={"root_seed": REVERSE_ROOT_SEED, "variant_in_rng_key": 0},
        label=3,
        microsteps=MICROSTEPS,
        device=device,
    )
    calls_after_first = calls["count"]
    second = run_fused_reverse_family(
        torch.as_tensor(state, dtype=torch.float64, device=device),
        sequence=sequence,
        output_dir=restart_root,
        family_name="restart-control",
        segment_name="complete",
        row_specs=restart_specs,
        controller_bank=restart_bank,
        reference_factory=restart_factory,
        controller_binding=restart_binding,
        rng_binding={"root_seed": REVERSE_ROOT_SEED, "variant_in_rng_key": 0},
        label=3,
        microsteps=MICROSTEPS,
        device=device,
    )
    restart_equal = bool(
        np.array_equal(first.final_state, second.final_state)
        and calls["count"] == calls_after_first
    )
    invariance_body = {key: value for key, value in invariance.items() if key != "semantic_sha256"}
    invariance_body.update(
        restart_invariant=int(restart_equal),
        restart_sampler_calls_for_committed_shards=0,
        passed=int(int(invariance["passed"]) and restart_equal),
    )
    invariance = _semantic(invariance_body)
    return {"phase": phase, "shard": shard, "invariance": invariance}


def _fused_test_preflight(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Non-authorizing orchestration fixture for the fused continuation."""

    preflight_dir = run_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    controls = {
        "duplicate_transition_id_control.json": {
            "duplicate_ids_accepted": 1,
            "identical_duplicate_rows_bit_identical": 1,
            "different_rows_match_singletons": 1,
        },
        "fused_singleton_phase_equivalence.json": {
            "zero_bit_identical": 1,
            "learned_bit_identical": 1,
            "oracle_bit_identical": 1,
        },
        "fused_singleton_shard_equivalence.json": {
            "eight_step_states_bit_identical": 1,
            "captures_bit_identical": 1,
            "transition_ids_bit_identical": 1,
        },
        "permutation_chunk_invariance.json": {
            "permutation_invariant": 1,
            "chunk_invariant": 1,
            "joined_evaluation_invariant": 1,
            "restart_invariant": 1,
        },
        "exact_cuda_smoke.json": {
            "exact_cuda_backend_exercised": 0,
            "test_only": 1,
            "certificate_fraction": 1.0,
        },
        "oracle_identity_control.json": {
            "interior_identity_test_passed": 1,
            "boundary_behavior_structural": 1,
        },
        "paired_rng_control.json": {
            "variant_in_rng_key": 0,
            "duplicate_canonical_ids_share_reference_bits": 1,
        },
    }
    control_records: dict[str, dict[str, Any]] = {}
    for filename, body in controls.items():
        record = _semantic(
            {
                "schema": FUSED_TEST_RUN_SCHEMA + "-" + filename[:-5].replace("_", "-"),
                "schema_version": 1,
                **body,
                "passed": 1,
            }
        )
        atomic_write_json(preflight_dir / filename, record)
        control_records[filename] = record
    warmup = _semantic(
        {
            "schema": FUSED_TEST_RUN_SCHEMA + "-warmup-record",
            "schema_version": 2,
            "path_id": FUSED_PREFLIGHT_PATH_IDS["warmup"],
            "timed": 0,
            "complete_eight_step_fused_unit": 1,
            "row_count": 6,
            "transition_count": FUSED_PROFILE_TRANSITIONS["reverse_p6"],
            "elapsed_seconds": 0.001,
            "output_sha256": "test-warmup",
            "passed": 1,
        }
    )
    atomic_write_json(preflight_dir / "warmup_record.json", warmup)
    repeats: dict[str, list[dict[str, Any]]] = {}
    for profile, transition_count in FUSED_PROFILE_TRANSITIONS.items():
        profile_rows = [
            {
                "profile": profile,
                "repeat": repeat,
                "elapsed_seconds": 0.01,
                "transition_count": transition_count,
                "output_sha256": f"test-{profile}-output",
                "health_passed": 1,
            }
            for repeat in range(3)
        ]
        repeats[profile] = profile_rows
        atomic_write_csv(preflight_dir / f"{profile}_repeat_metrics.csv", profile_rows)
    projection = _fused_resource_projection(repeats, test_only=True)
    atomic_write_json(preflight_dir / "fused_resource_projection.json", projection)
    # Preserve the legacy lookup used by downstream resource ledgers while
    # binding it to the fused profile projection.
    atomic_write_json(preflight_dir / "resource_projection.json", projection)
    unopened = _semantic(
        {
            "schema": FUSED_TEST_RUN_SCHEMA + "-objective-roles-unopened",
            "schema_version": 1,
            "objective_path_ids": {
                role: PATH_IDS[role]
                for role in ("development", "evaluation", "replication")
            },
            "realized_at_preflight": {
                role: 0 for role in ("development", "evaluation", "replication")
            },
            "passed": 1,
        }
    )
    atomic_write_json(preflight_dir / "objective_roles_unopened.json", unopened)
    checks = {
        "continuation_binding": int(
            _load_semantic(run_dir / "continuation_binding.json").get("passed", 0)
        )
        == 1,
        "parent_provenance": True,
        "source_target": True,
        "controller_interface": True,
        "duplicate_transition_ids": int(
            control_records["duplicate_transition_id_control.json"]["passed"]
        )
        == 1,
        "singleton_phase_equivalence": int(
            control_records["fused_singleton_phase_equivalence.json"]["passed"]
        )
        == 1,
        "singleton_shard_equivalence": int(
            control_records["fused_singleton_shard_equivalence.json"]["passed"]
        )
        == 1,
        "permutation_chunk_join_restart": int(
            control_records["permutation_chunk_invariance.json"]["passed"]
        )
        == 1,
        "oracle_identity": int(
            control_records["oracle_identity_control.json"]["passed"]
        )
        == 1,
        "paired_rng": int(control_records["paired_rng_control.json"]["passed"]) == 1,
        "exact_backend": int(control_records["exact_cuda_smoke.json"]["passed"]) == 1,
        "resource_projection": int(projection["passed"]) == 1,
        "objective_roles_unopened": int(unopened["passed"]) == 1,
    }
    gate = _semantic(
        {
            "schema": FUSED_TEST_RUN_SCHEMA + "-preflight-gate",
            "schema_version": 1,
            "gate_type": "execution_integrity",
            "downstream_action_controlled": (
                "same-process fresh development forward and objective rollout"
            ),
            "checks": checks,
            "passed": int(all(checks.values())),
            "runtime": {"test_only": 1, "device": "cpu", "passed": 1},
            "resource_projection_sha256": projection["semantic_sha256"],
            "continuation_binding_sha256": _load_semantic(
                run_dir / "continuation_binding.json"
            )["semantic_sha256"],
            "preflight_evidence_sha256": config_fingerprint(
                {
                    "controls": {
                        name: record["semantic_sha256"]
                        for name, record in control_records.items()
                    },
                    "warmup": warmup["semantic_sha256"],
                    "profile_rows": {
                        profile: [
                            {
                                **row,
                                "peak_memory_fraction": float(
                                    row.get("peak_memory_fraction", 0.0)
                                ),
                                "persisted_bytes": int(row.get("persisted_bytes", 0)),
                            }
                            for row in rows
                        ]
                        for profile, rows in repeats.items()
                    },
                }
            ),
            "same_process_continuation_required": 1,
            "scientific_evidence_complete": 0,
        }
    )
    atomic_write_json(run_dir / "preflight_gate.json", gate)
    if not int(gate["passed"]):
        raise RolloutCLIError("fused continuation test preflight failed")
    _status(run_dir, state="ready_for_forward", stage="preflight")
    return gate


_FUSED_PREFLIGHT_CONTROL_FILES = (
    "duplicate_transition_id_control.json",
    "fused_singleton_phase_equivalence.json",
    "fused_singleton_shard_equivalence.json",
    "permutation_chunk_invariance.json",
    "exact_cuda_smoke.json",
    "oracle_identity_control.json",
    "paired_rng_control.json",
)


def _read_fused_profile_csv(path: Path, profile: str) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ArtifactCompatibilityError(f"cannot read fused profile table: {path}") from exc
    if len(rows) != 3:
        raise ArtifactCompatibilityError(f"fused profile {profile} must have three rows")
    normalized: list[dict[str, Any]] = []
    for expected_repeat, row in enumerate(rows):
        if str(row.get("profile")) != profile or int(row.get("repeat", -1)) != expected_repeat:
            raise ArtifactCompatibilityError(f"fused profile {profile} ordering changed")
        normalized.append(
            {
                "profile": profile,
                "repeat": expected_repeat,
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "transition_count": int(row["transition_count"]),
                "output_sha256": str(row["output_sha256"]),
                "health_passed": int(row["health_passed"]),
                "peak_memory_fraction": float(row.get("peak_memory_fraction", 0.0)),
                "persisted_bytes": int(row.get("persisted_bytes", 0)),
            }
        )
    return normalized


def _verify_fused_forward_anchor(run_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Verify the committed FC001 forward chain and its two bound anchors."""

    summary = _load_semantic(
        run_dir / "preflight/forward_anchor/forward_anchor_summary.json",
        "fused preflight forward anchor",
    )
    _source, target = _source_arrays(run_dir)
    chain = _verify_restart_chain(
        run_dir
        / "preflight/forward_anchor/forward_shards/fused-preflight-anchor",
        expected_shards=64,
        initial_state=target,
    )
    short = chain[(SHORT_ANCHOR + 1) // 8 - 1][1]
    full = chain[(FULL_ANCHOR + 1) // 8 - 1][1]
    if (
        int(summary.get("path_id", -1))
        != FUSED_PREFLIGHT_PATH_IDS["forward_profile"]
        or summary.get("execution_role")
        != "untimed_forward_anchor_infrastructure"
        or int(summary.get("throughput_gate_applied", -1)) != 0
        or int(summary.get("passed", 0)) != 1
        or int(summary.get("health", {}).get("passed", 0)) != 1
        or int(summary.get("health", {}).get("throughput_gate_applied", -1))
        != 0
        or int(summary.get("health", {}).get("transition_count", -1))
        != 512 * 7 * 392
        or int(summary.get("health", {}).get("authorized_count", -1))
        != 512 * 7 * 392
        or summary.get("anchors")
        != {
            str(SHORT_ANCHOR): _array_sha256(short),
            str(FULL_ANCHOR): _array_sha256(full),
        }
    ):
        raise ArtifactCompatibilityError("fused preflight forward anchor changed")
    return short, full, summary


def _fused_profile_rows(profile: str) -> tuple[int, tuple[tuple[str, str, str, float | None], ...], str]:
    if profile == "reverse_p3":
        return (
            FUSED_PREFLIGHT_PATH_IDS["reverse_p3_profile"],
            (
                ("profile-zero", "zero", "short", None),
                ("profile-learned", "learned", "short", FUSED_PREFLIGHT_LEARNED_GAIN),
                ("profile-oracle", "oracle", "short", None),
            ),
            "frequency1-fused-profile-reverse_p3",
        )
    if profile == "reverse_p6":
        return (
            FUSED_PREFLIGHT_PATH_IDS["reverse_p6_profile"],
            (
                ("profile-zero", "zero", "short", None),
                *tuple(
                    (f"profile-learned-{_fused_gain_token(gain)}", "learned", "short", gain)
                    for gain in LEARNED_GAINS
                ),
                ("profile-oracle", "oracle", "short", None),
            ),
            "frequency1-fused-profile-reverse_p6",
        )
    raise ArtifactCompatibilityError("unknown fused reverse profile")


def _verify_fused_profile_summary(
    run_dir: Path,
    *,
    profile: str,
    repeat: int,
    short_anchor: np.ndarray,
) -> dict[str, Any]:
    """Reconstruct one timed profile row from its immutable attempt evidence."""

    repeat_root = run_dir / "preflight/profiles" / profile / f"repeat-{repeat}"
    summary = _load_semantic(repeat_root / "profile_summary.json", "fused profile summary")
    attempt_relative = str(summary.get("attempt_path", ""))
    root = repeat_root / attempt_relative
    if attempt_relative != "attempts/attempt-0000" or not root.is_dir():
        raise ArtifactCompatibilityError("fused profile attempt authority changed")
    if profile == "forward_p1":
        _source, target = _source_arrays(run_dir)
        chain = _verify_restart_chain(
            root / "forward_shards/forward-p1",
            expected_shards=1,
            initial_state=target,
        )
        output_hash = _array_sha256(chain[-1][1])
        transition_count = FUSED_PROFILE_TRANSITIONS[profile]
    else:
        from mnist.d0_jacobi_rb_tangent_fused import FusedRowSpec

        path_id, rows, stream = _fused_profile_rows(profile)
        specs = tuple(
            FusedRowSpec(
                row_key=key,
                canonical_path_id=path_id,
                controller_kind=kind,
                variant=kind,
                horizon=horizon,
                gain=gain,
                controller_binding={
                    "checkpoint_state_sha256": (
                        CHECKPOINT_STATE_SHA256 if kind == "learned" else None
                    ),
                    "target_measure_sha256": MIXED_TARGET_SHA256 if kind == "oracle" else None,
                    "microsteps": MICROSTEPS,
                },
            )
            for key, kind, horizon, gain in rows
        )
        controller_binding = {
            "row_table": [spec.to_record() for spec in specs],
            "checkpoint_state_sha256": CHECKPOINT_STATE_SHA256,
            "target_measure_sha256": MIXED_TARGET_SHA256,
            "controller_dispatch": "stable_one_row_canonical_order",
            "model_input_contract": "exact_six_fields_only",
        }
        prefix = _verify_fused_family_prefix(
            root
            / "fused_families"
            / profile.replace("_", "-")
            / "complete",
            initial_state=np.repeat(short_anchor[None, :], len(specs), axis=0),
            sequence=_reverse_sequence(SHORT_ANCHOR)[:56],
            row_specs=specs,
            controller_binding=controller_binding,
            rng_binding={
                "root_seed": REVERSE_ROOT_SEED,
                "stream_role": stream,
                "variant_in_rng_key": 0,
            },
            family_name=profile.replace("_", "-"),
            segment_name="complete",
        )
        if int(prefix["shard_count"]) != 1:
            raise ArtifactCompatibilityError("fused reverse profile shard count changed")
        output_hash = str(prefix["last_state_sha256"])
        transition_count = int(prefix["transition_count"])
    health = summary.get("health", {})
    expected_learned_gain: Any = None
    if profile == "reverse_p3":
        expected_learned_gain = FUSED_PREFLIGHT_LEARNED_GAIN
    elif profile == "reverse_p6":
        expected_learned_gain = list(LEARNED_GAINS)
    if (
        summary.get("profile") != profile
        or int(summary.get("repeat", -1)) != repeat
        or int(summary.get("transition_count", -1)) != transition_count
        or summary.get("output_sha256") != output_hash
        or int(summary.get("health_passed", 0)) != 1
        or (
            profile == "forward_p1"
            and summary.get("health_classification") != "passed"
        )
        or (
            profile == "forward_p1"
            and int(summary.get("throughput_gate_applied", -1)) != 1
        )
        or int(summary.get("restart_verified", 0)) != 1
        or int(health.get("passed", 0)) != 1
        or int(summary.get("persisted_bytes", -1)) != _directory_bytes(root)
        or (
            profile != "forward_p1"
            and summary.get("learned_gain") != expected_learned_gain
        )
        or int(health.get("transition_count", -1)) != transition_count
        or float(summary.get("elapsed_seconds", -1.0)) < 0.0
        or float(summary.get("committed_prefix_elapsed_seconds", -1.0)) < 0.0
        or float(summary.get("current_execution_and_restart_verification_seconds", -1.0)) < 0.0
        or float(summary.get("elapsed_seconds"))
        != float(summary.get("committed_prefix_elapsed_seconds"))
        + float(summary.get("current_execution_and_restart_verification_seconds"))
    ):
        raise ArtifactCompatibilityError("fused profile summary changed")
    return {
        "profile": profile,
        "repeat": repeat,
        "elapsed_seconds": float(summary["elapsed_seconds"]),
        "transition_count": transition_count,
        "output_sha256": output_hash,
        "health_passed": 1,
        "peak_memory_fraction": float(summary.get("peak_memory_fraction", 0.0)),
        "persisted_bytes": int(summary["persisted_bytes"]),
    }


def _verify_fused_preflight(run_dir: Path) -> dict[str, Any]:
    """Reconstruct every authorizing fused-preflight commitment on resume."""

    config = _load_semantic(run_dir / "scientific_config.json", "scientific configuration")
    if not _fused_mode(config):
        raise ArtifactCompatibilityError("fused preflight verifier received singleton run")
    binding = _load_semantic(run_dir / "continuation_binding.json", "continuation binding")
    carrier = Path(str(binding.get("carrier_run_dir", "")))
    expected_binding = _verify_continuation_carrier(
        carrier,
        test_only=bool(int(config.get("test_only", 0))),
    )
    if expected_binding != binding:
        raise ArtifactCompatibilityError("fused continuation binding changed")
    schedule = _load_semantic(run_dir / "fused_schedule_plan.json", "fused schedule plan")
    if schedule != _fused_schedule_plan(test_only=bool(int(config.get("test_only", 0)))):
        raise ArtifactCompatibilityError("fused schedule plan changed")
    controls = {
        name: _load_semantic(run_dir / "preflight" / name, name)
        for name in _FUSED_PREFLIGHT_CONTROL_FILES
    }
    if any(int(record.get("passed", 0)) != 1 for record in controls.values()):
        raise ArtifactCompatibilityError("fused preflight control is not passing")
    if bool(int(config.get("test_only", 0))):
        warmup = _load_semantic(
            run_dir / "preflight/warmup_record.json", "test fused warmup"
        )
        repeats = {
            profile: _read_fused_profile_csv(
                run_dir / "preflight" / f"{profile}_repeat_metrics.csv", profile
            )
            for profile in FUSED_PROFILE_TRANSITIONS
        }
        projection = _load_semantic(
            run_dir / "preflight/fused_resource_projection.json",
            "test fused resource projection",
        )
        if projection != _fused_resource_projection(repeats, test_only=True):
            raise ArtifactCompatibilityError("test fused resource projection changed")
        if _load_semantic(
            run_dir / "preflight/resource_projection.json", "resource projection"
        ) != projection:
            raise ArtifactCompatibilityError("test fused projection aliases differ")
        unopened = _load_semantic(
            run_dir / "preflight/objective_roles_unopened.json",
            "test fused objective firewall",
        )
        gate = _load_semantic(run_dir / "preflight_gate.json", "test fused preflight gate")
        expected_evidence = config_fingerprint(
            {
                "controls": {
                    name: record["semantic_sha256"] for name, record in controls.items()
                },
                "warmup": warmup["semantic_sha256"],
                "profile_rows": repeats,
            }
        )
        if (
            int(warmup.get("complete_eight_step_fused_unit", 0)) != 1
            or int(warmup.get("passed", 0)) != 1
            or int(unopened.get("passed", 0)) != 1
            or int(gate.get("passed", 0)) != 1
            or gate.get("resource_projection_sha256") != projection["semantic_sha256"]
            or gate.get("continuation_binding_sha256") != binding["semantic_sha256"]
            or gate.get("preflight_evidence_sha256") != expected_evidence
        ):
            raise ArtifactCompatibilityError("test fused preflight authority changed")
        return gate
    warmup = _verify_fused_warmup(run_dir)
    short_anchor, _full_anchor, forward_anchor = _verify_fused_forward_anchor(run_dir)
    csv_repeats = {
        profile: _read_fused_profile_csv(
            run_dir / "preflight" / f"{profile}_repeat_metrics.csv", profile
        )
        for profile in FUSED_PROFILE_TRANSITIONS
    }
    repeats = {
        profile: [
            _verify_fused_profile_summary(
                run_dir,
                profile=profile,
                repeat=repeat,
                short_anchor=short_anchor,
            )
            for repeat in range(3)
        ]
        for profile in FUSED_PROFILE_TRANSITIONS
    }
    if repeats != csv_repeats:
        raise ArtifactCompatibilityError("fused profile CSV differs from committed attempts")
    projection = _load_semantic(
        run_dir / "preflight/fused_resource_projection.json",
        "fused resource projection",
    )
    expected_projection = _fused_resource_projection(
        repeats,
        test_only=bool(int(config.get("test_only", 0))),
        current_persisted_bytes=int(projection.get("current_persisted_bytes", 0)),
    )
    if projection != expected_projection:
        raise ArtifactCompatibilityError("fused resource projection changed")
    legacy_projection = _load_semantic(
        run_dir / "preflight/resource_projection.json", "resource projection"
    )
    if legacy_projection != projection:
        raise ArtifactCompatibilityError("fused resource projection aliases differ")
    unopened = _load_semantic(
        run_dir / "preflight/objective_roles_unopened.json",
        "preflight objective-role firewall",
    )
    if (
        int(unopened.get("passed", 0)) != 1
        or unopened.get("objective_path_ids")
        != {role: PATH_IDS[role] for role in ("development", "evaluation", "replication")}
        or any(int(value) for value in unopened.get("realized_at_preflight", {}).values())
    ):
        raise ArtifactCompatibilityError("preflight objective-role firewall changed")
    # Before the next stage has committed authority, a supposedly completed
    # preflight may not coexist with objective evidence.  Once a later gate or
    # an interrupted objective shard exists, the immutable preflight-time
    # firewall above remains the authority and normal exact resume is allowed.
    later_stage_started = any(
        (run_dir / name).is_file()
        for name in ("forward_gate.json", "development_gate.json", "evaluation_gate.json")
    ) or any((run_dir / root).exists() for root in ("forward/development", "development"))
    if not later_stage_started and any(
        (run_dir / root).exists()
        for root in ("forward/evaluation", "evaluation", "replication")
    ):
        raise ArtifactCompatibilityError("objective role opened before forward authority")
    gate = _load_semantic(run_dir / "preflight_gate.json", "fused preflight gate")
    expected_checks = {
        "continuation_binding",
        "parent_provenance",
        "source_target",
        "controller_interface",
        "duplicate_transition_ids",
        "singleton_phase_equivalence",
        "singleton_shard_equivalence",
        "permutation_chunk_join_restart",
        "oracle_identity",
        "paired_rng",
        "exact_backend",
        "resource_projection",
        "objective_roles_unopened",
    }
    checks = gate.get("checks", {})
    if (
        not isinstance(checks, Mapping)
        or set(checks) != expected_checks
        or not all(bool(checks[name]) for name in expected_checks)
        or int(gate.get("passed", 0)) != 1
        or gate.get("resource_projection_sha256") != projection["semantic_sha256"]
        or gate.get("continuation_binding_sha256") != binding["semantic_sha256"]
        or gate.get("preflight_evidence_sha256")
        != config_fingerprint(
            {
                "controls": {
                    name: record["semantic_sha256"] for name, record in controls.items()
                },
                "warmup": warmup["semantic_sha256"],
                "forward_anchor": forward_anchor["semantic_sha256"],
                "profile_rows": repeats,
            }
        )
    ):
        raise ArtifactCompatibilityError("fused preflight gate authority changed")
    return gate


def _fused_preflight_forward_anchor(
    run_dir: Path,
    *,
    target: np.ndarray,
    device: torch.device,
    profile: JacobiRBCudaProfile,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from mnist.d0_jacobi_rb_tangent_rollout import run_forward_trajectory

    root = run_dir / "preflight/forward_anchor"
    result = run_forward_trajectory(
        torch.as_tensor(target[None, :], dtype=torch.float64, device=device).contiguous(),
        anchor_steps=(SHORT_ANCHOR, FULL_ANCHOR),
        output_dir=root,
        trajectory_name="fused-preflight-anchor",
        path_ids=(FUSED_PREFLIGHT_PATH_IDS["forward_profile"],),
        root_seed=FORWARD_ROOT_SEED,
        profile=profile,
        step_limit=512,
        device=device,
    )
    diagnostics = dict(_field(result, "diagnostics", {}))
    health = _health_record(
        diagnostics,
        expected_transition_count=512 * 7 * 392,
        test_only=False,
        enforce_throughput=False,
    )
    anchors = _field(result, "anchors", {})
    if not isinstance(anchors, Mapping) or set(anchors) != {SHORT_ANCHOR, FULL_ANCHOR}:
        raise RolloutCLIError("fused preflight forward anchors are incomplete")
    short = _state_row(anchors[SHORT_ANCHOR])
    full = _state_row(anchors[FULL_ANCHOR])
    record = _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-forward-anchor",
            "schema_version": 1,
            "path_id": FUSED_PREFLIGHT_PATH_IDS["forward_profile"],
            "anchors": {
                str(SHORT_ANCHOR): _array_sha256(short),
                str(FULL_ANCHOR): _array_sha256(full),
            },
            "execution_role": "untimed_forward_anchor_infrastructure",
            "throughput_gate_applied": 0,
            "observed_transitions_per_second": health[
                "transitions_per_second"
            ],
            "health": health,
            "passed": int(health["passed"]),
        }
    )
    atomic_write_json(root / "forward_anchor_summary.json", record)
    if not int(health["passed"]):
        raise RolloutCLIError(
            "fused preflight forward anchor health failed",
            failure_domain="execution_integrity",
            failure_code="fused_preflight_forward_health_invalid",
        )
    return short, full, record


def _fused_preflight_forward_repeat(
    run_dir: Path,
    *,
    repeat: int,
    target: np.ndarray,
    device: torch.device,
    profile: JacobiRBCudaProfile,
) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_tangent_rollout import run_forward_trajectory

    repeat_root = run_dir / "preflight/profiles/forward_p1" / f"repeat-{repeat}"
    summary_path = repeat_root / "profile_summary.json"
    existing = (
        _load_semantic(summary_path, "forward fused profile summary")
        if summary_path.is_file()
        else None
    )
    if existing is not None:
        attempt_relative = str(existing.get("attempt_path", ""))
        root = repeat_root / attempt_relative
        if not root.is_dir():
            raise ArtifactCompatibilityError("forward profile attempt is missing")
    else:
        # A repeat has exactly one frozen attempt.  An interrupted NPZ/JSON
        # tail resumes here; it is never replaced by a fresh favorable draw.
        root = repeat_root / "attempts/attempt-0000"
        attempt_relative = root.relative_to(repeat_root).as_posix()
    arguments = dict(
        anchor_steps=(7,),
        output_dir=root,
        trajectory_name="forward-p1",
        path_ids=(FUSED_PREFLIGHT_PATH_IDS["forward_profile"],),
        root_seed=FORWARD_ROOT_SEED,
        profile=profile,
        step_limit=8,
        device=device,
    )
    state = torch.as_tensor(target[None, :], dtype=torch.float64, device=device).contiguous()
    prior_elapsed = math.fsum(
        float(_load_semantic(path, "partial forward profile shard").get("elapsed_seconds", 0.0))
        for path in sorted(root.glob("forward_shards/forward-p1/shard-*.json"))
    )
    started = time.perf_counter()
    result = run_forward_trajectory(state, **arguments)
    replay = run_forward_trajectory(state, **arguments)
    current_elapsed = time.perf_counter() - started
    elapsed = prior_elapsed + current_elapsed
    output_hash = _array_sha256(_state_row(_field(result, "final_state")))
    if output_hash != _array_sha256(_state_row(_field(replay, "final_state"))):
        raise RolloutCLIError("forward profile restart output changed")
    if existing is not None:
        existing_health = existing.get("health", {})
        existing_classification = (
            _profile_health_classification(existing_health)
            if isinstance(existing_health, Mapping)
            else "integrity"
        )
        if (
            existing.get("profile") != "forward_p1"
            or int(existing.get("repeat", -1)) != int(repeat)
            or existing.get("attempt_path") != attempt_relative
            or int(existing.get("transition_count", -1))
            != FUSED_PROFILE_TRANSITIONS["forward_p1"]
            or existing.get("output_sha256") != output_hash
            or int(existing.get("health_passed", -1))
            != int(existing_classification == "passed")
            or existing.get("health_classification") != existing_classification
            or int(existing.get("throughput_gate_applied", -1)) != 1
            or int(existing.get("restart_verified", 0)) != 1
            or int(existing.get("persisted_bytes", -1)) != _directory_bytes(root)
        ):
            raise ArtifactCompatibilityError("completed forward profile changed")
        if existing_classification == "integrity":
            raise RolloutCLIError(
                "timed forward profile failed exact execution health",
                failure_domain="execution_integrity",
                failure_code="fused_forward_profile_health_invalid",
            )
        return existing
    diagnostics = dict(_field(result, "diagnostics", {}))
    diagnostics.update(
        elapsed_seconds=elapsed,
        transitions_per_second=FUSED_PROFILE_TRANSITIONS["forward_p1"] / elapsed,
    )
    health = _health_record(
        diagnostics,
        expected_transition_count=FUSED_PROFILE_TRANSITIONS["forward_p1"],
        test_only=False,
    )
    health_classification = _profile_health_classification(health)
    body = {
            "schema": FUSED_RUN_SCHEMA + "-profile-repeat",
            "schema_version": 1,
            "profile": "forward_p1",
            "repeat": int(repeat),
            "attempt_path": attempt_relative,
            "elapsed_seconds": elapsed,
            "committed_prefix_elapsed_seconds": prior_elapsed,
            "current_execution_and_restart_verification_seconds": current_elapsed,
            "transition_count": FUSED_PROFILE_TRANSITIONS["forward_p1"],
            "output_sha256": output_hash,
            "health_passed": int(health["passed"]),
            "health_classification": health_classification,
            "throughput_gate_applied": int(health["throughput_gate_applied"]),
            "peak_memory_fraction": float(health["peak_memory_fraction"]),
            "persisted_bytes": _directory_bytes(root),
            "restart_verified": 1,
            "health": health,
        }
    record = _semantic(body)
    atomic_write_json(summary_path, record)
    if health_classification == "integrity":
        raise RolloutCLIError(
            "timed forward profile failed exact execution health",
            failure_domain="execution_integrity",
            failure_code="fused_forward_profile_health_invalid",
        )
    return record


def _fused_preflight_reverse_repeat(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    profile_name: str,
    repeat: int,
    anchor: np.ndarray,
    prepared: Any,
    profile: JacobiRBCudaProfile,
) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_tangent_fused import run_fused_reverse_family

    if profile_name == "reverse_p3":
        path_id = FUSED_PREFLIGHT_PATH_IDS["reverse_p3_profile"]
        rows = (
            ("profile-zero", "zero", "short", None),
            ("profile-learned", "learned", "short", FUSED_PREFLIGHT_LEARNED_GAIN),
            ("profile-oracle", "oracle", "short", None),
        )
    elif profile_name == "reverse_p6":
        path_id = FUSED_PREFLIGHT_PATH_IDS["reverse_p6_profile"]
        rows = (
            ("profile-zero", "zero", "short", None),
            *tuple(
                (f"profile-learned-{_fused_gain_token(gain)}", "learned", "short", gain)
                for gain in LEARNED_GAINS
            ),
            ("profile-oracle", "oracle", "short", None),
        )
    else:
        raise RolloutCLIError("unknown fused reverse profile")
    specs, bank, binding = _fused_controller_family(
        run_dir, args, rows=rows, canonical_path_id=path_id
    )
    state = np.repeat(anchor[None, :], len(specs), axis=0)
    repeat_root = run_dir / "preflight/profiles" / profile_name / f"repeat-{repeat}"
    summary_path = repeat_root / "profile_summary.json"
    existing = (
        _load_semantic(summary_path, f"{profile_name} fused profile summary")
        if summary_path.is_file()
        else None
    )
    if existing is not None:
        attempt_relative = str(existing.get("attempt_path", ""))
        root = repeat_root / attempt_relative
        if not root.is_dir():
            raise ArtifactCompatibilityError("reverse profile attempt is missing")
    else:
        root = repeat_root / "attempts/attempt-0000"
        attempt_relative = root.relative_to(repeat_root).as_posix()
    stream = f"frequency1-fused-profile-{profile_name}"
    calls = {"count": 0}
    prepared_reference_factory = _fused_reference_factory(
        prepared=prepared, profile=profile, stream_role=stream
    )

    def factory(index: int) -> Any:
        calls["count"] += 1
        return prepared_reference_factory(index)

    common = dict(
        sequence=_reverse_sequence(SHORT_ANCHOR)[: 8 * 7],
        output_dir=root,
        family_name=profile_name.replace("_", "-"),
        segment_name="complete",
        row_specs=specs,
        controller_bank=bank,
        reference_factory=factory,
        controller_binding=binding,
        rng_binding={
            "root_seed": REVERSE_ROOT_SEED,
            "stream_role": stream,
            "variant_in_rng_key": 0,
        },
        label=3,
        microsteps=MICROSTEPS,
        device=torch.device(args.device),
    )
    shard_root = (
        root
        / "fused_families"
        / profile_name.replace("_", "-")
        / "complete"
    )
    prior_elapsed = math.fsum(
        float(_load_semantic(path, "partial reverse profile shard").get("elapsed_seconds", 0.0))
        for path in sorted(shard_root.glob("shard-*.json"))
    )
    started = time.perf_counter()
    result = run_fused_reverse_family(
        torch.as_tensor(state, dtype=torch.float64, device=args.device), **common
    )
    calls_after_first = calls["count"]
    replay = run_fused_reverse_family(
        torch.as_tensor(state, dtype=torch.float64, device=args.device), **common
    )
    current_elapsed = time.perf_counter() - started
    elapsed = prior_elapsed + current_elapsed
    output_hash = _core_array_sha256(_field(result, "final_state"))
    restart_verified = bool(
        output_hash == _core_array_sha256(_field(replay, "final_state"))
        and calls["count"] == calls_after_first
    )
    if existing is not None:
        if (
            existing.get("profile") != profile_name
            or int(existing.get("repeat", -1)) != int(repeat)
            or existing.get("attempt_path") != attempt_relative
            or int(existing.get("transition_count", -1))
            != FUSED_PROFILE_TRANSITIONS[profile_name]
            or existing.get("output_sha256") != output_hash
            or int(existing.get("health_passed", 0)) != 1
            or int(existing.get("restart_verified", 0)) != 1
            or not restart_verified
            or int(existing.get("persisted_bytes", -1)) != _directory_bytes(root)
        ):
            raise ArtifactCompatibilityError("completed reverse profile changed")
        return existing
    health = _fused_family_health(
        result,
        expected_transition_count=FUSED_PROFILE_TRANSITIONS[profile_name],
        test_only=False,
        end_to_end_elapsed_seconds=elapsed,
    )
    record = _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-profile-repeat",
            "schema_version": 1,
            "profile": profile_name,
            "repeat": int(repeat),
            "attempt_path": attempt_relative,
            "learned_gain": (
                FUSED_PREFLIGHT_LEARNED_GAIN if profile_name == "reverse_p3" else list(LEARNED_GAINS)
            ),
            "elapsed_seconds": elapsed,
            "committed_prefix_elapsed_seconds": prior_elapsed,
            "current_execution_and_restart_verification_seconds": current_elapsed,
            "transition_count": FUSED_PROFILE_TRANSITIONS[profile_name],
            "output_sha256": output_hash,
            "health_passed": int(health["passed"] and restart_verified),
            "peak_memory_fraction": float(health["peak_memory_fraction"]),
            "persisted_bytes": _directory_bytes(root),
            "restart_verified": int(restart_verified),
            "health": health,
        }
    )
    atomic_write_json(summary_path, record)
    return record


def _verify_fused_warmup(run_dir: Path) -> dict[str, Any]:
    """Verify the one frozen, complete, untimed FC000 fused warm-up unit."""

    record = _load_semantic(run_dir / "preflight/warmup_record.json", "fused warmup")
    root = run_dir / "preflight/warmup_unit/fused_families/warmup-p6/complete"
    shard_path = root / "shard-0000.json"
    state_path = root / "shard-0000.npz"
    if not shard_path.is_file() or not state_path.is_file():
        raise ArtifactCompatibilityError("fused warmup shard is incomplete")
    shard = _load_semantic(shard_path, "fused warmup shard")
    arrays = _load_npz(state_path)
    state = np.asarray(arrays.get("state"))
    expected_keys = [
        "warmup-zero",
        *[f"warmup-learned-{_fused_gain_token(gain)}" for gain in LEARNED_GAINS],
        "warmup-oracle",
    ]
    if (
        int(record.get("path_id", -1)) != FUSED_PREFLIGHT_PATH_IDS["warmup"]
        or int(record.get("timed", -1)) != 0
        or int(record.get("complete_eight_step_fused_unit", 0)) != 1
        or int(record.get("transition_count", -1))
        != FUSED_PROFILE_TRANSITIONS["reverse_p6"]
        or int(record.get("row_count", -1)) != 6
        or int(record.get("passed", 0)) != 1
        or int(record.get("health", {}).get("passed", 0)) != 1
        or record.get("shard_semantic_sha256") != shard.get("semantic_sha256")
        or record.get("shard_file_sha256") != file_fingerprint(shard_path)
        or shard.get("state_file_sha256") != file_fingerprint(state_path)
        or shard.get("row_keys") != expected_keys
        or shard.get("canonical_path_ids")
        != [FUSED_PREFLIGHT_PATH_IDS["warmup"]] * 6
        or int(shard.get("transition_count", -1))
        != FUSED_PROFILE_TRANSITIONS["reverse_p6"]
        or state.dtype != np.float64
        or state.shape != (6, 784)
        or record.get("output_sha256") != _core_array_sha256(state)
    ):
        raise ArtifactCompatibilityError("fused warmup commitment changed")
    _verify_fused_shard_health(
        shard,
        expected_transitions=FUSED_PROFILE_TRANSITIONS["reverse_p6"],
        row_count=6,
    )
    return record


def _fused_preflight_warmup(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    anchor: np.ndarray,
    prepared: Any,
    profile: JacobiRBCudaProfile,
    backend_prepare_seconds: float,
) -> dict[str, Any]:
    """Execute one complete P6 eight-step fused unit before timed profiles."""

    record_path = run_dir / "preflight/warmup_record.json"
    if record_path.is_file():
        return _verify_fused_warmup(run_dir)
    from mnist.d0_jacobi_rb_tangent_fused import run_fused_reverse_family

    rows = (
        ("warmup-zero", "zero", "short", None),
        *tuple(
            (f"warmup-learned-{_fused_gain_token(gain)}", "learned", "short", gain)
            for gain in LEARNED_GAINS
        ),
        ("warmup-oracle", "oracle", "short", None),
    )
    specs, bank, binding = _fused_controller_family(
        run_dir,
        args,
        rows=rows,
        canonical_path_id=FUSED_PREFLIGHT_PATH_IDS["warmup"],
    )
    state = np.repeat(np.asarray(anchor, dtype=np.float64)[None, :], 6, axis=0)
    stream = "frequency1-fused-warmup-p6"
    reference_factory = _fused_reference_factory(
        prepared=prepared,
        profile=profile,
        stream_role=stream,
    )
    started = time.perf_counter()
    result = run_fused_reverse_family(
        torch.as_tensor(state, dtype=torch.float64, device=args.device).contiguous(),
        sequence=_reverse_sequence(SHORT_ANCHOR)[: 8 * 7],
        output_dir=run_dir / "preflight/warmup_unit",
        family_name="warmup-p6",
        segment_name="complete",
        row_specs=specs,
        controller_bank=bank,
        reference_factory=reference_factory,
        controller_binding=binding,
        rng_binding={
            "root_seed": REVERSE_ROOT_SEED,
            "stream_role": stream,
            "variant_in_rng_key": 0,
        },
        label=3,
        microsteps=MICROSTEPS,
        device=torch.device(args.device),
    )
    unit_elapsed = time.perf_counter() - started
    health = _fused_family_health(
        result,
        expected_transition_count=FUSED_PROFILE_TRANSITIONS["reverse_p6"],
        test_only=False,
        end_to_end_elapsed_seconds=unit_elapsed,
    )
    shard_path = (
        run_dir
        / "preflight/warmup_unit/fused_families/warmup-p6/complete/shard-0000.json"
    )
    shard = _load_semantic(shard_path, "fused warmup shard")
    record = _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-warmup-record",
            "schema_version": 2,
            "path_id": FUSED_PREFLIGHT_PATH_IDS["warmup"],
            "timed": 0,
            "complete_eight_step_fused_unit": 1,
            "row_count": 6,
            "transition_count": FUSED_PROFILE_TRANSITIONS["reverse_p6"],
            "backend_prepare_seconds": float(backend_prepare_seconds),
            "complete_unit_elapsed_seconds": unit_elapsed,
            "elapsed_seconds": float(backend_prepare_seconds) + unit_elapsed,
            "prepared_backend_process_cached": 1,
            "output_sha256": _core_array_sha256(_field(result, "final_state")),
            "shard_semantic_sha256": shard["semantic_sha256"],
            "shard_file_sha256": file_fingerprint(shard_path),
            "health": health,
            "passed": int(health["passed"]),
        }
    )
    atomic_write_json(record_path, record)
    return _verify_fused_warmup(run_dir)


def _fused_cuda_preflight(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = _load_semantic(run_dir / "scientific_config.json", "scientific configuration")
    source, target = _source_arrays(run_dir)
    parent = _verify_parent_registry(args.frequency1_run_dir, test_only=False)
    binding = _load_semantic(run_dir / "continuation_binding.json", "continuation binding")
    if binding != _verify_continuation_carrier(args.continuation_run_dir, test_only=False):
        raise ArtifactCompatibilityError("continuation carrier changed before CUDA preflight")
    input_binding = _load_semantic(
        run_dir / "input_bindings/input_binding.json", "input binding"
    )
    for relative, expected in (
        ("checkpoint.pt", CHECKPOINT_FILE_SHA256),
        ("source_image.npz", input_binding["source_image_npz_sha256"]),
        ("source_image.json", input_binding["source_image_json_sha256"]),
    ):
        if file_fingerprint(run_dir / "input_bindings" / relative) != expected:
            raise ArtifactCompatibilityError(f"copied input changed: {relative}")
    device = torch.device(args.device)
    runtime = configure_exact_torch_backend(device)
    model = _load_model(run_dir, device, test_only=False)
    from mnist.d0_jacobi_rb_tangent_rollout import (
        TargetFractionOracleController,
        ZeroTangentScoreController,
        exploratory_reference_rng_key,
        target_oracle_identity_control,
    )

    zero = ZeroTangentScoreController()
    oracle = TargetFractionOracleController(target, microsteps=MICROSTEPS).to(device=device)
    interface = _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-controller-interface-preflight",
            "schema_version": 1,
            "model_has_score_prediction": int(callable(getattr(model, "score_prediction", None))),
            "zero_has_score_prediction": int(callable(getattr(zero, "score_prediction", None))),
            "oracle_has_score_prediction": int(callable(getattr(oracle, "score_prediction", None))),
            "checkpoint_parameters_frozen": int(
                not any(bool(parameter.requires_grad) for parameter in model.parameters())
            ),
            "passed": int(
                callable(getattr(model, "score_prediction", None))
                and callable(getattr(zero, "score_prediction", None))
                and callable(getattr(oracle, "score_prediction", None))
            ),
        }
    )
    atomic_write_json(run_dir / "preflight/controller_interface.json", interface)
    identity = target_oracle_identity_control(microsteps=MICROSTEPS)
    oracle_identity = _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-oracle-identity-control",
            "schema_version": 1,
            "identity_control": identity,
            "interior_identity_test_passed": int(identity.get("passed", 0)),
            "boundary_behavior_structural": int(
                identity.get("boundary_score_zero", 0)
                and identity.get("boundary_fraction_unchanged", 0)
            ),
            "passed": int(identity.get("passed", 0)),
        }
    )
    atomic_write_json(run_dir / "preflight/oracle_identity_control.json", oracle_identity)
    rng_keys = {
        role: exploratory_reference_rng_key(REVERSE_ROOT_SEED, stream, role="reference")
        for role, stream in STREAM_ROLES.items()
    }
    paired_rng = _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-paired-rng-control",
            "schema_version": 1,
            "keys_by_role": {name: config_fingerprint(value) for name, value in rng_keys.items()},
            "variant_in_rng_key": 0,
            "duplicate_canonical_ids_share_reference_bits": 1,
            "passed": 1,
        }
    )
    atomic_write_json(run_dir / "preflight/paired_rng_control.json", paired_rng)

    profile = JacobiRBCudaProfile()
    prepare_started = time.perf_counter()
    prepared = _prepared_fused_reference(device, profile)
    prepare_elapsed = time.perf_counter() - prepare_started
    duplicate_path = run_dir / "preflight/duplicate_transition_id_control.json"
    if duplicate_path.is_file():
        duplicate = _load_semantic(duplicate_path, "duplicate transition-ID control")
        if int(duplicate.get("passed", 0)) != 1:
            raise ArtifactCompatibilityError("duplicate transition-ID control is not passing")
    else:
        duplicate = _fused_cuda_duplicate_control(device=device, profile=profile)
        atomic_write_json(duplicate_path, duplicate)
    smoke_candidate = _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-exact-cuda-smoke",
            "schema_version": 1,
            "path_id": FUSED_PREFLIGHT_PATH_IDS["warmup"],
            "exact_cuda_backend_exercised": 1,
            "certificate_fraction": duplicate["certificate_fraction"],
            "output_sha256": duplicate["semantic_sha256"],
            "passed": int(duplicate["passed"]),
        }
    )
    smoke_path = run_dir / "preflight/exact_cuda_smoke.json"
    if smoke_path.is_file():
        smoke = _load_semantic(smoke_path, "exact CUDA smoke")
        if smoke != smoke_candidate:
            raise ArtifactCompatibilityError("exact CUDA smoke changed")
    else:
        smoke = smoke_candidate
        atomic_write_json(smoke_path, smoke)
    short_anchor, _full_anchor, anchor_record = _fused_preflight_forward_anchor(
        run_dir, target=target, device=device, profile=profile
    )
    warmup = _fused_preflight_warmup(
        run_dir,
        args,
        anchor=short_anchor,
        prepared=prepared,
        profile=profile,
        backend_prepare_seconds=prepare_elapsed,
    )
    control_paths = {
        "phase": run_dir / "preflight/fused_singleton_phase_equivalence.json",
        "shard": run_dir / "preflight/fused_singleton_shard_equivalence.json",
        "invariance": run_dir / "preflight/permutation_chunk_invariance.json",
    }
    if all(path.is_file() for path in control_paths.values()):
        controls = {
            name: _load_semantic(path, f"fused {name} equivalence control")
            for name, path in control_paths.items()
        }
        if any(int(record.get("passed", 0)) != 1 for record in controls.values()):
            raise ArtifactCompatibilityError("saved fused equivalence control is not passing")
    else:
        measured_controls = _fused_cuda_equivalence_controls(
            run_dir,
            args,
            anchor=short_anchor,
            prepared=prepared,
            profile=profile,
        )
        controls = {}
        for name, path in control_paths.items():
            if path.is_file():
                existing = _load_semantic(path, f"fused {name} equivalence control")
                if existing != measured_controls[name]:
                    raise ArtifactCompatibilityError("partial fused equivalence control changed")
                controls[name] = existing
            else:
                controls[name] = measured_controls[name]
                atomic_write_json(path, controls[name])

    repeats: dict[str, list[dict[str, Any]]] = {name: [] for name in FUSED_PROFILE_TRANSITIONS}
    for repeat in range(3):
        repeats["forward_p1"].append(
            _fused_preflight_forward_repeat(
                run_dir,
                repeat=repeat,
                target=target,
                device=device,
                profile=profile,
            )
        )
        for profile_name in ("reverse_p3", "reverse_p6"):
            repeats[profile_name].append(
                _fused_preflight_reverse_repeat(
                    run_dir,
                    args,
                    profile_name=profile_name,
                    repeat=repeat,
                    anchor=short_anchor,
                    prepared=prepared,
                    profile=profile,
                )
            )
    for profile_name, rows in repeats.items():
        atomic_write_csv(
            run_dir / "preflight" / f"{profile_name}_repeat_metrics.csv",
            [
                {
                    key: row[key]
                    for key in (
                        "profile",
                        "repeat",
                        "elapsed_seconds",
                        "transition_count",
                        "output_sha256",
                        "health_passed",
                        "peak_memory_fraction",
                        "persisted_bytes",
                    )
                }
                for row in rows
            ],
        )
    projection = _fused_resource_projection(
        repeats,
        current_persisted_bytes=_directory_bytes(run_dir),
    )
    atomic_write_json(run_dir / "preflight/fused_resource_projection.json", projection)
    atomic_write_json(run_dir / "preflight/resource_projection.json", projection)
    realized = {
        role: int(any((run_dir / root).exists() for root in (f"forward/{role}", role)))
        for role in ("development", "evaluation", "replication")
    }
    unopened = _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-objective-roles-unopened",
            "schema_version": 1,
            "objective_path_ids": {
                role: PATH_IDS[role]
                for role in ("development", "evaluation", "replication")
            },
            "realized_at_preflight": realized,
            "passed": int(not any(realized.values())),
        }
    )
    atomic_write_json(run_dir / "preflight/objective_roles_unopened.json", unopened)
    checks = {
        "continuation_binding": int(binding.get("passed", 0)) == 1,
        "parent_provenance": int(parent.get("passed", 0)) == 1,
        "source_target": (
            _array_sha256(source) == input_binding["source_image_array_sha256"]
            and _array_sha256(target) == input_binding["mixed_target_array_sha256"]
        ),
        "controller_interface": int(interface["passed"]) == 1,
        "duplicate_transition_ids": int(duplicate["passed"]) == 1,
        "singleton_phase_equivalence": int(controls["phase"]["passed"]) == 1,
        "singleton_shard_equivalence": int(controls["shard"]["passed"]) == 1,
        "permutation_chunk_join_restart": int(controls["invariance"]["passed"]) == 1,
        "oracle_identity": int(oracle_identity["passed"]) == 1,
        "paired_rng": int(paired_rng["passed"]) == 1,
        "exact_backend": (
            int(smoke["passed"]) == 1
            and int(anchor_record["passed"]) == 1
            and int(warmup["passed"]) == 1
        ),
        "resource_projection": int(projection["passed"]) == 1,
        "objective_roles_unopened": int(unopened["passed"]) == 1,
    }
    gate = _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-preflight-gate",
            "schema_version": 1,
            "gate_type": "execution_integrity_and_resource_budget",
            "downstream_action_controlled": "same-process fresh objective rollout",
            "checks": checks,
            "passed": int(all(checks.values())),
            "runtime": runtime,
            "resource_projection_sha256": projection["semantic_sha256"],
            "continuation_binding_sha256": binding["semantic_sha256"],
            "preflight_evidence_sha256": config_fingerprint(
                {
                    "controls": {
                        "duplicate_transition_id_control.json": duplicate[
                            "semantic_sha256"
                        ],
                        "fused_singleton_phase_equivalence.json": controls[
                            "phase"
                        ]["semantic_sha256"],
                        "fused_singleton_shard_equivalence.json": controls[
                            "shard"
                        ]["semantic_sha256"],
                        "permutation_chunk_invariance.json": controls[
                            "invariance"
                        ]["semantic_sha256"],
                        "exact_cuda_smoke.json": smoke["semantic_sha256"],
                        "oracle_identity_control.json": oracle_identity[
                            "semantic_sha256"
                        ],
                        "paired_rng_control.json": paired_rng["semantic_sha256"],
                    },
                    "warmup": warmup["semantic_sha256"],
                    "forward_anchor": anchor_record["semantic_sha256"],
                    "profile_rows": repeats,
                }
            ),
            "same_process_continuation_required": 1,
            "scientific_evidence_complete": 0,
        }
    )
    atomic_write_json(run_dir / "preflight_gate.json", gate)
    if not int(gate["passed"]):
        if not int(projection["passed"]):
            raise RolloutCLIError(
                "exact fused laptop schedule exceeds the frozen six-hour contract",
                failure_domain="resource_budget",
                failure_code="rollout_main_workflow_computationally_infeasible",
            )
        raise RolloutCLIError(
            "fused scheduler equivalence or integrity preflight failed",
            failure_domain="execution_integrity",
            failure_code="fused_scheduler_equivalence_invalid",
        )
    _verify_fused_preflight(run_dir)
    _status(run_dir, state="ready_for_forward", stage="preflight")
    return gate


def _preflight_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if _fused_mode(args):
        if args.test_only:
            return _fused_test_preflight(run_dir, args)
        return _fused_cuda_preflight(run_dir, args)
    config = _load_semantic(run_dir / "scientific_config.json", "scientific configuration")
    source, target = _source_arrays(run_dir)
    parent = _verify_parent_registry(args.frequency1_run_dir, test_only=args.test_only)
    input_binding = _load_semantic(run_dir / "input_bindings/input_binding.json", "input binding")
    if not args.test_only:
        for relative, expected in (
            ("checkpoint.pt", CHECKPOINT_FILE_SHA256),
            ("source_image.npz", input_binding["source_image_npz_sha256"]),
            ("source_image.json", input_binding["source_image_json_sha256"]),
        ):
            if file_fingerprint(run_dir / "input_bindings" / relative) != expected:
                raise ArtifactCompatibilityError(f"copied input changed: {relative}")

    device = torch.device(args.device)
    runtime = (
        {"test_only": 1, "device": "cpu", "passed": 1}
        if args.test_only
        else configure_exact_torch_backend(device)
    )
    model = _load_model(run_dir, device, test_only=args.test_only)
    from mnist.d0_jacobi_rb_tangent_rollout import (
        TargetFractionOracleController,
        ZeroTangentScoreController,
        exploratory_reference_rng_key,
        target_oracle_identity_control,
    )

    zero = ZeroTangentScoreController()
    oracle = TargetFractionOracleController(target, microsteps=MICROSTEPS)
    interface = _semantic(
        {
            "schema": _schema(config) + "-controller-interface-preflight",
            "schema_version": 1,
            "model_has_score_prediction": int(callable(getattr(model, "score_prediction", None))),
            "zero_has_score_prediction": int(callable(getattr(zero, "score_prediction", None))),
            "oracle_has_score_prediction": int(callable(getattr(oracle, "score_prediction", None))),
            "frequency1_checkpoint_strict_loaded": 1,
            "checkpoint_parameters_frozen": int(
                not any(bool(parameter.requires_grad) for parameter in model.parameters())
                if hasattr(model, "parameters") else 1
            ),
            "passed": int(
                callable(getattr(model, "score_prediction", None))
                and callable(getattr(zero, "score_prediction", None))
                and callable(getattr(oracle, "score_prediction", None))
            ),
        }
    )
    atomic_write_json(run_dir / "preflight/controller_interface.json", interface)

    # This algebraic control is implemented by the reusable core and does not
    # consume a production path or CUDA RNG stream.
    oracle_record = _record(oracle.record())
    identity_record = target_oracle_identity_control(microsteps=MICROSTEPS)
    oracle_identity = _semantic(
        {
            "schema": _schema(config) + "-oracle-identity-control",
            "schema_version": 1,
            "controller_record": oracle_record,
            "identity_control": identity_record,
            "interior_identity_test_passed": int(identity_record.get("passed", 0)),
            "boundary_behavior_structural": int(
                identity_record.get("boundary_score_zero", 0)
                and identity_record.get("boundary_fraction_unchanged", 0)
            ),
            "passed": int(identity_record.get("passed", 0)),
        }
    )
    atomic_write_json(run_dir / "preflight/oracle_identity_control.json", oracle_identity)

    rng_keys = {
        role: exploratory_reference_rng_key(REVERSE_ROOT_SEED, stream, role="reference")
        for role, stream in STREAM_ROLES.items()
    }
    rng_record = _semantic(
        {
            "schema": _schema(config) + "-paired-rng-control",
            "schema_version": 1,
            "keys_by_role": {name: config_fingerprint(value) for name, value in rng_keys.items()},
            "variant_name_in_key": 0,
            "development_evaluation_distinct": int(
                config_fingerprint(rng_keys["development"])
                != config_fingerprint(rng_keys["evaluation"])
            ),
            "passed": 1,
        }
    )
    atomic_write_json(run_dir / "preflight/paired_rng_control.json", rng_record)

    if args.test_only:
        smoke = {
            "test_only": 1,
            "exact_cuda_backend_exercised": 0,
            "certificate_fraction": 1.0,
            "maximum_mass_error": 0.0,
            "passed": 1,
        }
    else:
        from mnist.d0_jacobi_rb_cuda import sample_alpha1_rb_transition_batch_cuda

        x = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64, device=device)
        u = torch.full_like(x, 1.0e-3)
        ids = torch.tensor(
            [PATH_IDS["preflight"] * (1 << 23) + index for index in range(3)],
            dtype=torch.uint64,
            device=device,
        )
        batch = sample_alpha1_rb_transition_batch_cuda(
            x,
            u,
            rng_key=(REVERSE_ROOT_SEED, "frequency1-rollout-preflight-v1"),
            transition_ids=ids,
            profile=JacobiRBCudaProfile(),
        )
        codes = batch.certificate_codes.detach().cpu().numpy()
        certified = (codes.astype(np.uint8) & np.uint8(0b1111)) == np.uint8(0b1111)
        later = batch.later_head_fraction.detach().cpu().numpy()
        target_values = batch.denoising_target.detach().cpu().numpy()
        smoke = {
            "test_only": 0,
            "exact_cuda_backend_exercised": 1,
            "transition_count": 3,
            "certificate_fraction": float(certified.mean()),
            "finite_outputs": int(np.isfinite(later).all() and np.isfinite(target_values).all()),
            "passed": int(certified.all() and np.isfinite(later).all() and np.isfinite(target_values).all()),
        }
    smoke = _semantic(
        {
            "schema": _schema(config) + "-exact-cuda-smoke",
            "schema_version": 1,
            **smoke,
        }
    )
    atomic_write_json(run_dir / "preflight/exact_cuda_smoke.json", smoke)

    if args.test_only:
        benchmark = {
            "test_only": 1,
            "repeat_elapsed_seconds": [0.001, 0.001, 0.001],
            "repeat_transition_counts": [1_568, 1_568, 1_568],
            "slowest_complete_repeat_rate": 1_568_000.0,
            "projected_main_transition_count": MAIN_WORKFLOW_TRANSITIONS,
            "projected_main_wall_seconds": MAIN_WORKFLOW_TRANSITIONS / 1_568_000.0,
            "maximum_main_wall_seconds": MAXIMUM_MAIN_WALL_SECONDS,
            "passed": 1,
        }
    else:
        from mnist.d0_jacobi_rb_tangent_rollout import (
            CertifiedExploratoryReference,
            ZeroTangentScoreController,
            benchmark_tangent_phase,
        )

        state_tensor = torch.as_tensor(
            target[None, :], dtype=torch.float64, device=device
        ).contiguous()
        rates: list[float] = []
        elapsed_values: list[float] = []
        transition_counts: list[int] = []
        benchmark_records: list[dict[str, Any]] = []
        def benchmark_reference_factory(repeat: int) -> Any:
            return CertifiedExploratoryReference(
                profile=JacobiRBCudaProfile(),
                root_seed=REVERSE_ROOT_SEED,
                stream_role=f"frequency1-rollout-preflight-benchmark-{repeat}",
            )

        measured = benchmark_tangent_phase(
            state_tensor,
            controller=ZeroTangentScoreController(),
            path_ids=(PATH_IDS["preflight"],),
            outer_step=127,
            # A one-phase reverse shard must begin at phase 6.  Phase 6 is
            # the same frozen H0/2 duration/color as phase 0, so it provides
            # the intended complete-phase performance profile without
            # weakening the reverse-shard contiguity contract.
            phase=6,
            reference_factory=benchmark_reference_factory,
            label=3,
            microsteps=MICROSTEPS,
            repeats=3,
        )
        measured_record = _record(measured)
        for record in measured_record["repeats"]:
            elapsed = float(record["elapsed_seconds"])
            count = int(record["transition_count"])
            rate = float(record.get("transitions_per_second", count / elapsed))
            if elapsed <= 0.0 or count != 4 * 392 or not math.isfinite(rate):
                raise RolloutCLIError(
                    "complete tangent-phase resource benchmark is malformed",
                    failure_domain="resource_budget",
                    failure_code="rollout_resource_projection_invalid",
                )
            elapsed_values.append(elapsed)
            transition_counts.append(count)
            rates.append(rate)
            benchmark_records.append(dict(record))
        slowest_rate = min(rates)
        projected_seconds = MAIN_WORKFLOW_TRANSITIONS / slowest_rate
        benchmark = {
            "test_only": 0,
            "repeat_elapsed_seconds": elapsed_values,
            "repeat_transition_counts": transition_counts,
            "repeat_rates": rates,
            "repeat_records": benchmark_records,
            "slowest_complete_repeat_rate": slowest_rate,
            "projected_main_transition_count": MAIN_WORKFLOW_TRANSITIONS,
            "projected_main_wall_seconds": projected_seconds,
            "maximum_main_wall_seconds": MAXIMUM_MAIN_WALL_SECONDS,
            "minimum_profile_rate": MINIMUM_TRANSITIONS_PER_SECOND,
            "passed": int(
                slowest_rate >= MINIMUM_TRANSITIONS_PER_SECOND
                and projected_seconds <= MAXIMUM_MAIN_WALL_SECONDS
            ),
        }
    benchmark = _semantic(
        {
            "schema": _schema(config) + "-resource-projection",
            "schema_version": 1,
            "gate_type": "execution_integrity_resource_budget",
            "scientific_decision_purchased": "fresh paired objective-bearing rollout",
            "why_smaller_experiment_is_insufficient": (
                "the 128-step suffix alone cannot reveal long-horizon on-policy accumulation"
            ),
            **benchmark,
        }
    )
    atomic_write_json(run_dir / "preflight/resource_projection.json", benchmark)
    checks = {
        "parent_provenance": int(parent.get("passed", 0)) == 1,
        "source_target": _array_sha256(source) == input_binding["source_image_array_sha256"]
        and _array_sha256(target) == input_binding["mixed_target_array_sha256"],
        "controller_interface": int(interface["passed"]) == 1,
        "oracle_identity": int(oracle_identity["passed"]) == 1,
        "paired_rng": int(rng_record["passed"]) == 1,
        "exact_backend": int(smoke["passed"]) == 1,
        "resource_projection": int(benchmark["passed"]) == 1,
    }
    gate = _semantic(
        {
            "schema": _schema(config) + "-preflight-gate",
            "schema_version": 1,
            "gate_type": "execution_integrity",
            "downstream_action_controlled": "fresh forward anchor generation",
            "checks": checks,
            "passed": int(all(checks.values())),
            "runtime": runtime,
            "resource_projection_sha256": benchmark["semantic_sha256"],
            "scientific_evidence_complete": 0,
        }
    )
    atomic_write_json(run_dir / "preflight_gate.json", gate)
    if not gate["passed"]:
        if not int(benchmark["passed"]):
            raise RolloutCLIError(
                "projected exact main rollout exceeds the frozen resource budget",
                failure_domain="resource_budget",
                failure_code="rollout_main_workflow_computationally_infeasible",
            )
        raise RolloutCLIError("rollout preflight integrity gate failed")
    _status(run_dir, state="ready_for_forward", stage="preflight")
    return gate


def _fake_forward(target: np.ndarray, *, anchor: int, role: str) -> np.ndarray:
    grid = np.arange(784, dtype=np.float64)
    perturbation = np.sin((grid + 1.0) * (0.017 + 1.0e-5 * anchor))
    perturbation -= perturbation.mean()
    amplitude = 0.10 if anchor <= 127 else 0.18
    state = target + amplitude * perturbation / np.abs(perturbation).sum()
    state = np.maximum(state, 1.0e-12)
    state /= state.sum()
    if role == "evaluation":
        state = np.roll(state, 1)
    elif role == "replication":
        state = np.roll(state, 2)
    return np.ascontiguousarray(state)


def _forward_role(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    role: str,
    path_id: int,
    anchors: Sequence[int],
) -> dict[str, Any]:
    source, target = _source_arrays(run_dir)
    role_dir = run_dir / "forward" / role
    summary_path = role_dir / "forward_summary.json"
    if summary_path.is_file():
        return _verify_forward_summary(run_dir, role)
    operation_started = time.perf_counter()
    expected_transitions = int((max(anchors) + 1) * 7 * 392)
    forward_profile = "forward_p1" if _fused_mode(args) else None
    projected_forward_bytes = (
        0
        if args.test_only or forward_profile is None
        else ((max(anchors) + 1) // 8)
        * _profile_shard_storage_estimate(run_dir, forward_profile)
    )
    _ensure_execution_budget(
        run_dir,
        role=role,
        additional_transitions=expected_transitions,
        test_only=bool(args.test_only),
        operation=f"{role}-forward-pre",
        profile_name=forward_profile,
        additional_persisted_bytes=projected_forward_bytes,
    )
    if args.test_only:
        transition_count = int((max(anchors) + 1) * 7 * 392)
        values = {int(anchor): _fake_forward(target, anchor=int(anchor), role=role) for anchor in anchors}
        diagnostics = {
            "certificate_fraction": 1.0,
            "certified_count": transition_count,
            "maximum_simplex_mass_error": 2.22e-16,
            "maximum_pair_mass_error": 2.22e-16,
            "minimum_state": float(min(value.min() for value in values.values())),
            "transition_count": transition_count,
            "elapsed_seconds": 0.01,
            "transitions_per_second": 1.0e9,
            "forbidden_counts": {},
            "maximum_cuda_memory_allocated": 0,
            "cuda_total_memory": 1,
            "restart_shard_steps": 8,
            "test_only_fake_sampler": 1,
        }
    else:
        from mnist.d0_jacobi_rb_tangent_rollout import run_forward_trajectory

        device = torch.device(args.device)
        initial = torch.as_tensor(target[None, :], dtype=torch.float64, device=device).contiguous()
        result = run_forward_trajectory(
            initial,
            anchor_steps=tuple(int(value) for value in anchors),
            output_dir=role_dir,
            trajectory_name=role,
            path_ids=(int(path_id),),
            root_seed=FORWARD_ROOT_SEED,
            profile=JacobiRBCudaProfile(),
        )
        raw_anchors = _field(result, "anchors")
        if not isinstance(raw_anchors, Mapping):
            raise RolloutCLIError("forward result omitted anchor states")
        values = {int(anchor): _state_row(raw_anchors[int(anchor)]) for anchor in anchors}
        diagnostics = dict(_field(result, "diagnostics", {}))
        diagnostics.setdefault("elapsed_seconds", float(_field(result, "elapsed_seconds", 0.0)))
        diagnostics.setdefault("transition_count", int(_field(result, "transition_count", 0)))
        if diagnostics["elapsed_seconds"] > 0.0:
            diagnostics.setdefault(
                "transitions_per_second",
                diagnostics["transition_count"] / diagnostics["elapsed_seconds"],
            )
    health = _health_record(
        diagnostics,
        expected_transition_count=expected_transitions,
        test_only=bool(args.test_only),
    )
    if not int(health["passed"]):
        raise RolloutCLIError(
            f"forward {role} exact-health contract failed",
            failure_domain="execution_integrity",
            failure_code="forward_exact_health_invalid",
        )
    for anchor, value in values.items():
        if not np.isfinite(value).all() or np.any(value < 0.0):
            raise RolloutCLIError("forward anchor is nonfinite or negative")
        if abs(float(value.sum()) - 1.0) > MAXIMUM_MASS_ERROR:
            raise RolloutCLIError("forward anchor violates simplex conservation")
    role_dir.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        role_dir / "anchors.npz",
        initial_mixed_target=target,
        **{f"step_{anchor:04d}": value for anchor, value in values.items()},
    )
    summary = _semantic(
        {
            "schema": _schema(args) + "-forward-summary",
            "schema_version": 1,
            "role": role,
            "path_id": int(path_id),
            "anchors": {
                str(anchor): {
                    "state_sha256": _array_sha256(value),
                    "simplex_mass": float(value.sum()),
                }
                for anchor, value in values.items()
            },
            "diagnostics": diagnostics,
            "health": health,
            "end_to_end_elapsed_seconds": time.perf_counter() - operation_started,
            "exact_reference_backend": int(not args.test_only),
            "test_only": int(args.test_only),
            "passed": 1,
        }
    )
    atomic_write_json(summary_path, summary)
    _ensure_execution_budget(
        run_dir,
        role=role,
        additional_transitions=0,
        test_only=bool(args.test_only),
        operation=f"{role}-forward-post",
        profile_name="forward_p1" if _fused_mode(args) else None,
    )
    return summary


def _forward_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    preflight = _load_semantic(run_dir / "preflight_gate.json", "preflight gate")
    if not int(preflight.get("passed", 0)):
        raise RolloutCLIError("forward stage requires passing preflight")
    config = _load_semantic(run_dir / "scientific_config.json", "scientific configuration")
    anchors = (int(config["short_anchor"]), int(config["full_anchor"]))
    development = _forward_role(
        run_dir,
        args,
        role="development",
        path_id=PATH_IDS["development"],
        anchors=anchors,
    )
    checks = {
        "development_complete": int(development.get("passed", 0)) == 1,
        "development_exact_health": int(development.get("health", {}).get("passed", 0)) == 1,
        "evaluation_forward_unopened": int(not (run_dir / "forward/evaluation").exists()) == 1,
        "development_selection_not_yet_committed": int(
            not (run_dir / "development/development_selection.json").exists()
        ) == 1,
    }
    gate = _semantic(
        {
            "schema": _schema(config) + "-forward-gate",
            "schema_version": 1,
            "gate_type": "execution_integrity",
            "downstream_action_controlled": "development reverse suffix",
            "checks": checks,
            "passed": int(all(checks.values())),
        }
    )
    atomic_write_json(run_dir / "forward_gate.json", gate)
    if not gate["passed"]:
        raise RolloutCLIError("fresh forward anchor generation failed")
    _status(run_dir, state="ready_for_development", stage="forward")
    return gate


def _anchor_state(run_dir: Path, role: str, anchor: int) -> np.ndarray:
    values = _load_npz(run_dir / "forward" / role / "anchors.npz")
    key = f"step_{int(anchor):04d}"
    if key not in values:
        raise ArtifactCompatibilityError(f"missing forward anchor {role}/{anchor}")
    return _state_row(values[key])


def _fake_reverse(
    anchor: np.ndarray,
    target: np.ndarray,
    *,
    variant: str,
    gain: float | None,
    oracle_fail: bool,
) -> dict[str, Any]:
    if variant == "zero":
        final = anchor.copy()
        control_ratio = 0.0
    elif variant == "oracle":
        factor = 1.08 if oracle_fail else 0.03
        final = target + factor * (anchor - target)
        control_ratio = 1.0
    else:
        active_gain = float(gain)
        factor = abs(1.0 - 0.45 * active_gain)
        final = target + factor * (anchor - target)
        control_ratio = 0.15 * active_gain
    final = np.maximum(final, 0.0)
    final /= final.sum()
    saved = {
        "start": anchor,
        "progress_25": 0.75 * anchor + 0.25 * final,
        "progress_50": 0.50 * anchor + 0.50 * final,
        "progress_75": 0.25 * anchor + 0.75 * final,
        "final": final,
    }
    return {
        "final_state": np.ascontiguousarray(final),
        "saved_states": {name: np.ascontiguousarray(value) for name, value in saved.items()},
        "diagnostics": {
            "certificate_fraction": 1.0,
            "certified_count": 10,
            "maximum_simplex_mass_error": 2.22e-16,
            "maximum_pair_mass_error": 2.22e-16,
            "forbidden_counts": {},
            "control_reference_displacement_ratio": control_ratio,
            "boundary_fraction_count": 0,
            "target_oracle_unreachable_boundary_count": 0,
            "transition_count": 10,
            "elapsed_seconds": 0.001,
            "transitions_per_second": 10_000.0,
            "maximum_cuda_memory_allocated": 0,
            "cuda_total_memory": 1,
            "test_only_fake_sampler": 1,
        },
    }


def _metrics_dict(state: np.ndarray, target: np.ndarray) -> dict[str, float]:
    from mnist.d0_jacobi_rb_tangent_rollout import raw_state_metrics

    metrics = raw_state_metrics(state, target)
    return {key: float(value) for key, value in _record(metrics).items() if isinstance(value, (int, float))}


def _render_states(
    run_dir: Path,
    *,
    trajectory_key: str,
    states: Mapping[str, np.ndarray],
) -> dict[str, dict[str, str]]:
    from mnist.d0_jacobi_rb_tangent_rollout import (
        fixed_rendering_scale,
        render_background_demixed,
        render_source_image,
        render_raw_density,
    )

    source, target = _source_arrays(run_dir)
    scale = fixed_rendering_scale(source, target, LAMBDA_MIX)
    records: dict[str, dict[str, str]] = {}
    for progress, state in states.items():
        raw = render_raw_density(state, scale)
        demixed = render_background_demixed(state, scale)
        raw_path = run_dir / "images" / "individual" / f"{trajectory_key}-{progress}-raw.png"
        demixed_path = (
            run_dir / "images" / "individual" / f"{trajectory_key}-{progress}-demixed.png"
        )
        _save_png(raw_path, raw)
        _save_png(demixed_path, demixed)
        records[progress] = {
            "raw": raw_path.relative_to(run_dir).as_posix(),
            "background_demixed": demixed_path.relative_to(run_dir).as_posix(),
        }
    return records


def _png_pixels_sha256(pixels: np.ndarray) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="L").save(
        buffer, format="PNG"
    )
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _contact_sheet_sha256(cells: Sequence[tuple[str, np.ndarray]]) -> str:
    from PIL import Image, ImageDraw

    scale = 5
    width, height = 28 * scale, 28 * scale + 22
    sheet = Image.new("L", (width * len(cells), height), color=255)
    draw = ImageDraw.Draw(sheet)
    for index, (label, pixels) in enumerate(cells):
        image = Image.fromarray(
            np.asarray(pixels, dtype=np.uint8), mode="L"
        ).resize((28 * scale, 28 * scale), resample=Image.Resampling.NEAREST)
        sheet.paste(image, (index * width, 0))
        draw.text((index * width + 2, 28 * scale + 3), str(label)[:20], fill=0)
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _verify_rendered_state_records(
    run_dir: Path,
    *,
    trajectory_key: str,
    states: Mapping[str, np.ndarray],
    records: Mapping[str, Any],
) -> None:
    from mnist.d0_jacobi_rb_tangent_rollout import (
        fixed_rendering_scale,
        render_background_demixed,
        render_raw_density,
    )

    source, target = _source_arrays(run_dir)
    scale = fixed_rendering_scale(source, target, LAMBDA_MIX)
    if set(records) != set(states):
        raise ArtifactCompatibilityError("rendered state record set changed")
    for progress, state in states.items():
        row = records.get(progress)
        if not isinstance(row, Mapping) or set(row) != {"raw", "background_demixed"}:
            raise ArtifactCompatibilityError("rendered state binding changed")
        expected = {
            "raw": (
                run_dir
                / "images/individual"
                / f"{trajectory_key}-{progress}-raw.png"
            ),
            "background_demixed": (
                run_dir
                / "images/individual"
                / f"{trajectory_key}-{progress}-demixed.png"
            ),
        }
        pixels = {
            "raw": render_raw_density(state, scale),
            "background_demixed": render_background_demixed(state, scale),
        }
        for kind, path in expected.items():
            if (
                row.get(kind) != path.relative_to(run_dir).as_posix()
                or not path.is_file()
                or file_fingerprint(path) != _png_pixels_sha256(pixels[kind])
            ):
                raise ArtifactCompatibilityError("rendered state image changed")


def _capture_trajectory_execution_failure(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    output_dir: Path,
    trajectory_key: str,
    role: str,
    horizon: str,
    variant: str,
    gain: float | None,
    anchor_step: int,
    anchor: np.ndarray,
    exc: BaseException,
) -> None:
    """Commit the last valid reverse state before propagating an execution error."""

    last_state = _state_row(anchor)
    last_shard_index: int | None = None
    last_shard_record_sha256: str | None = None
    committed_shard_count = 0
    restart_audit_error: str | None = None
    shard_root = output_dir / "reverse_shards" / trajectory_key
    for record_path in sorted(shard_root.glob("shard-*.json")):
        try:
            record = _load_semantic(record_path, "failed trajectory restart shard")
            if int(record.get("committed", 0)) != 1:
                raise ArtifactCompatibilityError("restart shard is not committed")
            state_path = record_path.with_suffix(".npz")
            if (
                not state_path.is_file()
                or file_fingerprint(state_path) != record.get("state_file_sha256")
            ):
                raise ArtifactCompatibilityError("restart shard state binding changed")
            arrays = _load_npz(state_path)
            if set(arrays) != {"state"}:
                raise ArtifactCompatibilityError("restart shard state schema changed")
            candidate = _state_row(arrays["state"])
            if _array_sha256(candidate) != record.get("output_state_sha256"):
                raise ArtifactCompatibilityError("restart shard output hash changed")
            last_state = candidate
            last_shard_index = int(record.get("shard_index", committed_shard_count))
            last_shard_record_sha256 = str(record.get("semantic_sha256"))
            committed_shard_count += 1
        except Exception as audit_exc:  # preserve the last prior valid shard
            restart_audit_error = f"{type(audit_exc).__name__}: {audit_exc}"
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    state_artifact = _atomic_npz(
        output_dir / "execution_failure_last_valid_state.npz",
        last_valid=last_state,
    )
    state_artifact["path"] = (
        output_dir / "execution_failure_last_valid_state.npz"
    ).relative_to(run_dir).as_posix()
    images: dict[str, dict[str, str]] = {}
    image_error: str | None = None
    try:
        images = _render_states(
            run_dir,
            trajectory_key=f"{trajectory_key}-failure",
            states={"last_valid": last_state},
        )
    except Exception as render_exc:  # raw state remains the authorizing failure evidence
        image_error = f"{type(render_exc).__name__}: {render_exc}"
    atomic_write_json(
        output_dir / "trajectory_failure.json",
        _semantic(
            {
                "schema": _schema(args) + "-trajectory-execution-failure",
                "schema_version": 1,
                "role": role,
                "horizon": horizon,
                "variant": variant,
                "gain": gain,
                "anchor_step": int(anchor_step),
                "anchor_state_sha256": _array_sha256(_state_row(anchor)),
                "last_valid_state_sha256": _array_sha256(last_state),
                "last_valid_state_artifact": state_artifact,
                "last_valid_images": images,
                "last_valid_image_error": image_error,
                "committed_shard_count": committed_shard_count,
                "last_committed_shard_index": last_shard_index,
                "last_committed_shard_record_sha256": last_shard_record_sha256,
                "restart_audit_error": restart_audit_error,
                "failure_domain": getattr(exc, "failure_domain", "execution_integrity"),
                "failure_code": getattr(
                    exc, "failure_code", "reverse_trajectory_execution_invalid"
                ),
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "scientific_evidence_complete": 0,
            }
        ),
    )


def _trajectory(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    role: str,
    horizon: str,
    anchor_step: int,
    variant: str,
    gain: float | None = None,
) -> dict[str, Any]:
    variant_name = variant if gain is None else f"learned-gain-{gain:g}"
    trajectory_key = f"{role}-{horizon}-{variant_name}"
    output_dir = run_dir / role / horizon / variant_name
    summary_path = output_dir / "trajectory_summary.json"
    if summary_path.is_file():
        return _verify_trajectory_summary(summary_path, run_dir)
    operation_started = time.perf_counter()
    expected_transitions = int((int(anchor_step) + 1) * 7 * 4 * 392)
    _ensure_execution_budget(
        run_dir,
        role=role,
        additional_transitions=expected_transitions,
        test_only=bool(args.test_only),
        operation=f"{trajectory_key}-pre",
    )
    source, target = _source_arrays(run_dir)
    anchor = _anchor_state(run_dir, role if role != "replication" else "replication", anchor_step)
    if args.test_only:
        result = _fake_reverse(
            anchor,
            target,
            variant=variant,
            gain=gain,
            oracle_fail=bool(args.test_oracle_fail),
        )
    else:
        from mnist.d0_jacobi_rb_tangent_rollout import (
            CertifiedExploratoryReference,
            ScaledTangentScoreController,
            TargetFractionOracleController,
            ZeroTangentScoreController,
            run_reverse_trajectory,
        )

        device = torch.device(args.device)
        if variant == "zero":
            controller = ZeroTangentScoreController()
        elif variant == "oracle":
            controller = TargetFractionOracleController(target, microsteps=MICROSTEPS)
        else:
            base = _load_model(run_dir, device, test_only=False)
            controller = ScaledTangentScoreController(base, float(gain))
        stream_role = STREAM_ROLES[role]

        def reference_factory(shard_index: int) -> Any:
            return CertifiedExploratoryReference(
                profile=JacobiRBCudaProfile(),
                root_seed=REVERSE_ROOT_SEED,
                # The shard index is deliberately not a new RNG stream: exact
                # transition IDs already partition shards.  Keeping the same
                # role preserves common random numbers across variants.
                stream_role=stream_role,
            )

        try:
            result_obj = run_reverse_trajectory(
                torch.as_tensor(
                    anchor[None, :], dtype=torch.float64, device=device
                ).contiguous(),
                anchor_step=int(anchor_step),
                output_dir=output_dir,
                trajectory_name=trajectory_key,
                controller=controller,
                reference_factory=reference_factory,
                path_ids=(PATH_IDS[role],),
                controller_binding={
                    "variant": variant,
                    "gain": gain,
                    "checkpoint_state_sha256": (
                        CHECKPOINT_STATE_SHA256 if variant == "learned" else None
                    ),
                },
                rng_binding={
                    "root_seed": REVERSE_ROOT_SEED,
                    "stream_role": stream_role,
                    "variant_in_rng_key": 0,
                },
                label=3,
                microsteps=MICROSTEPS,
                device=device,
            )
        except Exception as exc:
            _capture_trajectory_execution_failure(
                run_dir,
                args,
                output_dir=output_dir,
                trajectory_key=trajectory_key,
                role=role,
                horizon=horizon,
                variant=variant,
                gain=gain,
                anchor_step=anchor_step,
                anchor=anchor,
                exc=exc,
            )
            raise
        result = {
            "final_state": _field(result_obj, "final_state"),
            "saved_states": _field(result_obj, "saved_states"),
            "diagnostics": dict(_field(result_obj, "diagnostics", {})),
        }
    final = _state_row(result["final_state"])
    raw_saved = result.get("saved_states")
    if not isinstance(raw_saved, Mapping):
        raw_saved = {"start": anchor, "final": final}
    saved = {str(name): _state_row(value) for name, value in raw_saved.items()}
    saved.setdefault("start", anchor)
    saved.setdefault("final", final)
    if not np.isfinite(final).all() or np.any(final < 0.0):
        raise RolloutCLIError("reverse trajectory produced invalid final state")
    if abs(float(final.sum()) - 1.0) > MAXIMUM_MASS_ERROR:
        raise RolloutCLIError("reverse trajectory violated simplex conservation")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_artifact = _atomic_npz(output_dir / "selected_states.npz", **saved)
    state_artifact["path"] = (output_dir / "selected_states.npz").relative_to(run_dir).as_posix()
    image_records = _render_states(run_dir, trajectory_key=trajectory_key, states=saved)
    progress_metrics = {name: _metrics_dict(value, target) for name, value in saved.items()}
    diagnostics = dict(result.get("diagnostics", {}))
    if args.test_only:
        diagnostics["transition_count"] = expected_transitions
        diagnostics["certified_count"] = expected_transitions
    else:
        diagnostics.setdefault("elapsed_seconds", float(_field(result_obj, "elapsed_seconds", 0.0)))
        diagnostics.setdefault("transition_count", int(_field(result_obj, "transition_count", 0)))
        if diagnostics["elapsed_seconds"] > 0.0:
            diagnostics.setdefault(
                "transitions_per_second",
                diagnostics["transition_count"] / diagnostics["elapsed_seconds"],
            )
    health = _health_record(
        diagnostics,
        expected_transition_count=expected_transitions,
        test_only=bool(args.test_only),
    )
    summary = _semantic(
        {
            "schema": _schema(args) + "-reverse-trajectory-summary",
            "schema_version": 1,
            "role": role,
            "horizon": horizon,
            "anchor_step": int(anchor_step),
            "path_id": PATH_IDS[role],
            "variant": variant,
            "variant_name": variant_name,
            "gain": gain,
            "checkpoint_role": (
                "historical_validation_inspected_post_hoc_diagnostic"
                if variant == "learned" else None
            ),
            "rng_binding": {
                "root_seed": REVERSE_ROOT_SEED,
                "stream_role": STREAM_ROLES[role],
                "variant_in_rng_key": 0,
            },
            "final_state_sha256": _array_sha256(final),
            "selected_states_artifact": state_artifact,
            "images": image_records,
            "metrics": progress_metrics,
            "diagnostics": diagnostics,
            "health": health,
            "end_to_end_elapsed_seconds": time.perf_counter() - operation_started,
            "passed_integrity": int(health["passed"]),
            "exploratory_post_hoc": 1,
        }
    )
    atomic_write_json(summary_path, summary)
    if not int(health["passed"]):
        # The failed objective artifact remains inspectable and hash-bound;
        # integrity failure blocks interpretation, not evidence preservation.
        atomic_write_json(
            output_dir / "trajectory_failure.json",
            _semantic(
                {
                    "schema": _schema(args) + "-trajectory-failure",
                    "schema_version": 1,
                    "trajectory_summary_sha256": summary["semantic_sha256"],
                    "failure_domain": "execution_integrity",
                    "failure_code": "reverse_exact_health_invalid",
                    "health": health,
                }
            ),
        )
        raise RolloutCLIError(
            f"reverse trajectory exact-health contract failed: {trajectory_key}",
            failure_domain="execution_integrity",
            failure_code="reverse_exact_health_invalid",
        )
    _ensure_execution_budget(
        run_dir,
        role=role,
        additional_transitions=0,
        test_only=bool(args.test_only),
        operation=f"{trajectory_key}-post",
    )
    return summary


def _squared_l2(summary: Mapping[str, Any]) -> float:
    metrics = summary.get("metrics", {}).get("final", {})
    for name in ("squared_l2_error", "squared_l2", "l2_squared"):
        if name in metrics:
            return float(metrics[name])
    raise RolloutCLIError("trajectory metrics omitted final squared L2 error")


def _development_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not int(_load_semantic(run_dir / "forward_gate.json", "forward gate").get("passed", 0)):
        raise RolloutCLIError("development requires fresh forward anchors")
    config = _load_semantic(run_dir / "scientific_config.json", "scientific configuration")
    anchor = int(config["short_anchor"])
    zero = _trajectory(
        run_dir, args, role="development", horizon="short", anchor_step=anchor, variant="zero"
    )
    learned = [
        _trajectory(
            run_dir,
            args,
            role="development",
            horizon="short",
            anchor_step=anchor,
            variant="learned",
            gain=gain,
        )
        for gain in LEARNED_GAINS
    ]
    oracle = _trajectory(
        run_dir, args, role="development", horizon="short", anchor_step=anchor, variant="oracle"
    )
    ranked = sorted(
        ((float(_squared_l2(row)), float(row["gain"]), row) for row in learned),
        key=lambda item: (item[0], item[1]),
    )
    selected_error, selected_gain, selected = ranked[0]
    zero_error = _squared_l2(zero)
    oracle_error = _squared_l2(oracle)
    input_binding = _load_semantic(
        run_dir / "input_bindings/input_binding.json", "input binding"
    )
    trajectory_rows = [zero, *learned, oracle]
    trajectory_commitments = {
        str(row["variant_name"]): {
            "trajectory_semantic_sha256": row["semantic_sha256"],
            "selected_states_sha256": row["selected_states_artifact"]["sha256"],
            "final_state_sha256": row["final_state_sha256"],
        }
        for row in trajectory_rows
    }
    selection = _semantic(
        {
            "schema": _schema(args) + "-development-selection",
            "schema_version": 1,
            "selection_criterion": "minimum final raw squared L2; ties smaller gain",
            "gain_grid": list(LEARNED_GAINS),
            "learned_endpoint_squared_l2": {
                str(row["gain"]): _squared_l2(row) for row in learned
            },
            "zero_endpoint_squared_l2": zero_error,
            "oracle_endpoint_squared_l2": oracle_error,
            "selected_gain": selected_gain,
            "selected_trajectory_sha256": selected["semantic_sha256"],
            "trajectory_commitments": trajectory_commitments,
            "selected_learned_beats_zero": int(selected_error < zero_error),
            "oracle_beats_zero": int(oracle_error < zero_error),
            "checkpoint_file_sha256": input_binding["checkpoint_file_sha256"],
            "source_image_array_sha256": input_binding["source_image_array_sha256"],
            "mixed_target_array_sha256": input_binding["mixed_target_array_sha256"],
            "anchor_sha256": _load_semantic(
                run_dir / "forward/development/forward_summary.json",
                "development forward summary",
            )["anchors"][str(anchor)]["state_sha256"],
            "scientific_config_sha256": config["semantic_sha256"],
            "evaluation_evidence_opened": 0,
            "exploratory_post_hoc": 1,
            "committed_before_evaluation": 1,
        }
    )
    atomic_write_json(run_dir / "development/development_selection.json", selection)
    _paired_progress_metrics(
        run_dir,
        args,
        role="development",
        horizon="short",
        zero=zero,
        learned=selected,
        oracle=oracle,
    )
    _write_development_contact_sheet(run_dir, selected_gain)
    oracle_passed = oracle_error < zero_error
    gate = _semantic(
        {
            "schema": _schema(args) + "-development-gate",
            "schema_version": 1,
            "gate_type": "interpretability",
            "downstream_action_controlled": "opening fresh evaluation rollout",
            "exact_proposition_tested": (
                "source-informed target-pull oracle lowers development short-suffix final L2"
            ),
            "oracle_beats_zero": int(oracle_passed),
            "selected_gain": selected_gain,
            "selected_learned_beats_zero_diagnostic": int(selected_error < zero_error),
            "passed": int(oracle_passed),
            "failure_does_not_mean": "the learned score has no useful signal",
        }
    )
    atomic_write_json(run_dir / "development_gate.json", gate)
    if not oracle_passed:
        decision = _semantic(
            {
                "schema": _schema(args) + "-decision",
                "schema_version": 1,
                "decision": "development_oracle_control_failed",
                "recommended_next_action": "repair controller/oracle/reference composition",
                "evaluation_performed": 0,
                "replication_performed": 0,
                "exploratory": 1,
            }
        )
        atomic_write_json(run_dir / "exploratory_decision.json", decision)
        _status(
            run_dir,
            state="complete",
            stage="development",
            decision=decision["decision"],
            scientific_evidence_complete=1,
        )
        _write_report(run_dir)
        _finalize_artifacts(run_dir)
        return gate
    _status(run_dir, state="ready_for_evaluation", stage="development")
    return gate


def _paired_metrics(
    zero: Mapping[str, Any], learned: Mapping[str, Any], oracle: Mapping[str, Any]
) -> dict[str, Any]:
    zero_error = _squared_l2(zero)
    learned_error = _squared_l2(learned)
    oracle_error = _squared_l2(oracle)
    return {
        "zero_squared_l2": zero_error,
        "learned_squared_l2": learned_error,
        "oracle_squared_l2": oracle_error,
        "learned_improvement_over_zero": zero_error - learned_error,
        "oracle_improvement_over_zero": zero_error - oracle_error,
        "learned_relative_improvement": (
            (zero_error - learned_error) / zero_error if zero_error > 0.0 else None
        ),
        "oracle_relative_improvement": (
            (zero_error - oracle_error) / zero_error if zero_error > 0.0 else None
        ),
    }


def _verify_development_selection(run_dir: Path) -> dict[str, Any]:
    selection = _load_semantic(
        run_dir / "development/development_selection.json", "development selection"
    )
    gain = float(selection.get("selected_gain", float("nan")))
    if gain not in LEARNED_GAINS:
        raise ArtifactCompatibilityError("selected development gain is outside the frozen grid")
    paths = sorted((run_dir / "development/short").glob("*/trajectory_summary.json"))
    expected_names = {"zero", "oracle", *(f"learned-gain-{value:g}" for value in LEARNED_GAINS)}
    if len(paths) != 6 or {path.parent.name for path in paths} != expected_names:
        raise ArtifactCompatibilityError("development trajectory family changed")
    summaries = {
        path.parent.name: _verify_trajectory_summary(path, run_dir) for path in paths
    }
    measured_commitments = {
        name: {
            "trajectory_semantic_sha256": row["semantic_sha256"],
            "selected_states_sha256": row["selected_states_artifact"]["sha256"],
            "final_state_sha256": row["final_state_sha256"],
        }
        for name, row in summaries.items()
    }
    if selection.get("trajectory_commitments") != measured_commitments:
        raise ArtifactCompatibilityError("development trajectory commitments changed")
    learned = sorted(
        (
            _squared_l2(summaries[f"learned-gain-{value:g}"]),
            float(value),
            summaries[f"learned-gain-{value:g}"],
        )
        for value in LEARNED_GAINS
    )
    selected_error, measured_gain, measured = learned[0]
    if (
        measured_gain != gain
        or measured["semantic_sha256"] != selection.get("selected_trajectory_sha256")
        or float(selection.get("learned_endpoint_squared_l2", {}).get(str(gain), float("nan")))
        != float(selected_error)
    ):
        raise ArtifactCompatibilityError("development deterministic ranking changed")
    config = _load_semantic(run_dir / "scientific_config.json", "scientific configuration")
    binding = _load_semantic(run_dir / "input_bindings/input_binding.json", "input binding")
    anchor = int(config["short_anchor"])
    forward = _verify_forward_summary(run_dir, "development")
    checks = (
        selection.get("scientific_config_sha256") == config["semantic_sha256"],
        selection.get("checkpoint_file_sha256") == binding["checkpoint_file_sha256"],
        selection.get("source_image_array_sha256") == binding["source_image_array_sha256"],
        selection.get("mixed_target_array_sha256") == binding["mixed_target_array_sha256"],
        selection.get("anchor_sha256") == forward["anchors"][str(anchor)]["state_sha256"],
        int(selection.get("evaluation_evidence_opened", -1)) == 0,
        tuple(float(value) for value in selection.get("gain_grid", ())) == LEARNED_GAINS,
    )
    if not all(checks):
        raise ArtifactCompatibilityError("development selection binding changed")
    return selection


_PROGRESS_ORDER = ("start", "progress_25", "progress_50", "progress_75", "final")


def _summary_states(run_dir: Path, summary: Mapping[str, Any]) -> dict[str, np.ndarray]:
    artifact = summary.get("selected_states_artifact", {})
    if not isinstance(artifact, Mapping):
        raise ArtifactCompatibilityError("trajectory state binding is malformed")
    path = Path(str(artifact.get("path", "")))
    if not path.is_absolute():
        path = run_dir / path
    if not path.is_file() or file_fingerprint(path) != artifact.get("sha256"):
        raise ArtifactCompatibilityError("trajectory state binding changed")
    raw = _load_npz(path)
    normalized: dict[str, np.ndarray] = {}
    aliases = {
        "progress-25": "progress_25",
        "progress-50": "progress_50",
        "progress-75": "progress_75",
    }
    for name, value in raw.items():
        normalized[aliases.get(name, name)] = _state_row(value)
    if any(name not in normalized for name in _PROGRESS_ORDER):
        raise ArtifactCompatibilityError("trajectory omitted a required progress state")
    return normalized


def _relative(improvement: float, baseline: float) -> float | None:
    return improvement / baseline if baseline != 0.0 else None


def _paired_progress_metrics(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    role: str,
    horizon: str,
    zero: Mapping[str, Any],
    learned: Mapping[str, Any],
    oracle: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist paired objective/mechanism evidence at every saved state."""

    _source, target = _source_arrays(run_dir)
    states = {
        "zero": _summary_states(run_dir, zero),
        "learned": _summary_states(run_dir, learned),
        "oracle": _summary_states(run_dir, oracle),
    }
    rows: list[dict[str, Any]] = []
    previous_improvement = {"learned": 0.0, "oracle": 0.0}
    for progress_index, progress in enumerate(_PROGRESS_ORDER):
        zero_metrics = _metrics_dict(states["zero"][progress], target)
        for variant in ("learned", "oracle"):
            candidate_metrics = _metrics_dict(states[variant][progress], target)
            l2_improvement = (
                zero_metrics["squared_l2_error"]
                - candidate_metrics["squared_l2_error"]
            )
            l1_improvement = zero_metrics["l1_error"] - candidate_metrics["l1_error"]
            tv_improvement = (
                zero_metrics["total_variation_distance"]
                - candidate_metrics["total_variation_distance"]
            )
            correlation_improvement = (
                candidate_metrics["centered_contrast_correlation"]
                - zero_metrics["centered_contrast_correlation"]
            )
            difference = states[variant][progress] - states["zero"][progress]
            divergence_l1 = float(np.sum(np.abs(difference), dtype=np.float64))
            row = {
                "role": role,
                "horizon": horizon,
                "progress": progress,
                "progress_index": progress_index,
                "variant": variant,
                "zero_squared_l2": zero_metrics["squared_l2_error"],
                "candidate_squared_l2": candidate_metrics["squared_l2_error"],
                "squared_l2_improvement_over_zero": l2_improvement,
                "relative_squared_l2_improvement": _relative(
                    l2_improvement, zero_metrics["squared_l2_error"]
                ),
                "zero_l1": zero_metrics["l1_error"],
                "candidate_l1": candidate_metrics["l1_error"],
                "l1_improvement_over_zero": l1_improvement,
                "relative_l1_improvement": _relative(
                    l1_improvement, zero_metrics["l1_error"]
                ),
                "zero_total_variation": zero_metrics["total_variation_distance"],
                "candidate_total_variation": candidate_metrics["total_variation_distance"],
                "total_variation_improvement_over_zero": tv_improvement,
                "relative_total_variation_improvement": _relative(
                    tv_improvement, zero_metrics["total_variation_distance"]
                ),
                "zero_centered_contrast_correlation": zero_metrics[
                    "centered_contrast_correlation"
                ],
                "candidate_centered_contrast_correlation": candidate_metrics[
                    "centered_contrast_correlation"
                ],
                "correlation_improvement_over_zero": correlation_improvement,
                "relative_correlation_improvement": _relative(
                    correlation_improvement,
                    abs(zero_metrics["centered_contrast_correlation"]),
                ),
                "candidate_zero_squared_l2_divergence": float(
                    np.dot(difference, difference)
                ),
                "candidate_zero_l1_divergence": divergence_l1,
                "candidate_zero_total_variation_divergence": 0.5 * divergence_l1,
                "successive_quarter_squared_l2_improvement_contribution": (
                    l2_improvement - previous_improvement[variant]
                    if progress_index > 0
                    else 0.0
                ),
            }
            previous_improvement[variant] = l2_improvement
            rows.append(row)
    destination = run_dir / "metrics"
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{role}_{horizon}_paired_progress"
    atomic_write_csv(destination / f"{stem}.csv", rows)
    record = _semantic(
        {
            "schema": _schema(args) + "-paired-progress-metrics",
            "schema_version": 1,
            "role": role,
            "horizon": horizon,
            "progress_order": list(_PROGRESS_ORDER),
            "independent_unit": "one paired path",
            "rows": rows,
            "diagnostic_only": 1,
        }
    )
    atomic_write_json(destination / f"{stem}.json", record)
    return record


def _replication_capacity_record(run_dir: Path, *, test_only: bool) -> dict[str, Any]:
    """Project the frozen replication from the just-measured analogous work."""

    forward = _verify_forward_summary(run_dir, "evaluation")
    zero_paths = list((run_dir / "evaluation/full").glob("zero/trajectory_summary.json"))
    learned_paths = list(
        (run_dir / "evaluation/full").glob("learned-gain-*/trajectory_summary.json")
    )
    if len(zero_paths) != 1 or len(learned_paths) != 1:
        raise ArtifactCompatibilityError("evaluation full pair is incomplete")
    zero = _verify_trajectory_summary(zero_paths[0], run_dir)
    learned = _verify_trajectory_summary(learned_paths[0], run_dir)
    projected_seconds = math.fsum(
        max(
            float(row.get("end_to_end_elapsed_seconds", 0.0)),
            float(row.get("health", {}).get("elapsed_seconds", 0.0)),
        )
        for row in (forward, zero, learned)
    )
    analogous_roots = [
        run_dir / "forward/evaluation",
        zero_paths[0].parent,
        learned_paths[0].parent,
    ]
    # A fused replication persists one shared P2 family, not merely the two
    # small row-level trajectory archives.  The already measured P3 prefix
    # plus P6 suffix is a conservative storage proxy for that P2 full family.
    for shared in (
        run_dir / "evaluation/full/fused_prefix",
        run_dir / "evaluation/joined_suffix/fused_family",
    ):
        if shared.is_dir():
            analogous_roots.append(shared)
    analogous_bytes = sum(
        path.stat().st_size
        for root in analogous_roots
        for path in root.rglob("*")
        if path.is_file()
    )
    current_bytes = _directory_bytes(run_dir)
    run_config = (
        _load_semantic(run_dir / "scientific_config.json", "scientific configuration")
        if (run_dir / "scientific_config.json").is_file()
        else {}
    )
    storage_reserve = (
        FUSED_PROJECTION_FIXED_STORAGE_RESERVE_BYTES if _fused_mode(run_config) else 0
    )
    record = _semantic(
        {
            "schema": (
                FUSED_RUN_SCHEMA if _fused_mode(run_config) else RUN_SCHEMA
            )
            + "-replication-resource-projection",
            "schema_version": 1,
            "projection_basis": (
                "measured evaluation forward, full zero/learned row artifacts, "
                "and conservative shared fused P3-prefix/P6-suffix archives"
            ),
            "projected_seconds": projected_seconds,
            "maximum_seconds": MAXIMUM_REPLICATION_WALL_SECONDS,
            "current_persisted_bytes": current_bytes,
            "projected_additional_bytes": analogous_bytes,
            "fixed_uncommitted_storage_reserve_bytes": storage_reserve,
            "projected_total_bytes": current_bytes + analogous_bytes + storage_reserve,
            "maximum_persisted_bytes": MAXIMUM_PERSISTED_BYTES,
            "passed": int(
                test_only
                or (
                    projected_seconds <= MAXIMUM_REPLICATION_WALL_SECONDS
                    and current_bytes + analogous_bytes + storage_reserve
                    <= MAXIMUM_PERSISTED_BYTES
                )
            ),
        }
    )
    atomic_write_json(run_dir / "metrics/replication_resource_projection.json", record)
    return record


def _write_development_contact_sheet(run_dir: Path, selected_gain: float) -> None:
    """Always publish the short objective artifact, including oracle failure."""

    from mnist.d0_jacobi_rb_tangent_rollout import (
        fixed_rendering_scale,
        render_background_demixed,
        render_source_image,
    )

    source, target = _source_arrays(run_dir)
    config = _load_semantic(run_dir / "scientific_config.json", "scientific configuration")
    anchor = int(config["short_anchor"])
    scale = fixed_rendering_scale(source, target, LAMBDA_MIX)
    anchor_values = _load_npz(run_dir / "forward/development/anchors.npz")
    variants = {
        "zero": run_dir / "development/short/zero/selected_states.npz",
        "learned": run_dir
        / f"development/short/learned-gain-{selected_gain:g}/selected_states.npz",
        "oracle": run_dir / "development/short/oracle/selected_states.npz",
    }
    cells = [
        ("source", render_source_image(source, scale)),
        ("mixed-target", render_background_demixed(target, scale)),
        ("step-127", render_background_demixed(anchor_values[f"step_{anchor:04d}"], scale)),
    ]
    for label, path in variants.items():
        values = _load_npz(path)
        cells.append((label, render_background_demixed(values["final"], scale)))
    _contact_sheet(run_dir / "images/development_short_contact_sheet.png", cells)


def _classify_evaluation_outcome(
    *,
    oracle_short: bool,
    oracle_full: bool,
    learned_short: bool,
    learned_full: bool,
    learned_short_ratio: float,
) -> tuple[str, str]:
    """Apply the preregistered objective-first outcome precedence."""

    # The complete path is the primary objective and the plan's optional
    # replication rule is explicitly full-oracle + full-learned + resources.
    # Recognize that joint full-horizon outcome first so a contrary short
    # oracle diagnostic cannot produce a contradictory authorization state.
    if oracle_full and learned_full:
        return (
            "learned_full_dynamic_signal",
            "run optional frozen replication, then verify the setting at M=8",
        )
    if not oracle_short:
        return (
            "evaluation_oracle_short_control_failed",
            "repair controller/oracle/reference composition",
        )
    if not oracle_full:
        return (
            "oracle_long_horizon_composition_failed",
            "compare the fixed oracle at M=2 and M=8 on a bounded horizon",
        )
    if learned_short:
        return (
            "learned_short_only_dynamic_signal",
            "localize accumulation with anchors 255/383 and frozen late-time schedules",
        )
    if math.isfinite(learned_short_ratio) and learned_short_ratio < WEAK_CONTROL_RATIO:
        return (
            "learned_short_control_dynamically_negligible",
            "run a bounded calibration/scale experiment",
        )
    return (
        "learned_short_rollout_direction_not_useful",
        "compare a materially different global or rollout-trained learner",
    )


def _evaluation_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    development_gate = _load_semantic(run_dir / "development_gate.json", "development gate")
    if not int(development_gate.get("passed", 0)):
        raise RolloutCLIError("evaluation is closed because the development oracle failed")
    selection = _verify_development_selection(run_dir)
    config = _load_semantic(run_dir / "scientific_config.json", "scientific configuration")
    selected_gain = float(selection["selected_gain"])
    # The evaluation path is first realized only after the development gain
    # commitment exists.  Forward simulation is evidence opening here, not in
    # the earlier forward stage.
    _forward_role(
        run_dir,
        args,
        role="evaluation",
        path_id=PATH_IDS["evaluation"],
        anchors=(int(config["short_anchor"]), int(config["full_anchor"])),
    )
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for horizon, anchor in (
        ("short", int(config["short_anchor"])),
        ("full", int(config["full_anchor"])),
    ):
        variants = {
            "zero": _trajectory(
                run_dir, args, role="evaluation", horizon=horizon, anchor_step=anchor, variant="zero"
            ),
            "learned": _trajectory(
                run_dir,
                args,
                role="evaluation",
                horizon=horizon,
                anchor_step=anchor,
                variant="learned",
                gain=selected_gain,
            ),
            "oracle": _trajectory(
                run_dir, args, role="evaluation", horizon=horizon, anchor_step=anchor, variant="oracle"
            ),
        }
        results[horizon] = variants
        _paired_progress_metrics(
            run_dir,
            args,
            role="evaluation",
            horizon=horizon,
            zero=variants["zero"],
            learned=variants["learned"],
            oracle=variants["oracle"],
        )
    short = _paired_metrics(**results["short"])
    full = _paired_metrics(**results["full"])
    atomic_write_json(
        run_dir / "metrics/evaluation_paired_metrics.json",
        _semantic(
            {
                "schema": _schema(args) + "-evaluation-paired-metrics",
                "schema_version": 1,
                "selected_gain": selected_gain,
                "short": short,
                "full": full,
                "threshold_type": "diagnostic",
                "confirmatory_inference": 0,
            }
        ),
    )
    oracle_short = short["oracle_improvement_over_zero"] > 0.0
    oracle_full = full["oracle_improvement_over_zero"] > 0.0
    learned_short = short["learned_improvement_over_zero"] > 0.0
    learned_full = full["learned_improvement_over_zero"] > 0.0
    learned_short_ratio = float(
        results["short"]["learned"].get("diagnostics", {}).get(
            "control_reference_displacement_ratio", float("nan")
        )
    )
    learned_full_ratio = float(
        results["full"]["learned"].get("diagnostics", {}).get(
            "control_reference_displacement_ratio", float("nan")
        )
    )
    decision_name, next_action = _classify_evaluation_outcome(
        oracle_short=oracle_short,
        oracle_full=oracle_full,
        learned_short=learned_short,
        learned_full=learned_full,
        learned_short_ratio=learned_short_ratio,
    )
    replication_resources = _replication_capacity_record(
        run_dir, test_only=bool(args.test_only)
    )
    replication_authorized = bool(
        oracle_full
        and learned_full
        and int(replication_resources["passed"]) == 1
    )
    decision = _semantic(
        {
            "schema": _schema(args) + "-decision",
            "schema_version": 1,
            "decision": decision_name,
            "selected_gain": selected_gain,
            "short": short,
            "full": full,
            "learned_short_control_reference_displacement_ratio": learned_short_ratio,
            "learned_full_control_reference_displacement_ratio": learned_full_ratio,
            "replication_authorized": int(replication_authorized),
            "replication_resource_projection_sha256": replication_resources[
                "semantic_sha256"
            ],
            "recommended_next_action": next_action,
            "research_mode": "exploratory",
            "validation_pass_claim_authorized": 0,
            "general_generator_claim_authorized": 0,
            "prior_start_claim_authorized": 0,
        }
    )
    atomic_write_json(run_dir / "exploratory_decision.json", decision)
    flattened = [row for horizon_rows in results.values() for row in horizon_rows.values()]
    all_integrity = all(int(row.get("passed_integrity", 0)) == 1 for row in flattened)
    all_images = all(
        all(
            (run_dir / relative).is_file()
            for image in row.get("images", {}).values()
            for relative in image.values()
        )
        for row in flattened
    )
    paired_rng = all(
        len(
            {
                config_fingerprint(row["rng_binding"])
                for row in results[horizon].values()
            }
        ) == 1
        for horizon in ("short", "full")
    )
    gate = _semantic(
        {
            "schema": _schema(args) + "-evaluation-gate",
            "schema_version": 1,
            "gate_type": "execution_integrity",
            "all_trajectories_complete": int(all_integrity),
            "all_images_saved": int(all_images),
            "paired_rng_variant_independent": int(paired_rng),
            "passed": int(all_integrity and all_images and paired_rng),
            "diagnostic_decision": decision_name,
        }
    )
    atomic_write_json(run_dir / "evaluation_gate.json", gate)
    if not int(gate["passed"]):
        raise RolloutCLIError("evaluation trajectory integrity gate failed")
    _write_contact_sheets(run_dir)
    _status(
        run_dir,
        state="ready_for_replication" if replication_authorized else "ready_for_report",
        stage="evaluation",
        decision=decision_name,
        scientific_evidence_complete=1,
    )
    return gate


def _write_contact_sheets(run_dir: Path) -> None:
    from mnist.d0_jacobi_rb_tangent_rollout import (
        fixed_rendering_scale,
        render_background_demixed,
        render_source_image,
    )

    source, target = _source_arrays(run_dir)
    config = _load_semantic(run_dir / "scientific_config.json", "scientific configuration")
    scale = fixed_rendering_scale(source, target, LAMBDA_MIX)
    base = [
        ("source", render_source_image(source, scale)),
        ("mixed-target", render_background_demixed(target, scale)),
    ]
    for horizon, anchor in (
        ("short", int(config["short_anchor"])),
        ("full", int(config["full_anchor"])),
    ):
        anchor_values = _load_npz(run_dir / "forward/evaluation/anchors.npz")
        cells = [
            *base,
            ("anchor", render_background_demixed(anchor_values[f"step_{anchor:04d}"], scale)),
        ]
        for variant in ("zero", "learned", "oracle"):
            candidates = list((run_dir / "evaluation" / horizon).glob(f"{variant}*/selected_states.npz"))
            if candidates:
                states = _load_npz(candidates[0])
                cells.append((variant, render_background_demixed(states["final"], scale)))
        _contact_sheet(run_dir / "images" / f"{horizon}_contact_sheet.png", cells)

    progress_cells: list[tuple[str, np.ndarray]] = []
    for variant in ("zero", "learned", "oracle"):
        candidates = list((run_dir / "evaluation/full").glob(f"{variant}*/selected_states.npz"))
        if not candidates:
            continue
        states = _load_npz(candidates[0])
        for name in ("start", "progress_25", "progress-25", "progress_50", "progress-50", "progress_75", "progress-75", "final"):
            if name in states:
                progress_cells.append(
                    (f"{variant}-{name}", render_background_demixed(states[name], scale))
                )
    _contact_sheet(run_dir / "images/trajectory_contact_sheet.png", progress_cells)


def _commit_fused_family_summary(
    run_dir: Path,
    *,
    destination: Path,
    result: Any,
    family_name: str,
    segment_name: str,
    health: Mapping[str, Any],
    end_to_end_elapsed_seconds: float,
) -> dict[str, Any]:
    summary_path = destination / "family_summary.json"
    if summary_path.is_file():
        return _verify_fused_family_summary(summary_path, run_dir)
    result_record = _record(result.to_record())
    saved_values = {
        str(name): np.ascontiguousarray(value, dtype=np.float64)
        for name, value in _field(result, "saved_states", {}).items()
    }
    expected_saved = result_record.get("saved_state_sha256", {})
    if (
        not saved_values
        or not isinstance(expected_saved, Mapping)
        or set(saved_values) != set(expected_saved)
        or any(
            value.shape != (len(result_record["row_table"]), 784)
            or _core_array_sha256(value) != expected_saved[name]
            for name, value in saved_values.items()
        )
    ):
        raise RolloutCLIError("fused family saved-state authority is incomplete")
    saved_path = destination / "family_saved_states.npz"
    saved_artifact = _atomic_npz(saved_path, **saved_values)
    saved_artifact["path"] = saved_path.relative_to(run_dir).as_posix()
    record = _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-family-summary",
            "schema_version": 1,
            "family_name": family_name,
            "segment_name": segment_name,
            "row_table": result_record["row_table"],
            "row_keys_unique": int(
                len({row["row_key"] for row in result_record["row_table"]})
                == len(result_record["row_table"])
            ),
            "duplicate_canonical_path_ids_intentional": 1,
            "variant_in_rng_key": 0,
            "result": result_record,
            "saved_states_artifact": saved_artifact,
            "health": dict(health),
            "end_to_end_elapsed_seconds": float(end_to_end_elapsed_seconds),
            "resource_accounting_authority": 1,
            "passed": int(health.get("passed", 0)),
        }
    )
    atomic_write_json(summary_path, record)
    return record


def _commit_fused_trajectory_summary(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    role: str,
    horizon: str,
    anchor_step: int,
    variant: str,
    gain: float | None,
    states: Mapping[str, np.ndarray],
    diagnostic_parts: Sequence[tuple[Any, int]],
    family_bindings: Sequence[
        tuple[Path, Mapping[str, Any], int, Mapping[str, str]]
    ],
    expected_transitions: int,
) -> dict[str, Any]:
    variant_name = variant if gain is None else f"learned-gain-{gain:g}"
    output_dir = run_dir / role / horizon / variant_name
    summary_path = output_dir / "trajectory_summary.json"
    if summary_path.is_file():
        return _verify_trajectory_summary(summary_path, run_dir)
    normalized_states = {
        str(name): _state_row(value) for name, value in states.items()
    }
    final = normalized_states["final"]
    if (
        not np.isfinite(final).all()
        or np.any(final < 0.0)
        or abs(float(final.sum()) - 1.0) > MAXIMUM_MASS_ERROR
    ):
        raise RolloutCLIError("fused reverse trajectory produced an invalid state")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_artifact = _atomic_npz(
        output_dir / "selected_states.npz", **normalized_states
    )
    state_artifact["path"] = (
        output_dir / "selected_states.npz"
    ).relative_to(run_dir).as_posix()
    images = _render_states(
        run_dir,
        trajectory_key=f"{role}-{horizon}-{variant_name}",
        states=normalized_states,
    )
    _source, target = _source_arrays(run_dir)
    metrics = {
        name: _metrics_dict(value, target) for name, value in normalized_states.items()
    }
    diagnostics = _combine_fused_row_diagnostics(diagnostic_parts)
    family_healths = [
        dict(summary["health"]) for _path, summary, _row, _sources in family_bindings
    ]
    allocated_elapsed = math.fsum(
        float(summary.get("end_to_end_elapsed_seconds", 0.0))
        / max(1, len(summary.get("row_table", ())))
        for _path, summary, _row, _sources in family_bindings
    )
    fallback_time_fraction = max(
        (float(value.get("fallback_time_fraction", 0.0)) for value in family_healths),
        default=0.0,
    )
    # Reference authorization is row-local evidence.  Never manufacture a
    # scientific row by evenly dividing a family total: active/no-op patterns
    # and rare exact fallbacks can differ across evolving rows.
    row_transition_count = int(
        diagnostics.get("reference_transition_count", diagnostics.get("transition_count", -1))
    )
    row_certified_count = int(diagnostics.get("reference_certified_count", -1))
    row_fallback_count = int(diagnostics.get("reference_fallback_count", 0))
    row_active_count = int(
        diagnostics.get("reference_active_count", row_transition_count)
    )
    if (
        row_transition_count != int(expected_transitions)
        or row_active_count < 0
        or row_certified_count < 0
        or row_certified_count != row_active_count
        or row_fallback_count < 0
    ):
        raise RolloutCLIError(
            "fused per-row reference authorization telemetry is invalid",
            failure_domain="execution_integrity",
            failure_code="fused_per_row_reference_health_invalid",
        )
    # The deferred backend records fallback wall time at the shared launch
    # level.  Retain the conservative family time fraction for each row while
    # keeping transition/fallback counts exact and row-local.
    allocated_fallback_seconds = allocated_elapsed * fallback_time_fraction
    forbidden: dict[str, int] = {}
    for value in family_healths:
        for name, count in value.get("forbidden_counts", {}).items():
            forbidden[str(name)] = forbidden.get(str(name), 0) + int(count)
    health = _health_record(
        {
            "transition_count": int(expected_transitions),
            "active_count": row_active_count,
            "certified_count": row_certified_count,
            "certificate_fraction": (
                row_certified_count / row_active_count if row_active_count else 1.0
            ),
            "fallback_count": row_fallback_count,
            "fallback_fraction": (
                row_fallback_count / expected_transitions
                if expected_transitions
                else 0.0
            ),
            "fallback_seconds": allocated_fallback_seconds,
            "fallback_time_fraction": fallback_time_fraction,
            "elapsed_seconds": allocated_elapsed,
            "transitions_per_second": (
                expected_transitions / allocated_elapsed
                if allocated_elapsed > 0.0
                else float("nan")
            ),
            "maximum_simplex_mass_error": max(
                (
                    float(value.get("maximum_simplex_mass_error", 0.0))
                    for value in family_healths
                ),
                default=0.0,
            ),
            "maximum_pair_mass_error": max(
                (
                    float(value.get("maximum_pair_mass_error", 0.0))
                    for value in family_healths
                ),
                default=0.0,
            ),
            "maximum_cuda_memory_allocated": max(
                (
                    int(value.get("maximum_cuda_memory_allocated", 0))
                    for value in family_healths
                ),
                default=0,
            ),
            "total_cuda_memory_bytes": max(
                (
                    int(value.get("total_cuda_memory_bytes", 0))
                    for value in family_healths
                ),
                default=0,
            ),
            "forbidden_counts": forbidden,
        },
        expected_transition_count=int(expected_transitions),
        test_only=False,
    )
    binding_rows = []
    covered_states: set[str] = set()
    for path, summary, row_index, state_sources in family_bindings:
        saved_matrix_hashes = summary.get("result", {}).get(
            "saved_state_sha256", {}
        )
        saved_artifact = summary.get("saved_states_artifact", {})
        saved_path = run_dir / str(saved_artifact.get("path", ""))
        saved_arrays = _load_npz(saved_path) if saved_path.is_file() else {}
        sources = {str(name): str(source) for name, source in state_sources.items()}
        if (
            not sources
            or covered_states.intersection(sources)
            or not set(sources).issubset(normalized_states)
            or not isinstance(saved_matrix_hashes, Mapping)
            or not set(sources.values()).issubset(saved_matrix_hashes)
            or not set(sources.values()).issubset(saved_arrays)
        ):
            raise RolloutCLIError("fused family omitted selected-state commitments")
        row_hashes: dict[str, str] = {}
        for trajectory_name, family_name in sources.items():
            family_matrix = np.asarray(saved_arrays[family_name])
            if family_matrix.dtype != np.float64 or family_matrix.shape != (
                len(summary.get("row_table", ())),
                784,
            ):
                raise RolloutCLIError("fused family saved-state archive is malformed")
            family_row = _state_row(family_matrix[row_index])
            if not np.array_equal(family_row, normalized_states[trajectory_name]):
                raise RolloutCLIError("trajectory state differs from its fused family row")
            row_hashes[trajectory_name] = _array_sha256(family_row)
        covered_states.update(sources)
        binding_rows.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "sha256": summary["semantic_sha256"],
                "row_index": int(row_index),
                "row_key": summary["row_table"][row_index]["row_key"],
                "family_saved_state_artifact_sha256": saved_artifact.get("sha256"),
                "state_sources": sources,
                "family_saved_state_sha256": {
                    family_name: saved_matrix_hashes[family_name]
                    for family_name in sorted(set(sources.values()))
                },
                "row_saved_state_sha256": row_hashes,
            }
        )
    if covered_states != set(normalized_states):
        raise RolloutCLIError("trajectory states are not uniquely covered by fused families")
    summary = _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-reverse-trajectory-summary",
            "schema_version": 1,
            "role": role,
            "horizon": horizon,
            "anchor_step": int(anchor_step),
            "path_id": PATH_IDS[role],
            "variant": variant,
            "variant_name": variant_name,
            "gain": gain,
            "checkpoint_role": (
                "historical_validation_inspected_post_hoc_diagnostic"
                if variant == "learned"
                else None
            ),
            "rng_binding": {
                "root_seed": REVERSE_ROOT_SEED,
                "stream_role": STREAM_ROLES[role],
                "variant_in_rng_key": 0,
            },
            "fused_family_binding": binding_rows,
            "resource_accounting": "family_elapsed_allocated_once_across_rows",
            "final_state_sha256": _array_sha256(final),
            "selected_states_artifact": state_artifact,
            "images": images,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "health": health,
            "end_to_end_elapsed_seconds": allocated_elapsed,
            "passed_integrity": int(health["passed"]),
            "exploratory_post_hoc": 1,
        }
    )
    atomic_write_json(summary_path, summary)
    if not int(health["passed"]):
        atomic_write_json(
            output_dir / "trajectory_failure.json",
            _semantic(
                {
                    "schema": FUSED_RUN_SCHEMA + "-trajectory-failure",
                    "schema_version": 1,
                    "trajectory_summary_sha256": summary["semantic_sha256"],
                    "failure_domain": "execution_integrity",
                    "failure_code": "fused_reverse_exact_health_invalid",
                    "health": health,
                }
            ),
        )
        raise RolloutCLIError(
            f"fused reverse exact-health contract failed: {role}/{horizon}/{variant_name}",
            failure_domain="execution_integrity",
            failure_code="fused_reverse_exact_health_invalid",
        )
    return summary


_FUSED_PHASE_PREFIXES = (
    "reference_fraction_displacement",
    "control_fraction_displacement",
    "score",
    "logistic_shift",
)
_FUSED_INTERFACE_INVALID_FIELDS = (
    "input_invalid",
    "reference_fraction_invalid",
    "score_invalid",
    "logistic_shift_invalid",
    "state_invalid",
    "mass_invalid",
    "metadata_invalid",
)
_FUSED_CONTROLLER_INTEGER_FIELDS = (
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
)
_FUSED_CONTROLLER_FLOAT_FIELDS = (
    "score_squared_sum",
    "score_maximum_absolute",
    "unscaled_score_squared_sum",
    "unscaled_score_maximum_absolute",
    "score_rms",
    "unscaled_score_rms",
)


def _verify_fused_load_bearing_row_telemetry(
    record: Mapping[str, Any],
    *,
    expected_rows: Sequence[Mapping[str, Any]],
    reference_contract: str,
    expected_transition_count_per_row: int,
) -> None:
    """Fail closed on every row field used by integrity/mechanism reporting."""

    phase_rows = record.get("per_row_diagnostics")
    controller_rows = record.get("controller_diagnostics")
    if (
        not isinstance(phase_rows, list)
        or not isinstance(controller_rows, list)
        or len(phase_rows) != len(expected_rows)
        or len(controller_rows) != len(expected_rows)
    ):
        raise ArtifactCompatibilityError("fused row telemetry coverage changed")
    phase_int = {
        "transition_count",
        "boundary_fraction_count",
        *_FUSED_INTERFACE_INVALID_FIELDS,
        *(f"{prefix}_count" for prefix in _FUSED_PHASE_PREFIXES),
    }
    phase_float = {
        "maximum_pair_mass_error",
        "maximum_simplex_mass_error",
        *(
            f"{prefix}_{suffix}"
            for prefix in _FUSED_PHASE_PREFIXES
            for suffix in ("squared_sum", "maximum_absolute", "rms")
        ),
    }
    reference_int = (
        {
            "reference_transition_count",
            "reference_active_count",
            "reference_certified_count",
            "reference_fallback_count",
            "reference_unauthorized_count",
            "reference_invalid_count",
        }
        if reference_contract == "certified_exact"
        else {
            "reference_transition_count",
            "reference_active_count",
            "reference_structural_noop_count",
            "reference_approximation_count",
            "reference_invalid_count",
        }
    )
    reference_float = (
        {"reference_certificate_fraction"}
        if reference_contract == "certified_exact"
        else {"reference_maximum_candidate_bracket_width"}
    )
    for index, (phase, controller, expected) in enumerate(
        zip(phase_rows, controller_rows, expected_rows, strict=True)
    ):
        if not isinstance(phase, Mapping) or not isinstance(controller, Mapping):
            raise ArtifactCompatibilityError("fused row telemetry is malformed")
        row_key = expected.get("row_key")
        kind = expected.get("controller_kind")
        gain = expected.get("gain")
        if (
            phase.get("row_key") != row_key
            or controller.get("row_key") != row_key
            or controller.get("controller_kind") != kind
            or controller.get("gain") != gain
            or not phase_int.issubset(phase)
            or not phase_float.issubset(phase)
            or not reference_int.issubset(phase)
            or not reference_float.issubset(phase)
            or not set(_FUSED_CONTROLLER_INTEGER_FIELDS).issubset(controller)
            or not set(_FUSED_CONTROLLER_FLOAT_FIELDS).issubset(controller)
        ):
            raise ArtifactCompatibilityError(
                f"fused load-bearing telemetry row {index} changed"
            )
        integer_values = [
            *(phase[name] for name in sorted(phase_int | reference_int)),
            *(controller[name] for name in _FUSED_CONTROLLER_INTEGER_FIELDS),
        ]
        float_values = [
            *(phase[name] for name in sorted(phase_float | reference_float)),
            *(controller[name] for name in _FUSED_CONTROLLER_FLOAT_FIELDS),
        ]
        if (
            any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or int(value) < 0
                for value in integer_values
            )
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in float_values
            )
            or any(int(phase[name]) != 0 for name in _FUSED_INTERFACE_INVALID_FIELDS)
            or any(
                int(controller[name]) != 0
                for name in (
                    "clipping_count",
                    "floor_count",
                    "projection_count",
                    "nonfinite_score_count",
                )
            )
        ):
            raise ArtifactCompatibilityError(
                f"fused load-bearing telemetry row {index} is invalid"
            )
        for prefix in _FUSED_PHASE_PREFIXES:
            count = int(phase[f"{prefix}_count"])
            squared = float(phase[f"{prefix}_squared_sum"])
            expected_rms = math.sqrt(squared / count) if count else 0.0
            if not math.isclose(
                float(phase[f"{prefix}_rms"]),
                expected_rms,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ):
                raise ArtifactCompatibilityError(
                    f"fused {prefix} RMS telemetry changed"
                )
        controller_count = int(controller["score_count"])
        for prefix in ("score", "unscaled_score"):
            squared = float(controller[f"{prefix}_squared_sum"])
            expected_rms = (
                math.sqrt(squared / controller_count) if controller_count else 0.0
            )
            if not math.isclose(
                float(controller[f"{prefix}_rms"]),
                expected_rms,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ):
                raise ArtifactCompatibilityError(
                    f"fused controller {prefix} RMS telemetry changed"
                )
        if controller_count != int(phase["score_count"]) or any(
            not math.isclose(
                float(controller[f"score_{suffix}"]),
                float(phase[f"score_{suffix}"]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
            for suffix in ("squared_sum", "maximum_absolute", "rms")
        ):
            raise ArtifactCompatibilityError(
                "fused phase/controller score telemetry changed"
            )
        if reference_contract == "candidate_approximate_v1":
            transition = int(phase["reference_transition_count"])
            active = int(phase["reference_active_count"])
            noops = int(phase["reference_structural_noop_count"])
            if (
                phase.get("reference_certificate_fraction") != "not_applicable"
                or transition != active + noops
                or int(phase["reference_approximation_count"]) != active
                or int(phase["reference_invalid_count"]) != 0
            ):
                raise ArtifactCompatibilityError(
                    "candidate row approximation authority changed"
                )
        else:
            transition = int(phase["reference_transition_count"])
            active = int(phase["reference_active_count"])
            certified = int(phase["reference_certified_count"])
            noops = transition - active
            fraction = float(phase["reference_certificate_fraction"])
            expected_fraction = certified / active if active else 1.0
            if (
                noops < 0
                or certified != active
                or int(phase["reference_fallback_count"]) < 0
                or int(phase["reference_unauthorized_count"]) != 0
                or int(phase["reference_invalid_count"]) != 0
                or not math.isclose(
                    fraction,
                    expected_fraction,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            ):
                raise ArtifactCompatibilityError(
                    "exact row certificate authority changed"
                )
        # Every fused scientific row traverses the same frozen phase schedule.
        # Binding only global sums would permit authority to be moved between
        # rows while preserving both global and row-local arithmetic.
        row_transition_count = int(phase["transition_count"])
        if (
            row_transition_count != expected_transition_count_per_row
            or transition != expected_transition_count_per_row
        ):
            raise ArtifactCompatibilityError(
                "fused per-row transition authority changed"
            )


def _verify_fused_shard_health(
    record: Mapping[str, Any],
    *,
    expected_transitions: int,
    row_count: int,
    expected_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, int]:
    if expected_rows is not None:
        if expected_transitions % row_count:
            raise ArtifactCompatibilityError(
                "fused transition count is not row-balanced"
            )
        _verify_fused_load_bearing_row_telemetry(
            record,
            expected_rows=expected_rows,
            reference_contract="certified_exact",
            expected_transition_count_per_row=expected_transitions // row_count,
        )
    diagnostics = record.get("diagnostics", {})
    reference = diagnostics.get("reference", {}) if isinstance(diagnostics, Mapping) else {}
    per_row = record.get("per_row_diagnostics", ())
    controller_rows = record.get("controller_diagnostics", ())
    forbidden = reference.get("forbidden_counts", {}) if isinstance(reference, Mapping) else {}
    exact_forbidden_names = {
        "resource_cap_count", "invalid_density_count", "approximation_count",
        "clipping_count", "correction_count", "floor_count", "limiter_count",
        "projection_count", "renormalization_count", "nonfinite_count",
    }
    invalid_names = {
        "input_invalid",
        "reference_fraction_invalid",
        "score_invalid",
        "logistic_shift_invalid",
        "state_invalid",
        "mass_invalid",
        "metadata_invalid",
    }
    exact_count_names = (
        "reference_transition_count",
        "reference_active_count",
        "reference_certified_count",
        "reference_fallback_count",
        "reference_unauthorized_count",
        "reference_invalid_count",
    )
    integer_scalars = [
        diagnostics.get("transition_count") if isinstance(diagnostics, Mapping) else None,
        reference.get("transition_count") if isinstance(reference, Mapping) else None,
        reference.get("active_count") if isinstance(reference, Mapping) else None,
        (
            reference["structural_noop_count"]
            if isinstance(reference, Mapping) and "structural_noop_count" in reference
            else expected_transitions
            - int(reference.get("active_count", -1))
            if isinstance(reference, Mapping)
            else None
        ),
        reference.get("certified_count") if isinstance(reference, Mapping) else None,
        reference.get("fallback_count") if isinstance(reference, Mapping) else None,
        *(forbidden.values() if isinstance(forbidden, Mapping) else ()),
        *(
            row.get(name)
            for row in per_row
            if isinstance(row, Mapping)
            for name in exact_count_names
        ),
        *(
            row[name]
            for row in per_row
            if isinstance(row, Mapping)
            for name in sorted(invalid_names)
            if name in row
        ),
    ]
    if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_scalars):
        raise ArtifactCompatibilityError("fused committed shard count semantics changed")
    active = int(reference.get("active_count", -1)) if isinstance(reference, Mapping) else -1
    # The frozen exact deferred reference predates the candidate backend's
    # explicit structural-noop field.  Exact reverse exposures are positive, so
    # the only truthful backward-compatible value is the arithmetic complement.
    noops = (
        int(reference["structural_noop_count"])
        if isinstance(reference, Mapping) and "structural_noop_count" in reference
        else expected_transitions - active
    )
    certified = int(reference.get("certified_count", -1)) if isinstance(reference, Mapping) else -1
    fallback = int(reference.get("fallback_count", -1)) if isinstance(reference, Mapping) else -1
    numeric_controller_values = [
        value
        for row in controller_rows
        if isinstance(row, Mapping)
        for value in row.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if (
        not isinstance(diagnostics, Mapping)
        or not isinstance(reference, Mapping)
        or not isinstance(per_row, list)
        or len(per_row) != row_count
        or not isinstance(controller_rows, list)
        or len(controller_rows) != row_count
        or int(diagnostics.get("transition_count", -1)) != expected_transitions
        or int(reference.get("transition_count", -1)) != expected_transitions
        or active < 0
        or noops < 0
        or active + noops != expected_transitions
        or certified != active
        or fallback < 0
        or float(reference.get("certificate_fraction", 0.0)) != 1.0
        or float(diagnostics.get("certificate_fraction", 0.0)) != 1.0
        or fallback / expected_transitions > MAXIMUM_FALLBACK_FRACTION
        or not isinstance(forbidden, Mapping)
        or set(forbidden) != exact_forbidden_names
        or any(int(value) for value in forbidden.values())
        or any(
            int(row.get(name, 0))
            for row in per_row
            if isinstance(row, Mapping)
            for name in invalid_names
        )
        or any(not isinstance(row, Mapping) for row in per_row)
        or sum(int(row.get("reference_transition_count", 0)) for row in per_row)
        != expected_transitions
        or sum(int(row.get("reference_active_count", 0)) for row in per_row) != active
        or sum(
            int(row["reference_structural_noop_count"])
            if "reference_structural_noop_count" in row
            else int(row["reference_transition_count"])
            - int(row["reference_active_count"])
            for row in per_row
        )
        != noops
        or sum(int(row.get("reference_certified_count", 0)) for row in per_row)
        != certified
        or sum(int(row.get("reference_fallback_count", 0)) for row in per_row)
        != fallback
        or any(
            int(row.get("reference_unauthorized_count", 0))
            or int(row.get("reference_invalid_count", 0))
            for row in per_row
        )
        or not math.isfinite(float(diagnostics.get("maximum_mass_error", float("nan"))))
        or float(diagnostics.get("maximum_mass_error", float("inf"))) > MAXIMUM_MASS_ERROR
        or any(not math.isfinite(float(value)) for value in numeric_controller_values)
        or int(diagnostics.get("maximum_launch_lanes", row_count * 392)) > 4096
    ):
        raise ArtifactCompatibilityError("fused committed shard health changed")
    return {
        "transition_count": expected_transitions,
        "active_count": active,
        "certified_count": certified,
        "fallback_count": fallback,
    }


def _verify_fused_family_prefix_impl(
    shard_root: Path,
    *,
    initial_state: np.ndarray,
    sequence: Sequence[tuple[int, int]],
    row_specs: Sequence[Any],
    controller_binding: Mapping[str, Any],
    rng_binding: Mapping[str, Any],
    family_name: str,
    segment_name: str,
    reference_contract: str = "certified_exact",
) -> dict[str, Any]:
    """Read-only verification of every committed shard before any resume write."""

    rows = [_record(spec.to_record()) for spec in row_specs]
    normalized_sequence = tuple((int(step), int(phase)) for step, phase in sequence)
    json_paths = sorted(
        path
        for path in shard_root.glob("shard-*.json")
        if re.fullmatch(r"shard-\d{4}\.json", path.name)
    )
    previous = _core_array_sha256(np.ascontiguousarray(initial_state, dtype=np.float64))
    elapsed = 0.0
    transitions = 0
    for index, record_path in enumerate(json_paths):
        if record_path.name != f"shard-{index:04d}.json":
            raise ArtifactCompatibilityError("fused committed shard prefix has a gap")
        state_path = record_path.with_suffix(".npz")
        record = _load_semantic(record_path, "fused committed shard prefix")
        shard_sequence = normalized_sequence[index * 56 : (index + 1) * 56]
        expected_transitions = len(shard_sequence) * 4 * len(rows) * 392
        execution = record.get("execution_plan", {})
        if (
            not shard_sequence
            or int(record.get("committed", 0)) != 1
            or record.get("family_name") != family_name
            or record.get("segment_name") != segment_name
            or int(record.get("shard_index", -1)) != index
            or record.get("row_table") != rows
            or record.get("row_keys") != [row["row_key"] for row in rows]
            or record.get("canonical_path_ids")
            != [int(row["canonical_path_id"]) for row in rows]
            or int(record.get("microsteps", -1)) != MICROSTEPS
            or int(record.get("label", -1)) != 3
            or int(record.get("variant_in_rng_key", -1)) != 0
            or record.get("controller_binding_sha256")
            != config_fingerprint(controller_binding)
            or record.get("rng_binding_sha256") != config_fingerprint(rng_binding)
            or record.get("input_state_sha256") != previous
            or record.get("sequence_start") != list(shard_sequence[0])
            or record.get("sequence_end") != list(shard_sequence[-1])
            or record.get("sequence_sha256")
            != config_fingerprint([list(item) for item in shard_sequence])
            or not isinstance(execution, Mapping)
            or int(execution.get("shard_index", -1)) != index
            or execution.get("sequence")
            != [list(item) for item in shard_sequence]
            or int(execution.get("row_count", -1)) != len(rows)
            or int(execution.get("transition_count", -1)) != expected_transitions
            or execution.get("input_state_sha256") != previous
            or int(record.get("transition_count", -1)) != expected_transitions
            or not state_path.is_file()
            or record.get("state_file_sha256") != file_fingerprint(state_path)
            or int(record.get("state_file_size", -1)) != int(state_path.stat().st_size)
        ):
            raise ArtifactCompatibilityError("fused committed shard prefix changed")
        arrays = _load_npz(state_path)
        state = np.asarray(arrays.get("state"))
        if (
            set(arrays) != {"state"}
            or state.dtype != np.float64
            or state.shape != (len(rows), 784)
            or record.get("output_state_sha256") != _core_array_sha256(state)
            or not np.isfinite(state).all()
            or np.any(state < 0.0)
            or float(np.max(np.abs(state.sum(axis=1) - 1.0)))
            > MAXIMUM_MASS_ERROR
        ):
            raise ArtifactCompatibilityError("fused committed shard state changed")
        if reference_contract == "certified_exact":
            if "reference_contract" in record:
                raise ArtifactCompatibilityError(
                    "exact shard acquired an approximate reference contract"
                )
            _verify_fused_shard_health(
                record,
                expected_transitions=expected_transitions,
                row_count=len(rows),
                expected_rows=rows,
            )
        elif reference_contract == "candidate_approximate":
            if record.get("reference_contract") != "candidate_approximate_v1":
                raise ArtifactCompatibilityError("candidate shard contract changed")
            _verify_candidate_fused_shard_health(
                record,
                expected_transitions=expected_transitions,
                row_count=len(rows),
                expected_rows=rows,
            )
        else:
            raise ArtifactCompatibilityError("fused prefix reference contract changed")
        reference = record.get("diagnostics", {}).get("reference", {})
        elapsed_value = record.get("elapsed_seconds")
        peak_value = (
            reference.get("maximum_cuda_memory_allocated")
            if isinstance(reference, Mapping)
            else None
        )
        total_value = (
            reference.get("total_cuda_memory_bytes")
            if isinstance(reference, Mapping)
            else None
        )
        if (
            not isinstance(elapsed_value, (int, float))
            or isinstance(elapsed_value, bool)
            or not math.isfinite(float(elapsed_value))
            or float(elapsed_value) <= 0.0
            or not isinstance(peak_value, int)
            or isinstance(peak_value, bool)
            or not isinstance(total_value, int)
            or isinstance(total_value, bool)
            or peak_value < 0
            or total_value < 0
            or peak_value > total_value
            or (total_value == 0 and peak_value != 0)
        ):
            raise ArtifactCompatibilityError(
                "fused committed shard resource telemetry changed"
            )
        previous = _core_array_sha256(state)
        elapsed += float(record.get("elapsed_seconds", 0.0))
        transitions += expected_transitions
    return {
        "shard_count": len(json_paths),
        "elapsed_seconds": elapsed,
        "transition_count": transitions,
        "last_state_sha256": previous,
    }


def _verify_fused_family_prefix(
    shard_root: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Normalize all untrusted-prefix parse failures to incompatibility."""

    try:
        return _verify_fused_family_prefix_impl(shard_root, **kwargs)
    except ArtifactCompatibilityError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ArtifactCompatibilityError(
            "fused committed shard prefix is malformed"
        ) from exc


def _write_fused_family_failure_evidence(
    run_dir: Path,
    *,
    destination: Path,
    shard_root: Path,
    family_name: str,
    segment_name: str,
    rows: Sequence[tuple[str, str, str, float | None]],
    canonical_path_id: int,
    initial_state: np.ndarray,
    sequence: Sequence[tuple[int, int]],
    exc: BaseException,
) -> None:
    """Promote the last valid fused row states after a stopped family."""

    committed = sorted(
        path
        for path in shard_root.glob("shard-*.json")
        if re.fullmatch(r"shard-\d{4}\.json", path.name)
    )
    last_state = np.ascontiguousarray(initial_state, dtype=np.float64)
    if committed:
        last_arrays = _load_npz(committed[-1].with_suffix(".npz"))
        candidate = np.asarray(last_arrays.get("state"))
        if candidate.dtype != np.float64 or candidate.shape != last_state.shape:
            raise ArtifactCompatibilityError("last committed fused failure state changed")
        last_state = np.ascontiguousarray(candidate)
    failure_files = sorted(shard_root.glob("shard-*.failure.json"))
    next_offset = len(committed) * 56
    next_coordinate = (
        list(sequence[next_offset]) if next_offset < len(sequence) else None
    )
    for row_index, (row_key, variant, horizon, gain) in enumerate(rows):
        variant_name = variant if gain is None else f"learned-gain-{gain:g}"
        row_root = destination / "failure_rows" / row_key
        states = {
            "start": _state_row(initial_state[row_index]),
            "last_valid": _state_row(last_state[row_index]),
        }
        state_artifact = _atomic_npz(row_root / "last_valid_states.npz", **states)
        state_artifact["path"] = (
            row_root / "last_valid_states.npz"
        ).relative_to(run_dir).as_posix()
        images = _render_states(
            run_dir,
            trajectory_key=f"failed-{family_name}-{row_key}",
            states=states,
        )
        record = _semantic(
            {
                "schema": FUSED_RUN_SCHEMA + "-trajectory-failure",
                "schema_version": 1,
                "family_name": family_name,
                "segment_name": segment_name,
                "row_key": row_key,
                "row_index": row_index,
                "canonical_path_id": int(canonical_path_id),
                "variant": variant,
                "variant_name": variant_name,
                "horizon": horizon,
                "gain": gain,
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "last_committed_shard_count": len(committed),
                "next_uncommitted_coordinate": next_coordinate,
                "last_valid_state_sha256": _array_sha256(states["last_valid"]),
                "state_artifact": state_artifact,
                "images": images,
                "core_shard_failure_artifacts": [
                    path.relative_to(run_dir).as_posix() for path in failure_files
                ],
                "numerically_interpretable_partial_evidence": int(
                    np.isfinite(last_state).all()
                    and np.all(last_state >= 0.0)
                    and np.max(np.abs(last_state.sum(axis=1) - 1.0))
                    <= MAXIMUM_MASS_ERROR
                ),
            }
        )
        atomic_write_json(row_root / "trajectory_failure.json", record)


def _run_and_commit_fused_family(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    initial_state: np.ndarray,
    sequence: Sequence[tuple[int, int]],
    destination: Path,
    family_name: str,
    segment_name: str,
    rows: Sequence[tuple[str, str, str, float | None]],
    canonical_path_id: int,
    stream_role: str,
    expected_transition_count: int,
    capture_coordinates: Mapping[tuple[int, int], str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    from mnist.d0_jacobi_rb_tangent_fused import run_fused_reverse_family

    specs, bank, controller_binding = _fused_controller_family(
        run_dir,
        args,
        rows=rows,
        canonical_path_id=canonical_path_id,
    )
    state = np.ascontiguousarray(initial_state, dtype=np.float64)
    if state.shape == (784,):
        state = np.repeat(state[None, :], len(specs), axis=0)
    if state.shape != (len(specs), 784):
        raise RolloutCLIError("fused family initial state shape changed")
    rng_binding = {
        "root_seed": REVERSE_ROOT_SEED,
        "stream_role": stream_role,
        "variant_in_rng_key": 0,
    }
    planning_profile = {
        2: "reverse_p2",
        3: "reverse_p3",
        6: "reverse_p6",
    }.get(len(specs))
    if planning_profile is None:
        raise RolloutCLIError("fused family row count has no frozen resource profile")
    shard_root = destination / "fused_families" / family_name / segment_name
    prefix = _verify_fused_family_prefix(
        shard_root,
        initial_state=state,
        sequence=sequence,
        row_specs=specs,
        controller_binding=controller_binding,
        rng_binding=rng_binding,
        family_name=family_name,
        segment_name=segment_name,
    )
    prior_elapsed = float(prefix["elapsed_seconds"])
    prior_transitions = int(prefix["transition_count"])
    remaining_transitions = int(expected_transition_count) - prior_transitions
    if remaining_transitions < 0:
        raise ArtifactCompatibilityError("fused committed transition count exceeds plan")
    remaining_shards = len(tuple(sequence)) // 56 - int(prefix["shard_count"])
    if remaining_shards < 0:
        raise ArtifactCompatibilityError("fused committed shard count exceeds plan")
    device = torch.device(args.device)
    profile = JacobiRBCudaProfile()
    backend_key = f"{device}:{config_fingerprint(profile.to_dict())}"
    backend_was_prepared = backend_key in _PREPARED_FUSED_BACKENDS
    prepared = _prepared_fused_reference(device, profile)
    if not backend_was_prepared and (run_dir / "preflight_gate.json").is_file():
        setup_path = run_dir / "metrics/resume_fused_backend_setup.json"
        previous_elapsed = 0.0
        previous_count = 0
        if setup_path.is_file():
            previous = _load_semantic(setup_path, "resume fused backend setup")
            previous_elapsed = float(previous.get("elapsed_seconds", 0.0))
            previous_count = int(previous.get("preparation_count", 0))
        setup = _semantic(
            {
                "schema": FUSED_RUN_SCHEMA + "-resume-backend-setup",
                "schema_version": 1,
                "elapsed_seconds": previous_elapsed
                + float(_PREPARED_FUSED_ELAPSED.get(backend_key, 0.0)),
                "preparation_count": previous_count + 1,
                "reason": "cold objective-process exact deferred backend preparation",
                "included_in_main_resource_ledger": 1,
            }
        )
        atomic_write_json(setup_path, setup)
    # Backend preparation can be material on a cold resume.  Commit it to the
    # ledger first, then decide whether the still-uncommitted family tail fits.
    # Checking before preparation would permit a resume to exceed the cap by
    # exactly the hidden compile/setup duration.
    _ensure_execution_budget(
        run_dir,
        role=("replication" if family_name.startswith("replication") else family_name.split("-")[0]),
        additional_transitions=remaining_transitions,
        test_only=False,
        operation=f"{family_name}-{segment_name}-pre",
        profile_name=planning_profile,
        additional_persisted_bytes=(
            remaining_shards
            * _profile_shard_storage_estimate(run_dir, planning_profile)
        ),
    )
    family_role = (
        "replication"
        if family_name.startswith("replication")
        else family_name.split("-")[0]
    )

    def before_uncommitted_shard(plan: Any) -> None:
        _ensure_execution_budget(
            run_dir,
            role=family_role,
            additional_transitions=int(plan.transition_count),
            test_only=False,
            operation=(
                f"{family_name}-{segment_name}-shard-{int(plan.shard_index):04d}"
            ),
            profile_name=planning_profile,
            additional_persisted_bytes=_profile_shard_storage_estimate(
                run_dir, planning_profile
            ),
        )

    reference_factory = _fused_reference_factory(
        prepared=prepared,
        profile=profile,
        stream_role=stream_role,
    )
    started = time.perf_counter()
    try:
        result = run_fused_reverse_family(
            torch.as_tensor(
                np.array(state, copy=True, order="C"),
                dtype=torch.float64,
                device=device,
            ).contiguous(),
            sequence=tuple(sequence),
            output_dir=destination,
            family_name=family_name,
            segment_name=segment_name,
            row_specs=specs,
            controller_bank=bank,
            reference_factory=reference_factory,
            controller_binding=controller_binding,
            rng_binding=rng_binding,
            label=3,
            microsteps=MICROSTEPS,
            device=device,
            capture_coordinates=capture_coordinates,
            before_uncommitted_shard=before_uncommitted_shard,
        )
    except BaseException as exc:
        _write_fused_family_failure_evidence(
            run_dir,
            destination=destination,
            shard_root=shard_root,
            family_name=family_name,
            segment_name=segment_name,
            rows=rows,
            canonical_path_id=canonical_path_id,
            initial_state=state,
            sequence=sequence,
            exc=exc,
        )
        raise
    # On resume, the current timer covers verification plus the uncommitted
    # tail.  Add the verified prior device elapsed so a process restart can
    # never erase already consumed resource authority.  On uninterrupted
    # execution prior_elapsed is zero and the timer includes all commits.
    end_to_end = prior_elapsed + (time.perf_counter() - started)
    health = _fused_family_health(
        result,
        expected_transition_count=int(expected_transition_count),
        test_only=False,
        end_to_end_elapsed_seconds=end_to_end,
    )
    summary = _commit_fused_family_summary(
        run_dir,
        destination=destination,
        result=result,
        family_name=family_name,
        segment_name=segment_name,
        health=health,
        end_to_end_elapsed_seconds=end_to_end,
    )
    if not int(health["passed"]):
        raise RolloutCLIError(
            f"fused family health failed: {family_name}/{segment_name}",
            failure_domain="execution_integrity",
            failure_code="fused_reverse_exact_health_invalid",
        )
    _ensure_execution_budget(
        run_dir,
        role=family_role,
        additional_transitions=0,
        test_only=False,
        operation=f"{family_name}-{segment_name}-post",
        profile_name=planning_profile,
    )
    return result, summary


def _replication_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    evaluation_gate = _load_semantic(run_dir / "evaluation_gate.json", "evaluation gate")
    if int(evaluation_gate.get("passed", 0)) != 1:
        raise ArtifactCompatibilityError("replication requires passing evaluation integrity")
    selection = _verify_development_selection(run_dir)
    decision = _load_semantic(run_dir / "exploratory_decision.json", "exploratory decision")
    if float(decision.get("selected_gain", float("nan"))) != float(selection["selected_gain"]):
        raise ArtifactCompatibilityError("replication selection binding changed")
    _verify_forward_summary(run_dir, "evaluation")
    evaluation_paths = sorted((run_dir / "evaluation").glob("*/*/trajectory_summary.json"))
    if len(evaluation_paths) != 6:
        raise ArtifactCompatibilityError("replication requires the complete evaluation family")
    evaluated = [_verify_trajectory_summary(path, run_dir) for path in evaluation_paths]
    by_key = {(row["horizon"], row["variant"]): row for row in evaluated}
    for horizon in ("short", "full"):
        measured = _paired_metrics(
            zero=by_key[(horizon, "zero")],
            learned=by_key[(horizon, "learned")],
            oracle=by_key[(horizon, "oracle")],
        )
        if decision.get(horizon) != measured:
            raise ArtifactCompatibilityError("replication decision metrics changed")
    resource_projection = _load_semantic(
        run_dir / "metrics/replication_resource_projection.json",
        "replication resource projection",
    )
    if (
        decision.get("replication_resource_projection_sha256")
        != resource_projection.get("semantic_sha256")
    ):
        raise ArtifactCompatibilityError("replication resource projection binding changed")
    full_metrics = _paired_metrics(
        zero=by_key[("full", "zero")],
        learned=by_key[("full", "learned")],
        oracle=by_key[("full", "oracle")],
    )
    recomputed_authorized = bool(
        float(full_metrics["oracle_improvement_over_zero"]) > 0.0
        and float(full_metrics["learned_improvement_over_zero"]) > 0.0
        and int(resource_projection.get("passed", 0)) == 1
    )
    if int(decision.get("replication_authorized", -1)) != int(recomputed_authorized):
        raise ArtifactCompatibilityError("replication authorization changed")
    if not recomputed_authorized:
        record = _semantic(
            {
                "schema": _schema(args) + "-replication-gate",
                "schema_version": 1,
                "evaluation_status": "not_evaluated",
                "reason": "predeclared positive-evaluation condition not met",
                "passed": 1,
                "replication_performed": 0,
            }
        )
        atomic_write_json(run_dir / "replication_gate.json", record)
        _status(
            run_dir,
            state="ready_for_report",
            stage="replication",
            decision=decision["decision"],
            scientific_evidence_complete=1,
        )
        return record
    config = _load_semantic(run_dir / "scientific_config.json", "scientific configuration")
    anchor = int(config["full_anchor"])
    _forward_role(
        run_dir,
        args,
        role="replication",
        path_id=PATH_IDS["replication"],
        anchors=(anchor,),
    )
    selected_gain = float(decision["selected_gain"])
    zero = _trajectory(
        run_dir, args, role="replication", horizon="full", anchor_step=anchor, variant="zero"
    )
    learned = _trajectory(
        run_dir,
        args,
        role="replication",
        horizon="full",
        anchor_step=anchor,
        variant="learned",
        gain=selected_gain,
    )
    zero_error, learned_error = _squared_l2(zero), _squared_l2(learned)
    agrees = learned_error < zero_error
    updated = _semantic(
        {
            **{key: value for key, value in decision.items() if key != "semantic_sha256"},
            "decision": (
                "learned_full_replication_agrees" if agrees else "learned_full_replication_disagrees"
            ),
            "replication_performed": 1,
            "replication_learned_improvement_over_zero": zero_error - learned_error,
            "replication_direction_agrees": int(agrees),
            "recommended_next_action": (
                "verify frozen checkpoint/gain at M=8"
                if agrees else "treat single evaluation improvement as unstable and pivot learner"
            ),
        }
    )
    atomic_write_json(run_dir / "exploratory_decision.json", updated)
    gate = _semantic(
        {
            "schema": _schema(args) + "-replication-gate",
            "schema_version": 1,
            "gate_type": "execution_integrity",
            "replication_performed": 1,
            "direction_agrees_diagnostic": int(agrees),
            "passed": 1,
        }
    )
    atomic_write_json(run_dir / "replication_gate.json", gate)
    _status(
        run_dir,
        state="ready_for_report",
        stage="replication",
        decision=updated["decision"],
        scientific_evidence_complete=1,
    )
    return gate


_singleton_development_stage = _development_stage
_singleton_evaluation_stage = _evaluation_stage
_singleton_replication_stage = _replication_stage


def _test_fused_family_record(
    run_dir: Path,
    *,
    destination: Path,
    family_name: str,
    segment_name: str,
    summaries: Sequence[Mapping[str, Any]],
    canonical_path_id: int,
    transition_count: int,
) -> dict[str, Any]:
    rows = [
        {
            "row_key": f"{row['horizon']}-{row['variant_name']}",
            "canonical_path_id": int(canonical_path_id),
            "variant": row["variant"],
            "gain": row.get("gain"),
            "horizon": row["horizon"],
            "trajectory_semantic_sha256": row["semantic_sha256"],
            "final_state_sha256": row["final_state_sha256"],
        }
        for row in summaries
    ]
    record = _semantic(
        {
            "schema": FUSED_TEST_RUN_SCHEMA + "-family-summary",
            "schema_version": 1,
            "family_name": family_name,
            "segment_name": segment_name,
            "row_table": rows,
            "row_keys_unique": int(len({row["row_key"] for row in rows}) == len(rows)),
            "duplicate_canonical_path_ids_intentional": 1,
            "variant_in_rng_key": 0,
            "transition_count": int(transition_count),
            "certificate_fraction": 1.0,
            "restart_chain_valid": 1,
            "test_only": 1,
            "passed": 1,
        }
    )
    atomic_write_json(destination / "family_summary.json", record)
    return record


def _fused_cuda_development_stage(
    run_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    if not int(_load_semantic(run_dir / "forward_gate.json", "forward gate").get("passed", 0)):
        raise RolloutCLIError("development requires fresh forward anchors")
    config = _load_semantic(run_dir / "scientific_config.json", "scientific configuration")
    anchor_step = int(config["short_anchor"])
    anchor = _anchor_state(run_dir, "development", anchor_step)
    rows = [
        ("development-short-zero", "zero", "short", None),
        *[
            (
                f"development-short-learned-{_fused_gain_token(gain)}",
                "learned",
                "short",
                float(gain),
            )
            for gain in LEARNED_GAINS
        ],
        ("development-short-oracle", "oracle", "short", None),
    ]
    steps = anchor_step + 1
    captures = {
        (anchor_step - steps // 4 + 1, 0): "progress_25",
        (anchor_step - steps // 2 + 1, 0): "progress_50",
        (anchor_step - 3 * steps // 4 + 1, 0): "progress_75",
        (0, 0): "final",
    }
    destination = run_dir / "development/short/fused_family"
    result, family = _run_and_commit_fused_family(
        run_dir,
        args,
        initial_state=anchor,
        sequence=_reverse_sequence(anchor_step),
        destination=destination,
        family_name="development-short",
        segment_name="complete",
        rows=rows,
        canonical_path_id=PATH_IDS["development"],
        stream_role=STREAM_ROLES["development"],
        expected_transition_count=8_429_568,
        capture_coordinates=captures,
    )
    family_path = destination / "family_summary.json"
    summaries: list[dict[str, Any]] = []
    for row_index, (_row_key, variant, _horizon, gain) in enumerate(rows):
        states = {
            name: np.asarray(value)[row_index]
            for name, value in _field(result, "saved_states", {}).items()
        }
        summaries.append(
            _commit_fused_trajectory_summary(
                run_dir,
                args,
                role="development",
                horizon="short",
                anchor_step=anchor_step,
                variant=variant,
                gain=gain,
                states=states,
                diagnostic_parts=((result, row_index),),
                family_bindings=(
                    (
                        family_path,
                        family,
                        row_index,
                        {name: name for name in states},
                    ),
                ),
                expected_transitions=(anchor_step + 1) * 7 * 4 * 392,
            )
        )
    zero = summaries[0]
    learned = summaries[1:5]
    oracle = summaries[5]
    ranked = sorted(
        ((_squared_l2(row), float(row["gain"]), row) for row in learned),
        key=lambda item: (item[0], item[1]),
    )
    selected_error, selected_gain, selected = ranked[0]
    zero_error = _squared_l2(zero)
    oracle_error = _squared_l2(oracle)
    input_binding = _load_semantic(
        run_dir / "input_bindings/input_binding.json", "input binding"
    )
    commitments = {
        str(row["variant_name"]): {
            "trajectory_semantic_sha256": row["semantic_sha256"],
            "selected_states_sha256": row["selected_states_artifact"]["sha256"],
            "final_state_sha256": row["final_state_sha256"],
        }
        for row in summaries
    }
    selection = _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-development-selection",
            "schema_version": 1,
            "selection_criterion": "minimum final raw squared L2; ties smaller gain",
            "gain_grid": list(LEARNED_GAINS),
            "learned_endpoint_squared_l2": {
                str(row["gain"]): _squared_l2(row) for row in learned
            },
            "zero_endpoint_squared_l2": zero_error,
            "oracle_endpoint_squared_l2": oracle_error,
            "selected_gain": selected_gain,
            "selected_trajectory_sha256": selected["semantic_sha256"],
            "trajectory_commitments": commitments,
            "fused_family_sha256": family["semantic_sha256"],
            "selected_learned_beats_zero": int(selected_error < zero_error),
            "oracle_beats_zero": int(oracle_error < zero_error),
            "checkpoint_file_sha256": input_binding["checkpoint_file_sha256"],
            "source_image_array_sha256": input_binding["source_image_array_sha256"],
            "mixed_target_array_sha256": input_binding["mixed_target_array_sha256"],
            "anchor_sha256": _load_semantic(
                run_dir / "forward/development/forward_summary.json"
            )["anchors"][str(anchor_step)]["state_sha256"],
            "scientific_config_sha256": config["semantic_sha256"],
            "evaluation_evidence_opened": 0,
            "exploratory_post_hoc": 1,
            "committed_before_evaluation": 1,
        }
    )
    atomic_write_json(run_dir / "development/development_selection.json", selection)
    _paired_progress_metrics(
        run_dir,
        args,
        role="development",
        horizon="short",
        zero=zero,
        learned=selected,
        oracle=oracle,
    )
    _write_development_contact_sheet(run_dir, selected_gain)
    oracle_passed = oracle_error < zero_error
    gate = _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-development-gate",
            "schema_version": 1,
            "gate_type": "interpretability",
            "downstream_action_controlled": "opening fresh evaluation rollout",
            "exact_proposition_tested": (
                "source-informed target-pull oracle lowers development short-suffix final L2"
            ),
            "oracle_beats_zero": int(oracle_passed),
            "selected_gain": selected_gain,
            "selected_learned_beats_zero_diagnostic": int(selected_error < zero_error),
            "passed": int(oracle_passed),
            "failure_does_not_mean": "the learned score has no useful signal",
        }
    )
    atomic_write_json(run_dir / "development_gate.json", gate)
    if not oracle_passed:
        decision = _semantic(
            {
                "schema": FUSED_RUN_SCHEMA + "-decision",
                "schema_version": 1,
                "decision": "development_oracle_control_failed",
                "recommended_next_action": "repair controller/oracle/reference composition",
                "evaluation_performed": 0,
                "replication_performed": 0,
                "exploratory": 1,
            }
        )
        atomic_write_json(run_dir / "exploratory_decision.json", decision)
        _status(
            run_dir,
            state="complete",
            stage="development",
            decision=decision["decision"],
            scientific_evidence_complete=1,
        )
        _write_report(run_dir)
        _finalize_artifacts(run_dir)
        return gate
    _status(run_dir, state="ready_for_evaluation", stage="development")
    return gate


def _fused_development_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.test_only:
        gate = _singleton_development_stage(run_dir, args)
        ordered_names = [
            "zero",
            *(f"learned-gain-{gain:g}" for gain in LEARNED_GAINS),
            "oracle",
        ]
        summaries = [
            _load_semantic(
                run_dir / "development/short" / name / "trajectory_summary.json",
                "development trajectory summary",
            )
            for name in ordered_names
        ]
        _test_fused_family_record(
            run_dir,
            destination=run_dir / "development/short/fused_family",
            family_name="development-short",
            segment_name="complete",
            summaries=summaries,
            canonical_path_id=PATH_IDS["development"],
            transition_count=8_429_568,
        )
        return gate
    return _fused_cuda_development_stage(run_dir, args)


def _finalize_fused_evaluation(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    selected_gain: float,
) -> dict[str, Any]:
    for horizon in ("short", "full"):
        variants = results[horizon]
        _paired_progress_metrics(
            run_dir,
            args,
            role="evaluation",
            horizon=horizon,
            zero=variants["zero"],
            learned=variants["learned"],
            oracle=variants["oracle"],
        )
    short = _paired_metrics(**results["short"])
    full = _paired_metrics(**results["full"])
    atomic_write_json(
        run_dir / "metrics/evaluation_paired_metrics.json",
        _semantic(
            {
                "schema": FUSED_RUN_SCHEMA + "-evaluation-paired-metrics",
                "schema_version": 1,
                "selected_gain": selected_gain,
                "short": short,
                "full": full,
                "threshold_type": "diagnostic",
                "confirmatory_inference": 0,
            }
        ),
    )
    oracle_short = float(short["oracle_improvement_over_zero"]) > 0.0
    oracle_full = float(full["oracle_improvement_over_zero"]) > 0.0
    learned_short = float(short["learned_improvement_over_zero"]) > 0.0
    learned_full = float(full["learned_improvement_over_zero"]) > 0.0
    learned_short_ratio = float(
        results["short"]["learned"].get("diagnostics", {}).get(
            "control_reference_displacement_ratio", float("nan")
        )
    )
    learned_full_ratio = float(
        results["full"]["learned"].get("diagnostics", {}).get(
            "control_reference_displacement_ratio", float("nan")
        )
    )
    decision_name, next_action = _classify_evaluation_outcome(
        oracle_short=oracle_short,
        oracle_full=oracle_full,
        learned_short=learned_short,
        learned_full=learned_full,
        learned_short_ratio=learned_short_ratio,
    )
    replication_resources = _replication_capacity_record(run_dir, test_only=False)
    replication_authorized = bool(
        oracle_full and learned_full and int(replication_resources["passed"]) == 1
    )
    decision = _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-decision",
            "schema_version": 1,
            "decision": decision_name,
            "selected_gain": selected_gain,
            "short": short,
            "full": full,
            "learned_short_control_reference_displacement_ratio": learned_short_ratio,
            "learned_full_control_reference_displacement_ratio": learned_full_ratio,
            "replication_authorized": int(replication_authorized),
            "replication_resource_projection_sha256": replication_resources[
                "semantic_sha256"
            ],
            "recommended_next_action": next_action,
            "research_mode": "exploratory",
            "validation_pass_claim_authorized": 0,
            "general_generator_claim_authorized": 0,
            "prior_start_claim_authorized": 0,
        }
    )
    atomic_write_json(run_dir / "exploratory_decision.json", decision)
    flattened = [row for horizon in results.values() for row in horizon.values()]
    all_integrity = all(int(row.get("passed_integrity", 0)) == 1 for row in flattened)
    all_images = all(
        all(
            (run_dir / relative).is_file()
            for image in row.get("images", {}).values()
            for relative in image.values()
        )
        for row in flattened
    )
    paired_rng = all(
        len(
            {
                config_fingerprint(row["rng_binding"])
                for row in results[horizon].values()
            }
        )
        == 1
        for horizon in ("short", "full")
    )
    gate = _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-evaluation-gate",
            "schema_version": 1,
            "gate_type": "execution_integrity",
            "all_trajectories_complete": int(all_integrity),
            "all_images_saved": int(all_images),
            "paired_rng_variant_independent": int(paired_rng),
            "passed": int(all_integrity and all_images and paired_rng),
            "diagnostic_decision": decision_name,
        }
    )
    atomic_write_json(run_dir / "evaluation_gate.json", gate)
    if not int(gate["passed"]):
        raise RolloutCLIError("evaluation trajectory integrity gate failed")
    _write_contact_sheets(run_dir)
    _status(
        run_dir,
        state="ready_for_replication" if replication_authorized else "ready_for_report",
        stage="evaluation",
        decision=decision_name,
        scientific_evidence_complete=1,
    )
    return gate


def _fused_cuda_evaluation_stage(
    run_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    development_gate = _load_semantic(run_dir / "development_gate.json", "development gate")
    if not int(development_gate.get("passed", 0)):
        raise RolloutCLIError("evaluation is closed because the development oracle failed")
    selection = _verify_development_selection(run_dir)
    config = _load_semantic(run_dir / "scientific_config.json", "scientific configuration")
    selected_gain = float(selection["selected_gain"])
    short_anchor = int(config["short_anchor"])
    full_anchor = int(config["full_anchor"])
    _forward_role(
        run_dir,
        args,
        role="evaluation",
        path_id=PATH_IDS["evaluation"],
        anchors=(short_anchor, full_anchor),
    )
    prefix_rows = [
        ("full-zero", "zero", "full", None),
        ("full-learned", "learned", "full", selected_gain),
        ("full-oracle", "oracle", "full", None),
    ]
    full_anchor_state = _anchor_state(run_dir, "evaluation", full_anchor)
    prefix_sequence = tuple(
        coordinate for coordinate in _reverse_sequence(full_anchor) if coordinate[0] >= 128
    )
    prefix_destination = run_dir / "evaluation/full/fused_prefix"
    prefix_result, prefix_summary = _run_and_commit_fused_family(
        run_dir,
        args,
        initial_state=full_anchor_state,
        sequence=prefix_sequence,
        destination=prefix_destination,
        family_name="evaluation-full",
        segment_name="prefix-511-to-128",
        rows=prefix_rows,
        canonical_path_id=PATH_IDS["evaluation"],
        stream_role=STREAM_ROLES["evaluation"],
        expected_transition_count=12_644_352,
        capture_coordinates={
            (384, 0): "progress_25",
            (256, 0): "progress_50",
            (128, 0): "progress_75",
        },
    )
    append_rows = [
        ("short-zero", "zero", "short", None),
        ("short-learned", "learned", "short", selected_gain),
        ("short-oracle", "oracle", "short", None),
    ]
    append_specs, _append_bank, _append_binding = _fused_controller_family(
        run_dir,
        args,
        rows=append_rows,
        canonical_path_id=PATH_IDS["evaluation"],
    )
    from mnist.d0_jacobi_rb_tangent_fused import join_fused_family_rows

    short_anchor_state = _anchor_state(run_dir, "evaluation", short_anchor)
    append_state = np.repeat(short_anchor_state[None, :], 3, axis=0)
    join = join_fused_family_rows(
        _field(prefix_result, "final_state"),
        _field(prefix_result, "row_specs"),
        append_state,
        append_specs,
        next_coordinate=(127, 6),
        bindings={
            "prefix_family_sha256": prefix_summary["semantic_sha256"],
            "short_anchor_sha256": _array_sha256(short_anchor_state),
            "development_selection_sha256": selection["semantic_sha256"],
            "rng_root_seed": REVERSE_ROOT_SEED,
            "stream_role": STREAM_ROLES["evaluation"],
            "variant_in_rng_key": 0,
        },
    )
    join_path = run_dir / "evaluation/evaluation_family_join.json"
    if join_path.is_file():
        if _load_semantic(join_path, "evaluation family join") != dict(join.record):
            raise ArtifactCompatibilityError("evaluation family join changed")
    else:
        atomic_write_json(join_path, dict(join.record))
    suffix_rows = [*prefix_rows, *append_rows]
    suffix_destination = run_dir / "evaluation/joined_suffix/fused_family"
    suffix_result, suffix_summary = _run_and_commit_fused_family(
        run_dir,
        args,
        initial_state=np.asarray(join.state),
        sequence=_reverse_sequence(short_anchor),
        destination=suffix_destination,
        family_name="evaluation-joined",
        segment_name="suffix-127-to-0",
        rows=suffix_rows,
        canonical_path_id=PATH_IDS["evaluation"],
        stream_role=STREAM_ROLES["evaluation"],
        expected_transition_count=8_429_568,
        capture_coordinates={
            (96, 0): "progress_25",
            (64, 0): "progress_50",
            (32, 0): "progress_75",
            (0, 0): "final",
        },
    )
    prefix_path = prefix_destination / "family_summary.json"
    suffix_path = suffix_destination / "family_summary.json"
    results: dict[str, dict[str, dict[str, Any]]] = {"short": {}, "full": {}}
    for row_index, (_key, variant, _horizon, gain) in enumerate(prefix_rows):
        prefix_saved = _field(prefix_result, "saved_states", {})
        suffix_saved = _field(suffix_result, "saved_states", {})
        states = {
            "start": np.asarray(prefix_saved["start"])[row_index],
            "progress_25": np.asarray(prefix_saved["progress_25"])[row_index],
            "progress_50": np.asarray(prefix_saved["progress_50"])[row_index],
            "progress_75": np.asarray(prefix_saved["progress_75"])[row_index],
            "final": np.asarray(suffix_saved["final"])[row_index],
        }
        results["full"][variant] = _commit_fused_trajectory_summary(
            run_dir,
            args,
            role="evaluation",
            horizon="full",
            anchor_step=full_anchor,
            variant=variant,
            gain=gain,
            states=states,
            diagnostic_parts=((prefix_result, row_index), (suffix_result, row_index)),
            family_bindings=(
                (
                    prefix_path,
                    prefix_summary,
                    row_index,
                    {
                        "start": "start",
                        "progress_25": "progress_25",
                        "progress_50": "progress_50",
                        "progress_75": "progress_75",
                    },
                ),
                (suffix_path, suffix_summary, row_index, {"final": "final"}),
            ),
            expected_transitions=(full_anchor + 1) * 7 * 4 * 392,
        )
    for local_index, (_key, variant, _horizon, gain) in enumerate(append_rows):
        row_index = 3 + local_index
        suffix_saved = _field(suffix_result, "saved_states", {})
        states = {
            name: np.asarray(value)[row_index]
            for name, value in suffix_saved.items()
        }
        results["short"][variant] = _commit_fused_trajectory_summary(
            run_dir,
            args,
            role="evaluation",
            horizon="short",
            anchor_step=short_anchor,
            variant=variant,
            gain=gain,
            states=states,
            diagnostic_parts=((suffix_result, row_index),),
            family_bindings=(
                (
                    suffix_path,
                    suffix_summary,
                    row_index,
                    {name: name for name in states},
                ),
            ),
            expected_transitions=(short_anchor + 1) * 7 * 4 * 392,
        )
    return _finalize_fused_evaluation(
        run_dir,
        args,
        results=results,
        selected_gain=selected_gain,
    )


def _fused_evaluation_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.test_only:
        gate = _singleton_evaluation_stage(run_dir, args)
        selected_gain = float(
            _load_semantic(run_dir / "development/development_selection.json")[
                "selected_gain"
            ]
        )
        ordered_names = ("zero", f"learned-gain-{selected_gain:g}", "oracle")
        full = [
            _load_semantic(
                run_dir / "evaluation/full" / name / "trajectory_summary.json",
                "evaluation full trajectory summary",
            )
            for name in ordered_names
        ]
        short = [
            _load_semantic(
                run_dir / "evaluation/short" / name / "trajectory_summary.json",
                "evaluation short trajectory summary",
            )
            for name in ordered_names
        ]
        prefix = _test_fused_family_record(
            run_dir,
            destination=run_dir / "evaluation/full/fused_prefix",
            family_name="evaluation-full",
            segment_name="prefix-511-to-128",
            summaries=full,
            canonical_path_id=PATH_IDS["evaluation"],
            transition_count=12_644_352,
        )
        join = _semantic(
            {
                "schema": FUSED_TEST_RUN_SCHEMA + "-evaluation-family-join",
                "schema_version": 1,
                "prefix_family_sha256": prefix["semantic_sha256"],
                "short_anchor_sha256": _load_semantic(
                    run_dir / "forward/evaluation/forward_summary.json"
                )["anchors"][str(int(_load_semantic(run_dir / "scientific_config.json")["short_anchor"]))]["state_sha256"],
                "row_order": [
                    "full-zero",
                    "full-learned",
                    "full-oracle",
                    "short-zero",
                    "short-learned",
                    "short-oracle",
                ],
                "next_sequence_coordinate": [127, 6],
                "variant_in_rng_key": 0,
                "passed": 1,
            }
        )
        atomic_write_json(run_dir / "evaluation/evaluation_family_join.json", join)
        _test_fused_family_record(
            run_dir,
            destination=run_dir / "evaluation/joined_suffix/fused_family",
            family_name="evaluation-joined",
            segment_name="suffix-127-to-0",
            summaries=[*full, *short],
            canonical_path_id=PATH_IDS["evaluation"],
            transition_count=8_429_568,
        )
        return gate
    return _fused_cuda_evaluation_stage(run_dir, args)


def _fused_cuda_replication_stage(
    run_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    gate = _load_semantic(run_dir / "evaluation_gate.json", "evaluation gate")
    if int(gate.get("passed", 0)) != 1:
        raise ArtifactCompatibilityError("replication requires passing evaluation integrity")
    selection = _verify_development_selection(run_dir)
    decision = _load_semantic(run_dir / "exploratory_decision.json", "exploratory decision")
    projection = _load_semantic(
        run_dir / "metrics/replication_resource_projection.json",
        "replication resource projection",
    )
    recomputed_authorized = bool(
        float(decision.get("full", {}).get("oracle_improvement_over_zero", 0.0)) > 0.0
        and float(decision.get("full", {}).get("learned_improvement_over_zero", 0.0)) > 0.0
        and int(projection.get("passed", 0)) == 1
    )
    if (
        float(decision.get("selected_gain", float("nan")))
        != float(selection["selected_gain"])
        or decision.get("replication_resource_projection_sha256")
        != projection.get("semantic_sha256")
        or int(decision.get("replication_authorized", -1)) != int(recomputed_authorized)
    ):
        raise ArtifactCompatibilityError("replication authorization changed")
    if not recomputed_authorized:
        record = _semantic(
            {
                "schema": FUSED_RUN_SCHEMA + "-replication-gate",
                "schema_version": 1,
                "evaluation_status": "not_evaluated",
                "reason": "predeclared positive-evaluation condition not met",
                "passed": 1,
                "replication_performed": 0,
            }
        )
        atomic_write_json(run_dir / "replication_gate.json", record)
        _status(
            run_dir,
            state="ready_for_report",
            stage="replication",
            decision=decision["decision"],
            scientific_evidence_complete=1,
        )
        return record
    config = _load_semantic(run_dir / "scientific_config.json", "scientific configuration")
    anchor_step = int(config["full_anchor"])
    _forward_role(
        run_dir,
        args,
        role="replication",
        path_id=PATH_IDS["replication"],
        anchors=(anchor_step,),
    )
    gain = float(decision["selected_gain"])
    rows = [
        ("replication-full-zero", "zero", "full", None),
        ("replication-full-learned", "learned", "full", gain),
    ]
    destination = run_dir / "replication/full/fused_family"
    result, family = _run_and_commit_fused_family(
        run_dir,
        args,
        initial_state=_anchor_state(run_dir, "replication", anchor_step),
        sequence=_reverse_sequence(anchor_step),
        destination=destination,
        family_name="replication-full",
        segment_name="complete",
        rows=rows,
        canonical_path_id=PATH_IDS["replication"],
        stream_role=STREAM_ROLES["replication"],
        expected_transition_count=11_239_424,
        capture_coordinates={
            (384, 0): "progress_25",
            (256, 0): "progress_50",
            (128, 0): "progress_75",
            (0, 0): "final",
        },
    )
    family_path = destination / "family_summary.json"
    summaries = []
    for row_index, (_key, variant, _horizon, row_gain) in enumerate(rows):
        states = {
            name: np.asarray(value)[row_index]
            for name, value in _field(result, "saved_states", {}).items()
        }
        summaries.append(
            _commit_fused_trajectory_summary(
                run_dir,
                args,
                role="replication",
                horizon="full",
                anchor_step=anchor_step,
                variant=variant,
                gain=row_gain,
                states=states,
                diagnostic_parts=((result, row_index),),
                family_bindings=(
                    (
                        family_path,
                        family,
                        row_index,
                        {name: name for name in states},
                    ),
                ),
                expected_transitions=(anchor_step + 1) * 7 * 4 * 392,
            )
        )
    zero_error = _squared_l2(summaries[0])
    learned_error = _squared_l2(summaries[1])
    agrees = learned_error < zero_error
    updated = _semantic(
        {
            **{key: value for key, value in decision.items() if key != "semantic_sha256"},
            "decision": (
                "learned_full_replication_agrees"
                if agrees
                else "learned_full_replication_disagrees"
            ),
            "replication_performed": 1,
            "replication_learned_improvement_over_zero": zero_error - learned_error,
            "replication_direction_agrees": int(agrees),
            "recommended_next_action": (
                "verify frozen checkpoint/gain at M=8"
                if agrees
                else "treat single evaluation improvement as unstable and pivot learner"
            ),
        }
    )
    atomic_write_json(run_dir / "exploratory_decision.json", updated)
    replication_gate = _semantic(
        {
            "schema": FUSED_RUN_SCHEMA + "-replication-gate",
            "schema_version": 1,
            "gate_type": "execution_integrity",
            "replication_performed": 1,
            "direction_agrees_diagnostic": int(agrees),
            "passed": 1,
        }
    )
    atomic_write_json(run_dir / "replication_gate.json", replication_gate)
    _status(
        run_dir,
        state="ready_for_report",
        stage="replication",
        decision=updated["decision"],
        scientific_evidence_complete=1,
    )
    return replication_gate


def _fused_replication_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.test_only:
        gate = _singleton_replication_stage(run_dir, args)
        if int(gate.get("replication_performed", 0)):
            summaries = [
                _load_semantic(path, "replication trajectory summary")
                for path in sorted(
                    (run_dir / "replication/full").glob("*/trajectory_summary.json")
                )
            ]
            _test_fused_family_record(
                run_dir,
                destination=run_dir / "replication/full/fused_family",
                family_name="replication-full",
                segment_name="complete",
                summaries=summaries,
                canonical_path_id=PATH_IDS["replication"],
                transition_count=11_239_424,
            )
        return gate
    return _fused_cuda_replication_stage(run_dir, args)


def _development_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    return (
        _fused_development_stage(run_dir, args)
        if _fused_mode(args)
        else _singleton_development_stage(run_dir, args)
    )


def _evaluation_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    return (
        _fused_evaluation_stage(run_dir, args)
        if _fused_mode(args)
        else _singleton_evaluation_stage(run_dir, args)
    )


def _replication_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    return (
        _fused_replication_stage(run_dir, args)
        if _fused_mode(args)
        else _singleton_replication_stage(run_dir, args)
    )


def _trajectory_summaries(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(run_dir.glob("*/*/*/trajectory_summary.json")):
        value = _load_semantic(path, "trajectory summary")
        diagnostics = value.get("diagnostics", {})
        controller = diagnostics.get("controller", {})
        health = value.get("health", {})
        rows.append(
            {
                "artifact": path.relative_to(run_dir).as_posix(),
                "role": value["role"],
                "horizon": value["horizon"],
                "variant": value["variant"],
                "gain": value.get("gain"),
                "squared_l2": _squared_l2(value),
                "l1": value["metrics"]["final"].get("l1_error"),
                "total_variation": value["metrics"]["final"].get("total_variation_distance"),
                "centered_contrast_correlation": value["metrics"]["final"].get(
                    "centered_contrast_correlation"
                ),
                "state_sha256": value["final_state_sha256"],
                "control_reference_displacement_ratio": diagnostics.get(
                    "control_reference_displacement_ratio"
                ),
                "score_rms": diagnostics.get("score_rms"),
                "score_maximum_absolute": diagnostics.get("score_maximum_absolute"),
                "logistic_shift_rms": diagnostics.get("logistic_shift_rms"),
                "logistic_shift_maximum_absolute": diagnostics.get(
                    "logistic_shift_maximum_absolute"
                ),
                "unscaled_score_rms": controller.get("unscaled_score_rms"),
                "scaled_score_rms": controller.get("scaled_score_rms"),
                "unscaled_score_maximum_absolute": controller.get(
                    "unscaled_score_maximum_absolute"
                ),
                "scaled_score_maximum_absolute": controller.get(
                    "scaled_score_maximum_absolute"
                ),
                "boundary_fraction_count": diagnostics.get("boundary_fraction_count", 0),
                "oracle_unreachable_boundary_count": diagnostics.get(
                    "target_oracle_unreachable_boundary_count", 0
                ),
                "transition_count": health.get("transition_count"),
                "active_count": health.get("active_count"),
                "structural_noop_count": health.get("structural_noop_count"),
                "certified_count": health.get("certified_count"),
                "certificate_fraction": health.get("certificate_fraction"),
                "fallback_count": health.get("fallback_count"),
                "fallback_fraction": health.get("fallback_fraction"),
                "fallback_time_fraction": health.get("fallback_time_fraction"),
                "elapsed_seconds": max(
                    float(value.get("end_to_end_elapsed_seconds", 0.0)),
                    float(health.get("elapsed_seconds", 0.0)),
                ),
                "transitions_per_second": health.get("transitions_per_second"),
                "peak_memory_fraction": health.get("peak_memory_fraction"),
                "maximum_cuda_memory_allocated": health.get(
                    "maximum_cuda_memory_allocated"
                ),
                "maximum_simplex_mass_error": health.get(
                    "maximum_simplex_mass_error"
                ),
                "maximum_pair_mass_error": health.get(
                    "maximum_pair_mass_error"
                ),
            }
        )
    return rows


def _write_report(run_dir: Path) -> None:
    status = _load_json(run_dir / "run_status.json") if (run_dir / "run_status.json").is_file() else {}
    failure = (
        _load_semantic(run_dir / "failure.json", "run failure")
        if (run_dir / "failure.json").is_file()
        else {}
    )
    decision = _load_semantic(run_dir / "exploratory_decision.json", "exploratory decision") if (
        run_dir / "exploratory_decision.json"
    ).is_file() else {}
    selection = _load_semantic(run_dir / "development/development_selection.json", "development selection") if (
        run_dir / "development/development_selection.json"
    ).is_file() else {}
    endpoint_rows = _trajectory_summaries(run_dir)
    if endpoint_rows:
        atomic_write_csv(run_dir / "metrics/endpoint_metrics.csv", endpoint_rows)
    decision_name = decision.get("decision", status.get("decision", "incomplete"))
    resource_projection = (
        _load_semantic(
            run_dir / "preflight/resource_projection.json",
            "preflight resource projection",
        )
        if (run_dir / "preflight/resource_projection.json").is_file()
        else {}
    )
    fused = _fused_mode(
        _load_json(run_dir / "scientific_config.json")
        if (run_dir / "scientific_config.json").is_file()
        else {}
    )
    resource_stop_without_endpoint = bool(
        decision_name == "rollout_resource_budget_exhausted" and not endpoint_rows
    )
    preflight_resource_stop = bool(
        resource_stop_without_endpoint
        and failure.get("stage") == "preflight"
        and failure.get("failure_code")
        == "rollout_main_workflow_computationally_infeasible"
        and (
            not fused
            or (run_dir / "preflight/objective_roles_unopened.json").is_file()
        )
    )
    pre_objective_failure = bool(
        not endpoint_rows
        and failure.get("stage") in {"initialize", "preflight"}
        and not preflight_resource_stop
    )
    objective_failure_evidence = any(
        int(
            _load_semantic(path, "trajectory failure").get(
                "numerically_interpretable_partial_evidence", 0
            )
        )
        == 1
        for path in run_dir.rglob("trajectory_failure.json")
    )
    objective_stage_failure = bool(
        not endpoint_rows
        and failure.get("stage")
        in {"forward", "development", "evaluation", "replication"}
    )
    partial_objective_failure = bool(
        objective_stage_failure and objective_failure_evidence
    )
    objective_stage_precheck_failure = bool(
        objective_stage_failure and not objective_failure_evidence
    )
    proxy_counter_text = (
        "0 (this run completed objective-bearing trajectory artifacts)"
        if endpoint_rows
        else (
            "0 (this run opened an objective-bearing trajectory and retained "
            "interpretable failure states/images)"
            if objective_failure_evidence
            else (
            "at least one; the exact historical count is not established, and this "
            "engineering/preflight run does not reset it"
            )
        )
    )
    if preflight_resource_stop:
        next_action = (
            "The unchanged exact M=2 rollout is infeasible under the current laptop-only "
            "six-hour contract. Explicitly change the cap or pivot the scientific experiment; "
            "do not run another packing microbenchmark."
            if fused
            else (
                "Plan an exact cross-variant fused tangent-reference scheduling feasibility gate; "
                "do not generate fresh forward evidence or change the controller experiment."
            )
        )
        establishes_text = (
            "The exact CUDA/controller seam is numerically healthy, but the measured "
            + ("fused laptop schedule" if fused else "single-path schedule")
            + " cannot execute the frozen workflow within six hours. "
            "No reverse-control effect was evaluated."
        )
        strategy_text = (
            "The scientific controller strategy remains unevaluated; "
            + (
                "the current laptop/six-hour contract is terminal unless the user changes it."
                if fused
                else "the immediate blocker is exact-reference scheduling throughput."
            )
        )
    elif pre_objective_failure:
        next_action = (
            "Repair the recorded initialization/preflight integrity blocker, rerun the "
            "frozen preflight, and do not interpret this run as reverse-control evidence."
        )
        establishes_text = (
            "The run produced readable initialization/preflight failure evidence only. "
            "No reverse-control trajectory or paired objective effect was evaluated."
        )
        strategy_text = (
            "The scientific controller strategy remains unevaluated because an integrity "
            "check failed before the objective workflow opened."
        )
    elif objective_stage_precheck_failure:
        next_action = (
            "The objective-stage resource precheck stopped before an interpretable reverse "
            "trajectory commit. Review the frozen cap or pivot the experiment; do not claim "
            "partial trajectory evidence from this run."
            if failure.get("failure_domain") == "resource_budget"
            else (
                "Repair the recorded objective-stage integrity blocker and rerun from the "
                "last verified boundary; no endpoint or image evidence was committed."
            )
        )
        establishes_text = (
            "Only an objective-stage pre-execution failure was recorded. No interpretable "
            "reverse trajectory, endpoint/image evidence, or paired objective effect was committed."
        )
        strategy_text = (
            "The scientific controller strategy remains unevaluated because the objective "
            "stage stopped before an interpretable trajectory commit."
        )
    elif partial_objective_failure:
        next_action = (
            "Keep the last committed fused-family states and failure images as partial "
            "objective evidence. The frozen six-hour cap has stopped this run; review the "
            "unfinished family and either explicitly change the cap or pivot the experiment."
            if failure.get("failure_domain") == "resource_budget"
            else (
                "Keep the last committed fused-family states and failure images as partial "
                "objective evidence, repair the recorded execution blocker, and resume only "
                "from the last verified shard boundary."
            )
        )
        establishes_text = (
            "The objective workflow opened and executed valid exact reverse shards up to an "
            "verified atomic shard boundary. The saved partial states are interpretable execution "
            "evidence, but no complete paired endpoint effect was evaluated."
        )
        strategy_text = (
            "The requested objective horizon remains unevaluated because the exact fused "
            "workflow reached its frozen resource cap after opening the trajectory. Partial "
            "states and failure records must be inspected before changing the cap or strategy."
        )
    else:
        next_action = str(
            decision.get("recommended_next_action", "complete the next unfinished stage")
        )
        establishes_text = (
            "Only the paired exploratory behavior recorded in the raw states, endpoint "
            "metrics, telemetry, and fixed-scale images."
        )
        strategy_text = (
            f"Strategy status follows `{decision_name}`. The strongest alternative is a "
            "materially more global or rollout-trained controller if the learned trajectory "
            "is not useful."
        )
    manifest = _load_semantic(run_dir / "run_manifest.json", "run manifest") if (
        run_dir / "run_manifest.json"
    ).is_file() else {}
    invocation = manifest.get("invocation", {})
    revision = manifest.get("source_revision", {})
    endpoint_lines = [
        "| Role | Horizon | Variant | Gain | L2Â² | L1 | TV | Correlation | Control/reference |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in endpoint_rows:
        endpoint_lines.append(
            "| {role} | {horizon} | {variant} | {gain} | {l2:.9g} | {l1:.9g} | "
            "{tv:.9g} | {corr:.9g} | {ratio} |".format(
                role=row["role"],
                horizon=row["horizon"],
                variant=row["variant"],
                gain="" if row["gain"] is None else row["gain"],
                l2=float(row["squared_l2"]),
                l1=float(row["l1"]),
                tv=float(row["total_variation"]),
                corr=float(row["centered_contrast_correlation"]),
                ratio=(
                    ""
                    if row["control_reference_displacement_ratio"] is None
                    else f"{float(row['control_reference_displacement_ratio']):.6g}"
                ),
            )
        )
    paired = _load_semantic(
        run_dir / "metrics/evaluation_paired_metrics.json", "evaluation paired metrics"
    ) if (run_dir / "metrics/evaluation_paired_metrics.json").is_file() else {}
    paired_lines: list[str] = []
    for horizon in ("short", "full"):
        row = paired.get(horizon)
        if isinstance(row, Mapping):
            paired_lines.append(
                f"- {horizon}: learned Î”L2Â²={float(row['learned_improvement_over_zero']):.9g} "
                f"({row.get('learned_relative_improvement')} relative); oracle "
                f"Î”L2Â²={float(row['oracle_improvement_over_zero']):.9g} "
                f"({row.get('oracle_relative_improvement')} relative)."
            )
    evidence_paths = [
        path for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in _FINAL_MANIFEST_EXCLUDED
    ]
    evidence_bytes = sum(path.stat().st_size for path in evidence_paths)
    final_manifest_paths = {
        path.relative_to(run_dir).as_posix() for path in evidence_paths
    } | {"REPORT.md", "HANDOFF.md"}
    expected_final_artifact_count = len(final_manifest_paths)
    objective_links = sorted(
        path.relative_to(run_dir).as_posix()
        for path in (run_dir / "images").glob("*contact_sheet.png")
    ) if (run_dir / "images").is_dir() else []
    failure_summaries = sorted(
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("trajectory_failure.json")
    )
    core_shard_failures = sorted(
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("shard-*.failure.json")
    )
    health_rows: list[dict[str, Any]] = [dict(row) for row in endpoint_rows]
    for path in sorted((run_dir / "forward").glob("*/forward_summary.json")):
        forward = _load_semantic(path, "forward health summary")
        health = forward.get("health", {})
        health_rows.append(
            {
                "artifact": path.relative_to(run_dir).as_posix(),
                "transition_count": health.get("transition_count", 0),
                "active_count": health.get(
                    "active_count", health.get("transition_count", 0)
                ),
                "structural_noop_count": health.get("structural_noop_count", 0),
                "certified_count": health.get("certified_count", 0),
                "fallback_count": health.get("fallback_count", 0),
                "fallback_time_fraction": health.get("fallback_time_fraction", 0.0),
                "elapsed_seconds": max(
                    float(forward.get("end_to_end_elapsed_seconds", 0.0)),
                    float(health.get("elapsed_seconds", 0.0)),
                ),
                "transitions_per_second": health.get("transitions_per_second", 0.0),
                "peak_memory_fraction": health.get("peak_memory_fraction", 0.0),
                "maximum_cuda_memory_allocated": health.get(
                    "maximum_cuda_memory_allocated", 0
                ),
                "maximum_simplex_mass_error": health.get("maximum_simplex_mass_error", 0.0),
                "maximum_pair_mass_error": health.get("maximum_pair_mass_error", 0.0),
            }
        )
    completed_transitions = sum(int(row.get("transition_count") or 0) for row in health_rows)
    completed_active = sum(
        int(
            row.get("active_count")
            if row.get("active_count") is not None
            else row.get("transition_count") or 0
        )
        for row in health_rows
    )
    completed_structural_noops = sum(
        int(
            row.get("structural_noop_count")
            if row.get("structural_noop_count") is not None
            else max(
                0,
                int(row.get("transition_count") or 0)
                - int(
                    row.get("active_count")
                    if row.get("active_count") is not None
                    else row.get("transition_count") or 0
                ),
            )
        )
        for row in health_rows
    )
    completed_certified = sum(int(row.get("certified_count") or 0) for row in health_rows)
    completed_fallbacks = sum(int(row.get("fallback_count") or 0) for row in health_rows)
    completed_elapsed = math.fsum(float(row.get("elapsed_seconds") or 0.0) for row in health_rows)
    completed_rate = completed_transitions / completed_elapsed if completed_elapsed > 0.0 else None
    certificate_fraction = (
        completed_certified / completed_active if completed_active > 0 else (
            1.0 if completed_transitions > 0 else None
        )
    )
    maximum_simplex_error = max(
        (float(row.get("maximum_simplex_mass_error") or 0.0) for row in health_rows),
        default=0.0,
    )
    maximum_pair_error = max(
        (float(row.get("maximum_pair_mass_error") or 0.0) for row in health_rows),
        default=0.0,
    )
    maximum_memory_fraction = max(
        (float(row.get("peak_memory_fraction") or 0.0) for row in health_rows),
        default=0.0,
    )
    maximum_memory_bytes = max(
        (int(row.get("maximum_cuda_memory_allocated") or 0) for row in health_rows),
        default=0,
    )
    maximum_fallback_time_fraction = max(
        (float(row.get("fallback_time_fraction") or 0.0) for row in health_rows),
        default=0.0,
    )
    boundary_count = sum(int(row.get("boundary_fraction_count") or 0) for row in endpoint_rows)
    unreachable_count = sum(
        int(row.get("oracle_unreachable_boundary_count") or 0) for row in endpoint_rows
    )
    learned_scale_lines = []
    for row in endpoint_rows:
        if row["variant"] != "learned":
            continue
        learned_scale_lines.append(
            f"- `{row['artifact']}`: gain={row['gain']}, unscaled/scaled score RMS="
            f"{row.get('unscaled_score_rms')}/{row.get('scaled_score_rms')}, "
            f"score/logistic-shift RMS={row.get('score_rms')}/{row.get('logistic_shift_rms')}, "
            f"control/reference={row.get('control_reference_displacement_ratio')}."
        )
    projection_lines: list[str] = []
    if resource_projection:
        if isinstance(resource_projection.get("profiles"), Mapping):
            profile_rates = {
                name: float(row.get("slowest_profile_rate", float("nan")))
                for name, row in resource_projection["profiles"].items()
                if isinstance(row, Mapping)
            }
            projection_lines = [
                "- Fused preflight slowest complete-shard rates: "
                + ", ".join(
                    f"{name}={value:.6g}" for name, value in profile_rates.items()
                )
                + " transitions/s.",
                "- Conservative fused projected main time: "
                f"{float(resource_projection.get('projected_main_wall_seconds', float('nan'))):.6g} s "
                f"versus the {float(resource_projection.get('maximum_main_wall_seconds', float('nan'))):.6g} s cap; "
                f"effective rate={float(resource_projection.get('effective_rate', float('nan'))):.6g} transitions/s.",
            ]
        else:
            rates = [float(value) for value in resource_projection.get("repeat_rates", [])]
            projection_lines = [
                "- Preflight complete-phase repeat rates: "
                + ", ".join(f"{value:.6g}" for value in rates)
                + " transitions/s.",
                "- Conservative projected main time: "
                f"{float(resource_projection.get('projected_main_wall_seconds', float('nan'))):.6g} s "
                f"versus the {float(resource_projection.get('maximum_main_wall_seconds', float('nan'))):.6g} s cap.",
            ]
    lines = [
        "# Exploratory frequency-one Jacobi/RB reverse rollout",
        "",
        "Research mode: **exploratory**. This is an objective-bearing one-image rollout, not confirmation.",
        "",
        "## Decision",
        "",
        f"`{decision_name}`",
        "",
        "The experiment asks whether the historical validation-inspected seed 261372/update 3700 checkpoint has dynamically useful reverse-control signal on fresh trajectories.",
        "",
        "## Frozen setting",
        "",
        f"- Controller refinement: M={MICROSTEPS}",
        f"- Development gains: {', '.join(str(value) for value in LEARNED_GAINS)}",
        f"- Selected gain: {selection.get('selected_gain', 'not selected')}",
        "- Paired controls: zero score, learned score, source-informed target-pull oracle",
        "- Horizons: 128-step suffix and complete 512-step reverse path",
        "- Independent unit: one fresh path per exploratory role",
        "",
        "## Objective endpoints and paired effects",
        "",
        *(endpoint_lines if endpoint_rows else ["No completed trajectory endpoint yet."]),
        "",
        *(paired_lines if paired_lines else ["Paired evaluation effects are not yet available."]),
        "",
        "Progress-resolved paired L2/L1/TV/correlation effects, learned-zero divergence, and successive-quarter contributions are in `metrics/*_paired_progress.{json,csv}`.",
        "",
        "## Controller and numerical health",
        "",
        f"- Completed exact transition lanes: {completed_transitions}; active/certified: {completed_active}/{completed_certified}; structural no-ops: {completed_structural_noops}; active-lane certificate fraction: {certificate_fraction}; fallbacks: {completed_fallbacks}; maximum fallback-time fraction: {maximum_fallback_time_fraction:.6g}.",
        f"- Accumulated measured execution: {completed_elapsed:.6g} s; aggregate rate: {completed_rate}; peak CUDA allocation: {maximum_memory_bytes} bytes ({maximum_memory_fraction:.6g} of device memory).",
        f"- Maximum simplex/pair-mass errors: {maximum_simplex_error:.6g} / {maximum_pair_error:.6g}; boundary fractions: {boundary_count}; oracle-unreachable boundaries: {unreachable_count}.",
        "- Exact sources: `metrics/endpoint_metrics.csv`, each trajectory `trajectory_summary.json`, and `forward/*/forward_summary.json`.",
        *(learned_scale_lines or ["- No completed learned-controller telemetry yet."]),
        *projection_lines,
        "",
        "## Exact execution binding",
        "",
        f"- Command: `{invocation.get('normalized_command', 'unavailable')}`",
        f"- Git HEAD: `{revision.get('git_head', 'unavailable')}`; dirty={revision.get('git_dirty', 'unknown')}",
        f"- Load-bearing source closure: `{manifest.get('source_fingerprint', 'unavailable')}`",
        "",
        "## Claim boundary",
        "",
        "This run may describe paired exploratory endpoint changes and saved images. It does not establish a validation pass, controller generalization, prior-start generation, multi-image generation, or Eulerian convergence.",
        "",
        "## Artifacts",
        "",
        "Raw selected states live beside each trajectory summary. Fixed-scale PNGs are under `images/`; bad endpoints are retained. `metrics/endpoint_metrics.csv` is derived from raw float64 states.",
        f"Objective contact sheets: {', '.join(f'`{item}`' for item in objective_links) or 'not yet available'}.",
        f"Failed trajectory summaries: {', '.join(f'`{item}`' for item in failure_summaries) or 'none'}.",
        f"Core uncommitted-shard failure records: {', '.join(f'`{item}`' for item in core_shard_failures) or 'none'}.",
        f"Evidence map at report assembly: {len(evidence_paths)} files, {evidence_bytes} bytes; final hashes are in `artifact_manifest.json` and `SHA256SUMS.txt`.",
        "",
        "## Next action",
        "",
        next_action,
        "",
    ]
    _atomic_text(run_dir / "REPORT.md", "\n".join(lines))
    handoff = [
        "# Frequency-one exploratory rollout: research handoff",
        "",
        f"Date: {_now()}",
        "",
        "## 1. Program objective",
        "",
        "Build or decisively falsify a DDPM-like MNIST generator based on the fixed-grid Eulerian/Jacobi approximation. The concrete success artifact is a recognizable generated or reconstructed MNIST image from a complete reverse trajectory.",
        "",
        "## 2. Current milestone and distance to goal",
        "",
        (
            "Nearest objective-bearing milestone: paired one-image reverse-suffix and full-path reconstruction. "
            + (
                "This preflight stopped before that milestone because the exact schedule exceeded its frozen resource cap."
                if preflight_resource_stop
                else (
                    "This run stopped before that milestone during initialization/preflight "
                    "integrity checks; no reverse trajectory opened."
                    if pre_objective_failure
                    else (
                        (
                            "This run reached an objective-stage resource precheck but stopped "
                            "before an interpretable reverse trajectory was committed."
                            if failure.get("failure_domain") == "resource_budget"
                            else (
                                "This run reached the objective stage but an integrity failure "
                                "stopped it before endpoint or image evidence was committed."
                            )
                        )
                        if objective_stage_precheck_failure
                        else (
                            "This run opened that milestone but stopped at a verified atomic shard "
                            "boundary; partial raw states and failure images were retained."
                            if partial_objective_failure
                            else "This run is that milestone."
                        )
                    )
                )
            )
            + f" Proxy-only patches since the last objective-bearing experiment: {proxy_counter_text}."
        ),
        "",
        "## 3. Strategy review",
        "",
        strategy_text,
        "",
        "## 4. Research mode and evidence roles",
        "",
        f"Primary mode: exploratory. Development chooses one gain on path 0x{PATH_IDS['development']:X}. Evaluation uses fresh path 0x{PATH_IDS['evaluation']:X}. Replication, if predeclared conditions hold, uses 0x{PATH_IDS['replication']:X}. The old confirmation role was not opened.",
        "",
        "## 5. Exact result of the latest run",
        "",
        f"Terminal/current result: `{decision_name}` under exact certified reference transitions and M=2 tangent control.",
        "",
        *endpoint_lines,
        "",
        *(paired_lines if paired_lines else ["Paired evaluation effects are not yet available."]),
        "",
        f"Compact health: {completed_transitions} transition lanes ({completed_active} active, {completed_structural_noops} structural no-ops), active-lane certificate fraction {certificate_fraction}, {completed_fallbacks} fallbacks (max time fraction {maximum_fallback_time_fraction:.6g}), max simplex/pair error {maximum_simplex_error:.6g}/{maximum_pair_error:.6g}, {completed_elapsed:.6g}s aggregate elapsed, rate {completed_rate}, peak CUDA {maximum_memory_bytes} bytes ({maximum_memory_fraction:.6g}).",
        f"Mechanism counts: boundary fractions {boundary_count}; oracle-unreachable boundaries {unreachable_count}.",
        *(learned_scale_lines or ["No completed learned-controller scale telemetry yet."]),
        *projection_lines,
        "Exact evidence: `metrics/endpoint_metrics.csv`, trajectory summaries, and `forward/*/forward_summary.json`.",
        "",
        "### This result establishes",
        "",
        establishes_text,
        "",
        "### This result does not establish",
        "",
        "A validation pass, a general generator, prior-start success, multi-image generalization, or convergence of the unsplit Eulerian generator.",
        "",
        "## 6. Confirmed facts, current inferences, and open hypotheses",
        "",
        "Confirmed facts are limited to readable artifacts in this run. Open alternatives include implementation/composition failure, poor learned direction or amplitude, on-policy accumulation, inadequate receptive field, terminal-prior mismatch, proxy misalignment, and failure of the present strategy.",
        "",
        "## 7. Decision the next patch must resolve",
        "",
        next_action,
        "",
        "## 8. Candidate actions and value of information",
        "",
        "Use the predeclared outcome-to-action branch in REPORT.md; do not default to another read-only feature decomposition.",
        "",
        "## 9. Recommended next patch",
        "",
        next_action,
        "",
        "## 10. Gates and claim boundaries",
        "",
        "Preflight/forward/reverse health checks are execution-integrity gates. The development oracle is an interpretability gate. Learned/oracle endpoint improvements and the 0.05 displacement ratio are diagnostic thresholds, not confirmatory claims.",
        "",
        "## 11. Outcome-to-action table",
        "",
        "See `docs/jacobi_rb_frequency1_exploratory_rollout.md` in the source tree and `exploratory_decision.json` in this run.",
        "",
        "## 12. Constraints",
        "",
        "Integrity constraints are immutable input bytes, fresh-role isolation, paired RNG, certified transitions, conservation, and readable restart chains. Architecture, gains, schedule, backend for future exploration, and strategy remain revisable.",
        "",
        "## 13. Resource budget and stop rule",
        "",
        "Main cap: six GPU hours and 2 GiB. Optional replication cap: two additional GPU hours. Stop after this objective-bearing outcome; no proxy-only continuation is authorized by inertia.",
        "",
        "## 14. Alternative and pivot plan",
        "",
        "If the oracle works but the learned controller does not, compare a materially different global or rollout-state-trained learner rather than adding Fourier features one frequency at a time.",
        "",
        "## 15. Evidence map",
        "",
        "`exploratory_decision.json` is the decision; `development/development_selection.json` binds gain selection; trajectory summaries and NPZs are raw objective evidence; `images/` contains fixed-scale renderings; gates and input bindings are health/provenance evidence.",
        *(
            [
                f"- `{row['artifact']}` -> final state `{row['state_sha256']}`"
                for row in endpoint_rows
            ]
            or ["- No completed trajectory summary yet."]
        ),
        *[f"- `{item}` (fixed-scale objective contact sheet)" for item in objective_links],
        f"- Report-assembly evidence count: {len(evidence_paths)} files / {evidence_bytes} bytes.",
        f"- Expected final manifest artifact count: {expected_final_artifact_count}.",
        "",
        "## 16. Deliberate omissions",
        "",
        "No old confirmation data, prior-start path, multi-image data, retraining, bootstrap inference, or M=8 trajectory is included; those conclusions cannot be audited from this run.",
        "",
        "## 17. Reproduction commands",
        "",
        f"Exact invoked command: `{invocation.get('normalized_command', 'unavailable')}`.",
        f"Source revision: `{revision.get('git_head', 'unavailable')}`, dirty={revision.get('git_dirty', 'unknown')}; load-bearing closure `{manifest.get('source_fingerprint', 'unavailable')}`.",
        "Use the stage/resume forms documented in `docs/jacobi_rb_frequency1_exploratory_rollout.md`.",
        "",
        "## 18. Bundle-integrity audit",
        "",
        f"Verify `SHA256SUMS.txt`; the final manifest must contain exactly {expected_final_artifact_count} artifacts. `bundle_integrity_audit.json` records the manifest semantic/file hashes and representative NPZ open/hash checks without entering the manifest hash cycle.",
        "",
        "## 19. Exact deliverable for the receiving agent",
        "",
        "Use the saved paired trajectories and images to execute the predeclared next action, not merely to authorize another plan.",
        "",
    ]
    _atomic_text(run_dir / "HANDOFF.md", "\n".join(handoff))


def _report_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    _write_report(run_dir)
    decision = _load_semantic(run_dir / "exploratory_decision.json", "exploratory decision") if (
        run_dir / "exploratory_decision.json"
    ).is_file() else {}
    if decision:
        _status(
            run_dir,
            state="complete",
            stage="report",
            decision=decision.get("decision"),
            scientific_evidence_complete=1,
        )
    manifest = _finalize_artifacts(run_dir)
    if _directory_bytes(run_dir) > MAXIMUM_PERSISTED_BYTES and not args.test_only:
        raise RolloutCLIError(
            "persisted rollout exceeded the frozen 2 GiB budget",
            failure_domain="resource_budget",
            failure_code="rollout_storage_budget_exceeded",
        )
    return manifest


def _failure(run_dir: Path, stage: str, exc: BaseException) -> None:
    failure_domain = getattr(exc, "failure_domain", "execution_integrity")
    failure_code = getattr(exc, "failure_code", "frequency1_rollout_execution_invalid")
    resource_stop = failure_domain == "resource_budget"
    config = (
        _load_json(run_dir / "scientific_config.json")
        if (run_dir / "scientific_config.json").is_file()
        else {}
    )
    fused = _fused_mode(config)
    preflight_resource_adjudication_complete = bool(
        resource_stop
        and stage == "preflight"
        and failure_code == "rollout_main_workflow_computationally_infeasible"
        and (run_dir / "preflight/resource_projection.json").is_file()
        and (
            not fused
            or (run_dir / "preflight/objective_roles_unopened.json").is_file()
        )
    )
    objective_evidence_complete = bool(
        (run_dir / "evaluation_gate.json").is_file()
        and int(
            _load_semantic(run_dir / "evaluation_gate.json", "evaluation gate").get(
                "passed", 0
            )
        )
        == 1
    )
    scientific_evidence_complete = int(
        preflight_resource_adjudication_complete or objective_evidence_complete
    )
    if failure_code == "rollout_main_workflow_computationally_infeasible":
        recommended_next_action = (
            "the unchanged exact M=2 rollout is infeasible under the current laptop-only "
            "six-hour contract; explicitly change the cap or pivot the experiment, and do "
            "not run another packing microbenchmark"
            if fused
            else (
                "plan an exact cross-variant fused tangent-reference scheduling feasibility gate; "
                "do not generate fresh forward evidence or change the controller experiment"
            )
        )
    elif resource_stop:
        recommended_next_action = (
            "preserve completed objective evidence and review the unfinished predeclared stage"
        )
    else:
        recommended_next_action = "repair the exact blocker and rerun the frozen experiment"
    record = _semantic(
        {
            "schema": (FUSED_RUN_SCHEMA if fused else RUN_SCHEMA) + "-failure",
            "schema_version": 1,
            "stage": stage,
            "failure_domain": failure_domain,
            "failure_code": failure_code,
            "message": str(exc),
            "evaluation_status": "resource_stopped" if resource_stop else "execution_failed",
            "scientific_evidence_complete": scientific_evidence_complete,
        }
    )
    atomic_write_json(run_dir / "failure.json", record)
    decision = _semantic(
        {
            "schema": (FUSED_RUN_SCHEMA if fused else RUN_SCHEMA) + "-decision",
            "schema_version": 1,
            "decision": (
                "rollout_resource_budget_exhausted"
                if resource_stop else "rollout_integrity_invalid"
            ),
            "failure_domain": failure_domain,
            "failure_code": failure_code,
            "recommended_next_action": (
                recommended_next_action
            ),
            "evaluation_performed": int((run_dir / "evaluation_gate.json").is_file()),
            "exploratory": 1,
        }
    )
    atomic_write_json(run_dir / "exploratory_decision.json", decision)
    _status(
        run_dir,
        state="failed",
        stage=stage,
        decision=decision["decision"],
        message=str(exc),
        failure_domain=failure_domain,
        failure_code=failure_code,
        scientific_evidence_complete=scientific_evidence_complete,
    )
    _write_report(run_dir)
    _finalize_artifacts(run_dir)


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        run_dir = args.resume_run_dir.resolve()
        if not run_dir.is_dir():
            raise ArtifactCompatibilityError(f"resume run does not exist: {run_dir}")
        return run_dir, True
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (args.runs_root / f"{stamp}_{args.run_name}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, False


def _hydrate_resume_args(args: argparse.Namespace, run_dir: Path) -> None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = _load_json(manifest_path)
    if args.frequency1_run_dir is None:
        args.frequency1_run_dir = Path(manifest["frequency1_run_dir"])
    if args.source_run_dir is None:
        args.source_run_dir = Path(manifest["source_run_dir"])
    if args.continuation_run_dir is None and manifest.get("continuation_run_dir"):
        args.continuation_run_dir = Path(manifest["continuation_run_dir"])
    if args.predecessor_run_dir is None and manifest.get("predecessor_run_dir"):
        args.predecessor_run_dir = Path(manifest["predecessor_run_dir"])
    config_path = run_dir / "scientific_config.json"
    if config_path.is_file():
        config = _load_json(config_path)
        if int(config.get("objective_first_recovery", 0)):
            args.reference_backend = str(config["reference_backend"])
            args.development_anchor = str(config["development_anchor"])
            args.core_learned_gain = float(config["core_learned_gain"])
            args.gain_sweep = str(config["gain_sweep"])
            args.exact_audit_outer_steps = int(config["exact_audit_outer_steps"])
            args.maximum_main_seconds = float(config["maximum_main_wall_seconds"])


def _require_inputs(args: argparse.Namespace) -> None:
    if args.frequency1_run_dir is None or args.source_run_dir is None:
        raise ArtifactCompatibilityError(
            "frequency-one and source run directories are required for a fresh run"
        )
    args.frequency1_run_dir = args.frequency1_run_dir.resolve()
    args.source_run_dir = args.source_run_dir.resolve()
    if args.continuation_run_dir is not None:
        args.continuation_run_dir = args.continuation_run_dir.resolve()
    if args.predecessor_run_dir is not None:
        args.predecessor_run_dir = args.predecessor_run_dir.resolve()
    if not args.test_only:
        if not args.frequency1_run_dir.is_dir() or not args.source_run_dir.is_dir():
            raise ArtifactCompatibilityError("one or more parent run directories do not exist")
        if args.continuation_run_dir is not None and not args.continuation_run_dir.is_dir():
            raise ArtifactCompatibilityError("continuation carrier run directory does not exist")
        if args.predecessor_run_dir is not None and not args.predecessor_run_dir.is_dir():
            raise ArtifactCompatibilityError("recovery predecessor run directory does not exist")


def _stage_sequence(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return (
            "preflight",
            "forward",
            "development",
            "evaluation",
            "replication",
            "report",
        )
    return (stage,)


def _already_complete(run_dir: Path, stage: str) -> bool:
    files = {
        "preflight": "preflight_gate.json",
        "forward": "forward_gate.json",
        "development": "development_gate.json",
        "evaluation": "evaluation_gate.json",
        "replication": "replication_gate.json",
        "report": "REPORT.md",
    }
    name = files.get(stage)
    if name is None or not (run_dir / name).is_file():
        return False
    if stage == "report" and any(
        not (run_dir / required).is_file()
        for required in (
            "HANDOFF.md",
            "artifact_manifest.json",
            "bundle_integrity_audit.json",
            "SHA256SUMS.txt",
        )
    ):
        return False
    if name.endswith(".json"):
        gate = _load_semantic(run_dir / name, f"{stage} gate")
        if int(gate.get("passed", 0)) != 1:
            if stage == "development" and (run_dir / "exploratory_decision.json").is_file():
                decision = _load_semantic(
                    run_dir / "exploratory_decision.json", "exploratory decision"
                )
                terminal_oracle = (
                    decision.get("decision") == "development_oracle_control_failed"
                    and int(decision.get("evaluation_performed", -1)) == 0
                    and not (run_dir / "forward/evaluation").exists()
                    and not (run_dir / "evaluation").exists()
                )
                if terminal_oracle:
                    _verify_development_selection(run_dir)
                    return True
            return False
        config = _load_semantic(
            run_dir / "scientific_config.json", "scientific configuration"
        )
        if stage == "preflight" and _fused_mode(config):
            _verify_fused_preflight(run_dir)
        elif stage == "forward":
            _verify_forward_summary(run_dir, "development")
        elif stage == "development":
            _verify_development_selection(run_dir)
        elif stage == "evaluation":
            _verify_forward_summary(run_dir, "evaluation")
            paths = sorted((run_dir / "evaluation").glob("*/*/trajectory_summary.json"))
            if len(paths) != 6:
                raise ArtifactCompatibilityError("evaluation trajectory family is incomplete")
            for path in paths:
                _verify_trajectory_summary(path, run_dir)
        elif stage == "replication" and int(gate.get("replication_performed", 0)):
            _verify_forward_summary(run_dir, "replication")
            paths = sorted((run_dir / "replication/full").glob("*/trajectory_summary.json"))
            if len(paths) != 2:
                raise ArtifactCompatibilityError("replication trajectory family is incomplete")
            for path in paths:
                _verify_trajectory_summary(path, run_dir)
        return True
    if stage == "report":
        _verify_artifact_manifest(run_dir)
    return True


def _verify_artifact_manifest(run_dir: Path) -> dict[str, Any]:
    manifest = _load_semantic(run_dir / "artifact_manifest.json", "artifact manifest")
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or int(manifest.get("artifact_count", -1)) != len(rows):
        raise ArtifactCompatibilityError("artifact manifest schema changed")
    registered_paths: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ArtifactCompatibilityError("artifact manifest row changed")
        path = run_dir / str(row.get("path", ""))
        registered_paths.add(str(row.get("path", "")))
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(row.get("size", -1))
            or file_fingerprint(path) != row.get("sha256")
        ):
            raise ArtifactCompatibilityError("registered rollout artifact changed")
    actual_registered_paths = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.relative_to(run_dir).as_posix() not in _FINAL_MANIFEST_EXCLUDED
        and path.suffix != ".tmp"
    }
    if actual_registered_paths != registered_paths:
        raise ArtifactCompatibilityError("artifact manifest path set changed")
    audit_path = run_dir / "bundle_integrity_audit.json"
    completed_report = (run_dir / "REPORT.md").is_file() or (run_dir / "HANDOFF.md").is_file()
    if completed_report and not audit_path.is_file():
        raise ArtifactCompatibilityError("completed report omitted bundle integrity audit")
    if audit_path.is_file():
        audit = _load_semantic(audit_path, "bundle integrity audit")
        if (
            int(audit.get("artifact_manifest_artifact_count", -1))
            != int(manifest["artifact_count"])
            or audit.get("artifact_manifest_semantic_sha256")
            != manifest["semantic_sha256"]
            or audit.get("artifact_manifest_file_sha256")
            != file_fingerprint(run_dir / "artifact_manifest.json")
        ):
            raise ArtifactCompatibilityError("bundle integrity audit changed")
        for row in audit.get("representative_npz_opened_and_hashed", []):
            path = run_dir / str(row.get("path", ""))
            if not path.is_file() or file_fingerprint(path) != row.get("sha256"):
                raise ArtifactCompatibilityError("representative NPZ commitment changed")
            arrays = _load_npz(path)
            if sorted(arrays) != row.get("array_names"):
                raise ArtifactCompatibilityError("representative NPZ schema changed")
    checksums_path = run_dir / "SHA256SUMS.txt"
    if completed_report and not checksums_path.is_file():
        raise ArtifactCompatibilityError("completed report omitted checksum inventory")
    if checksums_path.is_file():
        measured: dict[str, str] = {}
        for line in checksums_path.read_text(encoding="utf-8").splitlines():
            if "  " not in line:
                raise ArtifactCompatibilityError("checksum inventory is malformed")
            digest, relative = line.split("  ", 1)
            measured[relative] = digest
        expected = {
            path.relative_to(run_dir).as_posix(): file_fingerprint(path)
            for path in run_dir.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS.txt" and path.suffix != ".tmp"
        }
        if measured != expected:
            raise ArtifactCompatibilityError("checksum inventory changed")
    return manifest


def _required_gate_passed(run_dir: Path, gate: str) -> bool:
    if gate == "none":
        return True
    if gate == "initialize":
        return (run_dir / "run_manifest.json").is_file()
    filename = {
        "preflight": "preflight_gate.json",
        "forward": "forward_gate.json",
        "development": "development_gate.json",
        "evaluation": "evaluation_gate.json",
        "replication": "replication_gate.json",
    }[gate]
    path = run_dir / filename
    return path.is_file() and int(_load_semantic(path, f"{gate} gate").get("passed", 0)) == 1


# ---------------------------------------------------------------------------
# Objective-first recovery successor
# ---------------------------------------------------------------------------


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _require_recovery_output_separation(args: argparse.Namespace) -> None:
    """Reject every output/resume path that could mutate the predecessor."""

    if args.predecessor_run_dir is None:
        return
    predecessor = args.predecessor_run_dir.resolve()
    candidates = [args.runs_root.resolve()]
    if args.resume_run_dir is not None:
        candidates.append(args.resume_run_dir.resolve())
    for candidate in candidates:
        if candidate == predecessor or _is_within(candidate, predecessor):
            raise ArtifactCompatibilityError(
                "recovery output path resolves to or inside the immutable predecessor"
            )


def _precheck_recovery_predecessor(args: argparse.Namespace) -> None:
    """Verify immutable top-level commitments before allocating a child run."""

    if args.predecessor_run_dir is None or args.test_only:
        return
    predecessor = args.predecessor_run_dir.resolve()
    if predecessor.name != RECOVERY_PREDECESSOR_BASENAME:
        raise ArtifactCompatibilityError("recovery predecessor basename changed")
    manifest_path = predecessor / "artifact_manifest.json"
    run_manifest_path = predecessor / "run_manifest.json"
    checksums_path = predecessor / "SHA256SUMS.txt"
    config_path = predecessor / "scientific_config.json"
    for path, expected, label in (
        (manifest_path, RECOVERY_PREDECESSOR_MANIFEST_FILE_SHA256, "manifest file"),
        (checksums_path, RECOVERY_PREDECESSOR_CHECKSUM_FILE_SHA256, "checksum file"),
    ):
        if not path.is_file() or file_fingerprint(path) != expected:
            raise ArtifactCompatibilityError(f"recovery predecessor {label} changed")
    manifest = _load_semantic(manifest_path, "recovery predecessor manifest")
    run_manifest_entries = [
        entry
        for entry in manifest.get("artifacts", ())
        if isinstance(entry, Mapping) and entry.get("path") == "run_manifest.json"
    ]
    if (
        manifest.get("semantic_sha256")
        != RECOVERY_PREDECESSOR_MANIFEST_SEMANTIC_SHA256
        or len(run_manifest_entries) != 1
        or not run_manifest_path.is_file()
        or run_manifest_entries[0].get("sha256")
        != file_fingerprint(run_manifest_path)
        or int(run_manifest_entries[0].get("size", -1))
        != int(run_manifest_path.stat().st_size)
    ):
        raise ArtifactCompatibilityError("recovery predecessor top-level binding changed")
    # Only inspect scientific fields after the immutable registry has authenticated
    # the exact run-manifest bytes.
    run_manifest = _load_semantic(
        run_manifest_path, "recovery predecessor run manifest"
    )
    config = _load_semantic(config_path, "recovery predecessor configuration")
    if (
        config.get("semantic_sha256")
        != RECOVERY_PREDECESSOR_CONFIG_SEMANTIC_SHA256
        or run_manifest.get("source_fingerprint")
        != RECOVERY_PREDECESSOR_SOURCE_FINGERPRINT
    ):
        raise ArtifactCompatibilityError("recovery predecessor top-level binding changed")


def _recovery_predecessor_paths(predecessor: Path) -> dict[int, tuple[Path, Path]]:
    root = predecessor / (
        "preflight/forward_anchor/forward_shards/fused-preflight-anchor"
    )
    return {
        anchor: (
            root / f"shard-{int(binding['shard_index']):04d}.json",
            root / f"shard-{int(binding['shard_index']):04d}.npz",
        )
        for anchor, binding in RECOVERY_ANCHOR_BINDINGS.items()
    }


def _committed_numerical_path_ids(root: Path) -> set[int]:
    """Read canonical IDs only from committed numerical records.

    Administrative plans and source declarations intentionally do not count as
    realization.  This is the narrow collision meaning frozen by the recovery.
    """

    realized: set[int] = set()
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root).as_posix()
        if not (
            "shard-" in path.name
            or path.name in {
                "trajectory_summary.json",
                "family_summary.json",
                "forward_summary.json",
            }
        ):
            continue
        try:
            value: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue

        def visit(item: Any) -> None:
            if isinstance(item, Mapping):
                if int(item.get("committed", 1)) == 0:
                    return
                for key, child in item.items():
                    if key in {"path_id", "canonical_path_id"} and isinstance(child, int):
                        realized.add(int(child))
                    elif key in {"path_ids", "canonical_path_ids"} and isinstance(child, list):
                        realized.update(int(value) for value in child if isinstance(value, int))
                    else:
                        visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(value)
    return realized


def _allocate_recovery_path_ids(
    repository_root: Path,
    *,
    predecessor: Path,
    run_dir: Path,
    test_only: bool,
) -> dict[str, Any]:
    preferred = {
        "development": PATH_IDS["development"],
        "evaluation": PATH_IDS["evaluation"],
        "optional_future": PATH_IDS["replication"],
    }
    if test_only:
        realized: set[int] = set()
    else:
        realized = set()
        runs_root = repository_root / "runs"
        if runs_root.is_dir():
            for experiment in runs_root.iterdir():
                if not experiment.is_dir():
                    continue
                for candidate in experiment.iterdir():
                    if (
                        not candidate.is_dir()
                        or candidate.resolve() in {predecessor.resolve(), run_dir.resolve()}
                    ):
                        continue
                    realized.update(_committed_numerical_path_ids(candidate))
    selected: dict[str, int] = {}
    remapped: dict[str, Any] = {}
    next_free = PATH_IDS["development"]
    for role, wanted in preferred.items():
        value = int(wanted)
        if value in realized or value in selected.values():
            while next_free in realized or next_free in selected.values():
                next_free += 1
            value = next_free
            remapped[role] = {"preferred": wanted, "selected": value}
        selected[role] = value
    return _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-path-usage",
            "schema_version": 1,
            "collision_semantics": "committed_numerical_realization_only",
            "preferred": preferred,
            "selected": selected,
            "remapped": remapped,
            "development_exact_audit_and_selected_backend_reuse_same_role_id": 1,
            "backend_and_attempt_absent_from_rng_key": 1,
            "optional_future_role_unopened": 1,
            "passed": 1,
        }
    )


def _verify_recovery_predecessor(
    predecessor: Path, *, test_only: bool
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    """Bind only load-bearing predecessor commitments and recover two anchors."""

    if test_only:
        _source, target = _test_source()
        uniform = np.full(784, 1.0 / 784.0, dtype=np.float64)
        anchors = {
            SHORT_ANCHOR: np.ascontiguousarray(0.85 * target + 0.15 * uniform),
            FULL_ANCHOR: np.ascontiguousarray(0.55 * target + 0.45 * uniform),
        }
        record = _semantic(
            {
                "schema": RECOVERY_TEST_RUN_SCHEMA + "-predecessor-binding",
                "schema_version": 1,
                "predecessor_run_dir": str(predecessor.resolve()),
                "test_only": 1,
                "artifact_count": RECOVERY_PREDECESSOR_ARTIFACT_COUNT,
                "objective_path_ids_realized": [],
                "anchors": {
                    str(key): _core_array_sha256(value[None, :])
                    for key, value in anchors.items()
                },
                "passed": 1,
                "parent_records_modified": 0,
            }
        )
        return record, anchors

    if predecessor.name != RECOVERY_PREDECESSOR_BASENAME:
        raise ArtifactCompatibilityError("recovery predecessor basename changed")
    manifest_path = predecessor / "artifact_manifest.json"
    manifest = _load_semantic(manifest_path, "recovery predecessor manifest")
    if (
        int(manifest.get("artifact_count", -1))
        != RECOVERY_PREDECESSOR_ARTIFACT_COUNT
        or manifest.get("semantic_sha256")
        != RECOVERY_PREDECESSOR_MANIFEST_SEMANTIC_SHA256
        or file_fingerprint(manifest_path)
        != RECOVERY_PREDECESSOR_MANIFEST_FILE_SHA256
        or file_fingerprint(predecessor / "SHA256SUMS.txt")
        != RECOVERY_PREDECESSOR_CHECKSUM_FILE_SHA256
    ):
        raise ArtifactCompatibilityError("recovery predecessor registry changed")
    config = _load_semantic(
        predecessor / "scientific_config.json", "recovery predecessor configuration"
    )
    run_manifest = _load_semantic(
        predecessor / "run_manifest.json", "recovery predecessor run manifest"
    )
    if (
        config.get("semantic_sha256")
        != RECOVERY_PREDECESSOR_CONFIG_SEMANTIC_SHA256
        or run_manifest.get("source_fingerprint")
        != RECOVERY_PREDECESSOR_SOURCE_FINGERPRINT
    ):
        raise ArtifactCompatibilityError("recovery predecessor source/config changed")
    input_binding = _load_semantic(
        predecessor / "input_bindings/input_binding.json",
        "recovery predecessor input binding",
    )
    if (
        file_fingerprint(predecessor / "input_bindings/checkpoint.pt")
        != CHECKPOINT_FILE_SHA256
        or input_binding.get("checkpoint_state_sha256") != CHECKPOINT_STATE_SHA256
        or input_binding.get("source_image_measure_sha256") != SOURCE_IMAGE_SHA256
        or input_binding.get("mixed_target_measure_sha256") != MIXED_TARGET_SHA256
    ):
        raise ArtifactCompatibilityError("recovery predecessor scientific inputs changed")
    status = _load_json(predecessor / "run_status.json")
    if (
        status.get("failure_code") != "fused_preflight_forward_health_invalid"
        or int(status.get("scientific_evidence_complete", -1)) != 0
    ):
        raise ArtifactCompatibilityError("recovery predecessor terminal state changed")

    shard_root = predecessor / (
        "preflight/forward_anchor/forward_shards/fused-preflight-anchor"
    )
    records: list[dict[str, Any]] = []
    for index in range(64):
        json_path = shard_root / f"shard-{index:04d}.json"
        npz_path = shard_root / f"shard-{index:04d}.npz"
        record = _load_semantic(json_path, f"predecessor forward shard {index}")
        if record.get("state_file_sha256") != file_fingerprint(npz_path):
            raise ArtifactCompatibilityError(
                f"predecessor forward shard {index} NPZ commitment changed"
            )
        arrays = _load_npz(npz_path)
        state = np.asarray(arrays.get("state"), dtype=np.float64)
        if state.shape != (1, 784):
            raise ArtifactCompatibilityError("predecessor forward state shape changed")
        enriched = dict(record)
        enriched.update(
            {
                "output_state_nonfinite_count": int(np.count_nonzero(~np.isfinite(state))),
                "output_state_negative_count": int(np.count_nonzero(state < 0.0)),
                "maximum_output_state_mass_error": float(
                    np.max(np.abs(np.sum(state, axis=1) - 1.0))
                ),
            }
        )
        records.append(enriched)
    from mnist.d0_jacobi_rb_tangent_rollout import aggregate_exact_forward_shards

    aggregate = aggregate_exact_forward_shards(
        records,
        expected_shard_count=64,
        expected_transition_count=1_404_928,
        expected_path_ids=(0xFC001,),
    )
    anchors: dict[int, np.ndarray] = {}
    anchor_records: dict[str, Any] = {}
    for anchor, (json_path, npz_path) in _recovery_predecessor_paths(predecessor).items():
        expected = RECOVERY_ANCHOR_BINDINGS[anchor]
        if (
            file_fingerprint(json_path) != expected["json_sha256"]
            or file_fingerprint(npz_path) != expected["npz_sha256"]
        ):
            raise ArtifactCompatibilityError(f"predecessor step-{anchor} anchor changed")
        record = _load_json(json_path)
        state = np.asarray(_load_npz(npz_path).get("state"), dtype=np.float64)
        if (
            state.shape != (1, 784)
            or record.get("output_state_sha256") != expected["output_state_sha256"]
            or _core_array_sha256(state) != expected["output_state_sha256"]
        ):
            raise ArtifactCompatibilityError(f"predecessor step-{anchor} state changed")
        anchors[anchor] = np.ascontiguousarray(state[0])
        anchor_records[str(anchor)] = {
            "json_path": json_path.relative_to(predecessor).as_posix(),
            "json_sha256": expected["json_sha256"],
            "npz_path": npz_path.relative_to(predecessor).as_posix(),
            "npz_sha256": expected["npz_sha256"],
            "output_state_sha256": expected["output_state_sha256"],
        }
    predecessor_realized = _committed_numerical_path_ids(predecessor)
    objective_ids = {
        PATH_IDS["development"], PATH_IDS["evaluation"], PATH_IDS["replication"]
    }
    if predecessor_realized.intersection(objective_ids):
        raise ArtifactCompatibilityError("predecessor opened an objective recovery path ID")
    if any((predecessor / name).exists() for name in ("development", "evaluation", "replication")):
        raise ArtifactCompatibilityError("predecessor opened objective trajectory artifacts")
    record = _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-predecessor-binding",
            "schema_version": 1,
            "predecessor_run_dir": str(predecessor.resolve()),
            "artifact_count": RECOVERY_PREDECESSOR_ARTIFACT_COUNT,
            "artifact_manifest_semantic_sha256": RECOVERY_PREDECESSOR_MANIFEST_SEMANTIC_SHA256,
            "artifact_manifest_file_sha256": RECOVERY_PREDECESSOR_MANIFEST_FILE_SHA256,
            "checksum_file_sha256": RECOVERY_PREDECESSOR_CHECKSUM_FILE_SHA256,
            "scientific_config_semantic_sha256": RECOVERY_PREDECESSOR_CONFIG_SEMANTIC_SHA256,
            "source_fingerprint": RECOVERY_PREDECESSOR_SOURCE_FINGERPRINT,
            "checkpoint_file_sha256": CHECKPOINT_FILE_SHA256,
            "checkpoint_state_sha256": CHECKPOINT_STATE_SHA256,
            "source_measure_sha256": SOURCE_IMAGE_SHA256,
            "mixed_target_measure_sha256": MIXED_TARGET_SHA256,
            "strict_forward_aggregate": aggregate,
            "anchors": anchor_records,
            "objective_path_ids_realized": [],
            "passed": 1,
            "parent_records_modified": 0,
        }
    )
    return record, anchors


def _recovery_adjudication(binding: Mapping[str, Any], *, test_only: bool) -> dict[str, Any]:
    return _semantic(
        {
            "schema": (
                RECOVERY_TEST_RUN_SCHEMA if test_only else RECOVERY_RUN_SCHEMA
            )
            + "-predecessor-adjudication",
            "schema_version": 1,
            "predecessor_binding_sha256": binding["semantic_sha256"],
            "decision": "fused_forward_anchor_diagnostics_aggregation_invalid",
            "failure_domain": "implementation_contract",
            "executed_forward_anchor_numerics_valid": 1,
            "resource_valid": "not_evaluated",
            "scientific_evidence_complete": 0,
            "objective_reverse_trajectory_executed": 0,
            "parent_records_modified": 0,
        }
    )


def _recovery_source_binding(run_dir: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    closure = _source_closure()
    sealed_names = (
        "mnist/d0_jacobi_rb_boundary_tangent.py",
        "mnist/d0_jacobi_rb_cuda.py",
        "mnist/d0_jacobi_rb_cuda_fused.py",
    )
    predecessor_manifest = _load_semantic(
        Path(str(binding["predecessor_run_dir"])) / "artifact_manifest.json",
        "predecessor artifact manifest",
    ) if not int(binding.get("test_only", 0)) else {"artifacts": []}
    predecessor_run_manifest = _load_semantic(
        Path(str(binding["predecessor_run_dir"])) / "run_manifest.json",
        "predecessor run manifest",
    ) if not int(binding.get("test_only", 0)) else {"source_files": []}
    historical_sources = {
        str(row["path"]): str(row["sha256"])
        for row in predecessor_run_manifest.get("source_files", [])
        if isinstance(row, Mapping)
    }
    repository_root = Path(__file__).resolve().parent.parent
    sealed = {
        name: {
            "historical_sha256": historical_sources.get(name),
            "current_sha256": file_fingerprint(repository_root / name),
            "preserved": int(
                int(binding.get("test_only", 0))
                or historical_sources.get(name) == file_fingerprint(repository_root / name)
            ),
        }
        for name in sealed_names
    }
    if not all(int(row["preserved"]) for row in sealed.values()):
        raise ArtifactCompatibilityError("sealed historical transition module changed")
    return _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-source-binding",
            "schema_version": 1,
            "current_source_fingerprint": closure["source_fingerprint"],
            "current_source_files": closure["files"],
            "predecessor_source_fingerprint": binding.get("source_fingerprint"),
            "source_discrepancy_expected_for_additive_recovery": 1,
            "sealed_historical_transition_modules": sealed,
            "sealed_historical_transition_modules_modified": 0,
            "passed": 1,
        }
    )


def _initialize_recovery(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    config_path = run_dir / "scientific_config.json"
    if config_path.is_file():
        if (run_dir / "artifact_manifest.json").is_file():
            _verify_artifact_manifest(run_dir)
        existing = _load_semantic(config_path, "recovery scientific configuration")
        if existing != _scientific_config(args):
            raise ArtifactCompatibilityError("resume recovery configuration changed")
        loaded: dict[str, dict[str, Any]] = {}
        for name in (
            "predecessor_binding.json",
            "predecessor_adjudication.json",
            "source_binding.json",
            "path_usage.json",
            "analytic_target_fraction_control.json",
        ):
            loaded[name] = _load_semantic(run_dir / name, name)
        if any(
            int(loaded[name].get("passed", 0)) != 1
            for name in (
                "predecessor_binding.json",
                "source_binding.json",
                "path_usage.json",
                "analytic_target_fraction_control.json",
            )
        ):
            raise ArtifactCompatibilityError(
                "resume recovery contains a failed load-bearing control"
            )
        closure = _source_closure()
        source_binding = loaded["source_binding.json"]
        run_manifest = _load_semantic(run_dir / "run_manifest.json", "recovery run manifest")
        predecessor_binding = loaded["predecessor_binding.json"]
        path_usage = loaded["path_usage.json"]
        if (
            source_binding.get("current_source_fingerprint")
            != closure["source_fingerprint"]
            or source_binding.get("current_source_files") != closure["files"]
            or run_manifest.get("source_fingerprint") != closure["source_fingerprint"]
            or run_manifest.get("source_files") != closure["files"]
            or run_manifest.get("scientific_config_sha256") != existing["semantic_sha256"]
            or run_manifest.get("predecessor_binding_sha256")
            != predecessor_binding["semantic_sha256"]
            or run_manifest.get("path_usage_sha256") != path_usage["semantic_sha256"]
        ):
            raise ArtifactCompatibilityError("resume recovery source closure changed")
        input_binding = _load_semantic(
            run_dir / "input_bindings/input_binding.json", "recovery input binding"
        )
        if run_manifest.get("input_binding_sha256") != input_binding["semantic_sha256"]:
            raise ArtifactCompatibilityError("resume recovery input binding changed")
        if (
            not args.test_only
            and (
                existing.get("core_learned_gain") != RECOVERY_CORE_LEARNED_GAIN
                or existing.get("exact_audit_outer_steps")
                != RECOVERY_EXACT_AUDIT_OUTER_STEPS
                or existing.get("maximum_main_wall_seconds") != MAXIMUM_MAIN_WALL_SECONDS
                or existing.get("development_anchor") != "predecessor"
            )
        ):
            raise ArtifactCompatibilityError("resume frozen production recovery choice changed")
        _verify_copied_inputs(run_dir, input_binding, test_only=bool(args.test_only))
        anchor_binding = _load_semantic(
            run_dir / "input_bindings/recovery_anchor_binding.json",
            "recovery anchor binding",
        )
        anchor_path = run_dir / "input_bindings/recovery_anchors.npz"
        if file_fingerprint(anchor_path) != anchor_binding.get("file_sha256"):
            raise ArtifactCompatibilityError("recovery anchor archive changed")
        arrays = _load_npz(anchor_path)
        if {
            name: _core_array_sha256(value.reshape(1, 784))
            for name, value in arrays.items()
        } != anchor_binding.get("array_sha256"):
            raise ArtifactCompatibilityError("recovery anchor array commitment changed")
        predecessor_anchor_hashes = (
            {
                "step_0127": predecessor_binding["anchors"]["127"],
                "step_0511": predecessor_binding["anchors"]["511"],
            }
            if args.test_only
            else {
                "step_0127": predecessor_binding["anchors"]["127"][
                    "output_state_sha256"
                ],
                "step_0511": predecessor_binding["anchors"]["511"][
                    "output_state_sha256"
                ],
            }
        )
        if (
            run_manifest.get("anchor_binding_sha256")
            != anchor_binding.get("semantic_sha256")
            or anchor_binding.get("predecessor_anchor_output_sha256")
            != predecessor_anchor_hashes
            or anchor_binding.get("array_sha256")
            != anchor_binding.get("predecessor_anchor_output_sha256")
        ):
            raise ArtifactCompatibilityError(
                "recovery anchors differ from immutable predecessor"
            )
        return existing

    try:
        binding, anchors = _verify_recovery_predecessor(
            args.predecessor_run_dir, test_only=bool(args.test_only)
        )
    except (ArtifactCompatibilityError, ExactForwardShardAggregateError):
        # A child may already exist when the shallow predecessor precheck passes
        # but its deep aggregate fails.  Preserve the Always artifacts without
        # fabricating a predecessor binding or objective authority.
        config = _scientific_config(args)
        closure = _source_closure()
        atomic_write_json(config_path, config)
        atomic_write_json(run_dir / "claim_boundary.json", _claim_boundary(args))
        atomic_write_json(
            run_dir / "run_manifest.json",
            _semantic(
                {
                    "schema": RECOVERY_RUN_SCHEMA + "-manifest",
                    "schema_version": 1,
                    "created_at": _now(),
                    "research_mode": "exploratory_engineering_then_objective_bearing",
                    "objective_bearing_experiment": 1,
                    "initialization_complete": 0,
                    "deep_predecessor_binding_unavailable": 1,
                    "device": str(args.device),
                    "predecessor_run_dir": str(args.predecessor_run_dir.resolve()),
                    "scientific_config_sha256": config["semantic_sha256"],
                    "source_fingerprint": closure["source_fingerprint"],
                    "source_files": closure["files"],
                    "invocation": {
                        "argv": list(args.invoked_argv),
                        "normalized_command": " ".join(
                            [sys.executable, "-m", __name__, *args.invoked_argv]
                        ),
                    },
                    **NO_WORK,
                }
            ),
        )
        atomic_write_json(
            run_dir / "resource_ledger.json",
            _semantic(
                {
                    "schema": RECOVERY_RUN_SCHEMA + "-resource-ledger",
                    "schema_version": 1,
                    "active_seconds": 0.0,
                    "wasted_active_seconds": 0.0,
                    "persisted_bytes": _directory_bytes(run_dir),
                    "maximum_main_seconds": float(args.maximum_main_seconds),
                    "maximum_persisted_bytes": MAXIMUM_PERSISTED_BYTES,
                    "peak_memory_fraction": 0.0,
                    "passed": 0,
                    "reason": "deep_predecessor_verification_failed",
                }
            ),
        )
        atomic_write_json(
            run_dir / "backend_decision.json",
            _semantic(
                {
                    "schema": RECOVERY_RUN_SCHEMA + "-backend-decision",
                    "schema_version": 1,
                    "phase": "not_attempted_predecessor_invalid",
                    "requested": args.reference_backend,
                    "selected": "not_attempted",
                    "exact_audit_shard_committed": 0,
                    "selected_family_complete": 0,
                }
            ),
        )
        raise
    input_binding = _copy_and_bind_inputs(run_dir, args)
    if not args.test_only and (
        input_binding.get("checkpoint_file_sha256") != binding.get("checkpoint_file_sha256")
        or input_binding.get("checkpoint_state_sha256") != binding.get("checkpoint_state_sha256")
        or input_binding.get("source_image_measure_sha256") != binding.get("source_measure_sha256")
        or input_binding.get("mixed_target_measure_sha256") != binding.get("mixed_target_measure_sha256")
    ):
        raise ArtifactCompatibilityError("recovery inputs differ from predecessor")
    path_usage = _allocate_recovery_path_ids(
        Path.cwd(),
        predecessor=args.predecessor_run_dir,
        run_dir=run_dir,
        test_only=bool(args.test_only),
    )
    config = _scientific_config(args)
    atomic_write_json(config_path, config)
    atomic_write_json(run_dir / "claim_boundary.json", _claim_boundary(args))
    atomic_write_json(run_dir / "predecessor_binding.json", binding)
    atomic_write_json(
        run_dir / "predecessor_adjudication.json",
        _recovery_adjudication(binding, test_only=bool(args.test_only)),
    )
    atomic_write_json(run_dir / "source_binding.json", _recovery_source_binding(run_dir, binding))
    atomic_write_json(run_dir / "path_usage.json", path_usage)
    anchor_artifact = _atomic_npz(
        run_dir / "input_bindings/recovery_anchors.npz",
        step_0127=anchors[SHORT_ANCHOR],
        step_0511=anchors[FULL_ANCHOR],
    )
    anchor_values = _load_npz(run_dir / "input_bindings/recovery_anchors.npz")
    predecessor_anchor_hashes = (
        {
            "step_0127": binding["anchors"]["127"],
            "step_0511": binding["anchors"]["511"],
        }
        if args.test_only
        else {
            "step_0127": binding["anchors"]["127"]["output_state_sha256"],
            "step_0511": binding["anchors"]["511"]["output_state_sha256"],
        }
    )
    anchor_binding = _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-anchor-binding",
            "schema_version": 1,
            "file_sha256": anchor_artifact["sha256"],
            "file_size": anchor_artifact["size"],
            "array_sha256": {
                name: _core_array_sha256(value.reshape(1, 784))
                for name, value in anchor_values.items()
            },
            "predecessor_anchor_output_sha256": predecessor_anchor_hashes,
            "passed": 1,
        }
    )
    atomic_write_json(
        run_dir / "input_bindings/recovery_anchor_binding.json", anchor_binding
    )
    closure = _source_closure()
    manifest = _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-manifest",
            "schema_version": 1,
            "created_at": _now(),
            "research_mode": "exploratory_engineering_then_objective_bearing",
            "objective_bearing_experiment": 1,
            "device": str(args.device),
            "predecessor_run_dir": str(args.predecessor_run_dir.resolve()),
            "frequency1_run_dir": str(args.frequency1_run_dir.resolve()),
            "source_run_dir": str(args.source_run_dir.resolve()),
            "scientific_config_sha256": config["semantic_sha256"],
            "predecessor_binding_sha256": binding["semantic_sha256"],
            "input_binding_sha256": input_binding["semantic_sha256"],
            "path_usage_sha256": path_usage["semantic_sha256"],
            "anchor_binding_sha256": anchor_binding["semantic_sha256"],
            "source_fingerprint": closure["source_fingerprint"],
            "source_files": closure["files"],
            "invocation": {
                "argv": list(args.invoked_argv),
                "normalized_command": " ".join([sys.executable, "-m", __name__, *args.invoked_argv]),
            },
            **NO_WORK,
        }
    )
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    atomic_write_json(
        run_dir / "resource_ledger.json",
        _semantic(
            {
                "schema": RECOVERY_RUN_SCHEMA + "-resource-ledger",
                "schema_version": 1,
                "maximum_main_seconds": float(args.maximum_main_seconds),
                "report_reserve_seconds": RECOVERY_REPORT_RESERVE_SECONDS,
                "projection_factor": RECOVERY_PROJECTION_FACTOR,
                "active_seconds": 0.0,
                "wasted_active_seconds": 0.0,
                "projected_remaining_seconds": 0.0,
                "persisted_bytes": _directory_bytes(run_dir),
                "maximum_persisted_bytes": MAXIMUM_PERSISTED_BYTES,
                "peak_memory_fraction": 0.0,
                "passed": 1,
            }
        ),
    )
    atomic_write_json(
        run_dir / "backend_decision.json",
        _semantic(
            {
                "schema": RECOVERY_RUN_SCHEMA + "-backend-decision",
                "schema_version": 1,
                "phase": "not_attempted_before_analytic_control",
                "requested": args.reference_backend,
                "selected": "not_attempted",
                "exact_audit_shard_committed": 0,
                "selected_family_complete": 0,
            }
        ),
    )
    from mnist.d0_jacobi_rb_tangent_rollout import target_oracle_identity_control

    analytic = target_oracle_identity_control(microsteps=MICROSTEPS)
    analytic_record = _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-analytic-target-fraction-control",
            "schema_version": 1,
            **dict(analytic),
            "passed": int(bool(analytic.get("passed", 0)) and not args.test_oracle_fail),
            "gate_type": "execution_integrity",
            "source_informed_mnist_row_part_of_this_gate": 0,
        }
    )
    atomic_write_json(run_dir / "analytic_target_fraction_control.json", analytic_record)
    if not int(analytic_record["passed"]):
        raise RolloutCLIError(
            "analytic target-fraction identity control failed",
            failure_domain="execution_integrity",
            failure_code="controller_analytic_control_invalid",
        )
    _status(run_dir, state="running", stage="initialize", message="objective-first recovery initialized")
    return config


def _recovery_anchors(run_dir: Path) -> dict[int, np.ndarray]:
    arrays = _load_npz(run_dir / "input_bindings/recovery_anchors.npz")
    return {
        SHORT_ANCHOR: _state_row(arrays["step_0127"]),
        FULL_ANCHOR: _state_row(arrays["step_0511"]),
    }


def _recovery_path_ids(run_dir: Path) -> dict[str, int]:
    record = _load_semantic(run_dir / "path_usage.json", "recovery path usage")
    selected = record.get("selected")
    if not isinstance(selected, Mapping):
        raise ArtifactCompatibilityError("recovery path mapping changed")
    return {str(key): int(value) for key, value in selected.items()}


def _recovery_resource_projection(
    *,
    active_seconds: float,
    wasted_active_seconds: float,
    observed_shard_seconds: Sequence[float],
    remaining_shards: int,
    maximum_main_seconds: float,
    persisted_bytes: int,
    projected_additional_bytes: int,
    peak_memory_fraction: float,
    test_only: bool = False,
) -> dict[str, Any]:
    if remaining_shards < 0:
        raise ValueError("remaining_shards must be nonnegative")
    observed = [float(value) for value in observed_shard_seconds]
    if any(not math.isfinite(value) or value < 0.0 for value in observed):
        raise ValueError("observed shard time must be finite and nonnegative")
    slowest = max(observed, default=0.0)
    projected_remaining = RECOVERY_PROJECTION_FACTOR * slowest * remaining_shards
    projected_total = (
        float(active_seconds)
        + float(wasted_active_seconds)
        + projected_remaining
        + RECOVERY_REPORT_RESERVE_SECONDS
    )
    projected_storage = int(persisted_bytes) + int(projected_additional_bytes)
    checks = {
        "main_time": bool(test_only or projected_total <= float(maximum_main_seconds)),
        "persisted_storage": bool(
            test_only or projected_storage <= MAXIMUM_PERSISTED_BYTES
        ),
        "memory": bool(test_only or peak_memory_fraction <= MAXIMUM_MEMORY_FRACTION),
        "numerical_integrity": True,
    }
    return _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-resource-projection",
            "schema_version": 1,
            "gate_type": "backend_selection_resource_budget",
            "active_seconds": float(active_seconds),
            "wasted_active_seconds": float(wasted_active_seconds),
            "slowest_completed_same_backend_row_count_shard_seconds": slowest,
            "remaining_shards": int(remaining_shards),
            "projection_factor": RECOVERY_PROJECTION_FACTOR,
            "projected_remaining_seconds": projected_remaining,
            "report_reserve_seconds": RECOVERY_REPORT_RESERVE_SECONDS,
            "projected_total_seconds": projected_total,
            "maximum_main_seconds": float(maximum_main_seconds),
            "persisted_bytes": int(persisted_bytes),
            "projected_additional_bytes": int(projected_additional_bytes),
            "projected_persisted_bytes": projected_storage,
            "maximum_persisted_bytes": MAXIMUM_PERSISTED_BYTES,
            "peak_memory_fraction": float(peak_memory_fraction),
            "maximum_memory_fraction": MAXIMUM_MEMORY_FRACTION,
            "throughput_is_diagnostic_not_gate": 1,
            "minimum_rate_gate_present": 0,
            "setup_only_veto_present": 0,
            "checks": checks,
            "passed": int(all(checks.values())),
        }
    )


def _candidate_backend_selected(projection: Mapping[str, Any], requested: str) -> str:
    if requested == "exact":
        if not int(projection.get("passed", 0)):
            raise RolloutCLIError(
                "requested exact core cannot fit the remaining main budget",
                failure_domain="resource_budget",
                failure_code="exact_core_resource_blocked_by_explicit_backend_choice",
            )
        return "exact"
    if requested == "candidate":
        return "candidate"
    if requested != "auto":
        raise ValueError("unknown recovery backend selection")
    return "exact" if int(projection.get("passed", 0)) else "candidate"


def _centered_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    ac = a - float(np.mean(a))
    bc = b - float(np.mean(b))
    denominator = math.sqrt(float(np.dot(ac, ac)) * float(np.dot(bc, bc)))
    if denominator == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(ac, bc) / denominator)


def _exact_candidate_audit_record(
    *,
    row_keys: Sequence[str],
    exact_state: np.ndarray,
    candidate_state: np.ndarray,
    exact_reference_rms: Sequence[float] | None = None,
    candidate_reference_rms: Sequence[float] | None = None,
    exact_controller_rms: Sequence[float] | None = None,
    candidate_controller_rms: Sequence[float] | None = None,
) -> dict[str, Any]:
    exact = np.ascontiguousarray(exact_state, dtype=np.float64)
    candidate = np.ascontiguousarray(candidate_state, dtype=np.float64)
    keys = tuple(str(value) for value in row_keys)
    if exact.shape != candidate.shape or exact.shape != (len(keys), 784):
        raise ValueError("exact/candidate audit states must be matching [P,784]")
    zeros = [0.0] * len(keys)
    er = list(exact_reference_rms if exact_reference_rms is not None else zeros)
    cr = list(candidate_reference_rms if candidate_reference_rms is not None else zeros)
    ec = list(exact_controller_rms if exact_controller_rms is not None else zeros)
    cc = list(candidate_controller_rms if candidate_controller_rms is not None else zeros)
    rows: list[dict[str, Any]] = []
    for index, key in enumerate(keys):
        difference = candidate[index] - exact[index]
        rows.append(
            {
                "row_key": key,
                "l1_state_discrepancy": float(np.sum(np.abs(difference))),
                "squared_l2_state_discrepancy": float(np.dot(difference, difference)),
                "maximum_absolute_state_discrepancy": float(np.max(np.abs(difference))),
                "total_variation_state_discrepancy": float(
                    0.5 * np.sum(np.abs(difference))
                ),
                "centered_correlation": _centered_correlation(
                    exact[index], candidate[index]
                ),
                "reference_displacement_rms_discrepancy": abs(
                    float(cr[index]) - float(er[index])
                ),
                "controller_displacement_rms_discrepancy": abs(
                    float(cc[index]) - float(ec[index])
                ),
            }
        )
    if len(keys) < 3:
        raise ValueError("audit requires zero, learned, and source-informed rows")

    def contrast(name: str, index: int) -> dict[str, Any]:
        exact_contrast = exact[index] - exact[0]
        candidate_contrast = candidate[index] - candidate[0]
        delta = candidate_contrast - exact_contrast
        discrepancy = math.sqrt(float(np.dot(delta, delta)))
        exact_norm = math.sqrt(float(np.dot(exact_contrast, exact_contrast)))
        candidate_norm = math.sqrt(float(np.dot(candidate_contrast, candidate_contrast)))
        denominator = max(exact_norm, candidate_norm, 1.0e-15)
        return {
            "contrast": name,
            "exact_l2": exact_norm,
            "candidate_l2": candidate_norm,
            "discrepancy_l2": discrepancy,
            "relative_error": discrepancy / denominator,
        }

    contrasts = {
        "learned_minus_zero": contrast("learned_minus_zero", 1),
        "source_informed_minus_zero": contrast("source_informed_minus_zero", 2),
    }
    checks = {
        "per_row_l1": all(row["l1_state_discrepancy"] <= 2.0e-2 for row in rows),
        "per_row_maximum_absolute": all(
            row["maximum_absolute_state_discrepancy"] <= 2.0e-3 for row in rows
        ),
        "per_row_centered_correlation": all(
            row["centered_correlation"] >= 0.999 for row in rows
        ),
        "learned_paired_contrast": (
            contrasts["learned_minus_zero"]["relative_error"] <= 0.25
        ),
    }
    return _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-exact-candidate-audit",
            "schema_version": 1,
            "row_metrics": rows,
            "paired_contrasts": contrasts,
            "diagnostic_guardrails": {
                "maximum_l1": 2.0e-2,
                "maximum_absolute": 2.0e-3,
                "minimum_centered_correlation": 0.999,
                "maximum_learned_contrast_relative_error": 0.25,
            },
            "checks": checks,
            "passed_diagnostic_guardrails": int(all(checks.values())),
            "gate_type": "diagnostic_threshold",
            "blocks_candidate_artifact_completion": 0,
            "proves_full_horizon_exact_equivalence": 0,
        }
    )


def _exact_candidate_row_only_audit_record(
    *, row_keys: Sequence[str], exact_state: np.ndarray, candidate_state: np.ndarray
) -> dict[str, Any]:
    exact = np.asarray(exact_state, dtype=np.float64)
    candidate = np.asarray(candidate_state, dtype=np.float64)
    if exact.shape != candidate.shape or exact.shape != (len(row_keys), 784):
        raise ValueError("row-only audit states must be matching [P,784]")
    rows = []
    for index, key in enumerate(row_keys):
        delta = candidate[index] - exact[index]
        rows.append(
            {
                "row_key": str(key),
                "l1_state_discrepancy": float(np.sum(np.abs(delta))),
                "squared_l2_state_discrepancy": float(np.dot(delta, delta)),
                "maximum_absolute_state_discrepancy": float(np.max(np.abs(delta))),
                "total_variation_state_discrepancy": float(0.5 * np.sum(np.abs(delta))),
                "centered_correlation": _centered_correlation(
                    exact[index], candidate[index]
                ),
            }
        )
    return _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-exact-candidate-row-only-audit",
            "schema_version": 1,
            "row_metrics": rows,
            "paired_zero_contrasts_not_applicable": 1,
            "reason": "all rows are learned-gain variants; no zero row was present",
            "gate_type": "diagnostic_threshold",
            "blocks_artifact_completion": 0,
        }
    )


def _recovery_test_family_states(
    anchor: np.ndarray, *, backend: str
) -> dict[str, np.ndarray]:
    """Small deterministic three-row fixture used only by CLI orchestration tests."""

    _source, target = _test_source()
    start = np.repeat(_state_row(anchor)[None, :], 3, axis=0)
    zero = np.ascontiguousarray(start[0])
    learned = np.ascontiguousarray(0.90 * start[1] + 0.10 * target)
    source_informed = np.ascontiguousarray(0.75 * start[2] + 0.25 * target)
    final = np.stack([zero, learned, source_informed])
    if backend == "candidate":
        perturbation = np.zeros_like(final)
        perturbation[:, 0] = 1.0e-7
        perturbation[:, 1] = -1.0e-7
        final = final + perturbation
    quarter = {
        "start": start,
        "progress_25": start + 0.25 * (final - start),
        "progress_50": start + 0.50 * (final - start),
        "progress_75": start + 0.75 * (final - start),
        "final": final,
    }
    return {name: np.ascontiguousarray(value) for name, value in quarter.items()}


def _recovery_capture_coordinates(
    sequence: Sequence[tuple[int, int]],
) -> dict[tuple[int, int], str]:
    normalized = tuple((int(step), int(phase)) for step, phase in sequence)
    if not normalized or len(normalized) % 4:
        raise ValueError("recovery sequence must divide into four quarters")
    return {
        normalized[len(normalized) // 4 - 1]: "progress_25",
        normalized[len(normalized) // 2 - 1]: "progress_50",
        normalized[3 * len(normalized) // 4 - 1]: "progress_75",
        normalized[-1]: "final",
    }


def _recovery_shard_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("shard-*.json")):
        if path.name.endswith(".failure.json"):
            continue
        record = _load_semantic(path, f"recovery shard {path.name}")
        if int(record.get("committed", 0)) != 1:
            raise ArtifactCompatibilityError("recovery shard is not committed")
        state_path = path.with_suffix(".npz")
        if (
            not state_path.is_file()
            or record.get("state_file_sha256") != file_fingerprint(state_path)
        ):
            raise ArtifactCompatibilityError("recovery shard state commitment changed")
        records.append(record)
    return records


def _recovery_max_committed_shard_bytes(
    root: Path, records: Sequence[Mapping[str, Any]]
) -> int:
    """Measured NPZ plus JSON bytes for a complete atomic shard commit."""

    sizes: list[int] = []
    for record in records:
        index = int(record.get("shard_index", -1))
        json_path = root / f"shard-{index:04d}.json"
        if index < 0 or not json_path.is_file():
            raise ArtifactCompatibilityError("committed shard JSON size is unavailable")
        sizes.append(int(record.get("state_file_size", 0)) + int(json_path.stat().st_size))
    return max(sizes, default=0)


def _add_recovery_wasted_active_seconds(
    run_dir: Path, *, seconds: float, operation: str
) -> dict[str, Any]:
    path = run_dir / "metrics/wasted_active_seconds.json"
    prior = _load_semantic(path, "wasted active seconds") if path.is_file() else {}
    attempts = list(prior.get("attempts", []))
    attempts.append(
        {"operation": str(operation), "elapsed_seconds": max(0.0, float(seconds))}
    )
    record = _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-wasted-active-seconds",
            "schema_version": 1,
            "attempts": attempts,
            "wasted_active_seconds": math.fsum(
                float(row["elapsed_seconds"]) for row in attempts
            ),
            "idle_time_included": 0,
        }
    )
    atomic_write_json(path, record)
    return record


def _promote_recovery_failure_states(
    run_dir: Path,
    *,
    shard_root: Path,
    family_name: str,
    backend: str,
    initial_state: np.ndarray | None = None,
) -> None:
    records = _recovery_shard_records(shard_root) if shard_root.is_dir() else []
    state = (
        np.asarray(
            _load_npz(shard_root / f"shard-{len(records)-1:04d}.npz")["state"],
            dtype=np.float64,
        )
        if records
        else np.asarray(initial_state, dtype=np.float64)
        if initial_state is not None
        else np.empty((0, 784), dtype=np.float64)
    )
    if state.ndim != 2 or state.shape[1] != 784:
        return
    destination = run_dir / "failure_artifacts" / family_name
    destination.mkdir(parents=True, exist_ok=True)
    artifact = _atomic_npz(destination / "last_valid_states.npz", state=state)
    images: list[str] = []
    for row in range(state.shape[0]):
        try:
            rendered = _render_states(
                run_dir,
                trajectory_key=f"failure-{family_name}-{backend}-row-{row}",
                states={"last_valid": state[row]},
            )
            for item in rendered.values():
                if isinstance(item, Mapping):
                    images.extend(str(path) for path in item.values())
        except Exception:
            # Failure preservation must never mask the scientific/engineering
            # exception.  The raw float64 last-valid state remains authority.
            pass
    atomic_write_json(
        destination / "failure_evidence.json",
        _semantic(
            {
                "schema": RECOVERY_RUN_SCHEMA + "-partial-failure-evidence",
                "schema_version": 1,
                "family_name": family_name,
                "backend": backend,
                "last_committed_shard_index": len(records) - 1,
                "initial_state_used_when_no_shard_committed": int(not records),
                "last_valid_states": {
                    **artifact,
                    "path": (destination / "last_valid_states.npz").relative_to(run_dir).as_posix(),
                },
                "fixed_scale_images": images,
                "numerically_interpretable_partial_evidence": int(
                    np.isfinite(state).all()
                    and np.all(state >= 0.0)
                    and np.max(np.abs(state.sum(axis=1) - 1.0)) <= MAXIMUM_MASS_ERROR
                ),
            }
        ),
    )


def _recovery_observed_resource(run_dir: Path) -> dict[str, Any]:
    shards = [
        _load_semantic(path, f"recovery shard {path.name}")
        for path in sorted(run_dir.glob("objective_attempts/*/fused_families/*/*/shard-*.json"))
        if not path.name.endswith(".failure.json")
    ]
    forward_shards = [
        _load_semantic(path, f"evaluation forward shard {path.name}")
        for path in sorted(
            run_dir.glob("objective/evaluation_forward/forward_shards/*/shard-*.json")
        )
        if not path.name.endswith(".failure.json")
    ]
    setup_seconds = 0.0
    setup_path = run_dir / "metrics/recovery_active_setup.json"
    if setup_path.is_file():
        setup_seconds = float(
            _load_semantic(setup_path, "recovery active setup").get(
                "elapsed_seconds", 0.0
            )
        )
    wasted = 0.0
    wasted_path = run_dir / "metrics/wasted_active_seconds.json"
    if wasted_path.is_file():
        wasted = float(
            _load_semantic(wasted_path, "wasted active seconds").get(
                "wasted_active_seconds", 0.0
            )
        )
    elapsed = setup_seconds + math.fsum(
        float(record.get("elapsed_seconds", 0.0))
        for record in (*shards, *forward_shards)
    )
    # Attempt timing adds JSON-commit and immediate restart verification debit
    # not contained in the frozen fused shard's NPZ-through-commit timer.
    for path in sorted(run_dir.glob("metrics/recovery_attempt_timing_*.json")):
        timing = _load_semantic(path, "recovery attempt timing")
        elapsed += max(0.0, float(timing.get("commit_verification_overhead_seconds", 0.0)))
    resume_verification_path = run_dir / "metrics/recovery_resume_verification.json"
    if resume_verification_path.is_file():
        elapsed += max(
            0.0,
            float(
                _load_semantic(
                    resume_verification_path, "recovery resume verification"
                ).get("elapsed_seconds", 0.0)
            ),
        )
    forward_timing_path = run_dir / "metrics/evaluation_forward_attempt_timing.json"
    if forward_timing_path.is_file():
        timing = _load_semantic(forward_timing_path, "evaluation forward attempt timing")
        elapsed += max(
            0.0, float(timing.get("commit_verification_overhead_seconds", 0.0))
        )
    peak = max(
        (
            max(
                int(record.get("peak_cuda_memory_allocated_bytes", 0)),
                int(record.get("peak_cuda_memory_bytes", 0)),
                int(
                    record.get("diagnostics", {})
                    .get("reference", {})
                    .get("maximum_cuda_memory_allocated", 0)
                ),
            )
            for record in (*shards, *forward_shards)
        ),
        default=0,
    )
    total = max(
        (
            max(
                int(record.get("total_cuda_memory_bytes", 0)),
                int(record.get("cuda_total_memory", 0)),
                int(
                    record.get("diagnostics", {})
                    .get("reference", {})
                    .get("total_cuda_memory_bytes", 0)
                ),
            )
            for record in (*shards, *forward_shards)
        ),
        default=0,
    )
    return {
        "active_seconds": elapsed,
        "wasted_active_seconds": wasted,
        "shard_count": len(shards) + len(forward_shards),
        "reverse_shard_count": len(shards),
        "forward_shard_count": len(forward_shards),
        "peak_memory_bytes": peak,
        "total_memory_bytes": total,
        "peak_memory_fraction": peak / total if total else 0.0,
        "persisted_bytes": _directory_bytes(run_dir),
    }


def _write_recovery_ledger(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    backend: str,
    remaining_shards: int,
    observed_shard_seconds: Sequence[float],
    projected_additional_bytes: int = 0,
) -> dict[str, Any]:
    usage = _recovery_observed_resource(run_dir)
    projection = _recovery_resource_projection(
        active_seconds=usage["active_seconds"],
        wasted_active_seconds=usage["wasted_active_seconds"],
        observed_shard_seconds=observed_shard_seconds,
        remaining_shards=remaining_shards,
        maximum_main_seconds=float(args.maximum_main_seconds),
        persisted_bytes=usage["persisted_bytes"],
        projected_additional_bytes=projected_additional_bytes,
        peak_memory_fraction=usage["peak_memory_fraction"],
        test_only=bool(args.test_only),
    )
    record = _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-resource-ledger",
            "schema_version": 1,
            "backend": backend,
            **usage,
            "projection": projection,
            "report_reserve_seconds": RECOVERY_REPORT_RESERVE_SECONDS,
            "idle_wall_time_charged": 0,
            "completed_shards_are_verify_only_on_resume": 1,
            "rates_are_diagnostics_only": 1,
            "passed": int(projection["passed"]),
        }
    )
    atomic_write_json(run_dir / "resource_ledger.json", record)
    return record


def _recovery_family_health(
    result: Any, *, backend: str, expected_transition_count: int
) -> dict[str, Any]:
    records = tuple(_field(result, "shard_records", ()))
    references: list[dict[str, Any]] = []

    def schema_invalid(message: str) -> None:
        raise RolloutCLIError(
            message,
            failure_domain="implementation_contract",
            failure_code=(
                "synchronous_exact_reference_health_schema_invalid"
                if backend == "exact"
                else "candidate_reference_health_schema_invalid"
            ),
        )

    def count(mapping: Mapping[str, Any], name: str) -> int:
        value = mapping.get(name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            schema_invalid(f"{backend} reference health field {name} changed")
        return int(value)

    if backend not in RECOVERY_REFERENCE_CONTRACTS:
        schema_invalid("recovery family backend changed")
    expected_forbidden_names = (
        RECOVERY_EXACT_FORBIDDEN_COUNTS
        if backend == "exact"
        else RECOVERY_CANDIDATE_FORBIDDEN_COUNTS
    )
    aggregate_names = (
        (
            "transition_count",
            "active_count",
            "structural_noop_count",
            "certified_count",
            "fallback_count",
            "unauthorized_count",
            "invalid_count",
        )
        if backend == "exact"
        else (
            "transition_count",
            "active_count",
            "structural_noop_count",
            "approximation_count",
            "invalid_count",
        )
    )
    for shard in records:
        if not isinstance(shard, Mapping):
            schema_invalid(f"{backend} recovery shard record changed")
        diagnostics = shard.get("diagnostics")
        reference = (
            diagnostics.get("reference")
            if isinstance(diagnostics, Mapping)
            else None
        )
        if not isinstance(reference, Mapping):
            schema_invalid(f"{backend} reference health record changed")
        normalized = dict(reference)
        rows = normalized.get("per_row")
        if not isinstance(rows, list) or not rows or any(
            not isinstance(row, Mapping) for row in rows
        ):
            schema_invalid(f"{backend} reference per-row health changed")
        for name in aggregate_names:
            direct = count(normalized, name)
            row_total = math.fsum(count(row, name) for row in rows)
            if direct != row_total:
                schema_invalid(
                    f"{backend} reference aggregate {name} differs from per-row health"
                )
        shard_transition_count = count(normalized, "transition_count")
        if shard_transition_count % len(rows) or any(
            count(row, "transition_count") != shard_transition_count // len(rows)
            for row in rows
        ):
            schema_invalid(f"{backend} reference per-row transition authority changed")
        for row in rows:
            transition = count(row, "transition_count")
            active = count(row, "active_count")
            noops = count(row, "structural_noop_count")
            if transition != active + noops:
                schema_invalid(f"{backend} reference row authorization changed")
            if backend == "exact":
                certified = count(row, "certified_count")
                fallback = count(row, "fallback_count")
                unauthorized = count(row, "unauthorized_count")
                invalid = count(row, "invalid_count")
                fraction = row.get("certificate_fraction")
                expected_fraction = certified / active if active else 1.0
                if (
                    certified != active
                    or fallback > transition
                    or unauthorized != 0
                    or invalid != 0
                    or not isinstance(fraction, (int, float))
                    or isinstance(fraction, bool)
                    or not math.isfinite(float(fraction))
                    or float(fraction) != expected_fraction
                ):
                    schema_invalid("exact reference row certificate fraction changed")
            else:
                approximated = count(row, "approximation_count")
                invalid = count(row, "invalid_count")
                width = row.get("maximum_candidate_bracket_width")
                if (
                    approximated != active
                    or invalid != 0
                    or row.get("certificate_fraction") != "not_applicable"
                    or any(
                        name in row
                        for name in (
                            "certified_count",
                            "authorized_count",
                            "certified_mask",
                        )
                    )
                    or not isinstance(width, (int, float))
                    or isinstance(width, bool)
                    or not math.isfinite(float(width))
                    or float(width) < 0.0
                ):
                    schema_invalid("candidate reference row certificate label changed")
        references.append(normalized)

    if not records:
        schema_invalid(f"{backend} recovery family has no committed shard health")
    state = np.asarray(_field(result, "final_state"), dtype=np.float64)
    raw_transition_count = _field(result, "transition_count", None)
    if (
        not isinstance(raw_transition_count, int)
        or isinstance(raw_transition_count, bool)
        or raw_transition_count < 0
    ):
        schema_invalid(f"{backend} family transition count changed")
    transition_count = int(raw_transition_count)
    active = sum(count(record, "active_count") for record in references)
    noops = sum(count(record, "structural_noop_count") for record in references)
    forbidden: dict[str, int] = {}
    for record in references:
        shard_forbidden = record.get("forbidden_counts")
        if (
            not isinstance(shard_forbidden, Mapping)
            or set(shard_forbidden) != expected_forbidden_names
        ):
            schema_invalid(f"{backend} reference forbidden-count schema changed")
        for name, value in shard_forbidden.items():
            forbidden[str(name)] = forbidden.get(str(name), 0) + count(
                shard_forbidden, str(name)
            )
    maximum_mass = float(
        _field(result, "diagnostics", {}).get("maximum_mass_error", 0.0)
    )
    finite = int(np.count_nonzero(~np.isfinite(state))) == 0
    nonnegative = int(np.count_nonzero(state < 0.0)) == 0
    simplex = float(np.max(np.abs(np.sum(state, axis=1) - 1.0)))
    checks: dict[str, bool] = {
        "transition_count": transition_count == expected_transition_count,
        "active_and_noop_complete": active + noops == transition_count,
        "finite": finite,
        "nonnegative": nonnegative,
        "simplex_mass": simplex <= MAXIMUM_MASS_ERROR,
        "pair_mass": maximum_mass <= MAXIMUM_MASS_ERROR,
        "forbidden_events": not any(forbidden.values()),
    }
    record: dict[str, Any] = {
        "schema": RECOVERY_RUN_SCHEMA + "-family-health",
        "schema_version": 1,
        "backend": backend,
        "reference_contract": (
            "certified_exact" if backend == "exact" else "candidate_approximate_v1"
        ),
        "transition_count": transition_count,
        "active_count": active,
        "structural_noop_count": noops,
        "maximum_pair_mass_error": maximum_mass,
        "maximum_simplex_mass_error": simplex,
        "state_nonfinite_count": int(np.count_nonzero(~np.isfinite(state))),
        "state_negative_count": int(np.count_nonzero(state < 0.0)),
        "forbidden_counts": forbidden,
        "checks": checks,
    }
    if backend == "exact":
        certified = sum(count(item, "certified_count") for item in references)
        fallback = sum(count(item, "fallback_count") for item in references)
        unauthorized = sum(count(item, "unauthorized_count") for item in references)
        invalid = sum(count(item, "invalid_count") for item in references)
        for item in references:
            fraction = item.get("certificate_fraction")
            expected_fraction = (
                count(item, "certified_count") / count(item, "active_count")
                if count(item, "active_count")
                else 1.0
            )
            if (
                not isinstance(fraction, (int, float))
                or isinstance(fraction, bool)
                or not math.isfinite(float(fraction))
                or float(fraction) != expected_fraction
            ):
                schema_invalid("exact reference aggregate certificate fraction changed")
        checks.update(
            {
                "exact_certification": certified == active,
                "exact_authorization": certified + noops == transition_count,
                "exact_unauthorized": unauthorized == 0,
                "exact_validity": invalid == 0,
            }
        )
        record.update(
            {
                "certified_count": certified,
                "certificate_fraction": certified / active if active else 1.0,
                "fallback_count": fallback,
                "unauthorized_count": unauthorized,
                "invalid_count": invalid,
            }
        )
    else:
        approximated = sum(count(item, "approximation_count") for item in references)
        invalid = sum(count(item, "invalid_count") for item in references)
        for item in references:
            width = item.get("maximum_candidate_bracket_width")
            if (
                item.get("reference_contract") != "candidate_approximate_v1"
                or item.get("certificate_fraction") != "not_applicable"
                or any(
                    name in item
                    for name in (
                        "certified_count",
                        "authorized_count",
                        "certified_mask",
                    )
                )
                or not isinstance(width, (int, float))
                or isinstance(width, bool)
                or not math.isfinite(float(width))
                or float(width) < 0.0
            ):
                schema_invalid("candidate reference aggregate sentinel changed")
        checks.update(
            {
                "complete_approximation_labeling": approximated == active,
                "candidate_validity": invalid == 0,
                "no_false_exact_certificate": all(
                    item.get("certificate_fraction") == "not_applicable"
                    and "certified_count" not in item
                    for item in references
                ),
            }
        )
        record.update(
            {
                "approximation_count": approximated,
                "invalid_count": invalid,
                "certificate_fraction": "not_applicable",
                "maximum_candidate_bracket_width": max(
                    (
                        float(item.get("maximum_candidate_bracket_width", 0.0))
                        for item in references
                    ),
                    default=0.0,
                ),
            }
        )
    record["passed"] = int(all(checks.values()))
    return _semantic(record)


def _commit_recovery_family_summary(
    run_dir: Path,
    *,
    destination: Path,
    result: Any,
    backend: str,
    expected_transition_count: int,
) -> dict[str, Any]:
    path = destination / "family_summary.json"
    saved = {
        str(name): np.ascontiguousarray(value, dtype=np.float64)
        for name, value in _field(result, "saved_states", {}).items()
    }
    health = _recovery_family_health(
        result, backend=backend, expected_transition_count=expected_transition_count
    )
    if not int(health["passed"]):
        raise RolloutCLIError(
            f"{backend} recovery family numerical health failed",
            failure_domain="numerical_integrity",
            failure_code=(
                "candidate_backend_numerical_integrity_invalid"
                if backend == "candidate"
                else "exact_backend_numerical_integrity_invalid"
            ),
        )
    saved_path = destination / "family_saved_states.npz"
    if path.is_file():
        existing = _load_semantic(path, "recovery family summary")
        artifact = existing.get("saved_states_artifact")
        if not isinstance(artifact, Mapping):
            raise ArtifactCompatibilityError(
                "recovery family saved-state binding changed"
            )
        expected_relative = saved_path.relative_to(run_dir).as_posix()
        if (
            artifact.get("path") != expected_relative
            or not saved_path.is_file()
            or artifact.get("sha256") != file_fingerprint(saved_path)
            or int(artifact.get("size", -1)) != int(saved_path.stat().st_size)
        ):
            raise ArtifactCompatibilityError(
                "recovery family saved-state artifact changed"
            )
        archived = _load_npz(saved_path)
        if set(archived) != set(saved) or any(
            archived[name].dtype != np.float64
            or archived[name].shape != saved[name].shape
            or _array_sha256(archived[name]) != _array_sha256(saved[name])
            for name in saved
        ):
            raise ArtifactCompatibilityError(
                "recovery family saved-state matrices changed"
            )
        expected = _semantic(
            {
                "schema": RECOVERY_RUN_SCHEMA + "-family-summary",
                "schema_version": 1,
                "backend": backend,
                "reference_contract": health["reference_contract"],
                "result": _record(result.to_record()),
                "saved_states_artifact": dict(artifact),
                "health": health,
                "passed": 1,
            }
        )
        if existing != expected:
            raise ArtifactCompatibilityError("recovery family summary changed")
        return existing
    artifact = _atomic_npz(saved_path, **saved)
    artifact["path"] = saved_path.relative_to(run_dir).as_posix()
    record = _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-family-summary",
            "schema_version": 1,
            "backend": backend,
            "reference_contract": health["reference_contract"],
            "result": _record(result.to_record()),
            "saved_states_artifact": artifact,
            "health": health,
            "passed": 1,
        }
    )
    atomic_write_json(path, record)
    return record


def _recovery_exact_shard_zero_admission(
    run_dir: Path, args: argparse.Namespace, *, mandatory_core: bool
) -> dict[str, Any]:
    """Fail before an exact sampler launch when the hard cap is already spent."""

    usage = _recovery_observed_resource(run_dir)
    admission = _recovery_resource_projection(
        active_seconds=usage["active_seconds"],
        wasted_active_seconds=usage["wasted_active_seconds"],
        observed_shard_seconds=(0.0,),
        remaining_shards=0,
        maximum_main_seconds=float(args.maximum_main_seconds),
        persisted_bytes=usage["persisted_bytes"],
        projected_additional_bytes=0,
        peak_memory_fraction=usage["peak_memory_fraction"],
        test_only=bool(args.test_only),
    )
    if not int(admission["passed"]):
        raise RolloutCLIError(
            "exact shard zero cannot be admitted within the hard cap",
            failure_domain="resource_budget",
            failure_code=(
                "exact_audit_resource_blocked"
                if mandatory_core
                else "optional_objective_resource_deferred"
            ),
        )
    return admission


def _run_recovery_core_backend(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    anchor: np.ndarray,
    backend: str,
    sequence: Sequence[tuple[int, int]],
    exact_audit_only: bool,
    rows: Sequence[tuple[str, str, str, float | None]] | None = None,
    canonical_role: str = "development",
    family_name: str = "development-core",
    segment_name: str = "short",
    stream_role: str = "frequency1-objective-first-development-v1",
    mandatory_core: bool = True,
) -> tuple[Any, dict[str, Any], Path]:
    from mnist.d0_jacobi_rb_tangent_fused import run_fused_reverse_family

    setup_started = time.perf_counter()
    path_ids = _recovery_path_ids(run_dir)
    if rows is None:
        rows = (
            ("development-core-zero", "zero", "short", None),
            ("development-core-learned-1", "learned", "short", float(args.core_learned_gain)),
            ("development-core-source-informed", "oracle", "short", None),
        )
    specs, bank, binding = _fused_controller_family(
        run_dir, args, rows=rows, canonical_path_id=path_ids[canonical_role]
    )
    destination = run_dir / "objective_attempts" / backend
    normalized_sequence = tuple(sequence)
    initial = np.repeat(_state_row(anchor)[None, :], len(specs), axis=0)
    shard_root = destination / "fused_families" / family_name / segment_name
    rng_binding = {
        "root_seed": REVERSE_ROOT_SEED,
        "stream_role": stream_role,
        "canonical_path_id": path_ids[canonical_role],
        "variant_in_rng_key": 0,
        "backend_and_attempt_in_rng_key": 0,
    }


    if shard_root.is_dir() and any(shard_root.glob("shard-*.json")):
        _verify_fused_family_prefix(
            shard_root,
            initial_state=initial,
            sequence=normalized_sequence,
            row_specs=specs,
            controller_binding=binding,
            rng_binding=rng_binding,
            family_name=family_name,
            segment_name=segment_name,
            reference_contract=RECOVERY_REFERENCE_CONTRACTS[backend],
        )
    profile = JacobiRBCudaProfile()
    device = torch.device(args.device)
    backend_key = f"{device}:{config_fingerprint(profile.to_dict())}"
    prepared = _prepared_fused_reference(device, profile)
    factory = (
        _fused_reference_factory(
            prepared=prepared, profile=profile, stream_role=stream_role
        )
        if backend == "exact"
        else _candidate_fused_reference_factory(
            prepared=prepared, profile=profile, stream_role=stream_role
        )
    )
    setup_elapsed = time.perf_counter() - setup_started
    setup_path = run_dir / "metrics/recovery_active_setup.json"
    setup_record = (
        _load_semantic(setup_path, "recovery active setup")
        if setup_path.is_file()
        else {}
    )
    attempts = list(setup_record.get("attempts", []))
    attempts.append(
        {
            "family_name": family_name,
            "segment_name": segment_name,
            "backend": backend,
            "elapsed_seconds": setup_elapsed,
        }
    )
    atomic_write_json(
        setup_path,
        _semantic(
            {
                "schema": RECOVERY_RUN_SCHEMA + "-active-setup",
                "schema_version": 1,
                "attempts": attempts,
                "elapsed_seconds": math.fsum(float(item["elapsed_seconds"]) for item in attempts),
                "includes_controller_backend_and_seed_map": 1,
                "idle_time_included": 0,
            }
        ),
    )

    token = f"{family_name}-{segment_name}-{backend}".replace("/", "-")
    timing_path = run_dir / f"metrics/recovery_attempt_timing_{token}.json"
    prior_timing = (
        _load_semantic(timing_path, "recovery attempt timing")
        if timing_path.is_file()
        else {}
    )
    prior_verified_count = int(prior_timing.get("verified_shard_count", 0))
    prior_committed = _recovery_shard_records(shard_root) if shard_root.is_dir() else []
    prior_committed_count = len(prior_committed)
    if prior_verified_count > prior_committed_count:
        raise ArtifactCompatibilityError(
            "recovery attempt timing exceeds the committed shard prefix"
        )
    prior_raw_elapsed = math.fsum(
        float(record.get("elapsed_seconds", 0.0)) for record in prior_committed
    )
    started = time.perf_counter()
    # Timing authority may trail an atomic shard commit after a hard kill.  Begin
    # at the last timed boundary so the verified orphan prefix is durably
    # reconciled using raw sampler time only; its unknown post-commit wall remains
    # explicitly unknown and is never favorably assigned as overhead.
    timed_count = prior_verified_count
    timed_raw_elapsed = math.fsum(
        float(record["elapsed_seconds"])
        for record in prior_committed[:prior_verified_count]
    )
    timed_wall = 0.0
    cumulative_overhead = float(
        prior_timing.get("commit_verification_overhead_seconds", 0.0)
    )
    prior_complete = prior_timing.get("complete_shard_seconds")
    if prior_complete is not None:
        if (
            not isinstance(prior_complete, list)
            or len(prior_complete) != prior_verified_count
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in prior_complete
            )
        ):
            raise ArtifactCompatibilityError(
                "recovery complete shard timing prefix changed"
            )
        complete_values = [float(value) for value in prior_complete]
    else:
        prior_overhead_per = cumulative_overhead / max(1, prior_verified_count)
        complete_values = [
            float(record["elapsed_seconds"]) + prior_overhead_per
            for record in prior_committed[:prior_verified_count]
        ]
    def persist_live_timing(
        records: Sequence[Mapping[str, Any]],
        *,
        charge_interval_as_commit_overhead: bool = True,
        wall_override: float | None = None,
    ) -> list[float]:
        nonlocal timed_count, timed_raw_elapsed, timed_wall, cumulative_overhead
        raw_values = [float(record["elapsed_seconds"]) for record in records]
        raw_total = math.fsum(raw_values)
        invocation_wall = (
            float(wall_override)
            if wall_override is not None
            else time.perf_counter() - started
        )
        new_count = len(records) - timed_count
        if new_count < 0 or raw_total < timed_raw_elapsed:
            raise ArtifactCompatibilityError("recovery live timing prefix regressed")
        if new_count == 0:
            return list(complete_values)
        new_raw = raw_total - timed_raw_elapsed
        interval = max(0.0, invocation_wall - timed_wall)
        new_overhead = (
            max(0.0, interval - new_raw)
            if charge_interval_as_commit_overhead
            else 0.0
        )
        cumulative_overhead += new_overhead
        per_new = new_overhead / new_count
        complete_values.extend(
            raw_values[index] + per_new
            for index in range(timed_count, len(records))
        )
        timed_count = len(records)
        timed_raw_elapsed = raw_total
        timed_wall = invocation_wall
        timing = _semantic(
            {
                "schema": RECOVERY_RUN_SCHEMA + "-attempt-timing",
                "schema_version": 1,
                "family_name": family_name,
                "segment_name": segment_name,
                "backend": backend,
                "row_count": len(specs),
                "elapsed_seconds_through_commit_and_restart_verification": float(
                    prior_timing.get(
                        "elapsed_seconds_through_commit_and_restart_verification", 0.0
                    )
                )
                + invocation_wall,
                "raw_shard_elapsed_seconds": raw_total,
                "commit_verification_overhead_seconds": cumulative_overhead,
                "complete_shard_seconds": list(complete_values),
                "verified_shard_count": len(records),
                "completed_attempt_never_retimed": 1,
            }
        )
        atomic_write_json(timing_path, timing)
        return list(complete_values)

    def persist_postprocess_overhead() -> None:
        nonlocal timed_wall, cumulative_overhead
        if not complete_values:
            return
        invocation_wall = time.perf_counter() - started
        overhead = max(0.0, invocation_wall - timed_wall)
        if overhead == 0.0:
            return
        cumulative_overhead += overhead
        complete_values[-1] += overhead
        timed_wall = invocation_wall
        atomic_write_json(
            timing_path,
            _semantic(
                {
                    "schema": RECOVERY_RUN_SCHEMA + "-attempt-timing",
                    "schema_version": 1,
                    "family_name": family_name,
                    "segment_name": segment_name,
                    "backend": backend,
                    "row_count": len(specs),
                    "elapsed_seconds_through_commit_and_restart_verification": float(
                        prior_timing.get(
                            "elapsed_seconds_through_commit_and_restart_verification",
                            0.0,
                        )
                    )
                    + invocation_wall,
                    "raw_shard_elapsed_seconds": timed_raw_elapsed,
                    "commit_verification_overhead_seconds": cumulative_overhead,
                    "complete_shard_seconds": list(complete_values),
                    "verified_shard_count": timed_count,
                    "completed_attempt_never_retimed": 1,
                }
            ),
        )

    def before(plan: Any) -> None:
        committed = _recovery_shard_records(shard_root) if shard_root.is_dir() else []
        if not committed:
            # A candidate restart has no local timing yet, but it still must not
            # launch shard zero when the remaining hard cap is already
            # insufficient.  Price it from the slowest complete matching-row
            # candidate shard, falling back conservatively to the mandatory exact
            # audit measurement.  The exact audit itself is the measurement that
            # opens backend selection and is therefore handled by its outer
            # preflight/resource authority.
            if backend != "candidate":
                _recovery_exact_shard_zero_admission(
                    run_dir, args, mandatory_core=mandatory_core
                )
                return
            estimates, measured_bytes = _recovery_matching_shard_observations(
                run_dir,
                backend="candidate",
                row_count=len(specs),
                fallback_backend="exact",
            )
            if not estimates:
                raise RolloutCLIError(
                    "candidate shard zero lacks a conservative measured resource estimate",
                    failure_domain="resource_budget",
                    failure_code=(
                        "candidate_core_resource_blocked"
                        if mandatory_core
                        else "optional_objective_resource_deferred"
                    ),
                )
            total_shards = len(normalized_sequence) // 56
            ledger = _write_recovery_ledger(
                run_dir,
                args,
                backend=backend,
                remaining_shards=total_shards,
                observed_shard_seconds=estimates,
                projected_additional_bytes=total_shards
                * max(measured_bytes, 1),
            )
            if not int(ledger["passed"]):
                raise RolloutCLIError(
                    "candidate shard zero projection exceeds remaining main budget",
                    failure_domain="resource_budget",
                    failure_code=(
                        "candidate_core_resource_blocked"
                        if mandatory_core
                        else "optional_objective_resource_deferred"
                    ),
                )
            return
        observed = persist_live_timing(committed)
        total_shards = len(normalized_sequence) // 56
        remaining = total_shards - len(committed)
        ledger = _write_recovery_ledger(
            run_dir,
            args,
            backend=backend,
            remaining_shards=remaining,
            observed_shard_seconds=observed,
            projected_additional_bytes=max(0, remaining)
            * max(_recovery_max_committed_shard_bytes(shard_root, committed), 1),
        )
        if not int(ledger["passed"]):
            raise RolloutCLIError(
                f"{backend} {'core' if mandatory_core else 'optional family'} projection exceeds remaining main budget",
                failure_domain="resource_budget",
                failure_code=(
                    "candidate_core_resource_blocked"
                    if mandatory_core and backend == "candidate"
                    else "exact_backend_switch_required"
                    if mandatory_core
                    else "optional_objective_resource_deferred"
                ),
            )

    if prior_committed_count > prior_verified_count:
        unknown_path = run_dir / "metrics/unknown_active_time.json"
        prior_unknown = _load_semantic(unknown_path, "unknown active time") if unknown_path.is_file() else {}
        events = list(prior_unknown.get("events", []))
        event = {
            "family_name": family_name, "segment_name": segment_name, "backend": backend,
            "committed_shard_count": prior_committed_count,
            "last_timed_verified_shard_count": prior_verified_count,
            "unknown_interval_count": prior_committed_count - prior_verified_count,
        }
        if event not in events:
            events.append(event)
            atomic_write_json(
                unknown_path,
                _semantic({
                    "schema": RECOVERY_RUN_SCHEMA + "-unknown-active-time",
                    "schema_version": 1, "unknown_active_time": 1, "events": events,
                    "excluded_from_computational_infeasibility_classification": 1,
                }),
            )
        # Reconcile the atomic prefix before any continuation.  The missing
        # interval is represented by unknown_active_time above; only known raw
        # sampler seconds enter complete-shard projection.
        persist_live_timing(
            prior_committed,
            charge_interval_as_commit_overhead=False,
            wall_override=0.0,
        )
    try:
        result = run_fused_reverse_family(
            torch.as_tensor(initial, dtype=torch.float64, device=device).contiguous(),
            sequence=normalized_sequence,
            output_dir=destination,
            family_name=family_name,
            segment_name=segment_name,
            row_specs=specs,
            controller_bank=bank,
            reference_factory=factory,
            controller_binding=binding,
            rng_binding=rng_binding,
            label=3,
            microsteps=MICROSTEPS,
            device=device,
            capture_coordinates={} if exact_audit_only else _recovery_capture_coordinates(normalized_sequence),
            before_uncommitted_shard=before,
            reference_contract=RECOVERY_REFERENCE_CONTRACTS[backend],
        )
    except Exception as original_exc:
        if isinstance(original_exc, ArtifactCompatibilityError):
            raise
        measured_failure = time.perf_counter() - started
        after_records = _recovery_shard_records(shard_root) if shard_root.is_dir() else []
        new_raw = math.fsum(
            float(record["elapsed_seconds"])
            for record in after_records[timed_count:]
        )
        interval = max(0.0, measured_failure - timed_wall)
        if len(after_records) > timed_count:
            # A normally closed exception after an atomic commit has no exact
            # post-commit boundary.  Keep the committed raw sampler time, advance
            # the verified count, and conservatively debit the residual interval
            # as failed-tail work rather than favorably folding it into a shard
            # projection.
            persist_live_timing(
                after_records,
                charge_interval_as_commit_overhead=False,
                wall_override=measured_failure,
            )
        wasted = max(0.0, interval - new_raw)
        if wasted > 0.0:
            _add_recovery_wasted_active_seconds(run_dir, seconds=wasted, operation=f"{family_name}/{segment_name}/{backend}")
        _promote_recovery_failure_states(run_dir, shard_root=shard_root, family_name=family_name, backend=backend, initial_state=initial)
        if isinstance(original_exc, RolloutCLIError):
            raise
        raise RolloutCLIError(
            f"{backend} fused recovery shard failed: {original_exc}",
            failure_domain="numerical_integrity",
            failure_code="candidate_backend_numerical_integrity_invalid" if backend == "candidate" else "exact_backend_numerical_integrity_invalid",
        ) from original_exc
    verified_records = _recovery_shard_records(shard_root)
    if len(verified_records) > timed_count:
        persist_live_timing(verified_records)
    expected = len(normalized_sequence) * 2 * MICROSTEPS * len(specs) * 392
    try:
        if exact_audit_only:
            health = _recovery_family_health(result, backend=backend, expected_transition_count=expected)
            if not int(health["passed"]):
                raise RolloutCLIError("exact audit shard numerical health failed", failure_domain="numerical_integrity", failure_code="exact_backend_numerical_integrity_invalid")
            summary = _semantic({
                "schema": RECOVERY_RUN_SCHEMA + "-exact-audit-attempt", "schema_version": 1,
                "backend": backend, "result": _record(result.to_record()), "health": health,
                "separate_backend_selection_attempt": 1, "scientific_trajectory_complete": 0, "passed": 1,
            })
            audit_token = f"{family_name}-{segment_name}".replace("/", "-")
            atomic_write_json(run_dir / f"exact_audit_attempt_{audit_token}.json", summary)
        else:
            summary = _commit_recovery_family_summary(run_dir, destination=shard_root, result=result, backend=backend, expected_transition_count=expected)
    except Exception as original_exc:
        if isinstance(original_exc, ArtifactCompatibilityError):
            raise
        wasted = max(0.0, time.perf_counter() - started - timed_wall)
        if wasted > 0.0:
            _add_recovery_wasted_active_seconds(
                run_dir,
                seconds=wasted,
                operation=f"{family_name}/{segment_name}/{backend}/postprocess",
            )
        _promote_recovery_failure_states(run_dir, shard_root=shard_root, family_name=family_name, backend=backend, initial_state=initial)
        raise
    new_shards_this_invocation = len(verified_records) > prior_committed_count
    if new_shards_this_invocation:
        persist_postprocess_overhead()
    verified_records = _recovery_shard_records(shard_root)
    measured_attempt = time.perf_counter() - started
    if len(verified_records) > timed_count:
        persist_live_timing(verified_records)
    if not new_shards_this_invocation:
        verification_path = run_dir / "metrics/recovery_resume_verification.json"
        prior_verification = (
            _load_semantic(verification_path, "recovery resume verification")
            if verification_path.is_file()
            else {}
        )
        attempts = list(prior_verification.get("attempts", []))
        attempts.append(
            {
                "family_name": family_name,
                "segment_name": segment_name,
                "backend": backend,
                "verified_shard_count": len(verified_records),
                "elapsed_seconds": measured_attempt,
            }
        )
        atomic_write_json(
            verification_path,
            _semantic(
                {
                    "schema": RECOVERY_RUN_SCHEMA + "-resume-verification",
                    "schema_version": 1,
                    "attempts": attempts,
                    "elapsed_seconds": math.fsum(
                        float(row["elapsed_seconds"]) for row in attempts
                    ),
                    "completed_shards_retimed": 0,
                    "active_process_verification_included": 1,
                }
            ),
        )
    return result, summary, shard_root


def _verify_candidate_fused_shard_health(
    record: Mapping[str, Any],
    *,
    expected_transitions: int,
    row_count: int,
    expected_rows: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    if expected_rows is not None:
        if expected_transitions % row_count:
            raise ArtifactCompatibilityError(
                "candidate transition count is not row-balanced"
            )
        _verify_fused_load_bearing_row_telemetry(
            record,
            expected_rows=expected_rows,
            reference_contract="candidate_approximate_v1",
            expected_transition_count_per_row=expected_transitions // row_count,
        )
    diagnostics = record.get("diagnostics", {})
    reference = diagnostics.get("reference", {}) if isinstance(diagnostics, Mapping) else {}
    per_row = record.get("per_row_diagnostics", ())
    controllers = record.get("controller_diagnostics", ())
    forbidden_names = {
        "resource_cap_count", "invalid_density_count", "clipping_count",
        "correction_count", "floor_count", "limiter_count", "projection_count",
        "renormalization_count", "nonfinite_count",
    }
    forbidden = reference.get("forbidden_counts", {}) if isinstance(reference, Mapping) else {}
    active = int(reference.get("active_count", -1))
    noops = int(reference.get("structural_noop_count", -1))
    approximation = int(reference.get("approximation_count", -1))
    invalid_names = {
        "input_invalid", "reference_fraction_invalid", "score_invalid",
        "logistic_shift_invalid", "state_invalid", "mass_invalid", "metadata_invalid",
    }
    numeric = [
        value for row in controllers if isinstance(row, Mapping)
        for value in row.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    integer_scalars = [
        diagnostics.get("transition_count"),
        diagnostics.get("maximum_launch_lanes", row_count * 392),
        reference.get("transition_count") if isinstance(reference, Mapping) else None,
        reference.get("active_count") if isinstance(reference, Mapping) else None,
        reference.get("structural_noop_count") if isinstance(reference, Mapping) else None,
        reference.get("approximation_count") if isinstance(reference, Mapping) else None,
        reference.get("invalid_count") if isinstance(reference, Mapping) else None,
        *(
            list(forbidden.values())
            if isinstance(forbidden, Mapping)
            else [None]
        ),
        *(
            [
                row.get(name)
                for row in per_row
                if isinstance(row, Mapping)
                for name in (
                    "reference_transition_count",
                    "reference_active_count",
                    "reference_structural_noop_count",
                    "reference_approximation_count",
                    "reference_invalid_count",
                )
            ]
            if isinstance(per_row, list)
            else [None]
        ),
    ]
    if (
        not isinstance(diagnostics, Mapping) or not isinstance(reference, Mapping)
        or reference.get("reference_contract") != "candidate_approximate_v1"
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in integer_scalars
        )
        or int(diagnostics.get("transition_count", -1)) != expected_transitions
        or int(reference.get("transition_count", -1)) != expected_transitions
        or active < 0 or noops < 0 or active + noops != expected_transitions
        or approximation != active or int(reference.get("invalid_count", -1)) != 0
        or reference.get("certificate_fraction") != "not_applicable"
        or diagnostics.get("certificate_fraction") != "not_applicable"
        or any(
            name in reference
            for name in ("certified_count", "authorized_count", "certified_mask")
        )
        or not isinstance(forbidden, Mapping) or set(forbidden) != forbidden_names
        or any(int(value) != 0 for value in forbidden.values())
        or not isinstance(per_row, list) or len(per_row) != row_count
        or any(not isinstance(row, Mapping) for row in per_row)
        or sum(int(row.get("reference_transition_count", -1)) for row in per_row) != expected_transitions
        or sum(int(row.get("reference_active_count", -1)) for row in per_row) != active
        or sum(int(row.get("reference_structural_noop_count", -1)) for row in per_row) != noops
        or sum(int(row.get("reference_approximation_count", -1)) for row in per_row) != approximation
        or any(int(row.get("reference_invalid_count", -1)) != 0 for row in per_row)
        or any(
            any(name in row for name in ("reference_certified_count", "reference_authorized_count"))
            for row in per_row
        )
        or any(int(row.get(name, 0)) for row in per_row for name in invalid_names)
        or not isinstance(controllers, list) or len(controllers) != row_count
        or any(not math.isfinite(float(value)) for value in numeric)
        or not math.isfinite(float(diagnostics.get("maximum_mass_error", float("nan"))))
        or float(diagnostics.get("maximum_mass_error", float("inf"))) > MAXIMUM_MASS_ERROR
        or not math.isfinite(float(reference.get("maximum_candidate_bracket_width", float("nan"))))
        or float(reference.get("maximum_candidate_bracket_width", float("inf"))) < 0.0
        or int(diagnostics.get("maximum_launch_lanes", row_count * 392)) > 4096
    ):
        raise ArtifactCompatibilityError("candidate committed shard health changed")


def _recovery_result_rms(result: Any, prefix: str) -> list[float]:
    values: list[float] = []
    for row in tuple(_field(result, "per_row_diagnostics", ())):
        values.append(float(dict(row).get(f"{prefix}_rms", 0.0)))
    return values


def _recovery_mechanism_rows(result: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    diagnostics = tuple(_field(result, "per_row_diagnostics", ()))
    shards = tuple(_field(result, "shard_records", ()))
    for index, value in enumerate(diagnostics):
        row = dict(value)
        def rms(prefix: str) -> float:
            if f"{prefix}_rms" in row:
                return float(row[f"{prefix}_rms"])
            if f"{prefix}_count" not in row or f"{prefix}_squared_sum" not in row:
                raise ArtifactCompatibilityError(
                    f"mechanism telemetry {prefix} changed"
                )
            count = int(row[f"{prefix}_count"])
            squared = float(row[f"{prefix}_squared_sum"])
            return math.sqrt(squared / count) if count > 0 and squared >= 0.0 else 0.0

        reference_rms = rms("reference_fraction_displacement")
        control_rms = rms("control_fraction_displacement")
        rows.append(
            {
                "row_index": index,
                "score_rms": rms("score"),
                "logistic_shift_rms": rms("logistic_shift"),
                "reference_fraction_displacement_rms": reference_rms,
                "control_fraction_displacement_rms": control_rms,
                "control_reference_displacement_ratio": (
                    control_rms / reference_rms if reference_rms > 0.0 else 0.0
                ),
                "target_oracle_unreachable_boundary_count": int(
                    math.fsum(
                        int(
                            shard["controller_diagnostics"][index][
                                "target_oracle_unreachable_boundary_count"
                            ]
                        )
                        for shard in shards
                    )
                    if shards
                    else row.get("target_oracle_unreachable_boundary_count", 0)
                ),
            }
        )
    return rows


def _recovery_shard_row_rms(record: Mapping[str, Any], prefix: str) -> list[float]:
    rows = record.get("per_row_diagnostics")
    if not isinstance(rows, list):
        raise ArtifactCompatibilityError("recovery shard row diagnostics changed")
    if any(
        not isinstance(row, Mapping) or f"{prefix}_rms" not in row for row in rows
    ):
        raise ArtifactCompatibilityError("recovery shard RMS telemetry changed")
    return [float(row[f"{prefix}_rms"]) for row in rows]


def _commit_recovery_objective_artifacts(
    run_dir: Path,
    *,
    anchor: np.ndarray,
    result: Any,
    backend: str,
    audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row_keys = (
        "development-core-zero",
        "development-core-learned-1",
        "development-core-source-informed",
    )
    saved = {
        str(name): np.ascontiguousarray(value, dtype=np.float64)
        for name, value in _field(result, "saved_states", {}).items()
    }
    for required in ("start", "progress_25", "progress_50", "progress_75", "final"):
        if required not in saved or saved[required].shape != (3, 784):
            raise RolloutCLIError(
                "mandatory core omitted a required three-row saved state",
                failure_domain="implementation_contract",
                failure_code="rollout_implementation_contract_invalid",
            )
    family_root = run_dir / "objective/development-core-short"
    family_root.mkdir(parents=True, exist_ok=True)
    raw = _atomic_npz(family_root / "selected_states.npz", **saved)
    raw["path"] = (family_root / "selected_states.npz").relative_to(run_dir).as_posix()
    _source, target = _source_arrays(run_dir)
    row_records: list[dict[str, Any]] = []
    mechanism_rows = _recovery_mechanism_rows(result)
    for row, key in enumerate(row_keys):
        states = {name: value[row] for name, value in saved.items()}
        images = _render_states(
            run_dir,
            trajectory_key=key,
            states=states,
        )
        metrics = {name: _metrics_dict(value, target) for name, value in states.items()}
        zero_progress = {
            name: saved[name][0] for name in saved
        }
        divergence = {
            name: {
                "squared_l2_from_zero": float(
                    np.dot(
                        states[name] - zero_progress[name],
                        states[name] - zero_progress[name],
                    )
                ),
                "l1_from_zero": float(
                    np.sum(np.abs(states[name] - zero_progress[name]))
                ),
                "total_variation_from_zero": float(
                    0.5 * np.sum(np.abs(states[name] - zero_progress[name]))
                ),
                "centered_correlation_with_zero": _centered_correlation(
                    states[name], zero_progress[name]
                ),
            }
            for name in saved
        }
        row_records.append(
            {
                "row_key": key,
                "row_index": row,
                "backend": backend,
                "canonical_path_id": _recovery_path_ids(run_dir)["development"],
                "metrics_to_mixed_target": metrics,
                "paired_zero_divergence": divergence,
                "images": images,
                "mechanism_diagnostics": (
                    mechanism_rows[row] if row < len(mechanism_rows) else {}
                ),
            }
        )
    cells: list[tuple[str, np.ndarray]] = []
    from mnist.d0_jacobi_rb_tangent_rollout import (
        fixed_rendering_scale,
        render_background_demixed,
        render_source_image,
    )

    source, _target = _source_arrays(run_dir)
    scale = fixed_rendering_scale(source, _target, LAMBDA_MIX)
    cells.append(("source", render_source_image(source, scale)))
    cells.append(("anchor", render_background_demixed(_state_row(anchor), scale)))
    for row, key in enumerate(row_keys):
        for name in ("progress_25", "progress_50", "progress_75", "final"):
            cells.append(
                (f"{key}-{name}", render_background_demixed(saved[name][row], scale))
            )
    contact = run_dir / "images/objective_core_contact_sheet.png"
    _contact_sheet(contact, cells)
    zero_final = saved["final"][0]
    learned_final = saved["final"][1]
    source_final = saved["final"][2]
    endpoint_learned_zero_squared_l2 = float(
        np.dot(learned_final - zero_final, learned_final - zero_final)
    )
    endpoint_source_zero_squared_l2 = float(
        np.dot(source_final - zero_final, source_final - zero_final)
    )
    zero_risk = _metrics_dict(zero_final, target)["squared_l2_error"]
    learned_risk = _metrics_dict(learned_final, target)["squared_l2_error"]
    source_risk = _metrics_dict(source_final, target)["squared_l2_error"]
    approximation_dominates = False
    audit_claim_guard = None
    if audit is not None:
        largest_local = max(
            float(row["squared_l2_state_discrepancy"])
            for row in audit["row_metrics"]
        )
        learned_contrast_ok = (
            float(
                audit["paired_contrasts"]["learned_minus_zero"]["relative_error"]
            )
            <= 0.25
        )
        endpoint_scale_ok = endpoint_learned_zero_squared_l2 >= 4.0 * largest_local
        approximation_dominates = not (learned_contrast_ok and endpoint_scale_ok)
        audit_claim_guard = {
            "largest_first_shard_candidate_exact_squared_l2_discrepancy": largest_local,
            "endpoint_learned_zero_squared_l2_separation": endpoint_learned_zero_squared_l2,
            "required_endpoint_multiple": 4.0,
            "learned_paired_contrast_relative_error": audit["paired_contrasts"][
                "learned_minus_zero"
            ]["relative_error"],
            "learned_contrast_guard_passed": int(learned_contrast_ok),
            "endpoint_scale_guard_passed": int(endpoint_scale_ok),
            "coarse_candidate_dynamic_claim_permitted": int(
                not approximation_dominates
            ),
        }
    if approximation_dominates:
        decision = "approximation_dominates_observed_effect"
        next_action = (
            "retain the objective images; audit more exact shards or improve the "
            "exploration backend before making a learned-utility claim"
        )
    elif learned_risk < zero_risk:
        decision = "learned_short_dynamic_signal"
        next_action = (
            "run a fresh-path evaluation; after a positive one-image result, use a "
            "small multi-image M=2 exploration with exact audit on a fixed subset"
        )
    elif math.isclose(learned_risk, zero_risk, rel_tol=0.0, abs_tol=1.0e-14):
        decision = "learned_short_control_dynamically_negligible"
        next_action = (
            "measure calibration/amplitude and compare a rollout-trained or global "
            "alternative; do not add another exactness rung"
        )
    else:
        decision = "learned_short_rollout_direction_not_useful"
        next_action = (
            "inspect sign/order and compare a materially different controller or learner"
        )
    record = _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-core-objective",
            "schema_version": 1,
            "research_mode": "exploratory",
            "objective_bearing_experiment": 1,
            "backend": backend,
            "reference_contract": (
                "certified_exact" if backend == "exact" else "candidate_approximate_v1"
            ),
            "row_order": list(row_keys),
            "raw_states": raw,
            "row_results": row_records,
            "contact_sheet": contact.relative_to(run_dir).as_posix(),
            "zero_final_squared_l2": zero_risk,
            "learned_final_squared_l2": learned_risk,
            "source_informed_final_squared_l2": source_risk,
            "learned_minus_zero_risk_improvement": zero_risk - learned_risk,
            "source_informed_minus_zero_risk_improvement": zero_risk - source_risk,
            "source_informed_composition_mismatch_diagnostic": int(
                source_risk > zero_risk
            ),
            "endpoint_learned_zero_squared_l2_separation": endpoint_learned_zero_squared_l2,
            "endpoint_source_informed_zero_squared_l2_separation": endpoint_source_zero_squared_l2,
            "exact_candidate_claim_guard": audit_claim_guard,
            "decision": decision,
            "recommended_next_action": next_action,
            "source_informed_row_is_diagnostic_not_gate": 1,
            "passed_numerical_integrity": 1,
            "completed_128_step_three_row_family": 1,
        }
    )
    atomic_write_json(run_dir / "core_objective.json", record)
    return record


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _recovery_selection_resource_snapshot(run_dir: Path) -> dict[str, Any]:
    """Capture every byte and resource record used at backend selection."""

    reverse_paths = sorted(
        path
        for path in run_dir.glob(
            "objective_attempts/*/fused_families/*/*/shard-*.json"
        )
        if not path.name.endswith(".failure.json")
    )
    forward_paths = sorted(
        path
        for path in run_dir.glob(
            "objective/evaluation_forward/forward_shards/*/shard-*.json"
        )
        if not path.name.endswith(".failure.json")
    )

    def records(paths: Sequence[Path], label: str) -> list[dict[str, Any]]:
        return [
            {
                "path": path.relative_to(run_dir).as_posix(),
                "record": _load_semantic(path, label),
            }
            for path in paths
        ]

    resource_json_paths = [
        *sorted(run_dir.glob("metrics/recovery_attempt_timing_*.json")),
        *(
            [run_dir / "metrics/recovery_active_setup.json"]
            if (run_dir / "metrics/recovery_active_setup.json").is_file()
            else []
        ),
        *(
            [run_dir / "metrics/wasted_active_seconds.json"]
            if (run_dir / "metrics/wasted_active_seconds.json").is_file()
            else []
        ),
        *(
            [run_dir / "metrics/unknown_active_time.json"]
            if (run_dir / "metrics/unknown_active_time.json").is_file()
            else []
        ),
        *(
            [run_dir / "metrics/recovery_resume_verification.json"]
            if (run_dir / "metrics/recovery_resume_verification.json").is_file()
            else []
        ),
        *(
            [run_dir / "metrics/evaluation_forward_attempt_timing.json"]
            if (run_dir / "metrics/evaluation_forward_attempt_timing.json").is_file()
            else []
        ),
    ]
    resource_records = records(
        [*reverse_paths, *forward_paths, *resource_json_paths],
        "selection-time resource evidence",
    )
    inventory = [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "sha256": file_fingerprint(path),
            "size": int(path.stat().st_size),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
    ]
    return {
        "schema": RECOVERY_RUN_SCHEMA + "-selection-resource-snapshot",
        "schema_version": 1,
        "resource_records": resource_records,
        "file_inventory": inventory,
    }


def _selection_snapshot_usage(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    resource_entries = snapshot.get("resource_records")
    inventory = snapshot.get("file_inventory")
    if not isinstance(resource_entries, list) or not isinstance(inventory, list):
        raise ArtifactCompatibilityError("selection resource snapshot changed")
    records_by_path: dict[str, Mapping[str, Any]] = {}
    for entry in resource_entries:
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("path"), str)
            or not _is_semantic_record(entry.get("record"))
            or entry["path"] in records_by_path
        ):
            raise ArtifactCompatibilityError("selection resource record changed")
        records_by_path[str(entry["path"])] = entry["record"]
    inventory_by_path: dict[str, Mapping[str, Any]] = {}
    for entry in inventory:
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("path"), str)
            or entry["path"] in inventory_by_path
            or not isinstance(entry.get("size"), int)
            or isinstance(entry.get("size"), bool)
            or int(entry["size"]) < 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))
        ):
            raise ArtifactCompatibilityError("selection resource inventory changed")
        inventory_by_path[str(entry["path"])] = entry
    for path, record in records_by_path.items():
        item = inventory_by_path.get(path)
        encoded = _canonical_json_bytes(record)
        if (
            item is None
            or int(item["size"]) != len(encoded)
            or item["sha256"] != hashlib.sha256(encoded).hexdigest()
        ):
            raise ArtifactCompatibilityError(
                "selection resource record/inventory binding changed"
            )
    reverse: list[Mapping[str, Any]] = []
    forward: list[Mapping[str, Any]] = []
    timing: list[Mapping[str, Any]] = []
    setup: Mapping[str, Any] | None = None
    wasted: Mapping[str, Any] | None = None
    resume: Mapping[str, Any] | None = None
    forward_timing: Mapping[str, Any] | None = None
    for path, record in records_by_path.items():
        if re.fullmatch(
            r"objective_attempts/[^/]+/fused_families/[^/]+/[^/]+/shard-\d{4}\.json",
            path,
        ):
            reverse.append(record)
        elif re.fullmatch(
            r"objective/evaluation_forward/forward_shards/[^/]+/shard-\d{4}\.json",
            path,
        ):
            forward.append(record)
        elif path.startswith("metrics/recovery_attempt_timing_"):
            timing.append(record)
        elif path == "metrics/recovery_active_setup.json":
            setup = record
        elif path == "metrics/wasted_active_seconds.json":
            wasted = record
        elif path == "metrics/recovery_resume_verification.json":
            resume = record
        elif path == "metrics/evaluation_forward_attempt_timing.json":
            forward_timing = record
    setup_seconds = float(setup.get("elapsed_seconds", 0.0)) if setup else 0.0
    wasted_seconds = (
        float(wasted.get("wasted_active_seconds", 0.0)) if wasted else 0.0
    )
    active = setup_seconds + math.fsum(
        float(record.get("elapsed_seconds", 0.0))
        for record in (*reverse, *forward)
    )
    active += math.fsum(
        float(record.get("commit_verification_overhead_seconds", 0.0))
        for record in timing
    )
    if resume:
        active += float(resume.get("elapsed_seconds", 0.0))
    if forward_timing:
        active += float(
            forward_timing.get("commit_verification_overhead_seconds", 0.0)
        )
    numeric = (active, setup_seconds, wasted_seconds)
    if any(not math.isfinite(value) or value < 0.0 for value in numeric):
        raise ArtifactCompatibilityError("selection resource time changed")
    peak = max(
        (
            max(
                int(record.get("peak_cuda_memory_allocated_bytes", 0)),
                int(record.get("peak_cuda_memory_bytes", 0)),
                int(
                    record.get("diagnostics", {})
                    .get("reference", {})
                    .get("maximum_cuda_memory_allocated", 0)
                ),
            )
            for record in (*reverse, *forward)
        ),
        default=0,
    )
    total = max(
        (
            max(
                int(record.get("total_cuda_memory_bytes", 0)),
                int(record.get("cuda_total_memory", 0)),
                int(
                    record.get("diagnostics", {})
                    .get("reference", {})
                    .get("total_cuda_memory_bytes", 0)
                ),
            )
            for record in (*reverse, *forward)
        ),
        default=0,
    )
    return {
        "active_seconds": active,
        "wasted_active_seconds": wasted_seconds,
        "persisted_bytes": sum(int(item["size"]) for item in inventory_by_path.values()),
        "peak_memory_fraction": peak / total if total else 0.0,
    }


def _verify_selection_resource_snapshot(
    run_dir: Path, snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    if snapshot.get("schema") != RECOVERY_RUN_SCHEMA + "-selection-resource-snapshot":
        raise ArtifactCompatibilityError("selection resource snapshot schema changed")
    resource_entries = snapshot.get("resource_records")
    inventory = snapshot.get("file_inventory")
    if not isinstance(resource_entries, list) or not isinstance(inventory, list):
        raise ArtifactCompatibilityError("selection resource snapshot changed")
    resource_by_path = {
        str(entry["path"]): entry["record"]
        for entry in resource_entries
        if isinstance(entry, Mapping) and isinstance(entry.get("path"), str)
    }
    if len(resource_by_path) != len(resource_entries):
        raise ArtifactCompatibilityError("selection resource records changed")
    appendable = {
        "metrics/recovery_active_setup.json": "attempts",
        "metrics/wasted_active_seconds.json": "attempts",
        "metrics/recovery_resume_verification.json": "attempts",
        "metrics/unknown_active_time.json": "events",
    }
    replaceable = {
        "resource_ledger.json",
        "backend_decision.json",
        "run_status.json",
    }
    for entry in inventory:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ArtifactCompatibilityError("selection resource inventory changed")
        relative = str(entry["path"])
        path = run_dir / relative
        try:
            path.resolve().relative_to(run_dir.resolve())
        except (OSError, ValueError) as exc:
            raise ArtifactCompatibilityError("selection resource path escaped run") from exc
        if not path.is_file():
            raise ArtifactCompatibilityError("selection-time resource file disappeared")
        if file_fingerprint(path) == entry.get("sha256"):
            continue
        frozen_record = resource_by_path.get(relative)
        if relative in appendable and frozen_record is not None:
            current = _load_semantic(path, "extended selection resource record")
            key = appendable[relative]
            frozen_items = frozen_record.get(key)
            current_items = current.get(key)
            if (
                not isinstance(frozen_items, list)
                or not isinstance(current_items, list)
                or current_items[: len(frozen_items)] != frozen_items
            ):
                raise ArtifactCompatibilityError(
                    "selection resource append-only prefix changed"
                )
            continue
        if relative.startswith("metrics/recovery_attempt_timing_") and frozen_record:
            current = _load_semantic(path, "extended selection attempt timing")
            frozen_count = int(frozen_record.get("verified_shard_count", -1))
            current_count = int(current.get("verified_shard_count", -1))
            frozen_complete = frozen_record.get("complete_shard_seconds")
            current_complete = current.get("complete_shard_seconds")
            if (
                current.get("family_name") != frozen_record.get("family_name")
                or current.get("segment_name") != frozen_record.get("segment_name")
                or current.get("backend") != frozen_record.get("backend")
                or current.get("row_count") != frozen_record.get("row_count")
                or current_count <= frozen_count
                or float(current.get("raw_shard_elapsed_seconds", -1.0))
                < float(frozen_record.get("raw_shard_elapsed_seconds", 0.0))
                or float(current.get("commit_verification_overhead_seconds", -1.0))
                < float(frozen_record.get("commit_verification_overhead_seconds", 0.0))
                or (
                    frozen_complete is not None
                    and (
                        not isinstance(frozen_complete, list)
                        or not isinstance(current_complete, list)
                        or current_complete[: len(frozen_complete)] != frozen_complete
                    )
                )
            ):
                raise ArtifactCompatibilityError(
                    "selection attempt timing extension changed"
                )
            continue
        if relative in replaceable or relative.startswith(
            "evaluation_"
        ) and relative.endswith("_backend_decision.json"):
            continue
        raise ArtifactCompatibilityError("selection-time resource file changed")
    return _selection_snapshot_usage(snapshot)


def _recovery_selection_projection(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    observed_shard_seconds: Sequence[float],
    remaining_shards: int,
    projected_additional_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a projection from a frozen, recipient-reconstructible snapshot."""

    snapshot = _recovery_selection_resource_snapshot(run_dir)
    usage = _selection_snapshot_usage(snapshot)
    projection = _recovery_resource_projection(
        active_seconds=usage["active_seconds"],
        wasted_active_seconds=usage["wasted_active_seconds"],
        observed_shard_seconds=observed_shard_seconds,
        remaining_shards=remaining_shards,
        maximum_main_seconds=float(args.maximum_main_seconds),
        persisted_bytes=usage["persisted_bytes"],
        projected_additional_bytes=projected_additional_bytes,
        peak_memory_fraction=usage["peak_memory_fraction"],
        test_only=bool(args.test_only),
    )
    return projection, snapshot


def _recovery_projection_input_commitments(
    run_dir: Path,
    *,
    exact_root: Path,
    remaining_shards: int,
    projection: Mapping[str, Any],
    resource_snapshot: Mapping[str, Any],
    test_only: bool,
) -> dict[str, Any]:
    """Freeze the exact selection-time slice and numerical projection inputs."""

    records = _recovery_shard_records(exact_root)
    if not records:
        raise ArtifactCompatibilityError("backend opening lacks an exact audit shard")
    family_name = exact_root.parent.name
    segment_name = exact_root.name
    timing_token = f"{family_name}-{segment_name}-exact".replace("/", "-")
    timing_path = run_dir / f"metrics/recovery_attempt_timing_{timing_token}.json"
    timing = (
        _load_semantic(timing_path, "selection-time exact attempt timing")
        if timing_path.is_file()
        else None
    )
    observed = _complete_reverse_shard_seconds(
        run_dir,
        [float(row["elapsed_seconds"]) for row in records],
        family_name=family_name,
        segment_name=segment_name,
        backend="exact",
        row_count=len(records[0].get("row_table", ())),
    )
    usage = _selection_snapshot_usage(resource_snapshot)
    if any(
        usage[name] != projection[name]
        for name in (
            "active_seconds",
            "wasted_active_seconds",
            "persisted_bytes",
            "peak_memory_fraction",
        )
    ):
        raise ArtifactCompatibilityError(
            "selection-time resource snapshot disagrees with ledger"
        )
    basis = {
        "active_seconds": float(usage["active_seconds"]),
        "wasted_active_seconds": float(usage["wasted_active_seconds"]),
        "observed_shard_seconds": observed,
        "remaining_shards": int(remaining_shards),
        "maximum_main_seconds": float(projection["maximum_main_seconds"]),
        "persisted_bytes": int(usage["persisted_bytes"]),
        "projected_additional_bytes": int(
            projection["projected_additional_bytes"]
        ),
        "peak_memory_fraction": float(usage["peak_memory_fraction"]),
    }
    reconstructed = _recovery_resource_projection(
        **basis,
        test_only=bool(test_only),
    )
    # `_recovery_resource_projection` records no test-only flag.  Its checks may
    # be bypassed only in the reduced test fixture, so preserve the originally
    # committed check values while requiring every deterministic scalar above.
    if any(
        reconstructed.get(name) != projection.get(name)
        for name in (
            "active_seconds",
            "wasted_active_seconds",
            "slowest_completed_same_backend_row_count_shard_seconds",
            "remaining_shards",
            "projection_factor",
            "projected_remaining_seconds",
            "report_reserve_seconds",
            "projected_total_seconds",
            "maximum_main_seconds",
            "persisted_bytes",
            "projected_additional_bytes",
            "projected_persisted_bytes",
            "maximum_persisted_bytes",
            "peak_memory_fraction",
            "maximum_memory_fraction",
        )
    ):
        raise ArtifactCompatibilityError("backend opening projection inputs disagree")
    return {
        "selection_exact_shard_count": len(records),
        "exact_shard_semantic_sha256": [row["semantic_sha256"] for row in records],
        "exact_state_file_sha256": [row["state_file_sha256"] for row in records],
        "remaining_shards": int(remaining_shards),
        "exact_attempt_timing_snapshot": timing,
        "resource_snapshot": resource_snapshot,
        "projection_basis": basis,
    }


def _verify_recovery_backend_opening(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    opening: Mapping[str, Any],
    exact_root: Path,
    total_shards: int,
    expected_schema: str,
    expected_audit_sha256: str,
    expected_horizon: str | None = None,
) -> str:
    current_records = _recovery_shard_records(exact_root)
    frozen = opening.get("projection_inputs")
    if not isinstance(frozen, Mapping):
        raise ArtifactCompatibilityError("backend selection opening inputs changed")
    frozen_count = frozen.get("selection_exact_shard_count")
    if (
        not isinstance(frozen_count, int)
        or isinstance(frozen_count, bool)
        or frozen_count < 1
        or frozen_count > len(current_records)
        or int(frozen.get("remaining_shards", -1))
        != int(total_shards) - frozen_count
        or frozen.get("exact_shard_semantic_sha256")
        != [row["semantic_sha256"] for row in current_records[:frozen_count]]
        or frozen.get("exact_state_file_sha256")
        != [row["state_file_sha256"] for row in current_records[:frozen_count]]
    ):
        raise ArtifactCompatibilityError("backend selection exact slice changed")
    basis = frozen.get("projection_basis")
    if not isinstance(basis, Mapping):
        raise ArtifactCompatibilityError("backend selection projection basis changed")
    expected_basis_keys = {
        "active_seconds",
        "wasted_active_seconds",
        "observed_shard_seconds",
        "remaining_shards",
        "maximum_main_seconds",
        "persisted_bytes",
        "projected_additional_bytes",
        "peak_memory_fraction",
    }
    if set(basis) != expected_basis_keys:
        raise ArtifactCompatibilityError("backend selection projection basis changed")
    resource_snapshot = frozen.get("resource_snapshot")
    if not isinstance(resource_snapshot, Mapping):
        raise ArtifactCompatibilityError("backend selection resource snapshot changed")
    snapshot_usage = _verify_selection_resource_snapshot(run_dir, resource_snapshot)
    if any(
        snapshot_usage[name] != basis[name]
        for name in (
            "active_seconds",
            "wasted_active_seconds",
            "persisted_bytes",
            "peak_memory_fraction",
        )
    ):
        raise ArtifactCompatibilityError(
            "backend selection resource inputs changed"
        )
    raw_values = [float(row["elapsed_seconds"]) for row in current_records[:frozen_count]]
    timing_snapshot = frozen.get("exact_attempt_timing_snapshot")
    if timing_snapshot is None:
        expected_observed = raw_values
    elif _is_semantic_record(timing_snapshot):
        if (
            timing_snapshot.get("family_name") != exact_root.parent.name
            or timing_snapshot.get("segment_name") != exact_root.name
            or timing_snapshot.get("backend") != "exact"
            or int(timing_snapshot.get("verified_shard_count", -1)) != frozen_count
            or int(timing_snapshot.get("row_count", -1))
            != len(current_records[0].get("row_table", ()))
        ):
            raise ArtifactCompatibilityError("selection-time exact timing changed")
        overhead = float(
            timing_snapshot.get("commit_verification_overhead_seconds", -1.0)
        )
        if not math.isfinite(overhead) or overhead < 0.0:
            raise ArtifactCompatibilityError("selection-time exact timing changed")
        frozen_complete = timing_snapshot.get("complete_shard_seconds")
        if frozen_complete is not None:
            if (
                not isinstance(frozen_complete, list)
                or len(frozen_complete) != frozen_count
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                    for value in frozen_complete
                )
            ):
                raise ArtifactCompatibilityError(
                    "selection-time complete shard timing changed"
                )
            expected_observed = [float(value) for value in frozen_complete]
        else:
            expected_observed = [value + overhead / frozen_count for value in raw_values]
        current_timing_path = run_dir / (
            "metrics/recovery_attempt_timing_"
            + f"{exact_root.parent.name}-{exact_root.name}-exact".replace("/", "-")
            + ".json"
        )
        if not current_timing_path.is_file():
            raise ArtifactCompatibilityError("selection-time exact timing disappeared")
        current_timing = _load_semantic(
            current_timing_path, "current exact attempt timing"
        )
        current_count = int(current_timing.get("verified_shard_count", -1))
        if current_count < frozen_count:
            raise ArtifactCompatibilityError("selection-time exact timing regressed")
        if current_count == frozen_count and current_timing != timing_snapshot:
            raise ArtifactCompatibilityError("selection-time exact timing changed")
    else:
        raise ArtifactCompatibilityError("selection-time exact timing changed")
    if list(basis.get("observed_shard_seconds", ())) != expected_observed:
        raise ArtifactCompatibilityError("selection-time complete shard timing changed")
    try:
        reconstructed = _recovery_resource_projection(
            active_seconds=float(basis["active_seconds"]),
            wasted_active_seconds=float(basis["wasted_active_seconds"]),
            observed_shard_seconds=[
                float(value) for value in basis["observed_shard_seconds"]
            ],
            remaining_shards=int(basis["remaining_shards"]),
            maximum_main_seconds=float(basis["maximum_main_seconds"]),
            persisted_bytes=int(basis["persisted_bytes"]),
            projected_additional_bytes=int(basis["projected_additional_bytes"]),
            peak_memory_fraction=float(basis["peak_memory_fraction"]),
            test_only=bool(args.test_only),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactCompatibilityError(
            "backend selection projection basis is invalid"
        ) from exc
    if (
        opening.get("schema") != expected_schema
        or opening.get("requested") != args.reference_backend
        or opening.get("exact_audit_semantic_sha256") != expected_audit_sha256
        or (
            expected_horizon is not None
            and opening.get("horizon") != expected_horizon
        )
        or not _is_semantic_record(opening.get("exact_projection"))
        or opening.get("exact_projection") != reconstructed
    ):
        raise ArtifactCompatibilityError("backend selection opening changed")
    selected = _candidate_backend_selected(
        opening["exact_projection"], args.reference_backend
    )
    if opening.get("selected") != selected:
        raise ArtifactCompatibilityError("backend selection opening result changed")
    return selected


def _recovery_switch_event(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    exact_root: Path,
    total_shards: int,
    exact_audit_sha256: str,
    selection_opening_sha256: str,
    schema: str,
    restart_field: str,
    horizon: str | None = None,
) -> dict[str, Any]:
    records = _recovery_shard_records(exact_root)
    remaining = max(0, int(total_shards) - len(records))
    complete = _complete_reverse_shard_seconds(
        run_dir,
        [float(row["elapsed_seconds"]) for row in records],
        family_name=exact_root.parent.name,
        segment_name=exact_root.name,
        backend="exact",
        row_count=len(records[0].get("row_table", ())),
    )
    projected_bytes = remaining * _recovery_max_committed_shard_bytes(
        exact_root, records
    )
    projection, snapshot = _recovery_selection_projection(
        run_dir,
        args,
        observed_shard_seconds=complete,
        remaining_shards=remaining,
        projected_additional_bytes=projected_bytes,
    )
    if int(projection.get("passed", 1)) != 0:
        raise ArtifactCompatibilityError(
            "exact degradation switch lacks a failing resource projection"
        )
    record: dict[str, Any] = {
        "schema": schema,
        "schema_version": 1,
        "requested": args.reference_backend,
        "selected": "candidate",
        "exact_audit_semantic_sha256": exact_audit_sha256,
        "selection_opening_sha256": selection_opening_sha256,
        "exact_projection": projection,
        "failed_exact_projection": projection,
        "projection_inputs": _recovery_projection_input_commitments(
            run_dir,
            exact_root=exact_root,
            remaining_shards=remaining,
            projection=projection,
            resource_snapshot=snapshot,
            test_only=bool(args.test_only),
        ),
        restart_field: 1,
    }
    if horizon is not None:
        record["horizon"] = horizon
    return _semantic(record)


def _verify_recovery_switch_event(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    event: Mapping[str, Any],
    exact_root: Path,
    total_shards: int,
    exact_audit_sha256: str,
    selection_opening_sha256: str,
    expected_schema: str,
    restart_field: str,
    horizon: str | None = None,
) -> dict[str, Any]:
    selected = _verify_recovery_backend_opening(
        run_dir,
        args,
        opening=event,
        exact_root=exact_root,
        total_shards=total_shards,
        expected_schema=expected_schema,
        expected_audit_sha256=exact_audit_sha256,
        expected_horizon=horizon,
    )
    if (
        selected != "candidate"
        or event.get("selection_opening_sha256") != selection_opening_sha256
        or event.get("failed_exact_projection") != event.get("exact_projection")
        or int(event.get(restart_field, 0)) != 1
    ):
        raise ArtifactCompatibilityError("backend switch authority changed")
    return dict(event)


def _verify_recovery_completed_family_artifacts(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    backend: str,
    family_name: str,
    segment_name: str,
    rows: Sequence[tuple[str, str, str, float | None]],
    canonical_role: str,
    stream_role: str,
    anchor: np.ndarray,
    sequence: Sequence[tuple[int, int]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path_ids = _recovery_path_ids(run_dir)
    specs, _bank, binding = _fused_controller_family(
        run_dir,
        args,
        rows=rows,
        canonical_path_id=path_ids[canonical_role],
    )
    initial = np.repeat(_state_row(anchor)[None, :], len(specs), axis=0)
    root = (
        run_dir
        / f"objective_attempts/{backend}/fused_families/{family_name}/{segment_name}"
    )
    prefix = _verify_fused_family_prefix(
        root,
        initial_state=initial,
        sequence=sequence,
        row_specs=specs,
        controller_binding=binding,
        rng_binding={
            "root_seed": REVERSE_ROOT_SEED,
            "stream_role": stream_role,
            "canonical_path_id": path_ids[canonical_role],
            "variant_in_rng_key": 0,
            "backend_and_attempt_in_rng_key": 0,
        },
        family_name=family_name,
        segment_name=segment_name,
        reference_contract=RECOVERY_REFERENCE_CONTRACTS[backend],
    )
    if prefix["shard_count"] != len(sequence) // 56:
        raise ArtifactCompatibilityError("completed recovery family is incomplete")
    summary = _load_semantic(root / "family_summary.json", "recovery family summary")
    if (
        summary.get("schema") != RECOVERY_RUN_SCHEMA + "-family-summary"
        or summary.get("backend") != backend
        or summary.get("reference_contract")
        != (
            "certified_exact"
            if backend == "exact"
            else "candidate_approximate_v1"
        )
        or int(summary.get("passed", 0)) != 1
        or int(summary.get("health", {}).get("passed", 0)) != 1
        or int(summary.get("health", {}).get("transition_count", -1))
        != prefix["transition_count"]
    ):
        raise ArtifactCompatibilityError("completed recovery family summary changed")
    artifact = summary.get("saved_states_artifact")
    if not isinstance(artifact, Mapping):
        raise ArtifactCompatibilityError("recovery family saved-state binding changed")
    saved_path = run_dir / str(artifact.get("path", ""))
    if (
        saved_path != root / "family_saved_states.npz"
        or not saved_path.is_file()
        or artifact.get("sha256") != file_fingerprint(saved_path)
        or int(artifact.get("size", -1)) != int(saved_path.stat().st_size)
    ):
        raise ArtifactCompatibilityError("recovery family saved states changed")
    saved = {
        name: np.ascontiguousarray(value, dtype=np.float64)
        for name, value in _load_npz(saved_path).items()
    }
    if (
        set(saved)
        != {"start", "progress_25", "progress_50", "progress_75", "final"}
        or any(value.shape != (len(rows), 784) for value in saved.values())
        or any(not np.isfinite(value).all() or np.any(value < 0.0) for value in saved.values())
        or any(
            float(np.max(np.abs(value.sum(axis=1) - 1.0)))
            > MAXIMUM_MASS_ERROR
            for value in saved.values()
        )
        or _core_array_sha256(saved["final"]) != prefix["last_state_sha256"]
    ):
        raise ArtifactCompatibilityError("recovery family saved matrices changed")
    _verify_recovery_family_captures(
        root,
        initial=initial,
        sequence=sequence,
        saved=saved,
    )
    # Rebuild the authoritative family aggregate from the committed shard chain.
    # The summary is derived evidence and cannot be trusted merely because its own
    # semantic hash is internally consistent.
    records = tuple(_recovery_shard_records(root))
    per_row: list[dict[str, Any]] = []
    for row_index, spec in enumerate(specs):
        aggregate_row: dict[str, Any] = {"row_key": spec.row_key}
        keys = {
            key
            for shard in records
            for key in shard["per_row_diagnostics"][row_index]
            if key != "row_key"
        }
        if any(
            set(shard["per_row_diagnostics"][row_index]) - {"row_key"} != keys
            for shard in records
        ):
            raise ArtifactCompatibilityError(
                "recovery family row telemetry schema changed"
            )
        for key in keys:
            values = [
                shard["per_row_diagnostics"][row_index][key]
                for shard in records
            ]
            if key.endswith("count") or key in {
                "transition_count",
                "input_invalid",
                "reference_fraction_invalid",
                "score_invalid",
                "logistic_shift_invalid",
                "state_invalid",
                "mass_invalid",
                "metadata_invalid",
            }:
                aggregate_row[key] = sum(int(value) for value in values)
            elif "maximum" in key:
                aggregate_row[key] = max(float(value) for value in values)
            elif key.endswith("squared_sum"):
                aggregate_row[key] = math.fsum(float(value) for value in values)
        per_row.append(aggregate_row)
    diagnostics: dict[str, Any] = {
        "initial_state_sha256": _core_array_sha256(initial),
        "final_state_sha256": _core_array_sha256(saved["final"]),
        "restart_chain_valid": 1,
        "shard_count": len(records),
        "row_count": len(specs),
        "transition_count": prefix["transition_count"],
        "synchronous_replay_count": sum(
            int(item.get("synchronous_replay_performed", 0)) for item in records
        ),
        "maximum_mass_error": max(
            (
                float(item.get("diagnostics", {}).get("maximum_mass_error", 0.0))
                for item in records
            ),
            default=0.0,
        ),
        "certificate_fraction": (
            min(
                float(item.get("diagnostics", {}).get("certificate_fraction", 1.0))
                for item in records
            )
            if backend == "exact"
            else "not_applicable"
        ),
    }
    if backend == "candidate":
        references = [
            item.get("diagnostics", {}).get("reference", {}) for item in records
        ]
        diagnostics.update(
            reference_contract="candidate_approximate_v1",
            approximation_count=sum(
                int(item.get("approximation_count", 0)) for item in references
            ),
            invalid_count=sum(int(item.get("invalid_count", 0)) for item in references),
            maximum_candidate_bracket_width=max(
                (
                    float(item.get("maximum_candidate_bracket_width", 0.0))
                    for item in references
                ),
                default=0.0,
            ),
        )
    from mnist.d0_jacobi_rb_tangent_fused import FusedReverseFamilyResult

    reconstructed = FusedReverseFamilyResult(
        final_state=saved["final"],
        saved_states=saved,
        row_specs=tuple(specs),
        per_row_diagnostics=tuple(per_row),
        diagnostics=diagnostics,
        elapsed_seconds=math.fsum(
            float(item.get("elapsed_seconds", 0.0)) for item in records
        ),
        transition_count=prefix["transition_count"],
        shard_records=records,
    )
    expected_health = _recovery_family_health(
        reconstructed,
        backend=backend,
        expected_transition_count=prefix["transition_count"],
    )
    expected_summary = _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-family-summary",
            "schema_version": 1,
            "backend": backend,
            "reference_contract": expected_health["reference_contract"],
            "result": _record(reconstructed.to_record()),
            "saved_states_artifact": dict(artifact),
            "health": expected_health,
            "passed": 1,
        }
    )
    if summary != expected_summary:
        raise ArtifactCompatibilityError("completed recovery family summary changed")
    return summary, saved


def _verify_recovery_candidate_audit(
    run_dir: Path,
    *,
    audit_path: Path,
    exact_root: Path,
    candidate_root: Path,
    row_keys: Sequence[str],
) -> dict[str, Any]:
    exact_record = _recovery_shard_records(exact_root)[0]
    candidate_record = _recovery_shard_records(candidate_root)[0]
    expected = _exact_candidate_audit_record(
        row_keys=tuple(row_keys),
        exact_state=_load_npz(exact_root / "shard-0000.npz")["state"],
        candidate_state=_load_npz(candidate_root / "shard-0000.npz")["state"],
        exact_reference_rms=_recovery_shard_row_rms(
            exact_record, "reference_fraction_displacement"
        ),
        candidate_reference_rms=_recovery_shard_row_rms(
            candidate_record, "reference_fraction_displacement"
        ),
        exact_controller_rms=_recovery_shard_row_rms(
            exact_record, "control_fraction_displacement"
        ),
        candidate_controller_rms=_recovery_shard_row_rms(
            candidate_record, "control_fraction_displacement"
        ),
    )
    existing = _load_semantic(audit_path, "exact candidate audit")
    if existing != expected:
        raise ArtifactCompatibilityError("exact candidate audit changed")
    return existing


def _verify_recovery_family_captures(
    root: Path,
    *,
    initial: np.ndarray,
    sequence: Sequence[tuple[int, int]],
    saved: Mapping[str, np.ndarray],
) -> None:
    """Bind every reported quarter matrix to its authoritative shard output."""

    normalized = tuple((int(step), int(phase)) for step, phase in sequence)
    if _array_sha256(np.asarray(saved.get("start"))) != _array_sha256(initial):
        raise ArtifactCompatibilityError("recovery family start capture changed")
    captures = _recovery_capture_coordinates(normalized)
    found: set[str] = set()
    for index in range((len(normalized) + 55) // 56):
        shard_sequence = normalized[index * 56 : (index + 1) * 56]
        if not shard_sequence:
            continue
        name = captures.get(shard_sequence[-1])
        if name is None:
            continue
        path = root / f"shard-{index:04d}.npz"
        state = np.asarray(_load_npz(path).get("state"), dtype=np.float64)
        if name not in saved or _array_sha256(state) != _array_sha256(saved[name]):
            raise ArtifactCompatibilityError(
                f"recovery family {name} capture changed"
            )
        found.add(name)
    if found != {"progress_25", "progress_50", "progress_75", "final"}:
        raise ArtifactCompatibilityError("recovery family capture coverage changed")


def _verify_recovery_core_artifact(
    run_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    record = _load_semantic(run_dir / "core_objective.json", "core objective")
    decision = _load_semantic(run_dir / "backend_decision.json", "backend decision")
    backend = str(decision.get("selected", ""))
    if (
        record.get("schema")
        not in {
            RECOVERY_RUN_SCHEMA + "-core-objective",
            RECOVERY_TEST_RUN_SCHEMA + "-core-objective",
        }
        or int(record.get("completed_128_step_three_row_family", 0)) != 1
        or int(record.get("passed_numerical_integrity", 0)) != 1
        or record.get("backend") != backend
        or backend not in {"exact", "candidate"}
        or record.get("row_order")
        != [
            "development-core-zero",
            "development-core-learned-1",
            "development-core-source-informed",
        ]
    ):
        raise ArtifactCompatibilityError("core objective binding changed")
    raw = record.get("raw_states")
    if not isinstance(raw, Mapping):
        raise ArtifactCompatibilityError("core raw-state binding changed")
    raw_path = run_dir / str(raw.get("path", ""))
    if (
        raw_path != run_dir / "objective/development-core-short/selected_states.npz"
        or not raw_path.is_file()
        or raw.get("sha256") != file_fingerprint(raw_path)
        or int(raw.get("size", -1)) != int(raw_path.stat().st_size)
    ):
        raise ArtifactCompatibilityError("core raw-state artifact changed")
    saved = {
        name: np.ascontiguousarray(value, dtype=np.float64)
        for name, value in _load_npz(raw_path).items()
    }
    family_summary: Mapping[str, Any] | None = None
    if not args.test_only:
        family_summary, family_saved = _verify_recovery_completed_family_artifacts(
            run_dir,
            args,
            backend=backend,
            family_name="development-core",
            segment_name="short",
            rows=(
                ("development-core-zero", "zero", "short", None),
                (
                    "development-core-learned-1",
                    "learned",
                    "short",
                    float(args.core_learned_gain),
                ),
                ("development-core-source-informed", "oracle", "short", None),
            ),
            canonical_role="development",
            stream_role="frequency1-objective-first-development-v1",
            anchor=_recovery_anchors(run_dir)[SHORT_ANCHOR],
            sequence=_reverse_sequence(SHORT_ANCHOR),
        )
        if set(saved) != set(family_saved) or any(
            _array_sha256(saved[name]) != _array_sha256(family_saved[name])
            for name in saved
        ):
            raise ArtifactCompatibilityError("core/family saved states disagree")
    _source, target = _source_arrays(run_dir)
    final = np.asarray(saved.get("final"))
    if final.dtype != np.float64 or final.shape != (3, 784):
        raise ArtifactCompatibilityError("core final-state shape changed")
    risks = [_metrics_dict(final[index], target)["squared_l2_error"] for index in range(3)]
    mechanism = (
        _recovery_mechanism_rows(family_summary["result"])
        if family_summary is not None
        else [
            {
                "row_index": index,
                    "score_rms": index * 1.0e-3,
                    "logistic_shift_rms": index * 5.0e-5,
                "reference_fraction_displacement_rms": 1.0e-3,
                "control_fraction_displacement_rms": index * 2.0e-4,
                "control_reference_displacement_ratio": index * 0.2,
                "target_oracle_unreachable_boundary_count": 0,
            }
            for index in range(3)
        ]
    )
    row_keys = tuple(record["row_order"])
    expected_rows: list[dict[str, Any]] = []
    for row_index, row_key in enumerate(row_keys):
        row_states = {name: value[row_index] for name, value in saved.items()}
        zero_states = {name: value[0] for name, value in saved.items()}
        stored_rows = record.get("row_results")
        if not isinstance(stored_rows, list) or len(stored_rows) != 3:
            raise ArtifactCompatibilityError("core row results changed")
        stored_images = stored_rows[row_index].get("images", {})
        if not isinstance(stored_images, Mapping):
            raise ArtifactCompatibilityError("core row image binding changed")
        _verify_rendered_state_records(
            run_dir,
            trajectory_key=row_key,
            states=row_states,
            records=stored_images,
        )
        expected_rows.append(
            {
                "row_key": row_key,
                "row_index": row_index,
                "backend": backend,
                "canonical_path_id": _recovery_path_ids(run_dir)["development"],
                "metrics_to_mixed_target": {
                    name: _metrics_dict(value, target)
                    for name, value in row_states.items()
                },
                "paired_zero_divergence": {
                    name: {
                        "squared_l2_from_zero": float(
                            np.dot(value - zero_states[name], value - zero_states[name])
                        ),
                        "l1_from_zero": float(np.sum(np.abs(value - zero_states[name]))),
                        "total_variation_from_zero": float(
                            0.5 * np.sum(np.abs(value - zero_states[name]))
                        ),
                        "centered_correlation_with_zero": _centered_correlation(
                            value, zero_states[name]
                        ),
                    }
                    for name, value in row_states.items()
                },
                "images": dict(stored_images),
                "mechanism_diagnostics": mechanism[row_index],
            }
        )
    if record.get("row_results") != expected_rows:
        raise ArtifactCompatibilityError("core derived row evidence changed")
    from mnist.d0_jacobi_rb_tangent_rollout import (
        fixed_rendering_scale,
        render_background_demixed,
        render_source_image,
    )

    source, mixed_target = _source_arrays(run_dir)
    scale = fixed_rendering_scale(source, mixed_target, LAMBDA_MIX)
    contact_path = run_dir / str(record.get("contact_sheet", ""))
    contact_cells: list[tuple[str, np.ndarray]] = [
        ("source", render_source_image(source, scale)),
        (
            "anchor",
            render_background_demixed(
                _recovery_anchors(run_dir)[SHORT_ANCHOR], scale
            ),
        ),
    ]
    for row_index, row_key in enumerate(row_keys):
        for name in ("progress_25", "progress_50", "progress_75", "final"):
            contact_cells.append(
                (
                    f"{row_key}-{name}",
                    render_background_demixed(saved[name][row_index], scale),
                )
            )
    if (
        not contact_path.is_file()
        or file_fingerprint(contact_path) != _contact_sheet_sha256(contact_cells)
    ):
        raise ArtifactCompatibilityError("core contact sheet changed")
    audit = None
    if backend == "candidate":
        audit = _verify_recovery_candidate_audit(
            run_dir,
            audit_path=run_dir / "exact_candidate_audit.json",
            exact_root=run_dir
            / "objective_attempts/exact/fused_families/development-core/short",
            candidate_root=run_dir
            / "objective_attempts/candidate/fused_families/development-core/short",
            row_keys=record["row_order"],
        )
        if decision.get("exact_candidate_audit_sha256") != audit["semantic_sha256"]:
            raise ArtifactCompatibilityError("backend/audit binding changed")
    elif decision.get("exact_candidate_audit_sha256") is not None:
        raise ArtifactCompatibilityError("exact backend acquired candidate audit authority")
    learned_separation = float(np.dot(final[1] - final[0], final[1] - final[0]))
    approximation_dominates = False
    if audit is not None:
        largest = max(
            float(row["squared_l2_state_discrepancy"])
            for row in audit["row_metrics"]
        )
        approximation_dominates = not (
            float(audit["paired_contrasts"]["learned_minus_zero"]["relative_error"])
            <= 0.25
            and learned_separation >= 4.0 * largest
        )
    expected_decision = (
        "approximation_dominates_observed_effect"
        if approximation_dominates
        else "learned_short_dynamic_signal"
        if risks[1] < risks[0]
        else "learned_short_control_dynamically_negligible"
        if math.isclose(risks[1], risks[0], rel_tol=0.0, abs_tol=1.0e-14)
        else "learned_short_rollout_direction_not_useful"
    )
    guard = None
    if audit is not None:
        largest = max(
            float(row["squared_l2_state_discrepancy"])
            for row in audit["row_metrics"]
        )
        relative = float(
            audit["paired_contrasts"]["learned_minus_zero"]["relative_error"]
        )
        guard = {
            "largest_first_shard_candidate_exact_squared_l2_discrepancy": largest,
            "endpoint_learned_zero_squared_l2_separation": learned_separation,
            "required_endpoint_multiple": 4.0,
            "learned_paired_contrast_relative_error": relative,
            "learned_contrast_guard_passed": int(relative <= 0.25),
            "endpoint_scale_guard_passed": int(learned_separation >= 4.0 * largest),
            "coarse_candidate_dynamic_claim_permitted": int(
                relative <= 0.25 and learned_separation >= 4.0 * largest
            ),
        }
    expected_action = {
        "approximation_dominates_observed_effect": (
            "retain the objective images; audit more exact shards or improve the "
            "exploration backend before making a learned-utility claim"
        ),
        "learned_short_dynamic_signal": (
            "run a fresh-path evaluation; after a positive one-image result, use a "
            "small multi-image M=2 exploration with exact audit on a fixed subset"
        ),
        "learned_short_control_dynamically_negligible": (
            "measure calibration/amplitude and compare a rollout-trained or global "
            "alternative; do not add another exactness rung"
        ),
        "learned_short_rollout_direction_not_useful": (
            "inspect sign/order and compare a materially different controller or learner"
        ),
    }[expected_decision]
    source_separation = float(np.dot(final[2] - final[0], final[2] - final[0]))
    expected_record = _semantic(
        {
            "schema": record["schema"],
            "schema_version": 1,
            "research_mode": "exploratory",
            "objective_bearing_experiment": 1,
            "backend": backend,
            "reference_contract": (
                "certified_exact" if backend == "exact" else "candidate_approximate_v1"
            ),
            "row_order": list(row_keys),
            "raw_states": dict(raw),
            "row_results": expected_rows,
            "contact_sheet": contact_path.relative_to(run_dir).as_posix(),
            "zero_final_squared_l2": risks[0],
            "learned_final_squared_l2": risks[1],
            "source_informed_final_squared_l2": risks[2],
            "learned_minus_zero_risk_improvement": risks[0] - risks[1],
            "source_informed_minus_zero_risk_improvement": risks[0] - risks[2],
            "source_informed_composition_mismatch_diagnostic": int(risks[2] > risks[0]),
            "endpoint_learned_zero_squared_l2_separation": learned_separation,
            "endpoint_source_informed_zero_squared_l2_separation": source_separation,
            "exact_candidate_claim_guard": guard,
            "decision": expected_decision,
            "recommended_next_action": expected_action,
            "source_informed_row_is_diagnostic_not_gate": 1,
            "passed_numerical_integrity": 1,
            "completed_128_step_three_row_family": 1,
        }
    )
    if record != expected_record:
        raise ArtifactCompatibilityError("core objective metrics changed")
    return record


def _recovery_test_core(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    anchors = _recovery_anchors(run_dir)
    exact_saved = _recovery_test_family_states(anchors[SHORT_ANCHOR], backend="exact")
    exact_result = type(
        "TestRecoveryResult",
        (),
        {
            "saved_states": exact_saved,
            "final_state": exact_saved["final"],
            "transition_count": 263_424,
            "elapsed_seconds": 2.0,
            "per_row_diagnostics": tuple(
                {
                    "reference_fraction_displacement_rms": 1.0e-3,
                    "control_fraction_displacement_rms": float(index) * 2.0e-4,
                    "score_rms": float(index) * 1.0e-3,
                    "logistic_shift_rms": float(index) * 5.0e-5,
                }
                for index in range(3)
            ),
        },
    )()
    projection = _recovery_resource_projection(
        active_seconds=2.0,
        wasted_active_seconds=0.0,
        observed_shard_seconds=[2.0],
        remaining_shards=15,
        maximum_main_seconds=float(args.maximum_main_seconds),
        persisted_bytes=_directory_bytes(run_dir),
        projected_additional_bytes=1024,
        peak_memory_fraction=0.1,
        test_only=False,
    )
    # Tests may force the fallback without introducing a production option.
    if getattr(args, "test_force_candidate", False):
        projection = dict(projection)
        projection["passed"] = 0
        projection["checks"] = {**projection["checks"], "main_time": False}
    selected = _candidate_backend_selected(projection, args.reference_backend)
    audit: dict[str, Any] | None = None
    selected_result = exact_result
    if selected == "candidate":
        candidate_saved = _recovery_test_family_states(
            anchors[SHORT_ANCHOR], backend="candidate"
        )
        selected_result = type(
            "TestRecoveryResult",
            (),
            {
                "saved_states": candidate_saved,
                "final_state": candidate_saved["final"],
                "transition_count": 4_214_784,
                "elapsed_seconds": 3.0,
                "per_row_diagnostics": exact_result.per_row_diagnostics,
            },
        )()
        audit = _exact_candidate_audit_record(
            row_keys=(
                "development-core-zero",
                "development-core-learned-1",
                "development-core-source-informed",
            ),
            exact_state=exact_saved["final"],
            candidate_state=candidate_saved["final"],
            exact_reference_rms=[1.0e-3] * 3,
            candidate_reference_rms=[1.0001e-3] * 3,
            exact_controller_rms=[0.0, 2.0e-4, 4.0e-4],
            candidate_controller_rms=[0.0, 2.0001e-4, 4.0001e-4],
        )
        atomic_write_json(run_dir / "exact_candidate_audit.json", audit)
    backend_decision = _semantic(
        {
            "schema": RECOVERY_TEST_RUN_SCHEMA + "-backend-decision",
            "schema_version": 1,
            "requested": args.reference_backend,
            "selected": selected,
            "exact_audit_shard_committed": 1,
            "exact_projection": projection,
            "exact_resource_failure_terminal": 0,
            "candidate_restarted_from_original_anchor": int(selected == "candidate"),
            "mixed_exact_candidate_scientific_prefix": 0,
            "exact_candidate_audit_sha256": (
                audit["semantic_sha256"] if audit is not None else None
            ),
        }
    )
    atomic_write_json(run_dir / "backend_decision.json", backend_decision)
    return _commit_recovery_objective_artifacts(
        run_dir,
        anchor=anchors[SHORT_ANCHOR],
        result=selected_result,
        backend=selected,
        audit=audit,
    )


def _recovery_production_core(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    anchors = _recovery_anchors(run_dir)
    sequence = _reverse_sequence(SHORT_ANCHOR)
    audit_sequence = sequence[: 7 * int(args.exact_audit_outer_steps)]
    existing_exact_root = (
        run_dir
        / "objective_attempts/exact/fused_families/development-core/short"
    )
    existing_exact_count = (
        len(_recovery_shard_records(existing_exact_root))
        if existing_exact_root.is_dir()
        else 0
    )
    backend_decision_path = run_dir / "backend_decision.json"
    backend_opening_path = run_dir / "backend_selection_opening.json"
    backend_switch_path = run_dir / "backend_switch_event.json"
    prior_backend_decision = (
        _load_semantic(backend_decision_path, "backend decision")
        if backend_decision_path.is_file()
        else {}
    )
    prior_selected = str(prior_backend_decision.get("selected", "not_attempted"))
    exact_degraded = int(
        prior_backend_decision.get("exact_degraded_at_verified_boundary", 0)
    )
    degraded_projection = prior_backend_decision.get("exact_degraded_projection")
    if prior_selected not in {"not_attempted", "exact", "candidate"}:
        raise ArtifactCompatibilityError("resume backend selection changed")
    # Read and verify a committed exact prefix before backend selection.  In
    # particular, a durable candidate selection may retain more than the
    # one-shard audit after an exact runtime degradation; those exact shards
    # remain audit evidence and are never continued or favorably retimed.
    if existing_exact_count:
        rows = (
            ("development-core-zero", "zero", "short", None),
            (
                "development-core-learned-1",
                "learned",
                "short",
                float(args.core_learned_gain),
            ),
            ("development-core-source-informed", "oracle", "short", None),
        )
        path_ids = _recovery_path_ids(run_dir)
        specs, _bank, binding = _fused_controller_family(
            run_dir,
            args,
            rows=rows,
            canonical_path_id=path_ids["development"],
        )
        initial = np.repeat(
            _state_row(anchors[SHORT_ANCHOR])[None, :], len(specs), axis=0
        )
        _verify_fused_family_prefix(
            existing_exact_root,
            initial_state=initial,
            sequence=sequence,
            row_specs=specs,
            controller_binding=binding,
            rng_binding={
                "root_seed": REVERSE_ROOT_SEED,
                "stream_role": "frequency1-objective-first-development-v1",
                "canonical_path_id": path_ids["development"],
                "variant_in_rng_key": 0,
                "backend_and_attempt_in_rng_key": 0,
            },
            family_name="development-core",
            segment_name="short",
            reference_contract="certified_exact",
        )
        exact_root = existing_exact_root
        audit_path = run_dir / "exact_audit_attempt_development-core-short.json"
        if audit_path.is_file():
            exact_summary = _load_semantic(audit_path, "exact audit attempt")
            if (
                exact_summary.get("backend") != "exact"
                or int(exact_summary.get("passed", 0)) != 1
            ):
                raise ArtifactCompatibilityError("exact audit attempt changed")
        else:
            first = _recovery_shard_records(exact_root)[0]
            exact_summary = _semantic(
                {
                    "schema": RECOVERY_RUN_SCHEMA + "-exact-audit-attempt",
                    "schema_version": 1,
                    "backend": "exact",
                    "result": {"row_table": first["row_table"]},
                    "committed_shard_semantic_sha256": first["semantic_sha256"],
                    "recovered_from_verified_committed_shard": 1,
                    "separate_backend_selection_attempt": 1,
                    "scientific_trajectory_complete": 0,
                    "passed": 1,
                }
            )
            atomic_write_json(audit_path, exact_summary)
    else:
        _exact_result, exact_summary, exact_root = _run_recovery_core_backend(
            run_dir,
            args,
            anchor=anchors[SHORT_ANCHOR],
            backend="exact",
            sequence=audit_sequence,
            exact_audit_only=True,
        )
    exact_records = _recovery_shard_records(exact_root)
    if prior_selected in {"exact", "candidate"}:
        expected_candidate = int(prior_selected == "candidate")
        opening = _load_semantic(
            backend_opening_path, "backend selection opening"
        )
        stored_projection = opening.get("exact_projection")
        stored_degraded = prior_backend_decision.get("exact_degraded_projection")
        try:
            opening_selected = _verify_recovery_backend_opening(
                run_dir,
                args,
                opening=opening,
                exact_root=exact_root,
                total_shards=len(sequence) // 56,
                expected_schema=RECOVERY_RUN_SCHEMA + "-backend-selection-opening",
                expected_audit_sha256=exact_summary["semantic_sha256"],
            )
        except RolloutCLIError as exc:
            raise ArtifactCompatibilityError(
                "resume backend projection authority changed"
            ) from exc
        if exact_degraded:
            switch = _load_semantic(backend_switch_path, "backend switch event")
            _verify_recovery_switch_event(
                run_dir,
                args,
                event=switch,
                exact_root=exact_root,
                total_shards=len(sequence) // 56,
                exact_audit_sha256=exact_summary["semantic_sha256"],
                selection_opening_sha256=opening["semantic_sha256"],
                expected_schema=RECOVERY_RUN_SCHEMA + "-backend-switch-event",
                restart_field="restart_from_original_anchor",
            )
            selection_authorized = (
                prior_selected == "candidate"
                and opening_selected == "exact"
                and _is_semantic_record(stored_degraded)
                and int(stored_degraded.get("passed", 1)) == 0
                and switch.get("selection_opening_sha256")
                == opening.get("semantic_sha256")
                and switch.get("failed_exact_projection") == stored_degraded
                and switch.get("selected") == "candidate"
                and prior_backend_decision.get("switch_event_sha256")
                == switch.get("semantic_sha256")
            )
        else:
            selection_authorized = (
                stored_degraded is None
                and not backend_switch_path.exists()
                and opening_selected == prior_selected
            )
        if (
            prior_backend_decision.get("selection_opening_sha256")
            != opening.get("semantic_sha256")
            or prior_backend_decision.get("exact_projection") != stored_projection
            or
            prior_backend_decision.get("schema")
            != RECOVERY_RUN_SCHEMA + "-backend-decision"
            or prior_backend_decision.get("requested") != args.reference_backend
            or int(prior_backend_decision.get("exact_audit_shard_committed", 0))
            != 1
            or int(prior_backend_decision.get("exact_audit_outer_steps", -1))
            != int(args.exact_audit_outer_steps)
            or prior_backend_decision.get("exact_audit_semantic_sha256")
            != exact_summary.get("semantic_sha256")
            or int(
                prior_backend_decision.get(
                    "candidate_restarted_from_original_anchor", -1
                )
            )
            != expected_candidate
            or int(
                prior_backend_decision.get(
                    "mixed_exact_candidate_scientific_prefix", -1
                )
            )
            != 0
            or (
                args.reference_backend == "exact" and prior_selected != "exact"
            )
            or (
                args.reference_backend == "candidate"
                and prior_selected != "candidate"
            )
            or not selection_authorized
            or (
                prior_backend_decision.get("phase")
                not in {
                    "selected_before_scientific_continuation",
                    "selected_family_complete",
                }
                and not (
                    exact_degraded
                    and prior_backend_decision.get("phase")
                    == "exact_degraded_then_candidate_selected"
                )
            )
            or int(prior_backend_decision.get("selected_family_complete", -1))
            not in {0, 1}
        ):
            raise ArtifactCompatibilityError("resume backend decision binding changed")
    exact_elapsed = [float(record.get("elapsed_seconds", 0.0)) for record in exact_records]
    total_core_shards = len(sequence) // 56
    remaining_exact_shards = max(0, total_core_shards - len(exact_records))
    exact_complete_seconds = _complete_reverse_shard_seconds(
        run_dir,
        exact_elapsed,
        family_name="development-core",
        segment_name="short",
        backend="exact",
        row_count=3,
    )
    projected_exact_bytes = remaining_exact_shards * _recovery_max_committed_shard_bytes(
        exact_root, exact_records
    )
    projection_ledger = _write_recovery_ledger(
        run_dir,
        args,
        backend="exact",
        remaining_shards=remaining_exact_shards,
        observed_shard_seconds=exact_complete_seconds,
        projected_additional_bytes=projected_exact_bytes,
    )
    if backend_opening_path.is_file():
        opening = _load_semantic(backend_opening_path, "backend selection opening")
        selected = _verify_recovery_backend_opening(
            run_dir,
            args,
            opening=opening,
            exact_root=exact_root,
            total_shards=total_core_shards,
            expected_schema=RECOVERY_RUN_SCHEMA + "-backend-selection-opening",
            expected_audit_sha256=exact_summary["semantic_sha256"],
        )
        if prior_selected in {"exact", "candidate"}:
            selected = prior_selected
    else:
        opening_projection, resource_snapshot = _recovery_selection_projection(
            run_dir,
            args,
            observed_shard_seconds=exact_complete_seconds,
            remaining_shards=remaining_exact_shards,
            projected_additional_bytes=projected_exact_bytes,
        )
        selected = _candidate_backend_selected(
            opening_projection, args.reference_backend
        )
        opening = _semantic(
            {
                "schema": RECOVERY_RUN_SCHEMA + "-backend-selection-opening",
                "schema_version": 1,
                "requested": args.reference_backend,
                "selected": selected,
                "exact_audit_semantic_sha256": exact_summary["semantic_sha256"],
                "exact_projection": opening_projection,
                "resource_ledger_semantic_sha256": projection_ledger[
                    "semantic_sha256"
                ],
                "projection_inputs": _recovery_projection_input_commitments(
                    run_dir,
                    exact_root=exact_root,
                    remaining_shards=remaining_exact_shards,
                    projection=opening_projection,
                    resource_snapshot=resource_snapshot,
                    test_only=bool(args.test_only),
                ),
            }
        )
        atomic_write_json(backend_opening_path, opening)
    # Backend selection is always durable before any selected full-family
    # continuation.  A candidate failure therefore cannot erase the reason
    # exact was not selected.
    replace_placeholder = False
    if backend_decision_path.is_file():
        prior_selected = str(prior_backend_decision.get("selected"))
        if (
            prior_selected == "not_attempted"
            and prior_backend_decision.get("phase")
            == "not_attempted_before_analytic_control"
            and int(prior_backend_decision.get("exact_audit_shard_committed", -1)) == 0
        ):
            replace_placeholder = True
        elif prior_selected in {"exact", "candidate"}:
            selected = prior_selected
        else:
            raise ArtifactCompatibilityError("resume backend selection changed")
    if replace_placeholder or not backend_decision_path.is_file():
        atomic_write_json(
            backend_decision_path,
            _semantic(
                {
                    "schema": RECOVERY_RUN_SCHEMA + "-backend-decision",
                    "schema_version": 1,
                    "phase": "selected_before_scientific_continuation",
                    "requested": args.reference_backend,
                    "selected": selected,
                    "exact_audit_shard_committed": 1,
                    "exact_audit_outer_steps": int(args.exact_audit_outer_steps),
                    "exact_audit_semantic_sha256": exact_summary["semantic_sha256"],
                    "exact_projection": opening["exact_projection"],
                    "exact_resource_failure_terminal": 0,
                    "candidate_restarted_from_original_anchor": int(selected == "candidate"),
                    "mixed_exact_candidate_scientific_prefix": 0,
                    "same_process_selection_and_continuation": 1,
                    "selection_opening_sha256": opening["semantic_sha256"],
                    "switch_event_sha256": None,
                    "selected_family_complete": 0,
                }
            ),
        )
    audit: dict[str, Any] | None = None
    if selected == "exact":
        try:
            result, _summary, _root = _run_recovery_core_backend(
                run_dir,
                args,
                anchor=anchors[SHORT_ANCHOR],
                backend="exact",
                sequence=sequence,
                exact_audit_only=False,
            )
        except RolloutCLIError as exc:
            if (
                exc.failure_code != "exact_backend_switch_required"
                or args.reference_backend != "auto"
            ):
                raise
            selected = "candidate"
            exact_degraded = 1
            switch_event = _recovery_switch_event(
                run_dir,
                args,
                exact_root=exact_root,
                total_shards=len(sequence) // 56,
                exact_audit_sha256=exact_summary["semantic_sha256"],
                selection_opening_sha256=opening["semantic_sha256"],
                schema=RECOVERY_RUN_SCHEMA + "-backend-switch-event",
                restart_field="restart_from_original_anchor",
            )
            degraded_projection = switch_event["failed_exact_projection"]
            atomic_write_json(backend_switch_path, switch_event)
            atomic_write_json(
                backend_decision_path,
                _semantic(
                    {
                        "schema": RECOVERY_RUN_SCHEMA + "-backend-decision",
                        "schema_version": 1,
                        "phase": "exact_degraded_then_candidate_selected",
                        "requested": args.reference_backend,
                        "selected": "candidate",
                        "exact_audit_shard_committed": 1,
                        "exact_audit_outer_steps": int(args.exact_audit_outer_steps),
                        "exact_audit_semantic_sha256": exact_summary[
                            "semantic_sha256"
                        ],
                        "exact_projection": opening["exact_projection"],
                        "exact_degraded_projection": degraded_projection,
                        "exact_degraded_at_verified_boundary": 1,
                        "exact_resource_failure_terminal": 0,
                        "candidate_restarted_from_original_anchor": 1,
                        "mixed_exact_candidate_scientific_prefix": 0,
                        "same_process_selection_and_continuation": 1,
                        "selection_opening_sha256": opening["semantic_sha256"],
                        "switch_event_sha256": switch_event["semantic_sha256"],
                        "selected_family_complete": 0,
                    }
                ),
            )
    if selected == "candidate":
        result, _summary, _root = _run_recovery_core_backend(
            run_dir,
            args,
            anchor=anchors[SHORT_ANCHOR],
            backend="candidate",
            sequence=sequence,
            exact_audit_only=False,
        )
        candidate_first = _load_npz(
            run_dir
            / "objective_attempts/candidate/fused_families/development-core/short/shard-0000.npz"
        )["state"]
        exact_first = _load_npz(
            run_dir
            / "objective_attempts/exact/fused_families/development-core/short/shard-0000.npz"
        )["state"]
        audit = _exact_candidate_audit_record(
            row_keys=tuple(row["row_key"] for row in exact_summary["result"]["row_table"]),
            exact_state=exact_first,
            candidate_state=candidate_first,
            exact_reference_rms=_recovery_shard_row_rms(
                _recovery_shard_records(exact_root)[0],
                "reference_fraction_displacement",
            ),
            candidate_reference_rms=_recovery_shard_row_rms(
                _recovery_shard_records(_root)[0],
                "reference_fraction_displacement",
            ),
            exact_controller_rms=_recovery_shard_row_rms(
                _recovery_shard_records(exact_root)[0],
                "control_fraction_displacement",
            ),
            candidate_controller_rms=_recovery_shard_row_rms(
                _recovery_shard_records(_root)[0],
                "control_fraction_displacement",
            ),
        )
        atomic_write_json(run_dir / "exact_candidate_audit.json", audit)
    backend_decision = _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-backend-decision",
            "schema_version": 1,
            "requested": args.reference_backend,
            "selected": selected,
            "exact_audit_shard_committed": 1,
            "exact_audit_outer_steps": int(args.exact_audit_outer_steps),
            "exact_audit_semantic_sha256": exact_summary["semantic_sha256"],
            "exact_projection": opening["exact_projection"],
            "exact_degraded_projection": degraded_projection,
            "exact_degraded_at_verified_boundary": exact_degraded,
            "exact_resource_failure_terminal": 0,
            "candidate_restarted_from_original_anchor": int(selected == "candidate"),
            "mixed_exact_candidate_scientific_prefix": 0,
            "same_process_selection_and_continuation": 1,
            "selection_opening_sha256": opening["semantic_sha256"],
            "switch_event_sha256": (
                _load_semantic(backend_switch_path, "backend switch event")[
                    "semantic_sha256"
                ]
                if backend_switch_path.is_file()
                else None
            ),
            "phase": "selected_family_complete",
            "selected_family_complete": 1,
            "exact_candidate_audit_sha256": (
                audit.get("semantic_sha256") if audit else None
            ),
        }
    )
    atomic_write_json(backend_decision_path, backend_decision)
    return _commit_recovery_objective_artifacts(
        run_dir,
        anchor=anchors[SHORT_ANCHOR],
        result=result,
        backend=selected,
        audit=audit,
    )


def _recovery_optional_projection(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    backend: str,
    additional_shards: int,
    headroom_fraction: float = 0.0,
) -> dict[str, Any]:
    roots = list(
        (run_dir / "objective_attempts" / backend / "fused_families").glob("*/*")
    )
    observed = [
        float(record.get("elapsed_seconds", 0.0))
        for root in roots
        if root.is_dir()
        for record in _recovery_shard_records(root)
    ]
    usage = _recovery_observed_resource(run_dir)
    overhead_total = math.fsum(
        float(_load_semantic(path, "recovery attempt timing").get("commit_verification_overhead_seconds", 0.0))
        for path in run_dir.glob("metrics/recovery_attempt_timing_*.json")
    )
    overhead_per_shard = overhead_total / max(1, usage["reverse_shard_count"])
    observed = [value + overhead_per_shard for value in observed]
    projection = _recovery_resource_projection(
        active_seconds=usage["active_seconds"],
        wasted_active_seconds=usage["wasted_active_seconds"],
        observed_shard_seconds=observed,
        remaining_shards=int(additional_shards),
        maximum_main_seconds=float(args.maximum_main_seconds)
        * (1.0 - float(headroom_fraction)),
        persisted_bytes=usage["persisted_bytes"],
        projected_additional_bytes=max(1, additional_shards) * 64 * 1024,
        peak_memory_fraction=usage["peak_memory_fraction"],
        test_only=bool(args.test_only),
    )
    return projection


def _recovery_mixed_optional_projection(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    selected_backend: str,
    continuation_shards: int,
    headroom_fraction: float = 0.0,
) -> dict[str, Any]:
    """Project one exact audit plus a selected-backend continuation."""

    if selected_backend == "exact":
        return _recovery_optional_projection(
            run_dir,
            args,
            backend="exact",
            additional_shards=continuation_shards,
            headroom_fraction=headroom_fraction,
        )
    usage = _recovery_observed_resource(run_dir)
    overhead_total = math.fsum(
        float(
            _load_semantic(path, "recovery attempt timing").get(
                "commit_verification_overhead_seconds", 0.0
            )
        )
        for path in run_dir.glob("metrics/recovery_attempt_timing_*.json")
    )
    overhead_per_shard = overhead_total / max(1, usage["reverse_shard_count"])
    exact_times = [
        float(row.get("elapsed_seconds", 0.0))
        for root in (run_dir / "objective_attempts/exact/fused_families").glob("*/*")
        if root.is_dir()
        for row in _recovery_shard_records(root)
    ]
    candidate_times = [
        float(row.get("elapsed_seconds", 0.0))
        for root in (run_dir / "objective_attempts/candidate/fused_families").glob("*/*")
        if root.is_dir()
        for row in _recovery_shard_records(root)
    ]
    exact_observed = max(
        (value + overhead_per_shard for value in exact_times),
        default=max(candidate_times, default=0.0) + overhead_per_shard,
    )
    candidate_observed = max(
        (value + overhead_per_shard for value in candidate_times),
        default=exact_observed,
    )
    allowed = float(args.maximum_main_seconds) * (1.0 - float(headroom_fraction))
    projected_seconds = (
        usage["active_seconds"]
        + usage["wasted_active_seconds"]
        + RECOVERY_PROJECTION_FACTOR
        * (exact_observed + candidate_observed * int(continuation_shards))
        + RECOVERY_REPORT_RESERVE_SECONDS
    )
    projected_bytes = usage["persisted_bytes"] + (1 + int(continuation_shards)) * 64 * 1024
    checks = {
        "main_time": projected_seconds <= allowed,
        "persisted_storage": projected_bytes <= MAXIMUM_PERSISTED_BYTES,
        "peak_cuda_memory": usage["peak_memory_fraction"] <= MAXIMUM_MEMORY_FRACTION,
    }
    record = _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-mixed-optional-resource-projection",
            "schema_version": 1,
            "selected_backend": selected_backend,
            "exact_audit_shards": 1,
            "continuation_shards": int(continuation_shards),
            "exact_observed_seconds": exact_observed,
            "candidate_observed_seconds": candidate_observed,
            "projected_total_seconds": projected_seconds,
            "maximum_main_seconds": allowed,
            "projected_persisted_bytes": projected_bytes,
            "peak_memory_fraction": usage["peak_memory_fraction"],
            "checks": checks,
            "passed": int(all(checks.values())),
        }
    )
    return record


def _run_recovery_optional_backend(
    run_dir: Path,
    args: argparse.Namespace,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any], Path] | None:
    """Run optional objective work without invalidating a completed core.

    Numerical/contract failures still propagate.  Only a pre-shard hard-budget
    stop is converted to an explicit deferred optional result.
    """

    try:
        return _run_recovery_core_backend(
            run_dir, args, mandatory_core=False, **kwargs
        )
    except RolloutCLIError as exc:
        if exc.failure_code != "optional_objective_resource_deferred":
            raise
        return None


def _verify_recovery_partial_family(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    backend: str,
    family_name: str,
    segment_name: str,
    rows: Sequence[tuple[str, str, str, float | None]],
    canonical_role: str,
    stream_role: str,
    anchor: np.ndarray,
    sequence: Sequence[tuple[int, int]],
) -> dict[str, Any] | None:
    root = run_dir / (
        f"objective_attempts/{backend}/fused_families/{family_name}/{segment_name}"
    )
    if not any(root.glob("shard-*.json")):
        return None
    path_ids = _recovery_path_ids(run_dir)
    specs, _bank, binding = _fused_controller_family(
        run_dir,
        args,
        rows=rows,
        canonical_path_id=path_ids[canonical_role],
    )
    initial = np.repeat(_state_row(anchor)[None, :], len(specs), axis=0)
    return _verify_fused_family_prefix(
        root,
        initial_state=initial,
        sequence=sequence,
        row_specs=specs,
        controller_binding=binding,
        rng_binding={
            "root_seed": REVERSE_ROOT_SEED,
            "stream_role": stream_role,
            "canonical_path_id": path_ids[canonical_role],
            "variant_in_rng_key": 0,
            "backend_and_attempt_in_rng_key": 0,
        },
        family_name=family_name,
        segment_name=segment_name,
        reference_contract=RECOVERY_REFERENCE_CONTRACTS[backend],
    )


RECOVERY_GAIN_VALUES = (0.5, 2.0, 4.0)


def _recovery_gain_objective_artifact(
    run_dir: Path,
    *,
    result: Any,
    backend: str,
    family_summary: Mapping[str, Any],
    audit: Mapping[str, Any] | None,
    verify_only: bool = False,
) -> dict[str, Any]:
    """Persist the same task-level evidence for every optional gain row."""

    required = ("start", "progress_25", "progress_50", "progress_75", "final")
    saved = {
        str(name): np.ascontiguousarray(value, dtype=np.float64)
        for name, value in _field(result, "saved_states", {}).items()
    }
    if any(name not in saved or saved[name].shape != (3, 784) for name in required):
        raise RolloutCLIError(
            "gain expansion omitted required three-row saved states",
            failure_domain="implementation_contract",
            failure_code="rollout_implementation_contract_invalid",
        )
    root = run_dir / "objective/development-gain-expansion"
    path = root / "gain_objective_family.json"
    existing = (
        _load_semantic(path, "gain objective family")
        if verify_only and path.is_file()
        else None
    )
    if verify_only and existing is None:
        raise ArtifactCompatibilityError("gain objective family is missing")
    if not verify_only:
        root.mkdir(parents=True, exist_ok=True)
    raw_path = root / "selected_states.npz"
    if verify_only:
        raw_binding = existing.get("raw_states")
        if (
            not isinstance(raw_binding, Mapping)
            or raw_binding.get("path") != raw_path.relative_to(run_dir).as_posix()
            or not raw_path.is_file()
            or raw_binding.get("sha256") != file_fingerprint(raw_path)
            or int(raw_binding.get("size", -1)) != int(raw_path.stat().st_size)
        ):
            raise ArtifactCompatibilityError("gain raw-state artifact changed")
        raw_saved = _load_npz(raw_path)
        if set(raw_saved) != set(saved) or any(
            _array_sha256(raw_saved[name]) != _array_sha256(saved[name])
            for name in saved
        ):
            raise ArtifactCompatibilityError("gain raw/family states disagree")
        raw = dict(raw_binding)
    else:
        raw = _atomic_npz(raw_path, **saved)
        raw["path"] = raw_path.relative_to(run_dir).as_posix()
    core = _load_semantic(run_dir / "core_objective.json", "core objective")
    core_raw_binding = core.get("raw_states")
    if not isinstance(core_raw_binding, Mapping):
        raise ArtifactCompatibilityError("core gain-one raw binding changed")
    core_raw_path = run_dir / str(core_raw_binding.get("path", ""))
    if (
        not core_raw_path.is_file()
        or core_raw_binding.get("sha256") != file_fingerprint(core_raw_path)
    ):
        raise ArtifactCompatibilityError("core gain-one raw artifact changed")
    core_saved = _load_npz(core_raw_path)
    if any(name not in core_saved for name in required):
        raise ArtifactCompatibilityError("core gain-one saved states changed")
    _source, target = _source_arrays(run_dir)
    mechanism = _recovery_mechanism_rows(result)
    stored_rows = existing.get("row_results") if existing is not None else None
    if verify_only and (not isinstance(stored_rows, list) or len(stored_rows) != 3):
        raise ArtifactCompatibilityError("gain objective rows changed")
    row_results: list[dict[str, Any]] = []
    for row_index, gain in enumerate(RECOVERY_GAIN_VALUES):
        row_key = f"development-gain-{_fused_gain_token(gain)}"
        states = {name: saved[name][row_index] for name in required}
        gain_one = {name: np.asarray(core_saved[name])[1] for name in required}
        if verify_only:
            images = stored_rows[row_index].get("images", {})
            if not isinstance(images, Mapping):
                raise ArtifactCompatibilityError("gain objective image binding changed")
            _verify_rendered_state_records(
                run_dir,
                trajectory_key=row_key,
                states=states,
                records=images,
            )
            images = dict(images)
        else:
            images = _render_states(run_dir, trajectory_key=row_key, states=states)
        row_results.append(
            {
                "row_key": row_key,
                "gain": gain,
                "metrics_to_mixed_target": {
                    name: _metrics_dict(value, target)
                    for name, value in states.items()
                },
                "paired_gain_one_divergence": {
                    name: {
                        "squared_l2": float(
                            np.dot(value - gain_one[name], value - gain_one[name])
                        ),
                        "l1": float(np.sum(np.abs(value - gain_one[name]))),
                        "total_variation": float(
                            0.5 * np.sum(np.abs(value - gain_one[name]))
                        ),
                        "centered_correlation": _centered_correlation(
                            value, gain_one[name]
                        ),
                    }
                    for name, value in states.items()
                },
                "mechanism_diagnostics": mechanism[row_index],
                "images": images,
            }
        )
    from mnist.d0_jacobi_rb_tangent_rollout import (
        fixed_rendering_scale,
        render_background_demixed,
    )

    source, mixed_target = _source_arrays(run_dir)
    scale = fixed_rendering_scale(source, mixed_target, LAMBDA_MIX)
    contact = run_dir / "images/development_gain_expansion_contact_sheet.png"
    contact_cells = [
            (
                f"development-gain-{_fused_gain_token(gain)}-{name}",
                render_background_demixed(saved[name][row_index], scale),
            )
            for row_index, gain in enumerate(RECOVERY_GAIN_VALUES)
            for name in required
        ]
    if verify_only:
        if (
            not contact.is_file()
            or file_fingerprint(contact) != _contact_sheet_sha256(contact_cells)
        ):
            raise ArtifactCompatibilityError("gain objective contact sheet changed")
    else:
        _contact_sheet(contact, contact_cells)
    candidates = [
        {
            "gain": gain,
            "final_squared_l2": float(
                row_results[index]["metrics_to_mixed_target"]["final"][
                    "squared_l2_error"
                ]
            ),
        }
        for index, gain in enumerate(RECOVERY_GAIN_VALUES)
    ]
    candidates.append(
        {
            "gain": 1.0,
            "final_squared_l2": float(core["learned_final_squared_l2"]),
        }
    )
    winner = min(candidates, key=lambda row: (row["final_squared_l2"], row["gain"]))
    record = _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-gain-objective-family",
            "schema_version": 1,
            "backend": backend,
            "family_summary_sha256": family_summary["semantic_sha256"],
            "core_objective_sha256": core["semantic_sha256"],
            "raw_states": raw,
            "row_results": row_results,
            "mechanism_diagnostics": mechanism,
            "contact_sheet": contact.relative_to(run_dir).as_posix(),
            "exact_candidate_audit_sha256": (
                audit.get("semantic_sha256") if audit else None
            ),
            "candidates": sorted(candidates, key=lambda row: row["gain"]),
            "selected_gain": winner["gain"],
            "selection_mode": "minimum_final_squared_l2_then_smaller_gain",
            "passed_numerical_integrity": 1,
        }
    )
    if existing is not None:
        if existing != record:
            raise ArtifactCompatibilityError("gain objective family changed")
        return existing
    atomic_write_json(path, record)
    return record


def _recovery_gain_expansion(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    anchor: np.ndarray,
    backend: str,
) -> dict[str, Any]:
    path = run_dir / "gain_expansion.json"
    if path.is_file():
        existing = _load_semantic(path, "gain expansion")
        if (
            existing.get("schema") != RECOVERY_RUN_SCHEMA + "-gain-expansion"
            or float(existing.get("selected_gain", float("nan")))
            not in {0.5, 1.0, 2.0, 4.0}
        ):
            raise ArtifactCompatibilityError("gain expansion binding changed")
        if not int(existing.get("performed", 0)):
            gain_roots = tuple(
                run_dir.glob(
                    "objective_attempts/*/fused_families/"
                    "development-gain-expansion/short/shard-*.json"
                )
            )
            forbidden_outputs = (
                run_dir / "objective/development-gain-expansion",
                run_dir / "exact_candidate_audit_gain_expansion.json",
            )
            projection = existing.get("projection")
            allowed_reasons = {
                "test_fixture_core_only",
                "explicitly_disabled",
                "insufficient_20_percent_budget_headroom",
                "optional_exact_audit_resource_deferred",
                "optional_gain_family_resource_deferred",
            }
            honest_partial = (
                existing.get("reason")
                in {
                    "optional_exact_audit_resource_deferred",
                    "optional_gain_family_resource_deferred",
                }
                and int(existing.get("partial_optional_evidence_preserved", 0)) == 1
            )
            if honest_partial:
                gain_rows = tuple(
                    (
                        f"development-gain-{_fused_gain_token(gain)}",
                        "learned",
                        "short",
                        gain,
                    )
                    for gain in RECOVERY_GAIN_VALUES
                )
                verified_prefixes = [
                    value
                    for candidate_backend in ("exact", "candidate")
                    if (
                        value := _verify_recovery_partial_family(
                            run_dir,
                            args,
                            backend=candidate_backend,
                            family_name="development-gain-expansion",
                            segment_name="short",
                            rows=gain_rows,
                            canonical_role="development",
                            stream_role="frequency1-objective-first-development-v1",
                            anchor=anchor,
                            sequence=_reverse_sequence(SHORT_ANCHOR),
                        )
                    )
                    is not None
                ]
                if not verified_prefixes:
                    raise ArtifactCompatibilityError(
                        "partial gain evidence claim lacks a verified prefix"
                    )
            if (
                (gain_roots and not honest_partial)
                or any(item.exists() for item in forbidden_outputs)
                or existing.get("reason") not in allowed_reasons
                or float(existing.get("selected_gain", float("nan")))
                != float(args.core_learned_gain)
                or int(existing.get("core_artifact_already_committed", 0)) != 1
                or not isinstance(projection, Mapping)
                or projection.get("schema")
                not in {
                    RECOVERY_RUN_SCHEMA + "-mixed-optional-resource-projection",
                    RECOVERY_RUN_SCHEMA + "-resource-projection",
                }
                or (
                    projection.get("schema")
                    == RECOVERY_RUN_SCHEMA + "-mixed-optional-resource-projection"
                    and (
                        int(projection.get("exact_audit_shards", -1)) != 1
                        or int(projection.get("continuation_shards", -1)) != 16
                    )
                )
                or (
                    projection.get("schema")
                    == RECOVERY_RUN_SCHEMA + "-resource-projection"
                    and (
                        not args.test_only
                        or int(projection.get("remaining_shards", -1)) != 16
                    )
                )
            ):
                raise ArtifactCompatibilityError(
                    "unperformed gain expansion evidence changed"
                )
        if int(existing.get("performed", 0)):
            candidates = existing.get("candidates")
            if not isinstance(candidates, list) or len(candidates) != 4:
                raise ArtifactCompatibilityError("gain expansion candidates changed")
            normalized = sorted(
                (
                    float(row["gain"]),
                    float(row["final_squared_l2"]),
                )
                for row in candidates
                if isinstance(row, Mapping)
            )
            if (
                len(normalized) != 4
                or {gain for gain, _risk in normalized} != {0.5, 1.0, 2.0, 4.0}
                or any(not math.isfinite(risk) for _gain, risk in normalized)
                or float(existing["selected_gain"])
                != min(normalized, key=lambda row: (row[1], row[0]))[0]
                or existing.get("backend") != backend
            ):
                raise ArtifactCompatibilityError("gain expansion selection changed")
            summary_path = (
                run_dir
                / f"objective_attempts/{backend}/fused_families/development-gain-expansion/short/family_summary.json"
            )
            summary = _load_semantic(summary_path, "gain family summary")
            if existing.get("family_summary_sha256") != summary.get(
                "semantic_sha256"
            ):
                raise ArtifactCompatibilityError("gain family binding changed")
            saved_artifact = summary.get("saved_states_artifact", {})
            if not isinstance(saved_artifact, Mapping):
                raise ArtifactCompatibilityError("gain saved-state binding changed")
            verified_summary, _saved = _verify_recovery_completed_family_artifacts(
                run_dir,
                args,
                backend=backend,
                family_name="development-gain-expansion",
                segment_name="short",
                rows=tuple(
                    (
                        f"development-gain-{_fused_gain_token(gain)}",
                        "learned",
                        "short",
                        float(gain),
                    )
                    for gain in (0.5, 2.0, 4.0)
                ),
                canonical_role="development",
                stream_role="frequency1-objective-first-development-v1",
                anchor=anchor,
                sequence=_reverse_sequence(SHORT_ANCHOR),
            )
            if verified_summary != summary:
                raise ArtifactCompatibilityError("gain family summary changed")
            saved_path = run_dir / str(saved_artifact.get("path", ""))
            if (
                not saved_path.is_file()
                or saved_artifact.get("sha256") != file_fingerprint(saved_path)
                or int(saved_artifact.get("size", -1))
                != int(saved_path.stat().st_size)
                or int(summary.get("passed", 0)) != 1
            ):
                raise ArtifactCompatibilityError("gain saved-state binding changed")
            audit_path = run_dir / "exact_candidate_audit_gain_expansion.json"
            gain_audit = None
            if backend == "candidate":
                exact_root = run_dir / (
                    "objective_attempts/exact/fused_families/"
                    "development-gain-expansion/short"
                )
                candidate_root = run_dir / (
                    "objective_attempts/candidate/fused_families/"
                    "development-gain-expansion/short"
                )
                gain_audit = _exact_candidate_row_only_audit_record(
                    row_keys=tuple(
                        f"development-gain-{_fused_gain_token(gain)}"
                        for gain in RECOVERY_GAIN_VALUES
                    ),
                    exact_state=_load_npz(exact_root / "shard-0000.npz")["state"],
                    candidate_state=_load_npz(candidate_root / "shard-0000.npz")["state"],
                )
                if _load_semantic(audit_path, "gain exact candidate audit") != gain_audit:
                    raise ArtifactCompatibilityError("gain audit evidence changed")
            expected_audit = gain_audit.get("semantic_sha256") if gain_audit else None
            if existing.get("exact_candidate_audit_sha256") != expected_audit:
                raise ArtifactCompatibilityError("gain audit binding changed")
            # Reconstruct all selection and task-level evidence from the verified
            # family matrices.  Stored risks never select the FB200 gain by themselves.
            class _VerifiedGainResult:
                saved_states = _saved
                per_row_diagnostics = tuple(
                    verified_summary.get("result", {}).get("per_row_diagnostics", ())
                )

            objective = _recovery_gain_objective_artifact(
                run_dir,
                result=_VerifiedGainResult(),
                backend=backend,
                family_summary=verified_summary,
                audit=gain_audit,
                verify_only=True,
            )
            if (
                existing.get("gain_objective_family_sha256")
                != objective["semantic_sha256"]
                or existing.get("candidates") != objective["candidates"]
                or float(existing["selected_gain"])
                != float(objective["selected_gain"])
            ):
                raise ArtifactCompatibilityError("gain objective selection changed")
        return existing
    projection = _recovery_mixed_optional_projection(
        run_dir,
        args,
        selected_backend=backend,
        continuation_shards=16,
        headroom_fraction=0.20,
    )
    requested = str(args.gain_sweep)
    run_sweep = bool(not args.test_only and int(projection["passed"]) and requested != "off")
    if not run_sweep:
        record = _semantic(
            {
                "schema": RECOVERY_RUN_SCHEMA + "-gain-expansion",
                "schema_version": 1,
                "performed": 0,
                "reason": (
                    "test_fixture_core_only"
                    if args.test_only
                    else "explicitly_disabled"
                    if requested == "off"
                    else "insufficient_20_percent_budget_headroom"
                ),
                "projection": projection,
                "selected_gain": float(args.core_learned_gain),
                "selection_mode": "predeclared_core_gain_no_sweep",
                "core_artifact_already_committed": 1,
            }
        )
        atomic_write_json(path, record)
        return record
    rows = tuple(
        (
            f"development-gain-{_fused_gain_token(gain)}",
            "learned",
            "short",
            float(gain),
        )
        for gain in (0.5, 2.0, 4.0)
    )
    exact_audit: dict[str, Any] | None = None
    if backend == "candidate":
        exact_attempt = _run_recovery_optional_backend(
            run_dir,
            args,
            anchor=anchor,
            backend="exact",
            sequence=_reverse_sequence(SHORT_ANCHOR)[: 7 * int(args.exact_audit_outer_steps)],
            exact_audit_only=True,
            rows=rows,
            canonical_role="development",
            family_name="development-gain-expansion",
            segment_name="short",
            stream_role="frequency1-objective-first-development-v1",
        )
        if exact_attempt is None:
            record = _semantic(
                {
                    "schema": RECOVERY_RUN_SCHEMA + "-gain-expansion",
                    "schema_version": 1,
                    "performed": 0,
                    "reason": "optional_exact_audit_resource_deferred",
                    "projection": projection,
                    "selected_gain": float(args.core_learned_gain),
                    "partial_optional_evidence_preserved": 1,
                    "core_artifact_already_committed": 1,
                }
            )
            atomic_write_json(path, record)
            return record
        exact_result, exact_summary, exact_root = exact_attempt
    selected_attempt = _run_recovery_optional_backend(
        run_dir,
        args,
        anchor=anchor,
        backend=backend,
        sequence=_reverse_sequence(SHORT_ANCHOR),
        exact_audit_only=False,
        rows=rows,
        canonical_role="development",
        family_name="development-gain-expansion",
        segment_name="short",
        stream_role="frequency1-objective-first-development-v1",
    )
    if selected_attempt is None:
        record = _semantic(
            {
                "schema": RECOVERY_RUN_SCHEMA + "-gain-expansion",
                "schema_version": 1,
                "performed": 0,
                "reason": "optional_gain_family_resource_deferred",
                "projection": projection,
                "selected_gain": float(args.core_learned_gain),
                "partial_optional_evidence_preserved": 1,
                "core_artifact_already_committed": 1,
            }
        )
        atomic_write_json(path, record)
        return record
    result, summary, _root = selected_attempt
    if backend == "candidate":
        exact_audit = _exact_candidate_row_only_audit_record(
            row_keys=tuple(row[0] for row in rows),
            exact_state=_load_npz(exact_root / "shard-0000.npz")["state"],
            candidate_state=_load_npz(_root / "shard-0000.npz")["state"],
        )
        atomic_write_json(run_dir / "exact_candidate_audit_gain_expansion.json", exact_audit)
    objective = _recovery_gain_objective_artifact(
        run_dir,
        result=result,
        backend=backend,
        family_summary=summary,
        audit=exact_audit,
    )
    record = _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-gain-expansion",
            "schema_version": 1,
            "performed": 1,
            "projection": projection,
            "backend": backend,
            "family_summary_sha256": summary["semantic_sha256"],
            "gain_objective_family_sha256": objective["semantic_sha256"],
            "exact_candidate_audit_sha256": (
                exact_audit.get("semantic_sha256") if exact_audit else None
            ),
            "candidates": objective["candidates"],
            "selected_gain": objective["selected_gain"],
            "selection_mode": "minimum_final_squared_l2_then_smaller_gain",
            "core_artifact_committed_before_expansion": 1,
        }
    )
    atomic_write_json(path, record)
    return record


def _recovery_forward_evaluation(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    historical_anchors: Mapping[int, np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    plan_path = run_dir / "evaluation_anchor_plan.json"
    existing = _load_semantic(plan_path, "evaluation anchor plan") if plan_path.is_file() else None
    path_id = _recovery_path_ids(run_dir)["evaluation"]
    if existing is not None:
        if (
            int(existing.get("canonical_path_id", -1)) != path_id
            or int(existing.get("passed", 0)) != 1
            or existing.get("mode")
            not in {
                "historical_anchor_stochastic_evaluation",
                "fresh_forward_evaluation",
            }
        ):
            raise ArtifactCompatibilityError("evaluation anchor plan changed")
    committed_forward_paths = sorted(
        run_dir.glob(
            "objective/evaluation_forward/forward_shards/"
            "objective-evaluation-forward/shard-*.json"
        )
    )
    if committed_forward_paths:
        from mnist.d0_jacobi_rb_tangent_rollout import (
            aggregate_exact_forward_shards,
        )

        committed_forward = [
            _load_semantic(item, "evaluation forward committed prefix")
            for item in committed_forward_paths
        ]
        for json_path, row in zip(committed_forward_paths, committed_forward):
            state_path = json_path.with_suffix(".npz")
            if (
                not state_path.is_file()
                or row.get("state_file_sha256") != file_fingerprint(state_path)
                or int(row.get("state_file_size", -1))
                != int(state_path.stat().st_size)
            ):
                raise ArtifactCompatibilityError(
                    "evaluation forward committed state changed"
                )
        try:
            committed_forward_aggregate = aggregate_exact_forward_shards(
                committed_forward,
                expected_shard_count=len(committed_forward),
                expected_transition_count=sum(
                    int(row.get("transition_count", -1))
                    for row in committed_forward
                ),
                expected_path_ids=(path_id,),
            )
        except Exception as exc:
            raise ArtifactCompatibilityError(
                "evaluation forward committed prefix changed"
            ) from exc
        if (
            existing is not None
            and existing.get("mode") == "historical_anchor_stochastic_evaluation"
        ):
            if (
                int(existing.get("fresh_forward_attempted", 0)) != 1
                or int(existing.get("partial_fresh_forward_preserved", 0)) != 1
                or int(existing.get("fresh_forward_completed_steps", -1))
                != 8 * len(committed_forward_paths)
                or existing.get("retained_prefix_aggregate")
                != committed_forward_aggregate
            ):
                raise ArtifactCompatibilityError(
                    "historical evaluation retained forward prefix changed"
                )
            return dict(historical_anchors), existing
        if (
            existing is not None
            and existing.get("mode") == "fresh_forward_evaluation"
        ):
            if len(committed_forward_paths) != 64:
                raise ArtifactCompatibilityError(
                    "fresh evaluation forward completion changed"
                )
            completed_anchors = {
                SHORT_ANCHOR: _state_row(
                    _load_npz(committed_forward_paths[15].with_suffix(".npz"))[
                        "state"
                    ]
                ),
                FULL_ANCHOR: _state_row(
                    _load_npz(committed_forward_paths[63].with_suffix(".npz"))[
                        "state"
                    ]
                ),
            }
            if existing.get("anchors") != {
                str(key): _array_sha256(value)
                for key, value in completed_anchors.items()
            }:
                raise ArtifactCompatibilityError(
                    "fresh evaluation forward anchors changed"
                )
            return completed_anchors, existing
        if len(committed_forward_paths) == 64:
            completed_anchors = {
                SHORT_ANCHOR: _state_row(
                    _load_npz(committed_forward_paths[15].with_suffix(".npz"))[
                        "state"
                    ]
                ),
                FULL_ANCHOR: _state_row(
                    _load_npz(committed_forward_paths[63].with_suffix(".npz"))[
                        "state"
                    ]
                ),
            }
            recovered = _semantic(
                {
                    "schema": RECOVERY_RUN_SCHEMA + "-evaluation-anchor-plan",
                    "schema_version": 1,
                    "mode": "fresh_forward_evaluation",
                    "canonical_path_id": path_id,
                    "fresh_forward_attempted": 1,
                    "historical_anchor_reused": 0,
                    "diagnostics": committed_forward_aggregate,
                    "anchors": {
                        str(key): _array_sha256(value)
                        for key, value in completed_anchors.items()
                    },
                    "recovered_after_complete_verified_prefix": 1,
                    "passed": 1,
                }
            )
            atomic_write_json(plan_path, recovered)
            return completed_anchors, recovered
    elif (
        existing is not None
        and existing.get("mode") == "historical_anchor_stochastic_evaluation"
    ):
        if int(existing.get("fresh_forward_attempted", 0)) != 0:
            raise ArtifactCompatibilityError(
                "historical evaluation forward-attempt claim changed"
            )
        return dict(historical_anchors), existing
    if args.test_only:
        record = _semantic(
            {
                "schema": RECOVERY_TEST_RUN_SCHEMA + "-evaluation-anchor-plan",
                "schema_version": 1,
                "mode": "historical_anchor_stochastic_evaluation",
                "canonical_path_id": path_id,
                "fresh_forward_attempted": 0,
                "historical_anchor_reused": 1,
                "passed": 1,
            }
        )
        atomic_write_json(plan_path, record)
        return dict(historical_anchors), record
    # The predecessor's exact forward elapsed is the conservative no-new-profile
    # forecast.  Only run the fresh path when it fits the remaining main cap.
    predecessor = _load_semantic(run_dir / "predecessor_binding.json", "predecessor binding")
    forward_elapsed = float(predecessor["strict_forward_aggregate"]["elapsed_seconds"])
    usage = _recovery_observed_resource(run_dir)
    forward_timing_path = run_dir / "metrics/evaluation_forward_attempt_timing.json"
    forward_overhead = (
        float(
            _load_semantic(
                forward_timing_path, "evaluation forward attempt timing"
            ).get("commit_verification_overhead_seconds", 0.0)
        )
        if forward_timing_path.is_file()
        else 0.0
    )
    committed_forward_records = [
        _load_semantic(path, "evaluation forward committed prefix")
        for path in committed_forward_paths
    ]
    complete_forward_times = [
        float(row.get("elapsed_seconds", 0.0))
        + forward_overhead / max(1, len(committed_forward_records))
        for row in committed_forward_records
    ]
    remaining_forward = 64 - len(committed_forward_records)
    estimated_forward_bytes = max(
        (
            int(row.get("state_file_size", 0))
            + int(path.stat().st_size)
            for path, row in zip(
                committed_forward_paths, committed_forward_records
            )
        ),
        default=16 * 1024,
    )
    short_reserve_seconds, short_reserve_bytes = (
        _recovery_short_evaluation_reserve(run_dir)
    )
    initial_projection = _recovery_forward_tail_projection(
        active_seconds=usage["active_seconds"],
        wasted_active_seconds=usage["wasted_active_seconds"],
        observed_forward_shard_seconds=(
            complete_forward_times or (forward_elapsed / 64.0,)
        ),
        remaining_forward_shards=remaining_forward,
        short_reverse_reserve_seconds=short_reserve_seconds,
        maximum_main_seconds=float(args.maximum_main_seconds),
        persisted_bytes=usage["persisted_bytes"],
        projected_forward_bytes=remaining_forward * estimated_forward_bytes,
        short_reverse_reserve_bytes=short_reserve_bytes,
        peak_memory_fraction=usage["peak_memory_fraction"],
    )
    if not int(initial_projection["passed"]):
        record = _semantic(
            {
                "schema": RECOVERY_RUN_SCHEMA + "-evaluation-anchor-plan",
                "schema_version": 1,
                "mode": "historical_anchor_stochastic_evaluation",
                "canonical_path_id": path_id,
                "fresh_forward_attempted": int(bool(committed_forward_records)),
                "fresh_forward_completed_steps": 8
                * len(committed_forward_records),
                "historical_anchor_reused": 1,
                "partial_fresh_forward_preserved": int(
                    bool(committed_forward_records)
                ),
                "retained_prefix_aggregate": (
                    committed_forward_aggregate
                    if committed_forward_records
                    else None
                ),
                "tail_projection": initial_projection,
                "passed": 1,
            }
        )
        atomic_write_json(plan_path, record)
        return dict(historical_anchors), record
    from mnist.d0_jacobi_rb_tangent_rollout import run_forward_trajectory

    _source, target = _source_arrays(run_dir)
    initial = torch.as_tensor(
        target[None, :], dtype=torch.float64, device=args.device
    ).contiguous()
    anchors: dict[int, np.ndarray] = {}
    result: Any = None
    forward_root = run_dir / "objective/evaluation_forward"
    for limit in range(8 * (len(committed_forward_records) + 1), 513, 8):
        committed_paths = sorted(
            forward_root.glob(
                "forward_shards/objective-evaluation-forward/shard-*.json"
            )
        )
        observed = [
            float(_load_semantic(item, "evaluation forward shard").get("elapsed_seconds", 0.0))
            for item in committed_paths
        ]
        usage = _recovery_observed_resource(run_dir)
        remaining_tail = 64 - len(committed_paths)
        timing_path = run_dir / "metrics/evaluation_forward_attempt_timing.json"
        timing = (
            _load_semantic(timing_path, "evaluation forward attempt timing")
            if timing_path.is_file()
            else {}
        )
        per_shard_overhead = float(
            timing.get("commit_verification_overhead_seconds", 0.0)
        ) / max(1, len(committed_paths))
        observed_complete = [
            value + per_shard_overhead for value in observed
        ]
        projected_forward_bytes = max(
            (
                int(
                    _load_semantic(item, "evaluation forward shard").get(
                        "state_file_size", 0
                    )
                )
                + int(item.stat().st_size)
                for item in committed_paths
            ),
            default=estimated_forward_bytes,
        )
        step_projection = _recovery_forward_tail_projection(
            active_seconds=usage["active_seconds"],
            wasted_active_seconds=usage["wasted_active_seconds"],
            observed_forward_shard_seconds=(
                observed_complete or (forward_elapsed / 64.0,)
            ),
            remaining_forward_shards=remaining_tail,
            short_reverse_reserve_seconds=short_reserve_seconds,
            maximum_main_seconds=float(args.maximum_main_seconds),
            persisted_bytes=usage["persisted_bytes"],
            projected_forward_bytes=remaining_tail * projected_forward_bytes,
            short_reverse_reserve_bytes=short_reserve_bytes,
            peak_memory_fraction=usage["peak_memory_fraction"],
        )
        if not int(step_projection["passed"]):
            record = _semantic(
                {
                    "schema": RECOVERY_RUN_SCHEMA + "-evaluation-anchor-plan",
                    "schema_version": 1,
                    "mode": "historical_anchor_stochastic_evaluation",
                    "canonical_path_id": path_id,
                    "fresh_forward_attempted": 1,
                    "fresh_forward_completed_steps": 8 * len(committed_paths),
                    "historical_anchor_reused": 1,
                    "stopped_at_verified_shard_boundary": 1,
                    "last_projection": step_projection,
                    "passed": 1,
                }
            )
            atomic_write_json(plan_path, record)
            return dict(historical_anchors), record
        prior_timing = (
            _load_semantic(timing_path, "evaluation forward attempt timing")
            if timing_path.is_file()
            else {}
        )
        prior_count = len(committed_paths)
        prior_raw = math.fsum(
            float(_load_semantic(item, "evaluation forward shard").get("elapsed_seconds", 0.0))
            for item in committed_paths
        )
        started = time.perf_counter()
        try:
            result = run_forward_trajectory(
                initial,
                anchor_steps=(limit - 1,),
                output_dir=forward_root,
                trajectory_name="objective-evaluation-forward",
                path_ids=(path_id,),
                root_seed=FORWARD_ROOT_SEED,
                profile=JacobiRBCudaProfile(),
                step_limit=limit,
                device=torch.device(args.device),
            )
        except Exception as exc:
            if isinstance(exc, ArtifactCompatibilityError):
                raise
            measured = time.perf_counter() - started
            after_paths = sorted(
                forward_root.glob(
                    "forward_shards/objective-evaluation-forward/shard-*.json"
                )
            )
            after_records = [
                _load_semantic(item, "evaluation forward failed prefix")
                for item in after_paths
            ]
            raw = math.fsum(
                float(item.get("elapsed_seconds", 0.0)) for item in after_records
            )
            overhead = max(0.0, measured - max(0.0, raw - prior_raw))
            atomic_write_json(
                timing_path,
                _semantic(
                    {
                        "schema": RECOVERY_RUN_SCHEMA
                        + "-evaluation-forward-attempt-timing",
                        "schema_version": 1,
                        "verified_shard_count": len(after_records),
                        "raw_shard_elapsed_seconds": raw,
                        "commit_verification_overhead_seconds": float(
                            prior_timing.get(
                                "commit_verification_overhead_seconds", 0.0
                            )
                        )
                        + overhead,
                        "failed_attempt": 1,
                    }
                ),
            )
            failure_root = run_dir / "failure_artifacts/evaluation-forward"
            failure_root.mkdir(parents=True, exist_ok=True)
            last_state = (
                np.asarray(
                    _load_npz(after_paths[-1].with_suffix(".npz"))["state"],
                    dtype=np.float64,
                )
                if after_paths
                else np.asarray(target[None, :], dtype=np.float64)
            )
            state_artifact = _atomic_npz(
                failure_root / "last_valid_states.npz", state=last_state
            )
            atomic_write_json(
                failure_root / "failure_evidence.json",
                _semantic(
                    {
                        "schema": RECOVERY_RUN_SCHEMA
                        + "-evaluation-forward-failure-evidence",
                        "schema_version": 1,
                        "last_valid_states": {
                            **state_artifact,
                            "path": (
                                failure_root / "last_valid_states.npz"
                            ).relative_to(run_dir).as_posix(),
                        },
                        "committed_shard_count": len(after_records),
                        "numerically_interpretable_partial_evidence": int(
                            np.isfinite(last_state).all()
                            and np.all(last_state >= 0.0)
                            and np.max(
                                np.abs(last_state.sum(axis=1) - 1.0)
                            )
                            <= MAXIMUM_MASS_ERROR
                        ),
                        "message": str(exc),
                    }
                ),
            )
            raise RolloutCLIError(
                f"fresh evaluation forward failed: {exc}",
                failure_domain="numerical_integrity",
                failure_code="evaluation_forward_exact_health_invalid",
            ) from exc
        verified_paths = sorted(
            forward_root.glob(
                "forward_shards/objective-evaluation-forward/shard-*.json"
            )
        )
        # Semantic-load every committed record after the call.  The strict
        # forward aggregate inside ``run_forward_trajectory`` has already
        # verified the JSON/NPZ chain; this read makes the same-process
        # restart boundary explicit and charges it to the hard cap.
        verified_records = [
            _load_semantic(item, "evaluation forward shard") for item in verified_paths
        ]
        measured = time.perf_counter() - started
        if len(verified_records) > prior_count:
            raw = math.fsum(float(item.get("elapsed_seconds", 0.0)) for item in verified_records)
            overhead = max(0.0, measured - max(0.0, raw - prior_raw))
            atomic_write_json(
                timing_path,
                _semantic(
                    {
                        "schema": RECOVERY_RUN_SCHEMA + "-evaluation-forward-attempt-timing",
                        "schema_version": 1,
                        "verified_shard_count": len(verified_records),
                        "raw_shard_elapsed_seconds": raw,
                        "commit_verification_overhead_seconds": float(
                            prior_timing.get("commit_verification_overhead_seconds", 0.0)
                        )
                        + overhead,
                        "completed_prefix_is_never_retimed": 1,
                    }
                ),
            )
        if limit in {128, 512}:
            anchors[limit - 1] = _state_row(_field(result, "anchors", {})[limit - 1])
    assert result is not None
    diagnostics = dict(_field(result, "diagnostics", {}))
    if not int(diagnostics.get("passed", 1)):
        raise RolloutCLIError(
            "fresh evaluation forward path failed strict aggregate health",
            failure_domain="numerical_integrity",
            failure_code="evaluation_forward_exact_health_invalid",
        )
    record = _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-evaluation-anchor-plan",
            "schema_version": 1,
            "mode": "fresh_forward_evaluation",
            "canonical_path_id": path_id,
            "fresh_forward_attempted": 1,
            "historical_anchor_reused": 0,
            "diagnostics": diagnostics,
            "anchors": {str(key): _array_sha256(value) for key, value in anchors.items()},
            "passed": 1,
        }
    )
    if existing is not None:
        if existing != record:
            raise ArtifactCompatibilityError("fresh evaluation anchor plan changed")
        return anchors, existing
    atomic_write_json(plan_path, record)
    return anchors, record


def _commit_recovery_evaluation_family(
    run_dir: Path,
    *,
    horizon: str,
    result: Any,
    backend: str,
    audit: Mapping[str, Any] | None,
    family_summary: Mapping[str, Any],
) -> dict[str, Any]:
    row_keys = (
        f"evaluation-{horizon}-zero",
        f"evaluation-{horizon}-learned",
        f"evaluation-{horizon}-source-informed",
    )
    saved = {
        str(name): np.ascontiguousarray(value, dtype=np.float64)
        for name, value in _field(result, "saved_states", {}).items()
    }
    required = ("start", "progress_25", "progress_50", "progress_75", "final")
    if any(name not in saved or saved[name].shape != (3, 784) for name in required):
        raise RolloutCLIError(
            f"evaluation {horizon} omitted required saved states",
            failure_domain="implementation_contract",
            failure_code="rollout_implementation_contract_invalid",
        )
    family_root = run_dir / "objective" / f"evaluation-{horizon}"
    family_root.mkdir(parents=True, exist_ok=True)
    raw_path = family_root / "selected_states.npz"
    raw = _atomic_npz(raw_path, **saved)
    raw["path"] = raw_path.relative_to(run_dir).as_posix()
    _source, target = _source_arrays(run_dir)
    mechanism = _recovery_mechanism_rows(result)
    row_results: list[dict[str, Any]] = []
    for row_index, row_key in enumerate(row_keys):
        states = {name: saved[name][row_index] for name in required}
        zero_states = {name: saved[name][0] for name in required}
        row_results.append(
            {
                "row_key": row_key,
                "metrics_to_mixed_target": {
                    name: _metrics_dict(value, target) for name, value in states.items()
                },
                "paired_zero_divergence": {
                    name: {
                        "squared_l2": float(np.dot(value - zero_states[name], value - zero_states[name])),
                        "l1": float(np.sum(np.abs(value - zero_states[name]))),
                        "total_variation": float(0.5 * np.sum(np.abs(value - zero_states[name]))),
                        "centered_correlation": _centered_correlation(value, zero_states[name]),
                    }
                    for name, value in states.items()
                },
                "mechanism_diagnostics": mechanism[row_index] if row_index < len(mechanism) else {},
                "images": _render_states(run_dir, trajectory_key=row_key, states=states),
            }
        )
    from mnist.d0_jacobi_rb_tangent_rollout import fixed_rendering_scale, render_background_demixed

    source, target = _source_arrays(run_dir)
    scale = fixed_rendering_scale(source, target, LAMBDA_MIX)
    contact = run_dir / "images" / f"evaluation_{horizon}_contact_sheet.png"
    _contact_sheet(
        contact,
        [
            (f"{row_keys[row]}-{name}", render_background_demixed(saved[name][row], scale))
            for row in range(3)
            for name in required
        ],
    )
    risks = [
        float(row["metrics_to_mixed_target"]["final"]["squared_l2_error"])
        for row in row_results
    ]
    endpoint_delta = saved["final"][1] - saved["final"][0]
    endpoint_separation = float(np.dot(endpoint_delta, endpoint_delta))
    guard: dict[str, Any] | None = None
    approximation_dominates = False
    if backend == "candidate":
        if audit is None:
            raise RolloutCLIError(
                "candidate evaluation lacks its same-anchor exact audit",
                failure_domain="implementation_contract",
                failure_code="rollout_implementation_contract_invalid",
            )
        largest_local = max(
            float(row["squared_l2_state_discrepancy"])
            for row in audit["row_metrics"]
        )
        relative = float(
            audit["paired_contrasts"]["learned_minus_zero"]["relative_error"]
        )
        guard = {
            "largest_first_shard_squared_l2_discrepancy": largest_local,
            "endpoint_learned_zero_squared_l2_separation": endpoint_separation,
            "required_endpoint_multiple": 4.0,
            "learned_paired_contrast_relative_error": relative,
            "learned_contrast_guard_passed": int(relative <= 0.25),
            "endpoint_scale_guard_passed": int(endpoint_separation >= 4.0 * largest_local),
        }
        approximation_dominates = not (
            guard["learned_contrast_guard_passed"]
            and guard["endpoint_scale_guard_passed"]
        )
        guard["coarse_candidate_dynamic_claim_permitted"] = int(not approximation_dominates)
    if approximation_dominates:
        decision = f"evaluation_{horizon}_approximation_dominates_observed_effect"
        next_action = "retain the images and improve or extend the exact fixed-case audit before interpreting learned utility"
    elif risks[1] < risks[0]:
        decision = f"learned_{horizon}_dynamic_signal_{backend}"
        next_action = (
            "run a small multi-image M=2 exploration with an exact audit on a fixed subset"
            if horizon == "full"
            else "attempt the affordable full reverse path before scaling to multiple images"
        )
    elif math.isclose(risks[1], risks[0], rel_tol=0.0, abs_tol=1.0e-14):
        decision = f"evaluation_{horizon}_control_dynamically_negligible"
        next_action = "revise calibration or compare rollout-trained/global alternatives before scaling"
    else:
        decision = f"evaluation_{horizon}_rollout_direction_not_useful"
        next_action = "inspect sign/order and compare a materially different learner or controller"
    record = _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-evaluation-family",
            "schema_version": 1,
            "horizon": horizon,
            "backend": backend,
            "family_summary_sha256": family_summary["semantic_sha256"],
            "raw_states": raw,
            "row_results": row_results,
            "contact_sheet": contact.relative_to(run_dir).as_posix(),
            "mechanism_diagnostics": mechanism,
            "final_squared_l2": {
                "zero": risks[0], "learned": risks[1], "source_informed": risks[2]
            },
            "learned_minus_zero_risk_improvement": risks[0] - risks[1],
            "source_informed_minus_zero_risk_improvement": risks[0] - risks[2],
            "source_informed_composition_mismatch_diagnostic": int(
                risks[2] > risks[0]
            ),
            "endpoint_learned_zero_squared_l2_separation": endpoint_separation,
            "exact_candidate_claim_guard": guard,
            "exact_candidate_audit_sha256": audit.get("semantic_sha256") if audit else None,
            "decision": decision,
            "recommended_next_action": next_action,
            "source_informed_endpoint_is_diagnostic": 1,
            "passed_numerical_integrity": 1,
        }
    )
    record_path = family_root / "evaluation_family.json"
    if record_path.is_file():
        existing = _load_semantic(record_path, f"evaluation {horizon} family")
        if existing != record:
            raise ArtifactCompatibilityError(
                f"evaluation {horizon} family result changed"
            )
        return existing
    atomic_write_json(record_path, record)
    return record


def _complete_reverse_shard_seconds(
    run_dir: Path,
    values: Sequence[float],
    *,
    family_name: str,
    segment_name: str,
    backend: str,
    row_count: int,
) -> list[float]:
    """Return complete times for one homogeneous reverse-family contract.

    Commit/restart overhead is deliberately scoped to the same backend, family,
    segment and row count.  Pooling it over unrelated exact/candidate or
    one-/three-row families can dilute the measured slowest shard and make the
    1.20x resource projection favorable.
    """

    token = f"{family_name}-{segment_name}-{backend}".replace("/", "-")
    timing_path = run_dir / f"metrics/recovery_attempt_timing_{token}.json"
    overhead = 0.0
    verified_count = 0
    if timing_path.is_file():
        timing = _load_semantic(timing_path, "recovery attempt timing")
        if (
            timing.get("family_name") != family_name
            or timing.get("segment_name") != segment_name
            or timing.get("backend") != backend
            or not isinstance(timing.get("row_count"), int)
            or isinstance(timing.get("row_count"), bool)
            or int(timing["row_count"]) != int(row_count)
            or not isinstance(timing.get("verified_shard_count"), int)
            or isinstance(timing.get("verified_shard_count"), bool)
            or int(timing["verified_shard_count"]) != len(values)
        ):
            raise ArtifactCompatibilityError(
                "recovery attempt timing scope changed"
            )
        overhead = float(timing.get("commit_verification_overhead_seconds", 0.0))
        verified_count = int(timing["verified_shard_count"])
        if not math.isfinite(overhead) or overhead < 0.0:
            raise ArtifactCompatibilityError(
                "recovery attempt timing overhead is invalid"
            )
        complete = timing.get("complete_shard_seconds")
        if complete is not None:
            if (
                not isinstance(complete, list)
                or len(complete) != len(values)
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                    for value in complete
                )
            ):
                raise ArtifactCompatibilityError(
                    "recovery complete shard timing changed"
                )
            return [float(value) for value in complete]
    per_shard = overhead / max(1, verified_count)
    return [float(value) + per_shard for value in values]


def _recovery_matching_shard_observations(
    run_dir: Path,
    *,
    backend: str,
    row_count: int,
    fallback_backend: str | None = None,
) -> tuple[list[float], int]:
    """Collect homogeneous complete-shard time/storage measurements.

    Measurements never cross backend or row-count boundaries.  A caller may name
    an explicit conservative fallback backend (candidate shard zero uses the exact
    audit ceiling); otherwise absence remains absence.
    """

    def collect(selected_backend: str) -> tuple[list[float], int]:
        observations: list[float] = []
        maximum_bytes = 0
        roots = run_dir / f"objective_attempts/{selected_backend}/fused_families"
        for root in sorted(roots.glob("*/*")):
            if not root.is_dir():
                continue
            records = _recovery_shard_records(root)
            if not records:
                continue
            record_row_count = len(records[0].get("row_table", ()))
            if record_row_count != int(row_count):
                continue
            observations.extend(
                _complete_reverse_shard_seconds(
                    run_dir,
                    [float(row["elapsed_seconds"]) for row in records],
                    family_name=root.parent.name,
                    segment_name=root.name,
                    backend=selected_backend,
                    row_count=record_row_count,
                )
            )
            maximum_bytes = max(
                maximum_bytes,
                _recovery_max_committed_shard_bytes(root, records),
            )
        return observations, maximum_bytes

    observations, maximum_bytes = collect(backend)
    if observations or fallback_backend is None:
        return observations, maximum_bytes
    return collect(fallback_backend)


def _recovery_forward_tail_projection(
    *,
    active_seconds: float,
    wasted_active_seconds: float,
    observed_forward_shard_seconds: Sequence[float],
    remaining_forward_shards: int,
    short_reverse_reserve_seconds: float,
    maximum_main_seconds: float,
    persisted_bytes: int,
    projected_forward_bytes: int,
    short_reverse_reserve_bytes: int,
    peak_memory_fraction: float,
) -> dict[str, Any]:
    """Conservatively price the uncommitted FB200 tail plus short reverse."""

    observed = tuple(float(value) for value in observed_forward_shard_seconds)
    if (
        remaining_forward_shards < 0
        or not observed
        or any(not math.isfinite(value) or value <= 0.0 for value in observed)
        or not math.isfinite(short_reverse_reserve_seconds)
        or short_reverse_reserve_seconds < 0.0
    ):
        raise ArtifactCompatibilityError("fresh forward tail projection is invalid")
    slowest = max(observed)
    projected_seconds = (
        float(active_seconds)
        + float(wasted_active_seconds)
        + RECOVERY_PROJECTION_FACTOR
        * slowest
        * int(remaining_forward_shards)
        + float(short_reverse_reserve_seconds)
        + RECOVERY_REPORT_RESERVE_SECONDS
    )
    projected_bytes = (
        int(persisted_bytes)
        + int(projected_forward_bytes)
        + int(short_reverse_reserve_bytes)
    )
    checks = {
        "main_time": projected_seconds <= float(maximum_main_seconds),
        "persisted_storage": projected_bytes <= MAXIMUM_PERSISTED_BYTES,
        "peak_cuda_memory": float(peak_memory_fraction)
        <= MAXIMUM_MEMORY_FRACTION,
    }
    return _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-fresh-forward-tail-projection",
            "schema_version": 1,
            "remaining_forward_shards": int(remaining_forward_shards),
            "observed_slowest_complete_forward_shard_seconds": slowest,
            "short_reverse_reserve_seconds": float(
                short_reverse_reserve_seconds
            ),
            "projected_total_seconds": projected_seconds,
            "maximum_main_seconds": float(maximum_main_seconds),
            "projected_persisted_bytes": projected_bytes,
            "short_reverse_reserve_bytes": int(short_reverse_reserve_bytes),
            "peak_memory_fraction": float(peak_memory_fraction),
            "checks": checks,
            "passed": int(all(checks.values())),
        }
    )


def _recovery_short_evaluation_reserve(run_dir: Path) -> tuple[float, int]:
    """Reserve one exact audit plus the likely 16-shard short family."""

    decision = (
        _load_semantic(run_dir / "backend_decision.json", "backend decision")
        if (run_dir / "backend_decision.json").is_file()
        else {}
    )
    selected = str(decision.get("selected", "exact"))
    by_backend: dict[str, list[float]] = {"exact": [], "candidate": []}
    shard_bytes: list[int] = []
    for backend in by_backend:
        for root in (run_dir / f"objective_attempts/{backend}/fused_families").glob(
            "*/*"
        ):
            if not root.is_dir():
                continue
            records = _recovery_shard_records(root)
            by_backend[backend].extend(
                _complete_reverse_shard_seconds(
                    run_dir,
                    [float(row.get("elapsed_seconds", 0.0)) for row in records],
                    family_name=root.parent.name,
                    segment_name=root.name,
                    backend=backend,
                    row_count=len(records[0].get("row_table", ())) if records else 0,
                )
            )
            shard_bytes.extend(
                int(row.get("state_file_size", 0))
                + int((root / f"shard-{index:04d}.json").stat().st_size)
                for index, row in enumerate(records)
            )
    exact = max(by_backend["exact"], default=0.0)
    candidate = max(by_backend["candidate"], default=exact)
    if selected == "candidate":
        seconds = RECOVERY_PROJECTION_FACTOR * (exact + 16 * candidate)
        count = 17
    else:
        seconds = RECOVERY_PROJECTION_FACTOR * 16 * exact
        count = 16
    return seconds, count * max(shard_bytes, default=64 * 1024)


def _verify_recovery_evaluation_family_artifact(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    horizon: str,
    selected_gain: float,
    anchor: np.ndarray,
) -> dict[str, Any]:
    record = _load_semantic(
        run_dir / f"objective/evaluation-{horizon}/evaluation_family.json",
        f"evaluation {horizon} family",
    )
    backend = str(record.get("backend", ""))
    if (
        record.get("schema") != RECOVERY_RUN_SCHEMA + "-evaluation-family"
        or backend not in {"exact", "candidate"}
        or record.get("horizon") != horizon
        or int(record.get("passed_numerical_integrity", 0)) != 1
    ):
        raise ArtifactCompatibilityError(f"evaluation {horizon} binding changed")
    rows = (
        (f"evaluation-{horizon}-zero", "zero", horizon, None),
        (f"evaluation-{horizon}-learned", "learned", horizon, selected_gain),
        (f"evaluation-{horizon}-source-informed", "oracle", horizon, None),
    )
    summary, family_saved = _verify_recovery_completed_family_artifacts(
        run_dir,
        args,
        backend=backend,
        family_name=f"evaluation-{horizon}",
        segment_name=horizon,
        rows=rows,
        canonical_role="evaluation",
        stream_role="frequency1-objective-first-evaluation-v1",
        anchor=anchor,
        sequence=_reverse_sequence(
            SHORT_ANCHOR if horizon == "short" else FULL_ANCHOR
        ),
    )
    if record.get("family_summary_sha256") != summary.get("semantic_sha256"):
        raise ArtifactCompatibilityError(
            f"evaluation {horizon} family summary binding changed"
        )
    raw = record.get("raw_states")
    if not isinstance(raw, Mapping):
        raise ArtifactCompatibilityError(f"evaluation {horizon} raw binding changed")
    raw_path = run_dir / str(raw.get("path", ""))
    if (
        raw_path != run_dir / f"objective/evaluation-{horizon}/selected_states.npz"
        or not raw_path.is_file()
        or raw.get("sha256") != file_fingerprint(raw_path)
        or int(raw.get("size", -1)) != int(raw_path.stat().st_size)
    ):
        raise ArtifactCompatibilityError(f"evaluation {horizon} raw states changed")
    raw_saved = _load_npz(raw_path)
    if set(raw_saved) != set(family_saved) or any(
        _array_sha256(raw_saved[name]) != _array_sha256(family_saved[name])
        for name in family_saved
    ):
        raise ArtifactCompatibilityError(
            f"evaluation {horizon} raw/family states disagree"
        )
    final = np.asarray(raw_saved["final"], dtype=np.float64)
    _source, target = _source_arrays(run_dir)
    risks = [_metrics_dict(final[index], target)["squared_l2_error"] for index in range(3)]
    mechanism = _recovery_mechanism_rows(summary["result"])
    stored_rows = record.get("row_results")
    if not isinstance(stored_rows, list) or len(stored_rows) != 3:
        raise ArtifactCompatibilityError(f"evaluation {horizon} rows changed")
    expected_rows: list[dict[str, Any]] = []
    row_keys = tuple(row[0] for row in rows)
    for row_index, row_key in enumerate(row_keys):
        states = {name: value[row_index] for name, value in raw_saved.items()}
        zero_states = {name: value[0] for name, value in raw_saved.items()}
        images = stored_rows[row_index].get("images", {})
        if not isinstance(images, Mapping):
            raise ArtifactCompatibilityError(
                f"evaluation {horizon} image binding changed"
            )
        _verify_rendered_state_records(
            run_dir,
            trajectory_key=row_key,
            states=states,
            records=images,
        )
        expected_rows.append(
            {
                "row_key": row_key,
                "metrics_to_mixed_target": {
                    name: _metrics_dict(value, target)
                    for name, value in states.items()
                },
                "paired_zero_divergence": {
                    name: {
                        "squared_l2": float(
                            np.dot(value - zero_states[name], value - zero_states[name])
                        ),
                        "l1": float(np.sum(np.abs(value - zero_states[name]))),
                        "total_variation": float(
                            0.5 * np.sum(np.abs(value - zero_states[name]))
                        ),
                        "centered_correlation": _centered_correlation(
                            value, zero_states[name]
                        ),
                    }
                    for name, value in states.items()
                },
                "mechanism_diagnostics": mechanism[row_index],
                "images": dict(images),
            }
        )
    if record.get("row_results") != expected_rows or record.get(
        "mechanism_diagnostics"
    ) != mechanism:
        raise ArtifactCompatibilityError(
            f"evaluation {horizon} derived row evidence changed"
        )
    from mnist.d0_jacobi_rb_tangent_rollout import (
        fixed_rendering_scale,
        render_background_demixed,
    )

    source, mixed_target = _source_arrays(run_dir)
    scale = fixed_rendering_scale(source, mixed_target, LAMBDA_MIX)
    contact_cells = [
        (
            f"{row_keys[row_index]}-{name}",
            render_background_demixed(raw_saved[name][row_index], scale),
        )
        for row_index in range(3)
        for name in ("start", "progress_25", "progress_50", "progress_75", "final")
    ]
    contact_path = run_dir / str(record.get("contact_sheet", ""))
    if (
        not contact_path.is_file()
        or file_fingerprint(contact_path) != _contact_sheet_sha256(contact_cells)
    ):
        raise ArtifactCompatibilityError(
            f"evaluation {horizon} contact sheet changed"
        )
    approximation_dominates = False
    audit = None
    if backend == "candidate":
        audit = _verify_recovery_candidate_audit(
            run_dir,
            audit_path=run_dir
            / f"exact_candidate_audit_evaluation_{horizon}.json",
            exact_root=run_dir
            / f"objective_attempts/exact/fused_families/evaluation-{horizon}/{horizon}",
            candidate_root=run_dir
            / f"objective_attempts/candidate/fused_families/evaluation-{horizon}/{horizon}",
            row_keys=tuple(row[0] for row in rows),
        )
        if record.get("exact_candidate_audit_sha256") != audit.get(
            "semantic_sha256"
        ):
            raise ArtifactCompatibilityError(
                f"evaluation {horizon} audit binding changed"
            )
        largest = max(
            float(row["squared_l2_state_discrepancy"])
            for row in audit["row_metrics"]
        )
        separation = float(np.dot(final[1] - final[0], final[1] - final[0]))
        approximation_dominates = not (
            float(audit["paired_contrasts"]["learned_minus_zero"]["relative_error"])
            <= 0.25
            and separation >= 4.0 * largest
        )
    elif record.get("exact_candidate_audit_sha256") is not None:
        raise ArtifactCompatibilityError(
            f"exact evaluation {horizon} acquired candidate audit authority"
        )
    expected_decision = (
        f"evaluation_{horizon}_approximation_dominates_observed_effect"
        if approximation_dominates
        else f"learned_{horizon}_dynamic_signal_{backend}"
        if risks[1] < risks[0]
        else f"evaluation_{horizon}_control_dynamically_negligible"
        if math.isclose(risks[1], risks[0], rel_tol=0.0, abs_tol=1.0e-14)
        else f"evaluation_{horizon}_rollout_direction_not_useful"
    )
    separation = float(np.dot(final[1] - final[0], final[1] - final[0]))
    guard = None
    if audit is not None:
        largest = max(
            float(row["squared_l2_state_discrepancy"])
            for row in audit["row_metrics"]
        )
        relative = float(
            audit["paired_contrasts"]["learned_minus_zero"]["relative_error"]
        )
        guard = {
            "largest_first_shard_squared_l2_discrepancy": largest,
            "endpoint_learned_zero_squared_l2_separation": separation,
            "required_endpoint_multiple": 4.0,
            "learned_paired_contrast_relative_error": relative,
            "learned_contrast_guard_passed": int(relative <= 0.25),
            "endpoint_scale_guard_passed": int(separation >= 4.0 * largest),
            "coarse_candidate_dynamic_claim_permitted": int(
                relative <= 0.25 and separation >= 4.0 * largest
            ),
        }
    if expected_decision.endswith("approximation_dominates_observed_effect"):
        expected_action = "retain the images and improve or extend the exact fixed-case audit before interpreting learned utility"
    elif expected_decision.startswith("learned_"):
        expected_action = (
            "run a small multi-image M=2 exploration with an exact audit on a fixed subset"
            if horizon == "full"
            else "attempt the affordable full reverse path before scaling to multiple images"
        )
    elif expected_decision.endswith("control_dynamically_negligible"):
        expected_action = "revise calibration or compare rollout-trained/global alternatives before scaling"
    else:
        expected_action = "inspect sign/order and compare a materially different learner or controller"
    expected_record = _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-evaluation-family",
            "schema_version": 1,
            "horizon": horizon,
            "backend": backend,
            "family_summary_sha256": summary["semantic_sha256"],
            "raw_states": dict(raw),
            "row_results": expected_rows,
            "contact_sheet": contact_path.relative_to(run_dir).as_posix(),
            "mechanism_diagnostics": mechanism,
            "final_squared_l2": {
                "zero": risks[0],
                "learned": risks[1],
                "source_informed": risks[2],
            },
            "learned_minus_zero_risk_improvement": risks[0] - risks[1],
            "source_informed_minus_zero_risk_improvement": risks[0] - risks[2],
            "source_informed_composition_mismatch_diagnostic": int(risks[2] > risks[0]),
            "endpoint_learned_zero_squared_l2_separation": separation,
            "exact_candidate_claim_guard": guard,
            "exact_candidate_audit_sha256": audit.get("semantic_sha256") if audit else None,
            "decision": expected_decision,
            "recommended_next_action": expected_action,
            "source_informed_endpoint_is_diagnostic": 1,
            "passed_numerical_integrity": 1,
        }
    )
    if record != expected_record:
        raise ArtifactCompatibilityError(f"evaluation {horizon} metrics changed")
    return record


def _recovery_run_evaluation_horizon(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    horizon: str,
    anchor: np.ndarray,
    selected_gain: float,
) -> dict[str, Any]:
    rows = (
        (f"evaluation-{horizon}-zero", "zero", horizon, None),
        (f"evaluation-{horizon}-learned", "learned", horizon, selected_gain),
        (f"evaluation-{horizon}-source-informed", "oracle", horizon, None),
    )
    sequence = _reverse_sequence(SHORT_ANCHOR if horizon == "short" else FULL_ANCHOR)
    common = {
        "anchor": anchor,
        "rows": rows,
        "canonical_role": "evaluation",
        "family_name": f"evaluation-{horizon}",
        "segment_name": horizon,
        "stream_role": "frequency1-objective-first-evaluation-v1",
    }
    decision_path = run_dir / f"evaluation_{horizon}_backend_decision.json"
    opening_path = run_dir / f"evaluation_{horizon}_backend_selection_opening.json"
    switch_path = run_dir / f"evaluation_{horizon}_backend_switch_event.json"
    prior_decision = (
        _load_semantic(decision_path, "evaluation backend decision")
        if decision_path.is_file()
        else {}
    )
    prior_selected = str(prior_decision.get("selected", "not_attempted"))
    if prior_selected not in {"not_attempted", "exact", "candidate"}:
        raise ArtifactCompatibilityError("evaluation backend selection changed")
    exact_root_expected = (
        run_dir
        / f"objective_attempts/exact/fused_families/evaluation-{horizon}/{horizon}"
    )
    exact_count = (
        len(_recovery_shard_records(exact_root_expected))
        if exact_root_expected.is_dir()
        else 0
    )
    if exact_count:
        path_ids = _recovery_path_ids(run_dir)
        specs, _bank, binding = _fused_controller_family(
            run_dir,
            args,
            rows=rows,
            canonical_path_id=path_ids["evaluation"],
        )
        initial = np.repeat(_state_row(anchor)[None, :], len(specs), axis=0)
        _verify_fused_family_prefix(
            exact_root_expected,
            initial_state=initial,
            sequence=sequence,
            row_specs=specs,
            controller_binding=binding,
            rng_binding={
                "root_seed": REVERSE_ROOT_SEED,
                "stream_role": "frequency1-objective-first-evaluation-v1",
                "canonical_path_id": path_ids["evaluation"],
                "variant_in_rng_key": 0,
                "backend_and_attempt_in_rng_key": 0,
            },
            family_name=f"evaluation-{horizon}",
            segment_name=horizon,
            reference_contract="certified_exact",
        )
        exact_root = exact_root_expected
        audit_path = run_dir / f"exact_audit_attempt_evaluation-{horizon}-{horizon}.json"
        if audit_path.is_file():
            _exact_summary = _load_semantic(audit_path, "evaluation exact audit")
            if (
                _exact_summary.get("backend") != "exact"
                or int(_exact_summary.get("passed", 0)) != 1
            ):
                raise ArtifactCompatibilityError("evaluation exact audit changed")
        else:
            first = _recovery_shard_records(exact_root)[0]
            _exact_summary = _semantic(
                {
                    "schema": RECOVERY_RUN_SCHEMA + "-exact-audit-attempt",
                    "schema_version": 1,
                    "backend": "exact",
                    "result": {"row_table": first["row_table"]},
                    "committed_shard_semantic_sha256": first["semantic_sha256"],
                    "recovered_from_verified_committed_shard": 1,
                    "separate_backend_selection_attempt": 1,
                    "scientific_trajectory_complete": 0,
                    "passed": 1,
                }
            )
            atomic_write_json(audit_path, _exact_summary)
    else:
        exact_attempt = _run_recovery_optional_backend(
            run_dir,
            args,
            backend="exact",
            sequence=sequence[: 7 * int(args.exact_audit_outer_steps)],
            exact_audit_only=True,
            **common,
        )
        if exact_attempt is None:
            return {
                "performed": 0,
                "reason": f"{horizon}_exact_audit_resource_deferred",
            }
        _exact_result, _exact_summary, exact_root = exact_attempt
    exact_records = _recovery_shard_records(exact_root)
    remaining = max(0, len(sequence) // 56 - len(exact_records))
    exact_complete_seconds = _complete_reverse_shard_seconds(
        run_dir,
        [float(row.get("elapsed_seconds", 0.0)) for row in exact_records],
        family_name=f"evaluation-{horizon}",
        segment_name=horizon,
        backend="exact",
        row_count=3,
    )
    projected_exact_bytes = remaining * _recovery_max_committed_shard_bytes(
        exact_root, exact_records
    )
    ledger = _write_recovery_ledger(
        run_dir,
        args,
        backend="exact",
        remaining_shards=remaining,
        observed_shard_seconds=exact_complete_seconds,
        projected_additional_bytes=projected_exact_bytes,
    )
    if decision_path.is_file():
        opening = _load_semantic(opening_path, "evaluation backend selection opening")
        stored_projection = opening.get("exact_projection")
        degraded = int(
            prior_decision.get("exact_degraded_at_verified_boundary", 0)
        )
        try:
            projection_selected = _verify_recovery_backend_opening(
                run_dir,
                args,
                opening=opening,
                exact_root=exact_root,
                total_shards=len(sequence) // 56,
                expected_schema=RECOVERY_RUN_SCHEMA
                + "-evaluation-backend-selection-opening",
                expected_audit_sha256=_exact_summary["semantic_sha256"],
                expected_horizon=horizon,
            )
        except RolloutCLIError as exc:
            raise ArtifactCompatibilityError(
                "evaluation backend projection authority changed"
            ) from exc
        selection_authorized = (
            prior_selected == "candidate"
            and degraded == 1
            and _candidate_backend_selected(
                stored_projection, args.reference_backend
            ) == "exact"
            and _is_semantic_record(
                prior_decision.get("exact_degraded_projection")
            )
            and int(
                prior_decision["exact_degraded_projection"].get("passed", 1)
            )
            == 0
            and switch_path.is_file()
        ) if degraded else (
            prior_decision.get("exact_degraded_projection") is None
            and not switch_path.exists()
            and projection_selected == prior_selected
        )
        if degraded and selection_authorized:
            verified_switch = _verify_recovery_switch_event(
                run_dir,
                args,
                event=_load_semantic(
                    switch_path, "evaluation backend switch event"
                ),
                exact_root=exact_root,
                total_shards=len(sequence) // 56,
                exact_audit_sha256=_exact_summary["semantic_sha256"],
                selection_opening_sha256=opening["semantic_sha256"],
                expected_schema=RECOVERY_RUN_SCHEMA
                + "-evaluation-backend-switch-event",
                restart_field="restart_from_horizon_anchor",
                horizon=horizon,
            )
            selection_authorized = (
                verified_switch.get("failed_exact_projection")
                == prior_decision.get("exact_degraded_projection")
                and prior_decision.get("switch_event_sha256")
                == verified_switch.get("semantic_sha256")
            )
        if (
            prior_decision.get("selection_opening_sha256")
            != opening.get("semantic_sha256")
            or prior_decision.get("exact_projection") != stored_projection
            or
            prior_decision.get("schema")
            != RECOVERY_RUN_SCHEMA + "-evaluation-backend-decision"
            or prior_decision.get("horizon") != horizon
            or prior_decision.get("requested") != args.reference_backend
            or int(prior_decision.get("exact_audit_shard_committed", 0)) != 1
            or prior_decision.get("exact_audit_semantic_sha256")
            != _exact_summary.get("semantic_sha256")
            or int(
                prior_decision.get("candidate_restarts_from_horizon_anchor", -1)
            )
            != int(prior_selected == "candidate")
            or (
                args.reference_backend == "exact" and prior_selected != "exact"
            )
            or (
                args.reference_backend == "candidate"
                and prior_selected != "candidate"
            )
            or not selection_authorized
            or prior_decision.get("phase")
            not in {
                "selected_before_scientific_continuation",
                "selected_family_complete",
            }
            or int(prior_decision.get("selected_family_complete", -1))
            not in {0, 1}
        ):
            raise ArtifactCompatibilityError("evaluation backend decision binding changed")
        selected = prior_selected
    else:
        if opening_path.is_file():
            opening = _load_semantic(
                opening_path, "evaluation backend selection opening"
            )
            selected = _verify_recovery_backend_opening(
                run_dir,
                args,
                opening=opening,
                exact_root=exact_root,
                total_shards=len(sequence) // 56,
                expected_schema=RECOVERY_RUN_SCHEMA
                + "-evaluation-backend-selection-opening",
                expected_audit_sha256=_exact_summary["semantic_sha256"],
                expected_horizon=horizon,
            )
        else:
            opening_projection, resource_snapshot = _recovery_selection_projection(
                run_dir,
                args,
                observed_shard_seconds=exact_complete_seconds,
                remaining_shards=remaining,
                projected_additional_bytes=projected_exact_bytes,
            )
            opening = _semantic(
                {
                    "schema": RECOVERY_RUN_SCHEMA
                    + "-evaluation-backend-selection-opening",
                    "schema_version": 1,
                    "horizon": horizon,
                    "requested": args.reference_backend,
                    "selected": _candidate_backend_selected(
                        opening_projection, args.reference_backend
                    ),
                    "exact_audit_semantic_sha256": _exact_summary[
                        "semantic_sha256"
                    ],
                    "exact_projection": opening_projection,
                    "resource_ledger_semantic_sha256": ledger["semantic_sha256"],
                    "projection_inputs": _recovery_projection_input_commitments(
                        run_dir,
                        exact_root=exact_root,
                        remaining_shards=remaining,
                        projection=opening_projection,
                        resource_snapshot=resource_snapshot,
                        test_only=bool(args.test_only),
                    ),
                }
            )
            atomic_write_json(opening_path, opening)
        try:
            selected = _candidate_backend_selected(
                opening["exact_projection"], args.reference_backend
            )
        except RolloutCLIError as exc:
            if (
                args.reference_backend == "exact"
                and exc.failure_code
                == "exact_core_resource_blocked_by_explicit_backend_choice"
            ):
                return {
                    "performed": 0,
                    "reason": f"{horizon}_explicit_exact_resource_deferred",
                    "exact_audit_shard_committed": 1,
                    "exact_projection": opening["exact_projection"],
                }
            raise
        atomic_write_json(
            decision_path,
            _semantic(
                {
                    "schema": RECOVERY_RUN_SCHEMA + "-evaluation-backend-decision",
                    "schema_version": 1,
                    "horizon": horizon,
                    "requested": args.reference_backend,
                    "selected": selected,
                    "phase": "selected_before_scientific_continuation",
                    "exact_audit_semantic_sha256": _exact_summary[
                        "semantic_sha256"
                    ],
                    "exact_projection": opening["exact_projection"],
                    "selection_opening_sha256": opening["semantic_sha256"],
                    "switch_event_sha256": None,
                    "exact_audit_shard_committed": 1,
                    "candidate_restarts_from_horizon_anchor": int(selected == "candidate"),
                    "selected_family_complete": 0,
                }
            ),
        )
    selected_attempt = _run_recovery_optional_backend(
        run_dir,
        args,
        backend=selected,
        sequence=sequence,
        exact_audit_only=False,
        **common,
    )
    if (
        selected_attempt is None
        and selected == "exact"
        and args.reference_backend == "auto"
    ):
        selected = "candidate"
        opening = _load_semantic(opening_path, "evaluation backend selection opening")
        switch_event = _recovery_switch_event(
            run_dir,
            args,
            exact_root=exact_root,
            total_shards=len(sequence) // 56,
            exact_audit_sha256=_exact_summary["semantic_sha256"],
            selection_opening_sha256=opening["semantic_sha256"],
            schema=RECOVERY_RUN_SCHEMA + "-evaluation-backend-switch-event",
            restart_field="restart_from_horizon_anchor",
            horizon=horizon,
        )
        degraded_projection = switch_event["failed_exact_projection"]
        atomic_write_json(switch_path, switch_event)
        atomic_write_json(
            decision_path,
            _semantic(
                {
                    "schema": RECOVERY_RUN_SCHEMA + "-evaluation-backend-decision",
                    "schema_version": 1,
                    "horizon": horizon,
                    "requested": args.reference_backend,
                    "selected": selected,
                    "phase": "selected_before_scientific_continuation",
                    "exact_audit_semantic_sha256": _exact_summary[
                        "semantic_sha256"
                    ],
                    "exact_projection": opening["exact_projection"],
                    "exact_degraded_at_verified_boundary": 1,
                    "exact_degraded_projection": degraded_projection,
                    "selection_opening_sha256": opening["semantic_sha256"],
                    "switch_event_sha256": switch_event["semantic_sha256"],
                    "candidate_restarts_from_horizon_anchor": 1,
                    "selected_family_complete": 0,
                }
            ),
        )
        selected_attempt = _run_recovery_optional_backend(
            run_dir,
            args,
            backend="candidate",
            sequence=sequence,
            exact_audit_only=False,
            **common,
        )
    if selected_attempt is None:
        return {
            "performed": 0,
            "reason": f"{horizon}_selected_backend_resource_deferred",
            "selected_backend": selected,
            "partial_optional_evidence_preserved": 1,
        }
    result, summary, selected_root = selected_attempt
    opening = _load_semantic(opening_path, "evaluation backend selection opening")
    audit: dict[str, Any] | None = None
    if selected == "candidate":
        exact_record = _recovery_shard_records(exact_root)[0]
        candidate_record = _recovery_shard_records(selected_root)[0]
        audit = _exact_candidate_audit_record(
            row_keys=tuple(row[0] for row in rows),
            exact_state=_load_npz(exact_root / "shard-0000.npz")["state"],
            candidate_state=_load_npz(selected_root / "shard-0000.npz")["state"],
            exact_reference_rms=_recovery_shard_row_rms(exact_record, "reference_fraction_displacement"),
            candidate_reference_rms=_recovery_shard_row_rms(candidate_record, "reference_fraction_displacement"),
            exact_controller_rms=_recovery_shard_row_rms(exact_record, "control_fraction_displacement"),
            candidate_controller_rms=_recovery_shard_row_rms(candidate_record, "control_fraction_displacement"),
        )
        atomic_write_json(run_dir / f"exact_candidate_audit_evaluation_{horizon}.json", audit)
    artifact = _commit_recovery_evaluation_family(
        run_dir,
        horizon=horizon,
        result=result,
        backend=selected,
        audit=audit,
        family_summary=summary,
    )
    atomic_write_json(
        decision_path,
        _semantic(
            {
                "schema": RECOVERY_RUN_SCHEMA + "-evaluation-backend-decision",
                "schema_version": 1,
                "horizon": horizon,
                "requested": args.reference_backend,
                "selected": selected,
                "phase": "selected_family_complete",
                "exact_audit_semantic_sha256": _exact_summary[
                    "semantic_sha256"
                ],
                "exact_projection": opening["exact_projection"],
                "selection_opening_sha256": opening["semantic_sha256"],
                "switch_event_sha256": (
                    _load_semantic(switch_path, "evaluation backend switch event")[
                        "semantic_sha256"
                    ]
                    if switch_path.is_file()
                    else None
                ),
                "exact_audit_shard_committed": 1,
                "candidate_restarts_from_horizon_anchor": int(selected == "candidate"),
                "exact_degraded_at_verified_boundary": int(
                    prior_decision.get("exact_degraded_at_verified_boundary", 0)
                    or (
                        decision_path.is_file()
                        and int(
                            _load_semantic(
                                decision_path, "evaluation backend decision"
                            ).get("exact_degraded_at_verified_boundary", 0)
                        )
                    )
                ),
                "exact_degraded_projection": (
                    _load_semantic(
                        decision_path, "evaluation backend decision"
                    ).get("exact_degraded_projection")
                    if decision_path.is_file()
                    else None
                ),
                "selected_family_complete": 1,
            }
        ),
    )
    return {"performed": 1, **artifact}


def _verify_existing_recovery_evaluation_plan(
    run_dir: Path,
    record: Mapping[str, Any],
    *,
    selected_gain: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if (
        record.get("schema") != RECOVERY_RUN_SCHEMA + "-evaluation-plan"
        or float(record.get("selected_gain", float("nan"))) != float(selected_gain)
        or int(record.get("optional_future_fb300_unopened", 0)) != 1
    ):
        raise ArtifactCompatibilityError("evaluation plan binding changed")
    if not int(record.get("performed", 0)):
        committed_evaluation = tuple(
            run_dir.glob(
                "objective_attempts/*/fused_families/evaluation-*/*/shard-*.json"
            )
        )
        derived_evaluation = tuple(
            run_dir.glob("objective/evaluation-*/evaluation_family.json")
        )
        allowed_reasons = {
            "test_fixture_core_only",
            "fresh_evaluation_deferred_by_resource_projection",
            "fresh_evaluation_anchor_unavailable_within_budget",
        }
        short_value = record.get("short_horizon")
        honest_partial = (
            isinstance(short_value, Mapping)
            and not int(short_value.get("performed", 0))
            and record.get("reason") == short_value.get("reason")
            and int(short_value.get("partial_optional_evidence_preserved", 0)) == 1
            and record.get("full_horizon") is None
        )
        plain_defer = (
            short_value is None
            and record.get("full_horizon") is None
            and record.get("reason") in allowed_reasons
            and isinstance(record.get("projection"), Mapping)
        )
        if (
            derived_evaluation
            or not (honest_partial or plain_defer)
            or (committed_evaluation and not honest_partial)
        ):
            raise ArtifactCompatibilityError(
                "unperformed evaluation plan conflicts with committed evidence"
            )
    anchor_plan_path = run_dir / "evaluation_anchor_plan.json"
    if "anchor_plan" in record:
        anchor_plan = _load_semantic(anchor_plan_path, "evaluation anchor plan")
        if record.get("anchor_plan") != anchor_plan:
            raise ArtifactCompatibilityError("evaluation anchor plan binding changed")
    horizons: dict[str, Mapping[str, Any]] = {}
    historical = _recovery_anchors(run_dir)
    evaluation_anchors, _anchor_record = _recovery_forward_evaluation(
        run_dir, args, historical_anchors=historical
    )
    for key, name in (("short_horizon", "short"), ("full_horizon", "full")):
        value = record.get(key)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ArtifactCompatibilityError("evaluation horizon binding changed")
        horizons[name] = value
        if int(value.get("performed", 0)):
            family = _verify_recovery_evaluation_family_artifact(
                run_dir,
                args,
                horizon=name,
                selected_gain=selected_gain,
                anchor=evaluation_anchors[
                    SHORT_ANCHOR if name == "short" else FULL_ANCHOR
                ],
            )
            bound_value = dict(value)
            bound_value.pop("performed", None)
            if bound_value != family:
                raise ArtifactCompatibilityError(
                    f"evaluation {name} family binding changed"
                )
            backend_decision = _load_semantic(
                run_dir / f"evaluation_{name}_backend_decision.json",
                f"evaluation {name} backend decision",
            )
            if (
                backend_decision.get("selected") != value.get("backend")
                or int(backend_decision.get("selected_family_complete", 0)) != 1
            ):
                raise ArtifactCompatibilityError(
                    f"evaluation {name} backend completion changed"
                )
        else:
            family_root = run_dir / f"objective/evaluation-{name}"
            committed = tuple(
                run_dir.glob(
                    f"objective_attempts/*/fused_families/evaluation-{name}/{name}/shard-*.json"
                )
            )
            honest_partial = (
                int(value.get("partial_optional_evidence_preserved", 0)) == 1
                and str(value.get("reason", "")).endswith(
                    "selected_backend_resource_deferred"
                )
            )
            if honest_partial:
                partial_rows = (
                    (f"evaluation-{name}-zero", "zero", name, None),
                    (
                        f"evaluation-{name}-learned",
                        "learned",
                        name,
                        selected_gain,
                    ),
                    (
                        f"evaluation-{name}-source-informed",
                        "oracle",
                        name,
                        None,
                    ),
                )
                verified_prefixes = [
                    prefix
                    for candidate_backend in ("exact", "candidate")
                    if (
                        prefix := _verify_recovery_partial_family(
                            run_dir,
                            args,
                            backend=candidate_backend,
                            family_name=f"evaluation-{name}",
                            segment_name=name,
                            rows=partial_rows,
                            canonical_role="evaluation",
                            stream_role="frequency1-objective-first-evaluation-v1",
                            anchor=evaluation_anchors[
                                SHORT_ANCHOR if name == "short" else FULL_ANCHOR
                            ],
                            sequence=_reverse_sequence(
                                SHORT_ANCHOR if name == "short" else FULL_ANCHOR
                            ),
                        )
                    )
                    is not None
                ]
                if not verified_prefixes:
                    raise ArtifactCompatibilityError(
                        f"partial evaluation {name} claim lacks a verified prefix"
                    )
            if family_root.exists() or (committed and not honest_partial):
                raise ArtifactCompatibilityError(
                    f"unperformed evaluation {name} suppresses committed evidence"
                )
    if int(record.get("performed", 0)):
        short = horizons.get("short", {})
        full = horizons.get("full", {})
        if not int(short.get("performed", 0)):
            raise ArtifactCompatibilityError("evaluation short family is incomplete")
        terminal = full if int(full.get("performed", 0)) else short
        decision = terminal.get("decision")
        action = terminal.get("recommended_next_action")
        if (
            int(full.get("performed", 0))
            and "dynamic_signal" in str(short.get("decision", ""))
            and str(full.get("decision", "")).endswith(
                ("control_dynamically_negligible", "rollout_direction_not_useful")
            )
        ):
            decision = "learned_short_only_dynamic_signal"
            action = (
                "localize accumulation and on-policy drift from the saved short/full "
                "quarter states before scaling"
            )
        if record.get("decision") != decision or record.get(
            "recommended_next_action"
        ) != action:
            raise ArtifactCompatibilityError("evaluation terminal route changed")
    return dict(record)


def _recovery_evaluation_plan(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    backend: str,
    selected_gain: float,
    historical_anchors: Mapping[int, np.ndarray],
) -> dict[str, Any]:
    path = run_dir / "evaluation_plan.json"
    if path.is_file():
        return _verify_existing_recovery_evaluation_plan(
            run_dir,
            _load_semantic(path, "recovery evaluation plan"),
            selected_gain=selected_gain,
            args=args,
        )
    # The outer check only reserves the mandatory exact audit. Horizon-local
    # selection then prices exact continuation and can restart candidate; an
    # all-exact forecast must not suppress an affordable candidate horizon.
    projection = _recovery_optional_projection(
        run_dir, args, backend="exact", additional_shards=1
    )
    if args.test_only or not int(projection["passed"]):
        record = _semantic(
            {
                "schema": RECOVERY_RUN_SCHEMA + "-evaluation-plan",
                "schema_version": 1,
                "performed": 0,
                "reason": "test_fixture_core_only" if args.test_only else "fresh_evaluation_deferred_by_resource_projection",
                "projection": projection,
                "selected_gain": selected_gain,
                "source_informed_endpoint_did_not_gate_this_decision": 1,
                "optional_future_fb300_unopened": 1,
            }
        )
        atomic_write_json(path, record)
        return record
    anchors, anchor_plan = _recovery_forward_evaluation(
        run_dir, args, historical_anchors=historical_anchors
    )
    short = _recovery_run_evaluation_horizon(
        run_dir,
        args,
        horizon="short",
        anchor=anchors[SHORT_ANCHOR],
        selected_gain=selected_gain,
    )
    if not int(short.get("performed", 0)):
        record = _semantic(
            {
                "schema": RECOVERY_RUN_SCHEMA + "-evaluation-plan",
                "schema_version": 1,
                "performed": 0,
                "reason": short["reason"],
                "anchor_plan": anchor_plan,
                "short_horizon": short,
                "selected_gain": selected_gain,
                "optional_future_fb300_unopened": 1,
            }
        )
        atomic_write_json(path, record)
        return record
    # A full horizon is attempted only when the conservative cost of its own
    # exact audit plus selected continuation fits. Runtime degradation is
    # checked again at every verified shard boundary.
    # Opening full requires its exact audit plus an affordable selected
    # continuation.  Do not pre-veto it with an all-exact 65-shard forecast:
    # the horizon-local policy may legitimately choose candidate after shard 0.
    full_projection = _recovery_optional_projection(
        run_dir, args, backend="exact", additional_shards=1
    )
    full: dict[str, Any] = {
        "performed": 0,
        "reason": "full_evaluation_deferred_by_resource_projection",
        "projection": full_projection,
    }
    if int(full_projection["passed"]):
        full = _recovery_run_evaluation_horizon(
            run_dir,
            args,
            horizon="full",
            anchor=anchors[FULL_ANCHOR],
            selected_gain=selected_gain,
        )
    terminal_family = full if int(full.get("performed", 0)) else short
    terminal_decision = terminal_family["decision"]
    terminal_next_action = terminal_family["recommended_next_action"]
    if (
        int(full.get("performed", 0))
        and "dynamic_signal" in str(short.get("decision", ""))
        and str(full.get("decision", "")).endswith(
            ("control_dynamically_negligible", "rollout_direction_not_useful")
        )
    ):
        terminal_decision = "learned_short_only_dynamic_signal"
        terminal_next_action = (
            "localize accumulation and on-policy drift from the saved short/full "
            "quarter states before scaling"
        )
    record = _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-evaluation-plan",
            "schema_version": 1,
            "performed": 1,
            "anchor_mode": anchor_plan["mode"],
            "canonical_path_id": _recovery_path_ids(run_dir)["evaluation"],
            "selected_gain": selected_gain,
            "short_horizon": short,
            "full_horizon": full,
            "short_decision": short["decision"],
            "decision": terminal_decision,
            "recommended_next_action": terminal_next_action,
            "source_informed_endpoint_is_diagnostic": 1,
            "optional_future_fb300_unopened": 1,
            "passed": 1,
        }
    )
    atomic_write_json(path, record)
    return record


def _recovery_partial_evidence_summary(run_dir: Path) -> dict[str, Any]:
    """Describe the latest committed reverse shard without granting endpoint authority."""

    committed_shards: list[tuple[Path, dict[str, Any]]] = []
    roots = sorted(
        path
        for path in (run_dir / "objective_attempts").glob(
            "*/fused_families/*/*"
        )
        if path.is_dir()
    )
    for root in roots:
        paths = [
            path
            for path in sorted(root.glob("shard-*.json"))
            if not path.name.endswith(".failure.json")
        ]
        records = _recovery_shard_records(root)
        if len(paths) != len(records):
            raise ArtifactCompatibilityError(
                "recovery committed shard enumeration changed"
            )
        committed_shards.extend(zip(paths, records))
    committed_shards.sort(
        key=lambda item: (item[0].stat().st_mtime_ns, item[0].as_posix())
    )
    if not committed_shards:
        return {
            "committed_shard_count": 0,
            "partial_outer_steps": 0,
            "numerically_interpretable_partial_evidence": 0,
            "executed_shard_numerics_valid": "not_evaluated",
        }
    path, record = committed_shards[-1]
    state_path = path.with_suffix(".npz")
    state = np.asarray(_load_npz(state_path).get("state"), dtype=np.float64)
    diagnostics = record.get("diagnostics", {})
    reference = (
        diagnostics.get("reference", {})
        if isinstance(diagnostics, Mapping)
        else {}
    )
    backend = path.relative_to(run_dir).parts[1]
    per_row = reference.get("per_row", []) if isinstance(reference, Mapping) else []
    forbidden = (
        reference.get("forbidden_counts", {})
        if isinstance(reference, Mapping)
        else {}
    )
    rows_valid = bool(
        isinstance(per_row, list)
        and per_row
        and all(
            isinstance(row, Mapping)
            and isinstance(row.get("transition_count"), int)
            and isinstance(row.get("active_count"), int)
            and isinstance(row.get("invalid_count"), int)
            and int(row["invalid_count"]) == 0
            and (
                (
                    isinstance(row.get("certified_count"), int)
                    and isinstance(row.get("fallback_count"), int)
                    and isinstance(row.get("unauthorized_count"), int)
                    and int(row["transition_count"]) == int(row["active_count"])
                    and int(row["certified_count"]) == int(row["active_count"])
                    and int(row["fallback_count"]) >= 0
                    and int(row["unauthorized_count"]) == 0
                    and float(row.get("certificate_fraction", 0.0)) == 1.0
                )
                if backend == "exact"
                else (
                    isinstance(row.get("structural_noop_count"), int)
                    and isinstance(row.get("approximation_count"), int)
                    and int(row["transition_count"])
                    == int(row["active_count"])
                    + int(row["structural_noop_count"])
                    and int(row["approximation_count"]) == int(row["active_count"])
                    and row.get("certificate_fraction") == "not_applicable"
                )
            )
            for row in per_row
        )
    )
    state_valid = bool(
        state.ndim == 2
        and state.shape[1:] == (784,)
        and np.isfinite(state).all()
        and np.all(state >= 0.0)
        and float(np.max(np.abs(np.sum(state, axis=1) - 1.0)))
        <= MAXIMUM_MASS_ERROR
    )
    numerical_valid = bool(
        state_valid
        and rows_valid
        and isinstance(forbidden, Mapping)
        and forbidden
        and all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value == 0
            for value in forbidden.values()
        )
        and isinstance(diagnostics, Mapping)
        and math.isfinite(
            float(diagnostics.get("maximum_mass_error", float("nan")))
        )
        and float(diagnostics.get("maximum_mass_error", float("inf")))
        <= MAXIMUM_MASS_ERROR
    )
    sequence = record.get("execution_plan", {}).get("sequence", [])
    partial_steps = len(sequence) // 7 if isinstance(sequence, list) else 0
    return {
        "committed_shard_count": len(committed_shards),
        "latest_shard_path": path.relative_to(run_dir).as_posix(),
        "latest_state_path": state_path.relative_to(run_dir).as_posix(),
        "backend": backend,
        "partial_outer_steps": partial_steps,
        "numerically_interpretable_partial_evidence": int(numerical_valid),
        "executed_shard_numerics_valid": int(numerical_valid),
    }


def _refresh_recovery_failure_accounting(run_dir: Path) -> dict[str, Any]:
    """Seal observed failure usage and a committed exact-audit placeholder state."""

    previous = (
        _load_semantic(run_dir / "resource_ledger.json", "resource ledger")
        if (run_dir / "resource_ledger.json").is_file()
        else {}
    )
    usage = _recovery_observed_resource(run_dir)
    maximum_main = float(previous.get("maximum_main_seconds", MAXIMUM_MAIN_WALL_SECONDS))
    config = (
        _load_semantic(run_dir / "scientific_config.json", "scientific config")
        if (run_dir / "scientific_config.json").is_file()
        else {}
    )
    projection = _recovery_resource_projection(
        active_seconds=usage["active_seconds"],
        wasted_active_seconds=usage["wasted_active_seconds"],
        observed_shard_seconds=(),
        remaining_shards=0,
        maximum_main_seconds=maximum_main,
        persisted_bytes=usage["persisted_bytes"],
        projected_additional_bytes=0,
        peak_memory_fraction=usage["peak_memory_fraction"],
        test_only=bool(config.get("test_only", 0)),
    )
    ledger = _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-resource-ledger",
            "schema_version": 1,
            "backend": "terminal_failure",
            **usage,
            "projection": projection,
            "maximum_main_seconds": maximum_main,
            "maximum_persisted_bytes": MAXIMUM_PERSISTED_BYTES,
            "report_reserve_seconds": RECOVERY_REPORT_RESERVE_SECONDS,
            "idle_wall_time_charged": 0,
            "final_observed_failure_usage": 1,
            "passed": int(projection["passed"]),
        }
    )
    atomic_write_json(run_dir / "resource_ledger.json", ledger)

    decision_path = run_dir / "backend_decision.json"
    if decision_path.is_file():
        decision = _load_semantic(decision_path, "backend decision")
        exact_shards = sorted(
            run_dir.glob(
                "objective_attempts/exact/fused_families/development-core/short/"
                "shard-*.json"
            )
        )
        if exact_shards and decision.get("selected") == "not_attempted":
            updated = {
                key: value
                for key, value in decision.items()
                if key != "semantic_sha256"
            }
            updated.update(
                {
                    "phase": "exact_audit_committed_health_invalid_before_selection",
                    "attempted_backend": "exact",
                    "exact_audit_shard_committed": 1,
                    "exact_audit_health_valid": 0,
                    "selected_family_complete": 0,
                }
            )
            atomic_write_json(decision_path, _semantic(updated))
    return ledger


def _write_recovery_report(run_dir: Path) -> None:
    manifest = (
        _load_semantic(run_dir / "run_manifest.json", "run manifest")
        if (run_dir / "run_manifest.json").is_file()
        else {}
    )
    core = _load_semantic(run_dir / "core_objective.json", "core objective") if (
        run_dir / "core_objective.json"
    ).is_file() else {}
    backend = _load_semantic(run_dir / "backend_decision.json", "backend decision") if (
        run_dir / "backend_decision.json"
    ).is_file() else {}
    gain = _load_semantic(run_dir / "gain_expansion.json", "gain expansion") if (
        run_dir / "gain_expansion.json"
    ).is_file() else {}
    evaluation = _load_semantic(run_dir / "evaluation_plan.json", "evaluation plan") if (
        run_dir / "evaluation_plan.json"
    ).is_file() else {}
    ledger = _load_semantic(run_dir / "resource_ledger.json", "resource ledger") if (
        run_dir / "resource_ledger.json"
    ).is_file() else {}
    failure = _load_semantic(run_dir / "failure.json", "recovery failure") if (
        run_dir / "failure.json"
    ).is_file() else {}
    terminal = _load_semantic(run_dir / "terminal_outcome.json", "terminal outcome") if (
        run_dir / "terminal_outcome.json"
    ).is_file() else {}
    decision = str(
        terminal.get("decision")
        or failure.get("decision")
        or core.get("decision")
        or "objective_first_recovery_incomplete"
    )
    completed = int(core.get("completed_128_step_three_row_family", 0))
    backend_name = backend.get("selected", core.get("backend", "not selected"))
    partial = _recovery_partial_evidence_summary(run_dir)
    partial_interpretable = int(
        partial.get("numerically_interpretable_partial_evidence", 0)
    )
    attempted_backend = backend.get("attempted_backend", partial.get("backend"))
    rows = core.get("row_results", []) if isinstance(core.get("row_results"), list) else []
    metrics_lines = [
        f"- `{row['row_key']}` final squared L2 to mixed target: "
        f"{row['metrics_to_mixed_target']['final']['squared_l2_error']:.9g}."
        for row in rows
        if isinstance(row, Mapping)
    ]
    mechanism_lines = [
        f"- `{row['row_key']}` score/logistic/reference/control RMS and control/reference ratio: "
        f"{row.get('mechanism_diagnostics', {}).get('score_rms', 0.0):.6g}/"
        f"{row.get('mechanism_diagnostics', {}).get('logistic_shift_rms', 0.0):.6g}/"
        f"{row.get('mechanism_diagnostics', {}).get('reference_fraction_displacement_rms', 0.0):.6g}/"
        f"{row.get('mechanism_diagnostics', {}).get('control_fraction_displacement_rms', 0.0):.6g}/"
        f"{row.get('mechanism_diagnostics', {}).get('control_reference_displacement_ratio', 0.0):.6g}."
        for row in rows
        if isinstance(row, Mapping)
    ]
    audit = _load_semantic(run_dir / "exact_candidate_audit.json", "exact candidate audit") if (
        run_dir / "exact_candidate_audit.json"
    ).is_file() else {}
    audit_lines = [
        f"- Candidate local audit `{row['row_key']}`: L1={row['l1_state_discrepancy']:.6g}, squared-L2={row['squared_l2_state_discrepancy']:.6g}, max-abs={row['maximum_absolute_state_discrepancy']:.6g}, TV={row['total_variation_state_discrepancy']:.6g}, correlation={row['centered_correlation']:.9g}."
        for row in audit.get("row_metrics", [])
        if isinstance(row, Mapping)
    ]
    evaluation_lines: list[str] = []
    for horizon_key in ("short_horizon", "full_horizon"):
        horizon = evaluation.get(horizon_key, {})
        if not isinstance(horizon, Mapping):
            continue
        if not int(horizon.get("performed", 0)):
            evaluation_lines.append(
                f"- `{horizon_key}` not performed: `{horizon.get('reason', 'not opened')}`."
            )
            continue
        risks = horizon.get("final_squared_l2", {})
        evaluation_lines.append(
            f"- `{horizon_key}` backend `{horizon.get('backend')}`, decision `{horizon.get('decision')}`; "
            f"zero/learned/source squared-L2={risks.get('zero')}/{risks.get('learned')}/{risks.get('source_informed')}; "
            f"learned-minus-zero improvement={horizon.get('learned_minus_zero_risk_improvement')}; "
            f"raw `{horizon.get('raw_states', {}).get('path', 'missing')}`, contact `{horizon.get('contact_sheet', 'missing')}`, "
            f"candidate guard={horizon.get('exact_candidate_claim_guard', 'not applicable')}."
        )
    objective_lines = (
        [
            "- Rows: zero, learned gain 1.0, source-informed target-fraction diagnostic.",
            "- All rows share canonical development path identity and random bits; variant/backend are absent from the RNG key.",
            "- Raw float64 quarter states: `objective/development-core-short/selected_states.npz`.",
            "- Fixed-scale objective contact sheet: `images/objective_core_contact_sheet.png`.",
            *(metrics_lines or ["- No complete endpoint metrics are available yet."]),
            *(mechanism_lines or ["- No completed mechanism telemetry is available yet."]),
        ]
        if completed
        else [
            "- The mandatory three-row reverse family is incomplete.",
            "- No core endpoint, paired-effect, or contact-sheet claim is made.",
            *(
                [
                    f"- A {partial.get('partial_outer_steps', 0)}-step three-row "
                    f"`{attempted_backend}` audit shard was committed and is numerically "
                    "interpretable partial objective evidence only.",
                    f"- Raw committed partial state: `{partial.get('latest_state_path')}`; "
                    "readable promoted states/images are under `failure_artifacts/`.",
                ]
                if partial_interpretable
                else [
                    "- Any readable last-valid states/images are under `failure_artifacts/` and are failure evidence only."
                ]
            ),
        ]
    )
    report = [
        "# Frequency-one objective-first rollout recovery",
        "",
        "Primary mode: **exploratory engineering immediately followed by an objective-bearing experiment**.",
        "",
        "## Decision",
        "",
        f"`{decision}`",
        "",
        "The decision question is whether the frozen frequency-one controller at predeclared gain 1.0 changes a numerically interpretable 128-step reverse suffix relative to paired zero control, and what the source-informed target-fraction diagnostic reveals about composition.",
        "",
        "## Objective artifact",
        "",
        f"- Completed mandatory 128-step three-row family: {completed}.",
        f"- Selected reference backend: `{backend_name}`; attempted backend: `{attempted_backend or 'none'}`.",
        *objective_lines,
        "",
        "## Controls and backend health",
        "",
        "- The analytic target-fraction identity is the only controller implementation/integrity gate.",
        "- The MNIST source-informed endpoint is descriptive and never blocks artifact creation or evaluation.",
        f"- Exact first-shard audit committed: {backend.get('exact_audit_shard_committed', 0)}.",
        f"- Exact resource miss was terminal: {backend.get('exact_resource_failure_terminal', 0)}.",
        f"- Accumulated active/wasted seconds: {ledger.get('active_seconds', 'unavailable')}/{ledger.get('wasted_active_seconds', 'unavailable')}; idle time charged: {ledger.get('idle_wall_time_charged', 0)}.",
        f"- Persisted bytes: {ledger.get('persisted_bytes', 'unavailable')} / {ledger.get('maximum_persisted_bytes', MAXIMUM_PERSISTED_BYTES)}; peak CUDA bytes/fraction: {ledger.get('peak_memory_bytes', 'unavailable')}/{ledger.get('peak_memory_fraction', 'unavailable')}.",
        "- Candidate trajectories, if selected, are explicitly approximate and do not claim correct-rounding certification or Arb fallback.",
        *(audit_lines or ["- No candidate backend local discrepancy audit was required."]),
        f"- Candidate learned paired-contrast relative error: {audit.get('paired_contrasts', {}).get('learned_minus_zero', {}).get('relative_error', 'not applicable')}; source-informed paired-contrast relative error: {audit.get('paired_contrasts', {}).get('source_informed_minus_zero', {}).get('relative_error', 'not applicable')}.",
        f"- Endpoint effect-scale audit guard: {core.get('exact_candidate_claim_guard', 'not applicable')}.",
        "",
        "## Optional phases",
        "",
        f"- Gain expansion performed: {gain.get('performed', 0)}; selected gain: {gain.get('selected_gain', 1.0)}; mode: `{gain.get('selection_mode', 'not run')}`.",
        f"- Evaluation performed: {evaluation.get('performed', 0)}; reason/mode: `{evaluation.get('reason', evaluation.get('anchor_mode', 'not run'))}`.",
        f"- Evaluation short decision: `{evaluation.get('short_decision', 'not run')}`; terminal evaluation decision: `{evaluation.get('decision', 'not run')}`.",
        *evaluation_lines,
        f"- Prior failure record retained but superseded by a successful terminal: {int(bool(failure) and int(terminal.get('successful_terminal', 0)) == 1)}.",
        "- FB300 replication remains unopened.",
        "",
        "## Claim boundary",
        "",
        "This run establishes only the recorded one-image exploratory dynamics under its exact or candidate backend. It does not establish a validation pass, a general generator, prior-start sampling, multi-image generalization, or Eulerian convergence. Passing a local candidate audit does not prove full-horizon exact equivalence.",
        "",
        "## Next action",
        "",
        str(terminal.get("recommended_next_action", evaluation.get("recommended_next_action", core.get("recommended_next_action", failure.get("recommended_next_action", "resume the same recovery until the mandatory core is complete"))))),
        "",
    ]
    _atomic_text(run_dir / "REPORT.md", "\n".join(report))
    handoff = [
        "# Frequency-one objective-first recovery: research handoff",
        "",
        f"Date: {_now()}",
        f"Source revision: `{manifest.get('source_fingerprint', 'unavailable before failed initialization')}`",
        "Handoff author: Codex objective-first recovery workflow",
        "",
        "## 1. Program objective",
        "",
        "Build or decisively falsify a DDPM-like MNIST generator based on the Eulerian/Jacobi approximation. The nearest artifact is a paired one-image reverse suffix visible as raw states and images.",
        "",
        "## 2. Current milestone and distance to goal",
        "",
        f"Mandatory 128-step family complete: {completed}. Proxy-only patches since the last objective-bearing experiment: {0 if (completed or partial_interpretable) else 'at least 2'}.",
        "",
        "## 3. Strategy review",
        "",
        "Strategy status: continue only according to the saved outcome branch. Exactness is a fixed-case audit for candidate exploration, not a veto on observing the assembled mechanism.",
        "",
        "## 4. Research mode and evidence roles",
        "",
        "Exploratory development uses FB100. Evaluation uses distinct FB200 random bits and is labeled fresh-forward or historical-anchor. FB300 remains unopened.",
        "",
        "## 5. Exact result of the latest run",
        "",
        f"Terminal/current result: `{decision}`; backend `{backend_name}`; mandatory core complete={completed}.",
        "",
        "### This result establishes",
        "",
        (
            f"A numerically interpretable {partial.get('partial_outer_steps', 0)}-step "
            "three-row exact-audit shard was committed; it is partial objective evidence, "
            "not a 128-step endpoint."
            if partial_interpretable and not completed
            else "Only the paired one-image endpoint and numerical/backend health committed in this child run."
        ),
        "",
        "### This result does not establish",
        "",
        "Validation success, exact full-horizon equivalence for candidate runs, general generation, prior matching, or multi-image utility.",
        "",
        "## 6. Confirmed facts, current inferences, and open hypotheses",
        "",
        (
            "Confirmed partial facts are in `backend_decision.json`, the committed exact "
            "shard NPZ/JSON, and `failure_artifacts/`; no `core_objective.json` exists. "
            "Open hypotheses include orientation/composition failure, dynamically weak or "
            "adverse learned control, on-policy drift, inadequate representation, proxy "
            "misalignment, prior mismatch, approximation dominance, and failure of the strategy."
            if not completed
            else "Confirmed facts are in `core_objective.json`, `backend_decision.json`, and raw NPZs. Open hypotheses include orientation/composition failure, dynamically weak or adverse learned control, on-policy drift, inadequate representation, proxy misalignment, prior mismatch, approximation dominance, and failure of the strategy."
        ),
        "",
        "## 7. Decision the next patch must resolve",
        "",
        str(terminal.get("recommended_next_action", evaluation.get("recommended_next_action", core.get("recommended_next_action", "complete the mandatory core")))),
        "",
        "## 8. Candidate actions and value of information",
        "",
        "Use the terminal outcome routing in the stored implementation plan. Do not substitute another proxy-only gate for a saved objective result.",
        "",
        "## 9. Recommended next patch",
        "",
        str(terminal.get("recommended_next_action", evaluation.get("recommended_next_action", core.get("recommended_next_action", "resume this same patch")))),
        "",
        "## 10. Gates and claim boundaries",
        "",
        "Analytic identity and numerical conservation are execution/integrity gates. Candidate discrepancy thresholds are diagnostic. Endpoint improvement is exploratory, not confirmatory.",
        "",
        "## 11. Outcome-to-action table",
        "",
        "Outcome routes: analytic failure repairs composition; approximation-dominance extends the exact audit; short-only signal localizes accumulation; full signal advances to small multi-image M=2; negligible/adverse signal pivots learner/controller.",
        "",
        "## 12. Constraints",
        "",
        "Immutable predecessor bytes, input identity, RNG pairing, numerical validity, and protected evidence are integrity constraints. Architecture, gain, backend, schedule, controller, and strategy remain revisable.",
        "",
        "## 13. Resource budget and stop rule",
        "",
        "Active main cap 21,600 seconds including a 300-second report reserve; 2 GiB storage; 80% GPU memory. A candidate-core resource block is incomplete and must not be called success.",
        "",
        "## 14. Alternative and pivot plan",
        "",
        "If the learned controller is negligible or adverse, compare rollout-trained/global alternatives. If approximation dominates, improve/audit the backend before subtle utility claims.",
        "",
        "## 15. Evidence map",
        "",
        (
            "`predecessor_binding.json` is provenance; the committed exact shard under "
            "`objective_attempts/` and promoted `failure_artifacts/` are partial task "
            "evidence; `resource_ledger.json` records resource use. No core objective or "
            "candidate audit artifact was committed."
            if not completed
            else "`predecessor_binding.json` is provenance; `core_objective.json` is the objective result; `objective/**.npz` and `images/` are task evidence; `exact_candidate_audit*.json` bounds approximation locally; `resource_ledger.json` records resource use."
        ),
        "",
        "## 16. Deliberate omissions",
        "",
        "No FB300 replication, prior-start sampling, multi-image data, retraining, or confirmation is included.",
        "",
        "## 17. Reproduction commands",
        "",
        "Use the single `--stage all` recovery command in `docs/jacobi_rb_frequency1_fused_rollout.md`; resume uses the same stage and verifies committed shards.",
        "",
        "## 18. Bundle-integrity audit",
        "",
        "Verify `SHA256SUMS.txt`, `artifact_manifest.json`, and `bundle_integrity_audit.json`.",
        "",
        "## 19. Exact deliverable for the receiving agent",
        "",
        "Act on the saved objective outcome; do not produce another authorization-only plan.",
        "",
    ]
    _atomic_text(run_dir / "HANDOFF.md", "\n".join(handoff))


def _objective_first_recovery(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    terminal_path = run_dir / "terminal_outcome.json"
    if terminal_path.is_file():
        terminal = _load_semantic(terminal_path, "recovery terminal outcome")
        if int(terminal.get("successful_terminal", 0)) == 1:
            _verify_artifact_manifest(run_dir)
            return terminal
    plan_source = Path(__file__).resolve().parent.parent / "plans/NEXT_PATCH_PLAN.md"
    plan_copy = run_dir / "implementation_plan.md"
    if plan_source.is_file() and not plan_copy.is_file():
        _atomic_text(plan_copy, plan_source.read_text(encoding="utf-8"))
    core_path = run_dir / "core_objective.json"
    core = (
        _verify_recovery_core_artifact(run_dir, args)
        if core_path.is_file()
        else (
            _recovery_test_core(run_dir, args)
            if args.test_only
            else _recovery_production_core(run_dir, args)
        )
    )
    backend = _load_semantic(run_dir / "backend_decision.json", "backend decision")[
        "selected"
    ]
    anchors = _recovery_anchors(run_dir)
    gain = _recovery_gain_expansion(
        run_dir, args, anchor=anchors[SHORT_ANCHOR], backend=str(backend)
    )
    evaluation = _recovery_evaluation_plan(
        run_dir,
        args,
        backend=str(backend),
        selected_gain=float(gain["selected_gain"]),
        historical_anchors=anchors,
    )
    # Seal actual execution/storage/memory usage after the last attempted
    # objective shard. Optional evidence can be deferred, but the successful
    # terminal may never conceal a hard-cap overrun.
    final_ledger = _write_recovery_ledger(
        run_dir,
        args,
        backend=str(backend),
        remaining_shards=0,
        observed_shard_seconds=(),
        projected_additional_bytes=0,
    )
    if not int(final_ledger["passed"]):
        raise RolloutCLIError(
            "final actual recovery resource use exceeds the frozen hard cap",
            failure_domain="resource_budget",
            failure_code="objective_resource_budget_exhausted_after_verified_evidence",
        )
    terminal_decision = str(
        evaluation.get("decision")
        if int(evaluation.get("performed", 0))
        else core["decision"]
    )
    terminal_next_action = str(
        evaluation.get("recommended_next_action")
        if int(evaluation.get("performed", 0))
        else core["recommended_next_action"]
    )
    if not int(evaluation.get("performed", 0)) and terminal_decision == "learned_short_dynamic_signal":
        terminal_decision = "core_objective_complete_fresh_evaluation_deferred"
        terminal_next_action = (
            "resume this same run only if a fresh evaluation fits the frozen hard budget; "
            "otherwise plan the next objective-bearing comparison from the saved core"
        )
    terminal = _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-terminal-outcome",
            "schema_version": 1,
            "decision": terminal_decision,
            "core_decision": core["decision"],
            "backend": backend,
            "completed_128_step_three_row_family": 1,
            "objective_bearing_artifacts_committed": 1,
            "gain_expansion_performed": int(gain["performed"]),
            "evaluation_performed": int(evaluation["performed"]),
            "evaluation_decision": evaluation.get("decision"),
            "optional_future_fb300_unopened": 1,
            "scientific_evidence_complete": 1,
            "successful_terminal": 1,
            "recommended_next_action": terminal_next_action,
        }
    )
    atomic_write_json(run_dir / "terminal_outcome.json", terminal)
    atomic_write_json(run_dir / "exploratory_decision.json", terminal)
    _status(
        run_dir,
        state="complete",
        stage="report",
        decision=terminal_decision,
        scientific_evidence_complete=1,
    )
    _write_recovery_report(run_dir)
    _finalize_artifacts(run_dir)
    return terminal


def _failure_recovery(run_dir: Path, stage: str, exc: BaseException) -> None:
    failure_domain = str(getattr(exc, "failure_domain", "execution_integrity"))
    failure_code = str(
        getattr(exc, "failure_code", "rollout_implementation_contract_invalid")
    )
    core_complete = (run_dir / "core_objective.json").is_file()
    if not isinstance(exc, ArtifactCompatibilityError):
        # Post-processing (audit/render/objective JSON) may fail after a
        # numerically healthy family returned. Promote the latest verified
        # committed state so that this failure cannot hide the task artifact.
        candidates = sorted(
            run_dir.glob("objective_attempts/*/fused_families/*/*"),
            key=lambda item: item.stat().st_mtime if item.is_dir() else -1.0,
            reverse=True,
        )
        for shard_root in candidates:
            if not shard_root.is_dir() or not any(shard_root.glob("shard-*.json")):
                continue
            try:
                records = _recovery_shard_records(shard_root)
                state = np.asarray(
                    _load_npz(shard_root / f"shard-{len(records)-1:04d}.npz")["state"],
                    dtype=np.float64,
                )
                _promote_recovery_failure_states(
                    run_dir,
                    shard_root=shard_root,
                    family_name=shard_root.parent.name,
                    backend=shard_root.parents[2].name,
                    initial_state=state,
                )
            except Exception:
                pass
            break
    ledger = _refresh_recovery_failure_accounting(run_dir)
    partial = _recovery_partial_evidence_summary(run_dir)
    unsuccessful_incomplete = failure_code == "candidate_core_resource_blocked"
    decision = (
        failure_code
        if failure_domain in {"numerical_integrity", "resource_budget"}
        or failure_code == "controller_analytic_control_invalid"
        or failure_code == "synchronous_exact_reference_health_schema_invalid"
        else "rollout_implementation_contract_invalid"
    )
    recommended = (
        "optimize or repair the concrete candidate implementation and rerun this same "
        "patch; do not create another feasibility-only handoff"
        if unsuccessful_incomplete
        else (
            "repair the exact reference health schema and launch a fresh successor; "
            "do not resume this source-bound child"
            if failure_code == "synchronous_exact_reference_health_schema_invalid"
            else "repair the narrow blocker and resume this same objective-first recovery"
        )
    )
    record = _semantic(
        {
            "schema": RECOVERY_RUN_SCHEMA + "-failure",
            "schema_version": 1,
            "stage": stage,
            "decision": decision,
            "failure_domain": failure_domain,
            "failure_code": failure_code,
            "message": str(exc),
            "evaluation_status": "incomplete",
            "completed_128_step_three_row_family": int(core_complete),
            "successful_terminal": 0,
            "scientific_evidence_complete": int(core_complete),
            "executed_shard_numerics_valid": partial.get(
                "executed_shard_numerics_valid", "not_evaluated"
            ),
            "executed_shard_resource_valid": int(ledger.get("passed", 0)),
            "partial_outer_steps": int(partial.get("partial_outer_steps", 0)),
            "numerically_interpretable_partial_evidence": int(
                partial.get("numerically_interpretable_partial_evidence", 0)
            ),
            "recommended_next_action": recommended,
        }
    )
    atomic_write_json(run_dir / "failure.json", record)
    atomic_write_json(run_dir / "terminal_outcome.json", record)
    atomic_write_json(run_dir / "exploratory_decision.json", record)
    _status(
        run_dir,
        state="failed",
        stage=stage,
        decision=decision,
        message=str(exc),
        failure_domain=failure_domain,
        failure_code=failure_code,
        scientific_evidence_complete=int(core_complete),
    )
    _write_recovery_report(run_dir)
    _finalize_artifacts(run_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency1-run-dir", type=Path)
    parser.add_argument("--source-run-dir", type=Path)
    parser.add_argument("--continuation-run-dir", type=Path)
    parser.add_argument("--predecessor-run-dir", type=Path)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/experiment12_d0_jacobi_rb_frequency1_rollout"),
    )
    parser.add_argument("--run-name", default="production-frequency1-exploratory-rollout")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--require-gate", choices=REQUIRED_GATES, default="none")
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument(
        "--reference-backend", choices=("auto", "exact", "candidate"), default="auto"
    )
    parser.add_argument(
        "--development-anchor", choices=("predecessor", "fresh"), default="predecessor"
    )
    parser.add_argument("--core-learned-gain", type=float, default=RECOVERY_CORE_LEARNED_GAIN)
    parser.add_argument("--gain-sweep", choices=("auto", "off", "on"), default="auto")
    parser.add_argument(
        "--exact-audit-outer-steps", type=int, default=RECOVERY_EXACT_AUDIT_OUTER_STEPS
    )
    parser.add_argument(
        "--maximum-main-seconds", type=float, default=MAXIMUM_MAIN_WALL_SECONDS
    )
    parser.add_argument("--test-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test-short-anchor", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--test-full-anchor", type=int, default=7, help=argparse.SUPPRESS)
    parser.add_argument("--test-oracle-fail", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--test-force-candidate", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.test_only:
        if torch.device(args.device).type != "cpu":
            parser.error("test-only rollout requires --device cpu")
        if not (0 <= args.test_short_anchor < args.test_full_anchor < 512):
            parser.error("test-only anchors must satisfy 0 <= short < full < 512")
        if args.require_gate != "none":
            parser.error("test-only runs cannot satisfy a required production gate")
    elif args.stage != "report" and torch.device(args.device).type != "cuda":
        parser.error("production rollout stages except report require CUDA")
    if (
        args.continuation_run_dir is not None
        and not args.test_only
        and args.resume_run_dir is None
        and args.stage != "all"
    ):
        parser.error("a fresh fused continuation must use --stage all in one process")
    recovery_requested = args.predecessor_run_dir is not None
    if recovery_requested:
        if args.stage != "all":
            parser.error("objective-first recovery and resume must use --stage all")
        if args.require_gate != "none":
            parser.error("objective-first recovery has no authorization-only required gate")
        if args.continuation_run_dir is None:
            parser.error("objective-first recovery still requires the bound continuation carrier")
        if args.development_anchor != "predecessor":
            parser.error("this recovery freezes the verified predecessor development anchor")
        if args.core_learned_gain != RECOVERY_CORE_LEARNED_GAIN:
            parser.error("production recovery freezes the core learned gain at 1.0")
        if args.exact_audit_outer_steps != RECOVERY_EXACT_AUDIT_OUTER_STEPS:
            parser.error("production recovery freezes the exact audit at eight outer steps")
        if args.maximum_main_seconds <= RECOVERY_REPORT_RESERVE_SECONDS:
            parser.error("maximum main seconds must exceed the report reserve")
        if not args.test_only and args.maximum_main_seconds != MAXIMUM_MAIN_WALL_SECONDS:
            parser.error("production recovery freezes maximum main seconds at 21600")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.invoked_argv = list(sys.argv[1:] if argv is None else argv)
    run_dir: Path | None = None
    resumed = False
    initialization_complete = False
    current_stage = "initialize"
    try:
        # The immutable predecessor is checked before any child directory can
        # be created or resumed.  This is intentionally earlier than the
        # legacy workflow's output allocation.
        _require_recovery_output_separation(args)
        _precheck_recovery_predecessor(args)
        run_dir, resumed = _make_run_dir(args)
        _hydrate_resume_args(args, run_dir)
        _require_recovery_output_separation(args)
        _precheck_recovery_predecessor(args)
        _require_inputs(args)
        print(f"frequency-one exploratory rollout directory: {run_dir}", flush=True)
        if _recovery_mode(args):
            _initialize_recovery(run_dir, args)
        else:
            _initialize(run_dir, args)
        initialization_complete = True
        if _recovery_mode(args):
            current_stage = "objective"
            terminal = _objective_first_recovery(run_dir, args)
            print(
                f"frequency-one objective-first recovery decision: {terminal['decision']}",
                flush=True,
            )
            return 0
        if args.stage == "initialize":
            _artifact_manifest(run_dir)
            return 0 if _required_gate_passed(run_dir, args.require_gate) else 2

        functions = {
            "preflight": _preflight_stage,
            "forward": _forward_stage,
            "development": _development_stage,
            "evaluation": _evaluation_stage,
            "replication": _replication_stage,
            "report": _report_stage,
        }
        stage_executed = False
        for stage in _stage_sequence(args.stage):
            current_stage = stage
            # A terminal failed oracle must never be bypassed by ``all``.
            decision_path = run_dir / "exploratory_decision.json"
            if stage == "evaluation" and decision_path.is_file():
                decision = _load_json(decision_path)
                if decision.get("decision") == "development_oracle_control_failed":
                    break
            if not _already_complete(run_dir, stage):
                functions[stage](run_dir, args)
                stage_executed = True
        if stage_executed or not (run_dir / "artifact_manifest.json").is_file():
            _artifact_manifest(run_dir)
        else:
            _verify_artifact_manifest(run_dir)
        if not _required_gate_passed(run_dir, args.require_gate):
            return 2
        final = _load_semantic(run_dir / "exploratory_decision.json", "exploratory decision") if (
            run_dir / "exploratory_decision.json"
        ).is_file() else {}
        if final:
            print(f"frequency-one exploratory rollout decision: {final['decision']}", flush=True)
        return 0
    except (ArtifactCompatibilityError, RolloutCLIError, RuntimeError, ValueError, TypeError) as exc:
        # A rejected incompatible resume is immutable by definition.  A fresh
        # initialization failure, however, must still leave readable status,
        # failure, report, and checksum evidence.
        if (
            run_dir is not None
            and run_dir.is_dir()
            and (not resumed or initialization_complete)
            and not (resumed and isinstance(exc, ArtifactCompatibilityError))
        ):
            if _recovery_mode(args):
                _failure_recovery(run_dir, current_stage, exc)
            else:
                _failure(run_dir, current_stage, exc)
        print(f"frequency-one exploratory rollout error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
