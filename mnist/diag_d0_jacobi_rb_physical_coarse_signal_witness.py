"""Fresh exact-K=512 physical coarse-signal witness.

This workflow generates two independent certified physical panels and tests a
coarse projection of the exact Rao--Blackwell denoising conditional mean.  It
does not train a model and it never invokes a reverse sampler.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_json,
    config_fingerprint,
    configure_exact_torch_backend,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_rb_bayes_power import load_bayes_label_cache
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_cuda_controls import RigorousCudaControlError
from mnist.d0_jacobi_rb_cuda_multipath import (
    EDGES_PER_PHASE,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    SHARD_STEPS,
    run_exact_multipath_shard,
)
from mnist.d0_jacobi_rb_physical_coarse_signal import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    COARSE_CELL_COUNT,
    PANEL_A_PATH_IDS,
    PANEL_B_PATH_IDS,
    PATHS_PER_PANEL,
    PREFLIGHT_BENCHMARK_PATH_IDS,
    RESOLUTION_TARGET,
    ROOT_SEED,
    PhysicalCoarsePanel,
    analyze_cross_panel_signal,
    coarse_cell_path_means,
    evaluate_bayes_control_replay,
    frozen_path_plan,
    frozen_statistic_plan,
)
from mnist.d0_jacobi_rb_physical_coarse_signal_gate import (
    CLAIM_FLAGS,
    NO_WORK,
    evaluate_panel,
    evaluate_preflight,
    evaluate_witness,
    execution_failed_gate,
    not_evaluated_gate,
    witness_decision,
)
from mnist.d0_jacobi_rb_physical_coarse_signal_panel import (
    FORBIDDEN_COUNTS,
    OUTER_STEPS,
    PhysicalPanelError,
    run_physical_panel,
    selected_target_contribution,
    update_cell_accumulator,
)
from mnist.d0_jacobi_rb_physical_coarse_signal_provenance import (
    verify_physical_coarse_signal_parents,
)
from mnist.d0_one_image_gate import (
    ArtifactCompatibilityError as ParentArtifactCompatibilityError,
)


RUN_SCHEMA = "experiment12-d0-jacobi-rb-physical-coarse-signal-witness"
RUN_SCHEMA_VERSION = 1
EXPECTED_PANEL_TRANSITIONS = (
    PATHS_PER_PANEL * OUTER_STEPS * len(PHASE_MATCHINGS) * EDGES_PER_PHASE
)
EXPECTED_TOTAL_TRANSITIONS = 2 * EXPECTED_PANEL_TRANSITIONS
IMAGE_SHA256 = "0bb39fec59853f789fe366251cd85ed79ffbffb5a1aaa32084d2dbd2bbb4ea7d"
MIXED_TARGET_SHA256 = (
    "00ae86fb69be6d86557f15f6f8fa00f8bb3c2514f331863c9638e36d23d135c5"
)
SOURCE_IMAGE_NPZ_SHA256 = (
    "81904cde32495eb11b73cb688cc458118eb2e5578513426d2f9b881ac4665914"
)
_REGISTRY_EXCLUDED = {"artifact_registry.json", "run_status.json"}
_GATE_FILES = {
    "preflight": "coarse_signal_preflight_gate.json",
    "panel-a": "coarse_signal_panel_a_gate.json",
    "panel-b": "coarse_signal_panel_b_gate.json",
    "witness": "coarse_signal_witness_gate.json",
}


class CoarseWitnessCLIError(RuntimeError):
    """Typed execution failure committed before a required-gate exit."""

    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "workflow_execution",
        failure_code: str = "physical_coarse_signal_execution_failed",
    ) -> None:
        super().__init__(message)
        self.failure_domain = str(failure_domain)
        self.failure_code = str(failure_code)


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"cannot read JSON {target}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"JSON artifact is not an object: {target}")
    return dict(value)


def _freeze_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    if path.is_file():
        if _load_json(path) != record:
            raise ArtifactCompatibilityError(f"frozen artifact changed: {path.name}")
    else:
        atomic_write_json(path, record)
    return record


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


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
                name: np.ascontiguousarray(np.asarray(archive[name]))
                for name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise ArtifactCompatibilityError(f"cannot read NPZ {path}: {exc}") from exc


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [dict(row) for row in rows]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        fields: list[str] = []
        for row in normalized:
            for key in row:
                if key not in fields:
                    fields.append(key)
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(normalized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _source_paths() -> tuple[Path, ...]:
    import mnist.d0_jacobi_artifacts as artifacts
    import mnist.d0_jacobi_rb_bayes_power as bayes
    import mnist.d0_jacobi_rb_cuda as cuda
    import mnist.d0_jacobi_rb_cuda_certificate as cuda_certificate
    import mnist.d0_jacobi_rb_cuda_controls as cuda_controls
    import mnist.d0_jacobi_rb_cuda_fused as cuda_fused
    import mnist.d0_jacobi_rb_cuda_multipath as scheduler
    import mnist.d0_jacobi_rb_controls as reference_controls
    import mnist.d0_jacobi_rb_learnability as learnability
    import mnist.d0_jacobi_rb_spectral as spectral
    import mnist.d0_jacobi_denoising as denoising
    import mnist.d0_jacobi_rb_physical_coarse_signal as core
    import mnist.d0_jacobi_rb_physical_coarse_signal_gate as gate
    import mnist.d0_jacobi_rb_physical_coarse_signal_panel as panel
    import mnist.d0_jacobi_rb_physical_coarse_signal_provenance as provenance

    return tuple(
        sorted(
            {
                Path(__file__).resolve(),
                Path(artifacts.__file__).resolve(),
                Path(bayes.__file__).resolve(),
                Path(cuda.__file__).resolve(),
                Path(cuda_certificate.__file__).resolve(),
                Path(cuda_controls.__file__).resolve(),
                Path(cuda_fused.__file__).resolve(),
                Path(scheduler.__file__).resolve(),
                Path(reference_controls.__file__).resolve(),
                Path(learnability.__file__).resolve(),
                Path(spectral.__file__).resolve(),
                Path(denoising.__file__).resolve(),
                Path(core.__file__).resolve(),
                Path(gate.__file__).resolve(),
                Path(panel.__file__).resolve(),
                Path(provenance.__file__).resolve(),
            },
            key=lambda item: item.as_posix(),
        )
    )


def _scientific_config() -> dict[str, Any]:
    path_plan = frozen_path_plan()
    statistic = frozen_statistic_plan()
    profile = JacobiRBCudaProfile()
    record: dict[str, Any] = {
        "schema": RUN_SCHEMA + "-scientific-config",
        "schema_version": 1,
        "claim_scope": (
            "coarse conditional-mean witness for the exact K=512 certified "
            "Jacobi/Rao-Blackwell split chain and one frozen image"
        ),
        "grid_size": 28,
        "alpha": 1.0,
        "outer_steps": OUTER_STEPS,
        "tau_eff": 5.0e-5,
        "phase_matchings": list(PHASE_MATCHINGS),
        "phase_durations": list(PHASE_DURATIONS),
        "edges_per_phase": EDGES_PER_PHASE,
        "root_seed": ROOT_SEED,
        "path_plan_sha256": path_plan.fingerprint,
        "statistic_plan_sha256": statistic.fingerprint,
        # Keep the in-memory record identical to its JSON representation.
        # JacobiRBCudaProfile.to_dict() intentionally preserves compile flags
        # as a tuple, while JSON persists them as a list.  Resume validation
        # compares the complete record, so canonicalize container types before
        # freezing the scientific configuration.
        "cuda_profile": json.loads(json.dumps(profile.to_dict())),
        "source_image": {
            "label": 3,
            "class_index": 0,
            "lambda_mix": 0.35,
            "image_sha256": IMAGE_SHA256,
            "mixed_target_sha256": MIXED_TARGET_SHA256,
            "source_image_npz_sha256": SOURCE_IMAGE_NPZ_SHA256,
        },
        "panel_path_count": PATHS_PER_PANEL,
        "panel_transition_count": EXPECTED_PANEL_TRANSITIONS,
        "total_transition_count": EXPECTED_TOTAL_TRANSITIONS,
        "resource_thresholds": {
            "maximum_projected_two_panel_hours": 24.0,
            "minimum_transitions_per_second": 1300.0,
            "maximum_peak_memory_fraction": 0.80,
            "maximum_fallback_fraction": 1.0e-4,
            "maximum_fallback_time_fraction": 0.10,
            "maximum_mass_error": 2.0e-12,
        },
        "analysis": statistic.to_record(),
        "target_modification_permitted": 0,
        "parent_state_label_prediction_reuse_permitted": 0,
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    record["semantic_sha256"] = config_fingerprint(record)
    return record


def _manifest(
    args: argparse.Namespace,
    *,
    sources: Sequence[Path],
    scientific: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "created_by": "mnist.diag_d0_jacobi_rb_physical_coarse_signal_witness",
        "stage_contract": [
            "preflight",
            "panel-a",
            "panel-b",
            "analyze",
            "report",
        ],
        "parent_one_image_run_dir": str(
            Path(args.parent_one_image_run_dir).resolve()
        ),
        "parent_zero_signal_run_dir": str(
            Path(args.parent_zero_signal_run_dir).resolve()
        ),
        "parent_bayes_power_run_dir": str(
            Path(args.parent_bayes_power_run_dir).resolve()
        ),
        "source_paths": [path.as_posix() for path in sources],
        "source_fingerprint": source_fingerprint(sources),
        "scientific_config_sha256": scientific["semantic_sha256"],
        "path_plan_sha256": scientific["path_plan_sha256"],
        "statistic_plan_sha256": scientific["statistic_plan_sha256"],
        **CLAIM_FLAGS,
        **NO_WORK,
    }


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        run_dir = Path(args.resume_run_dir).resolve()
        if not run_dir.is_dir():
            raise ArtifactCompatibilityError(f"resume run does not exist: {run_dir}")
        return run_dir, True
    root = Path(args.runs_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{timestamp}_{args.run_name}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir, False


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    records = []
    for path in sorted(run_dir.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name in _REGISTRY_EXCLUDED:
            continue
        records.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "sha256": file_fingerprint(path),
                "size": int(path.stat().st_size),
            }
        )
    record = {
        "schema": RUN_SCHEMA + "-artifact-registry",
        "schema_version": 1,
        "record_count": len(records),
        "records": records,
        "registry_sha256": config_fingerprint(records),
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "artifact_registry.json", record)
    return record


def _verify_own_registry(run_dir: Path) -> None:
    path = run_dir / "artifact_registry.json"
    if not path.is_file():
        return
    registry = _load_json(path)
    rows = registry.get("records")
    if (
        not isinstance(rows, list)
        or int(registry.get("record_count", -1)) != len(rows)
        or registry.get("registry_sha256") != config_fingerprint(rows)
    ):
        raise ArtifactCompatibilityError("witness artifact registry is malformed")
    expected: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ArtifactCompatibilityError("witness registry row is malformed")
        relative = str(row.get("path", ""))
        artifact = run_dir / relative
        if (
            not relative
            or relative in expected
            or not artifact.is_file()
            or int(row.get("size", -1)) != artifact.stat().st_size
            or row.get("sha256") != file_fingerprint(artifact)
        ):
            raise ArtifactCompatibilityError(
                f"registered witness artifact changed: {relative}"
            )
        expected.add(relative)
    status_path = run_dir / "run_status.json"
    if status_path.is_file() and "artifact_registry_sha256" in _load_json(status_path):
        actual = {
            item.relative_to(run_dir).as_posix()
            for item in run_dir.rglob("*")
            if item.is_file() and item.name not in _REGISTRY_EXCLUDED
        }
        if actual != expected:
            raise ArtifactCompatibilityError(
                "terminal witness artifact file set changed"
            )


def _status(
    run_dir: Path,
    *,
    stage: str,
    state: str,
    decision: str | None,
    message: str = "",
    registry: Mapping[str, Any] | None = None,
) -> None:
    binding: dict[str, Any] = {}
    if registry is not None:
        registry_path = run_dir / "artifact_registry.json"
        binding = {
            "artifact_registry_record_count": int(registry["record_count"]),
            "artifact_registry_sha256": str(registry["registry_sha256"]),
            "artifact_registry_file_sha256": file_fingerprint(registry_path),
            "artifact_registry_file_size": int(registry_path.stat().st_size),
        }
    atomic_write_json(
        run_dir / "run_status.json",
        {
            "schema": RUN_SCHEMA + "-status",
            "schema_version": 1,
            "updated_at": _now(),
            "stage": str(stage),
            "state": str(state),
            "decision": decision,
            "message": str(message),
            **binding,
            **CLAIM_FLAGS,
            **NO_WORK,
        },
    )


def _load_gate(run_dir: Path, name: str) -> dict[str, Any] | None:
    path = run_dir / _GATE_FILES[name]
    return _load_json(path) if path.is_file() else None


def _save_gate(run_dir: Path, name: str, gate: Mapping[str, Any]) -> None:
    atomic_write_json(run_dir / _GATE_FILES[name], dict(gate))


def _refresh_decision(
    run_dir: Path, *, scientific_outcome: str | None = None
) -> dict[str, Any]:
    execution_failure = next(
        (
            (stage, gate)
            for stage, gate in (
                ("preflight", _load_gate(run_dir, "preflight")),
                ("panel-a", _load_gate(run_dir, "panel-a")),
                ("panel-b", _load_gate(run_dir, "panel-b")),
                ("analyze", _load_gate(run_dir, "witness")),
            )
            if gate is not None
            and gate.get("evaluation_status") == "execution_failed"
        ),
        None,
    )
    if execution_failure is not None:
        failed_stage, failed_gate = execution_failure
        expected = _failure_decision(
            stage=failed_stage,
            failure_domain=str(failed_gate.get("failure_domain", "workflow_execution")),
            failure_code=str(
                failed_gate.get(
                    "failure_code", "physical_coarse_signal_execution_failed"
                )
            ),
            message=str(failed_gate.get("message", "")),
        )
        path = run_dir / "physical_coarse_signal_decision.json"
        if path.is_file() and _load_json(path) != expected:
            raise ArtifactCompatibilityError(
                "execution-failure decision no longer matches its gate"
            )
        if not path.is_file():
            atomic_write_json(path, expected)
        return expected
    if scientific_outcome is None:
        result_path = run_dir / "physical_coarse_signal_analysis.json"
        if result_path.is_file():
            analysis = _load_json(result_path)
            classification = analysis.get("classification")
            if isinstance(classification, Mapping):
                scientific_outcome = str(classification.get("decision"))
    decision = witness_decision(
        preflight_gate=_load_gate(run_dir, "preflight"),
        panel_a_gate=_load_gate(run_dir, "panel-a"),
        panel_b_gate=_load_gate(run_dir, "panel-b"),
        witness_gate=_load_gate(run_dir, "witness"),
        scientific_outcome=scientific_outcome,
    )
    atomic_write_json(run_dir / "physical_coarse_signal_decision.json", decision)
    return decision


def _finalize(
    run_dir: Path,
    *,
    stage: str,
    required_gate: str,
    message: str = "",
) -> int:
    decision = _refresh_decision(run_dir)
    execution_failed = any(
        gate is not None and gate.get("evaluation_status") == "execution_failed"
        for gate in (
            _load_gate(run_dir, "preflight"),
            _load_gate(run_dir, "panel-a"),
            _load_gate(run_dir, "panel-b"),
            _load_gate(run_dir, "witness"),
        )
    )
    gate = None if required_gate == "none" else _load_gate(run_dir, required_gate)
    gate_passed = bool(
        gate
        and gate.get("evaluation_status") == "evaluated"
        and int(gate.get("passed", 0)) == 1
    )
    state = (
        "gate_failed"
        if execution_failed
        else (
            "completed"
            if required_gate == "none" or gate_passed
            else "gate_failed"
        )
    )
    registry = _artifact_registry(run_dir)
    _status(
        run_dir,
        stage=stage,
        state=state,
        decision=str(decision["decision"]),
        message=message,
        registry=registry,
    )
    return 0 if state == "completed" else 1


def _initialize_or_validate(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    resumed: bool,
) -> tuple[dict[str, Any], dict[str, Any], tuple[Path, ...]]:
    sources = _source_paths()
    scientific = _scientific_config()
    manifest = _manifest(args, sources=sources, scientific=scientific)
    if resumed:
        _verify_own_registry(run_dir)
        if _load_json(run_dir / "run_manifest.json") != manifest:
            raise ArtifactCompatibilityError("resume manifest changed")
        if _load_json(run_dir / "scientific_config.json") != scientific:
            raise ArtifactCompatibilityError("resume scientific configuration changed")
        path_plan = frozen_path_plan().to_record()
        statistic_plan = frozen_statistic_plan().to_record()
        if _load_json(run_dir / "path_id_plan.json") != path_plan:
            raise ArtifactCompatibilityError("resume path plan changed")
        if _load_json(run_dir / "statistic_plan.json") != statistic_plan:
            raise ArtifactCompatibilityError("resume statistic plan changed")
    else:
        _freeze_json(run_dir / "run_manifest.json", manifest)
        _freeze_json(run_dir / "scientific_config.json", scientific)
        _freeze_json(run_dir / "path_id_plan.json", frozen_path_plan().to_record())
        _freeze_json(
            run_dir / "statistic_plan.json", frozen_statistic_plan().to_record()
        )
        _status(
            run_dir,
            stage="initialization",
            state="running",
            decision="ready_for_preflight",
        )
    return manifest, scientific, sources


def _require_gate_pass(run_dir: Path, name: str) -> dict[str, Any]:
    gate = _load_gate(run_dir, name)
    if (
        gate is None
        or gate.get("evaluation_status") != "evaluated"
        or int(gate.get("passed", 0)) != 1
    ):
        raise CoarseWitnessCLIError(
            f"{name} gate must pass before this stage",
            failure_domain="stage_prerequisite",
            failure_code=f"{name.replace('-', '_')}_gate_required",
        )
    return gate


def _load_source_image(parent_run_dir: str | Path) -> tuple[dict[str, Any], np.ndarray]:
    root = Path(parent_run_dir)
    metadata = _load_json(root / "source_image.json")
    npz_path = root / "source_image.npz"
    if (
        metadata.get("image_sha256") != IMAGE_SHA256
        or metadata.get("mixed_target_sha256") != MIXED_TARGET_SHA256
        or metadata.get("npz_sha256") != SOURCE_IMAGE_NPZ_SHA256
        or file_fingerprint(npz_path) != SOURCE_IMAGE_NPZ_SHA256
        or int(metadata.get("label", -1)) != 3
        or int(metadata.get("class_index", -1)) != 0
        or float(metadata.get("lambda_mix", math.nan)) != 0.35
    ):
        raise ArtifactCompatibilityError("frozen source image binding changed")
    arrays = _load_npz(npz_path)
    if set(arrays) != {"image", "mixed_target"}:
        raise ArtifactCompatibilityError("source image NPZ schema changed")
    mixed = arrays["mixed_target"]
    if (
        mixed.dtype != np.float64
        or mixed.shape != (784,)
        or not np.isfinite(mixed).all()
        or np.any(mixed < 0.0)
        or abs(float(mixed.sum()) - 1.0) > 1.0e-12
    ):
        raise ArtifactCompatibilityError("source mixed target is invalid")
    return metadata, np.array(mixed, dtype=np.float64, order="C", copy=True)


def _bayes_panels(parent_run_dir: str | Path) -> dict[str, dict[str, PhysicalCoarsePanel]]:
    cache_root = Path(parent_run_dir) / "cache"
    result: dict[str, dict[str, PhysicalCoarsePanel]] = {}
    for law in ("teacher", "null"):
        splits: dict[str, PhysicalCoarsePanel] = {}
        for split in ("train", "validation", "confirmation"):
            cache = load_bayes_label_cache(
                cache_root / f"{law}_{split}_labels.npz"
            )
            splits[split] = coarse_cell_path_means(
                cache.denoising_target,
                cache.path_id,
                cache.outer_step,
                cache.phase,
                role=f"{law}-{split}",
            )
        result[law] = splits
    return result


def _old_physical_forecast(parent_zero_signal_run_dir: str | Path) -> dict[str, Any]:
    source_path = Path(parent_zero_signal_run_dir) / "zero_signal_diagnostic.json"
    source = _load_json(source_path)
    rows = source.get("coarse_cross_split_signal")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ArtifactCompatibilityError("old coarse-signal forecast evidence changed")
    normalized = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ArtifactCompatibilityError("old coarse-signal row is malformed")
        lower = float(row["lower"])
        upper = float(row["upper"])
        point = float(row["cross_split_coarse_signal"])
        if not all(math.isfinite(value) for value in (lower, upper, point)):
            raise ArtifactCompatibilityError("old coarse-signal row is nonfinite")
        normalized.append(
            {
                "left_split": str(row["left_split"]),
                "right_split": str(row["right_split"]),
                "old_path_count_per_split": int(row["left_path_count"]),
                "point_estimate": point,
                "old_lower": lower,
                "old_upper": upper,
                "rough_64_path_half_width_forecast": (
                    0.5 * (upper - lower) * math.sqrt(8.0 / 64.0)
                ),
            }
        )
    return {
        "schema": RUN_SCHEMA + "-old-physical-power-forecast",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "authorizing": 0,
        "source_artifact": source_path.resolve().as_posix(),
        "source_sha256": file_fingerprint(source_path),
        "rows": normalized,
        "old_state_used_in_new_estimate": 0,
        "old_label_used_in_new_estimate": 0,
        "old_prediction_used_in_new_estimate": 0,
        **NO_WORK,
    }


def _benchmark_capture(
    run_dir: Path,
    *,
    mixed_target: np.ndarray,
    device: torch.device,
    scientific_config_sha256: str,
) -> dict[str, Any]:
    profile = JacobiRBCudaProfile()
    paths = PREFLIGHT_BENCHMARK_PATH_IDS
    initial = np.repeat(mixed_target[None, :], len(paths), axis=0)
    warmup_states = torch.as_tensor(
        np.array(initial, dtype=np.float64, order="C", copy=True),
        dtype=torch.float64,
        device=device,
    ).contiguous()
    warmup = run_exact_multipath_shard(
        warmup_states,
        path_ids=paths,
        start_step=0,
        root_seed=ROOT_SEED,
        profile=profile,
        group_sizes=(len(paths),),
        capture_training_payload=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    capture_states = torch.as_tensor(
        np.array(
            warmup.committed_final_states,
            dtype=np.float64,
            order="C",
            copy=True,
        ),
        dtype=torch.float64,
        device=device,
    ).contiguous()
    capture = run_exact_multipath_shard(
        capture_states,
        path_ids=paths,
        start_step=8,
        root_seed=ROOT_SEED,
        profile=profile,
        group_sizes=(len(paths),),
        capture_training_payload=True,
    )
    if capture.capture_payload is None:
        raise CoarseWitnessCLIError(
            "benchmark capture payload is missing",
            failure_domain="resource_benchmark",
            failure_code="benchmark_capture_missing",
        )
    targets = selected_target_contribution(
        capture.capture_payload,
        selected_outer_step=15,
        expected_path_ids=paths,
    )
    cell_sums = np.zeros((len(paths), 4, 7, 392), dtype=np.float64)
    cell_compensations = np.zeros_like(cell_sums)
    cell_counts = np.zeros(4, dtype=np.int16)
    update_cell_accumulator(
        cell_sums,
        cell_compensations,
        cell_counts,
        targets,
        quartile=0,
    )
    restart_npz = _atomic_npz(
        run_dir / "preflight_benchmark_restart_state.npz",
        {
            "path_ids": np.asarray(paths, dtype=np.int64),
            "final_states": capture.committed_final_states,
        },
    )
    accumulator_npz = _atomic_npz(
        run_dir / "preflight_benchmark_accumulator.npz",
        {
            "cell_sums": cell_sums,
            "cell_compensations": cell_compensations,
            "cell_counts": cell_counts,
        },
    )
    atomic_write_json(
        run_dir / "preflight_benchmark_shard_commit.json",
        {
            "schema": RUN_SCHEMA + "-benchmark-shard-commit",
            "schema_version": 1,
            "path_ids": list(paths),
            "start_step": 8,
            "step_count": SHARD_STEPS,
            "restart_artifact_sha256": restart_npz["sha256"],
            "accumulator_artifact_sha256": accumulator_npz["sha256"],
            "selected_target_sha256": _array_sha256(targets),
            "scheduler_record_sha256": config_fingerprint(capture.to_record()),
            "raw_target_observations_persisted": 0,
            **NO_WORK,
        },
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_bytes = int(torch.cuda.max_memory_allocated(device))
        total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    else:
        peak_bytes = 0
        total_bytes = 1
    elapsed = time.perf_counter() - started
    diagnostics = capture.diagnostics
    transitions = int(diagnostics["transition_count"])
    rate = transitions / elapsed if elapsed > 0.0 else 0.0
    fallback_seconds = float(diagnostics["fallback_elapsed_seconds"])
    forbidden = {
        name: int(diagnostics.get(name, 0)) for name in FORBIDDEN_COUNTS
    }
    record = {
        "schema": RUN_SCHEMA + "-capture-benchmark",
        "schema_version": 1,
        "scientific_config_sha256": scientific_config_sha256,
        "path_ids": list(paths),
        "warmup_start_step": 0,
        "measured_start_step": 8,
        "step_count": SHARD_STEPS,
        "selected_outer_step": 15,
        "transition_count": transitions,
        "certified_count": int(diagnostics["certified_count"]),
        "certificate_fraction": (
            int(diagnostics["certified_count"]) / transitions
            if transitions
            else 0.0
        ),
        "fallback_count": int(diagnostics["fallback_count"]),
        "fallback_fraction": (
            int(diagnostics["fallback_count"]) / transitions
            if transitions
            else math.inf
        ),
        "fallback_elapsed_seconds": fallback_seconds,
        "fallback_time_fraction": (
            fallback_seconds / elapsed if elapsed > 0.0 else math.inf
        ),
        "complete_capture_elapsed_seconds": elapsed,
        "transitions_per_second": rate,
        "projected_two_panel_hours": EXPECTED_TOTAL_TRANSITIONS / rate / 3600.0,
        "peak_memory_bytes": peak_bytes,
        "device_total_memory_bytes": total_bytes,
        "peak_memory_fraction": peak_bytes / total_bytes,
        "maximum_mass_error": float(diagnostics["maximum_mass_error"]),
        "states_finite": int(np.isfinite(capture.committed_final_states).all()),
        "targets_finite": int(np.isfinite(targets).all()),
        "capture_shape": list(targets.shape),
        "selected_target_sha256": _array_sha256(targets),
        "raw_target_observations_persisted": 0,
        "restart_artifact": restart_npz,
        "accumulator_artifact": accumulator_npz,
        "forbidden_counts": forbidden,
        "scheduler_record": capture.to_record(),
        **NO_WORK,
    }
    atomic_write_json(run_dir / "physical_capture_benchmark.json", record)
    return record


def _preflight_stage(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    scientific: Mapping[str, Any],
) -> dict[str, Any]:
    existing = _load_gate(run_dir, "preflight")
    if (
        existing is not None
        and existing.get("evaluation_status") == "evaluated"
    ):
        return existing
    if (
        existing is not None
        and existing.get("evaluation_status") != "execution_failed"
    ):
        raise ArtifactCompatibilityError("preflight gate has an invalid state")
    provenance = verify_physical_coarse_signal_parents(
        physical_run_dir=args.parent_one_image_run_dir,
        zero_signal_run_dir=args.parent_zero_signal_run_dir,
        bayes_power_run_dir=args.parent_bayes_power_run_dir,
    )
    atomic_write_json(run_dir / "parent_provenance.json", provenance)
    path_plan = frozen_path_plan()
    statistic_plan = frozen_statistic_plan()
    all_paths = (
        path_plan.panel_a
        + path_plan.panel_b
        + path_plan.preflight_benchmark
    )
    path_checks = {
        "schema": RUN_SCHEMA + "-path-plan-checks",
        "schema_version": 1,
        "all_ids_20_bit": int(all(0 <= value < (1 << 20) for value in all_paths)),
        "all_ids_unique": int(len(all_paths) == len(set(all_paths))),
        "panel_a_range_pass": int(
            path_plan.panel_a == tuple(range(0xE5000, 0xE5040))
        ),
        "panel_b_range_pass": int(
            path_plan.panel_b == tuple(range(0xE5100, 0xE5140))
        ),
        "benchmark_range_pass": int(
            path_plan.preflight_benchmark == tuple(range(0xE5200, 0xE5208))
        ),
        "path_plan_sha256": path_plan.fingerprint,
        "statistic_plan_sha256": statistic_plan.fingerprint,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "path_and_statistic_plan_checks.json", path_checks)

    bayes_panels = _bayes_panels(args.parent_bayes_power_run_dir)
    bayes_replay = evaluate_bayes_control_replay(
        teacher_panels=bayes_panels["teacher"],
        null_panels=bayes_panels["null"],
        seed=BOOTSTRAP_SEED,
        replicates=BOOTSTRAP_REPLICATES,
    )
    atomic_write_json(run_dir / "bayes_control_replay.json", bayes_replay)
    _write_csv(run_dir / "bayes_control_replay.csv", bayes_replay["rows"])
    forecast = _old_physical_forecast(args.parent_zero_signal_run_dir)
    atomic_write_json(run_dir / "old_physical_power_forecast.json", forecast)

    source_metadata, mixed_target = _load_source_image(
        args.parent_one_image_run_dir
    )
    atomic_write_json(
        run_dir / "source_image_binding.json",
        {
            "schema": RUN_SCHEMA + "-source-image-binding",
            "schema_version": 1,
            **source_metadata,
            "parent_source_image_npz": str(
                (Path(args.parent_one_image_run_dir) / "source_image.npz").resolve()
            ),
            "old_random_state_reused": 0,
            **NO_WORK,
        },
    )
    device = torch.device(args.device)
    runtime = configure_exact_torch_backend(device)
    runtime.update(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_argument": str(args.device),
        }
    )
    atomic_write_json(run_dir / "exact_backend_runtime.json", runtime)
    benchmark = _benchmark_capture(
        run_dir,
        mixed_target=mixed_target,
        device=device,
        scientific_config_sha256=str(scientific["semantic_sha256"]),
    )

    teacher_rows = [
        row for row in bayes_replay["rows"] if row["law"] == "teacher"
    ]
    null_rows = [row for row in bayes_replay["rows"] if row["law"] == "null"]
    forbidden_total = sum(benchmark["forbidden_counts"].values())
    metrics = {
        "schema": RUN_SCHEMA + "-preflight-metrics",
        "schema_version": 1,
        "physical_parent_verified": int(
            provenance.get("passed", 0) == 1
            and "physical_one_image" in provenance["parents"]
        ),
        "zero_signal_parent_verified": int(
            provenance.get("passed", 0) == 1
            and "zero_signal_diagnostic" in provenance["parents"]
        ),
        "bayes_parent_verified": int(
            provenance.get("passed", 0) == 1
            and "bayes_power_calibration" in provenance["parents"]
        ),
        "all_parent_registries_verified": int(provenance.get("passed", 0)),
        "all_parent_sources_verified": int(provenance.get("passed", 0)),
        "no_parent_artifact_reused_in_estimate": 1,
        "path_plan_valid": int(
            path_checks["all_ids_20_bit"]
            and path_checks["all_ids_unique"]
            and path_checks["panel_a_range_pass"]
            and path_checks["panel_b_range_pass"]
            and path_checks["benchmark_range_pass"]
        ),
        "path_roles_disjoint": int(path_checks["all_ids_unique"]),
        "statistic_plan_frozen": int(
            statistic_plan.bootstrap_replicates == 50_000
            and statistic_plan.bootstrap_seed == 261_242
            and statistic_plan.resolution_target == 5.0e-4
        ),
        "bayes_teacher_all_pairs_detected": int(
            len(teacher_rows) == 3
            and all(int(row["passed"]) == 1 for row in teacher_rows)
        ),
        "bayes_null_all_pairs_cover_zero": int(
            len(null_rows) == 3
            and all(int(row["passed"]) == 1 for row in null_rows)
        ),
        "bayes_replay_whole_path_only": int(
            bayes_replay.get("bootstrap_unit")
            == "whole_path_independently_within_split"
        ),
        "old_physical_forecast_nonauthorizing": int(
            forecast.get("authorizing") == 0
            and all(
                int(forecast.get(name, 1)) == 0
                for name in (
                    "old_state_used_in_new_estimate",
                    "old_label_used_in_new_estimate",
                    "old_prediction_used_in_new_estimate",
                )
            )
        ),
        "benchmark_complete_capture_path": int(
            benchmark["capture_shape"] == [8, 7, 392]
            and benchmark["transition_count"]
            == 8 * SHARD_STEPS * len(PHASE_MATCHINGS) * EDGES_PER_PHASE
        ),
        "benchmark_certificate_fraction_one": int(
            benchmark["certificate_fraction"] == 1.0
        ),
        "benchmark_forbidden_events_zero": int(forbidden_total == 0),
        "benchmark_states_finite": int(benchmark["states_finite"]),
        "benchmark_target_finite": int(benchmark["targets_finite"]),
        "benchmark_mass_conservation_pass": int(
            benchmark["maximum_mass_error"] <= 2.0e-12
        ),
        "benchmark_raw_targets_not_persisted": int(
            benchmark.get("raw_target_observations_persisted", -1) == 0
        ),
        "projected_two_panel_hours": benchmark["projected_two_panel_hours"],
        "transitions_per_second": benchmark["transitions_per_second"],
        "peak_memory_fraction": benchmark["peak_memory_fraction"],
        "fallback_fraction": benchmark["fallback_fraction"],
        "fallback_time_fraction": benchmark["fallback_time_fraction"],
        "maximum_mass_error": benchmark["maximum_mass_error"],
        **NO_WORK,
    }
    atomic_write_json(run_dir / "coarse_signal_preflight_metrics.json", metrics)
    gate = evaluate_preflight(metrics)
    _save_gate(run_dir, "preflight", gate)
    return gate


def _panel_paths(panel: str) -> tuple[int, ...]:
    if panel == "a":
        return PANEL_A_PATH_IDS
    if panel == "b":
        return PANEL_B_PATH_IDS
    raise ValueError("panel must be 'a' or 'b'")


def _panel_seal_path(run_dir: Path, panel: str) -> Path:
    return run_dir / f"panel_{panel}_seal.json"


def _load_panel_cell_means(
    run_dir: Path, panel: str
) -> tuple[np.ndarray, np.ndarray, Path]:
    path = run_dir / "panels" / panel / "cell_means.npz"
    arrays = _load_npz(path)
    if set(arrays) != {"path_ids", "cell_means"}:
        raise ArtifactCompatibilityError(f"panel {panel} cell-means schema changed")
    path_ids = arrays["path_ids"]
    means = arrays["cell_means"]
    if (
        path_ids.dtype != np.int64
        or path_ids.tolist() != list(_panel_paths(panel))
        or means.dtype != np.float64
        or means.shape != (64, 4, 7, 392)
        or not np.isfinite(means).all()
    ):
        raise ArtifactCompatibilityError(f"panel {panel} cell means are invalid")
    return path_ids, means, path


def _validate_panel_seal(run_dir: Path, panel: str) -> dict[str, Any]:
    seal_path = _panel_seal_path(run_dir, panel)
    seal = _load_json(seal_path)
    path_ids, means, data_path = _load_panel_cell_means(run_dir, panel)
    metrics_path = run_dir / "panels" / panel / "metrics.json"
    audit_path = run_dir / f"panel_{panel}_cell_mean_persistence_audit.json"
    resource_path = run_dir / f"panel_{panel}_resource_summary.json"
    if (
        seal.get("schema") != RUN_SCHEMA + "-panel-seal"
        or seal.get("panel") != panel
        or seal.get("path_ids") != path_ids.tolist()
        or seal.get("cell_means_file_sha256") != file_fingerprint(data_path)
        or seal.get("cell_means_array_sha256") != _array_sha256(means)
        or seal.get("path_plan_sha256") != frozen_path_plan().fingerprint
        or seal.get("statistic_plan_sha256") != frozen_statistic_plan().fingerprint
        or not metrics_path.is_file()
        or seal.get("execution_metrics_file_sha256")
        != file_fingerprint(metrics_path)
        or not audit_path.is_file()
        or seal.get("persistence_audit_file_sha256")
        != file_fingerprint(audit_path)
        or not resource_path.is_file()
        or seal.get("resource_summary_file_sha256")
        != file_fingerprint(resource_path)
    ):
        raise ArtifactCompatibilityError(f"panel {panel} seal changed")
    return seal


def _cell_mean_persistence_audit(
    run_dir: Path, panel: str, expected: np.ndarray
) -> dict[str, Any]:
    reconstructed = []
    for group_index in range(8):
        path = (
            run_dir
            / "panels"
            / panel
            / f"group-{group_index:02d}-cell-means.npz"
        )
        arrays = _load_npz(path)
        means = arrays.get("cell_means")
        path_ids = arrays.get("path_ids")
        expected_paths = np.asarray(
            _panel_paths(panel)[8 * group_index : 8 * (group_index + 1)],
            dtype=np.int64,
        )
        if (
            means is None
            or means.dtype != np.float64
            or means.shape != (8, 4, 7, 392)
            or path_ids is None
            or path_ids.dtype != np.int64
            or not np.array_equal(path_ids, expected_paths)
        ):
            raise ArtifactCompatibilityError(
                f"panel {panel} group cell means changed"
            )
        reconstructed.append(means)
    reassembled = np.ascontiguousarray(np.concatenate(reconstructed, axis=0))
    maximum_error = float(np.max(np.abs(reassembled - expected)))
    return {
        "maximum_absolute_error": maximum_error,
        "tolerance": 0.0,
        "passed": int(maximum_error == 0.0),
        "implementation": "canonical_group_file_reassembly",
        "raw_target_observations_persisted": 0,
        **NO_WORK,
    }


def _panel_metrics_for_gate(
    run_dir: Path,
    *,
    panel: str,
    execution_metrics: Mapping[str, Any],
    peak_memory_fraction: float,
    persistence_audit: Mapping[str, Any],
) -> dict[str, Any]:
    preflight = _load_json(run_dir / "coarse_signal_preflight_metrics.json")
    forbidden_total = sum(
        int(execution_metrics.get(name, 0)) for name in FORBIDDEN_COUNTS
    )
    return {
        "schema": RUN_SCHEMA + "-panel-gate-metrics",
        "schema_version": 1,
        "panel": panel,
        "path_plan_binding_pass": 1,
        "statistic_plan_binding_pass": 1,
        "panel_role_isolated": int(
            set(_panel_paths(panel)).isdisjoint(
                _panel_paths("b" if panel == "a" else "a")
            )
        ),
        "panel_sealed": int(_panel_seal_path(run_dir, panel).is_file()),
        "all_groups_complete": int(
            execution_metrics.get("all_shards_complete_pass", 0)
        ),
        "shard_chains_valid": int(
            execution_metrics.get("all_shards_complete_pass", 0)
        ),
        "resume_state_hashes_valid": int(
            execution_metrics.get("all_shards_complete_pass", 0)
        ),
        "path_count_pass": int(execution_metrics.get("path_count") == 64),
        "cell_shape_pass": int(
            execution_metrics.get("cell_shape") == [64, 4, 7, 392]
        ),
        "selected_step_coverage_pass": int(
            execution_metrics.get("selected_outer_steps")
            == list(frozen_statistic_plan().selected_outer_steps)
        ),
        "eight_observations_per_cell_pass": int(
            execution_metrics.get("observations_per_path_cell") == 8
        ),
        "cell_means_finite": int(
            execution_metrics.get("cell_means_finite_pass", 0)
        ),
        "cell_means_persistence_audit_pass": int(
            persistence_audit["passed"]
        ),
        "cell_means_persistence_maximum_error": persistence_audit[
            "maximum_absolute_error"
        ],
        "certificate_fraction_one": int(
            float(execution_metrics.get("certificate_fraction", 0.0)) == 1.0
        ),
        "forbidden_events_zero": int(forbidden_total == 0),
        "state_updates_device_resident": int(
            execution_metrics.get("state_updates_device_resident_pass", 0)
        ),
        "target_modification_count_zero": int(
            execution_metrics.get("target_modification_count", -1) == 0
        ),
        "raw_target_observations_not_persisted": int(
            execution_metrics.get("raw_target_observations_persisted", -1)
            == 0
        ),
        "path_count": int(execution_metrics.get("path_count", -1)),
        "transition_count": int(execution_metrics.get("transition_count", -1)),
        "maximum_mass_error": float(
            execution_metrics.get("maximum_mass_error", math.inf)
        ),
        "fallback_fraction": float(
            execution_metrics.get("fallback_fraction", math.inf)
        ),
        "fallback_time_fraction": float(
            execution_metrics.get("fallback_time_fraction", math.inf)
        ),
        "peak_memory_fraction": max(
            float(peak_memory_fraction),
            float(preflight.get("peak_memory_fraction", 0.0)),
        ),
        "transitions_per_second": float(
            execution_metrics.get(
                "complete_pipeline_transitions_per_second", -math.inf
            )
        ),
        **NO_WORK,
    }


def _ensure_joint_analysis_seal(run_dir: Path) -> None:
    panel_a_seal = _validate_panel_seal(run_dir, "a")
    panel_b_seal = _validate_panel_seal(run_dir, "b")
    joint_path = run_dir / "joint_analysis_seal.json"
    sealed_at = (
        _load_json(joint_path)["sealed_at"]
        if joint_path.is_file()
        else _now()
    )
    joint = {
        "schema": RUN_SCHEMA + "-joint-analysis-seal",
        "schema_version": 1,
        "sealed_at": sealed_at,
        "panel_a_seal_sha256": config_fingerprint(panel_a_seal),
        "panel_b_seal_sha256": config_fingerprint(panel_b_seal),
        "panel_a_file_sha256": panel_a_seal["cell_means_file_sha256"],
        "panel_b_file_sha256": panel_b_seal["cell_means_file_sha256"],
        "statistic_plan_sha256": frozen_statistic_plan().fingerprint,
        "analysis_definition_frozen_before_open": 1,
        "analysis_open_count": 0,
        **NO_WORK,
    }
    _freeze_json(run_dir / "joint_analysis_seal.json", joint)


def _finalize_panel_evidence(
    run_dir: Path,
    *,
    panel: str,
    prerequisite: Mapping[str, Any],
) -> dict[str, Any]:
    execution_metrics = _load_json(run_dir / "panels" / panel / "metrics.json")
    persistence_audit = _load_json(
        run_dir / f"panel_{panel}_cell_mean_persistence_audit.json"
    )
    resource = _load_json(run_dir / f"panel_{panel}_resource_summary.json")
    expected_metrics = _panel_metrics_for_gate(
        run_dir,
        panel=panel,
        execution_metrics=execution_metrics,
        peak_memory_fraction=float(resource["peak_memory_fraction"]),
        persistence_audit=persistence_audit,
    )
    metrics_path = run_dir / f"coarse_signal_panel_{panel}_metrics.json"
    if metrics_path.is_file() and _load_json(metrics_path) != expected_metrics:
        raise ArtifactCompatibilityError(
            f"panel {panel} gate metrics changed"
        )
    if not metrics_path.is_file():
        atomic_write_json(metrics_path, expected_metrics)
    gate = evaluate_panel(
        expected_metrics,
        panel=f"panel-{panel}",
        prerequisite_gate=prerequisite,
    )
    _save_gate(run_dir, f"panel-{panel}", gate)
    if panel == "b" and int(gate.get("passed", 0)) == 1:
        _ensure_joint_analysis_seal(run_dir)
    return gate


def _panel_stage(
    run_dir: Path,
    *,
    panel: str,
    args: argparse.Namespace,
    scientific: Mapping[str, Any],
) -> dict[str, Any]:
    prerequisite_name = "preflight" if panel == "a" else "panel-a"
    prerequisite = _require_gate_pass(run_dir, prerequisite_name)
    if panel == "a" and (
        _panel_seal_path(run_dir, "b").exists()
        or (run_dir / "joint_analysis_seal.json").exists()
    ):
        raise ArtifactCompatibilityError(
            "panel A cannot be reopened after panel B has started"
        )
    if panel == "b" and (run_dir / "analysis_open.json").exists():
        raise ArtifactCompatibilityError(
            "panel B cannot be reopened after joint analysis"
        )
    seal_path = _panel_seal_path(run_dir, panel)
    if seal_path.is_file():
        _validate_panel_seal(run_dir, panel)
        gate = _load_gate(run_dir, f"panel-{panel}")
        if (
            gate is not None
            and gate.get("evaluation_status") == "evaluated"
        ):
            if panel == "b" and int(gate.get("passed", 0)) == 1:
                _ensure_joint_analysis_seal(run_dir)
            return gate
        return _finalize_panel_evidence(
            run_dir, panel=panel, prerequisite=prerequisite
        )

    _metadata, mixed_target = _load_source_image(args.parent_one_image_run_dir)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise CoarseWitnessCLIError(
            "authorizing physical panels require CUDA",
            failure_domain="physical_panel_resource",
            failure_code="physical_panel_cuda_required",
        )
    torch.cuda.reset_peak_memory_stats(device)
    result = run_physical_panel(
        run_dir,
        panel=panel,
        path_ids=_panel_paths(panel),
        mixed_target=mixed_target,
        root_seed=ROOT_SEED,
        profile=JacobiRBCudaProfile(),
        device=device,
        scientific_config_sha256=str(scientific["semantic_sha256"]),
        path_plan_sha256=frozen_path_plan().fingerprint,
    )
    torch.cuda.synchronize(device)
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    peak_fraction = int(torch.cuda.max_memory_allocated(device)) / total_memory
    persistence_audit = _cell_mean_persistence_audit(
        run_dir, panel, result.cell_means
    )
    atomic_write_json(
        run_dir / f"panel_{panel}_cell_mean_persistence_audit.json",
        persistence_audit,
    )
    atomic_write_json(
        run_dir / f"panel_{panel}_resource_summary.json",
        {
            "schema": RUN_SCHEMA + "-panel-resource-summary",
            "schema_version": 1,
            "panel": panel,
            "peak_memory_fraction": peak_fraction,
            **NO_WORK,
        },
    )
    seal = {
        "schema": RUN_SCHEMA + "-panel-seal",
        "schema_version": 1,
        "panel": panel,
        "sealed_at": _now(),
        "path_ids": list(_panel_paths(panel)),
        "cell_means_file": result.cell_means_path.relative_to(run_dir).as_posix(),
        "cell_means_file_sha256": file_fingerprint(result.cell_means_path),
        "cell_means_array_sha256": _array_sha256(result.cell_means),
        "panel_fingerprint": PhysicalCoarsePanel(
            role=f"panel-{panel}",
            path_ids=np.asarray(_panel_paths(panel), dtype=np.int64),
            cell_means=result.cell_means,
        ).fingerprint,
        "path_plan_sha256": frozen_path_plan().fingerprint,
        "statistic_plan_sha256": frozen_statistic_plan().fingerprint,
        "execution_metrics_file_sha256": file_fingerprint(result.metrics_path),
        "persistence_audit_file_sha256": file_fingerprint(
            run_dir / f"panel_{panel}_cell_mean_persistence_audit.json"
        ),
        "resource_summary_file_sha256": file_fingerprint(
            run_dir / f"panel_{panel}_resource_summary.json"
        ),
        "analysis_opened": 0,
        **NO_WORK,
    }
    atomic_write_json(seal_path, seal)
    return _finalize_panel_evidence(
        run_dir, panel=panel, prerequisite=prerequisite
    )


def _validate_joint_analysis_seal(run_dir: Path) -> dict[str, Any]:
    joint = _load_json(run_dir / "joint_analysis_seal.json")
    panel_a = _validate_panel_seal(run_dir, "a")
    panel_b = _validate_panel_seal(run_dir, "b")
    if (
        joint.get("schema") != RUN_SCHEMA + "-joint-analysis-seal"
        or joint.get("panel_a_seal_sha256") != config_fingerprint(panel_a)
        or joint.get("panel_b_seal_sha256") != config_fingerprint(panel_b)
        or joint.get("panel_a_file_sha256")
        != panel_a["cell_means_file_sha256"]
        or joint.get("panel_b_file_sha256")
        != panel_b["cell_means_file_sha256"]
        or joint.get("statistic_plan_sha256")
        != frozen_statistic_plan().fingerprint
        or int(joint.get("analysis_definition_frozen_before_open", 0)) != 1
        or int(joint.get("analysis_open_count", -1)) != 0
    ):
        raise ArtifactCompatibilityError("joint analysis seal changed")
    return joint


def _analysis_completion_record(
    run_dir: Path,
    *,
    joint: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    paths = (
        run_dir / "physical_coarse_signal_analysis.json",
        run_dir / "coarse_signal_influence_components.csv",
        run_dir / "coarse_signal_inference_summary.csv",
        run_dir / "coarse_signal_witness_metrics.json",
        run_dir / _GATE_FILES["witness"],
        run_dir / "physical_coarse_signal_decision.json",
    )
    if not all(path.is_file() for path in paths):
        raise ArtifactCompatibilityError(
            "completed witness analysis is missing sealed evidence"
        )
    return {
        "schema": RUN_SCHEMA + "-analysis-completion",
        "schema_version": 1,
        "joint_analysis_seal_sha256": config_fingerprint(joint),
        "witness_gate_sha256": config_fingerprint(gate),
        "artifacts": {
            path.relative_to(run_dir).as_posix(): {
                "sha256": file_fingerprint(path),
                "size": int(path.stat().st_size),
            }
            for path in paths
        },
        **NO_WORK,
    }


def _analysis_stage(run_dir: Path) -> dict[str, Any]:
    panel_a_gate = _require_gate_pass(run_dir, "panel-a")
    panel_b_gate = _require_gate_pass(run_dir, "panel-b")
    joint = _validate_joint_analysis_seal(run_dir)
    open_path = run_dir / "analysis_open.json"
    open_record = {
        "schema": RUN_SCHEMA + "-analysis-open",
        "schema_version": 1,
        "opened_at_seal_sha256": config_fingerprint(joint),
        "analysis_open_count": 1,
        "panel_a_file_sha256": joint["panel_a_file_sha256"],
        "panel_b_file_sha256": joint["panel_b_file_sha256"],
        "statistic_plan_sha256": joint["statistic_plan_sha256"],
        **NO_WORK,
    }
    _freeze_json(open_path, open_record)
    existing_gate_path = run_dir / _GATE_FILES["witness"]
    if existing_gate_path.is_file():
        existing_gate = _load_json(existing_gate_path)
        if (
            existing_gate.get("evaluation_status") != "evaluated"
            or existing_gate.get("gate") != "witness"
        ):
            raise ArtifactCompatibilityError(
                "completed witness gate has an invalid schema"
            )
        metrics = _load_json(run_dir / "coarse_signal_witness_metrics.json")
        expected_gate = evaluate_witness(
            metrics, panel_a_gate=panel_a_gate, panel_b_gate=panel_b_gate
        )
        if existing_gate != expected_gate:
            raise ArtifactCompatibilityError(
                "completed witness gate no longer matches its metrics"
            )
        analysis = _load_json(
            run_dir / "physical_coarse_signal_analysis.json"
        )
        classification = analysis.get("classification")
        if not isinstance(classification, Mapping):
            raise ArtifactCompatibilityError(
                "completed witness analysis is missing its classification"
            )
        _refresh_decision(
            run_dir,
            scientific_outcome=str(classification.get("decision")),
        )
        completion = _analysis_completion_record(
            run_dir, joint=joint, gate=existing_gate
        )
        _freeze_json(run_dir / "analysis_completion.json", completion)
        return existing_gate

    a_ids, a_means, a_path = _load_panel_cell_means(run_dir, "a")
    b_ids, b_means, b_path = _load_panel_cell_means(run_dir, "b")
    panel_a = PhysicalCoarsePanel(
        role="physical-panel-a", path_ids=a_ids, cell_means=a_means
    )
    panel_b = PhysicalCoarsePanel(
        role="physical-panel-b", path_ids=b_ids, cell_means=b_means
    )
    analysis = analyze_cross_panel_signal(
        panel_a,
        panel_b,
        seed=BOOTSTRAP_SEED,
        replicates=BOOTSTRAP_REPLICATES,
        confidence=0.99,
        namespace=0,
    )
    atomic_write_json(run_dir / "physical_coarse_signal_analysis.json", analysis)
    bootstrap = analysis["bootstrap"]
    welch = analysis["welch_delta"]
    classification = analysis["classification"]
    _write_csv(
        run_dir / "coarse_signal_influence_components.csv",
        [
            {
                "panel": "a",
                "path_id": int(path_id),
                "influence": float(value),
            }
            for path_id, value in zip(
                a_ids.tolist(), welch["left_influence"], strict=True
            )
        ]
        + [
            {
                "panel": "b",
                "path_id": int(path_id),
                "influence": float(value),
            }
            for path_id, value in zip(
                b_ids.tolist(), welch["right_influence"], strict=True
            )
        ],
    )
    _write_csv(
        run_dir / "coarse_signal_inference_summary.csv",
        [
            {
                "method": "whole_path_bootstrap",
                "point_estimate": bootstrap["point_estimate"],
                "lower_bound_99_one_sided": bootstrap["lower_bound"],
                "upper_bound_99_one_sided": bootstrap["upper_bound"],
                "lower_bound_99_central": bootstrap[
                    "central_99_lower_bound"
                ],
                "upper_bound_99_central": bootstrap[
                    "central_99_upper_bound"
                ],
                "replicates": bootstrap["replicates"],
            },
            {
                "method": "delta_welch",
                "point_estimate": welch["point_estimate"],
                "lower_bound_99_one_sided": welch["lower_bound"],
                "upper_bound_99_one_sided": welch["upper_bound"],
                "lower_bound_99_central": welch["central_99_lower_bound"],
                "upper_bound_99_central": welch["central_99_upper_bound"],
                "replicates": "",
            },
        ],
    )
    direct_point = float(
        np.mean(
            np.mean(a_means, axis=0, dtype=np.float64)
            * np.mean(b_means, axis=0, dtype=np.float64),
            dtype=np.float64,
        )
    )
    recorded_point = float(bootstrap["point_estimate"])
    allowed = {
        "exact_physical_coarse_signal_detected",
        "coarse_signal_below_preregistered_resolution",
        "physical_coarse_signal_inconclusive",
    }
    metrics = {
        "schema": RUN_SCHEMA + "-witness-metrics",
        "schema_version": 1,
        "joint_analysis_seal_valid": 1,
        "panels_opened_once": int(_load_json(open_path)["analysis_open_count"] == 1),
        "panel_hashes_unchanged": int(
            file_fingerprint(a_path) == joint["panel_a_file_sha256"]
            and file_fingerprint(b_path) == joint["panel_b_file_sha256"]
        ),
        "panel_path_sets_disjoint": int(
            set(a_ids.tolist()).isdisjoint(b_ids.tolist())
        ),
        "cell_count": COARSE_CELL_COUNT,
        "panel_a_path_count": panel_a.path_count,
        "panel_b_path_count": panel_b.path_count,
        "estimator_algebra_pass": int(
            abs(direct_point - recorded_point) <= 1.0e-12
        ),
        "estimator_algebra_absolute_error": abs(
            direct_point - recorded_point
        ),
        "estimator_algebra_tolerance": 1.0e-12,
        "direct_point_estimate": direct_point,
        "recorded_point_estimate": recorded_point,
        "bootstrap_whole_path_only": int(
            bootstrap["bootstrap_unit"]
            == "whole_path_independently_within_panel"
        ),
        "bootstrap_replicates": int(bootstrap["replicates"]),
        "bootstrap_finite": int(
            all(
                math.isfinite(float(bootstrap[name]))
                for name in ("point_estimate", "lower_bound", "upper_bound")
            )
        ),
        "influence_components_finite": int(
            np.isfinite(np.asarray(welch["left_influence"], dtype=np.float64)).all()
            and np.isfinite(
                np.asarray(welch["right_influence"], dtype=np.float64)
            ).all()
        ),
        "welch_bound_finite": int(
            all(
                math.isfinite(float(welch[name]))
                for name in (
                    "point_estimate",
                    "lower_bound",
                    "upper_bound",
                    "standard_error",
                )
            )
        ),
        "one_sided_99_percent_bounds": int(
            float(bootstrap["confidence"]) == 0.99
            and float(welch["confidence"]) == 0.99
        ),
        "negative_values_not_truncated": int(
            bootstrap["negative_values_truncated"] == 0
            and welch["negative_values_truncated"] == 0
            and classification["negative_estimates_or_intervals_truncated"] == 0
        ),
        "decision_partition_pass": int(classification["decision"] in allowed),
        "old_physical_data_excluded": 1,
        "no_training_performed": 1,
        "no_sampling_performed": 1,
        "scientific_outcome": classification["decision"],
        "resolution_target": RESOLUTION_TARGET,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "coarse_signal_witness_metrics.json", metrics)
    gate = evaluate_witness(
        metrics, panel_a_gate=panel_a_gate, panel_b_gate=panel_b_gate
    )
    _save_gate(run_dir, "witness", gate)
    _refresh_decision(
        run_dir, scientific_outcome=str(classification["decision"])
    )
    _freeze_json(
        run_dir / "analysis_completion.json",
        _analysis_completion_record(run_dir, joint=joint, gate=gate),
    )
    return gate


def _failure_decision(
    *,
    stage: str,
    failure_domain: str,
    failure_code: str,
    message: str,
) -> dict[str, Any]:
    if "provenance" in failure_domain:
        decision = "control_provenance_invalid"
    elif "resource" in failure_domain or "comput" in failure_domain:
        decision = "physical_coarse_signal_computationally_infeasible"
    elif any(
        token in failure_domain
        for token in ("numerical", "certificate", "transition")
    ):
        decision = "physical_coarse_signal_numerically_unresolved"
    elif stage in {"panel-a", "panel-b"}:
        decision = "physical_coarse_signal_panel_integrity_invalid"
    elif stage == "analyze":
        decision = "physical_coarse_signal_estimator_invalid"
    else:
        decision = "physical_coarse_signal_preflight_invalid"
    return {
        "schema": RUN_SCHEMA + "-decision",
        "schema_version": 1,
        "evaluation_status": "execution_failed",
        "decision": decision,
        "failure_domain": failure_domain,
        "failure_code": failure_code,
        "message": message,
        "scientific_evidence_complete": 0,
        **CLAIM_FLAGS,
        **NO_WORK,
    }


def _record_failure_attempt(
    run_dir: Path,
    *,
    stage: str,
    gate: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> Path:
    directory = run_dir / "failure_attempts"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"{stage.replace('-', '_')}-attempt-"
    indices: list[int] = []
    for path in directory.glob(prefix + "*.json"):
        suffix = path.stem.removeprefix(prefix)
        if suffix.isdigit():
            indices.append(int(suffix))
    index = max(indices, default=0) + 1
    path = directory / f"{prefix}{index:03d}.json"
    _freeze_json(
        path,
        {
            "schema": RUN_SCHEMA + "-failure-attempt",
            "schema_version": 1,
            "attempt_index": index,
            "stage": stage,
            "gate": dict(gate),
            "decision": dict(decision),
            "scientific_evidence_complete": 0,
            **NO_WORK,
        },
    )
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight", "panel-a", "panel-b", "analyze", "report", "all"),
        default="all",
    )
    parser.add_argument(
        "--require-gate",
        choices=("none", "preflight", "panel-a", "panel-b", "witness"),
        default="none",
    )
    parser.add_argument("--parent-one-image-run-dir", required=True)
    parser.add_argument("--parent-zero-signal-run-dir", required=True)
    parser.add_argument("--parent-bayes-power-run-dir", required=True)
    parser.add_argument("--resume-run-dir")
    parser.add_argument(
        "--runs-root",
        default="runs/experiment12_d0_jacobi_rb_physical_coarse_signal_witness",
    )
    parser.add_argument(
        "--run-name", default="production-exact-k512-physical-coarse-signal"
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> int:
    run_dir: Path | None = None
    resumed = False
    active_stage = (
        "preflight" if str(args.stage) == "all" else str(args.stage)
    )
    try:
        run_dir, resumed = _make_run_dir(args)
        _manifest_record, scientific, _sources = _initialize_or_validate(
            run_dir, args=args, resumed=resumed
        )
        print(f"physical coarse-signal run directory: {run_dir}", flush=True)
        stage = str(args.stage)
        if stage == "report":
            return _finalize(
                run_dir, stage=stage, required_gate=str(args.require_gate)
            )
        if stage in {"preflight", "all"}:
            active_stage = "preflight"
            _status(
                run_dir,
                stage="preflight",
                state="running",
                decision="ready_for_preflight",
            )
            preflight = _preflight_stage(
                run_dir, args=args, scientific=scientific
            )
            if int(preflight.get("passed", 0)) != 1:
                return _finalize(
                    run_dir,
                    stage="preflight",
                    required_gate=(
                        "preflight"
                        if args.require_gate != "none"
                        else "none"
                    ),
                    message="preflight did not authorize panel generation",
                )
        if stage in {"panel-a", "all"}:
            active_stage = "panel-a"
            _status(
                run_dir,
                stage="panel-a",
                state="running",
                decision="ready_for_panel_a",
            )
            panel_a = _panel_stage(
                run_dir, panel="a", args=args, scientific=scientific
            )
            if int(panel_a.get("passed", 0)) != 1:
                return _finalize(
                    run_dir,
                    stage="panel-a",
                    required_gate=(
                        "panel-a" if args.require_gate != "none" else "none"
                    ),
                    message="panel A failed integrity or resource gates",
                )
        if stage in {"panel-b", "all"}:
            active_stage = "panel-b"
            _status(
                run_dir,
                stage="panel-b",
                state="running",
                decision="ready_for_panel_b",
            )
            panel_b = _panel_stage(
                run_dir, panel="b", args=args, scientific=scientific
            )
            if int(panel_b.get("passed", 0)) != 1:
                return _finalize(
                    run_dir,
                    stage="panel-b",
                    required_gate=(
                        "panel-b" if args.require_gate != "none" else "none"
                    ),
                    message="panel B failed integrity or resource gates",
                )
        if stage in {"analyze", "all"}:
            active_stage = "analyze"
            _status(
                run_dir,
                stage="analyze",
                state="running",
                decision="ready_for_witness_analysis",
            )
            _analysis_stage(run_dir)
        return _finalize(
            run_dir, stage=stage, required_gate=str(args.require_gate)
        )
    except (ArtifactCompatibilityError, ParentArtifactCompatibilityError) as exc:
        # A resumed run is immutable until every old binding has verified.
        if run_dir is None or resumed:
            print(f"physical coarse-signal compatibility error: {exc}", file=sys.stderr)
            return 1
        failure_domain = "control_provenance"
        failure_code = "control_provenance_invalid"
        message = str(exc)
    except (CoarseWitnessCLIError, PhysicalPanelError) as exc:
        if run_dir is None:
            print(f"physical coarse-signal error: {exc}", file=sys.stderr)
            return 1
        failure_domain = getattr(exc, "failure_domain", "workflow_execution")
        failure_code = getattr(
            exc, "failure_code", "physical_coarse_signal_execution_failed"
        )
        message = str(exc)
    except RigorousCudaControlError as exc:
        if run_dir is None:
            print(f"physical coarse-signal error: {exc}", file=sys.stderr)
            return 1
        failure_domain = "exact_transition_certificate_numerical"
        failure_code = "certified_jacobi_transition_failed"
        message = str(exc)
    except Exception as exc:  # evidence is committed; programming errors still fail closed
        if run_dir is None:
            raise
        failure_domain = "workflow_execution"
        failure_code = "physical_coarse_signal_unexpected_failure"
        message = f"{type(exc).__name__}: {exc}"

    assert run_dir is not None
    stage = active_stage
    gate_name = {
        "preflight": "preflight",
        "panel-a": "panel-a",
        "panel-b": "panel-b",
        "analyze": "witness",
        "report": "witness",
    }[stage]
    failure_gate = execution_failed_gate(
        gate_name,
        failure_code=failure_code,
        failure_domain=failure_domain,
        message=message,
    )
    _save_gate(run_dir, gate_name, failure_gate)
    atomic_write_json(
        run_dir / f"{stage.replace('-', '_')}_failure.json",
        {
            "schema": RUN_SCHEMA + "-failure",
            "schema_version": 1,
            "stage": stage,
            "failure_domain": failure_domain,
            "failure_code": failure_code,
            "message": message,
            "evaluation_status": "execution_failed",
            "scientific_evidence_complete": 0,
            **NO_WORK,
        },
    )
    decision = _failure_decision(
        stage=stage,
        failure_domain=failure_domain,
        failure_code=failure_code,
        message=message,
    )
    _record_failure_attempt(
        run_dir,
        stage=stage,
        gate=failure_gate,
        decision=decision,
    )
    atomic_write_json(run_dir / "physical_coarse_signal_decision.json", decision)
    registry = _artifact_registry(run_dir)
    _status(
        run_dir,
        stage=stage,
        state="gate_failed",
        decision=str(decision["decision"]),
        message=message,
        registry=registry,
    )
    print(f"physical coarse-signal error: {message}", file=sys.stderr)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
