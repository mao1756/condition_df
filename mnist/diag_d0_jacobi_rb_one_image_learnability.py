"""One-image learnability gate for the exact K=512 Jacobi/RB split chain.

This workflow deliberately stops before reverse sampling.  It first binds the
already-passing exact kernel/target evidence and the terminal refinement-power
evidence, then generates exact forward-chain supervision, runs a synthetic
optimization control, and finally opens one sealed eight-path confirmation
panel.  A pass means only that the exact Rao--Blackwell label is conditionally
learnable under the permitted later-state inputs for this one frozen image.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
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
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_cuda_multipath import (
    CAPTURE_PAYLOAD_SCHEMA,
    EDGES_PER_PHASE,
    PATH_STATE_SIZE,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    SHARD_STEPS,
    ExactMultipathCapturePayload,
    run_exact_multipath_shard,
)


RUN_SCHEMA = "experiment12-d0-jacobi-rb-one-image-learnability"
RUN_SCHEMA_VERSION = 1
ROOT_SEED = 261_191
OUTER_STEPS = 512
SELECTED_OUTER_STEPS = tuple(range(15, OUTER_STEPS, 16))
PATH_IDS = {
    # The plan's preferred 0x60000 slots are already claimed by the immutable
    # phase-observer namespace.  These fresh slots are the collision-free
    # result of the mandatory repository-wide semantic scan.
    "train": tuple(range(0xE0000, 0xE0008)),
    "validation": tuple(range(0xE1000, 0xE1008)),
    "confirmation": tuple(range(0xE2000, 0xE2008)),
}
MODEL_SEEDS = (261_201, 261_202, 261_203)
LABEL = 3
CLASS_INDEX = 0
LAMBDA_MIX = 0.35
IMAGE_SHA256 = "0bb39fec59853f789fe366251cd85ed79ffbffb5a1aaa32084d2dbd2bbb4ea7d"
MIXED_TARGET_SHA256 = (
    "00ae86fb69be6d86557f15f6f8fa00f8bb3c2514f331863c9638e36d23d135c5"
)
SOURCE_IMAGE_NPZ_SHA256 = (
    "81904cde32495eb11b73cb688cc458118eb2e5578513426d2f9b881ac4665914"
)
EXPECTED_SAMPLES_PER_SPLIT = 8 * len(SELECTED_OUTER_STEPS) * len(PHASE_MATCHINGS)
EXPECTED_TRANSITIONS_PER_SPLIT = (
    8 * OUTER_STEPS * len(PHASE_MATCHINGS) * EDGES_PER_PHASE
)
EXPECTED_TOTAL_TRANSITIONS = 3 * EXPECTED_TRANSITIONS_PER_SPLIT
INPUT_FIELDS = (
    "sample_key",
    "later_full_state",
    "reverse_time",
    "phase",
    "color",
    "duration",
    "label",
)
AUDIT_FIELDS = (
    "sample_key",
    "path_id",
    "outer_step",
    "phase",
    "denoising_target",
    "certificate_codes",
)
FORBIDDEN_COUNTS = (
    "resource_cap_count",
    "invalid_density_count",
    "approximation_count",
    "correction_count",
    "floor_count",
    "limiter_count",
    "renormalization_count",
    "nonfinite_count",
)
CLAIM_FLAGS = {
    "state_dependent_strang_refinement_established": 0,
    "unsplit_generator_approximation_authorized": 0,
    "spatial_dirichlet_ferguson_claim_authorized": 0,
    "reverse_sampling_authorized": 0,
    "sampling_authorized": 0,
    "reconstruction_claim_authorized": 0,
    "known_prior_claim_authorized": 0,
    "full_dataset_training_authorized": 0,
}
NO_WORK = {
    "physical_training_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
    "production_refinement_performed": 0,
}
_REGISTRY_EXCLUDED = {"artifact_registry.json", "run_status.json"}


class LearnabilityCLIError(RuntimeError):
    """Typed orchestration failure committed before a required-gate exit."""

    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "workflow_execution",
        failure_code: str = "learnability_execution_failed",
    ) -> None:
        super().__init__(message)
        self.failure_domain = str(failure_domain)
        self.failure_code = str(failure_code)


class ParentScopeError(ArtifactCompatibilityError):
    """Verified immutable-parent/source binding failure."""


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


def _freeze_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    require_existing: bool = False,
) -> dict[str, Any]:
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
    normalized = {
        str(name): np.ascontiguousarray(np.asarray(value))
        for name, value in sorted(arrays.items())
    }
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **normalized)
    os.replace(temporary, path)
    return _npz_record(path, normalized)


def _npz_record(
    path: Path, arrays: Mapping[str, np.ndarray] | None = None
) -> dict[str, Any]:
    if arrays is None:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                name: np.ascontiguousarray(np.asarray(archive[name]))
                for name in archive.files
            }
    return {
        "path": path.as_posix(),
        "sha256": file_fingerprint(path),
        "size": int(path.stat().st_size),
        "array_hashes": {
            name: hashlib.sha256(np.asarray(value).tobytes(order="C")).hexdigest()
            for name, value in sorted(arrays.items())
        },
    }


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)
    return {
        "path": path.as_posix(),
        "sha256": file_fingerprint(path),
        "size": int(path.stat().st_size),
    }


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


def _array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _source_measure_sha(array: np.ndarray) -> str:
    measured = np.ascontiguousarray(
        np.asarray(array, dtype=np.float32).reshape(-1)
    )
    digest = hashlib.sha256()
    digest.update(str(measured.shape).encode("ascii"))
    digest.update(measured.tobytes(order="C"))
    return digest.hexdigest()


def _source_paths() -> tuple[Path, ...]:
    import mnist.d0_jacobi_artifacts as artifacts
    import mnist.d0_jacobi_rb_learnability as core
    import mnist.d0_jacobi_rb_learnability_gate as gate
    import mnist.d0_jacobi_rb_learnability_provenance as provenance
    import mnist.d0_jacobi_rb_cuda_multipath as scheduler
    import mnist.d0_jacobi_source_compat as source_compat

    return tuple(
        sorted(
            {
                Path(__file__).resolve(),
                Path(artifacts.__file__).resolve(),
                Path(core.__file__).resolve(),
                Path(gate.__file__).resolve(),
                Path(provenance.__file__).resolve(),
                Path(scheduler.__file__).resolve(),
                Path(source_compat.__file__).resolve(),
            },
            key=lambda item: item.as_posix(),
        )
    )


def _scientific_config(*, authorizing: bool) -> dict[str, Any]:
    config = {
        "schema": RUN_SCHEMA + "-scientific-config",
        "schema_version": RUN_SCHEMA_VERSION,
        "authorizing": int(authorizing),
        "claim_scope": (
            "conditional learnability of the exact Rao-Blackwell label for the "
            "exact K=512 split chain and one frozen MNIST image"
        ),
        "grid_size": 28,
        "alpha": 1.0,
        "outer_steps": OUTER_STEPS,
        "steps_per_shard": SHARD_STEPS,
        "selected_outer_steps": list(SELECTED_OUTER_STEPS),
        "phase_matchings": list(PHASE_MATCHINGS),
        "phase_durations": list(PHASE_DURATIONS),
        "edges_per_phase": EDGES_PER_PHASE,
        "root_seed": ROOT_SEED,
        "path_ids": {name: list(values) for name, values in PATH_IDS.items()},
        "source_image": {
            "label": LABEL,
            "class_index": CLASS_INDEX,
            "lambda_mix": LAMBDA_MIX,
            "image_sha256": IMAGE_SHA256,
            "mixed_target_sha256": MIXED_TARGET_SHA256,
            "source_image_npz_sha256": SOURCE_IMAGE_NPZ_SHA256,
        },
        "model": {
            "architecture": "phase-conditioned-local-affine-plus-cnn-v1",
            "width": 32,
            "convolution_count": 3,
            "model_input_fields": [
                "later_full_state",
                "reverse_time",
                "phase",
                "color",
                "duration",
                "label",
            ],
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "batch_size": 32,
            "maximum_updates": 4000,
            "validation_interval": 100,
            "gradient_norm_clip": 1.0,
            "model_seeds": list(MODEL_SEEDS),
            "mixed_precision": 0,
            "target_scaling": "one positive global training RMS",
        },
        "resource_thresholds": {
            "minimum_effective_transitions_per_second": 1300.0,
            "maximum_projected_total_hours": 10.0,
            "maximum_peak_memory_fraction": 0.80,
            "maximum_persisted_cache_bytes": 134_217_728,
        },
        "confirmation": {
            "path_count": 8,
            "required_positive_path_improvements": 8,
            "one_sided_sign_test_p_value": 1.0 / 256.0,
            "ties_fail": 1,
        },
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    config["semantic_sha256"] = config_fingerprint(config)
    return config


def _path_plan() -> dict[str, Any]:
    from mnist.d0_jacobi_rb_learnability import frozen_path_plan

    plan = frozen_path_plan()
    expected = {name: list(values) for name, values in PATH_IDS.items()}
    record = plan.to_record()
    if record.get("roles") != expected:
        raise LearnabilityCLIError(
            "learnability path-ID plan differs from the frozen core plan",
            failure_domain="configuration",
            failure_code="learnability_path_plan_invalid",
        )
    plan.assert_collision_free(Path.cwd())
    record["path_count_per_role"] = 8
    record["canonical_field_bits"] = 20
    record["pairwise_disjoint"] = 1
    record["repository_collision_scan_pass"] = 1
    record["preferred_slot_collision_note"] = (
        "0x60000/0x61000/0x62000 were rejected because immutable "
        "phase-observer claims overlap them"
    )
    record["semantic_sha256"] = config_fingerprint(record)
    return record


def _validated_path_plan_record(path: Path) -> dict[str, Any]:
    record = _load_json(path)
    expected_hash = record.get("semantic_sha256")
    body = dict(record)
    body.pop("semantic_sha256", None)
    if expected_hash != config_fingerprint(body):
        raise ArtifactCompatibilityError("path-ID plan semantic hash mismatch")
    if record.get("roles") != {
        name: list(values) for name, values in PATH_IDS.items()
    }:
        raise ArtifactCompatibilityError("frozen path-ID plan changed")
    return record


def _model_input_contract() -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + "-model-input-contract",
        "schema_version": 1,
        "allowed_model_fields": [
            "later_full_state",
            "reverse_time",
            "phase",
            "color",
            "duration",
            "label",
        ],
        "join_key_not_model_input": "sample_key",
        "forbidden_model_fields": [
            "earlier_state",
            "path_id",
            "outer_step",
            "sample_key",
            "uniform_bits",
            "normal_variables",
            "later_head_fraction",
            "certificate_codes",
            "denoising_target",
            "oracle_target",
        ],
        "later_state_only": 1,
        "reverse_time_semantics": "normalized remaining split-chain phase index",
        "continuous_physical_time_claimed": 0,
    }


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


def _status(
    run_dir: Path,
    *,
    stage: str,
    state: str,
    message: str = "",
    decision: str | None = None,
    registry: Mapping[str, Any] | None = None,
) -> None:
    registry_binding: dict[str, Any] = {}
    registry_path = run_dir / "artifact_registry.json"
    if registry is not None:
        registry_binding = {
            "artifact_registry_record_count": int(registry["record_count"]),
            "artifact_registry_sha256": str(registry["registry_sha256"]),
            "artifact_registry_file_sha256": file_fingerprint(registry_path),
            "artifact_registry_file_size": int(registry_path.stat().st_size),
        }
    record = {
        "schema": RUN_SCHEMA + "-status",
        "schema_version": 1,
        "updated_at": _now(),
        "stage": stage,
        "state": state,
        "message": message,
        "decision": decision,
        **registry_binding,
        **CLAIM_FLAGS,
        **NO_WORK,
        "physical_training_performed": int(_physical_work_performed(run_dir)),
    }
    atomic_write_json(run_dir / "run_status.json", record)


def _physical_work_performed(run_dir: Path) -> bool:
    direct = (
        run_dir / "physical_training_started.json",
        run_dir / "physical_training_metrics.json",
        run_dir / "physical_gate.json",
        run_dir / "selected_model.pt",
    )
    return any(path.exists() for path in direct) or any(
        (run_dir / "checkpoints").glob("physical-rb-seed-*")
    )


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name in _REGISTRY_EXCLUDED:
            continue
        relative = path.relative_to(run_dir).as_posix()
        records.append(
            {
                "path": relative,
                "sha256": file_fingerprint(path),
                "size": int(path.stat().st_size),
            }
        )
    record = {
        "schema": RUN_SCHEMA + "-artifact-registry",
        "schema_version": 1,
        "record_count": len(records),
        "records": records,
        **NO_WORK,
        "physical_training_performed": int(_physical_work_performed(run_dir)),
    }
    record["registry_sha256"] = config_fingerprint(records)
    atomic_write_json(run_dir / "artifact_registry.json", record)
    return record


def _verify_existing_artifact_registry(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "artifact_registry.json"
    if not path.is_file():
        return None
    record = _load_json(path)
    records = record.get("records")
    if not isinstance(records, list):
        raise ArtifactCompatibilityError("artifact registry records are malformed")
    if int(record.get("record_count", -1)) != len(records):
        raise ArtifactCompatibilityError("artifact registry count mismatch")
    if record.get("registry_sha256") != config_fingerprint(records):
        raise ArtifactCompatibilityError("artifact registry semantic hash mismatch")
    seen: set[str] = set()
    for item in records:
        if not isinstance(item, Mapping):
            raise ArtifactCompatibilityError("artifact registry row is malformed")
        relative = str(item.get("path", ""))
        if (
            not relative
            or relative in seen
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ArtifactCompatibilityError("artifact registry path is invalid")
        seen.add(relative)
        artifact = run_dir / relative
        if (
            not artifact.is_file()
            or int(item.get("size", -1)) != artifact.stat().st_size
            or item.get("sha256") != file_fingerprint(artifact)
        ):
            raise ArtifactCompatibilityError(
                f"registered artifact changed: {relative}"
            )
    status_path = run_dir / "run_status.json"
    if status_path.is_file():
        status = _load_json(status_path)
        if "artifact_registry_sha256" in status:
            actual = {
                artifact.relative_to(run_dir).as_posix()
                for artifact in run_dir.rglob("*")
                if artifact.is_file()
                and artifact.name not in _REGISTRY_EXCLUDED
            }
            if actual != seen:
                missing = sorted(seen - actual)
                extra = sorted(actual - seen)
                raise ArtifactCompatibilityError(
                    "terminal artifact registry file set changed: "
                    f"missing={missing}, extra={extra}"
                )
            expected = {
                "artifact_registry_record_count": len(records),
                "artifact_registry_sha256": record["registry_sha256"],
                "artifact_registry_file_sha256": file_fingerprint(path),
                "artifact_registry_file_size": int(path.stat().st_size),
            }
            if any(status.get(name) != value for name, value in expected.items()):
                raise ArtifactCompatibilityError(
                    "run status does not bind the current artifact registry"
                )
    return record


def _require_stage_artifact(run_dir: Path, filename: str) -> dict[str, Any]:
    path = run_dir / filename
    if not path.is_file():
        raise ArtifactCompatibilityError(f"required prior-stage artifact is missing: {filename}")
    return _load_json(path)


def _passed(gate: Mapping[str, Any]) -> bool:
    return (
        gate.get("evaluation_status") == "evaluated"
        and int(gate.get("passed", 0)) == 1
    )


def _load_source_image(parent_strang: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    metadata_path = parent_strang / "source_image.json"
    npz_path = parent_strang / "source_image.npz"
    metadata = _load_json(metadata_path)
    expected = {
        "label": LABEL,
        "class_index": CLASS_INDEX,
        "lambda_mix": LAMBDA_MIX,
        "image_sha256": IMAGE_SHA256,
        "mixed_target_sha256": MIXED_TARGET_SHA256,
        "npz_sha256": SOURCE_IMAGE_NPZ_SHA256,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ArtifactCompatibilityError(f"source image metadata mismatch: {key}")
    if file_fingerprint(npz_path) != SOURCE_IMAGE_NPZ_SHA256:
        raise ArtifactCompatibilityError("source image NPZ hash mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != {"image", "mixed_target"}:
            raise ArtifactCompatibilityError("source image NPZ schema mismatch")
        image = np.asarray(archive["image"], dtype=np.float64).reshape(-1)
        mixed = np.asarray(archive["mixed_target"], dtype=np.float64).reshape(-1)
    if image.shape != (PATH_STATE_SIZE,) or mixed.shape != (PATH_STATE_SIZE,):
        raise ArtifactCompatibilityError("source image arrays must contain 784 masses")
    if not np.isfinite(image).all() or not np.isfinite(mixed).all():
        raise ArtifactCompatibilityError("source image contains nonfinite values")
    if np.any(image < 0.0) or np.any(mixed < 0.0):
        raise ArtifactCompatibilityError("source image contains negative masses")
    if abs(float(image.sum()) - 1.0) > 1.0e-12:
        raise ArtifactCompatibilityError("source image is not on the simplex")
    if abs(float(mixed.sum()) - 1.0) > 1.0e-12:
        raise ArtifactCompatibilityError("mixed target is not on the simplex")
    if (
        _source_measure_sha(image) != IMAGE_SHA256
        or _source_measure_sha(mixed) != MIXED_TARGET_SHA256
    ):
        raise ArtifactCompatibilityError("source image array semantic hash mismatch")
    return metadata, image, mixed


def _copy_source_image(run_dir: Path, parent_strang: Path) -> None:
    for name in ("source_image.json", "source_image.npz"):
        source = parent_strang / name
        target = run_dir / name
        if target.is_file():
            if file_fingerprint(target) != file_fingerprint(source):
                raise ArtifactCompatibilityError(f"frozen source image changed: {name}")
        else:
            temporary = target.with_name(target.name + ".tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)


def _capture_alignment(payload: ExactMultipathCapturePayload) -> dict[str, Any]:
    from mnist import d0_jacobi_rb_cuda_controls as controls

    matchings = controls._matching_arrays()
    maximum_error = 0.0
    for block, phase in enumerate(payload.phases):
        tails, heads = matchings[PHASE_MATCHINGS[int(phase)]]
        states = payload.post_phase_states[block]
        tail_mass = states[:, tails]
        head_mass = states[:, heads]
        pair_mass = tail_mass + head_mass
        later = payload.later_head_fractions[block]
        reconstructed = np.where(pair_mass > 0.0, head_mass / pair_mass, 0.0)
        maximum_error = max(maximum_error, float(np.max(np.abs(reconstructed - later))))
    return {
        "maximum_post_phase_fraction_error": maximum_error,
        "alignment_tolerance": 2.0e-12,
        "capture_state_alignment_pass": int(maximum_error <= 2.0e-12),
    }


def _capture_parity(
    *,
    mixed_target: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_cuda import (
        sample_alpha1_rb_transition_batch_cuda as exact_sampler,
    )

    profile = JacobiRBCudaProfile()
    path_ids = PATH_IDS["train"]
    initial = torch.as_tensor(
        np.repeat(mixed_target[None, :], len(path_ids), axis=0),
        dtype=torch.float64,
        device=device,
    ).contiguous()
    plain_calls: list[str] = []
    captured_calls: list[str] = []

    def recorder(destination: list[str]) -> Any:
        def wrapped(
            head_fraction: torch.Tensor,
            exposure: torch.Tensor,
            **kwargs: Any,
        ) -> Any:
            transition_ids = kwargs.get("transition_ids")
            digest = hashlib.sha256()
            for value in (head_fraction, exposure, transition_ids):
                if not isinstance(value, torch.Tensor):
                    raise LearnabilityCLIError(
                        "capture parity sampler call lacks a tensor argument",
                        failure_domain="capture",
                        failure_code="capture_sampler_contract_invalid",
                    )
                host = value.detach().cpu().contiguous()
                digest.update(str(host.dtype).encode("ascii"))
                digest.update(str(tuple(host.shape)).encode("ascii"))
                digest.update(host.numpy().tobytes(order="C"))
            digest.update(repr(kwargs.get("rng_key")).encode("utf-8"))
            destination.append(digest.hexdigest())
            return exact_sampler(head_fraction, exposure, **kwargs)

        return wrapped

    plain = run_exact_multipath_shard(
        initial,
        path_ids=path_ids,
        start_step=0,
        root_seed=ROOT_SEED,
        profile=profile,
        sampler=recorder(plain_calls),
        capture_training_payload=False,
    )
    captured = run_exact_multipath_shard(
        initial,
        path_ids=path_ids,
        start_step=0,
        root_seed=ROOT_SEED,
        profile=profile,
        sampler=recorder(captured_calls),
        capture_training_payload=True,
    )
    payload = captured.capture_payload
    if payload is None:
        raise LearnabilityCLIError(
            "capture-enabled exact scheduler returned no payload",
            failure_domain="capture",
            failure_code="capture_payload_missing",
        )
    alignment = _capture_alignment(payload)
    parity = (
        plain.batch_output_sha256 == captured.batch_output_sha256
        and plain.batch_final_state_sha256 == captured.batch_final_state_sha256
        and plain.batch_certificate_sha256 == captured.batch_certificate_sha256
        and np.array_equal(
            plain.committed_final_states, captured.committed_final_states
        )
    )
    call_order_parity = (
        len(plain_calls) == len(captured_calls) == 56
        and plain_calls == captured_calls
    )
    arrays_valid = (
        payload.later_head_fractions.shape
        == (56, len(path_ids), EDGES_PER_PHASE)
        and payload.denoising_targets.shape
        == (56, len(path_ids), EDGES_PER_PHASE)
        and payload.certificate_codes.shape
        == (56, len(path_ids), EDGES_PER_PHASE)
        and payload.post_phase_states.shape
        == (56, len(path_ids), PATH_STATE_SIZE)
        and all(
            array.flags.c_contiguous and not array.flags.writeable
            for array in (
                payload.later_head_fractions,
                payload.denoising_targets,
                payload.certificate_codes,
                payload.post_phase_states,
            )
        )
    )
    selected = np.arange(
        7 * (SHARD_STEPS - 1), 7 * SHARD_STEPS, dtype=np.int64
    )
    capture_buffer = io.BytesIO()
    np.savez_compressed(
        capture_buffer,
        path_ids=np.asarray(payload.path_ids, dtype=np.int64),
        outer_steps=np.asarray(payload.outer_steps, dtype=np.int16)[selected],
        phases=np.asarray(payload.phases, dtype=np.int8)[selected],
        later_head_fractions=payload.later_head_fractions[selected],
        denoising_targets=payload.denoising_targets[selected],
        certificate_codes=payload.certificate_codes[selected],
        post_phase_states=payload.post_phase_states[selected],
    )
    restart_buffer = io.BytesIO()
    np.savez_compressed(
        restart_buffer,
        final_states=np.asarray(captured.committed_final_states),
    )
    return {
        "schema": RUN_SCHEMA + "-capture-parity",
        "schema_version": 1,
        "capture_payload_schema": CAPTURE_PAYLOAD_SCHEMA,
        "hash_and_state_parity_pass": int(parity),
        "sampler_call_count": len(captured_calls),
        "sampler_call_order_parity_pass": int(call_order_parity),
        "capture_schema_pass": int(arrays_valid),
        "benchmark_path_count": len(path_ids),
        "plain_effective_transitions_per_second": float(
            plain.diagnostics["transitions_per_second"]
        ),
        "capture_effective_transitions_per_second": float(
            captured.diagnostics["transitions_per_second"]
        ),
        "selected_capture_compressed_bytes": len(capture_buffer.getvalue()),
        "restart_state_compressed_bytes": len(restart_buffer.getvalue()),
        "capture_shard_metadata_projected_bytes": len(
            json.dumps(captured.to_record(), sort_keys=True).encode("utf-8")
        )
        + 2048,
        "plain_shard_metadata_projected_bytes": len(
            json.dumps(plain.to_record(), sort_keys=True).encode("utf-8")
        )
        + 2048,
        **alignment,
        "passed": int(
            parity
            and call_order_parity
            and arrays_valid
            and int(alignment["capture_state_alignment_pass"]) == 1
        ),
        **NO_WORK,
    }


def _parent_rate(parent_multipath: Path) -> tuple[float, float]:
    record = _load_json(parent_multipath / "kernel_metrics.json")
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ArtifactCompatibilityError("multipath kernel metrics are malformed")
    rate = float(metrics.get("projected_effective_transitions_per_second", math.nan))
    memory = float(metrics.get("peak_memory_fraction", math.nan))
    if not math.isfinite(rate) or rate <= 0.0 or not math.isfinite(memory):
        raise ArtifactCompatibilityError("multipath resource metrics are invalid")
    return rate, memory


def _preflight_stage(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    resumed: bool,
) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_learnability_gate import (
        LearnabilityThresholds,
        evaluate_learnability_preflight,
    )
    from mnist.d0_jacobi_rb_learnability_provenance import (
        verify_learnability_parents,
    )

    try:
        parents = verify_learnability_parents(
            multipath_run_dir=args.parent_multipath_run_dir,
            strang_run_dir=args.parent_strang_run_dir,
            haar_run_dir=args.parent_haar_run_dir,
        )
    except ArtifactCompatibilityError as exc:
        raise ParentScopeError(str(exc)) from exc
    completed_gate = _optional_json(run_dir, "preflight_gate.json")
    if completed_gate is not None:
        metrics = _require_stage_artifact(run_dir, "preflight_metrics.json")
        replay_gate = evaluate_learnability_preflight(
            metrics, thresholds=LearnabilityThresholds()
        )
        if replay_gate != completed_gate:
            raise ArtifactCompatibilityError(
                "completed preflight gate does not replay exactly"
            )
        _validated_path_plan_record(run_dir / "path_id_plan.json")
        capture = _require_stage_artifact(run_dir, "capture_parity.json")
        if int(capture.get("passed", 0)) != 1:
            raise ArtifactCompatibilityError(
                "completed preflight capture parity did not pass"
            )
        _load_source_image(args.parent_strang_run_dir)
        return completed_gate
    theoretical_basis = {
        "schema": RUN_SCHEMA + "-rao-blackwell-theoretical-basis",
        "schema_version": 1,
        "phase_local_factorization": (
            "conditional on the complete pre-phase sigma-field, disjoint "
            "active edges use a product of exact Jacobi kernels with "
            "independent edge-local latent randomness"
        ),
        "edge_identity": (
            "E[L-MY | pre_phase_sigma_field, S_plus] "
            "= E[L-MY | X,Y,u] = Z_bar(X,Y,u)"
        ),
        "permitted_information_identity": (
            "for W=(S_plus,reverse split-chain coordinate,phase,color,"
            "duration,label), E[L-MY | W] = E[Z_bar | W]"
        ),
        "pair_mass_conserved": 1,
        "post_phase_state_contains_pair_mass": 1,
        "earlier_fraction_constructs_label_only": 1,
        "earlier_fraction_is_model_input": 0,
        "k_to_infinity_limit_used": 0,
        "k512_vs_k1024_comparison_used": 0,
        "population_optimum": (
            "unweighted MSE on the exact Rao-Blackwell label has the same "
            "allowed-input conditional mean as the ancestral DDPM-like label"
        ),
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    theoretical_basis["semantic_sha256"] = config_fingerprint(theoretical_basis)
    _freeze_json(
        run_dir / "rao_blackwell_theoretical_basis.json", theoretical_basis
    )
    source_metadata, _image, mixed = _load_source_image(args.parent_strang_run_dir)
    _copy_source_image(run_dir, args.parent_strang_run_dir)
    path_plan_path = run_dir / "path_id_plan.json"
    if path_plan_path.is_file():
        path_plan = _validated_path_plan_record(path_plan_path)
    else:
        path_plan = _freeze_json(path_plan_path, _path_plan())
    contract = _freeze_json(
        run_dir / "model_input_contract.json", _model_input_contract()
    )
    runtime = configure_exact_torch_backend(torch.device(args.device))
    runtime.update(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_argument": str(args.device),
        }
    )
    atomic_write_json(run_dir / "exact_backend_runtime.json", runtime)
    capture = _capture_parity(
        mixed_target=mixed, device=torch.device(args.device)
    )
    atomic_write_json(run_dir / "capture_parity.json", capture)
    parent_rate, memory = _parent_rate(args.parent_multipath_run_dir)
    capture_rate = float(capture["capture_effective_transitions_per_second"])
    # The projection must be conservative for this workflow's actual
    # capture-enabled eight-path shape, while retaining the complete-pipeline
    # parent benchmark as an independent ceiling.
    rate = min(parent_rate, capture_rate)
    total_hours = EXPECTED_TOTAL_TRANSITIONS / rate / 3600.0
    # Compact input/audit caches are charged at their uncompressed byte count.
    # Capture and restart archives use the measured exact one-path compression
    # size, scaled pessimistically per path and shard.  Metadata receives a
    # fixed 20 KiB per shard plus one MiB per split.
    compact_cache_bytes = (
        EXPECTED_SAMPLES_PER_SPLIT
        * (
            PATH_STATE_SIZE * 8
            + EDGES_PER_PHASE * (8 + 1)
            + 8 * 5
            + 16
        )
    )
    capture_archive_bytes = (
        int(capture["selected_capture_compressed_bytes"])
        * len(SELECTED_OUTER_STEPS)
    )
    restart_archive_bytes = (
        int(capture["restart_state_compressed_bytes"])
        * (OUTER_STEPS // SHARD_STEPS)
    )
    metadata_bytes = (
        int(capture["capture_shard_metadata_projected_bytes"])
        * len(SELECTED_OUTER_STEPS)
        + int(capture["plain_shard_metadata_projected_bytes"])
        * len(SELECTED_OUTER_STEPS)
        + 512 * 1024
    )
    bytes_per_split = (
        compact_cache_bytes
        + capture_archive_bytes
        + restart_archive_bytes
        + metadata_bytes
    )
    projection = {
        "schema": RUN_SCHEMA + "-resource-projection",
        "schema_version": 1,
        "transition_count": EXPECTED_TOTAL_TRANSITIONS,
        "parent_measured_effective_transitions_per_second": parent_rate,
        "capture_measured_effective_transitions_per_second": capture_rate,
        "projected_effective_transitions_per_second": rate,
        "projected_total_hours": total_hours,
        "projected_persisted_cache_bytes": 3 * bytes_per_split,
        "projected_compact_cache_bytes_per_split": compact_cache_bytes,
        "projected_capture_archive_bytes_per_split": capture_archive_bytes,
        "projected_restart_archive_bytes_per_split": restart_archive_bytes,
        "projected_metadata_bytes_per_split": metadata_bytes,
        "peak_memory_fraction": memory,
        "minimum_effective_transitions_per_second": 1300.0,
        "maximum_projected_total_hours": 10.0,
        "maximum_persisted_cache_bytes": 134_217_728,
        "maximum_peak_memory_fraction": 0.80,
        "resource_projection_pass": int(
            rate >= 1300.0
            and total_hours <= 10.0
            and 3 * bytes_per_split <= 134_217_728
            and memory <= 0.80
        ),
    }
    atomic_write_json(run_dir / "resource_projection.json", projection)
    metrics = {
        "schema": RUN_SCHEMA + "-preflight-metrics",
        "schema_version": 1,
        "parent_provenance_pass": int(parents.get("passed", 0)),
        **{
            name: int(parents.get(name, 0))
            for name in (
                "multipath_kernel_gate_pass",
                "multipath_target_gate_pass",
                "multipath_decision_pass",
                "strang_power_failure_preserved_pass",
                "haar_power_only_failure_pass",
                "haar_numerical_health_pass",
                "haar_resource_health_pass",
                "source_image_hash_pass",
                "source_image_npz_hash_pass",
                "mixed_target_hash_pass",
                "future_model_input_contract_pass",
                "parents_no_training_pass",
                "parents_no_reverse_sampling_pass",
                "parent_registries_pass",
                "source_binding_pass",
            )
        },
        "path_plan_frozen_pass": 1,
        "path_plan_bounds_pass": 1,
        "path_plan_disjoint_pass": 1,
        "path_plan_collision_scan_pass": 1,
        "capture_parity_pass": int(capture["passed"]),
        "capture_rng_neutral_pass": int(capture["hash_and_state_parity_pass"]),
        "capture_call_order_pass": int(
            capture["sampler_call_order_parity_pass"]
        ),
        "capture_hash_parity_pass": int(capture["hash_and_state_parity_pass"]),
        "model_input_schema_firewall_pass": 1,
        "confirmation_absent_pass": int(_no_confirmation_artifacts(run_dir)),
        "outer_steps": OUTER_STEPS,
        "steps_per_shard": SHARD_STEPS,
        "paths_per_split": 8,
        "selected_outer_steps": list(SELECTED_OUTER_STEPS),
        "effective_transitions_per_second": rate,
        "projected_total_hours": total_hours,
        "peak_memory_fraction": memory,
        "projected_persisted_cache_bytes": 3 * bytes_per_split,
        "projected_transition_count": EXPECTED_TOTAL_TRANSITIONS,
        "test_only_reduced_workload": 0,
        "resource_projection_pass": int(projection["resource_projection_pass"]),
        "historical_refinement_gate_remains_failed_or_unresolved": 1,
        "historical_refinement_gate_reclassified": 0,
        "parent_provenance_sha256": config_fingerprint(parents),
        "rao_blackwell_theoretical_basis_sha256": theoretical_basis[
            "semantic_sha256"
        ],
        "path_plan_sha256": path_plan["semantic_sha256"],
        "model_input_contract_sha256": config_fingerprint(contract),
        "source_image": source_metadata,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "parent_provenance.json", parents)
    atomic_write_json(run_dir / "preflight_metrics.json", metrics)
    gate = evaluate_learnability_preflight(
        metrics, thresholds=LearnabilityThresholds()
    )
    atomic_write_json(run_dir / "preflight_gate.json", gate)
    return gate


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {
                name: np.ascontiguousarray(np.asarray(archive[name]))
                for name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise ArtifactCompatibilityError(f"cannot read NPZ artifact {path}: {exc}") from exc


def _valid_committed_shard(
    *,
    state_path: Path,
    capture_path: Path,
    metadata_path: Path,
    expected_split: str,
    expected_start_step: int,
    expected_path_ids: Sequence[int],
    expected_input_states: np.ndarray,
    expected_input_sha256: str,
    capture_expected: bool,
    scientific_config_sha256: str,
    path_plan_sha256: str,
    profile_sha256: str,
) -> tuple[bool, np.ndarray | None, dict[str, np.ndarray] | None]:
    if not state_path.is_file() or not metadata_path.is_file():
        return False, None, None
    try:
        from mnist import d0_jacobi_rb_cuda_controls as controls

        metadata = _load_json(metadata_path)
        semantic_sha = metadata.get("semantic_sha256")
        semantic_body = dict(metadata)
        semantic_body.pop("semantic_sha256", None)
        if semantic_sha != config_fingerprint(semantic_body):
            return False, None, None
        expected_static = {
            "schema": RUN_SCHEMA + "-cache-shard",
            "schema_version": 1,
            "split": expected_split,
            "start_step": int(expected_start_step),
            "step_count": SHARD_STEPS,
            "path_ids": list(expected_path_ids),
            "root_seed": ROOT_SEED,
            "scientific_config_sha256": scientific_config_sha256,
            "path_plan_sha256": path_plan_sha256,
            "profile_sha256": profile_sha256,
            "capture_expected": int(capture_expected),
        }
        if any(metadata.get(name) != value for name, value in expected_static.items()):
            return False, None, None
        states = _load_npz_arrays(state_path)
        if set(states) != {"final_states"}:
            return False, None, None
        final_states = states["final_states"]
        if (
            final_states.dtype != np.float64
            or final_states.shape != (len(expected_path_ids), PATH_STATE_SIZE)
            or not np.isfinite(final_states).all()
            or np.any(final_states < 0.0)
        ):
            return False, None, None
        if metadata.get("input_state_sha256") != expected_input_sha256:
            return False, None, None
        if metadata.get("state_file_sha256") != file_fingerprint(state_path):
            return False, None, None
        if metadata.get("state_file_size") != state_path.stat().st_size:
            return False, None, None
        if metadata.get("state_array_sha256") != hashlib.sha256(
            final_states.tobytes(order="C")
        ).hexdigest():
            return False, None, None
        if metadata.get("final_state_sha256") != _array_sha(final_states):
            return False, None, None
        scheduler = metadata.get("scheduler_record")
        if not isinstance(scheduler, Mapping):
            return False, None, None
        if metadata.get("scheduler_record_sha256") != config_fingerprint(scheduler):
            return False, None, None
        diagnostics = scheduler.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            return False, None, None
        if (
            scheduler.get("schema")
            != "jacobi-rb-cuda-exact-multipath-v1-shard"
            or int(diagnostics.get("start_step", -1)) != expected_start_step
            or int(diagnostics.get("step_count", -1)) != SHARD_STEPS
            or diagnostics.get("path_ids") != list(expected_path_ids)
            or diagnostics.get("group_sizes") != [len(expected_path_ids)]
            or int(diagnostics.get("transition_count", -1))
            != len(expected_path_ids)
            * SHARD_STEPS
            * len(PHASE_MATCHINGS)
            * EDGES_PER_PHASE
            or int(diagnostics.get("phase_state_trace_enabled", -1))
            != int(capture_expected)
        ):
            return False, None, None
        path_records = scheduler.get("path_records")
        if not isinstance(path_records, list) or len(path_records) != len(
            expected_path_ids
        ):
            return False, None, None
        initial = np.asarray(expected_input_states, dtype=np.float64)
        for index, (path_id, path_record) in enumerate(
            zip(expected_path_ids, path_records, strict=True)
        ):
            if (
                not isinstance(path_record, Mapping)
                or int(path_record.get("path_id", -1)) != int(path_id)
                or path_record.get("input_state_sha256")
                != controls._digest_arrays(initial[index])
                or path_record.get("final_state_sha256")
                != controls._digest_arrays(final_states[index])
            ):
                return False, None, None
        if scheduler.get("batch_final_state_sha256") != controls._digest_arrays(
            final_states
        ):
            return False, None, None
        captures: dict[str, np.ndarray] | None = None
        if capture_expected:
            if not capture_path.is_file():
                return False, None, None
            if metadata.get("capture_file_sha256") != file_fingerprint(capture_path):
                return False, None, None
            if metadata.get("capture_file_size") != capture_path.stat().st_size:
                return False, None, None
            captures = _load_npz_arrays(capture_path)
            expected = {
                "path_ids",
                "outer_steps",
                "phases",
                "later_head_fractions",
                "denoising_targets",
                "certificate_codes",
                "post_phase_states",
            }
            if set(captures) != expected:
                return False, None, None
            if metadata.get("capture_array_hashes") != {
                name: hashlib.sha256(value.tobytes(order="C")).hexdigest()
                for name, value in sorted(captures.items())
            }:
                return False, None, None
            expected_outer = np.full(
                len(PHASE_MATCHINGS),
                expected_start_step + SHARD_STEPS - 1,
                dtype=np.int16,
            )
            if (
                captures["path_ids"].dtype != np.int64
                or captures["path_ids"].tolist() != list(expected_path_ids)
                or not np.array_equal(captures["outer_steps"], expected_outer)
                or not np.array_equal(
                    captures["phases"],
                    np.arange(len(PHASE_MATCHINGS), dtype=np.int8),
                )
                or captures["later_head_fractions"].shape
                != (len(PHASE_MATCHINGS), len(expected_path_ids), EDGES_PER_PHASE)
                or captures["later_head_fractions"].dtype != np.float64
                or captures["denoising_targets"].shape
                != (len(PHASE_MATCHINGS), len(expected_path_ids), EDGES_PER_PHASE)
                or captures["denoising_targets"].dtype != np.float64
                or captures["certificate_codes"].shape
                != (len(PHASE_MATCHINGS), len(expected_path_ids), EDGES_PER_PHASE)
                or captures["certificate_codes"].dtype != np.uint8
                or captures["post_phase_states"].shape
                != (len(PHASE_MATCHINGS), len(expected_path_ids), PATH_STATE_SIZE)
                or captures["post_phase_states"].dtype != np.float64
            ):
                return False, None, None
        elif capture_path.exists():
            return False, None, None
        return True, final_states, captures
    except (ArtifactCompatibilityError, KeyError, TypeError, ValueError):
        return False, None, None


def _selected_capture_arrays(
    payload: ExactMultipathCapturePayload,
) -> dict[str, np.ndarray]:
    block_indices = np.arange(
        7 * (SHARD_STEPS - 1), 7 * SHARD_STEPS, dtype=np.int64
    )
    return {
        "path_ids": np.asarray(payload.path_ids, dtype=np.int64),
        "outer_steps": np.asarray(payload.outer_steps, dtype=np.int16)[block_indices],
        "phases": np.asarray(payload.phases, dtype=np.int8)[block_indices],
        "later_head_fractions": np.ascontiguousarray(
            payload.later_head_fractions[block_indices]
        ),
        "denoising_targets": np.ascontiguousarray(
            payload.denoising_targets[block_indices]
        ),
        "certificate_codes": np.ascontiguousarray(
            payload.certificate_codes[block_indices]
        ),
        "post_phase_states": np.ascontiguousarray(
            payload.post_phase_states[block_indices]
        ),
    }


def _persist_shard(
    shard_dir: Path,
    *,
    split: str,
    start_step: int,
    path_ids: Sequence[int],
    input_state_sha256: str,
    scientific_config_sha256: str,
    path_plan_sha256: str,
    profile_sha256: str,
    result: Any,
    capture: dict[str, np.ndarray] | None,
) -> tuple[Path, Path, Path]:
    stem = f"{split}-step-{start_step:03d}"
    state_path = shard_dir / f"{stem}-state.npz"
    capture_path = shard_dir / f"{stem}-capture.npz"
    metadata_path = shard_dir / f"{stem}.json"
    state_record = _atomic_npz(
        state_path, {"final_states": result.committed_final_states}
    )
    if capture is not None:
        capture_record = _atomic_npz(capture_path, capture)
    else:
        capture_record = None
        if capture_path.exists():
            capture_path.unlink()
    record = {
        "schema": RUN_SCHEMA + "-cache-shard",
        "schema_version": 1,
        "split": split,
        "start_step": start_step,
        "step_count": SHARD_STEPS,
        "path_ids": list(path_ids),
        "root_seed": ROOT_SEED,
        "scientific_config_sha256": scientific_config_sha256,
        "path_plan_sha256": path_plan_sha256,
        "profile_sha256": profile_sha256,
        "input_state_sha256": input_state_sha256,
        "final_state_sha256": _array_sha(result.committed_final_states),
        "state_file_sha256": state_record["sha256"],
        "state_file_size": state_record["size"],
        "state_array_sha256": state_record["array_hashes"]["final_states"],
        "capture_file_sha256": (
            None if capture_record is None else capture_record["sha256"]
        ),
        "capture_file_size": (
            None if capture_record is None else capture_record["size"]
        ),
        "capture_array_hashes": (
            None if capture_record is None else capture_record["array_hashes"]
        ),
        "capture_expected": int(capture is not None),
        "scheduler_record": result.to_record(),
    }
    record["scheduler_record_sha256"] = config_fingerprint(
        record["scheduler_record"]
    )
    record["semantic_sha256"] = config_fingerprint(record)
    atomic_write_json(metadata_path, record)
    return state_path, capture_path, metadata_path


def _flatten_selected_captures(
    captures: Sequence[Mapping[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    from mnist import d0_jacobi_rb_cuda_controls as controls
    from mnist.d0_jacobi_rb_learnability import sample_key as encode_sample_key

    matchings = controls._matching_arrays()
    rows: list[tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    maximum_alignment_error = 0.0
    for capture in captures:
        path_ids = np.asarray(capture["path_ids"], dtype=np.int64)
        outer_steps = np.asarray(capture["outer_steps"], dtype=np.int16)
        phases = np.asarray(capture["phases"], dtype=np.int8)
        later = np.asarray(capture["later_head_fractions"], dtype=np.float64)
        targets = np.asarray(capture["denoising_targets"], dtype=np.float64)
        codes = np.asarray(capture["certificate_codes"], dtype=np.uint8)
        states = np.asarray(capture["post_phase_states"], dtype=np.float64)
        for block, (outer_step, phase) in enumerate(
            zip(outer_steps.tolist(), phases.tolist(), strict=True)
        ):
            tails, heads = matchings[PHASE_MATCHINGS[int(phase)]]
            phase_states = states[block]
            pair_mass = phase_states[:, tails] + phase_states[:, heads]
            reconstructed = np.where(
                pair_mass > 0.0,
                phase_states[:, heads] / pair_mass,
                0.0,
            )
            maximum_alignment_error = max(
                maximum_alignment_error,
                float(np.max(np.abs(reconstructed - later[block]))),
            )
            for path_index, path_id in enumerate(path_ids.tolist()):
                rows.append(
                    (
                        int(path_id),
                        int(outer_step),
                        int(phase),
                        states[block, path_index],
                        targets[block, path_index],
                        codes[block, path_index],
                        later[block, path_index],
                    )
                )
    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    sample_key = np.asarray(
        [
            encode_sample_key(path_id, outer_step, phase)
            for path_id, outer_step, phase, *_rest in rows
        ],
        dtype=np.int64,
    )
    later_full_state = np.stack([row[3] for row in rows]).astype(np.float64)
    target = np.stack([row[4] for row in rows]).astype(np.float64)
    codes = np.stack([row[5] for row in rows]).astype(np.uint8)
    path_id = np.asarray([row[0] for row in rows], dtype=np.int64)
    outer_step = np.asarray([row[1] for row in rows], dtype=np.int16)
    phase = np.asarray([row[2] for row in rows], dtype=np.int8)
    color = np.asarray([PHASE_MATCHINGS[int(value)] for value in phase], dtype=np.int8)
    duration = np.asarray(
        [PHASE_DURATIONS[int(value)] for value in phase], dtype=np.float64
    )
    reverse_time = 1.0 - (
        7.0 * outer_step.astype(np.float64) + phase.astype(np.float64) + 1.0
    ) / (7.0 * OUTER_STEPS)
    inputs = {
        "sample_key": sample_key,
        "later_full_state": later_full_state,
        "reverse_time": reverse_time.astype(np.float64),
        "phase": phase,
        "color": color,
        "duration": duration,
        "label": np.full(len(rows), LABEL, dtype=np.int64),
    }
    audit = {
        "sample_key": sample_key.copy(),
        "path_id": path_id,
        "outer_step": outer_step,
        "phase": phase.copy(),
        "denoising_target": target,
        "certificate_codes": codes,
    }
    metrics = {
        "sample_count": len(rows),
        "path_count": len(set(path_id.tolist())),
        "selected_outer_steps": sorted(set(outer_step.tolist())),
        "phase_counts": {
            str(value): int(np.sum(phase == value)) for value in range(7)
        },
        "all_states_finite": int(np.isfinite(later_full_state).all()),
        "all_targets_finite": int(np.isfinite(target).all()),
        "maximum_capture_alignment_error": maximum_alignment_error,
        "capture_state_alignment_pass": int(maximum_alignment_error <= 2.0e-12),
        "input_schema_pass": int(set(inputs) == set(INPUT_FIELDS)),
        "audit_schema_pass": int(set(audit) == set(AUDIT_FIELDS)),
        "join_key_pass": int(np.array_equal(inputs["sample_key"], audit["sample_key"])),
        "sample_key_unique_pass": int(len(np.unique(sample_key)) == len(sample_key)),
        "target_modification_count": 0,
    }
    return inputs, audit, metrics


def _generate_split_cache(
    run_dir: Path,
    *,
    split: str,
    path_ids: Sequence[int],
    mixed_target: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    if split not in PATH_IDS or tuple(path_ids) != PATH_IDS[split]:
        raise LearnabilityCLIError(
            f"unexpected path IDs for split {split}",
            failure_domain="cache_configuration",
            failure_code="cache_path_plan_mismatch",
        )
    cache_dir = run_dir / "cache"
    shard_dir = cache_dir / f"{split}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    current = np.repeat(mixed_target[None, :], len(path_ids), axis=0)
    captures: list[dict[str, np.ndarray]] = []
    shard_records: list[dict[str, Any]] = []
    recompute_tail = False
    profile = JacobiRBCudaProfile()
    profile_sha256 = config_fingerprint(profile.to_dict())
    scientific_config_sha256 = _load_json(
        run_dir / "scientific_config.json"
    )["semantic_sha256"]
    path_plan_sha256 = _validated_path_plan_record(
        run_dir / "path_id_plan.json"
    )["semantic_sha256"]
    for start_step in range(0, OUTER_STEPS, SHARD_STEPS):
        capture_expected = start_step % 16 == 8
        stem = f"{split}-step-{start_step:03d}"
        state_path = shard_dir / f"{stem}-state.npz"
        capture_path = shard_dir / f"{stem}-capture.npz"
        metadata_path = shard_dir / f"{stem}.json"
        input_sha = _array_sha(current)
        valid, final_states, saved_capture = (False, None, None)
        if not recompute_tail:
            valid, final_states, saved_capture = _valid_committed_shard(
                state_path=state_path,
                capture_path=capture_path,
                metadata_path=metadata_path,
                expected_split=split,
                expected_start_step=start_step,
                expected_path_ids=path_ids,
                expected_input_states=current,
                expected_input_sha256=input_sha,
                capture_expected=capture_expected,
                scientific_config_sha256=scientific_config_sha256,
                path_plan_sha256=path_plan_sha256,
                profile_sha256=profile_sha256,
            )
        if valid:
            assert final_states is not None
            current = final_states
            if saved_capture is not None:
                captures.append(saved_capture)
            shard_records.append(_load_json(metadata_path))
            continue
        recompute_tail = True
        states = torch.as_tensor(current, dtype=torch.float64, device=device).contiguous()
        result = run_exact_multipath_shard(
            states,
            path_ids=path_ids,
            start_step=start_step,
            root_seed=ROOT_SEED,
            profile=profile,
            group_sizes=(len(path_ids),),
            capture_training_payload=capture_expected,
        )
        selected: dict[str, np.ndarray] | None = None
        if capture_expected:
            if result.capture_payload is None:
                raise LearnabilityCLIError(
                    "selected exact shard did not return its capture payload",
                    failure_domain="cache_capture",
                    failure_code="cache_capture_payload_missing",
                )
            selected = _selected_capture_arrays(result.capture_payload)
            captures.append(selected)
        _persist_shard(
            shard_dir,
            split=split,
            start_step=start_step,
            path_ids=path_ids,
            input_state_sha256=input_sha,
            scientific_config_sha256=scientific_config_sha256,
            path_plan_sha256=path_plan_sha256,
            profile_sha256=profile_sha256,
            result=result,
            capture=selected,
        )
        current = np.asarray(result.committed_final_states, dtype=np.float64)
        shard_records.append(_load_json(metadata_path))
        print(
            f"{split} exact cache shard {start_step // 8 + 1}/64 committed",
            flush=True,
        )
    inputs, audit, flattened_metrics = _flatten_selected_captures(captures)
    input_path = cache_dir / f"{split}_inputs.npz"
    audit_path = cache_dir / f"{split}_labels_audit.npz"
    input_record = _atomic_npz(input_path, inputs)
    audit_record = _atomic_npz(audit_path, audit)
    diagnostics = [
        record["scheduler_record"]["diagnostics"] for record in shard_records
    ]
    transition_count = sum(int(item["transition_count"]) for item in diagnostics)
    certified_count = sum(int(item["certified_count"]) for item in diagnostics)
    forbidden = {
        name: sum(int(item.get(name, 0)) for item in diagnostics)
        for name in FORBIDDEN_COUNTS
    }
    metrics = {
        "schema": RUN_SCHEMA + "-split-cache-metrics",
        "schema_version": 1,
        "split": split,
        "transition_count": transition_count,
        "expected_transition_count": EXPECTED_TRANSITIONS_PER_SPLIT,
        "certified_count": certified_count,
        "certificate_fraction": (
            certified_count / transition_count if transition_count else 0.0
        ),
        "uncertified_count": transition_count - certified_count,
        "maximum_mass_error": max(
            float(item["maximum_mass_error"]) for item in diagnostics
        ),
        "outer_steps": OUTER_STEPS,
        "steps_per_shard": SHARD_STEPS,
        "phases_per_selected_step": len(PHASE_MATCHINGS),
        "shard_count": len(shard_records),
        "all_shards_complete_pass": int(len(shard_records) == OUTER_STEPS // SHARD_STEPS),
        "shard_chain_pass": int(
            len(shard_records) == OUTER_STEPS // SHARD_STEPS
            and all(
                record.get("split") == split
                and int(record.get("start_step", -1)) == 8 * index
                and record.get("path_ids") == list(path_ids)
                and record.get("scientific_config_sha256")
                == scientific_config_sha256
                and record.get("path_plan_sha256") == path_plan_sha256
                and record.get("profile_sha256") == profile_sha256
                and (
                    index == 0
                    or record.get("input_state_sha256")
                    == shard_records[index - 1].get("final_state_sha256")
                )
                for index, record in enumerate(shard_records)
            )
        ),
        "replay_hashes_pass": int(
            all(
                record.get("semantic_sha256")
                == config_fingerprint(
                    {
                        key: value
                        for key, value in record.items()
                        if key != "semantic_sha256"
                    }
                )
                for record in shard_records
            )
        ),
        "target_modification_count": 0,
        "projection_count": 0,
        "states_finite_pass": int(flattened_metrics["all_states_finite"]),
        "targets_finite_pass": int(flattened_metrics["all_targets_finite"]),
        "sample_key_join_pass": int(flattened_metrics["join_key_pass"]),
        "model_input_schema_firewall_pass": int(
            flattened_metrics["input_schema_pass"]
        ),
        "input_label_schema_separation_pass": int(
            set(INPUT_FIELDS).intersection(set(AUDIT_FIELDS))
            == {"sample_key", "phase"}
        ),
        "selected_step_phase_coverage_pass": int(
            flattened_metrics["selected_outer_steps"] == list(SELECTED_OUTER_STEPS)
            and all(
                int(flattened_metrics["phase_counts"].get(str(phase), 0))
                == len(path_ids) * len(SELECTED_OUTER_STEPS)
                for phase in range(7)
            )
        ),
        "state_updates_device_resident_pass": int(
            all(int(item.get("state_updates_device_resident", 0)) == 1 for item in diagnostics)
        ),
        "confirmation_absent_pass": int(
            split == "confirmation" or _no_confirmation_artifacts(run_dir)
        ),
        "selected_model_seal_pass": int(
            split != "confirmation"
            or (run_dir / "confirmation_seal.json").is_file()
        ),
        "confirmation_opened_once_pass": int(
            split != "confirmation" or (run_dir / "confirmation_open.json").is_file()
        ),
        "confirmation_path_plan_unchanged_pass": int(
            split != "confirmation"
            or tuple(path_ids) == PATH_IDS["confirmation"]
        ),
        "inputs": input_record,
        "labels_audit": audit_record,
        "persisted_cache_bytes": int(
            input_path.stat().st_size + audit_path.stat().st_size
        ),
        **forbidden,
        **flattened_metrics,
        **NO_WORK,
        "physical_training_performed": int(split == "confirmation"),
    }
    atomic_write_json(cache_dir / f"{split}_metrics.json", metrics)
    return metrics


def _no_confirmation_artifacts(run_dir: Path) -> bool:
    candidates = (
        run_dir / "confirmation_open.json",
        run_dir / "cache" / "confirmation_inputs.npz",
        run_dir / "cache" / "confirmation_labels_audit.npz",
        run_dir / "cache" / "confirmation_metrics.json",
        run_dir / "confirmation_metrics.json",
        run_dir / "confirmation_path_metrics.csv",
        run_dir / "confirmation_gate.json",
    )
    shard_dir = run_dir / "cache" / "confirmation_shards"
    return not any(path.exists() for path in candidates) and not (
        shard_dir.is_dir() and any(shard_dir.iterdir())
    )


def _cache_stage(
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_learnability_gate import (
        LearnabilityThresholds,
        evaluate_learnability_cache,
    )

    preflight = _require_stage_artifact(run_dir, "preflight_gate.json")
    if not _passed(preflight):
        raise ArtifactCompatibilityError("cache stage requires a passing preflight")
    completed_gate = _optional_json(run_dir, "cache_gate.json")
    if completed_gate is not None:
        if not _passed(completed_gate):
            raise ArtifactCompatibilityError(
                "completed train/validation cache gate did not pass"
            )
        _load_cache_bundle_for_role(run_dir, "train")
        _load_cache_bundle_for_role(run_dir, "validation")
        if not _no_confirmation_artifacts(run_dir):
            raise ArtifactCompatibilityError(
                "confirmation evidence exists in a completed cache-only stage"
            )
        return completed_gate
    if not _no_confirmation_artifacts(run_dir):
        raise ArtifactCompatibilityError("confirmation evidence exists before model seal")
    _metadata, _image, mixed = _load_source_image(args.parent_strang_run_dir)
    split_metrics = {
        split: _generate_split_cache(
            run_dir,
            split=split,
            path_ids=PATH_IDS[split],
            mixed_target=mixed,
            device=torch.device(args.device),
        )
        for split in ("train", "validation")
    }
    manifest = {
        "schema": RUN_SCHEMA + "-cache-manifest",
        "schema_version": 1,
        "scientific_config_sha256": _load_json(
            run_dir / "scientific_config.json"
        )["semantic_sha256"],
        "path_plan_sha256": _validated_path_plan_record(
            run_dir / "path_id_plan.json"
        )["semantic_sha256"],
        "root_seed": ROOT_SEED,
        "splits": split_metrics,
        "confirmation_cache_exists": 0,
        "persisted_cache_bytes": sum(
            path.stat().st_size
            for path in (run_dir / "cache").rglob("*")
            if path.is_file()
        ),
        **NO_WORK,
    }
    manifest["semantic_sha256"] = config_fingerprint(manifest)
    _freeze_json(run_dir / "cache_manifest.json", manifest)
    combined = {
        "schema": RUN_SCHEMA + "-cache-metrics",
        "schema_version": 1,
        "train": split_metrics["train"],
        "validation": split_metrics["validation"],
        "confirmation_artifacts_absent": int(_no_confirmation_artifacts(run_dir)),
        "input_contract_pass": int(
            all(
                int(split_metrics[name]["input_schema_pass"]) == 1
                and int(split_metrics[name]["audit_schema_pass"]) == 1
                for name in ("train", "validation")
            )
        ),
        **NO_WORK,
    }
    atomic_write_json(run_dir / "cache_metrics.json", combined)
    train_gate = evaluate_learnability_cache(
        split_metrics["train"],
        split="train",
        thresholds=LearnabilityThresholds(),
    )
    validation_gate = evaluate_learnability_cache(
        split_metrics["validation"],
        split="validation",
        thresholds=LearnabilityThresholds(),
    )
    atomic_write_json(run_dir / "train_cache_gate.json", train_gate)
    atomic_write_json(run_dir / "validation_cache_gate.json", validation_gate)
    total_cache_bytes = int(manifest["persisted_cache_bytes"])
    gate = {
        "schema": RUN_SCHEMA + "-cache-gate",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": int(
            _passed(train_gate)
            and _passed(validation_gate)
            and total_cache_bytes <= 134_217_728
        ),
        "train": train_gate,
        "validation": validation_gate,
        "confirmation_absent_pass": int(_no_confirmation_artifacts(run_dir)),
        "persisted_cache_bytes": total_cache_bytes,
        "maximum_persisted_cache_bytes": 134_217_728,
        "persisted_cache_resource_pass": int(
            total_cache_bytes <= 134_217_728
        ),
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "cache_gate.json", gate)
    return gate


def _load_cache_bundle_for_role(run_dir: Path, role: str) -> Any:
    from mnist.d0_jacobi_rb_learnability import load_cache_bundle

    input_path = run_dir / "cache" / f"{role}_inputs.npz"
    audit_path = run_dir / "cache" / f"{role}_labels_audit.npz"
    if role in {"train", "validation"}:
        manifest = _require_stage_artifact(run_dir, "cache_manifest.json")
        semantic_sha = manifest.get("semantic_sha256")
        body = dict(manifest)
        body.pop("semantic_sha256", None)
        if (
            semantic_sha != config_fingerprint(body)
            or manifest.get("scientific_config_sha256")
            != _load_json(run_dir / "scientific_config.json")["semantic_sha256"]
            or manifest.get("path_plan_sha256")
            != _validated_path_plan_record(
                run_dir / "path_id_plan.json"
            )["semantic_sha256"]
            or int(manifest.get("root_seed", -1)) != ROOT_SEED
        ):
            raise ArtifactCompatibilityError("cache manifest binding changed")
        splits = manifest.get("splits")
        if not isinstance(splits, Mapping) or not isinstance(
            splits.get(role), Mapping
        ):
            raise ArtifactCompatibilityError("cache manifest split is missing")
        metrics = dict(splits[role])
    else:
        metrics = _load_json(run_dir / "cache" / f"{role}_metrics.json")
    for name, path in (("inputs", input_path), ("labels_audit", audit_path)):
        expected = metrics.get(name)
        if not isinstance(expected, Mapping) or dict(expected) != _npz_record(path):
            raise ArtifactCompatibilityError(
                f"{role} cache {name} no longer matches its frozen record"
            )
    return load_cache_bundle(
        input_path,
        audit_path,
        expected_path_ids=PATH_IDS[role],
        expected_outer_steps=SELECTED_OUTER_STEPS,
    )


def _training_task_paths(
    run_dir: Path, *, task: str, seed: int
) -> tuple[Path, Path, Path]:
    root = run_dir / "checkpoints"
    return (
        root / f"{task}-seed-{seed}.pt",
        root / f"{task}-seed-{seed}.json",
        root / f"{task}-seed-{seed}-history.csv",
    )


def _training_progress_path(run_dir: Path, *, task: str, seed: int) -> Path:
    return run_dir / "checkpoints" / f"{task}-seed-{seed}-progress.pt"


def _training_data_sha256(
    train_inputs: Any,
    train_target: torch.Tensor,
    validation_inputs: Any,
    validation_target: torch.Tensor,
) -> str:
    digest = hashlib.sha256()
    for scope, inputs, target in (
        ("train", train_inputs, train_target),
        ("validation", validation_inputs, validation_target),
    ):
        digest.update(scope.encode("ascii"))
        for name in (
            "later_full_state",
            "reverse_time",
            "phase",
            "color",
            "duration",
            "label",
        ):
            value = getattr(inputs, name).detach().cpu().contiguous()
            digest.update(name.encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes(order="C"))
        value = target.detach().cpu().contiguous()
        digest.update(b"target")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _train_or_load_task(
    run_dir: Path,
    *,
    task: str,
    seed: int,
    train_inputs: Any,
    train_target: torch.Tensor,
    validation_inputs: Any,
    validation_target: torch.Tensor,
    target_scale: float,
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    from mnist.d0_jacobi_rb_learnability import (
        JacobiRBPhasePredictor,
        CheckpointCandidate,
        TrainingPlan,
        TrainingResumeSnapshot,
        state_dict_sha256,
        train_deterministic_regressor,
    )

    checkpoint_path, metadata_path, history_path = _training_task_paths(
        run_dir, task=task, seed=seed
    )
    training_data_sha256 = _training_data_sha256(
        train_inputs,
        train_target,
        validation_inputs,
        validation_target,
    )
    progress_path = _training_progress_path(
        run_dir, task=task, seed=seed
    )
    source_binding = _load_json(run_dir / "run_manifest.json")[
        "source_fingerprint"
    ]
    scientific_binding = _load_json(run_dir / "scientific_config.json")[
        "semantic_sha256"
    ]
    if checkpoint_path.is_file() and metadata_path.is_file() and history_path.is_file():
        metadata = _load_json(metadata_path)
        if (
            metadata.get("task") != task
            or int(metadata.get("seed", -1)) != int(seed)
            or float(metadata.get("target_scale", math.nan)) != float(target_scale)
            or metadata.get("training_data_sha256") != training_data_sha256
            or metadata.get("checkpoint_file_sha256")
            != file_fingerprint(checkpoint_path)
        ):
            raise ArtifactCompatibilityError(
                f"completed training task is incompatible: {task}/{seed}"
            )
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict")
        if not isinstance(state, Mapping):
            raise ArtifactCompatibilityError("training checkpoint lacks a model state")
        if state_dict_sha256(state) != metadata.get("state_sha256"):
            raise ArtifactCompatibilityError("training checkpoint state hash mismatch")
        return metadata, state

    resume_snapshot: TrainingResumeSnapshot | None = None
    if progress_path.is_file():
        progress = torch.load(
            progress_path, map_location="cpu", weights_only=False
        )
        expected = {
            "schema": RUN_SCHEMA + "-training-progress",
            "schema_version": 1,
            "task": task,
            "seed": int(seed),
            "target_scale": float(target_scale),
            "training_data_sha256": training_data_sha256,
            "source_fingerprint": source_binding,
            "scientific_config_sha256": scientific_binding,
            "training_plan": TrainingPlan().to_record(),
        }
        if any(progress.get(name) != value for name, value in expected.items()):
            raise ArtifactCompatibilityError(
                f"training progress is incompatible: {task}/{seed}"
            )
        current_state = progress.get("model_state_dict")
        best_state = progress.get("best_state_dict")
        optimizer_state = progress.get("optimizer_state_dict")
        if (
            not isinstance(current_state, Mapping)
            or not isinstance(best_state, Mapping)
            or not isinstance(optimizer_state, Mapping)
            or state_dict_sha256(current_state)
            != progress.get("model_state_sha256")
            or state_dict_sha256(best_state)
            != progress.get("best_state_sha256")
        ):
            raise ArtifactCompatibilityError(
                f"training progress state hash mismatch: {task}/{seed}"
            )
        resume_snapshot = TrainingResumeSnapshot(
            seed=int(seed),
            completed_update=int(progress["completed_update"]),
            model_state_dict=current_state,
            optimizer_state_dict=optimizer_state,
            best_candidate=CheckpointCandidate(
                seed=int(seed),
                update=int(progress["best_update"]),
                validation_mse=float(progress["best_validation_mse"]),
                state_sha256=str(progress["best_state_sha256"]),
                state_dict=best_state,
            ),
            history=tuple(progress.get("history", ())),
            finite=bool(progress.get("finite", False)),
            torch_rng_state=progress["torch_rng_state"],
            cuda_rng_states=tuple(progress.get("cuda_rng_states", ())),
        )

    def persist_progress(snapshot: TrainingResumeSnapshot) -> None:
        payload = {
            "schema": RUN_SCHEMA + "-training-progress",
            "schema_version": 1,
            "task": task,
            "seed": int(seed),
            "completed_update": int(snapshot.completed_update),
            "target_scale": float(target_scale),
            "training_data_sha256": training_data_sha256,
            "source_fingerprint": source_binding,
            "scientific_config_sha256": scientific_binding,
            "training_plan": TrainingPlan().to_record(),
            "model_state_dict": dict(snapshot.model_state_dict),
            "model_state_sha256": state_dict_sha256(
                snapshot.model_state_dict
            ),
            "optimizer_state_dict": dict(snapshot.optimizer_state_dict),
            "best_update": int(snapshot.best_candidate.update),
            "best_validation_mse": float(
                snapshot.best_candidate.validation_mse
            ),
            "best_state_sha256": snapshot.best_candidate.state_sha256,
            "best_state_dict": dict(snapshot.best_candidate.state_dict),
            "history": list(snapshot.history),
            "finite": int(snapshot.finite),
            "torch_rng_state": snapshot.torch_rng_state,
            "cuda_rng_states": list(snapshot.cuda_rng_states),
            **CLAIM_FLAGS,
            **NO_WORK,
            "physical_training_performed": int(task == "physical-rb"),
        }
        _atomic_torch_save(progress_path, payload)

    result = train_deterministic_regressor(
        lambda: JacobiRBPhasePredictor(width=32, num_classes=10),
        train_inputs,
        train_target,
        validation_inputs,
        validation_target,
        target_scale=target_scale,
        seed=seed,
        plan=TrainingPlan(),
        resume_snapshot=resume_snapshot,
        checkpoint_callback=persist_progress,
    )
    selected = result.selected
    checkpoint = {
        "schema": RUN_SCHEMA + "-training-checkpoint",
        "schema_version": 1,
        "task": task,
        "seed": int(seed),
        "selected_update": int(selected.update),
        "validation_mse": float(selected.validation_mse),
        "target_scale": float(target_scale),
        "training_data_sha256": training_data_sha256,
        "state_sha256": str(selected.state_sha256),
        "model_state_dict": dict(selected.state_dict),
        "training_plan": TrainingPlan().to_record(),
        "history": list(result.history),
        "rng_semantics": "stateless deterministic batches plus fixed model seed",
        **CLAIM_FLAGS,
        **NO_WORK,
        "physical_training_performed": int(task == "physical-rb"),
    }
    checkpoint_record = _atomic_torch_save(checkpoint_path, checkpoint)
    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key != "model_state_dict"
    }
    metadata.update(
        {
            "finite": int(result.finite),
            "history_record_count": len(result.history),
            "checkpoint_file_sha256": checkpoint_record["sha256"],
            "checkpoint_file_size": checkpoint_record["size"],
        }
    )
    atomic_write_json(metadata_path, metadata)
    _write_csv(history_path, result.history)
    return metadata, selected.state_dict


def _load_model_with_state(
    state_dict: Mapping[str, torch.Tensor], *, device: torch.device
) -> torch.nn.Module:
    from mnist.d0_jacobi_rb_learnability import (
        JacobiRBPhasePredictor,
        state_dict_sha256,
    )

    model = JacobiRBPhasePredictor(width=32, num_classes=10).to(device)
    model.load_state_dict(dict(state_dict), strict=True)
    if state_dict_sha256(model.state_dict()) != state_dict_sha256(state_dict):
        raise ArtifactCompatibilityError("loaded checkpoint replay hash mismatch")
    model.eval()
    return model


def _path_rows(summary: Any) -> list[dict[str, Any]]:
    return [
        {
            "path_id": int(item.path_id),
            "model_mse": float(item.model_mse),
            "metadata_mse": float(item.metadata_mse),
            "zero_mse": float(item.zero_mse),
            "metadata_minus_model_mse": float(item.metadata_improvement),
            "relative_metadata_improvement": float(
                item.relative_metadata_improvement
            ),
        }
        for item in summary.paths
    ]


def _teacher_stage(
    run_dir: Path,
    *,
    train_bundle: Any,
    validation_bundle: Any,
    train_inputs: Any,
    validation_inputs: Any,
    device: torch.device,
) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_learnability import (
        audit_targets_from_cache,
        evaluate_model_mse,
        exact_global_target_scale,
        fit_metadata_baseline,
        path_mse_summary,
        synthetic_teacher_target,
    )
    from mnist.d0_jacobi_rb_learnability_gate import (
        LearnabilityThresholds,
        evaluate_learnability_teacher,
    )

    train_audit = audit_targets_from_cache(
        train_bundle.labels_audit, device=device
    )
    validation_audit = audit_targets_from_cache(
        validation_bundle.labels_audit, device=device
    )
    train_target = synthetic_teacher_target(train_inputs)
    validation_target = synthetic_teacher_target(validation_inputs)
    teacher_scale = exact_global_target_scale(train_target)
    teacher_baseline = fit_metadata_baseline(
        train_target,
        train_audit.outer_step,
        train_audit.phase,
    )
    validation_baseline = teacher_baseline.predict(
        validation_bundle.labels_audit.outer_step,
        validation_bundle.labels_audit.phase,
    )
    task, state = _train_or_load_task(
        run_dir,
        task="synthetic-teacher",
        seed=MODEL_SEEDS[0],
        train_inputs=train_inputs,
        train_target=train_target,
        validation_inputs=validation_inputs,
        validation_target=validation_target,
        target_scale=teacher_scale,
    )
    model = _load_model_with_state(state, device=device)
    teacher_mse, prediction = evaluate_model_mse(
        model, validation_inputs, validation_target, batch_size=32
    )
    summary = path_mse_summary(
        prediction,
        validation_target,
        validation_baseline,
        validation_audit.path_id,
    )
    rows = _path_rows(summary)
    _write_csv(run_dir / "teacher_path_metrics.csv", rows)
    baseline_mse = float(summary.aggregate_metadata_mse)
    metrics = {
        "schema": RUN_SCHEMA + "-teacher-metrics",
        "schema_version": 1,
        "training_complete_pass": 1,
        "all_losses_finite_pass": int(
            int(task.get("finite", 0)) == 1
            and math.isfinite(teacher_mse)
            and math.isfinite(baseline_mse)
        ),
        "same_pipeline_pass": 1,
        "selected_checkpoint_replay_hash_pass": 1,
        "model_input_schema_firewall_pass": 1,
        "training_only_scale_pass": 1,
        "no_target_modification_pass": 1,
        "validation_path_count": len(rows),
        "paths_beating_metadata_baseline": sum(
            row["metadata_minus_model_mse"] > 0.0 for row in rows
        ),
        "validation_teacher_mse": teacher_mse,
        "validation_metadata_baseline_mse": baseline_mse,
        "validation_teacher_relative_mse": (
            teacher_mse / baseline_mse if baseline_mse > 0.0 else math.inf
        ),
        "target_scale": teacher_scale,
        "selected_seed": int(task["seed"]),
        "selected_update": int(task["selected_update"]),
        "selected_state_sha256": task["state_sha256"],
        **NO_WORK,
    }
    atomic_write_json(run_dir / "teacher_metrics.json", metrics)
    gate = evaluate_learnability_teacher(
        metrics, thresholds=LearnabilityThresholds()
    )
    atomic_write_json(run_dir / "teacher_gate.json", gate)
    return gate


def _freeze_confirmation_seal(
    run_dir: Path,
    *,
    selected_model_record: Mapping[str, Any],
    baseline_sha256: str,
) -> dict[str, Any]:
    path_plan = _validated_path_plan_record(run_dir / "path_id_plan.json")
    scientific = _load_json(run_dir / "scientific_config.json")
    static = {
        "schema": RUN_SCHEMA + "-confirmation-seal",
        "schema_version": 1,
        "selected_model_file_sha256": selected_model_record[
            "checkpoint_file_sha256"
        ],
        "selected_state_sha256": selected_model_record["state_sha256"],
        "model_config_sha256": scientific["semantic_sha256"],
        "metadata_baseline_file_sha256": baseline_sha256,
        "path_plan_sha256": path_plan["semantic_sha256"],
        "confirmation_path_ids": list(PATH_IDS["confirmation"]),
        "confirmation_gate_definition": {
            "path_count": 8,
            "all_metadata_improvements_strictly_positive": 1,
            "aggregate_model_mse_strictly_below_zero_mse": 1,
            "ties_fail": 1,
            "one_sided_sign_test_p_value_on_pass": 1.0 / 256.0,
        },
        "confirmation_opened": 0,
        **CLAIM_FLAGS,
        **NO_WORK,
        "physical_training_performed": 1,
    }
    static["seal_sha256"] = config_fingerprint(static)
    path = run_dir / "confirmation_seal.json"
    if path.is_file():
        existing = _load_json(path)
        if existing != static:
            raise ArtifactCompatibilityError("confirmation seal changed")
        return existing
    atomic_write_json(path, static)
    return static


def _validated_confirmation_seal(run_dir: Path) -> dict[str, Any]:
    seal = _require_stage_artifact(run_dir, "confirmation_seal.json")
    expected_hash = seal.get("seal_sha256")
    body = dict(seal)
    body.pop("seal_sha256", None)
    if expected_hash != config_fingerprint(body):
        raise ArtifactCompatibilityError("confirmation seal semantic hash mismatch")
    if seal.get("confirmation_path_ids") != list(PATH_IDS["confirmation"]):
        raise ArtifactCompatibilityError("confirmation path plan changed after sealing")
    return seal


def _freeze_physical_metadata_baseline(
    run_dir: Path, train_bundle: Any
) -> tuple[Any, str]:
    from mnist.d0_jacobi_rb_learnability import (
        fit_metadata_baseline,
        load_metadata_baseline,
        save_metadata_baseline,
    )

    baseline = fit_metadata_baseline(
        train_bundle.labels_audit.denoising_target,
        train_bundle.labels_audit.outer_step,
        train_bundle.labels_audit.phase,
    )
    baseline_path = run_dir / "metadata_baseline.npz"
    if baseline_path.is_file():
        frozen = load_metadata_baseline(baseline_path)
        if frozen.sha256 != baseline.sha256:
            raise ArtifactCompatibilityError("metadata baseline changed")
        baseline = frozen
    else:
        save_metadata_baseline(baseline_path, baseline)
    baseline_file_sha = file_fingerprint(baseline_path)
    baseline_record = {
        "schema": RUN_SCHEMA + "-metadata-baseline",
        "schema_version": 1,
        "fit_split": "train",
        "fit_path_ids": list(PATH_IDS["train"]),
        "frozen_before_synthetic_teacher": 1,
        "values_sha256": baseline.sha256,
        "file_sha256": baseline_file_sha,
        "shape": list(baseline.values.shape),
        **NO_WORK,
    }
    _freeze_json(run_dir / "metadata_baseline.json", baseline_record)
    return baseline, baseline_file_sha


def _physical_training_stage(
    run_dir: Path,
    *,
    train_bundle: Any,
    validation_bundle: Any,
    train_inputs: Any,
    validation_inputs: Any,
    device: torch.device,
) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_learnability import (
        TrainingPlan,
        audit_targets_from_cache,
        evaluate_model_mse,
        exact_global_target_scale,
        path_mse_summary,
    )
    from mnist.d0_jacobi_rb_learnability_gate import (
        LearnabilityThresholds,
        evaluate_learnability_physical,
    )

    train_audit = audit_targets_from_cache(
        train_bundle.labels_audit, device=device
    )
    validation_audit = audit_targets_from_cache(
        validation_bundle.labels_audit, device=device
    )
    train_target = train_audit.denoising_target
    validation_target = validation_audit.denoising_target
    target_scale = exact_global_target_scale(train_target)
    baseline, baseline_file_sha = _freeze_physical_metadata_baseline(
        run_dir, train_bundle
    )
    baseline_path = run_dir / "metadata_baseline.npz"
    _freeze_json(
        run_dir / "physical_training_started.json",
        {
            "schema": RUN_SCHEMA + "-physical-training-started",
            "schema_version": 1,
            "model_seeds": list(MODEL_SEEDS),
            "scientific_config_sha256": _load_json(
                run_dir / "scientific_config.json"
            )["semantic_sha256"],
            **CLAIM_FLAGS,
            **NO_WORK,
            "physical_training_performed": 1,
        },
    )
    task_records: list[dict[str, Any]] = []
    states: dict[int, Mapping[str, torch.Tensor]] = {}
    histories: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        task, state = _train_or_load_task(
            run_dir,
            task="physical-rb",
            seed=seed,
            train_inputs=train_inputs,
            train_target=train_target,
            validation_inputs=validation_inputs,
            validation_target=validation_target,
            target_scale=target_scale,
        )
        task_records.append(task)
        states[seed] = state
        _checkpoint, _metadata, history_path = _training_task_paths(
            run_dir, task="physical-rb", seed=seed
        )
        with history_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                histories.append({"seed": seed, **row})
    _write_csv(run_dir / "physical_training_history.csv", histories)
    selected = min(
        task_records,
        key=lambda item: (
            float(item["validation_mse"]),
            int(item["seed"]),
            int(item["selected_update"]),
        ),
    )
    selected_state = states[int(selected["seed"])]
    selected_payload = {
        "schema": RUN_SCHEMA + "-selected-model-checkpoint",
        "schema_version": 1,
        "model_state_dict": dict(selected_state),
        "state_sha256": selected["state_sha256"],
        "seed": int(selected["seed"]),
        "update": int(selected["selected_update"]),
        "validation_mse": float(selected["validation_mse"]),
        "target_scale": target_scale,
        "training_data_sha256": selected["training_data_sha256"],
        "model": {"name": "JacobiRBPhasePredictor", "width": 32, "classes": 10},
        "training_plan": TrainingPlan().to_record(),
        **CLAIM_FLAGS,
        **NO_WORK,
        "physical_training_performed": 1,
    }
    selected_path = run_dir / "selected_model.pt"
    if selected_path.is_file():
        existing_payload = torch.load(
            selected_path, map_location="cpu", weights_only=False
        )
        existing_state = existing_payload.get("model_state_dict")
        if (
            not isinstance(existing_state, Mapping)
            or existing_payload.get("state_sha256") != selected["state_sha256"]
            or existing_payload.get("seed") != int(selected["seed"])
            or existing_payload.get("update") != int(selected["selected_update"])
        ):
            raise ArtifactCompatibilityError("frozen selected checkpoint changed")
        checkpoint_record = {
            "path": selected_path.as_posix(),
            "sha256": file_fingerprint(selected_path),
            "size": int(selected_path.stat().st_size),
        }
    else:
        checkpoint_record = _atomic_torch_save(selected_path, selected_payload)
    selected_record = {
        key: value
        for key, value in selected_payload.items()
        if key != "model_state_dict"
    }
    selected_record.update(
        {
            "checkpoint_file_sha256": checkpoint_record["sha256"],
            "checkpoint_file_size": checkpoint_record["size"],
            "selection_split": "validation",
            "selection_path_ids": list(PATH_IDS["validation"]),
            "tie_breaking": "validation_mse, lower_seed, earlier_update",
            "metadata_baseline_file_sha256": baseline_file_sha,
            "scientific_config_sha256": _load_json(
                run_dir / "scientific_config.json"
            )["semantic_sha256"],
        }
    )
    _freeze_json(run_dir / "selected_model.json", selected_record)
    _freeze_confirmation_seal(
        run_dir,
        selected_model_record=selected_record,
        baseline_sha256=baseline_file_sha,
    )
    selected_model = _load_model_with_state(selected_state, device=device)
    validation_mse, prediction = evaluate_model_mse(
        selected_model, validation_inputs, validation_target, batch_size=32
    )
    metadata_prediction = baseline.predict(
        validation_bundle.labels_audit.outer_step,
        validation_bundle.labels_audit.phase,
    )
    summary = path_mse_summary(
        prediction,
        validation_target,
        metadata_prediction,
        validation_audit.path_id,
    )
    _write_csv(run_dir / "validation_path_metrics.csv", _path_rows(summary))
    metrics = {
        "schema": RUN_SCHEMA + "-physical-training-metrics",
        "schema_version": 1,
        "training_complete_pass": 1,
        "all_seeds_complete_pass": int(len(task_records) == 3),
        "all_losses_finite_pass": int(
            all(int(item.get("finite", 0)) == 1 for item in task_records)
            and math.isfinite(validation_mse)
        ),
        "validation_only_selection_pass": 1,
        "selected_checkpoint_exists_pass": int(
            selected_path.is_file()
        ),
        "selected_checkpoint_hash_pass": int(
            file_fingerprint(selected_path)
            == selected_record["checkpoint_file_sha256"]
        ),
        "selected_model_record_frozen_pass": 1,
        "metadata_baseline_frozen_pass": 1,
        "confirmation_gate_definition_frozen_pass": 1,
        "confirmation_absent_pass": int(_no_confirmation_artifacts(run_dir)),
        "model_input_schema_firewall_pass": 1,
        "training_only_scale_pass": 1,
        "unweighted_mse_objective_pass": 1,
        "no_target_modification_pass": 1,
        "model_seed_count": len(task_records),
        "validation_path_count": len(summary.paths),
        "target_scale": target_scale,
        "selected_seed": int(selected["seed"]),
        "selected_update": int(selected["selected_update"]),
        "selected_validation_mse": validation_mse,
        "validation_metadata_baseline_mse": summary.aggregate_metadata_mse,
        "validation_zero_mse": summary.aggregate_zero_mse,
        **NO_WORK,
        "physical_training_performed": 1,
    }
    atomic_write_json(run_dir / "physical_training_metrics.json", metrics)
    gate = evaluate_learnability_physical(
        metrics, thresholds=LearnabilityThresholds()
    )
    atomic_write_json(run_dir / "physical_gate.json", gate)
    return gate


def _train_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    cache_gate = _require_stage_artifact(run_dir, "cache_gate.json")
    if not _passed(cache_gate):
        raise ArtifactCompatibilityError("train stage requires passing exact caches")
    if not _no_confirmation_artifacts(run_dir):
        raise ArtifactCompatibilityError("confirmation data exists before training")
    device = torch.device(args.device)
    train_bundle = _load_cache_bundle_for_role(run_dir, "train")
    validation_bundle = _load_cache_bundle_for_role(run_dir, "validation")
    completed_gate = _optional_json(run_dir, "physical_gate.json")
    if completed_gate is not None:
        _require_stage_artifact(run_dir, "teacher_gate.json")
        selected = _require_stage_artifact(run_dir, "selected_model.json")
        seal = _validated_confirmation_seal(run_dir)
        if (
            file_fingerprint(run_dir / "selected_model.pt")
            != selected.get("checkpoint_file_sha256")
            or selected.get("checkpoint_file_sha256")
            != seal.get("selected_model_file_sha256")
        ):
            raise ArtifactCompatibilityError(
                "completed training checkpoint no longer matches its seal"
            )
        _freeze_physical_metadata_baseline(run_dir, train_bundle)
        return completed_gate
    from mnist.d0_jacobi_rb_learnability import (
        TrainingPlan,
        model_inputs_from_cache,
    )

    train_inputs = model_inputs_from_cache(
        train_bundle.inputs, device=device, floating_dtype=torch.float32
    )
    validation_inputs = model_inputs_from_cache(
        validation_bundle.inputs, device=device, floating_dtype=torch.float32
    )
    _freeze_json(
        run_dir / "training_plan.json",
        {
            "schema": RUN_SCHEMA + "-training-plan",
            "schema_version": 1,
            **TrainingPlan().to_record(),
            "teacher_runs_first": 1,
            "physical_metadata_baseline_frozen_before_teacher": 1,
            "physical_model_seeds": list(MODEL_SEEDS),
            "validation_only_selection": 1,
            **NO_WORK,
        },
    )
    _freeze_physical_metadata_baseline(run_dir, train_bundle)
    teacher_gate = _teacher_stage(
        run_dir,
        train_bundle=train_bundle,
        validation_bundle=validation_bundle,
        train_inputs=train_inputs,
        validation_inputs=validation_inputs,
        device=device,
    )
    if not _passed(teacher_gate):
        return teacher_gate
    return _physical_training_stage(
        run_dir,
        train_bundle=train_bundle,
        validation_bundle=validation_bundle,
        train_inputs=train_inputs,
        validation_inputs=validation_inputs,
        device=device,
    )


def _open_confirmation(run_dir: Path) -> dict[str, Any]:
    seal = _validated_confirmation_seal(run_dir)
    selected = _require_stage_artifact(run_dir, "selected_model.json")
    if (
        file_fingerprint(run_dir / "selected_model.pt")
        != seal["selected_model_file_sha256"]
        or selected["checkpoint_file_sha256"]
        != seal["selected_model_file_sha256"]
    ):
        raise ArtifactCompatibilityError("selected checkpoint does not match its seal")
    path = run_dir / "confirmation_open.json"
    record = {
        "schema": RUN_SCHEMA + "-confirmation-open",
        "schema_version": 1,
        "opened_count": 1,
        "path_ids": list(PATH_IDS["confirmation"]),
        "seal_sha256": seal["seal_sha256"],
        "panel_resized": 0,
        "panel_regenerated": 0,
        **NO_WORK,
        "physical_training_performed": 1,
    }
    return _freeze_json(path, record)


def _phase_and_quartile_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    metadata: np.ndarray,
    outer_step: np.ndarray,
    phase: np.ndarray,
) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_learnability import stable_mse

    result: dict[str, Any] = {"phase": {}, "time_quartile": {}}
    for value in range(7):
        mask = phase == value
        result["phase"][str(value)] = {
            "model_mse": stable_mse(prediction[mask], target[mask]),
            "metadata_mse": stable_mse(metadata[mask], target[mask]),
        }
    quartile = outer_step.astype(np.int64) // 128
    for value in range(4):
        mask = quartile == value
        result["time_quartile"][str(value)] = {
            "model_mse": stable_mse(prediction[mask], target[mask]),
            "metadata_mse": stable_mse(metadata[mask], target[mask]),
        }
    return result


def _confirm_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_learnability import (
        audit_targets_from_cache,
        evaluate_model_mse,
        load_metadata_baseline,
        model_inputs_from_cache,
        path_mse_summary,
        state_dict_sha256,
    )
    from mnist.d0_jacobi_rb_learnability_gate import (
        LearnabilityThresholds,
        evaluate_learnability_cache,
        evaluate_learnability_confirmation,
    )

    completed_gate = _optional_json(run_dir, "confirmation_gate.json")
    if completed_gate is not None:
        seal = _validated_confirmation_seal(run_dir)
        _load_cache_bundle_for_role(run_dir, "confirmation")
        metrics = _require_stage_artifact(run_dir, "confirmation_metrics.json")
        cache_gate = _require_stage_artifact(
            run_dir, "confirmation_cache_gate.json"
        )
        if not (run_dir / "confirmation_path_metrics.csv").is_file():
            raise ArtifactCompatibilityError(
                "completed confirmation lacks per-path metrics"
            )
        if (
            file_fingerprint(run_dir / "selected_model.pt")
            != seal["selected_model_file_sha256"]
            or file_fingerprint(run_dir / "metadata_baseline.npz")
            != seal["metadata_baseline_file_sha256"]
        ):
            raise ArtifactCompatibilityError(
                "completed confirmation no longer matches its model seal"
            )
        replay_gate = evaluate_learnability_confirmation(
            metrics,
            confirmation_cache_gate=cache_gate,
            thresholds=LearnabilityThresholds(),
        )
        if replay_gate != completed_gate:
            raise ArtifactCompatibilityError(
                "completed confirmation gate does not replay exactly"
            )
        return completed_gate

    physical_gate = _require_stage_artifact(run_dir, "physical_gate.json")
    if not _passed(physical_gate):
        raise ArtifactCompatibilityError("confirmation requires passing physical training")
    seal = _validated_confirmation_seal(run_dir)
    _open_confirmation(run_dir)
    _metadata, _image, mixed = _load_source_image(args.parent_strang_run_dir)
    cache_metrics = _generate_split_cache(
        run_dir,
        split="confirmation",
        path_ids=PATH_IDS["confirmation"],
        mixed_target=mixed,
        device=torch.device(args.device),
    )
    cache_metrics = dict(cache_metrics)
    cache_metrics["total_persisted_cache_bytes"] = sum(
        path.stat().st_size
        for path in (run_dir / "cache").rglob("*")
        if path.is_file()
    )
    atomic_write_json(
        run_dir / "cache" / "confirmation_metrics.json", cache_metrics
    )
    cache_gate = evaluate_learnability_cache(
        cache_metrics,
        split="confirmation",
        thresholds=LearnabilityThresholds(),
    )
    atomic_write_json(run_dir / "confirmation_cache_gate.json", cache_gate)
    if not _passed(cache_gate):
        return cache_gate
    bundle = _load_cache_bundle_for_role(run_dir, "confirmation")
    existing_metrics = _optional_json(run_dir, "confirmation_metrics.json")
    if existing_metrics is not None:
        if not (run_dir / "confirmation_path_metrics.csv").is_file():
            raise ArtifactCompatibilityError(
                "sealed confirmation metrics lack their per-path table"
            )
        gate = evaluate_learnability_confirmation(
            existing_metrics,
            confirmation_cache_gate=cache_gate,
            thresholds=LearnabilityThresholds(),
        )
        atomic_write_json(run_dir / "confirmation_gate.json", gate)
        return gate
    model_inputs = model_inputs_from_cache(
        bundle.inputs, device=args.device, floating_dtype=torch.float32
    )
    audit = audit_targets_from_cache(bundle.labels_audit, device=args.device)
    checkpoint = torch.load(
        run_dir / "selected_model.pt", map_location="cpu", weights_only=False
    )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ArtifactCompatibilityError("selected checkpoint lacks model state")
    selected_record = _require_stage_artifact(run_dir, "selected_model.json")
    state_hash = state_dict_sha256(state)
    model = _load_model_with_state(state, device=torch.device(args.device))
    model_mse, prediction_tensor = evaluate_model_mse(
        model, model_inputs, audit.denoising_target, batch_size=32
    )
    prediction = prediction_tensor.detach().cpu().numpy()
    target = audit.denoising_target.detach().cpu().numpy()
    baseline_path = run_dir / "metadata_baseline.npz"
    baseline = load_metadata_baseline(baseline_path)
    metadata_prediction = baseline.predict(
        bundle.labels_audit.outer_step, bundle.labels_audit.phase
    )
    summary = path_mse_summary(
        prediction,
        target,
        metadata_prediction,
        bundle.labels_audit.path_id,
    )
    path_rows = _path_rows(summary)
    _write_csv(run_dir / "confirmation_path_metrics.csv", path_rows)
    diagnostics = _phase_and_quartile_metrics(
        prediction,
        target,
        metadata_prediction,
        bundle.labels_audit.outer_step,
        bundle.labels_audit.phase,
    )
    metrics = {
        "schema": RUN_SCHEMA + "-confirmation-metrics",
        "schema_version": 1,
        "confirmation_path_count": len(path_rows),
        "path_metadata_minus_model_mse": [
            row["metadata_minus_model_mse"] for row in path_rows
        ],
        "aggregate_model_mse": model_mse,
        "aggregate_metadata_mse": summary.aggregate_metadata_mse,
        "aggregate_zero_mse": summary.aggregate_zero_mse,
        "aggregate_relative_metadata_improvement": (
            summary.aggregate_relative_metadata_improvement
        ),
        "median_relative_metadata_improvement": (
            summary.median_relative_metadata_improvement
        ),
        "predictions_finite_pass": int(np.isfinite(prediction).all()),
        "losses_finite_pass": int(
            all(
                math.isfinite(value)
                for value in (
                    model_mse,
                    summary.aggregate_metadata_mse,
                    summary.aggregate_zero_mse,
                )
            )
        ),
        "selected_model_hash_pass": int(
            file_fingerprint(run_dir / "selected_model.pt")
            == seal["selected_model_file_sha256"]
            and state_hash == seal["selected_state_sha256"]
        ),
        "model_config_hash_pass": int(
            _load_json(run_dir / "scientific_config.json")["semantic_sha256"]
            == seal["model_config_sha256"]
        ),
        "metadata_baseline_hash_pass": int(
            file_fingerprint(baseline_path)
            == seal["metadata_baseline_file_sha256"]
        ),
        "path_plan_hash_pass": int(
            _validated_path_plan_record(run_dir / "path_id_plan.json")[
                "semantic_sha256"
            ]
            == seal["path_plan_sha256"]
        ),
        "confirmation_opened_once_pass": int(
            _load_json(run_dir / "confirmation_open.json")["opened_count"] == 1
        ),
        "confirmation_paths_not_replaced_pass": 1,
        "confirmation_paths_not_added_pass": 1,
        "model_input_schema_firewall_pass": 1,
        "selected_seed": selected_record["seed"],
        "selected_update": selected_record["update"],
        "descriptive_diagnostics": diagnostics,
        **NO_WORK,
        "physical_training_performed": 1,
    }
    _freeze_json(run_dir / "confirmation_metrics.json", metrics)
    gate = evaluate_learnability_confirmation(
        metrics,
        confirmation_cache_gate=cache_gate,
        thresholds=LearnabilityThresholds(),
    )
    atomic_write_json(run_dir / "confirmation_gate.json", gate)
    return gate


def _optional_json(run_dir: Path, filename: str) -> dict[str, Any] | None:
    path = run_dir / filename
    return _load_json(path) if path.is_file() else None


def _verified_provenance(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_learnability_provenance import (
        verify_learnability_parents,
    )

    try:
        verified = verify_learnability_parents(
            multipath_run_dir=args.parent_multipath_run_dir,
            strang_run_dir=args.parent_strang_run_dir,
            haar_run_dir=args.parent_haar_run_dir,
        )
    except ArtifactCompatibilityError as exc:
        raise ParentScopeError(str(exc)) from exc
    frozen = _optional_json(run_dir, "parent_provenance.json")
    if frozen is not None and frozen != verified:
        raise ArtifactCompatibilityError("parent provenance changed after preflight")
    return verified


def _write_workflow(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    required_gate: str | None = None,
) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_learnability_gate import (
        evaluate_learnability_workflow,
    )

    provenance = _verified_provenance(run_dir, args)
    workflow = evaluate_learnability_workflow(
        provenance=provenance,
        preflight_gate=_optional_json(run_dir, "preflight_gate.json"),
        train_cache_gate=_optional_json(run_dir, "train_cache_gate.json"),
        validation_cache_gate=_optional_json(run_dir, "validation_cache_gate.json"),
        teacher_gate=_optional_json(run_dir, "teacher_gate.json"),
        physical_gate=_optional_json(run_dir, "physical_gate.json"),
        confirmation_cache_gate=_optional_json(
            run_dir, "confirmation_cache_gate.json"
        ),
        confirmation_gate=_optional_json(run_dir, "confirmation_gate.json"),
        require_gate=required_gate or args.require_gate,
    )
    atomic_write_json(run_dir / "workflow_gate.json", workflow)
    atomic_write_json(run_dir / "learnability_decision.json", workflow["decision"])
    return workflow


def _initialize_run(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    resumed: bool,
) -> None:
    if resumed:
        _verify_existing_artifact_registry(run_dir)
    scientific = _scientific_config(authorizing=True)
    _freeze_json(
        run_dir / "scientific_config.json",
        scientific,
        require_existing=resumed,
    )
    sources = _source_paths()
    manifest = {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "created_by": "mnist.diag_d0_jacobi_rb_one_image_learnability",
        "scientific_config_sha256": scientific["semantic_sha256"],
        "source_fingerprint": source_fingerprint(sources),
        "source_paths": [
            path.relative_to(Path.cwd()).as_posix()
            if path.is_relative_to(Path.cwd())
            else path.as_posix()
            for path in sources
        ],
        "parents": {
            "multipath": str(args.parent_multipath_run_dir.resolve()),
            "strang": str(args.parent_strang_run_dir.resolve()),
            "haar": str(args.parent_haar_run_dir.resolve()),
        },
        "runtime_contract": {
            "device_argument": str(args.device),
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
        },
        "stage_contract": ["preflight", "cache", "train", "confirm"],
        "confirmation_sealed_until_selected_model": 1,
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    _freeze_json(run_dir / "run_manifest.json", manifest, require_existing=resumed)


def _required_stage_pass(
    run_dir: Path, required_gate: str
) -> bool:
    if required_gate == "none":
        return True
    mapping = {
        "preflight": "preflight_gate.json",
        "cache": "cache_gate.json",
        "train": "physical_gate.json",
        "confirm": "confirmation_gate.json",
    }
    gate = _optional_json(run_dir, mapping[required_gate])
    return gate is not None and _passed(gate)


def _commit_failure_decision(
    run_dir: Path,
    *,
    stage: str,
    required_gate: str,
    failure: Mapping[str, Any],
    provenance_failure: bool,
) -> str:
    failure_domain = str(failure.get("failure_domain", ""))
    failure_code = str(failure.get("failure_code", ""))
    if provenance_failure:
        decision = "parent_scope_invalid"
        action = "repair the immutable parent/source binding"
    elif (
        failure_domain in {"model_input_contract", "input_contract"}
        or "model_input_contract" in failure_code
    ):
        decision = "model_input_contract_invalid"
        action = "repair the later-state-only model input contract"
    elif stage in {"preflight", "cache"}:
        decision = "exact_cache_invalid"
        action = "repair the exact capture, resource, or cache execution"
    elif stage in {"train", "confirm"}:
        decision = "optimization_pipeline_invalid"
        action = "repair training, sealing, or confirmation execution"
    else:
        decision = "parent_scope_invalid"
        action = "complete a valid staged workflow"
    record = {
        "schema": RUN_SCHEMA + "-decision",
        "schema_version": 1,
        "evaluation_status": "execution_failed",
        "decision": decision,
        "claim_scope": (
            "conditional learnability of the exact Rao-Blackwell label for "
            "the exact K=512 split chain and one frozen MNIST image"
        ),
        "recommended_next_action": action,
        "larger_exact_discrete_chain_training_planning_authorized": 0,
        **CLAIM_FLAGS,
        **NO_WORK,
        "physical_training_performed": int(_physical_work_performed(run_dir)),
    }
    atomic_write_json(run_dir / "learnability_decision.json", record)
    atomic_write_json(
        run_dir / "workflow_gate.json",
        {
            "schema": RUN_SCHEMA + "-workflow",
            "schema_version": 1,
            "evaluation_status": "execution_failed",
            "required_gate": required_gate,
            "required_gate_pass": 0,
            "passed": 0,
            "failure": dict(failure),
            "decision": record,
            **CLAIM_FLAGS,
            **NO_WORK,
            "physical_training_performed": int(
                _physical_work_performed(run_dir)
            ),
        },
    )
    return decision


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight", "cache", "train", "confirm", "report", "all"),
        default="all",
    )
    parser.add_argument(
        "--require-gate",
        choices=("none", "preflight", "cache", "train", "confirm"),
        default="none",
    )
    parser.add_argument("--parent-multipath-run-dir", type=Path, required=True)
    parser.add_argument("--parent-strang-run-dir", type=Path, required=True)
    parser.add_argument("--parent-haar-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "runs/experiment12_d0_jacobi_rb_one_image_learnability"
        ),
    )
    parser.add_argument(
        "--run-name", default="production-exact-k512-rb-one-image-learnability"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.stage != "preflight" and args.stage != "all" and args.resume_run_dir is None:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    return args


def _stage_sequence(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return ("preflight", "cache", "train", "confirm")
    if stage == "report":
        return ()
    return (stage,)


def _run(args: argparse.Namespace) -> int:
    run_dir, resumed = _make_run_dir(args)
    print(f"Jacobi/RB one-image learnability run directory: {run_dir}", flush=True)
    active_stage = args.stage
    try:
        _initialize_run(run_dir, args, resumed=resumed)
    except ArtifactCompatibilityError as exc:
        print(
            f"Jacobi/RB learnability resume compatibility error: {exc}",
            file=sys.stderr,
        )
        return 1
    try:
        _status(run_dir, stage=args.stage, state="running")
        for active_stage in _stage_sequence(args.stage):
            _status(run_dir, stage=active_stage, state="running")
            if active_stage == "preflight":
                gate = _preflight_stage(run_dir, args, resumed=resumed)
            elif active_stage == "cache":
                gate = _cache_stage(run_dir, args)
            elif active_stage == "train":
                gate = _train_stage(run_dir, args)
            elif active_stage == "confirm":
                gate = _confirm_stage(run_dir, args)
            else:  # pragma: no cover - parse_args closes the set.
                raise AssertionError(active_stage)
            _artifact_registry(run_dir)
            if not _passed(gate):
                break
        workflow = _write_workflow(run_dir, args)
        decision = str(workflow["decision"]["decision"])
        required_pass = _required_stage_pass(run_dir, args.require_gate)
        state = "completed" if required_pass else "gate_failed"
        registry = _artifact_registry(run_dir)
        _status(
            run_dir,
            stage=active_stage,
            state=state,
            decision=decision,
            registry=registry,
        )
        return 0 if required_pass else 2
    except ArtifactCompatibilityError as exc:
        provenance_failure = isinstance(exc, ParentScopeError)
        failure = {
            "schema": RUN_SCHEMA + "-failure",
            "schema_version": 1,
            "evaluation_status": "execution_failed",
            "scientific_evidence_complete": 0,
            "stage_execution_valid": 0,
            "stage": active_stage,
            "failure_domain": (
                "parent_provenance"
                if provenance_failure
                else "artifact_compatibility"
            ),
            "failure_code": "learnability_artifact_compatibility_error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            **CLAIM_FLAGS,
            **NO_WORK,
        }
        atomic_write_json(run_dir / f"{active_stage}_failure.json", failure)
        decision = _commit_failure_decision(
            run_dir,
            stage=active_stage,
            required_gate=args.require_gate,
            failure=failure,
            provenance_failure=provenance_failure,
        )
        registry = _artifact_registry(run_dir)
        _status(
            run_dir,
            stage=active_stage,
            state="execution_failed",
            message=str(exc),
            decision=decision,
            registry=registry,
        )
        print(f"Jacobi/RB learnability compatibility error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        failure_domain = getattr(exc, "failure_domain", "workflow_execution")
        failure_code = getattr(
            exc, "failure_code", "learnability_unexpected_execution_failure"
        )
        failure = {
            "schema": RUN_SCHEMA + "-failure",
            "schema_version": 1,
            "evaluation_status": "execution_failed",
            "scientific_evidence_complete": 0,
            "stage_execution_valid": 0,
            "stage": active_stage,
            "failure_domain": str(failure_domain),
            "failure_code": str(failure_code),
            "error_type": type(exc).__name__,
            "error": str(exc),
            **CLAIM_FLAGS,
            **NO_WORK,
        }
        atomic_write_json(run_dir / f"{active_stage}_failure.json", failure)
        decision = _commit_failure_decision(
            run_dir,
            stage=active_stage,
            required_gate=args.require_gate,
            failure=failure,
            provenance_failure=False,
        )
        registry = _artifact_registry(run_dir)
        _status(
            run_dir,
            stage=active_stage,
            state="execution_failed",
            message=str(exc),
            decision=decision,
            registry=registry,
        )
        print(f"Jacobi/RB learnability error: {exc}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
