"""Immutable-parent verification for the Jacobi/RB zero-signal diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)


EXPECTED_PARENT_BASENAME = (
    "20260729-015817_production-exact-k512-rb-one-image-learnability"
)
EXPECTED_REGISTRY_COUNT = 544
EXPECTED_REGISTRY_SHA256 = (
    "5e0b46328b6783614bdb7d394587b32e63d2d33b76f0279abdab6ecdf7d4e18a"
)
EXPECTED_SOURCE_FINGERPRINT = (
    "f651d7322384275f269de3442f8e7a03cf062994b6bd894db735541d9f2a699d"
)
EXPECTED_SCIENTIFIC_CONFIG_SHA256 = (
    "58ccdfc5df2c4b30c28da5a143aa2570e390b007e7825b89d164762d7d23b01c"
)
EXPECTED_SELECTED_MODEL_FILE_SHA256 = (
    "8852ec2b501cb4a38e539127a0357d632b1134e9b826b4276ebb77094a999401"
)
EXPECTED_SELECTED_STATE_SHA256 = (
    "30bab0e25742d2a385af8ba6f60f35c0f266d8f1fb9f11baf33225306188a9d6"
)
EXPECTED_CONFIRMATION_SEAL_SHA256 = (
    "bdb2cb495c6e30d8be30e1f1f0de78267f884d444bb3e45335ce98e3ec09dcfe"
)
EXPECTED_DECISION = "no_detectable_one_image_conditional_signal"


class ZeroSignalParentError(ArtifactCompatibilityError):
    """The completed one-image run does not match the frozen diagnostic scope."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ZeroSignalParentError(f"cannot read parent artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ZeroSignalParentError(f"parent artifact is not an object: {path}")
    return dict(value)


def _require_json(run_dir: Path, name: str) -> dict[str, Any]:
    path = run_dir / name
    if not path.is_file():
        raise ZeroSignalParentError(f"parent artifact is missing: {name}")
    return _load_json(path)


def _passed(gate: Mapping[str, Any]) -> bool:
    return (
        gate.get("evaluation_status") == "evaluated"
        and int(gate.get("passed", 0)) == 1
    )


def _verify_registry(run_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    registry_path = run_dir / "artifact_registry.json"
    status_path = run_dir / "run_status.json"
    registry = _require_json(run_dir, "artifact_registry.json")
    records = registry.get("records")
    if not isinstance(records, list):
        raise ZeroSignalParentError("parent registry records are malformed")
    if (
        int(registry.get("record_count", -1)) != EXPECTED_REGISTRY_COUNT
        or len(records) != EXPECTED_REGISTRY_COUNT
        or registry.get("registry_sha256") != EXPECTED_REGISTRY_SHA256
        or config_fingerprint(records) != EXPECTED_REGISTRY_SHA256
    ):
        raise ZeroSignalParentError("parent terminal registry binding changed")

    by_path: dict[str, dict[str, Any]] = {}
    for row in records:
        if not isinstance(row, Mapping):
            raise ZeroSignalParentError("parent registry row is malformed")
        relative = str(row.get("path", ""))
        if (
            not relative
            or relative in by_path
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ZeroSignalParentError("parent registry path is invalid")
        path = run_dir / relative
        if (
            not path.is_file()
            or int(row.get("size", -1)) != path.stat().st_size
            or row.get("sha256") != file_fingerprint(path)
        ):
            raise ZeroSignalParentError(f"parent artifact changed: {relative}")
        by_path[relative] = dict(row)

    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.name not in {"artifact_registry.json", "run_status.json"}
    }
    if actual != set(by_path):
        raise ZeroSignalParentError("parent terminal registry file set changed")

    status = _load_json(status_path)
    expected_status_binding = {
        "artifact_registry_record_count": EXPECTED_REGISTRY_COUNT,
        "artifact_registry_sha256": EXPECTED_REGISTRY_SHA256,
        "artifact_registry_file_sha256": file_fingerprint(registry_path),
        "artifact_registry_file_size": int(registry_path.stat().st_size),
    }
    if any(status.get(key) != value for key, value in expected_status_binding.items()):
        raise ZeroSignalParentError("parent status no longer binds its terminal registry")
    return registry, by_path


def verify_zero_signal_parent(path: str | Path) -> dict[str, Any]:
    """Verify the exact immutable negative one-image run and return its binding."""

    run_dir = Path(path).resolve()
    if not run_dir.is_dir() or run_dir.name != EXPECTED_PARENT_BASENAME:
        raise ZeroSignalParentError("unexpected one-image parent run")
    _, registry = _verify_registry(run_dir)

    status = _require_json(run_dir, "run_status.json")
    manifest = _require_json(run_dir, "run_manifest.json")
    decision = _require_json(run_dir, "learnability_decision.json")
    confirmation_gate = _require_json(run_dir, "confirmation_gate.json")
    confirmation_cache_gate = _require_json(run_dir, "confirmation_cache_gate.json")
    seal = _require_json(run_dir, "confirmation_seal.json")
    selected = _require_json(run_dir, "selected_model.json")
    confirmation_open = _require_json(run_dir, "confirmation_open.json")

    if (
        status.get("stage") != "confirm"
        or status.get("state") != "gate_failed"
        or status.get("decision") != EXPECTED_DECISION
        or decision.get("decision") != EXPECTED_DECISION
        or decision.get("evaluation_status") != "evaluated"
    ):
        raise ZeroSignalParentError("parent is not the frozen negative confirmation")
    if (
        manifest.get("source_fingerprint") != EXPECTED_SOURCE_FINGERPRINT
        or manifest.get("scientific_config_sha256")
        != EXPECTED_SCIENTIFIC_CONFIG_SHA256
    ):
        raise ZeroSignalParentError("parent source or scientific configuration changed")
    source_paths = manifest.get("source_paths")
    if not isinstance(source_paths, list) or not source_paths:
        raise ZeroSignalParentError("parent source path list is missing")
    repository_root = Path(__file__).resolve().parents[1]
    resolved_sources = []
    for value in source_paths:
        relative = Path(str(value))
        if relative.is_absolute() or ".." in relative.parts:
            raise ZeroSignalParentError("parent source path is invalid")
        resolved = repository_root / relative
        if not resolved.is_file():
            raise ZeroSignalParentError(f"parent source is missing: {relative}")
        resolved_sources.append(resolved)
    if source_fingerprint(resolved_sources) != EXPECTED_SOURCE_FINGERPRINT:
        raise ZeroSignalParentError("current parent source bytes changed")
    if not _passed(confirmation_cache_gate):
        raise ZeroSignalParentError("parent confirmation cache did not pass")
    if (
        confirmation_gate.get("evaluation_status") != "evaluated"
        or int(confirmation_gate.get("passed", 1)) != 0
    ):
        raise ZeroSignalParentError("parent confirmation gate scope changed")
    failed = sorted(
        name
        for name, row in dict(confirmation_gate.get("subchecks", {})).items()
        if int(dict(row).get("passed", 0)) != 1
    )
    if failed != ["aggregate_model_beats_zero"]:
        raise ZeroSignalParentError("parent failed checks are not the frozen singleton")

    for gate_name in (
        "preflight_gate.json",
        "cache_gate.json",
        "teacher_gate.json",
        "physical_gate.json",
        "train_cache_gate.json",
        "validation_cache_gate.json",
    ):
        if not _passed(_require_json(run_dir, gate_name)):
            raise ZeroSignalParentError(f"required parent gate did not pass: {gate_name}")

    if (
        seal.get("seal_sha256") != EXPECTED_CONFIRMATION_SEAL_SHA256
        or seal.get("selected_model_file_sha256")
        != EXPECTED_SELECTED_MODEL_FILE_SHA256
        or seal.get("selected_state_sha256") != EXPECTED_SELECTED_STATE_SHA256
        or selected.get("checkpoint_file_sha256")
        != EXPECTED_SELECTED_MODEL_FILE_SHA256
        or selected.get("state_sha256") != EXPECTED_SELECTED_STATE_SHA256
        or file_fingerprint(run_dir / "selected_model.pt")
        != EXPECTED_SELECTED_MODEL_FILE_SHA256
        or int(confirmation_open.get("opened_count", -1)) != 1
    ):
        raise ZeroSignalParentError("selected model or one-time confirmation seal changed")

    baseline_hash = str(seal.get("metadata_baseline_file_sha256", ""))
    if file_fingerprint(run_dir / "metadata_baseline.npz") != baseline_hash:
        raise ZeroSignalParentError("training-only metadata baseline changed")

    cache_files: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "confirmation"):
        for suffix in ("inputs.npz", "labels_audit.npz"):
            relative = f"cache/{split}_{suffix}"
            if relative not in registry:
                raise ZeroSignalParentError(f"parent cache is unregistered: {relative}")
            cache_files[relative] = registry[relative]

    for record in (status, manifest, decision):
        if any(
            int(record.get(name, 0)) != 0
            for name in (
                "sampling_performed",
                "reverse_sampling_performed",
                "production_refinement_performed",
            )
        ):
            raise ZeroSignalParentError("parent performed work outside diagnostic scope")

    binding = {
        "schema": "d0-jacobi-rb-zero-signal-parent-binding-v1",
        "schema_version": 1,
        "parent_run_dir": str(run_dir),
        "parent_basename": run_dir.name,
        "registry_record_count": EXPECTED_REGISTRY_COUNT,
        "registry_sha256": EXPECTED_REGISTRY_SHA256,
        "registry_file_sha256": file_fingerprint(
            run_dir / "artifact_registry.json"
        ),
        "run_status_sha256": file_fingerprint(run_dir / "run_status.json"),
        "source_fingerprint": EXPECTED_SOURCE_FINGERPRINT,
        "current_parent_source_verified": 1,
        "scientific_config_sha256": EXPECTED_SCIENTIFIC_CONFIG_SHA256,
        "terminal_decision": EXPECTED_DECISION,
        "confirmation_failed_subchecks": failed,
        "selected_model_file_sha256": EXPECTED_SELECTED_MODEL_FILE_SHA256,
        "selected_state_sha256": EXPECTED_SELECTED_STATE_SHA256,
        "confirmation_seal_sha256": EXPECTED_CONFIRMATION_SEAL_SHA256,
        "metadata_baseline_file_sha256": baseline_hash,
        "confirmation_opened_count": 1,
        "cache_files": cache_files,
        "parent_read_only": 1,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
        "production_refinement_performed": 0,
    }
    binding["semantic_sha256"] = config_fingerprint(binding)
    return binding


__all__ = [
    "EXPECTED_CONFIRMATION_SEAL_SHA256",
    "EXPECTED_DECISION",
    "EXPECTED_PARENT_BASENAME",
    "EXPECTED_REGISTRY_COUNT",
    "EXPECTED_REGISTRY_SHA256",
    "EXPECTED_SCIENTIFIC_CONFIG_SHA256",
    "EXPECTED_SELECTED_MODEL_FILE_SHA256",
    "EXPECTED_SELECTED_STATE_SHA256",
    "EXPECTED_SOURCE_FINGERPRINT",
    "ZeroSignalParentError",
    "verify_zero_signal_parent",
]
