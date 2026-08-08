"""Immutable parent binding for noisy-Jacobi Bayes power calibration.

The bound parent is the terminal exact-K=512 one-image experiment.  Its
confirmation is a sealed no-signal result: every implementation and numerical
gate passed, while the learned model failed only to beat the analytic zero
predictor.

The new calibration may use the parent's separated input caches as exposure
templates, but it must never open the physical Rao--Blackwell label-audit
files.  This module makes that distinction explicit.  The registry itself
binds the forbidden files' historical digests; verification deliberately does
not read their bytes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
)


SCHEMA = "experiment12-d0-jacobi-rb-bayes-power-calibration-provenance"
SCHEMA_VERSION = 1

PARENT_RUN_BASENAME = (
    "20260729-015817_production-exact-k512-rb-one-image-learnability"
)
PARENT_RUN_SCHEMA = "experiment12-d0-jacobi-rb-one-image-learnability"
PARENT_REGISTRY_SCHEMA = PARENT_RUN_SCHEMA + "-artifact-registry"
PARENT_REGISTRY_RECORD_COUNT = 544
PARENT_REGISTRY_SEMANTIC_SHA256 = (
    "5e0b46328b6783614bdb7d394587b32e63d2d33b76f0279abdab6ecdf7d4e18a"
)
PARENT_REGISTRY_FILE_SHA256 = (
    "26370722f9f7ce5a6675bc3b626710373b407f4a4134a3425c852af8b17259a5"
)
PARENT_REGISTRY_FILE_SIZE = 99_431
PARENT_SOURCE_FINGERPRINT = (
    "f651d7322384275f269de3442f8e7a03cf062994b6bd894db735541d9f2a699d"
)
PARENT_SCIENTIFIC_CONFIG_SHA256 = (
    "58ccdfc5df2c4b30c28da5a143aa2570e390b007e7825b89d164762d7d23b01c"
)
PARENT_DECISION = "no_detectable_one_image_conditional_signal"
PARENT_ONLY_FAILED_CONFIRMATION_CHECK = "aggregate_model_beats_zero"

PARENT_LABEL_AUDIT_RECORDS = {
    "cache/confirmation_labels_audit.npz": {
        "sha256": "d1a5e218bc542ba7fa7bc120a9f2c03b8fd3f63b09a51350ca683804a8cd8ac9",
        "size": 5_414_301,
    },
    "cache/train_labels_audit.npz": {
        "sha256": "f7ea5f3bbeae2cbd716f33729cd90d0bf5a7af99edfa84b37791fd5d138d31b3",
        "size": 5_414_478,
    },
    "cache/validation_labels_audit.npz": {
        "sha256": "bb88dd33a2fcce971dff45d9918a7c9f1aa5b6886a326a2b4d3409d360bea8bf",
        "size": 5_414_360,
    },
}

# These are the only parent artifacts the calibration data builder may open.
# Provenance verification itself reads the JSON evidence listed separately.
PARENT_TEMPLATE_ALLOWLIST = frozenset(
    {
        "cache/train_inputs.npz",
        "cache/validation_inputs.npz",
        "cache/confirmation_inputs.npz",
        "scientific_config.json",
    }
)

_REGISTRY_EXCLUDED = frozenset({"artifact_registry.json", "run_status.json"})
_REQUIRED_JSON = {
    "manifest": "run_manifest.json",
    "status": "run_status.json",
    "scientific_config": "scientific_config.json",
    "decision": "learnability_decision.json",
    "workflow": "workflow_gate.json",
    "preflight": "preflight_gate.json",
    "cache": "cache_gate.json",
    "train_cache": "train_cache_gate.json",
    "validation_cache": "validation_cache_gate.json",
    "confirmation_cache": "confirmation_cache_gate.json",
    "teacher": "teacher_gate.json",
    "physical": "physical_gate.json",
    "confirmation": "confirmation_gate.json",
    "confirmation_seal": "confirmation_seal.json",
    "confirmation_open": "confirmation_open.json",
    "selected_model": "selected_model.json",
}
_ZERO_SCOPE_FIELDS = (
    "production_refinement_performed",
    "sampling_performed",
    "reverse_sampling_performed",
    "sampling_authorized",
    "reverse_sampling_authorized",
    "reconstruction_claim_authorized",
    "full_dataset_training_authorized",
    "known_prior_claim_authorized",
    "state_dependent_strang_refinement_established",
    "unsplit_generator_approximation_authorized",
    "spatial_dirichlet_ferguson_claim_authorized",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactCompatibilityError(message)


def _one(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 1


def _zero(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 0


def _load(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"cannot read {description} artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"{description} artifact is not an object")
    return dict(value)


def _safe_relative_path(value: str) -> str:
    path = Path(str(value))
    _require(
        bool(str(value))
        and not path.is_absolute()
        and ".." not in path.parts,
        f"unsafe parent artifact path: {value}",
    )
    return path.as_posix()


def is_parent_physical_label_path(value: str | Path) -> bool:
    """Return whether a path names a separated parent physical-label cache."""

    normalized = Path(str(value).replace("\\", "/")).as_posix().lower()
    return normalized.endswith("_labels_audit.npz")


def assert_parent_label_firewall(paths: Iterable[str | Path]) -> tuple[str, ...]:
    """Reject any attempted access to a parent physical-label artifact."""

    normalized = tuple(Path(str(path).replace("\\", "/")).as_posix() for path in paths)
    forbidden = sorted(path for path in normalized if is_parent_physical_label_path(path))
    _require(
        not forbidden,
        "physical parent label access is forbidden: " + ", ".join(forbidden),
    )
    return normalized


def _relative_access_path(root: Path, value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ArtifactCompatibilityError(
                f"parent template access escapes immutable run: {path}"
            ) from exc
    return _safe_relative_path(path.as_posix())


def validate_parent_template_access(
    run_dir: str | Path,
    paths: Iterable[str | Path],
) -> tuple[str, ...]:
    """Validate the complete set of parent files opened by a data builder."""

    root = Path(run_dir).resolve()
    relative = tuple(_relative_access_path(root, path) for path in paths)
    assert_parent_label_firewall(relative)
    unexpected = sorted(set(relative) - PARENT_TEMPLATE_ALLOWLIST)
    _require(
        not unexpected,
        "parent template access is not allowlisted: " + ", ".join(unexpected),
    )
    _require(
        len(relative) == len(set(relative)),
        "parent template access list contains duplicates",
    )
    for name in relative:
        _require((root / name).is_file(), f"parent template artifact is missing: {name}")
    return relative


def _zero_scope_tree(value: Any, description: str) -> None:
    if isinstance(value, Mapping):
        for name, item in value.items():
            if name in _ZERO_SCOPE_FIELDS:
                _require(
                    _zero(item),
                    f"{description} records forbidden scope/work in {name}",
                )
            _zero_scope_tree(item, description)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            _zero_scope_tree(item, description)


def _failed_subchecks(gate: Mapping[str, Any]) -> set[str]:
    raw = gate.get("subchecks")
    _require(isinstance(raw, Mapping), "parent confirmation subchecks are malformed")
    return {
        str(name)
        for name, check in raw.items()
        if not isinstance(check, Mapping) or not _one(check.get("passed"))
    }


def _verify_registry(
    root: Path,
    status: Mapping[str, Any],
    *,
    verify_nonlabel_hashes: bool,
) -> dict[str, Any]:
    """Verify the terminal registry without opening physical label NPZ bytes."""

    path = root / "artifact_registry.json"
    _require(path.is_file(), "parent artifact registry is missing")
    _require(
        path.stat().st_size == PARENT_REGISTRY_FILE_SIZE,
        "parent artifact registry size changed",
    )
    _require(
        file_fingerprint(path) == PARENT_REGISTRY_FILE_SHA256,
        "parent artifact registry file SHA-256 changed",
    )
    registry = _load(path, "parent registry")
    _require(
        registry.get("schema") == PARENT_REGISTRY_SCHEMA
        and registry.get("schema_version") == 1,
        "parent artifact registry schema changed",
    )
    raw_records = registry.get("records")
    _require(isinstance(raw_records, list), "parent registry records are malformed")
    _require(
        registry.get("record_count") == PARENT_REGISTRY_RECORD_COUNT
        and len(raw_records) == PARENT_REGISTRY_RECORD_COUNT,
        "parent artifact registry count changed",
    )
    _require(
        registry.get("registry_sha256") == PARENT_REGISTRY_SEMANTIC_SHA256
        and config_fingerprint(raw_records) == PARENT_REGISTRY_SEMANTIC_SHA256,
        "parent artifact registry semantic SHA-256 changed",
    )
    _require(
        status.get("artifact_registry_record_count")
        == PARENT_REGISTRY_RECORD_COUNT
        and status.get("artifact_registry_sha256")
        == PARENT_REGISTRY_SEMANTIC_SHA256
        and status.get("artifact_registry_file_sha256")
        == PARENT_REGISTRY_FILE_SHA256
        and status.get("artifact_registry_file_size")
        == PARENT_REGISTRY_FILE_SIZE,
        "parent terminal status does not bind its registry",
    )

    records: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for raw in raw_records:
        _require(isinstance(raw, Mapping), "parent registry row is malformed")
        relative = _safe_relative_path(str(raw.get("path", "")))
        _require(relative not in records, f"duplicate parent registry path: {relative}")
        size = raw.get("size")
        digest = raw.get("sha256")
        _require(
            isinstance(size, int)
            and not isinstance(size, bool)
            and size >= 0
            and isinstance(digest, str)
            and len(digest) == 64,
            f"parent registry row is malformed: {relative}",
        )
        records[relative] = {"path": relative, "size": size, "sha256": digest}
        order.append(relative)
    _require(order == sorted(order), "parent artifact registry ordering changed")
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.name not in _REGISTRY_EXCLUDED
    }
    _require(actual == set(records), "parent terminal artifact file set changed")

    for relative, record in records.items():
        artifact = root / relative
        _require(
            artifact.is_file() and artifact.stat().st_size == record["size"],
            f"registered parent artifact size changed: {relative}",
        )
        if verify_nonlabel_hashes and not is_parent_physical_label_path(relative):
            _require(
                file_fingerprint(artifact) == record["sha256"],
                f"registered parent artifact SHA-256 changed: {relative}",
            )

    actual_label_records = {
        name: {"sha256": records[name]["sha256"], "size": records[name]["size"]}
        for name in records
        if is_parent_physical_label_path(name)
    }
    _require(
        actual_label_records == PARENT_LABEL_AUDIT_RECORDS,
        "parent physical-label registry binding changed",
    )
    _zero_scope_tree(registry, "parent terminal registry")
    return registry


def verify_no_signal_parent(
    run_dir: str | Path,
    *,
    accessed_parent_paths: Iterable[str | Path] = (),
    verify_nonlabel_hashes: bool = True,
) -> dict[str, Any]:
    """Verify the immutable sealed no-signal parent and access firewall."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"one-image parent does not exist: {root}")
    _require(
        root.name == PARENT_RUN_BASENAME,
        f"one-image parent basename must be {PARENT_RUN_BASENAME}",
    )
    artifacts = {
        name: _load(root / filename, f"parent {name}")
        for name, filename in _REQUIRED_JSON.items()
    }
    status = artifacts["status"]
    registry = _verify_registry(
        root,
        status,
        verify_nonlabel_hashes=bool(verify_nonlabel_hashes),
    )
    manifest = artifacts["manifest"]
    scientific = artifacts["scientific_config"]
    decision = artifacts["decision"]
    confirmation = artifacts["confirmation"]

    _require(
        manifest.get("schema") == PARENT_RUN_SCHEMA
        and manifest.get("schema_version") == 1
        and manifest.get("source_fingerprint") == PARENT_SOURCE_FINGERPRINT
        and manifest.get("scientific_config_sha256")
        == PARENT_SCIENTIFIC_CONFIG_SHA256,
        "parent manifest source/config binding changed",
    )
    _require(
        scientific.get("semantic_sha256") == PARENT_SCIENTIFIC_CONFIG_SHA256
        and scientific.get("schema") == PARENT_RUN_SCHEMA + "-scientific-config",
        "parent scientific configuration changed",
    )
    _require(
        status.get("schema") == PARENT_RUN_SCHEMA + "-status"
        and status.get("stage") == "confirm"
        and status.get("state") == "gate_failed"
        and status.get("decision") == PARENT_DECISION
        and _one(status.get("physical_training_performed")),
        "parent terminal status changed",
    )
    _require(
        decision.get("decision") == PARENT_DECISION
        and decision.get("evaluation_status") == "evaluated"
        and _zero(
            decision.get(
                "larger_exact_discrete_chain_training_planning_authorized"
            )
        ),
        "parent terminal decision changed",
    )
    for name in (
        "preflight",
        "cache",
        "train_cache",
        "validation_cache",
        "confirmation_cache",
        "teacher",
        "physical",
    ):
        gate = artifacts[name]
        _require(
            gate.get("evaluation_status") == "evaluated"
            and _one(gate.get("passed")),
            f"parent {name} gate is not a pass",
        )
    _require(
        confirmation.get("evaluation_status") == "evaluated"
        and _zero(confirmation.get("passed"))
        and _one(confirmation.get("cache_valid"))
        and _failed_subchecks(confirmation)
        == {PARENT_ONLY_FAILED_CONFIRMATION_CHECK},
        "parent confirmation is not the sealed aggregate-zero-only failure",
    )
    seal = artifacts["confirmation_seal"]
    opened = artifacts["confirmation_open"]
    _require(
        seal.get("schema") == PARENT_RUN_SCHEMA + "-confirmation-seal"
        and seal.get("schema_version") == 1
        and _zero(seal.get("confirmation_opened"))
        and isinstance(seal.get("seal_sha256"), str)
        and len(seal["seal_sha256"]) == 64
        and opened.get("schema") == PARENT_RUN_SCHEMA + "-confirmation-open"
        and _one(opened.get("opened_count"))
        and _zero(opened.get("panel_regenerated"))
        and _zero(opened.get("panel_resized"))
        and opened.get("seal_sha256") == seal.get("seal_sha256"),
        "parent confirmation seal/open binding is malformed",
    )
    # The selected model and seal are historical evidence only; they are never
    # reused by the new synthetic calibration.
    _require(
        isinstance(artifacts["selected_model"].get("checkpoint_file_sha256"), str)
        and len(artifacts["selected_model"]["checkpoint_file_sha256"]) == 64
        and artifacts["selected_model"]["checkpoint_file_sha256"]
        == seal.get("selected_model_file_sha256"),
        "parent selected-model seal changed",
    )
    for description, record in artifacts.items():
        _zero_scope_tree(record, f"parent {description}")

    accessed = validate_parent_template_access(root, accessed_parent_paths)
    raw_registry_records = registry["records"]
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": 1,
        "parent_run_dir": str(root),
        "parent_run_basename": PARENT_RUN_BASENAME,
        "parent_registry_record_count": len(raw_registry_records),
        "parent_registry_semantic_sha256": PARENT_REGISTRY_SEMANTIC_SHA256,
        "parent_registry_file_sha256": PARENT_REGISTRY_FILE_SHA256,
        "parent_source_fingerprint": PARENT_SOURCE_FINGERPRINT,
        "parent_scientific_config_sha256": PARENT_SCIENTIFIC_CONFIG_SHA256,
        "parent_decision": PARENT_DECISION,
        "parent_provenance_pass": 1,
        "parent_registry_pass": 1,
        "parent_terminal_no_signal_pass": 1,
        "parent_only_aggregate_zero_failure_pass": 1,
        "parent_exact_cache_pass": 1,
        "parent_teacher_pass": 1,
        "parent_optimizer_pass": 1,
        "parent_seal_pass": 1,
        "parent_no_sampling_pass": 1,
        "parent_label_firewall_pass": 1,
        "parent_template_allowlist_pass": 1,
        "source_binding_pass": 1,
        "accessed_parent_paths": list(accessed),
        "forbidden_parent_label_paths": sorted(PARENT_LABEL_AUDIT_RECORDS),
        "forbidden_label_bytes_opened": 0,
        "nonlabel_artifact_hashes_verified": int(bool(verify_nonlabel_hashes)),
        "parent_mutated": 0,
        "fresh_physical_witness_planning_authorized": 0,
        **{name: 0 for name in _ZERO_SCOPE_FIELDS},
        "physical_training_performed": 0,
    }


# Descriptive aliases for callers.
verify_bayes_power_parent = verify_no_signal_parent
verify_parent = verify_no_signal_parent


__all__ = [
    "PARENT_DECISION",
    "PARENT_LABEL_AUDIT_RECORDS",
    "PARENT_ONLY_FAILED_CONFIRMATION_CHECK",
    "PARENT_REGISTRY_FILE_SHA256",
    "PARENT_REGISTRY_RECORD_COUNT",
    "PARENT_REGISTRY_SEMANTIC_SHA256",
    "PARENT_RUN_BASENAME",
    "PARENT_SCIENTIFIC_CONFIG_SHA256",
    "PARENT_SOURCE_FINGERPRINT",
    "PARENT_TEMPLATE_ALLOWLIST",
    "SCHEMA",
    "SCHEMA_VERSION",
    "assert_parent_label_firewall",
    "is_parent_physical_label_path",
    "validate_parent_template_access",
    "verify_bayes_power_parent",
    "verify_no_signal_parent",
    "verify_parent",
]
