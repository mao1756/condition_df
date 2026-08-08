"""Immutable parent bindings for the physical coarse-signal witness.

The witness is a fresh scientific gate, but its motivation depends on three
completed runs:

* the exact-K=512 one-image learnability run;
* the sealed zero-signal diagnostic of its selected model; and
* the noisy-Jacobi Bayes-power calibration.

This module verifies those runs without importing a trainer, transition
sampler, or reconstruction workflow.  All registered artifacts are hashed,
the terminal decisions and gates are checked, and the two diagnostic
descendants are required to bind the same immutable physical parent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from mnist.d0_jacobi_artifacts import source_fingerprint
from mnist.d0_one_image_gate import ArtifactCompatibilityError


SCHEMA = "experiment12-d0-jacobi-rb-physical-coarse-signal-provenance"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GateSpec:
    path: str
    schema: str
    passed: int


@dataclass(frozen=True)
class ParentSpec:
    role: str
    basename: str
    run_schema: str
    config_schema: str
    status_schema: str
    registry_schema: str
    registry_record_count: int
    registry_sha256: str
    registry_file_sha256: str
    source_fingerprint: str
    scientific_config_sha256: str
    terminal_state: str
    terminal_stage: str
    terminal_status_decision: str
    decision_path: str
    decision_schema: str
    decision: str
    physical_training_performed: int
    gates: tuple[GateSpec, ...]


PHYSICAL_PARENT = ParentSpec(
    role="physical_one_image",
    basename="20260729-015817_production-exact-k512-rb-one-image-learnability",
    run_schema="experiment12-d0-jacobi-rb-one-image-learnability",
    config_schema=(
        "experiment12-d0-jacobi-rb-one-image-learnability-scientific-config"
    ),
    status_schema="experiment12-d0-jacobi-rb-one-image-learnability-status",
    registry_schema=(
        "experiment12-d0-jacobi-rb-one-image-learnability-artifact-registry"
    ),
    registry_record_count=544,
    registry_sha256=(
        "5e0b46328b6783614bdb7d394587b32e63d2d33b76f0279abdab6ecdf7d4e18a"
    ),
    registry_file_sha256=(
        "26370722f9f7ce5a6675bc3b626710373b407f4a4134a3425c852af8b17259a5"
    ),
    source_fingerprint=(
        "f651d7322384275f269de3442f8e7a03cf062994b6bd894db735541d9f2a699d"
    ),
    scientific_config_sha256=(
        "58ccdfc5df2c4b30c28da5a143aa2570e390b007e7825b89d164762d7d23b01c"
    ),
    terminal_state="gate_failed",
    terminal_stage="confirm",
    terminal_status_decision="no_detectable_one_image_conditional_signal",
    decision_path="learnability_decision.json",
    decision_schema=(
        "experiment12-d0-jacobi-rb-one-image-learnability-gate-decision"
    ),
    decision="no_detectable_one_image_conditional_signal",
    physical_training_performed=1,
    gates=(
        GateSpec(
            "preflight_gate.json",
            "experiment12-d0-jacobi-rb-one-image-learnability-gate",
            1,
        ),
        GateSpec(
            "cache_gate.json",
            "experiment12-d0-jacobi-rb-one-image-learnability-cache-gate",
            1,
        ),
        GateSpec(
            "teacher_gate.json",
            "experiment12-d0-jacobi-rb-one-image-learnability-gate",
            1,
        ),
        GateSpec(
            "physical_gate.json",
            "experiment12-d0-jacobi-rb-one-image-learnability-gate",
            1,
        ),
        GateSpec(
            "confirmation_gate.json",
            "experiment12-d0-jacobi-rb-one-image-learnability-gate",
            0,
        ),
    ),
)

ZERO_SIGNAL_PARENT = ParentSpec(
    role="zero_signal_diagnostic",
    basename="20260730-010919_production-sealed-rb-zero-signal-diagnostic",
    run_schema="experiment12-d0-jacobi-rb-zero-signal-diagnostic",
    config_schema="d0-jacobi-rb-zero-signal-diagnostic-v1-scientific-config",
    status_schema="experiment12-d0-jacobi-rb-zero-signal-diagnostic-status",
    registry_schema=(
        "experiment12-d0-jacobi-rb-zero-signal-diagnostic-artifact-registry"
    ),
    registry_record_count=18,
    registry_sha256=(
        "11d0a7272dd83b6535c1bc4426ad471f929ec0a1cd2f9c96e8ac80f01483a5e3"
    ),
    registry_file_sha256=(
        "106906fb6a6ca48309ceb470a0798bc344fb498f7a0ac161863e004fab224b4d"
    ),
    source_fingerprint=(
        "8a8ad169a2d520cd5047020a45594001c49ac9dcb605b57041cc73323cc79e0e"
    ),
    scientific_config_sha256=(
        "4af898365c85d8edb8b41e12ceea85db453d817a069a8506b7e35d4e87120554"
    ),
    terminal_state="completed",
    terminal_stage="analyze",
    terminal_status_decision="zero_signal_diagnostic_complete",
    decision_path="zero_signal_decision.json",
    decision_schema="d0-jacobi-rb-zero-signal-diagnostic-decision-v1",
    decision="zero_signal_diagnostic_complete",
    physical_training_performed=0,
    gates=(
        GateSpec(
            "zero_signal_preflight_gate.json",
            "d0-jacobi-rb-zero-signal-diagnostic-gate-v1",
            1,
        ),
        GateSpec(
            "zero_signal_analysis_gate.json",
            "d0-jacobi-rb-zero-signal-diagnostic-gate-v1",
            1,
        ),
    ),
)

BAYES_POWER_PARENT = ParentSpec(
    role="bayes_power_calibration",
    basename="20260730-012459_production-noisy-jacobi-bayes-power",
    run_schema="experiment12-d0-jacobi-rb-bayes-power-calibration",
    config_schema=(
        "experiment12-d0-jacobi-rb-bayes-power-calibration-scientific-config"
    ),
    status_schema="experiment12-d0-jacobi-rb-bayes-power-calibration-status",
    registry_schema=(
        "experiment12-d0-jacobi-rb-bayes-power-calibration-artifact-registry"
    ),
    registry_record_count=74,
    registry_sha256=(
        "01b5d772299611e9e17b886658b7eba80a7ab50805241e94d2e9a8ba36562e79"
    ),
    registry_file_sha256=(
        "4caa9597f1ce7e6e6180ea11bffe55138f10582791b60e5d529e38d9e3b13bec"
    ),
    source_fingerprint=(
        "bbd522fb4ce2219e6759d5e0c78b8fc1baa8c4f39c8fe356f902f676ec1e7462"
    ),
    scientific_config_sha256=(
        "05cdd8b9b2b03920ef51d099f4b29589b66297fe6664ff6b052aa5f59d08d1ac"
    ),
    terminal_state="completed",
    terminal_stage="confirm",
    terminal_status_decision="noisy_bayes_detection_pipeline_calibrated",
    decision_path="bayes_power_decision.json",
    decision_schema=(
        "experiment12-d0-jacobi-rb-bayes-power-calibration-gate-decision"
    ),
    decision="noisy_bayes_detection_pipeline_calibrated",
    physical_training_performed=0,
    gates=(
        GateSpec(
            "preflight_gate.json",
            "experiment12-d0-jacobi-rb-bayes-power-calibration-gate",
            1,
        ),
        GateSpec(
            "cache_gate.json",
            "experiment12-d0-jacobi-rb-bayes-power-calibration-gate",
            1,
        ),
        GateSpec(
            "train_gate.json",
            "experiment12-d0-jacobi-rb-bayes-power-calibration-gate",
            1,
        ),
        GateSpec(
            "controls_gate.json",
            "experiment12-d0-jacobi-rb-bayes-power-calibration-gate",
            1,
        ),
    ),
)

PARENT_SPECS: Mapping[str, ParentSpec] = {
    PHYSICAL_PARENT.role: PHYSICAL_PARENT,
    ZERO_SIGNAL_PARENT.role: ZERO_SIGNAL_PARENT,
    BAYES_POWER_PARENT.role: BAYES_POWER_PARENT,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactCompatibilityError(message)


def _load_json(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"invalid {description}: {path}"
        ) from exc
    _require(isinstance(value, dict), f"{description} must be a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _registry_digest(registry: Mapping[str, Any]) -> str | None:
    value = registry.get("registry_sha256")
    if value is None:
        value = registry.get("semantic_sha256")
    return str(value) if value is not None else None


def _safe_artifact_path(root: Path, raw: Any) -> Path:
    _require(isinstance(raw, str) and raw != "", "registry path is invalid")
    relative = PurePosixPath(raw)
    _require(
        not relative.is_absolute()
        and all(part not in {"", ".", ".."} for part in relative.parts),
        f"unsafe registry path: {raw!r}",
    )
    target = (root / Path(*relative.parts)).resolve()
    _require(
        target != root and root in target.parents,
        f"registry path escapes parent run: {raw!r}",
    )
    return target


def _verify_registered_artifacts(
    root: Path, registry: Mapping[str, Any], spec: ParentSpec
) -> None:
    records = registry.get("records")
    _require(isinstance(records, list), f"{spec.role} registry records are invalid")
    _require(
        len(records) == spec.registry_record_count,
        f"{spec.role} registry record list changed",
    )
    observed: set[str] = set()
    for record in records:
        _require(
            isinstance(record, Mapping),
            f"{spec.role} registry record is not an object",
        )
        raw_path = record.get("path")
        _require(
            isinstance(raw_path, str) and raw_path not in observed,
            f"{spec.role} registry paths are invalid or duplicated",
        )
        observed.add(raw_path)
        target = _safe_artifact_path(root, raw_path)
        _require(target.is_file(), f"{spec.role} artifact is missing: {raw_path}")
        _require(
            target.stat().st_size == record.get("size"),
            f"{spec.role} artifact size changed: {raw_path}",
        )
        _require(
            _file_sha256(target) == record.get("sha256"),
            f"{spec.role} artifact hash changed: {raw_path}",
        )


def _verify_scope(
    value: Mapping[str, Any],
    *,
    role: str,
    expected_physical_training: int,
) -> None:
    _require(
        value.get("physical_training_performed") == expected_physical_training,
        f"{role} physical-training scope changed",
    )
    for field in (
        "sampling_performed",
        "reverse_sampling_performed",
        "production_refinement_performed",
    ):
        if field in value:
            _require(value.get(field) == 0, f"{role} unexpectedly changed {field}")
    for field in (
        "sampling_authorized",
        "reverse_sampling_authorized",
        "reconstruction_claim_authorized",
        "full_dataset_training_authorized",
        "production_refinement_authorized",
    ):
        if field in value:
            _require(value.get(field) == 0, f"{role} unexpectedly changed {field}")


def _verify_live_source_fingerprint(
    manifest: Mapping[str, Any], spec: ParentSpec
) -> None:
    """Recompute the parent fingerprint with its original path-label semantics."""

    raw_paths = manifest.get("source_paths")
    _require(
        isinstance(raw_paths, list)
        and bool(raw_paths)
        and all(isinstance(item, str) and item != "" for item in raw_paths),
        f"{spec.role} source_paths are invalid",
    )
    stored_paths = [Path(item) for item in raw_paths]
    if spec.role == PHYSICAL_PARENT.role:
        _require(
            all(not path.is_absolute() for path in stored_paths),
            "physical_one_image source path semantics changed",
        )
        # This manifest serialized relative display paths even though its source
        # fingerprint was built from the corresponding absolute workspace paths.
        fingerprint_paths = [path.resolve() for path in stored_paths]
    elif spec.role == ZERO_SIGNAL_PARENT.role:
        _require(
            all(not path.is_absolute() for path in stored_paths),
            "zero_signal_diagnostic source path semantics changed",
        )
        # The zero-signal workflow deliberately retained the relative labels in
        # the fingerprint records.
        fingerprint_paths = stored_paths
    else:
        _require(
            all(path.is_absolute() for path in stored_paths),
            "bayes_power_calibration source path semantics changed",
        )
        fingerprint_paths = stored_paths

    _require(
        len({path.as_posix() for path in fingerprint_paths})
        == len(fingerprint_paths),
        f"{spec.role} source_paths are duplicated",
    )
    _require(
        all(path.is_file() for path in fingerprint_paths),
        f"{spec.role} source file is missing",
    )
    try:
        observed = source_fingerprint(fingerprint_paths)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ArtifactCompatibilityError(
            f"{spec.role} live source fingerprint could not be computed"
        ) from exc
    _require(
        observed == spec.source_fingerprint,
        f"{spec.role} live source fingerprint changed",
    )


def _verify_parent(root_value: str | Path, spec: ParentSpec) -> dict[str, Any]:
    root = Path(root_value).resolve()
    _require(root.is_dir(), f"{spec.role} run directory does not exist: {root}")
    _require(root.name == spec.basename, f"wrong {spec.role} parent basename")

    registry_path = root / "artifact_registry.json"
    _require(
        registry_path.is_file(),
        f"missing {spec.role} artifact registry: {registry_path}",
    )
    _require(
        _file_sha256(registry_path) == spec.registry_file_sha256,
        f"{spec.role} registry file hash changed",
    )
    registry = _load_json(registry_path, f"{spec.role} artifact registry")
    _require(
        registry.get("schema") == spec.registry_schema
        and registry.get("schema_version") == 1,
        f"{spec.role} registry schema changed",
    )
    _require(
        registry.get("record_count") == spec.registry_record_count,
        f"{spec.role} registry count changed",
    )
    _require(
        _registry_digest(registry) == spec.registry_sha256,
        f"{spec.role} registry semantic hash changed",
    )
    _verify_registered_artifacts(root, registry, spec)

    manifest = _load_json(root / "run_manifest.json", f"{spec.role} manifest")
    scientific = _load_json(
        root / "scientific_config.json", f"{spec.role} scientific config"
    )
    status = _load_json(root / "run_status.json", f"{spec.role} status")
    decision = _load_json(root / spec.decision_path, f"{spec.role} decision")
    _require(
        manifest.get("schema") == spec.run_schema
        and manifest.get("schema_version") == 1,
        f"{spec.role} manifest schema changed",
    )
    _require(
        manifest.get("source_fingerprint") == spec.source_fingerprint,
        f"{spec.role} source fingerprint changed",
    )
    _verify_live_source_fingerprint(manifest, spec)
    _require(
        manifest.get("scientific_config_sha256")
        == spec.scientific_config_sha256,
        f"{spec.role} manifest scientific config changed",
    )
    _require(
        scientific.get("schema") == spec.config_schema
        and scientific.get("schema_version") == 1
        and scientific.get("semantic_sha256")
        == spec.scientific_config_sha256,
        f"{spec.role} scientific config binding changed",
    )
    _require(
        status.get("schema") == spec.status_schema
        and status.get("schema_version") == 1,
        f"{spec.role} status schema changed",
    )
    _require(
        status.get("state") == spec.terminal_state
        and status.get("stage") == spec.terminal_stage
        and status.get("decision") == spec.terminal_status_decision,
        f"{spec.role} terminal status changed",
    )
    _require(
        status.get("artifact_registry_record_count")
        == spec.registry_record_count
        and status.get("artifact_registry_sha256") == spec.registry_sha256
        and status.get("artifact_registry_file_sha256")
        == spec.registry_file_sha256
        and status.get("artifact_registry_file_size")
        == registry_path.stat().st_size,
        f"{spec.role} terminal registry binding changed",
    )
    _require(
        decision.get("schema") == spec.decision_schema
        and decision.get("schema_version") == 1
        and decision.get("evaluation_status") == "evaluated"
        and decision.get("decision") == spec.decision,
        f"{spec.role} terminal decision changed",
    )
    _verify_scope(
        registry,
        role=spec.role,
        expected_physical_training=spec.physical_training_performed,
    )
    _verify_scope(
        status,
        role=spec.role,
        expected_physical_training=spec.physical_training_performed,
    )
    _verify_scope(
        decision,
        role=spec.role,
        expected_physical_training=spec.physical_training_performed,
    )

    gates: dict[str, dict[str, Any]] = {}
    for gate_spec in spec.gates:
        gate = _load_json(root / gate_spec.path, f"{spec.role} {gate_spec.path}")
        _require(
            gate.get("schema") == gate_spec.schema
            and gate.get("schema_version") == 1
            and gate.get("evaluation_status") == "evaluated"
            and gate.get("passed") == gate_spec.passed,
            f"{spec.role} gate changed: {gate_spec.path}",
        )
        gates[gate_spec.path] = {
            "schema": gate_spec.schema,
            "passed": gate_spec.passed,
        }

    if spec.role == PHYSICAL_PARENT.role:
        confirmation = _load_json(
            root / "confirmation_gate.json", "physical confirmation gate"
        )
        failed = {
            name
            for name, check in confirmation.get("subchecks", {}).items()
            if isinstance(check, Mapping) and check.get("passed") != 1
        }
        _require(
            failed == {"aggregate_model_beats_zero"},
            "physical parent did not fail only aggregate_model_beats_zero",
        )
    elif spec.role == ZERO_SIGNAL_PARENT.role:
        _require(
            decision.get("diagnostic_conclusion")
            == "frozen_model_does_not_beat_zero"
            and decision.get("conditional_mean_identically_zero_proven") == 0
            and decision.get("population_signal_absence_proven") == 0,
            "zero-signal diagnostic conclusion changed",
        )
    elif spec.role == BAYES_POWER_PARENT.role:
        _require(
            decision.get("fresh_physical_witness_planning_authorized") == 1,
            "Bayes calibration no longer authorizes witness planning",
        )

    return {
        "role": spec.role,
        "run_dir": str(root),
        "basename": root.name,
        "registry": {
            "schema": spec.registry_schema,
            "record_count": spec.registry_record_count,
            "sha256": spec.registry_sha256,
            "file_sha256": spec.registry_file_sha256,
        },
        "source_fingerprint": spec.source_fingerprint,
        "scientific_config_sha256": spec.scientific_config_sha256,
        "terminal": {
            "state": spec.terminal_state,
            "stage": spec.terminal_stage,
            "status_decision": spec.terminal_status_decision,
            "decision": spec.decision,
        },
        "physical_training_performed": spec.physical_training_performed,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
        "production_refinement_performed": 0,
        "gates": gates,
        "verified": 1,
    }


def _verify_transitive_physical_binding(
    descendant_root: Path,
    *,
    role: str,
    physical_binding: Mapping[str, Any],
) -> None:
    provenance = _load_json(
        descendant_root / "parent_provenance.json",
        f"{role} transitive physical binding",
    )
    registry = physical_binding["registry"]
    basename = physical_binding["basename"]
    source = physical_binding["source_fingerprint"]
    config = physical_binding["scientific_config_sha256"]
    basename_value = provenance.get(
        "parent_basename", provenance.get("parent_run_basename")
    )
    registry_count = provenance.get(
        "registry_record_count", provenance.get("parent_registry_record_count")
    )
    registry_sha = provenance.get(
        "registry_sha256", provenance.get("parent_registry_semantic_sha256")
    )
    registry_file_sha = provenance.get(
        "registry_file_sha256", provenance.get("parent_registry_file_sha256")
    )
    source_value = provenance.get(
        "source_fingerprint", provenance.get("parent_source_fingerprint")
    )
    config_value = provenance.get(
        "scientific_config_sha256",
        provenance.get("parent_scientific_config_sha256"),
    )
    _require(basename_value == basename, f"{role} physical basename binding changed")
    _require(
        registry_count == registry["record_count"]
        and registry_sha == registry["sha256"]
        and registry_file_sha == registry["file_sha256"],
        f"{role} physical registry binding changed",
    )
    _require(source_value == source, f"{role} physical source binding changed")
    _require(config_value == config, f"{role} physical config binding changed")


def verify_physical_coarse_signal_parents(
    *,
    physical_run_dir: str | Path,
    zero_signal_run_dir: str | Path,
    bayes_power_run_dir: str | Path,
) -> dict[str, Any]:
    """Verify and return immutable bindings for all three witness parents."""

    physical = _verify_parent(physical_run_dir, PARENT_SPECS["physical_one_image"])
    zero_signal = _verify_parent(
        zero_signal_run_dir, PARENT_SPECS["zero_signal_diagnostic"]
    )
    bayes_power = _verify_parent(
        bayes_power_run_dir, PARENT_SPECS["bayes_power_calibration"]
    )
    _verify_transitive_physical_binding(
        Path(zero_signal["run_dir"]),
        role="zero_signal_diagnostic",
        physical_binding=physical,
    )
    _verify_transitive_physical_binding(
        Path(bayes_power["run_dir"]),
        role="bayes_power_calibration",
        physical_binding=physical,
    )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": 1,
        "parents": {
            "physical_one_image": physical,
            "zero_signal_diagnostic": zero_signal,
            "bayes_power_calibration": bayes_power,
        },
        "physical_parent_shared_by_descendants": 1,
        "physical_training_performed": 0,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
        "production_refinement_performed": 0,
        "fresh_physical_evidence_used": 0,
    }


verify_physical_coarse_signal_parent_runs = verify_physical_coarse_signal_parents


__all__ = [
    "BAYES_POWER_PARENT",
    "GateSpec",
    "PARENT_SPECS",
    "PHYSICAL_PARENT",
    "ParentSpec",
    "SCHEMA",
    "SCHEMA_VERSION",
    "ZERO_SIGNAL_PARENT",
    "verify_physical_coarse_signal_parent_runs",
    "verify_physical_coarse_signal_parents",
]
