"""Immutable parent binding for exact Jacobi/RB one-image learnability.

The new experiment intentionally starts from three different pieces of
historical evidence:

* the successful exact multipath kernel/target run;
* the failed, numerically healthy Strang power run that owns the frozen source
  image;
* the terminal Haar recovery run whose failure was power-only.

This module verifies those artifacts without importing a trainer or sampler.
The historical source fingerprints are bound as recorded facts.  The current
workflow records its own source fingerprint separately, which is important
because the learnability patch adds an optional, hash-neutral capture mode to a
source file that was part of the old multipath source set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError, file_fingerprint


SCHEMA = "experiment12-d0-jacobi-rb-one-image-learnability-provenance"
SCHEMA_VERSION = 1

MULTIPATH_RUN_BASENAME = "20260723-092105_production-multipath-jacobi-rb"
MULTIPATH_RUN_SCHEMA = "experiment12-d0-jacobi-rb-cuda-multipath-confirmation"
MULTIPATH_REGISTRY_RECORD_COUNT = 891
MULTIPATH_REGISTRY_SHA256 = (
    "b1724cb1222baf315b3aff24858ac6d979a2ed36e0331995245220a5861545f5"
)
MULTIPATH_SOURCE_FINGERPRINT = (
    "151eaa6c3fbd3a4beaae61ad5337892187e4338fe629761716f281bb84f7d450"
)
MULTIPATH_SCIENTIFIC_CONFIG_SHA256 = (
    "13de55906b4a9b8696183f16f49e07d2b177d20b8429e6b086b3ce5c36bd1ee9"
)
MULTIPATH_DECISION = "exact_jacobi_rb_multipath_kernel_and_target_feasible"

STRANG_RUN_BASENAME = (
    "20260723-230629_production-state-dependent-strang-refinement"
)
STRANG_RUN_SCHEMA = "experiment12-d0-jacobi-rb-strang-refinement"
STRANG_REGISTRY_RECORD_COUNT = 1308
STRANG_REGISTRY_SHA256 = (
    "734c93e1e7d0be29041e1d567b36cbd8ea7aac50df7996d5f8c41fbddef8e632"
)
STRANG_SOURCE_FINGERPRINT = (
    "2f20297eb83b434aa782676119915e9f8883eb116cec0d2b08c2c8c9a8b5ddb0"
)
STRANG_SCIENTIFIC_CONFIG_SHA256 = (
    "884c181610426e8d0c2adb99fc2835aa98322116d7bfdf2c684fcb7a3c286396"
)
STRANG_DECISION = "refinement_power_infeasible"

HAAR_RUN_BASENAME = "20260726-085126_production-haar-power-recovery"
HAAR_RUN_SCHEMA = "experiment12-d0-jacobi-rb-haar-power-recovery-confirmation"
HAAR_REGISTRY_RECORD_COUNT = 511
HAAR_REGISTRY_SHA256 = (
    "8281cc9254fd91e824baea9f0a0e19386a045a21aee5ba377295dfcb734acfde"
)
HAAR_SOURCE_FINGERPRINT = (
    "ff0b17a9efb5c321fe3c9cc23e8ac5382f3c51f742d1b329fff97fdcd29685ec"
)
HAAR_SCIENTIFIC_CONFIG_SHA256 = (
    "075c40eb09b8b811ae394788449997eebe31a08392c68dbd3f949a4a69d62a64"
)
HAAR_DECISION = "hierarchical_power_infeasible"

SOURCE_IMAGE_SHA256 = (
    "0bb39fec59853f789fe366251cd85ed79ffbffb5a1aaa32084d2dbd2bbb4ea7d"
)
MIXED_TARGET_SHA256 = (
    "00ae86fb69be6d86557f15f6f8fa00f8bb3c2514f331863c9638e36d23d135c5"
)
SOURCE_IMAGE_NPZ_SHA256 = (
    "81904cde32495eb11b73cb688cc458118eb2e5578513426d2f9b881ac4665914"
)
FUTURE_MODEL_INPUT_CONTRACT_FILE_SHA256 = (
    "cc75cadf63795011c36c114c77cb1e6e1e1c2997318fb4d59dd3319949106a0d"
)
FUTURE_MODEL_INPUT_CONTRACT_SEMANTIC_SHA256 = (
    "e40c640938204366883af214faf5a4b8deb50aa909f4535b3ed5048f88c9e4f0"
)
ALLOWED_MODEL_INPUTS = (
    "later_full_state",
    "reverse_time",
    "phase",
    "color",
    "duration",
    "label",
)

NO_WORK_FIELDS = (
    "physical_training_performed",
    "sampling_performed",
    "reverse_sampling_performed",
)


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactCompatibilityError(message)


def _one(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 1


def _zero(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 0


def _assert_no_work(record: Mapping[str, Any], description: str) -> None:
    for name in NO_WORK_FIELDS:
        _require(
            _zero(record.get(name, 0)),
            f"{description} records forbidden work in {name}",
        )


def _safe_artifact_path(root: Path, relative: str) -> Path:
    value = Path(relative)
    _require(
        not value.is_absolute() and ".." not in value.parts,
        f"unsafe artifact registry path: {relative}",
    )
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:  # pragma: no cover - belt-and-suspenders on Windows
        raise ArtifactCompatibilityError(
            f"artifact registry path escapes its run: {relative}"
        ) from exc
    return resolved


def _verify_terminal_registry(
    root: Path,
    *,
    expected_schema: str,
    expected_count: int,
    expected_sha256: str,
    status: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a complete terminal registry, including every artifact byte."""

    path = root / "artifact_registry.json"
    _require(path.is_file(), f"parent lacks artifact registry: {root}")
    digest = file_fingerprint(path)
    _require(digest == expected_sha256, "parent artifact registry SHA-256 changed")
    registry = _load(path, "parent registry")
    _require(
        registry.get("schema") == expected_schema + "-artifact-registry"
        and registry.get("schema_version") == 1,
        "parent artifact registry schema changed",
    )
    raw_records = registry.get("records")
    _require(isinstance(raw_records, Mapping), "parent registry records are invalid")
    records = dict(raw_records)
    _require(
        len(records) == expected_count,
        f"parent registry must contain exactly {expected_count} records",
    )
    exclusions = set(
        registry.get("terminal_files_excluded_to_avoid_self_reference", ())
    )
    _require(
        exclusions == {"artifact_registry.json", "run_status.json"},
        "parent terminal registry exclusions changed",
    )
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in exclusions
    }
    _require(actual == set(records), "parent terminal artifact file set changed")
    for relative, raw in records.items():
        _require(isinstance(raw, Mapping), f"invalid registry row: {relative}")
        artifact = _safe_artifact_path(root, str(relative))
        _require(artifact.is_file(), f"registered artifact is missing: {relative}")
        _require(
            raw.get("size") == artifact.stat().st_size,
            f"registered artifact size changed: {relative}",
        )
        _require(
            raw.get("sha256") == file_fingerprint(artifact),
            f"registered artifact SHA-256 changed: {relative}",
        )
    _require(
        status.get("artifact_registry_record_count") == expected_count
        and status.get("artifact_registry_sha256") == expected_sha256
        and status.get("artifact_registry_size") == path.stat().st_size,
        "parent terminal status does not bind its registry",
    )
    _assert_no_work(registry, "parent terminal registry")
    return registry


def _require_files(root: Path, names: Iterable[str]) -> None:
    for name in names:
        _require((root / name).is_file(), f"parent lacks required artifact: {name}")


def _verify_manifest_binding(
    manifest: Mapping[str, Any],
    *,
    run_schema: str,
    source_fingerprint: str,
    scientific_config_sha256: str,
) -> None:
    _require(
        manifest.get("schema") == run_schema
        and manifest.get("schema_version") == 1
        and manifest.get("source_fingerprint") == source_fingerprint
        and manifest.get("scientific_config_sha256")
        == scientific_config_sha256,
        "historical manifest source/config binding changed",
    )
    raw_paths = manifest.get("source_paths")
    _require(
        isinstance(raw_paths, list)
        and raw_paths
        and all(isinstance(value, str) and value for value in raw_paths),
        "historical manifest source path set is invalid",
    )
    _assert_no_work(manifest, "parent manifest")


def verify_successful_multipath_parent(run_dir: str | Path) -> dict[str, Any]:
    """Verify the immutable successful exact kernel/target parent."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"multipath parent does not exist: {root}")
    _require(
        root.name == MULTIPATH_RUN_BASENAME,
        f"multipath parent basename must be {MULTIPATH_RUN_BASENAME}",
    )
    _require_files(
        root,
        (
            "run_manifest.json",
            "run_status.json",
            "scientific_config.json",
            "multipath_decision.json",
            "multipath_preflight_gate.json",
            "multipath_pilot_gate.json",
            "multipath_kernel_gate.json",
            "multipath_target_gate.json",
            "multipath_workflow_gate.json",
            "parent_provenance.json",
        ),
    )
    status = _load(root / "run_status.json", "multipath status")
    registry = _verify_terminal_registry(
        root,
        expected_schema=MULTIPATH_RUN_SCHEMA,
        expected_count=MULTIPATH_REGISTRY_RECORD_COUNT,
        expected_sha256=MULTIPATH_REGISTRY_SHA256,
        status=status,
    )
    manifest = _load(root / "run_manifest.json", "multipath manifest")
    _verify_manifest_binding(
        manifest,
        run_schema=MULTIPATH_RUN_SCHEMA,
        source_fingerprint=MULTIPATH_SOURCE_FINGERPRINT,
        scientific_config_sha256=MULTIPATH_SCIENTIFIC_CONFIG_SHA256,
    )
    decision = _load(root / "multipath_decision.json", "multipath decision")
    kernel = _load(root / "multipath_kernel_gate.json", "multipath kernel gate")
    target = _load(root / "multipath_target_gate.json", "multipath target gate")
    transitive = _load(root / "parent_provenance.json", "multipath provenance")
    _require(
        status.get("schema") == MULTIPATH_RUN_SCHEMA
        and status.get("status") == "complete"
        and status.get("outcome") == "complete"
        and status.get("phase") == "target"
        and _one(status.get("required_gate_pass"))
        and status.get("decision") == MULTIPATH_DECISION,
        "multipath terminal status changed",
    )
    _require(
        decision.get("decision") == MULTIPATH_DECISION
        and _one(decision.get("closed_terminal_scientific_outcome"))
        and _zero(decision.get("physical_training_authorized"))
        and _zero(decision.get("sampling_authorized")),
        "multipath decision scope changed",
    )
    _require(
        kernel.get("evaluation_status") == "evaluated"
        and _one(kernel.get("passed"))
        and _one(kernel.get("numerically_valid"))
        and _one(kernel.get("resource_valid")),
        "multipath kernel gate is not a complete numerical/resource pass",
    )
    _require(
        target.get("evaluation_status") == "evaluated"
        and _one(target.get("passed")),
        "multipath Rao--Blackwell target gate is not a pass",
    )
    _require(
        _one(transitive.get("passed"))
        and transitive.get("parent_artifact_record_count") == 219
        and transitive.get("parent_artifact_registry_sha256")
        == "74a538caa33fbc5ef28e76e7feeedc77287fc0af36b8679c59e241ca3e43a757"
        and _one(transitive.get("parent_certificate_valid"))
        and _one(transitive.get("parent_kernel_numerically_valid")),
        "multipath transitive certified-CUDA provenance changed",
    )
    for description, record in (
        ("multipath status", status),
        ("multipath decision", decision),
        ("multipath kernel", kernel),
        ("multipath target", target),
        ("multipath transitive provenance", transitive),
    ):
        _assert_no_work(record, description)
    return {
        "schema": SCHEMA + "-multipath-parent",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": 1,
        "parent_run_dir": str(root),
        "parent_run_basename": MULTIPATH_RUN_BASENAME,
        "artifact_registry_sha256": MULTIPATH_REGISTRY_SHA256,
        "artifact_registry_record_count": len(dict(registry["records"])),
        "historical_source_fingerprint": MULTIPATH_SOURCE_FINGERPRINT,
        "scientific_config_sha256": MULTIPATH_SCIENTIFIC_CONFIG_SHA256,
        "decision": MULTIPATH_DECISION,
        "kernel_gate_pass": 1,
        "target_gate_pass": 1,
        "historical_source_manifest_binding_pass": 1,
        "transitive_provenance_pass": 1,
        "all_artifact_hashes_pass": 1,
        "parent_mutated": 0,
        **{name: 0 for name in NO_WORK_FIELDS},
    }


def _verify_source_image(root: Path) -> dict[str, Any]:
    metadata_path = root / "source_image.json"
    npz_path = root / "source_image.npz"
    _require(metadata_path.is_file() and npz_path.is_file(), "source image is missing")
    metadata = _load(metadata_path, "Strang source image")
    _require(
        metadata.get("label") == 3
        and metadata.get("class_index") == 0
        and metadata.get("lambda_mix") == 0.35
        and metadata.get("image_sha256") == SOURCE_IMAGE_SHA256
        and metadata.get("mixed_target_sha256") == MIXED_TARGET_SHA256
        and metadata.get("npz_sha256") == SOURCE_IMAGE_NPZ_SHA256,
        "frozen source image metadata changed",
    )
    actual_npz_sha256 = file_fingerprint(npz_path)
    _require(
        actual_npz_sha256 == SOURCE_IMAGE_NPZ_SHA256,
        "frozen source_image.npz SHA-256 changed",
    )
    _require(
        metadata.get("npz_size") == npz_path.stat().st_size,
        "frozen source_image.npz size changed",
    )
    _assert_no_work(metadata, "Strang source image")
    return {
        "label": 3,
        "class_index": 0,
        "lambda_mix": 0.35,
        "image_sha256": SOURCE_IMAGE_SHA256,
        "mixed_target_sha256": MIXED_TARGET_SHA256,
        "source_image_npz_sha256": actual_npz_sha256,
        "source_image_npz_size": npz_path.stat().st_size,
        "source_image_hash_pass": 1,
        "mixed_target_hash_pass": 1,
        "source_image_npz_hash_pass": 1,
    }


def verify_failed_strang_parent(run_dir: str | Path) -> dict[str, Any]:
    """Verify the immutable power-only Strang failure and frozen image."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"Strang parent does not exist: {root}")
    _require(
        root.name == STRANG_RUN_BASENAME,
        f"Strang parent basename must be {STRANG_RUN_BASENAME}",
    )
    _require_files(
        root,
        (
            "run_manifest.json",
            "run_status.json",
            "scientific_config.json",
            "strang_refinement_decision.json",
            "strang_preflight_gate.json",
            "strang_power_gate.json",
            "strang_workflow_gate.json",
            "selected_refinement_design.json",
            "source_image.json",
            "source_image.npz",
            "parent_provenance.json",
        ),
    )
    status = _load(root / "run_status.json", "Strang status")
    registry = _verify_terminal_registry(
        root,
        expected_schema=STRANG_RUN_SCHEMA,
        expected_count=STRANG_REGISTRY_RECORD_COUNT,
        expected_sha256=STRANG_REGISTRY_SHA256,
        status=status,
    )
    manifest = _load(root / "run_manifest.json", "Strang manifest")
    _verify_manifest_binding(
        manifest,
        run_schema=STRANG_RUN_SCHEMA,
        source_fingerprint=STRANG_SOURCE_FINGERPRINT,
        scientific_config_sha256=STRANG_SCIENTIFIC_CONFIG_SHA256,
    )
    decision = _load(root / "strang_refinement_decision.json", "Strang decision")
    preflight = _load(root / "strang_preflight_gate.json", "Strang preflight")
    power = _load(root / "strang_power_gate.json", "Strang power gate")
    transitive = _load(root / "parent_provenance.json", "Strang provenance")
    _require(
        status.get("schema") == STRANG_RUN_SCHEMA
        and status.get("status") == "complete"
        and status.get("outcome") == "gate_failed"
        and status.get("phase") == "power"
        and _zero(status.get("required_gate_pass"))
        and status.get("decision") == STRANG_DECISION,
        "Strang terminal status changed",
    )
    _require(
        decision.get("decision") == STRANG_DECISION,
        "Strang terminal decision changed",
    )
    _require(
        preflight.get("evaluation_status") == "evaluated"
        and _one(preflight.get("passed"))
        and _one(preflight.get("numerically_valid"))
        and _one(preflight.get("resource_valid")),
        "Strang preflight is not a numerical/resource pass",
    )
    _require(
        power.get("evaluation_status") == "evaluated"
        and _zero(power.get("passed"))
        and _one(power.get("numerically_valid"))
        and _zero(power.get("power_valid")),
        "Strang failure is not the frozen power-only failure",
    )
    _require(
        _one(transitive.get("passed"))
        and transitive.get("parent_artifact_record_count")
        == MULTIPATH_REGISTRY_RECORD_COUNT
        and transitive.get("parent_artifact_registry_sha256")
        == MULTIPATH_REGISTRY_SHA256
        and transitive.get("parent_decision") == MULTIPATH_DECISION
        and _one(transitive.get("parent_kernel_pass"))
        and _one(transitive.get("parent_target_pass")),
        "Strang-to-multipath transitive provenance changed",
    )
    _require(
        not (root / "refinement_metrics.json").exists()
        and not (root / "refinement_observables.npz").exists(),
        "Strang parent unexpectedly contains production refinement",
    )
    source = _verify_source_image(root)
    for description, record in (
        ("Strang status", status),
        ("Strang decision", decision),
        ("Strang preflight", preflight),
        ("Strang power", power),
        ("Strang transitive provenance", transitive),
    ):
        _assert_no_work(record, description)
    return {
        "schema": SCHEMA + "-strang-parent",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": 1,
        "parent_run_dir": str(root),
        "parent_run_basename": STRANG_RUN_BASENAME,
        "artifact_registry_sha256": STRANG_REGISTRY_SHA256,
        "artifact_registry_record_count": len(dict(registry["records"])),
        "historical_source_fingerprint": STRANG_SOURCE_FINGERPRINT,
        "scientific_config_sha256": STRANG_SCIENTIFIC_CONFIG_SHA256,
        "decision": STRANG_DECISION,
        "preflight_pass": 1,
        "power_gate_pass": 0,
        "power_only_failure_pass": 1,
        "production_refinement_performed": 0,
        "historical_source_manifest_binding_pass": 1,
        "transitive_provenance_pass": 1,
        "all_artifact_hashes_pass": 1,
        "parent_mutated": 0,
        **source,
        **{name: 0 for name in NO_WORK_FIELDS},
    }


def _verify_future_model_contract(root: Path) -> dict[str, Any]:
    path = root / "future_model_input_contract.json"
    _require(path.is_file(), "Haar parent lacks future model input contract")
    _require(
        file_fingerprint(path) == FUTURE_MODEL_INPUT_CONTRACT_FILE_SHA256,
        "future model input contract file SHA-256 changed",
    )
    contract = _load(path, "future model input contract")
    _require(
        contract.get("schema")
        == (
            "experiment12-d0-jacobi-rb-hierarchical-coupling-confirmation-"
            "future-model-input-contract"
        )
        and contract.get("schema_version") == 1
        and tuple(contract.get("allowed_inputs", ())) == ALLOWED_MODEL_INPUTS
        and contract.get("model_input_contract_sha256")
        == FUTURE_MODEL_INPUT_CONTRACT_SEMANTIC_SHA256
        and _one(contract.get("later_state_only_contract_pass"))
        and _one(contract.get("earlier_state_forbidden"))
        and _one(contract.get("randomness_forbidden"))
        and _one(contract.get("certificate_forbidden"))
        and _one(contract.get("oracle_quantity_forbidden")),
        "future model input semantic contract changed",
    )
    _assert_no_work(contract, "future model input contract")
    return {
        "future_model_input_contract_file_sha256": (
            FUTURE_MODEL_INPUT_CONTRACT_FILE_SHA256
        ),
        "future_model_input_contract_semantic_sha256": (
            FUTURE_MODEL_INPUT_CONTRACT_SEMANTIC_SHA256
        ),
        "allowed_model_inputs": list(ALLOWED_MODEL_INPUTS),
        "future_model_input_contract_pass": 1,
    }


def verify_power_only_haar_parent(run_dir: str | Path) -> dict[str, Any]:
    """Verify the terminal, numerically healthy Haar power-only failure."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"Haar parent does not exist: {root}")
    _require(
        root.name == HAAR_RUN_BASENAME,
        f"Haar parent basename must be {HAAR_RUN_BASENAME}",
    )
    _require_files(
        root,
        (
            "run_manifest.json",
            "run_status.json",
            "scientific_config.json",
            "haar_power_recovery_decision.json",
            "haar_power_recovery_preflight_gate.json",
            "haar_power_recovery_replay_gate.json",
            "haar_power_recovery_pilot_gate.json",
            "haar_power_recovery_workflow_gate.json",
            "future_model_input_contract.json",
            "parent_provenance.json",
        ),
    )
    status = _load(root / "run_status.json", "Haar status")
    registry = _verify_terminal_registry(
        root,
        expected_schema=HAAR_RUN_SCHEMA,
        expected_count=HAAR_REGISTRY_RECORD_COUNT,
        expected_sha256=HAAR_REGISTRY_SHA256,
        status=status,
    )
    manifest = _load(root / "run_manifest.json", "Haar manifest")
    _verify_manifest_binding(
        manifest,
        run_schema=HAAR_RUN_SCHEMA,
        source_fingerprint=HAAR_SOURCE_FINGERPRINT,
        scientific_config_sha256=HAAR_SCIENTIFIC_CONFIG_SHA256,
    )
    decision = _load(root / "haar_power_recovery_decision.json", "Haar decision")
    preflight = _load(
        root / "haar_power_recovery_preflight_gate.json", "Haar preflight"
    )
    replay = _load(root / "haar_power_recovery_replay_gate.json", "Haar replay")
    pilot = _load(root / "haar_power_recovery_pilot_gate.json", "Haar pilot")
    transitive = _load(root / "parent_provenance.json", "Haar provenance")
    _require(
        status.get("schema") == HAAR_RUN_SCHEMA
        and status.get("status") == "complete"
        and status.get("outcome") == "gate_failed"
        and status.get("phase") == "pilot"
        and _zero(status.get("required_gate_pass"))
        and status.get("decision") == HAAR_DECISION,
        "Haar terminal status changed",
    )
    _require(
        decision.get("decision") == HAAR_DECISION
        and _zero(decision.get("production_refinement_patch_authorized"))
        and _zero(decision.get("physical_training_authorized"))
        and _zero(decision.get("sampling_authorized")),
        "Haar terminal decision scope changed",
    )
    _require(
        preflight.get("evaluation_status") == "evaluated"
        and _one(preflight.get("passed"))
        and replay.get("evaluation_status") == "evaluated"
        and _one(replay.get("passed")),
        "Haar preflight/replay evidence is incomplete",
    )
    _require(
        pilot.get("evaluation_status") == "evaluated"
        and _zero(pilot.get("passed"))
        and _one(pilot.get("numerically_valid"))
        and _one(pilot.get("resource_valid"))
        and _zero(pilot.get("power_valid"))
        and _zero(pilot.get("panel_a_nominated"))
        and _zero(pilot.get("panel_b_opened")),
        "Haar failure is not a sealed numerical/resource-healthy power failure",
    )
    _require(
        _one(transitive.get("passed"))
        and _one(transitive.get("parent_transitive_provenance_pass"))
        and _one(transitive.get("parent_all_artifact_hashes_pass"))
        and _one(transitive.get("parent_no_work_pass"))
        and _zero(transitive.get("parent_mutated"))
        and transitive.get("parent_artifact_record_count") == 197
        and transitive.get("parent_artifact_registry_sha256")
        == "4bf1dab4c0905533fe0df885521fb3309ed6344e13f1fd67faad7fa9ae11abfe",
        "Haar transitive provenance changed",
    )
    _require(
        not (root / "pairwise_haar_antithetic_panel_b_evidence.json").exists()
        and not (root / "production_refinement_metrics.json").exists(),
        "Haar parent unexpectedly opened panel B or ran production refinement",
    )
    contract = _verify_future_model_contract(root)
    for description, record in (
        ("Haar status", status),
        ("Haar decision", decision),
        ("Haar preflight", preflight),
        ("Haar replay", replay),
        ("Haar pilot", pilot),
        ("Haar transitive provenance", transitive),
    ):
        _assert_no_work(record, description)
    _require(
        _zero(status.get("production_refinement_performed", 0))
        and _zero(decision.get("production_refinement_performed", 0)),
        "Haar parent records production refinement",
    )
    return {
        "schema": SCHEMA + "-haar-parent",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": 1,
        "parent_run_dir": str(root),
        "parent_run_basename": HAAR_RUN_BASENAME,
        "artifact_registry_sha256": HAAR_REGISTRY_SHA256,
        "artifact_registry_record_count": len(dict(registry["records"])),
        "historical_source_fingerprint": HAAR_SOURCE_FINGERPRINT,
        "scientific_config_sha256": HAAR_SCIENTIFIC_CONFIG_SHA256,
        "decision": HAAR_DECISION,
        "preflight_pass": 1,
        "replay_pass": 1,
        "pilot_numerically_valid": 1,
        "pilot_resource_valid": 1,
        "pilot_power_valid": 0,
        "power_only_failure_pass": 1,
        "panel_b_absent_pass": 1,
        "production_refinement_performed": 0,
        "historical_source_manifest_binding_pass": 1,
        "transitive_provenance_pass": 1,
        "all_artifact_hashes_pass": 1,
        "parent_mutated": 0,
        **contract,
        **{name: 0 for name in NO_WORK_FIELDS},
    }


def verify_learnability_parents(
    *,
    parent_multipath_run_dir: str | Path | None = None,
    parent_strang_run_dir: str | Path | None = None,
    parent_haar_run_dir: str | Path | None = None,
    multipath_run_dir: str | Path | None = None,
    strang_run_dir: str | Path | None = None,
    haar_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Verify and combine all immutable evidence required by preflight."""

    def resolve(
        primary: str | Path | None,
        compatibility: str | Path | None,
        name: str,
    ) -> str | Path:
        if primary is not None and compatibility is not None:
            _require(
                Path(primary).resolve() == Path(compatibility).resolve(),
                f"conflicting {name} parent paths",
            )
        value = primary if primary is not None else compatibility
        _require(value is not None, f"missing {name} parent path")
        return value  # type: ignore[return-value]

    multipath_path = resolve(
        parent_multipath_run_dir, multipath_run_dir, "multipath"
    )
    strang_path = resolve(parent_strang_run_dir, strang_run_dir, "Strang")
    haar_path = resolve(parent_haar_run_dir, haar_run_dir, "Haar")
    multipath = verify_successful_multipath_parent(multipath_path)
    strang = verify_failed_strang_parent(strang_path)
    haar = verify_power_only_haar_parent(haar_path)
    _require(
        _one(multipath.get("passed"))
        and _one(strang.get("passed"))
        and _one(haar.get("passed")),
        "one or more learnability parents failed verification",
    )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": 1,
        "claim_scope": (
            "immutable evidence for a one-image learnability study on the "
            "exact K=512 split chain; the failed refinement claim is preserved"
        ),
        "multipath": multipath,
        "strang": strang,
        "haar": haar,
        "multipath_kernel_gate_pass": 1,
        "multipath_target_gate_pass": 1,
        "multipath_decision_pass": 1,
        "strang_power_failure_preserved_pass": 1,
        "haar_power_only_failure_pass": 1,
        "haar_numerical_health_pass": 1,
        "haar_resource_health_pass": 1,
        "source_image_hash_pass": 1,
        "source_image_npz_hash_pass": 1,
        "mixed_target_hash_pass": 1,
        "future_model_input_contract_pass": 1,
        "parent_registries_pass": 1,
        "source_binding_pass": 1,
        "parents_no_training_pass": 1,
        "parents_no_reverse_sampling_pass": 1,
        "state_dependent_strang_refinement_established": 0,
        "unsplit_generator_approximation_authorized": 0,
        "spatial_dirichlet_ferguson_claim_authorized": 0,
        "reverse_sampling_authorized": 0,
        "sampling_authorized": 0,
        "reconstruction_claim_authorized": 0,
        "known_prior_claim_authorized": 0,
        "full_dataset_training_authorized": 0,
        "larger_exact_discrete_chain_training_planning_authorized": 0,
        "parent_mutated": 0,
        **{name: 0 for name in NO_WORK_FIELDS},
    }


# Explicit alias matching the CLI/workflow name used in the patch plan.
verify_one_image_learnability_parents = verify_learnability_parents


__all__ = [
    "ALLOWED_MODEL_INPUTS",
    "FUTURE_MODEL_INPUT_CONTRACT_FILE_SHA256",
    "FUTURE_MODEL_INPUT_CONTRACT_SEMANTIC_SHA256",
    "HAAR_DECISION",
    "HAAR_REGISTRY_RECORD_COUNT",
    "HAAR_REGISTRY_SHA256",
    "HAAR_RUN_BASENAME",
    "HAAR_SCIENTIFIC_CONFIG_SHA256",
    "HAAR_SOURCE_FINGERPRINT",
    "MIXED_TARGET_SHA256",
    "MULTIPATH_DECISION",
    "MULTIPATH_REGISTRY_RECORD_COUNT",
    "MULTIPATH_REGISTRY_SHA256",
    "MULTIPATH_RUN_BASENAME",
    "MULTIPATH_SCIENTIFIC_CONFIG_SHA256",
    "MULTIPATH_SOURCE_FINGERPRINT",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SOURCE_IMAGE_NPZ_SHA256",
    "SOURCE_IMAGE_SHA256",
    "STRANG_DECISION",
    "STRANG_REGISTRY_RECORD_COUNT",
    "STRANG_REGISTRY_SHA256",
    "STRANG_RUN_BASENAME",
    "STRANG_SCIENTIFIC_CONFIG_SHA256",
    "STRANG_SOURCE_FINGERPRINT",
    "verify_failed_strang_parent",
    "verify_learnability_parents",
    "verify_one_image_learnability_parents",
    "verify_power_only_haar_parent",
    "verify_successful_multipath_parent",
]
