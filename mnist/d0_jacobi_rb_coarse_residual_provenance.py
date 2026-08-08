"""Immutable parents for the exact-K512 coarse-baseline residual study.

The proposed learner is a fresh experiment.  Its only scientific parents are
the terminal physical coarse-signal witness and the earlier, sealed one-image
learner.  This module verifies those runs byte-for-byte without importing a
trainer, a transition scheduler, or a reverse sampler.

The witness establishes a non-zero *coarse* conditional-mean projection.  The
failed learner establishes only that its frozen selected neural predictor did
not beat analytic zero.  Neither parent authorizes sampling, reconstruction,
an unsplit-generator claim, or a spatial Dirichlet--Ferguson claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)


SCHEMA = "experiment12-d0-jacobi-rb-coarse-residual-provenance"
SCHEMA_VERSION = 1

SOURCE_IMAGE_SHA256 = (
    "0bb39fec59853f789fe366251cd85ed79ffbffb5a1aaa32084d2dbd2bbb4ea7d"
)
MIXED_TARGET_SHA256 = (
    "00ae86fb69be6d86557f15f6f8fa00f8bb3c2514f331863c9638e36d23d135c5"
)
SELECTED_OUTER_STEPS = tuple(15 + 16 * index for index in range(32))


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
    decision_path: str
    decision_schema: str
    decision: str
    physical_training_performed: int
    gates: tuple[GateSpec, ...]


WITNESS_PARENT = ParentSpec(
    role="physical_coarse_signal_witness",
    basename=(
        "20260730-135059_production-exact-k512-physical-coarse-signal-jsonfix"
    ),
    run_schema="experiment12-d0-jacobi-rb-physical-coarse-signal-witness",
    config_schema=(
        "experiment12-d0-jacobi-rb-physical-coarse-signal-witness-"
        "scientific-config"
    ),
    status_schema=(
        "experiment12-d0-jacobi-rb-physical-coarse-signal-witness-status"
    ),
    registry_schema=(
        "experiment12-d0-jacobi-rb-physical-coarse-signal-witness-"
        "artifact-registry"
    ),
    registry_record_count=2_616,
    registry_sha256=(
        "ca405ea7c00d6efed470c0651b8ad28d31f797cf142a1bef5f75d464ee1c3ab3"
    ),
    registry_file_sha256=(
        "866984822ef14dbb13f0644a0f23697f6fb42ecec40b07341249d74043319747"
    ),
    source_fingerprint=(
        "31f1f15008c2db864e282c5d3fa047986a9b576b92c480d50a18d55138e9eafb"
    ),
    scientific_config_sha256=(
        "b2e28989ef6da6fa2d233b14ee475c04e10326079cf03750f1f427494de90f14"
    ),
    terminal_state="completed",
    terminal_stage="analyze",
    decision_path="physical_coarse_signal_decision.json",
    decision_schema="d0-jacobi-rb-physical-coarse-signal-decision-v1",
    decision="exact_physical_coarse_signal_detected",
    physical_training_performed=0,
    gates=(
        GateSpec(
            "coarse_signal_preflight_gate.json",
            "d0-jacobi-rb-physical-coarse-signal-gate-v1",
            1,
        ),
        GateSpec(
            "coarse_signal_panel_a_gate.json",
            "d0-jacobi-rb-physical-coarse-signal-gate-v1",
            1,
        ),
        GateSpec(
            "coarse_signal_panel_b_gate.json",
            "d0-jacobi-rb-physical-coarse-signal-gate-v1",
            1,
        ),
        GateSpec(
            "coarse_signal_witness_gate.json",
            "d0-jacobi-rb-physical-coarse-signal-gate-v1",
            1,
        ),
    ),
)

FAILED_LEARNER_PARENT = ParentSpec(
    role="failed_one_image_learner",
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
            "confirmation_cache_gate.json",
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

PARENT_SPECS: dict[str, ParentSpec] = {
    WITNESS_PARENT.role: WITNESS_PARENT,
    FAILED_LEARNER_PARENT.role: FAILED_LEARNER_PARENT,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactCompatibilityError(message)


def _load(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"invalid {description}: {path}") from exc
    _require(isinstance(value, dict), f"{description} must be a JSON object")
    return value


def _safe_path(root: Path, raw: Any) -> Path:
    _require(isinstance(raw, str) and raw != "", "registry path is invalid")
    relative = PurePosixPath(raw)
    _require(
        not relative.is_absolute()
        and all(part not in {"", ".", ".."} for part in relative.parts),
        f"unsafe registry path: {raw!r}",
    )
    target = (root / Path(*relative.parts)).resolve()
    _require(root in target.parents, f"registry path escapes run: {raw!r}")
    return target


def _verify_registry(
    root: Path, spec: ParentSpec
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = root / "artifact_registry.json"
    _require(
        file_fingerprint(path) == spec.registry_file_sha256,
        f"{spec.role} registry file hash changed",
    )
    registry = _load(path, f"{spec.role} artifact registry")
    records = registry.get("records")
    _require(
        registry.get("schema") == spec.registry_schema
        and registry.get("schema_version") == 1
        and registry.get("record_count") == spec.registry_record_count
        and isinstance(records, list)
        and len(records) == spec.registry_record_count
        and registry.get("registry_sha256") == spec.registry_sha256
        and config_fingerprint(records) == spec.registry_sha256,
        f"{spec.role} terminal registry binding changed",
    )
    by_path: dict[str, dict[str, Any]] = {}
    for raw_record in records:
        _require(isinstance(raw_record, Mapping), "registry row is malformed")
        record = dict(raw_record)
        raw = record.get("path")
        _require(
            isinstance(raw, str) and raw not in by_path,
            f"{spec.role} registry path is duplicated",
        )
        target = _safe_path(root, raw)
        _require(
            target.is_file()
            and target.stat().st_size == record.get("size")
            and file_fingerprint(target) == record.get("sha256"),
            f"{spec.role} registered artifact changed: {raw}",
        )
        by_path[raw] = record
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
        and item.name not in {"artifact_registry.json", "run_status.json"}
    }
    _require(
        actual == set(by_path),
        f"{spec.role} terminal registry file set changed",
    )
    return registry, by_path


def _verify_live_sources(manifest: Mapping[str, Any], spec: ParentSpec) -> None:
    raw = manifest.get("source_paths")
    _require(
        isinstance(raw, list)
        and raw
        and all(isinstance(item, str) and item for item in raw),
        f"{spec.role} source path list changed",
    )
    repository_root = Path(__file__).resolve().parents[1]
    stored = [Path(item) for item in raw]
    resolved = [
        item if item.is_absolute() else (repository_root / item).resolve()
        for item in stored
    ]
    _require(
        len({item.as_posix() for item in resolved}) == len(resolved)
        and all(item.is_file() for item in resolved),
        f"{spec.role} source files are missing or duplicated",
    )
    _require(
        source_fingerprint(resolved) == spec.source_fingerprint,
        f"{spec.role} live source fingerprint changed",
    )


def _no_sampling(record: Mapping[str, Any], description: str) -> None:
    for field in (
        "sampling_performed",
        "reverse_sampling_performed",
        "production_refinement_performed",
    ):
        _require(
            int(record.get(field, 0)) == 0,
            f"{description} unexpectedly records {field}",
        )
    for field in (
        "sampling_authorized",
        "reverse_sampling_authorized",
        "reconstruction_claim_authorized",
        "full_dataset_training_authorized",
        "production_refinement_authorized",
    ):
        if field in record:
            _require(
                int(record.get(field, 0)) == 0,
                f"{description} unexpectedly authorizes {field}",
            )


def _verify_parent(root_value: str | Path, spec: ParentSpec) -> dict[str, Any]:
    root = Path(root_value).resolve()
    _require(root.is_dir(), f"{spec.role} parent does not exist: {root}")
    _require(root.name == spec.basename, f"wrong {spec.role} parent basename")
    _, registry = _verify_registry(root, spec)
    manifest = _load(root / "run_manifest.json", f"{spec.role} manifest")
    status = _load(root / "run_status.json", f"{spec.role} status")
    config = _load(root / "scientific_config.json", f"{spec.role} config")
    decision = _load(root / spec.decision_path, f"{spec.role} decision")
    _require(
        manifest.get("schema") == spec.run_schema
        and manifest.get("schema_version") == 1
        and manifest.get("source_fingerprint") == spec.source_fingerprint
        and manifest.get("scientific_config_sha256")
        == spec.scientific_config_sha256,
        f"{spec.role} manifest binding changed",
    )
    _verify_live_sources(manifest, spec)
    _require(
        config.get("schema") == spec.config_schema
        and config.get("schema_version") == 1
        and config.get("semantic_sha256") == spec.scientific_config_sha256,
        f"{spec.role} scientific configuration changed",
    )
    registry_path = root / "artifact_registry.json"
    _require(
        status.get("schema") == spec.status_schema
        and status.get("schema_version") == 1
        and status.get("state") == spec.terminal_state
        and status.get("stage") == spec.terminal_stage
        and status.get("decision") == spec.decision
        and status.get("artifact_registry_record_count")
        == spec.registry_record_count
        and status.get("artifact_registry_sha256") == spec.registry_sha256
        and status.get("artifact_registry_file_sha256")
        == spec.registry_file_sha256
        and status.get("artifact_registry_file_size")
        == registry_path.stat().st_size,
        f"{spec.role} terminal status changed",
    )
    _require(
        decision.get("schema") == spec.decision_schema
        and decision.get("schema_version") == 1
        and decision.get("evaluation_status") == "evaluated"
        and decision.get("decision") == spec.decision,
        f"{spec.role} terminal decision changed",
    )
    for description, record in (
        ("manifest", manifest),
        ("status", status),
        ("config", config),
        ("decision", decision),
    ):
        _no_sampling(record, f"{spec.role} {description}")
    _require(
        int(status.get("physical_training_performed", -1))
        == spec.physical_training_performed
        and int(decision.get("physical_training_performed", -1))
        == spec.physical_training_performed,
        f"{spec.role} physical-training scope changed",
    )
    gates: dict[str, dict[str, Any]] = {}
    for gate_spec in spec.gates:
        gate = _load(root / gate_spec.path, f"{spec.role} {gate_spec.path}")
        _require(
            gate.get("schema") == gate_spec.schema
            and gate.get("schema_version") == 1
            and gate.get("evaluation_status") == "evaluated"
            and int(gate.get("passed", -1)) == gate_spec.passed,
            f"{spec.role} gate changed: {gate_spec.path}",
        )
        _no_sampling(gate, f"{spec.role} {gate_spec.path}")
        gates[gate_spec.path] = {
            "schema": gate_spec.schema,
            "passed": gate_spec.passed,
        }
    return {
        "role": spec.role,
        "run_dir": str(root),
        "basename": root.name,
        "registry": {
            "record_count": spec.registry_record_count,
            "sha256": spec.registry_sha256,
            "file_sha256": spec.registry_file_sha256,
        },
        "source_fingerprint": spec.source_fingerprint,
        "scientific_config_sha256": spec.scientific_config_sha256,
        "terminal": {
            "state": spec.terminal_state,
            "stage": spec.terminal_stage,
            "decision": spec.decision,
        },
        "config": config,
        "decision_record": decision,
        "gates": gates,
        "registered_paths": registry,
        "verified": 1,
    }


def _verify_witness_specific(binding: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(binding["run_dir"]))
    analysis = _load(root / "physical_coarse_signal_analysis.json", "witness analysis")
    classification = analysis.get("classification")
    _require(isinstance(classification, Mapping), "witness classification is missing")
    _require(
        classification.get("decision") == WITNESS_PARENT.decision
        and float(classification.get("bootstrap_lower_bound", 0.0)) > 0.0
        and float(classification.get("welch_lower_bound", 0.0)) > 0.0
        and int(analysis.get("lower_bound_on_full_allowed_input_conditional_mean_energy", 0))
        == 1
        and int(analysis.get("conditional_mean_identically_zero_proven", 1)) == 0,
        "witness no longer establishes the frozen positive coarse projection",
    )
    decision = binding["decision_record"]
    _require(
        decision.get("recommended_next_action")
        == (
            "plan a coarse-baseline plus exact-RB residual learner with "
            "unweighted MSE against the unchanged exact label"
        ),
        "witness recommended action changed",
    )
    return {
        "coarse_signal_point_estimate": float(classification.get("point_estimate", analysis["bootstrap"]["point_estimate"])),
        "bootstrap_lower_bound": float(classification["bootstrap_lower_bound"]),
        "welch_lower_bound": float(classification["welch_lower_bound"]),
        "coarse_signal_detected_pass": 1,
    }


def _verify_learner_specific(binding: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(binding["run_dir"]))
    gate = _load(root / "confirmation_gate.json", "learner confirmation gate")
    failed = sorted(
        name
        for name, raw in dict(gate.get("subchecks", {})).items()
        if isinstance(raw, Mapping) and int(raw.get("passed", 0)) != 1
    )
    _require(
        failed == ["aggregate_model_beats_zero"],
        "failed learner no longer has the frozen singleton failure",
    )
    return {
        "failed_confirmation_subchecks": failed,
        "only_aggregate_model_beats_zero_failed_pass": 1,
    }


def _same_scientific_problem(
    witness: Mapping[str, Any], learner: Mapping[str, Any]
) -> dict[str, Any]:
    wc = witness["config"]
    lc = learner["config"]
    wimage = wc.get("source_image")
    limage = lc.get("source_image")
    _require(
        isinstance(wimage, Mapping)
        and isinstance(limage, Mapping)
        and wimage.get("image_sha256") == SOURCE_IMAGE_SHA256
        and limage.get("image_sha256") == SOURCE_IMAGE_SHA256
        and wimage.get("mixed_target_sha256") == MIXED_TARGET_SHA256
        and limage.get("mixed_target_sha256") == MIXED_TARGET_SHA256
        and float(wimage.get("lambda_mix", -1.0)) == 0.35
        and float(limage.get("lambda_mix", -1.0)) == 0.35,
        "parent source-image binding differs",
    )
    for name, expected in (
        ("grid_size", 28),
        ("alpha", 1.0),
        ("outer_steps", 512),
        ("edges_per_phase", 392),
        ("tau_eff", 5.0e-5),
    ):
        if name in wc and name in lc:
            _require(
                wc.get(name) == expected and lc.get(name) == expected,
                f"parent kernel field changed: {name}",
            )
    wsteps = tuple(dict(wc.get("analysis", {})).get("selected_outer_steps", ()))
    lsteps = tuple(lc.get("selected_outer_steps", ()))
    _require(
        wsteps == SELECTED_OUTER_STEPS and lsteps == SELECTED_OUTER_STEPS,
        "parent selected outer steps differ",
    )
    _require(
        tuple(wc.get("phase_matchings", ())) == tuple(lc.get("phase_matchings", ()))
        and tuple(wc.get("phase_durations", ()))
        == tuple(lc.get("phase_durations", ())),
        "parent phase schedule differs",
    )
    return {
        "same_source_image_pass": 1,
        "same_mixed_target_pass": 1,
        "same_exact_k512_kernel_pass": 1,
        "same_selected_outer_steps_pass": 1,
        "same_phase_schedule_pass": 1,
    }


def _verify_transitive_binding(
    witness: Mapping[str, Any], learner: Mapping[str, Any]
) -> None:
    provenance = _load(
        Path(str(witness["run_dir"])) / "parent_provenance.json",
        "witness transitive provenance",
    )
    parents = provenance.get("parents")
    _require(isinstance(parents, Mapping), "witness parent map is missing")
    physical = parents.get("physical_one_image")
    _require(isinstance(physical, Mapping), "witness learner binding is missing")
    registry = physical.get("registry")
    _require(isinstance(registry, Mapping), "witness learner registry is missing")
    _require(
        physical.get("basename") == learner["basename"]
        and physical.get("source_fingerprint") == learner["source_fingerprint"]
        and physical.get("scientific_config_sha256")
        == learner["scientific_config_sha256"]
        and registry.get("record_count")
        == learner["registry"]["record_count"]
        and registry.get("sha256") == learner["registry"]["sha256"]
        and registry.get("file_sha256") == learner["registry"]["file_sha256"]
        and dict(physical.get("terminal", {})).get("decision")
        == FAILED_LEARNER_PARENT.decision
        and int(physical.get("verified", 0)) == 1,
        "witness transitive learner binding changed",
    )


def verify_coarse_residual_parents(
    *,
    witness_run_dir: str | Path | None = None,
    failed_learner_run_dir: str | Path | None = None,
    parent_witness_run_dir: str | Path | None = None,
    parent_one_image_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the terminal positive witness and its failed learner ancestor."""

    witness_value = (
        witness_run_dir if witness_run_dir is not None else parent_witness_run_dir
    )
    learner_value = (
        failed_learner_run_dir
        if failed_learner_run_dir is not None
        else parent_one_image_run_dir
    )
    _require(witness_value is not None, "missing witness parent path")
    _require(learner_value is not None, "missing failed learner parent path")
    witness = _verify_parent(
        witness_value, PARENT_SPECS["physical_coarse_signal_witness"]
    )
    learner = _verify_parent(
        learner_value, PARENT_SPECS["failed_one_image_learner"]
    )
    witness_specific = _verify_witness_specific(witness)
    learner_specific = _verify_learner_specific(learner)
    shared = _same_scientific_problem(witness, learner)
    _verify_transitive_binding(witness, learner)
    for binding in (witness, learner):
        binding.pop("registered_paths", None)
        binding.pop("config", None)
        binding.pop("decision_record", None)
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": 1,
        "claim_scope": (
            "fresh coarse-baseline plus exact-RB residual learnability for one "
            "image under the certified fixed-K512 split chain"
        ),
        "parents": {
            "physical_coarse_signal_witness": witness,
            "failed_one_image_learner": learner,
        },
        **witness_specific,
        **learner_specific,
        **shared,
        "transitive_provenance_pass": 1,
        "all_artifact_hashes_pass": 1,
        "parents_immutable_pass": 1,
        "physical_training_performed": 0,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
        "reconstruction_claim_authorized": 0,
    }


verify_coarse_residual_parent_runs = verify_coarse_residual_parents


__all__ = [
    "FAILED_LEARNER_PARENT",
    "GateSpec",
    "MIXED_TARGET_SHA256",
    "PARENT_SPECS",
    "ParentSpec",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SELECTED_OUTER_STEPS",
    "SOURCE_IMAGE_SHA256",
    "WITNESS_PARENT",
    "verify_coarse_residual_parent_runs",
    "verify_coarse_residual_parents",
]
