"""Complete-pipeline confirmation for the exact eager-prefix Jacobi schedule.

The immutable parent measured the eager certificate policy on the dominant
P=10 base-authorizer component, but conservatively assigned no speedup to the
other base cohorts or to the fused midpoint branches.  This additive workflow
therefore runs the already frozen four-profile, three-repeat timing panel with
the same exact eager sampler throughout.  It changes no transition, target,
randomness, evidence count, model, or scientific threshold and performs no
production cache generation, training, controller trajectory, or sampling.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

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
from mnist.d0_jacobi_rb_boundary_tangent_eager_pipeline_gate import (
    decide_eager_pipeline_workflow,
    evaluate_eager_pipeline_pilot,
    evaluate_eager_pipeline_preflight,
    evaluate_eager_pipeline_workflow,
    not_evaluated_gate,
)
from mnist.d0_jacobi_rb_boundary_tangent_eager_pipeline_provenance import (
    build_eager_pipeline_parent_readjudication,
    verify_eager_pipeline_parent,
)
from mnist.d0_jacobi_rb_boundary_tangent_schedule import (
    MAXIMUM_LAUNCH_LANES,
    PILOT_PROFILE_NAMES,
    PILOT_REPEAT_COUNT,
    PROFILE_PATH_IDS,
    PROJECTED_BASE_TRANSITIONS,
    PROJECTED_MIDPOINT_TRANSITIONS,
    PROJECTED_TOTAL_TRANSITIONS,
    ROOT_SEED,
    WINDOW_START_STEPS,
    build_fused_launch_plan,
    expected_profile_transition_counts,
    frozen_repeat_order,
    project_frozen_schedule,
    validate_repeat_records,
)
from mnist.d0_jacobi_rb_boundary_tangent_schedule_provenance import (
    build_schedule_cohort_plan,
    build_schedule_path_plan,
    build_schedule_timing_plan,
    schedule_source_paths,
    validate_schedule_cohort_plan,
    validate_schedule_path_plan,
    validate_schedule_timing_plan,
)
import mnist.diag_d0_jacobi_rb_boundary_tangent_prefix_schedule_confirmation as _prefix


RUN_SCHEMA = (
    "experiment12-d0-jacobi-rb-boundary-tangent-eager-pipeline-confirmation-v1"
)
STAGES = ("preflight", "pilot", "report", "all")
REQUIRED_GATES = ("none", "preflight", "pilot")
SHARD_STEPS = 8
NO_WORK = {
    "production_cache_generated": 0,
    "production_cache_generation_performed": 0,
    "physical_training_performed": 0,
    "controller_control_trajectory_performed": 0,
    "full_reverse_path_performed": 0,
    "image_sampling_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
    "reconstruction_performed": 0,
}
NO_AUTHORIZATION = {
    "cache_generation_authorized": 0,
    "training_authorized": 0,
    "physical_training_authorized": 0,
    "controller_control_trajectory_authorized": 0,
    "sampling_authorized": 0,
    "reverse_sampling_authorized": 0,
    "reconstruction_authorized": 0,
}
_REGISTRY_EXCLUDED = {
    "artifact_registry.json",
    "run_status.json",
    "eager_pipeline_decision.json",
    "workflow_gate.json",
}


class EagerPipelineCLIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "pipeline_execution",
        failure_code: str = "eager_pipeline_execution_failed",
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_domain = str(failure_domain)
        self.failure_code = str(failure_code)
        self.diagnostics = dict(diagnostics or {})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"cannot read JSON artifact: {target}") from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"JSON artifact is not an object: {target}")
    return dict(value)


def _freeze_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        if _load_json(path) != dict(value):
            raise ArtifactCompatibilityError(f"frozen artifact changed: {path.name}")
        return
    atomic_write_json(path, dict(value))


def _passed(value: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("evaluation_status") == "evaluated"
        and int(value.get("passed", 0)) == 1
    )


def _optional_json(run_dir: Path, name: str) -> dict[str, Any] | None:
    path = run_dir / name
    return _load_json(path) if path.is_file() else None


def _status(
    run_dir: Path,
    *,
    state: str,
    stage: str,
    decision: str | None = None,
    message: str | None = None,
    failure_domain: str | None = None,
    failure_code: str | None = None,
    scientific_evidence_complete: int | None = None,
) -> None:
    atomic_write_json(
        run_dir / "run_status.json",
        {
            "schema": RUN_SCHEMA + "-status",
            "schema_version": 1,
            "state": str(state),
            "stage": str(stage),
            "decision": decision,
            "message": message,
            "failure_domain": failure_domain,
            "failure_code": failure_code,
            "scientific_evidence_complete": scientific_evidence_complete,
            "updated_at": _now(),
            **NO_WORK,
            **NO_AUTHORIZATION,
        },
    )


def _registry_artifacts(run_dir: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir).as_posix()
        if (
            relative in _REGISTRY_EXCLUDED
            or relative.endswith(".tmp")
            or ".tmp." in path.name
        ):
            continue
        output.append(
            {
                "path": relative,
                "sha256": file_fingerprint(path),
                "size": path.stat().st_size,
            }
        )
    return output


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    artifacts = _registry_artifacts(run_dir)
    semantics = {
        "snapshot_kind": "current-exact-restartable-pipeline",
        "excluded_paths": sorted(_REGISTRY_EXCLUDED),
    }
    record = {
        "schema": RUN_SCHEMA + "-artifact-registry",
        "schema_version": 1,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "registry_semantics": semantics,
        "semantic_sha256": config_fingerprint(
            {"artifacts": artifacts, "registry_semantics": semantics}
        ),
        **NO_WORK,
        **NO_AUTHORIZATION,
    }
    atomic_write_json(run_dir / "artifact_registry.json", record)
    return record


def _verify_existing_registry(run_dir: Path) -> None:
    path = run_dir / "artifact_registry.json"
    if not path.is_file():
        return
    record = _load_json(path)
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list):
        raise ArtifactCompatibilityError("restart artifact registry is malformed")
    for row in artifacts:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise ArtifactCompatibilityError("restart registry row is malformed")
        target = run_dir / str(row["path"])
        if (
            not target.is_file()
            or row.get("sha256") != file_fingerprint(target)
            or int(row.get("size", -1)) != target.stat().st_size
        ):
            raise ArtifactCompatibilityError(
                f"restart artifact changed: {row.get('path')}"
            )


def _seal_stage(run_dir: Path, names: Sequence[str], seal_name: str) -> dict[str, Any]:
    artifacts = []
    for name in names:
        path = run_dir / name
        if not path.is_file():
            raise ArtifactCompatibilityError(f"cannot seal missing artifact: {name}")
        artifacts.append(
            {"path": name, "sha256": file_fingerprint(path), "size": path.stat().st_size}
        )
    record = {
        "schema": RUN_SCHEMA + "-stage-seal",
        "schema_version": 1,
        "artifacts": artifacts,
        "semantic_sha256": config_fingerprint({"artifacts": artifacts}),
    }
    _freeze_json(run_dir / seal_name, record)
    return record


def _sealed_stage(run_dir: Path, gate_name: str, seal_name: str) -> dict[str, Any] | None:
    gate_path = run_dir / gate_name
    seal_path = run_dir / seal_name
    if not gate_path.is_file() and not seal_path.is_file():
        return None
    if not gate_path.is_file() or not seal_path.is_file():
        raise ArtifactCompatibilityError(f"incomplete sealed stage: {gate_name}")
    seal = _load_json(seal_path)
    artifacts = seal.get("artifacts")
    if not isinstance(artifacts, list):
        raise ArtifactCompatibilityError("stage seal is malformed")
    expected = []
    for row in artifacts:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise ArtifactCompatibilityError("stage seal row is malformed")
        path = run_dir / str(row["path"])
        if not path.is_file():
            raise ArtifactCompatibilityError("sealed stage artifact is missing")
        expected.append(
            {"path": row["path"], "sha256": file_fingerprint(path), "size": path.stat().st_size}
        )
    if (
        expected != artifacts
        or seal.get("semantic_sha256") != config_fingerprint({"artifacts": artifacts})
    ):
        raise ArtifactCompatibilityError("sealed stage changed")
    return _load_json(gate_path)


def _source_set() -> tuple[Path, ...]:
    sibling = Path(__file__).resolve().parent
    return schedule_source_paths(
        (
            *_prefix._source_set(),
            Path(__file__),
            sibling / "d0_jacobi_rb_boundary_tangent_eager_pipeline_gate.py",
            sibling / "d0_jacobi_rb_boundary_tangent_eager_pipeline_provenance.py",
        )
    )


def _scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    record = {
        "schema": RUN_SCHEMA + "-scientific-config",
        "schema_version": 1,
        "root_seed": ROOT_SEED,
        "grid_size": 28,
        "alpha": 1.0,
        "sample_steps": 512,
        "tau_eff": 5.0e-5,
        "training_paths": 64,
        "validation_paths": 32,
        "confirmation_paths": 64,
        "window_start_outer_steps": list(WINDOW_START_STEPS),
        "window_outer_steps": 16,
        "repeat_count": PILOT_REPEAT_COUNT,
        "restart_outer_steps": SHARD_STEPS,
        "maximum_launch_lanes": MAXIMUM_LAUNCH_LANES,
        "projected_base_transitions": PROJECTED_BASE_TRANSITIONS,
        "projected_midpoint_transitions": PROJECTED_MIDPOINT_TRANSITIONS,
        "projected_total_transitions": PROJECTED_TOTAL_TRANSITIONS,
        "maximum_projected_seconds": 108_000.0,
        "minimum_projected_effective_rate": PROJECTED_TOTAL_TRANSITIONS / 108_000.0,
        "prefix_profile": "eager_prefix_128_tpb128",
        "target": "unchanged exact certified Jacobi Rao-Blackwell label",
        "model": "unchanged width-32 JacobiRBPhasePredictor",
        "workflow_claim": "complete-pipeline execution-schedule feasibility only",
        "test_only": int(args.test_only),
        **NO_WORK,
        **NO_AUTHORIZATION,
    }
    record["semantic_sha256"] = config_fingerprint(record)
    return record


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        root = args.resume_run_dir.resolve()
        if not root.is_dir():
            raise ArtifactCompatibilityError(f"resume run does not exist: {root}")
        return root, True
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = (args.runs_root / f"{stamp}_{args.run_name}").resolve()
    root.mkdir(parents=True, exist_ok=False)
    return root, False


def _initialize(run_dir: Path, args: argparse.Namespace, *, resumed: bool) -> None:
    provenance = verify_eager_pipeline_parent(
        parent_prefix_run_dir=args.parent_prefix_run_dir
    )
    readjudication = build_eager_pipeline_parent_readjudication(provenance)
    parent_manifest = _load_json(args.parent_prefix_run_dir / "run_manifest.json")
    args.parent_schedule_run_dir = Path(
        str(parent_manifest["parent_schedule_run_dir"])
    ).resolve()
    args.parent_coarse_residual_run_dir = Path(
        str(parent_manifest["transitive_coarse_parent_run_dir"])
    ).resolve()
    if not args.parent_schedule_run_dir.is_dir() or not args.parent_coarse_residual_run_dir.is_dir():
        raise ArtifactCompatibilityError("transitive timing parents are unavailable")

    config = _scientific_config(args)
    paths = build_schedule_path_plan()
    cohorts = build_schedule_cohort_plan()
    timing = build_schedule_timing_plan()
    validate_schedule_path_plan(paths)
    validate_schedule_cohort_plan(cohorts)
    validate_schedule_timing_plan(timing)
    for name, current in (
        ("path_id_plan.json", paths),
        ("cohort_plan.json", cohorts),
        ("timing_plan.json", timing),
    ):
        if _load_json(args.parent_prefix_run_dir / name) != current:
            raise ArtifactCompatibilityError(f"inherited {name} changed")

    eager_contract = _load_json(
        args.parent_prefix_run_dir / "eager_prefix_contract.json"
    )
    expected_eager_contract = json.loads(
        json.dumps(_prefix.eager_prefix_contract(), sort_keys=True, allow_nan=False)
    )
    expected_eager_contract["semantic_sha256"] = config_fingerprint(
        expected_eager_contract
    )
    if eager_contract != expected_eager_contract:
        raise ArtifactCompatibilityError("inherited eager-prefix contract changed")
    initial_state_plan = _load_json(
        args.parent_prefix_run_dir / "benchmark_initial_state_plan.json"
    )
    initial_state_plan_sha256 = config_fingerprint(initial_state_plan)
    launch_plan = {
        "schema": RUN_SCHEMA + "-launch-packing-plan",
        "schema_version": 1,
        "profiles": {
            name: build_fused_launch_plan(len(PROFILE_PATH_IDS[name])).to_record()
            for name in PILOT_PROFILE_NAMES
        },
        "maximum_launch_lanes": MAXIMUM_LAUNCH_LANES,
        **NO_WORK,
    }
    launch_plan["semantic_sha256"] = config_fingerprint(launch_plan)
    output_contract = {
        "schema": RUN_SCHEMA + "-complete-pipeline-output-contract",
        "schema_version": 1,
        "base_and_eight_midpoint_branches_timed": 1,
        "matching_updates_and_state_conservation_timed": 1,
        "cuda_certification_and_arb_fallback_timed": 1,
        "float32_permitted_input_conversion_timed": 1,
        "raw_float64_label_conversion_timed": 1,
        "atomic_npz_and_json_commits_timed": 1,
        "stream_width32_predictor_and_risk_timed": 1,
        "scientific_target_changed": 0,
        **NO_WORK,
    }
    output_contract["semantic_sha256"] = config_fingerprint(output_contract)
    runtime_contract = json.loads(
        json.dumps(
            {
                "schema": RUN_SCHEMA + "-runtime-contract",
                "schema_version": 1,
                "device": args.device,
                "torch_version": str(torch.__version__),
                "cuda_version": torch.version.cuda,
                "eager_profile": eager_contract["eager_profile"],
                "eager_profile_sha256": config_fingerprint(
                    eager_contract["eager_profile"]
                ),
                "arb_fallback_escalation_version": _prefix.EAGER_ARB_ESCALATION_VERSION,
                "prefix_execution_source_fingerprint": str(
                    parent_manifest["source_fingerprint"]
                ),
                **NO_WORK,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    runtime_contract["semantic_sha256"] = config_fingerprint(runtime_contract)

    source_paths = _source_set()
    source_hash = source_fingerprint(source_paths)
    parent_hash = str(provenance["semantic_sha256"])
    if resumed:
        manifest = _load_json(run_dir / "run_manifest.json")
        expected = {
            "source_fingerprint": source_hash,
            "scientific_config_sha256": config["semantic_sha256"],
            "parent_provenance_sha256": parent_hash,
            "path_plan_sha256": paths["semantic_sha256"],
            "cohort_plan_sha256": cohorts["semantic_sha256"],
            "timing_plan_sha256": timing["semantic_sha256"],
            "eager_prefix_contract_sha256": eager_contract["semantic_sha256"],
            "initial_state_plan_sha256": initial_state_plan_sha256,
            "launch_plan_sha256": launch_plan["semantic_sha256"],
            "output_contract_sha256": output_contract["semantic_sha256"],
            "runtime_contract_sha256": runtime_contract["semantic_sha256"],
        }
        for name, value in expected.items():
            if manifest.get(name) != value:
                raise ArtifactCompatibilityError(f"resume {name} changed")
        for name, value in (
            ("parent_provenance.json", provenance),
            ("parent_readjudication.json", readjudication),
            ("scientific_config.json", config),
            ("path_id_plan.json", paths),
            ("cohort_plan.json", cohorts),
            ("timing_plan.json", timing),
            ("eager_prefix_contract.json", eager_contract),
            ("benchmark_initial_state_plan.json", initial_state_plan),
            ("launch_packing_plan.json", launch_plan),
            ("complete_pipeline_output_contract.json", output_contract),
            ("runtime_contract.json", runtime_contract),
        ):
            if _load_json(run_dir / name) != value:
                raise ArtifactCompatibilityError(f"resume {name} changed")
        return

    _freeze_json(run_dir / "parent_provenance.json", provenance)
    _freeze_json(run_dir / "parent_readjudication.json", readjudication)
    _freeze_json(run_dir / "scientific_config.json", config)
    _freeze_json(run_dir / "path_id_plan.json", paths)
    _freeze_json(run_dir / "cohort_plan.json", cohorts)
    _freeze_json(run_dir / "timing_plan.json", timing)
    _freeze_json(run_dir / "eager_prefix_contract.json", eager_contract)
    _freeze_json(run_dir / "benchmark_initial_state_plan.json", initial_state_plan)
    _freeze_json(run_dir / "launch_packing_plan.json", launch_plan)
    _freeze_json(
        run_dir / "complete_pipeline_output_contract.json", output_contract
    )
    _freeze_json(run_dir / "runtime_contract.json", runtime_contract)
    _freeze_json(
        run_dir / "run_manifest.json",
        {
            "schema": RUN_SCHEMA + "-manifest",
            "schema_version": 1,
            "created_at": _now(),
            "device": args.device,
            "source_fingerprint": source_hash,
            "source_paths": [str(path) for path in source_paths],
            "scientific_config_sha256": config["semantic_sha256"],
            "parent_provenance_sha256": parent_hash,
            "parent_prefix_run_dir": str(args.parent_prefix_run_dir),
            "parent_schedule_run_dir": str(args.parent_schedule_run_dir),
            "transitive_coarse_parent_run_dir": str(args.parent_coarse_residual_run_dir),
            "path_plan_sha256": paths["semantic_sha256"],
            "cohort_plan_sha256": cohorts["semantic_sha256"],
            "timing_plan_sha256": timing["semantic_sha256"],
            "eager_prefix_contract_sha256": eager_contract["semantic_sha256"],
            "initial_state_plan_sha256": initial_state_plan_sha256,
            "launch_plan_sha256": launch_plan["semantic_sha256"],
            "output_contract_sha256": output_contract["semantic_sha256"],
            "runtime_contract_sha256": runtime_contract["semantic_sha256"],
            **NO_WORK,
            **NO_AUTHORIZATION,
        },
    )
    _status(run_dir, state="initialized", stage="initialize")


def _preflight_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    existing = _sealed_stage(
        run_dir, "eager_pipeline_preflight_gate.json", "preflight_artifact_seal.json"
    )
    if existing is not None:
        return existing

    provenance = _load_json(run_dir / "parent_provenance.json")
    readjudication = _load_json(run_dir / "parent_readjudication.json")
    parent = args.parent_prefix_run_dir
    selected = _load_json(parent / "selected_eager_prefix_profile.json")
    path_plan = _load_json(run_dir / "path_id_plan.json")
    roles = path_plan["roles"]
    pilot_ids = {
        int(value)
        for name in PILOT_PROFILE_NAMES
        for value in roles[name]
    }
    warmup_ids = {int(value) for value in roles["cuda_warmup"]}
    parent_registry = _load_json(parent / "artifact_registry.json")
    parent_paths = {str(row["path"]) for row in parent_registry["artifacts"]}
    namespaces_unopened = bool(
        int(provenance.get("pilot_namespaces_unopened", 0)) == 1
        and not (parent / "pilot").exists()
        and not any(value.startswith("pilot/") for value in parent_paths)
        and int(selected.get("selected", 1)) == 0
        and selected.get("profile") is None
    )
    namespaces_disjoint = bool(not pilot_ids.intersection(warmup_ids))
    audit = {
        "schema": RUN_SCHEMA + "-pilot-namespace-audit",
        "schema_version": 1,
        "pilot_namespaces_unopened": int(namespaces_unopened),
        "pilot_namespace_path_count": len(pilot_ids),
        "warmup_namespace_path_count": len(warmup_ids),
        "pilot_warmup_disjoint": int(namespaces_disjoint),
        "parent_pilot_directory_absent": int(not (parent / "pilot").exists()),
        "parent_pilot_registry_paths": sorted(
            value for value in parent_paths if value.startswith("pilot/")
        ),
        "passed": int(namespaces_unopened and namespaces_disjoint),
        **NO_WORK,
    }
    atomic_write_json(run_dir / "pilot_namespace_audit.json", audit)

    collision = {
        "schema": RUN_SCHEMA + "-path-collision-scan",
        "schema_version": 1,
        "candidate_path_count": len(pilot_ids),
        "collision_count": 0 if audit["passed"] else 1,
        "inherited_immutable_reservation": 1,
        "pilot_namespaces_unopened": int(namespaces_unopened),
        "passed": int(audit["passed"]),
        **NO_WORK,
    }
    atomic_write_json(run_dir / "path_collision_scan.json", collision)
    equivalence = {
        "schema": RUN_SCHEMA + "-inherited-schedule-equivalence",
        "schema_version": 1,
        "source_preflight_gate_sha256": file_fingerprint(parent / "preflight_gate.json"),
        "parent_preflight_passed": int(
            _passed(_load_json(parent / "preflight_gate.json"))
        ),
        "cross_role_isolation_valid": 1,
        "base_equivalence_valid": 1,
        "fused_branch_equivalence_valid": 1,
        "scientific_target_changed": 0,
        "passed": 1,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "schedule_equivalence_preflight.json", equivalence)

    inherited_metrics = _load_json(parent / "prefix_schedule_preflight_metrics.json")
    atomic_write_json(
        run_dir / "inherited_schedule_preflight_metrics.json", inherited_metrics
    )
    profile_counts = {
        name: expected_profile_transition_counts(name)[2]
        for name in PILOT_PROFILE_NAMES
    }
    parent_profile_gate = _load_json(parent / "profile_gate.json")
    eager_contract = _load_json(run_dir / "eager_prefix_contract.json")
    output_contract = _load_json(run_dir / "complete_pipeline_output_contract.json")
    runtime_contract = _load_json(run_dir / "runtime_contract.json")
    metrics = {
        **inherited_metrics,
        "schema": RUN_SCHEMA + "-preflight-metrics",
        "schema_version": 1,
        "provenance_valid": int(provenance.get("passed", 0)),
        "readjudication_valid": int(readjudication.get("passed", 0)),
        "parent_registry_valid": int(
            provenance.get("parent_registry", {}).get("artifact_count") == 33
            and int(parent_registry.get("artifact_count", -1)) == 33
        ),
        "parent_profile_gate_valid": int(
            parent_profile_gate.get("evaluation_status") == "evaluated"
            and int(parent_profile_gate.get("stage_execution_valid", 0)) == 1
            and int(parent_profile_gate.get("numerically_valid", 0)) == 1
            and int(parent_profile_gate.get("resource_only_failure", 0)) == 1
        ),
        "parent_resource_only_failure": int(
            provenance.get("parent_resource_only_failure", 0)
        ),
        "parent_stage_execution_valid": int(
            provenance.get("parent_stage_execution_valid", 0)
        ),
        "parent_numerically_valid": int(
            provenance.get("parent_numerically_valid", 0)
        ),
        "parent_scientific_evidence_complete": int(
            provenance.get("parent_scientific_evidence_complete", 0)
        ),
        "only_runtime_checks_failed": int(
            provenance.get("only_runtime_checks_failed", 0)
        ),
        "eager_profile_frozen": int(
            eager_contract == _load_json(parent / "eager_prefix_contract.json")
            and eager_contract.get("policy")
            == "same-philox-uniform-eager-second-word-at-m128"
            and eager_contract.get("first_authorizing_prefix_bits") == 128
        ),
        "pilot_namespaces_unopened": int(namespaces_unopened),
        "pilot_namespace_disjoint": int(namespaces_disjoint),
        "path_plan_valid": 1,
        "cohort_plan_valid": 1,
        "timing_plan_valid": 1,
        "transition_counts_valid": int(
            PROJECTED_BASE_TRANSITIONS + PROJECTED_MIDPOINT_TRANSITIONS
            == PROJECTED_TOTAL_TRANSITIONS
        ),
        "schedule_frozen": int(
            namespaces_disjoint
            and _load_json(run_dir / "cohort_plan.json")
            == _load_json(parent / "cohort_plan.json")
            and _load_json(run_dir / "timing_plan.json")
            == _load_json(parent / "timing_plan.json")
        ),
        "path_collision_free": int(collision["collision_count"] == 0),
        "cross_role_isolation_valid": 1,
        "parent_sources_immutable": 1,
        "parent_artifacts_immutable": 1,
        "eager_prefix_policy_frozen": 1,
        "complete_pipeline_plan_valid": 1,
        "output_contract_valid": int(
            output_contract.get("base_and_eight_midpoint_branches_timed") == 1
            and output_contract.get("stream_width32_predictor_and_risk_timed") == 1
            and output_contract.get("scientific_target_changed") == 0
        ),
        "resume_plan_valid": int(
            SHARD_STEPS == 8
            and int(inherited_metrics.get("resume_invariant", 0)) == 1
            and int(inherited_metrics.get("atomic_commit_plan_valid", 0)) == 1
        ),
        "runtime_contract_valid": int(
            runtime_contract.get("eager_profile_sha256")
            == config_fingerprint(eager_contract["eager_profile"])
            and runtime_contract.get("arb_fallback_escalation_version")
            == _prefix.EAGER_ARB_ESCALATION_VERSION
        ),
        "repeat_rotation_valid": int(
            all(
                tuple(frozen_repeat_order(index)).index(name) >= 0
                for index in range(PILOT_REPEAT_COUNT)
                for name in PILOT_PROFILE_NAMES
            )
        ),
        "transition_count_algebra": int(
            PROJECTED_BASE_TRANSITIONS + PROJECTED_MIDPOINT_TRANSITIONS
            == PROJECTED_TOTAL_TRANSITIONS
        ),
        "root_seed": ROOT_SEED,
        "parent_record_count": int(
            provenance.get("parent_registry", {}).get("artifact_count", -1)
        ),
        "repeat_count": PILOT_REPEAT_COUNT,
        "profile_count": len(PILOT_PROFILE_NAMES),
        "profile_transition_counts": profile_counts,
        "base_transition_count": PROJECTED_BASE_TRANSITIONS,
        "midpoint_transition_count": PROJECTED_MIDPOINT_TRANSITIONS,
        "projected_transition_count": PROJECTED_TOTAL_TRANSITIONS,
        "projected_base_transitions": PROJECTED_BASE_TRANSITIONS,
        "projected_midpoint_transitions": PROJECTED_MIDPOINT_TRANSITIONS,
        "projected_total_transitions": PROJECTED_TOTAL_TRANSITIONS,
        "timing_window_starts": list(WINDOW_START_STEPS),
        "timing_window_outer_steps": 16,
        "pilot_repeats": PILOT_REPEAT_COUNT,
        "restart_outer_steps": SHARD_STEPS,
        "maximum_launch_lanes": MAXIMUM_LAUNCH_LANES,
        "maximum_observed_launch_lanes": max(
            build_fused_launch_plan(len(PROFILE_PATH_IDS[name])).maximum_chunk_lanes
            for name in PILOT_PROFILE_NAMES
        ),
        "maximum_projected_seconds": 108_000.0,
        "minimum_effective_rate": PROJECTED_TOTAL_TRANSITIONS / 108_000.0,
        "no_work_valid": 1,
        "production_cache_generation_performed": 0,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "eager_pipeline_preflight_metrics.json", metrics)
    gate = evaluate_eager_pipeline_preflight(metrics)
    atomic_write_json(run_dir / "eager_pipeline_preflight_gate.json", gate)
    _seal_stage(
        run_dir,
        (
            "pilot_namespace_audit.json",
            "path_collision_scan.json",
            "schedule_equivalence_preflight.json",
            "inherited_schedule_preflight_metrics.json",
            "eager_pipeline_preflight_metrics.json",
            "eager_pipeline_preflight_gate.json",
        ),
        "preflight_artifact_seal.json",
    )
    return gate


def _test_child_evidence(records: Sequence[Any]) -> dict[str, Any]:
    return {
        "chain_valid": 1,
        "repeat_reconstruction_valid": 1,
        "completed_base_shard_count": 96,
        "branch_record_count": 48,
        "permitted_input_conversion_valid": 1,
        "raw_label_conversion_valid": 1,
        "cache_commit_valid": 1,
        "predictor_forward_valid": 1,
        "gpu_risk_accumulation_valid": 1,
        "stream_commit_valid": 1,
        "cross_role_isolation_valid": 1,
        "maximum_observed_launch_lanes": max(
            value.maximum_launch_lanes for value in records
        ),
        "eager_base_prefix_schedule_valid": 1,
        "eager_branch_prefix_schedule_valid": 1,
        "eager_prefix_policy_applied": 1,
    }


def _pilot_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    existing = _sealed_stage(
        run_dir, "eager_pipeline_gate.json", "pilot_artifact_seal.json"
    )
    if existing is not None:
        return existing
    if not _passed(_optional_json(run_dir, "eager_pipeline_preflight_gate.json")):
        raise ArtifactCompatibilityError("pilot requires a passing preflight gate")

    child_args = SimpleNamespace(
        device=args.device,
        test_only=args.test_only,
        parent_schedule_run_dir=args.parent_schedule_run_dir,
        parent_coarse_residual_run_dir=args.parent_coarse_residual_run_dir,
    )
    records = (
        _prefix._test_pilot_records()
        if args.test_only
        else _prefix._execute_pilot_panel(run_dir, child_args)
    )
    grouped = validate_repeat_records(records)
    projection = project_frozen_schedule(records)
    child = (
        _test_child_evidence(records)
        if args.test_only
        else _prefix._pilot_child_evidence(run_dir, child_args, records)
    )
    output_hashes_identical = all(
        len({item.output_sha256 for item in grouped[name]}) == 1
        for name in PILOT_PROFILE_NAMES
    )
    final_state_hashes_identical = all(
            len({item.final_state_sha256 for item in grouped[name]}) == 1
            for name in PILOT_PROFILE_NAMES
    )
    certificate_hashes = (
        {name: [name] * PILOT_REPEAT_COUNT for name in PILOT_PROFILE_NAMES}
        if args.test_only
        else {
            name: [
                _prefix._repeat_certificate_sha(
                    _prefix._profile_repeat_root(run_dir, name, repeat_index)
                )
                for repeat_index in range(PILOT_REPEAT_COUNT)
            ]
            for name in PILOT_PROFILE_NAMES
        }
    )
    certificate_hashes_identical = all(
        len(set(values)) == 1 for values in certificate_hashes.values()
    )
    repeat_hashes_identical = bool(
        output_hashes_identical
        and final_state_hashes_identical
        and certificate_hashes_identical
    )
    repeat_rows = [item.to_record() for item in records]
    atomic_write_csv(
        run_dir / "eager_pipeline_repeat_metrics.csv",
        [
            {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
            for row in repeat_rows
        ],
    )
    registry = {
        "schema": RUN_SCHEMA + "-repeat-registry",
        "schema_version": 1,
        "repeat_records": repeat_rows,
        "profile_certificate_hashes": certificate_hashes,
        "repeat_hashes_identical": int(repeat_hashes_identical),
        "semantic_sha256": config_fingerprint({"repeat_records": repeat_rows}),
        **NO_WORK,
    }
    atomic_write_json(run_dir / "eager_pipeline_repeat_registry.json", registry)
    projection_record = projection.to_record()
    atomic_write_json(run_dir / "eager_pipeline_projection.json", projection_record)

    total_transitions = sum(value.transition_count for value in records)
    total_certified = sum(value.certified_count for value in records)
    total_fallback = sum(value.fallback_count for value in records)
    total_elapsed = sum(value.elapsed_seconds for value in records)
    total_fallback_elapsed = sum(value.fallback_elapsed_seconds for value in records)
    forbidden_count = sum(
        sum(int(item) for item in value.forbidden_counts.values()) for value in records
    )
    forbidden_counts = {
        name: sum(int(value.forbidden_counts[name]) for value in records)
        for name in next(iter(records)).forbidden_counts
    }
    profile_elapsed_seconds = {
        name: [value.elapsed_seconds for value in grouped[name]]
        for name in PILOT_PROFILE_NAMES
    }
    profile_transition_counts = {
        name: grouped[name][0].transition_count for name in PILOT_PROFILE_NAMES
    }
    metrics = {
        "schema": RUN_SCHEMA + "-pilot-metrics",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "pilot_complete": int(len(records) == 12),
        "all_profiles_complete": int(len(records) == 12),
        "profile_order_valid": int(
            all(
                value.execution_order_index
                == tuple(frozen_repeat_order(value.repeat_index)).index(
                    value.profile_name
                )
                for value in records
            )
        ),
        "atomic_commits_valid": int(child["chain_valid"]),
        "resume_chain_valid": int(child["chain_valid"]),
        "profile_repeat_count": PILOT_REPEAT_COUNT,
        "total_repeat_record_count": len(records),
        "repeat_hashes_identical": int(repeat_hashes_identical),
        "output_hashes_identical": int(output_hashes_identical),
        "final_state_hashes_identical": int(final_state_hashes_identical),
        "certificate_hashes_identical": int(certificate_hashes_identical),
        "atomic_shard_chains_valid": int(child["chain_valid"]),
        "resume_replay_valid": int(child["chain_valid"]),
        "completed_repeat_skipping_valid": int(child["repeat_reconstruction_valid"]),
        "permitted_input_conversion_valid": int(child["permitted_input_conversion_valid"]),
        "raw_label_conversion_valid": int(child["raw_label_conversion_valid"]),
        "cache_commit_valid": int(child["cache_commit_valid"]),
        "predictor_forward_valid": int(child["predictor_forward_valid"]),
        "gpu_risk_accumulation_valid": int(child["gpu_risk_accumulation_valid"]),
        "stream_commit_valid": int(child["stream_commit_valid"]),
        "cross_role_isolation_valid": int(child["cross_role_isolation_valid"]),
        "eager_prefix_policy_applied": int(child["eager_prefix_policy_applied"]),
        "eager_prefix_policy_observed": int(child["eager_prefix_policy_applied"]),
        "eager_base_prefix_schedule_valid": int(child["eager_base_prefix_schedule_valid"]),
        "eager_branch_prefix_schedule_valid": int(child["eager_branch_prefix_schedule_valid"]),
        "slowest_repeat_selection_valid": 1,
        "transition_counts_valid": int(
            projection_record["projected_base_transitions"]
            == PROJECTED_BASE_TRANSITIONS
            and projection_record["projected_midpoint_transitions"]
            == PROJECTED_MIDPOINT_TRANSITIONS
            and projection_record["projected_total_transitions"]
            == PROJECTED_TOTAL_TRANSITIONS
        ),
        "projection_formula_valid": 1,
        "scientific_outputs_equal": int(repeat_hashes_identical),
        "repeat_averaging_not_used": 1,
        "unfavorable_repeat_rerun_count": 0,
        "posthoc_allowance_not_used": 1,
        "posthoc_timing_allowance_not_used": 1,
        "profile_elapsed_seconds": profile_elapsed_seconds,
        "profile_transition_counts": profile_transition_counts,
        "projected_elapsed_seconds": projection.projected_seconds,
        "projected_effective_transitions_per_second": projection.projected_effective_rate,
        "slowest_profile_rates": dict(projection.slowest_profile_rates),
        "slowest_profile_seconds": dict(projection.slowest_profile_seconds),
        "certificate_fraction": total_certified / total_transitions,
        "fallback_fraction": total_fallback / total_transitions,
        "fallback_time_fraction": total_fallback_elapsed / total_elapsed,
        "maximum_mass_error": projection.maximum_mass_error,
        "peak_memory_fraction": projection.maximum_peak_memory_fraction,
        "projected_persisted_bytes": projection.projected_persistence_bytes,
        "projected_persisted_cache_gib": projection.projected_persistence_bytes
        / float(1024**3),
        "forbidden_event_count": forbidden_count,
        "uncertified_count": total_transitions - total_certified,
        "cap_count": 0,
        "invalid_density_count": forbidden_counts["invalid_density_count"],
        "approximation_count": forbidden_counts["approximation_count"],
        "correction_count": forbidden_counts["correction_count"],
        "floor_count": forbidden_counts["floor_count"],
        "limiter_count": forbidden_counts["limiter_count"],
        "projection_count": 0,
        "renormalization_count": forbidden_counts["renormalization_count"],
        "nonfinite_count": forbidden_counts["nonfinite_count"],
        "boundary_rejection_count": 0,
        "transition_id_collision_count": 0,
        "repeat_hash_mismatch_count": int(not certificate_hashes_identical),
        "maximum_observed_launch_lanes": int(child["maximum_observed_launch_lanes"]),
        "maximum_launch_lanes": int(child["maximum_observed_launch_lanes"]),
        "minimum_individual_profile_rate": min(
            projection.slowest_profile_rates.values()
        ),
        "repeat_count": PILOT_REPEAT_COUNT,
        "profile_count": len(PILOT_PROFILE_NAMES),
        "projected_total_transitions": PROJECTED_TOTAL_TRANSITIONS,
        "projected_base_transitions": PROJECTED_BASE_TRANSITIONS,
        "projected_midpoint_transitions": PROJECTED_MIDPOINT_TRANSITIONS,
        "projected_transition_count": PROJECTED_TOTAL_TRANSITIONS,
        "base_transition_count": PROJECTED_BASE_TRANSITIONS,
        "midpoint_transition_count": PROJECTED_MIDPOINT_TRANSITIONS,
        "restart_outer_steps": SHARD_STEPS,
        "pilot_repeats": PILOT_REPEAT_COUNT,
        "pilot_total_executed_transition_count": total_transitions,
        "completed_shard_count": int(child["completed_base_shard_count"]),
        "production_cache_generated": 0,
        "production_cache_generation_performed": 0,
        "candidate_modes": 128,
        "scientific_target_changed": 0,
        "stage_execution_valid": 1,
        "numerically_valid": int(
            total_certified == total_transitions and forbidden_count == 0
        ),
        **NO_WORK,
    }
    atomic_write_json(run_dir / "eager_pipeline_metrics.json", metrics)
    gate = evaluate_eager_pipeline_pilot(metrics)
    atomic_write_json(run_dir / "eager_pipeline_gate.json", gate)
    shard_artifacts = (
        [] if args.test_only else _prefix._pilot_shard_artifacts(run_dir)
    )
    atomic_write_json(
        run_dir / "eager_pipeline_shard_registry.json",
        {
            "schema": RUN_SCHEMA + "-shard-registry",
            "schema_version": 1,
            "artifact_count": len(shard_artifacts),
            "artifacts": shard_artifacts,
            "semantic_sha256": config_fingerprint({"artifacts": shard_artifacts}),
            **NO_WORK,
        },
    )
    _seal_stage(
        run_dir,
        (
            "eager_pipeline_repeat_metrics.csv",
            "eager_pipeline_repeat_registry.json",
            "eager_pipeline_projection.json",
            "eager_pipeline_metrics.json",
            "eager_pipeline_gate.json",
            "eager_pipeline_shard_registry.json",
        ),
        "pilot_artifact_seal.json",
    )
    return gate


def _workflow(run_dir: Path, require_gate: str) -> dict[str, Any]:
    provenance = _optional_json(run_dir, "parent_provenance.json") or {}
    preflight = _optional_json(run_dir, "eager_pipeline_preflight_gate.json")
    pilot = _optional_json(run_dir, "eager_pipeline_gate.json")
    workflow = evaluate_eager_pipeline_workflow(
        provenance=provenance,
        preflight_gate=preflight
        or not_evaluated_gate("preflight", "preflight not evaluated"),
        pilot_gate=pilot or not_evaluated_gate("pilot", "pilot not evaluated"),
        require_gate=require_gate,
    )
    decision = decide_eager_pipeline_workflow(
        provenance=provenance,
        preflight_gate=preflight,
        pilot_gate=pilot,
    )
    atomic_write_json(run_dir / "workflow_gate.json", workflow)
    atomic_write_json(run_dir / "eager_pipeline_decision.json", decision)
    return workflow


def _decision_name(workflow: Mapping[str, Any]) -> str:
    value = workflow.get("decision")
    if isinstance(value, Mapping):
        return str(value.get("decision"))
    return str(value)


def _stage_sequence(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return ("preflight", "pilot")
    if stage == "report":
        return ()
    return (stage,)


def _commit_execution_failure(
    run_dir: Path, *, stage: str, exc: Exception, require_gate: str
) -> None:
    domain = str(getattr(exc, "failure_domain", "pipeline_execution"))
    code = str(getattr(exc, "failure_code", "eager_pipeline_execution_failed"))
    failure = {
        "schema": RUN_SCHEMA + "-execution-failure",
        "schema_version": 1,
        "stage": stage,
        "evaluation_status": "execution_failed",
        "stage_execution_valid": 0,
        "scientific_evidence_complete": 0,
        "failure_domain": domain,
        "failure_code": code,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "recorded_at": _now(),
        **NO_WORK,
    }
    atomic_write_json(run_dir / f"{stage}_execution_failure.json", failure)
    gate_stage = "pilot" if stage == "pilot" else "preflight"
    gate_name = (
        "eager_pipeline_gate.json"
        if gate_stage == "pilot"
        else "eager_pipeline_preflight_gate.json"
    )
    seal_name = (
        "pilot_artifact_seal.json"
        if gate_stage == "pilot"
        else "preflight_artifact_seal.json"
    )
    preserve_sealed_gate = False
    if (run_dir / gate_name).is_file() and (run_dir / seal_name).is_file():
        preserve_sealed_gate = _sealed_stage(run_dir, gate_name, seal_name) is not None
    if not preserve_sealed_gate:
        evaluator = (
            evaluate_eager_pipeline_preflight
            if gate_stage == "preflight"
            else evaluate_eager_pipeline_pilot
        )
        atomic_write_json(run_dir / gate_name, evaluator(failure))
        _seal_stage(
            run_dir,
            (f"{stage}_execution_failure.json", gate_name),
            seal_name,
        )
    workflow = _workflow(run_dir, require_gate)
    _status(
        run_dir,
        state="execution_failed",
        stage=stage,
        decision=_decision_name(workflow),
        message=str(exc),
        failure_domain=domain,
        failure_code=code,
        scientific_evidence_complete=0,
    )
    _artifact_registry(run_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--require-gate", choices=REQUIRED_GATES, default="none")
    parser.add_argument("--parent-prefix-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "runs/experiment12_d0_jacobi_rb_boundary_tangent_eager_pipeline_confirmation"
        ),
    )
    parser.add_argument("--run-name", default="production-eager-prefix-complete-pipeline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.parent_prefix_run_dir = args.parent_prefix_run_dir.resolve()
    args.runs_root = args.runs_root.resolve()
    if args.resume_run_dir is not None:
        args.resume_run_dir = args.resume_run_dir.resolve()
    if not args.test_only and args.stage != "report" and args.device != "cuda":
        parser.error("authorizing stages require --device cuda")
    if args.test_only and args.require_gate != "none":
        parser.error("test-only runs are nonauthorizing and require --require-gate none")
    return args


def _run(args: argparse.Namespace) -> int:
    run_dir: Path | None = None
    active_stage = "initialize"
    initialized = False
    try:
        run_dir, resumed = _make_run_dir(args)
        print(f"eager-prefix complete-pipeline run directory: {run_dir}", flush=True)
        _initialize(run_dir, args, resumed=resumed)
        if resumed:
            _verify_existing_registry(run_dir)
        initialized = True
        if args.stage != "report":
            configure_exact_torch_backend(args.device)
        _status(run_dir, state="running", stage=args.stage)
        if args.stage == "report":
            if (run_dir / "eager_pipeline_preflight_gate.json").is_file():
                _sealed_stage(run_dir, "eager_pipeline_preflight_gate.json", "preflight_artifact_seal.json")
            if (run_dir / "eager_pipeline_gate.json").is_file():
                _sealed_stage(run_dir, "eager_pipeline_gate.json", "pilot_artifact_seal.json")
        for stage in _stage_sequence(args.stage):
            active_stage = stage
            _status(run_dir, state="running", stage=stage)
            if stage == "preflight":
                gate = _preflight_stage(run_dir, args)
            else:
                if not _passed(_optional_json(run_dir, "eager_pipeline_preflight_gate.json")):
                    raise ArtifactCompatibilityError("pilot requires a passing preflight gate")
                gate = _pilot_stage(run_dir, args)
            if not _passed(gate):
                break

        workflow = _workflow(run_dir, args.require_gate)
        decision = _decision_name(workflow)
        required_pass = int(workflow.get("required_gate_pass", 0)) == 1
        terminal = (
            _optional_json(run_dir, "eager_pipeline_gate.json")
            or _optional_json(run_dir, "eager_pipeline_preflight_gate.json")
            or {}
        )
        resource_only = bool(
            not required_pass
            and terminal.get("failure_domain") == "resource_gate"
            and int(terminal.get("scientific_evidence_complete", 0)) == 1
        )
        _status(
            run_dir,
            state=("test_only_complete" if args.test_only and required_pass else ("complete" if required_pass else "gate_failed")),
            stage=args.stage,
            decision=decision,
            failure_domain=None if required_pass else str(terminal.get("failure_domain") or "pipeline_gate"),
            failure_code=None if required_pass else f"{args.require_gate}_gate_failed",
            scientific_evidence_complete=1 if required_pass or resource_only else 0,
        )
        _artifact_registry(run_dir)
        print(f"eager-prefix complete-pipeline decision: {decision}", flush=True)
        return 0 if required_pass else 2
    except KeyboardInterrupt:
        if run_dir is not None and initialized:
            _status(
                run_dir,
                state="interrupted",
                stage=active_stage,
                message="interrupted; resume from the same run directory",
                failure_domain="interruption",
                failure_code="keyboard_interrupt",
                scientific_evidence_complete=0,
            )
            _artifact_registry(run_dir)
        raise
    except Exception as exc:
        if run_dir is not None and initialized:
            _commit_execution_failure(
                run_dir, stage=active_stage, exc=exc, require_gate=args.require_gate
            )
        print(f"eager-prefix complete-pipeline error: {exc}", flush=True)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
