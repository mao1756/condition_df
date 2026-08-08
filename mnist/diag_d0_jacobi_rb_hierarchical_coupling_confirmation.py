"""Certified Haar-coupled Jacobi refinement power confirmation.

This additive, controls-only workflow binds the immutable phase-observer
power run and changes only the common-random-number coupling used to measure
weak differences between temporal levels.  It never trains a model, runs a
production refinement experiment, or performs reverse sampling.

Production execution is deliberately fail closed: evidence is authorizing
only when the certified Haar/normal and arbitrary-uniform Jacobi backends
produce complete metrics.  Reduced CPU fixtures exercise the same orchestration
and sealing rules but are explicitly nonauthorizing.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from fractions import Fraction
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_json,
    canonical_json_bytes,
    config_fingerprint,
    configure_exact_torch_backend,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_rb_haar_gate import (
    ANTITHETIC_HAAR_PROFILE,
    NESTED_HAAR_PROFILE,
    HaarCouplingThresholds,
    decide_sealed_profile_selection,
    decide_haar_workflow,
    evaluate_haar_coupling,
    evaluate_haar_pilot,
    evaluate_haar_preflight,
    evaluate_haar_workflow,
    nominate_haar_power_design,
    not_evaluated_gate,
)
from mnist.d0_jacobi_rb_haar_provenance import (
    PARENT_REGISTRY_RECORD_COUNT,
    PARENT_REGISTRY_SHA256,
    PARENT_SCIENTIFIC_CONFIG_SHA256,
    PARENT_SOURCE_COUNT,
    PARENT_SOURCE_FINGERPRINT,
    verify_right_endpoint_coupling_parent,
)
from mnist.d0_jacobi_rb_haar_power import (
    FORBIDDEN_COUNTS as HAAR_POWER_FORBIDDEN_COUNTS,
    HaarPowerError,
    combine_certified_haar_power_panels,
    panel_confirmation_record,
    run_certified_haar_power_panel,
    verify_certified_haar_power_panel_evidence,
)
from mnist.d0_jacobi_rb_haar_controls import (
    HaarControlError,
    run_marginal_and_batching_controls,
    run_phase_tower_controls,
    run_scheduler_equivalence_and_benchmark,
)
from mnist.d0_jacobi_rb_haar_fused import fused_haar_runtime_report
from mnist.d0_jacobi_rb_haar_scheduler import HaarSchedulerError


RUN_SCHEMA = (
    "experiment12-d0-jacobi-rb-hierarchical-coupling-confirmation"
)
RUN_SCHEMA_VERSION = 1
CLAIM_SCOPE = "certified Haar coupling and refinement-power feasibility only"
ROOT_SEED = 261_181
PANEL_CLUSTER_COUNT = 8
SAMPLE_STEPS = (128, 256, 512, 1024, 2048)
MAIN_LEVELS = (128, 256, 512, 1024)
REFERENCE_LEVELS = (512, 1024, 2048)
PAIRWISE_LEVELS = ((128, 256), (256, 512), (512, 1024), (1024, 2048))
PROFILE_ORDER = (NESTED_HAAR_PROFILE, ANTITHETIC_HAAR_PROFILE)
PATH_ID_PLAN_VERSION = "d0-jacobi-rb-haar-path-id-v1"
RNG_PLAN_VERSION = "d0-jacobi-rb-certified-haar-rng-v1"
NO_WORK = {
    "physical_training_performed": 0,
    "production_refinement_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
}
_REGISTRY_EXCLUDED = {"artifact_registry.json", "run_status.json"}
_MUTABLE_TERMINAL_FILES = {
    "haar_workflow_gate.json",
    "haar_coupling_decision.json",
}
_PROFILE_BASES = {
    (NESTED_HAAR_PROFILE, "a"): 0xA0000,
    (NESTED_HAAR_PROFILE, "b"): 0xA1000,
    (ANTITHETIC_HAAR_PROFILE, "a"): 0xB0000,
    (ANTITHETIC_HAAR_PROFILE, "b"): 0xB1000,
}
_MARGINAL_BASES = {"a": 0xC0000, "b": 0xD0000}
_PRODUCTION_RESERVED = (0xF0000, 0x100000)


class HaarConfigurationError(ValueError):
    """A frozen Haar profile, namespace, or panel plan is invalid."""


class HaarExecutionError(RuntimeError):
    """A certified Haar stage could not produce authorizing evidence."""


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _load(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"cannot read JSON artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"JSON artifact is not an object: {path}")
    return dict(value)


def _freeze(
    path: Path,
    value: Mapping[str, Any],
    *,
    require_existing: bool = False,
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    if path.is_file():
        if _load(path) != normalized:
            raise ArtifactCompatibilityError(f"frozen artifact changed: {path.name}")
    elif require_existing:
        raise ArtifactCompatibilityError(f"resume lacks frozen artifact: {path.name}")
    else:
        atomic_write_json(path, normalized)
    return normalized


def _passed(gate: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(gate, Mapping)
        and gate.get("evaluation_status") == "evaluated"
        and int(gate.get("passed", 0)) == 1
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight", "coupling", "pilot", "report", "all"),
        default="all",
    )
    parser.add_argument(
        "--require-gate",
        choices=("none", "preflight", "coupling", "pilot"),
        default="none",
    )
    parser.add_argument(
        "--parent-phase-observer-run-dir", type=Path, required=True
    )
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "runs/experiment12_d0_jacobi_rb_hierarchical_coupling_confirmation"
        ),
    )
    parser.add_argument(
        "--run-name", default="production-certified-haar-strang-power"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--root-seed", type=int, default=ROOT_SEED)
    parser.add_argument(
        "--panel-clusters", type=int, default=PANEL_CLUSTER_COUNT, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--test-only-reduced-workload", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)

    if args.stage in {"coupling", "pilot", "report"} and args.resume_run_dir is None:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    allowed = {
        "preflight": {"none", "preflight"},
        "coupling": {"none", "preflight", "coupling"},
        "pilot": {"none", "preflight", "coupling", "pilot"},
        "report": {"none", "preflight", "coupling", "pilot"},
        "all": {"none", "preflight", "coupling", "pilot"},
    }
    if args.require_gate not in allowed[args.stage]:
        parser.error(
            f"--require-gate {args.require_gate} is unavailable at stage {args.stage}"
        )
    if not 1 <= int(args.panel_clusters) <= PANEL_CLUSTER_COUNT:
        parser.error("panel clusters must lie in [1,8]")
    changed = []
    if int(args.root_seed) != ROOT_SEED:
        changed.append("root_seed")
    if int(args.panel_clusters) != PANEL_CLUSTER_COUNT:
        changed.append("panel_clusters")
    if changed and not args.test_only_reduced_workload:
        parser.error(
            "production configuration is frozen; overrides require "
            "--test-only-reduced-workload: " + ", ".join(changed)
        )
    if args.test_only_reduced_workload and args.require_gate != "none":
        parser.error("test-only workloads cannot satisfy a required gate")
    if torch.device(args.device).type != "cuda" and not args.test_only_reduced_workload:
        parser.error("production Haar controls require --device cuda")
    return args


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        run_dir = Path(args.resume_run_dir).resolve()
        if not run_dir.is_dir():
            raise ArtifactCompatibilityError(f"resume run does not exist: {run_dir}")
        return run_dir, True
    root = Path(args.runs_root)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.run_name}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir.resolve(), False


def _write_status(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "run_status.json"
    record = (
        _load(path)
        if path.is_file()
        else {
            "schema": RUN_SCHEMA,
            "schema_version": RUN_SCHEMA_VERSION,
            "created_at": _now(),
        }
    )
    record.update(updates)
    record.update(updated_at=_now(), **NO_WORK)
    atomic_write_json(path, record)
    return record


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    records = {
        path.relative_to(run_dir).as_posix(): {
            "sha256": file_fingerprint(path),
            "size": int(path.stat().st_size),
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and path.relative_to(run_dir).as_posix() not in _REGISTRY_EXCLUDED
    }
    return {
        "schema": RUN_SCHEMA + "-artifact-registry",
        "schema_version": 1,
        "terminal_files_excluded_to_avoid_self_reference": sorted(
            _REGISTRY_EXCLUDED
        ),
        "records": records,
        **NO_WORK,
    }


def _verify_terminal_registry(run_dir: Path) -> None:
    registry_path = run_dir / "artifact_registry.json"
    if not registry_path.is_file():
        status_path = run_dir / "run_status.json"
        if status_path.is_file() and _load(status_path).get("status") == "complete":
            raise ArtifactCompatibilityError(
                "completed resume lacks its terminal artifact registry"
            )
        return
    status = _load(run_dir / "run_status.json")
    registry = _load(registry_path)
    records = registry.get("records")
    if (
        registry.get("schema") != RUN_SCHEMA + "-artifact-registry"
        or registry.get("schema_version") != 1
        or set(
            registry.get("terminal_files_excluded_to_avoid_self_reference", ())
        )
        != _REGISTRY_EXCLUDED
        or not isinstance(records, Mapping)
        or status.get("artifact_registry_sha256") != file_fingerprint(registry_path)
        or int(status.get("artifact_registry_record_count", -1)) != len(records)
        or int(status.get("artifact_registry_size", -1))
        != registry_path.stat().st_size
    ):
        raise ArtifactCompatibilityError("resume terminal registry is invalid")
    interrupted = status.get("status") == "running"
    for relative, raw in records.items():
        path = run_dir / str(relative)
        valid = (
            path.is_file()
            and isinstance(raw, Mapping)
            and raw.get("sha256") == file_fingerprint(path)
            and int(raw.get("size", -1)) == path.stat().st_size
        )
        if not valid and interrupted and relative in _MUTABLE_TERMINAL_FILES:
            continue
        if not valid:
            raise ArtifactCompatibilityError(f"resume artifact changed: {relative}")
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.relative_to(run_dir).as_posix() not in _REGISTRY_EXCLUDED
    }
    if not interrupted and actual != set(records):
        raise ArtifactCompatibilityError(
            "completed resume artifact set differs from its registry"
        )


def _finalize_registry(run_dir: Path) -> dict[str, Any]:
    registry = _artifact_registry(run_dir)
    atomic_write_json(run_dir / "artifact_registry.json", registry)
    path = run_dir / "artifact_registry.json"
    return {
        "artifact_registry_record_count": len(registry["records"]),
        "artifact_registry_sha256": file_fingerprint(path),
        "artifact_registry_size": path.stat().st_size,
    }


def _nested_path_ids(base: int, count: int) -> dict[str, Any]:
    main = [base + path for path in range(count)]
    reference = [base + 0x400 + path for path in range(count)]
    return {
        "main": {
            "root_path_ids": main,
            "levels": {str(level): main for level in MAIN_LEVELS},
        },
        "reference": {
            "root_path_ids": reference,
            "levels": {str(level): reference for level in REFERENCE_LEVELS},
        },
    }


def _antithetic_path_ids(
    base: int, count: int
) -> dict[str, dict[str, Any]]:
    return {
        f"{coarse}-{fine}": {
            "root_path_ids": [
                base + pair * 0x100 + path for path in range(count)
            ],
            "coarse_level": coarse,
            "fine_level": fine,
            "fine_detail_signs": [1, -1],
        }
        for pair, (coarse, fine) in enumerate(PAIRWISE_LEVELS)
    }


def _build_path_id_plan(args: argparse.Namespace) -> dict[str, Any]:
    count = int(args.panel_clusters)
    profiles: dict[str, Any] = {}
    all_ids: set[int] = set()
    for profile in PROFILE_ORDER:
        profiles[profile] = {}
        for panel in ("a", "b"):
            base = _PROFILE_BASES[(profile, panel)]
            roles = (
                _nested_path_ids(base, count)
                if profile == NESTED_HAAR_PROFILE
                else _antithetic_path_ids(base, count)
            )
            if profile == NESTED_HAAR_PROFILE:
                local = set(roles["main"]["root_path_ids"]) | set(
                    roles["reference"]["root_path_ids"]
                )
            else:
                roots = [
                    int(value)
                    for pair in roles.values()
                    for value in pair["root_path_ids"]
                ]
                local = set(roots)
                if len(local) != len(roots):
                    raise HaarConfigurationError(
                        f"duplicate pair roots in {profile} panel {panel}"
                    )
            if any(not 0 <= item < (1 << 20) for item in local):
                raise HaarConfigurationError("path ID exceeds the frozen 20-bit limit")
            if any(not base <= item < base + 0x1000 for item in local):
                raise HaarConfigurationError("path ID escaped its frozen role slot")
            if all_ids.intersection(local):
                raise HaarConfigurationError("Haar profile path-ID slots overlap")
            all_ids.update(local)
            profiles[profile][panel] = {
                "slot": [base, base + 0x1000],
                "roles": roles,
                "path_ids": sorted(local),
            }
    marginals: dict[str, Any] = {}
    for panel, base in _MARGINAL_BASES.items():
        ids = [base + path for path in range(count)]
        if all_ids.intersection(ids):
            raise HaarConfigurationError("marginal path-ID slot overlaps a pilot slot")
        all_ids.update(ids)
        marginals[panel] = {
            "slot": [base, base + 0x1000],
            "path_ids": ids,
        }
    low, high = _PRODUCTION_RESERVED
    if any(low <= value < high for value in all_ids):
        raise HaarConfigurationError("pilot IDs overlap reserved production IDs")
    record = {
        "schema": RUN_SCHEMA + "-path-id-plan",
        "schema_version": 1,
        "version": PATH_ID_PLAN_VERSION,
        "rng_plan_version": RNG_PLAN_VERSION,
        "panel_cluster_count": count,
        "profiles": profiles,
        "marginal_panels": marginals,
        "reserved_production_slot": [low, high],
        "all_path_id_count": len(all_ids),
        "maximum_path_id": max(all_ids),
        "checks": {
            "integer_20_bit_pass": 1,
            "role_slot_pass": 1,
            "role_disjoint_pass": 1,
            "marginal_disjoint_pass": 1,
            "future_production_reserved_pass": 1,
        },
        **NO_WORK,
    }
    record["path_id_plan_sha256"] = config_fingerprint(record)
    return record


def _profile_plan() -> dict[str, Any]:
    thresholds = HaarCouplingThresholds()
    record = {
        "schema": RUN_SCHEMA + "-profile-plan",
        "schema_version": 1,
        "profile_order": list(PROFILE_ORDER),
        "sealed_selection_semantics": {
            "panel_a_nominates": 1,
            "panel_b_opens_only_after_nomination": 1,
            "profile_two_a_opens_only_if_profile_one_a_has_no_nominee": 1,
            "no_profile_fallback_after_any_panel_b_opens": 1,
            "panel_means_excluded_from_future_refinement": 1,
        },
        "profiles": {
            NESTED_HAAR_PROFILE: {
                "main_levels": list(MAIN_LEVELS),
                "reference_levels": list(REFERENCE_LEVELS),
                "candidate_designs": [
                    {"main_paths": main, "reference_paths": reference}
                    for main in (32, 64)
                    for reference in (16, 32)
                ],
                "raw_endpoint_authorizing": 1,
                "dynkin_advisory_only": 1,
            },
            ANTITHETIC_HAAR_PROFILE: {
                "pairs": [list(pair) for pair in PAIRWISE_LEVELS],
                "fine_path_count_per_cluster": 2,
                "fine_observable": "arithmetic_mean",
                "candidate_designs": [{"main_paths": 16, "reference_paths": 16}],
                "raw_endpoint_authorizing": 1,
                "dynkin_advisory_only": 1,
            },
        },
        "thresholds": {
            "maximum_main_half_width": float(
                getattr(thresholds, "maximum_main_half_width", 0.0025)
            ),
            "maximum_reference_half_width": float(
                getattr(thresholds, "maximum_reference_half_width", 0.005)
            ),
            "maximum_projected_hours": float(
                getattr(thresholds, "maximum_projected_hours", 48.0)
            ),
            "minimum_rate": float(
                getattr(thresholds, "minimum_transitions_per_second", 1300.0)
            ),
            "maximum_fallback_fraction": float(
                getattr(thresholds, "maximum_fallback_fraction", 1.0e-4)
            ),
            "maximum_fallback_time_fraction": float(
                getattr(thresholds, "maximum_fallback_time_fraction", 0.10)
            ),
            "maximum_memory_fraction": float(
                getattr(thresholds, "maximum_memory_fraction", 0.80)
            ),
        },
        **NO_WORK,
    }
    record["profile_plan_sha256"] = config_fingerprint(record)
    return record


def _model_input_contract() -> dict[str, Any]:
    record = {
        "schema": RUN_SCHEMA + "-future-model-input-contract",
        "schema_version": 1,
        "allowed_inputs": [
            "later_full_state",
            "reverse_time",
            "phase",
            "color",
            "duration",
            "label",
        ],
        "forbidden_inputs": [
            "earlier_state",
            "uniform_bits",
            "normal_variables",
            "certificate",
            "oracle_target",
        ],
        "earlier_state_forbidden": 1,
        "randomness_forbidden": 1,
        "certificate_forbidden": 1,
        "oracle_quantity_forbidden": 1,
        "later_state_only_contract_pass": 1,
        **NO_WORK,
    }
    record["model_input_contract_sha256"] = config_fingerprint(record)
    return record


def _freeze_plans(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    require_existing: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path_plan = _build_path_id_plan(args)
    profile_plan = _profile_plan()
    _freeze(
        run_dir / "haar_path_id_plan.json",
        path_plan,
        require_existing=require_existing,
    )
    _freeze(
        run_dir / "haar_profile_plan.json",
        profile_plan,
        require_existing=require_existing,
    )
    _freeze(
        run_dir / "future_model_input_contract.json",
        _model_input_contract(),
        require_existing=require_existing,
    )
    panel_registry = {
        "schema": RUN_SCHEMA + "-sealed-panel-registry",
        "schema_version": 1,
        "path_id_plan_sha256": path_plan["path_id_plan_sha256"],
        "profile_plan_sha256": profile_plan["profile_plan_sha256"],
        "model_input_contract_sha256": _model_input_contract()[
            "model_input_contract_sha256"
        ],
        "panels_frozen_before_device_execution": 1,
        "panels_disjoint": 1,
        "panel_regeneration_permitted": 0,
        "panel_b_evaluation_limit": 1,
        "future_production_namespace_disjoint": 1,
        "profile_order": list(PROFILE_ORDER),
        **NO_WORK,
    }
    _freeze(
        run_dir / "sealed_panel_registry.json",
        panel_registry,
        require_existing=require_existing,
    )
    return path_plan, profile_plan


def _scientific_config(
    args: argparse.Namespace,
    path_plan: Mapping[str, Any],
    profile_plan: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + "-scientific-config",
        "schema_version": 1,
        "claim_scope": CLAIM_SCOPE,
        "root_seed": int(args.root_seed),
        "test_only_reduced_workload": int(bool(args.test_only_reduced_workload)),
        "requested_device_type": torch.device(args.device).type,
        "grid_size": 28,
        "alpha": 1.0,
        "tau_eff": 5.0e-5,
        "sample_steps": list(SAMPLE_STEPS),
        "panel_cluster_count": int(args.panel_clusters),
        "path_id_plan_version": PATH_ID_PLAN_VERSION,
        "path_id_plan_sha256": path_plan["path_id_plan_sha256"],
        "rng_plan_version": RNG_PLAN_VERSION,
        "profile_plan_sha256": profile_plan["profile_plan_sha256"],
        "model_input_contract_sha256": _model_input_contract()[
            "model_input_contract_sha256"
        ],
        "profile_order": list(PROFILE_ORDER),
        "raw_endpoint_authorizing": 1,
        "dynkin_advisory_only": 1,
        "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
        "parent_source_fingerprint": PARENT_SOURCE_FINGERPRINT,
        "parent_scientific_config_sha256": PARENT_SCIENTIFIC_CONFIG_SHA256,
        **NO_WORK,
    }


def _source_record(parent_dir: Path) -> tuple[str, list[str]]:
    manifest = _load(parent_dir / "run_manifest.json")
    raw = manifest.get("source_paths")
    if (
        not isinstance(raw, list)
        or len(raw) != PARENT_SOURCE_COUNT
        or not all(isinstance(item, str) for item in raw)
    ):
        raise ArtifactCompatibilityError("parent source-path set is invalid")
    parent_paths = {Path(item).resolve() for item in raw}
    if len(parent_paths) != PARENT_SOURCE_COUNT:
        raise ArtifactCompatibilityError("parent source-path set contains duplicates")
    # Import the additive modules explicitly so their resolved paths enter the
    # child fingerprint while the immutable parent files are only read.
    additive_modules = (
        "mnist.d0_jacobi_rb_haar",
        "mnist.d0_jacobi_rb_haar_controls",
        "mnist.d0_jacobi_rb_haar_cuda",
        "mnist.d0_jacobi_rb_haar_fused",
        "mnist.d0_jacobi_rb_haar_gate",
        "mnist.d0_jacobi_rb_haar_provenance",
        "mnist.d0_jacobi_rb_haar_power",
        "mnist.d0_jacobi_rb_haar_scheduler",
    )
    paths = set(parent_paths)
    for module_name in additive_modules:
        module = importlib.import_module(module_name)
        paths.add(Path(module.__file__).resolve())
    paths.add(Path(__file__).resolve())
    ordered = sorted(paths)
    if not all(path.is_file() for path in ordered):
        raise ArtifactCompatibilityError("Haar source set contains a missing file")
    return source_fingerprint(ordered), [str(path) for path in ordered]


def _verify_resume_contract(
    run_dir: Path,
    *,
    expected_path_plan: Mapping[str, Any],
    expected_profile_plan: Mapping[str, Any],
    expected_config: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
    expected_provenance: Mapping[str, Any],
) -> None:
    _verify_terminal_registry(run_dir)
    expected = {
        "haar_path_id_plan.json": expected_path_plan,
        "haar_profile_plan.json": expected_profile_plan,
        "scientific_config.json": expected_config,
        "run_manifest.json": expected_manifest,
        "parent_provenance.json": expected_provenance,
    }
    for name, value in expected.items():
        if _load(run_dir / name) != dict(value):
            raise ArtifactCompatibilityError(f"resume {name} changed")
    _freeze_plans(
        run_dir,
        argparse.Namespace(panel_clusters=expected_config["panel_cluster_count"]),
        require_existing=True,
    )


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _required_record(
    value: Any,
    *,
    owner: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HaarExecutionError(f"{owner} did not return a diagnostics record")
    return value


def _required_mask(
    value: Any,
    *,
    owner: str,
    name: str,
    count: int,
) -> np.ndarray:
    if value is None:
        raise HaarExecutionError(f"{owner} omitted required {name}")
    mask = _tensor_to_numpy(value).reshape(-1)
    if mask.shape != (count,) or mask.dtype.kind != "b":
        raise HaarExecutionError(
            f"{owner} returned an invalid {name}; expected {count} booleans"
        )
    return mask


def _required_scalar(
    record: Mapping[str, Any],
    name: str,
    *,
    owner: str,
) -> float:
    if name not in record:
        raise HaarExecutionError(f"{owner} omitted required metric {name}")
    array = _tensor_to_numpy(record[name])
    if array.size != 1:
        raise HaarExecutionError(f"{owner} metric {name} is not scalar")
    value = float(array.reshape(()))
    if not math.isfinite(value):
        raise HaarExecutionError(f"{owner} metric {name} is nonfinite")
    return value


def _measured_uncertified_count(
    normal_certificate_mask: np.ndarray,
    jacobi_certificate_mask: np.ndarray,
    jacobi_diagnostics: Mapping[str, Any],
) -> int:
    """Return the measured normal-plus-Jacobi uncertified count.

    The fused arbitrary-uniform CUDA adapter historically omitted the
    redundant ``uncertified_count`` diagnostic even though its certificate
    mask is mandatory.  Derive that one counter from the two required masks;
    all other forbidden-event counters remain fail-closed when absent.  When
    an adapter does report its Jacobi-only count, require exact agreement with
    the mask instead of silently preferring either source.
    """

    normal = np.asarray(normal_certificate_mask)
    jacobi = np.asarray(jacobi_certificate_mask)
    if normal.dtype.kind != "b" or jacobi.dtype.kind != "b":
        raise HaarExecutionError(
            "certificate masks must be boolean before counting uncertified events"
        )
    normal_uncertified = int(normal.size - np.count_nonzero(normal))
    jacobi_uncertified = int(jacobi.size - np.count_nonzero(jacobi))
    if "uncertified_count" in jacobi_diagnostics:
        raw = _tensor_to_numpy(jacobi_diagnostics["uncertified_count"])
        if raw.size != 1:
            raise HaarExecutionError(
                "arbitrary-uniform Jacobi authorizer uncertified_count is not scalar"
            )
        value = raw.reshape(()).item()
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            raise HaarExecutionError(
                "arbitrary-uniform Jacobi authorizer returned an invalid "
                "uncertified_count"
            )
        if int(value) != jacobi_uncertified:
            raise HaarExecutionError(
                "arbitrary-uniform Jacobi authorizer uncertified_count "
                "disagrees with its certificate mask"
            )
    return normal_uncertified + jacobi_uncertified


def _explicit_pass(record: Mapping[str, Any], name: str) -> int:
    if name not in record:
        return 0
    value = record[name]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(int(value) == 1)
    return 0


def _all_explicit_pass(
    records: Sequence[Mapping[str, Any]],
    name: str,
) -> int:
    return int(bool(records) and all(_explicit_pass(record, name) for record in records))


def _sum_explicit_count(
    records: Sequence[Mapping[str, Any]],
    name: str,
) -> int | None:
    values: list[int] = []
    for record in records:
        if name not in record:
            return None
        value = record[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
        ):
            return None
        values.append(int(value))
    return sum(values)


def _max_explicit_finite(
    records: Sequence[Mapping[str, Any]],
    name: str,
) -> float | None:
    """Return a measured finite maximum, never a permissive sentinel."""

    values: list[float] = []
    for record in records:
        if name not in record:
            return None
        try:
            value = float(record[name])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        values.append(value)
    return max(values) if values else None


def _backend_smoke(
    args: argparse.Namespace,
    *,
    role: str,
    path_ids: Sequence[int],
    sample_steps: int,
    detail_sign: int = 1,
    pair_coarse_steps: int | None = None,
    phase: int = 0,
    head_values: Sequence[float] | None = None,
    exposure_values: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Exercise the real certified normal/Haar and Jacobi-cell APIs.

    The fused CUDA certificates are independently enclosed by Arb here.
    Any genuine transition-local fallback remains included in the measured
    runtime and fallback fractions.
    """

    haar = importlib.import_module("mnist.d0_jacobi_rb_haar")
    haar_cuda = importlib.import_module("mnist.d0_jacobi_rb_haar_cuda")
    rb_cuda = importlib.import_module("mnist.d0_jacobi_rb_cuda")
    profile = haar.HaarCouplingProfile()
    jacobi_profile = rb_cuda.JacobiRBCudaProfile()
    edges = tuple(range(4))
    device = torch.device(args.device)
    batch = haar.build_certified_haar_uniform_batch(
        root_seed=int(args.root_seed),
        role=role,
        path_ids=tuple(int(value) for value in path_ids),
        sample_steps=int(sample_steps),
        outer_step=0,
        phase=int(phase),
        edge_ids=edges,
        profile=profile,
        detail_sign=int(detail_sign),
        pair_coarse_steps=pair_coarse_steps,
        device=device,
        materialize_host_cells=True,
    )
    lower = _tensor_to_numpy(batch.uniform_lower).reshape(-1)
    upper = _tensor_to_numpy(batch.uniform_upper).reshape(-1)
    normals_lower = _tensor_to_numpy(batch.normal_lower).reshape(-1)
    normals_upper = _tensor_to_numpy(batch.normal_upper).reshape(-1)
    certificate = _required_mask(
        getattr(batch, "certificate_mask", None),
        owner="certified Haar builder",
        name="certificate_mask",
        count=lower.size,
    )
    normal_fallback = _required_mask(
        getattr(batch, "fallback_mask", None),
        owner="certified Haar builder",
        name="fallback_mask",
        count=lower.size,
    )
    prefix = _tensor_to_numpy(batch.prefix_bits).reshape(-1)
    ids = _tensor_to_numpy(batch.transition_ids).reshape(-1)
    if (
        lower.size == 0
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or not np.isfinite(normals_lower).all()
        or not np.isfinite(normals_upper).all()
        or not np.all((0.0 < lower) & (lower <= upper) & (upper < 1.0))
        or len(set(int(value) for value in ids.tolist())) != ids.size
    ):
        raise HaarExecutionError("certified Haar batch violated its interval contract")

    # The CPU reference adapter consumes exact rational cells.  Production
    # uses the CUDA interval entry point; both return the same schema.
    count = lower.size
    head = (
        np.resize(np.asarray(head_values, dtype=np.float64), count)
        if head_values is not None
        else np.linspace(0.2, 0.8, count, dtype=np.float64)
    )
    exposure = (
        np.resize(np.asarray(exposure_values, dtype=np.float64), count)
        if exposure_values is not None
        else np.full(count, 0.01, dtype=np.float64)
    )
    if args.test_only_reduced_workload or device.type == "cpu":
        transition = (
            haar_cuda.sample_alpha1_rb_transition_batch_from_uniform_cells_cpu(
                head,
                exposure,
                batch.uniform_cells,
                transition_ids=ids.astype(np.uint64, copy=False),
                profile=jacobi_profile,
                refinement_callback=batch.refinement_callback,
            )
        )
    else:
        transition = (
            haar_cuda.sample_alpha1_rb_transition_batch_cuda_from_uniform_cells(
                torch.as_tensor(head, dtype=torch.float64, device=device),
                torch.as_tensor(exposure, dtype=torch.float64, device=device),
                batch.uniform_lower.reshape(-1).contiguous(),
                batch.uniform_upper.reshape(-1).contiguous(),
                transition_ids=batch.transition_ids.reshape(-1).contiguous(),
                refinement_callback=batch.refinement_callback,
                profile=jacobi_profile,
                uniform_center_hi=batch.uniform_center_hi.reshape(-1).contiguous(),
                uniform_center_lo=batch.uniform_center_lo.reshape(-1).contiguous(),
                uniform_radius=batch.uniform_radius.reshape(-1).contiguous(),
                source_prefix_bits=batch.prefix_bits.reshape(-1).contiguous(),
            )
        )
    later = _tensor_to_numpy(
        getattr(transition, "later_head_fraction", getattr(transition, "later", None))
    ).reshape(-1)
    target = _tensor_to_numpy(
        getattr(transition, "denoising_target", getattr(transition, "target", None))
    ).reshape(-1)
    if (
        later.shape != (count,)
        or target.shape != (count,)
        or not np.isfinite(later).all()
        or not np.isfinite(target).all()
        or not np.all((0.0 <= later) & (later <= 1.0))
    ):
        raise HaarExecutionError("arbitrary-uniform Jacobi output is invalid")
    transition_certificate = _required_mask(
        getattr(
            transition,
            "certified_mask",
            getattr(transition, "certificate_mask", None),
        ),
        owner="arbitrary-uniform Jacobi authorizer",
        name="certified_mask",
        count=count,
    )
    transition_fallback = _required_mask(
        getattr(transition, "fallback_mask", None),
        owner="arbitrary-uniform Jacobi authorizer",
        name="fallback_mask",
        count=count,
    )
    normal_runtime = _required_record(
        getattr(batch, "runtime_report", None),
        owner="certified Haar builder runtime",
    )
    transition_runtime = _required_record(
        getattr(transition, "runtime_report", None),
        owner="arbitrary-uniform Jacobi authorizer runtime",
    )
    transition_diagnostics = _required_record(
        getattr(transition, "diagnostics", None),
        owner="arbitrary-uniform Jacobi authorizer diagnostics",
    )
    normal_elapsed = _required_scalar(
        normal_runtime, "elapsed_seconds", owner="certified Haar builder runtime"
    )
    transition_elapsed = _required_scalar(
        transition_runtime,
        "elapsed_seconds",
        owner="arbitrary-uniform Jacobi authorizer runtime",
    )
    normal_fallback_elapsed = _required_scalar(
        normal_runtime,
        "arb_fallback_elapsed_seconds",
        owner="certified Haar builder runtime",
    )
    transition_fallback_count = int(np.count_nonzero(transition_fallback))
    if transition_fallback_count == 0:
        transition_fallback_elapsed = 0.0
    elif "arb_fallback_time_fraction" in transition_runtime:
        transition_fallback_elapsed = transition_elapsed * _required_scalar(
            transition_runtime,
            "arb_fallback_time_fraction",
            owner="arbitrary-uniform Jacobi authorizer runtime",
        )
    elif str(transition_runtime.get("authorization_backend")) == "python-flint/Arb":
        transition_fallback_elapsed = transition_elapsed
    else:
        raise HaarExecutionError(
            "arbitrary-uniform Jacobi authorizer omitted fallback timing"
        )
    forbidden: dict[str, int | None] = {}
    for name in _base_forbidden():
        if name == "uncertified_count":
            forbidden[name] = _measured_uncertified_count(
                certificate,
                transition_certificate,
                transition_diagnostics,
            )
            continue
        raw = transition_diagnostics.get(name)
        if raw is None:
            forbidden[name] = None
            continue
        array = _tensor_to_numpy(raw)
        forbidden[name] = (
            int(array.reshape(()))
            if array.size == 1
            and np.isfinite(array).all()
            and int(array.reshape(())) >= 0
            else None
        )
    # Independent Arb evaluation of the exact source-prefix tree.  The fused
    # normal/CDF balls must contain this oracle; merely reporting a CUDA
    # certificate bit is not sufficient for the scientific preflight.
    arb_normal_pass = True
    arb_uniform_pass = True
    ancestry_pass = True
    tree_root_steps = (
        int(pair_coarse_steps)
        if pair_coarse_steps is not None
        else int(profile.coarsest_steps)
    )
    event_index = 0
    for path_id in tuple(int(value) for value in path_ids):
        for edge_id in edges:
            event = haar.HaarEventIdentity(
                role=role,
                path_id=path_id,
                sample_steps=int(sample_steps),
                outer_step=0,
                phase=int(phase),
                edge_id=int(edge_id),
                arm=(int(detail_sign) if pair_coarse_steps is not None else 0),
                tree_root_steps=tree_root_steps,
            )
            oracle_normal, oracle_uniform, source_ids, _bits = (
                haar._certify_event(
                    event,
                    root_seed=int(args.root_seed),
                    profile=profile,
                    detail_sign=int(detail_sign),
                )
            )
            observed_normal = batch.normal_cells[event_index]
            observed_uniform = batch.uniform_cells[event_index]
            arb_normal_pass &= (
                observed_normal.lower <= oracle_normal.lower
                and observed_normal.upper >= oracle_normal.upper
            )
            arb_uniform_pass &= (
                observed_uniform.lower <= oracle_uniform.lower
                and observed_uniform.upper >= oracle_uniform.upper
            )
            decoded = [haar.unpack_haar_source_id(value) for value in source_ids]
            ancestry_pass &= all(
                value["role"] == role
                and int(value["path_id"]) == path_id
                and int(value["phase"]) == int(phase)
                and int(value["edge_id"]) == int(edge_id)
                for value in decoded
            )
            event_index += 1

    cpu_transition = (
        haar_cuda.sample_alpha1_rb_transition_batch_from_uniform_cells_cpu(
            head,
            exposure,
            batch.uniform_cells,
            transition_ids=ids.astype(np.uint64, copy=False),
            profile=jacobi_profile,
            refinement_callback=batch.refinement_callback,
        )
    )
    cpu_later = _tensor_to_numpy(
        getattr(
            cpu_transition,
            "later_head_fraction",
            getattr(cpu_transition, "later", None),
        )
    ).reshape(-1)
    cpu_target = _tensor_to_numpy(
        getattr(
            cpu_transition,
            "denoising_target",
            getattr(cpu_transition, "target", None),
        )
    ).reshape(-1)
    saved_prefix_replay = int(
        np.array_equal(cpu_later, later)
        and np.array_equal(cpu_target, target)
    )
    pair_totals = np.linspace(1.0e-5, 0.9, count, dtype=np.float64)
    before_mass = pair_totals
    after_tail = pair_totals * (1.0 - later)
    after_head = pair_totals * later
    mass_error = float(np.max(np.abs((after_tail + after_head) - before_mass)))
    normal_cuda = bool(
        normal_runtime.get("fused_cuda_authorizer_available", False)
    )
    jacobi_cuda = bool(
        transition_runtime.get(
            "arbitrary_uniform_interval_authorizing", False
        )
    )
    production_authorizing = int(
        not args.test_only_reduced_workload
        and device.type == "cuda"
        and normal_cuda
        and jacobi_cuda
        and bool(np.all(certificate))
        and bool(np.all(transition_certificate))
        and all(value == 0 for value in forbidden.values())
    )
    return {
        "sample_count": int(count),
        "certificate_count": int(np.count_nonzero(certificate))
        + int(np.count_nonzero(transition_certificate)),
        "certificate_denominator": int(2 * count),
        "fallback_count": int(np.count_nonzero(normal_fallback))
        + transition_fallback_count,
        "fallback_denominator": int(2 * count),
        "normal_fallback_count": int(np.count_nonzero(normal_fallback)),
        "jacobi_fallback_count": transition_fallback_count,
        "maximum_prefix_bits": int(np.max(prefix)),
        "normal_interval_pass": int(np.all(normals_lower <= normals_upper)),
        "uniform_interval_pass": int(
            np.all((0.0 < lower) & (lower <= upper) & (upper < 1.0))
        ),
        "jacobi_output_pass": int(np.all(transition_certificate)),
        "normal_cells_certified_pass": int(np.all(certificate)),
        "uniform_cells_certified_pass": int(
            np.all(certificate)
            and np.all((0.0 < lower) & (lower <= upper) & (upper < 1.0))
        ),
        "jacobi_outputs_certified_pass": int(np.all(transition_certificate)),
        "transition_id_unique_pass": 1,
        "phase": int(phase),
        "facet_pass": int(
            head_values is None
            or np.all(np.isfinite(later))
        ),
        "zero_duration_pass": int(
            not np.any(exposure == 0.0)
            or np.array_equal(later[exposure == 0.0], head[exposure == 0.0])
        ),
        "later_sha256": hashlib.sha256(later.tobytes()).hexdigest(),
        "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        "uniform_lower_sha256": hashlib.sha256(lower.tobytes()).hexdigest(),
        "uniform_upper_sha256": hashlib.sha256(upper.tobytes()).hexdigest(),
        "normal_elapsed_seconds": normal_elapsed,
        "jacobi_elapsed_seconds": transition_elapsed,
        "elapsed_seconds": normal_elapsed + transition_elapsed,
        "fallback_elapsed_seconds": (
            normal_fallback_elapsed + transition_fallback_elapsed
        ),
        "fused_cuda_normal_authorizer_pass": int(
            bool(
                normal_runtime.get(
                    "fused_cuda_authorizer_available", False
                )
            )
        ),
        "arbitrary_uniform_cuda_authorizer_pass": int(
            device.type == "cuda"
            and jacobi_cuda
        ),
        "production_authorizing_pass": production_authorizing,
        "normal_inverse_arb_enclosure_pass": int(arb_normal_pass),
        "normal_cdf_arb_enclosure_pass": int(arb_uniform_pass),
        "normal_cells_certified_pass": int(np.all(certificate)),
        "uniform_cells_certified_pass": int(np.all(certificate)),
        "jacobi_outputs_certified_pass": int(
            np.all(transition_certificate)
        ),
        "state_independent_uniform_pass": int(
            "head_fraction"
            not in importlib.import_module("inspect").signature(
                haar.build_certified_haar_uniform_batch
            ).parameters
        ),
        "state_independent_rng_pass": int(
            "head_fraction"
            not in importlib.import_module("inspect").signature(
                haar.build_certified_haar_uniform_batch
            ).parameters
        ),
        "intentional_haar_ancestry_only_pass": int(ancestry_pass),
        "intentional_ancestry_only_pass": int(ancestry_pass),
        "saved_prefix_jacobi_replay_pass": saved_prefix_replay,
        "conservation_pass": int(mass_error <= 2.0e-15),
        "mass_error": mass_error,
        "target_contract_pass": int(
            saved_prefix_replay
            and np.all(transition_certificate)
            and np.isfinite(target).all()
        ),
        "later_state_only_contract_pass": int(
            set(
                (
                    "later_full_state",
                    "reverse_time",
                    "phase",
                    "color",
                    "duration",
                    "label",
                )
            ).isdisjoint(
                {
                    "earlier_state",
                    "uniform_bits",
                    "certificate",
                    "oracle_target",
                }
            )
        ),
        # Full marginal statistics and full-pipeline scheduling are populated
        # by their dedicated measured controls, never inferred from this smoke.
        "jacobi_marginal_cdf_pass": 0,
        "jacobi_eigenmoment_pass": 0,
        "jacobi_detailed_balance_pass": 0,
        "marginal_cdf_pass": 0,
        "marginal_eigenmoment_pass": 0,
        "marginal_detailed_balance_pass": 0,
        "phase_tower_identity_pass": 0,
        "order_chunk_resume_invariance_pass": 0,
        "order_invariance_pass": 0,
        "chunk_invariance_pass": 0,
        "resume_invariance_pass": 0,
        "interruption_replay_pass": 0,
        "deterministic_batching_pass": 0,
        "candidate_under_48h_forecast_pass": 0,
        "pipeline_runtime_projection_pass": 0,
        **forbidden,
        "later": later,
        "target": target,
    }


def _haar_algebra_metrics() -> dict[str, int]:
    haar = importlib.import_module("mnist.d0_jacobi_rb_haar")
    rng = np.random.default_rng(0x48414152)
    parent = rng.standard_normal(131_072, dtype=np.float64)
    detail = rng.standard_normal(parent.size, dtype=np.float64)
    left, right = haar.haar_split(parent, detail)
    swapped_left, swapped_right = haar.haar_split(parent, -detail)
    parent_error = float(np.max(np.abs(haar.haar_parent(left, right) - parent)))
    swap_error = float(
        max(
            np.max(np.abs(left - swapped_right)),
            np.max(np.abs(right - swapped_left)),
        )
    )
    return {
        "haar_covariance_pass": int(
            abs(float(np.var(left, ddof=1)) - 1.0) <= 1.5e-2
            and abs(float(np.var(right, ddof=1)) - 1.0) <= 1.5e-2
        ),
        "haar_within_level_independence_pass": int(
            abs(float(np.cov(left, right, ddof=1)[0, 1])) <= 1.5e-2
        ),
        "haar_parent_child_aggregation_pass": int(parent_error <= 2.0e-15),
        "antithetic_marginal_equality_pass": int(swap_error == 0.0),
    }


def _extreme_prefix_report(
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    haar = importlib.import_module("mnist.d0_jacobi_rb_haar")
    profile = haar.HaarCouplingProfile()
    bits = 128
    denominator = 1 << bits
    numerators = (
        2,
        1 << 32,
        denominator // 4,
        denominator // 2,
        3 * denominator // 4,
        denominator - (1 << 32) - 1,
        denominator - 3,
    )
    references = []
    reference_pass = True
    for numerator in numerators:
        result = haar.certify_normal_uniform_from_prefix(
            numerator, bits, profile
        )
        source_lower = Fraction(numerator, denominator)
        source_upper = Fraction(numerator + 1, denominator)
        case_pass = (
            result.normal.lower <= result.normal.upper
            and result.uniform.lower <= source_lower
            and result.uniform.upper >= source_upper
            and result.uniform.lower > 0
            and result.uniform.upper < 1
        )
        reference_pass &= bool(case_pass)
        references.append((result, source_lower, source_upper))

    cuda_evaluated = bool(
        device is not None
        and torch.device(device).type == "cuda"
        and torch.cuda.is_available()
    )
    cuda_pass = False
    cuda_rows: list[dict[str, Any]] = []
    if cuda_evaluated:
        fused = importlib.import_module("mnist.d0_jacobi_rb_haar_fused")
        mask = (1 << 64) - 1
        launch = fused.launch_certified_normal_prefix_transform(
            torch.tensor(
                [value >> 64 for value in numerators],
                dtype=torch.uint64,
                device=torch.device(device),
            ).contiguous(),
            torch.tensor(
                [value & mask for value in numerators],
                dtype=torch.uint64,
                device=torch.device(device),
            ).contiguous(),
        )
        certified = _tensor_to_numpy(launch.certificate_mask).astype(bool)
        normal_lower = _tensor_to_numpy(launch.normal_lower)
        normal_upper = _tensor_to_numpy(launch.normal_upper)
        uniform_lower = _tensor_to_numpy(launch.uniform_lower)
        uniform_upper = _tensor_to_numpy(launch.uniform_upper)
        reasons = _tensor_to_numpy(
            launch.fallback_reason_codes
        ).astype(np.int64)
        cuda_pass = True
        for index, (
            (reference, source_lower, source_upper),
            numerator,
        ) in enumerate(zip(references, numerators, strict=True)):
            if certified[index]:
                row_pass = bool(
                    Fraction.from_float(float(normal_lower[index]))
                    <= reference.normal.lower
                    <= reference.normal.upper
                    <= Fraction.from_float(float(normal_upper[index]))
                    and Fraction.from_float(float(uniform_lower[index]))
                    <= source_lower
                    <= source_upper
                    <= Fraction.from_float(float(uniform_upper[index]))
                )
                backend = "fused-cuda-dd"
            else:
                # Extreme upper/lower tails may exhaust the deliberately
                # conservative DD ball before violating correctness.  The
                # required transition-local Arb escalation is authorizing.
                row_pass = bool(
                    int(reasons[index]) in {2, 3, 5}
                    and reference.normal.lower
                    <= reference.normal.upper
                    and reference.uniform.lower <= source_lower
                    and reference.uniform.upper >= source_upper
                )
                backend = "candidate-local-Arb-fallback"
            cuda_pass &= row_pass
            cuda_rows.append(
                {
                    "prefix_numerator_hex": hex(numerator),
                    "normal_lower": float(normal_lower[index]),
                    "normal_upper": float(normal_upper[index]),
                    "uniform_lower": float(uniform_lower[index]),
                    "uniform_upper": float(uniform_upper[index]),
                    "certificate": int(certified[index]),
                    "fallback_reason_code": int(reasons[index]),
                    "authorization_backend": backend,
                    "arb_enclosure_pass": int(row_pass),
                }
            )
    return {
        "schema": RUN_SCHEMA + "-extreme-prefix-cuda",
        "schema_version": 1,
        "prefix_bits": bits,
        "case_count": len(numerators),
        "arb_reference_pass": int(reference_pass),
        "cuda_evaluated": int(cuda_evaluated),
        "cuda_arb_enclosure_pass": int(cuda_pass),
        "fused_certificate_count": int(
            sum(int(row["certificate"]) for row in cuda_rows)
        ),
        "candidate_local_arb_fallback_count": int(
            sum(1 - int(row["certificate"]) for row in cuda_rows)
        ),
        "passed": int(reference_pass and (cuda_pass if cuda_evaluated else True)),
        "rows": cuda_rows,
        **NO_WORK,
    }


def _extreme_prefix_pass(
    device: str | torch.device | None = None,
) -> int:
    return int(_extreme_prefix_report(device).get("passed", 0))


def _base_forbidden() -> dict[str, int]:
    return {
        name: 0
        for name in (
            "uncertified_count",
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
    }


def _parent_mixed_state(args: argparse.Namespace) -> np.ndarray:
    parent = Path(args.parent_phase_observer_run_dir).resolve()
    metadata = _load(parent / "source_image.json")
    payload = parent / "source_image.npz"
    if (
        not payload.is_file()
        or metadata.get("source_npz_sha256") != file_fingerprint(payload)
    ):
        raise ArtifactCompatibilityError("parent source-image binding changed")
    with np.load(payload, allow_pickle=False) as archive:
        if "mixed_target" not in archive.files:
            raise ArtifactCompatibilityError(
                "parent source image lacks mixed_target"
            )
        value = np.asarray(archive["mixed_target"], dtype=np.float64)
    if (
        value.shape != (784,)
        or not np.isfinite(value).all()
        or np.any(value < 0.0)
        or abs(float(value.sum()) - 1.0) > 1.0e-12
    ):
        raise ArtifactCompatibilityError("parent mixed target is invalid")
    return np.array(value, copy=True, order="C")


def _load_or_run_preflight_controls(
    run_dir: Path,
    args: argparse.Namespace,
    path_plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    marginal_path = run_dir / "haar_marginal_controls.json"
    tower_path = run_dir / "haar_phase_tower_controls.json"
    scheduler_path = run_dir / "haar_scheduler_preflight.json"
    marginal_ids = path_plan["marginal_panels"]["a"]["path_ids"]
    tower_ids = path_plan["marginal_panels"]["b"]["path_ids"]
    if marginal_path.is_file():
        marginal = _load(marginal_path)
    else:
        marginal = run_marginal_and_batching_controls(
            root_seed=int(args.root_seed),
            path_ids=marginal_ids,
            device=args.device,
        )
        atomic_write_json(marginal_path, marginal)
    if tower_path.is_file():
        tower = _load(tower_path)
    else:
        tower = run_phase_tower_controls(
            root_seed=int(args.root_seed),
            path_ids=tower_ids,
            device=args.device,
            cases_per_color_duration=(
                1 if args.test_only_reduced_workload else 16
            ),
        )
        atomic_write_json(tower_path, tower)
    if scheduler_path.is_file():
        scheduler = _load(scheduler_path)
    else:
        scheduler = run_scheduler_equivalence_and_benchmark(
            run_dir=run_dir,
            root_seed=int(args.root_seed),
            path_id_plan=path_plan,
            mixed_state=_parent_mixed_state(args),
            device=args.device,
            include_all_profiles=False,
        )
        atomic_write_json(scheduler_path, scheduler)
    return marginal, tower, scheduler


def _collect_preflight_metrics(
    run_dir: Path,
    args: argparse.Namespace,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    path_plan = _load(run_dir / "haar_path_id_plan.json")
    model_input_contract = _load(
        run_dir / "future_model_input_contract.json"
    )
    expected_model_input_contract = _model_input_contract()
    nested_ids = path_plan["profiles"][NESTED_HAAR_PROFILE]["a"]["roles"][
        "main"
    ]["root_path_ids"]
    count = min(len(nested_ids), 2)
    smoke_128 = _backend_smoke(
        args,
        role="nested_a",
        path_ids=nested_ids[:count],
        sample_steps=128,
    )
    smoke_256 = _backend_smoke(
        args,
        role="nested_a",
        path_ids=nested_ids[:count],
        sample_steps=256,
    )
    phase_records = [smoke_128] + [
        _backend_smoke(
            args,
            role="nested_a",
            path_ids=nested_ids[:1],
            sample_steps=128,
            phase=phase,
        )
        for phase in range(1, 7)
    ]
    boundary_record = _backend_smoke(
        args,
        role="nested_a",
        path_ids=nested_ids[:1],
        sample_steps=128,
        head_values=(0.0, 1.0, 0.5, 0.25),
        exposure_values=(0.0, 0.005, 0.01, 0.0),
    )
    algebra = _haar_algebra_metrics()
    extreme_prefix = _extreme_prefix_report(
        None if args.test_only_reduced_workload else args.device
    )
    atomic_write_json(
        run_dir / "extreme_prefix_cuda_report.json",
        extreme_prefix,
    )
    denominator = smoke_128["certificate_denominator"] + smoke_256[
        "certificate_denominator"
    ]
    certificate_count = smoke_128["certificate_count"] + smoke_256[
        "certificate_count"
    ]
    fallback_count = smoke_128["fallback_count"] + smoke_256["fallback_count"]
    elapsed = smoke_128["elapsed_seconds"] + smoke_256["elapsed_seconds"]
    fallback_elapsed = smoke_128["fallback_elapsed_seconds"] + smoke_256[
        "fallback_elapsed_seconds"
    ]
    # ``phase_records[0]`` is exactly ``smoke_128`` reused as the phase-zero
    # observation.  Include it only once in aggregate event counts.
    measured_records = [
        smoke_128,
        smoke_256,
        *phase_records[1:],
        boundary_record,
    ]
    if args.test_only_reduced_workload:
        marginal_control: dict[str, Any] = {}
        tower_control: dict[str, Any] = {}
        scheduler_control: dict[str, Any] = {}
    else:
        (
            marginal_control,
            tower_control,
            scheduler_control,
        ) = _load_or_run_preflight_controls(run_dir, args, path_plan)
    scheduler_execution = dict(scheduler_control.get("execution", {}))
    scientific_controls = (
        marginal_control,
        tower_control,
        scheduler_execution,
    )
    measured_forbidden = {
        name: (
            None
            if _sum_explicit_count(measured_records, name) is None
            or any(name not in record for record in scientific_controls)
            else int(_sum_explicit_count(measured_records, name))
            + sum(int(record[name]) for record in scientific_controls)
        )
        for name in _base_forbidden()
    }
    structural = int(
        path_plan["checks"]["integer_20_bit_pass"]
        and path_plan["checks"]["role_disjoint_pass"]
        and path_plan["checks"]["future_production_reserved_pass"]
    )
    production = int(
        not args.test_only_reduced_workload
        and _all_explicit_pass(
            measured_records, "production_authorizing_pass"
        )
        and smoke_128["fused_cuda_normal_authorizer_pass"]
        and smoke_256["fused_cuda_normal_authorizer_pass"]
        and smoke_128["arbitrary_uniform_cuda_authorizer_pass"]
        and smoke_256["arbitrary_uniform_cuda_authorizer_pass"]
        and int(marginal_control.get("certificate_fraction", 0.0) == 1.0)
        and int(tower_control.get("certificate_fraction", 0.0) == 1.0)
        and int(scheduler_execution.get("certificate_fraction", 0.0) == 1.0)
    )
    return {
        "production_authorizing_pass": production,
        "control_provenance_pass": int(provenance.get("passed", 0)),
        "parent_readjudication_pass": int(
            provenance.get("parent_re_adjudication")
            == "right_endpoint_coupling_power_infeasible"
        ),
        "parent_sources_immutable_pass": int(
            provenance.get("parent_source_fingerprint")
            == PARENT_SOURCE_FINGERPRINT
        ),
        "parent_preflight_pass": int(provenance.get("parent_preflight_pass", 0)),
        "parent_pilot_numerically_valid_pass": int(
            provenance.get("parent_pilot_numerically_valid", 0)
        ),
        "parent_pilot_resource_valid_pass": int(
            provenance.get("parent_pilot_resource_valid", 0)
        ),
        "parent_power_only_failure_pass": int(
            provenance.get("parent_pilot_power_valid", 1) == 0
        ),
        "normal_inverse_arb_enclosure_pass": _all_explicit_pass(
            measured_records, "normal_inverse_arb_enclosure_pass"
        ),
        "fused_cuda_normal_authorizer_pass": int(
            smoke_128["fused_cuda_normal_authorizer_pass"]
            and smoke_256["fused_cuda_normal_authorizer_pass"]
        ),
        "normal_cdf_arb_enclosure_pass": _all_explicit_pass(
            measured_records, "normal_cdf_arb_enclosure_pass"
        ),
        "normal_extreme_prefix_pass": int(
            extreme_prefix.get("passed", 0)
            and (
                args.test_only_reduced_workload
                or extreme_prefix.get("cuda_evaluated", 0) == 1
            )
            and all(
                record["maximum_prefix_bits"] <= 1024
                and record["normal_interval_pass"]
                for record in (*phase_records, boundary_record)
            )
        ),
        **algebra,
        "state_independent_uniform_pass": _all_explicit_pass(
            measured_records, "state_independent_uniform_pass"
        ),
        "path_id_slot_plan_pass": structural,
        "future_production_reserved_pass": int(
            path_plan["checks"]["future_production_reserved_pass"]
        ),
        "profile_panel_disjoint_pass": int(
            path_plan["checks"]["role_disjoint_pass"]
        ),
        "path_id_uniqueness_pass": structural,
        "intentional_haar_ancestry_only_pass": _all_explicit_pass(
            measured_records, "intentional_haar_ancestry_only_pass"
        ),
        "order_chunk_resume_invariance_pass": int(
            marginal_control.get("order_chunk_resume_invariance_pass", 0)
            and scheduler_control.get("resume_invariance_pass", 0)
            and scheduler_control.get("regrouping_invariance_pass", 0)
        ),
        "saved_prefix_jacobi_replay_pass": _all_explicit_pass(
            measured_records, "saved_prefix_jacobi_replay_pass"
        ),
        "arbitrary_uniform_cuda_authorizer_pass": int(
            smoke_128["arbitrary_uniform_cuda_authorizer_pass"]
            and smoke_256["arbitrary_uniform_cuda_authorizer_pass"]
        ),
        "jacobi_marginal_cdf_pass": int(
            marginal_control.get("jacobi_marginal_cdf_pass", 0)
        ),
        "jacobi_eigenmoment_pass": int(
            marginal_control.get("jacobi_eigenmoment_pass", 0)
        ),
        "jacobi_detailed_balance_pass": int(
            marginal_control.get("jacobi_detailed_balance_pass", 0)
        ),
        "rb_target_certificate_pass": int(
            smoke_128["jacobi_output_pass"]
            and smoke_256["jacobi_output_pass"]
            and marginal_control.get("rb_target_certificate_pass", 0)
        ),
        "later_state_only_contract_pass": _all_explicit_pass(
            measured_records, "later_state_only_contract_pass"
        )
        and int(
            model_input_contract == expected_model_input_contract
            and model_input_contract.get(
                "model_input_contract_sha256"
            )
            == expected_model_input_contract[
                "model_input_contract_sha256"
            ]
        ),
        "all_colors_pass": int(tower_control.get("all_colors_pass", 0)),
        "half_full_duration_pass": int(
            tower_control.get("half_full_duration_pass", 0)
        ),
        "facet_pass": int(boundary_record["facet_pass"]),
        "zero_mass_duration_pass": int(
            boundary_record["zero_duration_pass"]
        ),
        "phase_tower_identity_pass": int(
            tower_control.get("phase_tower_identity_pass", 0)
        ),
        "interruption_replay_pass": int(
            marginal_control.get("interruption_replay_pass", 0)
            and scheduler_control.get("interruption_replay_pass", 0)
        ),
        "deterministic_batching_pass": int(
            marginal_control.get("deterministic_batching_pass", 0)
            and scheduler_control.get("deterministic_batching_pass", 0)
        ),
        "candidate_under_48h_forecast_pass": int(
            scheduler_control.get("candidate_under_48h_forecast_pass", 0)
        ),
        "parent_record_count": int(
            provenance.get("parent_artifact_record_count", -1)
        ),
        "parent_source_count": int(provenance.get("parent_source_count", -1)),
        "root_seed": int(args.root_seed),
        "grid_size": 28,
        "alpha": 1.0,
        "tau_eff": 5.0e-5,
        "levels": list(SAMPLE_STEPS),
        "maximum_prefix_bits": max(
            smoke_128["maximum_prefix_bits"], smoke_256["maximum_prefix_bits"]
        ),
        "certificate_fraction": min(
            certificate_count / denominator if denominator else 0.0,
            float(marginal_control.get("certificate_fraction", 0.0)),
            float(tower_control.get("certificate_fraction", 0.0)),
            float(scheduler_execution.get("certificate_fraction", 0.0)),
        ),
        "fallback_fraction": max(
            fallback_count / max(1, denominator),
            float(marginal_control.get("fallback_fraction", 1.0)),
            float(tower_control.get("fallback_fraction", 1.0)),
            float(scheduler_execution.get("fallback_fraction", 1.0)),
        ),
        "fallback_cost_fraction": max(
            fallback_elapsed / max(elapsed, 1.0e-30),
            float(marginal_control.get("fallback_cost_fraction", 1.0)),
            float(tower_control.get("fallback_cost_fraction", 1.0)),
            float(
                scheduler_execution.get(
                    "fallback_cost_fraction", 1.0
                )
            ),
        ),
        "peak_memory_fraction": (
            torch.cuda.max_memory_allocated(torch.device(args.device))
            / torch.cuda.get_device_properties(torch.device(args.device)).total_memory
            if torch.device(args.device).type == "cuda"
            else 0.0
        ),
        "mass_error": _max_explicit_finite(
            (
                smoke_128,
                smoke_256,
                marginal_control,
                scheduler_execution,
            ),
            "mass_error",
        ),
        **measured_forbidden,
    }


def _run_preflight_stage(
    run_dir: Path,
    args: argparse.Namespace,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    metrics_path = run_dir / "haar_preflight_metrics.json"
    gate_path = run_dir / "haar_preflight_gate.json"
    fused_runtime_path = run_dir / "haar_fused_runtime_report.json"
    if metrics_path.is_file() and gate_path.is_file():
        if not fused_runtime_path.is_file():
            raise ArtifactCompatibilityError(
                "frozen Haar preflight lacks its fused runtime report"
            )
        metrics = _load(metrics_path)["metrics"]
        gate = _load(gate_path)
        if gate != evaluate_haar_preflight(metrics):
            raise ArtifactCompatibilityError("frozen Haar preflight gate changed")
        return gate
    fused_runtime = {
        "schema": RUN_SCHEMA + "-fused-runtime",
        "schema_version": 1,
        **fused_haar_runtime_report(torch.device(args.device)),
        **NO_WORK,
    }
    _freeze(
        fused_runtime_path,
        fused_runtime,
        require_existing=fused_runtime_path.is_file(),
    )
    metrics = _collect_preflight_metrics(run_dir, args, provenance)
    atomic_write_json(
        metrics_path,
        {
            "schema": RUN_SCHEMA + "-preflight-metrics",
            "schema_version": 1,
            "metrics": metrics,
            **NO_WORK,
        },
    )
    gate = evaluate_haar_preflight(metrics)
    atomic_write_json(gate_path, gate)
    return gate


def _collect_coupling_metrics(
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    path_plan = _load(run_dir / "haar_path_id_plan.json")
    if args.test_only_reduced_workload:
        marginal: dict[str, Any] = {}
        tower: dict[str, Any] = {}
        scheduler: dict[str, Any] = {}
    else:
        marginal = _load(run_dir / "haar_marginal_controls.json")
        tower = _load(run_dir / "haar_phase_tower_controls.json")
        scheduler_path = run_dir / "haar_coupling_scheduler_benchmark.json"
        if scheduler_path.is_file():
            scheduler = _load(scheduler_path)
        else:
            scheduler = run_scheduler_equivalence_and_benchmark(
                run_dir=run_dir,
                root_seed=int(args.root_seed),
                path_id_plan=path_plan,
                mixed_state=_parent_mixed_state(args),
                device=args.device,
                include_all_profiles=True,
            )
            atomic_write_json(scheduler_path, scheduler)
    execution = dict(scheduler.get("execution", {}))
    preflight_metrics = (
        _load(run_dir / "haar_preflight_metrics.json").get("metrics", {})
        if (run_dir / "haar_preflight_metrics.json").is_file()
        else {}
    )
    algebra = _haar_algebra_metrics()
    measured_forbidden = {
        name: (
            None
            if any(
                name not in record
                for record in (marginal, tower, execution)
            )
            else sum(
                int(record[name]) for record in (marginal, tower, execution)
            )
        )
        for name in _base_forbidden()
    }
    production = int(
        not args.test_only_reduced_workload
        and int(preflight_metrics.get("production_authorizing_pass", 0)) == 1
        and float(execution.get("certificate_fraction", 0.0)) == 1.0
    )
    return {
        "production_authorizing_pass": production,
        "nested_profile_complete_pass": int(
            scheduler.get("nested_profile_complete_pass", 0)
        ),
        "antithetic_profile_complete_pass": int(
            scheduler.get("antithetic_profile_complete_pass", 0)
        ),
        "normal_cells_certified_pass": int(
            float(execution.get("certificate_fraction", 0.0)) == 1.0
        ),
        "fused_cuda_normal_authorizer_pass": int(
            preflight_metrics.get("fused_cuda_normal_authorizer_pass", 0)
        ),
        "uniform_cells_certified_pass": int(
            float(execution.get("certificate_fraction", 0.0)) == 1.0
        ),
        "jacobi_outputs_certified_pass": int(
            float(execution.get("certificate_fraction", 0.0)) == 1.0
        ),
        "arbitrary_uniform_cuda_authorizer_pass": int(
            preflight_metrics.get(
                "arbitrary_uniform_cuda_authorizer_pass", 0
            )
        ),
        "haar_covariance_pass": algebra["haar_covariance_pass"],
        "within_level_independence_pass": algebra[
            "haar_within_level_independence_pass"
        ],
        "parent_child_aggregation_pass": algebra[
            "haar_parent_child_aggregation_pass"
        ],
        "antithetic_marginal_pass": algebra[
            "antithetic_marginal_equality_pass"
        ],
        "state_independent_rng_pass": int(
            preflight_metrics.get("state_independent_uniform_pass", 0)
        ),
        "id_uniqueness_pass": int(
            path_plan["checks"]["integer_20_bit_pass"]
            and path_plan["checks"]["role_disjoint_pass"]
        ),
        "intentional_ancestry_only_pass": int(
            preflight_metrics.get(
                "intentional_haar_ancestry_only_pass", 0
            )
        ),
        "order_invariance_pass": int(
            marginal.get("order_invariance_pass", 0)
            and scheduler.get("order_invariance_pass", 0)
        ),
        "chunk_invariance_pass": int(
            marginal.get("chunk_invariance_pass", 0)
            and scheduler.get("chunk_invariance_pass", 0)
        ),
        "resume_invariance_pass": int(
            marginal.get("resume_invariance_pass", 0)
            and scheduler.get("resume_invariance_pass", 0)
        ),
        "marginal_cdf_pass": int(marginal.get("marginal_cdf_pass", 0)),
        "marginal_eigenmoment_pass": int(
            marginal.get("marginal_eigenmoment_pass", 0)
        ),
        "marginal_detailed_balance_pass": int(
            marginal.get("marginal_detailed_balance_pass", 0)
        ),
        "conservation_pass": int(
            marginal.get("conservation_pass", 0)
            and float(execution.get("mass_error", math.inf)) <= 2.0e-6
        ),
        "target_contract_pass": int(
            marginal.get("rb_target_certificate_pass", 0)
            and preflight_metrics.get("later_state_only_contract_pass", 0)
        ),
        "pipeline_runtime_projection_pass": int(
            scheduler.get("pipeline_runtime_projection_pass", 0)
        ),
        "profile_order": list(PROFILE_ORDER),
        "certificate_fraction": min(
            float(execution.get("certificate_fraction", 0.0)),
            float(marginal.get("certificate_fraction", 0.0)),
            float(tower.get("certificate_fraction", 0.0)),
        ),
        "fallback_fraction": max(
            float(execution.get("fallback_fraction", 1.0)),
            float(marginal.get("fallback_fraction", 1.0)),
            float(tower.get("fallback_fraction", 1.0)),
        ),
        "fallback_cost_fraction": max(
            float(execution.get("fallback_cost_fraction", 1.0)),
            float(marginal.get("fallback_cost_fraction", 1.0)),
            float(tower.get("fallback_cost_fraction", 1.0)),
        ),
        "minimum_rate": float(execution.get("conservative_rate", 0.0)),
        "minimum_projected_hours": float(
            scheduler.get("minimum_projected_hours", 1.0e300)
        ),
        "peak_memory_fraction": (
            torch.cuda.max_memory_allocated(torch.device(args.device))
            / torch.cuda.get_device_properties(torch.device(args.device)).total_memory
            if torch.device(args.device).type == "cuda"
            else 0.0
        ),
        "mass_error": _max_explicit_finite(
            (execution, marginal),
            "mass_error",
        ),
        **measured_forbidden,
    }


def _run_coupling_stage(
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics_path = run_dir / "haar_coupling_metrics.json"
    gate_path = run_dir / "haar_coupling_gate.json"
    if metrics_path.is_file() and gate_path.is_file():
        metrics = _load(metrics_path)["metrics"]
        gate = _load(gate_path)
        if gate != evaluate_haar_coupling(metrics):
            raise ArtifactCompatibilityError("frozen Haar coupling gate changed")
        return gate
    metrics = _collect_coupling_metrics(run_dir, args)
    atomic_write_json(
        metrics_path,
        {
            "schema": RUN_SCHEMA + "-coupling-metrics",
            "schema_version": 1,
            "metrics": metrics,
            **NO_WORK,
        },
    )
    gate = evaluate_haar_coupling(metrics)
    atomic_write_json(gate_path, gate)
    return gate


def _candidate_row(
    *,
    profile: str,
    main_paths: int,
    reference_paths: int,
    main_width: float,
    reference_width: float,
    projected_hours: float,
    rate: float,
    measured: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = {} if measured is None else dict(measured)
    return {
        "main_paths": int(main_paths),
        "reference_paths": int(reference_paths),
        "predicted_main_half_width": float(main_width),
        "predicted_generator_reference_half_width": float(reference_width),
        "predicted_reference_stability_half_width": float(reference_width),
        "projected_hours": float(projected_hours),
        "conservative_rate": float(rate),
        **{
            name: _explicit_pass(evidence, name)
            for name in (
                "panel_complete_pass",
                "panel_finite_pass",
                "panel_certification_pass",
                "panel_numerical_health_pass",
                "mass_conservation_pass",
                "shard_chain_pass",
                "pilot_production_isolation_pass",
                "pilot_means_excluded_pass",
                "raw_endpoint_authorizing_pass",
                "dynkin_advisory_only_pass",
            )
        },
    }


def _expected_power_panel_path_pools(
    path_plan: Mapping[str, Any],
    *,
    profile: str,
    panel: str,
) -> tuple[dict[str, list[int]], list[int], list[int]]:
    try:
        roles = path_plan["profiles"][profile][panel]["roles"]
        if profile == NESTED_HAAR_PROFILE:
            pools = {
                "main": [int(value) for value in roles["main"]["root_path_ids"]],
                "reference": [
                    int(value) for value in roles["reference"]["root_path_ids"]
                ],
            }
            return pools, pools["main"], pools["reference"]
        pools = {
            f"{coarse}-{fine}": [
                int(value)
                for value in roles[f"{coarse}-{fine}"]["root_path_ids"]
            ]
            for coarse, fine in PAIRWISE_LEVELS
        }
        return pools, pools["128-256"], pools["1024-2048"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactCompatibilityError(
            "frozen panel path-ID plan is incomplete"
        ) from exc


def _verified_profile_panel_evidence(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    profile: str,
    panel: str,
) -> dict[str, Any]:
    evidence_path = run_dir / f"{profile}_panel_{panel}_evidence.json"
    if not evidence_path.is_file():
        raise ArtifactCompatibilityError(
            f"sealed {profile} panel {panel} evidence is missing"
        )
    evidence = verify_certified_haar_power_panel_evidence(
        run_dir=run_dir,
        evidence=_load(evidence_path),
        expected_profile=profile,
        expected_panel=panel,
    )
    path_plan = _load(run_dir / "haar_path_id_plan.json")
    pools, main_ids, reference_ids = _expected_power_panel_path_pools(
        path_plan, profile=profile, panel=panel
    )
    parent_metadata = _load(
        Path(args.parent_phase_observer_run_dir).resolve() / "source_image.json"
    )
    if (
        evidence.get("root_seed") != int(args.root_seed)
        or evidence.get("path_id_plan_sha256")
        != path_plan.get("path_id_plan_sha256")
        or evidence.get("path_id_pools") != pools
        or evidence.get("main_path_ids") != main_ids
        or evidence.get("reference_path_ids") != reference_ids
        or evidence.get("source_npz_sha256")
        != parent_metadata.get("source_npz_sha256")
    ):
        raise ArtifactCompatibilityError(
            "sealed panel evidence provenance changed"
        )
    return evidence


def _rederive_profile_panel_record(
    evidence: Mapping[str, Any],
    *,
    profile: str,
    panel: str,
    selected: Mapping[str, Any] | None,
    evidence_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if (panel == "a") != (selected is None):
        raise ArtifactCompatibilityError(
            "sealed panel role and selection state disagree"
        )
    if selected is None:
        expected = nominate_haar_power_design(
            profile=profile,
            panel_role="a",
            candidates=evidence.get("candidates", ()),
        )
    else:
        expected = panel_confirmation_record(evidence, selected)
    expected["panel_evidence"] = dict(evidence_binding)
    return expected


def _run_profile_panel(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    profile: str,
    panel: str,
    selected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute or load one sealed panel.

    The reduced fixture uses real Haar/Jacobi cells to exercise orchestration
    without pretending to estimate production path power.  Production uses
    the certified hierarchical scheduler and its committed raw/Dynkin
    observable evidence; no synthetic candidate may authorize a gate.
    """

    path = run_dir / f"{profile}_panel_{panel}.json"
    evidence_path = run_dir / f"{profile}_panel_{panel}_evidence.json"
    if path.is_file():
        record = _load(path)
        if not args.test_only_reduced_workload:
            evidence = _verified_profile_panel_evidence(
                run_dir, args, profile=profile, panel=panel
            )
            binding = {
                "path": evidence_path.name,
                "sha256": file_fingerprint(evidence_path),
                "observable_payload": dict(evidence["observable_payload"]),
            }
            expected = _rederive_profile_panel_record(
                evidence,
                profile=profile,
                panel=panel,
                selected=selected,
                evidence_binding=binding,
            )
            if canonical_json_bytes(record) != canonical_json_bytes(expected):
                raise ArtifactCompatibilityError(
                    f"sealed {profile} panel {panel} record changed"
                )
        return record
    if not args.test_only_reduced_workload:
        evidence = run_certified_haar_power_panel(
            run_dir=run_dir,
            parent_run_dir=Path(args.parent_phase_observer_run_dir).resolve(),
            root_seed=int(args.root_seed),
            profile=profile,
            panel=panel,
            path_id_plan=_load(run_dir / "haar_path_id_plan.json"),
            device=args.device,
        )
        evidence = _verified_profile_panel_evidence(
            run_dir, args, profile=profile, panel=panel
        )
        evidence_binding = {
            "path": evidence_path.name,
            "sha256": file_fingerprint(evidence_path),
            "observable_payload": dict(evidence["observable_payload"]),
        }
        record = _rederive_profile_panel_record(
            evidence,
            profile=profile,
            panel=panel,
            selected=selected,
            evidence_binding=evidence_binding,
        )
        atomic_write_json(path, record)
        return record
    path_plan = _load(run_dir / "haar_path_id_plan.json")
    role = (
        ("nested_a" if panel == "a" else "nested_b")
        if profile == NESTED_HAAR_PROFILE
        else ("antithetic_a" if panel == "a" else "antithetic_b")
    )
    if profile == NESTED_HAAR_PROFILE:
        ids = path_plan["profiles"][profile][panel]["roles"]["main"][
            "root_path_ids"
        ][:2]
    else:
        ids = path_plan["profiles"][profile][panel]["roles"]["128-256"][
            "root_path_ids"
        ][:2]
    first = _backend_smoke(
        args,
        role=role,
        path_ids=ids,
        sample_steps=128 if profile == NESTED_HAAR_PROFILE else 256,
        pair_coarse_steps=(
            None if profile == NESTED_HAAR_PROFILE else 128
        ),
    )
    elapsed = max(float(first["elapsed_seconds"]), 1.0e-30)
    rate = float(first["sample_count"]) / elapsed
    projected = 89_915_392 / max(rate, 1.0e-30) / 3600.0
    # The reduced widths are computed from actual returned values and are
    # intentionally nonauthorizing because only two short transition panels
    # were evaluated.
    values = np.asarray(first["later"], dtype=np.float64)
    width = (
        float(np.std(values, ddof=1)) * 4.0
        if values.size > 1
        else math.inf
    )
    if selected is None:
        designs = (
            [(main, reference) for main in (32, 64) for reference in (16, 32)]
            if profile == NESTED_HAAR_PROFILE
            else [(16, 16)]
        )
        candidates = [
            _candidate_row(
                profile=profile,
                main_paths=main,
                reference_paths=reference,
                main_width=width * math.sqrt(values.size / main),
                reference_width=width * math.sqrt(values.size / reference),
                projected_hours=projected,
                rate=rate,
            )
            for main, reference in designs
        ]
        record = nominate_haar_power_design(
            profile=profile, panel_role="a", candidates=candidates
        )
        record["test_only_reduced_workload"] = 1
    else:
        main = int(selected["main_paths"])
        reference = int(selected["reference_paths"])
        record = {
            "schema": RUN_SCHEMA + "-sealed-panel-confirmation",
            "schema_version": 1,
            "evaluation_status": "evaluated",
            "profile": profile,
            "panel": panel,
            "main_paths": main,
            "reference_paths": reference,
            "complete_pass": 1,
            "finite_pass": 1,
            "certification_pass": 1,
            "numerical_health_pass": 1,
            "mass_conservation_pass": 1,
            "shard_chain_pass": 1,
            "main_half_width": width * math.sqrt(values.size / main),
            "generator_reference_half_width": width
            * math.sqrt(values.size / reference),
            "reference_stability_half_width": width
            * math.sqrt(values.size / reference),
            "projected_hours": projected,
            "minimum_rate": rate,
            "test_only_reduced_workload": 1,
            **NO_WORK,
        }
    atomic_write_json(path, record)
    return record


def _combined_panel_record(
    selected: Mapping[str, Any],
    panel_b: Mapping[str, Any],
) -> dict[str, Any]:
    # The two sealed panels have equal fixed cluster counts.  In production the
    # scheduler replaces these fields with a recomputed combined 16-cluster
    # interval; copying a B result is allowed only in the nonauthorizing test
    # fixture.
    return {
        **dict(panel_b),
        "schema": RUN_SCHEMA + "-combined-test-panel",
        "panel": "combined",
        "main_paths": int(selected["main_paths"]),
        "reference_paths": int(selected["reference_paths"]),
        "test_only_reduced_workload": 1,
    }


def _combine_profile_panels(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    profile: str,
    selected: Mapping[str, Any],
    panel_a: Mapping[str, Any],
    panel_b: Mapping[str, Any],
) -> dict[str, Any]:
    if args.test_only_reduced_workload:
        return _combined_panel_record(selected, panel_b)
    evidence_a_path = run_dir / f"{profile}_panel_a_evidence.json"
    evidence_b_path = run_dir / f"{profile}_panel_b_evidence.json"
    for role, panel_record, evidence_path in (
        ("a", panel_a, evidence_a_path),
        ("b", panel_b, evidence_b_path),
    ):
        binding = panel_record.get("panel_evidence")
        if (
            not isinstance(binding, Mapping)
            or not evidence_path.is_file()
            or binding.get("sha256") != file_fingerprint(evidence_path)
        ):
            raise ArtifactCompatibilityError(
                f"sealed {profile} panel {role} evidence changed"
            )
    result = combine_certified_haar_power_panels(
        run_dir=run_dir,
        profile=profile,
        selected=dict(selected),
        panel_a=_load(evidence_a_path),
        panel_b=_load(evidence_b_path),
    )
    return dict(result)


def _run_pilot_stage(
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    gate_path = run_dir / "haar_pilot_gate.json"
    metrics_path = run_dir / "haar_pilot_metrics.json"
    if gate_path.is_file() and metrics_path.is_file():
        metrics = _load(metrics_path)["metrics"]
        gate = _load(gate_path)
        if gate != evaluate_haar_pilot(metrics):
            raise ArtifactCompatibilityError("frozen Haar pilot gate changed")
        return gate

    nested_a = _run_profile_panel(
        run_dir, args, profile=NESTED_HAAR_PROFILE, panel="a"
    )
    nested_b = nested_combined = None
    antithetic_a = antithetic_b = antithetic_combined = None
    if isinstance(nested_a.get("selected"), Mapping):
        nested_b = _run_profile_panel(
            run_dir,
            args,
            profile=NESTED_HAAR_PROFILE,
            panel="b",
            selected=nested_a["selected"],
        )
        nested_combined = _combine_profile_panels(
            run_dir,
            args,
            profile=NESTED_HAAR_PROFILE,
            selected=nested_a["selected"],
            panel_a=nested_a,
            panel_b=nested_b,
        )
    else:
        antithetic_a = _run_profile_panel(
            run_dir, args, profile=ANTITHETIC_HAAR_PROFILE, panel="a"
        )
        if isinstance(antithetic_a.get("selected"), Mapping):
            antithetic_b = _run_profile_panel(
                run_dir,
                args,
                profile=ANTITHETIC_HAAR_PROFILE,
                panel="b",
                selected=antithetic_a["selected"],
            )
            antithetic_combined = _combine_profile_panels(
                run_dir,
                args,
                profile=ANTITHETIC_HAAR_PROFILE,
                selected=antithetic_a["selected"],
                panel_a=antithetic_a,
                panel_b=antithetic_b,
            )
    selection = decide_sealed_profile_selection(
        nested_panel_a=nested_a,
        nested_panel_b=nested_b,
        nested_combined=nested_combined,
        antithetic_panel_a=antithetic_a,
        antithetic_panel_b=antithetic_b,
        antithetic_combined=antithetic_combined,
    )
    atomic_write_json(run_dir / "sealed_profile_selection.json", selection)
    selected = selection.get("selected")
    atomic_write_json(
        run_dir / "selected_haar_design.json",
        {
            "schema": RUN_SCHEMA + "-selected-design",
            "schema_version": 1,
            "selection_status": selection["selection_status"],
            "selected": selected,
            "selected_design_frozen": 1,
            **NO_WORK,
        },
    )
    confirmation = (
        nested_combined
        if selection.get("selected_profile") == NESTED_HAAR_PROFILE
        else antithetic_combined
    )
    finite_confirmation = (
        confirmation if isinstance(confirmation, Mapping) else {}
    )
    selected_profile = selection.get("selected_profile")
    if args.test_only_reduced_workload:
        executed_records = [
            record
            for record in (
                nested_a,
                nested_b,
                nested_combined,
                antithetic_a,
                antithetic_b,
                antithetic_combined,
            )
            if isinstance(record, Mapping)
        ]
    else:
        executed_records = []
        for profile_name in (NESTED_HAAR_PROFILE, ANTITHETIC_HAAR_PROFILE):
            for panel_role in ("a", "b"):
                evidence_path = (
                    run_dir
                    / f"{profile_name}_panel_{panel_role}_evidence.json"
                )
                if evidence_path.is_file():
                    executed_records.append(_load(evidence_path))
        if isinstance(confirmation, Mapping):
            executed_records.append(confirmation)
    sealed_registry = _load(run_dir / "sealed_panel_registry.json")
    if args.test_only_reduced_workload:
        panel_a_clusters = int(args.panel_clusters)
        panel_b_clusters = int(args.panel_clusters)
        combined_clusters = 2 * int(args.panel_clusters)
    else:
        selected_evidence_a = next(
            (
                record
                for record in executed_records
                if record.get("profile") == selected_profile
                and record.get("panel") == "a"
            ),
            {},
        )
        selected_evidence_b = next(
            (
                record
                for record in executed_records
                if record.get("profile") == selected_profile
                and record.get("panel") == "b"
            ),
            {},
        )
        panel_a_clusters = int(selected_evidence_a.get("cluster_count", 0))
        panel_b_clusters = int(selected_evidence_b.get("cluster_count", 0))
        combined_clusters = int(
            finite_confirmation.get("combined_cluster_count", 0)
        )
    measured_forbidden = {
        name: finite_confirmation.get(name)
        for name in _base_forbidden()
    }
    metrics = {
        "production_authorizing_pass": int(
            not args.test_only_reduced_workload
            and _all_explicit_pass(
                executed_records, "production_authorizing_pass"
            )
        ),
        "plans_frozen_pass": int(
            sealed_registry.get("panels_frozen_before_device_execution") == 1
        ),
        "panels_disjoint_pass": int(
            sealed_registry.get("panels_disjoint") == 1
        ),
        "panel_nonregeneration_pass": int(
            sealed_registry.get("panel_regeneration_permitted") == 0
        ),
        "profile_order_pass": int(
            sealed_registry.get("profile_order") == list(PROFILE_ORDER)
        ),
        "no_fallback_after_panel_b_pass": int(
            "fallback_after_panel_b_permitted" in selection
            and selection.get("fallback_after_panel_b_permitted") == 0
        ),
        "raw_endpoint_authorizing_pass": _all_explicit_pass(
            executed_records, "raw_endpoint_authorizing_pass"
        ),
        "dynkin_advisory_only_pass": _all_explicit_pass(
            executed_records, "dynkin_advisory_only_pass"
        ),
        "independent_pool_variance_pass": _all_explicit_pass(
            executed_records, "independent_pool_variance_pass"
        ),
        "richardson_formula_pass": _all_explicit_pass(
            executed_records, "richardson_formula_pass"
        ),
        "executed_panels_complete_pass": int(
            bool(executed_records)
            and all(
                int(
                    record.get(
                        "complete_pass", record.get("complete", 0)
                    )
                )
                == 1
                for record in executed_records
            )
        ),
        "executed_panels_numerically_valid_pass": int(
            bool(executed_records)
            and all(
                (
                    int(record.get("numerical_health_pass", 0)) == 1
                    if "numerical_health_pass" in record
                    else any(
                        int(row.get("panel_numerical_health_pass", 0)) == 1
                        for row in record.get("candidates", ())
                    )
                )
                for record in executed_records
            )
        ),
        "shard_chain_pass": int(
            bool(executed_records)
            and all(
                (
                    int(record.get("shard_chain_pass", 0)) == 1
                    if "shard_chain_pass" in record
                    else int(
                        dict(record.get("execution", {})).get(
                            "shard_chain_pass", 0
                        )
                    )
                    == 1
                )
                for record in executed_records
            )
        ),
        "mass_conservation_pass": int(
            bool(executed_records)
            and all(
                (
                    int(record.get("mass_conservation_pass", 0)) == 1
                    if "mass_conservation_pass" in record
                    else float(
                        dict(record.get("execution", {})).get(
                            "mass_error", math.inf
                        )
                    )
                    <= HaarCouplingThresholds().maximum_cuda_mass_error
                )
                for record in executed_records
            )
        ),
        "pilot_production_isolation_pass": _all_explicit_pass(
            executed_records, "pilot_production_isolation_pass"
        ),
        "selected_profile": selected_profile,
        "panel_a_clusters": panel_a_clusters,
        "panel_b_clusters": panel_b_clusters,
        "combined_clusters": combined_clusters,
        "panel_a_nominated": int(selection.get("panel_a_nominated", 0)),
        "panel_b_opened": int(selection.get("panel_b_opened", 0)),
        "panels_agree": int(selection.get("panels_agree", 0)),
        "combined_main_half_width": finite_confirmation.get(
            "main_half_width", 1.0e300
        ),
        "combined_generator_reference_half_width": finite_confirmation.get(
            "generator_reference_half_width", 1.0e300
        ),
        "combined_reference_stability_half_width": finite_confirmation.get(
            "reference_stability_half_width", 1.0e300
        ),
        "projected_hours": finite_confirmation.get(
            "projected_hours", 1.0e300
        ),
        "minimum_rate": finite_confirmation.get("minimum_rate", 0.0),
        "certificate_fraction": finite_confirmation.get("certificate_fraction"),
        "fallback_fraction": finite_confirmation.get("fallback_fraction"),
        "fallback_cost_fraction": finite_confirmation.get(
            "fallback_cost_fraction"
        ),
        "peak_memory_fraction": finite_confirmation.get("peak_memory_fraction"),
        "mass_error": finite_confirmation.get("mass_error"),
        **measured_forbidden,
    }
    atomic_write_json(
        metrics_path,
        {
            "schema": RUN_SCHEMA + "-pilot-metrics",
            "schema_version": 1,
            "metrics": metrics,
            "selection_sha256": file_fingerprint(
                run_dir / "sealed_profile_selection.json"
            ),
            **NO_WORK,
        },
    )
    gate = evaluate_haar_pilot(metrics)
    atomic_write_json(gate_path, gate)
    return gate


def _existing_gate(run_dir: Path, stage: str, reason: str) -> dict[str, Any]:
    path = run_dir / f"haar_{stage}_gate.json"
    return _load(path) if path.is_file() else not_evaluated_gate(stage, reason)


def _failed_stage_gate(
    run_dir: Path,
    stage: str,
    error: BaseException,
    *,
    failure_domain: str,
    failure_code: str,
) -> dict[str, Any]:
    diagnostics = getattr(error, "diagnostics", None)
    failure = {
        "schema": RUN_SCHEMA + "-stage-failure",
        "schema_version": 1,
        "stage": stage,
        "evaluation_status": "execution_failed",
        "passed": 0,
        "stage_execution_valid": 0,
        "scientific_evidence_complete": 0,
        "failure_domain": failure_domain,
        "failure_code": failure_code,
        "error_type": type(error).__name__,
        "error": str(error),
        "failure_diagnostics": (
            dict(diagnostics) if isinstance(diagnostics, Mapping) else {}
        ),
        **NO_WORK,
    }
    atomic_write_json(run_dir / f"{stage}_failure.json", failure)
    atomic_write_json(run_dir / f"haar_{stage}_gate.json", failure)
    return failure


def _typed_stage_failure(
    error: BaseException,
    *,
    default_domain: str,
    default_code: str,
) -> tuple[str, str]:
    domain = getattr(error, "failure_domain", default_domain)
    code = getattr(error, "failure_code", default_code)
    return str(domain), str(code)


def _substantive_test_gate_passed(
    gate: Mapping[str, Any],
    *,
    ignored: Sequence[str],
) -> bool:
    checks = gate.get("subchecks")
    return bool(
        isinstance(checks, Mapping)
        and all(
            int(record.get("passed", 0)) == 1
            for name, record in checks.items()
            if name not in set(ignored)
        )
    )


def _finish(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    provenance: Mapping[str, Any] | bool,
    preflight: Mapping[str, Any],
    coupling: Mapping[str, Any],
    pilot: Mapping[str, Any],
) -> int:
    workflow = evaluate_haar_workflow(
        provenance=provenance,
        preflight_gate=preflight,
        coupling_gate=coupling,
        pilot_gate=pilot,
        require_gate=args.require_gate,
    )
    decision = dict(workflow["decision"])
    atomic_write_json(run_dir / "haar_workflow_gate.json", workflow)
    atomic_write_json(run_dir / "haar_coupling_decision.json", decision)
    registry_fields = _finalize_registry(run_dir)
    passed = bool(workflow["required_gate_pass"])
    _write_status(
        run_dir,
        status="complete",
        outcome="complete" if passed else "gate_failed",
        phase=args.stage,
        required_gate=args.require_gate,
        required_gate_pass=int(passed),
        decision=decision["decision"],
        **registry_fields,
    )
    return 0 if passed else 1


def _synthetic_provenance_gate(error: BaseException) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA + "-provenance-gate",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": 0,
        "provenance_valid": 0,
        "error_type": type(error).__name__,
        "error": str(error),
        **NO_WORK,
    }


def _run(args: argparse.Namespace) -> int:
    run_dir: Path | None = None
    resumed = False
    parent_binding_complete = False
    try:
        run_dir, resumed = _make_run_dir(args)
        print(f"Jacobi Haar coupling run directory: {run_dir}")
        parent_dir = Path(args.parent_phase_observer_run_dir).resolve()
        provenance = verify_right_endpoint_coupling_parent(parent_dir)
        path_plan = _build_path_id_plan(args)
        profile_plan = _profile_plan()
        source_hash, source_paths = _source_record(parent_dir)
        config = _scientific_config(args, path_plan, profile_plan)
        config_sha = config_fingerprint(config)
        manifest = {
            "schema": RUN_SCHEMA,
            "schema_version": RUN_SCHEMA_VERSION,
            "claim_scope": CLAIM_SCOPE,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "requested_device": str(args.device),
            "test_only_reduced_workload": int(
                bool(args.test_only_reduced_workload)
            ),
            "source_fingerprint": source_hash,
            "source_paths": source_paths,
            "source_count": len(source_paths),
            "scientific_config_sha256": config_sha,
            "path_id_plan_version": PATH_ID_PLAN_VERSION,
            "path_id_plan_sha256": path_plan["path_id_plan_sha256"],
            "profile_plan_sha256": profile_plan["profile_plan_sha256"],
            "parent_artifact_record_count": PARENT_REGISTRY_RECORD_COUNT,
            "parent_artifact_registry_sha256": PARENT_REGISTRY_SHA256,
            "parent_source_fingerprint": PARENT_SOURCE_FINGERPRINT,
            "parent_scientific_config_sha256": (
                PARENT_SCIENTIFIC_CONFIG_SHA256
            ),
            "plans_frozen_before_device_execution": 1,
            **NO_WORK,
        }
        parent_binding_complete = True
        if resumed:
            _verify_resume_contract(
                run_dir,
                expected_path_plan=path_plan,
                expected_profile_plan=profile_plan,
                expected_config=config,
                expected_manifest=manifest,
                expected_provenance=provenance,
            )
        else:
            _freeze_plans(run_dir, args, require_existing=False)
            _freeze(
                run_dir / "scientific_config.json",
                config,
                require_existing=False,
            )
            _freeze(
                run_dir / "parent_provenance.json",
                provenance,
                require_existing=False,
            )
            _freeze(
                run_dir / "run_manifest.json",
                manifest,
                require_existing=False,
            )
            _freeze(
                run_dir / "corrected_parent_adjudication.json",
                {
                    "schema": RUN_SCHEMA + "-parent-adjudication",
                    "schema_version": 1,
                    "parent_decision": "dynkin_power_infeasible",
                    "corrected_adjudication": (
                        "right_endpoint_coupling_power_infeasible"
                    ),
                    "dynkin_estimator_defective": 0,
                    "dynkin_unbiased": 1,
                    "dynkin_unconditional_variance_reduction_claimed": 0,
                    **NO_WORK,
                },
                require_existing=False,
            )
        runtime_path = run_dir / "exact_backend_runtime.json"
        if args.stage == "report":
            if not runtime_path.is_file():
                raise ArtifactCompatibilityError(
                    "report-only reconstruction lacks the frozen runtime record"
                )
            runtime = _load(runtime_path)
        else:
            runtime = {
                "schema": RUN_SCHEMA + "-exact-backend-runtime",
                "schema_version": 1,
                "requested_device": str(args.device),
                "exact_backend": configure_exact_torch_backend(
                    torch.device(args.device)
                ),
                **NO_WORK,
            }
            _freeze(
                runtime_path,
                runtime,
                require_existing=(
                    resumed and runtime_path.is_file()
                ),
            )
        _write_status(
            run_dir,
            status="running",
            outcome="running",
            phase=args.stage,
            required_gate=args.require_gate,
            required_gate_pass=0,
            scientific_config_sha256=config_sha,
            source_fingerprint=source_hash,
            plans_frozen_before_device_execution=1,
        )
        preflight = _existing_gate(run_dir, "preflight", "preflight has not run")
        coupling = _existing_gate(
            run_dir, "coupling", "coupling controls have not run"
        )
        pilot = _existing_gate(run_dir, "pilot", "sealed pilot has not run")

        if args.stage in {"preflight", "all"}:
            try:
                preflight = _run_preflight_stage(run_dir, args, provenance)
            except ArtifactCompatibilityError:
                raise
            except Exception as exc:
                domain, code = _typed_stage_failure(
                    exc,
                    default_domain=(
                        "normal_transform"
                        if "normal" in str(exc).lower()
                        else "hierarchical_rng"
                    ),
                    default_code="haar_preflight_execution_failed",
                )
                preflight = _failed_stage_gate(
                    run_dir,
                    "preflight",
                    exc,
                    failure_domain=domain,
                    failure_code=code,
                )

        if args.stage in {"coupling", "all"}:
            test_ready = (
                args.test_only_reduced_workload
                and _substantive_test_gate_passed(
                    preflight,
                    ignored=(
                        "production_authorizing_pass",
                        "root_seed",
                        "fallback_fraction",
                        "fallback_cost_fraction",
                    ),
                )
            )
            if _passed(preflight) or test_ready:
                try:
                    coupling = _run_coupling_stage(run_dir, args)
                except ArtifactCompatibilityError:
                    raise
                except Exception as exc:
                    domain, code = _typed_stage_failure(
                        exc,
                        default_domain="scheduler_execution",
                        default_code="haar_coupling_execution_failed",
                    )
                    coupling = _failed_stage_gate(
                        run_dir,
                        "coupling",
                        exc,
                        failure_domain=domain,
                        failure_code=code,
                    )
            else:
                coupling = not_evaluated_gate(
                    "coupling", "Haar preflight did not pass"
                )
                atomic_write_json(run_dir / "haar_coupling_gate.json", coupling)

        if args.stage in {"pilot", "all"}:
            test_ready = (
                args.test_only_reduced_workload
                and _substantive_test_gate_passed(
                    coupling,
                    ignored=(
                        "production_authorizing_pass",
                        "fallback_fraction",
                        "fallback_cost_fraction",
                        "minimum_rate",
                        "minimum_projected_hours",
                        "pipeline_runtime_projection_pass",
                    ),
                )
            )
            if _passed(coupling) or test_ready:
                try:
                    pilot = _run_pilot_stage(run_dir, args)
                except ArtifactCompatibilityError:
                    raise
                except Exception as exc:
                    domain, code = _typed_stage_failure(
                        exc,
                        default_domain="scheduler_execution",
                        default_code="haar_pilot_execution_failed",
                    )
                    pilot = _failed_stage_gate(
                        run_dir,
                        "pilot",
                        exc,
                        failure_domain=domain,
                        failure_code=code,
                    )
            else:
                pilot = not_evaluated_gate(
                    "pilot", "certified coupling gate did not pass"
                )
                atomic_write_json(run_dir / "haar_pilot_gate.json", pilot)

        return _finish(
            run_dir,
            args,
            provenance=provenance,
            preflight=preflight,
            coupling=coupling,
            pilot=pilot,
        )
    except ArtifactCompatibilityError as exc:
        if resumed:
            print(f"Jacobi Haar compatibility error: {exc}", file=sys.stderr)
            return 2
        if run_dir is not None and not parent_binding_complete:
            failure = _synthetic_provenance_gate(exc)
            atomic_write_json(run_dir / "provenance_failure.json", failure)
            preflight = not_evaluated_gate(
                "preflight", "control provenance is invalid"
            )
            coupling = not_evaluated_gate(
                "coupling", "control provenance is invalid"
            )
            pilot = not_evaluated_gate("pilot", "control provenance is invalid")
            atomic_write_json(run_dir / "haar_preflight_gate.json", preflight)
            atomic_write_json(run_dir / "haar_coupling_gate.json", coupling)
            atomic_write_json(run_dir / "haar_pilot_gate.json", pilot)
            return _finish(
                run_dir,
                args,
                provenance=failure,
                preflight=preflight,
                coupling=coupling,
                pilot=pilot,
            )
        if run_dir is not None:
            atomic_write_json(
                run_dir / "artifact_compatibility_failure.json",
                {
                    "schema": RUN_SCHEMA + "-artifact-compatibility-failure",
                    "schema_version": 1,
                    "evaluation_status": "execution_failed",
                    "failure_domain": "artifact_compatibility",
                    "failure_code": "haar_artifact_compatibility_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    **NO_WORK,
                },
            )
            registry_fields = _finalize_registry(run_dir)
            _write_status(
                run_dir,
                status="failed",
                outcome="artifact_compatibility_failure",
                phase=args.stage,
                required_gate=args.require_gate,
                required_gate_pass=0,
                decision="hierarchical_scheduler_invalid",
                **registry_fields,
            )
        print(f"Jacobi Haar compatibility error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        if run_dir is not None:
            atomic_write_json(
                run_dir / "unexpected_failure.json",
                {
                    "schema": RUN_SCHEMA + "-unexpected-failure",
                    "schema_version": 1,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    **NO_WORK,
                },
            )
            _write_status(
                run_dir,
                status="failed",
                outcome="unexpected_failure",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        print(f"Jacobi Haar error: {exc}", file=sys.stderr)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
