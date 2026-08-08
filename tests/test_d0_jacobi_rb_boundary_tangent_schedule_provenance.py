from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnist.d0_jacobi_artifacts import config_fingerprint
import mnist.d0_jacobi_rb_boundary_tangent_schedule_provenance as provenance


FAILED_RUN = Path(
    "runs/experiment12_d0_jacobi_rb_boundary_tangent_controller_confirmation/"
    "20260802-140158_production-boundary-tangent-rb-controller"
)
COARSE_RUN = Path(
    "runs/experiment12_d0_jacobi_rb_coarse_residual_learnability/"
    "20260731-140333_production-exact-k512-coarse-residual-one-image"
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def test_real_parent_binding_and_resource_only_readjudication() -> None:
    record = provenance.verify_and_readjudicate_boundary_tangent_schedule_parents(
        failed_boundary_tangent_run_dir=FAILED_RUN,
        parent_coarse_residual_run_dir=COARSE_RUN,
    )
    assert record["passed"] == 1
    assert record["failed_registry"] == {
        "artifact_count": 14,
        "semantic_sha256": provenance.FAILED_REGISTRY_SEMANTIC_SHA256,
        "file_sha256": provenance.FAILED_REGISTRY_FILE_SHA256,
    }
    assert record["readjudicated_decision"] == (
        "eight_path_cache_schedule_resource_infeasible"
    )
    assert record["readjudicated_failure_domain"] == "resource_gate"
    assert record["scientific_evidence_complete"] == 1
    assert record["numerically_valid"] == 1
    assert record["resource_valid"] == 0
    assert record["physical_training_performed"] == 0
    assert record["controller_control_trajectory_performed"] == 0


def test_parent_hash_or_basename_change_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance, "FAILED_REGISTRY_COUNT", 15)
    with pytest.raises(provenance.BoundaryTangentScheduleProvenanceError):
        provenance.verify_and_readjudicate_boundary_tangent_schedule_parents(
            failed_boundary_tangent_run_dir=FAILED_RUN,
            parent_coarse_residual_run_dir=COARSE_RUN,
        )
    monkeypatch.setattr(provenance, "FAILED_REGISTRY_COUNT", 14)
    with pytest.raises(provenance.BoundaryTangentScheduleProvenanceError):
        provenance.verify_and_readjudicate_boundary_tangent_schedule_parents(
            failed_boundary_tangent_run_dir=FAILED_RUN.parent,
            parent_coarse_residual_run_dir=COARSE_RUN,
        )


def test_fresh_path_plan_exact_slots_and_collision_rejection() -> None:
    plan = provenance.build_schedule_path_plan()
    validation = provenance.validate_schedule_path_plan(plan)
    assert validation["passed"] == 1
    assert validation["path_count"] == 40
    assert plan["roles"]["cache_p10"] == list(range(0xEE000, 0xEE00A))
    assert plan["roles"]["cache_p6"] == list(range(0xEE010, 0xEE016))
    assert plan["roles"]["stream_p10"] == list(range(0xEE100, 0xEE10A))
    assert plan["roles"]["stream_p4"] == list(range(0xEE110, 0xEE114))
    assert plan["roles"]["cuda_warmup"] == list(range(0xEE200, 0xEE20A))
    with pytest.raises(provenance.BoundaryTangentScheduleProvenanceError):
        provenance.validate_schedule_path_plan(
            plan, claimed_ids={"historical": [0xEE000]}
        )
    with pytest.raises(provenance.BoundaryTangentScheduleProvenanceError):
        provenance.validate_schedule_path_plan(plan, claimed_ids=[1 << 20])


def test_cohort_and_timing_plans_are_frozen() -> None:
    cohorts = provenance.build_schedule_cohort_plan()
    timing = provenance.build_schedule_timing_plan()
    assert provenance.validate_schedule_cohort_plan(cohorts)["passed"] == 1
    assert provenance.validate_schedule_timing_plan(timing)["passed"] == 1
    production = cohorts["production"]
    assert production["train_validation"]["group_sizes"] == [10] * 9 + [6]
    assert production["confirmation"]["group_sizes"] == [10] * 6 + [4]
    assert production["train_validation"]["role_slices"] == {
        "train": [0, 64],
        "validation": [64, 96],
    }
    assert max(
        row["maximum_phase_launch_lanes"]
        for role in production.values()
        for row in role["cohorts"]
    ) == 3920
    assert timing["window_start_outer_steps"] == [0, 128, 256, 384]
    assert timing["branch_outer_steps"] == [15, 143, 271, 399]
    assert timing["repeat_profile_orders"][1] == [
        "cache_p6",
        "stream_p10",
        "stream_p4",
        "cache_p10",
    ]

    changed = dict(cohorts)
    changed["maximum_launch_lanes"] = 4095
    with pytest.raises(provenance.BoundaryTangentScheduleProvenanceError):
        provenance.validate_schedule_cohort_plan(changed)


def test_resume_binds_all_frozen_plans(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    parent = {"schema": "parent", "schema_version": 1, "passed": 1}
    parent["semantic_sha256"] = config_fingerprint(parent)
    path_plan = provenance.build_schedule_path_plan()
    cohort_plan = provenance.build_schedule_cohort_plan()
    timing_plan = provenance.build_schedule_timing_plan()
    config = {
        "schema": "schedule-config",
        "schema_version": 1,
        "maximum_projected_exact_cache_hours": 30.0,
    }
    config["semantic_sha256"] = config_fingerprint(config)
    source_hash = "source-hash"
    expected = {
        "source_fingerprint": source_hash,
        "scientific_config_sha256": config["semantic_sha256"],
        "parent_provenance_sha256": parent["semantic_sha256"],
        "path_plan_sha256": path_plan["semantic_sha256"],
        "cohort_plan_sha256": cohort_plan["semantic_sha256"],
        "timing_plan_sha256": timing_plan["semantic_sha256"],
    }
    _write_json(run / "run_manifest.json", {"schema": "manifest", **expected})
    _write_json(run / "scientific_config.json", config)
    _write_json(run / "parent_provenance.json", parent)
    _write_json(run / "path_id_plan.json", path_plan)
    _write_json(run / "cohort_plan.json", cohort_plan)
    _write_json(run / "timing_plan.json", timing_plan)
    assert provenance.verify_schedule_resume_compatibility(
        run, source_fingerprint_value=source_hash,
        scientific_config_sha256=config["semantic_sha256"],
        parent_provenance_sha256=parent["semantic_sha256"],
        path_plan_sha256=path_plan["semantic_sha256"],
        cohort_plan_sha256=cohort_plan["semantic_sha256"],
        timing_plan_sha256=timing_plan["semantic_sha256"],
    )["passed"] == 1

    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["timing_plan_sha256"] = "changed"
    _write_json(run / "run_manifest.json", manifest)
    with pytest.raises(provenance.BoundaryTangentScheduleProvenanceError):
        provenance.verify_schedule_resume_compatibility(
            run, source_fingerprint_value=source_hash,
            scientific_config_sha256=config["semantic_sha256"],
            parent_provenance_sha256=parent["semantic_sha256"],
            path_plan_sha256=path_plan["semantic_sha256"],
            cohort_plan_sha256=cohort_plan["semantic_sha256"],
            timing_plan_sha256=timing_plan["semantic_sha256"],
        )
