from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnist.d0_jacobi_artifacts import config_fingerprint
import mnist.d0_jacobi_rb_boundary_tangent_eager_provenance as provenance


EAGER_RUN = Path(
    "runs/experiment12_d0_jacobi_rb_boundary_tangent_eager_pipeline_confirmation"
) / provenance.EAGER_RUN_BASENAME
FAILED_TANGENT_RUN = Path(
    "runs/experiment12_d0_jacobi_rb_boundary_tangent_controller_confirmation"
) / provenance.FAILED_TANGENT_RUN_BASENAME
COARSE_RUN = Path(
    "runs/experiment12_d0_jacobi_rb_coarse_residual_learnability/"
    "20260731-140333_production-exact-k512-coarse-residual-one-image"
)
FAILED_AFFINE_RUN = Path(
    "runs/experiment12_d0_jacobi_rb_reverse_controller_control/"
    "20260802-040147_production-exact-rb-reverse-controller-control"
)
PARENTS_AVAILABLE = all(
    path.is_dir()
    for path in (EAGER_RUN, FAILED_TANGENT_RUN, COARSE_RUN, FAILED_AFFINE_RUN)
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _hashed(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["semantic_sha256"] = config_fingerprint(result)
    return result


@pytest.mark.skipif(not PARENTS_AVAILABLE, reason="immutable parents unavailable")
def test_exact_parents_and_eager_readjudication_are_accepted() -> None:
    record = provenance.verify_eager_boundary_tangent_parents(
        eager_pipeline_run_dir=EAGER_RUN,
        failed_boundary_tangent_run_dir=FAILED_TANGENT_RUN,
        parent_coarse_residual_run_dir=COARSE_RUN,
        failed_affine_controller_run_dir=FAILED_AFFINE_RUN,
    )
    assert record["passed"] == 1
    assert record["readjudicated_decision"] == (
        "legacy_schedule_resource_projection_superseded"
    )
    assert record["historical_boundary_tangent_decision"] == (
        "boundary_tangent_design_infeasible"
    )
    assert record["historical_schedule_readjudication"] == (
        "eight_path_cache_schedule_resource_infeasible"
    )
    eager = record["parents"]["successful_eager_pipeline"]
    assert eager["registry"] == {
        "artifact_count": 615,
        "semantic_sha256": provenance.EAGER_REGISTRY_SEMANTIC_SHA256,
        "file_sha256": provenance.EAGER_REGISTRY_FILE_SHA256,
    }
    assert eager["decision"] == "exact_boundary_tangent_eager_pipeline_feasible"
    assert eager["projected_transition_count"] == 337_182_720
    assert eager["projected_hours"] == pytest.approx(25.984910233561983)
    assert eager["projected_effective_rate"] == pytest.approx(3_604.471434567184)
    failed = record["parents"]["failed_boundary_tangent"]
    assert failed["registry"]["artifact_count"] == 14
    assert failed["preflight_benchmark_path_count"] == 8
    assert failed["unopened_production_path_count"] == 160
    assert failed["production_path_roles_unopened"] == 1
    assert set(failed["unopened_production_roles"]) == {
        "train",
        "validation",
        "confirmation",
    }
    assert record["parents"]["successful_coarse_residual"]["decision"] == (
        "exact_rb_coarse_residual_learnable"
    )
    assert record["parents"]["failed_affine_reverse_controller"]["decision"] == (
        "controller_boundary_or_conservation_failed"
    )
    assert record["schedule_integration_authorized"] == 1
    assert record["fresh_v2_workflow_authorized"] == 1
    assert record["cache_generation_authorized"] == 0
    assert record["physical_training_performed"] == 0
    body = dict(record)
    assert body.pop("semantic_sha256") == config_fingerprint(body)

    readjudication = provenance.build_eager_boundary_tangent_readjudication(record)
    assert readjudication["passed"] == 1
    assert readjudication["readjudicated_decision"] == (
        "legacy_schedule_resource_projection_superseded"
    )
    assert readjudication["historical_gates_mutated"] == 0
    assert readjudication["parent_artifacts_mutated"] == 0
    assert readjudication["schedule_integration_authorized"] == 1
    assert readjudication["training_authorized"] == 0
    body = dict(readjudication)
    assert body.pop("semantic_sha256") == config_fingerprint(body)


def test_v2_path_plan_uses_fresh_seam_and_preserves_v1_claim() -> None:
    plan = provenance.build_eager_boundary_tangent_path_plan()
    validation = provenance.validate_eager_boundary_tangent_path_plan(plan)
    assert validation["passed"] == 1
    assert validation["path_count"] == 168
    assert validation["historical_claim_count"] == 8
    assert plan["root_seed"] == 261_311
    assert plan["reserved_control_seed"] == 261_316
    assert plan["roles"]["train"] == list(range(0xEC100, 0xEC140))
    assert plan["roles"]["validation"] == list(range(0xEC200, 0xEC220))
    assert plan["roles"]["confirmation"] == list(range(0xED000, 0xED040))
    assert plan["roles"]["preflight_seam"] == list(range(0xEF000, 0xEF008))
    assert plan["preflight_seam_path_ids"] == list(range(0xEF000, 0xEF008))
    assert plan["forbidden_historical_v1_path_ids"] == list(
        range(0xEC000, 0xEC008)
    )
    active = {path_id for values in plan["roles"].values() for path_id in values}
    assert active.isdisjoint(plan["forbidden_historical_v1_path_ids"])
    with pytest.raises(
        provenance.EagerBoundaryTangentProvenanceError,
        match="collide with claims",
    ):
        provenance.validate_eager_boundary_tangent_path_plan(
            plan, claimed_ids={"other": [0xEF000]}
        )


def test_source_helpers_and_resume_binding(tmp_path: Path) -> None:
    source_paths = provenance.eager_boundary_tangent_source_paths()
    assert source_paths == (Path(provenance.__file__).resolve(),)
    assert provenance.eager_boundary_tangent_source_fingerprint() == (
        provenance.eager_boundary_tangent_source_fingerprint(source_paths)
    )

    run = tmp_path / "resume-run"
    run.mkdir()
    parent = _hashed({"schema": "parent", "schema_version": 1, "passed": 1})
    readjudication = _hashed(
        {"schema": "readjudication", "schema_version": 1, "passed": 1}
    )
    path_plan = provenance.build_eager_boundary_tangent_path_plan()
    config = _hashed(
        {
            "schema": "config",
            "schema_version": 1,
            "root_seed": 261_311,
            "reserved_control_seed": 261_316,
            "physical_training_performed": 0,
            "training_authorized": 0,
        }
    )
    expected = {
        "source_fingerprint": "source-hash",
        "scientific_config_sha256": config["semantic_sha256"],
        "parent_provenance_sha256": parent["semantic_sha256"],
        "parent_readjudication_sha256": readjudication["semantic_sha256"],
        "path_plan_sha256": path_plan["semantic_sha256"],
    }
    _write_json(run / "run_manifest.json", {"schema": "manifest", **expected})
    _write_json(run / "scientific_config.json", config)
    _write_json(run / "parent_provenance.json", parent)
    _write_json(run / "parent_readjudication.json", readjudication)
    _write_json(run / "path_id_plan.json", path_plan)
    result = provenance.verify_eager_boundary_tangent_resume_compatibility(
        run,
        source_fingerprint_value="source-hash",
        scientific_config_sha256=str(config["semantic_sha256"]),
        parent_provenance_sha256=str(parent["semantic_sha256"]),
        parent_readjudication_sha256=str(readjudication["semantic_sha256"]),
        path_plan_sha256=str(path_plan["semantic_sha256"]),
    )
    assert result["passed"] == 1
    body = dict(result)
    assert body.pop("semantic_sha256") == config_fingerprint(body)

    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["path_plan_sha256"] = "changed"
    _write_json(run / "run_manifest.json", manifest)
    with pytest.raises(
        provenance.EagerBoundaryTangentProvenanceError,
        match="resume manifest compatibility changed",
    ):
        provenance.verify_eager_boundary_tangent_resume_compatibility(
            run,
            source_fingerprint_value="source-hash",
            scientific_config_sha256=str(config["semantic_sha256"]),
            parent_provenance_sha256=str(parent["semantic_sha256"]),
            parent_readjudication_sha256=str(readjudication["semantic_sha256"]),
            path_plan_sha256=str(path_plan["semantic_sha256"]),
        )


def test_wrong_basename_and_unverified_readjudication_fail_closed(
    tmp_path: Path,
) -> None:
    wrong = tmp_path / "wrong-eager"
    wrong.mkdir()
    with pytest.raises(
        provenance.EagerBoundaryTangentProvenanceError,
        match="wrong eager run basename",
    ):
        provenance.verify_eager_boundary_tangent_parents(
            eager_pipeline_run_dir=wrong,
            failed_boundary_tangent_run_dir=FAILED_TANGENT_RUN,
            parent_coarse_residual_run_dir=COARSE_RUN,
        )
    with pytest.raises(
        provenance.EagerBoundaryTangentProvenanceError,
        match="provenance did not pass",
    ):
        provenance.build_eager_boundary_tangent_readjudication(
            {"schema": provenance.SCHEMA, "passed": 0}
        )


@pytest.mark.skipif(not EAGER_RUN.is_dir(), reason="immutable eager parent unavailable")
def test_eager_registry_or_live_source_change_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance, "EAGER_REGISTRY_COUNT", 616)
    with pytest.raises(
        provenance.EagerBoundaryTangentProvenanceError,
        match="terminal registry changed",
    ):
        provenance._verify_eager_parent(EAGER_RUN.resolve(), COARSE_RUN.resolve())

    monkeypatch.setattr(provenance, "EAGER_REGISTRY_COUNT", 615)
    manifest = json.loads((EAGER_RUN / "run_manifest.json").read_text(encoding="utf-8"))
    with pytest.raises(
        provenance.EagerBoundaryTangentProvenanceError,
        match="live source fingerprint changed",
    ):
        provenance._resolve_live_sources(
            manifest,
            expected_relative_paths=provenance._EAGER_SOURCE_PATHS,
            expected_fingerprint="0" * 64,
            description="successful eager pipeline",
        )
