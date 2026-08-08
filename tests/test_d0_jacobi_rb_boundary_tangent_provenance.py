from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from mnist.d0_jacobi_artifacts import (
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)
import mnist.d0_jacobi_rb_boundary_tangent_provenance as provenance


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _semantic(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["semantic_sha256"] = config_fingerprint(result)
    return result


def _gate(schema: str, *, passed: int = 1) -> dict[str, object]:
    return {
        "schema": schema,
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": passed,
        "sampling_performed": 0,
        "reverse_sampling_performed": 0,
        "reconstruction_performed": 0,
    }


def _finalize_registry(run: Path, run_schema: str) -> tuple[int, str, str]:
    rows = []
    for path in sorted(run.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and path.name not in {
            "artifact_registry.json",
            "run_status.json",
        }:
            rows.append(
                {
                    "path": path.relative_to(run).as_posix(),
                    "sha256": file_fingerprint(path),
                    "size": path.stat().st_size,
                }
            )
    semantic = config_fingerprint({"artifacts": rows})
    _write_json(
        run / "artifact_registry.json",
        {
            "schema": run_schema.removesuffix("-manifest")
            + "-artifact-registry",
            "schema_version": 1,
            "artifact_count": len(rows),
            "artifacts": rows,
            "semantic_sha256": semantic,
            "sampling_performed": 0,
            "reverse_sampling_performed": 0,
            "reconstruction_performed": 0,
        },
    )
    return len(rows), semantic, file_fingerprint(run / "artifact_registry.json")


def _fake_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    source = tmp_path / "frozen_source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    source_hash = source_fingerprint((source,))

    coarse = tmp_path / "coarse-parent"
    coarse.mkdir()
    coarse_config = _semantic(
        {
            "schema": provenance.COARSE_RESIDUAL_PARENT_SPEC.config_schema,
            "schema_version": 1,
            "sampling_performed": 0,
            "reverse_sampling_performed": 0,
            "reconstruction_performed": 0,
        }
    )
    coarse_config_hash = str(coarse_config["semantic_sha256"])
    _write_json(coarse / "scientific_config.json", coarse_config)
    _write_json(
        coarse / "run_manifest.json",
        {
            "schema": provenance.COARSE_RESIDUAL_PARENT_SPEC.run_schema,
            "schema_version": 1,
            "source_paths": [str(source.resolve())],
            "source_fingerprint": source_hash,
            "scientific_config_sha256": coarse_config_hash,
            "sampling_performed": 0,
            "reverse_sampling_performed": 0,
            "reconstruction_performed": 0,
        },
    )
    _write_json(
        coarse / "coarse_residual_decision.json",
        {
            "schema": provenance.COARSE_RESIDUAL_PARENT_SPEC.decision_schema,
            "schema_version": 1,
            "evaluation_status": "evaluated",
            "decision": provenance.COARSE_RESIDUAL_PARENT_SPEC.decision,
            "reverse_controller_planning_authorized": 1,
            "sampling_performed": 0,
            "reverse_sampling_performed": 0,
            "reconstruction_performed": 0,
        },
    )
    gate_schema = "experiment12-d0-jacobi-rb-coarse-residual-gate"
    for name in (
        "preflight_gate.json",
        "cache_gate.json",
        "train_gate.json",
        "confirmation_cache_gate.json",
        "confirmation_gate.json",
    ):
        _write_json(coarse / name, _gate(gate_schema))
    _write_json(coarse / "workflow_gate.json", _gate(gate_schema + "-workflow"))
    selected_model = coarse / "selected_model.pt"
    selected_model.write_bytes(b"selected boundary-parent model")
    selected_hash = file_fingerprint(selected_model)
    state_hash = "state-hash"
    baseline_hash = "baseline-hash"
    monkeypatch.setattr(provenance, "SELECTED_CHECKPOINT_SHA256", selected_hash)
    monkeypatch.setattr(provenance, "SELECTED_STATE_DICT_SHA256", state_hash)
    monkeypatch.setattr(provenance, "FROZEN_COARSE_BASELINE_SHA256", baseline_hash)
    _write_json(
        coarse / "selected_model.json",
        {
            "nonzero_residual_selected": 1,
            "confirmation_inspected": 0,
            "selected_model_sha256": selected_hash,
            "candidate": {"state_sha256": state_hash},
        },
    )
    _write_json(coarse / "confirmation_open.json", {"open_count": 1})
    coarse_count, coarse_registry, coarse_registry_file = _finalize_registry(
        coarse, provenance.COARSE_RESIDUAL_PARENT_SPEC.run_schema
    )
    _write_json(
        coarse / "run_status.json",
        {
            "schema": provenance.COARSE_RESIDUAL_PARENT_SPEC.status_schema,
            "schema_version": 1,
            "state": provenance.COARSE_RESIDUAL_PARENT_SPEC.terminal_state,
            "stage": provenance.COARSE_RESIDUAL_PARENT_SPEC.terminal_stage,
            "decision": provenance.COARSE_RESIDUAL_PARENT_SPEC.decision,
            "physical_training_performed": 1,
            "sampling_performed": 0,
            "reverse_sampling_performed": 0,
            "reconstruction_performed": 0,
        },
    )
    coarse_spec = replace(
        provenance.COARSE_RESIDUAL_PARENT_SPEC,
        basename=coarse.name,
        registry_count=coarse_count,
        registry_semantic_sha256=coarse_registry,
        registry_file_sha256=coarse_registry_file,
        source_fingerprint=source_hash,
        scientific_config_sha256=coarse_config_hash,
    )
    monkeypatch.setattr(provenance, "COARSE_RESIDUAL_PARENT_SPEC", coarse_spec)

    failed = tmp_path / "failed-controller"
    failed.mkdir()
    failed_config = _semantic(
        {
            "schema": provenance.FAILED_CONTROLLER_PARENT_SPEC.config_schema,
            "schema_version": 1,
            "sampling_performed": 0,
            "reverse_sampling_performed": 0,
            "reconstruction_performed": 0,
        }
    )
    failed_config_hash = str(failed_config["semantic_sha256"])
    _write_json(failed / "scientific_config.json", failed_config)
    _write_json(
        failed / "run_manifest.json",
        {
            "schema": provenance.FAILED_CONTROLLER_PARENT_SPEC.run_schema,
            "schema_version": 1,
            "source_paths": [str(source.resolve())],
            "source_fingerprint": source_hash,
            "scientific_config_sha256": failed_config_hash,
            "sampling_performed": 0,
            "reverse_sampling_performed": 0,
            "reconstruction_performed": 0,
        },
    )
    _write_json(
        failed / "controller_decision.json",
        {
            "schema": provenance.FAILED_CONTROLLER_PARENT_SPEC.decision_schema,
            "schema_version": 1,
            "evaluation_status": "evaluated",
            "decision": provenance.FAILED_CONTROLLER_PARENT_SPEC.decision,
            "controller_control_trajectory_performed": 0,
            "maximum_control_trajectory_phase_count": 0,
            "sampling_performed": 0,
            "reverse_sampling_performed": 0,
            "reconstruction_performed": 0,
        },
    )
    _write_json(
        failed / "preflight_gate.json",
        {
            **_gate(
                "experiment12-d0-jacobi-rb-reverse-controller-control-"
                "preflight-gate",
                passed=0,
            ),
            "boundary_rejection": {
                "failure_code": "controller_boundary_step_rejected"
            },
        },
    )
    for name, gate_name in (
        ("cache_gate.json", "cache"),
        ("oracle_gate.json", "oracle"),
        ("control_gate.json", "control"),
    ):
        _write_json(
            failed / name,
            {
                "evaluation_status": "not_evaluated",
                "passed": 0,
                "gate": gate_name,
                "reason": "skipped_after_failed_preflight_gate",
            },
        )
    _write_json(
        failed / "decide_gate.json",
        {
            **_gate("controller-decide", passed=0),
            "gate": "decide",
            "decision": provenance.FAILED_CONTROLLER_PARENT_SPEC.decision,
        },
    )
    _write_json(
        failed / "parent_provenance.json",
        {
            "passed": 1,
            "parent_run_dir": str(coarse.resolve()),
            "registry_count": coarse_count,
            "registry_file_sha256": coarse_registry_file,
            "registry_semantic_sha256": coarse_registry,
            "source_fingerprint": source_hash,
            "scientific_config_sha256": coarse_config_hash,
        },
    )
    _write_json(
        failed / "selected_checkpoint_binding.json",
        {
            "checkpoint_sha256": selected_hash,
            "state_dict_sha256": state_hash,
            "seed": 261254,
            "update": 3000,
        },
    )
    _write_json(
        failed / "frozen_baseline_binding.json",
        {"values_c_order_sha256": baseline_hash, "refit_performed": 0},
    )
    failed_count, failed_registry, failed_registry_file = _finalize_registry(
        failed, provenance.FAILED_CONTROLLER_PARENT_SPEC.run_schema
    )
    _write_json(
        failed / "run_status.json",
        {
            "schema": provenance.FAILED_CONTROLLER_PARENT_SPEC.status_schema,
            "schema_version": 1,
            "state": provenance.FAILED_CONTROLLER_PARENT_SPEC.terminal_state,
            "stage": provenance.FAILED_CONTROLLER_PARENT_SPEC.terminal_stage,
            "decision": provenance.FAILED_CONTROLLER_PARENT_SPEC.decision,
            "failure_domain": "controller_boundary",
            "failure_code": "controller_boundary_step_rejected",
            "controller_control_trajectory_performed": 0,
            "maximum_control_trajectory_phase_count": 0,
            "sampling_performed": 0,
            "reverse_sampling_performed": 0,
            "reconstruction_performed": 0,
        },
    )
    failed_spec = replace(
        provenance.FAILED_CONTROLLER_PARENT_SPEC,
        basename=failed.name,
        registry_count=failed_count,
        registry_semantic_sha256=failed_registry,
        registry_file_sha256=failed_registry_file,
        source_fingerprint=source_hash,
        scientific_config_sha256=failed_config_hash,
    )
    monkeypatch.setattr(provenance, "FAILED_CONTROLLER_PARENT_SPEC", failed_spec)
    return coarse, failed


def test_parent_binding_readjudication_and_tamper_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coarse, failed = _fake_parents(tmp_path, monkeypatch)
    binding = provenance.verify_boundary_tangent_parents(
        coarse_residual_run_dir=coarse,
        failed_controller_run_dir=failed,
    )
    assert binding["passed"] == 1
    assert binding["failed_controller_evidence"][
        "downstream_scientific_stages_opened"
    ] == 0
    readjudication = provenance.build_failed_controller_readjudication(binding)
    assert (
        readjudication["corrected_adjudication"]
        == "frozen_affine_controller_boundary_domain_invalid"
    )
    assert readjudication["exact_rao_blackwell_target_invalidated"] == 0

    (failed / "controller_decision.json").write_text("{}", encoding="utf-8")
    with pytest.raises(provenance.BoundaryTangentProvenanceError, match="changed"):
        provenance.verify_boundary_tangent_parents(
            coarse_residual_run_dir=coarse,
            failed_controller_run_dir=failed,
        )


def test_exact_path_slots_bounds_and_semantic_collision() -> None:
    plan = provenance.build_boundary_tangent_path_plan()
    result = provenance.validate_boundary_tangent_path_plan(plan)
    assert result["active_path_count"] == 8 + 64 + 32 + 64
    assert plan["roles"]["preflight_benchmark"] == list(
        range(0xEC000, 0xEC008)
    )
    assert plan["roles"]["train"] == list(range(0xEC100, 0xEC140))
    assert plan["roles"]["validation"] == list(range(0xEC200, 0xEC220))
    assert plan["roles"]["confirmation"] == list(range(0xED000, 0xED040))
    assert plan["reserved_roles"]["future_production"] == {
        "start": 0xF0000,
        "stop_exclusive": 0x100000,
    }

    changed = dict(plan)
    changed["fresh_path_count"] = 167
    with pytest.raises(provenance.BoundaryTangentProvenanceError, match="changed"):
        provenance.validate_boundary_tangent_path_plan(changed)
    with pytest.raises(provenance.BoundaryTangentProvenanceError, match="collide"):
        provenance.validate_boundary_tangent_path_plan(
            plan, claimed_ids={"historical": [0xEC100]}
        )
    with pytest.raises(provenance.BoundaryTangentProvenanceError, match="bounds"):
        provenance.validate_boundary_tangent_path_plan(
            plan, claimed_ids=[1 << 20]
        )


def test_source_fingerprint_and_resume_compatibility(tmp_path: Path) -> None:
    source_a = tmp_path / "a.py"
    source_b = tmp_path / "b.py"
    source_a.write_text("A = 1\n", encoding="utf-8")
    source_b.write_text("B = 2\n", encoding="utf-8")
    expected_source = provenance.boundary_tangent_source_fingerprint(
        (source_b, source_a, source_a)
    )
    assert expected_source == source_fingerprint((source_a.resolve(), source_b.resolve()))

    run = tmp_path / "resume"
    run.mkdir()
    plan = provenance.build_boundary_tangent_path_plan()
    config = _semantic(
        {"schema": "test-config", "schema_version": 1, "root_seed": 7}
    )
    _write_json(run / "path_id_plan.json", plan)
    _write_json(run / "scientific_config.json", config)
    _write_json(
        run / "run_manifest.json",
        {
            "source_fingerprint": expected_source,
            "scientific_config_sha256": config["semantic_sha256"],
            "path_plan_sha256": plan["semantic_sha256"],
        },
    )
    result = provenance.verify_resume_compatibility(
        run,
        scientific_config_sha256=str(config["semantic_sha256"]),
        path_plan_sha256=str(plan["semantic_sha256"]),
        source_fingerprint_value=expected_source,
    )
    assert result["passed"] == 1

    with pytest.raises(
        provenance.BoundaryTangentProvenanceError, match="manifest compatibility"
    ):
        provenance.verify_resume_compatibility(
            run,
            scientific_config_sha256=str(config["semantic_sha256"]),
            path_plan_sha256=str(plan["semantic_sha256"]),
            source_fingerprint_value="changed",
        )
