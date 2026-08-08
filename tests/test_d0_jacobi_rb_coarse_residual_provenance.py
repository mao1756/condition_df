from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_rb_coarse_residual_provenance import (
    FAILED_LEARNER_PARENT,
    MIXED_TARGET_SHA256,
    PARENT_SPECS,
    SELECTED_OUTER_STEPS,
    SOURCE_IMAGE_SHA256,
    WITNESS_PARENT,
    ParentSpec,
    verify_coarse_residual_parents,
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _scope(training: int) -> dict[str, int]:
    return {
        "physical_training_performed": training,
        "production_refinement_performed": 0,
        "sampling_performed": 0,
        "sampling_authorized": 0,
        "reverse_sampling_performed": 0,
        "reverse_sampling_authorized": 0,
        "reconstruction_claim_authorized": 0,
        "full_dataset_training_authorized": 0,
    }


def _config(spec: ParentSpec) -> dict[str, Any]:
    source = {
        "image_sha256": SOURCE_IMAGE_SHA256,
        "mixed_target_sha256": MIXED_TARGET_SHA256,
        "lambda_mix": 0.35,
    }
    body: dict[str, Any] = {
        "schema": spec.config_schema,
        "schema_version": 1,
        "grid_size": 28,
        "alpha": 1.0,
        "outer_steps": 512,
        "edges_per_phase": 392,
        "phase_matchings": [0, 1, 2, 3, 2, 1, 0],
        "phase_durations": [0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5],
        "source_image": source,
        **_scope(0),
    }
    if spec.role == WITNESS_PARENT.role:
        body.update(
            tau_eff=5.0e-5,
            analysis={"selected_outer_steps": list(SELECTED_OUTER_STEPS)},
        )
    else:
        body["selected_outer_steps"] = list(SELECTED_OUTER_STEPS)
    return body


def _finalize_parent(
    root: Path,
    spec: ParentSpec,
    *,
    learner_binding: dict[str, Any] | None = None,
) -> tuple[Path, ParentSpec]:
    run = root / spec.basename
    run.mkdir(parents=True)
    source = root / "sources" / f"{spec.role}.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(f'"""Fixture source for {spec.role}."""\n', encoding="utf-8")
    source_sha = source_fingerprint([source.resolve()])

    config = _config(spec)
    config_sha = config_fingerprint(
        {key: value for key, value in config.items() if key != "semantic_sha256"}
    )
    config["semantic_sha256"] = config_sha
    _write(run / "scientific_config.json", config)
    _write(
        run / "run_manifest.json",
        {
            "schema": spec.run_schema,
            "schema_version": 1,
            "source_paths": [str(source.resolve())],
            "source_fingerprint": source_sha,
            "scientific_config_sha256": config_sha,
            **_scope(0),
        },
    )
    decision = {
        "schema": spec.decision_schema,
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "decision": spec.decision,
        **_scope(spec.physical_training_performed),
    }
    if spec.role == WITNESS_PARENT.role:
        decision["recommended_next_action"] = (
            "plan a coarse-baseline plus exact-RB residual learner with "
            "unweighted MSE against the unchanged exact label"
        )
        _write(
            run / "physical_coarse_signal_analysis.json",
            {
                "classification": {
                    "decision": spec.decision,
                    "bootstrap_lower_bound": 0.0005,
                    "welch_lower_bound": 0.0005,
                },
                "bootstrap": {"point_estimate": 0.000648},
                "lower_bound_on_full_allowed_input_conditional_mean_energy": 1,
                "conditional_mean_identically_zero_proven": 0,
            },
        )
        assert learner_binding is not None
        _write(
            run / "parent_provenance.json",
            {
                "parents": {
                    "physical_one_image": {
                        "basename": learner_binding["basename"],
                        "source_fingerprint": learner_binding[
                            "source_fingerprint"
                        ],
                        "scientific_config_sha256": learner_binding[
                            "scientific_config_sha256"
                        ],
                        "registry": learner_binding["registry"],
                        "terminal": {
                            "decision": FAILED_LEARNER_PARENT.decision
                        },
                        "verified": 1,
                    }
                }
            },
        )
    _write(run / spec.decision_path, decision)

    for gate_spec in spec.gates:
        gate: dict[str, Any] = {
            "schema": gate_spec.schema,
            "schema_version": 1,
            "evaluation_status": "evaluated",
            "passed": gate_spec.passed,
            **_scope(
                spec.physical_training_performed
                if gate_spec.path in {"physical_gate.json", "confirmation_gate.json"}
                else 0
            ),
        }
        if (
            spec.role == FAILED_LEARNER_PARENT.role
            and gate_spec.path == "confirmation_gate.json"
        ):
            gate["subchecks"] = {
                "aggregate_model_beats_zero": {"passed": 0},
                "other": {"passed": 1},
            }
        _write(run / gate_spec.path, gate)

    files = sorted(
        item
        for item in run.rglob("*")
        if item.is_file()
        and item.name not in {"artifact_registry.json", "run_status.json"}
    )
    records = [
        {
            "path": item.relative_to(run).as_posix(),
            "size": item.stat().st_size,
            "sha256": file_fingerprint(item),
        }
        for item in files
    ]
    registry_sha = config_fingerprint(records)
    _write(
        run / "artifact_registry.json",
        {
            "schema": spec.registry_schema,
            "schema_version": 1,
            "record_count": len(records),
            "records": records,
            "registry_sha256": registry_sha,
            **_scope(spec.physical_training_performed),
        },
    )
    registry_file_sha = file_fingerprint(run / "artifact_registry.json")
    updated = replace(
        spec,
        registry_record_count=len(records),
        registry_sha256=registry_sha,
        registry_file_sha256=registry_file_sha,
        source_fingerprint=source_sha,
        scientific_config_sha256=config_sha,
    )
    _write(
        run / "run_status.json",
        {
            "schema": spec.status_schema,
            "schema_version": 1,
            "state": spec.terminal_state,
            "stage": spec.terminal_stage,
            "decision": spec.decision,
            "artifact_registry_record_count": len(records),
            "artifact_registry_sha256": registry_sha,
            "artifact_registry_file_sha256": registry_file_sha,
            "artifact_registry_file_size": (
                run / "artifact_registry.json"
            ).stat().st_size,
            **_scope(spec.physical_training_performed),
        },
    )
    return run, updated


def _parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    learner_run, learner = _finalize_parent(
        tmp_path, FAILED_LEARNER_PARENT
    )
    learner_binding = {
        "basename": learner.basename,
        "source_fingerprint": learner.source_fingerprint,
        "scientific_config_sha256": learner.scientific_config_sha256,
        "registry": {
            "record_count": learner.registry_record_count,
            "sha256": learner.registry_sha256,
            "file_sha256": learner.registry_file_sha256,
        },
    }
    witness_run, witness = _finalize_parent(
        tmp_path, WITNESS_PARENT, learner_binding=learner_binding
    )
    monkeypatch.setitem(PARENT_SPECS, FAILED_LEARNER_PARENT.role, learner)
    monkeypatch.setitem(PARENT_SPECS, WITNESS_PARENT.role, witness)
    return witness_run, learner_run


def test_frozen_production_parent_constants() -> None:
    assert WITNESS_PARENT.registry_record_count == 2_616
    assert (
        WITNESS_PARENT.registry_sha256
        == "ca405ea7c00d6efed470c0651b8ad28d31f797cf142a1bef5f75d464ee1c3ab3"
    )
    assert FAILED_LEARNER_PARENT.registry_record_count == 544
    assert (
        FAILED_LEARNER_PARENT.registry_sha256
        == "5e0b46328b6783614bdb7d394587b32e63d2d33b76f0279abdab6ecdf7d4e18a"
    )


def test_verifier_binds_both_parents_and_shared_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    witness, learner = _parents(tmp_path, monkeypatch)
    record = verify_coarse_residual_parents(
        witness_run_dir=witness,
        failed_learner_run_dir=learner,
    )
    assert record["passed"] == 1
    assert record["coarse_signal_detected_pass"] == 1
    assert record["only_aggregate_model_beats_zero_failed_pass"] == 1
    assert record["same_source_image_pass"] == 1
    assert record["same_exact_k512_kernel_pass"] == 1
    assert record["same_selected_outer_steps_pass"] == 1
    assert record["transitive_provenance_pass"] == 1
    assert record["sampling_performed"] == 0


def test_registered_artifact_tampering_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    witness, learner = _parents(tmp_path, monkeypatch)
    (witness / "physical_coarse_signal_analysis.json").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(
        ArtifactCompatibilityError, match="registered artifact changed"
    ):
        verify_coarse_residual_parents(
            witness_run_dir=witness,
            failed_learner_run_dir=learner,
        )


def test_extra_parent_artifact_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    witness, learner = _parents(tmp_path, monkeypatch)
    (learner / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        ArtifactCompatibilityError, match="registry file set changed"
    ):
        verify_coarse_residual_parents(
            witness_run_dir=witness,
            failed_learner_run_dir=learner,
        )


def test_transitive_learner_binding_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    witness, learner = _parents(tmp_path, monkeypatch)
    # Mutate before re-finalizing would be the normal fixture route; direct
    # mutation is enough here because the verifier must first reject bytes.
    provenance = json.loads(
        (witness / "parent_provenance.json").read_text(encoding="utf-8")
    )
    provenance["parents"]["physical_one_image"]["registry"]["sha256"] = "0" * 64
    _write(witness / "parent_provenance.json", provenance)
    with pytest.raises(ArtifactCompatibilityError):
        verify_coarse_residual_parents(
            witness_run_dir=witness,
            failed_learner_run_dir=learner,
        )


def test_wrong_parent_basename_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    witness, learner = _parents(tmp_path, monkeypatch)
    wrong = tmp_path / "wrong-parent"
    witness.rename(wrong)
    with pytest.raises(ArtifactCompatibilityError, match="wrong .* basename"):
        verify_coarse_residual_parents(
            witness_run_dir=wrong,
            failed_learner_run_dir=learner,
        )
