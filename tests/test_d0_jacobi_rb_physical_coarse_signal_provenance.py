from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from mnist.d0_jacobi_artifacts import source_fingerprint
from mnist.d0_jacobi_rb_physical_coarse_signal_provenance import (
    BAYES_POWER_PARENT,
    PARENT_SPECS,
    PHYSICAL_PARENT,
    ZERO_SIGNAL_PARENT,
    ParentSpec,
    verify_physical_coarse_signal_parents,
)
from mnist.d0_one_image_gate import ArtifactCompatibilityError


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scope(physical_training: int) -> dict[str, int]:
    return {
        "physical_training_performed": physical_training,
        "production_refinement_performed": 0,
        "production_refinement_authorized": 0,
        "sampling_performed": 0,
        "sampling_authorized": 0,
        "reverse_sampling_performed": 0,
        "reverse_sampling_authorized": 0,
        "reconstruction_claim_authorized": 0,
        "full_dataset_training_authorized": 0,
    }


def _fake_parent(
    root: Path,
    original: ParentSpec,
    *,
    physical: ParentSpec | None = None,
) -> tuple[Path, ParentSpec]:
    run = root / original.basename
    run.mkdir(parents=True)
    semantic = hashlib.sha256((original.role + "-registry").encode()).hexdigest()
    config = hashlib.sha256((original.role + "-config").encode()).hexdigest()
    live_source = root / "live_sources" / f"{original.role}.py"
    live_source.parent.mkdir(parents=True, exist_ok=True)
    live_source.write_text(
        f'"""Immutable fixture source for {original.role}."""\n',
        encoding="utf-8",
    )
    relative_source = live_source.relative_to(Path.cwd())
    if original.role == PHYSICAL_PARENT.role:
        source_paths = [relative_source.as_posix()]
        source = source_fingerprint([relative_source.resolve()])
    elif original.role == ZERO_SIGNAL_PARENT.role:
        source_paths = [relative_source.as_posix()]
        source = source_fingerprint([relative_source])
    else:
        source_paths = [str(live_source.resolve())]
        source = source_fingerprint([live_source.resolve()])

    _write_json(
        run / "run_manifest.json",
        {
            "schema": original.run_schema,
            "schema_version": 1,
            "source_fingerprint": source,
            "source_paths": source_paths,
            "scientific_config_sha256": config,
        },
    )
    _write_json(
        run / "scientific_config.json",
        {
            "schema": original.config_schema,
            "schema_version": 1,
            "semantic_sha256": config,
        },
    )
    decision = {
        "schema": original.decision_schema,
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "decision": original.decision,
        **_scope(original.physical_training_performed),
    }
    if original.role == PHYSICAL_PARENT.role:
        pass
    elif original.role == ZERO_SIGNAL_PARENT.role:
        decision.update(
            {
                "diagnostic_conclusion": "frozen_model_does_not_beat_zero",
                "conditional_mean_identically_zero_proven": 0,
                "population_signal_absence_proven": 0,
            }
        )
    else:
        decision["fresh_physical_witness_planning_authorized"] = 1
    _write_json(run / original.decision_path, decision)

    for gate in original.gates:
        value: dict[str, Any] = {
            "schema": gate.schema,
            "schema_version": 1,
            "evaluation_status": "evaluated",
            "passed": gate.passed,
        }
        if (
            original.role == PHYSICAL_PARENT.role
            and gate.path == "confirmation_gate.json"
        ):
            value["subchecks"] = {
                "aggregate_model_beats_zero": {"passed": 0},
                "other": {"passed": 1},
            }
        _write_json(run / gate.path, value)

    if physical is not None:
        _write_json(
            run / "parent_provenance.json",
            {
                "parent_basename": physical.basename,
                "registry_record_count": physical.registry_record_count,
                "registry_sha256": physical.registry_sha256,
                "registry_file_sha256": physical.registry_file_sha256,
                "source_fingerprint": physical.source_fingerprint,
                "scientific_config_sha256": physical.scientific_config_sha256,
            },
        )

    artifacts = sorted(
        path
        for path in run.iterdir()
        if path.name not in {"artifact_registry.json", "run_status.json"}
    )
    records = [
        {
            "path": path.name,
            "sha256": _sha(path),
            "size": path.stat().st_size,
        }
        for path in artifacts
    ]
    registry = {
        "schema": original.registry_schema,
        "schema_version": 1,
        "record_count": len(records),
        "records": records,
        "registry_sha256": semantic,
        **_scope(original.physical_training_performed),
    }
    _write_json(run / "artifact_registry.json", registry)
    registry_file_sha = _sha(run / "artifact_registry.json")
    spec = replace(
        original,
        registry_record_count=len(records),
        registry_sha256=semantic,
        registry_file_sha256=registry_file_sha,
        source_fingerprint=source,
        scientific_config_sha256=config,
    )
    _write_json(
        run / "run_status.json",
        {
            "schema": original.status_schema,
            "schema_version": 1,
            "state": original.terminal_state,
            "stage": original.terminal_stage,
            "decision": original.terminal_status_decision,
            "artifact_registry_record_count": len(records),
            "artifact_registry_sha256": semantic,
            "artifact_registry_file_sha256": registry_file_sha,
            "artifact_registry_file_size": (run / "artifact_registry.json").stat().st_size,
            **_scope(original.physical_training_performed),
        },
    )
    return run, spec


def _fake_three_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    monkeypatch.chdir(tmp_path)
    physical_run, physical = _fake_parent(tmp_path, PHYSICAL_PARENT)
    zero_run, zero = _fake_parent(
        tmp_path, ZERO_SIGNAL_PARENT, physical=physical
    )
    bayes_run, bayes = _fake_parent(
        tmp_path, BAYES_POWER_PARENT, physical=physical
    )
    monkeypatch.setitem(PARENT_SPECS, "physical_one_image", physical)
    monkeypatch.setitem(PARENT_SPECS, "zero_signal_diagnostic", zero)
    monkeypatch.setitem(PARENT_SPECS, "bayes_power_calibration", bayes)
    return physical_run, zero_run, bayes_run


def test_frozen_production_bindings_are_exact() -> None:
    assert PHYSICAL_PARENT.registry_record_count == 544
    assert (
        PHYSICAL_PARENT.registry_sha256
        == "5e0b46328b6783614bdb7d394587b32e63d2d33b76f0279abdab6ecdf7d4e18a"
    )
    assert ZERO_SIGNAL_PARENT.registry_record_count == 18
    assert (
        ZERO_SIGNAL_PARENT.registry_sha256
        == "11d0a7272dd83b6535c1bc4426ad471f929ec0a1cd2f9c96e8ac80f01483a5e3"
    )
    assert BAYES_POWER_PARENT.registry_record_count == 74
    assert (
        BAYES_POWER_PARENT.registry_sha256
        == "01b5d772299611e9e17b886658b7eba80a7ab50805241e94d2e9a8ba36562e79"
    )
    assert PHYSICAL_PARENT.physical_training_performed == 1
    assert ZERO_SIGNAL_PARENT.physical_training_performed == 0
    assert BAYES_POWER_PARENT.physical_training_performed == 0


def test_three_parent_verifier_returns_complete_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    physical, zero, bayes = _fake_three_parents(tmp_path, monkeypatch)
    result = verify_physical_coarse_signal_parents(
        physical_run_dir=physical,
        zero_signal_run_dir=zero,
        bayes_power_run_dir=bayes,
    )
    assert result["passed"] == 1
    assert result["physical_parent_shared_by_descendants"] == 1
    assert set(result["parents"]) == {
        "physical_one_image",
        "zero_signal_diagnostic",
        "bayes_power_calibration",
    }
    assert result["parents"]["physical_one_image"]["physical_training_performed"] == 1
    assert result["parents"]["zero_signal_diagnostic"]["sampling_performed"] == 0
    assert result["parents"]["bayes_power_calibration"]["terminal"]["decision"] == (
        "noisy_bayes_detection_pipeline_calibrated"
    )


def test_registered_artifact_tampering_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    physical, zero, bayes = _fake_three_parents(tmp_path, monkeypatch)
    (bayes / "controls_gate.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match="artifact (size|hash) changed"):
        verify_physical_coarse_signal_parents(
            physical_run_dir=physical,
            zero_signal_run_dir=zero,
            bayes_power_run_dir=bayes,
        )


def test_missing_parent_registry_is_a_typed_provenance_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    physical, zero, bayes = _fake_three_parents(tmp_path, monkeypatch)
    (zero / "artifact_registry.json").unlink()
    with pytest.raises(
        ArtifactCompatibilityError,
        match="missing zero_signal_diagnostic artifact registry",
    ):
        verify_physical_coarse_signal_parents(
            physical_run_dir=physical,
            zero_signal_run_dir=zero,
            bayes_power_run_dir=bayes,
        )


def test_live_source_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    physical, zero, bayes = _fake_three_parents(tmp_path, monkeypatch)
    source_path = tmp_path / "live_sources" / "bayes_power_calibration.py"
    source_path.write_text("# drift after the immutable run\n", encoding="utf-8")
    with pytest.raises(
        ArtifactCompatibilityError,
        match="bayes_power_calibration live source fingerprint changed",
    ):
        verify_physical_coarse_signal_parents(
            physical_run_dir=physical,
            zero_signal_run_dir=zero,
            bayes_power_run_dir=bayes,
        )


def test_wrong_parent_directory_and_transitive_binding_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    physical, zero, bayes = _fake_three_parents(tmp_path, monkeypatch)
    renamed = tmp_path / "wrong-physical"
    physical.rename(renamed)
    with pytest.raises(ArtifactCompatibilityError, match="wrong physical_one_image"):
        verify_physical_coarse_signal_parents(
            physical_run_dir=renamed,
            zero_signal_run_dir=zero,
            bayes_power_run_dir=bayes,
        )
