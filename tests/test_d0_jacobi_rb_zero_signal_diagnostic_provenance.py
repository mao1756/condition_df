from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnist.d0_jacobi_artifacts import config_fingerprint, file_fingerprint
import mnist.d0_jacobi_rb_zero_signal_diagnostic_provenance as provenance


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fake_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    run = tmp_path / "sealed-parent"
    run.mkdir()
    monkeypatch.setattr(provenance, "EXPECTED_PARENT_BASENAME", run.name)
    monkeypatch.setattr(provenance, "EXPECTED_SOURCE_FINGERPRINT", "source")
    monkeypatch.setattr(provenance, "source_fingerprint", lambda paths: "source")
    monkeypatch.setattr(
        provenance, "EXPECTED_SCIENTIFIC_CONFIG_SHA256", "scientific"
    )
    monkeypatch.setattr(provenance, "EXPECTED_SELECTED_STATE_SHA256", "state")
    monkeypatch.setattr(provenance, "EXPECTED_CONFIRMATION_SEAL_SHA256", "seal")

    selected_model = run / "selected_model.pt"
    selected_model.write_bytes(b"frozen model")
    model_hash = file_fingerprint(selected_model)
    monkeypatch.setattr(
        provenance, "EXPECTED_SELECTED_MODEL_FILE_SHA256", model_hash
    )
    baseline = run / "metadata_baseline.npz"
    baseline.write_bytes(b"baseline")
    baseline_hash = file_fingerprint(baseline)

    passed = {"evaluation_status": "evaluated", "passed": 1}
    for name in (
        "preflight_gate.json",
        "cache_gate.json",
        "teacher_gate.json",
        "physical_gate.json",
        "train_cache_gate.json",
        "validation_cache_gate.json",
        "confirmation_cache_gate.json",
    ):
        _write_json(run / name, passed)
    _write_json(
        run / "confirmation_gate.json",
        {
            "evaluation_status": "evaluated",
            "passed": 0,
            "subchecks": {
                "aggregate_model_beats_zero": {"passed": 0},
                "everything_else": {"passed": 1},
            },
        },
    )
    _write_json(
        run / "run_manifest.json",
        {
            "source_fingerprint": "source",
            "source_paths": ["mnist/d0_jacobi_rb_learnability.py"],
            "scientific_config_sha256": "scientific",
            "sampling_performed": 0,
            "reverse_sampling_performed": 0,
            "production_refinement_performed": 0,
        },
    )
    _write_json(
        run / "learnability_decision.json",
        {
            "evaluation_status": "evaluated",
            "decision": provenance.EXPECTED_DECISION,
            "sampling_performed": 0,
            "reverse_sampling_performed": 0,
            "production_refinement_performed": 0,
        },
    )
    _write_json(
        run / "confirmation_seal.json",
        {
            "seal_sha256": "seal",
            "selected_model_file_sha256": model_hash,
            "selected_state_sha256": "state",
            "metadata_baseline_file_sha256": baseline_hash,
        },
    )
    _write_json(
        run / "selected_model.json",
        {
            "checkpoint_file_sha256": model_hash,
            "state_sha256": "state",
        },
    )
    _write_json(run / "confirmation_open.json", {"opened_count": 1})
    for split in ("train", "validation", "confirmation"):
        for suffix in ("inputs.npz", "labels_audit.npz"):
            path = run / "cache" / f"{split}_{suffix}"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(f"{split}:{suffix}".encode())

    records = []
    for path in sorted(run.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and path.name not in {
            "artifact_registry.json",
            "run_status.json",
        }:
            records.append(
                {
                    "path": path.relative_to(run).as_posix(),
                    "sha256": file_fingerprint(path),
                    "size": path.stat().st_size,
                }
            )
    registry_sha = config_fingerprint(records)
    monkeypatch.setattr(provenance, "EXPECTED_REGISTRY_COUNT", len(records))
    monkeypatch.setattr(provenance, "EXPECTED_REGISTRY_SHA256", registry_sha)
    _write_json(
        run / "artifact_registry.json",
        {
            "record_count": len(records),
            "records": records,
            "registry_sha256": registry_sha,
        },
    )
    registry_file = run / "artifact_registry.json"
    _write_json(
        run / "run_status.json",
        {
            "stage": "confirm",
            "state": "gate_failed",
            "decision": provenance.EXPECTED_DECISION,
            "artifact_registry_record_count": len(records),
            "artifact_registry_sha256": registry_sha,
            "artifact_registry_file_sha256": file_fingerprint(registry_file),
            "artifact_registry_file_size": registry_file.stat().st_size,
            "sampling_performed": 0,
            "reverse_sampling_performed": 0,
            "production_refinement_performed": 0,
        },
    )
    return run


def test_exact_parent_verification_and_tamper_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _fake_parent(tmp_path, monkeypatch)
    record = provenance.verify_zero_signal_parent(run)
    assert record["parent_read_only"] == 1
    assert record["confirmation_failed_subchecks"] == [
        "aggregate_model_beats_zero"
    ]
    assert record["confirmation_opened_count"] == 1

    (run / "cache" / "train_inputs.npz").write_bytes(b"tampered")
    with pytest.raises(provenance.ZeroSignalParentError, match="changed"):
        provenance.verify_zero_signal_parent(run)
