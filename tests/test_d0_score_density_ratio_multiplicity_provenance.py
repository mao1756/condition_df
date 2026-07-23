from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import mnist.d0_score_density_ratio_multiplicity_provenance as provenance
from mnist.d0_one_image_gate import ArtifactCompatibilityError, file_fingerprint


PRODUCTION_PARENT = Path(
    "runs/experiment12_d0_score_density_ratio_selection_power_confirmation/"
    "20260720-204514_production-density-ratio-selection-power"
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record(path: Path) -> dict[str, object]:
    return {"sha256": file_fingerprint(path), "size": int(path.stat().st_size)}


def test_registry_requires_exact_123_records_and_detects_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "parent"
    root.mkdir()
    records: dict[str, object] = {}
    for index in range(provenance.PARENT_REGISTRY_RECORD_COUNT):
        path = root / f"artifact-{index:03d}.json"
        _write(path, {"index": index})
        records[path.name] = _record(path)
    registry_path = root / "artifact_registry.json"
    status_path = root / "run_status.json"
    _write(
        registry_path,
        {
            "schema": provenance.PARENT_REGISTRY_SCHEMA,
            "schema_version": 1,
            "terminal_files_excluded_to_avoid_self_reference": [
                "artifact_registry.json",
                "run_status.json",
            ],
            "records": records,
        },
    )
    _write(
        status_path,
        {
            "artifact_registry_sha256": file_fingerprint(registry_path),
            "artifact_registry_size": registry_path.stat().st_size,
        },
    )
    result = provenance._verify_registry(
        root, registry_path=registry_path, status_path=status_path
    )
    assert len(result["records"]) == 123

    with (root / "artifact-100.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ArtifactCompatibilityError, match="registered selection-power"):
        provenance._verify_registry(
            root, registry_path=registry_path, status_path=status_path
        )


def test_transitive_parent_requires_exact_125_332_222_381_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recomputed = {
        "schema": "recomputed-schema",
        "schema_version": 1,
        "passed": 1,
        "run_dir": "upstream",
        "terminal_registry_record_count": 125,
        "transitive_parent_provenance": {
            "terminal_registry_record_count": 332,
            "transitive_parent_provenance": {
                "terminal_registry_record_count": 222,
                "transitive_parent_provenance": {
                    "terminal_registry_record_count": 381,
                },
            },
        },
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    monkeypatch.setattr(
        provenance, "verify_parent_normalized_head_run", lambda path: recomputed
    )
    stored = {
        **recomputed,
        "schema": provenance.PARENT_RUN_SCHEMA + "-parent-provenance",
        "verifier_source_fingerprint": "source",
    }
    assert provenance._verify_transitive_parent(stored) == recomputed

    stored["transitive_parent_provenance"] = {
        "terminal_registry_record_count": 332,
        "transitive_parent_provenance": {
            "terminal_registry_record_count": 222,
            "transitive_parent_provenance": {
                "terminal_registry_record_count": 380,
            },
        },
    }
    with pytest.raises(
        ArtifactCompatibilityError,
        match="transitive normalized-head provenance changed|125/332/222/381",
    ):
        provenance._verify_transitive_parent(stored)


def test_zero_clipping_verifier_reads_every_training_row(tmp_path: Path) -> None:
    history = tmp_path / "training_history.csv"
    windows = tmp_path / "clipping_windows.csv"
    with history.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", "clipped"))
        writer.writeheader()
        for step in range(1, 2001):
            writer.writerow({"step": step, "clipped": int(step == 1999)})
    with windows.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("start_step", "stop_step", "clip_fraction")
        )
        writer.writeheader()
        writer.writerow(
            {"start_step": 1, "stop_step": 2000, "clip_fraction": 0.0}
        )
    metrics = {
        "post_warmup_clip_fraction": 0.0,
        "final_500_clip_fraction": 0.0,
        "final_200_clip_fraction": 0.0,
        "optimization_diagnostics": {
            "post_warmup_clip_fraction": 0.0,
            "final_500_clip_fraction": 0.0,
            "final_200_clip_fraction": 0.0,
            "clipping_windows": [
                {"start_step": 1, "stop_step": 2000, "clip_fraction": 0.0}
            ],
        },
    }
    with pytest.raises(ArtifactCompatibilityError, match="history contains clipping"):
        provenance._verify_zero_clipping(
            metrics,
            history_path=history,
            windows_path=windows,
            description="fixture",
        )


def test_a_only_failure_shape_is_exact() -> None:
    gate = {
        "subchecks": {
            "null_panel_a_lower_bounds": {"passed": 0},
            "null_panel_b_lower_bounds": {"passed": 1},
            "optimizer_health": {"passed": 1},
        }
    }
    assert provenance._failed_subchecks(gate) == ["null_panel_a_lower_bounds"]
    gate["subchecks"]["optimizer_health"]["passed"] = 0
    assert provenance._failed_subchecks(gate) == [
        "null_panel_a_lower_bounds",
        "optimizer_health",
    ]


def test_terminal_verifier_rejects_wrong_decision_before_registry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "parent"
    root.mkdir()
    for relative in provenance._REQUIRED_FILES.values():
        _write(root / relative, {})
    _write(
        root / "run_manifest.json",
        {
            "schema": provenance.PARENT_RUN_SCHEMA,
            "schema_version": 1,
            "run_dir": str(root.resolve()),
        },
    )
    _write(
        root / "run_status.json",
        {
            "schema": provenance.PARENT_RUN_SCHEMA,
            "schema_version": 1,
            "status": "complete",
            "outcome": "gate_failed",
            "phase": "pilot",
            "stage": "pilot",
            "decision": "classification_power_confirmation_unresolved",
            "required_gate": "pilot",
            "required_gate_pass": 0,
        },
    )
    with pytest.raises(ArtifactCompatibilityError, match="multiplicity precursor"):
        provenance.verify_parent_selection_power_run(root)


def test_real_finished_123_record_parent_is_accepted_when_available() -> None:
    if not PRODUCTION_PARENT.is_dir():
        pytest.skip("production selection-power parent is not part of this checkout")
    result = provenance.verify_parent_selection_power_run(PRODUCTION_PARENT)
    assert result["passed"] == 1
    assert result["terminal_registry_record_count"] == 123
    assert result["lineage_registry_record_counts"] == [125, 332, 222, 381]
    assert result["pilot_candidate_count"] == 2
    assert result["pilot_task_count"] == 4
    assert result["a_only_failure_count"] == 2
    assert result["sealed_b_rejection_count"] == 2
    assert result["all_tasks_complete_finite_boundary_admissible"] == 1
    assert result["all_task_clipping_zero"] == 1
    assert result["task_failure_count"] == 0
    assert result["selected_profile"] == 0
    assert result["confirmation_performed"] == 0
    assert result["physical_training_performed"] == 0
    assert result["sampling_performed"] == 0
