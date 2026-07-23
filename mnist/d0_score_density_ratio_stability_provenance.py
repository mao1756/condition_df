"""Strict provenance for paired-mixture density-ratio stability controls.

The additive stability workflow is authorized by one immutable parent result:
the density-ratio preflight passed, every pilot task completed finitely, the
stationary null behaved correctly, but no learning-rate profile satisfied the
frozen clipping and teacher-signal contract.  This verifier binds the exact
222-record terminal registry, recomputes the parent gates, verifies all eight
task/status pairs, and recursively verifies the 381-record streamed-control
parent.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from mnist.d0_one_image_gate import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_score_density_ratio_gate import (
    DensityRatioThresholds,
    evaluate_density_ratio_pilot,
    evaluate_density_ratio_workflow,
    evaluate_ratio_preflight,
)
from mnist.d0_score_density_ratio_provenance import verify_parent_stability_run


__all__ = [
    "PARENT_RUN_SCHEMA",
    "PARENT_REGISTRY_SCHEMA",
    "PARENT_REGISTRY_RECORD_COUNT",
    "PARENT_LOSS_SCALE",
    "EXPECTED_PARENT_DECISION",
    "verify_parent_density_ratio_run",
]


PARENT_RUN_SCHEMA = "experiment12-d0-score-density-ratio-controls"
PARENT_REGISTRY_SCHEMA = PARENT_RUN_SCHEMA + "-artifact-registry"
PARENT_REGISTRY_RECORD_COUNT = 222
PARENT_LOSS_SCALE = 0.05173607018770852
EXPECTED_PARENT_DECISION = "classification_optimizer_unresolved"
EXPECTED_PARENT_MODEL_SEED = 8463954073261664671
EXPECTED_PILOT_LEARNING_RATES = (3e-4, 1e-4, 3e-5, 1e-5)
EXPECTED_B_CONFIRMED_RATES = {3e-5, 1e-5}
EXPECTED_KERNEL: dict[str, Any] = {
    "grid_size": 28,
    "sample_steps": 512,
    "reference_substeps": 256,
    "tau_eff": 5e-5,
    "edge_alpha_mode": "alpha_eff",
    "alpha_eff": 1.0,
    "mass_floor": 1e-7,
    "limiter_fraction": 1.0,
    "lambda_mix": 0.35,
}

_REQUIRED_FILES: dict[str, str] = {
    "manifest": "run_manifest.json",
    "status": "run_status.json",
    "registry": "artifact_registry.json",
    "parent_provenance": "parent_provenance.json",
    "stream_plan": "density_ratio_stream_plan.json",
    "stream_replay": "density_ratio_stream_replay_preflight.json",
    "preflight_raw": "density_ratio_preflight.json",
    "preflight_gate": "density_ratio_preflight_gate.json",
    "calibration": "density_ratio_loss_scale_calibration.json",
    "pilot_panels": "pilot_panel_registry.json",
    "pilot_candidates": "pilot_candidate_registry.json",
    "pilot_gate": "density_ratio_pilot_gate.json",
    "pilot_failures": "pilot_task_failures.json",
    "report": "density_ratio_control_report.json",
    "decision": "density_ratio_control_decision.json",
}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"cannot read density-ratio parent artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(
            f"density-ratio parent artifact is not a JSON object: {path}"
        )
    return dict(value)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactCompatibilityError(message)


def _one(value: Any) -> bool:
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return False


def _zero(value: Any) -> bool:
    try:
        return int(value) == 0
    except (TypeError, ValueError):
        return False


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return _finite(actual) and math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-15
        )
    return actual == expected


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_fingerprint(path),
        "size": int(path.stat().st_size),
    }


def _no_work(value: Mapping[str, Any], description: str) -> None:
    _require(
        _zero(value.get("physical_training_performed", 0))
        and _zero(value.get("sampling_performed", 0)),
        f"{description} records physical training or sampling",
    )


def _verify_record(
    raw: Mapping[str, Any],
    *,
    root: Path,
    expected_path: Path,
    description: str,
) -> None:
    path = expected_path.resolve()
    raw_path = raw.get("path")
    if raw_path is not None:
        _require(Path(str(raw_path)).resolve() == path, f"{description} path mismatch")
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtifactCompatibilityError(f"{description} escapes the parent run") from exc
    _require(path.is_file(), f"{description} is missing")
    _require(
        raw.get("sha256") == file_fingerprint(path)
        and int(raw.get("size", -1)) == int(path.stat().st_size),
        f"{description} hash or size mismatch",
    )


def _verify_registry(
    run_dir: Path,
    *,
    registry_path: Path,
    status_path: Path,
    expected_count: int = PARENT_REGISTRY_RECORD_COUNT,
) -> dict[str, Any]:
    registry = _load(registry_path)
    status = _load(status_path)
    _require(
        registry.get("schema") == PARENT_REGISTRY_SCHEMA
        and int(registry.get("schema_version", -1)) == 1,
        "density-ratio terminal registry schema is incompatible",
    )
    _require(
        status.get("artifact_registry_sha256") == file_fingerprint(registry_path)
        and int(status.get("artifact_registry_size", -1))
        == int(registry_path.stat().st_size),
        "density-ratio status does not bind its terminal registry",
    )
    exclusions = set(registry.get("terminal_files_excluded_to_avoid_self_reference", []))
    _require(
        exclusions == {"artifact_registry.json", "run_status.json"},
        "density-ratio registry exclusions are incompatible",
    )
    raw_records = registry.get("records", {})
    _require(isinstance(raw_records, Mapping), "density-ratio registry records are invalid")
    records = dict(raw_records)
    _require(
        len(records) == int(expected_count),
        f"density-ratio terminal registry must contain exactly {expected_count} records",
    )
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in exclusions | {"artifact_registry.json"}
    }
    _require(
        set(records) == actual,
        "density-ratio registry is incomplete or contains stale records",
    )
    for relative, raw in records.items():
        _require(isinstance(raw, Mapping), f"invalid registry record {relative}")
        _verify_record(
            dict(raw),
            root=run_dir,
            expected_path=run_dir / relative,
            description=f"registered density-ratio artifact {relative}",
        )
    return registry


def _verify_scientific_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw = manifest.get("scientific_config", {})
    _require(isinstance(raw, Mapping), "density-ratio scientific configuration is missing")
    scientific = dict(raw)
    _require(
        scientific.get("algorithm") == PARENT_RUN_SCHEMA
        and int(scientific.get("algorithm_version", -1)) == 1,
        "density-ratio parent algorithm is incompatible",
    )
    _require(
        config_fingerprint(scientific) == manifest.get("scientific_fingerprint"),
        "density-ratio scientific fingerprint is inconsistent",
    )
    kernel = scientific.get("kernel", {})
    _require(isinstance(kernel, Mapping), "density-ratio kernel is missing")
    for key, expected in EXPECTED_KERNEL.items():
        _require(_same(kernel.get(key), expected), f"density-ratio kernel mismatch for {key}")
    _require(
        scientific.get("thresholds")
        == _json_value(DensityRatioThresholds().to_dict()),
        "density-ratio scientific thresholds changed",
    )
    _require(
        scientific.get("model_schema") == "d0-boundary-smooth-potential-unet-v1"
        and int(scientific.get("model_schema_version", -1)) == 1
        and scientific.get("objective_version") == "d0-balanced-raw-logit-bce-v1"
        and scientific.get("panel_version") == "d0-density-ratio-panel-v1",
        "density-ratio model/objective schema changed",
    )
    optimization = scientific.get("optimization", {})
    _require(isinstance(optimization, Mapping), "density-ratio optimizer configuration is missing")
    for key, expected in {
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "ema_decay": 0.99,
        "grad_clip": 1.0,
        "clip_warmup_steps": 500,
        "calibration_state_count": 256,
        "initial_grad_target": 0.1,
        "adaptive_loss_scaling": 0,
    }.items():
        _require(_same(optimization.get(key), expected), f"density-ratio optimizer mismatch for {key}")
    pilot = scientific.get("pilot", {})
    confirmation = scientific.get("confirmation", {})
    preflight = scientific.get("preflight", {})
    stream = scientific.get("stream", {})
    _require(
        isinstance(pilot, Mapping)
        and tuple(float(value) for value in pilot.get("learning_rates", []))
        == EXPECTED_PILOT_LEARNING_RATES
        and int(pilot.get("steps", -1)) == 2000
        and int(pilot.get("selection_paths_per_panel", -1)) == 16
        and int(pilot.get("audit_paths", -1)) == 0
        and list(pilot.get("selection_panels", [])) == ["a", "b"],
        "density-ratio pilot configuration changed",
    )
    _require(
        isinstance(confirmation, Mapping)
        and tuple(int(value) for value in confirmation.get("model_seeds", []))
        == (260831, 260832, 260833)
        and int(confirmation.get("steps", -1)) == 4000
        and int(confirmation.get("selection_paths_per_panel", -1)) == 32
        and int(confirmation.get("audit_paths_per_panel", -1)) == 32
        and list(confirmation.get("audit_panels", [])) == ["c", "d"],
        "density-ratio planned confirmation configuration changed",
    )
    _require(
        isinstance(preflight, Mapping)
        and int(preflight.get("paths", -1)) == 128
        and _same(preflight.get("confidence"), 0.99),
        "density-ratio preflight configuration changed",
    )
    _require(
        isinstance(stream, Mapping)
        and int(stream.get("batch_size", -1)) == 64
        and int(stream.get("examples_per_class", -1)) == 32
        and _same(stream.get("class_prior"), 0.5)
        and list(stream.get("class_bin_counts", [])) == [4, 4, 4, 4, 16],
        "density-ratio stream configuration changed",
    )
    _require(int(scientific.get("root_seed", -1)) == 260821, "density-ratio root seed changed")
    _no_work(scientific, "density-ratio scientific configuration")
    return scientific


def _verify_transitive_parent(parent: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        parent.get("schema")
        == "experiment12-d0-score-density-ratio-controls-parent-provenance"
        and int(parent.get("schema_version", -1)) == 1
        and _one(parent.get("passed")),
        "density-ratio transitive stability provenance is incompatible",
    )
    root = Path(str(parent.get("run_dir", ""))).resolve()
    recomputed = verify_parent_stability_run(root)
    normalized = dict(parent)
    normalized.pop("verifier_source_fingerprint", None)
    normalized["schema"] = recomputed["schema"]
    _require(
        normalized == recomputed,
        "density-ratio transitive 381-record stability provenance changed",
    )
    _require(
        int(recomputed.get("terminal_registry_record_count", -1)) == 381,
        "density-ratio transitive parent does not bind 381 artifacts",
    )
    _no_work(parent, "density-ratio transitive parent")
    return recomputed


def _verify_task_result(
    result: Mapping[str, Any],
    *,
    result_path: Path,
    status_path: Path,
    expected_task: str,
    expected_learning_rate: float,
) -> None:
    _require(result_path.is_file() and status_path.is_file(), f"missing {expected_task} task artifacts")
    disk = _load(result_path)
    status = _load(status_path)
    _require(disk == dict(result), f"{expected_task} aggregate/task result mismatch")
    _require(
        status.get("status") == "complete"
        and status.get("task") == expected_task
        and status.get("task_result_sha256") == file_fingerprint(result_path),
        f"{expected_task} task status is incomplete or unbound",
    )
    _require(
        int(status.get("model_seed", -1)) == EXPECTED_PARENT_MODEL_SEED
        and int(result.get("model_seed", -1)) == EXPECTED_PARENT_MODEL_SEED,
        f"{expected_task} model seed changed",
    )
    metrics = result.get("metrics", {})
    _require(isinstance(metrics, Mapping), f"{expected_task} task metrics are missing")
    _require(
        _one(metrics.get("complete"))
        and _one(metrics.get("finite"))
        and _one(metrics.get("boundary_admissible")),
        f"{expected_task} task is incomplete, nonfinite, or boundary-inadmissible",
    )
    fingerprints = result.get("fingerprints", {})
    _require(isinstance(fingerprints, Mapping), f"{expected_task} fingerprints are missing")
    _require(
        _same(fingerprints.get("learning_rate"), expected_learning_rate)
        and _same(fingerprints.get("loss_scale"), PARENT_LOSS_SCALE)
        and int(fingerprints.get("model_seed", -1)) == EXPECTED_PARENT_MODEL_SEED
        and fingerprints.get("task") == expected_task,
        f"{expected_task} task fingerprints changed",
    )
    _require(
        dict(status.get("fingerprints", {})) == dict(fingerprints),
        f"{expected_task} task status fingerprints changed",
    )
    _no_work(result, f"{expected_task} task result")
    _no_work(status, f"{expected_task} task status")


def _verify_calibration(value: Mapping[str, Any]) -> None:
    _require(
        value.get("objective_kind") == "density_ratio_balanced_raw_logit_bce"
        and value.get("calibration_split") == "train"
        and int(value.get("calibration_state_count", -1)) == 256
        and _one(value.get("training_only"))
        and _one(value.get("shared_by_teacher_and_null")),
        "density-ratio parent calibration is not training-only and shared",
    )
    _require(
        _same(value.get("target_initial_gradient_norm"), 0.1)
        and _same(value.get("scaled_initial_gradient_norm"), 0.1)
        and _same(value.get("loss_scale"), PARENT_LOSS_SCALE)
        and _finite(value.get("unscaled_initial_gradient_norm"))
        and float(value["unscaled_initial_gradient_norm"]) > 0.0,
        "density-ratio parent loss calibration changed",
    )
    _no_work(value, "density-ratio calibration")


def _verify_pilot(run_dir: Path, values: Mapping[str, Mapping[str, Any]]) -> None:
    raw_candidates = values["pilot_candidates"].get("candidates", [])
    _require(
        isinstance(raw_candidates, Sequence) and len(raw_candidates) == 4,
        "density-ratio parent must contain exactly four pilot candidates",
    )
    candidates = [dict(value) for value in raw_candidates if isinstance(value, Mapping)]
    _require(len(candidates) == 4, "density-ratio pilot candidates are invalid")
    for index, candidate in enumerate(candidates):
        learning_rate = EXPECTED_PILOT_LEARNING_RATES[index]
        _require(
            int(candidate.get("candidate_index", -1)) == index
            and _same(candidate.get("learning_rate"), learning_rate),
            "density-ratio pilot candidate ordering changed",
        )
        for key, task, directory in (
            ("teacher", "bounded_teacher", "bounded_teacher"),
            ("null", "dirichlet_null", "dirichlet_null"),
        ):
            raw = candidate.get(key, {})
            _require(isinstance(raw, Mapping), f"density-ratio {task} result is missing")
            root = run_dir / "pilot" / f"lr-{index:02d}" / directory
            _verify_task_result(
                dict(raw),
                result_path=root / "task_result.json",
                status_path=root / "task_status.json",
                expected_task=task,
                expected_learning_rate=learning_rate,
            )
    recomputed = _json_value(evaluate_density_ratio_pilot(candidates))
    _require(recomputed == values["pilot_gate"], "density-ratio pilot gate does not recompute")
    _require(
        _zero(recomputed.get("passed"))
        and _zero(dict(recomputed.get("selected_profile", {})).get("selected")),
        "density-ratio parent unexpectedly selected a pilot profile",
    )
    for candidate, gate in zip(candidates, recomputed.get("candidate_gates", []), strict=True):
        learning_rate = float(candidate["learning_rate"])
        teacher = dict(candidate["teacher"]).get("metrics", {})
        null = dict(candidate["null"]).get("metrics", {})
        _require(isinstance(teacher, Mapping) and isinstance(null, Mapping), "pilot metrics missing")
        teacher_clip = teacher.get("post_warmup_clip_fraction")
        null_clip = null.get("post_warmup_clip_fraction")
        _require(
            _finite(teacher_clip) and float(teacher_clip) > 0.10,
            "density-ratio teacher did not cross the frozen clipping boundary",
        )
        _require(
            _finite(null_clip) and 0.0 <= float(null_clip) <= 0.10,
            "density-ratio null optimizer was not healthy",
        )
        teacher_selection = dict(teacher.get("selection", {}))
        null_selection = dict(null.get("selection", {}))
        teacher_confirmed = _one(dict(teacher_selection.get("confirmation", {})).get("accepted"))
        _require(
            teacher_confirmed == (learning_rate in EXPECTED_B_CONFIRMED_RATES),
            "density-ratio teacher panel-B replication pattern changed",
        )
        _require(
            int(null_selection.get("selected_step", -1)) == 0
            and not _one(dict(null_selection.get("confirmation", {})).get("accepted")),
            "density-ratio null no longer selects analytic zero",
        )
        _require(_zero(dict(gate).get("passed")), "density-ratio parent candidate unexpectedly passes")
    failures = values["pilot_failures"]
    _require(
        int(failures.get("failure_count", -1)) == 0
        and list(failures.get("failures", [])) == [],
        "density-ratio pilot contains task failures",
    )
    panels = values["pilot_panels"]
    _require(
        _one(dict(panels.get("disjointness", {})).get("passed"))
        and int(dict(panels.get("disjointness", {})).get("panel_count", -1)) == 4,
        "density-ratio pilot panels are not isolated",
    )


def verify_parent_density_ratio_run(path: str | Path) -> dict[str, Any]:
    """Verify the exact terminal classifier pilot authorizing stabilization."""

    run_dir = Path(path).resolve()
    _require(run_dir.is_dir(), f"density-ratio parent run does not exist: {run_dir}")
    paths = {name: run_dir / relative for name, relative in _REQUIRED_FILES.items()}
    missing = [str(value) for value in paths.values() if not value.is_file()]
    _require(not missing, "density-ratio parent artifacts are missing: " + ", ".join(missing))
    values = {name: _load(value) for name, value in paths.items() if name != "registry"}
    manifest, status = values["manifest"], values["status"]
    _require(
        manifest.get("schema") == status.get("schema") == PARENT_RUN_SCHEMA
        and int(manifest.get("schema_version", -1)) == 1
        and int(status.get("schema_version", -1)) == 1,
        "density-ratio parent schema is incompatible",
    )
    _require(
        Path(str(manifest.get("run_dir", ""))).resolve() == run_dir,
        "density-ratio manifest run directory is inconsistent",
    )
    _require(
        status.get("status") == "complete"
        and status.get("outcome") == "gate_failed"
        and status.get("phase") == "pilot"
        and status.get("stage") == "pilot"
        and status.get("decision") == EXPECTED_PARENT_DECISION
        and status.get("required_gate") == "pilot"
        and _zero(status.get("required_gate_pass")),
        "density-ratio parent is not the recorded unresolved pilot",
    )
    skips = status.get("skips", [])
    _require(
        isinstance(skips, Sequence)
        and any(
            isinstance(value, Mapping)
            and value.get("stage") == "confirmation"
            and "no density-ratio pilot profile qualified" in str(value.get("reason", ""))
            for value in skips
        ),
        "density-ratio parent did not explicitly skip confirmation",
    )
    _require(
        not (run_dir / "confirmation").exists()
        and not (run_dir / "confirmation_panel_registry.json").exists(),
        "density-ratio parent contains forbidden confirmation evidence",
    )
    registry = _verify_registry(
        run_dir, registry_path=paths["registry"], status_path=paths["status"]
    )
    scientific = _verify_scientific_manifest(manifest)
    source_paths = [Path(str(value)).resolve() for value in manifest.get("source_paths", [])]
    _require(
        source_paths
        and all(value.is_file() for value in source_paths)
        and source_fingerprint(source_paths) == manifest.get("source_fingerprint"),
        "density-ratio parent source fingerprint no longer matches",
    )
    _require(
        manifest.get("parent_provenance_sha256")
        == file_fingerprint(paths["parent_provenance"]),
        "density-ratio manifest does not bind stability provenance",
    )
    transitive = _verify_transitive_parent(values["parent_provenance"])

    preflight = _json_value(evaluate_ratio_preflight(values["preflight_raw"]))
    stored_preflight = dict(values["preflight_gate"])
    provenance_pass = stored_preflight.pop("provenance_pass", None)
    _require(
        preflight == stored_preflight
        and _one(provenance_pass)
        and _one(preflight.get("passed")),
        "density-ratio preflight gate does not recompute as passing",
    )
    _require(
        values["stream_replay"].get("evaluation_status") == "evaluated"
        and _one(values["stream_replay"].get("passed"))
        and isinstance(values["stream_replay"].get("records"), Sequence)
        and len(values["stream_replay"]["records"]) == 8
        and all(
            isinstance(record, Mapping)
            and _one(dict(record.get("verification", {})).get("passed"))
            for record in values["stream_replay"]["records"]
        ),
        "density-ratio stream replay preflight is invalid",
    )
    _verify_calibration(values["calibration"])
    _verify_pilot(run_dir, values)

    report = _json_value(
        evaluate_density_ratio_workflow(
            provenance=values["parent_provenance"],
            preflight=values["preflight_gate"],
            pilot=values["pilot_gate"],
            teacher_results=[],
            null_results=[],
            require_gate="pilot",
        )
    )
    _require(report == values["report"], "density-ratio terminal report does not recompute")
    _require(
        dict(report.get("decision", {})) == values["decision"]
        and values["decision"].get("decision") == EXPECTED_PARENT_DECISION,
        "density-ratio terminal decision changed",
    )
    for name, value in values.items():
        _no_work(value, f"density-ratio {name}")
    _require(
        _zero(status.get("physical_training_authorized"))
        and _zero(status.get("sampling_authorized")),
        "density-ratio parent authorized physical training or sampling",
    )

    schedule = transitive.get("schedule_metadata", {})
    _require(isinstance(schedule, Mapping), "density-ratio parent schedule is missing")
    horizon = schedule.get("horizon", transitive.get("horizon"))
    _require(_finite(horizon) and float(horizon) > 0.0, "density-ratio parent horizon is invalid")
    candidate_gates = list(values["pilot_gate"].get("candidate_gates", []))
    return {
        "schema": PARENT_RUN_SCHEMA + "-paired-stability-parent-provenance",
        "schema_version": 1,
        "passed": 1,
        "run_dir": str(run_dir),
        "scientific_fingerprint": str(manifest.get("scientific_fingerprint", "")),
        "runtime_fingerprint": str(manifest.get("runtime_fingerprint", "")),
        "source_fingerprint": str(manifest.get("source_fingerprint", "")),
        "scientific_config": scientific,
        "kernel": dict(scientific["kernel"]),
        "model": dict(transitive.get("model", {})),
        "schedule_metadata": dict(schedule),
        "horizon": float(horizon),
        "parent_decision": EXPECTED_PARENT_DECISION,
        "terminal_registry_record_count": len(dict(registry["records"])),
        "parent_loss_scale": PARENT_LOSS_SCALE,
        "preflight_pass": 1,
        "pilot_pass": 0,
        "pilot_candidate_count": len(candidate_gates),
        "teacher_b_confirmed_learning_rates": sorted(EXPECTED_B_CONFIRMED_RATES),
        "all_teacher_clipping_failed": 1,
        "all_nulls_selected_zero": 1,
        "confirmation_performed": 0,
        "transitive_parent_provenance": values["parent_provenance"],
        "artifacts": {name: _record(value) for name, value in paths.items()},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
