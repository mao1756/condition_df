"""Strict normalized-head parent provenance for the selection-power repair.

The only admissible parent is the immutable normalized-head pilot that ended
``classification_coordinate_repair_unresolved``.  The verifier binds its
exact 125-record terminal registry, recomputes its pure preflight/pilot and
workflow gates, checks all four completed task/status pairs (including every
recorded clipping indicator), and recursively verifies the stored
332 -> 222 -> 381 artifact ancestry.

This module is additive: none of the source files fingerprinted by the parent
run need to change in order to report or verify that run.
"""

from __future__ import annotations

import csv
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
from mnist.d0_score_density_ratio_head_gate import (
    HeadCoordinateThresholds,
    evaluate_head_pilot,
    evaluate_head_workflow,
    evaluate_normalized_head_preflight,
)
from mnist.d0_score_density_ratio_head_provenance import (
    EXPECTED_KERNEL,
    PARENT_LOSS_SCALE,
    verify_parent_paired_ratio_run,
)


__all__ = [
    "PARENT_RUN_SCHEMA",
    "PARENT_REGISTRY_SCHEMA",
    "PARENT_REGISTRY_RECORD_COUNT",
    "PARENT_LOSS_SCALE",
    "EXPECTED_PARENT_DECISION",
    "EXPECTED_KERNEL",
    "verify_parent_normalized_head_run",
]


PARENT_RUN_SCHEMA = "experiment12-d0-score-density-ratio-head-confirmation"
PARENT_REGISTRY_SCHEMA = PARENT_RUN_SCHEMA + "-artifact-registry"
PARENT_REGISTRY_RECORD_COUNT = 125
EXPECTED_PARENT_DECISION = "classification_coordinate_repair_unresolved"
EXPECTED_ACCUMULATION = 8
EXPECTED_LEARNING_RATES = (3e-5, 1e-5)
EXPECTED_MODEL_SCHEMA = "d0-boundary-smooth-potential-unet-mean-head-v2"
EXPECTED_HEAD_COORDINATE_VERSION = "d0-spatial-sum-to-mean-head-coordinate-v1"
EXPECTED_OPTIMIZER_COORDINATE_VERSION = "d0-mean-head-coordinate-adamw-v1"

_REQUIRED_FILES: dict[str, str] = {
    "manifest": "run_manifest.json",
    "status": "run_status.json",
    "registry": "artifact_registry.json",
    "parent_provenance": "parent_provenance.json",
    "head_equivalence": "head_equivalence.json",
    "optimizer_conjugacy": "adamw_ema_conjugacy.json",
    "model_size_forensics": "model_size_forensics.json",
    "stream_plans": "normalized_head_stream_plans.json",
    "stream_replay": "paired_ratio_stream_replay.json",
    "structural_law": "paired_ratio_structural_law_certificate.json",
    "preflight_raw": "normalized_head_preflight.json",
    "preflight_gate": "normalized_head_preflight_gate.json",
    "pilot_gate": "normalized_head_pilot_gate.json",
    "pilot_failures": "pilot_task_failures.json",
    "workflow": "normalized_head_control_gate.json",
    "decision": "normalized_head_decision.json",
}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"cannot read normalized-head parent artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(
            f"normalized-head parent artifact is not a JSON object: {path}"
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
    """Verify the closed terminal registry and its exact file-set equality."""

    registry = _load(registry_path)
    status = _load(status_path)
    _require(
        registry.get("schema") == PARENT_REGISTRY_SCHEMA
        and int(registry.get("schema_version", -1)) == 1,
        "normalized-head terminal registry schema is incompatible",
    )
    _require(
        status.get("artifact_registry_sha256") == file_fingerprint(registry_path)
        and int(status.get("artifact_registry_size", -1))
        == int(registry_path.stat().st_size),
        "normalized-head status does not bind its terminal registry",
    )
    exclusions = set(registry.get("terminal_files_excluded_to_avoid_self_reference", []))
    _require(
        exclusions == {"artifact_registry.json", "run_status.json"},
        "normalized-head registry exclusions are incompatible",
    )
    raw_records = registry.get("records", {})
    _require(isinstance(raw_records, Mapping), "normalized-head registry records are invalid")
    records = dict(raw_records)
    _require(
        len(records) == int(expected_count),
        f"normalized-head terminal registry must contain exactly {expected_count} records",
    )
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in exclusions | {"artifact_registry.json"}
    }
    _require(
        set(records) == actual,
        "normalized-head registry is incomplete or contains stale records",
    )
    for relative, raw in records.items():
        _require(isinstance(raw, Mapping), f"invalid registry record {relative}")
        _verify_record(
            dict(raw),
            root=run_dir,
            expected_path=run_dir / relative,
            description=f"registered normalized-head artifact {relative}",
        )
    return registry


def _verify_scientific_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw = manifest.get("scientific_config", {})
    _require(isinstance(raw, Mapping), "normalized-head scientific configuration is missing")
    scientific = dict(raw)
    _require(
        scientific.get("algorithm") == PARENT_RUN_SCHEMA
        and int(scientific.get("algorithm_version", -1)) == 1,
        "normalized-head parent algorithm is incompatible",
    )
    _require(
        config_fingerprint(scientific) == manifest.get("scientific_fingerprint"),
        "normalized-head scientific fingerprint is inconsistent",
    )
    kernel = scientific.get("kernel", {})
    _require(isinstance(kernel, Mapping), "normalized-head kernel is missing")
    for key, expected in EXPECTED_KERNEL.items():
        _require(_same(kernel.get(key), expected), f"normalized-head kernel mismatch for {key}")
    _require(
        scientific.get("thresholds")
        == _json_value(HeadCoordinateThresholds().to_dict()),
        "normalized-head scientific thresholds changed",
    )
    for key, expected in {
        "model_schema": EXPECTED_MODEL_SCHEMA,
        "head_coordinate_version": EXPECTED_HEAD_COORDINATE_VERSION,
        "optimizer_coordinate_version": EXPECTED_OPTIMIZER_COORDINATE_VERSION,
        "paired_estimator_schema": "experiment12-d0-score-density-ratio-paired-mixture",
        "paired_objective_version": "d0-density-ratio-paired-mixture-weighted-softplus-v1",
        "paired_stream_version": "d0-density-ratio-paired-mixture-stream-v1",
        "paired_accumulation_version": "d0-density-ratio-deterministic-gradient-accumulation-v1",
    }.items():
        _require(scientific.get(key) == expected, f"normalized-head schema mismatch for {key}")
    _require(
        _same(scientific.get("loss_scale"), PARENT_LOSS_SCALE),
        "normalized-head loss multiplier changed",
    )
    optimization = scientific.get("optimization", {})
    _require(isinstance(optimization, Mapping), "normalized-head optimizer configuration is missing")
    for key, expected in {
        "optimizer": "coordinate-conjugate-AdamW",
        "body_weight_decay": 1e-4,
        "ema_decay": 0.99,
        "grad_clip": 1.0,
        "clip_warmup_steps": 500,
        "adaptive_loss_scaling": 0,
        "gradient_accumulation": "mean-then-clip-once",
        "head_lr_factor": 784,
        "head_eps_factor": 1.0 / 784.0,
        "head_weight_decay_factor": 1.0 / 784.0,
    }.items():
        _require(
            _same(optimization.get(key), expected),
            f"normalized-head optimizer mismatch for {key}",
        )
    pilot = scientific.get("pilot", {})
    confirmation = scientific.get("confirmation", {})
    microbatch = scientific.get("microbatch", {})
    preflight = scientific.get("preflight", {})
    bootstrap = scientific.get("bootstrap", {})
    _require(
        isinstance(pilot, Mapping)
        and tuple(int(value) for value in pilot.get("accumulation_levels", []))
        == (EXPECTED_ACCUMULATION,)
        and tuple(float(value) for value in pilot.get("learning_rates", []))
        == EXPECTED_LEARNING_RATES
        and int(pilot.get("steps", -1)) == 2000
        and int(pilot.get("selection_paths_per_panel", -1)) == 16,
        "normalized-head pilot configuration changed",
    )
    _require(
        isinstance(confirmation, Mapping)
        and tuple(int(value) for value in confirmation.get("model_seeds", []))
        == (260891, 260892, 260893)
        and int(confirmation.get("steps", -1)) == 4000
        and int(confirmation.get("selection_paths_per_panel", -1)) == 32
        and int(confirmation.get("audit_paths_per_panel", -1)) == 32,
        "normalized-head planned confirmation configuration changed",
    )
    _require(
        isinstance(microbatch, Mapping)
        and int(microbatch.get("clusters", -1)) == 32
        and list(microbatch.get("time_bin_counts", [])) == [4, 4, 4, 4, 16]
        and microbatch.get("teacher_coupling") == "common-gamma-stochastic-anchor"
        and microbatch.get("null_coupling")
        == "independent-dirichlet-pooled-label-swap",
        "normalized-head microbatch law changed",
    )
    _require(
        isinstance(preflight, Mapping)
        and int(preflight.get("paths", -1)) == 128
        and int(preflight.get("bootstrap_reps", -1)) == 10000
        and _same(preflight.get("confidence"), 0.99),
        "normalized-head preflight configuration changed",
    )
    _require(
        isinstance(bootstrap, Mapping)
        and int(bootstrap.get("reps", -1)) == 10000
        and _same(bootstrap.get("selection_audit_confidence"), 0.9),
        "normalized-head bootstrap configuration changed",
    )
    _require(int(scientific.get("root_seed", -1)) == 260881, "normalized-head root seed changed")
    _no_work(scientific, "normalized-head scientific configuration")
    return scientific


def _verify_transitive_parent(parent: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        parent.get("schema")
        == "experiment12-d0-score-density-ratio-head-confirmation-parent-provenance"
        and int(parent.get("schema_version", -1)) == 1
        and _one(parent.get("passed")),
        "normalized-head transitive paired-ratio provenance is incompatible",
    )
    root = Path(str(parent.get("run_dir", ""))).resolve()
    recomputed = verify_parent_paired_ratio_run(root)
    normalized = dict(parent)
    normalized.pop("verifier_source_fingerprint", None)
    normalized["schema"] = recomputed["schema"]
    _require(
        normalized == recomputed,
        "normalized-head transitive 332/222/381-record provenance changed",
    )
    _require(
        int(recomputed.get("terminal_registry_record_count", -1)) == 332
        and int(
            dict(recomputed.get("transitive_parent_provenance", {})).get(
                "terminal_registry_record_count", -1
            )
        )
        == 222
        and int(
            dict(
                dict(recomputed.get("transitive_parent_provenance", {})).get(
                    "transitive_parent_provenance", {}
                )
            ).get("terminal_registry_record_count", -1)
        )
        == 381,
        "normalized-head parent does not bind the exact 332/222/381 ancestry",
    )
    _no_work(parent, "normalized-head transitive parent")
    return recomputed


def _verify_zero_clipping(
    metrics: Mapping[str, Any],
    *,
    history_path: Path,
    windows_path: Path,
    description: str,
) -> None:
    for name in (
        "post_warmup_clip_fraction",
        "final_500_clip_fraction",
        "final_200_clip_fraction",
    ):
        _require(
            _finite(metrics.get(name)) and float(metrics[name]) == 0.0,
            f"{description} does not record zero {name}",
        )
    diagnostics = metrics.get("optimization_diagnostics", {})
    _require(isinstance(diagnostics, Mapping), f"{description} optimization diagnostics are missing")
    for name in (
        "post_warmup_clip_fraction",
        "final_500_clip_fraction",
        "final_200_clip_fraction",
    ):
        _require(
            _finite(diagnostics.get(name)) and float(diagnostics[name]) == 0.0,
            f"{description} diagnostics do not record zero {name}",
        )
    windows = diagnostics.get("clipping_windows", [])
    _require(isinstance(windows, Sequence) and len(windows) > 0, f"{description} clipping windows are missing")
    _require(
        all(
            isinstance(value, Mapping)
            and _finite(value.get("clip_fraction"))
            and float(value["clip_fraction"]) == 0.0
            for value in windows
        ),
        f"{description} contains a nonzero clipping window",
    )
    _require(history_path.is_file() and windows_path.is_file(), f"{description} clipping CSV artifacts are missing")
    try:
        with history_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        with windows_path.open("r", encoding="utf-8", newline="") as handle:
            window_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise ArtifactCompatibilityError(f"cannot read {description} clipping CSV: {exc}") from exc
    _require(len(rows) == 2000, f"{description} training history must contain 2000 updates")
    _require(
        all(_finite(row.get("clipped")) and float(row["clipped"]) == 0.0 for row in rows),
        f"{description} training history contains clipping",
    )
    _require(window_rows, f"{description} clipping-window CSV is empty")
    _require(
        all(
            _finite(row.get("clip_fraction")) and float(row["clip_fraction"]) == 0.0
            for row in window_rows
        ),
        f"{description} clipping-window CSV contains clipping",
    )


def _verify_task_result(
    aggregate: Mapping[str, Any],
    *,
    task_dir: Path,
    expected_task: str,
    expected_learning_rate: float,
    scientific_fingerprint: str,
    runtime_fingerprint: str,
    source_fingerprint_value: str,
) -> int:
    result_path = task_dir / "task_result.json"
    status_path = task_dir / "task_status.json"
    _require(result_path.is_file() and status_path.is_file(), f"missing {expected_task} task artifacts")
    result = _load(result_path)
    status = _load(status_path)
    _require(result == dict(aggregate), f"{expected_task} aggregate/task result mismatch")
    _require(
        status.get("status") == "complete"
        and status.get("task") == expected_task
        and status.get("task_result_sha256") == file_fingerprint(result_path)
        and int(status.get("training_step", -1)) == 2000,
        f"{expected_task} task status is incomplete or unbound",
    )
    metrics = result.get("metrics", {})
    _require(isinstance(metrics, Mapping), f"{expected_task} metrics are missing")
    _require(
        _one(metrics.get("complete"))
        and _one(metrics.get("finite"))
        and _one(metrics.get("boundary_admissible")),
        f"{expected_task} task is incomplete, nonfinite, or boundary-inadmissible",
    )
    gate = result.get("gate", {})
    _require(isinstance(gate, Mapping) and _one(gate.get("passed")), f"{expected_task} task gate failed")
    fingerprints = result.get("fingerprints", {})
    _require(isinstance(fingerprints, Mapping), f"{expected_task} fingerprints are missing")
    _require(
        int(fingerprints.get("accumulation_level", -1)) == EXPECTED_ACCUMULATION
        and _same(fingerprints.get("learning_rate"), expected_learning_rate)
        and _same(fingerprints.get("loss_scale"), PARENT_LOSS_SCALE)
        and fingerprints.get("task") == expected_task
        and fingerprints.get("phase") == "pilot"
        and fingerprints.get("model_schema") == EXPECTED_MODEL_SCHEMA
        and fingerprints.get("head_coordinate_version") == EXPECTED_HEAD_COORDINATE_VERSION
        and fingerprints.get("optimizer_coordinate_version")
        == EXPECTED_OPTIMIZER_COORDINATE_VERSION
        and fingerprints.get("scientific_fingerprint") == scientific_fingerprint
        and fingerprints.get("runtime_fingerprint") == runtime_fingerprint
        and fingerprints.get("source_fingerprint") == source_fingerprint_value,
        f"{expected_task} task fingerprints changed",
    )
    _require(
        dict(status.get("fingerprints", {})) == dict(fingerprints),
        f"{expected_task} task status fingerprints changed",
    )
    model_seed = int(result.get("model_seed", -1))
    _require(
        model_seed >= 0 and int(status.get("model_seed", -2)) == model_seed,
        f"{expected_task} model seed changed",
    )
    _verify_zero_clipping(
        dict(metrics),
        history_path=task_dir / "training_history.csv",
        windows_path=task_dir / "clipping_windows.csv",
        description=f"{expected_task} task",
    )
    _no_work(result, f"{expected_task} task result")
    _no_work(status, f"{expected_task} task status")
    return model_seed


def _verify_pilot(
    run_dir: Path,
    stored: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_candidates = stored.get("candidate_records", [])
    _require(
        isinstance(raw_candidates, Sequence) and len(raw_candidates) == 2,
        "normalized-head parent must contain exactly two pilot candidates",
    )
    candidates = [dict(value) for value in raw_candidates if isinstance(value, Mapping)]
    _require(len(candidates) == 2, "normalized-head pilot candidates are invalid")
    model_seeds: list[int] = []
    for index, (candidate, learning_rate) in enumerate(
        zip(candidates, EXPECTED_LEARNING_RATES, strict=True)
    ):
        _require(
            int(candidate.get("accumulation_steps", -1)) == EXPECTED_ACCUMULATION
            and _same(candidate.get("learning_rate"), learning_rate),
            "normalized-head pilot candidate ordering changed",
        )
        paired_seeds: list[int] = []
        for key, task, directory in (
            ("teacher", "bounded_teacher", "bounded_teacher"),
            ("null", "dirichlet_null", "dirichlet_null"),
        ):
            aggregate = candidate.get(key, {})
            _require(isinstance(aggregate, Mapping), f"normalized-head {task} result is missing")
            task_dir = (
                run_dir
                / "pilot"
                / "accum-08"
                / f"lr-{index:02d}"
                / directory
            )
            paired_seeds.append(
                _verify_task_result(
                    dict(aggregate),
                    task_dir=task_dir,
                    expected_task=task,
                    expected_learning_rate=learning_rate,
                    scientific_fingerprint=str(manifest.get("scientific_fingerprint", "")),
                    runtime_fingerprint=str(manifest.get("runtime_fingerprint", "")),
                    source_fingerprint_value=str(manifest.get("source_fingerprint", "")),
                )
            )
        _require(paired_seeds[0] == paired_seeds[1], "normalized-head teacher/null initialization is not paired")
        model_seeds.extend(paired_seeds)
    _require(len(set(model_seeds)) == 1, "normalized-head pilot profiles did not share initialization")
    recomputed = _json_value(evaluate_head_pilot(candidates))
    comparable = {key: stored.get(key) for key in recomputed}
    _require(recomputed == comparable, "normalized-head pilot gate does not recompute")
    _require(
        _zero(recomputed.get("passed"))
        and _zero(dict(recomputed.get("selected_profile", {})).get("selected")),
        "normalized-head parent unexpectedly selected a profile",
    )
    _no_work(stored, "normalized-head pilot gate")
    return candidates


def verify_parent_normalized_head_run(path: str | Path) -> dict[str, Any]:
    """Verify the exact 125-record failed normalized-head production pilot."""

    run_dir = Path(path).resolve()
    _require(run_dir.is_dir(), f"normalized-head parent run does not exist: {run_dir}")
    paths = {name: run_dir / relative for name, relative in _REQUIRED_FILES.items()}
    missing = [str(value) for value in paths.values() if not value.is_file()]
    _require(not missing, "normalized-head parent artifacts are missing: " + ", ".join(missing))
    values = {name: _load(value) for name, value in paths.items() if name != "registry"}
    manifest, status = values["manifest"], values["status"]
    _require(
        manifest.get("schema") == status.get("schema") == PARENT_RUN_SCHEMA
        and int(manifest.get("schema_version", -1)) == 1
        and int(status.get("schema_version", -1)) == 1,
        "normalized-head parent schema is incompatible",
    )
    _require(
        Path(str(manifest.get("run_dir", ""))).resolve() == run_dir,
        "normalized-head manifest run directory is inconsistent",
    )
    _require(
        status.get("status") == "complete"
        and status.get("outcome") == "gate_failed"
        and status.get("phase") == "pilot"
        and status.get("stage") == "pilot"
        and status.get("decision") == EXPECTED_PARENT_DECISION
        and status.get("required_gate") == "pilot"
        and _zero(status.get("required_gate_pass")),
        "normalized-head parent is not the recorded selection-power precursor",
    )
    skips = status.get("skips", [])
    _require(
        isinstance(skips, Sequence)
        and any(
            isinstance(value, Mapping)
            and value.get("stage") == "confirmation"
            and "no normalized-head profile qualified" in str(value.get("reason", ""))
            for value in skips
        ),
        "normalized-head parent did not explicitly skip confirmation",
    )
    _require(
        not (run_dir / "confirmation").exists()
        and not (run_dir / "confirmation_panel_registry.json").exists()
        and not (run_dir / "selected_stability_profile.json").exists(),
        "normalized-head parent contains forbidden profile or confirmation evidence",
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
        "normalized-head parent source fingerprint no longer matches",
    )
    _require(
        manifest.get("parent_provenance_sha256") == file_fingerprint(paths["parent_provenance"]),
        "normalized-head manifest does not bind paired-ratio provenance",
    )
    transitive = _verify_transitive_parent(values["parent_provenance"])
    _require(
        scientific.get("parent_artifact_registry_sha256")
        == transitive.get("artifact_registry_sha256")
        and scientific.get("parent_scientific_fingerprint")
        == transitive.get("scientific_fingerprint"),
        "normalized-head scientific configuration does not bind its parent",
    )

    preflight = _json_value(evaluate_normalized_head_preflight(values["preflight_raw"]))
    _require(
        preflight == values["preflight_gate"] and _one(preflight.get("passed")),
        "normalized-head preflight gate does not recompute as passing",
    )
    candidates = _verify_pilot(run_dir, values["pilot_gate"], manifest=manifest)
    failures = values["pilot_failures"]
    _require(
        int(failures.get("count", -1)) == 0 and list(failures.get("failures", [])) == [],
        "normalized-head pilot contains task failures",
    )

    report = _json_value(
        evaluate_head_workflow(
            provenance=values["parent_provenance"],
            preflight=values["preflight_gate"],
            pilot=values["pilot_gate"],
            teacher_results=[],
            null_results=[],
            require_gate="pilot",
        )
    )
    _require(report == values["workflow"], "normalized-head terminal workflow does not recompute")
    _require(
        dict(report.get("decision", {})) == values["decision"]
        and values["decision"].get("decision") == EXPECTED_PARENT_DECISION,
        "normalized-head terminal decision changed",
    )
    for name, value in values.items():
        _no_work(value, f"normalized-head {name}")
    _require(
        _zero(status.get("physical_training_authorized"))
        and _zero(status.get("sampling_authorized")),
        "normalized-head parent authorized physical training or sampling",
    )
    schedule = transitive.get("schedule_metadata", {})
    _require(isinstance(schedule, Mapping), "normalized-head parent schedule is missing")
    horizon = schedule.get("horizon", transitive.get("horizon"))
    _require(_finite(horizon) and float(horizon) > 0.0, "normalized-head parent horizon is invalid")
    return {
        "schema": PARENT_RUN_SCHEMA + "-selection-power-parent-provenance",
        "schema_version": 1,
        "passed": 1,
        "run_dir": str(run_dir),
        "scientific_fingerprint": str(manifest.get("scientific_fingerprint", "")),
        "runtime_fingerprint": str(manifest.get("runtime_fingerprint", "")),
        "source_fingerprint": str(manifest.get("source_fingerprint", "")),
        "scientific_config": scientific,
        "kernel": dict(scientific["kernel"]),
        "model": {
            "schema": EXPECTED_MODEL_SCHEMA,
            "base_channels": 32,
            "head_coordinate_version": EXPECTED_HEAD_COORDINATE_VERSION,
            "optimizer_coordinate_version": EXPECTED_OPTIMIZER_COORDINATE_VERSION,
        },
        "schedule_metadata": dict(schedule),
        "horizon": float(horizon),
        "parent_decision": EXPECTED_PARENT_DECISION,
        "terminal_registry_record_count": len(dict(registry["records"])),
        "artifact_registry_sha256": file_fingerprint(paths["registry"]),
        "artifact_registry_size": int(paths["registry"].stat().st_size),
        "parent_loss_scale": PARENT_LOSS_SCALE,
        "preflight_pass": 1,
        "pilot_pass": 0,
        "pilot_candidate_count": len(candidates),
        "pilot_task_count": 2 * len(candidates),
        "all_tasks_complete_finite_boundary_admissible": 1,
        "all_task_clipping_zero": 1,
        "task_failure_count": 0,
        "selected_profile": 0,
        "confirmation_performed": 0,
        "transitive_parent_provenance": values["parent_provenance"],
        "artifacts": {name: _record(value) for name, value in paths.items()},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
