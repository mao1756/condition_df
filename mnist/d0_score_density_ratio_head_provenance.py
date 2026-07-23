"""Strict parent provenance for the normalized-head coordinate repair.

The only admissible parent is the immutable paired-mixture stability pilot
that ended ``classification_variance_reduction_unresolved``.  This verifier
binds its exact 332-record terminal registry, recomputes its pure gates,
checks all twelve task/status pairs, and recursively verifies the stored
222-record density-ratio provenance (which itself binds the 381-record
streamed-control run).
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
from mnist.d0_score_density_ratio_stability_gate import (
    RatioStabilityThresholds,
    evaluate_paired_ratio_preflight,
    evaluate_ratio_stability_workflow,
    evaluate_stability_pilot,
)
from mnist.d0_score_density_ratio_stability_provenance import (
    PARENT_LOSS_SCALE,
    verify_parent_density_ratio_run,
)


__all__ = [
    "PARENT_RUN_SCHEMA",
    "PARENT_REGISTRY_SCHEMA",
    "PARENT_REGISTRY_RECORD_COUNT",
    "PARENT_LOSS_SCALE",
    "EXPECTED_PARENT_DECISION",
    "EXPECTED_KERNEL",
    "verify_parent_paired_ratio_run",
]


PARENT_RUN_SCHEMA = "experiment12-d0-score-density-ratio-stability-confirmation"
PARENT_REGISTRY_SCHEMA = PARENT_RUN_SCHEMA + "-artifact-registry"
PARENT_REGISTRY_RECORD_COUNT = 332
EXPECTED_PARENT_DECISION = "classification_variance_reduction_unresolved"
EXPECTED_ACCUMULATION_LEVELS = (2, 4, 8)
EXPECTED_LEARNING_RATES = (3e-5, 1e-5)
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
    "preflight_raw": "paired_ratio_preflight.json",
    "preflight_gate": "paired_ratio_preflight_gate.json",
    "stream_plans": "paired_ratio_stream_plans.json",
    "stream_replay": "paired_ratio_stream_replay.json",
    "structural_law": "paired_ratio_structural_law_certificate.json",
    "variance_forensics": "paired_ratio_variance_forensics.json",
    "pilot_gate": "stability_pilot_gate.json",
    "pilot_failures": "pilot_task_failures.json",
    "workflow": "boundary_control_stability_gate.json",
    "decision": "density_ratio_stability_decision.json",
}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            f"cannot read paired-ratio parent artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(
            f"paired-ratio parent artifact is not a JSON object: {path}"
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
        "paired-ratio terminal registry schema is incompatible",
    )
    _require(
        status.get("artifact_registry_sha256") == file_fingerprint(registry_path)
        and int(status.get("artifact_registry_size", -1))
        == int(registry_path.stat().st_size),
        "paired-ratio status does not bind its terminal registry",
    )
    exclusions = set(registry.get("terminal_files_excluded_to_avoid_self_reference", []))
    _require(
        exclusions == {"artifact_registry.json", "run_status.json"},
        "paired-ratio registry exclusions are incompatible",
    )
    raw_records = registry.get("records", {})
    _require(isinstance(raw_records, Mapping), "paired-ratio registry records are invalid")
    records = dict(raw_records)
    _require(
        len(records) == int(expected_count),
        f"paired-ratio terminal registry must contain exactly {expected_count} records",
    )
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in exclusions | {"artifact_registry.json"}
    }
    _require(
        set(records) == actual,
        "paired-ratio registry is incomplete or contains stale records",
    )
    for relative, raw in records.items():
        _require(isinstance(raw, Mapping), f"invalid registry record {relative}")
        _verify_record(
            dict(raw),
            root=run_dir,
            expected_path=run_dir / relative,
            description=f"registered paired-ratio artifact {relative}",
        )
    return registry


def _verify_scientific_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw = manifest.get("scientific_config", {})
    _require(isinstance(raw, Mapping), "paired-ratio scientific configuration is missing")
    scientific = dict(raw)
    _require(
        scientific.get("algorithm") == PARENT_RUN_SCHEMA
        and int(scientific.get("algorithm_version", -1)) == 1,
        "paired-ratio parent algorithm is incompatible",
    )
    _require(
        config_fingerprint(scientific) == manifest.get("scientific_fingerprint"),
        "paired-ratio scientific fingerprint is inconsistent",
    )
    kernel = scientific.get("kernel", {})
    _require(isinstance(kernel, Mapping), "paired-ratio kernel is missing")
    for key, expected in EXPECTED_KERNEL.items():
        _require(_same(kernel.get(key), expected), f"paired-ratio kernel mismatch for {key}")
    _require(
        scientific.get("thresholds")
        == _json_value(RatioStabilityThresholds().to_dict()),
        "paired-ratio scientific thresholds changed",
    )
    for key, expected in {
        "model_schema": "d0-boundary-smooth-potential-unet-v1",
        "paired_estimator_schema": "experiment12-d0-score-density-ratio-paired-mixture",
        "paired_objective_version": "d0-density-ratio-paired-mixture-weighted-softplus-v1",
        "paired_stream_version": "d0-density-ratio-paired-mixture-stream-v1",
        "paired_accumulation_version": "d0-density-ratio-deterministic-gradient-accumulation-v1",
    }.items():
        _require(scientific.get(key) == expected, f"paired-ratio schema mismatch for {key}")
    _require(
        _same(scientific.get("loss_scale"), PARENT_LOSS_SCALE),
        "paired-ratio loss multiplier changed",
    )
    optimization = scientific.get("optimization", {})
    _require(isinstance(optimization, Mapping), "paired-ratio optimizer configuration is missing")
    for key, expected in {
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "ema_decay": 0.99,
        "grad_clip": 1.0,
        "clip_warmup_steps": 500,
        "adaptive_loss_scaling": 0,
        "gradient_accumulation": "mean-then-clip-once",
    }.items():
        _require(_same(optimization.get(key), expected), f"paired-ratio optimizer mismatch for {key}")
    pilot = scientific.get("pilot", {})
    confirmation = scientific.get("confirmation", {})
    microbatch = scientific.get("microbatch", {})
    preflight = scientific.get("preflight", {})
    _require(
        isinstance(pilot, Mapping)
        and tuple(int(value) for value in pilot.get("accumulation_levels", []))
        == EXPECTED_ACCUMULATION_LEVELS
        and tuple(float(value) for value in pilot.get("learning_rates", []))
        == EXPECTED_LEARNING_RATES
        and int(pilot.get("steps", -1)) == 2000
        and int(pilot.get("selection_paths_per_panel", -1)) == 16,
        "paired-ratio pilot configuration changed",
    )
    _require(
        isinstance(confirmation, Mapping)
        and tuple(int(value) for value in confirmation.get("model_seeds", []))
        == (260861, 260862, 260863)
        and int(confirmation.get("steps", -1)) == 4000
        and int(confirmation.get("selection_paths_per_panel", -1)) == 32
        and int(confirmation.get("audit_paths_per_panel", -1)) == 32,
        "paired-ratio planned confirmation configuration changed",
    )
    _require(
        isinstance(microbatch, Mapping)
        and int(microbatch.get("clusters", -1)) == 32
        and list(microbatch.get("time_bin_counts", [])) == [4, 4, 4, 4, 16]
        and microbatch.get("teacher_coupling") == "common-gamma-stochastic-anchor"
        and microbatch.get("null_coupling")
        == "independent-dirichlet-pooled-label-swap",
        "paired-ratio microbatch law changed",
    )
    _require(
        isinstance(preflight, Mapping)
        and int(preflight.get("paths", -1)) == 128
        and _same(preflight.get("confidence"), 0.99),
        "paired-ratio preflight configuration changed",
    )
    _require(int(scientific.get("root_seed", -1)) == 260851, "paired-ratio root seed changed")
    _no_work(scientific, "paired-ratio scientific configuration")
    return scientific


def _verify_transitive_parent(parent: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        parent.get("schema")
        == "experiment12-d0-score-density-ratio-stability-confirmation-parent-provenance"
        and int(parent.get("schema_version", -1)) == 1
        and _one(parent.get("passed")),
        "paired-ratio transitive density-ratio provenance is incompatible",
    )
    root = Path(str(parent.get("run_dir", ""))).resolve()
    recomputed = verify_parent_density_ratio_run(root)
    normalized = dict(parent)
    normalized.pop("verifier_source_fingerprint", None)
    normalized["schema"] = recomputed["schema"]
    _require(
        normalized == recomputed,
        "paired-ratio transitive 222/381-record provenance changed",
    )
    _require(
        int(recomputed.get("terminal_registry_record_count", -1)) == 222
        and int(
            dict(recomputed.get("transitive_parent_provenance", {})).get(
                "terminal_registry_record_count", -1
            )
        )
        == 381,
        "paired-ratio transitive parent does not bind 222 and 381 artifacts",
    )
    _no_work(parent, "paired-ratio transitive parent")
    return recomputed


def _verify_task_result(
    aggregate: Mapping[str, Any],
    *,
    result_path: Path,
    status_path: Path,
    expected_task: str,
    expected_accumulation: int,
    expected_learning_rate: float,
) -> int:
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
        int(fingerprints.get("accumulation_level", -1)) == expected_accumulation
        and _same(fingerprints.get("learning_rate"), expected_learning_rate)
        and _same(fingerprints.get("loss_scale"), PARENT_LOSS_SCALE)
        and fingerprints.get("task") == expected_task
        and fingerprints.get("phase") == "pilot",
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
    _no_work(result, f"{expected_task} task result")
    _no_work(status, f"{expected_task} task status")
    return model_seed


def _verify_pilot(run_dir: Path, stored: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_candidates = stored.get("candidate_records", [])
    _require(
        isinstance(raw_candidates, Sequence) and len(raw_candidates) == 12 // 2,
        "paired-ratio parent must contain exactly six pilot candidates",
    )
    candidates = [dict(value) for value in raw_candidates if isinstance(value, Mapping)]
    _require(len(candidates) == 6, "paired-ratio pilot candidates are invalid")
    expected_pairs = [
        (level, learning_rate)
        for level in EXPECTED_ACCUMULATION_LEVELS
        for learning_rate in EXPECTED_LEARNING_RATES
    ]
    for index, (candidate, expected) in enumerate(zip(candidates, expected_pairs, strict=True)):
        accumulation, learning_rate = expected
        _require(
            int(candidate.get("accumulation_steps", -1)) == accumulation
            and _same(candidate.get("learning_rate"), learning_rate),
            "paired-ratio pilot candidate ordering changed",
        )
        seeds: list[int] = []
        for key, task, directory in (
            ("teacher", "bounded_teacher", "bounded_teacher"),
            ("null", "dirichlet_null", "dirichlet_null"),
        ):
            aggregate = candidate.get(key, {})
            _require(isinstance(aggregate, Mapping), f"paired-ratio {task} result is missing")
            root = run_dir / "pilot" / f"accum-{accumulation:02d}" / f"lr-{index % 2:02d}" / directory
            seeds.append(
                _verify_task_result(
                    dict(aggregate),
                    result_path=root / "task_result.json",
                    status_path=root / "task_status.json",
                    expected_task=task,
                    expected_accumulation=accumulation,
                    expected_learning_rate=learning_rate,
                )
            )
        _require(seeds[0] == seeds[1], "paired-ratio teacher/null initialization is not paired")
    recomputed = _json_value(evaluate_stability_pilot(candidates))
    comparable = {key: stored.get(key) for key in recomputed}
    _require(recomputed == comparable, "paired-ratio pilot gate does not recompute")
    _require(
        _zero(recomputed.get("passed"))
        and _one(recomputed.get("all_levels_complete"))
        and _zero(dict(recomputed.get("selected_profile", {})).get("selected")),
        "paired-ratio parent unexpectedly selected a profile",
    )
    _no_work(stored, "paired-ratio pilot gate")
    return candidates


def verify_parent_paired_ratio_run(path: str | Path) -> dict[str, Any]:
    """Verify the exact 332-record failed paired-ratio stability pilot."""

    run_dir = Path(path).resolve()
    _require(run_dir.is_dir(), f"paired-ratio parent run does not exist: {run_dir}")
    paths = {name: run_dir / relative for name, relative in _REQUIRED_FILES.items()}
    missing = [str(value) for value in paths.values() if not value.is_file()]
    _require(not missing, "paired-ratio parent artifacts are missing: " + ", ".join(missing))
    values = {name: _load(value) for name, value in paths.items() if name != "registry"}
    manifest, status = values["manifest"], values["status"]
    _require(
        manifest.get("schema") == status.get("schema") == PARENT_RUN_SCHEMA
        and int(manifest.get("schema_version", -1)) == 1
        and int(status.get("schema_version", -1)) == 1,
        "paired-ratio parent schema is incompatible",
    )
    _require(
        Path(str(manifest.get("run_dir", ""))).resolve() == run_dir,
        "paired-ratio manifest run directory is inconsistent",
    )
    _require(
        status.get("status") == "complete"
        and status.get("outcome") == "gate_failed"
        and status.get("phase") == "pilot"
        and status.get("stage") == "pilot"
        and status.get("decision") == EXPECTED_PARENT_DECISION
        and status.get("required_gate") == "pilot"
        and _zero(status.get("required_gate_pass")),
        "paired-ratio parent is not the recorded coordinate-repair precursor",
    )
    skips = status.get("skips", [])
    _require(
        isinstance(skips, Sequence)
        and any(
            isinstance(value, Mapping)
            and value.get("stage") == "confirmation"
            and "no paired variance-reduction profile qualified" in str(value.get("reason", ""))
            for value in skips
        ),
        "paired-ratio parent did not explicitly skip confirmation",
    )
    _require(
        not (run_dir / "confirmation").exists()
        and not (run_dir / "confirmation_panel_registry.json").exists()
        and not (run_dir / "selected_stability_profile.json").exists(),
        "paired-ratio parent contains forbidden profile or confirmation evidence",
    )
    registry = _verify_registry(run_dir, registry_path=paths["registry"], status_path=paths["status"])
    scientific = _verify_scientific_manifest(manifest)
    source_paths = [Path(str(value)).resolve() for value in manifest.get("source_paths", [])]
    _require(
        source_paths
        and all(value.is_file() for value in source_paths)
        and source_fingerprint(source_paths) == manifest.get("source_fingerprint"),
        "paired-ratio parent source fingerprint no longer matches",
    )
    _require(
        manifest.get("parent_provenance_sha256") == file_fingerprint(paths["parent_provenance"]),
        "paired-ratio manifest does not bind density-ratio provenance",
    )
    transitive = _verify_transitive_parent(values["parent_provenance"])

    preflight = _json_value(evaluate_paired_ratio_preflight(values["preflight_raw"]))
    _require(
        preflight == values["preflight_gate"] and _one(preflight.get("passed")),
        "paired-ratio preflight gate does not recompute as passing",
    )
    structural = values["structural_law"]
    _require(
        all(
            _one(structural.get(name))
            for name in (
                "dirichlet_marginal_construction_pass",
                "exact_seed_namespaces_pass",
                "null_pool_swap_structure_pass",
                "simplex_positive_finite_pass",
                "teacher_common_gamma_pass",
            )
        )
        and _zero(structural.get("stochastic_moment_thresholds_used"))
        and _one(values["stream_replay"].get("passed")),
        "paired-ratio structural-law or stream replay certificate failed",
    )
    candidates = _verify_pilot(run_dir, values["pilot_gate"])
    failures = values["pilot_failures"]
    _require(
        int(failures.get("count", -1)) == 0 and list(failures.get("failures", [])) == [],
        "paired-ratio pilot contains task failures",
    )

    report = _json_value(
        evaluate_ratio_stability_workflow(
            provenance=values["parent_provenance"],
            preflight=values["preflight_gate"],
            pilot=values["pilot_gate"],
            teacher_results=[],
            null_results=[],
            require_gate="pilot",
        )
    )
    _require(report == values["workflow"], "paired-ratio terminal workflow does not recompute")
    _require(
        dict(report.get("decision", {})) == values["decision"]
        and values["decision"].get("decision") == EXPECTED_PARENT_DECISION,
        "paired-ratio terminal decision changed",
    )
    for name, value in values.items():
        _no_work(value, f"paired-ratio {name}")
    _require(
        _zero(status.get("physical_training_authorized"))
        and _zero(status.get("sampling_authorized")),
        "paired-ratio parent authorized physical training or sampling",
    )
    schedule = transitive.get("schedule_metadata", {})
    _require(isinstance(schedule, Mapping), "paired-ratio parent schedule is missing")
    horizon = schedule.get("horizon", transitive.get("horizon"))
    _require(_finite(horizon) and float(horizon) > 0.0, "paired-ratio parent horizon is invalid")
    return {
        "schema": PARENT_RUN_SCHEMA + "-normalized-head-parent-provenance",
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
        # Expose the already-verified terminal registry binding so the child
        # scientific fingerprint (and therefore every task checkpoint) can
        # bind the immutable 332-artifact parent, rather than recording a
        # null placeholder.
        "artifact_registry_sha256": file_fingerprint(paths["registry"]),
        "artifact_registry_size": int(paths["registry"].stat().st_size),
        "parent_loss_scale": PARENT_LOSS_SCALE,
        "preflight_pass": 1,
        "pilot_pass": 0,
        "pilot_candidate_count": len(candidates),
        "pilot_task_count": 2 * len(candidates),
        "all_tasks_complete_finite": 1,
        "task_failure_count": 0,
        "selected_profile": 0,
        "confirmation_performed": 0,
        "transitive_parent_provenance": values["parent_provenance"],
        "artifacts": {name: _record(value) for name, value in paths.items()},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
