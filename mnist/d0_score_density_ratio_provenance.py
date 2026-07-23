"""Strict provenance binding for D0 density-ratio control experiments.

Only the completed streamed implicit-control run with the recorded
``implicit_objective_unstable`` outcome authorizes the density-ratio fallback.
This verifier checks its exact 381-record registry, recomputes every scientific
gate that controls the conclusion, verifies all fourteen task-result/status
bindings, and recursively verifies the scale-repair parent.
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
)
from mnist.d0_score_control_stability_gate import (
    ProbeBankStatus,
    StabilityThresholds,
    evaluate_stability_confirmation,
    evaluate_stability_pilot,
    evaluate_stability_workflow,
    evaluate_stein_identity_preflight,
)
from mnist.d0_score_control_stability_provenance import (
    PARENT_IMPLICIT_LOSS_SCALE,
    verify_parent_scale_repair_run,
)


__all__ = [
    "PARENT_RUN_SCHEMA",
    "PARENT_REGISTRY_RECORD_COUNT",
    "EXPECTED_PARENT_DECISION",
    "verify_parent_stability_run",
]


PARENT_RUN_SCHEMA = "experiment12-d0-score-control-stability-confirmation"
PARENT_REGISTRY_SCHEMA = PARENT_RUN_SCHEMA + "-artifact-registry"
PARENT_REGISTRY_RECORD_COUNT = 381
EXPECTED_PARENT_DECISION = "implicit_objective_unstable"
EXPECTED_CONFIRMATION_SEEDS = (260811, 260812, 260813)
EXPECTED_PILOT_LEARNING_RATES = (1e-4, 3e-5, 1e-5, 3e-6)

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
    "stein_raw": "stein_identity_preflight.json",
    "stein_gate": "stein_identity_preflight_gate.json",
    "preflight_gate": "stability_preflight_gate.json",
    "forensic": "parent_forensic_replay.json",
    "pilot_candidates": "pilot_candidate_registry.json",
    "pilot_gate": "stability_pilot_gate.json",
    "selected_profile": "selected_stability_profile.json",
    "pilot_failures": "pilot_task_failures.json",
    "profile_binding": "confirmation_profile_binding.json",
    "teacher_confirmation": "implicit_teacher_confirmation.json",
    "null_confirmation": "null_confirmation.json",
    "confirmation_gate": "boundary_control_stability_gate.json",
    "confirmation_failures": "confirmation_task_failures.json",
    "report": "stability_confirmation_report.json",
    "decision": "control_stability_decision.json",
}


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"cannot read stability artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"stability artifact is not a JSON object: {path}")
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
    """Normalize tuples exactly as an atomic JSON artifact does."""

    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": file_fingerprint(path),
        "size": int(path.stat().st_size),
    }


def _verify_record(
    raw: Mapping[str, Any],
    *,
    root: Path,
    expected_path: Path,
    description: str,
) -> None:
    raw_path = raw.get("path")
    path = expected_path.resolve() if raw_path is None else Path(str(raw_path)).resolve()
    _require(path == expected_path.resolve(), f"{description} path mismatch")
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
        "stability terminal registry schema is incompatible",
    )
    _require(
        status.get("artifact_registry_sha256") == file_fingerprint(registry_path)
        and int(status.get("artifact_registry_size", -1))
        == int(registry_path.stat().st_size),
        "stability terminal status does not bind its artifact registry",
    )
    exclusions = set(registry.get("terminal_files_excluded_to_avoid_self_reference", []))
    _require(
        exclusions == {"artifact_registry.json", "run_status.json"},
        "stability registry exclusions are incompatible",
    )
    raw_records = registry.get("records", {})
    _require(isinstance(raw_records, Mapping), "stability registry records are invalid")
    records = dict(raw_records)
    _require(
        len(records) == int(expected_count),
        f"stability terminal registry must contain exactly {expected_count} records",
    )
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in exclusions | {"artifact_registry.json"}
    }
    _require(
        set(records) == actual,
        "stability artifact registry is incomplete or contains stale records",
    )
    for relative, raw in records.items():
        _require(isinstance(raw, Mapping), f"invalid stability registry record {relative}")
        _verify_record(
            dict(raw),
            root=run_dir,
            expected_path=run_dir / relative,
            description=f"registered stability artifact {relative}",
        )
    return registry


def _no_work(value: Mapping[str, Any], description: str) -> None:
    _require(
        _zero(value.get("physical_training_performed", 0))
        and _zero(value.get("sampling_performed", 0)),
        f"{description} records physical training or sampling",
    )


def _verify_scientific_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    scientific = manifest.get("scientific_config", {})
    _require(isinstance(scientific, Mapping), "stability scientific configuration is missing")
    scientific = dict(scientific)
    _require(
        scientific.get("algorithm") == PARENT_RUN_SCHEMA
        and int(scientific.get("algorithm_version", -1)) == 1,
        "stability scientific algorithm is incompatible",
    )
    _require(
        config_fingerprint(scientific) == manifest.get("scientific_fingerprint"),
        "stability scientific fingerprint is inconsistent",
    )
    kernel = scientific.get("kernel", {})
    _require(isinstance(kernel, Mapping), "stability kernel configuration is missing")
    for key, expected in EXPECTED_KERNEL.items():
        _require(
            _same(kernel.get(key), expected),
            f"stability kernel mismatch for {key}: {kernel.get(key)!r}",
        )
    thresholds = _json_value(StabilityThresholds().to_dict())
    _require(scientific.get("thresholds") == thresholds, "stability thresholds changed")
    _require(
        scientific.get("model_schema") == "d0-boundary-smooth-potential-unet-v1"
        and int(scientific.get("model_schema_version", -1)) == 1,
        "stability model schema is incompatible",
    )
    optimization = scientific.get("optimization", {})
    _require(isinstance(optimization, Mapping), "stability optimizer configuration is missing")
    for key, expected in {
        "weight_decay": 1e-4,
        "ema_decay": 0.99,
        "grad_clip": 1.0,
        "clip_warmup_steps": 500,
        "implicit_loss_scale": PARENT_IMPLICIT_LOSS_SCALE,
        "adaptive_loss_scaling": 0,
    }.items():
        _require(_same(optimization.get(key), expected), f"stability optimizer mismatch for {key}")
    pilot = scientific.get("pilot", {})
    confirmation = scientific.get("confirmation", {})
    stream = scientific.get("stream", {})
    _require(
        isinstance(pilot, Mapping)
        and tuple(float(value) for value in pilot.get("learning_rates", []))
        == EXPECTED_PILOT_LEARNING_RATES
        and int(pilot.get("steps", -1)) == 1000
        and int(pilot.get("selection_paths", -1)) == 16
        and int(pilot.get("audit_paths", -1)) == 0,
        "stability pilot configuration is incompatible",
    )
    _require(
        isinstance(confirmation, Mapping)
        and tuple(int(value) for value in confirmation.get("model_seeds", []))
        == EXPECTED_CONFIRMATION_SEEDS
        and int(confirmation.get("steps", -1)) == 4000
        and int(confirmation.get("selection_paths", -1)) == 32
        and int(confirmation.get("audit_paths", -1)) == 32,
        "stability confirmation configuration is incompatible",
    )
    _require(
        isinstance(stream, Mapping)
        and int(stream.get("batch_size", -1)) == 64
        and int(stream.get("clusters_per_step", -1)) == 2
        and list(stream.get("anchor_bin_counts", [])) == [4, 4, 4, 4, 16]
        and int(stream.get("training_probe_banks", -1)) == 2
        and int(stream.get("training_probes_per_bank", -1)) == 4,
        "stability stream/probe configuration is incompatible",
    )
    _require(int(scientific.get("root_seed", -1)) == 260801, "stability root seed changed")
    _no_work(scientific, "stability scientific configuration")
    return scientific


def _verify_transitive_parent(parent: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        parent.get("schema")
        == "experiment12-d0-score-control-scale-repair-stability-parent-provenance"
        and int(parent.get("schema_version", -1)) == 1
        and _one(parent.get("passed")),
        "stability transitive parent provenance is incompatible",
    )
    parent_root = Path(str(parent.get("run_dir", ""))).resolve()
    recomputed = verify_parent_scale_repair_run(parent_root)
    for key, value in recomputed.items():
        _require(parent.get(key) == value, f"stability transitive parent mismatch for {key}")
    _no_work(parent, "stability transitive parent provenance")
    return recomputed


def _verify_task_result(
    result: Mapping[str, Any],
    *,
    result_path: Path,
    status_path: Path,
    expected_kind: str,
    expected_seed: int | None = None,
) -> None:
    _require(result_path.is_file() and status_path.is_file(), f"missing {expected_kind} task artifacts")
    disk_result = _load(result_path)
    status = _load(status_path)
    _require(disk_result == dict(result), f"{expected_kind} aggregate/task result mismatch")
    _require(
        status.get("status") == "complete"
        and status.get("task_kind") == expected_kind
        and status.get("task_result_sha256") == file_fingerprint(result_path),
        f"{expected_kind} task status is incomplete or unbound",
    )
    if expected_seed is not None:
        _require(
            int(status.get("model_seed", -1)) == int(expected_seed)
            and int(result.get("model_seed", -1)) == int(expected_seed),
            f"{expected_kind} task seed mismatch",
        )
    metrics = result.get("metrics", {})
    _require(isinstance(metrics, Mapping), f"{expected_kind} task metrics are missing")
    _require(
        _one(metrics.get("complete"))
        and _one(metrics.get("finite"))
        and _one(metrics.get("boundary_admissible")),
        f"{expected_kind} task is incomplete, nonfinite, or boundary-inadmissible",
    )
    _no_work(result, f"{expected_kind} task result")
    _no_work(status, f"{expected_kind} task status")


def _verify_pilot(run_dir: Path, values: Mapping[str, Mapping[str, Any]]) -> None:
    registry = values["pilot_candidates"]
    candidates = registry.get("candidates", [])
    _require(
        isinstance(candidates, Sequence) and len(candidates) == 4,
        "stability pilot candidate registry is incomplete",
    )
    for index, candidate in enumerate(candidates):
        _require(isinstance(candidate, Mapping), "invalid stability pilot candidate")
        _require(
            int(candidate.get("candidate_index", -1)) == index
            and _same(candidate.get("learning_rate"), EXPECTED_PILOT_LEARNING_RATES[index]),
            "stability pilot candidate ordering changed",
        )
        for key, kind, directory in (
            ("teacher", "implicit_teacher", "implicit_teacher"),
            ("null", "null", "null"),
        ):
            raw = candidate.get(key, {})
            _require(isinstance(raw, Mapping), f"pilot {kind} result is missing")
            root = run_dir / "pilot" / f"lr-{index:02d}" / directory
            _verify_task_result(
                dict(raw),
                result_path=root / "task_result.json",
                status_path=root / "task_status.json",
                expected_kind=kind,
            )
    recomputed = _json_value(evaluate_stability_pilot(candidates))
    _require(recomputed == values["pilot_gate"], "stability pilot gate does not recompute")
    _require(_one(recomputed.get("passed")), "stability pilot did not pass")
    selected = recomputed.get("selected_profile", {})
    _require(
        isinstance(selected, Mapping)
        and dict(selected) == values["selected_profile"]
        and _one(selected.get("selected"))
        and _same(dict(selected.get("profile", {})).get("learning_rate"), 1e-5),
        "stability selected profile is inconsistent",
    )
    binding = values["profile_binding"]
    _require(
        binding.get("selected_profile") == values["selected_profile"]
        and binding.get("selected_profile_sha256")
        == file_fingerprint(run_dir / _REQUIRED_FILES["selected_profile"])
        and binding.get("pilot_gate_sha256")
        == file_fingerprint(run_dir / _REQUIRED_FILES["pilot_gate"]),
        "stability confirmation profile binding is invalid",
    )
    failures = values["pilot_failures"]
    _require(
        int(failures.get("failure_count", -1)) == 0
        and list(failures.get("failures", [])) == [],
        "stability pilot contains task failures",
    )


def _confirmation_results(value: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    raw = value.get("task_results", [])
    _require(isinstance(raw, Sequence), f"stability {name} aggregate is invalid")
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _verify_confirmation(run_dir: Path, values: Mapping[str, Mapping[str, Any]]) -> None:
    teacher = _confirmation_results(values["teacher_confirmation"], "teacher")
    null = _confirmation_results(values["null_confirmation"], "null")
    _require(len(teacher) == len(null) == 3, "stability confirmation task set is incomplete")
    for kind, directory, results in (
        ("implicit_teacher", "implicit_teacher", teacher),
        ("null", "null", null),
    ):
        by_seed = {int(value.get("model_seed", -1)): value for value in results}
        _require(set(by_seed) == set(EXPECTED_CONFIRMATION_SEEDS), f"stability {kind} seeds changed")
        for seed in EXPECTED_CONFIRMATION_SEEDS:
            root = run_dir / "confirmation" / directory / f"seed-{seed}"
            _verify_task_result(
                by_seed[seed],
                result_path=root / "task_result.json",
                status_path=root / "task_status.json",
                expected_kind=kind,
                expected_seed=seed,
            )
            clip = dict(by_seed[seed].get("metrics", {})).get("post_warmup_clip_fraction")
            _require(
                _finite(clip) and 0.0 <= float(clip) <= 0.10,
                f"stability {kind} confirmation optimizer was not healthy",
            )
    recomputed = _json_value(
        evaluate_stability_confirmation(
            teacher,
            null,
            probe_bank_status=ProbeBankStatus.AGREE,
        )
    )
    stored = values["confirmation_gate"]
    base = {key: stored.get(key) for key in recomputed}
    _require(base == recomputed, "stability confirmation gate does not recompute")
    _require(
        stored.get("teacher_results") == teacher
        and stored.get("null_results") == null
        and stored.get("selected_profile") == values["selected_profile"],
        "stability confirmation gate does not bind its task evidence",
    )
    teacher_study = dict(stored.get("implicit_teacher_study", {}))
    null_study = dict(stored.get("null_study", {}))
    null_passes = sum(
        _one(dict(gate).get("passed"))
        for gate in null_study.get("seed_gates", [])
        if isinstance(gate, Mapping)
    )
    _require(
        _zero(stored.get("passed"))
        and _one(stored.get("optimizer_health_pass"))
        and stored.get("probe_bank_status") == ProbeBankStatus.AGREE.value
        and int(teacher_study.get("passing_seed_count", -1)) == 0
        and null_passes == 2,
        "stability confirmation is not the recorded optimizer-healthy science failure",
    )
    failures = values["confirmation_failures"]
    _require(
        int(failures.get("failure_count", -1)) == 0
        and list(failures.get("failures", [])) == [],
        "stability confirmation contains task failures",
    )


def verify_parent_stability_run(path: str | Path) -> dict[str, Any]:
    """Verify the immutable streamed-control result authorizing classification."""

    run_dir = Path(path).resolve()
    _require(run_dir.is_dir(), f"parent stability run does not exist: {run_dir}")
    paths = {name: run_dir / relative for name, relative in _REQUIRED_FILES.items()}
    missing = [str(value) for value in paths.values() if not value.is_file()]
    _require(not missing, "parent stability artifacts are missing: " + ", ".join(missing))
    values = {name: _load(value) for name, value in paths.items() if name != "registry"}
    manifest, status = values["manifest"], values["status"]
    _require(
        manifest.get("schema") == PARENT_RUN_SCHEMA
        and status.get("schema") == PARENT_RUN_SCHEMA
        and int(manifest.get("schema_version", -1)) == 1
        and int(status.get("schema_version", -1)) == 1,
        "parent stability schema is incompatible",
    )
    _require(
        Path(str(manifest.get("run_dir", ""))).resolve() == run_dir,
        "parent stability manifest run directory is inconsistent",
    )
    _require(
        status.get("status") == "complete"
        and status.get("outcome") == "gate_failed"
        and status.get("decision") == EXPECTED_PARENT_DECISION
        and status.get("probe_bank_status") == ProbeBankStatus.AGREE.value
        and status.get("required_gate") == "controls"
        and _zero(status.get("required_gate_pass")),
        "parent stability outcome is not the recorded implicit-objective failure",
    )
    registry = _verify_registry(
        run_dir,
        registry_path=paths["registry"],
        status_path=paths["status"],
    )
    scientific = _verify_scientific_manifest(manifest)
    _require(
        manifest.get("parent_provenance_sha256")
        == file_fingerprint(paths["parent_provenance"]),
        "stability manifest does not bind transitive provenance",
    )
    _verify_transitive_parent(values["parent_provenance"])

    stein = _json_value(evaluate_stein_identity_preflight(values["stein_raw"]))
    _require(
        stein == values["stein_gate"] and _one(stein.get("passed")),
        "stability Stein identity gate does not recompute as passing",
    )
    preflight = values["preflight_gate"]
    _require(
        preflight.get("schema") == PARENT_RUN_SCHEMA + "-preflight-gate"
        and _one(preflight.get("passed"))
        and preflight.get("provenance") == values["parent_provenance"]
        and preflight.get("stein_identity") == values["stein_raw"]
        and preflight.get("stein_identity_gate") == values["stein_gate"]
        and _one(dict(preflight.get("stream_replay", {})).get("passed")),
        "stability aggregate preflight is inconsistent",
    )
    forensic = values["forensic"]
    _require(
        forensic.get("schema") == PARENT_RUN_SCHEMA + "-parent-forensic-replay"
        and _one(forensic.get("complete"))
        and _one(forensic.get("finite"))
        and _zero(forensic.get("eligible_for_gate")),
        "stability parent forensic replay is incomplete or non-advisory",
    )
    _verify_pilot(run_dir, values)
    _verify_confirmation(run_dir, values)

    report = _json_value(
        evaluate_stability_workflow(
            provenance=values["parent_provenance"],
            stein=values["preflight_gate"],
            pilot=values["pilot_gate"],
            confirmation=values["confirmation_gate"],
            require_gate="controls",
        )
    )
    _require(report == values["report"], "stability terminal report does not recompute")
    _require(
        dict(report.get("decision", {})) == values["decision"]
        and values["decision"].get("decision") == EXPECTED_PARENT_DECISION,
        "stability terminal decision is inconsistent",
    )
    for name, value in values.items():
        _no_work(value, f"stability {name}")
    _require(
        _zero(status.get("physical_training_authorized"))
        and _zero(status.get("sampling_authorized")),
        "stability parent authorized physical training or sampling",
    )

    selected = dict(values["selected_profile"].get("profile", {}))
    schedule = values["parent_provenance"].get("schedule_metadata", {})
    _require(isinstance(schedule, Mapping), "stability parent schedule metadata is missing")
    horizon = schedule.get("horizon", values["parent_provenance"].get("horizon"))
    _require(_finite(horizon) and float(horizon) > 0.0, "stability parent horizon is invalid")
    null_seed_gates = dict(values["confirmation_gate"].get("null_study", {})).get(
        "seed_gates", []
    )
    return {
        "schema": PARENT_RUN_SCHEMA + "-density-ratio-parent-provenance",
        "schema_version": 1,
        "passed": 1,
        "run_dir": str(run_dir),
        "scientific_fingerprint": str(manifest.get("scientific_fingerprint", "")),
        "runtime_fingerprint": str(manifest.get("runtime_fingerprint", "")),
        "source_fingerprint": str(manifest.get("source_fingerprint", "")),
        "scientific_config": scientific,
        "kernel": dict(scientific["kernel"]),
        "model": dict(values["parent_provenance"].get("model", {})),
        "schedule_metadata": dict(schedule),
        "horizon": float(horizon),
        "parent_decision": EXPECTED_PARENT_DECISION,
        "probe_bank_status": ProbeBankStatus.AGREE.value,
        "terminal_registry_record_count": len(dict(registry["records"])),
        "pilot_pass": 1,
        "selected_learning_rate": float(selected["learning_rate"]),
        "confirmation_optimizer_health_pass": 1,
        "teacher_passing_seed_count": int(
            dict(values["confirmation_gate"].get("implicit_teacher_study", {})).get(
                "passing_seed_count", 0
            )
        ),
        "null_passing_seed_count": sum(
            _one(dict(gate).get("passed"))
            for gate in null_seed_gates
            if isinstance(gate, Mapping)
        ),
        "transitive_parent_provenance": values["parent_provenance"],
        "artifacts": {name: _record(value) for name, value in paths.items()},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
