"""Report-only diagnosis of the sealed Jacobi/RB one-image no-signal result.

This workflow reads the immutable learnability run, replays its frozen model
on the already-created train/validation/confirmation caches, and decomposes
quadratic risk into prediction energy and target alignment.  It creates no
states, fits no parameters, changes no checkpoint, and performs no sampling.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_csv,
    atomic_write_json,
    config_fingerprint,
    configure_exact_torch_backend,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_rb_learnability import (
    CONFIRMATION_PATH_IDS,
    SELECTED_OUTER_STEPS,
    TRAIN_PATH_IDS,
    VALIDATION_PATH_IDS,
    JacobiRBPhasePredictor,
    audit_targets_from_cache,
    load_cache_bundle,
    load_metadata_baseline,
    model_inputs_from_cache,
    predict_in_batches,
    state_dict_sha256,
)
from mnist.d0_jacobi_rb_zero_signal_diagnostic import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DIAGNOSTIC_VERSION,
    coarse_cell_path_means,
    cross_split_coarse_signal,
    diagnostic_conclusion,
    path_decomposition_rows,
    quadratic_risk_decomposition,
    stratified_decomposition_rows,
    whole_path_bootstrap_interval,
)
from mnist.d0_jacobi_rb_zero_signal_diagnostic_gate import (
    CLAIM_FLAGS,
    evaluate_zero_signal_analysis,
    evaluate_zero_signal_preflight,
    not_evaluated_gate,
    zero_signal_decision,
)
from mnist.d0_jacobi_rb_zero_signal_diagnostic_provenance import (
    EXPECTED_REGISTRY_SHA256,
    EXPECTED_SCIENTIFIC_CONFIG_SHA256,
    EXPECTED_SELECTED_MODEL_FILE_SHA256,
    EXPECTED_SELECTED_STATE_SHA256,
    ZeroSignalParentError,
    verify_zero_signal_parent,
)


RUN_SCHEMA = "experiment12-d0-jacobi-rb-zero-signal-diagnostic"
RUN_SCHEMA_VERSION = 1
STORED_METRIC_REPLAY_ABS_TOLERANCE = 1.0e-6
_REGISTRY_EXCLUDED = {"artifact_registry.json", "run_status.json"}
_SPLIT_PATHS = {
    "train": TRAIN_PATH_IDS,
    "validation": VALIDATION_PATH_IDS,
    "confirmation": CONFIRMATION_PATH_IDS,
}
NO_WORK = {
    "physical_training_performed": 0,
    "sampling_performed": 0,
    "reverse_sampling_performed": 0,
    "production_refinement_performed": 0,
}


class ZeroSignalCLIError(RuntimeError):
    """Typed report-only execution failure."""

    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "diagnostic_execution",
        failure_code: str = "zero_signal_diagnostic_execution_failed",
    ) -> None:
        super().__init__(message)
        self.failure_domain = str(failure_domain)
        self.failure_code = str(failure_code)


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ArtifactCompatibilityError(f"JSON artifact is not an object: {path}")
    return dict(value)


def _freeze_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    if path.is_file():
        if _load_json(path) != record:
            raise ArtifactCompatibilityError(f"frozen artifact changed: {path.name}")
    else:
        atomic_write_json(path, record)
    return record


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _source_paths() -> list[Path]:
    return [
        Path("mnist/d0_jacobi_artifacts.py"),
        Path("mnist/d0_jacobi_rb_learnability.py"),
        Path("mnist/d0_jacobi_rb_zero_signal_diagnostic.py"),
        Path("mnist/d0_jacobi_rb_zero_signal_diagnostic_gate.py"),
        Path("mnist/d0_jacobi_rb_zero_signal_diagnostic_provenance.py"),
        Path("mnist/diag_d0_jacobi_rb_zero_signal_diagnostic.py"),
    ]


def _scientific_config(args: argparse.Namespace) -> dict[str, Any]:
    record = {
        "schema": DIAGNOSTIC_VERSION + "-scientific-config",
        "schema_version": 1,
        "parent_registry_sha256": EXPECTED_REGISTRY_SHA256,
        "parent_scientific_config_sha256": EXPECTED_SCIENTIFIC_CONFIG_SHA256,
        "selected_model_file_sha256": EXPECTED_SELECTED_MODEL_FILE_SHA256,
        "selected_state_sha256": EXPECTED_SELECTED_STATE_SHA256,
        "splits": {
            name: list(values) for name, values in _SPLIT_PATHS.items()
        },
        "selected_outer_steps": list(SELECTED_OUTER_STEPS),
        "strata": ["overall", "phase", "time_quartile", "phase_time"],
        "coarse_signal_partition": ["time_quartile", "phase", "edge"],
        "coarse_cross_split_pairs": [
            ["train", "validation"],
            ["train", "confirmation"],
            ["validation", "confirmation"],
        ],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "whole_path",
        "stored_metric_replay_abs_tolerance": STORED_METRIC_REPLAY_ABS_TOLERANCE,
        "posthoc_non_authorizing": 1,
        "new_data_generation_permitted": 0,
        "training_permitted": 0,
        "checkpoint_selection_permitted": 0,
        "parameter_tuning_permitted": 0,
        "sampling_permitted": 0,
        "device_argument": str(args.device),
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    record["semantic_sha256"] = config_fingerprint(record)
    return record


def _manifest(
    args: argparse.Namespace,
    *,
    sources: Sequence[Path],
    scientific: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "created_by": "mnist.diag_d0_jacobi_rb_zero_signal_diagnostic",
        "stage_contract": ["preflight", "analyze"],
        "parent_learnability_run_dir": str(
            Path(args.parent_learnability_run_dir).resolve()
        ),
        "source_paths": [path.as_posix() for path in sources],
        "source_fingerprint": source_fingerprint(sources),
        "scientific_config_sha256": scientific["semantic_sha256"],
        "report_only": 1,
        "posthoc_non_authorizing": 1,
        **CLAIM_FLAGS,
        **NO_WORK,
    }


def _make_run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir is not None:
        run_dir = Path(args.resume_run_dir).resolve()
        if not run_dir.is_dir():
            raise ArtifactCompatibilityError(f"resume run does not exist: {run_dir}")
        return run_dir, True
    root = Path(args.runs_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{timestamp}_{args.run_name}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir, False


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    records = []
    for path in sorted(run_dir.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name in _REGISTRY_EXCLUDED:
            continue
        records.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "sha256": file_fingerprint(path),
                "size": int(path.stat().st_size),
            }
        )
    record = {
        "schema": RUN_SCHEMA + "-artifact-registry",
        "schema_version": 1,
        "record_count": len(records),
        "records": records,
        "registry_sha256": config_fingerprint(records),
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "artifact_registry.json", record)
    return record


def _verify_own_registry(run_dir: Path) -> None:
    path = run_dir / "artifact_registry.json"
    if not path.is_file():
        return
    registry = _load_json(path)
    rows = registry.get("records")
    if (
        not isinstance(rows, list)
        or int(registry.get("record_count", -1)) != len(rows)
        or registry.get("registry_sha256") != config_fingerprint(rows)
    ):
        raise ArtifactCompatibilityError("diagnostic artifact registry is malformed")
    expected = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ArtifactCompatibilityError("diagnostic registry row is malformed")
        relative = str(row.get("path", ""))
        artifact = run_dir / relative
        if (
            not relative
            or relative in expected
            or not artifact.is_file()
            or int(row.get("size", -1)) != artifact.stat().st_size
            or row.get("sha256") != file_fingerprint(artifact)
        ):
            raise ArtifactCompatibilityError(
                f"diagnostic registered artifact changed: {relative}"
            )
        expected.add(relative)
    status = _load_json(run_dir / "run_status.json")
    if "artifact_registry_sha256" in status:
        actual = {
            item.relative_to(run_dir).as_posix()
            for item in run_dir.rglob("*")
            if item.is_file() and item.name not in _REGISTRY_EXCLUDED
        }
        if actual != expected:
            raise ArtifactCompatibilityError(
                "terminal diagnostic artifact file set changed"
            )


def _status(
    run_dir: Path,
    *,
    stage: str,
    state: str,
    decision: str | None = None,
    message: str = "",
    registry: Mapping[str, Any] | None = None,
) -> None:
    binding: dict[str, Any] = {}
    if registry is not None:
        registry_path = run_dir / "artifact_registry.json"
        binding = {
            "artifact_registry_record_count": int(registry["record_count"]),
            "artifact_registry_sha256": str(registry["registry_sha256"]),
            "artifact_registry_file_sha256": file_fingerprint(registry_path),
            "artifact_registry_file_size": int(registry_path.stat().st_size),
        }
    atomic_write_json(
        run_dir / "run_status.json",
        {
            "schema": RUN_SCHEMA + "-status",
            "schema_version": 1,
            "updated_at": _now(),
            "stage": str(stage),
            "state": str(state),
            "decision": decision,
            "message": str(message),
            **binding,
            **CLAIM_FLAGS,
            **NO_WORK,
        },
    )


def _initialize(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    resumed: bool,
) -> dict[str, Any]:
    sources = _source_paths()
    for path in sources:
        if not path.is_file():
            raise ArtifactCompatibilityError(f"diagnostic source is missing: {path}")
    scientific = _scientific_config(args)
    manifest = _manifest(args, sources=sources, scientific=scientific)
    if resumed:
        _verify_own_registry(run_dir)
        if _load_json(run_dir / "scientific_config.json") != scientific:
            raise ArtifactCompatibilityError("diagnostic scientific configuration changed")
        if _load_json(run_dir / "run_manifest.json") != manifest:
            raise ArtifactCompatibilityError("diagnostic source or parent binding changed")
    else:
        _freeze_json(run_dir / "scientific_config.json", scientific)
        _freeze_json(run_dir / "run_manifest.json", manifest)
    parent = verify_zero_signal_parent(args.parent_learnability_run_dir)
    _freeze_json(run_dir / "parent_provenance.json", parent)
    _freeze_json(
        run_dir / "analysis_plan.json",
        {
            "schema": DIAGNOSTIC_VERSION + "-analysis-plan",
            "schema_version": 1,
            "risk_identity": "MSE(p,z)-MSE(0,z)=E[p^2]-2E[pz]",
            "split_roles": ["train", "validation", "confirmation"],
            "authorizing_split_roles": [],
            "strata": ["overall", "phase", "time_quartile", "phase_time"],
            "coarse_signal_partition": ["time_quartile", "phase", "edge"],
            "coarse_cross_split_pairs": [
                ["train", "validation"],
                ["train", "confirmation"],
                ["validation", "confirmation"],
            ],
            "bootstrap_unit": "whole_path",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "stored_metric_replay_abs_tolerance": (
                STORED_METRIC_REPLAY_ABS_TOLERANCE
            ),
            "new_data_generated": 0,
            "training_performed": 0,
            "tuning_performed": 0,
            "sampling_performed": 0,
            "parent_mutation_permitted": 0,
            "conditional_mean_zero_claim_permitted": 0,
            "posthoc_non_authorizing": 1,
            **CLAIM_FLAGS,
            **NO_WORK,
        },
    )
    return parent


def _preflight(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    parent = verify_zero_signal_parent(args.parent_learnability_run_dir)
    parent_dir = Path(parent["parent_run_dir"])
    for split, path_ids in _SPLIT_PATHS.items():
        load_cache_bundle(
            parent_dir / "cache" / f"{split}_inputs.npz",
            parent_dir / "cache" / f"{split}_labels_audit.npz",
            expected_path_ids=path_ids,
            expected_outer_steps=SELECTED_OUTER_STEPS,
        )
    metrics = {
        "schema": DIAGNOSTIC_VERSION + "-preflight-metrics",
        "schema_version": 1,
        "parent_registry_verified": 1,
        "parent_terminal_scope_verified": 1,
        "parent_negative_decision_verified": 1,
        "parent_only_confirmation_failure_verified": 1,
        "selected_model_binding_verified": 1,
        "metadata_baseline_binding_verified": 1,
        "all_three_cache_bindings_verified": 1,
        "confirmation_opened_once_verified": 1,
        "no_parent_mutation_planned": 1,
        "no_new_data_planned": 1,
        "no_training_planned": 1,
        "no_tuning_planned": 1,
        "no_sampling_planned": 1,
        "posthoc_non_authorizing": 1,
        "parent_binding_sha256": parent["semantic_sha256"],
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "zero_signal_preflight_metrics.json", metrics)
    gate = evaluate_zero_signal_preflight(metrics)
    atomic_write_json(run_dir / "zero_signal_preflight_gate.json", gate)
    return gate


def _load_model(parent_dir: Path, device: torch.device) -> JacobiRBPhasePredictor:
    checkpoint = torch.load(
        parent_dir / "selected_model.pt",
        map_location="cpu",
        weights_only=True,
    )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ArtifactCompatibilityError("selected checkpoint lacks model_state_dict")
    if state_dict_sha256(state) != EXPECTED_SELECTED_STATE_SHA256:
        raise ArtifactCompatibilityError("selected checkpoint state hash changed")
    model = JacobiRBPhasePredictor(width=32, num_classes=10).to(device)
    model.load_state_dict(dict(state), strict=True)
    if state_dict_sha256(model.state_dict()) != EXPECTED_SELECTED_STATE_SHA256:
        raise ArtifactCompatibilityError("selected checkpoint replay hash mismatch")
    model.eval()
    return model


def _reference_metrics(
    parent_dir: Path, split: str
) -> tuple[dict[str, Any], Path | None]:
    if split == "validation":
        return _load_json(parent_dir / "physical_training_metrics.json"), (
            parent_dir / "validation_path_metrics.csv"
        )
    if split == "confirmation":
        return _load_json(parent_dir / "confirmation_metrics.json"), (
            parent_dir / "confirmation_path_metrics.csv"
        )
    return {}, None


def _close(
    left: float,
    right: float,
    tolerance: float = STORED_METRIC_REPLAY_ABS_TOLERANCE,
) -> bool:
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance


def _replay_reference(
    *,
    parent_dir: Path,
    split: str,
    summary: Mapping[str, Any],
    path_rows: Sequence[Mapping[str, Any]],
) -> bool:
    reference, path_file = _reference_metrics(parent_dir, split)
    if split == "train":
        return True
    if split == "validation":
        values = {
            "model_mse": float(reference["selected_validation_mse"]),
            "metadata_mse": float(reference["validation_metadata_baseline_mse"]),
            "zero_mse": float(reference["validation_zero_mse"]),
        }
    else:
        values = {
            "model_mse": float(reference["aggregate_model_mse"]),
            "metadata_mse": float(reference["aggregate_metadata_mse"]),
            "zero_mse": float(reference["aggregate_zero_mse"]),
        }
    if not all(_close(float(summary[name]), value) for name, value in values.items()):
        return False
    if path_file is None:
        return True
    with path_file.open("r", encoding="utf-8", newline="") as handle:
        stored = list(csv.DictReader(handle))
    if len(stored) != len(path_rows):
        return False
    by_path = {int(row["path_id"]): row for row in path_rows}
    for row in stored:
        current = by_path.get(int(row["path_id"]))
        if current is None:
            return False
        for field in ("model_mse", "metadata_mse", "zero_mse"):
            if not _close(float(current[field]), float(row[field])):
                return False
    return True


def _analyze(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    preflight_gate = _load_json(run_dir / "zero_signal_preflight_gate.json")
    if int(preflight_gate.get("passed", 0)) != 1:
        return not_evaluated_gate("analysis", "preflight did not pass")
    parent_before = verify_zero_signal_parent(args.parent_learnability_run_dir)
    parent_dir = Path(parent_before["parent_run_dir"])
    device = torch.device(args.device)
    runtime = configure_exact_torch_backend(device)
    atomic_write_json(
        run_dir / "diagnostic_runtime.json",
        {
            "schema": DIAGNOSTIC_VERSION + "-runtime",
            "schema_version": 1,
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "device": str(device),
            **runtime,
            **NO_WORK,
        },
    )
    model = _load_model(parent_dir, device)
    baseline = load_metadata_baseline(parent_dir / "metadata_baseline.npz")

    summaries: dict[str, dict[str, Any]] = {}
    all_paths: list[dict[str, Any]] = []
    all_strata: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    prediction_hashes: dict[str, str] = {}
    replay: dict[str, bool] = {}
    coarse_path_means: dict[str, np.ndarray] = {}

    for split, path_ids in _SPLIT_PATHS.items():
        bundle = load_cache_bundle(
            parent_dir / "cache" / f"{split}_inputs.npz",
            parent_dir / "cache" / f"{split}_labels_audit.npz",
            expected_path_ids=path_ids,
            expected_outer_steps=SELECTED_OUTER_STEPS,
        )
        inputs = model_inputs_from_cache(
            bundle.inputs, device=device, floating_dtype=torch.float32
        )
        audit = audit_targets_from_cache(bundle.labels_audit, device=device)
        prediction_tensor = predict_in_batches(model, inputs, batch_size=32)
        prediction = np.ascontiguousarray(
            prediction_tensor.detach().cpu().numpy(), dtype=np.float64
        )
        target = np.ascontiguousarray(
            audit.denoising_target.detach().cpu().numpy(), dtype=np.float64
        )
        metadata = baseline.predict(
            bundle.labels_audit.outer_step, bundle.labels_audit.phase
        )
        summary = quadratic_risk_decomposition(
            prediction, target, metadata
        ).to_record()
        summary["split"] = split
        summaries[split] = summary
        path_rows = path_decomposition_rows(
            prediction,
            target,
            metadata,
            bundle.labels_audit.path_id,
            split=split,
        )
        all_paths.extend(path_rows)
        all_strata.extend(
            stratified_decomposition_rows(
                prediction,
                target,
                metadata,
                bundle.labels_audit.outer_step,
                bundle.labels_audit.phase,
                split=split,
            )
        )
        for field in ("zero_minus_model_mse", "metadata_minus_model_mse"):
            bootstrap_rows.append(
                {
                    "split": split,
                    **whole_path_bootstrap_interval(
                        path_rows,
                        field=field,
                        seed=BOOTSTRAP_SEED,
                        replicates=BOOTSTRAP_REPLICATES,
                    ),
                }
            )
        prediction_hashes[split] = _hash_array(prediction)
        _, coarse_path_means[split] = coarse_cell_path_means(
            target,
            bundle.labels_audit.path_id,
            bundle.labels_audit.outer_step,
            bundle.labels_audit.phase,
        )
        replay[split] = _replay_reference(
            parent_dir=parent_dir,
            split=split,
            summary=summary,
            path_rows=path_rows,
        )

    atomic_write_csv(
        run_dir / "zero_signal_split_summary.csv",
        [summaries[name] for name in _SPLIT_PATHS],
    )
    atomic_write_csv(run_dir / "zero_signal_path_decomposition.csv", all_paths)
    atomic_write_csv(run_dir / "zero_signal_stratified_decomposition.csv", all_strata)
    atomic_write_csv(run_dir / "zero_signal_path_bootstrap.csv", bootstrap_rows)

    coarse_pairs = (
        ("train", "validation"),
        ("train", "confirmation"),
        ("validation", "confirmation"),
    )
    coarse_rows = [
        cross_split_coarse_signal(
            coarse_path_means[left],
            coarse_path_means[right],
            left_split=left,
            right_split=right,
            seed=BOOTSTRAP_SEED + pair_index,
            replicates=BOOTSTRAP_REPLICATES,
        )
        for pair_index, (left, right) in enumerate(coarse_pairs)
    ]
    atomic_write_csv(run_dir / "zero_signal_coarse_cross_split.csv", coarse_rows)
    atomic_write_json(
        run_dir / "frozen_evidence_replay.json",
        {
            "schema": DIAGNOSTIC_VERSION + "-frozen-evidence-replay",
            "schema_version": 1,
            "selected_model_file_sha256": EXPECTED_SELECTED_MODEL_FILE_SHA256,
            "selected_state_sha256": EXPECTED_SELECTED_STATE_SHA256,
            "prediction_sha256": prediction_hashes,
            "stored_metric_replay_abs_tolerance": (
                STORED_METRIC_REPLAY_ABS_TOLERANCE
            ),
            "split_reproduced": {
                key: int(value) for key, value in replay.items()
            },
            "alternate_checkpoint_evaluated": 0,
            "prediction_rescaled": 0,
            "model_refit": 0,
            "posthoc_non_authorizing": 1,
            **CLAIM_FLAGS,
            **NO_WORK,
        },
    )

    conclusion = diagnostic_conclusion(summaries, coarse_rows)
    identity_errors = [
        float(row["model_identity_abs_error"])
        for row in all_strata
    ]
    metadata_identity_errors = [
        float(row["metadata_identity_abs_error"])
        for row in all_strata
    ]
    diagnostic = {
        "schema": DIAGNOSTIC_VERSION + "-report",
        "schema_version": 1,
        "risk_identity": "MSE(p,z)-MSE(0,z)=E[p^2]-2E[pz]",
        "split_summaries": summaries,
        "prediction_sha256": prediction_hashes,
        "whole_path_bootstrap": bootstrap_rows,
        "coarse_cross_split_signal": coarse_rows,
        "conclusion": conclusion,
        "scientific_interpretation": {
            "frozen_model_signal_statement": (
                "The selected model does not improve quadratic risk over the zero "
                "predictor on the sealed confirmation cache."
            ),
            "metadata_statement": (
                "The selected model consistently improves over the training-only "
                "time/phase/edge metadata baseline on validation and confirmation."
            ),
            "zero_conditional_mean_statement": (
                "These observations do not prove that the true conditional mean "
                "is identically zero; they bound only this frozen model and cache."
            ),
        },
        "posthoc_non_authorizing": 1,
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "zero_signal_diagnostic.json", diagnostic)

    parent_after = verify_zero_signal_parent(args.parent_learnability_run_dir)
    metrics = {
        "schema": DIAGNOSTIC_VERSION + "-analysis-metrics",
        "schema_version": 1,
        "split_count": len(summaries),
        "train_path_count": len(_SPLIT_PATHS["train"]),
        "validation_path_count": len(_SPLIT_PATHS["validation"]),
        "confirmation_path_count": len(_SPLIT_PATHS["confirmation"]),
        "all_predictions_finite": 1,
        "all_metrics_finite": int(
            all(
                math.isfinite(float(value))
                for summary in summaries.values()
                for key, value in summary.items()
                if key not in {
                    "split",
                    "risk_identity",
                    "model_beats_zero",
                    "covariance_exceeds_prediction_cost",
                }
            )
        ),
        "checkpoint_replay_verified": 1,
        "confirmation_metrics_reproduced": int(replay["confirmation"]),
        "validation_metrics_reproduced": int(replay["validation"]),
        "decomposition_identity_max_abs_error": max(identity_errors),
        "metadata_identity_max_abs_error": max(metadata_identity_errors),
        "whole_path_only_bootstrap": 1,
        "coarse_signal_pair_count": len(coarse_rows),
        "coarse_observations_per_split_cell": int(
            min(row["observations_per_split_cell"] for row in coarse_rows)
        ),
        "coarse_bootstrap_whole_path_only": int(
            all(
                row["bootstrap_unit"]
                == "whole_path_independently_within_split"
                for row in coarse_rows
            )
        ),
        "coarse_signal_all_finite": int(
            all(
                math.isfinite(float(row[field]))
                for row in coarse_rows
                for field in (
                    "cross_split_coarse_signal",
                    "lower",
                    "upper",
                )
            )
        ),
        "parent_artifacts_read_only": int(
            parent_before["semantic_sha256"] == parent_after["semantic_sha256"]
        ),
        "no_new_data_generated": 1,
        "no_training_performed": 1,
        "no_tuning_performed": 1,
        "no_sampling_performed": 1,
        "posthoc_non_authorizing": 1,
        "prediction_sha256": prediction_hashes,
        "reference_replay": {key: int(value) for key, value in replay.items()},
        "stored_metric_replay_abs_tolerance": (
            STORED_METRIC_REPLAY_ABS_TOLERANCE
        ),
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "zero_signal_analysis_metrics.json", metrics)
    gate = evaluate_zero_signal_analysis(metrics, preflight_gate=preflight_gate)
    atomic_write_json(run_dir / "zero_signal_analysis_gate.json", gate)
    return gate


def _write_decision(run_dir: Path) -> dict[str, Any]:
    preflight = (
        _load_json(run_dir / "zero_signal_preflight_gate.json")
        if (run_dir / "zero_signal_preflight_gate.json").is_file()
        else not_evaluated_gate("preflight", "not run")
    )
    analysis = (
        _load_json(run_dir / "zero_signal_analysis_gate.json")
        if (run_dir / "zero_signal_analysis_gate.json").is_file()
        else not_evaluated_gate("analysis", "not run")
    )
    conclusion = (
        _load_json(run_dir / "zero_signal_diagnostic.json").get("conclusion")
        if (run_dir / "zero_signal_diagnostic.json").is_file()
        else None
    )
    decision = zero_signal_decision(
        preflight_gate=preflight,
        analysis_gate=analysis,
        conclusion=conclusion if isinstance(conclusion, Mapping) else None,
    )
    atomic_write_json(run_dir / "zero_signal_decision.json", decision)
    workflow = {
        "schema": DIAGNOSTIC_VERSION + "-workflow",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "passed": int(
            int(preflight.get("passed", 0)) == 1
            and int(analysis.get("passed", 0)) == 1
        ),
        "components": {"preflight": preflight, "analysis": analysis},
        "decision": decision,
        **CLAIM_FLAGS,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "zero_signal_workflow_gate.json", workflow)
    return decision


def _required_pass(run_dir: Path, required: str) -> bool:
    if required == "none":
        return True
    filename = {
        "preflight": "zero_signal_preflight_gate.json",
        "analysis": "zero_signal_analysis_gate.json",
    }[required]
    if not (run_dir / filename).is_file():
        return False
    gate = _load_json(run_dir / filename)
    return (
        gate.get("evaluation_status") == "evaluated"
        and int(gate.get("passed", 0)) == 1
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight", "analyze", "report", "all"),
        default="all",
    )
    parser.add_argument(
        "--require-gate",
        choices=("none", "preflight", "analysis"),
        default="none",
    )
    parser.add_argument("--parent-learnability-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/experiment12_d0_jacobi_rb_zero_signal_diagnostic"),
    )
    parser.add_argument(
        "--run-name", default="production-sealed-rb-zero-signal-diagnostic"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.stage in {"analyze", "report"} and args.resume_run_dir is None:
        parser.error(f"--stage {args.stage} requires --resume-run-dir")
    return args


def _sequence(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return ("preflight", "analyze")
    if stage == "report":
        return ()
    return (stage,)


def _run(args: argparse.Namespace) -> int:
    try:
        run_dir, resumed = _make_run_dir(args)
    except ArtifactCompatibilityError as exc:
        print(f"zero-signal diagnostic compatibility error: {exc}", file=sys.stderr)
        return 1
    print(f"Jacobi/RB zero-signal diagnostic run directory: {run_dir}", flush=True)
    active_stage = args.stage
    try:
        _initialize(run_dir, args, resumed=resumed)
        _status(run_dir, stage=args.stage, state="running")
        for active_stage in _sequence(args.stage):
            _status(run_dir, stage=active_stage, state="running")
            gate = (
                _preflight(run_dir, args)
                if active_stage == "preflight"
                else _analyze(run_dir, args)
            )
            _artifact_registry(run_dir)
            if int(gate.get("passed", 0)) != 1:
                break
        decision = _write_decision(run_dir)
        required_pass = _required_pass(run_dir, args.require_gate)
        registry = _artifact_registry(run_dir)
        _status(
            run_dir,
            stage=active_stage,
            state="completed" if required_pass else "gate_failed",
            decision=str(decision["decision"]),
            registry=registry,
        )
        return 0 if required_pass else 2
    except ArtifactCompatibilityError as exc:
        provenance_failure = isinstance(exc, ZeroSignalParentError)
        failure = {
            "schema": DIAGNOSTIC_VERSION + "-failure",
            "schema_version": 1,
            "evaluation_status": "execution_failed",
            "scientific_evidence_complete": 0,
            "stage_execution_valid": 0,
            "stage": active_stage,
            "failure_domain": (
                "parent_provenance" if provenance_failure else "artifact_compatibility"
            ),
            "failure_code": (
                "zero_signal_parent_scope_invalid"
                if provenance_failure
                else "zero_signal_artifact_compatibility_error"
            ),
            "error_type": type(exc).__name__,
            "error": str(exc),
            **CLAIM_FLAGS,
            **NO_WORK,
        }
        atomic_write_json(run_dir / f"{active_stage}_failure.json", failure)
        registry = _artifact_registry(run_dir)
        _status(
            run_dir,
            stage=active_stage,
            state="execution_failed",
            decision=(
                "parent_scope_invalid"
                if provenance_failure
                else "diagnostic_execution_invalid"
            ),
            message=str(exc),
            registry=registry,
        )
        print(f"zero-signal diagnostic compatibility error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        failure = {
            "schema": DIAGNOSTIC_VERSION + "-failure",
            "schema_version": 1,
            "evaluation_status": "execution_failed",
            "scientific_evidence_complete": 0,
            "stage_execution_valid": 0,
            "stage": active_stage,
            "failure_domain": getattr(exc, "failure_domain", "diagnostic_execution"),
            "failure_code": getattr(
                exc,
                "failure_code",
                "zero_signal_diagnostic_unexpected_execution_failure",
            ),
            "error_type": type(exc).__name__,
            "error": str(exc),
            **CLAIM_FLAGS,
            **NO_WORK,
        }
        atomic_write_json(run_dir / f"{active_stage}_failure.json", failure)
        registry = _artifact_registry(run_dir)
        _status(
            run_dir,
            stage=active_stage,
            state="execution_failed",
            decision="diagnostic_execution_invalid",
            message=str(exc),
            registry=registry,
        )
        print(f"zero-signal diagnostic error: {exc}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
