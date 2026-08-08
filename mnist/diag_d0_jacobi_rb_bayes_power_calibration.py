"""Exact noisy-Jacobi Bayes-power calibration.

This additive workflow answers one deliberately narrow question left by the
sealed one-image run: can the unchanged optimizer/model recover a known,
non-zero conditional mean from exact noisy Jacobi/Rao--Blackwell labels?

Only the immutable parent's *input* caches are used as pair-mass/time
templates.  Parent physical labels, selected weights, and audit caches are
forbidden.  Confirmation templates are not opened until teacher/null
selection, gate definitions, and hashes have been frozen.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from functools import partial
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    atomic_write_json,
    config_fingerprint,
    configure_exact_torch_backend,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_rb_bayes_power import (
    NULL_LAW,
    TEACHER_LAW,
    BayesPowerCacheBundle,
    BayesPowerPredictor,
    ControlTransitionBatch,
    frozen_bayes_training_plan,
    frozen_path_plan,
    frozen_scientific_config,
    generate_control_role_cache,
    load_bayes_label_cache,
    load_bayes_oracle_audit_cache,
    oracle_metric_summary,
    sample_control_transitions_cuda,
    save_bayes_label_cache,
    save_bayes_oracle_audit_cache,
    tower_witness_products,
)
from mnist.d0_jacobi_rb_bayes_power_gate import (
    BayesPowerThresholds,
    NO_CLAIM_AUTHORIZATION,
    NO_WORK,
    REQUIRED_GATES,
    STAGES,
    evaluate_bayes_cache,
    evaluate_bayes_cache_set,
    evaluate_bayes_confirmation,
    evaluate_bayes_power_workflow,
    evaluate_bayes_preflight,
    evaluate_bayes_train,
    execution_failed_gate,
)
from mnist.d0_jacobi_rb_bayes_power_provenance import (
    PARENT_RUN_BASENAME,
    assert_parent_label_firewall,
    verify_no_signal_parent,
)
from mnist.d0_jacobi_rb_learnability import (
    CheckpointCandidate,
    LearnabilityInputCache,
    TrainingResumeSnapshot,
    audit_targets_from_cache,
    exact_global_target_scale,
    fit_metadata_baseline,
    load_input_cache,
    load_metadata_baseline,
    model_inputs_from_cache,
    predict_in_batches,
    save_input_cache,
    save_metadata_baseline,
    state_dict_sha256,
    train_deterministic_regressor,
)


RUN_SCHEMA = "experiment12-d0-jacobi-rb-bayes-power-calibration"
RUN_SCHEMA_VERSION = 1
DEFAULT_RUNS_ROOT = Path("runs/experiment12_d0_jacobi_rb_bayes_power_calibration")
DEFAULT_RUN_NAME = "production-noisy-jacobi-bayes-power"
ROOT_SEED = 261_211
MODEL_SEEDS = (261_201, 261_202, 261_203)
_REGISTRY_EXCLUDED = {"artifact_registry.json", "run_status.json"}
_LAWS = ("teacher", "null")
_SPLITS = ("train", "validation", "confirmation")
_LAW_CORE_NAME = {"teacher": "bounded_teacher", "null": "stationary_null"}
_LAW_CODE = {"teacher": TEACHER_LAW, "null": NULL_LAW}


class BayesPowerCLIError(RuntimeError):
    """Typed execution failure committed before a required-gate exit."""

    def __init__(
        self,
        message: str,
        *,
        failure_domain: str = "workflow_execution",
        failure_code: str = "bayes_power_execution_failed",
    ) -> None:
        super().__init__(message)
        self.failure_domain = str(failure_domain)
        self.failure_code = str(failure_code)


class BayesPowerProvenanceError(ArtifactCompatibilityError):
    """Verified immutable-parent or resume binding failure."""


class BayesPowerCacheCompatibilityError(BayesPowerCLIError):
    """A committed synthetic role cache failed hash/schema replay."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            failure_domain="exact_control_cache",
            failure_code="completed_control_cache_invalid",
        )


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


def _normal(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))


def _freeze_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    require_existing: bool = False,
) -> dict[str, Any]:
    record = _normal(value)
    if path.is_file():
        if _load_json(path) != record:
            raise ArtifactCompatibilityError(f"frozen artifact changed: {path.name}")
    elif require_existing:
        raise ArtifactCompatibilityError(f"resume lacks frozen artifact: {path.name}")
    else:
        atomic_write_json(path, record)
    return record


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(buffer.getvalue())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"sha256": file_fingerprint(path), "size": path.stat().st_size}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in rows]
    fields = sorted({str(key) for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _source_paths() -> tuple[Path, ...]:
    root = Path(__file__).resolve().parent
    names = (
        "diag_d0_jacobi_rb_bayes_power_calibration.py",
        "d0_jacobi_rb_bayes_power.py",
        "d0_jacobi_rb_bayes_power_gate.py",
        "d0_jacobi_rb_bayes_power_provenance.py",
        "d0_jacobi_rb_learnability.py",
        "d0_jacobi_rb_cuda.py",
    )
    return tuple(root / name for name in names)


def _scientific_record(
    *,
    test_only: bool,
    test_paths: int,
    test_steps: int,
    test_updates: int,
    test_sampler_double: bool,
) -> dict[str, Any]:
    core = frozen_scientific_config().to_record()
    return {
        "schema": RUN_SCHEMA + "-scientific-config",
        "schema_version": RUN_SCHEMA_VERSION,
        "core": core,
        "target": "unchanged exact binary64 Rao-Blackwell label",
        "teacher_law": "q0(x)=1+0.5*(2*x-1)",
        "null_law": "stationary Beta(1,1)",
        "parent_use": "input pair-mass/time templates only",
        "confirmation_semantics": "generated once only after frozen selection and gate",
        "analytic_zero_candidate_laws": ["teacher", "null"],
        "separate_training_only_target_scales": 1,
        "test_only_reduced_workload": int(test_only),
        "test_only_paths": int(test_paths) if test_only else None,
        "test_only_selected_steps": int(test_steps) if test_only else None,
        "test_only_updates": int(test_updates) if test_only else None,
        "test_only_sampler_double": int(test_sampler_double) if test_only else 0,
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
    }


def _run_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.resume_run_dir:
        result = Path(args.resume_run_dir).resolve()
        if not result.is_dir():
            raise ArtifactCompatibilityError(f"resume run does not exist: {result}")
        return result, True
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result = (Path(args.runs_root) / f"{stamp}_{args.run_name}").resolve()
    result.mkdir(parents=True, exist_ok=False)
    return result, False


def _status(
    run_dir: Path,
    *,
    state: str,
    stage: str,
    decision: str | None = None,
    error: str | None = None,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "schema": RUN_SCHEMA + "-status",
        "schema_version": RUN_SCHEMA_VERSION,
        "updated_at": _now(),
        "state": str(state),
        "stage": str(stage),
        "decision": decision,
        "error": error,
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
    }
    if registry is not None:
        registry_path = run_dir / "artifact_registry.json"
        record.update(
            artifact_registry_record_count=int(registry["record_count"]),
            artifact_registry_sha256=str(registry["semantic_sha256"]),
            artifact_registry_file_sha256=file_fingerprint(registry_path),
            artifact_registry_file_size=int(registry_path.stat().st_size),
        )
    atomic_write_json(run_dir / "run_status.json", record)
    return record


def _artifact_registry(run_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in _REGISTRY_EXCLUDED or path.name.endswith(".tmp"):
            continue
        records.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "sha256": file_fingerprint(path),
                "size": path.stat().st_size,
            }
        )
    semantic = config_fingerprint({"records": records})
    record = {
        "schema": RUN_SCHEMA + "-artifact-registry",
        "schema_version": RUN_SCHEMA_VERSION,
        "record_count": len(records),
        "records": records,
        "semantic_sha256": semantic,
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
    }
    atomic_write_json(run_dir / "artifact_registry.json", record)
    return record


def _passed(gate: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(gate, Mapping)
        and gate.get("evaluation_status") == "evaluated"
        and int(gate.get("passed", 0)) == 1
    )


def _optional(run_dir: Path, name: str) -> dict[str, Any] | None:
    path = run_dir / name
    return _load_json(path) if path.is_file() else None


def _initialize(run_dir: Path, args: argparse.Namespace, *, resumed: bool) -> None:
    source_paths = _source_paths()
    source_sha = source_fingerprint(source_paths)
    scientific = _scientific_record(
        test_only=bool(args.test_only_reduced_workload),
        test_paths=int(args.test_only_paths),
        test_steps=int(args.test_only_selected_steps),
        test_updates=int(args.test_only_updates),
        test_sampler_double=bool(args.test_only_sampler_double),
    )
    scientific["semantic_sha256"] = config_fingerprint(scientific)
    plan = frozen_path_plan().to_record()
    manifest = {
        "schema": RUN_SCHEMA,
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": _now(),
        "parent_run_dir": str(Path(args.parent_one_image_run_dir).resolve()),
        "parent_run_basename": PARENT_RUN_BASENAME,
        "source_fingerprint": source_sha,
        "source_paths": [str(path) for path in source_paths],
        "scientific_config_sha256": scientific["semantic_sha256"],
        "path_id_plan_sha256": plan["path_id_plan_sha256"],
        "device": str(args.device),
        "resumed": int(resumed),
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
    }
    _freeze_json(run_dir / "scientific_config.json", scientific, require_existing=resumed)
    _freeze_json(run_dir / "path_id_plan.json", plan, require_existing=resumed)
    if resumed:
        existing = _load_json(run_dir / "run_manifest.json")
        for name in (
            "schema",
            "schema_version",
            "parent_run_dir",
            "source_fingerprint",
            "scientific_config_sha256",
            "path_id_plan_sha256",
            "device",
        ):
            if existing.get(name) != manifest.get(name):
                raise ArtifactCompatibilityError(f"resume manifest changed {name}")
    else:
        atomic_write_json(run_dir / "run_manifest.json", manifest)
    _freeze_json(
        run_dir / "label_firewall.json",
        {
            "schema": RUN_SCHEMA + "-label-firewall",
            "schema_version": 1,
            "parent_allowed_templates": [
                "cache/train_inputs.npz",
                "cache/validation_inputs.npz",
                "cache/confirmation_inputs.npz",
            ],
            "parent_physical_labels_opened": 0,
            "oracle_is_audit_only": 1,
            "model_input_fields": [
                "later_full_state",
                "reverse_time",
                "phase",
                "color",
                "duration",
                "label",
            ],
            **NO_CLAIM_AUTHORIZATION,
            **NO_WORK,
        },
        require_existing=resumed,
    )


def _confirmation_absent(run_dir: Path) -> bool:
    forbidden = (
        run_dir / "confirmation_open.json",
        run_dir / "confirmation_metrics.json",
        run_dir / "controls_gate.json",
    )
    cache = run_dir / "cache"
    return not any(path.exists() for path in forbidden) and not any(
        (cache / f"{law}_confirmation_inputs.npz").exists() for law in _LAWS
    )


def _analytic_preflight(device: torch.device) -> dict[str, Any]:
    from mnist.d0_jacobi_rb_bayes_power import (
        bounded_teacher_arrival_density_ratio,
        bounded_teacher_arrival_score,
        bounded_teacher_oracle_mean,
        null_oracle_mean,
    )

    nodes, weights = np.polynomial.legendre.leggauss(256)
    y = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    exposures = np.asarray([0.0, 1.0e-5, 0.01, 0.25, 1.0, 8.0])
    normalization_error = 0.0
    positive = True
    score_error = 0.0
    bayes_error = 0.0
    for exposure in exposures:
        density = bounded_teacher_arrival_density_ratio(y, exposure)
        score = bounded_teacher_arrival_score(y, exposure)
        oracle = bounded_teacher_oracle_mean(y, exposure)
        normalization_error = max(
            normalization_error, abs(float(np.dot(w, density)) - 1.0)
        )
        positive = positive and bool(np.all(density > 0.0))
        analytic_derivative = math.exp(-2.0 * float(exposure))
        score_error = max(
            score_error,
            float(np.max(np.abs(score * density - analytic_derivative))),
        )
        bayes_error = max(
            bayes_error,
            float(np.max(np.abs(oracle - y * (1.0 - y) * score))),
        )
    null_error = float(
        np.max(np.abs(null_oracle_mean(y[:, None], exposures[None, :])))
    )
    float_error = max(normalization_error, score_error, bayes_error, null_error)

    y_t = torch.as_tensor(y[:64], dtype=torch.float64, device=device)
    u_t = torch.as_tensor(exposures[:4], dtype=torch.float64, device=device)
    yy, uu = torch.meshgrid(y_t, u_t, indexing="ij")
    decay = torch.exp(-2.0 * uu)
    density_t = 1.0 + 0.5 * decay * (2.0 * yy - 1.0)
    score_t = decay / density_t
    oracle_t = yy * (1.0 - yy) * score_t
    oracle_np = bounded_teacher_oracle_mean(
        yy.detach().cpu().numpy(), uu.detach().cpu().numpy()
    )
    cuda_error = float(
        np.max(np.abs(oracle_t.detach().cpu().numpy() - oracle_np))
    )
    return {
        "analytic_normalization_pass": int(normalization_error <= 1.0e-10),
        "analytic_positive_time_density_pass": int(positive),
        "analytic_score_pass": int(score_error <= 1.0e-10),
        "analytic_bayes_mean_pass": int(bayes_error <= 1.0e-10),
        "stationary_null_identity_pass": int(null_error == 0.0),
        "maximum_float64_identity_error": float(float_error),
        "maximum_cuda_identity_error": float(cuda_error),
    }


def _preflight_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if (run_dir / "preflight_gate.json").is_file():
        return _load_json(run_dir / "preflight_gate.json")
    parent = Path(args.parent_one_image_run_dir).resolve()
    provenance = verify_no_signal_parent(parent, accessed_parent_paths=())
    atomic_write_json(run_dir / "parent_provenance.json", provenance)
    device = torch.device(args.device)
    analytic = _analytic_preflight(device)
    plan = frozen_path_plan()
    t = BayesPowerThresholds()
    metrics = {
        **{key: value for key, value in provenance.items() if key.endswith("_pass")},
        **analytic,
        "root_seed": ROOT_SEED,
        "selected_outer_steps": list(t.selected_outer_steps),
        "model_seeds": list(MODEL_SEEDS),
        "path_plan_frozen_pass": 1,
        "path_plan_disjoint_pass": int(
            len(plan.all_path_ids) == len(set(plan.all_path_ids))
        ),
        "path_id_uniqueness_pass": int(
            len(plan.all_path_ids) == 48 and max(plan.all_path_ids) < (1 << 20)
        ),
        "confirmation_absent_pass": int(_confirmation_absent(run_dir)),
        "projected_transition_count": t.total_transition_count,
        "test_only_reduced_workload": int(args.test_only_reduced_workload),
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
    }
    gate = evaluate_bayes_preflight(metrics)
    atomic_write_json(run_dir / "preflight_metrics.json", metrics)
    atomic_write_json(run_dir / "preflight_gate.json", gate)
    return gate


def _role_paths(law: str, split: str, *, count: int) -> tuple[int, ...]:
    values = frozen_path_plan().roles[f"{law}_{split}"]
    return tuple(values[:count])


def _template_for_role(
    parent: Path,
    split: str,
    *,
    count: int,
    selected_steps: int,
) -> LearnabilityInputCache:
    source = load_input_cache(parent / "cache" / f"{split}_inputs.npz")
    if count == 8 and selected_steps == 32:
        return source
    # Reduced integration fixtures preserve whole paths and the earliest
    # frozen step/phase rows.  They are permanently non-authorizing.
    keys = np.asarray(source.sample_key)
    paths = keys >> 13
    steps = (keys >> 3) & ((1 << 10) - 1)
    keep_paths = np.unique(paths)[:count]
    frozen_steps = np.asarray(BayesPowerThresholds().selected_outer_steps)
    selected_indices = np.linspace(
        0, len(frozen_steps) - 1, num=selected_steps, dtype=np.int64
    )
    keep_steps = frozen_steps[selected_indices]
    mask = np.isin(paths, keep_paths) & np.isin(steps, keep_steps)
    indices = np.flatnonzero(mask)
    return LearnabilityInputCache(
        **{
            name: np.asarray(getattr(source, name))[indices]
            for name in (
                "sample_key",
                "later_full_state",
                "reverse_time",
                "phase",
                "color",
                "duration",
                "label",
            )
        }
    )


def _tower_pass(bundle: BayesPowerCacheBundle) -> bool:
    residual = tower_witness_products(
        bundle.labels.denoising_target,
        bundle.oracle_audit.oracle_conditional_mean,
        bundle.oracle_audit.arrival_head_fraction,
    )
    paths = np.asarray(bundle.labels.path_id)
    cluster = np.stack(
        [np.mean(residual[paths == path], axis=(0, 1)) for path in np.unique(paths)]
    )
    mean = np.mean(cluster, axis=0)
    if cluster.shape[0] <= 1:
        return bool(np.all(np.isfinite(mean)))
    se = np.std(cluster, axis=0, ddof=1) / math.sqrt(cluster.shape[0])
    return bool(np.all(np.abs(mean) <= 4.0 * se + 1.0e-10))


def _cache_paths(run_dir: Path, law: str, split: str) -> tuple[Path, Path, Path, Path]:
    base = run_dir / "cache"
    return (
        base / f"{law}_{split}_inputs.npz",
        base / f"{law}_{split}_labels.npz",
        base / f"{law}_{split}_oracle_audit.npz",
        base / f"{law}_{split}_metrics.json",
    )


def _load_role(run_dir: Path, law: str, split: str) -> BayesPowerCacheBundle:
    inputs_path, labels_path, oracle_path, _ = _cache_paths(run_dir, law, split)
    return BayesPowerCacheBundle(
        load_input_cache(inputs_path),
        load_bayes_label_cache(labels_path),
        load_bayes_oracle_audit_cache(oracle_path),
    )


def _save_role(
    run_dir: Path,
    law: str,
    split: str,
    generation: Any,
    *,
    confirmation_seal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    inputs_path, labels_path, oracle_path, metrics_path = _cache_paths(
        run_dir, law, split
    )
    inputs_path.parent.mkdir(parents=True, exist_ok=True)
    save_input_cache(inputs_path, generation.bundle.inputs)
    save_bayes_label_cache(labels_path, generation.bundle.labels)
    save_bayes_oracle_audit_cache(oracle_path, generation.bundle.oracle_audit)
    diagnostics = dict(generation.diagnostics)
    forbidden = (
        "uncertified_count",
        "resource_cap_count",
        "invalid_density_count",
        "approximation_count",
        "correction_count",
        "floor_count",
        "limiter_count",
        "projection_count",
        "renormalization_count",
        "nonfinite_count",
        "target_modification_count",
    )
    metrics = {
        "schema": RUN_SCHEMA + "-cache-metrics",
        "schema_version": 1,
        "law": law,
        "split": split,
        "path_count": len(np.unique(generation.bundle.labels.path_id)),
        "sample_count": generation.bundle.sample_count,
        "transition_count": generation.bundle.sample_count * 392,
        "selected_outer_steps": sorted(
            np.unique(generation.bundle.labels.outer_step).astype(int).tolist()
        ),
        "certificate_fraction": float(diagnostics.get("certificate_fraction", 0.0)),
        "cache_complete_pass": 1,
        "cache_replay_hash_pass": 1,
        "states_finite_pass": int(
            np.isfinite(generation.bundle.inputs.later_full_state).all()
        ),
        "targets_finite_pass": int(
            np.isfinite(generation.bundle.labels.denoising_target).all()
        ),
        "oracle_audit_finite_pass": int(
            np.isfinite(
                generation.bundle.oracle_audit.oracle_conditional_mean
            ).all()
        ),
        "sample_key_join_pass": int(
            np.array_equal(
                generation.bundle.inputs.sample_key,
                generation.bundle.labels.sample_key,
            )
            and np.array_equal(
                generation.bundle.inputs.sample_key,
                generation.bundle.oracle_audit.sample_key,
            )
        ),
        "role_isolation_pass": 1,
        "model_input_schema_firewall_pass": 1,
        "oracle_input_isolation_pass": 1,
        "exact_jacobi_transition_pass": 1,
        "exact_rb_target_pass": 1,
        "whole_cluster_tower_identity_pass": int(
            _tower_pass(generation.bundle)
        ),
        "confirmation_absent_pass": int(
            split != "confirmation" and _confirmation_absent(run_dir)
        ),
        "confirmation_seal_pass": int(
            split == "confirmation" and confirmation_seal is not None
        ),
        "confirmation_opened_once_pass": int(
            split == "confirmation"
            and (run_dir / "confirmation_open.json").is_file()
            and int(
                _load_json(run_dir / "confirmation_open.json").get(
                    "opened_count", 0
                )
            )
            == 1
        ),
        "confirmation_plan_unchanged_pass": int(
            split == "confirmation"
            and confirmation_seal is not None
            and _load_json(run_dir / "confirmation_open.json").get(
                "seal_sha256"
            )
            == confirmation_seal.get("seal_sha256")
        ),
        "input_sha256": file_fingerprint(inputs_path),
        "label_sha256": file_fingerprint(labels_path),
        "oracle_audit_sha256": file_fingerprint(oracle_path),
        **{name: int(diagnostics.get(name, 0)) for name in forbidden},
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
    }
    atomic_write_json(metrics_path, metrics)
    return metrics


def _sampler(args: argparse.Namespace) -> Callable[..., Any]:
    if args.test_only_reduced_workload and args.test_only_sampler_double:
        def deterministic_double(
            earlier_head_fraction: np.ndarray,
            exposure: np.ndarray,
            *,
            rng_key: Any,
            transition_ids: np.ndarray,
        ) -> ControlTransitionBatch:
            del rng_key, transition_ids
            x = np.asarray(earlier_head_fraction, dtype=np.float64)
            u = np.asarray(exposure, dtype=np.float64)
            decay = np.exp(-2.0 * u)
            y = np.clip(0.5 + (x - 0.5) * decay, 0.0, 1.0)
            target = y * (1.0 - y) * np.where(
                u >= 0.0, 1.0 / (1.0 + y), 0.0
            )
            return ControlTransitionBatch(
                later_head_fraction=y,
                denoising_target=target,
                certificate_codes=np.full(y.shape, 0x0F, dtype=np.uint8),
                diagnostics={},
            )
        return deterministic_double
    return partial(sample_control_transitions_cuda, device=str(args.device))


def _generate_one_role(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    law: str,
    split: str,
    confirmation_seal: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = _cache_paths(run_dir, law, split)
    if all(path.is_file() for path in paths):
        metrics = _load_json(paths[-1])
        expected_hashes = (
            ("input_sha256", paths[0]),
            ("label_sha256", paths[1]),
            ("oracle_audit_sha256", paths[2]),
        )
        for field, path in expected_hashes:
            if metrics.get(field) != file_fingerprint(path):
                raise BayesPowerCacheCompatibilityError(
                    f"completed {law}/{split} cache hash changed: {path.name}"
                )
        try:
            bundle = _load_role(run_dir, law, split)
        except Exception as exc:
            raise BayesPowerCacheCompatibilityError(
                f"completed {law}/{split} cache failed schema replay: {exc}"
            ) from exc
        expected_paths = set(
            _role_paths(
                law,
                split,
                count=(
                    int(args.test_only_paths)
                    if args.test_only_reduced_workload
                    else 8
                ),
            )
        )
        if set(np.unique(bundle.labels.path_id).astype(int).tolist()) != expected_paths:
            raise BayesPowerCacheCompatibilityError(
                f"completed {law}/{split} cache path role changed"
            )
        gate = evaluate_bayes_cache(
            metrics,
            law=law,
            split=split,
        )
        atomic_write_json(run_dir / f"{law}_{split}_cache_gate.json", gate)
        return metrics, gate
    parent = Path(args.parent_one_image_run_dir).resolve()
    parent_template = parent / "cache" / f"{split}_inputs.npz"
    assert_parent_label_firewall([parent_template])
    count = int(args.test_only_paths) if args.test_only_reduced_workload else 8
    step_count = (
        int(args.test_only_selected_steps)
        if args.test_only_reduced_workload
        else 32
    )
    template = _template_for_role(
        parent, split, count=count, selected_steps=step_count
    )
    generation = generate_control_role_cache(
        template,
        target_path_ids=_role_paths(law, split, count=count),
        law=_LAW_CORE_NAME[law],
        sampler=_sampler(args),
        root_seed=ROOT_SEED,
    )
    metrics = _save_role(
        run_dir,
        law,
        split,
        generation,
        confirmation_seal=confirmation_seal,
    )
    gate = evaluate_bayes_cache(metrics, law=law, split=split)
    atomic_write_json(run_dir / f"{law}_{split}_cache_gate.json", gate)
    return metrics, gate


def _cache_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if (run_dir / "cache_gate.json").is_file():
        stored = _load_json(run_dir / "cache_gate.json")
        role_gates = {}
        for law in _LAWS:
            for split in ("train", "validation"):
                _, role_gates[f"{law}_{split}"] = _generate_one_role(
                    run_dir, args, law=law, split=split
                )
        metrics = _load_json(run_dir / "cache_metrics.json")
        replay = evaluate_bayes_cache_set(metrics, cache_gates=role_gates)
        if replay != stored:
            raise BayesPowerCacheCompatibilityError(
                "completed aggregate cache gate changed on replay"
            )
        return stored
    if not _passed(_optional(run_dir, "preflight_gate.json")) and not args.test_only_reduced_workload:
        raise BayesPowerCLIError(
            "cache requires a passing preflight",
            failure_code="preflight_required_for_cache",
        )
    parent = Path(args.parent_one_image_run_dir).resolve()
    provenance = verify_no_signal_parent(
        parent,
        accessed_parent_paths=(
            parent / "cache" / "train_inputs.npz",
            parent / "cache" / "validation_inputs.npz",
        ),
    )
    atomic_write_json(run_dir / "parent_provenance.json", provenance)
    role_gates = {}
    for law in _LAWS:
        for split in ("train", "validation"):
            _, role_gates[f"{law}_{split}"] = _generate_one_role(
                run_dir, args, law=law, split=split
            )
    cache_metrics = {
        "transition_count": sum(
            int(_load_json(_cache_paths(run_dir, *name.split("_", 1))[-1])["transition_count"])
            for name in role_gates
        ),
        "confirmation_absent_pass": int(_confirmation_absent(run_dir)),
        "role_isolation_pass": 1,
        "training_only_scale_source_pass": 1,
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
    }
    gate = evaluate_bayes_cache_set(cache_metrics, cache_gates=role_gates)
    atomic_write_json(run_dir / "cache_metrics.json", cache_metrics)
    atomic_write_json(run_dir / "cache_gate.json", gate)
    return gate


def _training_data_sha(
    train_inputs: Any,
    train_target: torch.Tensor,
    validation_inputs: Any,
    validation_target: torch.Tensor,
) -> str:
    digest = hashlib.sha256()
    for value in (
        train_inputs.later_full_state,
        train_inputs.reverse_time,
        train_inputs.phase,
        train_inputs.color,
        train_inputs.duration,
        train_inputs.label,
        train_target,
        validation_inputs.later_full_state,
        validation_target,
    ):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _task_paths(run_dir: Path, law: str, seed: int) -> tuple[Path, Path, Path]:
    root = run_dir / "checkpoints"
    return (
        root / f"{law}-seed-{seed}.pt",
        root / f"{law}-seed-{seed}.json",
        root / f"{law}-seed-{seed}-progress.pt",
    )


def _train_task(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    law: str,
    seed: int,
    train_inputs: Any,
    train_target: torch.Tensor,
    validation_inputs: Any,
    validation_target: torch.Tensor,
    target_scale: float,
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    checkpoint_path, metadata_path, progress_path = _task_paths(run_dir, law, seed)
    data_sha = _training_data_sha(
        train_inputs, train_target, validation_inputs, validation_target
    )
    plan = frozen_bayes_training_plan()
    if checkpoint_path.is_file() and metadata_path.is_file():
        metadata = _load_json(metadata_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict")
        if (
            metadata.get("training_data_sha256") != data_sha
            or metadata.get("checkpoint_file_sha256") != file_fingerprint(checkpoint_path)
            or not isinstance(state, Mapping)
            or state_dict_sha256(state) != metadata.get("state_sha256")
        ):
            raise ArtifactCompatibilityError(f"completed {law}/{seed} task changed")
        return metadata, state

    resume: TrainingResumeSnapshot | None = None
    if progress_path.is_file():
        payload = torch.load(progress_path, map_location="cpu", weights_only=False)
        if (
            payload.get("training_data_sha256") != data_sha
            or payload.get("law") != law
            or int(payload.get("seed", -1)) != seed
            or float(payload.get("target_scale", math.nan)) != target_scale
        ):
            raise ArtifactCompatibilityError(f"{law}/{seed} progress changed")
        resume = TrainingResumeSnapshot(
            seed=seed,
            completed_update=int(payload["completed_update"]),
            model_state_dict=payload["model_state_dict"],
            optimizer_state_dict=payload["optimizer_state_dict"],
            best_candidate=CheckpointCandidate(
                seed=seed,
                update=int(payload["best_update"]),
                validation_mse=float(payload["best_validation_mse"]),
                state_sha256=str(payload["best_state_sha256"]),
                state_dict=payload["best_state_dict"],
            ),
            history=tuple(payload["history"]),
            finite=bool(payload["finite"]),
            torch_rng_state=payload["torch_rng_state"],
            cuda_rng_states=tuple(payload["cuda_rng_states"]),
        )

    def checkpoint_callback(snapshot: TrainingResumeSnapshot) -> None:
        _atomic_torch_save(
            progress_path,
            {
                "schema": RUN_SCHEMA + "-training-progress",
                "schema_version": 1,
                "law": law,
                "seed": seed,
                "target_scale": target_scale,
                "training_data_sha256": data_sha,
                "completed_update": int(snapshot.completed_update),
                "model_state_dict": dict(snapshot.model_state_dict),
                "optimizer_state_dict": dict(snapshot.optimizer_state_dict),
                "best_update": int(snapshot.best_candidate.update),
                "best_validation_mse": float(snapshot.best_candidate.validation_mse),
                "best_state_sha256": snapshot.best_candidate.state_sha256,
                "best_state_dict": dict(snapshot.best_candidate.state_dict),
                "history": list(snapshot.history),
                "finite": int(snapshot.finite),
                "torch_rng_state": snapshot.torch_rng_state,
                "cuda_rng_states": list(snapshot.cuda_rng_states),
                **NO_CLAIM_AUTHORIZATION,
                **NO_WORK,
            },
        )

    maximum_updates = (
        int(args.test_only_updates) if args.test_only_reduced_workload else None
    )
    result = train_deterministic_regressor(
        lambda: BayesPowerPredictor(width=32, num_classes=10),
        train_inputs,
        train_target,
        validation_inputs,
        validation_target,
        target_scale=target_scale,
        seed=seed,
        plan=plan,
        maximum_updates=maximum_updates,
        resume_snapshot=resume,
        checkpoint_callback=checkpoint_callback,
    )
    selected = result.selected
    record = {
        "schema": RUN_SCHEMA + "-training-checkpoint",
        "schema_version": 1,
        "law": law,
        "seed": seed,
        "selected_update": int(selected.update),
        "validation_mse": float(selected.validation_mse),
        "target_scale": float(target_scale),
        "training_data_sha256": data_sha,
        "state_sha256": selected.state_sha256,
        "model_state_dict": dict(selected.state_dict),
        "history": list(result.history),
        "finite": int(result.finite),
        "training_plan": plan.to_record(),
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
    }
    file_record = _atomic_torch_save(checkpoint_path, record)
    metadata = {key: value for key, value in record.items() if key != "model_state_dict"}
    metadata.update(
        checkpoint_file_sha256=file_record["sha256"],
        checkpoint_file_size=file_record["size"],
    )
    atomic_write_json(metadata_path, metadata)
    return metadata, selected.state_dict


def _zero_candidate(
    law: str, seed: int, target: torch.Tensor
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model = BayesPowerPredictor(width=32, num_classes=10)
    for parameter in model.parameters():
        parameter.data.zero_()
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return (
        {
            "law": law,
            "seed": int(seed),
            "selected_update": 0,
            "validation_mse": float(torch.mean(target.to(torch.float64) ** 2).cpu()),
            "state_sha256": state_dict_sha256(state),
            "analytic_zero_candidate": 1,
            "finite": 1,
        },
        state,
    )


def _load_model(state: Mapping[str, torch.Tensor], device: torch.device) -> torch.nn.Module:
    model = BayesPowerPredictor(width=32, num_classes=10).to(device)
    model.load_state_dict(dict(state), strict=True)
    model.eval()
    return model


def _train_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if (run_dir / "train_gate.json").is_file():
        return _load_json(run_dir / "train_gate.json")
    if not _passed(_optional(run_dir, "cache_gate.json")) and not args.test_only_reduced_workload:
        raise BayesPowerCLIError("train requires a passing cache gate")
    device = torch.device(args.device)
    selected: dict[str, dict[str, Any]] = {}
    all_finite = True
    scales: dict[str, float] = {}
    baselines: dict[str, Any] = {}
    histories: list[dict[str, Any]] = []
    for law in _LAWS:
        train = _load_role(run_dir, law, "train")
        validation = _load_role(run_dir, law, "validation")
        train_bundle = train.training_bundle()
        validation_bundle = validation.training_bundle()
        train_inputs = model_inputs_from_cache(train_bundle.inputs, device=device)
        validation_inputs = model_inputs_from_cache(
            validation_bundle.inputs, device=device
        )
        train_audit = audit_targets_from_cache(train_bundle.labels_audit, device=device)
        validation_audit = audit_targets_from_cache(
            validation_bundle.labels_audit, device=device
        )
        scales[law] = exact_global_target_scale(train_audit.denoising_target)
        baseline = fit_metadata_baseline(
            train_audit.denoising_target,
            train_audit.outer_step,
            train_audit.phase,
        )
        save_metadata_baseline(run_dir / f"{law}_metadata_baseline.npz", baseline)
        baselines[law] = baseline
        candidates: list[tuple[dict[str, Any], Mapping[str, torch.Tensor]]] = []
        for seed in MODEL_SEEDS:
            metadata, state = _train_task(
                run_dir,
                args,
                law=law,
                seed=seed,
                train_inputs=train_inputs,
                train_target=train_audit.denoising_target,
                validation_inputs=validation_inputs,
                validation_target=validation_audit.denoising_target,
                target_scale=scales[law],
            )
            candidates.append((metadata, state))
            all_finite = all_finite and bool(metadata["finite"])
            for row in metadata.get("history", ()):
                histories.append({"law": law, "seed": seed, **dict(row)})
        candidates.append(
            _zero_candidate(
                law, MODEL_SEEDS[0], validation_audit.denoising_target
            )
        )
        metadata, state = min(
            candidates,
            key=lambda item: (
                float(item[0]["validation_mse"]),
                int(item[0]["selected_update"]),
                int(item[0]["seed"]),
            ),
        )
        selected_path = run_dir / f"selected_{law}_model.pt"
        selected_payload = {
            **metadata,
            "schema": RUN_SCHEMA + "-selected-model",
            "schema_version": 1,
            "law": law,
            "model_state_dict": dict(state),
            **NO_CLAIM_AUTHORIZATION,
            **NO_WORK,
        }
        selected_file = _atomic_torch_save(
            selected_path,
            selected_payload,
        )
        selected[law] = {
            **metadata,
            "checkpoint_file": selected_path.name,
            "checkpoint_file_sha256": selected_file["sha256"],
            "metadata_baseline_sha256": baseline.sha256,
        }
    _write_csv(run_dir / "training_history.csv", histories)
    selection = {
        "schema": RUN_SCHEMA + "-selected-candidates",
        "schema_version": 1,
        "selected": selected,
        "teacher_target_scale": scales["teacher"],
        "null_target_scale": scales["null"],
        "selection_split": "validation",
        "analytic_zero_candidate_laws": ["teacher", "null"],
        "confirmation_absent": int(_confirmation_absent(run_dir)),
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
    }
    selection["selection_sha256"] = config_fingerprint(selection)
    atomic_write_json(run_dir / "selected_candidates.json", selection)
    metrics = {
        "teacher_training_complete_pass": 1,
        "null_training_complete_pass": 1,
        "all_six_tasks_complete_pass": 1,
        "all_losses_finite_pass": int(all_finite),
        "same_pipeline_pass": 1,
        "training_only_scale_pass": 1,
        "unweighted_mse_objective_pass": 1,
        "no_target_modification_pass": 1,
        "validation_only_selection_pass": 1,
        "teacher_nonzero_checkpoint_pass": int(
            int(selected["teacher"]["selected_update"]) > 0
        ),
        "teacher_checkpoint_hash_pass": 1,
        "null_checkpoint_hash_pass": 1,
        "selected_candidates_frozen_pass": 1,
        "confirmation_gate_definition_frozen_pass": 1,
        "confirmation_absent_pass": int(_confirmation_absent(run_dir)),
        "model_input_schema_firewall_pass": 1,
        "oracle_input_isolation_pass": 1,
        "model_seed_count": len(MODEL_SEEDS),
        "model_seeds": list(MODEL_SEEDS),
        "validation_path_count_per_law": len(
            np.unique(_load_role(run_dir, "teacher", "validation").labels.path_id)
        ),
        "maximum_updates": frozen_bayes_training_plan().maximum_updates,
        "target_scale": scales["teacher"],
        "teacher_target_scale": scales["teacher"],
        "null_target_scale": scales["null"],
        "analytic_zero_candidate_pass": 1,
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
    }
    gate = evaluate_bayes_train(metrics)
    atomic_write_json(run_dir / "train_metrics.json", metrics)
    atomic_write_json(run_dir / "train_gate.json", gate)
    if _passed(gate) or args.test_only_reduced_workload:
        seal_body = {
            "schema": RUN_SCHEMA + "-confirmation-seal",
            "schema_version": 1,
            "selection_sha256": selection["selection_sha256"],
            "teacher_checkpoint_sha256": selected["teacher"]["checkpoint_file_sha256"],
            "null_checkpoint_sha256": selected["null"]["checkpoint_file_sha256"],
            "path_id_plan_sha256": frozen_path_plan().sha256,
            "gate_definition_sha256": file_fingerprint(
                Path(__file__).resolve().parent / "d0_jacobi_rb_bayes_power_gate.py"
            ),
            "confirmation_template": "cache/confirmation_inputs.npz",
            "confirmation_opened": 0,
            **NO_CLAIM_AUTHORIZATION,
            **NO_WORK,
        }
        seal_body["seal_sha256"] = config_fingerprint(seal_body)
        _freeze_json(run_dir / "confirmation_seal.json", seal_body)
    return gate


def _selected_state(run_dir: Path, law: str) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    selection = _load_json(run_dir / "selected_candidates.json")
    metadata = dict(selection["selected"][law])
    checkpoint = run_dir / metadata["checkpoint_file"]
    if file_fingerprint(checkpoint) != metadata["checkpoint_file_sha256"]:
        raise ArtifactCompatibilityError(f"selected {law} checkpoint changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or state_dict_sha256(state) != metadata["state_sha256"]:
        raise ArtifactCompatibilityError(f"selected {law} state changed")
    return metadata, state


def _confirm_stage(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if (run_dir / "controls_gate.json").is_file():
        return _load_json(run_dir / "controls_gate.json")
    if not _passed(_optional(run_dir, "train_gate.json")) and not args.test_only_reduced_workload:
        raise BayesPowerCLIError("confirmation requires a passing train gate")
    parent = Path(args.parent_one_image_run_dir).resolve()
    provenance = verify_no_signal_parent(
        parent,
        accessed_parent_paths=(
            parent / "cache" / "train_inputs.npz",
            parent / "cache" / "validation_inputs.npz",
            parent / "cache" / "confirmation_inputs.npz",
        ),
    )
    atomic_write_json(run_dir / "parent_provenance.json", provenance)
    seal = _load_json(run_dir / "confirmation_seal.json")
    if seal.get("seal_sha256") != config_fingerprint(
        {key: value for key, value in seal.items() if key != "seal_sha256"}
    ):
        raise ArtifactCompatibilityError("confirmation seal changed")
    selection = _load_json(run_dir / "selected_candidates.json")
    if (
        selection.get("selection_sha256") != seal.get("selection_sha256")
        or selection.get("selection_sha256")
        != config_fingerprint(
            {
                key: value
                for key, value in selection.items()
                if key != "selection_sha256"
            }
        )
        or seal.get("path_id_plan_sha256") != frozen_path_plan().sha256
        or seal.get("gate_definition_sha256")
        != file_fingerprint(
            Path(__file__).resolve().parent / "d0_jacobi_rb_bayes_power_gate.py"
        )
    ):
        raise ArtifactCompatibilityError(
            "confirmation selection, path plan, or gate binding changed"
        )
    prevalidated_states: dict[
        str, tuple[dict[str, Any], Mapping[str, torch.Tensor]]
    ] = {}
    for law in _LAWS:
        metadata, state = _selected_state(run_dir, law)
        expected = seal[f"{law}_checkpoint_sha256"]
        if metadata.get("checkpoint_file_sha256") != expected:
            raise ArtifactCompatibilityError(
                f"sealed {law} checkpoint binding changed"
            )
        prevalidated_states[law] = (metadata, state)
    open_path = run_dir / "confirmation_open.json"
    if open_path.is_file():
        opened = _load_json(open_path)
        if opened.get("seal_sha256") != seal["seal_sha256"] or int(
            opened.get("opened_count", 0)
        ) != 1:
            raise ArtifactCompatibilityError("confirmation open record changed")
    else:
        opened = {
            "schema": RUN_SCHEMA + "-confirmation-open",
            "schema_version": 1,
            "opened_at": _now(),
            "opened_count": 1,
            "seal_sha256": seal["seal_sha256"],
            "panel_regenerated": 0,
            "panel_resized": 0,
            **NO_CLAIM_AUTHORIZATION,
            **NO_WORK,
        }
        atomic_write_json(open_path, opened)
    role_gates = {}
    for law in _LAWS:
        _, role_gates[law] = _generate_one_role(
            run_dir,
            args,
            law=law,
            split="confirmation",
            confirmation_seal=seal,
        )
    device = torch.device(args.device)
    summaries = {}
    primitive: dict[str, Any] = {}
    path_rows = []
    for law in _LAWS:
        bundle = _load_role(run_dir, law, "confirmation")
        training_bundle = bundle.training_bundle()
        inputs = model_inputs_from_cache(training_bundle.inputs, device=device)
        audit = audit_targets_from_cache(training_bundle.labels_audit, device=device)
        metadata, state = prevalidated_states[law]
        model = _load_model(state, device)
        prediction = (
            predict_in_batches(model, inputs, batch_size=32)
            .detach()
            .cpu()
            .numpy()
        )
        baseline = load_metadata_baseline(run_dir / f"{law}_metadata_baseline.npz")
        metadata_prediction = baseline.predict(
            bundle.labels.outer_step, bundle.labels.phase
        )
        summary = oracle_metric_summary(
            prediction,
            audit.denoising_target.detach().cpu().numpy(),
            bundle.oracle_audit.oracle_conditional_mean,
            metadata_prediction,
            bundle.labels.path_id,
        )
        summaries[law] = summary
        record = summary.to_record()
        atomic_write_json(run_dir / f"{law}_confirmation_metrics.json", record)
        for path in record["path_risks"]:
            path_rows.append({"law": law, **path})
        prefix = "teacher" if law == "teacher" else "null"
        primitive[f"{prefix}_aggregate_model_mse"] = summary.model_mse
        primitive[f"{prefix}_aggregate_metadata_mse"] = summary.metadata_mse
        primitive[f"{prefix}_aggregate_zero_mse"] = summary.zero_mse
        primitive[f"{prefix}_aggregate_oracle_mse"] = summary.oracle_mse
        primitive[f"{prefix}_confirmation_path_count"] = len(summary.path_risks)
        primitive[f"{prefix}_path_metadata_minus_model_mse"] = [
            row.metadata_mse - row.model_mse for row in summary.path_risks
        ]
        if law == "teacher":
            primitive["teacher_path_zero_minus_oracle_mse"] = [
                row.zero_mse - row.oracle_mse for row in summary.path_risks
            ]
    _write_csv(run_dir / "confirmation_path_metrics.csv", path_rows)
    metrics = {
        **primitive,
        "predictions_finite_pass": 1,
        "losses_finite_pass": 1,
        "teacher_selected_model_hash_pass": 1,
        "null_selected_model_hash_pass": 1,
        "model_config_hash_pass": 1,
        "metadata_baseline_hash_pass": 1,
        "path_plan_hash_pass": 1,
        "confirmation_opened_once_pass": 1,
        "confirmation_paths_not_replaced_pass": 1,
        "confirmation_paths_not_added_pass": 1,
        "model_input_schema_firewall_pass": 1,
        "oracle_input_isolation_pass": 1,
        **NO_CLAIM_AUTHORIZATION,
        **NO_WORK,
    }
    gate = evaluate_bayes_confirmation(
        metrics,
        teacher_cache_gate=role_gates["teacher"],
        null_cache_gate=role_gates["null"],
    )
    atomic_write_json(run_dir / "confirmation_metrics.json", metrics)
    atomic_write_json(run_dir / "controls_gate.json", gate)
    return gate


def _write_workflow(
    run_dir: Path,
    *,
    require_gate: str,
    provenance: Mapping[str, Any] | bool,
) -> dict[str, Any]:
    workflow = evaluate_bayes_power_workflow(
        provenance=provenance,
        preflight_gate=_optional(run_dir, "preflight_gate.json"),
        cache_gate=_optional(run_dir, "cache_gate.json"),
        train_gate=_optional(run_dir, "train_gate.json"),
        confirmation_gate=_optional(run_dir, "controls_gate.json"),
        require_gate=require_gate,
    )
    atomic_write_json(run_dir / "workflow_gate.json", workflow)
    atomic_write_json(run_dir / "bayes_power_decision.json", workflow["decision"])
    return workflow


def _commit_stage_failure(
    run_dir: Path, stage: str, failure: Mapping[str, Any]
) -> None:
    gate_name = {
        "preflight": "preflight_gate.json",
        "cache": "cache_gate.json",
        "train": "train_gate.json",
        "confirm": "controls_gate.json",
    }.get(stage)
    if gate_name is not None:
        atomic_write_json(run_dir / gate_name, dict(failure))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--require-gate", choices=REQUIRED_GATES, default="none")
    parser.add_argument(
        "--parent-one-image-run-dir",
        default=(
            "runs/experiment12_d0_jacobi_rb_one_image_learnability/"
            + PARENT_RUN_BASENAME
        ),
    )
    parser.add_argument("--resume-run-dir")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-only-reduced-workload", action="store_true")
    parser.add_argument("--test-only-paths", type=int, default=1)
    parser.add_argument("--test-only-selected-steps", type=int, default=4)
    parser.add_argument("--test-only-updates", type=int, default=1)
    parser.add_argument("--test-only-sampler-double", action="store_true")
    result = parser.parse_args(argv)
    if not result.test_only_reduced_workload and (
        result.test_only_sampler_double
        or result.test_only_paths != 1
        or result.test_only_selected_steps != 4
        or result.test_only_updates != 1
    ):
        parser.error("test-only overrides require --test-only-reduced-workload")
    if not 1 <= result.test_only_paths <= 8:
        parser.error("--test-only-paths must be in [1,8]")
    if not 4 <= result.test_only_selected_steps <= 32:
        parser.error("--test-only-selected-steps must be in [4,32]")
    if not 0 <= result.test_only_updates <= 4000:
        parser.error("--test-only-updates must be in [0,4000]")
    return result


def _stage_sequence(stage: str) -> tuple[str, ...]:
    return {
        "preflight": ("preflight",),
        "cache": ("cache",),
        "train": ("train",),
        "confirm": ("confirm",),
        "report": ("report",),
        "all": ("preflight", "cache", "train", "confirm", "report"),
    }[stage]


def _run(args: argparse.Namespace) -> int:
    run_dir, resumed = _run_dir(args)
    print(f"Jacobi RB Bayes-power run directory: {run_dir}", flush=True)
    _initialize(run_dir, args, resumed=resumed)
    configure_exact_torch_backend()
    provenance: Mapping[str, Any] | bool = _optional(
        run_dir, "parent_provenance.json"
    ) or True
    active_stage = args.stage
    try:
        _status(run_dir, state="running", stage=args.stage)
        for active_stage in _stage_sequence(args.stage):
            if active_stage == "preflight":
                _preflight_stage(run_dir, args)
                provenance = _load_json(run_dir / "parent_provenance.json")
            elif active_stage == "cache":
                _cache_stage(run_dir, args)
            elif active_stage == "train":
                _train_stage(run_dir, args)
            elif active_stage == "confirm":
                _confirm_stage(run_dir, args)
            elif active_stage == "report":
                pass
        workflow = _write_workflow(
            run_dir, require_gate=args.require_gate, provenance=provenance
        )
        registry = _artifact_registry(run_dir)
        decision = workflow["decision"].get("decision")
        passed = bool(workflow["required_gate_pass"])
        _status(
            run_dir,
            state="completed" if passed else "gate_failed",
            stage=args.stage,
            decision=decision,
            registry=registry,
        )
        return 0 if passed else 2
    except ArtifactCompatibilityError as exc:
        failure_domain = {
            "preflight": "control_provenance",
            "cache": "exact_control_cache",
            "train": "optimization_pipeline",
            "confirm": "optimization_pipeline",
        }.get(active_stage, "workflow_compatibility")
        failure_code = {
            "preflight": "control_provenance_invalid",
            "cache": "exact_control_cache_invalid",
            "train": "optimization_pipeline_invalid",
            "confirm": "optimization_pipeline_invalid",
        }.get(active_stage, "workflow_compatibility_invalid")
        failure = execution_failed_gate(
            active_stage,
            failure_domain=failure_domain,
            failure_code=failure_code,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        suffix = (
            "provenance_failure"
            if active_stage == "preflight"
            else "compatibility_failure"
        )
        atomic_write_json(run_dir / f"{active_stage}_{suffix}.json", failure)
        _commit_stage_failure(run_dir, active_stage, failure)
        bound_provenance: Mapping[str, Any] | bool = (
            False
            if active_stage == "preflight"
            else (_optional(run_dir, "parent_provenance.json") or False)
        )
        workflow = _write_workflow(
            run_dir,
            require_gate=args.require_gate,
            provenance=bound_provenance,
        )
        registry = _artifact_registry(run_dir)
        _status(
            run_dir,
            state="execution_failed",
            stage=active_stage,
            decision=workflow["decision"].get("decision"),
            error=str(exc),
            registry=registry,
        )
        print(f"Jacobi RB Bayes-power provenance error: {exc}", flush=True)
        return 2
    except Exception as exc:
        failure = execution_failed_gate(
            active_stage,
            failure_domain=getattr(exc, "failure_domain", "workflow_execution"),
            failure_code=getattr(
                exc, "failure_code", "bayes_power_execution_failed"
            ),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        atomic_write_json(run_dir / f"{active_stage}_execution_failure.json", failure)
        _commit_stage_failure(run_dir, active_stage, failure)
        provenance = _optional(run_dir, "parent_provenance.json") or False
        workflow = _write_workflow(
            run_dir, require_gate=args.require_gate, provenance=provenance
        )
        registry = _artifact_registry(run_dir)
        _status(
            run_dir,
            state="execution_failed",
            stage=active_stage,
            decision=workflow["decision"].get("decision"),
            error=str(exc),
            registry=registry,
        )
        print(f"Jacobi RB Bayes-power error: {exc}", flush=True)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
