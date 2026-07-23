from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mnist.d0_one_image_gate import (
    ArtifactCompatibilityError,
    OneImageGateThresholds,
    config_fingerprint,
    file_fingerprint,
)
from mnist.diag_d0_one_image_overfit import (
    LATEST_CHECKPOINT_SCHEMA,
    LATEST_CHECKPOINT_SCHEMA_VERSION,
    _finalize_gate,
    _load_latest_training_checkpoint,
    _make_configs,
    _recover_selected_best_checkpoint,
    _scientific_payload,
    _select_source_image,
    _validate_committed_cache_identity,
    _validate_preflight_binding,
    _validate_selected_checkpoint_binding,
    parse_args,
    verify_zero_residual_run,
)
from mnist.experiment12_d0 import _lambda_mixed_data_for_paths, synthetic_digit_measures


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _upstream_run(path: Path, *, limiter_fraction: float = 1.0) -> None:
    path.mkdir()
    _write_json(
        path / "aggregate_summary.json",
        {"training_ready": 1, "config_fingerprint": "upstream-fingerprint"},
    )
    _write_json(
        path / "run_status.json",
        {"required_gate_pass": 1, "config_fingerprint": "upstream-fingerprint"},
    )
    _write_json(
        path / "run_config.json",
        {
            "dynamics_config": {
                "grid_size": 28,
                "edge_alpha_mode": "alpha_eff",
                "alpha_eff": 1.0,
                "mass_floor": 1e-7,
                "limiter_fraction": limiter_fraction,
            },
            "diagnostic_config": {
                "sample_steps": 512,
                "substep_levels": [64, 128, 256],
                "tau_eff": 5e-5,
            },
        },
    )


def _production_args(tmp_path: Path):
    return parse_args(["--zero-residual-run-dir", str(tmp_path / "zero")])


def test_parser_freezes_production_defaults_and_report_only_is_not_gateable(
    tmp_path: Path,
) -> None:
    args = _production_args(tmp_path)
    assert args.stage == "all"
    assert args.validation_paths == 16
    assert args.overfit_eval_seeds == (260719, 260720)
    assert args.samples_per_seed == 8
    assert args.reference_substeps == 256
    assert args.limiter_fraction == 1.0
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--zero-residual-run-dir",
                str(tmp_path / "zero"),
                "--stage",
                "evaluate",
                "--checkpoint-path",
                str(tmp_path / "legacy.pt"),
                "--require-gate",
                "reconstruction",
            ]
        )


def test_required_gate_rejects_changed_production_defaults_or_thresholds(
    tmp_path: Path,
) -> None:
    base = ["--zero-residual-run-dir", str(tmp_path / "zero")]
    with pytest.raises(SystemExit):
        parse_args(base + ["--require-gate", "reconstruction", "--train-steps", "9999"])
    with pytest.raises(SystemExit):
        parse_args(
            base
            + [
                "--require-gate",
                "reconstruction",
                "--min-mean-correlation",
                "0.1",
            ]
        )
    exploratory = parse_args(
        base
        + [
            "--require-gate",
            "none",
            "--train-steps",
            "3",
            "--min-mean-correlation",
            "0.1",
        ]
    )
    assert exploratory.train_steps == 3
    assert exploratory.min_mean_correlation == 0.1


def test_behavior_affecting_runtime_options_change_scientific_fingerprint(
    tmp_path: Path,
) -> None:
    base = ["--zero-residual-run-dir", str(tmp_path / "zero")]
    default_args = parse_args(base)
    changed_args = parse_args(
        base
        + [
            "--no-amp",
            "--validation-batch-size",
            "17",
            "--cache-batch-size",
            "3",
            "--physical-target-scale-floor",
            "99",
        ]
    )
    source = {
        "label": 3,
        "class_index": 0,
        "dataset_index": 7,
        "image_sha256": "image",
        "mixed_target_sha256": "mixed",
    }
    upstream = {"config_fingerprint": "upstream"}

    def fingerprint(args) -> str:
        dynamics, d0 = _make_configs(args)
        payload = _scientific_payload(
            args,
            dynamics,
            d0,
            OneImageGateThresholds(),
            source,
            upstream,
        )
        return config_fingerprint(payload)

    assert fingerprint(default_args) != fingerprint(changed_args)


def test_preflight_record_is_bound_to_recomputed_cache_evidence() -> None:
    metrics = {"cache_paths": 8, "raw_limited_fraction": 0.001}
    gate = {"passed": 1, "checks": {"cache_paths": 1}}
    metrics_fingerprint = config_fingerprint(metrics)
    gate_fingerprint = config_fingerprint(gate)
    artifact = SimpleNamespace(
        cache_fingerprint="cache-fingerprint",
        metadata={
            "metrics_fingerprint": metrics_fingerprint,
            "gate_fingerprint": gate_fingerprint,
        },
    )
    record = {
        "cache_fingerprint": "cache-fingerprint",
        "metrics_fingerprint": metrics_fingerprint,
        "gate_fingerprint": gate_fingerprint,
        "metrics": metrics,
        "gate": gate,
    }
    _validate_preflight_binding(
        record,
        artifact,
        recomputed_metrics=metrics,
        recomputed_gate=gate,
    )
    edited = dict(record)
    edited["gate"] = {"passed": 0, "checks": {"cache_paths": 0}}
    with pytest.raises(ArtifactCompatibilityError, match="gate does not reproduce"):
        _validate_preflight_binding(
            edited,
            artifact,
            recomputed_metrics=metrics,
            recomputed_gate=gate,
        )


def test_resume_rebuilds_stale_best_file_from_selected_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    selected_step = checkpoints / "step-00000500.pt"
    selected_step.write_bytes(b"authoritative-step-placeholder")
    (checkpoints / "best_ema.pt").write_bytes(b"stale")

    def fake_load(path, **kwargs):
        assert Path(path) == selected_step
        assert kwargs["expected_fingerprints"] == {"cache": "frozen"}
        return {"step": 500, "marker": "selected"}

    monkeypatch.setattr(
        "mnist.diag_d0_one_image_overfit.load_training_checkpoint", fake_load
    )
    best = _recover_selected_best_checkpoint(
        checkpoints,
        {"step": 500, "primary_mse": 0.1},
        {"cache": "frozen"},
    )
    assert best.read_bytes() == selected_step.read_bytes()
    assert file_fingerprint(best) == file_fingerprint(selected_step)


def test_manifest_rejects_replacement_of_a_committed_training_cache() -> None:
    manifest = {
        "artifacts": {
            "cache": {
                "cache_fingerprint": "cache-a",
                "split_fingerprint": "split-a",
                "physical_target_scale": 2.5,
            }
        }
    }
    _validate_committed_cache_identity(
        manifest,
        cache_fingerprint="cache-a",
        split_fingerprint="split-a",
        physical_target_scale=2.5,
    )
    with pytest.raises(ArtifactCompatibilityError, match="committed"):
        _validate_committed_cache_identity(
            manifest,
            cache_fingerprint="cache-b",
            split_fingerprint="split-a",
            physical_target_scale=2.5,
        )
    # Absence is the explicitly supported cache-to-manifest crash window.
    _validate_committed_cache_identity(
        {"artifacts": {}},
        cache_fingerprint="cache-a",
        split_fingerprint="split-a",
        physical_target_scale=2.5,
    )


def test_latest_checkpoint_pointer_binds_step_fingerprints_and_file_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    checkpoint = checkpoints / "step-00000500.pt"
    checkpoint.write_bytes(b"checkpoint-content")
    latest = checkpoints / "latest.json"
    _write_json(
        latest,
        {
            "schema": LATEST_CHECKPOINT_SCHEMA,
            "schema_version": LATEST_CHECKPOINT_SCHEMA_VERSION,
            "filename": checkpoint.name,
            "step": 500,
            "fingerprints": {"cache": "frozen"},
            "checkpoint_sha256": file_fingerprint(checkpoint),
        },
    )

    def fake_load(path, **kwargs):
        assert Path(path) == checkpoint
        assert kwargs["expected_fingerprints"] == {"cache": "frozen"}
        return {"step": 500}

    monkeypatch.setattr(
        "mnist.diag_d0_one_image_overfit.load_training_checkpoint", fake_load
    )
    payload = _load_latest_training_checkpoint(
        latest,
        checkpoints,
        map_location="cpu",
        strict_fingerprints={"cache": "frozen"},
    )
    assert payload["step"] == 500
    checkpoint.write_bytes(b"replacement")
    with pytest.raises(ArtifactCompatibilityError, match="file hash"):
        _load_latest_training_checkpoint(
            latest,
            checkpoints,
            map_location="cpu",
            strict_fingerprints={"cache": "frozen"},
        )


def test_selected_checkpoint_is_bound_to_validation_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    checkpoint = checkpoints / "best_ema.pt"
    checkpoint.write_bytes(b"selected-checkpoint")
    (checkpoints / "step-00000500.pt").write_bytes(checkpoint.read_bytes())
    selected = {"step": 500, "primary_mse": 0.125, "prediction_gain": 0.5}
    _write_json(
        tmp_path / "checkpoint_selection.json",
        {
            "selected": selected,
            "checkpoint_sha256": file_fingerprint(checkpoint),
        },
    )
    latest_checkpoint = checkpoints / "step-00010000.pt"
    latest_checkpoint.write_bytes(b"latest-checkpoint")
    _write_json(
        checkpoints / "latest.json",
        {
            "schema": LATEST_CHECKPOINT_SCHEMA,
            "schema_version": LATEST_CHECKPOINT_SCHEMA_VERSION,
            "filename": latest_checkpoint.name,
            "step": 10_000,
            "fingerprints": {"cache": "frozen"},
            "checkpoint_sha256": file_fingerprint(latest_checkpoint),
        },
    )

    def fake_load(path, **kwargs):
        assert Path(path) == latest_checkpoint
        return {"step": 10_000, "best_validation": selected}

    monkeypatch.setattr(
        "mnist.diag_d0_one_image_overfit.load_training_checkpoint", fake_load
    )
    payload = {"step": 500, "best_validation": selected}
    _validate_selected_checkpoint_binding(
        tmp_path,
        checkpoint,
        payload,
        strict_fingerprints={"cache": "frozen"},
        expected_training_step=10_000,
    )
    checkpoint.write_bytes(b"replacement")
    with pytest.raises(ArtifactCompatibilityError, match="selection"):
        _validate_selected_checkpoint_binding(
            tmp_path,
            checkpoint,
            payload,
            strict_fingerprints={"cache": "frozen"},
            expected_training_step=10_000,
        )


def test_upstream_verification_uses_semantic_kernel_not_source_hashes(
    tmp_path: Path,
) -> None:
    args = _production_args(tmp_path)
    dynamics, d0 = _make_configs(args)
    upstream = tmp_path / "zero"
    _upstream_run(upstream)
    result = verify_zero_residual_run(
        upstream, dynamics_config=dynamics, d0_config=d0
    )
    assert result["training_ready"] == 1
    assert result["config_fingerprint"] == "upstream-fingerprint"
    assert result["semantic_kernel"]["reference_substeps"] == 256


def test_upstream_verification_rejects_changed_kernel(tmp_path: Path) -> None:
    args = _production_args(tmp_path)
    dynamics, d0 = _make_configs(args)
    upstream = tmp_path / "zero"
    _upstream_run(upstream, limiter_fraction=0.25)
    with pytest.raises(ValueError, match="kernel mismatch"):
        verify_zero_residual_run(upstream, dynamics_config=dynamics, d0_config=d0)


def test_recorded_mixed_target_exactly_matches_cache_source_construction() -> None:
    images, labels = synthetic_digit_measures(
        examples_per_class=1, grid_size=8, seed=4
    )
    selected = _select_source_image(
        images,
        labels,
        label=3,
        class_index=0,
        grid_size=8,
        lambda_mix=0.35,
    )
    mixed, requested, indices = _lambda_mixed_data_for_paths(
        images,
        labels,
        count=2,
        lambda_mix=0.35,
        grid_size=8,
        rng=np.random.default_rng(1),
        single_image_overfit=True,
        single_image_index=0,
        single_image_label=3,
    )
    assert requested.tolist() == [3, 3]
    assert indices.tolist() == [selected["dataset_index"], selected["dataset_index"]]
    assert np.array_equal(np.broadcast_to(selected["mixed_target"], mixed.shape), mixed)


def _passing_cache() -> dict[str, object]:
    return {
        "cache_paths": 8,
        "cache_build_mode": "substep",
        "cache_stride_substeps": 1,
        "physical_target_scale": 1.0,
        "target_finite_fraction": 1.0,
        "oracle_direct_l1": 0.1,
        "oracle_positive_free_only_l1": 0.2,
        "terminal_target_abs_corr_mean": 0.05,
        "nonfinite_edges": 0,
        "floor_touched_pixels": 0,
        "max_simplex_mass_error": 1e-7,
        "floor_correction_l1_per_path_substep": 0.0,
        "renorm_correction_l1_per_path_substep": 0.0,
        "raw_limited_fraction": 0.001,
        "mobility_weighted_limited_fraction": 1e-5,
        "noise_energy_weighted_limited_fraction": 1e-5,
    }


def _passing_optimization() -> dict[str, object]:
    return {
        "selected_ema_validation_gain": 0.1,
        "selected_ema_data_end_gain": 0.05,
        "selected_ema_data_end_count": 4,
    }


def _healthy_arm() -> dict[str, object]:
    return {
        "nonfinite_edges": 0,
        "floor_touched_pixels": 0,
        "max_simplex_mass_error": 1e-7,
        "floor_correction_l1_per_path_substep": 0.0,
        "renorm_correction_l1_per_path_substep": 0.0,
        "limiter_fraction": 0.001,
        "mobility_weighted_limiter_fraction": 1e-5,
        "noise_energy_weighted_limiter_fraction": 1e-5,
    }


def _passing_reconstruction() -> dict[str, object]:
    return {
        "complete": 1,
        "sample_count": 16,
        "strength_1_mean_corr": 0.95,
        "strength_1_mean_l1": 0.1,
        "strength_1_good_corr_fraction": 0.9,
        "paired_mean_corr_improvement": 0.3,
        "relative_l1_reduction": 0.4,
        "strength_0": _healthy_arm(),
        "strength_1": _healthy_arm(),
    }


def test_required_gate_artifact_is_written_before_failure(tmp_path: Path) -> None:
    reconstruction = _passing_reconstruction()
    reconstruction["strength_1_mean_corr"] = 0.89
    report = _finalize_gate(
        run_dir=tmp_path,
        cache_metrics=_passing_cache(),
        optimization_metrics=_passing_optimization(),
        reconstruction_metrics=reconstruction,
        require_gate="reconstruction",
        thresholds=OneImageGateThresholds(),
        strict_checkpoint_eligible=True,
        skips=[],
    )
    artifact = tmp_path / "overfit_gate.json"
    assert report["required_gate_pass"] == 0
    assert artifact.is_file()
    persisted = json.loads(artifact.read_text(encoding="utf-8"))
    assert persisted["required_gate_pass"] == 0
    assert persisted["artifacts_complete_before_exit"] == 1


def test_report_only_checkpoint_cannot_be_promoted_to_required_gate(
    tmp_path: Path,
) -> None:
    report = _finalize_gate(
        run_dir=tmp_path,
        cache_metrics=_passing_cache(),
        optimization_metrics=_passing_optimization(),
        reconstruction_metrics=_passing_reconstruction(),
        require_gate="reconstruction",
        thresholds=OneImageGateThresholds(),
        strict_checkpoint_eligible=False,
        skips=[],
    )
    assert report["reconstruction"]["passed"] == 1
    assert report["required_gate_pass"] == 0
    assert "cannot satisfy" in report["strict_eligibility_failure"]
