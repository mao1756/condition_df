from __future__ import annotations

import ast
import copy
import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from mnist.d0_one_image_gate import ArtifactCompatibilityError
import mnist.diag_d0_multiscale_learnability as multiscale
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig


SOURCE_IMAGE = {
    "label": 3,
    "class_index": 0,
    "dataset_index": 17,
    "image_sha256": "image-sha256",
    "mixed_target_sha256": "mixed-target-sha256",
}


class _TinyFluxModel(nn.Module):
    """One-parameter task model that keeps orchestration tests inexpensive."""

    def __init__(self, config: DirectFluxMNISTConfig, *, base_channels: int = 1) -> None:
        super().__init__()
        del base_channels
        self.grid_size = int(config.grid_size)
        self.multiplier = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def forward(
        self,
        tau: torch.Tensor,
        masses: torch.Tensor,
        labels: torch.Tensor,
        source_masses: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del tau, labels, source_masses
        grid = masses.reshape(-1, self.grid_size, self.grid_size)
        horizontal = torch.roll(grid, shifts=-1, dims=2) - grid
        vertical = torch.roll(grid, shifts=-1, dims=1) - grid
        return self.multiplier * torch.stack((horizontal, vertical), dim=1)


def _task_fixture(*, flip_audit_target: bool = False) -> multiscale.TaskArrays:
    grid_size = 4
    rows: list[np.ndarray] = []
    tau_fraction: list[float] = []
    path_ids: list[int] = []
    for path_id in range(3):
        for bin_index, fraction in enumerate((0.1, 0.3, 0.5, 0.7, 0.9)):
            raw = np.arange(1, grid_size * grid_size + 1, dtype=np.float64)
            raw = np.roll(raw, path_id * 2 + bin_index)
            raw = raw + 0.1 * path_id + 0.01 * bin_index
            rows.append(raw / raw.sum())
            tau_fraction.append(fraction)
            path_ids.append(path_id)
    states = torch.as_tensor(np.asarray(rows), dtype=torch.float32)
    grid = states.reshape(-1, grid_size, grid_size)
    targets = 0.4 * torch.stack(
        (
            torch.roll(grid, shifts=-1, dims=2) - grid,
            torch.roll(grid, shifts=-1, dims=1) - grid,
        ),
        dim=1,
    )
    path_array = np.asarray(path_ids, dtype=np.int64)
    if flip_audit_target:
        targets[path_array == 2] *= -3.0
    tau_fraction_tensor = torch.as_tensor(tau_fraction, dtype=torch.float32)
    return multiscale.TaskArrays(
        states=states,
        tau=tau_fraction_tensor * 0.25,
        tau_fraction=tau_fraction_tensor,
        labels=torch.full((states.shape[0],), 3, dtype=torch.long),
        targets=targets,
        path_ids=path_array,
    )


def _task_args() -> SimpleNamespace:
    return SimpleNamespace(
        base_channels=1,
        learning_rate=1e-2,
        weight_decay=0.0,
        no_amp=True,
        validation_batch_size=4,
        batch_size=4,
        grad_clip=0.0,
        ema_decay=0.9,
        validation_every=1,
        checkpoint_every=1,
    )


def _train_tiny_task(
    task_dir: Path,
    *,
    arrays: multiscale.TaskArrays,
    stride: int,
    seed: int = 77,
    train_steps: int = 2,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fingerprints = {"fixture": task_dir.name, "stride": int(stride)}
    result, debug = multiscale.train_task(
        task_dir=task_dir,
        task_name=task_dir.name,
        arrays=arrays,
        split_path_ids={"train": [0], "selection": [1], "audit": [2]},
        dynamics=DirectFluxMNISTConfig(
            grid_size=4,
            num_steps=2,
            source_lowfreq_size=2,
            source_blur_sigma=0.0,
            ot_lowres_size=2,
            ot_blur_sigma=0.0,
            condition_on_source=False,
            flux_parameterization="edge",
            limiter_fraction=1.0,
            edge_alpha_mode="alpha_eff",
            alpha_eff=1.0,
            mass_floor=1e-8,
        ),
        training_seed=int(seed),
        train_steps=int(train_steps),
        args=_task_args(),
        fingerprints=fingerprints,
        device=torch.device("cpu"),
        show_progress=False,
        stride=int(stride),
    )
    return result, debug, fingerprints


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_cli(tmp_path: Path) -> list[str]:
    return [
        "--zero-residual-run-dir",
        str(tmp_path / "zero-residual"),
        "--parent-one-image-run-dir",
        str(tmp_path / "one-image"),
    ]


def _production_args(tmp_path: Path):
    return multiscale.parse_args(_base_cli(tmp_path))


def _write_zero_residual_run(
    path: Path,
    *,
    training_ready: int = 1,
    required_gate_pass: int = 1,
    limiter_fraction: float = 1.0,
) -> None:
    path.mkdir()
    _write_json(
        path / "aggregate_summary.json",
        {
            "training_ready": training_ready,
            "config_fingerprint": "zero-residual-fingerprint",
        },
    )
    _write_json(
        path / "run_status.json",
        {
            "required_gate_pass": required_gate_pass,
            "config_fingerprint": "zero-residual-fingerprint",
        },
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


def _write_parent_one_image_run(
    path: Path,
    *,
    source_overrides: dict[str, Any] | None = None,
    kernel_overrides: dict[str, Any] | None = None,
    upstream_fingerprint: str = "zero-residual-fingerprint",
    cache_passed: int = 1,
    optimization_passed: int = 0,
    status: str = "complete",
    outcome: str = "gate_failed",
) -> None:
    path.mkdir()
    source = dict(SOURCE_IMAGE)
    source.update(source_overrides or {})
    kernel = dict(multiscale.EXPECTED_KERNEL)
    kernel.update(kernel_overrides or {})
    _write_json(
        path / "run_manifest.json",
        {
            "schema": "experiment12-d0-production-one-image",
            "scientific_fingerprint": "one-image-scientific-fingerprint",
            "scientific_config": {
                "source_image": source,
                "upstream_zero_residual_fingerprint": upstream_fingerprint,
                "kernel": kernel,
            },
            "artifacts": {
                "cache": {"cache_fingerprint": "one-image-cache-fingerprint"}
            },
        },
    )
    _write_json(
        path / "run_status.json",
        {"status": status, "outcome": outcome},
    )
    _write_json(
        path / "overfit_gate.json",
        {
            "cache": {"passed": cache_passed},
            "optimization": {"passed": optimization_passed},
        },
    )
    _write_json(
        path / "checkpoint_selection.json",
        {
            "selected": {
                "step": 500,
                "prediction_gain": -0.0204729,
                "data_end_prediction_gain": -0.0216950,
            }
        },
    )


def _write_parent_multiscale_run(
    path: Path,
    *,
    source_overrides: dict[str, Any] | None = None,
    status_overrides: dict[str, Any] | None = None,
    decision: str = "inconclusive",
    cache_passed: int = 1,
    teacher_passed: int = 1,
    incomplete_task: tuple[int, int] | None = None,
) -> None:
    path.mkdir()
    pilot = multiscale.PILOT_REQUIRED_DEFAULTS
    source = dict(SOURCE_IMAGE)
    source.update(source_overrides or {})
    scientific = {
        "algorithm": multiscale.RUN_SCHEMA,
        "algorithm_version": 1,
        "target_contract": multiscale.TARGET_CONTRACT,
        "kernel": {
            **multiscale.EXPECTED_KERNEL,
            "edge_alpha_value": 1.0,
            "integrator": "masked_reference_free_step_torch",
        },
        "cache": {
            "temporal_strides": list(pilot["temporal_strides"]),
            "paths": pilot["cache_paths"],
            "anchors_per_path": pilot["anchors_per_path"],
            "anchor_bin_counts": list(pilot["anchor_bin_counts"]),
            "preflight_paths": pilot["preflight_paths"],
            "shard_paths": pilot["cache_shard_paths"],
            "cache_seed": pilot["cache_seed"],
            "dataset_seed": pilot["dataset_seed"],
            "split_seed": pilot["split_seed"],
            "train_paths": pilot["train_paths"],
            "selection_paths": pilot["selection_paths"],
            "audit_paths": pilot["audit_paths"],
            "target_scale_floor": pilot["target_scale_floor"],
        },
        "training": {
            "seeds": list(pilot["training_seeds"]),
            "teacher_seed": pilot["teacher_seed"],
            "steps": pilot["train_steps"],
            "teacher_steps": pilot["teacher_steps"],
            "batch_size": pilot["batch_size"],
            "base_channels": pilot["base_channels"],
            "learning_rate": pilot["learning_rate"],
            "weight_decay": pilot["weight_decay"],
            "grad_clip": pilot["grad_clip"],
            "ema_decay": pilot["ema_decay"],
            "validation_every": pilot["validation_every"],
            "checkpoint_every": pilot["checkpoint_every"],
            "validation_batch_size": pilot["validation_batch_size"],
            "use_amp": not pilot["no_amp"],
        },
        "gate": {
            "bootstrap_seed": pilot["bootstrap_seed"],
            "bootstrap_reps": pilot["bootstrap_reps"],
            "bootstrap_confidence": pilot["bootstrap_confidence"],
            "teacher_min_gain": pilot["teacher_min_gain"],
            "max_raw_intervention": pilot["max_raw_intervention"],
            "max_weighted_intervention": pilot["max_weighted_intervention"],
            "max_floor_correction_l1": pilot["max_floor_correction_l1"],
            "max_renorm_correction_l1": pilot["max_renorm_correction_l1"],
            "max_simplex_mass_error": pilot["max_simplex_mass_error"],
        },
        "source_image": source,
        "upstream_zero_residual_fingerprint": "zero-residual-fingerprint",
        "parent_one_image_fingerprint": "one-image-scientific-fingerprint",
        "sampling_performed": 0,
    }
    _write_json(
        path / "run_manifest.json",
        {
            "schema": multiscale.RUN_SCHEMA,
            "schema_version": 1,
            "scientific_fingerprint": "pilot-scientific-fingerprint",
            "cache_semantic_fingerprint": "pilot-cache-semantic-fingerprint",
            "source_fingerprint": "old-source-fingerprint-is-advisory",
            "scientific_config": scientific,
            "artifacts": {"cache_index_fingerprint": "pilot-cache-index"},
            "sampling_performed": 0,
        },
    )
    status = {
        "status": "complete",
        "outcome": "gate_failed",
        "required_gate": "any-scale",
        "required_gate_pass": 0,
        "skips": [],
        "sampling_performed": 0,
    }
    status.update(status_overrides or {})
    _write_json(path / "run_status.json", status)
    stride_gates = {
        str(stride): {"passed": 0}
        for stride in pilot["temporal_strides"]
    }
    _write_json(
        path / "learnability_decision.json",
        {
            "required_gate": "any-scale",
            "required_gate_pass": 0,
            "cache": {"passed": cache_passed},
            "teacher": {"passed": teacher_passed},
            "strides": stride_gates,
            "decision": {"decision": decision, "sampling_performed": 0},
            "sampling_performed": 0,
        },
    )
    _write_json(path / "cache_gate.json", {"passed": cache_passed})
    _write_json(path / "teacher_control.json", {"gate": {"passed": teacher_passed}})
    _write_json(
        path / "task_failures.json",
        {"failure_count": 0, "failures": [], "sampling_performed": 0},
    )
    lines = ["stride,training_seed,complete,finite,selected_step"]
    for stride in pilot["temporal_strides"]:
        for seed in pilot["training_seeds"]:
            complete = 0 if incomplete_task == (stride, seed) else 1
            lines.append(f"{stride},{seed},{complete},1,250")
    (path / "stride_seed_metrics.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def test_required_gate_freezes_every_production_default(tmp_path: Path) -> None:
    args = multiscale.parse_args(_base_cli(tmp_path) + ["--require-gate", "cache"])

    for name, expected in multiscale.REQUIRED_DEFAULTS.items():
        assert getattr(args, name) == expected, name


def test_confirmation_profile_freezes_independent_128_path_defaults(
    tmp_path: Path,
) -> None:
    args = multiscale.parse_args(
        _base_cli(tmp_path)
        + [
            "--study-profile",
            "confirmation",
            "--parent-multiscale-run-dir",
            str(tmp_path / "pilot"),
            "--require-gate",
            "any-scale",
        ]
    )

    for name, expected in multiscale.CONFIRMATION_REQUIRED_DEFAULTS.items():
        assert getattr(args, name) == expected, name
    assert args.study_profile == "confirmation"
    assert args.profile_conformant is True
    assert args.cache_paths == 128
    assert (args.train_paths, args.selection_paths, args.audit_paths) == (80, 24, 24)
    assert len(args.training_seeds) == 5


def test_confirmation_profile_requires_pilot_parent(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        multiscale.parse_args(
            _base_cli(tmp_path) + ["--study-profile", "confirmation"]
        )


def test_confirmation_required_gate_rejects_profile_override(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        multiscale.parse_args(
            _base_cli(tmp_path)
            + [
                "--study-profile",
                "confirmation",
                "--parent-multiscale-run-dir",
                str(tmp_path / "pilot"),
                "--require-gate",
                "any-scale",
                "--cache-paths",
                "127",
            ]
        )


@pytest.mark.parametrize(
    "override",
    [
        ["--train-steps", "2999"],
        ["--teacher-min-gain", "0.8"],
        ["--mass-floor", "2e-7"],
        ["--no-amp"],
    ],
)
def test_required_gate_rejects_production_profile_overrides(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    override: list[str],
) -> None:
    with pytest.raises(SystemExit):
        multiscale.parse_args(
            _base_cli(tmp_path) + ["--require-gate", "any-scale"] + override
        )

    assert "required gates freeze the production learnability profile" in capsys.readouterr().err


def test_exploratory_mode_accepts_coherent_overrides(tmp_path: Path) -> None:
    args = multiscale.parse_args(
        _base_cli(tmp_path)
        + [
            "--require-gate",
            "none",
            "--synthetic-data",
            "--cache-paths",
            "9",
            "--train-paths",
            "3",
            "--selection-paths",
            "3",
            "--audit-paths",
            "3",
            "--anchors-per-path",
            "5",
            "--anchor-bin-counts",
            "1,1,1,1,1",
            "--temporal-strides",
            "64,1,16,16",
            "--training-seeds",
            "7,8,9",
            "--train-steps",
            "12",
            "--teacher-min-gain",
            "0.1",
            "--mass-floor",
            "2e-7",
            "--no-amp",
        ]
    )

    assert args.synthetic_data is True
    assert args.cache_paths == 9
    assert (args.train_paths, args.selection_paths, args.audit_paths) == (3, 3, 3)
    assert args.anchor_bin_counts == (1, 1, 1, 1, 1)
    assert args.temporal_strides == (1, 16, 64)
    assert args.training_seeds == (7, 8, 9)
    assert args.train_steps == 12
    assert args.teacher_min_gain == 0.1
    assert args.mass_floor == 2e-7
    assert args.no_amp is True


def test_zero_residual_verification_accepts_training_ready_semantics(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "zero-residual"
    _write_zero_residual_run(run_dir)

    result = multiscale.verify_zero_residual_run(run_dir, _production_args(tmp_path))

    assert result["config_fingerprint"] == "zero-residual-fingerprint"
    assert result["semantic_kernel"] == {
        "grid_size": 28,
        "sample_steps": 512,
        "reference_substeps": 256,
        "tau_eff": 5e-5,
        "edge_alpha_mode": "alpha_eff",
        "alpha_eff": 1.0,
        "mass_floor": 1e-7,
        "limiter_fraction": 1.0,
    }
    assert result["aggregate_sha256"]
    assert result["run_config_sha256"]


@pytest.mark.parametrize(
    ("training_ready", "required_gate_pass"),
    [(0, 1), (1, 0)],
)
def test_zero_residual_verification_requires_both_readiness_records(
    tmp_path: Path,
    training_ready: int,
    required_gate_pass: int,
) -> None:
    run_dir = tmp_path / "zero-residual"
    _write_zero_residual_run(
        run_dir,
        training_ready=training_ready,
        required_gate_pass=required_gate_pass,
    )

    with pytest.raises(ArtifactCompatibilityError, match="did not pass training-ready"):
        multiscale.verify_zero_residual_run(run_dir, _production_args(tmp_path))


def test_zero_residual_verification_rejects_upstream_or_current_kernel_changes(
    tmp_path: Path,
) -> None:
    mismatched_upstream = tmp_path / "mismatched-zero"
    _write_zero_residual_run(mismatched_upstream, limiter_fraction=0.25)
    with pytest.raises(
        ArtifactCompatibilityError,
        match="upstream zero-residual limiter_fraction",
    ):
        multiscale.verify_zero_residual_run(
            mismatched_upstream, _production_args(tmp_path)
        )

    matching_upstream = tmp_path / "matching-zero"
    _write_zero_residual_run(matching_upstream)
    exploratory_args = multiscale.parse_args(
        _base_cli(tmp_path)
        + ["--require-gate", "none", "--limiter-fraction", "0.25"]
    )
    with pytest.raises(ArtifactCompatibilityError, match="current limiter_fraction"):
        multiscale.verify_zero_residual_run(matching_upstream, exploratory_args)


def test_parent_verification_accepts_matching_completed_optimization_failure(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "one-image"
    _write_parent_one_image_run(run_dir)

    result = multiscale.verify_parent_one_image_run(
        run_dir,
        source_image=SOURCE_IMAGE,
        upstream_fingerprint="zero-residual-fingerprint",
    )

    assert result["scientific_fingerprint"] == "one-image-scientific-fingerprint"
    assert result["cache_fingerprint"] == "one-image-cache-fingerprint"
    assert result["selected_step"] == 500
    assert result["selected_prediction_gain"] == pytest.approx(-0.0204729)
    assert result["selected_data_end_gain"] == pytest.approx(-0.0216950)
    assert result["source_image"] == SOURCE_IMAGE


@pytest.mark.parametrize(
    ("fixture_kwargs", "error"),
    [
        ({"source_overrides": {"image_sha256": "different"}}, "source-image mismatch"),
        ({"upstream_fingerprint": "different"}, "fingerprints differ"),
        ({"kernel_overrides": {"lambda_mix": 0.5}}, "kernel mismatch for lambda_mix"),
    ],
)
def test_parent_verification_rejects_semantic_mismatches(
    tmp_path: Path,
    fixture_kwargs: dict[str, Any],
    error: str,
) -> None:
    run_dir = tmp_path / "one-image"
    _write_parent_one_image_run(run_dir, **fixture_kwargs)

    with pytest.raises(ArtifactCompatibilityError, match=error):
        multiscale.verify_parent_one_image_run(
            run_dir,
            source_image=SOURCE_IMAGE,
            upstream_fingerprint="zero-residual-fingerprint",
        )


@pytest.mark.parametrize(
    ("fixture_kwargs", "error"),
    [
        ({"cache_passed": 0}, "cache gate did not pass"),
        ({"optimization_passed": 1}, "optimization was not the recorded failure"),
        ({"outcome": "gate_passed"}, "not the completed failed gate"),
    ],
)
def test_parent_verification_requires_cache_pass_and_optimization_failure(
    tmp_path: Path,
    fixture_kwargs: dict[str, Any],
    error: str,
) -> None:
    run_dir = tmp_path / "one-image"
    _write_parent_one_image_run(run_dir, **fixture_kwargs)

    with pytest.raises(ArtifactCompatibilityError, match=error):
        multiscale.verify_parent_one_image_run(
            run_dir,
            source_image=SOURCE_IMAGE,
            upstream_fingerprint="zero-residual-fingerprint",
        )


def test_legacy_pilot_verification_binds_semantics_not_old_source_hash(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "pilot"
    _write_parent_multiscale_run(run_dir)

    result = multiscale.verify_parent_multiscale_run(
        run_dir,
        source_image=SOURCE_IMAGE,
        upstream_fingerprint="zero-residual-fingerprint",
        parent_one_image_fingerprint="one-image-scientific-fingerprint",
    )

    assert result["study_profile"] == "pilot"
    assert result["profile_inferred_from_legacy"] == 1
    assert result["task_count"] == 15
    assert result["decision"] == "inconclusive"
    assert result["seed_plan"]["cache_seed"] == 260721
    assert Path(result["decision_path"]).is_file()


@pytest.mark.parametrize(
    ("fixture_kwargs", "error"),
    [
        ({"status_overrides": {"skips": [{"stage": "physical"}]}}, "skipped work"),
        ({"decision": "no_detectable_conditional_signal"}, "not inconclusive"),
        ({"cache_passed": 0}, "cache gate did not pass"),
        ({"teacher_passed": 0}, "teacher gate did not pass"),
        ({"incomplete_task": (1024, 260725)}, "all 15 complete"),
        ({"source_overrides": {"image_sha256": "changed"}}, "source-image mismatch"),
    ],
)
def test_legacy_pilot_verification_rejects_incomplete_or_changed_evidence(
    tmp_path: Path,
    fixture_kwargs: dict[str, Any],
    error: str,
) -> None:
    run_dir = tmp_path / "pilot"
    _write_parent_multiscale_run(run_dir, **fixture_kwargs)

    with pytest.raises(ArtifactCompatibilityError, match=error):
        multiscale.verify_parent_multiscale_run(
            run_dir,
            source_image=SOURCE_IMAGE,
            upstream_fingerprint="zero-residual-fingerprint",
            parent_one_image_fingerprint="one-image-scientific-fingerprint",
        )


def test_confirmation_rejects_pilot_cache_before_destination_status(
    tmp_path: Path,
) -> None:
    pilot_dir = tmp_path / "pilot"
    _write_parent_multiscale_run(pilot_dir)
    parent = multiscale.verify_parent_multiscale_run(
        pilot_dir,
        source_image=SOURCE_IMAGE,
        upstream_fingerprint="zero-residual-fingerprint",
        parent_one_image_fingerprint="one-image-scientific-fingerprint",
    )

    with pytest.raises(ArtifactCompatibilityError, match="cannot reuse the pilot"):
        multiscale._prevalidate_external_cache_root(
            pilot_dir,
            study_profile="confirmation",
            cache_semantic_fingerprint="confirmation-cache",
            parent_multiscale=parent,
        )


def test_cache_semantics_bind_profile_parent_and_independence() -> None:
    base = {
        "target_contract": multiscale.TARGET_CONTRACT,
        "kernel": {"grid_size": 28},
        "cache": {"paths": 128},
        "source_image": dict(SOURCE_IMAGE),
        "upstream_zero_residual_fingerprint": "upstream",
        "parent_one_image_fingerprint": "one-image",
        "upstream_zero_residual_provenance": {"config_fingerprint": "upstream"},
        "parent_one_image_provenance": {"scientific_fingerprint": "one-image"},
    }
    pilot = {
        **base,
        "study_profile": {"name": "pilot", "version": 1},
        "parent_multiscale_fingerprint": None,
        "parent_multiscale_provenance": None,
        "independence_provenance": {"required": 0, "passed": 1},
    }
    confirmation = {
        **base,
        "study_profile": {"name": "confirmation", "version": 1},
        "parent_multiscale_fingerprint": "pilot-fingerprint",
        "parent_multiscale_provenance": {"decision_sha256": "pilot-decision"},
        "independence_provenance": {"required": 1, "passed": 1},
    }

    assert multiscale.config_fingerprint(
        multiscale._cache_semantic_payload(pilot)
    ) != multiscale.config_fingerprint(
        multiscale._cache_semantic_payload(confirmation)
    )


def test_checkpoint_validation_predicts_selection_only_and_audits_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(multiscale, "DirectFluxUNet", _TinyFluxModel)
    original_predict = multiscale._predict_task
    calls: list[tuple[int, ...] | None] = []

    def tracked_predict(*args: Any, **kwargs: Any) -> np.ndarray:
        rows = kwargs.get("row_indices")
        calls.append(
            None
            if rows is None
            else tuple(np.asarray(rows, dtype=np.int64).reshape(-1).tolist())
        )
        return original_predict(*args, **kwargs)

    monkeypatch.setattr(multiscale, "_predict_task", tracked_predict)
    arrays = _task_fixture()
    result, debug, _ = _train_tiny_task(
        tmp_path / "selection-isolation",
        arrays=arrays,
        stride=1,
    )

    selection_rows = tuple(np.flatnonzero(arrays.path_ids == 1).tolist())
    # Raw and EMA checkpoint validation both use only selection paths.  The
    # selected EMA is evaluated over all rows exactly once after training, which
    # is the only call that can reach audit paths.
    assert calls.count(None) == 1
    assert calls[-1] is None
    assert calls[:-1]
    assert all(rows == selection_rows for rows in calls[:-1])
    assert len(debug["validation_records"]) == 3  # step 0, 1, and 2
    assert any(row["split"] == "audit" for row in result["split_metrics"])


def test_audit_target_cannot_change_checkpoint_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(multiscale, "DirectFluxUNet", _TinyFluxModel)
    ordinary, ordinary_debug, _ = _train_tiny_task(
        tmp_path / "ordinary-audit",
        arrays=_task_fixture(),
        stride=16,
    )
    changed, changed_debug, _ = _train_tiny_task(
        tmp_path / "changed-audit",
        arrays=_task_fixture(flip_audit_target=True),
        stride=16,
    )

    assert ordinary_debug["validation_records"] == changed_debug["validation_records"]
    assert ordinary_debug["best"] == changed_debug["best"]
    ordinary_audit = next(
        row
        for row in ordinary["split_metrics"]
        if row["split"] == "audit" and row["bin_index"] == -1
    )
    changed_audit = next(
        row
        for row in changed["split_metrics"]
        if row["split"] == "audit" and row["bin_index"] == -1
    )
    assert ordinary_audit["primary_mse"] != changed_audit["primary_mse"]


def test_same_training_seed_resets_model_batches_and_ema_across_strides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(multiscale, "DirectFluxUNet", _TinyFluxModel)
    arrays = _task_fixture()
    first, first_debug, first_fingerprints = _train_tiny_task(
        tmp_path / "stride-one",
        arrays=arrays,
        stride=1,
        seed=123,
    )
    second, second_debug, second_fingerprints = _train_tiny_task(
        tmp_path / "stride-sixteen",
        arrays=arrays,
        stride=16,
        seed=123,
    )

    assert first_debug["validation_records"] == second_debug["validation_records"]
    assert first_debug["best"] == second_debug["best"]
    first_payload = multiscale._load_task_checkpoint(
        Path(first["checkpoint_path"]),
        map_location="cpu",
        fingerprints=first_fingerprints,
    )
    second_payload = multiscale._load_task_checkpoint(
        Path(second["checkpoint_path"]),
        map_location="cpu",
        fingerprints=second_fingerprints,
    )
    assert first_payload["history"] == second_payload["history"]
    for key in first_payload["model_state_dict"]:
        assert torch.equal(
            first_payload["model_state_dict"][key],
            second_payload["model_state_dict"][key],
        )
    for key in first_payload["ema_state_dict"]:
        assert torch.equal(
            first_payload["ema_state_dict"][key],
            second_payload["ema_state_dict"][key],
        )


def test_multiscale_cli_has_no_sampler_import() -> None:
    source_path = Path(multiscale.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imported_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported_names.extend(f"{module}.{alias.name}" for alias in node.names)

    assert not [name for name in imported_names if "sampler" in name.lower()]


def test_shard_diagnostics_merge_additive_counters_before_forming_fractions() -> None:
    shards = [
        SimpleNamespace(
            diagnostics={
                # Deliberately inconsistent cached percentages: the merge must
                # ignore them and use the additive numerators/denominators.
                "raw_limited_fraction": 0.99,
                "mobility_weighted_limited_fraction": 0.98,
                "noise_energy_weighted_limited_fraction": 0.97,
                "masked_edges": 5,
                "proposed_edges": 10,
                "mobility_weight_sum": 2.0,
                "limited_mobility_weight_sum": 1.0,
                "noise_energy_sum": 4.0,
                "limited_noise_energy_sum": 2.0,
                "floor_correction_l1": 1.5,
                "renorm_correction_l1": 2.5,
                "floor_touched_pixels": 3,
                "floor_proposed_pixels": 40,
                "nonfinite_edges": 1,
                "path_substep_count": 8,
                "builder_seed": 11,
                "prefix_aggregation": 1,
            }
        ),
        SimpleNamespace(
            diagnostics={
                "raw_limited_fraction": 0.01,
                "mobility_weighted_limited_fraction": 0.02,
                "noise_energy_weighted_limited_fraction": 0.03,
                "masked_edges": 0,
                "proposed_edges": 990,
                "mobility_weight_sum": 98.0,
                "limited_mobility_weight_sum": 0.0,
                "noise_energy_sum": 196.0,
                "limited_noise_energy_sum": 0.0,
                "floor_correction_l1": 3.5,
                "renorm_correction_l1": 4.5,
                "floor_touched_pixels": 7,
                "floor_proposed_pixels": 960,
                "nonfinite_edges": 2,
                "path_substep_count": 72,
                "builder_seed": 22,
                "prefix_aggregation": 1,
            }
        ),
    ]

    merged = multiscale._merge_cache_diagnostics(shards)

    assert merged["raw_limited_fraction"] == pytest.approx(5.0 / 1000.0)
    assert merged["mobility_weighted_limited_fraction"] == pytest.approx(1.0 / 100.0)
    assert merged["noise_energy_weighted_limited_fraction"] == pytest.approx(2.0 / 200.0)
    assert merged["masked_edges"] == 5
    assert merged["proposed_edges"] == 1000
    assert merged["floor_correction_l1"] == pytest.approx(5.0)
    assert merged["renorm_correction_l1"] == pytest.approx(7.0)
    assert merged["floor_touched_pixels"] == 10
    assert merged["floor_proposed_pixels"] == 1000
    assert merged["nonfinite_edges"] == 3
    assert merged["path_substep_count"] == 80
    assert merged["builder_seeds"] == [11, 22]
    assert merged["prefix_aggregation"] == 1
    assert merged["shard_count"] == 2


def test_shard_diagnostic_merge_keeps_missing_additive_counters_fail_closed() -> None:
    merged = multiscale._merge_cache_diagnostics(
        [SimpleNamespace(diagnostics={"builder_seed": 1, "prefix_aggregation": 1})]
    )

    for name in (
        "raw_limited_fraction",
        "mobility_weighted_limited_fraction",
        "noise_energy_weighted_limited_fraction",
        "masked_edges",
        "proposed_edges",
        "floor_touched_pixels",
        "path_substep_count",
    ):
        assert np.isnan(float(merged[name])), name


def test_physical_task_rows_keep_path_major_anchor_minor_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = 2
    anchors = 3
    grid_size = 4
    pixels = grid_size * grid_size
    later_states = torch.arange(paths * anchors * pixels, dtype=torch.float32).reshape(
        paths, anchors, pixels
    )
    tau = torch.as_tensor(
        [[0.90, 0.80, 0.70], [0.60, 0.50, 0.40]], dtype=torch.float32
    )
    expected_targets = torch.stack(
        [torch.full((2, grid_size, grid_size), float(row + 1)) for row in range(paths * anchors)]
    )
    cache = SimpleNamespace(
        anchors_per_path=anchors,
        grid_size=grid_size,
        later_states=later_states,
        tau=tau,
        horizon=1.0,
        labels=torch.as_tensor([3, 8], dtype=torch.long),
        path_ids=torch.as_tensor([41, 7], dtype=torch.long),
    )
    dynamics = DirectFluxMNISTConfig(
        grid_size=grid_size,
        source_lowfreq_size=grid_size,
        ot_lowres_size=grid_size,
    )

    def fake_targets(
        received_cache: Any,
        received_dynamics: DirectFluxMNISTConfig,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert received_cache is cache
        assert received_dynamics is dynamics
        assert kwargs == {
            "stride": 16,
            "scale": 2.5,
            "device": torch.device("cpu"),
            "batch_size": 4,
        }
        return expected_targets.clone()

    monkeypatch.setattr(multiscale, "block_residual_targets", fake_targets)
    arrays = multiscale.make_physical_task(
        cache,
        dynamics,
        stride=16,
        scale=2.5,
        device=torch.device("cpu"),
        batch_size=4,
    )

    torch.testing.assert_close(arrays.states, later_states.reshape(-1, pixels))
    torch.testing.assert_close(arrays.tau, tau.reshape(-1))
    torch.testing.assert_close(arrays.tau_fraction, tau.reshape(-1))
    torch.testing.assert_close(arrays.labels, torch.as_tensor([3, 3, 3, 8, 8, 8]))
    torch.testing.assert_close(arrays.targets, expected_targets)
    np.testing.assert_array_equal(arrays.path_ids, np.asarray([41, 41, 41, 7, 7, 7]))


def _write_completed_task_fixture(
    task_dir: Path,
    *,
    fingerprints: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    task_dir.mkdir()
    checkpoint = task_dir / "best_ema.pt"
    checkpoint.write_bytes(b"selected-checkpoint")
    status = {"status": "complete", "fingerprints": dict(fingerprints)}
    result = {
        "task_complete": 1,
        "fingerprints": dict(fingerprints),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": multiscale.file_fingerprint(checkpoint),
        "selected_step": 250,
    }
    _write_json(task_dir / "task_status.json", status)
    _write_json(task_dir / "task_result.json", result)
    return checkpoint, status, result


@pytest.mark.parametrize(
    ("corruption", "error"),
    [
        ("status-fingerprint", "fingerprints differ"),
        ("result-fingerprint", "fingerprints differ"),
        ("checkpoint-hash", "checkpoint hash mismatch"),
    ],
)
def test_completed_task_rejects_fingerprint_or_checkpoint_replacement(
    tmp_path: Path,
    corruption: str,
    error: str,
) -> None:
    fingerprints = {"scientific": "frozen", "training_seed": 260723}
    checkpoint, status, result = _write_completed_task_fixture(
        tmp_path / "task", fingerprints=fingerprints
    )
    loaded = multiscale.load_completed_task_result(
        tmp_path / "task", fingerprints=fingerprints
    )
    assert loaded is not None
    assert loaded["selected_step"] == 250

    if corruption == "status-fingerprint":
        status["fingerprints"] = {**fingerprints, "scientific": "changed"}
        _write_json(tmp_path / "task" / "task_status.json", status)
    elif corruption == "result-fingerprint":
        result["fingerprints"] = {**fingerprints, "scientific": "changed"}
        _write_json(tmp_path / "task" / "task_result.json", result)
    else:
        checkpoint.write_bytes(b"replacement-checkpoint")

    with pytest.raises(ArtifactCompatibilityError, match=error):
        multiscale.load_completed_task_result(
            tmp_path / "task", fingerprints=fingerprints
        )


def test_required_preflight_failure_writes_evidence_before_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = multiscale.parse_args(
        _base_cli(tmp_path)
        + [
            "--stage",
            "cache-preflight",
            "--require-gate",
            "cache",
            "--device",
            "cpu",
            "--no-progress",
        ]
    )
    run_dir = tmp_path / "run"
    manifest = {"schema": multiscale.RUN_SCHEMA, "artifacts": {}}
    gate_report = {
        "required_gate": "cache",
        "required_gate_pass": 0,
        "decision": {"decision": "fail"},
    }

    monkeypatch.setattr(multiscale, "configure_exact_torch_backend", lambda _device: {})
    monkeypatch.setattr(multiscale, "_make_dynamics", lambda _args: SimpleNamespace())
    monkeypatch.setattr(
        multiscale,
        "_load_dataset",
        lambda _args: (np.zeros((1, 1), dtype=np.float32), np.asarray([3])),
    )
    monkeypatch.setattr(
        multiscale,
        "_select_source_image",
        lambda *args, **kwargs: dict(SOURCE_IMAGE),
    )
    monkeypatch.setattr(
        multiscale,
        "verify_zero_residual_run",
        lambda *args, **kwargs: {"config_fingerprint": "upstream"},
    )
    monkeypatch.setattr(
        multiscale,
        "verify_parent_one_image_run",
        lambda *args, **kwargs: {"scientific_fingerprint": "parent"},
    )
    monkeypatch.setattr(
        multiscale,
        "_scientific_payload",
        lambda *args, **kwargs: {"scope": "unit-test"},
    )
    monkeypatch.setattr(
        multiscale,
        "_cache_semantic_payload",
        lambda scientific: {"scientific": dict(scientific)},
    )
    monkeypatch.setattr(multiscale, "_source_fingerprint", lambda: ("source", []))

    def make_run_dir(_args: Any) -> tuple[Path, bool]:
        run_dir.mkdir()
        return run_dir, False

    monkeypatch.setattr(multiscale, "_make_run_dir", make_run_dir)
    monkeypatch.setattr(
        multiscale,
        "_initial_manifest",
        lambda **kwargs: copy.deepcopy(manifest),
    )
    monkeypatch.setattr(
        multiscale,
        "_load_or_create_manifest",
        lambda _run_dir, **kwargs: copy.deepcopy(kwargs["candidate"]),
    )

    def fake_preflight(**kwargs: Any) -> tuple[SimpleNamespace, dict[str, Any]]:
        report = {
            "passed": 0,
            "checks": {"forced_failure": {"passed": False}},
            "sampling_performed": 0,
        }
        _write_json(kwargs["run_dir"] / "cache_preflight.json", report)
        return SimpleNamespace(), report

    monkeypatch.setattr(multiscale, "_preflight_cache", fake_preflight)
    monkeypatch.setattr(multiscale, "_validate_cache_source", lambda *args: None)
    monkeypatch.setattr(
        multiscale,
        "evaluate_multiscale_gates",
        lambda **kwargs: copy.deepcopy(gate_report),
    )

    def fake_report_writer(root: Path, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        path = root / "multiscale_gate.json"
        _write_json(path, gate_report)
        return {"multiscale_gate": str(path.resolve())}

    monkeypatch.setattr(multiscale, "_write_report_artifacts", fake_report_writer)
    real_finish = multiscale._finish_run

    def finish_after_evidence(**kwargs: Any) -> int:
        assert json.loads((run_dir / "cache_preflight.json").read_text(encoding="utf-8"))["passed"] == 0
        assert json.loads((run_dir / "multiscale_gate.json").read_text(encoding="utf-8"))["required_gate_pass"] == 0
        return real_finish(**kwargs)

    monkeypatch.setattr(multiscale, "_finish_run", finish_after_evidence)

    assert multiscale._run(args) == 2
    finalized_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert Path(finalized_manifest["artifacts"]["cache_preflight"]).is_file()
    assert Path(finalized_manifest["artifacts"]["multiscale_gate"]).is_file()
    assert status["status"] == "complete"
    assert status["outcome"] == "gate_failed"
    assert status["required_gate_pass"] == 0
    assert finalized_manifest["decision_summary"]["sampling_authorized"] == 0
    assert status["confirmation_exhausted"] == 0
    assert status["sampling_authorized"] == 0


def test_rejected_incompatible_resume_does_not_mutate_existing_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "existing-run"
    run_dir.mkdir()
    existing_manifest = {
        "schema": multiscale.RUN_SCHEMA,
        "schema_version": multiscale.RUN_SCHEMA_VERSION,
        "scientific_fingerprint": "old-scientific",
        "cache_semantic_fingerprint": "old-cache",
        "source_fingerprint": "old-source",
        "runtime_fingerprint": "old-runtime",
    }
    original_status = {
        "schema": multiscale.RUN_SCHEMA,
        "schema_version": multiscale.RUN_SCHEMA_VERSION,
        "status": "running",
        "phase": "physical-tasks",
        "attempt_count": 3,
    }
    _write_json(run_dir / "run_manifest.json", existing_manifest)
    _write_json(run_dir / "run_status.json", original_status)

    args = multiscale.parse_args(
        _base_cli(tmp_path)
        + [
            "--resume-run-dir",
            str(run_dir),
            "--device",
            "cpu",
            "--no-progress",
        ]
    )
    monkeypatch.setattr(multiscale, "configure_exact_torch_backend", lambda _device: {})
    monkeypatch.setattr(multiscale, "_make_dynamics", lambda _args: SimpleNamespace())
    monkeypatch.setattr(
        multiscale,
        "_load_dataset",
        lambda _args: (np.zeros((1, 1), dtype=np.float32), np.asarray([3])),
    )
    monkeypatch.setattr(
        multiscale,
        "_select_source_image",
        lambda *args, **kwargs: dict(SOURCE_IMAGE),
    )
    monkeypatch.setattr(
        multiscale,
        "verify_zero_residual_run",
        lambda *args, **kwargs: {"config_fingerprint": "upstream"},
    )
    monkeypatch.setattr(
        multiscale,
        "verify_parent_one_image_run",
        lambda *args, **kwargs: {"scientific_fingerprint": "parent"},
    )
    monkeypatch.setattr(
        multiscale,
        "_scientific_payload",
        lambda *args, **kwargs: {"scope": "changed"},
    )
    monkeypatch.setattr(
        multiscale,
        "_cache_semantic_payload",
        lambda scientific: {"scientific": dict(scientific)},
    )
    monkeypatch.setattr(multiscale, "_source_fingerprint", lambda: ("new-source", []))
    monkeypatch.setattr(
        multiscale,
        "_initial_manifest",
        lambda **kwargs: {
            "schema": multiscale.RUN_SCHEMA,
            "schema_version": multiscale.RUN_SCHEMA_VERSION,
            "scientific_fingerprint": "new-scientific",
            "cache_semantic_fingerprint": "new-cache",
            "source_fingerprint": "new-source",
            "runtime_fingerprint": "new-runtime",
        },
    )

    with pytest.raises(ArtifactCompatibilityError, match="resume run fingerprint mismatch"):
        multiscale._run(args)
    assert json.loads(
        (run_dir / "run_status.json").read_text(encoding="utf-8")
    ) == original_status


def test_tiny_cpu_all_stage_and_report_resume_never_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = _base_cli(tmp_path) + [
        "--runs-root",
        str(tmp_path / "runs"),
        "--run-name",
        "tiny-cpu",
        "--stage",
        "all",
        "--require-gate",
        "none",
        "--device",
        "cpu",
        "--synthetic-data",
        "--synthetic-examples-per-class",
        "1",
        "--grid-size",
        "4",
        "--sample-steps",
        "10",
        "--reference-substeps",
        "4",
        "--temporal-strides",
        "1,2,4",
        "--cache-paths",
        "6",
        "--anchors-per-path",
        "5",
        "--anchor-bin-counts",
        "1,1,1,1,1",
        "--train-paths",
        "2",
        "--selection-paths",
        "2",
        "--audit-paths",
        "2",
        "--preflight-paths",
        "2",
        "--cache-shard-paths",
        "2",
        "--training-seeds",
        "11,12,13",
        "--teacher-seed",
        "14",
        "--train-steps",
        "1",
        "--teacher-steps",
        "1",
        "--base-channels",
        "4",
        "--batch-size",
        "2",
        "--validation-batch-size",
        "4",
        "--validation-every",
        "1",
        "--checkpoint-every",
        "1",
        "--bootstrap-reps",
        "10",
        "--teacher-min-gain",
        "0.1",
        "--max-raw-intervention",
        "1",
        "--max-weighted-intervention",
        "1",
        "--max-floor-correction-l1",
        "1",
        "--max-renorm-correction-l1",
        "1",
        "--no-amp",
        "--no-progress",
    ]

    monkeypatch.setattr(
        multiscale,
        "verify_zero_residual_run",
        lambda *args, **kwargs: {
            "config_fingerprint": "tiny-upstream",
            "aggregate_sha256": "aggregate",
            "run_config_sha256": "config",
            "run_status_sha256": "status",
        },
    )
    monkeypatch.setattr(
        multiscale,
        "verify_parent_one_image_run",
        lambda *args, **kwargs: {
            "scientific_fingerprint": "tiny-parent",
            "manifest_sha256": "manifest",
            "gate_sha256": "gate",
            "status_sha256": "status",
            "selection_sha256": "selection",
            "cache_fingerprint": "cache",
            "selected_step": 1,
            "selected_prediction_gain": -0.1,
            "selected_data_end_gain": -0.1,
        },
    )

    args = multiscale.parse_args(argv)
    assert multiscale._run(args) == 0
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    decision = json.loads(
        (run_dir / "learnability_decision.json").read_text(encoding="utf-8")
    )
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert decision["sampling_performed"] == 0
    assert status["sampling_performed"] == 0
    assert not list(run_dir.glob("**/*sample*.npz"))
    assert (run_dir / "cache" / "cache_index.json").is_file()
    assert (run_dir / "teacher_control.json").is_file()
    assert len(list((run_dir / "tasks").glob("stride-*/seed-*/task_result.json"))) == 9
    assert (run_dir / "gain_vs_stride.png").is_file()
    assert (run_dir / "learning_curves.png").is_file()

    cache_artifacts = [
        run_dir / "cache" / "cache_index.json",
        *sorted((run_dir / "cache").glob("shard-*.npz")),
        run_dir / "cache_preflight" / "shard-00000.npz",
    ]
    frozen_hashes = {
        path: multiscale.file_fingerprint(path) for path in cache_artifacts
    }

    def forbid_report_rollout(**kwargs: Any) -> Any:
        del kwargs
        raise AssertionError("report stage must not rebuild a forward cache")

    report_args = multiscale.parse_args(argv)
    report_args.stage = "report"
    report_args.resume_run_dir = run_dir
    report_args.runs_root = tmp_path / "unused"
    with monkeypatch.context() as report_patch:
        report_patch.setattr(
            multiscale,
            "build_multiscale_cache_shard",
            forbid_report_rollout,
        )
        assert multiscale._run(report_args) == 0
    resumed = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert resumed["attempt_count"] == 2
    assert resumed["sampling_performed"] == 0
    assert {
        path: multiscale.file_fingerprint(path) for path in cache_artifacts
    } == frozen_hashes

    # A self-consistent replacement shard must be rejected against the frozen
    # index, and report-only mode must not repair or re-bless it.
    replaced_path = sorted((run_dir / "cache").glob("shard-*.npz"))[0]
    replaced_cache = multiscale.load_multiscale_cache_shard(replaced_path)
    replacement = replace(
        replaced_cache,
        diagnostics={**dict(replaced_cache.diagnostics), "tampered_fixture": 1},
    )
    multiscale.save_multiscale_cache_shard(
        replaced_path,
        replacement,
        shard_id=0,
        metadata={"scope": "self-consistent-test-replacement"},
    )
    replacement_hash = multiscale.file_fingerprint(replaced_path)
    with monkeypatch.context() as report_patch:
        report_patch.setattr(
            multiscale,
            "build_multiscale_cache_shard",
            forbid_report_rollout,
        )
        with pytest.raises(multiscale.D0MultiscaleCompatibilityError):
            multiscale._run(report_args)
    assert multiscale.file_fingerprint(replaced_path) == replacement_hash


def test_dataset_seed_and_source_operation_order_match_frozen_parent_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mnist import diag_d0_one_image_overfit as parent_one_image

    base_images = np.stack(
        (
            np.arange(1, 17, dtype=np.float64),
            2.0 * np.arange(16, 0, -1, dtype=np.float64),
            np.linspace(0.25, 4.0, 16, dtype=np.float64),
            np.full(16, 3.0, dtype=np.float64),
        )
    )
    base_labels = np.asarray([3, 3, 8, 1], dtype=np.int64)
    observed_seeds: list[int] = []

    def seeded_dataset(*args: Any, **kwargs: Any) -> SimpleNamespace:
        del args
        seed = int(kwargs["seed"])
        observed_seeds.append(seed)
        order = np.random.default_rng(seed).permutation(base_labels.size)
        return SimpleNamespace(
            train_images=base_images[order],
            train_labels=base_labels[order],
        )

    monkeypatch.setattr(multiscale, "load_mnist_measure_dataset", seeded_dataset)
    args = multiscale.parse_args(
        _base_cli(tmp_path)
        + [
            "--require-gate",
            "none",
            "--grid-size",
            "4",
        ]
    )
    assert args.dataset_seed == 260718
    assert args.cache_seed == 260721
    images, labels = multiscale._load_dataset(args)
    assert observed_seeds == [260718]

    selection_kwargs = {
        "label": 3,
        "class_index": 0,
        "grid_size": 4,
        "lambda_mix": 0.35,
    }
    parent = parent_one_image._select_source_image(
        images, labels, **selection_kwargs
    )
    current = multiscale._select_source_image(images, labels, **selection_kwargs)

    assert current["dataset_index"] == parent["dataset_index"] == 0
    assert current["image_sha256"] == parent["image_sha256"]
    assert current["mixed_target_sha256"] == parent["mixed_target_sha256"]
    assert current["image_sha256"] == (
        "996d4b52415d7c70cb2df5a8228678b6409d5887d086090f050ef97244b25ffe"
    )
    assert current["mixed_target_sha256"] == (
        "4d74a47ee7e5431dbf0718bdfebbe67bc8aa35435858c70d5b88cafe1c219f7b"
    )
    np.testing.assert_array_equal(current["image"], parent["image"])
    np.testing.assert_array_equal(current["mixed_target"], parent["mixed_target"])

    # Normalizing the source first is a different operation when the constructed
    # input is not already a probability measure. The frozen hash therefore
    # protects the parent's mix-then-clamp-then-normalize order.
    source_input = np.asarray(images[current["dataset_index"]], dtype=np.float64)
    normalized_first = np.maximum(source_input, 0.0)
    normalized_first /= normalized_first.sum()
    normalized_first = 0.65 * normalized_first + 0.35 / normalized_first.size
    normalized_first /= normalized_first.sum()
    assert multiscale._array_digest(normalized_first) != current["mixed_target_sha256"]


@pytest.mark.parametrize(
    ("extra_args", "error"),
    [
        (
            ["--require-gate", "none", "--temporal-strides", "1,3"],
            "every temporal stride must divide",
        ),
        (
            ["--require-gate", "none", "--training-seeds", "7,7,9"],
            "training seeds must be distinct",
        ),
    ],
)
def test_parser_rejects_nondividing_strides_and_duplicate_training_seeds(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
    error: str,
) -> None:
    with pytest.raises(SystemExit):
        multiscale.parse_args(_base_cli(tmp_path) + extra_args)

    assert error in capsys.readouterr().err


def test_pilot_confirmation_comparison_is_advisory_and_never_pooled(
    tmp_path: Path,
) -> None:
    def gate(overall: float, data_end: float, tau: float, lcb: float) -> dict[str, Any]:
        return {
            "passed": 0,
            "diagnostics": {
                "median_audit_overall_gain": overall,
                "median_audit_data_end_gain": data_end,
                "median_audit_gain_vs_tau_baseline": tau,
                "median_audit_target_prediction_covariance": 0.25,
            },
            "bootstrap": {
                "overall_vs_zero": {"lower_bound": lcb},
                "data_end_vs_zero": {"lower_bound": lcb + 0.01},
            },
            "subchecks": {"passing_seed_count": {"value": 2}},
        }

    parent_decision = tmp_path / "pilot_decision.json"
    _write_json(
        parent_decision,
        {"strides": {"1": gate(-0.02, -0.01, -0.03, -0.04)}},
    )
    artifacts = multiscale._pilot_confirmation_comparison(
        tmp_path,
        stride_gates={1: gate(0.03, 0.04, 0.02, 0.01)},
        parent_multiscale={"decision_path": str(parent_decision)},
    )

    comparison_path = Path(artifacts["pilot_confirmation_comparison"])
    plot_path = Path(artifacts["pilot_confirmation_plot"])
    with comparison_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["evidence_pooled"] == "0"
    assert rows[0]["gate_use"] == "advisory-only"
    assert float(rows[0]["pilot_median_audit_overall_gain"]) == -0.02
    assert float(rows[0]["confirmation_median_audit_overall_gain"]) == 0.03
    assert plot_path.is_file()


def test_documented_multiscale_commands_use_only_live_parser_flags() -> None:
    document = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "experiment12_d0_patch_plan.md"
    ).read_text(encoding="utf-8")
    powershell_blocks = [
        chunk.split("```", 1)[0]
        for chunk in document.split("```powershell")[1:]
    ]
    commands = [
        block
        for block in powershell_blocks
        if "-m mnist.diag_d0_multiscale_learnability" in block
    ]
    assert len(commands) == 4

    parsed_stages: set[str] = set()
    parsed_profiles: set[str] = set()
    confirmation_commands = 0
    for command in commands:
        argv: list[str] = []
        documented_flags: set[str] = set()
        for raw_line in command.splitlines():
            line = raw_line.strip().removesuffix("`").strip()
            if not line.startswith("--"):
                continue
            pieces = line.split(maxsplit=1)
            documented_flags.add(pieces[0])
            argv.extend(pieces)
        assert "--dataset-seed" in documented_flags
        parsed = multiscale.parse_args(argv)
        parsed_stages.add(str(parsed.stage))
        parsed_profiles.add(str(parsed.study_profile))
        if str(parsed.study_profile) == "confirmation":
            confirmation_commands += 1
            assert "--parent-multiscale-run-dir" in documented_flags

    assert parsed_stages == {"cache-preflight", "all"}
    assert parsed_profiles == {"pilot", "confirmation"}
    assert confirmation_commands == 2
