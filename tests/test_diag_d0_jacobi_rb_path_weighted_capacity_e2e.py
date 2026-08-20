from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mnist.diag_d0_jacobi_rb_path_weighted_capacity_e2e import (
    ExperimentConfig,
    _controller_metric,
    _parser,
    _scientific_interpretation,
)


def test_cli_defaults_define_full_unattended_experiment() -> None:
    args = _parser().parse_args(["run", "--run-dir", "/tmp/example"])
    assert args.train_paths == 64
    assert args.validation_paths == 32
    assert args.small_updates == args.large_updates == 12_000
    assert args.batch_size == 32
    assert args.mobility_floor == 1e-4
    assert args.include_old_control is True
    assert args.hard_wall_seconds == 21_600


def test_metric_is_exact_at_target_and_detects_noise() -> None:
    target = np.arange(1, 785, dtype=np.float64)
    target /= target.sum()
    exact = _controller_metric(target, target)
    assert exact["squared_l2"] == 0.0
    assert exact["l1"] == 0.0
    shifted = np.roll(target, 1)
    changed = _controller_metric(shifted, target)
    assert changed["squared_l2"] > 0.0
    assert changed["mass_error"] < 1e-15


def test_interpretation_prioritizes_oracle_composition_failure() -> None:
    def row(value: float) -> dict[str, object]:
        return {"to_mixed_source": {"squared_l2": value}}

    same = {
        "metrics": {
            "zero": row(1.0),
            "small-old": row(0.9),
            "small-weighted": row(0.8),
            "large-weighted": row(0.7),
            "oracle": row(1.1),
        }
    }
    prior = {"metrics": dict(same["metrics"])}
    result = _scientific_interpretation(same, prior, True)
    assert result["automatic_interpretation"] == "reverse-composition-or-oracle-control-failure"


def test_runpod_launcher_requires_durable_storage_before_delete() -> None:
    root = Path(__file__).resolve().parents[1] / "tools" / "runpod_weighted_e2e"
    launcher = (root / "launch.sh").read_text()
    worker = (root / "worker.sh").read_text()
    finalizer = (root / "finalize.sh").read_text()
    lifecycle = (root / "pod_lifecycle.sh").read_text()
    assert "RUNPOD_RESULTS_DURABLE" in launcher
    assert "Refusing unattended Pod deletion" in launcher
    assert "timeout --signal=TERM" in worker
    assert "/stop" in lifecycle
    assert "--request DELETE" in lifecycle
    assert "EXPORT_VERIFIED" in finalizer


def test_config_is_json_serializable() -> None:
    config = ExperimentConfig(
        run_dir="/tmp/run",
        device="cuda:0",
        data_dir="/tmp/data",
        mnist_index=0,
        download_mnist=True,
        train_paths=64,
        validation_paths=32,
        small_updates=12_000,
        large_updates=12_000,
        batch_size=32,
        validation_interval=250,
        mobility_floor=1e-4,
        include_old_control=True,
        hard_wall_seconds=21_600,
        seed=1,
    )
    json.dumps(config.to_record(), sort_keys=True)
