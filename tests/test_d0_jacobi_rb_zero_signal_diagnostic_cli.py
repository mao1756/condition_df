from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError
import mnist.diag_d0_jacobi_rb_zero_signal_diagnostic as cli


def test_cli_requires_resume_for_analysis() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--stage",
                "analyze",
                "--parent-learnability-run-dir",
                "parent",
            ]
        )
    args = cli.parse_args(
        [
            "--stage",
            "analyze",
            "--resume-run-dir",
            "run",
            "--parent-learnability-run-dir",
            "parent",
        ]
    )
    assert args.require_gate == "none"


def test_scientific_config_is_report_only() -> None:
    args = cli.parse_args(
        [
            "--stage",
            "preflight",
            "--parent-learnability-run-dir",
            "parent",
            "--device",
            "cpu",
        ]
    )
    record = cli._scientific_config(args)
    assert record["bootstrap_replicates"] == 20_000
    assert record["bootstrap_unit"] == "whole_path"
    assert record["new_data_generation_permitted"] == 0
    assert record["training_permitted"] == 0
    assert record["parameter_tuning_permitted"] == 0
    assert record["sampling_permitted"] == 0
    assert record["conditional_mean_identically_zero_proven"] == 0


def test_registry_binds_artifacts_before_terminal_status(tmp_path: Path) -> None:
    (tmp_path / "evidence.json").write_text('{"value":1}\n', encoding="utf-8")
    registry = cli._artifact_registry(tmp_path)
    cli._status(
        tmp_path,
        stage="preflight",
        state="completed",
        decision="zero_signal_diagnostic_complete",
        registry=registry,
    )
    cli._verify_own_registry(tmp_path)
    (tmp_path / "evidence.json").write_text('{"value":2}\n', encoding="utf-8")
    with pytest.raises(ArtifactCompatibilityError, match="changed"):
        cli._verify_own_registry(tmp_path)


def test_replay_tolerance_is_fixed_and_small(tmp_path: Path) -> None:
    parent = tmp_path
    (parent / "confirmation_metrics.json").write_text(
        json.dumps(
            {
                "aggregate_model_mse": 2.0,
                "aggregate_metadata_mse": 3.0,
                "aggregate_zero_mse": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (parent / "confirmation_path_metrics.csv").write_text(
        "path_id,model_mse,metadata_mse,zero_mse\n"
        "7,2.0,3.0,1.0\n",
        encoding="utf-8",
    )
    summary = {"model_mse": 2.0 + 5.0e-7, "metadata_mse": 3.0, "zero_mse": 1.0}
    rows = [{"path_id": 7, **summary}]
    assert cli._replay_reference(
        parent_dir=parent,
        split="confirmation",
        summary=summary,
        path_rows=rows,
    )
    summary["model_mse"] = 2.0 + 2.0e-6
    assert not cli._replay_reference(
        parent_dir=parent,
        split="confirmation",
        summary=summary,
        path_rows=rows,
    )


def test_cli_does_not_import_generation_training_or_sampling_workflows() -> None:
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert "run_exact_multipath_shard" not in source
    assert "train_deterministic_regressor" not in source
    assert "reverse_sampler" not in source
