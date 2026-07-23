from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest
import torch

import mnist.diag_d0_score_density_ratio_selection_power_confirmation as cli
from mnist.d0_one_image_gate import atomic_write_json
from mnist.d0_score_density_ratio_head_gate import HeadCoordinateThresholds
from mnist.d0_score_density_ratio_selection_power_gate import SelectionPowerThresholds


torch.set_num_threads(1)


def test_production_defaults_and_required_gate_overrides_are_frozen() -> None:
    args = cli.parse_args(
        [
            "--parent-normalized-head-run-dir",
            "parent",
            "--stage",
            "preflight",
            "--require-gate",
            "preflight",
        ]
    )
    assert args.root_seed == 260931
    assert args.oracle_calibration_paths == 256
    assert args.oracle_half_paths == 128
    assert args.pilot_selection_paths == 128
    assert args.confirm_selection_paths == 128
    assert args.confirm_audit_paths == 128
    assert args.pilot_learning_rates == (3e-5, 1e-5)
    assert args.accumulation_levels == (8,)
    assert args.confirm_model_seeds == (260941, 260942, 260943)
    assert args.loss_scale == pytest.approx(0.05173607018770852, abs=0.0)

    for override in (
        ("--oracle-calibration-paths", "128", "--oracle-half-paths", "64"),
        ("--pilot-selection-paths", "64"),
        ("--base-channels", "16"),
        ("--pilot-learning-rates", "1e-5"),
        ("--loss-scale", "0.1"),
    ):
        with pytest.raises(SystemExit):
            cli.parse_args(
                [
                    "--parent-normalized-head-run-dir",
                    "parent",
                    "--require-gate",
                    "preflight",
                    *override,
                ]
            )


def test_cli_and_bound_sources_have_no_sampler_import() -> None:
    tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("sampler" in value.lower() for value in imported)

    _, source_paths = cli._source_record()
    names = {Path(value).name for value in source_paths}
    assert {
        "diag_d0_score_density_ratio_selection_power_confirmation.py",
        "d0_score_density_ratio_selection_power.py",
        "d0_score_density_ratio_selection_power_gate.py",
        "d0_score_density_ratio_selection_power_provenance.py",
        "diag_d0_score_density_ratio_head_confirmation.py",
    }.issubset(names)
    assert not any("sampler" in name.lower() for name in names)


def test_oracle_failure_precedes_every_optimizer_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle = {
        "gate": "oracle_qualified_ab_panel_set",
        "evaluation_status": "evaluated",
        "passed": 0,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    monkeypatch.setattr(
        cli,
        "_pilot_panels",
        lambda *args, **kwargs: (
            {"bounded_teacher": {}, "dirichlet_null": {}},
            oracle,
        ),
    )

    def forbidden(*args, **kwargs):  # pragma: no cover - assertion path
        raise AssertionError("optimizer was entered before the oracle panel gate")

    monkeypatch.setattr(cli.head, "run_paired_density_ratio_task", forbidden)
    args = argparse.Namespace(
        bootstrap_confidence=0.9,
        bootstrap_reps=10_000,
        root_seed=260931,
    )
    pilot, selected, multiplicity = cli._run_pilot(
        tmp_path,
        args=args,
        manifest={"scientific_fingerprint": "fixture"},
        dynamics=object(),
        device=torch.device("cpu"),
        stream_plan=object(),
        paired_stream_plan=object(),
        calibration_panel=object(),
        thresholds=SelectionPowerThresholds(head=HeadCoordinateThresholds()),
    )
    assert pilot["evaluation_status"] == "not_evaluated"
    assert selected == {}
    assert multiplicity["status"] == "not_evaluated"
    assert not (tmp_path / "pilot").exists()
    assert (tmp_path / "selection_power_pilot_gate.json").is_file()
    assert (tmp_path / "pilot_null_multiplicity_analysis.json").is_file()


def test_preflight_stage_commits_readable_artifacts_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = {
        "passed": 1,
        "preflight_pass": 1,
        "scientific_fingerprint": "parent-science",
        "artifact_registry_sha256": "a" * 64,
        "horizon": 0.001,
        "kernel": dict(cli.EXPECTED_KERNEL),
    }
    monkeypatch.setattr(cli, "verify_parent_normalized_head_run", lambda path: parent)
    monkeypatch.setattr(cli, "_source_record", lambda: ("b" * 64, [str(Path(cli.__file__).resolve())]))

    def fake_preflight(run_dir: Path, **kwargs):
        del kwargs
        gate = {
            "gate": "selection_power_preflight",
            "evaluation_status": "evaluated",
            "passed": 1,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
        atomic_write_json(run_dir / "selection_power_preflight_gate.json", gate)
        return gate, object()

    monkeypatch.setattr(cli, "_run_preflight", fake_preflight)
    monkeypatch.setattr(cli, "_write_learning_plot", lambda run_dir: None)
    monkeypatch.setattr(cli, "_write_oracle_csv", lambda run_dir: None)
    monkeypatch.setattr(cli, "natural_horizon", lambda dynamics: 0.001)

    code = cli.main(
        [
            "--parent-normalized-head-run-dir",
            str(tmp_path / "parent"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--run-name",
            "fixture",
            "--device",
            "cpu",
            "--stage",
            "preflight",
            "--require-gate",
            "none",
            "--no-progress",
        ]
    )
    assert code == 0
    run_dir = next((tmp_path / "runs").iterdir())
    status = cli._json_load(run_dir / "run_status.json")
    assert status["status"] == "complete"
    assert status["decision"] == "selection_power_preflight_passed"
    assert status["physical_training_performed"] == 0
    assert status["sampling_performed"] == 0
    assert (run_dir / "selection_power_control_gate.json").is_file()
    assert (run_dir / "selection_power_decision.json").is_file()
    assert (run_dir / "artifact_registry.json").is_file()


def test_frozen_artifact_rejects_regeneration(tmp_path: Path) -> None:
    path = tmp_path / "fixed.json"
    cli._freeze_json(path, {"panel": "a", "passed": 1})
    assert cli._freeze_json(path, {"panel": "a", "passed": 1})["passed"] == 1
    with pytest.raises(Exception, match="frozen artifact changed"):
        cli._freeze_json(path, {"panel": "b", "passed": 1})
