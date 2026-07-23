from __future__ import annotations

import ast
from pathlib import Path

import pytest

import mnist.diag_d0_score_density_ratio_h1_gradient_control_confirmation as cli
from mnist.d0_score_density_ratio_h1_gradient_control_gate import (
    H1GradientControlThresholds,
    evaluate_gradient_control_workflow,
    not_evaluated_gate,
)


def _panel_record(values: list[float], *, teacher: bool = False) -> dict[str, object]:
    path_ids = list(range(len(values)))
    value: dict[str, object] = {
        "finite": 1,
        "overall": {
            "bce": 0.68,
            "bootstrap": {"path_ids": path_ids, "path_values": values},
        },
        "data_end": {
            "bce": 0.68,
            "bootstrap": {
                "path_ids": path_ids,
                "path_values": [0.5 * item for item in values],
            },
        },
    }
    if teacher:
        value["analytic"] = {
            "audit_overall_score_gain": 0.95,
            "audit_data_end_score_gain": 0.95,
            "overall_flux_cosine": 0.99,
            "overall_relative_flux_l2": 0.10,
            "time_bin_flux_cosines": [0.97] * 5,
            "time_bin_relative_flux_l2": [0.15] * 5,
        }
    return value


def _task_result(task: str, *, ratio: float) -> dict[str, object]:
    panel_a = _panel_record([0.02, 0.01, 0.02, 0.01], teacher=task == "bounded_teacher")
    panel_b = _panel_record(
        [0.02, 0.01, 0.02, 0.01]
        if task == "bounded_teacher"
        else [-0.02, -0.01, 0.0, -0.01],
        teacher=task == "bounded_teacher",
    )
    metrics: dict[str, object] = {
        "evaluation_status": "evaluated",
        "complete": 1,
        "finite": 1,
        "boundary_admissible": 1,
        "optimizer_health_pass": 1,
        "controller_health_pass": 1,
        "controller_active_fraction": 1.0 if ratio else 0.0,
        "maximum_ratio_relative_error": 0.0,
        "post_ramp_h1_floor_hit_count": 0,
        "nonfinite_coefficient_count": 0,
        "fixed_endpoint_step": 4_000,
        "nominee_step": 4_000,
        "selected_step": 4_000 if task == "bounded_teacher" else 0,
        "selection": {
            "confirmation": {
                "accepted": int(task == "bounded_teacher"),
                "panel_b_lower_bounds": [0.01, 0.01]
                if task == "bounded_teacher"
                else [-0.01, -0.01],
                "panel_b_overall_bce": 0.68,
            }
        },
        "checkpoints": [
            {"step": 4_000, "finite": 1, "panels": {"a": panel_a, "b": panel_b}}
        ],
        "post_warmup_clip_fraction": 0.0,
        "final_500_clip_fraction": 0.0,
        "final_200_clip_fraction": 0.0,
    }
    return {"task": task, "model_seed": 1, "metrics": metrics}


def test_production_defaults_and_required_ratio_grid_are_frozen() -> None:
    args = cli.parse_args(
        [
            "--parent-h1-run-dir",
            "parent",
            "--stage",
            "preflight",
            "--require-gate",
            "preflight",
        ]
    )
    assert args.root_seed == 261041
    assert args.pilot_model_seed == 261051
    assert args.confirm_model_seeds == (261061, 261062, 261063)
    assert args.target_gradient_ratios == (0.0, 0.1, 0.3, 1.0)
    assert args.pilot_steps == args.confirm_steps == 4_000
    assert args.pilot_paths == args.confirm_paths == 128
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--parent-h1-run-dir",
                "parent",
                "--stage",
                "preflight",
                "--require-gate",
                "preflight",
                "--target-gradient-ratios",
                "0,0.2",
            ]
        )


def test_cli_bound_sources_are_additive_and_import_no_sampler() -> None:
    tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("sampler" in value.lower() for value in imported)
    _, paths = cli._source_record()
    names = {Path(value).name for value in paths}
    assert {
        "diag_d0_score_density_ratio_h1_gradient_control_confirmation.py",
        "d0_score_density_ratio_h1_gradient_control.py",
        "d0_score_density_ratio_h1_gradient_control_task.py",
        "d0_score_density_ratio_h1_gradient_control_gate.py",
        "d0_score_density_ratio_h1_gradient_control_provenance.py",
        "d0_score_density_ratio_matched_flux.py",
    }.issubset(names)
    assert not any("sampler" in value.lower() for value in names)


def test_candidate_summary_exposes_fixed_endpoint_controller_health() -> None:
    args = cli.parse_args(["--parent-h1-run-dir", "parent", "--stage", "all"])
    row = cli._candidate_summary(
        multiplier=0.3,
        teacher=_task_result("bounded_teacher", ratio=0.3),
        null=_task_result("dirichlet_null", ratio=0.3),
        args=args,
        expose_panel_b=False,
    )
    assert row["optimizer_health_pass"] == 1
    assert row["controller_health_pass"] == 1
    assert row["fixed_endpoint_step"] == 4_000
    assert row["panel_b_evaluation_count"] == 0
    assert row["teacher"]["panels"]["b"]["evaluation_status"] == "not_evaluated"


def test_pilot_null_family_has_exact_four_ratio_by_two_scope_family() -> None:
    args = cli.parse_args(
        [
            "--parent-h1-run-dir", "parent", "--stage", "all",
            "--simultaneous-bootstrap-reps", "100",
        ]
    )
    results = [
        (ratio, _task_result("dirichlet_null", ratio=ratio))
        for ratio in (0.0, 0.1, 0.3, 1.0)
    ]
    record, gate = cli._null_family(results, args=args, phase="fixture")
    assert record["family_size"] == 8
    assert gate["passed"] == 1
    assert {row["panel_role"] for row in record["members"]} == {"b"}


def test_workflow_artifacts_fail_closed_before_confirmation(tmp_path: Path) -> None:
    pending = not_evaluated_gate("confirmation", "not run")
    report = evaluate_gradient_control_workflow(
        provenance={"passed": 1},
        controller_preflight={"passed": 1},
        preflight={"passed": 1},
        pilot_panel_power={"passed": 1},
        pilot={"evaluation_status": "evaluated", "passed": 1},
        confirmation_panel_power=pending,
        confirmation=pending,
        require_gate="controls",
        thresholds=H1GradientControlThresholds(),
    )
    cli._save_report(tmp_path, report)
    assert report["required_gate_pass"] == 0
    assert (tmp_path / "h1_gradient_control_gate.json").is_file()
    assert (tmp_path / "h1_gradient_control_decision.json").is_file()


def test_artifact_registry_excludes_terminal_self_reference(tmp_path: Path) -> None:
    (tmp_path / "evidence.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_status.json").write_text("{}", encoding="utf-8")
    (tmp_path / "artifact_registry.json").write_text("{}", encoding="utf-8")
    registry = cli._artifact_registry(tmp_path)
    assert set(registry["records"]) == {"evidence.json"}

