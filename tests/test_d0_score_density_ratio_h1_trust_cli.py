from __future__ import annotations

import ast
from pathlib import Path

import pytest

import mnist.diag_d0_score_density_ratio_h1_trust_confirmation as cli


def _panel_record(values: list[float]) -> dict[str, object]:
    path_ids = list(range(len(values)))
    return {
        "finite": 1,
        "overall": {
            "bce": 0.68,
            "bootstrap": {"path_ids": path_ids, "path_values": values},
        },
        "data_end": {
            "bce": 0.68,
            "bootstrap": {
                "path_ids": path_ids,
                "path_values": [0.5 * value for value in values],
            },
        },
    }


def _task_result(task: str, *, positive_null_margin: bool = False) -> dict[str, object]:
    panel_b = _panel_record(
        [0.02, -0.01, -0.01, 0.0]
        if positive_null_margin
        else [-0.02, -0.01, 0.0, -0.01]
    )
    analytic = {
        "audit_overall_score_gain": 0.95,
        "audit_data_end_score_gain": 0.95,
        "overall_flux_cosine": 0.99,
        "overall_relative_flux_l2": 0.10,
        "time_bin_flux_cosines": [0.97] * 5,
        "time_bin_relative_flux_l2": [0.15] * 5,
    }
    if task == "bounded_teacher":
        panel_b["analytic"] = analytic
    accepted = int(task == "bounded_teacher")
    metrics = {
        "complete": 1,
        "finite": 1,
        "boundary_admissible": 1,
        "h1_health_pass": 1,
        "nominee_step": 25,
        "selected_step": 25 if task == "bounded_teacher" else 0,
        "selection": {
            "confirmation": {
                "accepted": accepted,
                "panel_b_lower_bounds": [0.01, 0.01]
                if task == "bounded_teacher"
                else [0.001, -0.001],
                "panel_b_overall_bce": 0.68,
            }
        },
        "checkpoints": [{"step": 25, "panels": {"b": panel_b}}],
        "post_warmup_clip_fraction": 0.0,
        "final_500_clip_fraction": 0.0,
        "final_200_clip_fraction": 0.0,
        "optimization_diagnostics": {"h1": {"h1_health_pass": 1}},
    }
    if task == "bounded_teacher":
        metrics.update(
            {
                "selected_analytic_metrics": analytic,
                "selection_overall_score_gain": 0.95,
                "selection_data_end_score_gain": 0.95,
                "selection_overall_flux_cosine": 0.99,
                "selection_overall_relative_flux_l2": 0.10,
                "selection_data_end_relative_flux_l2": 0.10,
            }
        )
    return {"task": task, "model_seed": 1, "metrics": metrics}


def test_production_defaults_and_required_overrides_are_frozen() -> None:
    args = cli.parse_args(
        [
            "--parent-multiplicity-run-dir",
            "parent",
            "--stage",
            "preflight",
            "--require-gate",
            "preflight",
        ]
    )
    assert args.root_seed == 261001
    assert args.pilot_model_seed == 261011
    assert args.confirm_model_seeds == (261021, 261022, 261023)
    assert args.h1_multipliers == (0.0, 0.1, 0.3, 1.0)
    assert args.pilot_steps == args.confirm_steps == 4_000
    assert args.pilot_paths == args.confirm_paths == 128

    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--parent-multiplicity-run-dir",
                "parent",
                "--stage",
                "preflight",
                "--require-gate",
                "preflight",
                "--h1-multipliers",
                "0,0.2",
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
    _, paths = cli._source_record()
    names = {Path(value).name for value in paths}
    assert {
        "diag_d0_score_density_ratio_h1_trust_confirmation.py",
        "d0_score_density_ratio_h1_task.py",
        "d0_score_density_ratio_h1_trust.py",
        "d0_score_density_ratio_h1_trust_gate.py",
        "d0_score_density_ratio_h1_trust_provenance.py",
    }.issubset(names)
    assert not any("sampler" in value.lower() for value in names)


def test_candidate_summary_exposes_strict_derivative_and_h1_health() -> None:
    args = cli.parse_args(
        ["--parent-multiplicity-run-dir", "parent", "--stage", "all"]
    )
    row = cli._candidate_summary(
        multiplier=0.3,
        teacher=_task_result("bounded_teacher"),
        null=_task_result("dirichlet_null", positive_null_margin=True),
        args=args,
    )
    assert row["optimizer_health_pass"] == 1
    assert row["h1_health_pass"] == 1
    assert row["teacher_score_gain_overall"] == pytest.approx(0.95)
    assert row["teacher_time_bin_flux_cosines"] == [0.97] * 5
    # A marginal positive null component remains discovery evidence; the
    # simultaneous family, not this row, authorizes the stationary null.
    assert row["null_panel_b_lower_bounds"] == [0.001, -0.001]


def test_pilot_null_family_is_exactly_four_ratios_by_two_scopes() -> None:
    args = cli.parse_args(
        [
            "--parent-multiplicity-run-dir",
            "parent",
            "--stage",
            "all",
            "--simultaneous-bootstrap-reps",
            "100",
        ]
    )
    results = [
        (ratio, _task_result("dirichlet_null"))
        for ratio in (0.0, 0.1, 0.3, 1.0)
    ]
    record, gate = cli._null_family(results, args=args, phase="fixture")
    assert record["family_size"] == 8
    assert gate["passed"] == 1
    assert {row["panel_role"] for row in record["members"]} == {"b"}
    assert {row["resampling_block"] for row in record["members"]} == {
        "fixture-panel-b"
    }


def test_artifact_registry_excludes_only_self_referential_terminal_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "evidence.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_status.json").write_text("{}", encoding="utf-8")
    (tmp_path / "artifact_registry.json").write_text("{}", encoding="utf-8")
    registry = cli._artifact_registry(tmp_path)
    assert set(registry["records"]) == {"evidence.json"}
    assert registry["terminal_files_excluded_to_avoid_self_reference"] == [
        "artifact_registry.json",
        "run_status.json",
    ]
