from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import mnist.diag_d0_score_density_ratio_multiplicity_confirmation as cli
from mnist.d0_score_density_ratio_sealed_null_gate import SealedNullThresholds


torch.set_num_threads(1)


def _panel_record(role: str, values: list[float]) -> dict[str, object]:
    paths = list(range(len(values)))
    return {
        "role": role,
        "finite": 1,
        "overall": {
            "bootstrap": {"path_ids": paths, "path_values": values},
        },
        "data_end": {
            "bootstrap": {
                "path_ids": paths,
                "path_values": [0.5 * value for value in values],
            },
        },
    }


def _null_result(seed: int) -> dict[str, object]:
    a0 = _panel_record("a", [-0.1, 0.0, 0.1])
    a1 = _panel_record("a", [-0.2, 0.0, 0.2])
    b = _panel_record("b", [-0.3, -0.2, -0.1])
    c = _panel_record("c", [-0.4, -0.3, -0.2])
    d = _panel_record("d", [-0.5, -0.4, -0.3])
    return {
        "task": "dirichlet_null",
        "model_seed": seed,
        "metrics": {
            "model_seed": seed,
            "nominee_step": 25,
            "checkpoints": [
                {"step": 0, "panels": {"a": a0}},
                {"step": 25, "panels": {"a": a1, "b": b}},
            ],
            "audit_panels": {"c": c, "d": d},
        },
    }


def test_production_defaults_and_required_overrides_are_frozen() -> None:
    args = cli.parse_args(
        [
            "--parent-selection-power-run-dir",
            "parent",
            "--stage",
            "preflight",
            "--require-gate",
            "preflight",
        ]
    )
    assert args.root_seed == 260961
    assert args.simultaneous_bootstrap_reps == 50_000
    assert args.familywise_confidence == 0.95
    assert args.confirm_selection_paths == 128
    assert args.confirm_audit_paths == 128
    assert args.confirm_model_seeds == (260971, 260972, 260973)
    assert args.loss_scale == pytest.approx(0.05173607018770852, abs=0.0)

    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--parent-selection-power-run-dir",
                "parent",
                "--stage",
                "preflight",
                "--require-gate",
                "preflight",
                "--simultaneous-bootstrap-reps",
                "10000",
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
        "diag_d0_score_density_ratio_multiplicity_confirmation.py",
        "d0_score_density_ratio_sealed_null_gate.py",
        "d0_score_density_ratio_multiplicity_provenance.py",
        "diag_d0_score_density_ratio_head_confirmation.py",
    }.issubset(names)
    assert not any("sampler" in name.lower() for name in names)


def test_confirmation_family_is_exactly_a_advisory_and_b_c_d_authorizing() -> None:
    results = [_null_result(seed) for seed in (260971, 260972, 260973)]
    discovery, confirmatory = cli._confirmation_family_members(results)
    assert len(discovery) == 3 * 1 * 2
    assert len(confirmatory) == 3 * 3 * 2
    assert all("/a/" in name for name in discovery)
    assert not any("/a/" in name for name in confirmatory)
    role_counts = {role: 0 for role in ("b", "c", "d")}
    for value in confirmatory.values():
        role_counts[str(value["panel_role"])] += 1
    assert role_counts == {"b": 6, "c": 6, "d": 6}
    assert {value["resampling_block"] for value in confirmatory.values()} == {
        "confirmation-panel-b",
        "confirmation-panel-c",
        "confirmation-panel-d",
    }


def test_oracle_failure_precedes_every_optimizer_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panels = {
        "bounded_teacher": {name: object() for name in ("a", "b", "c", "d")},
        "dirichlet_null": {name: object() for name in ("a", "b", "c", "d")},
    }
    monkeypatch.setattr(cli, "_profile_binding", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        cli,
        "_prepare_confirmation_panels",
        lambda *args, **kwargs: (panels, {"passed": 1}),
    )
    monkeypatch.setattr(
        cli,
        "_oracle_panel_bundle",
        lambda *args, **kwargs: {
            "gate": "oracle_qualified_abcd_panel_set",
            "evaluation_status": "evaluated",
            "passed": 0,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )

    def forbidden(*args, **kwargs):  # pragma: no cover - assertion path
        raise AssertionError("optimizer entered before oracle qualification")

    monkeypatch.setattr(cli.head, "run_paired_density_ratio_task", forbidden)
    args = cli.parse_args(
        [
            "--parent-selection-power-run-dir",
            "parent",
            "--stage",
            "all",
            "--confirm-steps",
            "1",
        ]
    )
    teacher, null, oracle, null_gate, teacher_gate = cli._run_confirmation(
        tmp_path,
        args=args,
        manifest={"scientific_fingerprint": "fixture"},
        parent={"run_dir": str(tmp_path), "artifact_registry_sha256": "a" * 64},
        profile={},
        dynamics=object(),
        device=torch.device("cpu"),
        stream_plan=object(),
        paired_stream_plan=object(),
        thresholds=SealedNullThresholds(),
    )
    assert teacher == [] and null == []
    assert oracle["passed"] == 0
    assert null_gate["evaluation_status"] == "not_evaluated"
    assert teacher_gate["evaluation_status"] == "not_evaluated"
    assert not (tmp_path / "confirmation").exists()
    assert (tmp_path / "confirmation_oracle_feasibility.json").is_file()


def test_frozen_artifact_rejects_regeneration(tmp_path: Path) -> None:
    path = tmp_path / "fixed.json"
    cli._freeze_json(path, {"panel": "a", "passed": 1})
    assert cli._freeze_json(path, {"panel": "a", "passed": 1})["passed"] == 1
    with pytest.raises(Exception, match="frozen artifact changed"):
        cli._freeze_json(path, {"panel": "b", "passed": 1})
