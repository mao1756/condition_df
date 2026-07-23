from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import mnist.diag_d0_score_control_scale_repair as repair_cli
from mnist.d0_one_image_gate import (
    ArtifactCompatibilityError,
    array_fingerprint,
    atomic_write_json,
)
from mnist.d0_score_boundary_control_gate import (
    BoundaryControlThresholds,
    evaluate_supervised_teacher,
)
from mnist.d0_score_control_scale_repair_gate import (
    ProbeBankStatus,
    ScaleRepairDecision,
    decide_scale_repair,
    not_evaluated_study,
)
from mnist.diag_d0_score_boundary_controls import (
    ControlArrays,
    _arrays_identity,
    _build_control_arrays,
    _train_task,
)
from mnist.diag_d0_score_control_scale_repair import (
    _evaluate_saved_controls,
    _load_or_calibrate,
    _pending_preflight_report,
    _run_controls,
    _verify_terminal_registry,
    main,
    parse_args,
)
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, natural_horizon


def _dynamics(grid_size: int = 4, sample_steps: int = 8) -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=grid_size,
        alpha_eff=1.0,
        edge_alpha_mode="alpha_eff",
        num_steps=sample_steps,
        mass_floor=1e-7,
        limiter_fraction=1.0,
        source_lowfreq_size=min(4, grid_size),
        ot_lowres_size=min(4, grid_size),
    )


def _arrays(role: str, law: str, first_path: int, seed: int) -> ControlArrays:
    dynamics = _dynamics()
    return _build_control_arrays(
        role=role,
        law=law,
        path_count=2,
        first_path_id=first_path,
        bin_counts=(1, 1, 1, 1, 1),
        horizon=float(natural_horizon(dynamics)),
        grid_size=4,
        seed=seed,
    )


def _task_args() -> SimpleNamespace:
    return SimpleNamespace(
        base_channels=4,
        batch_size=2,
        validation_batch_size=2,
        train_steps=1,
        validation_every=1,
        checkpoint_every=1,
        learning_rate=1e-4,
        weight_decay=1e-4,
        ema_decay=0.99,
        grad_clip=1.0,
        clip_warmup_steps=0,
        training_probes=1,
        selection_probes=1,
        audit_probes=1,
        bootstrap_reps=16,
        bootstrap_confidence=0.90,
        training_probe_seed=31,
        calibration_seed=29,
        selection_probe_a_seed=37,
        selection_probe_b_seed=41,
        audit_probe_a_seed=43,
        audit_probe_b_seed=47,
        bootstrap_seed=53,
        batch_index_seed=59,
        supervised_seed=61,
        teacher_seeds=(67, 71, 73),
        null_seeds=(67, 71, 73),
        supervised_initial_grad_target=0.10,
        implicit_initial_grad_target=0.10,
        require_gate="none",
        no_progress=True,
    )


def _passing_supervised_metrics() -> dict[str, Any]:
    return {
        "complete": 1,
        "finite": 1,
        "selected_step": 1,
        "audit_overall_score_gain": 0.99,
        "audit_data_end_score_gain": 0.99,
        "overall_flux_cosine": 0.999,
        "time_bin_flux_cosines": [0.999] * 5,
        "overall_relative_flux_l2": 0.04,
        "time_bin_relative_flux_l2": [0.04] * 5,
        "boundary_admissible": 1,
        "post_warmup_clip_fraction": 0.0,
    }


def _complete_probe_banks() -> dict[str, Any]:
    return {
        bank: {
            scope: {"lower_bound": 0.1}
            for scope in ("overall", "data_end")
        }
        for bank in ("a", "b")
    }


def _calibration_record(
    *,
    scale: float,
    target: float = 0.10,
    count: int = 10,
    objective_kind: str = "supervised_teacher",
    args: SimpleNamespace | None = None,
) -> dict[str, Any]:
    raw = target / scale
    binding: dict[str, Any] = {"fixture": "training-only"}
    if args is not None:
        binding = {
            "scientific_fingerprint": "science",
            "runtime_fingerprint": "runtime",
            "source_fingerprint": "source",
            "objective_kind": objective_kind,
            "calibration_seed": int(args.calibration_seed),
            "model_initialization_seed": repair_cli.boundary._derived_seed(
                int(args.calibration_seed), objective_kind, "model"
            ),
            "training_probe_seed": int(args.training_probe_seed),
            "target_initial_gradient_norm": target,
            "calibration_state_count": count,
        }
    return {
        "complete": 1,
        "finite": 1,
        "training_only": 1,
        "calibration_split": "train",
        "calibration_state_count": count,
        "unscaled_initial_gradient_norm": raw,
        "scaled_initial_gradient_norm": target,
        "target_initial_gradient_norm": target,
        "loss_scale": scale,
        "objective_kind": objective_kind,
        "calibration_state_sha256": "fixture-state-sha256",
        "binding": binding,
        "sampling_performed": 0,
    }


def _install_tiny_preflight_fakes(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    production_args = parse_args(
        ["--parent-boundary-control-run-dir", "parent", "--device", "cpu"]
    )
    parent = {
        "passed": 1,
        "artifacts": {"status": {"sha256": "a" * 64}},
        "scientific_fingerprint": "parent-science",
        "schedule_metadata": {
            "horizon": float(
                natural_horizon(repair_cli.boundary._make_dynamics(production_args))
            )
        },
    }

    def fake_preflight(
        run_dir: Path, *, binding: dict[str, Any], **_: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        artifact = {
            "binding": binding,
            "gate_metrics": {"aggregate_preflight_pass": 1},
        }
        gate = {
            "gate": "boundary_preflight",
            "passed": 1,
            "sampling_performed": 0,
        }
        atomic_write_json(run_dir / "boundary_operator_preflight.json", artifact)
        atomic_write_json(run_dir / "boundary_preflight_gate.json", gate)
        return artifact, gate

    monkeypatch.setattr(repair_cli, "verify_parent_boundary_control_run", lambda _: parent)
    monkeypatch.setattr(repair_cli.boundary, "_run_preflight", fake_preflight)
    monkeypatch.setattr(repair_cli.boundary, "_write_report_artifacts", lambda _: None)
    monkeypatch.setattr(
        repair_cli.boundary,
        "configure_exact_torch_backend",
        lambda _: {"deterministic": 1, "fixture": "cpu"},
    )
    return parent


def _run_tiny_preflight(runs_root: Path) -> tuple[int, Path]:
    code = main(
        [
            "--runs-root",
            str(runs_root),
            "--run-name",
            "tiny-preflight",
            "--device",
            "cpu",
            "--stage",
            "preflight",
            "--parent-boundary-control-run-dir",
            "parent",
            "--require-gate",
            "preflight",
            "--no-progress",
        ]
    )
    return code, next(runs_root.iterdir())


def _resume_tiny(run_dir: Path, stage: str) -> int:
    return main(
        [
            "--resume-run-dir",
            str(run_dir),
            "--device",
            "cpu",
            "--stage",
            stage,
            "--parent-boundary-control-run-dir",
            "parent",
            "--require-gate",
            "preflight",
            "--no-progress",
        ]
    )


def test_parser_uses_fresh_paired_defaults_and_required_gate_locks_profile() -> None:
    args = parse_args(
        [
            "--parent-boundary-control-run-dir",
            "parent",
            "--require-gate",
            "controls",
        ]
    )
    assert args.supervised_initial_grad_target == pytest.approx(0.10)
    assert args.implicit_initial_grad_target == pytest.approx(0.10)
    assert args.teacher_data_seed == 260781
    assert args.teacher_seeds == (260785, 260786, 260787)
    assert args.null_seeds == args.teacher_seeds
    assert args.anchor_bin_counts == (4, 4, 4, 4, 16)

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--parent-boundary-control-run-dir",
                "parent",
                "--require-gate",
                "controls",
                "--supervised-initial-grad-target",
                "0.2",
            ]
        )
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--parent-boundary-control-run-dir",
                "parent",
                "--teacher-seeds",
                "1,2,3",
                "--null-seeds",
                "1,2,4",
            ]
        )


def test_supervised_calibration_artifact_is_training_only_exact_and_frozen(
    tmp_path: Path,
) -> None:
    train = _arrays("train", "bounded_teacher", 100, 11)
    selection = _arrays("selection", "bounded_teacher", 200, 12)
    args = _task_args()
    target = float(args.supervised_initial_grad_target)
    binding = {
        "objective_kind": "supervised_teacher",
        "teacher_train_identity": _arrays_identity(train),
        "target_initial_gradient_norm": target,
    }
    path = tmp_path / "supervised_loss_scale_calibration.json"

    record, gate = _load_or_calibrate(
        path,
        objective_kind="supervised_teacher",
        arrays=train,
        dynamics=_dynamics(),
        args=args,
        device=torch.device("cpu"),
        binding=binding,
        target=target,
    )
    assert gate["passed"] == 1
    assert record["training_only"] == 1
    assert record["calibration_split"] == "train"
    assert record["calibration_state_count"] == len(train.path_ids)
    assert record["calibration_state_sha256"] == array_fingerprint(
        np.asarray(train.states, dtype=np.float32)
    )
    assert record["loss_scale"] == pytest.approx(
        min(1.0, target / record["unscaled_initial_gradient_norm"])
    )
    assert "selection" not in json.dumps(record["binding"]).lower()

    repeated, repeated_gate = _load_or_calibrate(
        path,
        objective_kind="supervised_teacher",
        arrays=train,
        dynamics=_dynamics(),
        args=args,
        device=torch.device("cpu"),
        binding=binding,
        target=target,
    )
    assert repeated == record
    assert repeated_gate == gate

    leaked_binding = dict(binding)
    leaked_binding["teacher_train_identity"] = _arrays_identity(selection)
    with pytest.raises(ArtifactCompatibilityError, match="binding mismatch"):
        _load_or_calibrate(
            path,
            objective_kind="supervised_teacher",
            arrays=selection,
            dynamics=_dynamics(),
            args=args,
            device=torch.device("cpu"),
            binding=leaked_binding,
            target=target,
        )


def test_one_step_supervised_training_records_scaled_optimizer_units(tmp_path: Path) -> None:
    train = _arrays("train", "bounded_teacher", 100, 17)
    selection = _arrays("selection", "bounded_teacher", 200, 19)
    args = _task_args()
    loss_scale = 0.125
    _, summary = _train_task(
        task_dir=tmp_path / "supervised",
        task_kind="supervised_teacher",
        train=train,
        selection_arrays=selection,
        dynamics=_dynamics(),
        device=torch.device("cpu"),
        args=args,
        model_seed=int(args.supervised_seed),
        loss_scale=loss_scale,
        fingerprints={"task": "scaled-supervised-fixture", "loss_scale": loss_scale},
        show_progress=False,
    )
    with (tmp_path / "supervised" / "training_history.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        row = next(csv.DictReader(handle))

    unscaled_loss = float(row["unscaled_loss"])
    scaled_loss = float(row["scaled_loss"])
    raw_norm = float(row["raw_gradient_norm"])
    scaled_norm = float(row["scaled_preclip_gradient_norm"])
    assert float(row["loss_scale"]) == pytest.approx(loss_scale)
    assert scaled_loss == pytest.approx(unscaled_loss * loss_scale, rel=1e-6)
    assert scaled_norm == pytest.approx(raw_norm * loss_scale, rel=1e-6)
    assert float(row["grad_norm"]) == pytest.approx(scaled_norm)
    diagnostics = summary["optimization_diagnostics"]
    assert diagnostics["gradient_norm_source"] == "scaled_preclip_gradient_norm"
    assert diagnostics["recorded_clip_flags_consistent"] == 1


def test_control_orchestration_routes_supervised_and_shared_implicit_scales(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays = {
        "teacher_train": _arrays("train", "bounded_teacher", 100, 21),
        "teacher_selection": _arrays("selection", "bounded_teacher", 200, 22),
        "teacher_audit": _arrays("audit", "bounded_teacher", 300, 23),
        "null_train": _arrays("train", "dirichlet_null", 400, 24),
        "null_selection": _arrays("selection", "dirichlet_null", 500, 25),
        "null_audit": _arrays("audit", "dirichlet_null", 600, 26),
    }
    atomic_write_json(tmp_path / "synthetic_array_registry.json", {"records": {}})
    args = _task_args()
    calls: list[tuple[str, int, float]] = []

    def fake_calibrate(
        path: Path, *, objective_kind: str, target: float, **_: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        scale = 0.02 if objective_kind == "supervised_teacher" else 0.03
        record = _calibration_record(scale=scale, target=target)
        atomic_write_json(path, record)
        return record, {"gate": "loss_scale_calibration", "passed": 1}

    supervised_metrics = _passing_supervised_metrics()
    supervised_gate = evaluate_supervised_teacher(supervised_metrics)

    def fake_task(*, task_kind: str, model_seed: int, loss_scale: float, **_: Any) -> dict[str, Any]:
        calls.append((task_kind, model_seed, loss_scale))
        if task_kind == "supervised_teacher":
            return {"task_kind": task_kind, "metrics": supervised_metrics, "gate": supervised_gate}
        return {
            "task_kind": task_kind,
            "metrics": {
                "complete": 1,
                "finite": 1,
                "selected_step": 1 if task_kind == "implicit_teacher" else 0,
                "post_warmup_clip_fraction": 0.0,
            },
            "gate": {"passed": 1},
        }

    monkeypatch.setattr(repair_cli, "_load_or_calibrate", fake_calibrate)
    monkeypatch.setattr(repair_cli.boundary, "_run_control_task", fake_task)
    monkeypatch.setattr(
        repair_cli,
        "evaluate_implicit_teacher_study",
        lambda *_: {"gate": "implicit_teacher_study", "passed": 1},
    )
    monkeypatch.setattr(
        repair_cli,
        "evaluate_null_study",
        lambda *_: {"gate": "null_study", "passed": 1},
    )
    monkeypatch.setattr(
        repair_cli,
        "_evaluate_saved_controls",
        lambda *_, **__: {"required_gate": "none", "required_gate_pass": 1},
    )

    _run_controls(
        tmp_path,
        arrays=arrays,
        dynamics=_dynamics(),
        args=args,
        device=torch.device("cpu"),
        thresholds=BoundaryControlThresholds(),
        scientific_fingerprint="science",
        runtime_fingerprint="runtime",
        source_fingerprint_value="source",
        provenance={"passed": 1},
        preflight_gate={"passed": 1},
    )

    assert calls[0] == ("supervised_teacher", args.supervised_seed, 0.02)
    assert calls[1:4] == [
        ("implicit_teacher", seed, 0.03) for seed in args.teacher_seeds
    ]
    assert calls[4:] == [("null", seed, 0.03) for seed in args.null_seeds]


def test_saved_skips_remain_not_evaluated_and_probe_status_is_not_vacuous(
    tmp_path: Path,
) -> None:
    args = _task_args()
    atomic_write_json(
        tmp_path / "run_manifest.json",
        {
            "scientific_fingerprint": "science",
            "runtime_fingerprint": "runtime",
            "source_fingerprint": "source",
            "scientific_config": {
                "synthetic_data": {"train_paths": 8, "anchors_per_path": 32}
            },
        },
    )
    atomic_write_json(
        tmp_path / "supervised_loss_scale_calibration.json",
        _calibration_record(
            scale=0.02,
            count=repair_cli.CALIBRATION_STATE_COUNT,
            objective_kind="supervised_teacher",
            args=args,
        ),
    )
    atomic_write_json(
        tmp_path / "implicit_loss_scale_calibration.json",
        _calibration_record(
            scale=0.03,
            count=repair_cli.CALIBRATION_STATE_COUNT,
            objective_kind="implicit_teacher",
            args=args,
        ),
    )
    atomic_write_json(
        tmp_path / "supervised_teacher_control.json",
        repair_cli.boundary._failed_task_result(
            "supervised_teacher",
            int(args.supervised_seed),
            RuntimeError("supervised prerequisite failed"),
        ),
    )
    atomic_write_json(
        tmp_path / "implicit_teacher_study.json",
        not_evaluated_study("implicit_teacher_study", "supervised prerequisite failed"),
    )
    atomic_write_json(
        tmp_path / "null_study.json",
        not_evaluated_study("null_study", "supervised prerequisite failed"),
    )

    report = _evaluate_saved_controls(
        tmp_path,
        provenance={"passed": 1},
        preflight_gate={"passed": 1},
        thresholds=BoundaryControlThresholds(),
        args=args,
    )
    assert report["controls"]["probe_bank_status"] == "not_evaluated"
    assert report["controls"]["components"]["downstream_optimizer"]["evaluation_status"] == "not_evaluated"
    assert report["decision"]["probe_bank_status"] == "not_evaluated"
    assert report["decision"]["studies_evaluated"] == 0
    assert report["decision"]["physical_training_authorized"] == 0

    pending = _pending_preflight_report(
        provenance={"passed": 1},
        preflight_gate={"passed": 1},
        require_gate="preflight",
    )
    assert pending["controls"]["evaluation_status"] == "not_evaluated"
    assert pending["decision"]["probe_bank_status"] == "not_evaluated"


@pytest.mark.parametrize("defect", ["missing_bank", "missing_scope"])
def test_attempted_studies_with_incomplete_probe_banks_are_not_evaluated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    args = _task_args()
    atomic_write_json(
        tmp_path / "run_manifest.json",
        {
            "scientific_fingerprint": "science",
            "runtime_fingerprint": "runtime",
            "source_fingerprint": "source",
            "scientific_config": {
                "synthetic_data": {
                    "train_paths": 128,
                    "anchors_per_path": 32,
                }
            },
        },
    )
    for filename, scale, kind in (
        ("supervised_loss_scale_calibration.json", 0.02, "supervised_teacher"),
        ("implicit_loss_scale_calibration.json", 0.03, "implicit_teacher"),
    ):
        atomic_write_json(
            tmp_path / filename,
            _calibration_record(
                scale=scale,
                count=repair_cli.CALIBRATION_STATE_COUNT,
                objective_kind=kind,
                args=args,
            ),
        )
    supervised_metrics = _passing_supervised_metrics()
    atomic_write_json(
        tmp_path / "supervised_teacher_control.json",
        {
            "task_kind": "supervised_teacher",
            "model_seed": int(args.supervised_seed),
            "metrics": supervised_metrics,
            "gate": evaluate_supervised_teacher(supervised_metrics),
        },
    )

    def result(task_kind: str, seed: int, banks: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_kind": task_kind,
            "model_seed": int(seed),
            "metrics": {
                "complete": 1,
                "finite": 1,
                "post_warmup_clip_fraction": 0.0,
                "audit_objective_banks": banks,
            },
            "gate": {"passed": 1},
        }

    teacher_results = [
        result("implicit_teacher", seed, _complete_probe_banks())
        for seed in args.teacher_seeds
    ]
    null_results = [
        result("null", seed, _complete_probe_banks()) for seed in args.null_seeds
    ]
    defective = teacher_results[0]["metrics"]["audit_objective_banks"]
    if defect == "missing_bank":
        defective.pop("b")
    else:
        defective["b"].pop("data_end")
    atomic_write_json(
        tmp_path / "implicit_teacher_study.json",
        {
            "gate": "implicit_teacher_study",
            "passed": 1,
            "evaluation_status": "evaluated",
            "task_results": teacher_results,
        },
    )
    atomic_write_json(
        tmp_path / "null_study.json",
        {
            "gate": "null_study",
            "passed": 1,
            "evaluation_status": "evaluated",
            "task_results": null_results,
        },
    )

    monkeypatch.setattr(
        repair_cli, "_verify_saved_task_result", lambda _run_dir, *, result, **_: dict(result)
    )
    monkeypatch.setattr(
        repair_cli,
        "evaluate_implicit_teacher_study",
        lambda *_: {"gate": "implicit_teacher_study", "passed": 1},
    )
    monkeypatch.setattr(
        repair_cli,
        "evaluate_null_study",
        lambda *_: {"gate": "null_study", "passed": 1},
    )
    monkeypatch.setattr(
        repair_cli.boundary,
        "_probe_banks_agree",
        lambda **_: pytest.fail("incomplete probe banks must not be compared"),
    )

    report = _evaluate_saved_controls(
        tmp_path,
        provenance={"passed": 1},
        preflight_gate={"passed": 1},
        thresholds=BoundaryControlThresholds(),
        args=args,
    )
    assert report["controls"]["probe_bank_status"] == "not_evaluated"
    assert report["decision"]["probe_bank_status"] == "not_evaluated"
    assert (
        report["decision"]["decision"]
        == ScaleRepairDecision.IMPLICIT_OBJECTIVE_UNSTABLE.value
    )
    assert report["decision"]["physical_training_authorized"] == 0


@pytest.mark.parametrize("attempt", ["incomplete", "nonfinite"])
def test_attempted_downstream_task_failure_is_optimizer_scale_invalid(
    attempt: str,
) -> None:
    healthy = {
        "metrics": {
            "complete": 1,
            "finite": 1,
            "post_warmup_clip_fraction": 0.0,
        }
    }
    if attempt == "incomplete":
        results = [
            healthy,
            {
                "metrics": {
                    "complete": 0,
                    "finite": 0,
                    "post_warmup_clip_fraction": None,
                }
            },
        ]
    else:
        results = [
            healthy,
            {
                "metrics": {
                    "complete": 1,
                    "finite": 0,
                    "post_warmup_clip_fraction": 0.0,
                }
            },
        ]
    downstream = repair_cli._aggregate_optimizer_health(
        results,
        BoundaryControlThresholds(),
        expected_count=2,
    )
    assert downstream["evaluation_status"] == "evaluated"
    assert downstream["passed"] == 0

    decision = decide_scale_repair(
        provenance_pass=True,
        boundary_preflight=True,
        supervised_calibration=True,
        implicit_calibration=True,
        supervised_optimizer=True,
        supervised_representation=True,
        downstream_optimizer=downstream,
        implicit_teacher_study=not_evaluated_study(
            "implicit_teacher_study", "attempt did not complete"
        ),
        null_study=not_evaluated_study("null_study", "attempt did not complete"),
        probe_bank_status=ProbeBankStatus.NOT_EVALUATED,
    )
    assert decision["decision"] == ScaleRepairDecision.OPTIMIZER_SCALE_INVALID.value


def test_tiny_cpu_preflight_integration_writes_a_verifiable_terminal_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs"
    _install_tiny_preflight_fakes(monkeypatch)
    code, run_dir = _run_tiny_preflight(runs_root)
    assert code == 0
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["required_gate_pass"] == 1
    assert status["decision"] == "controls_not_run"
    assert status["probe_bank_status"] == "not_evaluated"
    assert status["sampling_performed"] == 0
    _verify_terminal_registry(run_dir)


def test_report_resume_of_terminal_preflight_run_preserves_pending_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_tiny_preflight_fakes(monkeypatch)
    code, run_dir = _run_tiny_preflight(tmp_path / "runs")
    assert code == 0

    report_code = _resume_tiny(run_dir, "report")
    assert report_code == 0
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    report = json.loads(
        (run_dir / "boundary_control_report.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "complete"
    assert status["phase"] == "report"
    assert status["stage"] == "report"
    assert status["attempt_count"] == 2
    assert status["required_gate_pass"] == 1
    assert status["decision"] == "controls_not_run"
    assert status["probe_bank_status"] == "not_evaluated"
    assert report["required_gate"] == "preflight"
    assert report["required_gate_pass"] == 1
    assert report["controls"]["evaluation_status"] == "not_evaluated"
    assert report["decision"]["decision"] == "controls_not_run"
    _verify_terminal_registry(run_dir)


@pytest.mark.parametrize("stage", ["preflight", "controls", "report", "all"])
def test_every_terminal_resume_rejects_registered_artifact_tampering_before_status_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    _install_tiny_preflight_fakes(monkeypatch)
    code, run_dir = _run_tiny_preflight(tmp_path / stage / "runs")
    assert code == 0
    status_path = run_dir / "run_status.json"
    registry_path = run_dir / "artifact_registry.json"
    status_before = status_path.read_bytes()
    registry_before = registry_path.read_bytes()
    files_before = {path.relative_to(run_dir) for path in run_dir.rglob("*") if path.is_file()}

    registered = run_dir / "boundary_operator_preflight.json"
    registered.write_text(
        registered.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    assert _resume_tiny(run_dir, stage) == 2

    assert status_path.read_bytes() == status_before
    assert registry_path.read_bytes() == registry_before
    assert {path.relative_to(run_dir) for path in run_dir.rglob("*") if path.is_file()} == files_before


def test_provenance_failure_commits_terminal_evidence_before_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs"

    def reject(_: Path) -> dict[str, Any]:
        raise ArtifactCompatibilityError("parent evidence was tampered")

    monkeypatch.setattr(repair_cli, "verify_parent_boundary_control_run", reject)
    monkeypatch.setattr(repair_cli.boundary, "_write_report_artifacts", lambda _: None)
    monkeypatch.setattr(
        repair_cli.boundary,
        "configure_exact_torch_backend",
        lambda _: {"deterministic": 1, "fixture": "cpu"},
    )
    code = main(
        [
            "--runs-root",
            str(runs_root),
            "--run-name",
            "bad-parent",
            "--device",
            "cpu",
            "--stage",
            "preflight",
            "--parent-boundary-control-run-dir",
            "bad-parent",
            "--require-gate",
            "preflight",
            "--no-progress",
        ]
    )
    assert code == 2
    run_dir = next(runs_root.iterdir())
    for name in (
        "failure.json",
        "boundary_control_gate.json",
        "control_repair_decision.json",
        "boundary_control_report.json",
        "artifact_registry.json",
        "run_status.json",
    ):
        assert (run_dir / name).is_file(), name
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["outcome"] == "implementation_error"
    assert status["decision"] == "control_provenance_invalid"
    assert status["required_gate_pass"] == 0
    assert status["sampling_performed"] == 0
    _verify_terminal_registry(run_dir)


def test_scale_repair_cli_does_not_import_a_sampler() -> None:
    source_path = Path(__file__).parents[1] / "mnist" / "diag_d0_score_control_scale_repair.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("sampler" in name.lower() for name in imported)
