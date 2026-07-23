from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import mnist.diag_d0_score_control_stability_confirmation as cli
from mnist.d0_one_image_gate import atomic_write_json
from mnist.d0_score_control_stability import build_stream_plan
from mnist.d0_score_control_stability_gate import StabilityThresholds
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, natural_horizon


def _dynamics() -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=4,
        alpha_eff=1.0,
        edge_alpha_mode="alpha_eff",
        num_steps=8,
        mass_floor=1e-7,
        limiter_fraction=1.0,
        source_lowfreq_size=4,
        ot_lowres_size=4,
    )


def _task_args() -> SimpleNamespace:
    return SimpleNamespace(
        base_channels=4,
        weight_decay=1e-4,
        ema_decay=0.99,
        grad_clip=1.0,
        clip_warmup_steps=0,
        train_steps=1,
        validation_steps=(0, 1),
        validation_batch_size=10,
        selection_probes=1,
        audit_probes=1,
        bootstrap_reps=16,
        bootstrap_confidence=0.90,
        selection_probe_a_seed=17,
        selection_probe_b_seed=19,
        audit_probe_a_seed=23,
        audit_probe_b_seed=29,
        bootstrap_seed=31,
    )


def test_parser_locks_production_stream_profile() -> None:
    args = cli.parse_args(
        [
            "--parent-scale-repair-run-dir",
            "parent",
            "--require-gate",
            "controls",
        ]
    )
    assert args.stage == "all"
    assert args.root_seed == 260801
    assert args.pilot_learning_rates == (1e-4, 3e-5, 1e-5, 3e-6)
    assert args.anchor_bin_counts == (4, 4, 4, 4, 16)
    assert args.training_probe_banks == 2
    assert args.training_probes_per_bank == 4
    assert args.confirm_model_seeds == (260811, 260812, 260813)
    assert args.implicit_loss_scale == pytest.approx(cli.FROZEN_IMPLICIT_LOSS_SCALE)

    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--parent-scale-repair-run-dir",
                "parent",
                "--require-gate",
                "pilot",
                "--pilot-learning-rates",
                "1e-5",
            ]
        )
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--parent-scale-repair-run-dir",
                "parent",
                "--training-probe-banks",
                "1",
            ]
        )
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--parent-scale-repair-run-dir",
                "parent",
                "--stage",
                "confirm",
            ]
        )


def test_parser_rejects_stream_dimensions_that_disagree_with_time_strata(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--parent-scale-repair-run-dir",
                "parent",
                "--anchors-per-cluster",
                "16",
                "--batch-size",
                "32",
            ]
        )
    assert "--anchors-per-cluster must equal the sum" in capsys.readouterr().err


def test_scientific_config_records_derived_stream_dimensions() -> None:
    args = cli.parse_args(["--parent-scale-repair-run-dir", "parent"])
    args.anchors_per_cluster = 16
    args.batch_size = 32
    scientific = cli._scientific_config(args, {}, StabilityThresholds())
    assert scientific["stream"]["anchors_per_cluster"] == sum(
        args.anchor_bin_counts
    )
    assert scientific["stream"]["batch_size"] == (
        args.clusters_per_step * sum(args.anchor_bin_counts)
    )


def test_task_fingerprint_binds_stream_and_disjoint_fixed_panels(tmp_path: Path) -> None:
    dynamics = _dynamics()
    args = cli.parse_args(["--parent-scale-repair-run-dir", "parent"])
    args.grid_size = 4
    args.anchor_bin_counts = (1, 1, 1, 1, 1)
    args.anchors_per_path = 5
    first = cli._prepare_fixed_panel(
        tmp_path,
        phase="confirm",
        law="bounded_teacher",
        role="selection",
        path_count=2,
        first_path_id=100,
        seed=11,
        horizon=float(natural_horizon(dynamics)),
        args=args,
        scientific_fingerprint="science",
    )
    second = cli._prepare_fixed_panel(
        tmp_path,
        phase="confirm",
        law="bounded_teacher",
        role="audit",
        path_count=2,
        first_path_id=200,
        seed=13,
        horizon=float(natural_horizon(dynamics)),
        args=args,
        scientific_fingerprint="science",
    )
    assert set(first.path_ids).isdisjoint(set(second.path_ids))
    plan = build_stream_plan(
        root_seed=7,
        grid_size=4,
        horizon=float(natural_horizon(dynamics)),
    )
    fingerprint = cli._task_fingerprints(
        manifest={
            "scientific_fingerprint": "science",
            "runtime_fingerprint": "runtime",
            "source_fingerprint": "source",
        },
        phase="confirm",
        law="bounded_teacher",
        model_seed=17,
        learning_rate=1e-5,
        loss_scale=cli.FROZEN_IMPLICIT_LOSS_SCALE,
        stream_plan=plan,
        selection=first,
        audit=second,
    )
    assert fingerprint["stream_plan_fingerprint"] == plan.fingerprint
    assert fingerprint["selection_identity"] != fingerprint["audit_identity"]
    assert fingerprint["sampling_performed"] == 0


def test_streamed_task_is_exactly_reusable_from_terminal_result(tmp_path: Path) -> None:
    dynamics = _dynamics()
    selection = cli.boundary._build_control_arrays(
        role="selection",
        law="bounded_teacher",
        path_count=2,
        first_path_id=100,
        bin_counts=(1, 1, 1, 1, 1),
        horizon=float(natural_horizon(dynamics)),
        grid_size=4,
        seed=11,
    )
    plan = build_stream_plan(
        root_seed=7,
        grid_size=4,
        horizon=float(natural_horizon(dynamics)),
        probes_per_bank=4,
    )
    fingerprints = {
        "fixture": "streamed-resume",
        "stream_plan_fingerprint": plan.fingerprint,
    }
    kwargs: dict[str, Any] = {
        "task_dir": tmp_path / "task",
        "task_kind": "implicit_teacher",
        "selection_arrays": selection,
        "audit_arrays": None,
        "dynamics": dynamics,
        "args": _task_args(),
        "device": torch.device("cpu"),
        "model_seed": 37,
        "learning_rate": 1e-5,
        "loss_scale": cli.FROZEN_IMPLICIT_LOSS_SCALE,
        "stream_plan": plan,
        "fingerprints": fingerprints,
        "phase": "pilot",
        "show_progress": False,
        "thresholds": cli.BoundaryControlThresholds(
            bootstrap_confidence=0.90,
            expected_implicit_teacher_seeds=1,
            minimum_passing_implicit_teacher_seeds=1,
        ),
    }
    first = cli.run_streamed_control_task(**kwargs)
    repeated = cli.run_streamed_control_task(**kwargs)
    assert repeated == first
    assert first["training_summary"]["training_step"] == 1
    assert first["training_summary"]["stream_plan"]["fingerprint"] == plan.fingerprint
    assert (tmp_path / "task" / "checkpoints" / "step-00000000.pt").is_file()
    assert (tmp_path / "task" / "checkpoints" / "step-00000001.pt").is_file()
    history = json.loads((tmp_path / "task" / "task_result.json").read_text())["training_summary"]
    assert history["sampling_performed"] == 0
    assert not any(key.startswith("audit_") for key in first["metrics"])
    assert "selection_overall_score_gain" in first["metrics"]
    assert (tmp_path / "task" / "selection_time_bin_metrics.csv").is_file()
    assert not (tmp_path / "task" / "audit_time_bin_metrics.csv").exists()
    assert not (tmp_path / "task" / "audit_path_risks.csv").exists()
    for pointer_name in ("latest.json", "best.json"):
        pointer = json.loads(
            (tmp_path / "task" / "checkpoints" / pointer_name).read_text()
        )
        assert pointer["physical_training_performed"] == 0
        assert pointer["sampling_performed"] == 0

    best_pointer_path = tmp_path / "task" / "checkpoints" / "best.json"
    tampered = json.loads(best_pointer_path.read_text())
    tampered["authoritative_sha256"] = "0" * 64
    atomic_write_json(best_pointer_path, tampered)
    with pytest.raises(cli.ArtifactCompatibilityError, match="checkpoint chain"):
        cli.run_streamed_control_task(**kwargs)


def _assert_nested_exact(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert left.device.type == right.device.type
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            _assert_nested_exact(left_value, right_value)
    else:
        assert left == right


def test_interrupted_checkpoint_resume_matches_uninterrupted_exactly(
    tmp_path: Path,
) -> None:
    dynamics = _dynamics()
    selection = cli.boundary._build_control_arrays(
        role="selection",
        law="bounded_teacher",
        path_count=2,
        first_path_id=300,
        bin_counts=(1, 1, 1, 1, 1),
        horizon=float(natural_horizon(dynamics)),
        grid_size=4,
        seed=43,
    )
    plan = build_stream_plan(
        root_seed=47,
        grid_size=4,
        horizon=float(natural_horizon(dynamics)),
        probes_per_bank=4,
    )
    args = _task_args()
    args.train_steps = 2
    args.validation_steps = (0, 1, 2)
    common: dict[str, Any] = {
        "task_kind": "implicit_teacher",
        "selection_arrays": selection,
        "audit_arrays": None,
        "dynamics": dynamics,
        "args": args,
        "device": torch.device("cpu"),
        "model_seed": 53,
        "learning_rate": 1e-5,
        "loss_scale": cli.FROZEN_IMPLICIT_LOSS_SCALE,
        "stream_plan": plan,
        "fingerprints": {
            "fixture": "interrupted-resume",
            "stream_plan_fingerprint": plan.fingerprint,
        },
        "phase": "pilot",
        "show_progress": False,
        "thresholds": cli.BoundaryControlThresholds(
            bootstrap_confidence=0.90,
            expected_implicit_teacher_seeds=1,
            minimum_passing_implicit_teacher_seeds=1,
        ),
    }
    interrupted_dir = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="injected interruption"):
        cli.run_streamed_control_task(
            task_dir=interrupted_dir,
            interrupt_after_checkpoint_step=1,
            **common,
        )
    resumed = cli.run_streamed_control_task(task_dir=interrupted_dir, **common)
    uninterrupted_dir = tmp_path / "uninterrupted"
    uninterrupted = cli.run_streamed_control_task(
        task_dir=uninterrupted_dir, **common
    )
    assert resumed["metrics"] == uninterrupted["metrics"]
    for key in (
        "selected_step", "training_step", "post_warmup_clip_fraction",
        "clip_fraction_steps_101_1000", "final_200_clip_fraction",
        "checkpoint_selection", "optimization_diagnostics",
    ):
        assert resumed["training_summary"][key] == uninterrupted["training_summary"][key]

    def latest_payload(task_dir: Path) -> dict[str, Any]:
        pointer = json.loads(
            (task_dir / "checkpoints" / "latest.json").read_text()
        )
        return torch.load(
            task_dir / "checkpoints" / pointer["filename"],
            map_location="cpu",
            weights_only=False,
        )

    interrupted_payload = latest_payload(interrupted_dir)
    uninterrupted_payload = latest_payload(uninterrupted_dir)
    for key in (
        "model_state_dict",
        "ema_state_dict",
        "optimizer_state_dict",
        "history",
        "validation_records",
        "checkpoint_selection",
        "stream_cursor",
    ):
        _assert_nested_exact(interrupted_payload[key], uninterrupted_payload[key])


def test_selected_profile_is_write_once_and_rejects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "selected_stability_profile.json"
    selected = {
        "selected": 1,
        "passed": 1,
        "profile": {"learning_rate": 1e-5},
    }
    assert cli._freeze_selected_profile(path, selected) == cli._json_load(path)
    assert cli._freeze_selected_profile(path, selected) == cli._json_load(path)
    tampered = cli._json_load(path)
    tampered["profile"]["learning_rate"] = 3e-6
    atomic_write_json(path, tampered)
    with pytest.raises(cli.ArtifactCompatibilityError, match="frozen selected"):
        cli._freeze_selected_profile(path, selected)


def _ineligible_pilot_task(task_kind: str, model_seed: int) -> dict[str, Any]:
    banks = {
        bank: {
            scope: {"model_score_risk": 0.0, "lower_bound": 0.0}
            for scope in ("overall", "data_end")
        }
        for bank in ("a", "b")
    }
    return {
        "task_kind": task_kind,
        "model_seed": model_seed,
        "metrics": {
            "complete": 0,
            "finite": 0,
            "boundary_admissible": 0,
            "selected_step": 0,
            "clip_fraction_steps_101_1000": 0.0,
            "final_200_clip_fraction": 0.0,
            "selection_objective_banks": banks,
            "mean_dual_bank_selection_risk": 0.0,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def test_pilot_records_numerical_task_failure_and_continues_all_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = cli.parse_args(["--parent-scale-repair-run-dir", "parent", "--no-progress"])
    args.grid_size = 4
    args.pilot_selection_paths = 1
    args.bootstrap_reps = 16
    dynamics = _dynamics()
    calls: list[tuple[str, float]] = []

    def fake_task(**kwargs: Any) -> dict[str, Any]:
        calls.append((kwargs["task_kind"], float(kwargs["learning_rate"])))
        if len(calls) == 1:
            raise FloatingPointError("controlled nonfinite high-LR candidate")
        return _ineligible_pilot_task(kwargs["task_kind"], kwargs["model_seed"])

    monkeypatch.setattr(cli, "run_streamed_control_task", fake_task)
    gate, selected = cli._run_pilot(
        tmp_path,
        args=args,
        manifest={
            "scientific_fingerprint": "science",
            "runtime_fingerprint": "runtime",
            "source_fingerprint": "source",
        },
        dynamics=dynamics,
        device=torch.device("cpu"),
        thresholds=StabilityThresholds(),
    )
    assert len(calls) == 8
    assert gate["passed"] == 0
    assert selected["selected"] == 0
    failures = json.loads((tmp_path / "pilot_task_failures.json").read_text())
    assert failures["failure_count"] == 1
    assert failures["failures"][0]["type"] == "FloatingPointError"
    assert failures["physical_training_performed"] == 0
    assert failures["sampling_performed"] == 0
    registry = json.loads((tmp_path / "pilot_candidate_registry.json").read_text())
    assert len(registry["candidates"]) == 4


def test_pilot_does_not_mask_unexpected_implementation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = cli.parse_args(["--parent-scale-repair-run-dir", "parent", "--no-progress"])
    args.grid_size = 4
    args.pilot_selection_paths = 1
    args.bootstrap_reps = 16
    monkeypatch.setattr(
        cli,
        "run_streamed_control_task",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("programming defect")),
    )
    with pytest.raises(RuntimeError, match="programming defect"):
        cli._run_pilot(
            tmp_path,
            args=args,
            manifest={
                "scientific_fingerprint": "science",
                "runtime_fingerprint": "runtime",
                "source_fingerprint": "source",
            },
            dynamics=_dynamics(),
            device=torch.device("cpu"),
            thresholds=StabilityThresholds(),
        )


def test_nonfinite_gradient_runtime_is_classified_but_other_runtime_is_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loss = torch.tensor(1.0, requires_grad=True)

    def nonfinite(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "The total norm of order 2.0 for gradients from parameters is "
            "non-finite, so it cannot be clipped."
        )

    monkeypatch.setattr(cli.boundary, "scaled_backward_and_clip", nonfinite)
    with pytest.raises(FloatingPointError, match="pre-clip gradient"):
        cli._scaled_backward_and_clip_checked(
            loss, [loss], loss_scale=1.0, grad_clip=1.0
        )

    monkeypatch.setattr(
        cli.boundary,
        "scaled_backward_and_clip",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("unrelated programming failure")
        ),
    )
    with pytest.raises(RuntimeError, match="unrelated programming failure"):
        cli._scaled_backward_and_clip_checked(
            loss, [loss], loss_scale=1.0, grad_clip=1.0
        )


def test_confirmation_fail_fast_clipping_count_boundary() -> None:
    history = [
        {"step": step, "clipped": int(step <= 500)}
        for step in range(1, 4001)
    ]
    warmup_only = cli._clipping_bound_status(
        history,
        warmup_steps=500,
        total_steps=4000,
        maximum_fraction=0.10,
    )
    assert warmup_only["maximum_allowed_clips"] == 350
    assert warmup_only["observed_clips"] == 0
    assert warmup_only["mathematically_impossible"] == 0

    for row in history:
        row["clipped"] = int(501 <= row["step"] <= 850)
    exactly_allowed = cli._clipping_bound_status(
        history,
        warmup_steps=500,
        total_steps=4000,
        maximum_fraction=0.10,
    )
    assert exactly_allowed["observed_clips"] == 350
    assert exactly_allowed["mathematically_impossible"] == 0

    history[850]["clipped"] = 1  # step 851: the 351st post-warmup clip.
    impossible = cli._clipping_bound_status(
        history,
        warmup_steps=500,
        total_steps=4000,
        maximum_fraction=0.10,
    )
    assert impossible["observed_clips"] == 351
    assert impossible["mathematically_impossible"] == 1


def test_legacy_streamed_checkpoint_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "schema": "legacy-d0-stream-checkpoint",
            "schema_version": 0,
            "task_kind": "implicit_teacher",
            "fingerprints": {"fixture": "legacy"},
        },
        path,
    )
    with pytest.raises(
        cli.ArtifactCompatibilityError,
        match="legacy, foreign, or mismatched streamed checkpoint",
    ):
        cli._load_stream_checkpoint(
            path,
            device=torch.device("cpu"),
            task_kind="implicit_teacher",
            fingerprints={"fixture": "legacy"},
        )


def test_confirmation_profile_binding_is_exact_and_enters_task_fingerprint(
    tmp_path: Path,
) -> None:
    selected = {
        "selected": 1,
        "passed": 1,
        "profile": {"learning_rate": 1e-5},
    }
    cli._freeze_selected_profile(
        tmp_path / "selected_stability_profile.json", selected
    )
    atomic_write_json(
        tmp_path / "stability_pilot_gate.json",
        {
            "selected_profile": selected,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        },
    )
    binding = cli._bind_confirmation_profile(tmp_path, selected)
    assert binding["selected_profile"] == selected
    assert binding["physical_training_performed"] == 0
    assert binding["sampling_performed"] == 0

    dynamics = _dynamics()
    panel = cli.boundary._build_control_arrays(
        role="selection", law="bounded_teacher", path_count=1,
        first_path_id=1, bin_counts=(1, 1, 1, 1, 1),
        horizon=float(natural_horizon(dynamics)), grid_size=4, seed=5,
    )
    plan = build_stream_plan(
        root_seed=7, grid_size=4, horizon=float(natural_horizon(dynamics))
    )
    fingerprints = cli._task_fingerprints(
        manifest={
            "scientific_fingerprint": "science",
            "runtime_fingerprint": "runtime",
            "source_fingerprint": "source",
        },
        phase="confirm", law="bounded_teacher", model_seed=11,
        learning_rate=1e-5, loss_scale=cli.FROZEN_IMPLICIT_LOSS_SCALE,
        stream_plan=plan, selection=panel, audit=panel,
        selected_profile_binding=binding,
    )
    assert fingerprints["selected_profile_binding"] == binding

    tampered = dict(selected)
    tampered["profile"] = {"learning_rate": 3e-6}
    atomic_write_json(tmp_path / "selected_stability_profile.json", tampered)
    with pytest.raises(cli.ArtifactCompatibilityError, match="pilot gate"):
        cli._bind_confirmation_profile(tmp_path, tampered)


def _install_preflight_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "verify_parent_scale_repair_run",
        lambda _: {
            "passed": 1,
            "scientific_fingerprint": "parent-science",
            "artifact_registry_sha256": "a" * 64,
            "implicit_loss_scale": cli.FROZEN_IMPLICIT_LOSS_SCALE,
        },
    )
    monkeypatch.setattr(
        cli.boundary,
        "configure_exact_torch_backend",
        lambda _: {"deterministic": 1, "fixture": "cpu"},
    )
    monkeypatch.setattr(
        cli,
        "run_stein_identity_preflight",
        lambda *args, **kwargs: {
            "schema": "fixture-stein",
            "evaluation_status": "evaluated",
            "passed": 1,
            "sampling_performed": 0,
        },
    )
    monkeypatch.setattr(
        cli,
        "evaluate_stein_identity_preflight",
        lambda evidence, thresholds: {
            "gate": "stein_identity_preflight",
            "evaluation_status": "evaluated",
            "passed": int(evidence.get("passed", 0)),
            "sampling_performed": 0,
        },
    )
    monkeypatch.setattr(
        cli,
        "run_parent_forensic_replay",
        lambda **kwargs: {
            "role": "advisory_only",
            "complete": 1,
            "sampling_performed": 0,
        },
    )


def test_preflight_cli_writes_terminal_evidence_and_report_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_preflight_fakes(monkeypatch)
    root = tmp_path / "runs"
    code = cli.main(
        [
            "--runs-root",
            str(root),
            "--run-name",
            "tiny",
            "--device",
            "cpu",
            "--stage",
            "preflight",
            "--parent-scale-repair-run-dir",
            "parent",
            "--require-gate",
            "preflight",
        ]
    )
    assert code == 0
    run_dir = next(root.iterdir())
    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["required_gate_pass"] == 1
    assert status["physical_training_performed"] == 0
    assert status["sampling_performed"] == 0
    assert (run_dir / "stability_preflight_gate.json").is_file()
    assert (run_dir / "parent_forensic_replay.json").is_file()
    assert (run_dir / "artifact_registry.json").is_file()

    report_code = cli.main(
        [
            "--resume-run-dir",
            str(run_dir),
            "--device",
            "cpu",
            "--stage",
            "report",
            "--parent-scale-repair-run-dir",
            "parent",
            "--require-gate",
            "preflight",
        ]
    )
    assert report_code == 0


def test_failed_required_preflight_commits_artifacts_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_preflight_fakes(monkeypatch)
    monkeypatch.setattr(
        cli,
        "run_stein_identity_preflight",
        lambda *args, **kwargs: {
            "schema": "fixture-stein",
            "evaluation_status": "evaluated",
            "passed": 0,
            "sampling_performed": 0,
        },
    )
    root = tmp_path / "runs"
    code = cli.main(
        [
            "--runs-root",
            str(root),
            "--device",
            "cpu",
            "--stage",
            "preflight",
            "--parent-scale-repair-run-dir",
            "parent",
            "--require-gate",
            "preflight",
            "--no-progress",
        ]
    )
    assert code == 2
    run_dir = next(root.iterdir())
    assert (run_dir / "control_stability_decision.json").is_file()
    assert (run_dir / "artifact_registry.json").is_file()
    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["outcome"] == "gate_failed"


def test_cli_has_no_sampler_dependency() -> None:
    source = Path(cli.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("sampler" in name for name in imported)
    assert "physical_training_performed" in source
    assert "sampling_performed" in source


def test_workflow_report_preserves_closed_decision() -> None:
    thresholds = StabilityThresholds()
    report = cli._workflow_report(
        provenance={"passed": 1},
        preflight={"passed": 1, "evaluation_status": "evaluated"},
        pilot=cli.not_evaluated_gate("pilot", "not run"),
        confirmation=cli.not_evaluated_gate("confirm", "not run"),
        require_gate="preflight",
        thresholds=thresholds,
    )
    assert report["required_gate_pass"] == 1
    assert report["decision"]["decision"] == "optimizer_stability_unresolved"
    assert report["sampling_performed"] == 0
