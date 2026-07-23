from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mnist.diag_d0_score_boundary_controls as boundary_cli
from mnist.diag_d0_score_boundary_controls import (
    ControlArrays,
    _arrays_identity,
    _build_control_arrays,
    _calibrate_loss_scale,
    _failed_task_result,
    _load_arrays,
    _save_arrays,
    _task_fingerprints,
    _train_task,
    parse_args,
    main,
)
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


def _production_dynamics() -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=28,
        alpha_eff=1.0,
        edge_alpha_mode="alpha_eff",
        num_steps=512,
        mass_floor=1e-7,
        limiter_fraction=1.0,
        source_lowfreq_size=7,
        ot_lowres_size=7,
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
        bootstrap_reps=16,
        bootstrap_confidence=0.90,
        training_probe_seed=31,
        calibration_seed=29,
        selection_probe_a_seed=37,
        selection_probe_b_seed=41,
        bootstrap_seed=43,
        batch_index_seed=47,
    )


def _arrays(role: str, law: str, first_path: int, seed: int) -> ControlArrays:
    return _build_control_arrays(
        role=role,
        law=law,
        path_count=2,
        first_path_id=first_path,
        bin_counts=(1, 1, 1, 1, 1),
        horizon=float(natural_horizon(_dynamics())),
        grid_size=4,
        seed=seed,
    )


def _assert_nested_equal(left: object, right: object) -> None:
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)  # type: ignore[arg-type]
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()  # type: ignore[union-attr]
        for key in left:
            _assert_nested_equal(left[key], right[key])  # type: ignore[index]
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)  # type: ignore[arg-type]
        for first, second in zip(left, right):  # type: ignore[arg-type]
            _assert_nested_equal(first, second)
    else:
        assert left == right


def test_required_gate_locks_boundary_control_profile() -> None:
    args = parse_args(
        [
            "--failed-score-run-dir", "failed",
            "--require-gate", "controls",
        ]
    )
    assert args.teacher_seeds == (260771, 260772, 260773)
    assert args.null_seeds == (260771, 260772, 260773)
    assert args.anchor_bin_counts == (4, 4, 4, 4, 16)
    assert args.training_probes == 4
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--failed-score-run-dir", "failed",
                "--require-gate", "controls",
                "--training-probes", "1",
            ]
        )


def test_synthetic_arrays_are_whole_path_isolated_and_round_trip(tmp_path: Path) -> None:
    teacher_train = _arrays("train", "bounded_teacher", 100, 11)
    teacher_selection = _arrays("selection", "bounded_teacher", 200, 12)
    null_train = _arrays("train", "dirichlet_null", 300, 13)
    assert set(teacher_train.path_ids).isdisjoint(teacher_selection.path_ids)
    assert set(teacher_train.path_ids).isdisjoint(null_train.path_ids)
    assert not torch.equal(teacher_train.states, null_train.states)
    binding = {"science": "frozen", "role": "train"}
    path = tmp_path / "teacher.npz"
    _save_arrays(path, teacher_train, binding)
    loaded, _ = _load_arrays(path, binding)
    assert _arrays_identity(loaded) == _arrays_identity(teacher_train)
    with pytest.raises(Exception, match="fingerprint"):
        _load_arrays(path, {"science": "changed", "role": "train"})


def test_tiny_implicit_task_checkpoint_resume_is_byte_stable(tmp_path: Path) -> None:
    train = _arrays("train", "bounded_teacher", 100, 17)
    selection = _arrays("selection", "bounded_teacher", 200, 19)
    args = _task_args()
    fingerprints = _task_fingerprints(
        scientific_fingerprint="science",
        runtime_fingerprint="runtime",
        source_fingerprint_value="source",
        arrays=train,
        selection_arrays=selection,
        audit_arrays=selection,
        task_kind="implicit_teacher",
        model_seed=23,
        loss_scale=1.0,
    )
    _, first = _train_task(
        task_dir=tmp_path / "task",
        task_kind="implicit_teacher",
        train=train,
        selection_arrays=selection,
        dynamics=_dynamics(),
        device=torch.device("cpu"),
        args=args,
        model_seed=23,
        loss_scale=1.0,
        fingerprints=fingerprints,
        show_progress=False,
    )
    (tmp_path / "task" / "checkpoints" / "best.json").unlink()
    (tmp_path / "task" / "checkpoints" / "best_ema.pt").unlink()
    _, resumed = _train_task(
        task_dir=tmp_path / "task",
        task_kind="implicit_teacher",
        train=train,
        selection_arrays=selection,
        dynamics=_dynamics(),
        device=torch.device("cpu"),
        args=args,
        model_seed=23,
        loss_scale=1.0,
        fingerprints=fingerprints,
        show_progress=False,
    )
    assert resumed["selected_step"] == first["selected_step"]
    assert resumed["checkpoint_sha256"] == first["checkpoint_sha256"]
    assert (tmp_path / "task" / "checkpoints" / "step-00000000.pt").is_file()


def test_interrupted_training_resume_matches_uninterrupted_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = _arrays("train", "bounded_teacher", 100, 51)
    selection = _arrays("selection", "bounded_teacher", 200, 52)
    audit = _arrays("audit", "bounded_teacher", 300, 53)
    args = copy.copy(_task_args())
    args.train_steps = 2
    fingerprints = _task_fingerprints(
        scientific_fingerprint="science",
        runtime_fingerprint="runtime",
        source_fingerprint_value="source",
        arrays=train,
        selection_arrays=selection,
        audit_arrays=audit,
        task_kind="implicit_teacher",
        model_seed=59,
        loss_scale=1.0,
    )
    _train_task(
        task_dir=tmp_path / "uninterrupted",
        task_kind="implicit_teacher",
        train=train,
        selection_arrays=selection,
        dynamics=_dynamics(),
        device=torch.device("cpu"),
        args=args,
        model_seed=59,
        loss_scale=1.0,
        fingerprints=fingerprints,
        show_progress=False,
    )

    original_update = boundary_cli.update_ema_state
    calls = 0

    def interrupt_on_second_update(*values: object, **keywords: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption after step-one checkpoint")
        return original_update(*values, **keywords)

    with monkeypatch.context() as context:
        context.setattr(boundary_cli, "update_ema_state", interrupt_on_second_update)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            _train_task(
                task_dir=tmp_path / "resumed",
                task_kind="implicit_teacher",
                train=train,
                selection_arrays=selection,
                dynamics=_dynamics(),
                device=torch.device("cpu"),
                args=args,
                model_seed=59,
                loss_scale=1.0,
                fingerprints=fingerprints,
                show_progress=False,
            )
    _train_task(
        task_dir=tmp_path / "resumed",
        task_kind="implicit_teacher",
        train=train,
        selection_arrays=selection,
        dynamics=_dynamics(),
        device=torch.device("cpu"),
        args=args,
        model_seed=59,
        loss_scale=1.0,
        fingerprints=fingerprints,
        show_progress=False,
    )
    uninterrupted = tmp_path / "uninterrupted" / "checkpoints" / "step-00000002.pt"
    resumed = tmp_path / "resumed" / "checkpoints" / "step-00000002.pt"
    uninterrupted_payload = torch.load(
        uninterrupted, map_location="cpu", weights_only=False
    )
    resumed_payload = torch.load(resumed, map_location="cpu", weights_only=False)
    _assert_nested_equal(resumed_payload, uninterrupted_payload)


def test_loss_scale_calibration_is_positive_training_only_and_frozen(tmp_path: Path) -> None:
    train = _arrays("train", "bounded_teacher", 100, 17)
    args = _task_args()
    binding = {"role": "teacher-train", "identity": _arrays_identity(train)}
    path = tmp_path / "loss_scale.json"
    first = _calibrate_loss_scale(
        path,
        arrays=train,
        dynamics=_dynamics(),
        args=args,
        device=torch.device("cpu"),
        binding=binding,
    )
    repeated = _calibrate_loss_scale(
        path,
        arrays=train,
        dynamics=_dynamics(),
        args=args,
        device=torch.device("cpu"),
        binding=binding,
    )
    assert first == repeated
    assert first["loss_scale"] > 0.0
    assert first["target_initial_gradient_norm"] == pytest.approx(0.5)
    with pytest.raises(Exception, match="fingerprint"):
        _calibrate_loss_scale(
            path,
            arrays=train,
            dynamics=_dynamics(),
            args=args,
            device=torch.device("cpu"),
            binding={"role": "audit"},
        )


def test_provenance_failure_writes_terminal_artifacts_before_nonzero_exit(tmp_path: Path) -> None:
    code = main(
        [
            "--runs-root", str(tmp_path / "runs"),
            "--run-name", "bad-parent",
            "--device", "cpu",
            "--stage", "preflight",
            "--failed-score-run-dir", str(tmp_path / "missing-parent"),
            "--require-gate", "preflight",
            "--no-progress",
        ]
    )
    assert code == 2
    run_dir = next((tmp_path / "runs").iterdir())
    assert (run_dir / "failure.json").is_file()
    assert (run_dir / "boundary_control_gate.json").is_file()
    assert (run_dir / "control_repair_decision.json").is_file()
    assert (run_dir / "run_status.json").is_file()
    assert (run_dir / "artifact_registry.json").is_file()


@pytest.mark.parametrize("task_kind", ["supervised_teacher", "implicit_teacher", "null"])
def test_task_failure_is_a_complete_named_gate_record(task_kind: str) -> None:
    result = _failed_task_result(task_kind, 17, RuntimeError("boom"))
    assert result["task_kind"] == task_kind
    assert result["model_seed"] == 17
    assert result["metrics"]["complete"] == 0
    assert result["gate"]["passed"] == 0
    assert result["failure"] == {"type": "RuntimeError", "message": "boom"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_production_shape_loss_scale_calibration_fits_gpu_budget(tmp_path: Path) -> None:
    dynamics = _production_dynamics()
    train = _build_control_arrays(
        role="train",
        law="bounded_teacher",
        path_count=8,
        first_path_id=100,
        bin_counts=(4, 4, 4, 4, 16),
        horizon=float(natural_horizon(dynamics)),
        grid_size=28,
        seed=47,
    )
    args = _task_args()
    args.base_channels = 32
    args.batch_size = 64
    args.training_probes = 4
    result = _calibrate_loss_scale(
        tmp_path / "production_loss_scale.json",
        arrays=train,
        dynamics=dynamics,
        args=args,
        device=torch.device("cuda"),
        binding={"shape": "production", "identity": _arrays_identity(train)},
    )
    assert result["calibration_state_count"] == 256
    assert result["loss_scale"] > 0.0
    assert torch.cuda.max_memory_allocated() < 7.5 * 1024**3


def test_control_cli_does_not_import_a_sampler() -> None:
    source_path = Path(__file__).parents[1] / "mnist" / "diag_d0_score_boundary_controls.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("sampler" in name for name in imported)
