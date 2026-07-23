from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist.d0_dirichlet_score import D0LinearSplinePotential, exact_generator_from_derivatives
from mnist.d0_one_image_gate import array_fingerprint, atomic_write_json, file_fingerprint
from mnist.d0_dirichlet_score_gate import (
    evaluate_control_bundle,
    evaluate_null_control,
    evaluate_positive_teacher_control,
)
from mnist.diag_d0_dirichlet_score_learnability import (
    RUN_SCHEMA,
    TASK_RESULT_SCHEMA,
    TASK_RESULT_SCHEMA_VERSION,
    TASK_STATUS_SCHEMA,
    TASK_STATUS_SCHEMA_VERSION,
    ScoreArrays,
    _CombinedPotential,
    _ZeroPotential,
    _atomic_save_npz,
    _complete_control_task,
    _control_evidence_records,
    _cross_seed_flux_cosines,
    _binwise_stein_discrepancy,
    _control_templates,
    _load_or_create_witness_plan,
    _load_completed_control_task,
    _load_completed_physical_task,
    _paired_risk_summary,
    _risk_components,
    _smooth_witness_bank,
    _stein_path_rows,
    _synthetic_arrays,
    _task_checkpoint_fingerprints,
    _terminal_artifact_registry,
    _thresholds,
    _train_potential_task,
    _validate_completed_controls_gate,
    _write_physical_audit_intent,
    _witness_training_statistics,
    parse_args,
)
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, natural_horizon


def _required_cli() -> list[str]:
    return [
        "--zero-residual-run-dir", "zero",
        "--parent-one-image-run-dir", "one",
        "--parent-multiscale-run-dir", "multi",
    ]


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


def _arrays() -> ScoreArrays:
    generator = torch.Generator().manual_seed(91)
    raw = torch.rand((8, 16), generator=generator) + 0.1
    states = raw / raw.sum(dim=1, keepdim=True)
    horizon = float(natural_horizon(_dynamics()))
    fractions = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 1.0])
    return ScoreArrays(
        states=states,
        tau=fractions * horizon,
        tau_fraction=fractions,
        labels=torch.full((8,), 3, dtype=torch.long),
        path_ids=np.repeat(np.asarray([100, 101], dtype=np.int64), 4),
        strata=np.asarray([0, 1, 2, 3, 4, 0, 3, 4], dtype=np.int64),
        end_substeps=np.arange(1, 9, dtype=np.int64),
        rates=torch.ones(8),
        horizon=horizon,
        role="audit",
    )


def test_required_gate_locks_the_production_profile() -> None:
    args = parse_args([*_required_cli(), "--require-gate", "score"])
    assert args.training_seeds == (260753, 260754, 260755)
    assert args.anchor_bin_counts == (4, 4, 4, 4, 16)
    assert args.reference_substeps == 256
    with pytest.raises(SystemExit):
        parse_args(
            [
                *_required_cli(),
                "--require-gate", "score",
                "--reference-substeps", "128",
            ]
        )


def test_common_probe_risk_evaluation_makes_identical_models_pair_exactly() -> None:
    dynamics = _dynamics()
    arrays = _arrays()
    coefficients = torch.randn((8, 16), generator=torch.Generator().manual_seed(3)) * 0.01
    baseline = D0LinearSplinePotential(dynamics, coefficients)
    full = _CombinedPotential(baseline, _ZeroPotential())
    components = _risk_components(
        full,
        baseline,
        arrays,
        dynamics,
        device=torch.device("cpu"),
        batch_size=4,
        probes_per_state=4,
        probe_seed=260757,
    )
    summary = _paired_risk_summary(components)
    assert summary["finite_fraction"] == 1.0
    assert summary["score_risk_delta_vs_linear"] == pytest.approx(0.0, abs=1e-12)


def test_cross_seed_flux_cosines_keep_whole_path_pair_coverage() -> None:
    arrays = _arrays()
    base = np.arange(8 * 6, dtype=np.float32).reshape(8, 6) + 1.0
    rows = _cross_seed_flux_cosines({1: base, 2: 2.0 * base, 3: -base}, arrays)
    assert len(rows) == 2 * 3
    by_pair = {(row["seed_a"], row["seed_b"]): row["cosine"] for row in rows if row["path_id"] == 100}
    assert by_pair[(1, 2)] == pytest.approx(1.0)
    assert by_pair[(1, 3)] == pytest.approx(-1.0)


def test_stein_witnesses_are_deterministic_standardized_and_binwise() -> None:
    arrays = _arrays()
    first = _smooth_witness_bank(4, seed=7)
    repeated = _smooth_witness_bank(4, seed=7)
    independent = _smooth_witness_bank(4, seed=8)
    assert first.shape == (32, 16)
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, independent)
    statistics = _witness_training_statistics(first, arrays.states)
    coordinates = arrays.states.double().numpy() @ first.T
    assert np.std(
        (coordinates - statistics["linear_mean"]) / statistics["linear_scale"], axis=0
    ) == pytest.approx(np.ones(32), rel=1e-10, abs=1e-10)
    quadratic = 0.5 * coordinates**2
    assert np.std(
        (quadratic - statistics["quadratic_mean"]) / statistics["quadratic_scale"], axis=0
    ) == pytest.approx(np.ones(32), rel=1e-10, abs=1e-10)

    # Pooling these values before squaring would cancel to zero.  The frozen
    # statistic squares within each time bin and therefore remains positive.
    residual = np.asarray([[1.0], [-1.0], [1.0], [-1.0], [0.0]], dtype=np.float64)
    discrepancy, counts = _binwise_stein_discrepancy(
        residual, path_mask=np.ones(5, dtype=bool), strata=np.arange(5)
    )
    assert residual.mean() ** 2 == pytest.approx(0.0)
    assert discrepancy == pytest.approx(0.8)
    assert counts == [1, 1, 1, 1, 1]


def _control_source_arrays(path_count: int, *, role: str, first_id: int) -> ScoreArrays:
    base = _arrays()
    anchors = 5
    states = base.states[:anchors].repeat(path_count, 1)
    fractions = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9]).repeat(path_count)
    return ScoreArrays(
        states=states,
        tau=fractions * base.horizon,
        tau_fraction=fractions,
        labels=torch.full((path_count * anchors,), 3, dtype=torch.long),
        path_ids=np.repeat(np.arange(first_id, first_id + path_count, dtype=np.int64), anchors),
        strata=np.tile(np.arange(5, dtype=np.int64), path_count),
        end_substeps=np.tile(np.arange(1, 6, dtype=np.int64), path_count),
        rates=torch.ones(path_count * anchors),
        horizon=base.horizon,
        role=role,
    )


def test_controls_use_independent_40_12_12_clusters(tmp_path: Path) -> None:
    train = _control_source_arrays(80, role="train", first_id=0)
    selection = _control_source_arrays(24, role="selection", first_id=100)
    audit = _control_source_arrays(24, role="audit", first_id=200)
    ctrain, cselection, caudit, plan = _control_templates(
        run_dir=tmp_path,
        train=train,
        selection=selection,
        audit=audit,
        scientific_fingerprint="science",
        cache_fingerprint="cache",
    )
    assert tuple(len(set(value.path_ids.tolist())) for value in (ctrain, cselection, caudit)) == (40, 12, 12)
    all_ids = np.concatenate([ctrain.path_ids, cselection.path_ids, caudit.path_ids])
    assert np.unique(all_ids).size == 64
    assert plan["physical_state_values_reused"] == 0
    assert torch.allclose(ctrain.states, torch.full_like(ctrain.states, 1.0 / 16.0))
    teacher = _synthetic_arrays(ctrain, dynamics=_dynamics(), seed=31, teacher=True)
    null = _synthetic_arrays(ctrain, dynamics=_dynamics(), seed=37, teacher=False)
    assert not torch.equal(teacher.states, null.states)


def test_witness_plan_round_trip_is_bound_to_training_states(tmp_path: Path) -> None:
    arrays = _arrays()
    args = SimpleNamespace(stein_a_seed=41, stein_b_seed=43)
    created = _load_or_create_witness_plan(
        run_dir=tmp_path,
        train=arrays,
        dynamics=_dynamics(),
        args=args,
        scientific_fingerprint="science",
        cache_fingerprint="cache",
        read_only=False,
    )
    loaded = _load_or_create_witness_plan(
        run_dir=tmp_path,
        train=arrays,
        dynamics=_dynamics(),
        args=args,
        scientific_fingerprint="science",
        cache_fingerprint="cache",
        read_only=True,
    )
    assert loaded["metadata"]["fingerprint"] == created["metadata"]["fingerprint"]
    changed = ScoreArrays(
        **{**arrays.__dict__, "states": torch.roll(arrays.states, 1, dims=0)}
    )
    with pytest.raises(Exception, match="binding"):
        _load_or_create_witness_plan(
            run_dir=tmp_path,
            train=changed,
            dynamics=_dynamics(),
            args=args,
            scientific_fingerprint="science",
            cache_fingerprint="cache",
            read_only=True,
        )


def test_quadratic_stein_witness_includes_the_exact_hessian_term(tmp_path: Path) -> None:
    arrays = _control_source_arrays(1, role="audit", first_id=700)
    args = SimpleNamespace(stein_a_seed=47, stein_b_seed=53)
    plan = _load_or_create_witness_plan(
        run_dir=tmp_path,
        train=arrays,
        dynamics=_dynamics(),
        args=args,
        scientific_fingerprint="science",
        cache_fingerprint="cache",
        read_only=False,
    )
    zeros = np.zeros_like(arrays.states.numpy())
    row = _stein_path_rows(
        full_gradients=zeros,
        linear_gradients=zeros,
        arrays=arrays,
        dynamics=_dynamics(),
        model_seed=1,
        witness_plan=plan,
        bank="stein_a",
        device=torch.device("cpu"),
    )[0]
    states = arrays.states.double()
    generators: list[np.ndarray] = []
    patterns = plan["arrays"]["stein_a_patterns"]
    linear_scales = plan["arrays"]["stein_a_linear_scale"]
    quadratic_scales = plan["arrays"]["stein_a_quadratic_scale"]
    for index, raw_pattern in enumerate(patterns):
        pattern = torch.as_tensor(raw_pattern, dtype=torch.float64)
        linear_gradient = (pattern / linear_scales[index]).expand(states.shape[0], -1)
        zero_hessian = torch.zeros(
            (states.shape[0], states.shape[1], states.shape[1]), dtype=torch.float64
        )
        generators.append(
            exact_generator_from_derivatives(
                states, linear_gradient, zero_hessian, _dynamics(), time_change=arrays.rates.double()
            ).numpy()
        )
        coordinate = states @ pattern
        quadratic_gradient = coordinate[:, None] * pattern[None, :] / quadratic_scales[index]
        quadratic_hessian = (
            torch.outer(pattern, pattern)[None, :, :].expand(states.shape[0], -1, -1)
            / quadratic_scales[index]
        )
        generators.append(
            exact_generator_from_derivatives(
                states, quadratic_gradient, quadratic_hessian, _dynamics(), time_change=arrays.rates.double()
            ).numpy()
        )
    exact_matrix = np.stack(generators, axis=1)
    expected, _ = _binwise_stein_discrepancy(
        exact_matrix,
        path_mask=np.ones(states.shape[0], dtype=bool),
        strata=arrays.strata,
    )
    assert row["full_discrepancy"] == pytest.approx(expected, rel=2e-6, abs=1e-8)


def test_orchestration_never_imports_a_reverse_sampler() -> None:
    source = Path("mnist/diag_d0_dirichlet_score_learnability.py").read_text(encoding="utf-8")
    assert "d0_one_image_sampler" not in source
    assert "sampling_performed" in source


def _write_fake_control_task(
    root: Path,
    *,
    name: str,
    artifact_name: str,
    schema: str,
    fingerprints: dict[str, object],
    training_seed: int,
    gate: dict[str, object],
    metrics: dict[str, object],
) -> dict[str, object]:
    task_dir = root / "controls" / name
    checkpoints = task_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    best_path = checkpoints / "best_ema.pt"
    best_path.write_bytes((name + "-checkpoint").encode("utf-8"))
    pointer = {
        "selected_step": 1,
        "fingerprints": fingerprints,
        "best_ema_sha256": file_fingerprint(best_path),
    }
    pointer_path = checkpoints / "best.json"
    atomic_write_json(pointer_path, pointer)
    record = {
        "schema": schema,
        "schema_version": 1,
        "metrics": metrics,
        "gate": gate,
        "training_summary": {
            "selected_step": 1,
            "checkpoint_sha256": file_fingerprint(best_path),
            "best_pointer_sha256": file_fingerprint(pointer_path),
        },
        "fingerprints": fingerprints,
        "sampling_performed": 0,
    }
    _complete_control_task(
        task_dir=task_dir,
        artifact_path=root / "controls" / artifact_name,
        artifact=record,
        fingerprints=fingerprints,
        training_seed=training_seed,
        selected_step=1,
    )
    return record


def test_passed_controls_gate_reuses_only_hash_bound_complete_evidence(tmp_path: Path) -> None:
    science, runtime, source, cache = "science", "runtime", "source", "cache"
    train = _control_source_arrays(80, role="train", first_id=0)
    selection = _control_source_arrays(24, role="selection", first_id=100)
    audit = _control_source_arrays(24, role="audit", first_id=200)
    _, _, _, split = _control_templates(
        run_dir=tmp_path,
        train=train,
        selection=selection,
        audit=audit,
        scientific_fingerprint=science,
        cache_fingerprint=cache,
    )
    baselines = tmp_path / "controls" / "baselines"
    baselines.mkdir(parents=True)
    teacher_baseline = baselines / "positive_teacher_linear_spline.pt"
    null_baseline = baselines / "null_linear_spline.pt"
    teacher_baseline.write_bytes(b"teacher-baseline")
    null_baseline.write_bytes(b"null-baseline")
    args = parse_args(_required_cli())
    args.positive_teacher_train_seed = 71
    args.null_train_seed = 73
    args.control_steps = 5
    binding = cache + ":control-split:" + split["fingerprint"]
    teacher_fp = _task_checkpoint_fingerprints(
        manifest_fingerprint=science,
        runtime_fingerprint=runtime,
        source_fingerprint_value=source,
        cache_fingerprint=binding + ":teacher",
        baseline_path=teacher_baseline,
        seed=args.positive_teacher_train_seed,
        train_steps=args.control_steps,
    )
    null_fp = _task_checkpoint_fingerprints(
        manifest_fingerprint=science,
        runtime_fingerprint=runtime,
        source_fingerprint_value=source,
        cache_fingerprint=binding + ":null",
        baseline_path=null_baseline,
        seed=args.null_train_seed,
        train_steps=args.control_steps,
    )
    teacher_metrics = {
        "complete": 1,
        "selected_step": 1,
        "audit_overall_score_gain": 1.0,
        "audit_data_end_score_gain": 1.0,
        "overall_flux_cosine": 1.0,
        "time_bin_flux_cosines": [1.0] * 5,
        "overall_relative_flux_l2": 0.0,
        "time_bin_relative_flux_l2": [0.0] * 5,
        "nonlinear_gain_vs_linear": 1.0,
    }
    null_metrics = {
        "complete": 1,
        "selected_step": 1,
        "audit_improvement_lower_bound": -1.0,
        "comparator": "frozen_training_only_linear_spline_step0",
    }
    teacher_gate = evaluate_positive_teacher_control(teacher_metrics, _thresholds(args))
    null_gate = evaluate_null_control(null_metrics)
    _write_fake_control_task(
        tmp_path,
        name="positive_teacher",
        artifact_name="positive_teacher.json",
        schema=RUN_SCHEMA + "-positive-teacher-control",
        fingerprints=teacher_fp,
        training_seed=args.positive_teacher_train_seed,
        gate=teacher_gate,
        metrics=teacher_metrics,
    )
    _write_fake_control_task(
        tmp_path,
        name="null",
        artifact_name="null_control.json",
        schema=RUN_SCHEMA + "-null-control",
        fingerprints=null_fp,
        training_seed=args.null_train_seed,
        gate=null_gate,
        metrics=null_metrics,
    )
    operator_gate = {"gate": "operator", "passed": 1}
    record = {
        "schema": RUN_SCHEMA + "-controls-gate",
        "schema_version": 1,
        "binding": {
            "scientific_fingerprint": science,
            "runtime_fingerprint": runtime,
            "source_fingerprint": source,
            "cache_fingerprint": cache,
            "control_split_fingerprint": split["fingerprint"],
        },
        "teacher": teacher_gate,
        "null": null_gate,
        "evidence": _control_evidence_records(tmp_path),
        **evaluate_control_bundle(
            operator_gate=operator_gate,
            positive_teacher_gate=teacher_gate,
            null_control_gate=null_gate,
        ),
        "sampling_performed": 0,
    }
    loaded = _validate_completed_controls_gate(
        run_dir=tmp_path,
        controls_gate=record,
        args=args,
        scientific_fingerprint=science,
        runtime_fingerprint=runtime,
        source_fingerprint_value=source,
        cache_fingerprint=cache,
        operator_gate=operator_gate,
        require_evidence=True,
    )
    assert loaded["passed"] == 1
    record["evidence"]["null_best_checkpoint"]["size"] += 1
    with pytest.raises(Exception, match="evidence"):
        _validate_completed_controls_gate(
            run_dir=tmp_path,
            controls_gate=record,
            args=args,
            scientific_fingerprint=science,
            runtime_fingerprint=runtime,
            source_fingerprint_value=source,
            cache_fingerprint=cache,
            operator_gate=operator_gate,
            require_evidence=True,
        )

    # Hash consistency cannot make a stored task gate authoritative when its
    # frozen metrics imply a different decision.
    teacher_path = tmp_path / "controls" / "positive_teacher.json"
    teacher_record = json.loads(teacher_path.read_text(encoding="utf-8"))
    teacher_record["metrics"]["audit_overall_score_gain"] = -1.0
    atomic_write_json(teacher_path, teacher_record)
    teacher_status_path = tmp_path / "controls" / "positive_teacher" / "task_status.json"
    teacher_status = json.loads(teacher_status_path.read_text(encoding="utf-8"))
    teacher_status["control_result_sha256"] = file_fingerprint(teacher_path)
    atomic_write_json(teacher_status_path, teacher_status)
    record["evidence"] = _control_evidence_records(tmp_path)
    with pytest.raises(Exception, match="metrics and stored task gate disagree"):
        _validate_completed_controls_gate(
            run_dir=tmp_path,
            controls_gate=record,
            args=args,
            scientific_fingerprint=science,
            runtime_fingerprint=runtime,
            source_fingerprint_value=source,
            cache_fingerprint=cache,
            operator_gate=operator_gate,
            require_evidence=True,
        )


def test_terminal_registry_includes_exact_checkpoints_control_csvs_and_plots(tmp_path: Path) -> None:
    paths = (
        tmp_path / "tasks" / "seed-1" / "checkpoints" / "step-00000001.pt",
        tmp_path / "tasks" / "seed-1" / "checkpoints" / "latest.json",
        tmp_path / "tasks" / "seed-1" / "checkpoint_metrics.csv",
        tmp_path / "controls" / "null" / "checkpoint_metrics.csv",
        tmp_path / "score_gain.png",
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    registry = _terminal_artifact_registry(tmp_path)["records"]
    assert {path.relative_to(tmp_path).as_posix() for path in paths} <= set(registry)


def test_tiny_score_task_checkpoint_is_restartable(tmp_path: Path) -> None:
    dynamics = _dynamics()
    arrays = _arrays()
    baseline = D0LinearSplinePotential(dynamics, torch.zeros((8, 16)))
    args = SimpleNamespace(
        base_channels=4,
        learning_rate=1e-4,
        weight_decay=0.0,
        ema_decay=0.9,
        validation_batch_size=4,
        selection_probes=1,
        selection_probe_seed=17,
        batch_size=2,
        train_probes=1,
        grad_clip=1.0,
        validation_every=1,
        checkpoint_every=1,
        training_probe_seed=19,
    )
    fingerprints = {"fixture": "tiny-score-resume-v1"}
    torch.backends.mkldnn.enabled = False
    _, first = _train_potential_task(
        task_dir=tmp_path / "task",
        train=arrays,
        selection=arrays,
        baseline=baseline,
        dynamics=dynamics,
        device=torch.device("cpu"),
        args=args,
        training_seed=11,
        train_steps=1,
        fingerprints=fingerprints,
        show_progress=False,
    )
    first_hash = first["checkpoint_sha256"]
    _, resumed = _train_potential_task(
        task_dir=tmp_path / "task",
        train=arrays,
        selection=arrays,
        baseline=baseline,
        dynamics=dynamics,
        device=torch.device("cpu"),
        args=args,
        training_seed=11,
        train_steps=1,
        fingerprints=fingerprints,
        show_progress=False,
    )
    assert resumed["selected_step"] == 1
    assert resumed["checkpoint_sha256"] == first_hash

    task_dir = tmp_path / "task"
    result_path = task_dir / "task_result.json"
    flux_path = task_dir / "audit_nonlinear_flux.npz"
    result = {
        "schema": TASK_RESULT_SCHEMA,
        "schema_version": TASK_RESULT_SCHEMA_VERSION,
        "model_seed": 11,
        "selected_step": int(resumed["selected_step"]),
        "fingerprints": fingerprints,
        "finite": 1,
        "complete": 1,
        "sampling_performed": 0,
        "audit_path_ids_sha256": array_fingerprint(arrays.path_ids),
        "audit_end_substeps_sha256": array_fingerprint(arrays.end_substeps),
        "checkpoint_sha256": resumed["checkpoint_sha256"],
        "best_pointer_sha256": resumed["best_pointer_sha256"],
        "stein_witness_plan_fingerprint": None,
    }
    _write_physical_audit_intent(
        task_dir=task_dir,
        fingerprints=fingerprints,
        audit=arrays,
        model_seed=11,
        training_summary=resumed,
    )
    atomic_write_json(result_path, result)
    flux = np.zeros((arrays.states.shape[0], 2 * dynamics.grid_size**2), dtype=np.float32)
    _atomic_save_npz(
        flux_path, flux=flux, path_ids=arrays.path_ids, end_substeps=arrays.end_substeps
    )
    status = {
        "schema": TASK_STATUS_SCHEMA,
        "schema_version": TASK_STATUS_SCHEMA_VERSION,
        "status": "complete",
        "training_seed": 11,
        "selected_step": int(resumed["selected_step"]),
        "fingerprints": fingerprints,
        "task_result_sha256": file_fingerprint(result_path),
        "flux_sha256": file_fingerprint(flux_path),
    }
    atomic_write_json(task_dir / "task_status.json", status)
    loaded_result, loaded_flux = _load_completed_physical_task(
        task_dir=task_dir, fingerprints=fingerprints, audit=arrays, model_seed=11
    )
    assert loaded_result["selected_step"] == 1
    assert np.array_equal(loaded_flux, flux)

    # A crash after committing result+flux but before the terminal status can
    # validate those immutable audit artifacts without evaluating audit again.
    interrupted_status = dict(status)
    interrupted_status["status"] = "running"
    interrupted_status.pop("task_result_sha256")
    interrupted_status.pop("flux_sha256")
    atomic_write_json(task_dir / "task_status.json", interrupted_status)
    recovered_result, recovered_flux = _load_completed_physical_task(
        task_dir=task_dir,
        fingerprints=fingerprints,
        audit=arrays,
        model_seed=11,
        require_complete_status=False,
    )
    assert recovered_result["selected_step"] == 1
    assert np.array_equal(recovered_flux, flux)
    atomic_write_json(task_dir / "task_status.json", status)

    # Updating the byte hash cannot make a semantically permuted audit artifact legal.
    _atomic_save_npz(
        flux_path,
        flux=flux,
        path_ids=arrays.path_ids[::-1].copy(),
        end_substeps=arrays.end_substeps,
    )
    status["flux_sha256"] = file_fingerprint(flux_path)
    atomic_write_json(task_dir / "task_status.json", status)
    with pytest.raises(Exception, match="audit-state identity"):
        _load_completed_physical_task(
            task_dir=task_dir, fingerprints=fingerprints, audit=arrays, model_seed=11
        )
