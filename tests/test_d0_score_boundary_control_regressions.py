from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import mnist.diag_d0_score_boundary_controls as boundary_cli
from mnist.d0_score_boundary_control_gate import (
    BoundaryControlDecision,
    BoundaryControlThresholds,
    evaluate_boundary_control_gates,
    evaluate_implicit_teacher_study,
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


def _passing_raw_preflight() -> dict[str, object]:
    return {
        "passed": True,
        "operator": {"passed": True},
        "orthogonal_probe_preflight": {"passed": True},
        "facet_ray": {
            "passed": True,
            "checks": {
                "smooth_quantities_finite": {"passed": True},
                "conormal_log_log_slope": {"value": 1.0, "passed": True},
                "conormal_four_decade_decay": {"value": 1e-4, "passed": True},
                "legacy_barrier_nonvanishing": {"passed": True},
            },
            "rows": [],
        },
        "legacy_log_barrier": {
            "passed": True,
            "empirical_relative_error": 0.0,
            "checks": {
                "expected_negative_coefficient": {"passed": True},
                "empirical_coefficient": {"passed": True},
                "conormal_does_not_vanish": {"passed": True},
                # This marker is intentionally always true in the raw
                # fixture.  The production gate must bind the substantive
                # conormal evidence above, not this declarative label alone.
                "fixture_rejected": {"passed": True},
            },
            "boundary_rows": [],
        },
    }


@pytest.mark.parametrize("broken_evidence", ["aggregate", "probe", "legacy_conormal"])
def test_boundary_preflight_fails_for_every_substantive_raw_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    broken_evidence: str,
) -> None:
    raw = _passing_raw_preflight()
    if broken_evidence == "aggregate":
        raw["passed"] = False
    elif broken_evidence == "probe":
        raw["passed"] = False
        raw["orthogonal_probe_preflight"]["passed"] = False  # type: ignore[index]
    elif broken_evidence == "legacy_conormal":
        raw["passed"] = False
        legacy = raw["legacy_log_barrier"]  # type: ignore[assignment]
        legacy["passed"] = False
        legacy["checks"]["conormal_does_not_vanish"]["passed"] = False
    else:  # pragma: no cover - guarded by parametrization
        raise AssertionError(broken_evidence)

    monkeypatch.setattr(
        boundary_cli,
        "run_boundary_operator_preflight",
        lambda *args, **kwargs: copy.deepcopy(raw),
    )
    run_dir = tmp_path / broken_evidence
    run_dir.mkdir()
    _, gate = boundary_cli._run_preflight(
        run_dir,
        dynamics=_dynamics(),
        args=SimpleNamespace(operator_hutchinson_probes=4),
        device=torch.device("cpu"),
        binding={"case": broken_evidence},
        thresholds=BoundaryControlThresholds(),
    )
    assert gate["passed"] == 0


def _bank_result(a: float, b: float) -> dict[str, object]:
    return {
        "metrics": {
            "audit_objective_banks": {
                "a": {
                    "overall": {"lower_bound": a},
                    "data_end": {"lower_bound": a},
                },
                "b": {
                    "overall": {"lower_bound": b},
                    "data_end": {"lower_bound": b},
                },
            }
        }
    }


def test_probe_bank_agreement_includes_stationary_null_results() -> None:
    teacher_results = [_bank_result(0.2, 0.1)]
    agreeing_null_results = [_bank_result(-0.2, -0.1)]
    disagreeing_null_results = [_bank_result(-0.2, 0.1)]

    assert boundary_cli._probe_banks_agree(
        teacher_results=teacher_results,
        null_results=agreeing_null_results,
    )
    assert not boundary_cli._probe_banks_agree(
        teacher_results=teacher_results,
        null_results=disagreeing_null_results,
    )


def _array_args() -> SimpleNamespace:
    return SimpleNamespace(
        anchor_bin_counts=(1, 1, 1, 1, 1),
        anchors_per_path=5,
        train_paths=2,
        selection_paths=1,
        audit_paths=1,
        grid_size=4,
        teacher_data_seed=101,
        null_data_seed=201,
        training_probes=1,
        selection_probes=1,
        audit_probes=1,
        training_probe_seed=301,
        selection_probe_a_seed=302,
        selection_probe_b_seed=303,
        audit_probe_a_seed=304,
        audit_probe_b_seed=305,
    )


@pytest.mark.parametrize("sidecar_state", ["missing", "mismatched"])
def test_prepare_control_arrays_recovers_incomplete_npz_sidecar_commit(
    tmp_path: Path,
    sidecar_state: str,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = _array_args()
    parent = {"schedule_metadata": {"horizon": 1.0}}
    first, _ = boundary_cli._prepare_control_arrays(
        run_dir,
        args=args,
        parent=parent,
        scientific_fingerprint="frozen-science",
        resumed=False,
    )
    expected_identity = boundary_cli._arrays_identity(first["teacher_train"])

    sidecar = run_dir / "synthetic_arrays" / "teacher_train.json"
    if sidecar_state == "missing":
        sidecar.unlink()
    else:
        sidecar.write_text(
            json.dumps({"schema": "stale-interrupted-sidecar"}),
            encoding="utf-8",
        )

    recovered, _ = boundary_cli._prepare_control_arrays(
        run_dir,
        args=args,
        parent=parent,
        scientific_fingerprint="frozen-science",
        resumed=True,
    )
    assert boundary_cli._arrays_identity(recovered["teacher_train"]) == expected_identity
    assert sidecar.is_file()


def _small_arrays(role: str, first_path: int, seed: int) -> boundary_cli.ControlArrays:
    return boundary_cli._build_control_arrays(
        role=role,
        law="bounded_teacher",
        path_count=1,
        first_path_id=first_path,
        bin_counts=(1, 1, 1, 1, 1),
        horizon=1.0,
        grid_size=4,
        seed=seed,
    )


def _fingerprints(
    train: boundary_cli.ControlArrays,
    selection: boundary_cli.ControlArrays,
    audit: boundary_cli.ControlArrays,
) -> dict[str, object]:
    return boundary_cli._task_fingerprints(
        scientific_fingerprint="science",
        runtime_fingerprint="runtime",
        source_fingerprint_value="source",
        arrays=train,
        selection_arrays=selection,
        audit_arrays=audit,
        task_kind="implicit_teacher",
        model_seed=17,
        loss_scale=1.0,
    )


def test_task_fingerprint_binds_selection_and_audit_array_identities() -> None:
    train = _small_arrays("train", 100, 1)
    selection = _small_arrays("selection", 200, 2)
    audit = _small_arrays("audit", 300, 3)
    baseline = _fingerprints(train, selection, audit)

    changed_selection = _small_arrays("selection", 200, 22)
    changed_audit = _small_arrays("audit", 300, 33)
    assert _fingerprints(train, changed_selection, audit) != baseline
    assert _fingerprints(train, selection, changed_audit) != baseline


def _passing_teacher_metrics(seed: int) -> dict[str, object]:
    return {
        "model_seed": seed,
        "complete": 1,
        "finite": 1,
        "selected_step": 500,
        "audit_overall_score_gain": 0.92,
        "audit_data_end_score_gain": 0.92,
        "overall_flux_cosine": 0.99,
        "time_bin_flux_cosines": [0.96] * 5,
        "overall_relative_flux_l2": 0.12,
        "time_bin_relative_flux_l2": [0.18] * 5,
        "boundary_admissible": 1,
        "post_warmup_clip_fraction": 0.05,
        "audit_objective_banks": {
            name: {
                "overall": {"lower_bound": 0.2},
                "data_end": {"lower_bound": 0.1},
            }
            for name in ("a", "b")
        },
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("complete", 0),
        ("finite", 0),
        ("boundary_admissible", 0),
        ("post_warmup_clip_fraction", 0.1000001),
    ],
)
def test_implicit_teacher_study_requires_all_three_tasks_to_be_healthy(
    field: str,
    value: object,
) -> None:
    rows = [_passing_teacher_metrics(seed) for seed in (11, 12, 13)]
    rows[2][field] = value
    gate = evaluate_implicit_teacher_study(rows, BoundaryControlThresholds())
    assert gate["passed"] == 0


def test_required_controls_gate_fails_on_probe_bank_disagreement() -> None:
    components = {
        "provenance_pass": {"passed": 1},
        "boundary_preflight": {"passed": 1},
        "supervised_teacher": {"passed": 1},
        "implicit_teacher_study": {"passed": 1},
        "null_study": {"passed": 1},
    }
    report = evaluate_boundary_control_gates(
        **components,
        require_gate="controls",
        probe_banks_agree=False,
    )
    assert report["required_gate_pass"] == 0
    assert (
        report["decision"]["decision"]
        == BoundaryControlDecision.TRACE_ESTIMATOR_INCONCLUSIVE.value
    )
    assert report["decision"]["physical_training_authorized"] == 0


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")


def _minimal_failed_parent(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Path]]:
    root.mkdir()
    scientific = {"kernel": dict(boundary_cli.EXPECTED_KERNEL)}
    fingerprint = boundary_cli.config_fingerprint(scientific)
    monkeypatch.setattr(
        boundary_cli,
        "EXPECTED_FAILED_SCORE_SCIENTIFIC_FINGERPRINT",
        fingerprint,
    )
    required = {
        "status": root / "run_status.json",
        "preflight": root / "preflight_gate.json",
        "cache": root / "cache_gate.json",
        "controls": root / "controls_gate.json",
        "operator": root / "operator_preflight.json",
        "cache_index": root / "cache" / "parent" / "cache_index.json",
    }
    _write_json(
        required["status"],
        {
            "status": "complete",
            "outcome": "gate_failed",
            "decision": "optimization_pipeline_invalid",
            "sampling_performed": 0,
        },
    )
    _write_json(required["preflight"], {"passed": 1})
    _write_json(required["cache"], {"passed": 1})
    _write_json(
        required["controls"],
        {
            "passed": 0,
            "teacher": {"passed": 0},
            "null": {"passed": 0},
            "evidence": {},
        },
    )
    _write_json(required["operator"], {"passed": 1})
    _write_json(
        required["cache_index"],
        {
            "metadata": {
                "schedule_metadata": {
                    "horizon": float(
                        natural_horizon(
                            DirectFluxMNISTConfig(
                                grid_size=28,
                                alpha_eff=1.0,
                                edge_alpha_mode="alpha_eff",
                                num_steps=512,
                                limiter_fraction=1.0,
                                mass_floor=1e-7,
                                source_lowfreq_size=7,
                                ot_lowres_size=7,
                            )
                        )
                    )
                }
            }
        },
    )

    registry_path = root / "artifact_registry.json"
    _write_json(
        registry_path,
        {
            "schema": "fixture-artifact-registry",
            "records": {
                path.relative_to(root).as_posix(): boundary_cli._artifact_record(path)
                for path in required.values()
            },
        },
    )
    _write_json(
        root / "run_manifest.json",
        {
            "schema": "experiment12-d0-dirichlet-score-learnability",
            "scientific_config": scientific,
            "scientific_fingerprint": fingerprint,
            "artifacts": {
                "artifact_registry": boundary_cli._artifact_record(registry_path)
            },
        },
    )
    return root, required


@pytest.mark.parametrize("artifact_name", ["cache", "cache_index"])
def test_failed_parent_registry_rejects_semantic_preserving_artifact_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    parent, required = _minimal_failed_parent(tmp_path / "failed-parent", monkeypatch)
    assert boundary_cli.verify_failed_score_run(parent)["passed"] == 1

    # Whitespace preserves the parsed JSON and every semantic check, so the
    # rejection below specifically exercises the frozen registry hash/size.
    with required[artifact_name].open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(boundary_cli.ArtifactCompatibilityError, match="registry"):
        boundary_cli.verify_failed_score_run(parent)


def test_tiny_successful_cli_preflight_writes_terminal_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, _ = _minimal_failed_parent(tmp_path / "failed-parent", monkeypatch)
    monkeypatch.setattr(
        boundary_cli,
        "run_boundary_operator_preflight",
        lambda *args, **kwargs: _passing_raw_preflight(),
    )
    monkeypatch.setattr(
        boundary_cli,
        "_run_production_workload_smoke",
        lambda *args, **kwargs: {"passed": 1, "production_shape": 0},
    )
    code = boundary_cli.main(
        [
            "--runs-root",
            str(tmp_path / "runs"),
            "--run-name",
            "tiny-success",
            "--device",
            "cpu",
            "--stage",
            "preflight",
            "--failed-score-run-dir",
            str(parent),
            "--require-gate",
            "none",
            "--no-progress",
        ]
    )
    assert code == 0
    run_dir = next((tmp_path / "runs").iterdir())
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["outcome"] == "complete"
    assert status["sampling_performed"] == 0
    assert (run_dir / "boundary_operator_preflight.json").is_file()
    assert (run_dir / "boundary_preflight_gate.json").is_file()
    assert (run_dir / "artifact_registry.json").is_file()
