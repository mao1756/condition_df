from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError, atomic_write_json
from mnist import diag_d0_jacobi_rb_reverse_controller as cli


PARENT = Path(
    "runs/experiment12_d0_jacobi_rb_coarse_residual_learnability/"
    "20260731-140333_production-exact-k512-coarse-residual-one-image"
)


def _complete_gate(name: str, *, passed: bool = True) -> dict[str, object]:
    checks = {key: 1 for key in cli._DECISION_GATE_CHECKS[name]}
    if not passed:
        if name == "cache":
            checks["local_all_simultaneous_lower_positive"] = 0
        elif name == "control":
            checks["one_phase_reverse_law"] = 0
        else:
            checks[next(iter(checks))] = 0
    record = cli._gate_record(
        name,
        checks,
        numerically_valid=1,
        resource_valid=1,
    )
    if name == "cache":
        record["terminal_near_reverse_start_controlled"] = int(passed)
    if name == "control":
        record.update(
            controller_control_trajectory_performed=1,
            maximum_control_trajectory_phase_count=8,
            one_phase_reverse_law_controlled=int(passed),
            eight_phase_reverse_law_controlled=int(passed),
            M8_refinement_controlled=int(passed),
        )
    return record


def _write_decision_gates(root: Path, *, failed: str | None = None) -> None:
    order = ("preflight", "oracle", "cache", "control")
    failure_seen = False
    for name in order:
        if failure_seen:
            value = cli._not_evaluated(name, f"skipped_after_failed_{failed}_gate")
        else:
            value = _complete_gate(name, passed=name != failed)
            failure_seen = name == failed
        atomic_write_json(root / f"{name}_gate.json", value)


def test_parent_binding_and_image_identity() -> None:
    if not PARENT.is_dir():
        pytest.skip("sealed production parent is unavailable")
    assert cli._verify_parent(PARENT)["passed"] == 1
    target = cli._load_mixed_target(PARENT)
    assert target.shape == (784,)
    assert target.dtype == np.float64
    assert np.isclose(target.sum(), 1.0)


def test_parser_exposes_no_scientific_override() -> None:
    args = cli.parse_args(
        ["--parent-coarse-residual-run-dir", str(PARENT), "--stage", "report"]
    )
    assert args.stage == "report"
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--parent-coarse-residual-run-dir",
                str(PARENT),
                "--microsteps",
                "16",
            ]
        )


@pytest.mark.parametrize(
    "value",
    [
        {"data_end": 1},
        {"nested": [{"old.data_end.signal": 1}]},
        {"nested": "legacy_data_end_metric"},
    ],
)
def test_ambiguous_metric_names_fail_closed(value: object) -> None:
    with pytest.raises(cli.ReverseControllerCLIError):
        cli._assert_unambiguous_metric_schema(value)


def test_path_plan_is_disjoint_and_20_bit() -> None:
    plan = cli._path_plan()
    values = [
        *plan["roles"]["preflight"],
        *plan["roles"]["physical_control"],
        *plan["roles"]["oracle"],
    ]
    assert len(values) == len(set(values))
    assert all(0 <= value < 2**20 for value in values)
    assert plan["collision_free"] == 1


def test_formula_controls_cover_every_phase_and_microstep() -> None:
    rows, negatives = cli._formula_controls()
    assert len(rows) == 7 * 4
    assert all(row["passed"] == 1 for row in rows)
    assert all(row["rejected"] == 1 for row in negatives)


def test_decide_requires_all_upstream_artifacts(tmp_path: Path) -> None:
    atomic_write_json(tmp_path / "preflight_gate.json", _complete_gate("preflight"))
    with pytest.raises(ArtifactCompatibilityError, match="oracle"):
        cli._decide_stage(tmp_path, SimpleNamespace())


def test_decide_rejects_missing_gate_fields(tmp_path: Path) -> None:
    _write_decision_gates(tmp_path)
    gate = cli._load_json(tmp_path / "preflight_gate.json")
    gate["checks"].pop("throughput")
    atomic_write_json(tmp_path / "preflight_gate.json", gate)
    with pytest.raises(ArtifactCompatibilityError, match="required decision fields"):
        cli._decide_stage(tmp_path, SimpleNamespace())


def test_passing_decision_finalizes_claim_and_is_orphan_idempotent(
    tmp_path: Path,
) -> None:
    _write_decision_gates(tmp_path)
    record = cli._decide_stage(tmp_path, SimpleNamespace())
    assert record["decision"] == "exact_rb_time_local_reverse_controller_controlled"
    assert record["one_image_reconstruction_planning_authorized"] == 1
    claim = cli._load_json(tmp_path / "claim_boundary.json")
    assert claim["one_image_reconstruction_planning_authorized"] == 1
    assert claim["maximum_control_trajectory_phase_count"] == 8
    (tmp_path / "claim_boundary.json").unlink()
    (tmp_path / "decide_gate.json").unlink()
    assert cli._decide_stage(tmp_path, SimpleNamespace()) == record
    assert (tmp_path / "claim_boundary.json").is_file()
    assert (tmp_path / "decide_gate.json").is_file()


def test_early_failure_does_not_claim_control_trajectory(tmp_path: Path) -> None:
    _write_decision_gates(tmp_path, failed="preflight")
    record = cli._decide_stage(tmp_path, SimpleNamespace())
    assert record["controller_control_trajectory_performed"] == 0
    assert record["maximum_control_trajectory_phase_count"] == 0


def test_failed_control_records_execution_but_not_planning(tmp_path: Path) -> None:
    _write_decision_gates(tmp_path, failed="control")
    record = cli._decide_stage(tmp_path, SimpleNamespace())
    assert record["controller_control_trajectory_performed"] == 1
    assert record["one_image_reconstruction_planning_authorized"] == 0
    claim = cli._load_json(tmp_path / "claim_boundary.json")
    assert claim["controller_control_trajectory_performed"] == 1
    assert claim["one_image_reconstruction_planning_authorized"] == 0


def test_status_reads_evaluated_control_phase_count(tmp_path: Path) -> None:
    atomic_write_json(tmp_path / "control_gate.json", _complete_gate("control"))
    status = cli._status(tmp_path, state="complete", stage="control")
    assert status["controller_control_trajectory_performed"] == 1
    assert status["maximum_control_trajectory_phase_count"] == 8


def test_boundary_rejection_is_evaluated_scientific_gate(tmp_path: Path) -> None:
    exc = cli._controller.ControllerBoundaryStepRejected("outward flow")
    gate = cli._commit_control_boundary_rejection(tmp_path, exc=exc)
    assert gate["evaluation_status"] == "evaluated"
    assert gate["numerically_valid"] == 0
    assert gate["checks"]["boundary_rejections"] == 0
    assert gate["controller_control_trajectory_performed"] == 1


def test_preflight_boundary_rejection_maps_to_boundary_decision(
    tmp_path: Path,
) -> None:
    exc = cli._controller.ControllerBoundaryStepRejected("outward flow")
    gate = cli._commit_preflight_boundary_rejection(tmp_path, exc=exc)
    assert gate["evaluation_status"] == "evaluated"
    assert gate["numerically_valid"] == 0
    assert cli._decision_from_gates(gate, None, None, None)[0] == (
        "controller_boundary_or_conservation_failed"
    )


def test_decision_prioritizes_numerical_and_resource_failures() -> None:
    preflight = _complete_gate("preflight")
    oracle = _complete_gate("oracle")
    cache = _complete_gate("cache", passed=False)
    cache["numerically_valid"] = 0
    assert cli._decision_from_gates(preflight, oracle, cache, None)[0] == (
        "controller_boundary_or_conservation_failed"
    )
    cache["numerically_valid"] = 1
    cache["resource_valid"] = 0
    assert cli._decision_from_gates(preflight, oracle, cache, None)[0] == (
        "reverse_controller_control_resource_infeasible"
    )


def _write_control_audit(root: Path, *, phase: int | None, wrong_ids: bool = False) -> None:
    state = np.full((64, 784), 1.0 / 784.0, dtype=np.float64)
    path_ids = np.asarray(cli.PHYSICAL_PATH_IDS, dtype=np.int64)
    if wrong_ids:
        path_ids = path_ids[::-1].copy()
    filename = (
        f"one-phase-anchor-0127-phase-{phase}.npz"
        if phase is not None
        else "eight-phase-anchor-0127.npz"
    )
    cli._atomic_npz(
        root / "control" / "audit" / filename,
        {"earlier_state": state, "later_state": state, "path_ids": path_ids},
    )


@pytest.mark.parametrize("phase", [0, 4, 5, 6])
def test_control_audit_repeated_occurrences_use_color_mapping(
    tmp_path: Path, phase: int
) -> None:
    _write_control_audit(tmp_path, phase=phase)
    audit = cli._load_control_audit(tmp_path, anchor=127, phase=phase)
    assert audit["later_state"].shape == (64, 784)


def test_control_audit_rejects_path_reordering(tmp_path: Path) -> None:
    _write_control_audit(tmp_path, phase=6, wrong_ids=True)
    with pytest.raises(ArtifactCompatibilityError, match="path order"):
        cli._load_control_audit(tmp_path, anchor=127, phase=6)


def test_certified_reference_rejects_oversize_before_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "sample_alpha1_rb_transition_batch_cuda", forbidden)
    reference = cli._CertifiedReference(
        root_seed=1, profile=cli.JacobiRBCudaProfile()
    )
    x = torch.zeros((11, 392), dtype=torch.float64)
    with pytest.raises(cli.ReverseControllerCLIError, match="4096"):
        reference(
            head_fraction=x,
            exposure=x,
            transition_ids=torch.zeros_like(x, dtype=torch.int64),
            role="fixture",
        )
    assert not called


def test_control_trajectory_slices_64_paths_into_eight_path_cohorts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    atomic_write_json(tmp_path / "run_manifest.json", {"source_fingerprint": "s"})
    batches: list[int] = []

    class FakeReference:
        def __init__(self, **kwargs: object) -> None:
            self.transition_count = 0
            self.certified_count = 0
            self.fallback_count = 0
            self.fallback_seconds = 0.0
            self.elapsed_seconds = 0.0
            self.maximum_transition_count_per_call = 0
            self.forbidden = {name: 0 for name in cli.FORBIDDEN_DIAGNOSTICS}

    def controlled(state: torch.Tensor, *args: object, **kwargs: object) -> object:
        batches.append(state.shape[0])
        reference = kwargs["reference_transition"]
        count = state.numel() // 2 * 2 * int(args[2])
        reference.transition_count += count
        reference.certified_count += count
        reference.maximum_transition_count_per_call = max(
            reference.maximum_transition_count_per_call, state.shape[0] * 392
        )
        return SimpleNamespace(
            state=state.clone(),
            maximum_pair_mass_error=0.0,
            maximum_simplex_mass_error=0.0,
        )

    monkeypatch.setattr(cli, "_CertifiedReference", FakeReference)
    monkeypatch.setattr(cli._controller, "controlled_reverse_phase", controlled)
    initial = np.full((64, 784), 1.0 / 784.0, dtype=np.float64)
    output, _ = cli._run_control_trajectory(
        tmp_path,
        stem="fixture",
        later_state=initial,
        sequence=((127, 6),),
        microsteps=2,
        controller=object(),
        device=torch.device("cpu"),
        profile=cli.JacobiRBCudaProfile(),
        stream_role="fixture",
    )
    assert batches == [8] * 8
    assert np.array_equal(output, initial)
    batches.clear()
    resumed, _ = cli._run_control_trajectory(
        tmp_path,
        stem="fixture",
        later_state=initial,
        sequence=((127, 6),),
        microsteps=2,
        controller=object(),
        device=torch.device("cpu"),
        profile=cli.JacobiRBCudaProfile(),
        stream_role="fixture",
    )
    assert batches == []
    assert np.array_equal(resumed, output)


def test_preflight_controller_partition_is_group_order_and_restart_invariant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeReference:
        def __init__(self, **kwargs: object) -> None:
            self.transition_count = 0
            self.certified_count = 0
            self.fallback_count = 0
            self.fallback_seconds = 0.0
            self.elapsed_seconds = 1.0
            self.maximum_transition_count_per_call = 0
            self.forbidden = {name: 0 for name in cli.FORBIDDEN_DIAGNOSTICS}

        def record(self) -> dict[str, object]:
            return {
                "transition_count": self.transition_count,
                "certified_count": self.certified_count,
                "certificate_fraction": 1.0,
                "fallback_count": 0,
                "fallback_fraction": 0.0,
                "fallback_seconds": 0.0,
                "fallback_time_fraction": 0.0,
                "elapsed_seconds": self.elapsed_seconds,
                "transitions_per_second": self.transition_count,
                "maximum_transition_count_per_call": self.maximum_transition_count_per_call,
                "forbidden_counts": dict(self.forbidden),
            }

    def controlled(state: torch.Tensor, *args: object, **kwargs: object) -> object:
        reference = kwargs["reference_transition"]
        path_ids = tuple(kwargs["path_ids"])
        count = state.shape[0] * 392 * 4
        reference.transition_count += count
        reference.certified_count += count
        reference.maximum_transition_count_per_call = max(
            reference.maximum_transition_count_per_call, state.shape[0] * 392
        )
        output = state.clone()
        delta = torch.tensor(
            [(int(path_id) % 7) * 1.0e-8 for path_id in path_ids],
            dtype=state.dtype,
            device=state.device,
        )
        output[:, 0] += delta
        output[:, 1] -= delta
        return SimpleNamespace(
            state=output,
            maximum_pair_mass_error=0.0,
            maximum_simplex_mass_error=0.0,
        )

    monkeypatch.setattr(cli, "_CertifiedReference", FakeReference)
    monkeypatch.setattr(cli._controller, "controlled_reverse_phase", controlled)
    initial = torch.full((8, 784), 1.0 / 784.0, dtype=torch.float64)
    ids = cli.PREFLIGHT_PATH_IDS
    sequence = ((511, 6), (511, 5))

    full, full_diag = cli._preflight_controller_partition(
        initial,
        path_ids=ids,
        group_sizes=(8,),
        sequence=sequence,
        controller=object(),
        profile=cli.JacobiRBCudaProfile(),
        stream_role="fixture",
    )
    grouped, grouped_diag = cli._preflight_controller_partition(
        initial,
        path_ids=ids,
        group_sizes=(4, 4),
        sequence=sequence,
        controller=object(),
        profile=cli.JacobiRBCudaProfile(),
        stream_role="fixture",
    )
    reversed_state, reversed_diag = cli._preflight_controller_partition(
        initial.flip(0),
        path_ids=tuple(reversed(ids)),
        group_sizes=(8,),
        sequence=sequence,
        controller=object(),
        profile=cli.JacobiRBCudaProfile(),
        stream_role="fixture",
    )
    first, first_diag = cli._preflight_controller_partition(
        initial,
        path_ids=ids,
        group_sizes=(8,),
        sequence=sequence[:1],
        controller=object(),
        profile=cli.JacobiRBCudaProfile(),
        stream_role="fixture",
    )
    cli._atomic_npz(tmp_path / "restart.npz", {"states": first.numpy()})
    reloaded = torch.from_numpy(cli._load_npz(tmp_path / "restart.npz")["states"])
    resumed, second_diag = cli._preflight_controller_partition(
        reloaded,
        path_ids=ids,
        group_sizes=(8,),
        sequence=sequence[1:],
        controller=object(),
        profile=cli.JacobiRBCudaProfile(),
        stream_role="fixture",
    )
    assert torch.equal(full, grouped)
    assert torch.equal(full, reversed_state.flip(0))
    assert torch.equal(full, resumed)
    for diagnostics in (full_diag, grouped_diag, reversed_diag):
        assert diagnostics["transition_count"] == 8 * 392 * 4 * 2
        assert diagnostics["certified_count"] == diagnostics["transition_count"]
        assert all(value == 0 for value in diagnostics["forbidden_counts"].values())
        assert diagnostics["maximum_transition_count_per_call"] <= 4096
    assert first_diag["transition_count"] + second_diag["transition_count"] == full_diag["transition_count"]


def test_registered_gate_only_trusts_sealed_registry(tmp_path: Path) -> None:
    gate_path = tmp_path / "preflight_gate.json"
    atomic_write_json(gate_path, _complete_gate("preflight"))
    assert cli._registered_stage_gate(tmp_path, gate_path) is None
    assert not gate_path.exists()
    atomic_write_json(gate_path, _complete_gate("preflight"))
    cli._artifact_registry(tmp_path)
    assert cli._registered_stage_gate(tmp_path, gate_path)["passed"] == 1


def test_unregistered_orphan_ledger_is_replaceable_but_registered_is_not(
    tmp_path: Path,
) -> None:
    cli._artifact_registry(tmp_path)
    ledger = tmp_path / "forward" / "checkpoint_hashes.json"
    ledger.parent.mkdir(parents=True)
    atomic_write_json(ledger, {"version": "orphan"})
    assert cli._commit_recoverable_json(
        tmp_path, ledger, {"version": "rederived"}
    ) == {"version": "rederived"}
    cli._artifact_registry(tmp_path)
    with pytest.raises(ArtifactCompatibilityError, match="registered artifact"):
        cli._commit_recoverable_json(tmp_path, ledger, {"version": "changed"})


def test_uncommitted_forward_checkpoint_is_never_resumed(tmp_path: Path) -> None:
    atomic_write_json(
        tmp_path / "run_manifest.json",
        {
            "source_fingerprint": "source",
            "scientific_config_sha256": "science",
            "path_plan_sha256": "paths",
        },
    )
    states = np.full((64, 784), 1.0 / 784.0, dtype=np.float64)
    record = cli._persist_state_checkpoint(
        tmp_path,
        step=8,
        states=states,
        input_state_sha256="input",
        scheduler_record={"diagnostics": {}},
        local_artifacts=(),
        control_artifacts=(),
        branch_diagnostics={},
        wall_elapsed_seconds=1.0,
    )
    loaded, _ = cli._load_valid_state_checkpoint(
        tmp_path, 8, expected_input_sha256="input"
    )
    assert loaded is not None
    record["committed"] = 0
    atomic_write_json(tmp_path / "forward" / "checkpoint-step-0008.json", record)
    loaded, loaded_record = cli._load_valid_state_checkpoint(
        tmp_path, 8, expected_input_sha256="input"
    )
    assert loaded is None and loaded_record is None


def test_incompatible_resume_rejects_without_mutating_existing_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    args = SimpleNamespace(
        parent_coarse_residual_run_dir=parent,
        device="cpu",
    )
    records = {
        "parent": {"schema": "parent", "passed": 1},
        "package": {"schema": "package", "all_hashes_verified": 1},
        "path": {"schema": "path", "semantic_sha256": "paths"},
        "namespace": {"schema": "namespace"},
        "contract": {"schema": "contract"},
        "convention": {"schema": "convention"},
        "config": {"schema": "config", "semantic_sha256": "science"},
        "collision": {"schema": "collision", "passed": 1},
    }
    monkeypatch.setattr(cli, "_verify_parent", lambda _parent: records["parent"])
    monkeypatch.setattr(cli, "_verify_package_manifest", lambda: records["package"])
    monkeypatch.setattr(cli, "_path_plan", lambda: records["path"])
    monkeypatch.setattr(cli, "_transition_namespace_record", lambda: records["namespace"])
    monkeypatch.setattr(cli, "_model_input_contract", lambda: records["contract"])
    monkeypatch.setattr(cli, "_controller_convention", lambda: records["convention"])
    monkeypatch.setattr(cli, "_scientific_config", lambda: records["config"])
    monkeypatch.setattr(
        cli, "_semantic_path_collision_scan", lambda _run_dir: records["collision"]
    )
    monkeypatch.setattr(cli, "_source_paths", lambda _parent: ())
    monkeypatch.setattr(cli, "source_fingerprint", lambda _paths: "source-a")

    cli._initialize_run(tmp_path, args, resumed=False)
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(cli, "source_fingerprint", lambda _paths: "source-b")
    with pytest.raises(ArtifactCompatibilityError, match="run_manifest.json"):
        cli._initialize_run(tmp_path, args, resumed=True)
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_max_t_helpers_preserve_frozen_family_shapes() -> None:
    rng = np.random.default_rng(5)
    local = cli._one_sided_matrix_max_t(
        rng.normal(size=(64, 228)),
        names=[f"f{i}" for i in range(228)],
        confidence=0.9,
        replicates=64,
        seed=7,
    )
    assert local["family_size"] == 228
    numerator = rng.normal(size=(64, 784))
    denominator = rng.normal(size=(64, 784))
    trajectory = cli._normalized_trajectory_max_t(
        numerator,
        denominator,
        confidence=0.9,
        replicates=32,
        seed=8,
        names=[f"t{i}" for i in range(784)],
    )
    assert trajectory["family_size"] == 784


def test_cache_schema_rejects_joined_input_target_npz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "SELECTED_OUTER_STEPS", ())
    cli._atomic_npz(
        tmp_path / "input_probe.npz",
        {"later_full_state": np.zeros((1, 784), dtype=np.float64)},
    )
    cli._atomic_npz(
        tmp_path / "label_audit_probe.npz",
        {"denoising_target": np.zeros((1, 392), dtype=np.float64)},
    )
    assert cli._separated_cache_schema_valid(tmp_path)
    cli._atomic_npz(
        tmp_path / "joined.npz",
        {
            "later_full_state": np.zeros((1, 784), dtype=np.float64),
            "denoising_target": np.zeros((1, 392), dtype=np.float64),
        },
    )
    assert not cli._separated_cache_schema_valid(tmp_path)


def test_cli_source_does_not_import_a_sampler_or_image_writer() -> None:
    text = Path(cli.__file__).read_text(encoding="utf-8")
    assert "reverse_sampler" not in text
    assert "save_image" not in text
    assert "matplotlib" not in text
