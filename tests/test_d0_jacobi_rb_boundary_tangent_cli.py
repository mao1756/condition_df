from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

from mnist import (
    diag_d0_jacobi_rb_boundary_tangent_controller_confirmation as cli,
)
from mnist.d0_jacobi_artifacts import atomic_write_json
from mnist.d0_jacobi_rb_learnability import MODEL_INPUT_FIELDS


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        runs_root=tmp_path,
        run_name="fixture",
        device="cpu",
        stage="preflight",
        require_gate="none",
        parent_coarse_residual_run_dir=tmp_path / "coarse",
        failed_controller_run_dir=tmp_path / "failed-controller",
        resume_run_dir=None,
        test_only=True,
        test_path_count=2,
        test_outer_steps=16,
        test_maximum_updates=1,
        test_bootstrap_replicates=100,
    )


def _write_cache_fixture(run_dir: Path, args: argparse.Namespace) -> None:
    args.parent_coarse_residual_run_dir.mkdir(parents=True, exist_ok=True)
    cli._atomic_npz(
        args.parent_coarse_residual_run_dir / "source_image.npz",
        {"mixed_target": np.full(784, 1.0 / 784.0, dtype=np.float64)},
    )
    config = cli._scientific_config(args)
    atomic_write_json(run_dir / "scientific_config.json", config)
    plan = cli.build_boundary_tangent_path_plan()
    atomic_write_json(run_dir / "path_id_plan.json", plan)


def test_effective_paths_use_frozen_preflight_benchmark_role(tmp_path: Path) -> None:
    args = _args(tmp_path)
    paths = cli._effective_paths(args)
    assert paths["preflight"] == (0xEC000, 0xEC001)
    assert paths["train"] == (0xEC100, 0xEC101)
    assert paths["validation"] == (0xEC200, 0xEC201)
    assert paths["confirmation"] == (0xED000, 0xED001)


def test_checkpoint_gate_uses_high_reverse_time_quartile() -> None:
    from mnist.d0_jacobi_rb_reverse_controller import internal_reverse_time

    forward_steps = (15, 143, 271, 399)
    phase = 0
    inputs = cli.ModelInputs(
        later_full_state=torch.full((4, 784), 1.0 / 784.0),
        reverse_time=torch.tensor(
            [internal_reverse_time(step, phase, 1.0 / 16.0) for step in forward_steps],
            dtype=torch.float64,
        ),
        phase=torch.full((4,), phase, dtype=torch.long),
        color=torch.full((4,), cli.PHASE_MATCHINGS[phase], dtype=torch.long),
        duration=torch.full(
            (4,), cli.PHASE_DURATIONS[phase], dtype=torch.float64
        ),
        label=torch.full((4,), 3, dtype=torch.long),
    )
    mask = cli._high_reverse_time_mask(inputs)
    assert mask.tolist() == [True, False, False, False]


def test_initialize_and_reduced_preflight_are_resume_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    parents = {"schema": "fixture-parent-provenance", "passed": 1}
    parents["semantic_sha256"] = cli.config_fingerprint(parents)
    readjudication = {"schema": "fixture-readjudication", "passed": 1}
    monkeypatch.setattr(cli, "verify_boundary_tangent_parents", lambda **_: parents)
    monkeypatch.setattr(
        cli,
        "build_failed_controller_readjudication",
        lambda value: readjudication if value == parents else None,
    )

    cli._initialize(run_dir, args, resumed=False)
    assert cli._load_json(run_dir / "parent_provenance.json") == parents
    assert cli._load_json(run_dir / "failed_controller_readjudication.json") == readjudication
    assert cli._load_json(run_dir / "run_manifest.json")["device"] == "cpu"

    # Repository history contains the real parent's reserved EC/ED slots.  A
    # hermetic fixture has no verified parent reservation record to consume,
    # so isolate collision-discovery I/O from the preflight unit test.
    def fixture_collision_scan(root: Path, _args: argparse.Namespace) -> dict[str, object]:
        record: dict[str, object] = {
            "schema": "fixture-collision-scan",
            "passed": 1,
        }
        atomic_write_json(root / "path_collision_scan.json", record)
        return record

    monkeypatch.setattr(cli, "_semantic_path_collision_scan", fixture_collision_scan)
    gate = cli._preflight_stage(run_dir, args)
    assert gate["passed"] == 1
    representation = cli._load_json(
        run_dir / "boundary_tangent_representation_preflight.json"
    )
    assert representation["checks"]["logistic_semigroup"] == 1
    assert representation["checks"]["full_flow_orientation"] == 1
    assert representation["checks"]["full_flow_pair_conservation"] == 1
    assert representation["checks"]["full_flow_simplex_conservation"] == 1
    assert representation["checks"]["direct_raw_mse_algebra"] == 1
    assert representation["logistic_semigroup_relative_error"] <= 2.0e-6
    assert cli._load_json(run_dir / "resource_projection.json")["passed"] == 1

    # Resume validation must be read-only and bind source, scientific config,
    # and the frozen path plan.
    manifest_before = (run_dir / "run_manifest.json").read_bytes()
    cli._initialize(run_dir, args, resumed=True)
    assert (run_dir / "run_manifest.json").read_bytes() == manifest_before

    seal_path = run_dir / "preflight_artifact_seal.json"
    sealed = seal_path.read_bytes()
    atomic_write_json(
        seal_path,
        {
            "schema": cli.RUN_SCHEMA + "-preflight-artifact-seal",
            "schema_version": 1,
            "artifacts": [],
            **cli.NO_WORK,
        },
    )
    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._preflight_stage(run_dir, args)
    seal_path.write_bytes(sealed)

    changed_parents = {"schema": "changed-parent-provenance", "passed": 1}
    changed_parents["semantic_sha256"] = cli.config_fingerprint(changed_parents)
    monkeypatch.setattr(
        cli, "verify_boundary_tangent_parents", lambda **_: changed_parents
    )
    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._initialize(run_dir, args, resumed=True)
    assert (run_dir / "run_manifest.json").read_bytes() == manifest_before


def test_collision_scan_consumes_only_hash_bound_parent_reservations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mnist import d0_jacobi_rb_learnability as learnability
    from mnist.d0_jacobi_rb_learnability import PathIDClaim

    args = _args(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(
        run_dir / "path_id_plan.json", cli.build_boundary_tangent_path_plan()
    )
    parent_plan = (args.failed_controller_run_dir / "path_id_plan.json").resolve()
    parent_claims = (
        PathIDClaim(
            str(parent_plan), "reserved_roles.fresh_selection", 0xEC000, 0xEC040
        ),
        PathIDClaim(
            str(parent_plan), "reserved_roles.fresh_confirmation", 0xED000, 0xED040
        ),
    )
    monkeypatch.setattr(
        learnability,
        "discover_repository_path_id_claims",
        lambda _: parent_claims,
    )
    accepted = cli._semantic_path_collision_scan(run_dir, args)
    assert accepted["passed"] == 1
    assert accepted["consumed_parent_reservation_count"] == 2

    distinct = PathIDClaim(
        str(tmp_path / "other-plan.json"), "roles.train", 0xEC100, 0xEC102
    )
    monkeypatch.setattr(
        learnability,
        "discover_repository_path_id_claims",
        lambda _: parent_claims + (distinct,),
    )
    rejected = cli._semantic_path_collision_scan(run_dir, args)
    assert rejected["passed"] == 0
    assert rejected["collision_count"] == 1


def test_reduced_cpu_cache_preserves_order_resume_and_separation(tmp_path: Path) -> None:
    args = _args(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_cache_fixture(run_dir, args)
    path_ids = cli._effective_paths(args)["train"]

    first = cli._generate_role_cache(
        run_dir, args, role="train", path_ids=path_ids
    )
    assert first["recomputed_shard_count"] == 2
    assert first["selected_row_count"] == 2 * 7 * 8
    assert first["certificate_fraction"] == 1.0
    assert first["maximum_mass_error"] == 0.0
    assert all(first[name] == 0 for name in cli.FORBIDDEN_COUNTS)

    inputs, audit = cli._load_role_cache_arrays(
        run_dir, "train", open_labels=True
    )
    assert audit is not None
    assert set(inputs) == {
        "sample_key",
        "path_id",
        "outer_step",
        "midpoint_index",
        "midpoint_fraction",
        *MODEL_INPUT_FIELDS,
    }
    model_inputs = cli._model_inputs_from_arrays(inputs, torch.device("cpu"))
    assert set(vars(model_inputs)) == set(MODEL_INPUT_FIELDS)
    assert set(audit) == {
        "sample_key",
        "path_id",
        "outer_step",
        "phase",
        "midpoint_index",
        "midpoint_fraction",
        "denoising_target",
        "certificate_codes",
    }
    assert inputs["later_full_state"].dtype == np.float32
    assert audit["denoising_target"].dtype == np.float64
    assert inputs["later_full_state"].shape == (112, 784)
    assert audit["denoising_target"].shape == (112, 392)
    coordinates = np.stack(
        [
            audit["path_id"],
            audit["outer_step"],
            audit["phase"],
            audit["midpoint_index"],
        ],
        axis=1,
    )
    expected = coordinates[
        np.lexsort(
            (
                coordinates[:, 3],
                coordinates[:, 2],
                coordinates[:, 1],
                coordinates[:, 0],
            )
        )
    ]
    assert np.array_equal(coordinates, expected)
    assert np.array_equal(inputs["sample_key"], audit["sample_key"])
    assert np.unique(inputs["sample_key"]).size == inputs["sample_key"].size

    second = cli._generate_role_cache(
        run_dir, args, role="train", path_ids=path_ids
    )
    assert second["recomputed_shard_count"] == 0

    # Corrupt the selected tail shard.  Resume must keep the valid prefix and
    # recompute the corrupt tail exactly.
    _, _, label_path, _ = cli._cache_paths(
        run_dir, role="train", cohort_index=0, start_step=8
    )
    label_path.write_bytes(b"corrupt")
    repaired = cli._generate_role_cache(
        run_dir, args, role="train", path_ids=path_ids
    )
    assert repaired["recomputed_shard_count"] == 1
    repaired_inputs, repaired_audit = cli._load_role_cache_arrays(
        run_dir, "train", open_labels=True
    )
    assert repaired_audit is not None
    assert np.array_equal(repaired_inputs["sample_key"], inputs["sample_key"])
    assert np.array_equal(
        repaired_audit["denoising_target"], audit["denoising_target"]
    )


def test_cache_stage_does_not_create_confirmation_evidence(tmp_path: Path) -> None:
    args = _args(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_cache_fixture(run_dir, args)
    atomic_write_json(run_dir / "preflight_gate.json", {"passed": 1})
    gate = cli._cache_stage(run_dir, args)
    assert gate["passed"] == 1
    assert gate["checks"]["confirmation_absent"] == 1
    assert not (run_dir / "cache" / "confirmation_shards").exists()


def test_completed_cache_integrity_rejects_rehashed_configuration_tamper(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_cache_fixture(run_dir, args)
    cli._generate_role_cache(
        run_dir,
        args,
        role="train",
        path_ids=cli._effective_paths(args)["train"],
    )

    *_, metadata_path = cli._cache_paths(
        run_dir, role="train", cohort_index=0, start_step=0
    )
    metadata = cli._load_json(metadata_path)
    metadata["scientific_config_sha256"] = "f" * 64
    metadata_body = dict(metadata)
    metadata_body.pop("semantic_sha256", None)
    metadata["semantic_sha256"] = cli.config_fingerprint(metadata_body)
    atomic_write_json(metadata_path, metadata)

    index_path = run_dir / "cache" / "train_index.json"
    index = cli._load_json(index_path)
    index["shards"][0]["metadata_sha256"] = cli.file_fingerprint(metadata_path)
    index_body = dict(index)
    index_body.pop("semantic_sha256", None)
    index["semantic_sha256"] = cli.config_fingerprint(index_body)
    atomic_write_json(index_path, index)

    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._verify_role_cache_integrity(run_dir, "train")


def test_completed_cache_integrity_rejects_rehashed_selected_step_tamper(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_cache_fixture(run_dir, args)
    cli._generate_role_cache(
        run_dir,
        args,
        role="train",
        path_ids=cli._effective_paths(args)["train"],
    )

    *_, metadata_path = cli._cache_paths(
        run_dir, role="train", cohort_index=0, start_step=8
    )
    metadata = cli._load_json(metadata_path)
    assert metadata["selected_step"] == 15
    metadata["selected_step"] = None
    metadata_body = dict(metadata)
    metadata_body.pop("semantic_sha256", None)
    metadata["semantic_sha256"] = cli.config_fingerprint(metadata_body)
    atomic_write_json(metadata_path, metadata)

    index_path = run_dir / "cache" / "train_index.json"
    index = cli._load_json(index_path)
    selected_item = next(
        item for item in index["shards"] if int(item["start_step"]) == 8
    )
    selected_item["selected_step"] = None
    selected_item["metadata_sha256"] = cli.file_fingerprint(metadata_path)
    index_body = dict(index)
    index_body.pop("semantic_sha256", None)
    index["semantic_sha256"] = cli.config_fingerprint(index_body)
    atomic_write_json(index_path, index)

    with pytest.raises(cli.ArtifactCompatibilityError):
        cli._verify_role_cache_integrity(run_dir, "train")


def _parser_argv(tmp_path: Path) -> list[str]:
    return [
        "--parent-coarse-residual-run-dir",
        str(tmp_path / "coarse"),
        "--failed-controller-run-dir",
        str(tmp_path / "failed"),
    ]


def test_parser_exposes_only_bounded_authorizing_runtime_choices(
    tmp_path: Path,
) -> None:
    report = cli.parse_args(
        _parser_argv(tmp_path) + ["--stage", "report", "--device", "cpu"]
    )
    assert report.stage == "report"
    assert report.require_gate == "none"
    assert report.parent_coarse_residual_run_dir.is_absolute()
    assert report.failed_controller_run_dir.is_absolute()

    with pytest.raises(SystemExit):
        cli.parse_args(
            _parser_argv(tmp_path) + ["--stage", "cache", "--device", "cpu"]
        )
    with pytest.raises(SystemExit):
        cli.parse_args(
            _parser_argv(tmp_path)
            + ["--test-path-count", "4", "--stage", "report"]
        )
    with pytest.raises(SystemExit):
        cli.parse_args(
            _parser_argv(tmp_path)
            + ["--test-only", "--require-gate", "preflight"]
        )
    with pytest.raises(SystemExit):
        cli.parse_args(
            _parser_argv(tmp_path)
            + ["--test-only", "--test-outer-steps", "17"]
        )

    reduced = cli.parse_args(
        _parser_argv(tmp_path)
        + [
            "--test-only",
            "--device",
            "cpu",
            "--test-path-count",
            "4",
            "--test-outer-steps",
            "24",
            "--test-maximum-updates",
            "0",
            "--test-bootstrap-replicates",
            "8",
        ]
    )
    assert reduced.test_only
    assert reduced.require_gate == "none"
    assert reduced.test_path_count == 4


def test_stage_sequence_is_fail_closed_and_report_only_is_empty() -> None:
    assert cli._stage_sequence("all") == (
        "preflight",
        "cache",
        "train",
        "confirm",
        "control",
    )
    assert cli._stage_sequence("report") == ()
    assert cli._stage_sequence("control") == ("control",)
    with pytest.raises(ValueError):
        cli._stage_sequence("sample")


def test_workflow_required_gate_uses_evaluated_gate_and_does_not_materialize_skips(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    preflight = cli._gate(
        "preflight",
        {"fixture": True},
        provenance_valid=1,
        failed_controller_adjudication_valid=1,
        boundary_tangent_representation_valid=1,
    )
    atomic_write_json(run_dir / "preflight_gate.json", preflight)

    passed = cli._workflow_record(run_dir, require_gate="preflight")
    assert passed["required_gate_pass"] == 1
    assert passed["decision"]["decision"] == "ready_for_cache"
    assert cli._load_json(run_dir / "controller_decision.json")["decision"] == (
        "ready_for_cache"
    )
    assert not (run_dir / "cache_gate.json").exists()

    missing = cli._workflow_record(run_dir, require_gate="cache")
    assert missing["required_gate_pass"] == 0
    assert missing["stage_gates"]["cache"]["evaluation_status"] == "not_evaluated"
    assert not (run_dir / "cache_gate.json").exists()


def test_report_revalidates_only_committed_passing_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_write_json(run_dir / "preflight_gate.json", cli._gate("preflight", {"ok": 1}))
    atomic_write_json(run_dir / "cache_gate.json", cli._gate("cache", {"ok": 1}))
    atomic_write_json(run_dir / "train_gate.json", cli._gate("train", {"ok": 0}))
    calls: list[str] = []
    monkeypatch.setattr(
        cli, "_preflight_stage", lambda *_: calls.append("preflight") or {"passed": 1}
    )
    monkeypatch.setattr(
        cli, "_cache_stage", lambda *_: calls.append("cache") or {"passed": 1}
    )
    monkeypatch.setattr(
        cli,
        "_train_stage",
        lambda *_: pytest.fail("report must not rerun a failed train stage"),
    )
    cli._verify_report(run_dir, args)
    assert calls == ["preflight", "cache"]


def test_required_gate_failure_commits_readable_terminal_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.stage = "all"
    args.require_gate = "preflight"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(cli, "_make_run_dir", lambda _: (run_dir, False))
    monkeypatch.setattr(cli, "_initialize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "configure_exact_torch_backend", lambda *_: {})

    def failed_preflight(root: Path, _: argparse.Namespace) -> dict[str, object]:
        gate = cli._gate(
            "preflight",
            {"fixture": False},
            provenance_valid=1,
            failed_controller_adjudication_valid=1,
            boundary_tangent_representation_valid=1,
        )
        atomic_write_json(root / "preflight_gate.json", gate)
        return gate

    monkeypatch.setattr(cli, "_preflight_stage", failed_preflight)
    monkeypatch.setattr(
        cli,
        "_cache_stage",
        lambda *_: pytest.fail("all must stop after the failed preflight"),
    )
    assert cli._run(args) == 2
    assert cli._load_json(run_dir / "run_status.json")["state"] == "gate_failed"
    assert cli._load_json(run_dir / "workflow_gate.json")["required_gate_pass"] == 0
    assert (run_dir / "controller_decision.json").is_file()
    assert (run_dir / "artifact_registry.json").is_file()


def test_rejected_resume_is_byte_for_byte_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.stage = "report"
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    marker = run_dir / "marker.bin"
    marker.write_bytes(b"immutable")
    args.resume_run_dir = run_dir
    before = {path.name: path.read_bytes() for path in run_dir.iterdir()}
    monkeypatch.setattr(
        cli,
        "_initialize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            cli.ArtifactCompatibilityError("source fingerprint changed")
        ),
    )
    assert cli._run(args) == 2
    after = {path.name: path.read_bytes() for path in run_dir.iterdir()}
    assert after == before


def test_execution_failure_gate_preserves_failure_domain() -> None:
    failure = cli.BoundaryTangentCLIError(
        "bad baseline",
        failure_domain="baseline",
        failure_code="fixture_baseline_invalid",
    )
    gate = cli._execution_failed_gate("train", failure)
    assert gate["evaluation_status"] == "execution_failed"
    assert gate["stage_execution_valid"] == 0
    assert gate["scientific_evidence_complete"] == 0
    assert gate["failure_domain"] == "baseline"
    assert gate["failure_code"] == "fixture_baseline_invalid"
    assert gate["boundary_tangent_baseline_valid"] == 0


def test_new_run_initialization_failure_commits_provenance_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.stage = "preflight"
    args.require_gate = "preflight"
    run_dir = tmp_path / "new-run"
    run_dir.mkdir()
    monkeypatch.setattr(cli, "_make_run_dir", lambda _: (run_dir, False))
    monkeypatch.setattr(
        cli,
        "_initialize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            cli.ArtifactCompatibilityError("parent registry changed")
        ),
    )
    assert cli._run(args) == 2
    preflight = cli._load_json(run_dir / "preflight_gate.json")
    assert preflight["evaluation_status"] == "execution_failed"
    assert preflight["provenance_valid"] == 0
    assert cli._load_json(run_dir / "controller_decision.json")["decision"] == (
        "control_provenance_invalid"
    )
    assert cli._load_json(run_dir / "run_status.json")["state"] == (
        "execution_failed"
    )
    assert (run_dir / "artifact_registry.json").is_file()


def test_confirmation_control_audit_checks_phase_pair_mass() -> None:
    one_before = np.full((7, 2, 784), 1.0 / 784.0, dtype=np.float64)
    one_after = one_before.copy()
    eight_before = np.full((2, 784), 1.0 / 784.0, dtype=np.float64)
    eight_after = eight_before.copy()
    pair_error, simplex_error = cli._confirmation_audit_mass_errors(
        one_before, one_after, eight_before, eight_after
    )
    assert pair_error == 0.0
    assert simplex_error <= 3.0e-16

    from mnist.d0_jacobi_rb_learnability import matching_indices

    tails, heads = matching_indices(device="cpu")
    color = cli.PHASE_MATCHINGS[0]
    head = int(heads[color, 0])
    unrelated = int(tails[color, 1])
    one_after[0, 0, head] += 1.0e-6
    one_after[0, 0, unrelated] -= 1.0e-6
    pair_error, simplex_error = cli._confirmation_audit_mass_errors(
        one_before, one_after, eight_before, eight_after
    )
    assert pair_error > 9.0e-7
    assert simplex_error <= 3.0e-16


def test_candidate_loader_rejects_rehashed_wrong_frozen_q(
    tmp_path: Path,
) -> None:
    from mnist.d0_jacobi_rb_boundary_tangent import (
        BoundaryTangentPredictor,
        save_tangent_baseline,
    )

    baseline = cli._zero_baseline((0xEC100, 0xEC101))
    model = BoundaryTangentPredictor(baseline, zero_residual=True)
    state = cli._clone_state_dict(model)
    fingerprint = "a" * 64
    seed = cli.MODEL_SEEDS[0]
    path = (
        tmp_path
        / "checkpoints"
        / "physical"
        / f"seed-{seed}"
        / "update-0100.pt"
    )
    payload = {
        "schema": cli.RUN_SCHEMA + "-candidate",
        "fingerprint": fingerprint,
        "task": "physical",
        "seed": seed,
        "update": 100,
        "state_dict": state,
        "state_sha256": cli.state_dict_sha256(state),
    }
    artifact = cli._atomic_torch(path, payload)
    candidate = {
        "task": "physical",
        "seed": seed,
        "update": 100,
        "training_fingerprint": fingerprint,
        "state_sha256": payload["state_sha256"],
        "checkpoint_path": path.relative_to(tmp_path).as_posix(),
        "checkpoint_file_sha256": artifact["sha256"],
    }
    loaded = cli._load_candidate_model(
        tmp_path, candidate, baseline, torch.device("cpu")
    )
    assert torch.equal(loaded._q_values.cpu(), model._q_values.cpu())

    baseline_artifact = save_tangent_baseline(
        tmp_path / "tangent_baseline.npz", baseline
    )
    selection = {
        "schema": cli.RUN_SCHEMA + "-checkpoint-selection",
        "schema_version": 1,
        "selection_role": "validation_only",
        "task": "physical",
        "selected_seed": seed,
        "selected_update": 100,
        "selected_state_sha256": payload["state_sha256"],
        "training_fingerprint": fingerprint,
        "checkpoint_path": path.relative_to(tmp_path).as_posix(),
        "checkpoint_file_sha256": artifact["sha256"],
        "baseline_path": "tangent_baseline.npz",
        "baseline_file_sha256": baseline_artifact["sha256"],
        "baseline_semantic_sha256": baseline.fingerprint,
        "target_scale": 1.0,
        "confirmation_paths_created": 0,
        "candidate_ranking": (
            "lowest_validation_mse_then_earliest_update_then_lower_seed"
        ),
        **cli.NO_WORK,
    }
    selection["semantic_sha256"] = cli.config_fingerprint(selection)
    atomic_write_json(tmp_path / "checkpoint_selection.json", selection)
    assert cli._verify_training_selection(tmp_path)["selected_update"] == 100

    bad_state = {name: value.clone() for name, value in state.items()}
    bad_state["_q_values"][0, 0, 0] = 1.0
    bad_payload = {
        **payload,
        "state_dict": bad_state,
        "state_sha256": cli.state_dict_sha256(bad_state),
    }
    bad_artifact = cli._atomic_torch(path, bad_payload)
    bad_candidate = {
        **candidate,
        "state_sha256": bad_payload["state_sha256"],
        "checkpoint_file_sha256": bad_artifact["sha256"],
    }
    with pytest.raises(cli.ArtifactCompatibilityError, match="sealed tangent baseline"):
        cli._load_candidate_model(
            tmp_path, bad_candidate, baseline, torch.device("cpu")
        )

    restored_artifact = cli._atomic_torch(path, payload)
    tampered_selection = {
        **selection,
        "checkpoint_file_sha256": restored_artifact["sha256"],
    }
    tampered_selection["selected_state_sha256"] = "f" * 64
    tampered_selection.pop("semantic_sha256")
    tampered_selection["semantic_sha256"] = cli.config_fingerprint(
        tampered_selection
    )
    atomic_write_json(tmp_path / "checkpoint_selection.json", tampered_selection)
    with pytest.raises(cli.ArtifactCompatibilityError, match="state changed"):
        cli._verify_training_selection(tmp_path)


def test_control_phase_checkpoint_rejects_wrong_path_cardinality(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "phase.npz"
    record_path = tmp_path / "phase.json"
    paths = [0xED000, 0xED001]
    state = np.full((2, 784), 1.0 / 784.0, dtype=np.float64)
    artifact = cli._atomic_npz(state_path, {"state": state})
    binding = {
        "schema": cli.RUN_SCHEMA + "-control-phase-checkpoint",
        "schema_version": 1,
        "microsteps": 2,
        "path_ids": paths,
    }
    record = {
        **binding,
        "state_file_sha256": artifact["sha256"],
        "state_file_size": artifact["size"],
        "state_array_sha256": cli._array_sha(state),
        "reference_diagnostics": {
            "transition_count": len(paths) * 392 * 2 * 2,
            "certified_count": len(paths) * 392 * 2 * 2,
            "fallback_count": 0,
            "fallback_seconds": 0.0,
            "elapsed_seconds": 1.0,
            "maximum_transition_count_per_call": 784,
            "forbidden_counts": {
                name: 0
                for name in cli.FORBIDDEN_COUNTS
                if name != "uncertified_count"
            },
        },
        "maximum_pair_mass_error": 0.0,
        "maximum_simplex_mass_error": 0.0,
        "states_finite": 1,
        "states_nonnegative": 1,
        "boundary_rejection_count": 0,
        "controller_forbidden_counts": {
            "clip_count": 0,
            "floor_count": 0,
            "limiter_count": 0,
            "projection_count": 0,
            "renormalization_count": 0,
        },
        "committed": 1,
    }
    record["semantic_sha256"] = cli.config_fingerprint(record)
    atomic_write_json(record_path, record)
    assert cli._valid_control_phase_checkpoint(
        state_path, record_path, binding=binding
    ) is not None

    short = state[:1].copy()
    artifact = cli._atomic_npz(state_path, {"state": short})
    record["state_file_sha256"] = artifact["sha256"]
    record["state_file_size"] = artifact["size"]
    record["state_array_sha256"] = cli._array_sha(short)
    record.pop("semantic_sha256")
    record["semantic_sha256"] = cli.config_fingerprint(record)
    atomic_write_json(record_path, record)
    assert (
        cli._valid_control_phase_checkpoint(
            state_path, record_path, binding=binding
        )
        is None
    )
