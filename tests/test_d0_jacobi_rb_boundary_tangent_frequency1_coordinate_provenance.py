from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnist.d0_jacobi_artifacts import config_fingerprint
from mnist.d0_jacobi_rb_boundary_tangent_frequency1_coordinate_provenance import (
    ABSOLUTE_PARENT_BASENAME,
    ABSOLUTE_PARENT_REGISTRY_FILE_SHA256,
    ABSOLUTE_PARENT_REGISTRY_SEMANTIC_SHA256,
    CONFIRMATION_BOOTSTRAP_NAMESPACE,
    CONFIRMATION_BOOTSTRAP_SEED,
    FrequencyOneCoordinateProvenanceError,
    PATH_ID_LIMIT,
    PATH_ROLE_RANGES,
    PHYSICAL_MODEL_SEEDS,
    ROOT_SEED,
    SELECTION_BOOTSTRAP_NAMESPACE,
    SELECTION_BOOTSTRAP_SEED,
    TRAIN_VALIDATION_COHORT_SIZES,
    build_cohort_plan,
    build_path_id_plan,
    build_seed_plan,
    frequency1_coordinate_source_fingerprint,
    frequency1_coordinate_source_paths,
    scan_historical_path_seed_claims,
    validate_cohort_plan,
    validate_path_id_plan,
    validate_role_open_order,
    validate_seed_plan,
    validate_semantic_config,
    verify_absolute_coordinate_design_parent,
    verify_resume_compatibility,
    _snapshot_tree,
)


def test_frozen_parent_path_seed_and_namespace_constants() -> None:
    assert ABSOLUTE_PARENT_BASENAME.startswith("20260810-211949")
    assert ABSOLUTE_PARENT_REGISTRY_SEMANTIC_SHA256 == (
        "77f7245cd52b8d4e210f75ceb17f2bf3c0b923bbbabde26173e9409a0d6d9218"
    )
    assert ABSOLUTE_PARENT_REGISTRY_FILE_SHA256 == (
        "d05804f468c6485c234a6f6f66d55e3c3075e85a9172efbd1af2ab5654a84158"
    )
    assert PATH_ROLE_RANGES == {
        "preflight_seam": (0xF8000, 0xF8008),
        "training": (0xF8100, 0xF8140),
        "validation": (0xF8200, 0xF8220),
        "confirmation": (0xF9000, 0xF9040),
    }
    assert ROOT_SEED == 261371
    assert PHYSICAL_MODEL_SEEDS == (261372, 261373, 261374)
    assert (SELECTION_BOOTSTRAP_SEED, SELECTION_BOOTSTRAP_NAMESPACE) == (
        261380,
        0x46435631,
    )
    assert (CONFIRMATION_BOOTSTRAP_SEED, CONFIRMATION_BOOTSTRAP_NAMESPACE) == (
        261382,
        0x46434331,
    )


def test_path_plan_exact_ranges_bounds_and_collision_scan() -> None:
    plan = build_path_id_plan()
    result = validate_path_id_plan(plan)
    assert result["passed"] == 1
    assert result["active_path_count"] == 8 + 64 + 32 + 64
    for role, (start, stop) in PATH_ROLE_RANGES.items():
        assert plan["roles"][role] == list(range(start, stop))
        assert 0 <= start < stop <= PATH_ID_LIMIT

    with pytest.raises(FrequencyOneCoordinateProvenanceError, match="collision"):
        validate_path_id_plan(plan, claimed_ids={"historical": [0xF8100]})
    with pytest.raises(FrequencyOneCoordinateProvenanceError, match="20-bit"):
        validate_path_id_plan(plan, claimed_ids={"bad": [PATH_ID_LIMIT]})
    with pytest.raises(FrequencyOneCoordinateProvenanceError, match="integer"):
        validate_path_id_plan(plan, claimed_ids={"bad": [True]})


def test_cohort_plan_freezes_mixed_seventh_cohort_and_confirmation_tail() -> None:
    paths = build_path_id_plan()
    cohorts = build_cohort_plan(paths)
    assert validate_cohort_plan(cohorts, path_plan=paths)["passed"] == 1
    assert tuple(map(len, cohorts["train_validation"])) == TRAIN_VALIDATION_COHORT_SIZES
    assert tuple(map(len, cohorts["confirmation"])) == (10, 10, 10, 10, 10, 10, 4)
    assert cohorts["cross_role_cohort_indices"] == [6]
    assert cohorts["train_validation"][6][:4] == paths["roles"]["training"][-4:]
    assert cohorts["train_validation"][6][4:] == paths["roles"]["validation"][:6]


def test_seed_plan_and_role_opening_are_fail_closed() -> None:
    seeds = build_seed_plan()
    assert validate_seed_plan(seeds)["passed"] == 1
    with pytest.raises(FrequencyOneCoordinateProvenanceError, match="history"):
        validate_seed_plan(seeds, historical_seeds=[ROOT_SEED])
    assert validate_role_open_order(())["next_role"] == "training"
    assert validate_role_open_order(("training", "validation"))["next_role"] == "confirmation"
    with pytest.raises(FrequencyOneCoordinateProvenanceError, match="skipped"):
        validate_role_open_order(("training", "confirmation"))
    with pytest.raises(FrequencyOneCoordinateProvenanceError, match="prelabel"):
        validate_role_open_order(
            ("training",), prerequisite_flags={"prelabel_controls_passed": 0}
        )


def test_recursive_historical_allocation_scan_finds_named_claims(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "path_id_plan.json").write_text(
        json.dumps({"roles": {"old": [10, 11]}, "role_slots": {"burned": {"start": 20, "stop_exclusive": 22}}}),
        encoding="utf-8",
    )
    (parent / "seed_plan.json").write_text(
        json.dumps({"root_seed": 123, "selection_namespace": 456}),
        encoding="utf-8",
    )
    scan = scan_historical_path_seed_claims(run_dirs=[parent])
    assert scan["path_ids"] == [10, 11, 20, 21]
    assert scan["seeds"] == [123]
    assert scan["namespaces"] == [456]


def test_parent_tree_snapshot_rejects_symlink_escape(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside immutable parent")
    link = parent / "escape.bin"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks are unavailable on this host")
    with pytest.raises(FrequencyOneCoordinateProvenanceError, match="escapes"):
        _snapshot_tree(parent, role="test")


def test_source_config_and_resume_helpers(tmp_path: Path) -> None:
    entries = (
        Path("mnist/d0_jacobi_rb_boundary_tangent_frequency1_coordinate_provenance.py"),
        Path("mnist/d0_jacobi_rb_boundary_tangent_frequency1_coordinate_gate.py"),
    )
    paths = frequency1_coordinate_source_paths(entries)
    assert all(path.is_file() for path in paths)
    assert frequency1_coordinate_source_fingerprint(entries) == (
        frequency1_coordinate_source_fingerprint(entries)
    )
    default_names = {path.name for path in frequency1_coordinate_source_paths()}
    assert {
        "d0_jacobi_rb_boundary_tangent_frequency1_coordinate.py",
        "d0_jacobi_rb_boundary_tangent_frequency1_coordinate_provenance.py",
        "d0_jacobi_rb_boundary_tangent_frequency1_coordinate_gate.py",
        "d0_jacobi_rb_boundary_tangent_frequency1_coordinate_learnability.py",
        "diag_d0_jacobi_rb_boundary_tangent_frequency1_coordinate_learnability.py",
    } <= default_names

    config = {"schema": "frequency1-test-config", "schema_version": 1}
    config["semantic_sha256"] = config_fingerprint(config)
    assert validate_semantic_config(
        config,
        expected_schema="frequency1-test-config",
        expected_sha256=config["semantic_sha256"],
    )["passed"] == 1

    run = tmp_path / "resume"
    run.mkdir()
    (run / "run_manifest.json").write_text(
        json.dumps({"source_fingerprint": "abc", "scientific_config_sha256": "def"}),
        encoding="utf-8",
    )
    (run / "path_id_plan.json").write_text(json.dumps(build_path_id_plan()), encoding="utf-8")
    assert verify_resume_compatibility(
        run,
        expected_bindings={"source_fingerprint": "abc"},
        artifact_bindings={
            "path_id_plan.json": build_path_id_plan()["semantic_sha256"]
        },
    )["passed"] == 1
    with pytest.raises(FrequencyOneCoordinateProvenanceError, match="manifest"):
        verify_resume_compatibility(run, expected_bindings={"source_fingerprint": "changed"})
    with pytest.raises(FrequencyOneCoordinateProvenanceError, match="unsafe|escapes"):
        verify_resume_compatibility(
            run,
            expected_bindings={"source_fingerprint": "abc"},
            artifact_bindings={"../outside.json": "irrelevant"},
        )


def test_real_absolute_coordinate_design_parent_when_available() -> None:
    parent = (
        Path("runs/experiment12_d0_jacobi_rb_absolute_coordinate_adjudication")
        / ABSOLUTE_PARENT_BASENAME
    )
    if not parent.is_dir():
        pytest.skip("exact absolute-coordinate parent is not installed")
    result = verify_absolute_coordinate_design_parent(parent)
    assert result["passed"] == 1
    assert result["all_registered_artifact_hashes_verified"] == 1
    assert result["frequency1_lower_bounds"] == [
        0.0006381522766084302,
        0.00028943218513485374,
        0.0001397448905623646,
        0.00005043911793781375,
    ]
