from __future__ import annotations

from pathlib import Path
import json

import pytest

from mnist.d0_jacobi_artifacts import config_fingerprint, file_fingerprint
from mnist.d0_jacobi_rb_boundary_tangent_quartile_provenance import (
    BAYES_PARENT_BASENAME,
    MEMORY_PARENT_BASENAME,
    PATH_ID_LIMIT,
    PATH_ROLE_RANGES,
    PHYSICAL_MODEL_SEEDS,
    ROLE_OPEN_ORDER,
    TIME_LOCAL_PARENT_BASENAME,
    WITNESS_PARENT_BASENAME,
    QuartileSpecialistProvenanceError,
    build_cohort_plan,
    build_path_id_plan,
    build_role_firewall,
    build_seed_plan,
    quartile_source_fingerprint,
    quartile_source_paths,
    validate_cohort_plan,
    validate_path_id_plan,
    validate_role_open_order,
    validate_role_firewall,
    validate_seed_plan,
    validate_semantic_config,
    verify_quartile_specialist_parents,
    verify_resume_compatibility,
)


def _parents() -> tuple[Path, Path, Path, Path]:
    time_local = (
        Path(
            "runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_time_local_adjudication"
        )
        / TIME_LOCAL_PARENT_BASENAME
    ).resolve()
    memory = (
        Path("runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_memory_confirmation")
        / MEMORY_PARENT_BASENAME
    ).resolve()
    witness = (
        Path("runs/experiment12_d0_jacobi_rb_physical_coarse_signal_witness")
        / WITNESS_PARENT_BASENAME
    ).resolve()
    bayes = (
        Path("runs/experiment12_d0_jacobi_rb_bayes_power_calibration")
        / BAYES_PARENT_BASENAME
    ).resolve()
    return time_local, memory, witness, bayes


def test_exact_four_parent_chain_verifies_without_mutation() -> None:
    time_local, memory, witness, bayes = _parents()
    protected = {
        path: file_fingerprint(path)
        for path in (
            time_local / "artifact_registry.json",
            time_local / "time_local_adjudication_decision.json",
            memory / "artifact_registry.json",
            witness / "artifact_registry.json",
            bayes / "artifact_registry.json",
        )
    }
    result = verify_quartile_specialist_parents(
        time_local_run_dir=time_local,
        memory_v3_run_dir=memory,
        coarse_witness_run_dir=witness,
        bayes_power_run_dir=bayes,
        verify_external_cache=False,
    )
    assert result["passed"] == 1
    assert result["all_four_parent_registries_verified"] == 1
    assert result["all_registered_artifact_hashes_verified"] == 1
    assert result["all_checkpoint_hashes_verified"] == 1
    assert result["confirmation_namespace_opened"] == 0
    assert result["historical_design_evidence_authorizing"] == 0
    assert result["historical_gain_or_checkpoint_reuse_authorized"] == 0
    assert set(result["transitive_parents"]) == {
        "memory_safe_v3_selection",
        "physical_coarse_signal_witness",
        "bayes_power_calibration",
    }
    assert protected == {path: file_fingerprint(path) for path in protected}


def test_wrong_authoritative_parent_is_rejected(tmp_path: Path) -> None:
    _, memory, witness, bayes = _parents()
    wrong = tmp_path / "wrong-time-local-parent"
    wrong.mkdir()
    with pytest.raises(QuartileSpecialistProvenanceError, match="basename"):
        verify_quartile_specialist_parents(
            time_local_run_dir=wrong,
            memory_v3_run_dir=memory,
            coarse_witness_run_dir=witness,
            bayes_power_run_dir=bayes,
            verify_external_cache=False,
        )


def test_fresh_path_plan_exact_ranges_and_collision_scan() -> None:
    plan = build_path_id_plan()
    result = validate_path_id_plan(
        plan,
        claimed_ids={"enclosing_reservation": (0xF0000, 0x100000)},
    )
    assert result["passed"] == 1
    assert result["active_path_count"] == 8 + 64 + 32 + 32 + 384 + 384
    assert result["allowed_allocator_claims"] == ["enclosing_reservation"]
    for role, (start, stop) in PATH_ROLE_RANGES.items():
        assert plan["roles"][role] == list(range(start, stop))
        assert 0 <= start < stop <= PATH_ID_LIMIT

    with pytest.raises(QuartileSpecialistProvenanceError, match="collision"):
        validate_path_id_plan(plan, claimed_ids={"other_run": [0xF5000]})
    with pytest.raises(QuartileSpecialistProvenanceError, match="20-bit"):
        validate_path_id_plan(plan, claimed_ids={"other_run": [PATH_ID_LIMIT]})
    with pytest.raises(QuartileSpecialistProvenanceError, match="integer"):
        validate_path_id_plan(plan, claimed_ids={"other_run": [True]})


def test_cohort_plan_has_exact_role_local_p10_tails() -> None:
    path_plan = build_path_id_plan()
    cohort_plan = build_cohort_plan(path_plan)
    result = validate_cohort_plan(cohort_plan, path_plan=path_plan)
    assert result["passed"] == 1
    assert result["cross_role_cohorts"] == 0
    assert [row["size"] for row in cohort_plan["roles"]["physical_fit"]] == (
        [10] * 6 + [4]
    )
    assert [row["size"] for row in cohort_plan["roles"]["gain_calibration"]] == (
        [10] * 3 + [2]
    )
    assert [row["size"] for row in cohort_plan["roles"]["fresh_selection"]] == (
        [10] * 38 + [4]
    )
    assert len(cohort_plan["roles"]["untouched_confirmation"]) == 39
    for role, rows in cohort_plan["roles"].items():
        assert all(row["role"] == role and row["size"] <= 10 for row in rows)


def test_seed_plan_is_exact_unique_and_role_order_is_fail_closed() -> None:
    seed_plan = build_seed_plan()
    assert validate_seed_plan(seed_plan)["passed"] == 1
    assert seed_plan["physical_model_seeds"] == {
        f"q{quartile}": list(seeds)
        for quartile, seeds in PHYSICAL_MODEL_SEEDS.items()
    }
    firewall = build_role_firewall()
    assert firewall["role_open_order"] == list(ROLE_OPEN_ORDER)
    assert validate_role_firewall(firewall)["passed"] == 1
    assert validate_role_open_order(())["next_role"] == "physical_fit"
    assert validate_role_open_order(ROLE_OPEN_ORDER[:3])["next_role"] == (
        "fresh_selection"
    )
    with pytest.raises(QuartileSpecialistProvenanceError, match="skipped"):
        validate_role_open_order(("physical_fit", "training_rank"))
    with pytest.raises(QuartileSpecialistProvenanceError, match="more than once"):
        validate_role_open_order(("physical_fit", "physical_fit"))
    with pytest.raises(QuartileSpecialistProvenanceError, match="prelabel_controls"):
        validate_role_open_order(
            ("physical_fit",), prerequisite_flags={"prelabel_controls_passed": 0}
        )
    checked = validate_role_open_order(
        ROLE_OPEN_ORDER[:3],
        prerequisite_flags={
            "prelabel_controls_passed": 1,
            "physical_training_complete": 1,
            "gain_calibration_seal_exists": 1,
        },
    )
    assert checked["prerequisite_flags_checked"] == 1


def test_source_config_and_resume_compatibility_helpers(tmp_path: Path) -> None:
    entries = (
        Path("mnist/d0_jacobi_rb_boundary_tangent_quartile_provenance.py"),
        Path("mnist/d0_jacobi_rb_boundary_tangent_quartile_gate.py"),
    )
    paths = quartile_source_paths(entries)
    assert all(path.is_file() for path in paths)
    assert quartile_source_fingerprint(entries) == quartile_source_fingerprint(entries)

    config = {
        "schema": "quartile-test-config",
        "schema_version": 1,
        "root_seed": 261331,
    }
    config["semantic_sha256"] = config_fingerprint(config)
    assert validate_semantic_config(
        config,
        expected_schema="quartile-test-config",
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
        expected_bindings={
            "source_fingerprint": "abc",
            "scientific_config_sha256": "def",
        },
        artifact_bindings={
            "path_id_plan.json": build_path_id_plan()["semantic_sha256"]
        },
    )["passed"] == 1
    with pytest.raises(QuartileSpecialistProvenanceError, match="manifest"):
        verify_resume_compatibility(
            run, expected_bindings={"source_fingerprint": "changed"}
        )
