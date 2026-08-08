from __future__ import annotations

import copy
from pathlib import Path

import pytest

from mnist.d0_jacobi_artifacts import file_fingerprint
from mnist.d0_jacobi_rb_haar_gate import NESTED_HAAR_PROFILE
from mnist.d0_jacobi_rb_haar_power_recovery import (
    EXPECTED_NESTED_CONSERVATIVE_RATE,
    EXPECTED_NESTED_FALLBACKS,
    EXPECTED_NESTED_FALLBACK_COST_FRACTION,
    EXPECTED_NESTED_MASS_ERROR,
    EXPECTED_NESTED_TRANSITIONS,
    FROZEN_PARENT_REGISTRY_SHA256,
    HaarPowerRecoveryError,
    canonical_schedule,
    replay_nested_panel_a,
    run_recovery_antithetic_panel,
)


REPOSITORY = Path(__file__).resolve().parents[1]
PARENT = (
    REPOSITORY
    / "runs"
    / "experiment12_d0_jacobi_rb_hierarchical_coupling_confirmation"
    / "20260725-212650_production-certified-haar-strang-power-adapter-fix-v2"
)


def _nested_record() -> dict:
    schedule = {
        "profile_name": "nested_haar_single_arm",
        "pool": "main",
        "role": "nested_a",
        "levels": [128, 256, 512, 1024],
        "coarsest_steps": 128,
        "finest_steps": 1024,
        "single_arm": 1,
    }
    return {"identity": {"schedule": schedule}}


def test_canonical_schedule_reads_identity_and_accepts_only_equal_alias() -> None:
    record = _nested_record()
    expected = copy.deepcopy(record["identity"]["schedule"])
    assert canonical_schedule(
        record,
        expected_profile=NESTED_HAAR_PROFILE,
        expected_pool="main",
        expected_role="nested_a",
    ) == expected

    record["schedule"] = copy.deepcopy(expected)
    assert canonical_schedule(record) == expected

    record["schedule"]["levels"][-1] = 2048
    with pytest.raises(
        HaarPowerRecoveryError,
        match="top-level schedule conflicts",
    ) as caught:
        canonical_schedule(record)
    assert caught.value.failure_code == "panel_schedule_binding_invalid"


@pytest.mark.parametrize(
    "mutation",
    (
        lambda record: record.pop("identity"),
        lambda record: record["identity"].pop("schedule"),
        lambda record: record["identity"]["schedule"].update(
            profile_name="pairwise_haar_antithetic"
        ),
        lambda record: record["identity"]["schedule"].update(
            levels=[128, 256, 512]
        ),
    ),
)
def test_canonical_schedule_fails_closed_on_missing_or_malformed_binding(
    mutation,
) -> None:
    record = _nested_record()
    mutation(record)
    with pytest.raises(HaarPowerRecoveryError) as caught:
        canonical_schedule(record)
    assert caught.value.failure_code == "panel_schedule_binding_invalid"


def test_canonical_schedule_rejects_profile_mismatch() -> None:
    with pytest.raises(HaarPowerRecoveryError) as caught:
        canonical_schedule(
            _nested_record(),
            expected_profile="pairwise_haar_antithetic",
        )
    assert caught.value.failure_code == "panel_schedule_binding_invalid"


@pytest.mark.skipif(not PARENT.is_dir(), reason="sealed production run unavailable")
def test_exact_nested_parent_replay_is_read_only_and_reconstructs_candidates() -> None:
    shard_root = (
        PARENT
        / "haar_power_shards"
        / "nested_haar_single_arm"
        / "a"
    )
    files = sorted(
        (
            path
            for path in shard_root.rglob("*")
            if path.is_file() and path.suffix in {".json", ".npz"}
        ),
        key=lambda path: path.as_posix(),
    )
    before = {
        path: (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            file_fingerprint(path),
        )
        for path in files
    }
    registry_before = file_fingerprint(PARENT / "artifact_registry.json")

    replay = replay_nested_panel_a(PARENT)

    after = {
        path: (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            file_fingerprint(path),
        )
        for path in files
    }
    assert before == after
    assert (
        registry_before
        == file_fingerprint(PARENT / "artifact_registry.json")
        == FROZEN_PARENT_REGISTRY_SHA256
    )
    assert len(files) == 160
    assert replay["schedule_binding_count"] == 80
    assert replay["shard_audit"] == {
        "main_shard_count": 16,
        "reference_shard_count": 64,
        "total_shard_count": 80,
        "registry_hash_pass": 1,
        "archive_hash_pass": 1,
        "predecessor_chain_pass": 1,
        "parent_mutated": 0,
        "nested_gpu_recomputation_performed": 0,
    }
    assert all(
        binding["schedule_source"] == "identity.schedule"
        for binding in replay["schedule_bindings"]
    )

    execution = replay["execution"]
    assert execution["transition_count"] == EXPECTED_NESTED_TRANSITIONS
    assert execution["certified_count"] == EXPECTED_NESTED_TRANSITIONS
    assert execution["certificate_fraction"] == 1.0
    assert execution["fallback_count"] == EXPECTED_NESTED_FALLBACKS
    assert execution["fallback_cost_fraction"] == pytest.approx(
        EXPECTED_NESTED_FALLBACK_COST_FRACTION,
        abs=1.0e-15,
    )
    assert execution["mass_error"] == EXPECTED_NESTED_MASS_ERROR
    assert execution["conservative_rate"] == pytest.approx(
        EXPECTED_NESTED_CONSERVATIVE_RATE,
        abs=1.0e-9,
    )
    assert execution["peak_memory_fraction"] == 0.006818991116410677
    assert sum(execution[name] for name in (
        "uncertified_count",
        "resource_cap_count",
        "invalid_density_count",
        "approximation_count",
        "correction_count",
        "floor_count",
        "limiter_count",
        "projection_count",
        "renormalization_count",
        "nonfinite_count",
    )) == 0

    assert replay["recovery_decision"] == "panel_a_no_eligible_design"
    assert replay["nomination"]["selected"] is None
    assert replay["nomination"]["eligible_candidate_count"] == 0
    widths = [
        (
            row["main_paths"],
            row["reference_paths"],
            row["predicted_main_half_width"],
            row["predicted_generator_reference_half_width"],
            row["predicted_reference_stability_half_width"],
        )
        for row in replay["candidates"]
    ]
    assert widths == pytest.approx(
        [
            (32, 16, 6.838086287284076, 9.440360137912025, 7.7704410333538565),
            (32, 32, 6.838086287284076, 7.792741042926053, 5.655631466198558),
            (64, 16, 4.835257184077312, 8.541298867157927, 7.653983505194456),
            (64, 32, 4.835257184077312, 6.675342670360765, 5.4945315474947165),
        ],
        abs=1.0e-12,
    )
    assert {
        name: value.shape
        for name, value in replay["observable_arrays"].items()
    } == {
        "raw_main_differences": (8, 120),
        "raw_d3_differences": (8, 40),
        "raw_d4_differences": (8, 40),
        "dynkin_main_differences": (8, 120),
        "dynkin_d3_differences": (8, 40),
        "dynkin_d4_differences": (8, 40),
    }


@pytest.mark.skipif(not PARENT.is_dir(), reason="sealed production run unavailable")
def test_antithetic_wrapper_refuses_to_write_inside_parent() -> None:
    with pytest.raises(HaarPowerRecoveryError) as caught:
        run_recovery_antithetic_panel(
            run_dir=PARENT / "forbidden-recovery-output",
            parent_haar_run_dir=PARENT,
            panel="a",
            device="cuda",
        )
    assert caught.value.failure_code == "control_provenance_invalid"
    assert not (PARENT / "forbidden-recovery-output").exists()
