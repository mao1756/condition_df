from __future__ import annotations

import math

import numpy as np
import pytest

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    file_fingerprint,
)
from mnist.d0_jacobi_rb_haar_controls import (
    _structural_marginal_profiles,
    _whole_cluster_max_t_zero_pass,
    _whole_path_means,
)
from mnist.d0_jacobi_rb_haar_gate import NESTED_HAAR_PROFILE
from mnist.d0_jacobi_rb_haar_power import (
    FORBIDDEN_COUNTS,
    POWER_MEAN_FAMILY_ERROR,
    POWER_TOTAL_FAMILY_ERROR,
    POWER_VARIANCE_FAMILY_ERROR,
    _array_hash,
    _discard_corrupt_tail,
    _execution_record,
    _panel_evidence_flags,
    _variance_upper,
    build_power_candidates,
    combine_certified_haar_power_panels,
    verify_certified_haar_power_panel_evidence,
)


def _diagnostics(*, transitions: int = 100) -> dict[str, object]:
    return {
        "transition_count": transitions,
        "certificate_fraction": 1.0,
        "fallback_count": 0,
        "fallback_elapsed_seconds": 0.0,
        "mass_error": 0.0,
        "state_updates_device_resident_pass": 1,
        **{name: 0 for name in FORBIDDEN_COUNTS},
    }


def _record(
    schedule: str,
    *,
    seconds: float,
    transitions: int = 100,
) -> dict[str, object]:
    return {
        "schedule": {"profile_name": schedule},
        "diagnostics": _diagnostics(transitions=transitions),
        "timing": {
            "complete_pipeline_including_state_shard_io_seconds": seconds
        },
    }


def test_whole_path_means_uses_paths_not_edges_as_sampling_units() -> None:
    values = np.arange(24, dtype=np.float64).reshape(12, 2)
    result = _whole_path_means(
        values,
        path_count=3,
        values_per_path=4,
    )
    assert result.shape == (3, 2)
    np.testing.assert_array_equal(
        result,
        values.reshape(3, 4, 2).mean(axis=1),
    )


def test_whole_cluster_max_t_is_deterministic_and_rejects_large_bias() -> None:
    rng = np.random.default_rng(4)
    centered = rng.normal(size=(32, 6))
    centered -= centered.mean(axis=0, keepdims=True)
    first = _whole_cluster_max_t_zero_pass(
        centered,
        family_error=0.01,
        bootstrap_seed=17,
        replicates=2_000,
    )
    second = _whole_cluster_max_t_zero_pass(
        centered,
        family_error=0.01,
        bootstrap_seed=17,
        replicates=2_000,
    )
    assert first == second
    assert first["passed"] == 1

    biased = centered.copy()
    biased[:, 2] += 10.0
    rejected = _whole_cluster_max_t_zero_pass(
        biased,
        family_error=0.01,
        bootstrap_seed=17,
        replicates=2_000,
    )
    assert rejected["passed"] == 0


def test_structural_marginal_profiles_cover_every_frozen_branch() -> None:
    profiles = _structural_marginal_profiles()
    assert len(profiles) == 19
    assert {
        (row["pool"], row["sample_steps"])
        for row in profiles
        if row["profile"] == NESTED_HAAR_PROFILE
    } == {
        ("main", 128),
        ("main", 256),
        ("main", 512),
        ("main", 1024),
        ("reference", 512),
        ("reference", 1024),
        ("reference", 2048),
    }
    assert sum(
        row.get("branch") == "fine_minus" and row["detail_sign"] == -1
        for row in profiles
    ) == 4


def test_execution_rate_is_conservative_across_complete_schedules() -> None:
    evidence = _execution_record(
        (
            _record("nested-main", seconds=1.0),
            _record("nested-main", seconds=1.0),
            _record("nested-reference", seconds=2.0),
        ),
        peak_memory_fraction=0.25,
    )
    assert evidence["aggregate_rate"] == pytest.approx(75.0)
    assert evidence["slowest_schedule_rate"] == pytest.approx(50.0)
    assert evidence["conservative_rate"] == pytest.approx(50.0)


def test_power_width_splits_alpha_for_joint_familywise_coverage() -> None:
    from scipy import stats

    samples = np.arange(24, dtype=np.float64).reshape(8, 3)
    upper, critical = _variance_upper(samples, family_size=3)
    variance = np.var(samples, axis=0, ddof=1)
    chi_square_lower = stats.chi2.ppf(
        POWER_VARIANCE_FAMILY_ERROR / 3.0,
        samples.shape[0] - 1,
    )
    expected_upper = (
        (samples.shape[0] - 1) * variance / chi_square_lower
    )
    expected_critical = math.sqrt(
        2.0 * math.log(2.0 * 3.0 / POWER_MEAN_FAMILY_ERROR)
    )

    np.testing.assert_allclose(upper, expected_upper, rtol=0.0, atol=0.0)
    assert critical == pytest.approx(expected_critical)
    assert (
        POWER_VARIANCE_FAMILY_ERROR + POWER_MEAN_FAMILY_ERROR
        == POWER_TOTAL_FAMILY_ERROR
    )
    assert 1.0 - POWER_TOTAL_FAMILY_ERROR == 0.99


def test_panel_health_requires_measured_timing_and_device_residency() -> None:
    execution = {
        **_diagnostics(),
        "fallback_fraction": 0.0,
        "fallback_cost_fraction": 0.0,
        "peak_memory_fraction": 0.1,
        "shard_chain_pass": 1,
        "timing_coverage_pass": 0,
    }
    flags = _panel_evidence_flags(execution)
    assert flags["panel_numerical_health_pass"] == 0

    execution["timing_coverage_pass"] = 1
    execution["state_updates_device_resident_pass"] = 0
    flags = _panel_evidence_flags(execution)
    assert flags["panel_numerical_health_pass"] == 0

    execution["state_updates_device_resident_pass"] = 1
    flags = _panel_evidence_flags(execution)
    assert flags["panel_numerical_health_pass"] == 1


def _panel(role: str, ids: list[int]) -> dict[str, object]:
    return {
        "profile": NESTED_HAAR_PROFILE,
        "panel": role,
        "root_seed": 261181,
        "path_id_plan_sha256": "plan",
        "source_npz_sha256": "source",
        "cluster_count": 8,
        "path_id_pools": {"main": ids},
    }


def test_combining_panels_rejects_overlapping_sealed_path_ids(
    tmp_path,
) -> None:
    panel_a = _panel("a", [1, 2])
    panel_b = _panel("b", [2, 3])
    with pytest.raises(
        ArtifactCompatibilityError,
        match="path IDs overlap",
    ):
        combine_certified_haar_power_panels(
            run_dir=tmp_path,
            profile=NESTED_HAAR_PROFILE,
            selected={
                "profile": NESTED_HAAR_PROFILE,
                "main_paths": 32,
                "reference_paths": 16,
            },
            panel_a=panel_a,
            panel_b=panel_b,
        )


def test_combining_panels_rejects_profile_mismatch_before_payload_read(
    tmp_path,
) -> None:
    panel_a = _panel("a", [1, 2])
    panel_b = _panel("b", [3, 4])
    panel_b["profile"] = "pairwise_haar_antithetic"
    with pytest.raises(ArtifactCompatibilityError, match="profiles differ"):
        combine_certified_haar_power_panels(
            run_dir=tmp_path,
            profile=NESTED_HAAR_PROFILE,
            selected={
                "profile": NESTED_HAAR_PROFILE,
                "main_paths": 32,
                "reference_paths": 16,
            },
            panel_a=panel_a,
            panel_b=panel_b,
        )


def test_missing_forbidden_event_counter_fails_closed() -> None:
    record = _record("nested-main", seconds=1.0)
    del record["diagnostics"]["projection_count"]  # type: ignore[index]
    with pytest.raises(
        Exception,
        match="forbidden-event diagnostics are incomplete",
    ):
        _execution_record((record,), peak_memory_fraction=0.1)


def test_missing_counter_cannot_cancel_a_positive_counter() -> None:
    missing = _record("nested-main", seconds=1.0)
    positive = _record("nested-main", seconds=1.0)
    del missing["diagnostics"]["projection_count"]  # type: ignore[index]
    positive["diagnostics"]["projection_count"] = 1  # type: ignore[index]
    with pytest.raises(Exception, match="diagnostics are incomplete"):
        _execution_record(
            (missing, positive),
            peak_memory_fraction=0.1,
        )


def test_nonfinite_timing_fails_closed() -> None:
    with pytest.raises(Exception, match="diagnostics are incomplete"):
        _execution_record(
            (_record("nested-main", seconds=math.inf),),
            peak_memory_fraction=0.1,
        )


def _valid_sealed_evidence(tmp_path) -> dict[str, object]:
    cluster_count = 8
    arrays = {
        "raw_main_differences": np.zeros((cluster_count, 120)),
        "raw_d3_differences": np.zeros((cluster_count, 40)),
        "raw_d4_differences": np.zeros((cluster_count, 40)),
        "dynkin_main_differences": np.zeros((cluster_count, 120)),
        "dynkin_d3_differences": np.zeros((cluster_count, 40)),
        "dynkin_d4_differences": np.zeros((cluster_count, 40)),
    }
    payload_path = (
        tmp_path / f"{NESTED_HAAR_PROFILE}_panel_a_observables.npz"
    )
    np.savez_compressed(payload_path, **arrays)
    execution = _execution_record(
        (_record("nested-main", seconds=1.0, transitions=100),),
        peak_memory_fraction=0.1,
    )
    execution.update(
        timing_coverage_pass=1,
        outer_wall_seconds=1.0,
        outer_wall_rate=100.0,
        complete_wall_upper_seconds=1.0,
    )
    flags = _panel_evidence_flags(execution)
    candidates = build_power_candidates(
        profile=NESTED_HAAR_PROFILE,
        main_differences=arrays["raw_main_differences"],
        d3_differences=arrays["raw_d3_differences"],
        d4_differences=arrays["raw_d4_differences"],
        execution=execution,
        evidence_flags=flags,
    )
    main_ids = list(range(0xA0000, 0xA0000 + cluster_count))
    reference_ids = list(range(0xA0400, 0xA0400 + cluster_count))
    return {
        "schema": "d0-jacobi-rb-certified-haar-power-v1-panel",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "profile": NESTED_HAAR_PROFILE,
        "panel": "a",
        "root_seed": 261181,
        "path_id_plan_sha256": "plan",
        "main_path_ids": main_ids,
        "reference_path_ids": reference_ids,
        "path_id_pools": {
            "main": main_ids,
            "reference": reference_ids,
        },
        "cluster_count": cluster_count,
        "main_feature_count": 120,
        "reference_feature_count": 80,
        "source_npz_sha256": "source",
        "observable_payload": {
            "path": payload_path.name,
            "sha256": file_fingerprint(payload_path),
            "size": payload_path.stat().st_size,
            "array_hashes": {
                name: _array_hash(value) for name, value in arrays.items()
            },
        },
        "execution": execution,
        "candidates": candidates,
        "complete": 1,
        "finite": 1,
        "raw_endpoint_authorizing": 1,
        "dynkin_advisory_only": 1,
        "production_authorizing_pass": 1,
        "raw_endpoint_authorizing_pass": 1,
        "dynkin_advisory_only_pass": 1,
        "independent_pool_variance_pass": 1,
        "richardson_formula_pass": 1,
        **flags,
        "production_authorizing": 1,
    }


def test_power_candidate_records_the_joint_99_percent_error_budget(
    tmp_path,
) -> None:
    evidence = _valid_sealed_evidence(tmp_path)
    candidate = evidence["candidates"][0]  # type: ignore[index]
    assert candidate["variance_upper_confidence"] == 0.995
    assert candidate["mean_bound_confidence"] == 0.995
    assert candidate["joint_power_width_confidence_lower_bound"] == 0.99
    assert candidate["familywise_error_budget"] == {
        "total": 0.01,
        "variance_envelope": 0.005,
        "future_mean_bound": 0.005,
        "combination": "union_bound",
    }


def test_sealed_evidence_rederives_candidates_from_bound_npz(tmp_path) -> None:
    evidence = _valid_sealed_evidence(tmp_path)
    verified = verify_certified_haar_power_panel_evidence(
        run_dir=tmp_path,
        evidence=evidence,
        expected_profile=NESTED_HAAR_PROFILE,
        expected_panel="a",
    )
    assert verified == evidence

    tampered = {**evidence, "candidates": [*evidence["candidates"]]}  # type: ignore[index]
    tampered["candidates"][0] = {  # type: ignore[index]
        **tampered["candidates"][0],  # type: ignore[index]
        "predicted_main_half_width": 0.001,
    }
    with pytest.raises(
        ArtifactCompatibilityError,
        match="do not match the bound observables",
    ):
        verify_certified_haar_power_panel_evidence(
            run_dir=tmp_path,
            evidence=tampered,
            expected_profile=NESTED_HAAR_PROFILE,
            expected_panel="a",
        )


def test_corrupt_tail_recovery_preserves_only_verified_prefix(tmp_path) -> None:
    for stem in ("verified", "corrupt", "later"):
        (tmp_path / f"{stem}.json").write_text("{}", encoding="utf-8")
        (tmp_path / f"{stem}.npz").write_bytes(b"payload")
    (tmp_path / ".uncommitted.tmp").write_bytes(b"temporary")

    _discard_corrupt_tail(tmp_path, preserved_stems={"verified"})

    assert (tmp_path / "verified.json").is_file()
    assert (tmp_path / "verified.npz").is_file()
    assert not (tmp_path / "corrupt.json").exists()
    assert not (tmp_path / "corrupt.npz").exists()
    assert not (tmp_path / "later.json").exists()
    assert not (tmp_path / "later.npz").exists()
    assert (tmp_path / ".uncommitted.tmp").is_file()
