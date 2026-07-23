from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from mnist.d0_score_density_ratio_sealed_null_gate import (
    MAX_T_VERSION,
    SealedNullDecision,
    SealedNullThresholds,
    decide_sealed_null_workflow,
    evaluate_confirmation_null_family,
    evaluate_max_t_null_family,
    evaluate_parent_pilot_replay,
    evaluate_parent_replay_candidate,
    evaluate_sealed_null_workflow,
    evaluate_simultaneous_bootstrap_preflight,
    studentized_whole_path_max_t,
)


def _member(
    values: list[float],
    *,
    path_ids: list[int] | None = None,
    block: str = "panel-a",
    role: str | None = None,
    scope: str = "overall",
) -> dict[str, object]:
    result: dict[str, object] = {
        "path_ids": list(range(len(values))) if path_ids is None else path_ids,
        "path_values": values,
        "resampling_block": block,
        "scope": scope,
    }
    if role is not None:
        result["panel_role"] = role
    return result


def _pass() -> dict[str, object]:
    return {"evaluation_status": "evaluated", "passed": 1}


def _pending() -> dict[str, object]:
    return {"evaluation_status": "not_evaluated", "passed": 0}


def _family_gate(
    *,
    role_counts: dict[str, int],
    positive_by_role: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    positive_by_role = positive_by_role or {}
    positive_count = sum(len(value) for value in positive_by_role.values())
    return {
        "evaluation_status": "evaluated",
        "passed": int(positive_count == 0),
        "role_counts": role_counts,
        "positive_member_names_by_role": positive_by_role,
        "familywise_false_discovery": int(positive_count > 0),
    }


def _candidate(
    learning_rate: float,
    *,
    risk: float,
    other_failure: bool = False,
    b_bounds: tuple[float, float] = (-0.01, -0.02),
) -> dict[str, object]:
    subchecks: dict[str, object] = {
        "null_panel_a_lower_bounds": {"passed": 0},
        "null_panel_b_lower_bounds": {"passed": 1},
        "null_b_rejected": {"passed": 1},
        "null_selected_zero": {"passed": 1},
        "teacher_b_confirmed": {"passed": 1},
        "teacher_selected_nonzero": {"passed": 1},
        "teacher_score_gains": {"passed": 1},
        "teacher_flux_cosines": {"passed": 1},
        "teacher_relative_flux_l2": {"passed": 1},
    }
    if other_failure:
        subchecks["teacher_b_confirmed"] = {"passed": 0}
    base = {
        "gate": "density_ratio_pilot_candidate",
        "evaluation_status": "evaluated",
        "passed": 0,
        "learning_rate": learning_rate,
        "teacher_mean_ab_bce": risk,
        "maximum_clip_fraction_observed": 0.0,
        "subchecks": subchecks,
        "null_selection": {
            "selected_step": 0,
            "nominee_step": 1000,
            "nomination": {
                "nominee_panel_a_lower_bounds": [-1e-6, 2e-5],
            },
            "confirmation": {
                "accepted": 0,
                "panel_b_lower_bounds": list(b_bounds),
            },
        },
    }
    return {
        "gate": "selection_power_pilot_candidate",
        "evaluation_status": "evaluated",
        "passed": 0,
        "learning_rate": learning_rate,
        "accumulation_steps": 8,
        "teacher_mean_ab_bce": risk,
        "maximum_clip_fraction_observed": 0.0,
        "optimizer_health_pass": 1,
        "frozen_normalized_head_gate": {"base_gate": base},
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _null(seed: int, *, legacy_false: int = 0) -> dict[str, object]:
    return {
        "evaluation_status": "evaluated",
        "model_seed": seed,
        "complete": 1,
        "finite": 1,
        "boundary_admissible": 1,
        "optimizer_health_pass": 1,
        "false_discovery": legacy_false,
        "selection": {
            "nominee_step": 1000,
            "selected_step": 0,
            "confirmation": {"accepted": 0},
        },
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _bootstrap_preflight() -> dict[str, object]:
    value: dict[str, object] = {
        "evaluation_status": "evaluated",
        "complete": 1,
        "finite": 1,
        "version": MAX_T_VERSION,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    for name in (
        "deterministic_replay_pass",
        "member_order_invariance_pass",
        "path_order_invariance_pass",
        "shared_block_coupling_pass",
        "disjoint_block_stream_pass",
        "studentization_reference_pass",
        "simultaneous_coverage_fixture_pass",
        "whole_path_only_pass",
        "parent_family_coverage_pass",
    ):
        value[name] = 1
    return value


def test_thresholds_freeze_inherited_science_and_multiplicity_defaults() -> None:
    thresholds = SealedNullThresholds()
    assert thresholds.confidence == 0.95
    assert thresholds.bootstrap_replicates == 50_000
    assert thresholds.expected_pilot_learning_rates == (3e-5, 1e-5)
    assert thresholds.selection_power.head.base_channels == 32
    with pytest.raises(ValueError, match="confidence"):
        SealedNullThresholds(confidence=0.90)


def test_studentized_max_t_is_deterministic_and_order_invariant() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(-0.1, 1.0, 32).tolist()
    y = rng.normal(-0.15, 1.0, 32).tolist()
    first = studentized_whole_path_max_t(
        {"overall": _member(x), "data_end": _member(y)},
        seed=71,
        reps=600,
        bootstrap_chunk_size=37,
    )
    reordered = studentized_whole_path_max_t(
        {
            "data_end": _member(list(reversed(y)), path_ids=list(reversed(range(32)))),
            "overall": _member(list(reversed(x)), path_ids=list(reversed(range(32)))),
        },
        seed=71,
        reps=600,
        bootstrap_chunk_size=91,
    )
    assert first["family_fingerprint"] == reordered["family_fingerprint"]
    assert first["critical_value"] == reordered["critical_value"]
    assert first["members"] == reordered["members"]
    assert first["familywise_false_discovery"] == 0


def test_max_t_preserves_shared_blocks_and_detects_strong_positive_member() -> None:
    values = [1.0 + 0.01 * index for index in range(24)]
    record = studentized_whole_path_max_t(
        {
            "positive": _member(values, block="shared", role="b"),
            "negative": _member([-value for value in values], block="shared", role="c"),
        },
        seed=9,
        reps=500,
    )
    assert record["resampling_block_count"] == 1
    assert record["familywise_false_discovery"] == 1
    assert record["positive_member_names"] == ["positive"]
    assert {value["panel_role"] for value in record["members"]} == {"b", "c"}
    json.dumps(record, allow_nan=False)


def test_max_t_rejects_duplicate_or_mismatched_whole_path_clusters() -> None:
    with pytest.raises(ValueError, match="duplicate path IDs"):
        studentized_whole_path_max_t(
            {"x": _member([0.0, 1.0], path_ids=[1, 1])}, seed=1, reps=10
        )
    with pytest.raises(ValueError, match="share path IDs"):
        studentized_whole_path_max_t(
            {
                "x": _member([0.0, 1.0], path_ids=[1, 2]),
                "y": _member([0.0, 1.0], path_ids=[1, 3]),
            },
            seed=1,
            reps=10,
        )


def test_max_t_accepts_exact_zero_but_rejects_other_degenerate_vectors() -> None:
    exact_zero = studentized_whole_path_max_t(
        {"zero": _member([0.0] * 8)}, seed=4, reps=100
    )
    assert exact_zero["members"][0]["exact_zero_path_vector"] == 1
    assert exact_zero["members"][0]["simultaneous_lower_bound"] == 0.0
    assert exact_zero["familywise_false_discovery"] == 0

    for value in (-0.25, 0.25):
        with pytest.raises(ValueError, match="nonzero degenerate"):
            studentized_whole_path_max_t(
                {"constant": _member([value] * 8)}, seed=4, reps=100
            )


def test_max_t_family_gate_requires_exact_predeclared_members() -> None:
    record = studentized_whole_path_max_t(
        {
            "a": _member([-1.0, -0.5, -0.25]),
            "b": _member([-0.8, -0.4, -0.2]),
        },
        seed=3,
        reps=100,
    )
    record["bootstrap_replicates"] = 50_000
    gate = evaluate_max_t_null_family(
        record,
        expected_members=("a", "b"),
        required_replicates=50_000,
    )
    assert gate["passed"] == 1
    assert evaluate_max_t_null_family(
        record,
        expected_members=("a", "b", "c"),
        required_replicates=50_000,
    )["passed"] == 0


def test_bootstrap_preflight_fails_closed_on_every_contract() -> None:
    assert evaluate_simultaneous_bootstrap_preflight(_bootstrap_preflight())["passed"] == 1
    broken = _bootstrap_preflight()
    broken["path_order_invariance_pass"] = 0
    assert evaluate_simultaneous_bootstrap_preflight(broken)["passed"] == 0


def test_parent_candidate_allows_only_a_failure_and_keeps_b_authoritative() -> None:
    gate = evaluate_parent_replay_candidate(
        _candidate(3e-5, risk=0.6807), sealed_b_binding=_pass()
    )
    assert gate["passed"] == 1
    assert gate["discovery_a_lower_bounds_advisory"][1] > 0.0
    assert all(value < 0.0 for value in gate["sealed_b_lower_bounds"])

    other = evaluate_parent_replay_candidate(
        _candidate(3e-5, risk=0.6807, other_failure=True),
        sealed_b_binding=_pass(),
    )
    assert other["passed"] == 0
    positive_b = evaluate_parent_replay_candidate(
        _candidate(3e-5, risk=0.6807, b_bounds=(1e-6, -1e-6)),
        sealed_b_binding=_pass(),
    )
    assert positive_b["passed"] == 0


def test_parent_replay_recovers_frozen_profile_without_optimizer_steps() -> None:
    candidates = [
        _candidate(3e-5, risk=0.680783),
        _candidate(1e-5, risk=0.681793),
    ]
    replay = evaluate_parent_pilot_replay(
        candidates,
        sealed_b_bindings=[_pass(), _pass()],
        discovery_family={
            "evaluation_status": "evaluated",
            "passed": 0,
            "familywise_false_discovery": 1,
        },
        sealed_b_family=_family_gate(role_counts={"b": 4}),
    )
    assert replay["passed"] == 1
    assert replay["selected_profile"]["profile"]["body_learning_rate"] == pytest.approx(3e-5)
    assert replay["replay_only"] == 1
    assert replay["optimizer_steps_performed"] == 0
    assert replay["discovery_family_authorizing"] == 0


def test_confirmation_null_family_treats_legacy_marginal_signal_as_advisory() -> None:
    nulls = [_null(11, legacy_false=1), _null(12), _null(13)]
    family = _family_gate(role_counts={"b": 6, "c": 6, "d": 6})
    gate = evaluate_confirmation_null_family(
        nulls,
        max_t_family=family,
        sealed_b_bindings=[_pass(), _pass(), _pass()],
    )
    assert gate["passed"] == 1
    assert gate["legacy_false_discovery_count_advisory"] == 1
    assert gate["familywise_false_discovery"] == 0

    family = _family_gate(
        role_counts={"b": 6, "c": 6, "d": 6},
        positive_by_role={"c": ["seed-11-c-overall"]},
    )
    assert evaluate_confirmation_null_family(
        nulls,
        max_t_family=family,
        sealed_b_bindings=[_pass(), _pass(), _pass()],
    )["passed"] == 0


def test_closed_decisions_match_frozen_outcomes() -> None:
    controls = {
        "evaluation_status": "evaluated",
        "passed": 0,
        "optimizer_health_pass": 1,
        "teacher_study": {
            "classification_passing_seed_count": 0,
            "positive_point_estimate_seed_count": 0,
            "panel_disagreement": 0,
        },
        "null_family": {"familywise_false_discovery": 0},
    }
    kwargs = {
        "provenance": _pass(),
        "simultaneous_bootstrap": _pass(),
        "replay": _pass(),
        "confirmation_panel_power": _pass(),
        "controls": controls,
    }
    assert decide_sealed_null_workflow(**kwargs)["decision"] == SealedNullDecision.NO_DETECTABLE_DENSITY_RATIO_SIGNAL.value

    bootstrap = copy.deepcopy(kwargs)
    bootstrap["simultaneous_bootstrap"] = {"evaluation_status": "evaluated", "passed": 0}
    assert decide_sealed_null_workflow(**bootstrap)["decision"] == SealedNullDecision.SIMULTANEOUS_BOOTSTRAP_INVALID.value

    replay = copy.deepcopy(kwargs)
    replay["replay"] = {"evaluation_status": "evaluated", "passed": 0, "familywise_false_discovery": 0}
    assert decide_sealed_null_workflow(**replay)["decision"] == SealedNullDecision.PROFILE_RECOVERY_INVALID.value

    false = copy.deepcopy(kwargs)
    false["controls"]["null_family"]["selection_false_discovery"] = 1
    assert decide_sealed_null_workflow(**false)["decision"] == SealedNullDecision.SELECTION_FALSE_DISCOVERY.value

    audit = copy.deepcopy(kwargs)
    audit["controls"]["null_family"]["audit_false_discovery"] = 1
    assert decide_sealed_null_workflow(**audit)["decision"] == SealedNullDecision.CLASSIFICATION_AUDIT_INCONCLUSIVE.value

    unresolved = copy.deepcopy(kwargs)
    unresolved["controls"]["teacher_study"]["positive_point_estimate_seed_count"] = 1
    assert decide_sealed_null_workflow(**unresolved)["decision"] == SealedNullDecision.CLASSIFICATION_POWER_CONFIRMATION_UNRESOLVED.value

    value_only = copy.deepcopy(kwargs)
    value_only["controls"]["teacher_study"]["classification_passing_seed_count"] = 2
    assert decide_sealed_null_workflow(**value_only)["decision"] == SealedNullDecision.DENSITY_RATIO_VALUE_ONLY.value

    repaired = copy.deepcopy(kwargs)
    repaired["controls"]["passed"] = 1
    decision = decide_sealed_null_workflow(**repaired)
    assert decision["decision"] == SealedNullDecision.DENSITY_RATIO_CONTROL_PIPELINE_REPAIRED.value
    assert decision["physical_training_authorized"] == 1
    assert decision["sampling_authorized"] == 0


def test_workflow_required_gates_and_json_are_fail_closed() -> None:
    teacher = {
        "evaluation_status": "evaluated",
        "passed": 1,
        "optimizer_health_pass": 1,
        "classification_passing_seed_count": 2,
        "panel_disagreement": 0,
    }
    null = {
        "evaluation_status": "evaluated",
        "passed": 1,
        "optimizer_health_pass": 1,
        "familywise_false_discovery": 0,
    }
    report = evaluate_sealed_null_workflow(
        provenance=_pass(),
        simultaneous_bootstrap=_pass(),
        replay=_pass(),
        confirmation_panel_power=_pass(),
        teacher_study=teacher,
        null_family=null,
        require_gate="controls",
    )
    assert report["required_gate_pass"] == 1
    assert report["decision"]["physical_training_authorized"] == 1
    json.dumps(report, allow_nan=False)

    with pytest.raises(ValueError, match="require_gate"):
        evaluate_sealed_null_workflow(
            provenance=_pass(),
            simultaneous_bootstrap=_pass(),
            replay=_pass(),
            confirmation_panel_power=_pass(),
            teacher_study=teacher,
            null_family=null,
            require_gate="pilot",
        )
