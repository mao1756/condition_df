"""Fail-closed gates for the read-only quartile representation adjudication.

The module is deliberately pure.  It validates the sealed stage records and
applies the preregistered decision hierarchy, but grants no authority to open
fresh evidence, train a model, run a controller, reconstruct, or sample.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


SCHEMA = "d0-jacobi-rb-quartile-directional-adjudication-gate-v1"
REQUIRED_GATES = (
    "none",
    "preflight",
    "replay",
    "controls",
    "fittrace",
    "nominate",
    "adjudicate",
)

PREFLIGHT_FLAGS = (
    "parent_provenance_valid",
    "parent_immutability_valid",
    "checkpoint_payloads_valid",
    "role_cache_payloads_valid",
    "scientific_contract_valid",
    "role_firewall_valid",
    "candidate_component_plan_sealed",
    "inference_plan_sealed",
    "bootstrap_indices_sealed",
    "resource_projection_valid",
)
REPLAY_FLAGS = (
    "historical_gain_table_replayed",
    "historical_rank_table_replayed",
    "candidate_order_valid",
    "numerical_agreement_valid",
    "raw_role_labels_unopened",
)
CONTROLS_FLAGS = (
    "quadratic_moment_algebra_valid",
    "component_recomposition_valid",
    "exact_zero_control_valid",
    "stable_positive_direction_control_valid",
    "nonpositive_direction_control_valid",
    "energy_dominated_control_valid",
    "path_instability_control_valid",
    "phase_midpoint_cancellation_control_valid",
    "branch_cancellation_control_valid",
    "direction_rotation_control_valid",
    "malformed_moments_fail_closed",
    "physical_labels_unopened",
)
FITTRACE_FLAGS = (
    "fit_role_open_order_valid",
    "all_fittrace_jobs_complete",
    "branch_recomposition_valid",
    "moment_algebra_valid",
    "trajectory_diagnostics_valid",
    "nomination_forbidden",
    "downstream_labels_unopened",
    "resource_limits_valid",
    "parent_unchanged",
)
NOMINATE_FLAGS = (
    "gain_role_open_order_valid",
    "all_gain_jobs_complete",
    "nomination_rule_valid",
    "all_thirty_six_streams_accounted",
    "direction_nomination_sealed",
    "rank_labels_unopened",
    "rank_search_forbidden",
    "resource_limits_valid",
    "parent_unchanged",
)
ADJUDICATE_FLAGS = (
    "nomination_seal_valid",
    "rank_role_open_order_valid",
    "all_rank_jobs_complete",
    "max_t_family_valid",
    "q0_positive_control_evaluated",
    "direction_rules_valid",
    "effect_rules_valid",
    "branch_algebra_valid",
    "mechanism_classification_valid",
    "path_count_forecast_valid",
    "parent_unchanged",
)
FIT_TRACE_FLAGS = FITTRACE_FLAGS
NOMINATION_FLAGS = NOMINATE_FLAGS
RANK_ADJUDICATION_FLAGS = ADJUDICATE_FLAGS

STAGE_FLAGS: dict[str, tuple[str, ...]] = {
    "preflight": PREFLIGHT_FLAGS,
    "replay": REPLAY_FLAGS,
    "controls": CONTROLS_FLAGS,
    "fittrace": FITTRACE_FLAGS,
    "nominate": NOMINATE_FLAGS,
    "adjudicate": ADJUDICATE_FLAGS,
}

DECISION_ORDER = (
    "quartile_directional_parent_provenance_invalid",
    "quartile_directional_scientific_contract_invalid",
    "quartile_directional_resource_plan_invalid",
    "quartile_directional_historical_replay_invalid",
    "quartile_directional_prelabel_controls_failed",
    "quartile_directional_fittrace_invalid",
    "quartile_directional_nomination_invalid",
    "quartile_directional_rank_adjudication_invalid",
    "quartile_directional_q0_positive_control_failed",
    "unique_representation_hypothesis_identified",
    "same_class_effect_detected_but_non_authorizing_stop",
    "representation_cancellation_nonidentifying_stop",
    "positive_direction_effect_unresolved_stop",
    "later_quartile_direction_unstable_across_roles_stop",
    "no_later_quartile_signal_detectable_under_permitted_class_stop",
)
DECISIONS = DECISION_ORDER
INVALID_DECISIONS = frozenset(DECISION_ORDER[:9])
SCIENTIFIC_DECISIONS = frozenset(DECISION_ORDER[9:])
PENDING_DECISIONS = (
    "ready_for_preflight",
    "ready_for_replay",
    "ready_for_controls",
    "ready_for_fittrace",
    "ready_for_nominate",
    "ready_for_adjudicate",
    "ready_for_report",
)

COMPONENTS = ("full", "local_affine", "spatial_cnn")
BRANCH_COMPONENTS = ("local_affine", "spatial_cnn")
LATER_QUARTILES = ("q1", "q2", "q3")

ZERO_AUTHORIZATION_FIELDS = (
    "cache_generation_authorized",
    "new_path_generation_authorized",
    "physical_training_authorized",
    "new_learner_training_authorized",
    "fresh_fit_authorized",
    "fresh_calibration_authorized",
    "fresh_rank_authorized",
    "fresh_selection_authorized",
    "confirmation_authorized",
    "fresh_learner_plan_authorized",
    "production_refinement_authorized",
    "controller_planning_authorized",
    "controller_execution_authorized",
    "reconstruction_authorized",
    "sampling_authorized",
)
ZERO_WORK_FIELDS = (
    "new_transitions_generated",
    "optimizer_updates_performed",
    "new_checkpoints_created",
    "parent_selection_opened",
    "parent_confirmation_opened",
    "controller_trajectories_executed",
    "reconstructions_created",
    "samples_created",
    "parent_files_modified",
    "historical_design_evidence_authorizing",
)


class QuartileDirectionalGateError(ValueError):
    """A gate record lies outside the frozen workflow schema."""


def _one(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 1


def _zero(value: Any) -> bool:
    return isinstance(value, (bool, int)) and int(value) == 0


def _bit(row: Mapping[str, Any], *names: str) -> bool | None:
    for name in names:
        if name in row:
            value = row[name]
            return bool(int(value)) if isinstance(value, (bool, int)) and int(value) in (0, 1) else None
    return None


def safety_record() -> dict[str, int]:
    return {
        **{name: 0 for name in ZERO_AUTHORIZATION_FIELDS},
        **{name: 0 for name in ZERO_WORK_FIELDS},
    }


def _status(gate: Mapping[str, Any] | None) -> str:
    if not isinstance(gate, Mapping):
        return "not_evaluated"
    return str(gate.get("evaluation_status", "not_evaluated"))


def _passed(gate: Mapping[str, Any] | None) -> bool:
    return _status(gate) == "evaluated" and _one((gate or {}).get("passed"))


def not_evaluated_gate(stage: str, reason: str = "not evaluated") -> dict[str, Any]:
    if stage not in STAGE_FLAGS:
        raise QuartileDirectionalGateError(f"unknown stage gate: {stage}")
    return {
        "schema": f"{SCHEMA}-{stage}",
        "schema_version": 1,
        "gate": stage,
        "evaluation_status": "not_evaluated",
        "passed": 0,
        "stage_execution_valid": 0,
        "scientific_evidence_complete": 0,
        "reason": str(reason),
        **safety_record(),
    }


def evaluate_stage_gate(stage: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one sealed stage and reject hidden work or authority."""

    if stage not in STAGE_FLAGS:
        raise QuartileDirectionalGateError(f"unknown stage gate: {stage}")
    if not isinstance(metrics, Mapping):
        raise QuartileDirectionalGateError("gate metrics must be a mapping")
    status = str(metrics.get("evaluation_status", "not_evaluated"))
    checks = {name: int(_one(metrics.get(name))) for name in STAGE_FLAGS[stage]}
    for name in ZERO_AUTHORIZATION_FIELDS + ZERO_WORK_FIELDS:
        checks[name] = int(_zero(metrics.get(name, 0)))
    execution_valid = _one(metrics.get("stage_execution_valid", 1))
    passed = int(status == "evaluated" and execution_valid and all(checks.values()))
    result: dict[str, Any] = {
        "schema": f"{SCHEMA}-{stage}",
        "schema_version": 1,
        "gate": stage,
        "evaluation_status": status,
        "passed": passed,
        "stage_execution_valid": int(status == "evaluated" and execution_valid),
        "scientific_evidence_complete": 0,
        "checks": checks,
        **safety_record(),
    }
    for key in (
        "failure_domain",
        "failure_code",
        "error",
        "resource_projection",
        "component_diagnostics",
    ):
        if key in metrics:
            result[key] = metrics[key]
    return result


def evaluate_preflight_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("preflight", metrics)


def evaluate_replay_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("replay", metrics)


def evaluate_controls_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("controls", metrics)


def evaluate_fittrace_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("fittrace", metrics)


def evaluate_fit_trace_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_fittrace_gate(metrics)


def evaluate_nominate_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("nominate", metrics)


def evaluate_nomination_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_nominate_gate(metrics)


def evaluate_adjudicate_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_stage_gate("adjudicate", metrics)


def evaluate_rank_adjudication_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_adjudicate_gate(metrics)


def _next_action(decision: str) -> str:
    actions = {
        "ready_for_preflight": "verify and seal the two immutable parents",
        "ready_for_replay": "replay the sealed historical gain/rank summaries",
        "ready_for_controls": "run the prelabel algebra and mechanism controls",
        "ready_for_fittrace": "open only physical_fit for in-sample diagnostics",
        "ready_for_nominate": "open only gain_calibration and seal 36 nominees",
        "ready_for_adjudicate": "open training_rank only after the nomination seal",
        "ready_for_report": "finalize the read-only scientific decision",
        "unique_representation_hypothesis_identified": (
            "write a separately reviewed fresh branch-restricted learner plan"
        ),
        "same_class_effect_detected_but_non_authorizing_stop": (
            "stop; historical same-class effects authorize no gain or training rerun"
        ),
        "representation_cancellation_nonidentifying_stop": (
            "stop; no single replacement representation is identified"
        ),
        "positive_direction_effect_unresolved_stop": (
            "stop; retain only the nonauthorizing historical power forecast"
        ),
        "later_quartile_direction_unstable_across_roles_stop": (
            "stop; do not rerun the unstable representation with more paths"
        ),
        "no_later_quartile_signal_detectable_under_permitted_class_stop": (
            "stop the permitted width-32 local-affine-plus-CNN repair loop"
        ),
    }
    return actions.get(decision, "repair the named immutable evidence gate")


def _decision_record(
    decision: str,
    *,
    complete: bool,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if decision not in set(DECISION_ORDER) | set(PENDING_DECISIONS):
        raise QuartileDirectionalGateError(f"unknown closed decision: {decision}")
    pending = decision in PENDING_DECISIONS
    unique = decision == "unique_representation_hypothesis_identified"
    result: dict[str, Any] = {
        "schema": f"{SCHEMA}-decision",
        "schema_version": 1,
        "evaluation_status": "not_evaluated" if pending else "evaluated",
        "decision": decision,
        "terminal": int(not pending),
        "scientific_evidence_complete": int(complete),
        "invalid_evidence": int(decision in INVALID_DECISIONS),
        "valid_scientific_stop": int(decision in SCIENTIFIC_DECISIONS and not unique),
        "unique_representation_identified": int(unique),
        "fresh_learner_plan_drafting_recommended": int(unique),
        "next_action": _next_action(decision),
        **safety_record(),
    }
    if evidence is not None:
        result["scientific_diagnostics"] = dict(evidence)
    return result


def _global_failure(
    gates: tuple[Mapping[str, Any] | None, ...],
    *,
    domain: str,
) -> bool:
    for gate in gates:
        if not isinstance(gate, Mapping) or _passed(gate) or _status(gate) == "not_evaluated":
            continue
        checks = gate.get("checks") if isinstance(gate.get("checks"), Mapping) else {}
        failure_domain = str(gate.get("failure_domain", ""))
        failure_code = str(gate.get("failure_code", ""))
        if domain == "provenance" and (
            failure_domain in {"provenance", "parent_immutability"}
            or "provenance" in failure_code
            or "parent_immutability" in failure_code
            or "parent_unchanged" in checks and not _one(checks.get("parent_unchanged"))
        ):
            return True
        if domain == "scientific_contract" and (
            failure_domain == "scientific_contract"
            or "scientific_contract" in failure_code
        ):
            return True
        if domain == "resource" and (
            failure_domain in {"resource", "resource_gate"}
            or "resource" in failure_code
        ):
            return True
    return False


def _lookup(mapping: Mapping[str, Any], names: tuple[Any, ...]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _component_record_table(
    evidence: Mapping[str, Any],
) -> dict[tuple[int, str], Mapping[str, Any]] | None:
    records: Any = evidence.get("component_rows")
    if records is None and isinstance(evidence.get("stream_adjudication"), Mapping):
        records = evidence["stream_adjudication"].get("component_rows")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        return None
    table: dict[tuple[int, str], Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            return None
        quartile = record.get("quartile")
        component = record.get("component")
        if (
            isinstance(quartile, bool)
            or not isinstance(quartile, int)
            or not 0 <= quartile <= 3
            or component not in COMPONENTS
            or (quartile, component) in table
        ):
            return None
        table[(quartile, str(component))] = record
    return table if len(table) == 12 else None


def _component_rows(evidence: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]] | None:
    source = _lookup(evidence, ("quartiles", "per_quartile_diagnostics", "later_quartiles"))
    record_table = _component_record_table(evidence)
    if not isinstance(source, Mapping) and record_table is None:
        return None
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for number, quartile in enumerate(LATER_QUARTILES, start=1):
        qrow = _lookup(source, (quartile, number, str(number))) if isinstance(source, Mapping) else {}
        if not isinstance(qrow, Mapping):
            return None
        components = qrow.get("components", qrow)
        if not isinstance(components, Mapping):
            return None
        normalized: dict[str, dict[str, Any]] = {}
        for component in COMPONENTS:
            row = components.get(component)
            if not isinstance(row, Mapping) and record_table is not None:
                row = record_table.get((number, component))
            if not isinstance(row, Mapping):
                return None
            direction = _bit(row, "stable_direction", "direction_stable", "stable_direction_pass")
            effect = _bit(row, "stable_effect", "effect_stable", "stable_effect_pass")
            if direction is None or effect is None:
                return None
            normalized[component] = {
                **dict(row),
                "stable_direction": int(direction),
                "stable_effect": int(effect),
            }
        normalized["_quartile"] = dict(qrow)
        result[quartile] = normalized
    return result


def _q0_control(evidence: Mapping[str, Any]) -> tuple[bool, bool] | None:
    q0 = _lookup(evidence, ("q0_full", "q0_positive_control"))
    if not isinstance(q0, Mapping):
        quartiles = evidence.get("quartiles")
        if isinstance(quartiles, Mapping):
            q0row = _lookup(quartiles, ("q0", 0, "0"))
            if isinstance(q0row, Mapping):
                components = q0row.get("components", q0row)
                q0 = components.get("full") if isinstance(components, Mapping) else None
    if not isinstance(q0, Mapping):
        table = _component_record_table(evidence)
        q0 = table.get((0, "full")) if table is not None else None
    if not isinstance(q0, Mapping):
        return None
    direction = _bit(q0, "stable_direction", "direction_stable", "stable_direction_pass")
    effect = _bit(q0, "stable_effect", "effect_stable", "stable_effect_pass")
    if direction is None or effect is None:
        return None
    return direction, effect


def _attribution_passes(
    evidence: Mapping[str, Any],
    quartile: str,
    passing_branch: str,
    qrow: Mapping[str, Any],
) -> bool:
    aliases = (
        "competing_branch_negative_in_full_failure_strata",
        "full_failure_attributed_to_competing_branch",
        "cancellation_attribution_valid",
    )
    for name in aliases:
        value = qrow.get(name)
        if isinstance(value, Mapping):
            bit = _bit(value, passing_branch)
        else:
            bit = _bit(qrow, name)
        if bit is not None:
            return bit
    top = evidence.get("cancellation_attribution")
    if isinstance(top, Mapping):
        per_q = top.get(quartile)
        if isinstance(per_q, Mapping):
            bit = _bit(per_q, passing_branch, "passed", "valid")
            if bit is not None:
                return bit
    records = evidence.get("component_cancellation_rows")
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        quartile_number = int(quartile[1:])
        for record in records:
            if not isinstance(record, Mapping):
                continue
            if record.get("quartile") not in (quartile, quartile_number):
                continue
            if record.get("passing_branch", record.get("identified_branch")) != passing_branch:
                continue
            bit = _bit(
                record,
                "competing_branch_negative_in_responsible_strata",
                "attribution_valid",
            )
            if bit is not None:
                return bit
    return False


def _flag_any(
    evidence: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    names: tuple[str, ...],
) -> bool:
    top = _bit(evidence, *names)
    if top:
        return True
    for quartile in LATER_QUARTILES:
        for component in COMPONENTS:
            if _bit(rows[quartile][component], *names):
                return True
        if _bit(rows[quartile]["_quartile"], *names):
            return True
    return False


def _scientific_decision(evidence: Mapping[str, Any]) -> str:
    rows = _component_rows(evidence)
    q0 = _q0_control(evidence)
    inferential = _bit(
        evidence,
        "inferential_and_role_order_valid",
        "all_inferential_and_role_order_gates_pass",
    )
    algebra = _bit(
        evidence,
        "branch_algebra_cancellation_valid",
        "exact_branch_algebra_valid",
    )
    if rows is None or q0 is None or inferential is not True or algebra is not True:
        return "quartile_directional_rank_adjudication_invalid"
    if not all(q0):
        return "quartile_directional_q0_positive_control_failed"

    qualifying: list[str] = []
    for branch in BRANCH_COMPONENTS:
        competitor = BRANCH_COMPONENTS[1 - BRANCH_COMPONENTS.index(branch)]
        branch_passes = all(
            _one(rows[q][branch]["stable_direction"])
            and _one(rows[q][branch]["stable_effect"])
            for q in LATER_QUARTILES
        )
        competitor_never_effective = all(
            not _one(rows[q][competitor]["stable_effect"])
            for q in LATER_QUARTILES
        )
        failed_full_quartiles = [
            q
            for q in LATER_QUARTILES
            if not (
                _one(rows[q]["full"]["stable_direction"])
                and _one(rows[q]["full"]["stable_effect"])
            )
        ]
        attribution = bool(failed_full_quartiles) and all(
            _attribution_passes(evidence, q, branch, rows[q]["_quartile"])
            for q in failed_full_quartiles
        )
        if branch_passes and competitor_never_effective and attribution:
            qualifying.append(branch)
    if len(qualifying) == 1:
        return "unique_representation_hypothesis_identified"

    if all(_one(rows[q]["full"]["stable_effect"]) for q in LATER_QUARTILES):
        return "same_class_effect_detected_but_non_authorizing_stop"

    cancellation_visible = _flag_any(
        evidence,
        rows,
        ("cancellation_visible", "branch_cancellation_visible", "local_incompatibility_visible"),
    )
    stable_branch_effect = any(
        _one(rows[q][branch]["stable_effect"])
        for q in LATER_QUARTILES
        for branch in BRANCH_COMPONENTS
    )
    if cancellation_visible or stable_branch_effect:
        return "representation_cancellation_nonidentifying_stop"

    if _flag_any(
        evidence,
        rows,
        (
            "positive_direction_effect_unresolved",
            "positive_effect_point_with_nonpositive_bound",
        ),
    ):
        return "positive_direction_effect_unresolved_stop"

    if _flag_any(
        evidence,
        rows,
        (
            "positive_gain_direction",
            "rank_transfer_failed",
            "rotation_or_path_instability",
            "direction_rotation",
            "path_instability",
        ),
    ):
        return "later_quartile_direction_unstable_across_roles_stop"

    return "no_later_quartile_signal_detectable_under_permitted_class_stop"


def _merge_gates(
    evidence: Mapping[str, Any] | None,
    gates: Mapping[str, Mapping[str, Any] | None] | None,
    explicit: Mapping[str, Mapping[str, Any] | None],
) -> tuple[dict[str, Mapping[str, Any] | None], bool]:
    merged: dict[str, Mapping[str, Any] | None] = {}
    provided = gates is not None or any(value is not None for value in explicit.values())
    embedded = evidence.get("gates") if isinstance(evidence, Mapping) else None
    if isinstance(embedded, Mapping):
        provided = True
    for stage in STAGE_FLAGS:
        value = explicit.get(stage)
        if value is None and isinstance(gates, Mapping):
            value = gates.get(stage)
        if value is None and isinstance(embedded, Mapping):
            value = embedded.get(stage)
        merged[stage] = value
    return merged, provided


def decide_workflow(
    evidence: Mapping[str, Any] | None = None,
    *,
    gates: Mapping[str, Mapping[str, Any] | None] | None = None,
    preflight_gate: Mapping[str, Any] | None = None,
    replay_gate: Mapping[str, Any] | None = None,
    controls_gate: Mapping[str, Any] | None = None,
    fittrace_gate: Mapping[str, Any] | None = None,
    nominate_gate: Mapping[str, Any] | None = None,
    adjudicate_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply exact gate precedence, then the closed scientific partition.

    Passing only ``evidence`` is supported for pure scientific fixture tests;
    production callers should also pass the six stage gates (or ``gates``).
    """

    explicit = {
        "preflight": preflight_gate,
        "replay": replay_gate,
        "controls": controls_gate,
        "fittrace": fittrace_gate,
        "nominate": nominate_gate,
        "adjudicate": adjudicate_gate,
    }
    merged, gates_provided = _merge_gates(evidence, gates, explicit)
    ordered = tuple(merged[stage] for stage in STAGE_FLAGS)
    if gates_provided:
        if _global_failure(ordered, domain="provenance"):
            return _decision_record(
                "quartile_directional_parent_provenance_invalid", complete=False
            )
        if _global_failure(ordered, domain="scientific_contract"):
            return _decision_record(
                "quartile_directional_scientific_contract_invalid", complete=False
            )
        if _global_failure(ordered, domain="resource"):
            return _decision_record(
                "quartile_directional_resource_plan_invalid", complete=False
            )

        preflight = merged["preflight"]
        if _status(preflight) == "not_evaluated":
            return _decision_record("ready_for_preflight", complete=False)
        if not _passed(preflight):
            checks = preflight.get("checks", {}) if isinstance(preflight, Mapping) else {}
            if any(
                name in checks and not _one(checks.get(name))
                for name in (
                    "parent_provenance_valid",
                    "parent_immutability_valid",
                    "checkpoint_payloads_valid",
                    "role_cache_payloads_valid",
                )
            ):
                decision = "quartile_directional_parent_provenance_invalid"
            elif any(
                name != "resource_projection_valid" and not _one(value)
                for name, value in checks.items()
                if name in PREFLIGHT_FLAGS
            ):
                decision = "quartile_directional_scientific_contract_invalid"
            elif not _one(checks.get("resource_projection_valid")):
                decision = "quartile_directional_resource_plan_invalid"
            else:
                decision = "quartile_directional_scientific_contract_invalid"
            return _decision_record(decision, complete=False)

        stage_failures = (
            ("replay", "quartile_directional_historical_replay_invalid"),
            ("controls", "quartile_directional_prelabel_controls_failed"),
            ("fittrace", "quartile_directional_fittrace_invalid"),
            ("nominate", "quartile_directional_nomination_invalid"),
            ("adjudicate", "quartile_directional_rank_adjudication_invalid"),
        )
        for stage, invalid_decision in stage_failures:
            gate = merged[stage]
            if _status(gate) == "not_evaluated":
                return _decision_record(f"ready_for_{stage}", complete=False)
            if not _passed(gate):
                return _decision_record(invalid_decision, complete=False)

    if not isinstance(evidence, Mapping):
        return _decision_record("ready_for_report", complete=False)
    decision = _scientific_decision(evidence)
    return _decision_record(
        decision,
        complete=decision in SCIENTIFIC_DECISIONS,
        evidence=evidence,
    )


def evaluate_required_gate(
    require_gate: str,
    gates: Mapping[str, Mapping[str, Any] | None] | None = None,
    decision: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    *,
    preflight_gate: Mapping[str, Any] | None = None,
    replay_gate: Mapping[str, Any] | None = None,
    controls_gate: Mapping[str, Any] | None = None,
    fittrace_gate: Mapping[str, Any] | None = None,
    nominate_gate: Mapping[str, Any] | None = None,
    adjudicate_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the cumulative requested stage without granting authority."""

    if require_gate not in REQUIRED_GATES:
        raise QuartileDirectionalGateError(f"unknown required gate: {require_gate}")
    explicit = {
        "preflight": preflight_gate,
        "replay": replay_gate,
        "controls": controls_gate,
        "fittrace": fittrace_gate,
        "nominate": nominate_gate,
        "adjudicate": adjudicate_gate,
    }
    merged, _ = _merge_gates(evidence, gates, explicit)
    if require_gate == "none":
        passed = True
    else:
        index = REQUIRED_GATES.index(require_gate)
        passed = all(
            _passed(merged[stage]) for stage in REQUIRED_GATES[1 : index + 1]
        )
    active = dict(
        decision
        or decide_workflow(
            evidence,
            gates=merged,
        )
    )
    return {
        "schema": f"{SCHEMA}-workflow",
        "schema_version": 1,
        "evaluation_status": "evaluated",
        "require_gate": require_gate,
        "required_gate_pass": int(passed),
        "required_gate_exit_code": 0 if passed else 1,
        "decision": active,
        **safety_record(),
    }


def decision_exit_code(decision: Mapping[str, Any]) -> int:
    """Valid scientific stops and the unique hypothesis both exit zero."""

    name = str(decision.get("decision", ""))
    if name in SCIENTIFIC_DECISIONS or name in PENDING_DECISIONS:
        return 0
    return 1


__all__ = [
    "ADJUDICATE_FLAGS",
    "BRANCH_COMPONENTS",
    "COMPONENTS",
    "CONTROLS_FLAGS",
    "DECISIONS",
    "DECISION_ORDER",
    "FITTRACE_FLAGS",
    "FIT_TRACE_FLAGS",
    "INVALID_DECISIONS",
    "LATER_QUARTILES",
    "NOMINATE_FLAGS",
    "NOMINATION_FLAGS",
    "PENDING_DECISIONS",
    "PREFLIGHT_FLAGS",
    "QuartileDirectionalGateError",
    "REPLAY_FLAGS",
    "REQUIRED_GATES",
    "RANK_ADJUDICATION_FLAGS",
    "SCHEMA",
    "SCIENTIFIC_DECISIONS",
    "STAGE_FLAGS",
    "ZERO_AUTHORIZATION_FIELDS",
    "ZERO_WORK_FIELDS",
    "decide_workflow",
    "decision_exit_code",
    "evaluate_adjudicate_gate",
    "evaluate_controls_gate",
    "evaluate_fit_trace_gate",
    "evaluate_fittrace_gate",
    "evaluate_nominate_gate",
    "evaluate_nomination_gate",
    "evaluate_preflight_gate",
    "evaluate_rank_adjudication_gate",
    "evaluate_replay_gate",
    "evaluate_required_gate",
    "evaluate_stage_gate",
    "not_evaluated_gate",
    "safety_record",
]
