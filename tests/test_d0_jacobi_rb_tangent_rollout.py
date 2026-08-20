from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist.d0_jacobi_rb_boundary_tangent import frozen_score_logistic_flow
from mnist.d0_jacobi_rb_boundary_tangent_frequency1_coordinate import (
    FrequencyOneCoordinateZeroBaselinePredictor,
)
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    ModelInputs,
    matching_indices,
    semantic_sha256,
    state_dict_sha256,
)
from mnist.d0_jacobi_rb_reverse_controller import phase_exposure
from mnist.d0_jacobi_rb_tangent_rollout import (
    CertifiedExploratoryReference,
    EXACT_FORWARD_FORBIDDEN_COUNTERS,
    EXACT_FORWARD_FORBIDDEN_COUNTERS_VERSION,
    ExactForwardShardAggregateError,
    ScaledTangentScoreController,
    SignedDiagnosticTangentScoreController,
    TangentRolloutContractError,
    TargetFractionOracleController,
    ZeroTangentScoreController,
    aggregate_exact_forward_shards,
    benchmark_tangent_phase,
    exploratory_reference_rng_key,
    fixed_rendering_scale,
    load_verified_frequency1_checkpoint,
    load_verified_source_target,
    raw_state_metrics,
    render_background_demixed,
    render_raw_density,
    render_source_image,
    reverse_suffix_sequence,
    rollout_array_sha256,
    run_forward_trajectory,
    run_reverse_trajectory,
    target_oracle_identity_control,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FAILED_FUSED_PREFLIGHT = (
    REPOSITORY_ROOT
    / "runs"
    / "experiment12_d0_jacobi_rb_frequency1_rollout"
    / "20260812-065538_production-frequency1-exploratory-rollout-fused-laptop-v1"
)


def _inputs(state: torch.Tensor, phase: int = 0) -> ModelInputs:
    rows = state.shape[0]
    return ModelInputs(
        later_full_state=state.to(dtype=torch.float32),
        reverse_time=torch.full((rows,), 0.5, dtype=torch.float64),
        phase=torch.full((rows,), phase, dtype=torch.long),
        color=torch.full((rows,), PHASE_MATCHINGS[phase], dtype=torch.long),
        duration=torch.full((rows,), PHASE_DURATIONS[phase], dtype=torch.float32),
        label=torch.full((rows,), 3, dtype=torch.long),
    )


class _ConstantController(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def score_prediction(self, inputs: ModelInputs) -> torch.Tensor:
        return torch.full(
            (inputs.later_full_state.shape[0], EDGES_PER_PHASE),
            self.value,
            dtype=torch.float64,
            device=inputs.later_full_state.device,
        )


class _CheckpointLikeController(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.linspace(-1.25, 1.75, EDGES_PER_PHASE, dtype=torch.float64)
        )

    def score_prediction(self, inputs: ModelInputs) -> torch.Tensor:
        return self.weight.expand(inputs.batch_size, -1)

    def score_prediction_prevalidated(self, inputs: ModelInputs) -> torch.Tensor:
        return self.score_prediction(inputs)


def test_zero_and_scaled_controllers_preserve_exact_contract() -> None:
    state = torch.full((2, 784), 1.0 / 784.0, dtype=torch.float64)
    inputs = _inputs(state)
    zero = ZeroTangentScoreController().score_prediction(inputs)
    assert zero.dtype == torch.float64
    assert zero.shape == (2, 392)
    assert torch.count_nonzero(zero).item() == 0

    base = _ConstantController(1.25)
    before = {name: value.detach().clone() for name, value in base.state_dict().items()}
    wrapped = ScaledTangentScoreController(base, 2.0)
    result = wrapped.score_prediction(inputs)
    assert torch.equal(result, torch.full_like(result, 2.5))
    assert all(torch.equal(before[name], value) for name, value in base.state_dict().items())
    assert wrapped.record()["scaled_score_maximum_absolute"] == 2.5


def test_signed_diagnostic_is_exact_negative_and_preserves_base_state() -> None:
    state = torch.full((2, 784), 1.0 / 784.0, dtype=torch.float64)
    inputs = _inputs(state)
    base = _CheckpointLikeController()
    before = state_dict_sha256(base.state_dict())
    positive = ScaledTangentScoreController(base, 0.5)
    negative = SignedDiagnosticTangentScoreController(base, -0.5)

    plus = positive.score_prediction(inputs)
    minus = negative.score_prediction(inputs)
    assert torch.equal(minus, -plus)
    assert state_dict_sha256(base.state_dict()) == before
    assert negative.record()["diagnostic_only"] == 1
    assert negative.record()["gain"] == -0.5

    plus_deferred = positive.score_prediction_deferred(inputs)
    minus_deferred = negative.score_prediction_deferred(inputs)
    assert torch.equal(minus_deferred, -plus_deferred)
    assert state_dict_sha256(base.state_dict()) == before


@pytest.mark.parametrize("gain", [-0.5, float("-inf"), float("nan")])
def test_production_scaled_controller_still_rejects_negative_or_nonfinite_gain(
    gain: float,
) -> None:
    with pytest.raises(TangentRolloutContractError):
        ScaledTangentScoreController(_CheckpointLikeController(), gain)


@pytest.mark.parametrize("gain", [0.0, 0.5, float("-inf"), float("nan")])
def test_signed_diagnostic_requires_a_finite_negative_gain(gain: float) -> None:
    with pytest.raises(TangentRolloutContractError):
        SignedDiagnosticTangentScoreController(_CheckpointLikeController(), gain)


def test_target_fraction_oracle_hits_interior_and_reports_unreachable_boundary() -> None:
    state = torch.full((1, 784), 1.0 / 784.0, dtype=torch.float64)
    target = state[0].clone()
    tails, heads = matching_indices(device="cpu")
    tail, head = int(tails[0, 0]), int(heads[0, 0])
    pair = float(target[tail] + target[head])
    target[tail] = 0.25 * pair
    target[head] = 0.75 * pair
    oracle = TargetFractionOracleController(target.numpy(), microsteps=2)
    inputs = _inputs(state)
    score = oracle.score_prediction(inputs)
    geometry_state = inputs.later_full_state.to(dtype=torch.float64)
    pair_mass = geometry_state[:, tails[0]] + geometry_state[:, heads[0]]
    delta_u = phase_exposure(pair_mass, PHASE_DURATIONS[0]) / 2.0
    output = frozen_score_logistic_flow(
        geometry_state, (tails[0], heads[0]), score, delta_u
    )
    observed = output[:, heads[0]] / (output[:, tails[0]] + output[:, heads[0]])
    target_fraction = target[heads[0]] / (target[tails[0]] + target[heads[0]])
    assert float(observed[0, 0]) == pytest.approx(float(target_fraction[0]), abs=2e-14)

    boundary = state.clone()
    boundary[0, tail] += boundary[0, head]
    boundary[0, head] = 0.0
    boundary_oracle = TargetFractionOracleController(target.numpy(), microsteps=2)
    boundary_score = boundary_oracle.score_prediction(_inputs(boundary))
    assert boundary_score[0, 0].item() == 0.0
    assert boundary_oracle.record()["target_oracle_unreachable_boundary_count"] >= 1
    identity = target_oracle_identity_control()
    assert identity["passed"] == 1
    assert (
        identity["maximum_interior_fraction_error"]
        <= identity["maximum_interior_fraction_error_threshold"]
    )
    assert identity["reference_call_count"] == 4
    assert identity["reference_sequence_valid"] == 1
    assert identity["canonical_transition_ids_valid"] == 1
    assert identity["identity_transition_ids_globally_unique"] == 1


def test_reverse_sequences_have_frozen_lengths_and_endpoints() -> None:
    short = reverse_suffix_sequence(127)
    full = reverse_suffix_sequence(511)
    assert (len(short), short[0], short[-1]) == (896, (127, 6), (0, 0))
    assert (len(full), full[0], full[-1]) == (3584, (511, 6), (0, 0))
    with pytest.raises(TangentRolloutContractError):
        reverse_suffix_sequence(512)


def test_paired_rng_key_has_no_variant_and_streams_are_distinct() -> None:
    first = exploratory_reference_rng_key(261402, "development", "pre")
    same = exploratory_reference_rng_key(261402, "development", "pre")
    other = exploratory_reference_rng_key(261402, "evaluation", "pre")
    assert first == same
    assert first != other
    assert "zero" not in first and "learned" not in first and "oracle" not in first
    with pytest.raises(TypeError):
        exploratory_reference_rng_key(261402, "development", "pre", "learned")  # type: ignore[call-arg]


@dataclass
class _ReferenceBatch:
    later_head_fraction: torch.Tensor
    certified_mask: torch.Tensor
    fallback_mask: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


def test_certified_reference_requires_all_lanes_and_reports_rate() -> None:
    captured: list[object] = []

    def sampler(head: torch.Tensor, exposure: torch.Tensor, **kwargs: object) -> _ReferenceBatch:
        captured.append(kwargs["rng_key"])
        return _ReferenceBatch(
            later_head_fraction=head.clone(),
            certified_mask=torch.ones_like(head, dtype=torch.bool),
            fallback_mask=torch.zeros_like(head, dtype=torch.bool),
            diagnostics={},
        )

    adapter = CertifiedExploratoryReference(
        profile=JacobiRBCudaProfile(),
        root_seed=9,
        stream_role="development",
        sampler=sampler,
    )
    x = torch.full((5,), 0.5, dtype=torch.float64)
    adapter(
        head_fraction=x,
        exposure=torch.ones_like(x),
        transition_ids=torch.arange(5, dtype=torch.int64).to(torch.uint64),
        role="pre",
    )
    record = adapter.record()
    assert record["certificate_fraction"] == 1.0
    assert record["transition_count"] == 5
    assert captured == [exploratory_reference_rng_key(9, "development", "pre")]

    def bad(head: torch.Tensor, exposure: torch.Tensor, **_: object) -> _ReferenceBatch:
        certified = torch.ones_like(head, dtype=torch.bool)
        certified[0] = False
        return _ReferenceBatch(head.clone(), certified, torch.zeros_like(certified), {})

    invalid = CertifiedExploratoryReference(
        profile=JacobiRBCudaProfile(), root_seed=9, stream_role="x", sampler=bad
    )
    with pytest.raises(TangentRolloutContractError, match="uncertified"):
        invalid(
            head_fraction=x,
            exposure=torch.ones_like(x),
            transition_ids=torch.arange(5, dtype=torch.int64).to(torch.uint64),
            role="pre",
        )


class _IdentityReference:
    def __call__(
        self,
        *,
        head_fraction: torch.Tensor,
        exposure: torch.Tensor,
        transition_ids: torch.Tensor,
        role: str,
    ) -> SimpleNamespace:
        del exposure, transition_ids, role
        return SimpleNamespace(later_head_fraction=head_fraction.clone())

    def record(self) -> dict[str, int]:
        return {"transition_count": 0, "certified_count": 0, "fallback_count": 0}


def test_complete_phase_benchmark_uses_a_valid_reverse_shard_head() -> None:
    state = np.full((1, 784), 1.0 / 784.0, dtype=np.float64)
    record = benchmark_tangent_phase(
        state,
        controller=ZeroTangentScoreController(),
        path_ids=(0xFA000,),
        outer_step=127,
        phase=6,
        reference_factory=lambda _: _IdentityReference(),
        microsteps=2,
        repeats=2,
        device="cpu",
    )
    assert [row["transition_count"] for row in record["repeats"]] == [1568, 1568]
    assert record["repeat_output_hashes_identical"] == 1


def test_reverse_restart_is_identical_and_skips_committed_shard(tmp_path: Path) -> None:
    initial = np.full((1, 784), 1.0 / 784.0, dtype=np.float64)
    calls = 0

    def factory(_: int) -> _IdentityReference:
        nonlocal calls
        calls += 1
        return _IdentityReference()

    arguments = dict(
        anchor_step=7,
        output_dir=tmp_path,
        trajectory_name="zero",
        controller=ZeroTangentScoreController(),
        reference_factory=factory,
        path_ids=(0xFA100,),
        controller_binding={"kind": "zero"},
        rng_binding={"root_seed": 261402, "stream_role": "development"},
        microsteps=2,
        device="cpu",
    )
    first = run_reverse_trajectory(initial, **arguments)
    assert calls == 1
    second = run_reverse_trajectory(initial, **arguments)
    assert calls == 1
    assert np.array_equal(first.final_state, second.final_state)
    assert first.to_record()["final_state_sha256"] == second.to_record()["final_state_sha256"]
    # Reduced eight-step fixtures have no checkpoint boundary at internal
    # quarters; production 128/512-step trajectories do.
    assert set(first.saved_states) == {"start", "final"}

    # An NPZ written before interruption has no commit authority.  It is
    # replayed and atomically replaced once its missing JSON is observed.
    record = tmp_path / "reverse_shards" / "zero" / "shard-0000.json"
    record.unlink()
    recovered = run_reverse_trajectory(initial, **arguments)
    assert calls == 2
    assert np.array_equal(recovered.final_state, first.final_state)
    assert record.is_file()


def test_reverse_resume_aggregates_persisted_oracle_telemetry(tmp_path: Path) -> None:
    initial = np.full((1, 784), 1.0 / 784.0, dtype=np.float64)
    common = dict(
        anchor_step=7,
        output_dir=tmp_path,
        trajectory_name="oracle",
        reference_factory=lambda _: _IdentityReference(),
        path_ids=(0xFA100,),
        controller_binding={"kind": "oracle"},
        rng_binding={"root_seed": 261402, "stream_role": "development"},
        microsteps=2,
        device="cpu",
    )
    first = run_reverse_trajectory(
        initial,
        controller=TargetFractionOracleController(initial[0], microsteps=2),
        **common,
    )
    assert first.diagnostics["controller"]["already_equal_count"] > 0
    replay = run_reverse_trajectory(
        initial,
        controller=TargetFractionOracleController(initial[0], microsteps=2),
        **common,
    )
    assert replay.diagnostics["controller"] == first.diagnostics["controller"]


def _forward_fake_sampler(
    head_fraction: torch.Tensor,
    exposure: torch.Tensor,
    **_: object,
) -> SimpleNamespace:
    del exposure
    shape = head_fraction.shape
    return SimpleNamespace(
        later_head_fraction=head_fraction.clone(),
        denoising_target=torch.zeros_like(head_fraction),
        certificate_codes=torch.full(shape, 15, dtype=torch.uint8, device=head_fraction.device),
        fallback_mask=torch.zeros(shape, dtype=torch.bool, device=head_fraction.device),
        strengthened_mask=torch.zeros(shape, dtype=torch.bool, device=head_fraction.device),
        mode_counts=torch.ones(shape, dtype=torch.int32, device=head_fraction.device),
        prefix_bits=torch.full(shape, 64, dtype=torch.int32, device=head_fraction.device),
        arb_fallback_reason_codes=torch.zeros(shape, dtype=torch.uint8, device=head_fraction.device),
        diagnostics={},
    )


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _strict_forward_record(
    *,
    histogram: dict[str, int] | None = None,
    shard_index: int = 0,
    input_sha256: str | None = None,
    output_health: bool = True,
) -> dict[str, object]:
    codes = {"15": 100} if histogram is None else dict(histogram)
    transition_count = sum(codes.values())
    active_count = sum(
        count
        for code, count in ((int(key), value) for key, value in codes.items())
        if code != 0 and code & 0xF == 0xF
    )
    structural_noop_count = codes.get("0", 0)
    diagnostics: dict[str, object] = {
        "transition_count": transition_count,
        "certified_count": active_count,
        "uncertified_count": structural_noop_count,
        "certificate_code_counts": codes,
        "fallback_count": 0,
        "fallback_elapsed_seconds": 0.0,
        "maximum_mass_error": 0.0,
        **{name: 0 for name in EXACT_FORWARD_FORBIDDEN_COUNTERS},
    }
    record: dict[str, object] = {
        "schema": "d0-jacobi-rb-tangent-rollout-v1-forward-shard",
        "schema_version": 2,
        "committed": 1,
        "shard_index": shard_index,
        "start_step": shard_index * 8,
        "step_count": 8,
        "path_ids": [0xFA100],
        "input_state_sha256": input_sha256 or _digest(f"input-{shard_index}"),
        "output_state_sha256": _digest(f"output-{shard_index}"),
        "state_file_sha256": _digest(f"state-file-{shard_index}"),
        "elapsed_seconds": 1.0,
        "transition_count": transition_count,
        "maximum_pair_mass_error": 0.0,
        "peak_cuda_memory_allocated_bytes": 0,
        "total_cuda_memory_bytes": 0,
        "scheduler_record": {"diagnostics": diagnostics},
    }
    if output_health:
        record.update(
            {
                "output_state_nonfinite_count": 0,
                "output_state_negative_count": 0,
                "maximum_output_state_mass_error": 0.0,
            }
        )
    return record


def _aggregate_one(record: dict[str, object]) -> dict[str, object]:
    diagnostics = record["scheduler_record"]["diagnostics"]  # type: ignore[index]
    return aggregate_exact_forward_shards(
        [record],
        expected_shard_count=1,
        expected_transition_count=int(
            record["transition_count"]
            if "transition_count" in record
            else diagnostics.get("transition_count", 100)
        ),
        expected_path_ids=(0xFA100,),
    )


def test_strict_forward_aggregate_reconstructs_immutable_predecessor() -> None:
    root = (
        FAILED_FUSED_PREFLIGHT
        / "preflight"
        / "forward_anchor"
        / "forward_shards"
        / "fused-preflight-anchor"
    )
    assert root.is_dir(), "immutable predecessor forward shards are unavailable"
    records = [
        json.loads((root / f"shard-{index:04d}.json").read_text(encoding="utf-8"))
        for index in range(64)
    ]
    aggregate = aggregate_exact_forward_shards(
        records,
        expected_shard_count=64,
        expected_transition_count=1_404_928,
        expected_path_ids=(0xFC001,),
    )
    assert aggregate["transition_count"] == 1_404_928
    assert aggregate["active_count"] == 1_404_928
    assert aggregate["certified_count"] == 1_404_928
    assert aggregate["structural_noop_count"] == 0
    assert aggregate["uncertified_count"] == 0
    assert aggregate["authorized_count"] == 1_404_928
    assert aggregate["fallback_count"] == 0
    assert aggregate["certificate_code_counts"] == {"15": 1_404_928}
    assert aggregate["forbidden_event_count"] == 0
    assert aggregate["forbidden_counter_schema"] == (
        EXACT_FORWARD_FORBIDDEN_COUNTERS_VERSION
    )
    assert aggregate["forbidden_counter_names"] == list(
        EXACT_FORWARD_FORBIDDEN_COUNTERS
    )
    assert aggregate["output_state_health_recorded"] == 0
    assert aggregate["output_state_nonfinite_count"] is None
    assert aggregate["maximum_pair_mass_error"] == pytest.approx(
        3.469446951953614e-18
    )
    assert aggregate["maximum_simplex_mass_error"] == pytest.approx(
        2.220446049250313e-16
    )


def test_strict_forward_aggregate_derives_structural_noop_semantics() -> None:
    aggregate = _aggregate_one(
        _strict_forward_record(histogram={"0": 4, "15": 96})
    )
    assert aggregate["transition_count"] == 100
    assert aggregate["active_count"] == 96
    assert aggregate["certified_count"] == 96
    assert aggregate["structural_noop_count"] == 4
    assert aggregate["uncertified_count"] == 4
    assert aggregate["authorized_count"] == 100
    assert aggregate["certificate_fraction"] == 1.0
    assert aggregate["authorization_fraction"] == 1.0


def test_strict_forward_aggregate_rejects_unknown_nonzero_certificate_code() -> None:
    record = _strict_forward_record(histogram={"14": 100})
    with pytest.raises(ExactForwardShardAggregateError) as caught:
        _aggregate_one(record)
    assert caught.value.failure_domain == "numerical_integrity"


@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("record", "transition_count"),
        ("record", "schema"),
        ("record", "schema_version"),
        ("record", "elapsed_seconds"),
        ("record", "peak_cuda_memory_allocated_bytes"),
        ("record", "total_cuda_memory_bytes"),
        ("record", "maximum_pair_mass_error"),
        ("record", "input_state_sha256"),
        ("record", "output_state_sha256"),
        ("record", "state_file_sha256"),
        ("diagnostics", "transition_count"),
        ("diagnostics", "certified_count"),
        ("diagnostics", "uncertified_count"),
        ("diagnostics", "certificate_code_counts"),
        ("diagnostics", "fallback_count"),
        ("diagnostics", "fallback_elapsed_seconds"),
        ("diagnostics", "maximum_mass_error"),
        *[("diagnostics", name) for name in EXACT_FORWARD_FORBIDDEN_COUNTERS],
    ],
)
def test_strict_forward_aggregate_rejects_missing_required_fields(
    location: str, field: str
) -> None:
    record = _strict_forward_record()
    target = (
        record
        if location == "record"
        else record["scheduler_record"]["diagnostics"]  # type: ignore[index]
    )
    del target[field]  # type: ignore[index]
    with pytest.raises(ExactForwardShardAggregateError) as caught:
        _aggregate_one(record)
    assert caught.value.failure_domain == "implementation_contract"
    assert caught.value.failure_code == "exact_forward_shard_contract_invalid"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("certified_count", 99),
        ("uncertified_count", 1),
        ("transition_count", 99),
    ],
)
def test_strict_forward_aggregate_rejects_count_disagreement(
    field: str, value: int
) -> None:
    record = _strict_forward_record()
    record["scheduler_record"]["diagnostics"][field] = value  # type: ignore[index]
    with pytest.raises(ExactForwardShardAggregateError) as caught:
        _aggregate_one(record)
    assert caught.value.failure_domain == "numerical_integrity"


@pytest.mark.parametrize("field", EXACT_FORWARD_FORBIDDEN_COUNTERS)
def test_strict_forward_aggregate_rejects_forbidden_events(field: str) -> None:
    record = _strict_forward_record()
    record["scheduler_record"]["diagnostics"][field] = 1  # type: ignore[index]
    with pytest.raises(ExactForwardShardAggregateError) as caught:
        _aggregate_one(record)
    assert caught.value.failure_domain == "numerical_integrity"


def test_strict_forward_aggregate_rejects_broken_chain_and_state_health() -> None:
    first = _strict_forward_record(shard_index=0)
    second = _strict_forward_record(shard_index=1, input_sha256=_digest("wrong"))
    with pytest.raises(ExactForwardShardAggregateError) as caught:
        aggregate_exact_forward_shards(
            [first, second],
            expected_shard_count=2,
            expected_transition_count=200,
            expected_path_ids=(0xFA100,),
        )
    assert caught.value.failure_domain == "numerical_integrity"

    invalid_state = _strict_forward_record()
    invalid_state["output_state_negative_count"] = 1
    with pytest.raises(ExactForwardShardAggregateError) as caught:
        _aggregate_one(invalid_state)
    assert caught.value.failure_domain == "numerical_integrity"


def test_strict_forward_aggregate_rejects_partial_output_health_schema() -> None:
    record = _strict_forward_record(output_health=False)
    record["output_state_nonfinite_count"] = 0
    with pytest.raises(ExactForwardShardAggregateError) as caught:
        _aggregate_one(record)
    assert caught.value.failure_domain == "implementation_contract"


def test_v2_forward_aggregate_requires_complete_output_health_schema() -> None:
    record = _strict_forward_record(output_health=False)
    with pytest.raises(ExactForwardShardAggregateError) as caught:
        _aggregate_one(record)
    assert caught.value.failure_domain == "implementation_contract"

    record["schema_version"] = 1
    aggregate = _aggregate_one(record)
    assert aggregate["output_state_health_recorded"] == 0


def test_forward_restart_commits_exact_anchor_and_skips_sampler(tmp_path: Path) -> None:
    initial = np.full((1, 784), 1.0 / 784.0, dtype=np.float64)
    result = run_forward_trajectory(
        initial,
        anchor_steps=(7,),
        output_dir=tmp_path,
        trajectory_name="development",
        path_ids=(0xFA100,),
        root_seed=261401,
        profile=JacobiRBCudaProfile(),
        sampler=_forward_fake_sampler,
        step_limit=8,
        device="cpu",
    )
    assert np.array_equal(result.anchors[7], initial)
    assert result.diagnostics["transition_count"] == result.transition_count
    assert result.diagnostics["active_count"] == result.diagnostics["certified_count"]
    assert (
        result.diagnostics["active_count"]
        + result.diagnostics["structural_noop_count"]
        == result.diagnostics["transition_count"]
    )
    assert result.diagnostics["authorized_count"] == result.transition_count
    assert result.diagnostics["authorization_fraction"] == 1.0
    assert result.diagnostics["authorization_semantics"] == (
        "active-lanes-certified-plus-exact-structural-noops-v1"
    )
    assert result.diagnostics["certificate_fraction"] == 1.0
    assert result.diagnostics["maximum_simplex_mass_error"] <= 2e-12
    assert result.diagnostics["maximum_pair_mass_error"] <= 2e-12
    assert result.diagnostics["peak_cuda_memory_bytes"] == 0
    assert result.diagnostics["total_cuda_memory_bytes"] == 0
    assert result.diagnostics["output_state_health_recorded"] == 1
    assert result.diagnostics["output_state_nonfinite_count"] == 0
    assert result.diagnostics["output_state_negative_count"] == 0
    assert result.diagnostics["maximum_output_state_mass_error"] <= 2e-12
    committed_record = json.loads(
        (
            tmp_path
            / "forward_shards"
            / "development"
            / "shard-0000.json"
        ).read_text(encoding="utf-8")
    )
    assert committed_record["output_state_nonfinite_count"] == 0
    assert committed_record["output_state_negative_count"] == 0
    assert committed_record["maximum_output_state_mass_error"] <= 2e-12
    with np.load(
        tmp_path / "forward_shards" / "development" / "shard-0000.npz",
        allow_pickle=False,
    ) as archive:
        persisted_state = np.array(archive["state"], copy=True, order="C")
    assert committed_record["output_state_sha256"] == rollout_array_sha256(
        persisted_state
    )

    def fail_if_called(*_: object, **__: object) -> object:
        raise AssertionError("committed forward shard was recomputed")

    # The persisted sampler binding protects accidental backend changes.
    with pytest.raises(TangentRolloutContractError, match="sampler_binding"):
        run_forward_trajectory(
            initial,
            anchor_steps=(7,),
            output_dir=tmp_path,
            trajectory_name="development",
            path_ids=(0xFA100,),
            root_seed=261401,
            profile=JacobiRBCudaProfile(),
            sampler=fail_if_called,
            step_limit=8,
            device="cpu",
        )
    replay = run_forward_trajectory(
        initial,
        anchor_steps=(7,),
        output_dir=tmp_path,
        trajectory_name="development",
        path_ids=(0xFA100,),
        root_seed=261401,
        profile=JacobiRBCudaProfile(),
        sampler=_forward_fake_sampler,
        step_limit=8,
        device="cpu",
    )
    assert np.array_equal(replay.final_state, result.final_state)

    record = tmp_path / "forward_shards" / "development" / "shard-0000.json"
    record.unlink()
    recovered = run_forward_trajectory(
        initial,
        anchor_steps=(7,),
        output_dir=tmp_path,
        trajectory_name="development",
        path_ids=(0xFA100,),
        root_seed=261401,
        profile=JacobiRBCudaProfile(),
        sampler=_forward_fake_sampler,
        step_limit=8,
        device="cpu",
    )
    assert np.array_equal(recovered.final_state, result.final_state)
    assert record.is_file()


def _write_checkpoint_run(root: Path, *, seed: int = 261372, update: int = 3700) -> None:
    model = FrequencyOneCoordinateZeroBaselinePredictor(zero_residual=False)
    state = model.state_dict()
    state_hash = state_dict_sha256(state)
    relative = Path("checkpoints") / "physical" / f"seed-{seed}" / f"update-{update:04d}.pt"
    path = root / relative
    path.parent.mkdir(parents=True)
    torch.save(
        {
            "schema": "test-candidate",
            "schema_version": 1,
            "seed": seed,
            "update": update,
            "state_dict": state,
            "state_sha256": state_hash,
        },
        path,
    )
    candidate = {
        "seed": seed,
        "update": update,
        "checkpoint_path": relative.as_posix(),
        "checkpoint_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "state_sha256": state_hash,
    }
    body = {"schema": "test-inventory", "checkpoints": [candidate]}
    (root / "candidate_inventory.json").write_text(
        json.dumps({**body, "semantic_sha256": semantic_sha256(body)}), encoding="utf-8"
    )


def test_checkpoint_loader_verifies_inventory_file_state_and_strict_model(tmp_path: Path) -> None:
    _write_checkpoint_run(tmp_path)
    loaded = load_verified_frequency1_checkpoint(tmp_path)
    assert (loaded.seed, loaded.update) == (261372, 3700)
    assert loaded.model.training is False
    assert all(not value.requires_grad for value in loaded.model.parameters())
    with pytest.raises(TangentRolloutContractError, match="absent or ambiguous"):
        load_verified_frequency1_checkpoint(tmp_path, expected_update=3600)
    loaded.checkpoint_path.write_bytes(loaded.checkpoint_path.read_bytes() + b"tamper")
    with pytest.raises(TangentRolloutContractError, match="file hash"):
        load_verified_frequency1_checkpoint(tmp_path)


def _source_hash(value: np.ndarray) -> str:
    measured = np.ascontiguousarray(value.astype(np.float32))
    digest = hashlib.sha256()
    digest.update(str(measured.shape).encode("ascii"))
    digest.update(measured.tobytes())
    return digest.hexdigest()


def _write_source(root: Path) -> tuple[np.ndarray, np.ndarray]:
    image = np.arange(1, 785, dtype=np.float64)
    image /= image.sum()
    mixed = 0.65 * image + 0.35 / 784.0
    archive = root / "source_image.npz"
    np.savez(archive, image=image, mixed_target=mixed)
    metadata = {
        "label": 3,
        "dataset_index": 7,
        "lambda_mix": 0.35,
        "image_sha256": _source_hash(image),
        "mixed_target_sha256": _source_hash(mixed),
        "npz_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "npz_size": archive.stat().st_size,
    }
    (root / "source_image.json").write_text(json.dumps(metadata), encoding="utf-8")
    return image, mixed


def test_source_loader_metrics_and_fixed_scale_rendering_are_raw(tmp_path: Path) -> None:
    image, mixed = _write_source(tmp_path)
    loaded = load_verified_source_target(tmp_path)
    assert np.array_equal(loaded.source_image, image)
    state = mixed.copy()
    state[0] += 0.01
    state[1] -= 0.01
    metric = raw_state_metrics(state, mixed)
    assert metric.squared_l2_error == pytest.approx(0.0002)
    scale = fixed_rendering_scale(image, mixed, 0.35)
    raw_a = render_raw_density(state, scale)
    raw_b = render_raw_density(2.0 * state, scale)
    assert raw_a.dtype == np.uint8 and raw_a.shape == (28, 28)
    assert np.any(raw_a != raw_b)  # no per-image autoscaling
    demixed = render_background_demixed(mixed, scale)
    assert np.array_equal(demixed, render_background_demixed(mixed.copy(), scale))
    assert np.array_equal(demixed, render_source_image(image, scale))

    # Display clipping cannot alter the scientific metric.
    clipped = np.clip(state, 0.0, scale.raw_density_scale)
    assert raw_state_metrics(clipped, mixed).squared_l2_error != metric.squared_l2_error

    archive = tmp_path / "source_image.npz"
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(TangentRolloutContractError, match="archive binding"):
        load_verified_source_target(tmp_path)


def test_core_has_no_diagnostic_cli_or_confirmation_import() -> None:
    source = Path("mnist/d0_jacobi_rb_tangent_rollout.py").read_text(encoding="utf-8")
    assert "import mnist.diag_" not in source
    assert "confirmation_index" not in source
