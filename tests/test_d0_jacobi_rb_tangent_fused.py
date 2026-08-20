from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist.d0_jacobi_rb_boundary_tangent_fused import (
    controlled_reverse_phase_tangent,
    controlled_reverse_phase_tangent_fused,
)
from mnist.d0_jacobi_rb_learnability import EDGES_PER_PHASE, ModelInputs, STATE_SIZE
from mnist.d0_jacobi_rb_reverse_controller import NAMESPACE_VERSION
from mnist.d0_jacobi_rb_tangent_fused import (
    build_fused_transition_id_plan,
    CANDIDATE_REFERENCE_CONTRACT,
    CandidateApproximateFusedReference,
    DEFERRED_REFERENCE_RNG_ROLES,
    DeferredCertifiedFusedReference,
    FusedRowSpec,
    FusedTangentContractError,
    FusedTangentControllerBank,
    fused_transition_ids,
    join_fused_family_rows,
    prepare_deferred_reference_rng_seed_map,
    run_fused_reverse_family,
    run_fused_reverse_shard,
    validate_fused_row_specs,
)
from mnist.d0_jacobi_rb_tangent_rollout import (
    CertifiedExploratoryReference,
    ScaledTangentScoreController,
    SignedDiagnosticTangentScoreController,
    TargetFractionOracleController,
    ZeroTangentScoreController,
)
from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile
from mnist.d0_jacobi_rb_cuda_deferred import (
    prepare_alpha1_rb_transition_batch_cuda_candidate,
    prepare_alpha1_rb_transition_batch_cuda_deferred,
)
from mnist import d0_jacobi_rb_boundary_tangent_fused as boundary_tangent_module
from mnist import d0_jacobi_rb_reverse_controller as reverse_controller_module
from mnist import d0_jacobi_rb_tangent_rollout as tangent_rollout_module


class _ConstantController(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)
        self.calls = 0

    def score_prediction(self, inputs: ModelInputs) -> torch.Tensor:
        self.calls += 1
        return torch.full(
            (inputs.batch_size, EDGES_PER_PHASE),
            self.value,
            dtype=torch.float64,
            device=inputs.later_full_state.device,
        )

    def score_prediction_deferred(self, inputs: ModelInputs) -> torch.Tensor:
        return self.score_prediction(inputs)


class _NaNController(_ConstantController):
    def score_prediction_deferred(self, inputs: ModelInputs) -> torch.Tensor:
        return torch.full(
            (inputs.batch_size, EDGES_PER_PHASE),
            float("nan"),
            dtype=torch.float64,
            device=inputs.later_full_state.device,
        )


class _PrevalidatedConstantController(_ConstantController):
    def score_prediction_prevalidated(self, inputs: ModelInputs) -> torch.Tensor:
        return self.score_prediction(inputs)


def test_signed_diagnostic_row_is_separately_typed_and_dispatched() -> None:
    path = 0xFB300
    specs = (
        FusedRowSpec(
            "v4-plus-0p5",
            path,
            "learned",
            "v4-plus",
            "short",
            0.5,
        ),
        FusedRowSpec(
            "v4-minus-0p5",
            path,
            "signed_diagnostic",
            "v4-minus",
            "short",
            -0.5,
        ),
    )
    base = _PrevalidatedConstantController(1.25)
    bank = FusedTangentControllerBank(
        specs,
        {
            "v4-plus-0p5": ScaledTangentScoreController(base, 0.5),
            "v4-minus-0p5": SignedDiagnosticTangentScoreController(base, -0.5),
        },
    )
    bank.prepare_device("cpu")
    state = _state(2)
    inputs = ModelInputs(
        later_full_state=state.to(dtype=torch.float32),
        reverse_time=torch.full((2,), 0.5, dtype=torch.float64),
        phase=torch.zeros(2, dtype=torch.long),
        color=torch.zeros(2, dtype=torch.long),
        duration=torch.full((2,), 1.0 / 128.0, dtype=torch.float32),
        label=torch.full((2,), 3, dtype=torch.long),
    )
    score = bank.score_prediction(inputs)
    assert torch.equal(score[1], -score[0])


@pytest.mark.parametrize("gain", [None, 0.0, 0.5, float("-inf"), float("nan")])
def test_signed_diagnostic_fused_row_requires_finite_negative_gain(
    gain: float | None,
) -> None:
    with pytest.raises(FusedTangentContractError):
        FusedRowSpec(
            "signed-row",
            0xFB300,
            "signed_diagnostic",
            "v4-minus",
            "short",
            gain,
        )


def test_signed_diagnostic_wrapper_cannot_hide_in_a_production_row() -> None:
    spec = FusedRowSpec(
        "learned-row", 0xFB300, "learned", "v4-plus", "short", 0.5
    )
    controller = SignedDiagnosticTangentScoreController(
        _PrevalidatedConstantController(1.0), -0.5
    )
    with pytest.raises(FusedTangentContractError):
        FusedTangentControllerBank((spec,), {"learned-row": controller})


@pytest.mark.parametrize(
    ("kind", "spec_gain", "wrapper_gain"),
    [("learned", 0.5, 1.0), ("signed_diagnostic", -0.5, -1.0)],
)
def test_fused_row_gain_must_equal_controller_wrapper_gain(
    kind: str, spec_gain: float, wrapper_gain: float
) -> None:
    spec = FusedRowSpec(
        "gain-bound-row", 0xFB300, kind, "gain-bound", "short", spec_gain
    )
    base = _PrevalidatedConstantController(1.0)
    controller = (
        ScaledTangentScoreController(base, wrapper_gain)
        if kind == "learned"
        else SignedDiagnosticTangentScoreController(base, wrapper_gain)
    )
    with pytest.raises(FusedTangentContractError, match="gain differs"):
        FusedTangentControllerBank((spec,), {"gain-bound-row": controller})


@dataclass
class _Batch:
    later_head_fraction: torch.Tensor
    denoising_target: torch.Tensor
    transition_ids: torch.Tensor
    active_mask: torch.Tensor
    authorized_mask: torch.Tensor
    certified_mask: torch.Tensor
    fallback_mask: torch.Tensor
    valid_mask: torch.Tensor
    certificate_codes: torch.Tensor
    prefix_bits: torch.Tensor
    mode_counts: torch.Tensor
    fallback_reason_codes: torch.Tensor
    device_diagnostics: dict[str, torch.Tensor]


class _DeferredSampler:
    def __init__(self, *, force_fallback: bool = False) -> None:
        self.force_fallback = force_fallback
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(
        self,
        head_fraction: torch.Tensor,
        exposure: torch.Tensor,
        *,
        rng_key: object,
        transition_ids: torch.Tensor,
        profile: JacobiRBCudaProfile,
    ) -> _Batch:
        del rng_key, profile
        self.calls.append((transition_ids.clone(), head_fraction.clone()))
        # Stateless deterministic proposal.  Equal IDs and inputs are equal;
        # different states keep their own singleton result.
        id_term = (transition_ids.to(torch.int64) & 7).to(torch.float64) * 1.0e-8
        later = torch.where(
            exposure > 0.0,
            torch.clamp(head_fraction * 0.999999 + id_term, 0.0, 1.0),
            head_fraction,
        )
        active = exposure > 0.0
        authorized = torch.ones_like(active)
        fallback = torch.zeros_like(active)
        if self.force_fallback and fallback.numel():
            fallback.reshape(-1)[0] = True
            authorized.reshape(-1)[0] = False
        zero = torch.zeros((), dtype=torch.int64, device=head_fraction.device)
        diagnostics = {
            name: zero.clone()
            for name in (
                "resource_cap_count",
                "invalid_density_count",
                "approximation_count",
                "clipping_count",
                "correction_count",
                "floor_count",
                "limiter_count",
                "projection_count",
                "renormalization_count",
                "nonfinite_count",
            )
        }
        return _Batch(
            later_head_fraction=later,
            denoising_target=torch.zeros_like(later),
            transition_ids=transition_ids,
            active_mask=active,
            authorized_mask=authorized,
            certified_mask=authorized,
            fallback_mask=fallback,
            valid_mask=torch.ones_like(active),
            certificate_codes=torch.full_like(transition_ids, 15, dtype=torch.uint8),
            prefix_bits=torch.full_like(transition_ids, 64, dtype=torch.int32),
            mode_counts=torch.ones_like(transition_ids, dtype=torch.int32),
            fallback_reason_codes=torch.zeros_like(transition_ids, dtype=torch.uint8),
            device_diagnostics=diagnostics,
        )


@dataclass
class _CandidateBatch:
    earlier_head_fraction: torch.Tensor
    later_head_fraction: torch.Tensor
    denoising_target: torch.Tensor
    exposure: torch.Tensor
    transition_ids: torch.Tensor
    active_mask: torch.Tensor
    structural_noop_mask: torch.Tensor
    approximation_mask: torch.Tensor
    valid_mask: torch.Tensor
    candidate_lower: torch.Tensor
    candidate_upper: torch.Tensor
    device_diagnostics: dict[str, torch.Tensor]


class _CandidateSampler:
    def __init__(
        self,
        *,
        invalid_bracket: bool = False,
        false_certificate: bool = False,
        omit_forbidden_counter: str | None = None,
        malformed_forbidden_counter: str | None = None,
    ) -> None:
        self.invalid_bracket = invalid_bracket
        self.false_certificate = false_certificate
        self.omit_forbidden_counter = omit_forbidden_counter
        self.malformed_forbidden_counter = malformed_forbidden_counter
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(
        self,
        head_fraction: torch.Tensor,
        exposure: torch.Tensor,
        *,
        rng_key: object,
        transition_ids: torch.Tensor,
        profile: JacobiRBCudaProfile,
    ) -> _CandidateBatch:
        del rng_key, profile
        self.calls.append((transition_ids.clone(), head_fraction.clone()))
        active = exposure > 0.0
        noop = exposure == 0.0
        id_term = (transition_ids.to(torch.int64) & 7).to(torch.float64) * 1.0e-8
        later = torch.where(
            active,
            torch.clamp(head_fraction * 0.999999 + id_term, 0.0, 1.0),
            head_fraction,
        )
        radius = torch.where(active, torch.full_like(later, 1.0e-12), torch.zeros_like(later))
        lower = torch.clamp(later - radius, 0.0, 1.0)
        upper = torch.clamp(later + radius, 0.0, 1.0)
        if self.invalid_bracket and lower.numel():
            lower.reshape(-1)[0] = later.reshape(-1)[0] + 0.1
        zero = torch.zeros((), dtype=torch.int64, device=head_fraction.device)
        diagnostics = {
            name: zero.clone()
            for name in (
                "resource_cap_count",
                "invalid_density_count",
                "clipping_count",
                "correction_count",
                "floor_count",
                "limiter_count",
                "projection_count",
                "renormalization_count",
                "nonfinite_count",
            )
        }
        if self.omit_forbidden_counter is not None:
            diagnostics.pop(self.omit_forbidden_counter)
        if self.malformed_forbidden_counter is not None:
            diagnostics[self.malformed_forbidden_counter] = torch.zeros(
                2, dtype=torch.int64, device=head_fraction.device
            )
        result = _CandidateBatch(
            earlier_head_fraction=head_fraction,
            later_head_fraction=later,
            denoising_target=torch.zeros_like(later),
            exposure=exposure,
            transition_ids=transition_ids,
            active_mask=active,
            structural_noop_mask=noop,
            approximation_mask=active,
            valid_mask=torch.ones_like(active),
            candidate_lower=lower,
            candidate_upper=upper,
            device_diagnostics=diagnostics,
        )
        if self.false_certificate:
            result.device_diagnostics["certified_count"] = torch.sum(
                active, dtype=torch.int64
            )
        return result


class _SynchronousSampler:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        head_fraction: torch.Tensor,
        exposure: torch.Tensor,
        *,
        rng_key: object,
        transition_ids: torch.Tensor,
        profile: JacobiRBCudaProfile,
    ) -> SimpleNamespace:
        del rng_key, profile
        self.calls += 1
        active = exposure > 0.0
        zero = torch.zeros((), dtype=torch.int64, device=head_fraction.device)
        return SimpleNamespace(
            later_head_fraction=head_fraction.clone(),
            certified_mask=torch.ones_like(active),
            fallback_mask=torch.zeros_like(active),
            diagnostics={
                "fallback_count": zero,
                "arb_fallback_elapsed_seconds": zero.to(torch.float64),
                **{
                    name: zero.clone()
                    for name in (
                        "resource_cap_count",
                        "invalid_density_count",
                        "approximation_count",
                        "clipping_count",
                        "correction_count",
                        "floor_count",
                        "limiter_count",
                        "projection_count",
                        "renormalization_count",
                        "nonfinite_count",
                    )
                },
            },
        )


def _state(rows: int) -> torch.Tensor:
    value = torch.arange(1, rows * STATE_SIZE + 1, dtype=torch.float64).reshape(
        rows, STATE_SIZE
    )
    return value / torch.sum(value, dim=1, keepdim=True)


def _specs(path: int = 0xFB100) -> tuple[FusedRowSpec, ...]:
    return (
        FusedRowSpec("short-zero", path, "zero", "zero", "short"),
        FusedRowSpec(
            "short-learned",
            path,
            "learned",
            "learned",
            "short",
            1.0,
            {"checkpoint": "fixture"},
        ),
        FusedRowSpec(
            "short-oracle",
            path,
            "oracle",
            "oracle",
            "short",
            None,
            {"target": "fixture"},
        ),
    )


def _bank(specs: tuple[FusedRowSpec, ...], target: np.ndarray | None = None) -> FusedTangentControllerBank:
    target_value = (
        np.full(STATE_SIZE, 1.0 / STATE_SIZE, dtype=np.float64)
        if target is None
        else target
    )
    bank = FusedTangentControllerBank(
        specs,
        {
            "short-learned": _ConstantController(0.2),
            "short-oracle": TargetFractionOracleController(target_value, 2),
        },
    )
    bank.prepare_device("cpu")
    return bank


def _identity_reference(**kwargs: object) -> dict[str, torch.Tensor]:
    return {"later_head_fraction": kwargs["head_fraction"].clone()}  # type: ignore[union-attr]


def _shard_sequence() -> tuple[tuple[int, int], ...]:
    return tuple(
        (step, phase)
        for step in range(7, -1, -1)
        for phase in range(6, -1, -1)
    )


def _phase_plan(specs: tuple[FusedRowSpec, ...], device: torch.device):
    return build_fused_transition_id_plan(
        specs, ((7, 6),), microsteps=2, device=device
    )


def test_row_identity_is_separate_from_duplicate_canonical_path_identity() -> None:
    specs = validate_fused_row_specs(_specs())
    assert len({item.row_key for item in specs}) == 3
    assert len({item.canonical_path_id for item in specs}) == 1
    ids = fused_transition_ids(
        specs,
        outer_step=127,
        phase=6,
        reverse_microstep=0,
        role="reverse_reference_pre_control_M2",
        device="cpu",
    )
    assert ids.shape == (3, EDGES_PER_PHASE)
    assert torch.equal(ids[0], ids[1]) and torch.equal(ids[1], ids[2])
    with pytest.raises(FusedTangentContractError, match="row keys"):
        validate_fused_row_specs((specs[0], specs[0]))
    with pytest.raises(FusedTangentContractError, match="lane cap"):
        validate_fused_row_specs(
            tuple(
                FusedRowSpec(f"row-{index}", index, "zero", "zero", "short")
                for index in range(11)
            )
        )


def test_row_dispatch_is_one_row_and_preserves_zero_learned_oracle() -> None:
    specs = _specs()
    state = _state(3)
    target = np.ascontiguousarray(state[2].numpy(), dtype=np.float64)
    bank = _bank(specs, target)
    plan = _phase_plan(specs, state.device)
    bank.prepare_device(
        state.device,
        matching_tensors=(plan.matching_tails, plan.matching_heads),
    )
    inputs = ModelInputs(
        later_full_state=state.to(torch.float32),
        reverse_time=torch.full((3,), 0.5, dtype=torch.float64),
        phase=torch.zeros(3, dtype=torch.long),
        color=torch.zeros(3, dtype=torch.long),
        duration=torch.full((3,), 0.5, dtype=torch.float32),
        label=torch.full((3,), 3, dtype=torch.long),
    )
    output = bank.score_prediction(inputs)
    assert torch.count_nonzero(output[0]) == 0
    assert torch.equal(output[1], torch.full_like(output[1], 0.2))
    oracle = bank.controllers["short-oracle"]
    expected = oracle.score_prediction(_slice_for_test(inputs, 2))
    assert torch.equal(output[2], expected[0])
    assert bank.controllers["short-learned"].calls == 1


def _slice_for_test(inputs: ModelInputs, row: int) -> ModelInputs:
    return ModelInputs(
        later_full_state=inputs.later_full_state[row : row + 1],
        reverse_time=inputs.reverse_time[row : row + 1],
        phase=inputs.phase[row : row + 1],
        color=inputs.color[row : row + 1],
        duration=inputs.duration[row : row + 1],
        label=inputs.label[row : row + 1],
    )


def test_fused_phase_matches_corresponding_singletons() -> None:
    specs = _specs()
    state = _state(3)
    target = np.ascontiguousarray(state[2].numpy(), dtype=np.float64)
    bank = _bank(specs, target)
    plan = _phase_plan(specs, state.device)
    bank.prepare_device(
        state.device,
        matching_tensors=(plan.matching_tails, plan.matching_heads),
    )
    fused = controlled_reverse_phase_tangent_fused(
        state,
        7,
        6,
        2,
        NAMESPACE_VERSION,
        controller_bank=bank,
        reference_transition=_identity_reference,
        row_keys=tuple(item.row_key for item in specs),
        canonical_path_ids=tuple(item.canonical_path_id for item in specs),
        label=3,
        prebuilt_transition_ids=plan.phase_ids(0),
        prebuilt_matching_tails=plan.matching_tails,
        prebuilt_matching_heads=plan.matching_heads,
    )
    controllers = (
        ZeroTangentScoreController(),
        _ConstantController(0.2),
        TargetFractionOracleController(target, 2),
    )
    for row, controller in enumerate(controllers):
        singleton = controlled_reverse_phase_tangent(
            state[row : row + 1],
            7,
            6,
            2,
            NAMESPACE_VERSION,
            controller=controller,
            reference_transition=_identity_reference,
            path_ids=(specs[row].canonical_path_id,),
            label=3,
        )
        assert torch.equal(fused.state[row], singleton.state[0])


def test_fused_hot_loop_uses_only_prebuilt_ids_matchings_and_dense_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = _specs()
    state = _state(3)
    target = np.ascontiguousarray(state[2].numpy(), dtype=np.float64)
    plan = _phase_plan(specs, state.device)
    bank = _bank(specs, target)
    bank.prepare_device(
        state.device,
        matching_tensors=(plan.matching_tails, plan.matching_heads),
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("fused phase attempted a host synchronization path")

    monkeypatch.setattr(
        reverse_controller_module, "controller_transition_ids", forbidden
    )
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    result = controlled_reverse_phase_tangent_fused(
        state,
        7,
        6,
        2,
        NAMESPACE_VERSION,
        controller_bank=bank,
        reference_transition=_identity_reference,
        row_keys=tuple(item.row_key for item in specs),
        canonical_path_ids=tuple(item.canonical_path_id for item in specs),
        label=3,
        prebuilt_transition_ids=plan.phase_ids(0),
        prebuilt_matching_tails=plan.matching_tails,
        prebuilt_matching_heads=plan.matching_heads,
    )
    assert result.state.shape == state.shape

    phase_source = inspect.getsource(
        boundary_tangent_module.controlled_reverse_phase_tangent_fused
    )
    bank_source = inspect.getsource(FusedTangentControllerBank.score_prediction)
    oracle_source = inspect.getsource(
        tangent_rollout_module.TargetFractionOracleController
        .score_prediction_deferred_prepared
    )
    assert "controller_transition_ids(" not in phase_source
    assert "matching_indices(" not in phase_source
    assert "matching_indices(" not in bank_source
    assert "matching_indices(" not in oracle_source
    for helper in (
        boundary_tangent_module._frozen_score_logistic_fraction_device,
        boundary_tangent_module._scatter_pair_fraction_device,
    ):
        source = inspect.getsource(helper)
        for dynamic_name in ("positive", "negative", "active"):
            assert f"[{dynamic_name}]" not in source


def test_deferred_reference_chunk_and_row_permutation_identity() -> None:
    specs = _specs()
    state = _state(3)
    target = np.ascontiguousarray(state[2].numpy(), dtype=np.float64)

    def run(active_specs: tuple[FusedRowSpec, ...], active_state: torch.Tensor, chunk: int | None):
        sampler = _DeferredSampler()
        reference = DeferredCertifiedFusedReference(
            profile=JacobiRBCudaProfile(),
            root_seed=261402,
            stream_role="fixture",
            sampler=sampler,
            synchronous_sampler=_SynchronousSampler(),
            row_chunk_size=chunk,
        )
        controllers = {
            item.row_key: (
                _ConstantController(0.2)
                if item.controller_kind == "learned"
                else TargetFractionOracleController(target, 2)
            )
            for item in active_specs
            if item.controller_kind != "zero"
        }
        result = run_fused_reverse_shard(
            active_state,
            ((7, 6),),
            row_specs=active_specs,
            controller_bank=FusedTangentControllerBank(active_specs, controllers),
            reference_transition=reference,
        )
        return result.final_state

    full = run(specs, state, None)
    chunked = run(specs, state, 1)
    assert np.array_equal(full, chunked)
    permutation = (2, 0, 1)
    permuted_specs = tuple(specs[index] for index in permutation)
    permuted = run(permuted_specs, state[list(permutation)], 2)
    inverse = np.argsort(np.asarray(permutation))
    assert np.array_equal(full, permuted[inverse])


def test_candidate_reference_labels_approximation_and_is_chunk_permutation_invariant() -> None:
    specs = _specs()
    state = _state(3)
    target = np.ascontiguousarray(state[2].numpy(), dtype=np.float64)

    def run(
        active_specs: tuple[FusedRowSpec, ...],
        active_state: torch.Tensor,
        chunk: int | None,
    ):
        reference = CandidateApproximateFusedReference(
            profile=JacobiRBCudaProfile(),
            root_seed=261402,
            stream_role="candidate-fixture",
            sampler=_CandidateSampler(),
            row_chunk_size=chunk,
        )
        result = run_fused_reverse_shard(
            active_state,
            ((7, 6),),
            row_specs=active_specs,
            controller_bank=_bank(active_specs, target),
            reference_transition=reference,
            reference_contract="candidate_approximate",
        )
        return result

    full = run(specs, state, None)
    chunked = run(specs, state, 1)
    assert np.array_equal(full.final_state, chunked.final_state)
    permutation = (2, 0, 1)
    permuted_specs = tuple(specs[index] for index in permutation)
    permuted = run(permuted_specs, state[list(permutation)], 2)
    inverse = np.argsort(np.asarray(permutation))
    assert np.array_equal(full.final_state, permuted.final_state[inverse])

    reference = full.diagnostics["reference"]
    assert reference["reference_contract"] == CANDIDATE_REFERENCE_CONTRACT
    assert reference["certificate_fraction"] == "not_applicable"
    assert reference["approximation_count"] == reference["active_count"]
    assert reference["active_count"] + reference["structural_noop_count"] == reference["transition_count"]
    assert reference["invalid_count"] == 0
    assert reference["needs_synchronous_replay"] == 0
    assert full.synchronous_replay_performed == 0
    assert full.diagnostics["certificate_fraction"] == "not_applicable"
    assert full.diagnostics["maximum_mass_error"] <= 2.0e-12
    assert not any(
        "certified" in name or "authorized" in name for name in reference
    )
    assert all(
        row["reference_certificate_fraction"] == "not_applicable"
        and row["reference_approximation_count"] == row["reference_active_count"]
        for row in full.per_row_diagnostics
    )


def test_candidate_reference_structural_noop_and_integrity_failures() -> None:
    width = EDGES_PER_PHASE
    head_fraction = torch.linspace(0.0, 1.0, width, dtype=torch.float64).reshape(1, -1)
    exposure = torch.full_like(head_fraction, 0.001)
    exposure[:, ::2] = 0.0
    transition_ids = torch.arange(width, dtype=torch.int64).to(torch.uint64).reshape(1, -1)
    candidate = CandidateApproximateFusedReference(
        profile=JacobiRBCudaProfile(),
        root_seed=261402,
        stream_role="candidate-noop",
        sampler=_CandidateSampler(),
    )
    output = candidate(
        head_fraction=head_fraction,
        exposure=exposure,
        transition_ids=transition_ids,
        role="reverse_reference_pre_control_M2",
    )
    assert torch.equal(output.later_head_fraction[:, ::2], head_fraction[:, ::2])
    record = candidate.finalize_shard({})
    assert record["transition_count"] == width
    assert record["active_count"] == width // 2
    assert record["structural_noop_count"] == width // 2
    assert record["approximation_count"] == width // 2
    assert record["invalid_count"] == 0

    specs = _specs()
    state = _state(3)
    target = np.ascontiguousarray(state[2].numpy(), dtype=np.float64)
    invalid = CandidateApproximateFusedReference(
        profile=JacobiRBCudaProfile(),
        root_seed=261402,
        stream_role="candidate-invalid-bracket",
        sampler=_CandidateSampler(invalid_bracket=True),
    )
    with pytest.raises(FusedTangentContractError, match="candidate reference integrity"):
        run_fused_reverse_shard(
            state,
            ((7, 6),),
            row_specs=specs,
            controller_bank=_bank(specs, target),
            reference_transition=invalid,
            reference_contract="candidate_approximate",
        )

    false_certificate = CandidateApproximateFusedReference(
        profile=JacobiRBCudaProfile(),
        root_seed=261402,
        stream_role="candidate-false-certificate",
        sampler=_CandidateSampler(false_certificate=True),
    )
    with pytest.raises(FusedTangentContractError, match="authorizing fields"):
        run_fused_reverse_shard(
            state,
            ((7, 6),),
            row_specs=specs,
            controller_bank=_bank(specs, target),
            reference_transition=false_certificate,
            reference_contract="candidate_approximate",
        )


@pytest.mark.parametrize(
    "counter",
    (
        "resource_cap_count",
        "invalid_density_count",
        "clipping_count",
        "correction_count",
        "floor_count",
        "limiter_count",
        "projection_count",
        "renormalization_count",
        "nonfinite_count",
    ),
)
def test_candidate_reference_requires_every_forbidden_counter(counter: str) -> None:
    width = EDGES_PER_PHASE
    head_fraction = torch.full((1, width), 0.5, dtype=torch.float64)
    exposure = torch.full_like(head_fraction, 0.001)
    transition_ids = torch.arange(width, dtype=torch.int64).to(torch.uint64).reshape(1, -1)
    candidate = CandidateApproximateFusedReference(
        profile=JacobiRBCudaProfile(),
        root_seed=261402,
        stream_role="candidate-missing-forbidden-counter",
        sampler=_CandidateSampler(omit_forbidden_counter=counter),
    )
    candidate(
        head_fraction=head_fraction,
        exposure=exposure,
        transition_ids=transition_ids,
        role="reverse_reference_pre_control_M2",
    )
    with pytest.raises(
        FusedTangentContractError,
        match=f"omit forbidden counter {counter}",
    ):
        candidate.finalize_shard({})


def test_candidate_reference_rejects_malformed_forbidden_counter() -> None:
    width = EDGES_PER_PHASE
    head_fraction = torch.full((1, width), 0.5, dtype=torch.float64)
    exposure = torch.full_like(head_fraction, 0.001)
    transition_ids = torch.arange(width, dtype=torch.int64).to(torch.uint64).reshape(1, -1)
    candidate = CandidateApproximateFusedReference(
        profile=JacobiRBCudaProfile(),
        root_seed=261402,
        stream_role="candidate-malformed-forbidden-counter",
        sampler=_CandidateSampler(
            malformed_forbidden_counter="invalid_density_count"
        ),
    )
    candidate(
        head_fraction=head_fraction,
        exposure=exposure,
        transition_ids=transition_ids,
        role="reverse_reference_pre_control_M2",
    )
    with pytest.raises(
        FusedTangentContractError,
        match="invalid_density_count is not a scalar int64 tensor",
    ):
        candidate.finalize_shard({})


def test_exact_reference_contract_default_remains_implicit() -> None:
    signature = inspect.signature(run_fused_reverse_shard)
    assert signature.parameters["reference_contract"].default == "certified_exact"
    family_signature = inspect.signature(run_fused_reverse_family)
    assert family_signature.parameters["reference_contract"].default == "certified_exact"
    assert "reference_contract" not in DeferredCertifiedFusedReference(
        profile=JacobiRBCudaProfile(),
        root_seed=261402,
        stream_role="exact-implicit",
        sampler=_DeferredSampler(),
        synchronous_sampler=_SynchronousSampler(),
    ).__dict__
    hot_source = inspect.getsource(DeferredCertifiedFusedReference.__call__) + inspect.getsource(
        CandidateApproximateFusedReference.__call__
    )
    for forbidden in (".cpu(", ".numpy(", ".item(", "cuda.synchronize("):
        assert forbidden not in hot_source


def test_unresolved_speculation_replays_whole_shard_synchronously() -> None:
    specs = _specs()
    state = _state(3)
    target = np.ascontiguousarray(state[2].numpy(), dtype=np.float64)
    synchronous = _SynchronousSampler()
    reference = DeferredCertifiedFusedReference(
        profile=JacobiRBCudaProfile(),
        root_seed=261402,
        stream_role="replay",
        sampler=_DeferredSampler(force_fallback=True),
        synchronous_sampler=synchronous,
    )
    result = run_fused_reverse_shard(
        state,
        ((7, 6),),
        row_specs=specs,
        controller_bank=_bank(specs, target),
        reference_transition=reference,
    )
    assert result.synchronous_replay_performed == 1
    assert synchronous.calls == 4
    reference_record = result.diagnostics["reference"]
    assert reference_record["transition_count"] == 3 * EDGES_PER_PHASE * 4
    assert reference_record["active_count"] == reference_record["transition_count"]
    assert reference_record["structural_noop_count"] == 0
    assert reference_record["certified_count"] == reference_record["active_count"]
    assert reference_record["unauthorized_count"] == 0
    assert reference_record["invalid_count"] == 0
    assert sum(row["active_count"] for row in reference_record["per_row"]) == reference_record["active_count"]
    assert all(
        row["structural_noop_count"] == 0
        and row["certified_count"] == row["active_count"]
        for row in reference_record["per_row"]
    )
    # The replay reference is identity, so compare with the identity fused phase.
    plan = _phase_plan(specs, state.device)
    expected_bank = _bank(specs, target)
    expected_bank.prepare_device(
        state.device,
        matching_tensors=(plan.matching_tails, plan.matching_heads),
    )
    expected = controlled_reverse_phase_tangent_fused(
        state,
        7,
        6,
        2,
        NAMESPACE_VERSION,
        controller_bank=expected_bank,
        reference_transition=_identity_reference,
        row_keys=tuple(item.row_key for item in specs),
        canonical_path_ids=tuple(item.canonical_path_id for item in specs),
        label=3,
        prebuilt_transition_ids=plan.phase_ids(0),
        prebuilt_matching_tails=plan.matching_tails,
        prebuilt_matching_heads=plan.matching_heads,
    )
    assert np.array_equal(result.final_state, expected.state.numpy())


def test_family_restart_orphan_replay_and_committed_skip(tmp_path: Path) -> None:
    specs = _specs()
    state = _state(3)
    target = np.ascontiguousarray(state[2].numpy(), dtype=np.float64)
    calls = {"factories": 0, "references": 0}
    resource_plans: list[object] = []

    class CountingIdentity:
        def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
            calls["references"] += 1
            return _identity_reference(**kwargs)

    def factory(shard_index: int) -> CountingIdentity:
        assert shard_index == 0
        calls["factories"] += 1
        return CountingIdentity()

    kwargs = dict(
        sequence=_shard_sequence(),
        output_dir=tmp_path,
        family_name="development",
        segment_name="short",
        row_specs=specs,
        controller_bank=_bank(specs, target),
        reference_factory=factory,
        controller_binding={"checkpoint": "fixture"},
        rng_binding={"root_seed": 261402, "variant_in_rng_key": 0},
        before_uncommitted_shard=resource_plans.append,
    )
    first = run_fused_reverse_family(state, **kwargs)
    first_counts = dict(calls)
    assert len(resource_plans) == 1
    plan = resource_plans[0]
    assert plan.shard_index == 0
    assert plan.sequence == _shard_sequence()
    assert plan.row_count == len(specs)
    assert plan.transition_count == 56 * 2 * 2 * len(specs) * EDGES_PER_PHASE
    resumed = run_fused_reverse_family(state, **kwargs)
    assert calls == first_counts
    assert len(resource_plans) == 1
    assert np.array_equal(first.final_state, resumed.final_state)

    committed = first.shard_records[0]
    assert committed["elapsed_scope"] == (
        "pre-resource-callback-through-atomic-npz-commit"
    )
    assert committed["elapsed_seconds"] >= committed["execution_elapsed_seconds"]
    assert committed["execution_plan"] == plan.to_record()
    assert "reference_contract" not in committed

    root = tmp_path / "fused_families" / "development" / "short"
    (root / "shard-0000.json").unlink()
    replayed = run_fused_reverse_family(state, **kwargs)
    assert calls["factories"] == first_counts["factories"] + 1
    assert len(resource_plans) == 2
    assert np.array_equal(first.final_state, replayed.final_state)

    (root / "shard-0000.npz").unlink()
    with pytest.raises(FusedTangentContractError, match="lacks its state"):
        run_fused_reverse_family(state, **kwargs)


def test_candidate_family_restart_binds_contract_and_rejects_exact_mix(
    tmp_path: Path,
) -> None:
    specs = _specs()
    state = _state(3)
    target = np.ascontiguousarray(state[2].numpy(), dtype=np.float64)
    factories = {"count": 0}

    def factory(_shard_index: int) -> CandidateApproximateFusedReference:
        factories["count"] += 1
        return CandidateApproximateFusedReference(
            profile=JacobiRBCudaProfile(),
            root_seed=261402,
            stream_role="candidate-family",
            sampler=_CandidateSampler(),
        )

    kwargs = dict(
        sequence=_shard_sequence(),
        output_dir=tmp_path,
        family_name="candidate",
        segment_name="complete-shard",
        row_specs=specs,
        controller_bank=_bank(specs, target),
        reference_factory=factory,
        controller_binding={"checkpoint": "fixture"},
        rng_binding={"root_seed": 261402, "variant_in_rng_key": 0},
        reference_contract="candidate_approximate",
    )
    first = run_fused_reverse_family(state, **kwargs)
    assert factories["count"] == 1
    resumed = run_fused_reverse_family(state, **kwargs)
    assert factories["count"] == 1
    assert np.array_equal(first.final_state, resumed.final_state)
    shard = first.shard_records[0]
    assert shard["reference_contract"] == CANDIDATE_REFERENCE_CONTRACT
    assert shard["diagnostics"]["reference_contract"] == CANDIDATE_REFERENCE_CONTRACT
    assert shard["diagnostics"]["certificate_fraction"] == "not_applicable"
    assert first.diagnostics["reference_contract"] == CANDIDATE_REFERENCE_CONTRACT
    assert first.diagnostics["certificate_fraction"] == "not_applicable"
    assert first.diagnostics["approximation_count"] == first.transition_count

    with pytest.raises(FusedTangentContractError, match="candidate restart prefix"):
        run_fused_reverse_family(
            state,
            **{**kwargs, "reference_contract": "certified_exact"},
        )


def test_family_uses_one_prebuilt_seed_map_and_never_prepares_in_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mnist import d0_jacobi_rb_cuda_deferred as deferred_cuda

    preparation_calls: list[object] = []

    def prepare_seed(*, rng_key: object, prepared: object) -> object:
        assert prepared is backend
        preparation_calls.append(rng_key)
        return object()

    monkeypatch.setattr(
        deferred_cuda,
        "prepare_alpha1_rb_transition_cuda_rng_seed",
        prepare_seed,
    )
    monkeypatch.setattr(
        deferred_cuda,
        "validate_prepared_alpha1_rb_transition_cuda_rng_seed",
        lambda **_kwargs: None,
    )
    backend = object()
    root_seed = 261402
    stream_role = "prebuilt-seed-map-family"
    prepared_rng_seeds = prepare_deferred_reference_rng_seed_map(
        prepared_backend=backend,
        root_seed=root_seed,
        stream_role=stream_role,
    )
    assert tuple(prepared_rng_seeds) == DEFERRED_REFERENCE_RNG_ROLES
    assert len(preparation_calls) == len(DEFERRED_REFERENCE_RNG_ROLES) == 6

    def forbidden_preparation(**_kwargs: object) -> object:
        raise AssertionError("seed preparation entered family/shard timing")

    monkeypatch.setattr(
        deferred_cuda,
        "prepare_alpha1_rb_transition_cuda_rng_seed",
        forbidden_preparation,
    )

    class PreparedSampler(_DeferredSampler):
        def __call__(
            self,
            head_fraction: torch.Tensor,
            exposure: torch.Tensor,
            *,
            rng_key: object,
            transition_ids: torch.Tensor,
            prepared: object,
            prepared_rng_seed: object,
        ) -> _Batch:
            assert prepared is backend
            assert prepared_rng_seed in prepared_rng_seeds.values()
            return super().__call__(
                head_fraction,
                exposure,
                rng_key=rng_key,
                transition_ids=transition_ids,
                profile=JacobiRBCudaProfile(),
            )

    sampler = PreparedSampler()
    specs = _specs()
    state = _state(3)
    target = np.ascontiguousarray(state[2].numpy(), dtype=np.float64)

    def factory(_shard_index: int) -> DeferredCertifiedFusedReference:
        return DeferredCertifiedFusedReference(
            profile=JacobiRBCudaProfile(),
            root_seed=root_seed,
            stream_role=stream_role,
            sampler=sampler,
            synchronous_sampler=_SynchronousSampler(),
            prepared_backend=backend,
            prepared_rng_seeds=prepared_rng_seeds,
        )

    result = run_fused_reverse_family(
        state,
        sequence=_shard_sequence(),
        output_dir=tmp_path,
        family_name="seed-map",
        segment_name="profile",
        row_specs=specs,
        controller_bank=_bank(specs, target),
        reference_factory=factory,
        controller_binding={"checkpoint": "fixture"},
        rng_binding={"root_seed": root_seed, "stream_role": stream_role},
    )
    assert result.transition_count > 0
    assert len(preparation_calls) == 6


def test_join_preserves_prefix_bytes_and_rejects_duplicate_row_keys() -> None:
    prefix_specs = _specs()
    append_specs = tuple(
        FusedRowSpec(
            item.row_key.replace("short", "full"),
            item.canonical_path_id,
            item.controller_kind,
            item.variant,
            "full",
            item.gain,
            item.controller_binding,
        )
        for item in prefix_specs
    )
    prefix = _state(3).numpy()
    append = torch.flip(_state(3), dims=(1,)).numpy()
    joined = join_fused_family_rows(
        prefix,
        prefix_specs,
        append,
        append_specs,
        next_coordinate=(127, 6),
        bindings={"prefix": "fixture"},
    )
    assert np.array_equal(joined.state[:3], prefix)
    assert np.array_equal(joined.state[3:], append)
    assert joined.record["next_coordinate"] == [127, 6]
    with pytest.raises(FusedTangentContractError, match="row keys"):
        join_fused_family_rows(
            prefix,
            prefix_specs,
            append,
            prefix_specs,
            next_coordinate=(127, 6),
        )


def test_invalid_shard_writes_failure_before_any_commit(tmp_path: Path) -> None:
    specs = (
        FusedRowSpec(
            "bad-learned",
            0xFC003,
            "learned",
            "learned",
            "preflight",
            1.0,
            {"fixture": 1},
        ),
    )
    bank = FusedTangentControllerBank(specs, {"bad-learned": _NaNController(0.0)})
    with pytest.raises(FusedTangentContractError, match="health validation"):
        run_fused_reverse_family(
            _state(1),
            sequence=_shard_sequence(),
            output_dir=tmp_path,
            family_name="bad",
            segment_name="short",
            row_specs=specs,
            controller_bank=bank,
            reference_factory=lambda _: _identity_reference,
            controller_binding={"fixture": 1},
            rng_binding={"fixture": 1},
        )
    root = tmp_path / "fused_families" / "bad" / "short"
    assert (root / "shard-0000.failure.json").is_file()
    assert not (root / "shard-0000.json").exists()
    assert not (root / "shard-0000.npz").exists()


def test_resource_callback_failure_is_recorded_before_reference_creation(
    tmp_path: Path,
) -> None:
    specs = _specs()
    factories = 0

    def factory(_: int) -> object:
        nonlocal factories
        factories += 1
        return _identity_reference

    def reject(_: object) -> None:
        raise RuntimeError("fixture resource cap")

    with pytest.raises(RuntimeError, match="resource cap"):
        run_fused_reverse_family(
            _state(3),
            sequence=_shard_sequence(),
            output_dir=tmp_path,
            family_name="resource-failure",
            segment_name="short",
            row_specs=specs,
            controller_bank=_bank(specs),
            reference_factory=factory,
            controller_binding={"fixture": 1},
            rng_binding={"fixture": 1},
            before_uncommitted_shard=reject,
        )
    assert factories == 0
    root = tmp_path / "fused_families" / "resource-failure" / "short"
    failure = next(root.glob("*.failure.json"))
    payload = json.loads(failure.read_text(encoding="utf-8"))
    assert payload["failure_type"] == "RuntimeError"
    assert payload["execution_plan"]["transition_count"] == (
        56 * 2 * 2 * len(specs) * EDGES_PER_PHASE
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_cuda_fused_phase_is_bit_identical_to_exact_singletons() -> None:
    device = torch.device("cuda")
    profile = JacobiRBCudaProfile()
    prepared = prepare_alpha1_rb_transition_batch_cuda_deferred(
        device=device, profile=profile
    )
    specs = _specs()
    state = _state(3)
    target = np.ascontiguousarray(state[2].numpy(), dtype=np.float64)
    root_seed = 261402
    stream_role = "real-cuda-fused-equivalence"
    prepared_rng_seeds = prepare_deferred_reference_rng_seed_map(
        prepared_backend=prepared,
        root_seed=root_seed,
        stream_role=stream_role,
    )
    reference = DeferredCertifiedFusedReference(
        profile=profile,
        root_seed=root_seed,
        stream_role=stream_role,
        prepared_backend=prepared,
        prepared_rng_seeds=prepared_rng_seeds,
    )
    fused = run_fused_reverse_shard(
        state,
        ((7, 6),),
        row_specs=specs,
        controller_bank=_bank(specs, target),
        reference_transition=reference,
        device=device,
    )
    controllers = (
        ZeroTangentScoreController(),
        _ConstantController(0.2),
        TargetFractionOracleController(target, 2),
    )
    singleton_states: list[np.ndarray] = []
    singleton_records: list[dict[str, object]] = []
    for row, controller in enumerate(controllers):
        controller.to(device=device)
        singleton_reference = CertifiedExploratoryReference(
            profile=profile,
            root_seed=root_seed,
            stream_role=stream_role,
        )
        singleton = controlled_reverse_phase_tangent(
            state[row : row + 1].to(device=device),
            7,
            6,
            2,
            NAMESPACE_VERSION,
            controller=controller,
            reference_transition=singleton_reference,
            path_ids=(specs[row].canonical_path_id,),
            label=3,
        )
        singleton_states.append(
            np.ascontiguousarray(
                singleton.state.detach().cpu().numpy(), dtype=np.float64
            )
        )
        singleton_records.append(singleton_reference.record())
    expected = np.concatenate(singleton_states, axis=0)
    assert np.array_equal(fused.final_state, expected)
    assert fused.diagnostics["certificate_fraction"] == 1.0
    assert fused.diagnostics["maximum_launch_lanes"] == 3 * EDGES_PER_PHASE
    for row_record, singleton_record in zip(
        fused.per_row_diagnostics, singleton_records, strict=True
    ):
        assert row_record["reference_certificate_fraction"] == 1.0
        assert row_record["reference_fallback_count"] == int(
            singleton_record["fallback_count"]
        )
        assert row_record["reference_transition_count"] == int(
            singleton_record["transition_count"]
        )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or os.environ.get("D0_RUN_REAL_CUDA_CANDIDATE_SMOKE") != "1",
    reason="real candidate CUDA smoke requires explicit opt-in",
)
def test_real_cuda_candidate_three_row_eight_step_shard_is_integrity_labeled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mnist import d0_jacobi_rb_cuda_deferred as deferred_cuda

    device = torch.device("cuda")
    profile = JacobiRBCudaProfile()
    specs = _specs()
    state = _state(3)
    target = np.ascontiguousarray(state[2].numpy(), dtype=np.float64)
    root_seed = 261402
    stream_role = "real-cuda-candidate-eight-step"
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("candidate fused shard invoked the exact authorizer")

    with monkeypatch.context() as patch:
        patch.setattr(deferred_cuda, "probe_fused_cuda_authorizer", forbidden)
        patch.setattr(deferred_cuda, "launch_fused_cuda_authorizer", forbidden)
        prepared = prepare_alpha1_rb_transition_batch_cuda_candidate(
            device=device, profile=profile
        )
        prepared_rng_seeds = prepare_deferred_reference_rng_seed_map(
            prepared_backend=prepared,
            root_seed=root_seed,
            stream_role=stream_role,
        )
        reference = CandidateApproximateFusedReference(
            profile=profile,
            root_seed=root_seed,
            stream_role=stream_role,
            prepared_backend=prepared,
            prepared_rng_seeds=prepared_rng_seeds,
        )
        result = run_fused_reverse_shard(
            state,
            _shard_sequence(),
            row_specs=specs,
            controller_bank=_bank(specs, target),
            reference_transition=reference,
            reference_contract="candidate_approximate",
            device=device,
        )
    expected_transitions = 56 * 2 * 2 * len(specs) * EDGES_PER_PHASE
    health = result.diagnostics["reference"]
    assert result.transition_count == expected_transitions
    assert health["transition_count"] == expected_transitions
    assert health["active_count"] == expected_transitions
    assert health["approximation_count"] == expected_transitions
    assert health["structural_noop_count"] == 0
    assert health["invalid_count"] == 0
    assert health["certificate_fraction"] == "not_applicable"
    assert result.diagnostics["maximum_mass_error"] <= 2.0e-12
    assert np.isfinite(result.final_state).all()
