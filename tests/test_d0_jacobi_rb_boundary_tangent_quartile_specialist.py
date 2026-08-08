from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from mnist.d0_jacobi_rb_boundary_tangent_quartile_specialist import (
    ALL_CANDIDATE_IDENTITIES,
    CANDIDATE_GRID_SHA256,
    CHECKPOINT_UPDATES,
    GAIN_CROSS_TERM_NONFINITE,
    GAIN_CROSS_TERM_NONPOSITIVE,
    GAIN_ELIGIBLE,
    GAIN_NONFINITE,
    GAIN_OUTSIDE_OPEN_UNIT,
    GAIN_PREDICTION_ENERGY_NONFINITE,
    GAIN_PREDICTION_ENERGY_NONPOSITIVE,
    MODEL_SEEDS_BY_QUARTILE,
    NONZERO_CANDIDATE_IDENTITIES,
    RANK_ELIGIBLE,
    RANK_FINE_CELLS_INSUFFICIENT,
    RANK_GAIN_INELIGIBLE,
    RANK_MIDPOINT_NONPOSITIVE,
    RANK_NONFINITE,
    RANK_PHASE_NONPOSITIVE,
    RANK_POOLED_NONPOSITIVE,
    RANK_Q1_SENTINEL_NONPOSITIVE,
    CandidateIdentity,
    HashBinding,
    NoEligibleQuartileCandidateError,
    QuartileSpecialistBoundaryTangentPredictor,
    QuartileSpecialistContractError,
    SelectedExpert,
    SelectedSystem,
    build_training_rank_record,
    calibrate_training_only_gain,
    candidate_grid_record,
    exact_quartile_target_scale,
    fixed_unit_gain_record,
    gain_record_from_moments,
    reconstruct_forward_outer_quartile,
    scaled_raw_target_mse,
    select_training_rank_candidate,
    select_training_rank_system,
)
from mnist.d0_jacobi_rb_boundary_tangent_zero_baseline import (
    ZeroBaselineBoundaryTangentPredictor,
)
from mnist.d0_jacobi_rb_learnability import (
    EDGES_PER_PHASE,
    PHASE_DURATIONS,
    PHASE_MATCHINGS,
    SELECTED_OUTER_STEPS,
    STATE_SIZE,
    ModelInputs,
)
from mnist.d0_jacobi_rb_reverse_controller import (
    MIDPOINT_FRACTIONS,
    internal_reverse_time,
)


def _inputs(
    steps: list[int],
    phases: list[int] | None = None,
    fractions: list[float] | None = None,
    *,
    zero_last_state: bool = False,
) -> ModelInputs:
    count = len(steps)
    phase_values = phases if phases is not None else [0] * count
    fraction_values = fractions if fractions is not None else [1.0 / 16.0] * count
    states = torch.full((count, STATE_SIZE), 1.0 / STATE_SIZE, dtype=torch.float32)
    if zero_last_state:
        states[-1].zero_()
    return ModelInputs(
        later_full_state=states,
        reverse_time=torch.tensor(
            [
                internal_reverse_time(step, phase, fraction)
                for step, phase, fraction in zip(
                    steps, phase_values, fraction_values, strict=True
                )
            ],
            dtype=torch.float64,
        ),
        phase=torch.tensor(phase_values, dtype=torch.long),
        color=torch.tensor(
            [PHASE_MATCHINGS[phase] for phase in phase_values], dtype=torch.long
        ),
        duration=torch.tensor(
            [PHASE_DURATIONS[phase] for phase in phase_values], dtype=torch.float64
        ),
        label=torch.full((count,), 3, dtype=torch.long),
    )


class _RecordingExpert(ZeroBaselineBoundaryTangentPredictor):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)
        self.batch_sizes: list[int] = []

    def forward(self, inputs: ModelInputs) -> torch.Tensor:
        self.batch_sizes.append(inputs.batch_size)
        return torch.full(
            (inputs.batch_size, EDGES_PER_PHASE),
            self.value,
            dtype=torch.float64,
            device=inputs.later_full_state.device,
        )


def _rank_record(
    candidate: CandidateIdentity,
    *,
    pooled: float = 1.0,
    phases: tuple[float, ...] = (1.0,) * 7,
    midpoints: tuple[float, ...] = (1.0,) * 8,
    cells: tuple[tuple[float, ...], ...] = ((1.0,) * 8,) * 7,
    gain_eligible: bool = True,
):
    if candidate.quartile < 2:
        gain = fixed_unit_gain_record(candidate)
    else:
        gain = gain_record_from_moments(
            candidate,
            cross_term=0.5 if gain_eligible else -0.5,
            prediction_energy=1.0,
            sample_count=100,
        )
    return build_training_rank_record(
        candidate,
        gain,
        pooled_improvement=pooled,
        phase_improvements=phases,
        midpoint_improvements=midpoints,
        fine_cell_improvements=cells,
    )


def test_candidate_grid_is_exact_and_canonical() -> None:
    assert len(NONZERO_CANDIDATE_IDENTITIES) == 480
    assert len(ALL_CANDIDATE_IDENTITIES) == 492
    assert NONZERO_CANDIDATE_IDENTITIES[0] == CandidateIdentity(0, 261_332, 100)
    assert NONZERO_CANDIDATE_IDENTITIES[-1] == CandidateIdentity(3, 261_343, 4_000)
    assert [candidate.update for candidate in ALL_CANDIDATE_IDENTITIES[:41]] == [
        0,
        *CHECKPOINT_UPDATES,
    ]
    assert candidate_grid_record()["seeds_by_quartile"] == [
        list(values) for values in MODEL_SEEDS_BY_QUARTILE
    ]
    assert len(CANDIDATE_GRID_SHA256) == 64
    with pytest.raises(QuartileSpecialistContractError, match="seed"):
        CandidateIdentity(0, 261_335, 100)
    with pytest.raises(QuartileSpecialistContractError, match="checkpoint grid"):
        CandidateIdentity(0, 261_332, 101)


def test_public_coordinate_reconstructs_every_selected_step_phase_and_midpoint() -> None:
    steps: list[int] = []
    phases: list[int] = []
    fractions: list[float] = []
    expected: list[int] = []
    for step in SELECTED_OUTER_STEPS:
        for phase in range(7):
            for fraction in MIDPOINT_FRACTIONS[8]:
                steps.append(step)
                phases.append(phase)
                fractions.append(float(fraction))
                expected.append(step // 128)
    inputs = _inputs(steps, phases, fractions)
    reconstructed = reconstruct_forward_outer_quartile(inputs)
    assert torch.equal(reconstructed, torch.tensor(expected, dtype=torch.long))


def test_composite_dispatch_restores_order_applies_gains_and_masks_boundary() -> None:
    experts = tuple(_RecordingExpert(float(index + 1)) for index in range(4))
    model = QuartileSpecialistBoundaryTangentPredictor(
        experts, gains=(1.0, 1.0, 0.5, 0.25), gains_sealed=True
    )
    # Deliberately scrambled quartiles, with the last q0 row structurally inactive.
    inputs = _inputs([260, 4, 390, 130, 5], zero_last_state=True)
    raw = model.raw_prediction(inputs)
    scaled = model(inputs)
    assert torch.equal(raw[:4, 0], torch.tensor([3.0, 1.0, 4.0, 2.0]))
    assert torch.equal(scaled[:4, 0], torch.tensor([1.5, 1.0, 1.0, 2.0]))
    assert bool(torch.all(raw[-1] == 0.0))
    assert bool(torch.all(scaled[-1] == 0.0))
    assert [expert.batch_sizes for expert in experts] == [[2, 2], [1, 1], [1, 1], [1, 1]]


def test_experts_are_independent_and_unsealed_or_invalid_gains_fail() -> None:
    model = QuartileSpecialistBoundaryTangentPredictor()
    parameter_sets = [set(map(id, expert.parameters())) for expert in model.experts]
    for left in range(4):
        for right in range(left + 1, 4):
            assert parameter_sets[left].isdisjoint(parameter_sets[right])
    with pytest.raises(QuartileSpecialistContractError, match="not sealed"):
        QuartileSpecialistBoundaryTangentPredictor(gains_sealed=False)
    with pytest.raises(QuartileSpecialistContractError, match="q0/q1"):
        QuartileSpecialistBoundaryTangentPredictor(gains=(0.5, 1.0, 0.5, 0.5))
    with pytest.raises(QuartileSpecialistContractError, match="positive"):
        QuartileSpecialistBoundaryTangentPredictor(gains=(1.0, 1.0, 0.0, 0.5))
    shared = ZeroBaselineBoundaryTangentPredictor()
    with pytest.raises(QuartileSpecialistContractError, match="shared"):
        QuartileSpecialistBoundaryTangentPredictor((shared,) * 4)


def test_quartile_scale_is_canonical_and_scaled_loss_has_same_minimizer() -> None:
    targets = np.stack(
        [np.full(EDGES_PER_PHASE, value, dtype=np.float64) for value in (1, 2, 3, 4)]
    )
    quartiles = np.arange(4, dtype=np.int64)
    assert exact_quartile_target_scale(targets, quartiles, 2) == 3.0
    prediction = torch.tensor([[1.0, 4.0]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[2.0, 2.0]], dtype=torch.float64)
    scaled, raw = scaled_raw_target_mse(prediction, target, 2.5)
    assert scaled.item() == raw.item() / 2.5**2
    raw_gradient = torch.autograd.grad(raw, prediction, retain_graph=True)[0]
    scaled_gradient = torch.autograd.grad(scaled, prediction)[0]
    assert torch.equal(scaled_gradient, raw_gradient / 2.5**2)


def test_gain_calibration_is_exact_unclipped_and_has_stable_reasons() -> None:
    candidate = CandidateIdentity(2, 261_338, 100)
    target = np.arange(1.0, 9.0, dtype=np.float64).reshape(2, 4)
    record = calibrate_training_only_gain(candidate, target, 2.0 * target)
    assert record.eligible
    assert record.reason_code == GAIN_ELIGIBLE
    assert math.isclose(record.gain or math.nan, 0.5, rel_tol=0.0, abs_tol=5e-15)

    fixtures = (
        (math.inf, 1.0, GAIN_CROSS_TERM_NONFINITE),
        (1.0, math.nan, GAIN_PREDICTION_ENERGY_NONFINITE),
        (0.0, 1.0, GAIN_CROSS_TERM_NONPOSITIVE),
        (1.0, 0.0, GAIN_PREDICTION_ENERGY_NONPOSITIVE),
        (1.0e308, 1.0e-308, GAIN_NONFINITE),
        (2.0, 1.0, GAIN_OUTSIDE_OPEN_UNIT),
    )
    for cross, energy, reason in fixtures:
        failed = gain_record_from_moments(
            candidate,
            cross_term=cross,
            prediction_energy=energy,
            sample_count=8,
        )
        assert not failed.eligible
        assert failed.reason_code == reason
    assert fixed_unit_gain_record(CandidateIdentity(1, 261_335, 100)).gain == 1.0
    with pytest.raises(QuartileSpecialistContractError, match="only q2/q3"):
        calibrate_training_only_gain(
            CandidateIdentity(1, 261_335, 100), target, target
        )


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"pooled": 0.0}, RANK_POOLED_NONPOSITIVE),
        ({"phases": (0.0,) + (1.0,) * 6}, RANK_PHASE_NONPOSITIVE),
        ({"midpoints": (0.0,) + (1.0,) * 7}, RANK_MIDPOINT_NONPOSITIVE),
        (
            {"cells": ((1.0,) * 8,) * 6 + ((-1.0,) * 8,)},
            RANK_FINE_CELLS_INSUFFICIENT,
        ),
        ({"pooled": math.nan}, RANK_NONFINITE),
    ],
)
def test_training_rank_eligibility_reasons(kwargs: dict[str, object], reason: str) -> None:
    record = _rank_record(CandidateIdentity(0, 261_332, 100), **kwargs)
    assert not record.eligible
    assert record.reason_code == reason


def test_training_rank_gain_and_q1_sentinel_reasons() -> None:
    gain_failed = _rank_record(
        CandidateIdentity(2, 261_338, 100), gain_eligible=False
    )
    assert gain_failed.reason_code == RANK_GAIN_INELIGIBLE
    cells = [[1.0] * 8 for _ in range(7)]
    cells[4][7] = -1.0
    sentinel = _rank_record(
        CandidateIdentity(1, 261_335, 100),
        cells=tuple(tuple(row) for row in cells),
    )
    assert sentinel.positive_fine_cells == 55
    assert sentinel.reason_code == RANK_Q1_SENTINEL_NONPOSITIVE


def test_training_rank_is_separable_with_deterministic_ties() -> None:
    records = []
    expected = []
    for quartile, seeds in enumerate(MODEL_SEEDS_BY_QUARTILE):
        # Exact pooled tie: smaller update wins, then smaller seed.
        records.extend(
            [
                _rank_record(CandidateIdentity(quartile, seeds[2], 200)),
                _rank_record(CandidateIdentity(quartile, seeds[1], 100)),
                _rank_record(CandidateIdentity(quartile, seeds[0], 100)),
            ]
        )
        expected.append(CandidateIdentity(quartile, seeds[0], 100))
    selected = select_training_rank_system(records)
    assert [record.candidate for record in selected] == expected
    assert all(record.reason_code == RANK_ELIGIBLE for record in selected)
    with pytest.raises(NoEligibleQuartileCandidateError, match="q0"):
        select_training_rank_candidate([], 0)


def _selected_system() -> SelectedSystem:
    experts = tuple(
        SelectedExpert(
            candidate=CandidateIdentity(quartile, seeds[0], 100),
            checkpoint_path=f"checkpoints/q{quartile}.pt",
            checkpoint_sha256=chr(ord("a") + quartile) * 64,
            model_state_sha256=("e", "f", "a", "b")[quartile] * 64,
            target_scale=0.1 + quartile,
            gain=(1.0, 1.0, 0.5, 0.25)[quartile],
        )
        for quartile, seeds in enumerate(MODEL_SEEDS_BY_QUARTILE)
    )
    return SelectedSystem(
        experts=experts,
        candidate_grid_sha256=CANDIDATE_GRID_SHA256,
        gain_table_sha256="1" * 64,
        rank_table_sha256="2" * 64,
        role_open_bindings=(
            HashBinding("gain_label_open", "3" * 64),
            HashBinding("rank_label_open", "4" * 64),
        ),
    )


def test_selected_system_round_trip_and_fingerprint_tamper_rejection() -> None:
    system = _selected_system()
    record = system.to_record()
    assert SelectedSystem.from_record(record) == system
    assert SelectedSystem.from_record(record).semantic_sha256 == system.semantic_sha256
    tampered = copy.deepcopy(record)
    tampered["experts"][2]["gain"] = 0.4
    with pytest.raises(QuartileSpecialistContractError, match="fingerprint"):
        SelectedSystem.from_record(tampered)
    malformed = copy.deepcopy(record)
    malformed["experts"][0]["gain"] = 0.9
    with pytest.raises(QuartileSpecialistContractError, match="q0/q1"):
        SelectedSystem.from_record(malformed)
