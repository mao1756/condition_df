from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from mnist.d0_score_boundary_controls import (
    bounded_teacher_anchor_indices,
    bounded_teacher_log_relative_potential,
    bounded_teacher_weights,
)
from mnist.d0_score_density_ratio_paired import (
    PAIRED_MIXTURE_ACCUMULATION_LEVELS,
    PAIRED_MIXTURE_CLUSTER_BIN_COUNTS,
    PAIRED_MIXTURE_MICROBATCH_CLUSTERS,
    PAIRED_MIXTURE_PRODUCTION_ROOT_SEED,
    PairedMixtureStreamPlan,
    accumulation_diagnostics,
    backward_accumulated_objective,
    build_paired_mixture_stream_plan,
    derive_paired_mixture_seed,
    generate_accumulated_paired_stream,
    generate_paired_mixture_microbatch,
    paired_mixture_replay_record,
    paired_mixture_stream_plan_record,
    verify_paired_mixture_replay,
    weighted_paired_softplus_components,
    weighted_paired_softplus_loss,
)


def _plan(root_seed: int = PAIRED_MIXTURE_PRODUCTION_ROOT_SEED):
    return build_paired_mixture_stream_plan(
        root_seed=root_seed,
        grid_size=4,
        horizon=5e-4,
    )


def test_production_plan_freezes_root_bins_microbatch_and_levels() -> None:
    plan = _plan()
    record = paired_mixture_stream_plan_record(plan)
    assert plan.root_seed == 260851
    assert plan.cluster_bin_counts == (4, 4, 4, 4, 16)
    assert plan.microbatch_clusters == 32
    assert plan.accumulation_levels == (2, 4, 8)
    assert record["production_defaults_match"] == 1
    assert record["effective_cluster_counts"] == [64, 128, 256]
    assert record["fingerprint"] == plan.fingerprint
    with pytest.raises(ValueError):
        PairedMixtureStreamPlan(
            root_seed=260851,
            grid_size=4,
            horizon=5e-4,
            microbatch_clusters=64,
        )
    with pytest.raises(ValueError):
        PairedMixtureStreamPlan(
            root_seed=260851,
            grid_size=4,
            horizon=5e-4,
            accumulation_levels=(1, 2, 4),
        )


def test_teacher_microbatch_obeys_exact_common_gamma_identity_and_replays() -> None:
    plan = _plan()
    batch = generate_paired_mixture_microbatch(
        plan,
        phase="unit",
        task="bounded_teacher",
        optimizer_step=7,
        microbatch_index=1,
        dtype=torch.float64,
    )
    assert batch.reference_states.shape == (32, 16)
    assert batch.component_states.shape == (32, 16)
    assert tuple(np.bincount(batch.strata, minlength=5)) == (4, 4, 4, 4, 16)
    assert set(batch.component_indices.tolist()) <= {0, 1, 2, 3}
    assert torch.all(batch.swap_bits == -1)
    assert set(batch.seeds) == {
        "cluster-permutation",
        "common-base-gamma",
        "tilt-increment",
        "mixture-choice",
    }

    gamma = batch.reference_states * batch.base_gamma_sums[:, None]
    anchors = bounded_teacher_anchor_indices(4)[batch.component_indices]
    gamma[torch.arange(32), anchors] += batch.tilt_increments
    reconstructed = gamma / (
        batch.base_gamma_sums + batch.tilt_increments
    )[:, None]
    assert torch.allclose(
        reconstructed, batch.component_states, rtol=2e-15, atol=2e-16
    )

    repeated = generate_paired_mixture_microbatch(
        plan,
        phase="unit",
        task="bounded_teacher",
        optimizer_step=7,
        microbatch_index=1,
        dtype=torch.float64,
    )
    assert repeated.fingerprint == batch.fingerprint
    assert torch.equal(repeated.reference_states, batch.reference_states)
    assert torch.equal(repeated.component_states, batch.component_states)


def test_common_gamma_teacher_has_exact_conditional_dirichlet_marginals() -> None:
    plan = _plan(root_seed=260852)
    references: list[torch.Tensor] = []
    tilted_selected: list[torch.Tensor] = []
    tilted_other: list[torch.Tensor] = []
    anchors = bounded_teacher_anchor_indices(4)
    for step in range(128):
        batch = generate_paired_mixture_microbatch(
            plan,
            phase="marginal",
            task="bounded_teacher",
            optimizer_step=step,
            microbatch_index=0,
            dtype=torch.float64,
        )
        references.append(batch.reference_states)
        selected = anchors[batch.component_indices]
        rows = torch.arange(batch.clusters)
        tilted_selected.append(batch.component_states[rows, selected])
        mask = torch.ones_like(batch.component_states, dtype=torch.bool)
        mask[rows, selected] = False
        tilted_other.append(batch.component_states[mask])

    reference = torch.cat(references)
    selected_values = torch.cat(tilted_selected)
    other_values = torch.cat(tilted_other)
    # S0 ~ Dir(1); SJ | J=j ~ Dir(1+e_j), exactly by Gamma additivity.
    assert abs(float(reference.mean()) - 1.0 / 16.0) < 8e-4
    assert abs(float(selected_values.mean()) - 2.0 / 17.0) < 4e-3
    assert abs(float(other_values.mean()) - 1.0 / 17.0) < 8e-4


def test_weighted_softplus_is_exact_and_zero_logit_risk_is_log_two() -> None:
    reference = torch.tensor([-1.2, 0.3, 2.0], dtype=torch.float64)
    component = torch.tensor([0.7, -0.8, 1.1], dtype=torch.float64)
    epsilon = 0.5
    expected = (
        0.5 * (1.0 - epsilon) * F.softplus(-reference)
        + 0.5 * epsilon * F.softplus(-component)
        + 0.5 * F.softplus(reference)
    )
    pieces = weighted_paired_softplus_components(
        reference,
        component,
        task="bounded_teacher",
        teacher_epsilon=epsilon,
    )
    assert torch.equal(pieces["total"], expected)
    assert torch.equal(
        weighted_paired_softplus_loss(
            reference,
            component,
            task="bounded_teacher",
            teacher_epsilon=epsilon,
            reduction="none",
        ),
        expected,
    )
    zeros = torch.zeros(32, dtype=torch.float64)
    assert float(
        weighted_paired_softplus_loss(
            zeros, zeros, task="bounded_teacher"
        )
    ) == pytest.approx(math.log(2.0), abs=1e-15)
    assert float(
        weighted_paired_softplus_loss(
            zeros, zeros, task="dirichlet_null"
        )
    ) == pytest.approx(math.log(2.0), abs=1e-15)


def test_sampled_j_weighted_objective_is_unbiased_for_full_mixture() -> None:
    plan = _plan(root_seed=260853)
    sampled_values: list[torch.Tensor] = []
    enumerated_values: list[torch.Tensor] = []
    anchors = bounded_teacher_anchor_indices(4)
    epsilon = float(plan.teacher_epsilon)
    for step in range(192):
        batch = generate_paired_mixture_microbatch(
            plan,
            phase="unbiased",
            task="bounded_teacher",
            optimizer_step=step,
            microbatch_index=0,
            dtype=torch.float64,
        )
        base_logits = bounded_teacher_log_relative_potential(
            batch.reference_states, batch.tau_fraction, epsilon=epsilon
        )
        sampled_logits = bounded_teacher_log_relative_potential(
            batch.component_states, batch.tau_fraction, epsilon=epsilon
        )
        sampled_values.append(
            weighted_paired_softplus_loss(
                base_logits,
                sampled_logits,
                task="bounded_teacher",
                teacher_epsilon=epsilon,
                reduction="none",
            )
        )

        gamma = batch.reference_states * batch.base_gamma_sums[:, None]
        component_losses = []
        for anchor in anchors:
            tilted = gamma.clone()
            tilted[:, anchor] += batch.tilt_increments
            tilted = tilted / (
                batch.base_gamma_sums + batch.tilt_increments
            )[:, None]
            logits = bounded_teacher_log_relative_potential(
                tilted, batch.tau_fraction, epsilon=epsilon
            )
            component_losses.append(F.softplus(-logits))
        component_losses_tensor = torch.stack(component_losses, dim=1)
        weights = bounded_teacher_weights(batch.tau_fraction)
        exact_component = (weights * component_losses_tensor).sum(dim=1)
        enumerated_values.append(
            0.5 * (1.0 - epsilon) * F.softplus(-base_logits)
            + 0.5 * epsilon * exact_component
            + 0.5 * F.softplus(base_logits)
        )
    sampled = torch.cat(sampled_values)
    enumerated = torch.cat(enumerated_values)
    # J is the only Monte Carlo difference between these two estimators.
    assert abs(float(sampled.mean() - enumerated.mean())) < 2.5e-4


def test_null_uses_one_pool_and_stateless_swaps_at_matched_times() -> None:
    plan = _plan(root_seed=260854)
    batch = generate_paired_mixture_microbatch(
        plan,
        phase="null",
        task="dirichlet_null",
        optimizer_step=11,
        microbatch_index=0,
        dtype=torch.float64,
    )
    assert set(batch.seeds) == {
        "cluster-permutation",
        "null-pool",
        "null-swaps",
    }
    assert torch.all(batch.component_indices == -1)
    assert set(batch.swap_bits.tolist()) == {0, 1}
    assert torch.all(batch.base_gamma_sums == 0.0)
    assert torch.all(batch.tilt_increments == 0.0)
    assert not torch.equal(batch.reference_states, batch.component_states)
    assert torch.equal(batch.tau, batch.tau_fraction * plan.horizon)

    differences: list[torch.Tensor] = []
    for step in range(128):
        value = generate_paired_mixture_microbatch(
            plan,
            phase="null-marginal",
            task="dirichlet_null",
            optimizer_step=step,
            microbatch_index=0,
            dtype=torch.float64,
        )
        differences.append(value.component_states - value.reference_states)
    assert float(torch.cat(differences).mean().abs()) < 5e-4


@pytest.mark.parametrize("level", PAIRED_MIXTURE_ACCUMULATION_LEVELS)
def test_accumulated_stream_is_replayable_order_invariant_and_stratified(
    level: int,
) -> None:
    plan = _plan(root_seed=260855)
    canonical = generate_accumulated_paired_stream(
        plan,
        phase="accumulate",
        task="bounded_teacher",
        optimizer_step=23,
        accumulation_level=level,
        dtype=torch.float64,
    )
    reversed_stream = generate_accumulated_paired_stream(
        plan,
        phase="accumulate",
        task="bounded_teacher",
        optimizer_step=23,
        accumulation_level=level,
        microbatch_order=tuple(reversed(range(level))),
        dtype=torch.float64,
    )
    assert canonical.fingerprint == reversed_stream.fingerprint
    assert [value.fingerprint for value in canonical.canonical_microbatches] == [
        value.fingerprint for value in reversed_stream.canonical_microbatches
    ]
    assert canonical.effective_clusters == 32 * level
    diagnostics = accumulation_diagnostics(reversed_stream)
    assert diagnostics["effective_clusters"] == 32 * level
    assert diagnostics["time_bin_counts"] == [
        count * level for count in PAIRED_MIXTURE_CLUSTER_BIN_COUNTS
    ]
    assert diagnostics["common_gamma_reconstruction_max_error"] < 2e-12
    assert diagnostics["simplex_max_error"] < 2e-12
    assert diagnostics["order_invariant_fingerprint"] == 1

    replay = paired_mixture_replay_record(
        plan,
        phase="accumulate",
        task="bounded_teacher",
        optimizer_step=23,
        accumulation_level=level,
    )
    assert verify_paired_mixture_replay(plan, replay)["passed"] == 1
    tampered = copy.deepcopy(replay)
    tampered["stream_fingerprint"] = "0" * 64
    assert verify_paired_mixture_replay(plan, tampered)["passed"] == 0


def test_seed_namespaces_and_global_rng_are_isolated() -> None:
    plan = _plan(root_seed=260856)
    seeds = {
        derive_paired_mixture_seed(plan, phase, task, step, micro, namespace)
        for phase in ("pilot", "confirm")
        for task in ("bounded_teacher", "dirichlet_null")
        for step in (1, 2)
        for micro in (0, 1)
        for namespace in ("states", "choices")
    }
    assert len(seeds) == 2 * 2 * 2 * 2 * 2

    torch.manual_seed(99)
    expected = torch.rand(4)
    torch.manual_seed(99)
    generate_accumulated_paired_stream(
        plan,
        phase="rng",
        task="dirichlet_null",
        optimizer_step=1,
        accumulation_level=2,
    )
    actual = torch.rand(4)
    assert torch.equal(actual, expected)


def _model_losses(
    model: nn.Module,
    microbatches: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[torch.Tensor, int]]:
    result: list[tuple[torch.Tensor, int]] = []
    for reference, component in microbatches:
        reference_logits = model(reference).reshape(-1)
        component_logits = model(component).reshape(-1)
        result.append(
            (
                weighted_paired_softplus_loss(
                    reference_logits,
                    component_logits,
                    task="bounded_teacher",
                ),
                int(reference.shape[0]),
            )
        )
    return result


def test_gradient_accumulation_matches_concatenated_objective_and_order() -> None:
    generator = torch.Generator().manual_seed(260857)
    pairs = [
        (
            torch.randn((32, 5), dtype=torch.float64, generator=generator),
            torch.randn((32, 5), dtype=torch.float64, generator=generator),
        )
        for _ in range(4)
    ]
    base = nn.Linear(5, 1, bias=True, dtype=torch.float64)
    with torch.no_grad():
        base.weight.copy_(
            torch.randn(base.weight.shape, dtype=torch.float64, generator=generator)
        )
        base.bias.copy_(
            torch.randn(base.bias.shape, dtype=torch.float64, generator=generator)
        )
    accumulated = copy.deepcopy(base)
    reversed_model = copy.deepcopy(base)
    concatenated = copy.deepcopy(base)
    scale = 0.03

    diagnostics = backward_accumulated_objective(
        _model_losses(accumulated, pairs),
        accumulated.parameters(),
        expected_microbatches=4,
        expected_clusters=128,
        loss_scale=scale,
    )
    reversed_diagnostics = backward_accumulated_objective(
        _model_losses(reversed_model, list(reversed(pairs))),
        reversed_model.parameters(),
        expected_microbatches=4,
        expected_clusters=128,
        loss_scale=scale,
    )
    reference = torch.cat([value[0] for value in pairs])
    component = torch.cat([value[1] for value in pairs])
    concat_loss = weighted_paired_softplus_loss(
        concatenated(reference).reshape(-1),
        concatenated(component).reshape(-1),
        task="bounded_teacher",
    )
    (concat_loss * scale).backward()

    assert diagnostics.cluster_count == 128
    assert diagnostics.microbatch_count == 4
    assert diagnostics.weight_sum == pytest.approx(1.0)
    assert diagnostics.unscaled_objective == pytest.approx(
        float(concat_loss.detach()), rel=1e-14, abs=1e-14
    )
    for accumulated_parameter, concat_parameter, reversed_parameter in zip(
        accumulated.parameters(),
        concatenated.parameters(),
        reversed_model.parameters(),
    ):
        assert torch.allclose(
            accumulated_parameter.grad,
            concat_parameter.grad,
            rtol=1e-13,
            atol=1e-14,
        )
        assert torch.allclose(
            accumulated_parameter.grad,
            reversed_parameter.grad,
            rtol=1e-13,
            atol=1e-14,
        )
    assert diagnostics.scaled_gradient_norm == pytest.approx(
        reversed_diagnostics.scaled_gradient_norm, rel=1e-14, abs=1e-14
    )


def test_gradient_accumulation_fails_closed_on_size_or_dirty_gradients() -> None:
    model = nn.Linear(2, 1, dtype=torch.float64)
    value = model(torch.ones((2, 2), dtype=torch.float64)).square().mean()
    with pytest.raises(ValueError):
        backward_accumulated_objective(
            [(value, 2)],
            model.parameters(),
            expected_microbatches=2,
            expected_clusters=4,
        )

    model.zero_grad(set_to_none=True)
    model(torch.ones((1, 2), dtype=torch.float64)).sum().backward()
    fresh_value = model(torch.ones((2, 2), dtype=torch.float64)).square().mean()
    with pytest.raises(ValueError, match="clear"):
        backward_accumulated_objective(
            [(fresh_value, 2)],
            model.parameters(),
            expected_microbatches=1,
            expected_clusters=2,
        )
