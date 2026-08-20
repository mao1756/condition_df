from __future__ import annotations

import dataclasses
import ast
import csv
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mnist import diag_d0_ddpm_eulerian_diversity_pilot as runner
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, flux_divergence_torch, natural_horizon


def test_frozen_scientific_constants_and_routes_match_the_experiment_note() -> None:
    assert runner.VERSION == "ddpm-eulerian-diversity-pilot-v1"
    assert runner.RESEARCH_MODE == "exploratory"
    assert (runner.PATH_COUNT, runner.PATHS_PER_CLASS) == (40, 4)
    assert runner.PATH_PREFIX == "d2e-v1-"
    assert runner.OUTER_STEPS == 256
    assert runner.ANCHORS == (0, 64, 128, 192, 256)
    assert runner.NATIVE_ANCHORS == (0, 250, 500, 750, 1000)
    assert runner.DECISION_HORIZONS == (64, 128, 256)
    assert runner.ROWS == ("null", "teacher", "historical", "ddpm_eulerian", "native_ddpm")
    assert runner.EULERIAN_ROWS == runner.ROWS[:-1]

    assert runner.INVENTORY_SEED == 0xE1600001
    assert runner.SOURCE_SEED_BASE == 0xE1601000
    assert runner.DDPM_LATENT_SEED_BASE == 0xE1602000
    assert runner.EULERIAN_EDGE_NOISE_ROOT == 0xE1603001
    assert runner.NATIVE_DDPM_REVERSE_SEED_BASE == 0xE1604000
    assert runner.REVIEW_SEED == 0xE1605001
    assert runner.SMOKE_SEED == 0xE160F001

    assert runner.MASS_SCALE_NUMERATOR == 25_471
    assert runner.MASS_SCALE_DENOMINATOR == 255
    assert runner.MASS_SCALE == np.float64(25_471) / np.float64(255)
    assert float(runner.MASS_SCALE).hex() == "0x1.8f8b8b8b8b8b9p+6"
    assert runner.MASS_SCALE_HEX == "0x1.8f8b8b8b8b8b9p+6"
    assert (runner.TRAIN_START, runner.TRAIN_STOP) == (0, 55_000)
    assert (runner.VALIDATION_START, runner.VALIDATION_STOP) == (55_000, 60_000)

    assert runner.LEGACY_CHECKPOINT_SHA256 == "8be77d1701887522f86099673431a928ad7dd2d350a06f7a94ade5c30a658cc3"
    assert runner.DDPM_CHECKPOINT_SHA256 == "5f4065da8753ad5611ec4efd61b6d13082ce3c9cccaa62258f8019118e95dfc8"
    assert runner.EVALUATOR_SHA256 == "3d31d42a14fee0ecc72adc1644718a037cc48e649948407da6c272b981840c92"
    assert runner.EVALUATOR_SELECTION_SHA256 == "e6cd9e49ca61237d3a10e9ad2fe0ad09f7a33ea22911fdd73fd99f3a07e4c668"
    assert runner.MNIST_ARFF_BYTES == 127_888_265
    assert runner.MNIST_ARFF_SHA256 == "418c0a60d2b4abc95db2e2bbf676f3af93ddaf18f79ba3f640624ab57007fb4b"

    assert runner.TEACHER_IMPROVED_MINIMUM == 36
    assert runner.TEACHER_SHORT_RELATIVE_L2_MAXIMUM == 0.80
    assert runner.TEACHER_FINAL_RELATIVE_L2_MAXIMUM == 0.20
    assert runner.TEACHER_CLASSIFIER_ACCURACY_MINIMUM == 0.80
    assert runner.CANDIDATE_HUMAN_RECOGNIZABILITY_MINIMUM == 0.90
    assert runner.CANDIDATE_HUMAN_AGREEMENT_MINIMUM == 0.80
    assert runner.CANDIDATE_DIVERSITY_REAL_RATIO_MINIMUM == 0.25
    assert runner.CANDIDATE_DIVERSITY_HISTORICAL_RATIO_MINIMUM == 2.0
    assert runner.NATIVE_CLASSIFIER_ACCURACY_MINIMUM == 0.80
    assert runner.NATIVE_DIVERSITY_RATIO_MINIMUM == 0.25
    assert runner.POISSON_RESIDUAL_MAXIMUM == 2e-4
    assert runner.MASS_ERROR_MAXIMUM == 2e-6

    assert runner.SCIENTIFIC_ROUTES == (
        "adapter_positive_freeze_replication",
        "native_ddpm_control_invalid",
        "adapter_fidelity_only_major_pivot_or_stop",
        "adapter_early_joint_horizon_replication",
        "adapter_diverse_not_faithful_major_pivot_or_stop",
        "composition_mode_loss_theory_bridge_or_stop",
        "off_policy_bridge_on_policy_or_stop",
        "historical_early_horizon_replication",
        "learned_eulerian_negative_stop_or_major_pivot",
        "unclassified_stop_redesign",
    )


def test_stage_order_freezes_population_before_scoring_review_and_outcome() -> None:
    assert runner.STAGE_ORDER == (
        "binding_preflight",
        "cpu_smoke_replay",
        "inventory_and_start_seal",
        "null_population",
        "teacher_population",
        "historical_population",
        "ddpm_eulerian_population",
        "native_ddpm_population",
        "population_seal",
        "machine_scoring",
        "render_and_review_bundle",
        "awaiting_human_review",
        "human_review_terminalization",
    )
    seal = runner.STAGE_ORDER.index("population_seal")
    assert all(runner.STAGE_ORDER.index(stage) < seal for stage in (
        "null_population",
        "teacher_population",
        "historical_population",
        "ddpm_eulerian_population",
        "native_ddpm_population",
    ))
    assert all(runner.STAGE_ORDER.index(stage) > seal for stage in (
        "machine_scoring",
        "render_and_review_bundle",
        "awaiting_human_review",
        "human_review_terminalization",
    ))


def test_path_inventory_is_exact_balanced_factor_one_and_path_local() -> None:
    inventory = runner.build_path_inventory()
    assert set(inventory) == {
        "path_ids",
        "requested_labels",
        "within_class_index",
        "source_seeds",
        "ddpm_latent_seeds",
        "native_reverse_seeds",
        "retained",
    }
    expected_ids = np.asarray([f"d2e-v1-{index:03d}" for index in range(40)])
    np.testing.assert_array_equal(inventory["path_ids"], expected_ids)
    np.testing.assert_array_equal(inventory["requested_labels"], np.repeat(np.arange(10), 4))
    np.testing.assert_array_equal(inventory["within_class_index"], np.tile(np.arange(4), 10))
    np.testing.assert_array_equal(inventory["source_seeds"], np.arange(0xE1601000, 0xE1601000 + 40, dtype=np.uint64))
    np.testing.assert_array_equal(inventory["ddpm_latent_seeds"], np.arange(0xE1602000, 0xE1602000 + 40, dtype=np.uint64))
    np.testing.assert_array_equal(inventory["native_reverse_seeds"], np.arange(0xE1604000, 0xE1604000 + 40, dtype=np.uint64))
    np.testing.assert_array_equal(inventory["retained"], np.ones(40, dtype=np.int64))
    assert len(set(inventory["path_ids"].tolist())) == 40
    assert all(np.count_nonzero(inventory["requested_labels"] == label) == 4 for label in range(10))
    assert not any("candidate" in key or "selector" in key or "score" in key for key in inventory)


def test_start_bank_uses_one_isolated_cpu_draw_per_declared_source_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = runner.build_path_inventory()
    observed: list[tuple[int, int, str, torch.dtype]] = []

    @dataclasses.dataclass
    class Source:
        masses: torch.Tensor

    def fake_sample(batch_size: int, config: DirectFluxMNISTConfig, *, device: torch.device,
                    dtype: torch.dtype) -> Source:
        seed = int(torch.initial_seed())
        observed.append((seed, batch_size, device.type, dtype))
        row = torch.full((1, 784), 1.0 / 784.0, dtype=dtype)
        # A tiny seed-dependent zero-sum perturbation makes every row distinct.
        amount = (seed - runner.SOURCE_SEED_BASE + 1) * 1e-8
        row[0, 0] += amount
        row[0, 1] -= amount
        return Source(row)

    monkeypatch.setattr(runner, "_sample_source_batch_torch", fake_sample)
    torch.manual_seed(8675309)
    before = torch.random.get_rng_state().clone()
    starts = runner.build_start_bank(DirectFluxMNISTConfig(), inventory)
    after = torch.random.get_rng_state().clone()
    assert torch.equal(before, after)
    assert starts.shape == (40, 784) and starts.dtype == np.float32
    assert [item[0] for item in observed] == list(range(runner.SOURCE_SEED_BASE, runner.SOURCE_SEED_BASE + 40))
    assert all(item[1:] == (1, "cpu", torch.float32) for item in observed)
    assert len({_digest.tobytes() for _digest in starts}) == 40


def test_teacher_bank_selects_first_four_validation_occurrences_and_keeps_image_authorities_distinct() -> None:
    labels = np.tile(np.arange(10, dtype=np.int64), 500)
    images = np.zeros((5_000, 28, 28), dtype=np.uint8)
    for index in range(5_000):
        images[index].flat[index % 784] = np.uint8(1 + index % 254)
    bank = runner.build_teacher_target_bank(images, labels)
    expected_local = np.concatenate([np.flatnonzero(labels == digit)[:4] for digit in range(10)])
    np.testing.assert_array_equal(bank["validation_local_ids"], expected_local)
    np.testing.assert_array_equal(bank["arff_global_row_ids"], expected_local + 55_000)
    np.testing.assert_array_equal(bank["requested_labels"], np.repeat(np.arange(10), 4))
    np.testing.assert_array_equal(bank["path_ids"], runner.build_path_inventory()["path_ids"])
    np.testing.assert_array_equal(bank["source_images_uint8"], images[expected_local])
    np.testing.assert_array_equal(bank["rendered_images_uint8"], runner.mass_to_uint8(bank["masses"]))
    assert not np.array_equal(bank["source_images_uint8"], bank["rendered_images_uint8"])
    np.testing.assert_allclose(bank["masses"].sum(axis=1), 1.0, atol=runner.MASS_ERROR_MAXIMUM)
    assert np.all(bank["masses"] > 0)
    assert bank["role"].item() == "teacher_only_validation_targets"


def test_latent_bank_is_path_local_chunk_independent_and_rng_neutral() -> None:
    inventory = runner.build_path_inventory()
    torch.manual_seed(112358)
    before = torch.random.get_rng_state().clone()
    bank = runner.build_ddpm_latent_bank(inventory)
    after = torch.random.get_rng_state().clone()
    assert torch.equal(before, after)
    assert bank["z"].shape == (40, 1, 28, 28) and bank["z"].dtype == np.float32
    for index in (0, 17, 39):
        expected = torch.randn(
            (1, 28, 28),
            generator=torch.Generator(device="cpu").manual_seed(runner.DDPM_LATENT_SEED_BASE + index),
        ).numpy()
        np.testing.assert_array_equal(bank["z"][index], expected)
    assert bank["z_sha256"].item() == runner._hash_array(bank["z"])
    np.testing.assert_array_equal(bank["path_ids"], inventory["path_ids"])
    np.testing.assert_array_equal(bank["latent_seeds"], inventory["ddpm_latent_seeds"])


@pytest.mark.parametrize("tamper", ["duplicate_id", "wrong_label", "duplicate_latent_seed"])
def test_all_banks_reject_inventory_identity_or_seed_tamper(tamper: str) -> None:
    inventory = {key: value.copy() for key, value in runner.build_path_inventory().items()}
    if tamper == "duplicate_id":
        inventory["path_ids"][1] = inventory["path_ids"][0]
    elif tamper == "wrong_label":
        inventory["requested_labels"][0] = 9
    else:
        inventory["ddpm_latent_seeds"][1] = inventory["ddpm_latent_seeds"][0]
    with pytest.raises(runner.IntegrityFailure):
        runner.build_ddpm_latent_bank(inventory)


def test_edge_noise_seed_is_exact_big_endian_and_key_sensitive() -> None:
    path_id = "d2e-v1-017"
    parts = (path_id, 91, 2, 1)
    payload = f"edge-noise-v1|{runner.EULERIAN_EDGE_NOISE_ROOT}|{path_id}|91|2|1"
    expected = int.from_bytes(hashlib.sha256(payload.encode("ascii")).digest()[:8], "big", signed=False)
    assert runner.derive_edge_noise_seed(*parts) == expected
    assert 0 <= expected < 2**64
    variants = {
        runner.derive_edge_noise_seed(path_id, 91, 2, 1),
        runner.derive_edge_noise_seed(path_id, 92, 2, 1),
        runner.derive_edge_noise_seed(path_id, 91, 4, 1),
        runner.derive_edge_noise_seed(path_id, 91, 2, 0),
        runner.derive_edge_noise_seed("d2e-v1-018", 91, 2, 1),
    }
    assert len(variants) == 5


def test_path_local_standard_normals_are_retry_stable_chunk_invariant_and_rng_neutral() -> None:
    paths = ["d2e-v1-000", "d2e-v1-019", "d2e-v1-039"]
    torch.manual_seed(123456)
    before = torch.random.get_rng_state().clone()
    full = runner.standard_normal_flat_for_paths(paths, 7, 4, 2, edge_count=19)
    after = torch.random.get_rng_state().clone()
    assert torch.equal(before, after)
    assert full.shape == (3, 19)

    chunks = torch.cat([
        runner.standard_normal_flat_for_paths(paths[:1], 7, 4, 2, edge_count=19),
        runner.standard_normal_flat_for_paths(paths[1:], 7, 4, 2, edge_count=19),
    ])
    assert torch.equal(full, chunks)
    permutation = [2, 0, 1]
    permuted = runner.standard_normal_flat_for_paths([paths[index] for index in permutation], 7, 4, 2, edge_count=19)
    assert torch.equal(permuted, full[permutation])
    assert torch.equal(full, runner.standard_normal_flat_for_paths(paths, 7, 4, 2, edge_count=19))
    assert not torch.equal(full, runner.standard_normal_flat_for_paths(paths, 7, 2, 2, edge_count=19))
    assert not torch.equal(full, runner.standard_normal_flat_for_paths(paths, 7, 4, 1, edge_count=19))


def test_supplied_noise_extension_is_local_to_new_adapter_and_preserves_v3_core_bytes() -> None:
    core_path = Path(runner.__file__).with_name("eulerian_flux_mnist.py")
    assert runner.sha256_file(core_path) == "4dca1c40f25eb04b3d615bd0094891c7cedb8cea8a673607eb02e1ab977e4f19"
    assert "standard_normal_flat" not in inspect.signature(runner.eulerian_flux_step_torch).parameters
    adapter_step = runner._adapter_module().eulerian_flux_step_with_standard_normal_torch
    assert "standard_normal_flat" in inspect.signature(adapter_step).parameters


def test_resource_budget_rejects_every_ceiling_overrun() -> None:
    budget = runner.ResourceBudget()
    assert budget == runner.ResourceBudget(
        max_wall_seconds=3600.0,
        max_accelerator_seconds=1800.0,
        max_storage_bytes=256 * 1024**2,
        max_cuda_fraction=0.50,
        reserve_seconds=runner.TERMINAL_RESERVE_SECONDS,
        maximum_quantum_seconds=runner.MAX_QUANTUM_SECONDS,
    )
    bad = (
        {"max_wall_seconds": 3600.0001},
        {"max_accelerator_seconds": 1800.0001},
        {"max_storage_bytes": 256 * 1024**2 + 1},
        {"max_cuda_fraction": 0.500001},
        {"max_wall_seconds": 30.0, "reserve_seconds": 30.0},
    )
    for overrides in bad:
        with pytest.raises(ValueError):
            runner.ResourceBudget(**overrides)


def test_target_authority_is_structurally_teacher_only() -> None:
    provider_types = (
        runner.NullControllerProvider,
        runner.TeacherControllerProvider,
        runner.HistoricalControllerProvider,
        runner.DDPMEulerianControllerProvider,
    )
    constructors_with_targets = {
        provider.__name__
        for provider in provider_types
        if "targets" in inspect.signature(provider.__init__).parameters
    }
    assert constructors_with_targets == {"TeacherControllerProvider"}
    for provider in provider_types:
        call_parameters = inspect.signature(provider.__call__).parameters
        assert "targets" not in call_parameters
        assert "target" not in call_parameters


def test_null_teacher_and_historical_use_the_common_controller_contract() -> None:
    config = DirectFluxMNISTConfig(grid_size=28, free_weight=0.015, noise_weight=0.002)
    path_ids = ("d2e-v1-000", "d2e-v1-001")
    masses = torch.full((2, 784), 1.0 / 784.0)
    labels = torch.tensor([0, 0], dtype=torch.int64)
    source_masses = masses.clone()
    remaining = torch.full((2,), float(natural_horizon(config)))

    null = runner.NullControllerProvider(config)(masses, labels, remaining, path_ids, source_masses)
    assert isinstance(null, runner.ControllerStep)
    assert null.conditioning_flux.shape == (2, 2, 28, 28)
    assert torch.count_nonzero(null.conditioning_flux).item() == 0

    targets = masses.clone()
    targets[0, 0] += 1e-3
    targets[0, 1] -= 1e-3
    teacher_provider = runner.TeacherControllerProvider(config, targets, path_ids)
    teacher = teacher_provider(masses, labels, remaining, path_ids, source_masses)
    total_flux = teacher.conditioning_flux + config.free_weight * runner.free_drift_flux_torch(masses, config)
    velocity = (targets - masses) / remaining[:, None]
    velocity -= velocity.mean(dim=1, keepdim=True)
    residual = flux_divergence_torch(total_flux).reshape_as(velocity) - velocity
    assert residual.abs().amax().item() <= runner.POISSON_RESIDUAL_MAXIMUM
    assert "poisson_divergence_residual" in teacher.telemetry
    with pytest.raises(runner.IntegrityFailure, match="path order"):
        teacher_provider(masses, labels, remaining, tuple(reversed(path_ids)), source_masses)

    class RecordingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.call: tuple[torch.Tensor, ...] | None = None

        def predict_flux(self, tau: torch.Tensor, state: torch.Tensor, requested: torch.Tensor,
                         *, source_masses: torch.Tensor) -> torch.Tensor:
            self.call = (tau, state, requested, source_masses)
            return torch.zeros((len(state), 2, 28, 28), dtype=state.dtype)

    model = RecordingModel()
    historical = runner.HistoricalControllerProvider(model)  # type: ignore[arg-type]
    result = historical(masses, labels, remaining, path_ids, source_masses)
    assert model.call is not None
    assert model.call[3] is source_masses
    assert result.conditioning_flux.shape == (2, 2, 28, 28)


def test_adapter_provider_uses_only_persistent_latent_and_rejects_path_drift() -> None:
    @dataclasses.dataclass(frozen=True)
    class Result:
        conditioning_flux: torch.Tensor
        predicted_mass: torch.Tensor
        desired_velocity: torch.Tensor
        ddpm_timestep: torch.Tensor
        epsilon_rms: torch.Tensor
        score_rms: torch.Tensor
        render_saturation_fraction: torch.Tensor
        x0_saturation_fraction: torch.Tensor
        divergence_residual_linf: torch.Tensor

    class RecordingAdapter:
        def __init__(self) -> None:
            self.calls: list[tuple[torch.Tensor, ...]] = []

        def predict(self, current_masses: torch.Tensor, labels: torch.Tensor,
                    remaining_time: torch.Tensor, latent_z: torch.Tensor) -> Result:
            self.calls.append((current_masses, labels, remaining_time, latent_z.clone()))
            return Result(
                torch.zeros((len(current_masses), 2, 28, 28), dtype=current_masses.dtype),
                current_masses.clone(),
                torch.zeros_like(current_masses),
                torch.zeros(len(current_masses), dtype=torch.int64),
                torch.zeros(len(current_masses)),
                torch.zeros(len(current_masses)),
                torch.zeros(len(current_masses)),
                torch.zeros(len(current_masses)),
                torch.zeros(len(current_masses)),
            )

    path_ids = ("d2e-v1-000", "d2e-v1-001")
    latent = np.arange(2 * 28 * 28, dtype=np.float32).reshape(2, 1, 28, 28)
    adapter = RecordingAdapter()
    provider = runner.DDPMEulerianControllerProvider(adapter, latent, path_ids)
    assert not any("target" in name.lower() for name in vars(provider))
    masses = torch.full((2, 784), 1.0 / 784.0)
    labels = torch.tensor([0, 0])
    remaining = torch.ones(2)
    first = provider(masses, labels, remaining, path_ids, masses)
    second = provider(masses, labels, remaining, path_ids, masses)
    assert torch.equal(adapter.calls[0][3], adapter.calls[1][3])
    assert set(first.telemetry) == {
        "predicted_mass",
        "desired_velocity",
        "ddpm_timestep",
        "epsilon_rms",
        "score_rms",
        "render_saturation_fraction",
        "x0_saturation_fraction",
        "divergence_residual_linf",
        "poisson_divergence_residual",
        "predicted_mass_entropy",
        "predicted_mass_maximum",
        "predicted_mass_distance_from_state",
        "desired_velocity_rms",
        "controller_flux_rms",
        "predicted_mass_within_class_nn_mse",
    }
    assert torch.equal(first.conditioning_flux, second.conditioning_flux)
    with pytest.raises(runner.IntegrityFailure, match="path order"):
        provider(masses, labels, remaining, tuple(reversed(path_ids)), masses)


def test_row_result_endpoint_is_the_frozen_final_anchor() -> None:
    anchors = np.arange(5 * 2 * 784, dtype=np.float32).reshape(5, 2, 784)
    result = runner.EulerianRowResult(
        row="null",
        anchors=anchors,
        anchor_steps=np.asarray(runner.ANCHORS, dtype=np.int64),
        path_ids=np.asarray(["d2e-v1-000", "d2e-v1-001"]),
        requested_labels=np.asarray([0, 0], dtype=np.int64),
        telemetry=[],
        crn_key_hashes=[],
        scientific_digest="a" * 64,
    )
    assert np.shares_memory(result.endpoints, anchors)
    np.testing.assert_array_equal(result.endpoints, anchors[-1])


def _two_path_uniform_starts() -> tuple[np.ndarray, np.ndarray, tuple[str, str]]:
    starts = np.full((2, 784), 1.0 / 784.0, dtype=np.float32)
    labels = np.asarray([0, 0], dtype=np.int64)
    return starts, labels, ("d2e-v1-000", "d2e-v1-001")


def test_common_row_integrator_replays_science_and_crn_but_excludes_timing() -> None:
    starts, labels, path_ids = _two_path_uniform_starts()
    config = DirectFluxMNISTConfig(
        free_weight=0.015,
        noise_weight=0.002,
        adaptive_sampling=False,
        max_substeps=4,
    )
    torch.manual_seed(314159)
    global_before = torch.random.get_rng_state().clone()
    first = runner.run_eulerian_row(
        starts,
        labels,
        path_ids,
        config,
        runner.NullControllerProvider(config),
        row="null",
        device="cpu",
        num_steps=4,
        schedule_steps=4,
        anchors=(0, 1, 2, 3, 4),
    )
    global_after_first = torch.random.get_rng_state().clone()
    second = runner.run_eulerian_row(
        starts,
        labels,
        path_ids,
        config,
        runner.NullControllerProvider(config),
        row="null",
        device="cpu",
        num_steps=4,
        schedule_steps=4,
        anchors=(0, 1, 2, 3, 4),
    )
    global_after_second = torch.random.get_rng_state().clone()
    assert torch.equal(global_before, global_after_first)
    assert torch.equal(global_before, global_after_second)
    np.testing.assert_array_equal(first.anchors, second.anchors)
    assert first.crn_key_hashes == second.crn_key_hashes
    assert first.scientific_digest == second.scientific_digest
    assert [item["elapsed_seconds"] for item in first.telemetry] != [item["elapsed_seconds"] for item in second.telemetry]

    class ZeroHistorical(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def predict_flux(self, tau: torch.Tensor, state: torch.Tensor, requested: torch.Tensor,
                         *, source_masses: torch.Tensor) -> torch.Tensor:
            del tau, requested, source_masses
            return torch.zeros((len(state), 2, 28, 28), dtype=state.dtype)

    historical = runner.run_eulerian_row(
        starts,
        labels,
        path_ids,
        config,
        runner.HistoricalControllerProvider(ZeroHistorical()),  # type: ignore[arg-type]
        row="historical",
        device="cpu",
        num_steps=4,
        schedule_steps=4,
        anchors=(0, 1, 2, 3, 4),
    )
    np.testing.assert_array_equal(first.anchors, historical.anchors)
    assert first.crn_key_hashes == historical.crn_key_hashes
    assert first.scientific_digest != historical.scientific_digest


def test_adaptive_retry_resets_state_and_uses_attempt_keyed_common_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    starts, labels, path_ids = _two_path_uniform_starts()
    config = DirectFluxMNISTConfig(
        free_weight=0.0,
        noise_weight=0.002,
        adaptive_sampling=True,
        clip_target=0.03,
        max_substeps=4,
    )
    provider_states: list[torch.Tensor] = []
    supplied_normals: list[torch.Tensor] = []

    class RecordingProvider:
        def __call__(self, masses: torch.Tensor, requested: torch.Tensor, remaining: torch.Tensor,
                     ids: list[str], source_masses: torch.Tensor) -> runner.ControllerStep:
            del requested, remaining, ids, source_masses
            provider_states.append(masses.clone())
            return runner.ControllerStep(torch.zeros((len(masses), 2, 28, 28)), {})

    horizon = float(natural_horizon(config))

    def fake_step(state: torch.Tensor, flux: torch.Tensor, dt: float, cfg: DirectFluxMNISTConfig,
                  **kwargs: object) -> tuple[torch.Tensor, int, int]:
        del flux, cfg
        if "standard_normal_flat" not in kwargs:
            # The runner separately measures the deterministic controller-only
            # clipping fraction. It must not consume a CRN key or mutate state.
            return state.clone(), 0, 100
        normals = kwargs["standard_normal_flat"]
        assert isinstance(normals, torch.Tensor)
        supplied_normals.append(normals.clone())
        changed = state.clone()
        changed[:, 0] += 1e-5
        changed[:, 1] -= 1e-5
        # The one-substep attempt is rejected; the two-substep attempt is accepted.
        return changed, (100 if np.isclose(dt, horizon) else 0), 100

    monkeypatch.setattr(runner, "eulerian_flux_step_torch", fake_step)
    monkeypatch.setattr(
        runner._adapter_module(),
        "eulerian_flux_step_with_standard_normal_torch",
        fake_step,
    )
    result = runner.run_eulerian_row(
        starts,
        labels,
        path_ids,
        config,
        RecordingProvider(),
        row="null",
        device="cpu",
        num_steps=1,
        schedule_steps=1,
        anchors=(0, 1),
    )
    assert result.telemetry[0]["accepted_substeps"] == 2
    assert [entry["substeps"] for entry in result.telemetry[0]["attempts"]] == [1, 2]
    assert len(provider_states) == 3 and len(supplied_normals) == 3
    assert torch.equal(provider_states[0], provider_states[1])
    assert not torch.equal(supplied_normals[0], supplied_normals[1])
    expected = runner.standard_normal_flat_for_paths(path_ids, 0, 2, 1)
    assert torch.equal(supplied_normals[2], expected)
    np.testing.assert_allclose(result.endpoints[:, :2], starts[:, :2] + np.asarray([[2e-5, -2e-5]], dtype=np.float32), atol=1e-9)


@pytest.mark.parametrize("residual", [runner.POISSON_RESIDUAL_MAXIMUM + 1e-8, float("nan")])
def test_provider_poisson_residual_is_an_execution_gate_not_telemetry_only(residual: float) -> None:
    starts, labels, path_ids = _two_path_uniform_starts()
    config = DirectFluxMNISTConfig(
        free_weight=0.0,
        noise_weight=0.0,
        adaptive_sampling=False,
        max_substeps=1,
    )

    class BadResidualProvider:
        def __call__(self, masses: torch.Tensor, requested: torch.Tensor, remaining: torch.Tensor,
                     ids: list[str], source_masses: torch.Tensor) -> runner.ControllerStep:
            del requested, remaining, ids, source_masses
            return runner.ControllerStep(
                torch.zeros((len(masses), 2, 28, 28), dtype=masses.dtype),
                {"poisson_divergence_residual": torch.full((len(masses),), residual)},
            )

    with pytest.raises(runner.IntegrityFailure, match="Poisson|poisson"):
        runner.run_eulerian_row(
            starts,
            labels,
            path_ids,
            config,
            BadResidualProvider(),
            row="teacher",
            device="cpu",
            num_steps=1,
            schedule_steps=1,
            anchors=(0, 1),
        )


def test_float64_teacher_and_adapter_poisson_fluxes_feed_the_float32_common_integrator() -> None:
    starts, labels, path_ids = _two_path_uniform_starts()
    config = DirectFluxMNISTConfig(
        free_weight=0.015,
        noise_weight=0.0,
        adaptive_sampling=False,
        max_substeps=1,
        mass_floor=1e-8,
    )
    targets = starts.copy()
    targets[:, 0] += np.float32(1e-4)
    targets[:, 1] -= np.float32(1e-4)

    adapter_module = runner._adapter_module()

    class ZeroEpsilon(torch.nn.Module):
        def forward(self, state: torch.Tensor, timestep: torch.Tensor, requested: torch.Tensor) -> torch.Tensor:
            del timestep, requested
            return torch.zeros_like(state)

    adapter = adapter_module.DDPMEulerianAdapter(
        ZeroEpsilon(),
        runner.make_linear_ddpm_schedule(num_steps=4),
        config,
        adapter_module.DDPMEulerianAdapterConfig(num_ddpm_steps=4),
    )
    providers = (
        runner.TeacherControllerProvider(config, targets, path_ids),
        runner.DDPMEulerianControllerProvider(adapter, np.zeros((2, 1, 28, 28), dtype=np.float32), path_ids),
    )
    for row, provider in zip(("teacher", "ddpm_eulerian"), providers, strict=True):
        direct = provider(
            torch.as_tensor(starts),
            torch.as_tensor(labels),
            torch.full((2,), float(natural_horizon(config))),
            path_ids,
            torch.as_tensor(starts),
        )
        assert direct.conditioning_flux.dtype == torch.float32
        residual = torch.as_tensor(direct.telemetry["poisson_divergence_residual"])
        assert residual.dtype == torch.float64
        assert residual.amax().item() <= runner.POISSON_RESIDUAL_MAXIMUM
        result = runner.run_eulerian_row(
            starts,
            labels,
            path_ids,
            config,
            provider,
            row=row,
            device="cpu",
            num_steps=1,
            schedule_steps=1,
            anchors=(0, 1),
        )
        assert result.anchors.dtype == np.float32
        assert np.isfinite(result.anchors).all()
        assert result.telemetry[0]["poisson_divergence_residual_maximum"] <= runner.POISSON_RESIDUAL_MAXIMUM


def test_historical_provider_freezes_model_runs_under_no_grad_and_preserves_state_digest() -> None:
    starts, labels, path_ids = _two_path_uniform_starts()
    config = DirectFluxMNISTConfig(
        free_weight=0.0,
        noise_weight=0.0,
        adaptive_sampling=False,
        max_substeps=1,
    )

    class TinyHistorical(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.0))
            self.grad_modes: list[bool] = []

        def predict_flux(self, tau: torch.Tensor, state: torch.Tensor, requested: torch.Tensor,
                         *, source_masses: torch.Tensor) -> torch.Tensor:
            del tau, requested, source_masses
            self.grad_modes.append(torch.is_grad_enabled())
            return self.scale * torch.ones((len(state), 2, 28, 28), dtype=state.dtype)

    model = TinyHistorical().train()
    provider = runner.HistoricalControllerProvider(model)  # type: ignore[arg-type]
    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    before = runner.v3._model_state_semantic_digest(model)
    result = runner.run_eulerian_row(
        starts,
        labels,
        path_ids,
        config,
        provider,
        row="historical",
        device="cpu",
        num_steps=1,
        schedule_steps=1,
        anchors=(0, 1),
    )
    assert model.grad_modes == [False]
    assert runner.v3._model_state_semantic_digest(model) == before
    assert result.anchors.dtype == np.float32


def test_runner_source_contains_no_optimizer_backward_or_training_call() -> None:
    tree = ast.parse(inspect.getsource(runner))
    forbidden_attributes = {"backward", "step", "zero_grad", "train"}
    forbidden_names = {"Optimizer", "Adam", "AdamW", "SGD", "train_image_classifier"}
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(call, ast.Attribute) and call.attr in forbidden_attributes for call in calls)
    assert not any(isinstance(call, ast.Name) and call.id in forbidden_names for call in calls)


def test_ddpm_checkpoint_and_population_identity_use_the_same_digest() -> None:
    adapter_module = runner._adapter_module()
    model = adapter_module.ClassConditionalUNet28()

    assert adapter_module._model_state_sha256(model) == runner.v3._model_state_semantic_digest(model)
    assert "bound.model_state_sha256 == v3._model_state_semantic_digest(bound.model)" in inspect.getsource(
        runner.run_production
    )


def test_native_ddpm_is_latent_linked_path_local_and_scientifically_deterministic() -> None:
    class ZeroEpsilon(torch.nn.Module):
        def forward(self, state: torch.Tensor, timestep: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            del timestep, labels
            return torch.zeros_like(state)

    schedule = runner.make_linear_ddpm_schedule(num_steps=4)
    inventory = runner.build_path_inventory()
    latent = np.linspace(-1.0, 1.0, 40 * 28 * 28, dtype=np.float32).reshape(40, 1, 28, 28)
    labels = inventory["requested_labels"]
    ids = inventory["path_ids"]
    seeds = inventory["native_reverse_seeds"]
    model = ZeroEpsilon().eval()
    torch.manual_seed(271828)
    before = torch.random.get_rng_state().clone()
    first = runner.run_native_ddpm_row(
        model,
        schedule,
        latent,
        labels,
        ids,
        seeds,
        device="cpu",
        anchor_steps=(0, 1, 2, 3, 4),
    )
    second = runner.run_native_ddpm_row(
        model,
        schedule,
        latent,
        labels,
        ids,
        seeds,
        device="cpu",
        anchor_steps=(0, 1, 2, 3, 4),
    )
    assert torch.equal(before, torch.random.get_rng_state())
    np.testing.assert_array_equal(first.model_anchors, second.model_anchors)
    np.testing.assert_array_equal(first.model_anchors[0], latent)
    np.testing.assert_array_equal(first.reverse_steps, np.arange(5, dtype=np.int64))
    assert first.latent_bank_sha256 == runner._hash_array(latent)
    assert first.scientific_digest == second.scientific_digest
    assert len(first.telemetry) == 4


def test_native_render_is_an_exact_model_space_raster_without_mass_normalization() -> None:
    values = np.asarray([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    states = np.resize(values, (1, 1, 28, 28)).astype(np.float32)
    rendered = runner._model_to_uint8(states)
    expected = np.rint(np.clip((states[:, 0] + 1.0) * 127.5, 0.0, 255.0)).astype(np.uint8)
    np.testing.assert_array_equal(rendered, expected)
    assert rendered.shape == (1, 28, 28)
    assert rendered.flat[0] == 0
    assert rendered.flat[1] == 0
    assert rendered.flat[3] == 128
    assert rendered.flat[5] == 255
    assert rendered.flat[6] == 255


@pytest.mark.parametrize("tamper", ["duplicate_path", "duplicate_seed", "wrong_label"])
def test_native_ddpm_rejects_inventory_identity_tamper(tamper: str) -> None:
    class ZeroEpsilon(torch.nn.Module):
        def forward(self, state: torch.Tensor, timestep: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            del timestep, labels
            return torch.zeros_like(state)

    inventory = runner.build_path_inventory()
    latent = np.zeros((40, 1, 28, 28), dtype=np.float32)
    ids = inventory["path_ids"].copy()
    labels = inventory["requested_labels"].copy()
    seeds = inventory["native_reverse_seeds"].copy()
    if tamper == "duplicate_path":
        ids[1] = ids[0]
    elif tamper == "duplicate_seed":
        seeds[1] = seeds[0]
    else:
        labels[0] = 9
    with pytest.raises(runner.IntegrityFailure):
        runner.run_native_ddpm_row(
            ZeroEpsilon(),
            runner.make_linear_ddpm_schedule(num_steps=1),
            latent,
            labels,
            ids,
            seeds,
            device="cpu",
            anchor_steps=(0, 1),
        )


def _synthetic_population_run(tmp_path: Path, *, historical_receipt: bool = False) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "synthetic-run"
    for directory in ("inventory", "populations", "telemetry"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    inventory = runner.build_path_inventory()
    starts = np.full((runner.PATH_COUNT, 784), np.float32(1.0 / 784.0), dtype=np.float32)
    for index in range(runner.PATH_COUNT):
        delta = np.float32((index + 1) * 1e-6)
        starts[index, 0] += delta
        starts[index, 1] -= delta
    runner._validate_masses(starts, runner.PATH_COUNT, "synthetic start bank")
    teacher = starts[:, ::-1].copy()
    latent = runner.build_ddpm_latent_bank(inventory)

    runner._write_csv(
        root / "inventory/path_inventory.csv",
        [
            {
                "path_id": inventory["path_ids"][index],
                "requested_label": inventory["requested_labels"][index],
                "within_class_index": inventory["within_class_index"][index],
                "source_seed": inventory["source_seeds"][index],
                "ddpm_latent_seed": inventory["ddpm_latent_seeds"][index],
                "native_reverse_seed": inventory["native_reverse_seeds"][index],
                "retained": inventory["retained"][index],
            }
            for index in range(runner.PATH_COUNT)
        ],
    )

    start_arrays = {
        "masses": starts,
        "path_ids": inventory["path_ids"],
        "requested_labels": inventory["requested_labels"],
        "source_seeds": inventory["source_seeds"],
    }
    runner._write_npz(
        root / "inventory/start_bank.npz",
        schema=np.asarray(runner.VERSION + "-start-bank"),
        version=np.asarray(runner.VERSION),
        **start_arrays,
        scientific_sha256=np.asarray(runner._scientific_digest(start_arrays, {})),
    )
    teacher_arrays = dict(
        masses=teacher,
        source_images_uint8=np.zeros((runner.PATH_COUNT, 28, 28), dtype=np.uint8),
        rendered_images_uint8=runner.mass_to_uint8(teacher),
        requested_labels=inventory["requested_labels"],
        validation_local_ids=np.arange(runner.PATH_COUNT, dtype=np.int64),
        arff_global_row_ids=np.arange(runner.VALIDATION_START, runner.VALIDATION_START + runner.PATH_COUNT, dtype=np.int64),
        path_ids=inventory["path_ids"],
        role=np.asarray("teacher_only_validation_targets"),
    )
    runner._write_npz(
        root / "inventory/teacher_target_bank.npz",
        schema=np.asarray(runner.VERSION + "-teacher-target-bank"),
        version=np.asarray(runner.VERSION),
        **teacher_arrays,
        scientific_sha256=np.asarray(runner._scientific_digest(teacher_arrays, {})),
    )
    latent_science = {key: value for key, value in latent.items() if key != "z_sha256"}
    runner._write_npz(
        root / "inventory/ddpm_latent_bank.npz",
        schema=np.asarray(runner.VERSION + "-latent-bank"),
        version=np.asarray(runner.VERSION),
        **latent,
        scientific_sha256=np.asarray(runner._scientific_digest(latent_science, {})),
    )
    rng_contract = {
        "schema": runner.VERSION + "-rng-contract",
        "edge_noise_root": runner.EULERIAN_EDGE_NOISE_ROOT,
        "payload": "edge-noise-v1|<root>|<path_id>|<outer_step>|<attempted_substeps>|<sub_index>",
        "sha256_prefix_bytes": 8,
        "byteorder": "big",
    }
    runner._write_json(root / "inventory/rng_contract.json", rng_contract)
    runner._write_json(root / "source_bindings.json", runner._source_bindings(runner._repository_root()))
    authority_dir = root / "synthetic-input-authorities"
    authority_dir.mkdir()
    checkpoint_bindings: dict[str, object] = {"schema": runner.VERSION + "-checkpoint-bindings"}
    for key in ("legacy_checkpoint", "arff", "ddpm_checkpoint", "evaluator_checkpoint", "evaluator_selection"):
        path = authority_dir / f"{key}.bin"
        runner.v3._atomic_bytes(path, f"synthetic-{key}\n".encode("ascii"))
        checkpoint_bindings[key] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": runner.sha256_file(path),
        }
    bound_paths: dict[str, Path] = {"checkpoint": Path(checkpoint_bindings["ddpm_checkpoint"]["path"])}  # type: ignore[index]
    for prefix in ("selection", "config", "schedule"):
        path = authority_dir / f"ddpm_bound_{prefix}.bin"
        runner.v3._atomic_bytes(path, f"synthetic-ddpm-bound-{prefix}\n".encode("ascii"))
        bound_paths[prefix] = path
    checkpoint_bindings["ddpm_bound"] = {
        key: value
        for prefix, path in bound_paths.items()
        for key, value in (
            (f"{prefix}_path", str(path.resolve())),
            (f"{prefix}_bytes", path.stat().st_size),
            (f"{prefix}_sha256", runner.sha256_file(path)),
        )
    }
    checkpoint_bindings["ddpm_bound"]["model_state_sha256"] = "synthetic-ddpm-model-state"  # type: ignore[index]
    if historical_receipt:
        checkpoint_bindings["historical_receipt"] = {"config": dataclasses.asdict(DirectFluxMNISTConfig())}
    runner._write_json(root / "checkpoint_bindings.json", checkpoint_bindings)
    budget = runner.ResourceBudget()
    runner._write_json(
        root / "config.json",
        {
            "schema": runner.VERSION + "-config",
            "version": runner.VERSION,
            "research_mode": runner.RESEARCH_MODE,
            "repository_root": str(runner._repository_root()),
            "scientific_configuration": {
                "path_count": runner.PATH_COUNT,
                "paths_per_class": runner.PATHS_PER_CLASS,
                "outer_steps": runner.OUTER_STEPS,
                "anchors": list(runner.ANCHORS),
                "native_anchors": list(runner.NATIVE_ANCHORS),
                "adapter": {
                    "num_ddpm_steps": 1000,
                    "beta_start": 0.0001,
                    "beta_end": 0.02,
                    "min_tau_fraction": 0.03,
                    "time_map": "linear_remaining_fraction_round",
                    "latent_policy": "persistent_path_latent",
                    "flux_projection": "periodic_minimum_energy_minus_free",
                },
                "mass_scale": {
                    "numerator": runner.MASS_SCALE_NUMERATOR,
                    "denominator": runner.MASS_SCALE_DENOMINATOR,
                    "hex": runner.MASS_SCALE_HEX,
                },
            },
            "execution_authority": {
                "approval_id": "synthetic-test-only",
                "device": "cpu",
                **dataclasses.asdict(budget),
            },
        },
    )

    runner._write_json(
        root / "inventory/STARTS_SEALED.json",
        {
            "schema": runner.VERSION + "-starts-seal",
            "start_bank_sha256": runner.sha256_file(root / "inventory/start_bank.npz"),
            "teacher_target_bank_sha256": runner.sha256_file(root / "inventory/teacher_target_bank.npz"),
            "latent_bank_sha256": runner.sha256_file(root / "inventory/ddpm_latent_bank.npz"),
            "rng_contract_sha256": runner.sha256_file(root / "inventory/rng_contract.json"),
            "source_bindings_sha256": runner.sha256_file(root / "source_bindings.json"),
            "checkpoint_bindings_sha256": runner.sha256_file(root / "checkpoint_bindings.json"),
            "config_sha256": runner.sha256_file(root / "config.json"),
            "data_audit": {"synthetic": 1},
        },
    )

    path_ids = inventory["path_ids"]
    labels = inventory["requested_labels"]
    key_hashes = [
        runner._sha256_bytes(
            np.asarray(
                [runner.derive_edge_noise_seed(str(path_id), outer, 1, 0) for path_id in path_ids],
                dtype=np.uint64,
            ).tobytes()
        )
        for outer in range(runner.OUTER_STEPS)
    ]
    for row_index, row in enumerate(runner.EULERIAN_ROWS):
        anchors = np.stack([starts.copy() for _ in runner.ANCHORS]).astype(np.float32)
        if row == "teacher":
            anchors[1:] = teacher
        telemetry = [
            {
                "row": row,
                "completed_step": step,
                "remaining_time": float(runner.OUTER_STEPS - step),
                "accepted_substeps": 1,
                "attempts": [{"substeps": 1, "clipped": 0, "proposed": runner.PATH_COUNT * 2 * 28 * 28, "clipping_fraction": 0.0}],
                "accepted_clipped": 0,
                "accepted_proposed": runner.PATH_COUNT * 2 * 28 * 28,
                "accepted_clipping_fraction": 0.0,
                "learned_step_rms": float(row_index),
                "free_step_rms": 0.0,
                "noise_step_rms": 0.0,
                "controller_to_free_ratio": 0.0,
                "controller_to_noise_ratio": 0.0,
                "state_displacement_rms": 0.0,
                "minimum_mass": float(starts.min()),
                "maximum_mass": float(starts.max()),
                "maximum_mass_error": float(np.max(np.abs(starts.sum(axis=1, dtype=np.float64) - 1.0))),
                "poisson_divergence_residual_maximum": 0.0,
                "finite": 1,
                "elapsed_seconds": float(step) / 10_000.0,
                "cuda_allocated_bytes": 0,
            }
            for step in range(1, runner.OUTER_STEPS + 1)
        ]
        deterministic = [
            {key: value for key, value in item.items() if key not in {"elapsed_seconds", "cuda_allocated_bytes"}}
            for item in telemetry
        ]
        digest = runner._scientific_digest(
            {
                "anchors": anchors,
                "anchor_steps": np.asarray(runner.ANCHORS, dtype=np.int64),
                "path_ids": path_ids,
                "requested_labels": labels,
            },
            {"row": row, "telemetry": deterministic, "crn_key_hashes": key_hashes},
        )
        provider_payload = None
        if row == "ddpm_eulerian":
            provider_payload = {
                "predicted_mass": np.stack([teacher for _ in runner.ANCHORS[1:]]).astype(np.float32),
                "ddpm_timestep": np.stack(
                    [np.full(runner.PATH_COUNT, value, dtype=np.int64) for value in (749, 500, 250, 0)]
                ),
            }
        result = runner.EulerianRowResult(
            row=row,
            anchors=anchors,
            anchor_steps=np.asarray(runner.ANCHORS, dtype=np.int64),
            path_ids=path_ids,
            requested_labels=labels,
            telemetry=telemetry,
            crn_key_hashes=key_hashes,
            scientific_digest=digest,
            provider_anchor_payload=provider_payload,
        )
        runner.save_eulerian_population(
            root,
            result,
            start_bank_sha256=runner.sha256_file(root / "inventory/start_bank.npz"),
            rng_contract_sha256=runner.sha256_file(root / "inventory/rng_contract.json"),
        )

    native_anchors = np.stack([latent["z"] for _ in runner.NATIVE_ANCHORS]).astype(np.float32)
    native_telemetry = [
        {
            "completed_step": step,
            "ddpm_timestep": 1_000 - step,
            "state_mean": float(latent["z"].mean()),
            "state_std": float(latent["z"].std()),
            "epsilon_rms": 0.0,
            "finite": 1,
            "elapsed_seconds": float(step) / 100_000.0,
        }
        for step in range(1, 1_001)
    ]
    native_metadata = {
        "row": "native_ddpm",
        "comparison_role": "contextual_latent_linked_state_unpaired",
        "telemetry": [{key: value for key, value in item.items() if key != "elapsed_seconds"} for item in native_telemetry],
    }
    native_digest = runner._scientific_digest(
        {
            "model_anchors": native_anchors,
            "reverse_steps": np.asarray(runner.NATIVE_ANCHORS, dtype=np.int64),
            "path_ids": path_ids,
            "requested_labels": labels,
            "reverse_seeds": inventory["native_reverse_seeds"],
        },
        native_metadata,
    )
    runner.save_native_population(
        root,
        runner.NativeDDPMResult(
            model_anchors=native_anchors,
            reverse_steps=np.asarray(runner.NATIVE_ANCHORS, dtype=np.int64),
            path_ids=path_ids,
            requested_labels=labels,
            reverse_seeds=inventory["native_reverse_seeds"],
            latent_bank_sha256=runner._hash_array(latent["z"]),
            scientific_digest=native_digest,
            telemetry=native_telemetry,
        ),
    )
    runner._write_json(
        root / "telemetry/crn_key_hashes.json",
        {
            row: runner._npz(root / f"populations/{row}.npz")["crn_key_hashes"].tolist()
            for row in runner.EULERIAN_ROWS
        },
    )
    historical_identity = "synthetic-historical-model-state"
    ddpm_identity = "synthetic-ddpm-model-state"
    runner._write_json(
        root / "telemetry/model_state_identity.json",
        {
            "schema": runner.VERSION + "-model-state-identity",
            "historical_pre": historical_identity,
            "historical_post": historical_identity,
            "ddpm_pre": ddpm_identity,
            "ddpm_post": ddpm_identity,
            "populations": [
                {
                    "row": row,
                    "pre": {"historical": historical_identity, "ddpm": ddpm_identity}
                    if row != "native_ddpm"
                    else {"ddpm": ddpm_identity},
                    "post": {"historical": historical_identity, "ddpm": ddpm_identity}
                    if row != "native_ddpm"
                    else {"ddpm": ddpm_identity},
                    "passed": 1,
                }
                for row in runner.ROWS
            ],
            "passed": 1,
        },
    )
    return root, runner.seal_populations(root)


def test_population_writers_seal_all_authorities_and_exact_native_telemetry(tmp_path: Path) -> None:
    root, seal = _synthetic_population_run(tmp_path)
    verified = runner._verify_population_seal(root)
    assert verified == seal
    expected_authorities = {
        "inventory/start_bank.npz",
        "inventory/teacher_target_bank.npz",
        "inventory/ddpm_latent_bank.npz",
        "inventory/rng_contract.json",
        "inventory/STARTS_SEALED.json",
        "inventory/path_inventory.csv",
        "source_bindings.json",
        "checkpoint_bindings.json",
        "telemetry/crn_key_hashes.json",
        "telemetry/model_state_identity.json",
        "telemetry/adapter_paths.npz",
        "telemetry/adapter_paths.csv",
        "telemetry/summary.json",
    }
    assert expected_authorities <= set(seal["files"])
    assert set(seal["authority_receipts"]) == {
        "inventory/path_inventory.csv",
        "inventory/start_bank.npz",
        "inventory/ddpm_latent_bank.npz",
        "inventory/rng_contract.json",
        "source_bindings.json",
        "checkpoint_bindings.json",
        "inventory/teacher_target_bank.npz",
        "inventory/STARTS_SEALED.json",
        "telemetry/crn_key_hashes.json",
        "telemetry/model_state_identity.json",
        "telemetry/adapter_paths.npz",
        "telemetry/adapter_paths.csv",
        "telemetry/summary.json",
    }
    assert set(seal["row_receipts"]) == set(runner.ROWS)
    for row in runner.ROWS:
        assert f"populations/{row}.npz" in seal["files"]
        assert f"populations/{row}_uint8.npz" in seal["files"]
        assert f"telemetry/{row}_steps.csv" in seal["files"]
        with (root / f"telemetry/{row}_steps.csv").open(newline="", encoding="utf-8") as handle:
            telemetry_rows = list(csv.DictReader(handle))
        expected_count = 1_000 if row == "native_ddpm" else runner.OUTER_STEPS
        assert len(telemetry_rows) == expected_count
        raw = runner._npz(root / f"populations/{row}.npz")
        assert len(str(raw["scientific_digest"])) == 64
        assert seal["row_receipts"][row] == {
            "raw": seal["files"][f"populations/{row}.npz"],
            "uint8": seal["files"][f"populations/{row}_uint8.npz"],
            "telemetry": seal["files"][f"telemetry/{row}_steps.csv"],
            "telemetry_row_count": expected_count,
            "scientific_digest": str(raw["scientific_digest"]),
        }
    native = runner._npz(root / "populations/native_ddpm.npz")
    latent = runner._npz(root / "inventory/ddpm_latent_bank.npz")
    inventory = runner.build_path_inventory()
    np.testing.assert_array_equal(native["model_anchors"][0], latent["z"])
    np.testing.assert_array_equal(native["reverse_steps"], np.asarray(runner.NATIVE_ANCHORS))
    np.testing.assert_array_equal(native["reverse_seeds"], inventory["native_reverse_seeds"])
    native_uint8 = runner._npz(root / "populations/native_ddpm_uint8.npz")
    np.testing.assert_array_equal(native_uint8["images_uint8"], runner._model_to_uint8(native["model_anchors"]))


def test_sealed_population_rendering_is_exact_complete_and_deterministic(tmp_path: Path) -> None:
    root, _ = _synthetic_population_run(tmp_path)
    first = runner.render_sealed_population_images(root)
    expected_paths = {
        *(f"images/{row}_final.png" for row in runner.ROWS),
        *(f"images/{row}_trajectory.png" for row in runner.ROWS),
        "images/comparison_final.png",
    }
    assert first["schema"] == runner.VERSION + "-image-receipts"
    assert set(first["files"]) == expected_paths
    second = runner.render_sealed_population_images(root)
    assert second == first
    final_images = []
    for row in runner.ROWS:
        population = runner._npz(root / f"populations/{row}_uint8.npz")
        images = population["images_uint8"]
        ids = population["path_ids"].astype(np.str_).tolist()
        steps = population["anchor_steps"].tolist()
        runner.v3._verify_sheet_pixels(
            root / f"images/{row}_final.png", images[-1], columns=10, scale=3, captions=ids
        )
        runner.v3._verify_sheet_pixels(
            root / f"images/{row}_trajectory.png",
            images.transpose(1, 0, 2, 3).reshape(-1, 28, 28),
            columns=5,
            scale=2,
            captions=[f"{path_id}@{step}" for path_id in ids for step in steps],
        )
        final_images.append(images[-1])
    runner.v3._verify_sheet_pixels(
        root / "images/comparison_final.png",
        np.concatenate(final_images),
        columns=10,
        scale=3,
        captions=[f"{row}:{index:03d}" for row in runner.ROWS for index in range(runner.PATH_COUNT)],
    )


def _refresh_population_seal_receipts(root: Path, relatives: tuple[str, ...]) -> None:
    seal_path = root / "populations/POPULATIONS_SEALED.json"
    seal = runner._read_json(seal_path)
    for relative in relatives:
        path = root / relative
        receipt = {"bytes": path.stat().st_size, "sha256": runner.sha256_file(path)}
        seal["files"][relative] = receipt
        for row in runner.ROWS:
            mapping = {
                f"populations/{row}.npz": "raw",
                f"populations/{row}_uint8.npz": "uint8",
                f"telemetry/{row}_steps.csv": "telemetry",
            }
            if relative in mapping:
                seal["row_receipts"][row][mapping[relative]] = receipt
        if relative in seal["authority_receipts"]:
            seal["authority_receipts"][relative] = receipt
    unsigned = {key: value for key, value in seal.items() if key != "seal_sha256"}
    seal["seal_sha256"] = runner._sha256_bytes(runner._canonical_json_bytes(unsigned))
    runner._write_json(seal_path, seal)


def _refresh_row_scientific_digest(root: Path, row: str, raw: dict[str, np.ndarray]) -> None:
    if row == "native_ddpm":
        metadata = {
            "row": row,
            "comparison_role": "contextual_latent_linked_state_unpaired",
            "telemetry": json.loads(str(raw["telemetry_scientific_json"])),
        }
        arrays = {
            "model_anchors": raw["model_anchors"],
            "reverse_steps": raw["reverse_steps"],
            "path_ids": raw["path_ids"],
            "requested_labels": raw["requested_labels"],
            "reverse_seeds": raw["reverse_seeds"],
        }
    else:
        metadata = {
            "row": row,
            "telemetry": json.loads(str(raw["telemetry_scientific_json"])),
            "crn_key_hashes": raw["crn_key_hashes"].tolist(),
        }
        arrays = {
            "anchors": raw["anchors"],
            "anchor_steps": raw["anchor_steps"],
            "path_ids": raw["path_ids"],
            "requested_labels": raw["requested_labels"],
        }
    raw["scientific_digest"] = np.asarray(runner._scientific_digest(arrays, metadata))
    runner._write_npz(root / f"populations/{row}.npz", **raw)
    seal = runner._read_json(root / "populations/POPULATIONS_SEALED.json")
    seal["row_receipts"][row]["scientific_digest"] = str(raw["scientific_digest"])
    runner._write_json(root / "populations/POPULATIONS_SEALED.json", seal)


@pytest.mark.parametrize(
    "tamper",
    (
        "native_telemetry_count",
        "native_reverse_seed",
        "crn_key_hash",
        "rng_root",
        "model_state_identity",
        "adapter_predicted_mass_shape",
        "raw_uint8_start_authority",
    ),
)
def test_population_seal_replays_semantics_after_coordinated_hash_updates(tmp_path: Path, tamper: str) -> None:
    root, _ = _synthetic_population_run(tmp_path)
    changed: tuple[str, ...]
    if tamper == "native_telemetry_count":
        relative = "telemetry/native_ddpm_steps.csv"
        lines = (root / relative).read_bytes().splitlines(keepends=True)
        runner.v3._atomic_bytes(root / relative, b"".join(lines[:-1]))
        changed = (relative,)
    elif tamper == "native_reverse_seed":
        relative = "populations/native_ddpm.npz"
        raw = runner._npz(root / relative)
        raw["reverse_seeds"] = raw["reverse_seeds"].copy()
        raw["reverse_seeds"][0] += np.uint64(1)
        _refresh_row_scientific_digest(root, "native_ddpm", raw)
        changed = (relative,)
    elif tamper == "crn_key_hash":
        relative = "populations/null.npz"
        raw = runner._npz(root / relative)
        raw["crn_key_hashes"] = raw["crn_key_hashes"].copy()
        raw["crn_key_hashes"][0] = "0" * 64
        _refresh_row_scientific_digest(root, "null", raw)
        changed = (relative,)
    elif tamper == "rng_root":
        relative = "inventory/rng_contract.json"
        contract = runner._read_json(root / relative)
        contract["edge_noise_root"] += 1
        runner._write_json(root / relative, contract)
        changed = (relative,)
    elif tamper == "model_state_identity":
        relative = "telemetry/model_state_identity.json"
        identity = runner._read_json(root / relative)
        changed_digest = "coordinated-but-unauthenticated-ddpm-state"
        identity["ddpm_pre"] = identity["ddpm_post"] = changed_digest
        for item in identity["populations"]:
            item["pre"]["ddpm"] = item["post"]["ddpm"] = changed_digest
        runner._write_json(root / relative, identity)
        changed = (relative,)
    elif tamper == "adapter_predicted_mass_shape":
        relative = "telemetry/adapter_paths.npz"
        paths = runner._npz(root / relative)
        paths["predicted_mass"] = paths["predicted_mass"][:, :-1]
        runner._write_npz(root / relative, **paths)
        changed = (relative,)
    else:
        raw_relative = "populations/null.npz"
        uint8_relative = "populations/null_uint8.npz"
        raw = runner._npz(root / raw_relative)
        raw["anchors"] = raw["anchors"].copy()
        raw["anchors"][0, 0, 0] += np.float32(1e-4)
        raw["anchors"][0, 0, 1] -= np.float32(1e-4)
        _refresh_row_scientific_digest(root, "null", raw)
        rendered = runner._npz(root / uint8_relative)
        rendered["images_uint8"] = np.stack([runner.mass_to_uint8(anchor) for anchor in raw["anchors"]])
        runner._write_npz(root / uint8_relative, **rendered)
        changed = (raw_relative, uint8_relative)
    _refresh_population_seal_receipts(root, changed)
    with pytest.raises(runner.IntegrityFailure):
        runner._verify_population_seal(root)


def test_runner_diversity_helper_preserves_real_benchmark_schema_and_per_class_evidence() -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), runner.PATHS_PER_CLASS)
    generated = np.zeros((runner.PATH_COUNT, 28, 28), dtype=np.uint8)
    reference = np.zeros_like(generated)
    for index in range(runner.PATH_COUNT):
        generated[index].flat[index] = np.uint8(80 + index)
        reference[index].flat[index] = np.uint8(160 + index)
    result = runner._diversity(generated, labels, reference, labels)
    assert set(result) == {"aggregate_median_ratio", "by_class"}
    assert np.isfinite(result["aggregate_median_ratio"])
    assert set(result["by_class"]) == {str(digit) for digit in range(10)}
    assert all(set(values) == {"count", "generated_median_nn_mse", "real_median_nn_mse", "ratio"}
               for values in result["by_class"].values())


def test_machine_evaluation_persists_confusion_per_class_and_sealed_predicted_mass_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _synthetic_population_run(tmp_path)
    arff = tmp_path / "synthetic-terminal.arff"
    runner.v3._atomic_bytes(arff, b"synthetic terminal reference")
    monkeypatch.setattr(runner, "MNIST_ARFF_BYTES", arff.stat().st_size)
    monkeypatch.setattr(runner, "MNIST_ARFF_SHA256", runner.sha256_file(arff))
    monkeypatch.setattr(runner, "_load_evaluator_after_seal", lambda *_args, **_kwargs: object())
    labels = runner.build_path_inventory()["requested_labels"]
    reference = np.zeros((runner.PATH_COUNT, 28, 28), dtype=np.uint8)
    for index in range(runner.PATH_COUNT):
        reference[index].flat[index] = np.uint8(1 + index)
    monkeypatch.setattr(runner, "read_mnist_arff_slice", lambda *_args, **_kwargs: (reference, labels))

    def fake_score(_model: object, _images: np.ndarray, requested: np.ndarray,
                   sample_ids: np.ndarray, **_kwargs: object) -> dict[str, object]:
        predictions = np.asarray(requested, dtype=np.int64).copy()
        logits = np.full((len(predictions), 10), -1.0, dtype=np.float64)
        logits[np.arange(len(predictions)), predictions] = 1.0
        return {
            "loss": 0.0,
            "accuracy": 1.0,
            "sample_ids": np.asarray(sample_ids),
            "requested_labels": predictions,
            "predictions": predictions,
            "logits": logits,
            "per_class": {str(digit): {"count": 4, "accuracy": 1.0} for digit in range(10)},
        }

    calls = {"diversity": 0}

    def fake_diversity(_images: np.ndarray, _labels: np.ndarray,
                       _reference: np.ndarray, _reference_labels: np.ndarray) -> dict[str, object]:
        call = calls["diversity"]
        calls["diversity"] += 1
        # Row-major five anchors: historical step64 is call 11, candidate step64 is 16.
        ratio = 0.1
        if call == 16:
            ratio = 0.3
        if call == 25:  # sealed adapter predicted-mass endpoint
            ratio = 0.5
        by_class_ratios = {
            digit: (0.2 if digit < 7 else 0.05) if call == 19 else ratio
            for digit in range(10)
        }
        return {
            "aggregate_median_ratio": ratio,
            "by_class": {
                str(digit): {
                    "count": 4,
                    "generated_median_nn_mse": by_class_ratios[digit],
                    "real_median_nn_mse": 1.0,
                    "ratio": by_class_ratios[digit],
                }
                for digit in range(10)
            },
        }

    monkeypatch.setattr(runner, "evaluate_generated_labels", fake_score)
    monkeypatch.setattr(runner, "within_class_nn_diversity", fake_diversity)
    monkeypatch.setattr(
        runner,
        "exact_duplicate_metrics",
        lambda *_args, **_kwargs: {"duplicate_pair_count": 0, "by_class": {}},
    )
    evaluated = runner.evaluate_sealed_populations(root, arff=arff, ddpm_run_dir=tmp_path, device="cpu")
    comparison = evaluated["comparison"]
    assert comparison["predicted_mass_diverse"] == 1
    assert comparison["predicted_mass_diversity_ratio"] == 0.5
    assert comparison["candidate_diversity_classes_exceeding_historical"] == 7
    assert comparison["candidate_diversity_class_supportive"] == 1
    assert comparison["candidate_early_joint_steps"] == [64]
    assert comparison["early_fidelity_role"] == "machine_only_exploratory_proxy_not_human_D1"
    assert calls["diversity"] == 26

    predictions = runner._npz(root / "evaluation/predictions.npz")
    for row in runner.ROWS:
        metrics = runner._read_json(root / f"evaluation/{row}_metrics.json")
        assert set(metrics["classifier_per_class"]) == {str(digit) for digit in range(10)}
        assert set(metrics["diversity_by_class"]) == {str(digit) for digit in range(10)}
        confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
        assert confusion.shape == (10, 10)
        np.testing.assert_array_equal(np.diag(confusion), np.full(10, 4, dtype=np.int64))
        final_anchor = runner.NATIVE_ANCHORS[-1] if row == "native_ddpm" else runner.ANCHORS[-1]
        np.testing.assert_array_equal(predictions[f"{row}_{final_anchor}_confusion_matrix"], confusion)
    with (root / "evaluation/per_class_metrics.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == len(runner.ROWS) * len(runner.ANCHORS) * 10


def _write_synthetic_evaluation(root: Path) -> None:
    runner._write_json(
        root / "evaluation/SCORING_READY.json",
        {
            "schema": runner.VERSION + "-scoring-ready",
            "population_seal_sha256": runner.sha256_file(root / "populations/POPULATIONS_SEALED.json"),
            "terminal_rows_opened_after_seal": 1,
        },
    )
    anchor_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    prediction_arrays: dict[str, np.ndarray] = {}
    endpoints: dict[str, dict[str, object]] = {}
    for row in runner.ROWS:
        population = runner._npz(root / f"populations/{row}_uint8.npz")
        images = population["images_uint8"]
        labels = population["requested_labels"].astype(np.int64)
        steps = population["anchor_steps"].astype(np.int64)
        ratio = 0.3 if row == "native_ddpm" else 0.1
        per_class = {str(digit): {"count": 4, "accuracy": 1.0} for digit in range(10)}
        diversity_by_class = {
            str(digit): {
                "count": 4,
                "generated_median_nn_mse": ratio,
                "real_median_nn_mse": 1.0,
                "ratio": ratio,
            }
            for digit in range(10)
        }
        for anchor_index, step in enumerate(steps.tolist()):
            predictions = labels.copy()
            logits = np.full((runner.PATH_COUNT, 10), -1.0, dtype=np.float64)
            logits[np.arange(runner.PATH_COUNT), predictions] = 1.0
            confusion = np.zeros((10, 10), dtype=np.int64)
            np.add.at(confusion, (labels, predictions), 1)
            prefix = f"{row}_{step}"
            prediction_arrays[f"{prefix}_predictions"] = predictions
            prediction_arrays[f"{prefix}_logits"] = logits
            prediction_arrays[f"{prefix}_confusion_matrix"] = confusion
            duplicates = int(runner.exact_duplicate_metrics(images[anchor_index], labels)["duplicate_pair_count"])
            anchor_rows.append(
                {
                    "row": row,
                    "anchor_step": int(step),
                    "classifier_accuracy": 1.0,
                    "diversity_ratio": ratio,
                    "exact_duplicate_pair_count": duplicates,
                    "class_coverage": 10,
                }
            )
            class_rows.extend(
                {
                    "row": row,
                    "anchor_step": int(step),
                    "requested_label": digit,
                    "classifier_accuracy": 1.0,
                    "diversity_ratio": ratio,
                }
                for digit in range(10)
            )
        endpoints[row] = {
            **anchor_rows[-1],
            "classifier_per_class": per_class,
            "diversity_by_class": diversity_by_class,
            "confusion_matrix": confusion.tolist(),
        }
        runner._write_json(root / f"evaluation/{row}_metrics.json", endpoints[row])
    runner._write_csv(root / "evaluation/per_anchor_metrics.csv", anchor_rows)
    runner._write_csv(root / "evaluation/per_class_metrics.csv", class_rows)
    runner._write_npz(root / "evaluation/predictions.npz", **prediction_arrays)
    historical = endpoints["historical"]
    candidate = endpoints["ddpm_eulerian"]
    comparison = {
        "candidate_diversity_ratio": candidate["diversity_ratio"],
        "historical_diversity_ratio": historical["diversity_ratio"],
        "candidate_to_historical_diversity_ratio": float(candidate["diversity_ratio"])
        / max(float(historical["diversity_ratio"]), 1e-12),
        "candidate_classifier_accuracy_difference": 0.0,
        "candidate_diversity_classes_exceeding_historical": 0,
        "candidate_diversity_class_supportive": 0,
        **runner.early_joint_machine_proxy(anchor_rows),
        "predicted_mass_diversity_ratio": 0.1,
        "predicted_mass_diversity_by_class": {
            str(digit): {"count": 4, "ratio": 0.1} for digit in range(10)
        },
        "predicted_mass_diverse": 0,
    }
    runner._write_json(root / "evaluation/ddpm_eulerian_minus_historical.json", comparison)
    runner._write_json(
        root / "evaluation/contextual_native_ddpm.json",
        endpoints["native_ddpm"] | {"comparison_role": "contextual_latent_linked_state_unpaired"},
    )
    starts = runner._npz(root / "inventory/start_bank.npz")["masses"]
    targets = runner._npz(root / "inventory/teacher_target_bank.npz")["masses"]
    teacher = runner._npz(root / "populations/teacher.npz")["anchors"]
    start_l2 = ((starts - targets) ** 2).sum(axis=1)
    short_l2 = ((teacher[1] - targets) ** 2).sum(axis=1)
    final_l2 = ((teacher[-1] - targets) ** 2).sum(axis=1)
    teacher_gate = {
        "improved_count": int(np.sum(final_l2 < start_l2)),
        "short_median_relative_l2": float(np.median(short_l2 / np.maximum(start_l2, 1e-20))),
        "final_median_relative_l2": float(np.median(final_l2 / np.maximum(start_l2, 1e-20))),
        "classifier_accuracy": 1.0,
    }
    teacher_gate["passed"] = int(
        teacher_gate["improved_count"] >= runner.TEACHER_IMPROVED_MINIMUM
        and teacher_gate["short_median_relative_l2"] <= runner.TEACHER_SHORT_RELATIVE_L2_MAXIMUM
        and teacher_gate["final_median_relative_l2"] <= runner.TEACHER_FINAL_RELATIVE_L2_MAXIMUM
        and teacher_gate["classifier_accuracy"] >= runner.TEACHER_CLASSIFIER_ACCURACY_MINIMUM
    )
    runner._write_json(root / "gates.json", runner._build_machine_gates(root, endpoints, comparison, teacher_gate))


def _synthetic_awaiting_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    historical_receipt: bool = False,
    semantic_terminalize: bool = True,
) -> Path:
    # The production helper is fail-closed in real code. Tests replace it only
    # in-process so compact synthetic trees do not need real ARFF/evaluator access.
    monkeypatch.setattr(runner, "_is_production_config", lambda _config: False)
    root, _ = _synthetic_population_run(tmp_path, historical_receipt=historical_receipt)
    runner.v3._atomic_bytes(root / "command.txt", b"synthetic test-only command\n")
    runner._write_json(
        root / "claim_boundary.json",
        {
            "schema": runner.VERSION + "-claim-boundary",
            "mode": runner.RESEARCH_MODE,
            "establishes_at_most": "synthetic test only",
            "does_not_establish": ["scientific evidence"],
        },
    )
    for stage in runner.STAGE_ORDER[:-1]:
        runner._record_stage(root, stage)
    runner.ResourceGovernor(root, runner.ResourceBudget(), device="cpu").write()

    seal_path = root / "populations/POPULATIONS_SEALED.json"
    _write_synthetic_evaluation(root)

    runner.prepare_blind_review(root)
    runner._status(root, "awaiting_human_review", route="awaiting_human_review")
    if semantic_terminalize:
        runner._terminalize(root)
    else:
        runner._write_json(
            root / "VERIFY_RECEIPT.json",
            {"schema": runner.VERSION + "-verify-receipt", "passed": 1, "written_by": "synthetic-test"},
        )
        runner._write_reports(root)
        runner._manifest(root)
    return root


def test_public_verifier_is_read_only_on_a_semantically_valid_awaiting_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _synthetic_awaiting_run(tmp_path, monkeypatch)
    before = {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns, runner.sha256_file(path))
        for path in root.rglob("*")
        if path.is_file()
    }
    receipt = runner.verify_run(root)
    after = {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns, runner.sha256_file(path))
        for path in root.rglob("*")
        if path.is_file()
    }
    assert receipt["passed"] == 1
    assert receipt["state"] == "awaiting_human_review"
    assert before == after


def test_public_verifier_rebuilds_start_seed_authority_after_coordinated_reseal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _synthetic_awaiting_run(
        tmp_path, monkeypatch, historical_receipt=True, semantic_terminalize=False
    )
    start_path = root / "inventory/start_bank.npz"
    original_starts = runner._npz(start_path)["masses"].copy()
    tampered_start = runner._npz(start_path)
    tampered_start["masses"] = tampered_start["masses"].copy()
    tampered_start["masses"][0, 0] += np.float32(1e-4)
    tampered_start["masses"][0, 1] -= np.float32(1e-4)
    start_science = {key: tampered_start[key] for key in ("masses", "path_ids", "requested_labels", "source_seeds")}
    tampered_start["scientific_sha256"] = np.asarray(runner._scientific_digest(start_science, {}))
    runner._write_npz(start_path, **tampered_start)

    changed = ["inventory/start_bank.npz"]
    for row in runner.EULERIAN_ROWS:
        raw_relative = f"populations/{row}.npz"
        uint8_relative = f"populations/{row}_uint8.npz"
        raw = runner._npz(root / raw_relative)
        raw["anchors"] = raw["anchors"].copy()
        raw["anchors"][0] = tampered_start["masses"]
        raw["start_bank_sha256"] = np.asarray(runner.sha256_file(start_path))
        _refresh_row_scientific_digest(root, row, raw)
        raster = runner._npz(root / uint8_relative)
        raster["images_uint8"] = np.stack([runner.mass_to_uint8(anchor) for anchor in raw["anchors"]])
        runner._write_npz(root / uint8_relative, **raster)
        changed.extend((raw_relative, uint8_relative))

    starts_seal_path = root / "inventory/STARTS_SEALED.json"
    starts_seal = runner._read_json(starts_seal_path)
    starts_seal["start_bank_sha256"] = runner.sha256_file(start_path)
    runner._write_json(starts_seal_path, starts_seal)
    changed.append("inventory/STARTS_SEALED.json")
    _refresh_population_seal_receipts(root, tuple(changed))
    population_seal_path = root / "populations/POPULATIONS_SEALED.json"
    population_seal = runner._read_json(population_seal_path)
    population_seal["starts_seal_sha256"] = runner.sha256_file(starts_seal_path)
    unsigned = {key: value for key, value in population_seal.items() if key != "seal_sha256"}
    population_seal["seal_sha256"] = runner._sha256_bytes(runner._canonical_json_bytes(unsigned))
    runner._write_json(population_seal_path, population_seal)
    runner._manifest(root)

    monkeypatch.setattr(runner, "build_start_bank", lambda _config, _inventory: original_starts.copy())
    before = {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns, runner.sha256_file(path))
        for path in root.rglob("*")
        if path.is_file()
    }
    with pytest.raises(runner.IntegrityFailure, match="rebuilt start bank"):
        runner.verify_run(root)
    after = {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns, runner.sha256_file(path))
        for path in root.rglob("*")
        if path.is_file()
    }
    assert before == after


def _synthetic_complete_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = _synthetic_awaiting_run(tmp_path, monkeypatch)
    key = runner._read_json(root / "review/private_key.json")["items"]
    answers = tmp_path / "manual-review.csv"
    runner._write_csv(
        answers,
        [
            {
                "blind_id": item["blind_id"],
                "recognizable": 1,
                "perceived_digit": item["requested_label"],
                "notes": "synthetic test answer",
            }
            for item in key
        ],
    )
    args = SimpleNamespace(
        run_dir=str(root),
        answers=str(answers),
        reviewer="synthetic-test-reviewer",
        confirm_manual_review=True,
    )
    assert runner.finalize_review(args) == 0
    assert runner.verify_run(root)["state"] == "complete"
    return root


def test_invalid_manual_review_is_logged_and_corrected_retry_is_transactional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _synthetic_awaiting_run(tmp_path, monkeypatch)
    population_before = (root / "populations/POPULATIONS_SEALED.json").read_bytes()
    key_before = (root / "review/private_key.json").read_bytes()
    key = runner._read_json(root / "review/private_key.json")["items"]
    invalid = tmp_path / "invalid-review.csv"
    assert root not in invalid.resolve().parents
    runner._write_csv(
        invalid,
        [
            {
                "blind_id": item["blind_id"],
                "recognizable": 2 if index == 0 else 1,
                "perceived_digit": item["requested_label"],
                "notes": "synthetic invalid answer",
            }
            for index, item in enumerate(key)
        ],
    )
    invalid_args = SimpleNamespace(
        run_dir=str(root),
        answers=str(invalid),
        reviewer="synthetic-test-reviewer",
        confirm_manual_review=True,
    )
    with pytest.raises((runner.IntegrityFailure, ValueError), match="review answer value"):
        runner.finalize_review(invalid_args)
    attempts_path = root / "review/submission_attempts.json"
    attempts = runner._read_json(attempts_path)
    assert attempts["schema"] == runner.VERSION + "-review-submission-attempts"
    assert len(attempts["attempts"]) == 1
    first = attempts["attempts"][0]
    assert first["reviewer"] == "synthetic-test-reviewer"
    assert Path(first["answers_path"]) == invalid.resolve()
    assert first["answers_sha256"] == runner.sha256_file(invalid)
    assert first["passed"] == 0
    assert "review answer value" in first["error"]
    assert runner._read_json(root / "status.json")["state"] == "awaiting_human_review"
    assert not (root / "review/human_review.json").exists()
    assert not (root / "review/human_review_by_row.json").exists()
    assert not (root / "outcome.json").exists()
    assert runner.verify_run(root)["state"] == "awaiting_human_review"

    corrected = tmp_path / "corrected-review.csv"
    runner._write_csv(
        corrected,
        [
            {
                "blind_id": item["blind_id"],
                "recognizable": 1,
                "perceived_digit": item["requested_label"],
                "notes": "synthetic corrected answer",
            }
            for item in key
        ],
    )
    corrected_args = SimpleNamespace(
        run_dir=str(root),
        answers=str(corrected),
        reviewer="synthetic-test-reviewer",
        confirm_manual_review=True,
    )
    assert runner.finalize_review(corrected_args) == 0
    attempts = runner._read_json(attempts_path)["attempts"]
    assert [attempt["passed"] for attempt in attempts] == [0, 1]
    assert attempts[1]["answers_sha256"] == runner.sha256_file(corrected)
    assert runner.verify_run(root)["state"] == "complete"
    assert (root / "populations/POPULATIONS_SEALED.json").read_bytes() == population_before
    assert (root / "review/private_key.json").read_bytes() == key_before


def test_finalize_review_verifies_awaiting_tree_before_adopting_external_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _synthetic_awaiting_run(tmp_path, monkeypatch)
    key = runner._read_json(root / "review/private_key.json")["items"]
    answers = tmp_path / "external-valid-review.csv"
    assert root not in answers.resolve().parents
    runner._write_csv(
        answers,
        [
            {
                "blind_id": item["blind_id"],
                "recognizable": 1,
                "perceived_digit": item["requested_label"],
                "notes": "synthetic external answer",
            }
            for item in key
        ],
    )
    membership = runner._read_json(root / "review/membership.json")
    membership["count"] = 79
    runner._write_json(root / "review/membership.json", membership)
    runner._manifest(root)
    before = {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns, runner.sha256_file(path))
        for path in root.rglob("*")
        if path.is_file()
    }
    args = SimpleNamespace(
        run_dir=str(root),
        answers=str(answers),
        reviewer="synthetic-test-reviewer",
        confirm_manual_review=True,
    )
    with pytest.raises(runner.IntegrityFailure, match="review|membership|awaiting|semantic"):
        runner.finalize_review(args)
    after = {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns, runner.sha256_file(path))
        for path in root.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert not (root / "review/human_review_answers.csv").exists()
    assert not (root / "review/human_review.json").exists()
    assert not (root / "outcome.json").exists()


@pytest.mark.parametrize("error_type", (OSError, KeyboardInterrupt))
def test_valid_review_commit_failure_is_cleanly_retryable_without_partial_adoption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    root = _synthetic_awaiting_run(tmp_path, monkeypatch)
    population_before = (root / "populations/POPULATIONS_SEALED.json").read_bytes()
    key_before = (root / "review/private_key.json").read_bytes()
    key = runner._read_json(root / "review/private_key.json")["items"]
    answers = tmp_path / "valid-review.csv"
    runner._write_csv(
        answers,
        [
            {
                "blind_id": item["blind_id"],
                "recognizable": 1,
                "perceived_digit": item["requested_label"],
                "notes": "synthetic valid answer",
            }
            for item in key
        ],
    )
    original_write_json = runner._write_json
    armed = {"value": True}

    def fail_during_commit(path: Path, value: object) -> None:
        if Path(path).name == "human_review_by_row.json" and armed["value"]:
            armed["value"] = False
            raise error_type("synthetic review commit failure")
        original_write_json(Path(path), value)

    monkeypatch.setattr(runner, "_write_json", fail_during_commit)
    args = SimpleNamespace(
        run_dir=str(root),
        answers=str(answers),
        reviewer="synthetic-test-reviewer",
        confirm_manual_review=True,
    )
    with pytest.raises(error_type, match="synthetic review commit failure"):
        runner.finalize_review(args)
    assert runner._read_json(root / "status.json")["state"] == "awaiting_human_review"
    attempts = runner._read_json(root / "review/submission_attempts.json")["attempts"]
    assert [attempt["passed"] for attempt in attempts] == [0]
    assert "synthetic review commit failure" in attempts[0]["error"]
    assert runner._read_json(root / "resource_ledger.json")["open_events"] == []
    for relative in (
        "review/human_review_answers.csv",
        "review/human_review.json",
        "review/human_review_by_row.json",
        "outcome.json",
    ):
        assert not (root / relative).exists()
    assert runner._read_json(root / "gates.json")["candidate_human_fidelity"]["state"] == "pending"
    assert runner.verify_run(root)["state"] == "awaiting_human_review"

    assert runner.finalize_review(args) == 0
    assert [
        attempt["passed"]
        for attempt in runner._read_json(root / "review/submission_attempts.json")["attempts"]
    ] == [0, 1]
    assert runner.verify_run(root)["state"] == "complete"
    assert (root / "populations/POPULATIONS_SEALED.json").read_bytes() == population_before
    assert (root / "review/private_key.json").read_bytes() == key_before


def test_valid_commit_rollback_cleanup_resource_stop_is_verifier_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _synthetic_awaiting_run(tmp_path, monkeypatch)
    population_before = (root / "populations/POPULATIONS_SEALED.json").read_bytes()
    key_before = (root / "review/private_key.json").read_bytes()
    key = runner._read_json(root / "review/private_key.json")["items"]
    answers = tmp_path / "valid-review-before-nested-failure.csv"
    runner._write_csv(
        answers,
        [
            {
                "blind_id": item["blind_id"],
                "recognizable": 1,
                "perceived_digit": item["requested_label"],
                "notes": "rollback then cleanup resource stop",
            }
            for item in key
        ],
    )
    original_write_json = runner._write_json
    write_armed = {"value": True}

    def fail_valid_adoption(path: Path, value: object) -> None:
        if Path(path).name == "human_review_by_row.json" and write_armed["value"]:
            write_armed["value"] = False
            raise OSError("synthetic valid-adoption write failure")
        original_write_json(Path(path), value)

    original_complete = runner.ResourceGovernor.complete
    resource_armed = {"value": True}

    def fail_cleanup_completion(
        self: runner.ResourceGovernor,
        kind: str,
        *,
        terminal_override: bool = False,
    ) -> dict[str, object]:
        receipt = original_complete(self, kind, terminal_override=terminal_override)
        if kind == "human_review_terminalization" and not terminal_override and resource_armed["value"]:
            resource_armed["value"] = False
            receipt["storage_bytes_after"] = self.budget.max_storage_bytes + 1
            checks = {
                "wall": float(receipt["wall_seconds_after"]) <= self.budget.max_wall_seconds,
                "accelerator": float(receipt["accelerator_seconds_after"])
                <= self.budget.max_accelerator_seconds,
                "storage": False,
                "cuda": float(receipt["cuda_fraction"]) <= self.budget.max_cuda_fraction,
                "quantum": float(receipt["elapsed_seconds"])
                <= self.budget.maximum_quantum_seconds,
            }
            self.failed_admission = {
                "kind": kind,
                "phase": "post-completion",
                "checks": checks,
                "receipt": receipt,
                "passed": 0,
            }
            self.write()
            raise runner.ResourceStop(
                f"resource post-completion check failed for {kind}: {checks}"
            )
        return receipt

    monkeypatch.setattr(runner, "_write_json", fail_valid_adoption)
    monkeypatch.setattr(runner.ResourceGovernor, "complete", fail_cleanup_completion)
    args = SimpleNamespace(
        run_dir=str(root),
        answers=str(answers),
        reviewer="synthetic-test-reviewer",
        confirm_manual_review=True,
    )
    with pytest.raises(runner.ResourceStop, match="post-completion.*human_review_terminalization"):
        runner.finalize_review(args)

    status = runner._read_json(root / "status.json")
    assert status["state"] == status["route"] == "resource_stopped"
    assert status["failed_stage"] == "human_review_terminalization"
    ledger = runner._read_json(root / "resource_ledger.json")
    assert ledger["open_events"] == []
    assert ledger["failed_admission"]["phase"] == "post-completion"
    attempts = runner._read_json(root / "review/submission_attempts.json")["attempts"]
    assert [attempt["passed"] for attempt in attempts] == [0]
    assert "synthetic valid-adoption write failure" in attempts[0]["error"]
    for relative in (
        "review/human_review_answers.csv",
        "review/human_review.json",
        "review/human_review_by_row.json",
        "outcome.json",
    ):
        assert not (root / relative).exists()
    assert runner._read_json(root / "gates.json")["candidate_human_fidelity"]["state"] == "pending"
    assert runner.verify_run(root)["state"] == "resource_stopped"
    assert (root / "populations/POPULATIONS_SEALED.json").read_bytes() == population_before
    assert (root / "review/private_key.json").read_bytes() == key_before


def test_human_review_prospective_resource_stop_is_terminal_verifier_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _synthetic_awaiting_run(tmp_path, monkeypatch)
    population_before = (root / "populations/POPULATIONS_SEALED.json").read_bytes()
    key_before = (root / "review/private_key.json").read_bytes()
    key = runner._read_json(root / "review/private_key.json")["items"]
    answers = tmp_path / "valid-but-resource-stopped-review.csv"
    runner._write_csv(
        answers,
        [
            {
                "blind_id": item["blind_id"],
                "recognizable": 1,
                "perceived_digit": item["requested_label"],
                "notes": "must not be adopted",
            }
            for item in key
        ],
    )
    budget = runner.ResourceBudget()
    original_storage_bytes = runner.v3._storage_bytes

    def saturated_storage(path: str | Path) -> int:
        if Path(path).resolve() == root.resolve():
            return budget.max_storage_bytes
        return original_storage_bytes(path)

    monkeypatch.setattr(runner.v3, "_storage_bytes", saturated_storage)
    args = SimpleNamespace(
        run_dir=str(root),
        answers=str(answers),
        reviewer="synthetic-test-reviewer",
        confirm_manual_review=True,
    )
    with pytest.raises(runner.ResourceStop, match="human_review_terminalization"):
        runner.finalize_review(args)

    status = runner._read_json(root / "status.json")
    assert status["state"] == status["route"] == "resource_stopped"
    assert status["failed_stage"] == "human_review_terminalization"
    assert runner._read_json(root / "resource_ledger.json")["open_events"] == []
    assert runner._read_json(root / "resource_ledger.json")["failed_admission"]["passed"] == 0
    for relative in (
        "review/human_review_answers.csv",
        "review/human_review.json",
        "review/human_review_by_row.json",
        "outcome.json",
    ):
        assert not (root / relative).exists()
    assert not (root / "review/submission_attempts.json").exists()
    assert runner.verify_run(root)["state"] == "resource_stopped"
    assert (root / "populations/POPULATIONS_SEALED.json").read_bytes() == population_before
    assert (root / "review/private_key.json").read_bytes() == key_before


def test_finalize_review_preliminary_semantic_replay_is_inside_charged_quantum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _synthetic_awaiting_run(tmp_path, monkeypatch)
    key = runner._read_json(root / "review/private_key.json")["items"]
    answers = tmp_path / "charged-review.csv"
    runner._write_csv(
        answers,
        [
            {
                "blind_id": item["blind_id"],
                "recognizable": 1,
                "perceived_digit": item["requested_label"],
                "notes": "charged semantic replay",
            }
            for item in key
        ],
    )
    original_verify = runner.verify_run
    replay_open_events: list[tuple[str, ...]] = []

    def observe_semantic_replay(path: str | Path) -> dict[str, object]:
        replay_open_events.append(
            tuple(runner._read_json(root / "resource_ledger.json")["open_events"])
        )
        return original_verify(path)

    monkeypatch.setattr(runner, "verify_run", observe_semantic_replay)
    args = SimpleNamespace(
        run_dir=str(root),
        answers=str(answers),
        reviewer="synthetic-test-reviewer",
        confirm_manual_review=True,
    )
    assert runner.finalize_review(args) == 0
    assert replay_open_events == [("human_review_terminalization",)]
    ledger = runner._read_json(root / "resource_ledger.json")
    assert [(event["event"], event["kind"]) for event in ledger["events"][-2:]] == [
        ("admit", "human_review_terminalization"),
        ("complete", "human_review_terminalization"),
    ]


def test_postprecheck_baseexception_closes_review_quantum_and_allows_clean_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _synthetic_awaiting_run(tmp_path, monkeypatch)
    key = runner._read_json(root / "review/private_key.json")["items"]
    answers = tmp_path / "postprecheck-retry.csv"
    runner._write_csv(
        answers,
        [
            {
                "blind_id": item["blind_id"],
                "recognizable": 1,
                "perceived_digit": item["requested_label"],
                "notes": "retry after transient post-precheck interruption",
            }
            for item in key
        ],
    )
    original_verify_bundle = runner._verify_review_bundle
    calls = {"count": 0}

    def interrupt_after_precheck(*args: object, **kwargs: object) -> dict[str, object]:
        calls["count"] += 1
        if calls["count"] == 2:
            raise KeyboardInterrupt("synthetic post-precheck review interruption")
        return original_verify_bundle(*args, **kwargs)

    monkeypatch.setattr(runner, "_verify_review_bundle", interrupt_after_precheck)
    args = SimpleNamespace(
        run_dir=str(root),
        answers=str(answers),
        reviewer="synthetic-test-reviewer",
        confirm_manual_review=True,
    )
    with pytest.raises(KeyboardInterrupt, match="post-precheck review interruption"):
        runner.finalize_review(args)
    assert calls["count"] >= 2
    assert runner._read_json(root / "status.json")["state"] == "awaiting_human_review"
    assert runner._read_json(root / "resource_ledger.json")["open_events"] == []
    assert not (root / "review/human_review_answers.csv").exists()
    assert not (root / "outcome.json").exists()
    assert runner.verify_run(root)["state"] == "awaiting_human_review"
    assert runner.finalize_review(args) == 0
    assert runner.verify_run(root)["state"] == "complete"


def test_invalid_review_postcompletion_resource_stop_is_verifier_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _synthetic_awaiting_run(tmp_path, monkeypatch)
    key = runner._read_json(root / "review/private_key.json")["items"]
    answers = tmp_path / "invalid-postcompletion-resource-review.csv"
    runner._write_csv(
        answers,
        [
            {
                "blind_id": item["blind_id"],
                "recognizable": 2 if index == 0 else 1,
                "perceived_digit": item["requested_label"],
                "notes": "invalid answer before post-completion resource stop",
            }
            for index, item in enumerate(key)
        ],
    )
    original_complete = runner.ResourceGovernor.complete
    armed = {"value": True}

    def postcompletion_overrun(
        self: runner.ResourceGovernor,
        kind: str,
        *,
        terminal_override: bool = False,
    ) -> dict[str, object]:
        receipt = original_complete(self, kind, terminal_override=terminal_override)
        if kind == "human_review_terminalization" and not terminal_override and armed["value"]:
            armed["value"] = False
            receipt["storage_bytes_after"] = self.budget.max_storage_bytes + 1
            checks = {
                "wall": float(receipt["wall_seconds_after"]) <= self.budget.max_wall_seconds,
                "accelerator": float(receipt["accelerator_seconds_after"])
                <= self.budget.max_accelerator_seconds,
                "storage": False,
                "cuda": float(receipt["cuda_fraction"]) <= self.budget.max_cuda_fraction,
                "quantum": float(receipt["elapsed_seconds"])
                <= self.budget.maximum_quantum_seconds,
            }
            self.failed_admission = {
                "kind": kind,
                "phase": "post-completion",
                "checks": checks,
                "receipt": receipt,
                "passed": 0,
            }
            self.write()
            raise runner.ResourceStop(
                f"resource post-completion check failed for {kind}: {checks}"
            )
        return receipt

    monkeypatch.setattr(runner.ResourceGovernor, "complete", postcompletion_overrun)
    args = SimpleNamespace(
        run_dir=str(root),
        answers=str(answers),
        reviewer="synthetic-test-reviewer",
        confirm_manual_review=True,
    )
    with pytest.raises(runner.ResourceStop, match="post-completion.*human_review_terminalization"):
        runner.finalize_review(args)
    status = runner._read_json(root / "status.json")
    assert status["state"] == status["route"] == "resource_stopped"
    assert status["failed_stage"] == "human_review_terminalization"
    ledger = runner._read_json(root / "resource_ledger.json")
    assert ledger["open_events"] == []
    assert ledger["failed_admission"]["phase"] == "post-completion"
    attempts = runner._read_json(root / "review/submission_attempts.json")["attempts"]
    assert [attempt["passed"] for attempt in attempts] == [0]
    assert "review answer value" in attempts[0]["error"]
    assert runner._read_json(root / "gates.json")["candidate_human_fidelity"]["state"] == "pending"
    assert not (root / "review/human_review_answers.csv").exists()
    assert not (root / "outcome.json").exists()
    assert runner.verify_run(root)["state"] == "resource_stopped"


def test_human_review_postcompletion_cap_failure_retains_science_and_is_verifier_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _synthetic_awaiting_run(tmp_path, monkeypatch)
    key = runner._read_json(root / "review/private_key.json")["items"]
    answers = tmp_path / "postcompletion-resource-review.csv"
    runner._write_csv(
        answers,
        [
            {
                "blind_id": item["blind_id"],
                "recognizable": 1,
                "perceived_digit": item["requested_label"],
                "notes": "retain after completed scientific adoption",
            }
            for item in key
        ],
    )
    original_complete = runner.ResourceGovernor.complete
    armed = {"value": True}

    def postcompletion_overrun(
        self: runner.ResourceGovernor,
        kind: str,
        *,
        terminal_override: bool = False,
    ) -> dict[str, object]:
        receipt = original_complete(self, kind, terminal_override=terminal_override)
        if kind == "human_review_terminalization" and not terminal_override and armed["value"]:
            armed["value"] = False
            receipt["storage_bytes_after"] = self.budget.max_storage_bytes + 1
            checks = {
                "wall": float(receipt["wall_seconds_after"]) <= self.budget.max_wall_seconds,
                "accelerator": float(receipt["accelerator_seconds_after"])
                <= self.budget.max_accelerator_seconds,
                "storage": False,
                "cuda": float(receipt["cuda_fraction"]) <= self.budget.max_cuda_fraction,
                "quantum": float(receipt["elapsed_seconds"])
                <= self.budget.maximum_quantum_seconds,
            }
            self.failed_admission = {
                "kind": kind,
                "phase": "post-completion",
                "checks": checks,
                "receipt": receipt,
                "passed": 0,
            }
            self.write()
            raise runner.ResourceStop(
                f"resource post-completion check failed for {kind}: {checks}"
            )
        return receipt

    monkeypatch.setattr(runner.ResourceGovernor, "complete", postcompletion_overrun)
    args = SimpleNamespace(
        run_dir=str(root),
        answers=str(answers),
        reviewer="synthetic-test-reviewer",
        confirm_manual_review=True,
    )
    with pytest.raises(runner.ResourceStop, match="post-completion.*human_review_terminalization"):
        runner.finalize_review(args)

    status = runner._read_json(root / "status.json")
    assert status["state"] == status["route"] == "resource_stopped"
    assert status["failed_stage"] == "human_review_terminalization"
    ledger = runner._read_json(root / "resource_ledger.json")
    assert ledger["open_events"] == []
    assert ledger["failed_admission"]["phase"] == "post-completion"
    assert ledger["failed_admission"]["kind"] == "human_review_terminalization"
    for relative in (
        "review/human_review_answers.csv",
        "review/human_review.json",
        "review/human_review_by_row.json",
        "outcome.json",
    ):
        assert (root / relative).is_file()
    assert runner._read_json(root / "gates.json")["candidate_human_fidelity"]["state"] == "complete"
    assert runner._read_json(root / "stage_ledger.json")["events"][-1]["stage"] == "human_review_terminalization"
    assert runner.verify_run(root)["state"] == "resource_stopped"


def test_machine_and_human_terminalization_are_cumulatively_resource_governed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _synthetic_awaiting_run(tmp_path, monkeypatch, semantic_terminalize=False)
    governor = runner.ResourceGovernor.rehydrate(root, device="cpu")
    runner._terminalize(root, governor)
    machine_ledger = runner._read_json(root / "resource_ledger.json")
    assert [(event["event"], event["kind"]) for event in machine_ledger["events"]] == [
        ("admit", "machine_terminalization"),
        ("complete", "machine_terminalization"),
    ]
    assert machine_ledger["events"][0]["reserve_remaining_seconds"] == 0.0

    key = runner._read_json(root / "review/private_key.json")["items"]
    answers = tmp_path / "manual-review.csv"
    runner._write_csv(
        answers,
        [
            {
                "blind_id": item["blind_id"],
                "recognizable": 1,
                "perceived_digit": item["requested_label"],
                "notes": "synthetic test answer",
            }
            for item in key
        ],
    )
    args = SimpleNamespace(
        run_dir=str(root),
        answers=str(answers),
        reviewer="synthetic-test-reviewer",
        confirm_manual_review=True,
    )
    assert runner.finalize_review(args) == 0
    ledger = runner._read_json(root / "resource_ledger.json")
    assert [(event["event"], event["kind"]) for event in ledger["events"]] == [
        ("admit", "machine_terminalization"),
        ("complete", "machine_terminalization"),
        ("admit", "human_review_terminalization"),
        ("complete", "human_review_terminalization"),
    ]
    assert ledger["events"][2]["reserve_remaining_seconds"] == 0.0
    assert float(ledger["wall_seconds"]) >= float(machine_ledger["wall_seconds"])


def test_public_verifier_rejects_coordinated_terminal_semantic_tamper_after_reseal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _synthetic_awaiting_run(tmp_path, monkeypatch)
    comparison_path = root / "evaluation/ddpm_eulerian_minus_historical.json"
    comparison = runner._read_json(comparison_path)
    comparison["candidate_diversity_ratio"] = 100.0
    comparison["candidate_to_historical_diversity_ratio"] = 100.0
    runner._write_json(comparison_path, comparison)
    gates = runner._read_json(root / "gates.json")
    gates["candidate_diversity"]["passed"] = 1
    runner._write_json(root / "gates.json", gates)
    membership = runner._read_json(root / "review/membership.json")
    membership["count"] = 79
    membership["rows"]["ddpm_eulerian"] = 39
    runner._write_json(root / "review/membership.json", membership)
    runner.v3._atomic_bytes(
        root / "REPORT.md",
        (root / "REPORT.md").read_bytes().replace(
            b"This exploratory run can establish feasibility only",
            b"This confirmatory run establishes population superiority",
        ),
    )
    runner._manifest(root)

    before = {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns, runner.sha256_file(path))
        for path in root.rglob("*")
        if path.is_file()
    }
    with pytest.raises(runner.IntegrityFailure):
        runner.verify_run(root)
    after = {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns, runner.sha256_file(path))
        for path in root.rglob("*")
        if path.is_file()
    }
    assert before == after


def test_public_verifier_replays_gate_i3_from_sealed_provider_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _synthetic_awaiting_run(tmp_path, monkeypatch)
    raw_relative = "populations/ddpm_eulerian.npz"
    raw = runner._npz(root / raw_relative)
    telemetry = json.loads(str(raw["telemetry_scientific_json"]))
    telemetry[0]["poisson_divergence_residual_maximum"] = (
        runner.POISSON_RESIDUAL_MAXIMUM * 2.0
    )
    raw["telemetry_scientific_json"] = np.asarray(
        json.dumps(telemetry, sort_keys=True, separators=(",", ":"))
    )
    _refresh_row_scientific_digest(root, "ddpm_eulerian", raw)
    _refresh_population_seal_receipts(root, (raw_relative,))
    scoring = runner._read_json(root / "evaluation/SCORING_READY.json")
    scoring["population_seal_sha256"] = runner.sha256_file(
        root / "populations/POPULATIONS_SEALED.json"
    )
    runner._write_json(root / "evaluation/SCORING_READY.json", scoring)
    runner._manifest(root)

    with pytest.raises(runner.IntegrityFailure, match="numerical|Poisson|poisson|gate"):
        runner.verify_run(root)


def test_public_verifier_replays_human_answers_gate_route_and_report_after_coordinated_reseal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _synthetic_complete_run(tmp_path, monkeypatch)
    report_before = (root / "REPORT.md").read_text(encoding="utf-8")
    assert "candidate_diversity_classes_exceeding_historical" in report_before
    assert "candidate_diversity_class_supportive" in report_before
    seal_before = (root / "populations/POPULATIONS_SEALED.json").read_bytes()
    key_before = (root / "review/private_key.json").read_bytes()
    comparison = runner._read_json(root / "evaluation/ddpm_eulerian_minus_historical.json")
    comparison["candidate_diversity_ratio"] = 1.0
    comparison["candidate_to_historical_diversity_ratio"] = 3.0
    runner._write_json(root / "evaluation/ddpm_eulerian_minus_historical.json", comparison)
    gates = runner._read_json(root / "gates.json")
    gates["candidate_diversity"]["passed"] = 1
    runner._write_json(root / "gates.json", gates)
    outcome = runner._read_json(root / "outcome.json")
    outcome["route"] = "adapter_positive_freeze_replication"
    outcome["next_action"] = runner._route_action(outcome["route"])
    outcome["machine_comparison"] = comparison
    runner._write_json(root / "outcome.json", outcome)
    runner._status(root, "complete", route=outcome["route"])
    runner._write_reports(root, outcome)
    runner._manifest(root)

    with pytest.raises(runner.IntegrityFailure):
        runner.verify_run(root)
    assert (root / "populations/POPULATIONS_SEALED.json").read_bytes() == seal_before
    assert (root / "review/private_key.json").read_bytes() == key_before


def test_resource_projection_and_governor_fail_closed_without_open_event(tmp_path: Path) -> None:
    projection = runner.resource_projection(
        smoke_path_seconds=1.0,
        path_count=40,
        native_step_seconds=0.01,
        stored_bytes_per_path=2_000_000,
    )
    assert projection == {
        "projected_wall_seconds": 230.0,
        "projected_accelerator_seconds": 170.0,
        "projected_storage_bytes": 80_000_000,
        "fits_frozen_maxima": 1,
    }
    assert runner.resource_projection(smoke_path_seconds=30.0)["fits_frozen_maxima"] == 0

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    budget = runner.ResourceBudget(
        max_wall_seconds=120.0,
        max_accelerator_seconds=60.0,
        max_storage_bytes=1_000_000,
        max_cuda_fraction=0.5,
        reserve_seconds=60.0,
        maximum_quantum_seconds=60.0,
    )
    governor = runner.ResourceGovernor(run_dir, budget, device="cpu")
    with pytest.raises(runner.ResourceStop, match="resource admission failed"):
        governor.admit(
            "population",
            predicted_wall_seconds=61.0,
            predicted_accelerator_seconds=1.0,
            predicted_next_bytes=1,
        )
    ledger = runner._read_json(run_dir / "resource_ledger.json")
    assert ledger["failed_admission"]["passed"] == 0
    assert ledger["failed_admission"]["checks"]["wall"] == 0
    assert ledger["failed_admission"]["checks"]["quantum"] == 0
    assert ledger["open_events"] == []


def test_resource_observed_failure_closes_the_only_active_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    governor = runner.ResourceGovernor(run_dir, runner.ResourceBudget(), device="cpu")
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(ticks))
    governor.admit(
        "synthetic",
        predicted_wall_seconds=5.0,
        predicted_accelerator_seconds=0.0,
        predicted_next_bytes=0,
    )
    governor.close_open_as_failed()
    ledger = runner._read_json(run_dir / "resource_ledger.json")
    assert ledger["open_events"] == []
    assert [event["event"] for event in ledger["events"]] == ["admit", "failed-complete"]
    assert ledger["events"][-1]["elapsed_seconds"] == 2.5
    assert ledger["wall_seconds"] == 2.5


def test_resource_post_completion_overrun_is_durable_and_has_no_open_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    budget = runner.ResourceBudget(
        max_wall_seconds=120.0,
        max_accelerator_seconds=60.0,
        max_storage_bytes=1_000_000,
        max_cuda_fraction=0.5,
        reserve_seconds=30.0,
        maximum_quantum_seconds=5.0,
    )
    governor = runner.ResourceGovernor(run_dir, budget, device="cpu")
    ticks = iter([20.0, 26.0])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(ticks))
    governor.admit(
        "synthetic",
        predicted_wall_seconds=4.0,
        predicted_accelerator_seconds=0.0,
        predicted_next_bytes=0,
    )
    with pytest.raises(runner.ResourceStop, match="post-completion"):
        governor.complete("synthetic")
    ledger = runner._read_json(run_dir / "resource_ledger.json")
    assert ledger["open_events"] == []
    assert [event["event"] for event in ledger["events"]] == ["admit", "complete"]
    assert ledger["failed_admission"]["phase"] == "post-completion"
    assert ledger["failed_admission"]["checks"]["quantum"] == 0
    assert ledger["failed_admission"]["receipt"] == ledger["events"][-1]


def _resource_config(budget: runner.ResourceBudget) -> dict[str, object]:
    return {"execution_authority": dataclasses.asdict(budget)}


def test_resource_verifier_accepts_real_receipts_and_replays_cumulative_arithmetic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    budget = runner.ResourceBudget(
        max_wall_seconds=120.0,
        max_accelerator_seconds=60.0,
        max_storage_bytes=1_000_000,
        max_cuda_fraction=0.5,
        reserve_seconds=30.0,
        maximum_quantum_seconds=60.0,
    )
    ticks = iter([10.0, 11.25])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(ticks))
    governor = runner.ResourceGovernor(run_dir, budget, device="cpu")
    governor.admit(
        "machine_scoring",
        predicted_wall_seconds=10.0,
        predicted_accelerator_seconds=0.0,
        predicted_next_bytes=0,
    )
    governor.complete("machine_scoring")
    status = {"state": "awaiting_human_review"}
    verified = runner._verify_resource_ledger(run_dir, _resource_config(budget), status)
    assert verified["events"][0]["reserve_remaining_seconds"] == budget.reserve_seconds
    assert verified["wall_seconds"] == 1.25

    ledger = runner._read_json(run_dir / "resource_ledger.json")
    ledger["wall_seconds"] = -1.0
    runner._write_json(run_dir / "resource_ledger.json", ledger)
    with pytest.raises(runner.IntegrityFailure, match="resource totals"):
        runner._verify_resource_ledger(run_dir, _resource_config(budget), status)


def test_resource_verifier_replays_failed_admission_instead_of_trusting_saved_checks(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    budget = runner.ResourceBudget(
        max_wall_seconds=100.0,
        max_accelerator_seconds=60.0,
        max_storage_bytes=1_000_000,
        max_cuda_fraction=0.5,
        reserve_seconds=30.0,
        maximum_quantum_seconds=60.0,
    )
    governor = runner.ResourceGovernor(run_dir, budget, device="cpu")
    with pytest.raises(runner.ResourceStop):
        governor.admit(
            "machine_scoring",
            predicted_wall_seconds=80.0,
            predicted_accelerator_seconds=0.0,
            predicted_next_bytes=0,
        )
    status = {"state": "resource_stopped"}
    assert runner._verify_resource_ledger(run_dir, _resource_config(budget), status)["failed_admission"]["passed"] == 0

    ledger = runner._read_json(run_dir / "resource_ledger.json")
    ledger["failed_admission"]["checks"]["wall"] = True
    ledger["failed_admission"]["passed"] = 1
    runner._write_json(run_dir / "resource_ledger.json", ledger)
    with pytest.raises(runner.IntegrityFailure, match="failed admission|resource"):
        runner._verify_resource_ledger(run_dir, _resource_config(budget), status)


def test_durable_row_boundary_retains_exact_last_valid_state_png_and_telemetry(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "failure").mkdir(parents=True)
    governor = runner.ResourceGovernor(run_dir, runner.ResourceBudget(), device="cpu")
    state = torch.full((runner.PATH_COUNT, 784), 1.0 / 784.0, dtype=torch.float32)
    inventory = runner.build_path_inventory()
    callback = runner._durable_step_callback(
        run_dir,
        "teacher",
        governor,
        runner.OUTER_STEPS,
        initial_state=state.numpy(),
        path_ids=inventory["path_ids"],
        requested_labels=inventory["requested_labels"],
    )
    initial = runner._npz(run_dir / "failure/last_valid_teacher.npz")
    assert int(initial["completed_step"]) == 0
    np.testing.assert_array_equal(initial["state"], state.numpy())
    np.testing.assert_array_equal(initial["path_ids"], inventory["path_ids"])
    np.testing.assert_array_equal(initial["requested_labels"], inventory["requested_labels"])
    runner.v3._verify_sheet_pixels(
        run_dir / "failure/last_valid_teacher.png",
        runner.mass_to_uint8(state.numpy()),
        columns=10,
        scale=2,
        captions=[f"teacher:{index:03d}" for index in range(runner.PATH_COUNT)],
    )
    telemetry = [{"completed_step": step, "finite": 1} for step in range(1, 9)]
    callback(
        {
            "row": "teacher",
            "completed_step": 8,
            "state": state,
            "telemetry": telemetry,
            "path_ids": inventory["path_ids"].tolist(),
            "requested_labels": inventory["requested_labels"].tolist(),
            "crn_key_hashes": ["synthetic-key"],
        }
    )
    governor.close_open_as_failed()

    retained = runner._npz(run_dir / "failure/last_valid_teacher.npz")
    assert int(retained["completed_step"]) == 8
    np.testing.assert_array_equal(retained["state"], state.numpy())
    np.testing.assert_array_equal(retained["requested_labels"], inventory["requested_labels"])
    np.testing.assert_array_equal(retained["path_ids"], inventory["path_ids"])
    runner.v3._verify_sheet_pixels(
        run_dir / "failure/last_valid_teacher.png",
        runner.mass_to_uint8(state.numpy()),
        columns=10,
        scale=2,
        captions=[f"teacher:{index:03d}" for index in range(runner.PATH_COUNT)],
    )
    tail = runner._read_json(run_dir / "failure/telemetry_tail.json")
    assert tail["row"] == "teacher"
    assert tail["completed_step"] == 8
    assert tail["tail"] == telemetry[-4:]
    assert tail["crn_key_hashes"] == ["synthetic-key"]
    assert tail["path_ids"] == inventory["path_ids"].tolist()
    assert tail["requested_labels"] == inventory["requested_labels"].tolist()
    ledger = runner._read_json(run_dir / "resource_ledger.json")
    assert ledger["open_events"] == []
    assert [event["event"] for event in ledger["events"]] == [
        "admit",
        "complete",
        "admit",
        "failed-complete",
    ]


def test_failure_evidence_binds_current_state_rng_models_resources_and_authorities(tmp_path: Path) -> None:
    root, _ = _synthetic_population_run(tmp_path)
    (root / "failure").mkdir(exist_ok=True)
    starts = runner._npz(root / "inventory/start_bank.npz")
    governor = runner.ResourceGovernor(root, runner.ResourceBudget(), device="cpu")
    callback = runner._durable_step_callback(
        root,
        "teacher",
        governor,
        runner.OUTER_STEPS,
        initial_state=starts["masses"],
        path_ids=starts["path_ids"],
        requested_labels=starts["requested_labels"],
    )
    del callback
    governor.close_open_as_failed()
    error = runner.IntegrityFailure("synthetic controller failure")
    try:
        raise error
    except runner.IntegrityFailure as observed:
        runner._save_failure_evidence(root, observed, "teacher_population", governor)

    failure = runner._read_json(root / "failure.json")
    assert failure["error_type"] == "IntegrityFailure"
    assert failure["message"] == "synthetic controller failure"
    assert failure["failed_stage"] == "teacher_population"
    assert failure["populations_sealed"] == 1
    assert failure["last_valid_files"] == ["failure/last_valid_teacher.npz"]
    required_authorities = {
        "config.json",
        "source_bindings.json",
        "checkpoint_bindings.json",
        "inventory/STARTS_SEALED.json",
        "inventory/start_bank.npz",
        "inventory/ddpm_latent_bank.npz",
        "inventory/rng_contract.json",
    }
    assert set(failure["authority_receipts"]) == required_authorities
    for relative, receipt in failure["authority_receipts"].items():
        path = root / relative
        assert receipt == {"bytes": path.stat().st_size, "sha256": runner.sha256_file(path)}
    assert failure["resource_ledger_sha256"] == runner.sha256_file(root / "resource_ledger.json")
    last_valid = runner._npz(root / "failure/last_valid_teacher.npz")
    snapshot = runner._npz(root / "failure/controller_snapshot.npz")
    assert set(snapshot) == set(last_valid)
    for key in snapshot:
        np.testing.assert_array_equal(snapshot[key], last_valid[key])
    assert runner._read_json(root / "failure/model_state_identity.json") == runner._read_json(
        root / "telemetry/model_state_identity.json"
    )
    assert runner._read_json(root / "failure/resource_snapshot.json")["ledger"] == runner._read_json(
        root / "resource_ledger.json"
    )
    assert "synthetic controller failure" in (root / "failure/traceback.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("arguments", "expected"),
    (
        (
            dict(integrity_passed=True, native_valid=True, adapter_human_fidelity=True, adapter_diversity=True),
            "adapter_positive_freeze_replication",
        ),
        (
            dict(integrity_passed=True, native_valid=False, adapter_human_fidelity=True, adapter_diversity=True),
            "native_ddpm_control_invalid",
        ),
        (
            dict(integrity_passed=True, native_valid=True, adapter_human_fidelity=True, adapter_diversity=False),
            "adapter_fidelity_only_major_pivot_or_stop",
        ),
        (
            dict(integrity_passed=True, native_valid=True, adapter_human_fidelity=False, adapter_diversity=False, early_joint=True),
            "adapter_early_joint_horizon_replication",
        ),
        (
            dict(integrity_passed=True, native_valid=True, adapter_human_fidelity=False, adapter_diversity=True),
            "adapter_diverse_not_faithful_major_pivot_or_stop",
        ),
        (
            dict(
                integrity_passed=True,
                native_valid=True,
                adapter_human_fidelity=True,
                adapter_diversity=False,
                predicted_mass_diverse=True,
                eulerian_collapsed=True,
            ),
            "composition_mode_loss_theory_bridge_or_stop",
        ),
        (
            dict(
                integrity_passed=True,
                native_valid=True,
                adapter_human_fidelity=False,
                adapter_diversity=False,
                predicted_mass_diverse=True,
                eulerian_collapsed=True,
            ),
            "composition_mode_loss_theory_bridge_or_stop",
        ),
        (
            dict(integrity_passed=True, native_valid=True, adapter_human_fidelity=False, adapter_diversity=False),
            "off_policy_bridge_on_policy_or_stop",
        ),
        (
            dict(
                integrity_passed=True,
                native_valid=True,
                adapter_human_fidelity=False,
                adapter_diversity=False,
                predicted_mass_diverse=True,
                historical_early_joint=True,
            ),
            "historical_early_horizon_replication",
        ),
        (
            dict(
                integrity_passed=True,
                native_valid=True,
                adapter_human_fidelity=False,
                adapter_diversity=False,
                predicted_mass_diverse=True,
                learned_both_negative=True,
            ),
            "learned_eulerian_negative_stop_or_major_pivot",
        ),
        (
            dict(
                integrity_passed=True,
                native_valid=True,
                adapter_human_fidelity=False,
                adapter_diversity=False,
                predicted_mass_diverse=True,
            ),
            "unclassified_stop_redesign",
        ),
    ),
)
def test_outcome_route_truth_table_is_frozen(arguments: dict[str, bool], expected: str) -> None:
    assert runner._outcome_route(**arguments) == expected


def test_outcome_routing_refuses_scientific_interpretation_after_integrity_failure() -> None:
    with pytest.raises(runner.IntegrityFailure, match="integrity gate"):
        runner._outcome_route(
            integrity_passed=False,
            native_valid=True,
            adapter_human_fidelity=True,
            adapter_diversity=True,
        )


@pytest.mark.parametrize(
    ("comparison_overrides", "expected"),
    (
        (
            {
                "candidate_diversity_ratio": 0.3,
                "predicted_mass_diverse": 0,
                "historical_early_joint": 1,
            },
            "historical_early_horizon_replication",
        ),
        (
            {
                "candidate_diversity_ratio": 0.3,
                "predicted_mass_diverse": 0,
                "historical_early_joint": 0,
            },
            "learned_eulerian_negative_stop_or_major_pivot",
        ),
        (
            {
                "candidate_diversity_ratio": 0.1,
                "predicted_mass_diverse": 0,
                "historical_early_joint": 0,
            },
            "off_policy_bridge_on_policy_or_stop",
        ),
        (
            {
                "candidate_diversity_ratio": 0.1,
                "predicted_mass_diverse": 1,
                "historical_early_joint": 0,
            },
            "composition_mode_loss_theory_bridge_or_stop",
        ),
    ),
)
def test_persisted_gate_and_comparison_evidence_reaches_every_negative_scientific_branch(
    comparison_overrides: dict[str, float | int], expected: str
) -> None:
    gates = {
        "integrity": {"passed": 1},
        "native_ddpm": {"passed": 1},
        "candidate_diversity": {"passed": 0},
    }
    comparison: dict[str, object] = {
        "candidate_early_joint": 0,
        "historical_early_joint": 0,
        "candidate_diversity_ratio": 0.3,
        "predicted_mass_diverse": 0,
    }
    comparison.update(comparison_overrides)
    human = {"recognizability": 0.0, "requested_label_agreement": 0.0}
    inputs = runner._route_inputs(gates, comparison, human)
    assert runner._outcome_route(**inputs) == expected
    if expected == "historical_early_horizon_replication":
        assert inputs["historical_early_joint"] and not inputs["learned_both_negative"]
    if expected == "learned_eulerian_negative_stop_or_major_pivot":
        assert inputs["learned_both_negative"] and not inputs["historical_early_joint"]


def test_evaluator_terminal_arff_and_review_are_unreachable_before_population_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []

    def reject_seal(_run_dir: Path) -> dict[str, object]:
        opened.append("seal")
        raise runner.IntegrityFailure("synthetic unsealed tree")

    def forbidden(*_args: object, **_kwargs: object) -> object:
        opened.append("forbidden")
        raise AssertionError("terminal authority opened before the population seal")

    monkeypatch.setattr(runner, "_verify_population_seal", reject_seal)
    monkeypatch.setattr(runner, "_load_evaluator_after_seal", forbidden)
    monkeypatch.setattr(runner, "read_mnist_arff_slice", forbidden)
    with pytest.raises(runner.IntegrityFailure, match="unsealed"):
        runner.evaluate_sealed_populations(
            tmp_path,
            arff=tmp_path / "must-not-open.arff",
            ddpm_run_dir=tmp_path / "must-not-open-ddpm",
            device="cpu",
        )
    with pytest.raises(runner.IntegrityFailure, match="unsealed"):
        runner.prepare_blind_review(tmp_path)
    assert opened == ["seal", "seal"]


def test_cpu_smoke_cli_is_deterministic_cpu_only_and_leaves_no_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("CPU smoke opened an external or production authority")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "_bind_real_authorities_impl", forbidden)
    monkeypatch.setattr(runner, "read_mnist_development_prefix", forbidden)
    monkeypatch.setattr(runner, "_load_evaluator_after_seal", forbidden)
    first_code = runner.main(["smoke", "--device", "cpu"])
    first_bytes = capsys.readouterr().out.encode("utf-8")
    second_code = runner.main(["smoke", "--device", "cpu"])
    second_bytes = capsys.readouterr().out.encode("utf-8")
    assert first_code == second_code == 0
    assert first_bytes == second_bytes
    receipt = json.loads(first_bytes)
    assert receipt["schema"] == runner.VERSION + "-cpu-smoke"
    assert receipt["passed"] == 1
    assert receipt["path_count"] == 2
    assert receipt["outer_steps"] == 4
    assert receipt["external_authorities_opened"] == 0
    assert receipt["scientific_evidence"] == 0
    assert receipt["rows"] == list(runner.ROWS)
    for key in (
        "inventory_and_latent_passed",
        "adapter_provider_passed",
        "native_sampler_stub_passed",
        "crn_replay_exact",
        "population_seal_passed",
        "metric_stub_passed",
        "rendering_passed",
        "forced_failure_retained",
        "postseal_resume_no_generation_passed",
        "read_only_verification_passed",
    ):
        assert receipt[key] == 1
    assert 0.0 <= receipt["metric_stub_endpoint_uint8_mean"] <= 255.0
    assert len(receipt["scientific_digest"]) == 64
    assert list(tmp_path.rglob("*")) == []


def test_unsealed_observed_failure_is_terminal_retained_and_verifier_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "failed-run"
    (root / "failure").mkdir(parents=True)
    budget = runner.ResourceBudget()
    args = SimpleNamespace(
        resume_post_seal=False,
        run_dir=str(root),
        legacy_checkpoint="must-not-open.pt",
        ddpm_run_dir="must-not-open-ddpm",
        arff="must-not-open.arff",
        device="cpu",
        approval_id="synthetic-test-only",
    )
    config = runner._config(args, runner._repository_root(), budget)
    runner._write_json(root / "config.json", config)
    runner._write_json(root / "source_bindings.json", runner._source_bindings(runner._repository_root()))
    runner._write_json(
        root / "claim_boundary.json",
        {
            "schema": runner.VERSION + "-claim-boundary",
            "mode": runner.RESEARCH_MODE,
            "establishes_at_most": "synthetic test only",
            "does_not_establish": ["scientific evidence"],
        },
    )
    runner.v3._atomic_bytes(root / "command.txt", b"synthetic test-only command\n")
    runner._status(root, "running")
    governor = runner.ResourceGovernor(root, budget, device="cpu")
    governor.write()
    runner._write_json(root / "stage_ledger.json", {"schema": runner.VERSION + "-stage-ledger", "events": []})

    monkeypatch.setattr(runner, "_is_production_config", lambda _config: False)
    monkeypatch.setattr(runner, "_initialize", lambda _args: (root, governor, config))
    monkeypatch.setattr(
        runner,
        "_bind_real_authorities_impl",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(runner.IntegrityFailure("synthetic binding failure")),
    )
    assert runner.run_production(args) == 2
    status = runner._read_json(root / "status.json")
    failure = runner._read_json(root / "failure.json")
    assert status["state"] == status["route"] == "failed_unsealed"
    assert status["whole_run_restart_required"] == 1
    assert failure["error_type"] == "IntegrityFailure"
    assert failure["failed_stage"] == "binding_preflight"
    assert failure["populations_sealed"] == 0
    assert "synthetic binding failure" in (root / "failure/traceback.txt").read_text(encoding="utf-8")
    assert (root / "artifact_manifest.json").is_file()
    assert (root / "VERIFY_RECEIPT.json").is_file()
    assert runner.verify_run(root)["state"] == "failed_unsealed"


@pytest.mark.parametrize(
    ("failure", "expected_return"),
    (
        (runner.IntegrityFailure("authenticated legacy checkpoint is absent"), 2),
        (KeyboardInterrupt("operator interrupted authenticated binding"), 130),
        (SystemExit(23), None),
    ),
)
def test_production_binding_failure_before_preflight_is_terminal_verifier_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    expected_return: int | None,
) -> None:
    root = tmp_path / "production-shaped-binding-failure"
    (root / "failure").mkdir(parents=True)
    budget = runner.ResourceBudget()
    args = SimpleNamespace(
        resume_post_seal=False,
        run_dir=str(root),
        legacy_checkpoint="missing-authenticated-legacy.pt",
        ddpm_run_dir="missing-authenticated-ddpm-run",
        arff="missing-authenticated-mnist.arff",
        device="cuda:0",
        approval_id="approved-binding-failure-test",
    )
    config = runner._config(args, runner._repository_root(), budget)
    assert runner._is_production_config(config) is True
    runner._write_json(root / "config.json", config)
    runner._write_json(root / "source_bindings.json", runner._source_bindings(runner._repository_root()))
    runner._write_json(
        root / "claim_boundary.json",
        {
            "schema": runner.VERSION + "-claim-boundary",
            "mode": runner.RESEARCH_MODE,
            "establishes_at_most": "early authenticated binding failure only",
            "does_not_establish": ["scientific evidence"],
        },
    )
    runner.v3._atomic_bytes(root / "command.txt", (config["command"] + "\n").encode("utf-8"))
    runner._status(root, "running")
    governor = runner.ResourceGovernor(root, budget, device="cuda:0")
    governor.write()
    runner._write_json(root / "stage_ledger.json", {"schema": runner.VERSION + "-stage-ledger", "events": []})

    monkeypatch.setattr(runner, "_initialize", lambda _args: (root, governor, config))
    monkeypatch.setattr(
        runner,
        "_bind_real_authorities_impl",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )
    if expected_return is None:
        with pytest.raises(SystemExit) as observed:
            runner.run_production(args)
        assert observed.value.code == 23
    else:
        assert runner.run_production(args) == expected_return
    assert not (root / "preflight/cpu_smoke.json").exists()
    status = runner._read_json(root / "status.json")
    assert status["state"] == status["route"] == "failed_unsealed"
    assert status["failed_stage"] == "binding_preflight"
    assert runner._read_json(root / "failure.json")["error_type"] == type(failure).__name__
    receipt = runner.verify_run(root)
    assert receipt["passed"] == 1
    assert receipt["state"] == "failed_unsealed"


def test_postseal_resume_dispatch_never_calls_generation_or_mutates_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "sealed-run"
    (root / "populations").mkdir(parents=True)
    sentinel = root / "populations/frozen.bin"
    runner.v3._atomic_bytes(sentinel, b"sealed-scientific-population")
    before = sentinel.read_bytes()
    governor = runner.ResourceGovernor(root, runner.ResourceBudget(), device="cpu")
    governor.write()
    runner._write_json(root / "stage_ledger.json", {"schema": runner.VERSION + "-stage-ledger", "events": []})
    calls: list[str] = []

    monkeypatch.setattr(runner, "_initialize", lambda _args: (root, governor, {"schema": runner.VERSION + "-config"}))
    monkeypatch.setattr(runner, "evaluate_sealed_populations", lambda *_args, **_kwargs: calls.append("score") or {})
    monkeypatch.setattr(runner, "prepare_blind_review", lambda *_args, **_kwargs: calls.append("review") or {})
    monkeypatch.setattr(runner, "_record_stage", lambda _root, stage: calls.append(f"stage:{stage}"))
    monkeypatch.setattr(runner, "_terminalize", lambda _root, *_args: calls.append("terminalize") or {})
    monkeypatch.setattr(runner, "verify_run", lambda _root: calls.append("verify") or {"passed": 1})

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("post-seal resume invoked population generation")

    for name in (
        "_bind_real_authorities_impl",
        "read_mnist_development_prefix",
        "build_start_bank",
        "build_teacher_target_bank",
        "build_ddpm_latent_bank",
        "run_eulerian_row",
        "run_native_ddpm_row",
        "save_eulerian_population",
        "save_native_population",
        "seal_populations",
    ):
        monkeypatch.setattr(runner, name, forbidden)
    args = SimpleNamespace(
        resume_post_seal=True,
        legacy_checkpoint="must-not-open.pt",
        ddpm_run_dir="synthetic-ddpm",
        arff="synthetic.arff",
        device="cpu",
    )
    assert runner.run_production(args) == 0
    assert sentinel.read_bytes() == before
    assert calls == [
        "score",
        "stage:machine_scoring",
        "review",
        "stage:render_and_review_bundle",
        "stage:awaiting_human_review",
        "terminalize",
    ]
    status = runner._read_json(root / "status.json")
    assert status["state"] == status["route"] == "awaiting_human_review"
    assert status["whole_run_restart_required"] == 0
    ledger = runner._read_json(root / "resource_ledger.json")
    assert [(event["event"], event["kind"]) for event in ledger["events"]] == [
        ("admit", "machine_scoring"),
        ("complete", "machine_scoring"),
        ("admit", "render_and_review_bundle"),
        ("complete", "render_and_review_bundle"),
    ]
    assert ledger["events"][0]["reserve_remaining_seconds"] == runner.TERMINAL_RESERVE_SECONDS
    assert ledger["events"][2]["reserve_remaining_seconds"] == runner.TERMINAL_RESERVE_SECONDS


def test_actual_postseal_resume_preserves_failed_complete_history_and_verifies_awaiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "_is_production_config", lambda _config: False)
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "synchronize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner.torch.cuda, "max_memory_allocated", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        runner.torch.cuda,
        "get_device_properties",
        lambda *_args, **_kwargs: SimpleNamespace(total_memory=16 * 1024**3),
    )
    root, _ = _synthetic_population_run(tmp_path)
    for directory in ("failure", "evaluation", "review", "images"):
        (root / directory).mkdir(exist_ok=True)
    config = runner._read_json(root / "config.json")
    config["execution_authority"]["device"] = "cuda:0"
    config["execution_authority"]["approval_id"] = "approved-postseal-resume-test"
    config["input_paths"] = {
        "legacy_checkpoint": runner._read_json(root / "checkpoint_bindings.json")[
            "legacy_checkpoint"
        ]["path"],
        "ddpm_run_dir": str((root / "synthetic-input-authorities").resolve()),
        "arff": runner._read_json(root / "checkpoint_bindings.json")["arff"]["path"],
    }
    runner._write_json(root / "config.json", config)
    runner.v3._atomic_bytes(root / "command.txt", b"synthetic post-seal resume\n")
    runner._write_json(
        root / "claim_boundary.json",
        {
            "schema": runner.VERSION + "-claim-boundary",
            "mode": runner.RESEARCH_MODE,
            "establishes_at_most": "synthetic post-seal recovery only",
            "does_not_establish": ["scientific evidence"],
        },
    )
    starts_seal_path = root / "inventory/STARTS_SEALED.json"
    starts_seal = runner._read_json(starts_seal_path)
    starts_seal["config_sha256"] = runner.sha256_file(root / "config.json")
    runner._write_json(starts_seal_path, starts_seal)
    _refresh_population_seal_receipts(root, ("inventory/STARTS_SEALED.json",))
    population_seal_path = root / "populations/POPULATIONS_SEALED.json"
    population_seal = runner._read_json(population_seal_path)
    population_seal["starts_seal_sha256"] = runner.sha256_file(starts_seal_path)
    unsigned = {key: value for key, value in population_seal.items() if key != "seal_sha256"}
    population_seal["seal_sha256"] = runner._sha256_bytes(
        runner._canonical_json_bytes(unsigned)
    )
    runner._write_json(population_seal_path, population_seal)
    for stage in runner.STAGE_ORDER[: runner.STAGE_ORDER.index("population_seal") + 1]:
        runner._record_stage(root, stage)

    budget = runner.ResourceBudget()
    governor = runner.ResourceGovernor(root, budget, device="cuda:0")
    governor.write()
    governor.admit(
        "machine_scoring",
        predicted_wall_seconds=1.0,
        predicted_accelerator_seconds=1.0,
        predicted_next_bytes=1,
    )
    governor.close_open_as_failed()
    try:
        raise OSError("synthetic scoring interruption after population seal")
    except OSError as error:
        runner._save_failure_evidence(root, error, "machine_scoring", governor)
    runner._status(
        root,
        "postseal_interrupted",
        route="postseal_interrupted",
        error="synthetic scoring interruption after population seal",
        failed_stage="machine_scoring",
        whole_run_restart_required=False,
    )
    runner._terminalize(root, governor)
    population_before = {
        path.name: path.read_bytes() for path in (root / "populations").iterdir() if path.is_file()
    }

    monkeypatch.setattr(
        runner,
        "evaluate_sealed_populations",
        lambda run_dir, **_kwargs: _write_synthetic_evaluation(Path(run_dir)),
    )
    args = SimpleNamespace(
        run_dir=str(root),
        resume_post_seal=True,
        device="cuda:0",
        approval_id="approved-postseal-resume-test",
        max_wall_seconds=runner.MAX_WALL_SECONDS,
        max_accelerator_seconds=runner.MAX_ACCELERATOR_SECONDS,
        max_storage_mib=runner.MAX_STORAGE_MIB,
        max_cuda_fraction=runner.MAX_CUDA_FRACTION,
        legacy_checkpoint=config["input_paths"]["legacy_checkpoint"],
        ddpm_run_dir=config["input_paths"]["ddpm_run_dir"],
        arff=config["input_paths"]["arff"],
    )
    assert runner.run_production(args) == 0
    assert runner._read_json(root / "status.json")["state"] == "awaiting_human_review"
    assert runner.verify_run(root)["state"] == "awaiting_human_review"
    ledger = runner._read_json(root / "resource_ledger.json")
    assert ("failed-complete", "machine_scoring") in [
        (event["event"], event["kind"]) for event in ledger["events"]
    ]
    assert ledger["open_events"] == []
    assert {
        path.name: path.read_bytes() for path in (root / "populations").iterdir() if path.is_file()
    } == population_before


def test_postseal_resume_replays_semantics_before_deleting_any_derived_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _synthetic_awaiting_run(tmp_path, monkeypatch)
    config = runner._read_json(root / "config.json")
    config["execution_authority"]["device"] = "cuda:0"
    config["input_paths"] = {
        "legacy_checkpoint": runner._read_json(root / "checkpoint_bindings.json")["legacy_checkpoint"]["path"],
        "ddpm_run_dir": str((root / "synthetic-input-authorities").resolve()),
        "arff": runner._read_json(root / "checkpoint_bindings.json")["arff"]["path"],
    }
    runner._write_json(root / "config.json", config)
    starts_seal_path = root / "inventory/STARTS_SEALED.json"
    starts_seal = runner._read_json(starts_seal_path)
    starts_seal["config_sha256"] = runner.sha256_file(root / "config.json")
    runner._write_json(starts_seal_path, starts_seal)
    _refresh_population_seal_receipts(root, ("inventory/STARTS_SEALED.json",))
    population_seal_path = root / "populations/POPULATIONS_SEALED.json"
    population_seal = runner._read_json(population_seal_path)
    population_seal["starts_seal_sha256"] = runner.sha256_file(starts_seal_path)
    unsigned = {key: value for key, value in population_seal.items() if key != "seal_sha256"}
    population_seal["seal_sha256"] = runner._sha256_bytes(runner._canonical_json_bytes(unsigned))
    runner._write_json(population_seal_path, population_seal)
    runner._status(root, "postseal_interrupted", route="postseal_interrupted", error="synthetic crash")
    ledger = runner._read_json(root / "resource_ledger.json")
    ledger["wall_seconds"] = -1.0
    runner._write_json(root / "resource_ledger.json", ledger)
    runner._write_reports(root)
    runner._manifest(root)
    before = {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns, runner.sha256_file(path))
        for path in root.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    args = SimpleNamespace(
        run_dir=str(root),
        resume_post_seal=True,
        device="cuda:0",
        approval_id="approved-resume-test",
        max_wall_seconds=runner.MAX_WALL_SECONDS,
        max_accelerator_seconds=runner.MAX_ACCELERATOR_SECONDS,
        max_storage_mib=runner.MAX_STORAGE_MIB,
        max_cuda_fraction=runner.MAX_CUDA_FRACTION,
        legacy_checkpoint=config["input_paths"]["legacy_checkpoint"],
        ddpm_run_dir=config["input_paths"]["ddpm_run_dir"],
        arff=config["input_paths"]["arff"],
    )
    with pytest.raises(runner.IntegrityFailure, match="resource|semantic|ledger|total"):
        runner._initialize(args)
    after = {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns, runner.sha256_file(path))
        for path in root.rglob("*")
        if path.is_file()
    }
    assert before == after


@pytest.mark.parametrize("state", ["awaiting_human_review", "complete"])
def test_terminal_sealed_run_cannot_be_reopened_and_have_review_evidence_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    root = tmp_path / state
    (root / "populations").mkdir(parents=True)
    (root / "review").mkdir()
    runner._write_json(root / "populations/POPULATIONS_SEALED.json", {"synthetic": 1})
    runner._write_json(root / "config.json", {"schema": runner.VERSION + "-config"})
    runner._status(root, state, route=state)
    evidence = root / "review/human_review_answers.csv"
    runner.v3._atomic_bytes(evidence, b"immutable-review-evidence\n")
    before = evidence.read_bytes()
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner, "_verify_population_seal", lambda _root: {"passed": 1})
    args = SimpleNamespace(
        run_dir=str(root),
        resume_post_seal=True,
        device="cuda:0",
        approval_id="approved-resume-test",
        max_wall_seconds=runner.MAX_WALL_SECONDS,
        max_accelerator_seconds=runner.MAX_ACCELERATOR_SECONDS,
        max_storage_mib=runner.MAX_STORAGE_MIB,
        max_cuda_fraction=runner.MAX_CUDA_FRACTION,
    )
    with pytest.raises(runner.IntegrityFailure, match="state|terminal|resume"):
        runner._initialize(args)
    assert evidence.read_bytes() == before


def test_cli_exposes_only_execution_authorities_and_no_scientific_knobs() -> None:
    parser = runner.build_parser()
    subparsers = next(action for action in parser._actions if getattr(action, "choices", None))
    assert set(subparsers.choices) == {"smoke", "run", "verify", "finalize-review"}
    run_parser = subparsers.choices["run"]
    run_flags = {flag for action in run_parser._actions for flag in action.option_strings}
    assert run_flags == {
        "-h",
        "--help",
        "--run-dir",
        "--legacy-checkpoint",
        "--ddpm-run-dir",
        "--arff",
        "--device",
        "--approval-id",
        "--max-wall-seconds",
        "--max-accelerator-seconds",
        "--max-cuda-fraction",
        "--max-storage-mib",
        "--resume-post-seal",
    }
    forbidden = ("gain", "path-count", "seed", "anchor", "time-map", "threshold", "selector", "candidate")
    assert not any(token in flag for flag in run_flags for token in forbidden)
    smoke_flags = {flag for action in subparsers.choices["smoke"]._actions for flag in action.option_strings}
    assert smoke_flags == {"-h", "--help", "--device"}
    with pytest.raises(SystemExit):
        parser.parse_args(["smoke", "--device", "cuda:0"])


def test_production_verifier_has_no_serialized_config_opt_out_for_terminal_replay() -> None:
    args = SimpleNamespace(
        approval_id="approved-production-test",
        device="cuda:0",
        legacy_checkpoint="legacy.pt",
        ddpm_run_dir="ddpm-run",
        arff="mnist.arff",
    )
    config = runner._config(args, runner._repository_root(), runner.ResourceBudget())
    assert runner._is_production_config(config) is True
    variants = []
    for missing in ("created_at", "command", "argv", "input_paths"):
        changed = dict(config)
        changed.pop(missing)
        variants.append(changed)
    wrong_device = json.loads(json.dumps(config))
    wrong_device["execution_authority"]["device"] = "cpu"
    variants.append(wrong_device)
    bad_approval = json.loads(json.dumps(config))
    bad_approval["execution_authority"]["approval_id"] = "placeholder"
    variants.append(bad_approval)
    for changed in variants:
        with pytest.raises(runner.IntegrityFailure, match="production|config|authority"):
            runner._is_production_config(changed)

    source = inspect.getsource(runner._verify_evaluation)
    assert "_is_production_config(config)" in source
    assert "read_mnist_arff_slice" in source
    assert "_load_evaluator_after_seal" in source
    assert 'config["execution_authority"]["device"]' in source
    seal_source = inspect.getsource(runner._verify_population_seal)
    assert 'bound_ddpm["model_state_sha256"]' in seal_source
    assert "v3._load_clean_model" in seal_source
    assert "v3._model_state_semantic_digest" in seal_source


def test_experiment_note_contains_every_load_bearing_contract() -> None:
    note = (Path(__file__).resolve().parents[1] / "docs" / "ddpm_eulerian_diversity_pilot.md").read_text(encoding="utf-8")
    required = (
        "exploratory, objective-bearing",
        "Proxy-only patches since the last objective-bearing experiment: 0",
        "d2e-v1-000",
        "0xE1600001",
        "0xE160F001",
        "0x1.8f8b8b8b8b8b9p+6",
        "31c8b3660317954d9cc507e4894891804b379bc1c70aa526f4101f93b4a45fed",
        "f5310576d5c1a36b1889c4e4cc959a5edc8a09921e818a6a1134edb46838ff2a",
        "edge-noise-v1",
        "Scoped deviation from Plan section 8.2",
        "4dca1c40f25eb04b3d615bd0094891c7cedb8cea8a673607eb02e1ab977e4f19",
        "J_{\\mathrm{ctrl}}",
        "failed_unsealed",
        "awaiting_human_review",
        "adapter_positive_freeze_replication",
        "learned_eulerian_negative_stop_or_major_pivot",
        "There are **no confirmatory-claim gates**",
        "Scoped deviation from the plan's universal NPZ preamble",
        "not independently self-certifying containers",
        "exact 80 unique `blind_id` values",
        "even when `recognizable=0`",
        "outside the immutable run directory",
        "Blind-human metrics are endpoint-only at step `256`",
        "hard cap: 60 minutes",
        "hard cap: 30 minutes",
        "hard cap: 256 MiB",
        "cannot establish confirmatory generator quality",
        "stopped before population rollout",
    )
    for phrase in required:
        assert phrase in note
