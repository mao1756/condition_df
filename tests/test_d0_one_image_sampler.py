from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mnist.d0_one_image_sampler as sampler
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig
from mnist.experiment12_d0 import Experiment12D0Config


def _dynamics() -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=4,
        num_steps=2,
        source_lowfreq_size=2,
        source_blur_sigma=0.0,
        source_uniform_mix=0.1,
        ot_lowres_size=2,
        ot_blur_sigma=0.0,
        condition_on_source=False,
        flux_parameterization="edge",
        limiter_fraction=1.0,
        edge_alpha_mode="alpha_eff",
        alpha_eff=1.0,
        mass_floor=1e-8,
    )


def _d0_config() -> Experiment12D0Config:
    return Experiment12D0Config(
        cache_build_mode="substep",
        teacher_stride_substeps=1,
        d0_target_space="doob-physical-residual",
        physical_target_normalization="global-rms",
        physical_target_scale=1.0,
        physical_loss_mask="all",
        physical_sampler_noise_mode="reference",
        eta_l2_weight=0.0,
        state_delta_loss_weight=0.0,
        rollout_loss_weight=0.0,
        trajectory_rollout_loss_weight=0.0,
        invalid_output_l2_weight=0.0,
        curl_loss_weight=0.0,
        edge_laplacian_loss_weight=0.0,
        control_output_clip=0.0,
        sample_steps=2,
        reference_substeps=2,
        tau_eff=0.1,
    )


class _CountingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        edge = torch.zeros(1, 2, 4, 4)
        edge[0, 0, 0, 0] = 0.02
        edge[0, 1, 1, 2] = -0.01
        self.register_buffer("edge", edge)
        self.calls = 0

    def forward(
        self,
        tau: torch.Tensor,
        states: torch.Tensor,
        labels: torch.Tensor,
        source: torch.Tensor | None,
    ) -> torch.Tensor:
        del tau, labels, source
        self.calls += 1
        return self.edge.expand(states.shape[0], -1, -1, -1)


def _terminals(count: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(19)
    states = rng.dirichlet(np.ones(16), size=count).astype(np.float32)
    labels = np.full(count, 3, dtype=np.int64)
    return states, labels


def _target() -> np.ndarray:
    value = np.arange(1, 17, dtype=np.float32)
    return value / value.sum()


def _run_seed(
    path: Path,
    *,
    model: torch.nn.Module | None = None,
    fingerprints: dict[str, str] | None = None,
    resume: bool = True,
    stop_after_outer_steps: int | None = None,
    terminal_indices: tuple[int, ...] = (0, 1, 2),
    show_progress: bool = False,
    device: torch.device = torch.device("cpu"),
    start_substep: int | None = None,
) -> sampler.PairedSeedSamplingResult:
    states, labels = _terminals()
    chosen_model = _CountingModel() if model is None else model
    chosen_model = chosen_model.to(device)
    return sampler.run_paired_seed_sampling(
        chosen_model,
        terminal_states=states,
        terminal_labels=labels,
        terminal_indices=terminal_indices,
        mixed_target=_target(),
        unmixed_target=np.roll(_target(), 1),
        dynamics_config=_dynamics(),
        d0_config=_d0_config(),
        rate_schedule=np.asarray([0.5, 0.75]),
        horizon=0.1,
        physical_target_scale=1.0,
        eval_seed=260719,
        device=device,
        checkpoint_path=path,
        fingerprints={"model": "abc", "cache": "def"} if fingerprints is None else fingerprints,
        sampler_config=sampler.PairedSamplerConfig(
            sample_batch_size=2,
            checkpoint_every_outer_steps=1,
            show_progress=show_progress,
        ),
        resume=resume,
        start_substep=start_substep,
        stop_after_outer_steps=stop_after_outer_steps,
    )


def test_terminal_assignments_are_persisted_disjoint_and_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "path_split.json"
    first = sampler.resolve_or_create_terminal_assignments(
        path,
        validation_terminal_indices=range(10, 26),
        eval_seeds=(260719, 260720),
        samples_per_seed=8,
    )
    second = sampler.resolve_or_create_terminal_assignments(
        path,
        validation_terminal_indices=range(10, 26),
        eval_seeds=(260719, 260720),
        samples_per_seed=8,
    )

    assert first == second
    assert set(first[260719]).isdisjoint(first[260720])
    assert set(first[260719]) | set(first[260720]) == set(range(10, 26))
    payload = json.loads(path.read_text(encoding="utf-8"))
    left = payload["assignments"]["260719"]
    right = payload["assignments"]["260720"]
    left[0], right[0] = right[0], left[0]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="deterministic selection"):
        sampler.resolve_or_create_terminal_assignments(
            path,
            validation_terminal_indices=range(10, 26),
            eval_seeds=(260719, 260720),
            samples_per_seed=8,
        )
    with pytest.raises(ValueError, match="does not match"):
        sampler.resolve_or_create_terminal_assignments(
            path,
            validation_terminal_indices=range(9, 25),
            eval_seeds=(260719, 260720),
            samples_per_seed=8,
        )


def test_strength_zero_bypass_matches_ordinary_zero_scalar_path() -> None:
    model = _CountingModel()
    states, labels = _terminals(2)
    states_t = torch.as_tensor(states)
    result = sampler.verify_strength_zero_bypass_equivalence(
        model,
        states=states_t,
        tau=torch.zeros(2),
        labels=torch.as_tensor(labels),
        rate=0.5,
        dt=0.01,
        dynamics_config=_dynamics(),
        physical_target_scale=2.0,
        standard_normal=torch.randn(2, 2, 4, 4, generator=torch.Generator().manual_seed(4)),
    )

    assert result["pass"] == 1
    assert result["max_state_error"] == 0.0
    assert model.calls == 1  # The bypass arm itself did not invoke the model.


def test_paired_arms_share_normals_and_zero_arm_bypasses_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[int, list[torch.Tensor]] = {0: [], 1: []}

    def fake_step(
        states: torch.Tensor,
        learned_delta: torch.Tensor,
        *,
        standard_normal: torch.Tensor,
        **kwargs: object,
    ) -> SimpleNamespace:
        del kwargs
        arm = int(bool(torch.count_nonzero(learned_delta)))
        captured[arm].append(standard_normal.detach().clone())
        zero = torch.zeros_like(learned_delta)
        scalar = states.new_zeros((), dtype=torch.float64)
        proposed = states.new_tensor(float(learned_delta.numel()), dtype=torch.float64)
        diagnostics = {
            "limited_edges": scalar,
            "proposed_edges": proposed,
            "nonfinite_edges": scalar,
            "mobility_weight_sum": proposed,
            "limited_mobility_weight_sum": scalar,
            "noise_energy_sum": proposed,
            "limited_noise_energy_sum": scalar,
            "floor_touched_pixels": scalar,
            "floor_proposed_pixels": states.new_tensor(float(states.numel()), dtype=torch.float64),
            "floor_correction_l1": scalar,
            "renorm_correction_l1": scalar,
            "max_simplex_mass_error": scalar,
        }
        return SimpleNamespace(
            states=states,
            free_delta=zero,
            learned_delta=learned_delta,
            noise_delta=standard_normal,
            diagnostics=diagnostics,
        )

    monkeypatch.setattr(sampler, "_direct_doob_reverse_substep", fake_step)
    model = _CountingModel()
    result = _run_seed(tmp_path / "paired.pt", model=model, terminal_indices=(0, 1))

    assert result.complete
    assert len(captured[0]) == len(captured[1]) == 4
    assert all(torch.equal(left, right) for left, right in zip(captured[0], captured[1]))
    assert model.calls == 4  # Four strength-one substeps; zero invokes none.
    assert result.arm_summaries["0"]["learned_step_rms"] == 0.0
    assert result.arm_summaries["1"]["learned_step_rms"] > 0.0


def test_exact_resume_matches_uninterrupted_sampling(tmp_path: Path) -> None:
    full = _run_seed(tmp_path / "full.pt")
    interrupted = _run_seed(tmp_path / "resume.pt", stop_after_outer_steps=3)
    assert not interrupted.complete
    resumed = _run_seed(tmp_path / "resume.pt", show_progress=True)

    assert resumed.complete
    assert np.array_equal(resumed.samples_strength0, full.samples_strength0)
    assert np.array_equal(resumed.samples_strength1, full.samples_strength1)
    assert resumed.per_sample_metrics == full.per_sample_metrics
    assert resumed.arm_summaries == full.arm_summaries
    assert resumed.time_bin_metrics == full.time_bin_metrics
    checkpoint = sampler._load_checkpoint(tmp_path / "resume.pt")
    backend = checkpoint["runtime_manifest"]["runtime"]["exact_backend"]
    assert backend["deterministic_algorithms"] is True
    assert backend["cudnn_benchmark"] is False
    assert backend["cuda_matmul_allow_tf32"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_sampler_runs_with_the_exact_backend_contract(tmp_path: Path) -> None:
    result = _run_seed(
        tmp_path / "cuda.pt",
        terminal_indices=(0, 1),
        device=torch.device("cuda"),
        start_substep=1,
    )
    assert result.complete
    assert np.isfinite(result.samples_strength0).all()
    assert np.isfinite(result.samples_strength1).all()
    checkpoint = sampler._load_checkpoint(tmp_path / "cuda.pt")
    backend = checkpoint["runtime_manifest"]["runtime"]["exact_backend"]
    assert backend["nvidia_driver_versions"]
    assert backend["cuda_matmul_allow_fp16_reduced_precision_reduction"] is False


def test_resume_rejects_fingerprint_and_sampling_configuration_changes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "resume.pt"
    partial = _run_seed(checkpoint, stop_after_outer_steps=1)
    assert not partial.complete

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _run_seed(checkpoint, fingerprints={"model": "changed", "cache": "def"})

    states, labels = _terminals()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        sampler.run_paired_seed_sampling(
            _CountingModel(),
            terminal_states=states,
            terminal_labels=labels,
            terminal_indices=(0, 1, 2),
            mixed_target=_target(),
            unmixed_target=np.roll(_target(), 1),
            dynamics_config=_dynamics(),
            d0_config=_d0_config(),
            rate_schedule=np.asarray([0.5, 0.75]),
            horizon=0.1,
            physical_target_scale=1.0,
            eval_seed=260719,
            device=torch.device("cpu"),
            checkpoint_path=checkpoint,
            fingerprints={"model": "abc", "cache": "def"},
            sampler_config=sampler.PairedSamplerConfig(
                sample_batch_size=3,
                checkpoint_every_outer_steps=1,
                show_progress=False,
            ),
        )


def test_batched_multi_seed_writer_preserves_requested_order_and_artifacts(tmp_path: Path) -> None:
    states, labels = _terminals()
    result = sampler.run_paired_d0_sampling(
        _CountingModel(),
        terminal_states=states,
        terminal_labels=labels,
        terminal_assignments={260719: (4, 1, 3), 260720: (0, 5)},
        mixed_target=_target(),
        unmixed_target=None,
        dynamics_config=_dynamics(),
        d0_config=_d0_config(),
        rate_schedule=np.asarray([0.5, 0.75]),
        horizon=0.1,
        physical_target_scale=1.0,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        fingerprints={"model": "abc", "cache": "def"},
        sampler_config=sampler.PairedSamplerConfig(
            sample_batch_size=2,
            checkpoint_every_outer_steps=2,
            show_progress=False,
        ),
    )

    assert result.complete
    assert result.terminal_indices.tolist() == [4, 1, 3, 0, 5]
    assert result.eval_seeds.tolist() == [260719, 260719, 260719, 260720, 260720]
    assert len(result.per_sample_metrics) == 5
    assert len(result.time_bin_metrics) == 2 * 2 * 5
    assert (tmp_path / "paired_samples.npz").exists()
    assert (tmp_path / "per_sample_metrics.csv").exists()
    assert (tmp_path / "sampling_time_bins.csv").exists()
    assert (tmp_path / "sampler_summary.json").exists()
