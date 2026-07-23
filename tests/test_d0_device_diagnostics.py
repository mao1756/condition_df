"""Device-side diagnostic accumulation contracts for the D0 reference kernels."""

from __future__ import annotations

import pytest
import torch

import mnist.experiment12_d0 as d0_module
from mnist.eulerian_flux_mnist import DirectFluxMNISTConfig, masked_reference_free_step_torch


def _dynamics() -> DirectFluxMNISTConfig:
    return DirectFluxMNISTConfig(
        grid_size=4,
        num_steps=1,
        source_lowfreq_size=2,
        source_blur_sigma=0.0,
        ot_lowres_size=2,
        ot_blur_sigma=0.0,
        limiter_fraction=1.0,
        edge_alpha_mode="alpha_eff",
        alpha_eff=1.0,
        mass_floor=1e-7,
    )


def _states() -> torch.Tensor:
    values = torch.arange(1, 33, dtype=torch.float64).reshape(2, 16)
    return values / values.sum(dim=1, keepdim=True)


def test_forward_device_diagnostics_match_host_totals_without_scalar_sync_contract() -> None:
    states = _states()
    dynamics = _dynamics()
    torch.manual_seed(917)
    host = masked_reference_free_step_torch(
        states,
        2e-4,
        dynamics,
        free_weight=0.8,
        noise_weight=0.6,
        substeps=3,
        collect_diagnostics=True,
    )
    torch.manual_seed(917)
    device = masked_reference_free_step_torch(
        states,
        2e-4,
        dynamics,
        free_weight=0.8,
        noise_weight=0.6,
        substeps=3,
        collect_diagnostics=True,
        diagnostics_device=True,
    )

    assert torch.equal(host.states, device.states)
    assert device.device_diagnostics is not None
    diagnostics = device.device_diagnostics
    assert all(value.ndim == 0 and value.device == states.device for value in diagnostics.values())
    expected = {
        "limited_edges": host.limited_edges,
        "proposed_edges": host.proposed_edges,
        "drift_limited_edges": host.drift_limited_edges,
        "noise_limited_edges": host.noise_limited_edges,
        "nonfinite_edges": host.nonfinite_edges,
        "floor_touched_pixels": host.floor_touched_pixels,
        "floor_proposed_pixels": states.numel() * 3,
        "floor_correction_l1": host.floor_correction_l1,
        "renorm_correction_l1": host.renorm_correction_l1,
        "mobility_weight_sum": host.mobility_weight_sum,
        "limited_mobility_weight_sum": host.limited_mobility_weight_sum,
        "noise_energy_sum": host.noise_energy_sum,
        "limited_noise_energy_sum": host.limited_noise_energy_sum,
    }
    for key, value in expected.items():
        assert float(diagnostics[key].cpu()) == pytest.approx(float(value), rel=1e-12, abs=1e-12)
    assert float(diagnostics["max_simplex_mass_error"].cpu()) <= 1e-14


def test_forward_device_diagnostics_require_collection() -> None:
    with pytest.raises(ValueError, match="requires collect_diagnostics"):
        masked_reference_free_step_torch(
            _states(),
            1e-4,
            _dynamics(),
            collect_diagnostics=False,
            diagnostics_device=True,
        )


def test_direct_device_diagnostics_match_host_values_and_states() -> None:
    states = _states()
    dynamics = _dynamics()
    learned = torch.zeros((2, 2, 4, 4), dtype=states.dtype)
    normal = torch.Generator().manual_seed(441)
    standard_normal = torch.randn(learned.shape, dtype=learned.dtype, generator=normal)

    host = d0_module._direct_doob_reverse_substep(
        states,
        learned,
        rate=0.7,
        dt=2e-4,
        dynamics_config=dynamics,
        standard_normal=standard_normal,
    )
    device = d0_module._direct_doob_reverse_substep(
        states,
        learned,
        rate=0.7,
        dt=2e-4,
        dynamics_config=dynamics,
        standard_normal=standard_normal,
        diagnostics_device=True,
    )

    assert torch.equal(host.states, device.states)
    assert host.diagnostics.keys() == device.diagnostics.keys()
    assert all(isinstance(value, torch.Tensor) for value in device.diagnostics.values())
    assert all(value.ndim == 0 and value.device == states.device for value in device.diagnostics.values())
    for key, expected in host.diagnostics.items():
        assert float(device.diagnostics[key].cpu()) == pytest.approx(float(expected), rel=1e-12, abs=1e-12)


def test_direct_device_mode_shares_one_mobility_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = d0_module.harmonic_mobility_channels
    calls = 0

    def counted(states, config):
        nonlocal calls
        calls += 1
        return original(states, config)

    monkeypatch.setattr(d0_module, "harmonic_mobility_channels", counted)
    states = _states()
    learned = torch.zeros((2, 2, 4, 4), dtype=states.dtype)
    d0_module._direct_doob_reverse_substep(
        states,
        learned,
        rate=0.7,
        dt=2e-4,
        dynamics_config=_dynamics(),
        standard_normal=torch.zeros_like(learned),
        diagnostics_device=True,
    )
    assert calls == 1
