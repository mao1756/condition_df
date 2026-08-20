from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import torch

from mnist.diag_d0_tiny_ddpm_one_image import (
    ZeroEpsilon,
    _prior_panel,
    _smoke_config,
    run,
)
from mnist.pixel_ddpm import make_linear_ddpm_schedule
from mnist.tiny_ddpm import TINY_DDPM_PARAMETER_COUNT, TinyClassConditionalUNet28


def test_tiny_unet_parameter_shape_gradient_conditioning_and_roundtrip(tmp_path) -> None:
    torch.manual_seed(41)
    model = TinyClassConditionalUNet28()
    assert sum(parameter.numel() for parameter in model.parameters()) == 29_913
    assert TINY_DDPM_PARAMETER_COUNT == 29_913

    images = torch.randn(2, 1, 28, 28, requires_grad=True)
    timesteps = torch.tensor([0, 999])
    labels = torch.tensor([3, 3])
    prediction = model(images, timesteps, labels)
    assert prediction.shape == images.shape
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)

    with torch.no_grad():
        first = model(images[:1].detach(), torch.tensor([400]), torch.tensor([3]))
        changed_time = model(
            images[:1].detach(), torch.tensor([401]), torch.tensor([3])
        )
        changed_label = model(
            images[:1].detach(), torch.tensor([400]), torch.tensor([8])
        )
    assert not torch.equal(first, changed_time)
    assert not torch.equal(first, changed_label)

    checkpoint = tmp_path / "tiny.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = TinyClassConditionalUNet28()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    for original, loaded in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(original, loaded)


def test_cpu_smoke_writes_controls_failed_images_and_manifest(tmp_path) -> None:
    args = argparse.Namespace(
        device="cpu",
        data_root=tmp_path / "unused-data",
        runs_root=tmp_path / "runs",
        run_name="test-smoke",
        smoke=True,
        no_progress=True,
    )
    run_dir = run(args)
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "complete"

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["model"]["parameter_count"] == 29_913
    assert config["schedule"]["steps"] == 8
    assert config["training"]["updates"] == 3

    oracle = json.loads(
        (run_dir / "controls/oracle_gate.json").read_text(encoding="utf-8")
    )
    assert oracle["preflight_integrity_pass"] == 1
    assert oracle["prior_integrity_pass"] == 1
    assert oracle["integrity_pass"] == 1
    assert oracle["preflight_saved_states_all_finite"] == 1
    assert oracle["prior_saved_states_all_finite"] == 1

    reconstruction = json.loads(
        (run_dir / "controls/reconstruction_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert reconstruction["complete_forward_terminal_horizon"] == 7
    assert reconstruction["complete_forward_terminal_reconstruction_pass"] in (0, 1)
    assert reconstruction["any_short_horizon_reconstruction_pass"] in (0, 1)

    sampling = json.loads(
        (run_dir / "sampling/summary.json").read_text(encoding="utf-8")
    )
    assert sampling["prior_oracle_integrity_pass"] == 1
    with np.load(run_dir / "sampling/prior_panel.npz", allow_pickle=False) as panel:
        assert panel["initial"].shape == (2, 1, 28, 28)
        assert panel["zero_final"].shape == (2, 1, 28, 28)
        assert panel["oracle_final"].shape == (2, 1, 28, 28)
        assert panel["learned_final"].shape == (2, 1, 28, 28)
        assert panel["learned_anchors"].shape == (3, 2, 1, 28, 28)
        assert np.isfinite(panel["oracle_anchors"]).all()

    checkpoint = torch.load(
        run_dir / "training/best.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["parameter_count"] == 29_913
    restored = TinyClassConditionalUNet28()
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)

    command = (run_dir / "command.txt").read_text(encoding="utf-8")
    assert "-m mnist.diag_d0_tiny_ddpm_one_image" in command
    assert "--smoke" in command
    assert "-m pytest" not in command
    environment = json.loads(
        (run_dir / "environment.json").read_text(encoding="utf-8")
    )
    assert {"os", "cuda_runtime", "cudnn_version", "cuda_devices"} <= set(
        environment
    )

    for relative in (
        "source/source.png",
        "training/best.pt",
        "controls/reconstruction_panel.npz",
        "sampling/paired-final.png",
        "sampling/trajectory-step-0000.png",
        "REPORT.md",
        "artifact_manifest.json",
    ):
        assert (run_dir / relative).is_file(), relative
    manifest = json.loads(
        (run_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifact_count"] > 10
    paths = {row["path"] for row in manifest["artifacts"]}
    assert "sampling/prior_panel.npz" in paths
    assert "REPORT.md" in paths
    for row in manifest["artifacts"]:
        artifact = run_dir / row["path"]
        assert artifact.stat().st_size == row["size"]
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == row["sha256"]

    report = (run_dir / "REPORT.md").read_text(encoding="utf-8")
    training = json.loads(
        (run_dir / "training/summary.json").read_text(encoding="utf-8")
    )
    assert training["selected_checkpoint_sha256"] in report
    assert "engineering_smoke_complete" in report
    assert "says nothing about MNIST learning or model capacity" in report
    assert "Source revision" in report
    assert "Replay from the repository root" in report


def test_prior_panel_reuses_reverse_random_stream_for_each_model() -> None:
    config = _smoke_config()
    source = torch.linspace(-1.0, 1.0, 28 * 28).reshape(1, 1, 28, 28)
    schedule = make_linear_ddpm_schedule(
        config["schedule"]["steps"],
        config["schedule"]["beta_start"],
        config["schedule"]["beta_end"],
        device=torch.device("cpu"),
    )
    arrays, _, _ = _prior_panel(
        ZeroEpsilon(), source, 3, schedule, config
    )
    assert np.array_equal(arrays["zero_final"], arrays["learned_final"])
    assert np.array_equal(arrays["zero_anchors"], arrays["learned_anchors"])
