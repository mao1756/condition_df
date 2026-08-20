from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from mnist.d0_jacobi_rb_cuda import JacobiRBCudaProfile

from mnist.d0_jacobi_rb_runpod_source_integrity import (
    PROTECTED_SOURCE_CANONICAL_LF_HASHES,
    canonicalize_newlines,
    verify_protected_sources,
)

from mnist.diag_d0_jacobi_rb_path_weighted_capacity_e2e import (
    ExperimentConfig,
    _controller_metric,
    _parser,
    _scientific_interpretation,
)



def test_h100_runtime_profile_is_explicit_and_does_not_change_historical_default() -> None:
    historical = JacobiRBCudaProfile()
    assert historical.schema_version == 1
    assert historical.frozen_torch_version == "2.11.0+cu128"
    assert historical.frozen_cuda_version == "12.8"
    assert historical.frozen_compute_capability == "12.0"

    h100 = JacobiRBCudaProfile.h100_pytorch28()
    assert h100.schema_version == 2
    assert h100.frozen_torch_version == "2.8.0+cu128"
    assert h100.frozen_cuda_version == "12.8"
    assert h100.frozen_compute_capability == "9.0"


def test_weighted_runner_selects_h100_runtime_profile() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "mnist"
        / "diag_d0_jacobi_rb_path_weighted_capacity_e2e.py"
    ).read_text()
    assert "JacobiRBCudaProfile.h100_pytorch28()" in source


def test_cli_defaults_define_full_unattended_experiment() -> None:
    args = _parser().parse_args(["run", "--run-dir", "/tmp/example"])
    assert args.train_paths == 64
    assert args.validation_paths == 32
    assert args.small_updates == args.large_updates == 12_000
    assert args.batch_size == 32
    assert args.mobility_floor == 1e-4
    assert args.include_old_control is True
    assert args.hard_wall_seconds == 21_600


def test_metric_is_exact_at_target_and_detects_noise() -> None:
    target = np.arange(1, 785, dtype=np.float64)
    target /= target.sum()
    exact = _controller_metric(target, target)
    assert exact["squared_l2"] == 0.0
    assert exact["l1"] == 0.0
    shifted = np.roll(target, 1)
    changed = _controller_metric(shifted, target)
    assert changed["squared_l2"] > 0.0
    assert changed["mass_error"] < 1e-15


def test_interpretation_prioritizes_oracle_composition_failure() -> None:
    def row(value: float) -> dict[str, object]:
        return {"to_mixed_source": {"squared_l2": value}}

    same = {
        "metrics": {
            "zero": row(1.0),
            "small-old": row(0.9),
            "small-weighted": row(0.8),
            "large-weighted": row(0.7),
            "oracle": row(1.1),
        }
    }
    prior = {"metrics": dict(same["metrics"])}
    result = _scientific_interpretation(same, prior, True)
    assert result["automatic_interpretation"] == "reverse-composition-or-oracle-control-failure"


def test_runpod_launcher_requires_durable_storage_before_delete() -> None:
    root = Path(__file__).resolve().parents[1] / "tools" / "runpod_weighted_e2e"
    launcher = (root / "launch.sh").read_text()
    worker = (root / "worker.sh").read_text()
    finalizer = (root / "finalize.sh").read_text()
    lifecycle = (root / "pod_lifecycle.sh").read_text()
    assert "RUNPOD_RESULTS_DURABLE" in launcher
    assert "Refusing unattended Pod deletion" in launcher
    assert launcher.index('mkdir -p "$(dirname "${RUN_DIR}")"') < launcher.index(
        'mkdir "${LOCK_DIR}"'
    )
    assert "timeout --signal=TERM" in worker
    assert "ruff check --select I --fix" in worker
    assert "sed -i 's/\\r$//'" in launcher
    assert "/stop" in lifecycle
    assert "--request DELETE" in lifecycle
    assert "runpodctl stop pod" in lifecycle
    assert "runpodctl remove pod" in lifecycle
    assert "EXPORT_VERIFIED" in finalizer


def test_config_is_json_serializable() -> None:
    config = ExperimentConfig(
        run_dir="/tmp/run",
        device="cuda:0",
        data_dir="/tmp/data",
        mnist_index=0,
        download_mnist=True,
        train_paths=64,
        validation_paths=32,
        small_updates=12_000,
        large_updates=12_000,
        batch_size=32,
        validation_interval=250,
        mobility_floor=1e-4,
        include_old_control=True,
        hard_wall_seconds=21_600,
        seed=1,
    )
    json.dumps(config.to_record(), sort_keys=True)



def test_runpod_lifecycle_falls_back_to_legacy_cli(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1] / "tools" / "runpod_weighted_e2e"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    record = tmp_path / "calls.txt"
    fake = fake_bin / "runpodctl"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "echo \"$*\" >> \"${FAKE_RUNPOD_RECORD}\"\n"
        "if [[ \"${1:-}\" == \"pod\" ]]; then exit 1; fi\n"
        "if [[ \"${1:-}\" == \"remove\" && \"${2:-}\" == \"pod\" ]]; then exit 0; fi\n"
        "exit 1\n"
    )
    fake.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env.get('PATH', '')}",
            "RUNPOD_POD_ID": "test-pod",
            "RUNPOD_API_KEY": "",
            "RUNPOD_RESULTS_DURABLE": "1",
            "FAKE_RUNPOD_RECORD": str(record),
        }
    )
    subprocess.run(
        ["bash", str(root / "pod_lifecycle.sh"), "delete"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    calls = record.read_text().splitlines()
    assert any(call == "pod --help" for call in calls)
    assert any(call == "remove pod test-pod" for call in calls)



def test_runpod_network_volume_stop_falls_back_to_delete(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1] / "tools" / "runpod_weighted_e2e"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    record = tmp_path / "calls.txt"
    fake = fake_bin / "runpodctl"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "echo \"$*\" >> \"${FAKE_RUNPOD_RECORD}\"\n"
        "if [[ \"${1:-}\" == \"pod\" ]]; then exit 1; fi\n"
        "if [[ \"${1:-}\" == \"stop\" ]]; then exit 1; fi\n"
        "if [[ \"${1:-}\" == \"remove\" && \"${2:-}\" == \"pod\" ]]; then exit 0; fi\n"
        "exit 1\n"
    )
    fake.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env.get('PATH', '')}",
            "RUNPOD_POD_ID": "test-pod",
            "RUNPOD_API_KEY": "",
            "RUNPOD_RESULTS_DURABLE": "1",
            "FAKE_RUNPOD_RECORD": str(record),
        }
    )
    subprocess.run(
        ["bash", str(root / "pod_lifecycle.sh"), "stop"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    calls = record.read_text().splitlines()
    assert any(call == "stop pod test-pod" for call in calls)
    assert any(call == "remove pod test-pod" for call in calls)


def test_runpod_shell_scripts_are_lf_only() -> None:
    root = Path(__file__).resolve().parents[1] / "tools" / "runpod_weighted_e2e"
    for script in sorted(root.glob("*.sh")):
        raw = script.read_bytes()
        assert b"\r\n" not in raw, f"{script.name} contains CRLF line endings"


def test_runpod_protected_source_integrity_is_newline_portable(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    assert len(PROTECTED_SOURCE_CANONICAL_LF_HASHES) == 29
    verify_protected_sources(root)

    mixed = b"first\r\nsecond\nthird\r\n"
    lf = b"first\nsecond\nthird\n"
    crlf = b"first\r\nsecond\r\nthird\r\n"
    assert canonicalize_newlines(mixed) == lf
    assert canonicalize_newlines(crlf) == lf

    source = root / "mnist/conditioned_diffusion.py"
    converted = tmp_path / "conditioned_diffusion.py"
    converted.write_bytes(canonicalize_newlines(source.read_bytes()))
    expected = {
        "conditioned_diffusion.py": PROTECTED_SOURCE_CANONICAL_LF_HASHES[
            "mnist/conditioned_diffusion.py"
        ]
    }
    verify_protected_sources(tmp_path, expected)
