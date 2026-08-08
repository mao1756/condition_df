from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError
from mnist.d0_jacobi_rb_bayes_power_provenance import PARENT_RUN_BASENAME
import mnist.diag_d0_jacobi_rb_bayes_power_calibration as cli
from mnist.diag_d0_jacobi_rb_bayes_power_calibration import (
    _stage_sequence,
    _template_for_role,
    main,
    parse_args,
)


REPOSITORY = Path(__file__).resolve().parents[1]
PARENT = (
    REPOSITORY
    / "runs"
    / "experiment12_d0_jacobi_rb_one_image_learnability"
    / PARENT_RUN_BASENAME
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest(root: Path) -> Path:
    values = sorted(path for path in root.iterdir() if path.is_dir())
    assert len(values) == 1
    return values[0]


def _reduced_args(root: Path) -> list[str]:
    return [
        "--runs-root",
        str(root),
        "--run-name",
        "reduced-cli-fixture",
        "--parent-one-image-run-dir",
        str(PARENT),
        "--device",
        "cpu",
        "--stage",
        "all",
        "--require-gate",
        "none",
        "--test-only-reduced-workload",
        "--test-only-paths",
        "1",
        "--test-only-selected-steps",
        "4",
        "--test-only-updates",
        "0",
        "--test-only-sampler-double",
    ]


def test_stage_sequence_is_explicit_and_confirmation_is_separate() -> None:
    assert _stage_sequence("all") == (
        "preflight",
        "cache",
        "train",
        "confirm",
        "report",
    )
    assert _stage_sequence("confirm") == ("confirm",)


def test_test_only_overrides_fail_closed_without_test_flag() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--test-only-paths", "2"])
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--test-only-reduced-workload",
                "--test-only-selected-steps",
                "3",
            ]
        )


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable parent run is unavailable")
def test_reduced_cpu_workflow_is_restartable_and_label_firewalled(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    assert main(_reduced_args(root)) == 0
    run_dir = _latest(root)

    status = _json(run_dir / "run_status.json")
    registry = _json(run_dir / "artifact_registry.json")
    provenance = _json(run_dir / "parent_provenance.json")
    seal = _json(run_dir / "confirmation_seal.json")
    opened = _json(run_dir / "confirmation_open.json")
    selection = _json(run_dir / "selected_candidates.json")

    assert status["state"] == "completed"
    assert status["physical_training_performed"] == 0
    assert status["sampling_performed"] == 0
    assert registry["record_count"] > 20
    assert status["artifact_registry_record_count"] == registry["record_count"]
    assert status["artifact_registry_sha256"] == registry["semantic_sha256"]
    assert opened["opened_count"] == 1
    assert opened["seal_sha256"] == seal["seal_sha256"]
    assert selection["analytic_zero_candidate_laws"] == ["teacher", "null"]
    assert selection["analytic_zero_candidate_laws"] == ["teacher", "null"]
    assert selection["teacher_target_scale"] > 0.0
    assert selection["null_target_scale"] > 0.0
    assert set(provenance["accessed_parent_paths"]) == {
        "cache/train_inputs.npz",
        "cache/validation_inputs.npz",
        "cache/confirmation_inputs.npz",
    }
    assert provenance["forbidden_label_bytes_opened"] == 0
    assert not any("labels_audit" in value for value in provenance["accessed_parent_paths"])

    # Oracle data is physically separate from the model-input and noisy-label
    # files, so accidental field widening is visible and testable.
    with np.load(run_dir / "cache" / "teacher_train_inputs.npz") as archive:
        assert "oracle_conditional_mean" not in archive.files
        assert "denoising_target" not in archive.files
    with np.load(run_dir / "cache" / "teacher_train_labels.npz") as archive:
        assert "oracle_conditional_mean" not in archive.files
        assert "denoising_target" in archive.files
    with np.load(
        run_dir / "cache" / "teacher_train_oracle_audit.npz"
    ) as archive:
        assert "oracle_conditional_mean" in archive.files

    assert main(
        [
            "--resume-run-dir",
            str(run_dir),
            "--parent-one-image-run-dir",
            str(PARENT),
            "--device",
            "cpu",
            "--stage",
            "report",
            "--require-gate",
            "none",
            "--test-only-reduced-workload",
            "--test-only-paths",
            "1",
            "--test-only-selected-steps",
            "4",
            "--test-only-updates",
            "0",
            "--test-only-sampler-double",
        ]
    ) == 0
    assert _json(run_dir / "confirmation_open.json")["opened_count"] == 1
    for law in ("teacher", "null"):
        payload = __import__("torch").load(
            run_dir / f"selected_{law}_model.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert payload["schema"].endswith("-selected-model")


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable parent run is unavailable")
def test_reduced_template_keeps_whole_paths_and_all_quartiles() -> None:
    template = _template_for_role(
        PARENT,
        "train",
        count=1,
        selected_steps=4,
    )
    keys = np.asarray(template.sample_key)
    paths = keys >> 13
    steps = (keys >> 3) & ((1 << 10) - 1)
    assert len(np.unique(paths)) == 1
    assert len(np.unique(steps)) == 4
    assert set((np.unique(steps) // 128).tolist()) == {0, 1, 2, 3}
    assert keys.size == 4 * 7


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable parent run is unavailable")
def test_resume_rejects_runtime_binding_change_before_stage_work(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    args = _reduced_args(root)
    args[args.index("all")] = "preflight"
    assert main(args) == 0
    run_dir = _latest(root)
    status_before = (run_dir / "run_status.json").read_bytes()
    with pytest.raises(ArtifactCompatibilityError):
        main(
            [
                "--resume-run-dir",
                str(run_dir),
                "--parent-one-image-run-dir",
                str(PARENT),
                "--device",
                "cuda",
                "--stage",
                "report",
                "--require-gate",
                "none",
                "--test-only-reduced-workload",
                "--test-only-paths",
                "1",
                "--test-only-selected-steps",
                "4",
                "--test-only-updates",
                "0",
                "--test-only-sampler-double",
            ]
        )
    assert (run_dir / "run_status.json").read_bytes() == status_before


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable parent run is unavailable")
def test_fresh_parent_provenance_failure_commits_terminal_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_parent(*args, **kwargs):
        raise ArtifactCompatibilityError("synthetic parent mismatch")

    monkeypatch.setattr(cli, "verify_no_signal_parent", reject_parent)
    root = tmp_path / "runs"
    code = main(
        [
            "--runs-root",
            str(root),
            "--run-name",
            "provenance-failure",
            "--parent-one-image-run-dir",
            str(PARENT),
            "--device",
            "cpu",
            "--stage",
            "preflight",
            "--require-gate",
            "preflight",
        ]
    )
    assert code == 2
    run_dir = _latest(root)
    assert (run_dir / "preflight_provenance_failure.json").is_file()
    assert (run_dir / "artifact_registry.json").is_file()
    status = _json(run_dir / "run_status.json")
    assert status["state"] == "execution_failed"
    assert status["decision"] == "control_provenance_invalid"


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable parent run is unavailable")
def test_completed_role_cache_tamper_fails_before_skip(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    assert main(_reduced_args(root)) == 0
    run_dir = _latest(root)
    target = run_dir / "cache" / "teacher_train_labels.npz"
    body = bytearray(target.read_bytes())
    body[len(body) // 2] ^= 0x01
    target.write_bytes(body)
    code = main(
        [
            "--resume-run-dir",
            str(run_dir),
            "--parent-one-image-run-dir",
            str(PARENT),
            "--device",
            "cpu",
            "--stage",
            "cache",
            "--require-gate",
            "none",
            "--test-only-reduced-workload",
            "--test-only-paths",
            "1",
            "--test-only-selected-steps",
            "4",
            "--test-only-updates",
            "0",
            "--test-only-sampler-double",
        ]
    )
    assert code == 2
    failure = _json(run_dir / "cache_execution_failure.json")
    assert failure["failure_domain"] == "exact_control_cache"
    assert failure["failure_code"] == "completed_control_cache_invalid"
