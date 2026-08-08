from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mnist import diag_d0_jacobi_rb_one_image_learnability as cli
from mnist.d0_jacobi_rb_learnability import (
    LearnabilityCacheBundle,
    LearnabilityInputCache,
    LearnabilityLabelAuditCache,
    frozen_path_plan,
    sample_key,
)


STRANG = Path(
    "runs/experiment12_d0_jacobi_rb_strang_refinement/"
    "20260723-230629_production-state-dependent-strang-refinement"
)


def _selected_capture(step: int) -> dict[str, np.ndarray]:
    path_ids = np.asarray(cli.PATH_IDS["train"], dtype=np.int64)
    states = np.full((7, 8, 784), 1.0 / 784.0, dtype=np.float64)
    return {
        "path_ids": path_ids,
        "outer_steps": np.full(7, step, dtype=np.int16),
        "phases": np.arange(7, dtype=np.int8),
        "later_head_fractions": np.full((7, 8, 392), 0.5, dtype=np.float64),
        "denoising_targets": np.full(
            (7, 8, 392), step / 512.0, dtype=np.float64
        ),
        "certificate_codes": np.ones((7, 8, 392), dtype=np.uint8),
        "post_phase_states": states,
    }


def test_source_image_binding_uses_historical_measure_digest() -> None:
    metadata, image, mixed = cli._load_source_image(STRANG)
    assert metadata["npz_sha256"] == cli.SOURCE_IMAGE_NPZ_SHA256
    assert cli._source_measure_sha(image) == cli.IMAGE_SHA256
    assert cli._source_measure_sha(mixed) == cli.MIXED_TARGET_SHA256
    assert image.sum() == pytest.approx(1.0, abs=1.0e-12)
    assert mixed.sum() == pytest.approx(1.0, abs=1.0e-12)


def test_frozen_cli_path_plan_matches_collision_scanned_core() -> None:
    record = cli._path_plan()
    core = frozen_path_plan()
    assert record["roles"] == core.to_record()["roles"]
    assert record["repository_collision_scan_pass"] == 1
    assert set(core.train).isdisjoint(core.validation)
    assert set(core.train).isdisjoint(core.confirmation)
    assert min(core.train) == 0xE0000


def test_capture_flattening_builds_exact_separated_cache_contract() -> None:
    captures = [_selected_capture(step) for step in cli.SELECTED_OUTER_STEPS]
    inputs, audit, metrics = cli._flatten_selected_captures(captures)
    assert metrics["sample_count"] == 1792
    assert metrics["capture_state_alignment_pass"] == 1
    assert set(inputs) == set(cli.INPUT_FIELDS)
    assert set(audit) == set(cli.AUDIT_FIELDS)
    assert np.array_equal(inputs["sample_key"], audit["sample_key"])
    assert inputs["sample_key"][0] == sample_key(
        cli.PATH_IDS["train"][0], 15, 0
    )
    bundle = LearnabilityCacheBundle(
        LearnabilityInputCache(**inputs),
        LearnabilityLabelAuditCache(**audit),
    )
    assert bundle.sample_count == 1792
    assert bundle.inputs.later_full_state.dtype == np.float64
    assert bundle.labels_audit.denoising_target.dtype == np.float64


def test_selected_capture_keeps_only_local_step_seven() -> None:
    from mnist.d0_jacobi_rb_cuda_multipath import ExactMultipathCapturePayload

    arrays = {
        "later_head_fractions": np.zeros((56, 1, 392), dtype=np.float64),
        "denoising_targets": np.zeros((56, 1, 392), dtype=np.float64),
        "certificate_codes": np.ones((56, 1, 392), dtype=np.uint8),
        "post_phase_states": np.full((56, 1, 784), 1.0 / 784.0),
    }
    payload = ExactMultipathCapturePayload(
        path_ids=(cli.PATH_IDS["train"][0],),
        start_step=8,
        outer_steps=tuple(8 + index // 7 for index in range(56)),
        phases=tuple(index % 7 for index in range(56)),
        **arrays,
    )
    selected = cli._selected_capture_arrays(payload)
    assert selected["outer_steps"].tolist() == [15] * 7
    assert selected["phases"].tolist() == list(range(7))
    assert selected["post_phase_states"].shape == (7, 1, 784)


def test_confirmation_open_requires_and_preserves_frozen_hashes(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "selected_model.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = cli.file_fingerprint(checkpoint)
    cli.atomic_write_json(
        tmp_path / "selected_model.json",
        {"checkpoint_file_sha256": checkpoint_sha},
    )
    seal = {
        "selected_model_file_sha256": checkpoint_sha,
        "confirmation_path_ids": list(cli.PATH_IDS["confirmation"]),
    }
    seal["seal_sha256"] = cli.config_fingerprint(seal)
    cli.atomic_write_json(tmp_path / "confirmation_seal.json", seal)
    opened = cli._open_confirmation(tmp_path)
    assert opened["opened_count"] == 1
    assert opened["path_ids"] == list(cli.PATH_IDS["confirmation"])
    assert cli._open_confirmation(tmp_path) == opened


def test_parser_requires_resume_after_preflight() -> None:
    parents = [
        "--parent-multipath-run-dir",
        "m",
        "--parent-strang-run-dir",
        "s",
        "--parent-haar-run-dir",
        "h",
    ]
    with pytest.raises(SystemExit):
        cli.parse_args(["--stage", "cache", *parents])
    args = cli.parse_args(
        ["--stage", "cache", "--resume-run-dir", "run", *parents]
    )
    assert args.stage == "cache"


def test_success_claim_flags_remain_fail_closed() -> None:
    config = cli._scientific_config(authorizing=True)
    for name, value in cli.CLAIM_FLAGS.items():
        assert config[name] == value == 0
    assert "exact K=512 split chain" in config["claim_scope"]


def test_cli_does_not_import_reverse_or_reconstruction_sampler() -> None:
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert "reverse_sampler" not in source
    assert "reconstruction_sampler" not in source


def test_new_run_binds_the_reviewed_historical_source_compatibility_table() -> None:
    names = {path.name for path in cli._source_paths()}
    assert "d0_jacobi_source_compat.py" in names
    assert "d0_jacobi_artifacts.py" in names
    assert "d0_jacobi_rb_cuda_multipath.py" in names


def test_historical_source_successor_is_whole_set_exact() -> None:
    run = Path(
        "runs/experiment12_d0_jacobi_rb_cuda_multipath_confirmation/"
        "20260723-092105_production-multipath-jacobi-rb"
    )
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    paths = [Path(value) for value in manifest["source_paths"]]
    assert cli.source_fingerprint(paths) == manifest["source_fingerprint"]
    expanded = [*paths, Path("mnist/d0_jacobi_source_compat.py")]
    assert cli.source_fingerprint(expanded) != manifest["source_fingerprint"]


def test_confirmation_open_and_partial_shards_count_as_open_evidence(
    tmp_path: Path,
) -> None:
    assert cli._no_confirmation_artifacts(tmp_path)
    cli.atomic_write_json(tmp_path / "confirmation_open.json", {"opened_count": 1})
    assert not cli._no_confirmation_artifacts(tmp_path)
    (tmp_path / "confirmation_open.json").unlink()
    shard_dir = tmp_path / "cache" / "confirmation_shards"
    shard_dir.mkdir(parents=True)
    (shard_dir / "partial.tmp").write_bytes(b"partial")
    assert not cli._no_confirmation_artifacts(tmp_path)


def test_registry_verification_detects_registered_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "evidence.json"
    cli.atomic_write_json(artifact, {"passed": 1})
    registry = cli._artifact_registry(tmp_path)
    cli._status(
        tmp_path,
        stage="preflight",
        state="completed",
        registry=registry,
    )
    assert cli._verify_existing_artifact_registry(tmp_path) == registry
    cli.atomic_write_json(artifact, {"passed": 0})
    with pytest.raises(cli.ArtifactCompatibilityError, match="changed"):
        cli._verify_existing_artifact_registry(tmp_path)


def test_terminal_registry_rejects_an_unregistered_extra_file(
    tmp_path: Path,
) -> None:
    cli.atomic_write_json(tmp_path / "evidence.json", {"passed": 1})
    registry = cli._artifact_registry(tmp_path)
    cli._status(
        tmp_path,
        stage="preflight",
        state="completed",
        registry=registry,
    )
    cli.atomic_write_json(tmp_path / "unregistered.json", {"unexpected": 1})
    with pytest.raises(
        cli.ArtifactCompatibilityError,
        match="terminal artifact registry file set changed",
    ):
        cli._verify_existing_artifact_registry(tmp_path)


def test_physical_work_detection_survives_interrupted_task(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "physical-rb-seed-261201-progress.pt").write_bytes(b"x")
    assert cli._physical_work_performed(tmp_path)


@pytest.mark.parametrize(
    ("stage", "provenance", "expected"),
    [
        ("preflight", False, "exact_cache_invalid"),
        ("cache", False, "exact_cache_invalid"),
        ("train", False, "optimization_pipeline_invalid"),
        ("confirm", False, "optimization_pipeline_invalid"),
        ("preflight", True, "parent_scope_invalid"),
        ("cache", True, "parent_scope_invalid"),
    ],
)
def test_failure_classification_is_stage_local(
    tmp_path: Path, stage: str, provenance: bool, expected: str
) -> None:
    decision = cli._commit_failure_decision(
        tmp_path,
        stage=stage,
        required_gate=stage if stage != "confirm" else "confirm",
        failure={"failure_code": "fixture"},
        provenance_failure=provenance,
    )
    assert decision == expected


def test_input_contract_failure_has_its_closed_decision(tmp_path: Path) -> None:
    decision = cli._commit_failure_decision(
        tmp_path,
        stage="preflight",
        required_gate="preflight",
        failure={
            "failure_domain": "model_input_contract",
            "failure_code": "model_input_contract_unexpected_field",
        },
        provenance_failure=False,
    )
    assert decision == "model_input_contract_invalid"


def test_cache_shard_resume_binds_scheduler_identity(tmp_path: Path) -> None:
    from mnist import d0_jacobi_rb_cuda_controls as controls

    path_ids = (cli.PATH_IDS["train"][0],)
    initial = np.full((1, cli.PATH_STATE_SIZE), 1.0 / cli.PATH_STATE_SIZE)
    final = initial.copy()
    scheduler_record = {
        "schema": "jacobi-rb-cuda-exact-multipath-v1-shard",
        "schema_version": 1,
        "path_records": [
            {
                "path_id": path_ids[0],
                "input_state_sha256": controls._digest_arrays(initial[0]),
                "final_state_sha256": controls._digest_arrays(final[0]),
            }
        ],
        "phase_state_records": [],
        "batch_output_sha256": "a" * 64,
        "batch_final_state_sha256": controls._digest_arrays(final),
        "batch_certificate_sha256": "b" * 64,
        "diagnostics": {
            "start_step": 0,
            "step_count": 8,
            "path_ids": list(path_ids),
            "group_sizes": [1],
            "transition_count": 8 * 7 * 392,
            "phase_state_trace_enabled": 0,
        },
    }
    result = SimpleNamespace(
        committed_final_states=final,
        to_record=lambda: scheduler_record,
    )
    state, capture, metadata = cli._persist_shard(
        tmp_path,
        split="train",
        start_step=0,
        path_ids=path_ids,
        input_state_sha256=cli._array_sha(initial),
        scientific_config_sha256="c" * 64,
        path_plan_sha256="d" * 64,
        profile_sha256="e" * 64,
        result=result,
        capture=None,
    )
    valid, restored, _ = cli._valid_committed_shard(
        state_path=state,
        capture_path=capture,
        metadata_path=metadata,
        expected_split="train",
        expected_start_step=0,
        expected_path_ids=path_ids,
        expected_input_states=initial,
        expected_input_sha256=cli._array_sha(initial),
        capture_expected=False,
        scientific_config_sha256="c" * 64,
        path_plan_sha256="d" * 64,
        profile_sha256="e" * 64,
    )
    assert valid and np.array_equal(restored, final)
    assert not cli._valid_committed_shard(
        state_path=state,
        capture_path=capture,
        metadata_path=metadata,
        expected_split="validation",
        expected_start_step=0,
        expected_path_ids=path_ids,
        expected_input_states=initial,
        expected_input_sha256=cli._array_sha(initial),
        capture_expected=False,
        scientific_config_sha256="c" * 64,
        path_plan_sha256="d" * 64,
        profile_sha256="e" * 64,
    )[0]
