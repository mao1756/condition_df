from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from mnist import diag_d0_jacobi_rb_global_dilated_continuation as workflow
from mnist.d0_jacobi_rb_learnability import semantic_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_RUNNER = REPOSITORY_ROOT / "mnist/diag_d0_jacobi_rb_global_dilated_rollout.py"
V3_RELATIVE = Path(
    "runs/experiment12-d0-jacobi-rb-global-dilated-rollout/"
    "20260813-233915_production-global-dilated-exact-five-row-v3"
)
SOURCE_RELATIVE = Path(
    "runs/experiment12_d0_jacobi_rb_frequency1_rollout/"
    "20260813-002414_production-frequency1-objective-first-recovery-v4/"
    "input_bindings"
)
V2_RELATIVE = Path(
    "runs/experiment12-d0-jacobi-rb-global-dilated-continuation/"
    "production-same-path-complete-v2"
)
V2_RESOURCE_LEDGER_FILE_SHA256 = (
    "47f9b491ac0e2616f1e03fe977a42cd0475bad80bf7e3e25501d7ccd1fba608e"
)
V2_RESOURCE_LEDGER_SEMANTIC_SHA256 = (
    "b559e0110178e51d6062f627c4bcc3f0d949294f20a902698a988910d7fb4335"
)
V2_TERMINAL_STORAGE_FILE_SHA256 = (
    "e7a7beb58aa2d3fa49fe9224aff5f46bae5b526d607d93e0311f53e7363e4948"
)
V2_TERMINAL_STORAGE_SEMANTIC_SHA256 = (
    "54db340a9ad49a2e679ebf8d5f6b75f3d92bce688e55e10c071612c24560ca24"
)
V2_ACTIVE_SECONDS = 1945.6831628999998


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fresh_argv(tmp_path: Path, *, stage: str = "prepare") -> list[str]:
    return [
        "--stage",
        stage,
        "--repository-root",
        str(REPOSITORY_ROOT),
        "--runs-root",
        str(tmp_path / "runs"),
        "--run-name",
        "child",
        "--prefix-run-dir",
        str(REPOSITORY_ROOT / V2_RELATIVE),
        "--parent-run-dir",
        str(REPOSITORY_ROOT / V3_RELATIVE),
        "--source-run-dir",
        str(REPOSITORY_ROOT / SOURCE_RELATIVE),
        "--device",
        "cuda",
    ]


def _resume_argv(tmp_path: Path, *, stage: str = "controls") -> list[str]:
    return [
        "--stage",
        stage,
        "--repository-root",
        str(REPOSITORY_ROOT),
        "--resume-run-dir",
        str(tmp_path / "child"),
        "--prefix-run-dir",
        str(REPOSITORY_ROOT / V2_RELATIVE),
        "--parent-run-dir",
        str(REPOSITORY_ROOT / V3_RELATIVE),
        "--source-run-dir",
        str(REPOSITORY_ROOT / SOURCE_RELATIVE),
        "--device",
        "cuda",
    ]


def _verify_argv(tmp_path: Path) -> list[str]:
    return [
        "--repository-root",
        str(REPOSITORY_ROOT),
        "--verify-run-dir",
        str(tmp_path / "child"),
        "--prefix-run-dir",
        str(REPOSITORY_ROOT / V2_RELATIVE),
        "--parent-run-dir",
        str(REPOSITORY_ROOT / V3_RELATIVE),
        "--source-run-dir",
        str(REPOSITORY_ROOT / SOURCE_RELATIVE),
    ]


def _validate(argv: list[str]) -> tuple[object, str]:
    args = workflow.parse_args(argv)
    mode = workflow._resolve_mode(args)
    workflow._validate_cli_combination(args, mode=mode)
    return args, mode


def _classification(
    *,
    zero: float,
    global_error: float,
    source: float,
    intermediate_relative: dict[int, float] | None = None,
) -> dict[str, object]:
    intermediate = {
        int(step): {
            "zero_error": 1.0,
            "global_error": 1.0 - float(relative),
        }
        for step, relative in (intermediate_relative or {}).items()
    }
    return dict(
        workflow._classify_complete_outcome(
            zero_error=zero,
            global_error=global_error,
            source_error=source,
            intermediate=intermediate,
        )
    )


def _field(record: dict[str, object], *names: str) -> object:
    for name in names:
        if name in record:
            return record[name]
    raise AssertionError(f"record omitted every expected field: {names}")


def _valid_strict_reverse_fixture() -> tuple[np.ndarray, list[dict[str, object]]]:
    """Build all 64 exact shard authorities without running the numerical backend."""

    per_row = (
        workflow.FUSED_SHARD_PHASES
        * 2
        * workflow.MICROSTEPS
        * workflow.EDGES_PER_PHASE
    )
    per_shard = per_row * len(workflow.ROW_ORDER)
    row_active = per_row - 8
    sequence = tuple(workflow.reverse_suffix_sequence(511))
    assert len(sequence) == workflow.REVERSE_SHARDS * workflow.FUSED_SHARD_PHASES
    row_table = [
        {
            "row_key": "zero",
            "canonical_path_id": workflow.PATH_ID,
            "controller_kind": "zero",
            "variant": "zero",
            "horizon": "same-path-complete",
            "gain": None,
            "controller_binding": {},
        },
        {
            "row_key": "global-plus-1",
            "canonical_path_id": workflow.PATH_ID,
            "controller_kind": "learned",
            "variant": "global-dilated",
            "horizon": "same-path-complete",
            "gain": 1.0,
            "controller_binding": {
                "checkpoint_state_sha256": workflow.CHECKPOINT_STATE_SHA256
            },
        },
        {
            "row_key": "source-informed",
            "canonical_path_id": workflow.PATH_ID,
            "controller_kind": "oracle",
            "variant": "mixed-target-fraction",
            "horizon": "same-path-complete",
            "gain": None,
            "controller_binding": {
                "target_sha256": workflow.MIXED_TARGET_ARRAY_SHA256
            },
        },
    ]
    reference_rows = [
        {
            "row_key": key,
            "transition_count": per_row,
            "active_count": row_active,
            "structural_noop_count": 8,
            "certified_count": row_active,
            "fallback_count": 2,
            "unauthorized_count": 0,
            "invalid_count": 0,
            "certificate_fraction": 1.0,
        }
        for key in workflow.ROW_ORDER
    ]
    phase_rows: list[dict[str, object]] = []
    controller_rows: list[dict[str, object]] = []
    for key, row_authority in zip(workflow.ROW_ORDER, row_table, strict=True):
        phase: dict[str, object] = {
            "row_key": key,
            "transition_count": per_row,
            "reference_transition_count": per_row,
            "reference_active_count": row_active,
            "reference_structural_noop_count": 8,
            "reference_certified_count": row_active,
            "reference_fallback_count": 2,
            "reference_unauthorized_count": 0,
            "reference_invalid_count": 0,
            "reference_certificate_fraction": 1.0,
            "boundary_fraction_count": 0,
            "maximum_pair_mass_error": 0.0,
            "maximum_simplex_mass_error": 0.0,
            **{name: 0 for name in workflow._FUSED_INVALID_FIELDS},
        }
        for prefix in workflow._FUSED_PHASE_PREFIXES:
            phase[prefix + "_count"] = row_active
            phase[prefix + "_squared_sum"] = 0.0
            phase[prefix + "_maximum_absolute"] = 0.0
            phase[prefix + "_rms"] = 0.0
        phase_rows.append(phase)
        controller = {
            "row_key": key,
            "controller_kind": row_authority["controller_kind"],
            "gain": row_authority["gain"],
            **{name: 0 for name in workflow._FUSED_CONTROLLER_INTEGER_FIELDS},
            **{name: 0.0 for name in workflow._FUSED_CONTROLLER_FLOAT_FIELDS},
        }
        controller["call_count"] = workflow.FUSED_SHARD_PHASES
        controller["lane_count"] = per_row
        controller["score_count"] = row_active
        controller_rows.append(controller)

    records: list[dict[str, object]] = []
    controller_binding = {
        "row_table": copy.deepcopy(row_table),
        "global_state_sha256": workflow.CHECKPOINT_STATE_SHA256,
        "target_sha256": workflow.MIXED_TARGET_ARRAY_SHA256,
        "dispatch": "stable_one_row_canonical_order",
        "model_input_contract": "exact_ModelInputs_six_fields",
    }
    rng_binding = {
        "root_seed": workflow.REVERSE_ROOT_SEED,
        "stream_role": workflow.REVERSE_STREAM_ROLE,
        "canonical_path_id": workflow.PATH_ID,
    }
    for shard_index in range(workflow.REVERSE_SHARDS):
        offset = shard_index * workflow.FUSED_SHARD_PHASES
        shard_sequence = sequence[offset : offset + workflow.FUSED_SHARD_PHASES]
        records.append(
            {
                "schema": "d0-jacobi-rb-tangent-fused-v1-reverse-shard",
                "schema_version": 1,
                "scheduler_version": "d0-jacobi-rb-tangent-fused-v1",
                "family_name": "same-path-three-row",
                "segment_name": "complete-512",
                "committed": 1,
                "shard_index": shard_index,
                "sequence_start": list(shard_sequence[0]),
                "sequence_end": list(shard_sequence[-1]),
                "sequence_sha256": semantic_sha256(
                    [list(item) for item in shard_sequence]
                ),
                "row_keys": list(workflow.ROW_ORDER),
                "canonical_path_ids": [workflow.PATH_ID] * len(workflow.ROW_ORDER),
                "label": 3,
                "variant_in_rng_key": 0,
                "microsteps": workflow.MICROSTEPS,
                "controller_binding_sha256": semantic_sha256(controller_binding),
                "rng_binding_sha256": semantic_sha256(rng_binding),
                "input_state_sha256": "0" * 64,
                "output_state_sha256": "1" * 64,
                "state_file_sha256": "2" * 64,
                "state_file_size": 1,
                "transition_count": per_shard,
                "synchronous_replay_performed": 1,
                "execution_plan": {
                    "row_count": len(workflow.ROW_ORDER),
                    "transition_count": per_shard,
                    "sequence": [list(item) for item in shard_sequence],
                },
                "row_table": copy.deepcopy(row_table),
                "per_row_diagnostics": copy.deepcopy(phase_rows),
                "controller_diagnostics": copy.deepcopy(controller_rows),
                "diagnostics": {
                    "transition_count": per_shard,
                    "certificate_fraction": 1.0,
                    "maximum_mass_error": 0.0,
                    "fallback_count": 6,
                    "forbidden_counts": {
                        name: 0 for name in workflow._FORBIDDEN_EXACT_COUNTS
                    },
                    "reference": {
                        "schema": "d0-jacobi-rb-tangent-rollout-v1-certified-reference",
                        "root_seed": workflow.REVERSE_ROOT_SEED,
                        "stream_role": workflow.REVERSE_STREAM_ROLE,
                        "rng_namespace": "d0-jacobi-rb-frequency1-exploratory-reference-v1",
                        "variant_in_rng_key": 0,
                        "needs_synchronous_replay": 0,
                        "speculative_attempt_discarded": 1,
                        "transition_count": per_shard,
                        "active_count": row_active * len(workflow.ROW_ORDER),
                        "structural_noop_count": 8 * len(workflow.ROW_ORDER),
                        "certified_count": row_active * len(workflow.ROW_ORDER),
                        "fallback_count": 6,
                        "unauthorized_count": 0,
                        "invalid_count": 0,
                        "certificate_fraction": 1.0,
                        "forbidden_counts": {
                            name: 0 for name in workflow._FORBIDDEN_EXACT_COUNTS
                        },
                        "per_row": copy.deepcopy(reference_rows),
                    },
                },
            }
        )
    final_state = np.full(
        (len(workflow.ROW_ORDER), workflow.STATE_SIZE),
        1.0 / workflow.STATE_SIZE,
        dtype=np.float64,
    )
    return final_state, records


def _write_synthetic_reverse_chain(
    run_dir: Path,
) -> tuple[SimpleNamespace, np.ndarray, list[dict[str, object]]]:
    """Write a healthy, scientifically directional raw chain for derived tests."""

    run_dir = Path(run_dir)
    uniform = np.full(workflow.STATE_SIZE, 1.0 / workflow.STATE_SIZE, dtype=np.float64)
    target = np.arange(1, workflow.STATE_SIZE + 1, dtype=np.float64)
    target /= np.sum(target)
    source = SimpleNamespace(
        source_image=target.copy(),
        mixed_target=target.copy(),
        metadata={"lambda_mix": 0.2},
    )
    workflow.atomic_rollout_npz(
        run_dir / "forward/anchor-step-0511.npz", {"state": uniform}
    )
    scale = workflow.fixed_rendering_scale(
        source.source_image, source.mixed_target, source.metadata["lambda_mix"]
    )
    workflow._write_semantic(
        run_dir / "continuation_freeze.json",
        {
            "schema": "fixture-freeze",
            "schema_version": 1,
            "rendering_scale": scale.to_dict(),
        },
    )
    validation = np.tile(
        np.linspace(0.0, 2.0, 14_336, dtype=np.float64), (4, 1)
    )
    workflow.atomic_rollout_npz(
        run_dir / "inputs/calibration/on_policy_validation_calibration.npz",
        {
            "training_means": np.tile(uniform, (4, 1)),
            "training_p95": np.ones(4, dtype=np.float64),
            "validation_sorted_ratios": validation,
            "validation_counts": np.full(4, 14_336, dtype=np.int64),
        },
    )
    _final, records = _valid_strict_reverse_fixture()
    root = run_dir / "reverse/fused_families/same-path-three-row/complete-512"
    boundaries = [np.repeat(uniform[None, :], len(workflow.ROW_ORDER), axis=0)]
    previous = workflow.rollout_array_sha256(boundaries[0])
    for shard_index, record in enumerate(records):
        progress = (shard_index + 1) / workflow.REVERSE_SHARDS
        state = np.ascontiguousarray(
            np.stack(
                (
                    uniform,
                    (1.0 - 0.25 * progress) * uniform + 0.25 * progress * target,
                    (1.0 - 0.90 * progress) * uniform + 0.90 * progress * target,
                )
            ),
            dtype=np.float64,
        )
        archive = root / f"shard-{shard_index:04d}.npz"
        workflow.atomic_rollout_npz(archive, {"state": state})
        record.update(
            family_name="same-path-three-row",
            segment_name="complete-512",
            input_state_sha256=previous,
            output_state_sha256=workflow.rollout_array_sha256(state),
            state_file_sha256=_sha256(archive),
            state_file_size=archive.stat().st_size,
            elapsed_seconds=0.01,
        )
        records[shard_index] = workflow._write_semantic(
            root / f"shard-{shard_index:04d}.json", record
        )
        previous = workflow.rollout_array_sha256(state)
        boundaries.append(state)
    expected = np.ascontiguousarray(np.stack(boundaries, axis=1), dtype=np.float64)
    return source, expected, records


def _compatible_resume_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    child = tmp_path / "child"
    prefix = tmp_path / "prefix"
    parent = tmp_path / "parent"
    source = tmp_path / "source"
    for root, payload in (
        (child, b"child"),
        (prefix, b"prefix"),
        (parent, b"parent"),
        (source, b"source"),
    ):
        root.mkdir()
        (root / "identity.bin").write_bytes(payload)
    module = REPOSITORY_ROOT / "mnist/diag_d0_jacobi_rb_global_dilated_continuation.py"
    relative = module.relative_to(REPOSITORY_ROOT).as_posix()
    identity: dict[str, object] = {
        "run_schema": workflow.RUN_SCHEMA,
        "source_closure": {
            relative: {"size": module.stat().st_size, "sha256": _sha256(module)}
        },
        "prefix_run_dir": str(prefix.resolve()),
        "parent_run_dir": str(parent.resolve()),
        "source_run_dir": str(source.resolve()),
        "parent_tree_sha256": workflow._tree_hash(workflow._snapshot_tree(parent)),
        "source_tree_sha256": workflow._tree_hash(workflow._snapshot_tree(source)),
    }
    workflow._write_semantic(
        child / "run_manifest.json",
        {
            "schema": workflow.RUN_SCHEMA,
            "schema_version": 1,
            "prefix_run_dir": str(prefix.resolve()),
            "parent_run_dir": str(parent.resolve()),
            "source_run_dir": str(source.resolve()),
            "source_closure": identity["source_closure"],
        },
    )
    return child, prefix, parent, source, identity


def test_frozen_scientific_constants_and_transition_authorities() -> None:
    assert workflow.VERSION == "d0-jacobi-rb-global-dilated-continuation-v2"
    assert workflow.RUN_SCHEMA == "experiment12-d0-jacobi-rb-global-dilated-continuation"
    assert tuple(workflow.STAGES) == (
        "prepare",
        "controls",
        "forward_tail",
        "reverse_complete",
        "report_verify",
    )
    assert workflow.PARENT_SELECTED_UPDATE == 3100
    assert workflow.PATH_ID == 1028864
    assert workflow.FORWARD_ROOT_SEED == 261401
    assert workflow.REVERSE_ROOT_SEED == 261402
    assert workflow.ROW_ORDER == ("zero", "global-plus-1", "source-informed")
    assert workflow.PRACTICAL_RELATIVE_THRESHOLD == pytest.approx(0.01, abs=0.0)
    assert workflow.FORWARD_TRANSITION_COUNT == 1_404_928
    assert workflow.PARENT_PREFIX_TRANSITION_COUNT == 351_232
    assert workflow.IMPORTED_FORWARD_SHARDS == 64
    assert workflow.REVERSE_TRANSITION_COUNT == 16_859_136
    assert workflow.IMPORTED_REVERSE_SHARDS == 1
    assert workflow.V2_CARRIED_ACTIVE_SECONDS == V2_ACTIVE_SECONDS
    assert workflow.ACTIVE_SECONDS_CAP == 22_500.0
    assert tuple(workflow.COMPETING_HYPOTHESES) == (
        "implementation_or_orientation_defect",
        "controller_integrator_or_interface_failure",
        "inadequate_global_architecture_or_parameterization",
        "gain_or_late_time_calibration_failure",
        "useful_short_horizon_signal_but_dynamically_negligible_complete_effect",
        "on_policy_distribution_shift",
        "terminal_reference_prior_mismatch",
        "proxy_gate_misaligned_with_trajectory_objective",
        "exact_backend_cost_or_discrepancy",
        "current_jacobi_rb_strategy_failure",
        "other_evidence_supported_explanation",
    )


def test_default_exact_profile_is_bound_to_authenticated_prefix() -> None:
    assert workflow.DEFAULT_PROFILE_SHA256 == (
        "75ed39fcdc20bb8c675bf9321ae3b31b8fa409370f9d5620f3c9f5b75821fda4"
    )
    profile = workflow._default_profile()
    assert semantic_sha256(profile.to_dict()) == workflow.DEFAULT_PROFILE_SHA256


@pytest.mark.parametrize(
    ("argv_factory", "expected_mode"),
    ((_fresh_argv, "fresh"), (_resume_argv, "resume"), (_verify_argv, "verify")),
)
def test_cli_modes_are_mutually_exclusive_and_valid(
    tmp_path: Path, argv_factory: object, expected_mode: str
) -> None:
    _, mode = _validate(argv_factory(tmp_path))  # type: ignore[operator]
    assert mode == expected_mode


def test_cli_rejects_fresh_noncanonical_stage_and_mode_mixtures(tmp_path: Path) -> None:
    with pytest.raises((workflow.ContinuationArgumentError, SystemExit)):
        _validate(_fresh_argv(tmp_path, stage="controls"))
    mixed = _fresh_argv(tmp_path) + ["--resume-run-dir", str(tmp_path / "child")]
    with pytest.raises((workflow.ContinuationArgumentError, SystemExit)):
        _validate(mixed)
    with pytest.raises((workflow.ContinuationArgumentError, SystemExit)):
        _validate(_verify_argv(tmp_path) + ["--device", "cuda"])


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "../escape",
        "..\\escape",
        ".",
        "..",
        "nested/child",
        "nested\\child",
        "with space",
        "",
    ),
)
def test_fresh_run_name_is_one_safe_stem(tmp_path: Path, unsafe_name: str) -> None:
    argv = _fresh_argv(tmp_path)
    argv[argv.index("--run-name") + 1] = unsafe_name
    with pytest.raises((workflow.ContinuationArgumentError, SystemExit)):
        _validate(argv)


def test_resolved_fresh_run_cannot_escape_runs_root(tmp_path: Path) -> None:
    args, mode = _validate(_fresh_argv(tmp_path))
    paths = workflow._resolve_paths(args, mode=mode)
    assert paths.run_dir.parent == (tmp_path / "runs").resolve()


@pytest.mark.parametrize("safe_name", ("a", "production-v1", "run_01", "run.01"))
def test_fresh_run_name_safe_stems_are_accepted(tmp_path: Path, safe_name: str) -> None:
    argv = _fresh_argv(tmp_path)
    argv[argv.index("--run-name") + 1] = safe_name
    args, mode = _validate(argv)
    paths = workflow._resolve_paths(args, mode=mode)
    assert paths.run_dir.name == safe_name


@pytest.mark.parametrize(
    "forbidden",
    (
        "--gain",
        "--path-id",
        "--checkpoint",
        "--training-seed",
        "--microsteps",
        "--backend",
        "--active-seconds-cap",
        "--safety-multiplier",
    ),
)
def test_cli_exposes_no_scientific_override(tmp_path: Path, forbidden: str) -> None:
    with pytest.raises(SystemExit):
        workflow.parse_args(_fresh_argv(tmp_path) + [forbidden, "1"])


def test_atomic_fresh_initialization_is_structurally_resumable(tmp_path: Path) -> None:
    args, mode = _validate(_fresh_argv(tmp_path))
    assert mode == "fresh"
    paths = workflow._resolve_paths(args, mode=mode)
    run_dir = workflow._initialize_child_atomically(args, paths=paths)
    assert Path(run_dir) == tmp_path / "runs/child"
    assert sorted(path.name for path in Path(run_dir).iterdir()) == [
        "exact_command.txt",
        "resource_ledger.json",
        "run_manifest.json",
        "scientific_config.json",
    ]
    assert not list((tmp_path / "runs").glob(".*.tmp*"))
    before = workflow._snapshot_tree(Path(run_dir))
    with pytest.raises((FileExistsError, workflow.ContinuationArgumentError)):
        workflow._initialize_child_atomically(args, paths=paths)
    assert workflow._snapshot_tree(Path(run_dir)) == before


def _real_v2_roots() -> tuple[Path, Path, Path]:
    return (
        REPOSITORY_ROOT / V2_RELATIVE,
        REPOSITORY_ROOT / V3_RELATIVE,
        REPOSITORY_ROOT / SOURCE_RELATIVE,
    )


_REAL_V2_AVAILABLE = all(path.is_dir() for path in _real_v2_roots())


@pytest.mark.skipif(not _REAL_V2_AVAILABLE, reason="sealed v2 prefix fixture unavailable")
def test_real_v2_prefix_bundle_audit_and_v3_marker_pin_are_read_only(tmp_path: Path) -> None:
    prefix, parent, source = _real_v2_roots()
    before = tuple(workflow._snapshot_tree(root) for root in (prefix, parent, source))
    authority = workflow._verify_v2_prefix_bundle_read_only(
        prefix, parent_run_dir=parent, source_run_dir=source
    )
    after = tuple(workflow._snapshot_tree(root) for root in (prefix, parent, source))
    assert authority["binding_passed"] == 1
    assert authority["prefix_file_count"] == 158
    assert authority["prefix_bytes"] == 2_887_822
    assert authority["manifest_artifact_count"] == 154
    assert authority["checksum_entry_count"] == 155
    assert authority["resource_ledger"]["active_seconds"] == V2_ACTIVE_SECONDS
    assert authority["resource_ledger"]["semantic_sha256"] == V2_RESOURCE_LEDGER_SEMANTIC_SHA256
    storage = authority["values"]["terminal_storage_authority.json"]
    assert storage["semantic_sha256"] == V2_TERMINAL_STORAGE_SEMANTIC_SHA256
    assert authority["forward_health"]["passed"] == 1
    assert authority["reverse_shard_0_health"]["passed"] == 1
    assert authority["reverse_shard_0_health"]["shard_count"] == 1
    assert authority["reverse_shard_0_health"]["transition_count"] == 263_424
    assert authority["reverse_shard_0_state_sha256"] == (
        "45a9413878822f67fd5a86a09b1f0b956d33da2671d4963e0ee39d723ee75126"
    )
    assert after == before
    copied = tmp_path / "v3-parent"
    shutil.copytree(parent, copied)
    marker = copied / "stages/report_verify.json"
    body = marker.read_bytes()
    pinned = workflow.PARENT_REPORT_MARKER_SEMANTIC_SHA256.encode("ascii")
    replacement = (b"0" if pinned[:1] != b"0" else b"1") + pinned[1:]
    tampered = body.replace(pinned, replacement, 1)
    assert tampered != body and len(tampered) == len(body)
    marker.write_bytes(tampered)
    tampered_before = (workflow._snapshot_tree(copied), workflow._snapshot_tree(source))
    with pytest.raises(workflow.ParentBindingError):
        workflow._verify_parent_bundle_read_only(copied, source_run_dir=source)
    assert (workflow._snapshot_tree(copied), workflow._snapshot_tree(source)) == tampered_before


@pytest.mark.skipif(not _REAL_V2_AVAILABLE, reason="sealed v2 prefix fixture unavailable")
def test_prepare_controls_and_forward_noop_preserve_imported_prefix_and_exact_carry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prefix, parent, source = _real_v2_roots()
    external_before = tuple(
        workflow._snapshot_tree(root) for root in (prefix, parent, source)
    )
    real_append = workflow._append_prefix_resource_carry
    append_authorities: list[dict[str, object]] = []

    def append_after_reopen(
        run_dir: Path, prefix_authority: dict[str, object]
    ) -> dict[str, object]:
        capsule = Path(run_dir) / "imports/v2"
        for relative in workflow._V2_CAPSULE_PATHS:
            source_path = prefix / str(relative)
            copied_path = capsule / str(relative)
            assert copied_path.is_file()
            assert _sha256(copied_path) == _sha256(source_path)
            assert not os.path.samestat(source_path.stat(), copied_path.stat())
        assert _sha256(capsule / "resource_ledger.json") == V2_RESOURCE_LEDGER_FILE_SHA256
        assert _sha256(capsule / "terminal_storage_authority.json") == (
            V2_TERMINAL_STORAGE_FILE_SHA256
        )
        append_authorities.append(prefix_authority)
        return real_append(run_dir, prefix_authority)

    monkeypatch.setattr(workflow, "_append_prefix_resource_carry", append_after_reopen)
    assert workflow.main(_fresh_argv(tmp_path)) == 0
    run_dir = tmp_path / "runs/child"
    assert len(append_authorities) == 1
    assert tuple(workflow._snapshot_tree(root) for root in (prefix, parent, source)) == (
        external_before
    )
    assert workflow._scan_forward_chain(run_dir)["first_missing"] == 64
    assert workflow._scan_reverse_chain(run_dir)["first_missing"] == 1
    assert len(list((run_dir / "imports/v2").rglob("*.*"))) == len(
        workflow._V2_CAPSULE_PATHS
    )

    ledger = workflow._validate_resource_ledger(run_dir)
    carry = next(
        event for event in ledger["events"] if event["role"] == "prefix_resource_carry"
    )
    assert carry["attempt"] == 1
    assert carry["elapsed_seconds"] == V2_ACTIVE_SECONDS
    assert carry["peak_cuda_bytes"] == 46_834_176
    assert carry["total_cuda_bytes"] == 8_546_484_224
    linkage = json.dumps(carry["detail"], sort_keys=True)
    for digest in (
        V2_RESOURCE_LEDGER_FILE_SHA256,
        V2_RESOURCE_LEDGER_SEMANTIC_SHA256,
        V2_TERMINAL_STORAGE_FILE_SHA256,
        V2_TERMINAL_STORAGE_SEMANTIC_SHA256,
    ):
        assert digest in linkage
    marker = workflow._read_stage_marker_exact(run_dir, "prepare")
    assert marker["detail"]["prefix_resource_event_id"] == carry["event_id"]
    terminal_pin = workflow._read_json(
        run_dir / "predecessor_binding.json", semantic=True
    )["pinned_terminal_hashes"]["terminal_storage_authority.json"]
    assert terminal_pin == {
        "file_sha256": V2_TERMINAL_STORAGE_FILE_SHA256,
        "semantic_sha256": V2_TERMINAL_STORAGE_SEMANTIC_SHA256,
    }

    operational_before = {
        relative: _sha256(run_dir / relative)
        for relative in workflow._v2_operational_paths()
    }
    args = workflow.parse_args(_fresh_argv(tmp_path))
    reverse_root = run_dir / "reverse/fused_families/same-path-three-row/complete-512"
    imported_state = workflow._load_npz(reverse_root / "shard-0000.npz")["state"]
    presampling_orphan = reverse_root / "shard-0001.npz"
    workflow.atomic_rollout_npz(presampling_orphan, {"state": imported_state})
    with monkeypatch.context() as control_patch:
        control_patch.setattr(
            workflow, "_begin_durable_attempt",
            lambda *_args, **_kwargs: {"started_monotonic": workflow.time.perf_counter()},
        )
        with pytest.raises(workflow.ContinuationIntegrityError, match="exactly 64 forward"):
            workflow._run_controls(run_dir, args)
    presampling_orphan.unlink()
    workflow._run_controls(run_dir, args)
    workflow._run_forward_tail(run_dir, args)
    assert operational_before == {
        relative: _sha256(run_dir / relative)
        for relative in workflow._v2_operational_paths()
    }
    freeze = workflow._read_json(run_dir / "continuation_freeze.json", semantic=True)
    assert freeze["imported_forward_shards_before_seal"] == 64
    assert freeze["imported_reverse_shards_before_seal"] == 1
    forward = workflow._read_json(run_dir / "forward/forward_summary.json", semantic=True)
    assert (forward["imported_shard_count"], forward["generated_child_shard_count"]) == (64, 0)
    assert forward["sampler_called"] == 0

    suffix_archive = reverse_root / "shard-0001.npz"
    workflow.atomic_rollout_npz(suffix_archive, {"state": imported_state})
    suffix = workflow._read_json(reverse_root / "shard-0000.json", semantic=True)
    suffix.pop("semantic_sha256")
    suffix.update(
        shard_index=1,
        input_state_sha256=workflow.rollout_array_sha256(imported_state),
        output_state_sha256=workflow.rollout_array_sha256(imported_state),
        state_file_sha256=_sha256(suffix_archive),
        state_file_size=suffix_archive.stat().st_size,
    )
    workflow._write_semantic(reverse_root / "shard-0001.json", suffix)
    assert workflow._verify_imported_inputs(run_dir)["passed"] == 1
    failure = reverse_root / "shard-0002.failure.json"
    workflow._write_semantic(failure, {"schema": "fixture", "schema_version": 1})
    assert workflow._verify_imported_inputs(run_dir)["passed"] == 1
    failure.unlink()
    orphan = reverse_root / "shard-0002.npz"
    workflow.atomic_rollout_npz(orphan, {"state": imported_state})
    assert workflow._verify_imported_inputs(run_dir)["passed"] == 1
    orphan.unlink()
    for authority_path in (
        reverse_root / "shard-0000.json", run_dir / "resource_ledger.json"
    ):
        original = authority_path.read_bytes()
        if authority_path.name == "resource_ledger.json":
            changed = workflow._read_json(authority_path, semantic=True)
            changed["events"][0]["elapsed_seconds"] = V2_ACTIVE_SECONDS + 1.0
            workflow._write_semantic(authority_path, changed)
        else:
            authority_path.write_bytes(original + b"\n")
        with pytest.raises(workflow.ContinuationIntegrityError):
            workflow._verify_imported_inputs(run_dir)
        authority_path.write_bytes(original)

    ledger_before = _sha256(run_dir / "resource_ledger.json")
    repeated = real_append(run_dir, append_authorities[0])
    assert repeated["event_id"] == carry["event_id"]
    assert _sha256(run_dir / "resource_ledger.json") == ledger_before
    storage = run_dir / "imports/v2/terminal_storage_authority.json"
    storage.write_bytes(storage.read_bytes() + b"\n")
    with pytest.raises(workflow.ContinuationIntegrityError):
        real_append(run_dir, append_authorities[0])


@pytest.mark.skipif(not _REAL_V2_AVAILABLE, reason="sealed v2 prefix fixture unavailable")
@pytest.mark.parametrize(
    "tamper",
    ("resource_ledger", "terminal_storage", "reverse_state", "hardlink"),
)
def test_v2_prefix_tamper_fails_read_only(tmp_path: Path, tamper: str) -> None:
    prefix, parent, source = _real_v2_roots()
    copied = tmp_path / prefix.name
    shutil.copytree(prefix, copied)
    relative = {
        "resource_ledger": "resource_ledger.json",
        "terminal_storage": "terminal_storage_authority.json",
        "reverse_state": "reverse/fused_families/same-path-three-row/complete-512/shard-0000.npz",
        "hardlink": "resource_ledger.json",
    }[tamper]
    target = copied / relative
    if tamper == "hardlink":
        external = tmp_path / "linked-evidence.bin"
        external.write_bytes(target.read_bytes())
        target.unlink()
        try:
            os.link(external, target)
        except OSError as exc:
            pytest.skip(f"hardlinks unavailable: {exc}")
    else:
        body = target.read_bytes()
        target.write_bytes((b"0" if body[:1] != b"0" else b"1") + body[1:])
    before = tuple(workflow._snapshot_tree(root) for root in (copied, parent, source))
    with pytest.raises(workflow.ContinuationIntegrityError):
        workflow._verify_v2_prefix_bundle_read_only(
            copied, parent_run_dir=parent, source_run_dir=source
        )
    assert tuple(
        workflow._snapshot_tree(root) for root in (copied, parent, source)
    ) == before


def test_frozen27_reverse_delta_reconstructs_exact_frozen26_without_writes() -> None:
    before = HISTORICAL_RUNNER.read_bytes()
    result = dict(workflow._audit_frozen26_producer_read_only(HISTORICAL_RUNNER))
    assert result["passed"] == 1
    assert result["occurrence_count"] == 1
    assert result["current_size"] == 387_863
    assert result["current_sha256"] == (
        "9258ad5c49474250b7f150c26fe78fa9db892a602e17d085511d4e39391fd98d"
    )
    assert result["reconstructed_size"] == 387_813
    assert result["reconstructed_sha256"] == (
        "2356ddb38d39e75689ca1193094fc9114660915933235dece67d0b8490e32351"
    )
    assert HISTORICAL_RUNNER.read_bytes() == before


def test_atomic_copy_is_independent_idempotent_and_tamper_closed(tmp_path: Path) -> None:
    source = tmp_path / "external/input.bin"
    destination = tmp_path / "child/input.bin"
    source.parent.mkdir()
    source.write_bytes(b"frozen-input")
    expected = _sha256(source)
    first = dict(
        workflow._copy_bound_file(source, destination, expected_sha256=expected)
    )
    assert destination.read_bytes() == source.read_bytes()
    assert first["samefile"] == 0
    assert not os.path.samestat(source.stat(), destination.stat())
    before_stat = destination.stat()
    second = dict(
        workflow._copy_bound_file(source, destination, expected_sha256=expected)
    )
    after_stat = destination.stat()
    assert second["sha256"] == expected
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    destination.write_bytes(b"tampered")
    with pytest.raises(workflow.ContinuationIntegrityError):
        workflow._copy_bound_file(source, destination, expected_sha256=expected)
    assert destination.read_bytes() == b"tampered"


def test_atomic_copy_rejects_symlink_source_when_supported(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    link = tmp_path / "source-link.bin"
    source.write_bytes(b"bytes")
    try:
        link.symlink_to(source)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    with pytest.raises(workflow.ContinuationIntegrityError):
        workflow._copy_bound_file(link, tmp_path / "child.bin")


def test_orphan_is_archived_byte_exact_before_replay(tmp_path: Path) -> None:
    orphan = tmp_path / "reverse/shard-0004.npz"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"npz-first-crash-evidence")
    original = orphan.read_bytes()
    record = dict(
        workflow._archive_orphan_before_replay(
            orphan,
            tmp_path / "recovery/orphans",
            role="reverse",
            shard_index=4,
            attempt_id="attempt-2",
        )
    )
    archive = tmp_path / str(record["archive_relative_path"])
    assert archive.read_bytes() == original
    assert record["original_sha256"] == hashlib.sha256(original).hexdigest()
    assert not os.path.samestat(orphan.stat(), archive.stat())
    orphan.write_bytes(b"replayed-output")
    assert archive.read_bytes() == original


def test_orphan_archival_keeps_prior_failure_attempts_distinct(tmp_path: Path) -> None:
    failure = tmp_path / "reverse/shard-0004.failure.json"
    failure.parent.mkdir(parents=True)
    failure.write_text('{"attempt": 1}', encoding="utf-8")
    one = dict(
        workflow._archive_orphan_before_replay(
            failure,
            tmp_path / "recovery/orphans",
            role="reverse-failure",
            shard_index=4,
            attempt_id="attempt-1",
        )
    )
    failure.write_text('{"attempt": 2}', encoding="utf-8")
    two = dict(
        workflow._archive_orphan_before_replay(
            failure,
            tmp_path / "recovery/orphans",
            role="reverse-failure",
            shard_index=4,
            attempt_id="attempt-2",
        )
    )
    assert one["archive_relative_path"] != two["archive_relative_path"]
    assert (tmp_path / str(one["archive_relative_path"])).read_text() == '{"attempt": 1}'
    assert (tmp_path / str(two["archive_relative_path"])).read_text() == '{"attempt": 2}'


def test_orphan_replay_mismatch_preserves_archived_and_replayed_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "forward/forward_shards/fresh-main-path"
    root.mkdir(parents=True)
    original = np.full((1, 784), 1 / 784, dtype=np.float64)
    orphan = root / "shard-0016.npz"
    workflow.atomic_rollout_npz(orphan, {"state": original})
    recovery, _ = workflow._archive_replay_evidence(
        tmp_path,
        root=root,
        role="forward",
        shard_index=16,
        attempt_id="attempt-1",
    )
    archived = tmp_path / str(recovery["archive_relative_path"])
    archived_bytes = archived.read_bytes()
    replayed = original.copy()
    replayed[0, 0] += 1e-6
    replayed[0, 1] -= 1e-6
    workflow.atomic_rollout_npz(orphan, {"state": replayed})
    with pytest.raises(workflow.ContinuationIntegrityError, match="orphan.*replay"):
        workflow._verify_orphan_replay_matches(
            replayed_path=orphan,
            archived_path=archived,
        )
    assert archived.read_bytes() == archived_bytes
    assert orphan.read_bytes() != archived_bytes


def test_strict_complete_reverse_health_accepts_only_full_certified_family() -> None:
    final_state, records = _valid_strict_reverse_fixture()
    records[0]["synchronous_replay_performed"] = 0
    records[0]["diagnostics"]["reference"]["schema"] = (
        "d0-jacobi-rb-tangent-fused-v1-deferred-reference-shard"
    )
    records[0]["diagnostics"]["reference"].pop(
        "speculative_attempt_discarded"
    )
    health = workflow._strict_fused_exact_health(
        final_state=final_state, shard_records=records
    )
    expected_active = workflow.REVERSE_TRANSITION_COUNT - 64 * 3 * 8
    assert health == {
        "passed": 1,
        "row_count": 3,
        "shard_count": 64,
        "transition_count": workflow.REVERSE_TRANSITION_COUNT,
        "active_count": expected_active,
        "certified_count": expected_active,
        "certificate_fraction": 1.0,
        "fallback_count": 64 * 3 * 2,
        "forbidden_event_count": 0,
        "maximum_mass_error": 0.0,
        "final_state_nonfinite_count": 0,
        "final_state_negative_count": 0,
    }
    prefix_health = workflow._strict_fused_exact_health(
        final_state=final_state,
        shard_records=records[:1],
        expected_shard_count=workflow.IMPORTED_REVERSE_SHARDS,
    )
    assert prefix_health["shard_count"] == 1
    assert prefix_health["transition_count"] == workflow.REVERSE_SHARD_TRANSITION_COUNT
    with pytest.raises(workflow.ContinuationIntegrityError, match="fused"):
        workflow._strict_fused_exact_health(
            final_state=final_state, shard_records=records[:1]
        )


@pytest.mark.parametrize(
    "tamper",
    (
        "shard_transition_count",
        "diagnostic_transition_count",
        "plan_transition_count",
        "certificate_fraction",
        "forbidden_count",
        "reference_unauthorized",
        "reference_invalid",
        "reference_row_unauthorized",
        "phase_invalid",
        "phase_rms",
        "phase_nonfinite_maximum",
        "controller_nonfinite",
        "row_permutation",
        "row_table_path",
        "plan_sequence",
        "final_nonfinite",
        "final_negative",
        "final_mass",
    ),
)
def test_strict_complete_reverse_health_rejects_every_load_bearing_tamper(
    tamper: str,
) -> None:
    final_state, records = _valid_strict_reverse_fixture()
    first = records[0]
    diagnostics = first["diagnostics"]
    assert isinstance(diagnostics, dict)
    reference = diagnostics["reference"]
    assert isinstance(reference, dict)
    if tamper == "shard_transition_count":
        first["transition_count"] = int(first["transition_count"]) - 1
    elif tamper == "diagnostic_transition_count":
        diagnostics["transition_count"] = int(diagnostics["transition_count"]) - 1
    elif tamper == "plan_transition_count":
        first["execution_plan"]["transition_count"] -= 1
    elif tamper == "certificate_fraction":
        diagnostics["certificate_fraction"] = np.nextafter(1.0, 0.0)
    elif tamper == "forbidden_count":
        diagnostics["forbidden_counts"]["projection_count"] = 1
    elif tamper == "reference_unauthorized":
        reference["unauthorized_count"] = 1
    elif tamper == "reference_invalid":
        reference["invalid_count"] = 1
    elif tamper == "reference_row_unauthorized":
        reference["per_row"][0]["unauthorized_count"] = 1
    elif tamper == "phase_invalid":
        first["per_row_diagnostics"][1]["state_invalid"] = 1
    elif tamper == "phase_rms":
        first["per_row_diagnostics"][1]["score_squared_sum"] = 1.0
    elif tamper == "phase_nonfinite_maximum":
        first["per_row_diagnostics"][1]["score_maximum_absolute"] = math.nan
    elif tamper == "controller_nonfinite":
        first["controller_diagnostics"][1]["score_rms"] = math.inf
    elif tamper == "row_permutation":
        first["row_keys"][0], first["row_keys"][1] = (
            first["row_keys"][1],
            first["row_keys"][0],
        )
    elif tamper == "row_table_path":
        for record in records:
            record["row_table"][0]["canonical_path_id"] = workflow.PATH_ID + 1
    elif tamper == "plan_sequence":
        first["execution_plan"]["sequence"][0] = [511, 0]
    elif tamper == "final_nonfinite":
        final_state[0, 0] = math.nan
    elif tamper == "final_negative":
        final_state[0, 0] = -1.0e-6
    elif tamper == "final_mass":
        final_state[0] *= 0.5
    else:  # pragma: no cover - the parameter table is deliberately exhaustive
        raise AssertionError(tamper)
    with pytest.raises(workflow.ContinuationIntegrityError, match="fused"):
        workflow._strict_fused_exact_health(
            final_state=final_state, shard_records=records
        )


def test_reverse_aggregation_and_derived_evidence_have_exact_order_and_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, expected, records = _write_synthetic_reverse_chain(tmp_path)
    states, milestones, aggregation = workflow._aggregate_reverse_boundaries(
        tmp_path, shard_records=records
    )
    assert states.dtype == np.float64
    assert states.flags.c_contiguous
    assert states.shape == (3, 65, 784)
    assert np.array_equal(states, expected)
    assert milestones.dtype == np.float64
    assert milestones.flags.c_contiguous
    assert milestones.shape == (3, 5, 784)
    assert np.array_equal(milestones, states[:, [0, 16, 32, 48, 64], :])
    assert aggregation["states_shape"] == [3, 65, 784]
    trajectory = workflow._load_npz(
        tmp_path / "reverse/trajectory_shard_boundaries.npz"
    )
    milestone_archive = workflow._load_npz(tmp_path / "reverse/milestones.npz")
    assert np.array_equal(
        trajectory["completed_reverse_steps"],
        np.arange(0, 513, 8, dtype=np.int64),
    )
    assert np.array_equal(
        milestone_archive["completed_reverse_steps"],
        np.asarray(workflow.MILESTONE_STEPS, dtype=np.int64),
    )

    monkeypatch.setattr(workflow, "_load_child_source", lambda _run: source)
    summary, mechanism, outcome = workflow._build_reverse_derived(
        tmp_path,
        states=states,
        milestones=milestones,
        shard_records=records,
    )
    with (tmp_path / "reverse/metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        metric_rows = list(csv.DictReader(handle))
    assert len(metric_rows) == 195
    assert [row["row_key"] for row in metric_rows] == (
        list(workflow.ROW_ORDER) * 65
    )
    assert [int(row["boundary_index"]) for row in metric_rows] == [
        boundary for boundary in range(65) for _ in workflow.ROW_ORDER
    ]
    assert [int(row["completed_reverse_steps"]) for row in metric_rows] == [
        boundary * 8 for boundary in range(65) for _ in workflow.ROW_ORDER
    ]

    assert summary["metric_row_count"] == 195
    assert summary["individual_image_count"] == 30
    assert summary["contact_sheet_count"] == 14
    assert len(summary["images"]) == 30
    assert [
        (row["completed_reverse_steps"], row["row_key"], row["rendering"])
        for row in summary["images"]
    ] == [
        (milestone, row_key, rendering)
        for milestone in workflow.MILESTONE_STEPS
        for row_key in workflow.ROW_ORDER
        for rendering in ("raw", "demixed")
    ]
    assert len(summary["contact_sheets"]) == 14
    image_paths = [row["path"] for row in summary["images"]]
    sheet_paths = [row["path"] for row in summary["contact_sheets"]]
    assert len(set(image_paths)) == 30
    assert len(set(sheet_paths)) == 14
    assert set(image_paths).isdisjoint(sheet_paths)
    for row in (*summary["images"], *summary["contact_sheets"]):
        assert _sha256(tmp_path / row["path"]) == row["file_sha256"]
    assert [
        (
            row["kind"],
            row.get("completed_reverse_steps"),
            row["rendering"],
            row["columns"],
            row["cell_count"],
        )
        for row in summary["contact_sheets"]
    ] == [
        ("milestone", milestone, rendering, 3, 3)
        for milestone in workflow.MILESTONE_STEPS
        for rendering in ("raw", "demixed")
    ] + [
        ("all-milestones", None, "raw", 3, 15),
        ("all-milestones", None, "demixed", 3, 15),
        ("final", None, "raw", 3, 3),
        ("final", None, "demixed", 3, 3),
    ]
    individual = sorted(
        path
        for path in (tmp_path / "images").rglob("*.png")
        if "contact-sheets" not in path.parts
    )
    sheets = sorted((tmp_path / "images/contact-sheets").glob("*.png"))
    assert len(individual) == 30
    assert len(sheets) == 14
    assert all(path.stat().st_size > 0 for path in (*individual, *sheets))
    image_hashes = {
        path.relative_to(tmp_path).as_posix(): _sha256(path)
        for path in (*individual, *sheets)
    }
    workflow._build_reverse_derived(
        tmp_path,
        states=states,
        milestones=milestones,
        shard_records=records,
    )
    assert image_hashes == {
        path.relative_to(tmp_path).as_posix(): _sha256(path)
        for path in (*individual, *sheets)
    }

    assert len(mechanism["on_policy_drift"]) == 65
    assert [
        mechanism["on_policy_drift"][index]["matching_training_quartile"]
        for index in (0, 16, 32, 48, 64)
    ] == [3, 2, 1, 0, 0]
    for row_key in workflow.ROW_ORDER:
        assert [
            mechanism["per_reverse_quarter"][row_key][str(quarter)]["shard_count"]
            for quarter in range(4)
        ] == [16, 16, 16, 16]
    assert outcome["source_control_passed"] == 1
    assert outcome["learned_interpretation_authorized"] == 1
    assert outcome["global_effect_label"] == "global_material_improvement"
    assert outcome["required_next_action"] == "run_stage_e_reference_prior"
    assert outcome["confirmatory_claim"] == 0


def test_reverse_complete_stage_exercises_exact_call_chain_health_and_marker_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "child"
    parent = tmp_path / "parent"
    source_dir = tmp_path / "source"
    parent.mkdir()
    source_dir.mkdir()
    uniform = np.full(workflow.STATE_SIZE, 1.0 / workflow.STATE_SIZE, dtype=np.float64)
    target = np.arange(1, workflow.STATE_SIZE + 1, dtype=np.float64)
    target /= np.sum(target)
    source = SimpleNamespace(
        source_image=target.copy(),
        mixed_target=target.copy(),
        metadata={"lambda_mix": 0.2},
    )
    workflow._write_semantic(
        run_dir / "resource_ledger.json", workflow._initial_resource_ledger()
    )
    workflow.atomic_rollout_npz(
        run_dir / "forward/anchor-step-0511.npz", {"state": uniform}
    )
    scale = workflow.fixed_rendering_scale(target, target, 0.2)
    sequence = tuple(workflow.reverse_suffix_sequence(511))
    workflow._write_semantic(
        run_dir / "continuation_freeze.json",
        {
            "schema": "fixture-freeze",
            "schema_version": 1,
            "sealed": 1,
            "reverse_sequence": [list(item) for item in sequence],
            "row_table": workflow._complete_row_table_authority(),
            "controller_binding": workflow._complete_controller_binding_authority(),
            "rng_binding": workflow._complete_rng_binding_authority(),
            "variant_in_rng_key": 0,
            "checkpoint_state_sha256": workflow.CHECKPOINT_STATE_SHA256,
            "source_target_sha256": workflow.MIXED_TARGET_ARRAY_SHA256,
            "rendering_scale": scale.to_dict(),
            "storage_reserves": {
                "forward_pair_bytes": 1,
                "reverse_pair_bytes": 1,
                "derived_and_terminal_bytes": 1,
            },
        },
    )
    validation = np.tile(
        np.linspace(0.0, 2.0, 14_336, dtype=np.float64), (4, 1)
    )
    workflow.atomic_rollout_npz(
        run_dir / "inputs/calibration/on_policy_validation_calibration.npz",
        {
            "training_means": np.tile(uniform, (4, 1)),
            "training_p95": np.ones(4, dtype=np.float64),
            "validation_sorted_ratios": validation,
            "validation_counts": np.full(4, 14_336, dtype=np.int64),
        },
    )
    _fixture_final, fixture_records = _valid_strict_reverse_fixture()
    reverse_root = (
        run_dir / "reverse/fused_families/same-path-three-row/complete-512"
    )
    imported_state = np.ascontiguousarray(
        np.stack(
            (
                uniform,
                (1.0 - 0.25 / 64.0) * uniform + (0.25 / 64.0) * target,
                (1.0 - 0.90 / 64.0) * uniform + (0.90 / 64.0) * target,
            )
        ),
        dtype=np.float64,
    )
    imported_archive = reverse_root / "shard-0000.npz"
    workflow.atomic_rollout_npz(imported_archive, {"state": imported_state})
    imported_record = copy.deepcopy(fixture_records[0])
    imported_record.update(
        input_state_sha256=workflow.rollout_array_sha256(
            np.repeat(uniform[None, :], 3, axis=0)
        ),
        output_state_sha256=workflow.rollout_array_sha256(imported_state),
        state_file_sha256=_sha256(imported_archive),
        state_file_size=imported_archive.stat().st_size,
        elapsed_seconds=0.01,
    )
    workflow._write_semantic(reverse_root / "shard-0000.json", imported_record)
    imported_bytes = tuple(
        _sha256(reverse_root / f"shard-0000.{suffix}") for suffix in ("json", "npz")
    )
    freeze = workflow._read_json(run_dir / "continuation_freeze.json", semantic=True)
    freeze.pop("semantic_sha256")
    freeze.update(
        imported_reverse_shard_0_json_sha256=imported_bytes[0],
        imported_reverse_shard_0_npz_sha256=imported_bytes[1],
    )
    workflow._write_semantic(run_dir / "continuation_freeze.json", freeze)
    specs = tuple(
        workflow.FusedRowSpec(**{
            "row_key": row["row_key"],
            "canonical_path_id": row["canonical_path_id"],
            "controller_kind": row["controller_kind"],
            "variant": row["variant"],
            "horizon": row["horizon"],
            "gain": row["gain"],
            "controller_binding": row["controller_binding"],
        })
        for row in workflow._complete_row_table_authority()
    )
    controller_binding = workflow._complete_controller_binding_authority()
    controller_bank = object()
    prepared = object()
    reference_factory = object()
    events: list[str] = []
    plans: list[object] = []
    exact_call: dict[str, object] = {}

    monkeypatch.setattr(workflow, "_verify_external_trees", lambda *_args: events.append("external"))
    monkeypatch.setattr(workflow, "_verify_imported_inputs", lambda *_args: events.append("inputs"))
    monkeypatch.setattr(workflow, "_verify_continuation_freeze_exact", lambda root: workflow._read_json(Path(root) / "continuation_freeze.json", semantic=True))
    monkeypatch.setattr(workflow.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(workflow, "_cuda_memory_snapshot", lambda _device: (1, 100))
    monkeypatch.setattr(workflow, "_load_child_source", lambda _run: source)
    monkeypatch.setattr(
        workflow, "_strict_load_global_checkpoint", lambda *_args, **_kwargs: (object(), {})
    )
    monkeypatch.setattr(
        workflow,
        "_build_complete_rows",
        lambda **_kwargs: (specs, controller_bank, controller_binding),
    )
    monkeypatch.setattr(workflow, "_prepared_exact_backend", lambda *_args: prepared)
    monkeypatch.setattr(
        workflow,
        "_exact_reference_factory",
        lambda **_kwargs: reference_factory,
    )

    def fake_fused(initial_state: np.ndarray, **kwargs: object) -> SimpleNamespace:
        exact_call.update(initial_state=initial_state.copy(), **kwargs)
        root = (
            Path(kwargs["output_dir"])
            / "fused_families/same-path-three-row/complete-512"
        )
        scan = workflow._scan_reverse_chain(run_dir)
        assert scan["first_missing"] == 1
        previous = workflow.rollout_array_sha256(imported_state)
        final_state = imported_state.copy()
        callback = kwargs["before_uncommitted_shard"]
        for shard_index, record in enumerate(
            copy.deepcopy(fixture_records[1:]), start=1
        ):
            offset = shard_index * workflow.FUSED_SHARD_PHASES
            shard_sequence = sequence[offset : offset + workflow.FUSED_SHARD_PHASES]
            plan = workflow.FusedShardExecutionPlan(
                shard_index=shard_index,
                sequence=shard_sequence,
                row_count=len(workflow.ROW_ORDER),
                transition_count=(
                    workflow.FUSED_SHARD_PHASES
                    * 2
                    * workflow.MICROSTEPS
                    * len(workflow.ROW_ORDER)
                    * workflow.EDGES_PER_PHASE
                ),
                input_state_sha256=previous,
            )
            callback(plan)
            plans.append(plan)
            progress = (shard_index + 1) / workflow.REVERSE_SHARDS
            final_state = np.ascontiguousarray(
                np.stack(
                    (
                        uniform,
                        (1.0 - 0.25 * progress) * uniform + 0.25 * progress * target,
                        (1.0 - 0.90 * progress) * uniform + 0.90 * progress * target,
                    )
                ),
                dtype=np.float64,
            )
            archive = root / f"shard-{shard_index:04d}.npz"
            workflow.atomic_rollout_npz(archive, {"state": final_state})
            record.update(
                input_state_sha256=previous,
                output_state_sha256=workflow.rollout_array_sha256(final_state),
                state_file_sha256=_sha256(archive),
                state_file_size=archive.stat().st_size,
                elapsed_seconds=0.01,
            )
            workflow._write_semantic(root / f"shard-{shard_index:04d}.json", record)
            previous = workflow.rollout_array_sha256(final_state)
        record = {
            "schema": workflow.FUSED_TANGENT_VERSION + "-reverse-family-result",
            "row_table": workflow._complete_row_table_authority(),
            "final_state_sha256": workflow.rollout_array_sha256(final_state),
            "transition_count": workflow.REVERSE_TRANSITION_COUNT,
            "shard_count": workflow.REVERSE_SHARDS,
        }
        return SimpleNamespace(final_state=final_state, to_record=lambda: record)

    monkeypatch.setattr(workflow, "run_fused_reverse_family", fake_fused)
    original_marker = workflow._stage_marker

    def marker_last(path: Path, stage: str, detail: dict[str, object]) -> None:
        assert stage == "reverse_complete"
        assert not (Path(path) / "stages/reverse_complete.json").exists()
        assert not list((Path(path) / "journals").glob("*.json"))
        assert (Path(path) / "reverse/family_summary.json").is_file()
        assert (Path(path) / "reverse/trajectory_shard_boundaries.npz").is_file()
        assert (Path(path) / "reverse/milestones.npz").is_file()
        assert (Path(path) / "reverse/metrics.csv").is_file()
        assert (Path(path) / "reverse/mechanism.json").is_file()
        assert (Path(path) / "reverse/summary.json").is_file()
        assert (Path(path) / "outcome.json").is_file()
        assert len(list((Path(path) / "images").rglob("*.png"))) == 44
        assert events == ["external", "inputs", "external"]
        events.append("marker")
        original_marker(path, stage, detail)

    monkeypatch.setattr(workflow, "_stage_marker", marker_last)
    args = SimpleNamespace(
        device="cuda", parent_run_dir=parent, source_run_dir=source_dir
    )
    workflow._run_reverse_complete(run_dir, args)

    assert events[-1] == "marker"
    assert len(plans) == 63
    assert [plan.shard_index for plan in plans] == list(range(1, 64))
    assert all(plan.row_count == 3 for plan in plans)
    assert all(plan.transition_count == 263_424 for plan in plans)
    assert np.array_equal(
        exact_call["initial_state"], np.repeat(uniform[None, :], 3, axis=0)
    )
    assert exact_call["sequence"] == sequence
    assert exact_call["family_name"] == "same-path-three-row"
    assert exact_call["segment_name"] == "complete-512"
    assert tuple(spec.row_key for spec in exact_call["row_specs"]) == workflow.ROW_ORDER
    assert [spec.canonical_path_id for spec in exact_call["row_specs"]] == [
        workflow.PATH_ID
    ] * 3
    assert exact_call["controller_bank"] is controller_bank
    assert exact_call["reference_factory"] is reference_factory
    assert exact_call["controller_binding"] == controller_binding
    assert exact_call["rng_binding"] == {
        "root_seed": workflow.REVERSE_ROOT_SEED,
        "stream_role": workflow.REVERSE_STREAM_ROLE,
        "canonical_path_id": workflow.PATH_ID,
    }
    assert exact_call["label"] == 3
    assert exact_call["microsteps"] == 2
    assert exact_call["device"] == workflow.torch.device("cuda")
    assert exact_call["capture_coordinates"] == workflow.CAPTURE_COORDINATES
    assert exact_call["reference_contract"] == "certified_exact"
    assert workflow._scan_reverse_chain(run_dir)["first_missing"] == 64
    assert imported_bytes == tuple(
        _sha256(reverse_root / f"shard-0000.{suffix}") for suffix in ("json", "npz")
    )
    family = workflow._read_json(run_dir / "reverse/family_summary.json", semantic=True)
    assert family["strict_exact_health"]["transition_count"] == (
        workflow.REVERSE_TRANSITION_COUNT
    )
    assert family["rng_binding_sha256"] == (
        "43b69f7eee068bb35f28ac578939a429832de6c00d4815ff005a58562830efd3"
    )
    outcome = workflow._read_json(run_dir / "outcome.json", semantic=True)
    assert outcome["source_control_passed"] == 1
    assert outcome["global_effect_label"] == "global_material_improvement"
    assert outcome["required_next_action"] == "run_stage_e_reference_prior"
    ledger = workflow._read_json(run_dir / "resource_ledger.json", semantic=True)
    assert [event["role"] for event in ledger["events"]] == [
        "reverse_complete",
        "reverse_postprocess",
    ]
    assert (run_dir / "stages/reverse_complete.json").is_file()


@pytest.mark.parametrize(
    "tamper", ("input_hash", "output_hash", "file_hash", "state_mass", "state_shape")
)
def test_reverse_aggregation_rejects_chain_archive_and_state_tamper(
    tmp_path: Path, tamper: str
) -> None:
    _source, _expected, records = _write_synthetic_reverse_chain(tmp_path)
    index = 17
    archive = (
        tmp_path
        / "reverse/fused_families/same-path-three-row/complete-512"
        / f"shard-{index:04d}.npz"
    )
    if tamper == "input_hash":
        records[index]["input_state_sha256"] = "0" * 64
    elif tamper == "output_hash":
        records[index]["output_state_sha256"] = "0" * 64
    elif tamper == "file_hash":
        records[index]["state_file_sha256"] = "0" * 64
    else:
        state = workflow._load_npz(archive)["state"]
        if tamper == "state_mass":
            state[0] *= 0.5
        elif tamper == "state_shape":
            state = state[:2]
        workflow.atomic_rollout_npz(archive, {"state": state})
        records[index]["state_file_sha256"] = _sha256(archive)
        records[index]["output_state_sha256"] = workflow.rollout_array_sha256(state)
    with pytest.raises(workflow.ContinuationIntegrityError, match="raw boundary"):
        workflow._aggregate_reverse_boundaries(tmp_path, shard_records=records)


@pytest.mark.parametrize(
    "tamper",
    (
        "schema",
        "scheduler_version",
        "label",
        "controller_binding_sha256",
        "rng_binding_sha256",
        "sequence_start",
        "sequence_end",
        "sequence_sha256",
        "unexpected_reference_contract",
        "reference_schema",
        "reference_root_seed",
        "reference_stream_role",
        "reference_rng_namespace",
        "reference_variant",
        "reference_needs_replay",
        "replay_schema_mismatch",
        "replay_discard_mismatch",
    ),
)
def test_complete_reverse_raw_identity_tamper_fails_closed(
    tamper: str,
) -> None:
    final_state, records = _valid_strict_reverse_fixture()
    record = records[7]
    reference = record["diagnostics"]["reference"]
    if tamper == "schema":
        record["schema"] = "candidate-schema"
    elif tamper == "scheduler_version":
        record["scheduler_version"] = "changed-scheduler"
    elif tamper == "label":
        record["label"] = 4
    elif tamper == "controller_binding_sha256":
        record["controller_binding_sha256"] = "0" * 64
    elif tamper == "rng_binding_sha256":
        record["rng_binding_sha256"] = "0" * 64
    elif tamper == "sequence_start":
        record["sequence_start"] = [0, 0]
    elif tamper == "sequence_end":
        record["sequence_end"] = [0, 0]
    elif tamper == "sequence_sha256":
        record["sequence_sha256"] = "0" * 64
    elif tamper == "unexpected_reference_contract":
        record["reference_contract"] = "certified_exact"
    elif tamper == "reference_schema":
        reference["schema"] = "candidate-reference"
    elif tamper == "reference_root_seed":
        reference["root_seed"] = workflow.REVERSE_ROOT_SEED + 1
    elif tamper == "reference_stream_role":
        reference["stream_role"] = "other-stream"
    elif tamper == "reference_rng_namespace":
        reference["rng_namespace"] = "other-namespace"
    elif tamper == "reference_variant":
        reference["variant_in_rng_key"] = 1
    elif tamper == "reference_needs_replay":
        reference["needs_synchronous_replay"] = 1
    elif tamper == "replay_schema_mismatch":
        record["synchronous_replay_performed"] = 0
    elif tamper == "replay_discard_mismatch":
        reference["speculative_attempt_discarded"] = 0
    else:  # pragma: no cover - parameter table is exhaustive
        raise AssertionError(tamper)
    with pytest.raises(workflow.ContinuationIntegrityError, match="fused"):
        workflow._strict_fused_exact_health(
            final_state=final_state,
            shard_records=records,
            row_count=len(workflow.ROW_ORDER),
        )


@pytest.mark.parametrize("tamper", ("family_name", "segment_name", "state_file_size"))
def test_reverse_chain_scanner_rejects_exact_record_or_file_binding_tamper(
    tmp_path: Path, tamper: str
) -> None:
    _source, _expected, _records = _write_synthetic_reverse_chain(tmp_path)
    root = tmp_path / "reverse/fused_families/same-path-three-row/complete-512"
    record_path = root / "shard-0007.json"
    record = workflow._read_json(record_path, semantic=True)
    if tamper == "family_name":
        record["family_name"] = "other-family"
    elif tamper == "segment_name":
        record["segment_name"] = "other-segment"
    else:
        record["state_file_size"] = int(record["state_file_size"]) + 1
    workflow._write_semantic(record_path, record)
    with pytest.raises(workflow.ContinuationIntegrityError, match="reverse"):
        workflow._scan_reverse_chain(tmp_path)


@pytest.mark.parametrize("wrong", ("states", "milestones"))
def test_reverse_derived_rejects_wrong_aggregate_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, wrong: str
) -> None:
    source, _expected, records = _write_synthetic_reverse_chain(tmp_path)
    states, milestones, _record = workflow._aggregate_reverse_boundaries(
        tmp_path, shard_records=records
    )
    monkeypatch.setattr(workflow, "_load_child_source", lambda _run: source)
    if wrong == "states":
        states = states[:, :-1]
    else:
        milestones = milestones[:, :-1]
    with pytest.raises(workflow.ContinuationIntegrityError, match="derived input shape"):
        workflow._build_reverse_derived(
            tmp_path,
            states=states,
            milestones=milestones,
            shard_records=records,
        )


@pytest.mark.parametrize(
    ("source_error", "label", "authorized"),
    (
        (0.98, "source_informative", 1),
        (0.99, "source_informative", 1),
        (0.995, "source_positive_small_uninformative", 0),
        (1.0, "source_adverse", 0),
        (1.01, "source_adverse", 0),
        (float("nan"), "invalid_objective", 0),
    ),
)
def test_source_gate_uses_one_percent_practical_scale(
    source_error: float, label: str, authorized: int
) -> None:
    outcome = _classification(
        zero=1.0, global_error=0.98, source=source_error
    )
    assert _field(outcome, "source_effect_label", "source_label") == label
    assert int(_field(outcome, "learned_interpretation_authorized")) == authorized


def test_source_gate_rejects_nonpositive_zero_objective() -> None:
    outcome = _classification(zero=0.0, global_error=0.0, source=0.0)
    assert _field(outcome, "source_effect_label", "source_label") == "invalid_objective"
    assert int(_field(outcome, "learned_interpretation_authorized")) == 0


@pytest.mark.parametrize(
    ("global_error", "intermediate", "label", "action"),
    (
        (
            0.98,
            {},
            "global_material_improvement",
            "run_stage_e_reference_prior",
        ),
        (
            0.995,
            {},
            "global_positive_small",
            "run_one_new_independent_path_replication",
        ),
        (
            1.01,
            {128: 0.01},
            "global_early_help_late_adverse",
            "run_predeclared_time_window_schedule_ablation",
        ),
        (
            1.01,
            {128: 0.005},
            "global_complete_adverse",
            "run_conventional_mnist_ddpm_reconstruction_sanity",
        ),
        (
            1.01,
            {},
            "global_complete_adverse",
            "run_conventional_mnist_ddpm_reconstruction_sanity",
        ),
    ),
)
def test_outcome_actions_are_exact_and_intermediate_help_must_be_material(
    global_error: float,
    intermediate: dict[int, float],
    label: str,
    action: str,
) -> None:
    outcome = _classification(
        zero=1.0,
        global_error=global_error,
        source=0.5,
        intermediate_relative=intermediate,
    )
    assert outcome["global_effect_label"] == label
    assert outcome["required_next_action"] == action


def test_weak_source_takes_precedence_over_learned_label() -> None:
    outcome = _classification(zero=1.0, global_error=0.5, source=0.995)
    assert int(outcome["learned_interpretation_authorized"]) == 0
    assert outcome["required_next_action"] == "audit_oracle_controller_composition"


@pytest.mark.parametrize("active", (0.0, 21_899.999, 21_900.0))
def test_report_reserve_is_prepaid_before_ledger_freeze(active: float) -> None:
    admission = dict(
        workflow._prepaid_report_admission(active_seconds=active, now_monotonic=100.0)
    )
    assert admission["admitted"] == 1
    assert admission["charged_seconds"] == pytest.approx(600.0, abs=0.0)
    assert admission["active_seconds_after_charge"] == pytest.approx(
        active + 600.0, abs=0.0
    )
    assert admission["deadline_monotonic"] == pytest.approx(700.0, abs=0.0)


def test_report_reserve_rejects_one_representable_amount_over_cap() -> None:
    just_over = np.nextafter(
        workflow.ACTIVE_SECONDS_CAP - workflow.REPORT_RESERVE_SECONDS,
        float("inf"),
    )
    admission = dict(
        workflow._prepaid_report_admission(
            active_seconds=float(just_over), now_monotonic=1.0
        )
    )
    assert admission["admitted"] == 0
    assert admission["charged_seconds"] == 0.0


def test_report_deadline_includes_manifest_terminalization_and_final_audit() -> None:
    admission = workflow._prepaid_report_admission(
        active_seconds=0.0, now_monotonic=10.0
    )
    workflow._require_report_deadline(admission, now_monotonic=610.0)
    with pytest.raises(workflow.ResourceBoundaryError):
        workflow._require_report_deadline(
            admission, now_monotonic=float(np.nextafter(610.0, float("inf")))
        )


def test_report_overrun_marks_prepaid_event_and_ledger_breached(tmp_path: Path) -> None:
    run_dir = tmp_path / "child"; run_dir.mkdir()
    workflow._write_semantic(run_dir / "resource_ledger.json", workflow._initial_resource_ledger())
    admission = workflow._prepaid_report_admission(active_seconds=0.0, now_monotonic=10.0)
    journal = workflow._begin_durable_attempt(run_dir, role="report_verify", attempt=1, durable_elapsed_seconds=0.0, now_monotonic=10.0)
    workflow._finish_durable_attempt(run_dir, journal, durable_elapsed_seconds=0.0, invocation_wall_seconds=600.0, detail={"prepaid_report_admission": admission})
    workflow._record_report_overrun(run_dir)
    ledger = workflow._validate_resource_ledger(run_dir)
    assert ledger["limits_passed"] == 0
    assert ledger["breached_limits"] == ["report_deadline"]
    assert ledger["events"][-1]["breaches"] == ["report_deadline"]


def test_forward_and_reverse_admission_use_remaining_objective_and_report_reserve() -> None:
    allowed = workflow._resource_projection(
        active_seconds=V2_ACTIVE_SECONDS,
        current_attempt_wall=0.0,
        remaining_forward_shards=0,
        remaining_reverse_shards=63,
        forward_next_seconds=0.0,
        reverse_next_seconds=303.34827828,
        postprocess_seconds=30.0,
        report_seconds=600.0,
    )
    assert allowed.projected_active_seconds == pytest.approx(
        21_686.62469454, rel=0.0, abs=1e-9
    )
    assert allowed.admitted
    rejected = workflow._resource_projection(
        active_seconds=float(np.nextafter(workflow.ACTIVE_SECONDS_CAP, float("inf"))),
        current_attempt_wall=0.0,
        remaining_forward_shards=0,
        remaining_reverse_shards=0,
        forward_next_seconds=0.0,
        reverse_next_seconds=303.34827828,
        postprocess_seconds=0.0,
        report_seconds=0.0,
    )
    assert not rejected.admitted


def test_durable_journal_reconciliation_is_idempotent(tmp_path: Path) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    workflow._write_semantic(
        run_dir / "resource_ledger.json", workflow._initial_resource_ledger()
    )
    journal = workflow._begin_durable_attempt(
        run_dir,
        role="forward_tail",
        attempt=1,
        durable_elapsed_seconds=0.0,
        now_utc="2026-08-14T00:00:00+00:00",
        now_monotonic=10.0,
    )
    one = workflow._reconcile_durable_attempt(
        run_dir,
        journal,
        durable_elapsed_seconds=3.0,
        now_utc="2026-08-14T00:00:04+00:00",
        now_monotonic=14.0,
    )
    two = workflow._reconcile_durable_attempt(
        run_dir,
        journal,
        durable_elapsed_seconds=3.0,
        now_utc="2026-08-14T00:00:04+00:00",
        now_monotonic=14.0,
    )
    assert one["event_id"] == two["event_id"]
    ledger = workflow._read_json(run_dir / "resource_ledger.json", semantic=True)
    assert len(ledger["events"]) == 1
    assert ledger["persisted_storage_bytes"] == workflow._directory_bytes(run_dir)


def test_crash_reconciliation_updates_resource_breach_authority(tmp_path: Path) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    workflow._write_semantic(
        run_dir / "resource_ledger.json", workflow._initial_resource_ledger()
    )
    journal = workflow._begin_durable_attempt(
        run_dir,
        role="forward_tail",
        attempt=1,
        durable_elapsed_seconds=0.0,
        now_utc="2026-08-14T00:00:00+00:00",
        now_monotonic=0.0,
    )
    with pytest.raises(workflow.ResourceBoundaryError, match="resource.*breach|boundary crossed"):
        workflow._reconcile_durable_attempt(
            run_dir,
            journal,
            durable_elapsed_seconds=0.0,
            now_utc="2026-08-14T06:15:01+00:00",
            now_monotonic=22_501.0,
        )
    ledger = workflow._read_json(run_dir / "resource_ledger.json", semantic=True)
    assert ledger["active_seconds"] == pytest.approx(22_506.0, abs=0.0)
    assert ledger["limits_passed"] == 0
    assert "active_seconds" in ledger["breached_limits"]
    assert len(ledger["events"]) == 1
    assert ledger["events"][0]["elapsed_seconds"] == pytest.approx(22_506.0, abs=0.0)
    assert not workflow._journal_path(run_dir, str(journal["attempt_id"])).exists()
    with pytest.raises(workflow.ResourceBoundaryError, match="resource.*breach|boundary crossed"):
        workflow._reconcile_durable_attempt(
            run_dir,
            journal,
            durable_elapsed_seconds=0.0,
            now_utc="2026-08-14T06:15:01+00:00",
            now_monotonic=22_501.0,
        )
    ledger_again = workflow._read_json(
        run_dir / "resource_ledger.json", semantic=True
    )
    assert len(ledger_again["events"]) == 1


def test_normal_durable_finish_uses_max_wall_or_new_commit_and_is_idempotent(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    workflow._write_semantic(
        run_dir / "resource_ledger.json", workflow._initial_resource_ledger()
    )
    journal = workflow._begin_durable_attempt(
        run_dir,
        role="reverse_complete",
        attempt=1,
        durable_elapsed_seconds=10.0,
        now_utc="2026-08-14T00:00:00+00:00",
        now_monotonic=100.0,
    )
    event = workflow._finish_durable_attempt(
        run_dir,
        journal,
        durable_elapsed_seconds=17.0,
        invocation_wall_seconds=5.0,
        peak_cuda_bytes=40,
        total_cuda_bytes=100,
    )
    assert event["elapsed_seconds"] == pytest.approx(7.0, abs=0.0)
    assert not (run_dir / f"attempts/{journal['attempt_id']}.json").exists()
    again = workflow._finish_durable_attempt(
        run_dir,
        journal,
        durable_elapsed_seconds=17.0,
        invocation_wall_seconds=20.0,
        peak_cuda_bytes=40,
        total_cuda_bytes=100,
    )
    assert again["event_id"] == event["event_id"]
    ledger = workflow._read_json(run_dir / "resource_ledger.json", semantic=True)
    assert len(ledger["events"]) == 1
    assert ledger["active_seconds"] == pytest.approx(7.0, abs=0.0)


def test_caught_failed_attempt_is_charged_and_disclosed(tmp_path: Path) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    workflow._write_semantic(
        run_dir / "resource_ledger.json", workflow._initial_resource_ledger()
    )
    journal = workflow._begin_durable_attempt(
        run_dir,
        role="postprocess",
        attempt=2,
        durable_elapsed_seconds=0.0,
        now_utc="2026-08-14T00:00:00+00:00",
        now_monotonic=0.0,
    )
    event = workflow._finish_durable_attempt(
        run_dir,
        journal,
        durable_elapsed_seconds=0.0,
        invocation_wall_seconds=3.5,
        failed=True,
        detail={"failure_type": "ValueError"},
    )
    assert event["failed"] == 1
    assert event["elapsed_seconds"] == pytest.approx(3.5, abs=0.0)
    assert event["detail"] == {"failure_type": "ValueError"}


def test_cuda_boundary_preserves_completed_event_then_stops(tmp_path: Path) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    workflow._write_semantic(
        run_dir / "resource_ledger.json", workflow._initial_resource_ledger()
    )
    journal = workflow._begin_durable_attempt(
        run_dir,
        role="forward_tail",
        attempt=1,
        durable_elapsed_seconds=0.0,
        now_utc="2026-08-14T00:00:00+00:00",
        now_monotonic=0.0,
    )
    with pytest.raises(workflow.ResourceBoundaryError, match="cuda_memory"):
        workflow._finish_durable_attempt(
            run_dir,
            journal,
            durable_elapsed_seconds=1.0,
            invocation_wall_seconds=1.0,
            peak_cuda_bytes=80,
            total_cuda_bytes=100,
        )
    ledger = workflow._read_json(run_dir / "resource_ledger.json", semantic=True)
    assert ledger["limits_passed"] == 0
    assert ledger["breached_limits"] == ["cuda_memory"]
    assert len(ledger["events"]) == 1


def test_caught_precommit_reconciliation_persists_live_cuda_peak(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "child"; run_dir.mkdir()
    workflow._write_semantic(run_dir / "resource_ledger.json", workflow._initial_resource_ledger())
    workflow._begin_durable_attempt(run_dir, role="forward_tail", attempt=1, durable_elapsed_seconds=0.0)
    monkeypatch.setattr(workflow, "_scan_forward_chain", lambda _run: {"records": []})
    monkeypatch.setattr(workflow, "_cuda_memory_snapshot", lambda _device: (80, 100))
    with pytest.raises(workflow.ResourceBoundaryError): workflow._reconcile_live_stage_journals(run_dir)
    ledger = workflow._validate_resource_ledger(run_dir)
    assert (ledger["peak_cuda_bytes"], ledger["total_cuda_bytes"]) == (80, 100)
    assert "cuda_memory" in ledger["breached_limits"]


def test_resume_validation_hard_crash_reconciles_then_retries(tmp_path: Path) -> None:
    run_dir = tmp_path / "child"; run_dir.mkdir()
    workflow._write_semantic(run_dir / "resource_ledger.json", workflow._initial_resource_ledger())
    now = workflow._utc_now(); monotonic = workflow.time.perf_counter()
    stale = workflow._begin_durable_attempt(run_dir, role="resume_validation", attempt=1, durable_elapsed_seconds=0.0, now_utc=now, now_monotonic=monotonic)
    recovered = workflow._reconcile_live_stage_journals(run_dir)
    assert recovered[0]["durable_attempt_id"] == stale["attempt_id"]
    retry = workflow._begin_durable_attempt(run_dir, role="resume_validation", attempt=2, durable_elapsed_seconds=0.0)
    assert retry["attempt"] == 2


def test_main_packages_resume_reconciliation_cap_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir, prefix, parent, source = (tmp_path / name for name in ("child", "prefix", "parent", "source"))
    for root in (run_dir, prefix, parent, source): root.mkdir()
    workflow._write_semantic(run_dir / "resource_ledger.json", workflow._initial_resource_ledger())
    args = SimpleNamespace(prefix_run_dir=prefix, parent_run_dir=parent, source_run_dir=source); paths = SimpleNamespace(run_dir=run_dir, prefix_run_dir=prefix, parent_run_dir=parent, source_run_dir=source)
    monkeypatch.setattr(workflow, "parse_args", lambda _argv: args); monkeypatch.setattr(workflow, "_resolve_mode", lambda _args: "resume"); monkeypatch.setattr(workflow, "_resolve_paths", lambda *_args, **_kwargs: paths)
    monkeypatch.setattr(workflow, "_load_child_identity_read_only", lambda *_args: {}); monkeypatch.setattr(workflow, "_verify_resume_compatibility_read_only", lambda *_args, **_kwargs: {})
    calls = {"count": 0}
    def reconcile(_run: Path) -> list[dict[str, object]]:
        calls["count"] += 1
        if calls["count"] == 1: raise workflow.ResourceBoundaryError("reconciled resource boundary crossed")
        return []
    monkeypatch.setattr(workflow, "_reconcile_live_stage_journals", reconcile)
    assert workflow.main([]) == 1
    capture = workflow._read_json(run_dir / "failure_capture.json", semantic=True)
    assert capture["failure_domain"] == "resource_boundary"
    assert workflow._read_json(run_dir / "verification.json", semantic=True)["terminal_kind"] == "failure"


def test_main_resume_validation_charges_precompatibility_wall(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir, prefix, parent, source = (tmp_path / name for name in ("child", "prefix", "parent", "source"))
    for root in (run_dir, prefix, parent, source): root.mkdir()
    workflow._write_semantic(run_dir / "resource_ledger.json", workflow._initial_resource_ledger())
    args = SimpleNamespace(prefix_run_dir=prefix, parent_run_dir=parent, source_run_dir=source); paths = SimpleNamespace(run_dir=run_dir, prefix_run_dir=prefix, parent_run_dir=parent, source_run_dir=source)
    monkeypatch.setattr(workflow, "parse_args", lambda _argv: args); monkeypatch.setattr(workflow, "_resolve_mode", lambda _args: "resume"); monkeypatch.setattr(workflow, "_resolve_paths", lambda *_args, **_kwargs: paths)
    clock = iter((100.0, 110.0, 120.0)); monkeypatch.setattr(workflow.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(workflow, "_load_child_identity_read_only", lambda *_args: {}); monkeypatch.setattr(workflow, "_verify_resume_compatibility_read_only", lambda *_args, **_kwargs: {}); monkeypatch.setattr(workflow, "_reconcile_live_stage_journals", lambda *_args: [])
    monkeypatch.setattr(workflow, "_run_requested_stages", lambda *_args: None)
    assert workflow.main([]) == 0
    event = workflow._validate_resource_ledger(run_dir)["events"][0]
    assert event["role"] == "resume_validation" and event["elapsed_seconds"] == 25.0


def test_main_does_not_reset_timer_for_preexisting_abandoned_event(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir, prefix, parent, source = (tmp_path / name for name in ("child", "prefix", "parent", "source"))
    for root in (run_dir, prefix, parent, source): root.mkdir()
    workflow._write_semantic(run_dir / "resource_ledger.json", workflow._initial_resource_ledger())
    now = workflow._utc_now(); monotonic = workflow.time.perf_counter()
    stale = workflow._begin_durable_attempt(run_dir, role="forward_tail", attempt=1, durable_elapsed_seconds=0.0, now_utc=now, now_monotonic=monotonic)
    workflow._reconcile_durable_attempt(run_dir, stale, durable_elapsed_seconds=0.0, now_utc=now, now_monotonic=monotonic)
    workflow._write_semantic(workflow._journal_path(run_dir, stale["attempt_id"]), {key: value for key, value in stale.items() if key != "semantic_sha256"})
    args = SimpleNamespace(prefix_run_dir=prefix, parent_run_dir=parent, source_run_dir=source); paths = SimpleNamespace(run_dir=run_dir, prefix_run_dir=prefix, parent_run_dir=parent, source_run_dir=source)
    monkeypatch.setattr(workflow, "parse_args", lambda _argv: args); monkeypatch.setattr(workflow, "_resolve_mode", lambda _args: "resume"); monkeypatch.setattr(workflow, "_resolve_paths", lambda *_args, **_kwargs: paths)
    clock = iter((100.0, 110.0, 120.0, 120.0, 120.0)); monkeypatch.setattr(workflow.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(workflow, "_load_child_identity_read_only", lambda *_args: {}); monkeypatch.setattr(workflow, "_verify_resume_compatibility_read_only", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(workflow, "_run_requested_stages", lambda *_args: None)
    assert workflow.main([]) == 0
    ledger = workflow._validate_resource_ledger(run_dir)
    assert ledger["events"][-1]["role"] == "resume_validation" and ledger["events"][-1]["elapsed_seconds"] == 25.0


def test_main_recovers_probe_from_hard_kill_before_child_journal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir, prefix, parent, source = (tmp_path / name for name in ("child", "prefix", "parent", "source"))
    for root in (run_dir, prefix, parent, source): root.mkdir()
    workflow._write_semantic(run_dir / "resource_ledger.json", workflow._initial_resource_ledger())
    probe = run_dir.parent / f".{run_dir.name}.resume-probe.json"
    workflow._write_semantic(probe, {**workflow._resume_probe_binding(run_dir, {}, prefix, parent, source), "ledger_event_ids": [], "started_at": workflow._utc_now(), "started_monotonic": 100.0, "owned_monotonic": 100.0})
    args = SimpleNamespace(prefix_run_dir=prefix, parent_run_dir=parent, source_run_dir=source); paths = SimpleNamespace(run_dir=run_dir, prefix_run_dir=prefix, parent_run_dir=parent, source_run_dir=source)
    monkeypatch.setattr(workflow, "parse_args", lambda _argv: args); monkeypatch.setattr(workflow, "_resolve_mode", lambda _args: "resume"); monkeypatch.setattr(workflow, "_resolve_paths", lambda *_args, **_kwargs: paths)
    clock = iter((110.0, 110.0, 120.0)); monkeypatch.setattr(workflow.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(workflow, "_load_child_identity_read_only", lambda *_args: {}); monkeypatch.setattr(workflow, "_verify_resume_compatibility_read_only", lambda *_args, **_kwargs: {}); monkeypatch.setattr(workflow, "_reconcile_live_stage_journals", lambda *_args: []); monkeypatch.setattr(workflow, "_run_requested_stages", lambda *_args: None)
    assert workflow.main([]) == 0 and not probe.exists()
    assert workflow._validate_resource_ledger(run_dir)["events"][0]["elapsed_seconds"] == 25.0


def test_resume_probe_uses_utc_across_monotonic_reboot() -> None:
    probe = {"started_at": "2026-08-14T00:00:00+00:00", "started_monotonic": 1000.0}
    assert workflow._resume_probe_elapsed(probe, now_utc="2026-08-14T00:00:30+00:00", now_monotonic=2.0) == 30.0


def test_resume_probe_is_bound_to_child_generation_and_locators(tmp_path: Path) -> None:
    child, prefix, parent, source = (tmp_path / name for name in ("child", "prefix", "parent", "source"))
    for root in (child, prefix, parent, source): root.mkdir()
    workflow._write_semantic(child / "resource_ledger.json", workflow._initial_resource_ledger())
    identity = {"run_manifest_semantic_sha256": "a" * 64, "scientific_config_semantic_sha256": "b" * 64, "source_closure_sha256": "c" * 64}
    path, _ = workflow._begin_resume_probe(child, identity, prefix, parent, source)
    with pytest.raises(workflow.ContinuationIntegrityError, match="ownership changed"):
        workflow._begin_resume_probe(child, {**identity, "run_manifest_semantic_sha256": "d" * 64}, prefix, parent, source)
    path.unlink()


def test_resume_probe_survives_process_control_interrupt_after_ownership(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir, prefix, parent, source = (tmp_path / name for name in ("child", "prefix", "parent", "source"))
    for root in (run_dir, prefix, parent, source): root.mkdir()
    workflow._write_semantic(run_dir / "resource_ledger.json", workflow._initial_resource_ledger())
    identity = {"run_manifest_semantic_sha256": "a" * 64, "scientific_config_semantic_sha256": "b" * 64, "source_closure_sha256": "c" * 64}
    args = SimpleNamespace(prefix_run_dir=prefix, parent_run_dir=parent, source_run_dir=source); paths = SimpleNamespace(run_dir=run_dir, prefix_run_dir=prefix, parent_run_dir=parent, source_run_dir=source)
    monkeypatch.setattr(workflow, "parse_args", lambda _argv: args); monkeypatch.setattr(workflow, "_resolve_mode", lambda _args: "resume"); monkeypatch.setattr(workflow, "_resolve_paths", lambda *_args, **_kwargs: paths); monkeypatch.setattr(workflow, "_load_child_identity_read_only", lambda *_args: identity)
    calls = {"count": 0}
    def verify(*_args: object, **_kwargs: object) -> None:
        calls["count"] += 1
        if calls["count"] == 2: raise KeyboardInterrupt
    monkeypatch.setattr(workflow, "_verify_resume_compatibility_read_only", verify)
    with pytest.raises(KeyboardInterrupt): workflow.main([])
    assert (run_dir.parent / f".{run_dir.name}.resume-probe.json").is_file()


def test_failure_domain_verifier_does_not_require_failed_predicate_to_pass(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "failure"
    run_dir.mkdir()
    failure = workflow._semantic(
        {
            "schema": workflow.VERSION + "-terminal-failure",
            "schema_version": 1,
            "failure_domain": "parent_binding",
            "expected": {"sha256": "0" * 64},
            "observed": {"sha256": "1" * 64},
            "predicates_passed": [],
            "learned_interpretation_authorized": 0,
        }
    )
    workflow._write_semantic(run_dir / "terminal_failure.json", failure)
    result = workflow._verify_failure_domain_read_only(
        run_dir,
        failure,
        require_terminal_bundle=False,
    )
    assert result["passed"] == 1
    assert result["documented_failed_predicate"] == "parent_binding"
    assert result["learned_interpretation_authorized"] == 0


def test_source_control_failure_verifies_raw_evidence_but_blocks_interpretation(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "reverse/shard-0000.npz"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw-complete-control-evidence")
    failure = {
        "failure_domain": "composition_control",
        "raw_objective_completed": 1,
        "available_raw_paths": [raw.relative_to(tmp_path).as_posix()],
        "learned_interpretation_authorized": 0,
        "expected": {"relative_source_minimum": 0.01},
        "observed": {"relative_source": 0.005},
    }
    result = workflow._verify_failure_domain_read_only(
        tmp_path,
        failure,
        require_terminal_bundle=False,
    )
    assert result["passed"] == 1
    assert result["raw_paths_verified"] == 1
    assert result["learned_interpretation_authorized"] == 0


def test_terminal_fixed_point_is_deterministic_and_final_bytes_exact(tmp_path: Path) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    (run_dir / "payload.bin").write_bytes(b"evidence")
    kwargs = {
        "run_dir": run_dir,
        "terminal_kind": "success",
        "terminalized_at": "2026-08-14T00:00:00+00:00",
        "resource_ledger_semantic_sha256": "a" * 64,
        "verification_body": {
            "schema": workflow.VERSION + "-verification",
            "schema_version": 1,
            "terminal_kind": "success",
            "passed": 1,
        },
        "stage_body": {
            "schema": workflow.VERSION + "-stage",
            "schema_version": 1,
            "stage": "report_verify",
            "passed": 1,
            "completed_at": "2026-08-14T00:00:00+00:00",
            "detail": {"fixture": 1},
        },
    }
    first = workflow._compute_terminal_records_fixed_point(**kwargs)
    second = workflow._compute_terminal_records_fixed_point(**kwargs)
    assert first == second
    assert 1 <= first["iterations"] <= 8
    candidate_bytes = sum(len(value) for value in first["serialized_bytes"].values())
    assert first["exact_recursive_file_bytes"] == (
        workflow._directory_bytes(run_dir) + candidate_bytes
    )
    authority = json.loads(
        first["serialized_bytes"]["terminal_storage_authority.json"]
    )
    verification = json.loads(first["serialized_bytes"]["verification.json"])
    marker = json.loads(first["serialized_bytes"]["stages/report_verify.json"])
    assert verification["terminal_storage_semantic_sha256"] == authority[
        "semantic_sha256"
    ]
    assert set(marker) == {
        "schema",
        "schema_version",
        "stage",
        "passed",
        "completed_at",
        "detail",
        "semantic_sha256",
    }
    assert marker["detail"]["fixture"] == 1
    assert marker["detail"]["terminal_storage_semantic_sha256"] == authority[
        "semantic_sha256"
    ]
    assert marker["detail"]["verification_semantic_sha256"] == verification[
        "semantic_sha256"
    ]


def test_terminal_fixed_point_replaces_partial_candidates_without_counting_stale_bytes(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "child"
    (run_dir / "stages").mkdir(parents=True)
    (run_dir / "payload.bin").write_bytes(b"immutable-evidence")
    for relative, body in (
        ("terminal_storage_authority.json", b"stale-authority" * 7),
        ("verification.json", b"stale-verification" * 11),
        ("stages/report_verify.json", b"stale-marker" * 13),
    ):
        (run_dir / relative).write_bytes(body)
    kwargs = {
        "run_dir": run_dir,
        "terminal_kind": "success",
        "terminalized_at": "2026-08-14T00:00:00+00:00",
        "resource_ledger_semantic_sha256": "b" * 64,
        "verification_body": {
            "schema": workflow.VERSION + "-verification",
            "schema_version": 1,
            "terminal_kind": "success",
            "passed": 1,
        },
        "stage_body": {
            "schema": workflow.VERSION + "-stage",
            "schema_version": 1,
            "stage": "report_verify",
            "passed": 1,
            "completed_at": "2026-08-14T00:00:00+00:00",
            "detail": {},
        },
    }
    result = workflow._compute_terminal_records_fixed_point(**kwargs)
    stale = sum(
        (run_dir / relative).stat().st_size
        for relative in (
            "terminal_storage_authority.json",
            "verification.json",
            "stages/report_verify.json",
        )
    )
    base = workflow._directory_bytes(run_dir) - stale
    assert result["exact_recursive_file_bytes"] == base + sum(
        len(value) for value in result["serialized_bytes"].values()
    )


def test_manifest_and_checksum_inventory_is_exact_sorted_and_self_excluding(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "child"
    (run_dir / "nested").mkdir(parents=True)
    (run_dir / "alpha.bin").write_bytes(b"alpha")
    (run_dir / "nested/beta.bin").write_bytes(b"beta")
    workflow._write_manifest_and_checksums(run_dir, terminal_kind="success")
    manifest = workflow._read_json(
        run_dir / "artifact_manifest.json", semantic=True
    )
    rows = manifest["artifacts"]
    paths = [row["path"] for row in rows]
    assert paths == sorted(paths)
    assert paths == ["alpha.bin", "nested/beta.bin"]
    assert manifest["artifact_count"] == len(rows)
    assert sorted(manifest["excluded_self_referential_paths"]) == sorted(
        workflow._TERMINAL_EXCLUSIONS
    )
    assert all(
        row == {
            "path": row["path"],
            "size": (run_dir / row["path"]).stat().st_size,
            "sha256": _sha256(run_dir / row["path"]),
        }
        for row in rows
    )
    checksum_lines = (run_dir / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    checksum_paths = [line.split("  ", 1)[1] for line in checksum_lines]
    assert checksum_paths == sorted((*paths, "artifact_manifest.json"))
    assert len(checksum_paths) == len(set(checksum_paths))
    for line in checksum_lines:
        digest, relative = line.split("  ", 1)
        assert digest == _sha256(run_dir / relative)


def test_failure_capture_is_first_commit_and_last_valid_binds_all_available_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "child"
    raw = run_dir / "reverse/fused_families/family/segment/shard-0063.npz"
    derived = run_dir / "reverse/summary.json"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"complete-raw-objective")
    workflow._write_semantic(
        derived,
        {"schema": "fixture-summary", "schema_version": 1, "completed": 1},
    )
    writes: list[str] = []
    real_write = workflow._write_bytes_atomic

    def record_write(path: Path, data: bytes) -> None:
        candidate = Path(path)
        if candidate == run_dir or run_dir in candidate.parents:
            writes.append(candidate.relative_to(run_dir).as_posix())
        real_write(path, data)

    monkeypatch.setattr(workflow, "_write_bytes_atomic", record_write)
    capture = workflow._capture_failure(
        run_dir,
        "reverse_complete",
        workflow.CompositionControlError("source control below one percent"),
    )
    assert writes == ["failure_capture.json"]
    assert capture["failure_domain"] == "composition_control"
    assert capture["learned_interpretation_authorized"] == 0
    assert not (run_dir / "terminal_failure.json").exists()
    evidence = workflow._last_valid_evidence(run_dir, capture)
    serialized = json.dumps(evidence, sort_keys=True)
    assert raw.relative_to(run_dir).as_posix() in serialized
    assert derived.relative_to(run_dir).as_posix() in serialized
    assert _sha256(raw) in serialized
    assert _sha256(derived) in serialized


def _synthetic_success_terminal_fixture(
    tmp_path: Path,
) -> tuple[Path, SimpleNamespace, dict[str, object], dict[str, object]]:
    run_dir = tmp_path / "child"
    prefix = tmp_path / "prefix"
    parent = tmp_path / "parent"
    source = tmp_path / "source"
    for root in (run_dir, prefix, parent, source):
        root.mkdir()
    workflow._write_semantic(
        run_dir / "resource_ledger.json", workflow._initial_resource_ledger()
    )
    (run_dir / "objective.bin").write_bytes(b"complete-objective-evidence")
    args = SimpleNamespace(
        prefix_run_dir=prefix, parent_run_dir=parent, source_run_dir=source
    )
    evidence: dict[str, object] = {
        "passed": 1,
        "outcome": {
            "global_effect_label": "global_complete_adverse",
            "source_effect_label": "source_informative",
            "learned_interpretation_authorized": 1,
            "required_next_action": "run_conventional_mnist_ddpm_sanity",
        },
        "imports": {"predecessor": {"supplied_prefix_path": str(prefix.resolve())}},
        "ledger": {
            "active_seconds": V2_ACTIVE_SECONDS + 10.0,
            "events": [{"role": "prefix_resource_carry"}],
        },
    }
    admission: dict[str, object] = {
        "schema": workflow.VERSION + "-report-admission",
        "schema_version": 1,
        "admitted": 1,
        "charged_seconds": workflow.REPORT_RESERVE_SECONDS,
        "deadline_monotonic": 700.0,
    }
    return run_dir, args, evidence, admission


def test_success_terminal_marker_is_last_and_inventory_audit_is_byte_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir, args, evidence, admission = _synthetic_success_terminal_fixture(
        tmp_path
    )
    deep_calls: list[Path] = []
    monkeypatch.setattr(
        workflow,
        "_deep_verify_scientific_evidence_read_only",
        lambda root, _args=None: (deep_calls.append(Path(root)) or evidence),
    )
    monkeypatch.setattr(workflow.time, "perf_counter", lambda: 100.0)
    writes: list[str] = []
    real_write = workflow._write_bytes_atomic

    def record_write(path: Path, data: bytes) -> None:
        writes.append(Path(path).relative_to(run_dir).as_posix())
        real_write(path, data)

    monkeypatch.setattr(workflow, "_write_bytes_atomic", record_write)
    result = workflow._finalize_success(run_dir, args, admission)
    assert result["passed"] == 1
    assert deep_calls == [run_dir, run_dir]
    assert writes[-1] == "stages/report_verify.json"
    for relative in ("REPORT.md", "HANDOFF.md"):
        human = (run_dir / relative).read_text(encoding="utf-8")
        assert str(args.prefix_run_dir.resolve()) in human
        assert "64 authenticated imports, 0 child-generated" in human
        assert "1 authenticated import (8 steps), 63 child-generated (504 steps)" in human
        assert str(V2_ACTIVE_SECONDS) in human
        assert "22500" in human
        assert "Mode: exploratory" in human and "one opened path" in human
        assert all(
            exclusion in human
            for exclusion in ("reference-prior", "population", "diversity", "confirmatory")
        )
    authority = workflow._read_json(
        run_dir / "terminal_storage_authority.json", semantic=True
    )
    assert authority["exact_recursive_file_bytes"] == workflow._directory_bytes(
        run_dir
    )
    before = workflow._snapshot_tree(run_dir)
    audit = workflow._verify_terminal_inventory_read_only(run_dir, "success")
    assert audit["passed"] == 1
    assert workflow._snapshot_tree(run_dir) == before


def test_success_terminal_partial_commit_retries_without_new_scientific_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir, args, evidence, admission = _synthetic_success_terminal_fixture(
        tmp_path
    )
    monkeypatch.setattr(
        workflow,
        "_deep_verify_scientific_evidence_read_only",
        lambda *_args, **_kwargs: evidence,
    )
    monkeypatch.setattr(workflow, "_utc_now", lambda: "2026-08-14T00:00:00+00:00")
    monkeypatch.setattr(workflow.time, "perf_counter", lambda: 100.0)
    real_write = workflow._write_bytes_atomic
    crash_once = {"armed": True}

    def crash_before_marker(path: Path, data: bytes) -> None:
        if Path(path) == run_dir / "stages/report_verify.json" and crash_once["armed"]:
            crash_once["armed"] = False
            raise RuntimeError("injected terminal marker crash")
        real_write(path, data)

    monkeypatch.setattr(workflow, "_write_bytes_atomic", crash_before_marker)
    with pytest.raises(RuntimeError, match="terminal marker crash"):
        workflow._finalize_success(run_dir, args, admission)
    assert (run_dir / "terminal_storage_authority.json").is_file()
    assert (run_dir / "verification.json").is_file()
    assert not (run_dir / "stages/report_verify.json").exists()
    assert workflow._finalize_success(run_dir, args, admission)["passed"] == 1
    assert workflow._verify_terminal_inventory_read_only(run_dir, "success")[
        "passed"
    ] == 1


def test_expired_report_deadline_cannot_commit_success_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir, args, evidence, admission = _synthetic_success_terminal_fixture(
        tmp_path
    )
    admission["deadline_monotonic"] = 99.0
    monkeypatch.setattr(
        workflow,
        "_deep_verify_scientific_evidence_read_only",
        lambda *_args, **_kwargs: evidence,
    )
    monkeypatch.setattr(workflow.time, "perf_counter", lambda: 100.0)
    with pytest.raises(workflow.ResourceBoundaryError, match="deadline"):
        workflow._finalize_success(run_dir, args, admission)
    assert not (run_dir / "stages/report_verify.json").exists()


@pytest.mark.parametrize("tamper", ("payload", "manifest", "checksum", "hardlink"))
def test_terminal_inventory_rejects_manifest_checksum_raw_or_link_tamper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tamper: str
) -> None:
    run_dir, args, evidence, admission = _synthetic_success_terminal_fixture(
        tmp_path
    )
    monkeypatch.setattr(
        workflow,
        "_deep_verify_scientific_evidence_read_only",
        lambda *_args, **_kwargs: evidence,
    )
    monkeypatch.setattr(workflow.time, "perf_counter", lambda: 100.0)
    workflow._finalize_success(run_dir, args, admission)
    payload = run_dir / "objective.bin"
    if tamper == "payload":
        payload.write_bytes(payload.read_bytes() + b"tamper")
    elif tamper == "manifest":
        manifest = run_dir / "artifact_manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
    elif tamper == "checksum":
        checksum = run_dir / "SHA256SUMS.txt"
        body = checksum.read_bytes()
        checksum.write_bytes((b"0" if body[:1] != b"0" else b"1") + body[1:])
    else:
        external = tmp_path / "external-objective.bin"
        external.write_bytes(payload.read_bytes())
        payload.unlink()
        try:
            os.link(external, payload)
        except OSError as exc:
            pytest.skip(f"hardlinks unavailable: {exc}")
    with pytest.raises(workflow.ContinuationIntegrityError):
        workflow._verify_terminal_inventory_read_only(run_dir, "success")


def test_terminal_child_verifier_reopens_inventory_and_preserves_all_four_trees(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir, args, evidence, admission = _synthetic_success_terminal_fixture(
        tmp_path
    )
    monkeypatch.setattr(
        workflow,
        "_deep_verify_scientific_evidence_read_only",
        lambda *_args, **_kwargs: evidence,
    )
    monkeypatch.setattr(workflow.time, "perf_counter", lambda: 100.0)
    workflow._finalize_success(run_dir, args, admission)
    before = tuple(
        workflow._snapshot_tree(root)
        for root in (
            run_dir,
            args.prefix_run_dir,
            args.parent_run_dir,
            args.source_run_dir,
        )
    )
    assert workflow._verify_terminal_child_read_only(
        run_dir,
        prefix_run_dir=args.prefix_run_dir,
        parent_run_dir=args.parent_run_dir,
        source_run_dir=args.source_run_dir,
    )["passed"] == 1
    assert before == tuple(
        workflow._snapshot_tree(root)
        for root in (
            run_dir,
            args.prefix_run_dir,
            args.parent_run_dir,
            args.source_run_dir,
        )
    )
    (run_dir / "objective.bin").write_bytes(b"same-path-but-tampered")
    with pytest.raises(workflow.ContinuationIntegrityError):
        workflow._verify_terminal_child_read_only(
            run_dir,
            prefix_run_dir=args.prefix_run_dir,
            parent_run_dir=args.parent_run_dir,
            source_run_dir=args.source_run_dir,
        )


def test_composition_failure_seals_verification_last_and_preserves_complete_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "child"
    prefix, parent, source = (
        tmp_path / "prefix",
        tmp_path / "parent",
        tmp_path / "source",
    )
    for root in (run_dir, prefix, parent, source):
        root.mkdir()
    workflow._write_semantic(run_dir / "run_manifest.json", {"schema": "fixture", "schema_version": 1})
    workflow._write_semantic(
        run_dir / "resource_ledger.json", workflow._initial_resource_ledger()
    )
    carry_journal = workflow._begin_durable_attempt(
        run_dir, role="prefix_resource_carry", attempt=1, durable_elapsed_seconds=0.0
    )
    workflow._finish_durable_attempt(
        run_dir, carry_journal, durable_elapsed_seconds=0.0,
        invocation_wall_seconds=V2_ACTIVE_SECONDS,
        peak_cuda_bytes=46_834_176, total_cuda_bytes=8_546_484_224,
        detail=workflow._prefix_carry_detail(),
    )
    workflow._write_semantic(
        run_dir / "predecessor_binding.json",
        {"schema": "fixture", "schema_version": 1, "supplied_prefix_path": str(prefix.resolve())},
    )
    (run_dir / "exact_command.txt").write_text("python -m fixture\n", encoding="utf-8")
    raw = run_dir / "reverse/fused_families/family/complete/shard-0063.npz"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"all-64-raw-shards-represented-by-fixture")
    for relative, body in (
        ("reverse/family_summary.json", {"completed": 1, "shard_count": 64}),
        ("reverse/summary.json", {"completed": 1, "failed_rows_suppressed": 0}),
        ("outcome.json", _classification(zero=1.0, global_error=0.8, source=0.5)),
    ):
        workflow._write_semantic(
            run_dir / relative,
            {"schema": "fixture", "schema_version": 1, **body},
        )
    (run_dir / "reverse/metrics.csv").write_text("row,error\nsource,1\n", encoding="utf-8")
    image = run_dir / "images/raw/source-informed/step-512.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"failed-control-image")
    workflow._stage_marker(run_dir, "prepare", {"fixture": 1})
    imported_verifications: list[Path] = []
    monkeypatch.setattr(
        workflow,
        "_completed_stage_artifacts_read_only",
        lambda root, stage: workflow._read_stage_marker_exact(root, stage),
    )
    monkeypatch.setattr(
        workflow,
        "_verify_imported_inputs",
        lambda root: imported_verifications.append(Path(root)) or {"passed": 1},
    )
    monkeypatch.setattr(
        workflow, "_scan_reverse_chain",
        lambda root: {"records": [{}] * (64 if (Path(root) / "outcome.json").is_file() else 1)},
    )
    monkeypatch.setattr(workflow, "_load_npz", lambda _path: {"state": np.zeros(1)})
    monkeypatch.setattr(
        workflow, "_strict_fused_exact_health",
        lambda **_kwargs: {"passed": 1, "certificate_fraction": 1.0, "maximum_mass_error": 0.0, "fallback_count": 0},
    )
    args = SimpleNamespace(prefix_run_dir=prefix, parent_run_dir=parent, source_run_dir=source)
    resource_dir = tmp_path / "resource-child"
    shutil.copytree(run_dir, resource_dir)
    shutil.rmtree(resource_dir / "reverse")
    (resource_dir / "outcome.json").unlink()
    resource_capture = workflow._capture_failure(
        resource_dir, "reverse_complete", workflow.ResourceBoundaryError("projection stop")
    )
    workflow._finalize_failure_package(resource_dir, args, resource_capture)
    resource_report = (resource_dir / "REPORT.md").read_text(encoding="utf-8")
    assert all(token in resource_report for token in (
        f"authenticated v2 carry = {V2_ACTIVE_SECONDS}", f"current ledger active = {V2_ACTIVE_SECONDS}",
        "Completed stage scope: ['prepare']", "Objective-bearing complete path: 0",
        "Committed reverse scope: 1 of 64 shards", "Exact committed reverse health: passed=1",
        "No complete-path metric, effect classification, or required-action decision is authorized.",
    ))
    assert "trajectory_shard_boundaries.npz" not in resource_report
    writes: list[str] = []
    real_write = workflow._write_bytes_atomic

    def record_write(path: Path, data: bytes) -> None:
        writes.append(Path(path).relative_to(run_dir).as_posix())
        real_write(path, data)

    monkeypatch.setattr(workflow, "_write_bytes_atomic", record_write)
    capture = workflow._capture_failure(
        run_dir,
        "reverse_complete",
        workflow.CompositionControlError("source effect was only 0.5 percent"),
    )
    result = workflow._finalize_failure_package(run_dir, args, capture)
    assert result["terminal_kind"] == "failure"
    assert imported_verifications == [resource_dir, run_dir]
    for relative in ("REPORT.md", "HANDOFF.md"):
        human = (run_dir / relative).read_text(encoding="utf-8")
        assert all(token in human for token in (
            str(prefix.resolve()), "64 authenticated imports, 0 child-generated",
            "1 authenticated import (8 steps), 63 child-generated (504 steps)",
            str(V2_ACTIVE_SECONDS), "22500",
        ))
        assert all(token in human for token in (
            "Objective-bearing complete path: 1", "Primary mixed-target squared-L2 errors:",
            "Relative improvements:", "Complete-path effect: `global_material_improvement`",
            "Required next action: `run_stage_e_reference_prior`",
        ))
    assert "failure_capture.json" in writes
    assert writes.index("failure_capture.json") < writes.index("terminal_failure.json")
    assert writes[-1] == "verification.json"
    assert not (run_dir / "stages/report_verify.json").exists()
    terminal = workflow._read_json(run_dir / "terminal_failure.json", semantic=True)
    assert terminal["scientific_objective_completed"] == 1
    assert terminal["learned_interpretation_authorized"] == 0
    assert terminal["resume_same_child_authorized"] == 0
    manifest_paths = {
        row["path"]
        for row in workflow._read_json(
            run_dir / "artifact_manifest.json", semantic=True
        )["artifacts"]
    }
    for path in (raw, run_dir / "reverse/metrics.csv", image):
        assert path.relative_to(run_dir).as_posix() in manifest_paths
        assert path.is_file()
    monkeypatch.setattr(
        workflow,
        "_verify_imported_inputs",
        lambda _root: (_ for _ in ()).throw(
            workflow.ContinuationIntegrityError("tampered completed import")
        ),
    )
    with pytest.raises(workflow.ContinuationIntegrityError, match="completed import"):
        workflow._verify_terminal_child_contents_read_only(
            run_dir,
            prefix_run_dir=prefix,
            parent_run_dir=parent,
            source_run_dir=source,
        )
    before = workflow._snapshot_tree(run_dir)
    with pytest.raises(workflow.TerminalRunError):
        workflow._verify_resume_compatibility_read_only(
            run_dir,
            identity={},
            prefix_run_dir=prefix,
            parent_run_dir=parent,
            source_run_dir=source,
        )
    assert workflow._snapshot_tree(run_dir) == before


def test_capture_only_resume_packages_failure_without_reentering_science(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir, prefix, parent, source = (
        tmp_path / name for name in ("child", "prefix", "parent", "source")
    )
    for root in (run_dir, prefix, parent, source):
        root.mkdir()
    workflow._write_semantic(
        run_dir / "resource_ledger.json", workflow._initial_resource_ledger()
    )
    capture = workflow._capture_failure(
        run_dir, "forward_tail", RuntimeError("captured engineering failure")
    )
    args = SimpleNamespace(prefix_run_dir=prefix, parent_run_dir=parent, source_run_dir=source)
    paths = SimpleNamespace(run_dir=run_dir, prefix_run_dir=prefix, parent_run_dir=parent, source_run_dir=source)
    monkeypatch.setattr(workflow, "parse_args", lambda _argv: args)
    monkeypatch.setattr(workflow, "_resolve_mode", lambda _args: "resume")
    monkeypatch.setattr(workflow, "_resolve_paths", lambda *_args, **_kwargs: paths)
    monkeypatch.setattr(workflow, "_load_child_identity_read_only", lambda *_args: {})
    monkeypatch.setattr(
        workflow, "_verify_resume_compatibility_read_only", lambda *_args, **_kwargs: {"passed": 1}
    )
    reconciled: list[Path] = []
    monkeypatch.setattr(workflow, "_reconcile_live_stage_journals", lambda path: reconciled.append(Path(path)) or [])
    monkeypatch.setattr(
        workflow,
        "_run_requested_stages",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("capture-only resume must not re-enter science")
        ),
    )
    assert workflow.main([]) == 1
    assert reconciled == [run_dir, run_dir]
    terminal = workflow._read_json(run_dir / "terminal_failure.json", semantic=True)
    assert terminal["failure_capture_semantic_sha256"] == capture["semantic_sha256"]
    assert workflow._read_json(run_dir / "verification.json", semantic=True)[
        "terminal_kind"
    ] == "failure"
    report = (run_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "V2 predecessor authentication/carry is incomplete" in report
    assert "are not authenticated or charged by this report" in report
    assert "Objective-bearing complete path: 0" in report


def test_caught_owned_stage_exception_captures_first_and_verifies_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir, prefix, parent, source = (
        tmp_path / name for name in ("child", "prefix", "parent", "source")
    )
    for root in (run_dir, prefix, parent, source):
        root.mkdir()
    workflow._write_semantic(
        run_dir / "resource_ledger.json", workflow._initial_resource_ledger()
    )
    args = SimpleNamespace(prefix_run_dir=prefix, parent_run_dir=parent, source_run_dir=source)
    paths = SimpleNamespace(run_dir=run_dir, prefix_run_dir=prefix, parent_run_dir=parent, source_run_dir=source)
    monkeypatch.setattr(workflow, "parse_args", lambda _argv: args)
    monkeypatch.setattr(workflow, "_resolve_mode", lambda _args: "resume")
    monkeypatch.setattr(workflow, "_resolve_paths", lambda *_args, **_kwargs: paths)
    monkeypatch.setattr(workflow, "_load_child_identity_read_only", lambda *_args: {})
    monkeypatch.setattr(workflow, "_verify_resume_compatibility_read_only", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(workflow, "_reconcile_live_stage_journals", lambda *_args: [])
    monkeypatch.setattr(
        workflow,
        "_run_requested_stages",
        lambda *_args: (_ for _ in ()).throw(
            workflow.CompositionControlError("known-positive composition failed")
        ),
    )
    writes: list[str] = []
    real_write = workflow._write_bytes_atomic

    def record_write(path: Path, data: bytes) -> None:
        candidate = Path(path)
        if candidate == run_dir or run_dir in candidate.parents:
            writes.append(candidate.relative_to(run_dir).as_posix())
        real_write(path, data)

    monkeypatch.setattr(workflow, "_write_bytes_atomic", record_write)
    assert workflow.main([]) == 1
    assert writes.index("failure_capture.json") < writes.index("terminal_failure.json")
    assert writes[-1] == "verification.json"
    assert workflow._read_json(run_dir / "terminal_failure.json", semantic=True)[
        "failure_domain"
    ] == "composition_control"


def test_read_only_verifier_preserves_all_tree_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    child = tmp_path / "child"
    prefix = tmp_path / "prefix"
    parent = tmp_path / "parent"
    source = tmp_path / "source"
    for root, content in (
        (child, b"c"),
        (prefix, b"v2"),
        (parent, b"p"),
        (source, b"s"),
    ):
        root.mkdir()
        (root / "record.bin").write_bytes(content)
    before = tuple(
        workflow._snapshot_tree(root) for root in (child, prefix, parent, source)
    )
    monkeypatch.setattr(
        workflow,
        "_verify_terminal_child_contents_read_only",
        lambda *_args, **_kwargs: {"passed": 1},
    )
    result = workflow._verify_terminal_child_read_only(
        child,
        prefix_run_dir=prefix,
        parent_run_dir=parent,
        source_run_dir=source,
    )
    after = tuple(
        workflow._snapshot_tree(root) for root in (child, prefix, parent, source)
    )
    assert result["passed"] == 1
    assert after == before


def test_stage_all_stub_integration_and_resume_after_forward_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    calls: list[str] = []
    crashed = {"value": False}

    def stage(name: str):
        def run(_run_dir: Path, _args: object) -> None:
            calls.append(name)
            if name == "forward_tail" and not crashed["value"]:
                crashed["value"] = True
                raise RuntimeError("simulated hard crash")
            workflow._stage_marker(_run_dir, name, {"fixture": 1})

        return run

    for name in workflow.STAGES:
        monkeypatch.setattr(workflow, f"_run_{name}", stage(name))
    monkeypatch.setattr(
        workflow,
        "_completed_stage_artifacts_read_only",
        lambda run, name: workflow._read_stage_marker_exact(run, name),
    )
    args = SimpleNamespace(stage="all")
    with pytest.raises(RuntimeError, match="simulated hard crash"):
        workflow._run_requested_stages(run_dir, args)
    assert calls == ["prepare", "controls", "forward_tail"]
    calls.clear()
    workflow._run_requested_stages(run_dir, args)
    assert calls == ["forward_tail", "reverse_complete", "report_verify"]


def test_stage_order_rejects_later_marker_with_missing_predecessor(tmp_path: Path) -> None:
    run_dir = tmp_path / "child"
    workflow._write_semantic(
        run_dir / "stages/controls.json",
        {"schema": "stub", "schema_version": 1, "passed": 1},
    )
    with pytest.raises(workflow.ContinuationIntegrityError):
        workflow._require_predecessors(run_dir, "forward_tail")


def test_source_closure_incompatibility_is_no_write_resume_rejection(tmp_path: Path) -> None:
    child, prefix, parent, source, identity = _compatible_resume_fixture(tmp_path)
    before = workflow._snapshot_tree(child)
    identity["source_closure"] = {
        "mnist/missing.py": {"size": 1, "sha256": "0" * 64}
    }
    with pytest.raises(workflow.ContinuationIntegrityError, match="source closure|identity"):
        workflow._verify_resume_compatibility_read_only(
            child,
            identity=identity,
            prefix_run_dir=prefix,
            parent_run_dir=parent,
            source_run_dir=source,
        )
    assert workflow._snapshot_tree(child) == before


@pytest.mark.parametrize("locator", ("prefix", "parent", "source"))
def test_resume_rejects_external_locator_switch_before_mutation(
    tmp_path: Path, locator: str
) -> None:
    child, prefix, parent, source, identity = _compatible_resume_fixture(tmp_path)
    supplied = {"prefix": prefix, "parent": parent, "source": source}
    switched = tmp_path / f"switched-{locator}"
    shutil.copytree(supplied[locator], switched)
    supplied[locator] = switched
    roots = (child, prefix, parent, source, switched)
    before = tuple(workflow._snapshot_tree(root) for root in roots)
    with pytest.raises(workflow.ContinuationIntegrityError):
        workflow._verify_resume_compatibility_read_only(
            child,
            identity=identity,
            prefix_run_dir=supplied["prefix"],
            parent_run_dir=supplied["parent"],
            source_run_dir=supplied["source"],
        )
    after = tuple(workflow._snapshot_tree(root) for root in roots)
    assert after == before


def test_resume_rejects_terminal_child_before_mutation(tmp_path: Path) -> None:
    child, prefix, parent, source, identity = _compatible_resume_fixture(tmp_path)
    workflow._write_semantic(
        child / "stages/report_verify.json",
        {
            "schema": workflow.VERSION + "-stage",
            "schema_version": 1,
            "stage": "report_verify",
            "passed": 1,
        },
    )
    before = workflow._snapshot_tree(child)
    with pytest.raises(workflow.TerminalRunError):
        workflow._verify_resume_compatibility_read_only(
            child,
            identity=identity,
            prefix_run_dir=prefix,
            parent_run_dir=parent,
            source_run_dir=source,
        )
    assert workflow._snapshot_tree(child) == before


def test_resume_rejects_bound_external_tree_tamper_before_mutation(tmp_path: Path) -> None:
    child, prefix, parent, source, identity = _compatible_resume_fixture(tmp_path)
    workflow._write_semantic(
        child / "predecessor_binding.json",
        {
            "schema": workflow.VERSION + "-predecessor-binding",
            "schema_version": 1,
            "supplied_prefix_path": str(prefix.resolve()),
            "prefix_tree_sha256": workflow._tree_hash(workflow._snapshot_tree(prefix)),
        },
    )
    workflow._write_semantic(
        child / "parent_binding.json",
        {
            "schema": workflow.VERSION + "-parent-binding",
            "schema_version": 1,
            "parent_tree_sha256": workflow._tree_hash(workflow._snapshot_tree(parent)),
            "source_tree_sha256": workflow._tree_hash(workflow._snapshot_tree(source)),
        },
    )
    (source / "identity.bin").write_bytes(b"tampered-source")
    before = workflow._snapshot_tree(child)
    with pytest.raises(workflow.ContinuationIntegrityError, match="external evidence tree"):
        workflow._verify_resume_compatibility_read_only(
            child,
            identity=identity,
            prefix_run_dir=prefix,
            parent_run_dir=parent,
            source_run_dir=source,
        )
    assert workflow._snapshot_tree(child) == before


def test_resume_main_verifies_compatibility_before_running_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    child, prefix, parent, source, identity = _compatible_resume_fixture(tmp_path)
    called = {"identity": 0, "compatibility": 0, "stage": 0}

    def load_identity(_path: Path) -> dict[str, object]:
        called["identity"] += 1
        return identity

    def reject(*_args: object, **_kwargs: object) -> None:
        called["compatibility"] += 1
        raise workflow.ContinuationIntegrityError("incompatible")

    monkeypatch.setattr(workflow, "_load_child_identity_read_only", load_identity)
    monkeypatch.setattr(workflow, "_verify_resume_compatibility_read_only", reject)
    monkeypatch.setattr(
        workflow,
        "_run_requested_stages",
        lambda *_args, **_kwargs: called.__setitem__("stage", 1),
    )
    before = workflow._snapshot_tree(child)
    argv = [
        "--stage",
        "controls",
        "--repository-root",
        str(REPOSITORY_ROOT),
        "--resume-run-dir",
        str(child),
        "--prefix-run-dir",
        str(prefix),
        "--parent-run-dir",
        str(parent),
        "--source-run-dir",
        str(source),
        "--device",
        "cuda",
    ]
    with pytest.raises(workflow.ContinuationIntegrityError, match="incompatible"):
        workflow.main(argv)
    assert called["identity"] == 1
    assert called["compatibility"] == 1
    assert called["stage"] == 0
    assert workflow._snapshot_tree(child) == before
    assert not (child.parent / f".{child.name}.resume-probe.json").exists()
