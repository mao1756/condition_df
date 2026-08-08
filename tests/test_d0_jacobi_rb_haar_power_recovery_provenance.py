from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

import mnist.d0_jacobi_rb_haar_power_recovery_provenance as provenance
from mnist.d0_jacobi_artifacts import ArtifactCompatibilityError


PARENT = (
    Path(
        "runs/experiment12_d0_jacobi_rb_hierarchical_coupling_confirmation"
    )
    / provenance.PARENT_RUN_BASENAME
)


def _nested_schedule() -> dict[str, object]:
    return {
        "profile_name": "nested_haar_single_arm",
        "pool": "main",
        "role": "nested_a",
        "levels": [128, 256, 512, 1024],
        "coarsest_steps": 128,
        "finest_steps": 1024,
        "single_arm": 1,
    }


def _nested_record() -> dict[str, object]:
    return {"identity": {"schedule": _nested_schedule()}}


def _patched_load(
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    mutate: object,
) -> None:
    original = provenance._load

    def load(path: Path) -> dict[str, object]:
        record = original(path)
        if path.name == filename:
            changed = copy.deepcopy(record)
            mutate(changed)
            return changed
        return record

    monkeypatch.setattr(provenance, "_load", load)


def test_canonical_schedule_is_read_from_identity() -> None:
    record = _nested_record()
    assert provenance.canonical_shard_schedule(record) == _nested_schedule()
    binding = provenance.canonical_shard_schedule_binding(
        record,
        expected_profile_name="nested_haar_single_arm",
        expected_pool="main",
        expected_role="nested_a",
    )
    assert binding == {
        "schema": provenance.SCHEDULE_BINDING_VERSION,
        "schema_version": 1,
        "binding_source": "identity.schedule",
        "top_level_schedule_present": 0,
        "schedule": _nested_schedule(),
    }


def test_equal_top_level_schedule_is_accepted_but_identity_stays_authoritative() -> None:
    record = _nested_record()
    record["schedule"] = copy.deepcopy(_nested_schedule())
    binding = provenance.canonical_shard_schedule_binding(record)
    assert binding["binding_source"] == "identity.schedule"
    assert binding["top_level_schedule_present"] == 1


@pytest.mark.parametrize(
    "record,match",
    [
        ({}, "identity is missing"),
        ({"identity": {}}, "identity schedule is missing"),
        (
            {
                "identity": {
                    "schedule": {
                        **_nested_schedule(),
                        "levels": [128, 256, 512],
                    }
                }
            },
            "malformed schedule",
        ),
        (
            {
                "identity": {
                    "schedule": {
                        **_nested_schedule(),
                        "unexpected": 1,
                    }
                }
            },
            "not canonical",
        ),
        (
            {
                "identity": {"schedule": _nested_schedule()},
                "schedule": {
                    **_nested_schedule(),
                    "pool": "reference",
                },
            },
            "schedules conflict",
        ),
    ],
)
def test_missing_malformed_or_conflicting_schedules_fail_closed(
    record: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ArtifactCompatibilityError, match=match):
        provenance.canonical_shard_schedule(record)


def test_profile_pool_and_role_mismatches_fail_closed() -> None:
    record = _nested_record()
    for keyword, value in (
        ("expected_profile_name", "pairwise_haar_antithetic"),
        ("expected_pool", "reference"),
        ("expected_role", "nested_b"),
    ):
        with pytest.raises(ArtifactCompatibilityError, match="incompatible"):
            provenance.canonical_shard_schedule(record, **{keyword: value})


def test_canonical_antithetic_schedule_is_supported() -> None:
    schedule = {
        "profile_name": "pairwise_haar_antithetic",
        "role": "antithetic_a",
        "coarse_steps": 128,
        "fine_steps": 256,
        "pair_local_tree": 1,
        "fine_arms": [-1, 1],
    }
    assert provenance.canonical_shard_schedule(
        {"identity": {"schedule": schedule}},
        expected_profile_name="pairwise_haar_antithetic",
        expected_role="antithetic_a",
    ) == schedule


def test_no_work_and_later_panel_checks_reject_tampering() -> None:
    with pytest.raises(ArtifactCompatibilityError, match="forbidden work"):
        provenance._no_work_tree(
            {"nested": {"physical_training_performed": 1}},
            "fixture",
        )
    with pytest.raises(ArtifactCompatibilityError, match="selection"):
        provenance._verify_absent_work({"selected_haar_design.json"})
    with pytest.raises(ArtifactCompatibilityError, match="panel-B"):
        provenance._verify_absent_work(
            {
                "haar_power_shards/nested_haar_single_arm/b/main/"
                + "0" * 64
                + ".json"
            }
        )


def test_wrong_parent_fails_closed() -> None:
    with pytest.raises(ArtifactCompatibilityError):
        provenance.verify_haar_power_recovery_parent(Path("tests"))


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable Haar parent unavailable")
def test_exact_parent_and_all_restart_chains_verify_without_mutation() -> None:
    before = {
        path.relative_to(PARENT).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in PARENT.rglob("*")
        if path.is_file()
    }
    record = provenance.verify_haar_power_recovery_parent(PARENT)
    after = {
        path.relative_to(PARENT).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in PARENT.rglob("*")
        if path.is_file()
    }

    assert before == after
    assert record["passed"] == 1
    assert (
        record["parent_artifact_registry_sha256"]
        == provenance.PARENT_REGISTRY_SHA256
    )
    assert (
        record["parent_artifact_record_count"]
        == provenance.PARENT_REGISTRY_RECORD_COUNT
    )
    assert record["parent_source_count"] == provenance.PARENT_SOURCE_COUNT
    assert (
        record["parent_source_fingerprint"]
        == provenance.PARENT_SOURCE_FINGERPRINT
    )
    assert (
        record["parent_scientific_config_sha256"]
        == provenance.PARENT_SCIENTIFIC_CONFIG_SHA256
    )
    assert record["parent_preflight_subcheck_count"] == 61
    assert record["parent_coupling_subcheck_count"] == 42
    assert record["parent_failure_code"] == provenance.PARENT_FAILURE_CODE
    assert (
        record["parent_re_adjudication"]
        == provenance.PARENT_RE_ADJUDICATION
    )
    assert (
        record["parent_nested_main_shard_count"]
        == provenance.PARENT_NESTED_MAIN_SHARD_COUNT
    )
    assert (
        record["parent_nested_reference_shard_count"]
        == provenance.PARENT_NESTED_REFERENCE_SHARD_COUNT
    )
    assert (
        record["parent_nested_shard_count"]
        == provenance.PARENT_NESTED_SHARD_COUNT
    )
    assert record["parent_transition_count"] == provenance.PARENT_TRANSITION_COUNT
    assert record["parent_fallback_count"] == provenance.PARENT_FALLBACK_COUNT
    assert record["parent_mass_error"] == provenance.PARENT_MAXIMUM_MASS_ERROR
    assert record["parent_conservative_rate"] == pytest.approx(
        provenance.PARENT_CONSERVATIVE_RATE,
        rel=1.0e-14,
    )
    assert len(record["canonical_schedule_bindings"]) == 80
    assert all(
        binding["binding_source"] == "identity.schedule"
        and binding["top_level_schedule_present"] == 0
        for binding in record["canonical_schedule_bindings"]
    )
    for name in (
        "parent_all_artifact_hashes_pass",
        "parent_sources_immutable_pass",
        "parent_scientific_config_pass",
        "parent_transitive_provenance_pass",
        "parent_schedule_binding_pass",
        "parent_shard_hash_pass",
        "parent_shard_chain_pass",
        "parent_checkpoint_chain_pass",
        "parent_no_antithetic_power_shards_pass",
        "parent_panel_b_absent_pass",
        "parent_selection_absent_pass",
        "parent_no_work_pass",
    ):
        assert record[name] == 1
    assert record["parent_mutated"] == 0


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable Haar parent unavailable")
def test_registry_digest_tampering_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance, "PARENT_REGISTRY_SHA256", "0" * 64)
    with pytest.raises(ArtifactCompatibilityError, match="registry SHA-256 changed"):
        provenance.verify_haar_power_recovery_parent(PARENT)


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable Haar parent unavailable")
def test_registered_artifact_hash_tampering_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = provenance.file_fingerprint

    def fingerprint(path: str | Path, **kwargs: object) -> str:
        value = original(path, **kwargs)
        if str(path).endswith(".npz") and "haar_power_shards" in str(path):
            return "0" * 64
        return value

    monkeypatch.setattr(provenance, "file_fingerprint", fingerprint)
    with pytest.raises(
        ArtifactCompatibilityError,
        match="registered parent artifact changed",
    ):
        provenance.verify_haar_power_recovery_parent(PARENT)


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable Haar parent unavailable")
def test_source_fingerprint_tampering_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provenance, "source_fingerprint", lambda paths: "0" * 64)
    with pytest.raises(
        ArtifactCompatibilityError,
        match="immutable parent sources changed",
    ):
        provenance.verify_haar_power_recovery_parent(PARENT)


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable Haar parent unavailable")
@pytest.mark.parametrize(
    "filename,mutate,match",
    [
        (
            "scientific_config.json",
            lambda record: record.__setitem__("root_seed", 0),
            "scientific configuration changed",
        ),
        (
            "run_status.json",
            lambda record: record.__setitem__("outcome", "success"),
            "terminal status changed",
        ),
        (
            "haar_preflight_gate.json",
            lambda record: record.__setitem__("passed", 0),
            "preflight no longer recomputes",
        ),
        (
            "haar_coupling_gate.json",
            lambda record: record.__setitem__("passed", 0),
            "coupling no longer recomputes",
        ),
        (
            "pilot_failure.json",
            lambda record: record.__setitem__(
                "failure_code", "different_failure"
            ),
            "exact schedule-binding execution failure",
        ),
    ],
)
def test_semantic_parent_tampering_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    mutate: object,
    match: str,
) -> None:
    _patched_load(monkeypatch, filename, mutate)
    with pytest.raises(ArtifactCompatibilityError, match=match):
        provenance.verify_haar_power_recovery_parent(PARENT)


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable Haar parent unavailable")
def test_transitive_provenance_tampering_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def changed(run_dir: str | Path) -> dict[str, object]:
        record = provenance._load(PARENT / "parent_provenance.json")
        record["parent_mutated"] = 1
        return record

    monkeypatch.setattr(
        provenance,
        "verify_right_endpoint_coupling_parent",
        changed,
    )
    with pytest.raises(
        ArtifactCompatibilityError,
        match="transitive provenance no longer recomputes",
    ):
        provenance.verify_haar_power_recovery_parent(PARENT)


@pytest.mark.skipif(not PARENT.is_dir(), reason="immutable Haar parent unavailable")
def test_shard_predecessor_chain_tampering_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_transitive = provenance._load(PARENT / "parent_provenance.json")
    monkeypatch.setattr(
        provenance,
        "verify_right_endpoint_coupling_parent",
        lambda run_dir: copy.deepcopy(stored_transitive),
    )
    original = provenance.load_committed_haar_shard
    changed = False

    def load(*args: object, **kwargs: object) -> object:
        nonlocal changed
        resumed = original(*args, **kwargs)
        if changed:
            return resumed
        changed = True
        metadata = dict(resumed.metadata)
        metadata["input_sha256"] = "0" * 64
        return replace(resumed, metadata=metadata)

    monkeypatch.setattr(provenance, "load_committed_haar_shard", load)
    with pytest.raises(ArtifactCompatibilityError, match="input chain changed"):
        provenance.verify_haar_power_recovery_parent(PARENT)


def test_module_does_not_import_training_or_reverse_sampling_code() -> None:
    source = Path(provenance.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "mnist.experiment12_d0",
        "mnist.d0_one_image_sampler",
        "mnist.conditioned_diffusion",
    ):
        assert forbidden not in source
