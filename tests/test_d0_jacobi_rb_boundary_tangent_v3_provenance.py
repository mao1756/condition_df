from __future__ import annotations

import json
from pathlib import Path
import shutil
import zipfile

import pytest

from mnist.d0_jacobi_artifacts import config_fingerprint
from mnist.d0_jacobi_rb_boundary_tangent_v3_provenance import (
    BoundaryTangentV3ProvenanceError,
    FAILED_V3_PREFLIGHT_READJUDICATED_DECISION,
    FAILED_V3_PREFLIGHT_REGISTRY_COUNT,
    FAILED_V3_PREFLIGHT_REGISTRY_FILE_SHA256,
    FAILED_V3_PREFLIGHT_REGISTRY_SEMANTIC_SHA256,
    FAILED_V3_PREFLIGHT_RUN_BASENAME,
    PRODUCTION_RESERVATION,
    build_v3_cohort_plan,
    build_v3_path_plan,
    v3_source_fingerprint,
    v3_source_paths,
    v3_transitive_source_paths,
    validate_no_v3_baseline_artifacts,
    validate_v3_cohort_plan,
    validate_v3_path_plan,
    verify_and_re_adjudicate_failed_v3_preflight,
    verify_v3_resume_compatibility,
    verify_v3_source_image_binding,
)


def _hashed(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["semantic_sha256"] = config_fingerprint(result)
    return result


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _failed_v3_preflight_run() -> Path:
    return Path(
        "runs/experiment12_d0_jacobi_rb_boundary_tangent_v3_learnability"
    ).resolve() / FAILED_V3_PREFLIGHT_RUN_BASENAME


def test_exact_path_plan_and_allocator_only_overlap() -> None:
    plan = build_v3_path_plan()
    roles = plan["roles"]
    assert roles["preflight_seam"] == list(range(0xF0000, 0xF0008))
    assert roles["train"] == list(range(0xF1000, 0xF1040))
    assert roles["validation"] == list(range(0xF1100, 0xF1120))
    assert roles["confirmation"] == list(range(0xF2000, 0xF2040))

    result = validate_v3_path_plan(
        plan,
        claimed_ids={
            "future_production_reserved": range(*PRODUCTION_RESERVATION),
            "historical": range(0xED000, 0xED040),
        },
    )
    assert result["passed"] == 1
    assert result["collision_count"] == 0

    with pytest.raises(BoundaryTangentV3ProvenanceError, match="collide"):
        validate_v3_path_plan(plan, claimed_ids={"active_run": [0xF1000]})
    with pytest.raises(BoundaryTangentV3ProvenanceError):
        validate_v3_path_plan(plan, claimed_ids={"bad": [1 << 20]})


def test_path_plan_tamper_and_exact_twenty_bit_boundary_fail() -> None:
    plan = build_v3_path_plan()
    changed = dict(plan)
    changed["root_seed"] = 1
    with pytest.raises(BoundaryTangentV3ProvenanceError, match="changed"):
        validate_v3_path_plan(changed)
    validate_v3_path_plan(plan, claimed_ids={"outside": [(1 << 20) - 1]})
    with pytest.raises(BoundaryTangentV3ProvenanceError, match="20-bit"):
        validate_v3_path_plan(plan, claimed_ids={"outside": [1 << 20]})
    with pytest.raises(BoundaryTangentV3ProvenanceError, match="integer"):
        validate_v3_path_plan(plan, claimed_ids={"outside": [True]})


def test_exact_cohort_plan_and_cross_role_split() -> None:
    path_plan = build_v3_path_plan()
    cohorts = build_v3_cohort_plan(path_plan)
    assert cohorts["train_validation_sizes"] == [10] * 9 + [6]
    assert cohorts["confirmation_sizes"] == [10] * 6 + [4]
    mixed = cohorts["train_validation"][6]
    assert mixed["path_roles"] == ["train"] * 4 + ["validation"] * 6
    assert max(record["size"] for record in cohorts["train_validation"]) == 10
    assert validate_v3_cohort_plan(cohorts, path_plan=path_plan)["passed"] == 1
    changed = dict(cohorts)
    changed["mixed_train_validation_cohort_index"] = 7
    with pytest.raises(BoundaryTangentV3ProvenanceError, match="changed"):
        validate_v3_cohort_plan(changed, path_plan=path_plan)


def test_source_closure_is_normalized_and_hashable(tmp_path: Path) -> None:
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    first.write_text("a = 1\n", encoding="utf-8")
    second.write_text("b = 2\n", encoding="utf-8")
    paths = v3_source_paths([second, first, first])
    assert paths == tuple(sorted((first.resolve(), second.resolve()), key=lambda p: p.as_posix()))
    assert v3_source_fingerprint([second, first]) == v3_source_fingerprint([first, second])
    with pytest.raises(BoundaryTangentV3ProvenanceError):
        v3_source_paths([tmp_path / "missing.py"])


def test_transitive_source_closure_contains_all_scientific_dependencies() -> None:
    entry = Path("mnist/diag_d0_jacobi_rb_boundary_tangent_v3_learnability.py")
    names = {path.name for path in v3_transitive_source_paths((entry,))}
    assert {
        "__init__.py",
        "d0_jacobi_rb_boundary_tangent_gate.py",
        "d0_jacobi_rb_boundary_tangent_provenance.py",
        "d0_jacobi_rb_boundary_tangent_eager_gate.py",
        "d0_jacobi_rb_boundary_tangent_eager_provenance.py",
        "d0_jacobi_rb_spectral.py",
        "d0_jacobi_rb_cuda_controls.py",
        "d0_jacobi_v3_source_compat.py",
    }.issubset(names)


def test_frozen_source_image_binding_is_measured_and_tamper_evident(
    tmp_path: Path,
) -> None:
    parent = Path(
        "runs/experiment12_d0_jacobi_rb_coarse_residual_learnability/"
        "20260731-140333_production-exact-k512-coarse-residual-one-image"
    ).resolve()
    record = verify_v3_source_image_binding(parent)
    assert record["passed"] == 1
    assert record["source_image_npz_sha256"] == (
        "81904cde32495eb11b73cb688cc458118eb2e5578513426d2f9b881ac4665914"
    )
    assert record["measured_image_sha256"] == (
        "0bb39fec59853f789fe366251cd85ed79ffbffb5a1aaa32084d2dbd2bbb4ea7d"
    )
    assert record["measured_mixed_target_sha256"] == (
        "00ae86fb69be6d86557f15f6f8fa00f8bb3c2514f331863c9638e36d23d135c5"
    )

    changed = tmp_path / parent.name
    changed.mkdir()
    shutil.copy2(parent / "source_image.json", changed / "source_image.json")
    shutil.copy2(parent / "source_image.npz", changed / "source_image.npz")
    with (changed / "source_image.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(BoundaryTangentV3ProvenanceError, match="archive binding"):
        verify_v3_source_image_binding(changed)


def test_baseline_artifact_scan_rejects_fitted_evidence(tmp_path: Path) -> None:
    assert validate_no_v3_baseline_artifacts(tmp_path)["passed"] == 1
    (tmp_path / "tangent_baseline.npz").write_bytes(b"not allowed")
    with pytest.raises(BoundaryTangentV3ProvenanceError, match="fitted-baseline"):
        validate_no_v3_baseline_artifacts(tmp_path)


def test_baseline_artifact_scan_inspects_checkpoint_pickle_without_loading_tensors(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed.pt"
    with zipfile.ZipFile(allowed, mode="w") as archive:
        archive.writestr("checkpoint/data.pkl", b"model.weight")
        archive.writestr("checkpoint/data/0", b"_q_values in tensor bytes is irrelevant")
    assert validate_no_v3_baseline_artifacts(tmp_path)["passed"] == 1

    checkpoint = tmp_path / "checkpoint.pt"
    with zipfile.ZipFile(checkpoint, mode="w") as archive:
        archive.writestr("checkpoint/data.pkl", b"model._q_values")
        archive.writestr("checkpoint/data/0", b"tensor payload is not loaded")
    with pytest.raises(BoundaryTangentV3ProvenanceError, match="fitted-baseline"):
        validate_no_v3_baseline_artifacts(tmp_path)


def test_exact_failed_v3_preflight_is_read_only_and_readjudicated() -> None:
    root = _failed_v3_preflight_run()
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    record = verify_and_re_adjudicate_failed_v3_preflight(root)
    after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    assert after == before
    assert record["passed"] == 1
    assert record["readjudicated_decision"] == (
        FAILED_V3_PREFLIGHT_READJUDICATED_DECISION
    )
    assert record["decision"] == FAILED_V3_PREFLIGHT_READJUDICATED_DECISION
    assert record["failure_domain"] == "implementation_contract"
    assert record["readjudicated_failure_domain"] == "implementation_contract"
    assert record["historical_decision"] == "exact_cache_invalid"
    assert record["stage_execution_valid"] == 1
    assert record["numerically_valid"] == 1
    assert record["resource_valid"] == 1
    assert record["scientific_evidence_complete"] == 1
    assert record["production_path_roles_opened"] == 0
    assert record["downstream_scientific_evidence_opened"] == 0
    assert record["fresh_preflight_required"] == 1
    assert record["cache_generation_authorized"] == 0
    assert record["immutable_registry"] == {
        "artifact_count": FAILED_V3_PREFLIGHT_REGISTRY_COUNT,
        "file_sha256": FAILED_V3_PREFLIGHT_REGISTRY_FILE_SHA256,
        "semantic_sha256": FAILED_V3_PREFLIGHT_REGISTRY_SEMANTIC_SHA256,
        "complete_file_set_verified": 1,
    }
    basis = record["readjudication_basis"]
    assert basis["base_output_digest_bit_identical"] == 1
    assert basis[
        "base_output_digest_includes_later_target_and_certificate_codes"
    ] == 1
    assert basis["proof_effort_metadata_digest_bit_identical"] == 0
    assert basis["proof_effort_metadata_equality_required"] == 0
    assert record["semantic_sha256"] == config_fingerprint(
        {name: value for name, value in record.items() if name != "semantic_sha256"}
    )


def test_failed_v3_preflight_registry_artifact_and_terminal_tamper_fail(
    tmp_path: Path,
) -> None:
    source = _failed_v3_preflight_run()

    artifact_copy = tmp_path / "artifact" / FAILED_V3_PREFLIGHT_RUN_BASENAME
    shutil.copytree(source, artifact_copy)
    with (artifact_copy / "preflight_scheduler_seam.json").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(BoundaryTangentV3ProvenanceError, match="artifact changed"):
        verify_and_re_adjudicate_failed_v3_preflight(artifact_copy)

    terminal_copy = tmp_path / "terminal" / FAILED_V3_PREFLIGHT_RUN_BASENAME
    shutil.copytree(source, terminal_copy)
    status = json.loads((terminal_copy / "run_status.json").read_text())
    status["decision"] = "tampered"
    _write(terminal_copy / "run_status.json", status)
    with pytest.raises(BoundaryTangentV3ProvenanceError, match="terminal file changed"):
        verify_and_re_adjudicate_failed_v3_preflight(terminal_copy)


def test_failed_v3_preflight_rejects_unregistered_downstream_evidence(
    tmp_path: Path,
) -> None:
    source = _failed_v3_preflight_run()
    changed = tmp_path / FAILED_V3_PREFLIGHT_RUN_BASENAME
    shutil.copytree(source, changed)
    _write(
        changed / "cache_metrics.json",
        {"production_cache_generation_performed": 1},
    )
    with pytest.raises(BoundaryTangentV3ProvenanceError, match="terminal file set"):
        verify_and_re_adjudicate_failed_v3_preflight(changed)


def test_resume_verifies_all_hash_bindings_before_mutation(tmp_path: Path) -> None:
    path_plan = build_v3_path_plan()
    cohort_plan = build_v3_cohort_plan(path_plan)
    parent = _hashed({"schema": "parent", "passed": 1})
    adjudication = _hashed({"schema": "adjudication", "passed": 1})
    authorization = _hashed({"schema": "authorization", "passed": 1})
    baseline = _hashed({"schema": "zero", "formula": "q_B := 0"})
    target = _hashed({"schema": "target", "target_modified": 0})
    config = _hashed(
        {
            "schema": "config",
            "root_seed": 261_311,
            "forbidden_scheduler_benchmark_seed": 261_321,
        }
    )
    records = {
        "parent_provenance.json": parent,
        "adjudication_provenance.json": adjudication,
        "adjudication_authorization.json": authorization,
        "path_id_plan.json": path_plan,
        "cohort_plan.json": cohort_plan,
        "zero_baseline_contract.json": baseline,
        "target_and_input_contract.json": target,
        "scientific_config.json": config,
    }
    for name, record in records.items():
        _write(tmp_path / name, record)
    expected = {
        "source_fingerprint": "1" * 64,
        "scientific_config_sha256": config["semantic_sha256"],
        "parent_provenance_sha256": parent["semantic_sha256"],
        "adjudication_provenance_sha256": adjudication["semantic_sha256"],
        "adjudication_authorization_sha256": authorization["semantic_sha256"],
        "path_plan_sha256": path_plan["semantic_sha256"],
        "cohort_plan_sha256": cohort_plan["semantic_sha256"],
        "zero_baseline_contract_sha256": baseline["semantic_sha256"],
        "target_and_input_contract_sha256": target["semantic_sha256"],
    }
    _write(tmp_path / "run_manifest.json", expected)
    result = verify_v3_resume_compatibility(
        tmp_path,
        source_fingerprint_value=expected["source_fingerprint"],
        scientific_config_sha256=expected["scientific_config_sha256"],
        parent_provenance_sha256=expected["parent_provenance_sha256"],
        adjudication_provenance_sha256=expected["adjudication_provenance_sha256"],
        adjudication_authorization_sha256=expected[
            "adjudication_authorization_sha256"
        ],
        path_plan_sha256=expected["path_plan_sha256"],
        cohort_plan_sha256=expected["cohort_plan_sha256"],
        zero_baseline_contract_sha256=expected["zero_baseline_contract_sha256"],
        target_and_input_contract_sha256=expected[
            "target_and_input_contract_sha256"
        ],
    )
    assert result["passed"] == 1

    manifest_path = tmp_path / "run_manifest.json"
    before = manifest_path.read_bytes()
    with pytest.raises(BoundaryTangentV3ProvenanceError, match="compatibility"):
        verify_v3_resume_compatibility(
            tmp_path,
            source_fingerprint_value="2" * 64,
            scientific_config_sha256=expected["scientific_config_sha256"],
            parent_provenance_sha256=expected["parent_provenance_sha256"],
            adjudication_provenance_sha256=expected["adjudication_provenance_sha256"],
            adjudication_authorization_sha256=expected[
                "adjudication_authorization_sha256"
            ],
            path_plan_sha256=expected["path_plan_sha256"],
            cohort_plan_sha256=expected["cohort_plan_sha256"],
            zero_baseline_contract_sha256=expected["zero_baseline_contract_sha256"],
            target_and_input_contract_sha256=expected[
                "target_and_input_contract_sha256"
            ],
        )
    assert manifest_path.read_bytes() == before
