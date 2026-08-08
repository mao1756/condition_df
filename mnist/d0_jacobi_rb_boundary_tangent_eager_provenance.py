"""Immutable evidence binding for the boundary-tangent v2 continuation.

The successful eager-prefix timing run supersedes the legacy schedule's
resource forecast, but it does not alter either historical decision.  This
module verifies both terminal runs, re-verifies the coarse/affine ancestry of
the failed boundary-tangent run, and proves that its production path roles
were never opened.  It deliberately imports no kernel, trainer, controller,
or sampler.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from mnist.d0_jacobi_artifacts import (
    ArtifactCompatibilityError,
    config_fingerprint,
    file_fingerprint,
    source_fingerprint,
)
from mnist.d0_jacobi_rb_boundary_tangent_schedule_provenance import (
    FAILED_REGISTRY_COUNT as LEGACY_FAILED_REGISTRY_COUNT,
    FAILED_REGISTRY_FILE_SHA256 as LEGACY_FAILED_REGISTRY_FILE_SHA256,
    FAILED_REGISTRY_SEMANTIC_SHA256 as LEGACY_FAILED_REGISTRY_SEMANTIC_SHA256,
    FAILED_RUN_BASENAME as LEGACY_FAILED_RUN_BASENAME,
    FAILED_SCIENTIFIC_CONFIG_SHA256 as LEGACY_FAILED_CONFIG_SHA256,
    FAILED_SOURCE_FINGERPRINT as LEGACY_FAILED_SOURCE_FINGERPRINT,
    HISTORICAL_DECISION as LEGACY_TANGENT_DECISION,
    READJUDICATED_DECISION as LEGACY_SCHEDULE_DECISION,
    verify_and_readjudicate_boundary_tangent_schedule_parents,
)


SCHEMA = "experiment12-d0-jacobi-rb-boundary-tangent-eager-provenance"
SCHEMA_VERSION = 1
READJUDICATED_DECISION = "legacy_schedule_resource_projection_superseded"
PATH_PLAN_SCHEMA = SCHEMA + "-path-id-plan"
PATH_PLAN_VERSION = "d0-jacobi-rb-boundary-tangent-eager-v2-path-ids-v1"
ROOT_SEED = 261_311
RESERVED_CONTROL_SEED = 261_316
PATH_ID_BITS = 20
PATH_ID_LIMIT = 1 << PATH_ID_BITS
PATH_ROLE_RANGES: dict[str, tuple[int, int]] = {
    "train": (0xEC100, 0xEC140),
    "validation": (0xEC200, 0xEC220),
    "confirmation": (0xED000, 0xED040),
    "preflight_seam": (0xEF000, 0xEF008),
}
HISTORICAL_V1_PREFLIGHT_RANGE = (0xEC000, 0xEC008)

EAGER_RUN_BASENAME = (
    "20260803-034008_production-eager-prefix-complete-pipeline"
)
EAGER_RUN_SCHEMA = (
    "experiment12-d0-jacobi-rb-boundary-tangent-"
    "eager-pipeline-confirmation-v1"
)
EAGER_REGISTRY_COUNT = 615
EAGER_REGISTRY_SEMANTIC_SHA256 = (
    "b85907645f1b11be581f1247268729478fb7b4ff49444181663ac90467792eb7"
)
EAGER_REGISTRY_FILE_SHA256 = (
    "25f936a6839d67789fb19846fd9beed68bdbeb7c86cf7cbd6c894d3bbf6576dd"
)
EAGER_SOURCE_FINGERPRINT = (
    "50e9ebef34ed7981c8dc3a3a72e8c39c75eda162898f6d6648c7a618d85e5b87"
)
EAGER_SCIENTIFIC_CONFIG_SHA256 = (
    "e8df891e1efac66cdda4bd0aeda54bdd1b581f192b30f5ce22fe96411c615eaa"
)
EAGER_DECISION = "exact_boundary_tangent_eager_pipeline_feasible"
EAGER_PREFIX_PARENT_DECISION = "eager_prefix_profile_computationally_infeasible"
EAGER_PREFIX_PARENT_PROVENANCE_SHA256 = (
    "ff22ef8e17f7d79ce93c660d52f8714d50ca00b91cbcf29a2bb6c939049be04d"
)

FAILED_TANGENT_RUN_BASENAME = LEGACY_FAILED_RUN_BASENAME
FAILED_TANGENT_RUN_SCHEMA = (
    "experiment12-d0-jacobi-rb-boundary-tangent-controller-v1"
)
FAILED_TANGENT_REGISTRY_COUNT = LEGACY_FAILED_REGISTRY_COUNT
FAILED_TANGENT_REGISTRY_SEMANTIC_SHA256 = (
    LEGACY_FAILED_REGISTRY_SEMANTIC_SHA256
)
FAILED_TANGENT_REGISTRY_FILE_SHA256 = LEGACY_FAILED_REGISTRY_FILE_SHA256
FAILED_TANGENT_SOURCE_FINGERPRINT = LEGACY_FAILED_SOURCE_FINGERPRINT
FAILED_TANGENT_SCIENTIFIC_CONFIG_SHA256 = LEGACY_FAILED_CONFIG_SHA256
FAILED_TANGENT_PARENT_PROVENANCE_SHA256 = (
    "3fa216445055636a690fbbebef939d5553146960b98063164f9ebd8b570ac859"
)
FAILED_TANGENT_PATH_PLAN_SHA256 = (
    "5553e1416b9eb9491e377222a5ccacb8fe26c00e5aeeba55749d8d0b50f94fad"
)
FAILED_TANGENT_DECISION = LEGACY_TANGENT_DECISION

COARSE_PARENT_DECISION = "exact_rb_coarse_residual_learnable"
FAILED_AFFINE_PARENT_DECISION = "controller_boundary_or_conservation_failed"

PROJECTED_BASE_TRANSITIONS = 224_788_480
PROJECTED_MIDPOINT_TRANSITIONS = 112_394_240
PROJECTED_TOTAL_TRANSITIONS = 337_182_720
PROJECTED_SECONDS = 93_545.67684082314
PROJECTED_HOURS = 25.984910233561983
PROJECTED_EFFECTIVE_RATE = 3_604.471434567184
PROJECTED_PERSISTED_BYTES = 1_214_005_704
MAXIMUM_PROJECTED_SECONDS = 108_000.0
MINIMUM_PROJECTED_EFFECTIVE_RATE = 3_122.0622222222223
MAXIMUM_MASS_ERROR = 4.440892098500626e-16

_EAGER_REGISTRY_SEMANTICS = {
    "excluded_paths": [
        "artifact_registry.json",
        "eager_pipeline_decision.json",
        "run_status.json",
        "workflow_gate.json",
    ],
    "snapshot_kind": "current-exact-restartable-pipeline",
}
_EAGER_FILE_SHA256 = {
    "artifact_registry.json": EAGER_REGISTRY_FILE_SHA256,
    "run_manifest.json": (
        "f75b7d8bf3c65c5e9be9bc7081096f24e276413f49dd5df1e628b97fe2fd2e34"
    ),
    "scientific_config.json": (
        "7bffd03e70df17a1b3c47ab63a3b42e0872d5aed6157bd55ddad3f77e5da401c"
    ),
    "run_status.json": (
        "1382f56f75e4635b156e8de2079e75f45a7ceef4649fea9c9451b97ef6e6aa81"
    ),
    "workflow_gate.json": (
        "60172717600f261c307a031d9aa36f7dab7d0e17950171aa4f2759a5f6913494"
    ),
    "eager_pipeline_decision.json": (
        "9418dbb59286d71d415f67559a00d580bd64b842f9e16cc7ad69834f44ae431e"
    ),
    "eager_pipeline_preflight_gate.json": (
        "db81e23ae38d58f2f6e52c4336d36df555aecff3a777e2dfe48e34551a928f65"
    ),
    "eager_pipeline_gate.json": (
        "0de68f188657cabd45f5139076c5976e95176f18c0aed0de4d6198c87b7ee88d"
    ),
    "eager_pipeline_metrics.json": (
        "b320bcf68e2364214518c2e958b3aa529a569df75b57c5ca3db995c763c09f55"
    ),
    "eager_pipeline_projection.json": (
        "872544be1c1c8549cd9386f66aeab2cc22646d3838123cf1ece1d552bbd66be6"
    ),
    "parent_provenance.json": (
        "decdd96f75b9a6c8b7e73bfdd7329fcb200794c10963a3517d58ee31ae71f976"
    ),
    "path_id_plan.json": (
        "9eeee20b9f109c7515391b1c8cff78d9184b61be003e435cabbb528dbed734e5"
    ),
}
_FAILED_TANGENT_FILE_SHA256 = {
    "artifact_registry.json": FAILED_TANGENT_REGISTRY_FILE_SHA256,
    "run_manifest.json": (
        "d775ff4c449ac12efcbb68ce881e79ac4b5ddbfdd00ffec9d6a10c1728e18846"
    ),
    "scientific_config.json": (
        "d4daed29e169ac0a9969d5c266e2ede94781d2f771d22f865462695bdd6d314c"
    ),
    "run_status.json": (
        "a9349db983f2a3456be68fbc4d783a64e55b5bc87ca47e5191f4cdfa300f0527"
    ),
    "workflow_gate.json": (
        "fbd27d019f40c641b900501f01c666de4b6d47586ac636cd4e6d870b66545153"
    ),
    "controller_decision.json": (
        "d66617a7c0892c79428b31072bf2ac94caaa1a814641409c08475a364ac2a9e5"
    ),
    "preflight_gate.json": (
        "54dc9c524f90979d6c58ba799793c2fdf92d3b11f26cbb7ac28f7bb95d382ad3"
    ),
    "resource_projection.json": (
        "e0276cf391bf8921a923f1ce7559d161fafa55bcf650367f1d30f090da219542"
    ),
    "parent_provenance.json": (
        "f543eec2ae709649cd49ecb2e2a2bf422ad2fc86701c70b9ac2993e7212ead13"
    ),
    "failed_controller_readjudication.json": (
        "0fdbdd9a2032321bb12c57b94c1fce409abac765b559e3ef36adddf9dcba159f"
    ),
    "path_id_plan.json": (
        "92481c7544c2ee6370f3443e6b81b2d8c8aa9a1fecd20a225f260b1c46596ab5"
    ),
}

_EAGER_SOURCE_PATHS = (
    "mnist/d0_jacobi_artifacts.py",
    "mnist/d0_jacobi_rb_boundary_tangent_cache.py",
    "mnist/d0_jacobi_rb_boundary_tangent_eager_pipeline_gate.py",
    "mnist/d0_jacobi_rb_boundary_tangent_eager_pipeline_provenance.py",
    "mnist/d0_jacobi_rb_boundary_tangent_eager_prefix_provenance.py",
    "mnist/d0_jacobi_rb_boundary_tangent_prefix_fallback.py",
    "mnist/d0_jacobi_rb_boundary_tangent_prefix_schedule.py",
    "mnist/d0_jacobi_rb_boundary_tangent_prefix_schedule_gate.py",
    "mnist/d0_jacobi_rb_boundary_tangent_schedule.py",
    "mnist/d0_jacobi_rb_boundary_tangent_schedule_gate.py",
    "mnist/d0_jacobi_rb_boundary_tangent_schedule_provenance.py",
    "mnist/d0_jacobi_rb_coarse_residual.py",
    "mnist/d0_jacobi_rb_controls.py",
    "mnist/d0_jacobi_rb_cuda.py",
    "mnist/d0_jacobi_rb_cuda_certificate.py",
    "mnist/d0_jacobi_rb_cuda_controls.py",
    "mnist/d0_jacobi_rb_cuda_fused.py",
    "mnist/d0_jacobi_rb_cuda_multipath.py",
    "mnist/d0_jacobi_rb_learnability.py",
    "mnist/d0_jacobi_rb_reverse_controller.py",
    "mnist/d0_jacobi_rb_spectral.py",
    "mnist/d0_jacobi_source_compat.py",
    "mnist/diag_d0_jacobi_rb_boundary_tangent_eager_pipeline_confirmation.py",
    "mnist/diag_d0_jacobi_rb_boundary_tangent_prefix_schedule_confirmation.py",
    "mnist/diag_d0_jacobi_rb_boundary_tangent_schedule_feasibility.py",
)
_FAILED_TANGENT_SOURCE_PATHS = (
    "mnist/d0_jacobi_rb_boundary_tangent.py",
    "mnist/d0_jacobi_rb_boundary_tangent_cache.py",
    "mnist/d0_jacobi_rb_boundary_tangent_confirmation.py",
    "mnist/d0_jacobi_rb_boundary_tangent_gate.py",
    "mnist/d0_jacobi_rb_boundary_tangent_provenance.py",
    "mnist/diag_d0_jacobi_rb_boundary_tangent_controller_confirmation.py",
)

_FAILED_TANGENT_PATH_ROLES = {
    "preflight_benchmark": (0xEC000, 0xEC008),
    "train": (0xEC100, 0xEC140),
    "validation": (0xEC200, 0xEC220),
    "confirmation": (0xED000, 0xED040),
}
_UNOPENED_PRODUCTION_ROLES = ("train", "validation", "confirmation")
_DOWNSTREAM_PATH_PREFIXES = (
    "cache/",
    "checkpoints/",
    "confirmation/",
    "control/",
    "train/",
    "validation/",
)
_DOWNSTREAM_PATHS = (
    "cache",
    "checkpoints",
    "confirmation",
    "control",
    "train",
    "validation",
    "cache_gate.json",
    "train_gate.json",
    "confirm_gate.json",
    "control_gate.json",
    "cache_index.json",
    "selected_model.pt",
    "confirmation_open.json",
)

NO_WORK_FIELDS = (
    "physical_training_performed",
    "controller_control_trajectory_performed",
    "full_reverse_path_performed",
    "image_sampling_performed",
    "sampling_performed",
    "reverse_sampling_performed",
    "reconstruction_performed",
    "production_cache_generation_performed",
)
NO_AUTHORIZATION_FIELDS = (
    "cache_generation_authorized",
    "physical_training_authorized",
    "training_authorized",
    "full_dataset_training_authorized",
    "controller_control_trajectory_authorized",
    "controller_trajectory_authorized",
    "image_sampling_authorized",
    "sampling_authorized",
    "reverse_sampling_authorized",
    "reconstruction_authorized",
    "reconstruction_claim_authorized",
    "one_image_reconstruction_control_planning_authorized",
)


class EagerBoundaryTangentProvenanceError(ArtifactCompatibilityError):
    """An immutable parent or its live source closure changed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EagerBoundaryTangentProvenanceError(message)


def _load_json(path: Path, description: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EagerBoundaryTangentProvenanceError(
            f"invalid {description}: {path}"
        ) from exc
    _require(isinstance(value, Mapping), f"{description} must be a JSON object")
    return dict(value)


def _semantic_body(record: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(record)
    body.pop("semantic_sha256", None)
    return body


def _assert_semantic(record: Mapping[str, Any], description: str) -> None:
    _require(
        record.get("semantic_sha256") == config_fingerprint(_semantic_body(record)),
        f"{description} semantic hash changed",
    )


def _assert_no_work(record: Mapping[str, Any], description: str) -> None:
    for field in NO_WORK_FIELDS:
        if field in record:
            _require(int(record[field]) == 0, f"{description} records {field}")
    if "production_cache_generated" in record:
        _require(
            int(record["production_cache_generated"]) == 0,
            f"{description} records production_cache_generated",
        )


def _assert_no_authorization(record: Mapping[str, Any], description: str) -> None:
    for field in NO_AUTHORIZATION_FIELDS:
        if field in record:
            _require(int(record[field]) == 0, f"{description} authorizes {field}")


def _safe_registry_path(root: Path, value: Any) -> tuple[str, Path]:
    _require(isinstance(value, str) and bool(value), "registry path is invalid")
    relative = PurePosixPath(value)
    _require(
        not relative.is_absolute()
        and all(part not in {"", ".", ".."} for part in relative.parts),
        f"unsafe registry path: {value!r}",
    )
    target = (root / Path(*relative.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise EagerBoundaryTangentProvenanceError(
            f"registry path escapes parent: {value!r}"
        ) from exc
    return relative.as_posix(), target


def _assert_file_hashes(
    root: Path, expected: Mapping[str, str], description: str
) -> None:
    for relative, expected_sha256 in expected.items():
        path = root / relative
        _require(
            path.is_file() and file_fingerprint(path) == expected_sha256,
            f"{description} file changed: {relative}",
        )


def _verify_registry(
    root: Path,
    *,
    description: str,
    schema: str,
    artifact_count: int,
    semantic_sha256: str,
    file_sha256: str,
    excluded_paths: set[str],
    registry_semantics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    path = root / "artifact_registry.json"
    _require(
        path.is_file() and file_fingerprint(path) == file_sha256,
        f"{description} registry file hash changed",
    )
    registry = _load_json(path, f"{description} registry")
    artifacts = registry.get("artifacts")
    _require(
        registry.get("schema") == schema + "-artifact-registry"
        and registry.get("schema_version") == 1
        and isinstance(artifacts, list)
        and int(registry.get("artifact_count", -1)) == artifact_count
        and len(artifacts) == artifact_count
        and registry.get("semantic_sha256") == semantic_sha256,
        f"{description} terminal registry changed",
    )
    if registry_semantics is None:
        _require(
            "registry_semantics" not in registry
            and config_fingerprint({"artifacts": artifacts}) == semantic_sha256,
            f"{description} registry semantics changed",
        )
    else:
        semantics = dict(registry_semantics)
        _require(
            registry.get("registry_semantics") == semantics
            and config_fingerprint(
                {"artifacts": artifacts, "registry_semantics": semantics}
            )
            == semantic_sha256,
            f"{description} registry semantics changed",
        )

    registered: set[str] = set()
    for raw in artifacts:
        _require(isinstance(raw, Mapping), f"{description} registry row is malformed")
        relative, target = _safe_registry_path(root, raw.get("path"))
        _require(relative not in registered, f"{description} registry path duplicated")
        _require(
            target.is_file()
            and int(raw.get("size", -1)) == target.stat().st_size
            and raw.get("sha256") == file_fingerprint(target),
            f"{description} artifact changed: {relative}",
        )
        registered.add(relative)
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
        and item.relative_to(root).as_posix() not in excluded_paths
    }
    _require(actual == registered, f"{description} terminal file set changed")
    _assert_no_work(registry, f"{description} registry")
    _assert_no_authorization(registry, f"{description} registry")
    return registry


def _resolve_live_sources(
    manifest: Mapping[str, Any],
    *,
    expected_relative_paths: tuple[str, ...],
    expected_fingerprint: str,
    description: str,
) -> tuple[Path, ...]:
    raw = manifest.get("source_paths")
    _require(
        isinstance(raw, list)
        and len(raw) == len(expected_relative_paths)
        and all(isinstance(item, str) and item for item in raw),
        f"{description} source list changed",
    )
    repository_root = Path(__file__).resolve().parents[1]
    actual = tuple(
        sorted(
            (
                Path(item).resolve()
                if Path(item).is_absolute()
                else (repository_root / item).resolve()
            )
            for item in raw
        )
    )
    expected = tuple(
        sorted((repository_root / item).resolve() for item in expected_relative_paths)
    )
    _require(actual == expected, f"{description} live source set changed")
    _require(all(path.is_file() for path in actual), f"{description} source is missing")
    _require(
        source_fingerprint(actual) == expected_fingerprint,
        f"{description} live source fingerprint changed",
    )
    return actual


def _all_checks_passed(gate: Mapping[str, Any], description: str) -> bool:
    checks = gate.get("checks")
    _require(
        isinstance(checks, Mapping) and bool(checks),
        f"{description} checks changed",
    )
    return all(
        isinstance(value, Mapping) and int(value.get("passed", 0)) == 1
        for value in checks.values()
    )


def _hashed_record(body: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["semantic_sha256"] = config_fingerprint(result)
    return result


def build_eager_boundary_tangent_path_plan() -> dict[str, Any]:
    """Return the frozen v2 production and seam-check path namespaces."""

    historical_start, historical_stop = HISTORICAL_V1_PREFLIGHT_RANGE
    return _hashed_record(
        {
            "schema": PATH_PLAN_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "path_id_plan_version": PATH_PLAN_VERSION,
            "root_seed": ROOT_SEED,
            "reserved_control_seed": RESERVED_CONTROL_SEED,
            "canonical_path_id_bits": PATH_ID_BITS,
            "roles": {
                role: list(range(start, stop))
                for role, (start, stop) in PATH_ROLE_RANGES.items()
            },
            "role_slots": {
                role: {
                    "start": start,
                    "stop_exclusive": stop,
                    "start_hex": f"0x{start:05X}",
                    "stop_exclusive_hex": f"0x{stop:05X}",
                }
                for role, (start, stop) in PATH_ROLE_RANGES.items()
            },
            "preflight_seam_path_ids": list(
                range(*PATH_ROLE_RANGES["preflight_seam"])
            ),
            "forbidden_historical_v1_path_ids": list(
                range(historical_start, historical_stop)
            ),
            "ancestor_claimed_roles": {
                "v1_preflight_benchmark": list(
                    range(historical_start, historical_stop)
                )
            },
            "fresh_path_count": sum(
                stop - start for start, stop in PATH_ROLE_RANGES.values()
            ),
            "historical_claim_count": historical_stop - historical_start,
            "collision_free": 1,
            "silent_remapping_performed": 0,
        }
    )


def validate_eager_boundary_tangent_path_plan(
    plan: Mapping[str, Any],
    *,
    claimed_ids: Mapping[str, Iterable[int]] | Iterable[int] | None = None,
) -> dict[str, Any]:
    """Reject path drift, overlap, ancestor reuse, and external collisions."""

    expected = build_eager_boundary_tangent_path_plan()
    _require(dict(plan) == expected, "eager boundary-tangent path plan changed")
    active: set[int] = set()
    intervals = list(PATH_ROLE_RANGES.items())
    for index, (role, (start, stop)) in enumerate(intervals):
        _require(0 <= start < stop <= PATH_ID_LIMIT, f"{role} path slot is invalid")
        for other, (other_start, other_stop) in intervals[index + 1 :]:
            _require(
                stop <= other_start or other_stop <= start,
                f"path roles overlap: {role} and {other}",
            )
        active.update(range(start, stop))
    historical = set(range(*HISTORICAL_V1_PREFLIGHT_RANGE))
    _require(
        active.isdisjoint(historical),
        "active path roles reuse the historical v1 preflight reservation",
    )
    if claimed_ids is None:
        claims: Mapping[str, Iterable[int]] = {}
    elif isinstance(claimed_ids, Mapping):
        claims = claimed_ids
    else:
        claims = {"external": claimed_ids}
    collisions: list[dict[str, Any]] = []
    for source, values in claims.items():
        for raw in values:
            _require(
                isinstance(raw, int) and not isinstance(raw, bool),
                "external path claim is not an integer",
            )
            value = int(raw)
            _require(0 <= value < PATH_ID_LIMIT, "external path claim is out of bounds")
            if value in active:
                collisions.append({"source": str(source), "path_id": value})
    _require(not collisions, "eager boundary-tangent path IDs collide with claims")
    return {
        "schema": PATH_PLAN_SCHEMA + "-validation",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": 1,
        "path_id_plan_sha256": expected["semantic_sha256"],
        "path_count": len(active),
        "historical_claim_count": len(historical),
        "collision_count": 0,
        "twenty_bit_bounds_pass": 1,
        "role_disjointness_pass": 1,
        "historical_v1_preflight_reuse": 0,
    }


def eager_boundary_tangent_source_paths(
    paths: Iterable[str | Path] | None = None,
) -> tuple[Path, ...]:
    """Resolve the exact source closure selected by the v2 caller."""

    values = (Path(__file__),) if paths is None else tuple(Path(item) for item in paths)
    resolved = tuple(
        sorted({item.resolve() for item in values}, key=lambda item: item.as_posix())
    )
    _require(bool(resolved), "eager boundary-tangent source set is empty")
    _require(
        all(item.is_file() for item in resolved),
        "eager boundary-tangent source is missing",
    )
    return resolved


def eager_boundary_tangent_source_fingerprint(
    paths: Iterable[str | Path] | None = None,
) -> str:
    """Hash the exact, normalized v2 source closure."""

    return source_fingerprint(eager_boundary_tangent_source_paths(paths))


def verify_eager_boundary_tangent_resume_compatibility(
    run_dir: str | Path,
    *,
    source_fingerprint_value: str,
    scientific_config_sha256: str,
    parent_provenance_sha256: str,
    parent_readjudication_sha256: str,
    path_plan_sha256: str,
) -> dict[str, Any]:
    """Verify every frozen v2 binding before a resumed run is mutated."""

    root = Path(run_dir).resolve()
    _require(root.is_dir(), f"resume run does not exist: {root}")
    manifest = _load_json(root / "run_manifest.json", "resume manifest")
    config = _load_json(root / "scientific_config.json", "resume config")
    records = {
        "parent_provenance": _load_json(
            root / "parent_provenance.json", "resume parent provenance"
        ),
        "parent_readjudication": _load_json(
            root / "parent_readjudication.json", "resume parent readjudication"
        ),
        "path_plan": _load_json(root / "path_id_plan.json", "resume path plan"),
    }
    expected = {
        "source_fingerprint": source_fingerprint_value,
        "scientific_config_sha256": scientific_config_sha256,
        "parent_provenance_sha256": parent_provenance_sha256,
        "parent_readjudication_sha256": parent_readjudication_sha256,
        "path_plan_sha256": path_plan_sha256,
    }
    _require(
        all(manifest.get(name) == value for name, value in expected.items()),
        "resume manifest compatibility changed",
    )
    _require(
        config.get("semantic_sha256") == scientific_config_sha256
        and int(config.get("root_seed", -1)) == ROOT_SEED
        and int(config.get("reserved_control_seed", -1)) == RESERVED_CONTROL_SEED,
        "resume scientific configuration changed",
    )
    _assert_semantic(config, "resume scientific configuration")
    hashes = {
        "parent_provenance": parent_provenance_sha256,
        "parent_readjudication": parent_readjudication_sha256,
        "path_plan": path_plan_sha256,
    }
    for name, record in records.items():
        _require(
            record.get("semantic_sha256") == hashes[name],
            f"resume {name.replace('_', ' ')} changed",
        )
        _assert_semantic(record, f"resume {name.replace('_', ' ')}")
    validate_eager_boundary_tangent_path_plan(records["path_plan"])
    for description, record in {"manifest": manifest, "config": config}.items():
        _assert_no_work(record, f"resume {description}")
        _assert_no_authorization(record, f"resume {description}")
    return _hashed_record(
        {
            "schema": SCHEMA + "-resume-compatibility",
            "schema_version": SCHEMA_VERSION,
            "evaluation_status": "evaluated",
            "passed": 1,
            "run_dir": str(root),
            **expected,
        }
    )


def _verify_eager_parent(root: Path, coarse_root: Path) -> dict[str, Any]:
    registry = _verify_registry(
        root,
        description="successful eager pipeline",
        schema=EAGER_RUN_SCHEMA,
        artifact_count=EAGER_REGISTRY_COUNT,
        semantic_sha256=EAGER_REGISTRY_SEMANTIC_SHA256,
        file_sha256=EAGER_REGISTRY_FILE_SHA256,
        excluded_paths=set(_EAGER_REGISTRY_SEMANTICS["excluded_paths"]),
        registry_semantics=_EAGER_REGISTRY_SEMANTICS,
    )
    _assert_file_hashes(root, _EAGER_FILE_SHA256, "successful eager pipeline")

    manifest = _load_json(root / "run_manifest.json", "eager manifest")
    config = _load_json(root / "scientific_config.json", "eager config")
    status = _load_json(root / "run_status.json", "eager status")
    preflight = _load_json(
        root / "eager_pipeline_preflight_gate.json", "eager preflight"
    )
    pilot = _load_json(root / "eager_pipeline_gate.json", "eager pilot gate")
    metrics = _load_json(root / "eager_pipeline_metrics.json", "eager metrics")
    projection = _load_json(
        root / "eager_pipeline_projection.json", "eager projection"
    )
    workflow = _load_json(root / "workflow_gate.json", "eager workflow")
    decision = _load_json(root / "eager_pipeline_decision.json", "eager decision")
    parent = _load_json(root / "parent_provenance.json", "eager parent provenance")
    path_plan = _load_json(root / "path_id_plan.json", "eager path plan")

    _require(
        manifest.get("schema") == EAGER_RUN_SCHEMA + "-manifest"
        and manifest.get("schema_version") == 1
        and manifest.get("device") == "cuda"
        and manifest.get("source_fingerprint") == EAGER_SOURCE_FINGERPRINT
        and manifest.get("scientific_config_sha256")
        == EAGER_SCIENTIFIC_CONFIG_SHA256
        and manifest.get("parent_provenance_sha256")
        == EAGER_PREFIX_PARENT_PROVENANCE_SHA256
        and Path(str(manifest.get("transitive_coarse_parent_run_dir", ""))).resolve()
        == coarse_root,
        "successful eager manifest binding changed",
    )
    _resolve_live_sources(
        manifest,
        expected_relative_paths=_EAGER_SOURCE_PATHS,
        expected_fingerprint=EAGER_SOURCE_FINGERPRINT,
        description="successful eager pipeline",
    )
    _assert_semantic(config, "successful eager scientific config")
    _require(
        config.get("schema") == EAGER_RUN_SCHEMA + "-scientific-config"
        and config.get("schema_version") == 1
        and config.get("semantic_sha256") == EAGER_SCIENTIFIC_CONFIG_SHA256
        and int(config.get("root_seed", -1)) == 261_321
        and int(config.get("training_paths", -1)) == 64
        and int(config.get("validation_paths", -1)) == 32
        and int(config.get("confirmation_paths", -1)) == 64
        and int(config.get("test_only", 1)) == 0
        and int(config.get("projected_base_transitions", -1))
        == PROJECTED_BASE_TRANSITIONS
        and int(config.get("projected_midpoint_transitions", -1))
        == PROJECTED_MIDPOINT_TRANSITIONS
        and int(config.get("projected_total_transitions", -1))
        == PROJECTED_TOTAL_TRANSITIONS
        and float(config.get("maximum_projected_seconds", -1.0))
        == MAXIMUM_PROJECTED_SECONDS
        and float(config.get("minimum_projected_effective_rate", -1.0))
        == MINIMUM_PROJECTED_EFFECTIVE_RATE,
        "successful eager scientific configuration changed",
    )
    _require(
        status.get("schema") == EAGER_RUN_SCHEMA + "-status"
        and status.get("state") == "complete"
        and status.get("stage") == "pilot"
        and status.get("decision") == EAGER_DECISION
        and status.get("failure_code") is None
        and status.get("failure_domain") is None
        and int(status.get("scientific_evidence_complete", 0)) == 1,
        "successful eager terminal status changed",
    )
    _require(
        preflight.get("schema")
        == "experiment12-d0-jacobi-rb-boundary-tangent-eager-pipeline-gate-preflight"
        and preflight.get("evaluation_status") == "evaluated"
        and int(preflight.get("passed", 0)) == 1
        and int(preflight.get("stage_execution_valid", 0)) == 1
        and int(preflight.get("scientific_evidence_complete", 0)) == 1
        and _all_checks_passed(preflight, "successful eager preflight"),
        "successful eager preflight changed",
    )
    _require(
        pilot.get("schema")
        == "experiment12-d0-jacobi-rb-boundary-tangent-eager-pipeline-gate-pilot"
        and pilot.get("evaluation_status") == "evaluated"
        and int(pilot.get("passed", 0)) == 1
        and pilot.get("failure_domain") is None
        and int(pilot.get("stage_execution_valid", 0)) == 1
        and int(pilot.get("numerically_valid", 0)) == 1
        and int(pilot.get("resource_valid", 0)) == 1
        and int(pilot.get("resource_only_failure", 1)) == 0
        and int(pilot.get("scientific_evidence_complete", 0)) == 1
        and _all_checks_passed(pilot, "successful eager pilot"),
        "successful eager pilot gate changed",
    )
    _require(
        metrics.get("evaluation_status") == "evaluated"
        and int(metrics.get("stage_execution_valid", 0)) == 1
        and int(metrics.get("numerically_valid", 0)) == 1
        and int(metrics.get("pilot_complete", 0)) == 1
        and int(metrics.get("all_profiles_complete", 0)) == 1
        and int(metrics.get("repeat_count", -1)) == 3
        and int(metrics.get("profile_count", -1)) == 4
        and int(metrics.get("projected_transition_count", -1))
        == PROJECTED_TOTAL_TRANSITIONS
        and float(metrics.get("projected_elapsed_seconds", -1.0))
        == PROJECTED_SECONDS
        and float(metrics.get("projected_effective_transitions_per_second", -1.0))
        == PROJECTED_EFFECTIVE_RATE
        and int(metrics.get("projected_persisted_bytes", -1))
        == PROJECTED_PERSISTED_BYTES
        and float(metrics.get("certificate_fraction", -1.0)) == 1.0
        and float(metrics.get("fallback_fraction", -1.0)) == 0.0
        and int(metrics.get("forbidden_event_count", -1)) == 0
        and float(metrics.get("maximum_mass_error", -1.0)) == MAXIMUM_MASS_ERROR
        and int(metrics.get("output_hashes_identical", 0)) == 1
        and int(metrics.get("final_state_hashes_identical", 0)) == 1
        and int(metrics.get("certificate_hashes_identical", 0)) == 1
        and int(metrics.get("repeat_hashes_identical", 0)) == 1
        and int(metrics.get("eager_base_prefix_schedule_valid", 0)) == 1
        and int(metrics.get("eager_branch_prefix_schedule_valid", 0)) == 1,
        "successful eager metrics changed",
    )
    _require(
        projection.get("passed") == 1
        and projection.get("failed_checks") == []
        and int(projection.get("projected_total_transitions", -1))
        == PROJECTED_TOTAL_TRANSITIONS
        and float(projection.get("projected_seconds", -1.0)) == PROJECTED_SECONDS
        and float(projection.get("projected_hours", -1.0)) == PROJECTED_HOURS
        and float(projection.get("projected_effective_rate", -1.0))
        == PROJECTED_EFFECTIVE_RATE
        and int(projection.get("projected_persistence_bytes", -1))
        == PROJECTED_PERSISTED_BYTES
        and float(projection.get("fallback_fraction", -1.0)) == 0.0
        and int(projection.get("forbidden_total", -1)) == 0
        and float(projection.get("maximum_mass_error", -1.0))
        == MAXIMUM_MASS_ERROR,
        "successful eager projection changed",
    )
    components = workflow.get("components")
    _require(
        workflow.get("required_gate") == "pilot"
        and int(workflow.get("required_gate_pass", 0)) == 1
        and workflow.get("evaluation_status") == "evaluated"
        and isinstance(components, Mapping)
        and set(components) == {"preflight", "pilot"}
        and all(
            isinstance(components[name], Mapping)
            and components[name].get("evaluation_status") == "evaluated"
            and int(components[name].get("passed", 0)) == 1
            for name in ("preflight", "pilot")
        )
        and dict(workflow.get("decision", {})).get("decision") == EAGER_DECISION,
        "successful eager workflow changed",
    )
    _require(
        decision.get("schema")
        == "experiment12-d0-jacobi-rb-boundary-tangent-eager-pipeline-gate-decision"
        and decision.get("evaluation_status") == "evaluated"
        and decision.get("decision") == EAGER_DECISION
        and int(decision.get("schedule_integration_authorized", 0)) == 1,
        "successful eager decision changed",
    )
    _assert_semantic(parent, "successful eager parent provenance")
    _require(
        parent.get("semantic_sha256") == EAGER_PREFIX_PARENT_PROVENANCE_SHA256
        and int(parent.get("passed", 0)) == 1
        and parent.get("parent_decision") == EAGER_PREFIX_PARENT_DECISION
        and int(parent.get("only_runtime_checks_failed", 0)) == 1
        and int(parent.get("parent_artifacts_mutated", 1)) == 0,
        "successful eager immediate-parent decision changed",
    )
    _assert_semantic(path_plan, "successful eager path plan")
    _require(
        int(path_plan.get("collision_free", 0)) == 1
        and int(path_plan.get("fresh_path_count", -1)) == 40,
        "successful eager path plan changed",
    )

    for description, record in {
        "manifest": manifest,
        "config": config,
        "status": status,
        "preflight": preflight,
        "pilot": pilot,
        "metrics": metrics,
        "workflow": workflow,
        "decision": decision,
        "parent provenance": parent,
    }.items():
        _assert_no_work(record, f"successful eager {description}")
        _assert_no_authorization(record, f"successful eager {description}")

    return {
        "run_dir": str(root),
        "basename": EAGER_RUN_BASENAME,
        "registry": {
            "artifact_count": int(registry["artifact_count"]),
            "semantic_sha256": EAGER_REGISTRY_SEMANTIC_SHA256,
            "file_sha256": EAGER_REGISTRY_FILE_SHA256,
        },
        "source_fingerprint": EAGER_SOURCE_FINGERPRINT,
        "scientific_config_sha256": EAGER_SCIENTIFIC_CONFIG_SHA256,
        "decision": EAGER_DECISION,
        "immediate_parent_decision": EAGER_PREFIX_PARENT_DECISION,
        "scientific_evidence_complete": 1,
        "numerically_valid": 1,
        "resource_valid": 1,
        "projected_transition_count": PROJECTED_TOTAL_TRANSITIONS,
        "projected_seconds": PROJECTED_SECONDS,
        "projected_hours": PROJECTED_HOURS,
        "projected_effective_rate": PROJECTED_EFFECTIVE_RATE,
        "projected_persisted_bytes": PROJECTED_PERSISTED_BYTES,
        "schedule_integration_authorized": 1,
    }


def _verify_unopened_production_roles(
    root: Path, registry: Mapping[str, Any]
) -> dict[str, Any]:
    path_plan = _load_json(root / "path_id_plan.json", "failed tangent path plan")
    _assert_semantic(path_plan, "failed tangent path plan")
    roles = path_plan.get("roles")
    slots = path_plan.get("role_slots")
    _require(
        path_plan.get("semantic_sha256") == FAILED_TANGENT_PATH_PLAN_SHA256
        and int(path_plan.get("collision_free", 0)) == 1
        and int(path_plan.get("fresh_path_count", -1)) == 168
        and isinstance(roles, Mapping)
        and isinstance(slots, Mapping)
        and set(roles) == set(_FAILED_TANGENT_PATH_ROLES),
        "failed tangent path plan changed",
    )
    for role, (start, stop) in _FAILED_TANGENT_PATH_ROLES.items():
        slot = slots.get(role)
        _require(
            roles.get(role) == list(range(start, stop))
            and isinstance(slot, Mapping)
            and int(slot.get("start", -1)) == start
            and int(slot.get("stop_exclusive", -1)) == stop,
            f"failed tangent {role} path role changed",
        )

    artifact_paths = {
        str(row.get("path"))
        for row in registry.get("artifacts", [])
        if isinstance(row, Mapping)
    }
    _require(
        not any(
            path.startswith(prefix)
            for path in artifact_paths
            for prefix in _DOWNSTREAM_PATH_PREFIXES
        )
        and not any((root / path).exists() for path in _DOWNSTREAM_PATHS),
        "failed tangent production path role was opened",
    )
    return {
        role: {
            "start": start,
            "start_hex": f"0x{start:05X}",
            "stop_exclusive": stop,
            "stop_exclusive_hex": f"0x{stop:05X}",
            "path_count": stop - start,
            "opened": 0,
        }
        for role, (start, stop) in _FAILED_TANGENT_PATH_ROLES.items()
        if role in _UNOPENED_PRODUCTION_ROLES
    }


def _verify_failed_tangent_parent(
    root: Path,
    coarse_root: Path,
    legacy: Mapping[str, Any],
    explicit_affine_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = _verify_registry(
        root,
        description="failed boundary-tangent",
        schema=FAILED_TANGENT_RUN_SCHEMA,
        artifact_count=FAILED_TANGENT_REGISTRY_COUNT,
        semantic_sha256=FAILED_TANGENT_REGISTRY_SEMANTIC_SHA256,
        file_sha256=FAILED_TANGENT_REGISTRY_FILE_SHA256,
        excluded_paths={"artifact_registry.json", "run_status.json"},
        registry_semantics=None,
    )
    _assert_file_hashes(
        root, _FAILED_TANGENT_FILE_SHA256, "failed boundary-tangent"
    )
    manifest = _load_json(root / "run_manifest.json", "failed tangent manifest")
    status = _load_json(root / "run_status.json", "failed tangent status")
    decision = _load_json(
        root / "controller_decision.json", "failed tangent decision"
    )
    parent = _load_json(
        root / "parent_provenance.json", "failed tangent parent provenance"
    )
    _require(
        manifest.get("schema") == FAILED_TANGENT_RUN_SCHEMA + "-manifest"
        and manifest.get("source_fingerprint")
        == FAILED_TANGENT_SOURCE_FINGERPRINT
        and manifest.get("scientific_config_sha256")
        == FAILED_TANGENT_SCIENTIFIC_CONFIG_SHA256
        and manifest.get("parent_provenance_sha256")
        == FAILED_TANGENT_PARENT_PROVENANCE_SHA256,
        "failed tangent manifest binding changed",
    )
    _resolve_live_sources(
        manifest,
        expected_relative_paths=_FAILED_TANGENT_SOURCE_PATHS,
        expected_fingerprint=FAILED_TANGENT_SOURCE_FINGERPRINT,
        description="failed boundary-tangent",
    )
    _require(
        status.get("state") == "gate_failed"
        and status.get("stage") == "preflight"
        and status.get("decision") == FAILED_TANGENT_DECISION,
        "failed tangent status changed",
    )
    _require(
        decision.get("decision") == FAILED_TANGENT_DECISION
        and decision.get("evaluation_status") == "evaluated"
        and int(decision.get("physical_training_performed", 1)) == 0
        and int(decision.get("controller_control_trajectory_performed", 1)) == 0,
        "failed tangent historical decision changed",
    )
    _assert_semantic(parent, "failed tangent parent provenance")
    _require(
        parent.get("semantic_sha256") == FAILED_TANGENT_PARENT_PROVENANCE_SHA256
        and int(parent.get("passed", 0)) == 1
        and int(parent.get("transitive_parent_binding_pass", 0)) == 1,
        "failed tangent transitive parent evidence changed",
    )
    parents = parent.get("parents")
    _require(isinstance(parents, Mapping), "failed tangent parent map changed")
    coarse = dict(parents.get("successful_coarse_residual", {}))
    affine = dict(parents.get("failed_affine_reverse_controller", {}))
    affine_root = Path(str(affine.get("run_dir", ""))).resolve()
    _require(
        Path(str(coarse.get("run_dir", ""))).resolve() == coarse_root
        and coarse.get("terminal", {}).get("decision") == COARSE_PARENT_DECISION
        and affine.get("terminal", {}).get("decision")
        == FAILED_AFFINE_PARENT_DECISION
        and int(coarse.get("verified", 0)) == 1
        and int(affine.get("verified", 0)) == 1,
        "failed tangent coarse/affine parent decisions changed",
    )
    if explicit_affine_root is not None:
        _require(
            affine_root == explicit_affine_root,
            "failed tangent transitive affine run path changed",
        )
    _require(
        int(legacy.get("passed", 0)) == 1
        and legacy.get("historical_decision") == FAILED_TANGENT_DECISION
        and legacy.get("readjudicated_decision") == LEGACY_SCHEDULE_DECISION
        and int(legacy.get("scientific_evidence_complete", 0)) == 1
        and int(legacy.get("numerically_valid", 0)) == 1
        and int(legacy.get("resource_valid", 1)) == 0,
        "legacy schedule readjudication changed",
    )
    unopened_roles = _verify_unopened_production_roles(root, registry)
    _require(
        sum(int(role["path_count"]) for role in unopened_roles.values()) == 160,
        "failed tangent unopened production path count changed",
    )
    for description, record in {
        "manifest": manifest,
        "status": status,
        "decision": decision,
    }.items():
        _assert_no_work(record, f"failed tangent {description}")
        _assert_no_authorization(record, f"failed tangent {description}")
    return (
        {
            "run_dir": str(root),
            "basename": FAILED_TANGENT_RUN_BASENAME,
            "registry": {
                "artifact_count": int(registry["artifact_count"]),
                "semantic_sha256": FAILED_TANGENT_REGISTRY_SEMANTIC_SHA256,
                "file_sha256": FAILED_TANGENT_REGISTRY_FILE_SHA256,
            },
            "source_fingerprint": FAILED_TANGENT_SOURCE_FINGERPRINT,
            "scientific_config_sha256": FAILED_TANGENT_SCIENTIFIC_CONFIG_SHA256,
            "historical_decision": FAILED_TANGENT_DECISION,
            "legacy_schedule_readjudication": LEGACY_SCHEDULE_DECISION,
            "preflight_benchmark_path_count": 8,
            "unopened_production_path_count": 160,
            "unopened_production_roles": unopened_roles,
            "production_path_roles_unopened": 1,
        },
        {
            "run_dir": str(affine_root),
            "basename": affine.get("basename"),
            "registry": dict(affine.get("registry", {})),
            "source_fingerprint": affine.get("source_fingerprint"),
            "scientific_config_sha256": affine.get("scientific_config_sha256"),
            "decision": FAILED_AFFINE_PARENT_DECISION,
            "verified": 1,
        },
    )


def verify_eager_boundary_tangent_parents(
    *,
    eager_pipeline_run_dir: str | Path,
    failed_boundary_tangent_run_dir: str | Path,
    parent_coarse_residual_run_dir: str | Path,
    failed_affine_controller_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the exact eager success and the unopened legacy v1 workflow."""

    eager_root = Path(eager_pipeline_run_dir).resolve()
    failed_root = Path(failed_boundary_tangent_run_dir).resolve()
    coarse_root = Path(parent_coarse_residual_run_dir).resolve()
    affine_root = (
        None
        if failed_affine_controller_run_dir is None
        else Path(failed_affine_controller_run_dir).resolve()
    )
    _require(eager_root.is_dir(), f"eager pipeline run does not exist: {eager_root}")
    _require(eager_root.name == EAGER_RUN_BASENAME, "wrong eager run basename")
    _require(
        failed_root.is_dir(),
        f"failed boundary-tangent run does not exist: {failed_root}",
    )
    _require(
        failed_root.name == FAILED_TANGENT_RUN_BASENAME,
        "wrong failed boundary-tangent run basename",
    )
    _require(coarse_root.is_dir(), f"coarse parent does not exist: {coarse_root}")
    if affine_root is not None:
        _require(affine_root.is_dir(), f"affine parent does not exist: {affine_root}")

    try:
        legacy = verify_and_readjudicate_boundary_tangent_schedule_parents(
            failed_boundary_tangent_run_dir=failed_root,
            parent_coarse_residual_run_dir=coarse_root,
        )
    except ArtifactCompatibilityError as exc:
        raise EagerBoundaryTangentProvenanceError(
            f"legacy boundary-tangent ancestry failed verification: {exc}"
        ) from exc

    eager = _verify_eager_parent(eager_root, coarse_root)
    failed, affine = _verify_failed_tangent_parent(
        failed_root, coarse_root, legacy, affine_root
    )
    coarse = dict(legacy.get("coarse_parent", {}))
    _require(
        coarse.get("run_dir") == str(coarse_root)
        and coarse.get("basename")
        == "20260731-140333_production-exact-k512-coarse-residual-one-image",
        "successful coarse parent binding changed",
    )

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": 1,
        "parents": {
            "successful_eager_pipeline": eager,
            "failed_boundary_tangent": failed,
            "successful_coarse_residual": {
                **coarse,
                "decision": COARSE_PARENT_DECISION,
                "verified": 1,
            },
            "failed_affine_reverse_controller": affine,
        },
        "historical_boundary_tangent_decision": FAILED_TANGENT_DECISION,
        "historical_schedule_readjudication": LEGACY_SCHEDULE_DECISION,
        "successful_eager_pipeline_decision": EAGER_DECISION,
        "readjudicated_decision": READJUDICATED_DECISION,
        "legacy_schedule_resource_projection_superseded": 1,
        "production_path_roles_unopened": 1,
        "training_path_ids_unopened": 1,
        "validation_path_ids_unopened": 1,
        "confirmation_path_ids_unopened": 1,
        "scientific_evidence_complete": 1,
        "numerically_valid": 1,
        "resource_valid": 1,
        "parent_artifacts_mutated": 0,
        "schedule_integration_authorized": 1,
        "fresh_v2_workflow_authorized": 1,
        **{field: 0 for field in NO_WORK_FIELDS},
        **{field: 0 for field in NO_AUTHORIZATION_FIELDS},
    }
    result["semantic_sha256"] = config_fingerprint(result)
    return result


def build_eager_boundary_tangent_readjudication(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the additive v2 readjudication without changing either parent."""

    _require(
        provenance.get("schema") == SCHEMA
        and int(provenance.get("passed", 0)) == 1,
        "eager boundary-tangent provenance did not pass",
    )
    _assert_semantic(provenance, "eager boundary-tangent provenance")
    _require(
        provenance.get("historical_boundary_tangent_decision")
        == FAILED_TANGENT_DECISION
        and provenance.get("historical_schedule_readjudication")
        == LEGACY_SCHEDULE_DECISION
        and provenance.get("successful_eager_pipeline_decision") == EAGER_DECISION
        and provenance.get("readjudicated_decision") == READJUDICATED_DECISION
        and int(provenance.get("production_path_roles_unopened", 0)) == 1,
        "eager boundary-tangent evidence cannot be readjudicated",
    )
    result: dict[str, Any] = {
        "schema": SCHEMA + "-readjudication",
        "schema_version": SCHEMA_VERSION,
        "evaluation_status": "evaluated",
        "passed": 1,
        "historical_boundary_tangent_decision": FAILED_TANGENT_DECISION,
        "historical_schedule_readjudication": LEGACY_SCHEDULE_DECISION,
        "successful_eager_pipeline_decision": EAGER_DECISION,
        "readjudicated_decision": READJUDICATED_DECISION,
        "historical_gates_mutated": 0,
        "parent_artifacts_mutated": 0,
        "production_path_roles_unopened": 1,
        "schedule_integration_authorized": 1,
        "fresh_v2_workflow_authorized": 1,
        **{field: 0 for field in NO_WORK_FIELDS},
        **{field: 0 for field in NO_AUTHORIZATION_FIELDS},
    }
    result["semantic_sha256"] = config_fingerprint(result)
    return result


def verify_boundary_tangent_eager_parents(
    *,
    successful_eager_pipeline_run_dir: str | Path,
    failed_boundary_tangent_run_dir: str | Path,
    parent_coarse_residual_run_dir: str | Path,
    failed_affine_controller_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compatibility spelling with the workflow name first."""

    return verify_eager_boundary_tangent_parents(
        eager_pipeline_run_dir=successful_eager_pipeline_run_dir,
        failed_boundary_tangent_run_dir=failed_boundary_tangent_run_dir,
        parent_coarse_residual_run_dir=parent_coarse_residual_run_dir,
        failed_affine_controller_run_dir=failed_affine_controller_run_dir,
    )


build_boundary_tangent_eager_readjudication = (
    build_eager_boundary_tangent_readjudication
)
verify_boundary_tangent_eager_parent_runs = verify_boundary_tangent_eager_parents
build_boundary_tangent_eager_path_plan = build_eager_boundary_tangent_path_plan
validate_boundary_tangent_eager_path_plan = (
    validate_eager_boundary_tangent_path_plan
)
boundary_tangent_eager_source_paths = eager_boundary_tangent_source_paths
boundary_tangent_eager_source_fingerprint = (
    eager_boundary_tangent_source_fingerprint
)
verify_boundary_tangent_eager_resume_compatibility = (
    verify_eager_boundary_tangent_resume_compatibility
)


__all__ = [
    "EAGER_DECISION",
    "EAGER_REGISTRY_COUNT",
    "EAGER_REGISTRY_FILE_SHA256",
    "EAGER_REGISTRY_SEMANTIC_SHA256",
    "EAGER_RUN_BASENAME",
    "EAGER_SCIENTIFIC_CONFIG_SHA256",
    "EAGER_SOURCE_FINGERPRINT",
    "EagerBoundaryTangentProvenanceError",
    "FAILED_TANGENT_DECISION",
    "FAILED_TANGENT_REGISTRY_COUNT",
    "FAILED_TANGENT_REGISTRY_FILE_SHA256",
    "FAILED_TANGENT_REGISTRY_SEMANTIC_SHA256",
    "FAILED_TANGENT_RUN_BASENAME",
    "FAILED_TANGENT_SCIENTIFIC_CONFIG_SHA256",
    "FAILED_TANGENT_SOURCE_FINGERPRINT",
    "HISTORICAL_V1_PREFLIGHT_RANGE",
    "LEGACY_SCHEDULE_DECISION",
    "NO_AUTHORIZATION_FIELDS",
    "NO_WORK_FIELDS",
    "PATH_ID_BITS",
    "PATH_ID_LIMIT",
    "PATH_PLAN_SCHEMA",
    "PATH_PLAN_VERSION",
    "PATH_ROLE_RANGES",
    "READJUDICATED_DECISION",
    "RESERVED_CONTROL_SEED",
    "ROOT_SEED",
    "SCHEMA",
    "SCHEMA_VERSION",
    "boundary_tangent_eager_source_fingerprint",
    "boundary_tangent_eager_source_paths",
    "build_boundary_tangent_eager_path_plan",
    "build_boundary_tangent_eager_readjudication",
    "build_eager_boundary_tangent_path_plan",
    "build_eager_boundary_tangent_readjudication",
    "eager_boundary_tangent_source_fingerprint",
    "eager_boundary_tangent_source_paths",
    "validate_boundary_tangent_eager_path_plan",
    "validate_eager_boundary_tangent_path_plan",
    "verify_boundary_tangent_eager_parent_runs",
    "verify_boundary_tangent_eager_parents",
    "verify_boundary_tangent_eager_resume_compatibility",
    "verify_eager_boundary_tangent_parents",
    "verify_eager_boundary_tangent_resume_compatibility",
]
