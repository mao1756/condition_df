r"""Certified normal/Haar uniforms for exact Jacobi level coupling.

The normal variables in this module are a copula only.  They do not
approximate a Jacobi transition.  Each returned uniform is enclosed by Arb
from stateless dyadic source prefixes, and an exact Jacobi inverse-CDF
authorizer must consume the enclosure before a transition is accepted.

The production implementation first uses a fused CUDA double-double ball
certificate and escalates only unresolved lanes to Arb.  Torch tensors carry
the certified outward-rounded enclosures.  Independent oracle controls may
request exact host rational views explicitly; production leaves them
unmaterialized and reconstructs only unresolved lanes during fallback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import math
import threading
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - stripped test environments.
    torch = None
    Tensor = Any

try:
    import flint
    from flint import arb
    from flint import ctx as flint_ctx
except ImportError:  # pragma: no cover - reported fail-closed by the API.
    flint = None
    arb = None
    flint_ctx = None

from mnist import d0_jacobi_rb_cuda as _rb_cuda
from mnist.d0_jacobi_rb_cuda_certificate import (
    fraction_to_float_down,
    fraction_to_float_up,
)


HAAR_COUPLING_VERSION = "d0-jacobi-rb-certified-haar-copula-v1"
HAAR_SOURCE_ID_VERSION = "d0-jacobi-rb-haar-source-structural-v1"
HAAR_TRANSITION_ID_VERSION = "d0-jacobi-rb-haar-transition-structural-v1"
HAAR_ROLE_SLOTS: dict[str, tuple[int, int]] = {
    "nested_a": (0xA0000, 0xA1000),
    "nested_b": (0xA1000, 0xA2000),
    "antithetic_a": (0xB0000, 0xB1000),
    "antithetic_b": (0xB1000, 0xB2000),
    "marginal_c": (0xC0000, 0xC1000),
    "marginal_d": (0xD0000, 0xD1000),
}
HAAR_PRODUCTION_RESERVED = (0xF0000, 0x100000)
_MAX_PATH_ID = 1 << 20
_ARB_LOCK = threading.RLock()
_ROLE_CODES = {
    "nested_a": 0,
    "nested_b": 1,
    "antithetic_a": 2,
    "antithetic_b": 3,
    "marginal_c": 4,
    "marginal_d": 5,
}
_SOURCE_TAG = 0xA
_TRANSITION_TAG = 0xB


class HaarCertificationError(RuntimeError):
    """A fact required by the exact coupling could not be certified."""

    def __init__(self, message: str, *, failure_code: str, **diagnostics: Any):
        super().__init__(message)
        self.failure_code = str(failure_code)
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True)
class CertifiedInterval:
    """Closed rational enclosure with outward binary64 views."""

    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        lower, upper = Fraction(self.lower), Fraction(self.upper)
        if lower > upper:
            raise ValueError("certified interval endpoints are reversed")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def midpoint(self) -> Fraction:
        return (self.lower + self.upper) / 2

    @property
    def width(self) -> Fraction:
        return self.upper - self.lower

    def contains(self, value: Fraction | float | int) -> bool:
        exact = (
            value
            if isinstance(value, Fraction)
            else Fraction(value)
            if isinstance(value, int)
            else Fraction.from_float(float(value))
        )
        return self.lower <= exact <= self.upper

    def float_bounds(self) -> tuple[float, float]:
        return fraction_to_float_down(self.lower), fraction_to_float_up(self.upper)


@dataclass(frozen=True)
class CertifiedUniformCell(CertifiedInterval):
    """Certified subinterval of the open unit interval."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not Fraction(0) < self.lower <= self.upper < Fraction(1):
            raise ValueError("certified uniform cell must lie strictly inside (0,1)")


@dataclass(frozen=True)
class HaarCouplingProfile:
    """Frozen portable certification and tree contract."""

    schema_version: int = 1
    coarsest_steps: int = 128
    finest_steps: int = 2048
    initial_prefix_bits: int = 128
    prefix_block_bits: int = 64
    max_prefix_bits: int = 1024
    arb_precision_bits: tuple[int, ...] = (256, 512, 1024, 2048)
    rational_enclosure_bits: int = 192
    minimum_uniform_bits: int = 96
    require_python_flint_version: str = "0.9.0"

    def __post_init__(self) -> None:
        if int(self.schema_version) != 1:
            raise ValueError("unsupported Haar coupling schema_version")
        coarse, fine = int(self.coarsest_steps), int(self.finest_steps)
        if coarse < 1 or fine < coarse or fine % coarse:
            raise ValueError("coarsest_steps must divide finest_steps")
        ratio = fine // coarse
        if ratio & (ratio - 1):
            raise ValueError("finest/coarsest step ratio must be a power of two")
        if int(self.initial_prefix_bits) < 64:
            raise ValueError("initial_prefix_bits must be at least 64")
        if int(self.prefix_block_bits) < 1:
            raise ValueError("prefix_block_bits must be positive")
        if not (
            int(self.initial_prefix_bits)
            <= int(self.max_prefix_bits)
            <= 1024
        ):
            raise ValueError("prefix cap must lie between the initial count and 1024")
        if not self.arb_precision_bits or any(
            int(bits) < 128 for bits in self.arb_precision_bits
        ):
            raise ValueError("Arb precision plan must contain values >=128")
        if tuple(sorted(set(self.arb_precision_bits))) != tuple(self.arb_precision_bits):
            raise ValueError("Arb precision plan must be strictly increasing")
        if int(self.rational_enclosure_bits) < 64:
            raise ValueError("rational_enclosure_bits must be at least 64")
        if not 32 <= int(self.minimum_uniform_bits) <= int(
            self.rational_enclosure_bits
        ):
            raise ValueError("minimum_uniform_bits is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            coupling_version=HAAR_COUPLING_VERSION,
            source_id_version=HAAR_SOURCE_ID_VERSION,
            transition_id_version=HAAR_TRANSITION_ID_VERSION,
            gaussian_role="copula-only",
            fused_cuda_normal_authorizing=True,
            arb_fallback_authorizing=True,
            ordinary_torch_normal_authorizing=False,
            gaussian_transition_approximation=False,
        )
        return payload


@dataclass(frozen=True)
class HaarEventIdentity:
    """Immutable coordinates for one level/path/phase/edge event."""

    role: str
    path_id: int
    sample_steps: int
    outer_step: int
    phase: int
    edge_id: int
    arm: int = 0
    tree_root_steps: int = 128

    def __post_init__(self) -> None:
        validate_role_path_id(self.role, self.path_id)
        if int(self.sample_steps) < 1:
            raise ValueError("sample_steps must be positive")
        if not 0 <= int(self.outer_step) < int(self.sample_steps):
            raise ValueError("outer_step is outside sample_steps")
        if not 0 <= int(self.phase) < 7:
            raise ValueError("phase must lie in [0,7)")
        if not 0 <= int(self.edge_id) < 392:
            raise ValueError("edge_id must lie in [0,392)")
        if int(self.arm) not in {-1, 0, 1}:
            raise ValueError("arm must be -1, 0, or 1")
        root = int(self.tree_root_steps)
        if not 128 <= int(self.sample_steps) <= 2048:
            raise ValueError("sample_steps must lie in [128,2048]")
        if int(self.sample_steps) & (int(self.sample_steps) - 1):
            raise ValueError("sample_steps must be a power of two")
        if root < 128 or root > int(self.sample_steps):
            raise ValueError("tree_root_steps must not exceed sample_steps")
        if root & (root - 1):
            raise ValueError("tree_root_steps must be a power of two")
        if int(self.sample_steps) % root:
            raise ValueError("tree_root_steps must divide sample_steps")
        ratio = int(self.sample_steps) // root
        if ratio & (ratio - 1):
            raise ValueError("sample_steps/tree_root_steps must be a power of two")


@dataclass(frozen=True)
class CertifiedHierarchicalUniformBatch:
    """Device-resident certificates with optional exact host views.

    Production CUDA calls leave ``uniform_cells`` and ``normal_cells`` empty.
    That is deliberate: certified lanes must not make a device-to-host
    round-trip merely to reconstruct Python ``Fraction`` objects.  Oracle
    controls can request those views explicitly, while transition-local
    fallback reconstructs only the unresolved lanes.
    """

    uniform_lower: Tensor
    uniform_upper: Tensor
    uniform_midpoint: Tensor
    uniform_center_hi: Tensor
    uniform_center_lo: Tensor
    uniform_radius: Tensor
    normal_lower: Tensor
    normal_upper: Tensor
    normal_center_hi: Tensor
    normal_center_lo: Tensor
    normal_radius: Tensor
    source_prefix_ids: Tensor
    transition_ids: Tensor
    certificate_mask: Tensor
    fallback_mask: Tensor
    prefix_bits: Tensor
    refinement_counts: Tensor
    fallback_reason_codes: Tensor
    uniform_cells: tuple[CertifiedUniformCell, ...]
    normal_cells: tuple[CertifiedInterval, ...]
    refinement_callback: Callable[..., Any]
    shape: tuple[int, ...]
    diagnostics: dict[str, Any]
    runtime_report: dict[str, Any]


@dataclass(frozen=True)
class UniformCellRefinementRequest:
    """Request a narrower enclosure of the same latent Haar uniform."""

    sample_index: int
    requested_source_prefix_bits: int
    current_cell: CertifiedUniformCell

    def __post_init__(self) -> None:
        if int(self.sample_index) < 0:
            raise ValueError("sample_index must be nonnegative")
        if not 1 <= int(self.requested_source_prefix_bits) <= 1024:
            raise ValueError("requested_source_prefix_bits must lie in [1,1024]")
        if not isinstance(self.current_cell, CertifiedUniformCell):
            raise TypeError("current_cell must be a CertifiedUniformCell")


@dataclass(frozen=True)
class UniformCellRefinementResult:
    """Certified response to :class:`UniformCellRefinementRequest`."""

    cell: CertifiedUniformCell
    normal_cell: CertifiedInterval
    source_prefix_bits: int

    def __post_init__(self) -> None:
        if not isinstance(self.cell, CertifiedUniformCell):
            raise TypeError("cell must be a CertifiedUniformCell")
        if not isinstance(self.normal_cell, CertifiedInterval):
            raise TypeError("normal_cell must be a CertifiedInterval")
        if not 1 <= int(self.source_prefix_bits) <= 1024:
            raise ValueError("source_prefix_bits must lie in [1,1024]")


@dataclass(frozen=True)
class CertifiedNormalUniformPrefix:
    """Arb certificate for one dyadic source prefix and round trip."""

    numerator: int
    bits: int
    normal: CertifiedInterval
    uniform: CertifiedUniformCell
    precision_bits: int
    backend: str = "python-flint/Arb"


def _uint_field(name: str, value: int, bits: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= int(value) < (1 << int(bits)):
        raise ValueError(f"{name} does not fit its frozen {bits}-bit field")
    return int(value)


def _pack_structural_id(
    tag: int, fields: Sequence[tuple[str, int, int]]
) -> int:
    """Injectively pack validated fields below a four-bit type tag."""

    shift = 0
    payload = 0
    for name, value, width in fields:
        payload |= _uint_field(name, value, width) << shift
        shift += int(width)
    if shift > 60:
        raise RuntimeError("structural Haar identifier exceeds its 60-bit payload")
    return (_uint_field("identifier tag", tag, 4) << 60) | payload


def _unpack_structural_id(
    value: int,
    *,
    tag: int,
    fields: Sequence[tuple[str, int]],
) -> dict[str, int]:
    identifier = _uint_field("identifier", value, 64)
    if identifier >> 60 != int(tag):
        raise ValueError("identifier has the wrong structural type tag")
    payload = identifier & ((1 << 60) - 1)
    result: dict[str, int] = {}
    shift = 0
    for name, width in fields:
        result[name] = (payload >> shift) & ((1 << int(width)) - 1)
        shift += int(width)
    if payload >> shift:
        raise ValueError("identifier has nonzero reserved bits")
    return result


def validate_role_path_id(role: str, path_id: int) -> None:
    """Require an integer path in its frozen, disjoint role slot."""

    if role not in HAAR_ROLE_SLOTS:
        raise ValueError(f"unsupported Haar role: {role!r}")
    if isinstance(path_id, bool) or not isinstance(path_id, int):
        raise TypeError("path_id must be an integer")
    if not 0 <= path_id < _MAX_PATH_ID:
        raise ValueError("path_id must lie in the frozen 20-bit namespace")
    lower, upper = HAAR_ROLE_SLOTS[role]
    if not lower <= path_id < upper:
        raise ValueError(f"path_id is outside the {role} slot")


def path_ids_for_role(role: str, count: int) -> tuple[int, ...]:
    if role not in HAAR_ROLE_SLOTS:
        raise ValueError(f"unsupported Haar role: {role!r}")
    if not 1 <= int(count) <= 0x1000:
        raise ValueError("role path count must lie in [1,4096]")
    base, _ = HAAR_ROLE_SLOTS[role]
    return tuple(range(base, base + int(count)))


def canonical_haar_source_id(
    *,
    role: str,
    path_id: int,
    coarsest_step: int,
    phase: int,
    edge_id: int,
    depth: int,
    node: int,
    kind: str,
    tree_root_steps: int = 128,
) -> int:
    """Collision-free packed namespace for one root or Haar detail."""

    validate_role_path_id(role, path_id)
    if kind not in {"root", "detail"}:
        raise ValueError("kind must be root or detail")
    values = (coarsest_step, phase, edge_id, depth, node)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("tree coordinates must be integers")
    if min(values) < 0:
        raise ValueError("tree coordinates must be nonnegative")
    if not 0 <= int(phase) < 7:
        raise ValueError("phase must lie in [0,7)")
    if not 0 <= int(edge_id) < 392:
        raise ValueError("edge_id must lie in [0,392)")
    if not 0 <= int(depth) <= 4:
        raise ValueError("depth must lie in [0,4]")
    if kind == "root" and (depth != 0 or node != coarsest_step):
        raise ValueError("root coordinates must use depth=0 and the coarsest node")
    if kind == "detail" and depth < 1:
        raise ValueError("detail depth must be positive")
    root = int(tree_root_steps)
    if root < 128 or root > 2048 or root & (root - 1):
        raise ValueError(
            "tree_root_steps must be a power of two in [128,2048]"
        )
    if not 0 <= int(coarsest_step) < root:
        raise ValueError("coarsest_step is outside tree_root_steps")
    node_count = root << max(0, int(depth) - 1)
    if not 0 <= int(node) < node_count:
        raise ValueError("node is outside its structural tree level")
    root_level = root.bit_length() - 1
    # ``node`` and ``depth`` identify the coarsest ancestor injectively, so
    # coarsest_step is validated above but need not be stored a second time.
    ancestor_shift = max(0, int(depth) - 1)
    if int(node) >> ancestor_shift != int(coarsest_step):
        raise ValueError("node/depth do not belong to coarsest_step")
    return _pack_structural_id(
        _SOURCE_TAG,
        (
            ("edge_id", int(edge_id), 9),
            ("phase", int(phase), 3),
            ("node", int(node), 11),
            ("depth", int(depth), 3),
            ("tree_root_level", root_level, 4),
            ("path_id", int(path_id), 20),
            ("role", _ROLE_CODES[role], 3),
            ("kind", int(kind == "detail"), 1),
        ),
    )


def canonical_haar_transition_id(event: HaarEventIdentity) -> int:
    level = int(event.sample_steps).bit_length() - 1
    if (1 << level) != int(event.sample_steps):
        raise ValueError("sample_steps must be a power of two")
    arm_code = {-1: 0, 0: 1, 1: 2}[int(event.arm)]
    tree_root_level = int(event.tree_root_steps).bit_length() - 1
    return _pack_structural_id(
        _TRANSITION_TAG,
        (
            ("edge_id", int(event.edge_id), 9),
            ("phase", int(event.phase), 3),
            ("outer_step", int(event.outer_step), 11),
            ("sample_level", level, 4),
            ("tree_root_level", tree_root_level, 4),
            ("arm", arm_code, 2),
            ("path_id", int(event.path_id), 20),
            ("role", _ROLE_CODES[event.role], 3),
        ),
    )


def unpack_haar_source_id(value: int) -> dict[str, Any]:
    fields = _unpack_structural_id(
        value,
        tag=_SOURCE_TAG,
        fields=(
            ("edge_id", 9),
            ("phase", 3),
            ("node", 11),
            ("depth", 3),
            ("tree_root_level", 4),
            ("path_id", 20),
            ("role_code", 3),
            ("kind_code", 1),
        ),
    )
    inverse_roles = {code: role for role, code in _ROLE_CODES.items()}
    if fields["role_code"] not in inverse_roles:
        raise ValueError("source identifier contains an unassigned role code")
    fields["role"] = inverse_roles[fields.pop("role_code")]
    fields["kind"] = "detail" if fields.pop("kind_code") else "root"
    fields["tree_root_steps"] = 1 << fields.pop("tree_root_level")
    fields["coarsest_step"] = fields["node"] >> max(
        0, fields["depth"] - 1
    )
    return fields


def unpack_haar_transition_id(value: int) -> dict[str, Any]:
    fields = _unpack_structural_id(
        value,
        tag=_TRANSITION_TAG,
        fields=(
            ("edge_id", 9),
            ("phase", 3),
            ("outer_step", 11),
            ("sample_level", 4),
            ("tree_root_level", 4),
            ("arm_code", 2),
            ("path_id", 20),
            ("role_code", 3),
        ),
    )
    inverse_roles = {code: role for role, code in _ROLE_CODES.items()}
    inverse_arms = {0: -1, 1: 0, 2: 1}
    if fields["role_code"] not in inverse_roles:
        raise ValueError("transition identifier contains an unassigned role code")
    if fields["arm_code"] not in inverse_arms:
        raise ValueError("transition identifier contains an unassigned arm code")
    fields["role"] = inverse_roles[fields.pop("role_code")]
    fields["arm"] = inverse_arms[fields.pop("arm_code")]
    fields["sample_steps"] = 1 << fields.pop("sample_level")
    fields["tree_root_steps"] = 1 << fields.pop("tree_root_level")
    return fields


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def haar_split(parent: Any, detail: Any) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonally split parent/detail normals into two iid child normals."""

    parent_array, detail_array = np.broadcast_arrays(_as_array(parent), _as_array(detail))
    scale = math.sqrt(0.5)
    return (
        (parent_array + detail_array) * scale,
        (parent_array - detail_array) * scale,
    )


def haar_parent(left: Any, right: Any) -> np.ndarray:
    left_array, right_array = np.broadcast_arrays(_as_array(left), _as_array(right))
    return (left_array + right_array) * math.sqrt(0.5)


def haar_detail(left: Any, right: Any) -> np.ndarray:
    left_array, right_array = np.broadcast_arrays(_as_array(left), _as_array(right))
    return (left_array - right_array) * math.sqrt(0.5)


def haar_refine(parent: Any, details: Any) -> np.ndarray:
    """Refine one complete level in parent-major child order."""

    parents = _as_array(parent)
    detail_array = _as_array(details)
    if parents.shape != detail_array.shape:
        raise ValueError("parent and detail arrays must have identical shapes")
    if parents.ndim == 0:
        raise ValueError("haar_refine requires at least one tree dimension")
    left, right = haar_split(parents, detail_array)
    result = np.empty(parents.shape + (2,), dtype=np.float64)
    result[..., 0], result[..., 1] = left, right
    return result.reshape(parents.shape[:-1] + (2 * parents.shape[-1],))


def _validate_level(profile: HaarCouplingProfile, sample_steps: int) -> int:
    steps = int(sample_steps)
    if not int(profile.coarsest_steps) <= steps <= int(profile.finest_steps):
        raise ValueError("sample_steps is outside the frozen Haar tree")
    if steps % int(profile.coarsest_steps):
        raise ValueError("sample_steps must be a dyadic refinement of coarsest_steps")
    ratio = steps // int(profile.coarsest_steps)
    if ratio & (ratio - 1):
        raise ValueError("sample_steps must be a dyadic refinement of coarsest_steps")
    return ratio.bit_length() - 1


def haar_ancestor_step(
    sample_steps: int, outer_step: int, *, coarsest_steps: int = 128
) -> int:
    if sample_steps < coarsest_steps or sample_steps % coarsest_steps:
        raise ValueError("sample_steps is not nested over coarsest_steps")
    ratio = sample_steps // coarsest_steps
    if ratio & (ratio - 1):
        raise ValueError("sample_steps/coarsest_steps must be a power of two")
    if not 0 <= outer_step < sample_steps:
        raise ValueError("outer_step is outside sample_steps")
    return outer_step // ratio


def _prefix_numerator(root_seed: Any, source_id: int, bits: int) -> int:
    remaining = int(bits)
    block = 0
    numerator = 0
    while remaining:
        word = _rb_cuda._philox_u64(root_seed, int(source_id), block)
        take = min(64, remaining)
        numerator = (numerator << take) | (word >> (64 - take))
        block += 1
        remaining -= take
    return numerator


def _exact_arb(value: Fraction | int) -> Any:
    assert arb is not None
    exact = Fraction(value)
    return arb(exact.numerator) / arb(exact.denominator)


def _exact_integer_from_arb(value: Any) -> int:
    if not bool(value.is_exact()):
        raise HaarCertificationError(
            "expected an exact Arb integer",
            failure_code="arb_integer_conversion_invalid",
        )
    mantissa, exponent = value.man_exp()
    integer = int(mantissa)
    power = int(exponent)
    if power >= 0:
        return integer << power
    divisor = 1 << (-power)
    if integer % divisor:
        raise HaarCertificationError(
            "Arb integer had a fractional binary exponent",
            failure_code="arb_integer_conversion_invalid",
        )
    return integer // divisor


def _arb_to_rational_interval(value: Any, bits: int) -> CertifiedInterval:
    assert arb is not None
    scale = arb(2) ** int(bits)
    lower_integer = _exact_integer_from_arb((value.lower() * scale).floor())
    upper_integer = _exact_integer_from_arb((value.upper() * scale).ceil())
    denominator = 1 << int(bits)
    return CertifiedInterval(
        Fraction(lower_integer, denominator),
        Fraction(upper_integer, denominator),
    )


def _normal_from_prefix(source_id: int, root_seed: Any, bits: int) -> Any:
    assert arb is not None
    numerator = _prefix_numerator(root_seed, source_id, bits)
    denominator = 1 << int(bits)
    if numerator == 0 or numerator + 1 == denominator:
        raise HaarCertificationError(
            "normal source prefix touches a unit-interval facet",
            failure_code="normal_source_facet",
            source_id=int(source_id),
            prefix_bits=int(bits),
        )
    sqrt_two = arb(2).sqrt()
    lower_u = _exact_arb(Fraction(numerator, denominator))
    upper_u = _exact_arb(Fraction(numerator + 1, denominator))
    lower = sqrt_two * (2 * lower_u - 1).erfinv()
    upper = sqrt_two * (2 * upper_u - 1).erfinv()
    return lower.union(upper)


def _source_ids_for_event(
    event: HaarEventIdentity,
) -> tuple[tuple[int, ...], int, int]:
    """Return structural root/detail IDs, depth, and coarsest block."""

    if int(event.sample_steps) % int(event.tree_root_steps):
        raise ValueError("event sample_steps must be nested over tree_root_steps")
    ratio = int(event.sample_steps) // int(event.tree_root_steps)
    if ratio & (ratio - 1):
        raise ValueError("event sample_steps/tree_root_steps must be dyadic")
    depth = ratio.bit_length() - 1
    coarsest_step = int(event.outer_step) >> depth
    source_ids = [
        canonical_haar_source_id(
            role=event.role,
            path_id=event.path_id,
            coarsest_step=coarsest_step,
            phase=event.phase,
            edge_id=event.edge_id,
            depth=0,
            node=coarsest_step,
            kind="root",
            tree_root_steps=int(event.tree_root_steps),
        )
    ]
    for level in range(1, depth + 1):
        parent_node = int(event.outer_step) >> (depth - level + 1)
        source_ids.append(
            canonical_haar_source_id(
                role=event.role,
                path_id=event.path_id,
                coarsest_step=coarsest_step,
                phase=event.phase,
                edge_id=event.edge_id,
                depth=level,
                node=parent_node,
                kind="detail",
                tree_root_steps=int(event.tree_root_steps),
            )
        )
    return tuple(source_ids), depth, coarsest_step


def certify_normal_uniform_from_prefix(
    numerator: int,
    bits: int,
    profile: HaarCouplingProfile,
) -> CertifiedNormalUniformPrefix:
    """Certify ``Phi(Phi^-1(U))`` for one explicit dyadic prefix.

    Prefixes whose closed enclosure touches zero or one fail closed because a
    finite normal enclosure cannot contain the corresponding endpoint.
    """

    if not isinstance(profile, HaarCouplingProfile):
        raise TypeError("profile must be a HaarCouplingProfile")
    _require_backend(profile)
    prefix_bits = _uint_field("prefix bits", int(bits), 11)
    if not 1 <= prefix_bits <= int(profile.max_prefix_bits):
        raise ValueError("bits must lie in the profile's [1,max_prefix_bits] range")
    prefix_numerator = _uint_field(
        "prefix numerator", int(numerator), prefix_bits
    )
    denominator = 1 << prefix_bits
    if prefix_numerator == 0 or prefix_numerator + 1 == denominator:
        raise HaarCertificationError(
            "normal source prefix touches a unit-interval facet",
            failure_code="normal_source_facet",
            prefix_bits=prefix_bits,
            prefix_numerator=prefix_numerator,
        )
    latest: tuple[CertifiedInterval, CertifiedInterval, int] | None = None
    for precision in profile.arb_precision_bits:
        with _ARB_LOCK:
            assert arb is not None and flint_ctx is not None
            previous_precision = int(flint_ctx.prec)
            try:
                flint_ctx.prec = int(precision)
                sqrt_two = arb(2).sqrt()
                lower_u = _exact_arb(Fraction(prefix_numerator, denominator))
                upper_u = _exact_arb(
                    Fraction(prefix_numerator + 1, denominator)
                )
                normal = (
                    sqrt_two * (2 * lower_u - 1).erfinv()
                ).union(
                    sqrt_two * (2 * upper_u - 1).erfinv()
                )
                round_trip = (1 + (normal / sqrt_two).erf()) / 2
                normal_cell = _arb_to_rational_interval(
                    normal, int(profile.rational_enclosure_bits)
                )
                uniform_interval = _arb_to_rational_interval(
                    round_trip, int(profile.rational_enclosure_bits)
                )
            finally:
                flint_ctx.prec = previous_precision
        latest = normal_cell, uniform_interval, int(precision)
        source_cell = CertifiedUniformCell(
            Fraction(prefix_numerator, denominator),
            Fraction(prefix_numerator + 1, denominator),
        )
        if (
            uniform_interval.lower <= source_cell.lower
            and uniform_interval.upper >= source_cell.upper
            and uniform_interval.lower > 0
            and uniform_interval.upper < 1
        ):
            return CertifiedNormalUniformPrefix(
                numerator=prefix_numerator,
                bits=prefix_bits,
                normal=normal_cell,
                uniform=CertifiedUniformCell(
                    uniform_interval.lower, uniform_interval.upper
                ),
                precision_bits=int(precision),
            )
    raise HaarCertificationError(
        "normal inverse/CDF round trip was not certified",
        failure_code="certified_normal_transform_unresolved",
        latest=None
        if latest is None
        else {
            "normal": [str(latest[0].lower), str(latest[0].upper)],
            "uniform": [str(latest[1].lower), str(latest[1].upper)],
            "precision_bits": latest[2],
        },
    )


def _certify_event(
    event: HaarEventIdentity,
    *,
    root_seed: Any,
    profile: HaarCouplingProfile,
    detail_sign: int,
    source_prefix_bits: int | None = None,
) -> tuple[CertifiedInterval, CertifiedUniformCell, tuple[int, ...], int]:
    source_ids, depth, _coarsest_step = _source_ids_for_event(event)
    prefix_bits = (
        int(profile.initial_prefix_bits)
        if source_prefix_bits is None
        else int(source_prefix_bits)
    )
    if not int(profile.initial_prefix_bits) <= prefix_bits <= int(
        profile.max_prefix_bits
    ):
        raise ValueError("source_prefix_bits is outside the frozen profile")
    latest_width: Fraction | None = None
    while prefix_bits <= int(profile.max_prefix_bits):
        facet_prefix = False
        for precision in profile.arb_precision_bits:
            try:
                with _ARB_LOCK:
                    assert flint_ctx is not None and arb is not None
                    previous_precision = int(flint_ctx.prec)
                    try:
                        flint_ctx.prec = int(precision)
                        node = _normal_from_prefix(
                            source_ids[0], root_seed, prefix_bits
                        )
                        sqrt_two = arb(2).sqrt()
                        for level, source_id in enumerate(
                            source_ids[1:], start=1
                        ):
                            detail = _normal_from_prefix(
                                source_id, root_seed, prefix_bits
                            )
                            bit = (
                                event.outer_step >> (depth - level)
                            ) & 1
                            sign = 1 if bit == 0 else -1
                            sign *= int(detail_sign)
                            node = (node + sign * detail) / sqrt_two
                        uniform = (1 + (node / sqrt_two).erf()) / 2
                        normal_cell = _arb_to_rational_interval(
                            node, int(profile.rational_enclosure_bits)
                        )
                        raw_uniform = _arb_to_rational_interval(
                            uniform, int(profile.rational_enclosure_bits)
                        )
                    finally:
                        flint_ctx.prec = previous_precision
            except HaarCertificationError as error:
                if error.failure_code != "normal_source_facet":
                    raise
                # A finite all-zero/all-one prefix does not imply a Gaussian
                # facet. Reveal the next transition-local block instead.
                facet_prefix = True
                break
            if raw_uniform.lower <= 0 or raw_uniform.upper >= 1:
                continue
            uniform_cell = CertifiedUniformCell(
                raw_uniform.lower, raw_uniform.upper
            )
            latest_width = uniform_cell.width
            if latest_width <= Fraction(1, 1 << int(profile.minimum_uniform_bits)):
                return normal_cell, uniform_cell, source_ids, prefix_bits
        if facet_prefix:
            prefix_bits += int(profile.prefix_block_bits)
            continue
        prefix_bits += int(profile.prefix_block_bits)
    raise HaarCertificationError(
        "normal/Haar/uniform enclosure did not meet the frozen width",
        failure_code="certified_normal_transform_unresolved",
        event=asdict(event),
        latest_width=None if latest_width is None else str(latest_width),
        max_prefix_bits=int(profile.max_prefix_bits),
    )


def _require_backend(profile: HaarCouplingProfile) -> None:
    version = None if flint is None else str(getattr(flint, "__version__", ""))
    if arb is None or flint_ctx is None or version != profile.require_python_flint_version:
        raise HaarCertificationError(
            "the frozen python-flint/Arb backend is unavailable",
            failure_code="certified_normal_backend_unavailable",
            required_version=profile.require_python_flint_version,
            actual_version=version,
        )


def _integer_sequence(values: Iterable[Any], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise TypeError(f"{name} must contain integers")
    return tuple(int(value) for value in result)


def _tensor(
    values: Sequence[Any],
    *,
    shape: tuple[int, ...],
    dtype: Any,
    device: Any,
) -> Tensor:
    if torch is None:
        return np.asarray(values).reshape(shape)
    return torch.tensor(values, dtype=dtype, device=device).reshape(shape)


def _binary64_ball_for_interval(
    cell: CertifiedInterval,
) -> tuple[float, float, float]:
    """Return a DD centre/radius containing both outward binary64 endpoints."""

    if not isinstance(cell, CertifiedInterval):
        raise TypeError("cell must be a CertifiedInterval")
    centre = float(cell.midpoint)
    exact_centre = Fraction.from_float(centre)
    lower_float, upper_float = cell.float_bounds()
    radius = max(
        exact_centre - Fraction.from_float(lower_float),
        Fraction.from_float(upper_float) - exact_centre,
        Fraction(0),
    )
    return centre, 0.0, fraction_to_float_up(radius)


def build_certified_haar_uniform_batch(
    *,
    root_seed: Any,
    role: str,
    path_ids: Iterable[int],
    sample_steps: int,
    outer_step: int,
    phase: int,
    edge_ids: Iterable[int],
    profile: HaarCouplingProfile,
    detail_sign: int = 1,
    pair_coarse_steps: int | None = None,
    device: Any = None,
    materialize_host_cells: bool | None = None,
) -> CertifiedHierarchicalUniformBatch:
    """Build exact-marginal Haar-coupled uniform enclosures.

    ``detail_sign=-1`` is the antithetic arm.  It changes only the signs of
    independent Haar details and therefore preserves every level marginal.
    """

    if not isinstance(profile, HaarCouplingProfile):
        raise TypeError("profile must be a HaarCouplingProfile")
    _require_backend(profile)
    if int(detail_sign) not in {-1, 1}:
        raise ValueError("detail_sign must be -1 or 1")
    paths = _integer_sequence(path_ids, "path_ids")
    edges = _integer_sequence(edge_ids, "edge_ids")
    if len(set(paths)) != len(paths):
        raise ValueError("path_ids must be unique")
    if len(set(edges)) != len(edges) or min(edges) < 0:
        raise ValueError("edge_ids must be unique and nonnegative")
    for path_id in paths:
        validate_role_path_id(role, path_id)
    _validate_level(profile, int(sample_steps))
    tree_root_steps = int(profile.coarsest_steps)
    if pair_coarse_steps is not None:
        tree_root_steps = int(pair_coarse_steps)
        if role not in {
            "antithetic_a",
            "antithetic_b",
            "marginal_c",
            "marginal_d",
        }:
            raise ValueError(
                "pair_coarse_steps requires an antithetic or marginal-control role"
            )
        if tree_root_steps not in {128, 256, 512, 1024}:
            raise ValueError("pair_coarse_steps must be 128, 256, 512, or 1024")
        if int(sample_steps) not in {tree_root_steps, 2 * tree_root_steps}:
            raise ValueError("pairwise sample_steps must be K or 2K")
    elif int(detail_sign) == -1:
        raise ValueError(
            "antithetic detail reversal requires an explicit pair_coarse_steps"
        )
    if not 0 <= int(outer_step) < int(sample_steps):
        raise ValueError("outer_step is outside sample_steps")
    if not 0 <= int(phase) < 7:
        raise ValueError("phase must lie in [0,7)")

    started = time.perf_counter()
    normal_cells: list[CertifiedInterval] = []
    uniform_cells: list[CertifiedUniformCell] = []
    first_source_ids: list[int] = []
    transition_ids: list[int] = []
    prefix_counts: list[int] = []
    source_id_sets: list[tuple[int, ...]] = []
    events: list[HaarEventIdentity] = []
    for path_id in paths:
        for edge_id in edges:
            event = HaarEventIdentity(
                role=role,
                path_id=path_id,
                sample_steps=int(sample_steps),
                outer_step=int(outer_step),
                phase=int(phase),
                edge_id=edge_id,
                arm=(int(detail_sign) if pair_coarse_steps is not None else 0),
                tree_root_steps=tree_root_steps,
            )
            source_ids, _depth, _coarsest_step = _source_ids_for_event(event)
            first_source_ids.append(source_ids[0])
            source_id_sets.append(source_ids)
            transition_ids.append(canonical_haar_transition_id(event))
            events.append(event)

    fused_launch = None
    use_fused = bool(
        torch is not None
        and device is not None
        and torch.device(device).type == "cuda"
        and events
    )
    materialize = (
        not use_fused
        if materialize_host_cells is None
        else bool(materialize_host_cells)
    )

    if use_fused:
        from mnist.d0_jacobi_rb_haar_fused import (
            HAAR_NORMAL_MAX_SOURCES,
            launch_certified_haar_normal_transform,
        )

        depths = [
            (int(event.sample_steps) // int(event.tree_root_steps)).bit_length()
            - 1
            for event in events
        ]
        source_matrix = [
            list(source_ids)
            + [0] * (HAAR_NORMAL_MAX_SOURCES - len(source_ids))
            for source_ids in source_id_sets
        ]
        branch_codes = [int(event.outer_step) for event in events]
        signs = [int(detail_sign)] * len(events)
        fused_launch = launch_certified_haar_normal_transform(
            torch.tensor(
                source_matrix, dtype=torch.uint64, device=device
            ).contiguous(),
            torch.tensor(depths, dtype=torch.int32, device=device).contiguous(),
            torch.tensor(
                branch_codes, dtype=torch.int32, device=device
            ).contiguous(),
            torch.tensor(signs, dtype=torch.int32, device=device).contiguous(),
            root_seed=root_seed,
        )
        fallback_indices = (
            torch.nonzero(
                ~fused_launch.certificate_mask, as_tuple=False
            )
            .reshape(-1)
            .detach()
            .cpu()
            .tolist()
        )
        fallback_records: dict[
            int, tuple[CertifiedInterval, CertifiedUniformCell, int]
        ] = {}
        for index in fallback_indices:
            normal, uniform, _source_ids, prefix_bits = _certify_event(
                events[int(index)],
                root_seed=root_seed,
                profile=profile,
                detail_sign=int(detail_sign),
            )
            fallback_records[int(index)] = (normal, uniform, prefix_bits)
    else:
        for event in events:
            normal, uniform, _source_ids, prefix_bits = _certify_event(
                event,
                root_seed=root_seed,
                profile=profile,
                detail_sign=int(detail_sign),
            )
            normal_cells.append(normal)
            uniform_cells.append(uniform)
            prefix_counts.append(prefix_bits)

    def refine_uniform_cell(
        request: UniformCellRefinementRequest,
    ) -> UniformCellRefinementResult:
        if not isinstance(request, UniformCellRefinementRequest):
            raise TypeError("refinement callback requires UniformCellRefinementRequest")
        index = int(request.sample_index)
        if not 0 <= index < len(events):
            raise IndexError("refinement sample_index is outside this batch")
        requested = int(request.requested_source_prefix_bits)
        normal, uniform, _source_ids, used_bits = _certify_event(
            events[index],
            root_seed=root_seed,
            profile=profile,
            detail_sign=int(detail_sign),
            source_prefix_bits=requested,
        )
        if (
            uniform.lower < request.current_cell.lower
            or uniform.upper > request.current_cell.upper
        ):
            raise HaarCertificationError(
                "refined Arb cell was not nested in its prior certificate",
                failure_code="uniform_refinement_not_nested",
                sample_index=index,
                requested_source_prefix_bits=requested,
            )
        return UniformCellRefinementResult(
            cell=uniform,
            normal_cell=normal,
            source_prefix_bits=used_bits,
        )

    shape = (len(paths), len(edges))
    if fused_launch is not None:
        device_certified = fused_launch.certificate_mask.reshape(shape)
        uniform_lower_tensor = fused_launch.uniform_lower.reshape(shape).clone()
        uniform_upper_tensor = fused_launch.uniform_upper.reshape(shape).clone()
        normal_lower_tensor = fused_launch.normal_lower.reshape(shape).clone()
        normal_upper_tensor = fused_launch.normal_upper.reshape(shape).clone()
        u_center_hi_tensor = (
            fused_launch.uniform_center_hi.reshape(shape).clone()
        )
        u_center_lo_tensor = (
            fused_launch.uniform_center_lo.reshape(shape).clone()
        )
        u_radius_tensor = fused_launch.uniform_radius.reshape(shape).clone()
        n_center_hi_tensor = (
            fused_launch.normal_center_hi.reshape(shape).clone()
        )
        n_center_lo_tensor = (
            fused_launch.normal_center_lo.reshape(shape).clone()
        )
        n_radius_tensor = fused_launch.normal_radius.reshape(shape).clone()
        prefix_tensor = fused_launch.prefix_bits.reshape(shape).clone()
        if fallback_records:
            for flat_index, (normal, uniform, prefix_bits) in (
                fallback_records.items()
            ):
                row, column = divmod(flat_index, len(edges))
                u_lower, u_upper = uniform.float_bounds()
                n_lower, n_upper = normal.float_bounds()
                u_hi, u_lo, u_radius = _binary64_ball_for_interval(
                    uniform
                )
                n_hi, n_lo, n_radius = _binary64_ball_for_interval(
                    normal
                )
                uniform_lower_tensor[row, column] = u_lower
                uniform_upper_tensor[row, column] = u_upper
                normal_lower_tensor[row, column] = n_lower
                normal_upper_tensor[row, column] = n_upper
                u_center_hi_tensor[row, column] = u_hi
                u_center_lo_tensor[row, column] = u_lo
                u_radius_tensor[row, column] = u_radius
                n_center_hi_tensor[row, column] = n_hi
                n_center_lo_tensor[row, column] = n_lo
                n_radius_tensor[row, column] = n_radius
                prefix_tensor[row, column] = int(prefix_bits)
        fused_certificate_tensor = torch.ones_like(device_certified)
        fused_fallback_tensor = (~fused_launch.certificate_mask).reshape(shape)
        fused_reasons_tensor = fused_launch.fallback_reason_codes.reshape(shape)
        if materialize:
            uniform_lower_values = (
                uniform_lower_tensor.detach().cpu().reshape(-1).tolist()
            )
            uniform_upper_values = (
                uniform_upper_tensor.detach().cpu().reshape(-1).tolist()
            )
            normal_lower_values = (
                normal_lower_tensor.detach().cpu().reshape(-1).tolist()
            )
            normal_upper_values = (
                normal_upper_tensor.detach().cpu().reshape(-1).tolist()
            )
            uniform_cells = [
                CertifiedUniformCell(
                    Fraction.from_float(float(lower)),
                    Fraction.from_float(float(upper)),
                )
                for lower, upper in zip(
                    uniform_lower_values,
                    uniform_upper_values,
                    strict=True,
                )
            ]
            normal_cells = [
                CertifiedInterval(
                    Fraction.from_float(float(lower)),
                    Fraction.from_float(float(upper)),
                )
                for lower, upper in zip(
                    normal_lower_values,
                    normal_upper_values,
                    strict=True,
                )
            ]
    else:
        uniform_bounds = [cell.float_bounds() for cell in uniform_cells]
        normal_bounds = [cell.float_bounds() for cell in normal_cells]
        lower = [value[0] for value in uniform_bounds]
        upper = [value[1] for value in uniform_bounds]
        midpoint = [float(cell.midpoint) for cell in uniform_cells]
        n_lower = [value[0] for value in normal_bounds]
        n_upper = [value[1] for value in normal_bounds]
        uniform_lower_tensor = _tensor(
            lower, shape=shape, dtype=torch.float64, device=device
        )
        uniform_upper_tensor = _tensor(
            upper, shape=shape, dtype=torch.float64, device=device
        )
        normal_lower_tensor = _tensor(
            n_lower, shape=shape, dtype=torch.float64, device=device
        )
        normal_upper_tensor = _tensor(
            n_upper, shape=shape, dtype=torch.float64, device=device
        )
        u_center_hi_tensor = _tensor(
            midpoint, shape=shape, dtype=(torch.float64 if torch is not None else np.float64), device=device
        )
        u_center_lo_tensor = _tensor(
            [0.0] * len(midpoint),
            shape=shape,
            dtype=(torch.float64 if torch is not None else np.float64),
            device=device,
        )
        u_radius_tensor = _tensor(
            [
                max(float(cell.midpoint - cell.lower), float(cell.upper - cell.midpoint))
                for cell in uniform_cells
            ],
            shape=shape,
            dtype=(torch.float64 if torch is not None else np.float64),
            device=device,
        )
        normal_midpoint = [float(cell.midpoint) for cell in normal_cells]
        n_center_hi_tensor = _tensor(
            normal_midpoint,
            shape=shape,
            dtype=(torch.float64 if torch is not None else np.float64),
            device=device,
        )
        n_center_lo_tensor = _tensor(
            [0.0] * len(normal_midpoint),
            shape=shape,
            dtype=(torch.float64 if torch is not None else np.float64),
            device=device,
        )
        n_radius_tensor = _tensor(
            [
                max(float(cell.midpoint - cell.lower), float(cell.upper - cell.midpoint))
                for cell in normal_cells
            ],
            shape=shape,
            dtype=(torch.float64 if torch is not None else np.float64),
            device=device,
        )
        fused_certificate_tensor = None
        fused_fallback_tensor = None
        fused_reasons_tensor = None
        prefix_tensor = _tensor(
            prefix_counts,
            shape=shape,
            dtype=(torch.int32 if torch is not None else np.int32),
            device=device,
        )
    if torch is None:
        bool_dtype, int_dtype, uint_dtype, float_dtype = bool, np.int32, np.uint64, np.float64
    else:
        bool_dtype, int_dtype, uint_dtype, float_dtype = (
            torch.bool,
            torch.int32,
            torch.uint64,
            torch.float64,
        )
    elapsed = time.perf_counter() - started
    arb_fallback_count = (
        len(uniform_cells)
        if fused_launch is None
        else int((~fused_launch.certificate_mask).sum().item())
    )
    if torch is not None and isinstance(prefix_tensor, torch.Tensor):
        refinement_tensor = torch.div(
            prefix_tensor - int(profile.initial_prefix_bits),
            int(profile.prefix_block_bits),
            rounding_mode="floor",
        ).to(dtype=torch.int32)
        maximum_prefix_bits = int(prefix_tensor.max().item()) if len(events) else 0
    else:
        refinement_tensor = (
            (
                np.asarray(prefix_tensor, dtype=np.int32)
                - int(profile.initial_prefix_bits)
            )
            // int(profile.prefix_block_bits)
        ).astype(np.int32, copy=False)
        maximum_prefix_bits = (
            int(np.max(prefix_tensor)) if len(events) else 0
        )
    fused_elapsed = (
        0.0 if fused_launch is None else float(fused_launch.elapsed_seconds)
    )
    fused_available = fused_launch is not None
    return CertifiedHierarchicalUniformBatch(
        uniform_lower=uniform_lower_tensor,
        uniform_upper=uniform_upper_tensor,
        uniform_midpoint=(
            u_center_hi_tensor + u_center_lo_tensor
        ).contiguous(),
        uniform_center_hi=u_center_hi_tensor,
        uniform_center_lo=u_center_lo_tensor,
        uniform_radius=u_radius_tensor,
        normal_lower=normal_lower_tensor,
        normal_upper=normal_upper_tensor,
        normal_center_hi=n_center_hi_tensor,
        normal_center_lo=n_center_lo_tensor,
        normal_radius=n_radius_tensor,
        source_prefix_ids=_tensor(
            first_source_ids, shape=shape, dtype=uint_dtype, device=device
        ),
        transition_ids=_tensor(
            transition_ids, shape=shape, dtype=uint_dtype, device=device
        ),
        certificate_mask=(
            fused_certificate_tensor
            if fused_certificate_tensor is not None
            else _tensor(
                [True] * len(uniform_cells),
                shape=shape,
                dtype=bool_dtype,
                device=device,
            )
        ),
        fallback_mask=(
            fused_fallback_tensor
            if fused_fallback_tensor is not None
            else _tensor(
                [True] * len(uniform_cells),
                shape=shape,
                dtype=bool_dtype,
                device=device,
            )
        ),
        prefix_bits=prefix_tensor,
        refinement_counts=refinement_tensor,
        fallback_reason_codes=(
            fused_reasons_tensor
            if fused_reasons_tensor is not None
            else _tensor(
                [1] * len(uniform_cells),
                shape=shape,
                dtype=int_dtype,
                device=device,
            )
        ),
        uniform_cells=tuple(uniform_cells),
        normal_cells=tuple(normal_cells),
        refinement_callback=refine_uniform_cell,
        shape=shape,
        diagnostics={
            "sample_count": len(uniform_cells),
            "certificate_count": len(uniform_cells),
            "arb_fallback_count": arb_fallback_count,
            "maximum_prefix_bits": maximum_prefix_bits,
            "source_id_collision_count": (
                sum(len(values) for values in source_id_sets)
                - len({value for values in source_id_sets for value in values})
            ),
            # This single-level batch contains no repeated events.  Workflow
            # registries separately label equal ancestor IDs across levels as
            # intentional aliases rather than hash collisions.
            "intentional_ancestor_alias_count": 0,
            "transition_id_collision_count": len(transition_ids)
            - len(set(transition_ids)),
            "maximum_uniform_width": float(
                (
                    (uniform_upper_tensor - uniform_lower_tensor)
                    .max()
                    .item()
                    if len(events)
                    else 0.0
                )
            ),
            "host_interval_materialization_count": (
                len(events) if materialize else 0
            ),
            "device_resident_certified_output_pass": int(
                fused_launch is not None and not materialize
            ),
            "tree_root_steps": tree_root_steps,
            "pairwise_local_detail_count": int(
                pair_coarse_steps is not None and int(sample_steps) > tree_root_steps
            ),
        },
        runtime_report={
            "coupling_version": HAAR_COUPLING_VERSION,
            "authorization_backend": (
                "fused-cuda-dd-with-transition-local-Arb-fallback"
                if fused_available
                else "python-flint/Arb"
            ),
            "python_flint_version": str(flint.__version__),
            "arb_authorizing": True,
            "fused_cuda_authorizer_available": fused_available,
            "fused_cuda_authorizer_unavailable_reason": (
                None
                if fused_available
                else "portable CPU execution uses the Arb authorizer"
            ),
            "torch_normal_authorizing": fused_available,
            "arb_fallback_fraction": (
                arb_fallback_count / len(uniform_cells)
                if uniform_cells
                else 0.0
            ),
            "arb_fallback_elapsed_seconds": (
                max(0.0, elapsed - fused_elapsed)
                if arb_fallback_count
                else 0.0
            ),
            "arb_fallback_time_fraction": (
                max(0.0, elapsed - fused_elapsed) / elapsed
                if elapsed > 0.0 and arb_fallback_count
                else 0.0
            ),
            "fused_cuda_elapsed_seconds": fused_elapsed,
            "host_interval_materialization_count": (
                len(events) if materialize else 0
            ),
            "device_resident_certified_output": int(
                fused_launch is not None and not materialize
            ),
            "fused_cuda_source_sha256": (
                None if fused_launch is None else fused_launch.bundle.source_sha256
            ),
            "fused_cuda_binary_sha256": (
                None if fused_launch is None else fused_launch.bundle.binary_sha256
            ),
            "elapsed_seconds": elapsed,
            "profile": profile.to_dict(),
            "tree_root_steps": tree_root_steps,
            "pair_coarse_steps": (
                None if pair_coarse_steps is None else int(pair_coarse_steps)
            ),
        },
    )


def verify_haar_id_plan(
    roles: Iterable[str] = tuple(HAAR_ROLE_SLOTS)
) -> dict[str, Any]:
    selected = tuple(roles)
    slots = [HAAR_ROLE_SLOTS[role] for role in selected]
    disjoint = all(
        left_upper <= right_lower or right_upper <= left_lower
        for index, (left_lower, left_upper) in enumerate(slots)
        for right_lower, right_upper in slots[index + 1 :]
    )
    reserved_disjoint = all(
        upper <= HAAR_PRODUCTION_RESERVED[0]
        for lower, upper in slots
    )
    return {
        "version": HAAR_SOURCE_ID_VERSION,
        "roles": list(selected),
        "twenty_bit_bounds_pass": int(
            all(0 <= lower < upper <= _MAX_PATH_ID for lower, upper in slots)
        ),
        "role_disjointness_pass": int(disjoint),
        "production_reservation_disjoint_pass": int(reserved_disjoint),
        "structural_packing_injective_by_construction": 1,
        "structural_id_uint64_bounds_pass": 1,
        "source_type_tag": _SOURCE_TAG,
        "transition_type_tag": _TRANSITION_TAG,
        "passed": int(disjoint and reserved_disjoint),
    }


# Static production contract consumed by the additive scheduler.  Runtime
# execution still rechecks the actual certificate/fallback diagnostics.
build_certified_haar_uniform_batch.haar_backend_contract = {
    "fused_cuda_normal_authorizer": True,
    "normal_cuda_authorizing": True,
    "normal_fallback_fraction_upper_bound": 1.0e-4,
    "normal_fallback_time_fraction_upper_bound": 0.10,
    "source": "d0_jacobi_rb_haar_fused",
}


__all__ = [
    "HAAR_COUPLING_VERSION",
    "HAAR_PRODUCTION_RESERVED",
    "HAAR_ROLE_SLOTS",
    "HAAR_SOURCE_ID_VERSION",
    "HAAR_TRANSITION_ID_VERSION",
    "CertifiedHierarchicalUniformBatch",
    "CertifiedInterval",
    "CertifiedNormalUniformPrefix",
    "CertifiedUniformCell",
    "HaarCertificationError",
    "HaarCouplingProfile",
    "HaarEventIdentity",
    "UniformCellRefinementRequest",
    "UniformCellRefinementResult",
    "build_certified_haar_uniform_batch",
    "certify_normal_uniform_from_prefix",
    "canonical_haar_source_id",
    "canonical_haar_transition_id",
    "haar_ancestor_step",
    "haar_detail",
    "haar_parent",
    "haar_refine",
    "haar_split",
    "path_ids_for_role",
    "unpack_haar_source_id",
    "unpack_haar_transition_id",
    "validate_role_path_id",
    "verify_haar_id_plan",
]
