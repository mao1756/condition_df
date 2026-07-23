"""Deterministic streamed controls for the D0 implicit-score stability gate.

This module is deliberately additive.  It does not alter the boundary-control
model, the Dirichlet operator, a physical score cache, or a sampler.  Its
responsibilities are limited to

* stateless synthetic teacher/null training batches;
* stateless paired orthogonal-Hadamard probe banks;
* byte-replayable stream provenance;
* exact analytic Stein-identity controls; and
* pure pilot-profile eligibility and selection.

The production stream contains two path clusters per optimizer step.  Each
cluster has the frozen five-bin counts ``(4, 4, 4, 4, 16)``, giving a batch of
64 with exact counts ``(8, 8, 8, 8, 32)``.  Model seeds are intentionally not
part of stream derivation: learning-rate candidates and paired model seeds see
the same states and probes for a given ``(phase, law, step)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .d0_dirichlet_score import (
    edge_ratio_channels,
    harmonic_mobility_exact,
)
from .d0_score_boundary_controls import (
    BOUNDED_TEACHER_VERSION,
    ORTHOGONAL_HADAMARD_PROBE_VERSION,
    bounded_teacher_edge_score,
    orthogonal_hadamard_edge_probes,
    sample_bounded_teacher_mixture,
)
from .eulerian_flux_mnist import DirectFluxMNISTConfig, edge_alpha_value


STREAM_SCHEMA = "experiment12-d0-score-control-stream"
STREAM_SCHEMA_VERSION = 1
STREAM_DERIVATION_VERSION = "d0-stateless-cluster-stream-v1"
STEIN_PREFLIGHT_VERSION = "d0-exact-bounded-stein-identities-v1"
PILOT_PROFILE_VERSION = "d0-streamed-stability-pilot-profile-v1"

FROZEN_CLUSTER_BIN_COUNTS = (4, 4, 4, 4, 16)
FROZEN_BATCH_BIN_COUNTS = (8, 8, 8, 8, 32)
SUPPORTED_STREAM_LAWS = ("bounded_teacher", "dirichlet_null")


__all__ = [
    "STREAM_SCHEMA",
    "STREAM_SCHEMA_VERSION",
    "STREAM_DERIVATION_VERSION",
    "STEIN_PREFLIGHT_VERSION",
    "PILOT_PROFILE_VERSION",
    "FROZEN_CLUSTER_BIN_COUNTS",
    "FROZEN_BATCH_BIN_COUNTS",
    "SUPPORTED_STREAM_LAWS",
    "StreamPlan",
    "StreamBatch",
    "ProbeBanks",
    "build_stream_plan",
    "stream_plan_record",
    "derive_stream_seed",
    "generate_stream_batch",
    "stateless_probe_banks",
    "stream_replay_record",
    "verify_stream_replay",
    "analytic_bounded_score_objective_terms",
    "bootstrap_path_mean_interval",
    "run_stein_identity_preflight",
    "evaluate_pilot_profile",
    "select_stability_profile",
]


def _canonical_fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_fingerprint(value: Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    array = tensor.numpy()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_fingerprint(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class StreamPlan:
    """Frozen derivation plan for optimizer-state and trace-probe streams."""

    root_seed: int
    grid_size: int
    horizon: float
    label: int = 3
    clusters_per_step: int = 2
    bin_counts: tuple[int, ...] = FROZEN_CLUSTER_BIN_COUNTS
    probes_per_bank: int = 4
    teacher_epsilon: float = 0.5
    schema: str = STREAM_SCHEMA
    schema_version: int = STREAM_SCHEMA_VERSION
    derivation_version: str = STREAM_DERIVATION_VERSION

    def __post_init__(self) -> None:
        if self.schema != STREAM_SCHEMA or int(self.schema_version) != STREAM_SCHEMA_VERSION:
            raise ValueError("incompatible stream-plan schema")
        if self.derivation_version != STREAM_DERIVATION_VERSION:
            raise ValueError("incompatible stream derivation")
        if int(self.grid_size) <= 0 or int(self.grid_size) % 4:
            raise ValueError("grid_size must be a positive multiple of four")
        if not _finite_number(self.horizon) or float(self.horizon) <= 0.0:
            raise ValueError("horizon must be finite and positive")
        if int(self.label) < 0:
            raise ValueError("label must be nonnegative")
        if int(self.clusters_per_step) != 2:
            raise ValueError("stream schema v1 requires exactly two clusters per step")
        if tuple(int(value) for value in self.bin_counts) != FROZEN_CLUSTER_BIN_COUNTS:
            raise ValueError(
                "stream schema v1 requires cluster bin counts (4,4,4,4,16)"
            )
        if int(self.probes_per_bank) != 4:
            raise ValueError("stream schema v1 requires four probes per bank")
        if not _finite_number(self.teacher_epsilon) or not (
            0.0 < float(self.teacher_epsilon) < 1.0
        ):
            raise ValueError("teacher_epsilon must lie strictly between zero and one")

    @property
    def anchors_per_cluster(self) -> int:
        return int(sum(int(value) for value in self.bin_counts))

    @property
    def batch_size(self) -> int:
        return int(self.clusters_per_step) * self.anchors_per_cluster

    @property
    def fingerprint(self) -> str:
        return _canonical_fingerprint(_stream_plan_payload(self))


@dataclass(frozen=True)
class StreamBatch:
    """One replayable synthetic optimizer batch."""

    states: Tensor
    tau: Tensor
    tau_fraction: Tensor
    labels: Tensor
    path_ids: np.ndarray
    strata: np.ndarray
    cluster_ids: np.ndarray
    phase: str
    law: str
    step: int
    seed: int
    plan_fingerprint: str
    fingerprint: str

    def __post_init__(self) -> None:
        rows = int(self.states.shape[0]) if self.states.ndim == 2 else -1
        if rows != 64:
            raise ValueError("stream batches must have shape (64, pixels)")
        if any(value.shape != (rows,) for value in (self.tau, self.tau_fraction, self.labels)):
            raise ValueError("stream tensor axes disagree")
        if any(
            np.asarray(value).shape != (rows,)
            for value in (self.path_ids, self.strata, self.cluster_ids)
        ):
            raise ValueError("stream metadata axes disagree")
        if self.law not in SUPPORTED_STREAM_LAWS:
            raise ValueError(f"unsupported stream law {self.law!r}")
        if not self.phase or int(self.step) < 0:
            raise ValueError("phase must be nonempty and step must be nonnegative")
        if not bool(torch.isfinite(self.states).all() and (self.states > 0.0).all()):
            raise ValueError("stream states must be finite and strictly positive")
        tolerance = 2e-12 if self.states.dtype == torch.float64 else 2e-6
        if float((self.states.sum(1) - 1.0).abs().max().detach().cpu()) > tolerance:
            raise ValueError("stream states are not simplex-valued")
        counts = tuple(
            int((np.asarray(self.strata, dtype=np.int64) == index).sum())
            for index in range(5)
        )
        if counts != FROZEN_BATCH_BIN_COUNTS:
            raise ValueError("stream batch does not have the frozen five-bin strata")

    def record(self) -> dict[str, Any]:
        return {
            "schema": STREAM_SCHEMA + "-batch",
            "schema_version": STREAM_SCHEMA_VERSION,
            "phase": self.phase,
            "law": self.law,
            "step": int(self.step),
            "seed": int(self.seed),
            "rows": int(self.states.shape[0]),
            "pixels": int(self.states.shape[1]),
            "bin_counts": [
                int((self.strata == index).sum()) for index in range(5)
            ],
            "state_sha256": _tensor_fingerprint(self.states),
            "tau_fraction_sha256": _tensor_fingerprint(self.tau_fraction),
            "path_ids_sha256": _array_fingerprint(self.path_ids),
            "strata_sha256": _array_fingerprint(self.strata),
            "cluster_ids_sha256": _array_fingerprint(self.cluster_ids),
            "plan_fingerprint": self.plan_fingerprint,
            "fingerprint": self.fingerprint,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }


@dataclass(frozen=True)
class ProbeBanks:
    """Two independent stateless four-probe Hadamard banks."""

    a: Tensor
    b: Tensor
    seeds: Mapping[str, int]
    phase: str
    law: str
    step: int
    plan_fingerprint: str
    fingerprint: str

    def __post_init__(self) -> None:
        if self.a.shape != self.b.shape or self.a.ndim != 5:
            raise ValueError("probe banks must have matching (M,B,2,H,W) shapes")
        if int(self.a.shape[0]) != 4 or int(self.a.shape[1]) != 64:
            raise ValueError("stream schema v1 requires two (4,64,2,H,W) banks")
        if set(self.seeds) != {"a", "b"} or int(self.seeds["a"]) == int(self.seeds["b"]):
            raise ValueError("probe banks require two distinct named seeds")
        if not bool(torch.all((self.a == -1) | (self.a == 1))):
            raise ValueError("probe bank a is not Rademacher")
        if not bool(torch.all((self.b == -1) | (self.b == 1))):
            raise ValueError("probe bank b is not Rademacher")

    def record(self) -> dict[str, Any]:
        return {
            "schema": STREAM_SCHEMA + "-probe-banks",
            "schema_version": STREAM_SCHEMA_VERSION,
            "probe_version": ORTHOGONAL_HADAMARD_PROBE_VERSION,
            "phase": self.phase,
            "law": self.law,
            "step": int(self.step),
            "shape": list(self.a.shape),
            "seeds": {key: int(value) for key, value in self.seeds.items()},
            "bank_a_sha256": _tensor_fingerprint(self.a),
            "bank_b_sha256": _tensor_fingerprint(self.b),
            "plan_fingerprint": self.plan_fingerprint,
            "fingerprint": self.fingerprint,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }


def _stream_plan_payload(plan: StreamPlan) -> dict[str, Any]:
    payload = asdict(plan)
    payload["bin_counts"] = list(plan.bin_counts)
    payload["batch_size"] = int(plan.batch_size)
    payload["batch_bin_counts"] = list(FROZEN_BATCH_BIN_COUNTS)
    payload["teacher_version"] = BOUNDED_TEACHER_VERSION
    payload["probe_version"] = ORTHOGONAL_HADAMARD_PROBE_VERSION
    return payload


def build_stream_plan(
    *,
    root_seed: int,
    grid_size: int,
    horizon: float,
    label: int = 3,
    clusters_per_step: int = 2,
    bin_counts: Sequence[int] = FROZEN_CLUSTER_BIN_COUNTS,
    probes_per_bank: int = 4,
    teacher_epsilon: float = 0.5,
) -> StreamPlan:
    """Build the frozen version-one stream plan."""

    return StreamPlan(
        root_seed=int(root_seed),
        grid_size=int(grid_size),
        horizon=float(horizon),
        label=int(label),
        clusters_per_step=int(clusters_per_step),
        bin_counts=tuple(int(value) for value in bin_counts),
        probes_per_bank=int(probes_per_bank),
        teacher_epsilon=float(teacher_epsilon),
    )


def stream_plan_record(plan: StreamPlan) -> dict[str, Any]:
    """Return the JSON-safe immutable plan record."""

    payload = _stream_plan_payload(plan)
    return {
        **payload,
        "fingerprint": plan.fingerprint,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def derive_stream_seed(
    plan_or_root_seed: StreamPlan | int,
    phase: str,
    law: str,
    step: int,
    namespace: str,
) -> int:
    """Derive one stateless seed without consulting global RNG state."""

    root_seed = (
        int(plan_or_root_seed.root_seed)
        if isinstance(plan_or_root_seed, StreamPlan)
        else int(plan_or_root_seed)
    )
    phase_value = str(phase)
    law_value = str(law)
    namespace_value = str(namespace)
    if not phase_value or not namespace_value:
        raise ValueError("phase and namespace must be nonempty")
    if law_value not in SUPPORTED_STREAM_LAWS:
        raise ValueError(f"unsupported stream law {law_value!r}")
    if int(step) < 0:
        raise ValueError("step must be nonnegative")
    payload = json.dumps(
        [
            STREAM_DERIVATION_VERSION,
            root_seed,
            phase_value,
            law_value,
            int(step),
            namespace_value,
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def _cluster_time_template() -> tuple[Tensor, np.ndarray]:
    fractions: list[float] = []
    strata: list[int] = []
    for bin_index, count in enumerate(FROZEN_CLUSTER_BIN_COUNTS):
        for offset in range(int(count)):
            fractions.append(
                (float(bin_index) + (float(offset) + 0.5) / float(count)) / 5.0
            )
            strata.append(int(bin_index))
    return torch.tensor(fractions, dtype=torch.float64), np.asarray(strata, dtype=np.int64)


def _sample_dirichlet_null(
    rows: int, pixels: int, *, seed: int, dtype: torch.dtype
) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    concentration = torch.ones((int(rows), int(pixels)), dtype=dtype)
    draws = torch._standard_gamma(concentration, generator=generator)
    return draws / draws.sum(dim=1, keepdim=True)


def _batch_fingerprint_payload(
    *,
    states: Tensor,
    fractions: Tensor,
    path_ids: np.ndarray,
    strata: np.ndarray,
    cluster_ids: np.ndarray,
    plan: StreamPlan,
    phase: str,
    law: str,
    step: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema": STREAM_SCHEMA + "-batch",
        "schema_version": STREAM_SCHEMA_VERSION,
        "phase": str(phase),
        "law": str(law),
        "step": int(step),
        "seed": int(seed),
        "state_sha256": _tensor_fingerprint(states),
        "tau_fraction_sha256": _tensor_fingerprint(fractions),
        "path_ids_sha256": _array_fingerprint(path_ids),
        "strata_sha256": _array_fingerprint(strata),
        "cluster_ids_sha256": _array_fingerprint(cluster_ids),
        "plan_fingerprint": plan.fingerprint,
    }


def generate_stream_batch(
    plan: StreamPlan,
    *,
    phase: str,
    law: str,
    step: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> StreamBatch:
    """Generate one fresh, stateless, exactly stratified training batch.

    Samples are generated on CPU and then transferred.  Consequently the
    scientific byte stream is independent of the selected training device.
    Only float32 and float64 are accepted because both are supported by the
    exact gamma sampler and the artifact fingerprint contract.
    """

    if dtype not in (torch.float32, torch.float64):
        raise ValueError("stream dtype must be float32 or float64")
    phase_value, law_value, step_value = str(phase), str(law), int(step)
    seed = derive_stream_seed(plan, phase_value, law_value, step_value, "states")
    per_cluster_fractions, per_cluster_strata = _cluster_time_template()
    fractions_cpu = per_cluster_fractions.repeat(int(plan.clusters_per_step)).to(dtype)
    strata = np.tile(per_cluster_strata, int(plan.clusters_per_step))
    cluster_values = np.arange(
        int(plan.clusters_per_step) * step_value,
        int(plan.clusters_per_step) * (step_value + 1),
        dtype=np.int64,
    )
    cluster_ids = np.repeat(cluster_values, int(plan.anchors_per_cluster))
    path_ids = cluster_ids.copy()
    if law_value == "bounded_teacher":
        states_cpu = sample_bounded_teacher_mixture(
            fractions_cpu,
            int(plan.grid_size),
            seed=int(seed),
            device="cpu",
            dtype=dtype,
            epsilon=float(plan.teacher_epsilon),
        )
    elif law_value == "dirichlet_null":
        states_cpu = _sample_dirichlet_null(
            int(plan.batch_size),
            int(plan.grid_size) ** 2,
            seed=int(seed),
            dtype=dtype,
        )
    else:
        raise ValueError(f"unsupported stream law {law_value!r}")
    payload = _batch_fingerprint_payload(
        states=states_cpu,
        fractions=fractions_cpu,
        path_ids=path_ids,
        strata=strata,
        cluster_ids=cluster_ids,
        plan=plan,
        phase=phase_value,
        law=law_value,
        step=step_value,
        seed=seed,
    )
    target = torch.device(device)
    states = states_cpu.to(target)
    fractions = fractions_cpu.to(target)
    return StreamBatch(
        states=states,
        tau=(fractions * float(plan.horizon)).contiguous(),
        tau_fraction=fractions.contiguous(),
        labels=torch.full(
            (int(plan.batch_size),), int(plan.label), device=target, dtype=torch.long
        ),
        path_ids=path_ids,
        strata=strata,
        cluster_ids=cluster_ids,
        phase=phase_value,
        law=law_value,
        step=step_value,
        seed=seed,
        plan_fingerprint=plan.fingerprint,
        fingerprint=_canonical_fingerprint(payload),
    )


def stateless_probe_banks(
    plan: StreamPlan,
    *,
    phase: str,
    law: str,
    step: int,
    batch_size: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ProbeBanks:
    """Generate two independent four-probe banks for one stream step."""

    rows = int(plan.batch_size if batch_size is None else batch_size)
    if rows != int(plan.batch_size):
        raise ValueError("probe batch size must equal the frozen stream batch size")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("probe dtype must be float32 or float64")
    seeds = {
        name: derive_stream_seed(plan, phase, law, int(step), f"probes-{name}")
        for name in ("a", "b")
    }

    def make(seed: int) -> Tensor:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        value = orthogonal_hadamard_edge_probes(
            int(plan.probes_per_bank),
            rows,
            int(plan.grid_size),
            device="cpu",
            dtype=dtype,
            generator=generator,
        )
        return value.to(torch.device(device))

    bank_a, bank_b = make(seeds["a"]), make(seeds["b"])
    payload = {
        "schema": STREAM_SCHEMA + "-probe-banks",
        "schema_version": STREAM_SCHEMA_VERSION,
        "phase": str(phase),
        "law": str(law),
        "step": int(step),
        "seeds": seeds,
        "bank_a_sha256": _tensor_fingerprint(bank_a),
        "bank_b_sha256": _tensor_fingerprint(bank_b),
        "plan_fingerprint": plan.fingerprint,
    }
    return ProbeBanks(
        a=bank_a,
        b=bank_b,
        seeds=seeds,
        phase=str(phase),
        law=str(law),
        step=int(step),
        plan_fingerprint=plan.fingerprint,
        fingerprint=_canonical_fingerprint(payload),
    )


def stream_replay_record(
    plan: StreamPlan,
    *,
    phase: str,
    law: str,
    step: int,
) -> dict[str, Any]:
    """Materialize one canonical CPU-float32 replay certificate."""

    batch = generate_stream_batch(
        plan, phase=phase, law=law, step=int(step), device="cpu", dtype=torch.float32
    )
    probes = stateless_probe_banks(
        plan, phase=phase, law=law, step=int(step), device="cpu", dtype=torch.float32
    )
    record = {
        "schema": STREAM_SCHEMA + "-replay",
        "schema_version": STREAM_SCHEMA_VERSION,
        "derivation_version": STREAM_DERIVATION_VERSION,
        "plan_fingerprint": plan.fingerprint,
        "phase": str(phase),
        "law": str(law),
        "step": int(step),
        "batch": batch.record(),
        "probe_banks": probes.record(),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    record["fingerprint"] = _canonical_fingerprint(record)
    return record


def verify_stream_replay(plan: StreamPlan, record: Mapping[str, Any]) -> dict[str, Any]:
    """Regenerate a replay record and return a fail-closed comparison."""

    try:
        expected = stream_replay_record(
            plan,
            phase=str(record["phase"]),
            law=str(record["law"]),
            step=int(record["step"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "passed": 0,
            "reason": f"invalid replay record: {type(exc).__name__}: {exc}",
            "plan_fingerprint": plan.fingerprint,
            "physical_training_performed": 0,
            "sampling_performed": 0,
        }
    actual = dict(record)
    passed = actual == expected
    return {
        "passed": int(passed),
        "reason": None if passed else "stream replay record differs from regeneration",
        "expected_fingerprint": expected["fingerprint"],
        "actual_fingerprint": actual.get("fingerprint"),
        "plan_fingerprint": plan.fingerprint,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def analytic_bounded_score_objective_terms(
    states: Tensor,
    reverse_fraction: Tensor | float,
    config: DirectFluxMNISTConfig,
    *,
    scale: float = 1.0,
    epsilon: float = 0.5,
) -> dict[str, Tensor]:
    """Evaluate exact normalized objective terms for ``scale * log(h)``.

    ``h`` is the bounded four-anchor teacher ratio.  Its Hessian is rank one,
    so the exact edge Hessian is ``-(D log h)^2``.  This avoids constructing a
    dense ``N x N`` Hessian at production resolution while matching
    :func:`mnist.d0_dirichlet_score.dirichlet_score_objective` exactly.
    """

    n = int(config.grid_size)
    if states.ndim != 2 or states.shape[1] != n * n:
        raise ValueError(f"states must have shape (B, {n * n})")
    value = float(scale)
    if not math.isfinite(value):
        raise ValueError("scale must be finite")
    base_edge = bounded_teacher_edge_score(
        states, reverse_fraction, epsilon=float(epsilon)
    )
    theta = harmonic_mobility_exact(states, config)
    ratio = edge_ratio_channels(states, n)
    edge_score = value * base_edge
    energy = (theta * edge_score.square()).flatten(1).mean(dim=1)
    trace = (
        theta * (-value * base_edge.square())
    ).flatten(1).sum(dim=1) / float(n * n)
    drift = (
        float(2.0 * float(edge_alpha_value(config)) + 1.0)
        * ratio
        * edge_score
    ).flatten(1).sum(dim=1) / float(n * n)
    generator = trace + drift
    return {
        "objective": energy + generator,
        "energy": energy,
        "generator": generator,
        "trace": trace,
        "drift": drift,
        "edge_score": edge_score,
        "base_edge_score": base_edge,
    }


def bootstrap_path_mean_interval(
    values: np.ndarray | Tensor,
    path_ids: np.ndarray | Tensor,
    *,
    reps: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Return a deterministic two-sided whole-path bootstrap interval."""

    if isinstance(values, Tensor):
        sample_values = values.detach().double().cpu().numpy()
    else:
        sample_values = np.asarray(values, dtype=np.float64)
    if isinstance(path_ids, Tensor):
        paths = path_ids.detach().long().cpu().numpy()
    else:
        paths = np.asarray(path_ids, dtype=np.int64)
    sample_values = sample_values.reshape(-1)
    paths = paths.reshape(-1)
    if sample_values.shape != paths.shape or sample_values.size == 0:
        raise ValueError("values and path_ids must be nonempty matching vectors")
    if int(reps) <= 0 or not (0.0 < float(confidence) < 1.0):
        raise ValueError("reps must be positive and confidence must lie in (0,1)")
    unique = np.unique(paths)
    path_means = np.asarray(
        [sample_values[paths == path].mean() for path in unique], dtype=np.float64
    )
    finite = bool(np.isfinite(path_means).all())
    if not finite:
        return {
            "finite": 0,
            "path_count": int(unique.size),
            "state_count": int(sample_values.size),
            "point_estimate": None,
            "lower_bound": None,
            "upper_bound": None,
            "confidence": float(confidence),
            "reps": int(reps),
        }
    generator = np.random.default_rng(int(seed))
    totals = np.empty(int(reps), dtype=np.float64)
    chunk = 1024
    for start in range(0, int(reps), chunk):
        count = min(chunk, int(reps) - start)
        indices = generator.integers(
            0, int(unique.size), size=(count, int(unique.size))
        )
        totals[start : start + count] = path_means[indices].mean(axis=1)
    tail = 0.5 * (1.0 - float(confidence))
    return {
        "finite": 1,
        "path_count": int(unique.size),
        "state_count": int(sample_values.size),
        "point_estimate": float(path_means.mean()),
        "lower_bound": float(np.quantile(totals, tail)),
        "upper_bound": float(np.quantile(totals, 1.0 - tail)),
        "confidence": float(confidence),
        "reps": int(reps),
        "path_ids": unique.tolist(),
        "path_values": path_means.tolist(),
    }


def _stein_panel(
    *,
    plan: StreamPlan,
    law: str,
    path_count: int,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, np.ndarray, np.ndarray]:
    per_path_fraction, per_path_strata = _cluster_time_template()
    fractions = per_path_fraction.repeat(int(path_count)).to(dtype)
    strata = np.tile(per_path_strata, int(path_count))
    path_ids = np.repeat(
        np.arange(int(path_count), dtype=np.int64), int(plan.anchors_per_cluster)
    )
    seed = derive_stream_seed(plan, "stein-preflight", law, 0, "states")
    if law == "bounded_teacher":
        states = sample_bounded_teacher_mixture(
            fractions,
            int(plan.grid_size),
            seed=int(seed),
            device="cpu",
            dtype=dtype,
            epsilon=float(plan.teacher_epsilon),
        )
    elif law == "dirichlet_null":
        states = _sample_dirichlet_null(
            int(fractions.numel()),
            int(plan.grid_size) ** 2,
            seed=int(seed),
            dtype=dtype,
        )
    else:  # pragma: no cover - caller validates laws.
        raise ValueError(law)
    return states, fractions, path_ids, strata


def _identity_record(
    measured: Tensor,
    predicted: Tensor,
    path_ids: np.ndarray,
    *,
    name: str,
    confidence: float,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    residual = measured - predicted
    interval = bootstrap_path_mean_interval(
        residual,
        path_ids,
        reps=int(reps),
        confidence=float(confidence),
        seed=int(seed),
    )
    finite = bool(
        torch.isfinite(measured).all()
        and torch.isfinite(predicted).all()
        and int(interval["finite"]) == 1
    )
    contains_zero = bool(
        finite
        and float(interval["lower_bound"]) <= 0.0
        and float(interval["upper_bound"]) >= 0.0
    )
    return {
        "name": str(name),
        "finite": int(finite),
        "measured_mean": (
            float(measured.double().mean().detach().cpu()) if finite else None
        ),
        "predicted_mean": (
            float(predicted.double().mean().detach().cpu()) if finite else None
        ),
        "measured_minus_predicted": interval,
        "zero_in_99pct_whole_path_interval": int(contains_zero),
        "passed": int(finite and contains_zero),
    }


def run_stein_identity_preflight(
    dynamics: DirectFluxMNISTConfig,
    *,
    root_seed: int = 260801,
    path_count: int = 128,
    anchors_per_path: int = 32,
    bootstrap_reps: int = 10_000,
    confidence: float = 0.99,
    teacher_scales: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0),
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Check the exact null and bounded-teacher population identities.

    Panels are sampled on CPU in float64, evaluated with exact analytic
    derivatives, and only then optionally moved to ``device``.  No Hutchinson
    probes or neural checkpoints enter this preflight.
    """

    if int(path_count) <= 1:
        raise ValueError("path_count must exceed one")
    if int(anchors_per_path) != sum(FROZEN_CLUSTER_BIN_COUNTS):
        raise ValueError("Stein preflight requires 32 anchors per path")
    if int(bootstrap_reps) <= 0 or not (0.0 < float(confidence) < 1.0):
        raise ValueError("invalid bootstrap configuration")
    scales = tuple(float(value) for value in teacher_scales)
    if scales != (0.0, 0.25, 0.5, 1.0, 2.0):
        raise ValueError("Stein preflight scales are frozen at 0,0.25,0.5,1,2")
    # The plan horizon affects tau only, whereas these analytic controls use
    # tau/T.  Set it to one to make this fact explicit in the provenance.
    plan = build_stream_plan(
        root_seed=int(root_seed), grid_size=int(dynamics.grid_size), horizon=1.0
    )
    target = torch.device(device)
    null_cpu, fractions_cpu, null_paths, _ = _stein_panel(
        plan=plan, law="dirichlet_null", path_count=int(path_count), dtype=torch.float64
    )
    teacher_cpu, teacher_fractions_cpu, teacher_paths, _ = _stein_panel(
        plan=plan, law="bounded_teacher", path_count=int(path_count), dtype=torch.float64
    )
    null_states = null_cpu.to(target)
    null_fractions = fractions_cpu.to(target)
    teacher_states = teacher_cpu.to(target)
    teacher_fractions = teacher_fractions_cpu.to(target)

    null_terms = analytic_bounded_score_objective_terms(
        null_states,
        null_fractions,
        dynamics,
        scale=1.0,
        epsilon=float(plan.teacher_epsilon),
    )
    null_record = _identity_record(
        null_terms["objective"],
        null_terms["energy"],
        null_paths,
        name="dirichlet_null",
        confidence=float(confidence),
        reps=int(bootstrap_reps),
        seed=derive_stream_seed(plan, "stein-preflight", "dirichlet_null", 0, "bootstrap"),
    )

    base_teacher_terms = analytic_bounded_score_objective_terms(
        teacher_states,
        teacher_fractions,
        dynamics,
        scale=1.0,
        epsilon=float(plan.teacher_epsilon),
    )
    base_energy = base_teacher_terms["energy"]
    teacher_records: list[dict[str, Any]] = []
    for index, scale in enumerate(scales):
        terms = analytic_bounded_score_objective_terms(
            teacher_states,
            teacher_fractions,
            dynamics,
            scale=float(scale),
            epsilon=float(plan.teacher_epsilon),
        )
        predicted = float(scale * scale - 2.0 * scale) * base_energy
        record = _identity_record(
            terms["objective"],
            predicted,
            teacher_paths,
            name=f"bounded_teacher_scale_{scale:g}",
            confidence=float(confidence),
            reps=int(bootstrap_reps),
            seed=derive_stream_seed(
                plan, "stein-preflight", "bounded_teacher", 0, f"bootstrap-{index}"
            ),
        )
        record["scale"] = float(scale)
        record["coefficient"] = float(scale * scale - 2.0 * scale)
        teacher_records.append(record)

    all_records = [null_record, *teacher_records]
    finite = all(int(record["finite"]) == 1 for record in all_records)
    passed = finite and all(int(record["passed"]) == 1 for record in all_records)
    return {
        "schema": STREAM_SCHEMA + "-stein-preflight",
        "schema_version": STREAM_SCHEMA_VERSION,
        "preflight_version": STEIN_PREFLIGHT_VERSION,
        "passed": int(passed),
        "finite": int(finite),
        "claim": "exact bounded analytic Stein identities on fresh synthetic panels",
        "path_count_per_law": int(path_count),
        "anchors_per_path": int(anchors_per_path),
        "states_per_law": int(path_count) * int(anchors_per_path),
        "confidence": float(confidence),
        "bootstrap_reps": int(bootstrap_reps),
        "teacher_scales": list(scales),
        "null_identity": null_record,
        "teacher_identities": teacher_records,
        "stream_plan": stream_plan_record(plan),
        "panel_fingerprints": {
            "null_states": _tensor_fingerprint(null_cpu),
            "teacher_states": _tensor_fingerprint(teacher_cpu),
            "tau_fraction": _tensor_fingerprint(fractions_cpu),
        },
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }


def _nested_mappings(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    names = (
        "metrics",
        "training_summary",
        "selection_metrics",
        "checkpoint_selection",
        "selected_checkpoint",
        "selected_record",
    )
    result: list[Mapping[str, Any]] = []
    pending: list[Mapping[str, Any]] = [value]
    seen: set[int] = set()
    while pending:
        mapping = pending.pop(0)
        if id(mapping) in seen:
            continue
        seen.add(id(mapping))
        result.append(mapping)
        for name in names:
            nested = mapping.get(name)
            if isinstance(nested, Mapping):
                pending.append(nested)
    return result


def _lookup(value: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for mapping in _nested_mappings(value):
        for name in names:
            if name in mapping:
                return mapping[name]
    return default


def _objective_banks(value: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate = _lookup(
        value,
        "selection_objective_banks",
        "objective_banks",
        "banks",
        default={},
    )
    return candidate if isinstance(candidate, Mapping) else {}


def _scope_metric(
    value: Mapping[str, Any], scope: str, *names: str, default: Any = None
) -> Any:
    for mapping in _nested_mappings(value):
        candidate = mapping.get(scope)
        if not isinstance(candidate, Mapping):
            continue
        for name in names:
            if name in candidate:
                return candidate[name]
    return default


def _bank_scope_value(
    banks: Mapping[str, Any], bank: str, scope: str, key: str
) -> float | None:
    try:
        value = dict(dict(banks[bank])[scope])[key]
        return float(value) if _finite_number(value) else None
    except (KeyError, TypeError, ValueError):
        return None


def _pilot_common_checks(
    result: Mapping[str, Any], *, max_clip_fraction: float
) -> tuple[list[dict[str, Any]], float]:
    clip = _lookup(
        result,
        "clip_fraction_steps_101_1000",
        "post_warmup_clip_fraction",
        default=None,
    )
    final_clip = _lookup(
        result,
        "final_200_clip_fraction",
        "tail_clip_fraction",
        default=None,
    )
    checks = [
        {
            "name": "complete",
            "value": _lookup(result, "complete", default=0),
            "operator": "==",
            "threshold": 1,
            "passed": int(bool(int(_lookup(result, "complete", default=0) or 0))),
        },
        {
            "name": "finite",
            "value": _lookup(result, "finite", default=0),
            "operator": "==",
            "threshold": 1,
            "passed": int(bool(int(_lookup(result, "finite", default=0) or 0))),
        },
        {
            "name": "boundary_admissible",
            "value": _lookup(result, "boundary_admissible", default=0),
            "operator": "==",
            "threshold": 1,
            "passed": int(
                bool(int(_lookup(result, "boundary_admissible", default=0) or 0))
            ),
        },
        {
            "name": "clip_fraction_steps_101_1000",
            "value": clip,
            "operator": "<=",
            "threshold": float(max_clip_fraction),
            "passed": int(
                _finite_number(clip) and float(clip) <= float(max_clip_fraction)
            ),
        },
        {
            "name": "final_200_clip_fraction",
            "value": final_clip,
            "operator": "<=",
            "threshold": float(max_clip_fraction),
            "passed": int(
                _finite_number(final_clip)
                and float(final_clip) <= float(max_clip_fraction)
            ),
        },
    ]
    finite_clips = [
        float(value) for value in (clip, final_clip) if _finite_number(value)
    ]
    return checks, max(finite_clips, default=math.inf)


def _append_numeric_check(
    checks: list[dict[str, Any]],
    name: str,
    value: Any,
    operator: str,
    threshold: float,
) -> None:
    finite = _finite_number(value)
    if operator == ">":
        passed = finite and float(value) > float(threshold)
    elif operator == "<":
        passed = finite and float(value) < float(threshold)
    elif operator == "==":
        passed = finite and float(value) == float(threshold)
    elif operator == "<=":
        passed = finite and float(value) <= float(threshold)
    else:  # pragma: no cover - private callers use frozen operators.
        raise ValueError(operator)
    checks.append(
        {
            "name": name,
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "passed": int(passed),
        }
    )


def evaluate_pilot_profile(
    teacher_result: Mapping[str, Any],
    null_result: Mapping[str, Any],
    *,
    learning_rate: float,
    max_clip_fraction: float = 0.10,
) -> dict[str, Any]:
    """Evaluate one coupled teacher/null pilot profile.

    The helper accepts either flat task summaries or the standard result shape
    with ``metrics`` and ``training_summary`` mappings.  Objective banks must
    expose ``a``/``b`` and ``overall``/``data_end`` records.
    """

    rate = float(learning_rate)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(float(max_clip_fraction)) or not (
        0.0 <= float(max_clip_fraction) <= 1.0
    ):
        raise ValueError("max_clip_fraction must lie in [0,1]")

    teacher_checks, teacher_clip = _pilot_common_checks(
        teacher_result, max_clip_fraction=float(max_clip_fraction)
    )
    null_checks, null_clip = _pilot_common_checks(
        null_result, max_clip_fraction=float(max_clip_fraction)
    )
    _append_numeric_check(
        teacher_checks,
        "selected_nonzero_checkpoint",
        _lookup(teacher_result, "selected_step", default=None),
        ">",
        0.0,
    )
    _append_numeric_check(
        null_checks,
        "selected_analytic_zero",
        _lookup(null_result, "selected_step", default=None),
        "==",
        0.0,
    )

    teacher_banks = _objective_banks(teacher_result)
    null_banks = _objective_banks(null_result)
    teacher_risks: list[float] = []
    for bank in ("a", "b"):
        for scope in ("overall", "data_end"):
            teacher_lcb = _bank_scope_value(
                teacher_banks, bank, scope, "lower_bound"
            )
            null_lcb = _bank_scope_value(null_banks, bank, scope, "lower_bound")
            _append_numeric_check(
                teacher_checks,
                f"{bank}_{scope}_objective_lower_bound",
                teacher_lcb,
                ">",
                0.0,
            )
            _append_numeric_check(
                null_checks,
                f"{bank}_{scope}_objective_lower_bound",
                null_lcb,
                "<=",
                0.0,
            )
            # Preserve the established dual-bank checkpoint convention: the
            # data-end record gates eligibility, while ranking averages the
            # two banks' overall risks.
            if scope == "overall":
                risk = _bank_scope_value(
                    teacher_banks, bank, scope, "model_score_risk"
                )
                if risk is not None:
                    teacher_risks.append(risk)

    teacher_metric_specs = (
        (
            "overall_score_gain",
            _lookup(
                teacher_result,
                "selection_overall_score_gain",
                "audit_overall_score_gain",
                "overall_score_gain",
            ),
            ">",
            0.0,
        ),
        (
            "data_end_score_gain",
            _lookup(
                teacher_result,
                "selection_data_end_score_gain",
                "audit_data_end_score_gain",
                "data_end_score_gain",
            ),
            ">",
            0.0,
        ),
        (
            "overall_flux_cosine",
            _lookup(teacher_result, "overall_flux_cosine"),
            ">",
            0.0,
        ),
        (
            "data_end_flux_cosine",
            _lookup(
                teacher_result,
                "data_end_flux_cosine",
                default=_scope_metric(
                    teacher_result, "data_end", "flux_cosine"
                ),
            ),
            ">",
            0.0,
        ),
        (
            "overall_relative_flux_l2",
            _lookup(teacher_result, "overall_relative_flux_l2"),
            "<",
            1.0,
        ),
        (
            "data_end_relative_flux_l2",
            _lookup(
                teacher_result,
                "data_end_relative_flux_l2",
                default=_scope_metric(
                    teacher_result,
                    "data_end",
                    "flux_relative_l2",
                    "relative_flux_l2",
                ),
            ),
            "<",
            1.0,
        ),
    )
    for name, value, operator, threshold in teacher_metric_specs:
        _append_numeric_check(
            teacher_checks, name, value, operator, float(threshold)
        )

    teacher_pass = all(bool(int(check["passed"])) for check in teacher_checks)
    null_pass = all(bool(int(check["passed"])) for check in null_checks)
    mean_risk = (
        float(np.mean(teacher_risks)) if len(teacher_risks) == 2 else math.inf
    )
    if not math.isfinite(mean_risk):
        teacher_pass = False
    clipping_fraction = max(teacher_clip, null_clip)
    eligible = teacher_pass and null_pass and math.isfinite(clipping_fraction)
    result = {
        "schema": STREAM_SCHEMA + "-pilot-profile",
        "schema_version": STREAM_SCHEMA_VERSION,
        "profile_version": PILOT_PROFILE_VERSION,
        "learning_rate": rate,
        "eligible": int(eligible),
        "teacher_pass": int(teacher_pass),
        "null_pass": int(null_pass),
        "mean_dual_bank_teacher_selection_risk": (
            mean_risk if math.isfinite(mean_risk) else None
        ),
        "clipping_fraction": (
            clipping_fraction if math.isfinite(clipping_fraction) else None
        ),
        "teacher_checks": teacher_checks,
        "null_checks": null_checks,
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    result["fingerprint"] = _canonical_fingerprint(result)
    return result


def select_stability_profile(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the lowest-risk eligible pilot, using the frozen tie breaks."""

    normalized = [dict(candidate) for candidate in candidates]
    rates = [float(candidate.get("learning_rate", math.nan)) for candidate in normalized]
    if not normalized or not all(math.isfinite(rate) and rate > 0.0 for rate in rates):
        raise ValueError("pilot candidates require positive learning rates")
    if len(set(rates)) != len(rates):
        raise ValueError("pilot candidate learning rates must be distinct")
    eligible = [
        candidate
        for candidate in normalized
        if int(candidate.get("eligible", 0)) == 1
        and _finite_number(candidate.get("mean_dual_bank_teacher_selection_risk"))
        and _finite_number(candidate.get("clipping_fraction"))
    ]
    eligible.sort(
        key=lambda candidate: (
            float(candidate["mean_dual_bank_teacher_selection_risk"]),
            float(candidate["clipping_fraction"]),
            float(candidate["learning_rate"]),
        )
    )
    selected = dict(eligible[0]) if eligible else None
    result = {
        "schema": STREAM_SCHEMA + "-selected-profile",
        "schema_version": STREAM_SCHEMA_VERSION,
        "profile_version": PILOT_PROFILE_VERSION,
        "passed": int(selected is not None),
        "candidate_count": len(normalized),
        "eligible_count": len(eligible),
        "ranking": [
            {
                "learning_rate": float(candidate["learning_rate"]),
                "mean_dual_bank_teacher_selection_risk": float(
                    candidate["mean_dual_bank_teacher_selection_risk"]
                ),
                "clipping_fraction": float(candidate["clipping_fraction"]),
            }
            for candidate in eligible
        ],
        "selected": selected,
        "reason": (
            None
            if selected is not None
            else "no pilot profile satisfied coupled teacher/null eligibility"
        ),
        "physical_training_performed": 0,
        "sampling_performed": 0,
    }
    result["fingerprint"] = _canonical_fingerprint(result)
    return result
