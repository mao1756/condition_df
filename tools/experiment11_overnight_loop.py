from __future__ import annotations

"""Overnight Experiment 11 run/diagnose helper.

This script is intentionally lightweight.  It launches a small suite of
Experiment 11 commands, diagnoses each completed run from saved JSON/CSV files,
and writes a Markdown/JSONL report that Codex can use for an overnight
experiment -> diagnose -> adjust loop.

Examples
--------
Diagnose existing runs only::

    python tools/experiment11_overnight_loop.py diagnose --runs-root runs/experiment11

Run the default overnight C2.1 debug suite::

    python tools/experiment11_overnight_loop.py run-suite --data-root mnist_data

The script never deletes runs.  Each command still writes into the normal
``runs/experiment11/<timestamp>_<run-name>/`` directory via Experiment 11 itself.
"""

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


@dataclass
class RunDiagnosis:
    run_dir: str
    run_id: str
    run_name: str | None
    sample_entropy: float | None
    sample_total_variation: float | None
    clipping_fraction: float | None
    learned_step_rms: float | None
    free_step_rms: float | None
    noise_step_rms: float | None
    learned_to_noise_ratio: float | None
    learned_to_free_ratio: float | None
    innovation_gain_last: float | None
    innovation_gain_tail_mean: float | None
    prediction_rms_last: float | None
    target_rms_last: float | None
    branch_centered_target_rms_last: float | None
    branch_signal_to_noise_ratio_last: float | None
    branch_ess_fraction_mean_last: float | None
    branch_weighted_minus_unweighted_dist2_last: float | None
    terminal_ess_fraction_last: float | None
    target_label_mismatch_fraction_last: float | None
    cache_weighted_terminal_dist2_last: float | None
    cache_unweighted_terminal_dist2_last: float | None
    score: float
    verdict: str
    advice: str


def _json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def _tail_mean(rows: list[dict[str, Any]], key: str, n: int = 200) -> float | None:
    vals: list[float] = []
    for row in rows[-n:]:
        val = _as_float(row.get(key))
        if val is not None:
            vals.append(val)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metrics_score(metrics: dict[str, Any], hist_last: dict[str, Any], cache_last: dict[str, Any]) -> tuple[float, str, str]:
    """A crude scalar score and advice for automated triage.

    This is not intended to replace visual inspection.  It pushes Codex toward
    the dominant failure mode: no useful bridge signal, signal learned on cache
    but not expressed during generation, or label/weight collapse.
    """
    entropy = _as_float(metrics.get("sample_entropy"))
    tv = _as_float(metrics.get("sample_total_variation"))
    learned = _as_float(metrics.get("learned_step_rms"))
    noise = _as_float(metrics.get("noise_step_rms"))
    gain = _as_float(hist_last.get("innovation_gain"))
    pred = _as_float(hist_last.get("prediction_rms"))
    target = _as_float(hist_last.get("target_rms"))
    centered = _as_float(hist_last.get("branch_centered_target_rms"))
    branch_snr = _as_float(hist_last.get("branch_signal_to_noise_ratio"))
    mismatch = _as_float(cache_last.get("target_label_mismatch_fraction"))
    branch_ess = _as_float(cache_last.get("branch_ess_fraction_mean"))
    weighted = _as_float(cache_last.get("weighted_terminal_dist2"))
    unweighted = _as_float(cache_last.get("unweighted_terminal_dist2"))

    learned_noise = (learned / noise) if learned and noise and noise > 0 else 0.0
    entropy_gap = max(0.0, (math.log(784.0) - entropy) if entropy is not None else 0.0)
    tv_term = max(0.0, min((tv or 0.0) / 0.35, 1.5))
    gain_term = max(0.0, min((gain or 0.0), 1.0))
    control_term = max(0.0, min(learned_noise, 1.0))
    score = 0.35 * tv_term + 0.30 * gain_term + 0.30 * control_term + 0.05 * min(entropy_gap / 0.5, 1.0)

    if mismatch is not None and mismatch > 1e-6:
        return score, "target-label-bug", "Fix target sampler first: target_label_mismatch_fraction is nonzero."
    if centered is not None and centered < 1e-3:
        return score, "no-branch-score", "Branch-centered target is near zero: terminal reward is not resolving an edge-score signal; try sharper/high-res reward or value-network estimator."
    if gain is not None and gain > 0.25 and learned_noise < 0.05:
        return score, "cache-learns-but-sampler-weak", "Cache objective is learned but learned_step/noise is tiny; run generation/control-strength sweep and log generation-state output RMS."
    if branch_ess is not None and branch_ess > 0.85:
        return score, "branch-weights-flat", "Branch ESS is too high: branch weights are nearly uniform; lower branch-terminal-ess-target or sharpen terminal reward."
    if weighted is not None and unweighted is not None and abs(weighted - unweighted) < 1e-3:
        return score, "terminal-reward-weak", "Weighted terminal distance barely improves over unweighted; reward/branching is too weak."
    if pred is not None and target is not None and target > 0 and pred / target < 0.25:
        return score, "underfit-target", "Model prediction RMS is much smaller than target RMS; increase training/width or reduce target variance."
    if score > 0.65:
        return score, "promising", "Promising metrics; run longer and save ablations/control-strength previews."
    return score, "still-noisy", "Still noisy; prioritize control-strength ablation plus reward/branch ESS diagnostics."


def diagnose_run(run_dir: Path, tail: int = 200) -> RunDiagnosis:
    metrics_path = run_dir / "experiment11_c0_metrics.json"
    history_path = run_dir / "experiment11_c0_history.json"
    cache_path = run_dir / "experiment11_c0_cache_diagnostics.csv"
    metadata_path = run_dir / "run_metadata.json"

    metrics = _json_load(metrics_path) if metrics_path.exists() else {}
    metadata = _json_load(metadata_path) if metadata_path.exists() else {}
    hist_obj = _json_load(history_path) if history_path.exists() else []
    if isinstance(hist_obj, dict):
        # Older code can write dict-of-lists.
        keys = list(hist_obj.keys())
        length = max((len(v) for v in hist_obj.values() if isinstance(v, list)), default=0)
        hist_rows = [{k: (hist_obj[k][i] if isinstance(hist_obj.get(k), list) and i < len(hist_obj[k]) else None) for k in keys} for i in range(length)]
    elif isinstance(hist_obj, list):
        hist_rows = [r for r in hist_obj if isinstance(r, dict)]
    else:
        hist_rows = []
    hist_last = hist_rows[-1] if hist_rows else {}
    cache_rows = _read_csv_rows(cache_path)
    cache_last = cache_rows[-1] if cache_rows else {}

    score, verdict, advice = _metrics_score(metrics, hist_last, cache_last)
    learned = _as_float(metrics.get("learned_step_rms"))
    noise = _as_float(metrics.get("noise_step_rms"))
    free = _as_float(metrics.get("free_step_rms"))
    return RunDiagnosis(
        run_dir=str(run_dir),
        run_id=run_dir.name,
        run_name=metadata.get("run_name") or metadata.get("args", {}).get("run_name") if isinstance(metadata, dict) else None,
        sample_entropy=_as_float(metrics.get("sample_entropy")),
        sample_total_variation=_as_float(metrics.get("sample_total_variation")),
        clipping_fraction=_as_float(metrics.get("clipping_fraction")),
        learned_step_rms=learned,
        free_step_rms=free,
        noise_step_rms=noise,
        learned_to_noise_ratio=(learned / noise if learned and noise and noise > 0 else None),
        learned_to_free_ratio=(learned / free if learned and free and free > 0 else None),
        innovation_gain_last=_as_float(hist_last.get("innovation_gain")),
        innovation_gain_tail_mean=_tail_mean(hist_rows, "innovation_gain", n=tail),
        prediction_rms_last=_as_float(hist_last.get("prediction_rms")),
        target_rms_last=_as_float(hist_last.get("target_rms")),
        branch_centered_target_rms_last=_as_float(hist_last.get("branch_centered_target_rms")),
        branch_signal_to_noise_ratio_last=_as_float(hist_last.get("branch_signal_to_noise_ratio")),
        branch_ess_fraction_mean_last=_as_float(cache_last.get("branch_ess_fraction_mean")),
        branch_weighted_minus_unweighted_dist2_last=_as_float(cache_last.get("branch_weighted_minus_unweighted_dist2")),
        terminal_ess_fraction_last=_as_float(cache_last.get("ess_fraction")),
        target_label_mismatch_fraction_last=_as_float(cache_last.get("target_label_mismatch_fraction")),
        cache_weighted_terminal_dist2_last=_as_float(cache_last.get("weighted_terminal_dist2")),
        cache_unweighted_terminal_dist2_last=_as_float(cache_last.get("unweighted_terminal_dist2")),
        score=score,
        verdict=verdict,
        advice=advice,
    )


def iter_run_dirs(runs_root: Path) -> Iterable[Path]:
    if not runs_root.exists():
        return []
    return sorted([p for p in runs_root.iterdir() if p.is_dir() and (p / "run_metadata.json").exists()])


def write_report(diagnostics: list[RunDiagnosis], out_md: Path, out_jsonl: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(diagnostics, key=lambda d: d.score, reverse=True)
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for diag in diagnostics:
            handle.write(json.dumps(asdict(diag), sort_keys=True) + "\n")
    with out_md.open("w", encoding="utf-8") as handle:
        handle.write("# Experiment 11 overnight diagnosis\n\n")
        handle.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        if not ordered:
            handle.write("No runs found.\n")
            return
        handle.write("## Ranked runs\n\n")
        handle.write("| rank | run | score | verdict | entropy | TV | learned/noise | innovation_gain | advice |\n")
        handle.write("|---:|---|---:|---|---:|---:|---:|---:|---|\n")
        for rank, diag in enumerate(ordered, start=1):
            handle.write(
                "| {rank} | `{run}` | {score:.3f} | {verdict} | {entropy} | {tv} | {ln} | {gain} | {advice} |\n".format(
                    rank=rank,
                    run=diag.run_id,
                    score=diag.score,
                    verdict=diag.verdict,
                    entropy="" if diag.sample_entropy is None else f"{diag.sample_entropy:.4f}",
                    tv="" if diag.sample_total_variation is None else f"{diag.sample_total_variation:.4f}",
                    ln="" if diag.learned_to_noise_ratio is None else f"{diag.learned_to_noise_ratio:.4g}",
                    gain="" if diag.innovation_gain_last is None else f"{diag.innovation_gain_last:.4f}",
                    advice=diag.advice.replace("|", "/"),
                )
            )
        handle.write("\n## Details\n\n")
        for diag in ordered:
            handle.write(f"### {diag.run_id}\n\n")
            for key, value in asdict(diag).items():
                handle.write(f"- `{key}`: {value}\n")
            handle.write("\n")


def _override_args(base: list[str], **updates: str) -> list[str]:
    """Return ``base`` with selected CLI options overridden safely."""
    out: list[str] = []
    skip_next = False
    update_keys = set(updates.keys())
    i = 0
    while i < len(base):
        token = base[i]
        if token in update_keys:
            i += 2
            continue
        out.append(token)
        if token.startswith("--") and i + 1 < len(base) and not base[i + 1].startswith("--"):
            out.append(base[i + 1])
            i += 2
        else:
            i += 1
    for key, value in updates.items():
        out.extend([key, str(value)])
    return out


def default_suite_args(args: argparse.Namespace) -> list[list[str]]:
    common = [
        "--data-root", str(args.data_root),
        "--runs-root", str(args.runs_root),
        "--base-channels", str(args.base_channels),
        "--train-steps", str(args.train_steps),
        "--batch-size", str(args.batch_size),
        "--cache-paths", str(args.cache_paths),
        "--cache-batch-size", str(args.cache_batch_size),
        "--cache-refresh-every", str(args.cache_refresh_every),
        "--teacher-mode", "branch-mean",
        "--branch-count", str(args.branch_count),
        "--branch-batch-size", str(args.branch_batch_size),
        "--branch-center-innovations",
        "--terminal-epsilon-mode", "branch-ess",
        "--branch-terminal-ess-target", str(args.branch_terminal_ess_target),
        "--teacher-stride", str(args.teacher_stride),
        "--time-slices-per-path", "1",
        "--model-output-mode", "innovation",
        "--loss-reweighting", "label-balanced",
        "--proposal-mode", "free",
        "--hybrid-loss-weight", "0",
        "--ot-match-mode", "topk",
        "--ot-nearest-top-k", "32",
        "--eta-l2-weight", "0",
        "--weight-decay", "0",
        "--reference-free-weight", "0.03",
        "--sample-steps", str(args.sample_steps),
        "--num-samples", str(args.num_samples),
        "--value-fd-diagnostic-states", str(args.value_fd_diagnostic_states),
        "--value-fd-diagnostic-edges", str(args.value_fd_diagnostic_edges),
        "--value-fd-diagnostic-branches", str(args.value_fd_diagnostic_branches),
        "--value-fd-epsilon", str(args.value_fd_epsilon),
        "--save-cache-previews",
    ]
    if args.no_amp:
        common.append("--no-amp")
    if args.no_progress:
        common.append("--no-progress")
    if args.device:
        common.extend(["--device", args.device])
    seed = int(args.seed)
    suite = [
        ["--run-name", "overnight-c2-1-noise002-strength5", "--reference-noise-weight", "0.02", "--control-strength", "5.0", "--control-output-clip", "20.0", "--seed", str(seed), *common],
        ["--run-name", "overnight-c2-1-noise002-branch64", "--reference-noise-weight", "0.02", "--control-strength", "5.0", "--control-output-clip", "20.0", "--seed", str(seed + 11), *_override_args(common, **{"--branch-count": "64"})],
        ["--run-name", "overnight-c2-1-noise005-strength10", "--reference-noise-weight", "0.05", "--control-strength", "10.0", "--control-output-clip", "30.0", "--seed", str(seed + 22), *common],
        ["--run-name", "overnight-c2-1-branchess015", "--reference-noise-weight", "0.02", "--control-strength", "5.0", "--control-output-clip", "20.0", "--seed", str(seed + 33), *_override_args(common, **{"--branch-terminal-ess-target": "0.15"})],
    ]
    return suite[: max(0, int(args.max_runs))]


def run_command(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def command_run_name(cmd_args: list[str]) -> str:
    if "--run-name" in cmd_args:
        i = cmd_args.index("--run-name")
        if i + 1 < len(cmd_args):
            return cmd_args[i + 1]
    return "experiment11"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diagnose", help="diagnose existing Experiment 11 runs")
    d.add_argument("--runs-root", type=Path, default=Path("runs/experiment11"))
    d.add_argument("--out-md", type=Path, default=Path("runs/experiment11/overnight_report.md"))
    d.add_argument("--out-jsonl", type=Path, default=Path("runs/experiment11/overnight_report.jsonl"))
    d.add_argument("--tail", type=int, default=200)

    r = sub.add_parser("run-suite", help="run a default overnight Experiment 11 suite")
    r.add_argument("--data-root", type=Path, default=Path("mnist_data"))
    r.add_argument("--runs-root", type=Path, default=Path("runs/experiment11"))
    r.add_argument("--logs-root", type=Path, default=Path("runs/experiment11/overnight_logs"))
    r.add_argument("--out-md", type=Path, default=Path("runs/experiment11/overnight_report.md"))
    r.add_argument("--out-jsonl", type=Path, default=Path("runs/experiment11/overnight_report.jsonl"))
    r.add_argument("--max-runs", type=int, default=4)
    r.add_argument("--train-steps", type=int, default=5000)
    r.add_argument("--batch-size", type=int, default=256)
    r.add_argument("--base-channels", type=int, default=48)
    r.add_argument("--cache-paths", type=int, default=1024)
    r.add_argument("--cache-batch-size", type=int, default=128)
    r.add_argument("--cache-refresh-every", type=int, default=1000)
    r.add_argument("--branch-count", type=int, default=32)
    r.add_argument("--branch-batch-size", type=int, default=128)
    r.add_argument("--branch-terminal-ess-target", type=float, default=0.35)
    r.add_argument("--teacher-stride", type=int, default=16)
    r.add_argument("--sample-steps", type=int, default=256)
    r.add_argument("--num-samples", type=int, default=64)
    r.add_argument("--value-fd-diagnostic-states", type=int, default=8)
    r.add_argument("--value-fd-diagnostic-edges", type=int, default=4)
    r.add_argument("--value-fd-diagnostic-branches", type=int, default=4)
    r.add_argument("--value-fd-epsilon", type=float, default=1e-4)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--device", type=str, default=None)
    r.add_argument("--no-amp", action="store_true")
    r.add_argument("--no-progress", action="store_true")

    args = parser.parse_args()
    if args.cmd == "diagnose":
        diags = [diagnose_run(p, tail=int(args.tail)) for p in iter_run_dirs(args.runs_root)]
        write_report(diags, args.out_md, args.out_jsonl)
        print(f"Wrote {args.out_md}")
        print(f"Wrote {args.out_jsonl}")
        if diags:
            best = max(diags, key=lambda x: x.score)
            print(f"Best: {best.run_id} score={best.score:.3f} verdict={best.verdict}: {best.advice}")
        return

    if args.cmd == "run-suite":
        args.runs_root.mkdir(parents=True, exist_ok=True)
        commands = default_suite_args(args)
        for i, cmd_args in enumerate(commands, start=1):
            name = command_run_name(cmd_args)
            cmd = [sys.executable, "-m", "mnist.experiment11_c0", *cmd_args]
            print(f"\n===== Overnight run {i}/{len(commands)}: {name} =====")
            code = run_command(cmd, args.logs_root / f"{i:02d}_{name}.log")
            print(f"Run {name} exited with code {code}")
            diags = [diagnose_run(p) for p in iter_run_dirs(args.runs_root)]
            write_report(diags, args.out_md, args.out_jsonl)
            if code != 0:
                print("Stopping suite because a run failed.")
                raise SystemExit(code)
        print(f"\nSuite done. Report: {args.out_md}")
        return


if __name__ == "__main__":
    main()
