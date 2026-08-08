#!/usr/bin/env python3
"""Aggregate clean-grace short trials into filter_matrix_summary.json."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

GFROOT = Path(__file__).resolve().parents[1]
OUT = GFROOT / "outputs" / "clean_grace_period"
TRIALS = OUT / "trials"
MB = 2048.0


def _load_metrics(trial_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(trial_dir.glob("metrics_*.json")):
        m = json.loads(p.read_text())
        prog = m.get("progress", {})
        prof = m.get("profile", {})
        valid = float(prog.get("valid_transitions") or 0.0)
        retention = float(prog.get("retention") or 0.0)
        retained = retention * valid
        rows.append(
            {
                "update_index": int(p.stem.split("_")[-1]),
                "rollout_transitions": prof.get("transitions"),
                "valid_transitions": valid,
                "retention": retention,
                "retained_transitions": retained,
                "retained_per_minibatch": retained / MB if MB > 0 else None,
                "filter_eta": prog.get("filter_eta"),
                "advantage_raw_p99": prog.get("advantage_raw_p99"),
                "advantage_raw_p99_9": prog.get("advantage_raw_p99_9"),
                "actor_grad_norm": prog.get("actor_grad_norm"),
                "approx_kl": prog.get("approx_kl"),
                "candidate_kl": prog.get("candidate_kl"),
                "clip_fraction": prog.get("clip_fraction"),
                "progress_mean": prog.get("progress_mean"),
                "transitions_per_s": prof.get("transitions_per_s"),
                "update_s": prof.get("update_s"),
            }
        )
    return rows


def _mean_key(rows: list[dict], key: str, tail: int = 10) -> float | None:
    vals = [
        float(r[key])
        for r in rows[-tail:]
        if r.get(key) is not None and math.isfinite(float(r[key]))
    ]
    return sum(vals) / len(vals) if vals else None


def _trial_entry(trial_dir: Path, trial_id: str, filter_semantics: str) -> dict:
    manifest_path = trial_dir / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    )
    rows = _load_metrics(trial_dir)
    cfg_path = trial_dir / "trial_config.yaml"
    if not cfg_path.is_file():
        cfg_path = trial_dir / "config.json"
    return {
        "trial_id": trial_id,
        "dir": str(trial_dir),
        "filter_semantics": filter_semantics,
        "updates": manifest.get("final_update_index"),
        "shutdown_reason": manifest.get("shutdown_reason"),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "mean_retention_tail10": _mean_key(rows, "retention"),
        "mean_retained_tail10": _mean_key(rows, "retained_transitions"),
        "mean_retained_per_mb_tail10": _mean_key(rows, "retained_per_minibatch"),
        "mean_adv_p99_tail10": _mean_key(rows, "advantage_raw_p99"),
        "mean_adv_p99_9_tail10": _mean_key(rows, "advantage_raw_p99_9"),
        "mean_actor_grad_norm_tail10": _mean_key(rows, "actor_grad_norm"),
        "mean_candidate_kl_tail10": _mean_key(rows, "candidate_kl"),
        "mean_approx_kl_tail10": _mean_key(rows, "approx_kl"),
        "mean_clip_fraction_tail10": _mean_key(rows, "clip_fraction"),
        "mean_throughput_tail10": _mean_key(rows, "transitions_per_s"),
        "progress_mean_last": rows[-1]["progress_mean"] if rows else None,
        "metrics_tail": rows[-3:],
    }


def _discover_filter_trials() -> list[tuple[str, str, str]]:
    mapping = []
    for d in sorted(TRIALS.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if "d_filter_eta005" in name:
            mapping.append((d, "d_filter_eta005", "eta_scale=0.005 (higher retention)"))
        elif "e_filter_off" in name:
            mapping.append((d, "e_filter_off", "adaptive_filter_enabled=false"))
    return mapping


def _pick_config(entries: list[dict]) -> tuple[str, str]:
    """Prefer highest retention with stable KL/grad and best progress."""
    scored = []
    for e in entries:
        if e.get("updates") is None or e.get("shutdown_reason") != "completed":
            continue
        retention = e.get("mean_retention_tail10") or 0.0
        progress = e.get("progress_mean_last") or 0.0
        kl = e.get("mean_candidate_kl_tail10") or 1.0
        grad = e.get("mean_actor_grad_norm_tail10") or 999.0
        scored.append((retention, progress, -kl, -grad, e))
    if not scored:
        return (
            str(GFROOT / "configs" / "clean_grace_rtx4080.yaml"),
            "fallback: clean_grace_rtx4080 (b_anneal baseline)",
        )
    scored.sort(reverse=True)
    winner = scored[0][4]
    tid = winner["trial_id"]
    if tid == "d_filter_eta005":
        return (
            str(GFROOT / "configs" / "clean_grace_filter_eta005.yaml"),
            "d_filter_eta005: eta_scale=0.005 higher retention",
        )
    if tid == "e_filter_off":
        return (
            str(GFROOT / "configs" / "clean_grace_filter_off.yaml"),
            "e_filter_off: unfiltered",
        )
    if tid == "b_anneal":
        return (
            str(GFROOT / "configs" / "clean_grace_rtx4080.yaml"),
            "b_anneal: adaptive eta_scale=0.01",
        )
    return (
        str(GFROOT / "configs" / "clean_grace_rtx4080.yaml"),
        f"fallback after {tid}",
    )


def main() -> int:
    entries: list[dict] = []

    baseline = [
        ("20260806_170800_a_fixed_ent", "a_fixed_ent", "fixed ent_coef=0.01"),
        ("20260806_170800_b_anneal", "b_anneal", "eta_scale=0.01 (default adaptive)"),
        (
            "20260806_170800_c_anneal_sparse_eval",
            "c_anneal_sparse_eval",
            "eta_scale=0.01 eval_interval=200",
        ),
    ]
    for dirname, tid, sem in baseline:
        d = TRIALS / dirname
        if d.is_dir():
            entries.append(_trial_entry(d, tid, sem))

    for d, tid, sem in _discover_filter_trials():
        entries.append(_trial_entry(d, tid, sem))

    config_path, rationale = _pick_config(entries)
    summary = {
        "entries": entries,
        "selected_1h_config": config_path,
        "selection_rationale": rationale,
    }
    out_path = OUT / "filter_matrix_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
