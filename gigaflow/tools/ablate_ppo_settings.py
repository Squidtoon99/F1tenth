#!/usr/bin/env python3
"""Controlled PPO ablations: KL early-stop, BF16 AMP, recurrent minibatch size."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.trainer import build_trainer

ROOT = Path(__file__).resolve().parents[1]


def _finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def run_trial(
    *,
    base_cfg_path: Path,
    label: str,
    num_worlds: int,
    max_agents: int,
    rollout_length: int,
    minibatch_size: int,
    amp: bool,
    target_kl: float,
    updates: int,
    warmup: int,
    seed: int,
) -> dict:
    cfg = load_config(base_cfg_path)
    worlds = replace(
        cfg.worlds,
        num_worlds=int(num_worlds),
        max_agents_per_world=int(max_agents),
        device="cuda",
    )
    ppo = replace(
        cfg.ppo,
        rollout_length=int(rollout_length),
        minibatch_size=int(minibatch_size),
        amp=bool(amp),
        target_kl=float(target_kl),
        total_updates=max(int(updates) + int(warmup) + 2, 8),
        num_epochs=3,
    )
    # Synthetic oval only needs a soft budget for ablations.
    profiling = replace(
        cfg.profiling,
        enabled=True,
        report_interval_updates=1,
        estimate_bytes_budget=max(cfg.profiling.estimate_bytes_budget, 82_000_000_000),
    )
    wandb = replace(cfg.wandb, enabled=False)
    cfg = replace(
        cfg,
        seed=int(seed),
        worlds=worlds,
        ppo=ppo,
        profiling=profiling,
        wandb=wandb,
        tracks=replace(cfg.tracks, num_tracks=1),
    )
    atlas = make_synthetic_oval_atlas(max_agents=int(max_agents))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    trainer = build_trainer(cfg, atlas=atlas, device="cuda", run_dir=None)
    trainer.setup()

    rows = []
    oom = False
    err = None
    try:
        for i in range(int(warmup) + int(updates)):
            t0 = time.perf_counter()
            progress = trainer.train_update()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            m = dict(progress.metrics)
            m["wall_update_s"] = elapsed
            m["update_i"] = i
            for k in ("ppo_s", "collect_s", "update_s", "reconstruct_s"):
                if k in trainer.profile:
                    m[k] = float(trainer.profile[k])
            if i >= int(warmup):
                rows.append(m)
    except torch.cuda.OutOfMemoryError as exc:
        oom = True
        err = f"OOM: {exc}"
    except Exception as exc:  # noqa: BLE001 — ablation harness records failures
        err = f"{type(exc).__name__}: {exc}"

    def mean(key: str) -> float | None:
        vals = [float(r[key]) for r in rows if key in r]
        if not vals:
            return None
        return sum(vals) / len(vals)

    peak_mb = (
        float(torch.cuda.max_memory_allocated() / 1024**2)
        if torch.cuda.is_available()
        else 0.0
    )
    out = {
        "label": label,
        "num_worlds": num_worlds,
        "max_agents": max_agents,
        "rollout_length": rollout_length,
        "minibatch_size": minibatch_size,
        "amp": amp,
        "target_kl": target_kl,
        "updates_measured": len(rows),
        "oom": oom,
        "error": err,
        "peak_alloc_mb": peak_mb,
        "mean_epochs_completed": mean("epochs_completed"),
        "mean_approx_kl": mean("approx_kl"),
        "mean_clip_fraction": mean("clip_fraction"),
        "mean_policy_loss": mean("policy_loss"),
        "mean_value_loss": mean("value_loss"),
        "mean_retention": mean("retention"),
        "mean_early_stopped": mean("early_stopped"),
        "mean_grad_norm": mean("grad_norm"),
        "mean_wall_update_s": mean("wall_update_s"),
        "mean_ppo_s": mean("ppo_s") if rows and "ppo_s" in rows[0] else None,
        "mean_collect_s": mean("collect_s") if rows and "collect_s" in rows[0] else None,
        "mean_transitions_per_s": mean("transitions_per_s"),
        "mean_pre_update_approx_kl": mean("pre_update_approx_kl"),
        "all_finite": all(
            _finite(float(r.get("policy_loss", float("nan"))))
            and _finite(float(r.get("value_loss", float("nan"))))
            and _finite(float(r.get("grad_norm", float("nan"))))
            and _finite(float(r.get("approx_kl", float("nan"))))
            for r in rows
        )
        if rows
        else False,
    }
    if rows and "ppo_s" not in rows[0] and trainer.profile:
        # Fall back to trainer.profile keys if metrics omit them.
        out["profile_last"] = {k: float(v) for k, v in trainer.profile.items()}
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "gpu_smoke.yaml")
    p.add_argument("--suite", choices=["kl", "amp_mb", "all"], default="all")
    p.add_argument("--num-worlds", type=int, default=64)
    p.add_argument("--max-agents", type=int, default=8)
    p.add_argument("--rollout-length", type=int, default=64)
    p.add_argument("--updates", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    trials: list[dict] = []
    if args.suite in ("kl", "all"):
        for kl, label in (
            (0.02, "kl_0.02"),
            (0.04, "kl_0.04"),
            (0.0, "kl_disabled"),
        ):
            trials.append(
                dict(
                    label=label,
                    minibatch_size=4096,
                    amp=False,
                    target_kl=kl,
                )
            )
    if args.suite in ("amp_mb", "all"):
        for amp in (False, True):
            for mb in (4096, 8192, 16384, 32768):
                trials.append(
                    dict(
                        label=f"amp_{int(amp)}_mb_{mb}",
                        minibatch_size=mb,
                        amp=amp,
                        target_kl=0.0,  # isolate throughput once KL decision known
                    )
                )

    results = []
    for t in trials:
        print(f"=== trial {t['label']} ===", flush=True)
        row = run_trial(
            base_cfg_path=args.config,
            label=t["label"],
            num_worlds=args.num_worlds,
            max_agents=args.max_agents,
            rollout_length=args.rollout_length,
            minibatch_size=t["minibatch_size"],
            amp=t["amp"],
            target_kl=t["target_kl"],
            updates=args.updates,
            warmup=args.warmup,
            seed=args.seed,
        )
        print(json.dumps(row, indent=2, sort_keys=True), flush=True)
        results.append(row)
        torch.cuda.empty_cache()

    payload = {
        "device": torch.cuda.get_device_name(0),
        "suite": args.suite,
        "num_worlds": args.num_worlds,
        "max_agents": args.max_agents,
        "rollout_length": args.rollout_length,
        "updates": args.updates,
        "warmup": args.warmup,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
