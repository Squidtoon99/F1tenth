#!/usr/bin/env python3
"""Measure the adaptive advantage filter's real retention at production scale.

D4 (chunked-BPTT segment skipping) only pays off if the filter discards most
transitions, the way the paper's ~80% discard rate does. Measured here on
production_rtx4080.yaml: retention runs 82-97% (rising over training), far
above that assumption, so segment skipping has ~0% of empty segments to skip.
Re-run this before reconsidering that optimization.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.trainer import build_trainer

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "production_rtx4080.yaml")
    p.add_argument("--updates", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for production-dimension measurement")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.config)
    tr = build_trainer(cfg, device="cuda")
    tr.setup()

    retentions = []
    update_times = []
    for i in range(int(args.updates)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        progress = tr.train_update()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        update_times.append(dt)
        retentions.append(float(progress.metrics["retention"]))
        print(
            f"update {i}: retention={retentions[-1]:.4f} "
            f"update_s={dt:.3f} profile={dict(tr.profile)}"
        )

    out = {
        "config": str(args.config),
        "num_worlds": cfg.worlds.num_worlds,
        "max_agents_per_world": cfg.worlds.max_agents_per_world,
        "rollout_length": cfg.ppo.rollout_length,
        "num_slots": tr.layout.num_slots,
        "retentions": retentions,
        "retention_mean": float(np.mean(retentions)),
        "retention_mean_excl_warmup": (
            float(np.mean(retentions[2:])) if len(retentions) > 2 else None
        ),
        "update_s_mean": float(np.mean(update_times)),
        "update_s_mean_excl_warmup": (
            float(np.mean(update_times[1:])) if len(update_times) > 1 else None
        ),
    }
    text = json.dumps(out, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
