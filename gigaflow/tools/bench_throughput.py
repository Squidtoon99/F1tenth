#!/usr/bin/env python3
"""Before/after learner+sim throughput probe (CUDA)."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.kernels import build_simulator, world_slot_layout
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.trainer import build_trainer

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "gpu_smoke.yaml")
    p.add_argument("--label", type=str, default="bench")
    p.add_argument("--num-worlds", type=int, default=8)
    p.add_argument("--max-agents", type=int, default=4)
    p.add_argument("--rollout-length", type=int, default=16)
    p.add_argument("--sim-steps", type=int, default=40)
    p.add_argument("--updates", type=int, default=2)
    p.add_argument(
        "--legacy-collect",
        action="store_true",
        help="Duplicate LiDAR rebuild + full-slot actor (pre-optimization path)",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    cfg = load_config(args.config)
    worlds = replace(
        cfg.worlds,
        num_worlds=int(args.num_worlds),
        max_agents_per_world=int(args.max_agents),
        device="cuda",
    )
    ppo = replace(
        cfg.ppo,
        rollout_length=int(args.rollout_length),
        num_epochs=1,
        minibatch_size=max(64, int(args.rollout_length) * 4),
        amp=False,
    )
    cfg = replace(cfg, worlds=worlds, ppo=ppo)
    atlas = make_synthetic_oval_atlas(max_agents=int(args.max_agents))

    sim = build_simulator(cfg, atlas, "cuda")
    n = world_slot_layout(cfg).num_slots
    actions = torch.zeros((n, 2), device="cuda", dtype=torch.float32)
    actions[:, 0] = 0.2
    for _ in range(5):
        sim.step(actions)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(int(args.sim_steps)):
        sim.step(actions)
    torch.cuda.synchronize()
    sim_s = time.perf_counter() - t0

    tr = build_trainer(cfg, atlas=atlas, device="cuda")
    tr.setup()
    if args.legacy_collect:
        tr.force_rebuild_sensors = True
        tr.disable_compact_actor = True
    tr.train_update()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    progress = None
    for _ in range(int(args.updates)):
        progress = tr.train_update()
    torch.cuda.synchronize()
    learn_s = time.perf_counter() - t0
    assert progress is not None

    out = {
        "label": args.label,
        "device": torch.cuda.get_device_name(0),
        "slots": n,
        "control_interval": int(cfg.agents.control_interval),
        "sim_dt": float(cfg.agents.sim_dt),
        "sim": {
            "steps": int(args.sim_steps),
            "elapsed_s": sim_s,
            "world_ticks_per_s": int(args.sim_steps) / max(sim_s, 1e-9),
            "active": int(sim.state().active.sum().item()),
        },
        "learner": {
            "updates": int(args.updates),
            "elapsed_s": learn_s,
            "updates_per_s": int(args.updates) / max(learn_s, 1e-9),
            "profile": dict(tr.profile),
            "last_tps": float(progress.metrics.get("transitions_per_s", 0.0)),
        },
        "peak_alloc_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
    }
    text = json.dumps(out, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
