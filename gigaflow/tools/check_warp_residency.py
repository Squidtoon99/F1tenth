#!/usr/bin/env python3
"""Fail if the Warp CUDA runtime path is host-resident or falls back to CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import warp as wp

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.kernels import build_simulator, world_slot_layout
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not wp.is_cuda_available():
        print(json.dumps({"ok": False, "error": "CUDA/Warp unavailable"}))
        return 2

    cfg = load_config(args.config)
    if not str(cfg.worlds.device).startswith("cuda"):
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"config.worlds.device={cfg.worlds.device!r} is not cuda",
                }
            )
        )
        return 2

    wp.init()
    atlas = make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world)
    sim = build_simulator(cfg, atlas, "cuda")
    n = world_slot_layout(cfg).num_slots
    actions = torch.zeros((n, 2), dtype=torch.float32, device="cuda")
    actions[:, 0] = 0.15

    # Warmup compile.
    out = sim.step(actions)
    torch.cuda.synchronize()

    alloc0 = torch.cuda.memory_allocated()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(args.steps):
        out = sim.step(actions)
    t1.record()
    torch.cuda.synchronize()
    ms = t0.elapsed_time(t1)

    x = sim.buffers.torch_arrays.x
    sensor = out["sensor_obs"]
    rewards = out["rewards"]
    payload = {
        "ok": True,
        "device": sim.buffers.device_str,
        "wp_device": str(sim.buffers.wp_device),
        "slots": n,
        "steps": args.steps,
        "step_ms_total": ms,
        "world_ticks_per_s": args.steps / max(ms / 1000.0, 1e-9),
        "x_device": str(x.device),
        "sensor_device": str(sensor.device),
        "rewards_device": str(rewards.device),
        "cuda_alloc_bytes": int(torch.cuda.memory_allocated()),
        "cuda_alloc_delta_bytes": int(torch.cuda.memory_allocated() - alloc0),
        "finite_sensor": bool(torch.isfinite(sensor).all().item()),
        "finite_rewards": bool(torch.isfinite(rewards).all().item()),
        "broadphase_overflow": int(out["broadphase_overflow"]),
    }
    if payload["device"] != "cuda" or not str(payload["x_device"]).startswith("cuda"):
        payload["ok"] = False
        payload["error"] = "simulator tensors are not CUDA-resident"
    if not payload["finite_sensor"] or not payload["finite_rewards"]:
        payload["ok"] = False
        payload["error"] = "non-finite sensor/reward outputs"
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
