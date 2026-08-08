#!/usr/bin/env python3
"""Throughput smoke benchmark for the N-agent Warp simulator."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.kernels import build_simulator, world_slot_layout
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sim = build_simulator(cfg, make_synthetic_oval_atlas(), args.device)
    n = world_slot_layout(cfg).num_slots
    actions = torch.zeros((n, 2), dtype=torch.float32)
    actions[:, 0] = 0.15

    # Warmup (kernel compile).
    sim.step(actions)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        sim.step(actions)
    elapsed = time.perf_counter() - t0
    active = int(sim.state().active.sum().item())
    print(
        f"device={args.device} slots={n} active~={active} steps={args.steps} "
        f"elapsed_s={elapsed:.3f} world_ticks_per_s={args.steps / elapsed:.1f} "
        f"agent_transitions_per_s={(active * args.steps) / elapsed:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
