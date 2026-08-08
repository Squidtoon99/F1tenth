#!/usr/bin/env python3
"""Incremental LiDAR/runtime optimization microbench (CUDA, conda env g2)."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
import warp as wp

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.kernels import build_simulator, world_slot_layout
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.sim.layout_local import LIDAR_DIM
from gigaflow_f1tenth.sim.sensors import (
    lidar_and_proprio_kernel,
    lidar_and_proprio_kernel_global_scan,
    lidar_beam_parallel_kernel,
)

ROOT = Path(__file__).resolve().parents[1]


def _time_ms(fn, reps: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "median_ms": times[len(times) // 2],
        "p20_ms": times[max(0, len(times) // 5)],
        "p80_ms": times[min(len(times) - 1, 4 * len(times) // 5)],
        "mean_ms": sum(times) / len(times),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--num-worlds", type=int, default=32)
    p.add_argument("--max-agents", type=int, default=8)
    p.add_argument("--reps", type=int, default=40)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    cfg = load_config(ROOT / "configs" / "gpu_smoke.yaml")
    cfg = replace(
        cfg,
        worlds=replace(
            cfg.worlds,
            num_worlds=int(args.num_worlds),
            max_agents_per_world=int(args.max_agents),
            device="cuda",
            density_bins=["dense"],
        ),
    )
    atlas = make_synthetic_oval_atlas(max_agents=int(args.max_agents))
    sim = build_simulator(cfg, atlas, "cuda")
    n = world_slot_layout(cfg).num_slots
    actions = torch.zeros((n, 2), device="cuda")
    actions[:, 0] = 0.25
    for _ in range(8):
        sim.step(actions)

    def launch(kernel, dim):
        inputs = list(sim._sensor_kernel_inputs())
        wp.launch(kernel, dim=dim, inputs=inputs, device=sim.buffers.wp_device)

    # Isolate kernels (bypass runtime graph / clone path).
    modes = {
        "global_serial": lambda: launch(lidar_and_proprio_kernel_global_scan, n),
        "world_local_serial": lambda: launch(lidar_and_proprio_kernel, n),
        "beam_parallel": lambda: launch(
            lidar_beam_parallel_kernel, (n, int(LIDAR_DIM))
        ),
    }
    kernel_results = {
        name: _time_ms(fn, args.reps, args.warmup) for name, fn in modes.items()
    }

    # Full-step with production path toggles.
    def configure(*, beam: bool, sensor_graph: bool):
        sim._use_beam_parallel_lidar = beam
        sim._sensor_cuda_graph = None
        sim._sensor_cuda_graph_enabled = sensor_graph
        sim._sensor_cuda_graph_warmup_left = 2 if sensor_graph else 0
        sim._sensor_cuda_graph_status = "pending" if sensor_graph else "disabled"
        sim._invalidate_cuda_graph()
        # Re-enable physics graph (invalidate clears it).
        sim._cuda_graph_enabled = True
        for _ in range(6):
            sim.step(actions)

    step_results = {}
    for label, beam, sgraph in (
        ("step_world_local_serial", False, False),
        ("step_beam_parallel", True, False),
        ("step_beam_parallel_sensor_graph", True, True),
    ):
        configure(beam=beam, sensor_graph=sgraph)
        step_results[label] = _time_ms(lambda: sim.step(actions), args.reps, args.warmup)
        step_results[label]["sensor_graph_status"] = sim._sensor_cuda_graph_status
        step_results[label]["physics_graph_status"] = sim._cuda_graph_status

    out = {
        "device": torch.cuda.get_device_name(0),
        "slots": n,
        "active": int(sim.state().active.sum().item()),
        "num_worlds": int(args.num_worlds),
        "max_agents": int(args.max_agents),
        "reps": int(args.reps),
        "kernels": kernel_results,
        "steps": step_results,
    }
    text = json.dumps(out, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
