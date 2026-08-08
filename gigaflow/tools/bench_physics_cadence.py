#!/usr/bin/env python3
"""Fidelity + throughput gate for physics substep cadences (control_dt=0.1s)."""

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

ROOT = Path(__file__).resolve().parents[1]

# Candidates: (control_interval, sim_dt) with product == 0.1 s.
CANDIDATES = (
    (20, 0.005),
    (10, 0.01),
    (5, 0.02),
    (4, 0.025),
)

# Trajectory fidelity vs 20-substep reference (real sim behavior, not synthetic).
MAX_POS_ERR_M = 0.05
MAX_YAW_ERR_RAD = 0.05
MAX_SPEED_ERR_MPS = 0.15
MAX_PROGRESS_ERR_M = 0.10


def _cfg_for(interval: int, sim_dt: float, *, device: str, worlds: int, agents: int):
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    agents_c = replace(
        cfg.agents,
        control_interval=int(interval),
        sim_dt=float(sim_dt),
        control_hz=10.0,
        async_respawn=False,
    )
    worlds_c = replace(
        cfg.worlds,
        num_worlds=int(worlds),
        max_agents_per_world=int(agents),
        device=device,
        solo_world_fraction=0.0,
        density_bins=("dense",),
    )
    return replace(cfg, agents=agents_c, worlds=worlds_c, seed=0)


def _force_all_active(sim) -> None:
    t = sim.buffers.torch_arrays
    t.active.fill_(1)
    t.trainable.fill_(1)


def _snapshot(sim) -> dict[str, torch.Tensor]:
    t = sim.buffers.torch_arrays
    return {
        "x": t.x.detach().clone(),
        "y": t.y.detach().clone(),
        "yaw": t.yaw.detach().clone(),
        "vx": t.vx.detach().clone(),
        "vy": t.vy.detach().clone(),
        "progress_s": t.progress_s.detach().clone(),
        "active": t.active.detach().clone(),
    }


def _restore(sim, snap: dict[str, torch.Tensor]) -> None:
    t = sim.buffers.torch_arrays
    for k, v in snap.items():
        getattr(t, k).copy_(v)


def run_fidelity(device: str, steps: int, worlds: int, agents: int) -> dict:
    ref_cfg = _cfg_for(20, 0.005, device=device, worlds=worlds, agents=agents)
    atlas = make_synthetic_oval_atlas(max_agents=agents)
    ref = build_simulator(ref_cfg, atlas, device)
    _force_all_active(ref)
    n = world_slot_layout(ref_cfg).num_slots
    # Aggressive open-loop schedule to stress tire/steer integration.
    actions = torch.zeros((steps, n, 2), device=device if device != "cpu" else "cpu")
    for i in range(steps):
        actions[i, :, 0] = 0.85 if (i // 8) % 2 == 0 else -0.35
        actions[i, :, 1] = 0.7 if (i // 5) % 2 == 0 else -0.7

    ref_snap0 = _snapshot(ref)
    ref_traj = []
    for i in range(steps):
        ref.step(actions[i])
        ref_traj.append(_snapshot(ref))

    results = []
    for interval, sim_dt in CANDIDATES:
        cfg = _cfg_for(interval, sim_dt, device=device, worlds=worlds, agents=agents)
        sim = build_simulator(cfg, atlas, device)
        _force_all_active(sim)
        _restore(sim, ref_snap0)
        max_pos = 0.0
        max_yaw = 0.0
        max_spd = 0.0
        max_prog = 0.0
        finite = True
        t0 = time.perf_counter()
        for i in range(steps):
            out = sim.step(actions[i])
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            st = _snapshot(sim)
            if not torch.isfinite(st["x"]).all() or not torch.isfinite(out["rewards"]).all():
                finite = False
                break
            dx = st["x"] - ref_traj[i]["x"]
            dy = st["y"] - ref_traj[i]["y"]
            pos = torch.sqrt(dx * dx + dy * dy).max().item()
            yaw = (st["yaw"] - ref_traj[i]["yaw"]).abs().max().item()
            spd = torch.sqrt(st["vx"] ** 2 + st["vy"] ** 2)
            spd_ref = torch.sqrt(
                ref_traj[i]["vx"] ** 2 + ref_traj[i]["vy"] ** 2
            )
            spde = (spd - spd_ref).abs().max().item()
            prog = (st["progress_s"] - ref_traj[i]["progress_s"]).abs().max().item()
            max_pos = max(max_pos, float(pos))
            max_yaw = max(max_yaw, float(yaw))
            max_spd = max(max_spd, float(spde))
            max_prog = max(max_prog, float(prog))
        elapsed = time.perf_counter() - t0
        passed = (
            finite
            and max_pos <= MAX_POS_ERR_M
            and max_yaw <= MAX_YAW_ERR_RAD
            and max_spd <= MAX_SPEED_ERR_MPS
            and max_prog <= MAX_PROGRESS_ERR_M
        )
        # Reference cadence always passes by definition.
        if interval == 20:
            passed = finite
            max_pos = 0.0
            max_yaw = 0.0
            max_spd = 0.0
            max_prog = 0.0
        results.append(
            {
                "control_interval": interval,
                "sim_dt": sim_dt,
                "passed": passed,
                "finite": finite,
                "max_pos_err_m": max_pos,
                "max_yaw_err_rad": max_yaw,
                "max_speed_err_mps": max_spd,
                "max_progress_err_m": max_prog,
                "elapsed_s": elapsed,
                "world_ticks_per_s": steps / max(elapsed, 1e-9),
            }
        )

    # Fastest (= fewest substeps) among those that pass.
    passing = [r for r in results if r["passed"]]
    selected = min(passing, key=lambda r: r["control_interval"]) if passing else None
    return {
        "ok": selected is not None,
        "device": device,
        "steps": steps,
        "gates": {
            "max_pos_err_m": MAX_POS_ERR_M,
            "max_yaw_err_rad": MAX_YAW_ERR_RAD,
            "max_speed_err_mps": MAX_SPEED_ERR_MPS,
            "max_progress_err_m": MAX_PROGRESS_ERR_M,
        },
        "candidates": results,
        "selected": selected,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--worlds", type=int, default=2)
    p.add_argument("--agents", type=int, default=2)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    report = run_fidelity(args.device, args.steps, args.worlds, args.agents)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
