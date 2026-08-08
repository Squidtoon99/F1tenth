#!/usr/bin/env python3
"""Measure learner VRAM vs num_worlds and extrapolate H100 60–70 GiB target."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.trainer import build_trainer

ROOT = Path(__file__).resolve().parents[1]


def _peak_mb_for_worlds(
    base_cfg,
    *,
    num_worlds: int,
    max_agents: int,
    rollout_length: int,
) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    worlds = replace(
        base_cfg.worlds,
        num_worlds=int(num_worlds),
        max_agents_per_world=int(max_agents),
        device="cuda",
    )
    ppo = replace(
        base_cfg.ppo,
        rollout_length=int(rollout_length),
        num_epochs=1,
        minibatch_size=min(2048, int(rollout_length) * max(8, num_worlds)),
        amp=False,
    )
    # Keep startup estimator from rejecting large probes.
    profiling = replace(base_cfg.profiling, estimate_bytes_budget=10**12)
    cfg = replace(base_cfg, worlds=worlds, ppo=ppo, profiling=profiling)
    atlas = make_synthetic_oval_atlas(max_agents=max_agents)
    tr = build_trainer(cfg, atlas=atlas, device="cuda")
    tr.setup()
    tr.train_update()
    torch.cuda.synchronize()
    alloc = float(torch.cuda.max_memory_allocated() / 1024**2)
    reserved = float(torch.cuda.max_memory_reserved() / 1024**2)
    active = float((tr.sim.buffers.torch_arrays.active > 0).sum().item())
    return {
        "num_worlds": int(num_worlds),
        "max_agents": int(max_agents),
        "slots": int(num_worlds * max_agents),
        "active_end": active,
        "peak_alloc_mb": alloc,
        "peak_reserved_mb": reserved,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "gpu_smoke.yaml")
    p.add_argument("--worlds", type=str, default="4,8,12")
    p.add_argument("--max-agents", type=int, default=8)
    p.add_argument("--rollout-length", type=int, default=32)
    p.add_argument("--h100-current-worlds", type=int, default=256)
    p.add_argument("--h100-current-mb", type=float, default=16351.0)
    p.add_argument("--target-gb-lo", type=float, default=60.0)
    p.add_argument("--target-gb-hi", type=float, default=70.0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    cfg = load_config(args.config)
    points = []
    for w in [int(x) for x in args.worlds.split(",") if x.strip()]:
        points.append(
            _peak_mb_for_worlds(
                cfg,
                num_worlds=w,
                max_agents=int(args.max_agents),
                rollout_length=int(args.rollout_length),
            )
        )

    # Fit alloc_mb ≈ a + b * worlds from local points.
    xs = [p["num_worlds"] for p in points]
    ys = [p["peak_alloc_mb"] for p in points]
    if len(points) >= 2 and xs[-1] != xs[0]:
        b = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    else:
        b = ys[0] / max(xs[0], 1)

    # Anchor slope with H100 production measurement when available.
    # Prefer H100 absolute point for target extrapolation:
    # mb(w) ≈ (h100_mb / h100_worlds) * w   if fixed overhead unknown,
    # else blend local intercept with H100 slope.
    h100_w = float(args.h100_current_worlds)
    h100_mb = float(args.h100_current_mb)
    slope_h100 = h100_mb / max(h100_w, 1.0)
    # Production target uses the H100 measured slope. Local points only validate
    # that allocation grows roughly linearly with worlds.
    intercept = 0.0
    slope = slope_h100
    local_slope = b

    def worlds_for_gb(gb: float) -> int:
        target_mb = gb * 1024.0
        return max(1, int((target_mb - intercept) / max(slope, 1e-6)))

    w60 = worlds_for_gb(args.target_gb_lo)
    w70 = worlds_for_gb(args.target_gb_hi)
    # Recommend midpoint of band, snapped to multiple of 64.
    w_rec = int(round(((w60 + w70) * 0.5) / 64.0) * 64)
    w_rec = max(64, w_rec)
    est_mb = intercept + slope * w_rec

    report = {
        "device": torch.cuda.get_device_name(0),
        "local_points": points,
        "fit": {
            "intercept_mb": intercept,
            "slope_mb_per_world": slope,
            "local_slope_mb_per_world": local_slope,
        },
        "h100_anchor": {
            "num_worlds": int(args.h100_current_worlds),
            "measured_mb": h100_mb,
            "slope_mb_per_world": slope_h100,
        },
        "target_band_gb": [args.target_gb_lo, args.target_gb_hi],
        "worlds_for_60gb": w60,
        "worlds_for_70gb": w70,
        "recommended_num_worlds": w_rec,
        "estimated_mb_at_recommended": est_mb,
        "estimated_gb_at_recommended": est_mb / 1024.0,
        "max_agents_per_world": int(args.max_agents),
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
