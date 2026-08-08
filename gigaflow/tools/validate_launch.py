#!/usr/bin/env python3
"""Validate-launch harness: atlas gates, invariants, soaks, smoke, trial."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gigaflow_f1tenth.config import load_config
from gigaflow_f1tenth.evaluation import run_soak
from gigaflow_f1tenth.kernels import build_simulator, world_slot_layout
from gigaflow_f1tenth.sim.geometry import make_synthetic_oval_atlas
from gigaflow_f1tenth.tracks import load_atlas
from gigaflow_f1tenth.trainer import build_trainer

ROOT = Path(__file__).resolve().parents[1]


def _device(preferred: str) -> str:
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return preferred


def check_atlas(cache_dir: Path) -> dict[str, Any]:
    atlas = load_atlas(str(cache_dir), device="cpu")
    man = atlas.manifest()
    names = [e.name for e in man]
    ok = len(man) == 23 and all(e.valid for e in man)
    return {
        "ok": ok,
        "num_tracks": len(man),
        "track_names": names,
        "total_length_m": sum(e.length_m for e in man),
        "capacities": [e.capacity for e in man],
        "all_valid": all(e.valid for e in man),
    }


def invariant_sweep(cache_dir: Path, device: str) -> dict[str, Any]:
    from gigaflow_f1tenth.sim.spawn import horizon_for_track

    atlas_obj = load_atlas(str(cache_dir), device="cpu")
    atlas = atlas_obj.view()
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    field_sizes = (1, 2, 4)
    anomalies: list[dict[str, Any]] = []
    cases = 0
    for n_agents in field_sizes:
        worlds = replace(
            cfg.worlds,
            num_worlds=1,
            max_agents_per_world=n_agents,
            device=device,
            density_bins=("dense",),
            solo_world_fraction=0.0,
        )
        agents = replace(cfg.agents, async_respawn=True)
        cfg_t = replace(cfg, worlds=worlds, agents=agents, seed=n_agents)
        sim = build_simulator(cfg_t, atlas, device)
        n = world_slot_layout(cfg_t).num_slots
        for tid, name in enumerate(atlas.track_ids):
            t = sim.buffers.torch_arrays
            t.track_id[:] = int(tid)
            horizon = horizon_for_track(cfg_t, sim.geom, int(tid))
            t.episode_horizon[:] = int(horizon)
            # Respawn onto the forced track (do not leave stale oval poses).
            sim.reset_agents(np.ones(n, dtype=np.uint8), seed=tid * 17 + n_agents)
            actions = torch.zeros((n, 2), dtype=torch.float32)
            actions[:, 0] = 0.15
            failed = False
            for step in range(30):
                out = sim.step(actions)
                state = sim.pack_state()
                if not torch.isfinite(state).all():
                    anomalies.append(
                        {"track": name, "agents": n_agents, "step": step, "kind": "state"}
                    )
                    failed = True
                    break
                if not torch.isfinite(out["rewards"]).all():
                    anomalies.append(
                        {"track": name, "agents": n_agents, "step": step, "kind": "reward"}
                    )
                    failed = True
                    break
                if not torch.isfinite(out["sensor_obs"]).all():
                    anomalies.append(
                        {"track": name, "agents": n_agents, "step": step, "kind": "lidar"}
                    )
                    failed = True
                    break
                overflow = out["broadphase_overflow"]
                ov = int(overflow.item()) if hasattr(overflow, "item") else int(overflow)
                if ov > 0:
                    anomalies.append(
                        {
                            "track": name,
                            "agents": n_agents,
                            "step": step,
                            "kind": "overflow",
                        }
                    )
                    failed = True
                    break
            cases += 1
            if failed:
                continue
    return {"ok": len(anomalies) == 0, "anomalies": anomalies, "cases": cases}


def scaling_benchmark(device: str) -> dict[str, Any]:
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    results = []
    for num_worlds, agents in ((4, 2), (8, 4), (16, 4)):
        worlds = replace(
            cfg.worlds,
            num_worlds=num_worlds,
            max_agents_per_world=agents,
            device=device,
        )
        cfg_b = replace(cfg, worlds=worlds)
        atlas = make_synthetic_oval_atlas(max_agents=agents)
        sim = build_simulator(cfg_b, atlas, device)
        n = world_slot_layout(cfg_b).num_slots
        act_dev = "cpu" if device == "cpu" else device
        actions = torch.zeros((n, 2), dtype=torch.float32, device=act_dev)
        actions[:, 0] = 0.2
        sim.step(actions)
        steps = 40
        t0 = time.perf_counter()
        for _ in range(steps):
            sim.step(actions)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        active = int(sim.state().active.sum().item())
        results.append(
            {
                "num_worlds": num_worlds,
                "max_agents": agents,
                "slots": n,
                "active": active,
                "world_ticks_per_s": steps / max(elapsed, 1e-9),
                "agent_transitions_per_s": (active * steps) / max(elapsed, 1e-9),
                "elapsed_s": elapsed,
            }
        )
    return {"ok": True, "device": device, "scales": results}


def convergence_smoke(run_dir: Path, device: str) -> dict[str, Any]:
    cfg = load_config(ROOT / "configs" / "convergence_smoke.yaml")
    worlds = replace(cfg.worlds, device=device)
    cfg = replace(cfg, worlds=worlds)
    atlas = make_synthetic_oval_atlas(max_agents=cfg.worlds.max_agents_per_world)
    trainer = build_trainer(cfg, atlas=atlas, device=device, run_dir=run_dir)
    trainer.setup()
    history = []
    for _ in range(int(cfg.ppo.total_updates)):
        progress = trainer.train_update()
        history.append(dict(progress.metrics))
    ckpt = run_dir / "ckpt_mid.pt"
    trainer.save_checkpoint(str(ckpt))
    filt = float(trainer.ppo.filter_state.ewma_max_abs_adv)
    idx = trainer.ppo.update_index
    trainer2 = build_trainer(cfg, atlas=atlas, device=device)
    trainer2.setup()
    trainer2.load_checkpoint(str(ckpt))
    assert trainer2.ppo.update_index == idx
    assert abs(float(trainer2.ppo.filter_state.ewma_max_abs_adv) - filt) < 1e-6
    trainer.export_actor(str(run_dir / "actor_final.pt"))
    value_losses = [h["value_loss"] for h in history]
    grads = [h["grad_norm"] for h in history]
    # Short smoke: require finite losses, nonzero grads, healthy entropy/retention,
    # and no entropy collapse. KL may spike under target_kl early-stop; require the
    # median late KL stay below a loose bound rather than every sample.
    late_kl = [h["approx_kl"] for h in history[len(history) // 2 :]]
    ok = (
        all(np.isfinite(value_losses))
        and all(g > 0.0 for g in grads)
        and all(h["entropy"] > 0.05 for h in history)
        and all(h["retention"] > 0.05 for h in history)
        and float(np.median(late_kl)) < 1.0
        and history[-1]["entropy"] > 0.1
    )
    return {
        "ok": ok,
        "updates": len(history),
        "value_loss_first": value_losses[0],
        "value_loss_last": value_losses[-1],
        "grad_norm_mean": float(np.mean(grads)),
        "entropy_last": history[-1]["entropy"],
        "approx_kl_last": history[-1]["approx_kl"],
        "retention_last": history[-1]["retention"],
        "checkpoint_resume_ok": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip-atlas-sweep", action="store_true")
    parser.add_argument("--skip-convergence", action="store_true")
    args = parser.parse_args()
    device = _device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"device": device}

    report["atlas"] = check_atlas(args.cache_dir)
    if report["atlas"]["ok"] and not args.skip_atlas_sweep:
        report["invariants"] = invariant_sweep(args.cache_dir, device)
    else:
        report["invariants"] = {"ok": False, "skipped": True}

    cfg_soak = load_config(ROOT / "configs" / "smoke.yaml")
    report["soak_random"] = run_soak(
        cfg_soak, steps=80, device=device, output_dir=out / "soak_random", mode="random"
    )
    report["soak_policy"] = run_soak(
        cfg_soak, steps=80, device=device, output_dir=out / "soak_policy", mode="policy"
    )
    report["scaling"] = scaling_benchmark(device)
    if not args.skip_convergence:
        report["convergence"] = convergence_smoke(out / "convergence", device)

    report["ok"] = all(
        report[k].get("ok", False)
        for k in ("atlas", "invariants", "soak_random", "soak_policy", "scaling")
        if k in report and not report[k].get("skipped")
    ) and report.get("convergence", {"ok": True}).get("ok", True)

    (out / "validate_launch_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
