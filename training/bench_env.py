"""Compare end-to-end env.step throughput: Genesis vs Torch backend.

Same env / obs / reward / termination pipeline; only ``physics_backend`` differs,
so the ratio isolates the physics engine. Run once per backend (Genesis needs its
own process because gs.init is global):

    python bench_env.py --physics torch   --envs 64 256 1024 4096
    python bench_env.py --physics genesis --envs 64 256 1024
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

# py-cpuinfo can't probe the CPU in some sandboxes; keep gs.init happy.
try:
    import cpuinfo

    _orig = cpuinfo.get_cpu_info

    def _patched():
        d = _orig() or {}
        d.setdefault("brand_raw", "Sandbox CPU")
        d.setdefault("vendor_id_raw", "sandbox")
        return d

    cpuinfo.get_cpu_info = _patched
except Exception:
    pass

import genesis as gs  # noqa: E402
import torch  # noqa: E402

from f1tenth_env import runtime as rt  # noqa: E402
from standalone_trainer import (  # noqa: E402
    DEFAULT_CONFIG,
    episode_length_for_track,
    select_device,
    _maybe_patch_headless_rasterizer,
)


def _cfg(track: str, physics: str) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["physics_backend"] = physics
    cfg["env"]["episode_length"] = episode_length_for_track(
        track=track, workspace_dir=str(Path(__file__).resolve().parent),
        ref_lap_speed_mps=3.5, lap_multiplier=3.0,
    )
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--physics", choices=["genesis", "torch"], required=True)
    ap.add_argument("--envs", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--track", type=str, default=DEFAULT_CONFIG["env"]["track"])
    args = ap.parse_args()

    device = select_device("cpu")
    if args.physics == "genesis":
        _maybe_patch_headless_rasterizer()
        gs.init(backend=gs.cpu, precision="32", performance_mode=True)
        rt.configure(float_dtype=gs.tc_float, int_dtype=gs.tc_int, dev=device,
                     eps=gs.EPS)
    else:
        rt.configure(float_dtype=torch.float32, int_dtype=torch.int32, dev=device,
                     eps=1e-12)

    from f1tenth_env.env import F1tenthEnv

    cfg = _cfg(args.track, args.physics)
    ci = int(cfg["env"]["control_interval"])
    print(f"physics={args.physics} control_interval={ci} steps={args.steps}")
    print(f"{'n_envs':>8} {'ctrl/s':>12} {'env-steps/s':>14} {'substep/s':>14}")
    for n in args.envs:
        env = F1tenthEnv(
            num_envs=n,
            env_cfg={"launch_strategy": "uniform_jittered",
                     "launch_strategy_data": {"num_cars": n}, **cfg["env"]},
            obs_cfg=cfg["obs"], reward_cfg=cfg["reward"],
            show_viewer=False, enable_recording=False,
        )
        env.reset()
        a = torch.zeros(n, 2, device=device, dtype=rt.tc_float)
        a[:, 0] = 0.5
        for _ in range(args.warmup):
            env.step(a, n_steps=ci)
        t0 = time.perf_counter()
        for _ in range(args.steps):
            env.step(a, n_steps=ci)
        dt = time.perf_counter() - t0
        ctrl = n * args.steps / dt
        print(f"{n:>8} {ctrl:>12,.0f} {ctrl:>14,.0f} {ctrl * ci:>14,.0f}")
        env.close()


if __name__ == "__main__":
    main()
