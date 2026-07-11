"""Benchmark end-to-end TorchSim environment throughput."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch

from config import DEFAULT_CONFIG
from f1tenth_env import runtime as rt
from f1tenth_env.env import F1tenthEnv
from f1tenth_env.utils import episode_length_for_track
from standalone_trainer import select_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--track", type=str, default=DEFAULT_CONFIG["env"]["track"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--precision", choices=["32", "64"], default="32")
    args = parser.parse_args()

    device = select_device(args.device)
    dtype = torch.float64 if args.precision == "64" else torch.float32
    rt.configure(
        float_dtype=dtype,
        int_dtype=torch.int32,
        dev=device,
        eps=1e-12,
    )
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["episode_length"] = episode_length_for_track(
        track=args.track,
        workspace_dir=str(Path(__file__).resolve().parent),
        ref_lap_speed_mps=3.5,
        lap_multiplier=3.0,
    )
    control_interval = int(cfg["env"]["control_interval"])
    print(
        f"device={device} precision={args.precision} "
        f"control_interval={control_interval} steps={args.steps}"
    )
    print(f"{'envs':>8} {'ticks/s':>12} {'transitions/s':>16} {'substeps/s':>14}")

    for num_envs in args.envs:
        env = F1tenthEnv(
            num_envs=num_envs,
            env_cfg={
                "launch_strategy": "uniform_jittered",
                "launch_strategy_data": {"num_cars": num_envs},
                **cfg["env"],
            },
            obs_cfg=cfg["obs"],
            reward_cfg=cfg["reward"],
        )
        actions = torch.zeros(num_envs, 2, device=device, dtype=rt.tc_float)
        actions[:, 0] = 0.5
        for _ in range(args.warmup):
            env.step(actions, n_steps=control_interval)
        start = time.perf_counter()
        for _ in range(args.steps):
            env.step(actions, n_steps=control_interval)
        elapsed = time.perf_counter() - start
        ticks_per_sec = args.steps / elapsed
        transitions_per_sec = num_envs * ticks_per_sec
        print(
            f"{num_envs:>8} {ticks_per_sec:>12,.1f} "
            f"{transitions_per_sec:>16,.0f} "
            f"{transitions_per_sec * control_interval:>14,.0f}"
        )
        env.close()


if __name__ == "__main__":
    main()
