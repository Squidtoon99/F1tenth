#!/usr/bin/env python3
"""Measure the Warp simulator's steady full-lock turning radius."""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

_TRAINING = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "training")
)
sys.path.insert(0, _TRAINING)

from config import DEFAULT_CONFIG  # noqa: E402
from f1tenth_sim import VehicleParams, WarpVehicleSim  # noqa: E402

REAL_RADIUS_M = 0.94
REAL_V_MS = 2.0
REAL_STEER_RAD = 0.336


def run(num_envs: int = 16, warmup_steps: int = 150, record_steps: int = 50):
    env_cfg = DEFAULT_CONFIG["env"]
    params = VehicleParams.from_config(env_cfg)
    sim_dt = float(env_cfg["sim_dt"])
    control_interval = int(env_cfg["control_interval"])
    sim = WarpVehicleSim(
        params,
        num_envs,
        device="cpu",
        sim_dt=sim_dt,
        control_dt=sim_dt * control_interval,
    )

    throttle = torch.linspace(0.08, 0.45, num_envs)
    actions = torch.stack([throttle, torch.ones(num_envs)], dim=-1)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1)
    sim.reset(None, torch.zeros(num_envs, 3), quat, torch.zeros(num_envs))

    for _ in range(warmup_steps):
        sim.step(actions, n_steps=control_interval)

    speed_sum = torch.zeros(num_envs)
    yaw_rate_sum = torch.zeros(num_envs)
    for _ in range(record_steps):
        sim.step(actions, n_steps=control_interval)
        state = sim.read_state()
        speed_sum += torch.hypot(
            state["base_lin_vel"][:, 0], state["base_lin_vel"][:, 1]
        )
        yaw_rate_sum += state["base_ang_vel"][:, 2].abs()

    speed = (speed_sum / record_steps).numpy()
    yaw_rate = (yaw_rate_sum / record_steps).numpy()
    radius = np.where(
        yaw_rate > 1.0e-4, speed / np.maximum(yaw_rate, 1.0e-9), np.nan
    )
    return {
        "params": params,
        "throttle": throttle.numpy(),
        "v": speed,
        "omega": yaw_rate,
        "radius": radius,
        "alat_g": speed * yaw_rate / params.gravity,
        "eff_steer": np.arctan(params.wheelbase / radius),
        "kin_radius": params.wheelbase / math.tan(abs(params.max_steer)),
    }


def main() -> int:
    result = run()
    params = result["params"]
    print("warp simulator  model=dynamic  longitudinal=force")
    print(
        f"max_steer = {params.max_steer:.3f} rad "
        f"({math.degrees(params.max_steer):.1f} deg)  "
        f"wheelbase = {params.wheelbase:.3f} m  "
        f"tire_mu = {params.tire_mu:.2f}"
    )
    print(
        f"kinematic min radius @ full lock = {result['kin_radius']:.3f} m\n"
    )
    print(f"{'thr':>5} {'v':>6} {'omega':>7} {'R':>7} {'a_lat':>7} {'steer_eff':>10}")
    for i in range(len(result["throttle"])):
        print(
            f"{result['throttle'][i]:5.2f} {result['v'][i]:6.3f} "
            f"{result['omega'][i]:7.3f} {result['radius'][i]:7.3f} "
            f"{result['alat_g'][i]:6.3f}g "
            f"{math.degrees(result['eff_steer'][i]):9.1f}d"
        )

    index = int(np.nanargmin(np.abs(result["v"] - REAL_V_MS)))
    radius = result["radius"][index]
    steering = result["eff_steer"][index]
    print(f"\ncomparison near {REAL_V_MS} m/s")
    print(
        f"real car: R={REAL_RADIUS_M:.3f} m, "
        f"steer={math.degrees(REAL_STEER_RAD):.1f} deg"
    )
    print(
        f"warp sim: R={radius:.3f} m, "
        f"steer={math.degrees(steering):.1f} deg"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
