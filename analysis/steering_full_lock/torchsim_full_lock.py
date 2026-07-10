#!/usr/bin/env python3
"""Torch-sim full-lock skidpad: measure the sim's steady turning radius and the
effective max steering angle it implies, and compare to the real car.

Mirrors the Genesis-era analysis/sim_skidpad.py, but drives the pure-Torch
f1tenth_sim.TorchVehicleSim (the "torch" physics backend) directly -- no Genesis,
no ROS. Full-lock steer (+1.0 -> params.max_steer) is held while throttle is swept
across a batch of envs; the steady-state v/omega radius at each speed is recorded.

Real-car truth comes from analysis/analyze_full_lock.py in the on-car repo, which
circle-fits the LiDAR particle-filter trajectory from the full_lock_{left,right}
rosbags: v/omega radius ~= 0.94 m in both directions at v ~= 2 m/s, i.e. an
effective max wheel angle of ~0.336 rad (19.3 deg). The servo hard-clamps at 0.85
-> 0.33 rad, which is why config.py sets delta_max = 0.33 rad.

Run (from the repo root, with the training venv):
    .venv/bin/python analysis/steering_full_lock/torchsim_full_lock.py
"""
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
from f1tenth_sim import TorchVehicleSim, VehicleParams  # noqa: E402

# Real-car full-lock truth (analyze_full_lock.py, v/omega estimate, both dirs).
REAL_RADIUS_M = 0.94
REAL_V_MS = 2.0
REAL_STEER_RAD = 0.336


def kinematic_radius(delta: float, wheelbase: float) -> float:
    return wheelbase / math.tan(abs(delta)) if abs(delta) > 1e-6 else float("inf")


def run(num_envs: int = 16, warmup_steps: int = 150, record_steps: int = 50):
    env_cfg = DEFAULT_CONFIG["env"]
    params = VehicleParams.from_config(env_cfg)

    sim_dt = float(env_cfg["sim_dt"])
    control_interval = int(env_cfg["control_interval"])
    control_dt = sim_dt * control_interval
    internal = int((env_cfg.get("torch_sim") or {}).get("internal_substeps", 1))

    sim = TorchVehicleSim(
        params, num_envs, sim_dt=sim_dt, control_dt=control_dt,
        internal_substeps=internal,
    )

    # Full lock left (+1.0 -> +max_steer); throttle sweep (speed-mode v_cmd = thr*max_speed).
    throttle = torch.linspace(0.08, 0.45, num_envs)
    steer = torch.ones(num_envs)
    actions = torch.stack([throttle, steer], dim=-1)

    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1)
    sim.reset(None, torch.zeros(num_envs, 3), quat, torch.zeros(num_envs))

    for _ in range(warmup_steps):
        sim.step(actions, n_steps=control_interval)

    v_acc = torch.zeros(num_envs)
    r_acc = torch.zeros(num_envs)
    for _ in range(record_steps):
        sim.step(actions, n_steps=control_interval)
        st = sim.read_state()
        vx = st["base_lin_vel"][:, 0]
        vy = st["base_lin_vel"][:, 1]
        v_acc += torch.hypot(vx, vy)
        r_acc += st["base_ang_vel"][:, 2].abs()

    v = (v_acc / record_steps).numpy()
    omega = (r_acc / record_steps).numpy()
    radius = np.where(omega > 1e-4, v / np.maximum(omega, 1e-9), np.nan)
    alat_g = v * omega / params.gravity
    eff_steer = np.arctan(params.wheelbase / radius)

    return {
        "params": params,
        "throttle": throttle.numpy(),
        "v": v,
        "omega": omega,
        "radius": radius,
        "alat_g": alat_g,
        "eff_steer": eff_steer,
        "kin_radius": kinematic_radius(params.max_steer, params.wheelbase),
    }


def main() -> int:
    res = run()
    p = res["params"]
    kin = res["kin_radius"]

    print(f"torch backend  model={p.model}  throttle_mode={p.throttle_mode}")
    print(f"max_steer = {p.max_steer:.3f} rad ({math.degrees(p.max_steer):.1f} deg)  "
          f"wheelbase = {p.wheelbase:.3f} m  tire_mu = {p.tire_mu:.2f}")
    print(f"kinematic min radius @ full lock = {kin:.3f} m\n")

    print(f"{'thr':>5} {'v':>6} {'omega':>7} {'R':>7} {'a_lat':>7} {'steer_eff':>10}")
    for i in range(len(res["throttle"])):
        print(f"{res['throttle'][i]:5.2f} {res['v'][i]:6.3f} {res['omega'][i]:7.3f} "
              f"{res['radius'][i]:7.3f} {res['alat_g'][i]:6.3f}g "
              f"{math.degrees(res['eff_steer'][i]):9.1f}d")

    # Row nearest the real-car operating point (v ~= 2 m/s).
    i = int(np.nanargmin(np.abs(res["v"] - REAL_V_MS)))
    r_sim = res["radius"][i]
    s_sim = res["eff_steer"][i]

    print(f"\n{'='*56}\n COMPARISON @ v ~= {REAL_V_MS} m/s (real-car operating point)\n{'='*56}")
    print(f"real car   : R = {REAL_RADIUS_M:.3f} m   steer_eff = "
          f"{math.degrees(REAL_STEER_RAD):.1f} deg ({REAL_STEER_RAD:.3f} rad)")
    print(f"torch sim  : R = {r_sim:.3f} m   steer_eff = "
          f"{math.degrees(s_sim):.1f} deg ({s_sim:.3f} rad)   "
          f"(v={res['v'][i]:.2f} m/s, a_lat={res['alat_g'][i]:.2f}g)")
    print(f"gap        : R {(r_sim/REAL_RADIUS_M - 1)*100:+.0f}%   "
          f"steer {math.degrees(s_sim - REAL_STEER_RAD):+.1f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
