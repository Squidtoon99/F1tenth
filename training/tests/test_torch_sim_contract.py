"""Observation-contract + throughput tests for the Torch backend.

`test_sim_state_feeds_obs_contract` proves that :class:`TorchVehicleSim` state
plugs straight into the deploy-side observation builder (`obs_core`, the same code
that runs on the physical car), producing a well-formed 380-dim vector whose
velocity/accel/action slots equal the sim's readback -- i.e. the sim honours the
observation contract. `test_throughput` is a smoke benchmark of batched steps/sec.
"""

from __future__ import annotations

import importlib.util
import os
import time

import numpy as np
import torch

from f1tenth_sim import TorchVehicleSim, VehicleParams

_OBS_CORE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "racing_rl",
        "f1tenth_rl_agent", "f1tenth_rl_agent", "obs_core.py",
    )
)


def _load_obs_core():
    spec = importlib.util.spec_from_file_location("obs_core_contract", _OBS_CORE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _circle_track(radius=8.0, n=400):
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False).astype(np.float32)
    cl = np.stack([radius * np.cos(th), radius * np.sin(th)], axis=-1)
    w = np.full(n, 1.5, np.float32)
    return cl.astype(np.float32), w, w


def _obs_cfg():
    return {
        "num_obs": 380,
        "obs_scales": {"lin_vel": 1.0, "ang_vel": 1.0, "lin_acc": 1.0},
        "clip_obs": 0.0,
        "contact_margin_m": 0.08,
        "future_track_num_points": 60,
        "future_track_horizon_s": 6.0,
        "future_track_min_lookahead_m": 5.0,
        "future_track_width": 2.2,
        "enable_opponent_obs": False,
    }


def _tyre_slip(sim, wheel_radius, slip_eps=0.1):
    ws = sim.read_wheel_state()
    v_long = ws["motion_link_vel"][:, :, 0]
    v_lat = ws["motion_link_vel"][:, :, 1]
    spin = ws["dof_vel"]
    slip_angle = torch.atan2(v_lat, v_long.abs().clamp_min(slip_eps))
    wheel_speed = wheel_radius * spin
    denom = torch.maximum(wheel_speed.abs(), v_long.abs()).clamp_min(slip_eps)
    slip_ratio = (wheel_speed - v_long) / denom
    return torch.cat([slip_ratio, slip_angle], dim=-1)


def test_sim_state_feeds_obs_contract():
    obs_core = _load_obs_core()
    cl, wl, wr = _circle_track()
    cfg = _obs_cfg()
    builder = obs_core.ObservationBuilder(cl, wl, wr, obs_cfg=cfg)

    n = 6
    p = VehicleParams.from_config({"tire_friction": 0.9, "max_speed": 8.0})
    sim = TorchVehicleSim(p, n, sim_dt=0.005, control_dt=0.1)
    pos = torch.zeros(n, 3)
    pos[:, 0] = 8.0
    quat = torch.zeros(n, 4)
    quat[:, 0] = 1.0
    sim.reset(None, pos, quat, torch.full((n,), 2.0))

    last_actions = torch.zeros(n, 2)
    for _ in range(10):
        last_actions = torch.tensor([[0.5, 0.2]]).repeat(n, 1)
        sim.step(last_actions, n_steps=20)

    st = sim.read_state()
    tyre_slip = _tyre_slip(sim, p.wheel_radius)
    obs = builder.build(
        base_lin_vel=st["base_lin_vel"],
        base_ang_vel=st["base_ang_vel"],
        base_lin_acc=st["base_lin_acc"],
        last_actions=last_actions,
        base_pos=st["base_pos"],
        base_quat_wxyz=st["base_quat"],
        tyre_slip=tyre_slip,
    )

    assert obs.shape == (n, 380)
    assert torch.isfinite(obs).all()
    # Contract slot wiring: leading dims are exactly the sim readback.
    assert torch.allclose(obs[:, :2], st["base_lin_vel"][:, :2], atol=1e-5)
    assert torch.allclose(obs[:, 2], st["base_ang_vel"][:, 2], atol=1e-5)
    assert torch.allclose(obs[:, 3:5], st["base_lin_acc"][:, :2], atol=1e-5)
    assert torch.allclose(obs[:, 5:7], last_actions, atol=1e-5)
    # tyre-slip block is the final 8 dims.
    assert torch.allclose(obs[:, -8:], tyre_slip, atol=1e-5)


def test_throughput():
    n = 512
    p = VehicleParams.from_config({"tire_friction": 0.9, "max_speed": 8.0})
    sim = TorchVehicleSim(p, n, sim_dt=0.005, control_dt=0.1)
    quat = torch.zeros(n, 4)
    quat[:, 0] = 1.0
    sim.reset(None, torch.zeros(n, 3), quat, torch.zeros(n))
    a = torch.zeros(n, 2)
    a[:, 0] = 0.5
    a[:, 1] = 0.1

    for _ in range(3):  # warmup
        sim.step(a, n_steps=20)

    control_steps = 50
    t0 = time.perf_counter()
    for _ in range(control_steps):
        sim.step(a, n_steps=20)
    dt = time.perf_counter() - t0

    env_control_steps_per_s = n * control_steps / dt
    sim_substeps_per_s = env_control_steps_per_s * 20
    print(
        f"\nthroughput: {env_control_steps_per_s:,.0f} env-control-steps/s, "
        f"{sim_substeps_per_s:,.0f} substeps/s (n={n}, cpu)"
    )
    st = sim.read_state()
    assert torch.isfinite(st["base_pos"]).all()
    # Very loose floor just to catch pathological slowdowns on CI CPU.
    assert env_control_steps_per_s > 1000.0
