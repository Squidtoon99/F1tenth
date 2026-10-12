"""Warp test: scripted-only opponents track the centerline and hold target speed."""

from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import DEFAULT_CONFIG  # noqa: E402
from f1tenth_env import F1tenthEnv  # noqa: E402


def _pin_deterministic_spawn(cfg: dict, *, gap_m: float) -> None:
    cfg["env"]["opponent_spawn_gap_min_m"] = gap_m
    cfg["env"]["opponent_spawn_gap_max_m"] = gap_m
    cfg["env"]["opponent_spawn_behind_prob"] = 0.0
    cfg["env"]["opponent_spawn_lateral_independent"] = False
    cfg["env"]["opponent_reset_speed_min_mps"] = 0.0
    cfg["env"]["opponent_reset_speed_max_mps"] = 0.0
    cfg["env"]["reset_speed_min_mps"] = 0.0
    cfg["env"]["reset_speed_max_mps"] = 0.0
    cfg["env"]["reset_stationary_probability"] = 0.0
    cfg["env"]["reset_spawn_yaw_jitter_rad"] = 0.0
    cfg["env"]["reset_spawn_margin_m"] = 5.0


def _build_cfg(*, track: str, target_speed: float, gap_m: float) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["track"] = track
    cfg["env"]["opponent_strategy"] = "scripted"
    cfg["env"]["opponent_target_speed"] = target_speed
    cfg["env"]["opponent_target_speed_range"] = None
    cfg["env"]["opponent_lateral_offset_m"] = 0.0
    cfg["env"]["opponent_kp_speed"] = 1.0
    cfg["env"]["domain_randomization"]["enabled"] = False
    _pin_deterministic_spawn(cfg, gap_m=gap_m)
    cfg["env"]["term_not_moving_time_s"] = 999.0
    cfg["env"]["term_on_collision"] = False
    cfg["env"]["term_heading_error_rad"] = 10.0
    cfg["env"]["episode_length"] = 999.0
    cfg["env"]["car_length"] = 0.0
    cfg["env"]["car_width"] = 0.0
    return cfg


def _make_env(cfg: dict, num_envs: int) -> F1tenthEnv:
    env_cfg = {
        "launch_strategy": "uniform_jittered",
        "launch_strategy_data": {"num_cars": num_envs},
        **cfg["env"],
    }
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )


def _wrap_ds(ds: torch.Tensor, length: float) -> torch.Tensor:
    half_l = 0.5 * length
    ds = torch.where(ds > half_l, ds - length, ds)
    ds = torch.where(ds < -half_l, ds + length, ds)
    return ds


def _opponent_corridor_clearance(env) -> np.ndarray:
    cl = np.asarray(env.centerline, dtype=np.float64)
    wl = np.asarray(env.w_tr_left, dtype=np.float64)
    wr = np.asarray(env.w_tr_right, dtype=np.float64)
    if np.linalg.norm(cl[0] - cl[-1]) > 1e-6:
        cl = np.concatenate([cl, cl[:1]], axis=0)
        wl = np.concatenate([wl, wl[:1]])
        wr = np.concatenate([wr, wr[:1]])
    start = cl[:-1]
    edge = cl[1:] - start
    edge_sq = np.maximum((edge * edge).sum(-1), 1e-10)
    opp = env._opponent.tensor
    pos = torch.stack([opp["x"], opp["y"]], dim=-1).detach().cpu().numpy()
    pos = np.asarray(pos, dtype=np.float64)
    delta = pos[:, None, :] - start[None, :, :]
    t = np.clip((delta * edge[None, :, :]).sum(-1) / edge_sq[None, :], 0.0, 1.0)
    proj = start[None, :, :] + t[..., None] * edge[None, :, :]
    dist2 = ((pos[:, None, :] - proj) ** 2).sum(-1)
    idx = dist2.argmin(axis=1)
    rows = np.arange(pos.shape[0])
    alpha = t[rows, idx]
    best_edge = edge[idx]
    tangent = best_edge / np.maximum(
        np.linalg.norm(best_edge, axis=-1, keepdims=True), 1e-8
    )
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
    ey = ((pos - proj[rows, idx]) * normal).sum(-1)
    w_left = (1.0 - alpha) * wl[idx] + alpha * wl[(idx + 1) % len(wl)]
    w_right = (1.0 - alpha) * wr[idx] + alpha * wr[(idx + 1) % len(wr)]
    return np.minimum(w_left - ey, w_right + ey)


@pytest.mark.parametrize(
    "track,target_speed,gap_m,warmup_steps,measure_steps,lateral_offset",
    [
        ("IV_2026_SIM", 3.0, 25.0, 600, 1400, 0.0),
        ("courtyard_2", 3.5, 4.0, 100, 500, 0.0),
        ("courtyard_2", 3.5, 4.0, 100, 500, 0.15),
    ],
)
def test_scripted_opponent_tracks_centerline_and_speed(
    warp_runtime,
    track,
    target_speed,
    gap_m,
    warmup_steps,
    measure_steps,
    lateral_offset,
):
    num_envs = 4
    control_interval = int(DEFAULT_CONFIG["env"]["control_interval"])
    cfg = _build_cfg(track=track, target_speed=target_speed, gap_m=gap_m)
    cfg["env"]["opponent_lateral_offset_m"] = lateral_offset

    torch.manual_seed(0)
    env = _make_env(cfg, num_envs)
    try:
        obs, _ = env.reset()
        assert obs.shape == (num_envs, cfg["obs"]["num_obs"])

        track_len = env.track_length
        control_dt = control_interval * float(DEFAULT_CONFIG["env"]["sim_dt"])
        prev_s = env.extras["metrics"]["opponent_s"].clone()
        body_speeds: list[torch.Tensor] = []
        clearances: list[np.ndarray] = []
        forward_progress = torch.zeros((num_envs,), dtype=torch.float32)

        total_steps = warmup_steps + measure_steps
        for step_i in range(total_steps):
            actions = torch.zeros(num_envs, 2, device=env.device)
            obs, reward, done, extras = env.step(
                actions, n_steps=control_interval
            )
            assert torch.isfinite(obs).all()
            assert torch.isfinite(reward).all()

            s_now = extras["metrics"]["opponent_s"]
            ds = _wrap_ds(s_now - prev_s, track_len)
            prev_s = s_now.clone()
            if step_i < warmup_steps:
                continue
            opp = env._opponent.tensor
            body_v = torch.stack([opp["vx"], opp["vy"]], dim=-1)
            body_speeds.append(torch.linalg.norm(body_v, dim=-1))
            clearances.append(_opponent_corridor_clearance(env))
            forward_progress += torch.clamp(ds, min=0.0)

        body_speeds_t = torch.stack(body_speeds)
        clearance_t = np.stack(clearances)

        mean_body_speed = float(body_speeds_t.mean().item())
        min_clearance = float(clearance_t.min())
        mean_progress = float(forward_progress.mean().item())
        expected_m = target_speed * measure_steps * control_dt

        label = f"{track} offset={lateral_offset}"
        assert min_clearance > 0.0, (
            f"{label}: opponent left the drivable corridor "
            f"(min clearance {min_clearance:.3f} m)"
        )
        assert mean_body_speed > 1.0, (
            f"{label}: body speed {mean_body_speed:.3f} m/s is too low"
        )
        assert mean_body_speed < target_speed + 1.5, (
            f"{label}: body speed {mean_body_speed:.3f} overshot "
            f"target {target_speed}"
        )
        assert mean_progress >= 0.5 * expected_m, (
            f"{label}: forward progress {mean_progress:.1f} m is below "
            f"half of target-speed distance {expected_m:.1f} m"
        )
        if expected_m >= 1.25 * track_len:
            assert mean_progress >= track_len, (
                f"{label}: opponent forward progress {mean_progress:.1f} m "
                f"did not complete one lap ({track_len:.1f} m)"
            )
    finally:
        env.close()
