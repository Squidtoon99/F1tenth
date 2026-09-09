"""Warp test: scripted-only opponents track the centerline and hold target speed."""

from __future__ import annotations

import copy
import math
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


def _closed_centerline(env):
    cl = np.asarray(env.centerline, dtype=np.float64)
    wl = np.asarray(env.w_tr_left, dtype=np.float64)
    wr = np.asarray(env.w_tr_right, dtype=np.float64)
    if np.linalg.norm(cl[0] - cl[-1]) > 1e-6:
        cl = np.concatenate([cl, cl[:1]], axis=0)
        wl = np.concatenate([wl, wl[:1]])
        wr = np.concatenate([wr, wr[:1]])
    return cl, wl, wr


def _project_xy(xy: np.ndarray, env):
    cl, wl, wr = _closed_centerline(env)
    start = cl[:-1]
    edge = cl[1:] - start
    edge_sq = np.maximum((edge * edge).sum(-1), 1e-10)
    pos = np.asarray(xy, dtype=np.float64)
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
    return ey, tangent, np.minimum(w_left - ey, w_right + ey)


def _ego_follow_actions(env, target_speed: float, delta_max: float) -> torch.Tensor:
    """Keep ego moving so it is not a parked obstacle for the scripted opponent."""
    ego = env._ego.tensor
    xy = torch.stack([ego["x"], ego["y"]], dim=-1).detach().cpu().numpy()
    yaw = ego["yaw"].detach().cpu().numpy()
    steer = ego["steer"].detach().cpu().numpy()
    vx = ego["vx"].detach().cpu().numpy()
    vy = ego["vy"].detach().cpu().numpy()
    ey, tangent, _ = _project_xy(xy, env)
    actions = torch.zeros((xy.shape[0], 2), dtype=torch.float32, device=env.device)
    for i in range(xy.shape[0]):
        heading = math.atan2(tangent[i, 1], tangent[i, 0])
        heading_err = math.atan2(
            math.sin(yaw[i] - heading), math.cos(yaw[i] - heading)
        )
        vw_x = math.cos(yaw[i]) * vx[i] - math.sin(yaw[i]) * vy[i]
        vw_y = math.sin(yaw[i]) * vx[i] + math.cos(yaw[i]) * vy[i]
        track_speed = vw_x * tangent[i, 0] + vw_y * tangent[i, 1]
        speed_denom = max(abs(track_speed) + 0.5, 0.5)
        desired = -(heading_err + math.atan(ey[i] / speed_denom))
        steer_cmd = (desired - steer[i]) / max(delta_max, 1e-6)
        throttle = target_speed - track_speed
        actions[i, 0] = float(np.clip(throttle, -1.0, 1.0))
        actions[i, 1] = float(np.clip(steer_cmd, -1.0, 1.0))
    return actions


def _opponent_corridor_clearance(env) -> np.ndarray:
    opp = env._opponent.tensor
    xy = torch.stack([opp["x"], opp["y"]], dim=-1).detach().cpu().numpy()
    _, _, clearance = _project_xy(xy, env)
    return clearance


@pytest.mark.parametrize(
    "track,target_speed,gap_m,warmup_steps,measure_steps",
    [
        ("IV_2026_SIM", 4.0, 25.0, 300, 700),
        ("courtyard_2", 3.5, 4.0, 50, 250),
    ],
)
def test_scripted_opponent_tracks_centerline_and_speed(
    warp_runtime, track, target_speed, gap_m, warmup_steps, measure_steps
):
    num_envs = 4
    control_interval = int(DEFAULT_CONFIG["env"]["control_interval"])
    delta_max = float(DEFAULT_CONFIG["env"]["steering_delta_max_rad"])
    car_half_width = 0.5 * float(DEFAULT_CONFIG["env"]["car_width"])
    cfg = _build_cfg(track=track, target_speed=target_speed, gap_m=gap_m)

    torch.manual_seed(0)
    env = _make_env(cfg, num_envs)
    try:
        obs, _ = env.reset()
        assert obs.shape == (num_envs, cfg["obs"]["num_obs"])

        track_len = env.track_length
        prev_s = env.extras["metrics"]["opponent_s"].clone()
        steer_cmds: list[torch.Tensor] = []
        body_speeds: list[torch.Tensor] = []
        clearances: list[np.ndarray] = []
        forward_progress = torch.zeros((num_envs,), dtype=torch.float32)

        total_steps = warmup_steps + measure_steps
        for step_i in range(total_steps):
            actions = _ego_follow_actions(env, target_speed, delta_max)
            obs, reward, done, extras = env.step(
                actions, n_steps=control_interval
            )
            assert torch.isfinite(obs).all()
            assert torch.isfinite(reward).all()

            if step_i >= warmup_steps:
                opp = env._opponent.tensor
                body_v = torch.stack([opp["vx"], opp["vy"]], dim=-1)
                steer_cmds.append(
                    env._env.tensor["opponent_last_action"][:, 1].clone()
                )
                body_speeds.append(torch.linalg.norm(body_v, dim=-1))
                clearances.append(_opponent_corridor_clearance(env))

            s_now = extras["metrics"]["opponent_s"]
            ds = _wrap_ds(s_now - prev_s, track_len)
            forward_progress += torch.clamp(ds, min=0.0)
            prev_s = s_now.clone()

        steer_cmds_t = torch.stack(steer_cmds)
        body_speeds_t = torch.stack(body_speeds)
        clearance_t = np.stack(clearances)

        sat_frac = float((steer_cmds_t.abs() > 0.99).float().mean().item())
        mean_body_speed = float(body_speeds_t.mean().item())
        min_clearance = float(clearance_t.min())
        mean_progress = float(forward_progress.mean().item())

        assert min_clearance > car_half_width, (
            f"{track}: opponent left the drivable corridor "
            f"(min clearance {min_clearance:.3f} m, need > {car_half_width:.3f} m)"
        )
        assert sat_frac < 0.50, (
            f"{track}: steer command saturated {sat_frac * 100:.1f}% of measured steps"
        )
        assert abs(mean_body_speed - target_speed) < 1.0, (
            f"{track}: body speed {mean_body_speed:.3f} not near target {target_speed}"
        )
        assert mean_progress >= track_len, (
            f"{track}: opponent forward progress {mean_progress:.1f} m "
            f"did not complete one lap ({track_len:.1f} m)"
        )
    finally:
        env.close()
