"""Warp test: scripted-only opponents track the centerline and hold target speed."""

from __future__ import annotations

import copy
import os
import sys

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


def _build_cfg(*, target_speed: float) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["opponent_strategy"] = "scripted"
    cfg["env"]["opponent_target_speed"] = target_speed
    cfg["env"]["opponent_target_speed_range"] = None
    cfg["env"]["opponent_lateral_offset_m"] = 0.0
    cfg["env"]["opponent_kp_speed"] = 1.0
    cfg["env"]["domain_randomization"]["enabled"] = False
    _pin_deterministic_spawn(cfg, gap_m=25.0)
    cfg["env"]["term_not_moving_time_s"] = 999.0
    cfg["env"]["term_on_collision"] = False
    cfg["env"]["term_oob_max_consecutive"] = 10_000
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


def test_scripted_opponent_tracks_centerline_and_speed(warp_runtime):
    target_speed = 4.0
    num_envs = 4
    control_interval = int(DEFAULT_CONFIG["env"]["control_interval"])
    warmup_steps = 1500
    measure_steps = 2500
    cfg = _build_cfg(target_speed=target_speed)

    torch.manual_seed(0)
    env = _make_env(cfg, num_envs)
    try:
        obs, _ = env.reset()
        assert obs.shape == (num_envs, cfg["obs"]["num_obs"])

        track_len = env.track_length
        prev_s = env.extras["metrics"]["opponent_s"].clone()
        steer_cmds: list[torch.Tensor] = []
        body_speeds: list[torch.Tensor] = []
        boundary_dists: list[torch.Tensor] = []
        forward_progress = torch.zeros((num_envs,), dtype=torch.float32)

        total_steps = warmup_steps + measure_steps
        for step_i in range(total_steps):
            actions = torch.zeros(num_envs, 2, device=env.device)
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
                boundary_dists.append(extras["metrics"]["boundary_dist"].clone())

            s_now = extras["metrics"]["opponent_s"]
            ds = _wrap_ds(s_now - prev_s, track_len)
            forward_progress += torch.clamp(ds, min=0.0)
            prev_s = s_now.clone()

        steer_cmds_t = torch.stack(steer_cmds)
        body_speeds_t = torch.stack(body_speeds)
        boundary_dists_t = torch.stack(boundary_dists)

        sat_frac = float((steer_cmds_t.abs() > 0.99).float().mean().item())
        mean_body_speed = float(body_speeds_t.mean().item())
        min_boundary = float(boundary_dists_t.min().item())
        mean_progress = float(forward_progress.mean().item())

        assert sat_frac < 0.50, (
            f"steer command saturated {sat_frac * 100:.1f}% of measured steps"
        )
        assert abs(mean_body_speed - target_speed) < 1.0, (
            f"body speed {mean_body_speed:.3f} not near target {target_speed}"
        )
        assert min_boundary > 0.0, "opponent left the drivable corridor"
        assert mean_progress >= track_len, (
            f"opponent forward progress {mean_progress:.1f} m "
            f"did not complete one lap ({track_len:.1f} m)"
        )
    finally:
        env.close()
