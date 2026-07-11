"""Real-module tests for distributed opponent spawn and reset launch speeds."""

from __future__ import annotations

import copy

import numpy as np
import torch

from f1tenth_env import geom as gu
from f1tenth_env import runtime as rt


def _configure_runtime():
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )


def _fake_track_state(track, workspace_dir, device):
    n = 400
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False).astype(np.float32)
    radius = 8.0
    cl = np.stack(
        [radius * np.cos(th), radius * np.sin(th)], axis=-1
    ).astype(np.float32)
    w = np.full(n, 1.5, np.float32)
    return {
        "centerline": cl,
        "w_tr_left": w,
        "w_tr_right": w,
        "w_tr_left_torch": torch.tensor(w, device=device),
        "w_tr_right_torch": torch.tensor(w, device=device),
        "track_geom_cache": {},
    }


def _make_1v1_env(monkeypatch, *, num_envs: int, env_overrides: dict | None = None):
    _configure_runtime()
    import f1tenth_env.utils as U
    import f1tenth_env.env as E
    from config import DEFAULT_CONFIG

    monkeypatch.setattr(U, "load_track_state", _fake_track_state, raising=True)
    monkeypatch.setattr(E, "load_track_state", _fake_track_state, raising=True)

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    env_cfg = dict(cfg["env"])
    env_cfg["opponent_strategy"] = "scripted"
    env_cfg.setdefault("launch_strategy", "uniform_jittered")
    env_cfg.setdefault("launch_strategy_data", {"num_cars": num_envs})
    if env_overrides:
        env_cfg.update(env_overrides)
    return E.F1tenthEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )


def _wrapped_gap_m(s_self, s_other, track_len):
    gap = s_other - s_self
    half = 0.5 * track_len
    gap = torch.where(gap > half, gap - track_len, gap)
    gap = torch.where(gap < -half, gap + track_len, gap)
    return gap


def test_distributed_spawn_batch(monkeypatch):
    torch.manual_seed(0)
    gap_min = 3.0
    gap_max = 20.0
    behind_prob = 0.3
    num_envs = 1024
    env = _make_1v1_env(
        monkeypatch,
        num_envs=num_envs,
        env_overrides={
            "opponent_spawn_gap_min_m": gap_min,
            "opponent_spawn_gap_max_m": gap_max,
            "opponent_spawn_behind_prob": behind_prob,
            "opponent_spawn_lateral_independent": True,
            "opponent_reset_speed_min_mps": 0.0,
            "opponent_reset_speed_max_mps": 0.0,
            "reset_speed_min_mps": 0.0,
            "reset_speed_max_mps": 0.0,
            "term_on_collision": False,
        },
    )
    env.reset()

    ego_ss = env._get_step_state()
    opp_ss = env._opponent_step_state(env.opp_base_pos)
    track_len = ego_ss["frenet"]["L"]
    gap_m = _wrapped_gap_m(
        ego_ss["frenet"]["s"], opp_ss["frenet"]["s"], track_len
    )
    behind_frac = float((gap_m < 0.0).float().mean().item())
    assert 0.15 < behind_frac < 0.45

    gap_mag = gap_m.abs()
    assert bool((gap_mag >= gap_min - 0.5).all())
    assert bool((gap_mag <= gap_max + 0.5).all())

    opp_ey = opp_ss["boundary"]["ey"].abs()
    opp_w = torch.minimum(
        env.w_tr_left_torch[
            env._closest_centerline_indices(env.opp_base_pos[:, :2])
        ],
        env.w_tr_right_torch[
            env._closest_centerline_indices(env.opp_base_pos[:, :2])
        ],
    )
    assert bool((opp_ey <= opp_w + 0.05).all())

    # Spawns must not already overlap under the same oriented-box predicate the
    # contact resolver / collision termination use (both cars as full-size boxes).
    from f1tenth_env.terminations import collision_mask

    car_len = float(env.env_cfg.get("car_length", 0.568))
    car_wid = float(env.env_cfg.get("car_width", 0.296))
    ego_yaw = gu.quat_to_xyz(env.base_quat, rpy=True, degrees=False)[:, 2]
    opp_yaw = gu.quat_to_xyz(env.opp_base_quat, rpy=True, degrees=False)[:, 2]
    overlap = collision_mask(
        env.base_pos[:, :2],
        env.opp_base_pos[:, :2],
        ego_yaw,
        opp_yaw,
        car_len,
        car_wid,
    )
    assert not bool(overlap.any())


def _heading_speed(vel_xy, yaw):
    speed = torch.linalg.norm(vel_xy, dim=-1)
    hx = torch.cos(yaw)
    hy = torch.sin(yaw)
    align = (vel_xy[:, 0] * hx + vel_xy[:, 1] * hy) / speed.clamp_min(1e-6)
    return speed, align


def test_reset_launch_speeds_ego_and_opponent(monkeypatch):
    ego_speed = 3.0
    opp_speed = 2.5
    env = _make_1v1_env(
        monkeypatch,
        num_envs=8,
        env_overrides={
            "opponent_spawn_gap_min_m": 7.0,
            "opponent_spawn_gap_max_m": 7.0,
            "opponent_spawn_behind_prob": 0.0,
            "opponent_spawn_lateral_independent": False,
            "reset_speed_min_mps": ego_speed,
            "reset_speed_max_mps": ego_speed,
            "opponent_reset_speed_min_mps": opp_speed,
            "opponent_reset_speed_max_mps": opp_speed,
            "term_on_collision": False,
        },
    )
    env.reset()

    ego_yaw = gu.quat_to_xyz(env.base_quat, rpy=True, degrees=False)[:, 2]
    opp_yaw = gu.quat_to_xyz(env.opp_base_quat, rpy=True, degrees=False)[:, 2]

    ego_spd, ego_align = _heading_speed(env.base_vel_world[:, :2], ego_yaw)
    opp_spd, opp_align = _heading_speed(env.opp_vel_world[:, :2], opp_yaw)

    assert torch.allclose(ego_spd, torch.full_like(ego_spd, ego_speed), atol=0.05)
    assert torch.allclose(opp_spd, torch.full_like(opp_spd, opp_speed), atol=0.05)
    assert bool((ego_align > 0.99).all())
    assert bool((opp_align > 0.99).all())

    ego_vx = env.backend.sim.s["vx"]
    opp_vx = env.backend.opp_sim.s["vx"]
    assert torch.allclose(ego_vx.abs(), torch.full_like(ego_vx, ego_speed), atol=0.05)
    assert torch.allclose(opp_vx.abs(), torch.full_like(opp_vx, opp_speed), atol=0.05)
