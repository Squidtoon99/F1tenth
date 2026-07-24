"""Real-module tests for distributed opponent spawn and reset launch speeds."""

from __future__ import annotations

import copy

import torch

from conftest import brute_force_frenet
from f1tenth_env import F1tenthEnv
from f1tenth_env import geom as gu
from f1tenth_env import runtime as rt


def _configure_runtime():
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )


def _make_1v1_env(*, num_envs: int, env_overrides: dict | None = None):
    _configure_runtime()
    from config import DEFAULT_CONFIG

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    env_cfg = dict(cfg["env"])
    env_cfg["opponent_strategy"] = "scripted"
    env_cfg.setdefault("launch_strategy", "uniform_jittered")
    env_cfg.setdefault("launch_strategy_data", {"num_cars": num_envs})
    if env_overrides:
        env_cfg.update(env_overrides)
    return F1tenthEnv(
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


def test_distributed_spawn_batch():
    torch.manual_seed(0)
    from config import DEFAULT_CONFIG

    gap_min = float(DEFAULT_CONFIG["env"]["opponent_spawn_gap_min_m"])
    gap_max = float(DEFAULT_CONFIG["env"]["opponent_spawn_gap_max_m"])
    behind_prob = float(DEFAULT_CONFIG["env"]["opponent_spawn_behind_prob"])
    assert gap_min == 3.0
    assert gap_max == 80.0
    assert behind_prob == 0.3
    ahead_gate = float(DEFAULT_CONFIG["obs"]["opp_obs_ahead_m"])
    behind_gate = float(DEFAULT_CONFIG["obs"]["opp_obs_behind_m"])
    num_envs = 1024
    env = _make_1v1_env(
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

    state = env.read_state()
    metrics = env.extras["metrics"]
    track_len = env.track_length
    gap_m = _wrapped_gap_m(
        metrics["s"], metrics["opponent_s"], track_len
    )
    behind_frac = float((gap_m < 0.0).float().mean().item())
    assert 0.15 < behind_frac < 0.45

    gap_mag = gap_m.abs()
    assert bool((gap_mag >= gap_min - 0.5).all())
    assert bool((gap_mag <= gap_max + 0.5).all())
    # Widened range must actually use distances beyond the old 20 m ceiling.
    assert float((gap_mag > 20.0).float().mean().item()) > 0.2

    opponent_projection = brute_force_frenet(
        state["opp_base_pos"][:, :2].cpu().numpy(), env.centerline
    )
    width = torch.as_tensor(
        torch.minimum(
            torch.as_tensor(env.w_tr_left),
            torch.as_tensor(env.w_tr_right),
        ).numpy()[opponent_projection["best_idx"]]
    )
    assert bool(
        (
            torch.as_tensor(opponent_projection["ey"]).abs()
            <= width + 0.05
        ).all()
    )

    # Spawns must not already overlap under the same oriented-box predicate the
    # contact resolver / collision termination use (both cars as full-size boxes).
    from f1tenth_env.terminations import collision_mask

    car_len = float(env.env_cfg.get("car_length", 0.568))
    car_wid = float(env.env_cfg.get("car_width", 0.296))
    ego_yaw = gu.quat_to_xyz(state["base_quat"], rpy=True, degrees=False)[:, 2]
    opp_yaw = gu.quat_to_xyz(
        state["opp_base_quat"], rpy=True, degrees=False
    )[:, 2]
    overlap = collision_mask(
        state["base_pos"][:, :2],
        state["opp_base_pos"][:, :2],
        ego_yaw,
        opp_yaw,
        car_len,
        car_wid,
    )
    assert not bool(overlap.any())

    # Both initial visibility classes under the 40 m ahead / 20 m behind gate.
    initially_visible = (gap_m <= ahead_gate) & (gap_m >= -behind_gate)
    vis_frac = float(initially_visible.float().mean().item())
    assert 0.2 < vis_frac < 0.8
    assert bool(initially_visible.any())
    assert bool((~initially_visible).any())

    # Wraparound: a behind spawn near s≈0 places the opponent near track end.
    near_start = metrics["s"] < 5.0
    if bool(near_start.any()):
        behind_near_start = near_start & (gap_m < 0.0)
        if bool(behind_near_start.any()):
            opp_s = metrics["opponent_s"][behind_near_start]
            assert bool((opp_s > track_len - gap_max - 1.0).any())


def _heading_speed(vel_xy, yaw):
    speed = torch.linalg.norm(vel_xy, dim=-1)
    hx = torch.cos(yaw)
    hy = torch.sin(yaw)
    align = (vel_xy[:, 0] * hx + vel_xy[:, 1] * hy) / speed.clamp_min(1e-6)
    return speed, align


def test_reset_launch_speeds_ego_and_opponent():
    ego_speed = 3.0
    opp_speed = 2.5
    env = _make_1v1_env(
        num_envs=8,
        env_overrides={
            "opponent_spawn_gap_min_m": 7.0,
            "opponent_spawn_gap_max_m": 7.0,
            "opponent_spawn_behind_prob": 0.0,
            "opponent_spawn_lateral_independent": False,
            "reset_speed_min_mps": ego_speed,
            "reset_speed_max_mps": ego_speed,
            "reset_stationary_probability": 0.0,
            "opponent_reset_speed_min_mps": opp_speed,
            "opponent_reset_speed_max_mps": opp_speed,
            "term_on_collision": False,
        },
    )
    env.reset()

    state = env.read_state()
    ego_yaw = gu.quat_to_xyz(state["base_quat"], rpy=True, degrees=False)[:, 2]
    opp_yaw = gu.quat_to_xyz(
        state["opp_base_quat"], rpy=True, degrees=False
    )[:, 2]

    ego_spd, ego_align = _heading_speed(
        state["base_vel_world"][:, :2], ego_yaw
    )
    opp_spd, opp_align = _heading_speed(
        state["opp_vel_world"][:, :2], opp_yaw
    )

    assert torch.allclose(ego_spd, torch.full_like(ego_spd, ego_speed), atol=0.05)
    assert torch.allclose(opp_spd, torch.full_like(opp_spd, opp_speed), atol=0.05)
    assert bool((ego_align > 0.99).all())
    assert bool((opp_align > 0.99).all())
