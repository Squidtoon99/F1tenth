"""Seeded random lateral + yaw spawn jitter (ego + opponent).

The Warp migration spawned every car exactly on the track tangent (ego) and on the
centerline (opponent). These tests lock in the restored seeded spawn jitter: the
ego heading is perturbed off the tangent, the opponent samples an off-centerline
lateral start, and everything stays fully seed-deterministic.
"""

from __future__ import annotations

import copy

import torch

from config import DEFAULT_CONFIG
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt


def _make_env(num_envs=64, opponent=False, **env_over):
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["domain_randomization"]["enabled"] = False
    cfg["env"]["reset_speed_min_mps"] = 0.0
    cfg["env"]["reset_speed_max_mps"] = 0.0
    if opponent:
        cfg["env"]["opponent_strategy"] = "scripted"
    cfg["env"].update(env_over)
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )


def _yaw(state):
    quat = state["base_quat"]
    return 2.0 * torch.atan2(quat[:, 3], quat[:, 0])


def test_ego_yaw_jitter_perturbs_tangent_without_moving_position():
    # random_range always consumes one RNG draw (even when the range is zero), so
    # segment/alpha/lateral are identical between the two configs at the same seed;
    # only the added heading offset differs.
    jitter = 0.12
    baseline = _make_env(reset_spawn_yaw_jitter_rad=0.0)
    perturbed = _make_env(reset_spawn_yaw_jitter_rad=jitter)
    try:
        baseline.reset(seed=17)
        perturbed.reset(seed=17)
        base_state = baseline.read_state()
        pert_state = perturbed.read_state()

        # Position is unaffected by the heading jitter at a fixed seed.
        assert torch.allclose(base_state["base_pos"], pert_state["base_pos"], atol=1e-5)

        dyaw = _yaw(pert_state) - _yaw(base_state)
        dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))
        assert dyaw.abs().max() <= jitter + 1e-5
        assert dyaw.abs().max() > 1e-3           # jitter actually applied
        assert dyaw.std() > 1e-3                  # varies across envs
    finally:
        baseline.close()
        perturbed.close()


def test_spawn_jitter_is_seed_deterministic():
    env = _make_env(reset_spawn_yaw_jitter_rad=0.15, opponent=True,
                    opponent_spawn_lateral_independent=True)
    try:
        env.reset(seed=9)
        first = env.read_state()
        first_pos = first["base_pos"].clone()
        first_yaw = _yaw(first).clone()
        first_opp = first["opp_base_pos"].clone()
        env.reset(seed=9)
        second = env.read_state()
        assert torch.equal(first_pos, second["base_pos"])
        assert torch.equal(first_yaw, _yaw(second))
        assert torch.equal(first_opp, second["opp_base_pos"])
    finally:
        env.close()


def test_opponent_lateral_spawn_moves_off_centerline():
    on = _make_env(opponent=True, opponent_spawn_lateral_independent=True,
                   reset_spawn_yaw_jitter_rad=0.0)
    off = _make_env(opponent=True, opponent_spawn_lateral_independent=False,
                    reset_spawn_yaw_jitter_rad=0.0)
    try:
        on.reset(seed=3)
        off.reset(seed=3)
        # With an independent lateral spawn the opponent no longer sits where the
        # centerline-only spawn placed it (for at least some envs).
        on_opp = on.read_state()["opp_base_pos"]
        off_opp = off.read_state()["opp_base_pos"]
        assert not torch.allclose(on_opp, off_opp, atol=1e-3)
        assert torch.isfinite(on_opp).all()
    finally:
        on.close()
        off.close()
