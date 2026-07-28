from __future__ import annotations

import copy

import torch

from config import DEFAULT_CONFIG
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt


def _make_env(num_envs: int) -> F1tenthEnv:
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["domain_randomization"]["enabled"] = False
    cfg["env"]["opponent_strategy"] = None
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )


def test_stationary_reset_mixture_is_seeded_and_bounded(warp_runtime):
    del warp_runtime
    env = _make_env(512)
    try:
        env.reset(seed=23)
        first = env.read_state()["base_lin_vel"][:, 0].clone()
        env.reset(seed=23)
        second = env.read_state()["base_lin_vel"][:, 0]

        assert torch.equal(first, second)
        stationary = first == 0.0
        fraction = float(stationary.float().mean())
        # Canonical Lee path: 10% stationary starts.
        assert 0.05 <= fraction <= 0.15
        moving_min = DEFAULT_CONFIG["env"]["reset_speed_min_mps"]
        assert torch.all(first[~stationary] >= moving_min)
        assert torch.all(first <= DEFAULT_CONFIG["env"]["reset_speed_max_mps"])
    finally:
        env.close()


def test_stationary_startup_grace_precedes_not_moving_window(warp_runtime):
    del warp_runtime
    env = _make_env(1)
    try:
        env.reset(seed=7)
        state = env.read_state()
        pose = state["base_pos"][0, :2].clone()
        quat = state["base_quat"][0]
        yaw = 2.0 * torch.atan2(quat[3], quat[0])
        env.reset_to(pose, yaw, 0.0, seed=7)
        grace_steps = round(
            DEFAULT_CONFIG["env"]["stationary_startup_grace_s"] / env.control_dt
        )
        stopped_steps = round(
            DEFAULT_CONFIG["env"]["term_not_moving_time_s"] / env.control_dt
        )
        for _ in range(grace_steps):
            _, _, done, extras = env.step(
                torch.zeros(1, 2), n_steps=env.control_interval
            )
            assert not bool(done[0])
            assert not bool(extras["termination"]["not_moving"][0])

        fired = False
        for _ in range(stopped_steps + 2):
            _, _, done, extras = env.step(
                torch.zeros(1, 2), n_steps=env.control_interval
            )
            if bool(extras["termination"]["not_moving"][0]):
                fired = True
                assert bool(done[0])
                break
        assert fired
    finally:
        env.close()
