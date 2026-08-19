"""Warp env preserves pre-reset privileged critic obs on pure episode timeout."""

from __future__ import annotations

import copy

import torch

from config import DEFAULT_CONFIG
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt
from ppo import pure_timeout_mask


def _timeout_env(*, num_envs=4):
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["domain_randomization"]["enabled"] = False
    cfg["env"]["opponent_strategy"] = "none"
    cfg["env"]["term_on_collision"] = False
    cfg["env"]["term_not_moving_time_s"] = 999.0
    cfg["env"]["term_oob_max_consecutive"] = 10_000
    cfg["env"]["term_heading_error_rad"] = 10.0
    cfg["env"]["reset_speed_min_mps"] = 2.0
    cfg["env"]["reset_speed_max_mps"] = 2.0
    cfg["env"]["episode_length"] = 0.1
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )


def test_pure_timeout_preserves_pre_reset_terminal_critic_obs():
    env = _timeout_env()
    try:
        env.reset()
        actions = torch.zeros(env.num_envs, 2)
        actions[:, 0] = 0.8
        obs, _reward, done, extras = env.step(
            actions, n_steps=env.control_interval, with_sensors=False
        )
        pure = pure_timeout_mask(extras["termination"])
        assert bool(pure.any())
        assert bool(done[pure].all())
        terminal = extras["observations"]["terminal_critic"]
        for row in pure.nonzero(as_tuple=False).reshape(-1).tolist():
            assert not torch.allclose(terminal[row], obs[row])
    finally:
        env.close()


def test_non_timeout_done_does_not_publish_terminal_critic_snapshot():
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["domain_randomization"]["enabled"] = False
    cfg["env"]["opponent_strategy"] = "none"
    cfg["env"]["term_on_collision"] = False
    cfg["env"]["term_not_moving_time_s"] = 999.0
    cfg["env"]["term_oob_max_consecutive"] = 1
    cfg["env"]["term_heading_error_rad"] = 10.0
    cfg["env"]["reset_speed_min_mps"] = 4.0
    cfg["env"]["reset_speed_max_mps"] = 4.0
    cfg["env"]["episode_length"] = 999.0
    env = F1tenthEnv(
        num_envs=2,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )
    try:
        env.reset()
        actions = torch.zeros(2, 2)
        actions[:, 0] = 1.0
        actions[:, 1] = 1.0
        for _ in range(40):
            _obs, _reward, done, extras = env.step(
                actions, n_steps=env.control_interval, with_sensors=False
            )
            pure = pure_timeout_mask(extras["termination"])
            if bool((~pure & done).any()):
                terminal = extras["observations"]["terminal_critic"]
                crashed = (~pure) & done
                for row in crashed.nonzero(as_tuple=False).reshape(-1).tolist():
                    assert torch.all(terminal[row] == 0.0)
                return
        raise AssertionError("expected a non-timeout termination within step budget")
    finally:
        env.close()
