"""Deterministic fixed-start reset API (reset_to).

Restores reproducible fixed-start launches (the ``fixed``/``eval_launch`` intent):
``reset_to`` places cars at explicit world poses, bypassing the random spawn, and
identical fixed starts produce byte-identical observations and trajectories.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch

from config import DEFAULT_CONFIG
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt


def _make_env(num_envs=4, opponent=False):
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["domain_randomization"]["enabled"] = False
    if opponent:
        cfg["env"]["opponent_strategy"] = "scripted"
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )


def _on_track_pose(env):
    env.reset(seed=1)
    return env.read_state()["base_pos"][0, :2].clone()


def test_reset_to_places_cars_at_requested_pose():
    env = _make_env(num_envs=4)
    try:
        pose = _on_track_pose(env)
        yaw, speed = 0.5, 2.0
        env.reset_to(pose, yaw, speed, seed=1)
        st = env.read_state()
        assert torch.allclose(st["base_pos"][:, :2], pose.expand(4, 2), atol=1e-4)
        heading = 2.0 * torch.atan2(st["base_quat"][:, 3], st["base_quat"][:, 0])
        assert torch.allclose(heading, torch.full((4,), yaw), atol=1e-4)
        assert torch.allclose(st["base_lin_vel"][:, 0], torch.full((4,), speed), atol=1e-4)
        assert torch.allclose(st["base_lin_vel"][:, 1], torch.zeros(4), atol=1e-4)
    finally:
        env.close()


def test_reset_to_is_repeatable_observations():
    env = _make_env(num_envs=4)
    try:
        pose = _on_track_pose(env)
        first, _ = env.reset_to(pose, 0.3, 1.5, seed=2)
        first = first.clone()
        second, _ = env.reset_to(pose, 0.3, 1.5, seed=2)
        assert torch.equal(first, second)
    finally:
        env.close()


@pytest.mark.parametrize("opponent", [False, True])
def test_reset_to_yields_identical_trajectories(opponent):
    env = _make_env(num_envs=4, opponent=opponent)
    try:
        pose = _on_track_pose(env)
        opp_pose = pose + torch.tensor([0.0, 0.4]) if opponent else None

        def rollout():
            env.reset_to(
                pose,
                0.2,
                1.0,
                opponent_pose=opp_pose,
                opponent_yaw=0.2 if opponent else None,
                opponent_speed=1.0 if opponent else None,
                seed=5,
            )
            trajectory = []
            for _ in range(12):
                action = torch.zeros(4, 2)
                action[:, 0] = 0.5
                action[:, 1] = 0.15
                env.step(action, n_steps=env.control_interval)
                trajectory.append(env.read_state()["base_pos"].clone())
            return torch.stack(trajectory)

        assert torch.equal(rollout(), rollout())
    finally:
        env.close()


def test_reset_to_opponent_defaults_to_ego_when_omitted():
    env = _make_env(num_envs=2, opponent=True)
    try:
        pose = _on_track_pose(env)
        env.reset_to(pose, 0.0, 1.0, seed=1)
        st = env.read_state()
        assert torch.isfinite(st["base_pos"]).all()
        assert torch.isfinite(st["opp_base_pos"]).all()
        assert math.isfinite(float(st["opp_base_pos"].sum()))
    finally:
        env.close()
