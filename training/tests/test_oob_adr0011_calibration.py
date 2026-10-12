"""Canonical first-footprint wall contact and Lee barrier reward."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from config import DEFAULT_CONFIG
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt


def _make_env():
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["domain_randomization"]["enabled"] = False
    cfg["env"]["enable_aero_drag"] = False
    cfg["env"]["reset_spawn_yaw_jitter_rad"] = 0.0
    cfg["env"]["term_not_moving_time_s"] = 1e6
    cfg["reward"]["reward_scales"] = {
        name: 0.0 for name in cfg["reward"]["reward_scales"]
    }
    cfg["reward"]["reward_scales"]["progress"] = 1.0
    return F1tenthEnv(
        num_envs=1,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )


def _place_at_lateral(env, target_ey: float, speed: float):
    env.reset(seed=5)
    state = env.read_state()
    pos = state["base_pos"][0, :2].clone()
    quat = state["base_quat"][0]
    yaw = float(2.0 * torch.atan2(quat[3], quat[0]))
    ey0 = float(env.extras["metrics"]["lateral_error"][0])
    segment = int(env._env.tensor["ego_segment"][0])
    w_l = float(env.w_tr_left[segment])
    n_hat = torch.tensor([-math.sin(yaw), math.cos(yaw)], dtype=torch.float32)
    target = pos + (target_ey - ey0) * n_hat
    env.reset_to(target, yaw, speed, seed=5)
    return w_l, yaw


def test_just_inside_footprint_does_not_contact_or_terminate():
    env = _make_env()
    try:
        w_l, _ = _place_at_lateral(env, target_ey=0.0, speed=0.0)
        half_width = 0.5 * float(env.env_cfg["car_width"])
        _place_at_lateral(env, target_ey=w_l - half_width - 0.10, speed=0.0)
        _, _, done, extras = env.step(
            torch.zeros(1, 2), n_steps=env.control_interval
        )
        assert not bool(done[0])
        assert not bool(extras["termination"]["out_of_bounds"][0])
        assert float(extras["rewards"]["terms"]["wall_contact"][0]) == 0.0
    finally:
        env.close()


@pytest.mark.parametrize(
    ("speed", "expected_reward"),
    [(1.0, -15.0), (2.0, -30.0), (5.0, -75.0)],
)
def test_first_footprint_intersection_terminates_with_one_lee_reward(
    speed, expected_reward
):
    env = _make_env()
    try:
        w_l, _ = _place_at_lateral(env, target_ey=0.0, speed=0.0)
        half_width = 0.5 * float(env.env_cfg["car_width"])
        _place_at_lateral(env, target_ey=w_l - half_width + 0.01, speed=speed)
        _, reward, done, extras = env.step(
            torch.zeros(1, 2), n_steps=env.control_interval
        )
        assert bool(extras["termination"]["out_of_bounds"][0])
        assert bool(done[0])
        terms = extras["rewards"]["terms"]
        assert "oob_penalty" not in terms
        assert float(terms["oob_impact"][0]) == 0.0
        assert float(terms["boundary_contact"][0]) == 0.0
        assert float(terms["progress"][0]) == 0.0
        assert float(terms["wall_contact"][0]) == pytest.approx(
            expected_reward, abs=1e-4
        )
        assert float(reward[0]) == pytest.approx(expected_reward, abs=1e-4)
    finally:
        env.close()
