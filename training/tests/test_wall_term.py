"""Warp-env wall-impact termination and center-penetration fail-safe."""

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
    cfg["env"]["reset_spawn_yaw_jitter_rad"] = 0.0
    cfg["env"]["term_not_moving_time_s"] = 1e6
    cfg["env"]["term_oob_max_consecutive"] = 2
    cfg["env"]["wall_impact_term_speed_mps"] = 4.0
    cfg["env"]["opponent_strategy"] = None
    return F1tenthEnv(
        num_envs=1,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )


def _place_into_left_wall(env, *, speed: float, penetrate_center: bool = False):
    """Place ego against the left corridor wall, nose pointed into the wall."""
    env.reset(seed=5)
    state = env.read_state()
    pos = state["base_pos"][0, :2].clone()
    quat = state["base_quat"][0]
    yaw = float(2.0 * torch.atan2(quat[3], quat[0]))
    ey0 = float(env.extras["metrics"]["lateral_error"][0])
    segment = int(env._env.tensor["ego_segment"][0])
    w_l = float(env.w_tr_left[segment])
    car_length = float(env.env_cfg.get("car_length", 0.568))
    # Point into the left wall; footprint half-extent along track normal is L/2.
    wall_yaw = yaw + 0.5 * math.pi
    half = 0.5 * car_length
    if penetrate_center:
        target_ey = w_l + 0.5
    else:
        target_ey = w_l - half + 0.01
    n_hat = torch.tensor([-math.sin(yaw), math.cos(yaw)], dtype=torch.float32)
    target = pos + (target_ey - ey0) * n_hat
    env.reset_to(target, wall_yaw, speed, seed=5)
    return w_l


def test_high_normal_speed_wall_impact_terminates_with_impact_reward():
    env = _make_env()
    try:
        _place_into_left_wall(env, speed=5.0)
        _, reward, done, extras = env.step(
            torch.zeros(1, 2), n_steps=env.control_interval
        )
        assert bool(extras["termination"]["wall_impact"][0])
        assert bool(done[0])
        assert not bool(extras["termination"]["out_of_bounds"][0])
        impact = float(extras["rewards"]["terms"]["wall_impact"][0])
        # Physics can change speed slightly over the control interval; keep a
        # loose band around the ideal -1 * 5^2 head-on impact.
        assert impact == pytest.approx(-25.0, abs=0.5)
        assert float(reward[0]) <= impact + 1e-3
    finally:
        env.close()


def test_low_normal_speed_glancing_does_not_wall_terminate():
    env = _make_env()
    try:
        _place_into_left_wall(env, speed=1.0)
        saw_contact = False
        for _ in range(4):
            _, _, done, extras = env.step(
                torch.zeros(1, 2), n_steps=env.control_interval
            )
            if float(extras["metrics"]["wall_contact"][0]) > 0:
                saw_contact = True
            assert not bool(extras["termination"]["wall_impact"][0])
            if bool(extras["termination"]["out_of_bounds"][0]):
                break
            assert not bool(done[0]) or bool(
                extras["termination"]["out_of_bounds"][0]
            )
        assert saw_contact, "never observed geometric wall contact"
    finally:
        env.close()


def test_center_penetration_still_oob_failsafe():
    env = _make_env()
    try:
        _place_into_left_wall(env, speed=0.0, penetrate_center=True)
        window = int(env.env_cfg["term_oob_max_consecutive"])
        fired = False
        for _ in range(window):
            _, _, done, extras = env.step(
                torch.zeros(1, 2), n_steps=env.control_interval
            )
            if bool(extras["termination"]["out_of_bounds"][0]):
                fired = True
                assert bool(done[0])
        assert fired, "center past the boundary did not OOB-terminate"
    finally:
        env.close()
