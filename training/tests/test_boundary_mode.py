"""Config-selectable boundary termination geometry and wall-cost shape.

Exercises the real Warp kernel through ``F1tenthEnv`` (no mocks). Covers the two
independent switches added on top of ADR 0013/0014's first-footprint-terminal,
one-shot wall event: ``env.boundary_mode`` ("first_contact_terminal", the
default, vs "recoverable_full_car_out") and ``reward.wall_cost_mode``
("one_shot", the default, vs "continuous" or "continuous_quadratic").
"""

from __future__ import annotations

import copy
import math

import pytest
import torch

from config import DEFAULT_CONFIG
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt


def _make_env(
    *,
    boundary_mode: str = "first_contact_terminal",
    wall_cost_mode: str = "one_shot",
    boundary_contact_coefficient: float = 0.0,
    oob_impact_coefficient: float = 0.0,
):
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
    cfg["env"]["boundary_mode"] = boundary_mode
    cfg["reward"]["wall_cost_mode"] = wall_cost_mode
    cfg["reward"]["boundary_contact_coefficient"] = boundary_contact_coefficient
    cfg["reward"]["oob_impact_coefficient"] = oob_impact_coefficient
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


def _move_to_lateral(env, target_ey: float, speed: float):
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


def _place_at_lateral(env, target_ey: float, speed: float):
    env.reset(seed=5)
    return _move_to_lateral(env, target_ey, speed)


def test_default_config_is_unchanged_first_contact_terminal():
    assert DEFAULT_CONFIG["env"]["boundary_mode"] == "first_contact_terminal"
    assert DEFAULT_CONFIG["reward"]["wall_cost_mode"] == "one_shot"
    assert DEFAULT_CONFIG["reward"]["boundary_contact_coefficient"] == 0.0
    assert DEFAULT_CONFIG["reward"]["oob_impact_coefficient"] == 0.0


@pytest.mark.parametrize(
    ("speed", "expected_reward"),
    [(1.0, -2.0), (2.0, -4.0), (5.0, -10.0)],
)
def test_continuous_wall_cost_is_dt_scaled_at_first_contact(speed, expected_reward):
    env = _make_env(wall_cost_mode="continuous")
    try:
        w_l, _ = _place_at_lateral(env, target_ey=0.0, speed=0.0)
        half_width = 0.5 * float(env.env_cfg["car_width"])
        _place_at_lateral(env, target_ey=w_l - half_width + 0.01, speed=speed)
        _, reward, done, extras = env.step(
            torch.zeros(1, 2), n_steps=env.control_interval
        )
        assert env.control_dt == pytest.approx(0.1)
        assert bool(extras["termination"]["out_of_bounds"][0])
        assert bool(done[0])
        terms = extras["rewards"]["terms"]
        assert float(terms["progress"][0]) == 0.0
        assert float(terms["wall_contact"][0]) == pytest.approx(
            expected_reward, abs=1e-4
        )
        assert float(reward[0]) == pytest.approx(expected_reward, abs=1e-4)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("speed", "expected_reward"),
    [(1.0, -2.0), (2.0, -8.0), (5.0, -50.0)],
)
def test_continuous_quadratic_wall_cost_is_dt_scaled_speed_squared(speed, expected_reward):
    env = _make_env(wall_cost_mode="continuous_quadratic")
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
        assert float(terms["wall_contact"][0]) == pytest.approx(
            expected_reward, abs=1e-4
        )
        assert float(reward[0]) == pytest.approx(expected_reward, abs=1e-4)
    finally:
        env.close()


def test_first_contact_does_not_terminate_under_recoverable_boundary():
    env = _make_env(boundary_mode="recoverable_full_car_out")
    try:
        w_l, _ = _place_at_lateral(env, target_ey=0.0, speed=0.0)
        half_width = 0.5 * float(env.env_cfg["car_width"])
        _place_at_lateral(env, target_ey=w_l - half_width + 0.01, speed=2.0)
        _, reward, done, extras = env.step(
            torch.zeros(1, 2), n_steps=env.control_interval
        )
        assert not bool(extras["termination"]["out_of_bounds"][0])
        assert not bool(done[0])
        terms = extras["rewards"]["terms"]
        assert float(terms["progress"][0]) == 0.0, "recoverable path masks progress on contact"
        assert float(terms["wall_contact"][0]) == pytest.approx(-40.0, abs=1e-4)
    finally:
        env.close()


def test_full_car_out_terminates_under_recoverable_boundary():
    env = _make_env(boundary_mode="recoverable_full_car_out")
    try:
        w_l, _ = _place_at_lateral(env, target_ey=0.0, speed=0.0)
        half_width = 0.5 * float(env.env_cfg["car_width"])
        _place_at_lateral(env, target_ey=w_l + half_width + 0.01, speed=2.0)
        _, reward, done, extras = env.step(
            torch.zeros(1, 2), n_steps=env.control_interval
        )
        assert bool(extras["termination"]["out_of_bounds"][0])
        assert bool(done[0])
        terms = extras["rewards"]["terms"]
        assert float(terms["boundary_contact"][0]) == 0.0
        assert float(terms["oob_impact"][0]) == 0.0
    finally:
        env.close()


def test_recoverable_boundary_with_continuous_cost_charges_every_contact_step():
    """The probe configuration: matches Lee's continuous linear-speed atom."""
    env = _make_env(boundary_mode="recoverable_full_car_out", wall_cost_mode="continuous")
    try:
        w_l, yaw = _place_at_lateral(env, target_ey=0.0, speed=0.0)
        half_width = 0.5 * float(env.env_cfg["car_width"])
        _place_at_lateral(env, target_ey=w_l - half_width + 0.01, speed=2.0)
        for _ in range(3):
            _, reward, done, extras = env.step(
                torch.zeros(1, 2), n_steps=env.control_interval
            )
            assert not bool(extras["termination"]["out_of_bounds"][0])
            assert not bool(done[0])
            terms = extras["rewards"]["terms"]
            assert float(terms["progress"][0]) == 0.0
            # Speed decays slightly step-to-step (no throttle); the cost stays
            # a small negative dt-scaled value rather than the -40 one-shot
            # magnitude, confirming the continuous formula is active every step.
            assert -4.5 < float(terms["wall_contact"][0]) < -0.1
    finally:
        env.close()


def test_just_inside_footprint_does_not_contact_or_terminate_when_recoverable():
    env = _make_env(boundary_mode="recoverable_full_car_out", wall_cost_mode="continuous")
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


def test_legacy_ladder_composes_graze_one_shot_contact_and_terminal_impact():
    """The legacy graduated ladder: a cheap continuous graze cost, a one-shot
    penalty on first contact, and a terminal impact shock on full-car-out,
    composed from independent knobs rather than a single reward-stack switch.
    """
    env = _make_env(
        boundary_mode="recoverable_full_car_out",
        wall_cost_mode="continuous_quadratic",
        boundary_contact_coefficient=4.0,
        oob_impact_coefficient=0.1296,
    )
    try:
        w_l, _ = _place_at_lateral(env, target_ey=0.0, speed=0.0)
        half_width = 0.5 * float(env.env_cfg["car_width"])
        wall_coefficient = float(env.reward_cfg["wall_contact_coefficient"])

        # Rung 1 + rung 2: first contact is a cheap continuous graze plus the
        # one-shot contact penalty; the episode is not terminated.
        _place_at_lateral(env, target_ey=w_l - half_width + 0.01, speed=2.0)
        _, _, done, extras = env.step(
            torch.zeros(1, 2), n_steps=env.control_interval
        )
        assert not bool(extras["termination"]["out_of_bounds"][0])
        assert not bool(done[0])
        terms = extras["rewards"]["terms"]
        assert float(terms["progress"][0]) == 0.0
        expected_graze = -wall_coefficient * env.control_dt * 4.0
        assert float(terms["wall_contact"][0]) == pytest.approx(
            expected_graze, abs=1e-4
        )
        assert float(terms["boundary_contact"][0]) == pytest.approx(
            -4.0, abs=1e-4
        )
        assert float(terms["oob_impact"][0]) == 0.0

        # Still in contact one tick later: the graze cost recurs but the
        # one-shot penalty does not fire a second time for the same excursion.
        _, _, done, extras = env.step(
            torch.zeros(1, 2), n_steps=env.control_interval
        )
        assert not bool(done[0])
        terms = extras["rewards"]["terms"]
        assert float(terms["wall_contact"][0]) < 0.0
        assert float(terms["boundary_contact"][0]) == 0.0

        # Rung 3: driving straight from clean track to fully out (skipping the
        # recoverable grace period) terminates the episode and adds the
        # terminal impact shock, scaled by the speed at exit.
        _place_at_lateral(env, target_ey=w_l + half_width + 0.01, speed=3.0)
        _, _, done, extras = env.step(
            torch.zeros(1, 2), n_steps=env.control_interval
        )
        assert bool(extras["termination"]["out_of_bounds"][0])
        assert bool(done[0])
        terms = extras["rewards"]["terms"]
        assert float(terms["progress"][0]) == 0.0
        assert float(terms["oob_impact"][0]) == pytest.approx(
            -0.1296 * 9.0, abs=1e-3
        )
    finally:
        env.close()
