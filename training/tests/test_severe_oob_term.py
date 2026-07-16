"""Severe out-of-bounds termination split (production Warp kernel).

Terminal OOB uses the vehicle *center* crossing the track boundary held for
``term_oob_max_consecutive`` steps, while the reward off-course signal stays
footprint-aware. So a car whose center is outside the boundary resets, but a car
that only has a footprint edge past the boundary (center still inside) keeps
racing even though its progress is masked off-course.
"""

from __future__ import annotations

import copy
import math

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
    return F1tenthEnv(
        num_envs=1,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )


def _place_at_lateral(env, target_ey: float):
    """reset_to the ego so its centre sits at signed offset ``target_ey`` (m) from
    the centerline, returning the local left half-width used for the target."""
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
    env.reset_to(target, yaw, 0.0, seed=5)
    return w_l


def test_center_outside_boundary_terminates():
    env = _make_env()
    try:
        # First measure the local left half-width, then place the centre well past it.
        w_l = _place_at_lateral(env, target_ey=0.0)
        _place_at_lateral(env, target_ey=w_l + 0.5)

        window = int(env.env_cfg["term_oob_max_consecutive"])
        fired = False
        for _ in range(window):
            _, _, done, extras = env.step(
                torch.zeros(1, 2), n_steps=env.control_interval
            )
            if bool(extras["termination"]["out_of_bounds"][0]):
                fired = True
                assert bool(done[0])
        assert fired, "center past the boundary for the window did not terminate"
    finally:
        env.close()


def test_footprint_edge_only_does_not_terminate():
    env = _make_env()
    try:
        w_l = _place_at_lateral(env, target_ey=0.0)
        # Centre inside the terminal margin (0.15 m) but the footprint edge (half
        # width 0.148 m) crosses the reward margin (0.2 m): off-course but not severe.
        _place_at_lateral(env, target_ey=w_l - 0.25)

        off_course_seen = False
        for _ in range(6):
            _, _, done, extras = env.step(
                torch.zeros(1, 2), n_steps=env.control_interval
            )
            assert not bool(extras["termination"]["out_of_bounds"][0])
            assert not bool(done[0])
            if bool(extras["metrics"]["oob_mask"][0] > 0):
                off_course_seen = True
        assert off_course_seen, "footprint edge never registered off-course"
    finally:
        env.close()
