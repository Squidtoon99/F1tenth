"""Piecewise high-speed progress bonus through the real Warp kernel."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from config import DEFAULT_CONFIG
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt
from f1tenth_env.rewards import _shape_forward_progress_ds


def _reward_cfg(
    *,
    threshold_mps: float = 0.0,
    multiplier: float = 1.0,
    saturation_mps: float = 0.0,
    progress_scale: float = 1.0,
):
    return {
        "progress_speed_threshold_mps": threshold_mps,
        "progress_high_speed_multiplier": multiplier,
        "progress_speed_saturation_mps": saturation_mps,
        "control_dt": 0.1,
        "global_reward_scale": 1.0,
        "reward_scales": {"progress": progress_scale},
    }


def _expected_shaped_ds(
    ds: float,
    *,
    threshold_mps: float,
    multiplier: float,
    saturation_mps: float,
    control_dt: float = 0.1,
) -> float:
    ds_thr = threshold_mps * control_dt
    ds_sat = saturation_mps * control_dt
    if ds <= ds_thr:
        return ds
    if saturation_mps > 0.0 and ds > ds_sat:
        return ds_thr + multiplier * (ds_sat - ds_thr) + (ds - ds_sat)
    return ds_thr + multiplier * (ds - ds_thr)


@pytest.mark.parametrize(
    ("ds", "expected"),
    [
        (0.3, 0.3),
        (0.5, 0.5),
        (0.6, 0.8),
        (0.8, 1.4),
        (1.0, 1.6),
    ],
)
def test_piecewise_progress_exact_values(ds, expected):
    cfg = _reward_cfg(threshold_mps=5.0, multiplier=3.0, saturation_mps=8.0)
    shaped = _shape_forward_progress_ds(torch.tensor([ds]), cfg)
    assert shaped[0].item() == pytest.approx(expected, abs=1e-6)


def test_defaults_match_linear_progress():
    cfg = _reward_cfg()
    ds = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.25])
    assert torch.equal(_shape_forward_progress_ds(ds, cfg), ds)


def test_continuity_at_threshold_and_saturation():
    cfg = _reward_cfg(threshold_mps=5.0, multiplier=3.0, saturation_mps=8.0)
    ds_thr = 0.5
    ds_sat = 0.8
    eps = 1e-4
    for pivot, label in ((ds_thr, "threshold"), (ds_sat, "saturation")):
        left = _shape_forward_progress_ds(torch.tensor([pivot - eps]), cfg)[0].item()
        at = _shape_forward_progress_ds(torch.tensor([pivot]), cfg)[0].item()
        right = _shape_forward_progress_ds(torch.tensor([pivot + eps]), cfg)[0].item()
        assert left == pytest.approx(at, abs=1e-3), label
        assert right == pytest.approx(at, abs=1e-3), label


def _make_env(**reward_overrides):
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
    cfg["env"]["boundary_mode"] = "recoverable_full_car_out"
    cfg["reward"]["reward_scales"] = {
        name: 0.0 for name in cfg["reward"]["reward_scales"]
    }
    cfg["reward"]["reward_scales"]["progress"] = 1.0
    cfg["reward"].update(reward_overrides)
    return F1tenthEnv(
        num_envs=1,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )


def _place_centerline(env, speed: float):
    env.reset(seed=3)
    state = env.read_state()
    pos = state["base_pos"][0, :2].clone()
    quat = state["base_quat"][0]
    yaw = float(2.0 * torch.atan2(quat[3], quat[0]))
    env.reset_to(pos, yaw, speed, seed=3)


def test_kernel_matches_mirror_under_treatment():
    env = _make_env(
        progress_speed_threshold_mps=5.0,
        progress_high_speed_multiplier=3.0,
        progress_speed_saturation_mps=8.0,
    )
    try:
        _place_centerline(env, speed=6.0)
        _, _, _, extras = env.step(
            torch.tensor([[0.8, 0.0]]), n_steps=env.control_interval
        )
        ds = float(extras["metrics"]["progress_ds"][0])
        expected = _expected_shaped_ds(
            ds,
            threshold_mps=5.0,
            multiplier=3.0,
            saturation_mps=8.0,
            control_dt=env.control_dt,
        )
        assert float(extras["rewards"]["terms"]["progress"][0]) == pytest.approx(
            expected, abs=1e-4
        )
    finally:
        env.close()


def test_boundary_event_zeros_shaped_progress():
    env = _make_env(
        progress_speed_threshold_mps=5.0,
        progress_high_speed_multiplier=3.0,
        progress_speed_saturation_mps=8.0,
        wall_cost_mode="continuous_quadratic",
        wall_contact_coefficient=0.1296,
    )
    try:
        w_l, _ = _place_at_lateral(env, target_ey=0.0, speed=0.0)
        half_width = 0.5 * float(env.env_cfg["car_width"])
        _place_at_lateral(env, target_ey=w_l - half_width + 0.01, speed=6.0)
        _, _, done, extras = env.step(
            torch.zeros(1, 2), n_steps=env.control_interval
        )
        assert not bool(done[0])
        assert float(extras["rewards"]["terms"]["progress"][0]) == 0.0
    finally:
        env.close()


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
