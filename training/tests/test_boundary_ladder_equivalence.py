"""Recoverable ladder configs must match legacy footprint-margin boundary semantics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt
from standalone_trainer import build_config, load_config_patch, parse_args

_REPO = Path(__file__).resolve().parents[2]
_LADDER_PATCH = (
    _REPO
    / "training/outputs/experiments/mainline-legacy-ladder/mainline-ladder600.json"
)


def _configure(device: str = "cpu") -> None:
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device(device),
        eps=1e-12,
    )


def _ladder_cfg(*, boundary_mode: str = "recoverable_full_car_out") -> dict:
    patch, _ = load_config_patch(str(_LADDER_PATCH))
    patch["env"]["boundary_mode"] = boundary_mode
    patch["env"]["domain_randomization"] = {"enabled": False}
    patch["env"]["term_not_moving_time_s"] = 1e6
    args, explicit = parse_args(["--seed", "42"])
    cfg = build_config(args, patch=patch, explicit=explicit)
    cfg["env"]["opponent_strategy"] = None
    return cfg


def _make_env(boundary_mode: str = "recoverable_full_car_out") -> F1tenthEnv:
    _configure()
    cfg = _ladder_cfg(boundary_mode=boundary_mode)
    return F1tenthEnv(
        num_envs=1,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )


def _move_to_lateral(env: F1tenthEnv, target_ey: float, speed: float):
    state = env.read_state()
    pos = state["base_pos"][0, :2].clone()
    quat = state["base_quat"][0]
    yaw = float(2.0 * torch.atan2(quat[3], quat[0]))
    ey0 = float(env.extras["metrics"]["lateral_error"][0])
    n_hat = torch.tensor([-math.sin(yaw), math.cos(yaw)], dtype=torch.float32)
    target = pos + (target_ey - ey0) * n_hat
    env.reset_to(target, yaw, speed, seed=5)
    return float(env.w_tr_left[int(env._env.tensor["ego_segment"][0])]), yaw


def _place_at_lateral(env: F1tenthEnv, target_ey: float, speed: float):
    env.reset(seed=5)
    return _move_to_lateral(env, target_ey, speed)


def _footprint_half_width(env: F1tenthEnv) -> float:
    return 0.5 * float(env.env_cfg["car_width"])


def test_recoverable_ladder_uses_footprint_margin_not_touch_contact():
    """Legacy ``oob_penalty_form=footprint_margin`` keeps graze costs at bay until
    the footprint penetrates the boundary, not when it merely touches it."""
    env = _make_env(boundary_mode="recoverable_full_car_out")
    try:
        w_l, yaw = _place_at_lateral(env, target_ey=0.0, speed=0.0)
        half_width = _footprint_half_width(env)
        _move_to_lateral(env, w_l - half_width, speed=2.0)
        _, reward, done, extras = env.step(
            torch.zeros(1, 2), n_steps=env.control_interval
        )
        terms = extras["rewards"]["terms"]
        assert not bool(extras["termination"]["out_of_bounds"][0])
        assert not bool(done[0])
        assert float(terms["wall_contact"][0]) == 0.0
        assert float(terms["boundary_contact"][0]) == 0.0
        assert float(terms["progress"][0]) != 0.0
        assert float(reward[0]) == pytest.approx(
            float(terms["progress"][0]), abs=1e-4
        )
    finally:
        env.close()


def test_mainline_ladder_patch_resolves_recoverable_mode():
    cfg = _ladder_cfg()
    assert cfg["env"]["boundary_mode"] == "recoverable_full_car_out"
    assert cfg["reward"]["wall_cost_mode"] == "continuous_quadratic"
    assert cfg["reward"]["wall_contact_coefficient"] == pytest.approx(0.1296)
    assert cfg["reward"]["boundary_contact_coefficient"] == pytest.approx(4.0)
    assert cfg["reward"]["oob_impact_coefficient"] == pytest.approx(0.1296)
    legacy = json.loads(
        (
            _REPO
            / "training/outputs/experiments/causal-2x2/configs/pc-plus.json"
        ).read_text()
    )
    assert legacy["reward"]["reward_scales"]["oob_penalty"] * 12.96 == pytest.approx(
        0.1296
    )
    assert (
        legacy["reward"]["reward_scales"]["oob_impact"]
        * legacy["reward"]["terminal_oob_skip_seconds"]
        * 12.96
    ) == pytest.approx(0.1296)
