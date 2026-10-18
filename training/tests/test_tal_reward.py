from __future__ import annotations

import copy
import os
import sys

import torch

from standalone_trainer import validate_config_patch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import DEFAULT_CONFIG  # noqa: E402
from f1tenth_env import F1tenthEnv  # noqa: E402


def _build_cfg(*, tal_scale: float) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["domain_randomization"]["enabled"] = False
    cfg["env"]["reset_spawn_yaw_jitter_rad"] = 0.0
    cfg["env"]["reset_stationary_probability"] = 0.0
    cfg["env"]["reset_speed_min_mps"] = 2.0
    cfg["env"]["reset_speed_max_mps"] = 2.0
    cfg["env"]["opponent_strategy"] = "none"
    cfg["env"]["term_not_moving_time_s"] = 1e6
    cfg["obs"]["enable_opponent_obs"] = False
    scales = cfg["reward"]["reward_scales"]
    for key in list(scales):
        scales[key] = 0.0
    scales["tal"] = float(tal_scale)
    cfg["reward"]["global_reward_scale"] = 1.0
    cfg["reward"]["wall_contact_coefficient"] = 0.0
    return cfg


def _make_env(cfg: dict) -> F1tenthEnv:
    return F1tenthEnv(
        num_envs=2,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )


def test_default_tal_scale_is_off():
    assert DEFAULT_CONFIG["reward"]["reward_scales"]["tal"] == 0.0
    validate_config_patch({"reward": {"reward_scales": {"tal": 0.2}}})


def test_tal_scale_zero_emits_zero_term(warp_runtime):
    cfg = _build_cfg(tal_scale=0.0)
    env = _make_env(cfg)
    try:
        env.reset()
        actions = torch.zeros(env.num_envs, 2, device=env.device)
        _, reward, _, extras = env.step(actions, n_steps=env.control_interval)
        tal = extras["rewards"]["terms"]["tal"]
        assert float(tal.abs().max()) == 0.0
        assert float(reward.abs().max()) == 0.0
    finally:
        env.close()


def test_tal_scale_changes_reward(warp_runtime):
    off = _build_cfg(tal_scale=0.0)
    on = _build_cfg(tal_scale=0.2)
    env_off = _make_env(off)
    env_on = _make_env(on)
    try:
        env_off.reset()
        env_on.reset()
        actions = torch.zeros(2, 2, device=env_off.device)
        _, reward_off, _, extras_off = env_off.step(
            actions, n_steps=env_off.control_interval
        )
        _, reward_on, _, extras_on = env_on.step(
            actions, n_steps=env_on.control_interval
        )
        tal = extras_on["rewards"]["terms"]["tal"]
        assert float(extras_off["rewards"]["terms"]["tal"].abs().max()) == 0.0
        assert not torch.equal(reward_on, reward_off)
        assert torch.equal(reward_on, tal)
        assert tal.abs().max() > 0.0
    finally:
        env_off.close()
        env_on.close()


def test_tal_ignores_fixed_speed_target(warp_runtime):
    slow = _build_cfg(tal_scale=0.2)
    fast = _build_cfg(tal_scale=0.2)
    slow["env"]["expected_lap_speed_mps"] = 3.5
    fast["env"]["expected_lap_speed_mps"] = 10.0
    env_slow = _make_env(slow)
    env_fast = _make_env(fast)
    try:
        env_slow.reset(seed=0)
        env_fast.reset(seed=0)
        actions = torch.zeros(2, 2, device=env_slow.device)
        _, _, _, extras_slow = env_slow.step(
            actions, n_steps=env_slow.control_interval
        )
        _, _, _, extras_fast = env_fast.step(
            actions, n_steps=env_fast.control_interval
        )
        tal_slow = extras_slow["rewards"]["terms"]["tal"]
        tal_fast = extras_fast["rewards"]["terms"]["tal"]
        assert torch.equal(tal_slow, tal_fast)
        assert tal_slow.abs().max() > 0.0
    finally:
        env_slow.close()
        env_fast.close()
