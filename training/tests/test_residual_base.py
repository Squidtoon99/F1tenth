from __future__ import annotations

import copy
import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import DEFAULT_CONFIG  # noqa: E402
from f1tenth_env import F1tenthEnv  # noqa: E402
from ppo import mix_residual_actions, residual_alpha  # noqa: E402
from standalone_trainer import (  # noqa: E402
    validate_algorithm_config,
    validate_config_patch,
)


def test_residual_alpha_schedule():
    assert residual_alpha(0, 64_000_000) == 1.0
    assert residual_alpha(16_000_000, 64_000_000) == 0.0
    assert residual_alpha(32_000_000, 64_000_000) == 0.0
    assert residual_alpha(8_000_000, 64_000_000) == 0.5


def test_mix_residual_actions_identity_and_clip():
    policy = torch.tensor([[0.25, -0.5]])
    base = torch.tensor([[1.0, 1.0]])
    assert torch.equal(mix_residual_actions(policy, base, 0.0), policy)
    mixed = mix_residual_actions(torch.zeros(1, 2), base, 1.0)
    assert torch.equal(mixed, base)
    overflow = mix_residual_actions(torch.ones(1, 2), base, 1.0)
    assert torch.equal(overflow, torch.ones(1, 2))


def test_default_residual_flag_is_off():
    assert DEFAULT_CONFIG["ppo"]["residual_base_enabled"] is False
    validate_config_patch({"ppo": {"residual_base_enabled": True}})
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["algorithm"] = "ppo"
    cfg["ppo"]["residual_base_enabled"] = True
    validate_algorithm_config(cfg, num_envs=8)


def _solo_cfg():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["domain_randomization"]["enabled"] = False
    cfg["env"]["opponent_strategy"] = "none"
    cfg["env"]["reset_spawn_yaw_jitter_rad"] = 0.0
    cfg["env"]["reset_speed_min_mps"] = 2.0
    cfg["env"]["reset_speed_max_mps"] = 2.0
    cfg["obs"]["enable_opponent_obs"] = False
    return cfg


def test_centerline_base_action_in_unit_box(warp_runtime):
    cfg = _solo_cfg()
    env = F1tenthEnv(
        num_envs=4,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )
    try:
        env.reset()
        base = env.centerline_base_action()
        assert base.shape == (4, 2)
        assert torch.isfinite(base).all()
        assert float(base.abs().max()) <= 1.0 + 1e-6
        mixed = mix_residual_actions(torch.zeros_like(base), base, 1.0)
        _, _, _, extras = env.step(mixed, n_steps=env.control_interval)
        assert extras["rewards"]["total"].shape == (4,)
    finally:
        env.close()
