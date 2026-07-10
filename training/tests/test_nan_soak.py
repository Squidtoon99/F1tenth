"""Real-Genesis long 1v1 contact soak: assert zero non-finite obs/reward/state."""

from __future__ import annotations

import copy
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

gs = pytest.importorskip("genesis")

from config import DEFAULT_CONFIG  # noqa: E402
from f1tenth_env import F1tenthEnv  # noqa: E402


def _build_cfg(*, spawn_gap_m: float) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["opponent_strategy"] = "scripted"
    cfg["env"]["opponent_spawn_gap_m"] = spawn_gap_m
    cfg["env"]["opponent_target_speed"] = 1.5
    cfg["env"]["term_not_moving_time_s"] = 999.0
    cfg["obs"]["enable_opponent_obs"] = True
    cfg["obs"]["num_obs"] = 384 + int(cfg["obs"]["opponent_obs_dim"])
    cfg["reward"]["reward_scales"]["passing"] = 0.5
    return cfg


def _make_env(cfg: dict, num_envs: int) -> F1tenthEnv:
    env_cfg = {
        "launch_strategy": "uniform_jittered",
        "launch_strategy_data": {"num_cars": num_envs},
        **cfg["env"],
    }
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )


def test_1v1_contact_soak_zero_nonfinite(genesis_backend_f64):
    """Spawn ego close behind opponent and chase for many steps with repeated contact."""
    num_envs = 4
    steps = 600
    control_interval = int(DEFAULT_CONFIG["env"]["control_interval"])
    cfg = _build_cfg(spawn_gap_m=2.0)
    env = _make_env(cfg, num_envs)
    clip = float(cfg["env"]["clip_actions"])

    try:
        obs, _ = env.reset()
        assert obs.shape == (num_envs, cfg["obs"]["num_obs"])

        total_nf_obs = 0
        total_nf_reward = 0
        total_nf_state = 0
        collisions = 0

        for _ in range(steps):
            actions = torch.zeros(num_envs, 2, device=env.device)
            actions[:, 0] = clip
            obs, reward, done, extras = env.step(actions, n_steps=control_interval)

            assert torch.isfinite(obs).all(), "non-finite observation during soak"
            assert torch.isfinite(reward).all(), "non-finite reward during soak"

            metrics = extras.get("metrics", {})
            total_nf_obs += int(metrics.get("nonfinite_obs_envs", torch.zeros(1)).sum())
            total_nf_reward += int(
                metrics.get("nonfinite_reward_envs", torch.zeros(1)).sum()
            )
            total_nf_state += int(
                metrics.get("nonfinite_state_envs", torch.zeros(1)).sum()
            )
            collisions += int(
                extras.get("termination", {}).get(
                    "collision", torch.zeros(1)
                ).sum()
            )

        assert total_nf_obs == 0, f"env reported nonfinite obs rows: {total_nf_obs}"
        assert total_nf_reward == 0, (
            f"env reported nonfinite reward rows: {total_nf_reward}"
        )
        assert total_nf_state == 0, (
            f"env reported nonfinite state rows: {total_nf_state}"
        )
        assert collisions > 0, "contact soak never triggered a collision termination"
    finally:
        env.close()
