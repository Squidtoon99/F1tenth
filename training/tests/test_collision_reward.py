"""Real-Genesis test: config-gated collision reward penalty."""

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


def _build_cfg(*, spawn_gap_m: float, enable_collision_reward: bool) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["opponent_strategy"] = "scripted"
    cfg["env"]["opponent_spawn_gap_m"] = spawn_gap_m
    cfg["env"]["term_not_moving_time_s"] = 999.0
    cfg["obs"]["enable_opponent_obs"] = True
    cfg["obs"]["num_obs"] = 380 + int(cfg["obs"]["opponent_obs_dim"])
    cfg["reward"]["reward_scales"]["passing"] = 0.5
    if enable_collision_reward:
        cfg["reward"]["reward_scales"]["collision"] = 1.0
    return cfg


def _make_env(cfg: dict, num_envs: int) -> F1tenthEnv:
    env_cfg = {
        "launch_strategy": "fixed",
        "launch_strategy_data": {"num_cars": num_envs},
        **cfg["env"],
    }
    env_cfg["car_spawn_pos"] = (0.42, 0.16, 0.01)
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )


@pytest.fixture(scope="module")
def genesis_backend():
    if not gs._initialized:
        gs.init(backend=gs.cpu, precision="32", logging_level="warning")
    return gs


def test_collision_penalty_fires_on_contact(genesis_backend):
    num_envs = 2
    control_interval = int(DEFAULT_CONFIG["env"]["control_interval"])
    cfg = _build_cfg(spawn_gap_m=0.25, enable_collision_reward=True)
    env = _make_env(cfg, num_envs)

    try:
        obs, _ = env.reset()
        assert obs.shape == (num_envs, cfg["obs"]["num_obs"])

        collision_terms: list[float] = []
        for _ in range(80):
            actions = torch.zeros(num_envs, 2, device=env.device)
            actions[:, 0] = 1.0
            obs, reward, done, extras = env.step(actions, n_steps=control_interval)
            term = extras["rewards"]["terms"].get("collision")
            if term is not None:
                collision_terms.append(float(term.min().item()))

        assert collision_terms, "collision reward term never present"
        assert min(collision_terms) < -1e-6, (
            f"expected negative collision penalty, got min={min(collision_terms)}"
        )
    finally:
        env.close()


def test_collision_penalty_zero_when_separated(genesis_backend):
    num_envs = 2
    control_interval = int(DEFAULT_CONFIG["env"]["control_interval"])
    cfg = _build_cfg(spawn_gap_m=25.0, enable_collision_reward=True)
    env = _make_env(cfg, num_envs)

    try:
        obs, _ = env.reset()
        for _ in range(40):
            actions = torch.zeros(num_envs, 2, device=env.device)
            obs, reward, done, extras = env.step(actions, n_steps=control_interval)
            term = extras["rewards"]["terms"].get("collision")
            if term is not None:
                assert float(term.max().item()) == 0.0
                assert float(term.min().item()) == 0.0
    finally:
        env.close()
