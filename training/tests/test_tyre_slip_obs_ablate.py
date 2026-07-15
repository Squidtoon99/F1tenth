"""Warp tests for config-gated tyre-slip observation ablation."""

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

TYRE_SLIP_SLICE = slice(372, 380)


def _make_env(*, zero_slip: bool, num_envs: int = 2) -> F1tenthEnv:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["obs"]["zero_tyre_slip_obs"] = zero_slip
    # Isolate the ablation-zeroing check from observation noise: obs DR is applied
    # after build_observation, so leaving it on would add noise to the zeroed slip
    # channels and mask the exact-zero assertion.
    cfg["env"]["domain_randomization"]["enabled"] = False
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


def test_zero_tyre_slip_obs_zeros_channels(warp_runtime):
    num_envs = 2
    expected_dim = int(DEFAULT_CONFIG["obs"]["num_obs"])
    env = _make_env(zero_slip=True, num_envs=num_envs)
    control_interval = int(DEFAULT_CONFIG["env"]["control_interval"])
    try:
        obs, _ = env.reset()
        assert obs.shape == (num_envs, expected_dim)
        assert torch.all(obs[:, TYRE_SLIP_SLICE] == 0.0)

        for _ in range(60):
            actions = torch.zeros(num_envs, 2, device=env.device)
            actions[:, 0] = 1.0
            obs, _, _, _ = env.step(actions, n_steps=control_interval)
            assert obs.shape == (num_envs, expected_dim)
            assert torch.all(obs[:, TYRE_SLIP_SLICE] == 0.0)
    finally:
        env.close()


def test_tyre_slip_obs_populated_when_not_zeroed(warp_runtime):
    num_envs = 2
    env = _make_env(zero_slip=False, num_envs=num_envs)
    control_interval = int(DEFAULT_CONFIG["env"]["control_interval"])
    try:
        obs, _ = env.reset()
        saw_nonzero = bool((obs[:, TYRE_SLIP_SLICE].abs() > 1e-6).any())
        for _ in range(80):
            actions = torch.zeros(num_envs, 2, device=env.device)
            actions[:, 0] = 1.0
            obs, _, _, _ = env.step(actions, n_steps=control_interval)
            if (obs[:, TYRE_SLIP_SLICE].abs() > 1e-6).any():
                saw_nonzero = True
                break
        assert saw_nonzero, "expected tyre-slip channels to carry signal when not ablated"
    finally:
        env.close()
