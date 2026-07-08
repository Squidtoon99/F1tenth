"""Real-Genesis tests for config-gated domain randomization."""

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


def _make_env(*, enable_dr: bool, num_envs: int = 4) -> F1tenthEnv:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    dr = dict(cfg["env"]["domain_randomization"])
    dr["enabled"] = enable_dr
    if enable_dr:
        dr["tire_friction_range"] = [0.5, 0.8]
        dr["vehicle_mass_range"] = [3.0, 4.0]
        dr["action_latency_steps_range"] = [0, 2]
        dr["obs_noise_std_range"] = [0.01, 0.05]
    cfg["env"]["domain_randomization"] = dr
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


def _collect_dr_samples(env: F1tenthEnv, resets: int) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {
        "tire_friction": [],
        "vehicle_mass": [],
        "action_latency_steps": [],
        "obs_noise_std": [],
    }
    for _ in range(resets):
        env.reset()
        metrics = env.extras.get("metrics", {})
        samples["tire_friction"].extend(
            metrics["dr/tire_friction"].detach().cpu().tolist()
        )
        samples["vehicle_mass"].extend(
            metrics["dr/vehicle_mass"].detach().cpu().tolist()
        )
        samples["action_latency_steps"].extend(
            metrics["dr/action_latency_steps"].detach().cpu().tolist()
        )
        samples["obs_noise_std"].extend(
            metrics["dr/obs_noise_std"].detach().cpu().tolist()
        )
    return samples


def test_dr_disabled_matches_baseline(genesis_backend):
    num_envs = 4
    base_tf = float(DEFAULT_CONFIG["env"]["tire_friction"])
    base_mass = 3.74
    env = _make_env(enable_dr=False, num_envs=num_envs)
    try:
        env.reset()
        metrics = env.extras["metrics"]
        assert torch.allclose(
            metrics["dr/tire_friction"],
            torch.full((num_envs,), base_tf),
        )
        assert torch.allclose(
            metrics["dr/vehicle_mass"],
            torch.full((num_envs,), base_mass),
        )
        assert torch.all(
            metrics["dr/action_latency_steps"] == 1.0
        ), "baseline keeps simulate_action_latency=1 step"
        assert torch.allclose(metrics["dr/obs_noise_std"], torch.zeros(num_envs))

        control_interval = int(DEFAULT_CONFIG["env"]["control_interval"])
        for _ in range(40):
            actions = torch.rand(num_envs, 2, device=env.device) * 0.2
            obs, reward, _, _ = env.step(actions, n_steps=control_interval)
            assert torch.isfinite(obs).all()
            assert torch.isfinite(reward).all()
    finally:
        env.close()


def test_dr_enabled_samples_vary_and_respect_bounds(genesis_backend):
    num_envs = 4
    env = _make_env(enable_dr=True, num_envs=num_envs)
    try:
        samples = _collect_dr_samples(env, resets=6)
        for key, vals in samples.items():
            assert len(set(round(v, 4) for v in vals)) > 1, f"{key} did not vary"
        assert all(0.5 <= v <= 0.8 for v in samples["tire_friction"])
        assert all(3.0 <= v <= 4.0 for v in samples["vehicle_mass"])
        assert all(0 <= v <= 2 for v in samples["action_latency_steps"])
        assert all(0.01 <= v <= 0.05 for v in samples["obs_noise_std"])

        control_interval = int(DEFAULT_CONFIG["env"]["control_interval"])
        env.reset()
        for _ in range(120):
            actions = torch.rand(num_envs, 2, device=env.device) * 0.4
            obs, reward, _, _ = env.step(actions, n_steps=control_interval)
            assert torch.isfinite(obs).all()
            assert torch.isfinite(reward).all()
    finally:
        env.close()
