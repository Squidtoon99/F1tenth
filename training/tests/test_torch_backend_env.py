"""End-to-end integration smoke for F1tenthEnv and TorchSim.

Runs the full env pipeline (reset -> obs -> reward -> termination -> step) with
Runs the full pipeline on a synthetic circular track, asserting the 384-dim
observation contract holds, rewards/terminations stay finite, and a forward
throttle produces forward progress (a learning-relevant signal).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from evaluation import deterministic_rollout
from f1tenth_env import runtime as rt
from qrsac import SquashedGaussianMLPActor


def _configure_runtime():
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )


def _fake_track_state(track, workspace_dir, device):
    n = 400
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False).astype(np.float32)
    radius = 8.0
    cl = np.stack([radius * np.cos(th), radius * np.sin(th)], axis=-1).astype(np.float32)
    w = np.full(n, 1.5, np.float32)
    return {
        "centerline": cl,
        "w_tr_left": w,
        "w_tr_right": w,
        "w_tr_left_torch": torch.tensor(w, device=device),
        "w_tr_right_torch": torch.tensor(w, device=device),
        "track_geom_cache": {},
    }


def _make_env(monkeypatch, num_envs=16, opponent=False):
    _configure_runtime()
    torch.manual_seed(0)
    import f1tenth_env.utils as U
    import f1tenth_env.env as E
    from config import DEFAULT_CONFIG

    monkeypatch.setattr(U, "load_track_state", _fake_track_state, raising=True)
    monkeypatch.setattr(E, "load_track_state", _fake_track_state, raising=True)

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    env_cfg = dict(cfg["env"])
    # Deterministic smoke test: start from rest with DR off so a forward throttle
    # produces a clean, monotonic forward-progress signal regardless of test order.
    env_cfg["domain_randomization"] = {
        **cfg["env"]["domain_randomization"], "enabled": False
    }
    env_cfg["reset_speed_min_mps"] = 0.0
    env_cfg["reset_speed_max_mps"] = 0.0
    obs_cfg = copy.deepcopy(cfg["obs"])
    if opponent:
        env_cfg["opponent_strategy"] = "scripted"
        obs_cfg["enable_opponent_obs"] = True
        obs_cfg["num_obs"] += obs_cfg["opponent_obs_dim"]
        cfg["reward"]["reward_scales"]["passing"] = 0.5
        cfg["reward"]["reward_scales"]["collision"] = 1.0
    env_cfg.setdefault("launch_strategy", "uniform_jittered")
    env_cfg.setdefault("launch_strategy_data", {"num_cars": num_envs})
    return E.F1tenthEnv(
        num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=cfg["reward"]
    )


def test_torch_env_end_to_end(monkeypatch):
    env = _make_env(monkeypatch, num_envs=16)
    obs, extras = env.reset()
    assert obs.shape == (16, 384)
    assert torch.isfinite(obs).all()

    rewards = []
    for _ in range(40):
        a = torch.zeros(16, 2)
        a[:, 0] = 0.5
        a[:, 1] = 0.1
        obs, rew, done, extras = env.step(a, n_steps=env.env_cfg["control_interval"])
        assert torch.isfinite(obs).all()
        assert torch.isfinite(rew).all()
        rewards.append(rew.mean().item())

    # Progress reward should be net-positive when driving forward on the track.
    assert np.mean(rewards[-10:]) > np.mean(rewards[:10])
    speed = float(extras["metrics"]["speed_xy"].mean().item())
    assert speed > 0.5


def test_torch_env_reset_is_finite(monkeypatch):
    env = _make_env(monkeypatch, num_envs=8)
    obs, _ = env.reset()
    assert torch.isfinite(obs).all()
    # A masked partial reset should not corrupt state.
    done = torch.zeros(8, dtype=torch.bool)
    done[::2] = True
    env.reset(done)
    assert torch.isfinite(env.obs_buf).all()


@pytest.mark.parametrize("opponent", [False, True])
def test_deterministic_rollout_repeats_trajectory(monkeypatch, opponent):
    env = _make_env(monkeypatch, num_envs=2, opponent=opponent)
    actor = SquashedGaussianMLPActor(
        env.num_obs, 2, [8], nn.ReLU, 1.0
    )
    for parameter in actor.parameters():
        parameter.data.zero_()

    positions = []

    def record(_step, rollout_env, _before, reward, _done, _extras):
        assert torch.isfinite(reward).all()
        positions.append(rollout_env.base_pos.detach().clone())

    first = deterministic_rollout(
        env,
        actor,
        lambda obs: obs,
        num_steps=5,
        control_interval=env.control_interval,
        clip_actions=1.0,
        seed=7,
        callback=record,
    )
    first_positions = torch.stack(positions)
    positions.clear()
    second = deterministic_rollout(
        env,
        actor,
        lambda obs: obs,
        num_steps=5,
        control_interval=env.control_interval,
        clip_actions=1.0,
        seed=7,
        callback=record,
    )

    assert first["finite"]
    assert second["finite"]
    assert torch.equal(first_positions, torch.stack(positions))
