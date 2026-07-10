"""End-to-end integration smoke for F1tenthEnv on the Torch physics backend.

Runs the full env pipeline (reset -> obs -> reward -> termination -> step) with
``physics_backend="torch"`` on a synthetic circular track, asserting the 384-dim
observation contract holds, rewards/terminations stay finite, and a forward
throttle produces forward progress (a learning-relevant signal). No Genesis
scene is built on this path, so it runs headless anywhere.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from f1tenth_env import runtime as rt


def _configure_gs():
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
        "frenet_step_cache": {},
    }


def _make_env(monkeypatch, num_envs=16):
    _configure_gs()
    import f1tenth_env.utils as U
    import f1tenth_env.env as E
    from config import DEFAULT_CONFIG

    monkeypatch.setattr(U, "load_track_state", _fake_track_state, raising=True)
    monkeypatch.setattr(E, "load_track_state", _fake_track_state, raising=True)

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    env_cfg = dict(cfg["env"])
    env_cfg["physics_backend"] = "torch"
    env_cfg.setdefault("launch_strategy", "uniform_jittered")
    env_cfg.setdefault("launch_strategy_data", {"num_cars": num_envs})
    return E.F1tenthEnv(
        num_envs=num_envs, env_cfg=env_cfg, obs_cfg=cfg["obs"], reward_cfg=cfg["reward"]
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
