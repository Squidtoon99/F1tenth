"""Regression tests for finish-line lap counting (lap_count / laps_completed)."""

from __future__ import annotations

import copy

import pytest
import torch

from f1tenth_env import runtime as rt
from f1tenth_env.rewards import ensure_progress_delta

try:
    import warp  # noqa: F401
    HAS_WARP = True
except ImportError:
    HAS_WARP = False


def _configure_runtime():
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )


def _lap_step_state(s, length=100.0):
    return {
        "frenet": {
            "L": torch.tensor(length, dtype=torch.float32),
            "s": torch.tensor([s], dtype=torch.float32),
        }
    }


def _reward_cfg():
    return {"progress_max_step_frac": 0.05}


def _reward_state():
    return {"prev_step_counter": None, "prev_s": None}


def test_finish_line_cross_semantics():
    _configure_runtime()
    reward_state = _reward_state()
    lap_count_buf = torch.zeros(1, dtype=torch.int32)
    cfg = _reward_cfg()

    ensure_progress_delta(
        _lap_step_state(96.0),
        torch.tensor([5], dtype=torch.int32),
        cfg,
        reward_state,
        lap_count_buf,
    )
    ensure_progress_delta(
        _lap_step_state(4.0),
        torch.tensor([6], dtype=torch.int32),
        cfg,
        reward_state,
        lap_count_buf,
    )

    assert int(lap_count_buf[0].item()) == 1
    assert bool(reward_state["last_lap_cross"][0])


def test_mid_track_step_semantics():
    _configure_runtime()
    reward_state = _reward_state()
    lap_count_buf = torch.zeros(1, dtype=torch.int32)
    cfg = _reward_cfg()

    ensure_progress_delta(
        _lap_step_state(50.0),
        torch.tensor([5], dtype=torch.int32),
        cfg,
        reward_state,
        lap_count_buf,
    )
    ensure_progress_delta(
        _lap_step_state(50.5),
        torch.tensor([6], dtype=torch.int32),
        cfg,
        reward_state,
        lap_count_buf,
    )

    assert int(lap_count_buf[0].item()) == 0
    assert not bool(reward_state["last_lap_cross"][0])


def test_reset_step_semantics():
    _configure_runtime()
    reward_state = _reward_state()
    lap_count_buf = torch.zeros(1, dtype=torch.int32)
    cfg = _reward_cfg()

    ensure_progress_delta(
        _lap_step_state(96.0),
        torch.tensor([5], dtype=torch.int32),
        cfg,
        reward_state,
        lap_count_buf,
    )
    ensure_progress_delta(
        _lap_step_state(4.0),
        torch.tensor([6], dtype=torch.int32),
        cfg,
        reward_state,
        lap_count_buf,
    )
    assert int(lap_count_buf[0].item()) == 1

    ensure_progress_delta(
        _lap_step_state(4.0),
        torch.tensor([1], dtype=torch.int32),
        cfg,
        reward_state,
        lap_count_buf,
    )

    assert int(lap_count_buf[0].item()) == 1
    assert not bool(reward_state["last_lap_cross"][0])


@pytest.mark.skipif(not HAS_WARP, reason="warp-lang not installed")
def test_warp_finish_line_cross_increments_metrics():
    from config import DEFAULT_CONFIG
    from f1tenth_env import F1tenthEnv

    _configure_runtime()
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["domain_randomization"]["enabled"] = False
    cfg["env"]["reset_speed_min_mps"] = 0.0
    cfg["env"]["reset_speed_max_mps"] = 0.0
    env = F1tenthEnv(
        num_envs=1,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )
    try:
        cl = env.track_state["centerline"]
        start_pose = torch.tensor(cl[0], dtype=torch.float32)
        env.reset_to(start_pose, 0.0, 0.0, seed=0)
        env._env.tensor["prev_s"][0] = 0.95 * env.track_length

        action = torch.zeros(1, 2, dtype=torch.float32)
        _, _, _, extras = env.step(action, n_steps=env.control_interval)
        metrics = extras["metrics"]

        assert int(metrics["lap_count"][0].item()) == 1
        assert float(metrics["laps_completed"][0].item()) == 1.0
    finally:
        env.close()


@pytest.mark.skipif(not HAS_WARP, reason="warp-lang not installed")
def test_warp_mid_track_step_does_not_increment_laps():
    from config import DEFAULT_CONFIG
    from f1tenth_env import F1tenthEnv

    _configure_runtime()
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["domain_randomization"]["enabled"] = False
    cfg["env"]["reset_speed_min_mps"] = 0.0
    cfg["env"]["reset_speed_max_mps"] = 0.0
    env = F1tenthEnv(
        num_envs=1,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )
    try:
        cl = env.track_state["centerline"]
        mid_pose = torch.tensor(cl[len(cl) // 2], dtype=torch.float32)
        env.reset_to(mid_pose, 0.0, 0.0, seed=0)
        length = env.track_length
        current_s = float(env.extras["metrics"]["s"][0].item())
        env._env.tensor["prev_s"][0] = current_s - 0.5

        action = torch.zeros(1, 2, dtype=torch.float32)
        _, _, _, extras = env.step(action, n_steps=env.control_interval)
        metrics = extras["metrics"]

        assert int(metrics["lap_count"][0].item()) == 0
        assert float(metrics["laps_completed"][0].item()) == 0.0
        assert abs(float(metrics["s"][0].item()) - current_s) < 0.2 * length
    finally:
        env.close()


@pytest.mark.skipif(not HAS_WARP, reason="warp-lang not installed")
def test_warp_reset_does_not_spuriously_increment_laps():
    from config import DEFAULT_CONFIG
    from f1tenth_env import F1tenthEnv

    _configure_runtime()
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["domain_randomization"]["enabled"] = False
    cfg["env"]["reset_speed_min_mps"] = 0.0
    cfg["env"]["reset_speed_max_mps"] = 0.0
    env = F1tenthEnv(
        num_envs=1,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )
    try:
        cl = env.track_state["centerline"]
        start_pose = torch.tensor(cl[0], dtype=torch.float32)
        env.reset_to(start_pose, 0.0, 0.0, seed=0)
        env._env.tensor["prev_s"][0] = 0.95 * env.track_length

        env.reset(seed=1)
        action = torch.zeros(1, 2, dtype=torch.float32)
        _, _, _, extras = env.step(action, n_steps=env.control_interval)
        metrics = extras["metrics"]

        assert int(metrics["lap_count"][0].item()) == 0
        assert float(metrics["laps_completed"][0].item()) == 0.0
    finally:
        env.close()
