"""Direct Warp tests for Lee et al. steering-change / history under delta mode.

Drives the real Warp environment with deterministic delta-steer sequences and
compares emitted terms to independently calculated Lee values on realized
wheel angles. Torch rewards.py is intentionally not exercised — these terms
are Warp-only.
"""

from __future__ import annotations

import copy
import math
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import DEFAULT_CONFIG  # noqa: E402
from f1tenth_env import F1tenthEnv  # noqa: E402

_C_S = 182.883569
_C_O = 0.034
_C_D = 0.014
_LAMBDA_S = 0.25
_LAMBDA_H = 0.5
_MAX_STEER = 0.33
_DELTA_MAX = math.pi / 60.0


def _lee_terms_from_thetas(
    thetas: list[float],
    *,
    lambda_s: float = _LAMBDA_S,
    lambda_h: float = _LAMBDA_H,
    c_s: float = _C_S,
    c_o: float = _C_O,
    c_d: float = _C_D,
) -> list[tuple[float, float]]:
    """Lee steering terms from a realized absolute wheel-angle sequence (rad)."""
    prev_theta = 0.0
    prev_delta = 0.0
    out: list[tuple[float, float]] = []
    for theta in thetas:
        delta = float(theta) - prev_theta
        r_s = -lambda_s * abs(delta)
        m_t = (
            abs(delta) > c_d
            and abs(prev_delta) > c_d
            and delta * prev_delta < 0.0
        )
        if m_t:
            delta_sum = abs(delta) + abs(prev_delta)
            r_h = -lambda_h * (1.0 + math.exp(-c_s * (delta_sum - c_o)))
        else:
            r_h = 0.0
        out.append((r_s, r_h))
        prev_theta = float(theta)
        prev_delta = delta
    return out


def _realized_thetas(delta_actions: list[float]) -> list[float]:
    theta = 0.0
    thetas: list[float] = []
    for action in delta_actions:
        clipped = max(-1.0, min(1.0, float(action)))
        theta = max(
            -_MAX_STEER,
            min(_MAX_STEER, theta + clipped * _DELTA_MAX),
        )
        thetas.append(theta)
    return thetas


def _build_cfg(*, latency_steps: int = 0) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["domain_randomization"]["enabled"] = False
    cfg["env"]["reset_spawn_yaw_jitter_rad"] = 0.0
    cfg["env"]["term_not_moving_time_s"] = 1e6
    cfg["env"]["reset_speed_min_mps"] = 0.0
    cfg["env"]["reset_speed_max_mps"] = 0.0
    cfg["env"]["opponent_strategy"] = "none"
    cfg["env"]["delta_max"] = _MAX_STEER
    cfg["env"]["steering_action_mode"] = "delta"
    cfg["env"]["steering_delta_max_rad"] = _DELTA_MAX
    cfg["env"]["simulate_action_latency"] = bool(latency_steps)
    cfg["obs"]["enable_opponent_obs"] = False
    scales = cfg["reward"]["reward_scales"]
    for key in list(scales):
        scales[key] = 0.0
    scales["steering_change"] = _LAMBDA_S
    scales["steering_history"] = _LAMBDA_H
    cfg["reward"]["steering_history_c_s"] = _C_S
    cfg["reward"]["steering_history_c_o"] = _C_O
    cfg["reward"]["steering_history_c_d"] = _C_D
    cfg["reward"]["global_reward_scale"] = 1.0
    return cfg


def _make_env(cfg: dict, num_envs: int = 1) -> F1tenthEnv:
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )


def _step_steers(env: F1tenthEnv, steers: list[float]) -> list[tuple[float, float]]:
    emitted: list[tuple[float, float]] = []
    for steer in steers:
        actions = torch.zeros(env.num_envs, 2, device=env.device)
        actions[:, 1] = float(steer)
        _, _, done, extras = env.step(actions, n_steps=env.control_interval)
        assert not bool(done.any()), "episode reset during steering sequence"
        change = float(extras["rewards"]["terms"]["steering_change"][0])
        history = float(extras["rewards"]["terms"]["steering_history"][0])
        emitted.append((change, history))
    return emitted


def _assert_terms_match(
    emitted: list[tuple[float, float]],
    expected: list[tuple[float, float]],
    *,
    atol: float = 1e-5,
) -> None:
    assert len(emitted) == len(expected)
    for step, ((got_s, got_h), (exp_s, exp_h)) in enumerate(
        zip(emitted, expected)
    ):
        assert got_s == pytest.approx(exp_s, abs=atol), (
            f"step {step} steering_change: got={got_s} expected={exp_s}"
        )
        assert got_h == pytest.approx(exp_h, abs=atol), (
            f"step {step} steering_history: got={got_h} expected={exp_h}"
        )


def test_delta_lee_zero_and_constant_command(warp_runtime):
    cfg = _build_cfg()
    env = _make_env(cfg)
    try:
        env.reset(seed=0)
        actions = [0.0, 1.0, 1.0, 1.0, 0.0]
        emitted = _step_steers(env, actions)
        expected = _lee_terms_from_thetas(_realized_thetas(actions))
        _assert_terms_match(emitted, expected)
        assert emitted[1][0] == pytest.approx(-_LAMBDA_S * _DELTA_MAX)
        assert emitted[2][0] == pytest.approx(-_LAMBDA_S * _DELTA_MAX)
        assert emitted[4][0] == pytest.approx(0.0)
        assert all(h == pytest.approx(0.0) for _, h in emitted)
    finally:
        env.close()


def test_delta_lee_same_sign_does_not_activate_history(warp_runtime):
    cfg = _build_cfg()
    env = _make_env(cfg)
    try:
        env.reset(seed=1)
        actions = [1.0, 1.0]
        emitted = _step_steers(env, actions)
        expected = _lee_terms_from_thetas(_realized_thetas(actions))
        _assert_terms_match(emitted, expected)
        assert all(h == pytest.approx(0.0) for _, h in emitted)
        assert emitted[0][0] < 0.0
        assert emitted[1][0] < 0.0
    finally:
        env.close()


def test_delta_lee_reversal_activates_history(warp_runtime):
    cfg = _build_cfg()
    env = _make_env(cfg)
    try:
        env.reset(seed=3)
        # Two full ±Δ steps produce |delta| = 2*Δ above c_d, then reverse.
        actions = [1.0, 1.0, -1.0, -1.0]
        emitted = _step_steers(env, actions)
        thetas = _realized_thetas(actions)
        expected = _lee_terms_from_thetas(thetas)
        _assert_terms_match(emitted, expected)
        assert emitted[0][1] == pytest.approx(0.0)
        assert emitted[2][1] < 0.0
        # Same-sign continuation after the reversal does not re-trigger history.
        assert emitted[3][1] == pytest.approx(0.0)
    finally:
        env.close()


def test_episode_reset_clears_steering_history(warp_runtime):
    cfg = _build_cfg()
    env = _make_env(cfg)
    try:
        env.reset(seed=4)
        warm = _step_steers(env, [1.0, 1.0, -1.0])
        assert warm[2][1] < 0.0

        env.reset(seed=5)
        after = _step_steers(env, [1.0, 1.0, -1.0])
        expected = _lee_terms_from_thetas(_realized_thetas([1.0, 1.0, -1.0]))
        _assert_terms_match(after, expected)
        assert after[0][1] == pytest.approx(0.0)
        assert after[0][0] == pytest.approx(-_LAMBDA_S * _DELTA_MAX)
    finally:
        env.close()


def test_delta_steering_integrates_three_degrees_and_saturates(warp_runtime):
    cfg = _build_cfg()
    env = _make_env(cfg)
    try:
        env.reset(seed=6, with_sensors=True)
        action = torch.tensor([[0.0, 1.0]])
        realized = []
        for _ in range(8):
            env.step(action, n_steps=env.control_interval, with_sensors=True)
            realized.append(float(env._ego.tensor["steer"][0]))
        assert realized[0] == pytest.approx(_DELTA_MAX, abs=1e-6)
        assert realized[1] == pytest.approx(2.0 * _DELTA_MAX, abs=1e-6)
        assert realized[-1] == pytest.approx(_MAX_STEER, abs=1e-6)
        assert float(env.actions[0, 1]) == pytest.approx(1.0)
        assert env.actor_obs_buf[0, 1091] == pytest.approx(_MAX_STEER, abs=1e-6)
    finally:
        env.close()


def test_delta_steering_applies_after_action_latency(warp_runtime):
    cfg = _build_cfg(latency_steps=1)
    env = _make_env(cfg)
    try:
        env.reset(seed=7)
        env.step(torch.tensor([[0.0, 1.0]]), n_steps=env.control_interval)
        assert float(env._ego.tensor["steer"][0]) == pytest.approx(0.0, abs=1e-7)
        env.step(torch.zeros(1, 2), n_steps=env.control_interval)
        assert float(env._ego.tensor["steer"][0]) == pytest.approx(
            _DELTA_MAX, abs=1e-6
        )
    finally:
        env.close()


def test_delta_reset_clears_realized_steering_and_history(warp_runtime):
    cfg = _build_cfg()
    env = _make_env(cfg)
    try:
        env.reset(seed=8)
        env.step(torch.tensor([[0.0, 1.0]]), n_steps=env.control_interval)
        assert float(env._ego.tensor["steer"][0]) > 0.0
        env.reset(seed=9)
        assert float(env._ego.tensor["steer"][0]) == pytest.approx(0.0)
        assert torch.equal(
            env._env.tensor["executed_steer_history"], torch.zeros(1, 4)
        )
    finally:
        env.close()


def test_delta_lee_rewards_use_realized_absolute_steering(warp_runtime):
    cfg = _build_cfg()
    env = _make_env(cfg)
    try:
        env.reset(seed=10)
        actions = [1.0, 1.0, -1.0]
        emitted = _step_steers(env, actions)
        expected = _lee_terms_from_thetas(_realized_thetas(actions))
        _assert_terms_match(emitted, expected)
        history = env._env.tensor["executed_steer_history"][0]
        assert history[0] == pytest.approx(_DELTA_MAX, abs=1e-6)
        assert history[1] == pytest.approx(2.0 * _DELTA_MAX, abs=1e-6)
    finally:
        env.close()
