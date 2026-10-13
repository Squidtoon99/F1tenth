"""Real-module checks that 10 Hz and 20 Hz control share the same seconds.

Warp tests drive F1tenthEnv at control_interval 20 (10 Hz) and 10 (20 Hz) with
dt-scaled delta-steer limits. Pure config tests pin gamma/lambda horizons,
history windows, and episode timeouts in seconds.
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

from config import (  # noqa: E402
    DEFAULT_CONFIG,
    REFERENCE_CONTROL_DT_S,
    control_dt_s,
    effective_ppo_horizon,
    effective_qrsac_horizon,
    effective_qrsac_sequence,
    history_stride_for_dt,
    steering_delta_max_for_dt,
)
from f1tenth_env import F1tenthEnv  # noqa: E402
from f1tenth_policy.layout import (  # noqa: E402
    HISTORY_STRIDE,
    LONG_HISTORY,
    STEER_T,
    STEER_T1,
    STEER_T2,
    STEERING_DELTA_MAX_RAD,
)


def _cfg_for_interval(control_interval: int) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["control_interval"] = int(control_interval)
    dt = cfg["env"]["sim_dt"] * float(control_interval)
    cfg["env"]["steering_delta_max_rad"] = steering_delta_max_for_dt(dt)
    cfg["env"]["domain_randomization"]["enabled"] = False
    cfg["env"]["reset_spawn_yaw_jitter_rad"] = 0.0
    cfg["env"]["term_not_moving_time_s"] = 1e6
    cfg["env"]["reset_speed_min_mps"] = 3.0
    cfg["env"]["reset_speed_max_mps"] = 3.0
    cfg["env"]["reset_stationary_probability"] = 0.0
    cfg["env"]["reset_post_brake_history_probability"] = 0.0
    cfg["env"]["opponent_strategy"] = "none"
    cfg["obs"]["enable_opponent_obs"] = False
    scales = cfg["reward"]["reward_scales"]
    for key in list(scales):
        scales[key] = 0.0
    scales["progress"] = 1.0
    scales["steering_change"] = 0.5
    cfg["reward"]["global_reward_scale"] = 1.0
    cfg["reward"]["wall_contact_coefficient"] = 0.0
    return cfg


def _make_env(cfg: dict) -> F1tenthEnv:
    return F1tenthEnv(
        num_envs=1,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
        show_viewer=False,
        enable_recording=False,
    )


def _run_fixed_action(env: F1tenthEnv, action, seconds: float):
    steps = int(round(float(seconds) / env.control_dt))
    rewards = []
    for _ in range(steps):
        act = torch.tensor([action], dtype=torch.float32, device=env.device)
        _, reward, done, _ = env.step(act, n_steps=env.control_interval)
        assert not bool(done.any()), "episode reset during dt-invariance rollout"
        rewards.append(float(reward[0]))
    return {
        "reward_sum": sum(rewards),
        "reward_per_s": sum(rewards) / float(seconds),
        "x": float(env._ego.tensor["x"][0]),
        "y": float(env._ego.tensor["y"][0]),
        "yaw": float(env._ego.tensor["yaw"][0]),
        "vx": float(env._ego.tensor["vx"][0]),
        "steer": float(env._ego.tensor["steer_logical"][0]),
        "steps": steps,
    }


def test_helpers_match_10hz_window_and_30deg_s():
    assert steering_delta_max_for_dt(0.1) == pytest.approx(math.pi / 60.0)
    assert steering_delta_max_for_dt(0.05) == pytest.approx(math.pi / 120.0)
    assert steering_delta_max_for_dt(0.05) == pytest.approx(STEERING_DELTA_MAX_RAD)
    assert history_stride_for_dt(0.1) == 1
    assert history_stride_for_dt(0.05) == HISTORY_STRIDE == 2
    assert LONG_HISTORY == 1 + HISTORY_STRIDE


def test_gamma_lambda_horizon_is_invariant_in_seconds():
    cfg20 = copy.deepcopy(DEFAULT_CONFIG)
    cfg10 = copy.deepcopy(DEFAULT_CONFIG)
    cfg10["env"]["control_interval"] = 20
    g20, l20 = effective_ppo_horizon(cfg20)
    g10, l10 = effective_ppo_horizon(cfg10)
    dt20 = control_dt_s(cfg20["env"])
    dt10 = control_dt_s(cfg10["env"])
    assert g20 ** (1.0 / dt20) == pytest.approx(g10 ** (1.0 / dt10), rel=1e-9)
    assert l20 ** (1.0 / dt20) == pytest.approx(l10 ** (1.0 / dt10), rel=1e-9)
    qg20, n20 = effective_qrsac_horizon(cfg20)
    qg10, n10 = effective_qrsac_horizon(cfg10)
    assert n20 * dt20 == pytest.approx(n10 * dt10)
    assert qg20 ** n20 == pytest.approx(qg10 ** n10, rel=1e-9)
    b20, t20 = effective_qrsac_sequence(cfg20)
    b10, t10 = effective_qrsac_sequence(cfg10)
    assert b20 * dt20 == pytest.approx(b10 * dt10)
    assert t20 * dt20 == pytest.approx(t10 * dt10)
    r20 = cfg20["ppo"]["rollout_steps"] * dt20
    r10 = 128 * dt10
    assert r20 == pytest.approx(r10)


def test_timeout_length_is_invariant_in_seconds(warp_runtime):
    results = {}
    for interval in (10, 20):
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["env"]["control_interval"] = interval
        cfg["env"]["steering_delta_max_rad"] = steering_delta_max_for_dt(
            cfg["env"]["sim_dt"] * float(interval)
        )
        cfg["env"]["domain_randomization"]["enabled"] = False
        cfg["env"]["opponent_strategy"] = "none"
        env = _make_env(cfg)
        try:
            results[interval] = env.max_episode_steps * env.control_dt
            cadence = env.control_dt / REFERENCE_CONTROL_DT_S
            assert env._reward_params.collision == pytest.approx(
                float(DEFAULT_CONFIG["reward"]["reward_scales"]["collision"])
                * cadence
            )
            assert env._reward_params.rear_end == pytest.approx(
                float(DEFAULT_CONFIG["reward"]["reward_scales"]["rear_end"])
                * cadence
            )
            assert env._reward_params.steering_c_d == pytest.approx(0.014 * cadence)
            assert env._reward_params.steering_c_o == pytest.approx(0.034 * cadence)
            assert env._reward_params.steering_c_s == pytest.approx(
                182.883569 / cadence
            )
        finally:
            env.close()
    assert results[10] == pytest.approx(results[20])
    assert results[10] == pytest.approx(float(DEFAULT_CONFIG["env"]["episode_length"]))


def test_progress_reward_per_second_matches_at_10_and_20hz(warp_runtime):
    outcomes = {}
    for interval in (10, 20):
        cfg = _cfg_for_interval(interval)
        cfg["reward"]["reward_scales"]["steering_change"] = 0.0
        env = _make_env(cfg)
        try:
            env.reset(seed=0)
            outcomes[interval] = _run_fixed_action(env, [0.4, 0.0], 0.5)
        finally:
            env.close()
    a = outcomes[10]
    b = outcomes[20]
    assert a["reward_per_s"] == pytest.approx(b["reward_per_s"], rel=0.05, abs=0.02)
    assert a["x"] == pytest.approx(b["x"], abs=0.05)
    assert a["y"] == pytest.approx(b["y"], abs=0.05)
    assert a["yaw"] == pytest.approx(b["yaw"], abs=0.05)
    assert a["vx"] == pytest.approx(b["vx"], abs=0.15)


def test_steering_rate_and_change_reward_match_in_seconds(warp_runtime):
    outcomes = {}
    for interval in (10, 20):
        cfg = _cfg_for_interval(interval)
        cfg["reward"]["reward_scales"]["progress"] = 0.0
        env = _make_env(cfg)
        try:
            env.reset(seed=1)
            outcomes[interval] = _run_fixed_action(env, [0.0, 1.0], 0.1)
        finally:
            env.close()
    a = outcomes[10]
    b = outcomes[20]
    expected = math.pi / 60.0
    assert a["steer"] == pytest.approx(expected, abs=1e-5)
    assert b["steer"] == pytest.approx(expected, abs=1e-5)
    assert a["reward_sum"] == pytest.approx(b["reward_sum"], rel=0.02, abs=1e-4)


def test_command_history_covers_0p3s_at_20hz(warp_runtime):
    cfg = _cfg_for_interval(10)
    env = _make_env(cfg)
    try:
        env.reset(seed=2, with_sensors=True)
        delta = float(env.vehicle_params.steering_delta_max)
        action = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=env.device)
        for _ in range(7):
            env.step(action, n_steps=env.control_interval, with_sensors=True)
        obs = env.actor_obs_buf[0]
        stride = int(env._sim_params.history_stride)
        hist = env._env.tensor["executed_steer_history"][0]
        assert stride == 2
        assert obs[STEER_T] == pytest.approx(7.0 * delta, abs=1e-5)
        assert obs[STEER_T1] == pytest.approx(5.0 * delta, abs=1e-5)
        assert obs[STEER_T2] == pytest.approx(3.0 * delta, abs=1e-5)
        assert obs[STEER_T] == pytest.approx(float(hist[0]), abs=1e-5)
        assert obs[STEER_T1] == pytest.approx(float(hist[stride]), abs=1e-5)
        assert stride * env.control_dt == pytest.approx(0.1)
        assert 3 * stride * env.control_dt == pytest.approx(0.3)
    finally:
        env.close()
