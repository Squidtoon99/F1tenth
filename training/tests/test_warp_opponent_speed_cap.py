"""Warp test: the mixed-opponent policy speed cap is applied in-kernel.

GT Sophy-style mixed opponents give a fraction of the policy-mode rows a per-env
forward-speed cap so they drive their normal line but coast at a slower cruise
(realistically driven yet passable). In the Warp env the mixed population is
resolved in the kernel (per-row ``opponent_mode`` + ``opponent_speed_cap``), so
this exercises that path end to end: the cap is sampled within range on reset and
a capped, full-throttle policy opponent is held near its cap while an uncapped one
accelerates well past it.
"""

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
from f1tenth_env import runtime as rt  # noqa: E402


def _cap_env(
    *,
    cap_prob: float,
    cap_lo: float,
    cap_hi: float,
    ego_cap: float = 0.0,
    num_envs: int = 8,
):
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["opponent_strategy"] = "mixed"
    # scripted_weight 0 -> every row is a policy-mode opponent.
    cfg["env"]["opponent_mix"] = {
        "scripted_weight": 0.0,
        "policy_weight": 1.0,
        "policy_speed_cap_prob": cap_prob,
        "policy_speed_cap_range": [cap_lo, cap_hi],
    }
    cfg["env"]["domain_randomization"]["enabled"] = False
    # Deterministic, non-overlapping spawn; both cars start at rest.
    cfg["env"]["opponent_spawn_gap_min_m"] = 25.0
    cfg["env"]["opponent_spawn_gap_max_m"] = 25.0
    cfg["env"]["opponent_spawn_behind_prob"] = 0.0
    cfg["env"]["opponent_spawn_lateral_independent"] = False
    cfg["env"]["opponent_reset_speed_min_mps"] = 0.0
    cfg["env"]["opponent_reset_speed_max_mps"] = 0.0
    cfg["env"]["reset_speed_min_mps"] = 0.0
    cfg["env"]["reset_speed_max_mps"] = 0.0
    cfg["env"]["ego_speed_cap_mps"] = ego_cap
    # Keep the ego episode from resetting (which would re-roll the opponent).
    cfg["env"]["term_not_moving_time_s"] = 999.0
    cfg["env"]["term_on_collision"] = False
    cfg["env"]["term_oob_max_consecutive"] = 10_000
    cfg["env"]["term_heading_error_rad"] = 10.0
    cfg["env"]["episode_length"] = 999.0
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )


def _load_full_throttle_policy(env):
    """Drive the env's real policy opponent to a deterministic full throttle, zero
    steer, independent of the observation: zero the actor body (ReLU -> 0) and set a
    large throttle mu bias so the tanh squash saturates at +1."""
    actor = env._policy_opponent.actor
    for parameter in actor.parameters():
        parameter.data.zero_()
    actor.mu_layer.bias.data = torch.tensor([10.0, 0.0])
    actor.eval()


def test_speed_cap_sampled_within_range():
    env = _cap_env(cap_prob=1.0, cap_lo=3.5, cap_hi=8.0)
    try:
        env.reset(seed=0)
        cap = env._env.tensor["opponent_speed_cap"]
        assert bool((env.opponent_mode_buf == 1).all()), "expected all policy rows"
        assert torch.isfinite(cap).all()
        assert bool((cap >= 3.5 - 1e-4).all())
        assert bool((cap <= 8.0 + 1e-4).all())
    finally:
        env.close()


def test_speed_cap_zero_probability_is_uncapped():
    env = _cap_env(cap_prob=0.0, cap_lo=3.5, cap_hi=8.0)
    try:
        env.reset(seed=0)
        cap = env._env.tensor["opponent_speed_cap"]
        assert bool((cap > 1e29).all()), "cap_prob=0 must leave every row uncapped"
    finally:
        env.close()


def _opponent_forward_progress(env, steps=50):
    torch.manual_seed(0)
    env.reset(seed=0)
    _load_full_throttle_policy(env)
    track_len = env.track_length
    prev = env.extras["metrics"]["opponent_s"].clone()
    progress = torch.zeros(env.num_envs)
    half = 0.5 * track_len
    for _ in range(steps):
        _, _, _, extras = env.step(
            torch.zeros(env.num_envs, 2), n_steps=env.control_interval
        )
        s = extras["metrics"]["opponent_s"]
        ds = s - prev
        ds = torch.where(ds > half, ds - track_len, ds)
        ds = torch.where(ds < -half, ds + track_len, ds)
        progress += torch.clamp(ds, min=0.0)
        prev = s.clone()
    return float(progress.mean().item())


def test_speed_cap_limits_full_throttle_policy_opponent():
    cap = 2.0
    capped = _cap_env(cap_prob=1.0, cap_lo=cap, cap_hi=cap)
    uncapped = _cap_env(cap_prob=0.0, cap_lo=cap, cap_hi=cap)
    try:
        # Identical full-throttle policy opponents: the capped one coasts whenever it
        # is over its cap, so it cruises slower and covers far less track than the
        # uncapped one over the same rollout.
        capped_progress = _opponent_forward_progress(capped)
        uncapped_progress = _opponent_forward_progress(uncapped)
        assert uncapped_progress > 35.0, uncapped_progress
        assert capped_progress > 1.0, capped_progress
        assert capped_progress < 0.5 * uncapped_progress, (
            capped_progress,
            uncapped_progress,
        )
    finally:
        capped.close()
        uncapped.close()


def _ego_tail_speed(env, steps=100):
    env.reset(seed=0)
    action = torch.zeros(env.num_envs, 2)
    action[:, 0] = 1.0
    speeds = []
    for step in range(steps):
        env.step(action, n_steps=env.control_interval)
        if step >= steps // 2:
            speeds.append(env._ego.tensor["vx"].clone())
    return float(torch.stack(speeds).mean().item())


def test_ego_speed_cap_limits_full_throttle_policy():
    cap = 2.0
    capped = _cap_env(cap_prob=0.0, cap_lo=2.0, cap_hi=2.0, ego_cap=cap)
    uncapped = _cap_env(cap_prob=0.0, cap_lo=2.0, cap_hi=2.0)
    try:
        capped_speed = _ego_tail_speed(capped)
        uncapped_speed = _ego_tail_speed(uncapped)
        assert capped_speed <= cap + 0.25, capped_speed
        assert uncapped_speed > cap + 0.5, uncapped_speed
    finally:
        capped.close()
        uncapped.close()
