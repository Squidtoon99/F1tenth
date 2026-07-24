"""Production-kernel vs Torch-mirror reward parity and distribution regression.

The Warp one-kernel ``compute_reward_and_done`` is the production reward path;
``rewards.py`` is the reference mirror. This test drives the real Warp env over a
short 1v1 rollout and, on identical observed state each step, evaluates the Torch
mirror and asserts shared reward terms match within a tight epsilon. Warp-only Lee
steering and off-course terms are covered by direct production-environment tests.
This test also pins a broad deterministic distribution panel and checks that the
global reward scale rescales totals without changing the per-term structure.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
import types

import pytest
import torch

from config import DEFAULT_CONFIG
from f1tenth_env import F1tenthEnv
from f1tenth_env import runtime as rt
from f1tenth_env.utils import build_step_state

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_TERMS = ("progress", "wall_contact", "passing", "collision", "rear_end")


@pytest.fixture(scope="session")
def rewards_mod(real_modules):
    pkg_name = "f1tenth_env_reward_parity_test"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [os.path.join(_REPO_ROOT, "f1tenth_env")]
    sys.modules[pkg_name] = pkg
    sys.modules[f"{pkg_name}.car"] = real_modules.car
    sys.modules[f"{pkg_name}.utils"] = real_modules.utils

    path = os.path.join(_REPO_ROOT, "f1tenth_env", "rewards.py")
    spec = importlib.util.spec_from_file_location(f"{pkg_name}.rewards", path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg_name
    sys.modules[f"{pkg_name}.rewards"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _cfg(*, opponent: bool, global_scale: float = 1.0) -> dict:
    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["domain_randomization"]["enabled"] = False
    cfg["env"]["reset_spawn_yaw_jitter_rad"] = 0.0
    cfg["env"]["reset_stationary_probability"] = 0.0
    cfg["env"]["term_not_moving_time_s"] = 1e6
    cfg["env"]["term_on_collision"] = False
    cfg["reward"]["global_reward_scale"] = global_scale
    if opponent:
        cfg["env"]["opponent_strategy"] = "scripted"
        # Spawn both cars on the centerline so the footprint-aware off-course signal
        # stays clear of the boundary and kernel/mirror parity can be asserted.
        cfg["env"]["reset_spawn_margin_m"] = 5.0
        cfg["env"]["opponent_spawn_gap_min_m"] = 0.9
        cfg["env"]["opponent_spawn_gap_max_m"] = 0.9
        cfg["env"]["opponent_spawn_behind_prob"] = 0.0
        cfg["env"]["opponent_spawn_lateral_independent"] = False
        cfg["env"]["opponent_target_speed"] = 1.0
        cfg["env"]["opponent_target_speed_range"] = [1.0, 1.0]
        cfg["env"]["opponent_reset_speed_min_mps"] = 1.0
        cfg["env"]["opponent_reset_speed_max_mps"] = 1.0
        cfg["env"]["reset_speed_min_mps"] = 2.0
        cfg["env"]["reset_speed_max_mps"] = 2.0
    return cfg


def _make_env(cfg: dict, num_envs: int) -> F1tenthEnv:
    return F1tenthEnv(
        num_envs=num_envs,
        env_cfg=cfg["env"],
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )


def _wrap_clamp(delta: torch.Tensor, length: float) -> torch.Tensor:
    half = 0.5 * length
    delta = torch.where(delta > half, delta - length, delta)
    delta = torch.where(delta < -half, delta + length, delta)
    return delta.clamp(min=-0.1 * length, max=0.1 * length)


def test_kernel_matches_torch_mirror(rewards_mod):
    num_envs = 6
    cfg = _cfg(opponent=True)
    env = _make_env(cfg, num_envs)
    length = float(env.track_length)
    mirror_cfg = dict(cfg["reward"])
    mirror_cfg["control_dt"] = env.control_dt
    scales = mirror_cfg["reward_scales"]

    reward_state = rewards_mod.init_reward_state(scales, num_envs, torch.device("cpu"))

    try:
        env.reset(seed=11)
        prev_opp_s = env.extras["metrics"]["opponent_s"].clone()

        actions = torch.zeros(num_envs, 2)
        actions[:, 0] = 0.4

        contact_seen = False
        compared = 0
        for step in range(16):
            _, _, done, extras = env.step(actions, n_steps=env.control_interval)
            assert not bool(done.any())
            on_track = extras["metrics"]["wall_contact"] == 0

            ego_s = extras["metrics"]["s"].clone()
            opp_s = extras["metrics"]["opponent_s"].clone()
            ego_ds = extras["metrics"]["progress_ds"].clone()
            opp_ds = _wrap_clamp(opp_s - prev_opp_s, length)
            prev_opp_s = opp_s

            state = env.read_state()
            wheels = env.read_wheel_state()
            contact = env._env.tensor["contact"].clone() > 0
            contact_seen = contact_seen or bool(contact.any())
            track_ss = build_step_state(
                base_pos=state["base_pos"],
                track_state=env.track_state,
                device=env.device,
                cache_id="reward_parity",
            )

            big = torch.full((num_envs,), 1.0e6)
            step_state = {
                "progress_ds": ego_ds,
                "opp_progress_ds": opp_ds,
                "opp_s": opp_s,
                "frenet": {
                    "s": ego_s,
                    "L": torch.tensor(length),
                    "seg_dir": track_ss["frenet"]["seg_dir"],
                },
                "boundary": {
                    "ey": torch.zeros(num_envs),
                    "w_l_s": big,
                    "w_r_s": big,
                    "boundary_dist": big,
                },
                "wall_contact": torch.zeros(num_envs, dtype=torch.bool),
                "base_lin_vel": state["base_lin_vel"],
                "tyre_slip": wheels["tyre_slip"],
                "car_collision": contact,
                "ego_vel_world": state["base_vel_world"],
                "opp_vel_world": state["opp_vel_world"],
            }
            episode_steps = extras["metrics"]["episode_steps"].to(torch.int32)
            rewards_mod.compute_rewards(
                step_state, mirror_cfg, reward_state, episode_steps,
                torch.zeros(num_envs, dtype=torch.int32),
            )
            mirror_terms = reward_state["last_reward_terms"]
            warp_terms = extras["rewards"]["terms"]
            if bool(on_track.any()):
                for name in _TERMS:
                    assert torch.allclose(
                        warp_terms[name][on_track],
                        mirror_terms[name][on_track],
                        atol=2e-4,
                        rtol=1e-3,
                    ), (
                        f"step {step} term {name}: "
                        f"warp={warp_terms[name][on_track]} "
                        f"mirror={mirror_terms[name][on_track]}"
                    )
                contact_seen = contact_seen or bool(
                    (contact & on_track).any()
                )
                compared += int(on_track.sum())

        assert compared >= 10
        assert contact_seen, "rollout never produced an on-track car-car contact"
    finally:
        env.close()


def test_distribution_regression_panel():
    """Deterministic broad-quantile panel guarding against gross reward drift."""
    num_envs = 64
    cfg = _cfg(opponent=True)
    env = _make_env(cfg, num_envs)
    try:
        env.reset(seed=7)
        actions = torch.zeros(num_envs, 2)
        actions[:, 0] = 0.7
        actions[:, 1] = 0.05
        totals = []
        for _ in range(40):
            _, reward, _, _ = env.step(actions, n_steps=env.control_interval)
            assert torch.isfinite(reward).all()
            totals.append(reward.clone())
        flat = torch.cat(totals)
        q = torch.quantile(
            flat, torch.tensor([0.05, 0.5, 0.95, 0.999])
        )
        # Close-spawn 1v1 under the canonical wall-contact treatment: car contact
        # still dominates the left tail; progress stays in the upper quantiles.
        assert -20.0 < float(q[0]) < -5.0
        assert -2.0 < float(q[1]) < 0.5
        assert 0.0 < float(q[2]) < 3.0
        assert float(q[3]) < 8.0
        assert -8.0 < float(flat.mean()) < -2.0
    finally:
        env.close()


def test_global_scale_changes_totals_not_proportions():
    num_envs = 4
    k = 3.0
    env_a = _make_env(_cfg(opponent=True, global_scale=1.0), num_envs)
    env_b = _make_env(_cfg(opponent=True, global_scale=k), num_envs)
    try:
        env_a.reset(seed=3)
        env_b.reset(seed=3)
        actions = torch.zeros(num_envs, 2)
        actions[:, 0] = 0.6
        for _ in range(12):
            _, reward_a, _, extras_a = env_a.step(actions, n_steps=env_a.control_interval)
            _, reward_b, _, extras_b = env_b.step(actions, n_steps=env_b.control_interval)
            # The global scale multiplies the summed total uniformly...
            assert torch.allclose(reward_b, k * reward_a, atol=1e-4, rtol=1e-4)
            # ...while the per-term breakdown (pre-global) is identical, so the
            # proportion between any two terms is unchanged.
            for name in _TERMS:
                assert torch.allclose(
                    extras_a["rewards"]["terms"][name],
                    extras_b["rewards"]["terms"][name],
                    atol=1e-5,
                )
    finally:
        env_a.close()
        env_b.close()
