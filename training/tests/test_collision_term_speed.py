"""Real-Genesis test: collision termination is gated by closing speed.

Drives the ego full-throttle into a (near-)stationary scripted opponent and checks:
- with a high ``collision_term_speed_mps`` threshold, overlaps still occur (the
  collision penalty fires) but the episode does NOT terminate on them, so the agent
  experiences low-speed contact physics and keeps driving;
- with the threshold at 0.0 (legacy), the same contact ends the episode.
"""

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


def _build_cfg(*, term_speed: float) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["opponent_strategy"] = "scripted"
    cfg["env"]["opponent_target_speed"] = 0.0  # near-stationary pace car to ram
    cfg["env"]["opponent_spawn_gap_m"] = 2.0
    cfg["env"]["term_on_collision"] = True
    cfg["env"]["collision_term_speed_mps"] = term_speed
    cfg["env"]["term_not_moving_time_s"] = 999.0
    cfg["obs"]["enable_opponent_obs"] = True
    cfg["obs"]["num_obs"] = 380 + int(cfg["obs"]["opponent_obs_dim"])
    # Enable the any-collision penalty so we can detect raw overlaps independently
    # of whether they terminate the episode.
    cfg["reward"]["reward_scales"]["collision"] = 1.0
    return cfg


def _make_env(cfg: dict, num_envs: int) -> F1tenthEnv:
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


def _rollout(cfg: dict, num_envs: int, steps: int) -> tuple[bool, float]:
    """Return (overlap_seen, total_collision_terminations) for a chase rollout."""
    control_interval = int(DEFAULT_CONFIG["env"]["control_interval"])
    clip = float(cfg["env"]["clip_actions"])
    env = _make_env(cfg, num_envs)
    try:
        env.reset()
        overlap_seen = False
        term_collisions = 0.0
        for _ in range(steps):
            actions = torch.zeros(num_envs, 2, device=env.device)
            actions[:, 0] = clip  # full throttle, hold heading -> rear-end the pace car
            _, _, _, extras = env.step(actions, n_steps=control_interval)
            terms = extras.get("rewards", {}).get("terms", {})
            if "collision" in terms and bool((terms["collision"] < 0).any()):
                overlap_seen = True
            coll = extras.get("termination", {}).get("collision")
            if coll is not None:
                term_collisions += float(coll.sum().item())
        return overlap_seen, term_collisions
    finally:
        env.close()


@pytest.fixture(scope="module")
def genesis_backend():
    if not gs._initialized:
        gs.init(backend=gs.cpu, precision="32", logging_level="warning")
    return gs


def test_high_threshold_does_not_terminate_on_contact(genesis_backend):
    torch.manual_seed(0)
    cfg = _build_cfg(term_speed=100.0)
    overlap_seen, term_collisions = _rollout(cfg, num_envs=8, steps=200)
    assert overlap_seen, "ego never contacted the pace car; cannot assess gating"
    assert term_collisions == 0.0, (
        "high closing-speed threshold should not terminate on (low-speed) contact"
    )


def test_zero_threshold_terminates_on_contact(genesis_backend):
    torch.manual_seed(0)
    cfg = _build_cfg(term_speed=0.0)
    overlap_seen, term_collisions = _rollout(cfg, num_envs=8, steps=200)
    assert overlap_seen, "ego never contacted the pace car; cannot assess gating"
    assert term_collisions > 0.0, (
        "zero threshold should terminate on any car-car overlap"
    )
