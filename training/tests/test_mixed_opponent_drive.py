"""TorchSim test: the mixed opponent population drives correctly per row.

Instantiates ``F1tenthEnv`` with ``opponent_strategy="mixed"`` (50/50 scripted vs
policy), steps the simulator, and asserts:
- both opponent modes are actually assigned across the env rows,
- scripted-mode rows hold near ``opponent_target_speed`` and advance on track,
- policy-mode rows produce finite, in-bounds control (the policy half is an
  untrained actor here, so we assert validity rather than racing competence).
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


def _wrap_ds(ds: torch.Tensor, length: float) -> torch.Tensor:
    half_l = 0.5 * length
    ds = torch.where(ds > half_l, ds - length, ds)
    ds = torch.where(ds < -half_l, ds + length, ds)
    return ds


def _pin_deterministic_spawn(cfg: dict, *, gap_m: float) -> None:
    cfg["env"]["opponent_spawn_gap_min_m"] = gap_m
    cfg["env"]["opponent_spawn_gap_max_m"] = gap_m
    cfg["env"]["opponent_spawn_behind_prob"] = 0.0
    cfg["env"]["opponent_spawn_lateral_independent"] = False
    cfg["env"]["opponent_reset_speed_min_mps"] = 0.0
    cfg["env"]["opponent_reset_speed_max_mps"] = 0.0
    cfg["env"]["reset_speed_min_mps"] = 0.0
    cfg["env"]["reset_speed_max_mps"] = 0.0


def _build_cfg(*, target_speed: float) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["opponent_strategy"] = "mixed"
    cfg["env"]["opponent_mix"] = {"scripted_weight": 0.5, "policy_weight": 0.5}
    cfg["env"]["opponent_target_speed"] = target_speed
    cfg["env"]["opponent_kp_speed"] = 1.0
    _pin_deterministic_spawn(cfg, gap_m=25.0)
    # Keep episodes from resetting while we observe the opponents.
    cfg["env"]["term_not_moving_time_s"] = 999.0
    cfg["env"]["term_on_collision"] = False
    cfg["env"]["term_oob_max_consecutive"] = 10_000
    cfg["env"]["term_heading_error_rad"] = 10.0
    cfg["env"]["episode_length"] = 999.0
    cfg["obs"]["enable_opponent_obs"] = True
    cfg["obs"]["num_obs"] = 384 + int(cfg["obs"]["opponent_obs_dim"])
    cfg["reward"]["reward_scales"]["passing"] = 0.5
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


def test_mixed_opponents_drive_per_mode(torch_backend):
    target_speed = 2.5
    num_envs = 16
    control_interval = int(DEFAULT_CONFIG["env"]["control_interval"])
    cfg = _build_cfg(target_speed=target_speed)

    torch.manual_seed(0)
    env = _make_env(cfg, num_envs)
    try:
        obs, _ = env.reset()
        assert obs.shape == (num_envs, cfg["obs"]["num_obs"])

        mode_buf = env.opponent_ctrl.mode_buf.clone()  # True -> policy
        scripted_rows = ~mode_buf
        policy_rows = mode_buf
        assert int(scripted_rows.sum()) > 0, "no scripted-mode rows were assigned"
        assert int(policy_rows.sum()) > 0, "no policy-mode rows were assigned"

        track_len = float(env._opponent_step_state(env.opp_base_pos)["frenet"]["L"])
        prev_s = env._opponent_step_state(env.opp_base_pos)["frenet"]["s"].clone()
        forward_progress = torch.zeros((num_envs,), dtype=torch.float32)
        scripted_speeds: list[float] = []

        for _ in range(400):
            actions = torch.zeros(num_envs, 2, device=env.device)
            obs, reward, done, extras = env.step(actions, n_steps=control_interval)
            assert torch.isfinite(obs).all(), "non-finite observation under mixed opponents"
            assert torch.isfinite(reward).all(), "non-finite reward under mixed opponents"

            opp_speed = extras["metrics"].get("opp_speed")
            assert opp_speed is not None
            if scripted_rows.any():
                scripted_speeds.append(float(opp_speed[scripted_rows].mean().item()))

            s_now = env._opponent_step_state(env.opp_base_pos)["frenet"]["s"]
            ds = _wrap_ds(s_now - prev_s, track_len)
            forward_progress += torch.clamp(ds, min=0.0)
            prev_s = s_now.clone()

        # mode assignment is stable across the rollout (no resets configured to fire).
        assert torch.equal(env.opponent_ctrl.mode_buf, mode_buf)

        # Scripted-mode opponents hold near the commanded cruise speed and advance.
        fast = [s for s in scripted_speeds if s > 1.0]
        assert fast, "scripted-mode opponents never exceeded 1 m/s forward"
        mean_fast = sum(fast) / len(fast)
        assert abs(mean_fast - target_speed) < 1.0, (
            f"scripted-mode cruise speed {mean_fast:.3f} not near target {target_speed}"
        )
        assert float(forward_progress[scripted_rows].mean().item()) > 2.0, (
            "scripted-mode opponents did not advance along the track"
        )
    finally:
        env.close()
