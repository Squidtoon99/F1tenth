"""TorchSim test: scripted opponent closed-loop speed control.

Instantiates ``F1tenthEnv`` with a scripted opponent, steps the simulator, and
asserts the opponent converges near ``opponent_target_speed`` and advances along
the centerline (not stuck at rest).
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


def _build_cfg(*, num_envs: int, target_speed: float) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["opponent_strategy"] = "scripted"
    cfg["env"]["opponent_target_speed"] = target_speed
    cfg["env"]["opponent_kp_speed"] = 1.0
    _pin_deterministic_spawn(cfg, gap_m=25.0)
    # Keep the ego from ending episodes while the opponent accelerates.
    cfg["env"]["term_not_moving_time_s"] = 999.0
    cfg["env"]["term_on_collision"] = False
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


def test_scripted_opponent_holds_target_speed(torch_backend):
    target_speed = 2.5
    num_envs = 4
    control_interval = int(DEFAULT_CONFIG["env"]["control_interval"])
    cfg = _build_cfg(num_envs=num_envs, target_speed=target_speed)
    env = _make_env(cfg, num_envs)

    try:
        obs, _ = env.reset()
        assert obs.shape == (num_envs, cfg["obs"]["num_obs"])

        track_len = float(env._opponent_step_state(env.opp_base_pos)["frenet"]["L"])
        prev_s = env._opponent_step_state(env.opp_base_pos)["frenet"]["s"].clone()
        opp_speeds: list[float] = []
        forward_progress = torch.zeros((num_envs,), dtype=torch.float32)

        for _ in range(400):
            actions = torch.zeros(num_envs, 2, device=env.device)
            obs, reward, done, extras = env.step(actions, n_steps=control_interval)
            assert torch.isfinite(obs).all()
            assert torch.isfinite(reward).all()

            opp_speed = extras["metrics"].get("opp_speed")
            assert opp_speed is not None
            opp_speeds.append(float(opp_speed.mean().item()))

            s_now = env._opponent_step_state(env.opp_base_pos)["frenet"]["s"]
            ds = _wrap_ds(s_now - prev_s, track_len)
            forward_progress += torch.clamp(ds, min=0.0)
            prev_s = s_now.clone()

        fast_speeds = [s for s in opp_speeds if s > 1.0]
        assert fast_speeds, "opponent never exceeded 1 m/s forward"
        mean_fast = sum(fast_speeds) / len(fast_speeds)
        assert abs(mean_fast - target_speed) < 1.0, (
            f"opponent cruise speed {mean_fast:.3f} not near target {target_speed}"
        )
        assert max(opp_speeds) >= target_speed - 0.3, (
            f"opponent peak speed {max(opp_speeds):.3f} did not approach {target_speed}"
        )
        assert float(forward_progress.mean().item()) > 2.0, (
            "opponent did not advance along track"
        )
    finally:
        env.close()
