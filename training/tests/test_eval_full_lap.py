"""Full-lap eval: progress-from-spawn metric and a real zero-action rollout."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from config import DEFAULT_CONFIG
from eval_full_lap import (
    classify_terminations,
    even_spawn_s,
    evaluate_policy_on_env,
    load_actor_bundle,
    pose_at_s,
    resolve_eval_config,
    spawn_speed_mps,
    summarize_episodes,
    wandb_metrics,
    wrapped_delta_s,
)
from f1tenth_env.utils import build_warp_track_data

_OVERNIGHT_CKPT = Path(
    "/home/ubuntu/projects/F1tenth-overnight/outputs/runs/"
    "overnight-20261012/checkpoints/policy_320000000.pt"
)

try:
    import warp  # noqa: F401
    HAS_WARP = True
except ImportError:
    HAS_WARP = False


class ZeroPolicy:
    def __call__(self, obs, deterministic=True, with_logprob=False):
        del deterministic, with_logprob
        zeros = torch.zeros(obs.shape[0], 2, device=obs.device, dtype=torch.float32)
        return zeros, None


def test_wrapped_delta_s_wraps_forward_across_finish_line():
    assert wrapped_delta_s(5.0, 95.0, 100.0) == pytest.approx(10.0)
    assert wrapped_delta_s(95.0, 5.0, 100.0) == pytest.approx(-10.0)
    assert wrapped_delta_s(50.0, 40.0, 100.0) == pytest.approx(10.0)


def test_progress_from_spawn_counts_full_lap_after_wrap():
    length = 100.0
    spawn_s = 90.0
    samples = [90.0, 95.0, 99.0, 2.0, 40.0, 80.0, 91.0]
    progress = 0.0
    prev = spawn_s
    lap_step = None
    for step, s_now in enumerate(samples[1:], start=1):
        progress += wrapped_delta_s(s_now, prev, length)
        prev = s_now
        if lap_step is None and progress >= length:
            lap_step = step
    assert progress == pytest.approx(101.0)
    assert lap_step == 6


def test_even_spawn_s_spreads_full_lap_including_origin():
    s = even_spawn_s(80.0, 8)
    assert s.shape == (8,)
    assert s[0] == pytest.approx(0.0)
    assert s[-1] == pytest.approx(70.0)
    assert np.allclose(np.diff(s), 10.0)
    assert s.max() < 80.0


def test_pose_at_s_circle_quarter_turns():
    n = 64
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    centerline = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
    width = np.full(n, 1.0, dtype=np.float32)
    track = build_warp_track_data(
        {"centerline": centerline, "w_tr_left": width, "w_tr_right": width}
    )
    origin, yaw0 = pose_at_s(track, 0.0)
    assert origin[0] == pytest.approx(1.0, abs=1e-3)
    assert origin[1] == pytest.approx(0.0, abs=1e-3)
    assert yaw0 == pytest.approx(math.pi / 2.0, abs=0.12)
    quarter, yaw_q = pose_at_s(track, 0.25 * track.length)
    assert quarter[0] == pytest.approx(0.0, abs=5e-2)
    assert quarter[1] == pytest.approx(1.0, abs=5e-2)
    assert yaw_q == pytest.approx(math.pi, abs=0.12)


def test_spawn_speed_standing_vs_flying():
    assert spawn_speed_mps({"reset_stationary_probability": 1.0}) == 0.0
    flying = spawn_speed_mps(
        {"reset_stationary_probability": 0.0, "expected_lap_speed_mps": 3.5}
    )
    assert flying == pytest.approx(3.5)


def test_summarize_episodes_rates_and_lap_times():
    episodes = [
        {
            "spawn_s": 0.0,
            "progress_m": 70.0,
            "full_lap": True,
            "lap_time_s": 20.0,
            "crash": False,
            "timeout": True,
            "crash_reason": None,
            "steps": 400,
        },
        {
            "spawn_s": 10.0,
            "progress_m": 12.0,
            "full_lap": False,
            "lap_time_s": None,
            "crash": True,
            "timeout": False,
            "crash_reason": "out_of_bounds",
            "steps": 40,
        },
        {
            "spawn_s": 20.0,
            "progress_m": 80.0,
            "full_lap": True,
            "lap_time_s": 18.0,
            "crash": True,
            "timeout": False,
            "crash_reason": "out_of_bounds",
            "steps": 500,
        },
        {
            "spawn_s": 30.0,
            "progress_m": 8.0,
            "full_lap": False,
            "lap_time_s": None,
            "crash": True,
            "timeout": False,
            "crash_reason": "not_moving",
            "steps": 30,
        },
    ]
    row = summarize_episodes(
        episodes, track="courtyard_2", track_length_m=70.0, spawn_speed_mps_value=3.5
    )
    assert row["n_episodes"] == 4
    assert row["n_full_laps"] == 2
    assert row["full_lap_completion_rate"] == pytest.approx(0.5)
    assert row["crash_rate"] == pytest.approx(0.75)
    assert row["timeout_rate"] == pytest.approx(0.25)
    assert row["mean_progress_m"] == pytest.approx(42.5)
    assert row["mean_lap_time_s"] == pytest.approx(19.0)
    assert row["best_lap_time_s"] == pytest.approx(18.0)
    assert row["crash_breakdown"]["out_of_bounds"] == 2
    assert row["crash_breakdown"]["not_moving"] == 1
    metrics = wandb_metrics({"scenarios": {"solo_flying": row}})
    assert metrics["eval/solo_flying/full_lap_completion_rate"] == pytest.approx(0.5)
    assert metrics["eval/solo_flying/best_lap_time_s"] == pytest.approx(18.0)


def test_classify_terminations_crash_beats_timeout():
    done = torch.tensor([True, True, False, True])
    extras = {
        "termination": {
            "time_out": torch.tensor([1.0, 0.0, 0.0, 1.0]),
            "out_of_bounds": torch.tensor([0.0, 1.0, 0.0, 1.0]),
            "not_moving": torch.tensor([0.0, 0.0, 0.0, 0.0]),
            "invalid_state": torch.tensor([0.0, 0.0, 0.0, 0.0]),
        }
    }
    crash, timeout = classify_terminations(done, extras)
    assert crash.tolist() == [False, True, False, True]
    assert timeout.tolist() == [True, False, False, False]


def _tiny_standing_cfg():
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["env"]["track"] = "courtyard_2"
    cfg["env"]["opponent_strategy"] = None
    cfg["env"]["domain_randomization"]["enabled"] = False
    cfg["env"]["reset_stationary_probability"] = 1.0
    cfg["env"]["reset_speed_min_mps"] = 0.0
    cfg["env"]["reset_speed_max_mps"] = 0.0
    cfg["env"]["reset_spawn_yaw_jitter_rad"] = 0.0
    cfg["env"]["episode_length"] = 4.0
    cfg["env"]["term_not_moving_time_s"] = 0.4
    cfg["env"]["stationary_startup_grace_s"] = 0.0
    cfg["env"]["term_heading_error_rad"] = 10.0
    cfg["sensor"]["lidar_map_yaml"] = None
    cfg["sensor"]["lidar_map_sha256"] = None
    return cfg


@pytest.mark.skipif(not _OVERNIGHT_CKPT.is_file(), reason="overnight ckpt absent")
def test_load_overnight_actor_bundle_mean_action():
    cfg, _source = resolve_eval_config(_OVERNIGHT_CKPT, None)
    actor, normalizer, architecture, _payload = load_actor_bundle(
        _OVERNIGHT_CKPT, torch.device("cpu"), cfg
    )
    assert architecture["name"] == "lidar_cnn_gru"
    obs = torch.zeros(2, int(cfg["obs"]["num_actor_obs"]))
    hidden = actor.initial_hidden(2, device=obs.device, dtype=torch.float32)
    action, _, _hidden = actor.step(
        normalizer.normalize(obs),
        hidden,
        deterministic=True,
        with_logprob=False,
    )
    assert action.shape == (2, 2)
    assert torch.isfinite(action).all()
    assert action.abs().max() <= 1.0 + 1e-5


@pytest.mark.skipif(not HAS_WARP, reason="warp-lang not installed")
def test_zero_action_standing_completes_no_laps(warp_runtime):
    del warp_runtime
    from eval_full_lap import _open_env

    cfg = _tiny_standing_cfg()
    env = _open_env(cfg, num_envs=2)
    try:
        row = evaluate_policy_on_env(
            env,
            ZeroPolicy(),
            lambda obs: obs,
            n_spawns=4,
            seed=0,
            with_sensors=False,
            clip_actions=1.0,
            spawn_speed=0.0,
        )
    finally:
        env.close()

    assert row["n_episodes"] == 4
    assert row["full_lap_completion_rate"] == 0.0
    assert row["n_full_laps"] == 0
    assert row["mean_lap_time_s"] is None
    assert row["crash_rate"] == pytest.approx(1.0)
    assert row["mean_progress_m"] < 1.0
    assert row["crash_breakdown"]["not_moving"] == 4
    spawn = [ep["spawn_s"] for ep in row["episodes"]]
    assert spawn[0] == pytest.approx(0.0)
    assert spawn[-1] == pytest.approx(0.75 * row["track_length_m"])


@pytest.mark.skipif(not HAS_WARP, reason="warp-lang not installed")
def test_reset_to_even_spawns_match_requested_s(warp_runtime):
    del warp_runtime
    from eval_full_lap import _open_env

    cfg = _tiny_standing_cfg()
    env = _open_env(cfg, num_envs=4)
    try:
        length = float(env.track_length)
        spawn_s = even_spawn_s(length, 4)
        poses = []
        yaws = []
        for s_m in spawn_s:
            xy, yaw = pose_at_s(env._track_host, float(s_m))
            poses.append(xy)
            yaws.append(yaw)
        env.reset_to(np.stack(poses), np.asarray(yaws), 0.0, seed=0)
        measured = env.extras["metrics"]["s"].detach().cpu().numpy()
        for got, want in zip(measured, spawn_s):
            err = abs(wrapped_delta_s(got, want, length))
            assert err < 0.35, f"spawn s {got} vs {want} (err {err})"
    finally:
        env.close()
