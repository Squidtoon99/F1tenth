"""Real Austin track load: occupancy LiDAR, spawn, step, and progress."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.map_lidar import raycast_distance_field
from config import DEFAULT_CONFIG
from f1tenth_env.utils import (
    episode_length_for_track,
    load_occupancy_distance_field,
    load_track_state,
    track_loop_length,
)
from standalone_trainer import (
    build_config,
    build_env_cfg,
    load_config_patch,
    parse_args,
    resolve_training_scenarios,
)

try:
    import warp  # noqa: F401
    HAS_WARP = True
except ImportError:
    HAS_WARP = False

_TRAINING_ROOT = Path(__file__).resolve().parents[1]
_AUSTIN_CFG = _TRAINING_ROOT / "configs" / "austin_e2e_ppo_multi_scenario.json"
_AUSTIN_LENGTH_M = 421.04198767875545


def _resolve(argv, patch=None):
    args, explicit = parse_args(argv)
    return args, build_config(args, patch=patch, explicit=explicit)


def test_austin_centerline_and_occupancy_assets():
    patch = json.loads(_AUSTIN_CFG.read_text())
    yaml_path = _TRAINING_ROOT / patch["sensor"]["lidar_map_yaml"]
    field = load_occupancy_distance_field(
        yaml_path, expected_sha256=patch["sensor"]["lidar_map_sha256"]
    )
    assert field.resolution == pytest.approx(0.08089)
    assert field.origin[0] == pytest.approx(-21.25772567260448)
    assert field.origin[1] == pytest.approx(-70.80398789934522)
    assert field.variant_count == 1
    assert field.width == 2000
    assert field.height == 2000

    state = load_track_state("Austin", str(_TRAINING_ROOT), torch.device("cpu"))
    length = track_loop_length(state["centerline"])
    assert length == pytest.approx(_AUSTIN_LENGTH_M, rel=1e-6)
    width = state["w_tr_left"] + state["w_tr_right"]
    assert float(width.min()) == pytest.approx(2.2)
    assert float(width.max()) == pytest.approx(2.2)

    xy = np.asarray(state["centerline"], dtype=np.float64)
    pose = (float(xy[0, 0]), float(xy[0, 1]), 0.0)
    angles = np.linspace(-math.pi, math.pi, 36, endpoint=False)
    scan = raycast_distance_field(
        field, pose, angles, range_min=0.06, range_max=30.0
    )
    assert scan.shape == (36,)
    assert np.any(scan < 30.0)
    assert np.all(scan >= 0.06)
    assert np.all(np.isfinite(scan))


def test_austin_pretrain_config_resolves():
    patch, _meta = load_config_patch(str(_AUSTIN_CFG))
    args, cfg = _resolve(
        ["--num-envs", "8", "--config", str(_AUSTIN_CFG)], patch=patch
    )
    del args
    assert cfg["algorithm"] == "ppo"
    assert cfg["env"]["track"] == "Austin"
    assert cfg["env"]["opponent_strategy"] is None
    names = [scenario["name"] for scenario in cfg["scenarios"]]
    assert names == ["solo_flying", "solo_standing", "solo_iv_2026"]
    resolved = resolve_training_scenarios(cfg, 8, str(_TRAINING_ROOT))
    assert [spec["num_envs"] for spec in resolved] == [4, 2, 2]
    assert resolved[0]["cfg"]["env"]["track"] == "Austin"
    assert resolved[0]["cfg"]["sensor"]["lidar_map_yaml"] == "assets/Austin.yaml"
    assert resolved[1]["cfg"]["env"]["reset_stationary_probability"] == 1.0
    assert resolved[2]["cfg"]["env"]["track"] == "IV_2026_SIM"
    assert resolved[2]["cfg"]["sensor"]["lidar_map_yaml"] is None
    assert resolved[0]["cfg"]["env"]["episode_length"] == pytest.approx(
        episode_length_for_track("Austin", str(_TRAINING_ROOT))
    )
    assert resolved[2]["cfg"]["env"]["episode_length"] == pytest.approx(
        episode_length_for_track("IV_2026_SIM", str(_TRAINING_ROOT))
    )
    assert cfg["ppo"]["entropy_coef"] == pytest.approx(0.01)
    assert cfg["ppo"]["actor_lr"] == pytest.approx(DEFAULT_CONFIG["ppo"]["actor_lr"])


@pytest.mark.skipif(not HAS_WARP, reason="warp-lang not installed")
def test_austin_env_spawns_and_steps_with_lidar_and_progress(warp_runtime):
    del warp_runtime
    from f1tenth_env import F1tenthEnv
    from f1tenth_env import runtime as rt

    rt.configure(
        float_dtype=torch.float32,
        int_dtype=torch.int32,
        dev=torch.device("cpu"),
        eps=1e-12,
    )
    patch, _meta = load_config_patch(str(_AUSTIN_CFG))
    args, cfg = _resolve(["--num-envs", "4", "--device", "cpu"], patch=patch)
    del args
    cfg["env"]["domain_randomization"]["enabled"] = False
    cfg["env"]["reset_spawn_yaw_jitter_rad"] = 0.0
    cfg["env"]["episode_length"] = 8.0
    env = F1tenthEnv(
        num_envs=4,
        env_cfg=build_env_cfg(
            cfg,
            launch_strategy="uniform_jittered",
            launch_strategy_data={"num_cars": 4},
        ),
        obs_cfg=cfg["obs"],
        reward_cfg=cfg["reward"],
    )
    try:
        assert env.env_cfg["track"] == "Austin"
        assert env.track_length == pytest.approx(_AUSTIN_LENGTH_M, rel=1e-5)
        obs, extras = env.reset(seed=0, with_sensors=True)
        lidar = obs["lidar"]
        assert lidar.shape[0] == 4
        assert torch.isfinite(lidar).all()
        assert float(lidar.min()) >= 0.06
        assert bool((lidar < 30.0).any())
        s0 = extras["metrics"]["s"].clone()
        assert torch.isfinite(s0).all()
        assert torch.all(s0 >= 0.0)
        assert torch.all(s0 <= env.track_length + 1e-3)

        actions = torch.zeros(4, 2, dtype=torch.float32)
        actions[:, 0] = 0.6
        prev_s = s0
        progress = torch.zeros(4, dtype=torch.float32)
        half = 0.5 * float(env.track_length)
        for _ in range(8):
            obs, reward, done, extras = env.step(
                actions, n_steps=env.control_interval, with_sensors=True
            )
            assert torch.isfinite(obs["lidar"]).all()
            assert torch.isfinite(reward).all()
            s_now = extras["metrics"]["s"]
            ds = s_now - prev_s
            ds = torch.where(ds > half, ds - env.track_length, ds)
            ds = torch.where(ds < -half, ds + env.track_length, ds)
            progress += ds.clamp(min=0.0)
            prev_s = s_now.clone()
        assert float(progress.mean()) > 0.05
        assert bool((obs["lidar"] < 30.0).any())
        assert torch.isfinite(extras["metrics"]["s"]).all()
    finally:
        env.close()
