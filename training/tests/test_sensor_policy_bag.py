"""Focused tests for sensor-policy bag preprocessing / action replay."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT,
    ROOT / "training",
    ROOT / "src/racing_rl/f1tenth_rl_agent",
    ROOT / "libs/f1tenth_contract",
    ROOT / "libs/f1tenth_policy",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.sensor_policy_bag import (  # noqa: E402
    policy_actions,
)
from f1tenth_policy import ObsNormalizer, actor_from_architecture  # noqa: E402
from f1tenth_rl_agent import sensor_interfaces as si  # noqa: E402


def _params() -> dict:
    params = {
        "twist_vx_sign": -1.0,
        "imu_accel_to_ms2": 1.0,
        "imu_gyro_to_rads": 1.0,
    }
    for name in ("ax", "ay", "az", "gx", "gy", "gz"):
        params[f"imu_{name}_sign"] = 1.0
        params[f"imu_{name}_bias"] = 0.0
    return params


def test_policy_actions_exercises_real_recurrent_actor_reset_semantics():
    torch.manual_seed(0)
    architecture = {
        "name": "lidar_cnn_gru",
        "version": 1,
        "obs_dim": si.NUM_OBS,
        "lidar_dim": si.LIDAR_DIM,
        "proprio_dim": si.PROPRIO_DIM,
        "conv_channels": [32, 64, 64],
        "kernels": [7, 5, 3],
        "strides": [3, 2, 2],
        "padding": [3, 2, 1],
        "pool": "adaptive_avg_1d",
        "pool_bins": 16,
        "projection_dim": 8,
        "gru_hidden_dim": 8,
        "hidden_layers": [8],
        "activation": "relu",
        "action_dim": 2,
    }
    actor = actor_from_architecture(architecture)
    normalizer = ObsNormalizer(si.NUM_OBS, torch.device("cpu"))
    observations = np.ones((3, si.NUM_OBS), dtype=np.float32)

    actions, hidden_before, hidden_after = policy_actions(
        actor, normalizer, observations, np.array([True, False, True])
    )

    assert actions.shape == (3, 2)
    np.testing.assert_allclose(hidden_before[[0, 2]], 0.0)
    assert hidden_before[1] > 0.0
    assert np.all(hidden_after > 0.0)
