"""Native Warp training traj vs ROS packing/inference replay (no mocks)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
for path in (
    _ROOT,
    _ROOT / "training",
    _ROOT / "src/racing_rl/f1tenth_rl_agent",
    _ROOT / "libs/f1tenth_policy",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.warp_ros_traj_parity import compare_replay  # noqa: E402

_CKPT = Path(
    "/home/ubuntu/projects/F1tenth-baseline-v5/outputs/runs/"
    "baseline-v5-s42/checkpoints/policy_43520000.pt"
)


@pytest.mark.skipif(not _CKPT.is_file(), reason="baseline-v5 s42 checkpoint missing")
def test_warp_ros_replay_matches_training_obs_and_action(warp_runtime):
    del warp_runtime
    report = compare_replay(_CKPT, steps=24, imu="physics")
    assert report["norm_stats"]["mean_mae"] < 1e-6, report["norm_stats"]
    assert report["norm_stats"]["var_mae"] < 1e-6, report["norm_stats"]
    assert report["first_norm_same_raw_obs"] is None, report["first_norm_same_raw_obs"]
    assert report["lidar_remap"] is None, report["lidar_remap"]
    assert report["first_obs"] is None, report["first_obs"]
    assert report["first_raw_action"] is None, report["first_raw_action"]
    assert report["first_realized"] is None, report["first_realized"]
    assert report["spawn_speed_mps"] == pytest.approx(3.5)
