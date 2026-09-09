"""Focused tests for sensor-policy bag preprocessing / action replay."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
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
from analysis import sensor_policy_bag as bag_analysis  # noqa: E402
from f1tenth_policy import ObsNormalizer, actor_from_architecture  # noqa: E402
from f1tenth_rl_agent import sensor_interfaces as si  # noqa: E402


def _params() -> dict:
    params = {
        "twist_vx_sign": -1.0,
        "i_drive_max_a": 80.0,
        "i_brake_max_a": 40.0,
        "max_steer": 0.33,
        "imu_accel_to_ms2": 1.0,
        "imu_gyro_to_rads": 1.0,
    }
    for name in ("ax", "ay", "az", "gx", "gy", "gz"):
        params[f"imu_{name}_sign"] = 1.0
        params[f"imu_{name}_bias"] = 0.0
    return params


def test_reconstructed_actor_converts_applied_steering_to_radians():
    diagnostics = np.zeros((1, 1 + si.DIAG_LEN), dtype=float)
    diagnostics[0, 0] = 1.0
    diagnostics[0, 1 + si.DIAG_VALID_TICK] = 1.0
    diagnostics[0, 1 + si.DIAG_APPLIED_SOURCE] = 1.0
    scan = np.zeros((1, 6 + si.LIDAR_DIM), dtype=float)
    scan[0, :6] = (0.99, 0.99, si.LIDAR_ANGLE_MIN, si.LIDAR_ANGLE_INCREMENT, 0.06, 30.0)
    scan[0, 6:] = 2.0
    series = {
        "/sensor_racer/diagnostics": diagnostics,
        "/rl/actuator/desired": np.array([[1.0, 1.0, 0.99, 0, 0, 0, 0, 0, 1]]),
        "/scan": scan,
        "/sensors/imu/raw": np.array([[0.99, 0, 0, si.GRAVITY_MS2, 0, 0, 0]]),
        "/odom": np.array([[0.99, -2.0]]),
        "/rl/actuator/applied": np.array([[0.99, 1, 0.99, 40, 0, 0, 0.5, 0.5, 1]]),
    }

    synchronized = bag_analysis.synchronize_observations(series, _params())

    assert synchronized["observations"][0, si.STEER_T] == pytest.approx(0.165)


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


def test_pose_free_lidar_gate_accepts_matching_distributions_and_rejects_dropouts():
    rng = np.random.default_rng(0)
    real = rng.uniform(0.5, 8.0, size=(240, 120)).astype(np.float32)
    simulated = np.clip(real + 0.03, 0.02, 30.0)

    passing = bag_analysis.evaluate_pose_free_lidar_gate(
        real,
        simulated,
        quantile_tolerance_m=0.10,
        temporal_tolerance_m=0.10,
        max_range_rate_tolerance=0.02,
        dropout_run_tolerance_beams=4.0,
    )
    dropout_sim = simulated.copy()
    dropout_sim[:, 40:64] = 30.0
    failing = bag_analysis.evaluate_pose_free_lidar_gate(
        real,
        dropout_sim,
        quantile_tolerance_m=0.10,
        temporal_tolerance_m=0.10,
        max_range_rate_tolerance=0.02,
        dropout_run_tolerance_beams=4.0,
    )

    assert passing.passed
    assert passing.passing_beam_fraction == 1.0
    assert not failing.passed
    assert failing.passing_beam_fraction < 0.95
    assert failing.dropout_run_p95_difference_beams >= 20.0


def test_pose_free_lidar_gate_matches_short_real_segment_to_simulated_support():
    rng = np.random.default_rng(4)
    simulated = rng.uniform(0.5, 12.0, size=(300, 90)).astype(np.float32)
    real = simulated[80:120].copy()

    result = bag_analysis.evaluate_pose_free_lidar_gate(
        real,
        simulated,
        quantile_tolerance_m=0.01,
        temporal_tolerance_m=0.01,
        frame_median_tolerance_m=0.01,
    )

    assert result.passed
    assert result.supported_frame_fraction == 1.0


def test_pose_free_support_prefers_matching_max_range_openings():
    rng = np.random.default_rng(3)
    simulated = rng.uniform(0.5, 8.0, size=(80, 90)).astype(np.float32)
    real = np.repeat(simulated[0][None, :], 24, axis=0)
    real[:, 20:50] = 30.0
    simulated[7] = real[0]

    indices, _ = bag_analysis.match_simulated_support(real, simulated)

    assert set(indices.tolist()) == {7}


def test_actor_gate_requires_all_non_lidar_and_99_percent_of_1097_dimensions():
    frames = 80
    real = np.zeros((frames, si.NUM_OBS), dtype=np.float32)
    simulated = np.zeros_like(real)
    state = np.linspace(-0.9, 0.9, frames, dtype=np.float32)
    for observations in (real, simulated):
        observations[:, si.VESC_SPEED] = 4.0 + state
        observations[:, si.THROTTLE_CURRENT] = state
        observations[:, si.STEER_T] = 0.2 * state

    passing = bag_analysis.evaluate_actor_distribution_gate(real, simulated)
    bad_proprio = simulated.copy()
    bad_proprio[:, si.STEER_DELTA2] += 1.0
    failing_proprio = bag_analysis.evaluate_actor_distribution_gate(
        real, bad_proprio
    )
    bad_lidar = simulated.copy()
    bad_lidar[:, :20] += 2.0
    failing_lidar = bag_analysis.evaluate_actor_distribution_gate(real, bad_lidar)

    assert passing.passed
    assert passing.passing_dimension_fraction == 1.0
    assert not failing_proprio.passed
    assert failing_proprio.non_lidar_passing_fraction < 1.0
    assert not failing_lidar.passed
    assert failing_lidar.passing_dimension_fraction < 0.99


def test_actor_gate_compares_observations_at_matching_operating_states():
    frames = 80
    state = np.linspace(-0.9, 0.9, frames, dtype=np.float32)
    real = np.zeros((frames, si.NUM_OBS), dtype=np.float32)
    real[:, si.VESC_SPEED] = 4.0 + state
    real[:, si.THROTTLE_CURRENT] = state
    real[:, si.STEER_T] = 0.2 * state
    real[:, :20] = state[:, None]

    shuffled = real[::-1].copy()
    mismatched = real.copy()
    mismatched[:, :20] = state[::-1, None]

    assert bag_analysis.evaluate_actor_distribution_gate(real, shuffled).passed
    assert not bag_analysis.evaluate_actor_distribution_gate(real, mismatched).passed


def test_actor_gate_compares_conditional_distributions_for_tied_states():
    frames = 80
    real = np.zeros((frames, si.NUM_OBS), dtype=np.float32)
    real[:, :20] = np.linspace(1.0, 2.0, frames, dtype=np.float32)[:, None]
    simulated = real[::-1].copy()

    assert bag_analysis.evaluate_actor_distribution_gate(real, simulated).passed


def test_export_reconstructed_observations_writes_npz_with_metadata(tmp_path):
    observations = np.zeros((24, si.NUM_OBS), dtype=np.float32)
    observations[:, si.VESC_SPEED] = 2.0
    path = tmp_path / "real_actor.npz"

    bag_analysis.save_observation_npz(
        path,
        observations=observations,
        checkpoint_sha256="abc123",
        config="galaxy.json",
        bag="clean_lap_policy_99840000",
    )
    loaded = np.load(path)

    np.testing.assert_array_equal(loaded["observations"], observations)
    assert str(loaded["checkpoint_sha256"]) == "abc123"
    assert str(loaded["bag"]) == "clean_lap_policy_99840000"


def test_actor_parity_diagnostics_report_failing_beams_and_proprio():
    frames = 80
    real = np.zeros((frames, si.NUM_OBS), dtype=np.float32)
    simulated = np.zeros_like(real)
    state = np.linspace(-0.9, 0.9, frames, dtype=np.float32)
    for observations in (real, simulated):
        observations[:, si.VESC_SPEED] = 4.0 + state
        observations[:, si.THROTTLE_CURRENT] = state
        observations[:, si.STEER_T] = 0.2 * state
    real[:, 10:40] = 1.0
    simulated[:, 10:40] = 8.0
    real[:, si.STEER_DELTA2] = 0.4
    simulated[:, si.STEER_DELTA2] = 0.0

    report = bag_analysis.build_parity_report(
        real,
        simulated,
        bag="clean_lap_policy_99840000",
        checkpoint_sha256="abc123",
    )
    diagnostics = report["diagnostics"]

    assert report["bag"] == "clean_lap_policy_99840000"
    assert report["actor"]["passed"] is False
    assert "steer_delta2" in diagnostics["failing_proprio"]
    assert diagnostics["longest_failing_beam_run"] >= 20
    assert diagnostics["failing_beam_sectors"][0]["n_beams"] >= 20
    assert diagnostics["lidar_residual_direction"] == "real_shorter"
    assert diagnostics["operating_state"]["speed"]["real_max"] == pytest.approx(4.9)


def test_support_ray_hits_attribute_openings_to_unknown_cells():
    from analysis.map_lidar import raycast_distance_field
    from f1tenth_env.utils import CorridorDistanceData, _euclidean_distance_transform
    from f1tenth_rl_agent.sensor_preprocessing import beam_angles_rad

    occupied = np.zeros((21, 41), dtype=bool)
    occupied[:, 20] = True
    occupied[10, 20] = False
    unknown = np.zeros_like(occupied)
    unknown[:, 21:] = True
    distance = _euclidean_distance_transform(occupied) * 0.05
    distance[unknown & ~occupied] = -1.0
    field = CorridorDistanceData(
        distance=distance.astype(np.float32).reshape(-1),
        width=41,
        height=21,
        origin=(0.0, 0.0),
        resolution=0.05,
    )
    pose = np.array([[0.5, 0.525, 0.0]], dtype=np.float32)
    scan = raycast_distance_field(
        field, pose[0], beam_angles_rad(), 0.02, 30.0, lidar_offset=(0.0, 0.0, 0.0)
    )
    simulated = np.repeat(scan[None, :], 24, axis=0)
    real = simulated.copy()
    real[:, 530:560] = 30.0
    report = bag_analysis.diagnose_support_ray_hits(
        real,
        simulated,
        np.repeat(pose, 24, axis=0),
        field,
        lidar_offset=(0.0, 0.0, 0.0),
    )
    assert report["matched_unique_poses"] == 1
    assert report["real_opening_beams"] > 0
    assert 0.0 <= report["real_opening_hits_wall_fraction"] <= 1.0
