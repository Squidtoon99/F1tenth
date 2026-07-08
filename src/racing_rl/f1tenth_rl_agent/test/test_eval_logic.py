"""Pure tests for the episode monitor logic."""

import math

import numpy as np

from f1tenth_rl_agent.eval_logic import EpisodeMonitor, opponent_pose_ahead, sample_centerline_pose


def test_lap_detection_and_time():
    mon = EpisodeMonitor()
    mon.reset(0.0)
    track_len = 100.0
    # drive forward through the lap, sampling progress
    t = 0.0
    for s in range(0, 100, 5):  # 0..95
        ev = mon.update(float(s), track_len, 0.0, 1.5, 1.5, 3.0, t)
        t += 1.0
    assert ev.lap_count == 0
    # wrap: previous progress 0.95 -> now 0.02 triggers a lap
    ev = mon.update(2.0, track_len, 0.0, 1.5, 1.5, 3.0, t)
    assert ev.lap_completed
    assert ev.lap_count == 1
    assert ev.lap_time is not None and ev.lap_time > 0.0


def test_out_of_bounds():
    mon = EpisodeMonitor(oob_margin_m=0.0)
    mon.reset(0.0)
    ev = mon.update(10.0, 100.0, 2.0, 1.5, 1.5, 3.0, 0.1)  # ey beyond left width
    assert ev.oob


def test_sample_centerline_pose_on_track():
    centerline = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    rng = np.random.default_rng(0)
    x, y, yaw = sample_centerline_pose(centerline, rng)
    dists = [math.hypot(x - p[0], y - p[1]) for p in centerline]
    assert min(dists) < 1e-5
    assert -math.pi <= yaw <= math.pi


def test_stuck_detection():
    mon = EpisodeMonitor(stuck_speed_mps=0.2, stuck_timeout_s=2.0)
    mon.reset(0.0)
    ev = mon.update(10.0, 100.0, 0.0, 1.5, 1.5, 0.0, 1.0)
    assert not ev.stuck
    ev = mon.update(10.0, 100.0, 0.0, 1.5, 1.5, 0.0, 3.5)  # >2s without moving
    assert ev.stuck


def test_opponent_pose_ahead_gap():
    centerline = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    ox, oy, _ = opponent_pose_ahead(centerline, 0.0, 0.0, gap_m=2.0)
    assert math.isclose(ox, 2.0, abs_tol=1e-4)
    assert math.isclose(oy, 0.0, abs_tol=1e-4)
