"""Host tests for gym certify plant and eval_full_lap net-progress laps."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from f1tenth_rl_agent.gym_certify import (
    GYM_LIDAR_TF_XYYAW,
    PROGRESS_EPS_M,
    NetLapCounter,
    WarpLongitudinalParams,
    WarpLongitudinalPlant,
    lidar_frame_pose,
    load_centerline,
    pose_error_xy_yaw,
    project_s,
    spawn_pose_from_centerline,
    summarize_repeat_scores,
    wrapped_delta_s,
    yaw_from_quat,
)

_REPO = Path(__file__).resolve().parents[4]
_AUSTIN_CL = _REPO / "training" / "assets" / "Austin_centerline.csv"
_SIM_SH = _REPO / "tools" / "sim.sh"


def test_wrapped_delta_forward_wrap():
    assert wrapped_delta_s(4.0, 96.0, 100.0) == 8.0
    assert wrapped_delta_s(96.0, 4.0, 100.0) == -8.0
    assert wrapped_delta_s(50.0, 40.0, 100.0) == 10.0


def test_start_line_wrap_is_not_a_lap():
    counter = NetLapCounter(length=100.0)
    counter.update(99.0, False, 0.0)
    counter.update(1.0, False, 1.0)
    assert counter.laps == 0
    assert counter.progress == 2.0


def test_net_progress_of_one_track_length_is_a_lap():
    counter = NetLapCounter(length=100.0)
    t = 0.0
    s = 0.0
    counter.update(s, False, t)
    for _ in range(20):
        t += 1.0
        s = (s + 5.0) % 100.0
        counter.update(s, False, t)
    assert counter.laps == 1
    assert counter.lap_times[0] == 20.0
    assert abs(counter.progress) < 1e-9


def test_crash_clears_progress_so_wrap_is_not_a_lap():
    counter = NetLapCounter(length=100.0)
    counter.update(90.0, False, 0.0)
    counter.update(95.0, False, 1.0)
    assert counter.progress == 5.0
    counter.update(98.0, True, 2.0)
    assert counter.crashes == 1
    assert counter.progress == 0.0
    counter.update(2.0, False, 3.0)
    counter.update(10.0, False, 4.0)
    assert counter.laps == 0
    assert counter.progress == 12.0


def test_teleport_to_spawn_counts_as_crash_not_lap():
    counter = NetLapCounter(length=100.0, max_step_frac=0.1)
    counter.update(40.0, False, 0.0)
    counter.update(45.0, False, 1.0)
    counter.update(0.0, False, 2.0)
    assert counter.crashes == 1
    assert counter.laps == 0
    assert counter.progress == 0.0


def test_two_laps_need_two_track_lengths():
    counter = NetLapCounter(length=50.0)
    s = 0.0
    t = 0.0
    counter.update(s, False, t)
    for _ in range(20):
        t += 0.5
        s = (s + 5.0) % 50.0
        counter.update(s, False, t)
    assert counter.laps == 2
    assert counter.lap_times == [5.0, 5.0]


def test_plus_one_throttle_is_not_five_mps():
    plant = WarpLongitudinalPlant()
    vx = plant.step(0.0, i_drive_cmd=80.0, i_brake_cmd=0.0)
    assert vx < 0.05
    assert vx > 0.0


def test_full_throttle_accel_matches_80a_force():
    params = WarpLongitudinalParams()
    plant = WarpLongitudinalPlant(params)
    expected = 0.0
    i_drive = 0.0
    vx = 0.0
    for _ in range(10):
        i_drive = min(80.0, i_drive + params.i_slew * params.sim_dt)
        force = i_drive * params.f_drive_max / params.i_drive_max
        expected += params.sim_dt * (force / params.mass)
        vx = plant.step(vx, 80.0, 0.0, control_dt=params.sim_dt)
    assert abs(vx - expected) < 1e-9
    assert expected < 0.03


def test_brake_cannot_reverse_through_zero():
    plant = WarpLongitudinalPlant()
    vx = 0.4
    for _ in range(40):
        vx = plant.step(vx, i_drive_cmd=0.0, i_brake_cmd=40.0)
    assert vx >= 0.0
    assert vx < plant.params.v_eps


def test_one_second_full_throttle_is_not_instant_speed_setpoint():
    plant = WarpLongitudinalPlant()
    vx = 0.0
    for _ in range(20):
        vx = plant.step(vx, i_drive_cmd=80.0, i_brake_cmd=0.0)
    assert 4.0 < vx < 8.0
    assert not math.isclose(vx, 5.0, abs_tol=0.05)


def test_austin_centerline_is_about_421m():
    from pathlib import Path

    from f1tenth_rl_agent.gym_certify import load_centerline

    path = (
        Path(__file__).resolve().parents[4]
        / "training"
        / "assets"
        / "Austin_centerline.csv"
    )
    _xy, _cum, length = load_centerline(path)
    assert 420.0 < length < 422.0


def test_progress_eps_matches_eval():
    assert PROGRESS_EPS_M == 1e-4
    counter = NetLapCounter(length=10.0)
    s = 0.0
    t = 0.0
    counter.update(s, False, t)
    for _ in range(20):
        t += 1.0
        s = (s + 0.5) % 10.0
        counter.update(s, False, t)
    assert counter.laps == 1
    assert counter.lap_times[0] == 20.0


def test_austin_spawn_matches_first_centerline_segment():
    x, y, yaw = spawn_pose_from_centerline(_AUSTIN_CL)
    assert x == 0.0
    assert y == 0.0
    assert yaw == math.atan2(-0.2321189023617661, 0.3038214682081728)
    assert abs(yaw + 0.6524) < 1e-3


def test_lidar_frame_pose_shifts_forward_along_heading():
    x, y, yaw = lidar_frame_pose(0.0, 0.0, 0.0, GYM_LIDAR_TF_XYYAW)
    assert x == 0.27
    assert y == 0.0
    assert yaw == 0.0
    x, y, yaw = lidar_frame_pose(1.0, 2.0, math.pi / 2.0, (0.27, 0.0, 0.0))
    assert abs(x - 1.0) < 1e-9
    assert abs(y - 2.27) < 1e-9
    assert abs(yaw - math.pi / 2.0) < 1e-9


def test_gym_scan_patch_uses_lidar_mount_frame():
    text = _SIM_SH.read_text(encoding="utf-8")
    assert "sim._lidar_pose_from_base(pose)" in text
    assert "scan_sims[0].scan(pose" in text
    assert '"lidar_noise_std": "0.0"' in text
    assert '"lidar_base_link_to_lidar_tf": "[0.27, 0.0, 0.0]"' in text
    assert "occupancy_map[unknown] = 205.0" in text
    assert "self.dt[self.map_img == 205] = self.max_range" in text
    assert "paused at spawn" in text
    assert "was_paused = self.sim_paused" in text
    assert "--repeats 3" in text


def test_scripted_centerline_pure_pursuit_accumulates_progress():
    xy, cum, length = load_centerline(_AUSTIN_CL)
    counter = NetLapCounter(length)
    speed = 3.0
    dt = 0.05
    lookahead = 1.2
    wheelbase = 0.33
    x, y, yaw = spawn_pose_from_centerline(_AUSTIN_CL)
    t = 0.0
    s, dist = project_s(x, y, xy, cum)
    assert dist < 0.05
    counter.update(s, False, t)
    for _ in range(200):
        target_s = (s + lookahead) % length
        idx = int(np.searchsorted(cum, target_s, side="right") - 1)
        idx = max(0, min(idx, xy.shape[0] - 2))
        alpha = 0.0
        span = float(cum[idx + 1] - cum[idx])
        if span > 1e-9:
            alpha = (target_s - float(cum[idx])) / span
        tx = float(xy[idx, 0] + alpha * (xy[idx + 1, 0] - xy[idx, 0]))
        ty = float(xy[idx, 1] + alpha * (xy[idx + 1, 1] - xy[idx, 1]))
        dx = tx - x
        dy = ty - y
        heading = math.atan2(dy, dx) - yaw
        heading = math.atan2(math.sin(heading), math.cos(heading))
        steer = math.atan2(2.0 * wheelbase * math.sin(heading), lookahead)
        yaw += (speed / wheelbase) * math.tan(steer) * dt
        x += speed * math.cos(yaw) * dt
        y += speed * math.sin(yaw) * dt
        t += dt
        s, dist = project_s(x, y, xy, cum)
        counter.update(s, dist > 1.2, t)
    progress = counter.laps * length + counter.progress
    assert counter.crashes == 0
    assert progress > 25.0
    assert dist < 1.2


def test_pose_error_is_zero_at_spawn():
    x, y, yaw = spawn_pose_from_centerline(_AUSTIN_CL)
    xy_err, yaw_err = pose_error_xy_yaw(x, y, yaw, x, y, yaw)
    assert xy_err == 0.0
    assert yaw_err == 0.0
    xy_err, yaw_err = pose_error_xy_yaw(6.3, -5.6, yaw, x, y, yaw)
    assert xy_err > 5.0
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    assert yaw_from_quat(0.0, 0.0, qz, qw) == pytest.approx(yaw)


def test_net_lap_clear_drops_laps_and_crashes():
    counter = NetLapCounter(length=100.0)
    counter.update(0.0, False, 0.0)
    counter.update(50.0, True, 1.0)
    assert counter.crashes == 1
    counter.clear()
    assert counter.crashes == 0
    assert counter.laps == 0
    assert counter.prev_s is None


def test_summarize_repeat_scores_reports_mean_and_var():
    scores = [
        {
            "pass": True,
            "progress_m": 10.0,
            "laps": 0,
            "crashes": 1,
            "deadline_misses": 0,
            "max_speed_mps": 2.0,
            "offtrack_samples": 4,
            "desired_hz": 20.0,
        },
        {
            "pass": True,
            "progress_m": 14.0,
            "laps": 0,
            "crashes": 1,
            "deadline_misses": 0,
            "max_speed_mps": 2.4,
            "offtrack_samples": 8,
            "desired_hz": 20.0,
        },
        {
            "pass": False,
            "progress_m": 12.0,
            "laps": 0,
            "crashes": 3,
            "deadline_misses": 0,
            "max_speed_mps": 2.2,
            "offtrack_samples": 6,
            "desired_hz": 20.0,
        },
    ]
    summary = summarize_repeat_scores(scores)
    assert summary["repeats"] == 3
    assert summary["pass"] is False
    assert summary["progress_m_mean"] == pytest.approx(12.0)
    assert summary["progress_m_var"] == pytest.approx(4.0)
    assert summary["deadline_misses_mean"] == 0.0
    assert summary["deadline_misses_var"] == 0.0
    assert summary["crashes_mean"] == pytest.approx(5.0 / 3.0)
