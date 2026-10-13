"""Host tests for gym certify plant and eval_full_lap net-progress laps."""

from __future__ import annotations

import math

from f1tenth_rl_agent.gym_certify import (
    PROGRESS_EPS_M,
    NetLapCounter,
    WarpLongitudinalParams,
    WarpLongitudinalPlant,
    wrapped_delta_s,
)


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
