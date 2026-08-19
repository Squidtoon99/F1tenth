"""Pure tests for current_gate arbitration and validation."""

import math

import pytest

from f1tenth_control.current_gate import (
    REJECT_BRAKE_LIMIT,
    REJECT_DRIVE_LIMIT,
    REJECT_NONE,
    SOURCE_RL,
    SOURCE_SAFE,
    SOURCE_SAFETY,
    SOURCE_TELEOP,
    AckermannInput,
    GateConfig,
    GateState,
    RlInput,
    ackermann_to_physical,
    apply_slew,
    arbitrate,
    safe_brake_command,
    rl_rejection_reason,
    validate_rl_command,
)


def _cfg(**overrides):
    defaults = dict(
        i_drive_max_a=80.0,
        i_brake_max_a=20.0,
        i_brake_safe_a=5.0,
        i_slew_a_per_s=200.0,
        rl_command_timeout_s=0.15,
        teleop_timeout_s=0.2,
        safety_timeout_s=0.1,
        steering_angle_to_servo_gain=-1.2135,
        steering_angle_to_servo_offset=0.4495,
        max_steer=0.33,
    )
    defaults.update(overrides)
    return GateConfig(**defaults)


def _rl(**overrides):
    defaults = dict(
        generation=1,
        received_s=1.0,
        drive_current_a=64.0,
        brake_current_a=0.0,
        servo_position=0.45,
        longitudinal=0.8,
        steering=0.1,
        source=SOURCE_RL,
        observation_stamp_sec=0,
        observation_stamp_nanosec=0,
    )
    defaults.update(overrides)
    return RlInput(**defaults)


def test_arbitrate_priority_safety_over_teleop_rl():
    cfg = _cfg()
    state = GateState()
    now = 1.0
    rl = _rl(received_s=now)
    teleop = AckermannInput(acceleration=0.5, steering_angle=0.1, received_s=now)
    safety = AckermannInput(acceleration=-1.0, steering_angle=0.0, received_s=now)

    cmd = arbitrate(cfg, state, now, rl, teleop, safety)
    assert cmd.source == SOURCE_SAFETY
    assert cmd.brake_current_a == pytest.approx(20.0)
    assert cmd.drive_current_a == pytest.approx(0.0)


def test_arbitrate_priority_teleop_over_rl():
    cfg = _cfg()
    state = GateState()
    now = 1.0
    rl = _rl(received_s=now)
    teleop = AckermannInput(acceleration=0.5, steering_angle=0.1, received_s=now)

    cmd = arbitrate(cfg, state, now, rl, teleop, None)
    assert cmd.source == SOURCE_TELEOP
    assert cmd.drive_current_a == pytest.approx(40.0)
    assert state.last_applied_generation == 0


def test_arbitrate_rl_when_teleop_stale():
    cfg = _cfg()
    state = GateState()
    now = 1.0
    rl = _rl(received_s=now, generation=3)
    teleop = AckermannInput(acceleration=0.5, steering_angle=0.0, received_s=0.5)

    cmd = arbitrate(cfg, state, now, rl, teleop, None)
    assert cmd.source == SOURCE_RL
    assert cmd.drive_current_a == pytest.approx(64.0)
    assert state.last_applied_generation == 3


def test_arbitrate_safe_brake_when_all_stale():
    cfg = _cfg()
    state = GateState()
    now = 1.0
    rl = _rl(received_s=0.0)
    teleop = AckermannInput(acceleration=0.5, steering_angle=0.0, received_s=0.5)

    cmd = arbitrate(cfg, state, now, rl, teleop, None)
    assert cmd.source == SOURCE_SAFE
    assert cmd.brake_current_a == pytest.approx(5.0)


def test_validate_rl_rejects_nonfinite_and_dual_current():
    cfg = _cfg()
    state = GateState()
    now = 1.0
    assert not validate_rl_command(
        _rl(drive_current_a=float("nan")), cfg, state.last_applied_generation, now
    )
    assert not validate_rl_command(
        _rl(drive_current_a=5.0, brake_current_a=2.0),
        cfg,
        state.last_applied_generation,
        now,
    )
    assert not validate_rl_command(
        _rl(source=SOURCE_TELEOP), cfg, state.last_applied_generation, now
    )


def test_validate_rl_rejects_out_of_range_and_stale_generation():
    cfg = _cfg()
    now = 1.0
    assert rl_rejection_reason(
        _rl(generation=6, drive_current_a=80.01), cfg, 5, now
    ) == REJECT_DRIVE_LIMIT
    assert rl_rejection_reason(
        _rl(generation=6, drive_current_a=0.0, brake_current_a=20.01), cfg, 5, now
    ) == REJECT_BRAKE_LIMIT
    assert not validate_rl_command(_rl(generation=5), cfg, 5, now)
    assert not validate_rl_command(
        _rl(generation=6, received_s=0.5), cfg, 5, now
    )


@pytest.mark.parametrize(
    "field,invalid",
    [
        ("i_drive_max_a", float("nan")),
        ("i_drive_max_a", float("inf")),
        ("i_brake_max_a", float("nan")),
        ("i_brake_max_a", float("inf")),
    ],
)
def test_invalid_gate_current_limits_cannot_admit_commands(field, invalid):
    cfg = _cfg(**{field: invalid})

    assert not validate_rl_command(
        _rl(generation=1, drive_current_a=79.0), cfg, 0, 1.0
    )


def test_gate_diagnoses_10a_producer_against_5a_limit():
    cfg = _cfg(i_drive_max_a=5.0, i_brake_max_a=5.0)
    state = GateState()
    rl = _rl(drive_current_a=10.0, received_s=1.0)

    cmd = arbitrate(cfg, state, 1.0, rl, None, None)

    assert cmd.source == SOURCE_SAFE
    assert rl_rejection_reason(rl, cfg, 0, 1.0) == REJECT_DRIVE_LIMIT
    assert state.rl_reject_total == 1
    assert state.rejection_counts[REJECT_DRIVE_LIMIT] == 1
    assert state.last_reject_reason == REJECT_DRIVE_LIMIT


def test_gate_accepts_5a_and_transitions_to_rl_ownership():
    cfg = _cfg(i_drive_max_a=5.0, i_brake_max_a=5.0)
    state = GateState()
    rejected = _rl(generation=1, drive_current_a=10.0, received_s=1.0)
    accepted = _rl(generation=2, drive_current_a=5.0, received_s=1.1)

    assert arbitrate(cfg, state, 1.0, rejected, None, None).source == SOURCE_SAFE
    cmd = arbitrate(cfg, state, 1.1, accepted, None, None)

    assert cmd.source == SOURCE_RL
    assert cmd.drive_current_a == pytest.approx(5.0)
    assert state.last_applied_generation == 2
    assert state.last_reject_reason == REJECT_NONE
    assert state.rl_reject_total == 1
    assert arbitrate(cfg, state, 1.12, accepted, None, None).source == SOURCE_RL
    assert state.rl_reject_total == 1


def test_apply_slew_limits_current_ramp():
    cfg = _cfg(i_slew_a_per_s=100.0)
    state = GateState(last_i_drive=0.0, last_i_brake=0.0, last_pub_s=0.0)
    target = safe_brake_command(cfg)
    target = target.__class__(
        drive_current_a=10.0,
        brake_current_a=0.0,
        servo_position=target.servo_position,
        longitudinal=1.0,
        steering=0.0,
        source=SOURCE_RL,
    )
    cmd = apply_slew(target, state, cfg, now_s=0.05)
    assert cmd.drive_current_a == pytest.approx(5.0, abs=0.01)
    assert cmd.brake_current_a == pytest.approx(0.0)


def test_apply_slew_enforces_drive_brake_exclusivity():
    cfg = _cfg(i_slew_a_per_s=1e6)
    state = GateState(last_i_drive=5.0, last_i_brake=0.0, last_pub_s=0.0)
    target = safe_brake_command(cfg)
    target = target.__class__(
        drive_current_a=3.0,
        brake_current_a=6.0,
        servo_position=target.servo_position,
        longitudinal=-0.6,
        steering=0.0,
        source=SOURCE_TELEOP,
    )
    cmd = apply_slew(target, state, cfg, now_s=0.02)
    assert cmd.drive_current_a == pytest.approx(0.0)
    assert cmd.brake_current_a == pytest.approx(6.0)


def test_ackermann_to_physical_maps_force_and_servo():
    cfg = _cfg()
    cmd = ackermann_to_physical(0.5, 0.165, cfg, SOURCE_TELEOP)
    assert cmd.drive_current_a == pytest.approx(40.0)
    assert cmd.brake_current_a == pytest.approx(0.0)
    assert cmd.servo_position == pytest.approx(
        cfg.steering_angle_to_servo_gain * 0.165 + cfg.steering_angle_to_servo_offset
    )
    assert cmd.steering == pytest.approx(0.5, abs=0.01)


def test_safe_brake_command():
    cfg = _cfg(i_brake_safe_a=4.0, i_brake_max_a=10.0)
    cmd = safe_brake_command(cfg)
    assert cmd.source == SOURCE_SAFE
    assert cmd.drive_current_a == 0.0
    assert cmd.brake_current_a == pytest.approx(4.0)
    assert cmd.longitudinal == pytest.approx(-0.4)
    assert math.isfinite(cmd.servo_position)
