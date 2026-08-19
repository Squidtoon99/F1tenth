"""Pure arbitration and validation for the RL current gate (no ROS imports)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from f1tenth_control.drive_math import force_to_motor_currents

SOURCE_SAFE = 0
SOURCE_RL = 1
SOURCE_TELEOP = 2
SOURCE_SAFETY = 3

REJECT_NONE = 0
REJECT_SOURCE = 1
REJECT_STALE = 2
REJECT_GENERATION = 3
REJECT_NONFINITE = 4
REJECT_NEGATIVE_CURRENT = 5
REJECT_DUAL_CURRENT = 6
REJECT_DRIVE_LIMIT = 7
REJECT_BRAKE_LIMIT = 8
REJECT_LONGITUDINAL = 9
REJECT_STEERING = 10
REJECT_REASON_COUNT = 11

GATE_DIAG_APPLIED_SOURCE = 0
GATE_DIAG_CONSECUTIVE_SAFE = 1
GATE_DIAG_RL_REJECT_TOTAL = 2
GATE_DIAG_LAST_REJECT_REASON = 3
GATE_DIAG_REJECT_COUNTS_START = 4
GATE_DIAG_LEN = GATE_DIAG_REJECT_COUNTS_START + REJECT_REASON_COUNT


@dataclass(frozen=True)
class GateConfig:
    i_drive_max_a: float
    i_brake_max_a: float
    i_brake_safe_a: float
    i_slew_a_per_s: float
    rl_command_timeout_s: float
    teleop_timeout_s: float
    safety_timeout_s: float
    steering_angle_to_servo_gain: float
    steering_angle_to_servo_offset: float
    max_steer: float


@dataclass(frozen=True)
class PhysicalCommand:
    drive_current_a: float
    brake_current_a: float
    servo_position: float
    longitudinal: float
    steering: float
    source: int
    observation_stamp_sec: int = 0
    observation_stamp_nanosec: int = 0


@dataclass
class RlInput:
    generation: int
    received_s: float
    drive_current_a: float
    brake_current_a: float
    servo_position: float
    longitudinal: float
    steering: float
    source: int
    observation_stamp_sec: int
    observation_stamp_nanosec: int


@dataclass
class AckermannInput:
    acceleration: float
    steering_angle: float
    received_s: float


@dataclass
class GateState:
    last_applied_generation: int = 0
    last_accepted_received_s: float | None = None
    last_i_drive: float = 0.0
    last_i_brake: float = 0.0
    last_pub_s: float = 0.0
    applied_generation: int = 0
    consecutive_safe_ticks: int = 0
    rl_reject_total: int = 0
    last_reject_reason: int = REJECT_NONE
    rejection_counts: list[int] = field(
        default_factory=lambda: [0] * REJECT_REASON_COUNT
    )
    last_rejection_key: tuple[int, float, int] | None = None


def is_fresh(received_s: float | None, now_s: float, timeout_s: float) -> bool:
    if received_s is None:
        return False
    if timeout_s <= 0.0:
        return False
    return (now_s - received_s) <= timeout_s


def validate_gate_config(cfg: GateConfig) -> None:
    for name in ("i_drive_max_a", "i_brake_max_a", "i_slew_a_per_s"):
        value = float(getattr(cfg, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0 (got {value!r})")
    safe_brake = float(cfg.i_brake_safe_a)
    if (
        not math.isfinite(safe_brake)
        or safe_brake < 0.0
        or safe_brake > cfg.i_brake_max_a
    ):
        raise ValueError(
            "i_brake_safe_a must be finite and within [0, i_brake_max_a] "
            f"(got {safe_brake!r})"
        )


def rl_rejection_reason(
    cmd: RlInput,
    cfg: GateConfig,
    last_accepted_generation: int,
    now_s: float,
) -> int:
    if cmd.source != SOURCE_RL:
        return REJECT_SOURCE
    if not is_fresh(cmd.received_s, now_s, cfg.rl_command_timeout_s):
        return REJECT_STALE
    if cmd.generation <= last_accepted_generation:
        return REJECT_GENERATION
    fields = (
        cmd.drive_current_a,
        cmd.brake_current_a,
        cmd.servo_position,
        cmd.longitudinal,
        cmd.steering,
    )
    if not all(math.isfinite(v) for v in fields):
        return REJECT_NONFINITE
    if cmd.drive_current_a < 0.0 or cmd.brake_current_a < 0.0:
        return REJECT_NEGATIVE_CURRENT
    if cmd.drive_current_a > 0.0 and cmd.brake_current_a > 0.0:
        return REJECT_DUAL_CURRENT
    if cmd.drive_current_a > cfg.i_drive_max_a + 1e-9:
        return REJECT_DRIVE_LIMIT
    if cmd.brake_current_a > cfg.i_brake_max_a + 1e-9:
        return REJECT_BRAKE_LIMIT
    if abs(cmd.longitudinal) > 1.0 + 1e-6:
        return REJECT_LONGITUDINAL
    if abs(cmd.steering) > 1.0 + 1e-6:
        return REJECT_STEERING
    return REJECT_NONE


def validate_rl_command(
    cmd: RlInput,
    cfg: GateConfig,
    last_accepted_generation: int,
    now_s: float,
) -> bool:
    try:
        validate_gate_config(cfg)
    except ValueError:
        return False
    return (
        rl_rejection_reason(cmd, cfg, last_accepted_generation, now_s)
        == REJECT_NONE
    )


def record_rl_rejection(state: GateState, cmd: RlInput, reason: int) -> None:
    if reason == REJECT_NONE:
        state.last_reject_reason = REJECT_NONE
        state.last_rejection_key = None
        return
    key = (cmd.generation, cmd.received_s, reason)
    state.last_reject_reason = reason
    if key == state.last_rejection_key:
        return
    state.last_rejection_key = key
    state.rl_reject_total += 1
    state.rejection_counts[reason] += 1


def validate_ackermann(acceleration: float, steering_angle: float) -> bool:
    return math.isfinite(acceleration) and math.isfinite(steering_angle)


def ackermann_to_physical(
    acceleration: float,
    steering_angle: float,
    cfg: GateConfig,
    source: int,
) -> PhysicalCommand:
    longitudinal = max(-1.0, min(1.0, float(acceleration)))
    steering = float(steering_angle)
    i_drive, i_brake = force_to_motor_currents(
        longitudinal, cfg.i_drive_max_a, cfg.i_brake_max_a
    )
    servo = (
        cfg.steering_angle_to_servo_gain * steering
        + cfg.steering_angle_to_servo_offset
    )
    steer_norm = steering / cfg.max_steer if cfg.max_steer > 0.0 else 0.0
    return PhysicalCommand(
        drive_current_a=i_drive,
        brake_current_a=i_brake,
        servo_position=servo,
        longitudinal=longitudinal,
        steering=max(-1.0, min(1.0, steer_norm)),
        source=source,
    )


def safe_brake_command(cfg: GateConfig) -> PhysicalCommand:
    longitudinal = (
        -cfg.i_brake_safe_a / cfg.i_brake_max_a if cfg.i_brake_max_a > 0.0 else -1.0
    )
    return PhysicalCommand(
        drive_current_a=0.0,
        brake_current_a=cfg.i_brake_safe_a,
        servo_position=cfg.steering_angle_to_servo_offset,
        longitudinal=longitudinal,
        steering=0.0,
        source=SOURCE_SAFE,
    )


def rl_to_physical(cmd: RlInput) -> PhysicalCommand:
    return PhysicalCommand(
        drive_current_a=cmd.drive_current_a,
        brake_current_a=cmd.brake_current_a,
        servo_position=cmd.servo_position,
        longitudinal=cmd.longitudinal,
        steering=cmd.steering,
        source=SOURCE_RL,
        observation_stamp_sec=cmd.observation_stamp_sec,
        observation_stamp_nanosec=cmd.observation_stamp_nanosec,
    )


def arbitrate(
    cfg: GateConfig,
    state: GateState,
    now_s: float,
    rl: RlInput | None,
    teleop: AckermannInput | None,
    safety: AckermannInput | None,
) -> PhysicalCommand:
    if safety is not None and is_fresh(
        safety.received_s, now_s, cfg.safety_timeout_s
    ) and validate_ackermann(safety.acceleration, safety.steering_angle):
        return ackermann_to_physical(
            safety.acceleration,
            safety.steering_angle,
            cfg,
            SOURCE_SAFETY,
        )

    if teleop is not None and is_fresh(
        teleop.received_s, now_s, cfg.teleop_timeout_s
    ) and validate_ackermann(teleop.acceleration, teleop.steering_angle):
        return ackermann_to_physical(
            teleop.acceleration,
            teleop.steering_angle,
            cfg,
            SOURCE_TELEOP,
        )

    if rl is not None:
        if (
            rl.generation == state.last_applied_generation
            and rl.received_s == state.last_accepted_received_s
            and is_fresh(rl.received_s, now_s, cfg.rl_command_timeout_s)
        ):
            return rl_to_physical(rl)
        reason = rl_rejection_reason(
            rl, cfg, state.last_applied_generation, now_s
        )
        if reason == REJECT_NONE:
            record_rl_rejection(state, rl, reason)
            state.last_applied_generation = rl.generation
            state.last_accepted_received_s = rl.received_s
            return rl_to_physical(rl)
        record_rl_rejection(state, rl, reason)

    return safe_brake_command(cfg)


def apply_slew(
    cmd: PhysicalCommand,
    state: GateState,
    cfg: GateConfig,
    now_s: float,
) -> PhysicalCommand:
    dt = now_s - state.last_pub_s
    state.last_pub_s = now_s
    if dt <= 0.0 or not math.isfinite(dt):
        return cmd

    max_step = abs(cfg.i_slew_a_per_s) * dt
    if max_step <= 0.0:
        return cmd

    def step(prev: float, tgt: float) -> float:
        delta = tgt - prev
        if abs(delta) <= max_step:
            return tgt
        return prev + math.copysign(max_step, delta)

    i_drive = step(state.last_i_drive, cmd.drive_current_a)
    i_brake = step(state.last_i_brake, cmd.brake_current_a)
    if i_drive > 0.0 and i_brake > 0.0:
        if cmd.drive_current_a >= cmd.brake_current_a:
            i_brake = 0.0
        else:
            i_drive = 0.0

    state.last_i_drive = i_drive
    state.last_i_brake = i_brake
    return PhysicalCommand(
        drive_current_a=i_drive,
        brake_current_a=i_brake,
        servo_position=cmd.servo_position,
        longitudinal=cmd.longitudinal,
        steering=cmd.steering,
        source=cmd.source,
        observation_stamp_sec=cmd.observation_stamp_sec,
        observation_stamp_nanosec=cmd.observation_stamp_nanosec,
    )


def next_applied_generation(state: GateState) -> int:
    state.applied_generation += 1
    return state.applied_generation
