# 0006 — Current-only VESC actuation (supersedes 0005)

- Status: Accepted
- Date: 2026-07-12
- Supersedes: [0005](0005-force-mode-vesc-actuation.md)

## Context

ADR 0005 introduced dual-mode support (`motor_mode: speed|force`) with speed/ERPM
as the default. Track performance requires direct motor/brake current, and keeping
two longitudinal semantics created misconfiguration risk (launch arg vs ROS param)
plus checkpoint incompatibility. Speed-trained checkpoints are intentionally
invalidated.

The installed drivetrain is a Traxxas Velineon 3500 (sensorless brushless, 3500 Kv)
driven by a TRAMPA VESC 6 MKV. VESC FOC **current** control maps phase current to
torque; duty cycle is voltage/PWM and is a poor RL action; ERPM adds an outer
speed PID that hides longitudinal dynamics.

## Decision

1. Normalized `action[0] ∈ [-1, 1]` means drive / coast / brake effort only.
   Simulation maps it to `f_drive_max` / `f_brake_max`; the car maps it to
   `/commands/motor/current` and `/commands/motor/brake`.
2. Remove `throttle_mode` / `motor_mode` switches. Training and deploy always use
   force/current semantics. Remap vendored ERPM publishers away permanently.
3. Bump `policy_format_version` to 2 and reject older checkpoints at
   `policy_inference` load time.
4. Raise the shared control rate to 20 Hz (`CONTROL_HZ`); publish current
   immediately on action arrival; keep a short watchdog (a few control cycles).
5. Deadman / estop / stale commands request zero drive current plus a bounded
   safe brake current.

## Consequences

- Existing speed-trained policies (including 686M) will not load until retrained
  with force semantics and a fresh `obs_norm`.
- `safety` / teleop / deadman must speak acceleration/current, not speed setpoints.
- Safe `i_drive_max_a` / `i_brake_max_a` still require VESC firmware export and
  boxed-wheel calibration before raising above bench values.
