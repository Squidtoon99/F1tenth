# 0005 — Force-mode VESC actuation (current/brake)

- Status: Superseded by [0006](0006-current-only-vesc-actuation.md)
- Date: 2026-07-12

## Context

The deployed car commands the VESC in closed-loop **speed (ERPM)** mode via
`ackermann_to_vesc` → `/commands/motor/speed`. Negative policy throttle becomes
`speed = 0` (`brake_behavior: stop`), so the car coasts rather than actively
braking. Maximum acceleration/braking for track performance needs direct
**motor current** and **brake current** commands.

Training already supports `throttle_mode: "force"` in
`training/f1tenth_sim/drivetrain.py`, but the on-car stack did not. Existing
speed-trained checkpoints (including the 686M policy) are semantically
incompatible with force-mode actions.

## Decision

1. Keep the 2-D action layout unchanged; redefine longitudinal semantics for
   force mode as normalized force/brake in `[-1, 1]`.
   *Why:* avoids a contract dimension change while making sim/car force
   semantics match.

2. Carry the longitudinal command on `AckermannDrive.acceleration` through the
   existing mux; keep `steering_angle` for steering; set `speed = 0` in force
   mode so the ERPM converter is inert when remapped away.
   *Why:* preserves mux priorities (teleop / deadman / safety) without a new
   message type.

3. Add non-vendored `f1tenth_control/vesc_actuator` as the exclusive publisher of
   `/commands/motor/current`, `/commands/motor/brake`, and servo position when
   `motor_mode:=force`. Remap vendored `ackermann_to_vesc` motor/servo outputs
   away in `car.launch.py` so ERPM and force commands never race.
   *Why:* avoids editing vendored `src/vehicle/**` while enforcing mutual
   exclusion in our node.

4. Default remains `motor_mode: speed` / `vesc_actuator.enabled: false` until
   firmware limits are exported, boxed-wheel calibration completes, and a
   force-trained checkpoint is certified.
   *Why:* no accidental force-mode deployment with speed-trained weights.

5. Watchdog / deadman / non-finite commands request zero drive current and a
   configurable safe brake current (`i_brake_safe_a`).
   *Why:* current-mode faults are higher risk than speed-mode coasts.

## Consequences

- Speed-mode checkpoints must not command the force actuator; retrain with
  `throttle_mode: force` and new `obs_norm`.
- VESC firmware motor/battery/brake limits must be read via VESC Tool (or
  headless `--getMcConf`) before raising `i_*_max` above bench-safe values.
- ROS overlay `current_max` / `brake_max` remain software clips, not hardware
  truth.
- Follow-up: boxed-wheel current/brake bags with `/sensors/core`, fit
  `f_drive_max` / `f_brake_max`, then enable force mode for shakedown.
