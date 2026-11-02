# control/

Algorithmic path-tracking controllers and the drive-command / safety layer.

- `f1tenth_control/` — pure pursuit (`pp_driver`), follow-the-gap (`gap_driver`),
  hybrid overtaking (`pp_driver_plus`, `pp_ftg_driver`), PID wall-follow
  (`pid_driver`), safety (`safety`), and RL deadman gate (`rl_deadman_gate`).
  The RL `drive_command` node also lives here.
- `vesc_actuator` — exclusive publisher of `/commands/motor/current`,
  `/commands/motor/brake`, and servo position. Maps
  `AckermannDrive.acceleration` in `[-1, 1]` to drive/brake current.
  `car.launch.py` starves vendored `ackermann_to_vesc` so the two never race
  (ADR 0006). Watchdog on stale/non-finite commands: zero drive + `i_brake_safe_a`.

Controllers read pose from `/pf/pose/odom` and publish `/drive`. Raceline CSV
defaults to `/config/maps/raceline.csv` (per-car overlay).
