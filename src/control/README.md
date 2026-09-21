# control/

Algorithmic path-tracking controllers, the drive-command / safety layer, and
force-mode VESC actuation.

- `f1tenth_control/` — pure pursuit (`pp_driver`), follow-the-gap (`gap_driver`),
  hybrid overtaking (`pp_driver_plus`, `pp_ftg_driver`), PID wall-follow
  (`pid_driver`), safety (`safety`), and RL deadman gate (`rl_deadman_gate`).
  The RL `drive_command` node also lives here.
- `vesc_actuator` — exclusive publisher of `/commands/motor/current`,
  `/commands/motor/brake`, and servo position. Maps
  `AckermannDrive.acceleration` to current/brake (ADR 0006). `car.launch.py`
  starves vendored `ackermann_to_vesc` at its input so eRPM never races
  current.

Controllers read pose from `/pf/pose/odom` and publish `/drive`. Raceline CSV
defaults to `/config/maps/raceline.csv` (per-car overlay).
