# control/

Algorithmic path-tracking controllers and the drive-command / safety layer.

- `f1tenth_control/` — pure pursuit (`pp_driver`), follow-the-gap (`gap_driver`),
  hybrid overtaking (`pp_driver_plus`, `pp_ftg_driver`), PID wall-follow
  (`pid_driver`), safety (`safety`), and RL deadman gate (`rl_deadman_gate`).
  The RL `drive_command` node also lives here.

Controllers read pose from `/pf/pose/odom` and publish `/drive`. Raceline CSV
defaults to `/config/maps/raceline.csv` (per-car overlay).
