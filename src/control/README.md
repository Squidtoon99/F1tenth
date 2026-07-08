# control/

Control module group. Tracks a reference trajectory and emits `/drive`
(`AckermannDriveStamped`).

- `f1tenth_control/` — pure-pursuit tracker (adaptive lookahead, curvature speed
  cap), optional MPC, and the drive-command layer with a safety watchdog.

The drive-command layer is shared: the algorithmic stack feeds it a tracked path,
while the RL stack feeds it a decoded policy action.

Migration note: `drive_command_node` from
`F1tenth-Genesis/ros2_deploy/f1tenth_rl_agent` and the pure-pursuit controller move
here.
