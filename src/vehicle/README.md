# vehicle/

The car hardware driver layer. These packages are **vendored** from the upstream
F1TENTH driver repository as a frozen snapshot (copied in as first-party code so we
can edit them directly). We deliberately do not track upstream; the exact upstream
commits are recorded in [`VENDORED.md`](VENDORED.md).

Vendored packages (drop upstream package contents into these directories):

- `f1tenth_stack/` — bringup launches + all parameter files + `throttle_interpolator`
  + SICK LiDAR launch files.
- `vesc/` — VESC driver stack (`vesc_driver`, `vesc_msgs`, `vesc_ackermann` with
  `ackermann_to_vesc` and `vesc_to_odom`).
- `ackermann_mux/` — twist/ackermann command multiplexer.

Generic dependencies are NOT vendored; they are installed via `rosdep`/apt in the
Docker base image:

- `joy`, `urg_node`, `ackermann_msgs`, `sick_scan_xd`, `teleop_tools`

## Topic contract

The driver layer subscribes to `/drive` (`AckermannDriveStamped`) and publishes
`/scan`, `/odom`, `/sensors/imu/raw`, `/sensors/core` — the interface the rest of
the stack (`control/`, `racing_algo/`, `racing_rl/`) targets.
