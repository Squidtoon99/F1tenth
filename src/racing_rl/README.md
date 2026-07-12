# racing_rl/

On-car reinforcement-learning racing (inference only). This is a colcon ROS 2
package that ships in the car image; it loads a trained policy checkpoint (`.pt`)
supplied at runtime.

- `f1tenth_rl_vehicle/` — C++ `vehicle_obs` (track + odom → 390-dim observation),
  `drive`, and optional `opponent_detector`.
- `f1tenth_rl_agent/` — Python `policy_inference` (loads the actor `.pt` + saved
  observation normalizer), plus gym helpers (`track_server`, `evaluation`,
  `obs_debug`). The observation math library `obs_core.py` is kept for parity
  tests and C++ fixture generation — there is no Python observation ROS node.

The **training** side lives entirely under top-level [`training/`](../../../training/)
and is never built by colcon nor included in the car image.

End-to-end bringup: `f1tenth_bringup/race.launch.py` (on-car) and
`f1tenth_bringup/sim.launch.py` (gym gate).
