# f1tenth_bringup

Top-level launch files that compose the whole stack:

- `launch/race.launch.py` — on-car racing. Default `stack:=algo`; pass `stack:=rl`
  for the policy. Starts drivers, particle-filter localization, and a read-only
  `localization_preflight` check.
- `launch/sim.launch.py` — gym-side agent graph (on-car C++ RL nodes against gym
  odom). Used by `tools/sim.sh`.
- `launch/car.launch.py` — vehicle drivers only (shakedown/teleop), plus
  `vesc_actuator` and the RL deadman gate.

Generic launch configs live in `config/`. Per-car parameters and maps are supplied
at runtime from [`deploy/cars/`](../../../deploy/cars/), not baked in here.
