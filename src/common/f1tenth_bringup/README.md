# f1tenth_bringup

Top-level launch files that compose the whole stack:

- `launch/car.launch.py` — vehicle drivers + deadman + `vesc_actuator` (no
  autonomy). Starves vendored `ackermann_to_vesc` at its command input.
- `launch/race.launch.py` — on-car racing. Defaults to `stack:=algo`; pass
  `stack:=rl` for the policy. Starts drivers, particle-filter localization, and
  a read-only localization preflight.
- `launch/sim.launch.py` — gym-side agent graph (on-car C++ nodes + evaluation),
  pointed at gym ground-truth odom.

Generic launch configs live in `config/`. Per-car parameters and maps are supplied
at runtime from [`deploy/cars/`](../../../deploy/cars/), not baked in here.
