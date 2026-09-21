# f1tenth_bringup

Top-level launch files that compose the whole stack:

- `launch/race.launch.py` — on-car racing. Defaults to `stack:=algo`; pass
  `stack:=rl` for the policy graph. Starts drivers, particle-filter localization,
  and `localization_preflight`.
- `launch/sim.launch.py` — gym agent graph (`vehicle_obs` → `policy_inference` →
  `drive`) against `IV_2026_SIM`.
- `launch/car.launch.py` — vehicle drivers + deadman + `vesc_actuator`. Starves
  vendored `ackermann_to_vesc` at `ackermann_cmd` so current/brake/servo have one
  owner (ADR 0006).

Generic launch configs live in `config/`. Per-car parameters and maps are supplied
at runtime from [`deploy/cars/`](../../../deploy/cars/), not baked in here.
