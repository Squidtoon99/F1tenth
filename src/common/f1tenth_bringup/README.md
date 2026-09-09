# f1tenth_bringup

Top-level launch files that compose the whole stack:

- `launch/race.launch.py` — on-car racing (select `stack:=algo|rl`).
- `launch/sim.launch.py` — simulation bringup (`stack:=vehicle` or `sensor_policy`).
- `launch/car.launch.py` — vehicle drivers only (shakedown/teleop).
- `launch/sensor_policy.launch.py` — no-localization recurrent RL (`sensor_racer`
  + `rl_current_gate`; classical race graph unchanged).
- `launch/sensor_policy_sim.launch.py` — gym sensor-policy loop (`sensor_racer` +
  `rl_current_gate` + `gym_sensor_bridge`; no hardware drivers).

Generic launch configs live in `config/`. Per-car parameters and maps are supplied
at runtime from [`deploy/cars/`](../../../deploy/cars/), not baked in here.
