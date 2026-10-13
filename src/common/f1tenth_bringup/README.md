# f1tenth_bringup

Top-level launch files that compose the whole stack:

- `launch/race.launch.py` — on-car racing for the legacy 392-D C++ graph
  (`stack:=algo|rl`). The 1,097-D sensor policy does **not** use this launch.
- `launch/sim.launch.py` — simulation bringup (`stack:=vehicle` or `sensor_policy`).
- `launch/car.launch.py` — vehicle drivers only (shakedown/teleop).
- `launch/sensor_policy.launch.py` — on-car 1,097-D graph: vendored drivers →
  `sensor_racer` → `rl_current_gate` → VESC `commands/motor/{current,brake}` and
  `commands/servo/position`. Default of the `f1tenth-sensor-policy` image.
- `launch/sensor_policy_sim.launch.py` — gym sensor-policy loop (`sensor_racer` +
  `rl_current_gate` + `gym_sensor_bridge`; no hardware drivers).

Generic launch configs live in `config/`. Per-car parameters and maps are supplied
at runtime from [`deploy/cars/`](../../../deploy/cars/), not baked in here.
