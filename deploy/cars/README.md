# deploy/cars/

Per-car config overlays. The car image is **generic**; everything car-specific
lives here and is supplied at runtime (mounted at `/config`), so one image serves
every car.

Each `carNN/` directory contains:

- `car.yaml` — the car's identity. Authored on the car; **never overwritten** by a
  deployed image or a config push.
- `params.yaml` — chassis calibration (VESC gains, servo offsets, wheelbase),
  safety limits, mux/joy tuning, RL observation parameters, and (optional)
  `sensor_policy` node parameters for the no-localization experiment. The
  ackermann `vesc_actuator` block remains the legacy race path; sensor-policy
  first-run current limits are launch args (see `docs/deployment.md`).
- `maps/` — the occupancy grid + centerline/raceline for the current track:
  - `map.yaml` + `map.pgm` — occupancy grid for map_server / particle filter
  - `centerline.csv` — centerline for PF track-spread relocalize
  - `raceline.csv` — raceline with speed column for algorithmic drivers

To add a car, copy `car01/` to `carNN/` and edit the values.

## RL checkpoint note

The on-car policy trained under the legacy 387-dim layout (380 base + 7-dim opponent
with presence flag) is **not** compatible with the monorepo's 392-dim contract
(384 base + 8-dim zero-sentinel opponent block). Retrain under the monorepo
contract before deploying RL from this repository.

## Vehicle geometry provenance

`params.yaml` carries the chassis geometry the observation and odometry nodes use.
Measured values and exact 68277-4 specifications override the generic URDF; mass
distribution and CG height remain model priors until the assembled car is measured.

| Parameter (node) | Value | Source |
| --- | --- | --- |
| `wheelbase` (`vesc_to_odom_node`) | 0.325 m | On-car calibration |
| `track_width_m` (`vehicle_obs`) | 0.253 m | 296 mm outer track minus 43 mm tyre width |
| `lf_m` / `lr_m` (`vehicle_obs`) | 0.1584 / 0.1666 m | URDF CoG split (assumed) |
| `wheel_radius_m` (`vehicle_obs`) | 0.053 m | Measured wheel radius |
| `cg_height_m` (`vehicle_obs`) | 0.05 m | URDF base_link offset (assumed) |
| `max_steer` (`drive`) | 0.33 rad | Measured servo full-lock (rosbag circle fit) |
| `roll_stiffness_front` (`vehicle_obs`) | 0.47 | Stock GTR front/rear spring-rate split |
| body length x width (training `config.py`) | 0.568 x 0.296 m | Official 68277-4 dimensions |

The RL policy checkpoint itself is delivered separately (mounted at `/policies`),
not stored here, since it is a large binary artifact.
