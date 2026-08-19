# deploy/

Everything needed to turn `develop` into a running car, plus offboard training
images.

## Layout

- `docker/` — layered images off one base:
  - `Dockerfile.base` — ROS 2 Humble + all deps + tooling (rarely changes).
  - `Dockerfile.dev` — base + dev conveniences; source mounted live (see
    [`tools/dev.sh`](../tools/dev.sh)).
  - `Dockerfile.runtime` — base + a baked colcon install. The **generic racing
    car image** (`f1tenth-racing`).
  - `Dockerfile.sensor_policy` — JetPack 6 iGPU PyTorch + bringup closure, no
    localization (`f1tenth-sensor-policy`). See ADR 0008 / 0025.
  - `smoke_sensor_policy.sh` — non-powered CUDA/ROS/artifact/launch smoke for the
    sensor-policy image.
  - `entrypoint.sh` — sources ROS + workspace and applies the mounted per-car overlay.
- `apptainer/` — `training.def` (HPC RL training) and `racing.def` (run the racing
  image where Apptainer is preferred).
- `cars/` — per-car config overlays (identity + params + map). See
  [`cars/README.md`](cars/README.md).
- `releases/manifest.csv` — git SHA -> image sha256 -> car, for traceability.
- `scripts/` — `build_image.sh`, `snapshot.sh`, `load_to_jetson.sh`,
  `rollback_jetson.sh` (`TARGET=racing|sensor_policy`).
- `snapshots/` — built image artifacts (gitignored).

## Release flow

```mermaid
flowchart LR
  develop[develop] --> build["build_image.sh (buildx linux/arm64, colcon)"]
  build --> img["generic image f1tenth-racing:gitsha"]
  img --> snap["snapshot.sh (.tar.gz / .sif)"]
  snap --> load["load_to_jetson.sh"]
  load --> jetson[Jetson on car]
  overlay["cars/carNN: car.yaml + params + map"] --> jetson
  policy["policy .pt (RL only)"] --> jetson
```

1. Build the generic image from `develop` (tagged by git SHA).
2. Snapshot it to a portable artifact and record it in `releases/manifest.csv`.
3. Load it onto the Jetson and run with the per-car overlay mounted at `/config`
   and the RL policy at `/policies`.

A release is a git tag on `develop` (e.g. `racing-v<ver>`) whose exact image is
snapshotted. See [`../docs/deployment.md`](../docs/deployment.md).

## Sensor-policy race staging

The 1097-D sensor-policy artifact, `sensor_racer`, and `rl_current_gate` share an
80 A drive / 20 A motor-brake command envelope and a 200 A/s physical slew
(2.5/s normalized Warp slew). Artifact loading fails if the ROS limits do not
match. The VESC's 25 A hard motor-brake and 4 A battery-regen settings are
separate firmware safeguards and must be verified manually on the car.

For overnight race preparation, train and run offline validation on the CUDA
host without building a Docker image. The next morning, build the native arm64
`f1tenth-sensor-policy` image on the Jetson and run its non-powered
CUDA/artifact/launch smoke before powered testing. Start with boxed-wheel,
low-demand commanded-action checks while retaining matching 80/20 ROS limits;
do not use mismatched 5/5 runtime limits with an 80/20 artifact.
