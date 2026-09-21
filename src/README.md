# `src/` — colcon workspace

This is the single ROS 2 Humble colcon workspace. Build it from the repo root,
inside the dev container:

```bash
./tools/build.sh                  # or colcon build --symlink-install
```

Packages are grouped by domain. Each group directory is code-owned by a team (see
[`../CODEOWNERS`](../CODEOWNERS)) so modules can evolve independently.

| Group | Purpose |
| --- | --- |
| `common/` | Shared interfaces, C++ obs mirror, top-level bringup |
| `perception/` | Package scaffold — live opponent detector is in `racing_rl` |
| `mapping/` | SLAM wrap + offline centerline post-process |
| `localization/` | Particle-filter localization (shared) |
| `planning/` | Package scaffold — racelines live in `deploy/cars/<car>/maps/` |
| `control/` | Path-tracking, safety, deadman, `vesc_actuator`, RL drive |
| `racing_rl/` | On-car RL inference (`vehicle_obs` → `policy_inference` → `drive`) |
| `racing_algo/` | Algorithmic racing stack composition/bringup |
| `vehicle/` | Vendored hardware driver layer (see `vehicle/VENDORED.md`) |
