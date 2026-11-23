# `src/` — colcon workspace

This is the single ROS 2 Humble colcon workspace. Build it from the repo root,
inside the dev container:

```bash
./tools/build.sh                      # or colcon build --symlink-install
./tools/build.sh -t racing_rl         # one module group
```

Packages are grouped by domain. Each group directory is code-owned by a team (see
[`../CODEOWNERS`](../CODEOWNERS)) so modules can evolve independently.

| Group | Purpose |
| --- | --- |
| `common/` | Shared interfaces, C++ obs mirror, top-level bringup |
| `perception/` | Scaffold only — opponent detection lives in `racing_rl` |
| `mapping/` | SLAM-based track mapping + offline postprocess (shared) |
| `localization/` | Particle-filter localization (shared) |
| `planning/` | Scaffold only — raceline CSVs come from mapping postprocess / overlays |
| `control/` | Pure-pursuit, safety, deadman, `vesc_actuator`, RL drive layer |
| `racing_rl/` | On-car RL inference (`vehicle_obs` → `policy_inference` → `drive`) |
| `racing_algo/` | Algorithmic racing stack composition/bringup |
| `vehicle/` | Vendored hardware driver layer (see `vehicle/VENDORED.md`) |

`perception/` and `planning/` are empty package scaffolds. Do not look there for
the opponent detector or raceline optimizer.
