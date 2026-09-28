# `src/` — colcon workspace

This is the single ROS 2 Humble colcon workspace. Build it from the repo root,
inside the dev container:

```bash
./tools/build.sh                 # whole workspace (--symlink-install)
./tools/build.sh -t control      # one module group
```

Packages are grouped by domain. Each group directory is code-owned (see
[`../CODEOWNERS`](../CODEOWNERS)).

| Group | Purpose |
| --- | --- |
| `common/` | Shared interfaces, C++ obs mirror, top-level bringup |
| `perception/` | Opponent/obstacle detection from sensors |
| `mapping/` | SLAM-based track mapping (shared) |
| `localization/` | Particle-filter localization (shared) |
| `planning/` | Raceline optimization + global planning |
| `control/` | Pure-pursuit, safety, deadman, `vesc_actuator` |
| `racing_rl/` | On-car RL inference (`vehicle_obs` → policy → `drive`) |
| `racing_algo/` | Algorithmic racing stack composition/bringup |
| `vehicle/` | Vendored hardware driver layer (see `vehicle/VENDORED.md`) |

Launch the composed stack from `f1tenth_bringup`
(`car.launch.py` / `race.launch.py` / `sim.launch.py`).
