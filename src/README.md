# `src/` — colcon workspace

This is the single ROS 2 Humble colcon workspace. Build it from the repo root:

```bash
colcon build --symlink-install        # or ./tools/build.sh
```

Packages are grouped by domain. Each group directory is code-owned by a team (see
[`../CODEOWNERS`](../CODEOWNERS)) so modules can evolve independently.

| Group | Purpose |
| --- | --- |
| `common/` | Shared interfaces, utilities, and top-level bringup |
| `perception/` | Opponent/obstacle detection from sensors |
| `mapping/` | SLAM-based track mapping (shared) |
| `localization/` | Particle-filter localization (shared) |
| `planning/` | Raceline optimization + global planning |
| `control/` | Pure-pursuit + drive-command layer |
| `racing_rl/` | On-car RL inference nodes (loads a trained policy) |
| `racing_algo/` | Algorithmic racing stack composition/bringup |
| `vehicle/` | Vendored hardware driver layer (see `vehicle/VENDORED.md`) |

> Scaffold status: packages here are minimal valid placeholders. Node logic is not
> implemented yet; each package README notes where existing code will be migrated
> from.
