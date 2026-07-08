# Vendored packages — provenance

These packages are copied in (vendored) as a frozen snapshot, not tracked as
submodules. Record the exact upstream source and commit here whenever you vendor or
re-vendor, so the snapshot is reproducible.

| Package (local path)      | Upstream repo                                   | Branch        | Commit SHA                                 | Vendored on |
| ------------------------- | ----------------------------------------------- | ------------- | ------------------------------------------ | ----------- |
| `f1tenth_stack/`          | f1tenth/f1tenth_system (`f1tenth_stack` subdir) | humble-devel  | `94cb8d7fb5439315316bf80aadbc7256b80cb4e2` | 2026-07-07  |
| `vesc/`                   | f1tenth/vesc                                    | ros2          | `153998df8545fe1781b975df88e411b4e71d4bfe` | 2026-07-07  |
| `ackermann_mux/`          | f1tenth/ackermann_mux                           | foxy-devel    | `b3c0b083ac03aa8c648537d7e4d22608fcd3440c` | 2026-07-07  |

Notes:
- `f1tenth_stack` was vendored from the `f1tenth_stack/` subdirectory of the
  `f1tenth/f1tenth_system` monorepo (MIT licensed). Only that package directory was
  copied; the repo's `.git`/`.gitmodules` were not.
- `vesc/` contains four packages: `vesc` (metapackage), `vesc_msgs`, `vesc_driver`,
  `vesc_ackermann`. `ackermann_mux/` is a single package. Upstream `.git` and
  `.github/` were dropped; `LICENSE` files are retained.
- In upstream, `vesc`, `ackermann_mux`, and `teleop_tools` are git submodules of
  `f1tenth_system` (not populated by a shallow clone). `vesc` and `ackermann_mux`
  are now vendored from their own repos. `teleop_tools`
  (`f1tenth/teleop_tools` @ `humble-devel`) is instead satisfied via rosdep/apt
  (`joy_teleop`, from `ros-humble-teleop-tools`).
- The vendored packages' non-ROS system deps are installed in the Docker base image:
  `serial_driver` (`ros-humble-serial-driver`, needed by `vesc_driver`),
  `diagnostic_updater` (`ros-humble-diagnostic-updater`, needed by `ackermann_mux`),
  and `rosbridge_server` (`ros-humble-rosbridge-server`, needed by `f1tenth_stack`).

## Not vendored (installed via rosdep/apt in the Docker base image)

- `joy`, `urg_node`, `ackermann_msgs`, `sick_scan_xd`, `teleop_tools`

## How to (re-)vendor

1. Clone the upstream repo at the desired commit.
2. Copy the package directory contents into the matching local path here.
3. Remove upstream `.git` metadata (this is a copy, not a submodule).
4. Update the table above with the source URL and exact commit SHA.
5. Apply any local edits directly; they are now first-party code.

Rationale: we only need the current version and want to edit these packages freely,
so a frozen copy keeps clones and CI simple and avoids nested-submodule friction.
