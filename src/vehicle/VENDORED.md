# Vendored packages — provenance

These packages are copied in (vendored) as a frozen snapshot, not tracked as
submodules. Record the exact upstream source and commit here whenever you vendor or
re-vendor, so the snapshot is reproducible.

| Package (local path)      | Upstream repo                                   | Branch        | Commit SHA | Vendored on |
| ------------------------- | ----------------------------------------------- | ------------- | ---------- | ----------- |
| `f1tenth_stack/`          | f1tenth/f1tenth_system (`f1tenth_stack` subdir) | humble-devel  | TODO       | TODO        |
| `vesc/`                   | f1tenth/vesc                                    | ros2          | TODO       | TODO        |
| `ackermann_mux/`          | f1tenth/ackermann_mux                           | (default)     | TODO       | TODO        |

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
