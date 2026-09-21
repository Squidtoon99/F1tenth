# localization/

Localization module group (shared by both racing stacks). Provides pose on a known
map via a particle filter with optional GPU ray casting.

- `particle_filter/` — vendored from `f1tenth/particle_filter` with on-car edits
  (track-centerline initialization, `/pf/relocalize_on_track`). Published frames
  are ROS 2 `map` / `laser` (not ROS 1 `/map` / `/laser`). See
  [`VENDORED.md`](VENDORED.md).
- `range_libc/` — git submodule (`f1tenth/range_libc`). CPU by default; CUDA
  (`PyRayMarchingGPU`) is compiled into the Jetson `runtime-cuda` image and needs
  L4T `libcudart` — see [`docs/deployment.md`](../../docs/deployment.md).
- `f1tenth_localization/` — launch files, PF/relocalize configs, and the
  `pf_relocalize` joystick helper node.

Per-car map and centerline artifacts live in `deploy/cars/<car>/maps/` and are
mounted at `/config/maps` at runtime.
