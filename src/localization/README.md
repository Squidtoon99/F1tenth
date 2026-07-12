# localization/

Localization module group (shared by both racing stacks). Provides pose on a known
map via a particle filter with GPU-accelerated ray casting.

- `particle_filter/` — vendored from `f1tenth/particle_filter` with on-car edits
  (track-centerline initialization, `/pf/relocalize_on_track`). See
  [`VENDORED.md`](VENDORED.md).
- `range_libc/` — git submodule (`f1tenth/range_libc`); build dependency for the
  particle filter.
- `f1tenth_localization/` — launch files, PF/relocalize configs, and the
  `pf_relocalize` joystick helper node.

Per-car map and centerline artifacts live in `deploy/cars/<car>/maps/` and are
mounted at `/config/maps` at runtime.
