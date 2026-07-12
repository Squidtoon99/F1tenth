# Vendored packages — provenance

| Package (local path) | Upstream repo                         | Branch | Commit SHA                                 | Vendored on |
| -------------------- | ------------------------------------- | ------ | ------------------------------------------ | ----------- |
| `particle_filter/`   | f1tenth/particle_filter               | master | `ec599a1` (2022-02-15) + on-car edits      | 2026-07-11  |

Only the runtime slice is vendored: `particle_filter/*.py`, `config/`, and package
metadata. Upstream `deprecated/`, `docs/`, `media/`, `maps/`, `launch/`, `rviz/`,
and `test/` were dropped (maps launch via `f1tenth_localization`; maps live in the
per-car overlay).

## On-car edits (particle_filter)

Copied from `shereef@f1tenth:~/f1tenth_ws_bak/src/particle_filter` with local
changes applied on the car:

- `particle_filter/particle_filter.py` — track-centerline initialization and
  `/pf/relocalize_on_track` subscription (`track_centerline_csv` parameter,
  `initialize_on_track()`).

PF parameters and launch live in `f1tenth_localization` (`pf_params.yaml`).

Heavy map artifacts (`*.pgm`, track-specific maps) are **not** vendored; they
live in the per-car overlay at `deploy/cars/<car>/maps/`.

## Git submodule (not vendored)

| Package        | Upstream repo           | Notes                                      |
| -------------- | ----------------------- | ------------------------------------------ |
| `range_libc/`  | f1tenth/range_libc        | GPU ray-casting dep for `particle_filter` |

`range_libc` is a submodule because we do not edit it; it is built at image
install time (Cython + optional CUDA on Jetson).
