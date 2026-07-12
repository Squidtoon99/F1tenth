# ADR 0004: On-car stack integration

## Status

Accepted (2026-07-11)

## Context

The physical car at `shereef@f1tenth` ran a fragmented workspace (`f1tenth_ws_bak`)
with first-party driver nodes embedded in a modified `f1tenth_stack`, a vendored
`particle_filter` with local edits, and RL nodes from `F1tenth-Genesis/ros2_deploy`.
The monorepo already held the RL contract (384 + 6 opponent, tyre loads) and a
generic deploy image, but `localization/`, `control/`, and `racing_algo/` were
mostly scaffolds.

## Decision

Integrate the on-car stack into the monorepo module groups without changing the
observation contract:

1. **Localization** — vendor `particle_filter` (with on-car track-init edits) under
   `src/localization/`; add `range_libc` as a git submodule; compose via
   `f1tenth_localization` (`localization.launch.py`, `pf_relocalize`).
2. **Algorithmic drivers** — lift first-party controllers out of `f1tenth_stack`
   into `src/control/f1tenth_control`; compose via `f1tenth_racing_algo/algo.launch.py`.
   Exclude the legacy bundled `rl_driver/` (superseded by `racing_rl`).
3. **Vehicle layer** — keep vendored `f1tenth_stack` pristine; move hardware tuning
   into `deploy/cars/<car>/params.yaml`; teleop uses vendored `joy` + `joy_teleop`.
4. **Deploy** — build `range_libc` in the runtime image; start localization from
   `race.launch.py`; mount maps at `/config/maps/`.

Vendor vs submodule rule: **vendor** when we edit (particle_filter); **submodule**
when we do not (range_libc).

## Consequences

- One colcon workspace + one generic image now covers drivers, localization, algo,
  and RL stacks.
- The car's deployed 387-dim checkpoint (380 + 7 opponent with presence flag) is
  **not** compatible with the monorepo 390-dim contract; retraining is required
  before RL deploy from this repo.
- `particle_filter` depends on a pip-built `range_libc`; Jetson images should use
  `WITH_CUDA=ON` when building range_libc for GPU ray casting.
