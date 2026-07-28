# 0009 — Privileged opponent relative acceleration (390 → 392)

- Status: Accepted
- Date: 2026-07-19

## Context

The privileged critic observation used a 6-dim opponent block
`[rel_x, rel_y, rel_vx, rel_vy, gap_norm, ey_o]` at `[384:390)`. Recurrent LiDAR
QR-SAC training needs opponent acceleration in the critic without dropping the
track-relative gap / lateral channels, and without checkpoint compatibility for
the prior layout.

## Decision

1. Extend the opponent block to **8 dims**: relative position (2), velocity (2),
   acceleration (2), normalized along-track gap, and opponent lateral offset.
   Privileged `num_obs` becomes **392**. *Why:* retain useful Frenet scalars while
   adding body-consistent relative acceleration.
2. Compute relative acceleration by rotating each vehicle’s body-frame `ax,ay` into
   world coordinates, subtracting, then rotating into the ego frame. *Why:* matches
   the existing body/world convention used for relative velocity in Warp and on-car
   builders.
3. Keep the all-zero absent-opponent sentinel and caller-side range/certainty mask.
   No migration path for 390-dim checkpoints. *Why:* clean breaking schema change
   authorized for the recurrent training stack.

## Consequences

- Python contract, C++ mirror, Warp kernel, deploy builders, fixtures, and parity
  tests all move together.
- Old 390-dim privileged policies are rejected; retrain under the 392-dim layout.
- On-car opponent acceleration is finite-differenced from world-frame odometry and
  rotated into the opponent body frame before the shared relative-accel math.
