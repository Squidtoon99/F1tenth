# 0027 — Randomize map-backed LiDAR geometry

- Status: Accepted
- Date: 2026-08-18

## Context

The sensor-policy simulator rendered LiDAR from idealized centerline widths, while
the real track is bounded by movable soft walls. Recorded collision runs show
coherent occlusion changes rather than independent beam noise. Exact scan pairing
is not reliable because the bags contain no synchronized global pose and the wall
geometry changed between runs.

## Decision

1. Warp may load a hash-pinned ROS occupancy map for LiDAR while retaining the
   centerline for vehicle state, rewards, and termination.
2. Training precomputes a small seeded bank of smoothly deformed occupancy maps.
   Variant zero is the unmodified map, and each episode deterministically selects
   one variant for all of its beams.
3. Simulator validation uses all-frame nearest-support comparisons between real
   scans and a comprehensive simulated scan set. It does not recover poses, fit
   residuals from paired scans, or discard unmatched real frames.
4. Maps, bags, generated scan sets, and reports remain ignored artifacts. Config
   records the source image hash and deformation parameters.

## Consequences

The simulator represents coherent wall motion without adding localization to the
training or deployment graph. The distance-field bank consumes memory proportional
to its variant count. Pose-free validation can prove that real scans are represented
by simulated states, but it cannot attribute a mismatch to an exact track position.
Training must not start until the held-out support and observation gates pass.
