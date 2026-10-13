# 0038 — Gym occupancy uses ROS trinary thresholds

- Status: Accepted
- Date: 2026-10-13

## Context

Every Austin gym certify (baseline-v5 and dr-v5) left the track at the same
spot: ~5 m, first right-hander, ~3 s. Beam-by-beam LiDAR at spawn and 3 m
ahead showed no left/right mirror, no angle shift, and no 0.27 m mount applied
twice or opposite. Map PNG, origin, resolution, and centerline matched Warp.
Spawn yaw is the first centerline tangent (−0.6524 rad).

The mismatch was occupancy semantics. Warp classifies Austin's anti-aliased
PNG with ROS `occupied_thresh` / `free_thresh` (unknown cells → max-range).
f1tenth_gym `Track` split pixels at 128, turning Warp's unknown band into
free space. At spawn the gym RIGHT beam was ~1.1 m against Warp's 30 m.
`sim.sh` already stamped 205 as max-range, but `Track.from_track_path`
wiped those cells before ScanSimulator saw them. The gym_bridge fallback
that preserved exact pixel 205 never ran, because Austin has a centerline.

## Decision

1. Patch gym `Track` loading (and the gym_bridge YAML fallback) to build
   occupancy from YAML `occupied_thresh` / `free_thresh` / `negate`: 0
   occupied, 205 unknown, 255 free.
   Why: that is the same ROS trinary rule Warp uses, including anti-aliased
   greyscale maps that are not `{0, 205, 254}`.
2. Keep the ScanSimulator patch that treats 205 as max-range in the distance
   field (unknown is an obstacle in the EDT so rays stop, then dt=max_range
   so the reported range is 30 m).
3. Diagnose Warp-behind-ROS by lockstepping Warp physics with
   `sensor_racer` packing (`pack_lidar_from_scan` / IMU / history). If that
   clears the first corner, gym occupancy/map was the gap.

On-car `sensor_policy.launch.py` is unchanged.

## Consequences

Gym `/scan` at Austin spawn matches Warp occupancy raycasts (no 1 m phantom
wall to the right). Certify of baseline-v5 s42 on Warp packing is the
ROS/obs-path check; gym certify after this patch is the remaining
physics/map check. Anti-aliased maps now produce ~5k unknown cells on Austin
instead of 92 exact-205 pixels.
