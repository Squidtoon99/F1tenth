# 0008 — Dedicated sensor-policy CUDA runtime image (full workspace)

- Status: Accepted (image scope superseded by [0025](0025-sensor-policy-image-no-localization.md))
- Date: 2026-07-18 (updated 2026-07-19)

## Context

The 1,097-D format-4 recurrent sensor-policy stack runs on-car without
particle-filter localization for the experimental RL-current graph. The car is an
8 GB Jetson Orin Nano on L4T R36.4.x / JetPack 6.2.x (CUDA 12.6). Inference must
support CPU and CUDA in one image. The certified generic `f1tenth-racing` image
(390-D graph) must never be silently retagged or replaced.

The graph is now `sensor_racer` → `/rl/actuator/desired` → `rl_current_gate` → VESC
hardware topics, with applied-command feedback closing the observation history loop.
External deadman (`rl_deadman_gate`) and manual teleop remain outside the racer.

## Decision

1. **Separate launch.** `f1tenth_bringup/sensor_policy.launch.py` composes vendored
   drivers, `joy_teleop`, `rl_deadman_gate`, optional safety brake, `sensor_racer`,
   and `rl_current_gate`. It does not start `ackermann_mux`, Ackermann-mode
   `vesc_actuator`, PF, map server, SLAM, or the 390-D RL graph. Staged enable flags
   (`enable_drivers`, `enable_racer`, `enable_gate`) support non-powered certification.

2. **Separate image — full workspace.** `deploy/docker/Dockerfile.sensor_policy` builds
   `f1tenth-sensor-policy:<gitsha>` from a digest-pinned JetPack-6-compatible NGC
   PyTorch iGPU Jammy base:

   `nvcr.io/nvidia/pytorch@sha256:c652f021080c2d327fe7c14ae898fef1ba7dd3b756f15bff1455612f32b7cc0c`
   (24.09-py3-igpu, Ubuntu 22.04, CUDA 12.6).

   The build stage colcon-builds the **complete** `src/` workspace (plus
   `libs/f1tenth_contract`); `training/` is never built. Vendored vehicle packages
   are built; non-vendored packages are tested in the build stage. Runtime copies
   `/opt/ros`, the merged install tree, range_libc, pins metadata, entrypoint, and
   smoke helper. Per-car `/config` and `/policies` remain read-only mounts.

3. **Separate moving tag and rollback.** Build/load scripts take
   `TARGET=racing|sensor_policy`. Racing keeps `:develop`; sensor-policy uses
   `:sensor-policy`. Load records the previous moving tag, checkpoint path, and
   `/etc/f1tenth/sensor_policy_pins.json` under `/opt/f1tenth/rollback/<target>/`.
   Checkpoints stage under unique SHA-suffixed names; `/opt/f1tenth/policies/policy.pt`
   is never overwritten. Rollback: `deploy/scripts/rollback_jetson.sh`.

4. **Non-powered smoke.** `deploy/docker/smoke_sensor_policy.sh` (also installed as
   `/usr/local/bin/smoke_sensor_policy.sh`) sources ROS, imports Torch/rclpy, asserts
   CUDA on target, optionally loads a format-4 artifact, inventories expected
   executables, and runs `sensor_policy.launch.py` in dry-run mode with drivers and
   gate disabled.

*Why:* isolates the CUDA recurrent stack and current-gate safety boundary from the
certified 390-D racing image while keeping one generic overlay model and a reversible
deploy path.

## Consequences

- Two images coexist on the car; operators must choose the tag explicitly.
- Sensor-policy image includes the full workspace (classical/race launches are
  available) but certification targets the sensor-policy graph only.
- Native **linux/arm64** build is required; amd64/QEMU builds are not certified.
- Manifest rows note `sensor_policy`, actor layout (2), policy format (4), JetPack/L4T,
  snapshot/checkpoint hashes, and optional benchmark results in the notes column.
- `git submodule update --init src/localization/range_libc` is required before image
  build (range_libc is pip-installed, not vendored).
