# Minimal Sensor-Policy Jetson Deployment Design

**Date:** 2026-08-17

## Goal

Produce and stage a deployment-ready, native ARM64 container for the 1097-D
end-to-end recurrent policy on `shereef@f1tenth`. The image runs only the sensor
policy, its required hardware drivers and safety controls, and joystick control.
The selected checkpoint remains a separate, read-only runtime artifact.

The ROS command envelope is 80 A drive and 20 A motor brake. The VESC battery
regen maximum of 4 A is an operator-verified firmware prerequisite. Deployment
must neither read nor change VESC firmware configuration.

## Runtime architecture

Continue the dedicated `f1tenth-sensor-policy` image and
`sensor_policy.launch.py` graph. The runtime data flow is:

```text
LiDAR + IMU + VESC state + action history
  -> sensor_racer
  -> desired current and steering
  -> rl_current_gate
  -> VESC motor/brake/servo commands

joystick
  -> joy_teleop + rl_deadman_gate
  -> manual/autonomous ownership
```

`rl_current_gate` is the sole VESC command publisher. Joystick deadman state
controls command ownership and provides manual takeover. Stale sensors, stale
commands, non-finite values, incompatible policy metadata, or lost ownership
must fail closed without drive current.

## Minimal image boundary

Use the existing multi-stage `deploy/docker/Dockerfile.sensor_policy` and retain
only the build and runtime closure required by `sensor_policy.launch.py`:

- ROS 2 Humble base runtime;
- Ackermann messages, joystick teleop, Hokuyo URG, and serial support;
- `f1tenth_bringup`, `f1tenth_stack`, `f1tenth_control`,
  `f1tenth_rl_agent`, the policy/contract libraries, interfaces, and required
  VESC packages;
- the JetPack-compatible NVIDIA PyTorch runtime needed for CUDA inference.

The final image excludes the source tree, compiler and colcon tooling, rosdep,
tests, maps, checkpoints, particle filtering, `range_libc`, localization, SLAM,
Nav2, mapping, planning, the classical raceline stack, the 392-D RL stack, and
`ackermann_mux`. The image inventory smoke test must prove excluded ROS packages
are unavailable, not merely unused at launch.

The NVIDIA PyTorch base dominates image size. Replacing it is outside this work
unless measurements show an equally compatible smaller JetPack CUDA/Torch
runtime; package removal must not compromise native CUDA inference.

## Configuration and policy contract

Mount `/config` and `/policies` read-only. The generic image contains neither
per-car parameters nor model weights. Stage the selected checkpoint under a
content-hashed name and atomically update
`/opt/f1tenth/policies/e2e_policy.pt`.

The eventual deployment candidate must have passed the existing deployment gate
and retain the current input/output contract. Artifact metadata and both ROS
nodes must agree on:

- policy format and actor layout;
- observation preprocessing version and 1097-D input;
- recurrent model layout;
- delta-steering mode and steering bounds;
- 80 A drive and 20 A motor-brake normalization and output limits.

Any mismatch rejects the checkpoint at startup. The container does not rescale
or reinterpret an incompatible artifact.

The 4 A battery-regen maximum is not a ROS output limit and cannot be guaranteed
by this image. It is recorded as a required manual VESC preflight item. The
deployment tooling must not invoke VESC configuration utilities or write
firmware settings.

## Native build and staging

The target is the inspected Jetson at `shereef@f1tenth`: ARM64, JetPack/L4T
R36.4.7, Docker with the NVIDIA runtime, and sufficient local storage. Build the
exact repository revision natively on that host and tag it with its git SHA plus
the isolated `sensor-policy` moving tag.

Deployment performs these steps in order:

1. Verify the repository tests and the selected model's deployment-gate record.
2. Build the generic image natively on the Jetson.
3. Run non-powered CUDA, Torch, ROS, package-inventory, and dry-launch smoke
   checks.
4. Stage the content-addressed checkpoint and `car01` overlay.
5. Record previous image and checkpoint targets for rollback.
6. Install or update the on-car run helper with NVIDIA runtime, host networking,
   `/dev`, and read-only config/policy mounts.
7. Stop before starting the hardware-driving graph.

Powered startup remains a separate explicit operator action after confirming
physical safety and the external 4 A VESC regen setting.

## Verification and rollback

Repository verification follows `AGENTS.md`: complete build, tests, and lint,
with ROS packages exercised inside the dev container. Relevant pure-Python
policy and artifact checks run in `.venv`. The selected checkpoint must pass the
existing offline and closed-loop deployment gate before staging.

On the Jetson, verify:

- CUDA-visible Torch inference on the native image;
- exact required ROS package inventory and absence of excluded subsystems;
- checkpoint metadata compatibility and a successful model load;
- dry-run launch without VESC command ownership;
- joystick, sensor, and VESC device visibility without powered actuation;
- preserved rollback metadata and a working rollback command.

Deployment never overwrites the classical `f1tenth-racing` image or
`policy.pt`. It preserves the previous sensor-policy image and checkpoint
target. A failed build or smoke test leaves the active deployment unchanged.

## Completion criteria

This work is complete when the minimal native image builds on the Jetson, all
repository and on-car non-powered checks pass, a gate-passing compatible model
is staged read-only, the on-car run helper and rollback path are ready, and no
powered container has been started automatically.
