# 0025 — Sensor-policy image excludes localization (supersedes 0008 workspace scope)

- Status: Accepted
- Date: 2026-08-02

## Context

ADR 0008 introduced `f1tenth-sensor-policy` as a JetPack iGPU runtime image that
colcon-built the **complete** `src/` workspace and pip-installed `range_libc` in
both build and runtime stages. That behavior was copied from
`Dockerfile.runtime` when the sensor-policy path was added; it was never a
deliberate requirement for the end-to-end graph.

The certified sensor-policy launch (`sensor_policy.launch.py`) performs no
localization — no map server, AMCL, SLAM, or particle filter. The only in-repo
consumer of `range_libc` is `particle_filter`, which is not started by that launch.
Building CUDA raycast kernels and shipping localization packages inflates image
build time and size on Jetson without serving the experimental graph.

The classical 392-D racing stack still needs localization. It remains on
`f1tenth-racing`, built from `deploy/docker/Dockerfile.runtime`, which keeps
`range_libc` and the full localization closure unchanged.

## Decision

1. **Trim the sensor-policy colcon scope.** `deploy/docker/Dockerfile.sensor_policy`
   builds `--packages-up-to f1tenth_bringup` with
   `--packages-ignore range_lib particle_filter f1tenth_localization
   f1tenth_rl_vehicle f1tenth_racing_algo f1tenth_planning ackermann_mux
   f1tenth_common f1tenth_mapping f1tenth_perception`. A `COLCON_IGNORE` guard
   on `src/localization/range_libc` remains so an uninitialized submodule in the
   build context does not trigger the upstream CMake project. The resolved install
   is eleven packages: `f1tenth_bringup`, `f1tenth_stack`, `f1tenth_control`,
   `f1tenth_rl_agent`, `f1tenth_interfaces`, `f1tenth_contract`, `f1tenth_policy`,
   `vesc`, `vesc_driver`, `vesc_ackermann`, `vesc_msgs`.
   *Why:* matches the runtime graph; drops classical 390-D nodes and planning
   packages that `sensor_policy.launch.py` never starts.

2. **Drop range_libc from the sensor-policy image entirely.** Remove the submodule
   existence check, build-stage pip install, runtime-stage `COPY` + pip install, and
   `RANGE_LIBC_WITH_CUDA` build-arg plumbing (`build_image.sh` sensor_policy target
   only).
   *Why:* no runtime importer in this image; racing image already owns the
   localization dependency.

3. **Trim runtime apt to the sensor-policy graph.** The `ros_runtime` stage installs
   only `ros-humble-ros-base`, `ackermann-msgs`, `joy`, `teleop-tools`, `urg-node`,
   and `serial-driver`. Mapping/localization (`slam-toolbox`, `nav2-map-server`,
   `nav2-lifecycle-manager`), telemetry (`rosbridge-server`), alternate LiDAR
   (`sick-scan-xd`), classical extras (`diagnostic-updater`, `tf-transformations`,
   `python3-scipy`, `python3-transforms3d`), and build-only apt (`colcon`, `rosdep`,
   `asio-cmake-module`, `git`, `pip`, `vcstool`) are dropped from the shipped image.
   Build-only apt lives in the `build` stage only.
   *Why:* `sensor_policy.launch.py` starts `urg_node` (Hokuyo; default
   `f1tenth_stack/config/sensors.yaml`), vendored VESC nodes (`serial-driver`), and
   joy teleop; it does not start SLAM, map server, rosbridge, SICK, or
   `ackermann_mux`.

4. **Do not pip-reinstall numpy on the NGC base.** The prior `pip3 install numpy
   cython` layer is removed. NGC ships a Torch-tuned numpy; overwriting it risks an
   ABI mismatch. `cython` was only needed for the removed `range_libc` build.
   *Why:* preserve the prebuilt iGPU Torch stack on Jetson.

5. **Keep ADR 0008 decisions that still hold.** Separate image name/moving tag,
   `sensor_policy.launch.py` composition, non-powered smoke, rollback/load scripts,
   and isolation from the certified racing image are unchanged.

6. **Distinct end-to-end policy artifact name.** The asymmetric 1,097-D stack loads
   `/policies/e2e_policy.pt` (symlink to a content-hashed file under
   `/opt/f1tenth/policies/`). Legacy symmetric checkpoints remain at
   `sensor_policy.pt`; classical racing keeps `policy.pt`.
   *Why:* avoids loading a format-4 asymmetric image against a symmetric v1-preprocessing
   checkpoint left from an earlier experiment.

*Why:* one image per certified graph; avoid paying localization and classical-stack
cost on the no-localization experimental path.

## Consequences

- Sensor-policy image build no longer requires
  `git submodule update --init src/localization/range_libc`.
- Faster, smaller Jetson builds for the sensor-policy target (no range_libc CUDA
  compile; fewer colcon packages and runtime apt packages).
- `race.launch.py`, `car.launch.py`, particle-filter nodes, and classical RL nodes
  (`vehicle_obs`, `policy_inference`, `drive`, `vesc_actuator`, `ackermann_mux`) are
  **not** runnable from `f1tenth-sensor-policy`; attempting them fails loudly at node
  start (missing executables/packages), not silently. `f1tenth_bringup` still installs
  those launch files as data.
- Classical localization certification stays on `f1tenth-racing` /
  `Dockerfile.runtime`.
- Smoke inventory and deployment docs target the sensor-policy graph only; operators
  must pick the image tag explicitly when switching stacks.
- End-to-end loads use `e2e_policy.pt`; legacy `sensor_policy.pt` on cars is not
  overwritten or repurposed.
- Cars with a SICK LiDAR need `sick-scan-xd` added back (or a custom image) and a
  `sensors_config` override; the default stack config targets Hokuyo URG over Ethernet.
