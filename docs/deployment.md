# Deployment

How a commit on `develop` becomes a running car.

## Principle: one generic image, per-car config at runtime

Every `develop` commit can be built into a **generic** car image, keyed by git SHA,
that contains no per-car configuration and no policy weights. Per-car identity,
parameters, maps, and the RL policy checkpoint are supplied at runtime by mounting
them into the container. So a single image serves every car, and a car's identity is
never overwritten by a deploy.

```mermaid
flowchart LR
  develop[develop branch] --> build["build_image.sh (buildx linux/arm64, colcon)"]
  build --> img["generic image f1tenth-racing:gitsha"]
  img --> snap["snapshot.sh (.tar.gz / .sif, SHA-keyed)"]
  snap --> load["load_to_jetson.sh (docker load / apptainer)"]
  load --> jetson[NVIDIA Jetson on car]
  overlay["cars/carNN: car.yaml identity + params + map"] --> jetson
  policy["policy .pt (RL only)"] --> jetson
```

## Steps

1. **Build** the generic image for the Jetson architecture:

   ```bash
   deploy/scripts/build_image.sh          # tags f1tenth-racing:<gitsha> and :develop
   ```

   Uses `docker buildx --platform linux/arm64` (QEMU emulation on x86 hosts) and a
   multi-stage `Dockerfile.runtime` that colcon-builds the racing packages. On a
   Jetson with JetPack, `build_image.sh` auto-enables CUDA range_libc
   (`RANGE_LIBC_WITH_CUDA=ON`, target `runtime-cuda`). Validate Dockerfile syntax
   before a full build:

   ```bash
   docker buildx build --check -f deploy/docker/Dockerfile.runtime .
   ```

2. **Snapshot** it to a portable, SHA-named artifact (no registry needed):

   ```bash
   deploy/scripts/snapshot.sh             # -> deploy/snapshots/f1tenth-racing-<gitsha>.tar.gz
   ```

   Record the result in [`../deploy/releases/manifest.csv`](../deploy/releases/manifest.csv)
   (`git_sha,image_sha256,car_id,deployed_at,notes`).

3. **Load** it onto the car and run with the per-car overlay:

   ```bash
   deploy/scripts/load_to_jetson.sh user@car01 deploy/snapshots/f1tenth-racing-<gitsha>.tar.gz car01
   ```

   On the car:

   **CPU range_libc** (default off Jetson, or explicit `RANGE_LIBC_WITH_CUDA=OFF`):

   ```bash
   docker run --rm -it --net=host --privileged \
     -v /dev:/dev \
     -v /opt/f1tenth/config:/config:ro \
     -v /opt/f1tenth/policies:/policies:ro \
     f1tenth-racing:develop
   ```

   **CUDA range_libc** (Jetson auto-build, or `RANGE_LIBC_WITH_CUDA=ON`): the image
   ships `libcudart.so.12` from `nvcr.io/nvidia/l4t-cuda:12.6.11-runtime` (JetPack
   6.1 / L4T r36.4, ~CUDA 12.6.11). That library is built for the Jetson driver
   (540.x). The NGC compile stage is desktop/SBSA CUDA — copying *its* `libcudart`
   makes `particle_filter` fail at `PyRayMarchingGPU` with
   `CUDA driver version is insufficient for CUDA runtime version`.

   Requires JetPack **6.1+** (nvidia-smi CUDA Version ≥ 12.6). For JetPack 6.0
   (CUDA 12.2), rebuild with
   `L4T_CUDA_IMAGE=nvcr.io/nvidia/l4t-cuda:12.2.12-runtime`.

   Jetson nvidia-container-toolkit does **not** inject user-space CUDA libs, so
   do not rely on `--runtime=nvidia` for `libcudart` resolution. Use
   `--runtime=nvidia` (or the Jetson default nvidia runtime) so **libcuda** (the
   driver) is visible; `libcudart` still comes from the image:

   ```bash
   docker run --rm -it --runtime=nvidia --net=host --privileged \
     -v /dev:/dev \
     -v /opt/f1tenth/config:/config:ro \
     -v /opt/f1tenth/policies:/policies:ro \
     f1tenth-racing:develop
   ```

   After a CUDA build, verify the extension imports (build checks `DT_NEEDED` and
   `ldd` resolution; this confirms numpy ABI + ROS deps at runtime):

   ```bash
   docker run --rm --entrypoint bash f1tenth-racing:develop -lc \
     'source /opt/ros/humble/setup.bash && \
      python3 -c "import range_libc; assert hasattr(range_libc, \"PyRayMarchingGPU\"); print(\"CUDA range_libc ok\")"'
   ```

   `-v /dev:/dev` is required: `--privileged` exposes device *nodes* (e.g.
   `/dev/ttyACM0`) but not the udev *symlinks* the drivers use (`/dev/sensors/vesc`,
   LiDAR, joystick). Mounting host `/dev` brings those stable symlinks into the
   container so `vesc_driver` (and the LiDAR/joy nodes) open their configured ports.

## Per-car config

Lives in [`../deploy/cars/`](../deploy/cars/). Each `carNN/` has:

- `car.yaml` — identity, authored on the car, **never overwritten**.
- `params.yaml` — **node-scoped ROS 2 parameters** (keyed by node name with a
  `ros__parameters` block) layered on top of the package launch defaults. Covers the
  RL `drive` limits, `vehicle_obs` observation mode (390-dim solo, opponent block
  zeroed), and the vendored VESC calibration. It is passed directly to the nodes —
  `race.launch.py` forwards it as `overlay_params_file:=/config/params.yaml` and
  `car.launch.py` as `vesc_config:=/config/params.yaml`; there is no custom parser.
- `maps/` — occupancy grid + centerline/raceline for the current track (`map.yaml`,
  `centerline.csv`, `raceline.csv`).

The same image runs the algorithmic or RL stack; select with `stack:=rl` (default)
or `stack:=algo`. The RL policy is mounted read-only at `/policies/policy.pt`.
`race.launch.py` starts vehicle drivers, particle-filter localization
(`enable_localization:=true`), and a read-only `localization_preflight` health check.
The RL graph emits no drive command until pose + twist are live.

**Checkpoint compatibility:** policies trained under the legacy on-car 387-dim layout
are not compatible with the monorepo 390-dim contract; retrain before RL deploy.

## Certification gate (f1tenth_gym)

Before a release, the stack is certified against the `f1tenth_gym` bridge with a
real 390-dim checkpoint. The on-car C++ autonomy graph (`vehicle_obs` →
`policy_inference` → `drive`) is the sole gate. All commands run from the repo root.

1. **Contract, parity, unit (host `.venv`)** — the 390-dim contract and the
   training↔deploy observation parity:

   ```bash
   PYTHONPATH="libs/f1tenth_contract:src/racing_rl/f1tenth_rl_agent:training" \
     .venv/bin/python -m pytest \
       libs/f1tenth_contract/test/test_contract.py \
       src/racing_rl/f1tenth_rl_agent/test/test_contract_parity.py \
       src/racing_rl/f1tenth_rl_agent/test/test_quasi_static_load.py
   NUMBA_DISABLE_JIT=1 PYTHONPATH="libs/f1tenth_contract:src/racing_rl/f1tenth_rl_agent:training" \
     .venv/bin/python -m pytest src/racing_rl/f1tenth_rl_agent/test/test_obs_parity.py
   ```

2. **Build + ROS/C++ tests (dev container)** — the C++ obs/action mirror, tyre
   slip/load math, fixture parity, node integration, and drive-watchdog / bad-obs
   safety tests:

   ```bash
   ./tools/build.sh -t racing_rl
   ./tools/test.sh --packages-select f1tenth_common f1tenth_rl_vehicle f1tenth_rl_agent f1tenth_control
   ```

3. **Closed-loop gym gate (multi-container)** — the exact on-car C++ graph against
   gym physics with a real checkpoint, then the automated acceptance validator:

   ```bash
   CHECKPOINT_DIR=/abs/dir CKPT=policy.pt tools/sim.sh up      # (leave running)
   CHECKPOINT_DIR=/abs/dir CKPT=policy.pt \
     VALIDATE_ARGS="--duration-s 600 --min-samples 2000" tools/sim.sh validate
   ```

4. **Runtime-image gym gate** — the same acceptance from the *built* generic image
   with the per-car overlay, so `/config` + `/policies` wiring is exercised:

   ```bash
   deploy/scripts/build_image.sh
   CHECKPOINT_DIR=/abs/dir CKPT=policy.pt CONFIG_DIR=deploy/cars/car01 \
     STACK=vehicle MODE=release tools/sim.sh up
   ```

### Acceptance criteria (validator PASS)

- observations are all finite and exactly 390-dim; actions in `[-1, 1]`; drive
  `acceleration` in `[-1, 1]`, `speed == 0`, and `|steer|` within `max_steer`;
- the car moves (odom speed above the floor for enough samples);
- the opponent block `[384:390)` stays zero in solo mode;
- ≥ 3 consecutive IV_2026 laps and a 10-minute soak with no process exits and no
  watchdog trips during normal operation;
- policy-loss and non-finite-input fault tests command a safe brake within the
  ~0.15 s (3-cycle @ 20 Hz) drive watchdog (covered by the `f1tenth_control` node
  tests).
- checkpoints must be `policy_format_version >= 2` with `longitudinal_mode: force`
  (speed-trained artifacts are rejected at load).

Retain the validator stdout and the sim/agent container logs (`tools/sim.sh logs`)
with the run, and record the release in
[`../deploy/releases/manifest.csv`](../deploy/releases/manifest.csv).

### Remaining prerequisite before a powered on-car run

Current/force rollout (ADR 0006) requires a fresh force-trained checkpoint with a
new `obs_norm` (`policy_format_version >= 2`). Do not deploy a speed-trained
policy. Stage on-car certification only after Part 2 current/brake calibration
and these gates: boxed wheels at conservative `i_*_max_a` → low-speed floor with
manual estop/deadman checks → braking-distance and acceleration characterization
→ raise limits only within the calibrated/simulated envelope. Promote the
single-mode build and calibrated `deploy/cars/car01/params.yaml` together.

Confirm `deploy/cars/car01/params.yaml` has measured VESC servo gains
(`steering_angle_to_servo_*`) and bench-safe current limits before any powered
run. The observation mode
(`enable_load_estimation` / `enable_slip_estimation`) must also match how the
deployed checkpoint was trained.
When the checkpoint was trained with live slip/load channels (`zero_tyre_slip_obs:
false`), enable both flags in the per-car overlay and set `slip_obs_mean` from the
checkpoint's saved `obs_norm` running mean:

```bash
.venv/bin/python -c "
import torch
p = torch.load('/path/to/policy.pt', map_location='cpu', weights_only=False)
print(p['obs_norm']['mean'][372:380].tolist())
"
```

Paste the eight values into `vehicle_obs.slip_obs_mean` in
`deploy/cars/<car_id>/params.yaml`. With estimation off, `vehicle_obs` emits zeros
`[372:380)` and ones `[380:384)` — out-of-distribution for slip/load-trained
policies.

### On-car slip/load calibration (first powered test)

IMU sign and filter tuning happens on the ground, not in the gym bridge (which has no
`/sensors/imu/raw`). During the first powered shakedown at conservative
`vesc_actuator.i_drive_max_a` / `i_brake_max_a` (boxed wheels, then low-speed floor):

1. Constant-speed roll → rear slip ratios near 0.
2. Throttle step → rear κ responds.
3. Hard turn → load transfer visible in `[380:384)`.
4. Tune `imu_ax_sign`, `imu_ay_sign`, `imu_yaw_rate_sign`, `vy_filter_tau_s`,
   `vx_ground_lp_alpha`, and `slip_speed_min_mps` from a rosbag if channels look
   wrong post-`obs_norm`.

## Releases

A release is a git tag on `develop` (e.g. `racing-v<ver>`). Build + snapshot that
exact commit and note the tag in the manifest. There is no release branch.

## Offboard training (HPC)

RL training can run on HPC via Apptainer:

```bash
apptainer build training.sif deploy/apptainer/training.def
apptainer run --nv training.sif --num-envs 4096 --total-steps 2000000
```

Weights produced there are delivered to the car as the mounted `/policies/policy.pt`
— they are never baked into the image.
