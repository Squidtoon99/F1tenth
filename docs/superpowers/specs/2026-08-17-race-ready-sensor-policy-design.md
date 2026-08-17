# Race-Ready 1097-D Sensor Policy Design

**Date:** 2026-08-17

## Goal

Prepare the 1097-D LiDAR + IMU sensor-policy stack for tomorrow's F1TENTH test
day by aligning the simulator, training configuration, observation
normalization, and ROS current-command contract; validating the aligned stack
against held-out real-car rosbags; and launching a monitored cold-start PPO run.

Tonight ends with verified training/deployment code and a healthy training run.
The Jetson image build and every powered-car gate happen tomorrow morning.

## Scope

This work targets the `experiments/e2e-ppo` sensor-policy path:

```text
1081-beam LiDAR + IMU + VESC/proprioception + action history
  -> 1097-D recurrent PPO actor
  -> sensor_racer
  -> /rl/actuator/desired
  -> rl_current_gate
  -> VESC motor/brake/servo commands
```

It does not modify or certify the separate 392-D localized QR-SAC deployment
path. The newer `experiments/self-play-rl` branch contains unrelated Gigaflow
documentation and contributes no simulator implementation to merge.

## Fixed physical and software envelope

The following values are locked for training, artifact metadata, ROS runtime,
and tomorrow's VESC preflight:

| Limit | Value | Enforcement |
| --- | ---: | --- |
| Policy/ROS drive-current maximum | 80 A | Trainer, artifact, `sensor_racer`, current gate |
| Policy/ROS motor-brake maximum | 20 A | Trainer, artifact, `sensor_racer`, current gate |
| VESC hard motor-brake maximum | 25 A | VESC firmware |
| VESC battery-regen maximum | 4 A | VESC firmware |
| Physical current slew | 200 A/s | Trainer, current gate |
| Normalized longitudinal slew | 2.5/s | Warp simulator (`200 A/s / 80 A`) |

The 4 A battery-regen limit is deliberately separate from the actor's 20 A
motor-brake normalization denominator. Code must not claim that the firmware
limit is active without reading or manually verifying the live VESC
configuration.

The 80 A drive and 20 A brake values are directional denominators for
preprocessing v3:

```text
drive: signed_current_fraction = applied_drive_current_a / 80 A
brake: signed_current_fraction = -applied_brake_current_a / 20 A
```

Producer, gate, artifact metadata, fixtures, launch defaults, and per-car
overlays must agree exactly. A mismatch fails closed at checkpoint load.

## Simulator and real-data alignment

The accepted Cursor sim-to-real work under
`/home/ubuntu/f1tenth-sim2real-analysis` remains the evidence baseline:

- observation preprocessing v3 with directional normalized current;
- measured powered-IMU noise and bias domain-randomization ranges;
- surface-friction randomization while retaining the validated nominal vehicle;
- native 1081-beam LiDAR geometry and max-range sentinel behavior;
- retained nominal dynamics where candidate changes did not beat the frozen
  trajectory-divergence baseline.

The current 100 A drive / 10 A brake candidate cannot be reused unchanged.
Moving to 80/20 changes both current feedback normalization and normalized slew,
so a new observation normalizer and a policy trained from scratch are required.
No preprocessing-v2 or 100/10 artifact is deployable under the new contract.

Before the long run, the exact 80/20 configuration must pass:

1. Real-module unit and parity tests for training, policy artifacts,
   preprocessing, action mapping, current gating, and launch configuration.
2. Warp physics, Torch parity, sensor-domain-randomization, and vehicle-geometry
   tests.
3. A short seeded PPO training smoke using the same config as the long run.
4. Fixed-seed nominal and unseen-domain-randomization evaluation.
5. No-actuation replay over held-out real rosbags, including the accepted
   branch-point datasets.

Acceptance requires finite 1097-D observations, bounded actions, zero
drive/brake overlap, no unexpected current-channel clipping or OOD rate, no
persistent period-2 current oscillation, and identical directional-current
normalization in simulator and ROS preprocessing. Offline parity does not prove
tire grip, stopping distance, startup behavior, battery-regen behavior, or
venue-specific LiDAR behavior; those remain tomorrow's physical gates.

## Training configuration and lifecycle

Use the strong, reproducible 1097-D PPO configuration from the phase-11 100 A
cold-start experiment and its W&B/local resolved configuration. Change only the
reviewed 80/20 envelope, its derived 2.5/s normalized slew, and changes required
to make the contract internally consistent.

The overnight run has these fixed properties:

- algorithm: PPO;
- seed: 55, the strongest prior aligned cold-start seed;
- initialization: actor, critic, optimizer, recurrent state, and observation
  normalizer all start fresh;
- transition ceiling: 2,000,000,000;
- execution: continuous until manually stopped, failed, or the ceiling is met;
- checkpoints: immutable policy selection artifacts at regular intervals;
- W&B: online logging with the complete resolved configuration and provenance.

Policy checkpoints do not include critic and optimizer state. They must never
be treated as resumable trainer checkpoints. If the process must restart, it
starts a new cold run with a new run ID.

W&B and the local run snapshot must record the git SHA, a hash of the deliberate
dirty diff if the source is not yet clean, seed, run ID, policy and preprocessing
versions, 80/20/200 limits, resolved configuration, and launch arguments.

Selection never defaults to the final checkpoint. Tomorrow's candidate is the
best Pareto checkpoint across clean laps/completion, lap time or speed,
OOB/collision exposure, tail failures, action saturation, clipping/OOD,
oscillation, and robustness under unseen domain randomization.

## Monitoring and failure handling

Check the run every 30 minutes until the overnight window ends or the user
stops it. Each check records:

- process liveness and advancing transition count;
- freshness of logs and W&B telemetry;
- GPU utilization and absence of persistent runtime errors;
- successful checkpoint creation;
- finite learner, environment, observation, and action metrics;
- reward, clean laps, speed, OOB/collision termination, entropy/KL, action
  saturation, clipping/OOD, and period-2 oscillation indicators.

Monitoring distinguishes these cases:

- **Infrastructure failure:** process exit, no transition progress, non-finite
  values, corrupted checkpoints, or persistent GPU/runtime errors. Preserve the
  failed run and logs, diagnose the root cause, add a regression test, implement
  the smallest verified fix, and launch a new cold run with a new ID.
- **Training-health concern:** sustained numerical or policy collapse across
  multiple evaluation windows. Confirm with raw logs and held-out evaluation
  before stopping.
- **Normal variance:** temporary regressions in reward, speed, lifespan, or lap
  completion while the run remains structurally healthy. Continue training.
- **Plateau:** several consecutive evaluation/checkpoint windows without Pareto
  improvement. Treat it as a human-reviewed stop/selection signal tomorrow,
  never as an automatic kill condition tonight.

No automated rule stops a structurally healthy run solely because performance
temporarily worsens.

## Deployment readiness

Tonight validates source behavior but does not build a sensor-policy runtime
image. The local x86_64 host cannot certify the arm64-only NVIDIA JetPack iGPU
base, and Docker is not currently available in this WSL environment. CUDA
training, simulator evaluation, and bag replay run directly through the local
`.venv` on the RTX 4080.

Tonight's ROS-facing source checks cover:

- `sensor_racer -> rl_current_gate` as the sole RL VESC command path;
- full actor output mapping to no more than 80 A drive or 20 A brake, never both;
- fail-closed handling for stale/non-finite input, policy loss, artifact mismatch,
  watchdog timeout, and deadman ownership;
- launch parsing, diagnostics, and checkpoint rejection behavior.

Collision-safety arbitration is not silently enabled tonight. It must be tested
explicitly with the real graph tomorrow before use.

## Tomorrow morning and track gates

When the Jetson and car are available:

1. Build the exact arm64 sensor-policy image on the Jetson.
2. Run the CUDA image smoke test and a non-powered ROS dry run.
3. Verify the live VESC configuration: 80 A drive, 25 A motor brake, 4 A battery
   regen, voltage cutoffs, and temperature protection.
4. Verify the selected artifact requires 80 A drive, 20 A policy brake,
   preprocessing v3, policy format v4, actor layout v2, force mode, and delta
   steering.
5. Box the wheels and begin with a 5 A software drive limit. Test deadman,
   estop, watchdog, steering direction, current normalization, safe braking,
   source ownership, and diagnostic counters.
6. Run low-speed floor tests. Raise the software drive limit incrementally only
   while voltage, temperature, faults, stopping distance, recurrent resets, and
   diagnostics remain healthy.
7. Run short observed-track laps with manual takeover coverage before race pace.

The software limit must not jump directly from 5 A to 80 A. Simulation and bag
replay are necessary release evidence, not authorization to skip physical gates.

## Error handling and rollback

- Preserve every failed training run, log, checkpoint, config, and W&B identity.
- Never overwrite the prior known-good image or policy during Jetson staging.
- Reject incompatible artifacts rather than adapting their normalization at
  runtime.
- A VESC fault, unexpected regen behavior, rising temperature, voltage anomaly,
  recurrent reset storm, stale sensor stream, diagnostic rejection, or unsafe
  stopping distance blocks raising the current limit.
- Keep manual takeover and the previous image/policy rollback available during
  every powered test.

## Completion criteria

Tonight is complete when the reviewed 80/20 code and configuration pass the
offline gates, the short smoke proves the launch configuration, and the 2B
cold-start seed-55 PPO run is healthy under 30-minute monitoring.

Tomorrow's model is a deployment candidate only after an appropriate checkpoint
passes offline Pareto selection, the arm64 CUDA image and ROS dry-run gates, and
the staged boxed-wheel, floor, and observed-track checks. Race-ready status is
not claimed before those physical gates pass.
