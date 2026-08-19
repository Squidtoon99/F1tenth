# Rosbag-Driven Sim2Real Parity

## Goal

Use selected real-car rosbags to calibrate and validate the simulator by aligning
recorded positions, actions, and sensors, then measuring how quickly simulated
trajectories diverge. Trajectory parity is the primary success criterion; policy
transfer is an end-to-end validation of simulator accuracy.

The work is autonomous through offline analysis, simulator/interface changes,
benchmarks, and training. On-car changes and physical validation are deferred to
a reviewable handoff.

## Settled design decisions

- Target pure current control. Assume the VESC sensorless startup issue is fixed
  outside this simulator-parity effort; do not add a speed kick or reproduce the
  broken startup behavior in training.
- Use real bags only for calibration and validation, not imitation learning or
  policy fine-tuning.
- Represent the installed car with one nominal vehicle model and randomize
  surface-dependent friction, traction, and response.
- Promote the best stock Traxxas Slash 4x4 motor/gearing prior as the initial
  nominal current-to-force model. Use reasonable DR around it and refine later
  when a controlled loaded current/brake capture exists.
- Change the actor current feedback from raw amperes to applied current divided
  by the configured current limit. Version the preprocessing semantics and
  retrain; retain raw amperes in diagnostics and offline fitting.
- Treat ECSS Floor as one venue in a broader multi-map simulator-parity target.
  An ECSS-specific policy may be trained as a diagnostic, but simulator changes
  must also pass non-ECSS and different-surface holdouts from the same car.
- Prefer continuous PF output for alignment when it is internally sane. Where PF
  is absent or clearly broken, recover pose from LiDAR+IMU+odom. Discard recovered
  absolute trajectories if their parity checks fail.
- Accept candidates by statistically meaningful improvement over a frozen
  baseline, not fixed absolute thresholds. Preserve bag-level and surface-aware
  holdouts.
- Permit measured or strongly inferred 2D LiDAR effects, including geometry,
  clipping, bias/noise, dropout, latency, motion distortion, reflectivity and
  multipath, only when they improve parity inside the performance budget.
- Simulator throughput may regress at most 5% normally, or 10% when a measured
  trajectory/policy-level improvement justifies it.
- Use **Cursor Grok Fast** as the default subagent for parallel bag inventory,
  analysis, simulator experiments, benchmarks, and training investigations.
  Keep final synthesis, promotion decisions, and cross-workstream consistency in
  the parent agent.

## Artifact boundary

Keep all heavy and exploratory work outside Git:

```text
~/f1tenth-sim2real-analysis/
  manifest/       # bag, config, map and checkpoint hashes
  extracted/      # synchronized native-rate and 10 Hz datasets
  masks/          # source, clipping, fault and collision windows
  splits/         # calibration, validation and permanent holdouts
  reports/        # metrics, plots and candidate comparisons
  candidates/     # proposed config/interface/simulator values
  runs/           # localization, replay, benchmark and training logs
```

Do not commit bags, converted datasets, notebooks, plots, reports, fit outputs or
generated checkpoints. Only small validated simulator, interface, configuration
and real-module test changes enter the repository.

## Curated evidence set

### Calibration

- `calibration/boxed_fixed_8a_20260713_223826`: powered-on stationary VESC IMU
  bias/noise.
- `calibration/boxed_45a_20260713_232349`: 20–30 A sensorless breakaway bracket;
  use for low-speed current response, not road-load force fitting.
- `calibration/full_accel_no_load`: no-load ERPM saturation, reverse and
  coast-down.
- `calibration/loaded_pull`: loaded straight speed/reverse response.
- `calibration/steering_205300`: long bidirectional steering/yaw dataset with
  continuous PF.
- `calibration/accel_205714`: mixed forward/reverse/coast and continuous PF.

### Independent validation

- `f1tenth_calib_bags/.../imu_static_on_stage2`: deployed preprocessing and
  powered IMU validation.
- `analysis/diag_lap_20260623_035120`: continuous scan/odom/PF plus legacy
  observation/action data. Pin its 387-element contract and checkpoint before
  interpreting action parity.
- `rosbag2_2026_07_31-21_20_50`: independent speed, steering, LiDAR and VESC
  operational validation.

### Failure and stress holdouts

- `diagnostics/e2e_rl_oscillation_20260803T020627Z`: recurrent policy/current
  feedback oscillation and source-transition regression.
- `2026-08-02/arjun/rosbag2_2026_08_02-21_20_48`: overcurrent, high-speed,
  clipping, LiDAR-sentinel and likely impact stress.
- `calibration/weird_20260622_180051`: stationary PF drift and LiDAR-distribution
  failure condition.

Exclude duplicate, short mixed 100 A, zero-IMU, uncaused-transient, redundant
low-current boxed and idle-only recordings from fitting. They may be consulted
only if a selected bag exposes a specific unresolved ambiguity.

## Phase 1: Manifest, extraction and source masks

1. Hash every selected bag, map, parameter file and checkpoint.
2. Extract both ROS receipt time and header time without altering raw values.
3. Preserve native-rate streams and construct a separate 10 Hz policy timeline.
4. Label TELEOP/RL/SAFE ownership, deadman transitions, clipping, current
   saturation, VESC faults, stale observations, GRU resets and impact candidates.
5. Separate speed-command and current-command experiments; never compare their
   commands as if they have the same semantics.
6. Freeze calibration, validation and surface-aware holdout assignments before
   fitting.

## Phase 2: Pose and action alignment

### PF-preferred trajectory

Use continuous PF poses from `steering_205300`, `accel_205714` and `diag_lap`
when these checks show no obvious failure:

- timestamps and update rate are coherent;
- map/frame transforms can be normalized into one chain;
- scan endpoints align with occupied map cells;
- trajectory is continuous and consistent with odometry/IMU direction;
- loop closures do not introduce implausible jumps.

The single PF pose in the July low-speed bag is an initialization hint only.

### Recovered trajectory

For bags without usable PF:

1. Reconstruct the recorded ECSS occupancy grid from `/map` and retain its
   original resolution, origin and hash.
2. Use odometry as a motion prior and IMU yaw rate as a local constraint.
3. Run 2D scan-to-map alignment and loop closure with multiple initializations.
4. Validate the exact pipeline first on simulated trajectories with known truth.
5. Cross-check real forward/reverse passes, scan reprojection, loop closure,
   odometry and IMU integration.
6. Discard absolute poses when checks are inconsistent; retain only local
   relative-motion and sensor-distribution analysis for those segments.

### Branch-point dataset

At regular accepted timestamps, store:

- map pose, heading and localization confidence;
- speed, yaw rate and recent IMU interval;
- recorded physical command and actuator source;
- LiDAR scan and reconstructed 1,097-D observation;
- recurrent reset/burn-in context.

These branch points are the common inputs to all parity tests.

## Phase 3: Establish the frozen baseline

Run the current simulator against every accepted branch point before fitting.
Record:

- LiDAR per-beam residuals and distribution differences;
- IMU, speed and yaw residuals;
- one-step and finite-horizon trajectory divergence;
- real-observation versus simulated-observation policy actions;
- policy saturation, sign disagreement and recurrent-state divergence;
- simulator throughput, sensor-kernel time, GPU memory and determinism.

Use bootstrap confidence intervals across contiguous segments. A candidate is
accepted only when primary trajectory-divergence metrics improve beyond baseline
uncertainty and no permanent holdout shows a statistically meaningful regression.

## Phase 4: Calibrate current and vehicle dynamics

1. Initialize current-to-wheel-force from the stock Velineon 3500, gearing,
   drivetrain efficiency, wheel radius and vehicle mass priors.
2. Fit ERPM↔speed, speed-loop lag, coast-down, reverse and saturation from
   no-load/loaded speed bags.
3. Fit current command→phase current lag, slew and low-speed response from boxed
   and current-control bags.
4. Fit steering servo map, limits, lag, hysteresis and speed-dependent yaw from
   unclipped steering/PF segments.
5. Estimate nominal drag, rolling resistance, tire force and surface friction,
   then represent observed surface differences with DR.
6. Keep brake force and loaded current-to-force uncertainty explicit until a
   future controlled road-load current/brake capture updates the model.
7. Do not model the known sensorless startup failure; treat reliable current-mode
   startup as an on-car prerequisite.

Promote nominal values into the existing dynamics/config surfaces under
[`training/f1tenth_sim/`](../../training/f1tenth_sim/) and
[`training/config.py`](../../training/config.py), using existing DR knobs before
adding new per-step logic.

## Phase 5: Normalize current feedback consistently

Update sim and deploy preprocessing so the actor sees:

```text
applied_current_fraction =
  signed_applied_current_a / configured_directional_current_limit_a
```

Requirements:

- drive and brake use their respective configured limits and preserve sign;
- gate/applied diagnostics continue to report raw amperes;
- simulator and deploy use the same function and edge-case behavior;
- zero/invalid limits fail closed;
- bump observation preprocessing/artifact metadata;
- update parity tests and retrain observation normalization and policy weights;
- do not load legacy artifacts under the new semantics.

## Phase 6: IMU parity

1. Fit raw units, axis signs, bias, sample timing, noise PSD and drift from the
   selected powered stationary bags.
2. Fit dynamic acceleration, yaw response, vibration and current-correlated noise
   from unclipped, non-fault motion segments.
3. Reproduce deploy interval averaging and frozen actor channels exactly.
4. Start with bias/noise/misalignment DR. Add more dynamics only when block
   substitution shows IMU residuals materially drive trajectory or action
   divergence.

## Phase 7: LiDAR and map parity

For every localized scan:

1. Render the current simulator scan at the same pose.
2. Compare geometry, beam-conditioned range residuals, declared/max-range
   encoding, contiguous dropout, temporal correlation and scan timing.
3. Measure residual dependence on range, incidence, motion, surface and map
   structure.
4. Add candidate effects incrementally:
   - extrinsic and angular alignment;
   - clipping/sentinel behavior;
   - bias, noise and dropout;
   - latency and 2D rolling-scan motion distortion;
   - incidence/reflectivity and multipath approximations.
5. Retain a complex effect only if it improves held-out trajectory/action parity
   or has a clearly inferred failure-mode benefit and remains inside the 10%
   maximum throughput budget.

Keep the native 1,081-beam contract. Consider WarpORacer-style texture-backed EDT
sampling or recorded Warp launches only as optimizations after fidelity is
demonstrated.

## Phase 8: On-rails parity

Force the simulator through each recorded accepted trajectory. At every branch
point compare:

1. reconstructed real observation;
2. simulated observation at the same pose/state;
3. real observation with one simulated sensor block substituted;
4. simulated observation with one real sensor block substituted.

Run the same version-compatible policy with real-history GRU burn-in. Measure:

- LiDAR/IMU/proprio observation divergence;
- normalized clipping and out-of-distribution rates;
- longitudinal and steering action error/sign agreement;
- saturation and hidden-state divergence;
- block-specific contribution to action and trajectory error.

Recorded actions are authoritative only when the bag’s checkpoint and interface
version are pinned. Manual trajectories produce counterfactual policy actions,
not deployment-parity claims.

## Phase 9: Open-loop and off-rails divergence

### Open-loop dynamics

From every branch point, apply the recorded physical action to the simulator and
measure position, heading, speed and yaw divergence over 0.1, 0.5, 1, 2 and 5
seconds.

Primary metrics:

- position-error growth and area under the error curve;
- heading, speed and yaw-rate error;
- time to 0.25 m and 0.5 m divergence;
- wall-clearance and collision-outcome mismatch.

### Off-rails policy

Initialize the simulator from the same state and recurrent history, then allow
the policy to drive freely. Compare trajectory divergence, action divergence,
wall clearance and collision time/location. Start with short horizons and expand
only after local dynamics pass.

## Phase 10: Autonomous candidate loop

Use Cursor Grok Fast subagents in parallel for:

- bag extraction/quality and localization;
- drivetrain/steering fits;
- IMU and LiDAR candidates;
- simulator benchmarks;
- on-rails/off-rails evaluation;
- PPO training and checkpoint evaluation.

The parent agent merges results into one candidate ledger and enforces:

- one subsystem change per attribution experiment;
- frozen bags/splits/baseline;
- relative improvement with confidence intervals;
- no significant permanent-holdout regression;
- deterministic tests and simulator parity tests;
- ≤5% normal throughput regression and ≤10% justified maximum.

Reject failed candidates but preserve their external reports and hashes.

## Phase 11: Policy training as simulator validation

1. After trajectory parity improves, train an ECSS-specific PPO policy from
   scratch as a diagnostic of the target map.
2. Train multiple seeds and reject improvements that depend on one seed.
3. Replay each candidate over compatible real observations and require no
   stationary drive/brake oscillation or excessive normalized clipping.
4. Run on-rails and off-rails divergence checks.
5. Then train/evaluate a multi-map policy with the same car model and
   surface-aware DR.
6. Reject simulator candidates that produce fast simulated policies but worse
   real-bag trajectory/action alignment.

Real bags remain calibration/validation evidence; do not behavior-clone or
fine-tune the policy on operator actions.

## Phase 12: Promotion and handoff

Promote only minimal validated changes:

- simulator/config/DR values;
- shared normalized-current preprocessing and version checks;
- targeted sensor kernels when justified;
- deterministic real-module parity and performance tests.

Run training tests, calibration tests, lint, ROS build/tests for shared deploy
surfaces, and standard simulator throughput benchmarks.

If observation semantics or deploy interfaces change, record the decision in an
ADR. Produce a separate on-car validation handoff covering artifact compatibility,
current-limit configuration, no-actuation replay, boxed-wheel validation and
observed track testing; do not perform those physical changes autonomously.

## Completion condition

Stop when two consecutive candidate iterations:

- improve the frozen primary trajectory-divergence metrics beyond baseline
  uncertainty;
- show no significant regression on surface-aware permanent holdouts;
- pass on-rails and off-rails policy checks;
- eliminate the known stationary action oscillation under the new current
  semantics;
- remain within the agreed simulator performance budget.
