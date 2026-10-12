# Simplify the simulator and establish courtyard parity

## Summary

Replace the detailed vehicle plant with a **reduced dynamic bicycle model**, retain the **1,097-input CNN/GRU actor**, and move training and deployment to **20 Hz**.

The first milestone is verified sensor, actuator, and trajectory parity against existing recordings. Physical lap certification and further reward tuning follow separately.

Treat Gengar's speed sign as corrected. Preserve that correction and investigate its remaining failures independently. Remove retired physics and calibration code from the active repository; Git history preserves it for later work.

## 1. Establish one consistent training/deployment baseline

- Create an isolated feature branch from the latest drivetrain implementation. Bring across the root checkout's confirmed runtime corrections deliberately; preserve unrelated dirty worktree changes. Do not inherit exploratory parameter changes automatically.
- Freeze the existing courtyard manual-current manifest, September skidpad recordings, and seven Gengar bags. Resolve their actual local paths, record hashes, and retain existing crash exclusions. Whole bags remain the unit of train/holdout separation.
- Use manual/skidpad data for identification. Use Gengar for observation reconstruction, ownership transitions, command replay, and failure analysis. Exclude contact and intervention intervals from plant fitting.
- Produce one report containing the source revision, configuration, bag identities, exclusions, baseline metrics, candidate metrics, and parameter decisions. Replace the elaborate staged promotion workflow with this single replay-and-comparison entry point.
- Preserve exact checkpoint identity when available. A reconstructed observation or plant comparison can proceed without the deployed checkpoint; claims about reproducing its policy actions require the exact artifact.

## 2. Replace detailed physics with the reduced bicycle

Replace the existing shared dynamics implementation used by standalone simulation and the training environment, rather than introducing a selectable backend.

**Retained physical state**

World position and heading; body-forward and lateral velocity; yaw rate; logical and physical steering; applied drive/brake command state. Keep the existing observation/history buffers.

**Vehicle dynamics**

- Use two axle tire forces with linear cornering stiffness and friction-circle saturation. Use static axle loads derived from mass and centre-of-mass geometry.
- Fix measured mass at **3.444 kg**, wheelbase at **0.325 m**, existing axle distances, and the existing **0.13 kg·m² yaw-inertia prior**. Label inertia as a prior rather than a bag measurement.
- Fit one shared lateral-stiffness scale, with the front/rear ratio fixed by static load distribution. Do not jointly fit geometry, inertia, steering gain, and tire stiffness.
- Retain a smooth low-speed transition to bicycle kinematics below **1 m/s** to avoid singular slip-angle calculations. Require continuity through the transition.
- Keep the **5 ms physics timestep**. Advance the shared dynamics identically in training and replay.

**Actuator behavior**

- Preserve **80 A drive / 40 A brake** and the actual gate's separate current-channel slew and switching behavior. Reuse its rules rather than approximating them with one normalized signed-effort ramp.
- Use empirical drive-current-to-acceleration and brake-current-to-deceleration gains, plus one constant coast-loss term. Fit coast loss only where qualifying coast data exist; otherwise retain zero and report it as unidentified.
- Braking opposes motion and cannot propel the vehicle through zero speed.
- Retain first-order physical steering response. Start from the independently identified gain **1.10714** and lag **0.09 s**, then validate them with low-speed skidpad data before fitting lateral stiffness.
- Keep logical steering bounded at **±0.33 rad**. The physical target may reach gain × logical steering; do not repeat the clamp that erased the gain correction.

**Remove**

FOC and battery electrical modeling; individual wheel angular dynamics; Pacejka curves and relaxation states; suspension, load transfer, load sensitivity, and associated tuning parameters; speed-conditioned steering compensation; obsolete fitting/promotion paths and their unused configuration.

Remove tests exclusively belonging to deleted components. Migrate tests covering retained behavior to the replacement model.

## 3. Preserve the policy contract and correct its physical meanings

Keep all **1,097 actor fields**, their ordering, the CNN/GRU architecture, and the two current/brake-plus-delta-steering actions.

- Verify the car's acceleration-axis mapping using existing bags. Apply the confirmed driver-to-body rotation before actor packing, including correctly transformed bias values.
- Preserve Gengar's corrected speed convention. Test historical sign failures separately; do not apply a sign flip to Gengar.
- Define the current field as the **normalized applied command fraction**, matching the runtime. Do not substitute modeled motor telemetry.
- Derive IMU acceleration from inertial acceleration expressed in body coordinates, including rotational terms. Average simulated IMU samples over the same **50 ms interval** used by deployment.
- Preserve gravity and constant IMU channels, commanded steering histories, applied longitudinal histories, and ownership/reset ordering.
- Use body-forward velocity as the reduced model's no-slip wheel-speed estimate. Report wheel-slip uncertainty explicitly when comparing ERPM-derived speed with chassis motion.
- Retain the 392-field critic layout for this milestone. Document reduced-model assumptions: zero longitudinal tire slip, static load ratios, and axle-derived lateral slip angles. Do not retain wheel dynamics merely to populate critic fields.

Use one nominal occupancy map for both LiDAR and vehicle-footprint contact. Replace the independent Frenet-width collision boundary. Keep centreline projection for progress and critic geometry. Disable wall-map variants initially; reintroduce measured randomization only after nominal parity passes.

**20 Hz compatibility**

- Set the control interval to ten 5 ms physics steps.
- Retain the **3° steering delta per decision**, giving approximately **1.05 rad/s** command authority.
- Update artifact validation and runtime timing together. Increment preprocessing compatibility because sensor semantics change; reject incompatible historical artifacts in the new runtime.
- Preserve learning horizons in seconds: use the square root of existing per-step discount and GAE factors, and double PPO rollout length. Scale continuous reward terms by elapsed time; keep event penalties per event.
- Retrain actor and normalizer from scratch. Historical checkpoints remain diagnostic baselines.

## 4. Calibrate and verify without another open-ended tuning loop

Fit in this order: **steering response → longitudinal gains → lateral stiffness**. Freeze accepted parameters before moving to the next stage.

Use the same robust objective for optimization and candidate-improvement decisions. Independently judge held-out physical metrics. Missing excitation produces an explicit unidentified result and blocks promotion of that parameter only.

Required verification:

- **Sensor semantics:** all 16 non-LiDAR fields agree within **1e-6** when actual production preprocessing and simulator packing receive equivalent physical inputs.
- **Actuators:** identical current requests produce matching gate/simulator applied-channel trajectories, including drive/brake switching, zero crossing, handoff, and reset.
- **Steering trajectories:** held-out yaw-rate MAE ≤ **0.10 rad/s**, one-second heading error ≤ **0.08 rad**, and median turn-radius error ≤ **5%** where independently identifiable.
- **Longitudinal trajectories:** median absolute speed error ≤ **0.25 m/s** at 0.5 and 1.0 seconds; separately report acceleration and integrated IMU velocity change.
- **Regression:** no eligible held-out bag worsens a scored trajectory metric by more than **5%** against the frozen baseline.
- **Geometry:** LiDAR and footprint contact agree on the same walls, including corners and near-wall starts.
- **Numerics:** deterministic CPU replay; finite standstill, turning, saturation, braking, and low-speed-transition behavior; existing contract tests remain valid.
- Run pure-Python tests/lint locally and required ROS build, integration tests, and lint in the dev container.

A failed gate produces a specific residual report. Do not respond by automatically adding another model term, widening randomization, or relaxing acceptance. Missing independent references are reported as unverified, never counted as passes.

## 5. Deliverables and boundaries

Deliver independently reviewable changes in this order:

1. Frozen evidence and corrected sensor semantics.
2. Reduced dynamics and matching actuator behavior.
3. Shared collision/LiDAR geometry.
4. Consistent 20 Hz training/runtime artifacts.
5. Held-out parity results and removal of obsolete code.

Record the architectural replacement in a superseding ADR. Finish with a local training smoke test and fresh artifact export/load test.

Full GPU training, Jetson deployment, physical lap certification, reward changes for chatter, and wider domain randomization are subsequent stages. Report steering saturation and drive/brake oscillation now so those stages have a baseline.

No new dependencies, ROS packages, vendored vehicle edits, or deployment-image changes are planned. Success for this milestone is a simpler, reproducible simulator with demonstrated parity—not a claim that physical racing is already solved.
