# 0030 — Replace the detailed plant with a reduced bicycle at 20 Hz

- Status: Proposed
- Date: 2026-10-12
- Supersedes: 0002's four-wheel Pacejka plant as the training physics source of
  truth; 0010's 10 Hz sensor-policy cadence. Does not supersede 0007's Warp
  backend.

## Context

The Warp plant carried FOC/battery electricals, per-wheel angular states,
Pacejka curves, relaxation, suspension, and load transfer. That detail did not
produce a courtyard racer that reliably finished laps, and it blocked fitting
steering, longitudinal gains, and lateral stiffness independently. Sensor-policy
training was still at 10 Hz after ADR 0010, while the identified steering lag
and gate current slew are naturally expressed on a 5 ms physics tick. Gengar
bags showed that packing VESC speed body-forward, not with a sign flip, is
required for a standing enable not to dive.

## Decision

1. Use one reduced dynamic bicycle in `training/f1tenth_sim`: two-axle linear
   cornering stiffness, friction-circle saturation of lateral force, and static
   axle loads from mass and centre-of-mass geometry.
   *Why:* One shared lateral-stiffness scale can be fit after steering and
   longitudinal gains, without jointly identifying geometry, inertia, and tires.
2. Keep mass **3.444 kg**, wheelbase **0.325 m**, existing axle distances, and
   yaw inertia **0.13 kg·m²** as a labeled prior. Coast loss stays **0** until
   qualifying coast data exist.
3. Match the 80 A / 40 A gate: separate drive and brake channels, 200 A/s slew,
   exclusive switching, and braking that cannot reverse through zero. Logical
   steer stays ±0.33 rad; the physical first-order target is gain × logical
   (1.10714, τ=0.09 s) and is not reclamped.
4. Share one occupancy distance field for LiDAR and footprint contact. Keep
   centreline projection for progress and the 392-D critic. Disable wall-map
   variants until nominal parity passes.
5. Train and deploy the 1,097-D sensor policy at **20 Hz** (ten 5 ms substeps).
   Preserve horizons in seconds with the square root of the 10 Hz discount and
   GAE factors and doubled PPO rollout length. Increment
   `observation_preprocessing_version` **3 → 4**.
6. Pack VESC speed body-forward (`TWIST_VX_SIGN_BODY_FORWARD = +1`). Rotate
   driver-frame IMU acceleration into body coordinates
   (`body_x = -driver_y`, `body_y = driver_x`) after native-axis conversion and
   bias correction. Do not sign-flip Gengar.

## Consequences

- Historical 10 Hz / preprocessing-v3 sensor artifacts are rejected. Actor and
  normalizer must retrain from scratch; old checkpoints remain diagnostic.
- Privileged 392-D `f1tenth_contract` cadence stays 10 Hz. Critic tyre fields
  report axle-derived lateral slip, zero longitudinal slip, and static load
  ratios.
- Bag fitting, held-out trajectory parity, GPU training, Jetson deploy, and
  physical lap certification are subsequent gates. Missing excitation is
  unidentified, not a pass.
- Steering saturation and drive/brake chatter are in-scope to report, not to
  retune by adding plant terms.
- Scripted opponents command unclamped delta-to-logical steer with
  speed-scaled lookahead and corner slow-in, and LUT recovery only replaces
  a farther window projection. Tight maps (courtyard) override lateral offset
  to 0.15 m.
