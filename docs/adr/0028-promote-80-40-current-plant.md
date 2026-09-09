# 0028 — Promote the verified 80/40 Warp plant into training

- Status: Accepted
- Date: 2026-09-08

## Context

[ADR 0027](0027-current-control-bag-identification.md) cataloged Sep 6
current-control bags at scale mass **3.444 kg** and an **80 A drive / 40 A
brake** envelope, and left Warp on the 20 A-era defaults (`mass=3.74`,
`f_drive_max=23.0`, `f_brake_max=5.2`, `tire_mu=0.65`, `i_brake_max_a=20`)
until command replay filled **Verified**. Open-loop Warp IPEM replay of those
bags stopped at diagnosis round 5: μ **0.71** greened train/holdout drive and
brake plateau `a_x` (holdout drive **0.027**); `t_delta` 0.07/0.13 lost; leftover
red is train TTS **0.160** vs 0.15 (one 0.02 s bin) and skidpad yaw ~**0.52**
with `odom.vy=0`. Joint Pacejka NLS is out of scope. Domain-randomization mass
`[3.6, 3.9]` sat entirely above the scale reading.

## Decision

1. Promote the round-5 champion mean plant into Warp and training defaults:
   mass **3.444 kg**, `i_drive_max_a` **80**, `i_brake_max_a` **40**,
   `f_drive_max` **26.5 N**, `f_brake_max` **23.1 N**, `tire_friction` **0.71**,
   `longitudinal_slew_rate_per_s` **2.5**, `t_delta` **0.1**.
   *Why:* holdout and train longitudinal `a_x` gates are green on this overlay;
   further `t_delta` and force knobs did not beat it.
2. Keep **80/40** as the only production envelope. Map skidpad teleop brake
   through `20/40` at replay time; do not train or deploy a 20 A brake plant.
   *Why:* the skidpad bag's 80/20 split is a recording mistake, not a supported
   mode.
3. Recenter `vehicle_mass_range` on 3.444 kg with similar width
   (`[3.29, 3.60]`). Shift `tire_friction_range` so its midpoint is ~0.71
   (`[0.39, 1.03]`) without collapsing it to a point.
   *Why:* DR must cover the identified mean; friction stays a sim-to-real band,
   not a fitted constant. **Superseded for the DR half by
   [0029](0029-bag-referenced-sensor-dr.md)** (`drive_scale` pinned, friction
   `[0.66, 0.85]`, bag-referenced IMU/VESC).
4. Leave Pacejka B/C/E, `vesc.yaml` wheelbase **0.325 m**, and on-car current
   gate config to their existing owners. Replay overlay wheelbase **0.324 m**
   is spec-sheet geometry used in verification, not a deploy retune.
   *Why:* leftover yaw is unobservable without `vy`; identification and firmware
   maps stay separate.

## Consequences

- New training artifacts declare `i_brake_max_a=40` and the 26.5/23.1 N force
  envelope. Shared `TRAINING_I_BRAKE_MAX_A` is 40 so trainer/artifact fallbacks
  match. Checkpoints trained on 20 A / 5.2 N are a different plant.
- Courtyard and Galaxy scenario patches use the same 80/40 envelope;
  slew is `200/80` = 2.5. Training no longer uses 1 policy step (100 ms at
  10 Hz) as a stand-in for teleop→VESC delay: live DR
  `action_latency_steps_range` is `[0, 0]`, and Warp applies a 10 ms
  (2×5 ms) command delay in `apply_command_and_integrate` so replay and
  training share the bag p50 latency.
- Skidpad yaw (~0.52 rad/s IPEM RMSE vs a 0.15 gate) remains unverified.
  Do not treat μ 0.71 as a lateral-friction ceiling from that bag.
- Diagnosis loop is stopped; further plant search needs new bags (OptiTrack
  or non-zero `vy`), not another round of one-knob IPEM hypotheses.
- **Verified (DR, 2026-09-09):** mean plant above is unchanged. Sensor and
  friction **ranges** moved to [0029](0029-bag-referenced-sensor-dr.md).
