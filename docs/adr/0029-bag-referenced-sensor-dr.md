# 0029 — Bag-referenced IMU / VESC DR and conservative friction band

- Status: Accepted
- Date: 2026-09-09

## Context

[ADR 0028](0028-promote-80-40-current-plant.md) promoted the Sep 6 champion
**mean** plant (mass 3.444 kg, 80/40 A, 26.5/23.1 N, μ 0.71, 10 ms delay) and
left domain randomization as a wide sim-to-real band: `tire_friction_range`
`[0.39, 1.03]`, Lee-era `drive_scale_range` `[0.78, 1.44]`, and Courtyard/Galaxy
scenario JSON that silently widened IMU/VESC DR past the static-on measured
block (`±0.4` accel bias, VESC speed `±0.3` m/s). Warp applies `drive_scale`
**after** the `μ m g` clamp, so those ranges undid the longitudinal fit.
Particle-filter pose is not mocap; Pacejka α remains unidentifiable.

Open-loop Warp IPEM replay (`calibration/warp_physics_verifier.py --stage imu`)
on `accel_4` / holdout `accel_1` locked the driver IMU long axis to **`-ay`**
(corr 0.93 vs odom `a_x`, matching ADR 0027) and gated holdout plateau `a_x` at
μ endpoints with `drive_scale=1`. Dynamic IMU vs plant `a_x` residual is large
(~4 m/s² p95): that is plant/IMU mismatch, not sensor noise. Accel-bag coast
slices (`n_idle=45`) give vibration std ~0.073 m/s² and a ~0.79 m/s² idle
offset (gravity leak / not a `static_on` bag).

## Decision

1. Pin `drive_scale_range` to **`[1.0, 1.0]`**. Identified `f_drive_max` /
   `f_brake_max` already absorb drivetrain efficiency.
   *Why:* the Lee leftover mid-1.11 scale was not in the bags and randomizes
   the fitted envelope after the friction clamp.
2. Keep mean `tire_friction=0.71`. Replace `[0.39, 1.03]` with **`[0.66, 0.85]`**.
   *Why:* μ=0.55 (odom-force implied) failed the holdout 0.3 m/s² plateau gate;
   0.66 is the first green low end. 0.85 is ~+20% dry slack, not skidpad p95
   `|a|/g` as μ, and not ice (0.39 caps drive at ~13 N).
3. Lock IMU conversion to body-forward **`-ay`** (SI) before any IMU DR. Do not
   widen accel bias to the dynamic residual or to coast idle p95 (~0.88). Bias
   DR stays the static-on prior **`±0.15`**. Coast idle std sets accel noise
   **`[0.073, 0.145]`** and gyro noise **`[0.013, 0.026]`**. VESC current DR
   **`±0.05`**; speed DR from idle odom jitter **`±0.032` / 0–0.022**.
   *Why:* idle-in-accel-bags is not `static_on`; a −0.79 m/s² coast offset is
   on-car bias/level, not a training randomization. Plant mismatch must not
   enter DR.
4. Courtyard and Galaxy scenario JSON keep only scenario DR (`lidar_dropout`,
   optional `action_latency_steps`). They inherit sensor/plant DR from
   `DEFAULT_CONFIG`.
   *Why:* scenario files must not silently reopen `±0.4` IMU or `±0.3` m/s VESC
   overlays.

## Consequences

- Next training jobs pick up the new DEFAULT DR. The live
  `courtyard-2-80a-continuous` freeze is historical; restart is a separate
  decision.
- `--stage imu` writes gitignored `calibration/results/dr_proposal.json`.
  Friction low end is re-gated on holdout plateau `a_x` whenever the plant
  mean changes.
- On-car `sensor_policy.yaml` bias is unchanged. The −0.79 m/s² coast offset
  after `-ay` conversion is a deploy calibration follow-up, not this DR patch.
- Leftover skidpad yaw (~0.52) is still unverified; this ADR does not fit
  Pacejka B/C/E.
