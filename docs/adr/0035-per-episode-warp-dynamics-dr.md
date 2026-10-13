# 0035 — Per-episode Warp dynamics domain randomization

- Status: Accepted
- Date: 2026-10-13

## Context

Warp trains a reduced bicycle (ADR 0033) with labeled prior Izz **0.13**,
μ **0.71**, `lat_stiffness_scale` **11** (C_f/C_r ≈ 191/181 N/rad), mass
**3.444 kg**, first-order steer lag **0.09 s**, and `drive_scale` pinned at
**1.0** (ADR 0029). Gym ST used for certify is lighter in yaw (Izz 0.047),
higher μ (1.05), softer cornering (~94/101 N/rad), and holds SPEED in
corners. Existing DR randomizes mass, μ, sensors, and latency; Izz,
C_α, steer lag/rate, and motor gain stay global `SimParams`. A policy
trained on the Warp mean plant then meets a gym plant that yaws more
eagerly at low speed and keeps speed in fast corners.

## Decision

1. Sample per env at reset, on GPU tensors: Izz, μ, C_f, C_r, mass,
   steer lag, steer rate, and `drive_scale`. Missing range keys keep the
   current formula (C_f/C_r from `lat_stiffness_scale · Fz(mass)`; Izz and
   lag collapsed to the nominal; rate unbounded).
   *Why:* opt-in; DEFAULT and Austin baseline-v5 stay byte-identical in
   dynamics. Rudin 2021 and Swift (Kaufmann 2023) randomize inertial and
   actuator parameters per episode on GPU.
2. Ranges in `training/configs/austin_e2e_ppo_dr.json` cover Warp and gym
   ST: Izz `[0.045, 0.14]`, μ `[0.65, 1.1]`, C_f/C_r `[90, 200]` N/rad,
   mass ±10% of 3.444 kg, steer lag `[0.03, 0.15]` s, steer rate
   `[2.5, 5.0]` rad/s, `drive_scale` ±20%.
   *Why:* gym Izz 0.047 and Warp 0.13; gym μ 1.05 vs Warp 0.71 / ADR 0029
   `[0.66, 0.85]`; gym C_α ~94–101 vs Warp ~181–191; gym mass 3.74; gym
   `sv_max` 3.2 rad/s vs Warp first-order peak ~4 rad/s; gym `a_max` 9.51
   vs Warp 7.69 (ratio 1.24, clipped to the requested ±20%).
3. Do not retune the Warp mean plant. ADR 0029 still pins DEFAULT
   `drive_scale_range` and the conservative μ band; the Austin DR config
   is the explicit widening.

## Consequences

- Training env metrics grow by `dr/yaw_inertia`, `dr/cornering_stiffness_*`,
  `dr/steer_time_constant`, `dr/steer_rate_max`.
- Eval / tournament paths that already set `domain_randomization.enabled`
  false keep the nominal bicycle.
- Gym SPEED-hold vs Warp corner speed bleed is not randomized; only the
  listed plant scalars are.
