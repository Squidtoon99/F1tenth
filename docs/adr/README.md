# Architecture Decision Records (ADRs)

Short, dated records of significant architectural decisions and their rationale.
Add a new numbered file per decision; never rewrite history — supersede instead.
Copy [0000-template.md](0000-template.md) to start a new one.

Write an ADR when a change alters the repo's structure, a public interface or the
observation/action contract, the build/deploy shape, or accepts a non-obvious
trade-off. Routine work is captured by the PR instead (see
[../development.md](../development.md)).

- [0000](0000-template.md) — Template (copy this).
- [0001](0001-single-monorepo-and-generic-car-image.md) — Single monorepo, single
  colcon workspace, and a generic car image with per-car runtime config.
- [0002](0002-physx-tyre-slip-and-tyre-load-observation.md) — PhysX-grounded tyre
  slip and a per-wheel tyre-load observation block (380 → 384 layout change).
- [0003](0003-opponent-obs-range-mask.md) — Remove opponent `present` channel;
  6-dim opponent block with caller-side range/certainty masking (391 → 390).
- [0004](0004-on-car-stack-integration.md) — Integrate the on-car localization,
  algorithmic drivers, and deploy image from `shereef@f1tenth`.
- [0005](0005-force-mode-vesc-actuation.md) — Force-mode VESC actuation via
  current/brake commands (superseded).
- [0006](0006-current-only-vesc-actuation.md) — Current/force is the only
  longitudinal action semantic; speed-mode removed; policy_format_version 2.
- [0007](0007-warp-simulator.md) — Warp physics backend for large-batch training.
- [0008](0008-sensor-policy-runtime-image.md) — Dedicated no-localization
  sensor-policy launch and JetPack iGPU runtime image, isolated from racing.
- [0009](0009-opponent-relative-acceleration.md) — Privileged opponent relative
  acceleration in an 8-dim block (390 → 392).
- [0010](0010-ten-hz-policy-cadence.md) — Align training, rewards, and deployment
  policy decisions at 10 Hz.
- [0011](0011-delta-steering-and-boundary-policy.md) — Scope three-degree
  steering deltas and full-footprint boundary behavior to the sensor experiment
  (partially superseded by 0012 for the sensor stack).
- [0012](0012-lee-core-and-fixed-opponents.md) — Canonical Lee/ADR-0011 sensor
  core, fixed champion opponents, and shared `f1tenth_policy` module
  (amended 2026-10-12: 20 Hz γ/λ rescale, multi-scenario mix).
- [0013](0013-first-footprint-wall-contact.md) — Canonical first-footprint wall
  termination with one linear-speed Lee barrier reward.
- [0014](0014-paper-literal-wall-event-scale.md) — Paper-literal wall-event
  reward without control-period scaling.
- [0015](0015-progress-normalized-steering-history.md) — Progress-normalize
  default `steering_history` to 0.5 (\(\approx 0.1\times\) Lee \(\lambda^h\))
  (superseded by 0016).
- [0016](0016-revert-steering-history-to-paper-literal.md) — Revert default
  `steering_history` to paper-literal 5.0 after clean legacy-ladder A/B (+0.43
  m/s @ 18M).
- [0017](0017-tightened-opponent-spawn-gaps.md) — Tighten default opponent spawn
  gap ceiling 80 → 58 m for ~70% presence at 5–6 m/s without near-total coupling.
- [0020](0020-warm-start-actor-freeze.md) — Critic warm-up via
  `actor_freeze_transitions` for `--init-ckpt` warm-start runs.
- [0021](0021-config-swappable-ppo-training.md) — Config-selectable recurrent PPO
  with on-policy rollouts and unchanged sensor-policy artifacts.
- [0022](0022-ppo-advantage-filtering.md) — Optional Gigaflow-style PPO advantage
  filtering with fixed-κ threshold (superseded by 0023).
- [0023](0023-ppo-advantage-filter-quantile.md) — Per-rollout quantile advantage
  filter (95% default retention) replacing unstable fixed-κ/EWMA rule.
- [0024](0024-ppo-kl-removal-entropy-wall-penalty.md) — Remove PPO KL early-stop,
  add entropy bonus, reduce wall penalty.
- [0025](0025-sensor-policy-image-no-localization.md) — Sensor-policy image excludes
  localization and range_libc (supersedes 0008 workspace scope).
- [0026](0026-normalized-current-observation.md) — Actor `VESC_CURRENT` is signed
  applied current divided by the configured directional limit; preprocessing v3.
- [0027](0027-current-control-bag-identification.md) — 80/40 current-control bag
  identification (mass 3.444 kg); Verified in 0028.
- [0028](0028-promote-80-40-current-plant.md) — Promote the verified 80/40 Warp
  plant (mass 3.444 kg, 26.5/23.1 N, μ 0.71) into training defaults.
- [0029](0029-bag-referenced-sensor-dr.md) — Pin `drive_scale`, conservative
  friction `[0.66, 0.85]`, bag-referenced IMU/VESC DR; scenario JSON inherits.
- [0033](0033-reduced-bicycle-simulator.md) — Reduced bicycle plant, occupancy
  contact, and 20 Hz sensor-policy cadence (supersedes 0002 plant and 0010
  sensor cadence).
- [0034](0034-gym-warp-plant-and-net-progress-laps.md) — Gym certify uses Warp
  current→vx and eval_full_lap net-progress laps on Austin.
- [0035](0035-gym-lockstep-paused-spawn.md) — Gym certify starts paused at spawn,
  fails on lockstep misses, and reports 3-repeat mean/var.
- [0036](0036-per-episode-warp-dynamics-dr.md) — Opt-in per-episode Warp
  Izz / μ / C_α / mass / steer-lag / drive-scale DR covering gym ST.
- [0037](0037-gym-harness-episode-reset.md) — Gym certify resets GRU /
  command history per episode, warms up before unpause, and waits for
  `/drive` before the next lockstep observation.
- [0038](0038-gym-ros-trinary-occupancy.md) — Gym occupancy uses ROS
  occupied/free thresholds so unknown cells stay max-range like Warp.
- [0039](0039-action-chunk-receding-horizon-ppo.md) — Receding-horizon PPO
  action chunks (H=4, execute first tick, 20 Hz plant).
- [0040](0040-lidar-dreamer-world-model.md) — LiDAR Dreamer RSSM
  (Cartesian-polar + LS3C), imagination-trained actor, encoder+actor deploy.
- [0041](0041-tinylidarnet-encoder.md) — TinyLidarNet 1D CNN on raw 1081
  beams; GRU and PPO kept; no 32-bin pool.
- [0042](0042-sim-ready-v2-fitted-courtyard-plant.md) — Austin v6 and courtyard
  PPO use the frozen SIM_READY_V2 bag-fit plant; Galaxy/defaults stay unfitted.
- [0042](0042-drop-fixed-speed-target.md) — Drop TAL and the 3.5 m/s horizon
  constant; freeze per-track episode seconds; start speeds are per-track
  (courtyard rest, Austin 2 m/s flying / 0–5 reset).
- [0043](0043-delete-tal-reward.md) — Delete the TAL speed-match reward
  and its config key so a later patch cannot turn it back on.
