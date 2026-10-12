# 0012 — Canonical Lee sensor core and fixed opponents

- Status: Accepted (course-limit clause superseded by 0013)
- Date: 2026-07-23
- Supersedes: [0011](0011-delta-steering-and-boundary-policy.md) promise to retain
  legacy absolute-action and alternate course-limit paths for the sensor stack

## Context

The asymmetric sensor experiment accreted alternate reward forms, actor types,
artifact migrations, rolling self-play, and duplicated train/deploy policy code.
Lee et al. 2025 plus a short list of F1TENTH adaptations define the only path we
intend to operate.

## Decision

1. Make one Lee / ADR-0011 training path canonical: 1,097-D sensor actor,
   392-D privileged critic, LiDAR-CNN+GRU, delta steering (±3° at 10 Hz),
   footprint off-course with masked progress, continuous
   `-0.02 * dt * speed_kph²` OOB cost, and full-car-out termination with a
   ten-second speed-squared impact.
2. Replace rolling self-play with a seeded fixed champion pool: one weighted
   champion selected once at startup, immutable across replay-full reinit, with
   per-reset 50/50 centerline vs champion sampling and 50% policy speed caps in
   5–7 m/s (high-end skewed).
3. Put layout, GRU actor, normalizer, artifact validation, and delta-steering
   semantics in editable `libs/f1tenth_policy`, with training and sensor deploy
   as thin adapters.
4. Delete alternate actor types, reward-form switches, `zero_tyre_slip_obs`,
   one-shot overtake bonuses, wall/impact shaping, and Python mixed-opponent
   controllers bypassed by Warp.

## Consequences

Sensor configs no longer carry absolute-steering or center-streak OOB modes.
Classical 392-D deploy remains separate (ADR 0008). Fixed-opponent runs require
`fixed_opponents.entries` in config. Shared policy changes are picked up via
editable install without rebuilding images for Python-only edits.

**Superseded in part by
[0013](0013-first-footprint-wall-contact.md):** decision 1's course-limit reward
and termination clause is replaced. Its sensor architecture and the
fixed-opponent decisions remain in force.

## Amendment (2026-10-12)

Parity status of this Lee/ADR-0011 sensor path against the reference paper.
Original decisions above are unchanged.

- **Reference.** arXiv 2504.09021 (Lee et al., vision-based GT7 agent with
  asymmetric QR-SAC). Pure simulation; no sim-to-real. Domain randomization and
  on-car transfer are our additions.
- **Policy.** The actor gets onboard sensors only (LiDAR, IMU, VESC speed,
  command history) as a 1097-d LiDAR-CNN+GRU. The critic gets privileged 392-d
  state through a separate feed-forward network. LiDAR stands in for the
  paper's camera. No encoder is shared.
- **Algorithm.** The live training path is PPO
  (`courtyard_2_e2e_ppo.json`, `courtyard_2_e2e_ppo_multi_scenario.json`,
  `galaxy_1_scripted_v3_80a_ppo.json`). That departs from QR-SAC. QR-SAC
  remains config-selectable.
- **Now true on `experiments/e2e-ppo`** (PRs #31–#33):
  - 20 Hz sensor-policy cadence is live
    ([0033](0033-reduced-bicycle-simulator.md)). Privileged critic contract
    cadence stays 10 Hz.
  - Time-based γ/λ rescale is on by default (`horizon_from_control_dt`). At
    20 Hz, PPO γ≈0.9948 and λ≈0.9747 from the 10 Hz refs 0.9896 / 0.95.
  - Multi-scenario mixing exists
    (`courtyard_2_e2e_ppo_multi_scenario.json`): flying / standing /
    IV_2026 with `scn/<name>/...` metrics.
  - Timeout bootstrapping: PPO bootstraps pure `term_timeout` rows; QR-SAC
    replay stores timeouts as truncation and drops truncated n-step tails.
  - Standing starts have a dedicated `solo_standing` scenario as well as
    `reset_stationary_probability` mixing.
- **Known gaps still open:**
  - Lap-completion eval: `eval_interval_transitions` still defaults to 0.
    Per-scenario `scn/<name>/lap_completion` is a training diagnostic, not
    a held-out eval protocol.
- **Legacy contract.** `libs/f1tenth_contract` and `observation_layout.hpp`
  hold the 392-d symmetric privileged layout. Only the legacy on-car graph
  (`vehicle_obs` → `policy_inference` → `drive`) uses it.
