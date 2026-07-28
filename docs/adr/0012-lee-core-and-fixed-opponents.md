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
