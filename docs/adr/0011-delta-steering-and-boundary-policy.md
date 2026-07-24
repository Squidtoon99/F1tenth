# 0011 — Scope delta steering and footprint boundary semantics

- Status: Superseded by 0012 (sensor stack)
- Date: 2026-07-23

## Context

The asymmetric recurrent sensor experiment follows Lee et al. (2025), whose
steering action is a bounded change in steering angle. Existing symmetric racer
artifacts instead encode an absolute normalized steering target. Changing the
shared action layout would silently reinterpret existing checkpoints.

The experiment also needs boundary behavior that distinguishes first contact,
continued off-course driving, and a complete course skip. Centerline penetration,
three-tire gates, continuous wall shaping, and wall-impact termination do not
represent those events.

## Decision

1. Add artifact-scoped steering metadata with `absolute` as the default and use
   `delta` only for the asymmetric sensor experiment.
   *Why:* Existing symmetric configs and checkpoints retain their action meaning.
2. Interpret a normalized delta action as at most three degrees per 10 Hz
   decision, integrate it after action latency, clamp the realized steering to
   ±0.33 rad, and bypass the absolute-target steering lag.
   *Why:* The integrated action is itself the steering rate limit.
3. Store normalized deltas in replay while using realized absolute steering
   angles and their differences in recurrent observations and Lee rewards.
   *Why:* Actor/critic actions retain policy coordinates while theta and delta
   match the paper's physical definitions.
4. For this experiment, mark off-course when any projected footprint crosses the
   true boundary, apply a one-time -4 contact penalty, mask progress, and apply
   `-0.01 * dt * speed_kph²` while off-course.
   *Why:* This separates contact from elapsed off-course cost without continuous
   wall or impact shaping.
5. Terminate only when the complete projected footprint is outside either side
   and apply `-0.01 * 1 s * speed_kph²` on that transition.
   *Why:* Partial excursions remain recoverable while complete course skips end
   immediately.

## Consequences

Delta artifacts declare both `steering_action_mode` and
`steering_delta_max_rad`; the sensor runtime rejects missing or incompatible
metadata. Simulation and deployment maintain steering integrator state and clear
it with episode, recurrent-history, ownership, or safety resets.

**Superseded for the sensor stack by [0012](0012-lee-core-and-fixed-opponents.md):**
the sensor path no longer retains absolute-action or alternate course-limit
compatibility modes. Unrelated classical experiments remain outside that ADR.
