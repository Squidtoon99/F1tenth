# 0026 — Normalize actor VESC current to directional limit fraction

- Status: Accepted
- Date: 2026-08-03

## Context

Deploy packed actor `VESC_CURRENT` as raw applied amperes
(`drive_current_a - brake_current_a`), while training packed
`vesc_current_scale_a * applied_effort` (default scale 5.0). On-car current
limits are configuration (`i_drive_max_a` / `i_brake_max_a`) and routinely
differ from that scale, so the same physical actuation produced incompatible
actor channels. Rosbag preprocessing replay showed saturated period-2
longitudinal commands correlated with raw current feedback and high
out-of-distribution rates under stand-in replay.

The 1,097-D layout offsets are unchanged. Only the preprocessing semantics of
the existing `VESC_CURRENT` slot change.

## Decision

1. Pack actor current feedback as
   `signed_applied_current_a / configured_directional_current_limit_a`,
   using drive vs brake limits by sign and preserving sign.
2. Share one helper (`f1tenth_policy.applied_current_fraction`) between
   training reconstruction/deploy packing; the Warp sim packs
   `applied_effort` directly because it is already that fraction.
3. Keep raw amperes on actuator diagnostics / fitting surfaces; do not put
   raw amps in the actor observation.
4. Fail closed on non-positive or non-finite configured limits.
5. Bump `observation_preprocessing_version` **2 → 3** and refuse to load
   legacy artifacts under the new semantics.
6. Remove `vesc_current_scale_a` from the training sensor config.

## Consequences

- Existing sensor-policy checkpoints with preprocessing version ≤2 are
  incompatible and must not be deployed; policies need retrain + fresh
  observation normalization under version 3.
- On-car `i_drive_max_a` / `i_brake_max_a` are now part of the observation
  contract edge: changing limits changes the actor current channel scale.
- Sim2real amplitude mismatch for the current channel is closed for new
  artifacts; bag-era exact action parity remains blocked until a matching
  checkpoint exists.
