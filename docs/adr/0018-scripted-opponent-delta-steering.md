# 0018 — Speed-aware scripted opponent steering under delta actions

- Status: Accepted
- Date: 2026-07-26

## Context

Mixed-opponent training assigns 10% of rows to a Warp-kernel scripted centerline
follower. After ADR 0011 delta steering, the follower still computed a lateral P
target in radians and converted it with `(target − current_steer) /
steering_delta_max` using gains tuned for absolute normalized steering. Typical
errors saturated the ±1 delta command (~98% duty cycle), causing weave and poor
speed tracking.

## Decision

Replace the lateral P term with a speed-scaled cross-track `atan(k·ey / (|v| +
0.5))` term, keep heading P on `heading_error`, scale the combined target by
`steering_delta_max / max_steer`, and close the loop with the existing delta
inner step `(target − current_steer) / steering_delta_max`.

*Why:* Cross-track demand must shrink as forward speed grows; the inner loop
then commands a bounded rate toward a target steer angle instead of bang-bang
against the slew limit.

Apply the same logic in `scripted_opponent_action` and
`scripted_physics_opponent_action`. The Python `ScriptedCenterlineOpponent` path
remains bypassed by Warp (ADR 0012) but was aligned to the same cross-track
formula for consistency.

## Consequences

Scripted-mode opponents track centerline with low steer saturation and reach
commanded cruise speed in closed-loop env tests. Long-horizon runs that include
scripted rows must be restarted when this changes — opponent mix is part of the
training distribution.
