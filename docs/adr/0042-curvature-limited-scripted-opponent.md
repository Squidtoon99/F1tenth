# 0042 — Curvature-limited scripted centerline opponent

- Status: Accepted
- Date: 2026-10-17

## Context

The scripted centerline follower slowed in proportion to the current steer
command (`speed_scale = max(0.3, 1 − 0.7 · |δ*| / max_steer)`). On Austin that
cap finished closed-loop laps at a 7.6 m/s cruise target and cut the infield at
7.8 m/s and above. Mean body speed stayed ~4.1 m/s because the 0.3× floor
applied after the car was already in the corner. Ego flying-lap pace is
~7.2 m/s, so the opponent was not a passable challenge at the speeds the ego
can hold.

## Decision

1. Replace steer-command slow-in with a curvature-limited cruise: sample
   heading change over 4 m chords for 32 m ahead and command
   `min(v_target, sqrt(a_lat / κ + 2 a_brake s))` with `a_lat = 8 m/s²` and
   `a_brake = 6 m/s²`.
   *Why:* Austin hairpins need ~4–5 m/s, but a car 12 m out can still run 9 m/s
   if it has braking room. Preview κ, not the current steer, is what keeps the
   car on the CSV corridor at 8–9 m/s.

2. Keep Stanley cross-track, add `atan(L κ)` feedforward from 0.75 s lookahead
   heading, and drop the 0.3× steer-proportional scale.
   *Why:* heading from the current tangent is late once steer lag (`τ = 0.09 s`)
   and 9 m/s are in play; feedforward turns in before the tire saturates and
   the car goes straight through the infield.

Apply the same law in `centerline_tracking_action` (scripted opponent and the
optional residual ego base). Courtyard and IV keep using the same preview, so
tight maps still slow for their own κ.

## Consequences

Closed-loop scripted opponents complete Austin laps at 8 and 9 m/s cruise
targets without leaving the CSV corridor. Mean speed stays near the target
instead of collapsing to 0.3× in every corner. Tight maps still slow in;
average speed there remains below the configured cruise. Training configs that
still sample 2–4 m/s are unchanged — this is the follower, not the default
speed range.
