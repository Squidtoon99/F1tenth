# 0042 — Curvature-limited scripted centerline opponent

- Status: Accepted
- Date: 2026-10-17

## Context

The scripted centerline follower slowed in proportion to the current steer
command (`speed_scale = max(0.3, 1 − 0.7 · |δ*| / max_steer)`). On Austin that
cap finished closed-loop laps at a 7.6 m/s cruise target and cut the infield at
7.8 m/s and above. Mean body speed stayed ~4 m/s because the 0.3× floor crawled
every bend. Ego flying-lap pace is ~7.2 m/s.

Austin T1 is a ~0.4 m CSV kink with κ ≈ 0.9 m⁻¹ on a 1.1 m half-width. Bicycle
`v = sqrt(μ g / κ) ≈ 2.8 m/s` if the car follows that centerline. Steer lag
`τ = 0.09 s` and `δ_max = π/60` per 20 Hz tick add meters of path before
`max_steer` is in. A 2.5 m Stanley look point on that kink is a chord through
the infield once the car is no longer crawling.

## Decision

1. Walk 40 m of centerline segments and command
   `min(v_steer, sqrt(a_lat / κ_seg + 2 a_brake s))` with `a_lat = 4.0 m/s²`
   and `a_brake = 1.2 m/s²`. Keep
   `v_steer = max(0.3, 1 − 0.7 |δ*| / max_steer) · v_target`.
   *Why:* per-segment κ catches T1 20 m out. Dropping the 0.3× floor, raising
   it to 0.45, replacing it with a quadratic, or switching to 3 m heading
   chords all produce `|ey|_max > 2.5 m` (infield) at 8–9 m/s cruise.

2. Cap Stanley lookahead at `min(2.5, 0.5 |v|)` with 1.4× heading gain.
   *Why:* this is the look that finished 8.0–9.0 m/s cruise on the CSV.
   Shorter looks with faster corners still left the 2.5 m `|ey|` bar.

3. Accept a centerline corner limit: on-track Austin mean body speed stays
   ~4 m/s. A 7–9 m/s flying-lap mean is not available without leaving the
   track or cutting the infield.
   *Why:* T1 and the other 0.4 m kinks require ~2.5–4 m/s on the centerline.
   Any closed-loop attempt to take them at ≥5 m/s went to `|ey|_max` 6–13 m.
   A 7 m/s mean needs those kinks at ≥5 m/s or a raceline that uses more
   than 2.5 m of lateral room.

Apply the same law in `centerline_tracking_action` (scripted opponent and the
optional residual ego base). Courtyard and IV keep using the same preview.

## Consequences

Closed-loop scripted opponents complete Austin flying laps at 8 and 9 m/s
cruise targets without cutting the infield (`|ey|_max ≈ 1.4–1.6 m`). Mean
body speed stays ~4 m/s. Straights see the cruise; hairpins crawl at 0.3×.
This does not keep up with a 7.2 m/s ego on average. Training configs that
sample 2–4 m/s are unchanged — this is the follower, not the default speed
range.
