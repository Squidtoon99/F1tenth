# 0042 — Curvature-limited scripted centerline opponent

- Status: Accepted
- Date: 2026-10-17

## Context

The scripted centerline follower slowed in proportion to the current steer
command (`speed_scale = max(0.3, 1 − 0.7 · |δ*| / max_steer)`). On Austin that
cap finished closed-loop laps at a 7.6 m/s cruise target and cut the infield at
7.8 m/s and above. Mean body speed stayed ~4 m/s because the 0.3× floor crawled
every bend. Ego flying-lap pace is ~7.2 m/s.

Austin CSV kinks are ~0.4 m segments with κ ≈ 0.9 m⁻¹. Using that per-segment
κ as `v = sqrt(a_lat / κ)` commands ~2 m/s through a heading jump that is
over in one sample. 3 m heading windows see the real hairpins
(κ_max ≈ 0.7 m⁻¹, v_c ≈ 3 m/s at `a_lat = 6`) without crawling every kink.
A 2.5 m Stanley look point on the 0.4 m kinks is a chord through the infield
once the car is no longer crawling.

## Decision

1. Command `min(v_target, sqrt(a_lat / κ + 2 a_brake s))` from 3 m heading
   chords for 39 m ahead (`for i in range(13)` calling `track_heading_ahead`)
   with `a_lat = 6.0 m/s²` and `a_brake = 1.5 m/s²`. Drop the 0.3×
   steer-proportional scale. Close the speed loop on body speed, not the
   tangent component.
   *Why:* 3 m κ is the corner that actually lasts; `a_brake = 1.5` starts
   slowing ~20 m out of a 9 m/s cruise so the bicycle can actually match the
   envelope. Tangent-speed P-control accelerates when yaw error kills
   `v · t`. The 0.3× floor was what held mean speed at 4 m/s.

2. Cap Stanley lookahead at `min(1.2, max(0.8, 0.20 |v|))` with 1.4× heading
   gain.
   *Why:* 1.2 m of heading on Austin's tightest 3 m window keeps sagitta
   under the 2.5 m `|ey|` bar. Longer chords at 8–9 m/s aim through the
   infield.

Apply the same law in `centerline_tracking_action` (scripted opponent and the
optional residual ego base). Courtyard and IV keep using the same preview.

## Consequences

Closed-loop scripted opponents complete Austin flying laps at 8–9 m/s cruise
with mean body speed in the 7–9 m/s band and `|ey|_max < 2.5 m`. Hairpins
still drop below cruise; they no longer crawl at 0.3×. Training configs that
sample 2–4 m/s are unchanged — this is the follower, not the default speed
range.
