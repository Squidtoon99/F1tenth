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

Austin's CSV hairpins are ~0.4 m segments with κ ≈ 0.9 m⁻¹. Averaging that
heading change over a 2–4 m chord dilutes κ, and `v² = v_c² + 2 a s` with
`a_brake = 5–6 m/s²` then still allows 8–9 m/s a few meters out. The bicycle
cannot both brake and slew `max_steer` (`δ_max = π/60` per 20 Hz tick plus
`τ = 0.09 s` lag) in that distance, so it drives straight through the infield.
Uncapped `0.75 |v|` lookahead at 8–9 m/s is a 6–7 m chord that also aims
through the infield.

## Decision

1. Add a per-segment curvature preview: walk 40 m of centerline and command
   `min(v_steer, sqrt(a_lat / κ_seg + 2 a_brake s))` with `a_lat = 4.0 m/s²`
   and `a_brake = 1.2 m/s²`. Keep
   `v_steer = max(0.3, 1 − 0.7 |δ*| / max_steer) · v_target`.
   *Why:* per-segment κ keeps 0.4 m hairpin peaks; `a_brake = 1.2` starts
   slowing ~35 m out of a 9 m/s cruise so the 3 deg/tick slew and `τ = 0.09 s`
   lag can still make the corner. Dropping the 0.3× floor lets the bicycle
   arrive hot and cut the infield even with preview. The loop only advances
   `segment` / `covered` — the same pattern as `track_heading_ahead`.

2. Cap Stanley lookahead at `min(2.5, 0.5 |v|)` and use 1.4× heading gain.
   *Why:* uncapped `0.75 |v|` at 8–9 m/s is a 6–7 m chord through the infield.
   The 2.5 m cap plus 0.3× corner crawl is the law that already finished
   8.0–8.5 m/s; preview extends that to 9 m/s.

Apply the same law in `centerline_tracking_action` (scripted opponent and the
optional residual ego base). Courtyard and IV keep using the same preview, so
tight maps still slow for their own κ.

## Consequences

Closed-loop scripted opponents complete Austin flying laps at 8 and 9 m/s
cruise targets without cutting the infield. Peak |ey| stays ~1.4 m on the
1.1 m half-width CSV (a scrape, not a cut). Mean body speed stays ~4 m/s
because the 0.3× steer-scale still crawls hairpins; straights see the
cruise. Tight maps still slow in. Training configs that still sample
2–4 m/s are unchanged — this is the follower, not the default speed range.
