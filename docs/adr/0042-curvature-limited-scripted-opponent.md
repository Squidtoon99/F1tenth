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

1. Replace steer-command slow-in with a per-segment curvature preview: walk
   40 m of centerline segments and command
   `min(v_target, sqrt(a_lat / κ_seg + 2 a_brake s))` with `a_lat = 4.0 m/s²`
   and `a_brake = 1.2 m/s²`.
   *Why:* per-segment κ keeps hairpin peaks; `a_brake = 1.2` starts slowing
   ~35 m out so steer lag and the 3 deg/tick slew can still make the corner
   from a 9 m/s cruise. `a_lat` sits below `μ g`. The loop only advances
   `segment` / `covered` — the same pattern as `track_heading_ahead`.

2. Heading target is the centerline 1 m ahead with 2× Stanley heading gain.
   Drop the 0.3× steer-proportional scale and drop long-chord feedforward.
   *Why:* a 2.5 m lookahead on Austin's R ≈ 1.1 m hairpins is a chord through
   the infield. 1 m of heading preview turns in without that sagitta; extra
   heading gain closes yaw before the 1.1 m half-width is spent.

Apply the same law in `centerline_tracking_action` (scripted opponent and the
optional residual ego base). Courtyard and IV keep using the same preview, so
tight maps still slow for their own κ.

## Consequences

Closed-loop scripted opponents complete Austin laps at 8 and 9 m/s cruise
targets without leaving the CSV corridor or cutting the infield. Straights
hold the cruise; hairpins drop to ~2.5 m/s. Tight maps still slow in.
Training configs that still sample 2–4 m/s are unchanged — this is the
follower, not the default speed range.
