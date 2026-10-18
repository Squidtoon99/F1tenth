# 0042 — Curvature-limited scripted centerline opponent

- Status: Accepted
- Date: 2026-10-17

## Context

The scripted centerline follower slowed in proportion to the current steer
command (`speed_scale = max(0.3, 1 − 0.7 · |δ*| / max_steer)`). On Austin that
cap finished closed-loop laps at a 7.6 m/s cruise target and cut the infield at
7.8 m/s and above. Mean body speed stayed ~4 m/s because the 0.3× floor crawled
every bend. Ego flying-lap pace is ~7.2 m/s.

Austin T1 is a ~0.4 m CSV kink with κ ≈ 0.9 m⁻¹. 3 m heading chords dilute
that peak and still arrive at ~7 m/s. Per-segment preview sees it 20 m out.
A 2.5 m Stanley look point on that kink is a chord through the infield once
the car is no longer crawling.

## Decision

1. Command `min(v_target, sqrt(a_lat / κ_seg + 2 a_brake s))` by walking 40 m
   of centerline segments with `a_lat = 7.0 m/s²` and `a_brake = 1.5 m/s²`.
   Drop the linear 0.3× steer-scale. If `|δ*| > 0.85 max_steer`, also cap
   `v` at 4.5 m/s. Close the speed loop on body speed, not `v · t`.
   *Why:* per-segment κ catches T1. The 0.3× floor crawled every moderate
   bend; a saturated-steer cap only hits hairpins. Tangent-speed P-control
   accelerates when yaw error kills `v · t`.

2. Cap Stanley lookahead at `min(1.2, max(0.8, 0.20 |v|))` with 1.4× heading
   gain.
   *Why:* 1.2 m of heading keeps hairpin sagitta under the 2.5 m `|ey|` bar.

Apply the same law in `centerline_tracking_action` (scripted opponent and the
optional residual ego base). Courtyard and IV keep using the same preview.

## Consequences

Closed-loop scripted opponents complete Austin flying laps at 8–9 m/s cruise
without cutting the infield (`|ey|_max < 2.5 m`). Hairpins drop to ~2.5–4.5 m/s;
moderate bends stay near cruise. Training configs that sample 2–4 m/s are
unchanged — this is the follower, not the default speed range.
