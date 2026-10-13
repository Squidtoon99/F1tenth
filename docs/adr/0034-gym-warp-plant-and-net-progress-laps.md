# 0034 — Gym certify uses the Warp current plant and net-progress laps

- Status: Accepted
- Date: 2026-10-13

## Context

`STACK=sensor_policy` gym certify scored start-line wraps as laps (3.65–16 s
on a 17 m courtyard) and commanded gym SPEED as `longitudinal × 5 m/s`. Warp
eval of the same v5 checkpoints is Austin (~421 m, 78–122 s flying, 0 crashes).
Those gym numbers were not comparable.

## Decision

1. Count a lap only when wrap-aware net centerline progress since the last
   reset is at least one track length, matching `eval_full_lap.py`. A crash
   (off-track rising edge or teleport larger than `0.1 L`) clears progress.
   *Why:* start-line crossings after a reset are not laps.
2. Map applied drive/brake current through the reduced-bicycle longitudinal
   plant (80 A → 26.5 N, 40 A → 23.1 N, 200 A/s slew, mass 3.444 kg, 5 ms
   substeps) and send the resulting `vx` as the gym SPEED command. Drop the
   `× 5 m/s` pin.
   *Why:* gym is SPEED-mode; this is the current→force→vx map the policy was
   trained against, not a speed setpoint.
3. Wire sensor-policy gym to the Austin occupancy grid and centerline used by
   Warp eval, and run certify for 180 s sim so a flying lap can complete.

## Consequences

- Gym still uses the f1tenth_gym SPEED integrator; only the commanded speed
  comes from the Warp longitudinal map. Lateral gym dynamics are not Warp.
- On-car `sensor_policy.launch.py` is unchanged.
- Certify scores are still not Jetson/host-networked car time.
