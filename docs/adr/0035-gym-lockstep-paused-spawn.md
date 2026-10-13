# 0035 — Gym certify starts paused at spawn and repeats with mean/var

- Status: Accepted
- Date: 2026-10-13

## Context

Sensor-policy Austin certify was not reproducible. Fast lockstep (5–13×
realtime) let the agent drive the car 6–14 m off spawn before the validator
attached, so `start_xy` was garbage and `deadline_misses` ran 1300–1800.
Pacing the gym at 1× wall time recovered a valid attach once (180 s / 232 m)
and failed to reproduce it on a 400 s rerun. The gym already stepped one
physics tick per unique `/drive` stamp; it did not hold the configured spawn
until scoring started.

## Decision

1. Lockstep gym (`async_mode=False`) **starts paused** at the configured
   centerline spawn and keeps publishing scan/odom/clock without stepping.
   `/pause_sim false` resets to that spawn, then the next unique `/drive`
   advances physics. `/initialpose` works while paused.
   *Why:* the validator can attach at the exact pose; wait-graph delays no
   longer burn track. Gym still blocks on drive once released.
2. The sensor-policy validator holds `/pause_sim true`, waits until odom is
   within `spawn_tol` of the centerline spawn, then releases. It fails the
   run if `start_xy` is off spawn or `deadline_misses` exceed zero.
   *Why:* a score with a missed step or a drifted origin is not a lap
   measurement.
3. `tools/sim.sh certify` defaults to **three repeats** and writes mean/var
   of progress, laps, crashes, misses, speed, offtrack, and `desired_hz`.
   *Why:* one lucky attach is not a certify result.

## Consequences

- Fast lockstep is allowed after release; wall pacing is not required.
- On-car `sensor_policy.launch.py` is unchanged.
- A paused gym with a frozen sim-time stamp ticks the GRU at most once
  before release, so wait duration does not change the first inference.
