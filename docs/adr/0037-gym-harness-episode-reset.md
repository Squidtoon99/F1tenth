# 0037 — Gym certify resets the policy and waits lockstep per tick

- Status: Accepted
- Date: 2026-10-13

## Context

After ADR 0035, certify attaches at the exact spawn, but a warm 180 s run
(245 m, 0 crashes, 7 misses) did not reproduce on cold 3-repeats (0 m,
700+ misses). The score was measuring the harness: GRU hidden state and
command history survived `/pause_sim` between repeats; SAFE→RL did not
reset the recurrent state; the agent's deadline was wall/node clock, so a
cold JIT or a fast lockstep tick reset the GRU; the gym timer could publish
the next unique-stamp observation while `/drive` for the current tick was
still outstanding.

## Decision

1. `sensor_racer` zeros GRU hidden state and command history on
   `/pause_sim false`, `/initialpose`, and SAFE→RL (mode-engage). Gym
   `gym_sensor_bridge` zeros the Warp current-plant slew on unpause.
   *Why:* each certify repeat is an independent episode from a cold
   policy and a stopped car.
2. The validator waits until odom is at spawn **and** warmup scans /
   diagnostics have arrived, then unpauses. The runtime does a dummy
   inference on each episode reset.
   *Why:* the first scored tick must not pay JIT or empty-sensor cost.
3. Lockstep gym (`async_mode=False`) sets `_awaiting_drive` after
   publishing a running observation and skips the publish timer until a
   unique `/drive` stamp arrives. In `tick_on_scan` (gym) mode the agent
   deadline is scan-stamp dt, not wall clock, and a miss does not reset
   the GRU.
   *Why:* the gym cannot advance without the current tick's command, so
   `deadline_misses` is 0 by construction. On-car wall-clock deadlines
   are unchanged.

## Consequences

- On-car `sensor_policy.launch.py` defaults stay unchanged
  (`tick_on_scan` remains false).
- A paused gym may still tick the GRU once on a frozen stamp; unpause
  zeros that state before the first scored step.
- Fast lockstep remains allowed after release; wall pacing is not
  required.
