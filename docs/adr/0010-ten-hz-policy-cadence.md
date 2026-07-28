# 0010 — Align policy control and rewards at 10 Hz

- Status: Accepted
- Date: 2026-07-20

## Context

The recurrent sensor policy was trained and deployed at 20 Hz while retaining the
GT Sophy QR-SAC discount and seven-step return configured for 10 Hz. This halved
their real-time horizons. Reward terms compensated with cadence scaling, but the
policy objective still discounted future outcomes at the faster rate.

## Decision

1. Train and deploy the policy at 10 Hz.
   *Why:* This restores the real-time discount and multi-step horizons associated
   with the adopted QR-SAC hyperparameters.
2. Keep reward coefficients expressed at their canonical 10 Hz rate.
   *Why:* With a 0.1 s control period, cadence-scaled state and event terms use a
   factor of one and elapsed-time penalties use the full control period.
3. Use 1,024 parallel training environments and a 3 million-transition replay.
   *Why:* This retains more sequential history per environment while preserving
   the selected replay capacity.

## Consequences

New checkpoints declare `control_hz=10.0` and are rejected by older 20 Hz
runtimes. Training, evaluation, sensor preprocessing, and on-car policy timers
must use a 0.1 s decision period. The lower parallelism reduces collection
throughput but increases replay history per environment from roughly 36 seconds
to roughly 293 seconds.
