# 0039 — Action-chunk receding-horizon PPO

- Status: Accepted
- Date: 2026-10-16

## Context

The 20 Hz LiDAR GRU actor emits a 2-d command `(throttle, Δsteer)` every
control tick. High-frequency steer chatter is a known failure mode on this car.
CAPS-family losses already ran or are live. Receding-horizon action chunking
(Zhao et al., ACT, RSS 2023; Chi et al., Diffusion Policy, RSS 2023 §3) makes
smoothness a representation: one decision covers H ticks, the plant still
receives a command every 50 ms, and unused tail actions are not executed.

This is not action-repeat and not a 10 Hz policy. H=8 would be 400 ms of open
loop; this ADR starts at H=4, execute the first tick only.

## Decision

1. Keep `env.num_actions=2` as the **plant** command dim. `env.action_horizon`
   (default 1) selects the actor head width `2H`. Default off is H=1.
   *Why:* existing 2-d artifacts, Warp `step`, and QR-SAC stay unchanged.
2. The squashed-Gaussian head outputs a length-`2H` chunk. PPO log-probability
   covers the full chunk. The trainer, eval, and `sensor_racer` apply only the
   first 2-d command and cache the unused tail until the next 20 Hz query.
   *Why:* receding horizon re-plans every tick; executing the tail would be
   open-loop action repeat.
3. H>1 requires `algorithm=ppo` and `policy_format_version>=5`. Residual-base
   mixing is rejected with chunks.
   *Why:* QR-SAC replay and residual 2-d bases are plant-shaped; chunk artifacts
   must not load as 2-d policies.

## Consequences

- The Austin action-chunk arm turns the flag on with H=4 /
  `policy_format_version` 5. Control stays 20 Hz (`sim_dt=0.005`,
  `control_interval=10`).
- Eval scores the executed 2-d trace, not the unused tail.
- Deployment `sensor_racer` sizes action buffers from `actor.act_dim` and
  realizes the first plant command.
