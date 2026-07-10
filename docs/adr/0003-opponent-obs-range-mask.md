# 0003 — Remove opponent presence channel; range-mask the opponent block

- Status: Accepted
- Date: 2026-07-10

## Context

The 1v1 observation appended a 7-dim opponent-relative block at `[384:391)`, with
the last channel a binary `present` flag. Training zeroed the whole block (including
`present`) when no opponent was relevant. On the car, opponent detection confidence
will eventually come from an LSTM; until then we need a single masking seam that
does not bake detection logic into the shared block builders.

The passing reward already gates on a forward/rear window (`passing_gate_ahead_m`
40 m, `passing_gate_behind_m` 20 m) using wrapped signed arc-length gap.

## Decision

1. Drop the `present` channel. The opponent block is now **6 dims**
   `[rel_x, rel_y, rel_vx, rel_vy, gap_norm, ey_o]` at `[384:390)`; 1v1 `num_obs`
   is **390** (base 384 unchanged). *Why:* all-zeros is the sole “no relevant
   opponent” sentinel; no redundant explicit presence bit.

2. Keep block builders **pure** (`obs_opponent`, `obs_core.build_opponent_block`,
   `rl_obs_core.cpp`): they always emit the 6 relative features with no internal
   masking. *Why:* Python/C++/training parity stays trivial; masking policy lives
   at one caller layer per runtime.

3. Apply masking at the **caller**:
   - **Training/sim** (`f1tenth_env`): zero the 6-dim block per env when wrapped
     signed gap `s_opp - s_self` is outside `[-opp_obs_behind_m, +opp_obs_ahead_m]`
     (defaults 20 m / 40 m, matching the passing gate). Symmetric for the
     opponent's egocentric view.
   - **Deploy** (`vehicle_obs_node`): zero the block when opponent detection is
     not confident (current stand-in: odom timeout); future work swaps in an LSTM
     certainty threshold at this seam.

## Consequences

- Checkpoints trained on 391-dim obs with `present` are **not** compatible; retrain
  or migrate weights manually.
- Parity tests compare **unmasked** builder output; env/deploy masking is tested
  separately.
- `OBS_DEBUG_OPP_PRESENT` in deploy debug scalars is now derived from
  `any(opponent_block != 0)` rather than a dedicated obs channel.
- Follow-up: wire LSTM certainty into `vehicle_obs_node` masking; consider sharing
  the range gate on deploy if desired.
