# 0019 — Per-episode fixed opponent pool sampling

- Status: Accepted
- Date: 2026-07-27

## Context

ADR 0012 replaced rolling self-play with a seeded fixed champion pool, but the
trainer selected one weighted entry once at startup. A single fast frozen
champion decouples from the learner once episode lifespans grow: with 70%
spawn-ahead sampling the gap exceeds the +40 m opponent observation cutoff, so
`opp_presence` and passing reward collapse while training continues in a nominal
1v1 setup. Self-play avoided this because opponent speed tracked the learner.

## Decision

1. Load every `fixed_opponents.entries` checkpoint at startup into an immutable
   pool; do not restore self-play machinery.
2. On every episode reset, sample one pool entry per env from the configured
   weights (independent draws). Re-draw only the envs in the reset mask.
3. Run opponent inference grouped by assigned policy: one deterministic GRU
   forward pass per distinct policy present in the batch, with per-env hidden
   state that resets on episode boundary and stays bound to the env row when the
   assigned policy changes on reset.

## Consequences

Throughput scales with the number of distinct policies active in a step, not
always `num_envs`. Pool entries must share one actor architecture; normalizer
statistics remain per checkpoint. Replay-full reinit keeps the pool weights
unchanged; live opponent assignments may be re-drawn on bootstrap. Artifact
export already stamps `steering_action_mode`; a backfill utility covers legacy
checkpoints missing that field without loosening validation.
