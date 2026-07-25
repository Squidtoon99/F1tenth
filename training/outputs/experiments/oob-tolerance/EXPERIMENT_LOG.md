# OOB tolerance — halving continuous boundary cost

## Hypothesis

Champion `642a7a80` drives faster with ~9× higher `oob_frac` (0.220 vs 0.025
for `pcplus300` @150M) despite identical per-event OOB magnitudes. The champion
learned an aggressive recover-from-off-track policy; reconstruction runs learned
conservative boundary avoidance. Lower continuous OOB cost during training may
shift the policy toward the champion's tolerance profile.

## Preregistration (before launch)

Single arm vs `pcplus300` baseline (`oob_penalty=0.01`):

| Arm | Run id | Config | `oob_penalty` scale |
| --- | --- | --- | --- |
| A | `oobhalf300` | `oobhalf300.json` | **0.005** (half baseline) |

All other fields match `pc-plus.json` (legacy ladder, self-play, seed 42, 300M).
Requires causal-2x2 reconstruction overlay on the target host.

**Success:** @150M `oob_frac` ≥ 0.10 AND 200M+ mean speed > 4.5 m/s.
**Refutation:** 200M+ mean ≤ 4.0 m/s OR `oob_frac` @150M ≤ 0.05 (no behavioral
shift vs `pcplus300`).

## Status

Launched on lark-2 @ 09:15 UTC after `replay10m-clean001` completed (100M,
20 checkpoints harvested+verified; run.log lost to watchdog restart but
`policy_100000768.pt` confirmed). Self-play telemetry active
(`selfplay: pool_size=1`).

**Replay capacity interim:** `replay3m-clean001` 100M mean ~2.7 m/s;
`replay10m-clean001` completed to 100M (trajectory log lost); neither arm
approaches pcplus300 — replay capacity not the bottleneck.
