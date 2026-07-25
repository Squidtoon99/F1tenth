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

## Interim results @ 202M (`oobhalf300`, seed 42)

Halving `oob_penalty` (0.01 → 0.005) produces a **clear behavioral shift**
toward champion-like boundary tolerance:

| Metric | `oobhalf300` @150M | `pcplus300` @150M | champion @150M |
| --- | ---: | ---: | ---: |
| speed (10M bucket mean) | 4.58 | 4.36 | 5.10 |
| oob_frac | **0.155** | 0.034 | 0.177 |

Early speed exceeds all reconstruction arms (peak 6.06 m/s @90M, 5.67 @100M).
Mid-training dip @170-196M (3.3-3.9 m/s) mirrors champion/pcplus oscillation
but recovers: **4.61 m/s @200M** vs pcplus300 4.59 and champion 5.40.

**Replication:** `oobhalf301` (seed 7) launched on lark-1 @ 10:45 UTC.

**Preregistration status (partial):** behavioral shift confirmed (oob_frac
≥0.10 @150M). 200M+ mean still pending (currently recovering through dip).

## Final result (`oobhalf300`, 300M, seed 42)

Completed 300,000,256 transitions. Harvest verified (59 checkpoints, sha256
match; `run.log` restored from wandb after accidental restart overwrote it).

| Milestone | `oobhalf300` | `pcplus300` | champion |
| --- | ---: | ---: | ---: |
| 150M speed | 4.58 | 4.36 | 5.10 |
| 150M oob_frac | **0.155** | 0.034 | 0.177 |
| 200M+ mean speed | **4.820** | 4.263 | 5.205 |
| 250M speed | 4.92 | 4.18 | 5.55 |

**Verdict: CONFIRMED.** Halving `oob_penalty` (0.01 → 0.005) produces both the
behavioral shift (oob_frac @150M 0.155, champion-like) and a measured +0.56
m/s improvement in 200M+ mean speed vs `pcplus300`. Still 0.4 m/s below
champion at 200M+ but trending up (4.92 @250M). Does not yet justify changing
the canonical default without a D2 fixed-opponent replication.

**Follow-up:** `oobhalf-d2-300` (D2 seed 42, halved oob, 300M) preregistered on
lark-2 after harvest. `oobhalf301` (seed 7 replication) running on lark-1.

## Final result (`oobhalf301`, 300M, seed 7)

Completed 300,000,256 transitions. Harvest verified (59 checkpoints).

| Milestone | `oobhalf301` | `oobhalf300` (s42) | champion |
| --- | ---: | ---: | ---: |
| 150M speed | 4.10 | 4.58 | 5.10 |
| 150M oob_frac | 0.077 | **0.155** | 0.177 |
| 200M+ mean speed | **3.496** | **4.820** | 5.205 |

**Verdict: seed-sensitive.** Seed 7 does not replicate seed 42's behavioral
shift (`oob_frac` @150M 0.077 vs 0.155) or speed gain (+0.56 m/s at 200M+).
Halving `oob_penalty` helps on seed 42 but is not a reliable lever across seeds.

## `oobhalf-d2-600` preregistration (2026-07-25 ~07:18 PDT)

Combines the two strongest interventions: D2 fixed champion (seed 7, matching
`d2fx600`) + halved `oob_penalty` (0.005) + 600M horizon.

| Field | Value |
| --- | --- |
| Run id | `oobhalf-d2-600` |
| Host | lark-1 |
| Config | `oobhalf-d2-600.json` |
| Seed | 7 |

**Success:** 200M+ mean ≥ 5.3 m/s AND `oob_frac` @150M ≥ 0.08 (champion-like
tolerance on the D2 cell that reached 5.17-5.25 @600M without oob halving).
**Refutation:** 200M+ mean ≤ `d2fx600` comparable (~4.5-5.0) OR no oob_frac
shift vs `d2fx600` @150M.

Early telemetry @~0M: `oob_frac=0.212`, champion-like from step zero.
