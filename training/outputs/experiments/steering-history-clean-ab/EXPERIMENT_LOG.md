# Steering history clean A/B — preregistration and status

**Date:** 2026-07-25 (takeover session)

## Correction to experiment record

**`steering_history` 5.0 vs 0.5 has never been isolated on the legacy ladder.**
Every fast legacy-ladder run (D2, PC+, r642a001) used `steering_history=5.0`.
The apparent Lee-path pair `spstr002` vs `sphst003` also changed
`steering_delta_max_rad` (0.1047→0.05236) and `delta_max` (0.4→0.33) in the
same step — not a clean A/B.

**Replay capacity was also never tested in isolation.** The 5M/10M arms
simultaneously changed reward stack (Lee path), `steering_history`, `batch_size`,
and self-play. Hidden coupling: `maybe_replay_full_reinit` fires when the buffer
first fills, so replay capacity silently controls when the one-shot network reset
happens (3M→2.998M, 5M→4.997M, 10M→9.994M). Any future replay experiment must
pin that transition across arms.

ADR-0015 changed the shipped mainline default from 5.0 to 0.5 without a clean
legacy-ladder validation. Lee and Vasco publish λ^h = 5.0 at the same ±3°
delta-steering scale — no rescaling justification exists.

## Preregistration (before launch)

Two 25M arms, legacy ladder exactly as `d2fx001` (fixed champion, seed 42),
**only** difference: `reward.reward_scales.steering_history` = 5.0 vs 0.5.

| Arm | Run id | Config | steering_history |
| --- | --- | --- | --- |
| A | `steerhist5-ab001` | `steerhist5-d2match.json` | 5.0 |
| B | `steerhist05-ab001` | `steerhist05-d2match.json` | 0.5 |

Configs differ in exactly one field (verified by construction):

```diff
--- steerhist5-d2match.json
+++ steerhist05-d2match.json
@@ -31,7 +31,7 @@
       "rear_end": 0.5,
       "steering_change": 0.5,
-      "steering_history": 5.0,
+      "steering_history": 0.5,
       "tyre_slip_penalty": 0.0,
```

All other fields match `d2-legacy-fixed.json` / `d2fx001` resolved config.

## Launch status (2026-07-25 ~06:25 UTC)

**Blocked at first attempt:** lark-3 was repurposed by a concurrent session to
`d2fx600` (D2 seed 7, 600M, launched ~06:13 UTC) before `steerhist5-ab001`
could start. Reconstruction overlay files were copied to lark-3; launch queued
for **lark-2 after `pcplus302` completes** (~254M/300M at handoff, est. ~20 min
remaining).

## Results (2026-07-25 ~07:15 UTC)

Both arms completed 25M on lark-2 (legacy ladder, seed 42, fixed champion).

| Milestone | `steerhist5-ab001` (λ=5.0) | `steerhist05-ab001` (λ=0.5) | Δ (5.0 − 0.5) |
| ---: | ---: | ---: | ---: |
| 6M | 2.03 | 4.62 | −2.59 (0.5 transient peak) |
| 18M (peak region) | **4.18** | 3.76 | **+0.43** |
| 24M | 3.86 | 4.01 | −0.15 |

**Verdict:** λ=5.0 wins the 18M peak region (+0.43 m/s), matching Lee/Vasco
(λ^h = 5.0 at ±3° delta-steering scale). λ=0.5 shows a transient 6M spike
(4.62 m/s) then underperforms through 18M; it closes slightly by 24M (+0.15
m/s for 0.5) but remains below λ=5.0's peak. Per preregistration, **ADR-0015
should be superseded** and the mainline default reverted to
`reward_scales.steering_history = 5.0`.

Harvest: both arms verified locally under `training/outputs/runs/`.
