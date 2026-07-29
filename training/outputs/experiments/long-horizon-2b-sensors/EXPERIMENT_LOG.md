# Long-horizon 2B sensor policy — convergence hold test

## Question

Does the best asymmetric (sensor) config **maintain convergence and keep
optimizing past 600M**, rather than collapsing like earlier asymmetric runs
(~250M OOB-dominated collapse)? The symmetric privileged racer trained cleanly
to 4.26B with monotone slow improvement (7.17→7.98 m/s). We want the same
*shape* for the sensor stack: sustained growth without late decay — not a
single mean that a fading run could satisfy.

**Benchmark re-baseline (2026-07-26):** judge against `pcplus600`'s **growth
curve**, not champion `642a7a80`. Measured from `run.log`, 50M bucket means:

| Transitions | champion | `pcplus600` |
| ---: | ---: | ---: |
| 200M | 5.35 | 4.27 |
| 300M | **5.38** (peak) | 4.44 |
| 350M | 5.32 | 4.80 |
| 400M | 4.82 | 5.02 |
| 450M | 4.76 | 4.84 |
| 500M | — | 5.20 |
| 550M | — | 5.28 |

Post-200M split: champion 5.32→4.97 (**decay**); `pcplus600` 4.44→5.08
(**growth**, still climbing when stopped @600M). The champion's "5.208 mean
over 200M–458M" masked peak-then-fade — a run could match that average while
trending down. `pcplus600` is the correct reference because it exhibits the
property we care about: sustained improvement without late collapse.

## Prerequisite runs (do not disturb)

| Run | Purpose | Horizon | Status |
| --- | --- | ---: | --- |
| `numenv4096-600m-a001` | 4096 envs @ 3M replay to 600M | 600M | in flight |
| `numenv8192-deep4m-a001` | 8192 envs @ 4M replay to 150M | 150M | queued after Task 1 |

Config selection is **data-driven**: prefer whichever of 4096 or 8192-deep wins
on completed Task 1 / Task 2 metrics (see `wait-and-launch-2b.sh`).

## Candidate configs (preregistered)

Both share: `mainline-ladder600` reward/env stack, delta steering + both
penalties, `boundary_mode=recoverable_full_car_out`, fixed champion
(`642a7a80` @ 200M), seed 42, `torch.compile reduce-overhead`, W&B online.

| Arm | Run id | `num_envs` | Replay | `replay_full_reinit` |
| --- | --- | ---: | ---: | --- |
| A (default) | `numenv4096-2b-a001` | 4096 | 3M | true (default) |
| B (if 8192-deep wins @150M) | `numenv8192-deep4m-2b-a001` | 8192 | 4M | false |

**Horizon:** 2,000,000,000 transitions (single uninterrupted run).

**Checkpoint cadence:** `export_interval_transitions=10_240_000` (~195 ckpts ×
~17 MB ≈ 3.3 GB). Disk headroom 747 GiB free — conservative vs default 5.12M
(~390 ckpts / ~7 GB) to preserve trajectory resolution without risk.

**Throughput estimate:** ~35k transitions/s → ~16 h wall-clock.

## Success criteria (re-baselined 2026-07-26)

Judge the **trajectory shape** against `pcplus600`, not a single terminal mean
and not the champion's plateau-then-decay. All thresholds are measured from
`run.log` using 50M bucket means (`extract_trajectory.py --bucket-m 50`) unless
noted.

### Reference curve — `pcplus600` (50M bucket means)

| Transitions | mean speed (m/s) | post-200M Δ |
| ---: | ---: | ---: |
| 200M | 4.273 | — |
| 250M | 4.254 | −0.019 |
| 300M | 4.439 | +0.185 |
| 350M | 4.799 | +0.360 |
| 400M | 5.017 | +0.218 |
| 450M | 4.839 | −0.178 |
| 500M | 5.203 | +0.364 |
| 550M | 5.281 | +0.078 |
| 598M (terminal tick) | 5.385 | still climbing |

Net post-200M growth through 550M: **+1.008 m/s**. Terminal 40M mean: 5.292.

Champion is retained only as a **negative control** for the late-decay failure
mode (peak ~5.38 @300M → 4.76 @450M).

### Spawn-gap caveat (approximate comparison)

The 2B run inherits tightened spawn gaps **3–58 m** (~70% opponent presence;
[ADR 0017](../../../docs/adr/0017-tightened-opponent-spawn-gaps.md)). `pcplus600`,
`numenv4096-600m-a001`, and the champion used **3–80 m** (~40% early, collapsing
to ~19% late — observed in Task 1 @522M). Comparisons to `pcplus600` through
600M are therefore **approximate, not matched**.

**How we account for this when judging:**

1. **Primary:** evaluate trajectory *shape* (growth vs decay, tripwire below),
   not pointwise speed parity. A 2B run that grows post-200M while holding
   `opp_presence ≥ 0.50` over every 50M window past 300M passes the convergence
   test even if absolute speed trails `pcplus600` by up to **0.15 m/s** at a
   matched milestone.
2. **Secondary:** if absolute speed *exceeds* `pcplus600` at a matched milestone
   despite higher opponent pressure, treat that as strong confirmation (harder
   training distribution).
3. **Do not** compare 2B opponent-interaction statistics (presence, passing
   rate, collision mix) directly to `pcplus600` without noting the gap-range
   change.

### Milestone parity (through 600M)

At each 50M boundary from 200M to 600M, bucket mean must satisfy:

`speed[M] ≥ pcplus600[M] − 0.15`

| Transitions | `pcplus600` ref | minimum @ milestone |
| ---: | ---: | ---: |
| 200M | 4.273 | 4.12 |
| 300M | 4.439 | 4.29 |
| 400M | 5.017 | 4.87 |
| 500M | 5.203 | 5.05 |
| 600M | ~5.33 (extrap.) | 5.18 |

The 600M reference is extrapolated from `pcplus600`'s decelerating late growth
(+0.078/50M from 500→550M; terminal tick 5.385). Task 1 will provide a
direct 600M measurement on 3–80 m for calibration.

### Post-600M growth requirement (600M → 2B)

For each consecutive 50M window after 600M:

`speed[M] − speed[M−50M] ≥ −0.05 m/s`

Cumulative post-600M growth by 2B should be **≥ +0.15 m/s** if `pcplus600`'s
decelerating extrapolation holds (~5.43 m/s @1B, ~5.53 m/s @2B from horizon
alone). Falling below +0.05 m/s net over 600M–2B while not tripping the decay
alarm is a **plateau**, not a pass on the convergence-hold question.

### Late-decay tripwire

Detect the champion failure mode (peak-then-sustained-fade). Compute running
peak `P[M] = max(speed[0], …, speed[M])` over 50M bucket means.

**Tripwire fires** when, for **two consecutive** 50M buckets:

`speed[M] < P[M−50M] − 0.25 m/s`

Champion would have tripped between 350M and 450M (peak 5.381 @300M, 4.760
@450M = −0.621 over 150M). `pcplus600` never tripped (monotone net growth
post-300M despite a single −0.178 dip @450M).

On tripwire: log as **convergence-hold failure** regardless of absolute speed.
Do not abort training unless pathology abort conditions also fire — the decay
signature is itself a result.

### Early-shape expectations (do not abort)

| Phase | Expected speed (m/s) | Notes |
| ---: | ---: | --- |
| 18–20M | ~5.4 peak | early peak (champion shape) |
| 26–56M | 3.1–3.6 dip | **never abort** |
| 60–120M | 4.0–4.4 | climbing |
| 150M | ~4.5–5.0 | `pcplus600` 4.45; 4096 arm 4.73 |

### Stretch target

6.0 m/s sustained: 50M bucket mean **≥ 6.0** for **two consecutive** buckets.
No asymmetric run has reached this. See **6 m/s plan** below for extrapolation
and ranked levers.

### Episode health

- Lifespan ≥ 200 s (`pcplus600` ~318 s @600M)
- Terminations time-out-dominated; OOB share < 20% of episodes in any 50M window
- `not_moving` termination share < 5% and not monotonically climbing
- Post-300M: `opp_presence` ≥ 0.50 mean per 50M window (validates 3–58 m fix)

## Abort conditions (pathology only)

- Monotonically climbing `not_moving` fraction (precedent: 0.2%→22%)
- Nonfinite states/rewards
- Collapsing lifespan with OOB terminations dominating while speed sinks below
  5.0 band for tens of millions of transitions

**On abort:** capture transition count, termination mix, per-term reward means,
surrounding checkpoints before stopping — collapse signature is a valuable result.

## Reference milestones (50M bucket means)

| Transitions | champion | `pcplus600` | `numenv4096-600m-a001` |
| ---: | ---: | ---: | ---: |
| 100M | 4.35 | 4.06 | **4.41** |
| 150M | 5.22 | 4.43 | **4.82** |
| 200M | 5.35 | 4.27 | **4.81** |
| 300M | 5.38 | 4.44 | **4.66** |
| 400M | 4.82 | 5.02 | 4.57 |
| 500M | — | 5.20 | 4.92 |
| 550M | — | 5.28 | (in flight) |
| 600M | — | ~5.33 | **pending** (~87% @23:22 PDT) |

Task 1 led `pcplus600` at every matched milestone through 200M (+0.54 m/s @200M)
but fell behind 350M–500M as `pcplus600` accelerated (+0.85 m/s over 350M→550M).
Late Task 1 recovery: 450M→500M +0.056 m/s; `opp_presence` collapsed to ~19% (same
pathology ADR 0017 addresses). Terminal 600M number expected ~23:55 PDT.

## Launch log

| Run | Started (PDT) | PID | GPU @45s | Status |
| --- | --- | --- | --- | --- |
| `ladder2b-a001` | 11:10 | 1224910 | — | **replaced** @81.8M (642a7a80 champion, broken scripted arm) |
| `ladder2b-a002` | 12:08 | 1258488 | 75%, 10.8 GiB | **live** (pcplus600 champion + fixed scripted) |
| `numenv4096-600m-a001` | 18:43 | 919781 | 86%, 11.6 GiB | **in flight** (~523M/600M @23:22 PDT) |
| `numenv8192-deep4m-a001` | (after Task 1) | — | — | queued via `wait-task1-then-task2.sh` (921920) |
| 2B long run | (after Task 2) | — | — | `wait-and-launch-2b.sh` (928082) polling |

### Task 1 trajectory (in progress, 2026-07-25 ~23:22 PDT)

| Transitions | `numenv4096-600m-a001` | `pcplus600` | Δ |
| ---: | ---: | ---: | ---: |
| 100M | 4.41 | 4.06 | +0.35 |
| 150M | 4.82 | 4.43 | +0.39 |
| 200M | 4.81 | 4.27 | +0.54 |
| 300M | 4.66 | 4.44 | +0.22 |
| 400M | 4.57 | 5.02 | −0.45 |
| 500M | 4.92 | 5.20 | −0.28 |
| 600M | **pending** | ~5.33 | — |

Task 1 ETA ~23:55 PDT. Task 2 ETA ~01:00 PDT after Task 1.
2B launch ETA ~01:30 PDT + ~16 h → ~17:30 PDT Jul 26.

## Config selection

(Filled automatically by `wait-and-launch-2b.sh` after Task 2 completes.)

## Spawn-gap change (2026-07-26)

Before the 2B launch, default opponent spawn gaps were tightened **3–80 m → 3–58
m** ([ADR 0017](../../../docs/adr/0017-tightened-opponent-spawn-gaps.md)).
Replay does not synthetically zero opponent observations at sample time — only
50/50 stratified sampling of already-visible vs already-absent stored windows —
so environment presence is the sole source of solo experience. Probe at ego
5.5 m/s / opponent 0.5 m/s: **~70%** step-mean presence (vs ~40% at 3–80 m),
preserving real solo running while beating the late-run ~19% collapse. Both
preregistered 2B arms inherit the new default at launch. `numenv8192-deep4m-a001`
(Task 2 diagnostic) remains pinned at 3–80 m. In-flight `numenv4096-600m-a001`
is unaffected (config loaded at startup).

## 6 m/s plan (ranked, evidence-based)

**Short answer:** horizon alone is **unlikely** to reach 6.0 m/s; **5.4–5.7 m/s
@2B** is the honest extrapolation from `pcplus600`'s decelerating growth curve.
Closing the last ~0.6 m/s probably requires a mechanism we have not validated —
most likely observation modality (~2 m/s gap vs symmetric privileged at matched
horizons) — unless the 2B run re-accelerates past the `pcplus600` late slope.

### `pcplus600` extrapolation (honest)

Post-350M growth per 50M window: +0.218, −0.178, +0.364, +0.078 — decelerating,
not flat. Fitting decelerating increments (rate × 0.65 per window, floor
+0.005/50M):

| Horizon | projected speed | 95% intuition band |
| ---: | ---: | --- |
| 600M | 5.33 m/s | 5.28–5.39 (terminal tick 5.385 observed) |
| 1B | 5.43 m/s | 5.35–5.55 |
| 2B | 5.53 m/s | 5.40–5.70 |

Linear extrapolation from the last window (+0.078/50M) reaches 6.0 m/s @1B —
**overoptimistic** given three prior windows averaged +0.12/50M with high
variance. Constant-rate extrapolation (+0.12/50M) reaches 6.37 @1B — also
overoptimistic. **6.0 m/s by 2B requires growth to re-accelerate** beyond what
`pcplus600` showed; not impossible (450→500M was +0.364) but not the base case.

Symmetric privileged (`darktoaster_warp_ro_8192`): 7.32 m/s @100M, 7.98 m/s
@4.26B — physics/track permit far more. Residual sensor gap @150M after 4096
env fix: 7.32 − 4.73 = **2.59 m/s**. Action space refuted (`obsab-a001`:
removing delta cap + both steering penalties tracked **below** control at every
checkpoint). 8192 envs on 3M replay **hurt** (2.53 @150M vs 4.73 for 4096).

### Is 6 m/s the right deploy target?

Probably not as a hard requirement. The symmetric racer's 7.98 m/s uses
privileged Frenet features unavailable on hardware. Removing deploy constraints
(delta cap, steering penalties) did not improve sim speed — so the ~0.6 m/s gap
to 6.0 is **not** the price of steering smoothness. The binding deploy cost is
more likely **LiDAR perception** (latency, noise, partial observability) — the
~2 m/s gap to symmetric — which sim training may not fully model. A defensible
deploy target is **5.0–5.5 m/s** sustained with healthy lifespan; 6.0 m/s is a
sim stretch goal useful for headroom, not a ship gate.

### Ranked levers

GPU-hour cost assumes RTX 4080 Super @ ~35k transitions/s
(1M transitions ≈ 0.008 h).

| Rank | Mechanism | Experiment | Cost (GPU-h) | Confirm | Refute |
| ---: | --- | --- | ---: | --- | --- |
| 1 | Sample efficiency / slow convergence | **2B run** (`numenv4096-2b-a001`, 3–58 m gaps) | ~16 | ≥5.5 m/s @1B with growth, no tripwire | Plateau <5.4 or tripwire fires |
| 2 | Opponent-presence collapse | **3–58 m spawn gaps** (in 2B config) | 0 (config) | `opp_presence ≥ 0.50` post-300M + growth shape | Presence still collapses to ~19% |
| 3 | Parallel env diversity | **4096 envs** (promoted from sweep) | 0 (in 2B) | Already confirmed +1.98 m/s @150M vs 1024 | — |
| 4 | Observation modality | **Symmetric-trainer revival** + sensor vs privileged ablation (`obs-modality-ablation` Arm B) | ~8–16 dev + ~2.4 per 300M run | Privileged sensor-matched stack closes ≥1 m/s of the 2.6 m/s gap | Gap persists at 300M → perception is ceiling |
| 5 | Replay depth vs diversity | **`numenv8192-deep4m-a001`** (8192, 4M buffer) | ~1.2 @150M | Beats 4096 @150M by >0.15 m/s | ≤4096 (8192@3M already refuted) |
| 6 | Opponent training dynamics | **`mainline-selfplay300`** (self-play vs fixed champion) | ~2.4 @300M | ≥4.8 m/s @200M+ sustained (`pcplus600` beat D2 by +0.245 @600M) | Tracks fixed champion within ±0.2 |
| 7 | Replay capacity timing | **`replay10m-clean001`** vs 3M | ~0.8 @100M (+ extension if promising) | Material 200M+ divergence | Overlap within ±0.3 m/s @50M+ |
| 8 | OOB tolerance / aggression | **`oobhalf-d2-600`** (seed-42 self-play benefit did not replicate on D2) | ~4.8 @600M | +0.3 m/s vs matched D2 arm | No gain (likely refuted) |

**Config-sweep only:** ranks 2, 3, 5–7. **Branch-level code work:** rank 4
(symmetric trainer path deliberately reverted on this branch).

### Highest-value next experiment after 2B completes

**If 2B shows growth without tripwire but plateaus <5.7 m/s:** observation-modality
ablation (rank 4) — the only lever sized to the remaining ~2 m/s symmetric gap
that horizon and env count cannot explain.

**If 2B trips the decay alarm:** forensics on termination mix and opponent
presence before any new lever — likely a training-distribution bug, not a capacity
ceiling.

**If 2B exceeds 5.7 m/s with continued growth:** extend horizon or begin deploy
eval; 6.0 m/s may be reachable by 2B without new mechanisms.

## Relaunch: scripted follower + pcplus600 champion (2026-07-26)

`ladder2b-a001` stopped at **81,766,400** transitions (~4% of 2B). Opponent
distribution cannot change mid-run.

**Scripted centerline follower** ([ADR 0018](../../../docs/adr/0018-scripted-opponent-delta-steering.md)):
the delta inner loop used absolute-tuned P gains; ~98% steer saturation caused
weave and near-zero signed track speed despite high path progress. Fixed with
speed-scaled cross-track `atan(k·ey/(|v|+0.5))` plus scaled delta inner loop.
Measured @1024-env scripted-only (seed 0, 1250-step measure window):

| Metric | Before (target 2.5 m/s) | After (target 2.5 m/s) | After (target 4.0 m/s) |
| --- | ---: | ---: | ---: |
| Steer sat (\|cmd\|>0.99) | 98.8% | 20.7% | 38.8% |
| Body speed (m/s) | 0.88 | 3.06 | 3.50 |
| Mean \|steer cmd\| | 0.995 | 0.418 | 0.637 |
| Min boundary dist (m) | — | 0.42 | 0.22 |

**Fixed champion:** `642a7a80` @199.68M → `pcplus600` @600M. Raw
`policy_600000512.pt` fails strict validation (`steering_action_mode` missing,
defaults to `absolute`; run config confirms delta training). Metadata-complete
copy at `champions/pcplus600-policy_600000512.pt` passes
`validate_sensor_policy_artifact`. Isolated opponent rollout (fresh GRU, ego
idle): pcplus600 **2.0 m/s** body speed vs 642a7a80 **1.6 m/s**; pcplus600
**1672 m** forward progress vs **182 m**.

**`ladder2b-a002`:** same `mainline-ladder2b.json` stack (1024 envs, 3M replay,
3–58 m gaps, seed 42, legacy ladder boundary/reward coefficients).

---

## Post-mortem: `ladder2b-a002` vs `pcplus600` (2026-07-26)

**Status:** `ladder2b-a002` stopped at **355,481,600** transitions (host reboot,
not code fault). Analysis only — no new training launched.

### Lead finding

**Self-play vs fixed champion is the dominant confound (~50–70% of the speed
gap, high confidence).** Every post-cleanup long-horizon arm that replaced
`pcplus600`'s `--self-play` with a fixed champion (`mainline-ladder600-fix`,
`ladder2b-a002`) tracks persistently behind at matched milestones. The opponent-
presence collapse in `ladder2b-a002` is a **symptom** of fixed-champion dynamics,
not evidence that ADR 0017's 3–58 m gaps failed.

### Config diff (resolved `config.json`, not hand-wavy)

| Knob | `pcplus600` | `ladder2b-a002` | Material? |
| --- | --- | --- | --- |
| Opponent mode | `--self-play`, `fixed_opponents: false` | `--fixed-opponents`, pcplus600 @600M ckpt | **Yes — primary** |
| `opponent_strategy` | `policy` (100% policy rows) | `mixed` (10% scripted / 90% champion) | **Yes — secondary** |
| Spawn gaps | 3–80 m | 3–58 m | Minor at reset; not collapse driver |
| Scripted arm | N/A (`policy`-only) | ADR 0018 fixed follower (~3.5 m/s) | Yes — distribution shift vs benchmark |
| Boundary/reward | legacy stack (`reward_stack=legacy`, coeff 20.0/m legacy path) | `boundary_mode=recoverable_full_car_out` + 0.1296/m | **No** — parity test pins equivalence |
| `reset_stationary_probability` | 0.3 (patch) | 0.3 (patch) | Match |
| `num_envs` / replay / LR / γ / batch | 1024 / 3M / defaults | same | Match |
| `export_interval_transitions` | 5.12M | 10.24M | No (logging only) |
| `total_transitions` | 600M | 2B | No (horizon only) |

Additional differences surfaced only in `args` (CLI reward scales on `pcplus600`
that ladder2b inherits from patch defaults): none material beyond the table.

Boundary equivalence: `training/tests/test_boundary_ladder_equivalence.py` asserts
`wall_contact_coefficient=0.1296`, `boundary_contact_coefficient=4.0`, and the
`oob_penalty × 12.96` mapping against the legacy `pc-plus.json` patch. It also
exercises footprint-margin graze semantics. **Covers the coefficients in use; does
not exhaust every legacy code path.**

### Milestone comparison (from `run.log`, nearest 51200-tick bucket)

| Transitions | `pcplus600` speed / lifespan / presence | `ladder2b-a002` | Δ speed |
| ---: | --- | --- | ---: |
| 150M | 4.45 m/s / 194 s / 0.154 | 3.63 / 88 s / 0.097 | −0.82 |
| 200M | 4.60 / 99 s / 0.152 | 4.04 / 276 s / 0.115 | −0.56 |
| 250M | 4.07 / 297 s / 0.161 | 3.97 / 210 s / 0.146 | −0.10 |
| 350M | 4.70 / 121 s / **0.486** | 4.32 / 152 s / **0.122** | −0.38 |
| 355M | 4.51 / 121 s / 0.305 | 4.36 / 212 s / **0.083** | −0.15 |

Reference: `mainline-ladder600-fix` (fixed `642a7a80` @200M, 3–80 m gaps) was
**2.75 m/s @150M** and **3.94 m/s @250M** — same fixed-champion family, same
deficit pattern vs `pcplus600`.

### Opponent-presence anomaly — root cause

**Definition (code):** `metric/opponent_presence` = step-mean fraction of envs
where critic obs block `[384:392)` has any non-zero element
(`standalone_trainer.py`). The Warp kernel zeroes opponent obs when centerline
gap ∉ `[−20 m, +40 m]` (`kernel.py` `write_raw_observation`).

**What happened in `ladder2b-a002`:**

| Phase | Transitions | presence | lifespan | learner / opp speed |
| --- | ---: | ---: | ---: | --- |
| Early | 0–100M | **0.48–0.67** | 1–3 s | 2.5–5.8 / 4.1–4.4 |
| Collapse | ~122M | **0.19** | 33 s | 3.3 / 4.5 |
| Late | 150–355M | **0.08–0.15** | 70–210 s | 3.6–4.4 / 4.5–4.6 |

**Evidence:** presence was *higher* than `pcplus600` early (0.6 vs 0.39 @51200)
despite tighter 3–58 m gaps — ADR 0017 did not cause low presence. The collapse
coincides with lifespan jumping from ~2 s to 30–200 s (~122M): fewer resets →
less respawn re-coupling.

**Mechanism (inference, consistent with logs):** fixed champion runs ~0.3–0.5 m/s
faster than the learner. With 70% spawn-ahead probability, the gap widens beyond
the +40 m obs gate within one long episode. Passing reward → ~0 (confirmed:
`passing ≈ −0.0007` late). This is **not** "learner outran the fleet" — the
learner is slower than the champion.

**`pcplus600` presence at same milestones:** also low early (0.13–0.16 @150–250M)
yet speed was higher. Divergence opens **after ~300M** when self-play opponent
speed tracks learner (~4–5 m/s) and presence recovers to 0.4–0.5 while speed
climbs toward 5.3 m/s. Self-play keeps relative speed ≈ 0 once coupled; fixed
champion does not.

**ADR 0017 verdict:** revert-friendly but **not the primary fix**. Tightening
3–58 m did not prevent collapse under fixed champion; it may even help at reset.
Pin 3–80 m only for strict `pcplus600` reproduction.

### Ranked causes of the deficit

| Rank | Cause | Est. share of gap | Evidence vs inference |
| ---: | --- | ---: | --- |
| 1 | Self-play → fixed champion | 50–70% | **Evidence:** config diff; `mainline-ladder600-fix` ~1 m/s down @200M; causal-2x2 opponent-mode arm; late speed divergence tracks co-evolving vs static opponent |
| 2 | Presence collapse (fixed champion symptom) | 15–25% | **Evidence:** presence 8–15% late, passing ≈ 0; **inference:** hurts multi-agent reward/obs training |
| 3 | Scripted-arm distribution (ADR 0018 fix vs broken follower) | 5–10% | **Evidence:** `pcplus600` was `policy`-only; ladder2b 10% fixed scripted; **inference:** benchmark trained against broken 0.88 m/s scripted |
| 4 | Boundary vocabulary (legacy vs `boundary_mode`) | 0–5% | **Evidence:** parity test green |
| 5 | Spawn gaps 3–58 vs 3–80 | 0–5% | **Evidence:** ladder2b early presence *exceeded* pcplus600 |

### Self-play restoration scope

Removed in commit `52487e3` (`racing: add sensor-policy training and deployment
path`). Recoverable from parent `b537c9d` (`training: restore wall and self-play
integration`):

| Component | ~lines | Notes |
| --- | ---: | --- |
| `SelfPlaySnapshot` + `SelfPlayManager` | ~250 | Pool, snapshot/refresh intervals, mixed sampling |
| Trainer loop hooks | ~150 | `maybe_snapshot`, `maybe_refresh`, win-rate logging, bootstrap |
| CLI (`--self-play`, `--selfplay-*`) | ~80 | All removed from current `standalone_trainer.py` |
| `test_selfplay_manager.py` | ~308 | Deleted; exists at `b537c9d` |

**Total estimate:** ~400–600 LOC re-integration against the current asymmetric
sensor trainer (2114 lines today vs 2835 at `b537c9d`). Not a straight revert —
`FixedChampionManager` must coexist or be feature-flagged. **Faithful recovery:**
snapshot/refresh intervals and mixed sampling from `pcplus600` config are
documented; subtle integration risk around dual-view obs refresh and compile path.

**Prerequisite:** restore self-play before any long-horizon attempt intended to
beat `pcplus600`.

### Recommended next run

**Phase 1 — reproduction gate (~5 GPU-h @600M):** confirm the current codebase
can hit the `pcplus600` curve before changing anything. We have **never**
reproduced it post architecture cleanup.

Preregistered patch: `pcplus600-repro.json` (this directory). Launch **after**
self-play restoration:

```bash
cd /home/ubuntu/projects/F1tenth/training
../.venv/bin/python standalone_trainer.py \
  --config outputs/experiments/long-horizon-2b-sensors/pcplus600-repro.json \
  --num-envs 1024 \
  --total-transitions 600000000 \
  --opponent policy \
  --self-play \
  --selfplay-sample mixed \
  --selfplay-mixed-latest-prob 0.25 \
  --device cuda \
  --seed 42 \
  --compile \
  --compile-mode reduce-overhead \
  --run-id pcplus600-repro-a001 \
  --wandb --wandb-mode online
```

**Pass @600M:** 50M-bucket mean speed ≥ 5.05 m/s (pcplus600 − 0.15), lifespan
≥ 250 s, no late-decay tripwire, post-300M presence mean ≥ 0.25 (pcplus600
achieved 0.4+).

**Phase 2 — improvement run (only after Phase 1 passes):** same self-play stack;
then consider incremental changes one at a time:

1. Keep self-play; bump `num_envs` to 4096 (Task 1 showed +0.54 m/s @200M vs
   1024 self-play baseline — separate evidence).
2. Do **not** carry fixed champion forward.
3. Do **not** re-tighten spawn gaps until coupled under self-play.
4. Leave scripted arm off (`opponent_strategy: policy`) until reproduction passes;
   reintroduce mixed opponents as a labeled ablation.

**Exact-repro uncomfortable option (broken scripted + 3–80 + self-play):** only
needed if Phase 1 fails with ADR 0018 scripted fix in the codebase. The broken
follower code is recoverable from pre-ADR-0018 commits but contradicts current
mainline; prefer policy-only reproduction first. **Worth ~5 GPU-h** because we
have zero post-cleanup reproduction proofs.

### Risk register (deviations from `pcplus600`)

| Change | Risk if applied before repro |
| --- | --- |
| Restore self-play | Low — this *is* the benchmark |
| 4096 envs | Medium — faster but different batch stats; validate after repro |
| 3–58 m gaps | Low–medium — may reduce early solo fraction; not urgent |
| Fixed champion | **High — refuted** for long horizon |
| ADR 0018 scripted fix | Medium — changes 10% arm vs benchmark's broken scripted |
| `boundary_mode` path | Low — tested equivalent |

### Open ambiguities (not picking a side)

- Whether low presence @150–250M is harmless in self-play (pcplus600 had it too)
  or a necessary phase — speed still diverged favorably for self-play after 300M.
- Exact share of gap from scripted-arm vs champion-only fixed mode — both differ
  from benchmark; not isolated in a clean A/B on this branch.
- `numenv4096-600m-a001` (4096 envs, self-play?, 3–80 m) final 600M number may
  calibrate how much env count alone explains vs opponent mode.

## pool600-a001 — per-episode opponent pool (2026-07-27)

**Hypothesis:** a weighted pool spanning measured solo speeds fixes the single
frozen-champion decoupling that collapsed `opp_presence` on `ladder2b-a002`
(~0.08–0.15 after 122M) without restoring self-play ([ADR
0019](../../../docs/adr/0019-per-episode-opponent-pool.md)).

**Measured solo speeds** (deterministic GRU, Austin, DR on, mainline-ladder2b env
stack; backfilled `steering_action_mode=delta` on legacy checkpoints):

| Checkpoint | transitions | solo m/s |
| ---: | ---: | ---: |
| pool-policy_10240000 | 10M | 2.23 |
| pool-policy_51200000 | 51M | 4.07 |
| pool-policy_256000000 | 256M | 4.27 |
| pool-policy_409600000 | 409M | 5.02 |
| pool-policy_512000000 | 512M | 5.20 |
| pool-policy_600000512 | 600M | 5.34 |

Pool weights (total 20): 1 / 2 / 2 / 4 / 5 / 6 → expected sample fractions
~5% / 10% / 10% / 20% / 25% / 30%.

**Short validation (`pool-validate-a001`, 5.24M transitions, 512 envs):**
`opp_presence` held **0.32–0.39** (final tick 0.324); `passing` term mean
−0.13…−0.25 (non-zero throughout); pool assignment fractions tracked configured
weights within ~2 pp. Throughput @1024 envs: **~21.3k transitions/s** vs
**~28.7k** single champion (−26%, six policies loaded).

**600M gate run:** `pool600-a001` — config
`pool600m-a001.json` (else matched to `mainline-ladder2b.json`), systemd user
unit `f1tenth-pool600-a001.service` (enabled, reboot-durable). Success criteria
unchanged vs `pcplus600` milestones; additionally require `opp_presence ≥ 0.25`
past 300M.

### 2026-07-27 — champion overlay contamination recovery

Host reboot at 20:58 auto-started disabled `f1tenth-champion-recovery300.service`,
which applied legacy reproduction overlays onto tracked `kernel.py` / `rewards.py`
(and left `warp_env.py` / `config.py` / `standalone_trainer.py` aligned to that
stack). A follow-on commit (`8b8e77b`) landed per-episode opponent pool wiring on
top of the contaminated tree, silently reverting ADR 0011/0012 boundary semantics
in `warp_env.py` and legacy `oob` diagnostics in `standalone_trainer.py` /
`config.py`. Restored mainline kernel/rewards from HEAD; separated pool feature
from boundary reversion; suite back to **376 passed / 0 failed**. Added
`training/overlay_launch.sh` (marker file `.overlay-active`, restore on exit) and
a launch-time guard in `standalone_trainer.py` (refuse dirty `training/` tree,
record `source_git` in run snapshot). Relaunched **`pool600-a002`** on clean
mainline ladder stack with the calibrated six-entry pool (same weights as
`pool600m-a001.json`).

## pool600-a002 — 150M gate verdict (2026-07-27)

**Decision point:** 150M transitions (systemd stopped/disabled @150.9M, 23:46 PDT).

**Verdict: FAIL on pace; PASS on opponent presence.** The per-episode opponent pool
fixed the presence-collapse pathology but bought **zero** speed improvement. This
is evidence *against* opponent coupling being the primary lever for pace, and it
weakens the earlier claim that opponent mode explained 50–70% of the fixed-champion
deficit — consistent with Wave-1 causal screens where the reward stack was
dominant and opponent mode only a secondary amplifier.

### Trajectory (10M bucket means, nearest 51200-tick log)

| Transitions | speed (m/s) | lifespan (s) | opp_presence |
| ---: | ---: | ---: | ---: |
| 10M | 3.10 | 1.35 | 0.624 |
| 20M | 4.55 | 1.85 | 0.644 |
| 30M | 3.36 | 2.02 | 0.542 |
| 40M | 2.99 | 2.07 | 0.498 |
| 50M | 2.99 | 2.13 | 0.492 |
| 60M | 3.01 | 2.02 | 0.496 |
| 70M | 2.82 | 2.22 | 0.492 |
| 80M | 2.86 | 2.17 | 0.481 |
| 90M | 2.78 | 2.09 | 0.491 |
| 100M | 2.74 | 2.19 | 0.483 |
| 110M | 2.56 | 2.02 | 0.502 |
| 120M | 2.48 | 2.18 | 0.459 |
| 130M | 2.42 | 2.20 | 0.463 |
| 140M | 2.52 | 2.16 | 0.475 |
| 150M | 2.36 | 2.47 | 0.437 |

**Sustained `opp_presence`:** mean **0.506** over 2885 logged ticks (range
0.278–0.683); no late collapse. Benchmark `pcplus600` at comparable milestones
held ~0.38 mean presence with intermittent recovery post-300M.

### Comparison to `pcplus600` phase change (same 150M gate)

| Metric | `pcplus600` @150M | `pool600-a002` @150M |
| --- | ---: | ---: |
| speed | 4.43 m/s | 2.36 m/s |
| lifespan | 153 s (@120M breakout from ~2 s) | 2.47 s (pinned) |
| opp_presence | ~0.38 | 0.47 sustained |

`pool600-a002` never exited the ~2 s lifespan regime. Speed declined monotonically
from an early 4.55 m/s peak @20M to 2.36 m/s @150M — the opposite of `pcplus600`'s
110M→150M acceleration (4.19→4.43 m/s with lifespan 42→153 s).

### What the pool did and did not buy

- **Did buy:** stable opponent coupling. Per-episode weighted sampling across six
  solo-speed checkpoints (2.23–5.34 m/s) held presence ~0.47–0.50 with non-zero
  passing reward throughout; the single-frozen-champion decoupling that collapsed
  `ladder2b-a002` to 8–15% presence did not recur.
- **Did not buy:** any improvement in pace. Terminal speed 2.36 m/s is *below*
  fixed-champion `ladder2b-a002` at 355M (4.36 m/s) and far below `pcplus600`
  milestones. Opponent-presence fix alone is insufficient for the convergence-hold
  question.

**Next:** warm-start probe (`warmstart-probe-a001`) — init from `pcplus600` 600M
actor (5.34 m/s solo) to test whether the ~100M from-noise phase is the bottleneck
or whether the reward stack destroys a known-good policy ([ADR
0020](../../../docs/adr/0020-warm-start-actor-freeze.md)).

## Warm-start plateau probes (2026-07-27)

All arms: `--init-ckpt` pcplus600 @600M, `replay_full_reinit: false`,
`actor_freeze_transitions: 5_000_000`, `alpha: 0.001`, 1024 envs, batch 1024,
3M replay, seed 42, legacy ladder boundary coefficients (0.1296 / 4.0 / 0.1296),
50M horizon. Metrics below: mean `env: speed` over 35–45M transitions window
(deduped log lines; probe run had 2× duplicate logging — see logging fix below).

| Run | Hypothesis / delta | speed @35–45M | lifespan @40M | Verdict |
| --- | --- | ---: | ---: | --- |
| `warmstart-probe-a001` | baseline warm-start | 5.331 m/s | 304 s | plateau (invalid post-26330f3 — see alpha annotation) |
| `warmstart-reward-relax-a001` | `steering_change` 0, `steering_history` 0, wall coeff halved | 5.331 m/s | 304 s | **invalid** — actor never updated |
| `warmstart-matched-opp-a001` | pcplus600 @600M champion, tight gaps, 2× passing | 5.330 m/s | 294 s | **invalid** — actor never updated |

Reward-relax runtime confirmed: `steer_chg=0.0000`, `wall_contact` half baseline.
Three materially different reward/opponent configs produced numerically
indistinguishable pace — consistent with the policy not responding to
training signal under the CUDA-graph actor-freeze bug (see alpha sweep
annotation below); **not** valid evidence against the exploration hypothesis.

**Logging bug (fixed):** `setup_trainer_logging` wrote to both a `FileHandler` and
stdout; systemd units append stdout to the same `run.log`, doubling every line.
Fix: file-only logging when `log_file` is set (`training/standalone_trainer.py`).

## Alpha / exploration probes (2026-07-27)

**Hypothesis:** `alpha: 0.001` (10× below default) plus 5M actor freeze prevents
the warm-started policy from exploring faster actions; critic-only updates for
the first 5M then near-deterministic actor updates hold the init at ~5.33 m/s.

Metrics: mean `env: speed` over 35–45M transitions (deduped log lines); lifespan
from tick line at nearest 40M boundary. Pass threshold: sustained speed **> 5.35
m/s** with lifespan ≥ 200 s and low OOB.

| Run | Delta from baseline | speed @35–45M | lifespan @40M | Verdict |
| --- | --- | ---: | ---: | --- |
| `warmstart-probe-a001` | baseline warm-start | 5.331 m/s | 304 s | plateau |
| `warmstart-reward-relax-a001` | `steering_change` 0, `steering_history` 0, wall coeff halved | 5.331 m/s | 304 s | **invalid** — actor never updated (see below) |
| `warmstart-matched-opp-a001` | pcplus600 @600M champion, tight gaps, 2× passing | 5.330 m/s | 294 s | **invalid** — actor never updated (see below) |
| `warmstart-alpha010-a001` | `alpha: 0.01`, freeze 5M | 5.331 m/s | 314 s | **invalid** — actor never updated (see below) |
| `warmstart-alpha010-freeze1m-a001` | `alpha: 0.01`, freeze 1M | 5.328 m/s @25–35M | — | **invalid** — actor never updated (see below) |
| `warmstart-alpha010-nofreeze-a001` | `alpha: 0.01`, freeze 0 | — | — | **invalid** — actor never updated (see below) |

### 2026-07-27 annotation — CUDA graph + actor-freeze bug (`26330f3`)

**Prior conclusion retracted:** the alpha sweep did **not** refute the exploration
hypothesis. With `actor_freeze_transitions > 0` and `--compile-mode reduce-overhead`,
the CUDA graph was captured while the actor was frozen, so the actor never updated
for the entire run (actor L2 distance from init was exactly 0.0). The identical
5.33 m/s plateaus across all warm-start arms are therefore **uninformative** for
entropy/exploration — the runs never exercised actor gradient steps.

**Affected runs (invalid for exploration/reward/opponent conclusions):**
`warmstart-probe-a001`, `warmstart-reward-relax-a001`, `warmstart-alpha010-a001`,
`warmstart-alpha010-freeze1m-a001`, `warmstart-matched-opp-a001`.

**Status:** exploration hypothesis **untested** (not refuted). Re-test requires
post-`26330f3` trainer with verified actor movement under freeze+compile, or
`--no-compile` / freeze 0.

**Logging bug:** confirmed fixed post-`e379bef`. `warmstart-probe-a001` log is 2×
deduped (27507→13807 lines); post-fix runs (`alpha010-a001`, `freeze1m-a001`) are
single-count (6884 / 4085 lines). Root cause was `FileHandler` + systemd stdout
both writing to `run.log`; fix is file-only logging when `log_file` is set.

## pool600-fast2b-a001 — 2B from-noise fast snapshot pool (2026-07-27)

**Hypothesis:** self-play's benefit was racing a rolling history of the learner's
own snapshots. A **fixed ladder of fast pcplus600 checkpoints** (excluding slow
members that taught pace-matching in `pool600-a002`) approximates that
distribution without self-play machinery.

**Deviation from ADR 0019:** dropped 10M (2.23 m/s) and 51M (4.07 m/s); retained
only checkpoints faster than the learner will be for most of the run. Weights
rebalanced toward the 600M anchor (45% vs ADR 0019's 30%).

### Pool composition (validated artifacts, measured solo speeds)

| Checkpoint | solo m/s | weight | sample fraction |
| ---: | ---: | ---: | ---: |
| pool-policy_600000512 | 5.34 | 5 | 45.5% |
| pool-policy_512000000 | 5.20 | 2 | 18.2% |
| pool-policy_409600000 | 5.02 | 2 | 18.2% |
| pool-policy_256000000 | 4.27 | 2 | 18.2% |

Total weight 11. All four pass `validate_sensor_policy_artifact` (backfilled
`steering_action_mode=delta` copies under `champions/`).

### Run spec

| Knob | Value |
| --- | --- |
| Run id | `pool600-fast2b-a001` |
| Init | **from noise** (no `--init-ckpt`; `replay_full_reinit` default true) |
| Opponent mix | 100% policy pool (`scripted_weight=0`, `policy_weight=1`) |
| Spawn gaps | **3–80 m** (pcplus600 distribution; overrides ADR 0017 default) |
| Boundary/reward | `recoverable_full_car_out` / `continuous_quadratic`, 0.1296 / 4.0 / 0.1296 |
| Env/batch/replay | 1024 / 1024 / 3M |
| `steering_history` | 5.0 |
| Horizon | 2B, export every 10.24M, seed 42 |
| Steering | delta, 10 Hz |

**pcplus600 opponent note:** resolved `config.json` shows `opponent_strategy:
policy` (100% policy rows). The 10% scripted arm in its patch was unused; this
run omits scripted entirely.

### Decision gates (kill / continue)

Judge against `pcplus600` at matched milestones:

| Gate | Criterion | Action |
| ---: | --- | --- |
| **~120M phase change** | pcplus600 broke out of ~2 s lifespan @110M (42 s) and hit 153 s @120M at 4.19 m/s | If still pinned in ~2 s lifespan regime at **150M**, **stop** — repeats `pool600-a002` |
| 150M | ≥ **4.43 m/s** | continue |
| 350M | ≥ **4.80 m/s** | continue |
| 600M | ≥ **5.05 m/s**, lifespan ≥ **250 s** | continue |
| 2B expectation | 5.4–5.7 m/s honest extrapolation; 6.0 m/s requires re-acceleration beyond pcplus600 |

**Kill criterion (plain):** if mean speed is still in the ~2 s lifespan / ~2–3 m/s
regime at 150M transitions, stop the run rather than burn ~16 more GPU-hours.

### Infrastructure

- Systemd: `f1tenth-pool600-fast2b-a001.service` (sole enabled f1tenth unit)
- GPU idle queue: `recovery_2b` target in `tools/gpu_idle_queue.json`
- Prior enabled unit `f1tenth-warmstart-alpha010-nofreeze-a001.service` disabled

### Launch record

| Field | Value |
| --- | --- |
| Started (PDT) | 2026-07-27 17:09 |
| PID | 154963 |
| GPU @60s | 70% util, 10.9 GiB |
| Throughput @1.2M | **~25.0k transitions/s** steady (24.3–25.8k over ticks 800–1350) |
| vs single champion | −13% vs 28.7k (6-policy pool was −26%) |
| 2B wall-clock est. | **~22.2 h** @25.0k/s (range 20–23 h if rate holds) |
| Status | **abandoned** @111.8M (2026-07-27) — fixed fast pool refuted; repeats ~2 s lifespan stall |

Early log (pool load + first tick @307k):

```
Fixed opponent pool loaded size=4 total_weight=11.0000
  pool[0] ... pool-policy_600000512.pt weight=5.0000 transitions=600000512
  pool[1] ... pool-policy_512000000.pt weight=2.0000 transitions=512000000
  pool[2] ... pool-policy_409600000.pt weight=2.0000 transitions=409600000
  pool[3] ... pool-policy_256000000.pt weight=2.0000 transitions=256000000
ticks=300 transitions=307200 ... transitions/s=23790.8 ... opp_presence=0.147
fixed_opponent_pool: assignments=[600M p=0.453, 512M p=0.182, 409M p=0.183, 256M p=0.182]
```

## pcplus2b-a001 — 2B self-play pcplus600 reproduction gate (2026-07-27)

**Hypothesis:** post-cleanup codebase can reproduce `pcplus600`'s self-play growth
curve when trained from noise for 2B transitions. Fixed-opponent variants
(`pool600-a002`, `pool600-fast2b-a001`) are refuted; self-play is the only regime
with evidence of reaching 5.3+ m/s sustained growth.

**Config:** `pcplus2b-a001.json` — mainline ladder boundary/reward
(`recoverable_full_car_out` / `continuous_quadratic`, 0.1296 / 4.0 / 0.1296;
equivalence asserted by `test_boundary_ladder_equivalence.py`), 1024 envs, batch
1024, 3M replay, `replay_full_reinit: true`, `actor_freeze_transitions: 0`,
`alpha: 0.01`, seed 42, spawn gaps **3–80 m**, policy-only opponents
(`scripted_weight: 0`), self-play pool size 10, mixed sampling
(`mixed_latest_prob: 0.25`), `--compile --compile-mode reduce-overhead`.

**Init:** from noise (no `--init-ckpt`).

**Reference yardstick:** measured from `pcplus600/run.log` (nearest 51200-tick lines):

| Transitions | `pcplus600` speed (m/s) | lifespan (s) |
| ---: | ---: | ---: |
| 110M | 3.82 | 4.0 (still ~2 s regime) |
| 120M | 3.91 | 83.6 (breakout begins ~113M, ~42 s bucket) |
| 150M | 4.45 | 193.9 |
| 300M | 4.09 | 76.2 |
| 600M | **5.39** | 294.6 |

### Decision gates (preregistered — kill / continue)

| Gate | `pcplus600` reference | Criterion | Kill | Continue |
| ---: | --- | --- | --- | --- |
| **120M phase change** | lifespan breakout ~113M → ~42 s mean; 120M → 83.6 s | If **still pinned in ~2 s lifespan regime** (mean < 15 s) | **STOP** — repeats fixed-pool failure mode | proceed |
| **150M** | 4.45 m/s / 194 s | speed < **3.5 m/s** OR lifespan < **30 s** | **STOP** — no phase change | proceed |
| **300M** | 4.09 m/s (dip before re-acceleration) | speed < **3.5 m/s** sustained | **STOP** — trajectory diverging | proceed |
| **600M reproduction** | **5.39 m/s** / 295 s | speed < **5.05 m/s** OR lifespan < **200 s** | **STOP** — reproduction gate failed | continue toward 2B |
| **2B stretch** | honest extrapolation ~5.4–5.7 m/s | — | — | target 6.0 m/s if growth re-accelerates |

**Kill criterion (plain):** still in the ~2 s lifespan / ~2–3 m/s regime at **120M**
transitions — same early falsifier that killed `pool600-a002` and
`pool600-fast2b-a001`.

**600M pass:** 50M-bucket mean speed ≥ **5.05 m/s** (pcplus600 − 0.15 tolerance)
with lifespan ≥ **250 s**. Terminal tick 5.385 m/s is the reproduction bar.

### Infrastructure (prepared, not started)

- Config: `training/outputs/experiments/long-horizon-2b-sensors/pcplus2b-a001.json`
- Systemd: `f1tenth-pcplus2b-a001.service` (enabled; prior `f1tenth-pool600-fast2b-a001` disabled)
- GPU idle queue: `recovery_2b.run_ids = ["pcplus2b-a001"]`
- Self-play restored in trainer (`training/selfplay.py` + CLI flags)

### Launch record

| Field | Value |
| --- | --- |
| Prepared (PDT) | 2026-07-27 |
| Status | **pending launch** (awaiting WSL reboot sequencing) |

---

## progab-a001 — piecewise high-speed progress A/B (2026-07-28)

**Question:** does a superlinear forward-progress bonus above 5 m/s widen the
risk budget enough for the policy to exceed the ~5 m/s durability plateau seen
in `pcplus2b-a001`, without sacrificing lifespan?

**Reward shape (forward arc-length `ds` per control step, `dt = control_dt`):**

Let `ds_thr = progress_speed_threshold_mps × dt`,
`ds_sat = progress_speed_saturation_mps × dt`,
`m = progress_high_speed_multiplier`. Shaped forward component `f(ds)`:

| Region | Condition | `f(ds)` |
| --- | --- | --- |
| baseline | `ds ≤ ds_thr` | `ds` |
| boosted | `ds_thr < ds ≤ ds_sat` | `ds_thr + m × (ds − ds_thr)` |
| saturated | `ds > ds_sat` | `ds_thr + m × (ds_sat − ds_thr) + (ds − ds_sat)` |

Total progress term: `progress_scale × f(max(progress_ds, 0))` plus unchanged
backward term on `min(progress_ds, 0)`. Zeroed on boundary events. Defaults
(`threshold=0`, `multiplier=1`, `saturation=0`) recover linear `ds` exactly.

**Champion warm-start:** `pcplus2b-a001/checkpoints/policy_1443840000.pt`
(nearest export to the 1.4466B peak tick @1446604800: **5.672 m/s / 237.0 s**
lifespan; alternates 1433600000 @5.454 m/s / 40 s and 1454080000 @5.451 m/s /
143 s are strictly worse on the speed×durability objective).

**Arms (300M transitions each, sequential on single GPU, seed 42):**

| Arm | Run id | Config | Progress delta |
| --- | --- | --- | --- |
| A (control) | `progab-control-a001` | `progab-control-a001.json` | disabled defaults |
| B (treatment) | `progab-super-a001` | `progab-super-a001.json` | threshold 5.0, mult 3.0, sat 8.0 |

Both: `--init-ckpt` champion, **`actor_freeze_transitions: 20_000_000`**
(critic-only warm-up — see below), self-play as `pcplus2b-a001`, 1024 envs, 3M
replay, **`replay_full_reinit: false`**, `alpha: 0.01`, compile reduce-overhead.

**Checkpoint contents (`policy_1443840000.pt`):** actor weights + actor
`obs_norm` only (`save_policy_artifact` is actor-only by design). No critic,
target-critic, or optimizer state. `load_init_ckpt` loads actor + actor
normalizer; critics always start random. No richer train-state checkpoint exists
in `pcplus2b-a001/checkpoints/` (195 `policy_*.pt` actor artifacts only).

**Critic warm-up:** `actor_freeze_transitions: 20_000_000` (~37.7k critic-only
gradient updates at ~1.89k updates/M; ~32.1k after the 3M buffer fills). Without
freeze, ~10.8k actor updates by 5.7M destroyed the champion (lifespan 1.2 s,
critic_loss still falling). Reference: `warmstart-probe-a001` with 5M freeze +
fixed opponent reached critic_loss ~109 and lifespan ~239 s at unfreeze; self-play
needs a longer window. 20M is ~4× ADR-0020's 5M probe default and 6.7% of the
300M arm horizon.

**Self-play pool @ warm-start:** seed from champion snapshot at
`init_transitions` (trainer default); do **not** rely on `reseed_after_reinit`
(that path only runs when `replay_full_reinit` fires). Pool begins with the
loaded champion policy, then grows via normal snapshot/refresh intervals.

**Learning rates:** keep default `actor_lr=critic_lr=2.5e-05` — same as
`pcplus2b-a001` and prior warm-start probes; appropriate for fine-tuning once
the critic is warm.

**Warm-start anchor (do not compare to zero):** champion metrics at init —
**5.67 m/s / ~237 s** lifespan, timeout-dominated terminations.

### Preregistered gates

Compare **speed at comparable lifespan**, not speed alone. A treatment arm that
gains +0.3 m/s while lifespan falls below 120 s is **not** a win.

**During actor freeze (0–20M):** lifespan must reach **≥ 100 s** and speed
**≥ 5.2 m/s** once the replay buffer is full (~3M). If still **< 20 s** lifespan
or **< 4.5 m/s** after 5M transitions, stop — actor not frozen or checkpoint not
loading. Log must show `actor_freeze_transitions=20000000` and **no** actor
updates (policy weights unchanged; `Actor unfreeze at transitions=…` only @20M).

**At unfreeze + 10M (~30M total):** speed **≥ 5.2 m/s**, lifespan **≥ 100 s**.
Modest dip at unfreeze is acceptable; collapse to ~1 s lifespan is not (critic
warm-up too short).

**Early check @50M:** log 50M-bucket mean speed and lifespan for both arms.
Continue unless an arm drops below **5.0 m/s with lifespan < 100 s** (clear
harm vs warm-start).

| Outcome | Criterion @300M (50M bucket means) |
| --- | --- |
| **Treatment wins** | B mean speed ≥ **5.85 m/s** with lifespan ≥ **200 s**, AND B speed ≥ A speed + **0.10 m/s** at matched lifespan (±15 s window) OR B lifespan ≥ A lifespan + **30 s** at matched speed (±0.05 m/s). Timeout share ≥ 60% for both. |
| **No effect** | \|B − A\| speed < **0.10 m/s** AND \|B − A\| lifespan < **30 s** — reward shape insufficient to shift operating point. |
| **Harm** | Either arm mean speed < **5.20 m/s** OR lifespan < **120 s** with OOB terminations > 40% of episodes; OR B speed > A but B lifespan < **150 s** while A ≥ **180 s** (speed-for-durability trade rejected). |

**Control arm secondary question:** does A reproduce the late `pcplus2b-a001`
collapse from the champion (lifespan decay toward OOB-dominated terminations)?
If A holds ≥ **200 s** lifespan through 300M, the collapse was training-state
specific; if A decays to < **120 s** by 200M, the champion peak was unstable.

### Infrastructure

- Systemd: `f1tenth-progab-control-a001.service` → `OnSuccess=` chain to
  `f1tenth-progab-super-a001.service`
- GPU idle queue: `pcplus2b-a001` marked complete/abandoned; progab arms queued
- Run dirs pre-created under `training/outputs/runs/`

### Launch record

| Attempt | Started (PDT) | Status | Notes |
| --- | --- | --- | --- |
| 1 | 2026-07-28 23:50 | **aborted @~5M** | `replay_full_reinit: true`; Lee reinit @2.998M wiped actor. ~3.6 m/s / 1.2 s. |
| 2 | 2026-07-28 23:57 | **aborted @~5.7M** | `replay_full_reinit: false` but no actor freeze; ~10.8k actor updates on random critic shredded champion (4.15 m/s / 1.2 s, 4055 OOB/0 timeout). |
| 3 | 2026-07-29 00:04 | **live** | `actor_freeze_transitions: 20M`; @1.43M: 5.53 m/s / 58 s lifespan / 37 OOB per window (vs attempt 2 @5.7M: 4.15 m/s / 1.2 s / 4055 OOB) |

| Field | Value |
| --- | --- |
| Prepared (PDT) | 2026-07-28 |
| Status | **relaunch pending** |
