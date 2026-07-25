# Long-horizon promotion: pivoting the cluster off 25-30M screens

## The finding, independently re-verified

Every causal-2x2 arm (and the local termination-semantics-probe arms) has been
screened at 25-30M transitions. `training/outputs/experiments/long-horizon-promotion/extract_trajectory.py`
was written to re-extract the champion run's speed trajectory directly from
`training/outputs/runs/642a7a80/run.log` (parsing `transitions=` / `speed=`
pairs, bucketed by transition count) without trusting any previously reported
numbers. Result (2M buckets, mean speed):

| Transitions | Mean speed (m/s) | Note |
| --- | --- | --- |
| 18M | 5.57 (max 5.91) | peak region |
| 20M | 4.50 | starting to fall |
| 26M | 3.49 | |
| 30M | 3.22 | trough region |
| 40M | 3.33 | |
| 56M | 3.57 | trough ends |
| 60M | 4.05 | climbing |
| 100M | 4.23 | |
| 120M | 4.21 | |
| 130M | 4.89 | |
| 150M | 5.12 | |
| 190-200M | 5.29-5.40 | |
| 250M | 5.42 | |
| 300M | 5.44 | |
| 450M | 4.85 (still >4.5 throughout) | sustained to end of run |

This independently confirms the user's reported shape almost exactly: **peak
~5.4-5.9 m/s at 18-20M, collapse to 3.1-3.6 m/s sustained from 26M through
~56M, a slow climb to 4.0-4.4 m/s across 60-120M, 4.9-5.2 m/s at 130-150M, and
5.3-5.7 m/s sustained from ~190M all the way to 450M.** My own point estimates
differ from the reference points quoted in the task by at most ~0.2 m/s (e.g.
my 18M=5.57 vs quoted 5.37; my 200M=5.40 vs quoted 5.62) -- consistent with
different bucket alignment/sampling, not a different shape. **The dip-and-
recover pattern is real and reproduces from raw log text.**

Second check: the two reconstruction runs (`r642a001`, `pcplus001`, distinct
files, seed 42, same config) were both killed at 24,985,600 transitions.
Re-extracting them the same way shows both decaying to 2.41 m/s (min 1.94) in
the 24-25M bucket, after peaking at 5.44 mean / 5.70 max in the 18M bucket --
**exactly the champion's dip, at exactly the point they were killed.** The
finding is confirmed; the plan is not void.

Conclusion: a 25-30M screen is a valid *early discriminator* (does the arm
reach ~5 m/s by 16-19M?) but not a verdict on final speed. Ranking arms by
speed@25M, or concluding an arm has "regressed" because it is at 3-3.5 m/s in
the 26-60M window, is invalid. The Lee reward stack is a separate case: per
the task brief it has independently been run to 150-200M in `68e197ed`,
`sp642a001`, `edc66f07` and plateaued at 1.7-3.3 m/s -- that conclusion is not
based on a truncated screen and stands.

## Step 1: cluster inventory and takeover

At takeover (~03:36 UTC 2026-07-25), another agent's 25M screens (`d2fx002`
lark-1, `termA001` lark-2, `termB001` lark-3) were all >73% complete and
finished naturally within the following ~3 minutes -- consistent with the
task's "let it finish if >80% done" rule (all three crossed that threshold,
or finished outright, before any action was needed). Their final trajectories
were pulled and used to inform the Step 2 arm choice (see below); nothing was
lost.

Immediately after those three finished, the other agent launched three new
runs on the same three hosts (`d2fx200`/lark-1 promoting D2 to 200M,
`termB002`/lark-2 and `pcplus002`/lark-3, two more 25M seed-7 screens). These
were only ~3-5 minutes old (7-9M / 25M or 200M transitions each -- 4.6-33%
progress) when this takeover began, so the "let it finish" exception did not
apply. Each was stopped with `kill -TERM`, confirmed dead, and given a
`STOPPED.md` next to its artifacts (`training/outputs/runs/{d2fx200,termB002,
pcplus002}/STOPPED.md` on the respective Brev host) recording the transition
count reached and the reason. Checkpoints (`policy_5120000.pt`), `run.log`,
`config.json`, and wandb run dirs were left in place on all three, nothing
deleted. The other agent's own `causal-2x2/EXPERIMENT_LOG.md` (this is a
shared local repo checkout -- the other agent's edits and commits land in the
same working tree) picked up and accurately documented this handoff within
minutes, referencing the `STOPPED.md` files, so no further write to that log
was needed from this side.

Local RTX 4080 Super: `f1tenth-termination-semantics-probe-costshape.service`
(`termsem002`, Lee continuous-quadratic wall cost + recoverable boundary +
fixed champion, 30M cap) had already finished cleanly (exit 0) at 20:48 PDT,
before this takeover began -- nothing to stop. Its own trajectory (peak 1.8
m/s @16M, 1.0-1.3 m/s plateau through 30M) reinforces the existing evidence
that the Lee reward stack plateaus low regardless of termination geometry;
not a candidate for promotion.

## Step 2: promotion launches

### Preregistration (written before launch)

- **lark-1 -> `pcplus300`**: PC+ config (legacy reward + self-play), seed 42,
  `--total-transitions 300000000`. Per the task brief, this is the direct
  champion-trajectory reproduction and single most important run -- the one
  arm with three independent confirmations of the rise-dip shape (champion,
  r642a001, pcplus001), so confirming it recovers past the dip and reaches
  5+ m/s at long horizon validates the whole reconstruction.
- **lark-2 -> `d2fx300`**: D2 config (legacy reward + fixed champion, no
  self-play), seed 42, `--total-transitions 300000000`. Tests whether the
  fixed-opponent variant also recovers/sustains -- strictly preferable for
  future work if so (reproducible, no self-play non-stationarity).
- **lark-3 -> `d2fx301`** (D2, seed 7, 300M) -- chosen over the task's other
  suggested candidates (second PC+ seed, or promoting TERM-A/TERM-B) for a
  concrete evidence-based reason found during takeover: the just-completed
  `d2fx002` (D2, seed 7, 25M screen) climbed **smoothly and monotonically**
  from 2.9 to 3.85 m/s over the full 25M with **no dip at all** -- a visibly
  different shape from self-play PC+'s hard dip. That is a novel,
  decision-relevant signal (self-play's non-stationarity may be what drives
  the mid-training dip, not the legacy reward stack itself) that a second PC+
  seed would not add, since PC+'s dip-recovery shape already has three
  independent confirmations. TERM-A (Lee reward: peaked only 1.2-1.4 m/s,
  disqualified per the "no known-plateau reward stack" rule) and TERM-B
  (legacy cost + first-contact termination: rose to 3.68 m/s @10M then
  collapsed to 1.1-1.5 m/s by 14-24M, with **no established long-horizon
  recovery precedent** for that specific termination geometry, unlike
  recoverable-boundary legacy) were both excluded.
- **Local RTX 4080 (lowest priority) -> `pcplus-local001`**: a *third* seed of
  PC+ (seed 123), also at 300M, run from an isolated scratch copy of
  `training/` (`/home/ubuntu/scratch/pcplus-local`, mainline files untouched,
  `outputs/` symlinked back to the real run-artifact tree) with the causal-2x2
  reconstruction overlaid on top -- verified identical via `sha256sum` against
  the files deployed on lark-1, except a 3-line wandb-panel diff already noted
  as a known, harmless fix in the causal-2x2 log. Chosen to fill the
  "second/third PC+ seed" gap the L40S allocation above deliberately left
  open, without competing with the three time-boxed L40S slots; local GPU
  does not expire so a slower per-step rate is an acceptable tradeoff.

### Verification at launch (resolved config + runtime telemetry, not just the config file)

| Run | Host | Seed | Reward stack | Opponent | `total_transitions` (resolved) | Runtime confirmation |
| --- | --- | --- | --- | --- | --- | --- |
| `pcplus300` | lark-1 | 42 | `legacy` / `term_oob_mode=full_car_out` | self-play | 300,000,000 | `args.self_play=true`; log shows `selfplay: pool_size=... win_rate=...` lines |
| `d2fx300` | lark-2 | 42 | `legacy` / `full_car_out` | fixed champion | 300,000,000 | `args.self_play=false`, `args.fixed_opponents=true`; log shows `fixed_champion: ckpt=.../642a7a80/checkpoints/policy_199680000.pt` |
| `d2fx301` | lark-3 | 7 | `legacy` / `full_car_out` | fixed champion | 300,000,000 | same fixed-champion signature as above, seed 7 |
| `pcplus-local001` | local RTX 4080 | 123 | `legacy` / `full_car_out` | self-play | 300,000,000 | `args.self_play=true`; `selfplay: pool_size=1 ...` present |

All four launched detached and durable: the three Brev runs via
`setsid nohup ... & disown` (matching the pre-existing pattern found already
running on the hosts -- their PIDs were already reparented to PID 1, i.e. the
same launch style); the local run via a systemd user unit
(`f1tenth-pcplus-local300.service`, `enabled`, so it also survives reboot),
matching the existing `f1tenth-termination-semantics-probe*` convention.

## Step 3: champion-relative milestone comparison

Reference trajectory (champion `642a7a80`, from the re-extraction above, closest
2M-bucket mean to each task-specified milestone) and an on-track band of
±0.7 m/s per the task's rule:

| Milestone | Champion (m/s) | Band |
| --- | --- | --- |
| 18M | 5.37 (my extraction: 5.57) | 4.67-6.07 |
| 30M | 3.17 (mine: 3.22) | 2.47-3.87 |
| 60M | 4.08 (mine: 4.05) | 3.38-4.78 |
| 100M | 4.35 (mine: 4.23) | 3.65-5.05 |
| 150M | 5.22 (mine: 5.12) | 4.52-5.92 |
| 200M | 5.62 (mine: 5.40) | 4.92-6.32 |
| 250M | 5.58 (mine: 5.42) | 4.88-6.28 |
| 300M | 5.37 (mine: 5.44) | 4.67-6.07 |

Per-arm status will be appended below as each run accumulates enough
transitions to reach a milestone; this is a multi-hour experiment and the
table will be filled in incrementally as of the time this report is
finalized (see "Status as of report time" below), not all at once.

### Status as of report time (see final report for the actual clock-time snapshot)

Runs were only launched moments before this section was written; the useful
signal available immediately is that all four are alive, GPU-bound, and
running the correct config (see verification table above), not yet a
milestone comparison. Progress will be tracked by re-running
`extract_trajectory.py` against each run's `run.log` at intervals; see the
per-arm sections appended below.
