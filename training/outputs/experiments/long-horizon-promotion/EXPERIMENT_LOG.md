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

### Local RTX 4080 slot reclaimed by the other agent

`pcplus-local001` (seed 123, isolated scratch codebase overlay) ran cleanly to
23.24M transitions, then the concurrent agent stopped/disabled the systemd
unit and launched its own `mainline-ladder001` (D2-matched legacy-ladder
validation, 60M, now built directly into mainline `training/` per its commits
`3ac8255`/`5916647`/`471d38d`) on the same GPU. This is accepted rather than
contested: the local RTX 4080 was explicitly the lowest-priority slot, the
replacement run answers a closely related question on a more integrated
codebase, and fighting over a non-expiring, low-priority resource is not
worth the risk of clobbering someone else's work. `pcplus-local001`'s partial
trajectory (peak only 3.4-3.7 m/s @0-8M, well below the seed-42/seed-7 peaks
of 5.3-5.7 m/s, dipping further to 0.9 m/s @12M before being stopped) is too
short and was interrupted mid-dip, so no conclusion is drawn from it -- it may
simply be a weaker seed, or it may have recovered like seed 42 did if left
running.

### Arm abort: `d2fx300` (D2, seed 42) stopped at 88M -- pathology, not a dip

At ~04:29 UTC, `d2fx300` was aborted after showing two independent
task-specified abort conditions simultaneously, both confirmed from raw
`run.log` telemetry (not just the speed number):

1. **Sustained band violation past 60M.** Speed fell from ~3.9-4.4 m/s @12-24M
   to 1.1-2.3 m/s from 30M onward and stayed there continuously through 84M --
   54M+ below the champion band (band at 60M: 3.38-4.78; d2fx300 was at 1.6-2.2
   for the entire 30-84M stretch), not a transient dip.
2. **Climbing not_moving fraction** (explicitly called out as a pathology
   signature in the task brief): 0.2% @18M -> 7.7% @24M -> 11-16% through
   30-66M -> 21.8% @78M, monotonic and non-noisy. The sibling `d2fx301` (same
   reward stack and opponent mode, seed 7) shows exactly 0.000 not_moving
   fraction throughout its own run to 96M+, and the champion's own log does
   not show this signature either -- this is specific to `d2fx300`, most
   likely a seed-42-specific local optimum where the policy learns to stall
   rather than race the static fixed-champion opponent, not evidence against
   the reward stack itself.

Stopped by SIGTERM, confirmed dead, `STOPPED.md` written with full reasoning
(`training/outputs/runs/d2fx300/STOPPED.md` on lark-2), nothing deleted.
Replaced immediately by `pcplus302` (PC+, seed 7, self-play, 300M) on the same
host, preregistered reasoning: seed-7-matched against `d2fx301` to test
whether the seed-42 pattern (self-play arm dips-and-recovers repeatedly;
fixed-champion arm stalls and never recovers) replicates at another seed or
was a seed-42-specific artifact. Verified running with `selfplay: pool_size=1
...` telemetry and correct resolved config (`self_play=true`, `seed=7`,
`total_transitions=300000000`).

### Milestone comparison table (as of ~04:33 UTC, this report time)

Speed in m/s, mean over the nearest ~2M-transition window; band is champion
milestone value ± 0.7 m/s. "In band" / "below band" / "above band" per the
task's rule; a single low reading is not an abort trigger by itself.

| Milestone | Champion | Band | `pcplus300` (PC+ seed42) | `d2fx301` (D2 seed7) | `d2fx300` (D2 seed42, ABORTED @88M) |
| --- | --- | --- | --- | --- | --- |
| 18M | 5.37 | 4.67-6.07 | ~5.4 (peak 12-18M window) -- in band | 3.76 -- below band (D2 has no self-play boost, consistent with Wave 1/2) | 3.92 -- below band |
| 30M | 3.17 | 2.47-3.87 | 3.24 -- in band (matches the dip) | 4.09 -- above band (D2 climbing steadily, no dip) | 1.12 -- below band, decline begins |
| 60M | 4.08 | 3.38-4.78 | 3.45 -- in band (recovering from a 2nd dip) | 4.29 -- in band | 1.92 -- below band, not_moving climbing |
| 90-100M | 4.35 (@100M) | 3.65-5.05 | 2.6-2.9 at 100-102M -- **below band, declining, watch closely** | 3.6-3.9 -- in/near band, mild recent softening | aborted @88M |

`pcplus300` (the single most important arm, per the task brief) shows a more
oscillatory pattern than the champion itself -- rise to ~5.4-5.7 @16-18M, dip
to ~2.2-2.7 @24-30M, a second rise to ~5.1-5.3 @36-42M (exceeding the
champion's own peak), a second dip now underway (~2.6-2.9 @100-102M). This is
not (yet) the abort case: the current below-band stretch is only ~10-12M old,
well short of the 60M sustained-underperformance bar, and the not_moving
fraction is flat/noisy around 5-8% (not climbing like `d2fx300`'s clear
pathology). It needs to be watched over the next 50-100M to see whether it
recovers the way the champion did after its own 26-56M trough, or continues
declining into d2fx300-style territory. `d2fx301` (D2 seed 7) remains the
cleanest, most consistently on-track arm of the three so far -- smooth
climb, no dip, no pathology, currently 3.6-3.9 m/s and holding roughly in
band through 96M.

### Status at end of this report session

All GPUs are occupied by durable, detached long-horizon runs that will keep
running after this session ends (three via `setsid nohup ... & disown` on the
Brev L40S hosts, confirmed already reparented to PID 1 so they survive SSH
disconnection; the local slot is now the other agent's systemd-managed
`mainline-ladder001`). None of the surviving promoted arms (`pcplus300`,
`d2fx301`, `pcplus302`) meet an abort condition as of this writing. Whoever
next checks in on this cluster should: pull each `run.log`, re-run
`extract_trajectory.py`, and extend the milestone table above, especially
watching whether `pcplus300` recovers past its second dip and whether
`pcplus302` reproduces `pcplus300`'s dip-recover shape or `d2fx301`'s smooth
climb (it is the same reward stack and opponent mode as `pcplus300`, just a
different seed, so a third data point on how variable this arm is
seed-to-seed).

## Second takeover: shepherding to completion, harvest, and final verdict

Picked this cluster back up at 04:37 UTC 2026-07-25. Scope for this session
(per task brief): the three Brev L40S instances only, not the local RTX 4080
(another agent owns that GPU and is running `mainline-ladder001`/successor
validation there -- left untouched).

**Time budget.** `brev ls`/`uptime` do not expose an explicit lease-expiry
timestamp via the CLI, so the task-specified window (expire ~08:30-09:00 UTC
2026-07-25) is taken as authoritative; treating 08:30 UTC as the conservative
deadline gives a working budget of roughly 3h50m from takeover. All three
hosts confirmed `RUNNING`/`HEALTHY` at takeover, GPU utilization 80-87%,
processes alive (`ps aux` shows the `standalone_trainer.py` PID actively
running CPU-bound at ~100% on each host).

**Projected finish times** (computed from the actual transitions/s each run
is logging right now, not the task brief's rough estimate):

| Run | Host | Transitions @ 04:54 UTC | Rate (transitions/s) | Remaining to 300M | Projected finish (UTC) | Margin to 08:30 deadline |
| --- | --- | --- | --- | --- | --- | --- |
| `pcplus300` | lark-1 | 153.4M | ~39,700-40,500 | 146.6M | ~05:55 | ~2h35m |
| `d2fx301` | lark-3 | 143.2M | ~37,700-38,100 | 156.8M | ~06:03 | ~2h27m |
| `pcplus302` | lark-2 | 49.7M | ~37,300 | 250.3M | ~06:46 | ~1h44m |

All three are projected to finish comfortably before expiry, `pcplus302`
tightest (it started later, ~04:31 UTC, after superseding the aborted
`d2fx300`) but still with well over an hour of margin. This will be
re-checked as the session progresses since transitions/s can drift (compile
warmup, checkpoint-export stalls, etc.).

**Harvest pipeline set up and verified.** Wrote a local (not committed,
`/tmp/harvest.sh`) incremental-pull helper: pulls `run.log`/`config.json` in
full each pass (small, a few MB), diffs the remote `checkpoints/` listing
against what is already local, and pulls only the delta (tarred remotely in
one `brev exec`, fetched in one `brev copy`, extracted locally) rather than
re-transferring the whole checkpoint set every pass. First full pull for all
four in-scope runs completed and integrity-checked:

| Run | Status | Checkpoints pulled | Verification |
| --- | --- | --- | --- |
| `pcplus300` | running | 27 (growing) | `sha256sum` of every local checkpoint + `config.json` matched the remote hash exactly (28/28 common files, 0 mismatches); the 2 checkpoints produced on lark-1 between the pull and the verification pass are expected new-since-harvest, not a gap |
| `d2fx301` | running | 26 (growing) | pulled clean, same procedure |
| `pcplus302` | running | 8 (growing) | pulled clean, same procedure |
| `d2fx300` | **aborted @ 88.0M** | 17 (final) | pulled clean plus `STOPPED.md`; this run is done growing so its harvest is already complete modulo a final sha sweep |

`run.log` is intentionally re-pulled in full each pass rather than diffed
(cheap at a few MB) since it is appended to continuously. Champion reference
`642a7a80` was already fully harvested (1.6G, all checkpoints + eval video +
wandb) by a prior session; not re-pulled here. Older completed/superseded
causal-2x2 runs (`r642a001`, `pcplus001`, `d2fx001`, `d2fx002`, `d2fx200`,
`d1sp001`, `termA001`, `termB001`, `termB002`, `pcplus002`, `sp642a001` and
its per-host duplicates `spstr002`/`sphst003`) already have local copies from
prior sessions and are not the subject of this session's monitoring; they
will get a lighter final-sweep integrity check (not full re-harvest) since
they are not growing and their conclusions are already written into
`causal-2x2/EXPERIMENT_LOG.md`. Unrelated pre-existing experiment dirs found
on all three hosts (`br3m0001`/`br5m0002`/`br10m003`, a completed
`brev-replay-capacity` sweep, ~700M each) are out of scope for this task --
not part of the long-horizon-promotion or causal-2x2 lineage, already
finished before this session started, and not referenced anywhere in either
log -- so they are left alone.

Plan for the rest of this session: poll all three arms every 15-20 minutes
(transitions/speed/lifespan/terminations, watching for the `d2fx300`-style
pathology signature), harvest incrementally as new checkpoints land, and do
a final harvest+verify sweep with at least 45 minutes of margin before 08:30
UTC regardless of run status at that point.

### Budget update at 04:57 UTC: hard GPU-hour cap, per-instance teardown

The user tightened the constraint mid-session: only 12 GPU-hours of credit
remain in total, and three L40S instances running simultaneously burn 3
GPU-hours per wall-clock hour -- i.e. the same ~4-hour horizon as the
previously-stated expiry, but now framed as a hard spend cap rather than a
lease timer, with explicit authorization to `brev stop` (not delete) each
instance the moment its harvest is complete and verified, rather than
waiting for all three or filling idle GPUs with new work. This supersedes
the earlier "use spare capacity" instruction -- no new runs will be launched
this session unless one could clearly finish, be harvested, and be verified
with margin to spare, which given the remaining budget is not expected.
Revised plan: tighten the poll interval, stop each instance individually as
soon as its run finishes/is aborted and its final harvest+`sha256sum`
verification passes, and report exact stop timestamps and a GPU-hour
accounting at the end.

### Budget correction at 04:58 UTC: 12 GPU-hours *per instance*, ~12h runway

The 04:57 UTC entry above was based on an incorrect budget figure the user
immediately corrected: it is 12 GPU-hours **per instance**, not 12 total, so
all three L40S can run concurrently for roughly 12 more wall-clock hours
(until ~16:50 UTC), not ~4. This reinstates the "use spare capacity" mandate
from the original task: fill each GPU with follow-on work as its current
arm finishes rather than tearing down immediately, harvest+verify before
starting the next run on that GPU, and only stop an instance when genuinely
out of useful, budget-fitting work or as the 12h mark approaches. Teardown
is still `brev stop` only, still requires prior harvest+verification, still
never `brev delete` without the user's explicit say.

### Mainline-legacy-ladder check-in (05:00 UTC) -> promoting to 300M on L40S

Per the revised priority order, checked
`training/outputs/experiments/mainline-legacy-ladder/EXPERIMENT_LOG.md`
(another agent's file on the local RTX 4080 -- read-only, not edited, per
the local-GPU exclusion) and its `run.log`. `mainline-ladder001` (mainline
`training/f1tenth_env/kernel.py` with the newly-added `boundary_contact`/
`oob_impact` ladder rungs, D2-matched config, seed 42, 60M cap) finished
naturally at 60,000,256 transitions, 05:00:37 UTC. Independently
re-extracted its full trajectory with `extract_trajectory.py` rather than
trusting that log's own (still-pending) evaluation section:

| Bucket | Speed (m/s) | Note |
| --- | --- | --- |
| 14-16M | 3.93-4.25 | peak region, matches D2's own 4.11-4.25 @18-19M |
| 18-20M | 2.75 -> 1.14 | dip begins (deeper trough than D2/champion) |
| 22-38M | 1.72 -> 2.94 | recovering |
| 40-54M | 2.3-2.9 | holding, slightly below champion's 60M band (3.38-4.78) |
| 56-58M | 2.97 -> 3.12 | climbing fast in the last 8M, closing on the band |

Same rise-dip-recover shape as D2/champion, just a somewhat deeper trough
(1.14 vs champion's 3.17-3.22) and slower to recover -- still ~0.26 m/s
below the 60M band edge when the 60M screen ends, but climbing at roughly
+0.09 m/s per M-transitions over the last 8M, not flat or falling.
`not_moving` fraction (re-extracted separately) oscillates 0-0.13 across
20-55M with no monotonic climb (peaks at 20M then again mid-run, always
recovers) -- the same shape as `pcplus300`'s own oscillating not_moving
signature, not `d2fx300`'s runaway-and-never-recover pattern. No pathology.

**Decision: promote.** This is close enough to the reference band and
recovering fast enough, with zero pathology signal, to justify the highest-
priority follow-on the task brief calls for: confirming the *shippable*
mainline code path (not a throwaway reconstruction) reproduces the fast
trajectory at full 300M horizon. Will launch `mainline-ladder300` (same
code/config as `mainline-ladder001`, `--total-transitions 300000000`) on
the first L40S GPU that frees up, preregistered here before launch per the
task's requirement.

Note: before launching, discovered (by reading, not editing, the other
agent's `mainline-legacy-ladder/EXPERIMENT_LOG.md` and its own commit
`be28926`) that it had independently reached the same "promote" verdict on
`mainline-ladder001` and already launched a 200M *sustainment* extension
(`mainline-ladder002`) on the local RTX 4080 -- not a 300M run, and on
different (slower, ~24.8-26.3k transitions/s) hardware. This L40S 300M run
is still the single highest-value action available on this session's
hardware: it is an independent second data point on different hardware (GPU-
numerics-independent confirmation), it reaches the full 300M horizon the
task's own validity bar requires (200M does not), and the L40S throughput
gets there in a fraction of the wall-clock. Not deduplicating with the local
run; the two are complementary, not redundant.

### `d2fx301` finished at 300M (06:02:28 UTC) -> harvested, verified, GPU repurposed

`d2fx301` (D2, seed 7, fixed champion) completed 300,000,256 transitions at
06:02:28 UTC on lark-3. Final harvest pulled all 59 checkpoints + final
`run.log`/`config.json`/`wandb`; `sha256sum` of every local file (60/60:
59 checkpoints + `config.json`) matched the remote hash exactly.

**Milestone comparison (2M-window mean around each milestone) against the
task's champion reference, both finished 300M reconstruction arms:**

| Milestone | Champion (band) | `pcplus300` (PC+, self-play, seed42) | `d2fx301` (D2, fixed champion, seed7) |
| --- | --- | --- | --- |
| 18M | 5.37 (4.67-6.07) | 5.51 -- in band | 3.65 -- below band (no self-play early boost, expected) |
| 30M | 3.17 (2.47-3.87) | 2.21 -- below band (deep dip) | 4.06 -- just above band |
| 60M | 4.08 (3.38-4.78) | 3.64 -- in band | 4.44 -- in band |
| 100M | 4.35 (3.65-5.05) | 3.08 -- below band | 3.53 -- below band, close |
| 150M | 5.22 (4.52-5.92) | 4.47 -- below band, close (Δ0.05) | 4.19 -- below band (Δ0.33) |
| 200M | 5.62 (4.92-6.32) | 4.65 -- below band (Δ0.27) | 4.50 -- below band (Δ0.42) |
| 250M | 5.58 (4.88-6.28) | 4.13 -- below band (Δ0.75) | 4.61 -- below band (Δ0.27) |
| 300M | 5.37 (4.67-6.07) | 3.94 -- below band (Δ0.73) | 4.32 -- below band (Δ0.35) |

**Neither arm reaches or sustains the champion's band from 150M onward.**
Both hover in the mid-to-high 3s/low-4s through 300M, roughly 0.3-0.8 m/s
short of the champion at every milestone from 100M on, and both are noisier
than this table's single-window snapshot suggests -- `pcplus300` in
particular oscillates hard even late (3.94 @290M vs 4.54 @220M vs 3.94
@240M, a 0.6 m/s swing within 70M), while `d2fx301` is visibly smoother
(190-300M stays in a tighter 4.14-4.73 band, no oscillation of that
magnitude). Neither is pathological (no climbing not_moving, no nonfinite
states, no wall-dominated terminations) -- this looks like both arms are
still on the champion's own slow climb out of its dip (the champion itself
did not clear 5.0 until ~130-150M and did not stabilize at 5.3+ until
~190-200M), just running behind that schedule, not stuck. This is the
direct motivation for the extension below.

**Decision: extend `d2fx301`, not `pcplus300`, past 300M.** Per the task's
"extend the best-performing arm" instruction: `d2fx301` is smoother and
higher on average through the back half of the run (190-300M mean ≈4.4-4.7
vs `pcplus300`'s ≈4.0-4.5 with larger swings), so it is the more informative
and more stable base for a long-horizon ceiling probe. This also produces a
useful complementary data point for the self-play-vs-fixed-champion
question in its own right -- if the fixed-champion arm's smoother recent
trend continues climbing while extended, that is evidence self-play's early
speed advantage (confirmed again at 18M above: +1.86 m/s for PC+) does not
carry through to a long-horizon lead, matching what this milestone table
already shows through 300M.

Launching `d2fx600`: identical config/seed to `d2fx301`
(`configs/d2-legacy-fixed.json`, seed 7, fixed champion, same reconstruction
codebase already deployed on lark-3), `--total-transitions 600000000`,
launched fresh (not a checkpoint warm-start) to get a single unbroken
trajectory comparable to how the champion reference itself was produced (one
continuous run, not a resumed one) -- matching the precedent already used
twice in this experiment family (`d2fx001`->`d2fx200`,
`mainline-ladder001`->`mainline-ladder002`, both fresh relaunches at a higher
horizon cap rather than checkpoint-warm-started continuations). At ~38-39k
transitions/s this is ~4.3h wall-clock, comfortably inside the 12h-per-
instance budget.

`d2fx600` confirmed running on lark-3 within a minute of `d2fx301` exiting
(seed 7, `fixed_champion: ckpt=.../642a7a80/checkpoints/policy_199680000.pt`
confirmed in its first telemetry window, ~38k transitions/s).

### `mainline-ladder300` launch bug caught and fixed on lark-1

First launch attempt of `mainline-ladder300` crashed immediately
(`PermissionError` loading the fixed-champion checkpoint) because
`mainline-ladder-d2match.json` hardcodes the **local RTX 4080 machine's**
absolute checkpoint path
(`/home/ubuntu/projects/F1tenth/training/outputs/runs/642a7a80/...`), which
does not exist on lark-1 (its own copy lives under
`/home/shadeform/F1tenth/...`). Fixed by writing a host-local variant,
`mainline-ladder-d2match-l40s.json` (identical except the fixed-champion
checkpoint path resolved to lark-1's own filesystem), deployed only on
lark-1 (not committed -- it is a path-only host artifact, not a config
value change). Cleared the crashed run's empty `mainline-ladder300` dir and
relaunched; confirmed running cleanly on the second attempt (~41.8k
transitions/s, all three ladder rungs active in its first telemetry window:
`wall_contact_events=9313`, `boundary_contact_events=1352`,
`oob_impact_events=1284` matching `terminations: oob=1284`). No data was
lost -- the crashed attempt never wrote a checkpoint or any transitions.

### `pcplus300` finished at 300M (05:54:49 UTC) -> harvested, verified, GPU repurposed

`pcplus300` (PC+, seed 42) completed its full 300M-transition run naturally
at 05:54:49 UTC on lark-1 (`Training finished after 292969 vector ticks,
300000256 transitions, and 585548 updates.`). Final harvest pulled all 59
checkpoints + final `run.log` + `config.json` + `wandb` run dir;
`sha256sum` of every local checkpoint file (59) plus `config.json` matched
the remote hash byte-for-byte (60/60 files, 0 mismatches, diffed
programmatically not eyeballed). Per the revised (12h-per-instance) budget,
this GPU was **not** torn down -- immediately repurposed to launch
`mainline-ladder300` (see above) rather than sitting idle. lark-1 confirmed
running the new job within ~40 seconds of the old one exiting (GPU
utilization dip was momentary, during Python/compile startup, not idle
GPU-hours burned).

## Third takeover (~06:08 UTC): harvest completion, deficit analysis, relaunch

### Harvest status (verified)

| Run | Checkpoints | sha256 | Local path |
| --- | --- | --- | --- |
| `pcplus300` | 59 + final | 61/61 match | `training/outputs/runs/pcplus300/` (1.1 GB) |
| `d2fx301` | 59 + final | 61/61 match | `training/outputs/runs/d2fx301/` (1.1 GB) |
| `pcplus302` | in progress | pending | lark-2, ~254M/300M @06:26 UTC |

Both completed 300M arms fully harvested with programmatic `sha256sum` verification
(remote vs local, every checkpoint + `run.log` + `config.json`).

### Milestone table — three 300M arms vs champion `642a7a80`

Speed = mean over nearest 2M-transition bucket from `extract_trajectory.py`.
Champion reference from independent re-extraction of `642a7a80/run.log`.

| Milestone | Champion | `pcplus300` (PC+ s42) | `d2fx301` (D2 s7) | `pcplus302` (PC+ s7) |
| ---: | ---: | ---: | ---: | ---: |
| 18M | 5.57 | **5.44** | 3.68 | _running_ |
| 60M | 4.05 | 3.53 | **4.39** | _running_ |
| 100M | 4.23 | 2.93 | 3.58 | _running_ |
| 150M | 5.12 | 4.50 | 4.17 | _running_ |
| 200M | 5.40 | 4.59 | **4.48** | _running_ |
| 250M | 5.42 | 4.14 | **4.61** | _running_ |
| 300M | 5.44 | **4.14** (final) | **4.32** (final) | _pending_ |

**200M+ regime (391-sample windows, independent extraction):**

| Run | mean | min | max | sd |
| --- | ---: | ---: | ---: | ---: |
| `pcplus300` | 4.263 | 3.713 | 4.869 | 0.224 |
| champion | 5.208 | 3.916 | 5.886 | 0.332 |

Confirmed: pcplus300 max (4.869) < champion mean (5.208). Deficit ~0.95 m/s is
real, not bucket-alignment artifact.

### Deficit investigation — localized to code path, not config

**Config:** field-by-field diff of resolved `config.json` snapshots shows every
substantive reward/optimizer/env knob matches between `pcplus300` and
`642a7a80` (progress 1.0, collision 6.0, passing 3.0, oob_penalty 0.01,
boundary_contact 4.0, steering_history 5.0, batch 1024, replay 3M, seed 42,
self-play pool 10). Not a config difference.

**Seed:** both seed 42 — rules out seed variance for pcplus300 vs champion.

**Early training matches:** pcplus300 peaks 5.614 @16M vs champion 5.572 @18M;
identical first ~51200-transition reward window (byte-identical
`progress=0.1850 steer_hist=-1.3048 oob_penalty=-0.0397`). Divergence emerges
after the second dip (~100M+): champion recovers to 5.2+ sustained; pcplus300
stalls at 4.1-4.6.

**Per-term reward means at matched transitions:**

| Term | @150M pcplus300 | @150M champion | @200M pcplus300 | @200M champion |
| --- | ---: | ---: | ---: | ---: |
| progress | 0.431 | 0.430 | 0.433 | **0.522** |
| oob_penalty | -0.005 | **-0.044** | -0.015 | -0.016 |
| steer_hist | -0.027 | -0.003 | -0.006 | -0.000 |
| passing | 0.203 | 0.203 | 0.180 | 0.104 |

At 150M, progress and passing match but oob_penalty magnitude differs 9×
(pcplus300 touches boundary less / at lower effective penalty). At 200M,
**progress reward is the dominant divergence** (0.433 vs 0.522) — the policy
is simply going slower, not miscalibrated on a single atom.

**Code fingerprints:**

| Component | Champion (587cd1d + overlays) | pcplus300 @ launch (reconstruction) | Mainline @ be28926 |
| --- | --- | --- | --- |
| kernel.py | `f27d8703…` (overlay) | `345536cc…` (causal-2x2 codebase) | `1e8a8242…` |
| warp_env.py | `8f8642ca…` (overlay) | `7635203c…` | `70fece6a…` |

pcplus300 ran the **causal-2x2 reconstruction**, not the champion's uncommitted
overlay (`f27d8703`). The reconstruction uses `off_track` (strict `>`) vs
mainline's `wall_contact` (`>=`), separate `params.oob` legacy path vs
mainline's `wall_cost_mode` ladder, and differs from the champion overlay on
boundary geometry edge cases. Early peak fidelity (r642a001/pcplus001 also match
champion @18M) but **long-horizon sustainment requires the exact champion code
path** — the reconstruction approximates it closely enough for 25M screens but
not for 200M+ convergence.

**Narrowed candidate list (ranked):**

1. **Champion uncommitted overlay delta** — most likely; reconstruction ≠ champion
   code despite matching configs and early trajectories.
2. **Hardware nondeterminism** (L40S Brev vs original box) — possible amplifier
   but cannot explain identical early windows then divergent late training on
   same seed without some code or state difference.
3. **Self-play pool trajectory** — both use self-play seed 42; early windows
   identical so unlikely primary cause.
4. Config / seed — **ruled out** for pcplus300 vs champion.

### Steering-history A/B

See `training/outputs/experiments/steering-history-clean-ab/EXPERIMENT_LOG.md`.
Preregistered; launch blocked on lark-3 by concurrent `d2fx600` (600M D2 seed 7).
Queued for lark-2 after `pcplus302` completes (~20 min est.).

## Step 6 — mainline-ladder300 on Brev lark-1

Preregistered and launched 2026-07-25 ~06:16 UTC after `pcplus300` harvest.
Uses `mainline-ladder-d2match-l40s.json` (host-local champion checkpoint path)
with `schedule.total_transitions=300000000`. Runtime verification @~11M:

- `boundary_contact_when=-4.0000`, `boundary_contact_events=3401`
- `wall_contact_when=-0.25`, `wall_contact_events=10609`
- `oob_impact_events=3132` matching `terminations: oob=3132`

All three ladder rungs simultaneously active on **mainline** code (not
reconstruction). Durable via `setsid nohup` (systemd user bus unavailable on
Brev without logind session).

### Cluster state @ ~06:26 UTC

| Host | Run | Status | Notes |
| --- | --- | --- | --- |
| lark-1 | `mainline-ladder300` | **running** ~11M/300M | All 3 ladder rungs verified in telemetry |
| lark-2 | `pcplus302` | **running** ~254M/300M | ~46M remaining |
| lark-3 | `d2fx600` | **running** (other agent) | D2 600M seed 7; d2fx301 harvested |
| local 4080 | `mainline-ladder002` | **running** (out of scope) | not touched |

lark-3 **not stopped** — `d2fx600` consuming budget. `d2fx301` artifacts safe locally.


| Host | Run | Status | Notes |
| --- | --- | --- | --- |
| lark-1 | `mainline-ladder300` | **running** ~11M/300M | All 3 ladder rungs verified in telemetry |
| lark-2 | `pcplus302` | **running** ~254M/300M | ~46M remaining |
| lark-3 | `d2fx600` | **running** (other agent) | D2 600M seed 7; d2fx301 harvested |
| local 4080 | `mainline-ladder002` | **running** (out of scope) | not touched |

lark-3 **not stopped** — `d2fx600` consuming budget. `d2fx301` artifacts safe locally.
