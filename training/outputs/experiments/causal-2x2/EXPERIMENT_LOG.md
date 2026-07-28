# Causal 2x2: reward stack x opponent mode — running log

## Preregistration (written before Wave 1 results were known)

**Question.** Two documented cells: legacy recoverable-OOB reward + rolling
self-play = FAST (r642a001, 5.61 m/s peak @17.5M); Lee first-contact
wall-contact reward + fixed champion = SLOW (sp642a001, ~3.3 m/s plateau over
200M). The 2x2 is confounded — reward stack and opponent mode both changed
between the two known cells. Wave 1 fills the missing diagonal.

| Cell | Reward stack | Opponent | Run id | Host | Status before Wave 1 |
| --- | --- | --- | --- | --- | --- |
| Known FAST | legacy | self-play | r642a001 | lark-1 (prior run) | done, 5.61 m/s peak |
| Known SLOW | Lee | fixed champion | sp642a001 | lark-1 (prior run) | done, ~3.3 m/s plateau |
| PC+ (positive control) | legacy | self-play | pcplus001 | lark-1 | **launching** |
| D1 | Lee | self-play | d1sp001 | lark-2 | **launching** |
| D2 | legacy | fixed champion | d2fx001 | lark-3 | **launching** |

**Frozen across all three new arms:** seed 42, num_envs 1024, batch_size 1024,
replay capacity 3,000,000, reset_stationary_probability 0.3, steering_history
reward scale 5.0, control at 10 Hz (control_interval=20, sim_dt=0.005),
steering_delta_max_rad=0.05235987755982988 (~3 deg/tick),
delta_max=0.33 rad (~18.91 deg absolute clamp), track=Austin,
total_transitions=25,000,000, actor=lidar_cnn_gru [1024,1024,1024],
num_quantiles=32, n_step=7, rew_gamma=0.9896.

**Decision rule at 20-25M** (positive-control reference: 5.0 m/s by 16M,
peak 5.61 m/s @17.5M in r642a001):
- D2 fast, D1 slow -> reward/termination stack is causal.
- D1 fast, D2 slow -> opponent regime is causal.
- Both fast -> either intervention alone suffices.
- Both slow -> the two interact; PC+ is the only viable base found so far.
- PC+ itself fails to reach 5.0 m/s by ~20M -> STOP, report immediately, the
  positive control does not replicate and everything downstream is invalid.

## Code-base decision (critical, resolved before launch)

Three candidate code bases were inspected:

1. **Current committed HEAD** (`4dd982f`, then `f434e43` after a concurrent
   push mid-session — see below): implements Lee first-contact wall-contact
   reward/termination (`kernel.py: oob_done = wall_contact`,
   `out.wall_contact = -wall_contact_coefficient * sqrt(v^2)`), but has **no
   rolling self-play** — only a fixed-champion pool
   (`champion_mgr.bootstrap_opponent`). It cannot express self-play, so it
   cannot run PC+ or D1.
2. **Legacy reconstruction** (the overlay files copied onto `lark-1` for
   r642a001 — `kernel.py`/`warp_env.py`/`rewards.py`/`config.py`, plus a
   locally-extended `standalone_trainer.py` with `--self-play` /
   `--mixed-opponents`): implements the legacy recoverable-OOB stack
   (`oob_done = full_out_left or full_out_right`, one-shot `boundary_contact`,
   continuous quadratic-in-speed `oob_penalty`, terminal `oob_impact` shock)
   and has self-play, but **no Lee wall-contact reward and no fixed-champion
   support** (`FixedChampionManager`/`select_champion` were never imported
   into this trainer). It cannot express D1 or D2 as originally written.

Neither base alone could run all three arms without confounding "which code
path is executing" with "which reward stack/opponent mode is active". Per the
task instructions, the **legacy reconstruction was extended** (not
replaced) to add the missing capabilities, so one identical file set runs all
three (and, in principle, all four) cells:

- `f1tenth_env/kernel.py` / `warp_env.py`: added `reward.lee_wall_mode` /
  reused `termination.full_car_out` (already declared but hardcoded to `1`
  and dead), both now driven by a single config key `env.term_oob_mode`
  (`"full_car_out"` = legacy formula, `"wall_contact"` = Lee formula, ported
  verbatim from committed HEAD). Legacy and Lee formulas are now two branches
  of the same `compute_reward_and_done` kernel function instead of two
  different files.
- `standalone_trainer.py`: ported `FixedChampionManager`/`select_champion`
  support (`--fixed-opponents` + config `fixed_opponents.entries`) from
  committed HEAD into the self-play-capable reconstruction trainer, so
  `--self-play` and `--fixed-opponents` are both available side by side.

This is the smallest change that lets PC+, D1, and D2 share **one file set**
(verified identical by `sha256sum` across all three Brev hosts before launch).
It does not touch the mainline `training/f1tenth_env/kernel.py` /
`training/standalone_trainer.py` (which a concurrent commit, `f434e43`, was
independently modifying on the same branch mid-session — see below); the
unified reconstruction lives only under
`training/outputs/experiments/causal-2x2/codebase/` and is deployed to the
Brev hosts by direct copy, the same mechanism r642a001 used.

**Concurrent mainline activity noticed mid-session:** commit `f434e43`
("training: make wall-boundary termination geometry and cost shape
config-selectable", author Arjun Nayak) landed on
`origin/experiments/e2e-sim-sensors` between this session's start and the
code-base decision above. It adds `env.boundary_mode`
(`first_contact_terminal` / `recoverable_full_car_out`) and
`reward.wall_cost_mode` (`one_shot` / `continuous`) directly to the mainline
Lee kernel — a narrower, orthogonal decomposition of the *same* two axes
raised in this task's follow-up item (b), but it does not restore the full
legacy stack (`oob_penalty`/`boundary_contact`/`oob_impact`) or self-play, so
it does not by itself satisfy Wave 1's need to run PC+. It was fetched,
confirmed already pushed, and left untouched; this experiment's commit
(`780591f`) was rebased cleanly on top of it (fast-forward, no conflicts).
Follow-up item (b) below should reconcile with `f434e43` rather than
duplicate it.

## Verification performed before/at launch (all three arms)

- Resolved `config.json` read back on each host: `term_oob_mode`,
  `opponent_strategy`, `reward_scales`, `fixed_opponents.entries`,
  `args.self_play`, `args.fixed_opponents`, and all frozen knobs confirmed
  correct per arm (see per-arm sections below).
- Runtime log lines: legacy arms (PC+, D2) show non-zero `oob_penalty`,
  `oob_impact_events`, and `boundary_contact`-driven terminal shocks, with
  `wall=nan` (Lee atom inert); Lee arm (D1) shows `oob_penalty=0.0000`
  exactly (legacy atom scaled to zero and never assigned) and elevated
  `term/out_of_bounds` event rate consistent with first-contact (not
  full-car-out) termination.
- Opponent mode: PC+/D1 show `Self-play opponent bootstrapped from snapshot`
  and periodic `selfplay: pool_size=... win_rate=...` lines; D2 shows
  `fixed_champion: ckpt=.../642a7a80/checkpoints/policy_199680000.pt
  transitions=199680000` (same champion checkpoint sp642a001 used).
- `nonfinite: obs_rate=0.00e+00 reward_rate=0.00e+00 state_rate=0.00e+00` on
  all three at first log interval.
- `gradient_updates` non-zero and increasing on all three within the first
  log interval.
- sha256 of all 6 deployed files identical across lark-1/2/3.

Known gap: the wandb `reward/wall_contact` panel was not wired into the
periodic `wandb_run.log(...)` dict in the version launched for Wave 1 (only
the console/`reward_term/*` diag path had it). Fixed in the saved codebase
for follow-up waves; Wave 1 verification instead relies on the resolved
config + the `oob_penalty`/termination-rate signatures above, which are
unambiguous.

## Launch record

| Arm | Run id | Host | PID | Launched (UTC) | W&B |
| --- | --- | --- | --- | --- | --- |
| PC+ | pcplus001 | lark-1 | 49644 | 2026-07-25 03:11 | group causal-2x2-780591f, tags cell-PCplus/reward-legacy/opponent-selfplay/seed42/sha780591f |
| D1 | d1sp001 | lark-2 | 47347 | 2026-07-25 03:11 | group causal-2x2-780591f, tags cell-D1/reward-lee/opponent-selfplay/seed42/sha780591f |
| D2 | d2fx001 | lark-3 | 45773 | 2026-07-25 03:11 | group causal-2x2-780591f, tags cell-D2/reward-legacy/opponent-fixed/seed42/sha780591f |

Code SHA at launch: `780591f` (`training: unify legacy OOB/boundary and Lee
wall-contact reward stacks for causal 2x2`), pushed to
`origin/experiments/e2e-sim-sensors`.

## Wave 1 results (all three arms completed, 25M transitions each)

| Cell | Reward stack | Opponent | Run id | Peak (m/s) | Speed @18M (m/s) |
| --- | --- | --- | --- | --- | --- |
| Known FAST (reference) | legacy | self-play | r642a001 | 5.699 @19.0M | 5.648 |
| Known SLOW (reference) | Lee | fixed champion | sp642a001 | 3.667 @3.0M (early transient; decays to ~1.7 by 18M, ~0.46 by 200M) | 1.677 |
| **PC+** (positive control) | legacy | self-play | pcplus001 | 5.554 @19M | 5.493 |
| **D1** | Lee | self-play | d1sp001 | 2.837 @3M (early transient; decays) | 0.701 |
| **D2** | legacy | fixed champion | d2fx001 | 4.252 @19M | 4.112 |

Full 0.5M-bucketed trajectories: `speed-trajectories.json`. Raw logs/configs:
`training/outputs/runs/{pcplus001,d1sp001,d2fx001}/`.

**Positive control check:** PC+ crosses 5.0 m/s at 16.0M (5.079) and peaks at
5.554 @19M — matches r642a001 (crossed 5 m/s at 16.0M, peak 5.614-5.699
@17.5-19M) closely enough to confirm the unified reconstruction code base
reproduces the known-fast cell. **PC+ replicates; proceed with the diagonal
read.**

**Causal read:**
- D2 (legacy + fixed) is **dramatically faster than D1** (Lee + self-play):
  4.112 vs 0.701 m/s @18M, a ~5.9x gap, and faster than the known-SLOW
  reference itself (sp642a001, Lee + fixed, 1.677 @18M).
- D1 (Lee + self-play) is **no better than — arguably worse than —** the
  known-SLOW reference (0.701 vs 1.677 @18M). Self-play does not rescue the
  Lee reward/termination stack.
- This matches decision-rule branch 1 exactly: **"D2 fast, D1 slow -> the
  reward/termination stack is causal."** Swapping only the reward stack
  (holding opponent mode fixed) moves speed@18M by +2.4 m/s (fixed-champion
  pair: sp642a001->D2) to +4.8 m/s (self-play pair: D1->PC+). Swapping only
  the opponent mode (holding reward stack fixed) moves speed@18M by only
  -1.0 m/s under Lee (sp642a001->D1, self-play is slightly *worse*) or +1.4
  m/s under legacy (D2->PC+, self-play helps once the reward stack already
  works).
- **Conclusion: the legacy recoverable-OOB/boundary reward and termination
  stack is the dominant causal factor for the speed ceiling, not opponent
  mode.** Opponent mode is a secondary, sign-flipping lever: self-play is a
  synergistic amplifier on top of a working (legacy) reward stack (+1.4 m/s)
  but does nothing — or mildly hurts — on top of the Lee stack. D2 alone
  (legacy + fixed, no self-play) already recovers most of the gap to the
  champion trajectory (4.1 vs 5.5-5.6 m/s @18M), so the reward stack is
  necessary and largely sufficient; self-play is a smaller additive
  contribution, not a second required ingredient.

## Wave 2 (launched immediately on the three freed GPUs)

Preregistered before launch:

1. **D2-seed7** (`d2fx002`, lark-1, seed=7 else identical to D2): variance
   check on the pivotal new result (legacy + fixed champion is fast). If
   seed=7 also clears ~4 m/s by 18M, the D2 finding is not a seed artifact.
2. **TERM-A** (`termA001`, lark-2): Lee's continuous per-step
   `wall_contact` cost (`reward.reward_stack=lee`) + **recoverable**
   full-car-out termination (`env.term_oob_mode=full_car_out`, instead of
   Lee's normal first-contact) + self-play (matches D1's opponent mode).
   Decomposes D1's failure: is it the cost *shape* (continuous linear-in-speed)
   or the *termination geometry* (immediate first-contact vs a recoverable
   grace period) that kills speed? If TERM-A is fast, recoverability (not
   cost shape) is what matters; if still slow, the linear wall_contact cost
   itself is insufficient regardless of termination.
3. **TERM-B** (`termB001`, lark-3): legacy quadratic
   `oob_penalty`/one-shot `boundary_contact`/terminal `oob_impact` cost
   (`reward.reward_stack=legacy`) + **first-contact** termination
   (`env.term_oob_mode=wall_contact`, instead of legacy's normal
   full-car-out) + fixed champion (matches D2's opponent mode). Mirror test:
   does legacy's cost shape survive Lee-style immediate termination, or does
   D2's speed depend on the recoverable grace period specifically?

This required one small, additive code change (decoupling the previously
single `env.term_oob_mode` flag, which drove both reward-atom selection and
termination geometry together, into two independent knobs:
`env.term_oob_mode` for termination geometry only, and the new
`reward.reward_stack` for cost-atom selection only — both already existed as
separate fields in the `compute_reward_and_done` kernel, they were just wired
to the same config key). Re-verified with `sha256sum` identical across all
three hosts after redeploy; Wave 1's already-completed configs were updated
to set `reward.reward_stack` explicitly (no behavior change, since it matches
what `term_oob_mode` implied before the decoupling).

## Wave 2 results (all three arms completed, 25M transitions each)

| Arm | Run id | Peak (m/s) | Speed @18M | Speed @25M (final) |
| --- | --- | --- | --- | --- |
| D2-seed7 | d2fx002 | 3.884 @21M | 3.630 | 3.838 |
| TERM-A (Lee cost + recoverable term + self-play) | termA001 | 2.232 @0M (transient) | 1.279 | 1.334 |
| TERM-B (legacy cost + first-contact term + fixed) | termB001 | 3.852 @11M | 2.477 | 1.503 |

**D2 variance check:** seed=7 (3.630-3.884 range) closely tracks seed=42's D2
(3.9-4.2 range) — the D2 finding (legacy + fixed champion is fast, ~2-3x the
Lee-stack baseline) is **not a seed artifact**.

**Decomposition read (cost shape vs termination geometry):**

| Cost shape | Termination | Opponent | Run | Speed @18M |
| --- | --- | --- | --- | --- |
| legacy | recoverable (full-car-out) | fixed | D2 (d2fx001) | 4.112 |
| legacy | first-contact | fixed | TERM-B (termB001) | 2.477 (and still falling — 1.503 @25M) |
| Lee | first-contact | self-play | D1 (d1sp001) | 0.701 |
| Lee | recoverable (full-car-out) | self-play | TERM-A (termA001) | 1.279 |

Neither ingredient alone is sufficient: Lee's cost shape stays slow even with
a recoverable grace period (TERM-A, 1.28 vs D1's 0.70 — a small, not
qualitative, improvement), and legacy's cost shape collapses under
first-contact termination (TERM-B falls from 4.11 to 2.48 @18M and keeps
decaying to 1.50 by 25M — the same downward trajectory shape as D1/TERM-A,
not a plateau). **The recoverable-termination and legacy-cost-shape
ingredients only work together; the causal reward/termination stack is a
package, not decomposable into one dominant sub-factor.** This refines Wave
1's conclusion: "reward/termination stack is causal" is correct, but the
mechanism is specifically recoverability (a grace period that lets the
policy recover from a brush with the wall instead of ending the episode)
combined with the legacy cost formula, not either alone.

## Wave 3 (launched immediately on the three freed GPUs)

Preregistered before launch:

1. **D2 promoted to 200M** (`d2fx200`, lark-1, seed=42, else identical to
   D2): item (c) from the task. D2 is the cleanest fast cell that isolates
   the reward stack without the self-play confound, and unlike PC+ (which
   r642a001 already showed regresses after ~20M) its long-horizon behavior
   is unknown. Tests whether removing self-play also removes the late-run
   regression seen in r642a001/PC+, or whether that regression is a property
   of the legacy reward stack itself regardless of opponent mode.
2. **TERM-B-seed7** (`termB002`, lark-2, seed=7): variance check on Wave 2's
   most surprising finding (first-contact termination alone collapses
   legacy's cost-shape advantage, continuing to decay rather than
   plateauing) — confirms it is not a seed artifact before trusting the
   "package, not decomposable" conclusion above.
3. **PC+-seed7** (`pcplus002`, lark-3, seed=7): variance check completing
   seed-robustness coverage for the two self-play arms (D2 already had a
   seed check; PC+, the strongest cell overall, did not yet).

| Arm | Run id | Host | PID | Launched (UTC) |
| --- | --- | --- | --- | --- |
| D2 @ 200M | d2fx200 | lark-1 | 51872 | 2026-07-25 03:40 |
| TERM-B seed7 | termB002 | lark-2 | 49684 | 2026-07-25 03:40 |
| PC+ seed7 | pcplus002 | lark-3 | 48121 | 2026-07-25 03:40 |

D2@200M is a long run (~85-90 min at ~38-40k transitions/s); TERM-B-seed7 and
PC+-seed7 are the standard 25M screens (~10-11 min). All three verified
running (high GPU utilization, process alive) shortly after launch.

### Wave 3 outcome: superseded by a concurrent cluster-wide pivot

All three Wave 3 arms were terminated by SIGTERM at ~03:45 UTC, roughly five
minutes after launch, by a concurrent agent session working on the same three
Brev hosts (`training/outputs/experiments/long-horizon-promotion/`). Each
stopped run carries a `STOPPED.md` written by that agent explaining the
takeover; nothing was deleted.

Its stated reason: champion run `642a7a80` was re-verified to follow a
rise–dip–recover speed trajectory (peak ~5.6-5.9 m/s @18-20M, collapse to
3.1-3.6 m/s sustained through 26-56M, recovery to 5.0-5.7 m/s by 130-200M+),
so the 25-30M window this whole 2x2 screened in is inside a normal mid-training
dip rather than a final-quality verdict. The cluster was repurposed to 300M
promotion runs: `pcplus300` (PC+, seed 42) on lark-1, `d2fx300` (D2, seed 42)
on lark-2, `d2fx301` (D2, seed 7) on lark-3 — all three confirmed running with
78-82% GPU utilisation at 03:52 UTC, so no GPU is idle.

Implication for this experiment: the Wave 1 and Wave 2 conclusions remain valid
as statements about *early* (≤25M) learning speed, which is what they measured,
but they should not be read as final-quality verdicts. The long-horizon runs
supersede them on that question.

Partial Wave 3 data (all pulled locally with sha256 verification, 10 files each,
one checkpoint at 5.12M transitions per run):

| Arm | Run id | Stopped at | Speed at stop | Note |
| --- | --- | --- | --- | --- |
| D2 @ 200M | d2fx200 | 9.22M | 3.63 m/s | climbing; superseded in place by d2fx300 |
| TERM-B seed7 | termB002 | 8.24M | 3.24 m/s | flat ~3.1-3.24 since 0.5M; seed42 twin had already begun decaying by this point |
| PC+ seed7 | pcplus002 | 7.94M | 3.02 m/s | tracks seed42 twin's early shape |

Trajectories for all three are merged into `speed-trajectories.json`. No
conclusions are drawn from them — each covers only about a third of its
intended screen.
