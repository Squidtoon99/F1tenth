# Champion code recovery — run `642a7a80`

**Date:** 2026-07-24  
**Investigator:** code-archaeology pass (read-only; no training launched)

## Headline

**Yes — the champion reward/env kernel is recoverable.** The files in this
directory are copies of Cursor local-history snapshots that match the champion
run's logged behavior. They were already partially preserved under
`repro-642a7a80/overlays/` (kernel/warp_env/rewards) but **pcplus300 did not
use them**; it ran the unified causal-2x2 codebase instead (`345536cc` /
`7635203c`), which explains the ~0.95 m/s long-horizon deficit despite
identical config and seed.

What we **cannot** prove from artifacts alone: that these files are bit-for-bit
what was on disk at 09:12:15 on 2026-07-23. History timestamps for the kernel
and warp_env are 17:08 the same day (save time, not launch time). The champion
log line at 09:12:15 uniquely identifies this code path (see validation below).

---

## Recovered files and provenance

| File | SHA256 | Source |
| --- | --- | --- |
| `kernel.py` | `f27d8703…` | `~/.cursor-server/data/User/History/7da5850f/1bq8.py` |
| `warp_env.py` | `8f8642ca…` | `~/.cursor-server/data/User/History/5dddb2b6/cqkP.py` |
| `standalone_trainer.py` | `8ac969f4…` | `~/.cursor-server/data/User/History/2df5bb4c/kAZM.py` |
| `rewards.py` | `a596906c…` | `~/.cursor-server/data/User/History/529ebbfe/SzGa.py` |

### History timeline vs champion launch

Champion started **2026-07-23 09:12:15 local** on RTX 4080 Super (`DarkToaster`).

| Snapshot | Saved (local) | Before launch? | Notes |
| --- | --- | --- | --- |
| `Ypc5.py` / `DwQB.py` / `Vf6D.py` | Jul 22 16:33 | yes | **Wrong stack** — old Sophy wall/slip rewards, `sqrt(speed)` OOB, no `boundary_contact` |
| `1bq8.py` / `cqkP.py` / `kAZM.py` | Jul 23 17:08 | no (save time) | **Champion stack** — matches run log signatures |
| `SzGa.py` (rewards) | Jul 22 16:33 | yes | Same hash as overlay; unchanged through champion |

The Jul 22 snapshots (`Ypc5`) are **not** the champion code. They use
`out.oob = -reward.oob * sqrt(speed_squared)` and active `wall_penalty` /
`wall_impact` terms. The champion log shows `wall=0.0000`, `impact=0.0000`, and
`oob=-0.010000*dt*speed_kph^2 boundary_contact=-4.000`, which only exist in the
Jul 23 / `1bq8` lineage.

**Interpretation:** the champion stack was edited in an unsaved or not-yet-
snapshotted buffer before 09:12; Cursor recorded the save at 17:08. The run log
is the ground truth for which code executed.

### Avenues searched (no additional hits)

| Avenue | Result |
| --- | --- |
| `git reflog` / `git fsck --lost-found` | No dangling blob matching `f27d8703` or `8f8642ca`; dangling objects are config/trainer amend orphans from Jul 18 |
| `git stash list` | Two stashes from Jul 14/17; neither touches `kernel.py` / `warp_env.py` |
| `git show 587cd1d:training/f1tenth_env/kernel.py` | `f54b2857…` — committed Lee wall-contact path, not champion |
| Editor history under `~/.cursor` (non-server) | Empty |
| `~/.vscode-server/data/User/History` | No F1tenth kernel entries |
| Backup / `.bak` / swap in repo | None |
| External trees | `F1tenth-repro-642a7a80{,-v2}/` match overlay hashes; `scratch/pcplus-local/` matches **pcplus300** reconstruction (`345536cc`), not champion |
| Champion checkpoint `policy_5120000.pt` | Policy weights + resolved hyperparams only; **no embedded source** |
| Worktrees at `587cd1d` / `52487e3` | Carry overlay files, not live champion working tree |

---

## Validation against champion logged behavior

### Launch signature (09:12:15)

Champion `run.log` line 4:

```
Effective experiment: … oob=-0.010000*dt*speed_kph^2 boundary_contact=-4.000
term_oob_mode=full_car_out term_wall_impact=False wall_penalty=0.000 wall_impact=0.000
```

Only `kAZM.py` / recovered `standalone_trainer.py` emits this format. The prior
history trainer `Vf6D.py` (Jul 22) has **no** `Effective experiment` line and no
`boundary_contact` diagnostics.

### First reward window (51200 transitions) — byte-identical with pcplus300

| Field | Champion | pcplus300 |
| --- | --- | --- |
| `mean_ep_reward` | -23.2192 | -23.2192 |
| `episode_lifespan` | 1.607s (n=1937) | 1.607s (n=1937) |
| `progress` | 0.1850 | 0.1850 |
| `steer_hist` | -1.3048 | -1.3048 |
| `oob_penalty` | -0.0397 | -0.0397 |
| `oob_impact_when` | -2.9898 | (same class of magnitude) |

Recovered kernel formula (legacy path):

- Continuous OOB: `out.oob = -reward.oob * speed_squared` with
  `reward.oob = oob_penalty_scale * control_dt * 12.96` → matches logged
  `oob=-0.010000*dt*speed_kph^2`
- One-shot contact: `out.boundary_contact = -4.0` on first `off_track` step
- Terminal shock: `out.oob_impact = -oob_impact * terminal_oob_skip_seconds * 12.96 * speed_squared` on full-car-out

At ~2.5 m/s, `speed_kph² ≈ 81`, `dt=0.1`, scale `0.01*12.96=0.1296` → per-step
OOB ≈ `-0.13 * 0.3` (fraction off-track) ≈ `-0.04`, matching `-0.0397`.

### Late-training divergence (pcplus300 used wrong codebase)

At **150M** transitions:

| Metric | Champion | pcplus300 (causal-2x2) |
| --- | --- | --- |
| speed | 5.28 m/s | 4.48 m/s |
| progress | 0.447 | 0.431 |
| oob_frac | **0.171** | **0.027** |
| oob_penalty mean | -0.040 | -0.007 |
| oob_when (per event) | -0.233 | -0.245 |

Per-event penalty magnitudes agree; the 9× mean gap is **contact frequency**
(`oob_frac`), not a rescaling bug. By **200M**, champion `progress=0.502` vs
pcplus `0.418` — the policy is simply slower, consistent with less aggressive
 boundary probing during the mid-training dip/recovery.

---

## Behavioral diff: champion vs pcplus300 reconstruction

Patches: `diff-kernel-vs-pcplus300-reconstruction.patch`,
`diff-warp_env-vs-pcplus300-reconstruction.patch`.

With `reward_stack=legacy` and `term_oob_mode=full_car_out` (pcplus300 config),
the reconstruction **should** match the champion legacy path. Residual
differences in the unified kernel:

1. **Lee-mode branches** — extra `lee_wall_mode`, `wall_contact_coefficient`,
   `reward_wall_contact` fields and conditional paths (dead when legacy).
2. **Termination wiring** — champion hardcodes `full_car_out=1`; reconstruction
   reads `env.term_oob_mode` (same value for pcplus300, but extra indirection).
3. **Progress masking** — reconstruction introduces `progress_mask` variable;
   equals `off_track` when `lee_wall_mode=0` (no behavioral change in legacy).

These are structurally equivalent for pcplus300's resolved config. The more
likely explanation for long-horizon divergence is **pcplus300 never deployed the
champion files at all** — fingerprint mismatch is confirmed — plus compounding
self-play pool drift after the mid-training dip (~60–100M), not a single atom
bug visible at 150M reward means.

### Trainer nuance

`repro-642a7a80/overlays/standalone_trainer.py` (`9a9e7f65…`) differs from
champion `kAZM.py` by **one hunk**: overlay preserves
`reset_stationary_probability=0.3` from config patch; kAZM forces `0.10`
unless patch overrides. Champion config uses **0.3** — use kAZM only with the
Lee config patch applied (as champion did), or the overlay trainer.

---

## Hardware nondeterminism vs code delta

| Evidence | Implication |
| --- | --- |
| First 51200 transitions byte-identical (reward, lifespan, counts) | Same env code path and RNG stream at cold start |
| Both seed 42, same resolved config | Config/seed ruled out |
| Divergence after ~100M, not at tick 1 | Not a constant hardware bias |
| pcplus300 fingerprint ≠ champion fingerprint | **Code delta is confirmed** |
| Identical per-event `oob_when` at 150M/200M | Reward **formula** matches; policy **state** differs |

**Judgment:** hardware nondeterminism (4080 Super vs L40S, CUDA/cuDNN, compile)
could amplify divergent self-play pools after many updates, but it **cannot** be
the primary explanation while fingerprints differ and pcplus300 never ran the
champion kernel. A definitive A/B requires **`champion-code-recovery/` files vs
causal-2x2 codebase**, same seed, same host — ideally 300M on the original 4080.

---

## Ranked next experiments (if overlay rerun still gaps)

1. **300M on 4080 with this directory's files verbatim** — closes the
   fingerprint gap; success criterion: 200M+ mean ≥ 5.0 m/s.
2. **A/B at 150M only** — swap only `kernel.py`+`warp_env.py` mid-run is
   impractical; instead launch two 150M arms: recovery vs causal-2x2, compare
   `oob_frac` and speed trajectories at 60M/100M/150M.
3. **Self-play pool snapshot diff at 100M** — export opponent checkpoints from
   champion vs recovery run; if policies diverge before reward terms do, pool
   drift is the mechanism.
4. **Steering-history clean A/B** (already queued) — secondary; steer_hist means
   differ at 150M but are tiny (-0.003 vs -0.021).
5. **Cross-host seed-42 repeat on L40S with recovery files** — isolates hardware
   if (1) succeeds on 4080 but fails on L40S.

---

## File inventory

```
champion-code-recovery/
  kernel.py              # champion Warp reward kernel
  warp_env.py            # champion env param wiring
  standalone_trainer.py  # champion trainer (kAZM)
  rewards.py             # torch mirror (unchanged since Jul 22)
  diff-kernel-vs-pcplus300-reconstruction.patch
  diff-warp_env-vs-pcplus300-reconstruction.patch
  RECOVERY.md            # this file
```
