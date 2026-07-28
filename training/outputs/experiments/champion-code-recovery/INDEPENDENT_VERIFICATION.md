# Independent verification — 2026-07-25

Second code-archaeology pass confirming commit `618232f` and extending analysis
with log-level evidence at matched transition counts.

## Recovery confirmed

| Check | Result |
| --- | --- |
| Cursor history `1bq8.py` / `cqkP.py` / `kAZM.py` SHA256 | Match `champion-code-recovery/` exactly |
| Overlay `repro-642a7a80/overlays/` kernel+warp_env | Same hashes as recovery |
| Jul 22 snapshot `Ypc5.py` | **Wrong stack** — `sqrt(speed_squared)` OOB, no `boundary_contact` |
| Jul 23 snapshot `1bq8.py` saved 17:08 PDT | **Champion stack** — matches 09:12 run log |
| `git fsck --lost-found` | No blob matching `f27d8703` or `8f8642ca` |
| Checkpoint `policy_5120000.pt` | Policy weights + hyperparams only; no embedded source |
| Remote push | `618232f` on `origin/experiments/e2e-sim-sensors` |

Champion launch signature (line 4, `642a7a80/run.log`):

```
Effective experiment: … oob=-0.010000*dt*speed_kph^2 boundary_contact=-4.000
term_oob_mode=full_car_out … wall_penalty=0.000 wall_impact=0.000
```

Only the recovered `standalone_trainer.py` (`kAZM.py`) emits this format.

## Critical nuance: legacy-mode kernel equivalence

Full `diff` of `f27d8703` (champion) vs `345536cc` (pcplus300 causal-2x2) shows
**no behavioral delta** when `reward_stack=legacy` and
`term_oob_mode=full_car_out` (pcplus300 resolved config):

- Added Lee branches (`lee_wall_mode`, `wall_contact`) are guarded and dead.
- `progress_mask` equals `off_track` when `lee_wall_mode=0`.
- `full_car_out` resolves to `1` from config, matching champion hardcode.
- `off_track` / `wall_contact` geometry predicates are **byte-identical**.

Fingerprint mismatch is real but, for pcplus300's config, structurally
equivalent. A 300M rerun swapping only `kernel.py`+`warp_env.py` to recovery
files **may not** close the gap unless pre-09:12 champion code differed from
`1bq8.py` (unsaved edits) or `standalone_trainer.py` differences matter.

`standalone_trainer.py` diff vs causal-2x2 is ~2800 lines (self-play manager
refactor, diagnostics, fixed-opponent support). Training loop semantics should
be checked before attributing deficit to env kernel alone.

## Per-term reward evidence (matched transitions)

### First window @ 51200 — byte-identical

| Field | Champion | pcplus300 |
| --- | --- | --- |
| progress | 0.1850 | 0.1850 |
| steer_hist | -1.3048 | -1.3048 |
| oob_penalty | -0.0397 | -0.0397 |
| speed | 2.535 | 2.535 |

### @ 150016000 — 9× oob_penalty is frequency, not formula

| Field | Champion | pcplus300 |
| --- | --- | --- |
| speed | **5.223** | 4.451 |
| progress | 0.4295 | 0.4308 |
| passing | 0.2033 | 0.2034 |
| oob_penalty mean | **-0.0442** | -0.0050 |
| oob_when (per event) | -0.2000 | -0.1946 |
| oob_frac | **0.220** | 0.025 |
| wall_contacts (diag) | 11264 | 0 |

Per-event penalty magnitudes agree (~−0.20). Mean gap =
`oob_frac` ratio (0.220/0.025 ≈ 8.8×). Champion policy runs faster **and**
spends ~9× more steps off-track — consistent with aggressive boundary probing
during recovery, not a rescaling bug.

### @ 200192000 — progress dominates

| Field | Champion | pcplus300 |
| --- | --- | --- |
| speed | **5.470** | 4.615 |
| progress | **0.5056** | 0.4260 |
| oob_penalty | -0.0174 | -0.0207 |
| oob_frac | 0.064 | 0.062 |
| passing | 0.0793 | 0.1905 |

By 200M, oob terms have converged; **progress reward** (0.506 vs 0.426) tracks
the speed deficit. The policy is simply slower, not miscalibrated on OOB atoms.

## Hardware vs code

| Evidence | Read |
| --- | --- |
| Byte-identical first 51200 window | Same env + RNG at cold start |
| Identical resolved reward/optimizer knobs | Config ruled out |
| Divergence after ~100M only | Not tick-1 hardware bias |
| Legacy kernels structurally equivalent | Fingerprint alone insufficient to explain gap |
| oob_when matches at 150M | Reward **formula** matches; **policy state** differs |

Hardware (4080 Super vs L40S, compile, cuDNN) could amplify divergent
self-play pools after many gradient updates, but cannot explain identical
early windows then late divergence **without** some state trajectory difference
(policy weights, opponent pool). Code vs hardware is not fully separable from
artifacts; definitive test is 300M on the **original 4080** with recovery
files vs causal-2x2 codebase, same seed.

## Ranked next experiments

1. **300M on RTX 4080 with `champion-code-recovery/` files verbatim** — closes
   deployment gap; success = 200M+ mean ≥ 5.0 m/s sustained.
2. **Kernel-swap null hypothesis** — if (1) matches pcplus300 despite recovery
   files, legacy kernels are equivalent and deficit is policy/self-play drift.
3. **Self-play pool checkpoint diff @ 100M** — export opponent policies from
   both runs; if weights diverge before reward terms, pool drift is the mechanism.
4. **Trainer A/B** — recovery `kAZM.py` vs causal-2x2 trainer, same recovery
   kernel, 150M — isolates ~2800-line trainer refactor.
5. **Cross-host seed-42 on L40S with recovery files** — isolates hardware if
   (1) succeeds on 4080 but fails on L40S.
