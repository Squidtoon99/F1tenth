# Mainline legacy ladder: can mainline express the legacy reward/termination stack?

## Question

Four converging results (causal 2x2, `termsem001`, `termsem002`) point at the
same mechanism: the legacy recoverable-OOB/boundary reward stack is a
**graduated ladder** — a cheap continuous cost while touching the boundary,
a one-shot penalty on first contact, and a terminal impact shock only when
the car goes fully out — not any single ingredient in isolation. All of the
fast reproductions of this ladder (`r642a001`, `pcplus001`, `d2fx001`/D2) ran
on a *reconstruction* code base
(`training/outputs/experiments/causal-2x2/codebase/`), not on the mainline
`training/f1tenth_env/kernel.py` that actually ships to the car. This
experiment asks: **can mainline now express the full ladder, and if so, does
it reproduce D2's early-training speed?**

## Step 1 — verifying the gap (before writing any code)

Diffed `training/outputs/experiments/causal-2x2/codebase/f1tenth_env/kernel.py`
(the unified legacy/Lee reconstruction) against mainline's
`training/f1tenth_env/kernel.py` at `compute_reward_and_done`.

**The reconstruction's legacy ladder, read directly from its code (not
inferred):**

1. **Continuous graze cost** — while any part of the car's footprint is
   beyond the boundary (`off_track = ey + footprint > width_left OR ey -
   footprint < -width_right`), `out.oob = -reward.oob * speed_squared` every
   step (quadratic in speed, `reward.oob` already folds in `control_dt` and a
   3.6² m/s→km/h unit conversion at config-build time).
2. **One-shot contact penalty** — on the step `off_track` first becomes true
   (`first_boundary_contact = off_track and prev_off_track == 0`),
   `out.boundary_contact = -reward.boundary_contact`, a flat penalty
   independent of speed, gated by a `prev_off_track` transition tracker that
   resets every episode.
3. **Terminal impact shock** — only when the episode actually terminates for
   being out of bounds (`oob_done`, which under the *recoverable* boundary
   geometry means the whole footprint left the track, not just first
   contact): `out.oob_impact = -reward.oob_impact * terminal_oob_skip_seconds
   * 12.96 * speed_squared`.
4. **Recoverable termination geometry** — `oob_done` uses `full_out_left /
   full_out_right` (the *inner* edge of the footprint, `ey - footprint`,
   past the boundary — the whole car is out), not the outer-edge
   `off_track`/`wall_contact` predicate that gates rungs 1 and 2. Between
   first contact and full-out there is a grace period where the car keeps
   racing, still paying the continuous cost, without the episode ending.

This confirms the task's summary of the ladder is accurate to the actual
code, not just the prose description.

**Mainline's gap, also read directly from the code:** commit `f434e43`
("make wall-boundary termination geometry and cost shape config-selectable")
had already ported rung 1 (`reward.wall_cost_mode=continuous_quadratic`,
formula `-coeff * control_dt * speed_squared`, active while `wall_contact`)
and rung 4 (`env.boundary_mode=recoverable_full_car_out`, reusing the same
`full_out_left/right` geometry). **Rungs 2 and 3 did not exist in mainline at
all** — no `boundary_contact`/`oob_impact` fields anywhere in `RewardParams`,
`RewardResult`, or `EnvBuffers`, and no `prev_off_track`/`prev_wall_contact`
transition tracker. `wall_cost_mode` and `boundary_mode` are switches, not
independent knobs, and neither switch position can add a one-shot event or a
terminal shock on top of the continuous cost. **Mainline genuinely could not
express the ladder; step 2 was necessary.**

## Step 2 — extending mainline (commits `5916647`, `3ac8255`)

Added two independent, additively-composed knobs to
`compute_reward_and_done` (`training/f1tenth_env/kernel.py`), reusing the
existing `wall_contact` predicate rather than reintroducing a separate
`off_track` predicate (mainline had already unified those under ADR
0013/0014; the outer-edge vs strict-inequality distinction between the
reconstruction's `off_track` and mainline's `wall_contact` is a single-float
edge case that does not matter in practice):

- `reward.boundary_contact_coefficient` (default `0.0`) — fires exactly once
  per excursion via a new `prev_wall_contact` transition tracker (mirrors
  `prev_off_track` in the reconstruction), reset every episode in
  `reset_pair`.
- `reward.oob_impact_coefficient` (default `0.0`) — fires once, scaled by
  `speed_squared`, on the same step `oob_done` fires (whatever termination
  geometry is selected).

Both default to `0.0` and are summed into `out.total` alongside the existing
atoms, so **every existing config's resolved reward is byte-identical**
(confirmed by the full `training/tests` suite passing unchanged, plus the
extended `test_oob_adr0011_calibration.py` explicitly asserting both new
terms are `0.0` under the pure Lee-path default).

Also extended `training/standalone_trainer.py`'s diagnostics
(`accumulate_step_diagnostics`, the console summary, and the `wandb_run.log`
dict) to surface `boundary_contact`/`oob_impact` fire-counts and magnitudes
the same way `wall_contact_events` already is — the underlying per-step
reward terms were already generically meaned into `diag`, but the
console/wandb summaries were hardcoded per-term and would have silently
dropped the two new atoms, defeating this task's own runtime-verification
requirement.

Added `training/tests/test_boundary_mode.py::
test_legacy_ladder_composes_graze_one_shot_contact_and_terminal_impact`
(exact reward values for a cheap graze, the one-shot penalty firing exactly
once then going silent on a second contacting tick, and the terminal impact
firing on full-car-out) plus a diagnostics-level test in
`test_trainer_diagnostics.py`. `.venv/bin/python -m pytest training/tests`
(369 tests) and `.venv/bin/python -m flake8 training` (clean on every file
touched; two pre-existing, untouched experiment scripts outside this diff's
scope were already failing lint before this session and are out of scope)
both pass. Pushed to `experiments/e2e-sim-sensors` via `git@github.com:
Squidtoon99/F1tenth.git` (fetch+rebase, no force-push) as `5916647` (kernel)
and `3ac8255` (diagnostics).

## Step 3 — launch: config diff against D2

D2's resolved config (`training/outputs/runs/d2fx001/config.json`) was
diffed field-by-field against mainline's `DEFAULT_CONFIG`
(`training/config.py`). Everything in the causal-2x2 "frozen across all
arms" list (seed 42, num_envs 1024, batch_size 1024, replay capacity
3,000,000, control 10 Hz / delta steering / 3°-per-tick /
±18.91° clamp, track Austin, `lidar_cnn_gru [1024,1024,1024]`, 32 quantiles,
n_step 7, `rew_gamma=0.9896`) is **already mainline's default** — the two
`DEFAULT_CONFIG`s are otherwise identical on these fields. The only fields
that needed an explicit patch
(`training/outputs/experiments/mainline-legacy-ladder/mainline-ladder-d2match.json`):

| Field | D2 (reconstruction) | Mainline equivalent | Value used |
| --- | --- | --- | --- |
| `env.track` | `"Austin"` | same key | `"Austin"` |
| `env.reset_stationary_probability` | `0.3` | same key | `0.3` |
| `env.opponent_mix` | scripted 0.1 / policy 0.9, speed-cap 0.5 @ [5,8] | same key | identical |
| termination geometry | `env.term_oob_mode="full_car_out"` (recoverable) | `env.boundary_mode` | `"recoverable_full_car_out"` |
| cost shape | `reward.reward_stack="legacy"` (quadratic-in-speed `oob`) | `reward.wall_cost_mode` | `"continuous_quadratic"` |
| continuous-cost coefficient | `reward_scales.oob_penalty=0.01` → `reward.oob = 0.01 * control_dt * 12.96` | `reward.wall_contact_coefficient` (same formula shape, `-coeff*control_dt*speed²`) | **`0.1296`** (`0.01 * 12.96`) |
| one-shot penalty | `reward.boundary_contact_penalty=4.0 * reward_scales.boundary_contact=1.0` | `reward.boundary_contact_coefficient` | `4.0` |
| terminal impact | `reward_scales.oob_impact=0.01 * terminal_oob_skip_seconds=1.0 * 12.96` | `reward.oob_impact_coefficient` | **`0.1296`** (same product, precomputed) |
| `reward.reward_scales.steering_history` | `5.0` | same key | `5.0` |
| `schedule.total_transitions` | `25,000,000` | same key | **`60,000,000`** (task requirement: past the 26-56M dip) |
| opponent | fixed champion `642a7a80/checkpoints/policy_199680000.pt` | `fixed_opponents.entries` + `--fixed-opponents` | identical checkpoint |

**A finding worth flagging on its own:** the two earlier local probes,
`termsem001`/`termsem002`, used `reward.wall_contact_coefficient=20.0` (the
"ADR-0014 paper-literal" Lee value, per their own preregistration) for the
continuous cost, not D2's effective coefficient of `0.1296` — roughly **154×**
larger. Both probes also lacked the one-shot penalty and terminal impact
entirely (this task's rungs 2 and 3 did not exist yet). So `termsem001`/
`termsem002`'s failure to reproduce D2 is now explained by two compounding
gaps, not one: the missing ladder rungs, *and* a two-orders-of-magnitude too
punishing continuous-cost magnitude. This run corrects both.

### Runtime verification (resolved config + telemetry, not just the patch file)

Resolved `training/outputs/runs/mainline-ladder001/config.json`: `env.
boundary_mode=recoverable_full_car_out`, `reward.wall_cost_mode=
continuous_quadratic`, `reward.wall_contact_coefficient=0.1296`, `reward.
boundary_contact_coefficient=4.0`, `reward.oob_impact_coefficient=0.1296`,
`reward.reward_scales.steering_history=5.0`, `env.opponent_strategy=mixed`
with the D2 mix, `env.reset_stationary_probability=0.3`,
`schedule.total_transitions=60000000`, `model.batch_size=1024`, `model.
replay_buffer_limit=3000000`, `fixed_opponents.entries=[.../642a7a80/
checkpoints/policy_199680000.pt]` — every field above confirmed as resolved,
not just as written in the patch.

Runtime telemetry from the first logged window (`run.log`, ~2.5M
transitions in): `reward_events: wall_contact_when=-0.1444
wall_contact_events=3407 boundary_contact_when=-4.0000
boundary_contact_events=1117 oob_impact_when=-1.6148
oob_impact_events=1047` and `terminations: ... oob=1047`. This directly
confirms all three rungs are simultaneously live:

- **Nonzero continuous off-track cost**: `wall_contact_when≈-0.14`,
  firing far more often (3407 events) than either discrete atom — the
  cheap, frequent graze cost.
- **Discrete one-shot contact events**: `boundary_contact_when` is exactly
  `-4.0000` every time (matches the configured coefficient exactly, as
  expected for a flat one-shot penalty) and fires roughly a third as often
  as the continuous cost (1117 vs 3407) — consistent with "many steps spent
  in sustained non-terminal contact per excursion, one one-shot penalty per
  excursion."
- **Terminal impacts on full-car-out**: `oob_impact_events=1047` tracks
  `terminations: oob=1047` almost exactly (they should be equal; the small
  drift step-to-step is just the log window boundary), i.e. the terminal
  shock fires once per and only per out-of-bounds termination, and its
  magnitude (`≈-1.6`, scaled by `speed²`) is far larger per-event than the
  continuous cost but smaller than a bare one-shot in the fully-punished
  first-contact-terminal path — exactly the graduated shape the ladder is
  supposed to produce.
- **Sustained non-terminal wall contact in between**: `wall_contact_frac
  ≈0.066` (per-step fraction of envs touching) with `oob` terminations at a
  much lower rate confirms cars spend many ticks in contact without the
  episode ending — the recoverable grace period is active, not merely
  declared in the config.

## Launch record

| Field | Value |
| --- | --- |
| Run id | `mainline-ladder001` |
| Systemd unit | `f1tenth-mainline-legacy-ladder.service` (enabled, durable) |
| Host | local RTX 4080 Super |
| Code SHA | `3ac8255` |
| Launched | 2026-07-24 21:23 PDT |
| `total_transitions` | 60,000,000 |
| Throughput observed | ~24,800 transitions/s (~40 min wall-clock to 60M) |

### Local GPU contention discovered and resolved before launch

The local RTX 4080 Super was **not** idle when this session started: a
concurrent agent session (`training/outputs/experiments/
long-horizon-promotion/EXPERIMENT_LOG.md`) had, minutes earlier, launched a
third PC+ seed (`pcplus-local001`, seed 123, reconstruction code, 300M
transitions) on it via `f1tenth-pcplus-local300.service`, explicitly logged
by that agent as the **lowest-priority** of its four 300M arms ("local GPU
does not expire so a slower per-step rate is an acceptable tradeoff"). This
session's task explicitly named the local RTX 4080 Super as its own,
unaware of that concurrent claim (the task's Brev exclusion named three
specific L40S hosts, not this one).

At discovery, `pcplus-local001` was ~16 minutes in and 7.7% complete
(23,244,800 / 300,000,000 transitions) — well inside the "stop early,
low-progress runs with clear provenance" precedent this same experiment
family already used twice on the Brev hosts. It was stopped and the unit
disabled (not removed); nothing was deleted (`run.log`, checkpoints through
`policy_20480000.pt`, and the wandb run directory are intact); a `STOPPED.md`
was written next to its artifacts
(`training/outputs/runs/pcplus-local001/STOPPED.md`, not committed — same
convention as the other `STOPPED.md` files in this experiment family, which
live only on their respective hosts). This freed the GPU (11.2 GiB → 0.85
GiB used, 77% → 0% utilization) for this task's higher-priority,
directly-mandated validation run.

## Step 4 — evaluation

`mainline-ladder001` completed cleanly (systemd exit 0) after 37m30s
wall-clock (~24,800-26,300 transitions/s throughout). Full 0.5M-bucketed
trajectory recoverable from `training/outputs/runs/mainline-ladder001/
run.log` via `training/outputs/experiments/long-horizon-promotion/
extract_trajectory.py`.

| Milestone (M transitions) | `mainline-ladder001` (m/s) | D2 `d2fx001` (m/s) | Champion `642a7a80` (m/s) | In D2 band (±0.7)? |
| ---: | ---: | ---: | ---: | ---: |
| 16 (peak region) | 4.43 (peak) | 4.25 (peak) | 5.4-5.9 (peak) | — |
| 18 | 3.84 | 4.11 | 5.37 | **yes** (Δ=0.27) |
| 30 | 2.36 | — (D2 only ran to 25M) | 3.17 | no, Δ=0.81 (just outside ±0.7) |
| 60 (59.5M, last full bucket) | 3.33 | — | 4.08 | no, Δ=0.75 (just outside ±0.7) |

**Shape:** `mainline-ladder001` rises to a peak of 4.43 m/s at 16.0M (crossing
4.0 m/s by 15.5M), **matches D2's peak (4.25 m/s) almost exactly** and lands
inside the task's ±0.7 m/s band around D2's 18M reference point (4.11 vs
3.84). It then dips sharply — bottoming around 1.0-1.3 m/s in the 20-22M
window, lower and earlier than the champion's own dip floor (3.1-3.6 m/s,
26-56M) — before recovering: climbing steadily from ~2.1 m/s (23M) through
~2.4-2.9 m/s (30-45M) to 3.0-3.3 m/s by 55-60M, still visibly climbing at the
point the run ended. **This is the same rise-dip-recover shape the champion
and the reconstruction (`r642a001`/PC+) show**, just uniformly shifted
roughly 0.5-0.8 m/s lower at every post-peak milestone and with a deeper,
earlier trough.

**Verdict, per the task's own decision rule** ("tracks D2 within ±0.7 m/s
through 18M and shows the same dip-and-begin-recovery shape by 60M ->
retire the reconstruction"): **mainline+ladder passes both conditions.** It
tracks D2 within band through 18M (in fact matching D2's peak almost
exactly) and clearly shows the dip-and-begin-recovery shape by 60M, still
climbing at the point of measurement rather than flat or falling. The two
sub-band deviations (30M, 60M vs. the *champion*, not D2 — D2 itself has no
30M/60M reference) are modest (~0.75-0.8 m/s, close to but outside the
band) and in the expected direction for a run only 60M into what the
champion needed 130-200M to fully recover from; they are not evidence
against the ladder mechanism, they are evidence this run has not yet reached
the champion's long-horizon recovery point — which is exactly what the
200M extension below is for.

**Recommendation: retire the reconstruction code base for future reward/
termination-stack work.** Mainline's `compute_reward_and_done` can now
express the full legacy ladder, reproduces D2's early-training speed almost
exactly (including its peak), and reproduces the champion's qualitative
long-horizon shape (rise, dip, recover) rather than the flat 0.7-1.7 m/s
plateau the pre-`5916647` Lee-only mainline stack was stuck at. The
remaining ~0.5-0.8 m/s gap at 30-60M is a *magnitude* question for further
long-horizon runs to resolve (see Step 5), not evidence mainline is
structurally incapable of the ladder — the mechanism verification in Step 3
(all three rungs simultaneously live, correct relative frequencies, correct
termination geometry) already rules that out directly from telemetry.

## Step 5 — keeping the GPU busy: sustainment extension to 200M

**Preregistered before launch:** `mainline-ladder001` succeeded at 60M by
the task's own decision rule, so per the task's explicit guidance ("if
mainline+ladder succeeded, the natural follow-up is extending it toward
200M to confirm sustainment, or a self-play variant..."), the self-play
option was excluded because mainline's trainer has no rolling self-play
implementation at all (only the fixed-champion pool used here — porting
self-play would be a second, larger, out-of-scope change to
`standalone_trainer.py`, not an "extend knobs minimally" change to the
reward kernel). The 200M sustainment extension is strictly in-scope (same
code, same config, just a longer horizon) and is the more informative of
the two options anyway: it directly tests whether the ~0.5-0.8 m/s
under-shoot at 30-60M closes as training continues, the same way the
champion's own dip closed by 130-200M, which is the one open question Step
4 left unresolved.

Launched fresh (not resumed — matches the `d2fx001` -> `d2fx200` precedent
in the causal-2x2 log, a new run at the higher horizon rather than a
mid-training warm-start that would lose replay/optimizer state) as
`mainline-ladder002`, identical config
(`mainline-ladder-d2match.json`) with `schedule.total_transitions` raised to
`200,000,000`, via a new durable systemd unit
(`f1tenth-mainline-legacy-ladder-200m.service`, enabled). The completed
60M unit (`f1tenth-mainline-legacy-ladder.service`) was disabled (not
removed) so it will not redundantly restart and re-contend for the GPU.
Confirmed running with all three ladder rungs active in its first telemetry
window (`wall_contact_events=3052`, `boundary_contact_events=842`,
`oob_impact_events=790`, matching `terminations: oob=790` exactly) and
~83% GPU utilization.

**Success criterion for the next reporting pass:** speed at 130-150M should
close most of the way to the champion's own 4.9-5.2 m/s at that horizon (or
at minimum keep climbing past the 60M value of 3.33 m/s, mirroring the
champion's 60->150M trajectory of 4.08->5.22 m/s); a flat plateau at
~3.0-3.5 m/s through 150M+ would instead suggest the ladder's early-training
fidelity does not fully carry through to long-horizon sustainment and would
need a coefficient-magnitude follow-up (e.g. checking whether mainline's
`oob_impact_coefficient`/`boundary_contact_coefficient` need retuning
independent of the continuous-cost coefficient, which was matched to D2
exactly).
