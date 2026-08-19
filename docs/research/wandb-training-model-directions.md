# W&B training evidence and future-model directions

**Date:** 2026-07-31  
**Scope:** Analysis only. No training or production behavior was changed.

## Executive summary

The strongest measured findings are:

1. **Training with a 4 m/s governor produced unusually stable aggregate speed.**
   In `ego-cap4-noedge-1b-a001`, the mean after 100M transitions was
   **3.948 m/s** with a per-log-window standard deviation of **0.089 m/s**.
   During 900M–1B it was **4.004 m/s ± 0.025 m/s**, with **99.3%** of logged
   windows between 3.9 and 4.1 m/s. This is strong evidence that the training
   system can sustain a safe operating envelope for a billion transitions.
2. **It is not yet evidence that the learned policy itself is stable at 4 m/s.**
   The run applied `ego_speed_cap_mps=4.0` inside the physics kernel and also
   lowered `reset_speed_max_mps` from 7 to 4 m/s. Its logged
   `action/throttle_max` still reached 1.0. The cap can therefore create the
   observed plateau even if the actor continues to request full throttle.
   There is no recorded deterministic evaluation with the cap removed.
3. **The uncapped matched control was faster and nearly as durable late in
   training.** Its final 50M averaged **5.158 m/s**, **227 s** lifespan, and
   14.5 OOB terminations per logging window. The capped run averaged
   **4.004 m/s**, **268 s**, and 16.5 OOB terminations. The governor buys a much
   tighter speed distribution, not a clear late-training safety advantage.
4. **A 4 m/s plateau is not a dynamics or model-capacity ceiling.** Uncapped
   runs reached **5.28 m/s** over 550–600M (`pcplus600`) and the champion reached
   **5.15 m/s** over 150–200M. Earlier 3.5–4.5 m/s plateaus at 300M were often
   horizon artifacts.
5. **Reward/termination semantics and horizon matter more than marginal reward
   coefficients.** A preregistered 2×2 showed that the recoverable boundary
   rule and legacy boundary-cost shape work as a package. A high-speed progress
   multiplier did not improve the 1B endpoint and sharply reduced durability.
6. **The current stack matches several GT Sophy fundamentals but not its
   training distribution.** Both use 10 Hz actions, 7-step returns,
   \(\gamma=0.9896\), 32 quantiles, and \(\alpha=0.01\). F1TENTH uses a smaller
   3M replay, a lower critic learning rate, a recurrent LiDAR actor, and mostly
   rolling self-play. GT Sophy emphasized mixed scenarios, curated historical
   policies, slower built-in opponents, and specialized skill scenarios.

**Recommendation:** retain 4 m/s as a curriculum/safety tool, not as the final
policy target. First run a replicated cap/reset factorial with cap-free
evaluation. Then test a 4→5→uncapped curriculum against an always-uncapped
control using a fixed evaluation suite. Do not select QR-SAC versus PPO from the
current data: the PPO runs are too new, short, and operationally confounded.

## Evidence and method

### Sources

- Authenticated W&B API access was available through the existing local
  credential setup. The project
  [`squidtoon99-ut-dallas/f1tenth-genesis`](https://wandb.ai/squidtoon99-ut-dallas/f1tenth-genesis)
  contained **359 runs** at query time; 42 names/tags matched the principal
  speed, curriculum, opponent, and long-horizon experiment families.
- W&B run identity, state, summary, tags, and resolved configuration were
  queried through the official `wandb.Api`. Exact trajectories were computed
  from the locally synced W&B `output.log`/run logs, using the repository's
  restart-filtering utilities. This avoids interpolation from sampled charts.
- Resolved `config.json` snapshots, config patches, preregistrations, experiment
  logs, reward/kernel code, and trainer telemetry were inspected locally.
- The primary literature source was Wurman et al.,
  [“Outracing champion Gran Turismo drivers with deep reinforcement learning”](https://doi.org/10.1038/s41586-021-04357-7),
  *Nature* 602, 223–228 (2022), plus the publisher-hosted
  [Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-04357-7/MediaObjects/41586_2021_4357_MOESM1_ESM.pdf).

### Quantitative protocol

- “Window” means all 51,200-transition log records in the preceding 50M
  transitions. Standard deviations describe telemetry variation across those
  records, not seed uncertainty.
- `env/speed_xy` is the mean speed across active vector environments during a
  logging interval. It is not lap speed, deterministic evaluation speed, or a
  confidence interval over policies.
- Lifespan and termination counts are reported alongside speed to avoid
  rewarding fast policies that immediately leave the track.
- Comparisons were restricted to runs with known resolved configs and aligned
  transition budgets. Causal language is reserved for preregistered matched
  comparisons; single-run findings are labelled accordingly.

### Data-quality cautions

- The 4 m/s run has **one seed (42)**. It changed two fields relative to the
  matched control: `ego_speed_cap_mps` and `reset_speed_max_mps`.
- Its source snapshot records a dirty training tree. The committed changes
  between the control and cap run heads were run-directory/logging guards, not
  reward, dynamics, or learner changes, but uncommitted provenance cannot be
  reconstructed completely.
- The control's resolved config was reconstructed after an accidental overwrite.
  Its W&B summary and terminal local log agree, and the recorded live process
  arguments establish its main settings, but this is weaker provenance than an
  untouched snapshot.
- Some historical experiment notes contain interim conclusions later superseded
  by longer runs. This report uses final/addendum results where available.
- Training speed is survival- and reset-distribution-weighted. A policy that
  survives longer contributes a different state distribution from one that
  terminates quickly.

## W&B findings

### The 4 m/s consistency finding

[`ego-cap4-noedge-1b-a001`](https://wandb.ai/squidtoon99-ut-dallas/f1tenth-genesis/runs/ego-cap4-noedge-1b-a001)
completed 1B transitions with no non-finite observations, rewards, or states.

| Window | Mean speed | SD | Range | Mean lifespan | Mean OOB/window |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50–100M | 3.430 m/s | 0.351 | 2.104–3.980 | 146 s | 80.1 |
| 150–200M | 3.902 | 0.080 | 3.657–4.015 | 345 s | 1.0 |
| 350–400M | 3.970 | 0.032 | 3.857–4.030 | 343 s | 1.2 |
| 550–600M | 3.965 | 0.040 | 3.787–4.018 | 341 s | 0.9 |
| 950M–1B | **4.004** | **0.026** | **3.854–4.047** | 268 s | 16.5 |

Across 100M–1B, 82.8% of records were within 3.9–4.1 m/s. Across 900M–1B,
that rose to 99.3%. The late lifespan decline and OOB increase show that speed
stability alone does not imply invariant driving quality.

The closest control,
[`progab-control-a001`](https://wandb.ai/squidtoon99-ut-dallas/f1tenth-genesis/runs/progab-control-a001),
used the same algorithm, seed, 1,024 environments, self-play settings, reward,
model, horizon, and domain randomization. Its resolved config differs only at:

| Field | 4 m/s run | Control |
| --- | ---: | ---: |
| `env.ego_speed_cap_mps` | 4.0 | absent/uncapped |
| `env.reset_speed_max_mps` | 4.0 | 7.0 |

The control's 950M–1B mean was **5.158 ± 0.114 m/s** with **227 s** lifespan.
Thus:

- **Measured:** the capped recipe tightly controls the training state
  distribution around 4 m/s and can maintain it for a long horizon.
- **Supported inference:** the speed governor is the dominant direct cause of
  the narrow band. It replaces positive longitudinal effort with braking/coast
  effort when forward speed is at the cap.
- **Not established:** the actor learned a transferable 4 m/s pace, that 4 m/s
  is an optimal curriculum threshold, or that lowering reset speed contributed
  materially.

This makes the discovery important for **curriculum and safe exploration**, not
evidence for choosing 4 m/s as the future model's terminal operating point.

### Long horizon changes the verdict

| Run | Configuration | Window | Speed | Lifespan |
| --- | --- | ---: | ---: | ---: |
| [`pcplus600`](https://wandb.ai/squidtoon99-ut-dallas/f1tenth-genesis/runs/pcplus600) | uncapped QR-SAC, self-play | 550–600M | **5.281 ± 0.087** | 319 s |
| [`642a7a80`](https://wandb.ai/squidtoon99-ut-dallas/f1tenth-genesis/runs/642a7a80) | champion reference | 150–200M | **5.154 ± 0.226** | 141 s |
| [`pcplus2b-a001`](https://wandb.ai/squidtoon99-ut-dallas/f1tenth-genesis/runs/pcplus2b-a001) | uncapped QR-SAC, self-play | 950M–1B | **5.158 ± 0.114** | 227 s |
| same | same | 1.95–2.0B | 5.026 ± 0.117 | **14 s** |

The last row is a warning against treating more transitions as monotonically
beneficial. Speed remained high while durability collapsed. Policy selection
must use held-out rollouts and Pareto criteria, as GT Sophy did, rather than the
last checkpoint.

### Reward and termination evidence

The preregistered causal 2×2 compared reward/termination packages and opponent
types at 18M:

| Reward/termination | Opponent | Speed |
| --- | --- | ---: |
| legacy recoverable-boundary package | fixed champion | **4.112 m/s** |
| Lee first-contact package | fixed champion | 1.677 |
| legacy recoverable-boundary package | self-play | **5.493** |
| Lee first-contact package | self-play | 0.701 |

Follow-up decomposition found that recoverable termination with the Lee cost
reached only 1.279 m/s, while legacy cost with first-contact termination fell
from 2.477 m/s at 18M to 1.503 at 25M. The evidence supports an interaction:
recoverability and cost shape work together. It does not support attributing
the low-speed regime to steering authority, opponents, or one coefficient.

Halving the OOB cost improved one self-play seed by 0.56 m/s after 200M but did
not replicate on seed 7 or fixed-opponent arms. That lever is seed-sensitive and
should not be promoted without further replication.

### High-speed progress shaping did not solve the problem

[`progab-super-noise-1b-a005`](https://wandb.ai/squidtoon99-ut-dallas/f1tenth-genesis/runs/progab-super-noise-1b-a005)
tripled marginal progress reward from 5 to 8 m/s. At 950M–1B it averaged
**4.938 m/s** and **34 s** lifespan, versus the linear control's
**5.158 m/s** and **227 s**. The treatment was slower and much less durable.
The result argues against another unreplicated “reward harder above threshold”
sweep.

### Parallelism helped early, then degraded

[`numenv4096-600m-a001`](https://wandb.ai/squidtoon99-ut-dallas/f1tenth-genesis/runs/numenv4096-600m-a001)
reached 4.817 m/s and 302 s at 150–200M, but fell to 4.539 m/s and 91 s at
550–600M. The earlier 8,192-env arm was worse with the fixed 3M replay,
consistent with insufficient per-environment sequence depth. More parallel
environments are not a monotonic model-quality lever.

## Configuration analysis

### What matches GT Sophy

| Item | GT Sophy | F1TENTH QR-SAC |
| --- | ---: | ---: |
| Action rate | 10 Hz | 10 Hz |
| Action dimensions | longitudinal + steering | longitudinal + steering |
| Discount | 0.9896 | 0.9896 |
| N-step return | 7 | 7 |
| Quantiles | 32 | 32 |
| Entropy coefficient | 0.01 | 0.01 |
| Actor learning rate | 2.5e-5 | 2.5e-5 |
| Batch size | 1,024 | 1,024 |
| Track horizon | ~6 s, 60 map points | ~6 s, 60 future points |

The real-time discount half-life is about **6.63 s** at 10 Hz; the 7-step
backup spans **0.7 s**. A 4 m/s cap therefore does not change the configured
temporal horizon. It changes the distance covered within it: approximately
26.5 m per discount half-life instead of 34.2 m at 5.16 m/s. The observation's
6 s track horizon similarly covers about 24 m at 4 m/s versus 31 m at
5.16 m/s. Because the representation is time-based, its point density in metres
changes with speed; this is a plausible curriculum effect, not a horizon change.

### Material differences from the paper

| Item | GT Sophy | F1TENTH current/default |
| --- | --- | --- |
| Replay capacity | 10M across 8–10 tables | 3M single replay |
| Minimum replay before learning | 40k | 200k |
| Critic learning rate | 5e-5 | 2.5e-5 |
| Network | 4×2,048 MLP, policy dropout 0.1 | LiDAR CNN + GRU + 3×1,024 heads |
| Observation | privileged car/map state | deployable LiDAR/proprioception actor; privileged critic |
| Opponents | curated old policies, PID, built-in AI | rolling self-play, scripted/policy mix |
| Scenario curriculum | weighted full-track and specialized mistake/skill tasks | full-track randomized resets |
| Warm-up | 20 cars, random actions, 0–60 mph launch | randomized 1–7 m/s resets, 10% stationary |
| Selection | Pareto filters + skill tests + human review | training telemetry/checkpoints; periodic eval default off |

The paper directly reports that straightforward self-play was inadequate
against human imprecision and used a mixed opponent population instead. It also
reports five-seed ablations, 7–12 days of traffic training, and policy selection
on lap time, off-course, collision, skill tests, and races. Those are more
consequential gaps than the exact 4 m/s value.

GT Sophy did **not** report a sequential speed curriculum. Its full-track,
grid-start, slipstream, chicane, and mistake-recovery scenarios remained in a
simultaneous weighted mixture, backed by separate replay tables and stratified
sampling. The supplement's 0–60 mph random-action warm-up populated replay
before learning; it was not progressive speed training. The staged speed-cap
experiment proposed below is therefore a F1TENTH-specific hypothesis motivated
by the local 4 m/s run, not a reproduction of the paper.

## Prioritized recommendations

1. **Establish cap-free evaluation before changing the model.** Every training
   checkpoint should be assessed with `ego_speed_cap_mps=0`, fixed seeds,
   nominal and randomized dynamics, solo laps, fixed opponents, and a curated
   traffic suite. Report lap completion, lap time, speed, OOB/contact exposure,
   collision rate, and tail failure rate.
2. **Treat 4 m/s as a curriculum stage.** Its value is reducing state-distribution
   variance while the actor learns steering and recovery. It should earn
   promotion only if capped pretraining improves later uncapped evaluation.
3. **Prefer staged speed release over superlinear reward.** The 1B reward-shaping
   arm lost durability without increasing terminal speed. A governor changes
   feasibility directly and is easier to interpret than changing reward scale.
4. **Build a Sophy-like scenario mixture.** Keep solo/time-trial data, curated
   fixed opponents of several speeds/styles, traffic starts, corner/recovery
   scenarios, and a bounded amount of self-play in explicit replay strata.
5. **Use long enough horizons and checkpoint selection.** The 300M 4 m/s band was
   not converged; 600M was often required. Conversely, 2B could preserve speed
   while destroying lifespan. Stop and select on held-out Pareto metrics, not
   final training speed.
6. **Do not choose PPO yet.** Current PPO runs are hours old, include crashes,
   differ in environment count, and have not reached comparable budgets.
   Compare algorithms only after the evaluation protocol and matched data
   budget are fixed.

## Proposed experiments and success criteria

### P0 — Isolate the 4 m/s discovery

Run a 2×2 factorial:

- speed cap: 4 m/s versus uncapped;
- reset maximum: 4 m/s versus 7 m/s;
- at least 3 seeds, identical code/GPU/config, 600M transitions;
- checkpoints evaluated both with the training cap and with no cap.

**Primary success:** capped training improves uncapped held-out lap completion
or collision-free completion by at least 10% relative, while uncapped speed is
not more than 0.2 m/s slower.  
**Transfer success:** after cap removal, mean speed exceeds 4.5 m/s by 100M
additional transitions with lifespan at least 200 s and no rise in tail OOB
rate.  
**Refutation:** the apparent benefit disappears when evaluated uncapped, or is
fully explained by `reset_speed_max_mps`.

### P1 — 4→5→uncapped curriculum

Compare always-uncapped, always-4, and staged 4→5→uncapped arms. Release on
held-out competence gates, not fixed training speed: lap completion, boundary
exposure, and steering saturation.

**Success:** final uncapped evaluation reaches at least **5.2 m/s**, at least
**200 s** lifespan, and at least **80%** clean completion across seeds, with
lower between-seed variance than always-uncapped.  
**Abort:** speed rises while clean completion drops by more than 10 percentage
points.

### P2 — Sophy-style scenario and opponent mixture

Compare rolling self-play with a frozen, versioned mixture of solo tasks,
scripted slow/early-braking opponents, curated historical policies, dense
traffic, and targeted recovery/corner scenarios. Hold learner and total samples
fixed.

**Success:** at least 20% lower collision/OOB rate on held-out opponent styles
with no more than 0.1 m/s solo-speed loss; lower run-to-run oscillation after
400M.  
**Diagnostic:** log replay composition and performance by scenario, not only
global means.

### P3 — Paper hyperparameter ablation

After P0/P2, test the paper's 5e-5 critic learning rate and a feasible replay
increase independently. Keep per-environment sequence depth explicit.

**Success:** reach the 5.0 m/s + 200 s regime at least 25% earlier in transitions
without higher critic divergence or late durability decay.  
**Do not combine** critic learning rate, replay size, and environment count in
one arm.

### P4 — QR-SAC versus PPO

Use the same actor, scenario sampler, 1,024 environments, seed set, transition
budget, and evaluation checkpoints.

**Success:** compare sample efficiency to the first 80%-clean 5.0 m/s policy,
final Pareto frontier, seed variance, and wall-clock cost. No conclusion should
be based on current sub-100M PPO runs.

## Risks and explicit unknowns

- **Governor dependence:** the capped actor may rely on intervention and fail
  immediately when uncapped.
- **Reset confound:** cap and reset maximum changed together.
- **Single-seed evidence:** telemetry stability is not reproducibility.
- **No independent evaluation:** most evidence is from stochastic training
  rollouts with changing opponents.
- **No real-car evidence:** simulator stability does not establish VESC,
  localization, LiDAR, tyre, or latency robustness on hardware.
- **Opponent health:** low late opponent speed/presence in some self-play runs
  can make apparent racing quality closer to solo driving.
- **Provenance:** several historical runs came from dirty trees, reconstructed
  configs, restarts, or remote overlays.
- **Metric ambiguity:** mean vector speed can reward reset states and survival
  distribution differently from clean lap time.
- **Observation limits:** no matched ablation establishes whether LiDAR/GRU
  partial observability is responsible for the ~3× sample-efficiency gap to the
  champion.
- **Action limits:** steering authority binds geometrically at the tightest
  centreline point, but the champion exceeded 5 m/s with the same limits; it is
  not the main explanation of the 4 m/s regime.

## Sources and reproducibility pointers

1. Wurman, P. R. et al. (2022),
   [Nature article](https://doi.org/10.1038/s41586-021-04357-7) and
   [publisher supplement](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-04357-7/MediaObjects/41586_2021_4357_MOESM1_ESM.pdf).
2. W&B project:
   [`f1tenth-genesis`](https://wandb.ai/squidtoon99-ut-dallas/f1tenth-genesis);
   individual run links are embedded above.
3. Local resolved data:
   `training/outputs/runs/<run-id>/{config.json,run.log}`.
4. Local experiment records:
   `training/outputs/experiments/long-horizon-promotion/EXPERIMENT_LOG.md`,
   `training/outputs/experiments/causal-2x2/EXPERIMENT_LOG.md`,
   `training/outputs/experiments/oob-tolerance/EXPERIMENT_LOG.md`, and
   `training/outputs/experiments/long-horizon-2b-sensors/EXPERIMENT_LOG.md`.
5. Implementation:
   `training/config.py`, `training/standalone_trainer.py`,
   `training/f1tenth_env/kernel.py`, `training/f1tenth_env/rewards.py`, and
   `training/qrsac/`.
