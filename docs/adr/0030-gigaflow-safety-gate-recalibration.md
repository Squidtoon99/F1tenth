# 0030 — Gigaflow safety-gate recalibration

- Status: Accepted
- Date: 2026-08-06
- Supersedes: calibration parameters in [0029](0029-gigaflow-transactional-actor-updates.md)

## Context

The first RTX PRO 6000 robust run (`tdmlti3k`) stopped at update 68 after three
actor rollbacks whose final candidate KL was only 0.060 against a hard threshold
of 0.05. Counterfactual replay of prior productive trajectories showed the
behavioral gate would also stop learning early: RTX 4080 at update 200, A100 on a
transient update-1,200 dip that recovered at 1,300, and likely H100 near update
200. Transactional actor rollback and entropy annealing remain valuable; the
stop thresholds and hysteresis were over-sensitive for the new post-step KL
estimator.

## Decision

Keep transactional actor rollback, split optimizers, and the entropy schedule
unchanged. Recalibrate stop policy as follows:

1. **Behavioral gate.** Require two consecutive full-suite passes before
   feasibility is latched. Before feasibility, sentinels run and may freeze
   best-safe actors, but low completion/progress is not a training stop; only
   missing feasibility by update 1,000 stops the run. After feasibility, every
   behavioral stop (including catastrophic-looking failures) requires two
   consecutive failed sentinels.
2. **KL rollback gate.** Raise soft/hard actor KL to 0.05/0.20. Keep immediate
   rollback at the hard threshold. Disable run-stop from rollbacks during a
   200-update warm-up window. After warm-up, stop on two consecutive rollbacks
   or five rollbacks in a rolling 500-update window. Recover actor safety LR by
   doubling the multiplier after 100 rollback-free accepted updates until 1.0.
3. **Persistence.** Gate feasibility, consecutive counters, rollback-free LR
   recovery, and best-safe references are checkpointed in trainer version 8 and
   PPO resume state version 3.

## Consequences

- Early benign KL excursions no longer terminate productive random-init runs.
- Transient behavioral dips after feasibility can recover without stopping.
- Persistent collapse (A100 updates 2,200/2,300) still stops on the second bad
  sentinel.
- Six-hour runs remain insufficient to validate the historical update-6,500 KL
  horizon; a fresh 7,000-update gate is still required for long-horizon proof.
