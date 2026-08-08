# 0029 — Gigaflow transactional actor updates

- Status: Accepted
- Date: 2026-08-06

## Context

Long-horizon PPO runs exposed two independent failures: a fixed entropy bonus
overwhelmed the clipped surrogate near update 2,500, and rare Adam steps later
moved the actor by KL values as high as 152.8. The joint actor/critic optimizer
made an actor rollback discard valid critic learning, while the existing KL
check measured the policy before the optimizer step and could not reject damage.

## Decision

Use a resume-safe half-cosine entropy schedule and separate Adam optimizers and
cosine schedulers for actor and critic. Each PPO update snapshots actor
parameters and actor optimizer state. Candidate actor steps are rescored on the
same recurrent minibatch; crossing the soft KL limit stops further actor steps,
and crossing the hard limit restores the update-start actor transaction. A
mandatory retained-rollout rescore applies the same hard limit to the final
actor state. Critic steps remain committed. Rejection backs off an actor-only
learning-rate safety multiplier, and repeated rejection stops training.

The learner resume state and trainer checkpoint versions are bumped. Older
single-optimizer checkpoints remain valid evaluation artifacts but are rejected
for training resume with an explicit version error.

## Consequences

- Actor trust-region rejection no longer destroys critic work.
- Accepted actor state is always checked after its final optimizer step.
- Checkpoints now persist two optimizers, two schedulers, actor safety state,
  and rollback history.
- The six-hour run can test the entropy-collapse horizon, but a fresh run must
  still pass 7,000 updates before long-horizon robustness is claimed.
