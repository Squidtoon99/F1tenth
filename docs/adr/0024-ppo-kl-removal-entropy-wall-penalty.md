# 0024 — Remove PPO KL early-stop, add entropy bonus, reduce wall penalty

- Status: Accepted
- Date: 2026-08-02

## Context

An overnight PPO run (`final-2026-08-02`) failed with a frozen actor. Log analysis
showed the KL early-stop gate (`approx_kl > target_kl`, default 0.02) fired on the
first minibatch of nearly every rollout: measured `approx_kl` settled at median
0.0398 (p95 0.047), so only 2 of 666 logged windows completed a full 128-minibatch
update. The critic kept training while the actor was effectively frozen. Policy
degenerated — throttle pinned at the lower bound, mean speed fell from 3.6 to 1.0
m/s, 78% out-of-bounds and 21% not-moving terminations, mean episode life ~3 s.
Braking to a stop was locally rational because wall-contact penalty dominated
progress reward (~25:1 on average, up to ~120:1 on contact steps).

Gigaflow (arXiv 2502.03349, Table A3 p. 24) uses plain clipped PPO: no KL early
stopping, no KL penalty, no trust region. Update size is governed by clip epsilon
0.2, max grad norm 0.5, 3 epochs, and cosine-decayed LR. It uses an entropy
coefficient of 0.01, which our implementation lacked.

Infinite-horizon timeout bootstrapping ([0021](0021-config-swappable-ppo-training.md))
is now in place, so a modest wall-penalty reduction is justified without removing
the contact signal entirely.

## Decision

1. **Remove the KL early-stop gate and delete `target_kl` from config.** Keep
   logging `approx_kl` as a diagnostic only.
   *Why:* measured KL at the old threshold blocked actor updates on 664/666 windows;
   the reference paper does not use this mechanism.
2. **Add entropy bonus with coefficient 0.01** to the PPO policy loss, masked over
   surviving advantage-filter rows only (same `keep_denom` as policy and value
   losses). Log mean `policy_entropy` per update.
   *Why:* matches Gigaflow Table A3; consistent masking keeps entropy scale aligned
   with the clipped surrogate on the same transition subset.
3. **Reduce `wall_contact_coefficient` from 20.0 to 15.0** (25% decrease, not the
   50% halving previously floated).
   *Why:* timeout bootstrapping reduces the incentive to brake-to-stop at horizon;
   a modest reduction lowers the progress:penalty ratio enough to discourage the
   degenerate stop-at-wall local optimum while preserving strong contact signal.

## Consequences

- Every PPO update now runs all configured epochs and minibatches; `actor_updates`
  equals `minibatches` per rollout. The removed `early_stop` metric is no longer
  logged.
- `policy_entropy` appears in training logs and wandb under `ppo/policy_entropy`.
- Wall-contact step penalty is `-15.0 × sqrt(speed)` instead of `-20.0 × sqrt(speed)`.
- If `approx_kl` grows large during training, we observe it in logs but do not
  automatically halt actor updates — clip epsilon and grad norm remain the sole
  update-size governors, matching the paper.
