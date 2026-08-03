# 0021 — Add config-swappable recurrent PPO training

- Status: Accepted
- Date: 2026-07-31

## Context

The sensor-policy trainer is built around recurrent asymmetric QR-SAC: the actor
uses LiDAR and on-car proprioception, while feed-forward critics use privileged
Frenet state. QR-SAC owns an off-policy trajectory replay, n-step targets, target
critics, and replay-driven update cadence. PPO needs an on-policy rollout,
behavior log probabilities, scalar values, GAE, and multi-epoch sequence updates,
so replacing only the numerical QR-SAC updater would be incorrect.

The vectorized Warp simulator can collect large on-policy batches efficiently.
The deployed actor artifact, evaluation path, and opponent policies must remain
algorithm-independent.

## Decision

1. Select `qrsac` or `ppo` through the run configuration and the existing
   `standalone_trainer.py` entrypoint. The proven QR-SAC collection/update path
   remains unchanged; `PPOTrainer` owns PPO rollout storage and optimization.
   *Why:* adding PPO and mechanically relocating QR-SAC in the same change would
   make behavioral regressions difficult to distinguish from algorithm effects.
2. Keep the artifact-compatible recurrent squashed-Gaussian actor. Add
   fixed-action recurrent log-probability evaluation without changing parameters
   or state-dict keys.
   *Why:* PPO ratios must score the actions produced by the behavior policy, while
   deployment and opponent snapshots must continue to load existing artifacts.
3. Use a feed-forward scalar value critic over the privileged critic observation.
   Train recurrent actor minibatches as complete time sequences grouped by
   environment.
   *Why:* the privileged observation is intended to be Markov; flattening actor
   timesteps would break GRU state and reset semantics.
4. Freeze actor and critic normalization statistics throughout each rollout and
   all PPO epochs, then update them after optimization.
   *Why:* behavior and current log probabilities must be evaluated from identical
   normalized observations.
5. Reset all collection GRU state after each PPO rollout update. Continue to reset
   individual rows on episode termination.
   *Why:* carrying hidden state produced by pre-update recurrent weights into the
   updated policy has undefined semantics. Exact history replay can replace the
   boundary reset if measurements justify the added complexity.
6. On pure episode timeout only, augment PPO step rewards with
   `gamma * V(s_terminal)` using the pre-reset privileged critic observation
   captured before auto-reset; all other `done` causes remain zero-bootstrap
   terminals in GAE.
   *Why:* timeout is an artificial horizon, not task failure; the simulator
   auto-resets before returning the next observation, so bootstrap must use the
   stored terminal critic obs, not the post-reset observation.
7. Refresh and snapshot self-play opponents only at PPO rollout boundaries.
   Keep QR-SAC self-play cadence unchanged.
   *Why:* changing the opponent policy during a rollout mixes environment
   dynamics inside one on-policy batch.
8. Run PPO collection and optimization eagerly at first. Keep policy artifacts
   actor-only and retain their existing format version.

## Consequences

- PPO and QR-SAC can be selected without changing simulation, reward, artifact,
  evaluation, or deployment interfaces.
- PPO allocates no QR-SAC critics or replay buffer; QR-SAC allocates no PPO
  rollout or scalar value critic.
- PPO pauses collection while optimizing each rollout. Asynchronous collection
  is a separate architectural change.
- Rollout-boundary hidden resets limit uninterrupted recurrent context to the
  configured rollout length.
- PPO timeout bootstrapping is gated to pure `term_timeout` rows; QR-SAC still
  treats all `done` flags as terminal for n-step targets.
- By default, each PPO update does not train on every rollout transition: the
  bottom 5% by raw GAE |A| are zero-weighted in both policy and value losses
  ([0023](0023-ppo-advantage-filter-quantile.md)). Fixed-κ filtering
  ([0022](0022-ppo-advantage-filtering.md), superseded by 0023) was replaced
  after measured discard drift across training phases. Set
  `advantage_filter_enabled` false to restore full-rollout updates.
- Quantile PPO is deferred. A later ablation may train quantile values against
  scalar returns and use their mean or a documented risk distortion for GAE, but
  it does not recover QR-SAC's off-policy, twin-Q, or entropy-objective behavior.
