# 0022 — PPO Gigaflow-style advantage filtering

- Status: Superseded by [0023](0023-ppo-advantage-filter-quantile.md)
- Date: 2026-08-02

## Context

Gigaflow ("Robust Autonomy Emerges from Self-Play", arXiv 2502.03349) filters
on-policy PPO transitions whose raw GAE advantage magnitude falls below an
adaptive threshold before both actor and critic updates. The threshold is
`eta = kappa * A_max_ewma`, where `A_max_ewma` is an exponential moving average
of each rollout's `max|A|` (decay `beta = 0.25`) and Gigaflow uses `kappa =
0.01`. At their rollout scale this discards roughly 80% of transitions.

Our PPO path uses comparatively small on-policy batches and a fixed transition
budget. Discarding most transitions would waste scarce samples. We still want
the mechanism available for A/B testing because low-magnitude advantages can
dominate PPO gradient noise when returns are heavy-tailed.

Concurrent work adds infinite-horizon bootstrapping for pure episode timeouts in
GAE (ADR-0021 item 6). That changes how advantages are computed upstream; filtering
must apply to the post-GAE advantage tensor without altering timeout semantics.

## Decision

1. After GAE, drop transitions with `|A| < eta` from both policy and value
   losses when `ppo.advantage_filter_enabled` is true (default). Compute `eta`
   from raw advantages; standardize only the retained set before the clipped
   losses.
   *Why:* Gigaflow's threshold is defined on unnormalized GAE magnitudes; filtering
   then re-standardizing survivors matches their estimator and keeps disabled
   runs identical to pre-filter behavior.
2. Default `advantage_filter_kappa = 0.001` and expose `advantage_filter_kappa`
   and `advantage_filter_ewma_decay` in `config.ppo` for sweeps without code
   edits.
   *Why:* Measured on GPU (50 PPO updates, 2048 envs, IV_2026_SIM solo) at
   `kappa = 0.002` the realized discard rate rises from ~3% early to ~10% late
   (mean 5.9%, peak 15%) as |A| median collapses while EWMA max|A| stays
   elevated. At `kappa = 0.001` late discard is ~3% (~97% retention). Our
   sample budget is the limiting factor, so we default to the smaller kappa
   that keeps discard in the low single digits throughout training.
3. Persist `A_max_ewma` on `PPOTrainer` across rollouts. Log discard rate,
   retained count, and `eta` through existing PPO metrics.
   *Why:* policy artifacts remain actor-only and runs do not resume; no
   checkpoint change is required. Observability is needed for the planned A/B.
4. When every transition is filtered, skip the optimizer step and return finite
   zero-loss metrics rather than dividing by an empty batch.
   *Why:* early training or all-zero advantages must not produce NaNs.

## Consequences

- PPO updates may use fewer effective transitions; default kappa targets a
  modest discard fraction rather than Gigaflow's majority discard.
- QR-SAC is unchanged; advantage filtering applies only where GAE exists.
- Filtering composes with timeout bootstrapping: both operate on the post-GAE
  advantage tensor at distinct code sites (`compute_gae` vs `PPOTrainer.update`).
- Disabling the flag restores prior PPO loss math exactly (aside from new
  diagnostic metric keys that report zero discard).
