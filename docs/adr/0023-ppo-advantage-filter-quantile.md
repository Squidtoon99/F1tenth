# 0023 — PPO advantage filter: per-rollout quantile threshold

- Status: Accepted
- Date: 2026-08-02

## Context

[0022](0022-ppo-advantage-filtering.md) adopted Gigaflow-style advantage
filtering with a fixed `kappa` times an EWMA of each rollout's `max|A|`. GPU
measurement (PPO, 2048 envs, 128-step rollouts, 50 updates, RTX 4080) showed
realized discard is not stable across training phases:

- `kappa = 0.001`: ~1.7% discard early, ~3.1% late.
- `kappa = 0.002`: ~2.9% early, ~9.5% late (peak 14.8%).
- `kappa = 0.005`: ~6.0% early, ~41.0% late.

Root cause: the threshold tracks EWMA `max|A|` (early ~165, late ~90) while the
bulk of |A| moves independently (median ~25 early → ~0.65 late; p90 ~12, p99 ~33).
Correlation between EWMA max|A| and discard rate at `kappa = 0.002` was
r = −0.74.

Concurrent work adds infinite-horizon bootstrapping for pure episode timeouts in
GAE (ADR-0021 item 6). Filtering still applies to the post-GAE advantage tensor
without altering timeout semantics.

## Decision

1. Replace the fixed-κ / EWMA rule with a per-rollout quantile target: drop the
   bottom `N%` of transitions by |A| within each rollout. Default
   `advantage_filter_discard_fraction = 0.05` (95% retention). Keep
   `advantage_filter_enabled` default-on.
   *Why:* Retention tracks the within-rollout |A| distribution, so discard rate
   stays near the target as value calibration shifts; remove kappa/EWMA config
   rather than carry two parameterizations.
2. Tie handling at the quantile boundary: `num_drop = floor(n * discard_fraction)`;
   stable-argsort |A| and drop exactly `num_drop` indices (never more than the
   target fraction). All-zero or tie-heavy rollouts therefore drop at most
   `num_drop` transitions; survivors are never empty at default 5% unless
   `discard_fraction ≥ 1`.
   *Why:* A strict `|A| < eta` cut can over-drop when many values tie at the
   boundary; rank-based selection with stable tie-breaking is deterministic and
   respects the budget.
3. Keep structural behavior from 0022: filter raw GAE advantages before
   standardization; standardize only survivors; zero-weight non-survivors in both
   policy and value losses; skip the optimizer step when every transition is
   filtered. Log discard rate, retained count, and the realized cutoff (largest
   |A| among dropped transitions) as `advantage_filter_eta`.
   *Why:* Matches Gigaflow's estimator shape and preserves disabled-run parity.
4. No checkpoint change: `A_max_ewma` was never persisted on policy artifacts or
   protocol state (0022 item 3). Removing trainer EWMA state needs no resume
   compatibility shim.

## Consequences

- Default discard is ~5% throughout training rather than phase-dependent drift.
- QR-SAC is unchanged; advantage filtering applies only where GAE exists.
- Filtering composes with timeout bootstrapping at distinct code sites.
- Disabling the flag restores prior PPO loss math exactly (diagnostic metrics
  report zero discard).
