# 0015 — Progress-normalize steering-history scale

- Status: Superseded by [0016](0016-revert-steering-history-to-paper-literal.md)
- Date: 2026-07-24

## Context

Lee et al. 2025 publish \(\lambda^h = 5.0\) for the steering-history reversal
penalty. The Warp implementation already matches the paper's formula, realized
10 Hz wheel-angle deltas, and constants \(c^s,c^o,c^d\). On F1TENTH, centreline
progress per control step is roughly an order of magnitude smaller than at
GT7 highway speeds, so the same \(\lambda^h\) weighs about \(10\)–\(20\times\)
harder relative to progress. Early run telemetry with \(\lambda^h=5.0\) showed
history often dominating progress until late training.

Already-running trainers load a frozen resolved config snapshot and must keep
their original scale.

## Decision

1. Set the canonical default `reward_scales.steering_history` to **0.5**
   (\(\approx 0.1\times\) Lee \(\lambda^h\)).
   *Why:* Progress-normalized adaptation for F1TENTH; leave the history
   formula, constants, and `steering_change` (\(0.5\)) unchanged.
2. Do not rewrite frozen run configs or interrupt in-flight trainers.
   *Why:* Those runs retain their snapshot (including \(\lambda^h=5.0\)).

## Consequences

New resolved configs inherit `0.5`. Existing runs that already snapshotted
`5.0` continue unchanged. Reward totals are not directly comparable across the
two defaults. Wall-event scale (ADR 0014) and other Lee reward terms are
unaffected.
