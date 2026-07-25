# 0016 — Revert default steering-history scale to paper-literal 5.0

- Status: Accepted
- Date: 2026-07-25
- Supersedes: [0015](0015-progress-normalized-steering-history.md)

## Context

ADR-0015 set the canonical default `reward_scales.steering_history` to **0.5**
(\(\approx 0.1\times\) Lee \(\lambda^h\)) on the assumption that F1TENTH's
smaller per-step progress made the paper-literal \(\lambda^h = 5.0\) weigh
10–20× harder relative to progress. That change was not validated with a clean
legacy-ladder A/B before landing.

On 2026-07-25 a preregistered two-arm experiment (`steerhist5-ab001` vs
`steerhist05-ab001`) isolated only `steering_history` on the legacy ladder
(fixed champion, seed 42, 25M transitions). At the 18M peak region,
\(\lambda^h = 5.0\) reached **4.18 m/s** versus **3.76 m/s** for
\(\lambda^h = 0.5\) (+0.43 m/s). Lee and Vasco publish \(\lambda^h = 5.0\) at
the same ±3° delta-steering action scale this stack uses; no rescaling
justification remains.

## Decision

1. Revert the canonical default `reward_scales.steering_history` to **5.0**.
   *Why:* Measured +0.43 m/s at 18M on a clean legacy-ladder A/B; matches
   published \(\lambda^h\) at our action scale.
2. Supersede ADR-0015; do not rewrite it.
   *Why:* Preserve the record of why 0.5 was tried and what evidence reversed it.
3. Do not rewrite frozen run configs or interrupt in-flight trainers.
   *Why:* Those runs retain their resolved snapshot.

## Consequences

New resolved configs inherit `5.0`. Runs that already snapshotted `0.5`
continue unchanged. Reward totals are not directly comparable across the two
defaults. The steering-history formula, constants, and `steering_change`
(\(0.5\)) are unchanged.
