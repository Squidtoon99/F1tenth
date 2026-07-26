# Architecture Decision Records (ADRs)

Short, dated records of significant architectural decisions and their rationale.
Add a new numbered file per decision; never rewrite history — supersede instead.
Copy [0000-template.md](0000-template.md) to start a new one.

Write an ADR when a change alters the repo's structure, a public interface or the
observation/action contract, the build/deploy shape, or accepts a non-obvious
trade-off. Routine work is captured by the PR instead (see
[../development.md](../development.md)).

- [0000](0000-template.md) — Template (copy this).
- [0001](0001-single-monorepo-and-generic-car-image.md) — Single monorepo, single
  colcon workspace, and a generic car image with per-car runtime config.
- [0002](0002-physx-tyre-slip-and-tyre-load-observation.md) — PhysX-grounded tyre
  slip and a per-wheel tyre-load observation block (380 → 384 layout change).
- [0003](0003-opponent-obs-range-mask.md) — Remove opponent `present` channel;
  6-dim opponent block with caller-side range/certainty masking (391 → 390).
- [0004](0004-on-car-stack-integration.md) — Integrate the on-car localization,
  algorithmic drivers, and deploy image from `shereef@f1tenth`.
- [0005](0005-force-mode-vesc-actuation.md) — Force-mode VESC actuation via
  current/brake commands (superseded).
- [0006](0006-current-only-vesc-actuation.md) — Current/force is the only
  longitudinal action semantic; speed-mode removed; policy_format_version 2.
- [0007](0007-warp-simulator.md) — Warp physics backend for large-batch training.
- [0008](0008-sensor-policy-runtime-image.md) — Dedicated no-localization
  sensor-policy launch and JetPack iGPU runtime image, isolated from racing.
- [0009](0009-opponent-relative-acceleration.md) — Privileged opponent relative
  acceleration in an 8-dim block (390 → 392).
- [0010](0010-ten-hz-policy-cadence.md) — Align training, rewards, and deployment
  policy decisions at 10 Hz.
- [0011](0011-delta-steering-and-boundary-policy.md) — Scope three-degree
  steering deltas and full-footprint boundary behavior to the sensor experiment
  (partially superseded by 0012 for the sensor stack).
- [0012](0012-lee-core-and-fixed-opponents.md) — Canonical Lee/ADR-0011 sensor
  core, fixed champion opponents, and shared `f1tenth_policy` module.
- [0013](0013-first-footprint-wall-contact.md) — Canonical first-footprint wall
  termination with one linear-speed Lee barrier reward.
- [0014](0014-paper-literal-wall-event-scale.md) — Paper-literal wall-event
  reward without control-period scaling.
- [0015](0015-progress-normalized-steering-history.md) — Progress-normalize
  default `steering_history` to 0.5 (\(\approx 0.1\times\) Lee \(\lambda^h\))
  (superseded by 0016).
- [0016](0016-revert-steering-history-to-paper-literal.md) — Revert default
  `steering_history` to paper-literal 5.0 after clean legacy-ladder A/B (+0.43
  m/s @ 18M).
- [0017](0017-tightened-opponent-spawn-gaps.md) — Tighten default opponent spawn
  gap ceiling 80 → 58 m for ~70% presence at 5–6 m/s without near-total coupling.
