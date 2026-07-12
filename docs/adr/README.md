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
  current/brake commands; speed-mode remains the default until validated.
