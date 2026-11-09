# Architecture Decision Records (ADRs)

Short, dated records of significant architectural decisions and their rationale.
Add a new numbered file per decision; never rewrite history — supersede instead.
Copy [0000-template.md](0000-template.md) to start a new one.

Write an ADR when a change alters the repo's structure, a public interface or the
observation/action contract, the build/deploy shape, or accepts a non-obvious
trade-off. Routine work is captured by the PR instead (see
[../development.md](../development.md)).

## Timeline

These records are the path to the current implementation. Accepted decisions
still hold unless a later ADR supersedes them.

**Current stack:** one monorepo, 390-dim observation (384 base + 6-dim opponent
block), NVIDIA Warp batched training sim, current/force VESC actuation
(`policy_format_version` 2).

| ADR | Decision | Status |
| --- | --- | --- |
| [0000](0000-template.md) | Template | — |
| [0001](0001-single-monorepo-and-generic-car-image.md) | Single monorepo, one colcon workspace, generic car image + per-car runtime config | Accepted |
| [0002](0002-physx-tyre-slip-and-tyre-load-observation.md) | PhysX tyre slip + 4-dim tyre-load block (380 → 384). TorchSim as sim source of truth later superseded by 0007 | Accepted (layout) |
| [0003](0003-opponent-obs-range-mask.md) | Drop opponent `present`; 6-dim block with caller-side masking (391 → 390) | Accepted |
| [0004](0004-on-car-stack-integration.md) | On-car localization, algo drivers, and deploy image in the monorepo | Accepted |
| [0005](0005-force-mode-vesc-actuation.md) | Dual speed/force VESC actuation | **Superseded by 0006** |
| [0006](0006-current-only-vesc-actuation.md) | Current/force is the only longitudinal action; speed-mode removed | Accepted |
| [0007](0007-warp-simulator.md) | NVIDIA Warp is the batched RL simulator (replaces TorchSim) | Accepted |
