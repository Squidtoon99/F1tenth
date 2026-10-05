# Architecture Decision Records (ADRs)

Short, dated records of significant architectural decisions and their rationale.
Add a new numbered file per decision; never rewrite history — supersede instead.
Copy [0000-template.md](0000-template.md) to start a new one.

Write an ADR when a change alters the repo's structure, a public interface or the
observation/action contract, the build/deploy shape, or accepts a non-obvious
trade-off. Routine work is captured by the PR instead (see
[../development.md](../development.md)).

## What's true now

Read this index, then the **Accepted** ADRs if you need the why. Superseded
records stay for the timeline; they are not current policy.

| Now | Source |
| --- | --- |
| One monorepo, one colcon workspace, generic SHA-keyed car image, `develop` trunk | [0001](0001-single-monorepo-and-generic-car-image.md) |
| 390-dim policy obs: 384 base (incl. 8 slip + 4 load) + 6-dim opponent block | [0002](0002-physx-tyre-slip-and-tyre-load-observation.md), [0003](0003-opponent-obs-range-mask.md) |
| Opponent block is unmasked in builders; callers zero it when out of range / undetected | [0003](0003-opponent-obs-range-mask.md) |
| On-car graph: PF localization + algo or RL, maps at `/config/maps/` | [0004](0004-on-car-stack-integration.md) |
| Longitudinal action is force/current only (`policy_format_version` 2); no speed mode | [0006](0006-current-only-vesc-actuation.md) |
| Batched RL physics is NVIDIA Warp; Torch is a committed parity fixture only | [0007](0007-warp-simulator.md) |

## Timeline

Oldest first. Status is the current reading of each record, not a rewrite of it.

- [0000](0000-template.md) — Template (copy this).
- [0001](0001-single-monorepo-and-generic-car-image.md) — **Accepted.** Single
  monorepo, single colcon workspace, generic car image + per-car runtime config.
- [0002](0002-physx-tyre-slip-and-tyre-load-observation.md) — **Accepted**
  (PhysX slip + 4-dim tyre-load at `[380:384)`). Simulation backend later
  superseded by 0007; the slip/load contract still stands.
- [0003](0003-opponent-obs-range-mask.md) — **Accepted.** Drop opponent `present`;
  6-dim block with caller-side range/certainty masking (391 → 390).
- [0004](0004-on-car-stack-integration.md) — **Accepted.** Integrate localization,
  algorithmic drivers, and the generic deploy image.
- [0005](0005-force-mode-vesc-actuation.md) — **Superseded by 0006.** Dual
  speed/force motor mode (historical).
- [0006](0006-current-only-vesc-actuation.md) — **Accepted.** Current/force is the
  only longitudinal semantic; speed-mode removed; `policy_format_version` 2.
- [0007](0007-warp-simulator.md) — **Accepted.** NVIDIA Warp is the batched RL
  simulator; TorchSim is not a runtime backend.
