# Architecture Decision Records (ADRs)

Short, dated records of significant architectural decisions and their rationale.
Add a new numbered file per decision; never rewrite history — supersede instead.
Copy [0000-template.md](0000-template.md) to start a new one.

Write an ADR when a change alters the repo's structure, a public interface or the
observation/action contract, the build/deploy shape, or accepts a non-obvious
trade-off. Routine work is captured by the PR instead (see
[../development.md](../development.md)).

## Current stack

Read in order. Superseded records stay as history; do not treat them as the
live rule.

| Now | ADR | What still holds |
| --- | --- | --- |
| Repo + deploy shape | [0001](0001-single-monorepo-and-generic-car-image.md) | One monorepo, generic SHA-keyed image, per-car overlay |
| Observation layout | [0002](0002-physx-tyre-slip-and-tyre-load-observation.md) + [0003](0003-opponent-obs-range-mask.md) | 384-dim base (slip + load) + 6-dim opponent = **390**; all-zero opponent sentinel |
| On-car composition | [0004](0004-on-car-stack-integration.md) | Vendored PF, `range_libc` submodule, algo in `f1tenth_control` |
| Longitudinal action | [0006](0006-current-only-vesc-actuation.md) (supersedes [0005](0005-force-mode-vesc-actuation.md)) | Current/force only; `policy_format_version >= 2` |
| Training simulator | [0007](0007-warp-simulator.md) (supersedes 0002's TorchSim source of truth) | NVIDIA Warp physics; PyTorch for the learner |

## Index

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
  current/brake commands (superseded by 0006).
- [0006](0006-current-only-vesc-actuation.md) — Current/force is the only
  longitudinal action semantic; speed-mode removed; policy_format_version 2.
- [0007](0007-warp-simulator.md) — Batched NVIDIA Warp training simulator
  (replaces TorchSim).
