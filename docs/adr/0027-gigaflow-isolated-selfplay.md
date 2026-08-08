# 0027 — Isolate Gigaflow self-play in a root-level package

- Status: Accepted
- Date: 2026-08-03

## Context

The existing QR-SAC stack under `training/` and the shared `libs/f1tenth_policy`
actor are built around 1v1 / frozen-opponent workflows, Gym-style env wrapping,
and a different learner. The Gigaflow-inspired program needs multi-track N-car
worlds, shared-current-policy PPO, private reward conditioning, a training-only
centralized critic, and compact GPU-resident collection. Coupling that work to
`training/` would force incompatible abstractions into a working stack and risk
deploy-contract drift through accidental shared imports.

## Decision

1. Put the complete self-play implementation in a new root-level `gigaflow/`
   project with its own package metadata, configs, CLI, tests, benchmarks, and
   design specification (`gigaflow/DESIGN.md`).
2. Keep the runtime independent of `training/`, `f1tenth_env`, `f1tenth_sim`,
   `qrsac`, `libs/f1tenth_policy`, and `libs/f1tenth_contract`. Reimplement
   required formulas and the 1097-D sensor / continuous force+delta-steer action
   contract inside `gigaflow/`; use the current code only as a reference and
   parity-fixture source.
3. Use a small set of flat modules (`config`, `tracks`, `buffers`, `kernels`,
   `model`, `critic`, `ppo`, `trainer`, `artifacts`, `evaluation`, `cli`) with
   typed seams so later phases can proceed in parallel without framework
   wrappers or per-agent object graphs.
4. Validate a versioned experiment config at startup for layout parity, control
   cadence, PPO constraints, and memory budget before kernels are expanded.

## Consequences

- Self-play can evolve without destabilizing the current QR-SAC training path.
- Deploy parity becomes an explicit local constant set plus future export tests,
  not an import of `f1tenth_policy`.
- Track assets remain prepared/cached outside git with license/attribution
  handled by the track-atlas CLI (GPL-3 upstream racetracks are not vendored).
- Parallel implementation phases must respect the interfaces locked in
  `gigaflow/DESIGN.md`; end-to-end training is owned by later phases.
