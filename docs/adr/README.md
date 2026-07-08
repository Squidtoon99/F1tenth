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
