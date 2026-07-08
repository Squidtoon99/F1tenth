# 0001 — Single monorepo, one colcon workspace, generic car image

- Status: Accepted
- Date: 2026-07-07

## Context

We are building a 1/10th-scale autonomous racing stack with two racing approaches
(reinforcement-learning and algorithmic) that share perception, mapping, and
localization. A small team needs to work on different parts in parallel, train RL
models locally and on HPC, and deploy to an on-car NVIDIA Jetson. The code is
developed primarily in Docker and lives in a public repository.

## Decisions

1. **One monorepo, one colcon workspace.** All ROS 2 packages live under `src/`,
   grouped by domain and code-owned per team. Shared cross-language libraries live
   under `libs/`. RL training is pure Python under `training/` and is never built by
   colcon nor shipped in the car image.

   *Why:* atomic cross-cutting changes (e.g. observation contract + inference node +
   reward) in a single PR; shared modules reused without cross-repo version juggling;
   one source of truth for what becomes the car image. The usual monorepo downside
   (artifact bloat) is avoided by gitignoring weights/rosbags/outputs/maps/build
   trees.

2. **Shared observation/action contract with no rebuild churn.** The layout lives
   once in `libs/f1tenth_contract` (dual pip + ament package): editable-installed for
   training and `colcon --symlink-install` for the inference node, so iterating on the
   observation space needs no rebuild. The C++ side keeps a mirror validated by a
   parity test.

3. **Vendored vehicle driver layer.** Upstream F1TENTH driver packages are copied in
   (frozen snapshot, SHAs recorded), not submoduled — we only need the current
   version and want to edit freely. Generic deps come from rosdep/apt.

4. **Generic car image + per-car runtime config.** Each `develop` commit builds one
   generic, SHA-keyed image (no car config, no weights). Identity/params/map and the
   RL policy are mounted at runtime, so one image serves all cars and a car's identity
   is never overwritten. Snapshots are shared as portable archives (no registry yet).

5. **`develop` is the trunk; releases are tags.** No `main`/`master`, no release
   branch. Feature branches merge into protected `develop`; a release is a git tag on
   `develop` whose exact image is snapshotted.

6. **No GPU CI.** CI does lint + CPU colcon build/test + CPU pytest + a docker-build
   sanity check, path-filtered so ROS and training pipelines are independent. GPU work
   (training) runs on developer machines or HPC.

## Consequences

- Simple toolchain (colcon + Docker, no Bazel) suited to a small team.
- Clear ownership boundaries via `CODEOWNERS` while keeping one buildable tree.
- One duplication is accepted on purpose (the C++ observation mirror), fenced off by a
  parity test.
- Reproducing a deployment means checking out a git SHA/tag and rebuilding — the
  manifest maps SHA -> image -> car.
