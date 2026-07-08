# Development

How to set up, build, run, and contribute.

## One-command setup

Everything runs in Docker; you don't need ROS 2 on your host.

```bash
./tools/dev-setup.sh     # once per machine: checks docker/buildx, enables cross-arch
./tools/dev.sh up        # build base+dev images, enter a shell with source mounted
```

Inside the dev container the source is mounted at `/ws`:

```bash
./tools/build.sh                 # colcon build everything (--symlink-install)
./tools/build.sh -t control      # build just one module group
```

Teammates can skip rebuilding images by sharing them directly (no registry):

```bash
deploy/scripts/save_dev_images.sh          # producer
deploy/scripts/load_dev_images.sh dev-images.tar.gz   # consumer
```

## Repository shape

See [`../ARCHITECTURE.md`](../ARCHITECTURE.md). In short: one colcon workspace under
`src/`, shared libs under `libs/`, pure-Python RL training under `training/`, and
deploy tooling under `deploy/`.

## Branching model

- **`develop` is the trunk.** There is no `main`/`master`.
- Do work on a branch off `develop`: `feature/<ticket>-<slug>` (or `fix/…`,
  `chore/…`). Keep changes scoped to your module group where possible.
- Open a PR back into `develop`. `develop` is protected: CI (lint + build/test) and
  review must pass. Reviews route by [`../CODEOWNERS`](../CODEOWNERS).
- **Releases are git tags on `develop`**, e.g. `racing-v0.3.0` — not branches. The
  image for that exact commit is snapshotted for the car. See
  [`deployment.md`](deployment.md).

```mermaid
gitGraph
  commit id: "develop"
  branch feature/ABC-raceline
  commit
  commit
  checkout develop
  merge feature/ABC-raceline
  commit tag: "racing-v0.3.0"
```

## Running

- Simulation — evaluate the deployed RL stack in the gym (two containers: the gym
  bridge in the fork image + our nodes in the dev image, sharing a docker bridge
  network; noVNC serves RViz, Foxglove serves at `ws://localhost:8765`):

  ```bash
  # CHECKPOINT_DIR is the host dir holding the trained .pt; CKPT its filename.
  CHECKPOINT_DIR=/abs/path/to/checkpoints CKPT=policy.pt ./tools/sim.sh up
  # ... watch: RViz at http://localhost:8080/vnc.html, or Foxglove -> ws://localhost:8765
  ./tools/sim.sh down
  ```

  `tools/sim.sh` imports the gym (`vcs import sim < sim/f1tenth_gym_ros.repos`),
  builds the agent workspace, launches the bridge-only sim
  (`rl_agent_sim_launch.py`) plus our `bringup_agent_launch.py`, and drives the car
  with the trained policy. The `evaluation` node spawns the car forward-facing along
  the centerline (matching training) and resets it on stuck/out-of-bounds. See
  [`../sim/README.md`](../sim/README.md).

- On the car: vehicle-only shakedown then full stack:

  ```bash
  ros2 launch f1tenth_bringup car.launch.py
  ros2 launch f1tenth_bringup race.launch.py stack:=algo
  ```

## RL training

Training is pure Python and never built by colcon. See
[`../training/README.md`](../training/README.md). The observation contract is
installed editable, so changing the observation space needs no rebuild.

## Conventions

- Python: `flake8` (`.flake8`); C++: `clang-format` (`.clang-format`). CI runs
  `ament_lint`.
- Don't commit heavy artifacts (weights, rosbags, outputs, maps, build trees) — they
  are gitignored.
- No company or internal product/project names anywhere in the repo (public repo).

## Decisions & PRs

- **Commits** follow a `scope: summary` convention, where `scope` is the module
  group or area and the summary is a short, imperative, lowercase phrase — e.g.
  `contract: seed f1tenth_contract with the real obs/action format`,
  `training: migrate standalone QR-SAC trainer`, `build: make workspace
  colcon-clean`.
- **PRs** use [`../.github/PULL_REQUEST_TEMPLATE.md`](../.github/PULL_REQUEST_TEMPLATE.md):
  say what/why, list the module group(s) touched and any downstream impact, and
  include build/test evidence. This is how routine work is documented.
- **ADRs** capture significant decisions. Write one when a change alters the repo's
  structure, a public interface or the observation/action contract, the
  build/deploy shape, or accepts a non-obvious trade-off. Copy
  [`adr/0000-template.md`](adr/0000-template.md); never rewrite an accepted ADR —
  supersede it.
- Agent-facing guidance lives in [`../AGENTS.md`](../AGENTS.md) and
  `../.cursor/rules/`; the domain primer is [`context.md`](context.md).
