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

Teammates can skip building by pulling the prebuilt seed images from Docker Hub
(published amd64):
[`squidtoon99/f1tenth-base`](https://hub.docker.com/r/squidtoon99/f1tenth-base),
[`squidtoon99/f1tenth-dev`](https://hub.docker.com/r/squidtoon99/f1tenth-dev).

```bash
./tools/dev.sh pull      # docker pull squidtoon99/f1tenth-{base,dev}:latest, retag local
./tools/dev.sh shell
```

Point at a different registry with `SEED_REGISTRY=<user> ./tools/dev.sh pull`.

On Apple Silicon the amd64 seed runs under emulation, so an arm64 host is usually
better off building natively (`./tools/dev.sh build`). Images can also be shared as
tarballs without any registry:

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

- **Gym (deployed ROS stack)** — two containers: the gym bridge in the fork image
  plus our nodes in the dev image, sharing a docker bridge network. noVNC serves
  RViz; Foxglove is `ws://localhost:8765`. `STACK=vehicle` is the only graph
  (`vehicle_obs` → `policy_inference` → `drive`).

  ```bash
  # CHECKPOINT_DIR is the host dir holding the trained .pt; CKPT its filename.
  CHECKPOINT_DIR=/abs/path/to/checkpoints CKPT=policy.pt ./tools/sim.sh up
  # RViz: http://localhost:8080/vnc.html   Foxglove: ws://localhost:8765
  CHECKPOINT_DIR=/abs/path/to/checkpoints CKPT=policy.pt ./tools/sim.sh validate
  ./tools/sim.sh down
  ```

  `MODE=dev` (default) builds live source. `MODE=release` runs the generic car
  image with `/config` + `/policies` mounted. Gym map is `IV_2026_SIM`. See
  [`../sim/README.md`](../sim/README.md).

- **On the car** — vehicle-only shakedown, then a racing stack.
  `race.launch.py` defaults to `stack:=algo`; pass `stack:=rl` for the policy.

  ```bash
  ros2 launch f1tenth_bringup car.launch.py
  ros2 launch f1tenth_bringup race.launch.py stack:=algo
  ros2 launch f1tenth_bringup race.launch.py stack:=rl
  ```

## RL training

Training is pure Python (NVIDIA Warp + QR-SAC) and never built by colcon. Create
`.venv` at the repo root, then run from `training/`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r training/requirements-gpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130
cd training
python standalone_trainer.py --device cuda --num-envs 4096 --track IV_2026_SIM
```

macOS: `pip install -r training/requirements-mac.txt` — Warp CPU tests only, no
MPS. There is no `--physics` flag. See [`../training/README.md`](../training/README.md).
The observation contract is installed editable, so changing the observation space
needs no rebuild (but it is a shared-surface change — see
[`observation_contract.md`](observation_contract.md)).

## Pitfalls

- ROS 2 nodes: Docker only (`./tools/dev.sh up`). Host `.venv` has no `rclpy`.
- Training: host `.venv` at the repo root. Run from `training/` (or
  `PYTHONPATH=training`). `.venv/bin/python training/standalone_trainer.py` from
  the repo root does not import `config` / `f1tenth_env` on its own.
- Default training track is `ecss_clean`. Gym eval is `IV_2026_SIM`. Match them.
- Checkpoints must be `policy_format_version >= 2` with `longitudinal_mode: force`.
- Slip/load: `vehicle_obs.enable_slip_estimation` / `enable_load_estimation` in
  the per-car overlay must match how the checkpoint was trained. Copy
  `obs_norm.mean[372:380]` into `slip_obs_mean` when estimation is on.
- `race.launch.py` defaults to the algorithmic stack. The runtime image CMD is
  that launch with no `stack:=` override.
- `-v /dev:/dev` on the car: `--privileged` exposes device nodes but not the udev
  symlinks (`/dev/sensors/vesc`, LiDAR, joystick).

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
