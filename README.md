# F1TENTH Racing Stack

A monorepo for a 1/10th-scale autonomous racing stack. It hosts two racing
approaches that share common perception, mapping, and localization modules:

- **Reinforcement-learning racing** — an end-to-end policy trained in simulation
  (see [`training/`](training/)) and run on the car by a lightweight inference
  node (see [`src/racing_rl/`](src/racing_rl/)).
- **Algorithmic racing** — a classical pipeline (raceline optimization +
  pure-pursuit tracking) in [`src/racing_algo/`](src/racing_algo/).

Everything targets **ROS 2 Humble** and is developed primarily inside Docker.

## Repository layout

| Path | What lives here |
| --- | --- |
| [`src/`](src/) | The single colcon workspace: all ROS 2 packages, grouped by domain |
| [`libs/`](libs/) | Cross-cutting libraries (e.g. the observation/action contract) |
| [`training/`](training/) | RL training frameworks (pure Python; never built by colcon) |
| [`sim/`](sim/) | Simulator integration (gym bridge pin, synthetic-data sim notes) |
| [`deploy/`](deploy/) | Docker/Apptainer images, per-car config overlays, deploy scripts |
| [`tools/`](tools/) | Build/dev helper scripts and lint configs |
| [`calibration/`](calibration/) | Sim-to-real physics calibration tooling |
| [`analysis/`](analysis/) | Offline analysis (heavy artifacts are gitignored) |
| [`docs/`](docs/) | Architecture, development workflow, deployment, ADRs |

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design and rationale.

## Quick start (newcomers)

New here? Start with **[`docs/getting-started.md`](docs/getting-started.md)** — it
covers what the repo is, first-time setup, and contributing a ROS node end to end.

Everything runs in Docker; you do not need ROS 2 installed on your host.

```bash
git clone https://github.com/Squidtoon99/F1tenth.git f1tenth && cd f1tenth
./tools/dev.sh up        # build/enter the dev container (or `./tools/dev.sh pull`)
```

Teammates on amd64 can skip building by pulling the prebuilt seed images
([`squidtoon99/f1tenth-base`](https://hub.docker.com/r/squidtoon99/f1tenth-base),
[`squidtoon99/f1tenth-dev`](https://hub.docker.com/r/squidtoon99/f1tenth-dev)) via
`./tools/dev.sh pull`. Then, inside the container:

```bash
./tools/build.sh          # colcon build the whole workspace
./tools/build.sh -t racing_algo   # or just one module group
```

To bring up the simulator + a racing stack, see [`docs/development.md`](docs/development.md).

## Branching

`develop` is the trunk. Do feature work on `feature/<ticket>-<slug>` branches and
open a PR back into `develop`. Releases are git tags on `develop`. Details in
[`docs/development.md`](docs/development.md).

## Deployment

`develop` is built into a **generic** car image (no per-car config baked in),
snapshotted, and loaded onto the on-car Jetson. Per-car identity, parameters, and
maps are supplied at runtime from [`deploy/cars/`](deploy/cars/). See
[`docs/deployment.md`](docs/deployment.md).

## License

MIT. See [`LICENSE`](LICENSE).
