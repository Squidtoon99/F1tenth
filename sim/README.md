# sim/

Simulator integration. Two simulators, for different jobs:

| Task | Simulator |
| --- | --- |
| RL training / synthetic data | [`training/f1tenth_sim/`](../training/f1tenth_sim/) (NVIDIA Warp) |
| Evaluating the deployed ROS stack | gym bridge (`f1tenth_gym_ros`) |

## Gym bridge (`f1tenth_gym_ros`)

A ROS 2 wrapper around the F1TENTH gym. Used to run the *deployed* racing stack
(algorithmic or RL) over ROS topics, closest to on-car behavior. Pinned via
[`f1tenth_gym_ros.repos`](f1tenth_gym_ros.repos) (the bridge fork + the gym engine
it wraps) and imported, not vendored:

```bash
vcs import sim < sim/f1tenth_gym_ros.repos
```

The imported clone (`sim/f1tenth_gym_ros/`, with the engine nested at
`f1tenth_gym/`) is gitignored. It is a sim-only dependency and is **not** part of
the car runtime image. Gym assets are **IV_2026_SIM**, not the training default
`ecss_clean`.

### Running the RL stack against the gym

Use [`../tools/sim.sh`](../tools/sim.sh) — it stands up a **two-container** stack
(the gym bridge in the fork's own image, our nodes in the dev image) on a shared
docker bridge network, plus a noVNC container for RViz:

```bash
# Dev: live source, exact on-car C++ graph. Policy mounted at /policies/${CKPT}.
CHECKPOINT_DIR=/abs/path/to/checkpoints CKPT=policy.pt ./tools/sim.sh up
#   RViz over VNC : http://localhost:8080/vnc.html
#   Foxglove      : ws://localhost:8765

CHECKPOINT_DIR=/abs/path/to/checkpoints CKPT=policy.pt ./tools/sim.sh validate
./tools/sim.sh down
```

`STACK` must be `vehicle` (the only supported graph). `MODE=dev` (default) builds
the live workspace; `MODE=release` runs the generic car image with `/config` +
`/policies` mounted:

```bash
CHECKPOINT_DIR=/abs/dir CKPT=policy.pt CONFIG_DIR=deploy/cars/car01 \
  STACK=vehicle MODE=release ./tools/sim.sh up
```

`tools/sim.sh` imports the gym (`vcs import` via the base image), builds the agent
workspace, launches the fork's bridge-only `rl_agent_sim_launch.py` (built-in
gap/pid/pp drivers omitted so they don't fight `/drive`) together with our
`f1tenth_bringup/sim.launch.py` (on-car C++ graph). The two ROS containers
discover each other over DDS via a shared `ROS_DOMAIN_ID` (no host networking —
unreliable on macOS Docker Desktop).

The `evaluation` node spawns the car forward-facing along the centerline so it
matches the training distribution. `tools/sim.sh validate` then runs the closed-loop
acceptance gate; see [`../docs/deployment.md`](../docs/deployment.md) for the full
certification sequence and acceptance criteria.

Needs a `policy_format_version >= 2` force-mode checkpoint trained on
`IV_2026_SIM`. Floor-track (`ecss_clean`) weights will load but will not race the
gym map.

## Training simulator

[`../training/f1tenth_sim/`](../training/f1tenth_sim/) is the batched NVIDIA Warp
vehicle model used for RL rollouts. Pure Python, not a ROS package. See
[`../training/README.md`](../training/README.md).

See [`docs/development.md`](../docs/development.md) for the rest of the workflow.
