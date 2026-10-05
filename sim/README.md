# sim/

Two simulators, different jobs. Training never talks ROS; gym eval never trains.

| Task | Simulator |
| --- | --- |
| RL training / synthetic data | `training/f1tenth_sim/` (NVIDIA Warp) |
| Evaluating the deployed ROS stack | gym bridge (`f1tenth_gym_ros`) |

```mermaid
flowchart LR
  warp["Warp (batched, no ROS)"] --> pt["policy.pt"]
  pt --> gym["gym bridge + vehicle_obs → policy_inference → drive"]
  gym --> vnc["RViz :8080 / Foxglove :8765"]
```

## Gym bridge (`f1tenth_gym_ros`)

A ROS 2 wrapper around the F1TENTH gym. Used to run the *deployed* racing stack
over ROS topics, closest to on-car behavior. Pinned via
[`f1tenth_gym_ros.repos`](f1tenth_gym_ros.repos) and imported, not vendored:

```bash
vcs import sim < sim/f1tenth_gym_ros.repos
```

`tools/sim.sh` runs that import inside `f1tenth-base` so the host needs no
`vcs`. The clone (`sim/f1tenth_gym_ros/`, engine at `f1tenth_gym/`) is gitignored
and is **not** in the car image.

Map is **IV_2026**. The gym centerline and
`src/racing_rl/f1tenth_rl_agent/assets/IV_2026_SIM_centerline.csv` must match
(`tools/sim.sh` checks). Train with `--track IV_2026_SIM` for a checkpoint that
can pass this gate.

### Run the on-car graph against the gym

[`../tools/sim.sh`](../tools/sim.sh) starts noVNC + the gym bridge + our agent:

```bash
# Policy is mounted at /policies/${CKPT}. Needs a force-mode 390-dim checkpoint.
CHECKPOINT_DIR=/abs/path/to/checkpoints CKPT=policy.pt ./tools/sim.sh up
#   RViz over VNC : http://localhost:8080/vnc.html
#   Foxglove      : ws://localhost:8765
CHECKPOINT_DIR=/abs/path/to/checkpoints CKPT=policy.pt ./tools/sim.sh validate
./tools/sim.sh down
```

Env:

| Var | Meaning |
| --- | --- |
| `CHECKPOINT_DIR` | Host dir with the `.pt` (required) |
| `CKPT` | Filename inside that dir (default `policy.pt`) |
| `STACK` | `vehicle` only — on-car C++ graph |
| `MODE` | `dev` (live source, default) or `release` (baked `f1tenth-racing` image) |
| `ROS_DOMAIN_ID` | Shared DDS domain (default 42; no host networking) |

It launches the fork's bridge-only `rl_agent_sim_launch.py` (gap/pid/pp omitted
so they don't fight `/drive`) plus `f1tenth_bringup/sim.launch.py`. The
`evaluation` node spawns forward-facing on the centerline. `validate` is the
closed-loop acceptance gate; criteria live in
[`../docs/deployment.md`](../docs/deployment.md).

Release-mode gym (image + `/config` + `/policies`):

```bash
deploy/scripts/build_image.sh
CHECKPOINT_DIR=/abs/dir CKPT=policy.pt CONFIG_DIR=deploy/cars/car01 \
  STACK=vehicle MODE=release ./tools/sim.sh up
```

## Training simulator

[`../training/f1tenth_sim/`](../training/f1tenth_sim/) is the batched Warp
vehicle model. Pure Python, not a ROS package. See
[`../training/README.md`](../training/README.md).
