# sim/

Simulator integration. Two simulators, for different jobs:

| Task | Simulator |
| --- | --- |
| RL training / synthetic data | `training/f1tenth_sim/` (NVIDIA Warp) |
| Evaluating the deployed ROS stack | gym bridge (`f1tenth_gym_ros`) |

## Gym bridge

A ROS 2 wrapper around the F1TENTH gym. Runs the *deployed* racing stack over ROS
topics, closest to on-car behavior. Pinned via
[`f1tenth_gym_ros.repos`](f1tenth_gym_ros.repos) and imported, not vendored:

```bash
vcs import sim < sim/f1tenth_gym_ros.repos
```

The clone (`sim/f1tenth_gym_ros/`) is gitignored. It is **not** in the car image.
Gym track assets are **`IV_2026_SIM`**.

### Run the RL stack against the gym

[`../tools/sim.sh`](../tools/sim.sh) stands up a **two-container** stack (gym
bridge in the fork image, our nodes in the dev image) on a docker bridge network,
plus noVNC for RViz:

```bash
# Exact on-car C++ graph. Policy mounted at /policies/${CKPT}.
CHECKPOINT_DIR=/abs/path/to/checkpoints CKPT=policy.pt ./tools/sim.sh up
#   RViz over VNC : http://localhost:8080/vnc.html
#   Foxglove      : ws://localhost:8765
CHECKPOINT_DIR=/abs/path/to/checkpoints CKPT=policy.pt ./tools/sim.sh validate
./tools/sim.sh down
```

`CHECKPOINT_DIR` must be absolute. `MODE=dev` (default) builds live source;
`MODE=release` runs the generic car image. `STACK=vehicle` is the only graph
(the release gate).

The fork's `rl_agent_sim_launch.py` omits gap/pid/pp drivers so they don't fight
`/drive`. Our `f1tenth_bringup/sim.launch.py` runs `vehicle_obs` →
`policy_inference` → `drive`. DDS discovery uses a shared `ROS_DOMAIN_ID`
(default 42) — no host networking (unreliable on macOS Docker Desktop).

The `evaluation` node spawns the car forward-facing along the centerline.
`validate` is the closed-loop acceptance gate; criteria live in
[`../docs/deployment.md`](../docs/deployment.md).

## Training simulator

[`../training/f1tenth_sim/`](../training/f1tenth_sim/) — batched Warp vehicle
model, pure Python. Default train track is `ecss_clean`; pass `--track IV_2026_SIM`
before gym certification. See [`../docs/training.md`](../docs/training.md).
