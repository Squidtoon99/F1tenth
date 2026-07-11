# sim/

Simulator integration. Two simulators are used, for different purposes:

- **Gym bridge** (`f1tenth_gym_ros`) — a ROS 2 wrapper around the F1TENTH gym. Used
  to run and evaluate the *deployed* racing stack (algorithmic or RL) over ROS
  topics, closest to on-car behavior. Pinned via
  [`f1tenth_gym_ros.repos`](f1tenth_gym_ros.repos) (the bridge fork + the gym engine
  it wraps) and imported, not vendored:

  ```bash
  vcs import sim < sim/f1tenth_gym_ros.repos
  ```

  The imported clone (`sim/f1tenth_gym_ros/`, with the engine nested at
  `f1tenth_gym/`) is gitignored. It is a sim-only dependency and is **not** part of
  the car runtime image.

### Running the RL stack against the gym

Use [`../tools/sim.sh`](../tools/sim.sh) — it stands up a **two-container** stack
(the gym bridge in the fork's own image, our nodes in the dev image) on a shared
docker bridge network, plus a noVNC container for RViz:

```bash
# STACK=vehicle (default) runs the exact on-car C++ graph; STACK=python is the
# regression check. The policy is mounted at /policies/${CKPT}.
CHECKPOINT_DIR=/abs/path/to/checkpoints CKPT=policy.pt STACK=vehicle ./tools/sim.sh up
#   RViz over VNC : http://localhost:8080/vnc.html
#   Foxglove      : ws://localhost:8765
CHECKPOINT_DIR=/abs/path/to/checkpoints CKPT=policy.pt STACK=vehicle ./tools/sim.sh validate
./tools/sim.sh down
```

It launches the fork's bridge-only `rl_agent_sim_launch.py` (the built-in
gap/pid/pp drivers are omitted so they don't fight the agent on `/drive`) together
with our `f1tenth_bringup/sim.launch.py` (the on-car C++ graph or the Python graph,
selected by `STACK`). The two ROS containers discover each other over DDS via a
shared `ROS_DOMAIN_ID` (no host networking — unreliable on macOS Docker Desktop).
The `evaluation` node spawns the car forward-facing along the centerline so it
matches the training distribution. `tools/sim.sh validate` then runs the closed-loop
acceptance gate; see [`../docs/deployment.md`](../docs/deployment.md) for the full
certification sequence and acceptance criteria.

- **Training simulator** ([`../training/f1tenth_sim/`](../training/f1tenth_sim/)) —
  the batched TorchSim vehicle model used for RL rollouts. It is pure Python/Torch,
  not a ROS package.

## Which sim, when?

| Task | Simulator |
| --- | --- |
| RL training / synthetic data | `training/f1tenth_sim/` (TorchSim) |
| Evaluating the deployed ROS stack | gym bridge (`f1tenth_gym_ros`) |

See [`docs/development.md`](../docs/development.md) for launch instructions.
