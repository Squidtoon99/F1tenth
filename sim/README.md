# sim/

Simulator integration. Two simulators are used, for different purposes:

- **Gym bridge** (`f1tenth_gym_ros`) — a ROS 2 wrapper around the F1TENTH gym. Used
  to run and evaluate the *deployed* racing stack (algorithmic or RL) over ROS
  topics, closest to on-car behavior. Pinned via
  [`f1tenth_gym_ros.repos`](f1tenth_gym_ros.repos) and imported, not vendored:

  ```bash
  vcs import sim < sim/f1tenth_gym_ros.repos
  ```

  The imported clone is gitignored. It is a sim-only dependency and is **not** part
  of the car runtime image.

- **Synthetic-data simulator** ([`genesis/`](genesis/)) — used by RL training for
  large-scale synthetic rollouts. It is a pip dependency of `training/`
  (`genesis-world`), not a ROS package.

## Which sim, when?

| Task | Simulator |
| --- | --- |
| RL training / synthetic data | `genesis/` (pip, in `training/`) |
| Evaluating the deployed ROS stack | gym bridge (`f1tenth_gym_ros`) |

See [`docs/development.md`](../docs/development.md) for launch instructions.
