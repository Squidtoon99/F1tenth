# racing_rl/

On-car reinforcement-learning racing (inference only). This is a colcon ROS 2
package that ships in the car image; it loads a trained policy checkpoint (`.pt`)
supplied at runtime.

- `f1tenth_racing_rl/` — nodes:
  - observation builder (odom/pose + track -> observation, using the shared
    contract in [`libs/f1tenth_contract`](../../../libs/f1tenth_contract/)),
  - policy inference (loads the actor `.pt` + saved observation normalizer),
  - action decode -> hands off to the shared drive-command layer in `control/`.

The **training** side lives entirely under top-level [`training/`](../../../training/)
and is never built by colcon nor included in the car image.

End-to-end variant: the RL policy can run standalone (no planner) via a bringup
variant in `f1tenth_bringup`.

Migration note: the nodes in
`F1tenth-Genesis/ros2_deploy/f1tenth_rl_agent` (observation_builder,
policy_inference, obs_core, policy_model) move here. Note the upstream nested-copy
bug (`f1tenth_rl_agent/f1tenth_rl_agent/...`) must be cleaned up during migration.
