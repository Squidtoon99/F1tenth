# f1tenth_common

Shared utilities used across ROS 2 packages. Most importantly, it holds the
**C++ mirror** of the observation/action layout (`include/f1tenth_common/observation_layout.hpp`).

The single source of truth is the Python package
[`libs/f1tenth_contract`](../../../libs/f1tenth_contract/). Because the on-car C++
vehicle node cannot import Python, this header duplicates the layout and is kept
honest by a deploy-time parity test (`test/test_obs_parity.cpp`). This duplication
is intentional and does not affect the training loop, which uses the Python
contract directly.

Migration note: the existing C++ observation math in
`F1tenth-Genesis/ros2_deploy/f1tenth_rl_vehicle/src/rl_obs_core.cpp` moves here.
