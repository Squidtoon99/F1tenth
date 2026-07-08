# localization/

Localization module group (shared by both racing stacks). Provides pose on a known
map, typically via a particle filter with GPU-accelerated ray casting.

- `f1tenth_localization/` — launch/config/wrappers around the particle-filter
  localizer (the filter itself is an upstream dependency installed via rosdep or
  vendored under `src/vehicle/` if a fork is needed).

Migration note: the particle-filter configuration used by
`F1tenth-Genesis/ros2_deploy/f1tenth_rl_vehicle` (PF pose feeding the observation
builder) is captured here.
