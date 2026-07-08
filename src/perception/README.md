# perception/

Perception module group. Turns raw sensor data into higher-level detections
consumed by the racing stacks.

- `f1tenth_perception/` — nodes such as LiDAR-based opponent/obstacle detection.

Migration note: the opponent detector in
`F1tenth-Genesis/ros2_deploy/f1tenth_rl_vehicle` (LiDAR `/scan` ->
`/rl/opponent/odom`) moves here.
