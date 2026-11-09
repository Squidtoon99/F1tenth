# perception/

Perception module group. Turns raw sensor data into higher-level detections
consumed by the racing stacks.

- `f1tenth_perception/` — empty package scaffold (no nodes registered).

The LiDAR opponent detector that publishes `/rl/opponent/odom` lives in
`src/racing_rl/f1tenth_rl_vehicle` (`opponent_detector_node.cpp`), launched from
`bringup_vehicle.launch.py` when `enable_opponent:=true`.
