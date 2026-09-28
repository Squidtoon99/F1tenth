# perception/

Perception module group. Turns raw sensor data into higher-level detections
consumed by the racing stacks.

- `f1tenth_perception/` — LiDAR opponent/obstacle detection
  (`/scan` → `/rl/opponent/odom`). Solo racing leaves the detector off; the
  6-dim opponent obs block stays zero.
