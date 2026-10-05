# perception/

Perception module group. Intended for opponent/obstacle detection from sensors.

`f1tenth_perception/` is a placeholder package (no nodes yet). The live LiDAR
opponent detector is `opponent_detector` in [`../racing_rl/`](../racing_rl/):
`/scan` + pose → `/rl/opponent/odom`. Solo racing leaves it off; the 6-dim
opponent obs block stays zero.
