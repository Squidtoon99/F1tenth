# mapping/

Track mapping module group (shared by both racing stacks). Wraps a SLAM backend
(`slam_toolbox`, installed via rosdep) and adds exploration + post-processing to
produce an occupancy grid and a centerline CSV.

- `f1tenth_mapping/` — `exploration`, `navigator`, `reactive_explorer`,
  `occupancy_mapper`, `scan_throttle`.

  ```bash
  ros2 launch f1tenth_mapping mapping_real.launch.py   # car (f1tenth_stack)
  ros2 launch f1tenth_mapping mapping_sim.launch.py    # gym
  ```

- `postprocess/` — offline (not colcon). Occupancy grid or raceline CSV →
  centerline. See [`postprocess/README.md`](postprocess/README.md).

Generated maps go in the per-car overlay
([`deploy/cars/`](../../../deploy/cars/)); large rendered grids are gitignored.
