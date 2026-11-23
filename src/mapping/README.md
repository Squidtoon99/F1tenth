# mapping/

Track mapping module group (shared by both racing stacks). Wraps a SLAM backend
(`slam_toolbox`, installed via rosdep) and adds exploration + post-processing to
produce an occupancy grid and a centerline CSV.

- `f1tenth_mapping/` — exploration/mapping nodes (`mapping_sim.launch.py`,
  `mapping_real.launch.py`, `reactive_mapping_car.launch.py`).
- `postprocess/` — offline SLAM map → centerline/raceline (not a colcon package).

Generated maps are consumed at deploy time via the per-car overlay
([`deploy/cars/`](../../../deploy/cars/)); large rendered grids are gitignored.
