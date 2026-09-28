# mapping/

Track mapping module group (shared by both racing stacks). Wraps a SLAM backend
(`slam_toolbox`, installed via rosdep) and adds exploration + post-processing to
produce an occupancy grid and a centerline CSV.

- `f1tenth_mapping/` — exploration/mapping nodes and map post-processing
  (`postprocess/` turns a SLAM grid into a centerline).

Generated maps are consumed at deploy time via the per-car overlay
([`deploy/cars/`](../../../deploy/cars/)); large rendered grids are gitignored.
