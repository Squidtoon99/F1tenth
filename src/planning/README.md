# planning/

Planning module group (algorithmic stack). Turns a map/centerline into a reference
trajectory for the controller.

- `f1tenth_planning/` — offline raceline (minimum-curvature/optimal) generation and
  online global/local planning.

Migration note: the centerline/raceline builders in `F1tenth-Genesis/scripts/` and
the navigator planning bits in `ros2_mapping/f1tenth_mapping` consolidate here.
