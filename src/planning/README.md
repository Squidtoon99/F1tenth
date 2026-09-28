# planning/

Planning module group (algorithmic stack). Turns a map/centerline into a reference
trajectory for the controller.

- `f1tenth_planning/` — offline raceline (minimum-curvature/optimal) generation and
  online global/local planning. Raceline artifacts are consumed from
  `/config/maps/raceline.csv` at runtime.
