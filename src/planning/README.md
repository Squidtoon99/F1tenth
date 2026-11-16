# planning/

Planning module group (algorithmic stack). Turns a map/centerline into a reference
trajectory for the controller.

- `f1tenth_planning/` — empty package scaffold (no nodes registered).

Centerline / raceline CSVs are produced by [`../mapping/postprocess/`](../mapping/postprocess/)
and consumed at runtime from the per-car overlay (`deploy/cars/<car>/maps/`).
Controllers in `src/control/` track `/config/maps/raceline.csv`.
