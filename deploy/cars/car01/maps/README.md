# Per-car map overlay (mounted at /config/maps)

Place track artifacts here before deploying:

- `map.yaml` + `map.pgm` — occupancy grid for map_server / particle filter
- `centerline.csv` — centerline in map frame (`x_m`, `y_m`, `psi_rad` columns)
- `raceline.csv` — raceline with speed (`vx_mps` column) for algorithmic drivers

These files are gitignored (heavy artifacts). Copy from the car's `~/maps/` or from
SLAM postprocess outputs under `src/mapping/postprocess/`.
