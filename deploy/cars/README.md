# deploy/cars/

Per-car config overlays. The car image is **generic**; everything car-specific
lives here and is supplied at runtime (mounted at `/config`), so one image serves
every car.

Each `carNN/` directory contains:

- `car.yaml` — the car's identity. Authored on the car; **never overwritten** by a
  deployed image or a config push.
- `params.yaml` — chassis calibration (VESC gains, servo offsets, wheelbase),
  safety limits, and the RL policy path. Overlaid on generic launch defaults.
- `maps/` — the occupancy grid + centerline/raceline for the current track.

To add a car, copy `car01/` to `carNN/` and edit the values.

The RL policy checkpoint itself is delivered separately (mounted at `/policies`),
not stored here, since it is a large binary artifact.
