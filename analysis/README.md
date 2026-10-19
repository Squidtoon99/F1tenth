# analysis/

Offline scripts. Heavy outputs (bags, plots, checkpoints) stay gitignored.

## Training (Warp env, no ROS)

Lives in [`training/analysis/`](../training/analysis/). Default `--track` is
`Austin` (fetched), not the trainer default `ecss_clean` or the gym map
`IV_2026_SIM`.

```bash
# Solo lap times for every checkpoint in a run (writes lap_timing/):
.venv/bin/python training/analysis/lap_timing.py \
  --run-dir outputs/runs/<id> --workers 8 --device cuda

# 1v1 triple-elim, seeded from that CSV (default path: <run>/lap_timing/car_lap_times.csv):
.venv/bin/python training/analysis/tournament.py \
  --run-dir outputs/runs/<id> --device cuda
```

Crashes in a tournament match are not terminal: the car respawns, holds 2 s, then
resumes. See the module docstrings for worker vs orchestrator flags.

## Physics / bags

- `warp_parity/generate_torch_reference.py` — regenerate the committed Torch
  trajectory fixture (`training/tests/data/torch_reference_trajectory.npz`). Needs
  a checkout that still has `TorchVehicleSim` (`F1TENTH_TORCH_ROOT`). Do not
  rebaseline unless you mean to.
- `friction/estimate_mu_from_bag.py` — peak |a|/g from a driving rosbag
  (`/odom`, `/sensors/imu/raw`, `/pf/pose/odom`).
- `steering_full_lock/warp_full_lock.py` — Warp steady full-lock turning radius
  vs the measured car.

Keep notebooks and scripts here. Data they consume is off-repo.
