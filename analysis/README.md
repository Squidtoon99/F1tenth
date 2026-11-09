# analysis/

Offline analysis of races, telemetry, rosbags, and training runs. Keep **notebooks
and scripts** here. The data they consume — rosbags, logs, checkpoints, wandb
exports, rendered plots — is heavy and **gitignored**.

Scripts in this tree:

- `warp_parity/generate_torch_reference.py` — regenerate the committed Warp↔Torch
  trajectory fixture (`training/tests/data/torch_reference_trajectory.npz`).
- `steering_full_lock/warp_full_lock.py` — Warp full-lock turning-radius check
  against the measured 0.33 rad servo clamp.
- `friction/estimate_mu_from_bag.py` — friction estimate from a rosbag.

Training-side harnesses live next to the trainer:

- [`../training/analysis/lap_timing.py`](../training/analysis/lap_timing.py) — solo
  lap-time distributions over a run's checkpoints.
- [`../training/analysis/tournament.py`](../training/analysis/tournament.py) — 1v1
  triple-elim bracket.

Suggested convention for new work:

```text
analysis/
  <topic>/            # e.g. laptime_study/, raceline_compare/
    notebook.ipynb    # or a .py script
    README.md         # what it does + where the data lives (off-repo)
```
