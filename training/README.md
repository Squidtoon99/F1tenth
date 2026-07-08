# training/

Reinforcement-learning training code. This is **pure Python**: it is never built by
colcon and never included in the car image. Training runs natively on developer
machines (macOS Apple Silicon uses the torch MPS/Metal or CPU backend) or on a
Linux + NVIDIA GPU box / HPC (there is intentionally **no GPU CI**).

## Layout

The modules are top-level (imported as `config`, `run_layout`, `standalone_trainer`,
`f1tenth_env`, `qrsac`), so run commands from this `training/` directory.

- `f1tenth_env/` — the simulation environment used for training (env, car,
  observations, rewards, terminations, opponents, domain randomization). Loads the
  vehicle model from `F110.export.urdf` and track geometry from
  `f1tenth_env/tracks.pickle`. Physics is provided by a pluggable **backend**
  (`f1tenth_env/backends.py`): `genesis` (rigid-body engine, default) or `torch`.
- `f1tenth_sim/` — a pure-Torch, Genesis-free vehicle simulator (Pacejka tyres,
  load transfer, wheel-spin, VESC/force drivetrain) used by the `torch` backend.
  Fully batched over `num_envs` and runs on CPU / CUDA / MPS.
- `qrsac/` — the RL algorithm: Quantile-Regression Soft Actor-Critic
  (distributional actor-critic) plus the spinning-up MLP building blocks.
- `standalone_trainer.py` — single-process trainer entry point (F1tenthEnv +
  in-memory n-step replay; no Reverb/Redis/S3).
- `run_layout.py` — per-run output directory layout (`outputs/runs/<run-id>/`).
- `config.py` — `DEFAULT_CONFIG` (the single source of the trainer's config).
- `tests/` — observation-geometry / reward / termination / opponent unit tests
  (they run against the real env, so Genesis must be installed).

## Setup

The observation/action layout is documented by the shared contract, installed
editable so format changes need no rebuild/reinstall:

```bash
cd training

# macOS (Apple Silicon) — the local dev/test target:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt        # includes: -e ../libs/f1tenth_contract

# Linux + NVIDIA GPU (HPC / training rig):
pip install -r requirements-gpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130
```

## Run

```bash
cd training
python standalone_trainer.py --num-envs 512 --total-steps 500000
```

### Physics backend (Genesis vs Torch)

The physics engine is selectable and the observation/action contract is identical
for both, so the trainer and the deployed car are backend-agnostic:

```bash
# Pure-Torch vehicle sim (no Genesis; runs headless on CPU/CUDA/MPS). The torch
# sim prioritizes stable physics + throughput over bit-for-bit Genesis parity.
# It defaults to the VESC-style speed-command throttle (v_cmd = throttle * max_speed,
# matching the deployed car's drive_math.py):
python standalone_trainer.py --physics torch --num-envs 4096

# Open-loop drive-force throttle envelope instead of the speed loop:
python standalone_trainer.py --physics torch --throttle-mode force
```

To make Torch the **default** backend, set `DEFAULT_CONFIG["env"]["physics_backend"]
= "torch"` in `config.py` (and optionally the `torch_sim` / `throttle_mode` keys
there). The `--physics` flag always overrides the config default.

### Eval visualization (live + mp4)

`eval_visualize.py` runs a trained checkpoint deterministically in a single env and
renders a top-down view (track corridor, car + heading, breadcrumb trail, live
speed). It is **eval-only** — it never trains or touches a training run — and is
backend-agnostic (`f1tenth_env/eval_viz.py`).

```bash
# Watch LIVE in a Rerun viewer — updates in real time, no waiting for a file:
python eval_visualize.py --checkpoint outputs/runs/<id>/checkpoints/ckpt_8000.pt --live

# Save an mp4 (e.g. to share or attach to W&B):
python eval_visualize.py --checkpoint .../ckpt_8000.pt --mp4 outputs/eval.mp4

# Overlay many env instances at once (swarm view of the behaviour spread):
python eval_visualize.py --checkpoint .../ckpt_8000.pt --live --num-show 24

# Both at once; headless box: stream to a saved .rrd instead of spawning a viewer:
python eval_visualize.py --checkpoint .../ckpt_8000.pt --live --no-spawn \
    --rr-save outputs/eval.rrd --mp4 outputs/eval.mp4
```

During training you can also emit periodic eval videos (rendered on a separate
1-env instance, logged to W&B; add `--eval-video-live` to also stream to Rerun):

```bash
python standalone_trainer.py --physics torch --num-envs 4096 \
    --eval-video-interval 5000 --eval-video-steps 600 --eval-video-num-envs 16
```

Live view needs `rerun-sdk`; mp4 needs `opencv-python` + `imageio[ffmpeg]` (all in
the requirements files).

Run the unit tests (skipped if Genesis is not installed):

```bash
cd training && pytest
```

## Offboard / HPC

Build an Apptainer image for HPC and run there — see
[`deploy/apptainer/training.def`](../deploy/apptainer/training.def) and
[`docs/deployment.md`](../docs/deployment.md). Outputs (checkpoints, logs, wandb) are
kept out of git (see [`.gitignore`](../.gitignore)).

The synthetic-data simulator (Genesis) is a pip dependency (`genesis-world`); see
[`../sim/genesis/`](../sim/genesis/).
