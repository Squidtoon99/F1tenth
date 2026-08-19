# training/

Reinforcement-learning training code. This is **pure Python**: it is never built by
colcon and never included in the car image. Training runs natively on developer
machines (macOS supports small Warp CPU tests) or on a Linux + NVIDIA GPU box /
HPC (there is intentionally **no GPU CI**).

## Layout

The modules are top-level (imported as `config`, `run_layout`, `standalone_trainer`,
`f1tenth_env`, `qrsac`, `ppo`), so run commands from this `training/` directory.

- `f1tenth_env/` — the simulation environment used for training (observations,
  rewards, terminations, opponents, and domain randomization).
- `f1tenth_sim/` — Warp vehicle dynamics (Pacejka tyres, load transfer,
  wheel-spin, and force drivetrain), batched over environments on CPU or CUDA.
- `qrsac/` — the RL algorithm: Quantile-Regression Soft Actor-Critic
  (distributional actor-critic) plus the spinning-up MLP building blocks.
- `ppo/` — recurrent clipped PPO with an asymmetric scalar value critic,
  on-policy rollout storage, and optional Gigaflow-style advantage filtering
  (`config.ppo.advantage_filter_*`).
- `standalone_trainer.py` — single-process trainer entry point for either
  algorithm; QR-SAC uses in-memory n-step replay.
- `run_layout.py` — per-run output directory layout (`outputs/runs/<run-id>/`).
- `config.py` — `DEFAULT_CONFIG` (the single source of the trainer's config).
- `tests/` — learner, replay, environment, observation, reward, termination,
  opponent, export, and evaluation tests.

## Setup

The observation/action layout is documented by the shared contract, and the Lee
sensor actor/layout/artifact helpers live in `libs/f1tenth_policy`. Both are
installed editable so source changes are picked up immediately:

```bash
cd training

# macOS (Apple Silicon) — the local dev/test target:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt
# includes: -e ../libs/f1tenth_contract and -e ../libs/f1tenth_policy

# Linux + NVIDIA GPU (HPC / training rig):
pip install -r requirements-gpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130
```

Default config is the sole Lee sensor path (delta steering, termination at the
first mapped-wall intersection of the projected footprint, and one
`-20 * speed_mps` contact-event reward, with 10% stationary resets).
Fixed opponents use `fixed_opponents.entries`; every entry loads at startup and each
env samples a weighted opponent on every episode reset.

## Run

```bash
cd training
python standalone_trainer.py --num-envs 1024 --total-transitions 256000000

# Select recurrent PPO instead of the default QR-SAC trainer.
python standalone_trainer.py --algorithm ppo --num-envs 1024 \
    --total-transitions 256000000
```

### Warp physics

```bash
# GPU training uses float32 Warp kernels and zero-copy PyTorch interoperability.
# Reward/model/schedule knobs belong in a --config JSON patch, not the CLI.
python standalone_trainer.py --device cuda --num-envs 1024 --config my_run.json
```

Training domain randomization is always enabled. Evaluation and deterministic
tests use the nominal profile.

Galaxy sensor-policy runs use the ignored real-track occupancy map configured by
`sensor.lidar_map_yaml`. Warp keeps the cleaned map as variant zero and selects a
seeded, smoothly deformed wall-geometry variant per episode. The configured image
SHA-256 must match before the environment starts. Maps and generated parity reports
stay under `outputs/sim2real/`; they are never committed.

Budgets and all cadences are cumulative environment transitions. QR-SAC telemetry
tracks replay inserts and sampled rows; PPO telemetry tracks optimized rollout
rows and optimizer steps.

Policy exports under `outputs/runs/<id>/checkpoints/` are compact artifacts. They
retain the required `actor` and `obs_norm` keys plus transition count, dimensions,
action scaling/semantics, format version, and the physical current envelope
(`i_drive_max_a=80`, `i_brake_max_a=20`, `i_slew_a_per_s=200`) paired with
`f_drive_max≈23 N` at normalized effort=1. Warp derives a normalized longitudinal
slew of `2.5/s` (`200 A/s / 80 A`). Critic and optimizer state are not
persisted, and runs do not resume. Observation preprocessing v3 still packs
VESC current as a directional-limit fraction. Deploy refuses artifacts whose
embedded current scale does not match on-car limits. The Lee sensor experiment
uses normalized steering deltas of at most `pi/60` rad per 10 Hz decision;
replay keeps those normalized deltas while sensor history and steering rewards
use realized angles.

For the race-ready cold-start PPO run, use the reviewed configuration and keep
the 2B-transition ceiling so a healthy run can continue through temporary
performance regressions:

```bash
cd training
../.venv/bin/python standalone_trainer.py \
  --config configs/ecss_solo_v3_80a_ppo.json --algorithm ppo \
  --num-envs 4096 --total-transitions 2000000000 --opponent none \
  --device cuda --seed 55 --run-id race-ready-80a-ppo-s55-a001
```

This is a cold run: do not supply an initialization or resume checkpoint.

### Eval visualization (live + mp4)

`eval_visualize.py` runs a policy artifact deterministically in a single env and
renders a top-down view (track corridor, car + heading, breadcrumb trail, live
speed). It is **eval-only** — it never trains or touches a training run — and is
uses the same rollout path as in-training and diagnostic evaluation.

```bash
# Watch LIVE in a Rerun viewer — updates in real time, no waiting for a file:
python eval_visualize.py --checkpoint outputs/runs/<id>/checkpoints/policy_8000.pt --live

# Save an mp4 (e.g. to share or attach to W&B):
python eval_visualize.py --checkpoint .../policy_8000.pt --mp4 outputs/eval.mp4

# Overlay many env instances at once (swarm view of the behaviour spread):
python eval_visualize.py --checkpoint .../policy_8000.pt --live --num-show 24

# Both at once; headless box: stream to a saved .rrd instead of spawning a viewer:
python eval_visualize.py --checkpoint .../policy_8000.pt --live --no-spawn \
    --rr-save outputs/eval.rrd --mp4 outputs/eval.mp4
```

During training you can also emit periodic eval videos (rendered on a separate
1-env instance, logged to W&B; add `--eval-video-live` to also stream to Rerun):

```bash
python standalone_trainer.py --num-envs 1024 \
    --eval-interval-transitions 20480000 --eval-video-steps 600 \
    --eval-video-num-envs 16
```

Live view needs `rerun-sdk`; mp4 needs `opencv-python` + `imageio[ffmpeg]` (all in
the requirements files).

Run the unit tests and lint:

```bash
cd training
../.venv/bin/python -m pytest tests
../.venv/bin/python -m flake8 .
```

## Offboard / HPC

Build an Apptainer image for HPC and run there — see
[`deploy/apptainer/training.def`](../deploy/apptainer/training.def) and
[`docs/deployment.md`](../docs/deployment.md). Outputs (checkpoints, logs, wandb) are
kept out of git (see [`.gitignore`](../.gitignore)).
