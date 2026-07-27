# training/

Reinforcement-learning training code. This is **pure Python**: it is never built by
colcon and never included in the car image. Training runs natively on developer
machines (macOS supports small Warp CPU tests) or on a Linux + NVIDIA GPU box /
HPC (there is intentionally **no GPU CI**).

## Layout

The modules are top-level (imported as `config`, `run_layout`, `standalone_trainer`,
`f1tenth_env`, `qrsac`), so run commands from this `training/` directory.

- `f1tenth_env/` — the simulation environment used for training (observations,
  rewards, terminations, opponents, and domain randomization).
- `f1tenth_sim/` — Warp vehicle dynamics (Pacejka tyres, load transfer,
  wheel-spin, and force drivetrain), batched over environments on CPU or CUDA.
- `qrsac/` — the RL algorithm: Quantile-Regression Soft Actor-Critic
  (distributional actor-critic) plus the spinning-up MLP building blocks.
- `standalone_trainer.py` — single-process trainer entry point (F1tenthEnv +
  in-memory n-step replay; no Reverb/Redis/S3).
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
```

### Warp physics

```bash
# GPU training uses float32 Warp kernels and zero-copy PyTorch interoperability.
# Reward/model/schedule knobs belong in a --config JSON patch, not the CLI.
python standalone_trainer.py --device cuda --num-envs 1024 --config my_run.json
```

Training domain randomization is always enabled. Evaluation and deterministic
tests use the nominal profile.

Budgets and all cadences are cumulative environment transitions. Telemetry keeps
separate counts for vector ticks, environment transitions, replay inserts,
sampled replay rows, and gradient updates. The default learner budget samples two
replay rows per collected transition, preserving the former 512-env,
1024-row-batch update ratio independently of vector width.

Policy exports under `outputs/runs/<id>/checkpoints/` are compact artifacts. They
retain the required `actor` and `obs_norm` keys plus transition count, dimensions,
action scaling/semantics, and format version; critic and optimizer state are not
persisted, and runs do not resume. The Lee sensor experiment uses normalized
steering deltas of at most `pi/60` rad per 10 Hz decision; replay keeps those
normalized deltas while sensor history and steering rewards use realized angles.

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
