# training/

Reinforcement-learning training code. This is **pure Python**: it is never built by
colcon and never included in the car image. Training runs natively on developer
machines (macOS supports small Warp CPU tests) or on a Linux + NVIDIA GPU box /
HPC (there is intentionally **no GPU CI**).

How to train, eval, and export a checkpoint: [`docs/training.md`](../docs/training.md).

## Layout

The modules are top-level (imported as `config`, `run_layout`, `standalone_trainer`,
`f1tenth_env`, `qrsac`), so the trainer script adds this directory to `sys.path`
when run as `python training/standalone_trainer.py` from the repo root.

- `f1tenth_env/` — the simulation environment used for training (observations,
  rewards, terminations, opponents, and domain randomization).
- `f1tenth_sim/` — Warp vehicle dynamics (Pacejka tyres, load transfer,
  wheel-spin, and force drivetrain), batched over environments on CPU or CUDA.
- `qrsac/` — the RL algorithm: Quantile-Regression Soft Actor-Critic
  (distributional actor-critic) plus the spinning-up MLP building blocks.
- `standalone_trainer.py` — single-process trainer entry point (F1tenthEnv +
  in-memory n-step replay; no Reverb/Redis/S3).
- `run_layout.py` — per-run output directory layout (`outputs/runs/<run-id>/`).
- `config.py` — `DEFAULT_CONFIG` (trainer config; default track `ecss_clean`).
- `analysis/tournament.py` — 1v1 checkpoint races (crash-hold respawn).
- `tests/` — learner, replay, environment, observation, reward, termination,
  opponent, export, and evaluation tests.

## Setup

Venv lives at the **repo root** (not `training/.venv`):

```bash
python3 -m venv .venv && source .venv/bin/activate
# macOS (Apple Silicon) — local CPU tests:
pip install -r training/requirements-mac.txt
# Linux + NVIDIA GPU:
pip install -r training/requirements-gpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130
pip install pytest flake8
```

The contract is installed editable (`-e libs/f1tenth_contract`).

## Run

```bash
.venv/bin/python training/standalone_trainer.py \
  --device cuda --precision 32 --num-envs 4096
```

Training domain randomization is always enabled. Evaluation and deterministic
tests use the nominal profile. Budgets are cumulative environment transitions.

Policy exports under `outputs/runs/<id>/checkpoints/` keep `actor` + `obs_norm`
plus format metadata; critic and optimizer state are not persisted, and runs do
not resume.

### Eval visualization

```bash
.venv/bin/python training/eval_visualize.py \
  --checkpoint outputs/runs/<id>/checkpoints/policy_8000.pt --live
```

Live view needs `rerun-sdk`; mp4 needs `opencv-python` + `imageio[ffmpeg]` (both
in the requirements files). More flags: [`docs/training.md`](../docs/training.md).

```bash
.venv/bin/python -m pytest training/tests
.venv/bin/python -m flake8 training
```

## Offboard / HPC

Build an Apptainer image for HPC and run there — see
[`deploy/apptainer/training.def`](../deploy/apptainer/training.def) and
[`docs/deployment.md`](../docs/deployment.md). Outputs (checkpoints, logs, wandb) are
kept out of git (see [`.gitignore`](../.gitignore)).
