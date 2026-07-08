# training/

Reinforcement-learning training code. This is **pure Python**: it is never built by
colcon and never included in the car image. Training runs natively on developer
machines (macOS Apple Silicon uses the torch MPS/Metal or CPU backend) or on a
Linux + NVIDIA GPU box / HPC (there is intentionally **no GPU CI**).

## Layout

The modules are top-level (imported as `config`, `run_layout`, `standalone_trainer`,
`f1tenth_env`, `qrsac`), so run commands from this `training/` directory.

- `f1tenth_env/` — the Genesis simulation environment used for training (env, car,
  observations, rewards, terminations, opponents, domain randomization). Loads the
  vehicle model from `F110.export.urdf` and track geometry from
  `f1tenth_env/tracks.pickle`.
- `qrsac/` — the RL algorithm: Quantile-Regression Soft Actor-Critic
  (distributional actor-critic) plus the spinning-up MLP building blocks.
- `standalone_trainer.py` — single-process trainer entry point (F1tenthEnv +
  in-memory n-step replay; no Reverb/Redis/S3).
- `run_layout.py` — per-run output directory layout (`outputs/runs/<run-id>/`).
- `config.py` — `DEFAULT_CONFIG` (the single source of the trainer's config).
- `tests/` — observation-geometry / reward / termination / opponent unit tests
  (they `pytest.importorskip("genesis")`, so they are skipped where Genesis is not
  installed).

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
