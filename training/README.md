# training/

Reinforcement-learning training frameworks. This is **pure Python**: it is never
built by colcon and never included in the car image. Training runs on developer
machines or on HPC (there is intentionally **no GPU CI**).

## Layout

- `f1tenth_env/` — the simulation environment wrapper used for training (synthetic
  rollouts + reward/termination logic).
- `qrsac/` — the RL algorithm (distributional actor-critic).
- `trainer/` — training entry points (single-process and any distributed variants).
- `configs/` — default training configuration.

## Setup (editable contract = no rebuild on obs changes)

The observation/action layout comes from the shared contract. Install it editable so
changing the observation space is picked up on the next run with no rebuild:

```bash
pip install -r training/requirements.txt   # includes: -e ../libs/f1tenth_contract
# or: conda env create -f training/environment.yml
```

## Run

```bash
python -m training.trainer.standalone_trainer --num-envs 512 --total-steps 500000
```

## Offboard / HPC

Build an Apptainer image for HPC and run there — see
[`deploy/apptainer/training.def`](../deploy/apptainer/training.def) and
[`docs/deployment.md`](../docs/deployment.md). Outputs (checkpoints, logs, wandb) are
kept out of git (see [`.gitignore`](../.gitignore)).

## Migration note

Maps onto the current `F1tenth-Genesis` repo:

| Here | From F1tenth-Genesis |
| --- | --- |
| `f1tenth_env/` | `f1tenth_env/` (Genesis env: env, car, observations, rewards, ...) |
| `qrsac/` | `qrsac/` |
| `trainer/standalone_trainer.py` | `standalone_trainer.py` |
| `configs/` | `config.py` (`DEFAULT_CONFIG`) / `param.py` / `task.py` |

The synthetic-data simulator (Genesis) is a pip dependency (`genesis-world`); see
[`../sim/genesis/`](../sim/genesis/).
