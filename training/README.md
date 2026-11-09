# training/

Reinforcement-learning training code. This is **pure Python**: it is never built by
colcon and never included in the car image. GPU training is Linux + NVIDIA CUDA.
macOS supports small Warp CPU tests only (no MPS). There is intentionally **no GPU
CI**.

## Layout

Trainer modules are top-level (`config`, `run_layout`, `standalone_trainer`,
`f1tenth_env`, `qrsac`). Run from this `training/` directory, or set
`PYTHONPATH=training` from the repo root.

- `f1tenth_env/` — Warp environment (observations, rewards, terminations,
  opponents, domain randomization). `F1tenthEnv` is `WarpF1tenthEnv`.
- `f1tenth_sim/` — Warp vehicle dynamics (Pacejka tyres, load transfer,
  wheel-spin, force drivetrain), batched on CPU or CUDA.
- `qrsac/` — Quantile-Regression Soft Actor-Critic plus spinning-up MLP blocks.
- `standalone_trainer.py` — single-process trainer (in-memory n-step replay).
- `run_layout.py` — per-run output directory layout (`outputs/runs/<run-id>/`).
- `config.py` — `DEFAULT_CONFIG` (trainer config source of truth).
- `eval_visualize.py` — eval-only live / mp4 rollout of a saved policy.
- `analysis/` — `lap_timing.py` (solo pace) and `tournament.py` (1v1 bracket).
- `assets/` — bundled centerlines (`ecss_clean`, `IV_2026_SIM`).
- `tests/` — learner, replay, environment, observation, reward, Warp, export.

There is no `--physics` switch. Warp is the only simulator
([ADR 0007](../docs/adr/0007-warp-simulator.md)).

## Setup

Create the venv at the **repo root** (same `.venv` used by calibration and the
deploy parity commands):

```bash
# from the repo root
python3 -m venv .venv && source .venv/bin/activate

# macOS (Apple Silicon) — small Warp CPU tests:
pip install -r training/requirements-mac.txt        # includes: -e libs/f1tenth_contract

# Linux + NVIDIA GPU (HPC / training rig):
pip install -r training/requirements-gpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130
```

`conda env create -f training/environment.yml` is the macOS-oriented conda
equivalent. Prefer the pip files on a GPU box.

## Train a policy

```bash
cd training

# Solo (1v0) on the default building-floor track:
python standalone_trainer.py --device cuda --num-envs 4096 --track ecss_clean

# Gym closed-loop gate uses IV_2026_SIM — train that track before tools/sim.sh:
python standalone_trainer.py --device cuda --num-envs 4096 --track IV_2026_SIM

# 1v1 mixed field (scripted + self-play snapshots):
python standalone_trainer.py --device cuda --num-envs 4096 --mixed-opponents

# JSON patch over DEFAULT_CONFIG (patch < explicitly-passed CLI flags):
python standalone_trainer.py --config path/to/patch.json --device cuda --num-envs 4096
```

`--precision` is float32 only. `--device` is `auto` / `cpu` / `cuda`. Training
domain randomization is always on. Evaluation harnesses and deterministic tests
use the nominal profile unless they say otherwise.

Budgets and cadences are cumulative environment transitions. The default learner
samples two replay rows per collected transition, independent of vector width.

Policy exports under `outputs/runs/<id>/checkpoints/` are compact artifacts:
`actor`, `obs_norm`, transition count, dims, action scaling, `policy_format_version`
(2), and `longitudinal_mode: force`. Critic and optimizer state are not persisted,
and runs do not resume. Speed-trained checkpoints will not load on the car.

### Opponents

| Flag | What it does |
| --- | --- |
| (default `--opponent none`) | Solo / 1v0. Opponent block `[384:390)` is zeros. |
| `--opponent scripted` | Centerline follower below ego pace. |
| `--opponent policy --opponent-ckpt PATH` | Frozen-policy opponent. |
| `--self-play` | Snapshot the learner into a pool and refresh the policy opponent. |
| `--mixed-opponents` | Per-env mix of scripted + self-play (implies `--self-play`). Mix weights live in `config.py` `opponent_mix`. |
| `--selfplay-anchor-ckpt PATH` | Immutable incumbent that is never evicted from the pool. |

### Eval visualization (live + mp4)

`eval_visualize.py` runs a policy artifact deterministically in a single env and
renders a top-down view (track corridor, car + heading, breadcrumb trail, live
speed). It is **eval-only** — it never trains or touches a training run.

```bash
# Watch LIVE in a Rerun viewer:
python eval_visualize.py --checkpoint outputs/runs/<id>/checkpoints/policy_8000.pt --live

# Save an mp4:
python eval_visualize.py --checkpoint .../policy_8000.pt --mp4 outputs/eval.mp4

# Overlay many env instances (swarm view):
python eval_visualize.py --checkpoint .../policy_8000.pt --live --num-show 24

# Headless: stream to a saved .rrd instead of spawning a viewer:
python eval_visualize.py --checkpoint .../policy_8000.pt --live --no-spawn \
    --rr-save outputs/eval.rrd --mp4 outputs/eval.mp4
```

During training, emit periodic eval videos (separate 1-env instance, logged to
W&B; add `--eval-video-live` to also stream to Rerun):

```bash
python standalone_trainer.py --num-envs 4096 \
    --eval-interval-transitions 20480000 --eval-video-steps 600 \
    --eval-video-num-envs 16
```

Live view needs `rerun-sdk`; mp4 needs `opencv-python` + `imageio[ffmpeg]` (all in
the requirements files).

### After training

```bash
# Solo lap-time distribution across a run's checkpoints:
python analysis/lap_timing.py --run-dir outputs/runs/<id> --workers 8

# 1v1 triple-elim bracket (seed from the lap-timing CSV):
python analysis/tournament.py --run-dir outputs/runs/<id> --seed-csv <laps.csv>
```

Then certify the on-car graph in the gym: see [`../sim/README.md`](../sim/README.md)
and [`../docs/deployment.md`](../docs/deployment.md).

## Tests and lint

```bash
cd training
../.venv/bin/python -m pytest tests
../.venv/bin/python -m flake8 .
```

From the repo root: `PYTHONPATH=training .venv/bin/python -m pytest training/tests`.

## Pitfalls

- Default track is `ecss_clean`. The gym bridge and `tools/sim.sh validate` use
  `IV_2026_SIM`. A floor-trained policy is out of distribution in the gym.
- Observation is always 390-dim. Zeros in `[384:390)` mean “no relevant opponent”.
- Match deploy slip/load flags to training (`zero_tyre_slip_obs`,
  `enable_slip_estimation` / `enable_load_estimation` in the per-car overlay).
- macOS: Warp CPU tests only. Do not expect MPS training.
- Run from `training/` (or set `PYTHONPATH=training`). Repo-root
  `.venv/bin/python training/standalone_trainer.py` does not import `config` /
  `f1tenth_env` on its own.

## Offboard / HPC

Build an Apptainer image for HPC and run there — see
[`deploy/apptainer/training.def`](../deploy/apptainer/training.def) and
[`docs/deployment.md`](../docs/deployment.md). Outputs (checkpoints, logs, wandb) are
kept out of git (see [`.gitignore`](../.gitignore)).
