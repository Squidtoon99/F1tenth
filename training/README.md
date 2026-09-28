# training/

Reinforcement-learning training code. This is **pure Python**: it is never built by
colcon and never included in the car image. Training runs natively on a Linux +
NVIDIA GPU box / HPC. macOS supports small Warp **CPU** tests, not MPS training.
There is intentionally **no GPU CI**.

Physics is NVIDIA Warp ([ADR 0007](../docs/adr/0007-warp-simulator.md)). PyTorch
is for the policy, replay, and optimizer only.

## Layout

The modules are top-level (imported as `config`, `run_layout`, `standalone_trainer`,
`f1tenth_env`, `qrsac`). Commands below run from the **repo root**.

- `f1tenth_env/` — env loop: observations, rewards, terminations, opponents,
  domain randomization. `F1tenthEnv` is `WarpF1tenthEnv`.
- `f1tenth_sim/` — Warp vehicle dynamics (Pacejka tyres, load transfer,
  wheel-spin, force drivetrain), batched on CPU or CUDA.
- `qrsac/` — Quantile-Regression Soft Actor-Critic plus spinning-up MLPs.
- `standalone_trainer.py` — single-process trainer (F1tenthEnv + in-memory n-step
  replay; no Reverb/Redis/S3).
- `run_layout.py` — per-run output (`outputs/runs/<run-id>/`).
- `config.py` — `DEFAULT_CONFIG`.
- `assets/` — bundled centerlines: `ecss_clean` (trainer default), `IV_2026_SIM`
  (gym / on-car cert map).
- `tests/` — learner, replay, Warp physics, observation, reward, termination,
  opponent, export, and evaluation tests.

## Setup

Put the venv at the **repo root** as `.venv`:

```bash
python3 -m venv .venv

# macOS (Apple Silicon) — CPU tests only:
.venv/bin/pip install -r training/requirements-mac.txt

# Linux + NVIDIA GPU (HPC / training rig):
.venv/bin/pip install -r training/requirements-gpu.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130
```

Both files install `libs/f1tenth_contract` editable, so format changes need no
reinstall. Warp is pinned to 1.15.

```bash
.venv/bin/python training/standalone_trainer.py --help
.venv/bin/python -m pytest training/tests
.venv/bin/python -m flake8 training
```

## Train a new model

Budgets are cumulative environment transitions. Default learner samples two
replay rows per collected transition. Domain randomization is always on in
training; eval and deterministic tests use the nominal profile.

Exports under `outputs/runs/<id>/checkpoints/` keep `actor` + `obs_norm`,
transition count, dimensions, action scaling, and `policy_format_version` 2.
Critic and optimizer state are not persisted; runs do not resume. Warm-start
with `--init-ckpt`.

### 1. Solo (1v0)

```bash
.venv/bin/python training/standalone_trainer.py \
  --device cuda --precision 32 --num-envs 4096 \
  --track ecss_clean
```

For a checkpoint that must pass the gym gate, use the gym map:

```bash
.venv/bin/python training/standalone_trainer.py \
  --device cuda --precision 32 --num-envs 4096 \
  --track IV_2026_SIM
```

`--track` also accepts a bundled name, a GitHub `f1tenth_racetracks` name (fetched
on first run), or a centerline CSV path.

### 2. Head-to-head

```bash
# Scripted slow centerline opponent:
.venv/bin/python training/standalone_trainer.py \
  --device cuda --num-envs 4096 --opponent scripted --track IV_2026_SIM

# Delayed self-play (frozen snapshots of the learner):
.venv/bin/python training/standalone_trainer.py \
  --device cuda --num-envs 4096 --self-play --track IV_2026_SIM

# Mixed scripted + self-play (GT Sophy-style population):
.venv/bin/python training/standalone_trainer.py \
  --device cuda --num-envs 4096 --mixed-opponents --track IV_2026_SIM
```

`--selfplay-anchor-ckpt` adds an immutable incumbent that is never evicted from
the rolling pool. Mix weights live in `config.py` `env.opponent_mix`.

### Constraints

- `--device` is `auto|cpu|cuda`. There is no MPS path. `--precision` is `32` only.
- Action is force/current (`longitudinal_mode: force`). Speed-trained artifacts
  will not load on the car.
- Observation is always 390-dim. `--zero-tyre-slip-obs` zeros `[372:380)` if the
  car will not publish slip.
- macOS: keep `--num-envs` small (tens, not thousands).
- `torch.compile` is on by default; `--no-compile` is the deterministic test path.

## Eval visualization (live + mp4)

`eval_visualize.py` rolls a policy in a single env and draws a top-down view. It
never trains. Same rollout path as in-training eval.

```bash
.venv/bin/python training/eval_visualize.py \
  --checkpoint outputs/runs/<id>/checkpoints/policy_8000.pt --live

.venv/bin/python training/eval_visualize.py \
  --checkpoint .../policy_8000.pt --mp4 outputs/eval.mp4

.venv/bin/python training/eval_visualize.py \
  --checkpoint .../policy_8000.pt --live --num-show 24
```

During training, periodic eval video:

```bash
.venv/bin/python training/standalone_trainer.py --num-envs 4096 \
    --eval-interval-transitions 20480000 --eval-video-steps 600 \
    --eval-video-num-envs 16
```

Live view needs `rerun-sdk`; mp4 needs `opencv-python` + `imageio[ffmpeg]` (in
the requirements files).

## Offboard / HPC

```bash
apptainer build training.sif deploy/apptainer/training.def
apptainer run --nv training.sif --num-envs 4096 --total-transitions 256000000
```

See [`deploy/apptainer/training.def`](../deploy/apptainer/training.def) and
[`docs/deployment.md`](../docs/deployment.md). Checkpoints, logs, and wandb stay
out of git.
