# gigaflow — isolated F1TENTH self-play

Standalone multi-track, multi-car self-play project. It does **not** depend on
`training/` or `libs/f1tenth_policy` at runtime. Design authority:
[DESIGN.md](DESIGN.md). Repo decision record: [ADR 0027](../docs/adr/0027-gigaflow-isolated-selfplay.md).

## Layout

```text
gigaflow/
  configs/               # versioned experiment YAML + track_pin.json
  src/gigaflow_f1tenth/  # flat runtime modules (+ viewer/ replay server)
  viewer/                # Vite + Three.js live WebGL UI
  tests/                 # unit + integration tests
  benchmarks/            # throughput harnesses
  tools/                 # isolation checks, helpers
  DESIGN.md              # shapes, masks, seams
```

## Setup

```bash
cd gigaflow
pip install -r requirements-cpu.txt   # or requirements-gpu.txt on CUDA hosts
pip install -e .
```

## Config validation

```bash
gigaflow validate-config --config configs/smoke.yaml
python -m gigaflow_f1tenth validate-config --config configs/default.yaml
```

Startup validation checks sensor layout parity (1097-D), control cadence
(`20 × 0.005 s = 10 Hz`), PPO constraints, ablation seams, and a rough memory budget.

## Track atlas preparation

Upstream `f1tenth/f1tenth_racetracks` assets are **GPL-3.0**. They are downloaded
against the pinned revision/checksums in [`configs/track_pin.json`](configs/track_pin.json)
into a cache directory outside git (never vendored into this MIT tree). The pin
ships **23** centerlines; `TracksConfig.num_tracks` defaults to 23 to match.

```bash
# Full pinned atlas (network + checksum). Cache stays outside the repo.
gigaflow prepare-tracks \
  --config configs/default.yaml \
  --cache-dir "$HOME/.cache/gigaflow/tracks"

# Smoke / offline: place fixture CSVs in <cache>/local/*_centerline.csv
mkdir -p /tmp/gigaflow-tracks/local
cp tests/fixtures/tracks/oval_centerline.csv /tmp/gigaflow-tracks/local/
gigaflow prepare-tracks \
  --config configs/smoke.yaml \
  --cache-dir /tmp/gigaflow-tracks \
  --skip-download \
  --lut-resolution 0.5 \
  --edt-resolution 0.25
```

## Runtime: Warp sim + PyTorch PPO

Performance-critical non-neural paths (dynamics, contact, LiDAR, Frenet,
reset/spawn, rewards, packing/reconstruction, critic features) run as NVIDIA
Warp kernels over device-resident buffers with zero-copy Torch interop. CPU
paths remain for tests/reference and offline track preparation only. Actor/critic
PPO stays compiled PyTorch (AMP / `torch.compile` where configured).

```bash
# Assert CUDA steps stay device-resident (no .cpu()/.numpy() in hot paths)
pytest tests/test_device_residency.py
python tools/check_warp_residency.py
```

## Train / evaluate / soak / benchmark

```bash
# Short smoke training (CPU-friendly configs/smoke.yaml)
gigaflow train --config configs/smoke.yaml --smoke --run-dir /tmp/gigaflow-run

# Fixed-seed evaluation + multi-car diagnostics
gigaflow evaluate --config configs/smoke.yaml --output-dir /tmp/gigaflow-eval

# Random-action soak
gigaflow soak --config configs/smoke.yaml --steps 40 --output-dir /tmp/gigaflow-soak

# Throughput
gigaflow benchmark --config configs/smoke.yaml --mode sim --steps 20
gigaflow benchmark --config configs/smoke.yaml --mode learner --steps 1
```

### Optional Weights & Biases

W&B is **off by default**. Local `metrics_*.json` / checkpoints stay authoritative
if W&B is disabled or a recoverable log failure occurs. Enable via config `wandb:`
or CLI. Never put API keys in YAML.

**Logged at `profiling.report_interval_updates` (async, no hot-path sync):**
PPO parity/KL/entropy/loss/grad/filter, sim reward/progress/collision/OOB/reset,
track/density, throughput, GPU/memory; plus config + provenance (git SHA, track
manifest, hardware). Checkpoints/actor paths are referenced on save. Optional
mid-train eval/media via `wandb.eval_interval_updates`. When
`evaluation.device: cpu`, cadenced eval spawns an isolated CPU subprocess
(hidden CUDA, actor-only snapshot, at most one in-flight) and the parent later
uploads local `eval_*/` artifacts to the same W&B run.

```bash
# Offline smoke (no network / no login)
gigaflow train --config configs/smoke.yaml --smoke \
  --run-dir /tmp/gigaflow-run \
  --wandb --wandb-mode offline \
  --wandb-project f1tenth-gigaflow --wandb-name local_smoke \
  --wandb-tag smoke --wandb-tag cpu

# Next corrected H100 production run (online; host must already be logged in)
# Entity is inferred from the authenticated account unless --wandb-entity is set.
# Never put API keys in YAML.
RUN_ID=gigaflow_h100_$(date +%Y%m%d_%H%M%S)
RUN_DIR=/tmp/gigaflow_runs/$RUN_ID
mkdir -p "$RUN_DIR"
gigaflow train --config configs/production_h100.yaml --device cuda \
  --run-dir "$RUN_DIR" --checkpoint-interval 10 \
  --wandb --wandb-mode online \
  --wandb-project f1tenth-gigaflow \
  --wandb-group prod_h100 --wandb-name "$RUN_ID" \
  --wandb-tag production --wandb-tag h100 \
  --wandb-resume allow

# Exact resume — same W&B run id (from ckpt / run_meta.json); no duplicate run
gigaflow train --config configs/production_h100.yaml --device cuda \
  --run-dir "$RUN_DIR" --resume-from "$RUN_DIR/ckpt_final.pt" \
  --checkpoint-interval 10 \
  --wandb --wandb-mode online --wandb-resume must \
  --wandb-project f1tenth-gigaflow
```

### Production all-track launch

Validate gates first (`tools/validate_launch.py`), then pick a host-local
production config. RTX 4080 SUPER (16 GiB) scale, measured on the card:
`configs/production_rtx4080.yaml` (48 worlds × 8 agents, 4.9 GiB allocated peak,
11.1 GiB of 16 GiB device memory in use). H100 starting point:
`configs/production_h100.yaml`.

```bash
# Full validate-launch harness (atlas, invariants, soaks, trial)
python tools/validate_launch.py --cache-dir "$HOME/.cache/gigaflow/tracks" \
  --report /tmp/gigaflow_validate/validate_launch_report.json

# Production train (example)
RUN_ID=gigaflow_prod_$(date +%Y%m%d_%H%M%S)
RUN_DIR=/tmp/gigaflow_runs/$RUN_ID
mkdir -p "$RUN_DIR"
nohup python -m gigaflow_f1tenth train \
  --config configs/production_rtx4080.yaml --device cuda \
  --run-dir "$RUN_DIR" --checkpoint-interval 10 \
  > "$RUN_DIR/train.log" 2>&1 &
```

Resume from the latest checkpoint in `$RUN_DIR` via the trainer checkpoint
loader (see `DESIGN.md`). Monitor with `tail -f "$RUN_DIR/train.log"` and
`nvidia-smi`.

**Stopping a run.** The trainer does not install a `SIGTERM` handler and does
not flush a checkpoint on shutdown, so killing it loses at most
`--checkpoint-interval` updates of progress (whatever landed since the last
periodic `ckpt_*.pt`/`actor_*.pt`, or none if `--checkpoint-interval 0`). Stop
it with `kill <PID>` against the exact PID from launch — never a pattern-based
`pkill`, which can also match its own invoking command line or an unrelated
process. Confirm the process is gone (`ps -p <PID>`) and that the run directory
still has the checkpoint you expect before reusing the GPU.

## Live checkpoint viewer (local replay)

Runs a deterministic solo / head-to-head / dense sim from a local actor
checkpoint and streams track geometry + car poses over WebSocket. Training /
remote H100 processes are never touched. Switch track and density suite from
the UI without restarting the server.

```bash
# One-time UI build
cd viewer && npm install && npm run build && cd ..

# Example: local actor + prepared atlas + production-shaped config
gigaflow view \
  --checkpoint .cache/checkpoints/actor_000900.pt \
  --config configs/production_h100.yaml \
  --cache-dir "$HOME/.cache/gigaflow/tracks" \
  --track Austin \
  --suite solo \
  --seed 0 \
  --device cpu \
  --host 127.0.0.1 \
  --ws-port 8765 \
  --http-port 8766
```

Open `http://127.0.0.1:8766/`. Controls: pause/resume, reset, track, suite,
checkpoint, follow car, camera (`follow` tight / `dynamic` chase), whole-track,
orbit drag. The checkpoint selector lists every `actor_*.pt` found under
`--checkpoint-root` (repeatable; defaults to `outputs/`), grouped by the run
directory holding it, and served as JSON at `/api/checkpoints`. Picking one
hot-swaps the live policy and resets the GRU state; an incompatible checkpoint is
reported in the UI while the previous policy keeps running.
Fails clearly on missing atlas or checkpoint/config architecture mismatch.
Checkpoint pulling from Brev is intentionally not part of this command.

## Dense-traffic experiment (optional)

Queued A/B for denser worlds (`max_agents` 8 vs 10 vs 12) without changing
production defaults. See [`configs/experiments/README.md`](configs/experiments/README.md).

```bash
# CPU-only validation + capacity-aware atlas variants + H100 queue JSON
CUDA_VISIBLE_DEVICES= python tools/dense_traffic_experiment.py \
  --rebuild-atlases \
  --output-dir outputs/dense_traffic_experiment
```

## Tests / lint

```bash
pytest
python tools/check_isolation.py
flake8 src tests tools
cd viewer && npm run typecheck && npm test && npm run build
```

## Deviations from the reference paper

This follows the GT Sophy approach (Wurman et al., *Nature* 2022) but is not a
1:1 reimplementation. Known, deliberate deviations:

- **Reward set.** Only the racing subset is implemented: uncapped Frenet
  progress, collision, boundary, bounded-linear lane-center shaping, and
  conditioned N-car passing. The paper's urban goal/stop-line, lane-align,
  reverse, velocity, and active-timestep terms are omitted; finish-rank,
  blocking, and zero-sum are disabled by default. See `rewards.py`.
- **Private condition schema.** `CONDITION_DIM` is 10 (5 reward-weight alphas +
  5 estimable dynamics scales), a reduced schema matching the racing-only
  reward set, not the paper's full private-condition vector.
- **Critic inputs.** The critic's ego branch concatenates a `K=20` ego-relative
  track preview (centerline samples, widths, curvature; see the
  `sample_track_lookahead` speed-scaled lookahead in `tracks.py`) alongside the
  compact state, as an ordered feature — never a Deep Sets element, since
  max-pooling would destroy arc order. The opponent set it pools stays
  unchanged. The actor stays LiDAR-only, so the deploy contract is unaffected.
- **Static opponents in training.** The paper models static obstacles as
  immobile agents. That placement/pinning logic exists in the live checkpoint
  viewer today; landing it in the training loop itself is tracked separately.
- **Sensor observation normalization.** The paper normalizes every observation
  to `[-1, 1]`. Gigaflow tracks a running per-dimension mean/std (Welford) over
  the 1097-D sensor vector and maps to a `[-1, 1]` clipped z-score (`+/-5`
  running standard deviations), rather than fixed, known per-channel bounds.
  Owned by the actor (`normalization.py`) so it is applied identically in
  collection, reconstruction, and any loaded checkpoint or deployable artifact.
- **Recurrent actor vs. the paper's feed-forward one.** The paper's advantage
  filter drops ~80% of transitions outright; our GRU actor must still score
  every transition in a kept sequence to preserve recurrence, so it does
  roughly 5x more actor work per filtered rollout. A chunked-BPTT mitigation is
  tracked separately (see the audit remediation plan's D4).

Full architectural detail lives in [DESIGN.md](DESIGN.md).

## Status

Integrated one-machine loop: Warp device-resident sim, compact-state collection,
sensor/critic reconstruction, conditioned rewards, recurrent PPO with
carry/reset, exact resume checkpoints, evaluation suites, visualization, soak,
and validate-launch gates. Production configs:
`configs/production_rtx4080.yaml`, `configs/production_h100.yaml`.
