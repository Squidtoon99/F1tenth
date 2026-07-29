#!/usr/bin/env bash
# 100M from-noise validation: pcplus2b-a001 recipe on scalar-LR fix (do not chain).
# Reference @100M: pcplus2b-a001 4.26 m/s @ 26.7 s; regressed arm 2.72 @ 17.0 s.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING="$REPO/training"
PY="$REPO/.venv/bin/python"
CONFIG="$TRAINING/outputs/experiments/long-horizon-2b-sensors/scalar-lr-fix-validate-a001.json"
RUN_ID="${RUN_ID:-scalar-lr-fix-validate-a001}"
SEED="${SEED:-42}"

log() {
  echo "[$(date -Is)] scalar-lr-fix-validate: $*"
}

run_dir="$TRAINING/outputs/runs/$RUN_ID"
mkdir -p "$run_dir"

log "starting $RUN_ID seed=$SEED config=$CONFIG"
(
  cd "$TRAINING"
  export WANDB_MODE=online
  export WANDB_NAME="${RUN_ID}-seed${SEED}"
  export WANDB_GROUP=long-horizon-2b-sensors
  export WANDB_TAGS="scalar-lr-fix,validate,from-noise,100M,seed${SEED},pcplus2b-recipe"
  export PYTHONUNBUFFERED=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  exec "$PY" standalone_trainer.py \
    --config "$CONFIG" \
    --num-envs 1024 \
    --total-transitions 100000000 \
    --opponent policy \
    --self-play \
    --selfplay-sample mixed \
    --selfplay-mixed-latest-prob 0.25 \
    --selfplay-pool-size 10 \
    --device cuda \
    --seed "$SEED" \
    --compile \
    --compile-mode reduce-overhead \
    --run-id "$RUN_ID" \
    --wandb \
    --wandb-mode online
) >>"$run_dir/run.log" 2>&1
