#!/usr/bin/env bash
# Fast end-to-end repro for post-freeze CUDAGraph capture (~2 min loop).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING="$REPO/training"
PY="$REPO/.venv/bin/python"
RUN_ID="scratch-capture-repro"
RUN_DIR="$TRAINING/outputs/runs/$RUN_ID"
CONFIG="$TRAINING/outputs/experiments/long-horizon-2b-sensors/scratch-capture-repro.json"
CHAMPION="$TRAINING/outputs/runs/pcplus2b-a001/checkpoints/policy_1443840000.pt"
FREEZE="${FREEZE:-500000}"
TOTAL="${TOTAL:-800000}"
EXPANDABLE="${EXPANDABLE:-0}"

mkdir -p "$RUN_DIR"
: >"$RUN_DIR/run.log"

if [ "$EXPANDABLE" = "1" ]; then
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
else
  unset PYTORCH_CUDA_ALLOC_CONF
fi

echo "[$(date -Is)] scratch repro expandable=$EXPANDABLE freeze=$FREEZE total=$TOTAL"

cd "$TRAINING"
python3 - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("$CONFIG").read_text())
cfg["model"]["actor_freeze_transitions"] = int("$FREEZE")
cfg["schedule"]["total_transitions"] = int("$TOTAL")
out = Path("$RUN_DIR/config.json")
out.write_text(json.dumps(cfg, indent=2) + "\\n")
PY

cd "$TRAINING"
"$PY" standalone_trainer.py \
  --config "$RUN_DIR/config.json" \
  --init-ckpt "$CHAMPION" \
  --num-envs 1024 \
  --total-transitions "$TOTAL" \
  --opponent policy \
  --self-play \
  --selfplay-sample mixed \
  --selfplay-mixed-latest-prob 0.25 \
  --selfplay-pool-size 10 \
  --device cuda \
  --seed 42 \
  --compile \
  --compile-mode reduce-overhead \
  --run-id "$RUN_ID" \
  --no-wandb \
  >>"$RUN_DIR/run.log" 2>&1

echo "[$(date -Is)] scratch repro finished — checking gates"
grep -E "Actor unfreeze|Sequence CUDAGraph capture|Traceback|CUDNN" "$RUN_DIR/run.log" | tail -10
grep "standalone_trainer INFO: ticks=" "$RUN_DIR/run.log" | awk '
/transitions=4[89]|transitions=5[0-9]|transitions=6[0-9]|transitions=7[0-9]/' | tail -5
