#!/usr/bin/env bash
# Launch 1h+ clean grace run using filter_matrix_summary selected config.
set -euo pipefail
GFROOT="/home/ubuntu/.cursor/worktrees/F1tenth__SSH__darktoaster_/5e81/gigaflow"
export PYTHONPATH="$GFROOT/src"
PY="/home/ubuntu/miniconda3/envs/g2/bin/python"
SUMMARY="$GFROOT/outputs/clean_grace_period/filter_matrix_summary.json"
CONFIG="${1:-}"
if [[ -z "$CONFIG" ]]; then
  CONFIG="$("$PY" - <<PY
import json
from pathlib import Path
s = json.loads(Path("$SUMMARY").read_text())
print(s["selected_1h_config"])
PY
)"
fi
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$GFROOT/outputs/clean_grace_period/run_1h_${TS}"
mkdir -p "$RUN_DIR"
cp "$CONFIG" "$RUN_DIR/run_config.yaml"
"$PY" - <<PY
import yaml
from pathlib import Path
p = Path("$RUN_DIR/run_config.yaml")
raw = yaml.safe_load(p.read_text())
raw["seed"] = 200
raw["wandb"]["name"] = "clean_grace_1h_${TS}"
raw["wandb"]["group"] = "clean-grace-rtx4080"
raw["wandb"]["tags"] = ["clean-grace", "rtx4080", "fresh-init", "no-quality-stop", "filter-selected"]
p.write_text(yaml.dump(raw, sort_keys=False))
PY
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$RUN_DIR/start.txt"
echo "$RUN_DIR" > "$GFROOT/outputs/clean_grace_period/active_run.txt"
setsid "$PY" -m gigaflow_f1tenth.cli train \
  --config "$RUN_DIR/run_config.yaml" \
  --device cuda \
  --num-updates 500 \
  --run-dir "$RUN_DIR" \
  --checkpoint-interval 0 \
  >> "$RUN_DIR/train.log" 2>&1 < /dev/null &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$GFROOT/outputs/clean_grace_period/active_pid.txt"
{
  echo "run_dir=$RUN_DIR"
  echo "pid=$TRAIN_PID"
  echo "config=$CONFIG"
  echo "started=$(cat "$RUN_DIR/start.txt")"
  echo "stop: kill $TRAIN_PID"
} | tee "$GFROOT/outputs/clean_grace_period/active_run_info.txt"
echo "Launched PID $TRAIN_PID -> $RUN_DIR"
