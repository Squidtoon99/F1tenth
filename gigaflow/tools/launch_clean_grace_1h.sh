#!/usr/bin/env bash
# Fresh-init 1h+ clean grace run (no quality stops).
set -euo pipefail
GFROOT="/home/ubuntu/.cursor/worktrees/F1tenth__SSH__darktoaster_/5e81/gigaflow"
export PYTHONPATH="$GFROOT/src"
PY="/home/ubuntu/miniconda3/envs/g2/bin/python"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$GFROOT/outputs/clean_grace_period/run_1h_${TS}"
mkdir -p "$RUN_DIR"
cp "$GFROOT/configs/clean_grace_rtx4080.yaml" "$RUN_DIR/run_config.yaml"
"$PY" - <<PY
import yaml
from pathlib import Path
p = Path("$RUN_DIR/run_config.yaml")
raw = yaml.safe_load(p.read_text())
raw["seed"] = 200
raw["wandb"]["name"] = "clean_grace_1h_${TS}"
p.write_text(yaml.dump(raw, sort_keys=False))
PY
date -u +%Y-%m-%dT%H:%M:%SZ | tee "$RUN_DIR/start.txt"
echo "$RUN_DIR" > "$GFROOT/outputs/clean_grace_period/active_run.txt"
exec "$PY" -m gigaflow_f1tenth.cli train \
  --config "$RUN_DIR/run_config.yaml" \
  --device cuda \
  --num-updates 500 \
  --run-dir "$RUN_DIR" \
  --checkpoint-interval 0 \
  2>&1 | tee "$RUN_DIR/train.log"
