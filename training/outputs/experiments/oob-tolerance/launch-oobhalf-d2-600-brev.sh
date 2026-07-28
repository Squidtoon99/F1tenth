#!/usr/bin/env bash
set -euo pipefail
HOST="${1:?usage: launch-oobhalf-d2-600-brev.sh <brev-host>}"
REPO="/home/shadeform/F1tenth"
RUN_ID="oobhalf-d2-600"
CONFIG="outputs/experiments/oob-tolerance/oobhalf-d2-600.json"

brev exec "$HOST" bash -lc "
set -euo pipefail
cd $REPO/training
git fetch origin experiments/e2e-sim-sensors 2>/dev/null || true
git checkout experiments/e2e-sim-sensors 2>/dev/null || true
bash outputs/experiments/causal-2x2/apply-overlay-brev.sh
mkdir -p outputs/runs/$RUN_ID
setsid nohup ../.venv/bin/python standalone_trainer.py \
  --config $CONFIG \
  --num-envs 1024 \
  --total-transitions 600000000 \
  --opponent policy \
  --device cuda \
  --seed 7 \
  --compile \
  --compile-mode reduce-overhead \
  --run-id $RUN_ID \
  --wandb \
  --wandb-mode online \
  > outputs/runs/$RUN_ID/run.log 2>&1 &
disown
sleep 15
pgrep -af standalone_trainer || echo 'FAILED TO START'
tail -5 outputs/runs/$RUN_ID/run.log 2>/dev/null || true
"
