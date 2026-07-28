#!/usr/bin/env bash
# Launch oobhalf300 on a Brev host after applying causal-2x2 overlay.
set -euo pipefail
HOST="${1:?usage: launch-oobhalf300-brev.sh <brev-host>}"
REPO="/home/shadeform/F1tenth"
RUN_ID="oobhalf300"
CONFIG="outputs/experiments/oob-tolerance/oobhalf300.json"

brev exec "$HOST" bash -lc "
set -euo pipefail
cd $REPO
git fetch origin experiments/e2e-sim-sensors 2>/dev/null || true
git checkout experiments/e2e-sim-sensors 2>/dev/null || true
bash training/outputs/experiments/causal-2x2/apply-overlay-brev.sh
mkdir -p training/outputs/runs/$RUN_ID
setsid nohup ../.venv/bin/python standalone_trainer.py \
  --config $CONFIG \
  --num-envs 1024 \
  --total-transitions 300000000 \
  --opponent policy \
  --self-play \
  --selfplay-mixed-latest-prob 0.25 \
  --device cuda \
  --seed 42 \
  --compile \
  --compile-mode reduce-overhead \
  --run-id $RUN_ID \
  --wandb \
  --wandb-mode online \
  > training/outputs/runs/$RUN_ID/run.log 2>&1 &
disown
sleep 3
pgrep -af standalone_trainer || echo 'FAILED TO START'
tail -3 training/outputs/runs/$RUN_ID/run.log 2>/dev/null || true
"
