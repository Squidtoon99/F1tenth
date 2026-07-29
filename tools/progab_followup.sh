#!/usr/bin/env bash
# Launch post-verdict follow-up (single long run, Restart=no via systemd-run).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING="$REPO/training"
PY="$REPO/.venv/bin/python"
EXPERIMENT_DIR="$TRAINING/outputs/experiments/long-horizon-2b-sensors"
FLOG="$TRAINING/outputs/runs/progab-followup.log"

OUTCOME="${1:?usage: progab_followup.sh win|null|harm|void [threshold]}"
EXTRA="${2:-4.5}"

log() {
  echo "[$(date -Is)] progab-followup: $*" | tee -a "$FLOG"
}

case "$OUTCOME" in
  win)
    RUN_ID="progab-super-extend-a001"
    CONFIG="$EXPERIMENT_DIR/progab-super-noise-2b-a001.json"
    CKPT_DIR="$TRAINING/outputs/runs/progab-super-a001/checkpoints"
    INIT=$(ls "$CKPT_DIR"/policy_*.pt 2>/dev/null | sort -V | tail -1 || true)
    TARGET=2000000000
    TAGS="progab,treatment,extend,2B,from-noise"
    USE_INIT=1
    ;;
  null|harm)
    RUN_ID="pcplus2b-a002"
    CONFIG="$EXPERIMENT_DIR/pcplus2b-a001.json"
    INIT=""
    TARGET=2000000000
    TAGS="pcplus2b,reproduction,from-noise,2B,followup"
    USE_INIT=0
    ;;
  void)
    RUN_ID="progab-super-lowthr-a001"
    CONFIG="$EXPERIMENT_DIR/progab-super-lowthr-a001.json"
    python3 - "$CONFIG" "$EXTRA" <<'PY'
import json, sys
from pathlib import Path
path, thr = Path(sys.argv[1]), float(sys.argv[2])
cfg = json.loads(path.read_text())
cfg["reward"]["progress_speed_threshold_mps"] = thr
path.write_text(json.dumps(cfg, indent=2) + "\n")
PY
    INIT=""
    TARGET=2000000000
    TAGS="progab,treatment,low-threshold,${EXTRA}mps,2B,void-followup"
    USE_INIT=0
    ;;
  *)
    log "ERROR unknown outcome $OUTCOME"
    exit 2
    ;;
esac

RUN_DIR="$TRAINING/outputs/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
log "outcome=$OUTCOME run_id=$RUN_ID target=$TARGET init=${INIT:-none}"

CMD=(
  "$PY" standalone_trainer.py
  --config "$CONFIG"
  --num-envs 1024
  --total-transitions "$TARGET"
  --opponent policy
  --self-play
  --selfplay-sample mixed
  --selfplay-mixed-latest-prob 0.25
  --selfplay-pool-size 10
  --device cuda
  --seed 42
  --compile
  --compile-mode reduce-overhead
  --run-id "$RUN_ID"
  --wandb
  --wandb-mode online
)
if [ "$USE_INIT" = 1 ] && [ -n "$INIT" ] && [ -f "$INIT" ]; then
  CMD+=(--init-ckpt "$INIT")
fi

systemctl --user stop f1tenth-progab-a001.service 2>/dev/null || true
sleep 3

systemd-run --user \
  --unit=f1tenth-progab-followup.service \
  --description="F1TENTH progab follow-up $OUTCOME ($RUN_ID)" \
  --working-directory="$TRAINING" \
  -p Restart=no \
  --setenv=WANDB_MODE=online \
  --setenv=WANDB_NAME="$RUN_ID" \
  --setenv=WANDB_GROUP=long-horizon-2b-sensors \
  --setenv=WANDB_TAGS="$TAGS" \
  --setenv=PYTHONUNBUFFERED=1 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash -lc "$(printf '%q ' "${CMD[@]}") >> $(printf '%q' "$RUN_DIR/run.log") 2>&1"

log "systemd unit f1tenth-progab-followup.service started"
