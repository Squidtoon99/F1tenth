#!/usr/bin/env bash
# Sequential progab A/B: Arm A (control) then Arm B (treatment), one GPU tenant.
# PROGAB_MODE=warm (default): champion warm-start, 300M/arm, LR ramp after freeze.
# PROGAB_MODE=noise: from-noise pcplus2b recipe, 600M/arm, no init ckpt.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING="$REPO/training"
PY="$REPO/.venv/bin/python"
EXPERIMENT_DIR="$TRAINING/outputs/experiments/long-horizon-2b-sensors"
CHAMPION="$TRAINING/outputs/runs/pcplus2b-a001/checkpoints/policy_1443840000.pt"
PROGAB_MODE="${PROGAB_MODE:-warm}"

log() {
  echo "[$(date -Is)] progab-chain ($PROGAB_MODE): $*"
}

case "$PROGAB_MODE" in
  warm)
    TARGET_TRANSITIONS=300000000
    CONTROL_CONFIG="$EXPERIMENT_DIR/progab-control-a001.json"
    SUPER_CONFIG="$EXPERIMENT_DIR/progab-super-a001.json"
    TAG_SUFFIX="warm-start,300M,attempt6"
    ;;
  noise)
    TARGET_TRANSITIONS=600000000
    CONTROL_CONFIG="$EXPERIMENT_DIR/progab-control-noise-a001.json"
    SUPER_CONFIG="$EXPERIMENT_DIR/progab-super-noise-a001.json"
    TAG_SUFFIX="from-noise,600M"
    ;;
  *)
    log "ERROR: unknown PROGAB_MODE=$PROGAB_MODE (use warm or noise)"
    exit 2
    ;;
esac

latest_transitions() {
  local log_file="$1"
  python3 - "$log_file" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print(0)
    raise SystemExit(0)
vals = re.findall(
    r"standalone_trainer INFO: ticks=\d+ transitions=(\d+)",
    path.read_text(errors="replace"),
)
print(int(vals[-1]) if vals else 0)
PY
}

verify_arm_complete() {
  local run_id="$1"
  local log_file="$TRAINING/outputs/runs/$run_id/run.log"
  if [ ! -f "$log_file" ]; then
    log "ERROR: missing log for $run_id ($log_file)"
    return 1
  fi
  if ! grep -q "Training finished" "$log_file"; then
    log "ERROR: $run_id log has no 'Training finished' line"
    return 1
  fi
  local trans
  trans=$(latest_transitions "$log_file")
  if [ "$trans" -lt "$TARGET_TRANSITIONS" ]; then
    log "ERROR: $run_id finished at $trans transitions (need >= $TARGET_TRANSITIONS)"
    return 1
  fi
  log "$run_id verified: Training finished at $trans transitions"
  return 0
}

run_arm() {
  local run_id="$1"
  local config="$2"
  local wandb_name="$3"
  local wandb_tags="$4"

  local run_dir="$TRAINING/outputs/runs/$run_id"
  mkdir -p "$run_dir"

  log "starting $run_id (config=$config)"
  (
    cd "$TRAINING"
    export WANDB_MODE=online
    export WANDB_NAME="$wandb_name"
    export WANDB_GROUP=long-horizon-2b-sensors
    export WANDB_TAGS="$wandb_tags"
    export PYTHONUNBUFFERED=1
    if [ "$PROGAB_MODE" = "warm" ]; then
      exec "$PY" standalone_trainer.py \
        --config "$config" \
        --init-ckpt "$CHAMPION" \
        --num-envs 1024 \
        --total-transitions "$TARGET_TRANSITIONS" \
        --opponent policy \
        --self-play \
        --selfplay-sample mixed \
        --selfplay-mixed-latest-prob 0.25 \
        --selfplay-pool-size 10 \
        --device cuda \
        --seed 42 \
        --compile \
        --compile-mode reduce-overhead \
        --run-id "$run_id" \
        --wandb \
        --wandb-mode online
    else
      exec "$PY" standalone_trainer.py \
        --config "$config" \
        --num-envs 1024 \
        --total-transitions "$TARGET_TRANSITIONS" \
        --opponent policy \
        --self-play \
        --selfplay-sample mixed \
        --selfplay-mixed-latest-prob 0.25 \
        --selfplay-pool-size 10 \
        --device cuda \
        --seed 42 \
        --compile \
        --compile-mode reduce-overhead \
        --run-id "$run_id" \
        --wandb \
        --wandb-mode online
    fi
  ) >>"$run_dir/run.log" 2>&1
}

log "progab A/B chain starting (mode=$PROGAB_MODE target=${TARGET_TRANSITIONS})"

run_arm \
  "progab-control-a001" \
  "$CONTROL_CONFIG" \
  "progab-control-a001-seed42-${PROGAB_MODE}" \
  "progab,progress-shape,control,seed42,rtx4080super,${TAG_SUFFIX}"

control_rc=$?
if [ "$control_rc" -ne 0 ]; then
  log "ERROR: Arm A (control) exited $control_rc — not starting Arm B"
  exit "$control_rc"
fi

if ! verify_arm_complete "progab-control-a001"; then
  log "ERROR: Arm A completion checks failed — not starting Arm B"
  exit 1
fi

run_arm \
  "progab-super-a001" \
  "$SUPER_CONFIG" \
  "progab-super-a001-seed42-${PROGAB_MODE}" \
  "progab,progress-shape,treatment,superlinear,seed42,rtx4080super,${TAG_SUFFIX}"

super_rc=$?
if [ "$super_rc" -ne 0 ]; then
  log "ERROR: Arm B (treatment) exited $super_rc"
  exit "$super_rc"
fi

if ! verify_arm_complete "progab-super-a001"; then
  log "ERROR: Arm B completion checks failed"
  exit 1
fi

log "progab A/B chain complete (both arms reached ${TARGET_TRANSITIONS})"
exit 0
