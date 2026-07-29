#!/usr/bin/env bash
# Fill idle GPU time by launching queued training runs when no trainer is active
# and utilization stays low across several consecutive checks. Never preempts a
# healthy job; uses a lockfile so it cannot race existing launch watchers.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/gpu-idle-supervisor"
LOG="$STATE_DIR/supervisor.log"
IDLE_COUNT_FILE="$STATE_DIR/idle_count"
COMPLETED_FILE="$STATE_DIR/completed.txt"
ABANDONED_FILE="$STATE_DIR/abandoned.txt"
FAILURES_FILE="$STATE_DIR/failures.json"
PENDING_VERIFY_FILE="$STATE_DIR/pending_verify.json"
LOCK_FILE="$STATE_DIR/launch.lock"
QUEUE="${GPU_IDLE_QUEUE:-$REPO/tools/gpu_idle_queue.json}"
PY="$REPO/.venv/bin/python"
TRAINING="$REPO/training"
CHAMPION_CKPT="$TRAINING/outputs/runs/642a7a80/checkpoints/policy_199680000.pt"

IDLE_GPU_UTIL_MAX="${IDLE_GPU_UTIL_MAX:-15}"
IDLE_CHECKS_TO_LAUNCH="${IDLE_CHECKS_TO_LAUNCH:-3}"
LAUNCH_VERIFY_SEC="${LAUNCH_VERIFY_SEC:-60}"
MAX_LAUNCH_FAILURES="${MAX_LAUNCH_FAILURES:-2}"
DRY_RUN=0
SIMULATE_IDLE=0

usage() {
  echo "usage: $(basename "$0") [--dry-run] [--simulate-idle]" >&2
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --simulate-idle) SIMULATE_IDLE=1; shift ;;
    -h|--help) usage ;;
    *) usage ;;
  esac
done

mkdir -p "$STATE_DIR"
touch "$COMPLETED_FILE" "$ABANDONED_FILE"
[ -f "$FAILURES_FILE" ] || echo '{}' >"$FAILURES_FILE"

NVIDIA_SMI=""

log() {
  echo "[$(date -Is)] $*" >>"$LOG"
}

resolve_nvidia_smi() {
  if [ -n "$NVIDIA_SMI" ]; then
    return 0
  fi
  local candidate
  for candidate in \
    "$(command -v nvidia-smi 2>/dev/null || true)" \
    /usr/lib/wsl/lib/nvidia-smi \
    /usr/bin/nvidia-smi \
    /usr/local/bin/nvidia-smi; do
    [ -n "$candidate" ] && [ -x "$candidate" ] || continue
    NVIDIA_SMI="$candidate"
    return 0
  done
  return 1
}

trainer_count() {
  if [ "$SIMULATE_IDLE" -eq 1 ]; then
    echo 0
    return
  fi
  local count
  count=$(pgrep -cf '[.]venv/bin/python.*standalone_trainer\.py' 2>/dev/null || true)
  if [ "${count:-0}" -eq 0 ]; then
    count=$(pgrep -cf 'python.*[s]tandalone_trainer\.py' 2>/dev/null || true)
  fi
  echo "${count:-0}"
}

gpu_util_pct() {
  if [ "$SIMULATE_IDLE" -eq 1 ]; then
    echo 0
    return 0
  fi
  if ! resolve_nvidia_smi; then
    log "ERROR: nvidia-smi not found (PATH and /usr/lib/wsl/lib/nvidia-smi); refusing idle launch"
    return 1
  fi
  local raw util
  raw=$("$NVIDIA_SMI" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1) || {
    log "ERROR: nvidia-smi query failed ($NVIDIA_SMI); refusing idle launch"
    return 1
  }
  util=$(printf '%s' "$raw" | tr -dc '0-9')
  if [ -z "$util" ]; then
    log "ERROR: nvidia-smi returned unreadable utilization (raw=${raw:-empty}); refusing idle launch"
    return 1
  fi
  echo "$util"
  return 0
}

reset_idle_count() {
  echo 0 >"$IDLE_COUNT_FILE"
}

read_idle_count() {
  cat "$IDLE_COUNT_FILE" 2>/dev/null || echo 0
}

is_completed() {
  local run_id="$1"
  grep -qx "$run_id" "$COMPLETED_FILE" 2>/dev/null
}

mark_completed() {
  local run_id="$1"
  grep -qx "$run_id" "$COMPLETED_FILE" 2>/dev/null || echo "$run_id" >>"$COMPLETED_FILE"
}

is_abandoned() {
  local job_key="$1"
  grep -qx "$job_key" "$ABANDONED_FILE" 2>/dev/null && return 0
  "$PY" - "$QUEUE" "$job_key" <<'PY'
import json, sys
from pathlib import Path
queue_path, job_key = sys.argv[1:3]
data = json.loads(Path(queue_path).read_text())
if job_key in data.get("abandoned_run_ids", []):
    sys.exit(0)
sys.exit(1)
PY
}

mark_abandoned() {
  local job_key="$1"
  grep -qx "$job_key" "$ABANDONED_FILE" 2>/dev/null || echo "$job_key" >>"$ABANDONED_FILE"
}

failure_count() {
  local job_key="$1"
  "$PY" - "$FAILURES_FILE" "$job_key" <<'PY'
import json, sys
from pathlib import Path
path, job_key = sys.argv[1:3]
data = json.loads(Path(path).read_text())
print(int(data.get(job_key, 0)))
PY
}

record_launch_failure() {
  local job_key="$1"
  local run_id="$2"
  local reason="$3"
  local count
  count=$("$PY" - "$FAILURES_FILE" "$job_key" <<'PY'
import json, sys
from pathlib import Path
path, job_key = sys.argv[1:3]
data = json.loads(Path(path).read_text())
data[job_key] = int(data.get(job_key, 0)) + 1
Path(path).write_text(json.dumps(data, indent=2) + "\n")
print(data[job_key])
PY
)
  log "launch failure job_key=$job_key run_id=$run_id count=$count/$MAX_LAUNCH_FAILURES reason=$reason"
  if [ "$count" -ge "$MAX_LAUNCH_FAILURES" ]; then
    mark_abandoned "$job_key"
    log "abandoned job_key=$job_key after $count launch failures"
  fi
}

clear_failure_count() {
  local job_key="$1"
  "$PY" - "$FAILURES_FILE" "$job_key" <<'PY'
import json, sys
from pathlib import Path
path, job_key = sys.argv[1:3]
data = json.loads(Path(path).read_text())
data.pop(job_key, None)
Path(path).write_text(json.dumps(data, indent=2) + "\n")
PY
}

write_pending_verify() {
  local job_key="$1"
  local run_id="$2"
  local unit="$3"
  local launched_at="$4"
  local verify_after="$5"
  "$PY" - "$PENDING_VERIFY_FILE" "$job_key" "$run_id" "$unit" "$launched_at" "$verify_after" <<'PY'
import json, sys
path, job_key, run_id, unit, launched_at, verify_after = sys.argv[1:7]
from pathlib import Path
Path(path).write_text(json.dumps({
    "job_key": job_key,
    "run_id": run_id,
    "unit": unit,
    "launched_at": int(launched_at),
    "verify_after": int(verify_after),
}, indent=2) + "\n")
PY
}

clear_pending_verify() {
  rm -f "$PENDING_VERIFY_FILE"
}

launch_unit_active() {
  local unit="$1"
  systemctl --user is-active --quiet "$unit" 2>/dev/null
}

run_id_alive() {
  local run_id="$1"
  if [ -n "${STAND_IN_CMD:-}" ]; then
    pgrep -f "gpu-idle-standin-${run_id}" >/dev/null 2>&1
    return
  fi
  pgrep -f "standalone_trainer.*${run_id}" >/dev/null 2>&1
}

verify_pending_launch() {
  if [ ! -f "$PENDING_VERIFY_FILE" ]; then
    return 0
  fi

  local pending now job_key run_id unit verify_after remaining
  pending=$("$PY" - "$PENDING_VERIFY_FILE" <<'PY'
import json, sys
from pathlib import Path
print(Path(sys.argv[1]).read_text())
PY
)
  job_key=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["job_key"])' "$pending")
  run_id=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["run_id"])' "$pending")
  unit=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["unit"])' "$pending")
  verify_after=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["verify_after"])' "$pending")
  now=$(date +%s)

  if [ "$now" -lt "$verify_after" ]; then
    remaining=$((verify_after - now))
    log "awaiting post-exit verify for run_id=$run_id (${remaining}s remaining)"
    return 2
  fi

  if launch_unit_active "$unit" || run_id_alive "$run_id"; then
    clear_failure_count "$job_key"
    clear_pending_verify
    log "verified post-exit: run_id=$run_id unit=$unit still active"
    return 0
  fi

  record_launch_failure "$job_key" "$run_id" "died before post-exit verify"
  clear_pending_verify
  if [ -f "$TRAINING/outputs/runs/$run_id/launch.out" ]; then
    tail -20 "$TRAINING/outputs/runs/$run_id/launch.out" >>"$LOG" 2>/dev/null || true
  fi
  journalctl --user -u "$unit" -n 20 --no-pager >>"$LOG" 2>/dev/null || true
  return 1
}

run_finished() {
  local log_file="$1"
  [ -f "$log_file" ] && grep -q "Training finished" "$log_file" 2>/dev/null
}

highest_checkpoint() {
  local ckpt_dir="$1"
  ls "$ckpt_dir"/policy_*.pt 2>/dev/null | while read -r f; do
    step=$(basename "$f" .pt | sed 's/policy_//')
    echo "$step $f"
  done | sort -n | tail -1 | cut -d' ' -f2-
}

validate_config() {
  local config_path="$1"
  local requires_selfplay="${2:-false}"
  if [ ! -f "$config_path" ]; then
    log "validate: missing config $config_path"
    return 1
  fi
  if [ ! -f "$CHAMPION_CKPT" ]; then
    log "validate: missing champion checkpoint $CHAMPION_CKPT"
    return 1
  fi
  if [ "$requires_selfplay" = "true" ]; then
    if ! "$PY" -c 'import json,sys; print("selfplay" in json.load(open(sys.argv[1])))' "$config_path"; then
      log "validate: requires_selfplay but config lacks selfplay block"
      return 1
    fi
  fi
  return 0
}

queue_next_job() {
  "$PY" - "$QUEUE" "$COMPLETED_FILE" "$ABANDONED_FILE" "$REPO" <<'PY'
import json
import sys
from pathlib import Path

queue_path, completed_path, abandoned_path, repo = sys.argv[1:5]
completed = {
    line.strip()
    for line in Path(completed_path).read_text().splitlines()
    if line.strip()
}
abandoned = {
    line.strip()
    for line in Path(abandoned_path).read_text().splitlines()
    if line.strip()
}
data = json.loads(Path(queue_path).read_text())
abandoned.update(data.get("abandoned_run_ids", []))
runs_root = Path(repo) / "training/outputs/runs"
jobs = sorted(data.get("jobs", []), key=lambda j: j.get("priority", 999))
for job in jobs:
    run_id = job["run_id"]
    if run_id in completed or run_id in abandoned:
        continue
    log_file = runs_root / run_id / "run.log"
    if log_file.exists():
        text = log_file.read_text(errors="replace")
        if "Training finished" in text:
            continue
    job = dict(job)
    job["config"] = str(Path(repo) / job["config"])
    print(json.dumps(job))
    break
PY
}

find_incomplete_2b() {
  "$PY" - "$QUEUE" "$ABANDONED_FILE" "$REPO" <<'PY'
import json
import re
import sys
from pathlib import Path

queue_path, abandoned_path, repo = sys.argv[1:4]
abandoned = {
    line.strip()
    for line in Path(abandoned_path).read_text().splitlines()
    if line.strip()
}
data = json.loads(Path(queue_path).read_text())
recovery = data.get("recovery_2b", {})
abandoned.update(data.get("abandoned_run_ids", []))
horizon = int(recovery.get("horizon", 2_000_000_000))
runs_root = Path(repo) / "training/outputs/runs"


def highest(ckpt_dir: Path) -> str | None:
    best = None
    best_step = -1
    if not ckpt_dir.is_dir():
        return None
    for path in ckpt_dir.glob("policy_*.pt"):
        step = int(path.stem.split("_", 1)[1])
        if step > best_step:
            best_step = step
            best = str(path)
    return best


def latest_transitions(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    vals = re.findall(
        r"standalone_trainer INFO: ticks=\d+ transitions=(\d+)",
        log_path.read_text(errors="replace"),
    )
    return int(vals[-1]) if vals else None


for run_id in recovery.get("run_ids", []):
    if run_id in abandoned:
        continue
    run_dir = runs_root / run_id
    log_file = run_dir / "run.log"
    if not log_file.exists():
        continue
    if "Training finished" in log_file.read_text(errors="replace"):
        continue
    trans = latest_transitions(log_file)
    if trans is None:
        continue
    if trans >= horizon:
        continue
    meta = recovery.get("config_by_run_id", {}).get(run_id)
    if not meta:
        continue
    init_ckpt = highest(run_dir / "checkpoints")
    print(json.dumps({
        "kind": "recovery_2b",
        "run_id": run_id,
        "transitions": trans,
        "horizon": horizon,
        "config": str(Path(repo) / meta["config"]),
        "num_envs": int(meta["num_envs"]),
        "init_ckpt": init_ckpt,
    }))
    break
PY
}

launch_trainer() {
  local job_key="$1"
  local run_id="$2"
  local config="$3"
  local num_envs="$4"
  local total_transitions="$5"
  local init_ckpt="${6:-}"

  local run_dir="$TRAINING/outputs/runs/$run_id"
  mkdir -p "$run_dir"

  local unit="f1tenth-trainer-${run_id}-$(date +%s).service"
  local -a run_cmd=()

  if [ -n "${STAND_IN_CMD:-}" ]; then
    run_cmd=(bash -lc "exec -a gpu-idle-standin-${run_id} ${STAND_IN_CMD}")
  else
    local -a cmd=(
      "$PY" standalone_trainer.py
      --config "$config"
      --num-envs "$num_envs"
      --total-transitions "$total_transitions"
      --opponent policy
      --fixed-opponents
      --device cuda
      --seed 42
      --compile
      --compile-mode reduce-overhead
      --run-id "$run_id"
      --wandb
      --wandb-mode online
    )
    if [ -n "$init_ckpt" ] && [ -f "$init_ckpt" ]; then
      cmd+=(--init-ckpt "$init_ckpt")
    fi
    run_cmd=("${cmd[@]}")
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    log "DRY-RUN would launch run_id=$run_id unit=$unit num_envs=$num_envs transitions=$total_transitions config=$config init_ckpt=${init_ckpt:-none}"
    return 0
  fi

  if ! systemd-run --user \
    --unit="$unit" \
    --description="F1TENTH idle GPU trainer ${run_id}" \
    --working-directory="$TRAINING" \
    --collect \
    --setenv=WANDB_MODE=online \
    -- "${run_cmd[@]}" >>"${run_dir}/launch.out" 2>&1; then
    log "ERROR: systemd-run failed for run_id=$run_id"
    record_launch_failure "$job_key" "$run_id" "systemd-run failed"
    return 1
  fi

  local now verify_after
  now=$(date +%s)
  verify_after=$((now + LAUNCH_VERIFY_SEC))
  write_pending_verify "$job_key" "$run_id" "$unit" "$now" "$verify_after"
  log "launched run_id=$run_id unit=$unit; post-exit verify scheduled in ${LAUNCH_VERIFY_SEC}s"
  return 0
}

pick_job() {
  local job=""
  job=$(find_incomplete_2b 2>/dev/null || true)
  if [ -n "$job" ]; then
    echo "$job"
    return
  fi
  queue_next_job 2>/dev/null || true
}

main() {
  local verify_rc=0
  verify_pending_launch || verify_rc=$?
  if [ "$verify_rc" -eq 2 ]; then
    exit 0
  fi

  local procs util idle idle_target job gpu_ok=1
  procs=$(trainer_count)
  util=$(gpu_util_pct) || gpu_ok=0

  if [ "$DRY_RUN" -eq 0 ] && [ "$SIMULATE_IDLE" -eq 0 ]; then
    if [ "$gpu_ok" -eq 0 ]; then
      reset_idle_count
      log "abort: GPU utilization unreadable — not launching (see ERROR above)"
      exit 0
    fi
    if [ "${procs:-0}" -gt 0 ]; then
      reset_idle_count
      log "busy: $procs trainer proc(s), gpu=${util}%"
      exit 0
    fi
    if [ "${util:-100}" -gt "$IDLE_GPU_UTIL_MAX" ]; then
      reset_idle_count
      log "busy: gpu=${util}% (> ${IDLE_GPU_UTIL_MAX}%), no trainer match"
      exit 0
    fi
  fi

  idle=$(read_idle_count)
  idle=$((idle + 1))
  echo "$idle" >"$IDLE_COUNT_FILE"
  if [ "$gpu_ok" -eq 0 ]; then
    log "idle check ${idle}/${IDLE_CHECKS_TO_LAUNCH}: trainers=${procs:-0} gpu=unreadable"
  else
    log "idle check ${idle}/${IDLE_CHECKS_TO_LAUNCH}: trainers=${procs:-0} gpu=${util}%"
  fi

  if [ "$idle" -lt "$IDLE_CHECKS_TO_LAUNCH" ]; then
    exit 0
  fi

  reset_idle_count

  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    log "lock held, skipping launch"
    exit 0
  fi

  procs=$(trainer_count)
  util=$(gpu_util_pct) || gpu_ok=0
  if [ "$SIMULATE_IDLE" -eq 0 ] && { [ "$gpu_ok" -eq 0 ] || [ "${procs:-0}" -gt 0 ] || [ "${util:-100}" -gt "$IDLE_GPU_UTIL_MAX" ]; }; then
    if [ "$gpu_ok" -eq 0 ]; then
      log "idle threshold met but GPU utilization unreadable, aborting launch"
    else
      log "idle threshold met but GPU no longer idle (trainers=${procs:-0} gpu=${util}%), aborting launch"
    fi
    exit 0
  fi

  job=$(pick_job)
  if [ -z "$job" ]; then
    log "idle GPU but queue empty — nothing to launch"
    exit 0
  fi

  while [ -n "$job" ]; do
    kind=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1]).get("kind","queue"))' "$job")
    run_id=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["run_id"])' "$job")
    config=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["config"])' "$job")
    num_envs=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["num_envs"])' "$job")

    if is_abandoned "$run_id"; then
      log "skipping abandoned job $run_id"
      job=$(queue_next_job 2>/dev/null || true)
      continue
    fi

    if [ "$kind" = "recovery_2b" ]; then
      total_transitions=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["horizon"])' "$job")
      init_ckpt=$("$PY" -c 'import json,sys; v=json.loads(sys.argv[1]).get("init_ckpt"); print(v or "")' "$job")
      trans=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1]).get("transitions",0))' "$job")
      recovery_run_id="${run_id}-r$(date +%s)"
      log "2B recovery: ${run_id} stopped @${trans}; relaunching as ${recovery_run_id} init_ckpt=${init_ckpt:-none} (warm-start only, not full resume)"
      if ! validate_config "$config" false; then
        log "2B recovery config invalid, skipping"
        exit 1
      fi
      launch_trainer "$run_id" "$recovery_run_id" "$config" "$num_envs" "$total_transitions" "$init_ckpt"
      exit 0
    fi

    total_transitions=$("$PY" -c 'import json,sys; print(json.loads(sys.argv[1])["total_transitions"])' "$job")
    requires_selfplay=$("$PY" -c 'import json,sys; print("true" if json.loads(sys.argv[1]).get("requires_selfplay") else "false")' "$job")
    init_ckpt=""

    if ! validate_config "$config" "$requires_selfplay"; then
      mark_completed "$run_id"
      log "skipped invalid/disabled job $run_id, trying next queue item"
      job=$(queue_next_job 2>/dev/null || true)
      continue
    fi

    if pgrep -f "standalone_trainer.*${run_id}" >/dev/null 2>&1; then
      log "run_id=$run_id already running"
      exit 0
    fi

    local run_dir="$TRAINING/outputs/runs/$run_id"
    local run_log="$run_dir/run.log"
    local launch_id="$run_id"
    if [ -f "$run_log" ] && ! run_finished "$run_log"; then
      init_ckpt=$(highest_checkpoint "$run_dir/checkpoints")
      launch_id="${run_id}-r$(date +%s)"
      log "recovering partial queue run ${run_id} as ${launch_id} init_ckpt=${init_ckpt:-none}"
    fi

    log "launching queue job run_id=$launch_id"
    launch_trainer "$run_id" "$launch_id" "$config" "$num_envs" "$total_transitions" "$init_ckpt"
    exit 0
  done

  log "no launchable queue jobs remain"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
