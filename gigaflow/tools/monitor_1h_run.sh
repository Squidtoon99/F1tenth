#!/usr/bin/env bash
# Poll a clean-grace 1h run; emit observation checkpoints until >=OBS_MIN minutes.
set -euo pipefail
GFROOT="/home/ubuntu/.cursor/worktrees/F1tenth__SSH__darktoaster_/5e81/gigaflow"
RUN_DIR="${1:?run dir required}"
OBS_MIN="${2:-60}"
INTERVAL="${3:-300}"
PY="/home/ubuntu/miniconda3/envs/g2/bin/python"
EVIDENCE="$GFROOT/outputs/clean_grace_period/observation_log.jsonl"
PID_FILE="$GFROOT/outputs/clean_grace_period/active_pid.txt"

started_epoch="$("$PY" - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone
p = Path("$RUN_DIR/start.txt")
if p.is_file():
    ts = p.read_text().strip()
    dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    print(int(dt.timestamp()))
else:
    print(int(__import__("time").time()))
PY
)"

while true; do
  now_epoch=$(date +%s)
  elapsed_min=$(( (now_epoch - started_epoch) / 60 ))
  latest="$("$PY" - <<PY
import json
from pathlib import Path
root = Path("$RUN_DIR")
metrics = sorted(root.glob("metrics_*.json"))
if not metrics:
    print("{}")
else:
    m = json.loads(metrics[-1].read_text())
    prog = m.get("progress", {})
    prof = m.get("profile", {})
    print(json.dumps({
        "latest_metrics": metrics[-1].name,
        "update_index": int(metrics[-1].stem.split("_")[-1]),
        "progress_mean": prog.get("progress_mean"),
        "retention": prog.get("retention"),
        "filter_eta": prog.get("filter_eta"),
        "approx_kl": prog.get("approx_kl"),
        "candidate_kl": prog.get("candidate_kl"),
        "actor_grad_norm": prog.get("actor_grad_norm"),
        "clip_fraction": prog.get("clip_fraction"),
        "transitions_per_s": prof.get("transitions_per_s"),
        "rollback_stop_requested": prog.get("rollback_stop_requested"),
        "early_stopped": prog.get("early_stopped"),
    }))
PY
)"
  pid=""
  if [[ -f "$PID_FILE" ]]; then
    pid=$(cat "$PID_FILE")
  fi
  alive=0
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    alive=1
  fi
  checkpoint=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  line=$(printf '{"ts":"%s","elapsed_min":%d,"run_dir":"%s","pid":"%s","alive":%d,"snapshot":%s}' \
    "$checkpoint" "$elapsed_min" "$RUN_DIR" "$pid" "$alive" "$latest")
  echo "$line" >> "$EVIDENCE"
  echo "[$checkpoint] elapsed=${elapsed_min}m alive=$alive $latest"
  if [[ "$elapsed_min" -ge "$OBS_MIN" ]]; then
    echo "OBSERVATION_COMPLETE elapsed=${elapsed_min}m" | tee -a "$GFROOT/outputs/clean_grace_period/observation_complete.txt"
    exit 0
  fi
  if [[ "$alive" -eq 0 ]]; then
    echo "TRAINING_EXITED before ${OBS_MIN}m observation" | tee -a "$GFROOT/outputs/clean_grace_period/observation_complete.txt"
    exit 1
  fi
  sleep "$INTERVAL"
done
