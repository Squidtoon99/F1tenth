#!/usr/bin/env bash
# Bounded filter comparison trials (no quality stops).
set -euo pipefail
GFROOT="/home/ubuntu/.cursor/worktrees/F1tenth__SSH__darktoaster_/5e81/gigaflow"
export PYTHONPATH="$GFROOT/src"
PY="/home/ubuntu/miniconda3/envs/g2/bin/python"
OUT="$GFROOT/outputs/clean_grace_period/trials"
UPDATES=45
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$GFROOT/outputs/clean_grace_period/trials/filter_matrix_${TS}.log"

run_trial() {
  local id="$1" config="$2" seed="$3"
  local dir="$OUT/${TS}_${id}"
  mkdir -p "$dir"
  {
    echo "=== FILTER TRIAL $id seed=$seed dir=$dir ==="
    date -u +%Y-%m-%dT%H:%M:%SZ
  } | tee -a "$LOG" "$dir/start.txt"
  cp "$config" "$dir/trial_config.yaml"
  "$PY" - <<PY
import yaml
from pathlib import Path
p = Path("$dir/trial_config.yaml")
raw = yaml.safe_load(p.read_text())
raw["seed"] = int("$seed")
p.write_text(yaml.dump(raw, sort_keys=False))
PY
  "$PY" -m gigaflow_f1tenth.cli train \
    --config "$dir/trial_config.yaml" \
    --device cuda \
    --num-updates "$UPDATES" \
    --run-dir "$dir" \
    --checkpoint-interval 0 \
    2>&1 | tee "$dir/train.log"
  date -u +%Y-%m-%dT%H:%M:%SZ | tee "$dir/end.txt"
  "$PY" - <<PY
import json
import math
from pathlib import Path

root = Path("$dir")
manifest = json.loads((root / "run_manifest.json").read_text())
rows = []
for p in sorted(root.glob("metrics_*.json")):
    m = json.loads(p.read_text())
    prog = m.get("progress", {})
    prof = m.get("profile", {})
    valid = float(prog.get("valid_transitions") or 0.0)
    retention = float(prog.get("retention") or 0.0)
    retained = retention * valid
    mb = 2048.0
    rows.append({
        "file": p.name,
        "update_index": int(p.stem.split("_")[-1]),
        "rollout_transitions": prof.get("transitions"),
        "valid_transitions": valid,
        "retention": retention,
        "retained_transitions": retained,
        "retained_per_minibatch": retained / mb if mb > 0 else None,
        "filter_eta": prog.get("filter_eta"),
        "advantage_raw_p99": prog.get("advantage_raw_p99"),
        "advantage_raw_p99_9": prog.get("advantage_raw_p99_9"),
        "actor_grad_norm": prog.get("actor_grad_norm"),
        "approx_kl": prog.get("approx_kl"),
        "candidate_kl": prog.get("candidate_kl"),
        "clip_fraction": prog.get("clip_fraction"),
        "progress_mean": prog.get("progress_mean"),
        "update_s": prof.get("update_s"),
    })

def mean_key(key, tail=10):
    vals = [r[key] for r in rows[-tail:] if r.get(key) is not None and math.isfinite(float(r[key]))]
    return sum(vals) / len(vals) if vals else None

summary = {
    "trial_id": "$id",
    "seed": int("$seed"),
    "updates": manifest.get("final_update_index"),
    "shutdown_reason": manifest.get("shutdown_reason"),
    "started_at": manifest.get("started_at"),
    "finished_at": manifest.get("finished_at"),
    "mean_retention_tail10": mean_key("retention"),
    "mean_retained_tail10": mean_key("retained_transitions"),
    "mean_retained_per_mb_tail10": mean_key("retained_per_minibatch"),
    "mean_actor_grad_norm_tail10": mean_key("actor_grad_norm"),
    "mean_candidate_kl_tail10": mean_key("candidate_kl"),
    "mean_approx_kl_tail10": mean_key("approx_kl"),
    "mean_clip_fraction_tail10": mean_key("clip_fraction"),
    "progress_mean_last": rows[-1]["progress_mean"] if rows else None,
    "metrics_tail": rows[-3:],
    "metrics_all": rows,
}
(root / "filter_trial_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
}

run_trial "d_filter_eta005" "$GFROOT/configs/clean_grace_filter_eta005.yaml" 104
run_trial "e_filter_off" "$GFROOT/configs/clean_grace_filter_off.yaml" 105

echo "FILTER TRIALS DONE" | tee -a "$LOG"
