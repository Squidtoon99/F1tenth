#!/usr/bin/env bash
# Bounded clean-grace trial matrix (no quality stops).
set -euo pipefail
GFROOT="/home/ubuntu/.cursor/worktrees/F1tenth__SSH__darktoaster_/5e81/gigaflow"
export PYTHONPATH="$GFROOT/src"
PY="/home/ubuntu/miniconda3/envs/g2/bin/python"
OUT="$GFROOT/outputs/clean_grace_period/trials"
UPDATES=45
TS="$(date +%Y%m%d_%H%M%S)"

run_trial() {
  local id="$1" config="$2" seed="$3"
  local dir="$OUT/${TS}_${id}"
  mkdir -p "$dir"
  echo "=== TRIAL $id seed=$seed dir=$dir ==="
  date -u +%Y-%m-%dT%H:%M:%SZ | tee "$dir/start.txt"
  # Fresh init: resume never; unique seed via config copy
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
from pathlib import Path
root = Path("$dir")
manifest = json.loads((root / "run_manifest.json").read_text())
metrics = []
for p in sorted(root.glob("metrics_*.json")):
    m = json.loads(p.read_text())
    prog = m.get("progress", {})
    prof = m.get("profile", {})
    metrics.append({
        "file": p.name,
        "update_s": prof.get("update_s"),
        "approx_kl": prog.get("approx_kl"),
        "candidate_kl": prog.get("candidate_kl"),
        "entropy": prog.get("entropy"),
        "entropy_coef": prog.get("entropy_coef"),
        "progress_mean": prog.get("progress_mean"),
        "rollback_stop_requested": prog.get("rollback_stop_requested"),
        "early_stopped": prog.get("early_stopped"),
        "actor_rollback_count": prog.get("actor_rollback_count"),
    })
summary = {
    "trial_id": "$id",
    "seed": int("$seed"),
    "updates": manifest.get("final_update_index"),
    "shutdown_reason": manifest.get("shutdown_reason"),
    "started_at": manifest.get("started_at"),
    "finished_at": manifest.get("finished_at"),
    "metrics_tail": metrics[-3:],
}
(root / "trial_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
}

run_trial "a_fixed_ent" "$GFROOT/configs/clean_grace_trial_a_fixed_ent.yaml" 101
run_trial "b_anneal" "$GFROOT/configs/clean_grace_rtx4080.yaml" 102
run_trial "c_anneal_sparse_eval" "$GFROOT/configs/clean_grace_rtx4080.yaml" 103

echo "ALL TRIALS DONE"
