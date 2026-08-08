#!/usr/bin/env bash
# Preserve stale run → pytest → reduced long-horizon pre-KL gate → W&B gate.
# Does NOT launch H100 production until online W&B auth + short W&B resume gate pass.
set -euo pipefail

ROOT="${GIGAFLOW_ROOT:-/home/shadeform/work/gigaflow}"
VENV="${GIGAFLOW_VENV:-/home/shadeform/work/venv}"
CACHE="${GIGAFLOW_TRACK_CACHE:-/home/shadeform/.cache/gigaflow/tracks}"
RUNS="${GIGAFLOW_RUNS:-/home/shadeform/work/runs}"
LOGS="${GIGAFLOW_LOGS:-/home/shadeform/work/logs}"
STATE="${GIGAFLOW_STATE:-/home/shadeform/work/supervisor}"
mkdir -p "$RUNS" "$LOGS" "$STATE"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export GIGAFLOW_TRACK_CACHE="$CACHE"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOGS/learning_gate.log"; }
mark() {
  echo "$1" > "$STATE/current_job"
  echo "$(ts)" > "$STATE/current_job_started"
  echo "$$" > "$STATE/current_job_pid"
}

preserve_run() {
  local src="$1" dest="$2"
  if [[ -d "$src" ]]; then
    mkdir -p "$(dirname "$dest")"
    if [[ -e "$dest" ]]; then
      dest="${dest}_$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    mv "$src" "$dest"
    log "preserved $src -> $dest"
  fi
}

start_soak() {
  mark "warp_probe_fallback"
  pkill -f '/tmp/warp_soak.py' 2>/dev/null || true
  pkill -f 'warp_gpu_probe' 2>/dev/null || true
  sleep 1
  nohup python -u /tmp/warp_soak.py >>"$LOGS/warp_soak.log" 2>&1 &
  echo $! > "$STATE/current_job_pid"
  echo warp_probe_fallback > "$STATE/current_job"
  log "started warp soak pid=$(cat "$STATE/current_job_pid")"
}

# --- stop any stale train; keep GPU busy ---
mark "stop_stale_train"
pkill -f "gigaflow train" 2>/dev/null || true
sleep 1
if [[ ! -f /tmp/warp_soak.py ]]; then
  log "ERROR: /tmp/warp_soak.py missing; copy from workstation first"
  exit 2
fi
start_soak

preserve_run "$RUNS/production_h100" "$RUNS/diagnostic_stale_prekl/production_h100"
preserve_run "$RUNS/reduced_gate_learning" "$RUNS/diagnostic_stale_prekl/reduced_gate_learning"

# Immutable runtime configs (manifest pinned to packed atlas)
python - <<'PY'
from pathlib import Path
import yaml

root = Path("/home/shadeform/work/gigaflow")
cache = Path("/home/shadeform/.cache/gigaflow/tracks/manifest.json")

def write_runtime(src_name: str, dst_name: str, overrides: dict):
    src = yaml.safe_load((root / "configs" / src_name).read_text())
    src["tracks"]["manifest_path"] = str(cache)
    for section, vals in overrides.items():
        src.setdefault(section, {}).update(vals)
    dst = root / "configs" / dst_name
    dst.write_text(yaml.safe_dump(src, sort_keys=False))
    print(f"wrote {dst}")

write_runtime(
    "gpu_reduced_gate.yaml",
    "_runtime_reduced_gate.yaml",
    {
        "ppo": {
            "amp": False,
            "total_updates": 80,
            "max_pre_update_kl": 1.0e-3,
            "max_logp_delta": 5.0e-3,
        },
        "wandb": {"enabled": False, "mode": "online", "project": "f1tenth-gigaflow"},
    },
)
write_runtime(
    "production_h100.yaml",
    "_runtime_h100_production.yaml",
    {
        "worlds": {"num_worlds": 256, "max_agents_per_world": 8},
        "ppo": {
            "amp": False,
            "total_updates": 10000,
            "rollout_length": 128,
            "minibatch_size": 24576,
            "max_pre_update_kl": 1.0e-3,
            "max_logp_delta": 5.0e-3,
        },
        "wandb": {"enabled": False, "mode": "online", "project": "f1tenth-gigaflow"},
    },
)
PY

# Stop soak for GPU tests / gate
SOAK_PID="$(cat "$STATE/current_job_pid" 2>/dev/null || true)"
if [[ -n "${SOAK_PID}" ]] && kill -0 "$SOAK_PID" 2>/dev/null; then
  kill "$SOAK_PID" 2>/dev/null || true
  wait "$SOAK_PID" 2>/dev/null || true
fi

mark "pytest_learning_fixes"
pytest -q \
  tests/test_ppo_logp_parity.py \
  tests/test_async_train_gate.py::test_training_sim_defaults_to_async_respawn \
  tests/test_async_train_gate.py::test_collect_keeps_population_across_updates \
  tests/test_ppo.py::test_adaptive_filter_ignores_nonfinite_advantages \
  tests/test_ppo.py::test_gae_sanitizes_nonfinite_rewards_values \
  | tee "$LOGS/pytest_learning_fixes.log"

mark "train_reduced_gate"
rm -rf "$RUNS/reduced_gate_learning"
gigaflow train \
  --config configs/_runtime_reduced_gate.yaml \
  --device cuda \
  --run-dir "$RUNS/reduced_gate_learning" \
  --checkpoint-interval 10 \
  | tee "$LOGS/train_reduced_gate_learning.log"

mark "validate_reduced_gate"
python - <<'PY'
import json
import math
from pathlib import Path

run = Path("/home/shadeform/work/runs/reduced_gate_learning")
files = sorted(run.glob("metrics_*.json"))
assert files, "no metrics written"
assert len(files) >= 50, f"need long-horizon metrics, got {len(files)}"
ok = 0
pre_kls = []
ents = []
epochs = []
for p in files:
    d = json.loads(p.read_text())["progress"]
    pre_kls.append(abs(float(d["pre_update_approx_kl"])))
    ents.append(float(d["entropy"]))
    epochs.append(float(d["epochs_completed"]))
    checks = {
        "valid_transitions": d["valid_transitions"] > 0,
        "epochs_completed": d["epochs_completed"] > 0,
        "entropy": 0.0 < d["entropy"] < 5.0 and math.isfinite(d["entropy"]),
        "pre_kl": abs(d["pre_update_approx_kl"]) < 1e-3,
        "logp_delta_max": float(d.get("logp_delta_max", 0.0)) < 5e-3,
        "grad_norm": d["grad_norm"] > 0 and math.isfinite(d["grad_norm"]),
        "filter_eta": math.isfinite(d["filter_eta"]),
        "value_finite": math.isfinite(d["value_loss"]),
        "transitions_per_s": d["transitions_per_s"] > 0,
    }
    failed = [k for k, v in checks.items() if not v]
    status = "OK" if not failed else f"FAIL:{failed}"
    print(
        f"{p.name}: {status} pre_kl={d['pre_update_approx_kl']:.3e} "
        f"ent={d['entropy']:.3g} epochs={d['epochs_completed']} "
        f"val={d['value_loss']:.4g} grad={d['grad_norm']:.4g}"
    )
    if not failed:
        ok += 1
frac = ok / len(files)
print(f"PASS_FRAC={frac:.3f} ({ok}/{len(files)})")
print(f"pre_kl_max={max(pre_kls):.3e} entropy_max={max(ents):.3g} epochs_max={max(epochs)}")
if frac < 0.9:
    raise SystemExit(f"reduced gate learning signal too weak: {frac}")
if max(pre_kls) >= 1e-3:
    raise SystemExit(f"pre_update_approx_kl drifted: max={max(pre_kls)}")
if max(ents) >= 5.0:
    raise SystemExit(f"entropy saturating: max={max(ents)}")
ckpts = list(run.glob("ckpt_*.pt"))
assert ckpts, "no checkpoints written"
print("GATE_OK")
PY

mark "wandb_blocker_check"
python - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/shadeform/work/gigaflow/src")
from gigaflow_f1tenth.wandb_log import has_wandb_auth

blockers = []
if not Path("/home/shadeform/work/gigaflow/src/gigaflow_f1tenth/wandb_log.py").is_file():
    blockers.append("wandb_log.py not synced to Brev gigaflow tree")
if not has_wandb_auth():
    blockers.append(
        "no W&B credentials on host (set WANDB_API_KEY or run `wandb login`)"
    )
# entity/project must be explicitly provided for production promotion
entity = os.environ.get("WANDB_ENTITY", "").strip()
project = os.environ.get("WANDB_PROJECT", "").strip()
if not entity:
    blockers.append("WANDB_ENTITY unset (required before production promote)")
if not project:
    blockers.append(
        "WANDB_PROJECT unset (required before production promote; "
        "default config project=f1tenth-gigaflow is not sufficient alone)"
    )
if blockers:
    print("WANDB_BLOCKERS:")
    for b in blockers:
        print(f" - {b}")
    raise SystemExit(2)
print("WANDB_AUTH_OK")
PY

# Only reached when credentials exist — short offline/online W&B resume gate.
mark "wandb_resume_gate"
rm -rf "$RUNS/wandb_resume_gate"
gigaflow train \
  --config configs/_runtime_reduced_gate.yaml \
  --device cuda \
  --run-dir "$RUNS/wandb_resume_gate" \
  --checkpoint-interval 2 \
  --wandb \
  --wandb-mode online \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-tag wandb-gate \
  | tee "$LOGS/train_wandb_resume_gate.log" || {
    log "W&B online gate failed; restarting soak and aborting production promote"
    start_soak
    exit 3
  }

log "pre-KL gate + W&B gate passed; production promote still requires explicit operator launch"
start_soak
mark "awaiting_production_promote"
log "NOT launching production_h100 (policy: wait for explicit promote after W&B proof)"
echo "READY_FOR_PRODUCTION_PROMOTE" | tee "$LOGS/ready_for_production_promote"
