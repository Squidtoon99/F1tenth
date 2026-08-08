#!/usr/bin/env bash
# Persistent GPU work queue for gigaflow on a Brev/CUDA host.
# Priority: Warp residency + scaling benches → PPO smoke → reduced train →
# production train → Warp soak fallback. Never idles while rent remains.
set -u
set -o pipefail

ROOT="${GIGAFLOW_ROOT:-/home/shadeform/work/gigaflow}"
VENV="${GIGAFLOW_VENV:-/home/shadeform/work/venv}"
CACHE="${GIGAFLOW_TRACK_CACHE:-/home/shadeform/.cache/gigaflow/tracks}"
RUNS="${GIGAFLOW_RUNS:-/home/shadeform/work/runs}"
LOGS="${GIGAFLOW_LOGS:-/home/shadeform/work/logs}"
STATE_DIR="${GIGAFLOW_STATE:-/home/shadeform/work/supervisor}"
mkdir -p "$RUNS" "$LOGS" "$STATE_DIR"

# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$ROOT"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOGS/supervisor.log"; }
mark() { echo "$1" > "$STATE_DIR/current_job"; echo "$(ts)" > "$STATE_DIR/current_job_started"; }
set_next() { printf "%s\n" "$@" > "$STATE_DIR/queue_next.txt"; }

run_job() {
  local name="$1"; shift
  local logfile="$LOGS/${name}.log"
  mark "$name"
  log "START job=$name cmd=$*"
  set +e
  "$@" >>"$logfile" 2>&1
  local rc=$?
  set -e
  echo "$rc" > "$STATE_DIR/last_rc_$name"
  if [[ $rc -eq 0 ]]; then
    log "OK job=$name rc=0"
  else
    log "FAIL job=$name rc=$rc logfile=$logfile"
  fi
  return $rc
}

warp_soak() {
  local hours="${1:-1}"
  local iters=$((hours * 200000))
  mark "warp_soak"
  log "FALLBACK warp soak iters=$iters"
  PROBE_ITERS="$iters" python -u /home/shadeform/work/warp_gpu_probe.py \
    >>"$LOGS/warp_soak.log" 2>&1 || true
}

# Wait for track atlas if another process is still packing it.
wait_tracks() {
  local manifest="$CACHE/manifest.json"
  local i=0
  while [[ ! -f "$manifest" ]]; do
    if ! pgrep -f "gigaflow prepare-tracks" >/dev/null; then
      log "tracks missing; launching prepare-tracks"
      run_job prepare_tracks gigaflow prepare-tracks \
        --config configs/default.yaml --cache-dir "$CACHE" || return 1
      break
    fi
    log "waiting for prepare-tracks... ($i)"
    sleep 30
    i=$((i + 1))
    if [[ $i -gt 240 ]]; then
      log "prepare-tracks wait timed out"
      return 1
    fi
  done
  return 0
}

export GIGAFLOW_WARP_GATE_OK=0

set_next \
  "validate_config" \
  "warp_residency" \
  "bench_sim_scale" \
  "bench_learner_smoke" \
  "pytest_gpu" \
  "train_gpu_smoke" \
  "train_gpu_reduced" \
  "train_production" \
  "warp_soak_loop"

log "supervisor starting root=$ROOT"

run_job validate_config gigaflow validate-config --config configs/default.yaml || true
wait_tracks || log "WARN track wait failed; continuing with synthetic atlas jobs"

# 1) Required Warp CUDA residency / validation
if run_job warp_residency python tools/check_warp_residency.py \
  --config configs/gpu_smoke.yaml --steps 32; then
  GIGAFLOW_WARP_GATE_OK=1
  echo 1 > "$STATE_DIR/warp_gate_ok"
else
  echo 0 > "$STATE_DIR/warp_gate_ok"
  log "BLOCKER: Warp CUDA residency gate failed; not graduating to production train"
fi

# 2) Scaling benchmarks (always useful GPU work)
run_job bench_sim_scale gigaflow benchmark \
  --config configs/gpu_smoke.yaml --mode sim --steps 200 --device cuda || true
run_job bench_sim_default gigaflow benchmark \
  --config configs/default.yaml --mode sim --steps 50 --device cuda || true
run_job bench_learner_smoke gigaflow benchmark \
  --config configs/gpu_smoke.yaml --mode learner --steps 2 --device cuda || true

# 3) GPU-focused pytest subset (real modules)
run_job pytest_gpu pytest -q tests/test_n_agent_sim.py tests/test_rewards.py \
  tests/test_integration.py -k "not cpu_contact and not cpu_ray" || true

# 4) One-track / two-car PPO smoke (only if residency gate passed)
if [[ "$GIGAFLOW_WARP_GATE_OK" == "1" ]]; then
  run_job train_gpu_smoke gigaflow train \
    --config configs/gpu_smoke.yaml --smoke \
    --device cuda \
    --run-dir "$RUNS/gpu_smoke" \
    --checkpoint-interval 1 || true
else
  log "skip train_gpu_smoke (warp gate closed)"
  warp_soak 1
fi

# 5) Reduced all-track multi-car training
if [[ "$GIGAFLOW_WARP_GATE_OK" == "1" && -f "$CACHE/manifest.json" ]]; then
  # Point config at prepared atlas via env override file.
  python - <<'PY'
from pathlib import Path
import yaml
root = Path("/home/shadeform/work/gigaflow")
cache = Path("/home/shadeform/.cache/gigaflow/tracks/manifest.json")
cfg = yaml.safe_load((root / "configs/gpu_reduced.yaml").read_text())
cfg["tracks"]["manifest_path"] = str(cache)
out = root / "configs" / "_runtime_gpu_reduced.yaml"
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(out)
PY
  run_job train_gpu_reduced gigaflow train \
    --config configs/_runtime_gpu_reduced.yaml \
    --num-updates 200 \
    --device cuda \
    --run-dir "$RUNS/gpu_reduced" \
    --checkpoint-interval 20 || true
else
  log "skip train_gpu_reduced (gate/tracks)"
fi

# 6) Largest memory-safe production training
if [[ "$GIGAFLOW_WARP_GATE_OK" == "1" && -f "$CACHE/manifest.json" ]]; then
  python - <<'PY'
from pathlib import Path
import yaml
root = Path("/home/shadeform/work/gigaflow")
cache = Path("/home/shadeform/.cache/gigaflow/tracks/manifest.json")
cfg = yaml.safe_load((root / "configs/default.yaml").read_text())
cfg["tracks"]["manifest_path"] = str(cache)
# Memory-safe production on 80GB: keep default 64x8 but allow OOM fallback later.
out = root / "configs" / "_runtime_production.yaml"
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(out)
PY
  run_job train_production gigaflow train \
    --config configs/_runtime_production.yaml \
    --device cuda \
    --run-dir "$RUNS/production" \
    --checkpoint-interval 50 || {
      log "production train failed; falling back to reduced long-run + soak"
      run_job train_gpu_reduced_long gigaflow train \
        --config configs/_runtime_gpu_reduced.yaml \
        --num-updates 5000 \
        --device cuda \
        --run-dir "$RUNS/gpu_reduced_long" \
        --checkpoint-interval 50 || true
    }
else
  log "skip train_production (gate/tracks)"
fi

# 7) Keep GPU busy for the rental window with Warp soak / re-benchmarks
while true; do
  set_next "warp_soak_loop" "bench_sim_scale" "bench_learner_smoke"
  run_job bench_sim_recheck gigaflow benchmark \
    --config configs/gpu_smoke.yaml --mode sim --steps 400 --device cuda || true
  if [[ "$GIGAFLOW_WARP_GATE_OK" == "1" && -f "$CACHE/manifest.json" ]]; then
    run_job train_reduced_continue gigaflow train \
      --config configs/_runtime_gpu_reduced.yaml \
      --num-updates 500 \
      --device cuda \
      --run-dir "$RUNS/gpu_reduced_continue" \
      --checkpoint-interval 25 || warp_soak 2
  else
    warp_soak 2
  fi
done
