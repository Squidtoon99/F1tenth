#!/usr/bin/env bash
# Launch online courtyard_2 PPO fine-tuning from a mounted actor checkpoint.
# Does not start until the container is run with the checkpoint mounted.
# Loads actor-only weights (baked_condition in actor_state_dict is ignored).
# Current limits come from courtyard_ppo.yaml (60 A drive / 20 A brake).
set -euo pipefail

CKPT="${CHECKPOINT:-/opt/f1tenth/policies/actor_005700.pt}"
OUT="${RUN_DIR:-/opt/f1tenth/outputs/courtyard_ppo}"
DEVICE="${DEVICE:-cuda}"
CONFIG="${GIGAFLOW_CONFIG:-/opt/gigaflow/configs/courtyard_ppo.yaml}"

mkdir -p "${OUT}"

if [ ! -f "${CKPT}" ]; then
  echo "checkpoint not found: ${CKPT}" >&2
  echo "mount actor_005700.pt at /opt/f1tenth/policies/actor_005700.pt" >&2
  exit 1
fi

if [ "${DEVICE}" = "cuda" ]; then
  python3 -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' \
    || { echo "DEVICE=cuda but torch.cuda.is_available() is False" >&2; exit 1; }
fi

exec python3 -m gigaflow_f1tenth.cli train-courtyard \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --run-dir "${OUT}" \
  --device "${DEVICE}" \
  "$@"
