#!/usr/bin/env bash
# Restore the previous sensor-policy (or racing) image tag recorded by
# load_to_jetson.sh under /opt/f1tenth/rollback on the car.
#
# Usage:
#   deploy/scripts/rollback_jetson.sh <user@jetson-host> [sensor_policy|racing]
#
# Does not delete the newer image; it only re-points the moving tag and prints
# the docker run command for the restored artifact.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

JETSON="${1:?usage: rollback_jetson.sh <user@host> [sensor_policy|racing]}"
TARGET="${2:-sensor_policy}"

case "${TARGET}" in
  sensor_policy)
    IMAGE="${IMAGE:-f1tenth-sensor-policy}"
    MOVING_TAG="sensor-policy"
    ;;
  racing)
    IMAGE="${IMAGE:-f1tenth-racing}"
    MOVING_TAG="develop"
    ;;
  *)
    echo "TARGET must be sensor_policy or racing (got: ${TARGET})" >&2
    exit 1
    ;;
esac

ROLLBACK_DIR="/opt/f1tenth/rollback/${TARGET}"

echo "==> Rolling back ${IMAGE}:${MOVING_TAG} on ${JETSON} from ${ROLLBACK_DIR}"
ssh "${JETSON}" bash -s -- "${IMAGE}" "${MOVING_TAG}" "${ROLLBACK_DIR}" <<'EOF'
set -euo pipefail
IMAGE="$1"
MOVING_TAG="$2"
ROLLBACK_DIR="$3"

if [ ! -f "${ROLLBACK_DIR}/image.tag" ]; then
  echo "No rollback record at ${ROLLBACK_DIR}/image.tag" >&2
  exit 1
fi

PREV_TAG="$(cat "${ROLLBACK_DIR}/image.tag")"
if ! docker image inspect "${PREV_TAG}" >/dev/null 2>&1; then
  echo "Rollback image missing locally: ${PREV_TAG}" >&2
  echo "Load the saved tar from ${ROLLBACK_DIR}/ if present, then retry." >&2
  exit 1
fi

docker tag "${PREV_TAG}" "${IMAGE}:${MOVING_TAG}"
echo "Restored ${IMAGE}:${MOVING_TAG} -> ${PREV_TAG}"

if [ -f "${ROLLBACK_DIR}/checkpoint.path" ]; then
  PREV_CKPT="$(cat "${ROLLBACK_DIR}/checkpoint.path")"
  echo "Previous checkpoint path: ${PREV_CKPT}"
fi
if [ -f "${ROLLBACK_DIR}/pins.json" ]; then
  echo "Previous image pins:"
  cat "${ROLLBACK_DIR}/pins.json"
fi
if [ -f "${ROLLBACK_DIR}/run.sh" ]; then
  echo "Previous run helper:"
  cat "${ROLLBACK_DIR}/run.sh"
fi
EOF

echo "==> Rollback complete. Start the restored image with the printed run helper."
