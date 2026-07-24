#!/usr/bin/env bash
# Freeze a built image into a portable, SHA-named artifact for manual sharing/deploy
# (no registry yet). Optionally also produce an Apptainer .sif.
#
# Usage:   deploy/scripts/snapshot.sh
# Env:     TARGET (racing|sensor_policy), IMAGE, GITSHA, OUT_DIR
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

TARGET="${TARGET:-racing}"
case "${TARGET}" in
  racing) IMAGE="${IMAGE:-f1tenth-racing}" ;;
  sensor_policy) IMAGE="${IMAGE:-f1tenth-sensor-policy}" ;;
  *) echo "TARGET must be racing or sensor_policy (got: ${TARGET})" >&2; exit 1 ;;
esac

GITSHA="${GITSHA:-$(git rev-parse --short HEAD)}"
OUT_DIR="${OUT_DIR:-deploy/snapshots}"   # gitignored
mkdir -p "${OUT_DIR}"

if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  echo "docker not available (need a working daemon); cannot snapshot ${IMAGE}:${GITSHA}" >&2
  exit 1
fi

TAR="${OUT_DIR}/${IMAGE}-${GITSHA}.tar.gz"
echo "==> Saving ${IMAGE}:${GITSHA} -> ${TAR}"
docker save "${IMAGE}:${GITSHA}" | gzip > "${TAR}"

if command -v shasum >/dev/null 2>&1; then
  SHA256="$(shasum -a 256 "${TAR}" | awk '{print $1}')"
else
  SHA256="$(sha256sum "${TAR}" | awk '{print $1}')"
fi

echo "==> Snapshot: ${TAR}"
echo "    sha256:  ${SHA256}"
if [ "${TARGET}" = "sensor_policy" ]; then
  echo "    pins:    $(docker run --rm --entrypoint cat "${IMAGE}:${GITSHA}" \
    /etc/f1tenth/sensor_policy_pins.json 2>/dev/null | tr -d '\n' || echo '(unavailable)')"
fi
echo "    Record this in deploy/releases/manifest.csv (git_sha,image_sha256,car_id,...)."
echo "    Load: TARGET=${TARGET} deploy/scripts/load_to_jetson.sh user@host ${TAR} car01"

# Optional Apptainer image (uncomment if apptainer is installed):
# apptainer build "${OUT_DIR}/${IMAGE}-${GITSHA}.sif" "docker-daemon://${IMAGE}:${GITSHA}"
