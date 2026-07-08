#!/usr/bin/env bash
# Copy a snapshot to the Jetson, load it, and stage the per-car overlay.
#
# Usage:
#   deploy/scripts/load_to_jetson.sh <user@jetson-host> <snapshot.tar.gz> <carNN>
#
# The per-car identity (car.yaml) on the car is never overwritten; this only stages
# params/map. Adjust to your car's conventions.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

JETSON="${1:?usage: load_to_jetson.sh <user@host> <snapshot.tar.gz> <carNN>}"
SNAPSHOT="${2:?path to snapshot .tar.gz}"
CAR="${3:?car id, e.g. car01}"
IMAGE="${IMAGE:-f1tenth-racing}"

if [ ! -d "deploy/cars/${CAR}" ]; then
  echo "No overlay found at deploy/cars/${CAR}" >&2
  exit 1
fi

echo "==> Copying image snapshot to ${JETSON}"
scp "${SNAPSHOT}" "${JETSON}:/tmp/f1tenth-image.tar.gz"

echo "==> Staging per-car overlay (${CAR}) to ${JETSON}:/opt/f1tenth/config"
ssh "${JETSON}" "mkdir -p /opt/f1tenth/config /opt/f1tenth/policies"
# Do not clobber an existing on-car car.yaml identity.
scp "deploy/cars/${CAR}/params.yaml" "${JETSON}:/opt/f1tenth/config/params.yaml"
scp -r "deploy/cars/${CAR}/maps" "${JETSON}:/opt/f1tenth/config/maps"
ssh "${JETSON}" "test -f /opt/f1tenth/config/car.yaml || cp /dev/stdin /opt/f1tenth/config/car.yaml" \
  < "deploy/cars/${CAR}/car.yaml"

echo "==> Loading image on ${JETSON}"
ssh "${JETSON}" 'gunzip -c /tmp/f1tenth-image.tar.gz | docker load'

cat <<EOF

==> Done. Run on the car (mount the overlay + policy):

  docker run --rm -it --net=host --privileged \\
    -v /opt/f1tenth/config:/config:ro \\
    -v /opt/f1tenth/policies:/policies:ro \\
    ${IMAGE}:develop
EOF
