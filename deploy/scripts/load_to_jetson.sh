#!/usr/bin/env bash
# Copy a snapshot to the Jetson, load it under a SHA tag, stage overlay/checkpoint,
# and record rollback metadata. Never overwrites the other stack's moving tag
# (racing :develop vs sensor-policy :sensor-policy) and never clobbers
# /opt/f1tenth/policies/policy.pt.
#
# Usage:
#   deploy/scripts/load_to_jetson.sh <user@jetson-host> <snapshot.tar.gz> <carNN>
#   TARGET=sensor_policy CHECKPOINT=/path/to/policy_271360000.pt \
#     deploy/scripts/load_to_jetson.sh user@car01 deploy/snapshots/....tar.gz car01
#
# Env:
#   TARGET        racing (default) | sensor_policy
#   IMAGE         override image name
#   CHECKPOINT    optional local .pt to stage under a unique name
#   ACTOR_LAYOUT  optional manifest note (default 2 for format-4 sensor-policy)
#   POLICY_FORMAT optional manifest note (default 4)
#   JETPACK_L4T   optional manifest note (e.g. R36.4.7)
#   DRY_RUN_DOCS  if 1, only print remote inventory/rollback plan (no copy/load)
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

JETSON="${1:?usage: load_to_jetson.sh <user@host> <snapshot.tar.gz|-> <carNN>}"
SNAPSHOT="${2:?path to snapshot .tar.gz (or '-' with DRY_RUN_DOCS=1)}"
CAR="${3:?car id, e.g. car01}"
TARGET="${TARGET:-racing}"
DRY_RUN_DOCS="${DRY_RUN_DOCS:-0}"
ACTOR_LAYOUT="${ACTOR_LAYOUT:-2}"
POLICY_FORMAT="${POLICY_FORMAT:-4}"
JETPACK_L4T="${JETPACK_L4T:-R36.4.x}"

case "${TARGET}" in
  racing)
    IMAGE="${IMAGE:-f1tenth-racing}"
    MOVING_TAG="develop"
    DEFAULT_LAUNCH='ros2 launch f1tenth_bringup race.launch.py'
    RUNTIME_FLAGS="--privileged --net=host"
    ;;
  sensor_policy)
    IMAGE="${IMAGE:-f1tenth-sensor-policy}"
    MOVING_TAG="sensor-policy"
    DEFAULT_LAUNCH='ros2 launch f1tenth_bringup sensor_policy.launch.py'
    # NVIDIA runtime required for CUDA; CPU inference still works with it present.
    RUNTIME_FLAGS="--runtime nvidia --privileged --net=host"
    ;;
  *)
    echo "TARGET must be racing or sensor_policy (got: ${TARGET})" >&2
    exit 1
    ;;
esac

if [ ! -d "deploy/cars/${CAR}" ]; then
  echo "No overlay found at deploy/cars/${CAR}" >&2
  exit 1
fi
if [ "${SNAPSHOT}" != "-" ] && [ ! -f "${SNAPSHOT}" ]; then
  echo "Snapshot not found: ${SNAPSHOT}" >&2
  exit 1
fi
if [ "${SNAPSHOT}" = "-" ] && [ "${DRY_RUN_DOCS}" != "1" ]; then
  echo "SNAPSHOT='-' is only valid with DRY_RUN_DOCS=1" >&2
  exit 1
fi

if [ "${SNAPSHOT}" = "-" ]; then
  GITSHA="$(git rev-parse --short HEAD)"
  SNAP_SHA="(no-snapshot)"
else
  GITSHA="$(basename "${SNAPSHOT}" | sed -n "s/^${IMAGE}-\([^.]*\)\.tar\.gz$/\1/p")"
  if [ -z "${GITSHA}" ]; then
    GITSHA="$(git rev-parse --short HEAD)"
    echo "warning: could not parse git sha from snapshot name; using ${GITSHA}" >&2
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    SNAP_SHA="$(sha256sum "${SNAPSHOT}" | awk '{print $1}')"
  else
    SNAP_SHA="$(shasum -a 256 "${SNAPSHOT}" | awk '{print $1}')"
  fi
fi

CKPT_REMOTE=""
CKPT_SHA=""
if [ -n "${CHECKPOINT:-}" ]; then
  if [ ! -f "${CHECKPOINT}" ]; then
    echo "CHECKPOINT not found: ${CHECKPOINT}" >&2
    exit 1
  fi
  CKPT_BASE="$(basename "${CHECKPOINT}")"
  if command -v sha256sum >/dev/null 2>&1; then
    CKPT_SHA="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
  else
    CKPT_SHA="$(shasum -a 256 "${CHECKPOINT}" | awk '{print $1}')"
  fi
  CKPT_REMOTE="/opt/f1tenth/policies/${CKPT_BASE%.pt}-${CKPT_SHA:0:12}.pt"
fi

ROLLBACK_DIR="/opt/f1tenth/rollback/${TARGET}"
REMOTE_SNAP="/tmp/${IMAGE}-${GITSHA}-${SNAP_SHA:0:12}.tar.gz"

echo "==> Pre-flight inventory on ${JETSON}"
ssh "${JETSON}" bash -s -- "${IMAGE}" "${MOVING_TAG}" "${ROLLBACK_DIR}" <<'EOF'
set -euo pipefail
IMAGE="$1"
MOVING_TAG="$2"
ROLLBACK_DIR="$3"

echo "-- docker / nvidia --"
docker version --format 'server={{.Server.Version}} arch={{.Server.Arch}}' || {
  echo "docker not usable on car (need group membership or passwordless sudo docker)." >&2
  exit 1
}
if docker info 2>/dev/null | grep -qi nvidia; then
  echo "nvidia runtime: present (docker info)"
else
  echo "nvidia runtime: not reported by docker info (CPU still ok; CUDA needs nvidia-container-toolkit)"
fi

echo "-- images matching ${IMAGE} --"
docker image ls --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep -E "^${IMAGE}:" || true

echo "-- running containers --"
docker ps --format '{{.ID}} {{.Image}} {{.Names}} {{.Status}}' || true

echo "-- preserving on-car overlay / policies --"
mkdir -p /opt/f1tenth/config /opt/f1tenth/policies "${ROLLBACK_DIR}"
ls -la /opt/f1tenth/config /opt/f1tenth/policies || true

PREV=""
if docker image inspect "${IMAGE}:${MOVING_TAG}" >/dev/null 2>&1; then
  PREV_ID="$(docker image inspect --format '{{.Id}}' "${IMAGE}:${MOVING_TAG}")"
  PREV_TAG="$(docker image inspect --format '{{range .RepoTags}}{{.}}{{"\n"}}{{end}}' "${IMAGE}:${MOVING_TAG}" \
    | grep -v ":${MOVING_TAG}$" | head -1)"
  if [ -z "${PREV_TAG}" ]; then
    PREV_TAG="${IMAGE}:${MOVING_TAG}"
  fi
  PREV="${PREV_TAG}|${PREV_ID}"
  echo "${PREV_TAG}" > "${ROLLBACK_DIR}/image.tag"
  echo "${PREV_ID}" > "${ROLLBACK_DIR}/image.id"
  echo "Recorded rollback image tag ${PREV_TAG}"
else
  echo "No existing ${IMAGE}:${MOVING_TAG} to record for rollback"
fi
echo "prev=${PREV}"
EOF

if [ "${DRY_RUN_DOCS}" = "1" ]; then
  echo "==> DRY_RUN_DOCS=1: stopping before copy/load"
  echo "    snapshot sha256: ${SNAP_SHA}"
  echo "    checkpoint: ${CHECKPOINT:-'(none)'} -> ${CKPT_REMOTE:-'(n/a)'}"
  exit 0
fi

echo "==> Copying snapshot to ${JETSON}:${REMOTE_SNAP}"
scp "${SNAPSHOT}" "${JETSON}:${REMOTE_SNAP}"

echo "==> Staging per-car overlay (${CAR}) — never overwrites car.yaml"
ssh "${JETSON}" "mkdir -p /opt/f1tenth/config /opt/f1tenth/policies ${ROLLBACK_DIR}"
scp "deploy/cars/${CAR}/params.yaml" "${JETSON}:/opt/f1tenth/config/params.yaml"
scp -r "deploy/cars/${CAR}/maps" "${JETSON}:/opt/f1tenth/config/maps"
ssh "${JETSON}" "test -f /opt/f1tenth/config/car.yaml || cp /dev/stdin /opt/f1tenth/config/car.yaml" \
  < "deploy/cars/${CAR}/car.yaml"

if [ -n "${CKPT_REMOTE}" ]; then
  echo "==> Staging checkpoint (unique name, leaves policy.pt untouched)"
  scp "${CHECKPOINT}" "${JETSON}:${CKPT_REMOTE}"
  if [ "${TARGET}" = "sensor_policy" ]; then
    ssh "${JETSON}" "ln -sfn '${CKPT_REMOTE}' /opt/f1tenth/policies/sensor_policy.pt"
  fi
  ssh "${JETSON}" "printf '%s\n' '${CKPT_REMOTE}' > '${ROLLBACK_DIR}/checkpoint.path'"
fi

echo "==> Loading image on ${JETSON} as ${IMAGE}:${GITSHA}"
ssh "${JETSON}" bash -s -- "${REMOTE_SNAP}" "${IMAGE}" "${GITSHA}" "${MOVING_TAG}" "${ROLLBACK_DIR}" <<'EOF'
set -euo pipefail
REMOTE_SNAP="$1"
IMAGE="$2"
GITSHA="$3"
MOVING_TAG="$4"
ROLLBACK_DIR="$5"

gunzip -c "${REMOTE_SNAP}" | docker load
docker tag "${IMAGE}:${GITSHA}" "${IMAGE}:${MOVING_TAG}"
docker image inspect --format '{{.Id}}' "${IMAGE}:${GITSHA}"
docker run --rm --entrypoint cat "${IMAGE}:${GITSHA}" /etc/f1tenth/sensor_policy_pins.json \
  > "${ROLLBACK_DIR}/pins.json" 2>/dev/null \
  || echo '{}' > "${ROLLBACK_DIR}/pins.json"
rm -f "${REMOTE_SNAP}"
EOF

RUN_HELPER="${ROLLBACK_DIR}/run.sh"
LAUNCH_CKPT_ARG=""
if [ "${TARGET}" = "sensor_policy" ]; then
  if [ -n "${CKPT_REMOTE}" ]; then
    LAUNCH_CKPT_ARG="checkpoint_path:=/policies/$(basename "${CKPT_REMOTE}")"
  else
    LAUNCH_CKPT_ARG="checkpoint_path:=/policies/sensor_policy.pt"
  fi
fi

ssh "${JETSON}" bash -s -- "${RUN_HELPER}" "${IMAGE}" "${MOVING_TAG}" "${RUNTIME_FLAGS}" \
  "${DEFAULT_LAUNCH}" "${LAUNCH_CKPT_ARG}" "${TARGET}" <<'EOF'
set -euo pipefail
RUN_HELPER="$1"
IMAGE="$2"
MOVING_TAG="$3"
RUNTIME_FLAGS="$4"
DEFAULT_LAUNCH="$5"
LAUNCH_CKPT_ARG="$6"
TARGET="$7"

cat > "${RUN_HELPER}" <<RUN
#!/usr/bin/env bash
set -euo pipefail
if [ "${TARGET}" = "sensor_policy" ] && fuser /dev/sensors/vesc >/dev/null 2>&1; then
  echo "VESC serial device already has an owner; refusing to start sensor-policy." >&2
  fuser -v /dev/sensors/vesc >&2 || true
  exit 1
fi
docker run --rm -it ${RUNTIME_FLAGS} \\
  -v /dev:/dev \\
  -v /opt/f1tenth/config:/config:ro \\
  -v /opt/f1tenth/policies:/policies:ro \\
  ${IMAGE}:${MOVING_TAG} \\
  ${DEFAULT_LAUNCH} ${LAUNCH_CKPT_ARG}
RUN
chmod +x "${RUN_HELPER}"
EOF

MANIFEST="deploy/releases/manifest.csv"
NOTES="${TARGET} git=${GITSHA} snap=${SNAP_SHA} actor_layout=${ACTOR_LAYOUT} policy_format=${POLICY_FORMAT} jetpack=${JETPACK_L4T}"
if [ -n "${CKPT_SHA}" ]; then
  NOTES="${NOTES} ckpt=${CKPT_SHA:0:16}"
fi
echo "${GITSHA},${SNAP_SHA},${CAR},$(date -u +%Y-%m-%dT%H:%M:%SZ),${NOTES}" >> "${MANIFEST}"

cat <<EOF

==> Loaded ${IMAGE}:${GITSHA} and retagged :${MOVING_TAG}
    snapshot sha256: ${SNAP_SHA}
    checkpoint:      ${CKPT_REMOTE:-'(not staged)'}
    manifest:        ${MANIFEST}

Run on the car:
  ssh ${JETSON} ${RUN_HELPER}

Dry-run (VESC *_dryrun topics; drivers/gate/racer disabled) for sensor_policy:
  docker run --rm -it ${RUNTIME_FLAGS} \\
    -v /dev:/dev \\
    -v /opt/f1tenth/config:/config:ro \\
    -v /opt/f1tenth/policies:/policies:ro \\
    ${IMAGE}:${MOVING_TAG} \\
    ros2 launch f1tenth_bringup sensor_policy.launch.py \\
      dry_run:=true device:=cuda \\
      enable_drivers:=false enable_gate:=false enable_racer:=false

Non-powered smoke (inventory + artifact + dry-run launch):
  docker run --rm --runtime nvidia \\
    -v /opt/f1tenth/policies:/policies:ro \\
    --entrypoint smoke_sensor_policy.sh ${IMAGE}:${MOVING_TAG}

Rollback metadata: ${ROLLBACK_DIR}/pins.json (Torch/CUDA/base digest)

Rollback:
  deploy/scripts/rollback_jetson.sh ${JETSON} ${TARGET}
EOF
